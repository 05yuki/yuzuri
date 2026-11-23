"""NVFP4 MoE experts for a few rows (decode): four short kernels a layer instead of cuTile's eight.

FlashInfer's cuTile MoE runs a decode token's experts as a permute, an input quantize, GEMM1, a split-K reduce, the
gated activation, its quantize, GEMM2 and a combine: eight launches a layer whose work is reading 1.4 MiB an expert
(Flash-Next NVFP4 on two 16 GB cards: 2.3 ms a token for about 0.6 GB, 29% of the card's bandwidth). Here:

  _gemm1  one program for 16 intermediate rows of one route: the route's input quantized as cuTile does it, gate and
          up from the checkpoint's packed e2m1 bytes and e4m3 block scales, their alphas, the activation, and the
          16 activations quantized for GEMM2 (one block of 16)
  _gemm2  one program for a block of hidden rows of one route: its down projection and alpha
  _combine  a token's routes with their weights, summed in order, with no atomics
  (and _quant_x, the input quantized once a row)

The arithmetic is cuTile's (FlashInfer fused_moe/cutile with Yuzuri's patches/flashinfer-cutile-fp4.py): the input
times 1 / g1 (the checkpoint's static activation global), a block's scale e4m3(amax / 6), codes by the midpoints
between e2m1 values, the products of codes and scales in fp32, gate and up rounded to bf16, the activation in fp32
rounded to bf16, its blocks quantized with no global (GEMM2's global rides on the alphas), each route's down
projection rounded to bf16 and the routes summed in fp32 with their weights. The sum over K runs in another order
than the tensor-core MMA, so the outputs agree to rounding, not bit for bit.

KT_NVFP4_DECODE_ROWS (default 4): the most rows this path takes; 0 turns it off.
KT_NVFP4_DECODE_CHECK=N: the first N calls outside a graph capture also run cuTile and log the difference.
"""
import os

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

ROWS = int(os.environ.get("KT_NVFP4_DECODE_ROWS", "4") or 0)
CHECK = int(os.environ.get("KT_NVFP4_DECODE_CHECK", "0") or 0)
_CHECKED = [0]
BN1, CK1, W1 = (int(v) for v in os.environ.get("KT_NVFP4_DECODE_G1", "16,128,8").split(","))
BK2, W2 = (int(v) for v in os.environ.get("KT_NVFP4_DECODE_G2", "16,4").split(","))
_BUFS = {}


@triton.jit
def _e2m1(c):
    """e2m1 codes (int32, 0..15) to their values, by their fp32 bits: e = 0 is 0 or 0.5, else 2^(e-1) x (1 + m/2)."""
    e = (c >> 1) & 3
    mt = c & 1
    bits = tl.where(e > 0, ((e + 126) << 23) | (mt << 22), mt * 0x3F000000) | ((c & 8) << 28)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _codes(v):
    """The e2m1 code of v already divided by its block's scale: cuTile's midpoints, ties down, sign from v < 0."""
    a = tl.abs(v)
    c = ((a > 0.25).to(tl.int32) + (a > 0.75).to(tl.int32) + (a > 1.25).to(tl.int32) + (a > 1.75).to(tl.int32)
         + (a > 2.5).to(tl.int32) + (a > 3.5).to(tl.int32) + (a > 5.0).to(tl.int32))
    return c + (v < 0.0).to(tl.int32) * 8


@triton.jit
def _block_scale(amax):
    """A block's e4m3 scale of amax / 6 (saturating at 448) as fp32, and the divisor the codes are made with."""
    s = tl.minimum(tl.math.div_rn(amax, 6.0), 448.0).to(tl.float8e4nv).to(tl.float32)
    return s, tl.maximum(s, 0.001953125)


@triton.jit
def _quant_blocks(v, G: tl.constexpr):
    """v [G, 16] to its FP4 values (codes times the block's scale)."""
    s, d = _block_scale(tl.max(tl.abs(v), axis=1))
    return _e2m1(_codes(tl.math.div_rn(v, d[:, None]))) * s[:, None]


@triton.jit
def _e2m1_lo(v):
    """The e2m1 value in the low nibble of v (int32), times 2^-14: its 3 magnitude bits placed as an fp16's low exponent
    bits and top mantissa bit give 2^(e-15) x (1 + m/2) (e > 0) or m/2 x 2^-14 (e = 0), e2m1's values over 2^14."""
    bits = ((v & 7) << 9) | ((v & 8) << 12)
    return bits.to(tl.int16).to(tl.float16, bitcast=True).to(tl.float32)


@triton.jit
def _quant_x(X, INV_G1, QX, K: tl.constexpr, A4: tl.constexpr, GB: tl.constexpr):
    """GB blocks of a row's GEMM1 input as cuTile quantizes it (times 1 / g1, blocks of 16), as fp32 values."""
    m = tl.program_id(0)
    g = tl.program_id(1) * GB + tl.arange(0, GB)
    i = tl.arange(0, 16)
    msk = (g < K // 16)[:, None]
    v = tl.load(X + m * K + g[:, None] * 16 + i[None, :], mask=msk, other=0.0).to(tl.float32)
    if A4:
        v = _quant_blocks(v * tl.load(INV_G1), GB)
    tl.store(QX + m * K + g[:, None] * 16 + i[None, :], v, mask=msk)


@triton.jit
def _gemm1(QX, IDS, W13, S13, A13, H,
           sw_e, sw_n, ss_e, ss_n, a13_e, A13_UP: tl.constexpr,
           TOPK: tl.constexpr, N: tl.constexpr, K: tl.constexpr, A4: tl.constexpr,
           ALPHA: tl.constexpr, BETA: tl.constexpr, LIMIT: tl.constexpr, BN: tl.constexpr, CK: tl.constexpr):
    """W13 as int32 words (8 e2m1 values a word, value i in nibble i), [E, 2N, K/8]; S13 e4m3 bytes [E, 2N, K/16]."""
    r = tl.program_id(0)          # route: token r // TOPK
    nb = tl.program_id(1)         # BN intermediate rows
    e0 = tl.load(IDS + r)
    if e0 >= 0:
        e = e0.to(tl.int64)
        m = r // TOPK
        rows = nb * BN + tl.arange(0, BN)
        GW: tl.constexpr = CK // 8    # words a step
        w = tl.arange(0, GW)
        acc_g = tl.zeros((BN,), tl.float32)
        acc_u = tl.zeros((BN,), tl.float32)
        wg = W13 + e * sw_e + rows[:, None] * sw_n
        wu = wg + N * sw_n
        sg = S13 + e * ss_e + rows[:, None] * ss_n
        su = sg + N * ss_n
        for k0 in range(0, K, CK):
            cw = k0 // 8 + w
            qg = tl.load(wg + cw[None, :])
            qu = tl.load(wu + cw[None, :])
            cg = tl.load(sg + (cw // 2)[None, :]).to(tl.float8e4nv, bitcast=True).to(tl.float32)
            cu = tl.load(su + (cw // 2)[None, :]).to(tl.float8e4nv, bitcast=True).to(tl.float32)
            pg = tl.zeros((BN, GW), tl.float32)
            pu = tl.zeros((BN, GW), tl.float32)
            for i in tl.static_range(8):
                xi = tl.load(QX + m * K + cw * 8 + i)[None, :]
                pg += _e2m1_lo(qg >> (4 * i)) * xi
                pu += _e2m1_lo(qu >> (4 * i)) * xi
            acc_g += tl.sum(pg * cg, axis=1)
            acc_u += tl.sum(pu * cu, axis=1)
        ag = tl.load(A13 + e * a13_e) * 16384.0
        if A13_UP:
            au = tl.load(A13 + e * a13_e + 1) * 16384.0
        else:
            au = ag
        gate = (acc_g * ag).to(tl.bfloat16).to(tl.float32)
        up = (acc_u * au).to(tl.bfloat16).to(tl.float32)
        if LIMIT > 0.0:
            gate = tl.minimum(gate, LIMIT)
            up = tl.maximum(tl.minimum(up, LIMIT), -LIMIT)
        # (IEEE division and libdevice's exp: Triton's / and tl.exp are approximate, and a bf16 or FP4 step away
        # from cuTile's rounding moved 0.6% of the output)
        sig = tl.math.div_rn(1.0, 1.0 + libdevice.exp(-(ALPHA * gate)))
        h = (gate * sig * (up + BETA)).to(tl.bfloat16).to(tl.float32)
        if A4:
            h = tl.reshape(_quant_blocks(tl.reshape(h, (BN // 16, 16)), BN // 16), (BN,))
        tl.store(H + r * N + rows, h)


@triton.jit
def _gemm2(H, IDS, W2, S2, A2, Y,
           sw_e, sw_k, ss_e, ss_k, a2_e,
           N: tl.constexpr, NW: tl.constexpr, NWP: tl.constexpr, K: tl.constexpr, BK: tl.constexpr):
    """One route's down projection for BK hidden rows (an expert's rows lie next to each other: BK x N/2 bytes in
    one read), rounded to bf16 as cuTile's GEMM2 output is, into Y [R, K] fp32. W2 as int32 words [E, K, N/8]; S2 e4m3
    bytes [E, K, N/16]; H the routes' FP4 activations [R, N] fp32."""
    r = tl.program_id(0)
    kb = tl.program_id(1)
    e0 = tl.load(IDS + r)
    if e0 >= 0:
        e = e0.to(tl.int64)
        rows = kb * BK + tl.arange(0, BK)
        rm = rows < K
        w = tl.arange(0, NWP)
        wm = w < NW
        msk = rm[:, None] & wm[None, :]
        q = tl.load(W2 + e * sw_e + rows[:, None] * sw_k + w[None, :], mask=msk, other=0)
        c = tl.load(S2 + e * ss_e + rows[:, None] * ss_k + (w // 2)[None, :], mask=msk,
                    other=0).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        p = tl.zeros((BK, NWP), tl.float32)
        for i in tl.static_range(8):
            hv = tl.load(H + r * N + w * 8 + i, mask=wm, other=0.0)[None, :]
            p += _e2m1_lo(q >> (4 * i)) * hv
        y = (tl.sum(p * c, axis=1) * (tl.load(A2 + e * a2_e) * 16384.0)).to(tl.bfloat16).to(tl.float32)
        tl.store(Y + r * K + rows, y, mask=rm)


@triton.jit
def _combine(Y, IDS, WTS, OUT, TOPK: tl.constexpr, K: tl.constexpr, BK: tl.constexpr):
    """A token's routes summed in their order with their weights, as cuTile's combine (CPU routes skipped)."""
    m = tl.program_id(0)
    kb = tl.program_id(1)
    rows = kb * BK + tl.arange(0, BK)
    rm = rows < K
    total = tl.zeros((BK,), tl.float32)
    for t in tl.static_range(TOPK):
        r = m * TOPK + t
        ok = tl.load(IDS + r) >= 0
        y = tl.load(Y + r * K + rows, mask=rm & ok, other=0.0)
        total += y * tl.where(ok, tl.load(WTS + r), 0.0)
    tl.store(OUT + m * K + rows, total.to(OUT.dtype.element_ty), mask=rm)


def supports(x, act, w13, w2) -> str:
    """'' when this path takes the call, else why not."""
    if not ROWS or x.shape[0] > ROWS:
        return "rows"
    if x.dtype != torch.bfloat16 or not x.is_contiguous():
        return "input"
    kind = getattr(act, "type", "swiglu")
    kind = str(getattr(kind, "name", kind)).lower()  # (an IntEnum's str is its number)
    if getattr(act, "is_gated", True) is False or kind != "swiglu":
        return f"activation {act}"
    K, N = x.shape[1], w2.shape[2] * 2
    if K % 64 or N % 16 or w13.shape[1] != 2 * N or w13.shape[2] * 2 != K or w13.stride(2) != 1 or w2.stride(2) != 1:
        return f"shapes {tuple(w13.shape)} {tuple(w2.shape)}"
    return ""


def run(x, ids, weights, w13, s13, a13, w2, s2, a2, inv_g1, a4: bool, act) -> torch.Tensor:
    """The experts' output [M, K] for x [M, K] bf16, ids [M, TOPK] int32 (a slot, -1 for a CPU route), weights
    [M, TOPK] fp32 (0 for a CPU route)."""
    M, K = x.shape
    TOPK = ids.shape[1]
    N = w2.shape[2] * 2
    dev = x.device
    key = (dev.index, M, TOPK, N)
    h = _BUFS.get(key)
    if h is None:
        h = _BUFS[key] = torch.empty((M * TOPK, N), dtype=torch.float32, device=dev)
    out = torch.empty((M, K), dtype=x.dtype, device=dev)
    ids = ids.contiguous()
    weights = weights.contiguous()
    a13 = a13.reshape(a13.shape[0], -1)
    a2 = a2.reshape(-1)
    alpha = float(getattr(act, "alpha", 1.0) or 1.0)
    beta = float(getattr(act, "beta", 0.0) or 0.0)
    limit = float(getattr(act, "limit", 0.0) or 0.0)
    limit = limit if 0.0 < limit < 1e30 else 0.0
    s13b, s2b = s13.view(torch.uint8), s2.view(torch.uint8)
    w13w, w2w = w13.view(torch.int32), w2.view(torch.int32)
    qx = _BUFS.get(("qx",) + key)
    if qx is None:
        qx = _BUFS[("qx",) + key] = torch.empty((M, K), dtype=torch.float32, device=dev)
    _quant_x[(M, triton.cdiv(K // 16, 8))](x, inv_g1, qx, K=K, A4=a4, GB=8, num_warps=1)
    bn = BN1 if N % BN1 == 0 else 16
    ck = CK1 if K % CK1 == 0 else 64
    _gemm1[(M * TOPK, N // bn)](
        qx, ids, w13w, s13b, a13, h,
        w13w.stride(0), w13w.stride(1), s13b.stride(0), s13b.stride(1), a13.stride(0), a13.shape[1] > 1,
        TOPK=TOPK, N=N, K=K, A4=a4, ALPHA=alpha, BETA=beta, LIMIT=limit, BN=bn, CK=ck, num_warps=W1)
    NW = N // 8
    y = _BUFS.get(("y",) + key)
    if y is None:
        y = _BUFS[("y",) + key] = torch.empty((M * TOPK, K), dtype=torch.float32, device=dev)
    _gemm2[(M * TOPK, triton.cdiv(K, BK2))](
        h, ids, w2w, s2b, a2, y,
        w2w.stride(0), w2w.stride(1), s2b.stride(0), s2b.stride(1), a2.stride(0),
        N=N, NW=NW, NWP=triton.next_power_of_2(NW), K=K, BK=BK2, num_warps=W2)
    _combine[(M, triton.cdiv(K, 512))](y, ids, weights, out, TOPK=TOPK, K=K, BK=512, num_warps=4)
    return out


def ref(x, ids, weights, w13, s13, a13, w2, s2, a2, inv_g1, a4: bool, act) -> torch.Tensor:
    """The same arithmetic in torch, route by route (for tests)."""
    E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device=x.device)

    def deq(b, s):  # packed [R, K/2], e4m3 [R, K/16]
        v = torch.stack((E2M1[(b & 15).long()], E2M1[(b >> 4).long()]), -1).reshape(b.shape[0], -1)
        return (v.view(b.shape[0], -1, 16) * s.float()[:, :, None]).reshape(b.shape[0], -1)

    def quant(v):  # [K] fp32 -> its dequantized FP4
        g = v.view(-1, 16)
        sc = (g.abs().amax(1) / 6).clamp(max=448).to(torch.float8_e4m3fn).float()
        d = sc.clamp(min=2.0 ** -9)
        a = (g / d[:, None]).abs()
        c = sum((a > t).long() for t in (.25, .75, 1.25, 1.75, 2.5, 3.5, 5.0)) + (g / d[:, None] < 0).long() * 8
        return (E2M1[c] * sc[:, None]).reshape(-1)

    M, K = x.shape
    N = w2.shape[2] * 2
    a13 = a13.reshape(a13.shape[0], -1)
    alpha, beta = float(getattr(act, "alpha", 1.0)), float(getattr(act, "beta", 0.0))
    out = torch.zeros((M, K), dtype=torch.float32, device=x.device)
    for m in range(M):
        xm = x[m].float() * (float(inv_g1) if a4 else 1.0)
        xq = quant(xm) if a4 else xm
        for t in range(ids.shape[1]):
            e = int(ids[m, t])
            if e < 0:
                continue
            gu = deq(w13[e], s13[e]) @ xq
            gate = (gu[:N] * a13[e, 0]).bfloat16().float()
            up = (gu[N:] * a13[e, -1]).bfloat16().float()
            h = (gate * torch.sigmoid(alpha * gate) * (up + beta)).bfloat16().float()
            hq = quant(h) if a4 else h
            y = ((deq(w2[e], s2[e]) @ hq) * a2.reshape(-1)[e]).bfloat16().float()
            out[m] += y * weights[m, t]
    return out.to(x.dtype)
