"""HyperConnection low-rank mix + optional in-place FP8 for HC mix (KT_HC_MIX_FP8=1, per-row)
and BF16 dense projections (KT_DENSE_FP8=1, MXFP8 block-32 UE8M0 for both M=1 decode GEMV and
M>=64 SM120 flashinfer.mm_mxfp8 prefill, optional KT_DENSE_FP8_LM_HEAD=1)
+ 100% BF16 Split-K GEMV for small unquantized M=1 linears like router (KT_DENSE_BF16_GEMV=1).

Quantization runs inside DefaultModelLoader.postprocess_weights right before kt_lend.after_load(),
so kt_lend repacks the halved FP8 weights into smaller VMM spans and frees VRAM for hot expert slots.
"""
import logging
import os

from yuzuri._finder import after_import
from yuzuri.hooks._seam import wrap

REPLACES = ["kernels/ops/gemm/hc_mix.py"]
logger = logging.getLogger("yuzuri.hc_mix_m1")

_ENABLED = os.environ.get("KT_HC_MIX_M1", "1") == "1"
_FP8_ENABLED = os.environ.get("KT_HC_MIX_FP8", "0") == "1"
_DENSE_FP8 = os.environ.get("KT_DENSE_FP8", "0") == "1"
_DENSE_FP8_LM_HEAD = os.environ.get("KT_DENSE_FP8_LM_HEAD", "0") == "1"
_DENSE_BF16_GEMV = os.environ.get("KT_DENSE_BF16_GEMV", "1") == "1"
_PREFILL_MXFP8 = os.environ.get("KT_DENSE_PREFILL_MXFP8", "1") == "1"
_SPLIT_K = int(os.environ.get("KT_HC_MIX_M1_SPLIT_K", "8") or 8)
_KERNELS = None
_BUFS = {}
_FI_MOD = [None]


def _get_flashinfer():
    if _FI_MOD[0] is None:
        try:
            import flashinfer

            if (
                hasattr(flashinfer, "mxfp8_quantize")
                and hasattr(flashinfer, "mm_mxfp8")
                and hasattr(flashinfer, "block_scale_interleave")
            ):
                _FI_MOD[0] = (
                    flashinfer.mxfp8_quantize,
                    flashinfer.mm_mxfp8,
                    flashinfer.block_scale_interleave,
                )
            else:
                _FI_MOD[0] = False
        except Exception:
            _FI_MOD[0] = False
    return _FI_MOD[0]


def _get_kernels():
    global _KERNELS
    if _KERNELS is not None:
        return _KERNELS
    import triton
    import triton.language as tl

    @triton.jit
    def _hc_down_sk(
        x_ptr, wd_ptr, t_ptr,
        K: tl.constexpr, LOW: tl.constexpr,
        BN: tl.constexpr, BK: tl.constexpr, SPLIT_K: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        n = pid_n * BN + tl.arange(0, BN)
        mn = n < LOW
        k_per = K // SPLIT_K
        k_start = pid_k * k_per
        acc = tl.zeros((BN,), dtype=tl.float32)
        for k0 in range(k_start, k_start + k_per, BK):
            k = k0 + tl.arange(0, BK)
            xv = tl.load(x_ptr + k).to(tl.float32)
            wv = tl.load(wd_ptr + n[:, None] * K + k[None, :], mask=mn[:, None], other=0.0).to(tl.float32)
            acc += tl.sum(wv * xv[None, :], axis=1)
        tl.store(t_ptr + pid_k * LOW + n, acc, mask=mn)

    @triton.jit
    def _hc_up_sk(
        x_ptr, wu_ptr, t_ptr, out_ptr,
        HS: tl.constexpr, LOW: tl.constexpr, LOW_PAD: tl.constexpr,
        HC: tl.constexpr, BJ: tl.constexpr, SPLIT_K: tl.constexpr, inv_hc,
    ):
        pid_j = tl.program_id(0)
        j = pid_j * BJ + tl.arange(0, BJ)
        mj = j < HS
        r = tl.arange(0, LOW_PAD)
        mr = r < LOW
        a = tl.zeros((LOW_PAD,), dtype=tl.float32)
        for sk in tl.static_range(SPLIT_K):
            a += tl.load(t_ptr + sk * LOW + r, mask=mr, other=0.0)
        a = a * inv_hc
        tv = (a * tl.sigmoid(a)).to(x_ptr.dtype.element_ty).to(tl.float32)
        out = tl.zeros((BJ,), dtype=tl.float32)
        for g in tl.static_range(HC):
            row = g * HS + j
            wv = tl.load(wu_ptr + row[:, None] * LOW + r[None, :], mask=mj[:, None] & mr[None, :], other=0.0).to(tl.float32)
            acc = tl.sum(wv * tv[None, :], axis=1)
            gate = tl.sigmoid(acc)
            xg = tl.load(x_ptr + row, mask=mj, other=0.0).to(tl.float32)
            out += gate * xg
        tl.store(out_ptr + j, (out * inv_hc).to(out_ptr.dtype.element_ty), mask=mj)

    @triton.jit
    def _hc_up_sk_fp8(
        x_ptr, wu_ptr, sd_ptr, su_ptr, t_ptr, out_ptr,
        HS: tl.constexpr, LOW: tl.constexpr, LOW_PAD: tl.constexpr,
        HC: tl.constexpr, BJ: tl.constexpr, SPLIT_K: tl.constexpr, inv_hc,
    ):
        pid_j = tl.program_id(0)
        j = pid_j * BJ + tl.arange(0, BJ)
        mj = j < HS
        r = tl.arange(0, LOW_PAD)
        mr = r < LOW
        a = tl.zeros((LOW_PAD,), dtype=tl.float32)
        for sk in tl.static_range(SPLIT_K):
            a += tl.load(t_ptr + sk * LOW + r, mask=mr, other=0.0)
        sd = tl.load(sd_ptr + r, mask=mr, other=1.0)
        a = a * sd * inv_hc
        tv = (a * tl.sigmoid(a)).to(x_ptr.dtype.element_ty).to(tl.float32)
        out = tl.zeros((BJ,), dtype=tl.float32)
        for g in tl.static_range(HC):
            row = g * HS + j
            wv = tl.load(wu_ptr + row[:, None] * LOW + r[None, :], mask=mj[:, None] & mr[None, :], other=0.0).to(tl.float32)
            su = tl.load(su_ptr + row, mask=mj, other=1.0)
            acc = tl.sum(wv * tv[None, :], axis=1) * su
            gate = tl.sigmoid(acc)
            xg = tl.load(x_ptr + row, mask=mj, other=0.0).to(tl.float32)
            out += gate * xg
        tl.store(out_ptr + j, (out * inv_hc).to(out_ptr.dtype.element_ty), mask=mj)

    @triton.jit
    def _quant_per_token_fp8(x_ptr, q_ptr, s_ptr, K: tl.constexpr, BK: tl.constexpr):
        row = tl.program_id(0)
        off = row * K
        amax = 0.0
        for k0 in range(0, K, BK):
            k = k0 + tl.arange(0, BK)
            mk = k < K
            xv = tl.load(x_ptr + off + k, mask=mk, other=0.0).to(tl.float32)
            amax = tl.maximum(amax, tl.max(tl.abs(xv), axis=0))
        scale = tl.maximum(amax / 448.0, 1e-12)
        inv_scale = 1.0 / scale
        tl.store(s_ptr + row, scale)
        for k0 in range(0, K, BK):
            k = k0 + tl.arange(0, BK)
            mk = k < K
            xv = tl.load(x_ptr + off + k, mask=mk, other=0.0).to(tl.float32)
            qv = tl.clamp(xv * inv_scale, -448.0, 448.0).to(tl.float8e4nv)
            tl.store(q_ptr + off + k, qv, mask=mk)

    @triton.jit
    def _silu_div_quant_fp8(x_ptr, q_ptr, s_ptr, K: tl.constexpr, KPAD: tl.constexpr, inv_hc):
        row = tl.program_id(0)
        k = tl.arange(0, KPAD)
        mk = k < K
        off = row * K + k
        xv = tl.load(x_ptr + off, mask=mk, other=0.0).to(tl.float32) * inv_hc
        tv = xv * tl.sigmoid(xv)
        amax = tl.max(tl.abs(tv), axis=0)
        scale = tl.maximum(amax / 448.0, 1e-12)
        tl.store(s_ptr + row, scale)
        qv = tl.clamp(tv / scale, -448.0, 448.0).to(tl.float8e4nv)
        tl.store(q_ptr + off, qv, mask=mk)

    @triton.jit
    def _hc_gate_mean(x_ptr, g_ptr, out_ptr, HS: tl.constexpr, HC: tl.constexpr, BJ: tl.constexpr):
        row = tl.program_id(0)
        pid_j = tl.program_id(1)
        j = pid_j * BJ + tl.arange(0, BJ)
        mj = j < HS
        acc = tl.zeros((BJ,), dtype=tl.float32)
        base = row * (HC * HS)
        for c in tl.static_range(HC):
            off = base + c * HS + j
            gv = tl.load(g_ptr + off, mask=mj, other=0.0).to(tl.float32)
            xv = tl.load(x_ptr + off, mask=mj, other=0.0).to(tl.float32)
            acc += tl.sigmoid(gv) * xv
        tl.store(out_ptr + row * HS + j, (acc * (1.0 / HC)).to(out_ptr.dtype.element_ty), mask=mj)

    @triton.jit
    def _gemv_mxfp8_w8a16_sk(
        x_ptr, w_ptr, s_ptr, partial_ptr,
        N: tl.constexpr, K: tl.constexpr, NG: tl.constexpr,
        BN: tl.constexpr, BK: tl.constexpr, BG: tl.constexpr, SPLIT_K: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        n = pid_n * BN + tl.arange(0, BN)
        mn = n < N
        k_per = K // SPLIT_K
        k_start = pid_k * k_per
        acc = tl.zeros((BN,), dtype=tl.float32)
        g_off = tl.arange(0, BG)
        for k0 in range(k_start, k_start + k_per, BK):
            k = k0 + tl.arange(0, BK)
            xv = tl.load(x_ptr + k).to(tl.float32)
            wv = tl.load(w_ptr + n[:, None] * K + k[None, :], mask=mn[:, None], other=0.0).to(tl.float32)
            prod = tl.reshape(wv * xv[None, :], (BN, BG, 32))
            s_u8 = tl.load(s_ptr + n[:, None] * NG + (k0 // 32 + g_off)[None, :], mask=mn[:, None], other=127)
            sc = (s_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
            acc += tl.sum(tl.sum(prod, axis=2) * sc, axis=1)
        if SPLIT_K == 1:
            tl.store(partial_ptr + n, acc.to(partial_ptr.dtype.element_ty), mask=mn)
        else:
            tl.store(partial_ptr + pid_k * N + n, acc, mask=mn)

    @triton.jit
    def _gemv_bf16_sk(
        x_ptr, w_ptr, partial_ptr,
        N: tl.constexpr, K: tl.constexpr,
        BN: tl.constexpr, BK: tl.constexpr, SPLIT_K: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        n = pid_n * BN + tl.arange(0, BN)
        mn = n < N
        k_per = K // SPLIT_K
        k_start = pid_k * k_per
        acc = tl.zeros((BN,), dtype=tl.float32)
        for k0 in range(k_start, k_start + k_per, BK):
            k = k0 + tl.arange(0, BK)
            xv = tl.load(x_ptr + k).to(tl.float32)
            wv = tl.load(w_ptr + n[:, None] * K + k[None, :], mask=mn[:, None], other=0.0).to(tl.float32)
            acc += tl.sum(wv * xv[None, :], axis=1)
        if SPLIT_K == 1:
            tl.store(partial_ptr + n, acc.to(partial_ptr.dtype.element_ty), mask=mn)
        else:
            tl.store(partial_ptr + pid_k * N + n, acc, mask=mn)

    @triton.jit
    def _reduce_gemv_sk(partial_ptr, out_ptr, N: tl.constexpr, SPLIT_K: tl.constexpr, BN: tl.constexpr):
        pid = tl.program_id(0)
        n = pid * BN + tl.arange(0, BN)
        mn = n < N
        acc = tl.zeros((BN,), dtype=tl.float32)
        for sk in tl.static_range(SPLIT_K):
            acc += tl.load(partial_ptr + sk * N + n, mask=mn, other=0.0)
        tl.store(out_ptr + n, acc.to(out_ptr.dtype.element_ty), mask=mn)

    _KERNELS = (
        _hc_down_sk,
        _hc_up_sk,
        _hc_up_sk_fp8,
        _quant_per_token_fp8,
        _silu_div_quant_fp8,
        _hc_gate_mean,
        _gemv_mxfp8_w8a16_sk,
        _gemv_bf16_sk,
        _reduce_gemv_sk,
    )
    return _KERNELS


def _quant_fp8_row(w):
    import torch

    fmax = torch.finfo(torch.float8_e4m3fn).max
    wf = w.float()
    scale = (wf.abs().amax(dim=-1) / fmax).clamp(min=1e-12)
    q = (wf / scale[:, None]).clamp(-fmax, fmax).to(torch.float8_e4m3fn).contiguous()
    return q, scale.contiguous()


def _quant_mxfp8_dense(w):
    """Returns (q_fp8[N, K], scale_ue8m0_u8[N, K//32]) using FlashInfer when available."""
    import torch

    n, k = w.shape
    ng = k // 32
    fi = _get_flashinfer()
    if fi and n <= 16384:
        mxfp8_quantize, _, _ = fi
        q, s = mxfp8_quantize(w.contiguous(), is_sf_swizzled_layout=False)
        return q.contiguous(), s.view(n, ng).contiguous()
    q = torch.empty((n, k), dtype=torch.float8_e4m3fn, device=w.device)
    s = torch.empty((n, ng), dtype=torch.uint8, device=w.device)
    for i in range(0, n, 16384):
        j = min(i + 16384, n)
        if fi:
            mxfp8_quantize, _, _ = fi
            qc, sc = mxfp8_quantize(w[i:j].contiguous(), is_sf_swizzled_layout=False)
            q[i:j] = qc
            s[i:j] = sc.view(j - i, ng)
        else:
            wf = w[i:j].float().view(j - i, ng, 32)
            req = (wf.abs().amax(dim=-1) / 448.0).clamp(min=1e-30)
            bits = req.view(torch.int32)
            exp = (((bits + 0x7FFFFF) >> 23) & 0xFF).to(torch.uint8)
            sc_f = (exp.to(torch.int32) << 23).view(torch.float32)
            q[i:j] = (wf / sc_f[:, :, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).reshape(j - i, k)
            s[i:j] = exp
    return q, s


def _ue8m0_to_f32(s_u8):
    import torch

    return (s_u8.to(torch.int32) << 23).view(torch.float32)


def _fp8_linear(x, w, s_u8, bias=None):
    import torch
    import torch.nn.functional as F
    import triton

    orig_shape = x.shape
    x2 = x.reshape(-1, orig_shape[-1])
    rows, k = x2.shape
    n = w.shape[0]
    ng = s_u8.shape[1]

    # 1. Decode M=1: MXFP8 W8A16 Split-K Triton GEMV
    if rows == 1 and bias is None and x2.is_cuda and k % 64 == 0:
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        _, _, _, _, _, _, gemv_k, _, red_k = _get_kernels()
        if k % 128 != 0:
            sk, bn, bk, nw = 1, 8, 64, 4
        elif n >= 32768:
            sk, bn, bk, nw = 1, 8, 128, 4
        elif k % (5 * 128) == 0 and n >= 512:
            sk, bn, bk, nw = 5, 8, 128, 8
        elif k % (4 * 128) == 0:
            sk, bn, bk, nw = 4, 16, 128, 4
        else:
            sk, bn, bk, nw = 1, 8, 128, 4
        out = torch.empty((1, n), dtype=x2.dtype, device=x2.device)
        if sk == 1:
            gemv_k[(triton.cdiv(n, bn), 1)](
                x2, w, s_u8, out, N=n, K=k, NG=ng, BN=bn, BK=bk, BG=bk // 32, SPLIT_K=1, num_warps=nw
            )
        else:
            key = ("gemv", x2.device, sk, n)
            pbuf = _BUFS.get(key)
            if pbuf is None:
                pbuf = _BUFS[key] = torch.empty((sk, n), dtype=torch.float32, device=x2.device)
            gemv_k[(triton.cdiv(n, bn), sk)](
                x2, w, s_u8, pbuf, N=n, K=k, NG=ng, BN=bn, BK=bk, BG=bk // 32, SPLIT_K=sk, num_warps=nw
            )
            red_k[(triton.cdiv(n, 256),)](pbuf, out, N=n, SPLIT_K=sk, BN=256, num_warps=4)
        return out.view(*orig_shape[:-1], n)

    # 2. Prefill M>=64: SM120 flashinfer.mm_mxfp8 (2.6-3.1x faster than BF16, zero extra permanent VRAM)
    if (
        _PREFILL_MXFP8
        and rows >= 64
        and bias is None
        and x2.is_cuda
        and x2.dtype in (torch.bfloat16, torch.float16)
        and 128 <= n <= 16384  # mm_mxfp8 takes n and k from 128 up (a one-GPU in_proj_ba is 96 rows wide)
        and k >= 128
        and n % 32 == 0
        and k % 32 == 0
        and not torch.cuda.is_current_stream_capturing()
    ):
        fi = _get_flashinfer()
        if fi:
            mxfp8_quantize, mm_mxfp8, block_scale_interleave = fi
            try:
                from sglang.srt.layers.moe import kt_lend
            except ImportError:
                kt_lend = None
            in_prefill = kt_lend is None or (not kt_lend.ENABLED) or kt_lend.phase() == "prefill"
            if in_prefill:
                x_c = x2.contiguous()
                pad = (16 - (rows % 16)) % 16
                if pad:
                    x_c = F.pad(x_c, (0, 0, 0, pad))
                xq, xs_swiz = mxfp8_quantize(x_c, is_sf_swizzled_layout=True)
                ws_swiz = block_scale_interleave(s_u8)
                backend = "b12x" if (x_c.shape[0] < 1024 and k % 128 == 0) else "cudnn"
                out = torch.empty((x_c.shape[0], n), dtype=x2.dtype, device=x2.device)
                if kt_lend is not None and kt_lend.ENABLED:
                    with kt_lend.scratch_region():
                        mm_mxfp8(xq, w.T, xs_swiz, ws_swiz, out=out, out_dtype=x2.dtype, backend=backend)
                else:
                    mm_mxfp8(xq, w.T, xs_swiz, ws_swiz, out=out, out_dtype=x2.dtype, backend=backend)
                if pad:
                    out = out[:rows]
                return out.view(*orig_shape[:-1], n)

    # 3. Fallback (small M 2..63, N % 32 != 0, or large lm_head M > 1): chunked dequant F.linear
    dt = x.dtype
    if n <= 16384:
        w_d = (w.view(n, ng, 32).to(dt) * _ue8m0_to_f32(s_u8)[:, :, None].to(dt)).reshape(n, k)
        return F.linear(x, w_d, bias)
    out = torch.empty((*orig_shape[:-1], n), dtype=dt, device=x.device)
    for i in range(0, n, 16384):
        j = min(i + 16384, n)
        wc = (w[i:j].view(j - i, ng, 32).to(dt) * _ue8m0_to_f32(s_u8[i:j])[:, :, None].to(dt)).reshape(j - i, k)
        bc = bias[i:j] if bias is not None else None
        out[..., i:j] = F.linear(x, wc, bc)
    return out


def _bf16_small_gemv(x, w):
    import torch
    import triton

    orig_shape = x.shape
    x2 = x.reshape(-1, orig_shape[-1])
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    n, k = w.shape
    _, _, _, _, _, _, _, bf16_k, red_k = _get_kernels()
    sk, bn, bk, nw = 5, 8, 128, 8
    out = torch.empty((1, n), dtype=x2.dtype, device=x2.device)
    key = ("gemv", x2.device, sk, n)
    pbuf = _BUFS.get(key)
    if pbuf is None:
        pbuf = _BUFS[key] = torch.empty((sk, n), dtype=torch.float32, device=x2.device)
    bf16_k[(triton.cdiv(n, bn), sk)](
        x2, w, pbuf, N=n, K=k, BN=bn, BK=bk, SPLIT_K=sk, num_warps=nw
    )
    red_k[(triton.cdiv(n, 256),)](pbuf, out, N=n, SPLIT_K=sk, BN=256, num_warps=4)
    return out.view(*orig_shape[:-1], n)


def _quant_gated_residual(mod, st):
    import torch
    import torch.nn as nn

    wd_lin = getattr(mod, "input_mix_weight_down", None)
    wu_lin = getattr(mod, "input_mix_weight_up", None)
    if not (isinstance(wd_lin, nn.Linear) and isinstance(wu_lin, nn.Linear)):
        return
    wd = wd_lin.weight
    wu = wu_lin.weight
    if not (wd.is_cuda and wu.is_cuda and wd.dtype == torch.bfloat16 and wu.dtype == torch.bfloat16):
        return
    before = wd.numel() * wd.element_size() + wu.numel() * wu.element_size()
    wd_q, sd = _quant_fp8_row(wd.data)
    wu_q, su = _quant_fp8_row(wu.data)
    p_d = nn.Parameter(wd_q, requires_grad=False)
    p_u = nn.Parameter(wu_q, requires_grad=False)
    p_d._hc_fp8_scale = sd
    p_u._hc_fp8_scale = su
    wd_lin.weight = p_d
    wu_lin.weight = p_u
    mod.register_buffer("_hc_down_scale", sd, persistent=False)
    mod.register_buffer("_hc_up_scale", su, persistent=False)
    after = (
        wd_q.numel() * wd_q.element_size()
        + wu_q.numel() * wu_q.element_size()
        + sd.numel() * sd.element_size()
        + su.numel() * su.element_size()
    )
    st["hc_freed"] += before - after
    st["hc"] += 1


def _quant_fused_in_proj(mod, st, handled):
    import torch
    import torch.nn as nn

    fw = getattr(mod, "_fused_in_proj_weight", None)
    if not (
        fw is not None
        and isinstance(fw, torch.Tensor)
        and fw.is_cuda
        and fw.dim() == 2
        and fw.dtype == torch.bfloat16
        and fw.shape[1] % 32 == 0
        and hasattr(mod, "in_proj_qkvz")
        and hasattr(mod, "in_proj_ba")
        and getattr(mod, "_fused_in_proj_qkvz_width", 0) > 0
    ):
        return
    before = fw.numel() * fw.element_size()
    fq, fs = _quant_mxfp8_dense(fw)
    w1 = mod._fused_in_proj_qkvz_width
    mod._fused_in_proj_weight = fq
    mod.register_buffer("_fused_in_proj_fp8_scale", fs, persistent=False)
    mod.in_proj_qkvz.weight = nn.Parameter(fq[:w1], requires_grad=False)
    mod.in_proj_qkvz.register_buffer("_dense_fp8_scale", fs[:w1].contiguous(), persistent=False)
    mod.in_proj_ba.weight = nn.Parameter(fq[w1:], requires_grad=False)
    mod.in_proj_ba.register_buffer("_dense_fp8_scale", fs[w1:].contiguous(), persistent=False)
    handled.add(id(mod.in_proj_qkvz))
    handled.add(id(mod.in_proj_ba))
    st["freed"] += before - (fq.numel() * fq.element_size() + fs.numel() * fs.element_size())
    st["fused"] += 1


def _quant_dense_linear(name, mod, st, unquant):
    import torch
    import torch.nn as nn

    if not isinstance(getattr(mod, "quant_method", None), unquant):
        return
    if any(skip in name for skip in ("embed_tokens", "hyper_connection", "visual", "mtp", "shared_expert_gate", "lm_head")):
        return
    if name.endswith(".gate") or ".mlp.gate" in name:
        return
    w = getattr(mod, "weight", None)
    if w is None or not w.is_cuda or w.dim() != 2 or w.dtype != torch.bfloat16 or getattr(mod, "bias", None) is not None:
        return
    n, k = w.shape
    if n < 256 or k < 256 or k % 64 != 0:
        return
    before = w.numel() * w.element_size()
    wq, ws = _quant_mxfp8_dense(w.data)
    mod.weight = nn.Parameter(wq, requires_grad=False)
    mod.register_buffer("_dense_fp8_scale", ws, persistent=False)
    st["freed"] += before - (wq.numel() * wq.element_size() + ws.numel() * ws.element_size())
    st["lin"] += 1


def _quant_lm_head(mod, st):
    """The output head through host memory: its bf16 bytes leave the card before the FP8 ones are made, so the card
    holds the larger of the two and one 16,384-row piece, not both (a one-GPU launch has no room for both)."""
    import torch
    import torch.nn as nn

    w = getattr(mod, "weight", None)
    if not (w is not None and w.is_cuda and w.dim() == 2 and w.dtype == torch.bfloat16 and w.shape[1] % 128 == 0):
        return
    n, k = w.shape
    ng = k // 32
    dev = w.device
    before = w.numel() * w.element_size()
    host = w.detach().to("cpu")
    mod.weight = None
    del w
    torch.cuda.empty_cache()
    q = torch.empty((n, k), dtype=torch.float8_e4m3fn, device=dev)
    s = torch.empty((n, ng), dtype=torch.uint8, device=dev)
    for i in range(0, n, 16384):
        j = min(i + 16384, n)
        qc, sc = _quant_mxfp8_dense(host[i:j].to(dev))
        q[i:j] = qc
        s[i:j] = sc
        del qc, sc
    del host
    mod.weight = nn.Parameter(q, requires_grad=False)
    mod.register_buffer("_dense_fp8_scale", s, persistent=False)
    st["freed"] += before - (q.numel() * q.element_size() + s.numel() * s.element_size())
    st["head"] += 1


def _quantize_model(model):
    """One pass over the modules. A decoder layer the prefill lend holds in a span is taken back into that span as
    soon as its modules are done (kt_lend.after_module), so the FP8 copies made outside the spans are one layer's at
    a time: made for the whole model before the lend's end of load, they did not fit beside the spans on one card."""
    import torch
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    try:
        from sglang.srt.layers.moe import kt_lend
    except ImportError:
        kt_lend = None
    st = dict(hc=0, hc_freed=0, fused=0, lin=0, head=0, freed=0)
    handled = set()
    heads = []
    cur = cur_mod = None

    def close():
        if cur_mod is not None and kt_lend is not None:
            kt_lend.after_module(cur_mod)
            torch.cuda.empty_cache()

    at = "?"
    try:
        for name, mod in model.named_modules():
            at = name
            i = getattr(mod, "_kt_lend_layer", None)
            if i != cur:
                close()
                cur, cur_mod = i, (mod if i is not None else None)
            if _FP8_ENABLED:
                _quant_gated_residual(mod, st)
            if not _DENSE_FP8 or id(mod) in handled:
                continue
            if name == "lm_head" or name.endswith(".lm_head"):
                if _DENSE_FP8_LM_HEAD:
                    heads.append((name, mod))
                continue
            _quant_fused_in_proj(mod, st, handled)
            _quant_dense_linear(name, mod, st, UnquantizedLinearMethod)
        close()
        cur_mod = None
        for name, mod in heads:
            at = name
            _quant_lm_head(mod, st)
    except torch.OutOfMemoryError as e:
        dev = torch.cuda.current_device()
        raise RuntimeError(
            f"KT_HC_MIX_FP8 / KT_DENSE_FP8: no room on cuda:{dev} to make the FP8 copy of {at} "
            f"({torch.cuda.mem_get_info(dev)[0] >> 20} MiB free): launch without them, or on more cards"
        ) from e
    if st["hc"]:
        logger.info(
            "[yuzuri] hc_mix_fp8: quantized %d GatedResidual modules in-place (pre-kt_lend), freed %.1f MiB",
            st["hc"],
            st["hc_freed"] / (1024 * 1024),
        )
    if st["fused"] or st["lin"] or st["head"]:
        logger.info(
            "[yuzuri] dense_fp8(mxfp8-b32): quantized %d fused in_proj + %d dense linears + %d lm_head in-place (pre-kt_lend), freed %.1f MiB",
            st["fused"],
            st["lin"],
            st["head"],
            st["freed"] / (1024 * 1024),
        )
    return st["hc"] + st["fused"] + st["lin"] + st["head"]


def _install_hc_mix(m):
    import torch
    import torch.nn.functional as F
    import triton

    if getattr(m, "_is_npu", False):
        return

    def make_supported(orig):
        def fused_hc_mix_supported(hyper_input_normed, w_down, w_up):
            if (
                _FP8_ENABLED
                and hyper_input_normed.is_cuda
                and hyper_input_normed.dtype in (torch.bfloat16, torch.float16)
                and w_down.dtype == torch.float8_e4m3fn
                and w_up.dtype == torch.float8_e4m3fn
                and hasattr(w_down, "_hc_fp8_scale")
                and hasattr(w_up, "_hc_fp8_scale")
            ):
                return True
            return orig(hyper_input_normed, w_down, w_up)

        return fused_hc_mix_supported

    def make_fused(orig):
        def fused_hc_mix(hyper_input_normed, w_down, w_up, hc, hs):
            rows, k = hyper_input_normed.shape
            lowrank = w_down.shape[0]
            if w_down.dtype == torch.float8_e4m3fn and hasattr(w_down, "_hc_fp8_scale"):
                sd = w_down._hc_fp8_scale
                su = w_up._hc_fp8_scale
                if rows == 1 and lowrank <= 1024 and hyper_input_normed.is_cuda:
                    if k % (10 * 512) == 0:
                        sk, bn, bk, bj = 10, 4, 512, 8
                    elif k % (8 * 256) == 0:
                        sk, bn, bk, bj = 8, 4, 256, 8
                    else:
                        sk = 0
                    if sk:
                        down_k, _, up_k_fp8, _, _, _, _, _, _ = _get_kernels()
                        low_pad = 1 << (lowrank - 1).bit_length()
                        key = (hyper_input_normed.device, lowrank, sk)
                        t_buf = _BUFS.get(key)
                        if t_buf is None:
                            t_buf = _BUFS[key] = torch.empty(
                                (sk, lowrank), dtype=torch.float32, device=hyper_input_normed.device
                            )
                        out = torch.empty((1, hs), dtype=hyper_input_normed.dtype, device=hyper_input_normed.device)
                        down_k[(triton.cdiv(lowrank, bn), sk)](
                            hyper_input_normed, w_down, t_buf,
                            K=k, LOW=lowrank, BN=bn, BK=bk, SPLIT_K=sk, num_warps=4,
                        )
                        up_k_fp8[(triton.cdiv(hs, bj),)](
                            hyper_input_normed, w_up, sd, su, t_buf, out,
                            HS=hs, LOW=lowrank, LOW_PAD=low_pad, HC=hc, BJ=bj, SPLIT_K=sk,
                            inv_hc=1.0 / hc, num_warps=4,
                        )
                        return out

                # Prefill M>=64: FP8 _scaled_mm + fused silu_quant + fused gate_mean (4096-row chunks)
                dt = hyper_input_normed.dtype
                if (
                    rows >= 64
                    and hyper_input_normed.is_cuda
                    and dt == torch.bfloat16
                    and k % 16 == 0
                    and lowrank % 16 == 0
                    and lowrank <= 1024
                ):
                    _, _, _, q_tok_k, silu_q_k, gate_mean_k, _, _, _ = _get_kernels()
                    low_pad = 1 << (lowrank - 1).bit_length()
                    wd_t = w_down.t()
                    wu_t = w_up.t()
                    sd_row = sd[None, :]
                    su_row = su[None, :]
                    out = torch.empty((rows, hs), dtype=dt, device=hyper_input_normed.device)
                    bk_q = 1024 if k % 1024 == 0 else 512
                    for i in range(0, rows, 4096):
                        xc = hyper_input_normed[i : i + 4096].contiguous()
                        rc = xc.shape[0]
                        pad = (16 - (rc % 16)) % 16
                        xp = F.pad(xc, (0, 0, 0, pad)) if pad else xc
                        rp = xp.shape[0]
                        xq = torch.empty((rp, k), dtype=torch.float8_e4m3fn, device=xc.device)
                        xs = torch.empty((rp, 1), dtype=torch.float32, device=xc.device)
                        q_tok_k[(rp,)](xp, xq, xs, K=k, BK=bk_q, num_warps=4)
                        t = torch._scaled_mm(xq, wd_t, scale_a=xs, scale_b=sd_row, out_dtype=dt)
                        tq = torch.empty((rp, lowrank), dtype=torch.float8_e4m3fn, device=xc.device)
                        ts = torch.empty((rp, 1), dtype=torch.float32, device=xc.device)
                        silu_q_k[(rp,)](t, tq, ts, K=lowrank, KPAD=low_pad, inv_hc=1.0 / hc, num_warps=4)
                        g = torch._scaled_mm(tq, wu_t, scale_a=ts, scale_b=su_row, out_dtype=dt)
                        gate_mean_k[(rc, triton.cdiv(hs, 256))](
                            xc, g, out[i : i + rc], HS=hs, HC=hc, BJ=256, num_warps=4
                        )
                    return out

                wd_d = w_down.to(dt) * sd[:, None].to(dt)
                wu_d = w_up.to(dt) * su[:, None].to(dt)
                t = F.silu(F.linear(hyper_input_normed, wd_d) / hc)
                g = torch.sigmoid(F.linear(t, wu_d)).unflatten(-1, (hc, hs))
                return (g * hyper_input_normed.unflatten(-1, (hc, hs))).mean(dim=-2)

            sk = _SPLIT_K
            if (
                not _ENABLED
                or rows != 1
                or k % (sk * 256) != 0
                or lowrank > 1024
                or not hyper_input_normed.is_cuda
            ):
                return orig(hyper_input_normed, w_down, w_up, hc, hs)
            down_k, up_k, _, _, _, _, _, _, _ = _get_kernels()
            low_pad = 1 << (lowrank - 1).bit_length()
            key = (hyper_input_normed.device, lowrank, sk)
            t_buf = _BUFS.get(key)
            if t_buf is None:
                t_buf = _BUFS[key] = torch.empty(
                    (sk, lowrank), dtype=torch.float32, device=hyper_input_normed.device
                )
            out = torch.empty((1, hs), dtype=hyper_input_normed.dtype, device=hyper_input_normed.device)
            bn, bj = 2, 8
            down_k[(triton.cdiv(lowrank, bn), sk)](
                hyper_input_normed, w_down, t_buf,
                K=k, LOW=lowrank, BN=bn, BK=256, SPLIT_K=sk, num_warps=4,
            )
            up_k[(triton.cdiv(hs, bj),)](
                hyper_input_normed, w_up, t_buf, out,
                HS=hs, LOW=lowrank, LOW_PAD=low_pad, HC=hc, BJ=bj, SPLIT_K=sk,
                inv_hc=1.0 / hc, num_warps=4,
            )
            return out

        return fused_hc_mix

    wrap(m, "fused_hc_mix_supported", ["hyper_input_normed", "w_down", "w_up"], make_supported)
    wrap(m, "fused_hc_mix", ["hyper_input_normed", "w_down", "w_up", "hc", "hs"], make_fused)


def _install_unquant(m):
    import torch

    if not (_DENSE_FP8 or _DENSE_BF16_GEMV):
        return
    cls = getattr(m, "UnquantizedLinearMethod", None)
    if cls is None:
        return

    def make_apply(orig):
        def apply(self, layer, x, bias=None):
            s = getattr(layer, "_dense_fp8_scale", None)
            if s is not None:
                return _fp8_linear(x, layer.weight, s, bias)
            if (
                _DENSE_BF16_GEMV
                and bias is None
                and x.is_cuda
                and x.dtype == torch.bfloat16
                and x.numel() == x.shape[-1]
                and layer.weight.dtype == torch.bfloat16
                and layer.weight.shape[0] <= 1024
                and layer.weight.shape[1] % 640 == 0
            ):
                return _bf16_small_gemv(x, layer.weight)
            return orig(self, layer, x, bias)

        return apply

    wrap(cls, "apply", ["self", "layer", "x", "bias"], make_apply)


def _install_qwen35(m):
    import torch

    if not _DENSE_FP8:
        return
    cls = getattr(m, "Qwen3_5GatedDeltaNet", None)
    if cls is None:
        return

    def make_fwd(orig):
        def _forward_input_proj(self, hidden_states):
            s = getattr(self, "_fused_in_proj_fp8_scale", None)
            fw = getattr(self, "_fused_in_proj_weight", None)
            if (
                s is not None
                and fw is not None
                and not isinstance(hidden_states, tuple)
                and hidden_states.dtype == torch.bfloat16
                and hidden_states.shape[0] <= 1024
            ):
                fused_out = _fp8_linear(hidden_states, fw, s, None)
                w1 = self._fused_in_proj_qkvz_width
                return fused_out[:, :w1], fused_out[:, w1:]
            return orig(self, hidden_states)

        return _forward_input_proj

    wrap(cls, "_forward_input_proj", ["self", "hidden_states"], make_fwd)


def _install_logits(m):
    if not (_DENSE_FP8 and _DENSE_FP8_LM_HEAD):
        return
    cls = getattr(m, "LogitsProcessor", None)
    if cls is None:
        return

    def make_compute(orig):
        def _compute_lm_head(self, hidden_states, lm_head, embedding_bias=None, out=None):
            s = getattr(lm_head, "_dense_fp8_scale", None)
            if s is not None and not (getattr(self, "use_fp32_lm_head", False) or getattr(self, "use_fp32_lm_head_output", False)):
                return _fp8_linear(hidden_states, lm_head.weight, s, embedding_bias)
            return orig(self, hidden_states, lm_head, embedding_bias, out=out)

        return _compute_lm_head

    wrap(cls, "_compute_lm_head", ["self", "hidden_states", "lm_head", "embedding_bias", "out"], make_compute)


def _install_loader(m):
    import torch

    if not (_FP8_ENABLED or _DENSE_FP8):
        return
    L = getattr(m, "DefaultModelLoader", None)
    if L is None:
        return

    def make_postprocess(orig):
        def postprocess_weights(model, target_device):
            got = orig(model, target_device)
            did = _quantize_model(model)
            if did:
                torch.cuda.empty_cache()
            return got

        return postprocess_weights

    wrap(L, "postprocess_weights", ["model", "target_device"], make_postprocess)


after_import("sglang.kernels.ops.gemm.hc_mix", _install_hc_mix)
after_import("sglang.srt.layers.quantization.unquant", _install_unquant)
after_import("sglang.srt.models.qwen3_5", _install_qwen35)
after_import("sglang.srt.layers.logits_processor", _install_logits)
after_import("sglang.srt.model_loader.loader", _install_loader)
