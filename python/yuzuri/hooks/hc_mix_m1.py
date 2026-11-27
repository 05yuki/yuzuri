"""Barrier-free Split-K M=1 GEMV for HyperConnection low-rank mix (KT_HC_MIX_M1=1).

Replaces sglang's persistent ROWS=16 tl.dot kernel on single-row (M=1) decode calls
with a two-stage Split-K GEMV (no ROWS=16 padding, no software grid barrier, 100% BF16).
"""
import os

from yuzuri._finder import after_import
from yuzuri.hooks._seam import wrap

REPLACES = ["kernels/ops/gemm/hc_mix.py"]
_ENABLED = os.environ.get("KT_HC_MIX_M1", "1") == "1"
_SPLIT_K = int(os.environ.get("KT_HC_MIX_M1_SPLIT_K", "8") or 8)
_KERNELS = None
_BUFS = {}


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

    _KERNELS = (_hc_down_sk, _hc_up_sk)
    return _KERNELS


def _install(m):
    import torch
    import triton

    if getattr(m, "_is_npu", False):
        return

    def _fused_hc_mix(orig, hyper_input_normed, w_down, w_up, hc, hs):
        rows, k = hyper_input_normed.shape
        lowrank = w_down.shape[0]
        sk = _SPLIT_K
        if (
            not _ENABLED
            or rows != 1
            or k % (sk * 256) != 0
            or lowrank > 1024
            or not hyper_input_normed.is_cuda
        ):
            return orig(hyper_input_normed, w_down, w_up, hc, hs)
        down_k, up_k = _get_kernels()
        low_pad = 1 << (lowrank - 1).bit_length()
        key = (hyper_input_normed.device, lowrank, sk)
        t_buf = _BUFS.get(key)
        if t_buf is None:
            t_buf = _BUFS[key] = torch.empty((sk, lowrank), dtype=torch.float32, device=hyper_input_normed.device)
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

    wrap(
        m,
        "fused_hc_mix",
        _fused_hc_mix,
        expected=("hyper_input_normed", "w_down", "w_up", "hc", "hs"),
    )


after_import("sglang.kernels.ops.gemm.hc_mix", _install)
