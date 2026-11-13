"""SGLANG_MARLIN_MOE_FP32_GLUE=1: the Marlin MoE's two steps between its GEMMs in fp32, each rounded once.

  SwiGLU        silu(gate) * up from the first GEMM's bf16 output, computed in fp32 (the clamp included)
  top-k sum     the second GEMM leaves the routes' outputs unweighted; they are weighted and summed in fp32

Without the variable nothing is attached and fused_marlin_moe is SGLang's.
"""
import os

from yuzuri._finder import after_import
from yuzuri.hooks._seam import define, rewrite

REPLACES = ["srt/layers/moe/fused_moe_triton/fused_marlin_moe.py"]
_KERNELS = '''@triton.jit
def _swiglu_fp32_kernel(in_ptr, out_ptr, d, limit, HAS_LIMIT: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = cols < d
    g = tl.load(in_ptr + row * 2 * d + cols, mask=m, other=0.0).to(tl.float32)
    u = tl.load(in_ptr + row * 2 * d + d + cols, mask=m, other=0.0).to(tl.float32)
    if HAS_LIMIT:
        g = tl.minimum(g, limit)
        u = tl.minimum(tl.maximum(u, -limit), limit)
    h = g / (1.0 + tl.exp(-g)) * u
    tl.store(out_ptr + row * d + cols, h.to(out_ptr.dtype.element_ty), mask=m)


def _swiglu_fp32(output: torch.Tensor, input: torch.Tensor, limit) -> None:
    """silu(gate) * up of [rows, 2d] bf16 (gate first) in fp32, one rounding into output."""
    rows, d = input.shape[0], input.shape[1] // 2
    if rows == 0:
        return
    assert input.is_contiguous() and output.is_contiguous()
    lim = float(limit) if limit else 0.0
    BLOCK = 1024
    _swiglu_fp32_kernel[(rows, triton.cdiv(d, BLOCK))](input, output, d, lim, HAS_LIMIT=lim > 0, BLOCK=BLOCK)


@triton.jit
def _topk_wsum_fp32_kernel(y_ptr, w_ptr, out_ptr, K, scale, TOPK: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = cols < K
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for t in tl.static_range(TOPK):
        wt = tl.load(w_ptr + row * TOPK + t).to(tl.float32)
        y = tl.load(y_ptr + (row * TOPK + t) * K + cols, mask=m, other=0.0).to(tl.float32)
        acc += wt * y
    tl.store(out_ptr + row * K + cols, (acc * scale).to(out_ptr.dtype.element_ty), mask=m)


def _topk_wsum_fp32(y: torch.Tensor, weights: torch.Tensor, output: torch.Tensor, scale: float) -> None:
    """output[m] = scale * sum_t weights[m, t] * y[m, t] in fp32, one rounding."""
    M, topk, K = y.shape
    if M == 0:
        return
    weights = weights.contiguous()
    assert y.is_contiguous() and output.is_contiguous() and weights.shape == (M, topk)
    BLOCK = 1024
    _topk_wsum_fp32_kernel[(M, triton.cdiv(K, BLOCK))](y, weights, output, K, float(scale), TOPK=topk, BLOCK=BLOCK)
'''
_ACT = '    elif activation == "silu" and is_gated and clamp_limit is not None:\n'
_ACT_FP32 = '''    elif activation == "silu" and is_gated:
        _swiglu_fp32(intermediate_cache2, intermediate_cache1.view(-1, gemm1_n), clamp_limit)
'''
_MUL = '        mul_topk_weights=True,\n'
_SUM = '    if is_mxfp4_marlin:\n        # Top-k weights (incl. routed scaling)'
_SUM_FP32 = '''    # the routes' outputs times their weights, summed in fp32; the MXFP4 path's weights already carry routed scaling
    _topk_wsum_fp32(intermediate_cache3, topk_weights, output,
                    1.0 if is_mxfp4_marlin or routed_scaling_factor is None else routed_scaling_factor)
    return output

'''


def _install(m):
    if os.environ.get("SGLANG_MARLIN_MOE_FP32_GLUE", "0") != "1":
        return
    define(m, _KERNELS, "the fp32 SwiGLU and top-k sum")
    rewrite(m, "fused_marlin_moe",
            [(_ACT, _ACT_FP32 + _ACT), (_MUL, "        mul_topk_weights=False,\n"), (_SUM, _SUM_FP32 + _SUM)],
            adds=("_swiglu_fp32", "_topk_wsum_fp32"))


after_import("sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe", _install)
