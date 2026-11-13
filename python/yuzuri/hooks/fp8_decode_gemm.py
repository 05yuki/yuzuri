"""Decode-sized block-FP8 GEMMs on kernels made for them. w8a8_block_fp8_matmul_triton keeps every other call.

  M <= 4    fp8_block_gemv: steps K by several scale groups (the block kernel steps by one).
            SGLANG_FP8_GEMV=0 turns it off.
  M <= 16   where SGLang has no tuned config for the shape: its split-K kernel with a 16-row tile. The default
            tiling launches N/128 CTAs, each streaming a K x 128 strip alone (250-500 GB/s against ~900 on an
            RTX 5070 Ti); measured 3-6x faster on N=2048..7168 and ~1.2x on N=16384+. SGLANG_FP8_SKINNY=0 turns it
            off; SGLANG_FP8_SKINNY_SPLIT_K / _BLOCK_N override the tile.
"""
import os

from yuzuri._finder import after_import
from yuzuri.hooks._seam import wrap

REPLACES = ["kernels/ops/gemm/fp8_kernel.py"]
_GEMV = os.environ.get("SGLANG_FP8_GEMV", "1") == "1"
_seen = set()


def _install(m):
    import torch
    import triton

    # the kernels below are the CUDA ones: a module built for another platform keeps its own
    if getattr(m, "_is_hip", False) or getattr(m, "_is_gfx1250", False):
        return

    def theirs() -> bool:
        """Calls SGLang keeps whole: a HIP platform, and deterministic inference (batch-invariant ops: the kernel
        here changes with the row count and accumulates K in split partials)."""
        if m.get_platform().is_hip:
            return True
        ctx, ex = getattr(m, "get_context", None), getattr(m, "get_exec", None)
        return bool(ctx and ex and ctx().is_config_namespace_published("exec")
                    and ex().deterministic.enable_deterministic_inference)

    def make(orig):
        def w8a8_block_fp8_matmul_triton(A, B, As, Bs, block_size, output_dtype=torch.float16):
            # the row count as prepare_block_fp8_matmul_inputs gives it, before anything is allocated
            if A.shape[-1] == 0 or A.numel() // A.shape[-1] > 16 or theirs():
                return orig(A, B, As, Bs, block_size, output_dtype)
            skinny = os.environ.get("SGLANG_FP8_SKINNY", "1") == "1"
            M, N, K, C = m.prepare_block_fp8_matmul_inputs(A, B, As, Bs, block_size, output_dtype)
            block_n, block_k = block_size
            if M <= 4 and _GEMV and As.dtype == torch.float and Bs.dtype == torch.float:
                from sglang.kernels.ops.quantization.fp8_block_gemv import fp8_block_gemv

                fp8_block_gemv(A.view(M, K), B, As.reshape(M, -1), Bs, block_size, C.view(M, N))
                return C
            if not skinny or m.get_w8a8_block_fp8_configs(N, K, block_n, block_k):
                return orig(A, B, As, Bs, block_size, output_dtype)
            config = {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": int(os.environ.get("SGLANG_FP8_SKINNY_BLOCK_N", "128")),
                "BLOCK_SIZE_K": block_k,
                "GROUP_SIZE_M": 1,
                "num_warps": 4,
                "num_stages": 3,
                "SWAP_AB": False,
                "SPLIT_K": int(os.environ.get("SGLANG_FP8_SKINNY_SPLIT_K", "4")),
            }
            if os.environ.get("SGLANG_FP8_SKINNY_LOG") == "1" and (M, N, K) not in _seen:
                _seen.add((M, N, K))
                m.logger.info("fp8 skinny gemm shape M=%d N=%d K=%d", M, N, K)
            # split-K accumulates K in SPLIT_K fp32 partials, reduced after
            split_k = config["SPLIT_K"]
            assert split_k & (split_k - 1) == 0
            partials = torch.empty((split_k, M, N), device=A.device, dtype=torch.float32) if split_k > 1 else C
            blocks = triton.cdiv(M, config["BLOCK_SIZE_M"]) * triton.cdiv(N, config["BLOCK_SIZE_N"])
            m._w8a8_block_fp8_matmul_hopper[(blocks, split_k)](
                A, B, partials, As, Bs, M, N, K, block_n, block_k,
                A.stride(-2), A.stride(-1), B.stride(1), B.stride(0), C.stride(-2), C.stride(-1),
                As.stride(-2), As.stride(-1), Bs.stride(1), Bs.stride(0),
                **config, needs_masking=bool(K % config["BLOCK_SIZE_K"] != 0),
            )
            if split_k > 1:
                m._reduce_block_fp8_split_k[(triton.cdiv(M * N, 256),)](partials, C, M * N, split_k, 256)
            return C

        return w8a8_block_fp8_matmul_triton

    wrap(m, "w8a8_block_fp8_matmul_triton", ["A", "B", "As", "Bs", "block_size", "output_dtype"], make,
         adds=("fp8_block_gemv", "skinny"),
         needs=("prepare_block_fp8_matmul_inputs(A, B, As, Bs, block_size, output_dtype)",
                "get_w8a8_block_fp8_configs(N, K, block_size[0], block_size[1])",
                "get_platform()", "_w8a8_block_fp8_matmul_hopper",
                "_reduce_block_fp8_split_k[(triton.cdiv(M * N, 256),)]",
                "needs_masking=needs_masking"))


after_import("sglang.kernels.ops.gemm.fp8_kernel", _install)
