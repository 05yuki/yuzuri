"""Decode-sized and SM100+ block-[32,32] UE8M0 prefill FP8 GEMMs on kernels made for them.
w8a8_block_fp8_matmul_triton keeps every other call.

  M <= 4    fp8_block_gemv: steps K by several scale groups (the block kernel steps by one).
            SGLANG_FP8_GEMV=0 turns it off.
  M <= 16   where SGLang has no tuned config for the shape: its split-K kernel with a 16-row tile. The default
            tiling launches N/128 CTAs, each streaming a K x 128 strip alone (250-500 GB/s against ~900 on an
            RTX 5070 Ti); measured 3-6x faster on N=2048..7168 and ~1.2x on N=16384+. SGLANG_FP8_SKINNY=0 turns it
            off; SGLANG_FP8_SKINNY_SPLIT_K / _BLOCK_N override the tile.
  M >= 32   on SM100+ (Blackwell) with block_size == [32, 32] and exact UE8M0 (power-of-two) fp32 scales:
            FlashInfer's cuDNN MXFP8 GEMM (mm_mxfp8, backend="cudnn"), converting the fp32 scales to swizzled
            UE8M0 bytes on the fly (zero permanent VRAM). SGLANG_FP8_PREFILL_MXFP8=0 turns it off.
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

    _mxfp8_cap = {}
    _ue8m0_ok = {}
    _mxfp8_calls = [0]
    _mxfp8_mod = [None]  # False if unavailable, else (mm_mxfp8, block_scale_interleave)

    def _get_mxfp8():
        if _mxfp8_mod[0] is None:
            try:
                import flashinfer

                if hasattr(flashinfer, "mm_mxfp8") and hasattr(flashinfer, "block_scale_interleave"):
                    _mxfp8_mod[0] = (flashinfer.mm_mxfp8, flashinfer.block_scale_interleave)
                else:
                    _mxfp8_mod[0] = False
            except Exception:
                _mxfp8_mod[0] = False
        return _mxfp8_mod[0]

    def _is_blackwell(dev) -> bool:
        idx = dev.index if dev.index is not None else torch.cuda.current_device()
        ok = _mxfp8_cap.get(idx)
        if ok is None:
            ok = _mxfp8_cap[idx] = torch.cuda.get_device_capability(idx)[0] >= 10
        return ok

    def _try_mxfp8(A, B, As, Bs, block_size, output_dtype):
        if os.environ.get("SGLANG_FP8_PREFILL_MXFP8", "1") != "1":
            return None
        if block_size[0] != 32 or block_size[1] != 32:
            return None
        if A.dtype != torch.float8_e4m3fn or B.dtype != torch.float8_e4m3fn:
            return None
        if As.dtype != torch.float32 or Bs.dtype != torch.float32:
            return None
        if output_dtype not in (torch.bfloat16, torch.float16):
            return None
        K = A.shape[-1]
        M = A.numel() // K
        N = B.shape[0]
        min_m = int(os.environ.get("SGLANG_FP8_PREFILL_MXFP8_MIN_M", "32") or 32)
        if M < min_m or N % 32 != 0 or K % 32 != 0:
            return None
        if not _is_blackwell(A.device):
            return None
        # Under KT_PREFILL_LEND, keep FlashInfer's workspace inside TAG_SCRATCH so decode loses zero hot slots.
        try:
            from sglang.srt.layers.moe import kt_lend
        except ImportError:
            kt_lend = None
        if kt_lend is not None and kt_lend.ENABLED and kt_lend.phase() != "prefill":
            return None
        fns = _get_mxfp8()
        if not fns:
            return None
        As2 = As.reshape(M, -1)
        if As2.shape[1] != K // 32 or Bs.shape != (N // 32, K // 32):
            return None
        # Verify exact positive UE8M0 (zero mantissa bits, >0) once per weight tensor; never sync inside CUDA graph capture.
        key = (Bs.device.index, Bs.data_ptr(), Bs.shape)
        ok = _ue8m0_ok.get(key)
        if ok is None:
            if torch.cuda.is_current_stream_capturing():
                return None
            bs_c = Bs.contiguous()
            as_c = As2.contiguous()
            bs_exact = not bool((bs_c.view(torch.int32) & 0x7FFFFF).any().item()) and bool((bs_c > 0).all().item())
            as_exact = not bool((as_c.view(torch.int32) & 0x7FFFFF).any().item()) and bool((as_c > 0).all().item())
            ok = _ue8m0_ok[key] = bool(bs_exact and as_exact)
        if not ok:
            return None
        mm_mxfp8, block_scale_interleave = fns
        as_u8 = ((As2.contiguous().view(torch.int32) >> 23) & 0xFF).to(torch.uint8)
        bs_u8 = ((Bs.contiguous().view(torch.int32) >> 23) & 0xFF).to(torch.uint8).repeat_interleave(32, dim=0)
        as_swiz = block_scale_interleave(as_u8)
        bs_swiz = block_scale_interleave(bs_u8)
        A2 = A.reshape(M, K).contiguous()
        B2 = B.contiguous()
        backend = os.environ.get("SGLANG_FP8_PREFILL_MXFP8_BACKEND", "")
        if not backend:
            backend = "b12x" if (M < 1024 and K % 128 == 0) else "cudnn"
        out = torch.empty((M, N), device=A.device, dtype=output_dtype)
        if kt_lend is not None and kt_lend.ENABLED:
            with kt_lend.scratch_region():
                mm_mxfp8(A2, B2.T, as_swiz, bs_swiz, out=out, out_dtype=output_dtype, backend=backend)
        else:
            mm_mxfp8(A2, B2.T, as_swiz, bs_swiz, out=out, out_dtype=output_dtype, backend=backend)
        _mxfp8_calls[0] += 1
        if _mxfp8_calls[0] == 1:
            m.logger.info("[yuzuri] fp8 block-[32,32] prefill using flashinfer.mm_mxfp8(backend=%s), first shape M=%d N=%d K=%d",
                          backend, M, N, K)
        return out.view(*A.shape[:-1], N)

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
            if A.shape[-1] == 0 or theirs():
                return orig(A, B, As, Bs, block_size, output_dtype)
            rows = A.numel() // A.shape[-1]
            if rows > 16:
                fast = _try_mxfp8(A, B, As, Bs, block_size, output_dtype)
                if fast is not None:
                    return fast
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
