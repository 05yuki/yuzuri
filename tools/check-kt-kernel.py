#!/usr/bin/env python3
"""What the venv's kt-kernel extension (the C++ side, which setup.sh --package does not rebuild) can do for Yuzuri.

Run with the server venv's python, after `import kt_kernel` works:

    <venv>/bin/python tools/check-kt-kernel.py

Prints one line per capability and, for each one the extension lacks, what Yuzuri turns off and what to build.
Exit 0 either way (a lacking capability costs speed or an expert format, not the server); 2 when kt-kernel does
not import.
"""
import mmap
import sys


def main() -> int:
    try:
        import kt_kernel  # noqa: F401
        from kt_kernel.utils.amx import MOEConfig
    except Exception as e:  # noqa: BLE001
        print(f"kt-kernel: does not import in this python ({type(e).__name__}: {e})", file=sys.stderr)
        return 2
    ext = next((getattr(m, "__file__", None) for k, m in sys.modules.items()
                if k.endswith("kt_kernel_ext") and getattr(m, "__file__", None)), None)
    print(f"kt-kernel extension: {ext or 'loaded, file unknown'}")

    # 1. NVFP4 block scales as the checkpoint's e4m3 bytes (GeneralMOEConfig.scale_codes)
    if hasattr(MOEConfig, "scale_codes"):
        print("  NVFP4 scale codes: yes")
    else:
        print("  NVFP4 scale codes: no")
        print("    effect: on an NVFP4 checkpoint the CPU arenas hold the block scales folded with scale_2 into fp32.")
        print("            The CPU experts read those, or one-byte codes the extension makes from them where it reads")
        print("            KT_MXFP4_SCALE_CODES. The hot lend's cuTile slots recover the checkpoint's bytes from the fp32")
        print("            scales, checked against the checkpoint at launch ('block scales recovered' in the log).")
        print("    build:  kt-kernel with GeneralMOEConfig.scale_codes (kvcache-ai/ktransformers#2268),")
        print("            then run setup.sh --package again.")

    # 2. GGUF (LLAMAFILE) experts: a load-time check that every weight type has a CPU kernel
    guard = None
    e8m0_arena = None
    if ext:
        try:
            with open(ext, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
                guard = m.find(b"this build has no CPU kernel for the") >= 0
                e8m0_arena = m.find(b"KT_MXFP4_SCALE_ARENA_E8M0") >= 0
        except OSError:
            guard = None
            e8m0_arena = None
    if guard:
        print("  GGUF kernel check at load: yes (a weight type without a CPU kernel stops the load with its name)")
    else:
        print("  GGUF kernel check at load: " + ("no" if guard is False else "unknown (extension file not read)"))
        print("    effect: only GGUF expert checkpoints. A weight type this build has no CPU kernel for (IQ2_S,")
        print("            IQ3_XXS, IQ3_S, IQ4_NL and Q2_0 are missing from builds made for V4.1's types alone) is")
        print("            not caught at load: the server aborts at CUDA graph capture with")
        print("            'terminate called after throwing std::runtime_error'.")
        print("    build:  kt-kernel with those types' kernels in third_party/llamafile and the load check;")
        print("            safetensors checkpoints (bf16, FP8, MXFP4, NVFP4) are not affected.")

    # 3. MXFP4 1-byte E8M0 scale arenas (KT_MXFP4_SCALE_ARENA_E8M0=1)
    if e8m0_arena:
        print("  MXFP4 1-byte E8M0 scale arenas: yes (saves ~68 GB host RAM on 1017-class MXFP4 models)")
    else:
        print("  MXFP4 1-byte E8M0 scale arenas: " + ("no" if e8m0_arena is False else "unknown (extension file not read)"))
        print("    effect: MXFP4 experts keep 4-byte fp32 scale arenas; 2/4-bit packed scale DMA (KT_GPU_STREAM_SCALE_PACK=1)")
        print("            still runs automatically and cuts scale H2D 13.3x, but host RAM is ~68 GB higher on 1017-class models.")
        print("    build:  kt-kernel with patches/kt-kernel-mxfp4-e8m0-arena.patch, then run setup.sh --package again.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
