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
        print("    effect: on an NVFP4 checkpoint the CPU arenas hold the scales folded to bf16, so the hot lend's")
        print("            slots cannot be refilled from them: the server logs 'dynamic slots off' and the slots keep")
        print("            the placement's experts (decode is slower on prompts unlike the profile's hot list).")
        print("    build:  kt-kernel with GeneralMOEConfig.scale_codes (kvcache-ai/ktransformers#2268),")
        print("            then run setup.sh --package again.")

    # 2. GGUF (LLAMAFILE) experts: a load-time check that every weight type has a CPU kernel
    guard = None
    if ext:
        try:
            with open(ext, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
                guard = m.find(b"this build has no CPU kernel for the") >= 0
        except OSError:
            guard = None
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
