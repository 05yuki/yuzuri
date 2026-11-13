#!/usr/bin/env python3
"""kt-kernel's SwiGLU clamp (swiglu_limit, GLM-5.x's 10.0) for NVFP4 experts as well: the clamp lives in the
shared AVX2 MoE activation, and NVFP4 weights run the same FP4 kernel as MXFP4 (only the scale encoding differs),
but three python gates refuse the limit for any method but MXFP4 / MXFP8. They take NVFP4 too.

The swigluoai alpha (gate * sigmoid(alpha * gate) * (up + 1)) reached MOEConfig in the MXFP8 branch
only, so a swigluoai model's NVFP4 experts ran plain SiLU: layer 3 against a float32 reference from the checkpoint
bytes, cosine 0.06-0.17 a token (rel. error 0.99); with the alpha set for every method, cosine 1.0 (rel. 0.0043).

Usage: python kt-kernel-clamp-nvfp4.py <site-packages/kt_kernel>
"""
import sys
from pathlib import Path

K = Path(sys.argv[1])
n = 0
for rel, mark, old, new in (  # mark: what an applied edit leaves, whatever a comment beside it says
    ("experts.py", 'if method in ("MXFP4", "MXFP8", "NVFP4"):', '''    if method in ("MXFP4", "MXFP8"):
        extra_kwargs["swiglu_limit"] = swiglu_limit
        extra_kwargs["swiglu_alpha"] = swiglu_alpha
''', '''    if method in ("MXFP4", "MXFP8", "NVFP4"):  # kt-kernel-clamp-nvfp4.py: the FP4 kernel clamps too
        extra_kwargs["swiglu_limit"] = swiglu_limit
        extra_kwargs["swiglu_alpha"] = swiglu_alpha
'''),
    ("utils/amx.py", 'if swiglu_limit != 0.0 and method not in ("MXFP4", "MXFP8", "NVFP4"):',
     '''        if swiglu_limit != 0.0 and method not in ("MXFP4", "MXFP8"):
            raise ValueError(
                f"NativeMoEWrapper received swiglu_limit={swiglu_limit} with "
''', '''        if swiglu_limit != 0.0 and method not in ("MXFP4", "MXFP8", "NVFP4"):  # kt-kernel-clamp-nvfp4.py
            raise ValueError(
                f"NativeMoEWrapper received swiglu_limit={swiglu_limit} with "
'''),
    ("utils/amx.py", 'if self.swiglu_limit != 0.0 and self.method not in ("MXFP4", "MXFP8", "NVFP4"):',
     '''        if self.swiglu_limit != 0.0 and self.method not in ("MXFP4", "MXFP8"):
            raise ValueError(
                f"NativeMoEWrapper.load_weights: swiglu_limit="
''', '''        if self.swiglu_limit != 0.0 and self.method not in ("MXFP4", "MXFP8", "NVFP4"):  # kt-kernel-clamp-nvfp4.py
            raise ValueError(
                f"NativeMoEWrapper.load_weights: swiglu_limit="
'''),
    ("utils/amx.py", 'moe_config.swiglu_alpha = getattr(self, "_swiglu_alpha", 0.0)\n        moe_config.activation_type',
     '''        moe_config.swiglu_limit = self.swiglu_limit
        moe_config.activation_type = (''', '''        moe_config.swiglu_limit = self.swiglu_limit
        # swigluoai (gate * sigmoid(alpha * gate) * (up + 1)) on every FP4/FP8 kernel; 0.0 is SiLU. It
        # was set in the MXFP8 branch only, so NVFP4 experts ran plain SiLU.
        moe_config.swiglu_alpha = getattr(self, "_swiglu_alpha", 0.0)
        moe_config.activation_type = ('''),
):
    p = K / rel
    s = p.read_text()
    if mark in s:
        continue
    if s.count(old) != 1:
        raise SystemExit(f"anchor found {s.count(old)} times in {p}:\n{old}")
    p.write_text(s.replace(old, new, 1))
    n += 1
print(f"patched {n} places under {K}" if n else f"already patched {K}")
