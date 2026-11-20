#!/usr/bin/env python3
"""kt-kernel's SwiGLU clamp (swiglu_limit, GLM-5.x's 10.0) for NVFP4 experts as well: the clamp lives in the
shared AVX2 MoE activation, and NVFP4 weights run the same FP4 kernel as MXFP4 (only the scale encoding differs),
but three python gates refuse the limit for every method they do not name. They take NVFP4 too. The gates are
found by the statement around them, so the methods a kt-kernel version names there do not matter.

The swigluoai alpha (gate * sigmoid(alpha * gate) * (up + 1)) reached MOEConfig in the MXFP8 branch
only, so a swigluoai model's NVFP4 experts ran plain SiLU: layer 3 against a float32 reference from the checkpoint
bytes, cosine 0.06-0.17 a token (rel. error 0.99); with the alpha set for every method, cosine 1.0 (rel. 0.0043).

Usage: python kt-kernel-clamp-nvfp4.py <site-packages/kt_kernel>
"""
import re
import sys
from pathlib import Path

K = Path(sys.argv[1])
n = 0


def add_nvfp4(rel, head, tail):
    """`head (methods) tail` with "NVFP4" among the methods, whichever methods the tuple names and however it is
    laid out; 1 when the file changed."""
    p = K / rel
    s = p.read_text()
    found = list(re.finditer(re.escape(head) + r"\(([^()]*)\)(:[^\n]*\n)" + re.escape(tail), s))
    if len(found) != 1:
        raise SystemExit(f"anchor found {len(found)} times in {p}:\n{head}(...):\n{tail}")
    m = found[0]
    body = m.group(1)
    if '"NVFP4"' in body:
        return 0
    last = list(re.finditer(r'"[A-Z0-9_]+"', body))[-1]
    if "\n" in body:  # one method a line
        indent = body[: last.start()].rsplit("\n", 1)[1]
        body = body[: last.end()] + ",\n" + indent + '"NVFP4"' + body[last.end():]
    else:
        body = body[: last.end()] + ', "NVFP4"' + body[last.end():]
    p.write_text(s[: m.start(1)] + body + s[m.end(1):])
    return 1


# the three gates that name the methods the clamp is allowed for
n += add_nvfp4("experts.py", "    if method in ",
               '        extra_kwargs["swiglu_limit"] = swiglu_limit\n        extra_kwargs["swiglu_alpha"] = swiglu_alpha\n')
n += add_nvfp4("utils/amx.py", "        if swiglu_limit != 0.0 and method not in ",
               '            raise ValueError(\n                f"NativeMoEWrapper received swiglu_limit=')
n += add_nvfp4("utils/amx.py", "        if self.swiglu_limit != 0.0 and self.method not in ",
               '            raise ValueError(\n                f"NativeMoEWrapper.load_weights: swiglu_limit=')

for rel, mark, old, new in (  # mark: what an applied edit leaves, whatever a comment beside it says
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
