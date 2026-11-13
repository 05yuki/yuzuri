"""Triton has no e4m3 type below SM89: the compressed KV's scale is rounded to e4m3 in integer arithmetic there
(nearest, ties to even: the value the cast gives). SM89 and newer keep SGLang's kernel as it is."""
from yuzuri._finder import after_import
from yuzuri.hooks._seam import define, rewrite

REPLACES = ["kernels/ops/attention/dsv4/fp4_rope_fake_quant.py"]
_ROUND = '''@triton.jit
def _round_e4m3(x):
    """A positive float32 in [2^-9, 448] rounded to the nearest e4m3 value (ties to even), as a float32."""
    e = tl.maximum(((x.to(tl.int32, bitcast=True) >> 23) & 0xFF) - 127, -6)
    q = ((e + 124) << 23).to(tl.float32, bitcast=True)  # 2^(e - 3), the e4m3 spacing at x
    return tl.minimum(libdevice.rint(x / q) * q, 448.0)
'''


def install(m, force=False):
    import torch

    if not force and not (torch.cuda.is_available() and torch.version.hip is None
                          and torch.cuda.get_device_capability() < (8, 9)):
        return
    define(m, _ROUND, "fp4_rope_fake_quant._round_e4m3")
    rewrite(m, "rope_tail_fake_quant_fp4_row",
            [("scale = scale.to(tl.float8e4nv).to(tl.float32)", "scale = _round_e4m3(scale)")])


after_import("sglang.kernels.ops.attention.dsv4.fp4_rope_fake_quant", install)
