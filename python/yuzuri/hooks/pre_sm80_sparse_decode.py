"""Below SM80 Triton has no FP8 type and no bf16 dot: the fused sparse MLA decode kernel reads the KV's e4m3 bytes
as float32 built from their bit fields (e4m3fn; no NaN bytes are written) and takes its dot products in float32
there. SM80 and newer keep SGLang's kernel as it is."""
from yuzuri._finder import after_import
from yuzuri.hooks._seam import define, rewrite

REPLACES = ["kernels/ops/attention/dsv4/triton_sparse_decode/triton_mla_kernels_decode_fused.py"]
_HELPERS = '''@triton.jit
def _kv_fp8(x_u8):
    """e4m3 bytes as float32 values."""
    x = x_u8.to(tl.int32)
    e = (x >> 3) & 0xF
    m = x & 7
    normal = (((x >> 7) << 31) | ((e + 120) << 23) | (m << 20)).to(tl.float32, bitcast=True)
    sub = m.to(tl.float32) * 0.001953125
    sub = tl.where((x >> 7) != 0, -sub, sub)
    return tl.where(e == 0, sub, normal)


@triton.jit
def _kdot(a, b):
    return tl.dot(a.to(tl.float32), b.to(tl.float32), input_precision="ieee")
'''


def install(m, force=False):
    import torch

    if not force and not (torch.cuda.is_available() and torch.version.hip is None
                          and torch.cuda.get_device_capability() < (8, 0)):
        return
    define(m, _HELPERS, "triton_mla_kernels_decode_fused._kv_fp8, _kdot")
    edits = [(f"nope_uint8_{i}.to(_KV_FP8_TY, bitcast=True)", f"_kv_fp8(nope_uint8_{i})") for i in range(7)]
    edits += [(f"tl.dot(q_{i}, tl.trans(kv_{i}))", f"_kdot(q_{i}, tl.trans(kv_{i}))") for i in range(8)]
    edits += [(f"tl.dot(p_bf16, kv_{i})", f"_kdot(p_bf16, kv_{i})") for i in range(8)]
    rewrite(m, "_process_kv_block_aggressive", edits, adds=("_kv_fp8", "_kdot"))


after_import("sglang.kernels.ops.attention.dsv4.triton_sparse_decode.triton_mla_kernels_decode_fused", install)
