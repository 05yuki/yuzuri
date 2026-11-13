"""Below SM80 cuBLAS has no bf16-in / fp32-out GEMM: linear_bf16_fp32 runs on fp32 operands there (bf16 values and
their products are exact in fp32; only the order of the sums differs)."""
from yuzuri._finder import after_import
from yuzuri.hooks._seam import wrap

REPLACES = ["kernels/ops/gemm/bf16_fp32.py"]


def _install(m):
    import torch

    if not (torch.cuda.is_available() and torch.version.hip is None and torch.cuda.get_device_capability() < (8, 0)):
        return

    def make(orig):
        def linear_bf16_fp32(x, y, *, hpc_kernel_min_m=None):
            return torch.nn.functional.linear(x.float(), y.float())

        return linear_bf16_fp32

    wrap(m, "linear_bf16_fp32", ["x", "y", "hpc_kernel_min_m"], make)


after_import("sglang.kernels.ops.gemm.bf16_fp32", _install)
