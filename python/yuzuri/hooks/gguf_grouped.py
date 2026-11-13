"""A model file that multiplies by a grouped GGUF weight group by group (a function of a [M, G, K] input, qweight,
qweight_type, the rows of a group and a matmul_fn) does it through kt_gguf_dense_ops.grouped_mul_mat_gguf: every
group in one launch at decode, MMQ or one batched GEMM at prefill sizes. The model file is found by that shape, not by name; a
call that passes its own matmul_fn, or a weight no path there takes, runs the model file's loop."""
import inspect
import sys

from yuzuri._finder import after_import
from yuzuri.hooks._seam import wrap

REPLACES = []
_TAIL = ["qweight", "qweight_type"]


def _install(m):
    for name, fn in list(vars(m).items()):
        if not inspect.isfunction(fn) or fn.__module__ != m.__name__:
            continue
        p = list(inspect.signature(fn).parameters)
        if len(p) != 5 or p[1:3] != _TAIL or p[4] != "matmul_fn":
            continue

        def make(orig):
            def grouped(x, qweight, qweight_type, rank, matmul_fn=None):
                if matmul_fn is None and x.dim() == 3:
                    from sglang.srt.layers.quantization.kt_gguf_dense_ops import grouped_mul_mat_gguf

                    out = grouped_mul_mat_gguf(x, qweight, qweight_type, rank)
                    if out is not None:
                        return out
                return orig(x, qweight, qweight_type, rank, matmul_fn)

            return grouped

        wrap(m, name, p, make)
        print(f"[yuzuri] grouped GGUF product: {m.__name__}.{name}", file=sys.stderr, flush=True)


after_import("sglang.srt.models.*", _install)
