"""KT experts and the routed scaling factor: DeepseekV2MoE scales the experts' output whenever the experts are KT's,
also where the top-k already folded the factor into its weights (KT applies the weights as given), so such a model's
routed output was scaled twice. The block's test of the experts' method answers no for a block whose top-k folds the
factor; nothing else of the block changes."""
import inspect

from yuzuri._finder import after_import
from yuzuri.hooks._seam import SeamError, wrap

REPLACES = ["srt/models/deepseek_v2.py"]

_TEST = "isinstance(self.experts.quant_method, KTEPWrapperMethod)"


def _install(m):
    real = getattr(m, "KTEPWrapperMethod", None)
    src = inspect.getsource(m.DeepseekV2MoE)
    if real is None or _TEST not in src:
        raise SeamError(f"yuzuri: sglang.srt.models.deepseek_v2.DeepseekV2MoE no longer has {_TEST!r}: the hook "
                        "answers that test and has nothing to attach to in this SGLang")

    class _Unfolded(type):
        def __instancecheck__(cls, obj):
            return isinstance(obj, real) and not getattr(obj, "_yuzuri_scale_in_topk", False)

    class KTEPWrapperMethod(metaclass=_Unfolded):
        """Stands for kt_ep_wrapper.KTEPWrapperMethod in this module's isinstance tests."""

    m.KTEPWrapperMethod = KTEPWrapperMethod

    def make(orig):
        def __init__(self, *args, **kwargs):
            orig(self, *args, **kwargs)
            method = getattr(getattr(self, "experts", None), "quant_method", None)
            if isinstance(method, real):
                method._yuzuri_scale_in_topk = bool(getattr(self, "routed_scaling_factor_fused_in_topk", False))

        return __init__

    wrap(m.DeepseekV2MoE, "__init__", ["self", "config", "layer_id", "..."], make)


after_import("sglang.srt.models.deepseek_v2", _install)
