"""SGLANG_MXFP4_MOE_FP32=1: an MXFP4 expert checkpoint under Fp8Config runs its GPU experts in Mxfp4Fp32MoEMethod (the
checkpoint's layout, one rounding) where SGLang picks Marlin."""
import os

from yuzuri._finder import after_import
from yuzuri.hooks._seam import wrap

REPLACES = ["srt/layers/quantization/fp8.py", "srt/layers/quantization/mxfp4_marlin_moe.py"]


_MADE = {}


def _method(marlin):
    """marlin (Mxfp4MarlinMoEMethod) in the checkpoint's layout through Triton kernels with one rounding. The Marlin
    runner is still created for its config (the SwiGLU clamp the streamed prefill and the kernels read); its kernel
    is not used."""
    if marlin in _MADE:
        return _MADE[marlin]

    class Mxfp4Fp32MoEMethod(marlin):
        def process_weights_after_loading(self, layer) -> None:
            self._fp8.process_weights_after_loading(layer)
            if getattr(layer, "_mega_moe_weights_built", False):
                return
            if self.runner.config.gemm1_alpha is not None:
                raise RuntimeError("SGLANG_MXFP4_MOE_FP32: gemm1_alpha (GPT-OSS SwiGLU) is not supported")
            layer._dsv4_mxfp4_backend = "mxfp4_fp32"

        def apply(self, layer, dispatch_output):
            from sglang.srt.layers.moe.mxfp4_fp32_moe import mxfp4_fp32_moe
            from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

            topk_output = dispatch_output.topk_output
            out = mxfp4_fp32_moe(
                dispatch_output.hidden_states,
                layer.w13_weight,
                layer.w13_weight_scale_inv,
                layer.w2_weight,
                layer.w2_weight_scale_inv,
                topk_output.topk_weights,
                topk_output.topk_ids,
                swiglu_limit=self.runner.config.swiglu_limit,
            )
            return StandardCombineInput(hidden_states=out)

    _MADE[marlin] = Mxfp4Fp32MoEMethod
    return Mxfp4Fp32MoEMethod


def _install(m):
    def make(orig):
        def get_quant_method(self, layer, prefix):
            got = orig(self, layer, prefix)
            if os.environ.get("SGLANG_MXFP4_MOE_FP32") == "1" and type(got).__name__ == "Mxfp4MarlinMoEMethod":
                return _method(type(got))(got._fp8, prefix=got.prefix)
            return got

        return get_quant_method

    wrap(m.Fp8Config, "get_quant_method", ["self", "layer", "prefix"], make)


after_import("sglang.srt.layers.quantization.fp8", _install)
