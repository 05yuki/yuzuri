"""Model loading:
- SGLANG_UNBALANCED_LOAD_TIMEOUT_S: the barrier after an unbalanced load (480 s upstream). Loading CPU experts can
  take longer than that.
- KT_SM75_BF16=1 keeps bf16 below SM80: DSV4's JIT kernels take bf16, not fp16."""
import os

from yuzuri._finder import after_import
from yuzuri.hooks._seam import wrap

REPLACES = ["srt/model_executor/model_runner_components/load_model_utils.py"]


def _install(m):
    if os.environ.get("SGLANG_UNBALANCED_LOAD_TIMEOUT_S"):
        if not isinstance(getattr(m, "UNBALANCED_MODEL_LOADING_TIMEOUT_S", None), int):
            from yuzuri.hooks._seam import SeamError

            raise SeamError(f"yuzuri: {m.__name__} has no UNBALANCED_MODEL_LOADING_TIMEOUT_S")
        m.UNBALANCED_MODEL_LOADING_TIMEOUT_S = int(os.environ["SGLANG_UNBALANCED_LOAD_TIMEOUT_S"])

    def make(orig):
        def maybe_downgrade_dtype_for_legacy_gpu(*, model_config):
            import torch

            if torch.cuda.get_device_capability()[0] < 8 and os.environ.get("KT_SM75_BF16") == "1":
                return None
            return orig(model_config=model_config)

        return maybe_downgrade_dtype_for_legacy_gpu

    wrap(m, "maybe_downgrade_dtype_for_legacy_gpu", ["model_config"], make)


after_import("sglang.srt.model_executor.model_runner_components.load_model_utils", _install)
