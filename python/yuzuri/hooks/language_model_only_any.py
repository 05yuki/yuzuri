"""--language-model-only for a model outside SGLang's list: a model whose implementation reads language_model_only
(skips its visual weights itself) takes the flag. The capability decides, not the name; SGLang's own checks before
the list run unchanged."""
from yuzuri._finder import after_import
from yuzuri.hooks._seam import wrap

REPLACES = ["srt/arg_groups/model_hook.py"]

_REFUSAL = "--language-model-only does not support"


def _reads_the_flag(server_args) -> bool:
    import inspect
    import sys

    try:
        from sglang.srt.arg_groups.model_hook import model_config_of
        from sglang.srt.models.registry import ModelRegistry

        cls, _ = ModelRegistry.resolve_model_cls(model_config_of(server_args).hf_config.architectures)
        return "language_model_only" in inspect.getsource(sys.modules[cls.__module__])
    except Exception:
        return False


def _install(m):
    def make(orig):
        def handle_language_model_only(server_args):
            try:
                return orig(server_args)
            except ValueError as e:
                # the list is the function's last check: nothing after it is skipped
                if _REFUSAL not in str(e) or not _reads_the_flag(server_args):
                    raise

        return handle_language_model_only

    wrap(m, "handle_language_model_only", ["server_args"], make, adds=("_model_handles_language_model_only",),
         needs=(_REFUSAL, "LANGUAGE_MODEL_ONLY_ARCHITECTURES"))


after_import("sglang.srt.arg_groups.model_hook", _install)
