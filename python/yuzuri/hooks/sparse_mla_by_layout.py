"""The SM120/SM121 sparse MLA kernel by the latent's layout, not by the model's name.

SGLang picks flashinfer_sparse_mla for the architectures it names. The kernel itself reads a page of
_NOPE_DIM no-RoPE values and _ROPE_DIM RoPE values a token: a model SGLang does not name, with that latent and an
FP8 KV cache, runs on it as well. Three places decide by name, and each is left to SGLang for a model it accepts:

  overrides._dsa_split_backend_resolution         the backend a launch gets when the user names none: the kernel
                                                  where the layout is its own, TileLang's sparse MLA where it is not
  flash_mla_sm120._validate_flashinfer_sparse_mla_backend   the name check: a model whose layout fits passes it
  DeepseekSparseAttnBackend.__init__              where the layout is read (the validation gets the name only)
"""
import threading

from yuzuri._finder import after_import
from yuzuri.hooks._seam import wrap

REPLACES = [
    "kernels/ops/attention/flash_mla_sm120.py",
    "srt/arg_groups/overrides.py",
    "srt/layers/attention/dsa_backend.py",
]
_KERNEL = "sglang.kernels.ops.attention.flash_mla_sm120"
_BUILDING = threading.local()


def mla_layout_fits(text_config) -> bool:
    """Whether a model's MLA latent is the kernel's layout, read from its config."""
    import importlib

    k = importlib.import_module(_KERNEL)
    rank = getattr(text_config, "kv_lora_rank", None)
    rope = getattr(text_config, "qk_rope_head_dim", None)
    return rank == k._NOPE_DIM + k._ROPE_DIM and rope == k._ROPE_DIM


def _install_kernel(m):
    def make(orig):
        def _validate_flashinfer_sparse_mla_backend(*, model_arch, **kwargs):
            # a model the kernel's layout fits is validated as one SGLang names; any other as itself
            if model_arch not in m._GLM_DSA_MODEL_ARCHS and getattr(_BUILDING, "fits", False):
                model_arch = m._GLM_DSA_MODEL_ARCHS[0]
            return orig(model_arch=model_arch, **kwargs)

        return _validate_flashinfer_sparse_mla_backend

    wrap(m, "_validate_flashinfer_sparse_mla_backend",
         ["model_arch", "device_sm_major", "kv_cache_dtype", "prefill_impl", "decode_impl"], make,
         adds=("layout_fits",), needs=("model_arch in _GLM_DSA_MODEL_ARCHS",))


def _install_backend(m):
    def make(orig):
        def __init__(self, model_runner, *args, **kwargs):
            before = getattr(_BUILDING, "fits", False)
            _BUILDING.fits = mla_layout_fits(model_runner.model_config.hf_text_config)
            try:
                return orig(self, model_runner, *args, **kwargs)
            finally:
                _BUILDING.fits = before

        return __init__

    wrap(m.DeepseekSparseAttnBackend, "__init__", ["self", "model_runner", "..."], make,
         adds=("layout_fits",), needs=("_validate_flashinfer_sparse_mla_backend(",))


def _install_overrides(m):
    def make(orig):
        def _dsa_split_backend_resolution(view):
            import torch

            from sglang.srt.configs.model_config import is_deepseek_dsa

            hf_config = m.model_config_of(view).hf_config
            platform = m.get_platform()
            # what SGLang decides before it looks at the name stays its own
            if (hf_config.architectures[0] not in m._DEEPSEEK_FAMILY_ARCHS or not is_deepseek_dsa(hf_config)
                    or platform.is_npu or platform.is_xpu or platform.is_hip
                    or getattr(hf_config, "learnable_sink", False)):
                return orig(view)
            major, _ = torch.cuda.get_device_capability()
            if major != 12 or view.kv_cache_dtype != "fp8_e4m3":
                return orig(view)
            user_set_prefill = view.dsa_prefill_backend is not None
            user_set_decode = view.dsa_decode_backend is not None
            declared = {}
            if mla_layout_fits(getattr(hf_config, "text_config", None) or hf_config):
                backend = "flashinfer_sparse_mla"
                if not user_set_prefill:
                    declared["dsa_prefill_backend"] = backend
                if not user_set_decode:
                    declared["dsa_decode_backend"] = backend
                m.logger.warning("Set DSA backends for an FP8 KV cache on SM120/SM121 (the latent's layout is "
                                 f"flashinfer_sparse_mla's): prefill={backend}, decode={backend}.")
                return declared
            if not user_set_prefill and not user_set_decode and not view.enable_hisparse:
                declared["dsa_prefill_backend"] = declared["dsa_decode_backend"] = "tilelang"
                m.logger.warning("Set DSA backends for an FP8 KV cache on SM120/SM121 (the latent's layout is not "
                                 "flashinfer_sparse_mla's): prefill=tilelang, decode=tilelang.")
                return declared
            return orig(view)

        return _dsa_split_backend_resolution

    old = m._dsa_split_backend_resolution
    wrap(m, "_dsa_split_backend_resolution", ["view"], make, adds=("mla_layout_fits",),
         needs=('model_arch == "GlmMoeDsaForCausalLM"', "is_glm_sm12_fp8"))
    new = m._dsa_split_backend_resolution
    # the pass is registered when it is defined, and model_hook holds it by name
    passes = getattr(m, "POST_PROCESS_PASSES", None)
    if isinstance(passes, list):
        passes[:] = [new if p is old else p for p in passes]

    def _rebind(mh):
        if getattr(mh, "_dsa_split_backend_resolution", None) is old:
            mh._dsa_split_backend_resolution = new

    after_import("sglang.srt.arg_groups.model_hook", _rebind)


after_import(_KERNEL, _install_kernel)
after_import("sglang.srt.layers.attention.dsa_backend", _install_backend)
after_import("sglang.srt.arg_groups.overrides", _install_overrides)
