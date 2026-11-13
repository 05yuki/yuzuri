"""FusedMoE's part in three of the package's expert methods, on the lines of its own methods where they belong:

  __init__                  a layer whose CPU experts are a GGUF (KT method LLAMAFILE) holds its hot slots as the same
                            raw GGUF blocks (kt_gguf_hot; KT_GGUF_HOT=0 leaves the layer's own method), and a draft
                            model whose experts ship as a GGUF of their own (KT_MTP_GGUF) reads them from it
  weight_loader,            a KT layer's experts past the GPU's count never reach the layer's own tensors, with
  _weight_loader_impl       expert-location metadata or without it: they load into the hot lend's shadow
                            (kt_kv_hot_lend) where it has one for the layer, and are dropped where it has none.
                            However SGLang resolves an expert's destinations (the two methods' signatures are all
                            the hook relies on), the choice is made where every destination arrives
  _weight_loader_impl,      a method that wraps another to run it on other kernels (kt_cutile_nvfp4) loads through
  weight_loader_fused       the one it wraps
"""
import threading

from yuzuri._finder import after_import
from yuzuri.hooks._seam import rewrite, wrap

REPLACES = ["srt/layers/moe/fused_moe_triton/layer.py"]


def _d(text: str) -> str:
    """The lines as rewrite() sees a method's: the class body's indentation taken off."""
    return "".join(line[4:] if line.strip() else line for line in text.splitlines(True))


_KT = _d("        kt_config = create_kt_config_from_server_args(server_args, layer_id)\n")
_DRAFT = _d('''        import os

        from sglang.srt.layers.moe import kt_ep_methods as _kt_ep

        if getattr(_kt_ep, "_BUILDING_DRAFT", [False])[0] and os.environ.get("KT_MTP_GGUF"):
            # a draft model whose experts ship as a GGUF of their own (kt_gguf_draft.py): the raw blocks on the
            # GPU, whatever quantization the draft's config names for its other layers
            from sglang.srt.layers.moe.kt_gguf_draft import KTGGUFDraftExpertsMethod

            self.quant_method = KTGGUFDraftExpertsMethod(os.environ["KT_MTP_GGUF"], self)
''')
_GPU = _d("            if self.quant_method is not None:\n                gpu_method = self.quant_method\n")
_GPU_GGUF = _d('''            if kt_config.method == "LLAMAFILE" and os.environ.get("KT_GGUF_HOT", "1") == "1":
                # the experts are a GGUF, wherever the dense layers come from: the hot slots hold the same raw
                # GGUF blocks the CPU experts read (kt_gguf_hot.py), never another checkpoint's copy of them
                from sglang.srt.layers.moe.kt_gguf_hot import KTGGUFHotExpertsMethod

                gpu_method = KTGGUFHotExpertsMethod(kt_config, self)
            elif self.quant_method is not None:
                gpu_method = self.quant_method
''')
_IMPL = _d('        if method.__class__.__name__ == "KTEPWrapperMethod":\n'
           "            method = method.gpu_method\n")
_FUSED = _d("        if isinstance(method, KTEPWrapperMethod):\n"
            "            method = method.gpu_method\n")
_INNER = _d('        if method.__class__.__name__ == "CutileNvfp4Method":  # kt_cutile_nvfp4: the method it wraps loads\n'
            "            method = method.inner\n")

_LOADING = threading.local()
_LOCK = threading.RLock()


def _install(m):
    F = m.FusedMoE
    rewrite(F, "__init__", [(_KT, _DRAFT + _KT), (_GPU, _GPU_GGUF)], adds=("KTGGUFHotExpertsMethod", "KT_MTP_GGUF"))
    rewrite(F, "_weight_loader_impl", [(_IMPL, _IMPL + _INNER)], adds=("CutileNvfp4Method",))
    rewrite(F, "weight_loader_fused", [(_FUSED, _FUSED + _INNER)], adds=("CutileNvfp4Method",))

    def make_loader(orig):
        def weight_loader(self, param, loaded_weight, weight_name, shard_id, expert_id):
            qm = self.quant_method
            if type(qm).__name__ != "KTEPWrapperMethod" or expert_id is None:
                return orig(self, param, loaded_weight, weight_name, shard_id, expert_id)
            # the count is read once a layer, and one loader at a time runs in a layer: SGLang may load a model's
            # tensors from several threads, and the count below is taken away for the length of a call
            with _LOCK:
                state = qm.__dict__.get("_yuzuri_load")
                if state is None:
                    state = qm.__dict__["_yuzuri_load"] = (threading.RLock(), int(getattr(qm, "num_gpu_experts", -1)))
            lock, n = state
            if n == -1:
                return orig(self, param, loaded_weight, weight_name, shard_id, expert_id)
            # SGLang drops the experts past the GPU's count on some of its paths and not on others: for this call
            # it drops none, and _weight_loader_impl below places every destination it resolves
            with lock:
                before = getattr(_LOADING, "kt", None)
                _LOADING.kt = (qm, n)
                qm.num_gpu_experts = -1
                try:
                    return orig(self, param, loaded_weight, weight_name, shard_id, expert_id)
                finally:
                    qm.num_gpu_experts = n
                    _LOADING.kt = before

        return weight_loader

    def make_impl(orig):
        def _weight_loader_impl(self, param, loaded_weight, weight_name, shard_id, expert_id):
            kt = getattr(_LOADING, "kt", None)
            if kt is None or kt[0] is not self.quant_method or expert_id is None or expert_id < kt[1]:
                return orig(self, param=param, loaded_weight=loaded_weight, weight_name=weight_name,
                            shard_id=shard_id, expert_id=expert_id)
            from sglang.srt.layers.moe import kt_kv_hot_lend

            qm, n = kt
            qm.num_gpu_experts = n
            try:
                hot = kt_kv_hot_lend.shadow_param(qm, self, param, expert_id)
            finally:
                qm.num_gpu_experts = -1
            if hot is not None:
                orig(self, param=hot[0], loaded_weight=loaded_weight, weight_name=weight_name,
                     shard_id=shard_id, expert_id=hot[1])

        return _weight_loader_impl

    args = ["self", "param", "loaded_weight", "weight_name", "shard_id", "expert_id"]
    wrap(F, "weight_loader", args, make_loader, adds=("kt_kv_hot_lend",), needs=("self._weight_loader_impl(",))
    wrap(F, "_weight_loader_impl", args, make_impl, adds=("kt_kv_hot_lend",))


after_import("sglang.srt.layers.moe.fused_moe_triton.layer", _install)
