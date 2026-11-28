"""Where the model loader meets Yuzuri:
- the model is built with the KT wrapper knowing whether it is a draft (kt_ep_methods.building), and the prefill
  lend binds the target's layers before any weight loads (kt_lend.bind_model);
- KT_GGUF_DENSE: the primary weights come from the GGUF under the checkpoint's own names (kt_gguf_dense); the
  secondary sources stay SGLang's;
- the experts of an MoE layer whose GPU slots take their weights from somewhere else (the method says so:
  takes_no_checkpoint_experts) never reach the model's load_weights;
- after each module's weight processing the lend takes back what a repack left outside its span, and after the
  load it closes (kt_lend.after_module, after_load), before the KV pool is sized; SGLANG_EMBED_ON_HOST=1 moves the
  token embedding to pinned host memory before that."""
import logging
import os
import re
import time

from yuzuri._finder import after_import
from yuzuri.hooks._seam import SeamError, wrap

REPLACES = ["srt/model_loader/loader.py"]
logger = logging.getLogger("sglang.srt.model_loader.loader")


def without_checkpoint_experts(model, weights):
    """weights minus the experts of every such layer. A layer is matched by its module path from the component
    before its first index on (layers.3.mlp.experts.), which is what a checkpoint keeps of the path under any
    prefix."""
    tails = []
    for name, module in model.named_modules():
        gpu_method = getattr(getattr(module, "quant_method", None), "gpu_method", None)
        if getattr(gpu_method, "takes_no_checkpoint_experts", False):
            parts = name.split(".")
            first = next((i for i, c in enumerate(parts) if c.isdigit()), 0)
            tails.append(re.escape(".".join(parts[max(0, first - 1):]) + "."))
    if not tails:
        return weights
    under = re.compile(r"(?:^|\.)(?:" + "|".join(tails) + ")")

    def rest():
        dropped = 0
        for name, loaded_weight in weights:
            if under.search(name):
                dropped += 1
                continue
            yield name, loaded_weight
        logger.info(f"{dropped} expert tensors of the checkpoint left unread into the model: "
                    f"{len(tails)} MoE layers fill their GPU slots from the experts' own file")

    return rest()


def _install(m):
    L = getattr(m, "DefaultModelLoader", None)
    if L is None:
        raise SeamError(f"yuzuri: {m.__name__} has no DefaultModelLoader")

    def initialize(orig):
        def _initialize_model(model_config, load_config, quant_config=None):
            from sglang.srt.layers.moe import kt_ep_methods, kt_lend

            is_draft = bool(getattr(model_config, "is_draft_model", False))
            with kt_ep_methods.building(is_draft):
                model = orig(model_config, load_config, quant_config)
            if not is_draft:
                kt_lend.bind_model(model)
            return model

        return _initialize_model

    def all_weights(orig):
        def _get_all_weights(self, model_config, model):
            from sglang.srt.layers.moe import kt_gguf_dense

            gguf = kt_gguf_dense.gguf_dir()
            if gguf and not getattr(model_config, "is_draft_model", False):
                # the next source read is the primary one (_get_weights_iterator below)
                self._yuzuri_gguf_dense = (gguf, model_config, model)
            try:
                yield from orig(self, model_config, model)
            finally:
                self._yuzuri_gguf_dense = None

        return _get_all_weights

    def weights_iterator(orig):
        def _get_weights_iterator(self, source, *args, **kwargs):
            from sglang.srt.layers.moe import kt_gguf_dense

            primary, self._yuzuri_gguf_dense = getattr(self, "_yuzuri_gguf_dense", None), None
            if primary is None:
                return orig(self, source, *args, **kwargs)
            gguf, model_config, model = primary
            logger.info("loading the dense weights from GGUF %s", gguf)
            # the loader's own start mark, where the SGLang served has one (gone by a9a27e92)
            if getattr(self, "counter_before_loading_weights", None) == 0.0:
                self.counter_before_loading_weights = time.perf_counter()
            return kt_gguf_dense.weights(gguf, model_config, model)

        return _get_weights_iterator

    def monokernel(orig):
        def monokernel_weights_iterator(*, model, weights):
            return orig(model=model, weights=without_checkpoint_experts(model, weights))

        return monokernel_weights_iterator

    def with_quant_method(orig):
        def _modules_with_quant_method(model):
            from sglang.srt.layers.moe import kt_lend

            for module, quant_method in orig(model):
                yield module, quant_method
                kt_lend.after_module(module)  # the loop's body for this module is over

        return _modules_with_quant_method

    def postprocess(orig):
        def postprocess_weights(model, target_device):
            from sglang.srt.layers.moe import kt_lend

            got = orig(model, target_device)
            if os.environ.get("SGLANG_EMBED_ON_HOST", "0") == "1":
                from sglang.srt.layers.host_embedding import move_embedding_to_host
                from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding

                for name, mod in model.named_modules():
                    if name.endswith("embed_tokens") and type(mod) is VocabParallelEmbedding:
                        nb = move_embedding_to_host(mod)
                        logger.info("token embedding %s on host: %.1f MiB moved to pinned host memory", name, nb / 2**20)
            kt_lend.after_load()
            return got

        return postprocess_weights

    wrap(m, "_initialize_model", ["model_config", "load_config", "quant_config"], initialize,
         adds=("kt_ep_wrapper", "kt_ep_methods", "kt_lend"))
    wrap(L, "_get_all_weights", ["self", "model_config", "model"], all_weights)
    wrap(L, "_get_weights_iterator", ["self", "source", "..."], weights_iterator)
    wrap(m, "monokernel_weights_iterator", ["model", "weights"], monokernel)
    wrap(m, "_modules_with_quant_method", ["model"], with_quant_method)
    wrap(L, "postprocess_weights", ["model", "target_device"], postprocess, adds=("kt_lend",))


after_import("sglang.srt.model_loader.loader", _install)
