"""SGLang's KT expert-parallel method (KTEPWrapperMethod: some experts on the GPU, the rest on kt-kernel's CPU
kernels) runs Yuzuri's bodies for the steps Yuzuri's features live in; the class, its constructor and every other
method stay SGLang's. kt_ep_methods holds the bodies.

  create_kt_config_from_server_args   None while the loader builds a draft model (its MoE layers keep their own
                                      checkpoint's experts on the GPU); the layer count from text_config where the
                                      config has none of its own
  create_weights, process_weights_after_loading, submit, apply   see kt_ep_methods
  create_moe_runner                   the runner never works in place: the hot lend's slots and the streamed prefill
                                      read the layer's input after the resident experts
  sync                                zeros on the prefill lend's calibration server with a GGUF pack (it measures
                                      GPU memory only and loads no CPU experts)

Each is attached after checking the method SGLang has is the one these were written against: its parameters, and
the calls its body makes. A tree that already carries one of the mechanisms stops the launch.
"""
from yuzuri._finder import after_import
from yuzuri.hooks._seam import wrap

REPLACES = ["srt/layers/moe/kt_ep_wrapper.py"]


def _y():
    from sglang.srt.layers.moe import kt_ep_methods

    return kt_ep_methods


def _install(m):
    def make_config(orig):
        def create_kt_config_from_server_args(server_args, layer_idx):
            if _y()._BUILDING_DRAFT[0]:
                return None
            config = orig(server_args, layer_idx)
            if config is not None and config.num_layers is None:
                hf_config = m.model_config_of(server_args).hf_config
                config.num_layers = getattr(getattr(hf_config, "text_config", None), "num_hidden_layers", None)
            return config

        return create_kt_config_from_server_args

    def make_create(orig):
        def create_weights(self, layer, num_experts, hidden_size, intermediate_size_per_partition, params_dtype,
                           **extra_weight_attrs):
            return _y().create_weights(self, layer, num_experts, hidden_size, intermediate_size_per_partition,
                                       params_dtype, **extra_weight_attrs)

        return create_weights

    def make_loaded(orig):
        def process_weights_after_loading(self, layer):
            return _y().process_weights_after_loading(self, layer)

        return process_weights_after_loading

    def make_runner(orig):
        def create_moe_runner(self, layer, moe_runner_config):
            moe_runner_config.inplace = False
            return orig(self, layer, moe_runner_config)

        return create_moe_runner

    def make_submit(orig):
        def submit(self, layer, dispatch_output):
            return _y().submit(self, layer, dispatch_output)

        return submit

    def make_sync(orig):
        def sync(self, x):
            if _y()._CALIBRATING:
                import torch

                return torch.zeros_like(x)
            return orig(self, x)

        return sync

    def make_apply(orig):
        def apply(self, layer, dispatch_output):
            return _y().apply(self, layer, dispatch_output)

        return apply

    K = m.KTEPWrapperMethod
    wrap(m, "create_kt_config_from_server_args", ["server_args", "layer_idx"], make_config,
         adds=("_BUILDING_DRAFT", "text_config"), needs=("num_hidden_layers", "KTConfig("))
    wrap(K, "create_weights",
         ["self", "layer", "num_experts", "hidden_size", "intermediate_size_per_partition", "params_dtype",
          "extra_weight_attrs"], make_create, adds=("gpu_experts_mask", "kt_kv_hot_lend", "kt_cutile_nvfp4"),
         needs=("self.gpu_method.create_weights(", "self.wrapper = KTMoEWrapper(",
                "max_deferred_experts_per_token=layer_max_deferred"))
    wrap(K, "process_weights_after_loading", ["self", "layer"], make_loaded,
         adds=("kt_kv_hot_lend", "kt_hot_cache", "_drop_checkpoint_cache"),
         needs=("self.gpu_method.process_weights_after_loading(layer)",
                "self.wrapper.load_weights(physical_to_logical_map_cpu)"))
    wrap(K, "create_moe_runner", ["self", "layer", "moe_runner_config"], make_runner, adds=("inplace",),
         needs=("self.moe_runner_config = moe_runner_config",))
    wrap(K, "submit", ["self", "layer", "dispatch_output"], make_submit, adds=("pack_to_host", "_CALIBRATING"),
         needs=("self.wrapper.submit_forward(",))
    wrap(K, "sync", ["self", "x"], make_sync, adds=("_CALIBRATING",), needs=("torch.zeros_like(x)",))
    wrap(K, "apply", ["self", "layer", "dispatch_output"], make_apply,
         adds=("kt_kv_hot_lend", "stream_prefill_for", "kt_hot_cache"),
         needs=("self.submit(layer, dispatch_output)", "mask_cpu_expert_ids(topk_ids, self.num_gpu_experts)",
                "self.gpu_method.apply(layer, masked_dispatch_output)", "self.sync(x)"))


after_import("sglang.srt.layers.moe.kt_ep_wrapper", _install)
