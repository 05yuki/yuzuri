"""KT serves the routed experts on the CPU, which holds no fused shared expert: shared experts fusion off whenever
--kt-weight-path is set (kt-shared-experts-unfused.py)."""
from yuzuri._finder import after_import
from yuzuri.hooks._seam import wrap

REPLACES = ["srt/layers/moe/utils.py"]


def _install(m):
    def make(orig):
        def resolve_shared_experts_config(model_class, hf_config, quant_config):
            moe = m.get_exec().moe
            if moe.disable_shared_experts_fusion or getattr(moe, "kt_weight_path", None) is None:
                return orig(model_class, hf_config, quant_config)
            m.log_info_on_rank0(m.logger, "KT serves the routed experts on the CPU, which holds no fused shared "
                                "expert. Shared experts fusion optimization is disabled.")
            return m.SharedExpertsConfig(fusion_disabled=True, num_fused_shared_experts=0, use_waterfill=False)

        return resolve_shared_experts_config

    wrap(m, "resolve_shared_experts_config", ["model_class", "hf_config", "quant_config"], make)


after_import("sglang.srt.layers.moe.utils", _install)
