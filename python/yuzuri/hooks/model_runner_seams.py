"""Where the model runner meets Yuzuri:
- load_model: KT_LAUNCH_TIMING before it; after it, what leaves the card for any model that has it (the multimodal
  encoder on demand, the token embedding on the host, the RoPE tables mapped as the context grows) and the shared
  experts' late start (kt_handoff_pack);
- every forward: the streamed prefill's weights out / scratch in (kt_lend), the stream threshold's clock, the RoPE
  rows this batch reaches, the hot cache's turn."""
import logging
import os

from yuzuri._finder import after_import
from yuzuri.hooks._seam import wrap

REPLACES = ["srt/model_executor/model_runner.py"]
logger = logging.getLogger("sglang.srt.model_executor.model_runner")
MIB = 2**20


def _install(m):
    def load_model(orig):
        def load_model(self):
            if os.environ.get("KT_LAUNCH_TIMING") == "1":
                from sglang.srt.layers.moe import kt_launch_timing

                kt_launch_timing.install(self)
            got = orig(self)
            from sglang.srt.layers import mm_on_demand
            from sglang.srt.layers.moe import kt_kv_vmm
            from sglang.srt.layers.moe.kt_handoff_pack import bind_shared_streams

            b = mm_on_demand.install(self.model)
            if b:
                logger.info(f"multimodal encoder on demand: {b / MIB:.1f} MiB in pinned host memory")
            b = bind_shared_streams(self.model)
            if b:
                logger.info(f"shared experts start after the CPU request in {b} MoE blocks")
            if os.environ.get("SGLANG_EMBED_ON_HOST", "0") == "1":
                from sglang.srt.layers.host_embedding import move_all_to_host

                for n, b in move_all_to_host(self.model):
                    logger.info(f"token embedding {n} on host: {b / MIB:.1f} MiB moved to pinned host memory")
            if os.environ.get("SGLANG_ROPE_LAZY", "0") == "1" and kt_kv_vmm.ENABLED:
                from sglang.srt.layers import lazy_rope_table

                b = lazy_rope_table.install(self.model)
                if b:
                    logger.info(f"RoPE tables mapped as the context grows: {b / MIB:.1f} MiB no longer resident")
            return got

        return load_model

    def forward_raw(orig):
        def _forward_raw(self, forward_batch, pp_proxy_tensors):
            from sglang.srt.layers import lazy_rope_table
            from sglang.srt.layers.moe import kt_hot_cache, kt_lend, kt_stream_threshold

            kt_stream_threshold.begin(forward_batch)
            kt_lend.before_forward(forward_batch)
            lazy_rope_table.ensure(forward_batch)
            kt_hot_cache.before_forward(forward_batch)
            got = orig(self, forward_batch, pp_proxy_tensors)
            kt_stream_threshold.end()
            return got

        return _forward_raw

    wrap(m.ModelRunner, "load_model", ["self"], load_model, adds=("mm_on_demand", "lazy_rope_table", "kt_handoff_pack"))
    wrap(m.ModelRunner, "_forward_raw", ["self", "forward_batch", "pp_proxy_tensors"], forward_raw,
         adds=("kt_lend", "kt_hot_cache", "lazy_rope_table"))


after_import("sglang.srt.model_executor.model_runner", _install)
