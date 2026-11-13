"""SWA eviction between a prefill's chunks under the overlap scheduler: SGLang keeps the launch chunk
(chunked_prefill_size) behind the prefix; the chunk in flight is the one the plan gave this request
(kt_prefill_plan.chunk_in_flight). With the launch chunk a 35K request held every chunk's SWA until it ended."""
import threading

from yuzuri._finder import after_import
from yuzuri.hooks._seam import wrap

REPLACES = ["srt/managers/schedule_batch.py"]

_IN = threading.local()


def _install(m):
    def make_outer(orig):
        def maybe_evict_swa(self):
            # the one call of _evict_swa that subtracts the launch chunk: an extend batch, overlap on, no prefix sharing
            _IN.on = bool(self.tree_cache.supports_swa() and self.forward_mode.is_extend() and self.enable_overlap
                          and not self.tree_cache.supports_prefix_sharing())
            try:
                return orig(self)
            finally:
                _IN.on = False

        return maybe_evict_swa

    def make_inner(orig):
        def _evict_swa(self, req, pre_len, *args, **kwargs):
            if getattr(_IN, "on", False):
                from sglang.srt.layers.moe import kt_prefill_plan

                launch = m.get_schedule().chunked_prefill_size
                flight = kt_prefill_plan.chunk_in_flight(req, launch)
                pre_len = pre_len + (launch if launch > 0 else 0) - (flight if flight > 0 else 0)
            return orig(self, req, pre_len, *args, **kwargs)

        return _evict_swa

    wrap(m.ScheduleBatch, "maybe_evict_swa", ["self"], make_outer, adds=("kt_prefill_plan",),
         needs=("get_schedule().chunked_prefill_size", "self._evict_swa(req, pre_len)"))
    wrap(m.ScheduleBatch, "_evict_swa", ["self", "req", "pre_len", "..."], make_inner)


after_import("sglang.srt.managers.schedule_batch", _install)
