"""Where the scheduler meets the lend, the prefill plan and the host-made KV pool:
- after the worker is up, the pool rows made in host memory are placed (kt_kv_vmm.host_place_now);
- a queued request's prompt length reaches the prefill's loan of free KV pages (kt_kv_prefill_lend.note_request);
- the prefill adder is made with the chunk the plan gives for the request about to run (kt_prefill_plan.chunk_for)
  where SGLang passes the launch chunk;
- KT_PREFILL_LEND_PROBE measures the lend before the event loop starts."""
import inspect
import os

from yuzuri._finder import after_import
from yuzuri.hooks._seam import SeamError, wrap

REPLACES = ["srt/managers/scheduler.py"]
CHUNK = "rem_chunk_tokens"


def _planned(base, scheduler):
    """base, made with the planned chunk."""
    sig = inspect.signature(base.__init__)
    if CHUNK not in sig.parameters:
        raise SeamError(f"yuzuri: {base.__module__}.{base.__qualname__} takes no {CHUNK}: the prefill plan has no "
                        f"place to give its chunk")

    class Planned(base):
        def __init__(self, *a, **kw):
            from sglang.srt.layers.moe import kt_prefill_plan

            b = sig.bind(self, *a, **kw)
            b.arguments[CHUNK] = kt_prefill_plan.chunk_for(scheduler, b.arguments.get(CHUNK))
            base.__init__(*b.args, **b.kwargs)

    Planned.__name__, Planned.__qualname__ = base.__name__, base.__qualname__
    return Planned


def _install(m):
    def init_model_worker(orig):
        def init_model_worker(self):
            got = orig(self)
            from sglang.srt.layers.moe import kt_kv_vmm

            if getattr(kt_kv_vmm, "HOST", None) is not None:
                r = self.tp_worker.model_runner
                kt_kv_vmm.host_place_now(r.device, r.token_to_kv_pool_allocator)
            return got

        return init_model_worker

    def add_request(orig):
        def _add_request_to_queue(self, req, *a, **kw):
            from sglang.srt.layers.moe import kt_kv_prefill_lend

            kt_kv_prefill_lend.note_request(req.rid, len(req.origin_input_ids))
            return orig(self, req, *a, **kw)

        return _add_request_to_queue

    def adder_class(orig):
        def get_prefill_adder_class(self):
            base = orig(self)
            made = self.__dict__.setdefault("_yuzuri_adders", {})
            if base not in made:
                made[base] = _planned(base, self)
            return made[base]

        return get_prefill_adder_class

    def event_loop(orig):
        def dispatch_event_loop(scheduler):
            if os.environ.get("KT_PREFILL_LEND_PROBE"):
                from sglang.srt.layers.moe import kt_lend_probe

                kt_lend_probe.run(scheduler)
            return orig(scheduler)

        return dispatch_event_loop

    S = m.Scheduler
    wrap(S, "init_model_worker", ["self"], init_model_worker, adds=("kt_kv_vmm",))
    got = list(inspect.signature(S._add_request_to_queue).parameters)
    wrap(S, "_add_request_to_queue", ["self", "req"] + got[2:], add_request)  # the request first; the rest as it is
    wrap(S, "get_prefill_adder_class", ["self"], adder_class, adds=("kt_prefill_plan",))
    wrap(m, "dispatch_event_loop", ["scheduler"], event_loop)


after_import("sglang.srt.managers.scheduler", _install)
