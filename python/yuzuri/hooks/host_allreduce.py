"""Two cards without peer access: SGLANG_HOST_STAGED_ALLREDUCE=1 stages the large all-reduces (prefill) through
pinned host memory with the copy engines where NCCL takes its SHM transport, and SGLANG_LL_ALLREDUCE=1 runs the
decode-sized ones as one kernel through host memory in NCCL's LL form."""
import os

from yuzuri._finder import after_import
from yuzuri.hooks._seam import SeamError, wrap

REPLACES = ["srt/distributed/parallel_state.py"]
NEEDS = ("cpu_group", "rank_in_group", "world_size", "device")


def _install(m):
    import inspect

    G = m.GroupCoordinator

    def init(orig):
        def __init__(self, *a, **kw):
            orig(self, *a, **kw)
            self._yuzuri_host_ar = False
            if hasattr(self, "host_staged_comm") or hasattr(self, "ll_comm"):
                return  # this SGLang already makes them (a tree patched with the same all-reduces): its own stay
            self.host_staged_comm = self.ll_comm = None
            self._yuzuri_host_ar = True
            staged = os.environ.get("SGLANG_HOST_STAGED_ALLREDUCE") == "1"
            ll = os.environ.get("SGLANG_LL_ALLREDUCE") == "1"
            if not (staged or ll):
                return
            gone = [n for n in NEEDS if not hasattr(self, n)]
            if gone:
                raise SeamError(f"yuzuri: GroupCoordinator has no {gone}: the host all-reduces cannot be made")
            if self.world_size != 2 or not m.is_cuda_alike():
                return
            if staged:
                from sglang.srt.distributed.device_communicators.host_staged_allreduce import HostStagedAllReduce

                self.host_staged_comm = HostStagedAllReduce(
                    self.cpu_group, self.rank_in_group, self.world_size, self.device)
            if ll:
                from sglang.srt.distributed.device_communicators.ll_allreduce import LLAllReduce

                self.ll_comm = LLAllReduce(self.cpu_group, self.rank_in_group, self.world_size, self.device)

        return __init__

    def all_reduce(orig):
        def all_reduce(self, input_):
            if self._yuzuri_host_ar and not input_.is_cpu:  # a comm exists at two ranks only
                if self.host_staged_comm is not None and self.host_staged_comm.should_use(input_):
                    return self.host_staged_comm.all_reduce(input_)
                if self.ll_comm is not None and self.ll_comm.should_use(input_):
                    return self.ll_comm.all_reduce(input_)
            return orig(self, input_)

        return all_reduce

    wrap(G, "__init__", list(inspect.signature(G.__init__).parameters), init)
    wrap(G, "all_reduce", ["self", "input_"], all_reduce)


after_import("sglang.srt.distributed.parallel_state", _install)
