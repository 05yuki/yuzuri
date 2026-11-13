"""How many threads a helper pool takes, from the machine instead of a fixed number.

The pools that copy, touch or convert memory while kt-kernel's CPU experts are not computing (loading, a prefill
window's start and end) take this rank's share of the physical cores it may run on: the cores in its CPU affinity,
divided by the tensor-parallel ranks that share the host when the rank is not pinned to its own CPUs. An
environment variable still sets a pool's count.
"""
import functools
import os


@functools.lru_cache(maxsize=None)
def _core_of(cpu: int):
    """(package, core) of a logical CPU, from sysfs (None when the topology is not there)."""
    base = f"/sys/devices/system/cpu/cpu{cpu}/topology"
    try:
        with open(f"{base}/physical_package_id") as f:
            pkg = int(f.read())
        with open(f"{base}/core_id") as f:
            core = int(f.read())
    except OSError:
        return None
    return pkg, core


def _ranks_on_host() -> int:
    try:
        from sglang.srt.runtime_context import get_server_args

        args = get_server_args()
        tp = int(getattr(args, "tp_size", 1) or 1)
        nodes = int(getattr(args, "nnodes", 1) or 1)
        return max(1, tp // max(1, nodes))
    except Exception:
        return 1


def cores_here() -> int:
    """This rank's share of the physical cores it may run on."""
    try:
        cpus = sorted(os.sched_getaffinity(0))
    except AttributeError:
        cpus = list(range(os.cpu_count() or 1))
    cores = {_core_of(c) or ("cpu", c) for c in cpus}
    total = len({_core_of(c) for c in range(os.cpu_count() or 1)} - {None}) or (os.cpu_count() or 1)
    n = len(cores)
    # an unpinned rank sees every core of the host, which the other ranks on it see too
    if n >= total:
        n = n // _ranks_on_host()
    return max(1, n)


def threads(env: str) -> int:
    """The thread count for a helper pool: env when it is set, else cores_here()."""
    v = os.environ.get(env, "")
    return max(1, int(v)) if v.strip() else cores_here()
