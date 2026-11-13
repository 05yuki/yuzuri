"""What the safetensors loader does not have to read, and what it reads ahead.

  buffered_multi_thread_safetensors_weights_iterator
      a routed expert whose every physical slot is a KT CPU expert is dropped unread by the MoE weight_loader:
      its key is left out before the read (KT_LOAD_SKIP_CPU_EXPERTS=0 reads it). KT_LOAD_PREFETCH=<threads> preads
      a shard's other tensors in 16 MiB pieces first, so the loaders' mmap faults find the pages in the cache.
      The iterator opens a shard with safetensors.safe_open and lists it with keys(): while one runs, the module's
      safe_open hands it a shard whose keys() leaves those out.
  get_quant_config
      a config dir written for a GGUF checkpoint names the modules that stay unquantized: its quantization_config
      goes to the GGUF config (SGLang passes none).
"""
import concurrent.futures
import json
import os
import re
import struct
import threading

from yuzuri._finder import after_import
from yuzuri.hooks._seam import SeamError, wrap

REPLACES = ["srt/model_loader/weight_utils.py"]
_SKIP = os.environ.get("KT_LOAD_SKIP_CPU_EXPERTS", "1") == "1"
_SKIP_RE = re.compile(os.environ.get("KT_LOAD_SKIP_CPU_EXPERTS_RE", r"(?:^|\.)layers\.(\d+)\..*\.experts\.(\d+)\."))
_state = {}
_PREFETCH_THREADS = int(os.environ.get("KT_LOAD_PREFETCH", "0") or 0)
_PREFETCH_PIECE = int(os.environ.get("KT_LOAD_PREFETCH_PIECE_MB", "16")) << 20
_PREFETCH_SKIP = re.compile(os.environ.get("KT_LOAD_PREFETCH_SKIP", r"\.experts\.|\.engram\.embed\."))
_ACTIVE = [0]
_LOCK = threading.Lock()


def cpu_expert_key(name: str) -> bool:
    """True for a checkpoint key the MoE weight_loader would drop unread: a routed expert whose every physical slot
    is a KT CPU expert (the test FusedMoE's weight_loader makes, asked before the read)."""
    if not _SKIP:
        return False
    mt = _SKIP_RE.search(name)
    if mt is None:
        return False
    st = _state
    if "n" not in st:
        # The loader's threads ask at once. Work in locals and publish "n" last: a thread that finds "n" finds the
        # final count and "meta" with it.
        try:
            from sglang.srt.runtime_context import get_exec

            moe = get_exec().moe
            n = int(moe.kt_num_gpu_experts) if moe.kt_weight_path is not None else -1
            # the experts the hot lend's shadows hold are read too
            try:
                from sglang.srt.layers.moe import kt_kv_hot_lend

                n = kt_kv_hot_lend.load_count(n)
            except ImportError:
                pass
        except Exception:
            n = -1
        try:
            from sglang.srt.eplb.expert_location import get_global_expert_location_metadata

            meta = get_global_expert_location_metadata()
        except Exception:
            meta = None
        st.setdefault("skipped", 0)
        st["meta"] = meta
        st["n"] = n
    n = st["n"]
    if n < 0:
        return False
    layer, eid = int(mt.group(1)), int(mt.group(2))
    meta = st["meta"]
    phys = [eid]
    if meta is not None:
        try:
            phys = meta.logical_to_all_physical(layer, eid)
        except Exception:
            phys = [eid]
    if any(p < n for p in phys):
        return False
    st["skipped"] += 1
    if st["skipped"] % 50000 == 0:
        print(f"[kt-load] {st['skipped']} CPU-expert tensors skipped unread (GPU experts per layer: {n})", flush=True)
    return True


def prefetch_dense_tensors(st_file: str, logger=None) -> None:
    """KT_LOAD_PREFETCH=<threads>: pread this shard's tensors whose names do not match KT_LOAD_PREFETCH_SKIP in
    16 MiB pieces on a thread pool."""
    if _PREFETCH_THREADS <= 0:
        return
    import time

    t0 = time.time()
    with open(st_file, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        meta = json.loads(fh.read(n))
    ranges = []
    for k, v in meta.items():
        if k == "__metadata__" or _PREFETCH_SKIP.search(k):
            continue
        b, e = v["data_offsets"]
        ranges.append((8 + n + b, e - b))
    if not ranges:
        return
    ranges.sort()
    merged = []
    cur_off, cur_end = ranges[0][0], ranges[0][0] + ranges[0][1]
    for off, sz in ranges[1:]:
        if off <= cur_end + (1 << 20):
            cur_end = max(cur_end, off + sz)
        else:
            merged.append((cur_off, cur_end))
            cur_off, cur_end = off, off + sz
    merged.append((cur_off, cur_end))
    pieces = []
    for a, b in merged:
        p = a
        while p < b:
            pieces.append((p, min(_PREFETCH_PIECE, b - p)))
            p += _PREFETCH_PIECE
    local = threading.local()
    fd = os.open(st_file, os.O_RDONLY)

    def read(piece):
        off, sz = piece
        buf = getattr(local, "buf", None)
        if buf is None or len(buf) < sz:
            buf = local.buf = bytearray(max(sz, _PREFETCH_PIECE))
        mv = memoryview(buf)[:sz]
        got = 0
        while got < sz:
            r = os.preadv(fd, [mv[got:]], off + got)
            if r <= 0:
                break
            got += r
        return got

    try:
        with concurrent.futures.ThreadPoolExecutor(_PREFETCH_THREADS) as ex:
            done = sum(ex.map(read, pieces))
    finally:
        os.close(fd)
    if logger is not None:
        logger.info("[dense prefetch] %s: %.2f GiB in %.2f s (%d tensors)",
                    os.path.basename(st_file), done / 2**30, time.time() - t0, len(ranges))


class _Shard:
    """An open shard whose keys() leaves out what is not read."""

    def __init__(self, inner):
        self._inner = inner

    def __enter__(self):
        self._f = self._inner.__enter__()
        return self

    def __exit__(self, *exc):
        return self._inner.__exit__(*exc)

    def keys(self):
        return [k for k in self._f.keys() if not cpu_expert_key(k)]

    def __getattr__(self, name):
        return getattr(self.__dict__.get("_f", self._inner), name)


class _Safetensors:
    """The safetensors package as weight_utils sees it: safe_open as above while a buffered iterator runs."""

    def __init__(self, real, logger):
        self._real, self._logger = real, logger

    def safe_open(self, st_file, *args, **kwargs):
        if _ACTIVE[0] <= 0:
            return self._real.safe_open(st_file, *args, **kwargs)
        prefetch_dense_tensors(st_file, self._logger)
        return _Shard(self._real.safe_open(st_file, *args, **kwargs))

    def __getattr__(self, name):
        return getattr(self._real, name)


def _install(m):
    import safetensors

    def make_iter(orig):
        def buffered_multi_thread_safetensors_weights_iterator(hf_weights_files, max_workers, *args, **kwargs):
            with _LOCK:
                _ACTIVE[0] += 1
            try:
                yield from orig(hf_weights_files, max_workers, *args, **kwargs)
            finally:
                with _LOCK:
                    _ACTIVE[0] -= 1

        return buffered_multi_thread_safetensors_weights_iterator

    def make_quant(orig):
        def get_quant_config(model_config, *args, **kwargs):
            if model_config.quantization == "gguf":
                spec = getattr(model_config.hf_config, "quantization_config", None)
                if isinstance(spec, dict):
                    return m.get_quantization_config("gguf").from_config(spec)
            return orig(model_config, *args, **kwargs)

        return get_quant_config

    if getattr(m, "safetensors", None) is not safetensors:
        raise SeamError("yuzuri: sglang.srt.model_loader.weight_utils no longer opens its shards through the "
                        "safetensors package by that name: the hook that leaves CPU experts unread has nothing to "
                        "attach to in this SGLang")
    wrap(m, "buffered_multi_thread_safetensors_weights_iterator", ["hf_weights_files", "max_workers", "..."],
         make_iter, adds=("_kt_cpu_expert_key", "_prefetch_dense_tensors"),
         needs=('safetensors.safe_open(st_file, framework="pt", device="cpu")', "for k in f.keys()"))
    wrap(m, "get_quant_config", ["model_config", "..."], make_quant, adds=("quant_cls.from_config(spec)", "from_config(model_config.hf_config.quantization_config"),
         needs=("quant_cls.from_config({})", "get_quantization_config(model_config.quantization)"))
    m.safetensors = _Safetensors(safetensors, m.logger)


after_import("sglang.srt.model_loader.weight_utils", _install)
