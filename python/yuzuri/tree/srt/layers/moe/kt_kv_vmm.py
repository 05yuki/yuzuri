"""The KV pool on remappable memory (KT_KV_VMM=1; libkv_vmm.so, kv_vmm.c).

Every tensor made inside region() is one virtual range backed by physical
chunks that can be unmapped (release) and mapped again, zeroed (restore),
while the range, the tensors on it and the CUDA graphs that read it stay put.
sglang makes the KV pool inside memory_saver_adapter.region("kv_cache")
(hooks/kv_vmm_region.py routes that region here), so the free pages of the pool
can lend their physical memory to a streamed prefill and take it back before
decode (kt_lend's prefill window, M2).

KT_KV_VMM_LIB overrides the library path (default: libkv_vmm.so next to this
module, which setup.sh --package builds).
"""
import contextlib
import ctypes
import os

import torch

_SAID = set()


def _say_once(where, exc):
    """One line the first time a guarded step fails for a reason other than its module being off, so an SGLang that
    changed under it shows in the log."""
    if isinstance(exc, ImportError) or where in _SAID:
        return
    _SAID.add(where)
    print(f"[yuzuri] {where}: {type(exc).__name__}: {exc} (said once)", flush=True)


def _torch_ok() -> bool:
    """torch 2.13 runs the KV pool on this allocator through a MemPool; torch 2.9.1
    (GLM-5.3's venv) segfaults at the first allocation in the pluggable-allocator
    MemPool, CUDA initialised or not (10-15), so older torch takes the direct
    path: region() makes the tensors itself (KT_KV_VMM_DIRECT=1 forces it)."""
    try:
        major, minor = (int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    except ValueError:
        return False
    return (major, minor) >= (2, 13)


ENABLED = os.environ.get("KT_KV_VMM", "0") == "1"
DIRECT = os.environ.get("KT_KV_VMM_DIRECT", "0" if _torch_ok() else "1") == "1"


def _lib_path() -> str:
    """KT_KV_VMM_LIB, else libkv_vmm.so next to this module (setup.sh --package builds it there)."""
    env = os.environ.get("KT_KV_VMM_LIB")
    if env:
        return env
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libkv_vmm.so")
    return here


_LIB_PATH = _lib_path()

_lib = None
_pool = None


def _load():
    global _lib, _pool
    if _lib is not None:
        return _lib
    # 8 MiB chunks: a quarter of the map / unmap calls of the library's 2 MiB
    # (a window's loan took ~0.2 s each way at 2 MiB, 10-15)
    os.environ.setdefault("KV_VMM_CHUNK_MB", "8")
    _lib = ctypes.CDLL(_LIB_PATH)
    _lib.kv_vmm_release.restype = ctypes.c_longlong
    _lib.kv_vmm_release.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
    _lib.kv_vmm_restore.restype = ctypes.c_longlong
    _lib.kv_vmm_restore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
    _lib.kv_vmm_released_bytes.restype = ctypes.c_size_t
    _lib.kv_vmm_released_bytes.argtypes = [ctypes.c_int]
    _lib.kv_vmm_chunk_at.restype = ctypes.c_size_t
    _lib.kv_vmm_chunk_at.argtypes = [ctypes.c_void_p]
    _lib.kv_vmm_chunk_bytes.restype = ctypes.c_size_t
    _lib.kv_vmm_chunk_bytes.argtypes = [ctypes.c_int]
    _lib.kv_vmm_malloc.restype = ctypes.c_void_p
    _lib.kv_vmm_malloc.argtypes = [ctypes.c_ssize_t, ctypes.c_int, ctypes.c_void_p]
    _lib.kv_vmm_malloc_lazy.restype = ctypes.c_void_p
    _lib.kv_vmm_malloc_lazy.argtypes = [ctypes.c_ssize_t, ctypes.c_int]
    if hasattr(_lib, "kv_vmm_malloc_shadow"):
        _lib.kv_vmm_malloc_shadow.restype = ctypes.c_void_p
        _lib.kv_vmm_malloc_shadow.argtypes = [ctypes.c_ssize_t, ctypes.c_int]
    if hasattr(_lib, "kv_vmm_malloc_split"):
        _lib.kv_vmm_malloc_split.restype = ctypes.c_void_p
        _lib.kv_vmm_malloc_split.argtypes = [ctypes.c_ssize_t, ctypes.c_int, ctypes.c_size_t, ctypes.c_int]
        _lib.kv_vmm_device_bytes.restype = ctypes.c_size_t
        _lib.kv_vmm_device_bytes.argtypes = [ctypes.c_void_p]
    if hasattr(_lib, "kv_vmm_park"):  # a library built before the park has neither
        _lib.kv_vmm_park.restype = ctypes.c_longlong
        _lib.kv_vmm_park.argtypes = [ctypes.c_int]
        _lib.kv_vmm_unpark.restype = ctypes.c_longlong
        _lib.kv_vmm_unpark.argtypes = [ctypes.c_int]
    _lib.kv_vmm_free.restype = None
    _lib.kv_vmm_free.argtypes = [ctypes.c_void_p, ctypes.c_ssize_t, ctypes.c_int, ctypes.c_void_p]
    if not DIRECT:
        alloc = torch.cuda.memory.CUDAPluggableAllocator(_LIB_PATH, "kv_vmm_malloc", "kv_vmm_free")
        _pool = torch.cuda.MemPool(alloc.allocator())
    return _lib


class _Block:
    """One kv_vmm allocation, owned by the tensor made over it (torch keeps the
    object of a __cuda_array_interface__ alive as long as the tensor)."""

    def __init__(self, nbytes: int, device: int, lazy: bool = False, shadow: bool = False, split=None):
        self.nbytes, self.device = nbytes, device
        if split is not None:
            self.ptr = _lib.kv_vmm_malloc_split(nbytes, device, split[0], split[1])
        elif shadow:
            self.ptr = _lib.kv_vmm_malloc_shadow(nbytes, device)
        else:
            self.ptr = _lib.kv_vmm_malloc_lazy(nbytes, device) if lazy else _lib.kv_vmm_malloc(nbytes, device, None)
        if not self.ptr:
            raise torch.OutOfMemoryError(f"kv_vmm: {nbytes} bytes on cuda:{device}")
        self.__cuda_array_interface__ = {"shape": (nbytes,), "typestr": "|u1", "data": (self.ptr, False),
                                         "strides": None, "version": 3, "stream": None}

    def __del__(self):
        if getattr(self, "ptr", None) and _lib is not None:
            _lib.kv_vmm_free(self.ptr, self.nbytes, self.device, None)
            self.ptr = None


# LAZY(shape, nbytes) -> bytes to map from the start, or None: a tensor made in
# region() that it claims is a range with only those bytes and its last chunk
# mapped (zeroed); the rest is mapped by restore when used (kt_kv_prefill_lend's
# SWA pool widened for the prefill plan). LAZY_MADE lists them.
LAZY = None
LAZY_MADE = []


def _lazy_tensor(shape, dtype, index: int, lead: int) -> torch.Tensor:
    n = 1
    for d in shape:
        n *= int(d)
    size_b = n * dtype.itemsize
    with torch.cuda.device(index):
        t = torch.as_tensor(_Block(max(1, size_b), index, lazy=True), device=f"cuda:{index}")
    c = int(_lib.kv_vmm_chunk_at(ctypes.c_void_p(t.data_ptr())))
    # a lead is the first bytes, or (offset, bytes) spans (a fused K/V buffer's two halves)
    ok = True
    for off, n in ([(0, lead)] if isinstance(lead, int) else lead):
        n = min(size_b - off, -(-max(0, n) // c) * c)
        ok = ok and (n <= 0 or restore(t, off, n) >= 0)
    # the last chunks too: a pool may write its last row at setup (DeepSeek V4's
    # compress-state sentinel)
    tail = max(0, size_b - c) // c * c
    ok = ok and restore(t, tail, 2 * c) >= 0
    if not ok:
        raise torch.OutOfMemoryError(f"kv_vmm: mapping the used part of a {size_b}-byte pool on cuda:{index}")
    torch.cuda.synchronize(index)
    t = t[:size_b].view(dtype).view(shape)
    LAZY_MADE.append(t)
    return t


# Host KV for any pool. HOST: the pool's row counts (tokens, pages) set by the pool
# configurator when the pool covers more than the card holds; a tensor made in region() whose leading dimension
# is one of them is made in host memory and listed in HOST_MADE, and host_place() moves the first rows of each
# to the card once the server is up.
HOST = None
HOST_MADE = []


def _host_numa(index: int) -> int:
    try:
        p = torch.cuda.get_device_properties(index)
        path = f"/sys/bus/pci/devices/{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0/numa_node"
        return max(0, int(open(path).read().strip()))
    except Exception:
        return 0


def _host_tensor(shape, dtype, index: int) -> torch.Tensor:
    _place_sig()
    t = split_tensor(shape, dtype, index, 0, _host_numa(index))
    HOST_MADE.append(t)
    return t


def host_place(device: int, free_bytes: int) -> dict:
    """place in tokens. Move the first rows of the host-made pool tensors to the card,
    within free_bytes. Tensors grouped by bytes a token (a row is one token or one page of the pool), fewest
    first: a group that fits goes whole; the first that does not, and every later one, share one token count.
    Returns {"rows": the shared token count or None, "whole": bytes placed whole, "split": bytes placed}."""
    lib = _load()
    ts = [t for t in HOST_MADE if t.device.index == device]
    if not ts:
        return {"rows": None, "whole": 0, "split": 0}
    full = int(HOST["full"]) if HOST else 0

    def tokens_a_row(t):
        return max(1, round(full / max(1, t.shape[0] - 1))) if full else 1

    def bytes_a_token(t):
        return (t.nbytes // max(1, t.shape[0])) / tokens_a_row(t)

    groups = {}
    for t in ts:
        groups.setdefault(round(bytes_a_token(t), 3), []).append(t)
    left = free_bytes
    whole = 0
    rows = None
    order = sorted(groups)
    for k, bt in enumerate(order):
        size = sum(t.nbytes for t in groups[bt])
        if size <= left:
            for t in groups[bt]:
                if lib.kv_vmm_place(ctypes.c_void_p(t.data_ptr()), ctypes.c_size_t(t.nbytes)) < 0:
                    raise torch.OutOfMemoryError(f"kv_vmm: placing a host KV tensor on cuda:{device}")
            left -= size
            whole += size
            continue
        rest = [t for r in order[k:] for t in groups[r]]
        tokens = int(left // max(1e-9, sum(bytes_a_token(t) for t in rest)))
        rows = tokens
        for t in rest:
            n = (tokens // tokens_a_row(t)) * (t.nbytes // max(1, t.shape[0]))
            if lib.kv_vmm_place(ctypes.c_void_p(t.data_ptr()), ctypes.c_size_t(n)) < 0:
                raise torch.OutOfMemoryError(f"kv_vmm: placing a host KV tensor on cuda:{device}")
            left -= n
        break
    torch.cuda.synchronize(device)
    return {"rows": rows, "whole": whole, "split": free_bytes - left}


def _free_debug(pages: int) -> None:
    """KT_HOST_KV_DEBUG=1: who frees 32 pages or more (a cached prompt's node went free inside another's prefill)."""
    if pages < 32 or os.environ.get("KT_HOST_KV_DEBUG") != "1":
        return
    import traceback

    who = " < ".join(f"{f.name}:{f.lineno}" for f in reversed(traceback.extract_stack(limit=16)[:-2]))
    print(f"[kt-host-kv] {pages} pages freed by {who}", flush=True)


def keep_lowest_first(allocator) -> int:
    """Freed pages go back in order, so the lowest free pages (the card's part of a host KV pool) are handed out
    first. Every allocator reachable from `allocator` (it and its *_allocator attributes): a paged one through
    _release_page_ids, a token one through free. Returns how many were wrapped."""
    seen, n = set(), 0
    todo = [allocator]
    while todo:
        a = todo.pop()
        if a is None or id(a) in seen:
            continue
        seen.add(id(a))
        todo += [v for k, v in vars(a).items() if k.endswith("allocator") and v is not a]
        if getattr(a, "_kt_lowest_first", False) or getattr(a, "free_pages", None) is None:
            continue
        if hasattr(a, "_release_page_ids"):
            orig = a._release_page_ids

            def release(*page_ids, a=a, orig=orig):
                _free_debug(sum(len(x) for x in page_ids))
                if getattr(a, "need_sort", False):
                    return orig(*page_ids)
                a.free_pages = torch.sort(torch.cat((*page_ids, a.free_pages))).values

            a._release_page_ids = release
        elif hasattr(a, "free"):
            orig = a.free

            def free(free_index, a=a, orig=orig):
                _free_debug(len(free_index) // max(1, getattr(a, "page_size", 1)))
                r = orig(free_index)
                if not getattr(a, "need_sort", False) and len(a.free_pages):
                    a.free_pages = torch.sort(a.free_pages).values
                return r

            a.free = free
        else:
            continue
        a._kt_lowest_first = True
        n += 1
    return n


def _pool_allocator():
    """rows in use v3. The allocator the KV lend works on (the pool's full tokens), else
    the largest-sized one reachable from the pool's allocator."""
    try:
        from sglang.srt.layers.moe import kt_kv_prefill_lend

        a = kt_kv_prefill_lend.STATE.get("alloc")
        if a is not None and getattr(a, "free_pages", None) is not None:
            return a
    except Exception:
        pass
    seen, todo, best = set(), [HOST.get("allocator")], None
    while todo:
        a = todo.pop()
        if a is None or id(a) in seen:
            continue
        seen.add(id(a))
        todo += [v for k, v in vars(a).items() if k.endswith("allocator") and v is not a]
        if getattr(a, "free_pages", None) is not None and int(getattr(a, "size", 0) or 0) > 0:
            if best is None or int(a.size) > int(best.size):
                best = a
    return best


def _rows_in_use(t) -> int:
    """rows in use v2. How many leading rows of a host-made pool tensor some request may
    hold: above the highest page that is not free in the pool's allocator (the tensor's rows count tokens or
    pages)."""
    a = _pool_allocator()
    if a is None:
        return t.shape[0]
    p = int(getattr(a, "page_size", 1) or 1)
    pages = int(a.size) // p
    free = a.free_pages
    rel = getattr(a, "release_pages", None)
    if rel is not None and len(rel):
        free = torch.cat((free, rel))
    try:
        from sglang.srt.layers.moe import kt_kv_prefill_lend

        lent = kt_kv_prefill_lend.STATE.get("lent")
        if kt_kv_prefill_lend.STATE.get("alloc") is a and lent is not None and len(lent):
            free = torch.cat((free, lent.to(free.dtype).to(free.device)))
    except Exception as e:
        _say_once("kv-vmm: the lent pages were not counted as free rows", e)
    used = torch.ones(pages + 1, dtype=torch.bool, device=free.device)  # rows range
    free = free.long()
    used[free[free <= pages]] = False
    used[0] = False
    idx = used.nonzero()
    top = int(idx.max()) + 1 if len(idx) else 1
    by_tokens = t.shape[0] > pages + 1
    if os.environ.get("KT_HOST_KV_DEBUG") == "1":  # rows debug
        print(f"[kt-host-kv] rows in use: {type(a).__name__} size {int(a.size)} page {p}, {len(a.free_pages)} free, "
              f"{0 if rel is None else len(rel)} released, {int(free.numel())} counted free, {int(used.sum())} used, "
              f"top {top}, rows {t.shape[0]} by {'tokens' if by_tokens else 'pages'}", flush=True)
    return min(t.shape[0], (top + 1) * p if by_tokens else top + 1)


def host_place_now(device, allocator=None) -> None:
    """After the graphs (Scheduler.init_model_worker): what is free on the card now, less KT_HOST_KV_MARGIN_MB
    (256), takes the first rows of the host-made pool tensors; the allocator hands out the lowest pages first."""
    if allocator is not None:
        keep_lowest_first(allocator)
        if HOST is not None:
            HOST["allocator"] = allocator
    if os.environ.get("KT_HOST_KV_PLACE", "after-extend") != "start" and HOST is not None and not HOST.get("go"):
        # placement waits for the first decode step after an extend (maybe_place)
        HOST["pending"] = device
        print("[kt-host-kv] the pool's rows stay in host memory until the first decode step after an extend",
              flush=True)
        return
    index = torch.device(device).index
    index = torch.cuda.current_device() if index is None else index
    if not any(t.device.index == index for t in HOST_MADE):
        return
    torch.cuda.synchronize(index)
    torch.cuda.empty_cache()
    # logits headroom (margin, inside host_fill's room)
    # the SWA pages a prefill window maps back stay free
    from sglang.srt.layers.moe import kt_kv_prefill_lend

    swa = kt_kv_prefill_lend.swa_window_bytes()
    if swa:
        print(f"[kt-host-kv] cuda:{index}: {swa / (1 << 30):.2f} GiB kept free for the SWA pages of a prefill window",
              flush=True)
    total = sum(t.nbytes for t in HOST_MADE if t.device.index == index)
    # every row the room holds (a later prompt's prefill reads them); the rows in flight
    # come first at later decodes (host_in_flight)
    moved = host_fill(index)
    got = {"rows": _card_tokens(index), "whole": 0, "split": moved}
    rows = HOST["rows_of_pool"] if HOST else None
    print(f"[kt-host-kv] cuda:{index}: {len(HOST_MADE)} pool tensors, {total / (1 << 30):.2f} GiB; "
          f"{got['split'] / (1 << 30):.2f} GiB on the card ({got['whole'] / (1 << 30):.2f} GiB of them whole), "
          + (f"the first {got['rows']} of {rows} tokens of the rest" if got["rows"] is not None else "all of it")
          + "; host memory the remainder", flush=True)


def _extend_rows(index: int, forward_batch, reserve: int = 0) -> bool:
    """A forward that does not stream writes its KV where the allocator put it, and nothing planned those rows: the
    pool's card rows reach them before it runs (host_in_flight: the hot lend's slots give what the card is short
    of). A 4,096-token prompt wrote one page past the card's 49,152 (Qwen3.5 with streaming from 8,192, 11-11).
    Whether any rank is past its card is one count over the ranks: host_in_flight exchanges counts itself.
    reserve: what the forward itself takes, kept out of the room the rows come into.
    host_in_flight takes the cheaper of the card's first rows up to them and their own chunks (spots)."""
    loc = getattr(forward_batch, "out_cache_loc", None)
    held = _card_tokens(index)
    over = torch.tensor([int(loc is not None and held is not None and loc.numel() > 0 and int(loc.max()) >= held
                             and not _spot_covered(index, loc))])
    try:
        from sglang.srt.distributed.parallel_state import get_tp_group

        g = get_tp_group()
        if g.world_size > 1:
            torch.distributed.all_reduce(over, op=torch.distributed.ReduceOp.MAX, group=g.cpu_group)
    except ImportError:
        pass
    if int(over):
        host_in_flight(index, "an extend that does not stream", window=False, reserve=reserve, loc=loc)
    return False


def _rows_now(index: int, loc, seq: int = 0) -> None:
    """maybe_place, an extend of more than 256 rows: the rows it writes that are not on the card come now. The plan
    brought the rows of the pages the allocator would hand out next (prompt_rows, _next_locs); pages freed after it
    are lower and the allocator hands those out first: 162 pages of a cached prompt went free twice during a 131K
    prompt and its sixth window wrote pages 1216..66239 with spots at 66240 and up (Flash-Next NVFP4 on one card,
    11-12). A launch's first prompts run before any plan has measured a row: 1,024 and 4,096 rows written with no
    row of the pool on the card (Gemma 4, 11-12).
    In decode's memory (a prompt's first forward) host_in_flight brings them, the hot slots giving what is short.
    In a window the plan has priced what is free against its chunk: the spots no request reads and no page ahead
    will use go out and the forward's rows take their bytes, no more.
    Whether any rank is short is one count over the ranks: host_in_flight exchanges counts itself."""
    held = _card_tokens(index)
    over = torch.tensor([int(held is not None and loc.numel() > 0 and int(loc.max()) >= held
                             and not _spot_covered(index, loc))])
    try:
        from sglang.srt.distributed.parallel_state import get_tp_group

        g = get_tp_group()
        if g.world_size > 1:
            torch.distributed.all_reduce(over, op=torch.distributed.ReduceOp.MAX, group=g.cpu_group)
    except ImportError:
        pass
    if not int(over):
        return
    try:
        from sglang.srt.layers.moe import kt_lend

        in_window = kt_lend.STATE.get("phase") == "prefill"
    except ImportError:
        in_window = False
    if not in_window:
        host_in_flight(index, "an extend's rows", window=False, loc=loc)
        return
    ts = [t for t in HOST_MADE if t.device.index == index]
    if not ts or not all(_spot_able(t) for t in ts):
        return
    _flight_refresh(index)  # a flushed cache's requests are gone: their spots are no one's
    SPOT_GEN[0] += 1
    gen = SPOT_GEN[0]
    now = loc.long()
    kept = _flight_locs()
    if kept is not None:
        now = torch.cat((now.to(kept.device), kept))
    first = _spot_want(index, now, gen)  # this forward's rows and those its request reads
    cost = sum(len(k) * c for _, k, c in first)
    ahead = _next_locs(int(FLIGHT.get("ahead", 0) or 0)) if FLIGHT.get("ahead") else None
    far = {}  # the spots of the pages ahead that were there, before this forward's own marked them
    if ahead is not None:
        for t in ts:
            sp = SPOTS.get(t.data_ptr(), {})
            far[t.data_ptr()] = [k for k in _spot_chunks(t, ahead) if k in sp and sp[k] < gen]
    later = _spot_want(index, ahead.to(now.device), gen) if ahead is not None else []
    got = _spots_out(index, gen)
    if got < cost and far:
        # the pages ahead give way to the rows this forward writes, the farthest first (their own forwards bring
        # them back the same way): 376 MiB of spots no one read against 384 for a window's rows, one chunk written
        # in host memory (Flash-Next NVFP4 on one card, 11-12)
        lib = _load()
        torch.cuda.synchronize(index)
        while got < cost and any(far.values()):
            for t in ts:
                ks = far.get(t.data_ptr())
                if not ks or got >= cost:
                    continue
                k, c = ks.pop(), chunk_of(t)
                if _prefix_chunk(t, k, c):
                    continue
                n = lib.kv_vmm_place_range(ctypes.c_void_p(t.data_ptr()), ctypes.c_size_t(k * c), ctypes.c_size_t(c), 0)
                got += max(0, n)
                SPOTS[t.data_ptr()].pop(k, None)
        torch.cuda.synchronize(index)
        later = []
    if got < cost:
        # and what the card has free past this forward's own memory as the plan measured it (nothing measured at
        # a launch's first prompts: 4,096 rows against the 80 MiB of spots a flushed 1,024-token prompt left,
        # Gemma 4, 11-12)
        try:
            from sglang.srt.layers.moe import kt_prefill_plan as p

            rows = int(loc.numel())
            own = int(p.cost(rows, seq))
        except Exception:
            own = 0
        got += max(0, min(cost - got, _room_now(index) - own))
    if got <= 0:
        return
    used = _spot_place(index, first, gen, int(loc.numel()), "a window's rows", got) if first else 0
    if later and got - used > 0:
        _spot_place(index, later, gen, int(FLIGHT.get("ahead", 0) or 0), "the pages ahead", got - used)


VOCAB = 0  # the model's vocabulary without a host pool to say it (kt_lend.before_forward; -1: asked, unknown)


def extend_room(device, rows=None, logprobs: bool = True, forward_batch=None) -> None:
    """kt_lend.before_forward, before a forward that does not stream (decode's memory): its KV rows on the card
    (_extend_rows), then the card's free memory up to what this forward takes (KT_HOST_KV_MARGIN_MB; rows: what the
    plan measured a row of a forward costs; with its input logprobs two vocab-wide fp32 rows a token). The hot lend
    gives the slots that cover what is short, the same count on every rank; the pool's rows yield only what the
    slots could not. The next decode brings them back (host_in_flight)."""
    if device is None:
        return
    index = torch.device(device).index
    index = torch.cuda.current_device() if index is None else index
    # without a host pool on this card the hot lend's slots are the only room there is: they took all that was free,
    # and a 480-token extend found 0.35 GiB and ran out on a 256 MiB buffer (Flash-Next GGUF on one card, 11-17)
    host = HOST is not None and bool(HOST.get("go")) and any(t.device.index == index for t in HOST_MADE)
    own = 0
    if rows is not None:
        try:
            from sglang.srt.layers.moe import kt_prefill_plan

            row = float(kt_prefill_plan.ROW or 0)
        except Exception:
            row = 0.0
        own = int(row * int(rows)) + (2 * 4 * int(rows) * (int((HOST or {}).get("vocab", 0) or 0) or max(VOCAB, 0)) if logprobs else 0)
    # the forward's own memory stays out of the room its rows come into: without it the rows took what was free,
    # the forward was short of its logprobs, the rows went back, and the next forward brought them in again; twelve
    # 256-token scoring forwards after a 131K prompt each moved 0.84 GiB in and 0.94 out until none were left on the
    # card and the thirteenth did not fit (Flash-Next NVFP4 on one card, 11-11)
    if host and forward_batch is not None:
        _extend_rows(index, forward_batch, reserve=own + (64 << 20))
    need = margin() + own
    torch.cuda.empty_cache()
    short = need - card_free(index)
    # the slots first: with the margin of the longest forward that could come (3.9 GiB at 2,048 tokens of input
    # logprobs) against the 1 GiB the hot lend leaves free, every 256-token scoring forward after a long prompt sent
    # all of the pool's rows to host memory, and the 658-token ones wrote theirs there (Flash-Next GGUF, 11-10)
    try:
        from sglang.srt.layers.moe import kt_kv_hot_lend as h
    except ImportError:
        h = None
    if h is not None and h.LAYERS:
        lent = int(h.STATE.get("lent_slots", 0) or 0)
        v = [min(lent, h.slots_for(short + (64 << 20))) if short > 0 and lent else 0, -lent]
        if h._world() > 1:
            v = h._over_ranks(v, torch.distributed.ReduceOp.MAX)
        if v[0]:
            if v[0] >= -v[1]:
                from sglang.srt.layers.moe import kt_kv_prefill_lend as kpl

                if kpl.STATE.get("lent") is not None and kpl.STATE.get("owner") == "hot":
                    kpl.hot_release("a short extend")
                else:
                    h.release("a short extend")
            else:
                h.shrink(0, "a short extend", slots=v[0])
            torch.cuda.empty_cache()
            short = need - card_free(index)
    if host and short > 0 and host_yield(index, short + (64 << 20), why="a short extend") < short:
        try:  # who holds the card when the pool's rows could not cover a forward
            from sglang.srt.layers.moe import kt_ledger

            kt_ledger.snapshot(index, "D", "a short extend the pool's rows did not cover")
        except Exception:
            pass


def maybe_place(forward_batch) -> None:
    """kt_lend.before_forward, every forward: notes an extend, and at the first decode step after one moves the
    pool's rows to the card (host_place_now, what is free then). At every later first
    decode after an extend, the rows in flight (host_in_flight)."""
    if HOST is None or (HOST.get("pending") is None and not HOST.get("go")):
        return
    if HOST.get("pending") is None:
        mode = getattr(forward_batch, "forward_mode", None)
        if mode is not None and mode.is_extend():
            HOST["seen_extend"] = True
            loc = getattr(forward_batch, "out_cache_loc", None)
            index = torch.cuda.current_device()
            held = _card_tokens(index) if loc is not None and loc.numel() > 256 else None
            if held is not None:
                FLIGHT["ahead"] = max(0, int(FLIGHT.get("ahead", 0) or 0) - int(loc.numel()))
                lens = getattr(forward_batch, "seq_lens", None)
                _rows_now(index, loc, int(lens.max()) if lens is not None and lens.numel() else int(loc.numel()))
            if held is not None and int(loc.max()) >= held and not _spot_covered(index, loc):
                # never meant to happen: the plan brings a prompt's rows to the card before its chunks run
                # (prompt_rows), and _rows_now those of the pages the allocator gave instead. Attention over rows
                # in host memory runs at a quarter of the speed
                HOST["past_card"] = int(HOST.get("past_card", 0) or 0) + 1
                print(f"[kt-host-kv] HOST ROWS IN A PREFILL: an extend of {int(loc.numel())} rows writes pages up to "
                      f"{int(loc.max())}, the card holds the first {held} ({HOST['past_card']} so far)", flush=True)
            if os.environ.get("KT_HOST_KV_DEBUG") == "1":
                a = _pool_allocator()
                if loc is not None and loc.numel() > 256:
                    fp = getattr(a, "free_pages", None)
                    print(f"[kt-host-kv] extend of {int(loc.numel())} rows writes pages {int(loc.min())}.."
                          f"{int(loc.max())}, the card holds the first {_card_tokens(torch.cuda.current_device())}; "
                          f"{type(a).__name__} lowest-first {getattr(a, '_kt_lowest_first', None)}, need_sort "
                          f"{getattr(a, 'need_sort', None)}, free {None if fp is None else len(fp)}"
                          f"{'' if fp is None or not len(fp) else f' from {int(fp[0])} (min {int(fp.min())})'}",
                          flush=True)
        elif mode is not None and mode.is_decode() and HOST.get("seen_extend"):
            HOST["seen_extend"] = False
            FLIGHT["ahead"] = 0
            host_in_flight(torch.cuda.current_device(), "decode after an extend")
        return
    mode = getattr(forward_batch, "forward_mode", None)
    if mode is None:
        return
    if mode.is_extend():
        HOST["seen_extend"] = True
        return
    if mode.is_decode() and HOST.get("seen_extend"):
        device = HOST.pop("pending")
        HOST["go"] = True
        host_place_now(device)


def _direct(orig, zero: bool):
    """torch.zeros / torch.empty for CUDA tensors in region(): a tensor LAZY claims
    is made lazily; on old torch every other one is its own kv_vmm allocation
    (DIRECT), else it goes to the original (inside the MemPool); anything
    unusual goes to the original."""

    def make(*size, dtype=None, device=None, **kw):
        dev = torch.device(device) if device is not None else None
        if (dev is None or dev.type != "cuda" or kw.get("out") is not None or kw.get("pin_memory")
                or kw.get("layout", torch.strided) != torch.strided
                or kw.get("memory_format", torch.contiguous_format) != torch.contiguous_format
                or ("size" in kw) == bool(size)):
            return orig(*size, dtype=dtype, device=device, **kw)
        if "size" in kw:  # torch.zeros(size=(...)) as well as torch.zeros(a, b) and torch.zeros((a, b))
            size = (kw.pop("size"),)
        kw.pop("memory_format", None)
        shape = tuple(size[0]) if len(size) == 1 and isinstance(size[0], (tuple, list, torch.Size)) else size
        shape = tuple(int(d) for d in shape)
        dtype = dtype or torch.get_default_dtype()
        index = dev.index if dev.index is not None else torch.cuda.current_device()
        n = 1
        for d in shape:
            n *= int(d)
        size_b = n * dtype.itemsize
        if (HOST is not None and shape and _token_dim(shape) is not None and size_b >= HOST.get("min_bytes", 1 << 20)
                and not kw.get("requires_grad")):
            return _host_tensor(shape, dtype, index)  # zeroed, in host memory
        if HOST is not None and os.environ.get("KT_HOST_KV_DEBUG") == "1" and size_b >= (64 << 20):
            # region debug
            print(f"[kt-host-kv] on the card: {shape} {dtype} {size_b / (1 << 20):.0f} MiB", flush=True)
        lead = LAZY(shape, size_b) if LAZY is not None and not kw.get("requires_grad") else None
        if lead is not None:
            return _lazy_tensor(shape, dtype, index, lead)
        if not DIRECT:
            return orig(*size, dtype=dtype, device=device, **kw)
        with torch.cuda.device(index):
            t = torch.as_tensor(_Block(max(1, size_b), index), device=f"cuda:{index}")
        t = t[:size_b].view(dtype).view(shape)
        if kw.get("requires_grad"):
            t.requires_grad_(True)
        return t.zero_() if zero else t

    return make


@contextlib.contextmanager
def region():
    """Tensors made inside sit on remappable memory."""
    if not ENABLED:
        yield
        return
    _load()
    zeros, empty = torch.zeros, torch.empty
    torch.zeros, torch.empty = _direct(zeros, True), _direct(empty, False)
    try:
        if DIRECT:
            yield
        else:
            with torch.cuda.use_mem_pool(_pool):
                yield
    finally:
        torch.zeros, torch.empty = zeros, empty

def shadow_tensor(shape, dtype, device: int) -> torch.Tensor:
    """A tensor on a shadow range (kv_vmm_malloc_shadow): every byte reads as zero
    until restore maps real chunks under it, and again after release. Its
    chunk is the device granularity (2 MiB)."""
    lib = _load()
    if not hasattr(lib, "kv_vmm_malloc_shadow"):
        raise RuntimeError(f"{_LIB_PATH} has no kv_vmm_malloc_shadow: rebuild it from kv_vmm.c")
    n = 1
    for d in shape:
        n *= int(d)
    size_b = n * dtype.itemsize
    with torch.cuda.device(device):
        t = torch.as_tensor(_Block(max(1, size_b), device, shadow=True), device=f"cuda:{device}")
    return t[:size_b].view(dtype).view(tuple(int(d) for d in shape))


def split_tensor(shape, dtype, device: int, device_bytes: int, numa: int) -> torch.Tensor:
    """A zeroed tensor whose first device_bytes are device memory and the rest host
    memory of NUMA node numa, one range the GPU addresses as a whole (DeepSeek-V4.1's
    compressed KV past what the GPU holds). release / restore leave the host part."""
    lib = _load()
    if not hasattr(lib, "kv_vmm_malloc_split"):
        raise RuntimeError(f"{_LIB_PATH} has no kv_vmm_malloc_split: rebuild it from kv_vmm.c")
    n = 1
    for d in shape:
        n *= int(d)
    size_b = n * dtype.itemsize
    with torch.cuda.device(device):
        t = torch.as_tensor(_Block(max(1, size_b), device, split=(max(0, device_bytes), numa)), device=f"cuda:{device}")
    return t[:size_b].view(dtype).view(tuple(int(d) for d in shape)).zero_()


def _place_sig():
    lib = _load()
    if not hasattr(lib, "kv_vmm_place"):
        raise RuntimeError(f"{_LIB_PATH} has no kv_vmm_place: rebuild it from kv_vmm.c")
    lib.kv_vmm_place.restype = ctypes.c_longlong
    lib.kv_vmm_place.argtypes = [ctypes.c_void_p, ctypes.c_size_t]


def device_bytes(t: torch.Tensor) -> int:
    """How many bytes from the start of t's kv_vmm range are device memory."""
    lib = _load()
    if not hasattr(lib, "kv_vmm_device_bytes"):
        return t.nbytes
    return int(lib.kv_vmm_device_bytes(ctypes.c_void_p(t.data_ptr())))


def chunk_bytes(device: int) -> int:
    return int(_load().kv_vmm_chunk_bytes(device))


def chunk_of(t: torch.Tensor) -> int:
    """The chunk size of the kv_vmm allocation holding t (0: t is not on kv_vmm memory).
    Large allocations (the K/V buffers, KV_VMM_BIG_MB and up) use KV_VMM_CHUNK_MB,
    smaller ones the device granularity."""
    return int(_load().kv_vmm_chunk_at(ctypes.c_void_p(t.data_ptr())))


def released_bytes(device: int) -> int:
    return int(_load().kv_vmm_released_bytes(device)) if _lib is not None else 0


def release(t: torch.Tensor, off: int, length: int) -> int:
    """Unmap the chunks wholly inside t's bytes [off, off + length). The caller has
    synchronised the device: nothing may be reading them."""
    return int(_load().kv_vmm_release(ctypes.c_void_p(t.data_ptr()), off, length))


def restore(t: torch.Tensor, off: int, length: int) -> int:
    """Map zeroed chunks wherever t's bytes [off, off + length) have none. -1: no memory."""
    return int(_load().kv_vmm_restore(ctypes.c_void_p(t.data_ptr()), off, length))


# token dim. A pool tensor's token rows are the first dimension of the pool's row count with
# at most a few blocks before it (K and V stacked: (2, tokens + page, ...)); each block's first rows go to the card,
# through kv_vmm_place_range. These definitions replace the prefix-only ones above.
HOST_INFO = {}


# The blocks a pool tensor may have before its token rows: K and V stacked ((2, tokens + page, ...)), or one per layer
# (a page-row slab, (layers, pages + 1, row bytes)). Only the KV pool's tensors are made in the region.
HOST_MAX_BLOCKS = int(os.environ.get("KT_HOST_KV_MAX_BLOCKS", "1024"))


def _token_dim(shape):
    outer = 1
    for d, n in enumerate(shape):
        if int(n) in HOST["rows"]:
            return d if outer <= HOST_MAX_BLOCKS else None
        outer *= int(n)
    return None


def _drop_model_cache() -> None:
    """Before the first host tensor: posix_fadvise(DONTNEED) on the model's files (KT_GPU_STREAM_DROP_CACHE). The host
    part of a split range is 2 MiB chunks of one NUMA node, and the driver does not push the shards' page cache aside
    for them: a pool stopped at 39 of its 1 GiB tensors on node 1 with 30 GiB free and 165 GiB of file
    pages. The weights are in the arenas by then."""
    root_dir = os.environ.get("KT_GPU_STREAM_DROP_CACHE", "")
    if HOST.get("dropped") or not root_dir:
        return
    HOST["dropped"] = True
    import time

    t0, n = time.time(), 0
    for root, _, files in os.walk(root_dir, followlinks=True):
        for name in files:
            try:
                fd = os.open(os.path.join(root, name), os.O_RDONLY)
            except OSError:
                continue
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                n += 1
            finally:
                os.close(fd)
    print(f"[kt-host-kv] dropped the page cache of {n} files under {root_dir} before the pool's host tensors "
          f"({time.time() - t0:.1f} s)", flush=True)


def _node_memory(node: int) -> str:
    """A node's free memory, the part of it in free blocks of 2 MiB or more, and the free pages the kernel counts as
    CMA (only movable allocations may take those; the driver's pages are not movable)."""
    free = cma = big = 0
    for line in open(f"/sys/devices/system/node/node{node}/meminfo"):
        p = line.split()
        if p[2] == "MemFree:":
            free = int(p[3]) / (1 << 20)
    for line in open(f"/sys/devices/system/node/node{node}/vmstat"):
        p = line.split()
        if p[0] == "nr_free_cma":
            cma = int(p[1]) * 4096 / (1 << 30)
    for line in open("/proc/buddyinfo"):
        p = line.split()
        if p[1].rstrip(",") == str(node) and p[3] == "Normal":
            big += sum(int(c) << o for o, c in enumerate(p[4:]) if o >= 9) * 4096 / (1 << 30)
    cma_kb = " / ".join(l.split()[1] for l in open("/proc/meminfo") if l.startswith("Cma"))
    return (f"free {free:.1f} GiB, {big:.1f} GiB in 2 MiB blocks, {cma:.1f} GiB counted as free CMA "
            f"(CmaTotal / CmaFree {cma_kb} kB)")


def _nodes_by_distance(node: int) -> list:
    """The NUMA nodes with memory, node first and the rest nearest first (the node's distance row)."""
    try:
        dist = [int(x) for x in open(f"/sys/devices/system/node/node{node}/distance").read().split()]
        with_mem = open("/sys/devices/system/node/has_normal_memory").read().strip()
    except OSError:
        return [node]
    have = set()
    for part in with_mem.split(","):
        a, _, b = part.partition("-")
        have.update(range(int(a), int(b or a) + 1))
    return [node] + sorted((n for n in have if n != node and n < len(dist)), key=lambda n: dist[n])


# KT_HOST_KV_NODE_RESERVE_MB: what a node keeps free under the pool's host tensors (the process's own later
# allocations, the CUDA graphs' capture, the kernel's watermarks). 8 GiB still let the OOM killer take the scheduler
# bound to node 1 in one 1M start of three (11-08)
NODE_RESERVE = int(os.environ.get("KT_HOST_KV_NODE_RESERVE_MB", "16384")) << 20


def _node_usable(node: int) -> int:
    """Bytes the driver's unmovable pages may still take on a node: free, less the free pages the kernel counts as
    CMA (no unmovable allocation gets those), less the zones' min watermark (no allocation goes under it, and it
    carries the boost the driver's own fragmenting leaves: 12.9 GiB on node 1, 4.0 on node 0, 11-11), less
    NODE_RESERVE."""
    free = cma = wmark = 0
    try:
        here = False
        for line in open("/proc/zoneinfo"):
            p = line.split()
            if p and p[0] == "Node":
                here = int(p[1].rstrip(",")) == node
            elif here and len(p) == 2 and p[0] == "min":
                wmark += int(p[1]) * 4096
    except (OSError, ValueError):
        wmark = 0
    try:
        for line in open(f"/sys/devices/system/node/node{node}/meminfo"):
            p = line.split()
            if p[2] == "MemFree:":
                free = int(p[3]) << 10
        for line in open(f"/sys/devices/system/node/node{node}/vmstat"):
            p = line.split()
            if p[0] == "nr_free_cma":
                cma = int(p[1]) * 4096
    except OSError:
        return 1 << 62
    return free - cma - wmark - NODE_RESERVE


def _host_tensor(shape, dtype, index: int) -> torch.Tensor:
    _drop_model_cache()
    _place_sig()
    lib = _load()
    lib.kv_vmm_place_range.restype = ctypes.c_longlong
    lib.kv_vmm_place_range.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int]
    # The host part is the driver's own pages (GFP_KERNEL on the node), and the kernel keeps what it counts as free
    # CMA from those. A count with no CMA area (CmaTotal 0, CmaFree 43 GB from boot on 7.0.0-38, 0.8 GB on -42) leaves
    # tens of GiB free that the driver cannot take: a pool stopped 9 GiB short on node 1 with 45 GiB
    # free. The tensors that do not fit go to the next nearest node; the GPU reads them across the socket link.
    order = HOST.setdefault("nodes", {}).setdefault(index, _nodes_by_distance(_host_numa(index)))
    size = torch.Size(shape).numel() * torch.empty((), dtype=dtype).element_size()
    now, failed = HOST.setdefault("node_now", {}), HOST.setdefault("node_failed", {}).setdefault(index, set())
    while True:
        # The nearest node that keeps NODE_RESERVE under the tensor: filling a node to the last page let the
        # process's next allocation wake the OOM killer (a 1M pool, 11-08: the scheduler bound to node 1 was
        # killed capturing the graphs, 119 of 120 host tensors on that node). When no node does, the pool is still
        # the context: the tensor goes into the reserve of the node with the most left (a 1M pool needs
        # 150 GiB of host KV where the two nodes have 153 after the arenas, 11-11). Read for every tensor: the
        # other ranks are filling the same nodes.
        left = [n for n in order if n not in failed]
        usable = {n: _node_usable(n) for n in left}
        fit = [n for n in left if usable[n] >= size]
        node = fit[0] if fit else max(left, key=lambda n: usable[n])
        if now.get(index, order[0]) != node or (not fit and not HOST.get("into_reserve")):
            why = (f"keeps {NODE_RESERVE >> 30} GiB free" if fit else
                   f"no node keeps {NODE_RESERVE >> 30} GiB free under the pool, the rest goes into the reserves")
            print(f"[kt-host-kv] cuda:{index}: host KV from here on node {node} ({why}); "
                  + "; ".join(f"node {n}: {_node_memory(n)}" for n in left), flush=True)
            HOST["into_reserve"] = HOST.get("into_reserve") or not fit
        now[index] = node
        try:
            t = split_tensor(shape, dtype, index, 0, node)
            break
        except torch.OutOfMemoryError:
            failed.add(node)
            print(f"[kt-host-kv] cuda:{index}: no host pages for the KV on node {node}: {_node_memory(node)}",
                  flush=True)
            if len(failed) == len(order):
                raise
    nodes = [node]
    d = _token_dim(shape)
    outer = 1
    for n in shape[:d]:
        outer *= int(n)
    rows = int(shape[d])
    HOST_INFO[t.data_ptr()] = {"outer": outer, "rows": rows, "block": t.nbytes // outer,
                               "row": t.nbytes // (outer * rows), "dev": 0, "numa": nodes[0]}
    HOST_MADE.append(t)
    if os.environ.get("KT_HOST_KV_DEBUG") == "1":  # the pool's host tensors as they are made, and their sum
        total = sum(x.nbytes for x in HOST_MADE if x.device.index == index)
        print(f"[kt-host-kv] cuda:{index} host tensor {tuple(shape)} {dtype} {t.nbytes / (1 << 30):.2f} GiB, "
              f"{len(HOST_MADE)} made, {total / (1 << 30):.2f} GiB, this one on node {nodes[0]}", flush=True)
    return t


def _info(t):
    return HOST_INFO[t.data_ptr()]


def _set_rows(t, rows: int, live: bool = False) -> int:
    """The first rows of every block of t on the card, the rest in host memory; returns the bytes moved or -1.
    live: on the way to host memory, chunks with nothing mapped (lent pages) are left as they are (a park)."""
    lib = _load()
    place = lib.kv_vmm_place_range
    if live and hasattr(lib, "kv_vmm_place_range_live"):
        place = lib.kv_vmm_place_range_live
        place.restype = ctypes.c_longlong
        place.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int]
    i = _info(t)
    rows = max(0, min(rows, i["rows"]))
    moved = 0
    for o in range(i["outer"]):
        base = o * i["block"]
        if rows > i["dev"]:
            n = lib.kv_vmm_place_range(ctypes.c_void_p(t.data_ptr()), ctypes.c_size_t(base + i["dev"] * i["row"]),
                                       ctypes.c_size_t((rows - i["dev"]) * i["row"]), 1)
        elif rows < i["dev"]:
            n = place(ctypes.c_void_p(t.data_ptr()), ctypes.c_size_t(base + rows * i["row"]),
                                       ctypes.c_size_t((i["dev"] - rows) * i["row"]), 0)
        else:
            n = 0
        if n < 0:
            return -1
        moved += n
    grew = rows > i["dev"]
    i["dev"] = rows
    s = SPOTS.get(t.data_ptr())
    if grew and s:  # the first rows now hold these chunks
        c = chunk_of(t)
        for k in [k for k in s if _prefix_chunk(t, k, c)]:
            del s[k]
    return moved


def _tokens_a_row(t) -> int:
    full = int(HOST["full"]) if HOST else 0
    return max(1, round(full / max(1, _info(t)["rows"] - 1))) if full else 1


def _bytes_a_token(t) -> float:
    i = _info(t)
    return i["row"] * i["outer"] / _tokens_a_row(t)


def host_place(device: int, free_bytes: int) -> dict:
    """Tensors grouped by bytes a token, fewest first: a group that fits goes whole; the first that does not, and
    every later one, share one token count (in each block, its first rows)."""
    ts = [t for t in HOST_MADE if t.device.index == device]
    if not ts:
        return {"rows": None, "whole": 0, "split": 0}
    groups = {}
    for t in ts:
        groups.setdefault(round(_bytes_a_token(t), 3), []).append(t)
    left, whole, tokens = free_bytes, 0, None
    order = sorted(groups)

    def edge(tensors):  # a block's rows ending mid-chunk bring that whole chunk along
        return sum(2 * chunk_of(t) * _info(t)["outer"] for t in tensors)

    for k, bt in enumerate(order):
        size = sum(t.nbytes for t in groups[bt])
        if size + edge(groups[bt]) <= left:
            for t in groups[bt]:
                n = _set_rows(t, _info(t)["rows"])
                if n < 0:
                    raise torch.OutOfMemoryError(f"kv_vmm: placing a host KV tensor on cuda:{device}")
                left -= n
            whole += size
            continue
        rest = [t for r in order[k:] for t in groups[r]]
        tokens = int(max(0, left - edge(rest)) // max(1e-9, sum(_bytes_a_token(t) for t in rest)))
        for t in rest:
            n = _set_rows(t, tokens // _tokens_a_row(t))
            if n < 0:
                raise torch.OutOfMemoryError(f"kv_vmm: placing a host KV tensor on cuda:{device}")
            left -= n
        break
    torch.cuda.synchronize(device)
    return {"rows": tokens, "whole": whole, "split": free_bytes - left}


def _rows_in_use(t) -> int:
    """The leading rows of each block some request may hold (the pool's allocator, pages 1..size // page)."""
    a = _pool_allocator()
    i = _info(t)
    if a is None:
        return i["rows"]
    p = int(getattr(a, "page_size", 1) or 1)
    pages = int(a.size) // p
    free = a.free_pages
    rel = getattr(a, "release_pages", None)
    if rel is not None and len(rel):
        free = torch.cat((free, rel))
    try:
        from sglang.srt.layers.moe import kt_kv_prefill_lend

        lent = kt_kv_prefill_lend.STATE.get("lent")
        if kt_kv_prefill_lend.STATE.get("alloc") is a and lent is not None and len(lent):
            free = torch.cat((free, lent.to(free.dtype).to(free.device)))
    except Exception as e:
        _say_once("kv-vmm: the lent pages were not counted as free rows (map)", e)
    used = torch.ones(pages + 1, dtype=torch.bool, device=free.device)
    free = free.long()
    used[free[free <= pages]] = False
    used[0] = False
    idx = used.nonzero()
    top = int(idx.max()) + 1 if len(idx) else 1
    by_tokens = i["rows"] > pages + 1
    keep = (top + 1) * p if by_tokens else top + 1
    if os.environ.get("KT_HOST_KV_DEBUG") == "1":
        print(f"[kt-host-kv] rows in use: {type(a).__name__} size {int(a.size)} page {p}, top {top}, "
              f"rows {i['rows']} by {'tokens' if by_tokens else 'pages'}, keep {keep}", flush=True)
    return min(i["rows"], keep)


def host_yield(device: int, need: int, why: str = "the weights", held_too: bool = True) -> int:
    """The card's rows go back to host memory until need bytes are free: first those above what any request holds,
    then, if that is not enough, the rows the requests hold (the copy keeps their contents and the GPU reads them in
    host memory, as it reads every row past the card's share; a 58K prompt in a 64K pool held every card
    row, and the lend's weights could not come back, 1.4 GB short).
    The tensors with the most bytes a token give first (host_place's order reversed), all of a group down to one
    shared token count, the smallest that frees enough; a group with fewer bytes a token gives only when the larger
    ones down to nothing are not enough. Emptying one tensor after another left some layers with no card rows and
    every decode step read them in host memory (a 1M pool, 11-08: long-context decode 12.2 -> 7.8 tok/s).
    held_too=False (the hot lend, level 5 before its slots): only the rows past those in flight and HEADROOM go."""
    torch.cuda.synchronize(device)
    _flight_refresh(device)
    ts = [t for t in HOST_MADE if t.device.index == device]
    groups = {}
    for t in ts:
        groups.setdefault(round(_bytes_a_token(t), 3), []).append(t)
    order = [groups[k] for k in sorted(groups, reverse=True)]

    # read once: the searches below ask for it at every step, and each reading walks the allocator's free list (180
    # tensors of a 1M pool: the hot lend's yield took a second of the first decode after a window, 11-11)
    # held_too=False: with the rows _target names, those a request in flight reads among the card's first. A request
    # whose last tokens sit in spots is not in FLIGHT["now"], _target was 0 and the hot slots' refill during a
    # 131K prompt's decode sent its 1.91 GiB of card rows to host memory, its spots of the window before with
    # them: decode after the prompt 40 -> 14 tok/s (Gemma 4 26B, 11-13)
    locs = None if held_too else _flight_locs()

    def read(t):
        if locs is None or not locs.numel():
            return 0
        r = locs // _tokens_a_row(t)
        r = r[r < _info(t)["dev"]]
        return int(r.max()) + 1 if r.numel() else 0

    kept = {t.data_ptr(): min(_info(t)["dev"], _rows_in_use(t) if held_too else max(_target(t), read(t)))
            for t in ts}

    def target(t, tokens, held):
        i = _info(t)
        keep = 0 if held else kept[t.data_ptr()]
        return max(keep, min(i["dev"], tokens // _tokens_a_row(t)))

    def freed(sel, tokens, held):
        # only whole chunks move: each block's range may lose part of one at either end (two chunks a block less)
        out = 0
        for t in sel:
            i = _info(t)
            rows = target(t, tokens, held)
            if rows < i["dev"]:
                out += max(0, (i["dev"] - rows) * i["row"] * i["outer"] - 2 * chunk_of(t) * i["outer"])
        return out

    # the spots of earlier forwards first (held_too=False: not those a request in flight reads)
    got, held, sel, tokens = _spots_out(device, SPOT_GEN[0], locs), False, [], 0
    for held in ((False, True) if held_too else (False,)) if got < need else ():
        for k in range(1, len(order) + 1):
            sel = [t for g in order[:k] for t in g]
            if freed(sel, 0, held) >= need - got:
                lo, hi = 0, max(_info(t)["dev"] * _tokens_a_row(t) for t in sel)
                while lo < hi:  # the largest shared token count that still frees enough
                    mid = (lo + hi + 1) // 2
                    if freed(sel, mid, held) >= need - got:
                        lo = mid
                    else:
                        hi = mid - 1
                tokens = lo
                break
        else:
            tokens = 0  # every group down to what is held (the first pass) or to nothing
        for t in sel:
            i = _info(t)
            rows = target(t, tokens, held)
            if rows >= i["dev"]:
                continue
            n = _set_rows(t, rows)
            if n > 0:
                got += n
            if os.environ.get("KT_HOST_KV_DEBUG") == "1" or n < 0:
                print(f"[kt-host-kv] yield {tuple(t.shape)}: down to {rows} {'held ' if held else ''}rows: {n} bytes",
                      flush=True)
        if got >= need:
            break
    if got < need and held_too:
        got += _spots_out(device)
    torch.cuda.synchronize(device)
    if not got and not held_too:
        return 0
    print(f"[kt-host-kv] cuda:{device}: {got / (1 << 30):.2f} GiB of the pool's rows back to host memory "
          f"for {why} ({need / (1 << 30):.2f} GiB short{', rows in use among them' if held else ''}), "
          f"the card holds the first {_card_tokens(device)} tokens", flush=True)
    return got


# the ledger's order for the pool's card rows: those the requests in flight read come
# before the hot slots (level 2), the others after them (level 5); level 1 stays free at the next window's start
HEADROOM = int(os.environ.get("KT_HOST_KV_HEADROOM_TOKENS", "8192"))


def _hot_on() -> bool:
    try:
        from sglang.srt.layers.moe import kt_kv_hot_lend

        return bool(kt_kv_hot_lend.ENABLED and kt_kv_hot_lend.LAYERS)
    except ImportError:
        return False


def _card_tokens(index: int):
    ts = [t for t in HOST_MADE if t.device.index == index]
    return min(_info(t)["dev"] * _tokens_a_row(t) for t in ts) if ts else None


def margin(floor: int = 0) -> int:
    """What the pool's card rows leave free in decode: KT_HOST_KV_MARGIN_MB, at least floor. A forward that does not
    stream makes its own room when it comes (extend_room: its rows, its input logprobs). Kept free for the longest
    such forward with logprobs (two vocab-wide fp32 rows a token up to the streaming length), the margin grew with
    that length: 16 GiB at 8,192 tokens, no row of the pool on the card and every request's rows in host memory
    (Qwen3.5, 11-11)."""
    return max(int(os.environ.get("KT_HOST_KV_MARGIN_MB", "256")) << 20, int(floor))


def margin_standing() -> int:
    """The margin that stands whatever comes: KT_HOST_KV_MARGIN_MB (margin() without a floor). A short forward's
    logprob rows are not in it: extend_room makes room for them when one arrives (hot slots first, then rows to the
    host), so a new prompt's rows planned in a window may take that memory (GLM-5.3 kept 2.4 GB for it and wrote
    its calibration prompts' rows past the card, 11-11)."""
    return int(os.environ.get("KT_HOST_KV_MARGIN_MB", "256")) << 20


def host_room(index: int, floor: int = 0) -> int:
    """Bytes the pool's card rows may still take: free now less the margin (at least floor), and at the next
    window's start (the ledger) level 1 kept free: the smallest window at the longest sequence measured, its SWA
    pages, the lend's margin."""
    from sglang.srt.layers.moe import kt_kv_prefill_lend, kt_ledger, kt_lend, kt_prefill_plan as p

    margin_ = margin(floor)
    torch.cuda.empty_cache()
    room = card_free(index) - margin_
    swa = int(kt_kv_prefill_lend.swa_window_bytes() or 0)
    w = kt_ledger.window_free(index)
    if w is None:
        return room - swa
    first = int(p.cost(p.ALIGN, p.longest()))
    return min(room, int(w) - first - swa - kt_lend.MARGIN)


def card_free(index: int) -> int:
    """The driver's free memory on the card, less the KV pages a lend handed over unmapped that the next lend has
    not decided on yet (kt_kv_prefill_lend.held_bytes): they are the pool's, mapped back the moment it needs them."""
    free = int(torch.cuda.mem_get_info(index)[0])
    try:
        from sglang.srt.layers.moe import kt_kv_prefill_lend
    except ImportError:
        return free
    return free - int(getattr(kt_kv_prefill_lend, "held_bytes", lambda: 0)())


def _reach(t, tokens: int) -> int:
    """Rows of t the allocator's next `tokens` tokens reach: it hands out the head of its free list, its lowest free
    pages once that is short (below), and the pages a lend holds are not among them. Counted from the highest page in use, a new prompt's rows were taken to start there:
    with 32,768 low pages lent to hot slots a 35,000-token prompt was given pages 4,097 short of them and the rest
    from 36,865 up to 67,769, 18,618 past the card's 49,151, and its one chunk ran at 2,494 tok/s for 6,860
    (Qwen3.5, the first window of a launch that planned it whole, 11-10). 0 without a free list to read."""
    a = _pool_allocator()
    if a is None or tokens <= 0:
        return 0
    i = _info(t)
    p = int(getattr(a, "page_size", 1) or 1)
    pages = int(a.size) // p
    k = -(-int(tokens) // p)
    free = a.free_pages
    rel = getattr(a, "release_pages", None)
    if len(free) >= k:
        # the allocator's own order: the head of its free list while that is long enough, the freed pages merged in
        # and sorted only when it is short. Taken as the lowest of both, a 131,072-token prompt after the launch's
        # shorter ones was counted up to page 131,072 and given 114,690 to 245,761: a chunk of 65,536 rows wrote up
        # to 180,225 past the card's 131,072 and ran at 1,239 tok/s for 13,100 (Qwen3.5, a first launch, 11-11)
        head = free[:k].long()
        head = head[head <= pages]
        page = int(head.max()) if len(head) else 0
        return min(i["rows"], (page + 1) * p if i["rows"] > pages + 1 else page + 1) if page else 0
    if rel is not None and len(rel):
        free = torch.cat((free, rel))
    free = free.long()
    free = free[free <= pages]
    page = int(torch.kthvalue(free, min(k, len(free))).values) if len(free) else 0
    if len(free) < k:
        # its own pages first, to the last of them, and only then the lend's lowest for what it is short of. Taken as
        # one sorted list with the lend's, a 131,072-token prompt with 100,537 pages on the list was counted inside
        # the card's 212,992 rows and its first chunk wrote up to page 227,143 (69 s for 20, 11-10)
        try:
            from sglang.srt.layers.moe import kt_kv_prefill_lend

            lent = kt_kv_prefill_lend.STATE.get("lent")
            if kt_kv_prefill_lend.STATE.get("alloc") is a and lent is not None and len(lent):
                lent = lent.long()
                lent = lent[lent <= pages]
                if len(lent):
                    page = max(page, int(torch.kthvalue(lent, min(k - len(free), len(lent))).values))
        except Exception as e:
            _say_once("kv-vmm: the lent pages were left out of the highest row a prefill writes", e)
    if not page:
        return 0
    return min(i["rows"], (page + 1) * p if i["rows"] > pages + 1 else page + 1)


def _rows_are_pages(t) -> bool:
    """A tensor with a row a page of the pool's allocator (GLM-5.3's (4097, 8448) for 4,096 pages of 256 tokens):
    the allocator's page numbers are its rows, and a new prompt's rows are where _reach finds them. Counted as rows
    in use plus the prompt's, they stayed at the low pages while the hot lend held those and the allocator handed
    out pages 2,497 and up, past the card's 2,482 (GLM-5.3, 3 extends over host rows a launch, 11-11)."""
    a = _pool_allocator()
    if a is None:
        return False
    p = int(getattr(a, "page_size", 1) or 1)
    return p > 1 and _info(t)["rows"] <= int(a.size) // p + 1


def _target(t, extra: int = 0) -> int:
    if _tokens_a_row(t) > 1 and not _rows_are_pages(t):  # a row of several tokens: pages do not name its rows
        return min(_info(t)["rows"], _rows_in_use(t) + -(-(HEADROOM + extra) // _tokens_a_row(t)))
    top = FLIGHT.get("now")
    if top is None:  # no request seen yet: what the allocator has handed out
        return min(_info(t)["rows"], max(_rows_in_use(t), _reach(t, HEADROOM + extra)))
    if top < 0 and not extra:
        # no request in flight (or each one's rows in spots) and no prompt planned: the rows of prompts kept for
        # their prefix are level 5, after the hot slots. Counted as held, a finished 131K prompt's rows kept the
        # card and the hot lend never lent again: 392 slots out at its window's start, none back (Flash-Next NVFP4
        # on one card, 11-12)
        return 0
    return min(_info(t)["rows"], max(min(_rows_in_use(t), top // _tokens_a_row(t) + 2) if top >= 0 else 0,
                                     _reach(t, HEADROOM + extra)))


def _room_now(index: int) -> int:
    """Free on the card now less the margin: the room for a prompt's rows the plan has already priced against its
    window (host_room's window reserve would count that window twice)."""
    torch.cuda.empty_cache()
    return card_free(index) - margin()


def _moves(t, rows: int) -> int:
    """Bytes that come to the card when t's blocks hold their first rows there: whole chunks, from the one holding a
    block's first new row (already on the card unless the tensor has no card rows yet) to the one holding its
    last."""
    i = _info(t)
    if rows <= i["dev"]:
        return 0
    c = chunk_of(t)
    if c <= 0:
        return (rows - i["dev"]) * i["row"] * i["outer"]
    n = 0
    for o in range(i["outer"]):
        base = o * i["block"]
        start, end = base + i["dev"] * i["row"], base + rows * i["row"]
        first = start // c if i["dev"] == 0 else -(-start // c)
        n += max(0, -(-end // c) - first)
    return n * c


def host_in_flight(index: int, why: str, extra: int = 0, limit=None, window: bool = True, reserve: int = 0,
                   loc=None) -> int:
    """Level 2: every pool tensor's card rows cover the rows the requests hold, plus HEADROOM tokens. When the room
    is short and hot slots are lent, every rank's hot lend first gives back the slots that cover the shortage, the
    same count on each (its next tick lends again with what is left). Returns the bytes moved to the card.
    Level 3 (extra, limit: kt_prefill_plan.chunk_for, a new prompt's plan): the rows held plus the prompt's extra
    tokens, within limit bytes (what the plan's chunks leave of the window's free memory). A prompt's prefill writes
    its rows and its attention reads them, past the card's share in host memory: Qwen3.5's 131K prompt after a
    decode, 43,195 tokens on the card, 192 s; the same prompt again with 139,267 there, 13 s (11-09)."""
    ts = [t for t in HOST_MADE if t.device.index == index]
    if not ts:
        return 0
    if extra:
        torch.cuda.synchronize(index)  # the scheduler's thread: a forward may still read the rows that move
    # in the chunks that move (_moves), the unit the card gives and takes: counted in rows, the wanted bytes fell
    # short of the moved ones and the tensors filled one after another left the last without room (Qwen3.5 131K:
    # 0.86 GiB wanted, 0.88 moved, cuda:0 at its first 67,263 tokens with cuda:1 at 139,266, 11-10)
    _flight_refresh(index)
    need = sum(_moves(t, _target(t, extra)) for t in ts)
    # spots: the chunks of the rows themselves (loc: a forward's own, a planned prompt's next pages; else those of
    # the requests in flight) when they cost less than the card's first rows up to them. With idle prompts cached on
    # the low pages the allocator gave a 658-token scoring forward pages near 182,609: 4.50 GiB of first rows
    # wanted against 0.13 of room, 26 forwards a check wrote their rows in host memory, and the slots went out for
    # rows no request read (Flash-Next NVFP4 on one card after a 131K prompt, 11-12). Its own chunks are 0.19 GiB.
    # The ranks take the same steps either way: the exchange below is the same
    spots = None
    if all(_spot_able(t) for t in ts):
        held = _flight_locs()  # with a forward's own rows, those its request reads
        if loc is None:
            loc = held
        elif held is not None:
            loc = torch.cat((loc.long().to(held.device), held))
        if loc is not None and loc.numel():
            SPOT_GEN[0] += 1
            w = _spot_want(index, loc, SPOT_GEN[0])
            cost = sum(len(k) * c for _, k, c in w)
            if FLIGHT.get("ahead"):  # the spots of the pages a planned prompt has yet to write stay (host_yield)
                a = _next_locs(int(FLIGHT["ahead"]))
                if a is not None:
                    _spot_want(index, a, SPOT_GEN[0])
            # (no first rows wanted with rows of this forward off the card: its request's last token is in a spot
            # an earlier forward brought, the rest are not; a 615-token scoring forward wrote there, Gemma 4, 11-12)
            if cost < need or (w and not need):
                spots, need = w, cost
    if limit is not None:
        need = min(need, max(0, int(limit)))
    def room_of():
        return (host_room(index) if window else _room_now(index)) - int(reserve)

    room = room_of()  # every rank on to the count below: no return before it
    try:
        from sglang.srt.layers.moe import kt_kv_hot_lend as h
    except ImportError:
        h = None
    if h is not None and h.LAYERS:
        from sglang.srt.layers.moe import kt_kv_prefill_lend as kpl

        partial = bool(getattr(kpl, "PARTIAL", False))
        while True:
            # the shortage in whole slots, the hot lend's own unit, and the largest count over the ranks: each
            # rank's slots give as many and no more, the rest stay lent and filled. The whole lend back at every
            # request cost a refill of every slot (GGUF hot slots, 2,600 a rank: 0.7 s out and 5-6 s in, 11-10).
            # One count is all the ranks exchange; counted in bytes each rank measured its own card (Qwen3.5 131K:
            # 0.86 GiB on one, 0.94 on the other) and a shortfall after the slots went took the whole lend back
            lent = int(h.STATE.get("lent_slots", 0) or 0)
            n = 0
            if need > 0 and need > room and lent:  # (nothing wanted with the room below the margin: 574 slots
                # went out and 567 came back at a launch's first decode, 1.2 s, Gemma 4, 11-12)
                n = min(lent, h.slots_for(need - room)) if partial else lent
            v = [n, -lent]  # the largest count wanted, the fewest slots a rank has lent
            if h._world() > 1:
                v = h._over_ranks(v, torch.distributed.ReduceOp.MAX)
            if not v[0]:
                break
            if v[0] >= -v[1]:
                # every slot goes: through the prefill lend when it holds the slots' KV pages (it maps them back
                # and clears its lend), else the hot lend's next tick sees a lend out and never lends again
                # (step3bQwen: decode 48 tok/s without slots)
                if kpl.STATE.get("lent") is not None and kpl.STATE.get("owner") == "hot":
                    kpl.hot_release("host KV rows in flight")
                else:
                    h.release("host KV rows in flight")
                room = room_of()
                break
            h.shrink(0, "host KV rows in flight", slots=v[0])
            room = room_of()
    if limit is not None:
        room = min(room, int(limit))
    if spots is not None:
        if need > room:  # the first rows no request reads give way (level 5), and the spots of earlier forwards
            host_yield(index, need - room, why="rows in flight", held_too=False)
            room = room_of() if limit is None else min(room_of(), int(limit))
        moved = _spot_place(index, spots, SPOT_GEN[0], int(loc.numel()), why, room)
        return moved
    # every tensor to its target, or with the room short to one shared token count (host_fill: a forward reads
    # every layer's rows of a token, the tensor with the fewest decides)
    want = {t.data_ptr(): _target(t, extra) for t in ts}

    def rows_at(t, tokens):
        return min(want[t.data_ptr()], max(_info(t)["dev"], tokens // _tokens_a_row(t)))

    lo, hi = 0, max(want[t.data_ptr()] * _tokens_a_row(t) for t in ts)
    if sum(_moves(t, want[t.data_ptr()]) for t in ts) <= room:
        lo = hi
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if sum(_moves(t, rows_at(t, mid)) for t in ts) <= room:
            lo = mid
        else:
            hi = mid - 1
    moved = 0
    for t in ts:
        rows = rows_at(t, lo)
        if rows <= _info(t)["dev"]:
            continue
        n = _set_rows(t, rows)
        if n < 0:
            raise torch.OutOfMemoryError(f"kv_vmm: placing the pool's rows in flight on cuda:{index}")
        moved += n
    torch.cuda.synchronize(index)
    if moved or need:
        print(f"[kt-host-kv] cuda:{index} {why}: {moved / (1 << 30):.2f} GiB of rows in flight to the card "
              f"({need / (1 << 30):.2f} GiB wanted, room {room / (1 << 30):.2f} GiB), the card holds the first "
              f"{_card_tokens(index)} tokens", flush=True)
    return moved


def prompt_rows_need(tokens: int, prefix=None) -> int:
    """Bytes a new prompt of this many tokens needs moved to the card: every pool tensor's rows up to those the
    requests hold, the prompt and HEADROOM (whole chunks, the edges included). 0 without host KV."""
    if HOST is None or not HOST.get("go") or not HOST_MADE:
        return 0
    index = torch.cuda.current_device()
    FLIGHT["plan"] = torch.as_tensor(prefix) if prefix is not None and len(prefix) else None
    FLIGHT["idle"] = 0
    _flight_refresh(index)
    need = 0
    ts = [t for t in HOST_MADE if t.device.index == index]
    for t in ts:
        need += _moves(t, _target(t, int(tokens)))
    if need and all(_spot_able(t) for t in ts):  # or the chunks of the pages it will be given (host_in_flight)
        loc = _next_locs(int(tokens))
        if loc is not None:
            held = _flight_locs()
            if held is not None:
                loc = torch.cat((loc.long().to(held.device), held))
            spot = idle = 0
            for t in ts:
                want = set(_spot_chunks(t, loc))
                have = SPOTS.get(t.data_ptr(), {})
                spot += chunk_of(t) * sum(1 for k in want if k not in have)
                idle += chunk_of(t) * sum(1 for k in have if k not in want)
                i = _info(t)  # and the first rows no request reads
                idle += max(0, (i["dev"] - _target(t)) * i["row"] * i["outer"] - 2 * chunk_of(t) * i["outer"])
            if spot < need:
                FLIGHT["idle"] = idle
                return spot
    return need


def prompt_rows_bound(tokens: int) -> int:
    """The most bytes the rows of this many tokens (and HEADROOM) bring to the card: every pool tensor's rows for
    them in whole chunks, an edge chunk at each end of a block. From the tensors' shapes: the same on every rank."""
    if HOST is None or not HOST.get("go") or not HOST_MADE:
        return 0
    index = torch.cuda.current_device()
    n = 0
    for t in HOST_MADE:
        if t.device.index != index:
            continue
        i = _info(t)
        rows = min(i["rows"], -(-(int(tokens) + HEADROOM) // _tokens_a_row(t)) + 1)
        n += (rows * i["row"] + 2 * max(0, chunk_of(t))) * i["outer"]
    return n


def prompt_rows_idle() -> int:
    """Bytes the card gets back when the prompt prompt_rows_need just counted takes its rows as spots: the spots and
    first rows of prompts no request reads (host_in_flight sends them out before the hot slots). They were in no
    one's room: a 131K prompt behind two cached 35K ones was given 585 MiB for rows, 1,536 MiB of theirs went out
    as it took them, and 311 of its 432 chunks stayed in host memory (Flash-Next NVFP4 on one card, 11-12)."""
    return int(FLIGHT.get("idle", 0) or 0)


def prompt_rows_room() -> int:
    """Bytes of a new prompt's rows the card takes as it stands in decode: what it has free past the margin and what
    its hot slots hold (host_in_flight sends them out for the rows). The plan sets no more aside: with the weights
    resident a 16 GB card took 0.19 GiB of the 3.00 GiB a 131K prompt's plan had taken out of its window's room for
    them, and the window ran 2,048-row chunks in what was left (Flash-Next NVFP4, TP 1, 11-12)."""
    room = _room_now(torch.cuda.current_device())
    try:
        from sglang.srt.layers.moe import kt_kv_hot_lend as h

        if h.LAYERS and h.STATE.get("lent_slots"):
            room += int(h.standing_bytes())
    except ImportError:
        pass
    return max(0, int(room))


def prompt_rows(tokens: int, limit: int, prefix=None) -> int:
    """kt_prefill_plan.chunk_for, a new prompt planned: its rows to the card within limit bytes, what the plan took
    out of its room for them (host_in_flight, level 3; the window's own needs are in the plan's chunk). Returns the
    bytes moved."""
    if HOST is None or not HOST.get("go") or not HOST_MADE:
        return 0
    FLIGHT["plan"] = torch.as_tensor(prefix) if prefix is not None and len(prefix) else None
    FLIGHT["ahead"] = int(tokens)  # the tokens this prompt still writes (maybe_place counts its extends down)
    return host_in_flight(torch.cuda.current_device(), "a prompt planned", extra=int(tokens),
                          limit=max(0, int(limit)), window=False, loc=_next_locs(int(tokens)))


def host_fill(index: int, floor: int = 0) -> int:
    """Level 5: more of the pool's rows within the room (at least floor left in decode), every tensor up to one
    shared token count: a prefill reads every layer's rows for a token, so the tensor with the fewest decides.
    Filling the tensors one after another left Qwen3.5's largest at 9,920 tokens where the shared count gives
    49,324 (step3bQwen, 11-06). Returns the bytes moved to the card."""
    ts = [t for t in HOST_MADE if t.device.index == index]
    if not ts:
        return 0
    room = host_room(index, floor)
    try:
        # the slots a shrink sent out to come back at the next tick: what they freed waits for them (the encoder's
        # 827 MiB went to the rows at the first decode after an image, and the slots stayed out: 11-11)
        from sglang.srt.layers.moe import kt_kv_hot_lend

        room -= kt_kv_hot_lend.owed_bytes()
    except ImportError:
        pass
    if room <= 0:
        return 0

    def rows_at(t, tokens):
        i = _info(t)
        return min(i["rows"], max(i["dev"], tokens // _tokens_a_row(t)))

    # whole chunks move, a block's rows ending mid-chunk bring that chunk along. Counted
    # chunk by chunk: those from the one holding the block's first new row (already on the card unless the tensor
    # has no card rows yet, when its earlier rows brought it) to the one holding its last. Rows alone fell short
    # and the fill ran the card out (a 1M pool, 11-08: 5.62 GiB placed, 0.01 GiB left); two chunks a block
    # on top kept a 64K pool to its first 44,747 of 65,664 tokens, and a second 35K prompt's
    # attention read the rest in host memory (29 s -> 103 s)
    def cost(tokens):
        return sum(_moves(t, rows_at(t, tokens)) for t in ts)

    lo, hi = 0, max(_info(t)["rows"] * _tokens_a_row(t) for t in ts)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if cost(mid) <= room:
            lo = mid
        else:
            hi = mid - 1
    est = cost(lo)
    moved = 0
    for t in ts:
        rows = rows_at(t, lo)
        if rows <= _info(t)["dev"]:
            continue
        n = _set_rows(t, rows)
        if n < 0:
            raise torch.OutOfMemoryError(f"kv_vmm: placing the pool's rows on cuda:{index}")
        moved += n
    torch.cuda.synchronize(index)
    if os.environ.get("KT_HOST_KV_DEBUG") == "1":
        print(f"[kt-host-kv] cuda:{index}: fill room {room / (1 << 30):.2f} GiB (floor {floor >> 20} MiB), "
              f"estimated {est / (1 << 30):.2f} GiB, moved {moved / (1 << 30):.2f} GiB, "
              f"{torch.cuda.mem_get_info(index)[0] / (1 << 30):.2f} GiB free after", flush=True)
    if moved:
        print(f"[kt-host-kv] cuda:{index}: {moved / (1 << 30):.2f} GiB more of the pool's rows to the card, "
              f"the first {_card_tokens(index)} tokens now", flush=True)
    return moved


# Spots: chunks of the pool on the card past its first rows, for a forward whose own rows the first rows cannot reach
# (kv_vmm_place_range moves any chunk; the first-rows count of a tensor stays what it was). Per tensor, chunk index
# -> the generation of the forward that asked for it.
SPOTS = {}
SPOT_GEN = [0]


def _spot_able(t) -> bool:
    """A token's row in t is known from its page: a row a token, or a row a page of the allocator."""
    return chunk_of(t) > 0 and (_tokens_a_row(t) == 1 or _rows_are_pages(t))


def _prefix_chunk(t, k: int, c: int) -> bool:
    """Chunk k of t holds some block's first rows on the card."""
    i = _info(t)
    if i["dev"] <= 0:
        return False
    lo, hi = k * c, (k + 1) * c
    for o in {min(i["outer"] - 1, lo // i["block"]), min(i["outer"] - 1, (hi - 1) // i["block"])}:
        base = o * i["block"]
        if lo < base + i["dev"] * i["row"] and hi > base:
            return True
    return False


def _spot_chunks(t, loc) -> list:
    """The chunks of t that hold these tokens' rows in every block, those of the card's first rows left out."""
    if loc is None or not loc.numel() or not _spot_able(t):
        return []
    i, c = _info(t), chunk_of(t)
    rows = torch.unique(loc.long() // _tokens_a_row(t))
    rows = rows[(rows >= i["dev"]) & (rows < i["rows"])]
    if not len(rows):
        return []
    start = rows * i["row"]
    ks = torch.unique(torch.cat((start // c, (start + i["row"] - 1) // c)))  # within a block, from its base
    out = set()
    lo, hi = int(start.min()), int(start.max()) + i["row"]
    if (hi - lo) // c <= len(ks):  # one run of chunks: its ends for each block
        for o in range(i["outer"]):
            base = o * i["block"]
            out.update(range((base + lo) // c, (base + hi - 1) // c + 1))
    else:
        for o in range(i["outer"]):
            b = o * i["block"] + start
            out.update(int(k) for k in torch.unique(torch.cat((b // c, (b + i["row"] - 1) // c))))
    return sorted(k for k in out if not _prefix_chunk(t, k, c))


def _spot_want(index: int, loc, gen: int) -> list:
    """(tensor, chunks to bring, chunk bytes) for these tokens; the spots already there are kept for this forward."""
    out = []
    for t in (t for t in HOST_MADE if t.device.index == index):
        s = SPOTS.setdefault(t.data_ptr(), {})
        new = []
        for k in _spot_chunks(t, loc):
            if k in s:
                s[k] = gen
            else:
                new.append(k)
        if new:
            out.append((t, new, chunk_of(t)))
    return out


def _spot_place(index: int, spots: list, gen: int, tokens: int, why: str, room: int) -> int:
    lib = _load()
    torch.cuda.synchronize(index)
    moved = left = 0
    for t, ks, c in spots:
        s = SPOTS.setdefault(t.data_ptr(), {})
        for k in ks:
            if room - moved < c:
                left += 1
                continue
            n = lib.kv_vmm_place_range(ctypes.c_void_p(t.data_ptr()), ctypes.c_size_t(k * c), ctypes.c_size_t(c), 1)
            if n < 0:
                left += 1
                continue
            s[k] = gen
            moved += n
    torch.cuda.synchronize(index)
    print(f"[kt-host-kv] cuda:{index} {why}: spots, {moved / (1 << 20):.0f} MiB of the pool to the card for "
          f"{tokens} tokens past its first {_card_tokens(index)}"
          + (f", {left} chunks left in host memory (room {room / (1 << 20):.0f} MiB)" if left else "")
          + f"; {sum(len(v) for v in SPOTS.values())} chunks in spots", flush=True)
    return moved


def _spots_out(index: int, before=None, keep=None) -> int:
    """The spots back to host memory: those of forwards before generation `before`, or all; not those holding the
    rows of the tokens in keep. Returns the bytes."""
    if not any(SPOTS.values()):
        return 0
    lib = _load()
    torch.cuda.synchronize(index)
    got = 0
    for t in (t for t in HOST_MADE if t.device.index == index):
        s = SPOTS.get(t.data_ptr())
        if not s:
            continue
        c = chunk_of(t)
        read = set(_spot_chunks(t, keep)) if keep is not None else ()
        for k in [k for k, g in s.items() if (before is None or g < before) and k not in read]:
            if not _prefix_chunk(t, k, c):
                n = lib.kv_vmm_place_range(ctypes.c_void_p(t.data_ptr()), ctypes.c_size_t(k * c), ctypes.c_size_t(c), 0)
                got += max(0, n)
            del s[k]
    if got:
        torch.cuda.synchronize(index)
        print(f"[kt-host-kv] cuda:{index} spots: {got / (1 << 20):.0f} MiB back to host memory", flush=True)
    return got


def _spot_covered(index: int, loc) -> bool:
    """Every row these tokens write is on the card: in its first rows or in a spot."""
    for t in (t for t in HOST_MADE if t.device.index == index):
        if not _spot_able(t):
            return False
        s = SPOTS.get(t.data_ptr(), {})
        if any(k not in s for k in _spot_chunks(t, loc)):
            return False
    return True


# The requests in flight: req pool index -> (the highest token its rows reach, its length), noted at its extends and
# every 256th decode forward (kt_lend.before_forward); a request whose pool slot is free again is gone.
# plan: the cached prefix a planned prompt will read (its tokens), until the prompt's first forward names its rows.
# ahead: the tokens a planned prompt has yet to write.
FLIGHT = {"pool": None, "top": {}, "n": 0, "now": None, "plan": None, "ahead": 0}


def flight_note(forward_batch) -> None:
    pool = FLIGHT["pool"]
    if pool is None or HOST is None or not HOST.get("go"):
        return
    mode = getattr(forward_batch, "forward_mode", None)
    if mode is None:
        return
    ext = mode.is_extend()
    if ext:
        FLIGHT["plan"] = None
    if not ext:
        if not mode.is_decode():
            return
        FLIGHT["n"] += 1
        if FLIGHT["n"] % 256:
            return
    idx, lens = getattr(forward_batch, "req_pool_indices", None), getattr(forward_batch, "seq_lens", None)
    if idx is None or lens is None:
        return
    r2t = pool.req_to_token
    for i, n in zip(idx.tolist(), lens.tolist()):
        i, n = int(i), min(int(n), r2t.shape[1])
        if i < 0 or n <= 0 or i >= r2t.shape[0]:
            continue
        FLIGHT["top"][i] = (int(r2t[i, :n].max()), n)
        if not ext and any(SPOTS.values()):
            # a request decoding in spots: the chunk its last tokens moved into, while there is room
            index = torch.cuda.current_device()
            w = _spot_want(index, r2t[i, max(0, n - 256):n], SPOT_GEN[0])
            if w:
                _spot_place(index, w, SPOT_GEN[0], 256, "decode in spots", _room_now(index))


def _top_spotted(index: int, top: int) -> bool:
    """The row of this token is on the card in a spot, in every tensor."""
    ts = [t for t in HOST_MADE if t.device.index == index]
    for t in ts:
        if not _spot_able(t):
            return False
        i, c = _info(t), chunk_of(t)
        row = top // _tokens_a_row(t)
        if row < i["dev"]:
            continue
        if (row * i["row"]) // c not in SPOTS.get(t.data_ptr(), {}):
            return False
    return bool(ts)


def _flight_refresh(index: int) -> None:
    """FLIGHT["now"]: the highest token a request in flight holds outside spots; -1 with none, None when unknown."""
    pool = FLIGHT["pool"]
    if pool is None:
        FLIGHT["now"] = None
        return
    free = set(pool.free_slots)
    for i in [i for i in FLIGHT["top"] if i in free]:
        del FLIGHT["top"][i]
    tops = [v[0] for v in FLIGHT["top"].values()]
    if FLIGHT["plan"] is not None and FLIGHT["plan"].numel():
        tops.append(int(FLIGHT["plan"].max()))
    FLIGHT["now"] = max((v for v in tops if not _top_spotted(index, v)), default=-1)


def _flight_locs():
    """The tokens of the requests in flight (their pool rows), None when unknown or none."""
    pool = FLIGHT["pool"]
    if pool is None:
        return None
    locs = [pool.req_to_token[i, :n[1]].long() for i, n in FLIGHT["top"].items()]
    if FLIGHT["plan"] is not None and FLIGHT["plan"].numel():
        locs.append(FLIGHT["plan"].long().to(pool.req_to_token.device))
    return torch.cat(locs) if locs else None


def _next_locs(tokens: int):
    """The tokens the allocator hands out next for a prompt of this many and HEADROOM (_reach's order: the head of
    its free list, its lowest free pages once that is short; the pages a lend holds are not read here: None then)."""
    a = _pool_allocator()
    if a is None or tokens <= 0:
        return None
    p = int(getattr(a, "page_size", 1) or 1)
    pages = int(a.size) // p
    k = -(-(int(tokens) + HEADROOM) // p)
    free = a.free_pages
    if len(free) >= k:
        head = free[:k].long()
    else:
        rel = getattr(a, "release_pages", None)
        if rel is not None and len(rel):
            free = torch.cat((free, rel))
        if len(free) < k:
            return None
        head = torch.sort(free.long()).values[:k]
    head = head[head <= pages]
    if not len(head):
        return None
    if p == 1:
        return head
    return (head[:, None] * p + torch.arange(p, device=head.device)).flatten()
