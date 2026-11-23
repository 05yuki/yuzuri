"""The KV pool's free pages lend their physical memory to a streamed prefill
(KT_PREFILL_KV_LEND=1, with KT_KV_VMM=1 and the prefill lend KT_PREFILL_LEND=1).

The KV always wins:
- at the start of a prefill window (kt_lend.before_forward, after the weights
  leave) the free pages are split: the lowest-numbered ones that the request
  still needs for the rest of its prompt, plus KT_PREFILL_KV_LEND_RESERVE
  tokens, stay in the free list; the others leave it for the window, and every
  2 MiB chunk of a K/V tensor that holds only such pages is unmapped
  (kt_kv_vmm.release), so the driver, and torch, have that memory for the
  prefill's activations;
- at the end of the window (kt_lend.end_window, before the weights come back)
  the chunks are mapped again, zeroed, and the pages rejoin the free list;
- the pool's size never changes: available_size counts the lent pages, and an
  alloc that needs more pages than the free list holds maps everything back
  first (a device sync, then kt_kv_vmm.restore), so nothing the scheduler
  admitted can fail; that path is logged, it means the request's remaining
  prompt was not known.

The request's prompt length comes from the scheduler (note_request, by rid);
the window's forward batch carries the rids and the sequence lengths.

Outside prefill the same free pages can lend their memory to more hot experts
(KT_KV_HOT_LEND=1, kt_kv_hot_lend.py): decode_lend keeps KT_KV_HOT_RESERVE
tokens free and unmaps the rest the same way; restore runs the hot lend's
release (PRE_RESTORE) before it maps anything back, and a window that opens
takes everything back first (hot_release).

The sliding-window pool of a hybrid SWA model lends the other way round
(KT_PREFILL_SWA_LEND=1, with KT_KV_VMM=1): it must hold a whole prefill chunk,
but decode needs only the window. So it is sized for the largest chunk the plan
may take (KT_PREFILL_SWA_TOKENS, set in the KV cache configurator) and, outside
prefill, every free page beyond KT_PREFILL_SWA_LEND_KEEP tokens is unmapped: at
the end of pool setup, at the end of each prefill window, and in decode once
the window's frees have piled up. An allocation that needs more pages than are
mapped maps back the lowest-numbered lent ones first (swa_map), and the plan
counts what a chunk would map (swa_terms).
"""
import contextlib
import os
import time
from typing import Dict, Optional

import torch

from sglang.srt.layers.moe import kt_kv_vmm

ENABLED = os.environ.get("KT_PREFILL_KV_LEND", "0") == "1" and kt_kv_vmm.ENABLED
HOT = os.environ.get("KT_KV_HOT_LEND", "0") == "1" and kt_kv_vmm.ENABLED
PRE_RESTORE = []  # fn(why), run before lent pages are mapped back (the hot lend's slots out)
# fn(why), run when a window opens before the weights are paused (the hot lend's resident
# rows back to experts 0..N-1)
WINDOW_START = []
RESERVE = int(os.environ.get("KT_PREFILL_KV_LEND_RESERVE", "4096"))  # tokens kept free beyond the prompt
CHECK = os.environ.get("KT_PREFILL_KV_LEND_CHECK", "0") == "1"
# KT_PREFILL_KV_LEND_VERIFY=1: every page in use when a window lends (but the
# window's own batch's) must read the same after it, checked by a per-page hash
VERIFY = os.environ.get("KT_PREFILL_KV_LEND_VERIFY", "0") == "1"

PROMPT: Dict[str, int] = {}  # rid -> prompt tokens
STATE = {"alloc": None, "tensors": None, "lent": None, "ranges": [], "bytes": 0, "owner": None,
         "windows": 0, "take_backs": 0, "unknown": 0, "hot_lends": 0}
# A window that opens while the hot lend holds free pages takes them over unmapped (KT_KV_HOT_HANDOFF=1): its slots
# go, the pages rejoin the free list as they are, and lend() maps back only those the window keeps. Mapping them all
# back and unmapping most of them again a moment later cost MiMo-V2.6 0.7 s a window (5.5 GB each way, 11-09).
# Between the two the pages are HELD; the memory they leave free is not the window's until lend() has decided, so
# the window start's measurements leave it out (held_bytes).
HANDOFF = os.environ.get("KT_KV_HOT_HANDOFF", "1") == "1"
HELD = {"pages": None, "ranges": [], "bytes": 0}
# An allocation that needs pages the hot lend holds takes back those pages only (KT_KV_HOT_PARTIAL=1): the K/V chunks
# under the lowest-numbered lent pages are mapped again, after the hot lend gave up slots worth their bytes
# (PARTIAL_SHRINK: fn(bytes, why)), and the rest stays lent. The cost follows the pages taken; giving back the whole
# lend and lending it again a forward later cost 1.6-2.0 s a time on MiMo-V2.6 (1,130 slots, 11-09), which is what
# KT_KV_HOT_RESERVE's tokens kept free were there to make rare. decode_lend_more is the other direction: pages
# freed while the lend stands join it.
PARTIAL = os.environ.get("KT_KV_HOT_PARTIAL", "1") == "1"
PARTIAL_SHRINK = []


def held_bytes() -> int:
    """The bytes the hot lend handed to the opening window that are still unmapped and undecided."""
    return int(HELD["bytes"]) if HELD["pages"] is not None else 0


def _log(msg: str) -> None:
    print(f"[kt-kv-pf-lend {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def note_request(rid: str, prompt_tokens: int) -> None:
    """The scheduler, when a request is queued."""
    if not ENABLED:
        return
    if len(PROMPT) > 4096:
        PROMPT.clear()
    PROMPT[rid] = int(prompt_tokens)


def _pool_tensors(kvp) -> list:
    """A pool's per-layer buffers, its sub-pools' included: K and V (MHA), one KV
    buffer (MLA, DeepSeek V4), and DeepSeek V4's pool per compress ratio and its
    indexer's (slot = token loc // ratio, page = page_size // ratio rows: their
    pages are the full pool's)."""
    out = []
    for name in ("k_buffer", "v_buffer", "kv_buffer", "index_k_with_scale_buffer"):
        v = getattr(kvp, name, None)
        if isinstance(v, (list, tuple)):
            out += [t for t in v if isinstance(t, torch.Tensor)]
    for name in ("kv_pools", "index_pools"):
        v = getattr(kvp, name, None)
        if isinstance(v, dict):
            for sub in v.values():
                out += _pool_tensors(sub)
    return out


def _in_pages(t, n: int, ps: int):
    """(rows a page) when the buffer is laid out by the allocator's pages, else 0:
    k rows a page for the n + 1 pages and up to k + 1 rows of padding, or k rows a page for the n pages and a
    reserve page counted in the buffer's own smaller pages (a power of two of rows up to the page size, k rows a
    page: DeepSeek V4's c4 holds n + 4, its c128 n + 128)."""
    rows = t.shape[0] if t.dim() >= 1 else 0
    if not (t.is_cuda and t.is_contiguous()) or n <= 0:
        return 0
    k = rows // (n + 1)
    if k >= 1 and rows - k * (n + 1) <= k + 1:
        return k
    for k in range(1, max(1, int(ps)) + 1):
        r = rows - k * n
        if r < k:
            break
        if r % k == 0 and (r // k) & (r // k - 1) == 0 and r // k <= max(1, int(ps)):
            return k
    return 0


def _kv_tensors(kvp, n: int, ps: int = 1, what: str = "K/V", extra=()):
    """[(flat uint8 view, bytes a page, chunk, first chunk boundary)] of a pool's
    buffers laid out by the allocator's pages: k rows a page from row 0 for the n + 1
    pages (1 for a row a page, page_size for a row a token, or as many as a compress
    ratio's or an indexer's smaller pages make one), and up to k + 1 rows of padding
    after them, on kv_vmm memory; [] when there are none. A buffer of any other
    length (another page space) is left alone. extra: (tensor, bytes a page) pairs
    known to follow the same pages from byte 0."""
    tensors, other = [(t.view(-1).view(torch.uint8), pb) for t, pb in extra], 0
    for t in _pool_tensors(kvp):
        rows = t.shape[0] if t.dim() >= 1 else 0
        k = _in_pages(t, n, ps)
        if k:
            tensors.append((t.view(-1).view(torch.uint8), t.nbytes // rows * k))
        else:
            other += 1
    if other:
        _log(f"{other} buffers of {kvp.__class__.__name__} not in the allocator's pages: left out")
        # what they look like
        shapes = {}
        for t in _pool_tensors(kvp):
            if not _in_pages(t, n, ps):
                key = (tuple(t.shape), str(t.dtype), t.is_cuda, t.is_contiguous())
                shapes[key] = shapes.get(key, 0) + 1
        _log(f"  pages n + 1 = {n + 1}, page size {ps}; left out: "
             + "; ".join(f"{c} x {k[0]} {k[1]}{'' if k[2] else ' (host)'}{'' if k[3] else ' (strided)'}"
                         for k, c in list(shapes.items())[:8]))
    if not tensors:
        _log(f"no {what} tensors with whole pages in {kvp.__class__.__name__}: off")
        return []
    # each tensor's own chunk (large K/V buffers use KV_VMM_CHUNK_MB, a small
    # pool's the device granularity). Chunks sit at multiples of their size (each
    # range is reserved aligned to it), but torch packs small buffers into one
    # segment, so a tensor may start inside one: its first whole chunk begins c0
    # bytes in, and only whole chunks inside the tensor are ever unmapped
    tensors = [(t, pb, kt_kv_vmm.chunk_of(t)) for t, pb in tensors]
    off = [t for t, _, c in tensors if not c]
    if off:
        # made outside the KV region (GLM-5.3's DSA indexer buffer): their pages stay
        # mapped, unused while lent; the others still lend
        _log(f"{len(off)} {what} tensors not on kv_vmm memory ({sum(t.nbytes for t in off) >> 20} MiB): "
             "left mapped")
    tensors = [(t, pb, c, (c - t.data_ptr() % c) % c) for t, pb, c in tensors if c]
    if not tensors:
        _log(f"no {what} tensors on kv_vmm memory: off")
    return tensors


def _whole_chunks(tensors, mark, n: int, dev):
    """Per tensor, the runs of chunks every page of which is marked: [(ti, off, len)]."""
    csum = torch.cat((torch.zeros(1, dtype=torch.int64, device=dev), mark.cumsum(0)))
    runs = []
    for ti, (flat, pb, C, c0) in enumerate(tensors):
        nch = max(0, flat.numel() - c0) // C
        j = torch.arange(nch, device=dev, dtype=torch.int64)
        p0 = (c0 + j * C) // pb
        p1 = (c0 + (j + 1) * C - 1) // pb  # inclusive
        # a buffer may run past page n + 1 (a compress state's padding rows):
        # those chunks are never whole
        inside = p1 <= n + 1
        p0, p1 = torch.clamp(p0, max=n + 1), torch.clamp(p1, max=n + 1)
        whole = inside & ((csum[p1 + 1] - csum[p0]) == (p1 - p0 + 1))
        idx = torch.nonzero(whole).flatten().cpu().tolist()
        i = 0
        while i < len(idx):
            k = i
            while k + 1 < len(idx) and idx[k + 1] == idx[k] + 1:
                k += 1
            runs.append((ti, c0 + idx[i] * C, (idx[k] - idx[i] + 1) * C))
            i = k + 1
    return runs


def _setup(runner) -> bool:
    if STATE["alloc"] is not None:
        return not STATE.get("spare_only")
    if STATE.get("off"):
        return False
    STATE["off"] = True  # until set up
    a = runner.token_to_kv_pool_allocator
    # an SWA allocator (V4.1, Vision, MiMo) hands full-attention allocations to its
    # full_attn_allocator: that one's free pages are the full pool's, and wrapping
    # it covers every path the SWA allocator takes into it
    a = getattr(a, "full_attn_allocator", None) or a
    # one free list of pages (or tokens, page size 1) handed out from its front:
    # the paged allocator and the token one (Qwen3.5's, and SWA's full side); the
    # unified and sparse ones keep more than one list and are not handled yet
    if type(a).__name__ not in ("PagedTokenToKVPoolAllocator", "TokenToKVPoolAllocator") \
            or not isinstance(getattr(a, "free_pages", None), torch.Tensor):
        _log(f"allocator {type(a).__name__} has no single free list: off")
        return False
    pool = runner.token_to_kv_pool
    kvp = getattr(pool, "full_kv_pool", None) or pool
    n = int(getattr(a, "num_pages", 0) or a.size // a.page_size)
    tensors = _kv_tensors(kvp, n, a.page_size)
    if not tensors:
        if HOT:
            # nothing to unmap (V4.1's compressed KV in host memory): the hot lend can
            # still take the driver's spare, and gives it back at the same windows
            STATE.update(alloc=a, tensors=[], chunk=0, ps=a.page_size, n=n, off=False, spare_only=True)
            _log("the hot lend takes the driver's spare only")
        return False
    chunk = max(t[2] for t in tensors)
    # pages wholly in every tensor's device part (a split range's tail is host memory)
    dev_pages = min(kt_kv_vmm.device_bytes(t[0]) // t[1] for t in tensors)
    STATE.update(alloc=a, tensors=tensors, chunk=chunk, ps=a.page_size, n=n, off=False,
                 split=dev_pages < n + 1 or bool(getattr(kt_kv_vmm, "HOST_MADE", None)))
    _wrap_allocator(a)
    _log(f"{len(tensors)} K/V tensors of {kvp.__class__.__name__}, {n} pages of {a.page_size} tokens "
         f"({min(t[1] for t in tensors)}-{max(t[1] for t in tensors)} B a page per tensor), "
         f"chunks {min(t[2] for t in tensors) >> 20}-{chunk >> 20} MiB, "
         f"{sum(1 for t in tensors if t[3])} not chunk-aligned, reserve {RESERVE} tokens")
    return True


def _give_back(a, pages) -> None:
    """Lent pages back on the allocator's free list. A host KV pool's allocator (kt_kv_vmm.keep_lowest_first) hands
    out the lowest pages first, the ones whose rows the card holds: its list stays sorted. Appended, the pages a
    window returned went out last, and the next chunks took pages past the card's rows while lower ones were free
    (Qwen3.5: a 44,000-row chunk at pages 87,537-163,588 with the card at 139,266 ran at 604 tok/s, the same
    chunk on the card 2,437; a 65,536-row one 757 against 8,371, 11-10)."""
    free = torch.cat((a.free_pages, pages.to(a.free_pages.dtype)))
    if getattr(a, "_kt_lowest_first", False) and not getattr(a, "need_sort", False):
        free = torch.sort(free).values
    a.free_pages = free


def _free_pages(a) -> torch.Tensor:
    """Every free page of the allocator: the free list, and the pages freed but not
    merged into it yet (release_pages, or the paged allocator's staged_pages)."""
    parts = [a.free_pages]
    if getattr(a, "release_pages", None) is not None and len(a.release_pages):
        parts.append(a.release_pages)
    parts += [p for p in getattr(a, "staged_pages", None) or [] if len(p)]
    return torch.cat(parts) if len(parts) > 1 else parts[0]


def _free_count(a) -> int:
    """len(_free_pages(a)) without the concatenation: what the allocator can hand
    out, merging what it freed when its free list runs short."""
    return (len(a.free_pages) + (len(a.release_pages) if getattr(a, "release_pages", None) is not None else 0)
            + int(getattr(a, "num_staged_pages", 0) or 0))


def _merge_free(a) -> None:
    """The freed pages into the free list, as the allocator does when it runs short:
    pages are taken out of the free list only, so none is left behind elsewhere."""
    merge = getattr(a, "merge_and_sort_free", None)
    if merge is not None:
        merge()


def _remaining(fb) -> Optional[int]:
    """Prompt tokens the window's requests have not reached yet (None: unknown)."""
    rids = getattr(fb, "rids", None)
    seqs = getattr(fb, "seq_lens_cpu", None)
    if not rids or seqs is None:
        return None
    left = 0
    for rid, s in zip(rids, seqs.tolist()):
        p = PROMPT.get(rid)
        if p is None:
            return None
        left += max(0, p - int(s))
    return left


def _runner_of(scheduler):
    w = getattr(scheduler, "tp_worker", None)
    for path in (("model_runner",), ("_model_runner",), ("worker", "model_runner"), ("worker", "_model_runner")):
        o = w
        for p in path:
            o = getattr(o, p, None)
        if o is not None:
            return o
    return None


def _dev_pages():
    """The pages wholly on the card in every K/V tensor, None when the pool is all there. A host KV pool's rows move
    to the card as the memory allows, so it is read at each use: read once at the first window, it stayed at the
    rows of that moment, and Qwen3.5's 262K pool with 139,266 tokens on the card lent nothing (35K in two chunks,
    11-09). The smallest over the TP ranks: a lend takes pages out of the allocator's free list, which every rank
    keeps the same."""
    if not STATE.get("split") or not STATE.get("tensors"):
        return None
    dp = min(kt_kv_vmm.device_bytes(t[0]) // t[1] for t in STATE["tensors"])
    from sglang.srt.layers.moe import kt_kv_hot_lend as h

    if h._world() > 1:
        dp = h._over_ranks([dp])[0]
    return dp if dp < STATE["n"] + 1 else None


def window_back(rem: int) -> int:
    """Bytes a new prompt of rem tokens maps back in the window that is open: a window's lend goes back whole when an
    allocation runs past the pages it kept (it kept them for the prompt that opened it), or in part
    (_take_back). The plan takes them out of the room first: a second 35,000-token prompt planned its chunk on the 5.23 GB that were lent, its first
    allocation mapped them back, and the forward ran out of memory (Flash-Next GGUF, one card, no host KV, 11-17)."""
    if STATE.get("lent") is None or STATE.get("owner") == "hot" or STATE.get("alloc") is None:
        return 0
    held = int(HELD["pages"].numel()) if HELD["pages"] is not None else 0
    short = -(-(int(rem) + RESERVE) // STATE["ps"]) + 1 - (_free_count(STATE["alloc"]) - held)
    if short <= 0:
        return 0
    if not PARTIAL or not STATE.get("tensors"):
        return int(STATE["bytes"])
    # the pages it is short of and a band at each end (_take_back), and the part chunks at a run's ends
    band = max(C // pb for _, pb, C, _ in STATE["tensors"])
    per_page = sum(pb for _, pb, _, _ in STATE["tensors"])
    return min(int(STATE["bytes"]), (short + 2 * band) * per_page + sum(2 * C for _, _, C, _ in STATE["tensors"]))


def loan_estimate(scheduler, rem: int) -> int:
    """Bytes a window for a request with rem prompt tokens to go could borrow
    (kt_prefill_plan). Only the allocator's free page count enters it, which is
    the same on every TP rank."""
    if not ENABLED:
        return 0
    if STATE["alloc"] is None:
        runner = _runner_of(scheduler)
        if runner is None or not _setup(runner):
            return 0
    a, ps = STATE["alloc"], STATE["ps"]
    dp = _dev_pages()
    if dp is None:
        free = _free_count(a)
        if STATE["lent"] is not None:
            free += int(STATE["lent"].numel())
    else:
        # a pool whose tail is host memory: the window lends the free
        # pages below dev_pages only, so only those count
        free = int((_free_pages(a) < dp).sum())
        if STATE["lent"] is not None:
            free += int((STATE["lent"] < dp).sum())
    keep = -(-(rem + RESERVE) // ps) + 1
    pages = max(0, free - keep)
    # only whole chunks are unmapped: a tensor loses up to one chunk at the
    # boundary of the pages it keeps
    return sum(max(0, pages * pb - 2 * c) for _, pb, c, _ in STATE["tensors"])


def _chunk_runs(ti: int, idx) -> list:
    """[(ti, off, len)] for the sorted chunk indices idx of tensor ti, one run of consecutive chunks each."""
    _, _, C, c0 = STATE["tensors"][ti]
    runs, i = [], 0
    while i < len(idx):
        k = i
        while k + 1 < len(idx) and idx[k + 1] == idx[k] + 1:
            k += 1
        runs.append((ti, c0 + idx[i] * C, (idx[k] - idx[i] + 1) * C))
        i = k + 1
    return runs


def _chunks_of(ranges) -> Dict[int, set]:
    """{tensor: the chunk indices} of [(ti, off, len)] runs."""
    out: Dict[int, set] = {}
    for ti, off, ln in ranges:
        _, _, C, c0 = STATE["tensors"][ti]
        out.setdefault(ti, set()).update(range((off - c0) // C, (off - c0 + ln) // C))
    return out


def _hand_off(why: str) -> None:
    """hot_release at a window's start: the slots go, the hot lend's pages rejoin the free list unmapped (HELD)."""
    t0 = time.perf_counter()
    a = STATE["alloc"]
    dev = a.free_pages.device
    for fn in PRE_RESTORE:
        fn(why)
    torch.cuda.synchronize(dev)
    torch.cuda.empty_cache()
    HELD.update(pages=STATE["lent"], ranges=list(STATE["ranges"]), bytes=STATE["bytes"])
    _give_back(a, STATE["lent"])
    if STATE["hot_lends"] in (1, 2, 10, 100) or CHECK:
        _log(f"hot lend {STATE['hot_lends']}, {why}: {STATE['bytes'] / 1e9:.2f} GB handed to the window unmapped, "
             f"{STATE['lent'].numel()} pages returned in {(time.perf_counter() - t0) * 1e3:.1f} ms")
    STATE.update(lent=None, ranges=[], bytes=0, owner=None)


def _hand_back() -> None:
    """restore at a window's end: the window's lent pages rejoin the free list unmapped (HELD) for the hot lend
    of the next decode step, which keeps them lent; whatever it does not take is mapped back then."""
    t0 = time.perf_counter()
    a = STATE["alloc"]
    torch.cuda.synchronize(a.free_pages.device)
    HELD.update(pages=STATE["lent"], ranges=list(STATE["ranges"]), bytes=STATE["bytes"])
    _give_back(a, STATE["lent"])
    if STATE["windows"] in (1, 2, 10, 100) or CHECK:
        _log(f"end of window: {STATE['bytes'] / 1e9:.2f} GB handed to the hot lend unmapped, "
             f"{STATE['lent'].numel()} pages returned in {(time.perf_counter() - t0) * 1e3:.1f} ms")
    STATE.update(lent=None, ranges=[], bytes=0, owner=None)


def _settle_held(keep=(), why: str = "") -> int:
    """Map back the chunks the hot lend handed over that the window does not keep lent (keep: its runs).
    Returns the bytes left unmapped (the kept part)."""
    if HELD["pages"] is None:
        return 0
    t0 = time.perf_counter()
    kept = _chunks_of(keep)
    back, left = 0, 0
    for ti, idx in sorted(_chunks_of(HELD["ranges"]).items()):
        stay = kept.get(ti, set())
        go = sorted(i for i in idx if i not in stay)
        flat, _, C, _ = STATE["tensors"][ti]
        left += (len(idx) - len(go)) * C
        for _, off, ln in _chunk_runs(ti, go):
            got = kt_kv_vmm.restore(flat, off, ln)
            if got < 0:
                dev = STATE["alloc"].free_pages.device
                raise RuntimeError(
                    f"KT_KV_HOT_HANDOFF: {ln >> 20} MiB of KV tensor {ti} cannot be mapped back on {dev} "
                    f"({torch.cuda.mem_get_info(dev)[0] >> 20} MiB free)")
            back += got
    HELD.update(pages=None, ranges=[], bytes=0)
    if why and (STATE["windows"] in (0, 1, 2, 10, 100) or CHECK):
        _log(f"handed-over pages, {why}: {back / 1e9:.2f} GB mapped back, {left / 1e9:.2f} GB stay lent, "
             f"in {(time.perf_counter() - t0) * 1e3:.1f} ms")
    return left


def _release_free(keep_pages: int, need_pages: Optional[int] = None):
    """Take free pages but the keep_pages lowest-numbered out of the free list and unmap each chunk of a K/V tensor
    that holds only such pages (a device sync first). need_pages: how many the window needs, the highest-numbered
    taken (None: every one past keep_pages). Pages the hot lend handed over unmapped are taken first and all of them,
    what the window does not keep is mapped back. (lent pages, mark [n + 2] of them, [(ti, off, len)] unmapped,
    bytes)."""
    a, n = STATE["alloc"], STATE["n"]
    _merge_free(a)
    dev = a.free_pages.device
    dp = _dev_pages()  # every rank, before the early returns below (it reduces over the ranks)
    free = torch.sort(_free_pages(a)).values
    if free.numel() <= keep_pages:
        _settle_held(why="nothing past the kept pages")
        return None
    cand = free[keep_pages:]
    if dp is not None:
        # a pool whose tail is host memory: only device pages lend
        cand = cand[cand < dp]
    if HELD["pages"] is not None or need_pages is not None:
        held = torch.zeros(n + 2, dtype=torch.bool, device=dev)
        if HELD["pages"] is not None:
            held[HELD["pages"].long()] = True
        h = held[cand.long()]
        cand_held, rest = cand[h], cand[~h]
        want = rest.numel() if need_pages is None else max(0, need_pages - cand_held.numel())
        cand = torch.cat((cand_held, rest[rest.numel() - min(want, rest.numel()):]))
    if cand.numel() == 0:
        _settle_held(why="nothing to lend")
        return None
    mark = torch.zeros(n + 2, dtype=torch.int32, device=dev)
    mark[cand] = 1
    torch.cuda.synchronize(dev)
    ranges, total = [], 0
    # runs of chunks, one release call each; a run the hot lend left unmapped releases nothing and stays lent
    runs = _whole_chunks(STATE["tensors"], mark, n, dev)
    for ti, off, ln in runs:
        got = kt_kv_vmm.release(STATE["tensors"][ti][0], off, ln)
        if got > 0 or HELD["pages"] is not None:
            ranges.append((ti, off, ln))
            total += max(0, got)
    total += _settle_held(keep=ranges, why="the window took them")
    keep_mask = mark[a.free_pages] == 0
    a.free_pages = a.free_pages[keep_mask]
    if getattr(a, "release_pages", None) is not None and len(a.release_pages):
        a.release_pages = a.release_pages[mark[a.release_pages] == 0]
    return cand, mark, ranges, total


def _spare_only() -> int:
    """A hot lend of the driver's spare memory alone: no page leaves the pool (1: lent)."""
    STATE.update(lent=STATE["alloc"].free_pages[:0].clone(), ranges=[], bytes=0, owner="hot")
    STATE["hot_lends"] += 1
    return 1


def decode_lend(runner, reserve_tokens: int, min_bytes: int) -> int:
    """The hot lend, outside prefill: every free page beyond reserve_tokens (and a
    page a running request) leaves the free list and its whole chunks are unmapped,
    when that is at least min_bytes. Returns the bytes unmapped (0: nothing lent).
    Pages a window handed back unmapped are taken first; when nothing is lent they are mapped back."""
    try:
        return _decode_lend(runner, reserve_tokens, min_bytes)
    finally:
        _settle_held(why="the hot lend did not take them")


def _decode_lend(runner, reserve_tokens: int, min_bytes: int) -> int:
    if not HOT or STATE["lent"] is not None or runner is None:
        return 0
    if not _setup(runner):
        if not STATE.get("spare_only") or min_bytes > 1:
            return 0
        return _spare_only()
    a, ps = STATE["alloc"], STATE["ps"]
    keep = -(-reserve_tokens // ps) + 1
    free = _free_count(a)
    if free <= keep:
        # no page past the reserve (a pool no larger than it): the driver's spare alone when it covers the floor
        # (min_bytes 1); a 64K pool under the 65536-token reserve never lent its card's 7 GB free (Flash-Next, 11-09)
        return _spare_only() if min_bytes <= 1 else 0
    split = STATE.get("split")
    if split:
        dp = _dev_pages()
        if dp is not None:
            free = min(free, dp)  # only device pages lend
    est = sum(max(0, (free - keep) * pb - 2 * c) for _, pb, c, _ in STATE["tensors"])
    if est < min_bytes and not (split and min_bytes <= 1):
        # the pages past the reserve make no whole chunk (kept prefixes hold the rest): the driver's spare alone
        # when it covers the floor
        return _spare_only() if min_bytes <= 1 else 0
    got = _release_free(keep)
    if got is None:
        # no free page to lend (a split pool with no free device page among them): the driver's spare alone
        return _spare_only() if min_bytes <= 1 else 0
    cand, _, ranges, total = got
    if not ranges:
        # free pages with no whole chunk among them (a long prompt holds the rest): they stay in the free list and
        # the driver's spare alone is lent. Kept as a lend of nothing, they left the slots at 0 for as long as the
        # prompt stayed (Flash-Next after 220,000 tokens, 11-10)
        _give_back(a, cand)
        return _spare_only() if min_bytes <= 1 else 0
    STATE.update(lent=cand, ranges=ranges, bytes=total, owner="hot", more_floor=0)
    STATE["hot_lends"] += 1
    if PARTIAL:
        # (no chunk unmapped: every page of it)
        _return_mapped()
    return total


def _return_mapped(unm=None) -> int:
    """The lent pages no unmapped chunk reaches go back to the free list: a page is whole when no tensor has an
    unmapped chunk at or below its bytes. After a lend these are the pages under the first chunk boundary, which
    were never unmapped: the next short request allocates from them with nothing to take back."""
    a = STATE["alloc"]
    if unm is None:
        unm = _chunks_of(STATE["ranges"])
    whole = STATE["n"] + 2
    for ti, idx in unm.items():
        if idx:
            _, pb, C, c0 = STATE["tensors"][ti]
            whole = min(whole, (c0 + min(idx) * C) // pb)
    lent = STATE["lent"]
    low = lent < whole
    ret = lent[low]
    if ret.numel():
        STATE["lent"] = lent[~low]
        _give_back(a, ret)
    return int(ret.numel())


def _take_back(short: int, why: str) -> bool:
    """The hot lend's lowest-numbered pages back to the free list, short of them and the pages one more chunk
    holds: the chunks under them are mapped again, the others stay unmapped. False: the whole lend goes back
    (restore)."""
    # a window's lend too: its memory is the window's free room, which the plan left for it (window_back). Whole, a
    # second prompt's pages mapped all 5.23 GB back and the weights found 3.31 GB of the 3.87 they need at the
    # window's end (Flash-Next GGUF, one card, no host KV, 11-17)
    if (not PARTIAL or STATE.get("owner") not in ("hot", "window") or not STATE.get("tensors")
            or STATE["lent"].numel() < short):
        return False
    t0 = time.perf_counter()
    a = STATE["alloc"]
    hot = STATE.get("owner") == "hot"
    lent = STATE["lent"] = torch.sort(STATE["lent"]).values
    band = max(C // pb for _, pb, C, _ in STATE["tensors"])
    cut = int(lent[min(int(lent.numel()), short + band) - 1])
    unm = _chunks_of(STATE["ranges"])
    go, need = {}, 0
    for ti, idx in unm.items():
        _, pb, C, c0 = STATE["tensors"][ti]
        go[ti] = sorted(i for i in idx if c0 + i * C < (cut + 1) * pb)
        need += len(go[ti]) * C
    if need and STATE.get("owner") == "hot":
        for fn in PARTIAL_SHRINK:
            fn(need, why)
    back, ok = 0, True
    for ti in sorted(go):
        flat = STATE["tensors"][ti][0]
        for _, off, ln in _chunk_runs(ti, go[ti]):
            got = kt_kv_vmm.restore(flat, off, ln)
            if got < 0:
                ok = False
                break
            back += got
            C, c0 = STATE["tensors"][ti][2], STATE["tensors"][ti][3]
            unm[ti].difference_update(range((off - c0) // C, (off - c0 + ln) // C))
        if not ok:
            break
    STATE["ranges"] = [r for ti in sorted(unm) for r in _chunk_runs(ti, sorted(unm[ti]))]
    STATE["bytes"] = max(0, STATE["bytes"] - back)
    if not ok:
        _log(f"partial take back for {why}: mapping {need >> 20} MiB back failed after {back >> 20} MiB: "
             f"the whole lend goes back")
        return False
    ret = _return_mapped(unm)
    dt = (time.perf_counter() - t0) * 1e3
    STATE["partial"] = STATE.get("partial", 0) + 1
    STATE["partial_ms"] = STATE.get("partial_ms", 0.0) + dt
    STATE["partial_max"] = max(STATE.get("partial_max", 0.0), dt)
    if STATE["partial"] in (1, 2, 3, 10, 100, 1000, 10000) or CHECK:
        _log(f"partial take back {STATE['partial']} for {why}: {ret} pages returned "
             f"({short} short), {back / 2**20:.0f} MiB mapped back, {STATE['bytes'] / 2**30:.2f} GiB stay lent, in "
             f"{dt:.1f} ms ({STATE['partial_ms'] / STATE['partial']:.1f} ms a time so far, largest "
             f"{STATE['partial_max']:.1f})")
    # the allocation that asked takes its pages next; the rest of the band is not lent again until a page is freed
    # (a grow 1 s after a take-back unmapped the same 384 MiB, and the next allocation took it back, 11-10)
    if hot:
        STATE["more_floor"] = _free_count(a) - short + 1
    return True


def more_room(reserve_tokens: int, min_bytes: int) -> bool:
    """Whether the free pages past the reserve may be worth a decode_lend_more: page counts only (the same on
    every rank, nothing synchronized)."""
    if not PARTIAL or not HOT or STATE["lent"] is None or STATE.get("owner") != "hot" or not STATE.get("tensors"):
        return False
    a = STATE["alloc"]
    free = _free_count(a)
    over = free - (-(-reserve_tokens // STATE["ps"]) + 1)
    pb_all = STATE.get("pb_all")
    if pb_all is None:
        pb_all = STATE["pb_all"] = sum(pb for _, pb, _, _ in STATE["tensors"])
    if over * pb_all < min_bytes:
        return False
    # a try that found too few whole chunks: not again until as many more pages are free
    return free >= STATE.get("more_floor", 0)


def decode_lend_more(runner, reserve_tokens: int, min_bytes: int) -> int:
    """The hot lend stands and pages were freed since (a request ended): the free pages beyond reserve_tokens
    join it, when the whole chunks they make with the pages already lent are at least min_bytes. Returns the bytes
    newly unmapped (0: nothing changed)."""
    if not more_room(reserve_tokens, min_bytes):
        return 0
    a, n = STATE["alloc"], STATE["n"]
    keep = -(-reserve_tokens // STATE["ps"]) + 1
    _merge_free(a)
    dev = a.free_pages.device
    dp = _dev_pages()  # every rank (it reduces over the ranks)
    free = torch.sort(_free_pages(a)).values
    cand = free[keep:]
    if dp is not None:
        cand = cand[cand < dp]
    new_mark = torch.zeros(n + 2, dtype=torch.int32, device=dev)
    new_mark[cand.long()] = 1
    mark = new_mark.clone()
    mark[STATE["lent"].long()] = 1
    had = _chunks_of(STATE["ranges"])
    now = _chunks_of(_whole_chunks(STATE["tensors"], mark, n, dev))
    add = {ti: sorted(idx - had.get(ti, set())) for ti, idx in now.items()}
    est = sum(len(idx) * STATE["tensors"][ti][2] for ti, idx in add.items())
    if est < min_bytes:
        STATE["more_floor"] = int(free.numel()) + -(-min_bytes // STATE["pb_all"])
        return 0
    torch.cuda.synchronize(dev)
    total = 0
    for ti in sorted(add):
        for _, off, ln in _chunk_runs(ti, add[ti]):
            got = kt_kv_vmm.release(STATE["tensors"][ti][0], off, ln)
            if got > 0:
                had.setdefault(ti, set()).update(
                    range((off - STATE["tensors"][ti][3]) // STATE["tensors"][ti][2],
                          (off - STATE["tensors"][ti][3] + ln) // STATE["tensors"][ti][2]))
                total += got
    a.free_pages = a.free_pages[new_mark[a.free_pages.long()] == 0]
    if getattr(a, "release_pages", None) is not None and len(a.release_pages):
        a.release_pages = a.release_pages[new_mark[a.release_pages.long()] == 0]
    STATE["ranges"] = [r for ti in sorted(had) for r in _chunk_runs(ti, sorted(had[ti]))]
    STATE["lent"] = torch.cat((STATE["lent"], cand.to(STATE["lent"].dtype)))
    STATE["bytes"] += total
    STATE["more_floor"] = 0
    _return_mapped(had)
    return total


def hot_release(why: str) -> None:
    """Give back what the hot lend holds (a window opens, the pool is cleared)."""
    if why == "window start":
        for fn in WINDOW_START:
            fn(why)
    if STATE["lent"] is not None and STATE.get("owner") == "hot":
        # kt_lend calls this at a window's start and lend() right after, in the same step: the pages wait there
        # (a park as well: the pages stay unmapped through it, and the lend after the resume takes them as they are)
        if why in ("window start", "park") and HANDOFF and ENABLED and STATE["ranges"]:
            _hand_off(why)
        else:
            restore(why)


def keep_for(fb) -> bool:
    """A window opens over a hot lend that stays (kt_lend._hot_stays): the pages the window's prompts have yet to
    take, and RESERVE, are in the free list as a window's own loan leaves them, taken back from the hot lend where it
    holds them. False: the prompts' lengths are not known, or the pages could not be mapped back."""
    if STATE["lent"] is None or STATE.get("owner") != "hot":
        return False
    if not STATE.get("tensors"):
        return True  # the driver's spare alone is lent: no page left the pool
    rem = _remaining(fb)
    if rem is None:
        return False
    a = STATE["alloc"]
    short = -(-(rem + RESERVE) // STATE["ps"]) + len(fb.rids) - _free_count(a)
    short = min(short, int(STATE["lent"].numel()))
    return short <= 0 or _take_back(short, "window start")


def lend(runner, fb, hot_stays: bool = False) -> None:
    """kt_lend.before_forward, opening a prefill window, after the weights left."""
    if hot_stays and STATE["lent"] is not None and STATE.get("owner") == "hot":
        # the hot lend stands through this window (kt_lend._hot_stays): the pages the prompts take are back in the
        # free list (keep_for), the chunk's memory comes from the slots (kt_lend._hot_topup), nothing else is lent.
        # Releasing here took every slot at each window's start, a 256-token prompt's too (1.2 s, and decode began
        # on empty slots: 11-11)
        return
    hot_release("window start")
    try:
        _lend(runner, fb)
    finally:
        # every way out: what the hot lend handed over and the window did not take is mapped back
        _settle_held(why="the window did not lend")


def _need_pages(fb) -> Optional[int]:
    """The pages the window's planned loan comes to (kt_prefill_plan.loan_bytes), None for the whole loan: its bytes
    over the bytes a page holds in every tensor, and the two part chunks a tensor loses at the run's ends."""
    from sglang.srt.layers.moe import kt_prefill_plan

    need = getattr(kt_prefill_plan, "loan_bytes", lambda fb: None)(fb)
    if need is None:
        return None
    if need <= 0:
        return 0
    per_page = sum(pb for _, pb, _, _ in STATE["tensors"])
    edges = sum(2 * C for _, _, C, _ in STATE["tensors"])
    return -(-(need + edges) // max(1, per_page))


def _lend(runner, fb) -> None:
    if not ENABLED or STATE["lent"] is not None or runner is None or not _setup(runner):
        return
    from sglang.srt.layers.moe import kt_prefill_plan

    planned = kt_prefill_plan.needs_loan(fb)
    if not planned and HELD["pages"] is None:
        STATE["skipped"] = STATE.get("skipped", 0) + 1
        return
    t0 = time.perf_counter()
    a, ps, n = STATE["alloc"], STATE["ps"], STATE["n"]
    dev = a.free_pages.device
    rem = _remaining(fb)
    if rem is None:
        STATE["unknown"] += 1
        if STATE["unknown"] in (1, 10, 100):
            _log(f"window without a known prompt length ({STATE['unknown']} so far): nothing lent")
        return
    keep_pages = -(-(rem + RESERVE) // ps) + len(fb.rids)
    # a window the plan did not lend for keeps lent only what the hot lend handed over (it costs nothing to keep)
    need_pages = _need_pages(fb) if planned else 0
    # the free pages past the kept ones: the whole loan this window could borrow. kt_lend measures the window's free
    # memory before the loan goes out, so the hint records this beside it (loan_room; tools/serve.py measured_ceiling
    # adds the two), or a ceiling measured from the free memory alone can never exceed the loan-less room: GLM
    # 36864 -> 16384, 35K-2 18.4 -> 37.2 s (11-09)
    past_keep = _free_pages(a)
    dp = _dev_pages()
    if dp is not None:
        past_keep = past_keep[past_keep < dp]
    STATE["loan_room"] = max(0, int(past_keep.numel()) - keep_pages) * sum(pb for _, pb, _, _ in STATE["tensors"])
    got = _release_free(keep_pages, need_pages)
    if got is None:
        return
    cand, mark, ranges, total = got
    if VERIFY:
        # the release above synchronised; the pages hashed are in use, so never
        # in a released chunk
        used = torch.ones(n + 2, dtype=torch.bool, device=dev)
        used[_free_pages(a).long()] = False
        used[cand.long()] = False  # lent (out of the free list already): unmapped
        used[0] = False  # the padding page takes padded tokens' writes
        used[n + 1:] = False
        cur = getattr(fb, "out_cache_loc", None)
        if isinstance(cur, torch.Tensor) and cur.numel():
            used[(cur.long() // ps).clamp(0, n + 1)] = False  # this batch writes them
        STATE["verify"] = (used, _page_hashes(used, n))
    STATE.update(lent=cand, ranges=ranges, bytes=total, owner="window")
    STATE["windows"] += 1
    if STATE["windows"] in (1, 2, 10, 100) or CHECK:
        _log(f"window {STATE['windows']}: {rem} prompt tokens to go, {keep_pages} pages kept, {cand.numel()} lent"
             f"{' (all past them)' if need_pages is None else f' ({need_pages} planned)'}, "
             f"{total / 1e9:.2f} GB unmapped in {len(ranges)} ranges in {(time.perf_counter() - t0) * 1e3:.1f} ms")


def _page_hashes(mask, n: int) -> torch.Tensor:
    """A 64-bit hash of every page of every K/V tensor where mask is set (else 0):
    the int32 words of a page times fixed odd weights, summed, over the tensors."""
    dev = mask.device
    out = torch.zeros(n + 1, dtype=torch.int64, device=dev)
    idx = torch.nonzero(mask[: n + 1]).flatten()
    if not idx.numel():
        return out
    for ti, (flat, pb, C, c0) in enumerate(STATE["tensors"]):
        if pb % 4:
            continue
        pages = flat[: (n + 1) * pb].view(n + 1, pb).view(torch.int32)
        g = torch.Generator(device="cpu").manual_seed(1000 + ti)
        w = (torch.randint(1, 1 << 30, (pages.shape[1],), generator=g, dtype=torch.int64) * 2 + 1).to(dev)
        for i in range(0, idx.numel(), 256):
            sel = idx[i:i + 256]
            out[sel] += (pages[sel].to(torch.int64) * w).sum(dim=1)
    return out


def restore(why: str = "end of window") -> None:
    """kt_lend.end_window before the weights come back (or an alloc that needs the pages)."""
    _settle_held(why=why)
    if STATE["lent"] is None:
        return
    if why == "end of window" and STATE.get("owner") == "hot":
        return  # the hot lend stood through the window (kt_lend._hot_stays)
    if (why == "end of window" and HANDOFF and HOT and STATE.get("owner") == "window" and STATE["ranges"]
            and not VERIFY):
        # the next decode's hot lend takes the same pages (kt_kv_hot_lend.tick): they wait unmapped
        _hand_back()
        return
    t0 = time.perf_counter()
    a = STATE["alloc"]
    dev = a.free_pages.device
    owner = STATE.get("owner")
    if owner == "hot":
        # the slots give their memory back first (and leave the routing)
        for fn in PRE_RESTORE:
            fn(why)
    torch.cuda.synchronize(dev)
    torch.cuda.empty_cache()
    for ti, off, ln in STATE["ranges"]:
        flat = STATE["tensors"][ti][0]
        if kt_kv_vmm.restore(flat, off, ln) < 0:
            raise RuntimeError(
                f"KT_PREFILL_KV_LEND: {ln >> 20} MiB of KV tensor {ti} cannot be mapped back on {dev} "
                f"({torch.cuda.mem_get_info(dev)[0] >> 20} MiB free): the prefill kept memory it was lent")
    torch.cuda.synchronize(dev)
    if VERIFY and STATE.get("verify") is not None:
        used, before = STATE.pop("verify")
        still = used.clone()
        still[_free_pages(a).long()] = False  # freed in the window: no longer ours to judge
        still[STATE["lent"].long()] = False
        after = _page_hashes(still, STATE["n"])
        bad = torch.nonzero(still[: STATE["n"] + 1] & (after != before)).flatten()
        STATE["verified"] = STATE.get("verified", 0) + 1
        STATE["verify_bad"] = STATE.get("verify_bad", 0) + int(bad.numel())
        _log(f"verify window {STATE['windows']}: {int(still.sum())} pages in use checked, "
             f"{int(bad.numel())} changed{f' (pages {bad[:8].tolist()})' if bad.numel() else ''}; "
             f"{STATE['verified']} windows so far, {STATE['verify_bad']} pages changed in all")
    _give_back(a, STATE["lent"])
    if owner == "hot":
        if STATE["hot_lends"] in (1, 2, 10, 100) or CHECK or why not in ("window start",):
            _log(f"hot lend {STATE['hot_lends']}, {why}: {STATE['bytes'] / 1e9:.2f} GB mapped back, "
                 f"{STATE['lent'].numel()} pages returned in {(time.perf_counter() - t0) * 1e3:.1f} ms")
    elif STATE["windows"] in (1, 2, 10, 100) or CHECK or why != "end of window":
        _log(f"{why}: {STATE['bytes'] / 1e9:.2f} GB mapped back, {STATE['lent'].numel()} pages returned in "
             f"{(time.perf_counter() - t0) * 1e3:.1f} ms")
    STATE.update(lent=None, ranges=[], bytes=0, owner=None)


def _extend_pages(a, args, kw) -> int:
    """Pages an alloc_extend(prefix_lens, prefix_lens_cpu, seq_lens, seq_lens_cpu,
    last_loc, extend_num_tokens, ...) may take: the new tokens' pages and two
    partial ones a request."""
    prefix_lens = kw.get("prefix_lens", args[0] if args else ())
    tokens = kw.get("extend_num_tokens", args[5] if len(args) > 5 else 0)
    return int(tokens) // a.page_size + 2 * len(prefix_lens) + 1


def _wrap_allocator(a) -> None:
    if getattr(a, "_kt_kv_pf_lend", False):
        return
    a._kt_kv_pf_lend = True
    # the token allocator has alloc only (its callers extend and decode through it)
    orig = {k: getattr(a, k) for k in ("alloc", "alloc_extend", "alloc_decode", "available_size", "clear")
            if hasattr(a, k)}

    def _lent() -> int:
        return 0 if STATE["lent"] is None else int(STATE["lent"].numel())

    def ensure(pages: int, why: str) -> None:
        # pages handed over unmapped and not yet taken are in the free list, at its end (the highest-numbered: a lend
        # takes the free pages past the lowest it keeps, and the allocator hands out from the front): an allocation
        # that reaches them maps them back first
        if HELD["pages"] is not None and pages > _free_count(a) - int(HELD["pages"].numel()):
            _settle_held(why=f"{why} reached them ({pages} pages needed)")
        # the allocator merges what it freed when its free list runs short: those
        # pages count (the paged allocator stages every freed page until then)
        # a lend of the driver's spare alone has no page to give: taking it back for an allocation would hand the
        # slots back and load them again for nothing (a window or a pool clear still releases them)
        if _lent() > 0 and pages > _free_count(a):
            STATE["take_backs"] += 1
            if not _take_back(pages - _free_count(a), why) or pages > _free_count(a):
                restore(f"take back for {why} ({pages} pages needed, {_free_count(a)} kept)")

    def alloc(need_size: int):
        ensure(-(-need_size // a.page_size), "alloc")
        return orig["alloc"](need_size)

    # the arguments pass through as given: trees differ in them (num_new_pages)
    def alloc_extend(*args, **kw):
        if STATE["lent"] is not None:
            ensure(_extend_pages(a, args, kw), "alloc_extend")
        return orig["alloc_extend"](*args, **kw)

    def alloc_decode(*args, **kw):
        ensure(len(kw.get("seq_lens", args[0] if args else ())) + 1, "alloc_decode")
        return orig["alloc_decode"](*args, **kw)

    def available_size():
        return orig["available_size"]() + _lent() * a.page_size

    def clear():
        restore("clear")
        return orig["clear"]()

    for k, fn in (("alloc", alloc), ("alloc_extend", alloc_extend), ("alloc_decode", alloc_decode),
                  ("available_size", available_size), ("clear", clear)):
        if k in orig:
            setattr(a, k, fn)


# -- the sliding-window pool: lent outside prefill ---------------------------------

SWA_ENABLED = os.environ.get("KT_PREFILL_SWA_LEND", "0") == "1" and kt_kv_vmm.ENABLED


def swa_auto(is_hybrid_swa: bool) -> None:
    """KT_PREFILL_SWA_LEND=auto, decided by the KV cache configurator (its is_hybrid_swa)
    before the pools are budgeted; the pool holds the plan's cap (KT_PREFILL_LEND_MAX_CHUNK) unless
    KT_PREFILL_SWA_TOKENS says otherwise."""
    global SWA_ENABLED, SWA_TOKENS
    if os.environ.get("KT_PREFILL_SWA_LEND") != "auto" or not kt_kv_vmm.ENABLED:
        return
    SWA_ENABLED = bool(is_hybrid_swa)
    if SWA_ENABLED and not SWA_TOKENS:
        cap = int(os.environ.get("KT_PREFILL_LEND_MAX_CHUNK", "0") or 0)
        SWA_TOKENS = (cap + 512) * 102 // 100 if cap > 0 else 0
    if SWA_ENABLED:
        _log(f"SWA lend on: a hybrid sliding-window model, its pool at least {SWA_TOKENS} tokens")
SWA_TOKENS = int(os.environ.get("KT_PREFILL_SWA_TOKENS", "0") or 0)
SWA_KEEP = int(os.environ.get("KT_PREFILL_SWA_LEND_KEEP", "4096"))  # tokens kept mapped outside prefill
SWA_MIN = int(os.environ.get("KT_PREFILL_SWA_LEND_MIN_MB", "64")) << 20  # smallest lend worth a sync
SWA_SLACK = 1024  # tokens mapped beyond an allocation's shortfall
# with HiCache, the trim keeps SWA_KEEP tokens outside prefill
SWA_HOST_TRIM = os.environ.get("KT_PREFILL_SWA_HOST_TRIM", "1") == "1"
SWA = {"alloc": None, "tensors": None, "lent": None, "bytes": 0, "lends": 0, "maps": 0, "ticks": 0}


SAFE = int(os.environ.get("KT_PREFILL_PLAN_SAFE", "0") or 0)
# the chunk the pools are budgeted at under the SWA lend: small, so the SWA pool
# takes no more of the KV budget than a launcher's own ratio does (at the lend's
# chunk MiMo's full pool fell 262,144 -> 225,340); the rest of it is lent
SIZING = int(os.environ.get("KT_PREFILL_SWA_SIZING_CHUNK", "2048") or 0)


def _in_flight() -> int:
    """Chunks of one request the SWA pool holds at once: sglang's own rule (its SWA
    request cap), two with the overlap scheduler, whose next chunk is allocated
    before the last one's slots are freed, else one. V4.1 mapped 0.17, 0.34 and
    0.25 GB of SWA over three chunks of one window."""
    env = os.environ.get("KT_PREFILL_SWA_CHUNKS_IN_FLIGHT")
    if env:
        return max(1, int(env))
    try:
        from sglang.srt.runtime_context import get_server_args

        return 1 if get_server_args().disable_overlap_schedule else 2
    except Exception:
        return 2


@contextlib.contextmanager
def swa_sizing():
    """KV cache configurator, around the memory budget: with the SWA lend the pools
    are budgeted at a small chunk (KT_PREFILL_SWA_SIZING_CHUNK), not at the plan's
    cap, so a pool that counts chunks in flight (DeepSeek V4's SWA cap) takes no
    more of the budget than before; swa_size then widens the SWA pool alone."""
    try:
        from sglang.srt.runtime_context import get_schedule

        bag = get_schedule()
        chunk = int(bag.chunked_prefill_size or 0)
    except Exception:
        bag, chunk = None, 0
    at = min(x for x in (SIZING, SAFE) if x > 0) if (SIZING > 0 or SAFE > 0) else 0
    if not SWA_ENABLED or bag is None or at <= 0 or chunk <= at:
        yield
        return
    SWA["sized_at"] = (at, chunk)
    _log(f"pools budgeted at chunk {at}, the plan's cap {chunk} widens the SWA pool only")
    with bag.override(chunked_prefill_size=at):
        yield


def swa_size(size: int, page: int, follow):
    """KV cache configurator: (SWA pool tokens, the pool that follows its pages).
    At least KT_PREFILL_SWA_TOKENS, and as many more chunks of the cap over the
    budgeted chunk as the pool holds at once (_in_flight); whole pages. A pool
    sized per SWA page (DeepSeek V4's c4 compress state: pages x ring rows) grows
    with it; when it does not divide into the pages the SWA pool is not widened."""
    if not SWA_ENABLED or not size:
        return size, follow
    want = SWA_TOKENS
    # what stays mapped outside prefill at most (swa_trim): the pool a launch
    # without the plan would have had at the lend's own chunk
    SWA["budget"] = size
    if "sized_at" in SWA:
        at, cap = SWA["sized_at"]
        want = max(want, size + _in_flight() * (cap - at))
        SWA["budget"] = size + max(0, SAFE - at)
    page = max(1, int(page))
    want = -(-want // page) * page
    if want <= size:
        # budget-sized: its buffers made with only the kept pages mapped all the same
        _log(f"SWA pool {size} tokens (the budget's, at least the {want} the lend needs): made with its kept pages "
             "mapped, its free pages lent outside prefill")
        _lazy_claim(size, page, follow)
        return size, follow
    if follow:
        pages = size // page
        if size % page or not pages or follow % pages:
            _log(f"SWA pool {size} not widened: {follow} rows that follow it do not divide into its {pages} pages")
            return size, follow
        follow = want // page * (follow // pages)
    _log(f"SWA pool {size} -> {want} tokens{f' ({follow} compress-state rows)' if follow else ''}: "
         "sized for the largest chunk, its free pages lent outside prefill")
    _lazy_claim(want, page, follow)
    return want, follow


def _lazy_claim(tokens: int, page: int, follow):
    """The widened SWA pool's buffers are made with only their kept pages (and last
    chunk) mapped: the widening costs no memory at setup (MiMo at 74,874 tokens
    asked 3.5 GB there). Claimed by their first dimension: tokens + page (a token
    layout), the pages + 1 (a page layout), or the follower's rows plus its ring
    padding; swa_init checks every claimed buffer is one it lends."""
    pages = tokens // page
    dims = {tokens + page, pages + 1}
    ring = (follow // pages) if follow and pages else 0
    lo, hi = (follow, follow + ring + 8) if ring else (0, -1)
    keep = min(1.0, (SWA_KEEP + SWA_SLACK + 2 * page) / max(1, tokens))

    def claim(shape, nbytes):
        if not shape or nbytes < (2 << 20):
            return None
        d = int(shape[0])
        if d in dims or lo <= d <= hi:
            return int(nbytes * keep) + 1
        # K and V as the two halves of one allocation; each half keeps its lead
        if d == 2 and len(shape) > 1 and int(shape[1]) in dims:
            half = nbytes // 2
            return [(0, int(half * keep) + 1), (half, int(half * keep) + 1)]
        return None

    kt_kv_vmm.LAZY = claim


def swa_tokens(size: int) -> int:
    """The SWA pool's size alone (a configurator patched before swa_size)."""
    return swa_size(size, 1, None)[0]


def _swa_state_tensors(pool, n: int):
    """DeepSeek V4's c4 compress-state buffers: ring rows per SWA page
    (state loc = SWA page x ring + loc % ring), so they lend with the SWA pages.
    Only the ratio-4 pools: their size follows the SWA pool (c4_state_pool_size);
    the ratio-128 ones are request-scoped. The pool may hold more rows than its
    ring uses a page (the configurator's ring is the larger): those stay mapped."""
    out, why = [], {}
    for name in ("compress_state_pools", "indexer_compress_state_pools"):
        for sp in getattr(pool, name, None) or []:
            if sp is None or getattr(sp, "ratio", 0) != 4:
                continue
            buf = getattr(getattr(sp, "kv_score_buffer", None), "kv_score", None)
            k = int(getattr(sp, "ring_size", 0) or 0)
            if getattr(sp, "online", False) or not getattr(sp, "swa_page_size", 0):
                r = f"online {getattr(sp, 'online', None)} swa_page_size {getattr(sp, 'swa_page_size', None)}"
            elif not isinstance(buf, torch.Tensor):
                r = f"no kv_score tensor ({type(getattr(sp, 'kv_score_buffer', None)).__name__})"
            elif not (buf.is_cuda and buf.is_contiguous()):
                r = "not a contiguous CUDA tensor"
            elif k <= 0 or buf.shape[0] < n * k:
                r = f"{buf.shape[0]} rows for {n} pages x ring {k}"
            else:
                out.append((buf, k * buf[0].nbytes))
                continue
            why[r] = why.get(r, 0) + 1
    if out or why:
        _log(f"{len(out)} compress-state buffers lend with the SWA pages"
             + "".join(f"; {c} left: {r}" for r, c in why.items()))
    return out


def swa_init(allocator, pool) -> None:
    """KV cache configurator, after the pools are made: set up and lend."""
    kt_kv_vmm.LAZY = None
    if not SWA_ENABLED or SWA["alloc"] is not None or SWA.get("off"):
        _map_lazy("SWA lend not set up")
        return
    SWA["off"] = True  # until set up
    try:
        _swa_init(allocator, pool)
    finally:
        # a lazily made buffer the lend does not hold is mapped whole now
        _map_lazy("not lent", keep={t[0].data_ptr() for t in (SWA["tensors"] or [])} if not SWA.get("off") else set())


def _map_lazy(why: str, keep=frozenset()) -> None:
    made, kt_kv_vmm.LAZY_MADE[:] = list(kt_kv_vmm.LAZY_MADE), []
    rest = [t for t in made if t.data_ptr() not in keep]
    for t in rest:
        if kt_kv_vmm.restore(t.view(-1).view(torch.uint8), 0, t.nbytes) < 0:
            raise RuntimeError(f"KT_PREFILL_SWA_LEND: a lazily made pool buffer cannot be mapped ({why})")
    if rest:
        torch.cuda.synchronize()
        _log(f"{len(rest)} lazily made buffers mapped whole ({why})")
    if made:
        _log(f"{len(made) - len(rest)} SWA buffers made with only their kept pages mapped")


def _swa_init(allocator, pool) -> None:
    a = getattr(allocator, "swa_attn_allocator", None)
    kvp = getattr(pool, "swa_kv_pool", None)
    if a is None or kvp is None:
        _log("no SWA allocator or pool: the SWA lend is off")
        return
    if type(a).__name__ not in ("PagedTokenToKVPoolAllocator", "TokenToKVPoolAllocator") \
            or not isinstance(getattr(a, "free_pages", None), torch.Tensor):
        _log(f"SWA allocator {type(a).__name__} has no single free list: the SWA lend is off")
        return
    n = int(getattr(a, "num_pages", 0) or a.size // a.page_size)
    tensors = _kv_tensors(kvp, n, a.page_size, "SWA K/V", _swa_state_tensors(pool, n))
    if not tensors:
        return
    SWA.update(alloc=a, tensors=tensors, ps=a.page_size, n=n, off=False,
               row=sum(t[1] for t in tensors) / a.page_size)
    _wrap_swa(a)
    _log(f"SWA lend: {len(tensors)} K/V tensors of {kvp.__class__.__name__}, {n} pages of {a.page_size} tokens, "
         f"{SWA['row'] / 1024:.1f} KiB a token, chunks {min(t[2] for t in tensors) >> 10}-"
         f"{max(t[2] for t in tensors) >> 10} KiB, {SWA_KEEP} tokens kept mapped")
    swa_lend("pool setup", force=True)
    # what is unmapped now, the lent pages (a lazily made buffer never mapped them). The
    # device's released count holds every lazily made range: 81 GiB on MiMo against a 6.4 GiB SWA pool
    lent = 0 if SWA["lent"] is None else int(SWA["lent"].numel())
    SWA["bytes"] = int(lent * a.page_size * SWA["row"])


def _swa_free(a) -> int:
    return _free_count(a)


def swa_lend(why: str, force: bool = False) -> None:
    """Unmap the free SWA pages beyond the kept ones (outside prefill only). force:
    whatever the surplus (at setup, where the pages past the kept ones of a lazily
    made pool have never been mapped and must leave the free list)."""
    a = SWA["alloc"]
    if a is None:
        return
    swa_flush(why)  # pages handed back but not mapped yet are not lent twice
    if not force:
        swa_trim(why)
    keep = -(-SWA_KEEP // SWA["ps"]) + 1
    surplus = _swa_free(a) - keep
    if surplus <= 0 or (not force and surplus * SWA["row"] * SWA["ps"] < SWA_MIN):
        return
    t0 = time.perf_counter()
    _merge_free(a)
    n, dev = SWA["n"], a.free_pages.device
    free = torch.sort(_free_pages(a)).values
    cand = free[keep:]
    lent = cand if SWA["lent"] is None else torch.cat((SWA["lent"], cand))
    mark = torch.zeros(n + 2, dtype=torch.int32, device=dev)
    mark[lent] = 1
    torch.cuda.synchronize(dev)
    got = 0
    for ti, off, ln in _whole_chunks(SWA["tensors"], mark, n, dev):
        got += max(0, kt_kv_vmm.release(SWA["tensors"][ti][0], off, ln))
    a.free_pages = a.free_pages[mark[a.free_pages] == 0]
    if getattr(a, "release_pages", None) is not None and len(a.release_pages):
        a.release_pages = a.release_pages[mark[a.release_pages] == 0]
    SWA["lent"] = torch.sort(lent).values
    SWA["bytes"] += got
    SWA["lends"] += 1
    if SWA["lends"] <= 3 or SWA["lends"] in (10, 100, 1000) or CHECK:
        _log(f"SWA lend ({why}): {cand.numel()} pages more, {SWA['lent'].numel()} lent, "
             f"{SWA['bytes'] / 1e9:.2f} GB unmapped (+{got / 1e9:.2f}) in {(time.perf_counter() - t0) * 1e3:.1f} ms")


def _defer_ok() -> bool:
    """A lent page handed out now may be mapped at the next forward's start: the
    lend sees every forward (kt_lend.before_forward flushes), and a window maps
    after the weights have left, when there is room."""
    try:
        from sglang.srt.layers.moe import kt_lend
    except ImportError:
        return False
    return bool(kt_lend.ENABLED) and kt_lend.STATE.get("phase") in ("decode", "prefill")


def bind_tree(scheduler) -> None:
    """kt_prefill_plan, from the scheduler: its prefix cache, for swa_trim (and its
    HiCache controller, for _wrap_hicache_load)."""
    if SWA_ENABLED and "tree" not in SWA:
        SWA["tree"] = getattr(scheduler, "tree_cache", None)
        _wrap_hicache_load(SWA["tree"])


def _wrap_hicache_load(tree) -> None:
    """A HiCache load-back allocates its SWA slots in the scheduler and starts the
    host-to-device copy before the next forward, which is where swa_map(defer=True)
    maps the pages it handed back: map them before the copy starts."""
    cc = getattr(tree, "cache_controller", None)
    if cc is None or getattr(cc, "_kt_swa_load", False) or not hasattr(cc, "start_loading"):
        return
    orig = cc.start_loading

    def start_loading(*args, **kw):
        swa_flush("HiCache load", quiet=SWA["maps"] > 3 and not CHECK)
        return orig(*args, **kw)

    cc.start_loading = start_loading
    cc._kt_swa_load = True
    _log(f"SWA lend: {type(cc).__name__}.start_loading maps the SWA pages it loads into first")


def _swa_in_use(a) -> int:
    lent = 0 if SWA["lent"] is None else int(SWA["lent"].numel())
    return (SWA["n"] - _swa_free(a) - lent) * SWA["ps"]


def _swa_evict_for(a, short_pages: int) -> None:
    """the prefix cache's least recently used SWA tokens, as many as an allocation is
    short of, before lent pages are mapped for it."""
    tree = SWA.get("tree")
    if tree is None or SWA.get("trim_off"):
        return
    try:
        got = _evict_swa(tree, int(short_pages * SWA["ps"]))
    except Exception as e:  # a cache without SWA eviction: map as before
        SWA["trim_off"] = True
        _log(f"SWA eviction before mapping unavailable ({type(e).__name__}: {e})")
        return
    SWA["evicts"] = SWA.get("evicts", 0) + 1
    if SWA["evicts"] <= 3 or SWA["evicts"] in (10, 100, 1000) or CHECK:
        _log(f"SWA eviction before mapping: {short_pages} pages short, the cache gave back "
             f"{got} tokens")


def _evict_swa(tree, tokens: int) -> int:
    """Ask the prefix cache for tokens of the SWA pool back, the way this SGLang's EvictParams words it: one
    count for the SWA pool, or a count for each window sub-pool by its name. The tokens it gave back."""
    import dataclasses

    from sglang.srt.mem_cache.base_prefix_cache import EvictParams

    fields = {f.name for f in dataclasses.fields(EvictParams)}
    if "swa_num_tokens" in fields:
        res = tree.evict(EvictParams(swa_num_tokens=tokens))
        return int(getattr(res, "swa_num_tokens_evicted", 0))
    if "window_num_tokens" in fields:
        # the names are the cache's own allocator's (the one that holds the SWA pool's allocator, where it is a
        # part of another); the SWA pool's allocator alone names none
        owners = [getattr(tree, "token_to_kv_pool_allocator", None), SWA["alloc"]]
        names = next((n for n in (o.window_pool_names() for o in owners if o is not None) if n), ())
        if not names:
            raise RuntimeError(f"yuzuri: {[type(o).__name__ for o in owners]} name no window pool: nothing to "
                               f"ask the cache for")
        res = tree.evict(EvictParams(window_num_tokens={n: tokens for n in names}))
        return int(sum(getattr(res, "window_num_tokens_evicted", {}).values()))
    raise RuntimeError(f"yuzuri: EvictParams has {sorted(fields)}: no count for the SWA pool in this SGLang")


def swa_trim(why: str) -> None:
    """Outside prefill the SWA pages in use stay within the budget: a prefix
    cache keeps a cached prompt's SWA tokens until its pool runs short, which with
    the lent pages counted it never does (MiMo kept ~1 GB a cached 24K prompt and
    the weights could not come back after the fifth). The cache gives back the
    excess, least recently used first, as it would under pressure (sglang's
    tree_cache.evict with swa_num_tokens)."""
    a, tree, budget = SWA["alloc"], SWA.get("tree"), SWA.get("budget")
    if a is None or tree is None or not budget:
        return
    if SWA_HOST_TRIM and getattr(tree, "cache_controller", None) is not None:
        # HiCache backs up what the cache gives back and loads it on the next match:
        # keep only what the SWA lend keeps, the rest goes to the lends
        if not SWA.get("host_trim_logged"):
            SWA["host_trim_logged"] = True
            _log(f"SWA trim: HiCache backs the cache in host memory, {SWA_KEEP} tokens kept outside prefill "
                 f"(not the {budget}-token budget)")
        budget = SWA_KEEP
    over = _swa_in_use(a) - budget
    if over <= 4 * SWA["ps"]:
        return
    try:
        got = _evict_swa(tree, int(over))
    except Exception as e:  # a cache without SWA eviction: nothing to trim
        if not SWA.get("trim_off"):
            SWA["trim_off"] = True
            _log(f"SWA trim unavailable ({type(e).__name__}: {e})")
        return
    SWA["trims"] = SWA.get("trims", 0) + 1
    if SWA["trims"] <= 3 or SWA["trims"] in (10, 100) or CHECK:
        _log(f"SWA trim ({why}): {over} tokens over the {budget}-token budget, the cache gave back "
             f"{got}")


def swa_map(short: int, why: str, defer: bool = False) -> None:
    """Hand back the lowest-numbered lent pages, short pages and some slack more.
    Their chunks are mapped now, or with defer at the next forward's start
    (swa_flush): an allocation for a prefill comes before its window, while
    decode still holds the weights (V4.1's second 35K could not map 279 MiB there)."""
    a, lent = SWA["alloc"], SWA["lent"]
    if lent is None or not lent.numel():
        return
    k = min(int(lent.numel()), short + -(-SWA_SLACK // SWA["ps"]))
    top = int(lent[k - 1])
    # each tensor from its first byte up to a chunk boundary past page top (the
    # chunks still mapped are skipped); the pages every tensor then has rejoin
    pend = SWA.setdefault("pending", [0] * len(SWA["tensors"]))
    upto = None
    for i, (flat, pb, C, c0) in enumerate(SWA["tensors"]):
        # the first chunk boundary at or past page top's end, at most the last whole chunk's end
        last = c0 + max(0, flat.numel() - c0) // C * C
        end = min(c0 + -(-max(0, (top + 1) * pb - c0) // C) * C, last)
        pend[i] = max(pend[i], end)
        # a tensor's last, partial chunk is never lent: mapped through to its end
        has = SWA["n"] + 1 if end >= last else end // pb - 1
        upto = has if upto is None else min(upto, has)
    back = lent[lent <= upto]
    SWA["lent"] = lent[lent > upto] if back.numel() < lent.numel() else None
    _give_back(a, back)
    SWA["maps"] += 1
    note = SWA["maps"] <= 3 or SWA["maps"] in (10, 100, 1000) or CHECK
    if note:
        _log(f"SWA pages back for {why}: {back.numel()} ({short} short), "
             f"{'mapped at the next forward' if defer else 'mapping now'}")
    if not defer:
        swa_flush(why, quiet=not note)


def swa_flush(why: str, quiet: bool = False) -> None:
    """Map the chunks of the pages handed back since the last flush (zeroed)."""
    pend = SWA.get("pending")
    if not pend or not any(pend):
        return
    t0 = time.perf_counter()
    dev = SWA["alloc"].free_pages.device
    got = 0
    for (flat, _, _, _), end in zip(SWA["tensors"], pend):
        if end <= 0:
            continue
        r = kt_kv_vmm.restore(flat, 0, end)
        if r < 0:
            # inside a window torch's cache holds the earlier chunks' activations,
            # free to torch but not to the driver (V4.1 failed here with 2 MiB
            # free and a plan that fit): hand it back and map again
            torch.cuda.synchronize(dev)
            torch.cuda.empty_cache()
            r = kt_kv_vmm.restore(flat, 0, end)
            SWA["map_retries"] = SWA.get("map_retries", 0) + 1
            if SWA["map_retries"] in (1, 10, 100):
                _log(f"SWA map at {why} needed torch's cache handed back ({SWA['map_retries']} so far)")
        if r < 0:
            raise RuntimeError(
                f"KT_PREFILL_SWA_LEND: SWA pages for {why} cannot be mapped back on {dev} "
                f"({torch.cuda.mem_get_info(dev)[0] >> 20} MiB free)")
        got += r
    torch.cuda.synchronize(dev)  # the zeroing is done before any kernel reads them
    SWA["pending"] = [0] * len(SWA["tensors"])
    SWA["bytes"] -= got
    if not quiet:
        _log(f"SWA map at {why}: {got / 1e9:.2f} GB mapped, {SWA['bytes'] / 1e9:.2f} GB still lent, "
             f"{(time.perf_counter() - t0) * 1e3:.1f} ms")


def swa_restore_all(why: str) -> None:
    lent = SWA["lent"]
    if lent is None:
        return
    swa_map(int(lent.numel()), why)


def swa_window_bytes() -> int:
    """what a prefill window at the plan's cap maps back of the lent SWA pages (host KV
    placement keeps it free)."""
    if SWA["alloc"] is None or SWA["lent"] is None or SWA.get("off"):
        return 0
    return int(SWA["row"] * max(0, SWA_TOKENS - SWA_KEEP))


def swa_terms():
    """kt_prefill_plan: (SWA tokens a request takes without mapping, bytes a token
    beyond them, bytes of rounding and slack, chunks held at once); None without
    the SWA lend."""
    a = SWA["alloc"]
    if a is None or SWA["lent"] is None:
        return None
    ps = SWA["ps"]
    have = _swa_free(a) * ps - ps  # the request's first page may be partly used
    extra = sum(t[2] for t in SWA["tensors"]) + SWA_SLACK * SWA["row"]
    # a later chunk is allocated before the last one's slots are freed
    return have, SWA["row"], int(extra), _in_flight()


SWA_EAGER = int(os.environ.get("KT_PREFILL_SWA_LEND_EAGER_MB", "256")) << 20


def swa_tick() -> None:
    """kt_lend, before every forward outside a window: lend what the frees gave
    back, at once when it is SWA_EAGER or more (a long prompt left 2.5 GB of SWA
    mapped on MiMo, and extend-only requests after it ran out of memory waiting
    for the 32-forward check), else every 32 forwards."""
    a = SWA["alloc"]
    if a is None:
        return
    SWA["ticks"] += 1
    surplus = (_swa_free(a) - -(-SWA_KEEP // SWA["ps"]) - 1) * SWA["row"] * SWA["ps"]
    over = SWA.get("budget") and _swa_in_use(a) > SWA["budget"] + 4 * SWA["ps"]
    if over or surplus >= SWA_EAGER or (SWA["ticks"] % 32 == 0 and surplus >= SWA_MIN):
        swa_lend("decode")


def _wrap_swa(a) -> None:
    if getattr(a, "_kt_swa_lend", False):
        return
    a._kt_swa_lend = True
    orig = {k: getattr(a, k) for k in ("alloc", "alloc_extend", "alloc_decode", "available_size", "clear")
            if hasattr(a, k)}

    def _lent() -> int:
        return 0 if SWA["lent"] is None else int(SWA["lent"].numel())

    def ensure(pages: int, why: str) -> None:
        if SWA["lent"] is not None and pages > _swa_free(a):
            _swa_evict_for(a, pages - _swa_free(a))
        if SWA["lent"] is not None and pages > _swa_free(a):
            swa_map(pages - _swa_free(a), why, defer=_defer_ok())

    def alloc(need_size: int):
        ensure(-(-need_size // a.page_size), "alloc")
        return orig["alloc"](need_size)

    def alloc_extend(*args, **kw):
        if SWA["lent"] is not None:
            ensure(_extend_pages(a, args, kw), "alloc_extend")
        return orig["alloc_extend"](*args, **kw)

    def alloc_decode(*args, **kw):
        ensure(len(kw.get("seq_lens", args[0] if args else ())) + 1, "alloc_decode")
        return orig["alloc_decode"](*args, **kw)

    def available_size():
        return orig["available_size"]() + _lent() * a.page_size

    def clear():
        # the lent pages stay lent: the list the clear rebuilds loses them again. Mapped back first, they had to fit
        # beside whatever took their memory since, and with the KV pool's rows on the card nothing was left (Gemma 4:
        # "SWA pages for clear cannot be mapped back (8 MiB free)" at the flush after a launch's first prompts, 11-10)
        r = orig["clear"]()
        lent = SWA["lent"]
        if lent is not None and lent.numel():
            mark = torch.zeros(SWA["n"] + 2, dtype=torch.int32, device=a.free_pages.device)
            mark[lent] = 1
            a.free_pages = a.free_pages[mark[a.free_pages] == 0]
            if getattr(a, "release_pages", None) is not None and len(a.release_pages):
                a.release_pages = a.release_pages[mark[a.release_pages] == 0]
        return r

    for k, fn in (("alloc", alloc), ("alloc_extend", alloc_extend), ("alloc_decode", alloc_decode),
                  ("available_size", available_size), ("clear", clear)):
        if k in orig:
            setattr(a, k, fn)
    _wrap_budget()


# the lent pages the ledger can map, for the scheduler's SWA budget
ROOM = {"lent": None, "lent_at": 0, "notes": 0}


def swa_room(tokens, why: str = "") -> None:
    """kt_prefill_plan: SWA tokens a prefill may hold (the mapped free pages and the lent ones the plan's room can
    map), the same on every rank; None: no bound. The scheduler's budget leaves out the lent pages past them."""
    a = SWA["alloc"]
    if a is None or SWA["lent"] is None or tokens is None:
        ROOM["lent"] = None
        return
    ps = SWA["ps"]
    lent = int(SWA["lent"].numel())
    have = _swa_free(a)
    can = max(0, -(-int(tokens) // ps) - have)
    ROOM.update(lent=can, lent_at=lent)
    ROOM["notes"] += 1
    # a plan of another shape, or the bound starting or stopping to leave pages out (the NLL's 24 pieces of one
    # shape logged 48 lines, chain74)
    key = (why, can < lent)
    if ROOM["notes"] <= 3 or ROOM["notes"] in (10, 100, 1000) or CHECK or key != ROOM.get("key"):
        ROOM["key"] = key
        _log(f"SWA room ({why}): {int(tokens)} tokens, {have * ps} mapped free and {min(can, lent) * ps} of the "
             f"{lent * ps} lent; the scheduler's budget leaves out {max(0, lent - can) * ps}")


def _budget_hidden() -> int:
    """SWA tokens of lent pages the room cannot map, less those mapped since it was measured."""
    a, can = SWA["alloc"], ROOM["lent"]
    if a is None or can is None or SWA["lent"] is None:
        return 0
    lent = int(SWA["lent"].numel())
    left = max(0, can - max(0, ROOM["lent_at"] - lent))
    return max(0, lent - left) * SWA["ps"]


def _wrap_budget() -> None:
    """sglang's SWA prefill budget without the lent pages the room cannot map. available_size still counts them:
    the scheduler's idle check conserves every page of the pool."""
    try:
        from sglang.srt.mem_cache.prefill_budget import SWAPrefillBudget
    except ImportError as e:
        _log(f"SWA room: no SWA prefill budget ({e}), the scheduler counts every lent page")
        return
    if getattr(SWAPrefillBudget, "_kt_swa_room", False):
        return
    # the property is found by what it computes (SGLang has renamed it): the window pool's free pages less the
    # request's offset
    import inspect

    def _is_it(v):
        try:
            src = inspect.getsource(v.fget)
        except (OSError, TypeError):
            return False
        return "self.allocator.swa_available_size()" in src and "self.swa_offset" in src

    found = [k for k, v in vars(SWAPrefillBudget).items() if isinstance(v, property) and _is_it(v)]
    if len(found) != 1:
        raise RuntimeError(f"yuzuri: SWAPrefillBudget has {found or 'no'} property that returns the window pool's "
                           f"free pages less the request's offset, the lend was written for one: this SGLang "
                           f"changed how it budgets a prefill")
    name, orig = found[0], vars(SWAPrefillBudget)[found[0]]

    def remaining(self):
        v = orig.fget(self)
        if getattr(self.allocator, "swa_attn_allocator", None) is SWA["alloc"]:
            return v - _budget_hidden()
        if not ROOM.get("other_logged"):
            ROOM["other_logged"] = True
            _log(f"SWA room: a budget on {type(self.allocator).__name__}, not the lent pool's: not bounded")
        return v

    setattr(SWAPrefillBudget, name, property(remaining))
    SWAPrefillBudget._kt_swa_room = True
