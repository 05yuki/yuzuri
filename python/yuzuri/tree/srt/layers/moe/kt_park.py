"""YUZURI_PARK=1: a server that leaves the cards and comes back with its sessions.

Two servers on one host, one model each: the one that waits holds nothing on the cards but its CUDA context, so the
one that runs has them whole. Parked:
- the hot slots are given back (the experts are in kt-kernel's arenas);
- every row of the KV pool is in host memory (host KV keeps the pool's addresses; nothing is flushed, the prefix
  cache keeps its entries);
- the weights the prefill lend moves are out, as at a window's start;
- everything else of the model is paused by torch_memory_saver with a copy in host memory: after the load, every
  parameter and buffer that owns its storage is copied once into one region and the model points at the copy
  (adopt). Loading inside the region instead kept every block the load freed mapped: 11.8 GB a card against 5.5
  on Flash-Next NVFP4 (a region gives nothing back before it is paused).
Resume maps and fills them again in the reverse order; the hot slots come back with the lend's next growth.
What stays: the CUDA context, the kernels, the libraries' own workspaces, and what was allocated after the load
outside a region (the report says how much)."""
import contextlib
import os
import time

import torch

ENABLED = os.environ.get("YUZURI_PARK") == "1"
TAG = "yuzuri_park"
STATE = {"parked": False, "rows": {}, "lend": False}
GiB = 1 << 30


def _log(msg: str) -> None:
    print(f"[kt-park] {msg}", flush=True)


def _used(index: int) -> int:
    free, total = torch.cuda.mem_get_info(index)
    return total - free


def region():
    """Around the model's load: what it allocates can be paused, with a host copy."""
    if not ENABLED:
        return contextlib.nullcontext()
    from sglang.srt.layers.moe import kt_lend

    return kt_lend._tms().region(tag=TAG, enable_cpu_backup=True)


def _vmm_bytes(nbytes: int, index: int):
    """nbytes of the card on kv_vmm's remappable memory, or None without the library: a park takes them to pinned
    host memory with the pools (kv_vmm_park, about 15 GB/s from the second park on), where the saver's region with
    its host copy took 1.3 s for 1.6 GiB."""
    try:
        from sglang.srt.layers.moe import kt_kv_vmm

        kt_kv_vmm._load()
        blk = kt_kv_vmm._Block(int(nbytes), int(index))
        return torch.as_tensor(blk, device=f"cuda:{index}")
    except Exception as exc:
        _log(f"kv_vmm holds no buffer for the park ({type(exc).__name__}: {exc}): the saver's region takes it")
        return None


def buffers():
    """Around the making of a persistent buffer that must leave the card at a park and come back at its address:
    kv_vmm's pool when torch can allocate from it, else the saver's region."""
    if not ENABLED:
        return contextlib.nullcontext()
    try:
        from sglang.srt.layers.moe import kt_kv_vmm

        kt_kv_vmm._load()
        if kt_kv_vmm._pool is not None:
            return torch.cuda.use_mem_pool(kt_kv_vmm._pool)
    except Exception:
        pass
    return region()


def adopt(model) -> None:
    """Once, after the load and before any graph is captured: the model's own tensors into the region."""
    if not ENABLED:
        return
    i = torch.cuda.current_device()
    if os.environ.get("YUZURI_PARK_LIST") == "1":
        # where each block that stays was allocated (the listing reads the stacks)
        torch.cuda.memory._record_memory_history(enabled="all", context="alloc", stacks="python", max_entries=200000)
    before = _used(i)
    moved = {}  # old storage -> its place in the buffer (two parameters on one storage stay on one)
    took = skipped = shared = 0
    plan, total = [], 0
    for kind in (model.named_parameters(), model.named_buffers()):
        for name, t in list(kind):
            d = t.data
            if not d.is_cuda or d.numel() == 0:
                continue
            st = d.untyped_storage()
            key = st.data_ptr()
            if key in moved:
                first = plan[moved[key]][1]
                if first.shape == d.shape and first.dtype == d.dtype:
                    plan.append((t, d, plan[moved[key]][2], True))
                    shared += 1
                else:
                    skipped += d.numel() * d.element_size()
                continue
            if d.storage_offset() != 0 or st.nbytes() != d.numel() * d.element_size() or not d.is_contiguous():
                # a view into a larger buffer (the lend's spans, a method's arena): its owner moves it
                skipped += d.numel() * d.element_size()
                continue
            moved[key] = len(plan)
            plan.append((t, d, total, False))
            total += (d.numel() * d.element_size() + 255) & ~255
            took += d.numel() * d.element_size()
    # one buffer for all of them: the region's allocations are whole 2 MiB pages each, and a model's hundreds of
    # small tensors one allocation each cost 0.15 to 0.19 GiB a card (Flash-Next, 1.19 GiB of tensors)
    made = {}
    if total:
        buf = _vmm_bytes(total, i)
        if buf is None:
            with region():
                buf = torch.empty(total, dtype=torch.uint8, device=f"cuda:{i}")
        STATE.setdefault("adopted_ptrs", set()).add(buf.untyped_storage().data_ptr())
        for t, d, off, again in plan:
            if not again:
                made[off] = buf[off:off + d.numel() * d.element_size()].view(d.dtype).view(d.shape)
                made[off].copy_(d)
            t.data = made[off]
        del buf
    del plan, made
    del moved
    torch.cuda.synchronize(i)
    torch.cuda.empty_cache()
    STATE["adopted"] = took
    _log(f"adopted {took / GiB:.2f} GiB of the model's tensors into the region ({shared} sharing a storage kept "
         f"together; {skipped / GiB:.2f} GiB are views into buffers of their own and stay with their owners); "
         f"cuda:{i} {before / GiB:.2f} -> {_used(i) / GiB:.2f} GiB")


def _listing(index: int, top: int = 28) -> str:
    """YUZURI_PARK_LIST=1: the card's tensors this process still holds outside the park's own region, largest first
    (one line a storage): what a park leaves, to find its owner."""
    import gc

    seen, rows = set(), []
    adopted = STATE.get("adopted_ptrs") or set()
    # torch's own pool only: a saver region's allocations are in pools of their own and leave with their tag
    plain, plain_bytes = set(), 0
    for seg in torch.cuda.memory_snapshot():
        if seg["device"] == index and tuple(seg.get("segment_pool_id", (0, 0))) == (0, 0):
            for b in seg["blocks"]:
                if b["state"].startswith("active"):
                    plain.add(int(b["address"]))
                    plain_bytes += int(b["size"])
    for o in gc.get_objects():
        try:
            if not (torch.is_tensor(o) and o.is_cuda and o.device.index == index and o.numel()):
                continue
            st = o.untyped_storage()
            key = st.data_ptr()
        except Exception:
            continue
        if key in seen or key in adopted or key not in plain:
            continue
        seen.add(key)
        rows.append((st.nbytes(), tuple(o.shape), str(o.dtype).replace("torch.", "")))
    import collections

    total = sum(r[0] for r in rows)
    kinds = collections.Counter()
    for n, sh, dt in rows:
        kinds[(n, sh, dt)] += 1
    by = sorted(kinds.items(), key=lambda kv: -kv[0][0] * kv[1])[:top]
    import collections as _c

    pools, blocks = _c.defaultdict(lambda: [0, 0, 0]), _c.Counter()
    for seg in torch.cuda.memory_snapshot():
        if seg["device"] != index:
            continue
        pid = tuple(seg.get("segment_pool_id", (0, 0)))
        pools[pid][0] += 1
        pools[pid][1] += int(seg["total_size"])
        pools[pid][2] += int(seg["allocated_size"])
        if pid == (0, 0):
            for b in seg["blocks"]:
                if b["state"].startswith("active") and int(b["address"]) not in seen:
                    blocks[int(b["size"])] += 1
    try:
        where = _c.Counter()
        for seg in torch.cuda.memory._snapshot()["segments"]:
            if seg.get("device") != index:
                continue
            for b in seg["blocks"]:
                if b["state"] == "active_allocated" and int(b["size"]) >= (8 << 20) and int(b["address"]) not in seen:
                    fr = [f for f in (b.get("frames") or []) if "site-packages/torch/" not in f["filename"]][:4]
                    where[(int(b["size"]) >> 20, " < ".join(f"{os.path.basename(f['filename'])}:{f['line']} {f['name']}" for f in fr) or "no Python frame")] += 1
        for (mb, at), c in sorted(where.items(), key=lambda kv: -kv[0][0] * kv[1])[:16]:
            _log(f"cuda:{index} stays: {c} x {mb} MiB allocated at {at}")
    except Exception as e:
        _log(f"no allocation stacks: {type(e).__name__}: {e}")
    big = sorted(blocks.items(), key=lambda kv: -kv[0] * kv[1])[:14]
    _log(f"cuda:{index} pools (id: segments, MiB reserved, MiB allocated): "
         + "; ".join(f"{pid}: {v[0]}, {v[1] >> 20}, {v[2] >> 20}" for pid, v in sorted(pools.items(), key=lambda kv: -kv[1][1])[:10])
         + f" | blocks of torch's own pool no tensor here owns: "
         + ", ".join(f"{c} x {n / (1 << 20):.1f} MiB" for n, c in big))
    return (f"{plain_bytes / GiB:.2f} GiB allocated in torch's own pool; {len(rows)} tensors' storages there, "
            f"{total / GiB:.2f} GiB: "
            + ", ".join(f"{c} x {n / (1 << 20):.0f} MiB {sh} {dt}" for (n, sh, dt), c in by))


def park() -> None:
    if not ENABLED:
        raise RuntimeError("YUZURI_PARK=1 was not set when this server started: its weights are not in a region")
    if STATE["parked"]:
        return
    from sglang.srt.layers.moe import kt_kv_hot_lend, kt_kv_vmm, kt_lend

    i = torch.cuda.current_device()
    torch.cuda.synchronize(i)
    t0 = time.perf_counter()
    said = [f"{_used(i) / GiB:.2f} GiB on cuda:{i}"]

    def step(name):
        torch.cuda.synchronize(i)
        torch.cuda.empty_cache()
        said.append(f"{name} -> {_used(i) / GiB:.2f} ({(time.perf_counter() - t0) * 1e3:.0f} ms)")

    from sglang.srt.layers.moe import kt_kv_prefill_lend

    # the slots leave the routing and give their memory back; the pool pages they had borrowed stay unmapped and
    # rejoin the free list (as at a window's start): they hold nothing, so they are neither mapped nor copied to
    # host memory and back, and the lend after the resume takes them as they are
    kt_kv_hot_lend.release("park")
    kt_kv_prefill_lend.hot_release("park")
    step("hot slots out")
    rows = {}
    for t in kt_kv_vmm.HOST_MADE:
        if t.device.index == i:
            rows[t.data_ptr()] = (t, int(kt_kv_vmm._info(t)["dev"]))
            if kt_kv_vmm._set_rows(t, 0, live=True) < 0:
                raise RuntimeError("kv_vmm: a pool tensor's rows did not move to host memory")
    STATE["rows"] = rows
    step(f"KV rows of {len(rows)} pool tensors to host")
    STATE["lend"] = kt_lend.park_out(i)
    step("the lend's weights out" if STATE["lend"] else "no lent weights")
    # what else sits on kv_vmm's ranges and is still mapped (the pools that are not token KV: recurrent states,
    # tables): to host memory chunk by chunk, their contents kept
    lib = kt_kv_vmm._load() if kt_kv_vmm.ENABLED else None
    STATE["vmm"] = bool(lib is not None and hasattr(lib, "kv_vmm_park"))
    if STATE["vmm"]:
        n = lib.kv_vmm_park(i)
        if n < 0:
            raise RuntimeError("kv_vmm: the mapped chunks did not move to host memory")
        step(f"{n / GiB:.2f} GiB of other pool memory to host")
    kt_lend._tms().pause(TAG)
    step("the load's region paused")
    STATE["parked"] = True
    if os.environ.get("YUZURI_PARK_LIST") == "1":
        _log(_listing(i))
        try:
            kt_lend._ledger("D", "parked", torch.device(f"cuda:{i}"))
        except Exception as e:  # a listing only
            _log(f"no ledger line: {e}")
    _log("parked: " + "; ".join(said) + f"; of what stays, torch holds {torch.cuda.memory_allocated(i) / GiB:.2f} GiB "
         f"allocated and {torch.cuda.memory_reserved(i) / GiB:.2f} GiB reserved by its count")


def resume() -> None:
    if not STATE["parked"]:
        return
    from sglang.srt.layers.moe import kt_kv_vmm, kt_lend

    i = torch.cuda.current_device()
    t0 = time.perf_counter()
    said = [f"{_used(i) / GiB:.2f} GiB on cuda:{i}"]

    def step(name):
        torch.cuda.synchronize(i)
        said.append(f"{name} -> {_used(i) / GiB:.2f} ({(time.perf_counter() - t0) * 1e3:.0f} ms)")

    kt_lend._tms().resume(TAG)
    step("the load's region back")
    if STATE.get("vmm"):
        if kt_kv_vmm._load().kv_vmm_unpark(i) < 0:
            raise torch.OutOfMemoryError("kv_vmm: the parked chunks did not come back to the card")
        step("other pool memory back")
    if STATE["lend"]:
        kt_lend.park_back(i)
        step("the lend's weights back")
    for t, n in STATE["rows"].values():
        if kt_kv_vmm._set_rows(t, n) < 0:
            raise torch.OutOfMemoryError("kv_vmm: a pool tensor's rows did not come back to the card")
    step("KV rows back")
    STATE.update(parked=False, rows={}, lend=False, vmm=False)
    _log("resumed: " + "; ".join(said) + "; the hot slots return with the lend")
