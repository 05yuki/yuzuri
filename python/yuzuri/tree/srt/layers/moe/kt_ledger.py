"""The lend ledger, step 1: one account of a device's memory at each change of the lend's state,
read only. See the patch's docstring for the parts."""

import ctypes
import os

import torch

ENABLED = os.environ.get("KT_LEDGER", "1") == "1"
TOL = int(os.environ.get("KT_LEDGER_TOL_MB", "64") or 64) << 20
LAST = {}  # (device, state) -> (why, other bytes)
_FNS = {}


def _vmm(index: int):
    """(device bytes kv-vmm has mapped, virtual bytes of its kv-vmm MemPool in torch's count), or None."""
    from sglang.srt.layers.moe import kt_kv_vmm

    if not kt_kv_vmm.ENABLED:
        return 0, 0
    lib = kt_kv_vmm._load()
    if not _FNS:
        for name in ("kv_vmm_mapped_bytes", "kv_vmm_virtual_bytes"):
            f = getattr(lib, name, None)
            if f is None:
                return None
            f.restype, f.argtypes = ctypes.c_size_t, [ctypes.c_int]
            _FNS[name] = f
    pool = getattr(kt_kv_vmm, "_pool", None)
    in_torch = 0
    if pool is not None and hasattr(pool, "snapshot"):
        in_torch = sum(int(seg.get("total_size", 0)) for seg in pool.snapshot())
    return int(_FNS["kv_vmm_mapped_bytes"](index)), in_torch


def _claims() -> dict:
    """What each lend says it holds now (bytes)."""
    out = {}
    try:
        from sglang.srt.layers.moe import kt_kv_hot_lend

        out["hot spare"] = int(kt_kv_hot_lend.STATE.get("spare_held", 0) or 0)
    except ImportError:
        pass
    try:
        from sglang.srt.layers.moe import kt_kv_prefill_lend as pl

        out["KV loan"] = int(pl.STATE.get("bytes", 0) or 0)
        out["SWA lent"] = int(pl.SWA.get("bytes", 0) or 0) if pl.SWA.get("alloc") is not None else 0
    except ImportError:
        pass
    from sglang.srt.layers.moe import kt_lend

    out["scratch kept"] = int(kt_lend.STATE.get("scratch_kept", 0) or 0)
    return out


def snapshot(dev, state: str, why: str) -> None:
    """state: "D" (decode weights mapped, scratch out) or "W" (weights out, scratch mapped)."""
    if not ENABLED:
        return
    from sglang.srt.layers.moe import kt_lend

    dev = torch.device(dev)
    index = dev.index if dev.index is not None else torch.cuda.current_device()
    v = _vmm(index)
    if v is None:
        return
    vmm_dev, vmm_in_torch = v
    free, total = torch.cuda.mem_get_info(index)
    used = total - free
    reserved = torch.cuda.memory_reserved(index)
    st = kt_lend.STATE
    weights = int(st.get("nbytes", 0) or 0)
    # the scratch a window made and keeps for the next ones lies in the scratch spans: reserved in torch's count,
    # mapped in W and out in D like the rest of the scratch
    scratch = _scratch(index) + int(st.get("scratch_kept", 0) or 0)
    lend = int(st.get("dec_nbytes", weights) or 0) if state == "D" else scratch
    own = reserved - vmm_in_torch - weights - scratch
    other = used - vmm_dev - lend - own
    claims = _claims()
    g = 1 << 30
    prev = LAST.get((index, state))
    LAST[(index, state)] = (why, other)
    drift = "" if prev is None else f", other {(other - prev[1]) / g:+.2f} GiB since {prev[0]}"
    line = (f"[kt-ledger] cuda:{index} {state} {why}: used {used / g:.2f} GiB = kv-vmm {vmm_dev / g:.2f} + lend "
            f"{lend / g:.2f} + torch {own / g:.2f} + other {other / g:.2f}{drift}; lends: "
            + ", ".join(f"{k} {b / g:.2f}" for k, b in claims.items()))
    if prev is not None and abs(other - prev[1]) > TOL:
        line += " (past the tolerance)"
    print(line, flush=True)
    if state == "D" and why == "window end" and _LISTED.get(index, 0) < TENSORS:
        _LISTED[index] = _LISTED.get(index, 0) + 1
        _tensors(index, reserved)


TENSORS = int(os.environ.get("KT_LEDGER_TENSORS", "0") or 0)
_LISTED = {}


def _holders(t) -> str:
    """Who refers to this tensor, two steps up: the containers and the objects that own them."""
    import gc
    import types

    if t is None:
        return "?"
    out = []
    for r in gc.get_referrers(t):
        if isinstance(r, types.FrameType) or r is t:
            continue
        if isinstance(r, (dict, list, tuple)):
            ups = []
            for u in gc.get_referrers(r):
                if isinstance(u, types.FrameType):
                    continue
                if isinstance(u, dict):
                    ups += [type(w).__name__ for w in gc.get_referrers(u)
                            if not isinstance(w, (types.FrameType, dict, list, tuple))][:2] or ["dict"]
                elif not isinstance(u, (list, tuple)):
                    ups.append(type(u).__name__)
            key = ""
            if isinstance(r, dict):
                key = next((f"[{k!r}]" for k, v in r.items() if v is t), "")
            out.append(f"{type(r).__name__}{key} of {'/'.join(sorted(set(ups))[:3]) or '?'}")
        else:
            out.append(type(r).__name__)
    return ", ".join(sorted(set(out))[:4]) or "nothing gc sees"


def listing(index: int) -> None:
    """The same listing, asked for from outside (the hot lend's regrow, before it empties torch's cache)."""
    if TENSORS:
        _tensors(index, torch.cuda.memory_reserved(index))


def _tensors(index: int, reserved: int) -> None:
    """KT_LEDGER_TENSORS=N: at a card's first N window ends, the tensors alive on it by storage, largest
    first, named where a module holds them. What "torch" in the line above is made of."""
    import gc

    names, seen, objs_by = {}, {}, {}
    objs = gc.get_objects()
    for o in objs:
        try:
            if isinstance(o, torch.nn.Module):
                for kind in (o._parameters, o._buffers):
                    for n, t in kind.items():
                        if t is not None and t.is_cuda:
                            names.setdefault(t.untyped_storage().data_ptr(), f"{type(o).__name__}.{n}")
        except Exception:
            pass
    for o in objs:
        try:
            if torch.is_tensor(o) and o.is_cuda and o.device.index == index:
                st = o.untyped_storage()
                k = st.data_ptr()
                if k not in seen:
                    seen[k] = (int(st.nbytes()), tuple(o.shape), str(o.dtype).replace("torch.", ""),
                               names.get(k, "?"))
                    objs_by[k] = o
        except Exception:
            pass
    rows = sorted(seen.values(), reverse=True)
    m = 1 << 20
    print(f"[kt-ledger] cuda:{index} tensors: {len(rows)} storages, {sum(r[0] for r in rows) / m:.0f} MiB named by "
          f"their storages; torch allocated {torch.cuda.memory_allocated(index) / m:.0f} MiB, reserved "
          f"{reserved / m:.0f} MiB", flush=True)
    # the segments torch cannot give back: partly used, with what sits in them (a tensor's storage is found by
    # its address)
    try:
        ptrs = {k: v for k, v in seen.items()}
        segs = [sg for sg in torch.cuda.memory_snapshot() if sg["device"] == index]
        part = [sg for sg in segs if 0 < sg["active_size"] < sg["total_size"]
                and sg["total_size"] - sg["active_size"] >= 8 * m]
        held = sum(sg["total_size"] - sg["active_size"] for sg in part)
        print(f"[kt-ledger] cuda:{index} segments: {len(segs)}, {len(part)} partly used with 8 MiB or more idle, "
              f"{held / m:.0f} MiB idle in them", flush=True)
        for sg in sorted(part, key=lambda sg: sg["active_size"] - sg["total_size"])[:12]:
            act = []
            for blk in sg["blocks"]:
                if blk["state"] == "active_allocated":
                    who = ptrs.get(blk["address"])
                    act.append(f"{blk['size'] / m:.1f}" + (f" {who[3]} {who[1]} {who[2]}" if who else " (no tensor)"))
                    if who and sg["total_size"] - sg["active_size"] >= 64 * m:
                        act[-1] += " held by " + _holders(objs_by.get(blk["address"]))
            print(f"[kt-ledger] cuda:{index}   segment {sg['total_size'] / m:.0f} MiB, idle "
                  f"{(sg['total_size'] - sg['active_size']) / m:.0f}: " + "; ".join(act[:6]), flush=True)
    except Exception as e:
        print(f"[kt-ledger] cuda:{index} segments: {e!r}", flush=True)
    groups = {}
    for nb, shape, dt, name in rows:
        g = groups.setdefault((name, shape, dt, nb), [0])
        g[0] += 1
    top = sorted(groups.items(), key=lambda kv: -kv[0][3] * kv[1][0])[:60]
    for (name, shape, dt, nb), (cnt,) in top:
        print(f"[kt-ledger] cuda:{index}   {nb * cnt / m:9.1f} MiB = {cnt} x {nb / m:.1f}  {name} {shape} {dt}",
              flush=True)


# step 2 (kt_prefill_plan.py reads it): the free memory a window starts with, from what each holder gives back
# or takes at the start, checked at every start against the driver
ERRS = {}  # device -> the last few errors at a window's start (measured free less the holders' sum)
# device -> what the scratch mapped at the last window start, less what earlier windows kept. A scratch tensor maps
# whole chunks, so the named bytes run short: Qwen3.5's 0.08 GiB mapped 0.45 (step2dQwen, 11-06)
SCRATCH = {}
PRED = {}  # device -> (free before the start, the holders' parts) taken at the D window start
KEEP_ERRS = 4


def _scratch(index: int) -> int:
    """The scratch a window start maps, before what earlier windows kept: measured once a start has, else named."""
    if index in SCRATCH:
        return SCRATCH[index]
    from sglang.srt.layers.moe import kt_lend

    return int(kt_lend.STATE.get("scratch_bytes", 0) or 0)


def _index(dev) -> int:
    dev = torch.device(dev)
    return dev.index if dev.index is not None else torch.cuda.current_device()


def _releasable(index: int) -> int:
    """What torch.cuda.empty_cache hands back on the card: the default pool's segments with nothing allocated.
    A private pool's (torch_memory_saver's regions: the lend's scratch, the weights) stays reserved; counting it
    predicted +1.57 GiB at a window start that measured +0.00 (a 1M pool, 11-08), and the plan's chunks
    did not fit."""
    return sum(int(s["total_size"]) for s in torch.cuda.memory_snapshot()
               if s["device"] == index and int(s["allocated_size"]) == 0
               and tuple(s.get("segment_pool_id", (0, 0))) == (0, 0))


def holders(index: int) -> dict:
    """What changes hands at the next window's start, by holder (bytes; positive frees, negative takes). In a
    window already, nothing."""
    from sglang.srt.layers.moe import kt_lend

    st = kt_lend.STATE
    if st.get("phase") == "prefill":
        return {}
    out = {
        "weights out": int(st.get("dec_nbytes", st.get("nbytes", 0)) or 0),
        "scratch in": -(_scratch(index) + int(st.get("scratch_kept", 0) or 0)),
        # the window's start empties torch's cache, which hands back only the segments with nothing allocated: the
        # free pieces of a split segment stay (predicted +0.23 GiB against +0.00 to +0.03 measured, step2Glm 11-06)
        "torch cache": _releasable(index),
    }
    try:
        from sglang.srt.layers.moe import kt_kv_hot_lend

        # the slots' memory beyond the KV pages (those come back through the KV loan)
        out["hot spare"] = int(kt_kv_hot_lend.STATE.get("spare_held", 0) or 0)
    except ImportError:
        pass
    return out


def window_free(index: int):
    """The free memory the next window starts with: free now, plus the holders' parts, plus the smallest error
    measured at the last window starts. None until a window start has been measured.
    In a window, the free memory its chunks have, on the base the forwards' cost is measured on: free now plus
    what torch reserved since the window's start (the cost counts that growth). The KV loan is already in the
    free memory. What else the window mapped since its start stays taken: the scratch grown to a larger chunk, and
    the SWA pages."""
    from sglang.srt.layers.moe import kt_lend

    st = kt_lend.STATE
    if st.get("phase") == "prefill" and st.get("pf_base") is not None:
        return int(torch.cuda.mem_get_info(index)[0]) + int(torch.cuda.memory_reserved(index) - st["pf_base"])
    errs = ERRS.get(index)
    if not errs:
        return None
    from sglang.srt.layers.moe import kt_kv_vmm

    return kt_kv_vmm.card_free(index) + sum(holders(index).values()) + min(errs)


def carry_out(index: int):
    """For the next launch's first window (kt_prefill_plan.record): the smallest start error kept and the scratch a
    start mapped. None before a window start has been measured."""
    if not ERRS.get(index) or index not in SCRATCH:
        return None
    return {"err": int(min(ERRS[index])), "scratch": int(SCRATCH[index])}


def carry_in(index: int, rec) -> None:
    """An earlier launch's start error and scratch, until this launch's first window start measures its own (the
    error joins the kept ones; the scratch is replaced): window_free then answers before the first window."""
    if ERRS.get(index) or not rec:
        return
    ERRS[index] = [int(rec["err"])]
    SCRATCH.setdefault(index, int(rec["scratch"]))


def predict_start(dev, held: int = 0, hot_stays: bool = False) -> None:
    """kt_lend, at the D window start (the hot slots already out, the weights still mapped). held: the hot lend's
    pages handed to the window unmapped, not free for it until the loan has decided (kt_kv_prefill_lend.held_bytes).
    hot_stays: the hot slots the window's plan does not need stay through it, and their spare with them."""
    index = _index(dev)
    parts = holders(index)
    if hot_stays:
        parts.pop("hot spare", None)
    PRED[index] = (int(torch.cuda.mem_get_info(index)[0]) - int(held), parts)


def check_start(dev, free: int, measured: dict) -> None:
    """kt_lend, at the W window start: the driver's free memory against the prediction. measured: what the lend
    saw each part move (weights out, scratch in), for the line."""
    index = _index(dev)
    p = PRED.pop(index, None)
    if p is None:
        return
    before, parts = p
    if "scratch in" in measured:
        from sglang.srt.layers.moe import kt_lend

        SCRATCH[index] = max(0, -int(measured["scratch in"]) - int(kt_lend.STATE.get("scratch_kept", 0) or 0))
    said = before + sum(parts.values())
    err = int(free) - said
    errs = ERRS.setdefault(index, [])
    # the error kept for the next starts is with the scratch as measured now, which the next prediction uses
    kept = err
    if "scratch in" in parts and index in SCRATCH:
        from sglang.srt.layers.moe import kt_lend

        kept += parts["scratch in"] + SCRATCH[index] + int(kt_lend.STATE.get("scratch_kept", 0) or 0)
    errs.append(kept)
    del errs[:-KEEP_ERRS]
    g = 1 << 30
    print(f"[kt-ledger] cuda:{index} window start free {free / g:.2f} GiB, the holders said {said / g:.2f} GiB, "
          f"error {err / g:+.2f} GiB (smallest of the last {len(errs)}: {min(errs) / g:+.2f}); "
          + ", ".join(f"{k} {v / g:+.2f}" + (f" (measured {measured[k] / g:+.2f})" if k in measured else "")
                      for k, v in parts.items()), flush=True)
