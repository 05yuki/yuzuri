"""The prefill chunk planned per request (KT_PREFILL_PLAN=1, with the prefill lend
and, for the room of the free KV pages, KT_PREFILL_KV_LEND=1).

The launch chunk (--chunked-prefill-size) becomes the largest chunk the
streamer's buffers are sized for; each request then gets the fewest chunks its
remaining prompt needs and, for that count, the smallest equal chunk.

A chunk of c rows whose forward ends at sequence s needs, above a window's start,

  cost(c, s) = FIXED + ROW * c + SEQ * s          (KT_PREFILL_PLAN_COST="FIXED_MiB,ROW_B,SEQ_B,CARRY_MiB",
                                                   fitted over the lend's per-forward records)

(+ CARRY for a chunk after the first of a window) and has
  room = free at the window's start + the free KV pages it borrows - kt_lend.MARGIN.
The free memory is live: each TP rank's decode free memory now plus the step
kt_lend measured from decode to a window's start (weights out, scratch in),
the smallest over the ranks (an all-reduce on the scheduler's TP CPU group, so
every rank plans the same chunk). Before the first window has measured that
step, KT_PREFILL_PLAN_FREE_MB stands in. The last chunk of a request is the
one with the longest sequence, so it is the one checked.

Without KT_PREFILL_PLAN_COST the older model applies: cost = KT_PREFILL_PLAN_ROW_KB * rows.
"""
import math
import os

_SAID = set()


def _say_once(where, exc):
    """One line the first time a guarded step fails for a reason other than its module being off, so an SGLang that
    changed under it shows in the log."""
    if isinstance(exc, ImportError) or where in _SAID:
        return
    _SAID.add(where)
    print(f"[yuzuri] {where}: {type(exc).__name__}: {exc} (said once)", flush=True)


ENABLED = os.environ.get("KT_PREFILL_PLAN", "0") == "1"
FREE = int(os.environ.get("KT_PREFILL_PLAN_FREE_MB", "0") or 0) << 20
ROW_KB = int(os.environ.get("KT_PREFILL_PLAN_ROW_KB", "0") or 0) << 10
_cost = [x for x in os.environ.get("KT_PREFILL_PLAN_COST", "").split(",") if x]
FIXED, ROW, SEQ = ((int(float(_cost[0]) * (1 << 20)), float(_cost[1]), float(_cost[2]))
                   if len(_cost) >= 3 else (0, float(ROW_KB), 0.0))
XS = 0.0  # cross term, bytes a row a sequence token
# what a later chunk of a window holds from the earlier ones (4th field, MiB)
CARRY = int(float(_cost[3]) * (1 << 20)) if len(_cost) >= 4 else 0
LIVE = os.environ.get("KT_PREFILL_PLAN_LIVE", "1") == "1"
DELTA = int(os.environ.get("KT_PREFILL_PLAN_DELTA_MB", "0") or 0) << 20
ALIGN = int(os.environ.get("KT_PREFILL_PLAN_ALIGN", "1024"))
# a GGUF pack (KT_GGUF_V41_DENSE) only: the chunks may run TAIL_STEPS aligned steps over the
# launch chunk, and the free memory is read without torch's cached blocks
_GGUF = bool(os.environ.get("KT_GGUF_V41_DENSE"))
TAIL_STEPS = int(os.environ.get("KT_PREFILL_PLAN_TAIL_STEPS", "2" if _GGUF else "0"))
# the chunk this setup is known to take without a plan (the lend's hint): what a
# request gets while there is no free-memory step to plan with (the first window
# of a launch without KT_PREFILL_PLAN_DELTA_MB); unset, the launch chunk
SAFE = int(os.environ.get("KT_PREFILL_PLAN_SAFE", "0") or 0)
# no cost in the environment: measured from the windows this server runs
ONLINE = not _cost and ROW_KB == 0
MEASURED = {"windows": 0, "row": 0.0, "max_rows": 0}
LP = 0.0  # bytes a row whose input logprobs are computed, past a plain row (vocabulary-wide logits)
# rid -> the chunk a request was planned at when the online cap bound it (capped plans)
CAPPED = {}


def _lp_row(lp_pts) -> float:
    """What a logprob row grew past the plain fit, the largest over the forwards that computed them."""
    return max([0.0] + [max(0.0, g - FIXED - ROW * r - SEQ * s - XS * r * s) / lp for r, s, g, lp in lp_pts])


# The forwards an earlier launch of this setup measured, carried into this one (kt_lend's hint file,
# <KT_PREFILL_LEND_HINT>-rank<r>.json "plan"): the first window is then planned from that launch's whole fit, not at
# the unmeasured chunk and the doublings after it (Qwen3.5: a first 35K prompt in five chunks at 2,919 tok/s, the third
# in one at 5,019). Every forward of the fit is carried, the runs of equal chunks with them: the sequence term has the
# evidence it had (only the largest window carried fitted it at half, 11-09). A forward of this launch that grows past
# what the carried fit says drops them, on every rank, and the launch measures for itself (_drop_carried).
CARRY_HINT = os.environ.get("KT_PREFILL_LEND_HINT", "")
CARRY_ON = os.environ.get("KT_PREFILL_PLAN_CARRY", "1") == "1"


def _stamp() -> str:
    """What the carried forwards are good for: this package's code (a change to it may change a row's cost)."""
    import hashlib

    if "stamp" in MEASURED:
        return MEASURED["stamp"]

    here = os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha1()
    for n in sorted(os.listdir(here)):
        if n.endswith(".py"):
            with open(os.path.join(here, n), "rb") as f:
                h.update(n.encode() + f.read())
    MEASURED["stamp"] = h.hexdigest()[:16]
    return MEASURED["stamp"]


def record():
    """The fit's forwards for the next launch (kt_lend._write_hint), the carried ones that still stand with them;
    None until two chunk sizes are measured."""
    pts = MEASURED.get("points") or []
    if not ONLINE or not MEASURED.get("fitted") or len({p[0] for p in pts}) < 2:
        return None
    import torch

    from sglang.srt.layers.moe import kt_ledger

    return {"stamp": _stamp(), "points": [list(p) for p in pts],
            "lp_points": [list(p) for p in MEASURED.get("lp_points", [])],
            # the free memory a window starts with is the ledger's, known once a start has been measured
            "start": kt_ledger.carry_out(torch.cuda.current_device())}


def _carry_in() -> None:
    """Once, at the first plan of a launch: the earlier launch's forwards as this one's first points."""
    global FIXED, ROW, SEQ, LP
    if MEASURED.get("carry_tried"):
        return
    MEASURED["carry_tried"] = True
    if not (ONLINE and CARRY_ON and CARRY_HINT) or MEASURED.get("points"):
        return
    try:
        import json

        from sglang.srt.layers.moe import kt_lend

        with open(f"{CARRY_HINT}-rank{kt_lend._rank()}.json") as f:
            rec = json.load(f).get("plan") or {}
    except Exception:
        return
    pts = [tuple(float(v) if i != 4 else int(v) for i, v in enumerate(p)) for p in rec.get("points", []) if len(p) == 6]
    if rec.get("stamp") != _stamp() or len({p[0] for p in pts}) < 2:
        if rec:
            print("[kt-plan] an earlier launch's forwards are not carried: "
                  + ("the package changed since" if rec.get("stamp") != _stamp() else "one chunk size only"), flush=True)
        return
    MEASURED["points"] = pts[-64:]
    MEASURED["lp_points"] = [tuple(float(v) for v in p) for p in rec.get("lp_points", []) if len(p) == 4][-64:]
    MEASURED["carried"] = len(MEASURED["points"])
    MEASURED["own"] = 0
    MEASURED["windows"] = len(pts)
    MEASURED["max_rows"] = int(max(p[0] for p in pts))
    FIXED, ROW, SEQ = _fit(MEASURED["points"])
    MEASURED["row"] = ROW
    MEASURED["fitted"] = True
    LP = _lp_row(MEASURED["lp_points"]) if MEASURED["lp_points"] else 0.0
    try:
        import torch

        from sglang.srt.layers.moe import kt_ledger

        kt_ledger.carry_in(torch.cuda.current_device(), rec.get("start"))
    except Exception:
        pass
    print(f"[kt-plan] {len(pts)} forwards carried from an earlier launch (up to {MEASURED['max_rows']} rows): fixed "
          f"{FIXED >> 20} MiB, {ROW / 1024:.1f} KiB a row, {SEQ:.1f} B a sequence token, {XS * 1e3:.2f} mB a row a "
          f"sequence token ({SEQ_WHY})", flush=True)


def _drop_carried(why: str = "") -> None:
    """The carried forwards leave the fit: this launch's own stand alone, and the envelope doubles from the largest
    of them (online_cap), as on a launch with nothing carried."""
    global FIXED, ROW, SEQ, LP, XS
    if not MEASURED.get("carried"):
        return
    own = int(MEASURED.get("own", 0))
    pts = MEASURED.get("points", [])
    pts[:] = pts[len(pts) - own:] if own > 0 else []
    lp = MEASURED.get("lp_points", [])
    own_lp = int(MEASURED.get("own_lp", 0))
    lp[:] = lp[len(lp) - own_lp:] if own_lp > 0 else []
    MEASURED["carried"] = 0
    MEASURED["dropped"] = True
    MEASURED["windows"] = len(pts)
    MEASURED["max_rows"] = int(max([p[0] for p in pts] or [0]))
    if pts:
        FIXED, ROW, SEQ = _fit(pts)
        MEASURED["fitted"] = len({p[0] for p in pts}) >= 2 and ROW < max(p[2] / p[0] for p in pts)
    else:
        FIXED, ROW, SEQ, XS = 0, 0.0, 0.0, 0.0
        MEASURED["fitted"] = False
    MEASURED["row"] = ROW
    LP = _lp_row(lp) if lp and pts else 0.0
    print(f"[kt-plan] the carried forwards dropped{': ' + why if why else ''}; {len(pts)} of this launch stand, "
          f"the largest {MEASURED['max_rows']} rows", flush=True)


def observe_window(rows: int, seq: int, grown: int, alloc: int = 0, window: int = 0, lp_rows: int = 0) -> None:
    """A closed prefill window (kt_lend.end_window): its rows and how many bytes torch's allocation grew
    by. Under ONLINE the plan's cost a row is the largest growth a row seen; the fixed and sequence
    terms stay 0 and the margin covers them. A window may grow to twice the largest one closed so far:
    what a longer sequence costs a row (an indexer scored over the prefix) enters the envelope one
    doubling at a time instead of one extrapolation (a 4096-row first window planned a 1M-context Vision
    into an OOM at once, 11-05)."""
    global FIXED, ROW, SEQ, LP
    if not ONLINE or rows <= 0 or grown <= 0:
        return
    if MEASURED.get("carried") and lp_rows <= 0:
        said = FIXED + ROW * rows + SEQ * seq + XS * rows * seq
        if grown > said + max(SEQ_TOL, 0.02 * said):
            MEASURED["drop"] = (f"{rows} rows at sequence {seq} grew {grown >> 20} MiB, the carried fit said "
                                f"{int(said) >> 20} MiB")
    MEASURED["windows"] += 1
    MEASURED["max_rows"] = max(MEASURED["max_rows"], rows)
    if MEASURED.get("carried"):
        MEASURED["own_lp" if lp_rows > 0 else "own"] = min(64, int(MEASURED.get("own_lp" if lp_rows > 0 else "own", 0)) + 1)
    if os.environ.get("KT_PREFILL_PLAN_DEBUG") == "1" and lp_rows > 0:
        print(f"[kt-plan] window {window}: {rows} rows at sequence {seq}, {lp_rows} with input logprobs, grew "
              f"{grown >> 20} MiB ({alloc >> 20} MiB allocated)", flush=True)
    # forwards that computed input logprobs are not a plain row's cost: 668-row scored chat pieces
    # grew 1.9 GB against 0.95 GB for a 1024-row prefill chunk, and a 35K prompt after them was planned at 2.9 MB a
    # row, 1024-row chunks. They price the logprob rows on top of the plain fit; until a plain forward is
    # seen they stand in for it, as before
    if lp_rows > 0:
        lp_pts = MEASURED.setdefault("lp_points", [])
        lp_pts.append((float(rows), float(seq), float(grown), float(lp_rows)))
        del lp_pts[:-64]
        if MEASURED.get("points"):
            LP = _lp_row(lp_pts)
            return
    pts = MEASURED.setdefault("points", [])
    pts.append((float(rows), float(seq), float(grown), float(alloc), int(window), float(lp_rows)))
    del pts[:-64]
    if os.environ.get("KT_PREFILL_PLAN_DEBUG") == "1":  # each forward the fit learns from
        print(f"[kt-plan] window {window}: {rows} rows at sequence {seq}, grew {grown >> 20} MiB "
              f"({alloc >> 20} MiB allocated)", flush=True)
    fixed, row, sq = _fit(pts)
    # per forward. A fit stands once the points hold two sizes.
    MEASURED["fitted"] = MEASURED.get("fitted", False) or (
        len({p[0] for p in pts}) >= 2 and row < max(p[2] / p[0] for p in pts))
    if lp_rows <= 0 and MEASURED.get("lp_points") and all(len(p) > 5 and p[5] > 0 for p in pts[:-1]):
        # the first plain forward: the logprob forwards that stood in for it leave the plain fit
        pts[:] = pts[-1:]
        fixed, row, sq = _fit(pts)
    if (fixed, row, sq) != (FIXED, ROW, SEQ):
        FIXED, ROW, SEQ = fixed, row, sq
        MEASURED["row"] = row
    if lp_rows <= 0 and MEASURED.get("lp_points"):
        LP = _lp_row(MEASURED["lp_points"])
        print(f"[kt-plan] cost from {len(pts)} windows: fixed {fixed >> 20} MiB, {row / 1024:.1f} KiB a row, "
              f"{sq:.1f} B a sequence token, {XS * 1e3:.2f} mB a row a sequence token ({SEQ_WHY})", flush=True)


def _fit(pts):
    """the fit. FIXED + ROW * rows + SEQ * seq by least squares over the closed windows,
    then FIXED raised until no window grew more than the line says (an envelope from above). One window, or
    windows of one size, cannot separate the parts: all of the growth is then a row's, as before. A negative
    sequence term is dropped, a row term that is not positive falls back to the largest growth a row."""
    import torch

    worst = max(g / r for r, _, g in pts)
    if len({r for r, _, _ in pts}) < 2:
        return 0, worst, 0.0
    A = torch.tensor([[1.0, r, s] for r, s, _ in pts], dtype=torch.float64)
    y = torch.tensor([g for _, _, g in pts], dtype=torch.float64)
    row = sq = None
    if len({s for _, s, _ in pts}) >= 2 and len(pts) >= 3:
        sol = torch.linalg.lstsq(A, y.unsqueeze(1)).solution.squeeze(1)
        if float(sol[1]) > 0 and float(sol[2]) >= 0:
            row, sq = float(sol[1]), float(sol[2])
    if row is None:
        sol = torch.linalg.lstsq(A[:, :2], y.unsqueeze(1)).solution.squeeze(1)
        row, sq = float(sol[1]), 0.0
    if row <= 0:
        return 0, worst, 0.0
    fixed = max(0.0, max(g - row * r - sq * s for r, s, g in pts))
    return int(fixed), row, sq


def _sync_measured(scheduler) -> None:
    """The online cost terms as their maximum over the TP ranks (every rank reaches chunk_for alike)."""
    global FIXED, ROW, SEQ, XS, LP
    import torch

    group = getattr(scheduler, "tp_cpu_group", None)
    if group is None or not torch.distributed.is_initialized():
        return
    def terms():
        return torch.tensor([float(FIXED), float(ROW), float(SEQ), float(bool(MEASURED.get("fitted"))),
                             float(MEASURED["max_rows"]), float(MEASURED["windows"]), float(XS), float(LP),
                             float(bool(MEASURED.get("drop")))], dtype=torch.float64)

    t = terms()
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX, group=group)
    if t[8] > 0:
        # a rank's forward outgrew the carried fit: every rank drops the carried forwards, then the terms again
        _drop_carried(MEASURED.pop("drop", None) or "another rank's forward outgrew the carried fit")
        t = terms()
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX, group=group)
    FIXED, ROW, SEQ = int(t[0]), float(t[1]), float(t[2])
    MEASURED["fitted"] = bool(t[3] > 0)
    MEASURED["max_rows"] = int(t[4])
    MEASURED["windows"] = int(t[5])
    XS = float(t[6])
    LP = float(t[7])


def _lp_cost(req) -> float:
    """A row's logprob cost for a request that asked for its input logprobs (LP), else 0."""
    if LP <= 0 or not getattr(req, "return_logprob", False):
        return 0.0
    start = int(getattr(req, "logprob_start_len", -1) or 0)
    n = len(getattr(req, "origin_input_ids", ()) or ())
    return LP if 0 <= start < n - 1 else 0.0


def online_cap(chunk: int) -> int:
    """The chunk the online envelope allows now: twice the largest closed window. This launch's only: seeded with an
    earlier launch's largest window, the first prompts ran in fewer, larger windows, the sequence term was fitted
    from them at half its size, and a 131K prompt on DeepSeek V4-Flash-Vision outgrew its windows and stopped the
    server (11-09)."""
    if not ONLINE or MEASURED["max_rows"] <= 0:
        return chunk
    return max(ALIGN, min(chunk, 2 * MEASURED["max_rows"]))
STATS = {"plans": 0, "last": None}
# what a request's own forward puts back on the GPU before its rows run, by name (bytes):
# a model that keeps its vision tower on the host between images registers it
#, and a request with images is planned with that
# much less room
ON_DEMAND = {}


def on_demand(name: str, nbytes: int) -> None:
    ON_DEMAND[name] = int(nbytes)


def _on_demand_for(req) -> int:
    if not ON_DEMAND:
        return 0
    mm = getattr(req, "multimodal_inputs", None)
    if mm is None or not getattr(mm, "mm_items", None):
        return 0
    return sum(ON_DEMAND.values())


# room kept free in a planned window (MiB); unset: the lend's own margin. The
# fitted cost already lies above every recorded forward (its envelope), so this
# is the second allowance, not the only one
_PLAN_MARGIN = os.environ.get("KT_PREFILL_PLAN_MARGIN_MB")


def _margin() -> int:
    if _PLAN_MARGIN not in (None, ""):
        return int(float(_PLAN_MARGIN) * (1 << 20))
    from sglang.srt.layers.moe import kt_lend

    return kt_lend.MARGIN


def _ready() -> bool:
    return ENABLED and ROW > 0 and (FREE > 0 or LIVE)


# KT_PREFILL_KV_LEND_FORCE=1: every window lends whatever the plan says (for checks
# such as KT_PREFILL_KV_LEND_VERIFY, which see only windows that lend)
FORCE_LOAN = os.environ.get("KT_PREFILL_KV_LEND_FORCE", "0") == "1"
NEED = {}  # rid -> the plan needs the KV loan (decided on the all-reduced free memory)
# rid -> the bytes the plan's chunk needs past the free memory (the loan's part), None when the loan is taken whole
# (a chunk past the measured one, whose cost is extrapolated)
SHORT = {}
PLAN = {}  # rid -> the chunk planned when the request was first taken
# rid -> the free memory the plan's chunk was fitted in (its cost, the margin, what comes back on demand), on the base
# _free_now reads; None where the cost is extrapolated. The hot lend's slots that stay through a window (kt_lend)
WANT = {}
ROWS_UP = set()  # rids whose prompt rows went to the card at their plan (host KV, kt_kv_vmm.prompt_rows)
# rid -> [bytes of its rows the plan set aside that the card could not take yet, later chunks left to try at]: the
# weights were resident when the prompt was planned and are out once its window is open
ROWS_LATER = {}
# rid -> (the bytes the plan's chunk was fitted to cost at the prompt's end, its carry, its logprob bytes a row): what
# each window's own chunk is sized in (_window_chunk). Not kept where the sliding-window pool's lent pages are in the
# cost (their room is set for the plan's chunk)
WIN = {}
WIN_BASE = {}  # rid -> (WANT, SHORT) as the plan set them: a window whose chunk costs more raises them by it
ISSUED = {}  # rid -> the largest chunk a window of the request was given (chunk_in_flight)
WIN_SAID = {}


STEP = [ALIGN]  # the step the chunks even out in: the scheduler's page once a plan has read it


def _even(rem: int, n: int) -> int:
    """rem rows in n chunks of one size, in whole STEP rows."""
    return -(-math.ceil(rem / n) // STEP[0]) * STEP[0]


def _no_tail(c: int, rem: int) -> int:
    """c, or half of what is left when c would leave fewer rows than a streamed forward takes: that tail runs as a
    decode forward, with the weights back on the card and no room for its rows. Windows of their own sizes left 320
    of a 131K prompt's rows behind 5,056-row chunks, and they were written in host memory (Flash-Next NVFP4 on one
    card, 11-13). Never more rows than c."""
    if c < rem < c + max(ALIGN, _stream_from()):
        return min(c, _even(rem, 2))
    return c


def _window_chunk(req, rem: int, done: int, cap: int, scheduler=None) -> int:
    """The chunk of one window of a planned request. The plan's chunk is the one that fits at the prompt's end, where
    a row costs most (the term a row a sequence token); a window nearer the start runs more rows in the same bytes.
    Equal chunks of the size the end allows ran a 131K prompt in 26 forwards of 5,120 rows, a row costing 423 KiB
    at the start and 857 KiB at the end (Flash-Next NVFP4 on one card, 11-12). Each window takes the most rows
    whose cost at its own end is within the bytes the plan fitted, evened over the chunks left at that size; never
    fewer than the plan's. Nothing live is read (the previous forward may still hold its activations): the same
    on every rank.
    The prompt's rows the card could not take at its plan (ROWS_LATER, host KV) come window by window, each
    window's own: what the plan set aside for them and no window has used yet is the window's to run rows in, less
    the most its own tokens' rows can take. Set aside whole and moved whole at the second chunk, a 131K prompt
    kept 2.37 GiB out of its windows for rows of pages it never wrote (sglang evicted the idle prompts as it went
    and gave it their pages, on the card already), and ran 43 chunks of 3,072 rows in a 6.8 GiB room (Flash-Next
    NVFP4 on one card, 11-13)."""
    rid = req.rid
    base = int(PLAN[rid])
    w = WIN.get(rid)
    e = ROWS_LATER.get(rid) if scheduler is not None else None
    if e:
        from sglang.srt.layers.moe import kt_kv_vmm, kt_lend

        if kt_lend.STATE.get("phase") != "prefill":
            e = None
    if w is None or cap <= base or rem <= base:
        c = _no_tail(min(cap, base), rem)
        if e:
            _rows_later(req, rem)
            e = None
    else:
        budget, carry, lpc = w
        top = int(MEASURED.get("max_rows", 0) or 0) if ONLINE else 0
        spare = int(e[0]) if e else 0
        bound = getattr(kt_kv_vmm, "prompt_rows_bound", None) if e else None

        def cost(x: int) -> float:
            past = max(0, x - top) if top > 0 else 0  # as the plan counts them (fits)
            s = done + min(x, rem)
            return FIXED + carry + (ROW + lpc) * (x + past) + SEQ * s + XS * (x + past) * s

        # the rows the largest window would bring, the most over the ranks: a window holds back no more than that.
        # The bound of the tensors' shapes alone kept some 700 MiB out of every window for rows of 0 to 192 MiB
        # (the pages a prompt is given are mostly on the card already), and the later windows fell to the plan's
        # chunk (Flash-Next NVFP4 on one card, 11-13)
        prefix = getattr(req, "prefix_indices", None)
        need_of = getattr(kt_kv_vmm, "prompt_rows_need", lambda t, p=None: 0) if e else None
        most = _max_over_ranks(scheduler, need_of(min(cap, rem), prefix)) if e else 0

        def rows(x: int) -> int:
            return min(spare, most, int(bound(min(x, rem)))) if bound is not None else min(spare, most)

        c = base
        while c + ALIGN <= cap and c < rem and cost(c + ALIGN) + rows(c + ALIGN) <= budget + spare:
            c += ALIGN
        n = max(1, math.ceil(rem / c))
        even = _even(rem, n)
        c = _no_tail(max(base, min(c, even)), rem)
        if e:
            # this window's rows: the chunks of the pages its tokens are given that the card does not hold, the
            # most over the ranks, within what is left of the plan's share
            cap_c = int(bound(min(c, rem))) if bound is not None else spare
            need = max(0, min(spare, cap_c, _max_over_ranks(scheduler, need_of(min(c, rem), prefix))))
            while c - ALIGN >= base and cost(c) + need > budget + spare:  # the chunk's own rows are more than `most`
                c -= ALIGN
            c = _no_tail(c, rem)
            over = max(0, int(cost(c) - budget))
            if os.environ.get("KT_PREFILL_PLAN_DEBUG") == "1":
                print(f"[kt-plan] window rows for {str(rid)[:8]}: {need >> 20} MiB of {spare >> 20} MiB set aside "
                      f"(the largest window's {most >> 20}), chunk {c} against the plan's {base}, {over >> 20} MiB past "
                      f"its cost", flush=True)
            if need > 0:
                getattr(kt_kv_vmm, "prompt_rows", lambda t, b, p=None: 0)(min(c, rem), need, prefix)
                e[0] = spare - need
                if e[0] <= 0:
                    ROWS_LATER.pop(rid, None)
            # the lend frees what the window runs in (window_want, loan_bytes): its chunk's cost past the plan's
            b = WIN_BASE.setdefault(rid, (WANT.get(rid), SHORT.get(rid)))
            if b[0] is not None:
                WANT[rid] = int(b[0]) + over
            if b[1] is not None:
                SHORT[rid] = int(b[1]) + over
        if c != base and WIN_SAID.get(rid) != c:
            WIN_SAID[rid] = c
            print(f"[kt-prefill-plan] a window of {c} rows at {done} of {done + rem} tokens (the plan's chunk "
                  f"{base} is the one that fits at the end), {max(1, math.ceil(rem / c))} more at this size",
                  flush=True)
    ISSUED[rid] = max(int(ISSUED.get(rid, 0)), c)
    return c


def _rows_later(req, rem: int) -> None:
    """A later chunk of a prompt whose rows did not all fit at its plan: the rest come to the card now, in the
    window, within what the plan took out of the room for them. Decode's card had 2,099 MiB for the 2,921 MiB a 131K
    prompt's rows take; 122 of its 432 chunks stayed in host memory and 21 of its 22 forwards wrote there, with
    the weights out and the room set aside for them unused (Flash-Next NVFP4 on one card, 11-12). The same chunks
    try on every rank (host_in_flight exchanges counts); the weights' return sends back what decode cannot hold
    (kt_lend._check_room)."""
    from sglang.srt.layers.moe import kt_kv_vmm, kt_lend

    e = ROWS_LATER.get(req.rid)
    if not e or kt_lend.STATE.get("phase") != "prefill":
        return
    e[1] -= 1
    if e[1] <= 0:
        ROWS_LATER.pop(req.rid, None)
    getattr(kt_kv_vmm, "prompt_rows", lambda t, b, p=None: 0)(rem, e[0], getattr(req, "prefix_indices", None))


def _stream_from() -> int:
    from sglang.srt.layers.moe import kt_stream_threshold

    return kt_stream_threshold.get()


def chunk_in_flight(req, default: int) -> int:
    """sglang's chunk-cache SWA eviction under the overlap scheduler: the chunk still
    in flight when the next is scheduled is the one planned for this request, not
    the launch chunk (the cap); with the cap, pre_len - 36,864 left a 35K request
    holding every chunk's SWA until it ended."""
    if not ENABLED:
        return default
    rid = getattr(req, "rid", None)
    # the largest window the request was given: its windows differ in size (_window_chunk), and holding a chunk's
    # SWA longer than needed is the safe side
    return int(ISSUED.get(rid) or PLAN.get(rid, default) or default)


def needs_loan(fb) -> bool:
    """kt_kv_prefill_lend.lend: whether the window's requests were planned with the
    loan. The plan decided it from values every rank shares, so every rank lends
    or none does; a request the plan never saw is lent for."""
    if not _ready() or FORCE_LOAN:
        return True
    rids = getattr(fb, "rids", None) or []
    return not rids or any(NEED.get(r, True) for r in rids)


def loan_bytes(fb):
    """kt_kv_prefill_lend.lend: the bytes the window's requests were planned to borrow, the most over them; None
    when the loan is taken whole (no plan, a request the plan never saw, or a chunk whose cost is extrapolated).
    From values every rank shares, as needs_loan."""
    if not _ready() or FORCE_LOAN:
        return None
    rids = getattr(fb, "rids", None) or []
    if not rids:
        return None
    most = 0
    for r in rids:
        s = SHORT.get(r, None) if r in SHORT else None
        if s is None:
            return None
        most = max(most, int(s))
    return most


def window_want(fb):
    """kt_lend, at a window's start and before each of its chunks: the free memory the window's requests were
    planned in, the most over them; None when the plan cannot say (no plan, a request it never saw, a chunk whose
    cost is extrapolated): then every hot slot goes, as the loan is taken whole. From values every rank shares."""
    if not _ready() or FORCE_LOAN:
        return None
    rids = getattr(fb, "rids", None) or []
    if not rids:
        return None
    most = 0
    for r in rids:
        w = WANT.get(r)
        if w is None:
            return None
        most = max(most, int(w))
    return most


def kept_share(fb, kept: int) -> int:
    """kt_lend._hot_topup: the part of the scratch earlier windows grew (kept, mapped at every start since) that the
    cost of this window's chunk counts. The largest chunk measured grew it, and a forward's cost was recorded with
    the scratch it needed, so a smaller chunk's cost holds its rows' share of it and no more: a 2,048-row chunk
    planned at 1.2 GiB over 1.92 GiB kept left the forward 0.15 GiB (11-11)."""
    rows = max((int(PLAN.get(r, 0) or 0) for r in (getattr(fb, "rids", None) or [])), default=0)
    most = int(MEASURED.get("max_rows", 0) or 0) if ONLINE else 0
    if kept <= 0 or rows <= 0 or most <= 0 or rows >= most:
        return max(0, int(kept))
    return int(kept) * rows // most


def _remaining(req):
    """(tokens to prefill, tokens already in the cache) for the request."""
    total = len(getattr(req, "origin_input_ids", None) or []) + len(getattr(req, "output_ids", None) or [])
    done = int(getattr(req, "prefix_len", 0) or 0)
    return max(0, total - done), done


def _free_now(scheduler) -> int:
    """Free memory a window would start with, the smallest over the TP ranks."""
    if not LIVE:
        return FREE
    import torch

    from sglang.srt.layers.moe import kt_lend

    # the ledger's free memory alone: in decode at the next window's start, in a window for its
    # chunks, with what earlier windows kept in the scratch added back (the start maps it, and the forwards' fitted
    # cost includes it). The decode-to-window step (pf_delta, or KT_PREFILL_PLAN_DELTA_MB from an earlier launch),
    # the in-window reading and the hot lend's spare are the ledger's holders now. Before a window start has been
    # measured: the constant (FREE)
    from sglang.srt.layers.moe import kt_ledger

    led = kt_ledger.window_free(torch.cuda.current_device())
    mine = -1 if led is None else led + int(kt_lend.STATE.get("scratch_kept", 0) or 0)
    t = torch.tensor([mine], dtype=torch.int64)
    group = getattr(scheduler, "tp_cpu_group", None)
    if group is not None and torch.distributed.is_initialized():
        # every rank reaches this call under the same conditions (the queue and
        # the allocator are the same on every rank); a rank without a measured
        # step sends -1 and all of them fall back to the constant
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN, group=group)
    v = int(t.item())
    return v if v >= 0 else FREE


def _max_over_ranks(scheduler, v: int) -> int:
    """v reduced by MAX over the TP ranks (the scheduler's CPU group), as _free_now reduces the free memory."""
    import torch

    t = torch.tensor([int(v)], dtype=torch.int64)
    group = getattr(scheduler, "tp_cpu_group", None)
    if group is not None and torch.distributed.is_initialized():
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX, group=group)
    return int(t.item())


def chunk_for(scheduler, chunk: int) -> int:
    """Scheduler.get_new_batch_prefill: the chunk for the batch it is about to make."""
    from sglang.srt.layers.moe import kt_kv_prefill_lend

    getattr(kt_kv_prefill_lend, "bind_tree", lambda s: None)(scheduler)
    if os.environ.get("KT_PREFILL_PLAN_DEBUG") == "2":
        q = getattr(scheduler, "waiting_queue", None)
        print(f"[kt-plan] chunk_for({chunk}): chunked {scheduler.chunked_req is not None}, queue {type(q).__name__} "
              f"of {len(q) if q is not None else None}, ready {_ready()}, windows {MEASURED['windows']}, row {ROW}",
              flush=True)
    if not chunk or chunk <= 0:
        return chunk
    # the first window reaches the planner before anything is measured
    if not _ready() and not (ENABLED and ONLINE and MEASURED["windows"] == 0):
        return chunk
    req = scheduler.chunked_req
    if req is None:
        q = scheduler.waiting_queue
        req = q[0] if q else None
    if req is None:
        return chunk
    if req is not scheduler.chunked_req:
        # a queued request: the prefix the radix cache already holds is not work
        # (the same read-only match the scheduling policy makes; the adder
        # matches again when it takes the request)
        try:
            from sglang.srt.mem_cache.common import match_kv_cache

            # under the bound the adder's own match has (Req.init_next_round_input): a prompt the cache holds whole
            # still prefills its last token, and a cache that resumes only where it saved a state (a recurrent
            # layer's checkpoints) then resumes a chunk back. Unbounded, a repeated 131K prompt read as nothing to
            # prefill, went unplanned, and its 13,312 rows were written in host memory (11-12)
            ids = req.origin_input_ids + req.output_ids
            bound = getattr(req, "_compute_max_prefix_len", None)
            match_kv_cache(req, scheduler.tree_cache, ids, max_prefix_len=bound(len(ids)) if bound else None)
        except Exception as e:
            _say_once("prefill plan: the cached prefix of a waiting request was not matched", e)
    rem, done = _remaining(req)
    if os.environ.get("KT_PREFILL_PLAN_DEBUG") == "1" and (rem > 2048 or done > 2048):
        print(f"[kt-plan] chunk_for {str(req.rid)[:8]}: {'a later chunk' if req is scheduler.chunked_req else 'queued'}"
              f", {rem} to prefill after {done}, planned {req.rid in PLAN}, rows up {req.rid in ROWS_UP}, "
              f"stream from {_stream_from()}", flush=True)
    if req is not scheduler.chunked_req:
        # no SWA bound until this request is planned (a chunked one keeps its plan's)
        getattr(kt_kv_prefill_lend, "swa_room", lambda t, w="": None)(None)
    if rem <= 0:
        return chunk
    if ONLINE:
        _carry_in()
        _sync_measured(scheduler)  # same on every rank
        if MEASURED.get("drop"):  # one rank, no group to tell
            _drop_carried(MEASURED.pop("drop"))
    if ONLINE and MEASURED["windows"] == 0:
        # nothing measured yet: the lend's own chunk, the first window measures
        MEASURED["blind"] = min(chunk, SAFE) if SAFE > 0 else chunk
        if os.environ.get("KT_PREFILL_PLAN_DEBUG") == "1":
            print(f"[kt-plan] first window allowed {MEASURED['blind']} rows unmeasured", flush=True)
        return MEASURED["blind"]
    if not _ready():
        return chunk
    if (ONLINE and _lp_cost(req) == 0
            and all(len(p) > 5 and p[5] > 0 for p in MEASURED.get("points", []))):
        # only forwards with input logprobs measured: a plain prompt's first window is as unmeasured as a fresh
        # server's, and its chunks get what that one's did
        return MEASURED.get("blind") or (min(chunk, SAFE) if SAFE > 0 else chunk)
    capped_at = online_cap(chunk)
    capped = capped_at < chunk
    chunk = capped_at
    if (req is scheduler.chunked_req and req.rid in PLAN and not (ONLINE and not MEASURED.get("fitted"))
            and CAPPED.get(req.rid) != PLAN[req.rid]):
        # a later chunk keeps the request's plan: with the overlap scheduler this
        # runs while the previous chunk's forward still holds its activations, so
        # the live free memory would be low by them
        return _window_chunk(req, rem, done, chunk, scheduler)
    from sglang.srt.layers.moe import kt_kv_prefill_lend

    free = _free_now(scheduler)
    if free <= 0 and os.environ.get("KT_PREFILL_PLAN_DEBUG") == "1":
        print(f"[kt-plan] chunk_for {str(req.rid)[:8]}: no free memory read ({free}), unplanned", flush=True)
    if free <= 0:
        return min(chunk, SAFE) if SAFE > 0 else chunk
    # in a window the loan is out already, in its free memory
    from sglang.srt.layers.moe import kt_lend

    # (a hot lend standing through the window gives its slots to a chunk that needs them: kt_lend._hot_topup)
    loan = (-_max_over_ranks(scheduler, -kt_lend.hot_standing()) if kt_lend.STATE.get("phase") == "prefill"
            else kt_kv_prefill_lend.loan_estimate(scheduler, rem))
    back = _on_demand_for(req)
    room = free + loan - _margin() - back
    if req is not scheduler.chunked_req and kt_lend.STATE.get("phase") == "prefill":
        # a prompt that comes into an open window: the pages the window lent come back mapped for it
        room -= _max_over_ranks(scheduler, getattr(kt_kv_prefill_lend, "window_back", lambda rem: 0)(rem))
    # a new prompt's KV rows the card does not hold yet (host KV) come out of the room before its chunk is chosen:
    # moved after it from what the chunk left, a 131K prompt found the room spent and wrote its KV past the card's
    # 145,555 rows in host memory (Gemma 4, 258 s against 56 s, 11-09). The largest need over the ranks, so every
    # rank plans one chunk; never more than leaves the smallest chunk its cost
    kv_need = idle = kv_later = 0
    new_prompt = (req is not scheduler.chunked_req and req.rid not in ROWS_UP
                  and rem >= _stream_from())  # a forward of exactly that many rows streams (kt_lend._streamed)
    if new_prompt:
        from sglang.srt.layers.moe import kt_kv_vmm

        kv_need = kv_asked = _max_over_ranks(scheduler, getattr(kt_kv_vmm, "prompt_rows_need", lambda t, p=None: 0)(
            rem, getattr(req, "prefix_indices", None)))
        # what the rows of prompts no request reads give back as these come in (the least over the ranks)
        idle = max(0, -_max_over_ranks(scheduler, -getattr(kt_kv_vmm, "prompt_rows_idle", lambda: 0)()))
        kv_need = max(0, min(kv_need, int(room + idle - (FIXED + (ROW + _lp_cost(req)) * ALIGN))))
        if os.environ.get("KT_PREFILL_PLAN_DEBUG") == "1":
            print(f"[kt-plan] rows for {req.rid[:8]}: asked {kv_asked >> 20} MiB, room {int(room) >> 20} MiB = free "
                  f"{int(free) >> 20} + loan {int(loan) >> 20} - margin {_margin() >> 20} - on demand {int(back) >> 20}"
                  f", smallest chunk {int(FIXED + (ROW + _lp_cost(req)) * ALIGN) >> 20}, dec_free "
                  f"{int(kt_lend.STATE.get('dec_free', 0) or 0) >> 20}, rows in window "
                  f"{int(kt_lend.STATE.get('rows_in_window', 0) or 0) >> 20}", flush=True)
        if kv_need and kt_lend.STATE.get("phase") == "prefill":
            # inside a window the free memory has the lent weights in it: rows moved now must still fit decode's
            # memory when they come back (decode's free memory at the window's start, less what earlier prompts of
            # this window took), or the window cannot close (V4-Flash-Vision stopped there at 131K, 11-09)
            # (the hot slots that stayed through the window give theirs at its end: kt_lend._check_room)
            left = int(kt_lend.STATE.get("dec_free", 0) or 0) + kt_lend.hot_standing() - kt_kv_vmm.margin_standing() - int(
                kt_lend.STATE.get("rows_in_window", 0) or 0)
            kv_need = max(0, min(kv_need, -_max_over_ranks(scheduler, -left) + idle))
        elif kv_need:
            # in decode the rows go to the card as it stands, its weights resident: what it does not take stays
            # set aside in the window's room and comes at the prompt's next chunks (_rows_later)
            now = max(0, min(kv_need, -_max_over_ranks(scheduler, -kt_kv_vmm.prompt_rows_room()) + idle))
            kv_later, kv_need = kv_need - now, now
        room -= max(0, kv_need - idle) + kv_later
    # the sliding-window pool's lent pages a request's chunks map back
    # (KT_PREFILL_SWA_LEND): nothing up to the mapped free ones, then a token's SWA
    # bytes a row, for every row the request may hold at once
    swa = getattr(kt_kv_prefill_lend, "swa_terms", lambda: None)()
    if swa is not None and len(swa) < 4:
        swa = tuple(swa) + (1,)

    def swa_cost(c: int) -> int:
        if swa is None:
            return 0
        # one chunk holds its own rows; with more, the chunks held at once (swa[3]:
        # two under the overlap scheduler, whose next chunk is allocated before the
        # last one's slots are freed). Not the whole remaining prompt: a 512K prompt
        # in 2,048-row chunks held 6% of a 77K-token pool throughout (2.25 chunks),
        # 7,168-row chunks 19% (2.04), 18,432-row 46% (1.92) (10-18), while charging
        # the remaining prompt planned 11.5 GB and kept a 512K prefill at 2,048 rows.
        need = c if rem <= c else c * max(1, int(swa[3]))
        return 0 if need <= swa[0] else int((need - swa[0]) * swa[1] + swa[2])

    lpc = _lp_cost(req)

    def fits(c: int, n: int) -> bool:
        """The last chunk ends at the longest sequence and, when there is more than
        one, carries what the earlier ones left: it is the one that must fit."""
        carry = CARRY if n > 1 else 0
        # the rows past the largest forward measured count twice: the fit is a line through smaller forwards, and
        # what it misses there is not in its envelope. a 131K calibration prompt went from 22,528-row
        # forwards to 32,768 at 5,218 MiB of a 5,526 MiB room and ran the card out in its second chunk (11-11);
        # counted twice it takes 27,648, and the next step up is planned from a forward of that size
        past = max(0, c - MEASURED["max_rows"]) if ONLINE and MEASURED["max_rows"] > 0 else 0
        return (FIXED + carry + (ROW + lpc) * (c + past) + SEQ * (done + rem) + XS * (c + past) * (done + rem)
                + swa_cost(c) <= room)

    # the fewest chunks whose equal size, in whole ALIGN steps, fits (the step
    # rounds a chunk up: 35,000 in three is 11,667 rows, 12,288 once aligned)
    # over the launch chunk by up to TAIL_STEPS aligned steps when that spares a chunk (a
    # chunk of the streamed prefill sweeps every expert once; a tail chunk pays the same)
    slack = chunk + ALIGN * TAIL_STEPS
    for n in range(1, 257):
        c = math.ceil(rem / n)
        c = max(ALIGN, -(-c // ALIGN) * ALIGN)
        if not TAIL_STEPS:
            c = min(chunk, c)
        elif c > slack:
            continue
        elif c > chunk and math.ceil(rem / chunk) <= n:
            c = chunk  # over the launch chunk without sparing a chunk: the launch chunk
        if fits(c, n) or c <= ALIGN:
            break
    else:
        if TAIL_STEPS:
            c = min(chunk, ALIGN)
    n = max(1, math.ceil(rem / c))  # the launch chunk may cap it (V4.1's SWA pool)
    # the chunks even out in the scheduler's own step (a page of the pool; it cuts a chunk to whole pages), not in
    # ALIGN steps: 35,000 in two was 18,432 and 16,568, the 1,024-row step having rounded the half up and left the
    # last window short by it (11-13). A smaller chunk fits where the larger did
    STEP[0] = max(1, int(getattr(scheduler, "page_size", 0) or getattr(getattr(scheduler, "tree_cache", None),
                                                                        "page_size", 0) or 1))
    if n > 1:
        c = min(c, _even(rem, n))
    if swa is not None and swa[1] > 0:
        # the SWA tokens the room holds past the chunk's own cost, at least the chunk's rows
        # held at once plus the window and two pages (sglang's need for it)
        own = FIXED + (CARRY if n > 1 else 0) + (ROW + lpc) * c + SEQ * (done + rem) + XS * c * (done + rem)
        held = c if rem <= c else c * max(1, int(swa[3]))
        win = int(getattr(getattr(scheduler, "tree_cache", None), "sliding_window_size", 0) or 0)
        ps = int(getattr(kt_kv_prefill_lend, "SWA", {}).get("ps", 1) or 1)
        tokens = max(0, swa[0]) + int(max(0, room - own - swa[2]) // swa[1])
        kt_kv_prefill_lend.swa_room(max(tokens, held + win + 2 * ps), f"{rem} tokens in chunks of {c}")
    if new_prompt:
        # the prompt's rows to the card in what the room set aside for them, before the chunk runs (the free memory
        # the loan test below reads is less by them, the same on every rank)
        from sglang.srt.layers.moe import kt_kv_vmm

        ROWS_UP.add(req.rid)
        if kv_later > 0 and n > 1:
            ROWS_LATER[req.rid] = [kv_later, 2]
            free -= kv_later
        if os.environ.get("KT_PREFILL_PLAN_DEBUG") == "1":
            print(f"[kt-plan] rows for {req.rid[:8]}: phase {kt_lend.STATE.get('phase')}, rem {rem}, "
                  f"set aside {kv_need >> 20} MiB, chunk {c}", flush=True)
        if kv_need > 0:
            getattr(kt_kv_vmm, "prompt_rows", lambda t, b, p=None: 0)(
                rem, kv_need, getattr(req, "prefix_indices", None))
            free -= max(0, kv_need - idle)
            if kt_lend.STATE.get("phase") == "prefill":
                kt_lend.STATE["rows_in_window"] = (int(kt_lend.STATE.get("rows_in_window", 0) or 0)
                                                   + max(0, kv_need - idle))
    if len(NEED) > 4096:
        NEED.clear()
        SHORT.clear()
        PLAN.clear()
        WANT.clear()
        CAPPED.clear()
        ROWS_UP.clear()
        ROWS_LATER.clear()
        WIN.clear()
        WIN_BASE.clear()
        ISSUED.clear()
        WIN_SAID.clear()
    PLAN[req.rid] = c
    ISSUED.pop(req.rid, None)
    WIN_BASE.pop(req.rid, None)
    WIN_SAID.pop(req.rid, None)
    if swa is None and n > 1:
        _past = max(0, c - MEASURED["max_rows"]) if ONLINE and MEASURED["max_rows"] > 0 else 0
        WIN[req.rid] = (FIXED + CARRY + (ROW + lpc) * (c + _past) + SEQ * (done + rem)
                        + XS * (c + _past) * (done + rem), CARRY, lpc)
    else:
        WIN.pop(req.rid, None)
    if capped and c >= chunk:
        CAPPED[req.rid] = c
    else:
        CAPPED.pop(req.rid, None)
    # the loan when the estimate needs it, and always past the lend's own chunk:
    # there the estimate is extrapolated, and lending costs a few tens of ms
    short = (FIXED + (CARRY if n > 1 else 0) + (ROW + lpc) * c + SEQ * (done + rem) + XS * c * (done + rem)
             + swa_cost(c) + _margin() + back - free)
    # past the largest window measured the cost is extrapolated: the loan is taken whole there (before anything is
    # measured, past the lend's own chunk)
    measured = int(MEASURED.get("max_rows", 0) or 0) if ONLINE else 0
    extrapolated = c > measured if measured > 0 else (SAFE > 0 and c > SAFE)
    NEED[req.rid] = short > 0 or extrapolated
    # what the window borrows: the shortfall, or the whole loan where the cost is extrapolated
    SHORT[req.rid] = None if extrapolated else max(0, int(short))
    WANT[req.rid] = None if extrapolated else int(short + free)
    STATS["plans"] += 1
    plan = (rem, n, c, back)
    if plan != STATS["last"]:
        STATS["last"] = plan
        print(f"[kt-prefill-plan] {rem} tokens to prefill after {done} cached: {n} chunk(s) of {c} "
              f"(free {free >> 20} MiB + loan {loan >> 20} MiB, fixed {FIXED >> 20} MiB, "
              f"{ROW / 1024:.0f} KiB a row, {SEQ:.0f} B a sequence token, {XS * 1e3:.2f} mB a row a "
              f"sequence token, carry {CARRY >> 20} MiB, "
              f"SWA mapped back {swa_cost(c) >> 20} MiB"
              + (f", {back >> 20} MiB back on the GPU for the images ({', '.join(ON_DEMAND)})" if back else "")
              + f", cap {chunk})", flush=True)
    return _window_chunk(req, rem, done, max(chunk, c))


def _fit(pts):  # non-negative (this definition replaces the ones above)
    """FIXED + ROW * rows + SEQ * seq + XS * rows * seq with no negative term: every subset of the terms that keeps
    ROW is fitted by least squares, the one with the least error among those with no negative coefficient stands,
    then FIXED is raised until no forward grew more than the fit says. Sets XS; returns (FIXED, ROW, SEQ)."""
    global XS
    import itertools

    import torch

    worst = max(g / r for r, _, g in pts)
    XS = 0.0
    if len({r for r, _, _ in pts}) < 2:
        return 0, worst, 0.0
    cols = torch.tensor([[1.0, r, s, r * s] for r, s, _ in pts], dtype=torch.float64)
    y = torch.tensor([g for _, _, g in pts], dtype=torch.float64)
    best = None
    for k in range(1, 5):
        for sub in itertools.combinations(range(4), k):
            if 1 not in sub or len(pts) < len(sub):
                continue
            A = cols[:, list(sub)]
            if int(torch.linalg.matrix_rank(A)) < len(sub):
                continue
            sol = torch.linalg.lstsq(A, y.unsqueeze(1)).solution.squeeze(1)
            if any(float(v) < 0 for v in sol) or float(sol[list(sub).index(1)]) <= 0:
                continue
            err = float(((A @ sol - y) ** 2).sum())
            if best is None or err < best[0] * (1 - 1e-9):
                coef = [0.0, 0.0, 0.0, 0.0]
                for i, v in zip(sub, sol.tolist()):
                    coef[i] = v
                best = (err, coef)
    if best is None:
        return 0, worst, 0.0
    _, (c0, row, sq, xs) = best
    XS = xs
    fixed = max(0.0, max(g - row * r - sq * s - xs * r * s for r, s, g in pts))
    return int(fixed), row, sq


# the terms on the allocated bytes, a sequence term only where a request's chunks grew
SEQ_TOL = int(os.environ.get("KT_PREFILL_PLAN_SEQ_TOL_MB", "32")) << 20
SEQ_WHY = "no request has run two equal chunks yet"


def _seq_growth(pts):
    """What the run of equal chunks over the widest stretch of sequence grew in allocated bytes from its shortest
    sequence to its longest, and whether any run had two sequences. A run is one request's chunks: the same rows and
    the same state. The widest run decides: a request's second chunk allocates more than its first and the ones
    after it no more (three 11,712-row chunks grew 230 MiB, twenty of 5,056 rows over 100K tokens 1 MiB, and the
    three put a cross term in the fit that planned a 131K prompt in 1,024-row chunks, Flash-Next on one card, 11-13)."""
    runs, key = [], None
    for p in pts:
        if len(p) < 5 or p[3] <= 0:
            key = None
            continue
        if (p[4], p[0]) != key:
            key = (p[4], p[0])
            runs.append([])
        runs[-1].append((p[1], p[3]))
    grow, seen, span = 0.0, False, -1.0
    for run in runs:
        seqs = [s for s, _ in run]
        if len(set(seqs)) < 2:
            continue
        seen = True
        if max(seqs) - min(seqs) > span:
            span, grow = max(seqs) - min(seqs), max(run)[1] - min(run)[1]
    return seen, grow


def _fit(pts):  # this definition replaces the ones above
    """FIXED + ROW * rows + SEQ * seq + XS * rows * seq. The terms are chosen on the allocated bytes: a sequence term
    only when a run of equal chunks grew past SEQ_TOL along its sequence, then with the cross term first (rows x
    sequence; SEQ alone under-stated Vision's indexer logits at a larger chunk), and the fewest terms within 16 MiB a
    forward of the least error. Their sizes are fitted on the forwards' cost, none negative, then FIXED is raised until
    no forward cost more than the fit says. Sets XS and SEQ_WHY; returns (FIXED, ROW, SEQ)."""
    global XS, SEQ_WHY
    import itertools

    import torch

    worst = max(p[2] / p[0] for p in pts)
    XS = 0.0
    if len({p[0] for p in pts}) < 2:
        return 0, worst, 0.0
    seen, grow = _seq_growth(pts)
    seq = seen and grow > SEQ_TOL
    SEQ_WHY = (f"a request's equal chunks grew {int(grow) >> 20} MiB allocated along the sequence" if seen
               else "no request has run two equal chunks yet")
    cols = torch.tensor([[1.0, p[0], p[1], p[0] * p[1]] for p in pts], dtype=torch.float64)
    cost = torch.tensor([p[2] for p in pts], dtype=torch.float64)
    alloc = torch.tensor([p[3] if len(p) > 3 and p[3] > 0 else p[2] for p in pts], dtype=torch.float64)

    def solve(sub, y):
        A = cols[:, list(sub)]
        if len(pts) < len(sub) or int(torch.linalg.matrix_rank(A)) < len(sub):
            return None
        sol = torch.linalg.lstsq(A, y.unsqueeze(1)).solution.squeeze(1)
        if any(float(v) < 0 for v in sol) or float(sol[list(sub).index(1)]) <= 0:
            return None
        return sol, float(((A @ sol - y) ** 2).sum())

    def ranked(with_seq):
        subs = [sub for k in range(1, 5) for sub in itertools.combinations(range(4), k)
                if 1 in sub and with_seq == bool({2, 3} & set(sub))]
        # a subset the allocated bytes cannot fit without a negative term ranks after those they can (the cost
        # decides: Vision's rank 1 fitted the cross term on the cost and not on the allocated bytes)
        got = [(r[1] if r is not None else math.inf, sub) for sub in subs for r in [solve(sub, alloc)]]
        # with the cross term first when the sequence counts, then the fewest terms within 16 MiB a forward of the
        # least error, then the others by terms and error
        near = min([e for e, _ in got if e < math.inf] or [0.0]) + len(pts) * float(16 << 20) ** 2
        return [t[4] for t in sorted((with_seq and 3 not in sub, e > near, len(sub), e, sub) for e, sub in got)]

    for sub in ranked(seq) + (ranked(False) if seq else []):
        r = solve(sub, cost)
        if r is None:
            continue
        coef = [0.0, 0.0, 0.0, 0.0]
        for i, v in zip(sub, r[0].tolist()):
            coef[i] = v
        _, row, sq, xs = coef
        XS = xs
        fixed = max(0.0, max(p[2] - row * p[0] - sq * p[1] - xs * p[0] * p[1] for p in pts))
        return int(fixed), row, sq
    return 0, worst, 0.0
