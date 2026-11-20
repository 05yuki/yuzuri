"""The prefill chunk planned per request (KT_PREFILL_PLAN=1, with the prefill lend
and, for the room of the free KV pages, KT_PREFILL_KV_LEND=1).

The launch chunk (--chunked-prefill-size) becomes the largest chunk the
streamer's buffers are sized for; each request then gets the fewest chunks its
remaining prompt needs and, for that count, the smallest equal chunk.

What a chunk costs is read from what this server's forwards measured (TABLE): for each chunk size, how many bytes
torch's allocation grew by at each sequence a forward of that size ended at.

  a size measured, at a sequence it was measured to   the most a forward of it grew up to that sequence
  past the sequence it was measured to                the rise the size measured furthest along a sequence had there;
                                                      past that one's own, the rise into its last third carried on
  between two sizes                                   the line between them
  under the smallest size                             its cost a row
  past the largest size                               its cost a row, or the slope from the size before it if steeper

and has
  room = free at the window's start + the free KV pages it borrows - kt_lend.MARGIN.
The free memory is live: each TP rank's decode free memory now plus the step
kt_lend measured from decode to a window's start (weights out, scratch in),
the smallest over the ranks (an all-reduce on the scheduler's TP CPU group, so
every rank plans the same chunk). Before the first window has measured that
step, KT_PREFILL_PLAN_FREE_MB stands in. The last chunk of a request is the
one with the longest sequence, so it is the one checked.

A chunk may be twice the largest one measured and no more (online_cap). The table goes to the lend's hint file as
it grows, and the next launch of the same code starts from it.

With a cost in the environment nothing is measured: KT_PREFILL_PLAN_COST="FIXED_MiB,ROW_B,SEQ_B,CARRY_MiB" is
cost(c, s) = FIXED + ROW * c + SEQ * s (+ CARRY for a chunk after the first of a window), and
KT_PREFILL_PLAN_ROW_KB alone is cost = ROW * c.
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
# the cost in the environment; measured (ONLINE), ROW is the smallest measured chunk's cost a row
FIXED, ROW, SEQ = ((int(float(_cost[0]) * (1 << 20)), float(_cost[1]), float(_cost[2]))
                   if len(_cost) >= 3 else (0, float(ROW_KB), 0.0))
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
# request gets while nothing is measured; unset, the launch chunk
SAFE = int(os.environ.get("KT_PREFILL_PLAN_SAFE", "0") or 0)
# no cost in the environment: measured from the forwards this server runs
ONLINE = not _cost and ROW_KB == 0
DEBUG = os.environ.get("KT_PREFILL_PLAN_DEBUG", "")

TABLE = {}  # rows -> [(sequence, bytes grown)] by sequence, one forward size each
LP_PTS = []  # (rows, sequence, bytes grown, rows with input logprobs): forwards that computed input logprobs
LP = 0.0  # bytes a row whose input logprobs are computed, past a plain row (vocabulary-wide logits)
REF = [None]  # the size measured furthest along a sequence, by three forwards or more
PENDING = []  # forwards closed since the last plan: (rows, sequence, grown, logprob rows)
SCRATCH = [0]  # bytes the earlier launch's windows left in the scratch: this launch's first window makes its own
MEASURED = {"windows": 0, "max_rows": 0}
_CURVE = {}
POINTS = 32  # forwards kept of one chunk size
SIZES = 48  # chunk sizes kept
# rid -> the chunk a request was planned at when the online cap bound it (capped plans)
CAPPED = {}


def _at(pts, s):
    """One measured size at sequence s: the most a forward of it grew up to s, and past its last forward the rise
    into the last third of its sequence carried on."""
    # a request's first chunks grow less than its later ones: a window near the start runs more rows
    v = max([g for q, g in pts if q <= s] or [pts[0][1]])
    last = pts[-1][0]
    if s <= last or len(pts) < 3:
        return v
    # no rise, and the size costs past its sequence what it cost. Six 22,016-row chunks of a 131K prompt grew 7,282,
    # 8,601, 8,799, 8,010, 8,784 MiB and a 524K prompt in such chunks stayed there (V4.1 on two cards, 11-18); a
    # model whose indexer scores every row against the whole prefix keeps rising (V4-Flash-Vision, 11-09)
    third = (last - pts[0][0]) / 3
    if third <= 0:
        return v
    a = max(g for q, g in pts if q <= last - third)
    b = max([g for q, g in pts if q > last - third] or [a])
    return v + max(0.0, b - a) / third * (s - last)


def _entry(r, s):
    """What a forward of the measured size r costs at sequence s. Past the sequence it was measured to it rises as
    the size measured furthest along a sequence (REF) did, a row for a row: V4.1 on one card grows 2.0 to 2.6 GiB
    over the first 65K tokens of a prompt whatever the chunk, and no more after."""
    pts = TABLE[r]
    last, ref = pts[-1][0], REF[0]
    if s <= last or ref is None or ref == r:
        return _at(pts, s)
    rise = max(0.0, _at(TABLE[ref], s) - _at(TABLE[ref], last))
    return max(g for _, g in pts) + rise * max(1.0, r / ref)


def _curve(s):
    """(rows, cost at sequence s) over the measured sizes; a larger chunk costs no less than a smaller one."""
    got = _CURVE.get(s)
    if got is None:
        if len(_CURVE) > 256:
            _CURVE.clear()
        got, top = [], 0.0
        for r in sorted(TABLE):
            top = max(top, _entry(r, s))
            got.append((r, top))
        _CURVE[s] = got
    return got


def _own(c, s, lpc=0.0):
    """What a forward of c rows that ends at sequence s costs, lpc more a row for its input logprobs."""
    if not ONLINE:
        return FIXED + (ROW + lpc) * c + SEQ * s
    curve = _curve(s)
    if not curve:
        return lpc * c
    (r0, g0), (rt, gt) = curve[0], curve[-1]
    if c <= r0:
        v = g0 * c / r0
    elif c > rt:
        # past the largest size measured: its cost a row, or the slope from the size before it where that is
        # steeper. online_cap keeps the step within twice the largest
        slope = gt / rt
        if len(curve) > 1:
            slope = max(slope, (gt - curve[-2][1]) / (rt - curve[-2][0]))
        v = gt + slope * (c - rt)
    else:
        i = next(k for k, (r, _) in enumerate(curve) if r >= c)
        (ra, ga), (rb, gb) = curve[i - 1], curve[i]
        v = ga + (gb - ga) * (c - ra) / (rb - ra)
    return v + lpc * c


cost = _own  # kt_kv_vmm: a forward's own memory


def longest() -> int:
    """The longest sequence a forward was measured at."""
    return max([pts[-1][0] for pts in TABLE.values()] or [0])


def _put(pts, seq, grown) -> None:
    """One forward into its size's list, by sequence. The list keeps what the cost reads: the first forward, each
    one that grew more than every one before it, and the last. Over POINTS, the two nearest in what they grew become
    one, at the earlier's sequence and the later's bytes (the size costs no less at any sequence than it did)."""
    pts.append((seq, grown))
    pts.sort()
    kept, top = [], -1
    for q, g in pts:
        if kept and kept[-1][0] == q:
            kept[-1] = (q, max(g, kept[-1][1]))
        elif g > top:
            kept.append((q, g))
        top = max(top, g)
    if kept[-1][0] != pts[-1][0]:
        kept.append(pts[-1])
    while len(kept) > POINTS:
        i = min(range(1, len(kept) - 2), key=lambda k: kept[k + 1][1] - kept[k][1])
        kept[i:i + 2] = [(kept[i][0], kept[i + 1][1])]
    pts[:] = kept


def _refresh() -> None:
    global ROW, LP
    _CURVE.clear()
    REF[0] = max([r for r, pts in TABLE.items() if len(pts) >= 3], key=lambda r: (TABLE[r][-1][0], r), default=None)
    MEASURED["max_rows"] = max(TABLE or [0])
    ROW = _own(ALIGN, 0) / ALIGN if TABLE else 0.0
    # forwards that computed input logprobs are not a plain row's cost: 668-row scored chat pieces grew 1.9 GB
    # against 0.95 GB for a 1,024-row prefill chunk. They price the logprob rows past the plain cost
    LP = max([0.0] + [max(0.0, g - _own(r, s)) / lp for r, s, g, lp in LP_PTS]) if TABLE else 0.0


def _fold(rows, seq, grown, lp_rows) -> None:
    MEASURED["windows"] += 1
    if lp_rows > 0:
        LP_PTS.append((rows, seq, grown, lp_rows))
        del LP_PTS[:-64]
        return
    _put(TABLE.setdefault(rows, []), seq, grown)
    while len(TABLE) > SIZES:
        # the size with the fewest forwards joins the next larger one: more rows cost no less
        sizes = sorted(TABLE)
        r = min(sizes[:-1], key=lambda x: (len(TABLE[x]), x))
        for q, g in TABLE.pop(r):
            _put(TABLE[sizes[sizes.index(r) + 1]], q, g)


def observe_window(rows: int, seq: int, grown: int, alloc: int = 0, window: int = 0, lp_rows: int = 0,
                   kept: int = 0) -> None:
    """A closed prefill forward (kt_lend): its rows, the sequence it ended at and how many bytes torch's allocation
    grew by. It enters the table at the next plan, where every rank has it (_sync).

    kept is what earlier windows left in the scratch, which kt_lend counts in grown and in the free memory alike.
    The table holds a forward without it, and the plan's free memory is without it too (_free_now): how much a
    launch keeps depends on its first window (570 MiB after a 12,296-row one, 1,232 MiB after a 22,016-row one), so
    a table with it in was 662 MiB too high in the next launch (11-19)."""
    if not ONLINE or rows <= 0:
        return
    grown, alloc = max(0, grown - kept), max(0, alloc - kept)
    if DEBUG == "1":  # each forward the plan learns from
        print(f"[kt-plan] window {window}: {rows} rows at sequence {seq}, "
              + (f"{lp_rows} with input logprobs, " if lp_rows > 0 else "")
              + f"grew {grown >> 20} MiB ({alloc >> 20} MiB allocated)", flush=True)
    PENDING.append((int(rows), int(seq), max(0, int(grown)), int(lp_rows)))


def _reduce(scheduler, values, low=False):
    """values as their most (least with low) over the TP ranks, on the scheduler's CPU group."""
    group = getattr(scheduler, "tp_cpu_group", None)
    if group is None:
        return list(values)
    import torch

    if not torch.distributed.is_initialized():
        return list(values)
    t = torch.tensor([-v if low else v for v in values], dtype=torch.int64)
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX, group=group)
    return [-v if low else v for v in t.tolist()]


def _sync(scheduler) -> None:
    """The forwards closed since the last plan into the table, each at the most it grew on any rank: every rank
    reaches chunk_for alike and holds the same table, so every rank plans the same chunk."""
    n = _reduce(scheduler, [len(PENDING)], low=True)[0]
    if n <= 0:
        return
    grown = _reduce(scheduler, [p[2] for p in PENDING[:n]])
    for (rows, seq, _, lp_rows), g in zip(PENDING[:n], grown):
        if g > 0:
            _fold(rows, seq, g, lp_rows)
    del PENDING[:n]
    _refresh()
    if DEBUG == "1":
        print(f"[kt-plan] {len(TABLE)} chunk sizes measured, the largest {MEASURED['max_rows']} rows at "
              f"{int(_own(MEASURED['max_rows'], longest())) >> 20} MiB; along a sequence as the {REF[0]}-row forwards, "
              f"{LP / 1024:.0f} KiB a logprob row", flush=True)


# The table an earlier launch of this setup measured, carried into this one (kt_lend's hint file,
# <KT_PREFILL_LEND_HINT>-rank<r>.json "plan"): the first window is then planned from it, not at the unmeasured chunk
# and the doublings after it (Qwen3.5: a first 35K prompt in five chunks at 2,919 tok/s, the third in one at 5,019).
# This launch's forwards join it: one that grows more than the carried ones raises its size's cost.
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


def scratch_ahead() -> int:
    """What a launch's first window is expected to leave in the scratch (the earlier launch's did): out of the free
    memory until this launch has left its own, which the free memory at a window's start then has in it."""
    from sglang.srt.layers.moe import kt_lend

    return 0 if int(kt_lend.STATE.get("scratch_kept", 0) or 0) > 0 else SCRATCH[0]


def _table_out():
    return {"points": [[r, q, g] for r in sorted(TABLE) for q, g in TABLE[r]], "lp_points": [list(p) for p in LP_PTS]}


def _table_in(rec) -> None:
    for r, q, g in rec.get("points", []):
        _put(TABLE.setdefault(int(r), []), int(q), int(g))
    LP_PTS[:] = [tuple(int(v) for v in p) for p in rec.get("lp_points", []) if len(p) == 4][-64:]
    _refresh()


def record():
    """The table for the next launch (kt_lend writes it to the hint file); None until two chunk sizes are measured."""
    if not ONLINE or len(TABLE) < 2:
        return None
    rec = dict(_table_out(), stamp=_stamp(), rev=MEASURED["windows"], start=None)
    try:
        import torch

        from sglang.srt.layers.moe import kt_ledger

        # the free memory a window starts with is the ledger's, known once a start has been measured
        rec["start"] = kt_ledger.carry_out(torch.cuda.current_device())
        from sglang.srt.layers.moe import kt_lend

        rec["scratch"] = int(kt_lend.STATE.get("scratch_kept", 0) or 0) or SCRATCH[0]
    except Exception as e:
        _say_once("prefill plan: the ledger's start was not recorded", e)
    return rec


def _carry_in(scheduler) -> None:
    """Once, at the first plan of a launch: the earlier launch's table as this one's. Every rank must hold the same
    one, or none is carried."""
    if MEASURED.get("carry_tried"):
        return
    MEASURED["carry_tried"] = True
    if not (ONLINE and CARRY_ON and CARRY_HINT) or TABLE:
        return
    rec = {}
    try:
        import json

        from sglang.srt.layers.moe import kt_lend

        with open(f"{CARRY_HINT}-rank{kt_lend._rank()}.json") as f:
            rec = json.load(f).get("plan") or {}
    except Exception:
        pass
    pts = [p for p in rec.get("points", []) if len(p) == 3]
    ok = rec.get("stamp") == _stamp() and len({p[0] for p in pts}) >= 2
    if rec and not ok:
        print("[kt-plan] an earlier launch's forwards are not carried: "
              + ("the package changed since" if rec.get("stamp") != _stamp() else "one chunk size only"), flush=True)
    import zlib

    mark = zlib.crc32(repr(sorted(map(tuple, pts))).encode()) if ok else 0
    if _reduce(scheduler, [mark]) != _reduce(scheduler, [mark], low=True):
        print("[kt-plan] an earlier launch's forwards are not carried: the ranks' hint files differ", flush=True)
        return
    if not ok:
        return
    _table_in({"points": pts, "lp_points": rec.get("lp_points", [])})
    MEASURED["carried"] = len(pts)
    SCRATCH[0] = max(0, int(rec.get("scratch", 0) or 0))
    try:
        import torch

        from sglang.srt.layers.moe import kt_ledger

        kt_ledger.carry_in(torch.cuda.current_device(), rec.get("start"))
    except Exception as e:
        _say_once("prefill plan: the ledger's start was not carried", e)
    print(f"[kt-plan] {len(pts)} forwards carried from an earlier launch: {len(TABLE)} chunk sizes up to "
          f"{MEASURED['max_rows']} rows at {int(_own(MEASURED['max_rows'], longest())) >> 20} MiB", flush=True)


def _lp_cost(req) -> float:
    """A row's logprob cost for a request that asked for its input logprobs (LP), else 0."""
    if LP <= 0 or not getattr(req, "return_logprob", False):
        return 0.0
    start = int(getattr(req, "logprob_start_len", -1) or 0)
    n = len(getattr(req, "origin_input_ids", ()) or ())
    return LP if 0 <= start < n - 1 else 0.0


def online_cap(chunk: int) -> int:
    """The chunk the table allows now: twice the largest forward measured. What a larger chunk costs enters the
    table one doubling at a time instead of one extrapolation (a 4,096-row first window planned a 1M-context Vision
    into an OOM at once, 11-05)."""
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
    return ENABLED and (FREE > 0 or LIVE) and (bool(TABLE) if ONLINE else ROW > 0)


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
# rid -> the longest sequence measured when the request was planned, where its end lay past it: the plan priced its
# chunk on a rise carried on from the forwards before, and is made again as forwards further along come in
AHEAD = {}


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
        spare = int(e[0]) if e else 0
        bound = getattr(kt_kv_vmm, "prompt_rows_bound", None) if e else None

        def cost(x: int) -> float:
            return carry + _own(x, done + min(x, rem), lpc)

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
            if DEBUG == "1":
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
    # chunks. What earlier windows kept in the scratch is mapped by then and not in it, as it is not in a forward's
    # cost; before this launch's first window has left any, what the earlier launch's left comes out (scratch_ahead). The decode-to-window step (pf_delta, or KT_PREFILL_PLAN_DELTA_MB from an earlier launch),
    # the in-window reading and the hot lend's spare are the ledger's holders now. Before a window start has been
    # measured: the constant (FREE)
    from sglang.srt.layers.moe import kt_ledger

    led = kt_ledger.window_free(torch.cuda.current_device())
    mine = -1 if led is None else max(0, int(led) - scratch_ahead())
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


def rid_ahead(rid, rem: int) -> bool:
    """Whether a planned request is planned again at this chunk: its plan priced the prompt's end past every forward
    measured, forwards further along have come in since, and more than one chunk is left. A 131K prompt after a
    65K one was planned from three 16,384-row forwards up to sequence 49,152, whose rise carried on to 131,072 made
    a 1,024-row chunk cost 10.3 GiB: 43 chunks of 3,072 rows, 192 s against 63 s. The fourth forward, at 65,536,
    came in after the plan and grew less than the third (V4.1 on two cards, 11-20)."""
    at = AHEAD.get(rid)
    if at is None or rem <= int(PLAN.get(rid, 0)) or longest() <= at:
        return False
    AHEAD[rid] = longest()
    return True


def chunk_for(scheduler, chunk: int) -> int:
    """Scheduler.get_new_batch_prefill: the chunk for the batch it is about to make."""
    from sglang.srt.layers.moe import kt_kv_prefill_lend

    getattr(kt_kv_prefill_lend, "bind_tree", lambda s: None)(scheduler)
    if DEBUG == "2":
        q = getattr(scheduler, "waiting_queue", None)
        print(f"[kt-plan] chunk_for({chunk}): chunked {scheduler.chunked_req is not None}, queue {type(q).__name__} "
              f"of {len(q) if q is not None else None}, ready {_ready()}, windows {MEASURED['windows']}, row {ROW}",
              flush=True)
    if not chunk or chunk <= 0:
        return chunk
    # the first window reaches the planner before anything is measured
    if not ENABLED or not (ONLINE or _ready()):
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
    if DEBUG == "1" and (rem > 2048 or done > 2048):
        print(f"[kt-plan] chunk_for {str(req.rid)[:8]}: {'a later chunk' if req is scheduler.chunked_req else 'queued'}"
              f", {rem} to prefill after {done}, planned {req.rid in PLAN}, rows up {req.rid in ROWS_UP}, "
              f"stream from {_stream_from()}", flush=True)
    if req is not scheduler.chunked_req:
        # no SWA bound until this request is planned (a chunked one keeps its plan's)
        getattr(kt_kv_prefill_lend, "swa_room", lambda t, w="": None)(None)
    if rem <= 0:
        return chunk
    if ONLINE:
        _carry_in(scheduler)
        _sync(scheduler)  # same on every rank
    if ONLINE and not TABLE:
        # no plain forward measured yet (none at all, or only ones with input logprobs): the lend's own chunk
        blind = min(chunk, SAFE) if SAFE > 0 else chunk
        if DEBUG == "1":
            print(f"[kt-plan] first window allowed {blind} rows unmeasured", flush=True)
        return blind
    if not _ready():
        return chunk
    capped_at = online_cap(chunk)
    capped = capped_at < chunk
    chunk = capped_at
    if req is scheduler.chunked_req and req.rid in PLAN and CAPPED.get(req.rid) != PLAN[req.rid]:
        # a later chunk keeps the request's plan: with the overlap scheduler this
        # runs while the previous chunk's forward still holds its activations, so
        # the live free memory would be low by them
        again = rid_ahead(req.rid, rem)
        if not again:
            return _window_chunk(req, rem, done, chunk, scheduler)
    else:
        again = False
    from sglang.srt.layers.moe import kt_kv_prefill_lend

    free = _free_now(scheduler)
    if free <= 0 and DEBUG == "1":
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
        kv_need = max(0, min(kv_need, int(room + idle - _own(ALIGN, done + rem, _lp_cost(req)))))
        if DEBUG == "1":
            print(f"[kt-plan] rows for {req.rid[:8]}: asked {kv_asked >> 20} MiB, room {int(room) >> 20} MiB = free "
                  f"{int(free) >> 20} + loan {int(loan) >> 20} - margin {_margin() >> 20} - on demand {int(back) >> 20}"
                  f", smallest chunk {int(_own(ALIGN, done + rem, _lp_cost(req))) >> 20}, dec_free "
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
        return (CARRY if n > 1 else 0) + _own(c, done + rem, lpc) + swa_cost(c) <= room

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
    if again:
        # only a larger chunk replaces the plan's: the plan was the careful one
        if c <= PLAN[req.rid]:
            return _window_chunk(req, rem, done, chunk, scheduler)
        print(f"[kt-prefill-plan] {rem} tokens left of a prompt planned in chunks of {PLAN[req.rid]}: forwards "
              f"measured to sequence {longest()} since, planned again", flush=True)
    if swa is not None and swa[1] > 0:
        # the SWA tokens the room holds past the chunk's own cost, at least the chunk's rows
        # held at once plus the window and two pages (sglang's need for it)
        own = (CARRY if n > 1 else 0) + _own(c, done + rem, lpc)
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
        if DEBUG == "1":
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
        AHEAD.clear()
    PLAN[req.rid] = c
    if done + rem > longest():
        AHEAD[req.rid] = longest()
    else:
        AHEAD.pop(req.rid, None)
    if not again:
        ISSUED.pop(req.rid, None)
    WIN_BASE.pop(req.rid, None)
    WIN_SAID.pop(req.rid, None)
    if swa is None and n > 1:
        WIN[req.rid] = (CARRY + _own(c, done + rem, lpc), CARRY, lpc)
    else:
        WIN.pop(req.rid, None)
    if capped and c >= chunk:
        CAPPED[req.rid] = c
    else:
        CAPPED.pop(req.rid, None)
    # the loan when the estimate needs it, and always past the lend's own chunk:
    # there the estimate is extrapolated, and lending costs a few tens of ms
    short = (CARRY if n > 1 else 0) + _own(c, done + rem, lpc) + swa_cost(c) + _margin() + back - free
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
              f"(free {free >> 20} MiB + loan {loan >> 20} MiB, {int(_own(c, done + rem, lpc)) >> 20} MiB a chunk "
              f"at the prompt's end, {len(TABLE)} sizes measured up to {MEASURED['max_rows']} rows, "
              f"carry {CARRY >> 20} MiB, "
              f"SWA mapped back {swa_cost(c) >> 20} MiB"
              + (f", {back >> 20} MiB back on the GPU for the images ({', '.join(ON_DEMAND)})" if back else "")
              + f", cap {chunk})", flush=True)
    return _window_chunk(req, rem, done, max(chunk, c))
