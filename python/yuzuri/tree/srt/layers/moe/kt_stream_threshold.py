"""The prompt length from which a prefill streams its CPU experts through the GPU (KT_GPU_STREAM_PREFILL).

A streamed prefill opens a lend window: the resident weights leave, the streamer passes every layer's experts over
PCIe, the weights come back. That costs the same for 256 tokens as for 4,096 (Qwen3.5 bf16, TP 2: 1.35 s and
0.05 s a 1,000 tokens), while the same prompt on the resident path costs nothing fixed and more a token (0.27 s and
0.47 s a 1,000). Where the two lines cross is a property of the model, the cards and the host's memory, so it is
measured, not set:

  KT_GPU_STREAM_PREFILL=auto   on; the length is measured (below) and carried to the next launch
  KT_GPU_STREAM_PREFILL=<n>    on from n tokens, nothing measured (a test of one length)
  KT_GPU_STREAM_PREFILL=0      off

Both lines are measured on fresh single prompts without input logprobs (rows == sequence), from before the lend's
hook to the forward's end, the slowest rank's time (one all-reduce, so every rank derives the same length at the same
forward):

  streamed   the launch's calibration prompts (warmup "prefill_plan"): REPS forwards each at START and 4 x START
             tokens, the fastest of each. A window passes every expert whatever the prompt, so their random tokens
             measure it; the line is carried with the record.
  resident   the prompts the server is given, the last KEEP of them of MIN_ROWS tokens or more (FIRST before the
             first line), by the median of their pairwise slopes, again at every EVERY more. Not the calibration's: random tokens route to every expert alike and none of
             them is in a hot slot, 1.64 s a 1,000 tokens where the same lengths of text took 0.47, and the crossing
             came out at 640 tokens for 2,340 (Qwen3.5 bf16, 11-11). What the resident path costs is what the hot
             slots hold of the prompts served, so it is read from those.

Until a resident line exists the length is START (the plan's chunk alignment) or the record's, and it moves no
further than twice the longest prompt the resident line was read from.

The record is <KT_PREFILL_LEND_HINT>.stream.json, one file for all ranks, good for this package's code
(kt_prefill_plan._stamp).
"""
import json
import os
import time

import torch

_ENV = (os.environ.get("KT_GPU_STREAM_PREFILL", "0") or "0").strip().lower()
AUTO = _ENV == "auto"
ALIGN = int(os.environ.get("KT_PREFILL_PLAN_ALIGN", "1024"))
# what the streamers' on/off tests read, and the length before anything is measured
START = ALIGN if AUTO else int(_ENV)
REPS = int(os.environ.get("KT_GPU_STREAM_REPS", "3"))
KEEP = int(os.environ.get("KT_GPU_STREAM_KEEP", "32"))
EVERY = int(os.environ.get("KT_GPU_STREAM_EVERY", "4"))
FIRST = int(os.environ.get("KT_GPU_STREAM_FIRST", "16"))  # prompts before the first resident line
MIN_ROWS = 128  # shorter forwards are the launch's own (warmup) and all fixed cost
HINT = os.environ.get("KT_PREFILL_LEND_HINT", "")
STATE = {"value": START, "loaded": not (AUTO and HINT), "streamed": None, "resident": [], "new": 0, "cal": {}}


def calibration_lengths():
    """The prompts a launch's calibration sends, REPS rounds of them: two lengths that stream, far enough apart for
    a slope."""
    return [START, 4 * START]


def _log(msg: str) -> None:
    print(f"[kt-stream-threshold {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _path() -> str:
    return f"{HINT}.stream.json"


def _stamp() -> str:
    from sglang.srt.layers.moe import kt_prefill_plan

    return kt_prefill_plan._stamp()


def _carry_in() -> None:
    STATE["loaded"] = True
    try:
        with open(_path()) as f:
            rec = json.load(f)
        if rec.get("stamp") != _stamp():
            return
        v, line = int(rec["threshold"]), [float(x) for x in rec["streamed"]]
    except Exception:
        return
    if v > 0 and len(line) == 2:
        STATE["value"], STATE["streamed"] = v, line
        _log(f"streamed prefill from {v} tokens (an earlier launch: {rec.get('fit')})")


def get() -> int:
    """Rows from which a forward streams; 0 when streaming is off."""
    if not STATE["loaded"]:
        _carry_in()
    return STATE["value"]


def begin(forward_batch) -> None:
    """The model runner, before the lend's hook: a forward that counts starts its clock."""
    if not AUTO:
        return
    STATE.pop("t0", None)
    if not STATE["loaded"]:
        _carry_in()
    mode = getattr(forward_batch, "forward_mode", None)
    if mode is None or not mode.is_extend() or torch.cuda.is_current_stream_capturing():
        return
    lens = getattr(forward_batch, "seq_lens_cpu", None)
    rows = int(forward_batch.input_ids.shape[0])
    if lens is None or len(lens) != 1 or int(lens[0]) != rows or getattr(forward_batch, "return_logprob", False):
        return
    streams = rows >= STATE["value"]
    if rows < MIN_ROWS or (streams and STATE["streamed"] is not None):
        return
    if not streams and STATE["streamed"] is None:
        return  # before the calibration's: the launch's own short forwards
    torch.cuda.synchronize()
    STATE["t0"] = (time.perf_counter(), rows, streams)


def end() -> None:
    """The model runner, after the forward."""
    t = STATE.pop("t0", None)
    if t is None:
        return
    torch.cuda.synchronize()
    dt = time.perf_counter() - t[0]
    if t[2]:
        STATE["cal"].setdefault(t[1], []).append(dt)
        _streamed()
        return
    r = STATE["resident"]
    r.append((t[1], dt))
    del r[:-KEEP]
    STATE["new"] += 1
    if STATE["new"] >= EVERY and len(r) >= FIRST and max(n for n, _ in r) >= 2 * min(n for n, _ in r):
        STATE["new"] = 0
        _derive()


def _over_ranks(values):
    """The slowest rank's of each, the same vector on every rank."""
    v = torch.tensor(values, dtype=torch.float64)
    world = 1
    try:
        from sglang.srt.distributed.parallel_state import get_tp_group

        g = get_tp_group()
        world = g.world_size
        if world > 1:
            torch.distributed.all_reduce(v, op=torch.distributed.ReduceOp.MAX, group=g.cpu_group)
    except Exception:
        if world > 1:
            raise
    return v.tolist()


def _rank() -> int:
    try:
        from sglang.srt.runtime_context import get_parallel

        return int(get_parallel().tp_rank)
    except Exception:
        return 0


def _streamed() -> None:
    c = STATE["cal"]
    n = sorted(k for k, v in c.items() if len(v) >= REPS)
    if len(n) < 2:
        return
    t = _over_ranks([min(c[n[0]]), min(c[n[-1]])])
    b = (t[1] - t[0]) / (n[-1] - n[0])
    STATE["streamed"] = [t[0] - b * n[0], b]
    c.clear()
    if _rank() == 0:
        _log(f"a window costs {STATE['streamed'][0]:.2f} s + {b * 1e3:.3f} s/1000 tokens ({n[0]}, {n[-1]}); streamed "
             f"from {STATE['value']} until the prompts served give the resident path's cost")
        _record(f"streamed {STATE['streamed'][0]:.2f} s + {b * 1e3:.3f} s/1000, the resident path not measured")


def _derive() -> None:
    r = list(STATE["resident"])
    t = _over_ranks([dt for _, dt in r])
    k = len(r)
    # the resident path: seconds a row, and at no rows. The median of the slopes between every two prompts of
    # different lengths and of what each leaves at no rows, not least squares: the first prompts after a launch run
    # cold (2,000 tokens in 4.9 s where they take 1.1), and one of those in eight turned the line to 2.24 s a 1,000
    # tokens and the length to 832 (Qwen3.5 bf16, 11-11)
    ns = [n for n, _ in r]
    slopes = sorted((t[j] - t[i]) / (ns[j] - ns[i]) for i in range(k) for j in range(i + 1, k) if ns[i] != ns[j])
    d = slopes[len(slopes) // 2]
    rest = sorted(y - d * n for n, y in zip(ns, t))
    c = rest[k // 2]
    a, b = STATE["streamed"]
    safe = int(os.environ.get("KT_PREFILL_PLAN_SAFE", "0") or 0)
    # no further than twice the longest prompt the line was read from: the length moves out as longer ones are served
    lo, hi = max(64, ALIGN // 4), min(max(safe, 16 * ALIGN), 2 * max(n for n, _ in r))
    if a <= c:
        value = lo
    elif d <= b:
        value = hi
    else:
        value = min(hi, max(lo, -(-int((a - c) / (d - b)) // 64) * 64))
    was = STATE["value"]
    STATE["value"] = int(value)
    if value == was or _rank() != 0:
        return
    fit = (f"resident {c:.2f} s + {d * 1e3:.3f} s/1000 ({k} prompts of {min(n for n, _ in r)}-{max(n for n, _ in r)} "
           f"tokens), streamed {a:.2f} s + {b * 1e3:.3f} s/1000")
    _log(f"streamed prefill from {value} tokens (was {was}): {fit}")
    _record(fit)


def _record(fit: str) -> None:
    if not HINT:
        return
    try:
        with open(_path() + ".tmp", "w") as f:
            json.dump({"stamp": _stamp(), "threshold": int(STATE["value"]), "streamed": STATE["streamed"],
                       "fit": fit}, f)
        os.replace(_path() + ".tmp", _path())
    except Exception as e:
        _log(f"the record was not written ({e})")
