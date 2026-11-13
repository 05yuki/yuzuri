#!/usr/bin/env python3
"""A short check of a running server: decode, a long prompt twice, decode after it, and the server log's errors.

Start the server (tools/serve.py ... ), wait for "The server is fired up", then:

    tools/check.py [--url http://127.0.0.1:8080] [--prompt 35000] [--long 131072] [--log <run dir>/server.log]
                   [--json after.json]
    tools/check.py --compare before.json after.json

The same command against the server started with `serve.py --features none` gives the "before" file. Only the
standard library is used; the prompts are the server's own tokens of a fixed English text (random token ids when
the server has no /tokenize), each with 16 random ids first so no prefix is cached.
"""
import argparse
import json
import random
import re
import sys
import time
import urllib.request

TEXT = (
    "The river had been rising for three days when the surveyor came down from the upper camp with his "
    "instruments wrapped in oilcloth. He set the level on the last dry stone of the old quay, wrote the "
    "reading in his book, and compared it with the marks the harbour master's father had cut into the wall "
    "forty years before. The ferry still ran, though the rope sang under the load, and on the far bank the "
    "millers were carrying sacks to the loft by lantern. Nobody spoke of leaving. The baker lit his oven at "
    "the usual hour, the schoolmistress rang her bell, and the carter who had lost a wheel on the north road "
    "sat in the smithy doorway and waited for the iron to cool.\n\n"
)
ERRORS = re.compile(r"OutOfMemory|out of memory|Scheduler hit an exception|Traceback \(most recent call last\)")
NOTES = re.compile(r"dynamic slots (on|off)[^\n]{0,160}|HOST ROWS IN A PREFILL")


def post(url, body, timeout):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def get(url, timeout=10):
    return json.load(urllib.request.urlopen(url, timeout=timeout))


def base_ids(url):
    """The fixed text in the server's tokens; None when the server does not tokenize."""
    try:
        r = post(url + "/tokenize", {"prompt": TEXT, "add_special_tokens": False}, 30)
        ids = r.get("tokens") or r.get("token_ids")
        if ids and isinstance(ids[0], int) and len(ids) > 32:
            return ids
    except Exception:  # noqa: BLE001
        pass
    return None


def generate(url, ids, new, timeout):
    s = time.time()
    r = post(url + "/generate", {"input_ids": ids, "sampling_params": {"max_new_tokens": new, "temperature": 0,
                                                                     "ignore_eos": True}}, timeout)
    dt = time.time() - s
    m = r["meta_info"] if isinstance(r, dict) else r[0]["meta_info"]
    return dt, int(m.get("completion_tokens", new))


def prompt_ids(base, rng, n):
    head = [rng.randrange(1000, 20000) for _ in range(16)]
    if base is None:
        return head + [rng.randrange(1000, 20000) for _ in range(n - 16)]
    return (head + base * (n // len(base) + 2))[:n]


def decode_rate(url, ids, timeout, new=256):
    """Tokens a second of `new` tokens after `ids`: the time of new+1 tokens less the time of 1."""
    t1, _ = generate(url, ids, 1, timeout)
    t2, got = generate(url, ids, new + 1, timeout)
    return (got - 1) / max(t2 - t1, 1e-6)


def scan(log, start):
    with open(log, "rb") as f:
        f.seek(start)
        text = f.read().decode("utf-8", "replace")
    return len(ERRORS.findall(text))


def compare(a, b):
    A, B = json.load(open(a)), json.load(open(b))
    print(f"{'':28}{'before':>12}{'after':>12}{'ratio':>9}")
    for k, v in A["rows"].items():
        w = B["rows"].get(k)
        if w is None:
            continue
        print(f"{k:28}{v:12.1f}{w:12.1f}{w / v if v else float('nan'):9.2f}")
    print(f"{'server log error lines':28}{A.get('errors', '-')!s:>12}{B.get('errors', '-')!s:>12}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--prompt", type=int, default=35000, help="the long prompt's tokens (sent twice)")
    ap.add_argument("--long", type=int, default=0, help="one more prompt of this many tokens, and decode after it")
    ap.add_argument("--log", help="the server log: error lines written during the check are counted")
    ap.add_argument("--json", help="write the rows here, for --compare")
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    a = ap.parse_args()
    if a.compare:
        return compare(*a.compare)
    url = a.url.rstrip("/")
    try:
        info = get(url + "/get_model_info")
    except Exception as e:  # noqa: BLE001
        sys.exit(f"no server at {url} ({type(e).__name__}: {e}); start it with tools/serve.py first")
    print(f"model: {info.get('model_path')}")
    pool = 0
    try:
        with urllib.request.urlopen(url + "/get_server_info", timeout=30) as r:
            m = re.search(r'"max_total_num_tokens":\s*(\d+)', r.read().decode("utf-8", "replace"))
            pool = int(m.group(1)) if m else 0
    except Exception:  # noqa: BLE001
        pass
    if pool:
        print(f"pool: {pool} tokens")

    def fits(n):
        if pool and n + 512 > pool:
            print(f"  a {n}-token prompt is skipped: the server's pool is {pool} tokens")
            return False
        return True
    start = 0
    if a.log:
        with open(a.log, "rb") as f:
            start = f.seek(0, 2)
    rng = random.Random()
    base = base_ids(url)
    print("prompts: " + ("the fixed text in the server's tokens" if base else "random token ids (no /tokenize)"))
    rows = {}

    def row(name, value, unit="tok/s"):
        rows[name] = round(value, 2)
        print(f"  {name:26} {value:10.1f} {unit}")

    row("decode", decode_rate(url, prompt_ids(base, rng, 256), a.timeout))
    for i in (1, 2) if fits(a.prompt) else ():
        ids = prompt_ids(base, rng, a.prompt)
        dt, _ = generate(url, ids, 1, a.timeout)
        row(f"prefill {a.prompt} #{i}", a.prompt / dt)
    row("decode after the prompts", decode_rate(url, prompt_ids(base, rng, 256), a.timeout))
    if a.long and fits(a.long):
        ids = prompt_ids(base, rng, a.long)
        dt, _ = generate(url, ids, 1, a.timeout)
        row(f"prefill {a.long}", a.long / dt)
        # the same prompt again: its prefix is cached, so the difference of the two times is decode at that length
        t1, _ = generate(url, ids, 1, a.timeout)
        t2, got = generate(url, ids, 257, a.timeout)
        row(f"decode at {a.long}", (got - 1) / max(t2 - t1, 1e-6))
    out = {"model": info.get("model_path"), "rows": rows}
    if a.log:
        n = scan(a.log, start)
        out["errors"] = n
        print(f"server log, during the check: {n} error lines")
        with open(a.log, "rb") as f:
            whole = f.read().decode("utf-8", "replace")
        for line in sorted({m.group(0)[:170] for m in NOTES.finditer(whole)}):
            print(f"  log: {line}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"wrote {a.json}")
    return 1 if out.get("errors") else 0


if __name__ == "__main__":
    sys.exit(main())
