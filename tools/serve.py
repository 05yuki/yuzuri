#!/usr/bin/env python3
"""Serve a model with Yuzuri: probe the host, size the launch, set the profile's knobs, start SGLang.

usage: serve.py --package --venv SERVER_VENV [--profile auto|NAME] --config-dir DIR [--experts DIR | --gguf DIR]
                [--draft-dir DIR] [--tp 1|2] [--context T] [--host 127.0.0.1] [--port 8080] [--hot N] [--chunk C]
                [--kv-dtype bf16|fp8] [--mamba-slots N] [--threads N] [--placement FILE] [--hot-seed JSON]
                [--draft-hot FILE] [--log-dir DIR] [--cache-dir DIR] [--env KEY=VALUE ...] [--background] [--dry-run]
                [--no-plan]

  --package --venv   the server's venv, with Yuzuri installed into it by ./setup.sh (yuzuri.pth): the SGLang and
                 kt-kernel installed there serve, with Yuzuri's hooks on
  --recipe       instead of --package: a deepseek-v4.1-flash-1017-ktransformers checkout whose tree carries the
                 same changes as patches: its .venv, third_party/sglang, prebuilt/lib and build/cuda-home serve
  --host         the address the server listens on (default 127.0.0.1, this machine only; SGLang asks for no key
                 unless --server-arg="--api-key KEY")
  --profile      auto (the default): the command line and environment built from the checkpoint (auto_profile.py);
                 or profiles/<NAME>.json, a measured launch's command line, env defaults (per TP) and weights layout
  --config-dir   the model directory SGLang loads (config + tokenizer; for a GGUF serve, tools/make-config-dir.py)
  --gguf / --experts   where the experts are (a GGUF dir, or a second checkpoint dir); default: the config dir
  --draft-dir    the MTP draft directory when the profile serves one (tools/make-draft-dir.py)
  --context      default: the model's context (the profile's max_context, else max_position_embeddings); the KV
                 pool is the context, never cut to fit: the KV the card cannot hold goes to host memory
  --hot / --chunk / --threads   override the plan (tools/plan-launch.py prints the arithmetic). There is no memory
                 fraction to set (--mem-fraction-static 1.0: the pool's pages are mapped as they are used)
  --plan         the prefill chunk planned per request over the KV pool's free pages (kv-vmm/; README, "The chunk
                 per request"): the launch chunk becomes --plan-cap, the lend's own chunk the fallback; a setup's
                 first launch runs the lend at its chunk and measures, the next ones plan
The server log and a manifest (command, env, plan, package versions) go to --log-dir/<timestamp>/.
"""
import argparse
import datetime
import glob
import importlib.metadata as md
import json
import os
import shlex
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROFILES = os.path.join(os.path.dirname(HERE), "profiles")
PACKAGE_TREE = os.path.join(os.path.dirname(HERE), "python", "yuzuri", "tree")
FEATURES = os.path.join(os.path.dirname(HERE), "python", "yuzuri", "features.json")


# the chunk ceiling before this setup has measured a window: SGLang sizes some buffers by its chunked-prefill size at
# startup (the sliding-window and linear-attention pools, DeepSeek V4's hyper-connection buffers), so it needs one
FIRST_CEILING = 36864


def measured_ceiling(cache_dir, key, context, align=1024):
    """The most rows a prefill window on these cards has room for, from the lend's records of an earlier launch
    (lend-hint-<key>-rank<N>.json: each measured window's free memory, the KV loan it could borrow and its cost a
    row): per rank the largest over its windows, the smallest over the ranks, in whole steps of align and no longer
    than the context; None before a launch has measured one. The free memory is measured before the loan goes out,
    so without the loan the ceiling could never exceed the loan-less room (GLM 36864 -> 16384, 11-09)."""
    per_rank = []
    for f in sorted(glob.glob(os.path.join(cache_dir, f"lend-hint-{key}-rank*.json"))):
        try:
            with open(f) as fh:
                h = json.load(fh)
        except (OSError, ValueError):
            continue
        margin = int(h.get("margin_mb", 256)) << 20
        best = 0
        for p in (h.get("points") or {}).values():
            room = p.get("free", 0) + p.get("loan_room", 0)
            if p.get("per_row", 0) > 0 and room > margin:
                best = max(best, int((room - margin) // p["per_row"]))
        if best:
            per_rank.append(best)
    if not per_rank:
        return None
    return max(4 * align, min(min(per_rank) // align * align, context))


def plan_carried(cache_dir, key, moe_dir, stream_auto=False):
    """Whether the server will plan its first window from an earlier launch's forwards (kt_prefill_plan._carry_in):
    every rank's hint file holds them, measured with this package's code (the stamp kt_prefill_plan._stamp makes)."""
    import hashlib

    h = hashlib.sha1()
    try:
        for n in sorted(os.listdir(moe_dir)):
            if n.endswith(".py"):
                with open(os.path.join(moe_dir, n), "rb") as fh:
                    h.update(n.encode() + fh.read())
    except OSError:
        return False
    files = sorted(glob.glob(os.path.join(cache_dir, f"lend-hint-{key}-rank*.json")))
    for f in files:
        try:
            with open(f) as fh:
                plan = json.load(fh).get("plan") or {}
        except (OSError, ValueError):
            return False
        if plan.get("stamp") != h.hexdigest()[:16] or len({p[0] for p in plan.get("points", [])}) < 2:
            return False
    if files and stream_auto:
        # the length a prefill streams from, measured by the same prompts (kt_stream_threshold)
        try:
            with open(os.path.join(cache_dir, f"lend-hint-{key}.stream.json")) as fh:
                rec = json.load(fh)
        except (OSError, ValueError):
            return False
        if rec.get("stamp") != h.hexdigest()[:16] or not rec.get("threshold") or not rec.get("streamed"):
            return False
    return bool(files)


def feature_table():
    with open(FEATURES) as f:
        return json.load(f)


def list_features():
    t = feature_table()
    for f in t["features"]:
        need = f", needs {', '.join(f['requires'])}" if f["requires"] else ""
        rec = f", recipe tier 1 {f['recipe']}" if f.get("recipe") else ""
        print(f"{f['name']:<20} {f['for']}{need}{rec}{' (in base)' if f.get('base') else ''}")
    for f in t["always"]:
        rec = f", recipe {f['recipe']}" if f.get("recipe", "-") != "-" else ""
        print(f"{f['name']:<20} {f['for']} (always on: no switch{rec})")


def pick_features(on_arg, without_arg):
    """(the features off, the names on): every feature on unless --features names the ones wanted (or none);
    --without drops some. A feature on whose prerequisite is off stops the launch with what to add."""
    t = feature_table()
    by = {f["name"]: f for f in t["features"]}

    alias = {f["recipe"]: f["name"] for f in t["features"] if f.get("recipe")}

    def names(arg):
        got = {alias.get(n.strip(), n.strip()) for n in (arg or "").split(",") if n.strip()}
        if "base" in got:
            got = got - {"base"} | {f["name"] for f in t["features"] if f.get("base")}
        bad = got - set(by) - {"all", "none"}
        if bad:
            raise SystemExit(f"unknown feature {', '.join(sorted(bad))} (serve.py --list-features)")
        return got

    want = names(on_arg)
    want = set(by) if not want or "all" in want else set() if want == {"none"} else want
    want -= names(without_arg)
    for n in sorted(want):
        miss = [r for r in by[n]["requires"] if r not in want]
        if miss:
            raise SystemExit(f"feature {n} needs {', '.join(miss)}: add them, or --without {n}")
    return [by[n] for n in by if n not in want], [n for n in by if n in want]


def sh(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode:
        raise SystemExit(f"{' '.join(cmd[:3])} failed:\n{r.stderr[-2000:]}")
    return r.stdout




def model_layers(config_dir):
    c = json.load(open(os.path.join(config_dir, "config.json")))
    t = c.get("text_config", c)
    return int(t.get("num_hidden_layers") or 1)


def fill(items, vals):
    out = []
    for it in items:
        s = it
        for k, v in vals.items():
            s = s.replace("{" + k + "}", str(v))
        out.append(s)
    return out


def io_threads(env: dict, checkpoint: str, cache_dir: str, nodes) -> str:
    """The drive's reader pools when a feature reads the checkpoint while serving (kt-kernel's MXFP4 expert cache,
    the GGUF NVMe streamer): as many readers as reach 90% of the drive's best rate (tools/io-probe.py, measured once
    a drive and kept in cache_dir), on the second hardware threads of the serving nodes' cores (KT_IO_CPUS; none
    without SMT). Variables already set stay. Returns the line to print, '' when nothing reads the drive."""
    mxc = env.get("KT_MXFP4_EXPERT_CACHE_GIB", "0") not in ("", "0")
    nvme = env.get("KT_GGUF_STREAM_SOURCE") == "nvme"
    if not (mxc or nvme):
        return ""
    import importlib.util

    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("io_probe", os.path.join(here, "io-probe.py"))
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    files = sorted([os.path.join(d, f) for d, _, fs in os.walk(checkpoint) for f in fs
                    if f.endswith((".gguf", ".safetensors"))], key=os.path.getsize, reverse=True)
    if not files:
        return f"drive reads: no checkpoint files under {checkpoint} to probe; the reader pools keep their defaults"
    dev = probe.block_device(files[0])
    key = "".join(c if c.isalnum() else "-" for c in f"{dev['device']}-{dev.get('model') or ''}").strip("-")
    out = os.path.join(cache_dir, f"io-probe-{key}.json")
    if not os.path.exists(out):
        os.makedirs(cache_dir, exist_ok=True)
        print(f"probing the drive under {checkpoint} ({dev['device']}, {dev.get('model')}): about a minute, once",
              flush=True)
        subprocess.run([sys.executable, os.path.join(here, "io-probe.py"), checkpoint, out,
                        "--piece-kib", "1024,2048"], check=True)
    pieces = json.load(open(out))["pieces"]
    said = []
    for var, kib, on in (("KT_MXFP4_STREAM_READERS", "1024", mxc), ("KT_GGUF_STREAM_READERS", "2048", nvme)):
        if on and var not in env and kib in pieces:
            env[var] = str(pieces[kib]["saturating_threads"])
            said.append(f"{var}={env[var]} ({pieces[kib]['best_gb_per_s']:.1f} GB/s at {kib} KiB)")
    if "KT_IO_CPUS" not in env:
        rows = [ln.split(",") for ln in subprocess.run(["lscpu", "-p=CPU,CORE,NODE"], capture_output=True,
                                                         text=True).stdout.splitlines() if not ln.startswith("#")]
        seen, second = set(), []
        for cpu, core, node in rows:
            if node.isdigit() and int(node) in set(nodes):
                if (node, core) in seen:
                    second.append(cpu)
                seen.add((node, core))
        if second:
            env["KT_IO_CPUS"] = ",".join(second)
            said.append(f"KT_IO_CPUS on the {len(second)} second hardware threads of nodes {sorted(set(nodes))}")
    return f"drive reads ({dev['device']}): " + ("; ".join(said) if said else "everything already set")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="auto", help="profiles/<NAME>.json, or auto (built from the checkpoint)")
    ap.add_argument("--venv", help="the server's Python venv (SGLang and kt-kernel installed in it)")
    ap.add_argument("--recipe", help="a recipe checkout (deepseek-v4.1-flash-1017-ktransformers): its .venv, "
                                     "third_party/sglang, prebuilt/lib and build/ when --venv is not given")
    ap.add_argument("--config-dir", required=True)
    ap.add_argument("--gguf")
    ap.add_argument("--experts")
    ap.add_argument("--draft-dir")
    ap.add_argument("--tp", type=int)
    ap.add_argument("--context", type=int)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1",
                    help="address the server listens on (default 127.0.0.1: this machine only). SGLang has no "
                         "authentication unless --server-arg=\"--api-key KEY\" sets one: give 0.0.0.0 only on a "
                         "network where every machine may use the model")
    ap.add_argument("--max-running-requests", type=int, default=1,
                    help="requests SGLang runs at once (default 1); a long prompt holds the others while it prefills")
    ap.add_argument("--hot", type=int)
    ap.add_argument("--free-min-gb", type=float,
                    help="GiB of each card the plan keeps free of weights and pool (default: the profile's, 1)")
    ap.add_argument("--chunk", type=int)
    ap.add_argument("--kv-dtype", choices=["bf16", "fp8"])
    ap.add_argument("--mamba-slots", type=int)
    ap.add_argument("--threads", type=int)
    ap.add_argument("--placement")
    ap.add_argument("--hot-seed")
    ap.add_argument("--draft-hot")
    ap.add_argument("--served-model-name")
    ap.add_argument("--log-dir", default=os.path.join(os.getcwd(), "logs"))
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/.cache/kt-serve"))
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE, overrides any default")
    ap.add_argument("--server-arg", action="append", default=[], metavar="ARGS",
                    help="arguments passed on to sglang.launch_server as they are, e.g. --server-arg='--enable-hierarchical-cache --hicache-write-policy write_back --hicache-ratio 2'")
    ap.add_argument("--no-calibrate", action="store_true",
                    help="a setup's first launch measures the prefill plan on the first requests, not on prompts of "
                         "its own before it reports ready")
    ap.add_argument("--wrap", default="", help="run the server under this command (a profiler: 'nsys launch ...'); "
                                              "server.pid is then the wrapper's")
    ap.add_argument("--background", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-plan", action="store_true", help="the prefill lend alone: no KV loan, no chunk per request")
    ap.add_argument("--package", action="store_true",
                    help="Yuzuri as a package an unchanged SGLang loads at startup (yuzuri.pth in the venv: "
                         "tools/install-pth.py), not patches in the recipe's tree")
    ap.add_argument("--sglang", help="the SGLang checkout to serve (default: the recipe's third_party/sglang)")
    ap.add_argument("--features", help="the features on, comma-separated, or all / base / none (default: all; "
                                       "--list-features)")
    ap.add_argument("--without", help="features off, comma-separated")
    if "--list-features" in sys.argv[1:]:
        list_features()
        return 0
    a = ap.parse_args()
    feat_off, feat_on = pick_features(a.features, a.without)

    if a.profile == "auto":
        # a model no profile names: the command and environment from what the checkpoint holds (auto_profile.py)
        sys.path.insert(0, HERE)
        import auto_profile

        pf = None
        prof = auto_profile.build(os.path.abspath(a.config_dir), a.experts and os.path.abspath(a.experts),
                                  a.gguf and os.path.abspath(a.gguf), a.tp or 2)
        print(f"profile {prof['description']}", flush=True)
    else:
        pf = a.profile if a.profile.endswith(".json") else os.path.join(PROFILES, a.profile + ".json")
        prof = json.load(open(pf))
    w = prof["weights"]
    for k in ("recipe", "config_dir", "gguf", "experts", "draft_dir", "placement", "hot_seed", "draft_hot", "log_dir", "cache_dir"):
        v = getattr(a, k)
        if v:
            setattr(a, k, os.path.abspath(os.path.expanduser(v)))
    if not a.venv and not a.recipe:
        raise SystemExit("--venv <the server's venv> (or --recipe <checkout>) is needed")
    a.venv = os.path.abspath(os.path.expanduser(a.venv)) if a.venv else os.path.join(a.recipe, ".venv")
    # a venv inside a recipe checkout (its parent holds prebuilt/lib: kt-kernel's hwloc): that checkout's libraries,
    # toolkit and caches serve, as its own serve.sh gives them
    if not a.recipe and os.path.isdir(os.path.join(os.path.dirname(a.venv), "prebuilt", "lib")):
        a.recipe = os.path.dirname(a.venv)
    # the SGLang served: --sglang, the recipe's tree, else the one installed in the venv (no PYTHONPATH)
    a.tree = (os.path.abspath(os.path.expanduser(a.sglang)) if a.sglang
              else os.path.join(a.recipe, "third_party", "sglang") if a.recipe and not a.package else None)
    py = os.path.join(a.venv, "bin", "python")
    must = [(py, "venv python"), (os.path.join(a.config_dir, "config.json"), "config dir")]
    if a.tree:
        must.append((os.path.join(a.tree, "python", "sglang"), "sglang tree"))
    for p, what in must:
        if not os.path.exists(p):
            raise SystemExit(f"{what} not found: {p}")
    if a.package:
        pth = glob.glob(os.path.join(a.venv, "lib", "python3*", "site-packages", "yuzuri.pth"))
        want = os.path.join(os.path.dirname(HERE), "python")
        if not pth or want not in open(pth[0]).read().split("\n"):
            raise SystemExit(f"yuzuri.pth for {want} not in {a.venv}: run setup.sh --package {a.venv}")
    if not a.package and not a.recipe:
        raise SystemExit("without --package the recipe's patched tree serves: --recipe <checkout>")
    # libkv_vmm.so: the package's own (setup.sh --package builds it next to kt_kv_vmm.py), else the recipe's
    kv_vmm_lib = os.path.join(PACKAGE_TREE, "srt", "layers", "moe", "libkv_vmm.so")
    if not (a.package and os.path.exists(kv_vmm_lib)):
        kv_vmm_lib = os.path.join(a.recipe, "build", "libkv_vmm.so") if a.recipe else ""
    if not kv_vmm_lib or not os.path.exists(kv_vmm_lib):
        raise SystemExit("libkv_vmm.so not found: setup.sh --package builds it")
    cache_root = os.path.join(a.recipe, ".cache") if a.recipe else a.cache_dir
    experts_dir = a.gguf or a.experts or a.config_dir
    if w.get("gguf") and not a.gguf:
        raise SystemExit(f"profile {prof['name']} serves from a GGUF: pass --gguf DIR")
    if w.get("draft") == "dir" and a.draft_dir and not os.path.exists(os.path.join(a.draft_dir, "kt_mtp_gguf.json")):
        raise SystemExit(f"draft dir without kt_mtp_gguf.json: {a.draft_dir}")
    os.makedirs(a.cache_dir, exist_ok=True)
    cuda_home = os.environ.get("CUDA_HOME")
    if not cuda_home and a.recipe and os.path.exists(os.path.join(a.recipe, "build", "cuda-home", "bin", "nvcc")):
        cuda_home = os.path.join(a.recipe, "build", "cuda-home")  # the recipe's toolkit wheels
    if not cuda_home:
        nvcc = shutil.which("nvcc")
        cuda_home = os.path.dirname(os.path.dirname(nvcc)) if nvcc else None
    if not cuda_home and os.path.exists("/usr/local/cuda/bin/nvcc"):
        cuda_home = "/usr/local/cuda"
    if not cuda_home:
        # the venv's toolkit wheels last, as the recipe's serve.sh: their nvcc and the cccl headers torch's JIT reads
        # are separate wheels, and a venv may hold two minors of them (nvcc 13.4 against cccl 13.3: the all-reduce's
        # JIT stops at "CUDA compiler and CUDA toolkit headers are incompatible")
        wheels = glob.glob(os.path.join(a.venv, "lib", "python3*", "site-packages", "nvidia", "cu13"))
        cuda_home = wheels[0] if wheels and os.path.exists(os.path.join(wheels[0], "bin", "nvcc")) else "/usr/local/cuda"

    # --- host, defaults for this TP, model, plan
    host = json.loads(sh([sys.executable, os.path.join(HERE, "hostprobe.py"), "--json", "--mounts", experts_dir]))
    if host["gpu_in_use"]:
        raise SystemExit("GPUs in use: " + ", ".join(f"{u['name']}({u['pid']})" for u in host["gpu_in_use"]))
    gpus = sorted(host["gpus"], key=lambda g: g["numa_node"])
    tp = a.tp or int(sorted(prof["defaults"])[0])
    if str(tp) not in prof["defaults"]:
        raise SystemExit(f"profile {prof['name']} has no measured defaults for tp {tp} (has {list(prof['defaults'])})")
    if len(gpus) < tp:
        raise SystemExit(f"{len(gpus)} GPUs, tp {tp}")
    d = prof["defaults"][str(tp)]
    kv_dtype = a.kv_dtype or d["kv_dtype"]
    # the profile's state slots are one running request's; more running requests take as many times more
    mamba_slots = a.mamba_slots or d["mamba_slots"] * max(1, a.max_running_requests)
    pools = d["pools"]
    ranks = gpus[:tp]
    plan_cmd = [sys.executable, os.path.join(HERE, "plan-launch.py"), a.config_dir, "--tp", str(tp),
                "--kv-dtype", kv_dtype, "--mamba-slots", str(mamba_slots), "--embed-host", str(w.get("embed_host", 0)),
                "--min-hot", str(prof.get("min_hot", 1)), "--free-min-gb", str(a.free_min_gb if a.free_min_gb is not None else d.get("free_min_gb", 1.0)), "--json"]
    if a.context:
        plan_cmd += ["--context", str(a.context)]
    if prof.get("max_context"):
        plan_cmd += ["--max-context", str(prof["max_context"])]
    if a.gguf:
        plan_cmd += ["--gguf", a.gguf]
    elif a.experts and w.get("experts_from"):
        plan_cmd += ["--experts-from", a.experts]
    if w.get("draft") == "none" or not a.draft_dir:
        plan_cmd += ["--draft-bytes", "0"]
    elif w.get("draft_bytes_assumed"):
        plan_cmd += ["--draft-bytes", str(int(w["draft_bytes_assumed"] * 2**30 / tp))]  # the draft dir's share per rank
    # the profile's measured constants when they are for this TP:
    # they only make the plan's printed arithmetic closer; nothing sized by it moves with them
    cal_tp = str((prof.get("constants") or {}).get("from", {}).get("tp") or (list(prof["defaults"])[0] if len(prof["defaults"]) == 1 else ""))
    if prof.get("constants") and cal_tp == str(tp):
        plan_cmd += ["--constants", pf]
    plan = json.loads(sh(plan_cmd))
    context = context_length = plan["context"]
    mem_fraction = plan["mem_fraction"]
    names_off = {f["name"] for f in feat_off}
    if names_off & {"kv-vmm", "host-kv"}:
        # the pool at the context needs both: without them it is what the card holds, sized the way SGLang does
        context = context_length = plan["card_context"]
        if "kv-vmm" in names_off:
            # the pool is allocated whole, the way SGLang does it: the share is the plan's static budget of the card
            # (what a user of SGLang passes by hand), and what the pool leaves goes to fixed hot experts below
            mem_fraction = plan["card_fraction"]
    kv_lend = prof.get("env_by_tp", {}).get(str(tp), {}).get("KT_HOT_KV_LEND") == "1"
    # the KV hot lend (kv-vmm/kt_kv_hot_lend.py): the empty pool's pages hold KT_KV_HOT_EXTRA={lent} more
    # experts a layer outside prefill; it needs the plan's KV pool and at least one resident hot expert
    kv_hot = prof.get("env", {}).get("KT_KV_HOT_LEND") == "1"
    key = f"tp{tp}-ctx{context}-mamba{mamba_slots}-{os.path.basename(experts_dir.rstrip('/'))}"
    if a.hot is not None:
        hot = a.hot
    else:
        hot = plan["hot_fixed_with_kv_lend"] if kv_lend else plan["hot_experts_per_layer"]
        hot = max(hot, prof.get("min_hot", 1))
        per_layer = plan.get("expert_slot_per_rank", 0) * plan.get("moe_layers", 0)
        if "kv-vmm" in names_off and per_layer:
            # no lend: the fixed count is what the card holds beside the pool at this context, as a user sizes it
            room = plan["left_for_hot"] + (plan["context"] - context) * plan["kv_bytes_per_token"]
            hot = max(hot, int(room // per_layer))
    # the first window's chunk: small enough to fit any model, measured by the plan for the windows after it
    chunk = a.chunk or d.get("chunk") or 4096
    threads = a.threads or d.get("threads") or plan["threads"]
    # the lends, the KV loan and the plan come with the profile's environment (captured from the recipe's stack);
    # --no-plan keeps the lend alone. The plan's cost before this setup has measured its own: the profile's prior
    # (the recipe's V4.1 values when a profile has none: conservative on a model with a lighter row)
    # the plan (kt_prefill_plan.py): every request's chunk from the free memory, up to the cap; its cost a row is
    # measured from the first window this server runs, or taken from the profile when one
    # gives plan_cost_prior. A hybrid sliding-window pool is sized for the cap and lent outside prefill. --no-plan
    # keeps the lend alone at the launch chunk.
    vmm = os.path.join("srt", "layers", "moe", "kt_kv_vmm.py")
    plan_on = not a.no_plan and "kv-lend-plan" not in names_off and os.path.exists(os.path.join(PACKAGE_TREE if a.package else os.path.join(a.tree, "python", "sglang"), vmm))
    plan_env, plan_note = {}, None
    if plan_on:
        key += "-plan"
        # the ceiling is what these cards hold, measured by an earlier launch; the plan picks each chunk below it
        measured = measured_ceiling(a.cache_dir, key, context)
        cap = measured or FIRST_CEILING
        plan_env = {"KT_KV_VMM": "1", "KT_PREFILL_KV_LEND": "1", "KT_PREFILL_PLAN": "1",
                    "KT_PREFILL_PLAN_SAFE": str(chunk), "KT_PREFILL_PLAN_MARGIN_MB": "128",
                    "KT_PREFILL_LEND_MAX_CHUNK": str(cap)}
        if prof.get("plan_cost_prior"):
            plan_env["KT_PREFILL_PLAN_COST"] = prof["plan_cost_prior"]
        # a hybrid sliding-window pool (as SGLang's KV cache configurator finds it) holds the cap and is lent
        # outside prefill
        plan_env["KT_PREFILL_SWA_LEND"] = "auto"
        tree_srt = os.path.join(PACKAGE_TREE if a.package else os.path.join(a.tree, "python", "sglang"), "srt")
        carried = not prof.get("plan_cost_prior") and plan_carried(
            a.cache_dir, key, os.path.join(tree_srt, "layers", "moe"),
            stream_auto=prof.get("env", {}).get("KT_GPU_STREAM_PREFILL") == "auto")
        # nothing to carry: the plan measures on prompts of the launch's own before the server reports ready
        # (the prefill_plan warmup), when the tree has it
        has_warmup = False
        for path in ([os.path.join(os.path.dirname(PACKAGE_TREE), "hooks", "warmup_prefill.py")] if a.package else []) + [
                os.path.join(tree_srt, "entrypoints", "warmup.py")]:
            try:
                with open(path) as fh:
                    has_warmup = has_warmup or '@warmup("prefill_plan")' in fh.read()
            except OSError:
                pass
        calibrate = not carried and not prof.get("plan_cost_prior") and not a.no_calibrate and has_warmup
        # a carried plan measures nothing at launch, and the first long prompt then pays the kernel loads and the
        # staging of a first window (V4.1 on two cards: the first 35,000-token prompt 21.7 s, the second 19.0 s)
        warm = bool(carried) and has_warmup and not a.no_calibrate
        plan_note = ((f"prefill plan: the first window from the forwards an earlier launch measured" if carried else
                      f"prefill plan: the first window at {chunk}, the cost measured from it"
                      + (", on the launch's own prompts before it reports ready" if calibrate else ""))
                     + (f" (the profile's prior {prof['plan_cost_prior']})" if prof.get("plan_cost_prior") else "")
                     + f", chunk ceiling {cap}"
                     + (" (what an earlier launch measured these cards hold)" if measured
                        else " (the first launch's; the next takes what this one measures)"))
        chunk = cap
    else:
        plan_note = f"prefill lend at chunk {chunk}, no plan"
        calibrate = False
        warm = False
    vals = {"config_dir": a.config_dir, "experts_dir": experts_dir, "draft_dir": a.draft_dir or "", "port": a.port, "host": a.host, "tp": tp,
            "numa_nodes": " ".join(str(g["numa_node"]) for g in ranks), "context": context, "context_length": context_length,
            "mem_fraction": mem_fraction,
            "kv_cache_dtype": "fp8_e4m3" if kv_dtype == "fp8" else "bfloat16", "mamba_slots": mamba_slots, "hot": hot,
            "threads": threads, "pools": pools, "chunk": chunk, "cache_dir": a.cache_dir, "key": key,
            "hot_dynamic": "1" if hot else "0", "served_model_name": a.served_model_name or prof["name"],
            "placement": a.placement or "", "hot_seed": a.hot_seed or "", "draft_hot": a.draft_hot or "", "draft_gguf": "",
            "recipe": a.recipe or "", "kv_vmm_lib": kv_vmm_lib, "cache_root": cache_root, "lent": 0}
    if a.draft_dir and (w.get("draft") == "dir" or os.path.exists(os.path.join(a.draft_dir, "kt_mtp_gguf.json"))):
        with open(os.path.join(a.draft_dir, "kt_mtp_gguf.json")) as f:
            vals["draft_gguf"] = json.load(f)["gguf"]

    # --- environment and command from the profile
    env = dict(os.environ)
    env.update({"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": ",".join(str(g["index"]) for g in ranks),  # indices: SGLang parses this as integers
                "CUDA_HOME": cuda_home,
                # kt-kernel's hwloc and the recipe's small libraries, as its serve.sh
                "LD_LIBRARY_PATH": os.pathsep.join(([os.path.join(a.recipe, "prebuilt", "lib")] if a.recipe else [])
                                                   + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else [])),
                "PATH": os.path.join(a.venv, "bin") + os.pathsep + os.path.join(cuda_home, "bin") + os.pathsep + env.get("PATH", ""),
                "PYTHONPATH": os.pathsep.join(([os.path.join(a.tree, "python")] if a.tree else [])
                                              + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])),
                "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": str(threads),
                "YUZURI_PACKAGE": "1" if a.package else "0",
                # the runtime JITs (the LL all-reduce, kt's kernels) build under the recipe, as its serve.sh:
                # a build dir shared with another venv keeps a dead build's lock and every later launch waits
                "TORCH_EXTENSIONS_DIR": os.path.join(cache_root, "torch_extensions"),
                "SGLANG_CACHE_DIR": os.path.join(cache_root, "sglang"),
                # a multimodal model's image processor on the CPU: on the GPU the tokenizer process holds a CUDA
                # context of its own, outside every budget
                "SGLANG_MM_PROCESSOR_DEVICE": "cpu",
                # tier-1 25 and 24 for any model: the token embedding read from pinned host memory, the RoPE tables
                # mapped as the context grows
                "SGLANG_EMBED_ON_HOST": "1", "SGLANG_ROPE_LAZY": "1",
                # tier-1 06 for any multimodal model
                "SGLANG_MM_ON_DEMAND": "1"})
    # JIT builds at serve time (sgl-kernel / flashinfer modules compiled on first use) need the same glibc
    # compat header as the install on a glibc 2.41+ host with CUDA 13 (the kt-kernel build sets it the same way)
    compat = os.path.join(HERE, "cuda13-glibc-compat.h")
    if os.path.exists(compat) and "cuda13-glibc-compat" not in env.get("NVCC_PREPEND_FLAGS", ""):
        env["NVCC_PREPEND_FLAGS"] = (env.get("NVCC_PREPEND_FLAGS", "") + f" -U_GNU_SOURCE -D_DEFAULT_SOURCE -include {compat}").strip()
    # what every model runs with (auto_profile.COMMON_ENV), under the profile's own environment: a named profile
    # writes only what differs, and a switch added for all reaches every profile
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import auto_profile as _ap

    env.update(dict(zip(_ap.COMMON_ENV.keys(), fill(_ap.COMMON_ENV.values(), vals))))
    # the experts' format, from the command's --kt-method: its own switches (auto_profile.FORMAT_ENV) for every
    # profile, a named one's own environment over them
    args = fill(prof["command"], vals)
    method = args[args.index("--kt-method") + 1] if "--kt-method" in args[:-1] else ""
    env.update(_ap.FORMAT_ENV.get(method, {}))
    env.update(dict(zip(prof["env"].keys(), fill(prof["env"].values(), vals))))
    env.update(plan_env)
    if not plan_on:
        for k in [k for k in env if k.startswith(("KT_PREFILL_KV_LEND", "KT_PREFILL_PLAN", "KT_PREFILL_SWA"))]:
            env.pop(k)
    e_tp = prof.get("env_by_tp", {}).get(str(tp), {})
    env.update(dict(zip(e_tp.keys(), fill(e_tp.values(), vals))))
    for cond, block in (("draft_dir", "env_with_draft"), ("draft_hot", "env_with_draft_hot"), ("hot_seed", "env_with_hot_seed")):
        if getattr(a, cond) and block in prof:
            env.update(dict(zip(prof[block].keys(), fill(prof[block].values(), vals))))
    if not env.get("KT_MTP_GGUF", "x"):
        env.pop("KT_MTP_GGUF")  # a draft directory with its own safetensors
    # NVFP4 experts keep the checkpoint's one-byte e4m3 scale codes in kt-kernel's arenas when the installed
    # extension reads them (GeneralMOEConfig.scale_codes; the streamers and the hot lend then read the codes too),
    # else the scales folded to bf16: read from the venv, not set by a profile
    codes_note = None
    if method == "NVFP4":
        probe = subprocess.run(
            [py, "-c", "from kt_kernel.utils.amx import MOEConfig; print(hasattr(MOEConfig, 'scale_codes'))"],
            capture_output=True, text=True,
            env={**env, "CUDA_VISIBLE_DEVICES": "", "YUZURI_PACKAGE": "0"})
        takes = probe.stdout.strip().endswith("True")
        env["KT_SCALE_ARENA_CODES"] = "1" if takes else "0"
        codes_note = ("NVFP4: kt-kernel keeps the checkpoint's e4m3 scale bytes; GPU experts and streamed groups on "
                      "cuTile from them" if takes else
                      "NVFP4: this kt-kernel reads no e4m3 scale codes, so the CPU experts' block scales are folded "
                      "to bf16; the GPU stays FP4 (streamed groups on CUTLASS with the checkpoint's scale bytes), and "
                      "the hot lend's cuTile slots are off")
    # a feature off: its switches at their off values, over the profile's and the plan's, under --env
    for f in feat_off:
        env.update(f["off"])
    if feat_off:  # the workers' report names them: a hook stays attached while its switch is at the off value
        env["YUZURI_FEATURES_OFF"] = ",".join(f["name"] for f in feat_off)
    hooks_off = [h for f in feat_off for h in f.get("hooks_off", [])]
    if hooks_off:
        env["YUZURI_HOOKS_OFF"] = ",".join(hooks_off)
    for kv in a.env:
        k, _, v = kv.partition("=")
        env[k] = v
    env = {k: v for k, v in env.items() if v != ""}
    io_note = io_threads(env, experts_dir, a.cache_dir, [g["numa_node"] for g in ranks])
    cmd = [py]
    for arg in fill(prof["command"], vals):
        cmd += arg.split(" ") if arg == vals["numa_nodes"] else [arg]  # "--numa-node 0 1": one placeholder, two values
    if a.placement and "command_with_placement" in prof:
        cmd += fill(prof["command_with_placement"], vals)
    if a.draft_dir and "command_with_draft" in prof:
        cmd += fill(prof["command_with_draft"], vals)
    if warm and "--warmups" not in cmd:
        # two fresh prompts before ready: one of the plan's first chunk, and one of three chunks and eight rows (a row
        # count 16 does not divide selects other kernel variants than one it divides, and a longer one other tiles).
        # SGLANG_WARMUP_PREFILL_LENGTHS given by hand stands, and given empty there is no warmup
        given = "SGLANG_WARMUP_PREFILL_LENGTHS" in os.environ or any(
            kv.startswith("SGLANG_WARMUP_PREFILL_LENGTHS=") for kv in a.env)
        first = int(env.get("KT_PREFILL_PLAN_SAFE", "0") or 0)
        if not given and first > 0:
            env["SGLANG_WARMUP_PREFILL_LENGTHS"] = f"{first},{3 * first + 8}"
        if env.get("SGLANG_WARMUP_PREFILL_LENGTHS"):
            cmd += ["--warmups", "prefill_lengths"]
    if calibrate:
        if "--warmups" in cmd:
            cmd[cmd.index("--warmups") + 1] += ",prefill_plan"
        else:
            cmd += ["--warmups", "prefill_plan"]
    if mem_fraction is None and "--mem-fraction-static" in cmd:
        i = cmd.index("--mem-fraction-static")
        del cmd[i:i + 2]
    if a.max_running_requests < 1:
        raise SystemExit("--max-running-requests is 1 or more")
    if "--max-running-requests" in cmd:
        cmd[cmd.index("--max-running-requests") + 1] = str(a.max_running_requests)
    else:
        cmd += ["--max-running-requests", str(a.max_running_requests)]
    for more in a.server_arg:
        cmd += shlex.split(more)
    if a.wrap:
        cmd = shlex.split(a.wrap) + cmd
    # a model with linear-attention state keeps the plan's requests' worth of state slots, as SGLang counts a
    # request's, not the share of the budget SGLang gives them otherwise (GLM-5.3: 44 slots,
    # 3 GB a card)
    if ("--max-mamba-cache-size" not in cmd and "KT_MAMBA_REQUESTS" not in env
            and (plan.get("terms") or {}).get("linear_states", 0) > 0):
        env["KT_MAMBA_REQUESTS"] = str(mamba_slots)

    run = os.path.join(a.log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run, exist_ok=True)
    env["YUZURI_RUN_DIR"] = run  # each worker writes hooks-<pid>.json there: what every hook became
    manifest = {"profile": prof["name"], "command": cmd, "features": feat_on,
                "env": {k: v for k, v in env.items() if k.startswith(("KT_", "SGLANG_", "CUDA_", "OMP_", "NCCL_", "TORCH"))},
                "plan": plan, "host": {k: host[k] for k in ("hostname", "kernel", "driver", "cuda_toolkit")}, "packages": {}}
    for n in ("torch", "flashinfer-python", "triton", "kt-kernel", "sglang"):
        try:
            manifest["packages"][n] = md.version(n)
        except md.PackageNotFoundError:
            manifest["packages"][n] = None
    with open(os.path.join(run, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    lent_note = (f", lent ~{plan['lent_from_empty_pool']}" if kv_lend else
                 f", up to {vals['lent']} more on the empty pool" if vals["lent"] else "")
    print(f"{prof['name']} tp {tp}: hot {hot} per layer (plan {plan['hot_experts_per_layer']}{lent_note}), "
          f"chunk {chunk}, threads {threads} on nodes {[g['numa_node'] for g in ranks]}, context {context}"
          + (f" of the model's {plan['model_context']}" if context != plan["model_context"] else "")
          + f" (pool = context), KV {kv_dtype}, static "
          + (f"{mem_fraction} of the card" if mem_fraction is not None else "share of the card left to SGLang"))
    for n in plan["notes"] + ([plan_note] if plan_note else []):
        print("! " + n)
    if feat_off:
        print(f"features off: {', '.join(f['name'] for f in feat_off)}")
    if codes_note:
        print("! " + codes_note)
    if io_note:
        print("! " + io_note)
    print("log directory:", run)
    if a.dry_run:
        print(" ".join(cmd))
        return 0
    log = open(os.path.join(run, "server.log"), "ab")
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    with open(os.path.join(run, "server.pid"), "w") as f:
        f.write(str(proc.pid))
    if a.background:
        print(f"started pid {proc.pid}")
        print(f"tail the log for 'max_total_num_tokens=': the pool the server sized against context {context}")
        return 0
    try:
        while proc.poll() is None:
            time.sleep(2)
    except KeyboardInterrupt:
        proc.terminate()
    return proc.returncode or 0


if __name__ == "__main__":
    sys.exit(main())
