"""What each startup hook brings, and what became of it in this process.

TABLE names every hook module (hooks/), the feature it carries under the name the DeepSeek-V4.1 recipe's
docs/tier1.md gives it (or the tier 0 / tier 0.5 patch; "-" for what only Yuzuri has), and whether the server is
still right without it:
  optional  the hook makes the server faster or lighter. When this SGLang no longer has what the hook attaches to,
            the hook stays off, the launch goes on and the report says so.
  needed    without the hook the output is wrong or the server falls over: the launch stops, naming what is gone.
A tree that already carries the same mechanism as a patch keeps its own in both cases.

report() prints one line a hook once the model has loaded (`python -m yuzuri._features` prints the table alone):
  on        Yuzuri's hook is in
  tree      this SGLang carries the same mechanism: its own runs
  off       optional, and this SGLang has nothing for it to attach to (the reason follows)
  gone      needed, and the SGLang module the hook waits for is not in this tree at all: a model that goes through
            what the module held runs without the hook, so the report says it loudest
  idle      the SGLang module the hook waits for was not loaded by this launch (the model does not use it)
"""
import importlib.util
import sys

OPT, NEED = True, False
TABLE = {
    # hook module: (recipe name, optional, what it is)
    "fp8_decode_gemm": ("tier 1 01 fp8-decode", OPT, "decode-sized block-FP8 GEMMs on their own kernels"),
    "mxfp4_fp32": ("tier 1 02 gpu-moe-fp32", NEED, "MXFP4 GPU experts in fp32 between the GEMMs"),
    "marlin_fp32_glue": ("tier 1 02 gpu-moe-fp32", NEED, "the Marlin MoE's glue in fp32"),
    "mxfp4_runner_resident": ("tier 1 02 gpu-moe-fp32, 03 cutlass-kt-expert-count", NEED, "the MXFP4 CUTLASS method under kt-kernel"),
    "cutlass_shared_workspace": ("-", OPT, "one CUTLASS workspace a process"),
    "mm_encoder_on_demand": ("tier 1 06 vision-on-demand", OPT, "the multimodal encoder on the card only when called"),
    "mm_processor_device": ("tier 1 06 vision-on-demand", OPT, "the image processor's device"),
    "thinking_default": ("tier 1 07 thinking-default-parser", OPT, "the thinking default of explicit-toggle templates"),
    "load_model": ("tier 1 08 loading", OPT, "model loading"),
    "load_skip_cpu_experts": ("tier 1 08 loading", OPT, "the loader skips what the CPU experts hold, reads ahead"),
    "loader_seams": ("tier 1 08 loading, 13 prefill-lend", NEED, "the loader's part in the lend and the hot slots"),
    "model_runner_seams": ("tier 1 09, 13-17, 21 hicache, 24-26", NEED, "load and every forward: lend, hot cache, RoPE rows"),
    "swa_pool_guard": ("tier 1 10 pool-swa-guard", NEED, "a sliding-window pool that seats no request stops the launch"),
    "swa_chunk_in_flight": ("-", NEED, "SWA eviction between a prefill's chunks"),
    "kv_vmm_region": ("tier 1 12 kv-vmm", NEED, "the KV pool on remappable memory"),
    "kv_pool_seams": ("tier 1 13 prefill-lend, 15 kv-lend-plan, 17 host-kv", NEED, "the pool configurator's part"),
    "scheduler_seams": ("tier 1 13 prefill-lend, 15 kv-lend-plan, 17 host-kv", NEED, "the scheduler's part"),
    "lend_no_replay_graph": ("tier 1 13 prefill-lend", NEED, "no replay-tail graph over lent weights"),
    "moe_layer_kt_methods": ("tier 1 14 stream-prefill, 16 hot-lend, 26 gguf-lend", NEED, "FusedMoE's part in the expert methods"),
    "kt_ep_methods": ("tier 1 11 placement, 14 stream-prefill, 16 hot-lend, 18 stream-mmq, 19 cpu-gpu-overlap, 20 expert-cache, 26 gguf-lend", NEED, "the KT expert method"),
    "hicache_staging_cap": ("tier 1 21 hicache", OPT, "HiCache's write-back staging buffer within SGLANG_HICACHE_STAGING_MB"),
    "warmup_prefill": ("tier 1 22 prefill-warmup", OPT, "prefill warmups before the server reports ready"),
    "prefix_cache_replay_window": ("tier 1 23 swa-checkpoint-tail", OPT, "prefix cache hits past a chunk boundary"),
    "mqa_scratch_and_rope": ("tier 1 24 rope-tables", OPT, "rotary_emb not built, attention scratch"),
    "indexer_live_topk": ("tier 1 27 live-topk", OPT, "the indexer's decode on live positions only"),
    "kt_routed_scale_once": ("tier 0 10 routed-scale-once", NEED, "the routed scaling factor applied once"),
    "kt_shared_experts": ("-", NEED, "shared-experts fusion off under KT"),
    "host_allreduce": ("-", OPT, "large all-reduces through host memory"),
    "indexer_no_deepgemm_sm12x": ("-", NEED, "the torch indexer on SM12x"),
    "indexer_small_cards": ("-", OPT, "the indexer's score product reduced in place"),
    "sparse_mla_by_layout": ("tier 0 11 v41-kv-layout", NEED, "the sparse MLA kernel by the latent's layout"),
    "sparse_mla_window_calls": ("-", OPT, "FlashInfer's prepared calls of a prefill window dropped at its end"),
    "pre_sm90_sparse_attention": ("tier 0 13 sm89-attention", NEED, "DSV4 sparse attention on SM89 and SM75"),
    "pre_sm89_e4m3_round": ("tier 0 13 sm89-attention", NEED, "e4m3 rounding without Triton's type"),
    "pre_sm80_gemm": ("tier 0 13 sm89-attention", NEED, "bf16-in / fp32-out GEMM below SM80"),
    "pre_sm80_sparse_decode": ("tier 0 13 sm89-attention", NEED, "the fused sparse decode below SM80"),
    "gguf_dense": ("tier 0.5 01 gguf", NEED, "GGUF dense weights"),
    "gguf_grouped": ("tier 0.5 01 gguf", NEED, "grouped GGUF products"),
    "language_model_only_any": ("-", OPT, "--language-model-only for any model that reads it"),
}

_STATE = {}  # hook module -> [status, note]
_WAITS = {}  # hook module -> the SGLang modules it waits for
_REPORTED = False


def _short(fn) -> str:
    return getattr(fn, "__module__", "").rpartition(".")[2]


def optional(name: str) -> bool:
    return TABLE.get(name, ("", NEED, ""))[1]


def waits(fn, module_name: str) -> None:
    if _short(fn) in TABLE:
        _WAITS.setdefault(_short(fn), []).append(module_name)


def note(name: str, status: str, text: str = "") -> None:
    """off stays off; tree beside on is a hook made of several, some of them the tree's own."""
    have = _STATE.get(name)
    if have and have[0] == "off":
        return
    if have and have[0] != status and status != "off":
        status, text = "on", "; ".join(t for t in (have[1], text, "part of it is the tree's own"
                                                  if "tree" in (have[0], status) else "") if t)
    _STATE[name] = [status, text or (have[1] if have else "")]
    if _REPORTED and name in TABLE and (not have or have[0] != status):
        # a module first imported after the model loaded (FlashInfer's prepared calls come with the first attention
        # call): the report showed the hook idle, this line is its row
        recipe, opt, what = TABLE[name]
        print(f"[yuzuri-features] {status:4} {name} [{recipe}; {'optional' if opt else 'needed'}] "
              f"{_STATE[name][1] or what} (after the report: its module was first imported now)", flush=True)
        _write(rows())  # the file follows: a hook that came in with the first attention call is no longer idle there


def rows() -> list:
    out = []
    for name, (recipe, opt, what) in TABLE.items():
        status, text = _STATE.get(name, ["idle", ""])
        if status == "idle":
            gone = []
            for mod in _WAITS.get(name, []):
                if mod.endswith(".*") or mod in sys.modules:
                    continue
                try:
                    if importlib.util.find_spec(mod) is None:
                        gone.append(mod)
                except (ImportError, ValueError):
                    gone.append(mod)
            if gone:
                status, text = "off" if opt else "gone", f"this SGLang has no {', '.join(gone)}"
        out.append((name, recipe, "optional" if opt else "needed", status, text or what))
    return out


def _write(got: list) -> None:
    """hooks.json beside the launcher's manifest.json (YUZURI_RUN_DIR): what each hook became in this launch."""
    import json
    import os

    run = os.environ.get("YUZURI_RUN_DIR")
    if not run or not os.path.isdir(run):
        return
    try:
        path = os.path.join(run, f"hooks-{os.getpid()}.json")
        with open(path, "w") as f:
            json.dump({"features_off": [x for x in os.environ.get("YUZURI_FEATURES_OFF", "").split(",") if x],
                       "hooks": [dict(hook=n, recipe=r, kind=k, status=st, note=t) for n, r, k, st, t in got]},
                      f, indent=1)
    except OSError as e:
        print(f"[yuzuri-features] hooks.json not written: {e}", flush=True)


def report(force: bool = False) -> None:
    global _REPORTED
    if _REPORTED and not force:
        return
    _REPORTED = True
    got = rows()
    off = [r for r in got if r[3] == "off"]
    tree = [r for r in got if r[3] == "tree"]
    print(f"[yuzuri-features] {sum(r[3] == 'on' for r in got)} hooks on, {len(tree)} the tree's own, {len(off)} off, "
          f"{sum(r[3] == 'gone' for r in got)} gone, {sum(r[3] == 'idle' for r in got)} idle (names as in the recipe's docs/tier1.md)", flush=True)
    for name, recipe, kind, status, text in got:
        if status != "idle":
            print(f"[yuzuri-features] {status:4} {name} [{recipe}; {kind}] {text}", flush=True)
    gone = [r for r in got if r[3] == "gone"]
    if gone:
        print(f"[yuzuri-features] NEEDED AND WITHOUT THEIR MODULE in this SGLang: {', '.join(r[0] for r in gone)}. "
              f"SGLang moved or dropped what they attach to; a model that used it is not covered. Compare the output "
              f"with a launch without Yuzuri (--features base) before trusting this one", flush=True)
    import os

    switched = os.environ.get("YUZURI_FEATURES_OFF", "")
    if switched:
        print(f"[yuzuri-features] features switched off for this launch (--features / --without): "
              f"{switched.replace(',', ', ')}. A hook above that carries only these is attached and does nothing: "
              f"\"on\" is the hook, the feature is off", flush=True)
    idle = [r[0] for r in got if r[3] == "idle"]
    if idle:
        print(f"[yuzuri-features] idle (this model never loaded the SGLang module they attach to; they do nothing "
              f"here): {', '.join(idle)}", flush=True)
    _write(got)
    if off:
        print(f"[yuzuri-features] NOT RUNNING on this SGLang: {', '.join(r[0] for r in off)}. The server is right "
              f"without them and slower or heavier; the hooks need a look against this SGLang", flush=True)


def _report_after_load(m) -> None:
    """The report once the model is in, whatever became of the hooks on the runner itself."""
    orig = m.ModelRunner.load_model

    def load_model(self, *args, **kwargs):
        got = orig(self, *args, **kwargs)
        report()
        return got

    m.ModelRunner.load_model = load_model


def arm() -> None:
    from yuzuri._finder import after_import

    after_import("sglang.srt.model_executor.model_runner", _report_after_load)


if __name__ == "__main__":
    print("| hook | recipe | without it | what |\n|---|---|---|---|")
    for name, (recipe, opt, what) in TABLE.items():
        print(f"| `{name}` | {recipe} | {'launch goes on, reported' if opt else 'launch stops'} | {what} |")
