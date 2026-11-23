#!/usr/bin/env python3
"""A profile for a model no profile names (serve.py --profile auto): the SGLang command and the environment built
from what the checkpoint holds, not from a list of models.

  * the experts' format (kt-kernel's --kt-method): the dtype of one expert's weight and the names and dtypes of its
    scales, read from the safetensors headers (a GGUF directory: LLAMAFILE);
  * the GPU MoE runner for that format (SGLang's own choice for any other);
  * reasoning and tool-call parsers: SGLang's "auto" (from the chat template);
  * linear-attention state slots when the model has linear layers (modelprobe);
  * the streamed prefill's group: as many experts as keep about KT_AUTO_GROUP_MB (800) on a rank, counting
    everything the streamer keeps there (its two slots, and the bf16 group buffer where it expands the group);
  * the environment every measured profile shares, and the format's own streamer switches.

Usage: auto_profile.py <config dir> [--experts DIR | --gguf DIR] [--tp N] [--recipe DIR]   (prints the profile JSON)
"""
import argparse
import json
import os
import re
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# what every model runs with, under any profile (serve.py lays it under the profile's own environment)
COMMON_ENV = {
    "KT_EXPERT_SHM": "1", "KT_KV_VMM": "1", "KT_PREFILL_LEND": "1",
    "KT_PREFILL_LEND_HINT": "{cache_dir}/lend-hint-{key}",
    "KT_KV_HOT_LEND": "1", "KT_KV_HOT_EXTRA": "auto", "KT_KV_HOT_SPARE_FLOOR_MB": "1024",
    "KT_KV_HOT_DYNAMIC_SLOTS": "1", "KT_KV_HOT_DYNAMIC_RESIDENT": "1",
    "NCCL_IB_DISABLE": "1", "NCCL_P2P_DISABLE": "1", "SGLANG_HOST_STAGED_ALLREDUCE": "1", "SGLANG_LL_ALLREDUCE": "1",
    "KT_STREAM_MEMOPS": "1", "KT_TASKQUEUE_SPIN_US": "300", "KT_LOAD_PREFETCH": "4",
    # tier-1 19 (cpu-gpu-overlap): the hand-off packed by one kernel, the shared expert forked after the CPU request
    "KT_HANDOFF_PACK": "1", "KT_SHARED_AFTER_SUBMIT": "1",
    "KT_KV_VMM_LIB": "{kv_vmm_lib}", "KT_GPU_STREAM_DROP_CACHE": "{config_dir}",
    # the barrier after loading: a rank pinning its expert arenas can take minutes past the other (SGLang's 480 s)
    "SGLANG_UNBALANCED_LOAD_TIMEOUT_S": "1800",
    # the prompt length a prefill streams from is measured at the first launch (kt_stream_threshold)
    "KT_GPU_STREAM_PREFILL": "auto",
    # the streamer and its card buffers are made at load. Made at the first forward that qualifies, they came after
    # the hot lend had counted the card's free memory: an MXFP4 launch on one card kept 0.00 GiB of the lend's
    # 1 GiB floor and the first 1,024-token forward ran out (11-17)
    "KT_GPU_STREAM_EARLY_INIT": "1",
    # every streamed kernel call over blocks of this many rows: the CUTLASS workspaces are sized by it, not by the
    # chunk ceiling, so the ceiling is what the card holds (serve.py) and costs no scratch it does not use
    "KT_GPU_STREAM_ROW_BLOCK": "8192",
}
# the streamer's switches each format runs with in the measured profiles
FORMAT_ENV = {
    # KT_MXFP4_SCALE_CODES: kt-kernel's CPU decode reads a one-byte code a group (the checkpoint's e8m0 exponent)
    # beside the FP32 arena the streamer reads; an extension without the codes reads no such variable
    "MXFP4": {"KT_GPU_STREAM_W16": "1", "KT_GPU_STREAM_ZEROCOPY": "1", "KT_GPU_STREAM_GRAPH": "1",
              "KT_GPU_STREAM_MMQ": "1", "KT_GPU_STREAM_HOST_BUFS": "3", "KT_WRITE_UP_FIRST": "1", "KT_GPU_MASK_ZERO": "1",
              "KT_MXFP4_SCALE_CODES": "1",
              # two bits a group in the decode loop where a matrix holds at most four scales (kt-kernel decides
              # per matrix at load; bit-identical). As above: an extension without it reads no such variable
              "KT_MXFP4_SCALE_CODES2": "1",
              # kt-kernel's CPU decode: the activation row converted with PSHUFB, the weights prefetched 1 KB ahead
              # (bit-identical; V4.1 MXFP4 decode 51.7 -> 55.6 tok/s on two EPYC 7452, 11-14). As above: an
              # extension without them reads no such variable
              "KT_MXFP4_DEC_VCONV": "1", "KT_MXFP4_DEC_PFS": "1024"},
    # KT_MXFP4_SCALE_CODES: the same one-byte codes for NVFP4's block scales (kt-kernel's FP4 kernel serves both;
    # a matrix has at most 127 e4m3 values times one scale_2). Flash-Next, every expert on the CPU: 43.9 -> 47
    # tok/s, the same tokens (11-23)
    "NVFP4": {"KT_GPU_STREAM_ROW_BLOCK": "8192", "KT_GPU_STREAM_SCALE_BYTES": "1", "KT_MXFP4_SCALE_CODES": "1"},
    "BF16": {"KT_GPU_STREAM_GRAPH": "1"},
    "LLAMAFILE": {"KT_GGUF_NOCOPY": "1", "KT_GGUF_ZEROCOPY": "1", "KT_GGUF_STREAM_MMQ": "1",
                  "KT_GGUF_STREAM_SOURCE": "bank",
                  # kt-kernel's copy of the experts and the ranks' banks are placed by first touch: under SGLang's
                  # hard bind they all land on rank 0's node, however much the other nodes have free
                  "SGLANG_NUMA_BIND_V2": "0"},
}
# SGLang's MoE runner for the GPU experts of a format (none: SGLang chooses)
FORMAT_ARGS = {
    "MXFP4": ["--moe-runner-backend", "marlin"],
    "NVFP4": ["--moe-runner-backend", "flashinfer_cutlass", "--fp4-gemm-backend", "flashinfer_cutlass"],
    "LLAMAFILE": ["--moe-runner-backend", "flashinfer_cutlass"],
}


def _declares_modelopt(d):
    """The checkpoint declares ModelOpt quantization (hf_quant_config.json, or quant_method modelopt). SGLang's
    FlashInfer CUTLASS MoE takes only ModelOpt's quantizations; another NVFP4 declaration (compressed-tensors, a
    model's own method) keeps the runner SGLang and the model choose."""
    if os.path.exists(os.path.join(d, "hf_quant_config.json")):
        return True
    try:
        c = json.load(open(os.path.join(d, "config.json")))
    except (OSError, ValueError):
        return False
    q = c.get("quantization_config") or (c.get("text_config") or {}).get("quantization_config") or {}
    return str(q.get("quant_method", "")).lower() == "modelopt"


def _headers(d):
    """{tensor name: (dtype, shape)} of the safetensors files of d."""
    idx = os.path.join(d, "model.safetensors.index.json")
    files = sorted(set(json.load(open(idx))["weight_map"].values())) if os.path.exists(idx) else \
        [f for f in os.listdir(d) if f.endswith(".safetensors")]
    out = {}
    for f in files:
        with open(os.path.join(d, f), "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            for k, v in json.loads(fh.read(n)).items():
                if k != "__metadata__":
                    out[k] = (v["dtype"], v["shape"])
    return out


def _names_experts(d):
    """Whether the safetensors under d name any MoE expert tensor."""
    r = re.compile(r"^(?:.*\.)?layers\.\d+\.(?:[^.]+\.)*experts\.")
    return any(r.match(k) for k in _headers(d))


def expert_format(d):
    """kt-kernel's method for the experts under d, from one expert's tensors: (method, why)."""
    if any(f.endswith(".gguf") for f in os.listdir(d)):
        return "LLAMAFILE", "GGUF files"
    h = _headers(d)
    # one expert's tensors per layer (named per expert, or stacked with the experts first); a checkpoint may keep a
    # draft layer in another format, so the layers vote
    per = re.compile(r"^((?:.*\.)?layers\.(\d+)\.(?:[^.]+\.)*experts)\.0\.([^.]+)\.([^.]+)$")
    stacked = re.compile(r"^((?:.*\.)?layers\.(\d+)\.(?:[^.]+\.)*experts)\.([^.]+)$")
    layers = {}
    for k, (dt, shape) in h.items():
        m = per.match(k)
        if m:
            layers.setdefault(m.group(2), {}).setdefault(m.group(4), dt)
    if not layers:
        for k, (dt, shape) in h.items():
            m = stacked.match(k)
            if m and len(shape) == 3:
                layers.setdefault(m.group(2), {}).setdefault("weight", dt)
    if not layers:
        raise SystemExit(f"no MoE experts under {d}: no tensor named ...layers.<n>...experts... in its safetensors "
                         f"headers. Yuzuri moves a KTransformers server's experts and KV pages between the CPU and the "
                         f"cards; a dense model has none on the CPU. Experts kept elsewhere: --experts DIR or --gguf DIR.")
    votes = {}
    for suffix in layers.values():
        f = _format_of(suffix)
        votes[f] = votes.get(f, 0) + 1
    best = max(votes, key=votes.get)
    return best[0], best[1] + (f" ({votes[best]} of {len(layers)} layers)" if len(votes) > 1 else "")


def _format_of(suffix):
    """(method, why) from one layer's {tensor suffix: dtype}."""
    w = suffix.get("weight") or suffix.get("weight_packed")
    if w in ("BF16", "F16", "F32"):
        return "BF16", f"expert weights {w}"
    if w == "F8_E4M3":
        return "FP8", "expert weights F8_E4M3"
    scale = suffix.get("weight_scale") or suffix.get("scale") or suffix.get("scales")
    if "weight_scale_2" in suffix or "weight_global_scale" in suffix or scale == "F8_E4M3":
        return "NVFP4", f"packed weights {w}, block scales {scale}, a second-level scale"
    return "MXFP4", f"packed weights {w}, block scales {scale}"


def _has_weights(d):
    """The directory holds a weight file SGLang's loader reads."""
    return any(f.endswith((".safetensors", ".bin", ".pt")) for f in os.listdir(d))


def probe(d):
    return json.loads(subprocess.run([sys.executable, os.path.join(HERE, "modelprobe.py"), d, "--json"],
                                     capture_output=True, text=True, check=True).stdout)


def _gguf_block(t):
    """Values in one block of a GGML type: the unit a quantized row can be cut at."""
    return 1 if t in (0, 1, 30) else 32 if t in (2, 3, 6, 7, 8, 9) else 256


def shared_expert_cut(gguf, tp):
    """A shared expert's down projection whose input width, split over tp ranks, is not whole GGUF blocks:
    (name, width, block) of the first one, else None. Such a model keeps the shared expert whole on every rank."""
    if tp < 2:
        return None
    sys.path.insert(0, HERE)
    import modelprobe

    files = sorted(os.path.join(gguf, f) for f in os.listdir(gguf) if f.endswith(".gguf")) if os.path.isdir(gguf) else [gguf]
    for path in files:
        for name, _n, t, shape in modelprobe.gguf_tensors(path, dims=True):
            if "down_shexp" in name and len(shape) == 2 and (shape[0] % tp or shape[0] // tp % _gguf_block(t)):
                return name, shape[0], _gguf_block(t)
    return None


def build(config_dir, experts_dir=None, gguf=None, tp=2):
    src = gguf or experts_dir or config_dir
    method, why = expert_format(src)
    p = probe(config_dir)
    c = p.get("config", {})
    e = (probe(src) if src != config_dir and not gguf else p).get("experts") or {}
    per_rank = (e.get("bytes_per_expert_mean") or 0) / max(1, tp)
    fmt_env = FORMAT_ENV.get(method, {})
    no_hot = ""
    if experts_dir and experts_dir != config_dir and not gguf and _has_weights(config_dir) and not _names_experts(config_dir):
        # the hot slots are filled from the expert tensors the server's own loader hands each layer; a checkpoint
        # that keeps none (its experts live only under --experts) leaves them empty, so the slots are not made
        no_hot = "the config directory's own weights hold no expert tensor, and a hot expert is filled from them"
    if per_rank and fmt_env.get("KT_GPU_STREAM_W16") == "1":
        # the W16 streamer keeps, for every expert of a group, two packed DMA slots and the bf16 copy the kernel
        # reads (three matrices of hidden x intermediate), plus the MMQ buffers: count what stays on the card,
        # not the packed file (V4.1 MXFP4: 8.8 MB packed, 62 MB kept an expert a rank)
        H, I = c.get("hidden_size") or 0, c.get("moe_intermediate_size") or 0
        bf16 = (3 * H * I * 2) / max(1, tp) if H and I else 4 * per_rank
        kept = 2 * per_rank + bf16 + (per_rank if fmt_env.get("KT_GPU_STREAM_MMQ") == "1" else 0)
    elif method == "NVFP4" and no_hot:
        # with no expert resident on the card the layer holds no activation scale, and the NVFP4 streamer expands
        # each group to bf16: it runs a quarter of the group it is given, in two packed slots of that quarter and a
        # bf16 buffer four times one of them (GLM, 2026-11-17: the group counted as packed slots alone ran at 8)
        kept = (2 * per_rank + 4 * per_rank) / 4
    else:
        kept = 2 * per_rank  # two device slots
    group = 16
    if kept:
        budget = int(os.environ.get("KT_AUTO_GROUP_MB", "800")) << 20
        group = max(4, min(64, 1 << max(0, int(budget // kept).bit_length() - 1)))
    cmd = ["-m", "sglang.launch_server", "--model-path", "{config_dir}", "--served-model-name", "{served_model_name}",
           "--host", "{host}", "--port", "{port}", "--tp", "{tp}", "--numa-node", "{numa_nodes}",
           "--disable-custom-all-reduce", "--trust-remote-code", "--kv-cache-dtype", "{kv_cache_dtype}",
           "--context-length", "{context_length}", "--max-total-tokens", "{context}",
           "--mem-fraction-static", "{mem_fraction}", "--max-running-requests", "1",
           "--chunked-prefill-size", "{chunk}", "--max-prefill-tokens", "{chunk}",
           "--kt-weight-path", "{experts_dir}", "--kt-method", method, "--kt-num-gpu-experts", "{hot}",
           "--kt-cpuinfer", "{threads}", "--kt-threadpool-count", "{pools}", "--kt-max-deferred-experts-per-token", "0",
           "--sampling-backend", "pytorch", "--disable-flashinfer-autotune", "--watchdog-timeout", "1800",
           "--enable-metrics", "--cuda-graph-backend-prefill", "disabled",
           "--reasoning-parser", "auto", "--tool-call-parser", "auto"]
    if method != "NVFP4" or _declares_modelopt(config_dir):
        cmd += FORMAT_ARGS.get(method, [])
    linear = bool(c.get("linear_state_bytes_per_seq_bf16"))
    if linear:
        cmd += ["--max-mamba-cache-size", "{mamba_slots}"]
    env = dict(COMMON_ENV)
    env.update(FORMAT_ENV.get(method, {}))
    note = ""
    env["KT_GPU_STREAM_GROUP"] = str(group)
    if no_hot:
        env["KT_KV_HOT_LEND"] = "0"
        note = "; no hot expert, fixed or lent: no expert tensor in the config directory's own weights"
    if gguf and not _has_weights(config_dir):
        # a config directory with no weight file (tools/make-config-dir.py): the dense tensors come from the GGUF
        env["KT_GGUF_DENSE"] = "{experts_dir}"
        if c.get("ple"):
            # the config declares a per-layer token table: from a GGUF it is a sparse file mapping the CPU reads
            # (kt_gguf_dense_qwen4exp), so SGLang's check that the card reads pageable memory does not apply
            cmd += ["--ple-offload-embedding", "--ple-offload-backend", "file"]
            env["SGLANG_QWEN4_PLE_FILE_SKIP_DEVICE_CHECK"] = "1"
        cut = shared_expert_cut(gguf, tp)
        if cut:
            # a GGUF block cannot be cut: the shared expert whole on every rank
            env["SGLANG_SHARED_EXPERT_TP1"] = "1"
            note += f"; shared expert whole on every rank ({cut[0]}: {cut[1]} inputs over {tp} ranks are not blocks of {cut[2]})"
    return {
        "name": os.path.basename(config_dir.rstrip("/")),
        "description": f"auto: {c.get('architecture')}, experts {method} ({why}), stream group {group}{note}",
        "command": cmd, "command_with_placement": ["--init-expert-location", "{placement}"],
        "env": env, "env_by_tp": {str(tp): {}},
        "weights": {"gguf": bool(gguf), "experts_from": bool(experts_dir and experts_dir != config_dir) or bool(gguf),
                    "embed_host": 0, "draft": "auto"},
        # --draft-dir DIR: one draft step a pass; the draft's own tensors come from a GGUF when DIR names one
        "command_with_draft": ["--speculative-algorithm", "NEXTN", "--speculative-num-steps", "1", "--speculative-eagle-topk", "1",
                               "--speculative-num-draft-tokens", "2", "--speculative-draft-model-path", "{draft_dir}"],
        "env_with_draft": {"KT_MTP_GGUF": "{draft_gguf}"},
        "defaults": {str(tp): {"kv_dtype": c.get("kv_store") or "bf16", "mamba_slots": 8 if linear else 1, "pools": tp, "chunk": None,
                               "threads": None, "free_min_gb": 1.0}},
        "min_hot": 0 if no_hot else 1, "no_hot": no_hot,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config_dir")
    ap.add_argument("--experts")
    ap.add_argument("--gguf")
    ap.add_argument("--tp", type=int, default=2)
    a = ap.parse_args()
    print(json.dumps(build(a.config_dir, a.experts, a.gguf, a.tp), indent=2))


if __name__ == "__main__":
    main()
