"""NVFP4 experts of a KT layer's GPU part through FlashInfer's cuTile MoE, from the checkpoint's bytes.

The layer's GPU method (compressed-tensors' or ModelOpt's NVFP4) creates and loads its parameters as before; this
method then leaves them as they are instead of repacking them (Marlin) or swizzling their scales:
- the e2m1 weights [E, 2I, H/2] (gate rows, then up rows) and [E, H, I/2];
- the e4m3 block scales row-major [E, 2I, H/16] and [E, H, I/16] (patches/flashinfer-cutile-fp4.py reads that
  layout);
- per expert, the weights' global multiplier: 1 / weight_global_scale (compressed-tensors) or weight_scale_2
  (ModelOpt), [E] or [E, 2] for gate and up, a few bytes made once.
The routes run at the checkpoint's own precision, read from the checkpoint (its quantization_config and tensor
names, never a model name; one model comes in several quantizations):
- weight-only NVFP4 (no activation quantization): W4A16;
- activations with static global scales (ModelOpt input_scale, compressed-tensors input_global_scale): W4A4 with
  them. The quantize kernel multiplies GEMM1's input by a [1] tensor holding 1 / g1 (patches/flashinfer-cutile-fp4.py
  part 7, the weight view's input_inv_global) and the alphas carry the rest: gate times g1, up times g1 / g2, w2
  times g2, which quantizes silu(gate) * up / g2 for GEMM2. g1 and g2 are the largest over the layer's
  experts, as SGLang's FlashInfer CUTLASS and TRT-LLM paths fold them. Folding g2 into up needs an activation
  linear in up (no clamp, no beta); otherwise, with g2 other than 1, the layer stays on the GPU method's own path;
- activations with dynamic global scales: the layer stays on the GPU method's own path until the kernel computes
  them.
A W4A4 checkpoint never runs W4A16. A route the CPU serves comes as id -1 and takes no row.

Layers with the same shape share one MoELayer. Its workspace per token bucket stays only for the buckets decode's
CUDA graphs reach; an eager call larger than those gives its workspace back after the call (as the Marlin runner's
intermediates go), on the lend scratch when only a streamed prefill (above KT_GPU_STREAM_PREFILL) reaches it.

Env:
  KT_CUTILE_NVFP4=0   the GPU method's own path
"""
import contextlib
import os

import torch

ENABLED = os.environ.get("KT_CUTILE_NVFP4", "1") == "1"

# (convention, parameter suffixes: weights, block scales, the weights' global, the activations' global, and
# whether the weights' global divides the block scales)
_CONVENTIONS = (
    ("compressed-tensors", ("weight_packed", "weight_scale", "weight_global_scale", "input_global_scale"), True),
    ("ModelOpt", ("weight", "weight_scale", "weight_scale_2", "input_scale"), False),
)
_LOGGED = set()


def _log(msg: str, once: str = "") -> None:
    if once:
        if once in _LOGGED:
            return
        _LOGGED.add(once)
    print(f"[kt-cutile] {msg}", flush=True)


def _params(layer):
    """(convention, {w13/w2: (weights, scales, weights' global, activations' global or None)}, divides) of a
    layer holding NVFP4 experts as their checkpoint stores them, else None."""
    for name, (w, s, g, i), divides in _CONVENTIONS:
        p = {}
        for proj in ("w13", "w2"):
            t = [getattr(layer, f"{proj}_{k}", None) for k in (w, s, g, i)]
            if any(x is None for x in t[:3]):
                break
            p[proj] = t
        else:
            if p["w13"][0].dtype == torch.uint8 and p["w13"][1].dtype == torch.float8_e4m3fn \
                    and p["w2"][0].dtype == torch.uint8 and p["w2"][1].dtype == torch.float8_e4m3fn:
                return name, p, divides
    return None


def _activation(cfg):
    """FlashInfer's activation for the layer's runner config, as the Marlin runner reads it, or (None, why)."""
    from flashinfer.fused_moe import SwiGLU

    act = str(getattr(cfg, "activation", None) or "silu").lower()
    if act != "silu" or not getattr(cfg, "is_gated", True):
        return None, f"activation {act!r} (gated {getattr(cfg, 'is_gated', True)})"
    alpha = getattr(cfg, "gemm1_alpha", None)
    if alpha is not None:
        limit = getattr(cfg, "gemm1_clamp_limit", None)
        if limit is None:
            return None, "gemm1_alpha without gemm1_clamp_limit"
        # swiglu_oai: gate * sigmoid(alpha * gate) * (up + 1), gate <= limit, |up| <= limit
        return SwiGLU(alpha=float(alpha), beta=1.0, limit=float(limit)), ""
    limit = getattr(cfg, "swiglu_limit", None)
    return (SwiGLU(limit=float(limit)) if limit else SwiGLU()), ""


def _safetensors_dir(weight_path):
    """The checkpoint directory the GPU method's NVFP4 experts come from: KT's weight path when it is a safetensors
    checkpoint, else the model's own (KT's experts from a GGUF while the GPU loads the model directory); None when
    neither has an index (the layer's own tensors decide)."""
    from sglang.srt.runtime_context import get_server_args

    for d in (weight_path, getattr(get_server_args(), "model_path", None)):
        if d and os.path.exists(os.path.join(d, "model.safetensors.index.json")):
            return d
    return None


def maybe_wrap(gpu_method, layer, weight_path=None, layer_idx=None):
    """The KT wrapper's GPU method, wrapped when the layer's experts are NVFP4 the cuTile MoE can run as they are;
    else gpu_method itself (with the reason in the log once)."""
    if not ENABLED:
        return gpu_method
    found = _params(layer)
    if found is None:
        return gpu_method
    conv, p, _ = found
    weight_path = _safetensors_dir(weight_path)
    acts = checkpoint_activations(weight_path) if weight_path else ("static" if p["w13"][3] is not None else "none")
    if acts == "dynamic":
        _log(f"{conv} NVFP4 experts with dynamic activation global scales stay on {type(gpu_method).__name__}: "
             f"cuTile's W4A4 does not compute them yet", once="dynamic")
        return gpu_method
    act, why = _activation(getattr(layer, "moe_runner_config", None))
    if act is None:
        _log(f"{conv} NVFP4 experts stay on {type(gpu_method).__name__}: {why}", once="act")
        return gpu_method
    # the layer's static activation globals from the checkpoint: the layer's own input-scale parameters hold only
    # the GPU experts' values, the others stay at their initial 1 (Flash-Next NVFP4: g1 read 1 for 0.001825)
    g12 = (1.0, 1.0)
    if acts == "static" and weight_path and layer_idx is not None:
        g12 = checkpoint_layer_globals(weight_path).get(int(layer_idx), (1.0, 1.0))
        _cpu_a4(int(layer_idx), *g12)
    return CutileNvfp4Method(gpu_method, act, acts, g12)


def _inner_up_first(inner) -> bool:
    """Whether the wrapped method's own kernels take w13 up rows first (asked once its runner config is set)."""
    try:
        return bool(getattr(inner, "load_up_proj_weight_first", False))
    except AttributeError:
        return False


def _swap_w13_halves(p) -> None:
    """w13's weights and block scales from gate-then-up rows to up-then-gate, in place, an expert at a time."""
    for t in (p["w13"][0], p["w13"][1]):
        n = t.shape[1] // 2
        for e in range(t.shape[0]):
            gate = t[e, :n].clone()
            t[e, :n].copy_(t[e, n:])
            t[e, n:].copy_(gate)


def _alpha(g: torch.Tensor, divides: bool, experts: int) -> torch.Tensor:
    """The weights' global multiplier per expert: [E], or [E, 2] (gate, up) when the two differ."""
    a = g.detach().float().reshape(experts, -1)
    a = 1.0 / a if divides else a
    if a.shape[1] == 1 or torch.equal(a[:, 0], a[:, 1]):
        return a[:, 0].contiguous()
    return a[:, :2].contiguous()


CPU_A4 = os.environ.get("KT_CPU_A4", "1") == "1"
_CPU_A4_SAID = [False]


def _cpu_a4(layer_idx: int, g1: float, g2: float) -> None:
    """The CPU experts of a W4A4 checkpoint quantize their inputs as the card does: kt-kernel's knobs
    KT_CPU_A4_G13_L<layer> / KT_CPU_A4_G2_L<layer> take this layer's activation globals (float bits; 0 leaves the
    inputs as they are). Without it kt-kernel computes W4A16: the reference of the checkpoint's own arithmetic
    (tools/ref-nll-hf.py) scores 1.6015 on Flash-Next NVFP4, the reference without activation quantization 1.5874,
    and a server with every expert on the CPU scored 1.5894, so the answer moved with the share of experts in hot
    slots. KT_CPU_A4=0 leaves the CPU side W4A16."""
    if not CPU_A4:
        return
    import struct

    try:
        import kt_kernel.kt_kernel_ext as ext

        set_knob = ext.set_knob
    except (ImportError, AttributeError) as exc:
        if not _CPU_A4_SAID[0]:
            _CPU_A4_SAID[0] = True
            print(f"[kt-cutile] this kt-kernel takes no activation globals ({type(exc).__name__}): its CPU experts "
                  "compute W4A16 beside the card's W4A4", flush=True)
        return
    for name, g in ((f"KT_CPU_A4_G13_L{layer_idx}", g1), (f"KT_CPU_A4_G2_L{layer_idx}", g2)):
        set_knob(name, struct.unpack("<i", struct.pack("<f", float(g)))[0])
    if not _CPU_A4_SAID[0]:
        _CPU_A4_SAID[0] = True
        print(f"[kt-cutile] the CPU experts quantize their inputs to FP4 as the card does (layer {layer_idx}: "
              f"globals {g1:.4g} / {g2:.4g})", flush=True)


def checkpoint_activations(path: str) -> str:
    """How the checkpoint's NVFP4 experts take their activations: "none" (weight-only), "static" (global scales
    stored per expert) or "dynamic" (global scales computed at run time). From quantization_config's
    input_activations (compressed-tensors and ModelOpt both write it) and, without one, the experts' tensor names."""
    import json

    def qcfg(name):
        try:
            c = json.load(open(os.path.join(path, name)))
        except (OSError, ValueError):
            return {}
        return c.get("quantization_config") or c.get("text_config", {}).get("quantization_config") or \
            c.get("quantization") or {}

    q = qcfg("config.json") or qcfg("hf_quant_config.json")
    acts = [g.get("input_activations") for g in (q.get("config_groups") or {}).values() if isinstance(g, dict)]
    if any(a and a.get("dynamic") for a in acts):
        return "dynamic"
    if any(acts):
        return "static"
    idx = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
    if any(".experts." in k and k.endswith((".input_global_scale", ".input_scale")) for k in idx):
        # named but not calibrated: every stored activation global exactly 1 is a placeholder (one checkpoint's
        # experts; its own method runs them W4A16), and W4A4 under it would clip each block at e4m3's range
        checkpoint_layer_globals(path)
        return "none" if _UNIT_GLOBALS.get(path) else "static"
    return "none"


_LAYER_GLOBALS = {}
_UNIT_GLOBALS = {}  # path -> whether every stored expert activation global is exactly 1


def checkpoint_layer_globals(path: str) -> dict:
    """{layer index: (g1, g2)} from the checkpoint's expert activation scales (static ones), the largest over a
    layer's experts: gate and up for GEMM1's input, down for GEMM2's. The tensor name says the layer, the role
    (gate_proj / up_proj / down_proj, or w1 / w3 / w2) and whether the value divides (input_global_scale)."""
    import json
    import re

    from safetensors import safe_open

    if path in _LAYER_GLOBALS:
        return _LAYER_GLOBALS[path]
    idx = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
    pat = re.compile(r"\.layers\.(\d+)\..*\.experts\.\d+\.(gate_proj|up_proj|down_proj|w1|w3|w2)\."
                     r"(input_scale|input_global_scale)$")
    out, files, unit = {}, {}, True
    for k, fn in idx.items():
        m = pat.search(k)
        if not m:
            continue
        f = files.get(fn) or files.setdefault(fn, safe_open(os.path.join(path, fn), "pt"))
        t = f.get_tensor(k).float()
        unit = unit and bool((t == 1.0).all())
        v = float(t.max() if m.group(3) == "input_scale" else (1.0 / t).max())
        layer = int(m.group(1))
        g = list(out.get(layer, (0.0, 0.0)))
        i = 1 if m.group(2) in ("down_proj", "w2") else 0
        g[i] = max(g[i], v)
        out[layer] = tuple(g)
    _LAYER_GLOBALS[path] = {L: tuple(x or 1.0 for x in g) for L, g in out.items()}
    _UNIT_GLOBALS[path] = unit and bool(out)
    return _LAYER_GLOBALS[path]


def _linear_in_up(act) -> bool:
    """act(gate) * up with nothing else on up (no clamp, no beta): scaling up scales the product."""
    return float(getattr(act, "beta", 0.0) or 0.0) == 0.0 and float(getattr(act, "limit", 3e38) or 3e38) >= 3e38


def fold_globals(a13: torch.Tensor, a2: torch.Tensor, g1: float, g2: float, experts: int) -> tuple:
    """The alphas carrying the activation globals: gate times g1, up times g1 / g2, w2 times g2."""
    if g1 == 1.0 and g2 == 1.0:
        return a13, a2
    a13 = a13.reshape(experts, -1).expand(experts, 2).clone()
    a13[:, 0] *= g1
    a13[:, 1] *= g1 / g2
    return a13.contiguous(), (a2 * g2).contiguous()


_MOE = {}

# rows per cuTile call (the streamer's KT_GPU_STREAM_ROW_BLOCK, else 8192): the runner's workspace follows the
# row count's bucket, a power of two past 8192, so a 13312-row chunk in one call made the 16384 bucket's
# (1.5 GiB more than 8192's, TP2, 11-08) and the plan's per-row cost no longer held
ROWS = int(os.environ.get("KT_GPU_STREAM_ROW_BLOCK", "0") or 0) or 8192


def _ceiling() -> int:
    """Tokens a forward can hand the layer: the prefill chunk (larger forwards are split)."""
    try:
        from sglang.srt.runtime_context import get_schedule

        return max(1, int(get_schedule().chunked_prefill_size or 0) or 8192)
    except Exception:
        return 8192


def _moe(experts: int, hidden: int, inter: int, top_k: int, act, a4: bool, device):
    """One MoELayer per shape, shared by the layers (their workspaces are per token bucket)."""
    key = (experts, hidden, inter, top_k, repr(act), a4, str(device))
    m = _MOE.get(key)
    if m is None:
        from flashinfer.fused_moe import (
            BackendOptions, CuTileNvfp4Bf16Config, CuTileNvfp4Config, ExecutionConfig, ExpertConfig, MoEConfig,
            MoELayer, QuantConfig, QuantFormat, RoutingConfig,
        )

        qc = QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4 if a4 else QuantFormat.BF16)
        be = CuTileNvfp4Config() if a4 else CuTileNvfp4Bf16Config()
        ceiling = _ceiling()
        cfg = MoEConfig(routing=RoutingConfig(num_experts=experts, top_k=top_k), quant=qc,
                        experts=ExpertConfig(intermediate_size=inter), activation=act,
                        backend=BackendOptions(candidates=(be,)),
                        execution=ExecutionConfig(tune_max_num_tokens=ceiling))
        m = _MOE[key] = (MoELayer(cfg, device=device), ceiling)
    return m


def _graph_max_bs() -> int:
    try:
        from sglang.srt.runtime_context import get_exec

        return int(get_exec().graph.cuda_graph_config.decode.max_bs or 0)
    except Exception:
        return 0


def _release(moe, n: int, ceiling: int) -> None:
    """An eager call's workspace back to torch's cache, as the Marlin runner's intermediates go back: FlashInfer's
    runner keeps a workspace per token bucket, which after a lend took the free memory left a short extend's logits
    no room (chain143). The buckets decode's CUDA graphs reach stay (a replay reads them, a capture cannot make
    one)."""
    from flashinfer.fused_moe.utils import map_to_hybrid_bucket

    if torch.cuda.is_current_stream_capturing():
        return
    cap = map_to_hybrid_bucket(n, ceiling)
    if cap <= map_to_hybrid_bucket(max(1, _graph_max_bs()), ceiling):
        return
    for r in moe.runners:
        cache = getattr(r, "_workspace_cache", None)
        if not cache:
            continue
        for key in [k for k in cache if k[0] == cap]:
            if getattr(r, "_workspace", None) is cache[key]:
                r._workspace = None
            del cache[key]


def _scratch(n: int, ceiling: int):
    """The lend scratch for a token bucket only a streamed prefill reaches (decode's buckets stay plain)."""
    from flashinfer.fused_moe.utils import map_to_hybrid_bucket

    from sglang.srt.layers.moe import kt_lend
    from sglang.srt.layers.moe import kt_stream_threshold

    t = kt_stream_threshold.get()
    if t <= 0 or n <= t or map_to_hybrid_bucket(n, ceiling) <= map_to_hybrid_bucket(t, ceiling):
        return contextlib.nullcontext()
    return kt_lend.scratch_region()


_STREAM = {}


def stream_runner(experts: int, hidden: int, inter: int, top_k: int, act, a4: bool, ceiling: int, device):
    """A cuTile FP4 runner of the streamer's own for its group slots. Its workspace per token bucket stays (the group
    graphs bake its addresses in); the streamer's eager call before a capture makes it, on the lend scratch."""
    key = (experts, hidden, inter, top_k, repr(act), a4, ceiling, str(device))
    r = _STREAM.get(key)
    if r is None:
        from flashinfer.fused_moe import (
            BackendOptions, CuTileNvfp4Bf16Config, CuTileNvfp4Config, ExecutionConfig, ExpertConfig, MoEConfig,
            QuantConfig, QuantFormat, RoutingConfig,
        )
        from flashinfer.fused_moe.runners import CuTileNvfp4Bf16Runner, CuTileNvfp4Runner

        qc = QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4 if a4 else QuantFormat.BF16)
        be = CuTileNvfp4Config() if a4 else CuTileNvfp4Bf16Config()
        cfg = MoEConfig(routing=RoutingConfig(num_experts=experts, top_k=top_k), quant=qc,
                        experts=ExpertConfig(intermediate_size=inter), activation=act,
                        backend=BackendOptions(candidates=(be,)),
                        execution=ExecutionConfig(tune_max_num_tokens=ceiling))
        r = (CuTileNvfp4Runner if a4 else CuTileNvfp4Bf16Runner)(cfg, device)
        r.check_support()
        r.build()
        _STREAM[key] = r
    return r


def stream_run(runner, x, ids, weights, w13, w13_scale, w13_alpha, w2, w2_scale, w2_alpha, out, inv_g1=None) -> None:
    """The group's MoE into out (the caller's buffer, so a capture allocates nothing): x [T, H] bf16, ids [T, k] int32
    local to the group (-1: another group's or the CPU's), weights [T, k] fp32 (0 where ids is -1), the slots'
    checkpoint bytes (row-major e4m3 scales) and per-expert global multipliers; inv_g1 a [1] tensor, GEMM1's input
    activation global's inverse (patches/flashinfer-cutile-fp4.py part 7), read when the graph replays."""
    from sglang.srt.layers.moe import kt_lend

    # the eager call before a capture makes the bucket's workspace (on the scratch); a capture allocates nothing
    ctx = contextlib.nullcontext() if torch.cuda.is_current_stream_capturing() else kt_lend.scratch_region()
    with ctx:
        runner.forward([out, x, ids, weights, w13, w13_scale, w13_alpha, w2, w2_scale, w2_alpha]
                       + ([inv_g1] if inv_g1 is not None else []), tactic=-1)


class CutileNvfp4Method:
    """The GPU method of a KT layer's NVFP4 experts on FlashInfer's cuTile MoE; everything but the processing after
    loading, the runner and apply is the wrapped method's."""

    def __init__(self, inner, activation, acts="static", g12=(1.0, 1.0)):
        self.__dict__["inner"] = inner
        self.activation = activation
        self.acts = acts  # checkpoint_activations: "none" or "static"
        self.g12 = g12  # the layer's static activation globals (g1, g2), from the checkpoint
        self.moe_runner_config = None

    @property
    def load_up_proj_weight_first(self) -> bool:
        """w13 as the checkpoint names it, gate rows then up: FlashInfer's cuTile MoE reads the first half as gate.
        The loader asks this while loading (layer._load_w13); ModelOpt's own answer, up first for its CUTLASS
        layout, needs the runner config, which the wrap comes before (its property raised there, so a check of it
        at wrap time read False while the loads put up first: Flash-Next NVFP4 ran gate and up swapped, 11-09)."""
        return False

    def __getattr__(self, name):
        inner = self.__dict__.get("inner")
        if inner is None:  # copy.copy looks up hooks before the dict is filled
            raise AttributeError(name)
        return getattr(inner, name)

    def create_weights(self, layer, **kwargs):
        return self.inner.create_weights(layer=layer, **kwargs)

    def process_weights_after_loading(self, layer) -> None:
        """The checkpoint's bytes stay; only the per-expert global multipliers are made."""
        conv, p, divides = _params(layer)
        experts = int(p["w13"][0].shape[0])
        a4 = self.acts != "none"
        g1, g2 = self.g12 if a4 else (1.0, 1.0)
        if g2 != 1.0 and not _linear_in_up(self.activation):
            _log(f"{conv} NVFP4 experts with GEMM2 activation global {g2:.4g} and an activation with a clamp or beta "
                 f"on up ({self.activation}) stay on {type(self.inner).__name__}: the global cannot ride on up's alpha",
                 once="unfoldable")
            layer.__dict__["_kt_cutile"] = {"inner": True}
            if _inner_up_first(self.inner):
                _swap_w13_halves(p)  # loaded gate first (load_up_proj_weight_first above), as the method wants up first
            return self.inner.process_weights_after_loading(layer)
        a13 = _alpha(p["w13"][2], divides, experts).to(p["w13"][0].device)
        a2 = _alpha(p["w2"][2], divides, experts).to(p["w2"][0].device)
        layer.kt_cutile_w13_alpha, layer.kt_cutile_w2_alpha = fold_globals(a13, a2, g1, g2, experts)
        layer.kt_cutile_inv_g1 = torch.full((1,), 1.0 / g1, dtype=torch.float32, device=p["w13"][0].device)
        for proj in ("w13", "w2"):
            # the method's swizzled block-scale copy (ModelOpt's create_weights makes one) is never read here: an
            # alias of the checkpoint's scales, as the method's TRTLLM path makes it, so neither the resident rows
            # nor the hot lend's slots hold the bytes twice
            name = f"{proj}_blockscale_swizzled"
            if isinstance(getattr(layer, name, None), torch.Tensor) and getattr(layer, name) is not p[proj][1]:
                setattr(layer, name, p[proj][1])
        layer.__dict__["_kt_cutile"] = {"a4": a4, "pack": None}
        if not getattr(layer, "kt_hot_slots", False):
            what = ("W4A16 (the checkpoint does not quantize activations)" if not a4 else
                    f"W4A4 with the activations' static globals g1 {g1:.4g}, g2 {g2:.4g}")
            _log(f"{conv} NVFP4 experts on cuTile {what} from the checkpoint's bytes ({experts} experts, w13 "
                 f"{tuple(p['w13'][0].shape)}, block scales row-major)", once="on")

    def create_moe_runner(self, layer, moe_runner_config) -> None:
        self.moe_runner_config = moe_runner_config
        # the method's own runner too: a layer whose activation global scales are not 1 runs on it
        self.inner.create_moe_runner(layer, moe_runner_config)

    def _pack(self, layer):
        """The layer's weight pack, made again when a tensor moved (the hot lend's slots map under the shadow)."""
        _, p, _ = _params(layer)
        ts = (p["w13"][0], p["w13"][1], layer.kt_cutile_w13_alpha, p["w2"][0], p["w2"][1], layer.kt_cutile_w2_alpha,
              layer.kt_cutile_inv_g1)
        key = tuple(t.data_ptr() for t in ts)
        st = layer.__dict__["_kt_cutile"]
        if st["pack"] is None or st["pack"][0] != key:
            from flashinfer.fused_moe import MoEWeightPack

            wp = MoEWeightPack()
            wp.prepare_for("cutile_nvfp4", {
                "w1": ts[0].data, "w1_scale": ts[1].data, "w1_global_scale": ts[2],
                "w2": ts[3].data, "w2_scale": ts[4].data, "w2_global_scale": ts[5], "input_inv_global": ts[6]})
            st["pack"] = (key, wp)
        return st["pack"][1], ts

    def apply(self, layer, dispatch_output):
        from flashinfer.fused_moe import MoEActivationPack
        from flashinfer.fused_moe.core import RoutingInputMode

        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        if layer.__dict__["_kt_cutile"].get("inner"):
            return self.inner.apply(layer, dispatch_output)
        x = dispatch_output.hidden_states
        topk = dispatch_output.topk_output
        ids = topk.topk_ids
        ids = ids if ids.dtype == torch.int32 else ids.to(torch.int32)
        w = topk.topk_weights.float()
        w = torch.where(ids < 0, torch.zeros_like(w), w)  # the CPU's routes add nothing
        wp, ts = self._pack(layer)
        moe, ceiling = _moe(int(ts[0].shape[0]), int(x.shape[1]), int(ts[3].shape[2]) * 2, int(ids.shape[1]),
                            self.activation, layer.__dict__["_kt_cutile"]["a4"], x.device)
        outs = []
        step = min(ceiling, ROWS)
        for lo in range(0, x.shape[0], step):
            hi = min(x.shape[0], lo + step)
            act = MoEActivationPack(x[lo:hi], None, ids[lo:hi], w[lo:hi],
                                    routing_input_mode=RoutingInputMode.PackedPrecomputed)
            with _scratch(hi - lo, ceiling):
                outs.append(moe(act, wp))
            _release(moe, hi - lo, ceiling)
        out = outs[0] if len(outs) == 1 else torch.cat(outs)
        rsf = getattr(self.moe_runner_config, "routed_scaling_factor", None)
        if rsf is not None and float(rsf) != 1.0:
            out.mul_(float(rsf))  # where the Marlin runner applies it
        return StandardCombineInput(hidden_states=out)


def base_method(method):
    """The method a CutileNvfp4Method wraps, else method: what checks on the GPU method's class look at."""
    return method.__dict__["inner"] if isinstance(method, CutileNvfp4Method) else method
