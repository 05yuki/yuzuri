#!/usr/bin/env python3
"""Per-token log-probabilities from the model's implementation in transformers, streamed through one GPU.

  python tools/ref-nll-hf.py <model dir> <out.json> [--modelopt <dir with nvidia-modelopt>] [--fp32-experts]

A reference for checkpoints that ship no inference code of their own: the class the checkpoint's config names
(`architectures[0]`) is built without weights, and each block of the language model gets its weights from the
safetensors files for the time of its forward only, so a model far larger than the card is scored on it. Nothing of
SGLang, KTransformers or Yuzuri runs. It scores tools/nll_pieces.py's 24 pieces in one batch and writes
tools/nll.py's format:

  python tools/nll.py compare ref.json served.json

What is not transformers' code, and why:
  - A table split into shards in the checkpoint (`<name>.shard_<i>.weight`, a per-layer n-gram table of 96 GiB): its
    rows are gathered from a memory map of the files instead of being loaded.
  - ModelOpt NVFP4 experts (`experts.<i>.<proj>.weight` as packed bytes with `weight_scale`, `weight_scale_2`,
    `input_scale`): transformers has no such layer. The experts' forward keeps transformers' routing and order
    (gate and up from the input, the activation times up, down, the routing weight) and takes the arithmetic from
    NVIDIA ModelOpt, the quantizer that wrote the checkpoint (`NVFP4QTensor`): weights dequantized from the packed
    e2m1 values, the e4m3 block scales and the tensor's scale_2; each projection's input quantized to FP4 in blocks
    of 16 with its static `input_scale` and dequantized, as the checkpoint's W4A4 definition says. The products are
    computed in fp32. --modelopt names a directory holding the nvidia-modelopt package.
  - --fp32-experts: unquantized experts in fp32 as well (default: the model's own dtype).
  - --no-act-quant leaves the checkpoint's definition (it skips the activation quantization): for telling apart a
    server that computes its experts W4A16 from one that is wrong.
"""
import argparse
import json
import mmap
import os
import re
import struct
import sys
import time

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nll_pieces  # noqa: E402

DT = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16, "I64": torch.int64, "U8": torch.uint8,
      "F8_E4M3": torch.float8_e4m3fn, "I32": torch.int32, "F64": torch.float64, "BOOL": torch.bool}


class Files:
    """The checkpoint's tensors by name, read from memory maps of the safetensors files."""

    def __init__(self, model_dir):
        self.dir = model_dir
        self.where = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
        self.maps = {}

    def _file(self, fn):
        got = self.maps.get(fn)
        if got is None:
            f = open(os.path.join(self.dir, fn), "rb")
            n = struct.unpack("<Q", f.read(8))[0]
            head = json.loads(f.read(n))
            got = self.maps[fn] = (mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ), 8 + n, head)
        return got

    def __contains__(self, key):
        return key in self.where

    def view(self, key):
        """A CPU tensor over the file's bytes (read-only memory; copy or move before writing)."""
        mm, base, head = self._file(self.where[key])
        h = head[key]
        a, b = h["data_offsets"]
        t = torch.frombuffer(memoryview(mm)[base + a:base + b], dtype=DT[h["dtype"]]) if b > a else torch.empty(0, dtype=DT[h["dtype"]])
        return t.view(h["shape"])


class ShardedRows(nn.Module):
    """An embedding whose weight the checkpoint holds as equal shards: rows gathered from the files, on the CPU."""

    def __init__(self, files, prefix, n):
        super().__init__()
        self.shards = [files.view(f"{prefix}.shard_{i}.weight") for i in range(n)]
        self.rows = self.shards[0].shape[0]
        self.weight = torch.zeros(1, dtype=self.shards[0].dtype)  # transformers reads its device (cpu)

    def forward(self, ids):
        flat = ids.reshape(-1).cpu()
        out = torch.empty((flat.numel(), self.shards[0].shape[1]), dtype=self.shards[0].dtype)
        which = torch.div(flat, self.rows, rounding_mode="floor")
        for s in which.unique().tolist():
            sel = (which == s).nonzero().squeeze(1)
            out[sel] = self.shards[s][flat[sel] - s * self.rows]
        return out.view(*ids.shape, -1)


def build(cls, config):
    """The model with every parameter on the meta device (no memory) and its computed buffers real."""
    from transformers.initialization import TORCH_INIT_FUNCTIONS

    saved = {name: getattr(nn.init, name) for name in TORCH_INIT_FUNCTIONS}
    old_reg = nn.Module.register_parameter
    old_empty, old_zeros, old_ones = torch.empty, torch.zeros, torch.ones

    def reg(module, name, param):
        if param is not None and param.device.type != "meta":
            param = nn.Parameter(torch.empty(param.shape, dtype=param.dtype, device="meta"), requires_grad=False)
        old_reg(module, name, param)

    def lazy(fn):
        # a parameter's first tensor is made on meta when it is large: nn.Embedding(320M, 160) would take 205 GiB
        def make(*size, **kw):
            shape = size[0] if len(size) == 1 and isinstance(size[0], (tuple, list, torch.Size)) else size
            n = 1
            for d in shape:
                n *= int(d)
            if n >= (1 << 22) and "device" not in kw:
                kw["device"] = "meta"
            return fn(*size, **kw)
        return make

    try:
        for name in saved:
            setattr(nn.init, name, lambda t, *a, **k: t)
        nn.Module.register_parameter = reg
        torch.empty, torch.zeros, torch.ones = lazy(old_empty), lazy(old_zeros), lazy(old_ones)
        model = cls(config)
    finally:
        for name, fn in saved.items():
            setattr(nn.init, name, fn)
        nn.Module.register_parameter = old_reg
        torch.empty, torch.zeros, torch.ones = old_empty, old_zeros, old_ones
    return model.eval()


def tensors(module, prefix):
    """(owner module, attribute, is parameter, checkpoint key) of every parameter and buffer under module."""
    for mname, m in module.named_modules():
        p = prefix + ("." + mname if mname else "")
        for name in list(m._parameters):
            if m._parameters[name] is not None:
                yield m, name, True, f"{p}.{name}"
        for name in list(m._buffers):
            if m._buffers[name] is not None:
                yield m, name, False, f"{p}.{name}"


class Unit:
    """A block whose weights are on the card only while it runs."""

    def __init__(self, module, prefix, files, dev, stats):
        self.m, self.prefix, self.files, self.dev, self.stats = module, prefix, files, dev, stats
        self.inner = module.forward
        module.forward = self

    def __call__(self, *a, **k):
        t0 = time.time()
        held, missing = [], []
        for owner, name, is_param, key in tensors(self.m, self.prefix):
            if key in self.files:
                t = self.files.view(key).to(self.dev)
                if is_param:
                    held.append((owner, name, owner._parameters[name]))
                    owner._parameters[name] = nn.Parameter(t, requires_grad=False)
                else:
                    owner._buffers[name] = t
            elif is_param:
                missing.append(key)
            else:
                owner._buffers[name] = owner._buffers[name].to(self.dev)  # computed at construction
        if missing:
            raise SystemExit(f"{self.prefix}: no tensor in the checkpoint for {missing[:4]} ({len(missing)} in all)")
        try:
            return self.inner(*a, **k)
        finally:
            for owner, name, meta in held:
                owner._parameters[name] = meta
            self.stats["units"] += 1
            self.stats["s"] += time.time() - t0


def nvfp4_experts(experts, prefix, files, dev, modelopt_dir, quantize_acts=True, layer_scale=False):
    """experts.forward for ModelOpt NVFP4 experts: transformers' loop, ModelOpt's arithmetic (see the docstring)."""
    if modelopt_dir:
        sys.path.insert(0, modelopt_dir)
    try:
        from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor
    except ImportError as exc:
        raise SystemExit(f"NVFP4 experts need NVIDIA ModelOpt for their arithmetic ({exc}): "
                         "pip install --no-deps --target <dir> nvidia-modelopt omegaconf pulp, then --modelopt <dir>")

    def weight(key):
        packed = files.view(key + ".weight").to(dev)
        shape = (packed.shape[0], packed.shape[1] * 2)
        q = NVFP4QTensor(shape, torch.float32, packed)
        return q.dequantize(dtype=torch.float32, scale=files.view(key + ".weight_scale").to(dev),
                            double_scale=files.view(key + ".weight_scale_2").to(dev).float(), block_sizes={-1: 16})

    maxes = {}

    def layer_max(proj):
        # --layer-input-scale: one global for gate and up and one for down, the largest over the layer's experts
        kind = "down" if proj == "down_proj" else "gate_up"
        if kind not in maxes:
            names = ("down_proj",) if kind == "down" else ("gate_proj", "up_proj")
            vals = [files.view(f"{prefix}.{e}.{nm}.input_scale").float() for e in range(experts.num_experts) for nm in names]
            maxes[kind] = torch.stack(vals).max().to(dev)
        return maxes[kind]

    def act(x, key):
        if not quantize_acts or key + ".input_scale" not in files:
            return x  # the checkpoint quantizes no activation here (W4A16), or --no-act-quant
        s2 = layer_max(key.rsplit(".", 1)[1]) if layer_scale else files.view(key + ".input_scale").to(dev).float()
        q, s, s2 = NVFP4QTensor.quantize(x.clone(), 16, None, s2)
        return q.dequantize(dtype=torch.float32, scale=s, double_scale=s2, block_sizes={-1: 16})

    def forward(hidden_states, top_k_index, top_k_weights):
        out = torch.zeros_like(hidden_states)
        for e in top_k_index.unique().tolist():
            pos, tok = torch.where((top_k_index == e).t())
            x = hidden_states[tok].float()
            p = f"{prefix}.{e}"
            gate = nn.functional.linear(act(x, p + ".gate_proj"), weight(p + ".gate_proj"))
            up = nn.functional.linear(act(x, p + ".up_proj"), weight(p + ".up_proj"))
            h = experts.act_fn(gate) * up
            y = nn.functional.linear(act(h, p + ".down_proj"), weight(p + ".down_proj"))
            y = y * top_k_weights[tok, pos, None].float()
            out.index_add_(0, tok, y.to(out.dtype))
        return out

    return forward


def fp32_experts(experts):
    inner = experts.forward

    def forward(hidden_states, top_k_index, top_k_weights):
        gu, dn = experts._parameters["gate_up_proj"], experts._parameters["down_proj"]
        experts._parameters["gate_up_proj"] = nn.Parameter(gu.float(), requires_grad=False)
        experts._parameters["down_proj"] = nn.Parameter(dn.float(), requires_grad=False)
        try:
            return inner(hidden_states.float(), top_k_index, top_k_weights.float()).to(hidden_states.dtype)
        finally:
            experts._parameters["gate_up_proj"], experts._parameters["down_proj"] = gu, dn

    return forward


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir")
    ap.add_argument("out")
    ap.add_argument("--modelopt")
    ap.add_argument("--fp32-experts", action="store_true")
    ap.add_argument("--no-act-quant", action="store_true",
                    help="NVFP4 experts without the activation quantization the checkpoint defines: not the reference, a "
                         "comparison for a server whose experts compute W4A16")
    ap.add_argument("--layer-input-scale", action="store_true",
                    help="NVFP4 activations quantized with one input scale a layer (the largest over its experts, gate "
                         "and up together, down apart) instead of each projection's own: not the reference, what "
                         "kernels with one global a layer compute")
    ap.add_argument("--pieces", type=int, help="score only the first N pieces (a quick check)")
    a = ap.parse_args()
    import transformers

    t_start = time.time()
    dev = torch.device("cuda")
    config = transformers.AutoConfig.from_pretrained(a.model_dir)
    cls = getattr(transformers, config.architectures[0])
    files = Files(a.model_dir)
    model = build(cls, config)
    base = model.base_model_prefix  # "model"
    lm = model.get_decoder() if hasattr(model, "get_decoder") else getattr(model, base)
    lm_prefix = next(n for n, m in model.named_modules() if m is lm)
    stats = {"units": 0, "s": 0.0}
    notes = []

    # tables the checkpoint holds as shards, and quantized experts: found by the tensors that are there
    for name, m in list(lm.named_modules()):
        p = f"{lm_prefix}.{name}"
        if isinstance(m, nn.Embedding) and f"{p}.weight" not in files and f"{p}.shard_0.weight" in files:
            n = len({k for k in files.where if k.startswith(p + ".shard_")})
            owner = lm.get_submodule(name.rsplit(".", 1)[0])
            owner._modules[name.rsplit(".", 1)[1]] = ShardedRows(files, p, n)
            notes.append(f"{p}: {n} shards gathered from the files")
        if "gate_up_proj" in m._parameters and f"{p}.gate_up_proj" not in files:
            if f"{p}.0.gate_proj.weight_scale" not in files:
                raise SystemExit(f"{p}: neither stacked experts nor per-expert quantized tensors in the checkpoint")
            m.forward = nvfp4_experts(m, p, files, dev, a.modelopt, not a.no_act_quant, a.layer_input_scale)
            for nm in ("gate_up_proj", "down_proj"):
                del m._parameters[nm]
            if not any("NVFP4" in x for x in notes):
                acts = f"{p}.0.gate_proj.input_scale" in files and not a.no_act_quant
                notes.append("experts: ModelOpt NVFP4, " + ("activations quantized (W4A4)" if acts else "activations as they are (W4A16)"))
        elif "gate_up_proj" in m._parameters and a.fp32_experts:
            m.forward = fp32_experts(m)

    # the units: each element of a module list on its own, every other child whole
    for name, child in lm.named_children():
        if isinstance(child, nn.ModuleList):
            for i, m in enumerate(child):
                Unit(m, f"{lm_prefix}.{name}.{i}", files, dev, stats)
        elif any(True for _ in child.parameters()) or any(True for _ in child.buffers()):
            Unit(child, f"{lm_prefix}.{name}", files, dev, stats)
    direct = [n for n in lm._parameters if lm._parameters[n] is not None]
    for n in direct:
        lm._parameters[n] = nn.Parameter(files.view(f"{lm_prefix}.{n}").to(dev), requires_grad=False)

    tok = nll_pieces.tokenizer(a.model_dir)
    ids = nll_pieces.pieces(tok)
    if a.pieces:
        ids = ids[:a.pieces]
    x = torch.tensor(ids, device=dev)
    print(f"{config.architectures[0]} from transformers {transformers.__version__}; {len(ids)} pieces of {x.shape[1]} tokens", flush=True)
    for n in notes:
        print("  " + n, flush=True)
    with torch.no_grad():
        out = lm(input_ids=x, use_cache=False)
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        head = model.get_output_embeddings()
        hp = next(n for n, m in model.named_modules() if m is head)
        w = files.view(f"{hp}.weight").to(dev)
        result = []
        for b in range(x.shape[0]):
            lp = torch.log_softmax(nn.functional.linear(h[b].to(w.dtype), w).float(), dim=-1)
            result.append(lp[:-1].gather(1, x[b, 1:, None]).squeeze(1).tolist())
    json.dump(result, open(a.out, "w"))
    n = sum(len(r) for r in result)
    print("scored %d tokens in %d pieces, mean NLL %.4f (%.0f s, %d blocks streamed)"
          % (n, len(result), -sum(sum(r) for r in result) / n, time.time() - t_start, stats["units"]))


if __name__ == "__main__":
    main()
