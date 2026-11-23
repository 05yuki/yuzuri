"""kt_nvfp4_decode against its torch reference on a checkpoint's own NVFP4 experts (one layer, this rank's half),
and its time against the bytes it reads. Needs a CUDA card and an NVFP4 ModelOpt checkpoint:

  python tests/gpu/test_nvfp4_decode.py <checkpoint dir> [layer]
"""
import importlib.util
import json
import os
import sys
import time

import torch
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, "..", "..", "python", "yuzuri", "tree", "srt", "layers", "moe", "kt_nvfp4_decode.py")
spec = importlib.util.spec_from_file_location("kt_nvfp4_decode", P)
dec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dec)


class Act:
    alpha, beta, limit, is_gated, type = 1.0, 0.0, 3.4e38, True, "Swiglu"


def main():
    d, layer = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 5
    idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
    tpl = next(k for k in idx if f".layers.{layer}.mlp.experts.0.gate_proj.weight" in k).replace(".0.gate_proj.weight", ".{e}.{p}")
    dev = torch.device("cuda")
    E, TOPK, tp = 24, 10, 2

    def get(e, p):
        k = tpl.format(e=e, p=p)
        with safe_open(os.path.join(d, idx[k]), "pt") as f:
            return f.get_tensor(k)

    w13, s13, w2, s2, a13, a2 = [], [], [], [], [], []
    g1 = g2 = 0.0
    for e in range(E):
        g, u, dn = get(e, "gate_proj.weight"), get(e, "up_proj.weight"), get(e, "down_proj.weight")
        N = g.shape[0] // tp
        w13.append(torch.cat((g[:N], u[:N])))
        s13.append(torch.cat((get(e, "gate_proj.weight_scale")[:N], get(e, "up_proj.weight_scale")[:N])))
        w2.append(dn[:, : N // 2].contiguous())
        s2.append(get(e, "down_proj.weight_scale")[:, : N // 16].contiguous())
        g1 = max(g1, float(get(e, "gate_proj.input_scale")), float(get(e, "up_proj.input_scale")))
        g2 = max(g2, float(get(e, "down_proj.input_scale")))
        a13.append(torch.tensor([float(get(e, "gate_proj.weight_scale_2")) * g1, float(get(e, "up_proj.weight_scale_2")) * g1 / g2]))
        a2.append(torch.tensor([float(get(e, "down_proj.weight_scale_2")) * g2]))
    w13, s13, w2, s2 = (torch.stack(t).to(dev) for t in (w13, s13, w2, s2))
    a13, a2 = torch.stack(a13).float().to(dev), torch.cat(a2).float().to(dev)
    inv = torch.tensor([1.0 / g1], device=dev)
    K = w13.shape[2] * 2
    torch.manual_seed(0)
    worst = 0.0
    for M in (1, 2, 4):
        x = (torch.randn(M, K, device=dev) * 0.6).bfloat16()
        ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]).int()
        ids[0, 3] = -1  # a CPU route
        w = torch.softmax(torch.randn(M, TOPK, device=dev), -1)
        w = torch.where(ids < 0, torch.zeros_like(w), w)
        for a4 in (True, False):
            got = dec.run(x, ids, w, w13, s13, a13, w2, s2, a2, inv, a4, Act())
            want = dec.ref(x, ids, w, w13, s13, a13, w2, s2, a2, inv, a4, Act())
            rel = float((got.float() - want.float()).norm() / want.float().norm())
            worst = max(worst, rel)
            print(f"M {M} A4 {a4}: relative difference to the reference {rel:.2e}, max abs "
                  f"{float((got.float() - want.float()).abs().max()):.3e} of {float(want.float().abs().max()):.3e}")
    # against FlashInfer's cuTile MoE (YUZURI_PACKAGE=1: with Yuzuri's patches/flashinfer-cutile-fp4.py)
    try:
        from flashinfer.fused_moe import (BackendOptions, CuTileNvfp4Config, ExecutionConfig, ExpertConfig, MoEConfig,
                                          MoELayer, MoEWeightPack, QuantConfig, QuantFormat, RoutingConfig, SwiGLU,
                                          MoEActivationPack)
        from flashinfer.fused_moe.core import RoutingInputMode
    except ImportError as exc:
        print(f"no cuTile comparison ({exc})")
    else:
        cfg = MoEConfig(routing=RoutingConfig(num_experts=E, top_k=TOPK),
                        quant=QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4),
                        experts=ExpertConfig(intermediate_size=w2.shape[2] * 2), activation=SwiGLU(),
                        backend=BackendOptions(candidates=(CuTileNvfp4Config(),)),
                        execution=ExecutionConfig(tune_max_num_tokens=64))
        layer = MoELayer(cfg, device=dev)
        wp = MoEWeightPack()
        wp.prepare_for("cutile_nvfp4", {"w1": w13, "w1_scale": s13, "w1_global_scale": a13, "w2": w2, "w2_scale": s2,
                                        "w2_global_scale": a2, "input_inv_global": inv})
        for M in (1, 2, 4):
            x = (torch.randn(M, K, device=dev) * 0.6).bfloat16()
            ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]).int()
            ids[0, 3] = -1
            w = torch.softmax(torch.randn(M, TOPK, device=dev), -1)
            w = torch.where(ids < 0, torch.zeros_like(w), w)
            ct = layer(MoEActivationPack(x, None, ids, w, routing_input_mode=RoutingInputMode.PackedPrecomputed), wp)
            got = dec.run(x, ids, w, w13, s13, a13, w2, s2, a2, inv, True, SwiGLU())
            want = dec.ref(x, ids, w, w13, s13, a13, w2, s2, a2, inv, True, SwiGLU())
            n = lambda a, b: float((a.float() - b.float()).norm() / b.float().norm())
            print(f"M {M}: kernel vs cuTile {n(got, ct):.2e}, torch reference vs cuTile {n(want, ct):.2e}, "
                  f"kernel vs torch reference {n(got, want):.2e}")
    # time: one row, ten routes, as in decode
    x = (torch.randn(1, K, device=dev) * 0.6).bfloat16()
    ids = torch.randperm(E, device=dev)[:TOPK].view(1, -1).int()
    w = torch.softmax(torch.randn(1, TOPK, device=dev), -1)
    g = torch.cuda.CUDAGraph()
    dec.run(x, ids, w, w13, s13, a13, w2, s2, a2, inv, True, Act())
    with torch.cuda.graph(g):
        for _ in range(20):
            dec.run(x, ids, w, w13, s13, a13, w2, s2, a2, inv, True, Act())
    g.replay(); torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(50):
        g.replay()
    torch.cuda.synchronize()
    us = (time.perf_counter() - t) / 1000 * 1e6
    nbytes = TOPK * (w13[0].numel() + s13[0].numel() + w2[0].numel() + s2[0].numel())
    print(f"one row, {TOPK} routes: {us:.1f} us a layer (graph replay), {nbytes / 2**20:.2f} MiB read, "
          f"{nbytes / us / 1e3:.0f} GB/s")
    print("PASS" if worst < 2e-2 else "FAIL", f"(worst relative difference {worst:.2e})")


if __name__ == "__main__":
    main()
