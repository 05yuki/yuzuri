"""GPU-streamed prefill for KT CPU experts, NVFP4 (ModelOpt) edition, beside the MXFP4 streamer
(kt_stream_prefill.py picks it by the experts' format).

Above KT_GPU_STREAM_PREFILL tokens a MoE layer's CPU experts are not computed
on the CPU: each rank DMAs its TP part of the experts, G at a time, straight
out of kt-kernel's memfd arenas (KT_EXPERT_SHM=1) into a device slot,
repacks them into the FlashInfer CUTLASS NVFP4 layout on the device, and
runs the same cutlass_fused_moe the resident GPU experts use. Two slots
double-buffer DMA against compute; each (token count, group) is one CUDA
graph so the per-launch host cost under a saturated PCIe link stays off the
critical path (the DSV4.1 lesson, 09-13).

Per-layer fixed cost is the PCIe transfer of the whole expert bank:
Flash-Next is 512 experts x 3 x 2560 x 640 nibbles = 1.26 GB per layer,
split over two cards, ~24 ms at 26 GB/s each.

kt-kernel keeps NVFP4 block scales as fp32 (e4m3 x per-tensor scale_2, folded
via bf16 by its loader); the e4m3 bytes the GPU kernel wants are recovered
exactly by dividing by scale_2 and rounding to e4m3 (the bf16 error of
2^-8 is far inside e4m3's 2^-3 spacing). scale_2 per expert and projection
is read from the checkpoint at init.

Env:
  KT_GPU_STREAM_PREFILL=<tokens>   threshold (0 = off)
  KT_GPU_STREAM_GROUP=<G>          experts per slot (default 8)
  KT_EXPERT_SHM=1                  required (kt-kernel arenas)
  KT_GPU_STREAM_TIMING=1           per-layer timing lines
  KT_GPU_STREAM_SELFTEST=1         rank 0 checks arena bytes and recovered
                                   scales against the checkpoint at init
  KT_GPU_STREAM_DEBUG=<s>          periodic main-thread stack dump
"""
import json
import os
import time
import traceback

# KT_STREAM_GRAPH_KEEP: CUDA graphs kept for row counts other than the
# largest (a prompt's last chunk has its own)
GRAPH_KEEP = int(os.environ.get("KT_STREAM_GRAPH_KEEP", "8"))
from typing import Dict, List, Optional

import torch

from sglang.srt.runtime_context import get_parallel


def get_tensor_model_parallel_rank() -> int:
    return get_parallel().tp_rank


def get_tensor_model_parallel_world_size() -> int:
    return get_parallel().tp_size

from sglang.srt.layers.moe import kt_stream_threshold  # noqa: E402

# on or off, and the length before one is measured; the length in force is kt_stream_threshold.get()
STREAM_THRESHOLD = kt_stream_threshold.START
GROUP_SIZE = int(os.environ.get("KT_GPU_STREAM_GROUP", "8") or 8)
TIMING = os.environ.get("KT_GPU_STREAM_TIMING") == "1"
# Chunks shorter than the prefill chunk run padded through the full chunk's
# graphs.
FIXED_ROWS = os.environ.get("KT_GPU_STREAM_FIXED_ROWS", "0") == "1"


# rows per CUTLASS MoE call inside a group (0 = the chunk)
ROW_BLOCK = int(os.environ.get("KT_GPU_STREAM_ROW_BLOCK", "0") or 0)


def _block_rows(T: int) -> int:
    return min(T, ROW_BLOCK) if ROW_BLOCK > 0 else T


def _chunk_rows() -> int:
    try:
        from sglang.srt.runtime_context import get_schedule

        return int(get_schedule().chunked_prefill_size or 0)
    except Exception:
        return 0
# One-byte block scales from pinned host memory instead of kt-kernel's FP32
# scale arenas.
SCALE_BYTES = os.environ.get("KT_GPU_STREAM_SCALE_BYTES", "0") == "1"
SCALE_BYTES_VERIFY = os.environ.get("KT_GPU_STREAM_SCALE_BYTES_VERIFY", "0") == "1"


def _log(msg: str):
    print(f"[kt-stream] {msg}", flush=True)


def _swizzle(s: torch.Tensor) -> torch.Tensor:
    """FlashInfer's NVFP4 block-scale layout (sglang swizzle_blockscale), on
    the device, no padding: [n, M, Kc] -> [n, M, Kc] with M % 128 == 0 and
    Kc % 4 == 0."""
    n, M, Kc = s.shape
    return (
        s.reshape(n, M // 128, 4, 32, Kc // 4, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .contiguous()
        .view(n, M, Kc)
    )


def _to_e4m3(d: torch.Tensor, scale_2: torch.Tensor) -> torch.Tensor:
    """fp32 folded scales [n, M, Kc] / scale_2 [n] -> e4m3 bytes."""
    return (d / scale_2.view(-1, 1, 1)).to(torch.float8_e4m3fn).view(torch.uint8)


def _physical_to_logical(layer_idx: int, n: int) -> List[int]:
    """The checkpoint's expert id of each of a layer's n physical experts (the arenas'
    order): the hot-expert placement's map, the identity without one."""
    from sglang.srt.eplb.expert_location_dispatch import get_global_expert_location_metadata

    meta = get_global_expert_location_metadata()
    if meta is None:
        return list(range(n))
    return [int(v) for v in meta.physical_to_logical_map_cpu[layer_idx][:n].tolist()]


def _align64(n: int) -> int:
    return (n + 63) & ~63


# KT_SCALE_ARENA_CODES=1 (kt-kernel): a scale arena block is a one-byte code per 16 elements
# (64-aligned), then the matrix's 256-entry fp32 table. The streamer copies those blocks and expands them into the
# fp32 layout the rest of it reads (raw_s), on the device.
CODES = os.environ.get("KT_SCALE_ARENA_CODES", "0") == "1"


def _codes_stride(count: int) -> int:
    """Bytes of one matrix's arena block under CODES: count codes, 64-aligned, then the 1 KiB table."""
    return _align64(count) + 1024


def _host_scales(arena: torch.Tensor, e: int, stride: int, rows: int, cols: int) -> torch.Tensor:
    """Expert e's fp32 scales [1, rows, cols] from a host arena (fp32 blocks, or codes and table under CODES)."""
    base = e * stride
    if not CODES:
        return arena[base:base + 4 * rows * cols].view(torch.float32).view(1, rows, cols)
    lut_at = base + _align64(rows * cols)
    lut = arena[lut_at:lut_at + 1024].view(torch.float32)
    return lut[arena[base:base + rows * cols].long()].view(1, rows, cols)


import triton
import triton.language as tl


@triton.jit
def _nvfp4_dequant_kernel(
    packed_ptr, scale_ptr, out_ptr, n_elems, cols, packed_stride, scale_stride, out_stride, out_rs,
    BLOCK: tl.constexpr,
):
    """One expert per program_id(1). packed: e2m1 pairs, low nibble first, cols // 2 bytes a row; scale: fp32 per
    16 elements (kt-kernel's arenas: the e4m3 block scale times the global scale); out: bf16, out_rs a row."""
    e = tl.program_id(1)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n_elems
    row = offs // cols
    col = offs % cols
    byte = tl.load(packed_ptr + e * packed_stride + row * (cols // 2) + col // 2, mask=m, other=0).to(tl.int32)
    nib = (byte >> ((col % 2) * 4)) & 0xF
    sign = tl.where((nib & 8) != 0, -1.0, 1.0)
    ex = (nib >> 1) & 3
    man = (nib & 1).to(tl.float32)
    mag = tl.where(ex == 0, man * 0.5, (1.0 + man * 0.5) * tl.exp2((ex - 1).to(tl.float32)))
    sc = tl.load(scale_ptr + e * scale_stride + row * (cols // 16) + col // 16, mask=m, other=0.0)
    tl.store(out_ptr + e * out_stride + row * out_rs + col, (sign * mag * sc).to(tl.bfloat16), mask=m)


@triton.jit
def _codes_expand_kernel(codes_ptr, out_ptr, count, code_stride, lut_off, out_stride, BLOCK: tl.constexpr):
    """One expert per program_id(1): out[i] = table[code[i]], the table lut_off bytes into the expert's block."""
    e = tl.program_id(1)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < count
    blk = codes_ptr + e * code_stride
    c = tl.load(blk + offs, mask=m, other=0).to(tl.int32)
    lut = (blk + lut_off).to(tl.pointer_type(tl.float32))
    tl.store(out_ptr + e * out_stride + offs, tl.load(lut + c, mask=m, other=0.0), mask=m)


def _expand_codes(raw_c, raw_s, n: int, count: int):
    """raw_c [3, G, code stride] (CODES arena blocks) -> raw_s [3, G, fp32 stride bytes], experts [0, n)."""
    BLOCK = 4096
    for pi in range(3):
        _codes_expand_kernel[(triton.cdiv(count, BLOCK), n)](
            raw_c[pi], raw_s[pi].view(torch.float32), count, raw_c.shape[2], _align64(count), raw_s.shape[2] // 4,
            BLOCK=BLOCK)


def _nvfp4_dequant(packed, packed_stride: int, scales_f32, scale_stride: int, out, n: int, rows: int, cols: int):
    """packed / scales: the slot's raw tensors (expert-major, strides in bytes / floats); out [n, rows, cols] bf16."""
    assert out.stride(2) == 1, "dequant target columns must be contiguous"
    BLOCK = 4096
    n_elems = rows * cols
    _nvfp4_dequant_kernel[(triton.cdiv(n_elems, BLOCK), n)](
        packed, scales_f32, out, n_elems, cols, packed_stride, scale_stride, out.stride(0), out.stride(1),
        BLOCK=BLOCK)


class _DeviceSlot:
    def __init__(self, G: int, N: int, K: int, device, raw: bool = True):
        from sglang.srt.layers.moe import kt_lend

        u8 = torch.uint8

        def empty(shape):
            # resident only during a prefill
            return kt_lend.scratch_empty(shape, u8, device)

        self.w13_weight = empty((G, 2 * N, K // 2))
        self.w2_weight = empty((G, K, N // 2))
        self.w13_scale = empty((G, 2 * N, K // 16))
        self.w2_scale = empty((G, K, N // 16))
        self.dma_done = torch.cuda.Event()
        self.compute_done = torch.cuda.Event()
        self.compute_used = False
        if not raw:  # the cuTile path: the arenas' bytes land in the four tensors above as they are
            self.raw = self.raw_s = self.raw_c = None
            return
        # The group's gate/up/down blocks exactly as kt-kernel stores them
        # (packed weights; fp32 scales in separate arenas), G consecutive
        # experts per arena, 64-aligned stride (kt-kernel's arena stride).
        self.raw = empty((3, G, _align64(N * K // 2)))
        self.raw_s = empty((3, G, _align64(N * (K // 16) * 4)))
        # CODES: the arena blocks as they are (gate/up and down hold the same count, N * K / 16)
        self.raw_c = empty((3, G, _codes_stride(N * (K // 16)))) if CODES else None


# Device buffers made before SGLang sizes its KV pool (preallocate() from
# ModelOptNvFp4FusedMoEMethod.create_moe_runner), so the streamer's ~1 GB
# is not taken from the headroom the pool leaves: with 32 hot experts the
# first 2048-token chunk OOMed when it was the first request (09-14).
_prealloc = {}
# device index -> (chunk rows, graph inputs sized for it, in the lend scratch)
_prealloc_gin = {}
_CUDART_2D = None  # cudaMemcpy2DAsync, loaded at the first cuTile group stage
# Pad short chunks to the next multiple of this many rows, not to the chunk.
ROW_STEP = int(os.environ.get("KT_GPU_STREAM_ROW_STEP", "2048") or 0)


def _layer_device(layer) -> torch.device:
    """The card a MoE layer runs on: its resident experts' weights, or with none on the card (--hot 0, the layer then
    holds no w13_weight) the current device."""
    w = getattr(layer, "w13_weight", None)
    return w.device if w is not None else torch.device("cuda", torch.cuda.current_device())


def preallocate(layer):
    if STREAM_THRESHOLD <= 0 or not hasattr(layer, "intermediate_size_per_partition"):
        return
    device = _layer_device(layer)
    key = device.index
    if key in _prealloc:
        return
    N = layer.intermediate_size_per_partition
    K = layer.hidden_size
    slots = [_DeviceSlot(GROUP_SIZE, N, K, device) for _ in range(2)]
    _prealloc[key] = slots
    # the graph inputs for the one chunk length the streamer runs (shorter
    # steps use views of their head), made now so no long-lived buffer is
    # allocated inside a prefill
    if FIXED_ROWS and _chunk_rows() > 0:
        from sglang.srt.layers.moe import kt_lend

        T, topk, f32 = _chunk_rows(), layer.top_k, torch.float32

        def buf(shape, dtype):
            return kt_lend.scratch_empty(shape, dtype, device, per_chunk=len(shape) > 0 and shape[0] == T)

        _prealloc_gin[key] = (T, {
            "x": buf((T, K), torch.bfloat16), "ids": buf((T, topk), torch.int32),
            "w": buf((T, topk), f32), "acc": buf((T, K), torch.bfloat16),
            "out": buf((T, K), torch.bfloat16), "a1": buf((), f32), "a2": buf((), f32),
            "s2g": buf((GROUP_SIZE,), f32), "s2d": buf((GROUP_SIZE,), f32),
            "g1": buf((GROUP_SIZE,), f32), "g2": buf((GROUP_SIZE,), f32),
        })
    mb = sum(t.numel() for sl in slots for t in (sl.w13_weight, sl.w2_weight, sl.w13_scale, sl.w2_scale, sl.raw, sl.raw_s)) / 1e6
    _log(f"preallocated 2 device slots ({mb:.0f} MB) on {device} before the KV pool")


class StreamUnsupported(RuntimeError):
    """The experts' checkpoint lacks what the W4A4 path reads (KT_NVFP4_W4A16 streams it in W16)."""


_GLOBAL_SUFFIXES = (".weight_scale_2", ".weight_global_scale")  # ModelOpt: scale_2; compressed-tensors: its reciprocal
_ROLE_WORDS = {"gate": ("gate", "w1"), "down": ("down", "w2")}
_UP_WORDS = ("up", "w3")  # read too when present (the cuTile path keeps gate's and up's globals apart)


def _check_modelopt_scales(path: str) -> None:
    """An NVFP4 checkpoint keeps a per-tensor global scale per expert projection: ModelOpt's `weight_scale_2`, or
    compressed-tensors' `weight_global_scale` (1 / scale_2). Without one the alphas cannot be made."""
    idx = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
    if not any(".experts." in k and k.endswith(_GLOBAL_SUFFIXES) for k in idx):
        raise StreamUnsupported(f"no per-expert weight_scale_2 or weight_global_scale under {path}")


def _scale_2_keys(idx):
    """The checkpoint's per-expert global scale keys as a template, for any expert path and projection names:
    (template with {L} {E}, {"gate": name, "down": name}, reciprocal?)."""
    import re

    pat = re.compile(r"^(.*\.layers\.)(\d+)(\..*\.experts\.)0\.([^.]+)(\.weight_scale_2|\.weight_global_scale)$")
    for k in idx:
        m = pat.match(k)
        if m is None:
            continue
        head, layer, mid, _, suffix = m.groups()
        names = {mm.group(4) for kk in idx if (mm := pat.match(kk)) and mm.group(2) == layer}
        roles = {}
        for role, words in _ROLE_WORDS.items():
            for nm in sorted(names):
                low = nm.lower()
                if any(w in low.split("_") or low.startswith(w) or low.endswith(w) for w in words):
                    roles.setdefault(role, nm)
        if set(roles) == set(_ROLE_WORDS):
            for nm in sorted(names):
                low = nm.lower()
                if any(w in low.split("_") or low.startswith(w) or low.endswith(w) for w in _UP_WORDS):
                    roles.setdefault("up", nm)
            return head + "{L}" + mid + "{E}.{P}" + suffix, roles, suffix == ".weight_global_scale"
    raise StreamUnsupported("no per-expert global scale keys with gate and down projections")


def _input_quant(layer, which: str):
    """The activation quant scale of a layer's w13 / w2 for the CUTLASS W4A4 MoE: the resident method's
    `*_input_scale_quant` (ModelOpt), else compressed-tensors' `*_input_global_scale` (its min over experts, as
    SGLang's own compressed-tensors scheme takes it), else neutral."""
    q = getattr(layer, f"{which}_input_scale_quant", None)
    if q is not None:
        return q.reshape(-1)[0]
    g = getattr(layer, f"{which}_input_global_scale", None)
    if g is not None:
        return g.min()
    return torch.ones((), device=_layer_device(layer))


class KTStreamPrefill:
    """Process-wide (per rank) streaming state, shared by every MoE layer."""

    def __init__(self, method, layer):
        from sglang.srt.layers.moe import kt_cutile_nvfp4

        # kt_cutile_nvfp4: the resident experts run from the checkpoint's bytes on FlashInfer's cuTile MoE, and so do
        # the streamed groups: the arenas' weights and one-byte scale codes (the checkpoint's e4m3 bytes) land in the
        # slots as they are. Without the codes the arenas hold folded fp32 scales, and the CUTLASS path takes them.
        self.cutile = isinstance(getattr(method, "gpu_method", None), kt_cutile_nvfp4.CutileNvfp4Method)
        if self.cutile and not CODES:
            _log("cuTile groups need KT_SCALE_ARENA_CODES=1 (the arenas' e4m3 bytes); CUTLASS takes the groups")
            self.cutile = False
        # the checkpoint's precision (kt_cutile_nvfp4.checkpoint_activations, decided at the wrap): W4A16 when it
        # does not quantize activations, else W4A4 with its static activation globals per layer (g1 for GEMM1's
        # input, g2 for GEMM2's); a g2 other than 1 needs an activation linear in up, else CUTLASS takes the groups
        self._a4 = None
        self._g12 = {}  # layer index -> (g1, g2)
        self._inv_g1 = None  # [1]: the current layer's 1 / g1, read by the group graphs (some layer's g1 is not 1)
        self._needs_inv_g1 = False
        if self.cutile:
            acts = method.gpu_method.acts
            self._a4 = acts != "none"
            if self._a4:
                self._g12 = kt_cutile_nvfp4.checkpoint_layer_globals(method.kt_config.weight_path)
            act = method.gpu_method.activation
            if any(g2 != 1.0 for _, g2 in self._g12.values()) and not kt_cutile_nvfp4._linear_in_up(act):
                _log(f"GEMM2 activation globals other than 1 with an activation that clamps or shifts up ({act}): "
                     "CUTLASS takes the groups")
                self.cutile = False
            self._needs_inv_g1 = any(g1 != 1.0 for g1, _ in self._g12.values())
        if self.cutile:
            # the globals per expert are the checkpoint's: without them no group can run (no other arithmetic)
            _check_modelopt_scales(method.kt_config.weight_path)
            self.w16 = False
        else:
            # without ModelOpt's scales the checkpoint is W4A16 and streams in W16
            try:
                _check_modelopt_scales(method.kt_config.weight_path)
                self.w16 = os.environ.get("KT_NVFP4_STREAM_W16", "0") == "1"
                if not self.w16 and not any(hasattr(layer, f"w13_{a}") for a in ("input_scale_quant", "input_global_scale")):
                    # with no resident experts (--hot 0) the layer holds no activation scale, and W4A4 at a made-up
                    # one is not the checkpoint's arithmetic
                    raise StreamUnsupported("the layer holds no activation scale (no resident experts)")
            except StreamUnsupported as e:
                self.w16 = True
                _log(f"W16: {e}")
        from flashinfer.fused_moe import cutlass_fused_moe
        from flashinfer.fused_moe.core import ActivationType
        from sglang.srt.environ import envs
        from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import _activation_type

        self.rank = get_tensor_model_parallel_rank()
        self.world = get_tensor_model_parallel_world_size()
        self.device = _layer_device(layer)
        self.N = layer.intermediate_size_per_partition
        self.K = layer.hidden_size
        self.G = GROUP_SIZE
        if self.cutile:
            pass  # the slots hold the checkpoint's layout; the cuTile MoE takes any N, K a multiple of 64
        elif not self.w16:
            # the slots hold FlashInfer's CUTLASS NVFP4 layout whatever the resident method keeps (one checkpoint's
            # experts are Marlin-repacked on SM120): only its unpadded block-scale shapes are required here
            if (2 * self.N) % 128 or (self.K // 16) % 4 or self.K % 128 or (self.N // 16) % 4:
                raise RuntimeError(
                    f"KT_GPU_STREAM_PREFILL: N={self.N} K={self.K} would need padded block scales")
        else:
            # the bf16 group buffers take four times a packed slot's bytes; the
            # group shrinks by that factor so they stay one packed slot's size
            self.G = max(1, self.G // 4)
        self.slots = _prealloc.get(self.device.index) or [
            _DeviceSlot(self.G, self.N, self.K, self.device, raw=not self.cutile) for _ in range(2)]
        self._tp_size, self._tp_rank = layer.moe_tp_size, layer.moe_tp_rank
        self._ep_size, self._ep_rank = layer.moe_ep_size, layer.moe_ep_rank
        self.transfer_stream = torch.cuda.Stream(device=self.device)
        self._cutlass_fused_moe = cutlass_fused_moe
        self._activation_type = _activation_type(layer.moe_runner_config)
        self.swiglu_limit = self.swiglu_alpha = self.swiglu_beta = None
        # the activation SGLang's CUTLASS mapping gives the layer, whatever it is
        self._use_fused_finalize = envs.SGLANG_FLASHINFER_MOE_FUSED_FINALIZE.get()
        self._workspace = None
        self._workspace_tokens = 0
        self._graphs = {}
        self._graph_use = {}  # row count -> None, least recently used first
        self._gin = {}
        self._capture_stream = torch.cuda.Stream(device=self.device)
        self._graph_pool = torch.cuda.graph_pool_handle()
        self._arenas = {}
        self._registered = set()
        self._map_expert_arenas(method)
        self._s2 = {}      # layer_idx -> (scale_2 gate/up [E], scale_2 down [E]) on device
        self._s2u = {}     # layer_idx -> scale_2 up [E] on device, when the checkpoint names up (cuTile path)
        self._alphas = {}  # layer_idx -> (g1 [E], g2 [E]) on device, from the layer's input scales
        if not self.w16:
            self._load_scale_2(method)
        self._sb = {}      # layer_idx -> (w13 scale bytes [E, 2N, K/16], w2 scale bytes [E, K, N/16]), pinned host
        if self.w16:
            self._w16_setup(layer)
        self._swiglu_oai(getattr(layer, "moe_runner_config", None))
        if self.cutile:
            self._ct_act = method.gpu_method.activation
        self.calls = 0
        _log(
            f"rank {self.rank}: streamed prefill above {STREAM_THRESHOLD} tokens, G={self.G}, "
            f"N={self.N} K={self.K}, zero-copy from kt-kernel's expert arenas, "
            + (f"cuTile {'W4A4' if self._a4 else 'W4A16'}"
               f"{' with static activation globals for ' + str(len(self._g12)) + ' layers' if self._g12 else ''}"
               f" on the checkpoint's bytes, " if self.cutile else "")
            + f"2 device slots x {(self.slots[0].w13_weight.numel() + self.slots[0].w2_weight.numel())/1e6:.0f} MB")
        if self.rank == 0 and os.environ.get("KT_GPU_STREAM_SELFTEST") == "1":
            self._selftest(method)

    def _swiglu_oai(self, cfg):
        """A layer with gemm1_alpha (swigluoai: gate * sigmoid(alpha * gate) * (up + 1), clamped): the
        per-expert alpha, beta and limit SGLang's own CUTLASS methods pass (materialize_swiglu_params_for_cutlass).
        Without them the streamed experts ran SiLU."""
        if cfg is None or getattr(cfg, "gemm1_alpha", None) is None:
            return
        from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import materialize_swiglu_params_for_cutlass

        alpha, beta, limit = materialize_swiglu_params_for_cutlass(cfg, self.G, self.device)
        if alpha is None:
            raise RuntimeError("streamed experts: gemm1_alpha without a clamp limit, "
                               "no CUTLASS swiglu parameters for it")
        self.swiglu_alpha, self.swiglu_beta, self.swiglu_limit = alpha, beta, limit
        _log(f"rank {self.rank}: streamed experts swiglu alpha={float(alpha[0])} beta={float(beta[0])} "
             f"limit={float(limit[0])}")

    def _w16_setup(self, layer):
        """The bf16 group buffers (in the lend scratch) and the clamp the resident runner applies
        (moe_runner_config.swiglu_limit: gate <= limit, |up| <= limit), for the bf16 CUTLASS MoE."""
        from flashinfer.fused_moe.core import ActivationType
        from sglang.srt.layers.moe import kt_lend

        self.w13_bf16 = self.w2_bf16 = None  # made on the first streamed call (_w16_buffers)
        cfg = getattr(layer, "moe_runner_config", None)
        lim = getattr(cfg, "swiglu_limit", None) if os.environ.get("KT_SWIGLU_CLAMP", "1") != "0" else None
        self.swiglu_limit = (None if not lim else
                             torch.full((self.G,), float(lim), dtype=torch.float32, device=self.device))
        _log(f"rank {self.rank}: W16, groups of {self.G}, swiglu_limit={lim}")

    def _w16_buffers(self):
        """The bf16 group buffers, in the lend scratch: resident while a prefill streams, gone in decode."""
        from sglang.srt.layers.moe import kt_lend

        with kt_lend.scratch_region():
            self.w13_bf16 = torch.empty((self.G, 2 * self.N, self.K), dtype=torch.bfloat16, device=self.device)
            self.w2_bf16 = torch.empty((self.G, self.K, self.N), dtype=torch.bfloat16, device=self.device)
        _log(f"rank {self.rank}: W16 bf16 group buffers "
             f"{(self.w13_bf16.numel() + self.w2_bf16.numel()) * 2 / 1e6:.0f} MB")

    def _group_body_w16(self, layer, gi: int, base: int, n: int, g):
        """Dequantize the group to bf16 and run the bf16 CUTLASS MoE ([up; gate] rows, as the W4A4 body)."""
        slot = self.slots[gi % 2]
        N, K = self.N, self.K
        x, topk_weights, ids32, acc, out = g["x"], g["w"], g["ids"], g["acc"], g["out"]
        ws, ss = slot.raw.shape[2], slot.raw_s.shape[2] // 4
        if CODES:
            _expand_codes(slot.raw_c, slot.raw_s, n, N * (K // 16))
        rs = slot.raw_s.view(torch.float32)
        _nvfp4_dequant(slot.raw[1], ws, rs[1], ss, self.w13_bf16[:n, :N], n, N, K)
        _nvfp4_dequant(slot.raw[0], ws, rs[0], ss, self.w13_bf16[:n, N:], n, N, K)
        _nvfp4_dequant(slot.raw[2], ws, rs[2], ss, self.w2_bf16[:n], n, K, N)
        local = torch.where((ids32 >= base) & (ids32 < base + n), ids32 - base, torch.full_like(ids32, -1))
        out.zero_()
        T = x.shape[0]
        R = _block_rows(T)
        for r0 in range(0, T, R):
            r1 = min(T, r0 + R)
            self._cutlass_fused_moe(
                input=x[r0:r1],
                token_selected_experts=local[r0:r1],
                token_final_scales=topk_weights[r0:r1],
                fc1_expert_weights=self.w13_bf16[:n],
                fc2_expert_weights=self.w2_bf16[:n],
                output_dtype=torch.bfloat16,
                quant_scales=None,
                input_sf=None,
                swiglu_alpha=None if self.swiglu_alpha is None else self.swiglu_alpha[:n],
                swiglu_beta=None if self.swiglu_beta is None else self.swiglu_beta[:n],
                swiglu_limit=None if self.swiglu_limit is None else self.swiglu_limit[:n],
                tp_size=self._tp_size,
                tp_rank=self._tp_rank,
                ep_size=self._ep_size,
                ep_rank=self._ep_rank,
                activation_type=self._activation_type,
                tune_max_num_tokens=self._tune_tokens,
                output=out[r0:r1],
                use_fused_finalize=self._use_fused_finalize,
                workspace_buffer=self._workspace,
            )
        acc.add_(out)

    # -- checkpoint scale_2 ---------------------------------------------------

    def _load_scale_2(self, method):
        """Per expert and projection scale_2 for every KT layer, from the
        safetensors index; g1/g2 alphas = the layer's input scale x scale_2."""
        from safetensors import safe_open

        path = method.kt_config.weight_path
        idx = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
        # One expert's keys tell us the pattern: ModelOpt's scale_2 or compressed-tensors' reciprocal
        tpl, roles, reciprocal = _scale_2_keys(idx)
        files = {}
        t0 = time.time()
        # Every rank (the wrapper and KT_LAYERS exist on rank 0 only): the
        # layers and expert count come from the mapped arenas.
        # the cuTile path: up's globals too
        up = self.cutile and "up" in roles
        for layer_idx in sorted(self._arenas):
            E = self._num_experts(layer_idx)
            g = torch.empty(E, dtype=torch.float32)
            d = torch.empty(E, dtype=torch.float32)
            u = torch.empty(E, dtype=torch.float32) if up else None
            p2l = _physical_to_logical(layer_idx, E)
            for e in range(E):
                for role, dst in (("gate", g), ("down", d)) + ((("up", u),) if up else ()):
                    k = tpl.format(L=layer_idx, E=p2l[e], P=roles[role])
                    fn = idx[k]
                    f = files.get(fn)
                    if f is None:
                        f = files[fn] = safe_open(os.path.join(path, fn), "pt")
                    v = f.get_tensor(k).float().item()
                    dst[e] = 1.0 / v if reciprocal else v
            self._s2[layer_idx] = (g.to(self.device), d.to(self.device))
            if up:
                self._s2u[layer_idx] = u.to(self.device)
        _log(f"rank {self.rank}: scale_2 for {len(self._s2)} layers read in {time.time()-t0:.1f} s")

    def _num_experts(self, layer_idx: int) -> int:
        t, stride, _ = self._arenas[layer_idx][0]
        return t.numel() // stride

    def _alphas_for(self, layer, layer_idx: int):
        """g1/g2 = input scale x scale_2 per expert, as the resident method's
        g1_alphas/g2_alphas; the layer keeps 1/input_scale."""
        a = self._alphas.get(layer_idx)
        if a is None:
            s2g, s2d = self._s2[layer_idx]
            g1 = (s2g / _input_quant(layer, "w13")).to(torch.float32)
            g2 = (s2d / _input_quant(layer, "w2")).to(torch.float32)
            a = self._alphas[layer_idx] = (g1, g2)
        return a

    # -- zero-copy: map kt-kernel's expert arenas -----------------------------

    def _map_expert_arenas(self, method):
        """Rank 0 reads every KT layer's (fd, size, stride) for the part that
        feeds each rank and passes the other ranks their fds over a unix
        socket; every rank mmaps its part."""
        import mmap
        import socket

        from sglang.srt.layers.moe.kt_ep_methods import KT_LAYERS

        sock_path = f"/tmp/kt_stream_{os.getppid()}_r{self.rank}.sock"
        infos = {}
        if self.rank == 0:
            per_rank = {r: {} for r in range(self.world)}
            for layer_idx, m in sorted(KT_LAYERS.items()):
                parts = m.wrapper.moe.expert_arena_infos()
                if not parts or len(parts[0]) != 6:
                    raise RuntimeError(
                        "KT_GPU_STREAM_PREFILL needs kt-kernel built with expert arenas and KT_EXPERT_SHM=1 "
                        f"(got {0 if not parts else len(parts[0])} arenas per part, want 6)")
                cpu_tp = len(parts)
                for r in range(self.world):
                    part = r * cpu_tp // self.world if cpu_tp >= self.world else r // (self.world // cpu_tp)
                    per_rank[r][layer_idx] = parts[part]
            infos = per_rank[0]
            for r in range(1, self.world):
                path = f"/tmp/kt_stream_{os.getppid()}_r{r}.sock"
                t0 = time.time()
                while True:
                    try:
                        c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                        c.connect(path)
                        break
                    except (FileNotFoundError, ConnectionRefusedError):
                        c.close()
                        if time.time() - t0 > 600:
                            raise
                        time.sleep(0.05)
                layers = sorted(per_rank[r])
                fds = [fd for l in layers for (fd, _, _) in per_rank[r][l]]
                meta = repr([(l, [(sz, st) for (_, sz, st) in per_rank[r][l]]) for l in layers]).encode()
                c.sendall(len(meta).to_bytes(8, "little"))
                c.sendall(meta)
                for i in range(0, len(fds), 200):
                    socket.send_fds(c, [b"x"], fds[i:i + 200])
                c.recv(1)
                c.close()
        else:
            try:
                os.unlink(sock_path)
            except FileNotFoundError:
                pass
            srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            srv.bind(sock_path)
            srv.listen(1)
            c, _ = srv.accept()
            n = int.from_bytes(c.recv(8), "little")
            buf = b""
            while len(buf) < n:
                buf += c.recv(n - len(buf))
            meta = eval(buf.decode())  # our own repr of ints
            nfds = sum(len(v) for _, v in meta)
            fds = []
            while len(fds) < nfds:
                _, got, _, _ = socket.recv_fds(c, 1, 200)
                fds.extend(got)
            c.sendall(b"k")
            c.close()
            srv.close()
            os.unlink(sock_path)
            i = 0
            for layer_idx, sizes in meta:
                infos[layer_idx] = [(fds[i + j], sizes[j][0], sizes[j][1]) for j in range(6)]
                i += 6
        for layer_idx, arenas in sorted(infos.items()):
            tensors = []
            for (fd, size, stride) in arenas:
                mm = mmap.mmap(fd, size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)
                t = torch.frombuffer(mm, dtype=torch.uint8)
                tensors.append((t, stride, mm))
            self._arenas[layer_idx] = tensors
        _log(f"rank {self.rank}: mapped {len(infos)} layers of expert arenas")

    def _register_arenas(self, layer_idx: int):
        """cudaHostRegister on a layer's first streamed group, so only layers
        that stream get pinned (0.2 s/layer with shmem THP)."""
        if layer_idx in self._registered:
            return
        t0 = time.time()
        total = 0
        for (t, _, _) in self._arenas[layer_idx]:
            rc = torch.cuda.cudart().cudaHostRegister(t.data_ptr(), t.numel(), 0)
            if int(rc) != 0:
                raise RuntimeError(f"cudaHostRegister(expert arena, layer {layer_idx}) failed: {int(rc)}")
            total += t.numel()
        self._registered.add(layer_idx)
        if TIMING:
            _log(f"rank {self.rank}: registered layer {layer_idx} arenas, {total/1e9:.1f} GB in {time.time()-t0:.1f} s")

    def _selftest(self, method):
        """Rank 0, layer min: arena bytes of this rank's part against the
        checkpoint, and the recovered e4m3 scales against the checkpoint's."""
        from safetensors import safe_open
        from sglang.srt.layers.moe.kt_ep_methods import KT_LAYERS

        layer_idx = min(KT_LAYERS)
        m = KT_LAYERS[layer_idx]
        path = method.kt_config.weight_path
        idx = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
        sample = next(k for k in idx if ".mlp.experts.0.gate_proj.weight" in k and k.endswith(".weight"))
        tpl = sample.split(".layers.")[0] + ".layers.{L}.mlp.experts.{E}.{P}_proj.{T}"
        N, K = self.N, self.K
        r = self.rank
        arenas = self._arenas[layer_idx]
        wst, sst = arenas[0][1], arenas[3][1]
        s2g, s2d = self._s2[layer_idx]
        p2l = _physical_to_logical(layer_idx, m.global_num_experts)

        def get(e, proj, t):
            k = tpl.format(L=layer_idx, E=p2l[e], P=proj, T=t)
            with safe_open(os.path.join(path, idx[k]), "pt") as f:
                return f.get_tensor(k)

        for e in (m.num_gpu_experts, m.num_gpu_experts + 93, m.global_num_experts - 1):
            ck_gate = get(e, "gate", "weight")[r * N:(r + 1) * N]          # [N, K/2]
            ck_up = get(e, "up", "weight")[r * N:(r + 1) * N]
            ck_down = get(e, "down", "weight")[:, r * (N // 2):(r + 1) * (N // 2)].contiguous()  # [K, N/2]
            ck_gs = get(e, "gate", "weight_scale")[r * N:(r + 1) * N].view(torch.uint8)
            ck_us = get(e, "up", "weight_scale")[r * N:(r + 1) * N].view(torch.uint8)
            ck_ds = get(e, "down", "weight_scale")[:, r * (N // 16):(r + 1) * (N // 16)].contiguous().view(torch.uint8)
            a_gate = arenas[0][0][e * wst:e * wst + N * K // 2].view(N, K // 2)
            a_up = arenas[1][0][e * wst:e * wst + N * K // 2].view(N, K // 2)
            a_down = arenas[2][0][e * wst:e * wst + K * N // 2].view(K, N // 2)
            d_gs = _host_scales(arenas[3][0], e, sst, N, K // 16)
            d_us = _host_scales(arenas[4][0], e, sst, N, K // 16)
            d_ds = _host_scales(arenas[5][0], e, sst, K, N // 16)
            r_gs = _to_e4m3(d_gs, s2g[e:e + 1].cpu())[0]
            r_us = _to_e4m3(d_us, s2g[e:e + 1].cpu())[0]
            r_ds = _to_e4m3(d_ds, s2d[e:e + 1].cpu())[0]

            def mism(a, b):
                return int((a != b).sum())

            _log(f"selftest layer {layer_idx} expert {e} part {r}: weight mismatches gate={mism(a_gate, ck_gate)}/{a_gate.numel()} "
                 f"up={mism(a_up, ck_up)} down={mism(a_down, ck_down)}/{a_down.numel()}; recovered e4m3 scale mismatches "
                 f"gate={mism(r_gs, ck_gs)}/{r_gs.numel()} up={mism(r_us, ck_us)} down={mism(r_ds, ck_ds)}/{r_ds.numel()}; "
                 f"gate_w[:6] arena={a_gate.flatten()[:6].tolist()} ckpt={ck_gate.flatten()[:6].tolist()}")

    # -- staging ---------------------------------------------------------------

    def _stage(self, layer_idx: int, gi: int, experts, stats):
        """Six contiguous H2D copies: the group's consecutive expert blocks of
        the gate, up and down weight and scale arenas into the slot's raw
        landing, on the transfer stream."""
        self._register_arenas(layer_idx)
        if self.cutile:
            return self._stage_cutile(layer_idx, gi, experts, stats)
        slot = self.slots[gi % 2]
        n = len(experts)
        base = experts[0]
        arenas = self._arenas[layer_idx]
        wstride = slot.raw.shape[2]
        land = self._scale_landing(slot)
        sstride = land.shape[2]
        assert all(a[1] == wstride for a in arenas[:3]) and all(a[1] == sstride for a in arenas[3:]),             f"arena strides {[a[1] for a in arenas]} vs slot {wstride}/{sstride}"
        t0 = time.perf_counter()
        with torch.cuda.stream(self.transfer_stream):
            if slot.compute_used:
                self.transfer_stream.wait_event(slot.compute_done)
            if TIMING:
                ev0 = torch.cuda.Event(enable_timing=True)
                ev0.record(self.transfer_stream)
            for pi in range(3):
                slot.raw[pi, :n].view(-1).copy_(arenas[pi][0][base * wstride:(base + n) * wstride], non_blocking=True)
                if not SCALE_BYTES or self.w16:
                    land[pi, :n].view(-1).copy_(
                        arenas[3 + pi][0][base * sstride:(base + n) * sstride], non_blocking=True)
            if SCALE_BYTES and not self.w16:
                h13, h2 = self._sb[layer_idx]
                slot.w13_scale[:n].copy_(h13[base:base + n], non_blocking=True)
                slot.w2_scale[:n].copy_(h2[base:base + n], non_blocking=True)
            slot.dma_done.record(self.transfer_stream)
            if TIMING:
                ev1 = torch.cuda.Event(enable_timing=True)
                ev1.record(self.transfer_stream)
                stats.setdefault("dma_ev", []).append((ev0, ev1))
        stats["dma_enq"] += time.perf_counter() - t0

    @staticmethod
    def _memcpy2d(dst: int, dpitch: int, src: int, spitch: int, width: int, height: int, stream) -> None:
        """One strided host-to-device copy (cudaMemcpy2DAsync from the libcudart torch loaded): height rows of width
        bytes, src rows spitch apart, dst rows dpitch apart."""
        global _CUDART_2D
        if _CUDART_2D is None:
            import ctypes

            libs = sorted({line.split()[-1] for line in open("/proc/self/maps") if "libcudart" in line})
            f = ctypes.CDLL(libs[0]).cudaMemcpy2DAsync
            f.restype = ctypes.c_int
            f.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
                          ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
            _CUDART_2D = f
        rc = _CUDART_2D(dst, dpitch, src, spitch, width, height, 1, stream.cuda_stream)  # 1: host to device
        if rc != 0:
            raise RuntimeError(f"cudaMemcpy2DAsync failed: error {rc}")

    def _stage_cutile(self, layer_idx: int, gi: int, experts, stats):
        """The group's experts from the arenas into the slot as they are, on the transfer stream: per expert the gate
        and up weights side by side (w13 rows: gate, then up), the down weights, and the three matrices' one-byte
        scale codes (the checkpoint's e4m3 bytes, row-major; each arena block's 1 KiB table stays behind). Six
        strided copies, one per arena: copying expert by expert (six each) took 58% more transfer time than the
        CUTLASS path's six contiguous copies (Flash-Next NVFP4, 64 experts a group, 11-08)."""
        slot = self.slots[gi % 2]
        arenas = self._arenas[layer_idx]
        N, K = self.N, self.K
        wb, cb = N * K // 2, N * (K // 16)  # bytes of one matrix's weights, codes (gate, up and down alike)
        ws, cs = arenas[0][1], arenas[3][1]
        assert all(a[1] == ws for a in arenas[:3]) and all(a[1] == cs for a in arenas[3:]) and cs == _codes_stride(cb), \
            f"arena strides {[a[1] for a in arenas]} vs weights {wb} codes {cb}"
        w13 = slot.w13_weight.view(self.G, -1)
        w2 = slot.w2_weight.view(self.G, -1)
        s13 = slot.w13_scale.view(self.G, -1)
        s2 = slot.w2_scale.view(self.G, -1)
        t0 = time.perf_counter()
        with torch.cuda.stream(self.transfer_stream):
            if slot.compute_used:
                self.transfer_stream.wait_event(slot.compute_done)
            if TIMING:
                ev0 = torch.cuda.Event(enable_timing=True)
                ev0.record(self.transfer_stream)
            n, base = len(experts), experts[0]
            assert list(experts) == list(range(base, base + n)), "a group's experts are consecutive"
            st = self.transfer_stream
            for src, dst, pitch, width, stride in (
                    (arenas[0][0], w13.data_ptr(), 2 * wb, wb, ws),
                    (arenas[1][0], w13.data_ptr() + wb, 2 * wb, wb, ws),
                    (arenas[2][0], w2.data_ptr(), wb, wb, ws),
                    (arenas[3][0], s13.data_ptr(), 2 * cb, cb, cs),
                    (arenas[4][0], s13.data_ptr() + cb, 2 * cb, cb, cs),
                    (arenas[5][0], s2.data_ptr(), cb, cb, cs)):
                self._memcpy2d(dst, pitch, src.data_ptr() + base * stride, stride, width, n, st)
            slot.dma_done.record(self.transfer_stream)
            if TIMING:
                ev1 = torch.cuda.Event(enable_timing=True)
                ev1.record(self.transfer_stream)
                stats.setdefault("dma_ev", []).append((ev0, ev1))
        stats["dma_enq"] += time.perf_counter() - t0

    @staticmethod
    def _scale_landing(slot):
        """Where a group's scale arena blocks land: raw_s (fp32), or raw_c (codes and table) under CODES."""
        return slot.raw_c if CODES else slot.raw_s

    def _land_scales(self, slot, arenas, base: int, n: int):
        """Synchronous copy of experts [base, base + n)'s scale blocks into the slot, expanded into raw_s."""
        land = self._scale_landing(slot)
        st = land.shape[2]
        for pi in range(3):
            land[pi, :n].view(-1).copy_(arenas[3 + pi][0][base * st:(base + n) * st])
        if CODES:
            _expand_codes(slot.raw_c, slot.raw_s, n, self.N * (self.K // 16))

    def _scales_from_fp32(self, raw_s, n: int, s2g, s2d):
        """kt-kernel's FP32 scale blocks [3, n, stride] -> the swizzled e4m3
        bytes FlashInfer takes: w13 [n, 2N, K/16] in [up; gate] rows, w2
        [n, K, N/16]."""
        N, K = self.N, self.K
        w13_s = N * (K // 16)
        w2_s = K * (N // 16)
        s13 = torch.empty((n, 2 * N, K // 16), dtype=torch.uint8, device=self.device)
        s13[:, :N] = _to_e4m3(raw_s[1, :, :4 * w13_s].view(torch.float32).view(n, N, K // 16), s2g)
        s13[:, N:] = _to_e4m3(raw_s[0, :, :4 * w13_s].view(torch.float32).view(n, N, K // 16), s2g)
        s2 = _to_e4m3(raw_s[2, :, :4 * w2_s].view(torch.float32).view(n, K, N // 16), s2d)
        return _swizzle(s13), _swizzle(s2)

    def _build_scale_bytes(self, layer_idx: int):
        """Run the FP32 -> e4m3 conversion once for every expert of the layer
        and keep the bytes in pinned host memory."""
        if layer_idx in self._sb:
            return
        self._register_arenas(layer_idx)
        N, K = self.N, self.K
        E = self._num_experts(layer_idx)
        arenas = self._arenas[layer_idx]
        slot = self.slots[0]
        s2g_all, s2d_all = self._s2[layer_idx]
        h13 = torch.empty((E, 2 * N, K // 16), dtype=torch.uint8).pin_memory()
        h2 = torch.empty((E, K, N // 16), dtype=torch.uint8).pin_memory()
        cur = torch.cuda.current_stream(self.device)
        if slot.compute_used:
            cur.wait_event(slot.compute_done)
        cur.wait_stream(self.transfer_stream)
        for base in range(0, E, self.G):
            n = min(self.G, E - base)
            self._land_scales(slot, arenas, base, n)
            s13, s2 = self._scales_from_fp32(slot.raw_s[:, :n], n, s2g_all[base:base + n], s2d_all[base:base + n])
            h13[base:base + n].copy_(s13)
            h2[base:base + n].copy_(s2)
        self._sb[layer_idx] = (h13, h2)

    def _verify_scale_bytes(self, layer_idx: int, gi: int, base: int, n: int):
        """KT_GPU_STREAM_SCALE_BYTES_VERIFY: the bytes the DMA put in the slot
        against the FP32 arenas converted now. Synchronous; a diagnostic."""
        slot = self.slots[gi % 2]
        arenas = self._arenas[layer_idx]
        torch.cuda.current_stream(self.device).wait_event(slot.dma_done)
        self._land_scales(slot, arenas, base, n)
        s2g_all, s2d_all = self._s2[layer_idx]
        s13, s2 = self._scales_from_fp32(slot.raw_s[:, :n], n, s2g_all[base:base + n], s2d_all[base:base + n])
        bad = int((s13 != slot.w13_scale[:n]).sum().item()) + int((s2 != slot.w2_scale[:n]).sum().item())
        self._verified = getattr(self, "_verified", 0) + 1
        if bad:
            raise RuntimeError(f"KT_GPU_STREAM_SCALE_BYTES: layer {layer_idx} group {gi}: {bad} scale bytes differ")
        if self._verified % 384 == 0:
            _log(f"rank{self.rank} scale bytes verified against the FP32 arenas: {self._verified} groups, 0 mismatches")

    # -- one group ---------------------------------------------------------------

    def _group_body(self, layer, gi: int, base: int, n: int, g):
        """Repack + MoE + accumulate for one group on the current stream;
        captured into a graph per (token count, group). Everything that
        differs per layer (scale_2, alphas) is a graph input in g."""
        if self.w16:
            return self._group_body_w16(layer, gi, base, n, g)
        if self.cutile:
            return self._group_body_cutile(gi, base, n, g)
        slot = self.slots[gi % 2]
        N, K = self.N, self.K
        w13_bytes = N * K // 2
        w13_s = N * (K // 16)
        w2_bytes = K * N // 2
        w2_s = K * (N // 16)
        x, topk_weights, ids32, acc, out = g["x"], g["w"], g["ids"], g["acc"], g["out"]
        s2g, s2d, g1, g2 = g["s2g"][:n], g["s2d"][:n], g["g1"][:n], g["g2"][:n]
        raw = slot.raw[:, :n]
        raw_s = slot.raw_s[:, :n]
        # FlashInfer's CUTLASS MoE takes fc1 as [up; gate] rows (w31), the
        # order SGLang's swap_w13_to_w31 produces for it; the checkpoint's
        # [gate; up] gave outputs with an error above the signal (09-14
        # refcheck on the bf16 edition). Same nibble packing as kt-kernel.
        slot.w13_weight[:n, :N].copy_(raw[1, :, :w13_bytes].view(n, N, K // 2))
        slot.w13_weight[:n, N:].copy_(raw[0, :, :w13_bytes].view(n, N, K // 2))
        slot.w2_weight[:n].copy_(raw[2, :, :w2_bytes].view(n, K, N // 2))
        if not SCALE_BYTES:
            if CODES:
                _expand_codes(slot.raw_c, slot.raw_s, n, w13_s)
            s13, s2 = self._scales_from_fp32(raw_s, n, s2g, s2d)
            slot.w13_scale[:n].copy_(s13)
            slot.w2_scale[:n].copy_(s2)
        # else: _stage put the finished bytes in slot.w13_scale / w2_scale
        local = torch.where((ids32 >= base) & (ids32 < base + n), ids32 - base, torch.full_like(ids32, -1))
        T = x.shape[0]
        R = _block_rows(T)
        for r0 in range(0, T, R):
            r1 = min(T, r0 + R)
            self._moe_rows(slot, g, g1, g2, n, x[r0:r1], local[r0:r1], topk_weights[r0:r1], out[r0:r1])
        acc.add_(out)

    def _group_body_cutile(self, gi: int, base: int, n: int, g):
        """The group's MoE on FlashInfer's cuTile path, from the slot's checkpoint bytes, into acc; captured into a
        graph per (token count, group). The whole slot (G experts) is passed: the ids reach only the n loaded."""
        from sglang.srt.layers.moe import kt_cutile_nvfp4

        slot = self.slots[gi % 2]
        x, w, ids32, acc, out = g["x"], g["w"], g["ids"], g["acc"], g["out"]
        local = torch.where((ids32 >= base) & (ids32 < base + n), ids32 - base, torch.full_like(ids32, -1))
        wl = torch.where(local >= 0, w, torch.zeros_like(w))
        T = x.shape[0]
        R = _block_rows(T)
        runner = kt_cutile_nvfp4.stream_runner(self.G, self.K, self.N, ids32.shape[1], self._ct_act, bool(self._a4),
                                               max(R, _block_rows(_chunk_rows())), self.device)
        f8 = torch.float8_e4m3fn
        for r0 in range(0, T, R):
            r1 = min(T, r0 + R)
            kt_cutile_nvfp4.stream_run(runner, x[r0:r1], local[r0:r1], wl[r0:r1], slot.w13_weight,
                                       slot.w13_scale.view(f8), g["a13"], slot.w2_weight, slot.w2_scale.view(f8),
                                       g["a2c"], out[r0:r1], self._inv_g1)
        acc.add_(out)

    def _moe_rows(self, slot, g, g1, g2, n, x, local, topk_weights, out):
        """One CUTLASS MoE call over a block of rows."""
        self._cutlass_fused_moe(
            input=x,
            token_selected_experts=local,
            token_final_scales=topk_weights,
            fc1_expert_weights=slot.w13_weight[:n].view(torch.int64),
            fc2_expert_weights=slot.w2_weight[:n].view(torch.int64),
            output_dtype=torch.bfloat16,
            quant_scales=[
                g["a1"],
                slot.w13_scale[:n].view(torch.int32),
                g1,
                g["a2"],
                slot.w2_scale[:n].view(torch.int32),
                g2,
            ],
            input_sf=None,
            swiglu_alpha=None if self.swiglu_alpha is None else self.swiglu_alpha[:n],
            swiglu_beta=None if self.swiglu_beta is None else self.swiglu_beta[:n],
            swiglu_limit=None if self.swiglu_alpha is None else self.swiglu_limit[:n],
            tp_size=self._tp_size,
            tp_rank=self._tp_rank,
            ep_size=self._ep_size,
            ep_rank=self._ep_rank,
            activation_type=self._activation_type,
            tune_max_num_tokens=self._tune_tokens,
            output=out,
            use_fused_finalize=self._use_fused_finalize,
            workspace_buffer=self._workspace,
        )

    def _reference_check(self, method, layer, layer_idx, x, ids, w, acc):
        """This rank's partial MoE output for the streamed experts from a
        torch dequantisation of the arenas (e2m1 nibbles x e4m3 block scale
        x scale_2, bf16 activations, i.e. W4A16), against acc (W4A4 on the
        GPU). Not bit-exact: the activation quantisation is the difference."""
        torch.cuda.current_stream(self.device).synchronize()
        N, K = self.N, self.K
        ar = self._arenas[layer_idx]
        wst, sst = ar[0][1], ar[3][1]
        first = 0 if getattr(method.gpu_method, "streams_hot_experts", False) else method.num_gpu_experts
        s2g, s2d = self._s2.get(layer_idx, (None, None))
        e2m1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0, -.5, -1, -1.5, -2, -3, -4, -6], device=self.device)

        def deq(wb, sb, rows, cols, s2):
            q = wb.to(self.device).view(rows, cols // 2)
            lo, hi = (q & 0xF).long(), (q >> 4).long()
            v = torch.stack([e2m1[lo], e2m1[hi]], -1).view(rows, cols)
            sc = sb.to(self.device).view(rows, cols // 16).float().repeat_interleave(16, 1)
            return v * sc * s2

        ref = torch.zeros_like(acc, dtype=torch.float32)
        ids64 = ids.to(torch.int64)
        xf = x.float()
        used = 0
        for e in range(first, self._num_experts(layer_idx)):
            rows, cols = torch.where(ids64 == e)
            if rows.numel() == 0:
                continue
            used += 1
            gs = ar[3][0][e * sst:e * sst + 4 * N * (K // 16)].view(torch.float32)
            us = ar[4][0][e * sst:e * sst + 4 * N * (K // 16)].view(torch.float32)
            ds = ar[5][0][e * sst:e * sst + 4 * K * (N // 16)].view(torch.float32)
            gate = deq(ar[0][0][e * wst:e * wst + N * K // 2], gs, N, K, 1.0)  # d already = e4m3 x scale_2
            up = deq(ar[1][0][e * wst:e * wst + N * K // 2], us, N, K, 1.0)
            down = deq(ar[2][0][e * wst:e * wst + K * N // 2], ds, K, N, 1.0)
            h = xf[rows]
            a = torch.nn.functional.silu(h @ gate.t()) * (h @ up.t())
            o = (a @ down.t()) * w[rows, cols].float().unsqueeze(1)
            ref.index_add_(0, rows, o)
        d = (acc.float() - ref).abs()
        cos = torch.nn.functional.cosine_similarity(acc.float().flatten(), ref.flatten(), dim=0).item()
        _log(f"rank{self.rank} refcheck layer{layer_idx} T={x.shape[0]} experts_used={used}: cos={cos:.5f} "
             f"max|acc-ref|={d.max().item():.4f} mean={d.mean().item():.5f} ref_mean|.|={ref.abs().mean().item():.5f} "
             f"acc_mean|.|={acc.float().abs().mean().item():.5f}")

    def _graph_inputs(self, T: int, x, physical_ids, topk_weights):
        """Persistent graph inputs for this token count; the per-layer
        constants (input scales, scale_2 and alphas of the current group)
        are copied in before each replay."""
        g = self._gin.get(T)
        n = x.shape[0]
        pre = _prealloc_gin.get(self.device.index)
        if g is None and pre is not None and pre[0] >= T:
            g = {k: (v[:T] if k in ("x", "ids", "w", "acc", "out") else v) for k, v in pre[1].items()}
            g["x"].zero_()
            g["acc"].zero_()
            for k in ("a1", "a2", "s2g", "s2d", "g1", "g2"):
                g[k].fill_(1.0)
            if self.cutile:
                from sglang.srt.layers.moe import kt_lend

                with kt_lend.scratch_region():
                    g["a13"] = torch.ones((self.G, 2), dtype=torch.float32, device=x.device)
                    g["a2c"] = torch.ones((self.G,), dtype=torch.float32, device=x.device)
            self._gin[T] = g
        if g is None:
            from sglang.srt.layers.moe import kt_lend

            f32 = torch.float32
            # in the lend scratch, as the CUTLASS workspace: a row count's inputs outlive its window (its graphs
            # read them), and made as plain allocations they stayed on the card through decode. After a
            # 35K prompt 0.82 GB of them (4096, 8192 and 10424 rows) left the weights 70 MB, and the next scored
            # chat piece ran out of memory
            with kt_lend.scratch_region():
                g = {
                    "x": torch.zeros((T, x.shape[1]), dtype=x.dtype, device=x.device),
                    "ids": torch.empty((T, physical_ids.shape[1]), dtype=torch.int32, device=x.device),
                    "w": torch.empty((T, physical_ids.shape[1]), dtype=f32, device=x.device),
                    "acc": torch.zeros((T, self.K), dtype=torch.bfloat16, device=x.device),
                    "out": torch.empty((T, self.K), dtype=torch.bfloat16, device=x.device),
                    "a1": torch.ones((), dtype=f32, device=x.device),  # scalar: FlashInfer checks ndim
                    "a2": torch.ones((), dtype=f32, device=x.device),
                    "s2g": torch.ones((self.G,), dtype=f32, device=x.device),
                    "s2d": torch.ones((self.G,), dtype=f32, device=x.device),
                    "g1": torch.ones((self.G,), dtype=f32, device=x.device),
                    "g2": torch.ones((self.G,), dtype=f32, device=x.device),
                }
                if self.cutile:
                    g["a13"] = torch.ones((self.G, 2), dtype=f32, device=x.device)  # gate, up
                    g["a2c"] = torch.ones((self.G,), dtype=f32, device=x.device)
            self._gin[T] = g
        g["x"][:n].copy_(x)
        g["ids"][:n].copy_(physical_ids)
        g["w"][:n].copy_(topk_weights)
        if n < T:
            # padding rows: no expert in any group, nothing added to acc
            g["x"][n:].zero_()
            g["ids"][n:].fill_(-1)
            g["w"][n:].zero_()
        g["acc"].zero_()
        return g

    def _set_layer_inputs(self, g, layer, layer_idx: int = -1):
        if self.cutile and self._needs_inv_g1:
            # this layer's 1 / g1, which the group graphs' quantize kernel reads (patches/flashinfer-cutile-fp4.py
            # part 7); made at the first layer, before any group graph captures its address
            if self._inv_g1 is None:
                self._inv_g1 = torch.ones((1,), dtype=torch.float32, device=g["x"].device)
            self._inv_g1.fill_(1.0 / self._g12.get(layer_idx, (1.0, 1.0))[0])
        if self.w16 or self.cutile:
            return
        g["a1"].copy_(_input_quant(layer, "w13"))
        g["a2"].copy_(_input_quant(layer, "w2"))

    def _set_group_inputs(self, g, layer, layer_idx: int, base: int, n: int):
        if self.w16:
            return
        if self.cutile:
            s2g, s2d = self._s2[layer_idx]
            s2u = self._s2u.get(layer_idx, s2g)
            g1, g2 = self._g12.get(layer_idx, (1.0, 1.0))
            # kt_cutile_nvfp4.fold_globals: gate times g1, up times g1 / g2, w2 times g2
            g["a13"][:n, 0].copy_(s2g[base:base + n]).mul_(g1)
            g["a13"][:n, 1].copy_(s2u[base:base + n]).mul_(g1 / g2)
            g["a2c"][:n].copy_(s2d[base:base + n]).mul_(g2)
            return
        s2g, s2d = self._s2[layer_idx]
        g1, g2 = self._alphas_for(layer, layer_idx)
        g["s2g"][:n].copy_(s2g[base:base + n])
        g["s2d"][:n].copy_(s2d[base:base + n])
        g["g1"][:n].copy_(g1[base:base + n])
        g["g2"][:n].copy_(g2[base:base + n])

    def _trim_graphs(self):
        """Drop the graphs of the least recently used row counts beyond
        GRAPH_KEEP, never those of the largest (the full chunk)."""
        top = max(self._graph_use)
        old = [t for t in self._graph_use if t != top]
        if len(old) <= GRAPH_KEEP:
            return
        drop = set(old[: len(old) - GRAPH_KEEP])
        torch.cuda.synchronize(self.device)
        for k in [k for k in self._graphs if k[0] in drop]:
            del self._graphs[k]
        for t in drop:
            del self._graph_use[t]

    def _cutile_rows(self, T: int) -> None:
        """The cuTile path holds one row count's graphs, graph inputs and runner workspaces at a time: FlashInfer's
        runner keeps a workspace per token bucket, and those of earlier windows stayed while a larger one was made (a
        35K prompt after another: out of memory at a 16384-row window the plan had priced at one row count's cost).
        A new row count first gives the others back; its graphs are captured again (G of them, the kernels compiled
        before come from cuTile's disk cache)."""
        if getattr(self, "_ct_T", None) == T:
            return
        from sglang.srt.layers.moe import kt_cutile_nvfp4

        torch.cuda.synchronize(self.device)
        for k in [k for k in self._graphs if k[0] != T]:
            del self._graphs[k]
        for t in [t for t in self._graph_use if t != T]:
            del self._graph_use[t]
        for t in [t for t in self._gin if t != T]:
            del self._gin[t]
        for runner in kt_cutile_nvfp4._STREAM.values():
            runner._workspace_cache.clear()
            runner._workspace = None
        self._ct_T = T

    def _graph_for(self, layer, T: int, gi: int, base: int, n: int, g):
        key = (T, gi)
        graph = self._graphs.get(key)
        self._graph_use.pop(T, None)
        self._graph_use[T] = None
        if graph is None:
            t0 = time.perf_counter()
            cs = self._capture_stream
            cs.wait_stream(torch.cuda.current_stream(self.device))
            # The eager warm-up (lazy inits, kernel selection) must not touch
            # the real accumulator: it ran acc.add_ once more than the replay
            # and the first request per chunk size carried the streamed
            # experts twice (09-14 refcheck).
            gw = dict(g)
            gw["acc"] = torch.zeros_like(g["acc"])
            with torch.cuda.stream(cs):
                self._group_body(layer, gi, base, n, gw)
            cs.synchronize()
            del gw
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self._graph_pool, stream=cs):
                self._group_body(layer, gi, base, n, g)
            torch.cuda.current_stream(self.device).wait_stream(cs)
            self._graphs[key] = graph
            self._trim_graphs()
            if TIMING and gi == 0:
                _log(f"rank{self.rank} captured group graph T={T} gi={gi} n={n} in {(time.perf_counter()-t0)*1e3:.0f} ms")
        return graph

    # -- one layer ----------------------------------------------------------

    def run(self, method, layer, dispatch_output, physical_ids: torch.Tensor) -> torch.Tensor:
        """Sum of the streamed experts' outputs for this layer, [tokens, hidden]."""
        def next_power_of_2(n: int) -> int:
            return 1 << (n - 1).bit_length() if n > 0 else 1

        x = dispatch_output.hidden_states
        rows = x.shape[0]
        T = rows
        if FIXED_ROWS and rows < _chunk_rows():
            T = _chunk_rows()
            if ROW_STEP > 0:
                T = min(T, -(-rows // ROW_STEP) * ROW_STEP)
        layer_idx = method.kt_config.layer_idx
        Tb = _block_rows(T)
        # one value for every chunk size: the kernel's own buffers follow it (the bf16 edition, 09-14)
        self._tune_tokens = max(next_power_of_2(_block_rows(max(T, _chunk_rows()))), 2048)
        if self.w16 and self.w13_bf16 is None:
            self._w16_buffers()
        if self._workspace is None and not self.cutile:
            # sized once for the prefill chunk (in blocks) and never replaced: the group graphs
            # bake its address in; in the lend scratch, so decode does not hold it
            from flashinfer.fused_moe import cutlass_fused_moe_workspace_size
            from sglang.srt.layers.moe import kt_lend

            Tb = _block_rows(max(T, _chunk_rows()))
            nbytes = cutlass_fused_moe_workspace_size(
                Tb, self.K, self.N, self.G * layer.moe_ep_size, physical_ids.shape[1],
                x_dtype=torch.bfloat16, weight_dtype=torch.bfloat16 if self.w16 else torch.int64,
                output_dtype=torch.bfloat16,
                activation_type=self._activation_type, tp_size=layer.moe_tp_size, tp_rank=layer.moe_tp_rank,
                ep_size=layer.moe_ep_size, ep_rank=layer.moe_ep_rank, use_w4_group_scaling=False,
                use_mxfp8_act_scaling=False, use_fused_finalize=self._use_fused_finalize, device=x.device)
            with kt_lend.scratch_region():
                self._workspace = torch.empty(nbytes, dtype=torch.uint8, device=x.device)
            self._workspace_tokens = Tb
            _log(f"rank{self.rank} cutlass workspace {nbytes/1e6:.0f} MB for {Tb} tokens x {self.G} experts"
                 + (f" (chunk {T} in blocks)" if Tb < T else ""))
        topk_weights = dispatch_output.topk_output.topk_weights
        # the resident experts too when the gpu method streams them (KT_GPU_STREAM_RESIDENTS)
        first = 0 if getattr(method.gpu_method, "streams_hot_experts", False) else method.num_gpu_experts
        last = self._num_experts(layer_idx)
        groups = [list(range(b, min(b + self.G, last))) for b in range(first, last, self.G)]
        stats = {"dma_enq": 0.0, "graph_enq": 0.0}
        t0 = time.perf_counter()
        if TIMING:
            e_in = torch.cuda.Event(enable_timing=True)
            e_in.record(torch.cuda.current_stream(self.device))
            t_wait = time.perf_counter()
            torch.cuda.current_stream(self.device).synchronize()  # the work queued before the streamer
            t0 = time.perf_counter()
            stats["before"] = t0 - t_wait
        if self.cutile:
            self._cutile_rows(T)
        g = self._graph_inputs(T, x, physical_ids, topk_weights)
        self._set_layer_inputs(g, layer, layer_idx)
        cur = torch.cuda.current_stream(self.device)
        if SCALE_BYTES and not self.w16 and not self.cutile:  # W4A4's bytes
            self._build_scale_bytes(layer_idx)
        self._stage(layer_idx, 0, groups[0], stats)
        for gi, experts in enumerate(groups):
            slot = self.slots[gi % 2]
            graph = self._graph_for(layer, T, gi, experts[0], len(experts), g)
            t1 = time.perf_counter()
            self._set_group_inputs(g, layer, layer_idx, experts[0], len(experts))
            cur.wait_event(slot.dma_done)
            if SCALE_BYTES_VERIFY and SCALE_BYTES:
                self._verify_scale_bytes(layer_idx, gi, experts[0], len(experts))
            if TIMING:
                c0 = torch.cuda.Event(enable_timing=True)
                c0.record(cur)
            graph.replay()
            if TIMING:
                c1 = torch.cuda.Event(enable_timing=True)
                c1.record(cur)
                stats.setdefault("gpu_ev", []).append((c0, c1))
            slot.compute_done.record(cur)
            slot.compute_used = True
            stats["graph_enq"] += time.perf_counter() - t1
            if gi + 1 < len(groups):
                self._stage(layer_idx, gi + 1, groups[gi + 1], stats)
        self.calls += 1
        if TIMING:
            cur.synchronize()
            self.transfer_stream.synchronize()
            dma = sum(a.elapsed_time(b) for a, b in stats.get("dma_ev", []))
            gpu = sum(a.elapsed_time(b) for a, b in stats.get("gpu_ev", []))
            first = stats["dma_ev"][0][0] if stats.get("dma_ev") else None
            last = stats["gpu_ev"][-1][1] if stats.get("gpu_ev") else None
            span = first.elapsed_time(last) if first is not None and last is not None else 0.0
            _log(f"rank{self.rank} layer{layer_idx} tokens={rows}/{T} groups={len(groups)} "
                 f"dma_enq={stats['dma_enq']*1e3:.0f}ms graph_enq={stats['graph_enq']*1e3:.0f}ms "
                 f"dma_gpu={dma:.0f}ms moe_gpu={gpu:.0f}ms span_gpu={span:.0f}ms "
                 f"before={stats['before']*1e3:.0f}ms total={(time.perf_counter()-t0)*1e3:.0f}ms")
        if os.environ.get("KT_GPU_STREAM_SELFTEST") == "2" and layer_idx == min(self._arenas):
            self._reference_check(method, layer, layer_idx, x, physical_ids, topk_weights, g["acc"][:rows])
        # The caller adds this before the next layer zeroes it (same stream).
        return g["acc"] if rows == T else g["acc"][:rows]


_instance: Optional[KTStreamPrefill] = None
_failed = False


def _start_stack_dumper():
    period = float(os.environ.get("KT_GPU_STREAM_DEBUG", "0") or 0)
    if period <= 0:
        return
    import sys
    import threading

    main_id = threading.main_thread().ident
    rank = get_tensor_model_parallel_rank()

    def loop():
        while True:
            time.sleep(period)
            fr = sys._current_frames().get(main_id)
            if fr is not None:
                _log(f"rank{rank} main thread stack:" + chr(10) + "".join(traceback.format_stack(fr)[-12:]))

    threading.Thread(target=loop, daemon=True, name="kt-stream-stackdump").start()


# KT_GPU_STREAM_EARLY_INIT=1: build the streamer and cudaHostRegister every
# layer's arenas right after the weights load, instead of on the first
# qualifying request (09-26: 15.5 min on the first 38K prompt, 4K pages).
EARLY_INIT = os.environ.get("KT_GPU_STREAM_EARLY_INIT", "0") == "1"


def stream_prefill_init_early(method, layer) -> None:
    global _instance, _failed
    if not EARLY_INIT or STREAM_THRESHOLD <= 0 or _failed or _instance is not None:
        return
    rank = get_tensor_model_parallel_rank()
    t0 = time.time()
    try:
        _instance = KTStreamPrefill(method, layer)
        _start_stack_dumper()
        for layer_idx in sorted(_instance._arenas):
            _instance._register_arenas(layer_idx)
        if SCALE_BYTES and not _instance.w16 and not _instance.cutile:  # the cuTile path reads the arenas' codes
            t1 = time.time()
            for layer_idx in sorted(_instance._arenas):
                _instance._build_scale_bytes(layer_idx)
            nb = sum(a.numel() + b.numel() for a, b in _instance._sb.values())
            _log(f"rank {rank}: one-byte block scales for {len(_instance._sb)} layers, "
                 f"{nb/1e9:.2f} GB pinned, built in {time.time() - t1:.1f} s")
    except Exception:
        _failed = True
        _log("early init failed:\n" + traceback.format_exc())
        raise
    _log(f"rank {rank}: streamer built and {len(_instance._arenas)} layers of arenas registered "
         f"at load time in {time.time() - t0:.0f} s")


def stream_prefill_for(method, layer, num_tokens: int):
    """The per-rank streamer when this call qualifies, else None."""
    global _instance, _failed
    if STREAM_THRESHOLD <= 0 or _failed:
        return None
    if num_tokens < kt_stream_threshold.get() or torch.cuda.is_current_stream_capturing():
        return None
    if _instance is None:
        _log(f"rank {get_tensor_model_parallel_rank()}: first qualifying call, layer {method.kt_config.layer_idx}, tokens={num_tokens}")
        try:
            _instance = KTStreamPrefill(method, layer)
            _start_stack_dumper()
        except Exception:
            _failed = True
            _log("init failed:\n" + traceback.format_exc())
            raise
    return _instance
