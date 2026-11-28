"""More hot experts in the KV pool's free pages, outside prefill, for any GPU MoE
method of the KT wrapper (KT_KV_HOT_LEND=1, KT_KV_HOT_EXTRA=<X>; with
KT_KV_VMM=1).

The KT wrapper runs physical experts 0..N-1 on the GPU (N =
--kt-num-gpu-experts) and the rest on the CPU. With a hot-expert placement the
physical order is the hotness order, so the next X experts, N..N+X-1, are the
ones most worth adding. This module:

- at load, makes a shadow of each MoE layer outside the model, with a copy of
  the GPU method's weights for X experts on the CPU (make_shadow); the weight
  loader puts experts N..N+X-1 there (shadow_param); after the layer's own
  processing the shadow's are processed on the GPU into the method's format,
  copied to pinned host memory, and its per-expert tensors (dim 0 of X) become
  slots on kv_vmm shadow ranges, every byte of which reads as zero until mapped
  (split). The layer itself is exactly as without this; the CPU loads all of
  its experts as before, those X included;
- outside prefill, takes the KV pool's free pages beyond KT_KV_HOT_RESERVE
  tokens out of the free list and unmaps their whole chunks
  (kt_kv_prefill_lend.decode_lend), maps that much memory under the slots,
  layer by layer one slot at a time, copies the experts in from the host, and
  sets each layer's lent count k on the device;
- in the wrapper's apply, a route to a physical id in [N, N + k) goes to the
  CPU with weight 0 at id 0 (a GPU id, which the CPU skips) and to the shadow's
  GPU method as slot id - N; every other route is as before;
- when the pool needs the pages (an allocation the free list cannot serve, a
  prefill window, a cache flush), sets every k to 0, syncs, maps the zero chunk
  back under the slots, then the pages' chunks are mapped back, zeroed
  (kt_kv_prefill_lend.restore, which calls release first).

The pool never changes size and the lent pages count as available, as with the
prefill loan. k lives on the device and the slots never move, so the captured
decode graphs stay valid; a slot that holds nothing reads zeros. Every rank
lends at the same forward (the allocator's free page count is the same on each),
and every rank holds the same slots: the CPU runs on rank 0 with its routing, so
a slot another rank lacked would leave its share of that route uncomputed.

KT_KV_HOT_DYNAMIC_SLOTS=1: the slots follow what is routed, so whoever serves
whatever gets their own hot experts without recording a placement (a writer
placement's 21 a layer catch 3-6% of the routes of code, JSON or math, 10-21):
- a device table, physical id -> slot or -1, replaces the [N, N + k) range in
  route, and route counts every routed physical id on the device;
- every few decode forwards, outside the graph (the period and the bar a swap must
  clear come from what the rounds measure: "the period and the bar" below;
  KT_KV_HOT_DYN_EVERY and KT_KV_HOT_DYN_HYST fix them by hand), the counts
  come to the host as decayed scores (half-life KT_KV_HOT_DYN_HALF_LIFE forwards,
  256), and each layer swaps up to KT_KV_HOT_DYN_SWAPS a forward (2) of its held
  experts for better-scored ones; a swapped-out expert leaves the table at once
  (its routes go to the CPU, which holds every non-resident expert), the new one
  is copied into the slot on a side stream from kt-kernel's expert arenas (the
  zero-copy mapping the streamed prefill registered: no host copy) and enters
  the table on the next round, behind the copy;
- the decisions use only the counts, the same on every rank, and the slot counts
  are the smallest over the ranks: the CPU (rank 0's routing) skips exactly the
  routes every rank's slots take;
- a new lend loads what was held before the window, from the arenas.
"""
import copy
import os
import time
from typing import List

import numpy as np
import torch

from sglang.srt.layers.moe import kt_kv_vmm

ENABLED = os.environ.get("KT_KV_HOT_LEND", "0") == "1" and kt_kv_vmm.ENABLED
# KT_KV_HOT_EXTRA=<X> slots a layer, or auto: with the dynamic slots every expert past the
# resident ones (virtual address space only: how many hold memory follows the free memory
# at each lend), else 16
_EXTRA_RAW = (os.environ.get("KT_KV_HOT_EXTRA", "0") or "0").strip().lower()
EXTRA_AUTO = _EXTRA_RAW == "auto"
EXTRA = 0 if EXTRA_AUTO else int(_EXTRA_RAW)
# tokens kept free outside prefill. With the partial take-back (KT_KV_HOT_PARTIAL, the default) an allocation takes
# back the pages it needs and no more, so none are kept; without it every take-back gives the whole lend back
_PARTIAL = os.environ.get("KT_KV_HOT_PARTIAL", "1") == "1"
RESERVE = int(os.environ.get("KT_KV_HOT_RESERVE", "") or (0 if _PARTIAL else 4096))
# a test's switch: a new lend leaves this many tokens of pages more, which the next forward's grow takes (the grow
# with slots added and filled, without waiting for a request to free pages)
GROW_HOLD = int(os.environ.get("KT_KV_HOT_GROW_HOLD", "0") or 0)
MIN_MB = int(os.environ.get("KT_KV_HOT_MIN_MB", "256"))  # smallest lend worth a sync and a copy
RETRY = int(os.environ.get("KT_KV_HOT_RETRY", "256"))  # forwards between tries when nothing was lent
MARGIN_MB = int(os.environ.get("KT_KV_HOT_MARGIN_MB", "64"))  # of the unmapped bytes left unused
# KT_KV_HOT_SPARE_FLOOR_MB=<n>: the slots also take what the
# driver has free in decode beyond n MiB (the prefill's room, idle between windows)
SPARE_FLOOR_MB = int(os.environ.get("KT_KV_HOT_SPARE_FLOOR_MB", "-1") or -1)
# KT_KV_HOT_SLOTS_MAX=<n>: at most n slots a rank (0: no cap). Measurements set it below
# what every lend gets, so each lend, launch and rank holds the same slots and the same
# routes run on the GPU (the count otherwise follows the free memory)
SLOTS_MAX = int(os.environ.get("KT_KV_HOT_SLOTS_MAX", "0") or 0)
# KT_KV_HOT_CHECK=<n>: every n-th eager MoE call, the layer's output is compared
# with the same call computed without the lend (the resident experts again, the
# CPU with every route), summed over the TP ranks (check)
CHECK = int(os.environ.get("KT_KV_HOT_CHECK", "0") or 0)
_CHECK_DUMP = os.environ.get("KT_KV_HOT_CHECK_DUMP", "")  # a directory: see check
_CHECK_DUMP_FIRST = int(os.environ.get("KT_KV_HOT_CHECK_DUMP_FIRST", "2"))  # checks dumped whatever they say
_MASK_ZERO = os.environ.get("KT_GPU_MASK_ZERO", "0") == "1"
_SLOT_REL = float(os.environ.get("KT_KV_HOT_CHECK_SLOT_REL", "0.05") or 0.05)
_DUMP_REL = float(os.environ.get("KT_KV_HOT_CHECK_DUMP_REL", "0.2") or 0.2)  # a call dumped past this
_CHECK_REPEAT = os.environ.get("KT_KV_HOT_CHECK_REPEAT", "0") == "1"
DYNAMIC = os.environ.get("KT_KV_HOT_DYNAMIC_SLOTS", "0") == "1"
# The period of the rounds and the bar a swap must clear are derived from what the rounds measure (_DRV below).
# Until those measures stand, and where they never do, a round every 32 forwards and new > victim x 2.0.
# KT_KV_HOT_DYN_EVERY / KT_KV_HOT_DYN_HYST set by hand fix the period / the bar as that multiple.
_EVERY_ENV = os.environ.get("KT_KV_HOT_DYN_EVERY", "")
_HYST_ENV = os.environ.get("KT_KV_HOT_DYN_HYST", "")
DYN_EVERY = max(1, int(_EVERY_ENV or 32))
DYN_SWAPS = int(os.environ.get("KT_KV_HOT_DYN_SWAPS", "2") or 2)
DYN_HALF_LIFE = float(os.environ.get("KT_KV_HOT_DYN_HALF_LIFE", "256") or 256)
# (at 1.0 near-equal experts trade places every round, and each trade is a load the decode beside it pays for)
DYN_HYST = float(_HYST_ENV or 2.0)
DYN_LOG = int(os.environ.get("KT_KV_HOT_DYN_LOG", "4096") or 4096)  # forwards between log lines
# GGUF swaps loaded by a background pool into pinned staging
DYN_ASYNC = os.environ.get("KT_KV_HOT_DYN_ASYNC", "1") == "1"
# The staging holds one round's swaps: KT_KV_HOT_DYN_ASYNC_MB of pinned memory (512), no more items than a round's
# budget over every layer. (24 items until 11-16: slots that start on experts nobody routes to took thousands of
# forwards to reach the routed ones, and the cap went to the first layers.) KT_KV_HOT_DYN_ASYNC_CAP fixes the items.
DYN_ASYNC_MB = max(1, int(os.environ.get("KT_KV_HOT_DYN_ASYNC_MB", "512") or 512))
DYN_ASYNC_CAP = int(os.environ.get("KT_KV_HOT_DYN_ASYNC_CAP", "0") or 0)
# its threads: KT_KV_HOT_DYN_ASYNC_THREADS, else a quarter of the CPUs this process may run on, 2 to 8
DYN_ASYNC_THREADS = int(os.environ.get("KT_KV_HOT_DYN_ASYNC_THREADS", "0") or 0)
# KT_KV_HOT_DYN_VERIFY=1: before a batch's slots enter the table, read them back and compare them byte for byte with
# the expert's GGUF rows read straight from the file (a check: it waits for the GPU)
DYN_VERIFY = os.environ.get("KT_KV_HOT_DYN_VERIFY", "0") == "1"
# KT_KV_HOT_DYNAMIC_RESIDENT=1 (with the dynamic slots): the N resident experts swap too.
# kt-kernel then loads every expert (no GPU mask), so an expert that leaves a resident row
# runs on the CPU; route sends the CPU -1 for every route the GPU takes. In a prefill
# window the rows hold experts 0..N-1 again, as the streamer expects.
DYN_RES = DYNAMIC and os.environ.get("KT_KV_HOT_DYNAMIC_RESIDENT", "0") == "1"

# KT_KV_HOT_ROUTE_TORCH=1: the dynamic route as separate torch ops instead of one kernel
ROUTE_TORCH = os.environ.get("KT_KV_HOT_ROUTE_TORCH", "0") == "1"

try:
    import triton
    import triton.language as tl
except ImportError:  # the torch ops serve
    triton = None

if triton is not None:

    @triton.jit
    def _route_kernel(ids_ptr, w_ptr, tbl_ptr, utbl_ptr, rtbl_ptr, cnt_ptr, cid_ptr, cw_ptr, sid_ptr, sw_ptr, row_ptr, n,
                      HAS_RES: tl.constexpr, MASK_ZERO: tl.constexpr, CPU_SKIP: tl.constexpr,
                      BLOCK: tl.constexpr):
        """One layer's routes in one launch: count each routed expert, look it up in the
        slot table (and the resident rows' table), and write the CPU's ids and weights
        (CPU_SKIP at weight 0 for what the GPU takes), the slots' and the rows' ids."""
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        e = tl.load(ids_ptr + offs, mask=m, other=-1).to(tl.int64)
        w = tl.load(w_ptr + offs, mask=m, other=0.0)
        valid = m & (e >= 0)
        ec = tl.where(valid, e, 0)
        tl.atomic_add(cnt_ptr + ec, tl.where(valid, 1.0, 0.0), mask=valid)
        s = tl.load(tbl_ptr + ec, mask=valid, other=-1)
        u = tl.load(utbl_ptr + ec, mask=valid, other=-1)  # any rank's slot (this rank's table on one rank)
        if HAS_RES:
            r = tl.load(rtbl_ptr + ec, mask=valid, other=-1)
            res = valid & (r >= 0)
            lent = valid & (s >= 0) & (r < 0)
            tl.store(row_ptr + offs, tl.where(res, r, -1).to(row_ptr.dtype.element_ty), mask=m)
        else:
            res = valid & (s < -1)  # never
            lent = valid & (s >= 0)
        gpu = res | (valid & (u >= 0))
        tl.store(cid_ptr + offs, tl.where(gpu, CPU_SKIP, e).to(cid_ptr.dtype.element_ty), mask=m)
        tl.store(cw_ptr + offs, tl.where(gpu, 0.0, w).to(cw_ptr.dtype.element_ty), mask=m)
        if MASK_ZERO:
            tl.store(sid_ptr + offs, tl.where(lent, s, 0).to(sid_ptr.dtype.element_ty), mask=m)
            tl.store(sw_ptr + offs, tl.where(lent, w, 0.0).to(sw_ptr.dtype.element_ty), mask=m)
        else:
            tl.store(sid_ptr + offs, tl.where(lent, s, -1).to(sid_ptr.dtype.element_ty), mask=m)
            tl.store(sw_ptr + offs, w.to(sw_ptr.dtype.element_ty), mask=m)
else:
    _route_kernel = None


def _route_fused(L, ids, w):
    """The dynamic route in one kernel launch (route lists what each output is)."""
    ids, w = ids.contiguous(), w.contiguous()
    cid, cw, sw = torch.empty_like(ids), torch.empty_like(w), torch.empty_like(w)
    sid = torch.empty(ids.shape, dtype=torch.int64, device=ids.device) if _MASK_ZERO else torch.empty_like(ids)
    rows = torch.empty_like(ids) if DYN_RES else None
    n = ids.numel()
    if n:
        tbl, cnt = STATE["tbl"][L.idx], STATE["cnt"][L.idx]
        _route_kernel[(triton.cdiv(n, 128),)](
            ids, w, tbl, STATE["utbl"][L.idx], STATE["rtbl"][L.idx] if DYN_RES else tbl, cnt, cid, cw, sid, sw,
            rows if DYN_RES else cid, n, HAS_RES=DYN_RES, MASK_ZERO=_MASK_ZERO,
            CPU_SKIP=-1 if DYN_RES else 0, BLOCK=128)
    return cid, cw, sid, sw, rows


def _on() -> bool:
    return ENABLED and (EXTRA > 0 or EXTRA_AUTO)


def _banks() -> int:
    """Experts the shadow loads from the checkpoint and keeps in pinned memory: with the
    dynamic slots two (the self test's reference; the slots load from kt-kernel's arenas),
    else one a slot."""
    x = EXTRA if EXTRA > 0 else 16
    return min(2, x) if DYNAMIC else x


def _cap(wrapper) -> int:
    """Slots a layer can have (KT_KV_HOT_EXTRA)."""
    if EXTRA_AUTO:
        return int(wrapper.global_num_experts) - int(wrapper.num_gpu_experts) if DYNAMIC else 16
    return EXTRA


# slot tensors under this many bytes an expert live on plain device memory
_FIXED_BYTES = 1024

LAYERS: List["_Layer"] = []
STATE = {"k": None, "want": True, "since": 0, "lent_slots": 0, "lends": 0, "releases": 0, "hooked": False,
         "calls": 0, "checks": 0}


def _log(msg: str) -> None:
    print(f"[kt-kv-hot {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_count(n: int) -> int:
    """GPU experts a loader that skips CPU experts' tensors must still read: the
    resident N and, with the hot lend, the X the shadows hold."""
    return n + _banks() if _on() and n > 0 else n


def cpu_holds_all(wrapper) -> bool:
    """Whether kt-kernel is to load every expert, the resident ones too (no GPU mask):
    the resident rows swap (KT_KV_HOT_DYNAMIC_RESIDENT), so route decides alone what
    the CPU computes. Only where make_shadow makes a shadow, so route runs."""
    return (DYN_RES and _on() and int(wrapper.num_gpu_experts) > 0
            and os.environ.get("KT_HOT_DYNAMIC", "0") != "1")


def make_shadow(wrapper, layer, hidden_size: int, intermediate_size_per_partition: int,
                params_dtype, extra_weight_attrs) -> None:
    """The wrapper's create_weights, after the GPU method made the layer's N
    experts: a shadow of the layer, outside the model (so no span of the prefill
    lend holds it), with the method's weights for X experts on the CPU; the
    weight loader fills them (shadow_param), split processes them."""
    if not _on() or int(wrapper.num_gpu_experts) <= 0:
        return
    if os.environ.get("KT_HOT_DYNAMIC", "0") == "1":
        # the GGUF stack's dynamic cache routes the CPU and the GPU itself, and
        # lends the KV pool's pages its own way (KT_HOT_KV_LEND)
        raise RuntimeError("KT_KV_HOT_LEND=1 with KT_HOT_DYNAMIC=1: use KT_HOT_KV_LEND for the dynamic cache")
    shadow = copy.copy(layer)
    shadow.__dict__["kt_hot_slots"] = True  # the slots, not the resident experts (kt_gguf_hot.py under EP)
    shadow.__dict__["_parameters"] = {}
    shadow.__dict__["_buffers"] = {}
    shadow.__dict__["_modules"] = {}
    method = copy.copy(wrapper.gpu_method)
    from sglang.srt.layers.moe import kt_cutile_nvfp4

    if type(kt_cutile_nvfp4.base_method(method)).__name__ == "ModelOptNvFp4FusedMoEMethod":
        shadow.num_local_experts = _banks()  # it sizes its tensors by the layer
    with torch.device("cpu"):
        method.create_weights(layer=shadow, num_experts=_banks(), hidden_size=hidden_size,
                              intermediate_size_per_partition=intermediate_size_per_partition,
                              params_dtype=params_dtype, **extra_weight_attrs)
    for p in shadow._parameters.values():
        if isinstance(p, torch.Tensor) and p.is_cuda:
            p.data = p.data.cpu()
    wrapper.__dict__["_kt_kv_hot_shadow"] = (shadow, method, set(vars(shadow)))
    # a loader that skips the checkpoint tensors of experts with no GPU slot
    # (DeepSeek V4's KT_LOAD_SKIP_CPU_EXPERTS) must still read these X
    wrapper.__dict__["load_gpu_experts"] = int(wrapper.num_gpu_experts) + _banks()


def shadow_param(wrapper, layer, param, expert_id: int):
    """The layer's weight loader, for a physical expert the GPU does not hold:
    (the shadow's tensor of the same name, its slot) when it is one of the X,
    else None."""
    sh = wrapper.__dict__.get("_kt_kv_hot_shadow")
    if sh is None:
        return None
    n = int(wrapper.num_gpu_experts)
    if not n <= expert_id < n + _banks():
        return None
    for name, p in layer._parameters.items():
        if p is param:
            sp = sh[0]._parameters.get(name)
            if sp is None:
                return None
            wrapper.__dict__["_kt_kv_hot_loads"] = wrapper.__dict__.get("_kt_kv_hot_loads", 0) + 1
            return sp, expert_id - n
    return None


def _per_expert(module, rows: int):
    """(kind, name, tensor) of a module's own CUDA tensors with rows rows: its
    parameters, buffers and plain tensor attributes (some methods keep the
    repacked weights as attributes)."""
    out, seen = [], set()
    for kind, items in (("param", module._parameters.items()), ("buffer", module._buffers.items()),
                        ("attr", [(k, v) for k, v in vars(module).items() if not k.startswith("_")])):
        for name, t in items:
            if not isinstance(t, torch.Tensor) or not t.is_cuda or t.dim() < 1 or t.shape[0] != rows:
                continue
            if 0 in t.stride():  # one value seen at a per-expert shape: nothing per expert to move
                continue
            if id(t) in seen:
                continue
            seen.add(id(t))
            out.append((kind, name, t))
    return out


class _Layer:
    def __init__(self, wrapper, idx: int):
        self.wrapper, self.idx = wrapper, idx
        # x slots, the first b of which have a bank (the checkpoint's processed experts)
        self.n, self.x, self.b = int(wrapper.num_gpu_experts), max(_cap(wrapper), _banks()), _banks()
        self.k = 0
        self.names, self.slots, self.banks, self.sz = [], {}, {}, {}
        self.shadow = self.method = None
        self.layer_idx = int(wrapper.kt_config.layer_idx)
        self.e = int(wrapper.global_num_experts)
        # KT_KV_HOT_DYNAMIC_SLOTS: the physical expert each slot is to hold (-1: none);
        # the placement's N..N+X-1 to start with, as the banks hold
        self.held = [self.n + j for j in range(self.x)]
        # KT_KV_HOT_DYNAMIC_RESIDENT: the expert each resident row is to hold, the one it
        # holds now, and the layer's per-expert tensors by role
        self.res_held = list(range(self.n))
        self.res_now = list(range(self.n))
        self.res_t = None
        # slot tensors on plain device memory (a few bytes an expert), never lent
        self.fixed = set()
        # ModelOpt NVFP4: the method's w13 / w2 input scales ([1] fp32), for the alphas
        self.nv_isc = None
        self.layer = None


def split(wrapper, layer) -> None:
    """The wrapper's process_weights_after_loading, after the layer's own: the
    shadow's X experts are processed on the GPU by its copy of the method, moved
    to pinned host memory, and its per-expert tensors become slots."""
    sh = wrapper.__dict__.pop("_kt_kv_hot_shadow", None)
    if sh is None:
        return
    shadow, method, made = sh
    x = _banks()
    loads = wrapper.__dict__.pop("_kt_kv_hot_loads", 0)
    gguf = getattr(method, "reader", None) is not None and hasattr(method, "fill_experts")
    if loads == 0 and not gguf:
        # a loader that never hands these experts to the layer's weight loader
        # (one that skips the CPU experts' tensors) leaves the shadow empty: its
        # slots would compute nothing for the routes they take
        raise RuntimeError(f"KT_KV_HOT_LEND: no checkpoint tensor of experts {wrapper.num_gpu_experts}.."
                           f"{wrapper.num_gpu_experts + x - 1} reached layer {wrapper.kt_config.layer_idx}'s shadow")
    dev = next(p.device for p in layer.parameters() if p.is_cuda)
    torch.cuda.synchronize(dev)
    before = torch.cuda.memory_allocated(dev)
    # what the layer got after create_weights (the rest of its __init__): the
    # shadow takes its plain attributes, never its tensors
    for k, v in vars(layer).items():
        if k not in made and k not in shadow.__dict__ and not isinstance(v, torch.Tensor):
            shadow.__dict__[k] = v
    for p in shadow._parameters.values():
        if isinstance(p, torch.Tensor):
            p.data = p.data.to(dev)
    for k, b in list(shadow._buffers.items()):
        if isinstance(b, torch.Tensor):
            shadow._buffers[k] = b.to(dev)
    # the shadow's runner first: a method may read it while processing (Marlin
    # MXFP4 reads runner.config.gemm1_alpha)
    cfg = copy.copy(getattr(wrapper, "moe_runner_config", None))
    if cfg is not None and hasattr(method, "create_moe_runner"):
        cfg.num_local_experts = max(_cap(wrapper), x)
        method.create_moe_runner(shadow, cfg)
    # the method's global-experts parameters (ModelOpt NVFP4's input scales) load no row
    # into the shadow and kept their 1.0 fill, which the method's max took: the layer's
    # loaded rows instead
    try:
        from sglang.srt.layers.moe.kt_ep_wrapper import kt_fill_global_expert_params
    except ImportError:  # a tree without that patch
        kt_fill_global_expert_params = None
    if kt_fill_global_expert_params is not None:
        kt_fill_global_expert_params(shadow, src=layer)
    with torch.cuda.device(dev):
        if gguf:
            # the X experts past the resident ones, from the GGUF
            n0 = int(wrapper.num_gpu_experts)
            method.fill_experts(shadow, torch.arange(n0, n0 + x, dtype=torch.int64))
        elif hasattr(method, "process_weights_after_loading"):
            method.process_weights_after_loading(shadow)
    L = _Layer(wrapper, len(LAYERS))
    L.layer = layer
    found = _per_expert(shadow, x)
    # only the method's per-expert tensors (the layer has them with N rows): a bank of two
    # rows could match some other tensor's first dimension
    res_names = {nm for _, nm, _ in _per_expert(layer, L.n)}
    if res_names:
        found = [f for f in found if f[1] in res_names]
    if not found:
        raise RuntimeError(f"KT_KV_HOT_LEND: the shadow of layer {wrapper.kt_config.layer_idx} has no tensor "
                           f"of {x} experts")
    if any(nm == "g1_alphas" for _, nm, _ in found) and hasattr(shadow, "w13_input_scale"):
        # ModelOpt NVFP4 under CUTLASS: the method's alphas take the input scales' max
        L.nv_isc = (shadow.w13_input_scale.max().float().reshape(1).clone(),
                    shadow.w2_input_scale.max().float().reshape(1).clone())
    for kind, name, t in found:
        bank = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
        bank.copy_(t)
        if t[0].nbytes < _FIXED_BYTES:
            # a few bytes an expert (NVFP4's scale_2 and alphas): every slot on plain device
            # memory, not a 2 MiB chunk of its own under each lend
            slot = torch.zeros((L.x,) + tuple(t.shape[1:]), dtype=t.dtype, device=dev)
            L.fixed.add(name)
        else:
            slot = kt_kv_vmm.shadow_tensor((L.x,) + tuple(t.shape[1:]), t.dtype, dev.index)
        new = torch.nn.Parameter(slot, requires_grad=False) if kind == "param" else slot
        # every name the tensor goes by: a method may register its processed form
        # under a second name as an alias of the same Parameter (ModelOpt NVFP4's
        # w13_blockscale_swizzled is w13_weight_scale), and the kernel reads that one
        for store in (shadow._parameters, shadow._buffers, shadow.__dict__):
            for k in [k for k, v in store.items() if v is t]:
                store[k] = new
        L.names.append((kind, name))
        L.slots[name], L.banks[name] = slot, bank
        L.sz[name] = t[0].nbytes
    del found, t
    torch.cuda.synchronize(dev)
    torch.cuda.empty_cache()
    # what the shadow keeps on the GPU besides its slots (a method's per-layer
    # tensors that are not per-expert, a tensor still held under another name)
    left = torch.cuda.memory_allocated(dev) - before
    STATE["left"] = STATE.get("left", 0) + left
    if left > 1 << 20 and STATE.get("left_logged", 0) < 3:
        STATE["left_logged"] = STATE.get("left_logged", 0) + 1
        held = [(k, tuple(v.shape), v.nbytes) for store in (shadow._parameters, shadow._buffers, shadow.__dict__)
                for k, v in store.items() if isinstance(v, torch.Tensor) and v.is_cuda
                and kt_kv_vmm.chunk_of(v) == 0 and v.nbytes >= 1 << 16]
        _log(f"layer {L.idx}: the shadow keeps {left / 2**20:.1f} MiB on the GPU "
             f"outside its slots: {held}")
    L.shadow, L.method = shadow, method
    if _ep(L):
        # this rank's experts only (e % world == rank), in order, to start with
        own = [e for e in range(L.n, L.e) if e % method.world == method.rank]
        L.held = (own + [-1] * L.x)[: L.x]
    LAYERS.append(L)
    # the lent counts, made before any graph is captured (route reads them)
    STATE["k"] = torch.zeros(len(LAYERS), dtype=torch.int64, device=dev)
    if DYNAMIC:
        # the slot table and the route counts, also made before any capture
        STATE["tbl"] = torch.full((len(LAYERS), L.e), -1, dtype=torch.int64, device=dev)
        STATE["utbl"] = torch.full((len(LAYERS), L.e), -1, dtype=torch.int64, device=dev) if _ep(L) else STATE["tbl"]
        STATE["cnt"] = torch.zeros((len(LAYERS), L.e), dtype=torch.float32, device=dev)
    if DYN_RES:
        # the resident rows by role (None: not the packed layout the arenas give, so they
        # stay put) and their table, physical id -> row, the identity to start with
        rows = {}
        for _, name, t in _per_expert(layer, L.n):
            key = ("w13" if name.startswith("w13") else "w2" if name.startswith("w2") else None, "scale" in name)
            if key[0] is None or key in rows or (t.element_size() != 1 and t.dtype != torch.bfloat16):
                rows = None
                break
            rows[key] = t
        # two bf16 tensors are the resident layout too
        bf16 = bool(rows) and len(rows) == 2 and all(t.dtype == torch.bfloat16 for t in rows.values())
        L.res_t = rows if rows and (len(rows) == 4 or bf16 or (len(rows) == 2 and _is_gguf(L))) else None
        rtbl = torch.full((len(LAYERS), L.e), -1, dtype=torch.int64, device=dev)
        if "rtbl" in STATE:
            rtbl[: STATE["rtbl"].shape[0]] = STATE["rtbl"]
        rtbl[L.idx, : L.n] = torch.arange(L.n, device=dev)
        STATE["rtbl"] = rtbl
    wrapper.__dict__["_kt_kv_hot"] = L
    if L.idx == 0:
        _log(f"N {L.n} + X {L.x} slots ({L.b} with a bank): per-expert tensors "
             f"{[(nm, tuple(L.slots[nm].shape), L.sz[nm]) for _, nm in L.names]}, "
             f"{sum(L.sz.values()) * L.b / 2**20:.1f} MiB a layer in pinned memory, up to "
             f"{sum(L.sz.values()) * L.x / 2**20:.1f} MiB in slots")


def _k_dev(dev):
    return STATE["k"]


def _in_window() -> bool:
    try:
        from sglang.srt.layers.moe import kt_lend
    except ImportError:
        return False
    return kt_lend.ENABLED and kt_lend.phase() == "prefill"


def route(wrapper, dispatch_output):
    """The wrapper's apply, after the logical -> physical remap, before the CPU
    submit: (dispatch for the CPU, dispatch for the slots or None, the check's
    copy, the resident rows' ids or None: then the wrapper's own mask) or None."""
    L = wrapper.__dict__.get("_kt_kv_hot")
    if L is None:
        return None
    topk = dispatch_output.topk_output
    ids, w = topk.topk_ids, topk.topk_weights
    if _in_window():
        # a streamed prefill: nothing is lent, the streamer runs them, and the resident
        # rows hold 0..N-1 (res_guard); a CPU that holds every expert must skip those
        if not DYN_RES:
            return None
        res = (ids >= 0) & (ids < L.n)
        return dispatch_output._replace(topk_output=topk._replace(
            topk_ids=torch.where(res, torch.full_like(ids, -1), ids),
            topk_weights=torch.where(res, torch.zeros_like(w), w))), None, None, None
    if DYNAMIC and _route_kernel is not None and not ROUTE_TORCH and not CHECK:
        # ten-odd small kernels a layer cost 4% of decode with the resident rows' table
        # (10-21): one launch instead
        cid, cw, sid, sw, rows = _route_fused(L, ids, w)
        return (dispatch_output._replace(topk_output=topk._replace(topk_ids=cid, topk_weights=cw)),
                dispatch_output._replace(topk_output=topk._replace(topk_ids=sid, topk_weights=sw)),
                None, rows)
    rows = None
    if DYNAMIC:
        flat = ids.reshape(-1).long()
        STATE["cnt"][L.idx].index_add_(0, flat.clamp(min=0), (flat >= 0).float())
        ic = ids.long().clamp(min=0)
        slot = STATE["tbl"][L.idx][ic].to(ids.dtype)
        lent = (slot >= 0) & (ids >= 0)
        any_lent = (STATE["utbl"][L.idx][ic] >= 0) & (ids >= 0)
        if DYN_RES:
            row = STATE["rtbl"][L.idx][ic].to(ids.dtype)
            res = (row >= 0) & (ids >= 0)
            lent = lent & ~res
            rows = torch.where(res, row, torch.full_like(ids, -1))
    else:
        k = _k_dev(ids.device)[L.idx]
        lent = (ids >= L.n) & (ids < L.n + k)
        slot = ids - L.n
    if rows is not None:
        # kt-kernel holds every expert: -1 (skipped) for each route the GPU takes
        gpu = lent | (rows >= 0)
        cpu = dispatch_output._replace(topk_output=topk._replace(
            topk_ids=torch.where(gpu, torch.full_like(ids, -1), ids),
            topk_weights=torch.where(gpu, torch.zeros_like(w), w)))
    else:
        any_lent = any_lent if DYNAMIC else lent
        cpu = dispatch_output._replace(topk_output=topk._replace(
            topk_ids=torch.where(any_lent, torch.zeros_like(ids), ids),
            topk_weights=torch.where(any_lent, torch.zeros_like(w), w)))
    if _MASK_ZERO:
        # a runner that cannot take -1 (Marlin): slot 0 at weight 0, which reads
        # zeros when nothing is lent
        sid = torch.where(lent, slot, torch.zeros_like(ids)).to(torch.int64)
        sw = torch.where(lent, w, torch.zeros_like(w))
    else:
        sid = torch.where(lent, slot, torch.full_like(ids, -1))
        sw = w
    slots = dispatch_output._replace(topk_output=topk._replace(topk_ids=sid, topk_weights=sw))
    # (the check's copy; with the resident rows' table it checks the whole layer: lent None)
    saved = (ids.clone(), w.clone(), lent if rows is None else None) if CHECK else None
    return cpu, slots, saved, rows


def check(wrapper, layer, dispatch_output, hot, output) -> None:
    """KT_KV_HOT_CHECK: the slots' share of the output against the same routes on
    the CPU (the CPU with every route minus the CPU without the lent ones), and the
    whole output against the call without the lend; summed over the TP ranks
    (every rank joins: the call count is the same on each)."""
    if not CHECK or hot is None or hot[2] is None or torch.cuda.is_current_stream_capturing():
        return
    STATE["calls"] += 1
    if STATE["calls"] % CHECK:
        return
    from sglang.srt.distributed import tensor_model_parallel_all_reduce

    ids, w, lent = hot[2]
    L = wrapper.__dict__["_kt_kv_hot"]
    if lent is None:
        # KT_KV_HOT_DYNAMIC_RESIDENT: kt-kernel holds every expert, so the reference is the
        # CPU with every route; the layer's output, summed over the ranks, against it
        from sglang.srt.distributed import tensor_model_parallel_all_reduce as ar

        x = dispatch_output.hidden_states
        topk = dispatch_output.topk_output
        torch.cuda.synchronize(x.device)
        ref = torch.zeros_like(output, dtype=torch.float32)
        if wrapper.tp_rank == 0 and wrapper.wrapper is not None:
            wrapper.submit(layer, dispatch_output._replace(topk_output=topk._replace(topk_ids=ids, topk_weights=w)))
            ref = wrapper.sync(x).float().clone()
            torch.cuda.synchronize(x.device)
        ref, out = ar(ref.contiguous()), ar(output.float().contiguous())
        rel = float((out - ref).norm() / ref.norm().clamp(min=1e-30))
        on = int(((hot[3] >= 0) | (hot[1].topk_output.topk_ids >= 0)).sum()) if hot[1] is not None else 0
        STATE["checks"] += 1
        if wrapper.tp_rank == 0:
            _log(f"check {STATE['checks']}: layer {L.idx}, {x.shape[0]} rows, {on} of {ids.numel()} routes on the "
                 f"GPU; |out - all on the CPU| / |all on the CPU| = {rel:.3e}")
        return
    x = dispatch_output.hidden_states
    topk = dispatch_output.topk_output
    cpu_id = ids >= L.n
    if _MASK_ZERO:
        rid = torch.where(cpu_id, torch.zeros_like(ids), ids).to(torch.int64)
        rw = torch.where(cpu_id, torch.zeros_like(w), w)
    else:
        rid, rw = torch.where(cpu_id, torch.full_like(ids, -1), ids), w
    # every call finished and copied out before the next one starts: the CPU's result
    # comes through a host buffer the next submit writes again (10-18: rows that came
    # back wrong on whichever side was read second)
    torch.cuda.synchronize(x.device)
    res = wrapper.gpu_method.apply(layer, dispatch_output._replace(
        topk_output=topk._replace(topk_ids=rid, topk_weights=rw))).hidden_states.float().clone()
    slots = L.method.apply(L.shadow, hot[1]).hidden_states.float().clone()
    torch.cuda.synchronize(x.device)
    cpu_full = cpu_rest = torch.zeros_like(res)
    if wrapper.tp_rank == 0 and wrapper.wrapper is not None:
        wrapper.submit(layer, dispatch_output._replace(topk_output=topk._replace(topk_ids=ids, topk_weights=w)))
        cpu_full = wrapper.sync(x).float().clone()
        torch.cuda.synchronize(x.device)
        wrapper.submit(layer, hot[0])
        cpu_rest = wrapper.sync(x).float().clone()
        torch.cuda.synchronize(x.device)
        if _CHECK_REPEAT:
            # KT_KV_HOT_CHECK_REPEAT=1: both CPU calls once more, in the same order, so a
            # dump shows whether a call's result depends on what ran before it
            wrapper.submit(layer, dispatch_output._replace(topk_output=topk._replace(topk_ids=ids, topk_weights=w)))
            STATE["cpu_full2"] = wrapper.sync(x).float().clone()
            torch.cuda.synchronize(x.device)
            wrapper.submit(layer, hot[0])
            STATE["cpu_rest2"] = wrapper.sync(x).float().clone()
            torch.cuda.synchronize(x.device)
    ar = tensor_model_parallel_all_reduce
    slots, lent_cpu = ar(slots.contiguous()), ar((cpu_full - cpu_rest).contiguous())
    ref = ar((res + cpu_full).contiguous())
    diff = ar((output.float() - (res + cpu_full)).contiguous())
    n_lent = int(lent.sum())
    nr = ref.norm().clamp(min=1e-30)
    share = float(lent_cpu.norm() / nr)
    rel_slots = float((slots - lent_cpu).norm() / lent_cpu.norm().clamp(min=1e-30))
    rel = float(diff.norm() / nr)
    # the CPU returns each row's routes summed in bf16, so the lent routes' CPU side
    # (a difference of two such sums) is good to about 2^-8 of the row's CPU output:
    # where a lent route is a small part of its row, that floor is above the slots'
    # own error (10-17 Vision: 0.49 against the CPU, 5e-3 against a float32 reference)
    cpu_all = ar(cpu_full.contiguous())
    full_rows = cpu_all[lent.any(dim=1)]
    floor = float(2.0 ** -8 * full_rows.norm() / lent_cpu.norm().clamp(min=1e-30))
    STATE["checks"] += 1
    if _CHECK_DUMP and wrapper.tp_rank == 0 and n_lent and (
            STATE["checks"] <= _CHECK_DUMP_FIRST or (rel_slots > _DUMP_REL and STATE.get("dumps", 0) < 6)):
        # KT_KV_HOT_CHECK_DUMP=<dir>: the call's inputs and both sides of the lent
        # routes, for a reference computed from the checkpoint offline
        STATE["dumps"] = STATE.get("dumps", 0) + (STATE["checks"] > _CHECK_DUMP_FIRST)
        os.makedirs(_CHECK_DUMP, exist_ok=True)
        # res: the resident experts' routes on the GPU method, the same method as the
        # slots (a reference can tell a property of the method from one of the lend)
        # cpu_full / cpu_rest: the CPU's two calls (every route; the lent ones out), so a
        # reference can tell a wrong CPU row from a wrong slot
        torch.save({"layer": L.idx, "n": L.n, "x": x.cpu(), "ids": ids.cpu(), "w": w.cpu(), "lent": lent.cpu(),
                    "slots": slots.cpu(), "cpu": lent_cpu.cpu(), "res": (ref - cpu_all).cpu(),
                    "cpu_full": cpu_all.cpu(), "cpu_rest": ar(cpu_rest.contiguous()).cpu(),
                    "cpu_full_raw": cpu_full.cpu(), "cpu_rest_raw": cpu_rest.cpu(),
                    "cpu_full2": STATE.pop("cpu_full2", torch.zeros(0)).cpu(),
                    "cpu_rest2": STATE.pop("cpu_rest2", torch.zeros(0)).cpu(),
                    "cpu_ids": hot[0].topk_output.topk_ids.cpu(), "cpu_w": hot[0].topk_output.topk_weights.cpu(),
                    "rel_slots": rel_slots, "share": share},
                   os.path.join(_CHECK_DUMP, f"check{STATE['checks']}-layer{L.idx}.pt"))
    if n_lent and rel_slots > _SLOT_REL and STATE.get("slotdiag", 0) < 4:
        # KT_KV_HOT_CHECK_SLOT_REL: past this, which slot holds what its bank does not
        # (a slot read through an unmapped or stale chunk differs from its bank)
        STATE["slotdiag"] = STATE.get("slotdiag", 0) + 1
        k = int(_k_dev(x.device)[L.idx])
        # a dynamic slot holds its bank's expert only while it still holds N + j
        bad = [f"{nm}[{j}]" for _, nm in L.names for j in range(min(k, L.b)) if L.held[j] == L.n + j
               and not torch.equal(L.slots[nm][j].cpu(), L.banks[nm][j].cpu())]
        _log(f"slot diag rank {wrapper.tp_rank} layer {L.idx} (k {k}, slots vs CPU {rel_slots:.3e}): "
             f"{len(bad)} of {k * len(L.names)} slot tensors differ from their bank: {bad[:16]}")
    if n_lent and rel_slots > 0.5 and STATE.get("diag", 0) < 3:
        # the slots gave (almost) nothing for routes they hold: what do they hold?
        STATE["diag"] = STATE.get("diag", 0) + 1
        k = int(_k_dev(x.device)[L.idx])
        for kind, nm in L.names:
            cur = (L.shadow._parameters.get(nm) if kind == "param" else L.shadow._buffers.get(nm)
                   if kind == "buffer" else L.shadow.__dict__.get(nm))
            same = cur is not None and cur.data_ptr() == L.slots[nm].data_ptr()
            # on the host: in decode the driver's memory may be the lend's (a 226 MiB
            # float copy here ran 1017 out of memory, 10-17)
            sv = float(L.slots[nm][:k].cpu().view(torch.uint8).float().abs().sum()) if k else 0.0
            bv = float(L.banks[nm][:k].cpu().view(torch.uint8).float().abs().sum()) if k else 0.0
            _log(f"diag rank {wrapper.tp_rank} layer {L.idx} {kind} {nm}: shadow reads the slot {same}, "
                 f"getattr is the slot {getattr(L.shadow, nm).data_ptr() == L.slots[nm].data_ptr()}, "
                 f"slot bytes sum {sv:.4g}, bank {bv:.4g}")
        _log(f"diag rank {wrapper.tp_rank}: this rank's slot output norm "
             f"{float(L.method.apply(L.shadow, hot[1]).hidden_states.float().norm()):.4g}, "
             f"method {type(L.method).__name__}, runner {type(getattr(L.method, 'runner', None)).__name__}")
    if wrapper.tp_rank == 0:
        _log(f"check {STATE['checks']}: layer {L.idx}, {x.shape[0]} rows, {n_lent} of {lent.numel()} routes lent "
             f"(k {int(_k_dev(x.device)[L.idx])}); lent routes are {share:.3e} of |out|, slots vs CPU on them "
             f"{rel_slots:.3e} (the CPU's bf16 floor there {floor:.1e}); |out - ref| / |ref| = {rel:.3e}")


def apply(wrapper, slots_dispatch) -> torch.Tensor:
    """The slots' share of the layer's output (zero where nothing is lent)."""
    L = wrapper.__dict__["_kt_kv_hot"]
    return L.method.apply(L.shadow, slots_dispatch).hidden_states


# -- lending -------------------------------------------------------------------------

def _span(L: _Layer, name: str, k: int) -> int:
    """Bytes from a slot tensor's start that cover k slots, in whole chunks (the
    range is reserved from a chunk boundary and in whole chunks); 0 for a tensor on
    plain device memory (kv_vmm restore / release leave it alone)."""
    if name in L.fixed:
        return 0
    c = kt_kv_vmm.chunk_of(L.slots[name]) or (2 << 20)
    return -(-(k * L.sz[name]) // c) * c


def _cost(L: _Layer, k: int) -> int:
    """Bytes mapped under a layer's slots for k experts."""
    return sum(_span(L, nm, k) for _, nm in L.names) if k > 0 else 0


def _plan(budget: int) -> List[int]:
    """Slots per layer within budget bytes: one slot at a time to the layer with
    the fewest, while the next one's chunks fit."""
    ks = [0] * len(LAYERS)
    spent = 0
    while True:
        best = None
        for i, L in enumerate(LAYERS):
            if ks[i] >= L.x:
                continue
            if best is None or ks[i] < ks[best]:
                best = i
        if best is None or (SLOTS_MAX > 0 and sum(ks) >= SLOTS_MAX):
            break
        L = LAYERS[best]
        extra = _cost(L, ks[best] + 1) - _cost(L, ks[best])
        if spent + extra > budget:
            # another layer's next slot may still fit in a chunk already mapped
            cheaper = [i for i, M in enumerate(LAYERS) if ks[i] < M.x and ks[i] == ks[best]
                       and spent + _cost(M, ks[i] + 1) - _cost(M, ks[i]) <= budget]
            if not cheaper:
                break
            best = cheaper[0]
            L = LAYERS[best]
            extra = _cost(L, ks[best] + 1) - _cost(L, ks[best])
        ks[best] += 1
        spent += extra
    return ks


# -- KT_KV_HOT_DYNAMIC_SLOTS ---------------------------------------------------------

DYN = {"on": None, "order": -1, "roles": {}, "stage": None, "stream": None, "pending": None,
       "ev_unmap": None, "ev_copy": None, "ev_cnt": None, "buf": None, "scores": None, "seen": None,
       "calls": 0, "rounds": 0, "swaps": 0, "swaps_logged": 0}


def _over_ranks(vals: List[int], op=torch.distributed.ReduceOp.MIN) -> List[int]:
    """vals reduced with op over the TP ranks (the CPU group: every rank lends at the
    same forward, the free page count being the same on each)."""
    t = torch.tensor(vals, dtype=torch.int64)
    from sglang.srt.distributed.parallel_state import get_tp_group

    g = get_tp_group()
    if g.world_size > 1:
        torch.distributed.all_reduce(t, op=op, group=g.cpu_group)
    return t.tolist()


def _world() -> int:
    from sglang.srt.distributed.parallel_state import get_tp_group

    return get_tp_group().world_size


def _ep(L) -> bool:
    """The slots hold whole experts, this rank's share (kt_gguf_hot.py under KT_GGUF_HOT_EP)."""
    return bool(getattr(L.method, "ep", False))


def _union() -> None:
    """Under EP: utbl from every rank's slot table (an all-reduce, the same forward on every rank), so rank 0's
    CPU skips the routes any rank's slots take."""
    if not DYNAMIC or not LAYERS or not _ep(LAYERS[0]):
        return
    from sglang.srt.distributed.parallel_state import get_tp_group

    held = (STATE["tbl"] >= 0).to(torch.int32)
    torch.distributed.all_reduce(held, op=torch.distributed.ReduceOp.MAX, group=get_tp_group().device_group)
    STATE["utbl"].copy_(torch.where(held > 0, 0, -1))


def _is_gguf(L: _Layer) -> bool:
    """The slots' method is the GGUF hot method (kt_gguf_hot.py): raw GGUF rows, a reader for more."""
    return getattr(L.method, "reader", None) is not None and hasattr(L.method, "fill_experts")


_NV_PER_EXPERT = ("w13_weight_scale_2", "w2_weight_scale_2", "g1_alphas", "g1_alphas_up", "g2_alphas")


def _roles_nvfp4(names):
    """ModelOpt NVFP4 under the CUTLASS method: {(w13|w2, is scale): name} for the packed
    weights and their swizzled e4m3 block scales, {("nv", name): name} for the fp32 scale_2
    and alphas; None when the shadow holds anything else."""
    r = {}
    for nm in names:
        if nm in _NV_PER_EXPERT:
            r[("nv", nm)] = nm
        elif nm in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
            r[(nm[:nm.index("_")], nm.endswith("_scale"))] = nm
        else:
            return None
    need = {("w13", False), ("w13", True), ("w2", False), ("w2", True), ("nv", "g1_alphas"), ("nv", "g2_alphas")}
    return r if need <= set(r) else None


def _roles_cutile(names):
    """NVFP4 under kt_cutile_nvfp4 (ModelOpt or compressed-tensors names): {(w13|w2, is scale): name} for the
    packed weights and their row-major e4m3 block scales (the checkpoint's bytes; the swizzled copy is an alias
    of them there), {("nv", name): name} for the per-expert alphas the kernel reads (activation globals folded
    in). The weights' and activations' global slots stay as loaded: the kernel reads only the alphas. None when
    the shadow holds anything else."""
    r = {}
    for nm in names:
        proj, _, rest = nm.partition("_")
        if nm in ("kt_cutile_w13_alpha", "kt_cutile_w2_alpha"):
            r[("nv", nm)] = nm
        elif proj in ("w13", "w2") and rest in ("weight", "weight_packed", "weight_scale", "blockscale_swizzled"):
            # ModelOpt's weight / weight_scale, compressed-tensors' weight_packed / weight_scale
            key = (proj, rest not in ("weight", "weight_packed"))
            if key in r:
                return None
            r[key] = nm
        elif not (proj in ("w13", "w2") and rest in ("weight_scale_2", "weight_global_scale", "input_scale",
                                                       "input_global_scale")):
            # the globals the kernel does not read (the alphas carry them)
            return None
    need = {("w13", False), ("w13", True), ("w2", False), ("w2", True),
            ("nv", "kt_cutile_w13_alpha"), ("nv", "kt_cutile_w2_alpha")}
    return r if need <= set(r) else None


def _roles(L: _Layer):
    """{(w13|w2, is scale): slot name} when the shadow holds exactly the packed
    weights and the e8m0 scales of the two projections (bytes: uint8 or int8
    weights, uint8 or float8_e8m0fnu scales), or exactly the two projections'
    bf16 weights (the BF16 method: no scales), or ModelOpt NVFP4's tensors
    (_roles_nvfp4), else None. GGUF slots: the two weight tensors alone."""
    names = [nm for _, nm in L.names]
    if "kt_cutile_w13_alpha" in names:
        return _roles_cutile(names)
    if "g1_alphas" in names:
        return _roles_nvfp4(names)
    if len(names) == 2 and all(L.slots[nm].dtype == torch.bfloat16 for nm in names):
        r = {("w13" if nm.startswith("w13") else "w2" if nm.startswith("w2") else None, False): nm for nm in names}
        return r if set(r) == {("w13", False), ("w2", False)} else None
    r = {}
    for _, nm in L.names:
        key = ("w13" if nm.startswith("w13") else "w2" if nm.startswith("w2") else None, "scale" in nm)
        # Marlin's int32-packed weights beside e8m0 scales
        marlin_w = L.slots[nm].dtype == torch.int32 and not key[1]
        if key[0] is None or key in r or (L.slots[nm].element_size() != 1 and not marlin_w):
            return None
        r[key] = nm
    if _is_gguf(L):
        return r if len(r) == 2 and ("w13", False) in r and ("w2", False) in r else None
    return r if len(r) == 4 else None


_GGUF_PEEK = None


def _gguf_rows(L: _Layer, e: int):
    """Expert e's rows for this rank (gate, up, down) as CPU uint8 tensors: from kt-kernel's GGUF expert
    cache when resident there, else read from the file through the method's reader."""
    global _GGUF_PEEK
    m = L.method
    li, N, rank = L.layer_idx, m.N, getattr(m, "slice_rank", m.rank)
    if _GGUF_PEEK is None:
        try:
            import kt_kernel.kt_kernel_ext as _ext

            _GGUF_PEEK = (getattr(_ext, "kt_gguf_cache_peek", None), getattr(_ext, "kt_gguf_cache_unpin", None))
        except ImportError:
            _GGUF_PEEK = (None, None)
    peek, unpin = _GGUF_PEEK
    if peek is not None and m.world == 1:
        ptrs = peek(li, e)
        if ptrs is not None:
            import ctypes

            w13 = L.slots[DYN["roles"][L.idx][("w13", False)]] if L.idx in DYN.get("roles", {}) else None
            rb13 = (w13.shape[-1] if w13 is not None else L.sz[[nm for _, nm in L.names if nm.startswith("w13")][0]] // (2 * N))
            per = N * rb13
            g = torch.empty(per, dtype=torch.uint8)
            u = torch.empty(per, dtype=torch.uint8)
            w2nm = [nm for _, nm in L.names if nm.startswith("w2")][0]
            per2 = L.sz[w2nm]
            d = torch.empty(per2, dtype=torch.uint8)
            ctypes.memmove(g.data_ptr(), ptrs[0], per)
            ctypes.memmove(u.data_ptr(), ptrs[1], per)
            ctypes.memmove(d.data_ptr(), ptrs[2], per2)
            unpin(li, e)
            return g.view(N, -1), u.view(N, -1), d.view(-1, per2 // L.slots[w2nm].shape[-2] if False else d.numel() // L.slots[w2nm].shape[-1]), True
    idx = torch.tensor([e], dtype=torch.int64)
    g = m.reader.slice(li, "gate", N, rank, idx)[0]
    u = m.reader.slice(li, "up", N, rank, idx)[0]
    d = m.reader.slice(li, "down", N, rank, idx)[0]
    return g, u, d, False


def _gguf_load(L: _Layer, e: int, dst) -> None:
    if not _DYN_TIMING:
        return _gguf_load_impl(L, e, dst)
    t0 = time.perf_counter()
    _gguf_load_impl(L, e, dst)
    _DYN_T["load"] += time.perf_counter() - t0
    _DYN_T["loads"] += 1


def _gguf_load_impl(L: _Layer, e: int, dst) -> None:
    """Expert e into dst, the slot layout of the GGUF method (w13 [2N, rb]: gate rows then up rows; w2
    [K, rb2]); on the current stream (the caller's side stream for swaps)."""
    g, u, d, _ = _gguf_rows(L, e)
    N = L.method.N
    w13, w2 = dst[("w13", False)], dst[("w2", False)]
    w13[:N].copy_(g.view(N, -1), non_blocking=True)
    w13[N:].copy_(u.view(N, -1), non_blocking=True)
    w2.copy_(d.view(w2.shape), non_blocking=True)


class _AsyncGGUF:
    """A batch of GGUF swaps, their rows copied out of kt-kernel's cache into a
    pinned staging pool by background threads, then queued to their slots by the scheduler's thread."""

    pool = None  # one pinned uint8 tensor, `items` items of `item` bytes
    item = 0
    items = 0
    ex = None

    @classmethod
    def cap(cls) -> int:
        """The swaps one batch may hold (0: no GGUF layer with roles yet)."""
        if not cls.item:
            item = 0
            for M in LAYERS:
                if not _is_gguf(M) or M.idx not in DYN.get("roles", {}):
                    continue
                r = DYN["roles"][M.idx]
                item = max(item, M.slots[r[("w13", False)]][0].numel() + M.slots[r[("w2", False)]][0].numel())
            if not item:
                return 0
            cls.item = (item + 4095) // 4096 * 4096
            budget = DYN_SWAPS * DYN_EVERY * max(1, len(LAYERS))
            cls.items = DYN_ASYNC_CAP or max(1, min(budget, (DYN_ASYNC_MB << 20) // cls.item))
        return cls.items

    @classmethod
    def ready(cls, swaps) -> bool:
        if not DYN_ASYNC or not swaps:
            return False
        for L, w, _, _, _ in swaps:
            if w != 0 or not _is_gguf(L) or (getattr(L.method, "world", 1) != 1 and not _ep(L)):
                return False
        if cls.pool is None:
            from concurrent.futures import ThreadPoolExecutor

            if not cls.cap():
                return False
            n = DYN_ASYNC_THREADS or max(2, min(8, len(os.sched_getaffinity(0)) // 4))
            cls.pool = torch.empty(cls.items * cls.item, dtype=torch.uint8, pin_memory=True)
            cls.ex = ThreadPoolExecutor(max_workers=n, thread_name_prefix="kt-hot-swap")
            _log(f"dynamic GGUF swaps load in the background: {n} threads, "
                 f"{cls.items} x {cls.item / 2**20:.1f} MiB of pinned staging")
        return True

    @classmethod
    def _rows(cls, L: _Layer, e: int, k: int, pool=None, item=None) -> None:
        """Expert e's gate, up and down rows into staging item k (gate | up | down, the slot's byte order) of the
        swaps' pool, or of the pool given (a lend's fill, _gguf_fill_batch)."""
        global _GGUF_PEEK
        r = DYN["roles"][L.idx]
        w13, w2 = L.slots[r[("w13", False)]][0], L.slots[r[("w2", False)]][0]
        per, per2 = w13.numel() // 2, w2.numel()
        pool, item = (cls.pool, cls.item) if pool is None else (pool, item)
        buf = pool[k * item: k * item + 2 * per + per2]
        if _GGUF_PEEK is None:
            try:
                import kt_kernel.kt_kernel_ext as _ext

                _GGUF_PEEK = (getattr(_ext, "kt_gguf_cache_peek", None), getattr(_ext, "kt_gguf_cache_unpin", None))
            except ImportError:
                _GGUF_PEEK = (None, None)
        peek, unpin = _GGUF_PEEK
        # under EP the cache holds TP slices (kt-kernel splits each expert over its NUMA pools): read the file
        ptrs = peek(L.layer_idx, e) if peek is not None and not _ep(L) else None
        if ptrs is not None:
            import ctypes

            base = buf.data_ptr()
            ctypes.memmove(base, ptrs[0], per)
            ctypes.memmove(base + per, ptrs[1], per)
            ctypes.memmove(base + 2 * per, ptrs[2], per2)
            unpin(L.layer_idx, e)
            return
        m, N = L.method, L.method.N
        rank = getattr(m, "slice_rank", m.rank)
        idx = torch.tensor([e], dtype=torch.int64)
        buf[:per].copy_(m.reader.slice(L.layer_idx, "gate", N, rank, idx)[0].reshape(-1))
        buf[per:2 * per].copy_(m.reader.slice(L.layer_idx, "up", N, rank, idx)[0].reshape(-1))
        buf[2 * per:].copy_(m.reader.slice(L.layer_idx, "down", N, rank, idx)[0].reshape(-1))

    @classmethod
    def verify(cls, swaps) -> None:
        """Each swapped slot against the expert's rows read from the GGUF file (not through the cache)."""
        torch.cuda.synchronize()
        bad = []
        for L, _, i, _, e in swaps:
            r = DYN["roles"][L.idx]
            w13, w2 = L.slots[r[("w13", False)]][i], L.slots[r[("w2", False)]][i]
            m, N = L.method, w13.shape[0] // 2
            idx = torch.tensor([e], dtype=torch.int64)
            rank = getattr(m, "slice_rank", m.rank)
            want = (m.reader.slice(L.layer_idx, "gate", N, rank, idx)[0].reshape(-1),
                    m.reader.slice(L.layer_idx, "up", N, rank, idx)[0].reshape(-1),
                    m.reader.slice(L.layer_idx, "down", N, rank, idx)[0].reshape(-1))
            have = (w13[:N].reshape(-1).cpu(), w13[N:].reshape(-1).cpu(), w2.reshape(-1).cpu())
            diff = [int((a.view(torch.uint8) != b.view(torch.uint8)).sum()) for a, b in zip(have, want)]
            if any(diff):
                bad.append((L.layer_idx, i, e, diff))
        DYN["verified"] = DYN.get("verified", 0) + len(swaps)
        DYN["verify_bad"] = DYN.get("verify_bad", 0) + len(bad)
        if bad or DYN["verified"] <= len(swaps) or DYN["verified"] % 500 < len(swaps):
            _log(f"dynamic GGUF swaps verified: {DYN['verified']} slots, {DYN['verify_bad']} differing"
                 + (f"; this batch (layer, slot, expert, differing bytes gate/up/down): {bad[:4]}" if bad else ""))

    @classmethod
    def submit(cls, swaps):
        return [cls.ex.submit(cls._rows, L, e, k) for k, (L, _, _, _, e) in enumerate(swaps)]

    @classmethod
    def queue(cls, swaps) -> None:
        """On the current (side) stream: staging items to their slots."""
        for k, (L, _, i, _, e) in enumerate(swaps):
            r = DYN["roles"][L.idx]
            w13, w2 = L.slots[r[("w13", False)]][i], L.slots[r[("w2", False)]][i]
            per, per2 = w13.numel() // 2, w2.numel()
            buf = cls.pool[k * cls.item: k * cls.item + 2 * per + per2]
            N = w13.shape[0] // 2
            w13[:N].view(-1).copy_(buf[:per], non_blocking=True)
            w13[N:].view(-1).copy_(buf[per:2 * per], non_blocking=True)
            w2.view(-1).copy_(buf[2 * per:], non_blocking=True)


# a lend's fill of GGUF slots: threads that copy the experts' rows into pinned staging (0: half the CPUs this
# process may run on, 8 at most), and the staging's size (two halves, one filling while the other's copies run)
FILL_THREADS = int(os.environ.get("KT_KV_HOT_FILL_THREADS", "0") or 0)
FILL_STAGE_MB = int(os.environ.get("KT_KV_HOT_FILL_STAGE_MB", "256") or 256)


def _gguf_fill_batch(L: _Layer, loads) -> None:
    """A lend's (slot, expert) loads of a GGUF layer, on the current stream: the rows come out of kt-kernel's cache
    (or the file) on several threads into pinned staging, and go to the slots as non-blocking copies. One expert
    at a time through pageable memory (_gguf_load) filled 3,152 slots, 6.3 GiB, in 7.2 s: the pause at the first
    decode forward after a long prompt (11-12)."""
    r = DYN["roles"][L.idx]
    w13, w2 = L.slots[r[("w13", False)]], L.slots[r[("w2", False)]]
    n13, n2 = w13[0].numel(), w2[0].numel()
    # one staging for every layer: the largest expert among them (layers of other GGML types differ in size)
    need = max(M.slots[q[("w13", False)]][0].numel() + M.slots[q[("w2", False)]][0].numel()
               for M in LAYERS if _is_gguf(M) for q in [DYN["roles"].get(M.idx)] if q)
    need = (need + 4095) // 4096 * 4096
    st = DYN.get("gguf_fill")
    if st is None or st["item"] < need:
        from concurrent.futures import ThreadPoolExecutor

        if st is not None:
            for ev in st["ev"]:
                if ev is not None:
                    ev.synchronize()
            st["ex"].shutdown()
        n = FILL_THREADS or max(1, min(8, len(os.sched_getaffinity(0)) // 2))
        cap = max(n, (FILL_STAGE_MB << 20) // 2 // need)
        st = DYN["gguf_fill"] = {
            "item": need, "cap": cap, "turn": 0, "ev": [None, None],
            "pools": [torch.empty(cap * need, dtype=torch.uint8, pin_memory=True) for _ in range(2)],
            "ex": ThreadPoolExecutor(max_workers=n, thread_name_prefix="kt-hot-fill"),
        }
        _log(f"GGUF slots fill through {n} threads and 2 x {cap * need / 2**20:.0f} MiB of pinned staging")
    item, cap = st["item"], st["cap"]
    for a in range(0, len(loads), cap):
        part = loads[a: a + cap]
        p = st["turn"]
        st["turn"] ^= 1
        if st["ev"][p] is not None:
            st["ev"][p].synchronize()  # this half's copies of two batches ago have left it
        pool = st["pools"][p]
        for f in [st["ex"].submit(_AsyncGGUF._rows, L, e, k, pool, item) for k, (_, e) in enumerate(part)]:
            f.result()
        for k, (j, _) in enumerate(part):
            buf = pool[k * item: k * item + n13 + n2]
            w13[j].view(-1).copy_(buf[:n13], non_blocking=True)
            w2[j].view(-1).copy_(buf[n13:], non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        st["ev"][p] = ev


def _to_dev(rows, dev):
    """A small int64 table update through pinned memory: no wait for the stream (a pageable copy waits)."""
    return torch.tensor(rows, dtype=torch.int64).pin_memory().to(dev, non_blocking=True)


def _pending(swaps, dev):
    """The table entries a batch of swaps makes, split by table on the host: applying them needs no boolean mask,
    whose element count waited for the side stream's copies (10-26: most of a round's host time)."""
    out = []
    for which in (0, 1):
        rows = [(L.idx, e, i) for L, w, i, _, e in swaps if w == which]
        if rows:
            out.append((which, _to_dev(rows, dev)))
    return out


def _nv_setup(st) -> str:
    """What the NVFP4 loads need beyond the arenas, or why they cannot be made: the
    streamer's one-byte block scales of every layer, the shadows' input scales, and
    scale_2 of the up projections (the streamer reads gate and down)."""
    import json
    import sys

    from safetensors import safe_open

    from sglang.srt.layers.moe import kt_stream_prefill

    sb = getattr(st, "_sb", {})
    if any(M.layer_idx not in sb for M in LAYERS):
        return "the streamer has no one-byte block scales (KT_GPU_STREAM_SCALE_BYTES=1)"
    if any(M.nv_isc is None for M in LAYERS):
        return "a shadow has no input scales"
    return _nv_setup_up(st)


def _nv_setup_up(st) -> str:
    """DYN["nv"]: every layer's (gate, up, down) scale_2, the up ones read from the checkpoint here."""
    import json
    import sys

    from safetensors import safe_open

    from sglang.srt.layers.moe import kt_stream_prefill

    if any(M.layer_idx not in st._s2 for M in LAYERS):
        # the streamer reads its gate and down scale_2 at its first streamed prefill: a decode came first
        if not hasattr(st, "_load_scale_2"):
            return "the streamer holds no scale_2 of the layers"
        st._load_scale_2(LAYERS[0].wrapper)
        if any(M.layer_idx not in st._s2 for M in LAYERS):
            return "the streamer's scale_2 does not cover the layers"
    path = LAYERS[0].wrapper.kt_config.weight_path
    idx = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
    sample = next((k for k in idx if ".mlp.experts.0.up_proj.weight_scale_2" in k), None)
    if sample is None:
        return f"no up_proj.weight_scale_2 in {path}"
    tpl = sample.split(".layers.")[0] + ".layers.{L}.mlp.experts.{E}.up_proj.weight_scale_2"
    # the streamer's own module (the NVFP4 one is kt_stream_prefill_nvfp4)
    smod = sys.modules.get(type(st).__module__, kt_stream_prefill)
    if not hasattr(smod, "_physical_to_logical"):
        # its gate / down scale_2 would be the checkpoint's by arena index, not of the
        # expert the arena slot holds under a placement
        return "the streamer reads scale_2 by arena index"
    files, t0, nv = {}, time.time(), {}
    for M in LAYERS:
        li = M.layer_idx
        u = torch.empty(st._num_experts(li), dtype=torch.float32)
        p2l = smod._physical_to_logical(li, u.numel())
        for e in range(u.numel()):
            k = tpl.format(L=li, E=p2l[e])
            f = files.get(idx[k])
            if f is None:
                f = files[idx[k]] = safe_open(os.path.join(path, idx[k]), "pt")
            u[e] = f.get_tensor(k).float().item()
        nv[li] = (st._s2[li][0], u.to(st._s2[li][0].device), st._s2[li][1])
    DYN["nv"] = nv
    _log(f"NVFP4 slots: up scale_2 of {len(nv)} layers read in {time.time() - t0:.1f} s")
    return ""


def _arena_load_nvfp4(st, L: _Layer, e: int, dst, order: int) -> None:
    """_arena_load for ModelOpt NVFP4: the packed e2m1 weights as MXFP4's; the block
    scales from the streamer's one-byte copy ([up; gate] rows, swizzled); scale_2 from
    the checkpoint; the alphas as the method computes them (input scale x scale_2)."""
    li = L.layer_idx
    ar = st._arenas[li]
    w13, w2 = dst[("w13", False)].view(torch.uint8), dst[("w2", False)].view(torch.uint8)
    n, ws = w13.shape[0] // 2, ar[0][1]
    a, b = (0, 1) if order == 0 else (1, 0)
    nb = w13[:n].numel()
    w13[:n].view(-1).copy_(ar[a][0][e * ws:e * ws + nb], non_blocking=True)
    w13[n:].view(-1).copy_(ar[b][0][e * ws:e * ws + nb], non_blocking=True)
    w2.view(-1).copy_(ar[2][0][e * ws:e * ws + w2.numel()], non_blocking=True)
    h13, h2 = st._sb[li]
    dst[("w13", True)].view(torch.uint8).copy_(h13[e], non_blocking=True)
    dst[("w2", True)].view(torch.uint8).copy_(h2[e], non_blocking=True)
    s2g, s2u, s2d = DYN["nv"][li]
    i13, i2 = L.nv_isc
    for (kind, nm), t in dst.items():
        if kind != "nv":
            continue
        t = t.view(-1)
        if nm == "w13_weight_scale_2":
            t[:1].copy_(s2g[e:e + 1])
            if t.numel() > 1:
                t[1:2].copy_(s2u[e:e + 1])
        elif nm == "w2_weight_scale_2":
            t.copy_(s2d[e:e + 1])
        elif nm == "g1_alphas":
            torch.mul(i13, s2g[e:e + 1], out=t)
        elif nm == "g1_alphas_up":
            torch.mul(i13, s2u[e:e + 1], out=t)
        elif nm == "g2_alphas":
            torch.mul(i2, s2d[e:e + 1], out=t)


def _cutile_setup(st) -> str:
    """What the cuTile NVFP4 loads need beyond the arenas, or why they cannot be made: the block scales as the
    checkpoint's one-byte e4m3 codes, and every layer's gate, up and down scale_2 and its activation globals.
    With KT_SCALE_ARENA_CODES=1 the arenas hold the codes and the streamer (on cuTile) the tables. Without it (a
    kt-kernel that folds the scales to fp32) the codes are recovered from the arenas' fp32 scales, expert by expert
    as it is loaded (the fold is e4m3 x scale_2 rounded to bf16: dividing by scale_2 and rounding to e4m3 gives
    the checkpoint's byte back; checked against the checkpoint here), and the tables are read here."""
    from sglang.srt.layers.moe import kt_cutile_nvfp4
    from sglang.srt.layers.moe import kt_stream_prefill_nvfp4 as snv

    DYN.pop("cut", None)
    if snv.CODES:
        if not getattr(st, "cutile", False):
            return "the streamer is not on cuTile: it holds no up scale_2 or activation globals"
        missing = [M.layer_idx for M in LAYERS if M.layer_idx not in st._s2 or M.layer_idx not in st._s2u]
        if missing:
            return f"the streamer has no scale_2 of layers {missing[:4]}"
        return ""
    why = _nv_setup_up(st)  # DYN["nv"]: (gate, up, down) scale_2 of every layer
    if why:
        return why
    path = LAYERS[0].wrapper.kt_config.weight_path
    try:
        g12 = kt_cutile_nvfp4.checkpoint_layer_globals(path)
    except Exception as exc:
        return f"the checkpoint's activation globals could not be read ({type(exc).__name__}: {exc})"
    DYN["cut"] = {li: DYN["nv"][li] + tuple(g12.get(li, (1.0, 1.0))) for li in DYN["nv"]}
    bad = _cutile_recovered_check(st, path)
    if bad:
        DYN.pop("cut", None)
        return bad
    _log("cuTile slots: block scales recovered from the arenas' fp32 scales (this kt-kernel keeps no e4m3 codes); "
         "they equal the checkpoint's bytes on the experts checked")
    return ""


def _recovered_codes(st, li: int, e: int):
    """Expert e's e4m3 block-scale bytes (gate [N, K/16], up [N, K/16], down [K, N/16], this rank's part) from the
    arenas' fp32 scales."""
    from sglang.srt.layers.moe import kt_stream_prefill_nvfp4 as snv

    ar = st._arenas[li]
    cs = ar[3][1]
    N, K = st.N, st.K
    s2g, s2u, s2d = (t[e:e + 1].cpu() for t in DYN["cut"][li][:3])
    return (snv._to_e4m3(snv._host_scales(ar[3][0], e, cs, N, K // 16), s2g)[0],
            snv._to_e4m3(snv._host_scales(ar[4][0], e, cs, N, K // 16), s2u)[0],
            snv._to_e4m3(snv._host_scales(ar[5][0], e, cs, K, N // 16), s2d)[0])


def _cutile_recovered_check(st, path: str) -> str:
    """The recovered bytes against the checkpoint's, three experts of the first and the last layer; "" when equal."""
    import json

    from safetensors import safe_open

    from sglang.srt.layers.moe import kt_stream_prefill_nvfp4 as snv

    idx = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
    sample = next((k for k in idx if ".mlp.experts.0.gate_proj.weight_scale" in k and k.endswith(".weight_scale")), None)
    if sample is None:
        return f"no gate_proj.weight_scale in {path}"
    tpl = sample.split(".layers.")[0] + ".layers.{L}.mlp.experts.{E}.{P}_proj.weight_scale"
    N, K, r = st.N, st.K, st.rank
    for li in sorted({LAYERS[0].layer_idx, LAYERS[-1].layer_idx}):
        E = st._num_experts(li)
        p2l = snv._physical_to_logical(li, E)
        for e in sorted({0, E // 2, E - 1}):
            got = _recovered_codes(st, li, e)
            for proj, mine in zip(("gate", "up", "down"), got):
                k = tpl.format(L=li, E=p2l[e], P=proj)
                with safe_open(os.path.join(path, idx[k]), "pt") as f:
                    ck = f.get_tensor(k)
                ck = (ck[r * N:(r + 1) * N] if proj != "down"
                      else ck[:, r * (N // 16):(r + 1) * (N // 16)].contiguous()).view(torch.uint8)
                if ck.shape != mine.shape or bool((ck != mine).any()):
                    n = int((ck != mine).sum()) if ck.shape == mine.shape else -1
                    return (f"the block scales recovered from the arenas differ from the checkpoint's (layer {li}, "
                            f"expert {e}, {proj}: {n} of {mine.numel()} bytes)")
    return ""


def _arena_load_cutile(st, L: _Layer, e: int, dst, order: int) -> None:
    """_arena_load for ModelOpt NVFP4 under kt_cutile_nvfp4: the packed e2m1 weights and the e4m3 block-scale
    bytes as the arenas hold them (the streamer's cuTile staging, one expert), or the bytes recovered from the
    arenas' fp32 scales (_cutile_setup); the alphas as kt_cutile_nvfp4.fold_globals makes them from scale_2 (gate
    times g1, up times g1 / g2, down times g2)."""
    li = L.layer_idx
    ar = st._arenas[li]
    w13, w2 = dst[("w13", False)].view(torch.uint8), dst[("w2", False)].view(torch.uint8)
    s13, s2 = dst[("w13", True)].view(torch.uint8), dst[("w2", True)].view(torch.uint8)
    n, ws, cs = w13.shape[0] // 2, ar[0][1], ar[3][1]
    a, b = (0, 1) if order == 0 else (1, 0)
    nb, cb = w13[:n].numel(), s13[:n].numel()
    w13[:n].view(-1).copy_(ar[a][0][e * ws:e * ws + nb], non_blocking=True)
    w13[n:].view(-1).copy_(ar[b][0][e * ws:e * ws + nb], non_blocking=True)
    w2.view(-1).copy_(ar[2][0][e * ws:e * ws + w2.numel()], non_blocking=True)
    cut = DYN.get("cut")
    if cut is None:
        s13[:n].view(-1).copy_(ar[3 + a][0][e * cs:e * cs + cb], non_blocking=True)
        s13[n:].view(-1).copy_(ar[3 + b][0][e * cs:e * cs + cb], non_blocking=True)
        s2.view(-1).copy_(ar[5][0][e * cs:e * cs + s2.numel()], non_blocking=True)
        s2g, s2d = st._s2[li]
        s2u = st._s2u.get(li, s2g)
        g1, g2 = st._g12.get(li, (1.0, 1.0))
    else:
        s2g, s2u, s2d, g1, g2 = cut[li]
        # the fp32 scales go to the card as the weights do and become the bytes there: nothing waits on the host
        # (three host tensors copied one by one made a slot 1.45 ms, 7.2 s for a lend of 4,933)
        dev = s13.device
        for view, k, div in ((s13[:n], 3 + a, (s2g, s2u)[a]), (s13[n:], 3 + b, (s2g, s2u)[b]), (s2, 5, s2d)):
            raw = ar[k][0][e * cs:e * cs + 4 * view.numel()].to(dev, non_blocking=True)
            view.view(-1).copy_((raw.view(torch.float32) / div[e:e + 1]).to(torch.float8_e4m3fn).view(torch.uint8))
    t = dst[("nv", "kt_cutile_w13_alpha")].view(-1)
    t[:1].copy_(s2g[e:e + 1]).mul_(g1)
    if t.numel() > 1:
        t[1:2].copy_(s2u[e:e + 1]).mul_(g1 / g2)
    dst[("nv", "kt_cutile_w2_alpha")].view(-1).copy_(s2d[e:e + 1]).mul_(g2)


# KT_KV_HOT_CUTILE_BATCH=1: cuTile slots are filled by one kernel for all of a batch's slots, reading the registered
# expert arenas through their host addresses (the card reads them over PCIe), instead of six copies and the scale
# recovery's kernels a slot from the host: a round's 180 swaps took 30 to 50 ms of host time (11-23)
CUTILE_BATCH = os.environ.get("KT_KV_HOT_CUTILE_BATCH", "1") == "1"
CUTILE_CHUNK = 8192  # elements a program
_CB = {"ok": None, "verified": 0}


def _cutile_batch_ok(dev) -> bool:
    """The card reads registered host memory through the host pointer (CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_
    REGISTERED_MEM), and Triton is there."""
    if _CB["ok"] is None:
        ok = CUTILE_BATCH
        try:
            import ctypes

            import triton  # noqa: F401

            cu = ctypes.CDLL("libcuda.so.1")
            d, v = ctypes.c_int(), ctypes.c_int()
            index = dev.index if dev.index is not None else torch.cuda.current_device()
            ok = ok and cu.cuDeviceGet(ctypes.byref(d), index) == 0 and cu.cuDeviceGetAttribute(ctypes.byref(v), 91, d) == 0 \
                and v.value == 1
        except Exception:
            ok = False
        _CB["ok"] = bool(ok)
    return _CB["ok"]


def _cutile_layer_jobs(st, L: _Layer, js, es, order: int):
    """_cutile_jobs for all of one layer's (slot, expert) pairs at once, as numpy columns (src, dst, count, kind,
    divisor): the slots' addresses from each role tensor's base and stride, no tensor made a slot."""
    li = L.layer_idx
    ar = st._arenas[li]
    roles = DYN["roles"][L.idx]
    T = {key: L.slots[nm] for key, nm in roles.items()}
    js = np.asarray(js, np.int64)
    es = np.asarray(es, np.int64)

    def base(key):
        t = T[key]
        return t.data_ptr() + js * (t.stride(0) * t.element_size()), t[0].numel() * t.element_size()

    W, wb = base(("w13", False))
    S, sb = base(("w13", True))
    W2, w2b = base(("w2", False))
    S2, s2b = base(("w2", True))
    A13, a13b = base(("nv", "kt_cutile_w13_alpha"))
    A2, _ = base(("nv", "kt_cutile_w2_alpha"))
    nb, cb = wb // 2, sb // 2
    ws, cs = ar[0][1], ar[3][1]
    a, b = (0, 1) if order == 0 else (1, 0)
    p = [ar[k][0].data_ptr() for k in range(6)]
    cut = DYN.get("cut")
    f = np.float32
    if cut is None:
        s2g, s2d = st._s2[li]
        s2u = st._s2u.get(li, s2g)
        g1, g2 = st._g12.get(li, (1.0, 1.0))
        key = ("host_s2c", li)
    else:
        s2g, s2u, s2d, g1, g2 = cut[li]
        key = ("host_s2", li)
    hs = _CB.get(key)
    if hs is None:
        hs = _CB[key] = tuple(t.float().cpu().numpy() for t in (s2g, s2u, s2d))
    hg, hu, hd = hs
    one = np.ones_like(es, dtype=np.float32)
    cols = [  # (src, dst, count, kind, divisor)
        (p[a] + es * ws, W, nb // 4, 0, one), (p[b] + es * ws, W + nb, nb // 4, 0, one),
        (p[2] + es * ws, W2, w2b // 4, 0, one)]
    if cut is None:
        cols += [(p[3 + a] + es * cs, S, cb // 4, 0, one), (p[3 + b] + es * cs, S + cb, cb // 4, 0, one),
                 (p[5] + es * cs, S2, s2b // 4, 0, one)]
    else:
        sc = (hg, hu)
        cols += [(p[3 + a] + es * cs, S, cb, 1, sc[a][es]), (p[3 + b] + es * cs, S + cb, cb, 1, sc[b][es]),
                 (p[5] + es * cs, S2, s2b, 1, hd[es])]
    # the alphas in fp32 as _arena_load_cutile's mul_ by a Python float computes them (the scalar in fp32 first)
    zero = np.zeros_like(es)
    cols.append((zero, A13, 1, 2, hg[es].astype(f) * f(g1)))
    if a13b > 4:
        cols.append((zero, A13 + 4, 1, 2, hu[es].astype(f) * f(g1 / g2)))
    cols.append((zero, A2, 1, 2, hd[es].astype(f) * f(g2)))
    n = len(es)
    return [np.concatenate([np.broadcast_to(np.asarray(c[i], dtype=(np.float32 if i == 4 else np.int64)), (n,))
                            for c in cols]) for i in range(5)]


def _slot_copy_kernel():
    k = _CB.get("kernel")
    if k is None:
        import triton
        import triton.language as tl

        @triton.jit
        def _slot_copy(SRC, DST, CNT, KIND, DIV, BLOCK: tl.constexpr):
            i = tl.program_id(0)
            src = tl.load(SRC + i)
            dst = tl.load(DST + i)
            n = tl.load(CNT + i)
            kind = tl.load(KIND + i)
            dv = tl.load(DIV + i)
            off = tl.arange(0, BLOCK)
            if kind == 0:
                for o in range(0, n, BLOCK):
                    m = o + off < n
                    v = tl.load(src.to(tl.pointer_type(tl.int32)) + o + off, mask=m)
                    tl.store(dst.to(tl.pointer_type(tl.int32)) + o + off, v, mask=m)
            elif kind == 1:
                for o in range(0, n, BLOCK):
                    m = o + off < n
                    v = tl.load(src.to(tl.pointer_type(tl.float32)) + o + off, mask=m, other=0.0)
                    q = tl.math.div_rn(v, dv).to(tl.float8e4nv)
                    tl.store(dst.to(tl.pointer_type(tl.float8e4nv)) + o + off, q, mask=m)
            else:
                tl.store(dst.to(tl.pointer_type(tl.float32)), dv)

        k = _CB["kernel"] = _slot_copy
    return k


def _run_jobs(cols, dev) -> None:
    """The jobs (numpy columns src, dst, count, kind, divisor) on the current stream: cut into programs of
    CUTILE_CHUNK elements, one launch."""
    src, dst, cnt, kind, div = cols
    if not len(src):
        return
    C = CUTILE_CHUNK
    pieces = np.maximum(1, -(-cnt // C))
    rep = np.repeat(np.arange(len(src)), pieces)
    first = np.repeat(np.cumsum(pieces) - pieces, pieces)
    o = (np.arange(rep.size) - first) * C  # this program's first element in its job
    k = kind[rep]
    table = np.stack((src[rep] + o * 4, dst[rep] + o * np.where(k == 0, 4, 1), np.minimum(C, cnt[rep] - o), k,
                      div[rep].astype(np.float32).view(np.int32).astype(np.int64)))
    host = torch.from_numpy(np.ascontiguousarray(table).reshape(-1)).pin_memory()
    t = host.to(dev, non_blocking=True)
    R = rep.size
    _slot_copy_kernel()[(R,)](t[:R], t[R:2 * R], t[2 * R:3 * R], t[3 * R:4 * R],
                               t[4 * R:].to(torch.int32).view(torch.float32), BLOCK=1024, num_warps=4)
    _CB.setdefault("keep", []).append(host)  # the pinned table lives until its copy is done (a few rounds)
    del _CB["keep"][:-8]


def _cutile_fill_many(groups, order: int) -> list:
    """[(layer, [(slot, expert)])] into their cuTile slots by one _run_jobs; returns the groups this path did not
    take (the caller loads them one by one)."""
    from sglang.srt.layers.moe import kt_stream_prefill

    st = kt_stream_prefill._instance
    left, jobs, done, dev = [], [], [], None
    for L, loads in groups:
        roles = DYN["roles"][L.idx]
        if not loads or ("nv", "kt_cutile_w13_alpha") not in roles or st is None:
            left.append((L, loads))
            continue
        dev = L.slots[roles[("w13", False)]].device
        if not _cutile_batch_ok(dev):
            left.append((L, loads))
            continue
        st._register_arenas(L.layer_idx)
        jobs.append(_cutile_layer_jobs(st, L, [j for j, _ in loads], [e for _, e in loads], order))
        done.append((L, loads))
    if jobs:
        _run_jobs([np.concatenate([c[i] for c in jobs]) for i in range(5)], dev)
    if done and _CB["verified"] < 3 and not torch.cuda.is_current_stream_capturing():
        _CB["verified"] += 1
        L, loads = done[0]
        roles = DYN["roles"][L.idx]
        j, e = loads[0]
        got = {key: L.slots[nm][j] for key, nm in roles.items()}
        ref = {key: torch.empty_like(t) for key, t in got.items()}
        _arena_load(L, e, ref, order)
        torch.cuda.synchronize(dev)
        bad = {str(k): int((ref[k].reshape(-1).view(torch.uint8) != got[k].reshape(-1).view(torch.uint8)).sum())
               for k in ref}
        if any(bad.values()):
            raise RuntimeError(f"KT_KV_HOT_CUTILE_BATCH: layer {L.layer_idx} slot {j} (expert {e}) differs from the "
                               f"slot-by-slot load: {bad}")
        _log(f"cuTile batch fill: layer {L.layer_idx} slot {j} (expert {e}) equal to the slot-by-slot load "
             f"({sum(len(x) for _, x in done)} slots in one launch)")
    return left


def _cutile_fill(L: _Layer, loads, order: int) -> bool:
    return not _cutile_fill_many([(L, loads)], order)


def _arena_load_bf16(st, L: _Layer, e: int, dst, order: int) -> None:
    """_arena_load for the BF16 method: kt-kernel's arenas hold the plain row-major bf16
    rows ([N, K] gate and up, [K, N] down), the slot layout's own bytes: three copies."""
    st._register_arenas(L.layer_idx)
    w13, w2 = dst[("w13", False)].view(torch.uint8), dst[("w2", False)].view(torch.uint8)
    ar = st._arenas[L.layer_idx]
    n, ws = w13.shape[0] // 2, ar[0][1]
    a, b = (0, 1) if order == 0 else (1, 0)
    nb = w13[:n].numel()
    w13[:n].view(-1).copy_(ar[a][0][e * ws:e * ws + nb], non_blocking=True)
    w13[n:].view(-1).copy_(ar[b][0][e * ws:e * ws + nb], non_blocking=True)
    w2.view(-1).copy_(ar[2][0][e * ws:e * ws + w2.numel()], non_blocking=True)


def _arena_load(L: _Layer, e: int, dst, order: int) -> None:
    """Physical expert e of the layer from this rank's part of kt-kernel's arenas
    (gate, up, down packed e2m1; their fp32 group scales) into dst, the slot
    layout: w13 [2N, K/2] with gate rows first (order 0) or up rows first (1), w2
    [K, N/2], e8m0 scale bytes. On the current stream."""
    if _is_gguf(L):
        return _gguf_load(L, e, dst)
    from sglang.srt.layers.moe import kt_stream_prefill

    st = kt_stream_prefill._instance
    if ("nv", "kt_cutile_w13_alpha") in dst:
        st._register_arenas(L.layer_idx)
        return _arena_load_cutile(st, L, e, dst, order)
    if ("nv", "g1_alphas") in dst:
        st._register_arenas(L.layer_idx)
        return _arena_load_nvfp4(st, L, e, dst, order)
    if ("w13", True) not in dst:
        return _arena_load_bf16(st, L, e, dst, order)
    if dst[("w13", False)].dtype == torch.int32:
        return _arena_load_marlin(st, L, e, dst, order)
    w13, s13, w2, s2 = (dst[k].view(torch.uint8) for k in (("w13", False), ("w13", True), ("w2", False), ("w2", True)))
    if getattr(st, "_mxc", None) is not None:
        _mxc_load(st, L, e, w13, s13, w2, s2, order)
        return
    st._register_arenas(L.layer_idx)
    if not getattr(st, "_seg_whole", True):
        _arena_load_parts(st, L, e, w13, s13, w2, s2, order)
        return
    ar = st._arenas[L.layer_idx]
    n = w13.shape[0] // 2
    ws, ss = ar[0][1], ar[3][1]
    a, b = (0, 1) if order == 0 else (1, 0)
    nb = w13[:n].numel()
    w13[:n].view(-1).copy_(ar[a][0][e * ws:e * ws + nb], non_blocking=True)
    w13[n:].view(-1).copy_(ar[b][0][e * ws:e * ws + nb], non_blocking=True)
    w2.view(-1).copy_(ar[2][0][e * ws:e * ws + w2.numel()], non_blocking=True)
    if getattr(st, "_arena_e8m0", False):
        for src, out in ((ar[3 + a], s13[:n]), (ar[3 + b], s13[n:]), (ar[5], s2)):
            c = out.numel()
            out.view(-1).copy_(src[0][e * ss:e * ss + c], non_blocking=True)
    else:
        stage = DYN["stage"]
        for src, out in ((ar[3 + a], s13[:n]), (ar[3 + b], s13[n:]), (ar[5], s2)):
            c = out.numel()
            stage[:c].view(torch.uint8).copy_(src[0][e * ss:e * ss + 4 * c], non_blocking=True)
            out.view(-1).copy_(((stage[:c].view(torch.int32) >> 23) & 0xFF).to(torch.uint8))


def _mxc_unpin_done() -> None:
    """Gives the cache back the slots whose copies to the GPU are done."""
    left = []
    for moe, lid, ev in DYN.get("mxc_pins", []):
        if ev.query():
            moe.kt_cache_unpin(lid)
        else:
            left.append((moe, lid, ev))
    DYN["mxc_pins"] = left


def _mxc_load(st, L: _Layer, e: int, w13, s13, w2, s2, order: int) -> None:
    """_arena_load from kt-kernel's expert cache (KT_MXFP4_EXPERT_CACHE_GIB): the slot's gate, up, down weights and
    ue8m0 scale bytes as they are; the slot stays pinned in the cache until the copy is done. An expert the cache
    does not hold is read from the checkpoint through the streamer's bounce (and waited for)."""
    _mxc_unpin_done()
    X = st._mxc
    w = X["layers"][L.layer_idx]
    moe, lid = w.moe, w._mxc_p2l[int(e)]
    n = w13.shape[0] // 2
    a, b = (0, 1) if order == 0 else (1, 0)
    s = moe.kt_cache_pin(lid)
    if s >= 0:
        base, cache = s * X["sb"], X["cache"]
        src = [cache[base + o:base + o + c] for o, c in zip(X["off"], X["nb"])]
    else:
        src = st._mxc_read(0, [(0, lid)], w._mxc_where)[0]
    w13[:n].view(-1).copy_(src[a], non_blocking=True)
    w13[n:].view(-1).copy_(src[b], non_blocking=True)
    w2.view(-1).copy_(src[2], non_blocking=True)
    s13[:n].view(-1).copy_(src[3 + a], non_blocking=True)
    s13[n:].view(-1).copy_(src[3 + b], non_blocking=True)
    s2.view(-1).copy_(src[5], non_blocking=True)
    if s >= 0:
        ev = torch.cuda.Event()
        ev.record()
        DYN.setdefault("mxc_pins", []).append((moe, lid, ev))
    else:
        torch.cuda.current_stream().synchronize()  # the bounce is reused by the next read


def _arena_load_marlin(st, L: _Layer, e: int, dst, order: int) -> None:
    """expert e into Marlin MXFP4 slots: the arena bytes in the byte-packed layout first
    (w13 [2N, K/2], e8m0 [2N, K/32], w2 [K, N/2], e8m0 [K, N/32]), then SGLang's repack of one expert."""
    import types

    from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_mxfp4_layer_for_marlin

    layer = L.layer
    if layer is None:
        raise RuntimeError("Marlin hot slots: the layer is not bound to the hot lend")
    K = int(layer.hidden_size)
    st._register_arenas(L.layer_idx)
    if getattr(st, "_seg_whole", True) and L.layer_idx in getattr(st, "_arenas", {}):
        N = 2 * int(st._arenas[L.layer_idx][0][1]) // K  # this rank's rows: a gate part is N x K/2 bytes
    else:
        N = sum(int(seg[2]) for seg in st._segs)
    dev = dst[("w13", False)].device
    u8 = torch.uint8
    byte = {("w13", False): torch.empty(2 * N, K // 2, dtype=u8, device=dev),
            ("w13", True): torch.empty(2 * N, K // 32, dtype=u8, device=dev),
            ("w2", False): torch.empty(K, N // 2, dtype=u8, device=dev),
            ("w2", True): torch.empty(K, N // 32, dtype=u8, device=dev)}
    _arena_load(L, e, byte, order)
    one = types.SimpleNamespace(
        w13_weight=byte[("w13", False)][None], w2_weight=byte[("w2", False)][None],
        w13_weight_scale=byte[("w13", True)][None], w2_weight_scale=byte[("w2", True)][None],
        moe_runner_config=getattr(layer, "moe_runner_config", None),
        orig_dtype=getattr(layer, "orig_dtype", torch.bfloat16))
    prepare_moe_mxfp4_layer_for_marlin(one)
    for key, src in ((("w13", False), one.w13_weight), (("w13", True), one.w13_weight_scale),
                     (("w2", False), one.w2_weight), (("w2", True), one.w2_weight_scale)):
        out = dst[key]
        if tuple(src.shape[1:]) != tuple(out.shape) or src.dtype.itemsize != out.element_size():
            raise RuntimeError(f"Marlin hot slots: {key} repacks to {tuple(src.shape[1:])} {src.dtype}, "
                               f"the slot is {tuple(out.shape)} {out.dtype}")
        out.view(torch.uint8).copy_(src[0].contiguous().view(torch.uint8))


def _arena_load_parts(st, L: _Layer, e: int, w13, s13, w2, s2, order: int) -> None:
    """_arena_load when this rank's rows sit in other than one whole kt-kernel part
    (kt_stream_prefill._rank_segments): gate/up rows lo..lo+c of each part are contiguous;
    the down projection's columns of a part come through a staging buffer on the device."""
    n, k = w13.shape[0] // 2, w2.shape[0]
    a, b = (0, 1) if order == 0 else (1, 0)
    e8m0 = getattr(st, "_arena_e8m0", False)
    stage = DYN["stage"]
    for (_, lo, c, d, m), ar in zip(st._segs, st._arena_parts[L.layer_idx]):
        ws, ss = ar[0][1], ar[3][1]
        kb, kg = w13.shape[1], s13.shape[1]  # bytes and scale groups a gate/up row
        for src, row0 in ((a, 0), (b, n)):
            w13[row0 + d:row0 + d + c].view(-1).copy_(
                ar[src][0][e * ws + lo * kb:e * ws + (lo + c) * kb], non_blocking=True)
            cnt = c * kg
            if e8m0:
                s13[row0 + d:row0 + d + c].view(-1).copy_(
                    ar[3 + src][0][e * ss + lo * kg:e * ss + (lo + c) * kg], non_blocking=True)
            else:
                stage[:cnt].view(torch.uint8).copy_(
                    ar[3 + src][0][e * ss + 4 * lo * kg:e * ss + 4 * (lo + c) * kg], non_blocking=True)
                s13[row0 + d:row0 + d + c].view(-1).copy_(((stage[:cnt].view(torch.int32) >> 23) & 0xFF).to(torch.uint8))
        # down: k rows of m/2 bytes and m/32 scales in the part; columns lo..lo+c
        if DYN.get("stage_b") is None or DYN["stage_b"].numel() < k * m // 2:
            DYN["stage_b"] = torch.empty(k * m // 2, dtype=torch.uint8, device=w2.device)
        sb = DYN["stage_b"][: k * m // 2]
        sb.copy_(ar[2][0][e * ws:e * ws + k * m // 2], non_blocking=True)
        w2[:, d // 2:(d + c) // 2].copy_(sb.view(k, m // 2)[:, lo // 2:(lo + c) // 2])
        cnt = k * (m // 32)
        if e8m0:
            if DYN.get("stage_s") is None or DYN["stage_s"].numel() < cnt:
                DYN["stage_s"] = torch.empty(cnt, dtype=torch.uint8, device=w2.device)
            ss_b = DYN["stage_s"][:cnt]
            ss_b.copy_(ar[5][0][e * ss:e * ss + cnt], non_blocking=True)
            s2[:, d // 32:(d + c) // 32].copy_(ss_b.view(k, m // 32)[:, lo // 32:(lo + c) // 32])
        else:
            if stage.numel() < cnt:
                DYN["stage"] = stage = torch.empty(cnt, dtype=torch.float32, device=w2.device)
            stage[:cnt].view(torch.uint8).copy_(ar[5][0][e * ss:e * ss + 4 * cnt], non_blocking=True)
            s2[:, d // 32:(d + c) // 32].copy_(
                ((stage[:cnt].view(torch.int32).view(k, m // 32)[:, lo // 32:(lo + c) // 32] >> 23) & 0xFF).to(torch.uint8))


# KT_KV_HOT_FILL_BATCH (on by default): a lend fills a layer's slots from the MXFP4 arenas with every host copy
# issued back to back and each scale tensor's exponent bytes taken out for FILL_SLOTS slots at once. Slot by slot,
# the copies alternated with small kernels (the stage, the shift, the strided write) and the copy engine waited on
# each: MiMo-V2.6's 1,088 slots (8.7 GB) took 1.05 s of GPU time, 12 GB/s; back to back 0.40 s, 21.8 GB/s (11-09).
FILL_BATCH = os.environ.get("KT_KV_HOT_FILL_BATCH", "1") == "1"
FILL_SLOTS = int(os.environ.get("KT_KV_HOT_FILL_SLOTS", "8") or 8)
# KT_KV_HOT_FILL_VERIFY=N: the first N batched fills of each layer load their first slot again the slot-by-slot way
# into scratch and compare every byte
FILL_VERIFY = int(os.environ.get("KT_KV_HOT_FILL_VERIFY", "0") or 0)


def _batchable(L: _Layer, roles):
    """How the layer's slots fill from kt-kernel's MXFP4 arenas (FP32 scales): "bytes" (the packed bytes as they
    are, _arena_load's own path), "marlin" (repacked for Marlin, _arena_load_marlin), None (another way)."""
    if _is_gguf(L) or ("w13", True) not in roles or ("nv", "g1_alphas") in roles \
            or ("nv", "kt_cutile_w13_alpha") in roles:
        return None
    from sglang.srt.layers.moe import kt_stream_prefill

    st = kt_stream_prefill._instance
    if st is None or getattr(st, "_mxc", None) is not None:
        return None
    return "marlin" if L.slots[roles[("w13", False)]].dtype == torch.int32 else "bytes"


def _stage(name: str, numel: int, dtype, dev) -> torch.Tensor:
    t = DYN.get(name)
    if t is None or t.numel() < numel or t.device != dev:
        t = DYN[name] = torch.empty(numel, dtype=dtype, device=dev)
    return t[:numel]


def _runs(js):
    """[(first, last + 1, index in js of first)] of the consecutive slot numbers in js (sorted)."""
    out, i = [], 0
    while i < len(js):
        k = i
        while k + 1 < len(js) and js[k + 1] == js[k] + 1:
            k += 1
        out.append((js[i], js[k] + 1, i))
        i = k + 1
    return out


def _arena_fill_batch(L: _Layer, pairs, order: int, outs=None) -> None:
    """_arena_load for (slot, expert) pairs of one layer, the same bytes: every host copy first, then each scale
    tensor's e8m0 bytes taken out of its FP32 for the batch's slots at once, written in runs of consecutive slots.
    outs: (w13, s13, w2, s2) byte tensors [slots, rows, bytes] to fill instead of the layer's slots."""
    from sglang.srt.layers.moe import kt_stream_prefill

    st = kt_stream_prefill._instance
    st._register_arenas(L.layer_idx)
    roles = DYN["roles"][L.idx]
    W13, S13, W2, S2 = outs if outs is not None else (L.slots[roles[k]].view(torch.uint8) for k in
                                                      (("w13", False), ("w13", True), ("w2", False), ("w2", True)))
    dev = W13.device
    n = W13.shape[1] // 2
    kb, kg = W13.shape[2], S13.shape[2]  # bytes and scale groups a gate/up row
    K = W2.shape[1]
    a, b = (0, 1) if order == 0 else (1, 0)
    whole = getattr(st, "_seg_whole", True)
    e8m0 = getattr(st, "_arena_e8m0", False)
    segs = [(0, 0, n, 0, n)] if whole else st._segs
    parts = [st._arenas[L.layer_idx]] if whole else st._arena_parts[L.layer_idx]
    for b0 in range(0, len(pairs), FILL_SLOTS):
        batch = pairs[b0:b0 + FILL_SLOTS]
        q = len(batch)
        js = [j for j, _ in batch]
        for si, ((_, lo, c, d, m), ar) in enumerate(zip(segs, parts)):
            ws, ss = ar[0][1], ar[3][1]
            if e8m0:
                if not whole:
                    db = _stage("fbd", q * K * (m // 2), torch.uint8, dev).view(q, K, m // 2)
                    sb2 = _stage("fb2u", q * K * (m // 32), torch.uint8, dev).view(q, K, m // 32)
                for i, (j, e) in enumerate(batch):
                    for h, src in ((0, a), (1, b)):
                        W13[j, h * n + d:h * n + d + c].view(-1).copy_(
                            ar[src][0][e * ws + lo * kb:e * ws + (lo + c) * kb], non_blocking=True)
                        S13[j, h * n + d:h * n + d + c].view(-1).copy_(
                            ar[3 + src][0][e * ss + lo * kg:e * ss + (lo + c) * kg], non_blocking=True)
                    if whole:
                        W2[j].view(-1).copy_(ar[2][0][e * ws:e * ws + W2[j].numel()], non_blocking=True)
                        S2[j].view(-1).copy_(ar[5][0][e * ss:e * ss + S2[j].numel()], non_blocking=True)
                    else:
                        db[i].view(-1).copy_(ar[2][0][e * ws:e * ws + K * m // 2], non_blocking=True)
                        sb2[i].view(-1).copy_(ar[5][0][e * ss:e * ss + K * (m // 32)], non_blocking=True)
                if not whole:
                    for j0, j1, i0 in _runs(js):
                        r = slice(i0, i0 + j1 - j0)
                        W2[j0:j1, :, d // 2:(d + c) // 2].copy_(db[r, :, lo // 2:(lo + c) // 2])
                        S2[j0:j1, :, d // 32:(d + c) // 32].copy_(sb2[r, :, lo // 32:(lo + c) // 32])
                continue
            # the FP32 scales of the batch: gate/up rows lo..lo+c (two halves), then the down projection's
            f13 = _stage("fb13", q * 2 * c * kg, torch.int32, dev).view(q, 2, c, kg)
            if whole:
                f2 = _stage("fb2", q * K * S2.shape[2], torch.int32, dev).view(q, K, S2.shape[2])
            else:
                f2 = _stage("fb2", q * K * (m // 32), torch.int32, dev).view(q, K, m // 32)
                db = _stage("fbd", q * K * (m // 2), torch.uint8, dev).view(q, K, m // 2)
            for i, (j, e) in enumerate(batch):
                for h, src in ((0, a), (1, b)):
                    W13[j, h * n + d:h * n + d + c].view(-1).copy_(
                        ar[src][0][e * ws + lo * kb:e * ws + (lo + c) * kb], non_blocking=True)
                    f13[i, h].view(torch.uint8).view(-1).copy_(
                        ar[3 + src][0][e * ss + 4 * lo * kg:e * ss + 4 * (lo + c) * kg], non_blocking=True)
                if whole:
                    W2[j].view(-1).copy_(ar[2][0][e * ws:e * ws + W2[j].numel()], non_blocking=True)
                    f2[i].view(torch.uint8).view(-1).copy_(ar[5][0][e * ss:e * ss + 4 * f2[i].numel()],
                                                           non_blocking=True)
                else:
                    db[i].view(-1).copy_(ar[2][0][e * ws:e * ws + K * m // 2], non_blocking=True)
                    f2[i].view(torch.uint8).view(-1).copy_(ar[5][0][e * ss:e * ss + 4 * K * (m // 32)],
                                                           non_blocking=True)
            e13 = ((f13 >> 23) & 0xFF).to(torch.uint8)
            e2 = ((f2 >> 23) & 0xFF).to(torch.uint8)
            for j0, j1, i0 in _runs(js):
                r = slice(i0, i0 + j1 - j0)
                for h in (0, 1):
                    S13[j0:j1, h * n + d:h * n + d + c].copy_(e13[r, h])
                if whole:
                    S2[j0:j1].copy_(e2[r])
                else:
                    W2[j0:j1, :, d // 2:(d + c) // 2].copy_(db[r, :, lo // 2:(lo + c) // 2])
                    S2[j0:j1, :, d // 32:(d + c) // 32].copy_(e2[r, :, lo // 32:(lo + c) // 32])
        if outs is None and FILL_VERIFY and DYN.setdefault("fill_verified", {}).get(L.idx, 0) < FILL_VERIFY:
            DYN["fill_verified"][L.idx] = DYN["fill_verified"].get(L.idx, 0) + 1
            j, e = batch[0]
            ref = {key: torch.empty_like(L.slots[nm][j]) for key, nm in roles.items()}
            _arena_load(L, e, ref, order)
            torch.cuda.synchronize(dev)
            bad = {nm: int((ref[key].view(torch.uint8) != L.slots[nm][j].view(torch.uint8)).sum())
                   for key, nm in roles.items()}
            if any(bad.values()):
                raise RuntimeError(f"KT_KV_HOT_FILL_BATCH: layer {L.layer_idx} slot {j} (expert {e}) differs from "
                                   f"the slot-by-slot load: {bad}")
            _log(f"fill verify: layer {L.layer_idx} slot {j} (expert {e}) equal to the slot-by-slot load")


def _marlin_fill_batch(L: _Layer, pairs, order: int) -> None:
    """_arena_load_marlin for (slot, expert) pairs of one layer: FILL_SLOTS experts at a time read into byte tensors
    (_arena_fill_batch) and repacked for Marlin by one call of SGLang's repack, then written to the slots in runs."""
    import types

    from sglang.srt.layers.moe import kt_stream_prefill
    from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_mxfp4_layer_for_marlin

    st = kt_stream_prefill._instance
    layer = L.layer
    if layer is None:
        raise RuntimeError("Marlin hot slots: the layer is not bound to the hot lend")
    K = int(layer.hidden_size)
    st._register_arenas(L.layer_idx)
    if getattr(st, "_seg_whole", True) and L.layer_idx in getattr(st, "_arenas", {}):
        N = 2 * int(st._arenas[L.layer_idx][0][1]) // K
    else:
        N = sum(int(seg[2]) for seg in st._segs)
    roles = DYN["roles"][L.idx]
    dev = L.slots[roles[("w13", False)]].device
    u8 = torch.uint8
    for b0 in range(0, len(pairs), FILL_SLOTS):
        batch = pairs[b0:b0 + FILL_SLOTS]
        q = len(batch)
        outs = (_stage("mb13", q * 2 * N * (K // 2), u8, dev).view(q, 2 * N, K // 2),
                _stage("mbs13", q * 2 * N * (K // 32), u8, dev).view(q, 2 * N, K // 32),
                _stage("mb2", q * K * (N // 2), u8, dev).view(q, K, N // 2),
                _stage("mbs2", q * K * (N // 32), u8, dev).view(q, K, N // 32))
        _arena_fill_batch(L, [(i, e) for i, (_, e) in enumerate(batch)], order, outs=outs)
        one = types.SimpleNamespace(
            w13_weight=outs[0], w13_weight_scale=outs[1], w2_weight=outs[2], w2_weight_scale=outs[3],
            moe_runner_config=getattr(layer, "moe_runner_config", None),
            orig_dtype=getattr(layer, "orig_dtype", torch.bfloat16))
        prepare_moe_mxfp4_layer_for_marlin(one)
        js = [j for j, _ in batch]
        for key, src in ((("w13", False), one.w13_weight), (("w13", True), one.w13_weight_scale),
                         (("w2", False), one.w2_weight), (("w2", True), one.w2_weight_scale)):
            out = L.slots[roles[key]]
            if tuple(src.shape[1:]) != tuple(out.shape[1:]) or src.dtype.itemsize != out.element_size():
                raise RuntimeError(f"Marlin hot slots: {key} repacks to {tuple(src.shape[1:])} {src.dtype}, "
                                   f"the slot is {tuple(out.shape[1:])} {out.dtype}")
            for j0, j1, i0 in _runs(js):
                out[j0:j1].view(u8).copy_(src[i0:i0 + j1 - j0].contiguous().view(u8))
        if FILL_VERIFY and DYN.setdefault("fill_verified", {}).get(L.idx, 0) < FILL_VERIFY:
            DYN["fill_verified"][L.idx] = DYN["fill_verified"].get(L.idx, 0) + 1
            j, e = batch[-1]
            ref = {key: torch.empty_like(L.slots[nm][j]) for key, nm in roles.items()}
            _arena_load_marlin(st, L, e, ref, order)
            torch.cuda.synchronize(dev)
            bad = {nm: int((ref[key].view(u8) != L.slots[nm][j].view(u8)).sum()) for key, nm in roles.items()}
            if any(bad.values()):
                raise RuntimeError(f"KT_KV_HOT_FILL_BATCH: layer {L.layer_idx} slot {j} (expert {e}) differs from "
                                   f"the slot-by-slot Marlin load: {bad}")
            _log(f"fill verify (Marlin): layer {L.layer_idx} slot {j} (expert {e}) equal to the slot-by-slot load")


def _dyn_setup(dev) -> None:
    """First lend: whether the arenas give the bytes the slots hold. Experts N, N+1
    of the first layer loaded from the arenas against their banks (the method's
    processed form), in both row orders; on only when every rank found the same."""
    from sglang.srt.layers.moe import kt_stream_prefill

    order, why = -1, ""
    L = LAYERS[0]
    roles = _roles(L)
    st = getattr(kt_stream_prefill, "_instance", None)
    if st is None and getattr(kt_stream_prefill, "STREAM_THRESHOLD", 0) > 0 and getattr(L, "layer", None) is not None:
        # no streamed prefill yet: the streamer maps the arenas now (outside any graph)
        st = kt_stream_prefill.stream_prefill_for(L.wrapper, L.layer, kt_stream_prefill.kt_stream_threshold.get())
    if _is_gguf(L):
        if roles is None:
            why = f"slot tensors {[(nm, str(L.slots[nm].dtype)) for _, nm in L.names]} are not the GGUF hot layout"
        else:
            DYN["roles"] = {M.idx: _roles(M) for M in LAYERS}
            good = True
            for j in range(min(2, L.b)):
                dst = {key: torch.empty_like(L.slots[nm][j]) for key, nm in roles.items()}
                _gguf_load(L, L.n + j, dst)
                torch.cuda.synchronize(dev)
                for key, nm in roles.items():
                    bad = int((dst[key].reshape(-1).view(torch.uint8).cpu()
                               != L.banks[nm][j].reshape(-1).view(torch.uint8)).sum())
                    if bad:
                        good = False
                        why = f"GGUF rows of expert {L.n + j} differ from the bank ({nm}: {bad} bytes)"
            if good:
                order = 0
    elif st is None or (getattr(st, "_mxc", None) is None and L.layer_idx not in getattr(st, "_arenas", {})):
        why = "no zero-copy arenas (KT_GPU_STREAM_ZEROCOPY=1 with KT_EXPERT_SHM=1)"
    elif roles is None:
        why = (f"slot tensors {[(nm, str(L.slots[nm].dtype)) for _, nm in L.names]} are neither packed MXFP4 "
               f"bytes nor bf16 nor ModelOpt NVFP4 (CUTLASS or cuTile)")
    elif ("nv", "g1_alphas") in roles and (nv_why := _nv_setup(st)):
        why = nv_why
    elif ("nv", "kt_cutile_w13_alpha") in roles and (nv_why := _cutile_setup(st)):
        why = nv_why
    else:
        if ("w13", True) in roles and ("nv", "g1_alphas") not in roles and ("nv", "kt_cutile_w13_alpha") not in roles:
            DYN["stage"] = torch.empty(max(L.slots[roles[("w13", True)]][0].numel() // 2 + 1,
                                           L.slots[roles[("w2", True)]][0].numel()), dtype=torch.float32, device=dev)
        diff = {}
        for o in (0, 1):
            good = True
            for j in range(min(2, L.b)):
                dst = {key: torch.empty_like(L.slots[nm][j]) for key, nm in roles.items()}
                try:
                    _arena_load(L, L.n + j, dst, o)
                except RuntimeError as exc:
                    good = False
                    diff[(o, "load")] = str(exc)[:200]
                    break
                torch.cuda.synchronize(dev)
                for key, nm in roles.items():
                    bad = int((dst[key].reshape(-1).view(torch.uint8).cpu()
                               != L.banks[nm][j].reshape(-1).view(torch.uint8)).sum())
                    if bad:
                        good = False
                        diff[(o, nm)] = diff.get((o, nm), 0) + bad
            if good:
                order = o
                break
        if order < 0:
            why = f"arena bytes differ from the banks (row order, slot name: bytes): {diff}"
    lo, hi = _over_ranks([order, order], torch.distributed.ReduceOp.MIN)[0], \
        _over_ranks([order], torch.distributed.ReduceOp.MAX)[0]
    DYN["on"] = lo >= 0 and lo == hi
    if DYN["on"]:
        DYN["order"] = order
        DYN["roles"] = {M.idx: _roles(M) for M in LAYERS}
        DYN["stream"] = torch.cuda.Stream(device=dev)
        DYN["ev_unmap"], DYN["ev_copy"], DYN["ev_cnt"] = torch.cuda.Event(), torch.cuda.Event(), None
        DYN["buf"] = torch.zeros(tuple(STATE["cnt"].shape), dtype=torch.float32, pin_memory=True)
        DYN["scores"] = np.zeros(tuple(STATE["cnt"].shape))
        _log(f"dynamic slots on: arena experts match the banks ({'gate' if order == 0 else 'up'} rows first), "
             + (f"a round every {DYN_EVERY} forwards" if _EVERY_ENV else
                f"a round every {DYN_EVERY} forwards until the rounds' measures set the period")
             + f", up to {DYN_SWAPS} swaps a forward and layer, half-life {DYN_HALF_LIFE:g}")
    dyn_why = why
    if DYN_RES:
        # the resident rows swap when experts 0, 1 from the arenas match what the rows hold
        ok, why = 1, ""
        L = LAYERS[0]
        if not DYN["on"]:
            ok, why = 0, "the dynamic slots are off"
        elif _is_gguf(L):
            # the prefill lend unmaps the GGUF resident rows and writes
            # their load-time experts back (a swapped row crashed a back-to-back prefill, 10-25)
            ok, why = 0, "GGUF hot slots: the prefill lend restores the resident rows to their load-time experts"
        elif L.res_t is None:
            ok, why = 0, "the resident tensors are not the packed layout"
        else:
            for r in range(min(2, L.n)):
                dst = {key: torch.empty_like(t[r]) for key, t in L.res_t.items()}
                _arena_load(L, r, dst, order)
                torch.cuda.synchronize(dev)
                bad = {str(key): int((dst[key].view(torch.uint8) != t[r].view(torch.uint8)).sum())
                       for key, t in L.res_t.items()}
                if any(bad.values()):
                    ok, why = 0, f"arena expert {r} differs from resident row {r} (bytes by role): {bad}"
                    break
        if os.environ.get("KT_KV_HOT_DYN_RES_SWAP", "1") == "0":
            # (diagnosis: the CPU holds every expert and route sends it -1, but the rows stay)
            ok, why = 0, "KT_KV_HOT_DYN_RES_SWAP=0"
        DYN["res"] = bool(_over_ranks([ok], torch.distributed.ReduceOp.MIN)[0])
        _log("dynamic resident rows " + ("on: experts 0, 1 from the arenas match their rows" if DYN["res"]
                                          else f"off: {why or 'another rank could not'}; they hold 0..N-1"))
    # the slots shrink when the dynamic slots are off (this `else` was
    # the resident rows', which cut the slots under KT_KV_HOT_DYNAMIC_RESIDENT=0, 10-25)
    if not DYN["on"]:
        for M in LAYERS:
            # the placement's experts, as many as there are banks
            M.x = M.b
            M.held = [M.n + j for j in range(M.x)]
        _log(f"dynamic slots off: {dyn_why or 'another rank could not load from its arenas'}; "
             f"the slots hold the placement's experts")


def _dyn_fill(L: _Layer, k: int, lo=None) -> None:
    """A lend's copy into a layer's k slots: what each held before the window, from
    its bank when it is still the placement's expert, else from the arenas.
    lo (a lend that stood through a window, tick): slots 0..lo-1 stand as they are, and the slots mapped again
    take the experts the window's shrink sent out (L.back, the highest-scored last)."""
    if lo is not None:
        back = L.__dict__.get("back") or []
        for j in range(lo, k):
            L.held[j] = back.pop() if back else -1
    L.back, lo = [], lo or 0
    # an expert held twice (a slot past an earlier lend's k kept its expert while a
    # round put it in a lower slot, 10-21) takes one slot; the other is left empty
    seen = set(L.res_held) if DYN.get("res") and L.res_t is not None else set()
    for j in range(k):
        if L.held[j] in seen:
            L.held[j] = -1
        elif L.held[j] >= 0:
            seen.add(L.held[j])
    loads = []  # (slot, expert) from the arenas
    for j in range(lo, k):
        e = L.held[j]
        if (e == L.n + j or not DYN["on"]) and j < L.b:
            L.held[j] = L.n + j
            for _, nm in L.names:
                L.slots[nm][j].copy_(L.banks[nm][j], non_blocking=True)
        elif not DYN["on"]:
            L.held[j] = -1
        elif e >= 0:
            loads.append((j, e))
    way = _batchable(L, DYN["roles"][L.idx]) if loads and FILL_BATCH else None
    if loads and FILL_BATCH and _is_gguf(L):
        _gguf_fill_batch(L, loads)
    elif way == "bytes":
        _arena_fill_batch(L, loads, DYN["order"])
    elif way == "marlin":
        _marlin_fill_batch(L, loads, DYN["order"])
    elif FILL_BATCH and _cutile_fill(L, loads, DYN["order"]):
        pass
    else:
        for j, e in loads:
            _arena_load(L, e, {key: L.slots[nm][j] for key, nm in DYN["roles"][L.idx].items()}, DYN["order"])
    for j in range(k, L.x):
        L.held[j] = -1  # unmapped: what it held is gone
    if DYN.get("res") and L.res_t is not None:
        # the resident rows back to what they held before the window (res_guard put
        # 0..N-1 in them for the streamer)
        for r in range(L.n):
            if L.res_now[r] != L.res_held[r]:
                _arena_load(L, L.res_held[r], {key: t[r] for key, t in L.res_t.items()}, DYN["order"])
                L.res_now[r] = L.res_held[r]


def _dyn_table(dev) -> None:
    """The slot table from what each lent slot holds, and the resident rows' table."""
    tbl = STATE["tbl"]
    tbl.fill_(-1)
    rows = [(L.idx, L.held[j], j) for L in LAYERS for j in range(L.k) if L.held[j] >= 0]
    if rows:
        t = torch.tensor(rows, dtype=torch.int64).to(dev)
        tbl[t[:, 0], t[:, 1]] = t[:, 2]
    if DYN_RES:
        rtbl = STATE["rtbl"]
        rtbl.fill_(-1)
        t = torch.tensor([(L.idx, L.res_now[r], r) for L in LAYERS for r in range(L.n)], dtype=torch.int64).to(dev)
        rtbl[t[:, 0], t[:, 1]] = t[:, 2]


def res_guard() -> None:
    """The KT wrapper's apply, first thing: in a prefill window the resident rows hold
    experts 0..N-1 again. The prefill lend reads every layer's weights from its pinned
    mirror, made at load, and copies the mirror back when decode resumes, so no copy is
    made here (writing the rows inside a window hits the lent memory: a segfault,
    10-21); the table says so, and the next lend fills the rows again (_dyn_fill). A
    round's copies finished before the window paused the weights: its release syncs."""
    if not DYN_RES or not DYN.get("res") or not _in_window() or torch.cuda.is_current_stream_capturing():
        return
    if all(L.res_now == list(range(L.n)) for L in LAYERS):
        return
    dev = LAYERS[0].slots[LAYERS[0].names[0][1]].device
    DYN["pending"] = None
    rtbl = STATE["rtbl"]
    rtbl.fill_(-1)
    for L in LAYERS:
        L.res_now = list(range(L.n))
        rtbl[L.idx, : L.n] = torch.arange(L.n, device=dev)


# -- the period and the bar ---------------------------------------------------------
# What a round costs and what a swap wins are measured where they happen, in seconds of the decode:
#   r0  the host time of a round without its loads (counts, scores, the choice);
#   b   the host time of one swap's load (the copy from the arena);
#   g   what one route on the CPU adds to a forward: the slope of the forward time against the routes the tables
#       did not hold, over consecutive windows' differences (a slow drift, the context growing, cancels). It
#       stands once it is more than twice its standard error away from zero;
#   q   per forward, the rate the swaps wait for: for each swap half the rate it wins (the wait for a round,
#       P / 2 forwards on average) plus the new expert's rate (in no table until the next round, P forwards);
#   T   how long a place keeps an expert: the places over the swaps a forward (their turnover).
# A score is a rate x half-life / ln 2. A swap of a victim scored v for an expert scored e wins, over the T
# forwards it lasts, g x ln 2 / half-life x (e x (T - P) - v x T) and costs b:
#   swap when  e x (1 - P / T) - v > b x half-life / (g x ln 2 x T)    (the bar)
# and a period P costs r0 / P + g x q x P a forward, least at
#   P = sqrt(r0 / (g x q))                                             (between 1 and half the half-life)
# Until g, b and r0 stand on every rank (and where the loads run in the background, GGUF: no b), the period and
# new > victim x KT_KV_HOT_DYN_HYST stay as they start.
_LN2 = float(np.log(2.0))
_DRV = {"period": DYN_EVERY, "since": 0, "t_call": None, "gap_sum": 0.0, "gap_max": 0.0, "gaps": 0, "f": None,
        "win": None, "last_xy": None, "masks": [], "counted": None,
        "n": 0, "sxx": 0.0, "sxy": 0.0, "syy": 0.0, "g": None, "g_se": None,
        "r0": None, "b": None, "q": None, "s": None, "T": None, "bar": None}


def _drv_ema(key: str, v: float, period: int) -> None:
    D = _DRV
    D[key] = v if D[key] is None else D[key] + (v - D[key]) * (1.0 - 0.5 ** (period / DYN_HALF_LIFE))


def _drv_mask(new):
    """What the tables hold through the window that starts now: the held experts less the ones this round chose
    (those enter the table a round later)."""
    M = np.zeros(DYN["scores"].shape, dtype=bool)
    res = bool(DYN.get("res"))
    for L in LAYERS:
        on = [e for e in L.held[: L.k] if e >= 0] + (L.res_now if res else [])
        if on:
            M[L.idx, on] = True
    for i, e in new:
        M[i, e] = False
    return M


def _drv_counted():
    C = _DRV["counted"]
    if C is None or C.shape != DYN["scores"].shape:
        C = np.zeros(DYN["scores"].shape, dtype=bool)
        res = bool(DYN.get("res"))
        for L in LAYERS:
            C[L.idx, 0 if res else L.n:] = True
        _DRV["counted"] = C
    return C


def _drv_may(Ls, res, shape):
    """The experts each of these layers may take into a place: past the resident ones unless the resident rows
    swap too, and under EP this rank's."""
    D = _DRV
    key = (tuple(L.idx for L in Ls), tuple(res), shape)
    if D.get("may_key") != key:
        A = np.zeros(shape, dtype=bool)
        for i, (L, r) in enumerate(zip(Ls, res)):
            A[i, 0 if r else L.n:] = True
            if _ep(L):
                A[i, np.arange(shape[1]) % L.method.world != L.method.rank] = False
        D["may_key"], D["may"] = key, A
    return D["may"]


def _drv_window(x: float, y: float) -> None:
    """One window: x routes a forward the tables did not hold, y seconds a forward."""
    D = _DRV
    last, D["last_xy"] = D["last_xy"], (x, y)
    if last is None:
        return
    dx, dy = x - last[0], y - last[1]
    D["n"] += 1
    D["sxx"] += dx * dx
    D["sxy"] += dx * dy
    D["syy"] += dy * dy
    if D["n"] < 3 or D["sxx"] <= 0:
        return
    g = D["sxy"] / D["sxx"]
    se = (max(D["syy"] - g * D["sxy"], 0.0) / (D["n"] - 1) / D["sxx"]) ** 0.5
    D["g_se"] = se
    D["g"] = g if g > 2.0 * se else None


def _drv_settle(period: int) -> None:
    """The bar and the next period from the measures, where they stand on every rank. The ranks' tables must
    hold the same experts (a slot another rank lacked leaves its share of that route uncomputed) and their rounds
    fall on the same forward, so the values are the ranks' mean, as whole numbers."""
    D = _DRV
    g = D["g"]
    ok = bool(g) and D["r0"] is not None and D["b"] is not None
    P, T, bar = D["period"], 0, 0
    if ok:
        if not _EVERY_ENV:
            top = max(1, int(DYN_HALF_LIFE / 2))
            P = top if not D["q"] else int(min(top, max(1, round((D["r0"] / (g * D["q"])) ** 0.5))))
        places = sum(L.k + (L.n if DYN.get("res") and L.res_t is not None else 0) for L in LAYERS)
        T = int(min(1 << 30, max(2 * P, places / D["s"]))) if D["s"] else 1 << 30
        bar = int(D["b"] * DYN_HALF_LIFE / (g * _LN2 * T) * 1e6)
    world = _world()
    if world > 1:
        v = _over_ranks([int(ok), P, T, bar], torch.distributed.ReduceOp.SUM)
        ok, P, T, bar = v[0] == world, v[1] // world, v[2] // world, v[3] // world
    if not ok:
        D["bar"], D["T"] = None, None
        return
    D["period"], D["T"] = max(1, P), T
    D["bar"] = None if _HYST_ENV else bar / 1e6


def _drv_text() -> str:
    D = _DRV
    if D["g"] is None:
        why = "no slope yet" if D["g_se"] is None else "the forward time does not follow the CPU routes"
        return f"period {D['period']}, new > victim x {DYN_HYST:g} ({why}: {D['n']} windows)"
    bar = (f"bar {D['bar']:.1f} over {D['T']} forwards" if D["bar"] is not None else f"new > victim x {DYN_HYST:g}"
           + (" (set by hand)" if _HYST_ENV else " (not every rank's measures stand)"))
    return (f"period {D['period']}" + (" (set by hand)" if _EVERY_ENV else "") + f", {bar}; a round "
            f"{(D['r0'] or 0) * 1e3:.2f} ms, a swap {(D['b'] or 0) * 1e3:.2f} ms, a CPU route "
            f"{D['g'] * 1e6:.0f} +- {D['g_se'] * 1e6:.0f} us a forward ({D['n']} windows)")


# KT_KV_HOT_DYN_TIMING=1, the host time of the rounds
_DYN_TIMING = os.environ.get("KT_KV_HOT_DYN_TIMING", "0") == "1"
_DYN_T = {"n": 0, "swap_rounds": 0, "sum": 0.0, "max": 0.0, "swap_sum": 0.0, "load": 0.0, "loads": 0}


def _dyn_round(dev) -> None:
    if not _DYN_TIMING:
        return _dyn_round_impl(dev)
    t0 = time.perf_counter()
    swaps0 = DYN.get("swaps", 0)
    if os.environ.get("KT_KV_HOT_DYN_PROFILE", "0") == "1" and _DRV["since"] == _DRV["period"] - 1:
        import cProfile

        prof = _DYN_T.get("prof")
        if prof is None:
            prof = _DYN_T["prof"] = cProfile.Profile()
        prof.enable()
        _dyn_round_impl(dev)
        prof.disable()
    else:
        _dyn_round_impl(dev)
    if _DRV["since"]:
        return  # not a round
    dt = time.perf_counter() - t0
    T = _DYN_T
    T["n"] += 1
    T["sum"] += dt
    T["max"] = max(T["max"], dt)
    if DYN.get("swaps", 0) > swaps0:
        T["swap_rounds"] += 1
        T["swap_sum"] += dt
    if T["n"] % 64 == 0:
        if _DYN_T.get("sec"):
            _log("dynamic round sections (ms in all): " + ", ".join(f"{k} {v * 1e3:.0f}" for k, v in _DYN_T["sec"].items()))
            _DYN_T["sec"] = {}
        _log(f"dynamic round timing: {T['n']} rounds, {T['swap_rounds']} with swaps; host {T['sum'] * 1e3:.0f} ms in all "
             f"({T['sum'] / T['n'] * 1e3:.2f} ms a round, {T['swap_sum'] / max(1, T['swap_rounds']) * 1e3:.2f} ms a round "
             f"with swaps, largest {T['max'] * 1e3:.1f} ms); of it in {T['loads']} expert loads {T['load'] * 1e3:.0f} ms "
             f"({T['load'] / max(1, T['loads']) * 1e3:.2f} ms a load)")
        if T.get("prof") is not None:
            import io
            import pstats

            out = io.StringIO()
            pstats.Stats(T["prof"], stream=out).sort_stats("cumulative").print_stats(18)
            _log("dynamic round profile:\n" + out.getvalue())
            T["prof"] = None
        T.update(n=0, swap_rounds=0, sum=0.0, max=0.0, swap_sum=0.0, load=0.0, loads=0)


def _ck(name: str) -> None:
    if _DYN_TIMING:
        t = time.perf_counter()
        last = _DYN_T.get("ck_t")
        if last is not None and name != "start":
            _DYN_T.setdefault("sec", {})
            _DYN_T["sec"][name] = _DYN_T["sec"].get(name, 0.0) + t - last
        _DYN_T["ck_t"] = t


def _dyn_round_impl(dev) -> None:
    """Every period of decode forwards, outside the graph: last round's copies enter
    the table, last round's counts become scores, and each layer swaps held experts
    for better-scored ones (out of the table now, copied in on the side stream)."""
    DYN["calls"] += 1
    D = _DRV
    t_in = time.perf_counter()
    if D["t_call"] is not None:
        gap = t_in - D["t_call"]
        D["gap_sum"] += gap
        D["gap_max"] = max(D["gap_max"], gap)
        D["gaps"] += 1
    D["t_call"] = t_in
    D["since"] += 1
    if D["since"] < D["period"]:
        return
    period, D["since"] = D["since"], 0
    # this window's forward time: none when a prompt or a pause sat in it (one gap far above the others, or the
    # window far above the ones before)
    y = D["gap_sum"] / D["gaps"] if D["gaps"] else None
    if y is not None and (D["gap_max"] > 2.0 * y or (D["f"] is not None and y > 2.0 * D["f"])):
        y = None
    if y is not None:
        _drv_ema("f", y, period)
    D["gap_sum"], D["gap_max"], D["gaps"] = 0.0, 0.0, 0
    win, D["win"] = D["win"], (y, period)  # (the counts of a window come a round after its time)
    t_load = 0.0
    cur = torch.cuda.current_stream(dev)
    tbl = STATE["tbl"]
    rtbl = STATE.get("rtbl")
    _ck("start")
    if DYN["pending"] is not None:
        cur.wait_event(DYN["ev_copy"])
        if DYN_VERIFY and DYN.get("pending_swaps"):
            _AsyncGGUF.verify(DYN["pending_swaps"])
        DYN["pending_swaps"] = None
        for which, t in DYN["pending"]:  # split on the host
            table = tbl if which == 0 else rtbl
            if table is not None:  # (no resident rows' table without KT_KV_HOT_DYNAMIC_RESIDENT)
                table[t[:, 0], t[:, 1]] = t[:, 2]
        DYN["pending"] = None
    # a batch whose rows are staged goes to its slots now; its table entries the
    # round after (pending, above)
    _ck("apply pending")
    busy = False
    if DYN.get("prep") is not None:
        busy = True
        swaps_b, futs = DYN["prep"]
        if all(f.done() for f in futs):
            for f in futs:
                f.result()
            side = DYN["stream"]
            with torch.cuda.stream(side):
                side.wait_event(DYN["ev_unmap"])
                _AsyncGGUF.queue(swaps_b)
                DYN["ev_copy"].record(side)
            DYN["pending"] = _pending(swaps_b, dev)
            DYN["pending_swaps"] = swaps_b
            DYN["prep"] = None
    _ck("queue staged batch")
    if DYN["ev_cnt"] is not None:
        DYN["ev_cnt"].synchronize()
        seen = DYN["buf"].numpy().astype(np.float64)
        DYN["scores"] *= 0.5 ** ((win[1] if win else period) / DYN_HALF_LIFE)
        DYN["scores"] += seen
        DYN["seen"] = seen if DYN["seen"] is None else DYN["seen"] + seen
        DYN["last"] = seen
    else:
        seen = None
        DYN["ev_cnt"] = torch.cuda.Event()
    DYN["buf"].copy_(STATE["cnt"], non_blocking=True)
    STATE["cnt"].zero_()
    DYN["ev_cnt"].record(cur)
    DYN["rounds"] += 1
    _ck("counts and scores")
    # the window's routes against what the tables held through it (the mask made two rounds ago: one round for
    # the window to run, one for its counts to come)
    M = D["masks"].pop(0) if len(D["masks"]) >= 2 else None
    if seen is not None and M is not None and M.shape == seen.shape:
        C = _drv_counted()
        hit, every = float(seen[M & C].sum()), float(seen[C].sum())
        DYN["hit"] = DYN.get("hit", 0.0) + hit
        DYN["all"] = DYN.get("all", 0.0) + every
        if win is not None and win[0] is not None:
            _drv_window((every - hit) / win[1], win[0])
        else:
            D["last_xy"] = None
    else:
        D["last_xy"] = None
    _ck("hit accounting")
    _drv_settle(period)
    bar, tilt = D["bar"], (1.0 - D["period"] / D["T"] if D["T"] else 1.0)
    _ck("settle over the ranks")
    budget = 0 if busy else DYN_SWAPS * period  # none while a batch is out
    swaps = []  # (layer, 0: slot / 1: resident row, its index, expert out, expert in)
    moved = 0.0  # the scores the swaps move, as in q
    Ls = [L for L in LAYERS if L.k] if budget else []
    if Ls:
        # Every layer at once. A layer's places are its lent slots, and with the dynamic resident rows those rows
        # too (one pool: the lowest score anywhere goes). The places are given up in order: the empty ones as they
        # stand, then the lowest score first; the experts come in by score, the highest first; pair i swaps while
        # every pair before it did.
        S = DYN["scores"]
        res = [bool(DYN.get("res")) and L.res_t is not None for L in Ls]
        width = max(L.k + (L.n if r else 0) for L, r in zip(Ls, res))
        place = np.full((len(Ls), width), -2, dtype=np.int64)  # the expert a place holds; -1 empty, -2 no place
        for i, (L, r) in enumerate(zip(Ls, res)):
            place[i, : L.k] = L.held[: L.k]
            if r:
                place[i, L.k: L.k + L.n] = L.res_held
        Sl = S[[L.idx for L in Ls]]
        free = _drv_may(Ls, res, Sl.shape).copy()  # the experts a layer may take: not held, this rank's
        ii, jj = np.nonzero(place >= 0)
        free[ii, place[ii, jj]] = False
        cs = np.where(free, Sl, -np.inf)
        ci = np.argsort(-cs, axis=1, kind="stable")[:, :width]
        cv = np.take_along_axis(cs, ci, 1)
        vs = np.where(place >= 0, np.take_along_axis(Sl, np.maximum(place, 0), 1),
                      np.where(place == -1, -np.inf, np.inf))
        vi = np.argsort(vs, axis=1, kind="stable")
        vv = np.take_along_axis(vs, vi, 1)
        with np.errstate(invalid="ignore"):
            pays = cv > vv * DYN_HYST if bar is None else cv * tilt - vv > bar
        ok = (cv > 0) & (vv < np.inf) & ((vv == -np.inf) | pays)
        take = np.minimum(np.cumprod(ok, axis=1).sum(axis=1), budget)
        # The background loader's batch has a size (its staging). The pairs that gain the most go first, whichever
        # layer they are in: an empty slot before any other, and a layer's pairs in their own order (their gain
        # only falls), so a pair still swaps only while every pair before it did.
        pairs = [(i, t) for i in range(len(Ls)) for t in range(int(take[i]))]
        cap = _AsyncGGUF.cap() if DYN_ASYNC and any(_is_gguf(L) for L in Ls) else 0
        if cap and len(pairs) > cap:
            pairs.sort(key=lambda p: -(cv[p] - vv[p]))
        left, shut = cap, set()
        for i, t in pairs:
            L = Ls[i]
            if cap and _is_gguf(L):
                if not left:
                    shut.add(i)
                if i in shut:
                    continue
                left -= 1
            j, e = int(vi[i, t]), int(ci[i, t])
            old = int(place[i, j])
            se, so = float(cv[i, t]), (float(vv[i, t]) if old >= 0 else 0.0)
            if j < L.k:
                swaps.append((L, 0, j, old, e))
                L.held[j] = e
            else:
                swaps.append((L, 1, j - L.k, old, e))
                L.res_held[j - L.k] = e
            moved += (se - so) / 2.0 + se
    _ck("decide")
    if swaps:
        for which, table in ((0, tbl), (1, rtbl)):
            out = [(L.idx, old) for L, w, _, old, _ in swaps if w == which and old >= 0]
            if out and table is not None:
                # flat indices and index_fill_: -1 as a kernel argument (an index put of the
                # Python -1 waits for the stream)
                t = _to_dev([i * table.shape[1] + e for i, e in out], dev)
                table.view(-1).index_fill_(0, t, -1)
        DYN["ev_unmap"].record(cur)
    if swaps and _AsyncGGUF.ready(swaps):
        # the rows load in the background; a later round queues them. The last
        # batch's staging-to-slot copies must be done before the pool is written again (long done: a round ago)
        DYN["ev_copy"].synchronize()
        DYN["prep"] = (swaps, _AsyncGGUF.submit(swaps))
        DYN["swaps"] += len(swaps)
    elif swaps:
        t_load = time.perf_counter()
        side = DYN["stream"]
        with torch.cuda.stream(side):
            side.wait_event(DYN["ev_unmap"])
            rest = swaps
            if CUTILE_BATCH and all(w == 0 for _, w, _, _, _ in swaps):
                by = {}
                for L, w, i, _, e in swaps:
                    by.setdefault(L.idx, (L, []))[1].append((i, e))
                rest = [(L, 0, i, None, e) for L, loads in _cutile_fill_many(list(by.values()), DYN["order"])
                        for i, e in loads]
            for L, w, i, _, e in rest:
                if w == 0:
                    dst = {key: L.slots[nm][i] for key, nm in DYN["roles"][L.idx].items()}
                else:
                    dst = {key: t[i] for key, t in L.res_t.items()}
                    L.res_now[i] = e
                _arena_load(L, e, dst, DYN["order"])
            DYN["ev_copy"].record(side)
        DYN["pending"] = _pending(swaps, dev)
        DYN["swaps"] += len(swaps)
        t_load = time.perf_counter() - t_load
        _drv_ema("b", t_load / len(swaps), period)
    D["masks"].append(_drv_mask({(L.idx, e) for L, _, _, _, e in swaps}))
    _drv_ema("q", moved * _LN2 / DYN_HALF_LIFE / period, period)
    _drv_ema("s", len(swaps) / period, period)
    if DYN["calls"] - DYN.get("calls_logged", 0) >= DYN_LOG and DYN["seen"] is not None:
        res = bool(DYN.get("res"))
        hit, cpu = DYN.pop("hit", 0.0), DYN.pop("all", 0.0)
        # Under EP a rank's slots hold only its own experts, and the counts are every rank's routes: the share is
        # this rank's, and the ranks' shares add up to what the GPUs took.
        ep = bool(LAYERS) and _ep(LAYERS[0])
        _log(f"dynamic round {DYN['rounds']}: the GPU took {hit / max(cpu, 1):.1%} of the routes"
             + (" (resident rows and slots)" if res else " past the resident experts")
             + (f" on this rank's slots (rank {LAYERS[0].method.rank} of {LAYERS[0].method.world}: the ranks' shares add)" if ep else "")
             + f" over the last {DYN['rounds'] - DYN.get('rounds_logged', 0)} rounds, {DYN['swaps'] - DYN['swaps_logged']} swaps "
             f"({(DYN['swaps'] - DYN['swaps_logged']) / (max(DYN['calls'] - DYN.get('calls_logged', 0), 1) * max(len(LAYERS), 1)):.3f} "
             f"a forward and layer); {_drv_text()}")
        DYN["seen"], DYN["swaps_logged"], DYN["calls_logged"], DYN["rounds_logged"] = None, DYN["swaps"], DYN["calls"], DYN["rounds"]
    _union()
    _ck("unmap, submit, log")
    t_out = time.perf_counter()
    _drv_ema("r0", t_out - t_in - t_load, period)
    D["t_call"] = t_out  # (the round is not a forward's time)


def _hook() -> None:
    if STATE["hooked"]:
        return
    from sglang.srt.layers.moe import kt_kv_prefill_lend as m

    m.PRE_RESTORE.append(release)
    if hasattr(m, "PARTIAL_SHRINK"):
        m.PARTIAL_SHRINK.append(shrink)
    if hasattr(m, "WINDOW_START"):
        m.WINDOW_START.append(rows_home)
    STATE["hooked"] = True


def rows_home(why: str) -> None:
    """A prefill window opens, the weights not yet paused: the resident rows hold experts
    0..N-1 again, loaded from the arenas whole (the lend's mirror brings back only the
    tensors it lends: the scales stayed the swapped-in expert's), and the table says so."""
    if not DYN_RES or not DYN.get("res") or not LAYERS:
        return
    if all(L.res_now == list(range(L.n)) for L in LAYERS):
        return
    t0 = time.perf_counter()
    dev = LAYERS[0].slots[LAYERS[0].names[0][1]].device
    torch.cuda.synchronize(dev)  # a round's copies into the rows land first
    DYN["pending"] = None
    moved = 0
    for L in LAYERS:
        if L.res_t is None:
            continue
        for r in range(L.n):
            if L.res_now[r] != r:
                _arena_load(L, r, {key: t[r] for key, t in L.res_t.items()}, DYN["order"])
                L.res_now[r] = r
                moved += 1
    rtbl = STATE["rtbl"]
    rtbl.fill_(-1)
    for L in LAYERS:
        rtbl[L.idx, : L.n] = torch.arange(L.n, device=dev)
    torch.cuda.synchronize(dev)  # before the pause unmaps them
    if moved and DYN.get("home_logged", 0) < 3:
        DYN["home_logged"] = DYN.get("home_logged", 0) + 1
        _log(f"{why}: {moved} resident rows back to experts 0..N-1 in {(time.perf_counter() - t0) * 1e3:.1f} ms")


# KT_KV_HOT_PHASES=1: every lend and release logged with the time of each of its steps
PHASES = os.environ.get("KT_KV_HOT_PHASES", "0") == "1"


class _Phases:
    """Named steps of a lend or a release and their milliseconds, for the log line."""

    def __init__(self, t0: float):
        self.t, self.parts = t0, []

    def __call__(self, name: str) -> None:
        now = time.perf_counter()
        self.parts.append((name, (now - self.t) * 1e3))
        self.t = now

    def __str__(self) -> str:
        return ", ".join(f"{n} {ms:.0f}" for n, ms in self.parts)


# KT_KV_HOT_FILL_SPREAD=<forwards>: a fresh lend maps its slots at once and fills them over that many forwards, the
# highest-scored experts first, instead of filling them all before the forward that asked (4,456 slots: 1.4 s before
# the first token after a resume). 0: all at once.
FILL_SPREAD = int(os.environ.get("KT_KV_HOT_FILL_SPREAD", "64") or 0)


def _spread_hold(L: _Layer, k: int) -> list:
    """The slots of a fresh lend that wait: [(score, layer, slot, expert)], their held entries emptied until they
    are filled. Bank slots (a copy on the card) do not wait."""
    out = []
    S = DYN.get("scores")
    for j in range(k):
        e = L.held[j]
        if e < 0 or (e == L.n + j and j < L.b):
            continue
        out.append((float(S[L.idx, e]) if S is not None else 0.0, L.idx, j, e))
        L.held[j] = -1
    return out


def _spread_back() -> None:
    """The slots still waiting keep their experts in the held lists (a release: the next lend fills them)."""
    for _, li, j, e in STATE.pop("backlog", None) or ():
        L = LAYERS[li]
        if j < L.x and L.held[j] == -1 and e not in L.held:
            L.held[j] = e


def _spread_step(dev) -> None:
    """One forward's share of the waiting slots, on the current stream: the copies, then their table entries.
    Ranks whose slots hold the same experts fill the same rows a forward. Under EP each rank fills its own share,
    the lists differ in length, and one rank's list ends a forward before another's: the union and the end of
    the fill are then taken on the same forward by every rank (one all-reduce of the counts over the CPU group),
    or a rank waits in the union's all-reduce for a rank that has gone on to the next request."""
    bl, n = STATE["backlog"], STATE["backlog_n"]
    rows, have = [], {}
    while bl and len(rows) < n:
        _, li, j, e = bl.pop()
        L = LAYERS[li]
        h = have.get(li)
        if h is None:
            h = have[li] = set(L.held[: L.k])
        if j >= L.k or L.held[j] != -1 or e in h:
            continue  # the slot went (a shrink) or a round filled it, or put the expert elsewhere
        L.held[j] = e
        h.add(e)
        rows.append((li, e, j))
    by = {}
    for li, e, j in rows:
        by.setdefault(li, (LAYERS[li], []))[1].append((j, e))
    for L, loads in _cutile_fill_many(list(by.values()), DYN["order"]):
        for j, e in loads:
            _arena_load(L, e, {key: L.slots[nm][j] for key, nm in DYN["roles"][L.idx].items()}, DYN["order"])
    if rows:
        t = torch.tensor(rows, dtype=torch.int64).to(dev)
        STATE["tbl"][t[:, 0], t[:, 1]] = t[:, 2]
    filled, left = len(rows), len(bl)
    if _ep(LAYERS[0]) and _world() > 1:
        filled, left = _over_ranks([filled, left], torch.distributed.ReduceOp.MAX)
    if filled:
        _union()
    STATE["backlog_done"] = STATE.get("backlog_done", 0) + len(rows)
    if not left:
        STATE["backlog"] = None
        if STATE["lends"] in (1, 2, 3, 10, 100, 1000) or CHECK or PHASES:
            _log(f"lend {STATE['lends']}: {STATE.pop('backlog_done', 0)} waiting slots filled over "
                 f"{(time.perf_counter() - STATE['backlog_t']) * 1e3:.0f} ms of forwards")
        STATE.pop("backlog_done", None)


def tick(runner) -> None:
    """kt_lend.before_forward outside a prefill window (decode, and extends below
    the streaming threshold), outside any graph: lend when there is room."""
    if not ENABLED or not LAYERS or runner is None or getattr(runner, "is_draft_worker", False):
        return
    from sglang.srt.layers.moe import kt_kv_prefill_lend as m

    more = m.STATE["lent"] is not None
    # the lend stood through a prefill window (kt_lend._hot_stays): the slots the window's chunks took are mapped
    # and filled again, and the resident rows hold what they held before it
    # They come back once decode has run for as long as the last lend took to map and fill (measured, the largest
    # over the ranks' answers): a prompt followed by a few tokens and another prompt pays for neither the fill nor
    # the next window's shrink (35K + 16 tokens twice: 4,761 -> 3,539 tok/s when every window's end filled, 11-13),
    # and a long decode waits at most the time the fill itself costs.
    regrow = False
    if STATE.get("stood") and more:
        now = time.perf_counter()
        ripe = now - STATE.setdefault("stood_t", now) >= STATE.get("fill_s", 0.0)
        regrow = bool(_over_ranks([1 if ripe else 0])[0])
        if not regrow:
            if DYNAMIC and DYN["on"] and LAYERS[0].k:
                _dyn_round(LAYERS[0].slots[LAYERS[0].names[0][1]].device)
            return
    STATE.pop("stood", None)
    STATE.pop("stood_t", None)
    STATE["owed"] = 0
    if more and not regrow and STATE.get("backlog") is not None:
        # a fresh lend's slots are still being filled: no round and no growth until they are
        _spread_step(LAYERS[0].slots[LAYERS[0].names[0][1]].device)
        return
    if more and not regrow:
        if DYNAMIC and DYN["on"] and LAYERS[0].k:
            _dyn_round(LAYERS[0].slots[LAYERS[0].names[0][1]].device)
        # the lend stands: pages freed since (a request ended) join it, and slots are added for them
        if not (STATE["hooked"] and getattr(m, "more_room", None) and m.more_room(RESERVE, MIN_MB << 20)):
            return
    elif not regrow:
        if not STATE["want"]:
            STATE["since"] += 1
            if STATE["since"] < RETRY:
                return
        STATE["want"], STATE["since"] = False, 0
        _hook()
    # (a regrow waits for nothing: it fell into the wait above with the flag already taken, and the slots a window's
    # chunks took never came back, 11-10 to 11-13)
    t0 = time.perf_counter()
    ph = _Phases(t0)
    dev = LAYERS[0].slots[LAYERS[0].names[0][1]].device
    # MIN_MB is the whole lend's floor, free pages and the driver's spare together: a
    # smaller KV layout (V4.1's FP4 compressed caches halve the full pool's pages)
    # leaves fewer free-page bytes and puts the rest in the driver's free memory
    min_bytes = MIN_MB << 20
    # the slots each layer has already (a lend that stands): they stay as they are
    base = [L.k if more else 0 for L in LAYERS]
    have = sum(_cost(L, k) for L, k in zip(LAYERS, base))
    if regrow:
        from sglang.srt.layers.moe import kt_ledger

        kt_ledger.listing(dev.index if dev.index is not None else torch.cuda.current_device())
    if SPARE_FLOOR_MB >= 0 and (not more or regrow) and kt_kv_vmm.HOST is not None and kt_kv_vmm.HOST.get("go"):
        # the ledger's order: the pool's card rows no request holds (level 5) come after the hot slots. Once on the
        # card they stayed: a launch's calibration prompts left 4.44 GiB of them there after the cache was flushed,
        # and the lend that followed found 1.17 GiB, 56 slots for the 302 before (a 1M pool, 11-11). They go
        # back to host memory for as much as every slot costs; the fill below returns what the slots leave
        torch.cuda.empty_cache()
        index = dev.index if dev.index is not None else torch.cuda.current_device()
        # (after a window the lend stood through, the slots that stand are not asked for again)
        short = sum(_cost(L, L.x) for L in LAYERS) - have + (SPARE_FLOOR_MB << 20) - kt_kv_vmm.card_free(index)
        if short > 0:
            kt_kv_vmm.host_yield(index, short, why="the hot slots", held_too=False)
    if SPARE_FLOOR_MB >= 0 and not more:
        torch.cuda.empty_cache()
        spare_now = kt_kv_vmm.card_free(dev.index if dev.index is not None else torch.cuda.current_device()) \
            - (SPARE_FLOOR_MB << 20)
        min_bytes = max(1, min_bytes - max(0, spare_now))
    ph("spare")
    got = m.decode_lend_more(runner, RESERVE, min_bytes) if more else m.decode_lend(runner, RESERVE + GROW_HOLD, min_bytes)
    ph("KV pages out")
    if more and got <= 0 and not regrow:
        return
    got = max(0, got)
    if not more and _world() > 1 and not _over_ranks([1 if got > 0 else 0])[0]:
        # a rank with nothing to lend (cards that differ): none lends, so every rank takes the same steps
        if got > 0:
            m.hot_release("another rank had nothing to lend")
            # the release asks for a lend at the next forward, on this rank only: the others wait RETRY forwards,
            # and this rank's next tick met no peer in _over_ranks (TP2 hung, 11-08)
            STATE["want"], STATE["since"] = False, 0
        return
    if got <= 0 and not regrow:
        _host_fill(dev)  # nothing to lend, the pool's rows take the room
        return
    # (after a window: the slots it took left their K/V chunks unmapped, and those are the budget again)
    budget = (max(have + got, int(m.STATE["bytes"])) if regrow else have + got) - (MARGIN_MB << 20)
    spare = 0
    if SPARE_FLOOR_MB >= 0:
        # what the driver has free now counts the KV chunks just unmapped (and not the slots that stand)
        torch.cuda.empty_cache()
        free = have + kt_kv_vmm.card_free(dev.index if dev.index is not None else torch.cuda.current_device()) \
            - (SPARE_FLOOR_MB << 20)
        if free > budget:
            spare, budget = free - budget, free
    ks = [max(k, b) for k, b in zip(_plan(budget), base)]
    ph("plan")
    ep = _ep(LAYERS[0])
    # rank 0's routing decides what the CPU skips, so every rank's slots must take
    # the same routes: the slot counts are the smallest over the ranks (the spare
    # memory differs between the cards; the free pages do not). Under EP each rank's
    # slots hold its own experts and the CPU skips the union: each rank keeps its own count
    if not ep:
        ks = _over_ranks(ks)
    if DYNAMIC and DYN["on"] is None:
        _dyn_setup(dev)
        # the setup may have cut the slots to the banks
        ks = [min(k, L.x) for L, k in zip(LAYERS, ks)]
    ph("ranks and setup")
    for i, (L, k) in enumerate(zip(LAYERS, ks)):
        if k > base[i] and any(kt_kv_vmm.restore(L.slots[nm], 0, _span(L, nm, k)) < 0 for _, nm in L.names):
            # no memory after all: this layer and the rest get nothing (more than they had)
            _log(f"mapping layer {i}'s slots failed ({torch.cuda.mem_get_info(dev)[0] >> 20} MiB free): "
                 f"stopping at {sum(ks[:i]) + sum(base[i:])} slots")
            for M, b in zip(LAYERS[i:], base[i:]):
                for _, nm in M.names:
                    kt_kv_vmm.release(M.slots[nm], _span(M, nm, b), _span(M, nm, M.x) - _span(M, nm, b))
            ks = ks[:i] + base[i:]
            # the card is full now: the fill's repack and the next forward allocate beside the slots. The slots
            # mapped last go back until the floor is free again (V4.1 MXFP4 on a tree without the 1017 recipe's
            # tier 1: 820 slots mapped, 4 MiB free, and the Marlin repack of the first batch asked for 20 MiB, 11-14)
            floor = max(0, SPARE_FLOOR_MB) << 20
            index = dev.index if dev.index is not None else torch.cuda.current_device()
            back = 0
            while kt_kv_vmm.card_free(index) < floor:
                j = max(range(len(ks)), key=lambda q: (ks[q] - base[q], q))
                if ks[j] <= base[j]:
                    break
                M = LAYERS[j]
                for _, nm in M.names:
                    kt_kv_vmm.release(M.slots[nm], _span(M, nm, ks[j] - 1), _span(M, nm, ks[j]) - _span(M, nm, ks[j] - 1))
                ks[j] -= 1
                back += 1
            _log(f"the plan was {budget / 2**30:.2f} GiB ({got / 2**30:.2f} GiB of free pages, {spare / 2**30:.2f} GiB "
                 f"the driver had free); {back} slots more given back for the floor of {floor >> 20} MiB: {sum(ks)} "
                 f"slots, {kt_kv_vmm.card_free(index) >> 20} MiB free")
            break
    ph("slots mapped")
    # a rank that could not map them all: the others give back what it lacks
    agreed = ks if ep else _over_ranks(ks)
    for L, k, a in zip(LAYERS, ks, agreed):
        if a < k:
            for _, nm in L.names:
                kt_kv_vmm.release(L.slots[nm], _span(L, nm, a), _span(L, nm, k) - _span(L, nm, a))
    ks = agreed
    ph("ranks agreed")
    if PHASES:
        torch.cuda.synchronize(dev)
        ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        ev0.record()
    if regrow and DYNAMIC and DYN["on"]:
        _dyn_settle(dev)
    later, spread = [], False
    for L, k, b in zip(LAYERS, ks, base):
        if regrow and DYNAMIC and DYN["on"]:
            _dyn_fill(L, k, lo=b)
        elif more and DYNAMIC and DYN["on"]:
            pass  # the new slots are empty (held -1): the rounds fill them, as they fill a slot left empty
        elif more and DYNAMIC:
            for j in range(b, k):
                L.held[j] = L.n + j if j < L.b else -1
                if j < L.b:
                    for _, nm in L.names:
                        L.slots[nm][j].copy_(L.banks[nm][j], non_blocking=True)
        elif k and DYNAMIC:
            if FILL_SPREAD and DYN["on"]:
                later += _spread_hold(L, k)
                spread = True
            _dyn_fill(L, k)
        elif k > b:
            for _, nm in L.names:
                L.slots[nm][b:k].copy_(L.banks[nm][b:k], non_blocking=True)
        L.k = k
    if spread:  # also with nothing waiting on this rank: under EP another rank's share may wait
        later.sort()  # the highest score last: _spread_step pops from the end
        STATE["backlog"], STATE["backlog_t"] = later, time.perf_counter()
        STATE["backlog_n"] = -(-len(later) // FILL_SPREAD)
    if PHASES:
        ev1.record()
    ph("fill issued")
    if DYNAMIC and (regrow or not (more and DYN["on"])):
        # (a lend that stands under the rounds: the table holds what the slots hold, the copies in flight
        # enter it when they land)
        _dyn_table(dev)
        _union()
    ph("tables")
    kd = _k_dev(dev)
    kd.copy_(torch.tensor(ks, dtype=torch.int64), non_blocking=True)
    torch.cuda.synchronize(dev)
    ph("copies done")
    if PHASES:
        ph.parts.append(("fill on the GPU", ev0.elapsed_time(ev1)))
    STATE["lent_slots"] = sum(ks)
    # what the slots hold beyond the KV pages: back at the window's start, so the
    # prefill plan counts it as free
    STATE["spare_held"] = max(0, sum(_cost(L, k) for L, k in zip(LAYERS, ks)) - int(m.STATE["bytes"]))
    _host_fill(dev)  # the pool's rows past those in flight take what the slots left
    ph("host rows")
    if regrow or not more:
        STATE["fill_s"] = time.perf_counter() - t0  # (a grow adds a few slots: not the measure of a fill)
    if regrow:
        STATE["regrows"] = STATE.get("regrows", 0) + 1
        if STATE["regrows"] in (1, 2, 3, 10, 100, 1000) or CHECK or PHASES:
            _log(f"lend {STATE['lends']} after a window it stood through ({STATE['regrows']}): "
                 f"{sum(ks) - sum(base)} slots mapped and filled again beside {sum(base)} in "
                 f"{(time.perf_counter() - t0) * 1e3:.1f} ms" + (f" [{ph}]" if PHASES else ""))
        return
    if more:
        STATE["grows"] = STATE.get("grows", 0) + 1
        if STATE["grows"] in (1, 2, 3, 10, 100, 1000) or CHECK or PHASES:
            _log(f"lend {STATE['lends']} grows ({STATE['grows']}): {got / 2**30:.2f} GiB more of free pages unmapped, "
                 f"{sum(ks) - sum(base)} slots added to {sum(base)} in {(time.perf_counter() - t0) * 1e3:.1f} ms"
                 + (f" [{ph}]" if PHASES else ""))
        return
    STATE["lends"] += 1
    if STATE["lends"] in (1, 2, 10, 100, 1000) or CHECK or PHASES:
        _log(f"lend {STATE['lends']}: {got / 2**30:.2f} GiB of free pages unmapped"
             + (f" + {spare / 2**30:.2f} GiB the driver had free" if spare else "") + f", {sum(ks)} slots "
             f"({min(ks)}-{max(ks)} a layer of {LAYERS[0].x}) in {(time.perf_counter() - t0) * 1e3:.1f} ms"
             + (f"; the shadows keep {STATE.get('left', 0) / 2**20:.0f} MiB outside their slots"
                if STATE["lends"] == 1 else "") + (f" [{ph}]" if PHASES else ""))


def _host_fill(dev) -> None:
    """level 5 after the hot slots, the hot lend's floor kept free."""
    from sglang.srt.layers.moe import kt_kv_vmm

    if kt_kv_vmm.HOST is not None and kt_kv_vmm.HOST.get("go"):
        index = dev.index if dev.index is not None else torch.cuda.current_device()
        kt_kv_vmm.host_fill(index, max(0, SPARE_FLOOR_MB) << 20)


def _dyn_settle(dev) -> None:
    """The rounds' swaps in flight land and enter the tables now: after it every slot holds what its held entry
    says, and the table says the same."""
    if DYN.get("prep") is not None:
        swaps_b, futs = DYN["prep"]
        for f in futs:
            f.result()
        side = DYN["stream"]
        with torch.cuda.stream(side):
            side.wait_event(DYN["ev_unmap"])
            _AsyncGGUF.queue(swaps_b)
            DYN["ev_copy"].record(side)
        DYN["pending"] = _pending(swaps_b, dev)
        DYN["prep"] = None
    if DYN["pending"] is not None:
        torch.cuda.current_stream(dev).wait_event(DYN["ev_copy"])
        for which, t in DYN["pending"]:
            table = STATE["tbl"] if which == 0 else STATE.get("rtbl")
            if table is not None:
                table[t[:, 0], t[:, 1]] = t[:, 2]
        DYN["pending"] = None
    DYN["pending_swaps"] = None


def settle() -> None:
    """kt_lend._hot_stays, before a window the lend stands through: no swap is in flight past this, the tables say
    what the slots hold (rows_home and res_guard drop a round's pending table entries, which a release made moot)."""
    if not LAYERS or not (DYNAMIC and DYN["on"]):
        return
    dev = LAYERS[0].slots[LAYERS[0].names[0][1]].device
    _dyn_settle(dev)
    torch.cuda.synchronize(dev)


def standing_bytes() -> int:
    """What the lent slots hold of the card: what a shrink of all of them frees."""
    return sum(_cost(L, L.k) for L in LAYERS)


def owed_bytes() -> int:
    """What the slots a shrink sent out to come back (keep) took with them: the next tick maps them again, and the
    card's free memory they wait for is not the pool's rows' to fill before it (kt_kv_vmm.host_room)."""
    return int(STATE.get("owed", 0) or 0) if STATE.get("stood") else 0


def slots_for(need: int) -> int:
    """How many slots go (shrink's order: one at a time from the layer with the most) before their chunks free need
    bytes; every lent slot when all of them free less. The take-back's unit: a slot is the same size all through a
    model, and the ranks agree on one count where bytes measured on each card differed."""
    if not LAYERS or need <= 0:
        return 0
    ks, freed, n = [L.k for L in LAYERS], 0, 0
    while freed < need:
        i = max(range(len(ks)), key=lambda q: (ks[q], q))
        if ks[i] == 0:
            break
        freed += _cost(LAYERS[i], ks[i]) - _cost(LAYERS[i], ks[i] - 1)
        ks[i] -= 1
        n += 1
    return n


def shrink(need: int, why: str, exact: bool = False, slots=None, keep: bool = False) -> int:
    """kt_kv_prefill_lend._take_back, before it maps need bytes of K/V chunks back: slots worth as many bytes give
    their memory back, one at a time from the layer with the most. Under the rounds the experts that go are each
    layer's lowest-scored (an empty slot first), and an expert that stays moves from a slot that goes to one that
    opened. Every rank takes the same slots: the bytes, the slot counts and the scores are the same on each (under
    EP each rank's are its own, and the union table is made again). Returns the bytes freed.
    exact (kt_kv_vmm.host_in_flight: the pool's rows coming to the card, no K/V chunk mapped back): need is what
    the card is short of, already less what the driver has free, and the slots give all of it.
    slots (kt_kv_vmm.host_in_flight): that many slots go, the count the ranks agreed on (slots_for), need unused.
    keep (kt_lend, a window's chunk): each layer remembers the experts that go, for the slots mapped again after the
    window (tick)."""
    if slots is not None:
        exact, need = True, 0
        if slots <= 0:
            return 0
    if not LAYERS or not STATE["lent_slots"] or (need <= 0 and slots is None):
        return 0
    from sglang.srt.layers.moe import kt_kv_prefill_lend as m

    t0 = time.perf_counter()
    dev = LAYERS[0].slots[LAYERS[0].names[0][1]].device
    old = [L.k for L in LAYERS]
    ks, freed = list(old), 0
    want = 0 if exact else need
    if SPARE_FLOOR_MB >= 0 and not exact:
        # the chunks map into what the driver has free past the floor first; the slots give only the rest (the
        # smallest room over the ranks: every rank takes the same slots out)
        room = max(0, kt_kv_vmm.card_free(dev.index if dev.index is not None else torch.cuda.current_device())
                   - (SPARE_FLOOR_MB << 20))
        if _world() > 1:
            room = _over_ranks([room])[0]
        need = max(0, need - room)
        if need == 0:
            STATE["spare_held"] = max(0, sum(_cost(L, k) for L, k in zip(LAYERS, ks))
                                      - max(0, int(m.STATE["bytes"]) - want))
            STATE["shrinks_spared"] = STATE.get("shrinks_spared", 0) + 1
            if STATE["shrinks_spared"] in (1, 2, 3, 10, 100, 1000, 10000) or CHECK or PHASES:
                _log(f"take-back {STATE['shrinks_spared']} with no slot out ({why}): {want / 2**20:.0f} MiB of K/V "
                     f"map into the {room / 2**20:.0f} MiB the driver has free past the floor")
            return 0
    while (sum(old) - sum(ks) < slots) if slots is not None else (freed < need):
        i = max(range(len(ks)), key=lambda q: (ks[q], q))
        if ks[i] == 0:
            break
        freed += _cost(LAYERS[i], ks[i]) - _cost(LAYERS[i], ks[i] - 1)
        ks[i] -= 1
    if DYNAMIC and DYN["on"]:
        _dyn_settle(dev)
        tbl, sc = STATE["tbl"], DYN["scores"]
        out, moved = [], []
        for L, k in zip(LAYERS, ks):
            if k >= L.k:
                continue
            held = L.held
            order = sorted(range(L.k), key=lambda j: (held[j] >= 0, float(sc[L.idx][held[j]]) if held[j] >= 0 else 0.0, -j))
            drop = set(order[: L.k - k])
            if keep:  # the highest-scored last, after what an earlier chunk of the window sent out (lower still)
                L.back = (L.__dict__.get("back") or []) + [held[j] for j in order[: L.k - k] if held[j] >= 0]
            out += [L.idx * tbl.shape[1] + held[j] for j in drop if held[j] >= 0]
            holes = [j for j in range(k) if j in drop]
            stay = [j for j in range(k, L.k) if j not in drop]
            for h, t in zip(holes, stay):
                for _, nm in L.names:
                    L.slots[nm][h].copy_(L.slots[nm][t])
                held[h] = held[t]
                moved.append((L.idx, held[h], h))
            for j in range(k, L.x):
                held[j] = -1
        if out:
            tbl.view(-1).index_fill_(0, _to_dev(out, dev), -1)
        if moved:
            t = _to_dev(moved, dev)
            tbl[t[:, 0], t[:, 1]] = t[:, 2]
    elif DYNAMIC:
        # no rounds (the slots hold what the banks gave them): the slots that go leave the table
        tbl = STATE["tbl"]
        out = [L.idx * tbl.shape[1] + L.held[j] for L, k in zip(LAYERS, ks) for j in range(k, L.k) if L.held[j] >= 0]
        for L, k in zip(LAYERS, ks):
            for j in range(k, L.x):
                L.held[j] = -1
        if out:
            tbl.view(-1).index_fill_(0, _to_dev(out, dev), -1)
    kd = _k_dev(dev)
    kd.copy_(torch.tensor(ks, dtype=torch.int64))
    torch.cuda.synchronize(dev)
    for L, k in zip(LAYERS, ks):
        if k < L.k:
            for _, nm in L.names:
                a, b = _span(L, nm, k), _span(L, nm, L.k)
                if b > a:
                    kt_kv_vmm.release(L.slots[nm], a, b - a)
            L.k = k
    if DYNAMIC:
        _union()
    STATE["lent_slots"] = sum(ks)
    STATE["spare_held"] = max(0, sum(_cost(L, k) for L, k in zip(LAYERS, ks)) - max(0, int(m.STATE["bytes"]) - want))
    if keep:
        STATE["owed"] = int(STATE.get("owed", 0) or 0) + freed
    STATE["shrinks"] = STATE.get("shrinks", 0) + 1
    if STATE["shrinks"] in (1, 2, 3, 10, 100, 1000, 10000) or CHECK or PHASES:
        _log(f"shrink {STATE['shrinks']} ({why}): {sum(old) - sum(ks)} of {sum(old)} slots out for "
             + ("the rows the card is short of" if slots is not None else
                f"the {need / 2**20:.0f} MiB the card is short" if exact else f"{want / 2**20:.0f} MiB of K/V")
             + f" ({freed / 2**20:.0f} MiB freed) in {(time.perf_counter() - t0) * 1e3:.1f} ms")
    return freed


def release(why: str) -> None:
    """Before the lent pages are mapped back (kt_kv_prefill_lend.restore): the
    routing leaves the slots, then the slots give their memory back."""
    if not LAYERS:
        return
    t0 = time.perf_counter()
    ph = _Phases(t0)
    _spread_back()
    dev = LAYERS[0].slots[LAYERS[0].names[0][1]].device
    kd = _k_dev(dev)
    kd.zero_()
    if DYNAMIC:
        # the slots leave the table; a round's copies in flight finish in the sync,
        # and what they were for is in each layer's held list for the next lend
        STATE["tbl"].fill_(-1)
        if STATE["utbl"] is not STATE["tbl"]:
            STATE["utbl"].fill_(-1)
        DYN["pending"] = None
        # a batch still loading is for slots about to go; its rows land in the
        # staging pool only, and the next lend fills the slots from each layer's held list
        if DYN.get("prep") is not None:
            for f in DYN["prep"][1]:
                f.result()
            DYN["prep"] = None
    ph("tables and loads")
    torch.cuda.synchronize(dev)
    ph("sync")
    for L in LAYERS:
        if L.k:
            for _, name in L.names:
                kt_kv_vmm.release(L.slots[name], 0, _span(L, name, L.x))
        L.k = 0
    ph("unmap")
    torch.cuda.synchronize(dev)
    ph("sync")
    STATE["releases"] += 1
    STATE["want"] = True
    STATE["owed"] = 0
    if STATE["releases"] in (1, 2, 10, 100, 1000) or CHECK or PHASES or why not in ("window start",):
        _log(f"release {STATE['releases']} ({why}): {STATE['lent_slots']} slots out in "
             f"{(time.perf_counter() - t0) * 1e3:.1f} ms" + (f" [{ph}]" if PHASES else ""))
    STATE["lent_slots"] = 0
    STATE["spare_held"] = 0
