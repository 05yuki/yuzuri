"""The KT expert-parallel method as Yuzuri runs it: the bodies hooks/kt_ep_methods.py puts on SGLang's
KTEPWrapperMethod (create_weights, process_weights_after_loading, submit, apply), and what they share.

  create_weights                   the GPU method sized by the GPU experts' count, NVFP4 experts on cuTile from the
                                   checkpoint's bytes, the hot lend's shadow layer, and kt-kernel made with a mask of
                                   the experts the GPU holds, the layer's SwiGLU clamp and its activation
  process_weights_after_loading    no GPU repacking for an empty resident set, the page-cache drop before kt-kernel
                                   makes its arenas, the hot lend's split, the streamer's early init, the hot cache
  submit                           the request staged by one kernel where kt-kernel takes a packer
  apply                            the streamed prefill, the hot lend's slots, the dynamic hot cache, and a resident
                                   set of zero experts
"""
import contextlib
import inspect
import os

import torch

from sglang.srt.layers.moe import kt_ep_wrapper as _base
from sglang.srt.layers.moe import kt_hot_cache
from sglang.srt.layers.moe.kt_ep_wrapper import mask_cpu_expert_ids
from sglang.srt.layers.moe.kt_handoff_pack import PACK as _KT_HANDOFF_PACK
from sglang.srt.layers.moe.kt_handoff_pack import SHARED_AFTER_SUBMIT as _KT_SHARED_AFTER_SUBMIT
from sglang.srt.layers.moe.kt_handoff_pack import pack_to_host
from sglang.srt.layers.moe.kt_stream_prefill import stream_prefill_for

# the prefill lend's calibration server measures GPU memory only: with a GGUF pack the CPU experts
# are skipped there
_CALIBRATING = os.environ.get("KT_PREFILL_LEND_CALIBRATING") == "1" and bool(os.environ.get("KT_GGUF_V41_DENSE"))

_DROPPED_ONCE = []  # the layers that called _drop_checkpoint_cache


def _drop_checkpoint_cache(layer_idx: int, num_layers: int = 0) -> None:
    """KT_GPU_STREAM_DROP_CACHE=<dir>: posix_fadvise(DONTNEED) every file
    under it, so the arenas kt-kernel is about to create find free 2 MB
    blocks."""
    import os
    import time

    root_dir = os.environ.get("KT_GPU_STREAM_DROP_CACHE", "")
    if not root_dir:
        return
    # the loader's prefetch evicts each finished layer itself; a whole-tree
    # drop here would throw away the next layer's background read
    # (the loader reads the last KT_LOAD_PREFETCH_TAIL layers after their
    # arena exists; this drop runs again for them so the arena finds 2 MB
    # blocks, as it did before the prefetch)
    # the first MoE layer always drops: it is not layer 0 when the model's first layers are dense,
    # and without this drop the shards' page cache stayed on the nodes (165 GiB on node 1)
    first = not _DROPPED_ONCE
    _DROPPED_ONCE.append(layer_idx)
    if not first and int(os.environ.get("KT_LOAD_PREFETCH", "0") or 0) > 0:
        tail = int(os.environ.get("KT_LOAD_PREFETCH_TAIL", "4"))
        if not num_layers or layer_idx < num_layers - tail - 1:
            return
    t0, n = time.time(), 0
    for root, _, files in os.walk(root_dir, followlinks=True):
        for name in files:
            try:
                fd = os.open(os.path.join(root, name), os.O_RDONLY)
            except OSError:
                continue
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                n += 1
            finally:
                os.close(fd)
    if first or os.environ.get("KT_GPU_STREAM_TIMING") == "1":
        print(f"[kt-stream] layer {layer_idx}: dropped the page cache of {n} files in "
              f"{time.time() - t0:.1f} s", flush=True)


class _gpu_local_experts:
    """around the GPU method's weight steps, the
    layer's num_local_experts is the GPU experts' count for methods that size by it."""

    _SIZED_BY_LAYER = ("ModelOptNvFp4FusedMoEMethod",)

    def __init__(self, wrapper, layer):
        from sglang.srt.layers.moe import kt_cutile_nvfp4

        self.on = type(kt_cutile_nvfp4.base_method(wrapper.gpu_method)).__name__ in self._SIZED_BY_LAYER
        self.layer, self.n = layer, wrapper.num_gpu_experts

    def __enter__(self):
        if self.on:
            self.keep = self.layer.num_local_experts
            self.layer.num_local_experts = self.n

    def __exit__(self, *exc):
        if self.on:
            self.layer.num_local_experts = self.keep
        return False


# True while the loader builds a speculative draft model (loader.py _initialize_model). The experts' file
# (kt_weight_path) holds the target's experts, indexed by the target's layers: a draft model's MoE layers keep
# the experts of their own checkpoint on the GPU, with no CPU experts and no hot slots.
_BUILDING_DRAFT = [False]


@contextlib.contextmanager
def building(is_draft: bool):
    was, _BUILDING_DRAFT[0] = _BUILDING_DRAFT[0], bool(is_draft)
    try:
        yield
    finally:
        _BUILDING_DRAFT[0] = was


_routing_dump_prefix = os.environ.get("KT_ROUTING_DUMP")

_gpu_mask_zero = os.environ.get("KT_GPU_MASK_ZERO") == "1"
# KT_GPU_ONE_CALL=0: the resident rows and the lent slots as two calls of the GPU expert method, as before
_gpu_one_call = os.environ.get("KT_GPU_ONE_CALL", "1") != "0"


def _two_banks(self, layer, x, hot, streamer):
    """The two banks (w13, w13_scale, w2, w2_scale of the resident rows and of the lent slots) when a decode
    step can run both in one launch: the fp32 MXFP4 method on both, every masked route at weight 0, banks of
    one shape, no streamer and no check of the slots against the CPU. Else None."""
    if not (_gpu_one_call and _gpu_mask_zero and streamer is None and hot is not None and hot[1] is not None):
        return None
    from sglang.srt.layers.moe import kt_kv_hot_lend

    L = self.__dict__.get("_kt_kv_hot")
    if L is None or kt_kv_hot_lend.CHECK or not kt_kv_hot_lend._MASK_ZERO or L.shadow is None:
        return None
    if type(self.gpu_method).__name__ != "Mxfp4Fp32MoEMethod" or type(L.method).__name__ != "Mxfp4Fp32MoEMethod":
        return None
    from sglang.srt.layers.moe.mxfp4_fp32_moe import two_banks_ok

    names = ("w13_weight", "w13_weight_scale_inv", "w2_weight", "w2_weight_scale_inv")
    a = tuple(getattr(layer, n) for n in names)
    b = tuple(getattr(L.shadow, n) for n in names)
    return (a, b) if two_banks_ok(x, a, b) else None

_hot_check = os.environ.get("KT_HOT_CHECK") == "1"

_routing_state = {}

# layer_idx -> KTEPWrapperMethod on the rank that owns the kt-kernel wrapper
# (rank 0); the streamed-prefill zero-copy path hands every layer's expert
# arenas to the other ranks in one go.
KT_LAYERS = {}


def _routing_count(device, layer_idx, topk_ids, rank):
    import threading
    import time

    st = _routing_state.get(device.index)
    if st is None:
        counts = torch.zeros((64, 512), dtype=torch.int32, device=device)
        ones = torch.ones((1,), dtype=torch.int32, device=device)
        st = _routing_state[device.index] = {"counts": counts, "ones": ones}

        def dump():
            while True:
                time.sleep(30)
                try:
                    c = st["counts"].cpu()
                    torch.save(
                        {"rank": rank, "logical_count": c},
                        f"{_routing_dump_prefix}-rank{rank}.pt",
                    )
                except Exception as e:  # never take the scheduler down for a dump
                    print(f"[kt-routing] dump failed: {e}", flush=True)

        threading.Thread(target=dump, daemon=True, name="kt-routing-dump").start()
    ids = topk_ids.reshape(-1).to(torch.int64)
    st["counts"][layer_idx].index_add_(0, ids, st["ones"].expand(ids.numel()))


def _kt_activation(layer: torch.nn.Module) -> dict:
    """The routed experts' activation for kt-kernel: {} for SiLU (any kt-kernel build),
    gelu_pytorch_tanh for SGLang's gelu (Gemma 4); an activation kt-kernel does not compute stops the launch."""
    act = str(getattr(getattr(layer, "moe_runner_config", None), "activation", None) or "silu").lower()
    if act == "silu":
        return {}
    if act.startswith("gelu"):
        return {"activation": "gelu_pytorch_tanh"}
    raise ValueError(f"kt-kernel's CPU experts compute silu and gelu_pytorch_tanh; this layer's activation is {act!r}")


def _kt_swiglu_kwargs(layer: torch.nn.Module) -> dict:
    """{"swiglu_limit": limit} for KTMoEWrapper when the layer clamps its
    SwiGLU, else {} (older kt-kernel builds keep working for models without
    a clamp). The value is what the resident GPU runner applies:
    moe_runner_config.swiglu_limit (DeepSeek V4: gate <= limit, |up| <= limit
    before silu(gate) * up). KT_SWIGLU_CLAMP=0 leaves it out, the behaviour
    before 10-12."""
    cfg = getattr(layer, "moe_runner_config", None)
    alpha = getattr(cfg, "gemm1_alpha", None)
    if alpha is not None:
        # swigluoai: kt-kernel computes gate * sigmoid(alpha * gate) * (up + 1) with gate and up clamped
        # to +-limit (act_fn with swiglu_alpha > 0). It is the activation itself, so KT_SWIGLU_CLAMP does not drop it.
        # Before, a layer with gemm1_alpha and no swiglu_limit got {} here and its CPU experts ran SiLU.
        beta = getattr(cfg, "gemm1_beta", None)
        if beta is not None and float(beta) != 1.0:
            raise RuntimeError(f"kt-kernel's swigluoai adds 1 to up; this layer's gemm1_beta is {beta}")
        limit = getattr(cfg, "gemm1_clamp_limit", None) or getattr(cfg, "swiglu_limit", None)
        return {"swiglu_alpha": float(alpha), **({"swiglu_limit": float(limit)} if limit else {})}
    if os.environ.get("KT_SWIGLU_CLAMP", "1") == "0":
        return {}
    limit = getattr(cfg, "swiglu_limit", None)
    if not limit:
        return {}
    if getattr(cfg, "gemm1_alpha", None) is not None or getattr(cfg, "gemm1_clamp_limit", None) is not None:
        raise RuntimeError(
            "KT_SWIGLU_CLAMP: this layer uses gemm1_alpha/gemm1_clamp_limit, which "
            "the KT CPU wrapper in this tree does not map; set KT_SWIGLU_CLAMP=0"
        )
    return {"swiglu_limit": float(limit)}


def create_weights(
    self,
    layer: torch.nn.Module,
    num_experts: int,
    hidden_size: int,
    intermediate_size_per_partition: int,
    params_dtype: torch.dtype,
    **extra_weight_attrs,
):
    """Create weights for both GPU and CPU experts.

    Args:
        layer: The MoE layer module
        num_experts: Total number of experts (GPU + CPU)
        hidden_size: Hidden dimension size
        intermediate_size_per_partition: Intermediate size per TP partition
        params_dtype: Data type for parameters
        **extra_weight_attrs: Additional weight attributes
    """
    self.global_num_experts = num_experts
    self.hidden_size = hidden_size
    self.intermediate_size_per_partition = intermediate_size_per_partition

    # Get required parameters from layer object
    # top_k: number of experts selected per token
    num_experts_per_tok = layer.top_k

    # intermediate_size_full: full intermediate size before TP partitioning
    intermediate_size_full = (
        layer.intermediate_size_per_partition * layer.moe_tp_size
    )

    layer_max_deferred = self.kt_config.max_deferred_experts_per_token or 0
    if (
        self.kt_config.max_deferred_experts_per_token is not None
        and self.kt_config.num_layers is not None
        and self.kt_config.layer_idx == self.kt_config.num_layers - 1
    ):
        layer_max_deferred = 0

    # 1. Create weights for GPU experts using the wrapped method
    # GPU experts: 0 to num_gpu_experts-1
    with _gpu_local_experts(self, layer):
        self.gpu_method.create_weights(
            layer=layer,
            num_experts=self.num_gpu_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            params_dtype=params_dtype,
            **extra_weight_attrs,
        )
    # kt_cutile_nvfp4: NVFP4 experts run from the checkpoint's bytes on FlashInfer's cuTile MoE (before the
    # hot lend's shadow copies the method)
    from sglang.srt.layers.moe import kt_cutile_nvfp4

    self.gpu_method = kt_cutile_nvfp4.maybe_wrap(self.gpu_method, layer, self.kt_config.weight_path,
                                                 self.kt_config.layer_idx)
    # kt_kv_hot_lend: the next X experts' weights on a shadow layer, for the
    # KV pool's free pages
    from sglang.srt.layers.moe import kt_kv_hot_lend

    kt_kv_hot_lend.make_shadow(self, layer, hidden_size, intermediate_size_per_partition,
                               params_dtype, extra_weight_attrs)

    # 2. Initialize KT wrapper for CPU experts
    # CPU experts: num_gpu_experts to num_experts-1
    if self.tp_rank == 0:
        # kt-kernel 0.7.0 inference API takes an explicit boolean mask of
        # GPU-resident experts instead of num_gpu_experts (SFT-only).
        # GPU experts occupy ordinals 0..num_gpu_experts-1 (see above).
        gpu_mask = torch.zeros(num_experts, dtype=torch.bool)
        # kt_kv_hot_lend's dynamic resident rows: kt-kernel loads every expert, the
        # routes say what it computes (-1 for what the GPU takes)
        # KT_HOT_DYNAMIC=1 (kt_hot_cache.py): the slots hold what the cache put there and the CPU
        # gets the logical ids with the residents set to -1, so kt-kernel must not skip ids on its own
        if not kt_kv_hot_lend.cpu_holds_all(self) and not kt_hot_cache.ENABLED:
            gpu_mask[: self.num_gpu_experts] = True
        KT_LAYERS[self.kt_config.layer_idx] = self
        self.wrapper = _base.KTMoEWrapper(
            layer_idx=self.kt_config.layer_idx,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            hidden_size=hidden_size,
            moe_intermediate_size=intermediate_size_full,
            gpu_experts_mask=gpu_mask,
            cpuinfer_threads=self.kt_config.cpuinfer_threads,
            threadpool_count=self.kt_config.threadpool_count,
            weight_path=self.kt_config.weight_path,
            chunked_prefill_size=self.kt_config.chunked_prefill_size,
            method=self.kt_config.method,
            max_deferred_experts_per_token=layer_max_deferred,
            # KT_SWIGLU_CLAMP
            **_kt_swiglu_kwargs(layer),
            **_kt_activation(layer),
            # KT_NUMA_NODES="1" pins a single pool to node 1, the node
            # that does not host the TP0 scheduler, for the prefill-decay
            # isolation; unset keeps kt-kernel's sequential default.
            numa_nodes=(
                [int(x) for x in os.environ["KT_NUMA_NODES"].split(",")]
                if os.environ.get("KT_NUMA_NODES")
                else None
            ),
        )


def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
    """Process weights after loading from checkpoint.

    Args:
        layer: The MoE layer module
    """
    # 1. Process GPU weights
    # Empty resident weights need no GPU repacking; still load CPU experts below.
    # (dsv41-flash-dual-epyc#2)
    if self.num_gpu_experts > 0 and hasattr(
        self.gpu_method, "process_weights_after_loading"
    ):
        with _gpu_local_experts(self, layer):
            self.gpu_method.process_weights_after_loading(layer)

    # 2. Load CPU weights using KT wrapper
    if self.tp_rank == 0 and self.wrapper is not None:
        torch.cuda.synchronize()

        # Get expert location metadata for CPU expert mapping
        from sglang.srt.eplb.expert_location import (
            get_global_expert_location_metadata,
        )

        # without the expert-location metadata the physical order is the logical one
        _meta = get_global_expert_location_metadata()
        physical_to_logical_map_cpu = (
            _meta.physical_to_logical_map_cpu[self.kt_config.layer_idx].contiguous()
            if _meta is not None
            else torch.arange(int(self.global_num_experts), dtype=torch.int64)
        )
        _drop_checkpoint_cache(self.kt_config.layer_idx, getattr(self.kt_config, "num_layers", 0) or 0)
        self.wrapper.load_weights(physical_to_logical_map_cpu)
    # kt_kv_hot_lend: experts N..N+X-1 to pinned memory and the shadow slots
    from sglang.srt.layers.moe import kt_kv_hot_lend

    kt_kv_hot_lend.split(self, layer)
    # Last layer loaded on this rank: optionally build the streamer and
    # pin every layer's arenas now (KT_GPU_STREAM_EARLY_INIT).
    num_layers = getattr(self.kt_config, "num_layers", None)
    if num_layers and self.kt_config.layer_idx == num_layers - 1:
        from sglang.srt.layers.moe.kt_stream_prefill import stream_prefill_init_early

        stream_prefill_init_early(self, layer)
    kt_hot_cache.register(self, layer)


def submit(
    self,
    layer: torch.nn.Module,
    dispatch_output: "StandardDispatchOutput",
) -> None:
    """Submit CPU expert computation asynchronously (non-blocking).

    This method submits the CPU expert computation to AMX/AVX without waiting
    for completion, allowing GPU computation to proceed in parallel.

    Args:
        layer: The MoE layer module
        dispatch_output: Dispatched tokens and routing information
    """
    # the activation went to kt-kernel when the wrapper was made

    if self.tp_rank != 0 or self.wrapper is None or _CALIBRATING:
        return

    x = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output
    # attributes, not unpacking: the packed variant carries a fourth field
    topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids

    # KT_ROUTING_DUMP=<prefix>: accumulate a [layers, experts] histogram of
    # the routed expert ids on the GPU (a captured index_add_, so it keeps
    # counting under CUDA graph replay) and save it as
    # <prefix>-rank<r>.pt every 30 s. SGLang's expert distribution recorder
    # never sees the V4 router's ids (09-13); this does.
    # Submit forward task to CPU (non-blocking)
    stream = torch.cuda.current_stream(x.device)
    kwargs = {}
    if _KT_HANDOFF_PACK:
        takes_packer = getattr(self, "_kt_takes_packer", None)
        if takes_packer is None:
            # a kt_kernel without the packer hook keeps its copies
            takes_packer = (
                "packer" in inspect.signature(self.wrapper.submit_forward).parameters
            )
            self._kt_takes_packer = takes_packer
        if takes_packer:
            # KT_HANDOFF_PACK: one kernel stages the request in the pinned buffers
            kwargs["packer"] = pack_to_host
    self.wrapper.submit_forward(
        x, topk_ids, topk_weights, stream.cuda_stream, **kwargs
    )
    if _KT_SHARED_AFTER_SUBMIT:
        # KT_SHARED_AFTER_SUBMIT: DeepseekV2MoE forks its shared expert here
        self._kt_post_submit_event = stream.record_event()


def apply(
    self,
    layer: torch.nn.Module,
    dispatch_output: "StandardDispatchOutput",
) -> "CombineInput":
    """Execute hybrid CPU+GPU MoE forward pass with parallelism.

    This is the main computation method that coordinates:
    1. Submit CPU expert computation (non-blocking)
    2. Execute GPU expert computation in parallel
    3. Synchronize CPU results and merge with GPU results

    Args:
        layer: The MoE layer module
        dispatch_output: Dispatched tokens and routing information

    Returns:
        Combined computation results from CPU and GPU experts
    """
    from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    if TopKOutputChecker.format_is_bypassed(dispatch_output.topk_output):
        dispatch_output = dispatch_output._replace(
            topk_output=dispatch_output.topk_output.to_standard()
        )

    x = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output
    # The ids are physical: --init-expert-location moves experts between the GPU
    # slots (physical 0..num_gpu_experts-1) and the CPU, and TopK resolves the
    # placement for every router (resolve_routing_inputs).
    # KT_ROUTING_DUMP=<prefix>: a [layers, experts] histogram of the routed physical ids
    # (a captured index_add_, so it keeps counting under CUDA graph replay), saved as
    # <prefix>-rank<r>.pt every 30 s. Counted here, before the hot lend and the
    # streamer split the routes; SGLang's
    # expert distribution recorder never sees the V4 router's ids (09-13).
    if _routing_dump_prefix and self.tp_rank == 0:
        _routing_count(x.device, self.kt_config.layer_idx, topk_output.topk_ids, self.tp_rank)

    # KT_GPU_STREAM_PREFILL=<tokens>: above that many tokens the CPU
    # experts are streamed to the GPU in groups instead of computed here
    # (kt_stream_prefill.py). The ids are taken before the mask below
    # mutates them in place.
    # kt_kv_hot_lend: a prefill window finds the resident rows holding 0..N-1
    from sglang.srt.layers.moe import kt_kv_hot_lend

    kt_kv_hot_lend.res_guard()
    streamer = stream_prefill_for(self, layer, x.shape[0])
    cpu_share = False
    if streamer is not None:
        physical_ids = topk_output.topk_ids.clone()
        from sglang.srt.layers.moe.kt_stream_prefill import stream_cpu_cutoff

        cutoff = stream_cpu_cutoff(self, streamer, x.shape[0])
        if cutoff < self.global_num_experts and self.tp_rank == 0:
            if self.num_gpu_experts <= 0:
                raise RuntimeError("KT_STREAM_CPU_GROUPS / KT_STREAM_CPU_SHORT need --kt-num-gpu-experts > 0")
            # Everything below the cutoff is streamed or resident: send it
            # to GPU expert 0 at weight 0, which kt-kernel skips.
            theirs = physical_ids < cutoff
            # (-1 when kt-kernel holds every expert: it would compute expert 0)
            skip = -1 if kt_kv_hot_lend.cpu_holds_all(self) else 0
            cpu_do = dispatch_output._replace(
                topk_output=topk_output._replace(
                    topk_ids=torch.where(theirs, torch.full_like(physical_ids, skip), physical_ids),
                    topk_weights=torch.where(
                        theirs, torch.zeros_like(topk_output.topk_weights), topk_output.topk_weights
                    ),
                )
            )
            self.submit(layer, cpu_do)
            cpu_share = True

    # kt_kv_hot_lend: routes to the experts lent the KV pool's free pages go to
    # their slots, and to the CPU at weight 0
    hot = kt_kv_hot_lend.route(self, dispatch_output) if streamer is None else None

    # KT_HOT_DYNAMIC=1 (kt_hot_cache.py): the GPU slots follow the decode's routing; the tables give
    # the CPU the logical ids minus the resident ones and the GPU method the slot ids
    if streamer is None and kt_hot_cache.ENABLED and kt_hot_cache._cache is not None and kt_hot_cache._cache.ready:
        slot_ids, cpu_ids = kt_hot_cache._cache.route(self.kt_config.layer_idx, topk_output.topk_ids)
        if self.tp_rank == 0:
            self.submit(layer, dispatch_output._replace(topk_output=topk_output._replace(topk_ids=cpu_ids)))
        output = self.gpu_method.apply(
            layer, dispatch_output._replace(topk_output=topk_output._replace(topk_ids=slot_ids))
        ).hidden_states
        if self.tp_rank == 0:
            output = output + self.sync(x)
        return StandardCombineInput(hidden_states=output)

    # Step 1: Submit CPU expert computation (non-blocking)
    if streamer is None and self.tp_rank == 0:
        self.submit(layer, hot[0] if hot is not None else dispatch_output)

    # Step 2: Prepare GPU computation by masking CPU expert IDs
    # CPU expert IDs (>= num_gpu_experts) are set to -1 so GPU kernel skips them
    topk_ids = topk_output.topk_ids
    if hot is not None and hot[3] is not None:
        # kt_kv_hot_lend's dynamic resident rows: the row each route's expert is in
        masked_topk_ids = hot[3]
    else:
        masked_topk_ids = mask_cpu_expert_ids(topk_ids, self.num_gpu_experts)

    # Create modified dispatch output for GPU computation
    if _gpu_mask_zero:
        # KT_GPU_MASK_ZERO=1: the Marlin MoE path cannot take -1 ids (its
        # alignment kernels index weights with them: illegal address at
        # graph capture, 09-13). Send the CPU-bound rows to GPU expert 0
        # with a zero routing weight instead — the second GEMM multiplies
        # by the weight, so they add nothing — and hand the ids over as
        # int64 so fused_marlin_moe takes its generic alignment path,
        # which is fine with duplicate ids within a token.
        m = masked_topk_ids < 0
        masked_topk_output = topk_output._replace(
            topk_ids=torch.where(m, torch.zeros_like(masked_topk_ids), masked_topk_ids).to(torch.int64),
            topk_weights=torch.where(m, torch.zeros_like(topk_output.topk_weights), topk_output.topk_weights),
        )
    else:
        masked_topk_output = topk_output._replace(topk_ids=masked_topk_ids)
    masked_dispatch_output = dispatch_output._replace(
        topk_output=masked_topk_output
    )

    # Step 3: Execute GPU expert computation (any quantization method)
    # This runs in parallel with CPU computation. With no resident GPU
    # experts (--kt-num-gpu-experts 0, every slot given to the streamer
    # and the KV pool) there is nothing to run and the FlashInfer MXFP4
    # runner refuses a zero-expert workspace (09-16).
    if self.num_gpu_experts > 0 and not (
        streamer is not None
        and getattr(self.gpu_method, "streams_hot_experts", False)
    ):
        one = _two_banks(self, layer, x, hot, streamer)
        if one is not None:
            # the resident rows and the lent slots in one launch of each kernel (mxfp4_fp32_moe_two)
            from sglang.srt.layers.moe.mxfp4_fp32_moe import mxfp4_fp32_moe_two

            output = mxfp4_fp32_moe_two(
                x, one[0], one[1], masked_topk_output.topk_weights, masked_topk_output.topk_ids,
                hot[1].topk_output.topk_weights, hot[1].topk_output.topk_ids,
                swiglu_limit=self.gpu_method.runner.config.swiglu_limit)
            hot = (hot[0], None, hot[2], hot[3])
        else:
            gpu_combine_input = self.gpu_method.apply(layer, masked_dispatch_output)
            output = gpu_combine_input.hidden_states
        if streamer is not None and _hot_check:
            from sglang.srt.layers.moe.kt_stream_prefill import hot_reference_check

            hot_reference_check(self, layer, x, physical_ids, topk_output.topk_weights, output, streamer)
    else:
        output = torch.zeros_like(x)
    if hot is not None and hot[1] is not None:
        output = output + kt_kv_hot_lend.apply(self, hot[1])

    # Step 4: Synchronize CPU results and merge with GPU results
    if streamer is not None:
        if getattr(streamer, "accumulates_into", False) and output.is_contiguous() and \
                output.untyped_storage().data_ptr() != x.untyped_storage().data_ptr():
            # the GGUF MMQ streamer adds into the output: no chunk-sized accumulator and no sum
            output = streamer.run(self, layer, dispatch_output, physical_ids, into=output)
        else:
            output = output + streamer.run(self, layer, dispatch_output, physical_ids)
        if cpu_share:
            output = output + self.sync(x)
    elif self.tp_rank == 0:
        cpu_output = self.sync(x)
        output = output + cpu_output

    kt_kv_hot_lend.check(self, layer, dispatch_output, hot, output)
    return StandardCombineInput(hidden_states=output)
