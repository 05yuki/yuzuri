"""The CUTLASS MoE workspace every caller in the process shares (the resident experts of the FlashInfer MXFP4 runner,
hooks/cutlass_shared_workspace.py, and the streamed prefill's groups, kt_stream_prefill.py)."""
import logging
from typing import Optional

import torch

from sglang.srt.environ import envs

# One persistent CUTLASS MoE workspace per device, shared by every caller
# in the process (the resident-expert path here and the KT streamed-prefill
# groups). Without it the runner allocates ~244 MB per call through the
# caching allocator; beside a 1M-token KV pool on a 16 GB card that is an
# OOM waiting for a fragmented moment (09-13), and the allocation is one more
# host-side stall while a DMA holds the PCIe link.
logger = logging.getLogger(__name__)
_shared_cutlass_workspace = {}
# The buffer is baked into the decode CUDA graphs by address, so it is
# allocated once, before any capture (preallocate_shared_cutlass_workspace
# from Mxfp4FlashinferCutlassMoEMethod.create_moe_runner, hooks/mxfp4_runner_resident.py), sized for the
# prefill chunk and the largest expert group any caller uses, and never
# replaced. Replacing it (09-13: allocate small during capture, grow at the
# first prefill, drop the old one) left the decode graphs scribbling into
# freed memory; whether that hit live tensors depended on allocation
# history, and it did after a three-chunk prefill. Allocating inside a
# capture put the buffer in the graph's private pool, which was no better.


def _kt_streamed_prefill() -> bool:
    """Inside a KT prefill window (the lend's prefill phase)."""
    try:
        from sglang.srt.layers.moe import kt_lend
    except ImportError:
        return False
    return kt_lend.ENABLED and kt_lend.phase() == "prefill"


def get_shared_cutlass_workspace(
    nbytes: int, device: torch.device
) -> Optional[torch.Tensor]:
    """The preallocated buffer if it is large enough, else None (the caller
    lets FlashInfer allocate for that call) with a loud warning."""
    key = device.index if device.index is not None else torch.cuda.current_device()
    ws = _shared_cutlass_workspace.get(key)
    if ws is None or ws.numel() < nbytes:
        logger.warning(
            "shared CUTLASS MoE workspace too small or missing (%d MB needed, %s); "
            "falling back to a per-call allocation",
            nbytes >> 20,
            "none" if ws is None else f"{ws.numel() >> 20} MB",
        )
        return None
    return ws


def preallocate_shared_cutlass_workspace(
    nbytes: int, device: torch.device
) -> torch.Tensor:
    assert not torch.cuda.is_current_stream_capturing()
    key = device.index if device.index is not None else torch.cuda.current_device()
    ws = _shared_cutlass_workspace.get(key)
    if ws is None or ws.numel() < nbytes:
        ws = torch.empty(nbytes, dtype=torch.uint8, device=device)
        _shared_cutlass_workspace[key] = ws
        logger.info("shared CUTLASS MoE workspace: %d MB on %s", nbytes >> 20, device)
    return ws


def _workspace_max_tokens(num_tokens: int) -> int:
    """Size the shared workspace for the prefill chunk on the first call so it
    is not re-allocated (and the old one retired) on the first real prefill."""
    try:
        from sglang.srt.runtime_context import get_schedule

        chunk = get_schedule().chunked_prefill_size or 0
    except Exception:
        chunk = 0
    # a streamed prefill's calls see a block at most
    from sglang.srt.layers.moe.kt_stream_prefill import _block_rows

    return max(num_tokens, _block_rows(chunk))


def mxfp4_workspace_nbytes(
    num_tokens,
    hidden,
    inter,
    num_experts_total,
    topk,
    x_dtype,
    activation_type,
    tp_size,
    tp_rank,
    ep_size,
    ep_rank,
    use_mxfp8_act_scaling,
    device,
    weight_dtype=torch.int64,
):
    from flashinfer.fused_moe import cutlass_fused_moe_workspace_size

    return cutlass_fused_moe_workspace_size(
        _workspace_max_tokens(num_tokens),
        hidden,
        inter,
        num_experts_total,
        topk,
        x_dtype=x_dtype,
        weight_dtype=weight_dtype,
        output_dtype=torch.bfloat16,
        activation_type=activation_type,
        tp_size=tp_size,
        tp_rank=tp_rank,
        ep_size=ep_size,
        ep_rank=ep_rank,
        use_w4_group_scaling=not use_mxfp8_act_scaling,
        use_mxfp8_act_scaling=use_mxfp8_act_scaling,
        use_fused_finalize=envs.SGLANG_FLASHINFER_MOE_FUSED_FINALIZE.get(),
        device=device,
    )


def _mxfp4_workspace(x, quant_info, topk, activation_type, use_mxfp8_act_scaling):
    w2 = quant_info.w2_weight
    nbytes = mxfp4_workspace_nbytes(
        x.shape[0],
        w2.shape[1],
        w2.shape[2] * 2,
        w2.shape[0] * quant_info.moe_ep_size,
        topk,
        x.dtype,
        activation_type,
        quant_info.moe_tp_size,
        quant_info.moe_tp_rank,
        quant_info.moe_ep_size,
        quant_info.moe_ep_rank,
        use_mxfp8_act_scaling,
        x.device,
    )
    return get_shared_cutlass_workspace(nbytes, x.device)
