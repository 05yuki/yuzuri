"""The FlashInfer MXFP4 runner's resident experts on the process's one CUTLASS workspace (kt_cutlass_workspace), and
over blocks of rows inside a streamed prefill:

  the workspace   FlashInfer allocates ~244 MB a call otherwise; beside a large KV pool that is an OOM waiting for a
                  fragmented moment. The buffer is made once, before the decode graphs are captured
                  (hooks/mxfp4_runner_resident.py), and passed to every call
  row blocks      during a KT streamed prefill the resident experts run over KT_GPU_STREAM_ROW_BLOCK rows at a time,
                  each block quantizing its own activations, so the workspace is a block's
"""
from yuzuri._finder import after_import
from yuzuri.hooks._seam import rewrite

REPLACES = ["srt/layers/moe/moe_runner/flashinfer_cutlass.py"]
_HEAD = '    x = dispatch_output.hidden_states\n    if x.shape[0] == 0:\n        return StandardCombineInput(hidden_states=x)\n'
_OUT = '        out = torch.empty(x.shape[0], out_hidden, dtype=output_dtype, device=x.device)\n'
_CALL = '        **humming_kwargs,\n'
_SPLIT = '''    # During a KT streamed prefill the
    # resident experts run over blocks of rows (KT_GPU_STREAM_ROW_BLOCK), each
    # quantizing its own activations, so the workspace is a block's
    from sglang.srt.layers.moe.kt_cutlass_workspace import _kt_streamed_prefill, _mxfp4_workspace
    from sglang.srt.layers.moe.kt_stream_prefill import _block_rows

    rows = dispatch_output.hidden_states.shape[0]
    step = _block_rows(rows)
    if (
        step < rows
        and dispatch_output.hidden_states_scale is None
        and getattr(dispatch_output, "hidden_states_pre_quant", None) is None
        and _kt_streamed_prefill()
    ):
        topk = dispatch_output.topk_output
        if TopKOutputChecker.format_is_bypassed(topk):
            topk = topk.to_standard()
        out = None
        for r0 in range(0, rows, step):
            r1 = min(rows, r0 + step)
            part_topk = type(topk)(
                *[
                    f[r0:r1]
                    if isinstance(f, torch.Tensor)
                    and f.dim() > 0
                    and f.shape[0] == rows
                    else f
                    for f in topk
                ]
            )
            part = dispatch_output._replace(
                hidden_states=dispatch_output.hidden_states[r0:r1],
                topk_output=part_topk,
            )
            y = _fused_experts_flashinfer_mxfp4_cutlass(
                part, quant_info, runner_config
            ).hidden_states
            if out is None:
                out = torch.empty(
                    (rows,) + tuple(y.shape[1:]), dtype=y.dtype, device=y.device
                )
            out[r0:r1].copy_(y)
        return StandardCombineInput(hidden_states=out)

'''
_WORKSPACE = '''    activation_type = (
        ActivationType.SwigluStep
        if quant_info.use_swiglu_step
        else ActivationType.Swiglu
    )
    workspace_kwargs = {}
    if use_mxfp8_act_scaling and not use_wfp4afp8_humming:
        ws = _mxfp4_workspace(
            x, quant_info, topk_ids.shape[1], activation_type, use_mxfp8_act_scaling
        )
        if ws is not None:
            workspace_kwargs = {"workspace_buffer": ws}
'''


def _install(m):
    rewrite(m, "_fused_experts_flashinfer_mxfp4_cutlass",
            [(_HEAD, _SPLIT + _HEAD), (_OUT, _OUT + _WORKSPACE), (_CALL, _CALL + "        **workspace_kwargs,\n")],
            adds=("workspace_buffer", "_block_rows"))


after_import("sglang.srt.layers.moe.moe_runner.flashinfer_cutlass", _install)
