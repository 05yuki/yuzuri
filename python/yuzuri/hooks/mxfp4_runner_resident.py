"""The MXFP4 CUTLASS MoE method under kt-kernel, where the method holds the experts resident on the card only:

  E                     the per-expert tensors are sized from the weights that exist (layer.num_local_experts counts
                        every routed expert, and FlashInfer rejects a swiglu_limit of another length)
  _swiglu_limit_value   the clamp as a scalar: with no resident expert the tensor is empty and the streamed prefill
                        (kt_stream_prefill) needs the value
  the CUTLASS workspace made here, before the decode CUDA graphs are captured: FlashInfer makes it inside the capture
                        otherwise, and that scratch aliased live memory after long prefills. Sized for this rank's
                        experts or the streamed prefill's group (KT_GPU_STREAM_GROUP), whichever is larger, and for
                        the streamer's bf16 mode (KT_GPU_STREAM_W16) as well
"""
from yuzuri._finder import after_import
from yuzuri.hooks._seam import rewrite

REPLACES = ["srt/layers/quantization/mxfp4_flashinfer_cutlass_moe.py"]
_WORKSPACE = '''    self.runner = MoeRunner(MoeRunnerBackend.FLASHINFER_MXFP4, moe_runner_config)
    if self._use_mxfp8_act_scaling:
        from flashinfer.fused_moe.core import ActivationType

        from sglang.srt.layers.moe.kt_cutlass_workspace import (
            mxfp4_workspace_nbytes,
            preallocate_shared_cutlass_workspace,
        )

        w2 = layer.w2_weight
        act = ActivationType.SwigluStep if self._use_swiglu_step else ActivationType.Swiglu
        experts = max(w2.shape[0], int(os.environ.get("KT_GPU_STREAM_GROUP", "0") or 0))
        nbytes = mxfp4_workspace_nbytes(
            1, w2.shape[1], w2.shape[2] * 2, experts * layer.moe_ep_size, layer.top_k, torch.float8_e4m3fn, act,
            layer.moe_tp_size, layer.moe_tp_rank, layer.moe_ep_size, layer.moe_ep_rank, True, device)
        if os.environ.get("KT_GPU_STREAM_W16") == "1":
            nbytes = max(nbytes, mxfp4_workspace_nbytes(
                1, w2.shape[1], w2.shape[2] * 2, experts * layer.moe_ep_size, layer.top_k, torch.bfloat16, act,
                layer.moe_tp_size, layer.moe_tp_rank, layer.moe_ep_size, layer.moe_ep_rank, False, device,
                weight_dtype=torch.bfloat16))
        preallocate_shared_cutlass_workspace(nbytes, device)
'''


def _install(m):
    rewrite(m.Mxfp4FlashinferCutlassMoEMethod, "create_moe_runner", [
        ("E = layer.num_local_experts\n", "E = layer.w13_weight.shape[0]\n"),
        ("    if clamp_limit is not None:\n",
         "    self._swiglu_limit_value = None if clamp_limit is None else float(clamp_limit)\n"
         "    if clamp_limit is not None:\n"),
        ("    self.runner = MoeRunner(MoeRunnerBackend.FLASHINFER_MXFP4, moe_runner_config)\n", _WORKSPACE),
    ], adds=("_swiglu_limit_value", "preallocate_shared_cutlass_workspace"))


after_import("sglang.srt.layers.quantization.mxfp4_flashinfer_cutlass_moe", _install)
