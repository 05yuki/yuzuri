"""DSV4 sparse attention on SM89 and SM75. FlashMLA's kernels are SM90 and newer; Ada has bf16 tensor cores without
them, and Turing opens FP8 and takes the dot in fp32. On those two the Triton dual-scope kernel
(kernels/ops/attention/sm89_dual_scope.py) is the attention, FlashMLA's metadata is not made and its sparse prefill
is not taken. Any other card keeps SGLang's backend as it is: nothing is attached."""
from yuzuri._finder import after_import
from yuzuri.hooks._seam import rewrite, wrap

REPLACES = ["srt/layers/attention/deepseek_v4_backend.py"]
_CARDS = ((8, 9), (7, 5))
_SM89 = '''        if True:
            from sglang.kernels.ops.attention.sm89_dual_scope import flash_mla_sparse_sm89

            o = flash_mla_sparse_sm89(
                q=q,
                k_cache=swa_k_cache,
                indices=swa_page_indices,
                topk_length=swa_topk_lengths,
                attn_sink=attn_sink,
                head_dim_v=self.head_dim_v,
                softmax_scale=self.softmax_scale,
                extra_k_cache=extra_k_cache,
                extra_indices=extra_indices,
                extra_topk_length=extra_topk_lengths,
            )
        elif get_platform().is_sm120:
'''


def install(m, force=False):
    if not force and not (m._is_cuda and m.get_platform().device_capability in _CARDS):
        return

    def make(orig):
        def _create_flashmla_metadata():
            return None

        return _create_flashmla_metadata

    wrap(m, "_create_flashmla_metadata", [], make, adds=("sm89",), needs=("get_mla_metadata",))
    B = m.DeepseekV4AttnBackend
    rewrite(B, "_use_sparse_prefill",
            [("and not get_platform().is_sm120\n", "and not get_platform().is_sm120\n        and False\n")],
            adds=("sm89",))
    rewrite(B, "_forward_attention", [("        if get_platform().is_sm120:\n", _SM89)], adds=("sm89",))


after_import("sglang.srt.layers.attention.deepseek_v4_backend", install)
