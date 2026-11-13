"""Two lines of SGLang's MQA attention layer (srt/models/deepseek_v4.py), each compiled again with that line changed:

  _zero_padded_q     a prefill window's padded query buffer is made in the lend's scratch region (kt_lend), so it
                     goes back with the scratch when the window ends
  MQALayer.__init__  on CUDA nothing reads rotary_emb (the compressor and the C4 indexer keep it unused; the cos/sin
                     consumers are the NPU and ROCm paths): its fp32 cos/sin cache over max_position_embeddings (two
                     RoPE bases, 0.54 GB a card at 1M) is not built there
"""
from yuzuri._finder import after_import
from yuzuri.hooks._seam import SeamError, rewrite

REPLACES = ["srt/models/deepseek_v4.py"]


def _install(m):
    owners = [c for c in vars(m).values() if isinstance(c, type) and c.__module__ == m.__name__
              and "_zero_padded_q" in vars(c)]
    if len(owners) != 1:
        raise SeamError(f"yuzuri: {m.__name__} has {len(owners)} classes with _zero_padded_q, the hook was written "
                        f"for one")
    rewrite(owners[0], "_zero_padded_q", [
        ("""        buf = x.new_zeros(x.shape[0], kernel_num_heads, self.head_dim)
""", """        from sglang.srt.layers.moe import kt_lend

        with kt_lend.scratch_region():
            buf = x.new_zeros(x.shape[0], kernel_num_heads, self.head_dim)
"""),
    ], adds=("kt_lend",))
    if getattr(m, "_is_cuda", None) is None:
        raise SeamError(f"yuzuri: {m.__name__} no longer has _is_cuda: the hook that leaves rotary_emb unbuilt on "
                        f"CUDA has nothing to decide by in this SGLang")
    rewrite(m.MQALayer, "__init__", [
        ("self.rotary_emb = get_rope_wrapper(", "self.rotary_emb = None if _is_cuda else get_rope_wrapper("),
    ])


after_import("sglang.srt.models.deepseek_v4", _install)
