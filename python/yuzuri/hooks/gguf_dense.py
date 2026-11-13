"""GGUF dense weights: prefill-sized batches, Q5_K embeddings, and the weight's MMQ tail at load. SGLang's GGUF
module keeps every call these do not take (kt_gguf_dense_ops holds the kernels' glue).

  fused_mul_mat_gguf        a batch of SGLANG_GGUF_MMQ_ROWS rows and more on llama.cpp's MMQ (the weight stays in
                            its GGUF blocks; SGLANG_GGUF_PREFILL_MMQ=0 turns it off), or, for a type MMQ does not
                            build, a sliced dequant + bf16 GEMM from SGLANG_GGUF_DEQUANT_ROWS rows
  apply_gguf_embedding      Q5_K rows in float32 as gguf-py opens them, in steps
  GGUFLinearMethod.process_weights_after_loading   the bytes MMQ reads past a weight's last row, once, before any
                            decode graph records the weight
  GGUFLinearMethod.apply    a weight whose column blocks were reordered at load (_kt_in_perm) gets its input in
                            that order
  GGUFEmbeddingMethod.process_weights_after_loading   code that ties an output head to the embedding reads .weight
  kt_sm75_ops               sgl-kernel's GGUF ops built for GPUs older than the wheel's arches (nothing on SM80+)
"""
from yuzuri._finder import after_import
from yuzuri.hooks._seam import rewrite, wrap

REPLACES = ["srt/layers/quantization/gguf.py"]


def _install(m):
    import torch
    from torch.nn.parameter import Parameter

    from sglang.srt.layers import kt_sm75_ops
    from sglang.srt.layers.quantization import kt_gguf_dense_ops as ops

    kt_sm75_ops.load_if_needed()

    def make_mul(orig):
        def fused_mul_mat_gguf(x, qweight, qweight_type):
            if qweight_type in m.UNQUANTIZED_TYPES:
                return orig(x, qweight, qweight_type)
            if ops.mmq_takes(x, qweight, qweight_type):
                out = torch.empty(x.shape[0], qweight.shape[0], dtype=x.dtype, device=x.device)
                ops.mmq_mul_mat(x, qweight, qweight_type, out)
                return out
            if x.shape[0] >= ops._DEQUANT_ROWS and qweight_type in m.DEQUANT_TYPES:
                return ops.dequant_mul_mat(x, qweight, qweight_type)
            return orig(x, qweight, qweight_type)

        return fused_mul_mat_gguf

    def make_embedding(orig):
        def apply_gguf_embedding(x, qweight, qweight_type, hidden_size, dtype=None):
            if qweight_type != m.WeightType.Q5_K:
                return orig(x, qweight, qweight_type, hidden_size, dtype)
            x_flat = x.flatten()
            out = torch.empty(x_flat.shape[0], hidden_size, dtype=dtype or torch.bfloat16, device=qweight.device)
            for r0 in range(0, x_flat.shape[0], ops._Q5K_ROWS_A_STEP):
                ids = x_flat[r0 : r0 + ops._Q5K_ROWS_A_STEP]
                out[r0 : r0 + ids.shape[0]] = ops.q5k_rows_f32(torch.index_select(qweight, 0, ids))
            return out.view(*x.shape, hidden_size)

        return apply_gguf_embedding

    def make_loaded(orig):
        def process_weights_after_loading(self, layer):
            orig(self, layer)
            qweight = getattr(layer, "qweight", None)
            if not isinstance(qweight, Parameter) or isinstance(qweight, m.GGUFUninitializedParameter):
                return
            types = set(layer.qweight_type.shard_weight_type.values()) or {layer.qweight_type.weight_type}
            if ops._MMQ_ON and qweight.is_cuda and types & ops.MMQ_PREFILL_TYPES:
                ops.mmq_qbuf(qweight.device)
            padded = ops.pad_for_mmq(qweight.data, types)
            if padded is not None:
                padded_param = Parameter(padded, requires_grad=False)
                for key, value in vars(qweight).items():  # the attributes the loader reads off the parameter
                    setattr(padded_param, key, value)
                layer.register_parameter("qweight", padded_param)

        return process_weights_after_loading

    wrap(m, "fused_mul_mat_gguf", ["x", "qweight", "qweight_type"], make_mul,
         adds=("mmq_mul_mat", "_DEQUANT_ROWS"), needs=("qweight_type in UNQUANTIZED_TYPES",))
    wrap(m, "apply_gguf_embedding", ["x", "qweight", "qweight_type", "hidden_size", "dtype"], make_embedding,
         adds=("q5k_rows_f32",))
    L = m.GGUFLinearMethod
    wrap(L, "process_weights_after_loading", ["self", "layer"], make_loaded, adds=("pad_for_mmq",),
         needs=("self._create_padded_weight_param(layer)",))
    rewrite(L, "apply", [(
        "        out = fused_mul_mat_gguf(x, qweight, qweight_type)\n",
        "        perm = getattr(layer, \"_kt_in_perm\", None)\n"
        "        if perm is not None:\n"
        "            x = x.index_select(-1, perm)\n"
        "        out = fused_mul_mat_gguf(x, qweight, qweight_type)\n",
    )], adds=("_kt_in_perm",))

    E = m.GGUFEmbeddingMethod
    if "process_weights_after_loading" in vars(E):
        return  # SGLang gave the embedding its own: what it does there is its own

    def process_weights_after_loading(self, layer):
        L.process_weights_after_loading(self, layer)
        if "weight" not in layer._parameters:
            layer.__dict__["weight"] = layer.qweight

    process_weights_after_loading.__qualname__ = "GGUFEmbeddingMethod.process_weights_after_loading"
    E.process_weights_after_loading = process_weights_after_loading


after_import("sglang.srt.layers.quantization.gguf", _install)
