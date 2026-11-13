"""A speculative draft model's routed experts from a GGUF of its own (KT_MTP_GGUF=<gguf>), all of them on the GPU
in the file's own quantization.

The target's experts file (kt_weight_path) holds the target's layers only; a draft whose experts ship as a GGUF
(Strata's mtp-q2_0.gguf next to its IQ3_S model) names that file here. The GGUF keeps the experts under the
checkpoint's fused layout: experts.gate_up_proj [E, 2I, H] (gate rows, then up rows) and experts.down_proj
[E, H, I]. The raw blocks go to the GPU as they are, cut to this rank's TP part: rows N*r..N*(r+1) of gate and of
up, the down's quant blocks covering columns N*r..N*(r+1). Windows of up to kt_gguf_moe_multi.MAX_TOKENS tokens
(decode, verify) run the multi-token GEMV on the raw blocks; longer windows (the draft's extend over a prompt) run
llama.cpp's int8-mma MMQ (kt_gguf_moe_mmq.py). Both weights get ggml's row padding after their last row, which the
MMQ kernel reads; the rows stay packed, so the GEMV's strides do not change.

FusedMoE picks this method for every MoE layer of a draft model (fused_moe_triton/layer.py), by the loader's
"building a draft" span (kt_ep_methods.building), not by a layer name.
"""
import os
import time
import warnings

import numpy as np
import torch
from torch.nn import Parameter

from sglang.srt.layers.moe.kt_gguf_dequant import BLOCK_INFO
from sglang.srt.layers.moe.kt_gguf_hot import KTGGUFHotExpertsMethod, _log, _row_bytes


class _DraftExperts:
    """The two fused expert tensors of the draft's GGUF, as views over the file mapping."""

    def __init__(self, path: str):
        from kt_kernel.utils.loader import GGUFLoader

        self.path = path
        self.loader = GGUFLoader(path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path)))
        base = None if os.path.isdir(path) else os.path.basename(path)
        names = [n for n in self.loader.tensor_info
                 if ".experts." in n and (base is None or os.path.basename(str(self.loader.tensor_file_map[n])) == base)]
        gate_up = [n for n in names if n.endswith("gate_up_proj")]
        down = [n for n in names if n.endswith("down_proj")]
        if len(gate_up) != 1 or len(down) != 1:
            raise RuntimeError(f"{path}: expected one fused gate_up_proj and one down_proj expert tensor, "
                               f"found {len(gate_up)} and {len(down)}")
        self.gate_up, self.down = gate_up[0], down[0]
        E, I2, H = (int(x) for x in self.loader.tensor_info[self.gate_up]["shape"])
        self.E, self.inter, self.K = E, I2 // 2, H

    def raw(self, name: str):
        info = self.loader.tensor_info[name]
        t = int(info["dtype"])
        blck, ts = BLOCK_INFO[t]
        n_bytes = int(info["n_elements"]) // blck * ts
        mm = self.loader.file_data_map[self.loader.tensor_file_map[name]]
        raw = np.frombuffer(mm, dtype=np.uint8, count=n_bytes, offset=int(info["offset"]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # read-only buffer, only copied out
            return torch.from_numpy(raw), t


class KTGGUFDraftExpertsMethod(KTGGUFHotExpertsMethod):
    """Every routed expert of a draft model's MoE layer on the GPU, from the draft's GGUF."""

    streams_hot_experts = False
    # the draft's checkpoint carries no expert tensors for this layer; if it does, they are not read
    takes_no_checkpoint_experts = True

    def __init__(self, path: str, layer):
        self.reader = _DraftExperts(path)
        self.rank, self.world = layer.moe_tp_rank, layer.moe_tp_size
        self.ep = False
        self.slice_rank = self.rank
        _, t13 = self.reader.raw(self.reader.gate_up)
        _, t2 = self.reader.raw(self.reader.down)
        self.types = (t13, t13, t2)
        self.type13, self.type2 = t13, t2
        self.mmq_chunk = int(os.environ.get("KT_MTP_MMQ_CHUNK", "2048"))
        self.num_gpu_experts = self.reader.E
        self.moe_runner_config = None
        self.kt_config = None

    def create_weights(self, layer, num_experts, hidden_size, intermediate_size_per_partition, params_dtype,
                       **extra_weight_attrs):
        from sglang.srt.layers.moe import kt_gguf_moe_mmq as _mmq

        E, N, K = num_experts, intermediate_size_per_partition, hidden_size
        if E != self.reader.E or N * self.world != self.reader.inter or K != self.reader.K:
            raise RuntimeError(f"draft GGUF experts {self.reader.E}x{self.reader.inter}x{self.reader.K} "
                               f"vs layer {E}x{N}x{self.world}/{K}")
        self.N, self.K = N, K
        dev = torch.device("cuda", torch.cuda.current_device())
        w13 = Parameter(self._padded((E, 2 * N, _row_bytes(K, self.type13)), _mmq.tail_bytes(self.type13, K), dev),
                        requires_grad=False)
        w2 = Parameter(self._padded((E, K, _row_bytes(N, self.type2)), _mmq.tail_bytes(self.type2, N), dev),
                       requires_grad=False)
        w13.weight_loader = w2.weight_loader = self._refuse_loader
        layer.register_parameter("w13_qweight", w13)
        layer.register_parameter("w2_qweight", w2)

    @staticmethod
    def _padded(shape, tail: int, dev) -> torch.Tensor:
        n = shape[0] * shape[1] * shape[2]
        buf = torch.empty(n + tail, dtype=torch.uint8, device=dev)
        buf[n:].zero_()
        return buf[:n].view(shape)

    def process_weights_after_loading(self, layer) -> None:
        t0 = time.time()
        E, N, K, r, I = self.reader.E, self.N, self.K, self.rank, self.reader.inter
        data, t13 = self.reader.raw(self.reader.gate_up)
        blck, ts = BLOCK_INFO[t13]
        gu = data.view(E, 2 * I, (K // blck) * ts)
        w13 = torch.cat([gu[:, N * r:N * (r + 1)], gu[:, I + N * r:I + N * (r + 1)]], dim=1).contiguous()
        data, t2 = self.reader.raw(self.reader.down)
        blck, ts = BLOCK_INFO[t2]
        if (N * r) % blck or N % blck:
            raise RuntimeError(f"draft GGUF down_proj: type {t2} blocks of {blck} values do not cut at this rank's "
                               f"columns {N * r}..{N * (r + 1)}")
        dn = data.view(E, K, I // blck, ts)[:, :, (N * r) // blck:(N * (r + 1)) // blck, :].reshape(E, K, -1).contiguous()
        assert tuple(w13.shape) == tuple(layer.w13_qweight.shape), (w13.shape, layer.w13_qweight.shape)
        assert tuple(dn.shape) == tuple(layer.w2_qweight.shape), (dn.shape, layer.w2_qweight.shape)
        layer.w13_qweight.data.copy_(w13)
        layer.w2_qweight.data.copy_(dn)
        nbytes = layer.w13_qweight.untyped_storage().nbytes() + layer.w2_qweight.untyped_storage().nbytes()
        _log(f"rank {r}: draft experts from {os.path.basename(self.reader.path)}, ggml types {self.types} as they are, "
             f"{E} experts, {nbytes / 2**20:.1f} MiB on the GPU, {time.time() - t0:.1f} s")

    def apply(self, layer, dispatch_output):
        from sglang.srt.layers.moe import kt_gguf_moe_multi as _multi

        x = dispatch_output.hidden_states
        topk_weights, topk_ids, _ = dispatch_output.topk_output
        if x.shape[0] <= _multi.MAX_TOKENS and _multi.enabled():
            return super().apply(layer, dispatch_output)
        return self._apply_mmq(layer, x, topk_weights, topk_ids)

    def _apply_mmq(self, layer, x, topk_weights, topk_ids):
        """A long window: llama.cpp's MoE MMQ for gate/up and for down (fp32 out), SiLU-and-mul, the top-k weighted
        sum as one bmm. KT_MTP_MMQ_CHUNK tokens a pass (2048) bound the fp32 intermediates."""
        from sgl_kernel import silu_and_mul

        from sglang.srt.layers.moe import kt_gguf_moe_mmq as _mmq
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        T, top_k = topk_ids.shape
        w13, w2 = layer.w13_qweight, layer.w2_qweight
        result = torch.empty_like(x)
        step = self.mmq_chunk if self.mmq_chunk > 0 else T
        for c0 in range(0, T, step):
            c1 = min(T, c0 + step)
            n = c1 - c0
            ids = topk_ids[c0:c1]
            h = _mmq.moe_mmq(x[c0:c1], w13, ids, self.type13, 2 * self.N, False)
            h = silu_and_mul(h.view(n * top_k, 2 * self.N))
            out = _mmq.moe_mmq(h, w2, ids, self.type2, self.K, True)
            result[c0:c1] = torch.bmm(topk_weights[c0:c1].view(n, 1, top_k).float(), out).view(n, self.K).to(x.dtype)
        return StandardCombineInput(hidden_states=result)
