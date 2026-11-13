"""Dense (non-expert) weights of a GGUF made by llama.cpp's "qwen4exp" converter (Qwen3.8-Flash-Next) under the
checkpoint's own tensor names, so the model's load_weights loads the GGUF the way it loads the safetensors
checkpoint, and no weight is read from another checkpoint.

What the converter changed and what is undone here:
  A_log            -> ssm_a = -exp(A_log)                 : A_log = log(-ssm_a)
  dt_bias          -> ssm_dt.bias
  conv1d.weight    -> squeezed [C, k], V channels tiled   : [C, 1, k], grouped
  *norm.weight     -> +1 (all but linear_attn.norm)       : -1
  GDN V heads      -> tiled order                         : back to grouped order (in_proj_qkv V rows, in_proj_z /
                      in_proj_a / in_proj_b rows, A_log / dt_bias, out_proj columns)
  indexer qk_proj  -> split into indexer q_proj / k_proj  : concatenated back
  PLE shards       -> one per_layer_token_embd table      : served from its IQ4_NL rows (below)

A quantized tensor whose layer carries SGLang's GGUF method (the layer has a .qweight parameter: quantization_config
{"quant_method": "gguf", "modules_to_not_convert": [...]}) is yielded as its GGUF bytes (<name>.qweight, [rows,
row bytes] uint8) and type (<name>.qweight_type) and stays in its blocks on the GPU. The row permutations are done
on the byte rows. The GDN out_proj needs a column permutation: its blocks are reordered whole, rank-major, so a TP
rank's contiguous slice holds whole blocks of its own heads, and the layer gathers its input to match
(_kt_in_perm, read by GGUFLinearMethod.apply). Everything else (norms, hyper-connection mixes, indexer, router
gates, conv1d, the token embedding) is F32 / BF16 in the GGUF and viewed, or dequantized to bf16 with gguf-py.

The PLE table (per_layer_token_embd, IQ4_NL) is not dequantized: each TP rank reads the rows of its vocabulary range
into pinned host memory as they are in the file (90 bytes a row of 160) and the gather kernel below decodes the
gathered rows on the GPU. The model must be built with the PLE table offloaded to the host and its file backend
(--ple-offload-embedding --ple-offload-backend file): the model's own table is then a sparse mapping that costs
nothing and is dropped here.

Not yielded: the routed experts (blk.N.ffn_*_exps; kt-kernel reads them from the same GGUF).
KT_GGUF_DENSE_THREADS=<n> reads and dequantizes on n threads (default: the cores of this process).
"""
import os
import re
import time
import types
import warnings
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
import torch

import gguf.quants
from gguf.constants import GGMLQuantizationType

# GGUF tensor stem -> checkpoint name (layer tensors get "model.language_model.layers.{L}." in front)
_GLOBAL = {
    "token_embd": "model.language_model.embed_tokens.weight",
    "output": "lm_head.weight",
    "output_hc_norm": "model.language_model.hyper_connection_mixer.hc_norm.weight",
    "output_hc_down": "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
    "output_hc_up": "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight",
}
_LAYER = {
    "attn_norm": "input_layernorm.weight",
    "post_attention_norm": "post_attention_layernorm.weight",
    "attn_qkv": "linear_attn.in_proj_qkv.weight",
    "attn_gate": "linear_attn.in_proj_z.weight",
    "ssm_alpha": "linear_attn.in_proj_a.weight",
    "ssm_beta": "linear_attn.in_proj_b.weight",
    "ssm_a": "linear_attn.A_log",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_conv1d": "linear_attn.conv1d.weight",
    "ssm_norm": "linear_attn.norm.weight",
    "ssm_out": "linear_attn.out_proj.weight",
    "attn_q": "self_attn.q_proj.weight",
    "attn_k": "self_attn.k_proj.weight",
    "attn_v": "self_attn.v_proj.weight",
    "attn_output": "self_attn.o_proj.weight",
    "attn_q_norm": "self_attn.q_norm.weight",
    "attn_k_norm": "self_attn.k_norm.weight",
    "indexer.q_norm": "self_attn.indexer.q_layernorm.weight",
    "indexer.k_norm": "self_attn.indexer.k_layernorm.weight",
    "ffn_gate_inp": "mlp.gate.weight",
    "ffn_gate_inp_shexp": "mlp.shared_expert_gate.weight",
    "ffn_gate_shexp": "mlp.shared_expert.gate_proj.weight",
    "ffn_up_shexp": "mlp.shared_expert.up_proj.weight",
    "ffn_down_shexp": "mlp.shared_expert.down_proj.weight",
    "hc_attn_norm": "attn_hyper_connection.hc_norm.weight",
    "hc_attn_down": "attn_hyper_connection.input_mix_weight_down.weight",
    "hc_attn_up": "attn_hyper_connection.input_mix_weight_up.weight",
    "hc_attn_inject": "attn_hyper_connection.block_inject_weight.weight",
    "hc_ffn_norm": "mlp_hyper_connection.hc_norm.weight",
    "hc_ffn_down": "mlp_hyper_connection.input_mix_weight_down.weight",
    "hc_ffn_up": "mlp_hyper_connection.input_mix_weight_up.weight",
    "hc_ffn_inject": "mlp_hyper_connection.block_inject_weight.weight",
    "ple_conv1d": "ple.conv1d.weight",
    "ple_key": "ple.key_proj.weight",
    "ple_value": "ple.value_proj.weight",
    "ple_norm_conv": "ple.norm_conv.weight",
    "ple_norm_key": "ple.norm_key.weight",
    "ple_norm_query": "ple.norm_query.weight",
}
# norms the converter stored as gamma + 1 (every norm except linear_attn.norm)
_PLUS_ONE = {
    "attn_norm", "post_attention_norm", "attn_q_norm", "attn_k_norm", "indexer.q_norm", "indexer.k_norm",
    "hc_attn_norm", "hc_ffn_norm", "ple_norm_conv", "ple_norm_key", "ple_norm_query", "output_hc_norm",
}
_FLOAT_TYPES = {GGMLQuantizationType.F32, GGMLQuantizationType.F16, GGMLQuantizationType.BF16}
# checkpoint module -> the fused module of the model that holds it (the model's stacked_params_mapping)
_STACKED = (
    (".q_proj", ".qkv_proj"), (".k_proj", ".qkv_proj"), (".v_proj", ".qkv_proj"),
    (".gate_proj", ".gate_up_proj"), (".up_proj", ".gate_up_proj"),
    (".in_proj_qkv", ".in_proj_qkvz"), (".in_proj_z", ".in_proj_qkvz"),
)
PLE_TABLE = "per_layer_token_embd.weight"
_IQ4NL_VALUES = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)


def _threads() -> int:
    from sglang.srt.layers.moe import kt_cpus

    return kt_cpus.threads("KT_GGUF_DENSE_THREADS")


def _model_name(name: str) -> str:
    """The parameter name the model's load_weights gives a checkpoint name."""
    return name.replace("model.language_model.", "model.").replace(".self_attn", "")


def _tiled_to_grouped_rows(n_k: int, per_k: int, head_dim: int) -> torch.Tensor:
    """Row permutation that takes the converter's tiled V-head order back to the checkpoint's grouped order:
    tiled index t = (v * n_k + k) * head_dim + d holds grouped element (k * per_k + v) * head_dim + d."""
    n = n_k * per_k * head_dim
    tiled = torch.arange(n).reshape(n_k, per_k, head_dim).permute(1, 0, 2).reshape(n)
    inv = torch.empty_like(tiled)
    inv[tiled] = torch.arange(n)
    return inv


class GGUFDenseSource:
    def __init__(self, gguf_dir: str, hf_config, model=None, loader=None):
        if isinstance(hf_config, dict):
            t = hf_config.get("text_config", hf_config)
            g = lambda k, d=None: t.get(k, hf_config.get(k, d))  # noqa: E731
        else:
            t = getattr(hf_config, "text_config", hf_config)
            g = lambda k, d=None: getattr(t, k, getattr(hf_config, k, d))  # noqa: E731
        if loader is None:
            from kt_kernel.utils.loader import GGUFLoader

            loader = GGUFLoader(gguf_dir)
        self.loader = loader
        self.n_k = int(g("linear_num_key_heads"))
        self.n_v = int(g("linear_num_value_heads"))
        self.dk = int(g("linear_key_head_dim"))
        self.dv = int(g("linear_value_head_dim"))
        self.per_k = self.n_v // self.n_k
        self._row_perm_v = _tiled_to_grouped_rows(self.n_k, self.per_k, self.dv)
        self._row_perm_1 = _tiled_to_grouped_rows(self.n_k, self.per_k, 1)
        self.params = {n for n, _ in model.named_parameters()} if model is not None else set()
        self.tp = 1
        if model is not None:
            from sglang.srt.distributed.utils import get_group_rank_size

            for name, mod in model.named_modules():
                if name.endswith("linear_attn.out_proj") and hasattr(mod, "tp_group"):
                    self.tp = int(get_group_rank_size(mod.tp_group)[1])
                    break

    # -- names ---------------------------------------------------------------------------------------------

    def hf_name(self, gguf_name: str):
        stem = gguf_name[: -len(".weight")] if gguf_name.endswith(".weight") else gguf_name
        if stem in _GLOBAL:
            return _GLOBAL[stem], None, stem
        m = re.match(r"blk\.(\d+)\.(.+)$", stem)
        if not m:
            return None, None, stem
        layer, key = int(m.group(1)), m.group(2)
        if key.endswith("_exps") or key == "per_layer_token_embd":
            return None, layer, key
        if key in ("indexer.q_proj", "indexer.k_proj"):
            return f"model.language_model.layers.{layer}.self_attn.indexer.index_qk_proj.weight", layer, key
        if key not in _LAYER:
            raise KeyError(f"no checkpoint name for GGUF tensor {gguf_name}")
        return f"model.language_model.layers.{layer}.{_LAYER[key]}", layer, key

    def names(self):
        return sorted(n for n in self.loader.tensor_info if n != PLE_TABLE and self.hf_name(n)[0] is not None)

    def quantized(self, hf: str) -> bool:
        """True when the model keeps this tensor in a GGUF layer (a .qweight parameter)."""
        if not hf.endswith(".weight"):
            return False
        base = _model_name(hf)[: -len(".weight")]
        if base + ".qweight" in self.params:
            return True
        return any(base.endswith(a) and base[: -len(a)] + b + ".qweight" in self.params for a, b in _STACKED)

    # -- raw -> torch --------------------------------------------------------------------------------------

    def _info(self, gguf_name: str):
        info = self.loader.tensor_info[gguf_name]
        qt = GGMLQuantizationType(int(info["dtype"]))
        shape = tuple(int(s) for s in info["shape"])  # torch order
        mm = self.loader.file_data_map[self.loader.tensor_file_map[gguf_name]]
        return qt, shape, mm, int(info["offset"])

    def tensor(self, gguf_name: str) -> torch.Tensor:
        """The tensor's values: F32 as it is, everything else bf16 (quantized types through gguf-py)."""
        qt, shape, mm, off = self._info(gguf_name)
        n = int(np.prod(shape))
        if qt == GGMLQuantizationType.F32:
            return torch.from_numpy(np.frombuffer(mm, dtype=np.float32, count=n, offset=off).reshape(shape).copy())
        if qt == GGMLQuantizationType.F16:
            a = np.frombuffer(mm, dtype=np.float16, count=n, offset=off).reshape(shape).copy()
            return torch.from_numpy(a).to(torch.bfloat16)
        if qt == GGMLQuantizationType.BF16:
            a = np.frombuffer(mm, dtype=np.int16, count=n, offset=off).reshape(shape).copy()
            return torch.from_numpy(a).view(torch.bfloat16)
        rows, row_bytes = self._row_layout(qt, shape)
        raw = np.frombuffer(mm, dtype=np.uint8, count=rows * row_bytes, offset=off).reshape(rows, row_bytes)
        threads = _threads()
        if threads > 1 and rows >= 4 * threads and raw.nbytes >= (32 << 20):
            # a big table (token_embd) in row blocks on several threads; blocks never cross a row
            from concurrent.futures import ThreadPoolExecutor

            parts = np.array_split(raw, 4 * threads, axis=0)
            with ThreadPoolExecutor(max_workers=threads) as ex:
                f = np.concatenate(list(ex.map(lambda r: gguf.quants.dequantize(r, qt), parts)), axis=0)
        else:
            f = gguf.quants.dequantize(raw, qt)
        return torch.from_numpy(f.reshape(shape)).to(torch.bfloat16)

    @staticmethod
    def _row_layout(qt, shape) -> Tuple[int, int]:
        rows, cols = (shape[0], int(np.prod(shape[1:]))) if len(shape) > 1 else (1, shape[0])
        blck, ts = gguf.quants.GGML_QUANT_SIZES[qt]
        if cols % blck:
            raise ValueError(f"{qt.name} blocks of {blck} do not tile a row of {cols}")
        return rows, cols // blck * ts

    def raw_rows(self, gguf_name: str) -> Tuple[torch.Tensor, int]:
        """The tensor's bytes as [rows, row bytes] uint8 (a copy) and its type."""
        qt, shape, mm, off = self._info(gguf_name)
        rows, row_bytes = self._row_layout(qt, shape)
        raw = np.frombuffer(mm, dtype=np.uint8, count=rows * row_bytes, offset=off)
        return torch.from_numpy(raw.copy()).reshape(rows, row_bytes), int(qt)

    # -- inverse transforms ----------------------------------------------------------------------------------

    def _undo(self, key: str, t: torch.Tensor) -> torch.Tensor:
        if key in _PLUS_ONE:
            return (t.float() - 1.0).to(t.dtype)
        if key == "ssm_a":  # -exp(A_log) -> A_log, tiled -> grouped
            return torch.log(-t.float())[self._row_perm_1]
        if key == "ssm_dt.bias":
            return t[self._row_perm_1]
        if key == "ssm_conv1d":  # [C, k] squeezed, V channels tiled -> [C, 1, k], grouped
            qk = 2 * self.n_k * self.dk
            v = t[qk:]
            return torch.cat([t[:qk], v[self._row_perm_v]], 0).unsqueeze(1)
        if key == "ple_conv1d":
            return t.unsqueeze(1)
        if key == "ffn_gate_inp_shexp":  # the checkpoint keeps the shared-expert gate as [1, hidden]
            return t.reshape(1, -1)
        if key in ("ssm_alpha", "ssm_beta"):
            return t[self._row_perm_1]
        if key == "ssm_out":
            return t[:, self._row_perm_v]
        return self._undo_rows(key, t)

    def _undo_rows(self, key: str, t: torch.Tensor) -> torch.Tensor:
        """The row permutations, on a value matrix or on a byte matrix (one row of bytes is one output row)."""
        if key == "attn_qkv":
            qk = 2 * self.n_k * self.dk
            v = t[qk:]
            return torch.cat([t[:qk], v[self._row_perm_v]], 0)
        if key == "attn_gate":
            return t[self._row_perm_v]
        return t

    def ssm_out_layout(self, blk: int, tp: int):
        """Block order of an ssm_out row (rank-major, so a contiguous TP split hands each rank whole blocks of its
        own heads) and, per rank, the gather that brings the rank's local grouped input into that order."""
        dv, n_k, per_k = self.dv, self.n_k, self.per_k
        if blk % dv or n_k % tp:
            raise ValueError(f"ssm_out: block {blk} vs head dim {dv}, key heads {n_k} vs tp {tp}")
        hpb = blk // dv
        per_rank = n_k // tp
        if n_k % hpb or per_rank % hpb:
            raise ValueError(f"ssm_out: {hpb} heads per block does not tile {n_k} key heads over tp {tp}")
        n_blocks = n_k * per_k // hpb
        order, perms = [], []
        for r in range(tp):
            perm = []
            for b in range(n_blocks):
                t0 = b * hpb
                if (t0 % n_k) // per_rank != r:
                    continue
                order.append(b)
                for h in range(hpb):
                    t = t0 + h
                    v, k = t // n_k, t % n_k
                    base = ((k - r * per_rank) * per_k + v) * dv
                    perm.extend(range(base, base + dv))
            perms.append(perm)
        return order, perms

    # -- iteration -------------------------------------------------------------------------------------------

    def _fetch(self, n: str):
        hf, _layer, _key = self.hf_name(n)
        qt = GGMLQuantizationType(int(self.loader.tensor_info[n]["dtype"]))
        if self.quantized(hf) and qt not in _FLOAT_TYPES:
            return self.raw_rows(n)
        return self.tensor(n)

    def _fetched(self, names, fn):
        """fn(name) for each name, in order, computed up to 4N ahead on N threads (gguf-py's numpy and the mapping
        reads release the GIL)."""
        threads = _threads()
        if threads <= 1:
            for n in names:
                yield n, fn(n)
            return
        from collections import deque
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=threads, thread_name_prefix="kt-gguf-dense") as ex:
            q = deque()
            it = iter(names)
            for n in it:
                q.append((n, ex.submit(fn, n)))
                if len(q) >= 4 * threads:
                    break
            while q:
                n, f = q.popleft()
                nxt = next(it, None)
                if nxt is not None:
                    q.append((nxt, ex.submit(fn, nxt)))
                yield n, f.result()

    def __iter__(self) -> Iterator[Tuple[str, torch.Tensor]]:
        """(name, tensor) under the checkpoint's names. A tensor of a GGUF layer comes as <name>.qweight_type and
        <name>.qweight; the GDN in_proj_qkv of a fused GGUF in_proj_qkvz is handed to the layer here (delivered)."""
        pending_indexer: Dict[int, Dict[str, torch.Tensor]] = {}
        for n, fetched in self._fetched(self.names(), self._fetch):
            hf, layer, key = self.hf_name(n)
            if not self.quantized(hf):
                t = fetched
                if key in ("indexer.q_proj", "indexer.k_proj"):
                    d = pending_indexer.setdefault(layer, {})
                    d[key] = t
                    if len(d) == 2:
                        yield hf, torch.cat([d["indexer.q_proj"], d["indexer.k_proj"]], 0)
                        del pending_indexer[layer]
                    continue
                yield hf, self._undo(key, t)
                continue
            if isinstance(fetched, tuple):
                rows, qt = fetched
                rows = self._undo_rows(key, rows)
                if key == "ssm_out":
                    blk, ts = gguf.quants.GGML_QUANT_SIZES[GGMLQuantizationType(qt)]
                    order, _ = self.ssm_out_layout(blk, self.tp)
                    nb = rows.shape[1] // ts
                    rows = rows.view(rows.shape[0], nb, ts)[:, order].reshape(rows.shape[0], nb * ts).contiguous()
            else:
                # a GGUF layer whose weight the GGUF keeps unquantized: the bf16 matrix itself, typed BF16
                # (SGLang's GGUF method multiplies those directly)
                rows, qt = self._undo(key, fetched).to(torch.bfloat16), int(GGMLQuantizationType.BF16)
            wtype = torch.tensor(qt, dtype=torch.uint8)
            base = hf[: -len(".weight")]
            if key == "attn_qkv" and self._deliver_qkv(base, rows, wtype):
                continue
            yield base + ".qweight_type", wtype
            yield base + ".qweight", rows
        assert not pending_indexer, f"unpaired indexer projections in layers {sorted(pending_indexer)}"

    def _deliver_qkv(self, base: str, rows: torch.Tensor, wtype: torch.Tensor) -> bool:
        """The GDN q, k and v rows as shards 0, 1, 2 of the fused in_proj_qkvz: the checkpoint has them as one
        in_proj_qkv tensor, which the model maps to the shard tuple (0, 1, 2), and a GGUF layer takes its shards
        one at a time."""
        fused = _model_name(base)[: -len(".in_proj_qkv")] + ".in_proj_qkvz"
        if fused + ".qweight" not in self.params:
            return False
        params = dict(self._model.named_parameters())
        qw, qt = params[fused + ".qweight"], params[fused + ".qweight_type"]
        qk = self.n_k * self.dk
        for shard, sl in enumerate((slice(0, qk), slice(qk, 2 * qk), slice(2 * qk, None))):
            qt.weight_loader(qt, wtype, shard)
            qw.weight_loader(qw, rows[sl].contiguous(), shard)
        return True


# -- GDN out_proj ----------------------------------------------------------------------------------------------


def attach_ssm_out_perms(model, src: GGUFDenseSource) -> int:
    """After the load: each GDN out_proj holding GGUF blocks gets the input gather that matches its reordered
    columns (_kt_in_perm, read by GGUFLinearMethod.apply)."""
    from sglang.srt.distributed.utils import get_group_rank_size

    n = 0
    for name, mod in model.named_modules():
        if not name.endswith("linear_attn.out_proj") or not hasattr(mod, "qweight_type"):
            continue
        qt = GGMLQuantizationType(int(getattr(mod.qweight_type, "weight_type", 0)))
        if qt in _FLOAT_TYPES:
            continue
        blk, _ = gguf.quants.GGML_QUANT_SIZES[qt]
        rank, size = get_group_rank_size(mod.tp_group)
        _, perms = src.ssm_out_layout(blk, int(size))
        perm = torch.tensor(perms[int(rank)], dtype=torch.int64, device=mod.qweight.device)
        mod.register_buffer("_kt_in_perm", perm, persistent=False)
        n += 1
    return n


# -- PLE table -------------------------------------------------------------------------------------------------

_iq4nl_kernel = None


def _gather_kernel():
    global _iq4nl_kernel
    if _iq4nl_kernel is not None:
        return _iq4nl_kernel
    import triton
    import triton.language as tl

    @triton.jit
    def _gather_iq4nl_rows_kernel(
        table_ptr,
        ids_ptr,
        values_ptr,
        output_ptr,
        embedding_dim,
        row_bytes,
        tp_vocab_start,
        tp_vocab_end,
        BLOCK_D: tl.constexpr,
    ):
        # one gathered row a program: an IQ4_NL block is an fp16 scale and 16 bytes, the low nibbles index the
        # block's first 16 values and the high nibbles its last 16
        row_id = tl.program_id(0)
        global_idx = tl.load(ids_ptr + row_id)
        in_range = (global_idx >= tp_vocab_start) & (global_idx < tp_vocab_end)
        local_idx = tl.where(in_range, global_idx - tp_vocab_start, 0)
        offsets = tl.arange(0, BLOCK_D)
        mask = offsets < embedding_dim
        table = table_ptr.to(tl.int64).to(tl.pointer_type(tl.uint8))
        block = table + local_idx * row_bytes + (offsets // 32) * 18
        lo = tl.load(block, mask=mask, other=0).to(tl.uint16)
        hi = tl.load(block + 1, mask=mask, other=0).to(tl.uint16)
        d = (lo | (hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        within = offsets % 32
        q = tl.load(block + 2 + within % 16, mask=mask, other=0).to(tl.int32)
        nibble = (q >> ((within // 16) * 4)) & 0xF
        value = tl.load(values_ptr + nibble, mask=mask, other=0).to(tl.float32)
        out = (d * value).to(tl.bfloat16)
        tl.store(output_ptr + row_id * embedding_dim + offsets, tl.where(in_range, out, 0.0), mask=mask)

    _iq4nl_kernel = _gather_iq4nl_rows_kernel
    return _iq4nl_kernel


def _iq4nl_gather(self, input_ids: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Qwen4ExpPinnedHostEmbedding.gather over a table kept in IQ4_NL rows."""
    expected_shape = (*input_ids.shape, self.embedding_dim)
    if out is None:
        output = self.allocate_output(expected_shape, input_ids.device)
    else:
        if tuple(out.shape) != expected_shape:
            raise ValueError(f"invalid PLE prefetch output shape: {tuple(out.shape)} != {expected_shape}")
        if out.dtype != torch.bfloat16 or out.device != input_ids.device:
            raise ValueError("PLE prefetch output must be bfloat16 on the id device")
        output = out
    flat_ids = input_ids.reshape(-1).long()
    if flat_ids.numel():
        values = self._kt_iq4nl_values
        if values.device != flat_ids.device:
            values = self._kt_iq4nl_values = values.to(flat_ids.device)
        _gather_kernel()[(flat_ids.numel(),)](
            self.weight.data_ptr(),
            flat_ids,
            values,
            output,
            embedding_dim=self.embedding_dim,
            row_bytes=self.weight.shape[1],
            tp_vocab_start=self.shard_indices.org_vocab_start_index,
            tp_vocab_end=self.shard_indices.org_vocab_end_index,
            BLOCK_D=self._block_d,
        )
    return output


def _read_pinned(path: str, offset: int, rows: int, row_bytes: int) -> torch.Tensor:
    """rows x row_bytes of a file, from offset, into pinned host memory; the pages read leave the page cache."""
    table = torch.empty((rows, row_bytes), dtype=torch.uint8, pin_memory=True)
    buf = memoryview(table.numpy()).cast("B")
    total, step = rows * row_bytes, 64 << 20
    fd = os.open(path, os.O_RDONLY)
    try:

        def read(o: int) -> None:
            end, p = min(total, o + step), o
            while p < end:
                got = os.preadv(fd, [buf[p:end]], offset + p)
                if got <= 0:
                    raise IOError(f"{path}: short read at {offset + p}")
                p += got
            os.posix_fadvise(fd, offset + o, end - o, os.POSIX_FADV_DONTNEED)

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=max(1, min(_threads(), 16))) as ex:
            list(ex.map(read, range(0, total, step)))
    finally:
        os.close(fd)
    return table


def attach_ple_tables(model, src: GGUFDenseSource) -> int:
    """After the load: every offloaded PLE table of the model becomes the rows of its TP range of the GGUF's
    table, as they are in the file, in pinned host memory, with a gather that decodes them."""
    if PLE_TABLE not in src.loader.tensor_info:
        return 0
    qt, shape, mm, off = src._info(PLE_TABLE)
    if qt != GGMLQuantizationType.IQ4_NL:
        raise NotImplementedError(f"the GGUF's PLE table is {qt.name}: only IQ4_NL rows have a gather kernel")
    rows, row_bytes = src._row_layout(qt, shape)
    dim = shape[1]
    path = src.loader.tensor_file_map[PLE_TABLE]
    offset = int(getattr(mm, "offset", 0)) + off
    n = 0
    for name, mod in model.named_modules():
        emb = getattr(mod, "ngram_embedding", None)
        if emb is None or not hasattr(emb, "shard_indices"):
            continue
        if type(emb).__name__ != "Qwen4ExpPinnedHostEmbedding":
            raise NotImplementedError(
                f"{name}: the GGUF's PLE table is served from host memory in its IQ4_NL rows; "
                "launch with --ple-offload-embedding --ple-offload-backend file"
            )
        if int(emb.embedding_dim) != dim:
            raise ValueError(f"{name}: PLE rows of {emb.embedding_dim} values, the GGUF's have {dim}")
        start, end = int(emb.shard_indices.org_vocab_start_index), int(emb.shard_indices.org_vocab_end_index)
        if end > rows:
            raise ValueError(f"{name}: PLE rows {start}-{end} past the GGUF table's {rows}")
        old = emb._parameters["weight"]
        if not hasattr(old.data, "_sglang_ple_file_path") and emb._file_prefetcher is None and emb._file_rss_trimmer is None:
            # the model's own table is real pinned memory (backend pinned): this one would double it
            if old.numel() * old.element_size() > (1 << 30):
                raise NotImplementedError(
                    f"{name}: the model allocated its own PLE table in host memory ({old.numel() * old.element_size() / 2**30:.1f} GiB); "
                    "launch with --ple-offload-backend file so that table is a sparse mapping"
                )
        t0 = time.time()
        table = _read_pinned(path, offset + start * row_bytes, end - start, row_bytes)
        sparse = None
        pre, trim = emb._file_prefetcher, emb._file_rss_trimmer
        if pre is not None:
            try:
                sparse = os.readlink(f"/proc/self/fd/{pre._fd}")
            except (OSError, AttributeError):
                sparse = None
            pre.close()
        if trim is not None:
            trim.close()
        emb._file_prefetcher = None
        emb._file_rss_trimmer = None
        new = torch.nn.Parameter(table, requires_grad=False)
        new.weight_loader = getattr(old, "weight_loader", None)
        emb._parameters["weight"] = new
        del old
        if sparse and os.path.isfile(sparse) and os.stat(sparse).st_blocks == 0:
            os.unlink(sparse)  # the model's own table: a sparse file nothing was written to
        emb._kt_iq4nl_values = torch.tensor(_IQ4NL_VALUES, dtype=torch.int8, device=emb.weight_scale.device)
        emb.gather = types.MethodType(_iq4nl_gather, emb)
        print(
            f"[kt-gguf-dense] {name}: PLE rows {start}-{end} in IQ4_NL as they are in the file, "
            f"{table.numel() / 2**30:.1f} GiB of pinned host memory, {time.time() - t0:.1f} s",
            flush=True,
        )
        n += 1
    return n


# -- the loader's source -----------------------------------------------------------------------------------------


def gguf_dense_weights(gguf_dir: str, hf_config, model=None) -> Iterator[Tuple[str, torch.Tensor]]:
    """What the loader feeds the model's load_weights instead of the safetensors files when KT_GGUF_DENSE names
    a GGUF directory. Experts stay with kt-kernel (--kt-weight-path on the same GGUF)."""
    t0 = time.time()
    src = GGUFDenseSource(gguf_dir, hf_config, model)
    src._model = model
    n_q = n_f = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # numpy views of the read-only mapping; everything yielded is a copy
        for name, t in src:
            if name.endswith(".qweight"):
                n_q += 1
            elif not name.endswith(".qweight_type"):
                n_f += 1
            yield name, t
    # (runs once the model has taken the last tensor)
    if model is None:
        return
    n_q += sum(1 for n in src.params if n.endswith(".in_proj_qkvz.qweight"))
    n_perm = attach_ssm_out_perms(model, src)
    n_ple = attach_ple_tables(model, src)
    print(
        f"[kt-gguf-dense] {n_q} tensors in their GGUF blocks ({n_perm} GDN out_proj with an input gather), "
        f"{n_f} F32 / BF16 or dequantized to bf16, {n_ple} PLE tables in IQ4_NL rows, {time.time() - t0:.1f} s",
        flush=True,
    )
    # a fused GGUF layer stays uninitialized until its shards are padded together after the load
    left = [
        n
        for n, prm in model.named_parameters()
        if type(prm).__name__.endswith("UninitializedParameter") and not getattr(prm, "data_container", None)
    ]
    if left:
        print(f"[kt-gguf-dense] {len(left)} parameters left uninitialized, e.g. {left[:8]}", flush=True)


# kt_gguf_dense.py finds this mapper by the converter name the GGUF carries
ARCH = "qwen4exp"


def weights(gguf_dir: str, model_config, model=None) -> Iterator[Tuple[str, torch.Tensor]]:
    return gguf_dense_weights(gguf_dir, model_config.hf_config, model)
