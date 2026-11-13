"""GGUF dense weights at prefill sizes and in groups, the weight kept in its GGUF blocks (hooks/gguf_dense.py puts
these behind SGLang's fused_mul_mat_gguf, apply_gguf_embedding and GGUFLinearMethod).

  mmq_mul_mat            a prefill-sized batch on llama.cpp's MMQ (kt_gguf_moe_mmq), no copy of the weight
  dequant_mul_mat        a prefill-sized batch of a type MMQ does not build: dequantize in slices of rows, bf16 GEMM
  grouped_mul_mat_gguf   a grouped weight: every group in one launch at decode, MMQ or one batched GEMM at prefill
  q5k_rows_f32           Q5_K embedding rows in float32, in gguf-py's order of operations
"""
import os

import gguf
import torch

from sglang.srt.layers.quantization.gguf import (
    DEQUANT_TYPES,
    MMVQ_QUANT_TYPES,
    UNQUANTIZED_TYPES,
    _is_cuda,
    dequantize_gguf_weight,
)

_DEQUANT_ROWS = int(os.environ.get("SGLANG_GGUF_DEQUANT_ROWS", "64") or 1 << 30)
# Prefill-sized batches (SGLANG_GGUF_MMQ_ROWS rows and more) of the types below run on llama.cpp's MMQ
# (kt_gguf_moe_mmq, built for the GPU's own arch): the weight stays in its GGUF blocks, the activations are
# quantized to q8_1 in slices of rows whose buffer stays under SGLANG_GGUF_MMQ_SLICE_MB, and the output is
# written in the activations' dtype. SGLANG_GGUF_PREFILL_MMQ=0 keeps the dequant + bf16 GEMM path below.
_MMQ_ON = os.environ.get("SGLANG_GGUF_PREFILL_MMQ", "1") == "1"
_MMQ_ROWS = int(os.environ.get("SGLANG_GGUF_MMQ_ROWS", "64") or 1 << 30)
_MMQ_SLICE_MB = int(os.environ.get("SGLANG_GGUF_MMQ_SLICE_MB", "32") or 32)
# the template instances kt_gguf_moe_mmq builds: Q8_0, Q5_K, IQ2_XXS, IQ2_XS, IQ1_S, IQ1_M, Q2_0
MMQ_PREFILL_TYPES = {8, 13, 16, 17, 19, 29, 42}
_MATRIX_ROW_PADDING = 512


def mmq_tail_bytes(qweight_type: int, K: int) -> int:
    """Bytes the MMQ kernel reads past a weight's last row: ggml pads a row to 512 values."""
    if K % _MATRIX_ROW_PADDING == 0:
        return 0
    block_size, type_size = gguf.GGML_QUANT_SIZES[qweight_type]
    return (_MATRIX_ROW_PADDING - K % _MATRIX_ROW_PADDING) // block_size * type_size


def _mmq_weight_ok(qweight: torch.Tensor, qweight_type: int) -> bool:
    if qweight.dim() != 2 or qweight.dtype != torch.uint8 or qweight.stride(1) != 1 or not qweight.is_cuda:
        return False
    block_size, type_size = gguf.GGML_QUANT_SIZES[qweight_type]
    K = qweight.shape[1] // type_size * block_size
    needed = (qweight.shape[0] - 1) * qweight.stride(0) + qweight.shape[1] + mmq_tail_bytes(qweight_type, K)
    return qweight.untyped_storage().nbytes() - qweight.storage_offset() >= needed


_MMQ_QBUF: dict = {}
_MMQ_STEP: dict = {}


def mmq_qbuf(device: torch.device) -> torch.Tensor:
    """The q8_1 buffer every prefill MMQ call on a device quantizes its activations into (SGLANG_GGUF_MMQ_SLICE_MB),
    made at load: one block the caching allocator keeps, instead of one of each size a call."""
    buf = _MMQ_QBUF.get(device)
    if buf is None:
        buf = _MMQ_QBUF[device] = torch.empty(_MMQ_SLICE_MB << 20, dtype=torch.uint8, device=device)
    return buf


def mmq_mul_mat(x: torch.Tensor, qweight: torch.Tensor, qweight_type: int, out: torch.Tensor) -> None:
    """out[M, rows] = x @ W^T on llama.cpp's MMQ, M in slices whose q8_1 activations fit the device's buffer."""
    from sglang.srt.layers.moe import kt_gguf_moe_mmq

    M, K = x.shape
    qbuf = mmq_qbuf(x.device)
    step = _MMQ_STEP.get(K)
    if step is None:
        step = 64
        while kt_gguf_moe_mmq.q8_bytes(step * 2, K) <= qbuf.numel():
            step *= 2
        while kt_gguf_moe_mmq.q8_bytes(step + 64, K) <= qbuf.numel():
            step += 64
        _MMQ_STEP[K] = step
    for r0 in range(0, M, step):
        kt_gguf_moe_mmq.mul_mat(x[r0 : r0 + step], qweight, qweight_type, out[r0 : r0 + step], qbuf)


def pad_for_mmq(qweight: torch.Tensor, qweight_types) -> "torch.Tensor | None":
    """A copy of a GGUF weight whose storage extends the MMQ tail past its last row (zeros), or None when it
    already does or MMQ does not take it. Done once at load, before any CUDA graph records the weight."""
    if not _MMQ_ON or qweight.dim() != 2 or qweight.dtype != torch.uint8 or not qweight.is_cuda:
        return None
    if not qweight.is_contiguous():
        return None
    tail = 0
    for t in qweight_types:
        if t in MMQ_PREFILL_TYPES:
            block_size, type_size = gguf.GGML_QUANT_SIZES[t]
            tail = max(tail, mmq_tail_bytes(t, qweight.shape[1] // type_size * block_size))
    n = qweight.numel()
    if tail == 0 or qweight.untyped_storage().nbytes() - qweight.storage_offset() >= n + tail:
        return None
    buf = torch.zeros(n + tail, dtype=torch.uint8, device=qweight.device)
    padded = buf[:n].view(qweight.shape)
    padded.copy_(qweight)
    return padded


_DEQUANT_SLICE_MB = int(os.environ.get("SGLANG_GGUF_DEQUANT_SLICE_MB", "256") or 1 << 20)


def mmq_takes(x: torch.Tensor, qweight: torch.Tensor, qweight_type: int) -> bool:
    """Whether a batch is one for MMQ: prefill-sized, a type it builds, the weight's storage holding its tail."""
    return (_MMQ_ON and x.shape[0] >= _MMQ_ROWS and qweight_type in MMQ_PREFILL_TYPES and x.stride(1) == 1
            and _mmq_weight_ok(qweight, qweight_type))


def dequant_mul_mat(x: torch.Tensor, qweight: torch.Tensor, qweight_type: int) -> torch.Tensor:
    """x @ W^T on a bf16 GEMM, the copy of the weight made in slices of rows so the transient stays small (an
    lm_head's is 1.2 GiB whole). SGLang's a8 MMQ is a decode kernel: 71% of a 27K prefill's GPU time on Q5_K / Q8_0
    dense weights."""
    n = qweight.shape[0]
    step = max(256, (_DEQUANT_SLICE_MB << 20) // (x.shape[1] * x.element_size()) // 256 * 256)
    if step >= n:
        return x @ dequantize_gguf_weight(qweight, qweight_type, x.dtype).T
    out = torch.empty(x.shape[0], n, dtype=x.dtype, device=x.device)
    for r0 in range(0, n, step):
        w = dequantize_gguf_weight(qweight[r0 : r0 + step], qweight_type, x.dtype)
        out[:, r0 : r0 + w.shape[0]] = x @ w.T
    return out




_GROUP_IDS: dict = {}


def _group_ids(G: int, M: int, device) -> torch.Tensor:
    """[M * G, 1] int32: row m*G+g picks group g (kept per (G, M, device); captured decode graphs reuse it)."""
    key = (G, M, device)
    t = _GROUP_IDS.get(key)
    if t is None:
        t = _GROUP_IDS[key] = torch.arange(G, dtype=torch.int32, device=device).repeat(M).view(M * G, 1)
    return t


def grouped_mul_mat_gguf(x: torch.Tensor, qweight: torch.Tensor, qweight_type: int, rank: int):
    """x [M, G, K] times a grouped weight (G groups of rank rows, qweight [G * rank, row bytes]) -> [M, G, rank],
    the weight kept in its GGUF blocks; None when no path here takes it (the caller multiplies group by group)."""
    M, G, K = x.shape
    if M == 0 or qweight.shape[0] != G * rank:
        return None
    # fp16/bf16 rows need no kernel
    if qweight_type in UNQUANTIZED_TYPES:
        w = qweight.view(G, rank, K)
        return torch.bmm(x.transpose(0, 1), w.transpose(1, 2)).transpose(0, 1).contiguous()
    # prefill-sized batches on llama.cpp's MMQ, one group at a time (its q8_1 buffer is one group's
    # activations, sliced by rows)
    if (
        _MMQ_ON
        and M >= _MMQ_ROWS
        and qweight_type in MMQ_PREFILL_TYPES
        and x.stride(2) == 1
        and _mmq_weight_ok(qweight, qweight_type)
    ):
        out = torch.empty(M, G, rank, dtype=x.dtype, device=x.device)
        for g in range(G):
            mmq_mul_mat(x[:, g, :], qweight[g * rank : (g + 1) * rank], qweight_type, out[:, g, :])
        return out
    # otherwise prefill-sized batches dequantize the rows once and run one batched GEMM
    # (sgl_kernel's a8 MMQ a group took 72% of a 27K prefill's GPU time, 10-25)
    if M >= _DEQUANT_ROWS and qweight_type in DEQUANT_TYPES:
        w = dequantize_gguf_weight(qweight, qweight_type, x.dtype)
        w = w.view(G, rank, w.shape[-1])
        return torch.bmm(x.transpose(0, 1), w.transpose(1, 2)).transpose(0, 1).contiguous()
    # every group in one MoE-vector launch (one MMVQ a group was 521 launches a token)
    if qweight_type in MMVQ_QUANT_TYPES and _is_cuda:
        from sgl_kernel import ggml_moe_a8_vec

        xx = x.reshape(M * G, K)
        if xx.stride(1) != 1:
            xx = xx.contiguous()
        w = qweight.view(G, rank, qweight.shape[-1])
        return ggml_moe_a8_vec(xx, w, _group_ids(G, M, x.device), 1, qweight_type, rank, M * G).view(M, G, rank)
    return None




# Q5_K embedding rows in float32, in gguf-py's order of operations
_QK_K = 256
Q5_K_BLOCK = 176  # d (f16), dmin (f16), scales[12], qh[32], qs[128]
_SHIFTS = {}


def _shifts(device):
    # made once a device: a tensor built inside a CUDA graph capture would be a host copy
    if device not in _SHIFTS:
        _SHIFTS[device] = (
            torch.tensor([0, 4], dtype=torch.uint8, device=device).view(1, 1, 1, 2, 1),
            torch.arange(8, dtype=torch.uint8, device=device).view(1, 1, 8, 1),
        )
    return _SHIFTS[device]


def q5k_rows_f32(raw: torch.Tensor) -> torch.Tensor:
    """Q5_K rows (uint8 [n, blocks * 176]) to float32 [n, blocks * 256], in gguf-py's order of operations
    ((d * sc) * q - dmin * m, each a separate float32 op), so the values equal its dequantize."""
    n = raw.shape[0]
    b = raw.reshape(n, -1, Q5_K_BLOCK)
    nb = b.shape[1]
    d = b[..., 0:2].contiguous().view(torch.float16).float()  # [n, nb, 1]
    dmin = b[..., 2:4].contiguous().view(torch.float16).float()
    s = b[..., 4:16].reshape(n, nb, 3, 4)
    s_d, s_m, s_md = s[:, :, 0], s[:, :, 1], s[:, :, 2]
    sc = torch.cat([s_d & 0x3F, (s_md & 0x0F) | ((s_d >> 2) & 0x30)], dim=-1)  # [n, nb, 8]
    mn = torch.cat([s_m & 0x3F, (s_md >> 4) | ((s_m >> 2) & 0x30)], dim=-1)
    dd = (d * sc.float()).unsqueeze(-1)  # [n, nb, 8, 1]
    dm = (dmin * mn.float()).unsqueeze(-1)
    qh = b[..., 16:48]  # [n, nb, 32]
    qs = b[..., 48:176].reshape(n, nb, 4, 32)  # 4 groups of 32 bytes, two sub-blocks each
    shifts4, shifts8 = _shifts(raw.device)
    ql = ((qs.unsqueeze(3) >> shifts4) & 0x0F).reshape(n, nb, 8, 32)
    qhb = (qh.unsqueeze(2) >> shifts8) & 0x01  # [n, nb, 8, 32]
    q = (ql | (qhb << 4)).float()
    return (dd * q - dm).reshape(n, nb * _QK_K)


_Q5K_ROWS_A_STEP = 4096


