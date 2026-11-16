"""A VocabParallelEmbedding whose table lives in pinned host memory.

The token embedding is read one row a decode token and a prompt's rows a prefill: the GPU
reads the rows straight from pinned host memory (UVA) with a small Triton gather, so the
table's VRAM goes to what needs it. Values are copied, not computed: the output equals the
GPU table's. Out-of-shard rows are zero, as _embed_local_shard gives, and the caller's
all-reduce is unchanged. Graph capture works: the host pointer does not move.
"""
import types

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_rows_from_host_kernel(
    weight_ptr,
    ids_ptr,
    output_ptr,
    embedding_dim,
    vocab_start,
    vocab_end,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    idx = tl.load(ids_ptr + row).to(tl.int64)
    in_range = (idx >= vocab_start) & (idx < vocab_end)
    local = tl.where(in_range, idx - vocab_start, 0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < embedding_dim
    w = weight_ptr.to(tl.int64).to(tl.pointer_type(tl.bfloat16))
    vals = tl.load(w + local * embedding_dim + offs, mask=mask, other=0.0)
    tl.store(
        output_ptr + row.to(tl.int64) * embedding_dim + offs,
        tl.where(in_range, vals, 0.0),
        mask=mask,
    )


def _host_embed_local_shard(self, input_: torch.Tensor) -> torch.Tensor:
    ids = input_.reshape(-1)
    out = torch.empty(
        (*input_.shape, self.embedding_dim), dtype=torch.bfloat16, device=input_.device
    )
    if ids.numel():
        _gather_rows_from_host_kernel[(ids.numel(),)](
            self.weight.data_ptr(),
            ids,
            out,
            self.embedding_dim,
            self.shard_indices.org_vocab_start_index,
            self.shard_indices.org_vocab_end_index,
            BLOCK_D=triton.next_power_of_2(self.embedding_dim),
        )
    if self.output_dtype is not None:
        out = out.to(self.output_dtype)
    return out


def move_all_to_host(model) -> list:
    """every token embedding of model to pinned host memory: [(name, bytes)]. Skipped:
    an LM head (ParallelLMHead), a module reached under two names (an embedding that is also the LM head), a table
    that shares its storage with another parameter, a table with added vocabulary (the gather reads the original
    rows only), anything not a bf16 CUDA tensor."""
    from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding

    seen = {}
    for name, mod in model.named_modules(remove_duplicate=False):
        seen.setdefault(id(mod), []).append(name)
    owners = {}
    for mod in model.modules():
        for p in mod.parameters(recurse=False):
            owners.setdefault(p.data_ptr(), set()).add(id(mod))
    out = []
    for name, mod in model.named_modules():
        if not isinstance(mod, VocabParallelEmbedding) or isinstance(mod, ParallelLMHead):
            continue
        w = getattr(mod, "weight", None)
        si = getattr(mod, "shard_indices", None)
        if w is None or si is None or len(seen.get(id(mod), ())) > 1 or len(owners.get(w.data_ptr(), ())) > 1:
            continue
        if int(getattr(si, "num_added_elements", 0) or 0):
            continue
        n = move_embedding_to_host(mod)
        if n:
            out.append((name, n))
    return out


def move_embedding_to_host(embed) -> int:
    """The embedding table to pinned host memory, its GPU storage freed. Returns the bytes
    moved (0 when the table is not a bf16 CUDA tensor)."""
    w = embed.weight
    if not w.is_cuda or w.dtype != torch.bfloat16 or w.numel() == 0:
        return 0
    host = torch.empty(w.shape, dtype=w.dtype, pin_memory=True)
    host.copy_(w.data)
    torch.cuda.synchronize()
    w.data = host
    embed._embed_local_shard = types.MethodType(_host_embed_local_shard, embed)
    torch.cuda.empty_cache()
    return host.numel() * host.element_size()
