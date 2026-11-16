#!/usr/bin/env python3
"""Yuzuri's own GPU parts on this machine's GPU: kv_vmm's host/card split and row placement (host KV), its shadow ranges
with the lazily mapped RoPE tables and their dtype copies, and the host embedding's Triton gather. Each part is checked
against plain torch on the same values; a few hundred MiB of VRAM, seconds. Run it with no server up.

usage (after ./setup.sh <venv>, which builds libkv_vmm.so beside kt_kv_vmm.py and writes yuzuri.pth):
    YUZURI_PACKAGE=1 KT_KV_VMM=1 KT_KV_VMM_LIB=<yuzuri>/python/yuzuri/tree/srt/layers/moe/libkv_vmm.so \
        <venv>/bin/python tools/test_gpu_parts.py
"""
import os
import sys
import types

import torch

from sglang.srt.layers.moe import kt_kv_vmm as kv

dev = 0
torch.cuda.set_device(dev)
print("GPU:", torch.cuda.get_device_name(dev), "cc", torch.cuda.get_device_capability(dev), "torch", torch.__version__)
ok = True


def check(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'} {name} {extra}")


# 1. host KV: a pool tensor made in host memory, its first rows moved to the card and back, the values kept
kv.HOST = {"rows": {4097}, "rows_of_pool": 4097, "full": 4096 * 256}
shape = (2, 4097, 4, 256)  # K and V stacked, a token-dim pool tensor
t = kv._host_tensor(shape, torch.bfloat16, dev)
ref = torch.randn(shape, dtype=torch.bfloat16, device=dev)
t.copy_(ref)
torch.cuda.synchronize()
check("host KV: written in host memory", torch.equal(t, ref), f"(card bytes {kv.device_bytes(t)})")
for rows in (1024, 4097, 2048, 0, 3000):
    n = kv._set_rows(t, rows)
    torch.cuda.synchronize()
    same = torch.equal(t, ref)
    check(f"host KV: first {rows} rows of each block on the card", n >= 0 and same, f"({n} bytes moved)")
t[:, 100:200] += 1
ref[:, 100:200] += 1
torch.cuda.synchronize()
check("host KV: a write after placement", torch.equal(t, ref))

# 2. lazily mapped RoPE tables and their dtype copies
from sglang.srt.layers import lazy_rope_table as lr

os.environ["SGLANG_ROPE_LAZY"] = "1"
lr.INITIAL_ROWS, lr.STEP = 4096, 4096
table = torch.randn(65536, 128, dtype=torch.float32, device=dev)
mod = torch.nn.Module()
mod.register_buffer("cos_sin_cache", table.clone(), persistent=False)


def cos_sin_cache_as(self, dtype):
    return self.cos_sin_cache.to(dtype)


type(mod).cos_sin_cache_as = cos_sin_cache_as
lr.INITIAL_ROWS = 4096
freed = lr.install(mod)
check("RoPE: installed", freed == table.numel() * 4, f"({freed >> 20} MiB no longer resident)")
lazy = mod.cos_sin_cache
check("RoPE: first rows mapped", torch.equal(lazy[:4096], table[:4096]))
check("RoPE: unmapped rows read zero", float(lazy[30000:30016].abs().sum()) == 0.0)
bf = mod.cos_sin_cache_as(torch.bfloat16)
fb = types.SimpleNamespace(seq_lens_cpu=torch.tensor([40000]), forward_mode=None)
lr.ensure(fb)
torch.cuda.synchronize()
check("RoPE: grown to the sequence", torch.equal(lazy[:40000], table[:40000]))
check("RoPE: the bf16 copy grew with it", torch.equal(bf[:40000], table[:40000].to(torch.bfloat16)),
      "(same tensor object a captured graph would hold: %s)" % (mod.cos_sin_cache_as(torch.bfloat16) is bf))

# 3. the token embedding read from pinned host memory by the Triton gather
from sglang.srt.layers import host_embedding as he

vocab, dim = 32000, 2816
emb = torch.nn.Module()
emb.weight = torch.nn.Parameter(torch.randn(vocab, dim, dtype=torch.bfloat16, device=dev), requires_grad=False)
emb.embedding_dim, emb.output_dtype = dim, None
emb.shard_indices = types.SimpleNamespace(org_vocab_start_index=8000, org_vocab_end_index=24000)
w0 = emb.weight.detach().clone()
n = he.move_embedding_to_host(emb)
check("embedding: moved to host", n == vocab * dim * 2 and not emb.weight.is_cuda, f"({n >> 20} MiB)")
ids = torch.randint(0, vocab, (4096,), device=dev)
got = emb._embed_local_shard(ids)
inr = (ids >= 8000) & (ids < 24000)
want = torch.zeros(4096, dim, dtype=torch.bfloat16, device=dev)
want[inr] = w0[ids[inr] - 8000]  # the table holds this rank's shard: row = id - the shard's first id
check("embedding: rows equal the card's table, out-of-shard rows zero", torch.equal(got, want))
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
