"""DeepSeek-V4 RoPE tables mapped as the context grows.

Every V4 attention layer reads one of two complex64 freqs_cis tables (the plain base and the
compressed YaRN base) indexed by position, each built for max_position_embeddings (1M on
V4.1: 256 MiB each), whatever context is served. With SGLANG_DSV4_ROPE_LAZY=1 each table moves
to a kv_vmm shadow range of the same size (one virtual range: the tensor, its views and the
CUDA graphs that read it never move) with physical chunks mapped only for the rows a forward
can reach; a pinned host copy of the table fills each new chunk, so the values are the table's
bytes. ensure(forward_batch) runs before every forward (model_runner, next to
kt_lend.before_forward) and maps up to the batch's longest sequence plus one step.

An unmapped row of a shadow range reads as zero, so a forward must never reach past what
ensure mapped: positions are below seq_len, and capture-time dummy batches stay inside the
first INITIAL_ROWS.
"""
import logging
import os
import time

import torch

logger = logging.getLogger(__name__)

STEP = int(os.environ.get("SGLANG_DSV4_ROPE_LAZY_STEP", "16384"))
INITIAL_ROWS = int(os.environ.get("SGLANG_DSV4_ROPE_LAZY_INITIAL", "65536"))
_TABLES = []


_LEAD = [0]  # largest position minus sequence length seen in a prefill (M-RoPE)


def enabled() -> bool:
    return (os.environ.get("SGLANG_DSV4_ROPE_LAZY", "0") == "1"
            or os.environ.get("SGLANG_ROPE_LAZY", "0") == "1")


def _map(ent, rows: int) -> None:
    dev = ent["dev"]
    rows = min(dev.shape[0], -(-rows // STEP) * STEP)
    have = ent["rows"]
    if rows <= have:
        return
    from sglang.srt.layers.moe import kt_kv_vmm

    rb = ent["row_bytes"]
    if kt_kv_vmm.restore(dev, have * rb, (rows - have) * rb) < 0:
        raise torch.OutOfMemoryError(
            f"RoPE table: mapping rows {have}..{rows} on {dev.device}"
        )
    dev[have:rows].copy_(ent["host"][have:rows])
    # the activation-dtype copies grow with their table
    for d in ent.get("derived", {}).values():
        drb = d["row_bytes"]
        if kt_kv_vmm.restore(d["dev"], have * drb, (rows - have) * drb) < 0:
            raise torch.OutOfMemoryError(f"RoPE table copy: mapping rows {have}..{rows} on {dev.device}")
        d["dev"][have:rows].copy_(d["host"][have:rows])
    ent["rows"] = rows
    if os.environ.get("SGLANG_DSV4_ROPE_LAZY_VERIFY", "0") == "1":
        same = torch.equal(dev[:rows].cpu(), ent["host"][:rows])
        logger.info("RoPE table check: rows 0..%d %s the host copy", rows, "equal" if same else "DIFFER from")


def make_lazy(table: torch.Tensor) -> torch.Tensor:
    """A shadow-range copy of table with its first INITIAL_ROWS mapped."""
    from sglang.srt.layers.moe import kt_kv_vmm

    host = torch.empty(table.shape, dtype=table.dtype, pin_memory=True)
    host.copy_(table)
    dev = kt_kv_vmm.shadow_tensor(table.shape, table.dtype, table.device.index)
    ent = dict(dev=dev, host=host, rows=0, row_bytes=table[0].numel() * table.element_size())
    _map(ent, INITIAL_ROWS)
    torch.cuda.synchronize(table.device)
    _TABLES.append(ent)
    return dev


def _derived(ent, dtype):
    """a lazily mapped copy of ent's table in dtype, mapped as far as the table."""
    from sglang.srt.layers.moe import kt_kv_vmm

    ds = ent.setdefault("derived", {})
    if dtype in ds or ent["dev"].dtype == dtype:
        return ds.get(dtype, {}).get("dev", ent["dev"])
    host = ent["host"].to(dtype).pin_memory()
    dev = kt_kv_vmm.shadow_tensor(ent["dev"].shape, dtype, ent["dev"].device.index)
    d = dict(dev=dev, host=host, row_bytes=host[0].numel() * host.element_size())
    rows = ent["rows"]
    if rows and kt_kv_vmm.restore(dev, 0, rows * d["row_bytes"]) < 0:
        raise torch.OutOfMemoryError(f"RoPE table copy: mapping rows 0..{rows} on {dev.device}")
    dev[:rows].copy_(host[:rows])
    ds[dtype] = d
    return dev


def _lazy_cast(mod, ent) -> None:
    """mod.cos_sin_cache_as(dtype) answers with the lazy copy while its cache is the lazy
    table; a module that rebuilds its cache gets the stock behaviour back."""
    import types

    orig = getattr(type(mod), "cos_sin_cache_as", None)
    if orig is None:
        return

    def cos_sin_cache_as(self, dtype, _ent=ent, _orig=orig):
        cache = self.cos_sin_cache
        if cache.dtype == dtype or cache is not _ent["dev"]:
            return _orig(self, dtype)
        return _derived(_ent, dtype)

    mod.cos_sin_cache_as = types.MethodType(cos_sin_cache_as, mod)
    mod._cos_sin_cache_cast = None  # a full copy made before install is not read again


def ensure(forward_batch) -> None:
    if not _TABLES:
        return
    lens = getattr(forward_batch, "seq_lens_cpu", None)
    if lens is not None and lens.numel():
        need = int(lens.max())
    else:
        lens = getattr(forward_batch, "seq_lens", None)
        if lens is None or not lens.numel():
            return
        need = int(lens.max().item())
    # M-RoPE positions can run ahead of the sequence length (a prefill shows by how much)
    mp = getattr(forward_batch, "mrope_positions", None)
    fm = getattr(forward_batch, "forward_mode", None)
    if mp is not None and fm is not None and fm.is_extend() and mp.numel():
        _LEAD[0] = max(_LEAD[0], int(mp.max().item()) + 1 - need)
    need += STEP + _LEAD[0]
    if all(need <= ent["rows"] or ent["rows"] >= ent["dev"].shape[0] for ent in _TABLES):
        return
    t0 = time.perf_counter()
    for ent in _TABLES:
        _map(ent, need)
    logger.info(
        "RoPE tables: %d rows mapped (%.1f MiB a table) in %.1f ms",
        _TABLES[0]["rows"],
        _TABLES[0]["rows"] * _TABLES[0]["row_bytes"] / 2**20,
        (time.perf_counter() - t0) * 1e3,
    )


def install(model) -> int:
    """Every freqs_cis table of model's modules (buffers and plain attributes) moved to a
    lazily mapped range; returns the device bytes no longer resident."""
    if not enabled() or not torch.cuda.is_available():
        return 0
    found = {}
    lazy = {ent["dev"].data_ptr() for ent in _TABLES}  # not twice
    owners = {}
    for mod in model.modules():
        for name, buf in list(mod._buffers.items()):
            if name == "freqs_cis" and torch.is_tensor(buf) and buf.is_cuda and buf.data_ptr() not in lazy:
                found.setdefault(buf.data_ptr(), buf)
            # SGLang's RoPE cache, when it is worth a range of its own
            if (name == "cos_sin_cache" and torch.is_tensor(buf) and buf.is_cuda and buf.dim() == 2
                    and buf.shape[0] >= 2 * INITIAL_ROWS and buf.data_ptr() not in lazy):
                found.setdefault(buf.data_ptr(), buf)
                owners.setdefault(buf.data_ptr(), []).append(mod)
        v = vars(mod).get("freqs_cis")
        if torch.is_tensor(v) and v.is_cuda:
            found.setdefault(v.data_ptr(), v)
    if not found:
        return 0
    new = {ptr: make_lazy(t) for ptr, t in found.items()}
    for mod in model.modules():
        for name, buf in list(mod._buffers.items()):
            if torch.is_tensor(buf) and buf.is_cuda and buf.data_ptr() in new and buf.shape == new[buf.data_ptr()].shape:
                mod._buffers[name] = new[buf.data_ptr()]
        for k, v in list(vars(mod).items()):
            if k == "_buffers" or k == "_parameters":
                continue
            if torch.is_tensor(v) and v.is_cuda and v.data_ptr() in new and v.shape == new[v.data_ptr()].shape:
                setattr(mod, k, new[v.data_ptr()])
    freed = sum(t.numel() * t.element_size() for t in found.values())
    # cos_sin_cache_as hands out the lazy copy in the asked dtype, made at install for the
    # activation dtypes so that no range is made during a graph capture
    ents = {ent["dev"].data_ptr(): ent for ent in _TABLES}
    for ptr, mods in owners.items():
        ent = ents.get(new[ptr].data_ptr())
        if ent is None:
            continue
        for dt in (torch.bfloat16, torch.float16):
            _derived(ent, dt)
        for mod in mods:
            _lazy_cast(mod, ent)
    try:
        from sglang.kernels.ops.attention.dsv4.rope import precompute_freqs_cis

        precompute_freqs_cis.cache_clear()
    except Exception:
        pass
    found.clear()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return freed
