"""the parts of a multimodal model the text path does not reach live in pinned host memory
and come to the card only while the model's encoder runs (tier-1 06 for any model)."""
import contextlib
import logging
import os

import torch

logger = logging.getLogger(__name__)

ENABLED = os.environ.get("SGLANG_MM_ON_DEMAND", "0") == "1"
MIN_BYTES = int(os.environ.get("SGLANG_MM_ON_DEMAND_MIN_MB", "32")) << 20
_STATE = {}  # id(model) -> {"parts": [part], "depth": int}


def _nbytes(mod) -> int:
    return sum(p.numel() * p.element_size() for p in mod.parameters() if p.is_cuda)


def _encoder_modules(model):
    """The modules the model's own encoder methods (get_*_feature[s]) reach through self.<attr>[.<attr>...],
    following the model's own methods they call (Qwen3.5: get_image_feature -> _get_visual_feature)."""
    import inspect
    import re

    found = {}
    todo = [n for n in dir(type(model))
            if n.startswith("get_") and (n.endswith("_feature") or n.endswith("_features"))]
    seen = set()
    while todo:
        name = todo.pop()
        if name in seen or len(seen) > 64:
            continue
        seen.add(name)
        fn = getattr(type(model), name, None)
        try:
            src = inspect.getsource(fn)
        except (OSError, TypeError):
            continue
        for chain in re.findall(r"self\.((?:[A-Za-z_]\w*\.)*[A-Za-z_]\w*)", src):
            head = chain.split(".")[0]
            if inspect.isfunction(getattr(type(model), head, None)):
                todo.append(head)  # one of the model's own methods: read it too
                continue
            obj = model
            for part in chain.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
                if isinstance(obj, torch.nn.Module):
                    found.setdefault(id(obj), obj)
                    break
    return list(found.values())


def _candidates(model):
    """What the encoder methods reach that holds no decoder layer, no token embedding and no LM head."""
    from sglang.srt.layers.moe import kt_lend
    from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding

    layers = {id(l) for l in kt_lend._decoder_layers(model)}
    if not layers:
        return []
    out = []
    for mod in _encoder_modules(model):
        if any(id(m) in layers or isinstance(m, VocabParallelEmbedding) for m in mod.modules()):
            continue
        if _nbytes(mod) >= MIN_BYTES:
            out.append(mod)
    # a module inside another candidate goes with it
    inner = {id(m) for c in out for m in c.modules() if m is not c}
    return [c for c in out if id(c) not in inner]


def install(model) -> int:
    """After loading: the candidates' parameters to pinned host memory. Returns the bytes moved."""
    if not ENABLED or id(model) in _STATE:
        return 0
    parts, total = [], 0
    for mod in _candidates(model):
        store = []
        for p in mod.parameters():
            if not p.is_cuda or p.numel() == 0:
                continue
            host = torch.empty(p.shape, dtype=p.dtype, pin_memory=True)
            host.copy_(p.data)
            store.append((p, host))
        if not store:
            continue
        part = {"mod": mod, "store": store, "resident": False, "pinned": False,
                "bytes": sum(h.numel() * h.element_size() for _, h in store), "dev": store[0][0].device}
        for p, _ in store:
            p.data = torch.empty(0, dtype=p.dtype, device=p.device)
        mod.register_forward_pre_hook(lambda m, args, _part=part, _model=model: _guard(_model, _part))
        parts.append(part)
        total += part["bytes"]
        logger.info(f"on demand: {type(mod).__name__} ({part['bytes'] / 2**20:.1f} MiB, {len(store)} parameters) "
                    "moved to pinned host memory, on the card while an encoder runs")
    if not parts:
        return 0
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    _STATE[id(model)] = {"parts": parts, "depth": 0}
    try:
        from sglang.srt.layers.moe import kt_prefill_plan

        kt_prefill_plan.on_demand("multimodal encoder", total)
    except Exception:
        pass
    return total


def _bring(part) -> None:
    if part["resident"]:
        return
    for p, host in part["store"]:
        p.data = torch.empty(host.shape, dtype=host.dtype, device=part["dev"])
        p.data.copy_(host, non_blocking=True)
    part["resident"] = True


def _make_room(parts) -> None:
    """Under host KV the card's free memory is the pool's rows (kt_kv_vmm): the rows go back to host memory for what
    the encoder's parts are short of. A 1M pool's rows left 7 MiB free on each card and the encoder's first 20 MiB
    did not fit (11-11)."""
    need = {}
    for part in parts:
        if not part["resident"]:
            dev = torch.device(part["dev"])
            need[dev.index] = need.get(dev.index, 0) + part["bytes"]
    if not need:
        return
    try:
        from sglang.srt.layers.moe import kt_kv_vmm
    except ImportError:
        return
    for index, nbytes in need.items():
        if index is None or not any(t.device.index == index for t in getattr(kt_kv_vmm, "HOST_MADE", ())):
            continue
        torch.cuda.empty_cache()
        short = nbytes + (64 << 20) - kt_kv_vmm.card_free(index)
        if short > 0:
            kt_kv_vmm.host_yield(index, short, why="the multimodal encoder")


def _drop(part) -> None:
    if not part["resident"] or part["pinned"]:
        return
    for p, host in part["store"]:
        p.data = torch.empty(0, dtype=host.dtype, device=part["dev"])
    part["resident"] = False


def _guard(model, part) -> None:
    """A part called outside an encode is on the text path: it stays on the card from now on."""
    st = _STATE.get(id(model))
    if st is None or st["depth"] > 0 or part["pinned"]:
        return
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"SGLANG_MM_ON_DEMAND: {type(part['mod']).__name__} runs in a captured text forward, so it is not an "
            "encoder-only part; its parameters cannot come back inside a graph capture. Set SGLANG_MM_ON_DEMAND=0.")
    _bring(part)
    part["pinned"] = True
    logger.warning(f"on demand: {type(part['mod']).__name__} ran outside an encoder; kept on the card "
                   f"({part['bytes'] / 2**20:.1f} MiB)")


@contextlib.contextmanager
def on_gpu(model):
    """Around an encoder call: the model's on-demand parts on the card, freed after (stream-ordered)."""
    st = _STATE.get(id(model)) if model is not None else None
    if not st:
        yield
        return
    if st["depth"] == 0:
        try:
            from sglang.srt.layers.moe import kt_kv_prefill_lend

            kt_kv_prefill_lend.hot_release("multimodal encoder")
        except Exception:
            pass
        _make_room(st["parts"])
        for part in st["parts"]:
            _bring(part)
    st["depth"] += 1
    try:
        yield
    finally:
        st["depth"] -= 1
        if st["depth"] == 0:
            for part in st["parts"]:
                _drop(part)


def call(func, items):
    """mm_schedule: data_embedding_func(items) with its model's on-demand parts on the card."""
    with on_gpu(getattr(func, "__self__", None)):
        return func(items)
