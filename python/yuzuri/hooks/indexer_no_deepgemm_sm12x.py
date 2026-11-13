"""Where SGLang takes every card from SM100 up for DeepGEMM's indexer kernels, SM12x gets the torch indexer, as Hopper
does (the DeepGEMM of that SGLang has no SM12x build). An SGLang that decides another way is left to decide."""
from yuzuri._finder import after_import
from yuzuri.hooks._seam import wrap

REPLACES = ["srt/layers/attention/dsv4/v41_indexer/__init__.py"]


def _install(m):
    import functools

    import torch

    def make(orig):
        @functools.cache
        def is_sm100_or_newer() -> bool:
            return orig() and torch.cuda.get_device_capability()[0] != 12

        return is_sm100_or_newer

    import inspect

    fn = getattr(m, "is_sm100_or_newer", None)
    try:
        src = inspect.getsource(inspect.unwrap(fn)) if fn is not None else ""
    except (OSError, TypeError):
        src = ""
    if "torch.cuda.get_device_capability()[0] >= 10" not in src:
        # the hook corrects that one expression. An SGLang that decides another way has its own answer for SM12x
        # (later versions run DeepGEMM's arch-12 kernels there): the question is left to it
        return
    wrap(m, "is_sm100_or_newer", [], make, adds=("major != 12",))


after_import("sglang.srt.layers.attention.dsv4.v41_indexer", _install)
