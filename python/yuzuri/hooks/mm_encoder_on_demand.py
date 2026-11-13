"""The multimodal encoder on demand (layers/mm_on_demand.py): an encoder call brings the model's on-demand parts to
the card and frees them after. The embedding function SGLang's scheduler calls is handed over wrapped, for a model
that has such parts only; a cache hit calls nothing and moves nothing."""
from yuzuri._finder import after_import
from yuzuri.hooks._seam import wrap

REPLACES = ["srt/managers/mm_schedule.py"]


class _OnCard:
    """A bound embedding method that runs under mm_on_demand.on_gpu; __self__ and __name__ stay what SGLang reads."""

    def __init__(self, func):
        self._func = func
        self.__self__ = func.__self__
        self.__name__ = getattr(func, "__name__", None)

    def __call__(self, items):
        from sglang.srt.layers import mm_on_demand

        return mm_on_demand.call(self._func, items)

    def __getattr__(self, name):
        return getattr(self._func, name)


def _install(m):
    def make(orig):
        def get_embedding_and_mask(data_embedding_func, *args, **kwargs):
            from sglang.srt.layers import mm_on_demand

            owner = getattr(data_embedding_func, "__self__", None)
            if owner is not None and mm_on_demand._STATE.get(id(owner)):
                data_embedding_func = _OnCard(data_embedding_func)
            return orig(data_embedding_func, *args, **kwargs)

        return get_embedding_and_mask

    wrap(m, "get_embedding_and_mask", ["data_embedding_func", "..."], make, adds=("mm_on_demand",))


after_import("sglang.srt.managers.mm_schedule", _install)
