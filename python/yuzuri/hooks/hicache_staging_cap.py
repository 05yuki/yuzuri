"""HiCache's write-back staging buffer on the card, within SGLANG_HICACHE_STAGING_MB (32) MiB.

A host pool that stages whole pages of all its layers sizes the buffer by a page count alone (64 pages): on a
page_first pool of a 43-layer model that is 330 MiB of device memory held for the launch, which the lends then do
not get. The staged write-back loops over the buffer's capacity, so a smaller buffer writes the same pages in more
rounds. The pools taken are the ones whose _init_write_back_staging_buffers sizes the buffer from the module's
_WRITE_BACK_STAGING_PAGE_CHUNK and keeps no page count beside the buffer (read from the function, no list of names)."""
import inspect
import os

from yuzuri._finder import after_import
from yuzuri.hooks._seam import wrap

REPLACES = ["srt/mem_cache/memory_pool_host.py"]
_NAME = "_WRITE_BACK_STAGING_PAGE_CHUNK"


def _install(m):
    import torch

    cap = int(os.environ.get("SGLANG_HICACHE_STAGING_MB", "32") or 32) << 20
    if not hasattr(m, _NAME):
        return

    def make(orig):
        def _init_write_back_staging_buffers(self):
            full = getattr(m, _NAME)
            try:
                setattr(m, _NAME, 1)  # one page first: its shape gives a page's bytes
                got = orig(self)
                one = getattr(self, "staging_buffer", None)
                if not isinstance(one, torch.Tensor) or one.shape[0] != 1:
                    if one is None:
                        return got
                    setattr(m, _NAME, full)
                    return orig(self)
                pages = max(1, min(full, cap // max(1, one.nbytes)))
                if pages == 1:
                    return got
                self.staging_buffer = None
                del one
                setattr(m, _NAME, pages)
                got = orig(self)
                if pages < full:
                    print(f"[yuzuri] {type(self).__name__}: HiCache stages {self.staging_buffer.shape[0]} pages "
                          f"({self.staging_buffer.nbytes >> 20} MiB on the card), not up to {full}", flush=True)
                return got
            finally:
                setattr(m, _NAME, full)

        return _init_write_back_staging_buffers

    for cls in list(vars(m).values()):
        if not inspect.isclass(cls) or cls.__module__ != m.__name__:
            continue
        fn = cls.__dict__.get("_init_write_back_staging_buffers")
        if fn is None:
            continue
        try:
            src = inspect.getsource(fn)
        except (OSError, TypeError):
            continue
        if _NAME not in src or "staging_page_capacity =" not in src or "self.staging_page_capacity" in src:
            continue
        wrap(cls, "_init_write_back_staging_buffers", ["self"], make, adds=("_staging_page_capacity(",))


after_import("sglang.srt.mem_cache.memory_pool_host", _install)
