"""Prefix cache with a sliding window: SGLang frees the SWA slots behind a key it has just published, but a later
match of that key stops its re-prefill tail (swa_reprefill_tail_tokens, window_reprefill_tail_tokens in later
versions: the method is found by the end of its name) short of the key's end and needs the window before that point.
Without it the match falls back to the previous checkpoint (a prompt read again after decode matched only to the
last chunk boundary). The free stops that many tokens earlier. An SGLang that already subtracts the tail is left
alone."""
import inspect
import threading

from yuzuri._finder import after_import
from yuzuri.hooks._seam import SeamError, wrap

REPLACES = ["srt/mem_cache/unified_radix_cache.py"]

_IN = threading.local()
_CALL = "comp.free_out_of_window_slots(req, len(radix_key) - 1, insert_params)"
_TAIL = "reprefill_tail_tokens"


def _install(m):
    cache = m.UnifiedRadixCache
    todo = []
    # later versions keep checkpoint_into_tree's body, the free with it, in _checkpoint_into_tree
    inner = "_checkpoint_into_tree" if inspect.getattr_static(cache, "_checkpoint_into_tree", None) else "checkpoint_into_tree"
    for name, params in ((inner, ["self", "req", "..."]), ("advance_unpublished_req", ["self", "req"])):
        fn = inspect.getattr_static(cache, name, None)
        try:
            src = inspect.getsource(fn) if fn is not None else ""
        except (OSError, TypeError):
            src = ""
        if "- self." in src and _TAIL + "(" in src:
            continue  # theirs already keeps the tail's window
        todo.append((name, params))
    if not todo:
        return

    def make_outer(orig):
        def publishing(self, req, *args, **kwargs):
            before = getattr(_IN, "cache", None)
            _IN.cache = self
            try:
                return orig(self, req, *args, **kwargs)
            finally:
                _IN.cache = before

        return publishing

    def make_free(orig):
        def free_out_of_window_slots(self, req, pre_len, *args, **kwargs):
            c = getattr(_IN, "cache", None)
            if c is not None:
                pre_len = pre_len - int(getattr(c, tail_name)(req) or 0)
            return orig(self, req, pre_len, *args, **kwargs)

        return free_out_of_window_slots

    tails = [n for n in dir(cache) if n.endswith(_TAIL)
             and list(inspect.signature(inspect.getattr_static(cache, n)).parameters)[:2] == ["self", "req"]]
    if len(tails) != 1:
        raise SeamError(f"yuzuri: UnifiedRadixCache has {tails or 'no'} *{_TAIL}(self, req): the prefix cache replay "
                        "hook reads the tail from one such method and has nothing to read it from in this SGLang")
    tail_name = tails[0]
    for name, params in todo:
        wrap(cache, name, params, make_outer, needs=(_CALL,))
    from sglang.srt.mem_cache.unified_cache.components import swa

    wrap(swa.SWAComponent, "free_out_of_window_slots", ["self", "req", "pre_len", "..."], make_free)


after_import("sglang.srt.mem_cache.unified_radix_cache", _install)
