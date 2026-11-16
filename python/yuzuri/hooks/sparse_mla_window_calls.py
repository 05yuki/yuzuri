"""FlashInfer's sparse MLA wrapper keeps one prepared call a tensor shape for good, each with its own output buffer
(flashinfer/mla/_sparse_mla_sm120/_prepared.py: wrapper_run, prepare(owned=True)). A prefill window's shapes get
theirs while the window's large blocks are alive, so a buffer of a few MiB lands in a segment of hundreds, and torch
cannot give that segment back when the window ends: the hot slots grow back into less memory (a 2.2 MiB buffer held
690 MiB, 11-15).

The wrapper's two scratch arenas grow the same way: a window's 15 MiB arena held a 238 MiB segment (11-15).

The calls a window prepared are dropped when it ends (kt_lend.WINDOW_END); a later window with the same shape
prepares its own again. What was prepared outside a window (the decode shapes, the ones the graphs were captured
with, which keep buffers of their own) is never touched.

Every wrapper's arenas are dropped at the same moment, whoever made them: an arena only grows, so one made by a
short request before the window is the one the window used and left in a large segment. The next call that needs
one makes it at its own size, after torch gave the window's segments back."""
import inspect
import sys
import weakref

from yuzuri._finder import after_import
from yuzuri.hooks._seam import SeamError, _put

_MADE = []  # (the wrapper, the key of a call it prepared inside the window)
_SEEN = {}  # id(wrapper) -> weakref: every wrapper that ran, for its scratch arenas
_LEND = "sglang.srt.layers.moe.kt_lend"


def drop() -> int:
    n = 0
    for ref, key in _MADE:
        w = ref()
        if w is not None and w._prepared_calls.pop(key, None) is not None:
            n += 1
    _MADE.clear()
    for k, ref in list(_SEEN.items()):
        w = ref()
        if w is None:
            del _SEEN[k]
            continue
        arenas = getattr(w, "_scratch_arenas", None)
        for i in range(len(arenas or ())):
            if arenas[i] is not None:
                arenas[i] = None
                n += 1
    return n


_SAID = []


def _install(m):
    inner = getattr(m, "wrapper_run", None)
    if inner is None or "_prepared_calls" not in inspect.getsource(inner):
        raise SeamError(f"yuzuri: {m.__name__} has no wrapper_run that keeps _prepared_calls: this FlashInfer keeps "
                        f"its prepared calls elsewhere")

    def wrapper_run(wrapper, *a, **kw):
        lend = sys.modules.get(_LEND)
        if lend is None:
            return inner(wrapper, *a, **kw)
        if not hasattr(lend, "WINDOW_END"):  # the tree's own lend has no place to drop the calls: nothing is kept
            if not _SAID:
                _SAID.append(1)
                print(f"[yuzuri] {lend.__name__} is the tree's own and has no WINDOW_END: a prefill window's "
                      f"prepared calls are not reused", flush=True)
            return inner(wrapper, *a, **kw)
        if id(wrapper) not in _SEEN:
            _SEEN[id(wrapper)] = weakref.ref(wrapper)
        if drop not in lend.WINDOW_END:
            lend.WINDOW_END.append(drop)
        if lend.STATE.get("phase") != "prefill":
            return inner(wrapper, *a, **kw)
        calls = wrapper._prepared_calls
        n = len(calls)
        try:
            return inner(wrapper, *a, **kw)
        finally:
            if len(calls) > n:  # a dict keeps insertion order: the new keys are the last
                ref = weakref.ref(wrapper)
                _MADE.extend((ref, k) for k in list(calls)[n:])

    wrapper_run.__wrapped__ = inner
    _put(m, "wrapper_run", wrapper_run)


after_import("flashinfer.mla._sparse_mla_sm120._prepared", _install)
