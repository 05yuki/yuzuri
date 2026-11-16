"""The finder that puts Yuzuri's modules into SGLang's namespace, and the startup hooks onto SGLang's own.

manifest.json lists Yuzuri's modules by the name they take inside the sglang package (their files in tree/). No
SGLang module is replaced: after_import(name, fn) runs fn(module) once SGLang's module has executed, and the hooks
(hooks/) wrap the functions Yuzuri's features live in. Each hook checks what it attaches to and stops the launch,
naming it, when this SGLang does not have it. A tree that already carries the same mechanism keeps its own and the hook
stays off, said once in the log. SGLang's version is not
checked anywhere else. Six files of kt-kernel and FlashInfer are made from the installed ones with Yuzuri's edits
(_patched.py); nothing installed is written to.
"""
import importlib.abc
import importlib.util
import json
import os
import re
import sys

from yuzuri import _patched

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.join(HERE, "tree")
_HOOKS = {}


# what a switch Yuzuri sets to a word is as a number, for a tree's own module that reads the switch with int():
# KT_GPU_STREAM_PREFILL=auto is "measure the length, start at the plan's alignment" (kt_stream_threshold.START)
_AS_NUMBER = {"KT_GPU_STREAM_PREFILL": lambda environ: environ.get("KT_PREFILL_PLAN_ALIGN", "1024")}
_INT_READ = re.compile(r"""int\(\s*os\.environ\.get\(\s*["']([A-Z0-9_]+)["']""")


def _numbers_for(fullname: str, origin, environ=None) -> list:
    """The tree's own module runs with its own reading of the environment. A switch it reads with int() and the
    launcher set to a word (auto) is given as the number the word starts from, said once; a word with no number
    stops the launch naming the switch, not as a ValueError inside the tree's import."""
    environ = os.environ if environ is None else environ
    try:
        with open(origin) as f:
            names = sorted(set(_INT_READ.findall(f.read())))
    except (OSError, TypeError):
        return []
    done = []
    for n in names:
        v = (environ.get(n) or "").strip()
        if not v or v.lstrip("+-").isdigit():
            continue
        if n not in _AS_NUMBER:
            raise RuntimeError(f"yuzuri: {fullname} of this SGLang ({origin}) reads {n} as a number and the launch "
                               f"has {n}={v}: give it a number (serve.py --env {n}=<n>)")
        environ[n] = str(_AS_NUMBER[n](environ))
        print(f"[yuzuri] {n}={v} is {environ[n]} for this SGLang's own {fullname}: it reads a number", flush=True)
        done.append(n)
    return done


def _hooks_for(name: str) -> list:
    """The hooks of the module itself, then those registered for its package ("pkg.*")."""
    return _HOOKS.get(name, []) + _HOOKS.get(name.rpartition(".")[0] + ".*", [])


def _run(fn, module) -> None:
    """One hook on its module. A hook that cannot attach stops the launch when the server is wrong without it; an
    optional one (_features.TABLE) takes back what it had put in, stays off and is reported."""
    from yuzuri import _features
    from yuzuri.hooks import _seam

    name = _features._short(fn)
    if name not in _features.TABLE:
        return fn(module)
    if name in os.environ.get("YUZURI_HOOKS_OFF", "").split(",") and _features.optional(name):
        # serve.py --features / --without left out the feature this optional hook carries
        return _features.note(name, "off", "not picked (serve.py --features / --without)")
    mark = _seam.begin(name)
    try:
        fn(module)
    except _seam.SeamError as e:
        if not _features.optional(name):
            raise
        _seam.undo(mark)
        _features.note(name, "off", str(e).removeprefix("yuzuri: "))
        print(f"[yuzuri] {name} stays off (the server is right without it): {e}", flush=True)
    else:
        _features.note(name, "tree" if _seam.carried_since(mark) == "all" else "on")
    finally:
        _seam.end(mark)


def after_import(name: str, fn) -> None:
    """fn(module) after the module named name has executed (at once if it already has). "pkg.*" is every module
    directly under pkg: a hook that looks for a shape, where the module that has it is not known beforehand."""
    from yuzuri import _features

    _HOOKS.setdefault(name, []).append(fn)
    _features.waits(fn, name)
    if name.endswith(".*"):
        for n, m in list(sys.modules.items()):
            if n.rpartition(".")[0] + ".*" == name and getattr(m, "__spec__", None) is not None:
                _run(fn, m)
    elif name in sys.modules and getattr(sys.modules[name], "__spec__", None) is not None:
        _run(fn, sys.modules[name])


class _HookLoader(importlib.abc.Loader):
    """The module's own loader, then the hooks registered for it."""

    def __init__(self, inner, name):
        self.inner, self.name = inner, name

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        for fn in _hooks_for(self.name):
            _run(fn, module)

    def __getattr__(self, attr):  # get_source, get_code, is_package... for inspect, linecache, pkgutil
        return getattr(self.inner, attr)


class Finder(importlib.abc.MetaPathFinder):
    def __init__(self, manifest: dict):
        self.new = {m["module"]: m for m in manifest["modules"]}
        self.checked = False

    def _others(self, fullname, path=None):
        for f in sys.meta_path:
            if f is self or not hasattr(f, "find_spec"):
                continue
            spec = f.find_spec(fullname, path)
            if spec is not None:
                return spec
        return None

    def find_spec(self, fullname, path=None, target=None):
        if sys.meta_path[0] is not self:
            # a finder put in front after this one (a sitecustomize, which runs after the .pth files) that answers for
            # an SGLang module itself leaves the hooks for that module unrun: a storage adapter's finder took
            # sglang.srt.managers.scheduler, and the prefill plan never chose a chunk (a 32,768-token window in one
            # piece, out of memory, 11-14). First again from the next import on; the other finder's loader stays
            # inside the hook loader
            sys.meta_path.remove(self)
            sys.meta_path.insert(0, self)
        if fullname != "sglang" and not fullname.startswith("sglang."):
            if fullname not in _patched.MODULES:
                if not _HOOKS.get(fullname):
                    return None
                # a hook on a module of another package (FlashInfer's prepared calls): named whole, no "pkg.*"
                spec = self._others(fullname, path)
                if spec is not None and spec.loader is not None:
                    spec.loader = _HookLoader(spec.loader, fullname)
                return spec
            # kt-kernel's and FlashInfer's files with Yuzuri's edits: a copy made from the installed file
            theirs = self._others(fullname, path)
            return None if theirs is None else _patched.spec_for(fullname, theirs)
        if not self.checked:
            spec = self._others("sglang")
            if spec is None or not spec.submodule_search_locations:
                raise ImportError("yuzuri: no SGLang package found in this Python: Yuzuri goes on a server already "
                                  "built on SGLang and kt-kernel")
            self.checked = True
        m = self.new.get(fullname)
        if m is not None:
            # a module of this name in SGLang itself is a tree that carries Yuzuri's code already
            theirs = self._others(fullname, path)
            if theirs is not None:
                # the tree's own module runs (11-14 yuki: a tree with the same patch is used as it is, not refused)
                print(f"[yuzuri] this SGLang has its own {fullname} ({theirs.origin}): it carries the same "
                      f"mechanism, its own runs and Yuzuri's copy stays off", flush=True)
                _numbers_for(fullname, theirs.origin)
                return theirs
            spec = importlib.util.spec_from_file_location(fullname, os.path.join(TREE, m["path"]))
        elif _hooks_for(fullname):
            spec = self._others(fullname, path)
        else:
            return None
        if spec is not None and _hooks_for(fullname):
            spec.loader = _HookLoader(spec.loader, fullname)
        return spec


FINDER = None


def install() -> None:
    global FINDER
    if FINDER is not None:
        return
    with open(os.path.join(HERE, "manifest.json")) as f:
        FINDER = Finder(json.load(f))
    sys.meta_path.insert(0, FINDER)
    from yuzuri import hooks  # noqa: F401  (registers the startup hooks)
