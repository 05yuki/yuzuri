"""The finder that puts Yuzuri's modules into SGLang's namespace, and the startup hooks onto SGLang's own.

manifest.json lists Yuzuri's modules by the name they take inside the sglang package (their files in tree/). No
SGLang module is replaced: after_import(name, fn) runs fn(module) once SGLang's module has executed, and the hooks
(hooks/) wrap the functions Yuzuri's features live in. Each hook checks what it attaches to and stops the launch,
naming it, when this SGLang does not have it or already carries the same mechanism: SGLang's version is not
checked anywhere else. Six files of kt-kernel and FlashInfer are made from the installed ones with Yuzuri's edits
(_patched.py); nothing installed is written to.
"""
import importlib.abc
import importlib.util
import json
import os
import sys

from yuzuri import _patched

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.join(HERE, "tree")
_HOOKS = {}


def _hooks_for(name: str) -> list:
    """The hooks of the module itself, then those registered for its package ("pkg.*")."""
    return _HOOKS.get(name, []) + _HOOKS.get(name.rpartition(".")[0] + ".*", [])


def after_import(name: str, fn) -> None:
    """fn(module) after the module named name has executed (at once if it already has). "pkg.*" is every module
    directly under pkg: a hook that looks for a shape, where the module that has it is not known beforehand."""
    _HOOKS.setdefault(name, []).append(fn)
    if name.endswith(".*"):
        for n, m in list(sys.modules.items()):
            if n.rpartition(".")[0] + ".*" == name and getattr(m, "__spec__", None) is not None:
                fn(m)
    elif name in sys.modules and getattr(sys.modules[name], "__spec__", None) is not None:
        fn(sys.modules[name])


class _HookLoader(importlib.abc.Loader):
    """The module's own loader, then the hooks registered for it."""

    def __init__(self, inner, name):
        self.inner, self.name = inner, name

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        for fn in _hooks_for(self.name):
            fn(module)

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
        if fullname != "sglang" and not fullname.startswith("sglang."):
            if fullname not in _patched.MODULES:
                return None
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
                raise ImportError(f"yuzuri: this SGLang has its own {fullname} ({theirs.origin}): it carries a patch "
                                  f"with the same mechanism. Yuzuri goes on an SGLang without it")
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
