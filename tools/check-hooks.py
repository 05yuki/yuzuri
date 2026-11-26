#!/usr/bin/env python3
"""Whether every Yuzuri hook attaches to the venv's SGLang, on the CPU, before any launch.

Run with the server venv's python (the SGLang that will serve: the venv's own, or a tree on PYTHONPATH):

    <venv>/bin/python tools/check-hooks.py

It starts the package as a launch does, imports every SGLang module a hook waits for and prints the startup report:
each hook on, the tree's own, off or gone (README, "Checked against"). No GPU is used; the only files written are
the edited kt-kernel and FlashInfer copies a launch makes too (YUZURI_CACHE).
Exit 0 when no hook is off or gone, 1 when one is, 2 when a module does not import.
"""
import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    package = os.path.join(os.path.dirname(HERE), "python")
    if os.environ.get("YUZURI_PACKAGE") != "1":
        # the venv's yuzuri.pth starts the package when the interpreter starts, which it does only with
        # YUZURI_PACKAGE=1 and this checkout first on the path: run again so
        env = dict(os.environ, YUZURI_PACKAGE="1", CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                   PYTHONPATH=os.pathsep.join([package] + [p for p in [os.environ.get("PYTHONPATH")] if p]))
        env.pop("YUZURI_RUN_DIR", None)  # no hooks-<pid>.json
        os.execve(sys.executable, [sys.executable, os.path.abspath(__file__)] + sys.argv[1:], env)
    import yuzuri._boot  # noqa: F401  (the finder a launch installs; the venv's yuzuri.pth may have started it)
    from yuzuri import _features

    try:
        import sglang
    except Exception as e:  # noqa: BLE001
        print(f"sglang: does not import in this python ({type(e).__name__}: {e})", file=sys.stderr)
        return 2
    print(f"sglang: {os.path.dirname(sglang.__file__)}")
    mods = sorted({m for ms in _features._WAITS.values() for m in ms if not m.endswith(".*")})
    failed = []
    for m in mods:
        try:
            importlib.import_module(m)
        except Exception as e:  # noqa: BLE001
            failed.append(f"{m} ({type(e).__name__}: {str(e)[:200]})")
    print(f"modules the hooks wait for: {len(mods) - len(failed)} of {len(mods)} imported")
    for f in failed:
        print(f"  does not import: {f}")
    _features.report(force=True)
    if failed:
        return 2
    return 1 if any(r[3] in ("off", "gone") for r in _features.rows()) else 0


if __name__ == "__main__":
    sys.exit(main())
