"""kt-kernel's and FlashInfer's Python files with Yuzuri's edits, made at launch from the files this Python has.

Six files of the two packages carry lines Yuzuri's expert formats need (patches/*.py at the repository's top say
which and why). The installed packages are not written to: the first import of one of the six takes the installed
file (with whatever the user has changed in it), runs the edits on a copy under the cache directory, and the copy
is what the module is made from. An edit whose anchor is not in the file stops the launch with the anchor; a file
that already has every edit is used as it is. A copy is kept by the contents of the file and of the scripts, so a
new version of either makes a new one.
"""
import hashlib
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(HERE)), "patches")
CACHE = os.environ.get("YUZURI_CACHE") or os.path.join(
    os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "yuzuri", "patched")

# package -> (its files the scripts edit, [(script, the path it is given, relative to the package)])
PACKAGES = {
    "kt_kernel": (
        ["experts.py", "utils/amx.py", "utils/loader.py"],
        [("kt-loader-discover-experts.py", "utils/loader.py"),
         ("kt-kernel-clamp-nvfp4.py", ""),
         ("kt-kernel-nvfp4-scale-codes.py", "")],
    ),
    "flashinfer": (
        ["fused_moe/cutile/fp4.py", "fused_moe/cutile/moe.py", "fused_moe/runners.py"],
        [("flashinfer-cutile-fp4.py", "")],
    ),
}
MODULES = {f"{pkg}.{rel[:-3].replace('/', '.')}": (pkg, rel) for pkg, (files, _) in PACKAGES.items() for rel in files}
_made = {}


def _digest(paths: list) -> str:
    h = hashlib.sha256()
    for p in paths:
        with open(p, "rb") as f:
            h.update(hashlib.sha256(f.read()).digest())
    return h.hexdigest()[:24]


def _make(pkg: str, root: str) -> str:
    """The directory holding pkg's edited files (the package's layout), made once per contents."""
    files, scripts = PACKAGES[pkg]
    missing = [s for s, _ in scripts if not os.path.isfile(os.path.join(SCRIPTS, s))]
    if missing:
        raise ImportError(f"yuzuri: {missing} not found in {SCRIPTS}: the package runs from its repository checkout")
    gone = [r for r in files if not os.path.isfile(os.path.join(root, r))]
    if gone:
        raise ImportError(f"yuzuri: {pkg} at {root} has no {gone}: the edits were written for those files")
    key = _digest([os.path.join(root, r) for r in files] + [os.path.join(SCRIPTS, s) for s, _ in scripts])
    out = os.path.join(CACHE, f"{pkg}-{key}")
    if os.path.isdir(out):
        return out
    import shutil
    import subprocess
    import tempfile

    os.makedirs(CACHE, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=f"{pkg}-{key}.", dir=CACHE)
    try:
        for r in files:
            os.makedirs(os.path.dirname(os.path.join(tmp, r)), exist_ok=True)
            shutil.copyfile(os.path.join(root, r), os.path.join(tmp, r))
        for script, rel in scripts:
            # -I: the script reads and writes the copy and nothing else of this environment
            run = subprocess.run([sys.executable, "-I", os.path.join(SCRIPTS, script), os.path.join(tmp, rel)],
                                 capture_output=True, text=True)
            if run.returncode != 0:
                raise ImportError(f"yuzuri: {script} does not apply to this {pkg} ({root}):\n"
                                  f"{(run.stderr or run.stdout).strip()[-2000:]}")
        try:
            os.rename(tmp, out)  # another process of the launch may have made it meanwhile
        except OSError:
            if not os.path.isdir(out):
                raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def spec_for(fullname: str, theirs):
    """The spec a module of MODULES is made from: theirs when the installed file needs no edit, else the copy."""
    pkg, rel = MODULES[fullname]
    origin = getattr(theirs, "origin", None)
    if not origin or not origin.endswith(rel):
        raise ImportError(f"yuzuri: {fullname} is not a file ending in {rel} ({origin}): the edits were written "
                          f"for that file")
    root = origin[: -len(rel)].rstrip(os.sep)
    out = _made.get((pkg, root))
    if out is None:
        out = _made[(pkg, root)] = _make(pkg, root)
    mine = os.path.join(out, rel)
    with open(mine, "rb") as a, open(origin, "rb") as b:
        if a.read() == b.read():
            return theirs
    return importlib.util.spec_from_file_location(fullname, mine)
