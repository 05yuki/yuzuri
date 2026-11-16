#!/usr/bin/env python3
"""Install yuzuri.pth into a virtual environment's site-packages, so every Python process there (SGLang's spawned
schedulers too) runs yuzuri._boot at startup. It does nothing unless YUZURI_PACKAGE=1 (serve.py --package sets it).

  install-pth.py <venv> [--fetched-from <SGLang tree>] [--remove]

--fetched-from copies the sources the recipe's setup fetched into its SGLang tree (manifest.json "fetched": llama.cpp's
MMQ sources) next to Yuzuri's modules that build them.
"""
import glob
import os
import sys

venv = sys.argv[1]
sites = glob.glob(os.path.join(venv, "lib", "python3*", "site-packages"))
if len(sites) != 1:
    sys.exit(f"one site-packages expected under {venv}, found {sites}")
pth = os.path.join(sites[0], "yuzuri.pth")
if "--remove" in sys.argv[2:]:
    if os.path.exists(pth):
        os.remove(pth)
        print(f"removed {pth}")
    sys.exit(0)
here = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python")
if "--fetched-from" in sys.argv:
    import json
    import shutil

    src = os.path.join(sys.argv[sys.argv.index("--fetched-from") + 1], "python", "sglang")
    for rel in json.load(open(os.path.join(here, "yuzuri", "manifest.json"))).get("fetched", []):
        s, d = os.path.join(src, rel), os.path.join(here, "yuzuri", "tree", rel)
        if not os.path.isdir(s):
            sys.exit(f"{s} not found: run the recipe's setup first")
        shutil.copytree(s, d, dirs_exist_ok=True)
        print(f"copied {s} -> {d}")
# line 1 puts the package on sys.path (site skips it when the directory is gone); line 2 starts Yuzuri only where the
# package is still there, so a checkout that was moved or deleted leaves the venv's Python silent
old = f"{here}\nimport yuzuri._boot\n"
text = f"{here}\nimport os; os.path.isfile({os.path.join(here, 'yuzuri', '_boot.py')!r}) and __import__('yuzuri._boot')\n"
if os.path.exists(pth) and open(pth).read() not in (text, old):
    sys.exit(f"{pth} exists for another Yuzuri:\n{open(pth).read()}remove it first (--remove)")
with open(pth, "w") as f:
    f.write(text)
print(f"wrote {pth} -> {here}")
