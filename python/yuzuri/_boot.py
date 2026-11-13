"""Run by yuzuri.pth in every Python process of the environment (SGLang's spawned schedulers too). Nothing happens
unless YUZURI_PACKAGE=1."""
import os

if os.environ.get("YUZURI_PACKAGE", "0") == "1":
    from yuzuri import _finder

    _finder.install()
