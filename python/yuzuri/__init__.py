"""Yuzuri: the lend system for KTransformers + SGLang hybrids, as a package an existing SGLang loads at startup.

With YUZURI_PACKAGE=1, yuzuri.pth (site-packages) installs a finder ahead of SGLang's own. It serves:
- Yuzuri's own modules under their SGLang names (sglang.srt.layers.moe.kt_lend and the others);
- Yuzuri's versions of the SGLang modules it changes, each only while the installed SGLang's file is the one the
  version was made from (the git blob recorded in manifest.json). Another file stops the launch with the reason.
"""
