---
name: Something broke or ran slow
about: A launch that failed, a wrong output, or a number that looks off
---

**What happened**

**The launch command**

```
tools/serve.py ...
```

**From the run directory** (`--log-dir/<timestamp>/`)

- `manifest.json` (command, plan, package versions)
- one `hooks-<pid>.json` (which hooks went in)
- the `[yuzuri-features]` lines and the last 50 lines of `server.log`

**Machine**

GPU model and count, VRAM, CPU, RAM, SGLang commit (`git -C <sglang> rev-parse HEAD`).

**Does it happen with `--features base`?**
