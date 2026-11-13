#!/usr/bin/env bash
# Yuzuri as a package on a server already built, its SGLang untouched. Idempotent.
#
#   ./setup.sh --package <the server's venv> [--fetched-from <SGLang tree with llama.cpp's MMQ sources>]
#
# - checks that kt-kernel and the venv's SGLang import with the package attached, without a GPU;
# - installs the tileiras wheel of the venv's own nvcc version;
# - makes, once, the edited copies of six kt-kernel and FlashInfer Python files a launch loads
#   (python/yuzuri/_patched.py; the installed files are not written to), so an edit that does not fit stops here;
# - builds libkv_vmm.so (python/yuzuri/csrc/kv_vmm.c) next to the package's kt_kv_vmm.py;
# - writes yuzuri.pth into the venv (tools/install-pth.py), which does nothing unless serve.py --package starts it.
# - reports what the venv's kt-kernel extension lacks (tools/check-kt-kernel.py): it is not rebuilt here.
set -Eeuo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == "--package" ]]; then
  venv="$(cd "${2:?usage: setup.sh --package <venv>}" && pwd)"
  py="$venv/bin/python"
  [[ -x "$py" ]] || { echo "no python at $py" >&2; exit 2; }
  # the import, as a launch makes it, on the CPU
  # kt-kernel's own libraries as serve.py gives them: a venv inside a recipe checkout has them in its prebuilt/lib
  libs="$(dirname "$venv")/prebuilt/lib"
  [[ -d "$libs" ]] && export LD_LIBRARY_PATH="$libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  if ! CUDA_VISIBLE_DEVICES="" PYTHONPATH="$here/python${PYTHONPATH:+:$PYTHONPATH}" YUZURI_PACKAGE=1 "$py" -c \
       'import kt_kernel, yuzuri._boot, sglang.srt.layers.moe.kt_lend' 2>/tmp/yuzuri-setup-check.$$; then
    tail -50 /tmp/yuzuri-setup-check.$$ >&2; rm -f /tmp/yuzuri-setup-check.$$
    echo "kt-kernel or this venv's SGLang did not import under Yuzuri's package (above); nothing was changed" >&2
    exit 2
  fi
  rm -f /tmp/yuzuri-setup-check.$$
  site="$(ls -d "$venv"/lib/python3.*/site-packages | head -1)"
  loader="$site/kt_kernel/utils/loader.py"
  [[ -f "$loader" ]] || { echo "kt-kernel not installed in $venv" >&2; exit 2; }
  nvcc_ver="$("$py" -c 'import importlib.metadata as m; print(m.version("nvidia-cuda-nvcc"))' 2>/dev/null || true)"
  have="$("$py" -c 'import importlib.metadata as m; print(m.version("nvidia-cuda-tileiras"))' 2>/dev/null || true)"
  if [[ -n "$nvcc_ver" && "$have" != "$nvcc_ver" ]]; then
    if command -v uv >/dev/null 2>&1; then
      uv pip install --python "$py" --no-deps "nvidia-cuda-tileiras==$nvcc_ver"
    else
      "$py" -m pip install --no-deps "nvidia-cuda-tileiras==$nvcc_ver"
    fi
  fi
  # kt-kernel's and FlashInfer's Python files are not written to: a launch makes its copies of six of them with
  # Yuzuri's edits (python/yuzuri/_patched.py). Made here once, so an edit that does not fit this venv stops now
  CUDA_VISIBLE_DEVICES="" PYTHONPATH="$here/python${PYTHONPATH:+:$PYTHONPATH}" YUZURI_PACKAGE=1 "$py" -c '
import importlib.util, yuzuri._boot
from yuzuri import _patched
for name in sorted(_patched.MODULES):
    spec = importlib.util.find_spec(name)
    print(f"{name}: {spec.origin}")'
  # libkv_vmm.so beside the package's kt_kv_vmm.py, where it looks first; built aside and renamed into place, so a
  # running server that has the old library mapped keeps its copy
  src="$here/python/yuzuri/csrc/kv_vmm.c"
  lib="$here/python/yuzuri/tree/srt/layers/moe/libkv_vmm.so"
  if [[ ! -f "$lib" || "$src" -nt "$lib" ]]; then
    inc=""
    for d in "${CUDA_HOME:-/usr/local/cuda}/include" "$site"/nvidia/cu13/include; do
      [[ -f "$d/cuda.h" ]] && { inc="$d"; break; }
    done
    [[ -n "$inc" ]] || { echo "cuda.h not found (CUDA_HOME, or the venv's nvidia/cu13 wheels)" >&2; exit 2; }
    gcc -O2 -Wall -shared -fPIC "$src" -I"$inc" -l:libcuda.so.1 -o "$lib.new"
    mv "$lib.new" "$lib"
    echo "built $lib"
  fi
  "$py" "$here/tools/install-pth.py" "$venv" "${@:3}"
  # the C++ side of kt-kernel is the server's own build: say what it lacks, and what that turns off
  CUDA_VISIBLE_DEVICES="" "$py" "$here/tools/check-kt-kernel.py" || true
  echo "yuzuri ready on $venv: tools/serve.py --package --venv $venv --config-dir <model dir>"
  exit 0
fi
echo "usage: setup.sh --package <the server's venv> [--fetched-from <SGLang tree with llama.cpp's MMQ sources>]" >&2
exit 2
