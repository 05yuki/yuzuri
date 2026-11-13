"""The dense (non-expert) weights of a GGUF under the checkpoint's own tensor names: KT_GGUF_DENSE names the GGUF
directory, the GGUF says which converter wrote it (general.architecture), and the mapper for that converter is
the kt_gguf_dense_<name>.py next to this file whose ARCH is that string. No list of models here: a mapper is
found by what it declares, and a GGUF no mapper declares stops the launch with what was read and what exists."""
import glob
import importlib
import os
import pkgutil
import struct
from typing import Iterator, Tuple

import torch


def gguf_dir() -> str:
    return os.environ.get("KT_GGUF_DENSE") or os.environ.get("KT_GGUF_V41_DENSE") or ""


_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}  # GGUF value types; 8 string, 9 array


def _skip(f, vtype: int) -> None:
    if vtype == 8:
        f.seek(struct.unpack("<Q", f.read(8))[0], 1)
    elif vtype == 9:
        etype, n = struct.unpack("<IQ", f.read(12))
        if etype in _SIZES:
            f.seek(_SIZES[etype] * n, 1)
        else:
            for _ in range(n):
                _skip(f, etype)
    else:
        f.seek(_SIZES[vtype], 1)


def architecture(path: str) -> str:
    """general.architecture from the file's key/value header (the tensor table is not read: a pack may carry
    tensor types the installed gguf-py does not know)."""
    files = sorted(glob.glob(os.path.join(path, "*.gguf"))) if os.path.isdir(path) else [path]
    if not files:
        raise FileNotFoundError(f"KT_GGUF_DENSE: no .gguf file in {path}")
    with open(files[0], "rb") as f:
        magic, _version, _tensors, n_kv = struct.unpack("<4sIQQ", f.read(24))
        if magic != b"GGUF":
            raise ValueError(f"KT_GGUF_DENSE: {files[0]} is not a GGUF file")
        for _ in range(n_kv):
            key = f.read(struct.unpack("<Q", f.read(8))[0]).decode("utf-8")
            (vtype,) = struct.unpack("<I", f.read(4))
            if key == "general.architecture" and vtype == 8:
                return f.read(struct.unpack("<Q", f.read(8))[0]).decode("utf-8")
            _skip(f, vtype)
    raise ValueError(f"KT_GGUF_DENSE: {files[0]} has no general.architecture")


def mappers() -> dict:
    found = {}
    # The mappers sit next to this file. Under Yuzuri's finder that is not the package's own directory, so
    # both are listed.
    pkg = importlib.import_module(__package__)
    dirs = list(pkg.__path__) + [os.path.dirname(os.path.abspath(__file__))]
    for name in sorted({info.name for info in pkgutil.iter_modules(dirs)}):
        if not name.startswith("kt_gguf_dense_"):
            continue
        mod = importlib.import_module(f"{__package__}.{name}")
        arch, fn = getattr(mod, "ARCH", None), getattr(mod, "weights", None)
        if isinstance(arch, str) and callable(fn):
            found[arch] = fn
    return found


def weights(path: str, model_config, model) -> Iterator[Tuple[str, torch.Tensor]]:
    arch, have = architecture(path), mappers()
    if arch not in have:
        raise RuntimeError(
            f"KT_GGUF_DENSE: {path} was written by llama.cpp's \"{arch}\" converter and no dense-weight mapper "
            f"declares it (mappers: {', '.join(sorted(have)) or 'none'}). A mapper is a module "
            f"kt_gguf_dense_<name>.py with ARCH = \"{arch}\" and weights(gguf_dir, model_config, model).")
    print(f"[kt-gguf-dense] {path}: architecture \"{arch}\", mapper found", flush=True)
    yield from have[arch](path, model_config, model)
