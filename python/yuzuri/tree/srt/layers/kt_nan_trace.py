"""KT_NAN_TRACE=1: name the first module whose output is not finite.

A diagnostic, off by default. Every module gets a forward hook; the first outputs that hold a NaN or
an inf are printed with the module's name, whether its inputs were finite, and the shapes. A CUDA
graph replay runs no Python, so this sees prefill and ungraphed decode only; it synchronises on every
module and is slow.
"""

import os

import torch

LIMIT = int(os.environ.get("KT_NAN_TRACE_LINES", "12"))
_seen = [0]
_calls = [0]


def _tensors(x, out):
    if torch.is_tensor(x):
        out.append(x)
    elif isinstance(x, (tuple, list)):
        for v in x:
            _tensors(v, out)
    elif isinstance(x, dict):
        for v in x.values():
            _tensors(v, out)
    return out


def _bad(ts):
    n = 0
    for t in ts:
        if t.is_floating_point() and t.numel():
            n += int((~torch.isfinite(t)).sum())
    return n


def _desc(ts):
    return [f"{tuple(t.shape)} {str(t.dtype).replace('torch.', '')}" for t in ts if t.is_floating_point()][:4]


def install(model):
    def hook_for(name):
        def hook(mod, args, kwargs, output):
            if _seen[0] >= LIMIT or torch.cuda.is_current_stream_capturing():
                return
            _calls[0] += 1
            outs = _tensors(output, [])
            bad = _bad(outs)
            if not bad:
                return
            ins = _tensors(args, []) + _tensors(kwargs, [])
            bad_in = _bad(ins)
            _seen[0] += 1
            print(f"[kt-nan-trace] {name} ({type(mod).__name__}), hooked call {_calls[0]}: {bad} non-finite out "
                  f"{_desc(outs)}, {bad_in} non-finite in {_desc(ins)}"
                  f"{' <- FIRST made here' if not bad_in else ''}", flush=True)

        return hook

    n = 0
    for name, mod in model.named_modules():
        if name:
            mod.register_forward_hook(hook_for(name), with_kwargs=True)
            n += 1
    print(f"[kt-nan-trace] {n} modules hooked; up to {LIMIT} lines", flush=True)
    return n
