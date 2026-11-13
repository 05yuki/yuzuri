#!/usr/bin/env python3
"""kt-kernel's safetensors loaders find a layer's experts in the checkpoint instead of trying layouts by name:
the tensor names under `layers.<L>` that carry `.experts.0.` give the experts' path, the projection names and
the weight and scale names, whatever the model calls them (DeepSeek V4 `ffn.experts.{i}.w1/w3/w2.weight/.scale`,
MiMo and Qwen `mlp.experts.{i}.gate_proj/up_proj/down_proj.weight/.weight_scale`, Mixtral-style
`block_sparse_moe.experts`). The roles come from the names themselves (gate / up / down, llama.cpp's w1 / w3 / w2):
no list of models.

Usage: python kt-loader-discover-experts.py <kt_kernel/utils/loader.py>
"""
import sys
from pathlib import Path

P = Path(sys.argv[1])
s = P.read_text()


def once(src, old, new):
    assert src.count(old) == 1, old[:80]
    return src.replace(old, new, 1)


def extras(s):
    """The changes after the discovery itself; each checks its own text, so a loader patched by an earlier
    version of this script gets them as well."""
    # the scales under the suffix the checkpoint has (MXFP4: `.scale` for DeepSeek V4, `.weight_scale` for MiMo)
    s = s.replace('''self.load_tensor(f"{prefix}.{exp_id}.{proj}.scale", device)''',
                  '''self.load_tensor(f"{prefix}.{exp_id}.{proj}.{getattr(self, 'SCALE_SUFFIX', 'scale')}", device)''')
    # the weights under the name the checkpoint has (compressed-tensors: `weight_packed`)
    if "self.WEIGHT_SUFFIX = " not in s:
        s = once(s, '''        self._discovered_prefix = prefix
        self.PROJ_NAMES = names
''', '''        self._discovered_prefix = prefix
        self.PROJ_NAMES = names
        g0 = f"{prefix}.0.{names[0]}"
        self.WEIGHT_SUFFIX = ("weight_packed" if not self.has_tensor(f"{g0}.weight")
                              and self.has_tensor(f"{g0}.weight_packed") else "weight")
''')
    s = s.replace('''.{gate_name}.weight"):''', '''.{gate_name}.{getattr(self, 'WEIGHT_SUFFIX', 'weight')}"):''')
    s = s.replace('''self.load_tensor(f"{prefix}.{exp_id}.{proj}.weight", device)''',
                  '''self.load_tensor(f"{prefix}.{exp_id}.{proj}.{getattr(self, 'WEIGHT_SUFFIX', 'weight')}", device)''')
    # NVFP4's second-level scale: ModelOpt's weight_scale_2 multiplies, compressed-tensors' weight_global_scale
    # divides
    old_gs = '''                gs = self.load_tensor(f"{prefix}.{exp_id}.{proj}.weight_scale_2", device)
'''
    if old_gs in s and "the reciprocal of ModelOpt's weight_scale_2" not in s:
        s = once(s, old_gs, '''                if self.has_tensor(f"{prefix}.{exp_id}.{proj}.weight_scale_2"):
                    gs = self.load_tensor(f"{prefix}.{exp_id}.{proj}.weight_scale_2", device)
                else:
                    # compressed-tensors: weight_global_scale, the reciprocal of ModelOpt's weight_scale_2
                    gs = 1.0 / self.load_tensor(
                        f"{prefix}.{exp_id}.{proj}.weight_global_scale", device).to(torch.float32)
''')
    s = s.replace('''("weight", "weight_scale", "weight_scale_2")''',
                  '''("weight", "weight_packed", "weight_scale", "weight_scale_2", "weight_global_scale")''')
    return s


MARK = "def _discover_experts(self, base_key"
if MARK in s:
    t = extras(s)
    if t != s:
        P.write_text(t)
    print(f"{'updated' if t != s else 'already patched'} {P}")
    raise SystemExit(0)


# the discovery on the base class: every loader that tries prefixes gets it
# (before prefetch_layer where a tree has that method, else at the end of the base class)
AT = "    def prefetch_layer(self, prefix, suffixes, expert_count, proj_names):"
if AT not in s:
    AT = "class FP8SafeTensorLoader(SafeTensorLoader):"
s = once(s, AT, '''    _ROLE_WORDS = {"gate": ("gate", "w1"), "up": ("up", "w3"), "down": ("down", "w2")}

    def _discover_experts(self, base_key: str):
        """(experts path, (gate, up, down) names, scale suffix) read from the tensor names of this layer, or
        None when the checkpoint has no `.experts.0.` under it. `base_key` is `<anything>.layers.<L>`: the
        layer index is what is matched, so a model's own prefix (`model.`, `language_model.model.`, none)
        does not matter."""
        import re

        m = re.search(r"layers\\.(\\d+)$", base_key)
        if m is None:
            return None
        layer = m.group(1)
        pat = re.compile(r"^((?:.*\\.)?layers\\." + layer + r"\\.(?:[^.]+\\.)*experts)\\.0\\.([^.]+)\\.([^.]+)$")
        found = {}
        for key in self.tensor_file_map:
            mm = pat.match(key)
            if mm is None:
                continue
            prefix, proj, suffix = mm.groups()
            found.setdefault(prefix, {}).setdefault(proj, set()).add(suffix)
        if not found:
            return None
        if len(found) > 1:
            # a prefix that also matches a stripped form of itself: the longest is the real one
            prefix = max(found, key=len)
        else:
            prefix = next(iter(found))
        projs = found[prefix]
        roles = {}
        for proj in projs:
            low = proj.lower()
            for role, words in self._ROLE_WORDS.items():
                if any(w in low.split("_") or low.startswith(w) or low.endswith(w) for w in words):
                    roles.setdefault(role, proj)
        if set(roles) != {"gate", "up", "down"}:
            return None
        suffixes = projs[roles["gate"]]
        scale = next((x for x in ("scale", "weight_scale", "weight_scale_inv", "scales") if x in suffixes), None)
        return prefix, (roles["gate"], roles["up"], roles["down"]), scale

    def _apply_discovered(self, base_key: str) -> None:
        """Point this loader's layout at what the checkpoint holds for the layer (a no-op when it holds
        nothing recognisable: the loader's own layout stays)."""
        got = self._discover_experts(base_key)
        if got is None:
            return
        prefix, names, scale = got
        layer = base_key.rsplit(".", 1)[-1]
        # the loader formats EXPERTS_PATH_TPL with base=<...>.layers.<L>: the part after the layer is what varies
        self.EXPERTS_PATH_TPL = "{base}" + prefix[prefix.index(f"layers.{layer}") + len(f"layers.{layer}"):]
        self._discovered_prefix = prefix
        self.PROJ_NAMES = names
        if scale is not None:
            self.SCALE_SUFFIX = scale

''' + ("" if AT.startswith("    def") else "\n") + AT)

# every loader that tries prefixes: the discovery first (the MiMo probe, where a tree has it, becomes it)
if "_use_mimo_layout_if_present" in s:
    s = once(s, '''    def _use_mimo_layout_if_present(self, base_key: str) -> None:
        tpl, names, suffix = self.MIMO_LAYOUT
        if self.has_tensor(f"{tpl.format(base=base_key)}.0.{names[0]}.weight"):
            self.EXPERTS_PATH_TPL, self.PROJ_NAMES, self.SCALE_SUFFIX = tpl, names, suffix
''', '''    def _use_mimo_layout_if_present(self, base_key: str) -> None:
        # kt-loader-discover-experts.py: the checkpoint's own names, whatever the model
        self._apply_discovered(base_key)
''')
hook = '''        gate_name, up_name, down_name = self.PROJ_NAMES
        prefix = None
        expert_count = 0
        for cand in self._experts_prefix_candidates(base_key):'''
assert hook in s, "load_experts hook"
s = s.replace(hook, '''        self._apply_discovered(base_key)  # kt-loader-discover-experts.py
        gate_name, up_name, down_name = self.PROJ_NAMES
        prefix = None
        expert_count = 0
        for cand in self._experts_prefix_candidates(base_key):''')

# every loader's candidate list starts with the discovered prefix
s = s.replace('''    def _experts_prefix_candidates(self, base_key: str) -> list[str]:
        candidates = [self.EXPERTS_PATH_TPL.format(base=base_key)]
''', '''    def _experts_prefix_candidates(self, base_key: str) -> list[str]:
        candidates = [self.EXPERTS_PATH_TPL.format(base=base_key)]
        if getattr(self, "_discovered_prefix", None):
            candidates.insert(0, self._discovered_prefix)  # kt-loader-discover-experts.py
''')
s = extras(s)
P.write_text(s)
print(f"patched {P}")
