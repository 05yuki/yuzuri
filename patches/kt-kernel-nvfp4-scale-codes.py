#!/usr/bin/env python3
"""kt-kernel's NVFP4 experts keep the checkpoint's e4m3 block-scale bytes (the Python side; the extension needs
GeneralMOEConfig.scale_codes / *_scale_luts and BufferB.codes_in, a kt-kernel build with the scale-code change).

The loader handed each block scale over folded with the per-tensor global into bf16 (a rounding the FP32 arena
then widened). It now hands over the e4m3 bytes untouched and, per expert matrix, a float32[256] table: the e4m3
value of each byte times the global (ModelOpt's weight_scale_2, or 1 / compressed-tensors' weight_global_scale).
NativeMoEWrapper passes both and sets scale_codes; an extension without codes_in keeps the bf16 path.

Usage: python kt-kernel-nvfp4-scale-codes.py <site-packages/kt_kernel>
"""
import sys
from pathlib import Path

K = Path(sys.argv[1])
MARK = "kt-kernel-nvfp4-scale-codes.py"
n = 0
# the stop message an earlier version of this patch wrote, in a kt-kernel it patched then
amx = K / "utils/amx.py"
s = amx.read_text()
stale = '''"(dev repo patches/kt-nvfp4-scale-codes.py)")'''
if stale in s:
    amx.write_text(s.replace(stale, '''"(a kt-kernel build with the scale-code change); without "
                                       "KT_SCALE_ARENA_CODES=1 the experts take the bf16 scale path")'''))
    n += 1
for rel, old, new, *prev in (
    ("utils/loader.py", '''                sdst[exp_id] = self._fold_scales_to_bf16(bs, gs)
''', '''                if codes:
                    # kt-kernel-nvfp4-scale-codes.py: the e4m3 bytes as they are, and their values x the global
                    sdst[exp_id] = bs.contiguous().view(torch.uint8)
                    ldst[exp_id] = _e4m3_table(bs.device) * gs.to(torch.float32).reshape(())
                else:
                    sdst[exp_id] = self._fold_scales_to_bf16(bs, gs)
''', '''                # kt-kernel-nvfp4-scale-codes.py: the e4m3 bytes as they are, and their values x the global
                sdst[exp_id] = bs.contiguous().view(torch.uint8)
                ldst[exp_id] = _e4m3_table(bs.device) * gs.to(torch.float32).reshape(())
'''),
    ("utils/loader.py", '''        for exp_id in range(expert_count):
            for proj, wdst, sdst in (
                (gate_name, gate_weights, gate_scales),
                (up_name, up_weights, up_scales),
                (down_name, down_weights, down_scales),
            ):
                w = self.load_tensor(f"{prefix}.{exp_id}.{proj}.{getattr(self, 'WEIGHT_SUFFIX', 'weight')}", device).contiguous()''',
     '''        luts = {"gate": [None] * expert_count, "up": [None] * expert_count, "down": [None] * expert_count}
        codes = _ext_takes_codes()  # kt-kernel-nvfp4-scale-codes.py
        for exp_id in range(expert_count):
            for proj, wdst, sdst, ldst in (
                (gate_name, gate_weights, gate_scales, luts["gate"]),
                (up_name, up_weights, up_scales, luts["up"]),
                (down_name, down_weights, down_scales, luts["down"]),
            ):
                w = self.load_tensor(f"{prefix}.{exp_id}.{proj}.{getattr(self, 'WEIGHT_SUFFIX', 'weight')}", device).contiguous()''', '''        luts = {"gate": [None] * expert_count, "up": [None] * expert_count, "down": [None] * expert_count}
        for exp_id in range(expert_count):
            for proj, wdst, sdst, ldst in (
                (gate_name, gate_weights, gate_scales, luts["gate"]),
                (up_name, up_weights, up_scales, luts["up"]),
                (down_name, down_weights, down_scales, luts["down"]),
            ):
                w = self.load_tensor(f"{prefix}.{exp_id}.{proj}.{getattr(self, 'WEIGHT_SUFFIX', 'weight')}", device).contiguous()'''),
    ("utils/loader.py", '''        print(f"[NVFP4SafeTensorLoader] Loaded {expert_count} experts from {prefix}")
        return {
            "gate": gate_weights,
            "up": up_weights,
            "down": down_weights,
            "gate_scale": gate_scales,
            "up_scale": up_scales,
            "down_scale": down_scales,
        }''', '''        print(f"[NVFP4SafeTensorLoader] Loaded {expert_count} experts from {prefix}")
        return {
            "gate": gate_weights,
            "up": up_weights,
            "down": down_weights,
            "gate_scale": gate_scales,
            "up_scale": up_scales,
            "down_scale": down_scales,
            "gate_lut": luts["gate"] if codes else None,  # kt-kernel-nvfp4-scale-codes.py
            "up_lut": luts["up"] if codes else None,
            "down_lut": luts["down"] if codes else None,
        }''', '''        print(f"[NVFP4SafeTensorLoader] Loaded {expert_count} experts from {prefix}")
        return {
            "gate": gate_weights,
            "up": up_weights,
            "down": down_weights,
            "gate_scale": gate_scales,
            "up_scale": up_scales,
            "down_scale": down_scales,
            "gate_lut": luts["gate"],  # kt-kernel-nvfp4-scale-codes.py
            "up_lut": luts["up"],
            "down_lut": luts["down"],
        }'''),
    ("utils/loader.py", '''class NVFP4SafeTensorLoader(SafeTensorLoader):''', '''def _ext_takes_codes():
    """kt-kernel-nvfp4-scale-codes.py: whether the installed extension reads one-byte e4m3 scale codes
    (GeneralMOEConfig.scale_codes); an extension without them takes the scales folded to bf16."""
    try:
        from kt_kernel.utils.amx import MOEConfig
    except Exception:
        return False
    return hasattr(MOEConfig, "scale_codes")


def _e4m3_table(device="cpu"):
    """kt-kernel-nvfp4-scale-codes.py: the float32 value of each e4m3 byte (the two NaN codes as 0)."""
    t = torch.arange(256, dtype=torch.uint8, device=device).view(torch.float8_e4m3fn).to(torch.float32)
    return torch.nan_to_num(t, nan=0.0)


class NVFP4SafeTensorLoader(SafeTensorLoader):'''),
    ("utils/amx.py", '''            elif self.method == "NVFP4":
                # e4m3 block scale x per-tensor global, folded to bf16 by the loader.
                assert self.gate_scales[0].dtype == torch.bfloat16, "Expected bf16 scales for NVFP4"''',
     '''            elif self.method == "NVFP4":
                # kt-kernel-nvfp4-scale-codes.py: the checkpoint's e4m3 bytes and a table per matrix
                self._scale_luts = (weights.get("gate_lut"), weights.get("up_lut"), weights.get("down_lut"))
                if self._scale_luts[0] is not None and not hasattr(MOEConfig, "scale_codes"):
                    raise RuntimeError("NVFP4 e4m3 scale codes need kt_kernel_ext with GeneralMOEConfig.scale_codes "
                                       "(a kt-kernel build with the scale-code change); without "
                                       "KT_SCALE_ARENA_CODES=1 the experts take the bf16 scale path")
                want = torch.uint8 if self._scale_luts[0] is not None else torch.bfloat16
                assert self.gate_scales[0].dtype == want, f"Expected {want} scales for NVFP4"'''),
    ("utils/amx.py", '''        moe_config.gate_scales = gate_scale_ptrs
        moe_config.up_scales = up_scale_ptrs
        moe_config.down_scales = down_scale_ptrs

        # Infer group_size from scale shape (column-major layout)
''', '''        moe_config.gate_scales = gate_scale_ptrs
        moe_config.up_scales = up_scale_ptrs
        moe_config.down_scales = down_scale_ptrs

        # Infer group_size from scale shape (column-major layout)
        luts = getattr(self, "_scale_luts", None) if self.method == "NVFP4" else None
        if luts is not None and luts[0] is not None:  # kt-kernel-nvfp4-scale-codes.py
            moe_config.scale_codes = True
            moe_config.gate_scale_luts = [[t.data_ptr() for t in luts[0]]]
            moe_config.up_scale_luts = [[t.data_ptr() for t in luts[1]]]
            moe_config.down_scale_luts = [[t.data_ptr() for t in luts[2]]]
'''),
):
    p = K / rel
    s = p.read_text()
    if new in s:
        continue
    prev = prev[0] if prev else None
    if prev and s.count(prev) == 1:
        p.write_text(s.replace(prev, new, 1))
        n += 1
        continue
    if s.count(old) != 1:
        raise SystemExit(f"anchor found {s.count(old)} times in {p}:\\n{old[:200]}")
    p.write_text(s.replace(old, new, 1))
    n += 1
print(f"patched {n} places under {K}" if n else f"already patched {K}")
