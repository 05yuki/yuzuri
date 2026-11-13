#!/usr/bin/env python3
"""FlashInfer's cuTile FP4 grouped GEMMs (flashinfer/fused_moe/cutile/fp4.py), for Yuzuri's NVFP4 experts.

1. Row-major block scales: the weight-tile loader reads a [E, N, K/16] e4m3 scale tensor as it is (a checkpoint's
   layout, and kt-kernel's one-byte scale codes) as well as the 128x4-swizzled 5-D one FlashInfer prepares on
   SM100+. cuTile compiles a kernel per argument type, so the scale tensor's rank picks the load; nothing else
   changes and the outputs are the same bytes.
2. Until flashinfer-ai/flashinfer#9129: _grouped_gemm_w4a16 and _grouped_gemm_w4a4 compile their M-block loop only
   for the persistent grid (tileiras 13.4 ignores the occupancy hint on sm_120 when the gathers sit in a loop with
   a runtime trip count; NVIDIA/cutile-python#117).

3. Routes with a negative expert id are skipped (fused_moe/cutile/moe.py's permute): a KT layer hands its GPU
   experts the routes the CPU serves as -1. They took rows before, at the block before expert 0's.
   The combine (moe.py) reads no row of a zero-weight route: those rows were never written (6).
4. The FP4 runners (fused_moe/runners.py) take the row-major scale shapes on SM100+ as well.
5. Their W4A4 fallback tactic on row-major scales widens the K tile (same N tile and occupancy, up to 256) from 192
   rows an expert: a K tile of 64 or 128 gives 4- or 8-byte scale rows, too narrow for TMA (16K
   tokens: 12.5 -> 9.0 ms).

Usage: python flashinfer-cutile-fp4.py <site-packages/flashinfer>
"""
import sys
from pathlib import Path

p = Path(sys.argv[1]) / "fused_moe/cutile/fp4.py"
s = p.read_text()
n = 0


def one(text, old, new, start=0, end=None):
    end = len(text) if end is None else min(len(text), end + 1)  # a slice ends on the blank lines' first newline
    i = text.find(old, start, end)
    if i < 0 or text.find(old, i + 1, end) >= 0:
        raise SystemExit(f"anchor found {text.count(old, start, end)} times in {p}:\n{old[:200]}")
    return text[:i] + new + text[i + len(old):]


# 1. row-major block scales
SWZ = """    weight_scale_swizzled = ct.reshape(
        ct.load(
            weight_scale,
            index=(expert, n_block, k_tile, 0, 0),
            shape=(1, tile_n // 128, groups_per_tile // 4, 32, 16),
            padding_mode=ct.PaddingMode.ZERO,
            latency=3,
            allow_tma=True,
        ),
        (tile_n // 128, groups_per_tile // 4, 32, 16),
    )
    decoded_scale = ct.permute(_unswizzle_32_4_4(weight_scale_swizzled), (1, 0))
    return weight, decoded_scale
"""
ROW = """    if weight_scale.ndim == 3:
        # flashinfer-cutile-fp4.py: row-major block scales [E, N, K/16]
        decoded_scale = ct.permute(
            ct.reshape(
                ct.load(
                    weight_scale,
                    index=(expert, n_block, k_tile),
                    shape=(1, tile_n, groups_per_tile),
                    padding_mode=ct.PaddingMode.ZERO,
                    latency=3,
                ),
                (tile_n, groups_per_tile),
            ),
            (1, 0),
        )
    else:
""" + "".join(("    " + ln if ln.strip() and not ln.startswith("    return") else ln)
              for ln in SWZ.splitlines(keepends=True))
if "flashinfer-cutile-fp4.py: row-major block scales" not in s:
    a = s.index("\ndef _load_w4a4_weight_tile(")
    s = one(s, SWZ, ROW, a, s.index("\n\n\n", a))
    n += 1

# 2. the M-block loop only on the persistent grid (flashinfer-ai/flashinfer#9129)
LOOP = "    for iteration in range(num_iterations):\n"
NEW_LOOP = (
    "    # Only the persistent grid gives a CTA several M blocks. Otherwise the grid\n"
    "    # covers every live block, and the loop is compiled out: tileiras 13.4\n"
    "    # ignores the occupancy hint on sm_120 when the gathers sit in a loop with\n"
    "    # a runtime trip count.\n"
    "    for iteration in range(num_iterations if PERSISTENT else 1):\n"
)
if "PERSISTENT: ConstBool" not in s:
    for impl, kernels in (
        ("_grouped_gemm_w4a16_impl", ("_grouped_gemm_w4a16", "_grouped_gemm_w4a16_i64")),
        ("_grouped_gemm_w4a4_impl", ("_grouped_gemm_w4a4", "_grouped_gemm_w4a4_i64")),
    ):
        a = s.index(f"\ndef {impl}(")
        s = one(s, "    USE_INT64: ConstBool,\n):\n", "    USE_INT64: ConstBool,\n    PERSISTENT: ConstBool,\n):\n",
                a, s.index("\n\n\n", a))
        s = one(s, LOOP, NEW_LOOP, a, s.index("\n\n\n", a))
        for k in kernels:
            a = s.index(f"\ndef {k}(")
            s = one(s, f"\n):\n    {impl}(\n", f"\n    PERSISTENT: ConstBool,\n):\n    {impl}(\n", a, s.index("\n\n\n", a))
            flag = "True" if k.endswith("_i64") else "False"
            s = one(s, f"        {flag},\n    )\n", f"        {flag},\n        PERSISTENT,\n    )\n", a, s.index("\n\n\n", a))
    PERSIST = """    if k <= _PERSISTENT_MAX_K and num_assignment_rows >= _PERSISTENT_MIN_ASSIGNMENTS:
        n_blocks = (n + config.tile_n - 1) // config.tile_n
"""
    NEW_PERSIST = """    persistent = (
        k <= _PERSISTENT_MAX_K and num_assignment_rows >= _PERSISTENT_MIN_ASSIGNMENTS
    )
    if persistent:
        n_blocks = (n + config.tile_n - 1) // config.tile_n
"""
    for fn in ("_grouped_gemm", "_grouped_gemm_w4a16_launch"):
        a = s.index(f"\ndef {fn}(")
        s = one(s, PERSIST, NEW_PERSIST, a, s.index("\n\n\n", a))
    a = s.index("\ndef _grouped_gemm(")
    s = one(s, "        input_sorted,\n        output_sorted,\n    )\n",
            "        input_sorted,\n        output_sorted,\n        persistent,\n    )\n", a, s.index("\n\n\n", a))
    a = s.index("\ndef _grouped_gemm_w4a16_launch(")
    tail = "            weight_global_scale.ndim == 2,\n        ),\n    )\n"
    s = one(s, tail, tail.replace("\n        ),", "\n            persistent,\n        ),", 1), a, s.index("\n\n\n", a))
    if s.count("PERSISTENT: ConstBool") != 6 or s.count("persistent,\n") != 2:
        raise SystemExit(f"{p}: the M-block loop edit is incomplete")
    n += 1

if n:
    p.write_text(s)

# 3. negative expert ids skipped by the permute
pm = Path(sys.argv[1]) / "fused_moe/cutile/moe.py"
m = pm.read_text()
if "flashinfer-cutile-fp4.py: a negative id" not in m:
    for fn, old in (
        ("_lane_ids", "    ids = ct.where(valid, ids, ct.full((chunk_size,), -1, dtype=ct.int32))\n"),
        ("_permute_small", "    ids = ct.where(valid, ids, ct.full((CHUNK_SIZE,), -1, dtype=ct.int32))\n"),
    ):
        a = m.index(f"\ndef {fn}(")
        m = one(m, old, old + "    valid = valid & (ids >= 0)  # flashinfer-cutile-fp4.py: a negative id takes no row\n",
                a, m.index("\n\n\n", a))
    pm.write_text(m)
    n += 1

# 6. the combine reads no row of a zero-weight route (a skipped route's row was never written: 0 x NaN is NaN)
m = pm.read_text()
COMBINE = """        values = ct.gather(
            Y,
            ((token * top_k + expert_slot) * H + ct.minimum(h_offsets, H - 1),),
            padding_value=0,
        )
        accumulator = accumulator + ct.astype(values, ct.float32) * weight
"""
if "flashinfer-cutile-fp4.py: a zero-weight route" not in m:
    a = m.index("\ndef _combine_impl(")
    m = one(m, COMBINE, "        if weight != 0.0:  # flashinfer-cutile-fp4.py: a zero-weight route's row may be unwritten\n"
            + "".join("    " + ln if ln.strip() else ln for ln in COMBINE.splitlines(keepends=True)),
            a, m.index("\n\n\n", a))
    pm.write_text(m)
    n += 1

# 4. the FP4 runners take row-major block scales on every arch (part 1 reads them)
pr = Path(sys.argv[1]) / "fused_moe/runners.py"
r = pr.read_text()
CHECK = "        if tuple(w1_scale.shape) != expected_s1 or tuple(w2_scale.shape) != expected_s2:\n"
if "flashinfer-cutile-fp4.py: row-major scales" not in r:
    r = one(r, CHECK,
            "        row_major = (tuple(w1_scale.shape), tuple(w2_scale.shape)) == (linear_s1, linear_s2)"
            "  # flashinfer-cutile-fp4.py: row-major scales\n"
            "        if not row_major and (\n"
            "            tuple(w1_scale.shape) != expected_s1 or tuple(w2_scale.shape) != expected_s2\n"
            "        ):\n")
    pr.write_text(r)
    n += 1

# 5. the W4A4 fallback tactic: the default's N tile and occupancy with the K tile that pads K least. On row-major
# scales from 192 rows an expert (a bound the bucket fingerprint already splits on) the widest of those (up to 256),
# so a tile's scale rows are 16 bytes or more (TMA) instead of 4 or 8; fewer rows pay more in CTAs than they save in
# scale loads (32 experts, a quarter of the routes on the GPU: 1024 tokens 2.40 -> 2.55 ms, 2048
# 2.15 -> 1.96, 8192 6.85 -> 4.89). Flash-Next's GEMM2 (K 320) on 256 pads a tile to 512, on 128 to 384: a decode
# call 28.7 -> 26.7 us, an 8192-row group 1.196 -> 1.155 ms
r = pr.read_text()
TAIL = """        gemm1 = self._fp4_gemm_configs(
            self._fp4_gemm_problem(
                inputs,
                stage=1,
                block_size=block_size,
                fuse_gemm1=fuse_gemm1,
            )
        )[0]
        gemm2 = self._fp4_gemm_configs(
            self._fp4_gemm_problem(inputs, stage=2, block_size=block_size)
        )[0]
        return (block_size, int(fuse_gemm1), *gemm1, *gemm2)
"""
NEW_TAIL = """        gemm1_configs = self._fp4_gemm_configs(
            self._fp4_gemm_problem(
                inputs,
                stage=1,
                block_size=block_size,
                fuse_gemm1=fuse_gemm1,
            )
        )
        gemm2_configs = self._fp4_gemm_configs(
            self._fp4_gemm_problem(inputs, stage=2, block_size=block_size)
        )
        gemm1, gemm2 = gemm1_configs[0], gemm2_configs[0]
        rows_per_expert = (
            inputs[2].numel() + self.config.routing.num_experts - 1
        ) // self.config.routing.num_experts
        wide = self._activation_fp4 and inputs[5].ndim == 3 and rows_per_expert >= 192
        # flashinfer-cutile-fp4.py: the K tile padding K least (the default's N tile and occupancy); of those on
        # row-major scales from 192 rows an expert the widest, else the default's
        gemm1, gemm2 = (
            min(
                (c for c in configs if (c[0], c[2]) == (d[0], d[2]) and (c[1] <= 256 or c == d)),
                key=lambda c: (-(-k // c[1]) * c[1], -c[1] if wide else c != d),
                default=d,
            )
            for configs, d, k in (
                (gemm1_configs, gemm1, inputs[1].shape[1]),
                (gemm2_configs, gemm2, inputs[7].shape[2] * 2),
            )
        )
        return (block_size, int(fuse_gemm1), *gemm1, *gemm2)
"""
OLD_TAIL5 = """        gemm1, gemm2 = gemm1_configs[0], gemm2_configs[0]
        rows_per_expert = (
            inputs[2].numel() + self.config.routing.num_experts - 1
        ) // self.config.routing.num_experts
        if self._activation_fp4 and inputs[5].ndim == 3 and rows_per_expert >= 192:
            # flashinfer-cutile-fp4.py: row-major scales, the widest K tile
            gemm1, gemm2 = (
                max(
                    (c for c in configs if (c[0], c[2]) == (d[0], d[2]) and c[1] <= 256),
                    key=lambda c: c[1],
                    default=d,
                )
                for configs, d in ((gemm1_configs, gemm1), (gemm2_configs, gemm2))
            )
        return (block_size, int(fuse_gemm1), *gemm1, *gemm2)
"""
if "flashinfer-cutile-fp4.py: the K tile padding K least" not in r:
    a = r.index("\n    def _fp4_fallback_tactic(")
    if "flashinfer-cutile-fp4.py: row-major scales, the widest K tile" in r:  # the earlier part 5
        old5 = NEW_TAIL[:NEW_TAIL.index("        gemm1, gemm2 = gemm1_configs[0]")] + OLD_TAIL5
        r = one(r, old5, NEW_TAIL, a, r.index("\n\n    def ", a + 10))
    else:
        r = one(r, TAIL, NEW_TAIL, a, r.index("\n\n    def ", a + 10))
    pr.write_text(r)
    n += 1
# 7. an activation global scale for W4A4's GEMM1 input: a [1] float32 tensor the quantize kernel reads and
# multiplies the activations by (1 / g) before the block scales, so x / g is quantized from the bf16 input with no
# copy. The weight view's optional "input_inv_global" reaches it as the runner's 11th input. GEMM2's input global
# rides on the alphas (up's times 1 / g2, w2's times g2) and needs nothing here. ModelOpt input_scale,
# compressed-tensors input_global_scale; a dynamic global is written into the same tensor
q7 = Path(sys.argv[1]) / "fused_moe/cutile/fp4.py"
f7 = q7.read_text()
if "flashinfer-cutile-fp4.py: INV_GLOBAL" not in f7:
    a7 = f7.index("\ndef _quantize_fp4_impl(")
    b7 = f7.index("\n\n\n", a7)
    f7 = one(f7, "    SCALE_ROW_MAJOR: ConstBool,\n):\n", "    SCALE_ROW_MAJOR: ConstBool,\n    INV_GLOBAL,\n):\n", a7, b7)
    b7 = f7.index("\n\n\n", a7)
    f7 = one(f7, "    packed, scale = _encode_fp4_groups(groups, TILE_M, TILE_K, IS_MXFP4)\n",
             "    # flashinfer-cutile-fp4.py: INV_GLOBAL, the activations over their global scale\n"
             "    groups = groups * ct.load(INV_GLOBAL, (0,), (1,)).item()\n"
             "    packed, scale = _encode_fp4_groups(groups, TILE_M, TILE_K, IS_MXFP4)\n", a7, b7)
    for kname in ("_quantize_fp4", "_quantize_fp4_i64"):
        a7 = f7.index(f"\ndef {kname}(")
        b7 = f7.index("\n\n\n", a7)
        f7 = one(f7, "    SCALE_ROW_MAJOR: ConstBool,\n):\n", "    SCALE_ROW_MAJOR: ConstBool,\n    INV_GLOBAL,\n):\n", a7, b7)
        b7 = f7.index("\n\n\n", a7)
        f7 = one(f7, "        SCALE_ROW_MAJOR,\n    )\n", "        SCALE_ROW_MAJOR,\n        INV_GLOBAL,\n    )\n", a7, b7)
    a7 = f7.index("\ndef _quantize(")
    b7 = f7.index("\n\n\n", a7)
    f7 = one(f7, "    scale_row_major: bool = False,\n) -> None:\n",
             "    scale_row_major: bool = False,\n    inv_global: torch.Tensor | None = None,\n) -> None:\n", a7, b7)
    b7 = f7.index("\n\n\n", a7)
    f7 = one(f7, "            scale_block_size == _MXFP4_BLOCK_SIZE,\n            scale_row_major,\n        ),\n    )\n",
             "            scale_block_size == _MXFP4_BLOCK_SIZE,\n            scale_row_major,\n"
             "            _ones_like_global(x.device) if inv_global is None else inv_global,\n        ),\n    )\n",
             a7, b7)
    f7 = one(f7, "\ndef _quantize(", """
_ONES = {}


def _ones_like_global(device):
    \"\"\"flashinfer-cutile-fp4.py: INV_GLOBAL's default, 1.0.\"\"\"
    t = _ONES.get(device)
    if t is None:
        t = _ONES[device] = torch.ones((1,), dtype=torch.float32, device=device)
    return t


def _quantize(""")
    a7 = f7.index("\ndef run_moe(")
    b7 = f7.index("\n\n\n", a7)
    f7 = one(f7, "    gemm2_config: GemmConfig,\n) -> torch.Tensor:\n",
             "    gemm2_config: GemmConfig,\n    input_inv_global: torch.Tensor | None = None,\n) -> torch.Tensor:\n",
             a7, b7)
    b7 = f7.index("\n\n\n", a7)
    f7 = one(f7, "        scale_block_size=scale_block_size,\n        scale_row_major=workspace.scale_row_major,\n    )\n"
             "    gemm1_input = workspace.input_q\n",
             "        scale_block_size=scale_block_size,\n        scale_row_major=workspace.scale_row_major,\n"
             "        inv_global=input_inv_global,\n    )\n    gemm1_input = workspace.input_q\n", a7, b7)
    q7.write_text(f7)
    n += 1
pr7 = Path(sys.argv[1]) / "fused_moe/runners.py"
r7 = pr7.read_text()
if "flashinfer-cutile-fp4.py: input_inv_global" not in r7:
    a7 = r7.index("\nclass _CuTileFp4Runner(")
    b7 = r7.index("\nclass CuTileNvfp4Runner(", a7)
    r7 = one(r7, "            w2,\n            w2_scale,\n            w2_global,\n        ]\n",
             "            w2,\n            w2_scale,\n            w2_global,\n        ] + (\n"
             "            # flashinfer-cutile-fp4.py: input_inv_global, GEMM1's activation global (an 11th input)\n"
             "            [view[\"input_inv_global\"]] if \"input_inv_global\" in view else []\n        )\n", a7, b7)
    a7 = r7.index("\n    def forward(", a7)
    b7 = r7.index("\nclass CuTileNvfp4Runner(", a7)
    r7 = one(r7, "        if len(inputs) != 10:\n", "        if len(inputs) not in (10, 11):\n", a7, b7)
    b7 = r7.index("\nclass CuTileNvfp4Runner(", a7)
    r7 = one(r7, "            gemm2_config=self._kernel_module.GemmConfig(g2_n, g2_k, g2_occ),\n        )\n",
             "            gemm2_config=self._kernel_module.GemmConfig(g2_n, g2_k, g2_occ),\n"
             "            input_inv_global=inputs[10] if len(inputs) == 11 else None,\n        )\n", a7, b7)
    pr7.write_text(r7)
    n += 1
# 9. the small-row W4A4 defaults (up to 64 routes) take a K tile half as wide where that pads K less, down to 128:
# Flash-Next's GEMM2 (K 320) on 256 pads to 512, on 128 to 384 (10 routes on the GPU: a call 29.3 -> 26.7 us; 64 is
# slower again, 30.8). A narrowed tactic the tile set rejects keeps the default
r9 = pr.read_text()
if "flashinfer-cutile-fp4.py: a K tile half as wide" not in r9:
    a9 = r9.index("\n    def _fp4_fallback_tactic(")
    b9 = r9.index("\n\n    def ", a9 + 10)
    r9 = one(r9, """                gemm2_problem = self._fp4_gemm_problem(
                    inputs, stage=2, block_size=block_size
                )
                if (
""", """                gemm2_problem = self._fp4_gemm_problem(
                    inputs, stage=2, block_size=block_size
                )
                # flashinfer-cutile-fp4.py: a K tile half as wide where it pads K less
                narrowed = list(stage_tactics)
                for i, k in ((2, gemm1_problem.k), (5, gemm2_problem.k)):
                    t = narrowed[i]
                    if t >= 256 and -(-k // (t // 2)) * (t // 2) < -(-k // t) * t:
                        narrowed[i] = t // 2
                narrowed = tuple(narrowed)
                if (
                    self._fp4_config_rejection_reason(gemm1_problem, narrowed[1:4]) is None
                    and self._fp4_config_rejection_reason(gemm2_problem, narrowed[4:7]) is None
                ):
                    stage_tactics = narrowed
                if (
""", a9, b9)
    pr.write_text(r9)
    n += 1
# 8. split-K for W4A4 GEMMs of a few rows (decode): two CTAs share an output tile, each half its K tiles, writing
# acc x alpha in fp32 to a [2, rows, N] workspace that one more kernel sums in order into the output. One token's
# routes give GEMM1 (K 2560, N 640, 256 x 256 tiles) 3 CTAs an expert running 10 K tiles in turn: 15 us on a 0.8 MB
# expert. Flash-Next's shape, 1 / 4 / 10 routes on the GPU: a call 28.7 -> 20.7, 30.7 -> 22.5, 36.9 -> 29.1 us, the
# same output bytes. More slices gain nothing at 1 route and lose at 10. Only where the grid's most live CTAs fill
# half the SMs or fewer: a GEMM1 (N 3072, 24 CTAs an expert) lost with it at 8 routes, 89.5 -> 94.0.
# A padding slot's row (out of OUT's bounds) stays out of every slice
f8 = p.read_text()
if "flashinfer-cutile-fp4.py: split-K" not in f8:
    a8 = f8.index("\ndef _grouped_gemm_w4a4_impl(")
    b8 = f8.index("\n\n\n", a8)
    f8 = one(f8, "    OUT,\n    top_k,\n", "    OUT,\n    PART,\n    top_k,\n", a8, b8)
    b8 = f8.index("\n\n\n", a8)
    f8 = one(f8, "    global_scale_shard_width,\n    K_IN: ConstInt,\n",
             "    global_scale_shard_width,\n    part_rows,\n    K_IN: ConstInt,\n", a8, b8)
    b8 = f8.index("\n\n\n", a8)
    f8 = one(f8, "    PERSISTENT: ConstBool,\n):\n", "    PERSISTENT: ConstBool,\n    SPLIT_K: ConstInt,\n):\n", a8, b8)
    b8 = f8.index("\n\n\n", a8)
    f8 = one(f8, "            for k_tile in range(num_k_tiles):\n",
             "            # flashinfer-cutile-fp4.py: split-K, the CTA's slice of the K tiles\n"
             "            k_tiles_per_split = (num_k_tiles + SPLIT_K - 1) // SPLIT_K\n"
             "            for k_step in range(k_tiles_per_split):\n"
             "                k_tile = ct.bid(2) * k_tiles_per_split + k_step\n", a8, b8)
    b8 = f8.index("\n\n\n", a8)
    STORE = """            output_rows = m_offsets if OUTPUT_SORTED else slots
            ct.scatter(
                OUT,
                (
                    ct.reshape(output_rows, (TILE_M, 1)),
                    ct.reshape(n_offsets, (1, TILE_N)),
                ),
                ct.astype(accumulator * alpha, OUT.dtype),
                check_bounds=True,
            )
"""
    f8 = one(f8, STORE, """            output_rows = m_offsets if OUTPUT_SORTED else slots
            if SPLIT_K > 1:
                ct.scatter(
                    PART,
                    (
                        # a padding slot's row is out of OUT's bounds; kept out of every slice
                        ct.reshape(
                            ct.where(
                                (output_rows >= 0) & (output_rows < part_rows),
                                ct.bid(2) * part_rows + output_rows,
                                SPLIT_K * part_rows,
                            ),
                            (TILE_M, 1),
                        ),
                        ct.reshape(n_offsets, (1, TILE_N)),
                    ),
                    accumulator * alpha,
                    check_bounds=True,
                )
            else:
                ct.scatter(
                    OUT,
                    (
                        ct.reshape(output_rows, (TILE_M, 1)),
                        ct.reshape(n_offsets, (1, TILE_N)),
                    ),
                    ct.astype(accumulator * alpha, OUT.dtype),
                    check_bounds=True,
                )
""", a8, b8)
    for k8, ann in (("_grouped_gemm_w4a4", ""), ("_grouped_gemm_w4a4_i64", ": ct.IndexedWithInt64")):
        a8 = f8.index(f"\ndef {k8}(")
        for old, new in ((f"    OUT{ann},\n    top_k,\n", f"    OUT{ann},\n    PART{ann},\n    top_k,\n"),
                         ("    global_scale_shard_width,\n    K_IN: ConstInt,\n",
                          "    global_scale_shard_width,\n    part_rows,\n    K_IN: ConstInt,\n"),
                         ("    PERSISTENT: ConstBool,\n):\n", "    PERSISTENT: ConstBool,\n    SPLIT_K: ConstInt,\n):\n"),
                         ("        OUT,\n        top_k,\n", "        OUT,\n        PART,\n        top_k,\n"),
                         ("        global_scale_shard_width,\n        K_IN,\n",
                          "        global_scale_shard_width,\n        part_rows,\n        K_IN,\n"),
                         ("        PERSISTENT,\n    )\n", "        PERSISTENT,\n        SPLIT_K,\n    )\n")):
            f8 = one(f8, old, new, a8, f8.index("\n\n\n", a8))
    a8 = f8.index("\ndef _grouped_gemm(")
    f8 = f8[:a8] + """

_SPLIT_K_MAX_ASSIGNMENTS = 64


def _split_k(
    num_assignment_rows: int, num_experts: int, n_blocks: int, num_k_tiles: int, output_sorted: bool, device
) -> int:
    \"\"\"flashinfer-cutile-fp4.py: split-K slices for a W4A4 GEMM, 1 for none. Two when the most CTAs the grid can
    have live (an M block a route or an expert, whichever is fewer) fill no more than half the SMs.\"\"\"
    if output_sorted or num_assignment_rows > _SPLIT_K_MAX_ASSIGNMENTS or num_k_tiles < 4:
        return 1
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return 2 if 2 * min(num_assignment_rows, num_experts) * n_blocks <= sms else 1


@ct.kernel
def _split_k_reduce(PART, OUT, SPLIT_K: ConstInt, TILE_M: ConstInt, TILE_N: ConstInt):
    \"\"\"flashinfer-cutile-fp4.py: the split-K slices summed in order into OUT.\"\"\"
    m_block = ct.bid(0)
    n_block = ct.bid(1)
    total = ct.zeros((1, TILE_M, TILE_N), dtype=ct.float32)
    for s in range(SPLIT_K):
        total = total + ct.load(
            PART,
            index=(s, m_block, n_block),
            shape=(1, TILE_M, TILE_N),
            padding_mode=ct.PaddingMode.ZERO,
        )
    ct.store(OUT, index=(m_block, n_block), tile=ct.astype(ct.reshape(total, (TILE_M, TILE_N)), OUT.dtype))
""" + f8[a8:]
    a8 = f8.index("\ndef _grouped_gemm(")
    b8 = f8.index("\n\n\n", a8)
    f8 = one(f8, "    launch_args: tuple[object, ...] = (\n",
             "    # flashinfer-cutile-fp4.py: split-K\n"
             "    split_k = _split_k(\n"
             "        num_assignment_rows,\n"
             "        weights.shape[0],\n"
             "        (n + config.tile_n - 1) // config.tile_n,\n"
             "        (k + config.tile_k - 1) // config.tile_k,\n"
             "        output_sorted,\n"
             "        output.device,\n"
             "    )\n"
             "    part = (\n"
             "        torch.empty((split_k * num_assignment_rows, n), dtype=torch.float32, device=output.device)\n"
             "        if split_k > 1\n        else output\n    )\n"
             "    launch_args: tuple[object, ...] = (\n", a8, b8)
    b8 = f8.index("\n\n\n", a8)
    f8 = one(f8, "        output,\n    )\n    launch_args += (top_k,)\n",
             "        output,\n        part,\n    )\n    launch_args += (top_k,)\n", a8, b8)
    b8 = f8.index("\n\n\n", a8)
    f8 = one(f8, "        n // global_scale_shards,\n        k,\n",
             "        n // global_scale_shards,\n        num_assignment_rows,\n        k,\n", a8, b8)
    b8 = f8.index("\n\n\n", a8)
    f8 = one(f8, "        output_sorted,\n        persistent,\n    )\n",
             "        output_sorted,\n        persistent,\n        split_k,\n    )\n", a8, b8)
    b8 = f8.index("\n\n\n", a8)
    f8 = one(f8, """        (grid_m, (n + config.tile_n - 1) // config.tile_n),
        kernel,
        launch_args,
    )
""", """        (grid_m, (n + config.tile_n - 1) // config.tile_n, split_k),
        kernel,
        launch_args,
    )
    if split_k > 1:
        reduce_m, reduce_n = 16, 128
        ct.launch(
            torch.cuda.current_stream(x.device),
            ((num_assignment_rows + reduce_m - 1) // reduce_m, (n + reduce_n - 1) // reduce_n),
            _split_k_reduce,
            (part.view(split_k, num_assignment_rows, n), output, split_k, reduce_m, reduce_n),
        )
""", a8, b8)
    p.write_text(f8)
    n += 1
print(f"patched {n} parts under {Path(sys.argv[1])}" if n else f"already patched {Path(sys.argv[1])}")
