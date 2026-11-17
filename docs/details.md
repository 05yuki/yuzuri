# Yuzuri: details

What the [README](../README.md) leaves out: how the package sits in SGLang, what the server sizes and when, each
mechanism, the variables the tools set, and how to report a machine.

## The package

`python/yuzuri/` is loaded by `yuzuri.pth` in the venv, in every Python process there (SGLang's spawned schedulers
too).
It does nothing unless `YUZURI_PACKAGE=1`, which `serve.py --package` sets.

- `tree/` holds Yuzuri's own modules, by the path they take inside the `sglang` package; `manifest.json` lists them.
- No SGLang module is replaced. `hooks/` wraps SGLang's functions once their module has loaded.
- Each hook checks what it attaches to: the function's parameters, and the calls inside it the hook relies on.
- A function that gained or moved parameters still takes the hook.
  The hook is bound to the names it works with and the others pass through.
- A function that is gone, or lost a name the hook works with, is handled by what the hook is for
  (`python/yuzuri/_features.py`).
  A hook that only makes the server faster or lighter stays off: it takes back what it had put in and the launch goes on.
  A hook the output depends on stops the launch with the function's name.
- A function that already uses what the hook brings keeps its own: that SGLang carries the same mechanism as a
  patch (the DeepSeek-V4.1 recipe's tree does).
  The hook stays off there and the log names it once.
  The same holds for a module the tree already has under one of Yuzuri's names.
- Your own patches elsewhere in SGLang are left as they are.
- Once the model has loaded, the server log lists every hook under `[yuzuri-features]`.
  `on` is Yuzuri's hook, `tree` is the tree's own mechanism, `off` names the reason, `gone` is a needed hook whose
  SGLang module is not in the tree.
  Each line carries the patch name the DeepSeek-V4.1 recipe's `docs/tier1.md` uses, so the two can be read side by side.
  `PYTHONPATH=python python3 -m yuzuri._features` prints the whole table.
- `manifest.json` records the SGLang commit the hooks were checked on (`sglang_checked`). Nothing compares it.
- `csrc/kv_vmm.c` is the KV library; `setup.sh --package` builds it next to `kt_kv_vmm.py`.

## What is sized, and when

| | from | when |
|---|---|---|
| context and KV pool | the model's config (`max_position_embeddings`) | the launch, never cut |
| the pool's rows on the card | the memory free after the weights | after the first decode, and before each prompt |
| resident hot experts | the minimum, one a layer | the launch |
| lent hot experts | the pool's free pages | every lend, following the routing |
| prefill chunk | the memory free at that moment | every request |
| the chunk ceiling SGLang sizes its buffers by | what a window on these cards held in an earlier launch | the launch |
| CPU threads | the cores of each rank's NUMA node | the launch |
| memory fraction | 1.0, pages mapped as used | fixed |

`plan-launch.py` prints the arithmetic: VRAM, CUDA graphs, weights, KV a token, the hot slot size.
It decides the resident hot count and the threads, and nothing else.

## The mechanisms

**Prefill lend** (`kt_lend.py`).
During a streamed prefill the decoder layers' dense weights give their memory to the prefill's scratch.
That scratch is the streamer's slots and inputs and the kernel workspace.
Outside prefill the scratch holds no memory: its addresses stay and its pages are released.
They come back layer by layer, at the same addresses, just before each layer runs.
The addresses stay, so the decode CUDA graphs stay valid
([torch_memory_saver](https://github.com/fzyzcjy/torch_memory_saver)).
Each layer streams back from a pinned host copy.
Both phase changes wait for the GPU work still reading the weights.
The prefill's output is bit-identical to a plain server's at the same chunk.

**KV pool on remappable memory** (`kt_kv_vmm.py`, `libkv_vmm.so`).
The pool's physical pages can be unmapped and mapped again while the addresses stay.
Pages are mapped as they are used, so a pool made at the model's context costs only what is used.

**KV loan and plan** (`kt_kv_prefill_lend.py`, `kt_prefill_plan.py`).
A prefill window lends the pool's free pages to the prefill.
Pages in use are never lent, and the pool never shrinks.
Each request's chunk is planned from the free memory a window would start with (`kt_ledger.py`).
The plan's cost a row is measured from the first window after a launch, so that window runs at a small chunk.
After it, a chunk may grow to twice the largest window run so far, this launch's or an earlier one's of the same
setup (the lend's records in the cache directory).
The chunk ceiling SGLang sizes some buffers by at startup (the sliding-window and linear-attention pools, DeepSeek V4's
hyper-connection buffers) is the most rows a window on these cards had room for in an earlier launch.
A launch with no record yet takes 36,864.
Every streamed kernel call runs over blocks of 8,192 rows, so the CUTLASS workspaces do not grow with the ceiling.
A long prompt takes fewer, larger chunks, and each chunk streams the experts to the GPU once.
A hybrid model's sliding-window pool is sized for the largest planned chunk and lent outside prefill too.

**Hot lend** (`kt_kv_hot_lend.py`).
Outside prefill the pool's free pages hold more experts a layer on the GPU, and their routes go to the GPU.
Every expert has a slot in address space; how many hold memory follows the free memory at each lend.
Every 32 decode steps, outside the CUDA graph, each layer swaps up to 16 of its slots for experts routed more often.
The copies come straight from kt-kernel's expert memory, on a side stream.
Every slot is given back when a prefill starts or the pool needs its pages.
A lent expert runs in the GPU kernel instead of kt-kernel's, so decode differs from the CPU's by rounding.
The slot formats: bf16, NVFP4 on cuTile (ModelOpt and compressed-tensors names), MXFP4 on Marlin, GGUF rows.

**Host KV** (`kt_kv_vmm.py`).
The pool is made at the model's context in host memory, mapped into the GPU's address space.
Its first rows move to the card as the free memory allows:

- after the first decode, every row the room holds;
- when a prompt is planned, the rows its prefill will write, in what the plan's chunks leave (inside a window too:
  prompts back to back are planned before the last one's window closes);
- after each prompt, the rows the requests hold plus 8,192 tokens.

The room keeps the smallest prefill window free.
A row past the card's share is read and written in host memory, over PCIe.
The allocator hands out the lowest pages first, so requests fill the card's part before host memory.
The loans and the hot lend take card pages only.

**Streamed prefill** (`kt_stream_prefill*.py`).
A prompt above 512 tokens runs every expert on the GPU, in groups copied from kt-kernel's pinned arenas.
The groups are the checkpoint's bytes: bf16, NVFP4 with its e4m3 block scales, MXFP4, GGUF rows.
NVFP4 groups run on FlashInfer's cuTile MoE at the checkpoint's precision.

## NVFP4 on cuTile

`kt_cutile_nvfp4.py` runs a layer's NVFP4 GPU experts on FlashInfer's cuTile MoE from the checkpoint's bytes.

- W4A4 when the checkpoint stores activation scales, W4A16 when it does not.
- Stored activation scales that are all exactly 1.0 are read as none: W4A16.
- `setup.sh` patches FlashInfer's cuTile FP4 MoE: row-major block scales, negative expert ids skipped, split-K for
  few-row GEMMs, and the K tile that pads K least.
  The scale and expert-id parts are [flashinfer-ai/flashinfer#9129](https://github.com/flashinfer-ai/flashinfer/pull/9129).
- cuTile compiles with the `tileiras` wheel of the venv's own `nvcc` version, which `setup.sh` installs.
- `KT_CUTILE_NVFP4=0` gives the layers back to SGLang's own NVFP4 method.

## Serving a model

`tools/serve.py` builds the launch from the host and the checkpoint and starts SGLang:

```sh
tools/serve.py --package --venv <the server's venv> --config-dir <model dir> \
               [--experts DIR | --gguf DIR] [--tp 1|2] [--context T] [--features A,B | --without C] \
               [--dry-run] [--background]
```

- The SGLang served is the one installed in the venv; `--sglang <checkout>` serves another tree instead.
- `--recipe <checkout>` adds the recipe's `prebuilt/lib`, its toolkit wheels and its cache directory.
- `--list-features` prints the features, what each needs and what is always on.
- A feature off has its switches set to their off values over the profile's; `--env` still overrides both.

- `--profile auto`, the default, builds the profile from the checkpoint (`auto_profile.py`): the experts' format
  from one expert's tensor dtypes and scale names, the GPU MoE runner for it, the linear-attention slots when the
  model has linear layers, and the streamed group size.
- The repository keeps no profile of a named model.
- `tools/profile-from-launch.py --pid <launch_server pid>` writes a JSON from a launch running on your own machine,
  with every path and sized value as a placeholder; `--profile FILE.json` launches from it.
- `--hot`, `--chunk` and `--threads` override the plan for an experiment.
- The server log and a manifest (command, environment, plan, package versions) go to `--log-dir/<timestamp>/`.

A hot-expert placement file (`--placement`) chooses the resident experts.
Without one the loader's trivial placement serves.
Which experts are resident moves the NLL by about 0.01 nat; compare NLL only at the same placement and hot count.

## Variables the tools set

A user sets none of these: `serve.py` fills them from the profile and the plan.
They are listed so a log can be read.

- `KT_PREFILL_LEND=1`, `KT_PREFILL_LEND_HINT` (the measured chunk's file).
- `KT_KV_VMM=1`, `KT_KV_VMM_LIB` (the recipe's `build/libkv_vmm.so`).
- `KT_KV_HOT_LEND=1`, `KT_KV_HOT_EXTRA=auto`, `KT_KV_HOT_DYNAMIC_SLOTS=1`, `KT_KV_HOT_DYNAMIC_RESIDENT=1`,
  `KT_KV_HOT_SPARE_FLOOR_MB` (1,024). No free pages are kept aside for new requests: an allocation that needs
  pages the hot slots took gets back those pages and no more (`KT_KV_HOT_PARTIAL=0` gives the whole lend back
  instead, and `KT_KV_HOT_RESERVE` then sets the tokens kept free; a GGUF pack, whose slots are per rank,
  runs that way).
- `KT_GPU_STREAM_PREFILL=auto`: the prompt length the streamed prefill starts from is measured (where a window's
  fixed cost, timed at the first launch, meets the resident path's cost a token, read from the prompts the server is
  given) and carried to the next launch; a number sets it, `0` turns streaming off. The format's streamer switches.
- `KT_GPU_STREAM_DROP_CACHE` (the model directory: its page cache dropped before the host KV is made).
- `SGLANG_UNBALANCED_LOAD_TIMEOUT_S` (1,800: a rank pinning its experts can take minutes longer than the other).
- Diagnosis only: `KT_PREFILL_LEND_DEBUG=1`, `KT_HOST_KV_DEBUG=1`, `KT_PREFILL_PLAN_DEBUG=1`,
  `KT_KV_HOT_CHECK=<n>` (a lent layer's output against the CPU's every n eager calls).

## Requirements and limits

- Linux and NVIDIA GPUs, one or more, of any memory size; the toolkit the base server was built with.
- Host memory for the experts, plus the host KV: the whole pool at the model's context.
- `expandable_segments` is refused by torch_memory_saver.
- One request runs at a time by default; `serve.py --max-running-requests` raises it (README, Limits).
- Measured on one host with SM120 cards of 16 GB: TP=2 over two sockets, TP=1 on one card and one socket.
- Not tested: EP, DP attention, PP.

## Reporting your hardware

`tools/diag.py` prints what Yuzuri reads from a machine: CPU, ISA flags, caches, NUMA nodes, GPUs with their
compute capability, VRAM, PCIe link and NUMA node, driver, OS, glibc and the recipe's package versions.

```sh
python tools/diag.py --recipe <recipe checkout> --model-dir <model directory>
```

- `--bench` adds about 30 seconds of copy-rate measurements. Run it with no server up.
- `--json` prints the report as JSON.
- The report holds no hostname, GPU UUID, user name or home path.
