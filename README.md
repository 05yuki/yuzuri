# Yuzuri

Yuzuri is a speed mod for KTransformers + SGLang.

KTransformers + SGLang serves large mixture-of-experts models on small GPUs, with the experts in host memory.

Yuzuri adds three things to that server:

- faster long prompts
- faster decode
- the model's full context, up to 1,048,576 tokens

It is a Python package that an unchanged SGLang loads at startup.
No SGLang or kt-kernel file is replaced.

It serves models of 26B to over 300B parameters.

The number of cards, their memory and the host's sockets are read at launch.
None of them is assumed.

## What it does

- **Prefill lend.**
  The dense weights and the prefill's buffers take turns on the card.
  During a long prompt the decoder layers' dense weights give their VRAM to the prefill.
  They come back layer by layer, at the same addresses.
  During decode the prefill's buffers give their VRAM back: the streamed experts' slots, their inputs and the kernel workspace.
  The hot lend fills that memory with experts.

- **KV loan and plan.**
  The KV pool's free pages go to the prefill too.
  Each request's chunk is sized to its prompt, from the memory free at that moment.

- **Hot lend.**
  Between prompts the pool's free pages hold more experts on the GPU.
  The slots follow the routing.
  A prompt takes back only the slots its chunk needs; the rest keep their experts through the prefill.

- **Host KV.**
  The pool is made at the model's context.
  The rows a card cannot hold sit in pinned host memory.
  A prompt's own rows are on the card before its prefill reads them.

- **Streamed prefill.**
  A prompt's experts stream to the GPU from the CPU's own copy, in the checkpoint's format.

One command starts the server.

Nothing is sized by hand:

- no memory fraction
- no context to cut
- no hot-expert count to pick

## What it needs

- A server already built on KTransformers + SGLang: SGLang with kt-kernel's CPU experts, in one Python venv.

- A checkpoint with MoE experts.
  - Yuzuri moves experts and KV pages between the CPU and the cards.
  - A dense model has no experts on the CPU, and the launch stops saying so.

- SGLang as upstream ships it, or with your own patches.
  - Yuzuri replaces no SGLang file. It attaches to the functions its features live in when the server starts.
  - The launch prints one line for every hook: on, the tree's own, off, gone or idle.
  - An SGLang that already carries one of Yuzuri's mechanisms keeps its own.
  - A function that has moved or changed shape switches off a hook that only adds speed.
  - It stops the launch when the output would be wrong without the hook, and names what is gone.
  - The commits it was run on are under [Checked against](#checked-against).

- Linux and NVIDIA GPUs (CUDA), one or more, of any memory size. AMD GPUs are not supported.
  - What a card cannot hold stays in host memory: experts, and the KV rows past the card.
  - Measured on SM120 cards of 16 GB.

- Host memory for the experts and for the KV past the cards.

[deepseek-v4.1-flash-1017-ktransformers](https://github.com/05yuki/deepseek-v4.1-flash-1017-ktransformers)
(tag `2026-11-05-0936`) builds a venv with kt-kernel and the toolkit.

Its SGLang is patched with mechanisms Yuzuri brings itself, so Yuzuri stops on it.
Put an unpatched SGLang checkout ahead of it: `PYTHONPATH=<sglang checkout>/python`, for `setup.sh` and for the launch.

## Demo recipe

[yuzuri-deepseek-v4.1-flash-1017](https://github.com/05yuki/yuzuri-deepseek-v4.1-flash-1017) is a complete setup built on Yuzuri.

- It runs DeepSeek-V4.1-Flash-1017 on SGLang + KTransformers, with Yuzuri at a pinned commit.
- Its SGLang carries five patches, all in the model's own files.
- Everything else the launch needs comes from Yuzuri's hooks.
- `setup.sh` builds the venv, `serve.sh` starts the server and `tools/qualify.sh` checks and measures it.
- Its README has the numbers of that setup.

## Install

```sh
git clone https://github.com/05yuki/yuzuri.git
yuzuri/setup.sh --package <the server's venv>
```

What `setup.sh --package` does:

- It first imports the venv's SGLang and kt-kernel with Yuzuri's hooks on, without a GPU.
  Nothing is changed when that fails.

- It builds Yuzuri's KV library.

- It writes `yuzuri.pth` into the venv.

- Where the venv has `nvidia-cuda-nvcc`, it installs the `nvidia-cuda-tileiras` of the same version beside it.

What it leaves alone:

- SGLang is not modified.

- kt-kernel and FlashInfer are not modified either.
  - Six of their Python files need Yuzuri's edits (`patches/`).
  - A launch makes edited copies of the installed files under `~/.cache/yuzuri` (`YUZURI_CACHE`) and loads those.
  - An edit that does not fit the installed file stops `setup.sh`, and a launch, naming the line it looked for.

- kt-kernel's C++ extension is not rebuilt.
  - `setup.sh` reports what that build lacks and what Yuzuri turns off for it (`tools/check-kt-kernel.py`).
  - Those are NVFP4 one-byte scale codes for the hot lend's refill, and GGUF weight types.

To undo it:

- The `.pth` file does nothing unless the server is started through `tools/serve.py --package`.

- `tools/install-pth.py <venv> --remove` gives the server back.
  - `yuzuri.pth` is the only file of Yuzuri in the venv.
  - The `nvidia-cuda-tileiras` package stays; `pip uninstall nvidia-cuda-tileiras` removes it.
  - The edited copies are under `~/.cache/yuzuri`; delete that directory.
  - A venv whose Yuzuri checkout was moved or deleted starts as if Yuzuri were not there.

## Serve

```sh
yuzuri/tools/serve.py --package --venv <the server's venv> --config-dir <checkpoint dir> --tp 1
```

- `--tp 1` is one card and the CPU socket it hangs from.
  `--tp 2` is two cards, one rank a socket.
  Nothing else changes between them.

- The profile is built from the checkpoint.
  - The experts' format comes from the safetensors headers.
  - The model's layers and context come from its config.
  - The launch reads no list of models: an unknown checkpoint is sized the same way.
  - A GGUF pack's dense weights are the exception: their tensor names follow the converter, and two layouts are known (`kt_gguf_dense_*.py`).

- `--dry-run` prints the arithmetic and stops: weights, KV a token, threads, the resident hot experts.

- The server speaks SGLang's OpenAI-compatible API on `127.0.0.1:8080`, this machine only.
  - `--host 0.0.0.0` opens it to the network.
  - SGLang asks for no key unless one is set: `--server-arg="--api-key <key>"`.

- The first launch of an NVFP4 model compiles FlashInfer's cuTile kernels.

## Features

Every feature is on by default, and each can be turned off on its own.

```sh
yuzuri/tools/serve.py --list-features
yuzuri/tools/serve.py ... --without hot-lend,host-kv
yuzuri/tools/serve.py ... --features stream-prefill,prefill-lend
```

| feature | what it does | needs |
|---|---|---|
| `kv-vmm` | the KV pool on remappable memory, pages mapped as used | |
| `stream-prefill` | a prompt's experts streamed to the GPU from kt-kernel's arenas | |
| `prefill-lend` | the GPU-resident weights' VRAM lent to a streamed prefill | `stream-prefill` |
| `kv-lend-plan` | the KV pool's free pages lent to the prefill, each request's chunk planned | `kv-vmm`, `prefill-lend` |
| `swa-lend` | a hybrid sliding-window pool sized for the plan and lent outside prefill | `kv-lend-plan` |
| `hot-lend` | the pool's free pages hold more experts on the GPU, following the routing | `kv-vmm`, `prefill-lend`, `stream-prefill`, `kv-lend-plan` |
| `host-kv` | the pool at the model's context, the rows past the card in host memory | `kv-vmm`, `kv-lend-plan` |
| `cutile-nvfp4` | NVFP4 GPU experts on FlashInfer's cuTile MoE from the checkpoint's bytes | |
| `cpu-gpu-overlap` | the CPU hand-off packed by one kernel, the shared expert after the CPU request | |
| `host-allreduce` | all-reduce through host memory, for cards without peer access | |
| `live-topk` | the sparse-attention indexer's top-k selected while the scores are made | |
| `rope-tables` | the RoPE tables mapped as the context grows | `kv-vmm` |
| `embed-on-host` | the token embedding read from pinned host memory | |
| `mm-on-demand` | a multimodal model's encoder in host memory, on the card while it runs | |

- A feature turned on without what it needs stops the launch with what to add.

- The fixes (the thinking default, no shared-expert fusion under KT, the load timeout) have no switch.

## Expert formats

| format | CPU experts | GPU experts and streamed prefill |
|---|---|---|
| bf16 | kt-kernel BF16 | FlashInfer CUTLASS |
| ModelOpt or compressed-tensors NVFP4 | kt-kernel NVFP4 | FlashInfer cuTile, from the checkpoint's bytes |
| MXFP4 | kt-kernel MXFP4 | Marlin, W16 streamed groups |
| GGUF | kt-kernel LLAMAFILE | GGUF rows, MMQ streamed groups |

- Weights and scales stay in the checkpoint's form.

- An NVFP4 checkpoint runs at its own precision: W4A4 when it stores activation scales, W4A16 when it does not.

- Nothing is converted to a precision the checkpoint lacks.

- The GPU kernels named are the ones the measured SM120 cards ran.

## Results

Two RTX 5070 Ti 16 GB, two EPYC 7452, 512 GB DDR4, one request at a time.

TP=2 is both cards and both sockets.
TP=1 is one card and the socket next to it.

- **on** is `serve.py` with its defaults.
  Every pool is the model's context.

- **off** is `serve.py --features base`.
  It is the same SGLang, kt-kernel and checkpoint.
  Nothing is lent, streamed or moved to host memory; the kernels, the loader and the fixes stay.
  It has a 4,096-token prefill chunk.
  The memory settings are the ones a user of SGLang and kt-kernel sets by hand.
  `--mem-fraction-static` leaves 2.5 GiB of each card free for prefill (`serve.py --free-min-gb 2.5`).
  The hot experts are a fixed count a layer (`--hot`), the largest that kept the pool at the context and survived the check.
  Where the weights leave less than the context, the count is 1 and the pool is what fits.

- Hot slots: on, the experts on the cards, all layers and all cards together, lent from free pool pages, and the count a layer.
  It is the count at the first decode.
  Off lends none: it keeps its fixed count a layer.

- Decode: 1,024 tokens after a 384-token prompt, timed from the first token to the last.
  Five prompts of Japanese prose run (`tools/nll-text/`); the first is left out, and the number is the median of the other four.
  Sampling is temperature 1.0 and top-p 0.95, and the end-of-sequence token is ignored.

- 35K prefill: the second of two different 35,000-token prompts, in tokens a second.

- 131K prefill: a 131,072-token prompt, in tokens a second.

- Long decode: 1,024 greedy tokens after the 131K prompt, in one request, timed from the first token to the last.

- NLL: the mean over 24 public-domain pieces (`tools/nll.py`), in nats a token.
  On and off differ by the launch-to-launch spread of one configuration.
  Gemma answers in a chat template, so its number on bare text only compares on with off.

Each on row is one launch on 2026-11-17, `tools/check.py --prompt 35000 --long 131072`.
Each off row's decode and NLL are one launch on 2026-11-17.
Its pool, hot count, prefill and long decode are the same settings' launch on 2026-11-16.
The MiMo off row is the 2026-11-16 launch in every column: FP8 KV and `--swa-full-tokens-ratio 0.0657`, where on keeps the bf16 KV `--profile auto` reads.

TP=2:

| model | experts | | pool | hot slots | decode | 35K prefill | 131K prefill | long decode | NLL |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-35B-A3B | bf16 | on | 262,144 | 3,361 (84–85 a layer) | 112.0 tok/s | 9,023 tok/s | 6,556 tok/s | 74.1 tok/s | 1.7741 |
| | | off | 262,144 | 54 a layer | 55.0 tok/s | 440 tok/s | 435 tok/s | 49.6 tok/s | 1.7766 |
| Gemma 4 26B-A4B | bf16 | on | 262,144 | 1,484 (49–50 a layer) | 101.5 tok/s | 4,530 tok/s | 2,300 tok/s | 37.7 tok/s | 10.1885 |
| | | off | 88,859 | 1 a layer | 36.2 tok/s | 223 tok/s | does not fit the pool | | 10.1520 |
| Qwen3.8-Flash-Next | NVFP4 (ModelOpt, W4A4) | on | 262,144 | 4,435 (92–93 a layer) | 67.3 tok/s | 4,590 tok/s | 4,495 tok/s | 54.0 tok/s | 1.5918 |
| | | off | 262,144 | 27 a layer | 46.6 tok/s | 287 tok/s | 283 tok/s | 46.0 tok/s | 1.5882 |
| Qwen3.8-Flash-Next | GGUF IQ3_S | on | 262,144 | 4,290 (89–90 a layer) | 79.3 tok/s | 4,410 tok/s | 4,181 tok/s | 62.7 tok/s | 1.5966 |
| | | off | 262,144 | 28 a layer | 54.0 tok/s | 437 tok/s | 438 tok/s | 53.7 tok/s | 1.5973 |
| MiMo-V2.6-Flash | MXFP4 | on | 1,048,576 | 1,092 (23–24 a layer) | 49.7 tok/s | 1,930 tok/s | 1,158 tok/s | 37.8 tok/s | 1.2330 |
| | | off | 637,143 | 1 a layer | 27.3 tok/s | 95 tok/s | 94 tok/s | 22.0 tok/s | 1.2334 |
| GLM-5.3-Flash | NVFP4, FP8 dense | on | 1,048,576 | none | 24.8 tok/s | 1,750 tok/s | 2,173 tok/s | 24.9 tok/s | 1.0697 |
| | | off | 610,560 | none | 24.5 tok/s | 94 tok/s | 94 tok/s | 23.7 tok/s | 1.0752 |

TP=1:

| model | experts | | pool | hot slots | decode | 35K prefill | 131K prefill | long decode | NLL |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-35B-A3B | bf16 | on | 262,144 | 1,273 (31–32 a layer) | 57.7 tok/s | 5,596 tok/s | 4,044 tok/s | 39.7 tok/s | 1.7781 |
| | | off | 262,144 | 3 a layer | 36.2 tok/s | 227 tok/s | 223 tok/s | 32.2 tok/s | 1.7755 |
| Gemma 4 26B-A4B | bf16 | on | 262,144 | 577 (19–20 a layer) | 54.8 tok/s | 2,905 tok/s | 1,168 tok/s | 26.6 tok/s | 10.1858 |
| | | off | 31,762 | 1 a layer | 28.5 tok/s | 132 tok/s (30,762 tokens) | does not fit the pool | | 10.1983 |
| Qwen3.8-Flash-Next | NVFP4 (ModelOpt, W4A4) | on | 262,144 | 533 (11–12 a layer) | 37.4 tok/s | 3,010 tok/s | 2,059 tok/s | 33.2 tok/s | 1.5847 |
| | | off | 18,240 | 1 a layer | 33.2 tok/s | 166 tok/s (17,240 tokens) | does not fit the pool | | 1.5902 |
| Qwen3.8-Flash-Next | GGUF IQ3_S | on | 262,144 | 3,018 (62–63 a layer) | 52.9 tok/s | 3,036 tok/s | 2,663 tok/s | 39.5 tok/s | 1.5972 |
| | | off | 196,416 | 1 a layer | 39.5 tok/s | 267 tok/s | 269 tok/s | 39.2 tok/s | 1.5947 |

- The long decode is below the decode in every row, on and off.
  Attention reads 131K tokens of KV at each step.
  On falls further than off, by design: the hot slots sit in free KV pages, and a 131K prompt takes those pages back.
  The slots come back when the KV rows are freed.

- GLM runs without hot slots in both.
  Its config directory holds no expert tensor, and the hot slots are filled from the tensors the server's own loader reads.

- These are one host's numbers.
  On another server, compare the same prompts with and without Yuzuri:
  - `tools/check.py --json after.json` against the server
  - the same with `serve.py --features base`, for `before.json`
  - `tools/check.py --compare before.json after.json`

- The launches measured had an earlier launch's record of the prefill's memory cost (see Limits).

## Limits

- **Concurrent requests.**
  One runs at a time by default.
  `serve.py --max-running-requests 4` runs up to four.
  - At the default, a request that arrives while another runs waits in SGLang's queue.
  - With more, the streams decode together.
  - The CPU reads every expert any of them routes to, so streams of the same kind of text gain the most.
  - A long prompt holds the other streams while it prefills: the prefill lends the GPU's weights and free KV
    pages to itself.
    They resume when it returns.
  - The hot slots use the pool's free pages, so they shrink as running requests fill the pool.
  - A model with linear-attention state gets that many requests' state slots, and the plan counts their memory.

- **Four requests, measured.**
  Qwen3.5-35B-A3B bf16, four 512-token Japanese prose requests, tokens over wall time.
  - TP=2: 93.9 tok/s one after another, 198.2 tok/s together.
  - TP=1: 53.1 tok/s one after another, 85.2 tok/s together.
  - A 52,000-token prompt arriving while three streams decode: all four finish.
    No KV row is left in host memory during that prefill.
  - The numbers in Results are one request at a time.

- **One host.**
  Tested at TP=2 and TP=1.
  Other card counts, card sizes and socket counts are read at launch the same way and are not measured.

- **Not tested:** EP, DP attention, PP.

- **NVIDIA only.**
  AMD and Intel GPUs are not supported.
  - The KV pool, the lends and the hot slots are built on CUDA's virtual memory calls
    (`cuMemAddressReserve`, `cuMemMap`), which ROCm does not have.
  - The GPU experts run on FlashInfer's CUDA kernels.

- **GPU generation.**
  Among NVIDIA cards, Yuzuri asks for none.
  - A card the base server runs on is a card Yuzuri runs on.
  - Its lends, plan and host KV use CUDA memory calls, not a generation's kernels.
  - It also widens the base: the DSV4-family sparse attention gets paths for SM89 and for cards before SM80.
  - Measured on SM120 only.
    The other generations are supported and not yet launched with the package.

- **Launch time.**
  A launch takes minutes: kt-kernel pins every expert in host memory first.

- **1M context.**
  Host KV at 1M context needs host memory for the whole pool.

- **The first launch on new cards.**
  It measures the prefill's memory cost before it reports ready.
  It sizes SGLang's buffers for a 36,864-row chunk.
  Later launches read what it measured from the cache directory.

- **GGUF experts.**
  They need a kt-kernel build with their weight types' CPU kernels.
  `setup.sh --package` says when it lacks them.

## Checked against

Yuzuri pins no SGLang version.

These are the SGLang commits it has been launched on, on two RTX 5070 Ti (SM120):

| SGLang commit | Tree | Date |
|---|---|---|
| `58484246d2` | the DeepSeek-V4.1 recipe's patched tree | 2026-11-04 |
| `d193c34ae4` | upstream main, no patches | 2026-11-13 |
| `8c7ec05287` | upstream main with the recipe's tier 0 patches | 2026-11-15 |
| `8c7ec05287` | upstream main with the [demo recipe](#demo-recipe)'s five model patches | 2026-11-18 |

On another commit, the startup report says which hooks went in.

`[yuzuri-features]` lines name every hook as on, the tree's own, off, gone or idle.

The same table is written to `hooks-<pid>.json` beside `manifest.json` in the run directory.

A hook that is idle attaches to an SGLang module this model never loaded: it does nothing in that launch.

## Environment variables

- A first run sets none: the launcher sets what a run needs.

- [docs/env.md](docs/env.md) lists every variable the package reads, with its default and the file that reads it.
  It is generated from the code by `python3 tools/list-env.py`.
  The tests fail when the list and the code differ.

## Tests

```sh
python -m unittest discover -s tests
```

- These run without SGLang and without a GPU.
  - The hook helper is run against a stand-in module: a matching function, one that gained a parameter, one that
    lost one, a tree that already carries the change, a line a rewrite looks for that is gone.
  - The tables are checked against the tree: hooks, features, manifest, licenses, file names in the text.

- `tests/gpu/test_kv_vmm.py` and `tools/test_gpu_parts.py` need a CUDA card.
  They check the KV library and Yuzuri's own GPU parts against plain torch.

- `setup.sh` imports the venv's SGLang with every hook on before it changes anything.
  That is the check against a new SGLang commit.

## Related work

[Strata](https://github.com/Niko1221/Strata) lends slots of its GPU expert cache to the prompt path, inside its own engine.

Yuzuri lends the dense weights and the KV pool's pages as well, inside SGLang.

## License

Apache-2.0, like KTransformers and SGLang.

No SGLang file is carried: `python/yuzuri/tree/` holds Yuzuri's own modules (see `python/yuzuri/NOTICE`).

The GGUF matmul kernels under `kt_gguf_moe_mmq/llama_mmq/` are derived from llama.cpp, MIT.
Its license text is `LICENSES/MIT-llama.cpp.txt`.

