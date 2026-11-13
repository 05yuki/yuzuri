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
  During a long prompt the GPU-resident weights give their VRAM to the prefill.
  They come back layer by layer, at the same addresses.

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

- SGLang as upstream ships it, or with your own patches.
  - Yuzuri replaces no SGLang file. It attaches to the functions its features live in when the server starts.
  - A function that has moved or changed shape stops the launch and names it.
  - So does an SGLang that already carries one of Yuzuri's mechanisms.
  - Checked on commits `58484246d26523eaf1a7f16ab524deee288d2120` and
    `d193c34ae4b138aae4b8c0479ff395b0138a0156` (702 commits later).

- Linux and NVIDIA GPUs (CUDA), one or more, of any memory size. AMD GPUs are not supported.
  - What a card cannot hold stays in host memory: experts, and the KV rows past the card.
  - Measured on SM120 cards of 16 GB.

- Host memory for the experts and for the KV past the cards.

[deepseek-v4.1-flash-1017-ktransformers](https://github.com/05yuki/deepseek-v4.1-flash-1017-ktransformers)
(tag `2026-11-05-0936`) builds a venv with kt-kernel and the toolkit.

Its SGLang is patched with mechanisms Yuzuri brings itself, so Yuzuri stops on it.
Put an unpatched SGLang checkout ahead of it: `PYTHONPATH=<sglang checkout>/python`, for `setup.sh` and for the launch.

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
  Nothing else of Yuzuri is in the venv.

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
  - No model is listed anywhere.

- `--dry-run` prints the arithmetic and stops: weights, KV a token, threads, the resident hot experts.

- `--profile <name>` takes a measured command line from `profiles/` instead.

- The server speaks SGLang's OpenAI-compatible API on `127.0.0.1:8080`.

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

- **off** is `serve.py --features none`.
  It is the same SGLang, kt-kernel and checkpoint.
  It has a 4,096-token prefill chunk and SGLang's own share of the card for the pool.

- Decode: Japanese prose, 1,024 tokens, the median of four prompts.

- 35K prefill: the first 35,000-token prompt after the launch, in tokens a second.

- 131K prefill: a 131,072-token prompt, in tokens a second.

- Long decode: 256 tokens after the 131K prompt.

TP=2:

| model | experts | | pool | decode | 35K prefill | 131K prefill | long decode |
|---|---|---|---|---|---|---|---|
| Qwen3.5-35B-A3B | bf16 | on | 262,144 | 131.7 tok/s | 7,256 tok/s | 6,633 tok/s | 50.4 tok/s |
| | | off | 262,144 | 46.6 tok/s | 343 tok/s | 353 tok/s | 42.5 tok/s |
| Gemma 4 26B-A4B | bf16 | on | 262,144 | 104.2 tok/s | 4,113 tok/s | 2,387 tok/s | 33.6 tok/s |
| | | off | 37,474 | 36.4 tok/s | 220 tok/s | does not fit the pool | |
| Qwen3.8-Flash-Next | NVFP4 (ModelOpt, W4A4) | on | 262,144 | 67.0 tok/s | 4,149 tok/s | 4,201 tok/s (150K) | 47.6 tok/s |
| | | off | 82,368 | 45.6 tok/s | 270 tok/s | does not fit the pool | |
| Qwen3.8-Flash-Next | GGUF IQ3_S | on | 262,144 | 67.9 tok/s | 4,024 tok/s | 4,659 tok/s | 61.5 tok/s |
| | | off | 262,144 | 53.9 tok/s | 419 tok/s | 422 tok/s | 53.2 tok/s |
| MiMo-V2.6-Flash | MXFP4 | on | 1,048,576 | 44.2 tok/s | 1,675 tok/s | 1,463 tok/s | 33.4 tok/s |
| | | off | 115,480 | 25.7 tok/s | 94 tok/s | does not fit the pool | |
| GLM-5.3-Flash | NVFP4, FP8 dense | on | 1,048,576 | 24.6 tok/s | 1,871 tok/s | 2,178 tok/s | 24.7 tok/s |
| | | off | does not start | | | | |

TP=1:

| model | experts | | pool | decode | 35K prefill | 131K prefill | long decode |
|---|---|---|---|---|---|---|---|
| Qwen3.5-35B-A3B | bf16 | on | 262,144 | 77.8 tok/s | 4,961 tok/s | 3,891 tok/s | 35.5 tok/s |
| | | off | 87,048 | 36.2 tok/s | 218 tok/s | does not fit the pool | |
| Gemma 4 26B-A4B | bf16 | on | 262,144 | 69.0 tok/s | 3,009 tok/s | 1,343 tok/s | 25.9 tok/s |
| | | off | 6,366 | 28.5 tok/s | does not fit the pool | does not fit the pool | |
| Qwen3.8-Flash-Next | NVFP4 (ModelOpt, W4A4) | on | 262,144 | 43.2 tok/s | 2,741 tok/s | 1,879 tok/s | 31.5 tok/s |
| | | off | does not start | | | | |

- "Does not start": the weights leave no room for a pool under SGLang's default share of the card, and the
  launch stops asking for a larger `--mem-fraction-static`.

- These are one host's numbers.
  On another server, compare the same prompts with and without Yuzuri:
  - `tools/check.py --json after.json` against the server
  - the same with `serve.py --features none`, for `before.json`
  - `tools/check.py --compare before.json after.json`

- The launches measured had an earlier launch's record of the prefill's memory cost (see Limits).

- The Flash-Next TP=1 row is the mean of two launches.
  Its 131K prefill was 2,054 and 1,703 tok/s in the two.

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

## Related work

[Strata](https://github.com/Niko1221/Strata) lends slots of its GPU expert cache to the prompt path, inside its own engine.

Yuzuri lends the dense weights and the KV pool's pages as well, inside SGLang.

## License

Apache-2.0, like KTransformers and SGLang.

`python/yuzuri/tree/` holds Yuzuri's versions of SGLang modules (see `python/yuzuri/NOTICE`).

