# Paddock

[![Discord](https://img.shields.io/badge/Discord-join%20the%20server-5865F2?logo=discord&logoColor=white)](https://discord.gg/fMeKkPdrsW)

![Paddock Studio](assets/studio-graph.gif)

## Community

Questions, setups that do not work, benchmark results and ideas: the Paddock
Discord, <https://discord.gg/fMeKkPdrsW>. Bugs and model requests are best as GitHub
issues, where the templates ask for what makes a report actionable.

## What it is

AI inference server written in Rust, currently for NVIDIA GPUs, with an OpenAI and Anthropic compatible API
and a Studio for exploring and comparing models, with artifact support.

It is a native engine and NOT A WRAPPER - the scheduler, the paged KV cache, the memory
management and the CUDA kernels are all part of this repo.

Two binaries, paddock-runner is what runs models and is really the only thing needed but we made
paddock (manager) as a simpler way to get started running models quickly and we added the built-in Studio
that allows users to quickly work with models and test them against any cloud model through OpenRouter
or other endpoints. Paddock (manager) will bundle the Studio to just make it so simple for users
to download, just two binaries and that's all.

User only needs the NVIDIA driver.

## Our aim is being the fastest most modern inference platform

Paddock aims at being the fastest inference engine for models that fit on a single
GPU, until we support tensor parallelism, meaning splitting model layers across GPUs.

We currently beat llama.cpp, vLLM, SGLang on tested RTX PRO 6000, B200 with models such as
Qwen 3.8 27B, Qwen 3.6 27B, Gemma 4 31B on FP8/Q8_0 and Q4_K_XL and in most cases on NVFP4.
We recently added Qwen 3.8 Flash Next and results are promising for lower concurrency but still
have a bit of work in the higher concurrency tests.

Paddock is aimed at making it easier for organizations and companies to run open models
in production, mainly on Blackwell GPUs such as the RTX 5090, RTX PRO 6000 and B200. Ampere
is supported and tested as well: the RTX A6000 was the original bring-up card and still runs
the heavy parity suites. Ada Lovelace kernels ship in the pack, but the architecture has no
measured board yet, so the engine serves it with an `UNVALIDATED` warning in the log rather
than a supported claim - the same goes for any other card we have not measured.
Hopper and the A100 have kernels in the source tree but no board and no place in the release
pack. We hope contributors will help close those gaps.

## Supported quants

So far we support native FP8 and NVFP4, MXFP4, plus the Q8_0 and
Q4_K_XL/Q4_K_L/Q4_K_M quantizations. Weights load from both GGUF and safetensors.

## Available GPUs for contributors - Free

Contributors can request GPUs from the repo managers and are entitled to between 24 hours and 31 days
of free private usage. Each GPU is private so you are not sharing it with others. You have full access
to the GPU with edit clocks and use NVIDIA Nsight.

We offer freely today:
- NVIDIA RTX 5090
- NVIDIA RTX PRO 6000
- NVIDIA B200

We don't have unlimited capacity, so it is first come, first served, and we encourage you to
use the time rather than idle the machine. We start at 24 hours and gradually extend regular
contributors up to 31 days of "their own GPU".

## Models tested

Every model below runs on the engine today. Sizes are the checkpoint's own, and
`A3B`-style suffixes mean a mixture-of-experts model with that many active parameters.

**Text, vision and tool use**

- **Qwen** - 3.5 9B, 3.6 27B, 3.6 35B-A3B, 3.8 27B, and 3.8 Flash Next
- **Gemma 4** - 31B and 26B-A4B
- **GPT-OSS** - 20B and 120B
- **Granite** - 4.1 8B/30B, 4.2 8B/30B, and 4.1 Vision 4B
- **Laguna 2.1** - XS (33B-A3B) and S (118B-A8B)
- **Muse Glimmer** - 30B
- **Nemotron 3.5 Lightning** - 30B-A3B

**Documents and OCR**

- **PaddleOCR-VL** 1.6
- **Unlimited-OCR** 3B

**Speech**

- **Qwen3-ASR** 1.7B, and **Qwen3 Forced Aligner** 0.6B
- **Granite Speech 4.1** - 2B and 2B Plus
- **Whisper Large** fine-tunes - KB-Whisper (Swedish), NB-Whisper (Norwegian),
  Røst v3 (Danish)

**Embeddings and reranking**

- **Qwen 3 Embedding** - 0.6B, 4B, 8B
- **Qwen 3 Reranker** - 0.6B, 4B, 8B

## What it contains

**Serving.** Paged KV cache, continuous batching with chunked prefill, radix
prefix caching, and fair scheduling across concurrent sessions. The design
target is the agentic workload: several coding-agent sessions against one box,
long contexts, heavily shared prefixes.

**Model formats.** GGUF and safetensors. Quantization covers the k-quant family,
Q8_0, MXFP4, native FP8 (E4M3) and block-scale FP4/NVFP4, dispatched per tensor
rather than per model, because real checkpoints mix types.

**Memory.** Paged KV with per-layer cache kinds, FP8 KV storage, KV offload to
host RAM and optionally to disk, MoE expert streaming from RAM for models larger
than VRAM, and will-it-fit estimation (please note this is not accurate and we plan to fix this)
that reports honestly instead of failing at load.

**APIs.** OpenAI chat completions, completions, the Responses API, embeddings
and audio transcriptions; Anthropic messages and token counting; plus a
Model Context Protocol surface. The runner publishes its own `/openapi.json`.

**Beyond text.** Vision models, speech transcription, reranking, and document
extraction from PDF and Office files.

**Studio.** A built-in web UI for downloading and managing models, running them,
comparing them side by side, and serving them.

Side by side means across the boundary, not just within it: a model running on
your own box against one behind OpenRouter or any other OpenAI-compatible
endpoint, same prompt, same view.

![Comparing a local model with a cloud one](assets/studio-compare.gif)

## Status

Paddock is young and under active development.

- **Backend:** CUDA only. The release kernel pack carries SASS for sm_86, sm_89,
  sm_100 and sm_120; the build script can target more. There is no Vulkan,
  Metal or ROCm backend.
- **Platforms:** Windows and Linux, x64. macOS is out of scope for now.
- **Models:** most modern families are supported, feel free to add support.

## What is pdfium doing in the project ?

To make it easier to work with PDF files in the Studio, and to let vision models receive
rendered pages automatically, we include pdfium to render PDFs to images.
This is an optional piece, added mostly as a convenience.

## What is siftx doing in the project ?

This is another repo of ours, [truespar/siftx](https://github.com/truespar/siftx), a metadata
and PDF extraction library in Rust, MIT/Apache 2.0. It lets us inject data a model cannot
otherwise see or process, in-process, rather than shelling out to a sub-process.

## What is Traverse doing in the project ?

Traverse is a graph database, written in Rust, similar to Neo4j, supporting openCypher and GQL. It's going
open source under MIT/Apache 2.0 during Q4 2026 as part of our initiative that all our code will be open-source.
Traverse is compiled to a WASM version that is embedded and can run in the Studio to create graphs or consume graphs.

## What is Scriptor doing in the project ?

Scriptor is another repo of ours, [truespar/scriptor](https://github.com/truespar/scriptor),
open source under MIT/Apache 2.0. It is an OOXML editor and renderer in WASM. It lets LLMs
work with Word documents in the browser, viewing, understanding and editing them, with no
external shells. The Studio carries its browser packages (`core`, `vue` and the WebAssembly
build of the engine) under `studio/vendor/scriptor`, taken from that repository at the same
revision the Rust side pins; the `VERSION` file there records it.

## What is Lector doing in the project ?

Lector is another repo of ours, a WASM-based PDF viewer for the browser. It's going open source under
MIT/Apache 2.0 during September 2026 as part of our initiative that all our code will be open-source.
It's used in the Studio to offer better in-app viewing of PDFs. It also offers decryption, encryption,
signing, redaction, annotation and many other enterprise features common in advanced PDF viewers.
The Studio carries its `core`, `vue`, `utils` and `pdfium-wasm` packages under `studio/vendor/lector`,
already under the same licence.

## Building

The remaining prerequisites:

| | why |
|---|---|
| Rust, current stable | the toolchain file pins the channel, not a version |
| Node and npm | builds the Studio bundle embedded into `paddock` |
| cmake and a C/C++ compiler | the Opus codec is compiled from source at build time, by cmake; on Windows VS2022 covers it |
| CUDA toolkit 13.x | only for the kernel pack, not for `cargo build` |
| Windows: VS2022 | build from a `vcvars64` environment; `rc.exe` is also required |

Neither `siftx` nor `scriptor` is on crates.io, so cargo fetches them from GitHub at a
pinned revision. The first build needs network for that; nothing else does.

```sh
# 1. pdfium, a build input linked into paddock-runner: download our prebuilt
powershell -File packs/pdfium/fetch.ps1                 # Windows
bash packs/pdfium/fetch.sh                              # Linux

# 2. the Studio bundle, embedded by rust-embed
cd studio && npm ci && npm run build && cd ..

# 3. the binaries
cargo build --release -p paddock-manager -p paddock-runner
```

The fetch scripts read `packs/pdfium/prebuilt.json`, download our own build of
pdfium at the pin in `packs/pdfium/VERSION`, check the SHA-256 and stage the
library at `packs/pdfium/<platform>/`, where the build script looks for it.
They are idempotent, so they are safe to run on every build.

To build pdfium from source instead, `packs/pdfium/build/build-windows.ps1`
(native, needs depot_tools) and `packs/pdfium/build/build-linux.sh` (via
Docker) stage the same file; both take around 15 minutes. The Windows script
syncs the Chromium tree into a `pdfium-build` directory beside the checkout
(`-Root` puts it elsewhere).

The CUDA kernel pack is built separately, on purpose: the Rust build needs
neither a CUDA toolkit nor a GPU, and the pack is loaded over a stable C ABI at
runtime.

```sh
powershell -File packs/cuda/build.ps1              # every arch
powershell -File packs/cuda/build.ps1 -Arches 86   # just yours, much faster
bash packs/cuda/build.sh                           # Linux
```

Kernels are hand-written CUDA, with no Triton, no DSL and no Python anywhere in
the build. The sources are organised by domain under `packs/cuda/src/`, and
`pack.cu`'s include list is the one true order. The single exception is
`gemm/cutgemm.cu`, a CUTLASS-based fp8 GEMM for sm_100 kept in its own
translation unit so those headers never reach the main compile. It is built
only when you point `PD_CUTLASS_INC` at a CUTLASS checkout, and compiles to an
unsupported stub otherwise.

## Benchmarking

Use [aiperf](https://github.com/ai-dynamo/aiperf), NVIDIA's load generator, for
any number you intend to publish or to compare against another engine. It is
what users of every engine run, it treats every server as the same black box,
and it counts tokens on the client by re-tokenizing the visible stream, so a
result does not depend on what a server chooses to report. Our own comparison
boards are aiperf columns, one engine resident on the GPU at a time.

The repo also carries `paddock-bench` (`crates/paddock-bench`), an in-process
and HTTP harness. It is for engine work: timing a kernel change in isolation,
or a quick probe of a running server. Its numbers are not comparable to
anything outside this repo and we do not publish them.

**The scenario we run.** One YAML per cell. Engine, model, URL and tokenizer
enter through the environment, so the same file drives every engine. This is
`syn_128x128_c32`; the other cells change only the numbers under `prompts`,
`warmup` and `profiling`.

```yaml
schemaVersion: "2.0"
# prompt-generation seed: 7 plus the rep number, the same on every engine
randomSeed: ${PDK_PROMPT_SEED:7}

benchmark:
  model: ${PDK_MODEL:gemma-4-31B-it-Q8_0}
  endpoint:
    url: ${PDK_URL:http://localhost:11660}
    type: chat
    streaming: true
    timeout: 600.0
    extra:
      ignore_eos: true
      temperature: 0.7
      seed: 7                 # sampling seed, pinned on every engine
  tokenizer:
    name: ${PDK_TOKENIZER:google/gemma-4-31B-it}
  dataset:
    type: synthetic
    entries: 256              # unique prompt pool, larger than warmup + profiling
    prompts:
      isl: {mean: 128, stddev: 0}
      osl: {mean: 128, stddev: 0}
      corpus: coding
  warmup:
    type: concurrency
    requests: 32              # 1x concurrency
    concurrency: 32
  profiling:
    type: concurrency
    requests: 192             # 6x concurrency
    concurrency: 32
  artifacts:
    dir: ./artifacts/syn_128x128_c32
    summary: [json]
    records: [jsonl]
```

| cell | prompt / output tokens | concurrency | what it weighs |
|---|---|---|---|
| `syn_128x128_c{1,8,16,32,64,96,128}` | 128 / 128 | 1 to 128 | decode, batching, the scheduler |
| `syn_2048x128_c32` | 2048 / 128 | 32 | prefill |
| `syn_128x2048_c32` | 128 / 2048 | 32 | long decode |
| `syn_1024x1024_c{1,8}` | 1024 / 1024 | 1, 8 | balanced |
| `syn_8192x1024_c8` | 8192 / 1024 | 8 | long prompts |
| `imax_chat_1k1k_c32` | 1024 / 1024 | 32 | InferenceMAX chat |
| `imax_reason_1k8k_c8` | 1024 / 8192 | 8 | InferenceMAX reasoning |
| `imax_sum_8k1k_c8` | 8192 / 1024 | 8 | InferenceMAX summarization |
| `smoke_128x128_c4` | 128 / 128 | 4 | a regression smoke test, not a board cell |

**The commands.** Serve, then run the cell against the server. `max_batch` must
be at least the cell's concurrency and `max_ctx` must hold its prompt plus its
output.

```sh
# the server under test
paddock-runner --model <model>.gguf --kernel-pack packs/cuda/build/pd-cuda-sm86.dll \
  --port 11660 --max-batch 32 --max-ctx 8192

# the client, once per rep with its own prompt seed. The tokenizer must be the
# served model's own; aiperf takes a Hugging Face name or a local directory.
PDK_MODEL=<served id> PDK_URL=http://localhost:11660 PDK_TOKENIZER=<tokenizer> \
PDK_PROMPT_SEED=8 \
  aiperf profile --config syn_128x128_c32.yaml --artifact-dir out/syn_128x128_c32 --ui simple
```

We read `output_token_throughput.avg`, `time_to_first_token.p50` and
`inter_token_latency.p50` from `profile_export_aiperf.json` in the artifact
directory, and quote the median across reps.

## Licence

Paddock is dual-licensed under the [MIT licence](LICENSE-MIT) and the
[Apache Licence 2.0](LICENSE-APACHE), at your option. See [LICENSE](LICENSE).

Third-party components remain under their own licences, collected in
[THIRD-PARTY-NOTICES](THIRD-PARTY-NOTICES).

Where code is inspired by someone else's work, they are credited in the comments that apply.

## Principles

1. **No silent failures.** Context truncation is an error, not a trim. You can
   always see what is on the GPU, which quantization, and which context length.
2. **Your models, plainly stored.** Standard GGUF in visible paths. Import in
   place, export freely, no lock-in.
3. **Honest estimation.** Will-it-fit math across VRAM, RAM and MoE splits that
   tells the truth, including when the truth is no.
4. **Honest naming.** Models and quantizations are labelled for what they are.
5. **Local means local.** No account for the local path, ever.
6. **Secure by default.** Authenticated network API, signed updates,
   memory-safe parsers.
7. **Upstream citizenship.** Loud attribution, licence notices shipped with
   every artifact, fixes contributed back.
