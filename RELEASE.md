# Paddock 0.1.5

A feature release. Windows x64 and Linux x64, NVIDIA GPUs, driver 580 or newer.

The theme is small cards: 1-, 2- and 3-bit files now serve, and the largest
mixture-of-experts models reach a 16 GB card through expert offload.

## New

- **1-, 2- and 3-bit GGUF files serve.** The ggml i-quant family (IQ1_S,
  IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL) and the Q2_K and
  Q3_K formats load and serve end to end, for dense weights and for MoE
  experts alike, with decoders written here from the format specification.
  Prompt processing on these files runs on the same tiled GEMM as the 4-bit
  files, so a long prompt costs the same as it does at 4 bits. A file is
  named by its real bit width everywhere it appears. Contributed by
  NodeNestor (#17, #21).

- **Qwen 3.8 Flash-Next serves from GGUF on RTX 50-series cards.** The
  mixture-of-experts lane loads unsloth's UD files, and with expert offload
  the UD-IQ1_S file serves on a 16 GB card at llama.cpp parity. Prompt
  processing streams the offloaded experts through the cache expert by
  expert: a 427-token prompt went from 308 s to 37 s, and the first token of
  a 1,500-word prompt from 31 s to 15 s, on an RTX 5060 Ti. Blackwell only in
  this release; the family's kernels are built for sm_100 and sm_120.
  Contributed by NodeNestor (#18).

- **Qwen 3.8 27B has a Compact lane** in the catalog: the UD-Q3_K_XL file,
  for cards where the standard file does not fit.

## Improved

- **Shared-prefix serving under load.** Requests that arrive with the same
  prefix, the shape of an agent fleet, are now recognised at admission and
  warmed in one batched wave instead of each recomputing the prefix. With
  eight concurrent sessions on a shared prompt the time to first token fell
  from **1184 ms to 248 ms** and the p99 gap between tokens from 88 ms to
  31 ms.

- **Small cards decode faster with many streams.** The k-quant K-split path
  now takes batched decode, so on an RTX 5060 Ti eight concurrent streams of
  a Qwen 3.5/3.6 k-quant file step in **17.8 ms instead of 30.0 ms**, twice
  llama.cpp's aggregate on the same card.

- **Memory planned by demand.** Prompt-processing scratch is profiled at
  load, and state checkpoints and cache retention are sized by what the
  configuration actually needs, across Qwen 3.5/3.6, Gemma 4, Granite,
  Laguna and Nemotron. Laguna serves batched. A `graph_scratch_mib` setting
  in the Advanced tab reserves extra scratch by hand (ErikBPF, #14).

- **A GPU the engine has not validated serves under a startup warning**
  instead of a refusal.

## Fixed

- Blackwell kernels on Windows could fail: the Windows build laid out
  tensor-map kernel parameters at 8-byte alignment where the hardware wants
  128 (#6).
- The multi-column weight kernel faulted on narrow planes at two concurrent
  streams on sm_120 (#19, reported by L4GN).
- A dense i-quant kernel's launch grid overflowed past 134 million outputs.
- Qwen 3.5/3.6: a failed attempt to enable batching is released before the
  width ladder retries (ErikBPF, #15).

## Known

- The fp8 KV cache's paged attention on RTX 50-series cards shows a small
  numeric deviation in one split configuration (#5). Being fixed on an fp8
  card.
