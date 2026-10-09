# The Metal kernels

Hand-written Metal Shading Language for Apple Silicon, used by
`crates/paddock-metal`. Same rule as the CUDA pack next door: the kernels are
written here. Other projects are read for their algorithms, and the file that
learned from one says so in its opening comment.

Two files carry something that is not ours, and both say so:

- `iquant_tables.metal` holds ggml's IQ2_S / IQ3_S codebooks. They are the
  format, not a kernel, and must stay byte-identical (a test checks it).
- `splash.metal` has two traversal helpers adapted from Inco AI's Splash
  (Apache-2.0). `splash.NOTICE.md` has the revision and what was changed.

## No build step

There is nothing to compile ahead of time. `crates/paddock-metal/src/device.rs`
embeds every file with `include_str!`, joins them into one source string and
hands it to Metal when the runner starts.

That list in `device.rs` is the one true order, the way `pack.cu`'s include
list is for CUDA. It is one translation unit, so a file can use what a file
above it defined and nothing below it. Two things follow:

- a new `.metal` file that is not added to the list is never compiled;
- add it after the files it builds on, not alphabetically.

## What is where

The directory is deliberately flat: the file name is the grouping, and each
file's opening comment says what it holds and which graph or paper it follows.
The right-hand column is the folder under `crates/paddock-metal/src/` that
drives those kernels.

| files | Rust side |
|---|---|
| `linear` (Q8_0), `kquant`, `iquant` + `iquant_tables`, `mlx_affine` | shared: `weights`, `projection`, `iquant`, `affine` |
| `mlx_affine_few` | stable small-row packed loading; vector/MMA/occupancy experiments are test-only |
| `attention`, `moe` (MXFP4), `deltanet`, `spec`, `dflash` | shared across families |
| `granite`, `granite_attention64`, `granite_prefill`, `granite_speech`, `granite_vision` | `granite/` |
| `gemma4`, `gemma4_vision`, `gemma_moe`, `gemma_mlx`, `gemma_mlx_vision`, `muse`, `muse_vision` | `gemma4/` |
| `qwen_attention`, `qwen_projection`, `qwen_moe`, `mlx_qwen`, `vision` | `qwen35/` |
| `qwen4exp`, `qwen4exp_affine`, `qwen4exp_mlx`, `qwen4exp_moe`, `qwen4exp_qsa` | `qwen4exp/` |
| `qwen3_encoder`, `qwen3_asr`, `qwen3_aligner` | `qwen3/`, `qwen3_asr/` |
| `embedding_gemma2`, `embedding_gemma2_attention`, `embedding_gemma2_vision`, `embedding_gemma2_audio` | `embedding_gemma2/`: bidirectional encoder, request-local affine8 projection reductions, bounded-slab BF16 attention/softmax (GGUF F16 QKV/F32 softmax and accumulation, KT32), GPU mean pooling/MRL; GGUF and MLX image/audio towers, sampled video, grouped image projections, packed causal-isolated audio, GPU resize and compensated FFT; 128-row media-projection candidates are diagnostic-only |
| `splash`, `splash_attention`, `splash_draft`, `splash_draft_attention` | `splash` (the packed-Q4 package format) |
| `gpt_oss`, `laguna`, `nemotron`, `paddleocr`, `whisper` | the folder of the same name |
| `kolibri` | `kolibri/`: mixed affine4/8, raw-logit top-6, sandwich norms and BF16 full/sliding attention |
| `unlimited_ocr`, `unlimited_vision` | `unlimited_ocr/` |
| `clef`, `clef_quant`, `clef_attention`, `clef_head`, `clef_vision` | `clef/` |
| `sam3`, `sam3_image` | `sam3/`: image-encoder component bring-up (not serving support); window/global attention, detector/tracker necks, distinct picture/video input arithmetic; full-model reference gates still pending |

`granite.metal` comes first in the list and also defines the weight-decoding
helpers most later files call, which is why it is not only about Granite.

The Rust side mirrors this: one folder per family under
`crates/paddock-metal/src/`, shaped like the CUDA engine's `gpu_model/`
folders, with the shared pieces (`device`, `weights`, `projection`, `affine`,
`iquant`, `offload`, `schedule`) beside them.

## Checking a change

Nothing here can be built or run off macOS. On a Mac, `cargo test -p
paddock-metal` runs the GPU tests against the embedded source. The tests that
need a model file are `#[ignore]`d, and the reason string names the environment
variable each one wants.
