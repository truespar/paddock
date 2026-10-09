# Paddock 0.1.15

A fixes and polish release: the RAM context cache no longer switches itself
off when it fills up, Claude Code sessions on Kolibri 1 run longer and
faster, tool definitions reach every model exactly as its chat template
expects, and on the Mac EmbeddingGemma 2 now takes pictures and audio.
Windows x64, Linux x64 and the NVIDIA DGX Spark. NVIDIA GPUs, driver 580 or
newer. The macOS pre-release for Apple Silicon is built from the same commit.

## New

- **TIC Forestry v1** (The Intelligence Company). Land cover and canopy
  height at 1 m from Swedish aerial imagery, near-infrared included, in one
  pass. Over the API, `/v1/segmentations`. Like SAM 3 it is a gated
  download: accept the DINOv3 licence on its Hugging Face page and add a
  token in Manager > Settings. On NVIDIA GPUs.

- **Gated downloads are marked in the model picker** with a lock and
  "licence on Hugging Face", so you know before you start that the download
  needs your own token.

## Improved

- **Kolibri 1 in long agent sessions.** Reading long prompts and decoding
  deep into the context are faster with the 8-bit context cache, the
  default (Nemotron's long-context decode gains too), and so is the NVFP4
  version's prompt reading on Blackwell. After a reply that calls a tool,
  the next request resumes from the cache instead of reading the reply's
  reasoning again.

- **Tool definitions reach the model as its chat template writes them.**
  Tool schemas and earlier tool calls kept their keys in alphabetical order
  and had `<`, `>`, `&` and apostrophes escaped; they now arrive exactly as
  the reference renderer writes them, on every model family.

- **System messages in the middle of a conversation stay where they are**
  on models whose templates allow it - Kolibri 1, Laguna, Gemma 4, Granite
  and Muse Glimmer - instead of being folded into the next user turn.
  Claude Code ends every request with one.

- **Faster pictures on Gemma 4 and EmbeddingGemma 2**, which share a picture
  tower, and faster EmbeddingGemma 2 text embeddings on NVIDIA GPUs. The
  results are the same to the bit.

- **Kumo Tabular asks Size and Task** on the start form instead of offering
  six quality cards, and the Tables page offers to start the other task -
  predicting numbers instead of classes, or the reverse.

## Fixed

- **The RAM context cache no longer goes offline when it fills up.** A full
  cache could have free space and still no gap big enough for the next
  block; those stores counted as failures, and enough of them switched the
  cache off until the model restarted ("Cache offline" in the Studio). A
  full cache now makes room before it stores, and a cache that is simply
  full is never switched off.

- **Claude Code's thinking display setting** is accepted by the Messages
  API instead of refusing the request.

## macOS (pre-release)

- EmbeddingGemma 2 takes pictures and audio on Metal, from the GGUF with its
  picture and audio downloads or from an MLX 8-bit package that holds both;
  video comes in as sampled frames, without the soundtrack.
- The native app has an Embeddings workspace.
- MLX text embeddings no longer depend on what else is in the batch.

## Known

- **SAM 3 and TIC Forestry run on NVIDIA GPUs only**, and their weights
  download only with a Hugging Face account that has accepted the model's
  licence. TIC Forestry is built for Swedish imagery at 0.5 m; it is not a
  general-purpose aerial model.

- **Kolibri 1 has no same-weights llama.cpp reference yet:** no llama.cpp
  release reads it, so its correctness is checked against Aleph Alpha's own
  serving code.

- **EmbeddingGemma 2 takes audio clips of up to 30 seconds;** on NVIDIA GPUs
  video is not served yet.

- **Laya reads text only**, and its confidence on choices with more than ten
  options is not calibrated: the English checkpoint ships an out-of-range
  temperature for that case, which Paddock clamps, as Laya's own server does.

- **Kumo Tabular predicts up to ten classes.**

- **On Qwen 3.8 Flash-Next's NVFP4 versions a greedy speculative reply can
  part from plain decoding at a near tie** - the same model with slightly
  different numerics, not a wrong answer.

- On-demand loading covers Whisper only.

- The fp8 KV cache's paged attention on RTX 50-series cards shows a small
  numeric deviation in one split configuration (#5).
