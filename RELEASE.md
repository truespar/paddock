# Paddock 0.1.14

A new-models release: Aleph Alpha's Kolibri 1, Google's EmbeddingGemma 2,
and Meta's SAM 3, which masks every instance of what you name in a picture
or a live camera, with a Masks page in the Studio. Windows x64, Linux x64
and the NVIDIA DGX Spark. NVIDIA GPUs, driver 580 or newer. The macOS
pre-release for Apple Silicon is built from the same commit.

## New

- **Kolibri 1** (Aleph Alpha). A German-English mixture-of-experts model for
  reasoning, retrieval and tool calling, with a 262K context. On NVIDIA GPUs
  it serves a 4-bit GGUF on every card, with the NVFP4 checkpoint offered on
  Blackwell; on the Mac a mixed 4/8-bit build. Plan for at least 64 GB of GPU
  or unified memory.

- **EmbeddingGemma 2** (Google). Turns text, pictures and audio into vectors
  in one shared space, so photos and recordings can be searched with words.
  The text model is 0.3 GB; the picture tower (0.4 GB, on by default) and
  the audio tower (0.6 GB, off by default) are switches on the start form,
  and the download follows them. The Embeddings page takes pictures and
  audio. On NVIDIA GPUs.

- **SAM 3** (Meta). Name a thing - "person", "red car" - and it returns a
  mask, box and score for every one in the picture; draw boxes to say what
  to include or leave out, or click on a part of the picture to mask that
  object. Asking again about the same picture only costs the new prompt.
  Over the API, `/v1/masks` answers a picture and `/v1/masks/sessions`
  follows objects from frame to frame. On NVIDIA GPUs.

- **The Masks page.** Drop, paste or open a picture, or turn on the camera
  and track up to four named things at once, each in its own colour, with
  snapshots of the moments you keep. Every picture is kept in a side panel
  like Tables and Reads, and exports as COCO JSON, cut-outs or mask PNGs.

- **Gated downloads with your own Hugging Face token.** SAM 3's weights are
  not redistributed: accept Meta's licence on its Hugging Face page and add
  a token in Manager > Settings, and the download comes straight from
  Hugging Face. A refused download now says so, instead of "the file is
  gone".

## Improved

- **The context cache is 8-bit by default on most models**, so the same
  memory holds about twice the context and long agent sessions run faster.
  The OCR and speech models keep their 16-bit cache.

- **A start that names no context window gets a long one sized to the
  card.** Qwen 3.8 27B, Nemotron and Kolibri ask for 262K on NVIDIA GPUs and
  fall back to the largest window that fits; a window you type is never
  shrunk.

- **Nemotron answers faster on NVIDIA GPUs.** On Blackwell it defaults to
  its NVFP4 weights with their speculative drafter, and each speculative
  round does less work.

- **Agent sessions resume instead of starting over** when the end of a
  prompt is rewritten - the same document with another question - on
  Nemotron, Qwen 3.5 to 3.8, Laguna and Kolibri.

- **Models with sliding-window attention get the context they can hold.**
  The memory estimate charged their window layers as if they kept growing,
  so Gemma 4, gpt-oss, Laguna and Kolibri were offered a fraction of the
  context that fits.

- **Reads, Tables and Masks share one page layout**, and switching between
  them no longer flashes the side panel.

## Fixed

- **Shipped builds use the speed settings the engine picks for each card.**
  Some of those choices were dropped in release builds, among them a faster
  decode on the DGX Spark and a faster prompt read on Qwen 3.5 to 3.8.

## macOS (pre-release)

- Kolibri 1 runs natively on Metal, in a mixed 4/8-bit build.
- Qwen models in MLX format decode faster at small batch sizes, and their
  speculative rounds cost less.
- The settings pages share one layout.
- Updates are checked against a pinned signing identity.

## Known

- **SAM 3 runs on NVIDIA GPUs only**, and its weights download only with a
  Hugging Face account that has accepted Meta's licence.

- **Kolibri 1 has no same-weights llama.cpp reference yet:** no llama.cpp
  release reads it, so its correctness is checked against Aleph Alpha's own
  serving code.

- **EmbeddingGemma 2 takes audio clips of up to 30 seconds;** video is not
  served yet.

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
