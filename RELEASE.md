# Paddock 0.1.16

A document-reading release: LightOnOCR-3 joins the page readers,
PaddleOCR-VL reads pages through its own layout model the way its authors'
pipeline does, boxes on Qwen pictures land where they should, and the
context cache in RAM and on disk now brings back hybrid models such as
Qwen 3.8, after a restart too. Windows x64, Linux x64 and the NVIDIA DGX
Spark. NVIDIA GPUs, driver 580 or newer. The macOS pre-release for Apple
Silicon is built from the same commit.

## New

- **LightOnOCR-3** (LightOn), 0.8B and 4B. Reads a page as text, or in
  grounding mode as labelled blocks with their boxes, which the Studio
  draws over the page. PDF pages are rendered at the model's own
  2048-pixel page size. Over the API, send `ocr: {"mode": "grounding"}` on
  chat completions, Messages or Responses. It serves with the 8-bit context
  cache by default: 17-23% more pages a minute at 64 in flight, no page
  read worse.

- **PaddleOCR-VL reads documents the way its authors' pipeline does.** A
  layout model (PP-DocLayoutV3) finds the text, tables, formulas and
  figures on the page, PaddleOCR-VL reads each region in reading order, and
  the page comes back as Markdown. On our 64-page test set the word error
  rate is 0.65%, and 60 of the 64 pages come back byte-identical to the
  official pipeline's Markdown. The layout model downloads with
  PaddleOCR-VL and is the default for a page sent to chat completions. On
  NVIDIA GPUs.

- **The context cache restores hybrid models** - Qwen 3.5, 3.6 and 3.8
  27B, and Nemotron 3.5, whose layers carry a running state. Their saved
  states used to be found and never used; now a re-asked document or a
  later turn comes back from RAM or disk instead of being read again, and
  the disk copy survives a restart and is used from the first prompt after
  it. Qwen 3.8 27B on the DGX Spark: re-asked documents start answering in
  0.2-0.3 s instead of 7-12 s.

## Improved

- **Agents see tool calls as they are written.** A tool call used to arrive
  whole once its block closed, so writing a large file was minutes of
  silence and then one blob. On Qwen's tool-call format the arguments now
  stream on all three APIs; other formats send each call as soon as it
  closes.

- **Pictures inside tool results reach the model**, such as Claude Code
  reading an image file, on models whose chat template takes them (Qwen 3.5
  to 3.8). Elsewhere the model gets a note that a picture was there instead
  of the request failing.

- **Long agent sessions on Nemotron keep their place.** A new user turn
  resumes from the cache instead of reading the whole conversation again,
  which took 34-58 s at 150-216K tokens.

- **A conversation decoding beside another one's long prompt no longer
  stalls** for about a second per step: on the DGX Spark one stream beside
  a 16K-token prompt went from about 3 to 4.65 tokens a second.

- **Faster pictures and pages.** The picture towers of Qwen 3.5 to 3.8,
  LightOnOCR-3 and PaddleOCR-VL run their steps fused and their attention
  in half precision, with the same results to the bit: a LightOnOCR-3 page
  goes through the tower in 354 ms instead of 590 ms on the DGX Spark. Page
  readers take two pages a pass, so 64 pages in flight reach their first
  token about a quarter sooner.

- **PaddleOCR-VL stops a page at its first repeated phrase** instead of
  looping to the token limit, and the Studio marks such a page for review.
  `ocr.repetition_stop: false` turns it off.

- **Models with tied embeddings hold one copy** of the embedding matrix
  instead of two: LightOnOCR-3 4B needs about 0.7 GB less.

## Fixed

- **Boxes and points on Qwen 3.5, 3.6 and 3.8 27B pictures land where they
  should** on NVIDIA GPUs. Three differences from the reference implementation - how
  picture positions are rotated, how the position table is resized, and
  which picture rows see each other - moved the boxes, on some test
  pictures to no overlap with the right answer at all. Pictures are now
  also resized with the reference's bicubic filter. Text replies are
  unchanged.

- **Many requests in flight could crash the engine or stall every
  request** on models served without a draft model, Qwen 3.5 9B and the
  page readers among them. 64 LightOnOCR-3 pages at once now complete.

## macOS (pre-release)

- LightOnOCR-3 reads pages on Metal, from the GGUF or an MLX package, with
  the grounding switch in the native app and the Studio.
- LightOn's mark on the LightOnOCR-3 rows, and the OpenBMB and The
  Intelligence Company marks in the native app.
- SAM 3's text side is in place on Metal; SAM 3 itself is not served on the
  Mac yet.

## Known

- **PaddleOCR-VL's layout pipeline runs on NVIDIA GPUs only;** on the Mac
  it reads whole pages.

- **On the Mac, LightOnOCR-3's MLX build reads the same text as the
  reference on every test page, but a few grounding boxes sit up to about
  half a percent of the page away from the reference's.**

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
