# Paddock 0.1.13

A decisions and speech release: Cloudflare's Clef and Clef Flash answer
fixed questions about a text or a picture, NVIDIA's Nemotron 3 Diarization
tells who spoke when, and Reads gains a Camera mode that asks your questions
of the webcam live. The DGX Spark now reports its memory. Windows x64,
Linux x64 and the NVIDIA DGX Spark. NVIDIA GPUs, driver 580 or newer. The
macOS pre-release for Apple Silicon is built from the same commit.

## New

- **Clef and Clef Flash** (Cloudflare). Decision models: give them a text, a
  picture or both, and a set of yes/no, choice and score questions, and each
  answer comes back with a calibrated probability, every option scored in one
  pass. Clef Flash is built on a 9B model and Clef on a 27B one; both serve
  in 8-bit with their own vision, on NVIDIA GPUs and on the Mac, behind the
  same `/v1/systemone` call the Reads page uses. Clef Flash needs about 16 GB
  of GPU memory, Clef about 37 GB.

- **Nemotron 3 Diarization** (NVIDIA). Who spoke when, for a recording or
  live audio, on NVIDIA GPUs and on the Mac. The Studio shows a speaker
  timeline beside the transcript, with each word attributed to its speaker,
  and keeps it with the conversation.

- **Camera mode in Reads.** Switch Reads to Camera and your questions are
  asked of the webcam frame after frame. The answers ride on the picture as
  large cards - yes in green, no in red, a choice in white, each with its
  confidence - and a card flashes when its answer flips. Snapshot keeps a
  frame and its answers in the read's history. In the web Studio and the Mac
  app, with any reader that takes pictures.

## Improved

- **The start form proposes the weights `paddock serve` would start**, and a
  full-precision build is labelled "Full precision - no quantization" instead
  of being offered as a smaller one.

- **`paddock ps` names decision models** such as Laya and Clef instead of
  showing "-".

- **Models from Cloudflare show Cloudflare's logo.**

## Fixed

- **The DGX Spark reports its memory.** The Studio said "No GPU" beside every
  model on a Spark, and starting a model there was never checked against the
  memory it needs. Both now read the Spark's unified memory.

- **A cloud model refuses a malformed output cap.** A `max_output_tokens`
  that is not a whole number used to be forwarded, dropped or replaced
  depending on the provider; it is now refused with a 400 before anything is
  sent. Thanks to @DevChiniwala (#35).

## macOS (pre-release)

- Clef and Clef Flash run natively on Metal, in 8-bit, with vision.
- Reads has a native camera, with snapshots saved in the read's history.
- Nemotron 3 Diarization runs on Metal, with speaker timelines in the app.

## Known

- **Clef reads text and pictures; videos are refused.**

- **Nemotron 3 Diarization is qualified against NVIDIA's reference on the
  pyannote and AMI test sets;** other recordings are not measured yet.

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
