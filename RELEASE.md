# Paddock 0.1.20

Two new models and a context cache that survives a restart. NVIDIA's Nemotron 3.5 Super VL reads pictures and documents, Phonon turns text into speech on NVIDIA GPUs and the Mac, and conversations now come back after a restart with their cache intact.

Builds for Windows x64, Linux x64 and the DGX Spark, on NVIDIA GPUs with driver 580 or newer. The macOS pre-release is built from the same commit.

## New

- **Nemotron 3.5 Super VL 120B-A12B** on NVIDIA GPUs, from NVIDIA's NVFP4 checkpoint. It reads pictures and documents, thinks before answering, calls tools, and drafts with its own built-in MTP head. Only 8 of its 88 layers keep a KV cache, so long contexts stay cheap. The weights are about 75 GB, so it wants a DGX Spark or a 96 GB Blackwell card.
- **Phonon 31M** from Gradium turns English text into speech on NVIDIA GPUs and the Mac, as 24 kHz WAV or streaming PCM through `/v1/audio/speech`. On the DGX Spark it runs about 200 times faster than real time, and the first audio arrives in 1.6-3 ms.

## Changed

- API keys now have a scope. An inference key (the default) reaches the OpenAI and Anthropic endpoints only; an admin key also reaches the management API and the Studio login. Keys made before this release count as inference keys, so if you used one for `/api` or to sign in to the Studio, create an admin key with `{"scope": "admin"}` on `POST /api/keys`. The manager's configured key and its automatic network key are admin keys.
- A stdio MCP server's environment can no longer set `LD_*` or `DYLD_*` variables. Connecting to one that does fails with a clear error.

## Improved

- Conversations come back after a restart. With 24 agent sessions on the DGX Spark, 24 of 24 resumed from their cache after a restart, where 5 did before. Shutdown now saves what's still on the GPU, and the last copy of a session's state goes to disk before RAM evicts it.
- Restoring a conversation from disk is faster and no longer slows everyone else down. The median restore went from 3.44 s to 0.95 s, with reads at the drive's full speed and off the engine thread.
- The disk cache writes less and keeps more of what it writes. It stores each conversation's newest state once it has been idle for 30 s, instead of every change. On the same workload it wrote 18.9 GiB instead of 54.8 GiB, and 65% of it was read back, against 4% before.
- The disk cache backs off a full or failing disk instead of failing requests, keeps a tenth of its volume free (at most 32 GiB), and frees the space of what it evicts. Writes are capped at a quarter of the drive's size a day, at most 1 TiB.

## Fixed

- A request could restore an older checkpoint of a conversation while a newer one was still on its way into RAM, and recompute more than it needed to. It now waits for the newer one when that's quicker than recomputing.

## macOS (pre-release)

- Phonon runs natively on Metal, with streamed audio.

## Known issues

- Nemotron 3.5 Super VL, Parse, Embed, Rerank and ASR Streaming, and Gemma 4 12B, run on NVIDIA GPUs only for now.
- Phonon has one English voice, no voice cloning or speed control, and doesn't spell out numbers or currency.
- PaddleOCR-VL's layout pipeline is NVIDIA-only; on the Mac it reads whole pages.
- On the Mac, LightOnOCR-3 0.8B on MLX reads the same text as the reference, but a few boxes are off by up to about half a percent of the page.
- On the Mac, Nemotron on MLX matches the reference's first token on every test prompt, but not yet the full reply on all of them.
- SAM 3 and TIC Forestry are NVIDIA-only, and their downloads need a Hugging Face account that has accepted the model licence. TIC Forestry is made for Swedish aerial imagery at 0.5 m.
- EmbeddingGemma 2 audio clips are limited to 30 seconds.
- Laya is text-only, and its confidence isn't calibrated for more than ten options. The English checkpoint ships an out-of-range temperature for that case, which we clamp the same way Laya's own server does.
- On the NVFP4 versions of Qwen 3.8 Flash-Next, a greedy speculative reply can differ from plain decoding at a near tie. That's numerics, not a wrong answer.
- On-demand loading only covers Whisper so far.
