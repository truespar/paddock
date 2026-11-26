# Paddock 0.1.22

Two new models for agents. Thinking Machines' Inkling 2 Mini is a 26B mixture-of-experts that reads pictures and lets you choose how long it thinks, and Ricursive's Ripple 65B Preview is a dense coding model that speeds itself up with its own draft model. This release also fixes a race in the 8-bit activation path on Blackwell GPUs.

Builds for Windows x64, Linux x64 and the DGX Spark, on NVIDIA GPUs with driver 580 or newer. The macOS pre-release is built from the same commit.

## New

- **Inkling 2 Mini** from Thinking Machines, 26.4B with 5.4B active, served from the official NVFP4 and MXFP8 release. It's built for agents, tool use and coding, calls several tools at once, and reads pictures and screenshots at their own resolution, one token for every 40 x 40 pixel cell. Pick its thinking effort from minimal to max, or cap its thinking with `thinking.budget_tokens` on Messages or `reasoning.max_tokens` on chat completions and Responses, so Claude Code's extended thinking works with it. Its own draft model speeds up single chats, and a conversation's earlier turns come from the prompt cache. On the DGX Spark a single reply runs at about 39 tokens/s, a 3.2K-token prompt gets its first token in about 1 s, and the full 256K window is the default.
- **Ripple 65B Preview** from Ricursive, a dense 65B coding model for terminal work and long multi-step tasks, served from Ricursive's NVFP4 release with its DFlash2 draft model on by default. On the DGX Spark one reply runs at about 24 tokens/s, against about 4 without the draft model, and four at once at about 51 tokens/s in total. It needs about 65 GB, so a DGX Spark or a 96 GB RTX PRO Blackwell card.

## Fixed

- On Blackwell GPUs, the 8-bit activation path could read part of its input before the step before it had finished writing it, once a pass reached 96 or more rows on a wide layer, so some rows were computed from stale data. It turned up while bringing up Ripple's draft model, and it's fixed for every model that uses that path.

## Known issues

- Ripple 65B Preview, Inkling 2 Mini, Flint, Unlimited-OCR 2, Nemotron 3.5 Transcribe, Super VL, Parse, Embed, Rerank and ASR Streaming, and Gemma 4 12B, run on NVIDIA GPUs only for now. Flint needs Ada or newer. Ripple, Inkling, Transcribe and Super VL need an RTX 50-series or RTX PRO Blackwell card or the DGX Spark, and don't run on the B200 yet.
- Inkling 2 Mini still reasons at its lowest thinking effort, and in long reasoning it can get stuck re-checking its own work, so set output and turn limits. Prompts with pictures don't use its prompt cache yet.
- Ripple 65B Preview is text only.
- On the Mac, Whisper still fails clips shorter than about 3 s. The fix in 0.1.21 hasn't reached Metal yet.
- Unlimited-OCR 2 has one trained mode, multi-page parsing. There's no free-text or figure mode.
- On a live socket, Nemotron 3.5 Transcribe times and scores each word but doesn't label speakers, since each utterance would number its own.
- Phonon has one English voice, no voice cloning or speed control, and doesn't spell out numbers or currency.
- PaddleOCR-VL's layout pipeline is NVIDIA-only; on the Mac it reads whole pages.
- On the Mac, LightOnOCR-3 0.8B on MLX reads the same text as the reference, but a few boxes are off by up to about half a percent of the page.
- On the Mac, Nemotron on MLX matches the reference's first token on every test prompt, but not yet the full reply on all of them.
- SAM 3 and TIC Forestry are NVIDIA-only, and their downloads need a Hugging Face account that has accepted the model licence. TIC Forestry is made for Swedish aerial imagery at 0.5 m.
- EmbeddingGemma 2 audio clips are limited to 30 seconds.
- Laya is text-only, and its confidence isn't calibrated for more than ten options. The English checkpoint ships an out-of-range temperature for that case, which we clamp the same way Laya's own server does.
- On the NVFP4 versions of Qwen 3.8 Flash-Next, a greedy speculative reply can differ from plain decoding at a near tie. That's numerics, not a wrong answer.
- On-demand loading only covers Whisper so far.
