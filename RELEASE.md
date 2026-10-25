# Paddock 0.1.19

New models, batch inference and a lot of B200 work. NVIDIA's Nemotron 3.5 family arrives for reading pages, search and live transcription, Gemma 4 12B takes pictures and speech, and the manager can now run OpenAI and Anthropic batch jobs.

Builds for Windows x64, Linux x64 and the DGX Spark, on NVIDIA GPUs with driver 580 or newer. The macOS pre-release is built from the same commit.

## New

- **Nemotron 3.5 Parse** reads a scanned page, a photo of a document or a PDF page into its blocks in reading order, each with its box, its kind (title, text, table, caption and so on) and its text. Tables come back as LaTeX. It takes one page per request.
- **Nemotron 3.5 Embed 8B** puts text and pictures of pages in one vector space, and **Nemotron 3.5 Rerank 8B** scores how well each document answers a query, from its text, a picture of the page, or both.
- **Nemotron 3.5 ASR Streaming** transcribes live audio in 40 languages and locales. Words come out as they're heard and are never revised. It serves `/v1/audio/transcriptions` and the realtime WebSocket.
- **Gemma 4 12B** takes pictures and speech straight into the model, with no separate encoder. It has a 256K context, tools and reasoning, under Apache 2.0.
- **Batch inference** on the manager: OpenAI's `/v1/files` and `/v1/batches` and Anthropic's `/v1/messages/batches`, at the vendors' own paths, so the official SDKs only need a base URL. A batch can hold a million requests, uploads resume, a full disk pauses the batch instead of failing it, and every result is written exactly once, even across a manager crash.

## Improved

- Live requests now go ahead of batch work. With a batch running flat out on the DGX Spark, an agent session's time to first token went from about 33 s to 0.4-0.6 s, and its prompt cache hits again. Batch throughput didn't change.
- Qwen 3.8 27B on the B200 now runs its matrix work on Blackwell's newest tensor-core instructions, for NVFP4 and 8-bit weights. Reading pages with 128 in flight went from about 223 to about 370 pages a minute over this release.
- Speculative decoding on the DGX Spark keeps working past 8 requests at once, where it used to switch itself off. At 16 concurrent replies, Qwen 3.6 35B-A3B went from 22.2 to 25.0 tokens/s per reply, Qwen 3.5 9B from 23.4 to 27.4, and Qwen 3.8 27B from 7.5 to 12.2.
- At low concurrency every request now drafts to the full depth. A single Qwen 3.6 35B-A3B reply on the DGX Spark went from 80.3 to 87.1 tokens/s.
- Picture prompts reuse the prompt cache. With Qwen 3.8 27B NVFP4, a second page that shares the same instructions went from 3.52 s to 1.67 s, and four new pages at once from 13.9 s to 8.0 s.
- Gemma 4 at Q8_0 reads prompts faster: 9.2K tokens went from 3.31 s to 2.84 s, and 256 parallel streams run 11-13% faster. Output is bit-identical.
- Q8_0 models decode faster at low concurrency. The matrix kernel for 1 to 16 rows is 1.15-1.43x faster, with bit-identical output.
- Chat prompts are tokenized in cached pieces on several threads. A 119K-token agent turn took 145 ms every turn; now it's 21 ms the first time and about 1 ms after.
- The speculation controller keeps its timing across a restart, so the first replies after a restart no longer start slow.

## Fixed

- Concurrent picture requests on Qwen 3.5 to 3.8 could turn into "!!!!" partway through an answer, in 3-25% of answers at 8 to 32 parallel requests. It was a race between the engine's two lanes.
- Text inside a request (a pasted file, a tool result, a tool argument) could be read as a chat turn marker and open a new turn, a system turn for example. Agents hit it just by reading code that contains those markers. Request content is now always plain text to the model.
- NVIDIA's ModelOpt NVFP4 export of Qwen 3.8 27B served its attention and output layers from the Q8_0 base without saying so. They now load as shipped.
- Streamed Responses tool calls now keep the same IDs in the final response, so the OpenAI Agents SDK matches tool results to the right calls.

## macOS (pre-release)

- The native app keeps your chat sessions, saves composer drafts, and has searchable model pickers.
- The workspace keeps its layout through resizing and fullscreen.
- Bonsai and Gemma run faster on MLX, and Bonsai reuses its own earlier replies from the prompt cache.

## Known issues

- Nemotron 3.5 Parse, Embed, Rerank and ASR Streaming, and Gemma 4 12B, run on NVIDIA GPUs only for now.
- PaddleOCR-VL's layout pipeline is NVIDIA-only; on the Mac it reads whole pages.
- On the Mac, LightOnOCR-3 0.8B on MLX reads the same text as the reference, but a few boxes are off by up to about half a percent of the page.
- On the Mac, Nemotron on MLX matches the reference's first token on every test prompt, but not yet the full reply on all of them.
- SAM 3 and TIC Forestry are NVIDIA-only, and their downloads need a Hugging Face account that has accepted the model licence. TIC Forestry is made for Swedish aerial imagery at 0.5 m.
- Kolibri 1 can't be checked against llama.cpp yet, since no llama.cpp release reads it. We check it against Aleph Alpha's own serving code instead.
- EmbeddingGemma 2 audio clips are limited to 30 seconds.
- Laya is text-only, and its confidence isn't calibrated for more than ten options. The English checkpoint ships an out-of-range temperature for that case, which we clamp the same way Laya's own server does.
- On the NVFP4 versions of Qwen 3.8 Flash-Next, a greedy speculative reply can differ from plain decoding at a near tie. That's numerics, not a wrong answer.
- On-demand loading only covers Whisper so far.
