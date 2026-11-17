# Paddock 0.1.21

Three new models, and batches you can watch. Recursive's Flint is a coding model with 2.5B of its 40B active, Baidu's Unlimited-OCR 2 reads a whole document in one request, and NVIDIA's Nemotron 3.5 Transcribe turns recordings into text with word times and speakers. The Studio now shows every batch on the box.

Builds for Windows x64, Linux x64 and the DGX Spark, on NVIDIA GPUs with driver 580 or newer. The macOS pre-release is built from the same commit.

## New

- **Flint** from Recursive, 40B with 2.5B active, made for coding agents. It's served from Recursive's own NVFP4 GGUF on NVIDIA GPUs from Ada on, with tool calls and thinking. On the DGX Spark an 8K-token prompt is read in about 1 s, and a single reply runs at about 100-114 tokens/s.
- **Unlimited-OCR 2** from Baidu, 3B, reads a whole multi-page document in one request and returns every page as labelled blocks (titles with their level, text, tables as HTML), each with its box. Its memory stays flat however long the output runs, and about 30 dense pages fit its 32K context. Set `ocr.crop` to `"gundam"` to tile every page when the print is small.
- **Nemotron 3.5 Transcribe 30B-A4B** from NVIDIA transcribes recordings of up to an hour in 40 languages, with word timestamps and speaker labels, and also translates speech and answers questions about a clip. It serves `/v1/audio/transcriptions` (including `word` timestamps and `diarized_json`), `/v1/audio/translations`, chat completions and Responses. On the DGX Spark, 64 recordings at once transcribe about 225 times faster than real time.
- Manage > Batches in the Studio lists every OpenAI and Anthropic batch on the box with its progress, rate and time left. Open one to see its results and failures, download its files, or cancel or delete it.
- `/v1/audio/translations` on the models whose cards translate: Nemotron 3.5 Transcribe, Granite Speech and Gemma 4 12B.
- In the Studio, Live now works with models that both chat and hear audio, such as Transcribe, and a question typed beside a clip is answered about that clip.

## Changed

- Structured outputs accept nullable fields, written as `type: [T, "null"]` (OpenAI's strict mode) or as `anyOf` with null (pydantic), and enums without a `type`.
- Transcriptions take uploads up to 1 GiB, about 9 hours of WAV, and up to 8 hours of decoded audio.
- Diarization takes recordings up to 4 hours instead of 10 minutes, and a live session runs for as long as its audio keeps up with the clock.
- A request with both pictures and audio is refused on chat completions and Responses. Chat completions used to drop the audio.
- When a speech model's engine fails, the error is now a 500 `internal_error` instead of a 503, so clients stop retrying a request that can't succeed.

## Improved

- Unlimited-OCR reads faster under load, and Unlimited-OCR 2 shares the work. On the DGX Spark with 32 requests at once, pages per second went from 0.78 to 1.22, and the first token of a tiled burst arrives in 3.75 s instead of 5.27 s.
- Nemotron models keep their speculative drafts when a new prompt arrives mid-reply. Before, every reply in flight fell back to one token for that step.

## Fixed

- Whisper failed on any clip shorter than about 3 s, and on clips whose end fell in the first 3 s of a 30 s window, with "got the special token 50363". The decoder now suppresses those tokens the way OpenAI's does.
- The Responses API dropped audio parts and answered without the clip, on every audio model.
- On every Nemotron model, a race in the prompt pass could change a reply between identical runs while another request kept the GPU busy.

## Known issues

- Flint, Unlimited-OCR 2, Nemotron 3.5 Transcribe, Super VL, Parse, Embed, Rerank and ASR Streaming, and Gemma 4 12B, run on NVIDIA GPUs only for now. Flint needs Ada or newer. Transcribe and Super VL need an RTX 50-series or RTX PRO Blackwell card or the DGX Spark, and don't run on the B200 yet.
- On the Mac, Whisper still fails clips shorter than about 3 s. The fix above hasn't reached Metal yet.
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
