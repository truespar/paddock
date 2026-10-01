# Paddock 0.1.12

A tables release: Kumo Tabular, NVIDIA's model that predicts a missing
column from labelled rows, runs natively on NVIDIA GPUs and on the Mac, with
a Tables page in the Studio and the Mac app. Requests with many images or
long audio use far less memory, Gemma 4 no longer fails on the DGX Spark and
RTX 50-series cards with an f16 KV cache, and long Qwen 3.8 Flash-Next
prompts prefill several times faster. Windows x64, Linux x64 and the NVIDIA
DGX Spark. NVIDIA GPUs, driver 580 or newer. The macOS pre-release for Apple
Silicon is built from the same commit.

## New

- **Kumo Tabular.** Give it labelled rows and it predicts the missing column
  of new ones - a class with its confidence, or a number with an 80% range -
  with no training: the labelled rows are read in context on every call.
  Small, medium and large, each for classification and for regression, in
  the F32 weights NVIDIA ships, behind the same `/v1/tabular/predictions`
  API on NVIDIA GPUs and on the Mac.

- **Tables in the Studio and the Mac app.** Paste or open a CSV, pick the
  column to predict, and the rows with it empty come back filled in; copy
  the result as CSV, or see the equivalent curl. Every table and its runs
  are saved and shared between the web Studio and the Mac app, and each
  table has its own link.

## Improved

- **Pictures cost far less memory.** The images in a prompt are encoded as
  each prefill pass needs them, into memory the plan sets aside, instead of
  all at once and held twice. On the DGX Spark an 18K-token prompt with a
  picture peaks 1.3 GB above idle instead of 13.6 GB. This covers the Qwen
  3.5-3.8 models, Gemma 4, Granite Vision and PaddleOCR-VL.

- **A Qwen prompt with pictures no longer pauses other conversations** while
  its pictures are read: on the DGX Spark the longest pause another session
  saw went from 15.4 s to 3.3 s.

- **Long audio clips use a fixed amount of memory** in Qwen3-ASR and Granite
  Speech, with the same transcript as before.

- **Qwen 3.8 Flash-Next prefills long prompts much faster** on its NVFP4
  versions - a 16.5K-token document question on the DGX Spark went from
  85 s to under 15 s - keeps its prefix cache across agent sessions, and an
  exact re-send of a long prompt resumes from its last walk instead of from
  the start.

- **An idle server hands the memory it freed back** instead of holding it
  until the next request.

- **Quieter logs.** A warning now means a client was held up or a real
  fallback happened; long prompts and normal startup no longer warn.

## Fixed

- **A Qwen 3.8 server on the DGX Spark could run out of memory after about a
  day:** startup set aside up to 20 GB it never used. Under memory pressure
  the system now stops the runner, not the manager.

- **Gemma 4 with an f16 KV cache failed every request on the DGX Spark and
  RTX 50-series cards.** It now serves there.

- **A long Gemma 4 prompt with a picture could get a wrong answer** when the
  picture fell across two prefill passes.

- **Safetensors checkpoints start from the Studio.** The NVFP4 and FP8
  versions of Qwen 3.8 Flash-Next, Nemotron and Granite were refused when
  started from the Studio.

- **Switching Vision off switches it off.** A vision file beside the model
  is no longer loaded anyway, and saving the endpoint through the Advanced
  tab keeps Vision off.

- **The start form offers document and image features only to models that
  chat.**

## macOS (pre-release)

- Kumo Tabular runs natively on Metal, with a native Tables workspace.
- Qwen 3.8 Flash-Next decodes and prefills faster on Metal.

## Known

- **Laya reads text only.**

- **Laya's confidence on choices with more than ten options is not
  calibrated:** the English checkpoint ships an out-of-range temperature for
  that case, which Paddock clamps, as Laya's own server does.

- **Kumo Tabular predicts up to ten classes.**

- **On Qwen 3.8 Flash-Next's NVFP4 versions a greedy speculative reply can
  part from plain decoding at a near tie** - the same model with slightly
  different numerics, not a wrong answer.

- On-demand loading covers Whisper only.

- The fp8 KV cache's paged attention on RTX 50-series cards shows a small
  numeric deviation in one split configuration (#5).
