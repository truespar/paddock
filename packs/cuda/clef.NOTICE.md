# Clef

The request encoding in `crates/paddock-runner/src/systemone/clef/` (the
joint sequence, its token spans, the question and option rules and the
answer shapes) and the joint schema head in `packs/cuda/src/clef.cuh` and
`crates/paddock-engine/src/gpu_model/clef/` follow Cloudflare's reference
implementation, `joint_schema_model.py`, at revision
17f0b0ad64efb65d273590632833508766b2aae6 of
https://huggingface.co/Cloudflare/clef-flash (Apache-2.0). It is the
operation-order and numerical reference; no Python code is incorporated.

The backbone's tensor transforms (value heads reordered to tiled order,
`A_log` stored as `-exp(A_log)`, the `+1` folded into the zero-centred
RMSNorms) follow llama.cpp's `conversion/qwen.py` (MIT), so the engine's
Qwen3.5 kernels read the layout they were qualified on; no code is
incorporated.

The GGUF lane (`crates/paddock-engine/src/gpu_model/clef/gguf.rs`, the
head's GGUF loader, `ClefConfig::from_gguf`) reads the `clef` GGUF schema
llama.cpp defines for Clef (MIT; `conversion/clef.py`, `gguf-py`'s tensor
names, `src/models/clef.cpp`, studied at merge commit
99b95488cac0f00ce3f05af113a8c1e287753f87): its metadata keys, the
`dec.blk.N` / `decision.*` tensor names, the split of the packed attention
projections and the three head scalars stored as used. No code is
incorporated. The GGUF files Paddock's catalog serves are ggml-org's
conversions of Cloudflare's releases (`ggml-org/Clef-Flash-GGUF`,
`ggml-org/Clef-GGUF`, Apache-2.0 as the weights they convert).

The image lane follows the reference's own image path, studied in
source; no code is incorporated from any of it:

- Transformers 5.10.2 (Apache-2.0): the Qwen2-VL image processor's
  `smart_resize`, pixel normalization and patch order, the Qwen3.5 vision
  model (patch embedding, the bilinear position-grid interpolation, the 2D
  vision rotary embedding, the merger) and `get_rope_index`
  (`packs/cuda/src/clef_vision.cuh`, `crates/paddock-engine/src/gpu_model/clef/`,
  `crates/paddock-models/src/clef.rs`);
- PyTorch (BSD-3-Clause, aten `UpSampleKernel.cpp`): the uint8 antialiased
  bicubic resize the processor runs - the weight formation, the int16
  quantization and the integer passes (slots 735-736);
- libjpeg-turbo 3.2.0 (IJG License, BSD-3-Clause, zlib License): the
  default JPEG decompression torchvision and Pillow link - Huffman
  decoding, the accurate integer IDCT and its range limit, the upsamplers
  and the color conversion - which `crates/paddock-jpeg` reproduces in its
  own code; and torchvision's CMYK-to-RGB conversion (BSD-3-Clause).

The model weights are separate, under Apache-2.0; Paddock's code licence
does not relicense them. No weights are included in this repository.
