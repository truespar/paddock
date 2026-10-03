# Nemotron 3 Diarization

The streaming-state and arrival-order-cache algorithms in
`crates/paddock-engine/src/diarization.rs` and
`crates/paddock-models/src/diarization.rs` are adapted from MLX Audio,
revision 94c7716212b2228f178d2f9c7619a591fd1b0b78:
https://github.com/Blaizzy/mlx-audio

MIT License

Copyright (c) 2024 Prince Canuma

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Architecture and GGUF schema reference (no C++ code incorporated): NVIDIA
NeMo-Speech.cpp, revision 4c101bc7113f49101a3e11d2c994c519f41939f6,
https://github.com/NVIDIA/NeMo-Speech.cpp (Apache-2.0).

Operation-order and numerical reference for the CUDA lane
(`packs/cuda/src/diarization.cuh`; no Python code incorporated): Hugging Face
Transformers `nemotron3_diarization`, revision
a005fc82babfe8871d87746decad2dbee100a125,
https://github.com/huggingface/transformers (Apache-2.0).

The model weights are separate, under OpenMDW-1.1; Paddock's code licence
does not relicense them. No weights are included in this repository.
