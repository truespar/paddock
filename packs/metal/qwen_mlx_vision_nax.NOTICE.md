# MLX register-fragment attention

`qwen_mlx_vision_nax.metal` adapts the fragment mapping, 16x32x16 register
contractions and online-softmax traversal from Apple's MLX v0.32.3:

- https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/kernels/steel/attn/nax.h
- https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h

Copyright © 2024–2026 Apple Inc.

Paddock specializes the implementation to head width 64, 12/16-head ragged
image descriptors and F32 output storage with a BF16 rounding boundary. It
retains F32 probabilities and accumulators, bounds every tail load/store, and
uses this fixed register mapping only on Apple10. Earlier Apple GPU families
retain the layout-independent MPP implementation.
The MLX runtime is not linked or redistributed.

`qwen_mlx_attention_nax.metal` is a head-256 language-attention implementation
based on the same upstream head-dimension-split implementation.
It uses Paddock's paged KV addressing, 32-row scheduler tiles and bounded
causal loads. Its election is limited to the native 0.8B tied-head MLX vision
backbone on Apple10; 4B remains diagnostic-only. No reference replay is
available at runtime.

The companion `qwen_mlx_vision.metal` also preserves the normalization and
convolution arithmetic contracts of MLX v0.32.3. Reference sources:

- https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/kernels/layer_norm.metal
- https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/kernels/steel/conv/kernels/steel_conv_3d.h
- https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/kernels/steel/gemm/mma.h

Upstream license (https://github.com/ml-explore/mlx/blob/v0.32.3/LICENSE):

MIT License

Copyright © 2023 Apple Inc.

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
