// vision.cuh (formerly 06_vision.cuh) - vision encoder kernels (layernorm, gelu, vision attn, dp4a gemv)
// Textually-included segment of the single pack translation unit.
// Not standalone-compilable: include order is defined by ../pack.cu.
// --------------------------------------------------------------------- vision
// The Qwen3-VL-style ViT encoder ops (b9895 tools/mtmd/models/qwen3vl.cpp is the
// reference). All f32 - the encoder runs once per image; correctness first.

// Row-wise LayerNorm (mean/variance, weight + bias) - the ViT norm (the LLM uses
// RMSNorm; the vision tower uses classic LN). grid rows, block 256.
__global__ void pd_layernorm_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                    const float* __restrict__ b, float* __restrict__ out,
                                    uint32_t n, float eps) {
    uint32_t row = blockIdx.x;
    const float* xr = x + (size_t)row * n;
    float* orow = out + (size_t)row * n;
    uint32_t tid = threadIdx.x, nth = blockDim.x;
    __shared__ float wsum[32];
    __shared__ float s_mean, s_inv;
    uint32_t lane = tid & 31u, warp = tid >> 5, nwarps = (nth + 31u) >> 5;

    float acc = 0.0f;
    for (uint32_t i = tid; i < n; i += nth) acc += xr[i];
    for (uint32_t s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, s);
    if (lane == 0) wsum[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        float m = 0.0f;
        for (uint32_t i = 0; i < nwarps; ++i) m += wsum[i];
        s_mean = m / (float)n;
    }
    __syncthreads();
    float mean = s_mean;
    float vacc = 0.0f;
    for (uint32_t i = tid; i < n; i += nth) {
        float d = xr[i] - mean;
        vacc += d * d;
    }
    for (uint32_t s = 16; s > 0; s >>= 1) vacc += __shfl_down_sync(0xffffffffu, vacc, s);
    if (lane == 0) wsum[warp] = vacc;
    __syncthreads();
    if (tid == 0) {
        float v = 0.0f;
        for (uint32_t i = 0; i < nwarps; ++i) v += wsum[i];
        s_inv = rsqrtf(v / (float)n + eps);
    }
    __syncthreads();
    float inv = s_inv;
    for (uint32_t i = tid; i < n; i += nth) orow[i] = (xr[i] - mean) * inv * w[i] + b[i];
}

PD_EXPORT
int pd_layernorm(const void* x, const void* w, const void* b, void* out, uint32_t rows,
                 uint32_t n, float eps, void* stream) {
    if (rows == 0 || n == 0) return 0;
    pd_layernorm_kernel<<<rows, 256, 0, (cudaStream_t)stream>>>(
        (const float*)x, (const float*)w, (const float*)b, (float*)out, n, eps);
    return pd_launch_status();
}

// GELU (tanh approximation - Exactly ggml_gelu_f32, the parity reference) applied
// in place over n elements.
__global__ void pd_gelu_kernel(float* __restrict__ x, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    x[i] = 0.5f * v * (1.0f + tanhf(0.79788456080286535587989211986876f * v
                                    * (1.0f + 0.044715f * v * v)));
}

PD_EXPORT
int pd_gelu(void* x, uint64_t n, void* stream) {
    if (n == 0) return 0;
    uint32_t threads = 256;
    uint64_t blocks = (n + threads - 1) / threads;
    pd_gelu_kernel<<<(uint32_t)blocks, threads, 0, (cudaStream_t)stream>>>((float*)x, n);
    return pd_launch_status();
}

// Broadcast bias add: x[row][i] += bias[i] for rows x n.
__global__ void pd_bias_add_kernel(float* __restrict__ x, const float* __restrict__ bias,
                                   uint32_t rows, uint32_t n) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (uint64_t)rows * n) return;
    x[idx] += bias[idx % n];
}

PD_EXPORT
int pd_bias_add(void* x, const void* bias, uint32_t rows, uint32_t n, void* stream) {
    if (rows == 0 || n == 0) return 0;
    uint32_t threads = 256;
    uint64_t blocks = ((uint64_t)rows * n + threads - 1) / threads;
    pd_bias_add_kernel<<<(uint32_t)blocks, threads, 0, (cudaStream_t)stream>>>(
        (float*)x, (const float*)bias, rows, n);
    return pd_launch_status();
}

// Vision 2D rope (llama.cpp's clip_graph::build_rope_2d): the head splits into
// two INDEPENDENT rope blocks - dims [0, hd/2) roped by pos_x, dims [hd/2, hd)
// by pos_y; each block is its own ggml_rope_ext with n_dims = hd/2, so both
// restart their exponent at zero: theta_i = pos * ts^i with
// ts = theta_base^(-2/(hd/2)).
//
// NEOX is the pair layout ARGUMENT, not a tuning knob. The two towers we serve
// disagree on it and nothing in either GGUF says so - it is the `mode` argument
// llama.cpp passes to ggml_rope_ext, i.e. an architecture constant read off the
// reference graph:
//   NEOX  (gemma4v)       partner is half a half away: (i, i + hd/4)
//   NORM  (muse-glimmer)  partner is the ADJACENT element: (2i, 2i+1)
// Both cover the same hd/2 pairs per head and compute the same thetas in the
// same order, so the NEOX arm is bit-identical to the pre-template kernel.
//
// One THREAD per PAIR, the shape pd_mrope_kernel has always had.
// The first cut handed a whole head to a single thread and walked it
// serially, so consecutive threads sat head_dim floats apart and every
// 4-byte access was its own 32 B sector - the kernel ran at a fraction of
// the coalesced rate for arithmetic that is two multiplies. Here consecutive
// threads walk consecutive dims of the same head, so a warp's loads and
// stores fall in two contiguous runs. theta is rebuilt by the same repeated
// f32 multiply the serial loop used, so this stays bit-identical: pow() or a
// precomputed table would round differently.
template <bool NEOX>
__global__ void pd_rope2d_kernel(float* __restrict__ x,
                                 const unsigned int* __restrict__ pos_x,
                                 const unsigned int* __restrict__ pos_y,
                                 uint32_t n_tokens, uint32_t n_heads,
                                 uint32_t head_dim, float theta_scale) {
    uint32_t quarter = head_dim / 4;
    uint32_t pairs = 2u * quarter;                // per head, across both blocks
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_tokens * n_heads * pairs) return;
    uint32_t j = gid % pairs;
    uint32_t idx = gid / pairs;                   // (t, h) flat index
    uint32_t t = idx / n_heads;
    float* head = x + (size_t)idx * head_dim;
    // j < quarter is the x block at dims [0, half); the rest is the y block,
    // which restarts its exponent at its own base - two independent rope
    // blocks, not one sectioned sweep
    uint32_t i = (j < quarter) ? j : (j - quarter);
    uint32_t base = (j < quarter) ? 0u : (head_dim / 2);
    float theta = (j < quarter) ? (float)pos_x[t] : (float)pos_y[t];
    for (uint32_t k = 0; k < i; ++k) theta *= theta_scale;  // ts^i, serial order
    float sn = sinf(theta), cs = cosf(theta);
    const uint32_t e0 = NEOX ? (base + i) : (base + 2u * i);
    const uint32_t e1 = NEOX ? (base + i + quarter) : (base + 2u * i + 1u);
    float a = head[e0], b = head[e1];
    head[e0] = a * cs - b * sn;
    head[e1] = a * sn + b * cs;
}

static int pd_rope2d_impl(void* x, const void* pos_x, const void* pos_y, uint32_t n_tokens,
                          uint32_t n_heads, uint32_t head_dim, float theta_scale,
                          bool neox, void* stream) {
    uint32_t total = n_tokens * n_heads * 2u * (head_dim / 4);   // one thread per pair
    if (total == 0) return 0;
    uint32_t threads = 256;
    uint32_t blocks = (total + threads - 1) / threads;
    if (neox) {
        pd_rope2d_kernel<true><<<blocks, threads, 0, (cudaStream_t)stream>>>(
            (float*)x, (const unsigned int*)pos_x, (const unsigned int*)pos_y, n_tokens,
            n_heads, head_dim, theta_scale);
    } else {
        pd_rope2d_kernel<false><<<blocks, threads, 0, (cudaStream_t)stream>>>(
            (float*)x, (const unsigned int*)pos_x, (const unsigned int*)pos_y, n_tokens,
            n_heads, head_dim, theta_scale);
    }
    return pd_launch_status();
}

PD_EXPORT
int pd_rope2d_neox(void* x, const void* pos_x, const void* pos_y, uint32_t n_tokens,
                   uint32_t n_heads, uint32_t head_dim, float theta_scale, void* stream) {
    return pd_rope2d_impl(x, pos_x, pos_y, n_tokens, n_heads, head_dim, theta_scale,
                          /*neox=*/true, stream);
}

// Same rope with the pair layout as an argument - muse-glimmer's tower ropes
// NORM (see the header note). Older packs only ever roped NEOX, which is why
// this is a new export rather than a changed signature.
PD_EXPORT
int pd_rope2d(void* x, const void* pos_x, const void* pos_y, uint32_t n_tokens,
              uint32_t n_heads, uint32_t head_dim, float theta_scale, uint32_t neox,
              void* stream) {
    return pd_rope2d_impl(x, pos_x, pos_y, n_tokens, n_heads, head_dim, theta_scale,
                          neox != 0u, stream);
}

// Qwen3-VL vision M-RoPE (ggml GGML_ROPE_TYPE_VISION with indep_sects):
// full-head rotation, pair (p, p + head_dim/2); sections are head_dim/4 each
// and only the first two are reachable - pair p < s: theta = pos_y * ts^p,
// else theta = pos_x * ts^(p-s), with s = head_dim/4 and
// ts = base^(-2/(head_dim/2)). positions [4, n_tokens] axis-major
// = [y, x, y, x] per patch (b9895 clip.cpp fills it exactly so).
//
// One thread per pair, same reasoning as pd_rope2d_neox_kernel above
// `indep_sects` is what makes the exponent reconstructible in
// closed form: the serial loop's `theta_x = th_x` at the section boundary is
// the x section restarting its exponent at zero, so pair p wants
// (p < s) ? y·ts^p : x·ts^(p-s) - reached by the same repeated multiply, in
// the same order, for the same bits.
__global__ void pd_mrope_vision_kernel(float* __restrict__ x,
                                       const unsigned int* __restrict__ positions,
                                       uint32_t n_tokens, uint32_t n_heads,
                                       uint32_t head_dim, float theta_scale) {
    uint32_t half = head_dim / 2;
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_tokens * n_heads * half) return;
    uint32_t p = gid % half;
    uint32_t idx = gid / half;                    // (t, h) flat index
    uint32_t t = idx / n_heads;
    float* head = x + (size_t)idx * head_dim;
    uint32_t s = head_dim / 4;
    float theta = (p < s) ? (float)positions[t] : (float)positions[(size_t)n_tokens + t];
    uint32_t steps = (p < s) ? p : (p - s);
    for (uint32_t i = 0; i < steps; ++i) theta *= theta_scale;
    float sn = sinf(theta);
    float cs = cosf(theta);
    float a = head[p];
    float b = head[p + half];
    head[p] = a * cs - b * sn;
    head[p + half] = a * sn + b * cs;
}

PD_EXPORT
int pd_mrope_vision(void* x, const void* positions, uint32_t n_tokens, uint32_t n_heads,
                    uint32_t head_dim, float theta_scale, void* stream) {
    uint32_t total = n_tokens * n_heads * (head_dim / 2);   // one thread per pair
    if (total == 0) return 0;
    uint32_t threads = 256;
    uint32_t blocks = (total + threads - 1) / threads;
    pd_mrope_vision_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        (float*)x, (const unsigned int*)positions, n_tokens, n_heads, head_dim, theta_scale);
    return pd_launch_status();
}

// ----------------------------------------------------- tower elementwise fusion
//  The PaddleOCR-VL tower is elementwise-PASS-bound between its
// f16 GEMMs: per layer the unfused chain paid two LN->convert round-trips, six
// standalone bias passes, a separate GELU pass and a second f32->f16 convert on
// the wide FFN plane - ~625 MB of pure re-streaming per layer per 5040-patch
// image. Each fusion below keeps the unfused chain's arithmetic ORDER exactly
// (the same IEEE f32 adds in the same association, one final __float2half
// round), so every one is bit-identical to the two or three ops it replaces
// and the oracle-diffed vision gates hold unchanged.

// LayerNorm writing f16 directly - the GEMM staging dtype, so the LN->convert
// round-trip disappears. Body identical to pd_layernorm_kernel; f32 memory
// holds no extra precision, so rounding the register value here equals
// storing f32 then running pd_convert_f32_f16.
__global__ void pd_layernorm_f16_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                        const float* __restrict__ b, __half* __restrict__ out,
                                        uint32_t n, float eps) {
    uint32_t row = blockIdx.x;
    const float* xr = x + (size_t)row * n;
    __half* orow = out + (size_t)row * n;
    uint32_t tid = threadIdx.x, nth = blockDim.x;
    __shared__ float wsum[32];
    __shared__ float s_mean, s_inv;
    uint32_t lane = tid & 31u, warp = tid >> 5, nwarps = (nth + 31u) >> 5;

    float acc = 0.0f;
    for (uint32_t i = tid; i < n; i += nth) acc += xr[i];
    for (uint32_t s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, s);
    if (lane == 0) wsum[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        float m = 0.0f;
        for (uint32_t i = 0; i < nwarps; ++i) m += wsum[i];
        s_mean = m / (float)n;
    }
    __syncthreads();
    float mean = s_mean;
    float vacc = 0.0f;
    for (uint32_t i = tid; i < n; i += nth) {
        float d = xr[i] - mean;
        vacc += d * d;
    }
    for (uint32_t s = 16; s > 0; s >>= 1) vacc += __shfl_down_sync(0xffffffffu, vacc, s);
    if (lane == 0) wsum[warp] = vacc;
    __syncthreads();
    if (tid == 0) {
        float v = 0.0f;
        for (uint32_t i = 0; i < nwarps; ++i) v += wsum[i];
        s_inv = rsqrtf(v / (float)n + eps);
    }
    __syncthreads();
    float inv = s_inv;
    for (uint32_t i = tid; i < n; i += nth)
        orow[i] = __float2half((xr[i] - mean) * inv * w[i] + b[i]);
}

PD_EXPORT
int pd_layernorm_f16(const void* x, const void* w, const void* b, void* out, uint32_t rows,
                     uint32_t n, float eps, void* stream) {
    if (rows == 0 || n == 0) return 0;
    pd_layernorm_f16_kernel<<<rows, 256, 0, (cudaStream_t)stream>>>(
        (const float*)x, (const float*)w, (const float*)b, (__half*)out, n, eps);
    return pd_launch_status();
}

// bias + tanh-GELU + f16 store in one pass: replaces pd_bias_add(x,b) ->
// pd_gelu(x) -> pd_convert_f32_f16(x, out). x is read-only here - the f32
// intermediate never lands (nothing downstream reads it; the GEMM consumes
// the f16 plane). Constant + form exactly pd_gelu_kernel's.
__global__ void pd_gelu_bias_f16_kernel(const float* __restrict__ x, const float* __restrict__ bias,
                                        __half* __restrict__ out, uint32_t rows, uint32_t n) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (uint64_t)rows * n) return;
    float v = x[idx] + bias[idx % n];
    out[idx] = __float2half(0.5f * v * (1.0f + tanhf(0.79788456080286535587989211986876f * v
                                                     * (1.0f + 0.044715f * v * v))));
}

PD_EXPORT
int pd_gelu_bias_f16(const void* x, const void* bias, void* out, uint32_t rows, uint32_t n,
                     void* stream) {
    if (rows == 0 || n == 0) return 0;
    uint64_t blocks = ((uint64_t)rows * n + 255ull) / 256ull;
    pd_gelu_bias_f16_kernel<<<(uint32_t)blocks, 256, 0, (cudaStream_t)stream>>>(
        (const float*)x, (const float*)bias, (__half*)out, rows, n);
    return pd_launch_status();
}

// erf twin (the projector FFN's activation - see pd_gelu_erf's header note on
// why the two GELUs must never be swapped). Same fusion, erf formula.
__global__ void pd_gelu_erf_bias_f16_kernel(const float* __restrict__ x,
                                            const float* __restrict__ bias,
                                            __half* __restrict__ out, uint32_t rows, uint32_t n) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (uint64_t)rows * n) return;
    float v = x[idx] + bias[idx % n];
    out[idx] = __float2half(0.5f * v * (1.0f + erff(v * 0.70710678118654752440084436210484f)));
}

PD_EXPORT
int pd_gelu_erf_bias_f16(const void* x, const void* bias, void* out, uint32_t rows, uint32_t n,
                         void* stream) {
    if (rows == 0 || n == 0) return 0;
    uint64_t blocks = ((uint64_t)rows * n + 255ull) / 256ull;
    pd_gelu_erf_bias_f16_kernel<<<(uint32_t)blocks, 256, 0, (cudaStream_t)stream>>>(
        (const float*)x, (const float*)bias, (__half*)out, rows, n);
    return pd_launch_status();
}

// residual + projection-bias in one pass: x[r][i] += src[r][i] + bias[i].
// Replaces pd_bias_add(src, b) -> pd_add(x, src); src stays UNBIASED in
// memory (callers that need the biased value - the attn0 tap - add b on the
// host, same IEEE add). t first so the association matches the unfused chain.
__global__ void pd_add_bias_res_kernel(float* __restrict__ x, const float* __restrict__ src,
                                       const float* __restrict__ bias, uint32_t rows, uint32_t n) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (uint64_t)rows * n) return;
    float t = src[idx] + bias[idx % n];
    x[idx] += t;
}

PD_EXPORT
int pd_add_bias_res(void* x, const void* src, const void* bias, uint32_t rows, uint32_t n,
                    void* stream) {
    if (rows == 0 || n == 0) return 0;
    uint64_t blocks = ((uint64_t)rows * n + 255ull) / 256ull;
    pd_add_bias_res_kernel<<<(uint32_t)blocks, 256, 0, (cudaStream_t)stream>>>(
        (float*)x, (const float*)src, (const float*)bias, rows, n);
    return pd_launch_status();
}

// pd_mrope_vision with the q/k projection bias folded into the load: the pair
// walk touches every head element exactly once, so x = rope(x + b) lands in
// one pass instead of bias_add's read+write followed by rope's read+write.
// Rotation math identical to pd_mrope_vision_kernel (same serial theta).
__global__ void pd_mrope_vision_bias_kernel(float* __restrict__ x,
                                            const float* __restrict__ bias,
                                            const unsigned int* __restrict__ positions,
                                            uint32_t n_tokens, uint32_t n_heads,
                                            uint32_t head_dim, float theta_scale) {
    uint32_t half = head_dim / 2;
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_tokens * n_heads * half) return;
    uint32_t p = gid % half;
    uint32_t idx = gid / half;                    // (t, h) flat index
    uint32_t t = idx / n_heads;
    float* head = x + (size_t)idx * head_dim;
    const float* brow = bias + (size_t)(idx % n_heads) * head_dim;
    uint32_t s = head_dim / 4;
    float theta = (p < s) ? (float)positions[t] : (float)positions[(size_t)n_tokens + t];
    uint32_t steps = (p < s) ? p : (p - s);
    for (uint32_t i = 0; i < steps; ++i) theta *= theta_scale;
    float sn = sinf(theta);
    float cs = cosf(theta);
    float a = head[p] + brow[p];
    float b = head[p + half] + brow[p + half];
    head[p] = a * cs - b * sn;
    head[p + half] = a * sn + b * cs;
}

PD_EXPORT
int pd_mrope_vision_bias(void* x, const void* bias, const void* positions, uint32_t n_tokens,
                         uint32_t n_heads, uint32_t head_dim, float theta_scale, void* stream) {
    uint32_t total = n_tokens * n_heads * (head_dim / 2);   // one thread per pair
    if (total == 0) return 0;
    uint32_t threads = 256;
    uint32_t blocks = (total + threads - 1) / threads;
    pd_mrope_vision_bias_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        (float*)x, (const float*)bias, (const unsigned int*)positions, n_tokens, n_heads,
        head_dim, theta_scale);
    return pd_launch_status();
}

// Non-causal (bidirectional) ViT attention: q/k/v [n, heads, hd] f32, out
// [n, heads*hd]. grid (heads, n); block hd threads; online softmax over all n
// keys - the same structure as pd_attn_decode_batch minus the causal bound and
// the KV cache (activations stay f32; the encoder runs once per image).
// Bidirectional (non-causal) tower attention, tiled: grid (n_heads,
// ceil(n/8)), 256 threads = 8 warps, each warp owns one query. K/V arrive in
// shared 32-key tiles staged by the whole block (each staged element serves 8
// queries); within a tile each LANE owns a key - the dot runs 72 FMAs straight
// from shared with no shuffles - and the tile folds into the query's running
// online softmax with one warp-max + one warp-sum per 32 keys. Rows are padded
// to 129 floats so lane-per-row reads land in distinct banks. Two earlier
// shapes measured on a 188-SM GB202 at n=2304: the original one-72-thread-
// block-per-(head,query) with 4 syncs per key ran 10.7 ms/layer (90% of tower
// encode) AND truncated the dot to 64 of 72 dims (n_warps = hd>>5 dropped the
// partial warp's red[2] - fixed here, tower output changes deliberately); a
// warp-per-query serial-key variant with per-key shuffle reduces still ran
// 4.6 ms/layer. This shape: sub-ms. Numerics: per-tile combine is the
// standard flash-style m/l fold, exact softmax algebra in tile order.
//
// This KERNEL is no LONGER the DEFAULT. It is the
// exact-f32 reference and the fallback for shapes/devices the mma kernel below
// cannot take (pre-sm_80, head_dim not a multiple of 8); PADDOCK_NO_VIS_MMA=1
// forces everything back onto it, which is how the parity test gates the exact
// class and how any future A/B gets its baseline.
//
// Where it STOOD (A6000 sm_86, n=5760/hd=72/16 heads): 78 ms per
// layer, 91% of the whole tower encode - the tower's GEMMs were 5%. Profiled:
// L1/TEX 83.0%, Compute (SM) 81.9%, DRAM 16.6%, L2 21.9%, occupancy 33%
// (shared-memory-limited, 2 blocks/SM). SHARED-MEMORY-THROUGHPUT bound at 83%
// of that port - 168 shared words per lane per 32-key tile - at its
// algorithmic ceiling, not mistuned. Two tunings were built and measured and
// both did nothing (8-sample medians): moving the head off the fastest grid
// dimension to make one head's K/V L2-resident, and right-sizing the tiles to
// hd+1 for 4 blocks/SM. Do not retry either. The answer was never a tuning
// - it was fewer
// shared bytes per unit of math, which is the mma kernel below.
#define PD_VA_KT 32u
#define PD_VA_QT 8u
#define PD_VA_MAXD 128u
// Generalized for granite-vision's Q-Former: separate query and KV
// counts (cross-attention reads 16 queries against 64 encoder rows) plus a
// batch dimension on grid.z (the 9 windows per tile attend independently).
// `pd_vision_attn` keeps its old contract by passing nq == nkv and grid.z == 1,
// so the tower path is byte-identical to before - the batch bases fold to 0.
__global__ void __launch_bounds__(256) pd_vision_attn_kernel(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, float* __restrict__ out, uint32_t nq,
    uint32_t nkv, uint32_t n_heads, uint32_t hd, float scale) {
    const uint32_t h = blockIdx.x;
    const uint32_t i0 = blockIdx.y * PD_VA_QT;
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t i = i0 + warp;             // this warp's query row
    const uint32_t nd = (hd + 31u) >> 5;      // dims per lane (3 at hd=72)
    // q/out are indexed by query rows, k/v by KV rows - the two strides differ
    // whenever this is a cross-attention.
    const size_t qbase = (size_t)blockIdx.z * nq * n_heads * hd;
    const size_t kbase = (size_t)blockIdx.z * nkv * n_heads * hd;

    __shared__ float sh_k[PD_VA_KT][PD_VA_MAXD + 1u];
    __shared__ float sh_v[PD_VA_KT][PD_VA_MAXD + 1u];
    __shared__ float sh_q[PD_VA_QT][PD_VA_MAXD];   // broadcast reads: no pad needed
    __shared__ float sh_w[PD_VA_QT][PD_VA_KT];

    const bool live = i < nq;
    // stage this warp's query (pre-scaled) into shared once
    if (live) {
        for (uint32_t d = lane; d < hd; d += 32u)
            sh_q[warp][d] = q[qbase + ((size_t)i * n_heads + h) * hd + d] * scale;
    }
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float m = -3.402823e38f, l = 0.0f;

    for (uint32_t j0 = 0; j0 < nkv; j0 += PD_VA_KT) {
        const uint32_t jl = min(PD_VA_KT, nkv - j0);
        // cooperative stage: 256 threads copy jl rows of K and V
        for (uint32_t idx = threadIdx.x; idx < jl * hd; idx += 256u) {
            const uint32_t jj = idx / hd, dd = idx % hd;
            sh_k[jj][dd] = k[kbase + ((size_t)(j0 + jj) * n_heads + h) * hd + dd];
            sh_v[jj][dd] = v[kbase + ((size_t)(j0 + jj) * n_heads + h) * hd + dd];
        }
        __syncthreads();
        if (live) {
            // lane-per-key dot: 72 serial FMAs, zero shuffles. (A float4
            // variant was tried and measured slower, 273 -> 287 ms encode:
            // 32 lanes x float4 from shared is 4 transactions per
            // instruction regardless - vectorizing saves instructions, not
            // shared bandwidth, and the required even row stride broke the
            // conflict-free +1 padding. The remaining lever here is f16 K/V
            // tiles - halves shared traffic AND footprint - but that is a
            // numeric-class change needing a dual-kernel parity structure:
            // keep this f32 kernel exact-gated at 1e-4, add the f16 one
            // gated at its own class tolerance. Not done in Q2.)
            float dot = -3.402823e38f;
            if (lane < jl) {
                float p = 0.0f;
                for (uint32_t d = 0; d < hd; ++d) p += sh_q[warp][d] * sh_k[lane][d];
                dot = p;
            }
            // tile max + weights + tile weight-sum (one warp reduction each)
            float tmax = dot;
#pragma unroll
            for (uint32_t off = 16; off > 0; off >>= 1)
                tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, off));
            const float m_new = fmaxf(m, tmax);
            const float corr = __expf(m - m_new);
            const float w = (lane < jl) ? __expf(dot - m_new) : 0.0f;
            sh_w[warp][lane] = w;
            float wsum = w;
#pragma unroll
            for (uint32_t off = 16; off > 0; off >>= 1)
                wsum += __shfl_xor_sync(0xffffffffu, wsum, off);
            __syncwarp();
            // fold the tile: acc = acc*corr + sum_j w_j * V[j]
#pragma unroll
            for (uint32_t c = 0; c < 4; ++c) {
                uint32_t d = lane + c * 32u;
                if (c < nd && d < hd) {
                    float a = acc[c] * corr;
                    for (uint32_t j = 0; j < jl; ++j)
                        a = fmaf(sh_w[warp][j], sh_v[j][d], a);
                    acc[c] = a;
                }
            }
            l = l * corr + wsum;
            m = m_new;
        }
        __syncthreads();
    }
    if (live) {
#pragma unroll
        for (uint32_t c = 0; c < 4; ++c) {
            uint32_t d = lane + c * 32u;
            if (c < nd && d < hd)
                out[qbase + (size_t)i * n_heads * hd + (size_t)h * hd + d] = acc[c] / l;
        }
    }
}

// ~38 KB static shared per block: opt into the max carveout so two blocks can
// sit per SM (default 48 KB window fits only one). Shared by both entry points.
static void pd_vision_attn_carveout(void) {
    static bool va_attr = false;
    if (!va_attr) {
        pd_prefer_max_shared(pd_vision_attn_kernel);
        va_attr = true;
    }
}

// ---------------------------------------------------------------- vision mma
// Tensor-core form of exactly the attention above.
//
// The scalar kernel sits at 83% of the shared-memory port and 5% of fp32 peak:
// it spends its whole budget MOVING operands - one shared word per lane per
// FMA, and no arrangement of the tiles changes that ratio (two tunings were
// built and measured and did nothing; see the header comment). mma inverts it.
// One `m16n8k16` instruction consumes a 16x16 A fragment and a 16x8 B fragment
// that are already in registers, and one `ldmatrix.x4` fills four 8x8 B tiles
// with a single shared instruction where the scalar walk issued 256 scalar
// loads. Same math, ~30x fewer shared transactions per unit of it.
//
// The fragment algebra is lifted wholesale from the prefill v3c kernel
// (attn/prefill.cuh, serving this geometry since): non-trans
// ldmatrix for the score B-frags because K stored [key][dim] is the
// B[k=dim][n=key] orientation, trans ldmatrix for the PV B-frags because V
// needs [dim][key]. This is that shape minus paging, sinks, positions and the
// causal bound, plus the grid.z batch and the nq != nkv cross-attention the
// Q-Former needs.
//
// NUMERIC CLASS: q/k/v round to f16 into the fragments; the accumulator, the
// online softmax and the output stay f32 (mma's .f32.f16.f16.f32 form). That
// is a real class change from the exact-f32 kernel above - and it is the class
// the rest of the tower already runs in, because q/k/v come out of
// `matvec_batch_f16`, whose weights and activations are both f16. The f32
// kernel stays reachable (PADDOCK_NO_VIS_MMA=1) and stays exact-gated at 1e-4
// in tests/gpu_vision_attn_parity.rs; this path is gated at its own class
// tolerance in the same test.
//
// DP is the head dim rounded up to a multiple of 16 (the mma K step). Pad dims
// are zeroed in the tiles and in the q fragments, so they contribute exactly
// nothing to the score, and the output columns they produce are never stored.
// Keys past nkv are zeroed too AND masked to -inf - zeroing alone would leave
// them a weight of exp(0-m), the phantom-softmax-mass bug.
//
// The head goes on grid.y, not grid.x, so consecutive blocks share a head and
// its K/V (3.3 MB at n=5760) stays L2-resident instead of 16 heads' worth
// (53 MB) being in flight. The same reordering measured as nothing on the
// scalar kernel above - because that kernel was shared-bound and its DRAM
// throughput was 16.6%, so no amount of L2 help could show. Here it matters:
// mma raised the arithmetic per byte enough that global traffic is on the
// table again. A falsified lever is falsified for the kernel it was measured
// on, not forever.
//
// Where it STANDS (A6000 sm_86, same shape, 8-sample medians on
// the same harness that killed the two tunings above):
//   scalar f32   attn 2010 ms   tower 2208 ms
//   mma          attn   76 ms   tower  271 ms      26x on the kernel, 8x tower
// Profiled: L1/TEX 48.8% (was 83.0), Compute (SM) 42.5% (was 81.9), DRAM 7.7%,
// L2 35.8%, achieved occupancy 31.5%. Nothing is near a wall any more - the
// profiler calls it latency, Block Limit Registers is 2 (shared would allow 4) and only
// 0.61 warps per scheduler are eligible, one issue per 2.9 cycles. So the next
// lever here is registers and stage/compute overlap (cp.async, the v3c
// recipe), not another tile shape. It is the tower's biggest cell again
// now that fixed the rope thread mapping: 208 ms reads
// ln1 10 | qkv 29 | mrope 5 | attn 77 | out 15 | ffn 70 (was 271 with
// mrope 67). Attention and the FFN GEMMs are what is left.
#define PD_VM_KT 64u        // keys staged per tile
#define PD_VM_QW 8u         // warps per block, each owning 16 query rows
#define PD_VM_QT (PD_VM_QW * 16u)

// Shared bytes for one padded head dim - the launcher and the kernel must
// agree, so both go through this.
#define PD_VM_SMEM(DP) (2u * PD_VM_KT * ((DP) + 8u) * 2u)

// ---- the element interface: T = float (every tower so far) or __half ----
// The kernel has always rounded q/k/v to f16 on their way into the fragments
// and the tiles; reading them as f32 only to convert at the stage is the f32
// interface's cost, not the math's. A producer that already writes halves
// (dinov3's qkv split, which folds 1/sqrt(hd) into q before its round - the
// same single round the f32 path does here) gets T = __half: the q fragment
// is a word load, the stage is a 16 B copy, and the output lands as halves -
// so the f32 attention plane and the convert pass behind it both disappear.
// Bit-for-bit the f32 kernel's result rounded by __float2half
// (bench/vis_attn_h_bench.cu: 0 of 16.9 M differ), 18.6% faster on the
// dinov3 shape, and the three overload pairs below are the whole difference.
// A double-buffered cp.async stage was built on the half interface and is
// worth nothing at this key tile (KT 64: 980 vs 970 us, bit-equal; at KT 32 it
// only recovers what the smaller tile loses) - the kernel is issue/latency-
// bound (ncu: tensor pipe 56%, DRAM 19%), not stage-bound.
//
// f32: the original predicated form, verbatim. Half: clamped unconditional
// loads zeroed afterwards - a predicated load costs per-lane preserve-MOVs.
// `qrow` is the head's row, or a valid row when rowok is false (half only).
__device__ __forceinline__ uint32_t pd_vm_qpair(const float* qrow, bool rowok, uint32_t c,
                                                uint32_t hd, float scale) {
    const float f0 = (rowok && c < hd) ? qrow[c] * scale : 0.f;
    const float f1 = (rowok && c + 1u < hd) ? qrow[c + 1u] * scale : 0.f;
    const __half2 p = __floats2half2_rn(f0, f1);
    return *reinterpret_cast<const uint32_t*>(&p);
}
__device__ __forceinline__ uint32_t pd_vm_qpair(const __half* qrow, bool rowok, uint32_t c,
                                                uint32_t hd, float) {
    const uint32_t raw = *reinterpret_cast<const uint32_t*>(qrow + (c < hd ? c : 0u));
    return (rowok && c < hd) ? raw : 0u;
}
// 8 dims of one key into the f16 tile
__device__ __forceinline__ void pd_vm_stage8(__half* dst, const float* src) {
    const float4 a = *reinterpret_cast<const float4*>(src);
    const float4 b = *reinterpret_cast<const float4*>(src + 4);
    *reinterpret_cast<__half2*>(dst + 0) = __floats2half2_rn(a.x, a.y);
    *reinterpret_cast<__half2*>(dst + 2) = __floats2half2_rn(a.z, a.w);
    *reinterpret_cast<__half2*>(dst + 4) = __floats2half2_rn(b.x, b.y);
    *reinterpret_cast<__half2*>(dst + 6) = __floats2half2_rn(b.z, b.w);
}
__device__ __forceinline__ void pd_vm_stage8(__half* dst, const __half* src) {
    *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
}
// one output pair. hd is even on the half interface, so `second` is always
// true there and the pair is one 4 B store.
__device__ __forceinline__ void pd_vm_store2(float* op, float a, float b, bool second) {
    op[0] = a;
    if (second) op[1] = b;
}
__device__ __forceinline__ void pd_vm_store2(__half* op, float a, float b, bool) {
    *reinterpret_cast<__half2*>(op) = __floats2half2_rn(a, b);
}

// QW: warps a block, 16 query rows each - block geometry only, the per-row
// math is untouched, so every QW lands the same bits. PD_VM_QW (8) is what
// every f32 tower was measured at; the half path on cc 12.0 takes 4 (see
// pd_vision_attn_h): on the dinov3 shape (1029 rows, 8 chips, hd 64) the
// 64-row block ran 8.1% faster than the 128-row one - the same 16 warps an
// SM from twice the blocks, and a 1029-row sequence pads 5.7% of its rows
// instead of 12% (bench/vis_attn_h2_bench.cu, which also tried two m-tiles
// a warp: bit-equal, 216 registers, at best the same 6-8% from the block
// shape alone - the K/V fragment reuse itself is worth nothing here).
template <uint32_t DP, typename T = float, uint32_t QW = PD_VM_QW>
__global__ void __launch_bounds__(32u * QW) pd_vision_attn_mma_kernel(
    const T* __restrict__ q, const T* __restrict__ k,
    const T* __restrict__ v, T* __restrict__ out, uint32_t nq,
    uint32_t nkv, uint32_t n_heads, uint32_t hd, float scale) {
#if PD_FA_OK
    constexpr uint32_t KT = PD_VM_KT, DPD = DP + 8u, NT = 32u * QW;
    const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
    const uint32_t g8 = lane >> 2, t4 = lane & 3u, lg = lane >> 3;
    const uint32_t h = blockIdx.y;
    const uint32_t row0 = blockIdx.x * (QW * 16u) + warp * 16u;
    // q/out are indexed by query rows, k/v by KV rows - the two strides differ
    // whenever this is a cross-attention (the Q-Former's 16x64).
    const size_t qbase = (size_t)blockIdx.z * nq * n_heads * hd;
    const size_t kbase = (size_t)blockIdx.z * nkv * n_heads * hd;

    // +8 halves of row pad: the 8 rows an ldmatrix touches then land in
    // distinct bank groups (DPD*2 is never a multiple of 128 B).
    extern __shared__ unsigned char vmsh[];
    __half* sh_k = reinterpret_cast<__half*>(vmsh);
    __half* sh_v = sh_k + (size_t)KT * DPD;

    // A-fragment rows this lane owns (mma layout: row = lane/4, and +8)
    const uint32_t jr[2] = {g8, g8 + 8u};
    uint32_t qa[DP / 16u][4];
#pragma unroll
    for (uint32_t d0 = 0; d0 < DP / 16u; ++d0) {
#pragma unroll
        for (uint32_t e = 0; e < 2u; ++e) {
            const uint32_t b = row0 + jr[e];
            const bool rowok = b < nq;
            // past the last query the pointer stays on a valid row: the f32
            // overload never dereferences it, the half one zeroes what it read
            const T* qp = q + qbase + ((size_t)(rowok ? b : nq - 1u) * n_heads + h) * hd;
            const uint32_t c0 = d0 * 16u + 2u * t4, c1 = c0 + 8u;
            // scale folds into q before the round, exactly as v3c does it
            qa[d0][e] = pd_vm_qpair(qp, rowok, c0, hd, scale);
            qa[d0][e + 2u] = pd_vm_qpair(qp, rowok, c1, hd, scale);
        }
    }

    float m_st[2] = {-1e30f, -1e30f}, l_st[2] = {0.f, 0.f};
    float o_acc[DP / 8u][4];
#pragma unroll
    for (uint32_t nt = 0; nt < DP / 8u; ++nt)
#pragma unroll
        for (uint32_t e = 0; e < 4u; ++e) o_acc[nt][e] = 0.f;

    for (uint32_t t0 = 0; t0 < nkv; t0 += KT) {
        // co-stage K then V as f16, 8 dims per thread-step: two 16 B global
        // loads collapse into one 16 B shared store. Rows past nkv and dims
        // past hd store zeros so the fragments are always well-defined.
        constexpr uint32_t SPAN = KT * (DP / 8u);
        for (uint32_t u = tid; u < 2u * SPAN; u += NT) {
            const bool isv = u >= SPAN;
            const uint32_t ur = isv ? u - SPAN : u;
            const uint32_t kk = ur / (DP / 8u), d8 = (ur % (DP / 8u)) * 8u;
            const uint32_t ks = t0 + kk;
            __half* dst = (isv ? sh_v : sh_k) + (size_t)kk * DPD + d8;
            if (ks < nkv && d8 < hd) {
                const T* src = (isv ? v : k) + kbase
                               + ((size_t)ks * n_heads + h) * hd + d8;
                pd_vm_stage8(dst, src);
            } else {
                *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        __syncthreads();

        float s_acc[KT / 8u][4];
#pragma unroll
        for (uint32_t nt = 0; nt < KT / 8u; ++nt)
#pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) s_acc[nt][e] = 0.f;
        // scores: A = q rows x dims, B = K rows read as [k=dim][n=key], which
        // is the non-trans ldmatrix orientation of the [key][dim] tile
#pragma unroll
        for (uint32_t d0 = 0; d0 < DP / 16u; ++d0) {
#pragma unroll
            for (uint32_t np = 0; np < KT / 16u; ++np) {
                const __half* kp = sh_k
                    + (size_t)(np * 16u + (lg >> 1) * 8u + (lane & 7u)) * DPD
                    + d0 * 16u + (lg & 1u) * 8u;
                uint32_t kb4[4];
                const uint32_t ka = (uint32_t)__cvta_generic_to_shared(kp);
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                    : "=r"(kb4[0]), "=r"(kb4[1]), "=r"(kb4[2]), "=r"(kb4[3]) : "r"(ka));
                pd_fa_mma16(s_acc[np * 2u], qa[d0][0], qa[d0][1], qa[d0][2],
                            qa[d0][3], kb4[0], kb4[1]);
                pd_fa_mma16(s_acc[np * 2u + 1u], qa[d0][0], qa[d0][1], qa[d0][2],
                            qa[d0][3], kb4[2], kb4[3]);
            }
        }
        // online softmax: the C-fragment puts one query row across the four
        // lanes sharing a lane/4 group, so max/sum reduce over t4 only
        float mn[2] = {m_st[0], m_st[1]};
#pragma unroll
        for (uint32_t nt = 0; nt < KT / 8u; ++nt) {
            const uint32_t kb = t0 + nt * 8u + 2u * t4;
#pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) {
                if (kb + (e & 1u) >= nkv) s_acc[nt][e] = -1e30f;
                mn[e >> 1] = fmaxf(mn[e >> 1], s_acc[nt][e]);
            }
        }
#pragma unroll
        for (uint32_t o = 1; o <= 2u; o <<= 1) {
            mn[0] = fmaxf(mn[0], __shfl_xor_sync(0xffffffffu, mn[0], o));
            mn[1] = fmaxf(mn[1], __shfl_xor_sync(0xffffffffu, mn[1], o));
        }
        float ws[2] = {0.f, 0.f};
#pragma unroll
        for (uint32_t nt = 0; nt < KT / 8u; ++nt) {
#pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) {
                const float d = s_acc[nt][e] - mn[e >> 1];
                const float w = d >= -20.f ? __expf(d) : 0.f;
                s_acc[nt][e] = w;
                ws[e >> 1] += w;
            }
        }
#pragma unroll
        for (uint32_t o = 1; o <= 2u; o <<= 1) {
            ws[0] += __shfl_xor_sync(0xffffffffu, ws[0], o);
            ws[1] += __shfl_xor_sync(0xffffffffu, ws[1], o);
        }
        float corr[2];
#pragma unroll
        for (uint32_t r = 0; r < 2u; ++r) {
            const float dc = m_st[r] - mn[r];
            corr[r] = dc >= -20.f ? __expf(dc) : 0.f;
            l_st[r] = l_st[r] * corr[r] + ws[r];
            m_st[r] = mn[r];
        }
#pragma unroll
        for (uint32_t nt = 0; nt < DP / 8u; ++nt) {
            o_acc[nt][0] *= corr[0];
            o_acc[nt][1] *= corr[0];
            o_acc[nt][2] *= corr[1];
            o_acc[nt][3] *= corr[1];
        }
        // PV: A = the tile's weights (queries x keys) straight out of the
        // score C-fragment, B = V read as [k=key][n=dim] -> ldmatrix.trans
#pragma unroll
        for (uint32_t kf = 0; kf < KT / 16u; ++kf) {
            const uint32_t c0 = 2u * kf, c1 = c0 + 1u;
            const __half2 a0 = __floats2half2_rn(s_acc[c0][0], s_acc[c0][1]);
            const __half2 a1 = __floats2half2_rn(s_acc[c0][2], s_acc[c0][3]);
            const __half2 a2 = __floats2half2_rn(s_acc[c1][0], s_acc[c1][1]);
            const __half2 a3 = __floats2half2_rn(s_acc[c1][2], s_acc[c1][3]);
            const uint32_t pa0 = *reinterpret_cast<const uint32_t*>(&a0);
            const uint32_t pa1 = *reinterpret_cast<const uint32_t*>(&a1);
            const uint32_t pa2 = *reinterpret_cast<const uint32_t*>(&a2);
            const uint32_t pa3 = *reinterpret_cast<const uint32_t*>(&a3);
            const uint32_t vr = kf * 16u + (lg & 1u) * 8u + (lane & 7u);
            const __half* vp = sh_v + (size_t)vr * DPD + (lg >> 1) * 8u;
#pragma unroll
            for (uint32_t nt = 0; nt < DP / 8u; nt += 2u) {
                uint32_t vb4[4];
                const uint32_t va = (uint32_t)__cvta_generic_to_shared(vp + nt * 8u);
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                    : "=r"(vb4[0]), "=r"(vb4[1]), "=r"(vb4[2]), "=r"(vb4[3]) : "r"(va));
                pd_fa_mma16(o_acc[nt], pa0, pa1, pa2, pa3, vb4[0], vb4[1]);
                pd_fa_mma16(o_acc[nt + 1u], pa0, pa1, pa2, pa3, vb4[2], vb4[3]);
            }
        }
        __syncthreads();   // tiles read before the next stage overwrites them
    }

    const float nrm[2] = {l_st[0] > 0.f ? 1.f / l_st[0] : 0.f,
                          l_st[1] > 0.f ? 1.f / l_st[1] : 0.f};
#pragma unroll
    for (uint32_t nt = 0; nt < DP / 8u; ++nt) {
        const uint32_t dcol = nt * 8u + 2u * t4;
        if (dcol >= hd) continue;
#pragma unroll
        for (uint32_t r = 0; r < 2u; ++r) {
            const uint32_t b = row0 + jr[r];
            if (b >= nq) continue;
            T* op = out + qbase + (size_t)b * n_heads * hd + (size_t)h * hd + dcol;
            pd_vm_store2(op, o_acc[nt][2u * r] * nrm[r], o_acc[nt][2u * r + 1u] * nrm[r],
                         dcol + 1u < hd);
        }
    }
#else
    (void)q; (void)k; (void)v; (void)out; (void)nq; (void)nkv; (void)n_heads;
    (void)hd; (void)scale;
#endif
}

// pd_env() reads the CRT's copy of the environment, taken at process start -
// and this pack carries its own CRT instance, so a variable the host sets after
// load (Rust's std::env::set_var, which is just SetEnvironmentVariableW) is
// invisible to it. Measured: the parity test flipped the switch
// between legs and got byte-identical numbers both times, because nothing was
// flipping. Reading the Win32 block directly sees both a shell-set variable and
// an in-process one, which is what lets one test process gate both numeric
// classes. The pack's older getenv switches predate this and are shell-only.
#if defined(_WIN32)
extern "C" __declspec(dllimport) unsigned long __stdcall GetEnvironmentVariableA(
    const char*, char*, unsigned long);
#endif
static bool pd_vm_env_on(const char* name) {
#if defined(_WIN32)
    char buf[8];
    const unsigned long n = GetEnvironmentVariableA(name, buf, sizeof(buf));
    return n > 0 && n < sizeof(buf) && buf[0] == '1';
#else
    const char* e = pd_env(name);
    return e && e[0] == '1';
#endif
}

// One dispatcher behind both exports. Picks the mma kernel when the shape and
// the device allow it; anything else keeps the exact f32 kernel, which is a
// correct answer at a worse speed rather than no answer.
//   - head_dim must be a multiple of 8 (the tile staging moves 8 dims at a
//     time) and <= 128. Every tower we serve is 64, 72 or 128.
//   - sm_80+ for mma.m16n8k16 / ldmatrix.
//   - PADDOCK_NO_VIS_MMA=1 forces the f32 kernel. The read is deliberately not
//     cached: this launches once per layer per image (tens of times), an
//     environment probe is free against that, and a live read lets one test
//     process gate both numeric classes.
static int pd_vision_attn_dispatch(const float* q, const float* k, const float* v,
                                   float* out, uint32_t nq, uint32_t nkv,
                                   uint32_t n_heads, uint32_t hd, uint32_t n_batch,
                                   float scale, cudaStream_t stream) {
    static int mma_arch = -1;
    if (mma_arch < 0) {
        int dev = 0, cc = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&cc, cudaDevAttrComputeCapabilityMajor, dev);
        mma_arch = cc >= 8 ? 1 : 0;
    }
    const bool use_mma = mma_arch && !pd_vm_env_on("PADDOCK_NO_VIS_MMA")
                         && (hd % 8u) == 0u && hd >= 16u && hd <= PD_VA_MAXD;
    if (use_mma) {
        const uint32_t dp = (hd + 15u) & ~15u;
        dim3 grid((nq + PD_VM_QT - 1u) / PD_VM_QT, n_heads, n_batch);
        const uint32_t nth = 32u * PD_VM_QW;
        static bool vm_attr = false;
        if (!vm_attr) {
            pd_prefer_max_shared(pd_vision_attn_mma_kernel<16u>);
            pd_prefer_max_shared(pd_vision_attn_mma_kernel<32u>);
            pd_prefer_max_shared(pd_vision_attn_mma_kernel<48u>);
            pd_prefer_max_shared(pd_vision_attn_mma_kernel<64u>);
            pd_prefer_max_shared(pd_vision_attn_mma_kernel<80u>);
            pd_prefer_max_shared(pd_vision_attn_mma_kernel<96u>);
            pd_prefer_max_shared(pd_vision_attn_mma_kernel<112u>);
            pd_prefer_max_shared(pd_vision_attn_mma_kernel<128u>);
            vm_attr = true;
        }
#define PD_VM_LAUNCH(DP)                                                        \
    case DP:                                                                    \
        pd_vision_attn_mma_kernel<DP>                                           \
            <<<grid, nth, PD_VM_SMEM(DP), stream>>>(q, k, v, out, nq, nkv,       \
                                                    n_heads, hd, scale);        \
        break;
        switch (dp) {
            PD_VM_LAUNCH(16u)
            PD_VM_LAUNCH(32u)
            PD_VM_LAUNCH(48u)
            PD_VM_LAUNCH(64u)
            PD_VM_LAUNCH(80u)
            PD_VM_LAUNCH(96u)
            PD_VM_LAUNCH(112u)
            PD_VM_LAUNCH(128u)
            default: return cudaErrorInvalidValue;
        }
#undef PD_VM_LAUNCH
        return pd_launch_status();
    }
    pd_vision_attn_carveout();
    dim3 grid(n_heads, (nq + PD_VA_QT - 1u) / PD_VA_QT, n_batch);
    pd_vision_attn_kernel<<<grid, 256, 0, stream>>>(q, k, v, out, nq, nkv, n_heads,
                                                    hd, scale);
    return pd_launch_status();
}

PD_EXPORT
int pd_vision_attn(const void* q, const void* k, const void* v, void* out, uint32_t n,
                   uint32_t n_heads, uint32_t head_dim, float scale, void* stream) {
    if (n == 0 || n_heads == 0) return 0;
    if (head_dim > PD_VA_MAXD) return cudaErrorInvalidValue;
    return pd_vision_attn_dispatch((const float*)q, (const float*)k, (const float*)v,
                                   (float*)out, n, n, n_heads, head_dim, 1u, scale,
                                   (cudaStream_t)stream);
}

// Cross/self attention over a BATCH of independent groups - granite-vision's
// windowed Q-Former (311). Each of `n_batch` windows holds `nq`
// query rows and `nkv` KV rows laid out [batch][row][head][dim]; self-attention
// passes nq == nkv with q == k == v, cross-attention passes the 16x64 shape.
// One launch covers every window, which is the whole point: the per-window
// shapes are far too small to fill the GPU on their own.
PD_EXPORT
int pd_vision_attn_x(const void* q, const void* k, const void* v, void* out, uint32_t nq,
                     uint32_t nkv, uint32_t n_heads, uint32_t head_dim, uint32_t n_batch,
                     float scale, void* stream) {
    if (nq == 0 || nkv == 0 || n_heads == 0 || n_batch == 0) return 0;
    if (head_dim > PD_VA_MAXD) return cudaErrorInvalidValue;
    return pd_vision_attn_dispatch((const float*)q, (const float*)k, (const float*)v,
                                   (float*)out, nq, nkv, n_heads, head_dim, n_batch,
                                   scale, (cudaStream_t)stream);
}

// 620: the same attention on the half interface - q (PRE-SCALED by the caller,
// see pd_vm_qpair), k, v and out are all [batch][row][head][dim] halves. There
// is no f32 fallback to fall to: head_dim must be a multiple of 8 in
// [16, 128] and the device sm_80+, or this refuses and the caller keeps the
// f32 entry. 16 B-aligned planes (any pooled allocation is).
PD_EXPORT
int pd_vision_attn_h(const void* q, const void* k, const void* v, void* out, uint32_t nq,
                     uint32_t nkv, uint32_t n_heads, uint32_t head_dim, uint32_t n_batch,
                     void* stream) {
    if (nq == 0 || nkv == 0 || n_heads == 0 || n_batch == 0) return 0;
    if ((head_dim % 8u) != 0u || head_dim < 16u || head_dim > PD_VA_MAXD)
        return cudaErrorInvalidValue;
    int dev = 0, cc = 0, ccm = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&cc, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&ccm, cudaDevAttrComputeCapabilityMinor, dev);
    if (cc < 8) return cudaErrorInvalidValue;
    const uint32_t dp = (head_dim + 15u) & ~15u;
    // 64-row blocks on cc 12.0 (measured, see the kernel note) and cc 12.1
    // (GB10, Gemma 4's picture tower, 2394 rows x 12 heads x hd 64, one
    // sequence 307 -> 289 us, four 1273 -> 1099; bench/vis_attn_h2_bench.cu,
    // 0 mismatches); 128 elsewhere
    const bool qw4 = (cc == 12 && (ccm == 0 || ccm == 1));
    const uint32_t qw = qw4 ? 4u : PD_VM_QW;
    dim3 grid((nq + qw * 16u - 1u) / (qw * 16u), n_heads, n_batch);
    const uint32_t nth = 32u * qw;
    static bool vh_attr = false;
    if (!vh_attr) {
        pd_prefer_max_shared(pd_vision_attn_mma_kernel<16u, __half>);
        pd_prefer_max_shared(pd_vision_attn_mma_kernel<32u, __half>);
        pd_prefer_max_shared(pd_vision_attn_mma_kernel<48u, __half>);
        pd_prefer_max_shared(pd_vision_attn_mma_kernel<64u, __half>);
        pd_prefer_max_shared(pd_vision_attn_mma_kernel<80u, __half>);
        pd_prefer_max_shared(pd_vision_attn_mma_kernel<96u, __half>);
        pd_prefer_max_shared(pd_vision_attn_mma_kernel<112u, __half>);
        pd_prefer_max_shared(pd_vision_attn_mma_kernel<128u, __half>);
        pd_prefer_max_shared(pd_vision_attn_mma_kernel<16u, __half, 4u>);
        pd_prefer_max_shared(pd_vision_attn_mma_kernel<32u, __half, 4u>);
        pd_prefer_max_shared(pd_vision_attn_mma_kernel<48u, __half, 4u>);
        pd_prefer_max_shared(pd_vision_attn_mma_kernel<64u, __half, 4u>);
        pd_prefer_max_shared(pd_vision_attn_mma_kernel<80u, __half, 4u>);
        pd_prefer_max_shared(pd_vision_attn_mma_kernel<96u, __half, 4u>);
        pd_prefer_max_shared(pd_vision_attn_mma_kernel<112u, __half, 4u>);
        pd_prefer_max_shared(pd_vision_attn_mma_kernel<128u, __half, 4u>);
        vh_attr = true;
    }
#define PD_VH_LAUNCH(DP)                                                        \
    case DP:                                                                    \
        if (qw4)                                                                \
            pd_vision_attn_mma_kernel<DP, __half, 4u>                           \
                <<<grid, nth, PD_VM_SMEM(DP), (cudaStream_t)stream>>>(          \
                    (const __half*)q, (const __half*)k, (const __half*)v,       \
                    (__half*)out, nq, nkv, n_heads, head_dim, 1.0f);            \
        else                                                                    \
            pd_vision_attn_mma_kernel<DP, __half>                               \
                <<<grid, nth, PD_VM_SMEM(DP), (cudaStream_t)stream>>>(          \
                    (const __half*)q, (const __half*)k, (const __half*)v,       \
                    (__half*)out, nq, nkv, n_heads, head_dim, 1.0f);            \
        break;
    switch (dp) {
        PD_VH_LAUNCH(16u)
        PD_VH_LAUNCH(32u)
        PD_VH_LAUNCH(48u)
        PD_VH_LAUNCH(64u)
        PD_VH_LAUNCH(80u)
        PD_VH_LAUNCH(96u)
        PD_VH_LAUNCH(112u)
        PD_VH_LAUNCH(128u)
        default: return cudaErrorInvalidValue;
    }
#undef PD_VH_LAUNCH
    return pd_launch_status();
}

// Row gather with an averaging fan-in (312) - granite-vision's
// windowing AND both of its downsamplers in one op:
//   k == 1  plain get_rows: window/unwindow permutations, and the "pick one
//           cell per 2x2 block" spatial downsampler (offsets TL/TR/BL/BR).
//   k == 4  the area-interpolate downsampler = 2x2 average pool, expressed as
//           a 4-way gather whose index table names the pooled cells.
// Unifying them means the two downsampler branches differ only in the host
// index table, so there is a single device path to get right. Both fan-ins
// scale by an exact power of two (1.0, 0.25), so k == 1 is bit-exact with a
// plain gather and k == 4 matches ggml's pool_2d sum-then-divide.
__global__ void pd_gather_rows_avg_kernel(const float* __restrict__ src,
                                          const int32_t* __restrict__ idx,
                                          float* __restrict__ out, uint32_t k,
                                          uint32_t width) {
    const uint32_t r = blockIdx.x;
    const float inv = 1.0f / (float)k;
    for (uint32_t d = threadIdx.x; d < width; d += blockDim.x) {
        float a = 0.0f;
        for (uint32_t j = 0; j < k; ++j)
            a += src[(size_t)idx[(size_t)r * k + j] * width + d];
        out[(size_t)r * width + d] = a * inv;
    }
}

PD_EXPORT
int pd_gather_rows_avg(const void* src, const void* idx, void* out, uint32_t rows,
                       uint32_t k, uint32_t width, void* stream) {
    if (rows == 0 || k == 0 || width == 0) return 0;
    const uint32_t nth = width < 256u ? ((width + 31u) & ~31u) : 256u;
    pd_gather_rows_avg_kernel<<<rows, nth, 0, (cudaStream_t)stream>>>(
        (const float*)src, (const int32_t*)idx, (float*)out, k, width);
    return pd_launch_status();
}

// Pixel-shuffle merge (muse-glimmer): gather k source rows per
// output row and interleave them CHANNEL-OUTER -
//     out[o][c*k + s] = src[idx[o*k + s]][c]
// which is llama.cpp's get_rows -> reshape [c, s, o] -> permute(1,0,2,3) ->
// cont -> reshape [c*k, o] chain collapsed into one pass.
//
// Not `gather_rows_avg` with k=4: that one SUMS the k rows (a pool). This one
// keeps all k and concatenates them, and the interleave is what makes it
// different from a plain 4-row concat as well - qwen3vl's merger is
// out[o][s*width + c] (spatial-outer), muse is the transpose of that. Getting
// the two confused is silent: same shape, same norm, permuted features.
//
// One block per output row; the loop walks OUTPUT positions so the stores are
// fully coalesced and a warp's loads land as k runs of 32/k consecutive floats.
__global__ void pd_pixel_shuffle_rows_kernel(const float* __restrict__ src,
                                             const int32_t* __restrict__ idx,
                                             float* __restrict__ out, uint32_t k,
                                             uint32_t width) {
    const uint32_t o = blockIdx.x;
    const uint32_t rw = k * width;
    const int32_t* ix = idx + (size_t)o * k;
    float* orow = out + (size_t)o * rw;
    for (uint32_t p = threadIdx.x; p < rw; p += blockDim.x) {
        const uint32_t c = p / k, s = p - c * k;
        orow[p] = src[(size_t)ix[s] * width + c];
    }
}

PD_EXPORT
int pd_pixel_shuffle_rows(const void* src, const void* idx, void* out, uint32_t rows,
                          uint32_t k, uint32_t width, void* stream) {
    if (rows == 0 || k == 0 || width == 0) return 0;
    const uint32_t rw = k * width;
    const uint32_t nth = rw < 256u ? ((rw + 31u) & ~31u) : 256u;
    pd_pixel_shuffle_rows_kernel<<<rows, nth, 0, (cudaStream_t)stream>>>(
        (const float*)src, (const int32_t*)idx, (float*)out, k, width);
    return pd_launch_status();
}

// Exact-erf GELU, in place (313). Not interchangeable with `gelu`,
// which is the tanh approximation: granite-vision's tower uses tanh (its GGUF
// sets clip.use_gelu) while the Q-Former FFN uses this one (llama.cpp calls
// ggml_gelu_erf there explicitly, matching PyTorch's F.gelu default
// approximate="none"). Same formula and constant as ggml's vec.h so the two
// engines agree term for term.
__global__ void pd_gelu_erf_kernel(float* __restrict__ x, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float xi = x[i];
        x[i] = 0.5f * xi * (1.0f + erff(xi * 0.70710678118654752440084436210484f));
    }
}

PD_EXPORT
int pd_gelu_erf(void* x, uint64_t n, void* stream) {
    if (n == 0) return 0;
    const uint64_t blocks = (n + 255ull) / 256ull;
    pd_gelu_erf_kernel<<<(uint32_t)blocks, 256, 0, (cudaStream_t)stream>>>((float*)x, n);
    return pd_launch_status();
}

// Broadcast row add (314): `x[r][d] += src[r % src_rows][d]`. The
// Q-Former adds one learned query table [16, C] across every window and one
// learned encoder-position table [64, C] likewise, so the cycling source row is
// the natural shape. `bias_add` is the src_rows == 1 case of this; it stays a
// separate export because it is on hot text paths and specializes better.
__global__ void pd_add_rows_bcast_kernel(float* __restrict__ x,
                                         const float* __restrict__ src,
                                         uint32_t src_rows, uint32_t width) {
    const uint32_t r = blockIdx.x;
    const size_t sr = (size_t)(r % src_rows) * width;
    const size_t dr = (size_t)r * width;
    for (uint32_t d = threadIdx.x; d < width; d += blockDim.x) x[dr + d] += src[sr + d];
}

PD_EXPORT
int pd_add_rows_bcast(void* x, const void* src, uint32_t rows, uint32_t src_rows,
                      uint32_t width, void* stream) {
    if (rows == 0 || src_rows == 0 || width == 0) return 0;
    const uint32_t nth = width < 256u ? ((width + 31u) & ~31u) : 256u;
    pd_add_rows_bcast_kernel<<<rows, nth, 0, (cudaStream_t)stream>>>(
        (float*)x, (const float*)src, src_rows, width);
    return pd_launch_status();
}

// Multi-column dp4a GEMV (llama's mmvq shape) - the B=2..8 decode matmul. One
// block per OUTPUT row (out_dim-way parallelism, the proven 97%-DRAM gemv shape):
// the weight row is read once per block via streaming int4 loads and dotted
// against up to 8 quantized activation columns. The int8 x rows make the
// per-block re-reads L2-trivial (B*in_dim bytes), which is exactly what the f32
// tiled kernel could not achieve - at small B this is strictly better than the
// shared-staging MT kernel (which stays the B>8 route).
// (Raising this to 32 for a one-pass B<=32 dense rung measured worse twice
// over: the 32-col variant is compute-bound at 51 GB/s, and the fatter acc
// array alone regressed the ncols=1 lm_head 452 -> 652 us - the registers
// stay live regardless of ncols. 8 stands.)
#define PD_NC_MAX 8
// Row body shared by the single and multi-segment nc kernels - one block dots
// one output row against ncols staged int8 activation columns. Extracted
// verbatim from the single kernel so the multi merge below is
// bit-identical per segment by construction (same walk, same reduce order).
__device__ __forceinline__ void pd_q8nc_row_body(
    const int8_t* __restrict__ data, const __half* __restrict__ scale,
    const float* __restrict__ bias, const int8_t* __restrict__ xq,
    const float* __restrict__ xs, float* __restrict__ y, uint32_t in_dim,
    uint32_t out_dim, uint32_t ncols, uint32_t o) {
    uint32_t tid = threadIdx.x, nth = blockDim.x;
    uint32_t n_blocks = in_dim >> 5;
    __shared__ float wsum[32];
    const int8_t* row = data + (size_t)o * in_dim;
    const __half* srow = scale + (size_t)o * n_blocks;
    // compile-time-indexed accumulators (runtime indices would spill to local)
    float acc[PD_NC_MAX];
#pragma unroll
    for (uint32_t c = 0; c < PD_NC_MAX; ++c) acc[c] = 0.0f;

    for (uint32_t base = tid * 16u; base < in_dim; base += nth * 16u) {
        int4 wv = __ldcs(reinterpret_cast<const int4*>(row + base));
        float ws = __half2float(__ldcs(srow + (base >> 5)));
#pragma unroll
        for (uint32_t c = 0; c < PD_NC_MAX; ++c) {
            if (c >= ncols) break;
            int4 xv = *reinterpret_cast<const int4*>(xq + (size_t)c * in_dim + base);
            int s = __dp4a(wv.x, xv.x, 0);
            s = __dp4a(wv.y, xv.y, s);
            s = __dp4a(wv.z, xv.z, s);
            s = __dp4a(wv.w, xv.w, s);
            acc[c] += ws * xs[(size_t)c * n_blocks + (base >> 5)] * (float)s;
        }
    }
    uint32_t lane = tid & 31u, warp = tid >> 5, nwarps = (nth + 31u) >> 5;
#pragma unroll
    for (uint32_t c = 0; c < PD_NC_MAX; ++c) {
        if (c >= ncols) break;
        float a = acc[c];
        for (uint32_t s2 = 16; s2 > 0; s2 >>= 1) a += __shfl_down_sync(0xffffffffu, a, s2);
        if (lane == 0) wsum[warp] = a;
        __syncthreads();
        if (tid == 0) {
            float v = 0.0f;
            for (uint32_t w = 0; w < nwarps; ++w) v += wsum[w];
            if (bias) v += bias[o];
            y[(size_t)c * out_dim + o] = v;
        }
        __syncthreads();
    }
}

__global__ void __launch_bounds__(256) pd_q8_0_gemv_dp4a_nc_kernel(
    const int8_t* __restrict__ data, const __half* __restrict__ scale,
    const float* __restrict__ bias, const int8_t* __restrict__ xq,
    const float* __restrict__ xs, float* __restrict__ y, uint32_t in_dim,
    uint32_t out_dim, uint32_t ncols) {
    PD_PDL_ARM();
    uint32_t o = blockIdx.x;
    if (o >= out_dim) return;
    pd_q8nc_row_body(data, scale, bias, xq, xs, y, in_dim, out_dim, ncols, o);
}

// Multi-segment nc GEMV - up to four same-in_dim planes sharing one staged
// activation, one launch (the batched-decode q|k|v|g and shexp gate|up
// merges; entry 317's launch economics at the r=2..4 tick: laguna c4 ran 8
// nc launches/layer at a ~10.3us avg where the band's byte floor is ~3us).
// Segment resolve is a constant-index if-chain: runtime
// indexing into a by-value param struct spills to local (the granite-30b
// STACK:120 trap).
struct PdQ8NcSeg {
    const int8_t* data;
    const __half* scale;
    const float* bias;
    float* y;
    uint32_t out_dim;
};
struct PdQ8NcSegs4 {
    PdQ8NcSeg s[4];
};

__global__ void __launch_bounds__(256) pd_q8_0_gemv_dp4a_nc_multi_kernel(
    PdQ8NcSegs4 segs, const int8_t* __restrict__ xq,
    const float* __restrict__ xs, uint32_t in_dim, uint32_t ncols) {
    PD_PDL_ARM();
    uint32_t o = blockIdx.x;
    if (o < segs.s[0].out_dim) {
        pd_q8nc_row_body(segs.s[0].data, segs.s[0].scale, segs.s[0].bias, xq, xs,
                         segs.s[0].y, in_dim, segs.s[0].out_dim, ncols, o);
        return;
    }
    o -= segs.s[0].out_dim;
    if (o < segs.s[1].out_dim) {
        pd_q8nc_row_body(segs.s[1].data, segs.s[1].scale, segs.s[1].bias, xq, xs,
                         segs.s[1].y, in_dim, segs.s[1].out_dim, ncols, o);
        return;
    }
    o -= segs.s[1].out_dim;
    if (o < segs.s[2].out_dim) {
        pd_q8nc_row_body(segs.s[2].data, segs.s[2].scale, segs.s[2].bias, xq, xs,
                         segs.s[2].y, in_dim, segs.s[2].out_dim, ncols, o);
        return;
    }
    o -= segs.s[2].out_dim;
    if (o < segs.s[3].out_dim) {
        pd_q8nc_row_body(segs.s[3].data, segs.s[3].scale, segs.s[3].bias, xq, xs,
                         segs.s[3].y, in_dim, segs.s[3].out_dim, ncols, o);
    }
}

PD_EXPORT
int pd_q8_0_gemv_dp4a_nc(const void* data, const void* scale, const void* xq,
                         const void* xs, void* y, uint32_t in_dim, uint32_t out_dim,
                         uint32_t ncols, void* stream) {
    if (out_dim == 0 || ncols == 0) return 0;
    if (ncols > PD_NC_MAX) return cudaErrorInvalidValue;
    pd_pdl_go(pd_q8_0_gemv_dp4a_nc_kernel, dim3(out_dim), dim3(256), 0,
              (cudaStream_t)stream, (const int8_t*)data, (const __half*)scale,
              (const float*)0, (const int8_t*)xq, (const float*)xs, (float*)y,
              in_dim, out_dim, ncols);
    return pd_launch_status();
}

// Multi-segment launcher: unused trailing slots pad with slot-0 pointers and
// out_dim=0 (valid pointers, zero blocks resolve there - no work, no reads).
PD_EXPORT
int pd_q8_0_gemv_dp4a_nc_multi(
    const void* d0, const void* s0, const void* b0, void* y0, uint32_t out0,
    const void* d1, const void* s1, const void* b1, void* y1, uint32_t out1,
    const void* d2, const void* s2, const void* b2, void* y2, uint32_t out2,
    const void* d3, const void* s3, const void* b3, void* y3, uint32_t out3,
    const void* xq, const void* xs, uint32_t in_dim, uint32_t n_segs,
    uint32_t ncols, void* stream) {
    if (n_segs == 0 || n_segs > 4 || ncols == 0) return cudaErrorInvalidValue;
    if (ncols > PD_NC_MAX) return cudaErrorInvalidValue;
    PdQ8NcSegs4 segs;
    segs.s[0] = {(const int8_t*)d0, (const __half*)s0, (const float*)b0, (float*)y0, out0};
    segs.s[1] = n_segs > 1
        ? PdQ8NcSeg{(const int8_t*)d1, (const __half*)s1, (const float*)b1, (float*)y1, out1}
        : PdQ8NcSeg{(const int8_t*)d0, (const __half*)s0, (const float*)0, (float*)y0, 0u};
    segs.s[2] = n_segs > 2
        ? PdQ8NcSeg{(const int8_t*)d2, (const __half*)s2, (const float*)b2, (float*)y2, out2}
        : PdQ8NcSeg{(const int8_t*)d0, (const __half*)s0, (const float*)0, (float*)y0, 0u};
    segs.s[3] = n_segs > 3
        ? PdQ8NcSeg{(const int8_t*)d3, (const __half*)s3, (const float*)b3, (float*)y3, out3}
        : PdQ8NcSeg{(const int8_t*)d0, (const __half*)s0, (const float*)0, (float*)y0, 0u};
    uint32_t total = segs.s[0].out_dim + segs.s[1].out_dim + segs.s[2].out_dim +
                     segs.s[3].out_dim;
    if (total == 0) return 0;
    pd_pdl_go(pd_q8_0_gemv_dp4a_nc_multi_kernel, dim3(total), dim3(256), 0,
              (cudaStream_t)stream, segs, (const int8_t*)xq, (const float*)xs,
              in_dim, ncols);
    return pd_launch_status();
}

// Bias-carrying variant of the nc GEMV, same kernel body: the B=1 decode
// projections (wq/wk/wv/wo carry biases on gpt-oss) get the block-per-row
// repacked shape. The warp-per-row 34-byte-block GEMV they rode before tops
// out well under half of DRAM on many-SM parts (misaligned 34 B strides, and
// out_dim/8 blocks cannot fill a 188-SM die); this shape measured 1.0-1.4 TB/s
// where that one measured 0.6.
PD_EXPORT
int pd_q8_0_gemv_dp4a_nc_b(const void* data, const void* scale, const void* bias,
                           const void* xq, const void* xs, void* y, uint32_t in_dim,
                           uint32_t out_dim, uint32_t ncols, void* stream) {
    if (out_dim == 0 || ncols == 0) return 0;
    if (ncols > PD_NC_MAX) return cudaErrorInvalidValue;
    pd_pdl_go(pd_q8_0_gemv_dp4a_nc_kernel, dim3(out_dim), dim3(256), 0,
              (cudaStream_t)stream, (const int8_t*)data, (const __half*)scale,
              (const float*)bias, (const int8_t*)xq, (const float*)xs, (float*)y,
              in_dim, out_dim, ncols);
    return pd_launch_status();
}

// Small-batch GEMM tiling parameters (shared by the f32 and dp4a variants).
#define PD_MT_ROWS 16            // max batch rows per weight pass (f32 shared: 16*512*4 = 32 KB)
#define PD_MT_BM 16              // output rows per block (8 warps x 2)
#define PD_MT_CHUNK 512          // x elems staged per iteration

// int8 MMQ small-batch GEMM (the llama.cpp large-batch method): activations
// pre-quantized to per-32-block int8 (pd_quantize_q8), weights read once as int8,
// integer __dp4a inner product, f32 accumulate of per-16 partials. 4x fewer compute
// ops and 4x smaller x staging than the f32 tiled GEMM - the weight read runs at
// full DRAM bandwidth instead of latency-bound at ~40%. Not bit-identical to the
// f32 path (activation quantization ~4e-3 rel) - the same numeric class llama.cpp
// itself uses for its decode/batch matmuls; gated at the token level.
// Templated on the rows-per-weight-pass tile and threads-per-block: per-row
// math (chunk order, lane assignment, accumulation) is identical across
// instantiations, so a row's output is bit-identical whichever variant
// executes it (the 27B spec gate cross-proves the 16- and 24-row tiles).
template <uint32_t MT_ROWS, uint32_t TPB>
__global__ void __launch_bounds__(TPB) pd_q8_0_gemm_mt_dp4a_kernel(
    const int8_t* __restrict__ data, const __half* __restrict__ scale,
    const int8_t* __restrict__ xq, const float* __restrict__ xs,
    const float* __restrict__ bias, float* __restrict__ y, uint32_t in_dim,
    uint32_t out_dim, uint32_t batch) {
    uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    uint32_t o0 = blockIdx.x * (TPB / 16u) + warp * 2;   // two output rows per warp
    uint32_t n_blocks = in_dim >> 5;
    // grid.z tiles the batch: each z-tile re-reads the weight once, so weight
    // traffic scales with ceil(B/MT_ROWS). The host's tile choice lives in the
    // launcher below.
    {
        uint32_t b0 = blockIdx.z * MT_ROWS;
        xq += (size_t)b0 * in_dim;
        xs += (size_t)b0 * n_blocks;
        y += (size_t)b0 * out_dim;
        batch = (batch - b0 < MT_ROWS) ? (batch - b0) : MT_ROWS;
    }
    __shared__ int4 xqs[MT_ROWS * (PD_MT_CHUNK / 16)];   // int8 chunk rows
    __shared__ float xss[MT_ROWS * (PD_MT_CHUNK / 32)];  // per-32-block scales

    // compile-time-indexed accumulators (runtime indices would spill to local)
    float acc0[MT_ROWS], acc1[MT_ROWS];
#pragma unroll
    for (uint32_t t = 0; t < MT_ROWS; ++t) { acc0[t] = 0.0f; acc1[t] = 0.0f; }

    const int8_t* row0 = data + (size_t)o0 * in_dim;
    const int8_t* row1 = data + (size_t)(o0 + 1) * in_dim;
    const __half* sc0 = scale + (size_t)o0 * n_blocks;
    const __half* sc1 = scale + (size_t)(o0 + 1) * n_blocks;

    for (uint32_t c0 = 0; c0 < in_dim; c0 += PD_MT_CHUNK) {
        // stage the quantized x chunk (int4 = 16 int8 each) + its block scales
        for (uint32_t i = tid; i < batch * (PD_MT_CHUNK / 16); i += blockDim.x) {
            uint32_t t = i / (PD_MT_CHUNK / 16), k = i % (PD_MT_CHUNK / 16);
            uint32_t src = c0 + k * 16u;
            xqs[i] = (src < in_dim)
                ? *reinterpret_cast<const int4*>(xq + (size_t)t * in_dim + src)
                : make_int4(0, 0, 0, 0);
        }
        for (uint32_t i = tid; i < batch * (PD_MT_CHUNK / 32); i += blockDim.x) {
            uint32_t t = i / (PD_MT_CHUNK / 32), b = i % (PD_MT_CHUNK / 32);
            uint32_t blk = (c0 >> 5) + b;
            xss[i] = (blk < n_blocks) ? xs[(size_t)t * n_blocks + blk] : 0.0f;
        }
        __syncthreads();
        uint32_t base = c0 + lane * 16u;
        if (o0 < out_dim && base < in_dim) {
            int4 w0 = __ldcs(reinterpret_cast<const int4*>(row0 + base));
            int4 w1 = __ldcs(reinterpret_cast<const int4*>(row1 + base));
            float ws0 = __half2float(__ldcs(sc0 + (base >> 5)));
            float ws1 = __half2float(__ldcs(sc1 + (base >> 5)));
#pragma unroll
            for (uint32_t t = 0; t < MT_ROWS; ++t) {
                if (t >= batch) break;
                int4 xv = xqs[t * (PD_MT_CHUNK / 16) + lane];
                int s0 = __dp4a(w0.x, xv.x, 0);
                s0 = __dp4a(w0.y, xv.y, s0);
                s0 = __dp4a(w0.z, xv.z, s0);
                s0 = __dp4a(w0.w, xv.w, s0);
                int s1 = __dp4a(w1.x, xv.x, 0);
                s1 = __dp4a(w1.y, xv.y, s1);
                s1 = __dp4a(w1.z, xv.z, s1);
                s1 = __dp4a(w1.w, xv.w, s1);
                float xsc = xss[t * (PD_MT_CHUNK / 32) + ((lane * 16u) >> 5)];
                acc0[t] += ws0 * xsc * (float)s0;
                acc1[t] += ws1 * xsc * (float)s1;
            }
        }
        __syncthreads();
    }
    if (o0 >= out_dim) return;
#pragma unroll
    for (uint32_t t = 0; t < MT_ROWS; ++t) {
        if (t >= batch) break;
        float a0 = acc0[t], a1 = acc1[t];
        for (uint32_t s = 16; s > 0; s >>= 1) {
            a0 += __shfl_down_sync(0xffffffffu, a0, s);
            a1 += __shfl_down_sync(0xffffffffu, a1, s);
        }
        if (lane == 0) {
            // bias fold: identical rounding to store-then-bias_add (the sum is
            // complete before the single f32 bias add)
            y[(size_t)t * out_dim + o0] = bias ? a0 + bias[o0] : a0;
            if (o0 + 1 < out_dim)
                y[(size_t)t * out_dim + o0 + 1] = bias ? a1 + bias[o0 + 1] : a1;
        }
    }
}

static int pd_q8_0_gemm_mt_dp4a_impl(const void* data, const void* scale, const void* xq,
                                     const void* xs, const void* bias, void* y,
                                     uint32_t in_dim, uint32_t out_dim, uint32_t batch,
                                     void* stream) {
    if (out_dim == 0 || batch == 0) return 0;
    // The 24-row tile pays only for 17..24-row batches, where one partial-tile
    // pass (<=6 dp4a per weight byte) replaces two bandwidth-bound 16-row passes
    // - the 27B spec-verify chunk (B*(K+1)=20) measured +21%. Fatter or full
    // wide tiles choke the INT pipe at the fat variant's 2-block residency:
    // 32-row single/half passes AND 24x3 both measured slower than 16-row
    // z-tiling on the 9B at B=32/64 despite reading less weight traffic (those
    // regimes are not bandwidth-bound, so saved passes buy nothing). 256-thread
    // blocks won the sweep (128-thread 8-row blocks read ~9% slower); the gap
    // to the nc kernel's bandwidth is P6c's tensor-core tile, not dp4a tuning.
    // PD_MT_NOWIDE=1 pins the legacy 16-row tile (A/B diagnostics).
    static int wide = -1;
    if (wide < 0) {
        const char* e = pd_env("PD_MT_NOWIDE");
        wide = (e && e[0] == '1') ? 0 : 1;
    }
    uint32_t rows = (wide && batch > PD_MT_ROWS && batch <= 24u) ? 24u : PD_MT_ROWS;
    dim3 grid((out_dim + PD_MT_BM - 1) / PD_MT_BM, 1, (batch + rows - 1) / rows);
    if (rows == 24u) {
        pd_q8_0_gemm_mt_dp4a_kernel<24, 256><<<grid, 256, 0, (cudaStream_t)stream>>>(
            (const int8_t*)data, (const __half*)scale, (const int8_t*)xq, (const float*)xs,
            (const float*)bias, (float*)y, in_dim, out_dim, batch);
    } else {
        pd_q8_0_gemm_mt_dp4a_kernel<PD_MT_ROWS, 256><<<grid, 256, 0, (cudaStream_t)stream>>>(
            (const int8_t*)data, (const __half*)scale, (const int8_t*)xq, (const float*)xs,
            (const float*)bias, (float*)y, in_dim, out_dim, batch);
    }
    return pd_launch_status();
}

PD_EXPORT
int pd_q8_0_gemm_mt_dp4a(const void* data, const void* scale, const void* xq,
                         const void* xs, void* y, uint32_t in_dim, uint32_t out_dim,
                         uint32_t batch, void* stream) {
    return pd_q8_0_gemm_mt_dp4a_impl(data, scale, xq, xs, nullptr, y, in_dim, out_dim,
                                     batch, stream);
}

// Bias-carrying variant (the serving 5..=8 dense rung: gpt-oss projections are
// biased; folding kills the trailing pd_bias_add launch per projection).
PD_EXPORT
int pd_q8_0_gemm_mt_dp4a_b(const void* data, const void* scale, const void* xq,
                           const void* xs, const void* bias, void* y, uint32_t in_dim,
                           uint32_t out_dim, uint32_t batch, void* stream) {
    return pd_q8_0_gemm_mt_dp4a_impl(data, scale, xq, xs, bias, y, in_dim, out_dim,
                                     batch, stream);
}


// ------------------------------------------------------------------ sam attn
// SAM ViTDet attention (DeepSeek-OCR's first tower):
//
//   out = softmax(q·kᵀ·scale + rel_h + rel_w) · v
//
// where the bias is the DECOMPOSED relative position term from mvitv2 -
// per-axis tables contracted against the query itself, so unlike a fixed
// additive mask it is q-dependent:
//
//   rel_h[i, ky] = Σ_d q[i,d] · Rh[qy(i), ky, d]     (UNSCALED q - the
//   rel_w[i, kx] = Σ_d q[i,d] · Rw[qx(i), kx, d]      reference contracts raw
//   score[i, j] = q·k·scale + rel_h[i, j/side] + rel_w[i, j%side]   q, and SDPA
//                                                     scales only q·kᵀ)
//
// Rh/Rw arrive as full [side, side, hd] tables, already indexed by relative
// coordinate (and, for the 640px views, already linear-resampled) on the host:
// the checkpoint's (2·side-1, hd) parameter is a per-GEOMETRY constant, so the
// host does `get_rel_pos` once per (layer, side) and the kernel just reads
// [qy][ky]. The contraction itself cannot leave the kernel - it is 2·side dots
// per query row (~1/32nd of the q·k work at side 14) but its RESULT feeds the
// softmax, and materializing score matrices to add it outside is exactly what
// SDPA exists to avoid.
//
// One kernel serves both attention classes of the tower: the 8 windowed blocks
// launch with n_batch = number of 14×14 windows (window partition pads the
// grid, so every window is complete - side is always the full window/grid side
// and n = side²), the 4 global blocks with n_batch = views and side = the full
// grid (64 at 1024px, 40 at 640px).
//
// Shape is `pd_vision_attn_kernel`'s (warp-per-query, 32-key shared tiles,
// online softmax - see its header for why that shape), narrowed to MAXD 64
// because every SAM in this family is head_dim 64: the narrower tiles put the
// whole block at ~24 KB static shared, two blocks per SM in the default
// carveout with no opt-in. Exact-f32 class - this is the correctness kernel;
// an mma sibling is a measured-later rung, same history as the tower above.
#define PD_SAM_QT 8u
#define PD_SAM_MAXD 64u
#define PD_SAM_MAXSIDE 64u
__global__ void __launch_bounds__(256) pd_sam_attn_kernel(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ rh,
    const float* __restrict__ rw, float* __restrict__ out, uint32_t side,
    uint32_t n_heads, uint32_t hd, float scale) {
    const uint32_t n = side * side;
    const uint32_t h = blockIdx.x;
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t i = blockIdx.y * PD_SAM_QT + warp;  // this warp's query row
    const size_t base = (size_t)blockIdx.z * n * n_heads * hd;

    __shared__ float sh_k[32][PD_SAM_MAXD + 1u];
    __shared__ float sh_v[32][PD_SAM_MAXD + 1u];
    __shared__ float sh_q[PD_SAM_QT][PD_SAM_MAXD];
    __shared__ float sh_w[PD_SAM_QT][32];
    __shared__ float sh_rh[PD_SAM_QT][PD_SAM_MAXSIDE];
    __shared__ float sh_rw[PD_SAM_QT][PD_SAM_MAXSIDE];

    // i is warp-uniform, so these branches never split a warp and the
    // __syncwarp() ladder below is legal.
    const bool live = i < n;
    if (live) {
        // stage the query UNSCALED - the rel-pos dots contract raw q
        for (uint32_t d = lane; d < hd; d += 32u)
            sh_q[warp][d] = q[base + ((size_t)i * n_heads + h) * hd + d];
        __syncwarp();
        const uint32_t qy = i / side, qx = i % side;
        for (uint32_t j = lane; j < side; j += 32u) {
            const float* rhp = rh + ((size_t)qy * side + j) * hd;
            const float* rwp = rw + ((size_t)qx * side + j) * hd;
            float ph = 0.0f, pw = 0.0f;
            for (uint32_t d = 0; d < hd; ++d) {
                const float qd = sh_q[warp][d];
                ph = fmaf(qd, rhp[d], ph);
                pw = fmaf(qd, rwp[d], pw);
            }
            sh_rh[warp][j] = ph;
            sh_rw[warp][j] = pw;
        }
        __syncwarp();  // rel dots read every lane's sh_q - finish before scaling
        for (uint32_t d = lane; d < hd; d += 32u) sh_q[warp][d] *= scale;
    }

    float acc[2] = {0.0f, 0.0f};
    float m = -3.402823e38f, l = 0.0f;
    const uint32_t nd = (hd + 31u) >> 5;

    for (uint32_t j0 = 0; j0 < n; j0 += 32u) {
        const uint32_t jl = min(32u, n - j0);
        for (uint32_t idx = threadIdx.x; idx < jl * hd; idx += 256u) {
            const uint32_t jj = idx / hd, dd = idx % hd;
            sh_k[jj][dd] = k[base + ((size_t)(j0 + jj) * n_heads + h) * hd + dd];
            sh_v[jj][dd] = v[base + ((size_t)(j0 + jj) * n_heads + h) * hd + dd];
        }
        __syncthreads();  // also orders the pre-loop sh_q scale for tile 0
        if (live) {
            float dot = -3.402823e38f;
            if (lane < jl) {
                float p = 0.0f;
                for (uint32_t d = 0; d < hd; ++d) p += sh_q[warp][d] * sh_k[lane][d];
                const uint32_t j = j0 + lane;
                dot = p + sh_rh[warp][j / side] + sh_rw[warp][j % side];
            }
            float tmax = dot;
#pragma unroll
            for (uint32_t off = 16; off > 0; off >>= 1)
                tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, off));
            const float m_new = fmaxf(m, tmax);
            const float corr = __expf(m - m_new);
            const float w = (lane < jl) ? __expf(dot - m_new) : 0.0f;
            sh_w[warp][lane] = w;
            float wsum = w;
#pragma unroll
            for (uint32_t off = 16; off > 0; off >>= 1)
                wsum += __shfl_xor_sync(0xffffffffu, wsum, off);
            __syncwarp();
#pragma unroll
            for (uint32_t c = 0; c < 2; ++c) {
                uint32_t d = lane + c * 32u;
                if (c < nd && d < hd) {
                    float a = acc[c] * corr;
                    for (uint32_t j = 0; j < jl; ++j)
                        a = fmaf(sh_w[warp][j], sh_v[j][d], a);
                    acc[c] = a;
                }
            }
            l = l * corr + wsum;
            m = m_new;
        }
        __syncthreads();
    }
    if (live) {
#pragma unroll
        for (uint32_t c = 0; c < 2; ++c) {
            uint32_t d = lane + c * 32u;
            if (c < nd && d < hd)
                out[base + (size_t)i * n_heads * hd + (size_t)h * hd + d] = acc[c] / l;
        }
    }
}

// The mma sibling. An OCR smoke run found this one band at 37.8% of c8 GPU
// time - 2972 us/launch for the exact-f32 form over a 576-launch window,
// where a bf16 flash tile of this class runs two orders faster. Shape
// is `pd_vision_attn_mma_kernel`'s (m16n8k16 QK/PV, ldmatrix staging, online
// softmax in the C-fragment layout - the doctrine's current tile), pinned to
// SAM's hd 64, plus the one thing that kernel doesn't have: the decomposed
// rel-pos bias on the logits.
//
// The bias never becomes a materialized [q, k] term. Keys are row-major
// (j = jy·side + jx), so per query it splits into two small tables:
//   - Pw[q][jx] = raw_q · Rw[qx(q), jx]  - jx cycles through 0..side in every
//     key tile, so each warp computes its 16 rows × side table once up front
//     (f16 in shared: the QK dot already rounds q and k to f16, a bias table
//     in the same class adds nothing new to the error budget);
//   - Ph[q][jy] = raw_q · Rh[qy(q), jy]  - a 64-key tile spans at most
//     63/side+2 distinct jy rows, so the per-tile refresh is 16×span dots per
//     warp (f32, it is tiny). Sides below 11 would overflow the 8-slot span
//     and stay on the exact-f32 kernel above.
// Both dots now contract the A-FRAGMENTS' f16 q: hd 64 makes scale
// an exact power of two, so f16(q·scale)·(1/scale) is f16(q) bit for bit -
// see pd_samm_bias_partial for the lg_throttle numbers that forced this off
// the original raw-q-from-global form.
//
// NUMERIC CLASS: same as the vision mma arm - f16 fragments, f32 accumulate/
// softmax/output - gated at its own class tolerance in
// tests/gpu_sam_attn_parity.rs next to the 1e-4 exact-f32 leg.
// PADDOCK_NO_SAM_MMA=1 keeps the f32 kernel reachable, read live per launch
// (same reasoning as PADDOCK_NO_VIS_MMA).
//
// Static shared: K/V 18,432 + Pw 16,384 + Ph 4,096 ≈ 38.9 KB - two blocks/SM
// with the max carveout on sm_120 (<~51 KB/block per the probed per-SM
// ceiling), same occupancy shape as the f32 kernel it replaces.
#define PD_SAMM_KT 64u
#define PD_SAMM_HD 64u
#define PD_SAMM_DPD (PD_SAMM_HD + 8u)
#define PD_SAMM_QW 8u
#define PD_SAMM_QT (PD_SAMM_QW * 16u)
#define PD_SAMM_SPAN 8u

// Quad-cooperative bias-dot partial. The first cut of the
// bias tables contracted RAW q from GLOBAL, one lane per dot, 128 serial
// scalar LDGs each - live pc-sampling put lg_throttle at 2.5x every other
// stall (277k samples vs mio 112k) with DRAM at 2% of roof: the LSU issue
// queue was drowning in redundant scalar re-reads of the same q rows. But the
// A-fragments already hold q as f16(q x scale) in registers, and hd 64 pins
// scale to 1/8 - an exact power of two, so f16(q x 0.125) x 8 is f16(q) bit
// for bit (mantissa untouched). Each lane contracts its own 16 fragment dims
// against float2 loads of the table row; the caller folds the owning quad
// with two xor-shuffles and unscales once. Refresh LDG count drops ~16x and
// every remaining load is 8B. Numeric class: the bias dots move from raw-f32
// q to the fragments' f16-rounded q with a quad-regrouped summation - inside
// the mma arm's existing 1e-3 class (Pw storage already rounded f16).
__device__ __forceinline__ float pd_samm_bias_partial(
    const uint32_t (&qa)[PD_SAMM_HD / 16u][4], uint32_t e, uint32_t t4,
    const float* __restrict__ tp) {
    float p = 0.f;
#pragma unroll
    for (uint32_t d0 = 0; d0 < PD_SAMM_HD / 16u; ++d0) {
        const uint32_t c0 = d0 * 16u + 2u * t4;
        const __half2 q01 = *reinterpret_cast<const __half2*>(&qa[d0][e]);
        const __half2 q89 = *reinterpret_cast<const __half2*>(&qa[d0][e + 2u]);
        const float2 t01 = *reinterpret_cast<const float2*>(tp + c0);
        const float2 t89 = *reinterpret_cast<const float2*>(tp + c0 + 8u);
        const float2 f01 = __half22float2(q01);
        const float2 f89 = __half22float2(q89);
        p = fmaf(f01.x, t01.x, p);
        p = fmaf(f01.y, t01.y, p);
        p = fmaf(f89.x, t89.x, p);
        p = fmaf(f89.y, t89.y, p);
    }
    return p;
}
__global__ void __launch_bounds__(32u * PD_SAMM_QW) pd_sam_attn_mma_kernel(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ rh,
    const float* __restrict__ rw, float* __restrict__ out, uint32_t side,
    uint32_t n_heads, float scale) {
#if PD_FA_OK
    constexpr uint32_t KT = PD_SAMM_KT, HD = PD_SAMM_HD, DPD = PD_SAMM_DPD;
    constexpr uint32_t NT = 32u * PD_SAMM_QW;
    const uint32_t n = side * side;
    const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
    const uint32_t g8 = lane >> 2, t4 = lane & 3u, lg = lane >> 3;
    const uint32_t h = blockIdx.y;
    const uint32_t row0 = blockIdx.x * PD_SAMM_QT + warp * 16u;
    const size_t base = (size_t)blockIdx.z * n * n_heads * HD;

    __shared__ __half sm_k[KT][DPD];
    __shared__ __half sm_v[KT][DPD];
    __shared__ __half sm_pw[PD_SAMM_QW][16][PD_SAM_MAXSIDE];
    __shared__ float sm_ph[PD_SAMM_QW][16][PD_SAMM_SPAN];

    // A-fragment rows this lane owns (mma layout: row = lane/4, and +8),
    // scale folded into the f16 round exactly as the vision arm does it
    const uint32_t jr[2] = {g8, g8 + 8u};
    uint32_t qa[HD / 16u][4];
#pragma unroll
    for (uint32_t d0 = 0; d0 < HD / 16u; ++d0) {
#pragma unroll
        for (uint32_t e = 0; e < 2u; ++e) {
            const uint32_t b = row0 + jr[e];
            const float* qp =
                b < n ? q + base + ((size_t)b * n_heads + h) * HD : nullptr;
            const uint32_t c0 = d0 * 16u + 2u * t4, c1 = c0 + 8u;
            const float f0 = qp ? qp[c0] * scale : 0.f;
            const float f1 = qp ? qp[c0 + 1u] * scale : 0.f;
            const float f8 = qp ? qp[c1] * scale : 0.f;
            const float f9 = qp ? qp[c1 + 1u] * scale : 0.f;
            const __half2 p01 = __floats2half2_rn(f0, f1);
            const __half2 p89 = __floats2half2_rn(f8, f9);
            qa[d0][e] = *reinterpret_cast<const uint32_t*>(&p01);
            qa[d0][e + 2u] = *reinterpret_cast<const uint32_t*>(&p89);
        }
    }

    // Pw prelude: q · Rw for the warp's 16 rows, every jx - once, quad-
    // cooperative off the A-fragments (see pd_samm_bias_partial: the scaled
    // f16 fragments are f16(q)/8, so one exact x8 recovers the raw-q dot).
    // Only this warp reads its slice, so a warp sync is the whole fence.
    const float unscale = 1.0f / scale;  // hd 64 -> scale 0.125, both exact
    for (uint32_t jx = 0; jx < side; ++jx) {
#pragma unroll
        for (uint32_t e = 0; e < 2u; ++e) {
            const uint32_t b = row0 + jr[e];
            float p = 0.f;
            if (b < n) {
                const float* wp = rw + ((size_t)(b % side) * side + jx) * HD;
                p = pd_samm_bias_partial(qa, e, t4, wp);
            }
            p += __shfl_xor_sync(0xffffffffu, p, 1);
            p += __shfl_xor_sync(0xffffffffu, p, 2);
            if (t4 == 0) sm_pw[warp][jr[e]][jx] = __float2half_rn(p * unscale);
        }
    }
    __syncwarp();

    float m_st[2] = {-1e30f, -1e30f}, l_st[2] = {0.f, 0.f};
    float o_acc[HD / 8u][4];
#pragma unroll
    for (uint32_t nt = 0; nt < HD / 8u; ++nt)
#pragma unroll
        for (uint32_t e = 0; e < 4u; ++e) o_acc[nt][e] = 0.f;

    for (uint32_t t0 = 0; t0 < n; t0 += KT) {
        const uint32_t jl = min(KT, n - t0);
        // co-stage K then V as f16 (vision arm's staging, hd pinned to 64)
        constexpr uint32_t SPAN8 = KT * (HD / 8u);
        for (uint32_t u = tid; u < 2u * SPAN8; u += NT) {
            const bool isv = u >= SPAN8;
            const uint32_t ur = isv ? u - SPAN8 : u;
            const uint32_t kk = ur / (HD / 8u), d8 = (ur % (HD / 8u)) * 8u;
            const uint32_t ks = t0 + kk;
            __half* dst = (isv ? &sm_v[0][0] : &sm_k[0][0]) + (size_t)kk * DPD + d8;
            if (ks < n) {
                const float* src =
                    (isv ? v : k) + base + ((size_t)ks * n_heads + h) * HD + d8;
                const float4 a = *reinterpret_cast<const float4*>(src);
                const float4 bq = *reinterpret_cast<const float4*>(src + 4);
                *reinterpret_cast<__half2*>(dst + 0) = __floats2half2_rn(a.x, a.y);
                *reinterpret_cast<__half2*>(dst + 2) = __floats2half2_rn(a.z, a.w);
                *reinterpret_cast<__half2*>(dst + 4) = __floats2half2_rn(bq.x, bq.y);
                *reinterpret_cast<__half2*>(dst + 6) = __floats2half2_rn(bq.z, bq.w);
            } else {
                *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        // Ph refresh for this tile's jy range - same quad-cooperative
        // fragment dot as the Pw prelude; storage stays f32, layout unchanged
        const uint32_t jy0 = t0 / side;
        const uint32_t span = (t0 + jl - 1u) / side - jy0 + 1u;
        for (uint32_t s = 0; s < span; ++s) {
#pragma unroll
            for (uint32_t e = 0; e < 2u; ++e) {
                const uint32_t b = row0 + jr[e];
                float p = 0.f;
                if (b < n) {
                    const float* hp = rh + ((size_t)(b / side) * side + jy0 + s) * HD;
                    p = pd_samm_bias_partial(qa, e, t4, hp);
                }
                p += __shfl_xor_sync(0xffffffffu, p, 1);
                p += __shfl_xor_sync(0xffffffffu, p, 2);
                if (t4 == 0) sm_ph[warp][jr[e]][s] = p * unscale;
            }
        }
        __syncthreads();  // K/V staged block-wide; also fences sm_ph reuse

        float s_acc[KT / 8u][4];
#pragma unroll
        for (uint32_t nt = 0; nt < KT / 8u; ++nt)
#pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) s_acc[nt][e] = 0.f;
#pragma unroll
        for (uint32_t d0 = 0; d0 < HD / 16u; ++d0) {
#pragma unroll
            for (uint32_t np = 0; np < KT / 16u; ++np) {
                const __half* kp = &sm_k[0][0]
                    + (size_t)(np * 16u + (lg >> 1) * 8u + (lane & 7u)) * DPD
                    + d0 * 16u + (lg & 1u) * 8u;
                uint32_t kb4[4];
                const uint32_t ka = (uint32_t)__cvta_generic_to_shared(kp);
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                    : "=r"(kb4[0]), "=r"(kb4[1]), "=r"(kb4[2]), "=r"(kb4[3]) : "r"(ka));
                pd_fa_mma16(s_acc[np * 2u], qa[d0][0], qa[d0][1], qa[d0][2],
                            qa[d0][3], kb4[0], kb4[1]);
                pd_fa_mma16(s_acc[np * 2u + 1u], qa[d0][0], qa[d0][1], qa[d0][2],
                            qa[d0][3], kb4[2], kb4[3]);
            }
        }
        // bias add (before masking - the mask must win), then online softmax
        float mn[2] = {m_st[0], m_st[1]};
#pragma unroll
        for (uint32_t nt = 0; nt < KT / 8u; ++nt) {
            const uint32_t kb = t0 + nt * 8u + 2u * t4;
#pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) {
                const uint32_t kk = kb + (e & 1u);
                if (kk < n) {
                    const uint32_t jy = kk / side;
                    const uint32_t r = jr[e >> 1];
                    s_acc[nt][e] += sm_ph[warp][r][jy - jy0]
                                    + __half2float(sm_pw[warp][r][kk - jy * side]);
                } else {
                    s_acc[nt][e] = -1e30f;
                }
                mn[e >> 1] = fmaxf(mn[e >> 1], s_acc[nt][e]);
            }
        }
#pragma unroll
        for (uint32_t o = 1; o <= 2u; o <<= 1) {
            mn[0] = fmaxf(mn[0], __shfl_xor_sync(0xffffffffu, mn[0], o));
            mn[1] = fmaxf(mn[1], __shfl_xor_sync(0xffffffffu, mn[1], o));
        }
        float ws[2] = {0.f, 0.f};
#pragma unroll
        for (uint32_t nt = 0; nt < KT / 8u; ++nt) {
#pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) {
                const float d = s_acc[nt][e] - mn[e >> 1];
                const float w = d >= -20.f ? __expf(d) : 0.f;
                s_acc[nt][e] = w;
                ws[e >> 1] += w;
            }
        }
#pragma unroll
        for (uint32_t o = 1; o <= 2u; o <<= 1) {
            ws[0] += __shfl_xor_sync(0xffffffffu, ws[0], o);
            ws[1] += __shfl_xor_sync(0xffffffffu, ws[1], o);
        }
        float corr[2];
#pragma unroll
        for (uint32_t r = 0; r < 2u; ++r) {
            const float dc = m_st[r] - mn[r];
            corr[r] = dc >= -20.f ? __expf(dc) : 0.f;
            l_st[r] = l_st[r] * corr[r] + ws[r];
            m_st[r] = mn[r];
        }
#pragma unroll
        for (uint32_t nt = 0; nt < HD / 8u; ++nt) {
            o_acc[nt][0] *= corr[0];
            o_acc[nt][1] *= corr[0];
            o_acc[nt][2] *= corr[1];
            o_acc[nt][3] *= corr[1];
        }
        // PV: weights straight out of the score C-fragment, V via .trans
#pragma unroll
        for (uint32_t kf = 0; kf < KT / 16u; ++kf) {
            const uint32_t c0 = 2u * kf, c1 = c0 + 1u;
            const __half2 a0 = __floats2half2_rn(s_acc[c0][0], s_acc[c0][1]);
            const __half2 a1 = __floats2half2_rn(s_acc[c0][2], s_acc[c0][3]);
            const __half2 a2 = __floats2half2_rn(s_acc[c1][0], s_acc[c1][1]);
            const __half2 a3 = __floats2half2_rn(s_acc[c1][2], s_acc[c1][3]);
            const uint32_t pa0 = *reinterpret_cast<const uint32_t*>(&a0);
            const uint32_t pa1 = *reinterpret_cast<const uint32_t*>(&a1);
            const uint32_t pa2 = *reinterpret_cast<const uint32_t*>(&a2);
            const uint32_t pa3 = *reinterpret_cast<const uint32_t*>(&a3);
            const uint32_t vr = kf * 16u + (lg & 1u) * 8u + (lane & 7u);
            const __half* vp = &sm_v[0][0] + (size_t)vr * DPD + (lg >> 1) * 8u;
#pragma unroll
            for (uint32_t nt = 0; nt < HD / 8u; nt += 2u) {
                uint32_t vb4[4];
                const uint32_t va = (uint32_t)__cvta_generic_to_shared(vp + nt * 8u);
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                    : "=r"(vb4[0]), "=r"(vb4[1]), "=r"(vb4[2]), "=r"(vb4[3]) : "r"(va));
                pd_fa_mma16(o_acc[nt], pa0, pa1, pa2, pa3, vb4[0], vb4[1]);
                pd_fa_mma16(o_acc[nt + 1u], pa0, pa1, pa2, pa3, vb4[2], vb4[3]);
            }
        }
        __syncthreads();  // tiles read before the next stage overwrites them
    }

    const float nrm[2] = {l_st[0] > 0.f ? 1.f / l_st[0] : 0.f,
                          l_st[1] > 0.f ? 1.f / l_st[1] : 0.f};
#pragma unroll
    for (uint32_t nt = 0; nt < HD / 8u; ++nt) {
        const uint32_t dcol = nt * 8u + 2u * t4;
#pragma unroll
        for (uint32_t r = 0; r < 2u; ++r) {
            const uint32_t b = row0 + jr[r];
            if (b >= n) continue;
            float* op = out + base + (size_t)b * n_heads * HD + (size_t)h * HD + dcol;
            op[0] = o_acc[nt][2u * r] * nrm[r];
            op[1] = o_acc[nt][2u * r + 1u] * nrm[r];
        }
    }
#else
    (void)q; (void)k; (void)v; (void)rh; (void)rw; (void)out; (void)side;
    (void)n_heads; (void)scale;
#endif
}

// q/k/v/out are [n_batch, side², heads, hd] f32; rh/rw are [side, side, hd]
// f32, shared by every batch element (rel-pos is relative within a window, so
// all windows of one launch read the same tables). -3 = shape not covered,
// same convention as the other route-not-covered returns.
//
// Dispatch: the mma arm when the device has mma.m16n8k16 and the shape is the
// family's (hd exactly 64, side big enough that a 64-key tile spans ≤ 8 jy
// rows - side ≥ 11; the real sides are 14/40/64). Anything else keeps the
// exact-f32 kernel: a correct answer at a worse speed rather than no answer.
PD_EXPORT
int pd_sam_attn(const void* q, const void* k, const void* v, const void* rh,
                const void* rw, void* out, uint32_t n_batch, uint32_t side,
                uint32_t n_heads, uint32_t hd, float scale, void* stream) {
    if (n_batch == 0 || side == 0 || n_heads == 0) return 0;
    if (hd > PD_SAM_MAXD || side > PD_SAM_MAXSIDE) return -3;
    const uint32_t n = side * side;
    static int mma_arch = -1;
    if (mma_arch < 0) {
        int dev = 0, cc = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&cc, cudaDevAttrComputeCapabilityMajor, dev);
        mma_arch = cc >= 8 ? 1 : 0;
    }
    const bool use_mma = mma_arch && hd == PD_SAMM_HD && side >= 11u
                         && !pd_vm_env_on("PADDOCK_NO_SAM_MMA");
    if (use_mma) {
        static bool attr = false;
        if (!attr) {
            pd_prefer_max_shared(pd_sam_attn_mma_kernel);
            attr = true;
        }
        dim3 grid((n + PD_SAMM_QT - 1u) / PD_SAMM_QT, n_heads, n_batch);
        pd_sam_attn_mma_kernel<<<grid, 32u * PD_SAMM_QW, 0, (cudaStream_t)stream>>>(
            (const float*)q, (const float*)k, (const float*)v, (const float*)rh,
            (const float*)rw, (float*)out, side, n_heads, scale);
        return pd_launch_status();
    }
    dim3 grid(n_heads, (n + PD_SAM_QT - 1u) / PD_SAM_QT, n_batch);
    pd_sam_attn_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(
        (const float*)q, (const float*)k, (const float*)v, (const float*)rh,
        (const float*)rw, (float*)out, side, n_heads, hd, scale);
    return pd_launch_status();
}

// OCR tower patch stem: u8 interleaved-RGB views -> normalized
// f16 patch rows in the SAM conv stem's im2row order, one gather. Replaces
// the host normalize + im2row + f32 upload + f16 convert chain - the f32
// patch plane was 4x the PCIe bytes and ~1.5 ms/tower-chunk of serve-loop
// host time (the c8 idle probe's gap.enc segment). Math matches the host
// path bit for bit: (u8/255 - mean)/std in IEEE f32 (the pack builds
// without fast-math, so both divisions are round-to-nearest), then
// __float2half's RNE - identical to normalize_rgb8 + pd_convert_f32_f16.
__global__ void pd_ocr_patches_u8_kernel(const uint8_t* __restrict__ px8,
                                         __half* __restrict__ out,
                                         float m0, float m1, float m2,
                                         float s0, float s1, float s2,
                                         uint32_t g, uint32_t p, uint32_t px,
                                         uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // out[row, col]: row = (b*g + gy)*g + gx, col = c*p*p + ky*p + kx
    const uint32_t k = 3u * p * p;
    const uint64_t row = i / k;
    const uint32_t col = (uint32_t)(i - row * k);
    const uint32_t c = col / (p * p), rem = col % (p * p);
    const uint32_t ky = rem / p, kx = rem % p;
    const uint64_t b = row / ((uint64_t)g * g);
    const uint32_t r2 = (uint32_t)(row - b * (uint64_t)g * g);
    const uint32_t gy = r2 / g, gx = r2 % g;
    const uint64_t src =
        ((b * px + gy * p + ky) * px + (gx * p + kx)) * 3u + c;
    const float mean = c == 0u ? m0 : (c == 1u ? m1 : m2);
    const float sd = c == 0u ? s0 : (c == 1u ? s1 : s2);
    out[i] = __float2half(((float)px8[src] / 255.0f - mean) / sd);
}

PD_EXPORT
int pd_ocr_patches_u8(const void* pixels, void* out, float m0, float m1,
                      float m2, float s0, float s1, float s2, uint32_t views,
                      uint32_t px, uint32_t patch, void* stream) {
    if (views == 0) return 0;
    if (patch == 0 || px % patch != 0) return cudaErrorInvalidValue;
    const uint32_t g = px / patch;
    const uint64_t n = (uint64_t)views * px * px * 3u;
    const uint32_t threads = 256;
    const uint64_t blocks = (n + threads - 1) / threads;
    pd_ocr_patches_u8_kernel<<<(uint32_t)blocks, threads, 0,
                               (cudaStream_t)stream>>>(
        (const uint8_t*)pixels, (__half*)out, m0, m1, m2, s0, s1, s2, g,
        patch, px, n);
    return pd_launch_status();
}
