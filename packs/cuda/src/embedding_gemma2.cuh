// embedding_gemma2.cuh - EmbeddingGemma 2's text encoder glue (slots 806-812):
// its row norms, the sandwich seam, the q/k/v head transform, bidirectional
// varlen attention at head_dim 256 and 512, and the pooled output.
//
// The model: 24 layers of width 512, four query heads, five sliding layers
// (hd 256, two kv heads, |i - j| <= 512) for every full one (hd 512, one kv
// head), q/k RMS-normed with weights and v without, split-half rope on every
// dim (theta 1e4 sliding, 1e6 full), sandwich norms, GEGLU, a per-layer
// input (PLE) gate after the FFN and a learned scalar per layer; mean pooling
// over a 512 -> 768 projection. The GEMMs ride the mmq int8 class (Q8_0) and
// the bf16 tile (the PLE projection); everything else is here.
//
// BATCH INVARIANCE is the contract (as the Laya encoder's): a sequence's
// vector must not depend on what was packed with
// it. So every kernel below does a row's work in one fixed shape whatever the
// row count - the generic norms elect their block width and vector arm by
// rows, which is why the encoder has its own - and attention blocks own
// (sequence, query tile, kv head) with key tiles laid from the sequence
// start.
//
// The reference is llama.cpp's gemma-embedding2 graph on the same GGUF (read,
// not copied): its symmetric window masks |i - j| > n_swa / 2, the rope angle
// is pos * theta_scale^i with theta_scale = base^(-2 / n_dims) (host powf,
// passed in), attention scale 1, K/V rounded to f16.
//
// Needs decode_spec's pd_fa_mma16 / PD_FA_OK and f32_qkv's pd_launch_status.

#define PD_EG2_W 512u        // model width (also the PLE width and every norm row)
#define PD_EG2_LAYERS 24u
#define PD_EG2_KT 32u        // keys staged per attention tile

// Fixed-order block sum for a 128-thread row: butterfly within each warp,
// then the four warp totals in index order. Same bits for a row whether the
// launch carried one row or eight thousand.
__device__ __forceinline__ float pd_eg2_sum128(float v, float* red) {
#pragma unroll
    for (uint32_t o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const uint32_t tid = threadIdx.x;
    __syncthreads();   // red may still hold the previous reduction's values
    if ((tid & 31u) == 0) red[tid >> 5] = v;
    __syncthreads();
    return ((red[0] + red[1]) + red[2]) + red[3];
}

// The mmq activation quantize, fused into a row kernel's epilogue: thread t
// of a 128-thread row block holds columns 4t..4t+3, so warp w is 128-column
// chunk w and each 8-lane group one 32-value block - exactly the lanes
// pd_quantize_q8_mmq_kernel gives a block, with the same arithmetic (order-
// free max, scale = max / 127, round-to-nearest-even, clamp), so the bytes
// are identical to quantizing the f32 row afterwards. yq is the mmq layout:
// [chunk][column padded to 128][4 f32 scales + 128 int8].
__device__ __forceinline__ void pd_eg2_quant4(float4 v, uint8_t* __restrict__ yq,
                                              uint32_t row, uint32_t batch_pad) {
    const uint32_t t = threadIdx.x, lane = t & 31u, chunk = t >> 5;
    float a = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
#pragma unroll
    for (uint32_t s = 4; s > 0; s >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, s));
    const float scl = a * (1.0f / 127.0f);
    const float inv = scl > 0.0f ? 1.0f / scl : 0.0f;
    uint8_t* blk = yq + ((size_t)chunk * batch_pad + row) * 144u;
    int qi;
    char4 q;
    qi = __float2int_rn(v.x * inv); q.x = (char)(qi < -127 ? -127 : (qi > 127 ? 127 : qi));
    qi = __float2int_rn(v.y * inv); q.y = (char)(qi < -127 ? -127 : (qi > 127 ? 127 : qi));
    qi = __float2int_rn(v.z * inv); q.z = (char)(qi < -127 ? -127 : (qi > 127 ? 127 : qi));
    qi = __float2int_rn(v.w * inv); q.w = (char)(qi < -127 ? -127 : (qi > 127 ? 127 : qi));
    ((char4*)(blk + 16u))[lane] = q;
    if ((lane & 7u) == 0u) ((float*)blk)[lane >> 3] = scl;
}

// rmsnorm(x * scale) * w over 512-wide rows, one row a 128-thread block, into
// f32 `out` and/or the mmq-quantized `yq` (either may be null). PLE mode
// (ple_tokens > 0): the input is the PLE projection [tokens][24][512] and the
// output is laid LAYER-major [24][tokens][512], so each layer's slice is a
// plain [tokens][512] plane for the GEGLU quantize that consumes it.
__global__ void __launch_bounds__(128) pd_eg2_rms_kernel(
    const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ out,
    uint8_t* __restrict__ yq, uint32_t batch_pad, uint32_t ple_tokens, float scale, float eps) {
    __shared__ float red[4];
    const uint32_t r = blockIdx.x, t = threadIdx.x;
    float4 v = reinterpret_cast<const float4*>(x + (size_t)r * PD_EG2_W)[t];
    v.x *= scale; v.y *= scale; v.z *= scale; v.w *= scale;
    const float ss = pd_eg2_sum128(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w, red);
    const float inv = rsqrtf(ss / (float)PD_EG2_W + eps);
    const float4 g = reinterpret_cast<const float4*>(w)[t];
    const float4 o = make_float4(v.x * inv * g.x, v.y * inv * g.y, v.z * inv * g.z,
                                 v.w * inv * g.w);
    if (out != nullptr) {
        const size_t dst = ple_tokens
            ? ((size_t)(r % PD_EG2_LAYERS) * ple_tokens + r / PD_EG2_LAYERS) * PD_EG2_W
            : (size_t)r * PD_EG2_W;
        reinterpret_cast<float4*>(out + dst)[t] = o;
    }
    if (yq != nullptr) pd_eg2_quant4(o, yq, r, batch_pad);
}

// The sandwich seam, one row a block: x = (x + rmsnorm(proj) * wpost) * s, then
// (wnext set) xn = rmsnorm(x) * wnext - the post-norm of one block fused with
// the pre-norm of the next. s is the layer scalar after the PLE block, 1
// elsewhere (x * 1 is exact). `yq` (nullable) receives the next GEMM's
// quantized input: xn when wnext is set, else the updated x itself (the PLE
// gate reads the raw residual); `xn` (nullable) its f32 copy.
__global__ void __launch_bounds__(128) pd_eg2_sandwich_kernel(
    float* __restrict__ x, const float* __restrict__ proj, const float* __restrict__ wpost,
    const float* __restrict__ wnext, float* __restrict__ xn, uint8_t* __restrict__ yq,
    uint32_t batch_pad, float s, float eps) {
    __shared__ float red[4];
    const uint32_t r = blockIdx.x, t = threadIdx.x;
    const size_t off = (size_t)r * PD_EG2_W;
    const float4 p = reinterpret_cast<const float4*>(proj + off)[t];
    const float pss = pd_eg2_sum128(p.x * p.x + p.y * p.y + p.z * p.z + p.w * p.w, red);
    const float pinv = rsqrtf(pss / (float)PD_EG2_W + eps);
    const float4 g = reinterpret_cast<const float4*>(wpost)[t];
    float4 v = reinterpret_cast<const float4*>(x + off)[t];
    v.x = (p.x * pinv * g.x + v.x) * s;
    v.y = (p.y * pinv * g.y + v.y) * s;
    v.z = (p.z * pinv * g.z + v.z) * s;
    v.w = (p.w * pinv * g.w + v.w) * s;
    reinterpret_cast<float4*>(x + off)[t] = v;
    if (wnext == nullptr) {
        if (yq != nullptr) pd_eg2_quant4(v, yq, r, batch_pad);
        return;
    }
    const float ss = pd_eg2_sum128(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w, red);
    const float inv = rsqrtf(ss / (float)PD_EG2_W + eps);
    const float4 h = reinterpret_cast<const float4*>(wnext)[t];
    const float4 n = make_float4(v.x * inv * h.x, v.y * inv * h.y, v.z * inv * h.z, v.w * inv * h.w);
    if (xn != nullptr) reinterpret_cast<float4*>(xn + off)[t] = n;
    if (yq != nullptr) pd_eg2_quant4(n, yq, r, batch_pad);
}

// GEGLU into the mmq layout with strided inputs: v = gelu_tanh(gate) * up per
// element (the pd_quantize_q8_mmq_geglu_kernel formula and quantize, verbatim)
// where gate and up are `in_dim`-wide rows at row stride `ld` - so a fused
// gate|up GEMM landing ([rows][2 ff], up at column ff) is read in place, and
// so is the PLE gate against its layer-major plane (ld = in_dim).
__global__ void pd_eg2_geglu_q_kernel(const float* __restrict__ gate,
                                      const float* __restrict__ up, uint32_t ld,
                                      uint8_t* __restrict__ yq, uint32_t in_dim,
                                      uint32_t batch) {
    const uint32_t chunk = blockIdx.x, col = blockIdx.y, lane = threadIdx.x;
    uint8_t* blk = yq + ((size_t)chunk * gridDim.y + col) * 144u;
    const uint32_t k0 = chunk * 128u + lane * 4u;
    float v[4] = {};
    if (col < batch) {
#pragma unroll
        for (uint32_t j = 0; j < 4u; ++j)
            if (k0 + j < in_dim) {
                const float g = gate[(size_t)col * ld + k0 + j];
                const float u = up[(size_t)col * ld + k0 + j];
                const float gelu = 0.5f * g
                    * (1.0f + tanhf(0.79788456080286535587989211986876f * g
                                    * (1.0f + 0.044715f * g * g)));
                v[j] = gelu * u;
            }
    }
    float a = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
#pragma unroll
    for (uint32_t s = 4; s > 0; s >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, s));
    const float scl = a * (1.0f / 127.0f);
    const float inv = scl > 0.0f ? 1.0f / scl : 0.0f;
    char4 q;
    int qi;
    qi = __float2int_rn(v[0] * inv); q.x = (char)(qi < -127 ? -127 : (qi > 127 ? 127 : qi));
    qi = __float2int_rn(v[1] * inv); q.y = (char)(qi < -127 ? -127 : (qi > 127 ? 127 : qi));
    qi = __float2int_rn(v[2] * inv); q.z = (char)(qi < -127 ? -127 : (qi > 127 ? 127 : qi));
    qi = __float2int_rn(v[3] * inv); q.w = (char)(qi < -127 ? -127 : (qi > 127 ? 127 : qi));
    ((char4*)(blk + 16u))[lane] = q;
    if ((lane & 7u) == 0u) ((float*)blk)[lane >> 3] = scl;
}

// The small-pass twin of the mmq tile (pd_q8_0_gemm_mmq_kernel) over the
// same repacked Q8_0 rows and quantized activations. At a few dozen rows the
// tile's grid is out/128 x 1 blocks - 4 to 32 on a 48-SM die, each walking K
// alone - so the encoder's short passes were GEMM-latency bound. This is the
// GEMV shape instead: one block an SM per 32-token group, the group's
// activations staged ONCE for the whole K (<= 74 KB at K 2048), then every
// warp streams whole weight rows - a row's bytes are one coalesced load a
// lane, the next row's already in registers while this one computes from
// the warp's smem slot - and a lane is a token. DRAM sees the weights once.
// Per output it is the tile's arithmetic step for step: Q8_0 block b's exact
// int32 dot (dp4a here, int8 mma there - both exact), then acc += dW_b *
// dX_b * (float)dot in block order. So an output is bit-identical whichever
// kernel computed it, and a pass may switch kernels by row count without a
// sequence's vector depending on what it was packed with.
// Measured on the way (GB10, an 18-row pass): a warp a weight row reading
// K a block at a time from global, 14 us a launch (a DRAM round trip a
// step); 32-row x 32-token smem tiles, 13 us at 512-wide outputs (16
// blocks); 8-row tiles on a cp.async ring, 10-28 us (every block restaged
// 16 KB of activations a round against 4 KB of weights).
#define PD_EG2_GR_MAXQ 4u   // int4 of a weight row a lane holds (K <= 2048)

__device__ __forceinline__ void pd_eg2_cpa16(void* dst, const void* src) {
    const uint32_t sm = (uint32_t)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(sm), "l"(src));
}

__host__ __device__ constexpr uint32_t pd_eg2_gr_smem(uint32_t nb) {
    return 32u * (2u * nb + 1u) * 16u   // activations, [32 tokens][2 nb + 1 pad] int4
           + 32u * nb * 4u              // their scales
           + 8u * 2u * nb * 16u         // a weight row a warp
           + 8u * nb * 2u;              // its f16 scales
}

__global__ void __launch_bounds__(256, 1) pd_eg2_gemm_rows_kernel(
    const int8_t* __restrict__ data, const __half* __restrict__ scale,
    const uint8_t* __restrict__ yq, float* __restrict__ y, uint32_t in_dim, uint32_t out_dim,
    uint32_t batch) {
    extern __shared__ int4 eg2gr[];
    const uint32_t nb = in_dim >> 5, xs = 2u * nb + 1u;
    int4* sx = eg2gr;
    float* sdx = reinterpret_cast<float*>(sx + 32u * xs);
    int4* sw = reinterpret_cast<int4*>(sdx + 32u * nb);
    __half* sdw = reinterpret_cast<__half*>(sw + 8u * 2u * nb);
    const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
    const uint32_t c0 = blockIdx.y * 32u;
    const uint32_t tv = min(32u, batch - c0);   // live tokens in this group
    const uint32_t batch_pad = (batch + 127u) & ~127u;

    // this group's activations for all of K, and their scales
    for (uint32_t i = tid; i < tv * 2u * nb; i += 256u) {
        const uint32_t t = i / (2u * nb), j = i % (2u * nb), gb = j >> 1;
        pd_eg2_cpa16(&sx[t * xs + j], yq + ((size_t)(gb >> 2) * batch_pad + c0 + t) * 144u
                                          + 16u + (gb & 3u) * 32u + (j & 1u) * 16u);
    }
    for (uint32_t i = tid; i < tv * (nb >> 2); i += 256u) {
        const uint32_t t = i / (nb >> 2), q = i % (nb >> 2);
        pd_eg2_cpa16(&sdx[t * nb + q * 4u], yq + ((size_t)q * batch_pad + c0 + t) * 144u);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");

    // a lane's share of a weight row: int4 lane + 32 k, and (lanes < nb / 8)
    // one int4 of its f16 scales
    const uint32_t nq = (2u * nb) >> 5;   // int4 a lane holds (K 512: 1, K 2048: 4)
    const uint32_t stride = gridDim.x * 8u;
    uint32_t r = blockIdx.x * 8u + warp;
    int4 wr[PD_EG2_GR_MAXQ], ws = make_int4(0, 0, 0, 0);
    auto fetch = [&](uint32_t rr) {
        const int4* src = reinterpret_cast<const int4*>(data + (size_t)rr * (size_t)in_dim);
#pragma unroll
        for (uint32_t k = 0; k < PD_EG2_GR_MAXQ; ++k)
            if (k < nq) wr[k] = __ldg(src + lane + 32u * k);
        if (lane < (nb >> 3)) ws = __ldg(reinterpret_cast<const int4*>(scale + (size_t)rr * nb) + lane);
    };
    if (r < out_dim) fetch(r);
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();

    int4* mw = sw + warp * 2u * nb;
    __half* mdw = sdw + warp * nb;
    while (r < out_dim) {
#pragma unroll
        for (uint32_t k = 0; k < PD_EG2_GR_MAXQ; ++k)
            if (k < nq) mw[lane + 32u * k] = wr[k];
        if (lane < (nb >> 3)) reinterpret_cast<int4*>(mdw)[lane] = ws;
        __syncwarp();
        const uint32_t rn = r + stride;
        if (rn < out_dim) fetch(rn);   // in flight while this row computes
        if (lane < tv) {
            float acc = 0.f;
            for (uint32_t b = 0; b < nb; ++b) {
                const int4 w0 = mw[2u * b], w1 = mw[2u * b + 1u];
                const int4 x0 = sx[lane * xs + 2u * b], x1 = sx[lane * xs + 2u * b + 1u];
                int d = 0;
                d = __dp4a(w0.x, x0.x, d);
                d = __dp4a(w0.y, x0.y, d);
                d = __dp4a(w0.z, x0.z, d);
                d = __dp4a(w0.w, x0.w, d);
                d = __dp4a(w1.x, x1.x, d);
                d = __dp4a(w1.y, x1.y, d);
                d = __dp4a(w1.z, x1.z, d);
                d = __dp4a(w1.w, x1.w, d);
                const float dA = __half2float(mdw[b]);
                acc += dA * sdx[lane * nb + b] * (float)d;
            }
            y[(size_t)(c0 + lane) * out_dim + r] = acc;
        }
        __syncwarp();   // the slot is read before the next row lands in it
        r = rn;
    }
}

// The head transform off the fused q|k|v GEMM landing ([rows][stride] f32, q
// at 0, k at 4 * hd, v at 4 * hd + 512): per (row, head) RMS norm (q and k with
// their weights, v without), split-half rope on q and k at the row's position
// in its sequence, rounded once to f16 into [rows][heads][hd] planes. One head
// row a block of hd / 2 threads; thread t owns the rope pair (t, t + hd / 2).
template <uint32_t HD>
__global__ void __launch_bounds__(HD / 2u) pd_eg2_heads_kernel(
    const float* __restrict__ qkv, const uint32_t* __restrict__ pos,
    const float* __restrict__ qw, const float* __restrict__ kw, __half* __restrict__ q16,
    __half* __restrict__ k16, __half* __restrict__ v16, uint32_t stride, float theta_scale,
    float eps) {
    constexpr uint32_t KH = 512u / HD, NW = HD / 64u;   // kv heads; warps a block
    __shared__ float red[NW];
    const uint32_t r = blockIdx.x, hi = blockIdx.y, t = threadIdx.x;
    // hi: [0, 4) q heads, [4, 4 + KH) k heads, [4 + KH, 4 + 2 KH) v heads
    const bool isq = hi < 4u, isk = !isq && hi < 4u + KH;
    const uint32_t col = isq ? hi * HD : (isk ? 4u * HD + (hi - 4u) * HD
                                              : 4u * HD + 512u + (hi - 4u - KH) * HD);
    const float* src = qkv + (size_t)r * stride + col;
    float a = src[t], b = src[t + HD / 2u];
    float v = a * a + b * b;
#pragma unroll
    for (uint32_t o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if ((t & 31u) == 0) red[t >> 5] = v;
    __syncthreads();
    float ss = 0.f;
#pragma unroll
    for (uint32_t i = 0; i < NW; ++i) ss += red[i];
    const float inv = rsqrtf(ss / (float)HD + eps);
    a *= inv;
    b *= inv;
    __half* dst;
    if (isq || isk) {
        const float* w = isq ? qw : kw;
        a *= w[t];
        b *= w[t + HD / 2u];
        // llama.cpp's neox angle: pos * theta_scale^i, cos/sin of it (rope_yarn
        // at ext_factor 0, attn_factor 1)
        const float theta = (float)pos[r] * powf(theta_scale, (float)t);
        float sn, cs;
        sincosf(theta, &sn, &cs);
        const float ra = a * cs - b * sn;
        b = a * sn + b * cs;
        a = ra;
        dst = isq ? q16 + ((size_t)r * 4u + hi) * HD
                  : k16 + ((size_t)r * KH + (hi - 4u)) * HD;
    } else {
        dst = v16 + ((size_t)r * KH + (hi - 4u - KH)) * HD;
    }
    dst[t] = __float2half_rn(a);
    dst[t + HD / 2u] = __float2half_rn(b);
}

// ---- attention ----
//
// Bidirectional attention over PACKED sequences (cu[s] .. cu[s + 1]), one
// block a (query tile, kv head): the kv head's G = 4 / KH query heads share
// every staged K/V tile. A warp owns (16-row strip, query head, 256-wide D
// slice): at hd 256 that is the varlen kernel's register posture (Q 64 regs,
// O 128); at hd 512 each head splits across two D-half warps (gemma4's v3-512
// answer to the 255-register cap) whose partial S meet through one smem pane
// a warp - both halves then hold the same summed S, run the same softmax and
// multiply their own V half. Blocks are 8 warps either way: 4 strips x 2
// heads (64 rows) at hd 256, 1 strip x 4 heads x 2 halves (16 rows) at 512.
//
// Masks: keys outside the sequence, and (window > 0) |i - j| > window. A
// tile's key span is [q0 - w, q_last + w] clipped to the sequence. Numeric
// class: f16 q/k/v/P operands, f32 S/O and online softmax (FTZ at e^-20) -
// the varlen kernel's.
template <uint32_t HD>
struct PdEg2Geo {
    static constexpr uint32_t DS = HD / 256u;              // D slices a head
    static constexpr uint32_t G = HD / 128u;               // q heads a kv head
    static constexpr uint32_t KH = 512u / HD;              // kv heads
    static constexpr uint32_t STRIPS = 8u / (G * DS);      // 16-row strips a block
    static constexpr uint32_t ROWS = STRIPS * 16u;         // query rows a block
    static constexpr uint32_t DPD = HD + 8u;               // padded smem row
    static constexpr uint32_t SMEM = 2u * PD_EG2_KT * DPD * 2u
                                     + (DS > 1u ? 8u * 16u * PD_EG2_KT * 4u : 0u);
};

#define PD_EG2_TILE_SHIFT 12u   // a tile descriptor is (sequence << 12) | tile index

template <uint32_t HD>
__global__ void __launch_bounds__(256, 1) pd_eg2_attn_kernel(
    const __half* __restrict__ q16, const __half* __restrict__ k16,
    const __half* __restrict__ v16, const uint32_t* __restrict__ cu,
    const uint32_t* __restrict__ tiles, float* __restrict__ out, uint32_t window) {
#if PD_FA_OK
    using Geo = PdEg2Geo<HD>;
    constexpr uint32_t KT = PD_EG2_KT, DPD = Geo::DPD, DS = Geo::DS, G = Geo::G;
    constexpr uint32_t KH = Geo::KH;
    const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
    const uint32_t g8 = lane >> 2, t4 = lane & 3u, lg = lane >> 3;
    const uint32_t kvh = blockIdx.y;
    const uint32_t td = tiles[blockIdx.x];
    const uint32_t sq = td >> PD_EG2_TILE_SHIFT;
    const uint32_t q0 = (td & ((1u << PD_EG2_TILE_SHIFT) - 1u)) * Geo::ROWS;
    const uint32_t base = cu[sq], L = cu[sq + 1u] - base;
    const uint32_t strip = warp / (G * DS), gi = (warp / DS) % G, ds = warp % DS;
    const uint32_t h = kvh * G + gi;                // this warp's query head
    const uint32_t dcol = ds * 256u;                // this warp's D slice
    const uint32_t wq0 = q0 + strip * 16u;

    extern __shared__ unsigned char eg2sh[];
    __half* sh_k = reinterpret_cast<__half*>(eg2sh);
    __half* sh_v = sh_k + (size_t)KT * DPD;
    float* pane = reinterpret_cast<float*>(sh_v + (size_t)KT * DPD);

    // ---- q fragments for 16 rows x this 256-wide slice, straight from f16
    const uint32_t jr[2] = {g8, g8 + 8u};
    uint32_t qa[16][4];
#pragma unroll
    for (uint32_t d0 = 0; d0 < 16u; ++d0) {
#pragma unroll
        for (uint32_t e = 0; e < 2u; ++e) {
            const uint32_t qi = wq0 + jr[e];
            const bool ok = qi < L;
            const __half* qp = q16 + ((size_t)(base + (ok ? qi : 0u)) * 4u + h) * HD + dcol
                               + d0 * 16u + 2u * t4;
            qa[d0][e] = ok ? *reinterpret_cast<const uint32_t*>(qp) : 0u;
            qa[d0][e + 2u] = ok ? *reinterpret_cast<const uint32_t*>(qp + 8u) : 0u;
        }
    }

    const uint32_t qlast = (q0 + Geo::ROWS < L ? q0 + Geo::ROWS : L) - 1u;
    const uint32_t lo = window ? (q0 > window ? q0 - window : 0u) : 0u;
    const uint32_t hi = window ? (qlast + window + 1u < L ? qlast + window + 1u : L) : L;

    float m_st[2] = {-1e30f, -1e30f}, l_st[2] = {0.f, 0.f};
    float o_acc[32][4];
#pragma unroll
    for (uint32_t nt = 0; nt < 32u; ++nt)
#pragma unroll
        for (uint32_t e = 0; e < 4u; ++e) o_acc[nt][e] = 0.f;

    for (uint32_t t0 = lo; t0 < hi; t0 += KT) {
        // stage K and V (rows of this kv head), zero past the span
        constexpr uint32_t U = KT * (HD / 8u);
        for (uint32_t u = tid; u < U; u += 256u) {
            const uint32_t kk = u / (HD / 8u), d8 = (u % (HD / 8u)) * 8u;
            const uint32_t ks = t0 + kk;
            uint4 kv = make_uint4(0u, 0u, 0u, 0u), vv = kv;
            if (ks < hi) {
                const size_t at = ((size_t)(base + ks) * KH + kvh) * HD + d8;
                kv = *reinterpret_cast<const uint4*>(k16 + at);
                vv = *reinterpret_cast<const uint4*>(v16 + at);
            }
            *reinterpret_cast<uint4*>(sh_k + (size_t)kk * DPD + d8) = kv;
            *reinterpret_cast<uint4*>(sh_v + (size_t)kk * DPD + d8) = vv;
        }
        __syncthreads();

        float s_acc[KT / 8u][4];
#pragma unroll
        for (uint32_t nt = 0; nt < KT / 8u; ++nt)
#pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) s_acc[nt][e] = 0.f;
#pragma unroll
        for (uint32_t d0 = 0; d0 < 16u; ++d0) {
#pragma unroll
            for (uint32_t np = 0; np < KT / 16u; ++np) {
                const __half* kp = sh_k
                    + (size_t)(np * 16u + (lg >> 1) * 8u + (lane & 7u)) * DPD
                    + dcol + d0 * 16u + (lg & 1u) * 8u;
                uint32_t kb4[4];
                const uint32_t ka = (uint32_t)__cvta_generic_to_shared(kp);
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                    : "=r"(kb4[0]), "=r"(kb4[1]), "=r"(kb4[2]), "=r"(kb4[3]) : "r"(ka));
                pd_fa_mma16(s_acc[np * 2u], qa[d0][0], qa[d0][1], qa[d0][2], qa[d0][3],
                            kb4[0], kb4[1]);
                pd_fa_mma16(s_acc[np * 2u + 1u], qa[d0][0], qa[d0][1], qa[d0][2],
                            qa[d0][3], kb4[2], kb4[3]);
            }
        }
        if (DS > 1u) {
            // the two D halves of a head swap partial S; a + b == b + a, so
            // both hold bit-identical sums
            float* mine = pane + (size_t)warp * 16u * KT;
            const float* other = pane + (size_t)(warp ^ 1u) * 16u * KT;
#pragma unroll
            for (uint32_t nt = 0; nt < KT / 8u; ++nt)
#pragma unroll
                for (uint32_t e = 0; e < 4u; ++e) mine[(nt * 4u + e) * 32u + lane] = s_acc[nt][e];
            __syncthreads();
#pragma unroll
            for (uint32_t nt = 0; nt < KT / 8u; ++nt)
#pragma unroll
                for (uint32_t e = 0; e < 4u; ++e)
                    s_acc[nt][e] += other[(nt * 4u + e) * 32u + lane];
        }
        // mask: past the span, and (sliding layers) outside the row's window
        float mn[2] = {m_st[0], m_st[1]};
#pragma unroll
        for (uint32_t nt = 0; nt < KT / 8u; ++nt) {
            const uint32_t kb = t0 + nt * 8u + 2u * t4;
#pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) {
                const uint32_t kj = kb + (e & 1u);
                const uint32_t qi = wq0 + jr[e >> 1];
                const bool in_win = window == 0u || (kj + window >= qi && kj <= qi + window);
                if (kj >= hi || !in_win) s_acc[nt][e] = -1e30f;
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
                const float dd = s_acc[nt][e] - mn[e >> 1];
                const float w = dd >= -20.f ? __expf(dd) : 0.f;
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
        for (uint32_t nt = 0; nt < 32u; ++nt) {
            o_acc[nt][0] *= corr[0];
            o_acc[nt][1] *= corr[0];
            o_acc[nt][2] *= corr[1];
            o_acc[nt][3] *= corr[1];
        }
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
            const __half* vp = sh_v + (size_t)vr * DPD + dcol + (lg >> 1) * 8u;
#pragma unroll
            for (uint32_t nt = 0; nt < 32u; nt += 2u) {
                uint32_t vb4[4];
                const uint32_t va = (uint32_t)__cvta_generic_to_shared(vp + nt * 8u);
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                    : "=r"(vb4[0]), "=r"(vb4[1]), "=r"(vb4[2]), "=r"(vb4[3]) : "r"(va));
                pd_fa_mma16(o_acc[nt], pa0, pa1, pa2, pa3, vb4[0], vb4[1]);
                pd_fa_mma16(o_acc[nt + 1u], pa0, pa1, pa2, pa3, vb4[2], vb4[3]);
            }
        }
        __syncthreads();   // tiles and panes read before the next stage overwrites them
    }

    const float nrm[2] = {l_st[0] > 0.f ? 1.f / l_st[0] : 0.f,
                          l_st[1] > 0.f ? 1.f / l_st[1] : 0.f};
#pragma unroll
    for (uint32_t nt = 0; nt < 32u; ++nt) {
        const uint32_t c = dcol + nt * 8u + 2u * t4;
#pragma unroll
        for (uint32_t r = 0; r < 2u; ++r) {
            const uint32_t qi = wq0 + jr[r];
            if (qi >= L) continue;
            float* op = out + ((size_t)(base + qi) * 4u + h) * HD + c;
            *reinterpret_cast<float2*>(op) =
                make_float2(o_acc[nt][2u * r] * nrm[r], o_acc[nt][2u * r + 1u] * nrm[r]);
        }
    }
#else
    (void)q16; (void)k16; (void)v16; (void)cu; (void)tiles; (void)out; (void)window;
#endif
}

// Mean pooling per sequence over its token rows ([rows][768] f32), then L2
// normalization over the first `dims` components - Matryoshka truncation is
// the prefix, re-normalized. One sequence a block; the token sum runs in row
// order so a sequence's vector is independent of its neighbours.
__global__ void __launch_bounds__(256) pd_eg2_pool_kernel(
    const float* __restrict__ tok, const uint32_t* __restrict__ cu, float* __restrict__ out,
    uint32_t dims) {
    __shared__ float red[8];
    const uint32_t sq = blockIdx.x, tid = threadIdx.x;
    const uint32_t first = cu[sq], count = cu[sq + 1u] - first;
    float vals[3];
    float ss = 0.f;
#pragma unroll
    for (uint32_t i = 0; i < 3u; ++i) {
        const uint32_t j = tid + i * 256u;
        float v = 0.f;
        if (j < dims) {
            for (uint32_t t = 0; t < count; ++t) v += tok[(size_t)(first + t) * 768u + j];
            v /= (float)count;
        }
        vals[i] = v;
        ss += v * v;
    }
#pragma unroll
    for (uint32_t o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if ((tid & 31u) == 0) red[tid >> 5] = ss;
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for (uint32_t i = 0; i < 8u; ++i) tot += red[i];
    const float inv = 1.f / fmaxf(sqrtf(tot), 1e-12f);
#pragma unroll
    for (uint32_t i = 0; i < 3u; ++i) {
        const uint32_t j = tid + i * 256u;
        if (j < dims) out[(size_t)sq * dims + j] = vals[i] * inv;
    }
}

// 806: rmsnorm(x * scale) * w over [n_rows][512] f32 into `out` and/or the
// mmq-quantized `yq` (batch_pad = n_rows padded to 128; either output may be
// null); ple_tokens > 0 writes the PLE planes layer-major (n_rows = 24 *
// ple_tokens, f32 only).
PD_EXPORT
int pd_eg2_rms(const void* x, const void* w, void* out, void* yq, uint32_t n_rows,
               uint32_t ple_tokens, float scale, float eps, void* stream) {
    if (n_rows == 0) return 0;
    if (ple_tokens && (n_rows != ple_tokens * PD_EG2_LAYERS || yq != nullptr))
        return cudaErrorInvalidValue;
    if (out == nullptr && yq == nullptr) return cudaErrorInvalidValue;
    pd_eg2_rms_kernel<<<n_rows, 128, 0, (cudaStream_t)stream>>>(
        (const float*)x, (const float*)w, (float*)out, (uint8_t*)yq, (n_rows + 127u) & ~127u,
        ple_tokens, scale, eps);
    return pd_launch_status();
}

// 807: x = (x + rmsnorm(proj) * wpost) * s over [rows][512]; wnext non-null
// also forms xn = rmsnorm(x) * wnext. `xn` (f32, needs wnext) and `yq` (the
// mmq layout of xn, or of x when wnext is null) are each optional.
PD_EXPORT
int pd_eg2_sandwich(void* x, const void* proj, const void* wpost, const void* wnext, void* xn,
                    void* yq, uint32_t rows, float s, float eps, void* stream) {
    if (rows == 0) return 0;
    if (xn != nullptr && wnext == nullptr) return cudaErrorInvalidValue;
    pd_eg2_sandwich_kernel<<<rows, 128, 0, (cudaStream_t)stream>>>(
        (float*)x, (const float*)proj, (const float*)wpost, (const float*)wnext, (float*)xn,
        (uint8_t*)yq, (rows + 127u) & ~127u, s, eps);
    return pd_launch_status();
}

// 808: the q/k/v head transform off the fused landing [rows][stride] f32 (q at
// 0, k at 4 * hd, v at 4 * hd + 512) into f16 q [rows][4][hd], k / v
// [rows][512 / hd][hd]; pos [rows] u32 positions within each sequence.
// head_dim 256 or 512.
PD_EXPORT
int pd_eg2_heads(const void* qkv, const void* pos, const void* qw, const void* kw, void* q16,
                 void* k16, void* v16, uint32_t rows, uint32_t stride, uint32_t head_dim,
                 float theta_scale, float eps, void* stream) {
    if (rows == 0) return 0;
    if (stride < 4u * head_dim + 1024u) return cudaErrorInvalidValue;
    cudaStream_t st = (cudaStream_t)stream;
    const float* x = (const float*)qkv;
    const uint32_t* p = (const uint32_t*)pos;
    if (head_dim == 256u) {
        pd_eg2_heads_kernel<256u><<<dim3(rows, 8u), 128u, 0, st>>>(
            x, p, (const float*)qw, (const float*)kw, (__half*)q16, (__half*)k16, (__half*)v16,
            stride, theta_scale, eps);
    } else if (head_dim == 512u) {
        pd_eg2_heads_kernel<512u><<<dim3(rows, 6u), 256u, 0, st>>>(
            x, p, (const float*)qw, (const float*)kw, (__half*)q16, (__half*)k16, (__half*)v16,
            stride, theta_scale, eps);
    } else {
        return cudaErrorInvalidValue;
    }
    return pd_launch_status();
}

// 809: bidirectional varlen attention, f32 out [rows][4][hd]. cu [n_seq + 1]
// row offsets; tiles [n_tiles] (seq << 12) | tile, one per 64 query rows at
// hd 256 and per 16 at hd 512; window 0 = full, else |i - j| <= window.
// sm_80+.
PD_EXPORT
int pd_eg2_attn(const void* q16, const void* k16, const void* v16, const void* cu,
                const void* tiles, uint32_t n_tiles, void* out, uint32_t head_dim,
                uint32_t window, void* stream) {
    if (n_tiles == 0) return 0;
    int dev = 0, cc = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&cc, cudaDevAttrComputeCapabilityMajor, dev);
    if (cc < 8) return cudaErrorInvalidValue;
    cudaStream_t st = (cudaStream_t)stream;
    static cudaError_t a256 = cudaFuncSetAttribute(
        pd_eg2_attn_kernel<256u>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)PdEg2Geo<256u>::SMEM);
    static cudaError_t a512 = cudaFuncSetAttribute(
        pd_eg2_attn_kernel<512u>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)PdEg2Geo<512u>::SMEM);
    const __half* q = (const __half*)q16;
    const __half* k = (const __half*)k16;
    const __half* v = (const __half*)v16;
    const uint32_t* c = (const uint32_t*)cu;
    const uint32_t* t = (const uint32_t*)tiles;
    if (head_dim == 256u) {
        if (a256 != cudaSuccess) return a256;
        pd_eg2_attn_kernel<256u><<<dim3(n_tiles, 2u), 256u, PdEg2Geo<256u>::SMEM, st>>>(
            q, k, v, c, t, (float*)out, window);
    } else if (head_dim == 512u) {
        if (a512 != cudaSuccess) return a512;
        pd_eg2_attn_kernel<512u><<<dim3(n_tiles, 1u), 256u, PdEg2Geo<512u>::SMEM, st>>>(
            q, k, v, c, t, (float*)out, window);
    } else {
        return cudaErrorInvalidValue;
    }
    return pd_launch_status();
}

// 810: mean-pool + L2-normalize [rows][768] token outputs into [n_seq][dims],
// dims in {128, 256, 512, 768}.
PD_EXPORT
int pd_eg2_pool(const void* tok, const void* cu, uint32_t n_seq, uint32_t dims, void* out,
                void* stream) {
    if (n_seq == 0) return 0;
    if (dims != 128u && dims != 256u && dims != 512u && dims != 768u)
        return cudaErrorInvalidValue;
    pd_eg2_pool_kernel<<<n_seq, 256, 0, (cudaStream_t)stream>>>(
        (const float*)tok, (const uint32_t*)cu, (float*)out, dims);
    return pd_launch_status();
}

// 811: the small-pass Q8_0 GEMM - y [batch][out] = W . X over the repacked
// rows (data int8 [out][in], scale f16 [out][in / 32]) and mmq-quantized X,
// bit-identical per output to pd_q8_0_gemm_mmq's plain tile (see the kernel).
// in_dim a multiple of 512, at most 2048.
PD_EXPORT
int pd_eg2_gemm_rows(const void* data, const void* scale, const void* yq, void* y,
                     uint32_t in_dim, uint32_t out_dim, uint32_t batch, void* stream) {
    if (batch == 0 || out_dim == 0) return 0;
    // a weight row is a whole int4 a lane (K 512 each), at most PD_EG2_GR_MAXQ
    if (in_dim == 0 || (in_dim & 511u) || in_dim > 2048u) return cudaErrorInvalidValue;
    static int sms = 0;
    if (sms == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        if (sms <= 0) sms = 48;
    }
    static cudaError_t attr = cudaFuncSetAttribute(
        pd_eg2_gemm_rows_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)pd_eg2_gr_smem(64u));
    if (attr != cudaSuccess) return attr;
    const uint32_t groups = (out_dim + 7u) / 8u;
    dim3 grid(groups < (uint32_t)sms ? groups : (uint32_t)sms, (batch + 31u) / 32u);
    pd_eg2_gemm_rows_kernel<<<grid, 256, pd_eg2_gr_smem(in_dim >> 5), (cudaStream_t)stream>>>(
        (const int8_t*)data, (const __half*)scale, (const uint8_t*)yq, (float*)y, in_dim,
        out_dim, batch);
    return pd_launch_status();
}

// 812: GEGLU of strided gate / up rows (row stride ld) straight into the mmq
// layout - pd_quantize_q8_mmq_geglu with a stride.
PD_EXPORT
int pd_eg2_geglu_q(const void* gate, const void* up, uint32_t ld, void* yq, uint32_t in_dim,
                   uint32_t batch, void* stream) {
    if (in_dim == 0 || batch == 0) return 0;
    const uint32_t n_chunks = (in_dim + 127u) / 128u;
    dim3 grid(n_chunks, (batch + 127u) & ~127u);
    pd_eg2_geglu_q_kernel<<<grid, 32, 0, (cudaStream_t)stream>>>(
        (const float*)gate, (const float*)up, ld, (uint8_t*)yq, in_dim, batch);
    return pd_launch_status();
}

