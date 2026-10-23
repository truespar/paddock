// nemotron_embed.cuh - Nemotron 3.5 Embed 8B / Rerank 8B's own kernels
// (slots 896-902; the gated-SiLU GEMM landing is slot 895 in f16_dense.cuh).
//
// The model: a Pixtral tower (24 blocks, 1024 wide, 16 x 64 heads, 2D rope,
// gated-SiLU FFN, RMSNorm) and Mistral 3's projector feed a bidirectional
// Ministral 3 8B body (34 layers, 4096 wide, GQA 32 / 8 at hd 128, YaRN rope
// with Llama 4's query scaling, gated-SiLU FFN), mean-pooled over every
// token: the embedder L2-normalizes the mean, the reranker dots it with one
// score row. Every linear rides the f16 mma rings (f16 operands, f32
// accumulate - the bf16 checkpoint widened to f16 exactly); what is here is
// the glue around them:
//
// 896 pd_ne_patch_u8 - the processor's pixel math on the device: RGB8 ->
//     (u - mean') / std' (the Torchvision backend's fused rescale: mean and std
//     pre-multiplied by 255, one subtract and one divide in f32) -> the 14 x 14
//     patch flattened (c, ky, kx), one f16 row of 592 a patch (588 + four zero
//     columns: the GEMM stages 16-byte units).
// 897 pd_ne_rms_h - RMSNorm into the f16 operand of the next GEMM, the
//     reference's f32 order (x * rsqrt(mean(x^2) + eps), then the weight).
// 898 pd_ne_embed_rows - the body's input rows in one gather: a token id
//     widens its bf16 embedding row, a picture placeholder takes its projector
//     row (index | 0x80000000).
// 899 pd_ne_rope_qk - rotate-half rope on q and k IN PLACE in the fused
//     q|k|v landing, from (cos, sin) tables indexed by a per-row position, q
//     then scaled (1/sqrt(hd), log2(e) for the attention's exp2, and Llama 4's
//     per-position factor). The body reads the YaRN table at the token's
//     position in its sequence; the tower reads a per-pass 2D table at the
//     patch's own row.
// 900 pd_ne_attn - bidirectional attention over PACKED sequences straight off
//     the landing, one block a (query tile, kv head): the kv head's G query
//     heads share every staged K/V tile (EmbeddingGemma 2's layout, slot 809).
// 901 pd_ne_pool - per sequence: the mean over its rows (f64, row order),
//     then the unit vector over the first `dims` components or the score dot.
// 902 pd_ne_add_rms_h - the residual add folded into the next norm (the
//     projections land plain f32).
//
// BATCH INVARIANCE is the contract: a sequence's vector must not depend on
// what was packed with it. The row kernels do one row in one fixed shape
// whatever the row count, the rings never split K, attention blocks own
// (sequence, query tile, kv head) with key tiles laid from the sequence start,
// and the pool reads a sequence's rows in order in one block.
//
// Needs decode_spec's pd_fa_mma16 / PD_FA_OK and f32_qkv's pd_launch_status.

#define PD_NE_PATCH_K 592u      // 3 x 14 x 14 = 588, padded to a 16-byte multiple
#define PD_NE_KT 64u            // keys staged per attention tile
#define PD_NE_TILE_SHIFT 12u    // a tile descriptor is (sequence << 12) | tile index
#define PD_NE_EMB_PIC 0x80000000u

// 16 bytes global -> shared, zero-filled (no global read) when !ok
__device__ __forceinline__ void pd_ne_cp16(void* dst, const void* src, bool ok) {
    const uint32_t d = (uint32_t)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(d), "l"(src),
                 "r"(ok ? 16 : 0));
}

// ---- 896 ---------------------------------------------------------------

__global__ void __launch_bounds__(256) pd_ne_patch_u8_kernel(
    const uint8_t* __restrict__ canvas, __half* __restrict__ out, uint32_t w, float m0, float m1,
    float m2, float s0, float s1, float s2) {
    const uint32_t cols = w / 14u;
    const uint32_t p = blockIdx.x;
    const uint32_t pr = p / cols, pc = p - pr * cols;
    const float mean[3] = {m0, m1, m2};
    const float sd[3] = {s0, s1, s2};
    __half* orow = out + (size_t)p * PD_NE_PATCH_K;
    for (uint32_t i = threadIdx.x; i < PD_NE_PATCH_K; i += blockDim.x) {
        float v = 0.0f;
        if (i < 588u) {
            const uint32_t c = i / 196u, r = i - c * 196u;
            const uint32_t ky = r / 14u, kx = r - ky * 14u;
            const uint32_t y = pr * 14u + ky, x = pc * 14u + kx;
            const float u = (float)canvas[((size_t)y * w + x) * 3u + c];
            // torchvision's normalize: sub_ then div_, each its own round
            v = __fdiv_rn(__fsub_rn(u, mean[c]), sd[c]);
        }
        orow[i] = __float2half_rn(v);
    }
}

// 896: an RGB8 h x w canvas (both multiples of 14) -> one f16 row of 592 per
// 14 x 14 patch, (c, ky, kx) order, patches raster; m/s are the processor's
// mean and std already multiplied by 255.
PD_EXPORT
int pd_ne_patch_u8(const void* canvas, void* out, uint32_t h, uint32_t w, float m0, float m1,
                   float m2, float s0, float s1, float s2, void* stream) {
    if (h == 0 || w == 0) return 0;
    if ((h % 14u) != 0u || (w % 14u) != 0u) return cudaErrorInvalidValue;
    const uint32_t n = (h / 14u) * (w / 14u);
    pd_ne_patch_u8_kernel<<<n, 256, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)canvas, (__half*)out, w, m0, m1, m2, s0, s1, s2);
    return pd_launch_status();
}

// ---- 897 ---------------------------------------------------------------

// Fixed-order sum over a 256-thread block: butterfly within each warp, then
// the eight warp totals in index order - the same bits for a row whatever
// the launch's row count.
__device__ __forceinline__ float pd_ne_block_sum(float v, float* red) {
#pragma unroll
    for (uint32_t o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31u) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < 8u; ++i) t += red[i];
    return t;
}

// One row a 256-thread block; d a multiple of 1024 (each thread owns d / 1024
// float4 chunks: thread t chunk j is columns 4 (t + 256 j) .. + 3).
template <uint32_t NCH>
__global__ void __launch_bounds__(256) pd_ne_rms_h_kernel(
    const float* __restrict__ x, const float* __restrict__ w, __half* __restrict__ out,
    uint32_t d, float eps) {
    __shared__ float red[8];
    const float* xr = x + (size_t)blockIdx.x * d;
    float4 v[NCH];
    float ss = 0.0f;
#pragma unroll
    for (uint32_t j = 0; j < NCH; ++j) {
        v[j] = *reinterpret_cast<const float4*>(xr + 4u * (threadIdx.x + 256u * j));
        ss += v[j].x * v[j].x + v[j].y * v[j].y + v[j].z * v[j].z + v[j].w * v[j].w;
    }
    const float r = rsqrtf(pd_ne_block_sum(ss, red) / (float)d + eps);
    __half* orow = out + (size_t)blockIdx.x * d;
#pragma unroll
    for (uint32_t j = 0; j < NCH; ++j) {
        const uint32_t c = 4u * (threadIdx.x + 256u * j);
        const float4 wv = *reinterpret_cast<const float4*>(w + c);
        const __half2 a = __floats2half2_rn((v[j].x * r) * wv.x, (v[j].y * r) * wv.y);
        const __half2 b = __floats2half2_rn((v[j].z * r) * wv.z, (v[j].w * r) * wv.w);
        uint2 pk;
        pk.x = *reinterpret_cast<const uint32_t*>(&a);
        pk.y = *reinterpret_cast<const uint32_t*>(&b);
        *reinterpret_cast<uint2*>(orow + c) = pk;
    }
}

// 897: out16[r] = f16(x[r] * rsqrt(mean(x[r]^2) + eps) * w) over [rows][d] f32
// rows; d in {1024, 2048, 4096}.
PD_EXPORT
int pd_ne_rms_h(const void* x, const void* w, void* out16, uint32_t rows, uint32_t d, float eps,
                void* stream) {
    if (rows == 0) return 0;
    cudaStream_t st = (cudaStream_t)stream;
    const float* xp = (const float*)x;
    const float* wp = (const float*)w;
    __half* op = (__half*)out16;
    switch (d) {
        case 1024u: pd_ne_rms_h_kernel<1u><<<rows, 256, 0, st>>>(xp, wp, op, d, eps); break;
        case 2048u: pd_ne_rms_h_kernel<2u><<<rows, 256, 0, st>>>(xp, wp, op, d, eps); break;
        case 4096u: pd_ne_rms_h_kernel<4u><<<rows, 256, 0, st>>>(xp, wp, op, d, eps); break;
        default: return cudaErrorInvalidValue;
    }
    return pd_launch_status();
}

// ---- 902 ---------------------------------------------------------------

// The residual add folded into the next norm: x += proj (f32, in place - the
// landing's own y + acc, so the same bits as the ring's residual epilogue),
// then out16 = f16(rms(x) * w). The projections land plain: a residual
// read-modify-write in the GEMM's epilogue cost 20-25% of the projection on
// GB10 (8B body, down at 32768 rows 44.0 -> 57.8 ms) - more than its bytes,
// the residual plane evicting the GEMM's resident weight group - while here
// the add rides a pass that reads x anyway. w null: the add alone.
template <uint32_t NCH>
__global__ void __launch_bounds__(256) pd_ne_add_rms_h_kernel(
    float* __restrict__ x, const float* __restrict__ proj, const float* __restrict__ w,
    __half* __restrict__ out, uint32_t d, float eps) {
    __shared__ float red[8];
    float* xr = x + (size_t)blockIdx.x * d;
    const float* pr = proj + (size_t)blockIdx.x * d;
    float4 v[NCH];
    float ss = 0.0f;
#pragma unroll
    for (uint32_t j = 0; j < NCH; ++j) {
        const uint32_t c = 4u * (threadIdx.x + 256u * j);
        float4 a = *reinterpret_cast<const float4*>(xr + c);
        const float4 b = *reinterpret_cast<const float4*>(pr + c);
        a.x += b.x;
        a.y += b.y;
        a.z += b.z;
        a.w += b.w;
        *reinterpret_cast<float4*>(xr + c) = a;
        v[j] = a;
        ss += a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w;
    }
    if (w == nullptr) return;
    const float r = rsqrtf(pd_ne_block_sum(ss, red) / (float)d + eps);
    __half* orow = out + (size_t)blockIdx.x * d;
#pragma unroll
    for (uint32_t j = 0; j < NCH; ++j) {
        const uint32_t c = 4u * (threadIdx.x + 256u * j);
        const float4 wv = *reinterpret_cast<const float4*>(w + c);
        const __half2 a = __floats2half2_rn((v[j].x * r) * wv.x, (v[j].y * r) * wv.y);
        const __half2 b = __floats2half2_rn((v[j].z * r) * wv.z, (v[j].w * r) * wv.w);
        uint2 pk;
        pk.x = *reinterpret_cast<const uint32_t*>(&a);
        pk.y = *reinterpret_cast<const uint32_t*>(&b);
        *reinterpret_cast<uint2*>(orow + c) = pk;
    }
}

// 902: x += proj over [rows][d] f32, then (w non-null) out16 = f16(x *
// rsqrt(mean(x^2) + eps) * w) - 897 on the updated rows; d in {1024, 2048,
// 4096}.
PD_EXPORT
int pd_ne_add_rms_h(void* x, const void* proj, const void* w, void* out16, uint32_t rows,
                    uint32_t d, float eps, void* stream) {
    if (rows == 0) return 0;
    cudaStream_t st = (cudaStream_t)stream;
    float* xp = (float*)x;
    const float* pp = (const float*)proj;
    const float* wp = (const float*)w;
    __half* op = (__half*)out16;
    switch (d) {
        case 1024u: pd_ne_add_rms_h_kernel<1u><<<rows, 256, 0, st>>>(xp, pp, wp, op, d, eps); break;
        case 2048u: pd_ne_add_rms_h_kernel<2u><<<rows, 256, 0, st>>>(xp, pp, wp, op, d, eps); break;
        case 4096u: pd_ne_add_rms_h_kernel<4u><<<rows, 256, 0, st>>>(xp, pp, wp, op, d, eps); break;
        default: return cudaErrorInvalidValue;
    }
    return pd_launch_status();
}

// ---- 898 ---------------------------------------------------------------

__global__ void __launch_bounds__(256) pd_ne_embed_rows_kernel(
    const uint32_t* __restrict__ src, const uint16_t* __restrict__ emb,
    const float* __restrict__ pic, float* __restrict__ out, uint32_t d) {
    const uint32_t s = src[blockIdx.x];
    float* orow = out + (size_t)blockIdx.x * d;
    if (s & PD_NE_EMB_PIC) {
        const float4* pr = reinterpret_cast<const float4*>(pic + (size_t)(s & ~PD_NE_EMB_PIC) * d);
        for (uint32_t c = threadIdx.x; c < d / 4u; c += blockDim.x)
            reinterpret_cast<float4*>(orow)[c] = pr[c];
        return;
    }
    // bf16 is the top half of an f32: the widen is a shift, exact
    const uint4* er = reinterpret_cast<const uint4*>(emb + (size_t)s * d);
    for (uint32_t c = threadIdx.x; c < d / 8u; c += blockDim.x) {
        const uint4 raw = er[c];
        const uint32_t u[4] = {raw.x, raw.y, raw.z, raw.w};
        float4 lo, hi;
        lo.x = __uint_as_float(u[0] << 16);
        lo.y = __uint_as_float(u[0] & 0xffff0000u);
        lo.z = __uint_as_float(u[1] << 16);
        lo.w = __uint_as_float(u[1] & 0xffff0000u);
        hi.x = __uint_as_float(u[2] << 16);
        hi.y = __uint_as_float(u[2] & 0xffff0000u);
        hi.z = __uint_as_float(u[3] << 16);
        hi.w = __uint_as_float(u[3] & 0xffff0000u);
        reinterpret_cast<float4*>(orow)[2u * c] = lo;
        reinterpret_cast<float4*>(orow)[2u * c + 1u] = hi;
    }
}

// 898: out[r] (f32 [rows][d]) = the bf16 embedding row of src[r], or - with
// src[r]'s top bit set - row src[r] & 0x7fffffff of pic ([*][d] f32). The
// caller guarantees every id is in the vocabulary and every picture row
// exists. d % 8 == 0.
PD_EXPORT
int pd_ne_embed_rows(const void* src, const void* emb, const void* pic, void* out, uint32_t rows,
                     uint32_t d, void* stream) {
    if (rows == 0) return 0;
    if ((d % 8u) != 0u) return cudaErrorInvalidValue;
    pd_ne_embed_rows_kernel<<<rows, 256, 0, (cudaStream_t)stream>>>(
        (const uint32_t*)src, (const uint16_t*)emb, (const float*)pic, (float*)out, d);
    return pd_launch_status();
}

// ---- 899 ---------------------------------------------------------------

// One row a block. The row's first (nh + nkv) heads (q then k) rotate; v, the
// rest of the row, is untouched. Thread work item u = (head, pair pp): dims
// (2 pp, 2 pp + 1) and their partners (+ hd / 2) as two half2.
__global__ void __launch_bounds__(128) pd_ne_rope_qk_kernel(
    __half* __restrict__ qkv, const uint32_t* __restrict__ pos, const float* __restrict__ cosb,
    const float* __restrict__ sinb, uint32_t nh, uint32_t nkv, uint32_t hd, uint32_t stride,
    float qscale, float l4_beta, uint32_t l4_orig) {
    const uint32_t row = blockIdx.x;
    const uint32_t p = pos[row];
    const uint32_t half = hd / 2u;
    // Llama 4's factor in the reference's f32 ops: 1 + beta * log(1 + floor(p / orig))
    float qs = qscale;
    if (l4_beta != 0.0f) {
        const float fl = floorf((float)p / (float)l4_orig);
        qs = qscale * (1.0f + l4_beta * logf(1.0f + fl));
    }
    const float* cr = cosb + (size_t)p * half;
    const float* sr = sinb + (size_t)p * half;
    __half* r = qkv + (size_t)row * stride;
    const uint32_t pairs = half / 2u;
    for (uint32_t u = threadIdx.x; u < (nh + nkv) * pairs; u += blockDim.x) {
        const uint32_t h = u / pairs, i = 2u * (u - h * pairs);
        __half* a = r + (size_t)h * hd + i;
        const float2 x1 = __half22float2(*reinterpret_cast<const __half2*>(a));
        const float2 x2 = __half22float2(*reinterpret_cast<const __half2*>(a + half));
        const float c0 = cr[i], c1 = cr[i + 1u], s0 = sr[i], s1 = sr[i + 1u];
        float y1a = x1.x * c0 - x2.x * s0, y1b = x1.y * c1 - x2.y * s1;
        float y2a = x2.x * c0 + x1.x * s0, y2b = x2.y * c1 + x1.y * s1;
        if (h < nh) {
            y1a *= qs;
            y1b *= qs;
            y2a *= qs;
            y2b *= qs;
        }
        *reinterpret_cast<__half2*>(a) = __floats2half2_rn(y1a, y1b);
        *reinterpret_cast<__half2*>(a + half) = __floats2half2_rn(y2a, y2b);
    }
}

// 899: in-place rope over rows [0, rows) of the fused landing ([rows][stride]
// halves, q heads [nh][hd] then k heads [nkv][hd] at its start); cos / sin
// [*][hd / 2] f32 tables at row pos[r]; q (only) then multiplied by qscale x
// (1 + l4_beta ln(1 + floor(pos / l4_orig))) - l4_beta 0 = no Llama 4 factor.
// hd % 4 == 0.
PD_EXPORT
int pd_ne_rope_qk(void* qkv, const void* pos, const void* cosb, const void* sinb, uint32_t rows,
                  uint32_t nh, uint32_t nkv, uint32_t hd, uint32_t stride, float qscale,
                  float l4_beta, uint32_t l4_orig, void* stream) {
    if (rows == 0) return 0;
    if ((hd % 4u) != 0u || stride < (nh + nkv) * hd || (l4_beta != 0.0f && l4_orig == 0u))
        return cudaErrorInvalidValue;
    pd_ne_rope_qk_kernel<<<rows, 128, 0, (cudaStream_t)stream>>>(
        (__half*)qkv, (const uint32_t*)pos, (const float*)cosb, (const float*)sinb, nh, nkv, hd,
        stride, qscale, l4_beta, l4_orig);
    return pd_launch_status();
}

// ---- 900 ---------------------------------------------------------------
//
// Bidirectional attention over PACKED sequences (cu[s] .. cu[s + 1]) off the
// fused landing [rows][(nh + 2 nkv) hd]: q at head h * hd, k at (nh + j) hd, v
// at (nh + nkv + j) hd for kv head j. q arrives rotated and pre-scaled in the
// log2 domain (log2(e) / sqrt(hd) x Llama 4's factor, slot 899), so the
// softmax runs on exp2. One block a (query tile, kv head), eight warps: a
// warp owns (16-row strip, query head of the group), so G heads share each
// staged K/V tile - G = 4 at the body's 32 / 8 (32 query rows a block), 1 in
// the tower (128 rows). Numeric class: f16 q/k/v/P operands, f32 S/O and
// online softmax - the varlen kernels' (no flush-to-zero cutoff: an exp2 far
// below the row max is a true near-zero).
//
// K / V stage on a two-deep cp.async ring: the next tile's rows land while
// this one's mma run. At hd 128 a tile is 32 KB a block against ~4K cycles
// of mma, so the synchronous stage the varlen kernels use (it bought nothing
// at their shapes) stalled every warp for about as long as it computed: on
// GB10 a 16K-token document's attention went 2.59 -> ~1.8 s (58 -> ~82 TF/s).
// The bytes and the arithmetic order are the synchronous stage's, so the
// output bits are too.
template <uint32_t HD, uint32_t G>
struct PdNeGeo {
    static constexpr uint32_t STRIPS = 8u / G;      // 16-row strips a block
    static constexpr uint32_t ROWS = STRIPS * 16u;  // query rows a block
    static constexpr uint32_t DPD = HD + 8u;        // padded smem row
    static constexpr uint32_t STAGE = 2u * PD_NE_KT * DPD;  // halves: K then V
    static constexpr uint32_t SMEM = 2u * STAGE * 2u;         // two stages, bytes
};

template <uint32_t HD, uint32_t G>
__global__ void __launch_bounds__(256, 1) pd_ne_attn_kernel(
    const __half* __restrict__ qkv, const uint32_t* __restrict__ cu,
    const uint32_t* __restrict__ tiles, __half* __restrict__ out, uint32_t nh, uint32_t nkv) {
#if PD_FA_OK
    using Geo = PdNeGeo<HD, G>;
    constexpr uint32_t KT = PD_NE_KT, DPD = Geo::DPD, NT = 256u;
    const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
    const uint32_t g8 = lane >> 2, t4 = lane & 3u, lg = lane >> 3;
    const uint32_t kvh = blockIdx.y;
    const uint32_t td = tiles[blockIdx.x];
    const uint32_t sq = td >> PD_NE_TILE_SHIFT;
    const uint32_t q0 = (td & ((1u << PD_NE_TILE_SHIFT) - 1u)) * Geo::ROWS;
    const uint32_t base = cu[sq], L = cu[sq + 1u] - base;
    const uint32_t strip = warp / G, gi = warp % G;
    const uint32_t h = kvh * G + gi;  // this warp's query head
    const uint32_t wq0 = q0 + strip * 16u;
    const size_t rs = (size_t)(nh + 2u * nkv) * HD;
    const size_t koff = (size_t)(nh + kvh) * HD, voff = (size_t)(nh + nkv + kvh) * HD;

    extern __shared__ unsigned char nesh[];
    __half* const ring = reinterpret_cast<__half*>(nesh);
    // stage tile t0's K and V rows of this kv head into ring slot `slot`,
    // zero past the sequence
    auto stage = [&](uint32_t t0, uint32_t slot) {
        __half* sk = ring + (size_t)slot * Geo::STAGE;
        __half* sv = sk + (size_t)KT * DPD;
        constexpr uint32_t U = KT * (HD / 8u);
        for (uint32_t u = tid; u < U; u += NT) {
            const uint32_t kk = u / (HD / 8u), d8 = (u % (HD / 8u)) * 8u;
            const uint32_t ks = t0 + kk;
            const bool ok = ks < L;
            const __half* rp = qkv + (size_t)(base + (ok ? ks : 0u)) * rs + d8;
            pd_ne_cp16(sk + (size_t)kk * DPD + d8, rp + koff, ok);
            pd_ne_cp16(sv + (size_t)kk * DPD + d8, rp + voff, ok);
        }
        asm volatile("cp.async.commit_group;");
    };

    // ---- q fragments for this warp's 16 rows of head h
    const uint32_t jr[2] = {g8, g8 + 8u};
    uint32_t qa[HD / 16u][4];
#pragma unroll
    for (uint32_t d0 = 0; d0 < HD / 16u; ++d0) {
#pragma unroll
        for (uint32_t e = 0; e < 2u; ++e) {
            const uint32_t qi = wq0 + jr[e];
            const bool ok = qi < L;
            const __half* qp = qkv + (size_t)(base + (ok ? qi : 0u)) * rs + (size_t)h * HD
                               + d0 * 16u + 2u * t4;
            qa[d0][e] = ok ? *reinterpret_cast<const uint32_t*>(qp) : 0u;
            qa[d0][e + 2u] = ok ? *reinterpret_cast<const uint32_t*>(qp + 8u) : 0u;
        }
    }

    float m_st[2] = {-1e30f, -1e30f}, l_st[2] = {0.f, 0.f};
    float o_acc[HD / 8u][4];
#pragma unroll
    for (uint32_t nt = 0; nt < HD / 8u; ++nt)
#pragma unroll
        for (uint32_t e = 0; e < 4u; ++e) o_acc[nt][e] = 0.f;

    stage(0u, 0u);
    for (uint32_t t0 = 0, it = 0; t0 < L; t0 += KT, ++it) {
        // the next tile lands while this one computes
        if (t0 + KT < L) {
            stage(t0 + KT, (it + 1u) & 1u);
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();
        const __half* sh_k = ring + (size_t)(it & 1u) * Geo::STAGE;
        const __half* sh_v = sh_k + (size_t)KT * DPD;

        float s_acc[KT / 8u][4];
#pragma unroll
        for (uint32_t nt = 0; nt < KT / 8u; ++nt)
#pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) s_acc[nt][e] = 0.f;
#pragma unroll
        for (uint32_t d0 = 0; d0 < HD / 16u; ++d0) {
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
                pd_fa_mma16(s_acc[np * 2u], qa[d0][0], qa[d0][1], qa[d0][2], qa[d0][3],
                            kb4[0], kb4[1]);
                pd_fa_mma16(s_acc[np * 2u + 1u], qa[d0][0], qa[d0][1], qa[d0][2],
                            qa[d0][3], kb4[2], kb4[3]);
            }
        }
        // mask the keys past the sequence (the last tile only)
        float mn[2] = {m_st[0], m_st[1]};
#pragma unroll
        for (uint32_t nt = 0; nt < KT / 8u; ++nt) {
            const uint32_t kb = t0 + nt * 8u + 2u * t4;
#pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) {
                if (kb + (e & 1u) >= L) s_acc[nt][e] = -1e30f;
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
                const float w = exp2f(s_acc[nt][e] - mn[e >> 1]);
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
            corr[r] = exp2f(m_st[r] - mn[r]);
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
        __syncthreads();   // this slot read before the stage after next refills it
    }

    const float nrm[2] = {l_st[0] > 0.f ? 1.f / l_st[0] : 0.f,
                          l_st[1] > 0.f ? 1.f / l_st[1] : 0.f};
    const size_t od = (size_t)nh * HD;
#pragma unroll
    for (uint32_t nt = 0; nt < HD / 8u; ++nt) {
        const uint32_t c = nt * 8u + 2u * t4;
#pragma unroll
        for (uint32_t r = 0; r < 2u; ++r) {
            const uint32_t qi = wq0 + jr[r];
            if (qi >= L) continue;
            __half* op = out + (size_t)(base + qi) * od + (size_t)h * HD + c;
            *reinterpret_cast<__half2*>(op) =
                __floats2half2_rn(o_acc[nt][2u * r] * nrm[r], o_acc[nt][2u * r + 1u] * nrm[r]);
        }
    }
#else
    (void)qkv; (void)cu; (void)tiles; (void)out; (void)nh; (void)nkv;
#endif
}

template <uint32_t HD, uint32_t G>
static int pd_ne_attn_go(const __half* qkv, const uint32_t* cu, const uint32_t* tiles,
                         uint32_t n_tiles, __half* out, uint32_t nh, uint32_t nkv,
                         cudaStream_t st) {
    using Geo = PdNeGeo<HD, G>;
    static cudaError_t attr = cudaFuncSetAttribute(
        pd_ne_attn_kernel<HD, G>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)Geo::SMEM);
    if (attr != cudaSuccess) return attr;
    pd_ne_attn_kernel<HD, G><<<dim3(n_tiles, nkv), 256u, Geo::SMEM, st>>>(qkv, cu, tiles, out,
                                                                           nh, nkv);
    return pd_launch_status();
}

// 900: bidirectional varlen GQA attention off the fused landing (see above),
// out f16 [rows][nh][hd]. cu [n_seq + 1] row offsets; tiles [n_tiles] (seq <<
// 12) | tile, one per 16 x 8 / G query rows (G = nh / nkv). (hd, G) in
// {(128, 4), (64, 1)}. sm_80+.
PD_EXPORT
int pd_ne_attn(const void* qkv, const void* cu, const void* tiles, uint32_t n_tiles, void* out,
               uint32_t nh, uint32_t nkv, uint32_t hd, void* stream) {
    if (n_tiles == 0) return 0;
    if (nkv == 0 || nh % nkv != 0) return cudaErrorInvalidValue;
    int dev = 0, cc = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&cc, cudaDevAttrComputeCapabilityMajor, dev);
    if (cc < 8) return cudaErrorInvalidValue;
    const __half* q = (const __half*)qkv;
    const uint32_t* c = (const uint32_t*)cu;
    const uint32_t* t = (const uint32_t*)tiles;
    __half* o = (__half*)out;
    cudaStream_t st = (cudaStream_t)stream;
    const uint32_t g = nh / nkv;
    if (hd == 128u && g == 4u) return pd_ne_attn_go<128u, 4u>(q, c, t, n_tiles, o, nh, nkv, st);
    if (hd == 64u && g == 1u) return pd_ne_attn_go<64u, 1u>(q, c, t, n_tiles, o, nh, nkv, st);
    return cudaErrorInvalidValue;
}

// ---- 901 ---------------------------------------------------------------

// Fixed-order double sum over a 1024-thread block (32 warps), as
// pd_ne_block_sum.
__device__ __forceinline__ double pd_ne_block_sum_d(double v, double* red) {
#pragma unroll
    for (uint32_t o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31u) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    double t = 0.0;
    for (uint32_t i = 0; i < (blockDim.x >> 5); ++i) t += red[i];
    return t;
}

// One sequence a block of d / 4 threads, thread t owning columns 4t .. 4t + 3:
// the rows summed in order in f64, the mean in f64, then (score null) the
// unit vector over the first `dims` columns or (score set) the dot with the
// f32 score row - both reductions in a fixed tree.
__global__ void __launch_bounds__(1024) pd_ne_pool_kernel(
    const float* __restrict__ x, const uint32_t* __restrict__ cu, const float* __restrict__ score,
    float* __restrict__ mean_out, float* __restrict__ out, uint32_t d, uint32_t dims) {
    __shared__ double red[32];
    const uint32_t sq = blockIdx.x, c = 4u * threadIdx.x;
    const uint32_t first = cu[sq], count = cu[sq + 1u] - first;
    double a[4] = {0.0, 0.0, 0.0, 0.0};
    for (uint32_t t = 0; t < count; ++t) {
        const float4 v = *reinterpret_cast<const float4*>(x + (size_t)(first + t) * d + c);
        a[0] += (double)v.x;
        a[1] += (double)v.y;
        a[2] += (double)v.z;
        a[3] += (double)v.w;
    }
    const double inv_n = 1.0 / (double)count;
#pragma unroll
    for (uint32_t i = 0; i < 4u; ++i) a[i] *= inv_n;
    if (mean_out != nullptr) {
        *reinterpret_cast<float4*>(mean_out + (size_t)sq * d + c) =
            make_float4((float)a[0], (float)a[1], (float)a[2], (float)a[3]);
    }
    if (score != nullptr) {
        double p = 0.0;
#pragma unroll
        for (uint32_t i = 0; i < 4u; ++i) p += a[i] * (double)score[c + i];
        const double dot = pd_ne_block_sum_d(p, red);
        if (threadIdx.x == 0) out[sq] = (float)dot;
        return;
    }
    double ss = 0.0;
#pragma unroll
    for (uint32_t i = 0; i < 4u; ++i) ss += (c + i < dims) ? a[i] * a[i] : 0.0;
    const double nrm = sqrt(pd_ne_block_sum_d(ss, red));
    const double inv = 1.0 / fmax(nrm, 1e-12);
#pragma unroll
    for (uint32_t i = 0; i < 4u; ++i)
        if (c + i < dims) out[(size_t)sq * dims + c + i] = (float)(a[i] * inv);
}

// 901: per sequence s (rows cu[s] .. cu[s + 1] of x, [rows][d] f32): the mean
// (into mean_out [n_seq][d] f32 when non-null), then out [n_seq][dims] = the
// mean's first dims components over their own norm (score null), or out
// [n_seq] = mean . score (score [d] f32). d a multiple of 128 up to 4096;
// dims in 1..=d.
PD_EXPORT
int pd_ne_pool(const void* x, const void* cu, const void* score, void* mean_out, void* out,
               uint32_t n_seq, uint32_t d, uint32_t dims, void* stream) {
    if (n_seq == 0) return 0;
    if (d == 0 || (d % 128u) != 0u || d > 4096u || dims == 0 || dims > d)
        return cudaErrorInvalidValue;
    pd_ne_pool_kernel<<<n_seq, d / 4u, 0, (cudaStream_t)stream>>>(
        (const float*)x, (const uint32_t*)cu, (const float*)score, (float*)mean_out, (float*)out,
        d, dims);
    return pd_launch_status();
}
