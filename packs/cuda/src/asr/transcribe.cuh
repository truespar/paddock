// transcribe.cuh - Nemotron 3.5 Transcribe's speech encoder glue (slots
// 942-944) and its MTP block's FP8 expert GEMVs (945-946). The encoder's blocks ride the house tower kernels (LayerNorm into
// f16, the f16 tensor-core GEMMs with their bias / GELU / residual landings,
// the packed varlen bidirectional attention); what is here is what they do
// not cover. Textually-included segment of the single pack translation unit.
// Not standalone-compilable: include order is defined by ../pack.cu.
//
// The reference is NeMo's ParallelExpertEncoder (online inference) as the
// Transformers port spells it:
//
// 942 pd_tr_mel_norm - NeMo's `per_feature` normalization of the log-mel:
//     each of the 128 bins over the recording's valid frames, mean then the
//     UNBIASED variance (0 for a single frame), 1e-5 added to the standard
//     deviation (not under the root). The sums run in f64, so the statistics
//     are the f64 evaluation's rounded once; frames past `valid` (up to the
//     padded `frames`) are zero. The f16 rows ARE the stacked encoder input:
//     eight consecutive frames are one 1024-wide row.
// 943 pd_tr_qk_norm_rope - per head of the fused q|k|v landing
//     ([rows][3][heads][128] f16, in place): LayerNorm over the head (with its
//     own weight and bias) on q and on k, then the rope on the head's first 64
//     dims only (rotary_fraction 0.5, rotate-half pairing c <-> c + 32, cos /
//     sin of the row's position in its window from the host table), each
//     value rounded to f16 once. v is not touched.
// 944 pd_tr_fuse - the speaker fusion: LayerNorm(asr row) * w + b plus the
//     row's speaker infusion, one of 256 precomputed rows (the eight speakers'
//     0/1 activity pattern through the 8-wide LayerNorm and the fixed
//     sinusoid table - the host forms them in f64), rounded once to the f16
//     the projector GEMM reads.
//
// Every row is a function of its own inputs (942's statistics excepted, which
// are the recording's), so a window's rows are the same bits however many
// windows or recordings share a pass.

// ------------------------------------------------------------------ 942
// grid = 4 column groups of 32 bins, 32 x 32 threads: lane = bin, y = frame
// stride. Three sweeps: sum, centred square sum, the normalized f16 rows.
__global__ void __launch_bounds__(1024) pd_tr_mel_norm_kernel(const float* __restrict__ mel,
                                                              __half* __restrict__ out,
                                                              uint32_t valid, uint32_t frames) {
    const uint32_t c = blockIdx.x * 32u + threadIdx.x;
    const uint32_t ty = threadIdx.y;
    __shared__ double red[32][33];
    __shared__ double s_mean[32], s_inv[32];
    double acc = 0.0;
    for (uint32_t f = ty; f < valid; f += 32u) acc += (double)mel[(size_t)f * 128u + c];
    red[ty][threadIdx.x] = acc;
    __syncthreads();
    if (ty == 0) {
        double s = 0.0;
        for (uint32_t i = 0; i < 32u; ++i) s += red[i][threadIdx.x];
        s_mean[threadIdx.x] = valid > 0 ? s / (double)valid : 0.0;
    }
    __syncthreads();
    const double mean = s_mean[threadIdx.x];
    acc = 0.0;
    for (uint32_t f = ty; f < valid; f += 32u) {
        const double d = (double)mel[(size_t)f * 128u + c] - mean;
        acc += d * d;
    }
    __syncthreads();
    red[ty][threadIdx.x] = acc;
    __syncthreads();
    if (ty == 0) {
        double s = 0.0;
        for (uint32_t i = 0; i < 32u; ++i) s += red[i][threadIdx.x];
        const double var = valid > 1 ? s / (double)(valid - 1u) : 0.0;
        s_inv[threadIdx.x] = 1.0 / (sqrt(var) + 1e-5);
    }
    __syncthreads();
    const double inv = s_inv[threadIdx.x];
    for (uint32_t f = ty; f < frames; f += 32u) {
        const float v =
            f < valid ? (float)(((double)mel[(size_t)f * 128u + c] - mean) * inv) : 0.0f;
        out[(size_t)f * 128u + c] = __float2half(v);
    }
}

PD_EXPORT
int pd_tr_mel_norm(const void* mel, void* out, uint32_t valid, uint32_t frames, void* stream) {
    if (frames == 0) return 0;
    if (valid > frames) return cudaErrorInvalidValue;
    pd_tr_mel_norm_kernel<<<4, dim3(32, 32), 0, (cudaStream_t)stream>>>(
        (const float*)mel, (__half*)out, valid, frames);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 943
// one warp per (row, head, q|k): four dims a lane, lane l holds 4l .. 4l + 3,
// so the rope's partner of a dim under 32 sits eight lanes up.
__global__ void pd_tr_qk_norm_rope_kernel(__half* __restrict__ qkv, const float* __restrict__ wq,
                                          const float* __restrict__ bq,
                                          const float* __restrict__ wk,
                                          const float* __restrict__ bk,
                                          const uint32_t* __restrict__ pos,
                                          const float* __restrict__ rope, uint32_t rows,
                                          uint32_t heads, float eps) {
    const uint32_t warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const uint32_t lane = threadIdx.x & 31u;
    if (warp >= rows * heads * 2u) return;
    const uint32_t which = warp % 2u;  // 0 q, 1 k
    const uint32_t head = (warp / 2u) % heads;
    const uint32_t row = warp / (2u * heads);
    __half* p = qkv + ((size_t)row * 3u + which) * heads * 128u + (size_t)head * 128u + lane * 4u;
    const float* w = which ? wk : wq;
    const float* b = which ? bk : bq;
    float x[4];
    {
        const uint2 raw = *reinterpret_cast<const uint2*>(p);
        const __half2* h2 = reinterpret_cast<const __half2*>(&raw);
        const float2 a = __half22float2(h2[0]), c = __half22float2(h2[1]);
        x[0] = a.x;
        x[1] = a.y;
        x[2] = c.x;
        x[3] = c.y;
    }
    float s = x[0] + x[1] + x[2] + x[3];
#pragma unroll
    for (uint32_t o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    const float mean = s / 128.0f;
    float v = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < 4u; ++i) {
        const float d = x[i] - mean;
        v += d * d;
    }
#pragma unroll
    for (uint32_t o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const float inv = rsqrtf(v / 128.0f + eps);
    float y[4];
#pragma unroll
    for (uint32_t i = 0; i < 4u; ++i) {
        const uint32_t d = lane * 4u + i;
        y[i] = (x[i] - mean) * inv * w[d] + b[d];
    }
    // the rope over dims 0..63: lanes 0..7 pair with lanes 8..15
    float other[4];
#pragma unroll
    for (uint32_t i = 0; i < 4u; ++i) other[i] = __shfl_xor_sync(0xffffffffu, y[i], 8);
    if (lane < 16u) {
        const float* t = rope + (size_t)pos[row] * 64u;
#pragma unroll
        for (uint32_t i = 0; i < 4u; ++i) {
            const uint32_t f = (lane * 4u + i) & 31u;
            const float cs = t[2u * f], sn = t[2u * f + 1u];
            y[i] = lane < 8u ? y[i] * cs - other[i] * sn : y[i] * cs + other[i] * sn;
        }
    }
    uint2 raw;
    __half2* h2 = reinterpret_cast<__half2*>(&raw);
    h2[0] = __floats2half2_rn(y[0], y[1]);
    h2[1] = __floats2half2_rn(y[2], y[3]);
    *reinterpret_cast<uint2*>(p) = raw;
}

PD_EXPORT
int pd_tr_qk_norm_rope(void* qkv, const void* wq, const void* bq, const void* wk, const void* bk,
                       const void* pos, const void* rope, uint32_t rows, uint32_t heads,
                       uint32_t head_dim, float eps, void* stream) {
    if (rows == 0 || heads == 0) return 0;
    if (head_dim != 128u) return cudaErrorInvalidValue;
    const uint64_t warps = (uint64_t)rows * heads * 2u;
    const uint32_t blocks = (uint32_t)((warps + 7u) / 8u);
    pd_tr_qk_norm_rope_kernel<<<blocks, 256, 0, (cudaStream_t)stream>>>(
        (__half*)qkv, (const float*)wq, (const float*)bq, (const float*)wk, (const float*)bk,
        (const uint32_t*)pos, (const float*)rope, rows, heads, eps);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 944
// one block per row, pd_layernorm_f16_kernel's body (the same sums in the same
// order) with the infusion row added before the round.
__global__ void pd_tr_fuse_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                  const float* __restrict__ b, const uint32_t* __restrict__ codes,
                                  const float* __restrict__ table, __half* __restrict__ out,
                                  uint32_t n, float eps) {
    const uint32_t row = blockIdx.x;
    const float* xr = x + (size_t)row * n;
    const float* tr = table + (size_t)(codes[row] & 255u) * n;
    __half* orow = out + (size_t)row * n;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    __shared__ float wsum[32];
    __shared__ float s_mean, s_inv;
    const uint32_t lane = tid & 31u, warp = tid >> 5, nwarps = (nth + 31u) >> 5;

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
    const float mean = s_mean;
    float vacc = 0.0f;
    for (uint32_t i = tid; i < n; i += nth) {
        const float d = xr[i] - mean;
        vacc += d * d;
    }
    for (uint32_t s = 16; s > 0; s >>= 1) vacc += __shfl_down_sync(0xffffffffu, vacc, s);
    __syncthreads();
    if (lane == 0) wsum[warp] = vacc;
    __syncthreads();
    if (tid == 0) {
        float v = 0.0f;
        for (uint32_t i = 0; i < nwarps; ++i) v += wsum[i];
        s_inv = rsqrtf(v / (float)n + eps);
    }
    __syncthreads();
    const float inv = s_inv;
    for (uint32_t i = tid; i < n; i += nth)
        orow[i] = __float2half((xr[i] - mean) * inv * w[i] + b[i] + tr[i]);
}

PD_EXPORT
int pd_tr_fuse(const void* x, const void* w, const void* b, const void* codes, const void* table,
               void* out, uint32_t rows, uint32_t n, float eps, void* stream) {
    if (rows == 0 || n == 0) return 0;
    pd_tr_fuse_kernel<<<rows, 256, 0, (cudaStream_t)stream>>>(
        (const float*)x, (const float*)w, (const float*)b, (const uint32_t*)codes,
        (const float*)table, (__half*)out, n, eps);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 945/946
// The speech LM's MTP block keeps its routed and shared experts as the
// checkpoint ships them: e4m3 rows with one f32 scale per expert (broadcast
// per row at load). The block's MoE only runs on draft rows (a row per
// drafting slot, one chain step at a time), so these are decode GEMVs - the
// e4m3 bytes widened exactly, multiplied against f32 activations, the row
// scale applied to the f32 dot: W8A32, the checkpoint's own numbers.
//
// 945 pd_f8r_moe_up_relu2: out[r][j][o] = relu(scale[e o] * W[e ff + o] . x[r])^2
//     for e = idx[r][j]; one warp per output row, 16 weight bytes a lane.
// 946 pd_f8r_moe_down: out[r][o] (+)= sum_j w[r][j] * scale * W[e hidden + o] .
//     act[r][j], the k picks summed in pick order; `acc` adds onto out (the
//     shared expert onto the routed sum).

__device__ __forceinline__ float pd_f8r_dot16(const uint4 raw, const float* __restrict__ x) {
    const unsigned char* b = reinterpret_cast<const unsigned char*>(&raw);
    float s = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < 16u; ++i) {
        __nv_fp8_e4m3 v;
        v.__x = b[i];
        s = fmaf((float)v, x[i], s);
    }
    return s;
}

// one warp: the dot of a 16-byte-aligned e4m3 row of `in` against f32 x
__device__ __forceinline__ float pd_f8r_row_dot(const unsigned char* __restrict__ w,
                                                const float* __restrict__ x, uint32_t in,
                                                uint32_t lane) {
    float s = 0.0f;
    for (uint32_t c = lane * 16u; c < in; c += 32u * 16u)
        s += pd_f8r_dot16(*reinterpret_cast<const uint4*>(w + c), x + c);
#pragma unroll
    for (uint32_t o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    return s;
}

__global__ void pd_f8r_moe_up_relu2_kernel(const unsigned char* __restrict__ data,
                                           const float* __restrict__ rs,
                                           const uint32_t* __restrict__ idx,
                                           const float* __restrict__ x, float* __restrict__ out,
                                           uint32_t in, uint32_t ff, uint32_t k, uint32_t rows) {
    const uint64_t warp = ((uint64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const uint32_t lane = threadIdx.x & 31u;
    if (warp >= (uint64_t)rows * k * ff) return;
    const uint32_t o = (uint32_t)(warp % ff);
    const uint32_t rj = (uint32_t)(warp / ff);  // r * k + j
    const uint32_t r = rj / k;
    const size_t wrow = (size_t)idx[rj] * ff + o;
    const float d = pd_f8r_row_dot(data + wrow * in, x + (size_t)r * in, in, lane);
    if (lane == 0) {
        const float v = fmaxf(d * rs[wrow], 0.0f);
        out[(size_t)rj * ff + o] = v * v;
    }
}

PD_EXPORT
int pd_f8r_moe_up_relu2(const void* data, const void* rs, const void* idx, const void* x,
                        void* out, uint32_t in, uint32_t ff, uint32_t k, uint32_t rows,
                        void* stream) {
    if (rows == 0 || k == 0 || ff == 0) return 0;
    if ((in & 15u) != 0) return cudaErrorInvalidValue;
    const uint64_t warps = (uint64_t)rows * k * ff;
    pd_f8r_moe_up_relu2_kernel<<<(uint32_t)((warps + 7u) / 8u), 256, 0, (cudaStream_t)stream>>>(
        (const unsigned char*)data, (const float*)rs, (const uint32_t*)idx, (const float*)x,
        (float*)out, in, ff, k, rows);
    return pd_launch_status();
}

__global__ void pd_f8r_moe_down_kernel(const unsigned char* __restrict__ data,
                                       const float* __restrict__ rs,
                                       const uint32_t* __restrict__ idx,
                                       const float* __restrict__ w,
                                       const float* __restrict__ act, float* __restrict__ out,
                                       uint32_t ff, uint32_t hidden, uint32_t k, uint32_t rows,
                                       uint32_t acc) {
    const uint64_t warp = ((uint64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const uint32_t lane = threadIdx.x & 31u;
    if (warp >= (uint64_t)rows * hidden) return;
    const uint32_t o = (uint32_t)(warp % hidden);
    const uint32_t r = (uint32_t)(warp / hidden);
    float sum = 0.0f;
    for (uint32_t j = 0; j < k; ++j) {
        const size_t wrow = (size_t)idx[r * k + j] * hidden + o;
        const float d =
            pd_f8r_row_dot(data + wrow * ff, act + ((size_t)r * k + j) * ff, ff, lane);
        // the pick's weight times the scaled dot, accumulated by one fma -
        // spelled out so the sorted path (953's combine) rounds alike
        sum = fmaf(w[r * k + j], d * rs[wrow], sum);
    }
    if (lane == 0) {
        float* y = out + (size_t)r * hidden + o;
        *y = acc ? *y + sum : sum;
    }
}

PD_EXPORT
int pd_f8r_moe_down(const void* data, const void* rs, const void* idx, const void* w,
                    const void* act, void* out, uint32_t ff, uint32_t hidden, uint32_t k,
                    uint32_t rows, uint32_t acc, void* stream) {
    if (rows == 0 || k == 0 || hidden == 0) return 0;
    if ((ff & 15u) != 0) return cudaErrorInvalidValue;
    const uint64_t warps = (uint64_t)rows * hidden;
    pd_f8r_moe_down_kernel<<<(uint32_t)((warps + 7u) / 8u), 256, 0, (cudaStream_t)stream>>>(
        (const unsigned char*)data, (const float*)rs, (const uint32_t*)idx, (const float*)w,
        (const float*)act, (float*)out, ff, hidden, k, rows, acc);
    return pd_launch_status();
}

// ---------------------------------------------------------- 951-953
// The same FP8 experts over expert-sorted blocks (pd_moe_align_bm, BM = 8):
// 945/946 run a warp per (row, pick, output) and so stream an expert's rows
// once per pick that routes to it - a draft step's 32 rows x 6 picks read
// ~190 expert row sets for ~100 distinct experts, and the drafter's append
// over a pass's ~700 rows read every expert many times over. Here a CTA owns
// a block of one expert's tokens and walks each of its output rows once
// against all of them. Per (pick, output) the arithmetic is 945/946's
// verbatim (the same 16-byte chunks in the same order, the same xor
// reduction), and the down's pick sum moves to 953 in pick order with the
// same fma - so the outputs are 945/946's bits.
#define PD_F8R_BM 8u

// A CTA is one block's expert x PD_F8R_OUT output rows (8 warps x 4 rows).
// The block's token rows stage through shared memory 512 elements at a
// time - a warp re-reading eight f32 token rows for every output row it
// owned was the whole cost (L2 traffic ~8 x 10.7 KB an output row against
// 2.7 KB of weights). One 512-element stage is exactly one 16-byte chunk a
// lane, 945's chunk walk, so every (pick, output) sums in 945's order.
#define PD_F8R_OUT 32u
template <bool DOWN>
__global__ void __launch_bounds__(256) pd_f8r_moe_sorted_kernel(
        const unsigned char* __restrict__ data, const float* __restrict__ rs,
        const uint32_t* __restrict__ sorted_row, const uint32_t* __restrict__ sorted_slot,
        const uint32_t* __restrict__ block_expert, const float* __restrict__ pick_w,
        const float* __restrict__ x, float* __restrict__ out, uint32_t in, uint32_t n_out,
        uint32_t k) {
    constexpr uint32_t RW = PD_F8R_OUT / 8u;  // output rows a warp
    __shared__ __align__(16) float xs[PD_F8R_BM][512];
    const uint32_t b = blockIdx.x;
    const uint32_t e = block_expert[b];
    if (e == PD_MOE_PAD) return;
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    uint32_t pair[PD_F8R_BM];
    bool live[PD_F8R_BM];
#pragma unroll
    for (uint32_t t = 0; t < PD_F8R_BM; ++t) {
        const uint32_t r = sorted_row[b * PD_F8R_BM + t];
        live[t] = r != PD_MOE_PAD;
        pair[t] = live[t] ? r * k + sorted_slot[b * PD_F8R_BM + t] : 0u;
    }
    float s[RW][PD_F8R_BM];
#pragma unroll
    for (uint32_t q = 0; q < RW; ++q)
#pragma unroll
        for (uint32_t t = 0; t < PD_F8R_BM; ++t) s[q][t] = 0.0f;
    const uint32_t o0 = blockIdx.y * PD_F8R_OUT + warp * RW;
    for (uint32_t kc = 0; kc < in; kc += 512u) {
        const uint32_t kn = min(512u, in - kc);
        __syncthreads();  // the previous stage's readers are done
        for (uint32_t i = threadIdx.x; i < PD_F8R_BM * 128u; i += 256u) {
            const uint32_t t = i / 128u, q4 = (i % 128u) * 4u;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (live[t] && q4 < kn) {
                const size_t xr = DOWN ? (size_t)pair[t] : (size_t)(pair[t] / k);
                v = *reinterpret_cast<const float4*>(x + xr * in + kc + q4);
            }
            *reinterpret_cast<float4*>(&xs[t][q4]) = v;
        }
        __syncthreads();
        const uint32_t c = lane * 16u;
        if (c >= kn) continue;
#pragma unroll
        for (uint32_t q = 0; q < RW; ++q) {
            const uint32_t o = o0 + q;
            if (o >= n_out) break;
            const uint4 raw = *reinterpret_cast<const uint4*>(
                data + ((size_t)e * n_out + o) * in + kc + c);
#pragma unroll
            for (uint32_t t = 0; t < PD_F8R_BM; ++t)
                if (live[t]) s[q][t] += pd_f8r_dot16(raw, &xs[t][c]);
        }
    }
#pragma unroll
    for (uint32_t q = 0; q < RW; ++q) {
        const uint32_t o = o0 + q;
#pragma unroll
        for (uint32_t t = 0; t < PD_F8R_BM; ++t)
#pragma unroll
            for (uint32_t m = 16; m > 0; m >>= 1) s[q][t] += __shfl_xor_sync(0xffffffffu, s[q][t], m);
        if (lane != 0 || o >= n_out) continue;
        const size_t wrow = (size_t)e * n_out + o;
#pragma unroll
        for (uint32_t t = 0; t < PD_F8R_BM; ++t) {
            if (!live[t]) continue;
            if (DOWN) {
                // the pick's scaled dot; 953 applies its weight in pick order
                out[(size_t)pair[t] * n_out + o] = s[q][t] * rs[wrow];
            } else {
                const float v = fmaxf(s[q][t] * rs[wrow], 0.0f);
                out[(size_t)pair[t] * n_out + o] = v * v;
            }
        }
    }
    (void)pick_w;
}

// 951: 945 over sorted blocks - out[pair][o], pair = r * k + j.
PD_EXPORT
int pd_f8r_moe_up_relu2_sorted(const void* data, const void* rs, const void* sorted_row,
                               const void* sorted_slot, const void* block_expert, const void* x,
                               void* out, uint32_t in, uint32_t ff, uint32_t k, uint32_t nb,
                               void* stream) {
    if (nb == 0 || ff == 0 || k == 0) return 0;
    if ((in & 15u) != 0 || nb > 65535u * 8u) return cudaErrorInvalidValue;
    const dim3 grid(nb, (ff + PD_F8R_OUT - 1u) / PD_F8R_OUT);
    pd_f8r_moe_sorted_kernel<false><<<grid, 256, 0, (cudaStream_t)stream>>>(
        (const unsigned char*)data, (const float*)rs, (const uint32_t*)sorted_row,
        (const uint32_t*)sorted_slot, (const uint32_t*)block_expert, nullptr, (const float*)x,
        (float*)out, in, ff, k);
    return pd_launch_status();
}

// 952: 946's per-pick scaled dots over sorted blocks - part[pair][o].
PD_EXPORT
int pd_f8r_moe_down_sorted(const void* data, const void* rs, const void* sorted_row,
                           const void* sorted_slot, const void* block_expert, const void* act,
                           void* part, uint32_t ff, uint32_t hidden, uint32_t k, uint32_t nb,
                           void* stream) {
    if (nb == 0 || hidden == 0 || k == 0) return 0;
    if ((ff & 15u) != 0) return cudaErrorInvalidValue;
    const dim3 grid(nb, (hidden + PD_F8R_OUT - 1u) / PD_F8R_OUT);
    pd_f8r_moe_sorted_kernel<true><<<grid, 256, 0, (cudaStream_t)stream>>>(
        (const unsigned char*)data, (const float*)rs, (const uint32_t*)sorted_row,
        (const uint32_t*)sorted_slot, (const uint32_t*)block_expert, nullptr, (const float*)act,
        (float*)part, ff, hidden, k);
    return pd_launch_status();
}

// 953: out[r][o] (+)= sum_j w[r][j] * part[r k + j][o] in pick order, one
// fma a pick - 946's sum.
__global__ void pd_f8r_moe_combine_kernel(const float* __restrict__ part,
                                          const float* __restrict__ w, float* __restrict__ out,
                                          uint32_t hidden, uint32_t k, uint32_t rows,
                                          uint32_t acc) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)rows * hidden) return;
    const uint32_t r = (uint32_t)(i / hidden), o = (uint32_t)(i % hidden);
    float sum = 0.0f;
    for (uint32_t j = 0; j < k; ++j)
        sum = fmaf(w[r * k + j], part[((size_t)r * k + j) * hidden + o], sum);
    out[i] = acc ? out[i] + sum : sum;
}

PD_EXPORT
int pd_f8r_moe_combine(const void* part, const void* w, void* out, uint32_t hidden, uint32_t k,
                       uint32_t rows, uint32_t acc, void* stream) {
    if (rows == 0 || hidden == 0) return 0;
    const uint64_t n = (uint64_t)rows * hidden;
    pd_f8r_moe_combine_kernel<<<(uint32_t)((n + 255u) / 256u), 256, 0, (cudaStream_t)stream>>>(
        (const float*)part, (const float*)w, (float*)out, hidden, k, rows, acc);
    return pd_launch_status();
}

// 949 pd_tr_codes: the speaker branch's encoder-rate activity patterns on the
// device - row r's bit s = mean of speaker s's eight 10-ms probabilities
// > 0.5, rows past `valid` (padding) 0. The host walk it replaces summed the
// eight in order from -0.0 and divided by 8; this is that sum, so the bits are
// the host's. Keeps a recording's encode free of a readback.
__global__ void pd_tr_codes_kernel(const float* __restrict__ probs, uint32_t* __restrict__ codes,
                                   uint32_t rows, uint32_t valid) {
    const uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    uint32_t code = 0u;
    if (r < valid) {
        for (uint32_t s = 0; s < 8u; ++s) {
            float acc = -0.0f;
            for (uint32_t f = 0; f < 8u; ++f) acc = __fadd_rn(acc, probs[((size_t)r * 8u + f) * 8u + s]);
            if (__fdiv_rn(acc, 8.0f) > 0.5f) code |= 1u << s;
        }
    }
    codes[r] = code;
}

PD_EXPORT
int pd_tr_codes(const void* probs, void* codes, uint32_t rows, uint32_t valid, void* stream) {
    if (rows == 0) return 0;
    if (valid > rows) return cudaErrorInvalidValue;
    pd_tr_codes_kernel<<<(rows + 127u) / 128u, 128u, 0, (cudaStream_t)stream>>>(
        (const float*)probs, (uint32_t*)codes, rows, valid);
    return pd_launch_status();
}

// ---- the CTC timestamp head (slots 958-960) --------------------------------
// The checkpoint's deferred head over a window of the speech branch's states:
// ConvT x2 upsample, two pre-norm transformer layers (16 heads of 80, a
// LayerNorm on each q / k head, NeoX rope on the first 40 dims), the residual,
// a 23863-class classifier and its log-softmax. The GEMMs, norms and adds ride
// the f32 ops every lane has; these are the three pieces the head alone needs.
// Everything is f32 (the reference serves the head in bf16).

// 958: per-head LayerNorm (affine, `eps`) on q and k of a fused [q | k | v]
// row, then rope on the first `rot` dims of each head (x[0:rot/2] and
// x[rot/2:rot] rotated as a pair, NeoX's rotate-half) from a per-position
// (cos, sin) table [rows][rot/2][2]. A warp a (row, head, q/k).
__global__ void pd_tr_ctc_qk_norm_rope_kernel(float* __restrict__ qkv, const float* __restrict__ qn_w,
                                              const float* __restrict__ qn_b,
                                              const float* __restrict__ kn_w,
                                              const float* __restrict__ kn_b,
                                              const float* __restrict__ rope, uint32_t rows,
                                              uint32_t heads, uint32_t hd, uint32_t rot, float eps) {
    const uint32_t warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31u;
    const uint32_t per_row = 2u * heads;
    if (warp >= rows * per_row) return;
    const uint32_t r = warp / per_row, rem = warp % per_row, isk = rem / heads, h = rem % heads;
    float* x = qkv + (size_t)r * 3u * heads * hd + (size_t)isk * heads * hd + (size_t)h * hd;
    const float* w = isk ? kn_w : qn_w;
    const float* b = isk ? kn_b : qn_b;
    float v[4];
    const uint32_t per = (hd + 31u) / 32u;
    float s = 0.f;
    for (uint32_t k = 0; k < per; ++k) {
        const uint32_t d = lane + 32u * k;
        v[k] = d < hd ? x[d] : 0.f;
        s += v[k];
    }
    for (uint32_t o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    const float mean = s / (float)hd;
    float q = 0.f;
    for (uint32_t k = 0; k < per; ++k) {
        const uint32_t d = lane + 32u * k;
        const float c = d < hd ? v[k] - mean : 0.f;
        q += c * c;
    }
    for (uint32_t o = 16; o > 0; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
    const float inv = rsqrtf(q / (float)hd + eps);
    for (uint32_t k = 0; k < per; ++k) {
        const uint32_t d = lane + 32u * k;
        if (d < hd) v[k] = (v[k] - mean) * inv * w[d] + b[d];
    }
    // write the normed head, then rotate its first `rot` dims in place
    for (uint32_t k = 0; k < per; ++k) {
        const uint32_t d = lane + 32u * k;
        if (d < hd) x[d] = v[k];
    }
    __syncwarp();
    const uint32_t half = rot / 2u;
    if (lane < half) {
        const float* cs = rope + ((size_t)r * half + lane) * 2u;
        const float x1 = x[lane], x2 = x[lane + half];
        const float c = cs[0], sn = cs[1];
        x[lane] = x1 * c - x2 * sn;
        x[lane + half] = x2 * c + x1 * sn;
    }
}

PD_EXPORT
int pd_tr_ctc_qk_norm_rope(void* qkv, const void* qn_w, const void* qn_b, const void* kn_w,
                           const void* kn_b, const void* rope, uint32_t rows, uint32_t heads,
                           uint32_t hd, uint32_t rot, float eps, void* stream) {
    if (rows == 0 || heads == 0) return 0;
    if (hd == 0 || hd > 128u || rot > hd || (rot & 1u) || rot / 2u > 32u)
        return cudaErrorInvalidValue;
    const uint32_t warps = rows * 2u * heads;
    pd_tr_ctc_qk_norm_rope_kernel<<<(warps + 7u) / 8u, 256u, 0, (cudaStream_t)stream>>>(
        (float*)qkv, (const float*)qn_w, (const float*)qn_b, (const float*)kn_w,
        (const float*)kn_b, (const float*)rope, rows, heads, hd, rot, eps);
    return pd_launch_status();
}

// 959: full (unmasked) attention over a window, f32: a thread a query row
// with its q and running output in registers, keys and values staged through
// shared memory 32 at a time, the softmax kept online (running max and sum).
// [rows][3*heads*HD] in, [rows][heads*HD] out.
template <uint32_t HD>
__global__ void __launch_bounds__(128) pd_tr_ctc_attention_kernel(
        const float* __restrict__ qkv, float* __restrict__ out, uint32_t rows, uint32_t heads,
        float scale) {
    constexpr uint32_t KT = 32u;
    __shared__ float ks[KT][HD];
    __shared__ float vs[KT][HD];
    const uint32_t h = blockIdx.y;
    const uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = 3u * (size_t)heads * HD;
    float q[HD], acc[HD];
    const bool live = r < rows;
    for (uint32_t d = 0; d < HD; ++d) {
        q[d] = live ? qkv[(size_t)r * stride + (size_t)h * HD + d] * scale : 0.f;
        acc[d] = 0.f;
    }
    float m = -INFINITY, l = 0.f;
    for (uint32_t k0 = 0; k0 < rows; k0 += KT) {
        const uint32_t nk = min(KT, rows - k0);
        for (uint32_t i = threadIdx.x; i < KT * HD; i += blockDim.x) {
            const uint32_t j = i / HD, d = i % HD;
            const bool ok = j < nk;
            const size_t base = (size_t)(k0 + j) * stride + (size_t)h * HD + d;
            ks[j][d] = ok ? qkv[base + (size_t)heads * HD] : 0.f;
            vs[j][d] = ok ? qkv[base + 2u * (size_t)heads * HD] : 0.f;
        }
        __syncthreads();
        if (live) {
            for (uint32_t j = 0; j < nk; ++j) {
                float s = 0.f;
                for (uint32_t d = 0; d < HD; ++d) s = fmaf(q[d], ks[j][d], s);
                const float mn = fmaxf(m, s);
                const float corr = expf(m - mn), p = expf(s - mn);
                l = l * corr + p;
                for (uint32_t d = 0; d < HD; ++d) acc[d] = fmaf(acc[d], corr, p * vs[j][d]);
                m = mn;
            }
        }
        __syncthreads();
    }
    if (live) {
        const float inv = 1.f / l;
        for (uint32_t d = 0; d < HD; ++d) out[(size_t)r * heads * HD + (size_t)h * HD + d] = acc[d] * inv;
    }
}

PD_EXPORT
int pd_tr_ctc_attention(const void* qkv, void* out, uint32_t rows, uint32_t heads, uint32_t hd,
                        void* stream) {
    if (rows == 0 || heads == 0) return 0;
    if (hd != 80u || heads > 65535u) return cudaErrorInvalidValue;
    const dim3 grid((rows + 127u) / 128u, heads);
    pd_tr_ctc_attention_kernel<80u><<<grid, 128u, 0, (cudaStream_t)stream>>>(
        (const float*)qkv, (float*)out, rows, heads, 1.0f / sqrtf(80.0f));
    return pd_launch_status();
}

// 960: the classifier's log-softmax read back only at the alignment's
// columns: per row, logits + bias over all `classes` reduced to their log-sum-
// exp (a CTA a row), then out[row][j] = logit[cols[j]] + bias - lse.
__global__ void __launch_bounds__(256) pd_tr_ctc_lse_gather_kernel(
        const float* __restrict__ logits, const float* __restrict__ bias, uint32_t classes,
        const uint32_t* __restrict__ cols, uint32_t n_cols, float* __restrict__ out) {
    __shared__ float red[32];
    const uint32_t r = blockIdx.x;
    const float* x = logits + (size_t)r * classes;
    float mx = -INFINITY;
    for (uint32_t c = threadIdx.x; c < classes; c += blockDim.x) mx = fmaxf(mx, x[c] + bias[c]);
    for (uint32_t o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    if ((threadIdx.x & 31u) == 0) red[threadIdx.x >> 5] = mx;
    __syncthreads();
    if (threadIdx.x < 32u) {
        float v = threadIdx.x < blockDim.x / 32u ? red[threadIdx.x] : -INFINITY;
        for (uint32_t o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
        if (threadIdx.x == 0) red[0] = v;
    }
    __syncthreads();
    mx = red[0];
    __syncthreads();
    float s = 0.f;
    for (uint32_t c = threadIdx.x; c < classes; c += blockDim.x) s += expf(x[c] + bias[c] - mx);
    for (uint32_t o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if ((threadIdx.x & 31u) == 0) red[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x < 32u) {
        float v = threadIdx.x < blockDim.x / 32u ? red[threadIdx.x] : 0.f;
        for (uint32_t o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (threadIdx.x == 0) red[0] = v;
    }
    __syncthreads();
    const float lse = mx + logf(red[0]);
    for (uint32_t j = threadIdx.x; j < n_cols; j += blockDim.x) {
        const uint32_t c = cols[j];
        out[(size_t)r * n_cols + j] = x[c] + bias[c] - lse;
    }
}

PD_EXPORT
int pd_tr_ctc_lse_gather(const void* logits, const void* bias, uint32_t rows, uint32_t classes,
                         const void* cols, uint32_t n_cols, void* out, void* stream) {
    if (rows == 0 || n_cols == 0) return 0;
    if (classes == 0) return cudaErrorInvalidValue;
    pd_tr_ctc_lse_gather_kernel<<<rows, 256u, 0, (cudaStream_t)stream>>>(
        (const float*)logits, (const float*)bias, classes, (const uint32_t*)cols, n_cols,
        (float*)out);
    return pd_launch_status();
}
