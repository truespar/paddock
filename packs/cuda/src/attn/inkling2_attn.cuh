// attn/inkling2_attn.cuh - Inkling 2 Mini's attention over a contiguous f32
// K / V history (the serial lane; the paged serving arms are separate).
//
// No RoPE: a hidden-conditioned relative bias carries position. Per query
// row and head, `rel` holds 16 relative states; the layer's profile bank
// proj [16][1024] (bf16) turns them into one bias per backward distance:
// bias(d) = sum_i rel[i] * proj[i][d] for 0 <= d < 1024, 0 past it. A
// per-token sink logit (one a head, zero value) joins the softmax. Scores
// are q.k * scale + bias; sliding layers see the `window` newest keys (the
// current one included), full layers every earlier key.

// slot 986: rows consecutive positions p0 .. p0 + rows - 1 of one sequence,
// their K / V already appended (k / v [pos][kv_heads][128]). q [rows][heads
// * 128] (row stride qs), rel [rows][heads * 16] (rs), sink [rows][heads]
// (ss), out [rows][heads][128]. One CTA a (row, KV head): its G query heads
// (G <= 4) over key tiles of 32, online softmax seeded by the sink.
template <uint32_t G>
__global__ void __launch_bounds__(128) pd_inkling2_attn_kernel(
        const float* q, uint32_t qs, const float* rel, uint32_t rs, const float* sink, uint32_t ss,
        const float* k, const float* v, const __nv_bfloat16* __restrict__ proj, float* __restrict__ out,
        uint32_t p0, uint32_t heads, uint32_t kv_heads, uint32_t window, uint32_t extent,
        float scale) {
    constexpr uint32_t HD = 128u, DR = 16u, T = 32u;
    PD_PDL_ARM();
    __shared__ float sq[G][HD];
    __shared__ float srel[G][DR];
    __shared__ float sp[G][T];
    __shared__ float sm[G], sl[G], sfix[G];
    const uint32_t r = blockIdx.x, kh = blockIdx.y, tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t h0 = kh * G;
    const uint32_t p = p0 + r;
    const uint32_t kv_dim = kv_heads * HD;
    const float* qr = pd_pdl_after(q) + (size_t)r * qs;
    const float* rr = pd_pdl_after(rel) + (size_t)r * rs;
    #pragma unroll
    for (uint32_t g = 0; g < G; ++g) sq[g][tid] = qr[(h0 + g) * HD + tid];
    if (tid < G * DR) srel[tid / DR][tid % DR] = rr[(h0 + tid / DR) * DR + tid % DR];
    if (tid < G) {
        sm[tid] = pd_pdl_after(sink)[(size_t)r * ss + h0 + tid];
        sl[tid] = 1.0f;  // the sink: exp(sink - m) with m = sink, value 0
    }
    __syncthreads();
    const float* ks = pd_pdl_after(k);
    const float* vs = pd_pdl_after(v);
    const uint32_t lo = window ? (p + 1u > window ? p + 1u - window : 0u) : 0u;
    float acc[G];
    #pragma unroll
    for (uint32_t g = 0; g < G; ++g) acc[g] = 0.0f;
    for (uint32_t t0 = lo; t0 <= p; t0 += T) {
        // scores: each warp 8 keys, lanes over 4 dims each
        for (uint32_t kk = warp; kk < T; kk += 4u) {
            const uint32_t j = t0 + kk;
            float d[G];
            #pragma unroll
            for (uint32_t g = 0; g < G; ++g) d[g] = 0.0f;
            if (j <= p) {
                const float4 kv4 = *(const float4*)(ks + (size_t)j * kv_dim + kh * HD + lane * 4u);
                #pragma unroll
                for (uint32_t g = 0; g < G; ++g)
                    d[g] = sq[g][lane * 4u] * kv4.x + sq[g][lane * 4u + 1] * kv4.y +
                           sq[g][lane * 4u + 2] * kv4.z + sq[g][lane * 4u + 3] * kv4.w;
            }
            #pragma unroll
            for (uint32_t g = 0; g < G; ++g)
                for (uint32_t sh = 16; sh > 0; sh >>= 1) d[g] += __shfl_xor_sync(0xffffffffu, d[g], sh);
            if (lane < G) {
                float s = -INFINITY;
                if (j <= p) {
                    s = d[lane] * scale;
                    const uint32_t dist = p - j;
                    if (dist < extent) {
                        float b = 0.0f;
                        #pragma unroll
                        for (uint32_t i = 0; i < DR; ++i)
                            b = fmaf(srel[lane][i], __bfloat162float(proj[i * extent + dist]), b);
                        s += b;
                    }
                }
                sp[lane][kk] = s;
            }
        }
        __syncthreads();
        if (tid < G) {
            float m = sm[tid];
            #pragma unroll 4
            for (uint32_t kk = 0; kk < T; ++kk) m = fmaxf(m, sp[tid][kk]);
            const float fix = expf(sm[tid] - m);
            float l = sl[tid] * fix;
            for (uint32_t kk = 0; kk < T; ++kk) {
                const float e = expf(sp[tid][kk] - m);  // -inf past p -> 0
                sp[tid][kk] = e;
                l += e;
            }
            sm[tid] = m;
            sl[tid] = l;
            sfix[tid] = fix;
        }
        __syncthreads();
        #pragma unroll
        for (uint32_t g = 0; g < G; ++g) acc[g] *= sfix[g];
        const uint32_t n = min(T, p + 1u - t0);
        for (uint32_t kk = 0; kk < n; ++kk) {
            const float vv = vs[(size_t)(t0 + kk) * kv_dim + kh * HD + tid];
            #pragma unroll
            for (uint32_t g = 0; g < G; ++g) acc[g] = fmaf(sp[g][kk], vv, acc[g]);
        }
        __syncthreads();
    }
    #pragma unroll
    for (uint32_t g = 0; g < G; ++g) out[((size_t)r * heads + h0 + g) * HD + tid] = acc[g] / sl[g];
}

PD_EXPORT
int pd_inkling2_attn(const void* q, uint32_t qs, const void* rel, uint32_t rs, const void* sink,
                     uint32_t ss, const void* k, const void* v, const void* proj, void* out,
                     uint32_t rows, uint32_t p0, uint32_t heads, uint32_t kv_heads,
                     uint32_t head_dim, uint32_t d_rel, uint32_t window, uint32_t extent,
                     float scale, void* stream) {
    if (rows == 0) return 0;
    if (head_dim != 128u || d_rel != 16u || kv_heads == 0 || heads % kv_heads)
        return cudaErrorInvalidValue;
    const uint32_t G = heads / kv_heads;
    const dim3 grid(rows, kv_heads);
    #define PD_IK2_ATTN(GG)                                                                   \
        pd_pdl_go(pd_inkling2_attn_kernel<GG>, grid, 128, 0u, (cudaStream_t)stream,           \
                  (const float*)q, qs, (const float*)rel, rs, (const float*)sink, ss,         \
                  (const float*)k, (const float*)v, (const __nv_bfloat16*)proj, (float*)out,  \
                  p0, heads, kv_heads, window, extent, scale)
    switch (G) {
    case 1: PD_IK2_ATTN(1u); break;
    case 2: PD_IK2_ATTN(2u); break;
    case 3: PD_IK2_ATTN(3u); break;
    case 4: PD_IK2_ATTN(4u); break;
    default: return cudaErrorInvalidValue;
    }
    #undef PD_IK2_ATTN
    return pd_launch_status();
}

// ---- the serving lane's K / V: paged (full layers) and rings (sliding) -----
// A visit's K (or V) store: the full-attention visits page their rows in
// 16-row blocks through the slot's block table (every full visit shares the
// table - their positions are the same); the sliding visits keep a ring of
// `ring` rows a slot (a power of two >= the window plus a prompt chunk, so a
// chunk's first row still finds its whole window after the chunk landed).
// Rows are kv_dim wide in `dtype`: 0 f16, 1 fp8 e4m3, 2 f32 (the reference
// class). Keys are L2-clipped and values unit-RMS before they land, so e4m3
// holds them unscaled.
struct PdIk2Kv {
    void* base;
    const uint32_t* bt;  // paged: block tables [slots][bps]; null = ring
    uint32_t bps;
    uint32_t ring;
    uint32_t dtype;  // 0 f16, 1 fp8 e4m3, 2 f32, 3 bf16 (the draft's)
};

__device__ __forceinline__ size_t pd_ik2_kv_row(const PdIk2Kv& kv, uint32_t slot, uint32_t pos) {
    if (kv.bt) return (size_t)kv.bt[(size_t)slot * kv.bps + pos / 16u] * 16u + (pos & 15u);
    return (size_t)slot * kv.ring + (pos & (kv.ring - 1u));
}

// 4 consecutive elements of a stored row as f32
__device__ __forceinline__ float4 pd_ik2_kv_ld4(const PdIk2Kv& kv, size_t elem) {
    if (kv.dtype == 2u) return *(const float4*)((const float*)kv.base + elem);
    if (kv.dtype == 0u) {
        const uint2 h = *(const uint2*)((const __half*)kv.base + elem);
        const float2 a = __half22float2(*(const __half2*)&h.x), b = __half22float2(*(const __half2*)&h.y);
        return make_float4(a.x, a.y, b.x, b.y);
    }
    const uint32_t w = *(const uint32_t*)((const uint8_t*)kv.base + elem);
    return make_float4(pd_ik2_e4m3_val(w & 0xFFu), pd_ik2_e4m3_val((w >> 8) & 0xFFu),
                       pd_ik2_e4m3_val((w >> 16) & 0xFFu), pd_ik2_e4m3_val(w >> 24));
}

__device__ __forceinline__ float pd_ik2_kv_ld1(const PdIk2Kv& kv, size_t elem) {
    if (kv.dtype == 2u) return ((const float*)kv.base)[elem];
    if (kv.dtype == 0u) return __half2float(((const __half*)kv.base)[elem]);
    if (kv.dtype == 3u) return __bfloat162float(((const __nv_bfloat16*)kv.base)[elem]);
    return pd_ik2_e4m3_val(((const uint8_t*)kv.base)[elem]);
}

// slot 989: store `rows` prepped K (or V) rows (src [rows][kv_dim], row
// stride ss) at their (slot, pos) in a visit's store.
__global__ void __launch_bounds__(256) pd_inkling2_kv_store_kernel(
        const float* src, uint32_t ss, const uint32_t* slots, const uint32_t* pos, PdIk2Kv kv,
        uint32_t kv_dim) {
    PD_PDL_ARM();
    const uint32_t r = blockIdx.x;
    const float* s = pd_pdl_after(src) + (size_t)r * ss;
    const size_t row = pd_ik2_kv_row(kv, pd_pdl_after(slots)[r], pd_pdl_after(pos)[r]);
    for (uint32_t i = threadIdx.x; i < kv_dim; i += blockDim.x) {
        const float v = s[i];
        const size_t e = row * kv_dim + i;
        if (kv.dtype == 2u) ((float*)kv.base)[e] = v;
        else if (kv.dtype == 0u) ((__half*)kv.base)[e] = __float2half_rn(v);
        else if (kv.dtype == 3u) ((__nv_bfloat16*)kv.base)[e] = __float2bfloat16_rn(v);
        else ((uint8_t*)kv.base)[e] = __nv_fp8_e4m3(v).__x;
    }
}

PD_EXPORT
int pd_inkling2_kv_store(const void* src, uint32_t ss, const void* slots, const void* pos,
                         void* base, const void* bt, uint32_t bps, uint32_t ring, uint32_t dtype,
                         uint32_t kv_dim, uint32_t rows, void* stream) {
    if (rows == 0) return 0;
    // class 3 (bf16) holds the DSpark draft's K / V
    if (dtype > 3u || (bt == nullptr && (ring == 0 || (ring & (ring - 1u))))) return cudaErrorInvalidValue;
    const PdIk2Kv kv{base, (const uint32_t*)bt, bps, ring, dtype};
    pd_pdl_go(pd_inkling2_kv_store_kernel, rows, 256, 0u, (cudaStream_t)stream, (const float*)src,
              ss, (const uint32_t*)slots, (const uint32_t*)pos, kv, kv_dim);
    return pd_launch_status();
}

// slot 990: the relative-bias attention of `rows` rows, each at its own
// (slot, pos), over a visit's K / V stores - the serving lane's twin of slot
// 986. The keys a row sees are split `splits` ways: split s of row r writes
// its (acc, m, l) without the sink to part_o / part_ml, and the fold kernel
// below adds the sink; with one split the CTA folds the sink itself and
// writes `out`. q / rel / sink as slot 986's.
template <uint32_t G>
__global__ void __launch_bounds__(128) pd_inkling2_attn_kv_kernel(
        const float* q, uint32_t qs, const float* rel, uint32_t rs, const float* sink, uint32_t ss,
        PdIk2Kv kk, PdIk2Kv vv, const uint32_t* slots, const uint32_t* pos,
        const __nv_bfloat16* __restrict__ proj, float* __restrict__ out, float* __restrict__ part_o,
        float* __restrict__ part_ml, uint32_t heads, uint32_t kv_heads, uint32_t window,
        uint32_t extent, float scale) {
    constexpr uint32_t HD = 128u, DR = 16u, T = 32u;
    PD_PDL_ARM();
    __shared__ float sq[G][HD];
    __shared__ float srel[G][DR];
    __shared__ float sp[G][T];
    __shared__ float sm[G], sl[G], sfix[G];
    __shared__ uint32_t srow[T];
    const uint32_t r = blockIdx.x, kh = blockIdx.y, split = blockIdx.z, splits = gridDim.z;
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const uint32_t h0 = kh * G;
    const uint32_t kv_dim = kv_heads * HD;
    const uint32_t slot = pd_pdl_after(slots)[r], p = pd_pdl_after(pos)[r];
    const float* qr = pd_pdl_after(q) + (size_t)r * qs;
    const float* rr = pd_pdl_after(rel) + (size_t)r * rs;
    #pragma unroll
    for (uint32_t g = 0; g < G; ++g) sq[g][tid] = qr[(h0 + g) * HD + tid];
    if (tid < G * DR) srel[tid / DR][tid % DR] = rr[(h0 + tid / DR) * DR + tid % DR];
    // this split's keys [k0, k1) of the row's [lo, p]
    const uint32_t lo = window ? (p + 1u > window ? p + 1u - window : 0u) : 0u;
    const uint32_t n = p + 1u - lo, per = (n + splits - 1u) / splits;
    const uint32_t k0 = lo + min(n, split * per), k1 = lo + min(n, (split + 1u) * per);
    if (tid < G) {
        if (splits == 1u) {
            sm[tid] = pd_pdl_after(sink)[(size_t)r * ss + h0 + tid];
            sl[tid] = 1.0f;
        } else {
            sm[tid] = -INFINITY;
            sl[tid] = 0.0f;
        }
    }
    __syncthreads();
    float acc[G];
    #pragma unroll
    for (uint32_t g = 0; g < G; ++g) acc[g] = 0.0f;
    for (uint32_t t0 = k0; t0 < k1; t0 += T) {
        if (tid < T) srow[tid] = t0 + tid < k1 ? (uint32_t)pd_ik2_kv_row(kk, slot, t0 + tid) : 0u;
        __syncthreads();
        for (uint32_t kx = warp; kx < T; kx += 4u) {
            const uint32_t j = t0 + kx;
            float d[G];
            #pragma unroll
            for (uint32_t g = 0; g < G; ++g) d[g] = 0.0f;
            if (j < k1) {
                const float4 k4 = pd_ik2_kv_ld4(kk, (size_t)srow[kx] * kv_dim + kh * HD + lane * 4u);
                #pragma unroll
                for (uint32_t g = 0; g < G; ++g)
                    d[g] = sq[g][lane * 4u] * k4.x + sq[g][lane * 4u + 1] * k4.y +
                           sq[g][lane * 4u + 2] * k4.z + sq[g][lane * 4u + 3] * k4.w;
            }
            #pragma unroll
            for (uint32_t g = 0; g < G; ++g)
                for (uint32_t sh = 16; sh > 0; sh >>= 1) d[g] += __shfl_xor_sync(0xffffffffu, d[g], sh);
            if (lane < G) {
                float s = -INFINITY;
                if (j < k1) {
                    s = d[lane] * scale;
                    const uint32_t dist = p - j;
                    if (dist < extent) {
                        float b = 0.0f;
                        #pragma unroll
                        for (uint32_t i = 0; i < DR; ++i)
                            b = fmaf(srel[lane][i], __bfloat162float(proj[i * extent + dist]), b);
                        s += b;
                    }
                }
                sp[lane][kx] = s;
            }
        }
        __syncthreads();
        if (tid < G) {
            float m = sm[tid];
            for (uint32_t kx = 0; kx < T; ++kx) m = fmaxf(m, sp[tid][kx]);
            // a split with no finite score yet keeps m at -inf: nothing to fix
            const float fix = m == -INFINITY ? 1.0f : expf(sm[tid] - m);
            float l = sl[tid] * fix;
            for (uint32_t kx = 0; kx < T; ++kx) {
                const float e = m == -INFINITY ? 0.0f : expf(sp[tid][kx] - m);
                sp[tid][kx] = e;
                l += e;
            }
            sm[tid] = m;
            sl[tid] = l;
            sfix[tid] = fix;
        }
        __syncthreads();
        #pragma unroll
        for (uint32_t g = 0; g < G; ++g) acc[g] *= sfix[g];
        const uint32_t cnt = min(T, k1 - t0);
        for (uint32_t kx = 0; kx < cnt; ++kx) {
            const float vv4 = pd_ik2_kv_ld1(vv, (size_t)srow[kx] * kv_dim + kh * HD + tid);
            #pragma unroll
            for (uint32_t g = 0; g < G; ++g) acc[g] = fmaf(sp[g][kx], vv4, acc[g]);
        }
        __syncthreads();
    }
    if (splits == 1u) {
        #pragma unroll
        for (uint32_t g = 0; g < G; ++g) out[((size_t)r * heads + h0 + g) * HD + tid] = acc[g] / sl[g];
        return;
    }
    #pragma unroll
    for (uint32_t g = 0; g < G; ++g) {
        const size_t o = ((size_t)r * heads + h0 + g) * splits + split;
        part_o[o * HD + tid] = acc[g];
        if (tid == 0) {
            part_ml[o * 2u] = sm[g];
            part_ml[o * 2u + 1u] = sl[g];
        }
    }
}

// (slot 990's second launch) fold the splits with the row's sink logit:
// M = max(sink, m_s), out = sum_s acc_s e^(m_s - M) / (e^(sink - M) + sum_s
// l_s e^(m_s - M)). grid (rows, heads), 128 threads.
__global__ void __launch_bounds__(128) pd_inkling2_attn_combine_kernel(
        const float* part_o, const float* part_ml, const float* sink, uint32_t ss,
        float* __restrict__ out, uint32_t heads, uint32_t splits) {
    constexpr uint32_t HD = 128u;
    PD_PDL_ARM();
    const uint32_t r = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
    const float* ml = pd_pdl_after(part_ml) + ((size_t)r * heads + h) * splits * 2u;
    const float* po = pd_pdl_after(part_o) + ((size_t)r * heads + h) * splits * HD;
    const float sk = pd_pdl_after(sink)[(size_t)r * ss + h];
    float M = sk;
    for (uint32_t s = 0; s < splits; ++s) M = fmaxf(M, ml[2u * s]);
    float den = expf(sk - M), a = 0.0f;
    for (uint32_t s = 0; s < splits; ++s) {
        const float m = ml[2u * s];
        if (m == -INFINITY) continue;
        const float f = expf(m - M);
        den = fmaf(ml[2u * s + 1u], f, den);
        a = fmaf(po[(size_t)s * HD + tid], f, a);
    }
    out[((size_t)r * heads + h) * HD + tid] = a / den;
}

PD_EXPORT
int pd_inkling2_attn_kv(const void* q, uint32_t qs, const void* rel, uint32_t rs, const void* sink,
                        uint32_t ss, void* kbase, void* vbase, const void* bt, uint32_t bps,
                        uint32_t ring, uint32_t dtype, const void* slots, const void* pos,
                        const void* proj, void* out, void* part_o, void* part_ml, uint32_t rows,
                        uint32_t heads, uint32_t kv_heads, uint32_t window, uint32_t extent,
                        uint32_t splits, float scale, void* stream) {
    if (rows == 0) return 0;
    if (kv_heads == 0 || heads % kv_heads || dtype > 2u || splits == 0 ||
        (splits > 1u && (part_o == nullptr || part_ml == nullptr)) ||
        (bt == nullptr && (ring == 0 || (ring & (ring - 1u)))))
        return cudaErrorInvalidValue;
    const uint32_t G = heads / kv_heads;
    const PdIk2Kv kk{kbase, (const uint32_t*)bt, bps, ring, dtype};
    const PdIk2Kv vv{vbase, (const uint32_t*)bt, bps, ring, dtype};
    const dim3 grid(rows, kv_heads, splits);
    #define PD_IK2_ATTN_KV(GG)                                                                    \
        pd_pdl_go(pd_inkling2_attn_kv_kernel<GG>, grid, 128, 0u, (cudaStream_t)stream,            \
                  (const float*)q, qs, (const float*)rel, rs, (const float*)sink, ss, kk, vv,     \
                  (const uint32_t*)slots, (const uint32_t*)pos, (const __nv_bfloat16*)proj,      \
                  (float*)out, (float*)part_o, (float*)part_ml, heads, kv_heads, window, extent,  \
                  scale)
    switch (G) {
    case 1: PD_IK2_ATTN_KV(1u); break;
    case 2: PD_IK2_ATTN_KV(2u); break;
    case 3: PD_IK2_ATTN_KV(3u); break;
    case 4: PD_IK2_ATTN_KV(4u); break;
    default: return cudaErrorInvalidValue;
    }
    #undef PD_IK2_ATTN_KV
    if (splits > 1u)
        pd_pdl_go(pd_inkling2_attn_combine_kernel, dim3(rows, heads), 128, 0u, (cudaStream_t)stream,
                  (const float*)part_o, (const float*)part_ml, (const float*)sink, ss, (float*)out,
                  heads, splits);
    return pd_launch_status();
}

// ---- slot 993: prompt rows on the tensor cores -----------------------------
// The relative-bias attention of runs of consecutive positions (prompt
// chunks) over a visit's f16 / fp8 stores. A CTA takes a tile of up to 32
// rows of one run and the G heads sharing a K / V head (a warp: 16 rows of
// one head), stages 64-key tiles of K and V in smem as f16 (exact from
// either class), and runs QK and PV on mma.sync m16n8k16 (f16 operands, f32
// accumulators) - the queries round to f16 on the way in, the probabilities
// to f16 before PV, the flash-attention practice; everything else (scale,
// masks, the online softmax seeded with the row's sink) is f32 as slot
// 990's. `tiles` [n][4]: (first row, rows, slot, first position).
//
// The relative bias rel[r] . bank[pos_r - key] is Transformer-XL's position
// term: a warp's 16 rows against a 64-key tile reach only 79 distances, so
// G = rel (16 x 16) x bank (16 x 80 distances) is ten bf16 mma tiles on the
// tensor cores (rel split hi + lo in bf16, the bank bf16 as stored: the
// products exact, ~16 bits of rel kept), parked in smem and gathered a score
// at a time - where every score used to take 16 CUDA-core FMAs and two
// 16-byte bank loads. The fp8 class lands its next tile's rows with cp.async
// while the current one computes (the f16 class loads in step).
#define PD_IK2T_BM 32u
#define PD_IK2T_BN 64u
#define PD_IK2T_LD 136u  // f16 row stride (128 + 8): conflict-free ldmatrix
// a warp's G: row r at r * 99 + (r & 1), so the 32 lanes' gathers of a score
// (rows g / g + 8, keys 2tq + c) land on 32 banks; 1586 floats a warp
#define PD_IK2T_GLD 99u
#define PD_IK2T_GW 1586u
#define PD_IK2T_SMEM(G, F8)                                                        \
    (2u * PD_IK2T_BN * PD_IK2T_LD * 2u + 2u * (G) * PD_IK2T_GW * 4u +             \
     ((F8) ? 2u * PD_IK2T_BN * 128u : 0u))

__device__ __forceinline__ void pd_ik2_ldm4(uint32_t r[4], const void* p) {
    const unsigned sp = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(sp));
}
__device__ __forceinline__ void pd_ik2_ldm4t(uint32_t r[4], const void* p) {
    const unsigned sp = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(sp));
}
__device__ __forceinline__ void pd_ik2_mma16(float d[4], const uint32_t a[4], uint32_t b0,
                                             uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pd_ik2_h2(float a, float b) {
    const __half2 h = __floats2half2_rn(a, b);
    return *(const uint32_t*)&h;
}
// 8 consecutive stored elements as f16 (class 0 or 1)
__device__ __forceinline__ uint4 pd_ik2_kv_ld8h(const PdIk2Kv& kv, size_t elem) {
    if (kv.dtype == 0u) return *(const uint4*)((const __half*)kv.base + elem);
    const uint2 v8 = *(const uint2*)((const uint8_t*)kv.base + elem);
    uint4 h4;
    __half2* h = (__half2*)&h4;
    h[0] = __half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(v8.x & 0xffffu), __NV_E4M3));
    h[1] = __half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(v8.x >> 16), __NV_E4M3));
    h[2] = __half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(v8.y & 0xffffu), __NV_E4M3));
    h[3] = __half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(v8.y >> 16), __NV_E4M3));
    return h4;
}

__device__ __forceinline__ void pd_ik2_mma16bf(float d[4], const uint32_t a[4], uint32_t b0,
                                               uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// 8 e4m3 codes as f16 (exact)
__device__ __forceinline__ uint4 pd_ik2_f8x8_h(uint2 v8) {
    uint4 h4;
    __half2* h = (__half2*)&h4;
    h[0] = __half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(v8.x & 0xffffu), __NV_E4M3));
    h[1] = __half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(v8.x >> 16), __NV_E4M3));
    h[2] = __half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(v8.y & 0xffffu), __NV_E4M3));
    h[3] = __half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(v8.y >> 16), __NV_E4M3));
    return h4;
}

template <uint32_t G, bool F8>
__global__ void __launch_bounds__(G * 64u) pd_inkling2_attn_tile_kernel(
        const float* q, uint32_t qs, const float* rel, uint32_t rs, const float* sink, uint32_t ss,
        PdIk2Kv kk, PdIk2Kv vv, const uint32_t* tiles, const __nv_bfloat16* __restrict__ projT,
        float* __restrict__ out, uint32_t heads, uint32_t kv_heads, uint32_t window,
        uint32_t extent, float scale, const float* gate, uint32_t gs, uint32_t out_bf16,
        float* __restrict__ part_o, float* __restrict__ part_ml) {
    constexpr uint32_t HD = 128u, DR = 16u, BN = PD_IK2T_BN, LD = PD_IK2T_LD;
    constexpr uint32_t GLD = PD_IK2T_GLD;
    constexpr uint32_t NT = G * 64u;
    PD_PDL_ARM();
    extern __shared__ __align__(16) unsigned char pd_ik2t_sh[];
    __half* ks = (__half*)pd_ik2t_sh;
    __half* vs = ks + BN * LD;
    __shared__ uint32_t krow[BN];
    const uint32_t* td = pd_pdl_after(tiles) + (size_t)blockIdx.x * 4u;
    const uint32_t row0 = td[0], nr = td[1], slot = td[2], p0 = td[3];
    const uint32_t kh = blockIdx.y, tid = threadIdx.x;
    const uint32_t warp = tid >> 5, lane = tid & 31u, g = lane >> 2, tq = lane & 3u;
    const uint32_t h = kh * G + warp / 2u;
    const uint32_t kv_dim = kv_heads * HD;
    float* gw = (float*)(vs + BN * LD) + warp * PD_IK2T_GW;
    unsigned char* stage = (unsigned char*)((float*)(vs + BN * LD) + 2u * G * PD_IK2T_GW);
    // this thread's two rows of its warp's 16 (tile-local), valid below nr
    const uint32_t rb = (warp & 1u) * 16u;
    const uint32_t ra = rb + g, rc = ra + 8u;
    const bool va = ra < nr, vc = rc < nr;
    const float* qp = pd_pdl_after(q);
    uint32_t qf[8][4];
    #pragma unroll
    for (uint32_t k16 = 0; k16 < 8u; ++k16) {
        #pragma unroll
        for (uint32_t i = 0; i < 4u; ++i) {
            const uint32_t row = (i & 1u) ? rc : ra;
            const uint32_t col = k16 * 16u + 2u * tq + (i >> 1) * 8u;
            float x0 = 0.0f, x1 = 0.0f;
            if (row < nr) {
                const float* qr = qp + (size_t)(row0 + row) * qs + h * HD + col;
                x0 = qr[0];
                x1 = qr[1];
            }
            qf[k16][i] = pd_ik2_h2(x0, x1);
        }
    }
    // the relative states as the bias GEMM's A fragments, hi + lo in bf16
    uint32_t ahi[4], alo[4];
    {
        const float* rp = pd_pdl_after(rel);
        #pragma unroll
        for (uint32_t i = 0; i < 4u; ++i) {
            const uint32_t row = (i & 1u) ? rc : ra;
            const uint32_t col = 2u * tq + (i >> 1) * 8u;
            float x0 = 0.0f, x1 = 0.0f;
            if (row < nr) {
                const float* rr = rp + (size_t)(row0 + row) * rs + h * DR + col;
                x0 = rr[0];
                x1 = rr[1];
            }
            const __nv_bfloat162 hb = __floats2bfloat162_rn(x0, x1);
            const float2 hf = __bfloat1622float2(hb);
            const __nv_bfloat162 lb = __floats2bfloat162_rn(x0 - hf.x, x1 - hf.y);
            ahi[i] = *(const uint32_t*)&hb;
            alo[i] = *(const uint32_t*)&lb;
        }
    }
    // the sink seeds each row: m = its logit, l = 1 (held by one thread of
    // the quad; the partial sums fold at the end). Split over keys (slot
    // 1009, gridDim.z > 1): each CTA a BN-aligned share of the tile's keys,
    // no sink (the combine folds it once), m from -inf
    const uint32_t splits = gridDim.z, split = blockIdx.z;
    const float* sp = pd_pdl_after(sink);
    float ma = va && splits == 1u ? sp[(size_t)(row0 + ra) * ss + h] : 0.0f;
    float mc = vc && splits == 1u ? sp[(size_t)(row0 + rc) * ss + h] : 0.0f;
    float la = tq == 0u && splits == 1u ? 1.0f : 0.0f, lc = la;
    if (splits > 1u) ma = mc = -INFINITY;
    float o[16][4];
    #pragma unroll
    for (uint32_t d = 0; d < 16u; ++d) o[d][0] = o[d][1] = o[d][2] = o[d][3] = 0.0f;
    const uint32_t pa = p0 + ra, pc = p0 + rc;
    const uint32_t lo0 = window ? (p0 + 1u > window ? p0 + 1u - window : 0u) : 0u;
    const uint32_t hi0 = p0 + nr;
    const uint32_t per = ((hi0 - lo0 + splits - 1u) / splits + BN - 1u) / BN * BN;
    const uint32_t lo = min(hi0, lo0 + split * per);
    const uint32_t hi = min(hi0, lo + per);
    // fp8: a tile's K and V rows (128 B each) into the stage, PAST hi zeros
    auto issue = [&](uint32_t k0) {
        for (uint32_t u = tid; u < BN * 16u; u += NT) {
            const uint32_t key = u >> 4, part = u & 15u, c16 = (part & 7u) * 16u;
            const bool ok = k0 + key < hi;
            const size_t row = ok ? pd_ik2_kv_row(kk, slot, k0 + key) : 0u;
            const unsigned char* src =
                (const unsigned char*)(part < 8u ? kk.base : vv.base) + row * kv_dim + kh * HD + c16;
            pd_cp_async16(stage + (part < 8u ? 0u : BN * 128u) + key * 128u + c16, src, ok);
        }
        asm volatile("cp.async.commit_group;");
    };
    if constexpr (F8) {
        if (lo < hi) issue(lo);
    }
    for (uint32_t k0 = lo; k0 < hi; k0 += BN) {
        if constexpr (F8) {
            asm volatile("cp.async.wait_group 0;");
            // every thread's copies landed, every warp is past the last tile
            __syncthreads();
            for (uint32_t u = tid; u < BN * 16u; u += NT) {
                const uint32_t key = u >> 4, c8 = (u & 15u) * 8u;
                *(uint4*)(ks + key * LD + c8) = pd_ik2_f8x8_h(*(const uint2*)(stage + key * 128u + c8));
                *(uint4*)(vs + key * LD + c8) =
                    pd_ik2_f8x8_h(*(const uint2*)(stage + BN * 128u + key * 128u + c8));
            }
            __syncthreads();
            if (k0 + BN < hi) issue(k0 + BN);
        } else {
            if (tid < BN) krow[tid] = k0 + tid < hi ? (uint32_t)pd_ik2_kv_row(kk, slot, k0 + tid) : 0u;
            __syncthreads();
            for (uint32_t u = tid; u < BN * 16u; u += NT) {
                const uint32_t key = u >> 4, c8 = (u & 15u) * 8u;
                uint4 kx = make_uint4(0, 0, 0, 0), vx = kx;
                if (k0 + key < hi) {
                    const size_t e = (size_t)krow[key] * kv_dim + kh * HD + c8;
                    kx = pd_ik2_kv_ld8h(kk, e);
                    vx = pd_ik2_kv_ld8h(vv, e);
                }
                *(uint4*)(ks + key * LD + c8) = kx;
                *(uint4*)(vs + key * LD + c8) = vx;
            }
            __syncthreads();
        }
        // the warp's bias rows: G[r][c] = rel[rb + r] . bank[D0 + c], c < 80;
        // distances past the bank (or negative: masked) stay zero
        const int D0 = (int)(p0 + rb) - (int)k0 - 63;
        const bool bias = D0 < (int)extent && D0 + 78 >= 0;
        if (bias) {
            const uint32_t* pt = (const uint32_t*)projT;
            #pragma unroll
            for (uint32_t t = 0; t < 10u; ++t) {
                const int d = D0 + (int)(8u * t + g);
                uint32_t b0 = 0u, b1 = 0u;
                if (d >= 0 && d < (int)extent) {
                    b0 = __ldg(pt + (size_t)d * 8u + tq);
                    b1 = __ldg(pt + (size_t)d * 8u + 4u + tq);
                }
                float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                pd_ik2_mma16bf(acc, ahi, b0, b1);
                pd_ik2_mma16bf(acc, alo, b0, b1);
                const uint32_t c = 8u * t + 2u * tq;
                *(float2*)(gw + g * GLD + (g & 1u) + c) = make_float2(acc[0], acc[1]);
                *(float2*)(gw + (g + 8u) * GLD + (g & 1u) + c) = make_float2(acc[2], acc[3]);
            }
            __syncwarp();
        }
        float s[8][4];
        #pragma unroll
        for (uint32_t nt = 0; nt < 8u; ++nt) {
            s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.0f;
            #pragma unroll
            for (uint32_t kp = 0; kp < 4u; ++kp) {
                uint32_t b[4];
                pd_ik2_ldm4(b, ks + (nt * 8u + (lane & 7u)) * LD + kp * 32u + (lane >> 3) * 8u);
                pd_ik2_mma16(s[nt], qf[2u * kp], b[0], b[1]);
                pd_ik2_mma16(s[nt], qf[2u * kp + 1u], b[2], b[3]);
            }
        }
        // scale, the relative bias, masks; the tile max a row
        float ta = -INFINITY, tc = -INFINITY;
        #pragma unroll
        for (uint32_t nt = 0; nt < 8u; ++nt) {
            #pragma unroll
            for (uint32_t i = 0; i < 4u; ++i) {
                const bool up = i >= 2u;
                const uint32_t pr = up ? pc : pa;
                const uint32_t j = nt * 8u + 2u * tq + (i & 1u);
                const uint32_t key = k0 + j;
                const bool ok = (up ? vc : va) && key < hi && key <= pr &&
                                (window == 0u || pr - key < window);
                float x = -INFINITY;
                if (ok) {
                    x = s[nt][i] * scale;
                    if (bias) {
                        const uint32_t r = up ? g + 8u : g;
                        x += gw[r * GLD + (g & 1u) + r + 63u - j];
                    }
                }
                s[nt][i] = x;
                if (up) tc = fmaxf(tc, x);
                else ta = fmaxf(ta, x);
            }
        }
        ta = fmaxf(ta, __shfl_xor_sync(0xffffffffu, ta, 1));
        ta = fmaxf(ta, __shfl_xor_sync(0xffffffffu, ta, 2));
        tc = fmaxf(tc, __shfl_xor_sync(0xffffffffu, tc, 1));
        tc = fmaxf(tc, __shfl_xor_sync(0xffffffffu, tc, 2));
        // m starts at the sink's (finite) logit, so the new max is finite;
        // a split's rows may see no key yet (m -inf: nothing to rescale, and
        // every score of theirs is -inf too)
        const float na = fmaxf(ma, ta), nc = fmaxf(mc, tc);
        const float fa = na == -INFINITY ? 1.0f : expf(ma - na);
        const float fc = nc == -INFINITY ? 1.0f : expf(mc - nc);
        ma = na;
        mc = nc;
        const float ea = ma == -INFINITY ? 0.0f : ma, ec = mc == -INFINITY ? 0.0f : mc;
        la *= fa;
        lc *= fc;
        #pragma unroll
        for (uint32_t d = 0; d < 16u; ++d) {
            o[d][0] *= fa;
            o[d][1] *= fa;
            o[d][2] *= fc;
            o[d][3] *= fc;
        }
        #pragma unroll
        for (uint32_t nt = 0; nt < 8u; ++nt) {
            s[nt][0] = expf(s[nt][0] - ea);
            s[nt][1] = expf(s[nt][1] - ea);
            s[nt][2] = expf(s[nt][2] - ec);
            s[nt][3] = expf(s[nt][3] - ec);
            la += s[nt][0] + s[nt][1];
            lc += s[nt][2] + s[nt][3];
        }
        // O += P V, 16 keys a step
        #pragma unroll
        for (uint32_t j = 0; j < 4u; ++j) {
            const uint32_t a[4] = {pd_ik2_h2(s[2u * j][0], s[2u * j][1]),
                                   pd_ik2_h2(s[2u * j][2], s[2u * j][3]),
                                   pd_ik2_h2(s[2u * j + 1u][0], s[2u * j + 1u][1]),
                                   pd_ik2_h2(s[2u * j + 1u][2], s[2u * j + 1u][3])};
            #pragma unroll
            for (uint32_t dt = 0; dt < 16u; dt += 2u) {
                uint32_t b[4];
                pd_ik2_ldm4t(b, vs + (j * 16u + (lane & 7u) + ((lane >> 3) & 1u) * 8u) * LD +
                                    dt * 8u + (lane >> 4) * 8u);
                pd_ik2_mma16(o[dt], a, b[0], b[1]);
                pd_ik2_mma16(o[dt + 1u], a, b[2], b[3]);
            }
        }
        // the f16 class's next tile overwrites ks / vs and the warps' G rows
        // (fp8: the barrier after the stage wait does it)
        if constexpr (!F8) __syncthreads();
    }
    la += __shfl_xor_sync(0xffffffffu, la, 1);
    la += __shfl_xor_sync(0xffffffffu, la, 2);
    lc += __shfl_xor_sync(0xffffffffu, lc, 1);
    lc += __shfl_xor_sync(0xffffffffu, lc, 2);
    if (splits > 1u) {
        // slot 990's partial layout: (row x heads + head) x splits + split
        auto part = [&](uint32_t row, float m, float l, uint32_t c, float x0, float x1) {
            const size_t at = ((size_t)(row0 + row) * heads + h) * splits + split;
            part_o[at * HD + c] = x0;
            part_o[at * HD + c + 1u] = x1;
            if (c == 0u) {
                part_ml[at * 2u] = m;
                part_ml[at * 2u + 1u] = l;
            }
        };
        #pragma unroll
        for (uint32_t d = 0; d < 16u; ++d) {
            const uint32_t col = d * 8u + 2u * tq;
            if (va) part(ra, ma, la, col, o[d][0], o[d][1]);
            if (vc) part(rc, mc, lc, col, o[d][2], o[d][3]);
        }
        return;
    }
    // slot 1005: the output gate x sigmoid(g) here, as mul_sigmoid computes
    // it; with out_bf16 the rows land as the bf16 the next GEMM narrows to
    const float* gp = gate ? pd_pdl_after(gate) : nullptr;
    auto put = [&](uint32_t row, uint32_t col, float x0, float x1) {
        if (gp) {
            const float2 gv = *(const float2*)(gp + (size_t)(row0 + row) * gs + h * HD + col);
            x0 *= 1.0f / (1.0f + expf(-gv.x));
            x1 *= 1.0f / (1.0f + expf(-gv.y));
        }
        const size_t at = ((size_t)(row0 + row) * heads + h) * HD + col;
        if (out_bf16) {
            *(__nv_bfloat162*)((__nv_bfloat16*)out + at) = __floats2bfloat162_rn(x0, x1);
        } else {
            out[at] = x0;
            out[at + 1u] = x1;
        }
    };
    #pragma unroll
    for (uint32_t d = 0; d < 16u; ++d) {
        const uint32_t col = d * 8u + 2u * tq;
        if (va) put(ra, col, o[d][0] / la, o[d][1] / la);
        if (vc) put(rc, col, o[d][2] / lc, o[d][3] / lc);
    }
}

static int pd_inkling2_attn_tile_impl(const void* q, uint32_t qs, const void* rel, uint32_t rs,
                                      const void* sink, uint32_t ss, void* kbase, void* vbase,
                                      const void* bt, uint32_t bps, uint32_t ring, uint32_t dtype,
                                      const void* tiles, uint32_t n_tiles, const void* projT,
                                      void* out, uint32_t heads, uint32_t kv_heads,
                                      uint32_t window, uint32_t extent, float scale,
                                      const void* gate, uint32_t gstride, uint32_t out_bf16,
                                      void* part_o, void* part_ml, uint32_t splits,
                                      uint32_t rows, void* stream) {
    if (n_tiles == 0) return 0;
    if (splits == 0u || splits > 64u ||
        (splits > 1u && (!part_o || !part_ml || gate || out_bf16 || rows == 0u)))
        return cudaErrorInvalidValue;
    // the f16 / fp8 classes (the f32 reference class keeps slot 990); up to
    // 3 query heads a K / V head (4 would take past the opt-in smem)
    if (kv_heads == 0 || heads % kv_heads || heads / kv_heads > 3u || dtype > 1u ||
        (bt == nullptr && (ring == 0 || (ring & (ring - 1u)))))
        return cudaErrorInvalidValue;
    const PdIk2Kv kk{kbase, (const uint32_t*)bt, bps, ring, dtype};
    const PdIk2Kv vv{vbase, (const uint32_t*)bt, bps, ring, dtype};
    const dim3 grid(n_tiles, kv_heads, splits);
    #define PD_IK2_ATTN_TILE(GG, F)                                                               \
        do {                                                                                      \
            auto k = pd_inkling2_attn_tile_kernel<GG, F>;                                         \
            constexpr uint32_t smem = PD_IK2T_SMEM(GG, F);                                        \
            static const bool attr = cudaFuncSetAttribute(                                        \
                k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) == cudaSuccess ||      \
                (cudaGetLastError(), false);                                                      \
            if (!attr) return cudaErrorInvalidConfiguration;                                      \
            pd_pdl_go(k, grid, GG * 64u, smem, (cudaStream_t)stream, (const float*)q, qs,         \
                      (const float*)rel, rs, (const float*)sink, ss, kk, vv,                      \
                      (const uint32_t*)tiles, (const __nv_bfloat16*)projT, (float*)out, heads,   \
                      kv_heads, window, extent, scale, (const float*)gate, gstride, out_bf16,    \
                      (float*)part_o, (float*)part_ml);                                           \
        } while (0)
    const bool f8 = dtype == 1u;
    switch (heads / kv_heads) {
    case 1: if (f8) PD_IK2_ATTN_TILE(1u, true); else PD_IK2_ATTN_TILE(1u, false); break;
    case 2: if (f8) PD_IK2_ATTN_TILE(2u, true); else PD_IK2_ATTN_TILE(2u, false); break;
    default: if (f8) PD_IK2_ATTN_TILE(3u, true); else PD_IK2_ATTN_TILE(3u, false); break;
    }
    #undef PD_IK2_ATTN_TILE
    if (splits > 1u)
        pd_pdl_go(pd_inkling2_attn_combine_kernel, dim3(rows, heads), 128, 0u, (cudaStream_t)stream,
                  (const float*)part_o, (const float*)part_ml, (const float*)sink, ss, (float*)out,
                  heads, splits);
    return pd_launch_status();
}

PD_EXPORT
int pd_inkling2_attn_tile(const void* q, uint32_t qs, const void* rel, uint32_t rs, const void* sink,
                          uint32_t ss, void* kbase, void* vbase, const void* bt, uint32_t bps,
                          uint32_t ring, uint32_t dtype, const void* tiles, uint32_t n_tiles,
                          const void* projT, void* out, uint32_t heads, uint32_t kv_heads,
                          uint32_t window, uint32_t extent, float scale, void* stream) {
    return pd_inkling2_attn_tile_impl(q, qs, rel, rs, sink, ss, kbase, vbase, bt, bps, ring, dtype,
                                      tiles, n_tiles, projT, out, heads, kv_heads, window, extent,
                                      scale, nullptr, 0u, 0u, nullptr, nullptr, 1u, 0u, stream);
}

// slot 1005: slot 993 with the attention's output gate folded in, out =
// attention x sigmoid(gate) (gate [rows][heads x 128] at row stride gstride,
// the rows of the tiles') - mul_sigmoid's arithmetic, one pass fewer; with
// out_bf16 the rows land as bf16 (convert_f32_bf16 of them, for the o
// projection's prefill pair).
PD_EXPORT
int pd_inkling2_attn_tile_g(const void* q, uint32_t qs, const void* rel, uint32_t rs,
                            const void* sink, uint32_t ss, void* kbase, void* vbase, const void* bt,
                            uint32_t bps, uint32_t ring, uint32_t dtype, const void* tiles,
                            uint32_t n_tiles, const void* projT, void* out, uint32_t heads,
                            uint32_t kv_heads, uint32_t window, uint32_t extent, float scale,
                            const void* gate, uint32_t gstride, uint32_t out_bf16, void* stream) {
    if (!gate || out_bf16 > 1u) return cudaErrorInvalidValue;
    return pd_inkling2_attn_tile_impl(q, qs, rel, rs, sink, ss, kbase, vbase, bt, bps, ring, dtype,
                                      tiles, n_tiles, projT, out, heads, kv_heads, window, extent,
                                      scale, gate, gstride, out_bf16, nullptr, nullptr, 1u, 0u,
                                      stream);
}

// slot 1009: slot 993 split over keys for passes of few tiles (a verify
// pass's runs at a long context: a CTA a K / V head walked every key) -
// each (tile, K / V head, split) a BN-aligned share of the keys into slot
// 990's partials, folded with the sinks by its combine over rows [0, rows).
// The tiles must cover every one of those rows; splits <= 64, rows x splits
// within the caller's partials.
PD_EXPORT
int pd_inkling2_attn_tile_split(const void* q, uint32_t qs, const void* rel, uint32_t rs,
                                const void* sink, uint32_t ss, void* kbase, void* vbase,
                                const void* bt, uint32_t bps, uint32_t ring, uint32_t dtype,
                                const void* tiles, uint32_t n_tiles, const void* projT, void* out,
                                void* part_o, void* part_ml, uint32_t rows, uint32_t heads,
                                uint32_t kv_heads, uint32_t window, uint32_t extent, float scale,
                                uint32_t splits, void* stream) {
    return pd_inkling2_attn_tile_impl(q, qs, rel, rs, sink, ss, kbase, vbase, bt, bps, ring, dtype,
                                      tiles, n_tiles, projT, out, heads, kv_heads, window, extent,
                                      scale, nullptr, 0u, 0u, part_o, part_ml, splits, rows,
                                      stream);
}
