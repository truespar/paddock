// elementwise.cuh (formerly 02_elementwise.cuh) - dequant, rmsnorm, rope, softmax-sink, swiglu - scalar elementwise kernels
// Textually-included segment of the single pack translation unit.
// Not standalone-compilable: include order is defined by ../pack.cu.
// ------------------------------------------------------------------- kernels

// FP4 (E2M1) values as stored in MXFP4 nibbles - doubled, compensated by the
// halved E8M0 scale (the ggml/GGUF on-disk convention).
__constant__ float PD_FP4_VALUES[16] = {
    0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 12.0f,
    0.0f, -1.0f, -2.0f, -3.0f, -4.0f, -6.0f, -8.0f, -12.0f,
};

// E8M0 byte -> 2^(e-127) * 0.5, bit-exact with the CPU reference
// (paddock-kernels::reference::e8m0_half_to_f32).
__device__ __forceinline__ float pd_e8m0_half(uint8_t e) {
    uint32_t bits = (e < 2) ? (0x00200000u << e) : ((uint32_t)(e - 1) << 23);
    return __uint_as_float(bits);
}

// The same doubled FP4 values as PD_FP4_VALUES, but as int8 (they're integers) -
// the weight operand for the dp4a integer dot on the MXFP4 path.
__constant__ signed char PD_FP4_INT[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
};

// Unpack 8 MXFP4 nibbles (packed in q4) into 8 int8 values via __byte_perm - a
// 16-entry table lookup with no shared/constant LUT reads (so no bank conflicts
// or serialization). Returns int2: .x = the 4 low-nibble values (in order), .y =
// the 4 high-nibble values. Technique learned from ggml get_int_from_table_16;
// implementation ours. table[0..3] are the 16 int8 values as four uint32.
__device__ __forceinline__ int2 pd_fp4_unpack8(uint32_t q4) {
    const uint32_t* t = (const uint32_t*)PD_FP4_INT;
    uint32_t tmp[2];
    const uint32_t sel = 0x32103210u | ((q4 & 0x88888888u) >> 1);
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        uint32_t shift = 16u * i;
        uint32_t low = __byte_perm(t[0], t[1], q4 >> shift);
        uint32_t high = __byte_perm(t[2], t[3], q4 >> shift);
        tmp[i] = __byte_perm(low, high, sel >> shift);
    }
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
}

// MXFP4 block: 1 byte E8M0 scale + 16 bytes packed nibbles; low nibble of
// qs[j] is element j, high nibble is element j+16. One thread per element.
__global__ void pd_mxfp4_dequant_kernel(const uint8_t* __restrict__ in,
                                        float* __restrict__ out,
                                        uint64_t total_elems) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_elems) return;

    uint64_t block = i >> 5;          // / 32
    uint32_t elem  = (uint32_t)(i & 31);

    const uint8_t* blk = in + block * 17;
    float d = pd_e8m0_half(blk[0]);
    uint8_t packed = blk[1 + (elem & 15)];
    uint8_t nib = (elem < 16) ? (packed & 0x0F) : (packed >> 4);
    out[i] = PD_FP4_VALUES[nib] * d;
}

// Q8_0 block: f16 scale + 32 signed bytes. One thread per element.
__global__ void pd_q8_0_dequant_kernel(const uint8_t* __restrict__ in,
                                       float* __restrict__ out,
                                       uint64_t total_elems) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_elems) return;

    uint64_t block = i >> 5;
    uint32_t elem  = (uint32_t)(i & 31);

    const uint8_t* blk = in + block * 34;
    __half h;
    memcpy(&h, blk, sizeof(h));
    float d = __half2float(h);
    out[i] = (float)((int8_t)blk[2 + elem]) * d;
}

// RMSNorm, one block, shared-memory tree reduction. Correctness-first shape;
// fused/vectorized variants come with the perf pass.
__global__ void pd_rmsnorm_kernel(const float* __restrict__ x,
                                  const float* __restrict__ w,
                                  float* __restrict__ out,
                                  uint32_t n, float eps) {
    __shared__ float sred[256];
    __shared__ float s_inv;
    float acc = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        acc += x[i] * x[i];
    }
    sred[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sred[threadIdx.x] += sred[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        s_inv = 1.0f / sqrtf(sred[0] / (float)n + eps);
    }
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        out[i] = x[i] * s_inv * w[i];
    }
}

// YaRN NEOX rope: one thread per head, iterating pairs with the same
// multiplicative theta chain as the CPU reference (keeps parity tight; a
// pair-parallel version is a perf-pass change).
__global__ void pd_rope_yarn_kernel(float* __restrict__ x,
                                    uint32_t n_heads, uint32_t head_dim, uint32_t pos,
                                    float theta_scale, float freq_scale,
                                    float corr_low, float corr_high,
                                    float ext_factor, float mscale) {
    uint32_t h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= n_heads) return;
    float* head = x + (size_t)h * head_dim;
    uint32_t half = head_dim / 2;
    float theta = (float)pos;
    for (uint32_t k = 0; k < half; ++k) {
        float y = ((float)k - corr_low) / fmaxf(0.001f, corr_high - corr_low);
        float ramp = (1.0f - fminf(1.0f, fmaxf(0.0f, y))) * ext_factor;
        float angle = (freq_scale * theta) * (1.0f - ramp) + theta * ramp;
        float s = sinf(angle) * mscale;
        float c = cosf(angle) * mscale;
        float a = head[k];
        float b = head[k + half];
        head[k] = a * c - b * s;
        head[k + half] = a * s + b * c;
        theta *= theta_scale;
    }
}

// Softmax with sink, one block: max-reduce (sink joins), exp+sum-reduce
// (sink joins denominator only), scale.
__global__ void pd_softmax_sink_kernel(float* __restrict__ scores, uint32_t n, float sink) {
    __shared__ float sred[256];
    __shared__ float s_max;
    __shared__ float s_denom;
    float m = -INFINITY;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        m = fmaxf(m, scores[i]);
    }
    sred[threadIdx.x] = m;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sred[threadIdx.x] = fmaxf(sred[threadIdx.x], sred[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) s_max = fmaxf(sred[0], sink);
    __syncthreads();

    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        float e = expf(scores[i] - s_max);
        scores[i] = e;
        sum += e;
    }
    sred[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sred[threadIdx.x] += sred[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) s_denom = sred[0] + expf(sink - s_max);
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        scores[i] /= s_denom;
    }
}

// swiglu_oai elementwise, in-place on gate.
__global__ void pd_swiglu_oai_kernel(float* __restrict__ gate,
                                     const float* __restrict__ up,
                                     uint32_t n, float alpha, float limit) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float xg = fminf(gate[i], limit);
    float yu = fminf(fmaxf(up[i], -limit), limit);
    gate[i] = (xg / (1.0f + expf(-alpha * xg))) * (yu + 1.0f);
}

__global__ void pd_add_inplace_kernel(float* __restrict__ x,
                                      const float* __restrict__ y, uint32_t n) {
    // cascade (laguna chain arming): y is always the
    // immediately-preceding producer's output. No-op under plain launches.
    PD_PDL_ARM();
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

// GEGLU elementwise, in-place on gate: gate = gelu_tanh(gate) * up. The GELU
// constant + form are exactly ggml_gelu_f32 (same as pd_gelu) - gemma4's
// LLM_FFN_GELU/LLM_FFN_PAR pair, fused so the FFN pays one launch not two.
__global__ void pd_geglu_kernel(float* __restrict__ gate, const float* __restrict__ up,
                                uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    float gelu = 0.5f * g * (1.0f + tanhf(0.79788456080286535587989211986876f * g
                                          * (1.0f + 0.044715f * g * g)));
    gate[i] = gelu * up[i];
}

__global__ void pd_scale_add_kernel(float* __restrict__ x,
                                    const float* __restrict__ y, float w, uint32_t n) {
    PD_PDL_ARM();  // cascade (granite chain)
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += w * y[i];
}

// x[i] *= s - the standalone scalar multiply, ggml_scale's shape. Granite's
// embedding_multiplier (x12) and logits_scaling (/16) are exactly this; its
// residual_multiplier rides pd_scale_add_f32 (x += w*y) instead. Deliberately
// a separate kernel rather than pd_scale_add_kernel with y aliased onto x:
// both pointers there are __restrict__, so aliasing them is UB no matter how
// benign the elementwise body looks.
__global__ void pd_scale_kernel(float* __restrict__ x, float s, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s;
}

// Batched multi-head attention, one decode query token. One block per query
// head; blockDim.x = head_dim. Online (flash-style) softmax with a per-head
// sink logit (joins max + denominator, no value). GQA via kvh = h / group.
// Correctness-first shape (one launch replaces 192); flash-tiled variant later.
__global__ void pd_attn_decode_kernel(
    const float* __restrict__ q, const __half* __restrict__ kc,
    const __half* __restrict__ vc, const float* __restrict__ sinks,
    float* __restrict__ out,
    uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
    uint32_t first_pos, uint32_t n_pos, uint32_t kv_dim, float scale)
{
    uint32_t h = blockIdx.x;
    if (h >= n_heads) return;
    uint32_t d = threadIdx.x;                 // 0..head_dim-1
    uint32_t group = n_heads / n_kv_heads;
    uint32_t kvh = h / group;
    uint32_t n_warps = head_dim >> 5;         // head_dim a multiple of 32

    __shared__ float red[32];                 // one slot per warp
    __shared__ float s_m, s_l, s_score;

    float qd = q[(size_t)h * head_dim + d];
    float acc = 0.0f;
    if (d == 0) { s_m = sinks[h]; s_l = 1.0f; } // sink init: m=sink, l=exp(0)=1
    __syncthreads();

    for (uint32_t i = 0; i < n_pos; ++i) {
        size_t base = (size_t)(first_pos + i) * kv_dim + (size_t)kvh * head_dim;
        // warp-shuffle dot reduction (replaces the head_dim-step shared tree)
        float v = qd * __half2float(kc[base + d]);
        for (uint32_t off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
        if ((d & 31u) == 0) red[d >> 5] = v;
        __syncthreads();
        if (d == 0) {
            float sc = 0.0f;
            for (uint32_t w = 0; w < n_warps; ++w) sc += red[w];
            s_score = sc * scale;
        }
        __syncthreads();

        float score = s_score;
        float m_old = s_m;
        float m_new = fmaxf(m_old, score);
        float corr = __expf(m_old - m_new);
        float w = __expf(score - m_new);
        acc = acc * corr + w * __half2float(vc[base + d]);
        __syncthreads();
        if (d == 0) { s_l = s_l * corr + w; s_m = m_new; }
        __syncthreads();
    }
    out[(size_t)h * head_dim + d] = acc / s_l;
}

// Tiled online-softmax KV walk shared by the batched decode-attention kernels.
// Stages PD_ATTN_TILE positions of K and V through shared memory with block-
// coalesced loads, scores them all in parallel (thread p owns position p), and
// applies one online-softmax rescale per tile. The per-position walk this
// replaces paid 4 __syncthreads() and a full un-overlapped DRAM round trip per
// position (measured ~1.5 us/position on GB202 - 9-20 GB/s effective); the
// tile walk pays the same 4 syncs per 32 positions and the staging loop keeps
// ~32 loads in flight per thread. Numeric class: per-32-key-tile update order,
// the same class as the f16 prefill attention (not bit-identical to the
// per-position walk). Deterministic: fixed tile order, fixed reduce order.
// Contract: blockDim.x >= head_dim (both multiples of 32) - the extra threads
// join the staging loop (the memory phase, where a head_dim-sized block leaves
// the SM short of loads in flight) and idle through the dim-bound math; shared
// layout below; s_m/s_l initialized by the caller (thread 0) before the first
// call; per-thread acc only meaningful for threadIdx.x < head_dim.
// smem/s_m/s_l are deliberately not __restrict__: thread 0 updates *s_m/*s_l
// each tile and every thread re-reads them the next tile - restrict let ptxas
// cache the load across __syncthreads() and every thread but 0 kept tile 1's
// running max (0.05-0.13 abs logit error on any context past one tile).
#define PD_ATTN_TILE 32u
// hd=256 walkers take 16-position tiles: the 32-tile carveout is ~67 KB -
// one block per SM - and decode-class attention at short context becomes
// wave-after-wave of near-empty blocks (measured 53.6 us/layer at B=8
// ctx<=136 on the 35B, ~0.85 ms/step). 16 halves the carveout (~34 KB, two
// blocks/SM). NUMERIC CLASS NOTE: the online-softmax fold is per-TILE, so
// tile size is part of each qwen model's accumulation order (all qwen
// full-attn heads are 256-dim); gpt-oss (hd 64) keeps the 32-tile class
// untouched. Same-path A/B gates hold on both sides by construction.
#define PD_ATTN_TILE_HD256 16u
#define PD_ATTN_TILE_FOR(hd) ((hd) > 128u ? PD_ATTN_TILE_HD256 : PD_ATTN_TILE)
// dynamic-shared bytes the tile walk needs at a given head_dim
#define PD_ATTN_TILE_SMEM(hd) \
    (((hd) + 2u * PD_ATTN_TILE_FOR(hd) * ((hd) + 1u) + 2u * PD_ATTN_TILE_FOR(hd)) * \
     sizeof(float))
// The tile walks' score step: s_scores[p] = scale * (q . k_p) over the
// tile's n_t staged keys, between the walks' barriers. One body for the
// dense and paged walks, so both keep one summation order per PS.
template<uint32_t TILE, bool PS>
__device__ __forceinline__ void pd_attn_tile_scores(const float* s_q, const float* s_k,
                                                    float* s_scores, uint32_t n_t,
                                                    uint32_t head_dim, float scale) {
    const uint32_t d = threadIdx.x, nth = blockDim.x;
    if (PS) {
        // PARALLEL SCORE. The serial form below leaves n_t of nth threads
        // working - 16 of 256 at head_dim 256 - while the other seven warps
        // sit at the next __syncthreads() for the whole 256-step dot
        // product. ncu on the shipped kernel: 41.5% of the average 23.0
        // cycles between issued instructions is stall_barrier, its own rule
        // naming "diverging code paths before a barrier" at an estimated
        // 41.48% local speedup. DRAM throughput is 1.23% and achieved
        // occupancy 58.9%, so neither bandwidth nor occupancy binds here -
        // the idle warps do.
        //
        // Here every thread works: LPK threads cooperate on one key, each
        // striding the head, then a width-LPK shuffle reduces. The partial
        // sums are interleaved rather than sequential, so this is not
        // bit-identical to the serial walk - hence a template flag, with
        // every existing caller keeping PS=false.
        const uint32_t lpk = nth / TILE;
        if (lpk >= 2u) {
            const uint32_t key = d / lpk, lane = d % lpk;
            float sc = 0.0f;
            if (key < n_t) {
                const float* krow = s_k + key * (head_dim + 1u);
                for (uint32_t dd = lane; dd < head_dim; dd += lpk)
                    sc += s_q[dd] * krow[dd];
            }
            #pragma unroll
            for (uint32_t o = 16u; o >= 1u; o >>= 1)
                if (o < lpk) sc += __shfl_down_sync(0xffffffffu, sc, o, lpk);
            if (key < n_t && lane == 0u) s_scores[key] = sc * scale;
        } else if (d < n_t) {
            float sc = 0.0f;
            const float* krow = s_k + d * (head_dim + 1u);
            for (uint32_t dd = 0; dd < head_dim; ++dd) sc += s_q[dd] * krow[dd];
            s_scores[d] = sc * scale;
        }
    } else if (d < n_t) {
        float sc = 0.0f;
        const float* krow = s_k + d * (head_dim + 1u);
        for (uint32_t dd = 0; dd < head_dim; ++dd) sc += s_q[dd] * krow[dd];
        s_scores[d] = sc * scale;
    }
}

template<typename KV, uint32_t TILE = PD_ATTN_TILE, bool PS = false>
__device__ __forceinline__ void pd_attn_tile_walk(
    const KV* __restrict__ kcb, const KV* __restrict__ vcb, uint32_t first_pos,
    uint32_t lo, uint32_t hi, uint32_t kv_dim, uint32_t kvh, uint32_t head_dim,
    float scale, float* smem, float* s_m, float* s_l, float& acc) {
    uint32_t d = threadIdx.x, nth = blockDim.x;
    bool dim_active = d < head_dim;
    float* s_q = smem;                                      // [head_dim]
    float* s_k = s_q + head_dim;                            // [TILE][head_dim+1]
    float* s_v = s_k + TILE * (head_dim + 1u);      // [TILE][head_dim+1]
    float* s_scores = s_v + TILE * (head_dim + 1u); // [TILE]
    float* s_w = s_scores + TILE;                   // [TILE]

    for (uint32_t t0 = lo; t0 < hi; t0 += TILE) {
        uint32_t n_t = hi - t0 < TILE ? hi - t0 : TILE;
        __syncthreads();  // prior tile's s_v reads complete before overwrite
        uint32_t hd4 = head_dim >> 2;
        for (uint32_t idx = d; idx < n_t * hd4; idx += nth) {
            uint32_t p = idx / hd4, dd = (idx - p * hd4) << 2;
            size_t base = (size_t)(first_pos + t0 + p) * kv_dim + (size_t)kvh * head_dim + dd;
            float4 kf = pd_kv_load4(kcb + base);
            float4 vf = pd_kv_load4(vcb + base);
            float* kr = s_k + p * (head_dim + 1u) + dd;
            float* vr = s_v + p * (head_dim + 1u) + dd;
            kr[0] = kf.x; kr[1] = kf.y; kr[2] = kf.z; kr[3] = kf.w;
            vr[0] = vf.x; vr[1] = vf.y; vr[2] = vf.z; vr[3] = vf.w;
        }
        __syncthreads();
        pd_attn_tile_scores<TILE, PS>(s_q, s_k, s_scores, n_t, head_dim, scale);
        __syncthreads();
        // every thread derives the same m_new from shared (broadcast reads)
        float m_old = *s_m;
        float m_tile = -INFINITY;
        for (uint32_t p = 0; p < n_t; ++p) m_tile = fmaxf(m_tile, s_scores[p]);
        float m_new = fmaxf(m_old, m_tile);
        float corr = __expf(m_old - m_new);
        if (d < n_t) s_w[d] = __expf(s_scores[d] - m_new);
        __syncthreads();
        if (dim_active) {
            acc *= corr;
            for (uint32_t p = 0; p < n_t; ++p) acc += s_w[p] * s_v[p * (head_dim + 1u) + d];
        }
        if (d == 0) {
            float ws = 0.0f;
            for (uint32_t p = 0; p < n_t; ++p) ws += s_w[p];
            *s_l = *s_l * corr + ws;
            *s_m = m_new;
        }
    }
}

// Paged twin of pd_attn_tile_walk: identical online-softmax math, but K/V live
// in a shared block pool [n_blocks, 16, kv_dim] and are addressed through this
// slot's block table `bt` (bt[pos/16] = physical block, pos%16 = intra-block
// row). Each 16-token block is internally contiguous, so only the per-token
// base computation differs from the dense walk - the inner tile loop, the
// numerics, and every __syncthreads() are byte-identical -> bit-exact parity.
template<typename KV, uint32_t TILE = PD_ATTN_TILE, bool PS = false>
__device__ __forceinline__ void pd_attn_tile_walk_paged(
    const KV* __restrict__ pool_k, const KV* __restrict__ pool_v,
    const uint32_t* __restrict__ bt, uint32_t first_pos,
    uint32_t lo, uint32_t hi, uint32_t kv_dim, uint32_t kvh, uint32_t head_dim,
    float scale, float* smem, float* s_m, float* s_l, float& acc) {
    uint32_t d = threadIdx.x, nth = blockDim.x;
    bool dim_active = d < head_dim;
    float* s_q = smem;                                      // [head_dim]
    float* s_k = s_q + head_dim;                            // [TILE][head_dim+1]
    float* s_v = s_k + TILE * (head_dim + 1u);      // [TILE][head_dim+1]
    float* s_scores = s_v + TILE * (head_dim + 1u); // [TILE]
    float* s_w = s_scores + TILE;                   // [TILE]

    for (uint32_t t0 = lo; t0 < hi; t0 += TILE) {
        uint32_t n_t = hi - t0 < TILE ? hi - t0 : TILE;
        __syncthreads();  // prior tile's s_v reads complete before overwrite
        uint32_t hd4 = head_dim >> 2;
        for (uint32_t idx = d; idx < n_t * hd4; idx += nth) {
            uint32_t p = idx / hd4, dd = (idx - p * hd4) << 2;
            // paged address: resolve this token's physical block, then offset
            // within it - the only line that differs from pd_attn_tile_walk.
            uint32_t tok = first_pos + t0 + p;
            uint32_t blk = bt[tok >> 4];
            uint32_t within = tok & 15u;
            size_t base = (size_t)blk * 16u * kv_dim + (size_t)within * kv_dim
                          + (size_t)kvh * head_dim + dd;
            float4 kf = pd_kv_load4(pool_k + base);
            float4 vf = pd_kv_load4(pool_v + base);
            float* kr = s_k + p * (head_dim + 1u) + dd;
            float* vr = s_v + p * (head_dim + 1u) + dd;
            kr[0] = kf.x; kr[1] = kf.y; kr[2] = kf.z; kr[3] = kf.w;
            vr[0] = vf.x; vr[1] = vf.y; vr[2] = vf.z; vr[3] = vf.w;
        }
        __syncthreads();
        pd_attn_tile_scores<TILE, PS>(s_q, s_k, s_scores, n_t, head_dim, scale);
        __syncthreads();
        // every thread derives the same m_new from shared (broadcast reads)
        float m_old = *s_m;
        float m_tile = -INFINITY;
        for (uint32_t p = 0; p < n_t; ++p) m_tile = fmaxf(m_tile, s_scores[p]);
        float m_new = fmaxf(m_old, m_tile);
        float corr = __expf(m_old - m_new);
        if (d < n_t) s_w[d] = __expf(s_scores[d] - m_new);
        __syncthreads();
        if (dim_active) {
            acc *= corr;
            for (uint32_t p = 0; p < n_t; ++p) acc += s_w[p] * s_v[p * (head_dim + 1u) + d];
        }
        if (d == 0) {
            float ws = 0.0f;
            for (uint32_t p = 0; p < n_t; ++p) ws += s_w[p];
            *s_l = *s_l * corr + ws;
            *s_m = m_new;
        }
    }
}

// FMHA-style decode attention (slot 537).
//
// The tile walk above stages K/V into shared as f32 -- 2*TILE*(head_dim+1)*4 B,
// which is 32.9 KB at head_dim 256 and caps TILE at 16 and occupancy at 6
// blocks. It then pays three __syncthreads() per 16 keys, and inside each tile
// most of the block is idle or redundant: every one of 256 threads rescans the
// same 16 scores for m_tile, the PV accumulate is 16 shared reads per thread
// (ncu: L1/TEX 72.8% against DRAM 1.23%), and the running-sum update is serial
// on thread 0 while 255 threads wait. Over a 256-token context that is ~48
// barriers to move the KV once.
//
// This kernel gives every WARP its own independent key stream -- warp w walks
// keys w, w+nw, w+2*nw, ... -- carrying (m, l, acc) in registers, so there is
// no per-tile barrier at all. Lane i owns a contiguous DPL-dim slice of the
// head, so the QK dot is DPL FMAs plus one 32-lane butterfly, and the PV
// accumulate is DPL register FMAs against a vector load. The warps merge once
// through shared at the end. Shared drops to nw*head_dim + 2*nw floats
// (8.25 KB at head_dim 256, nw 8), so this is no longer smem-bound.
//
// NUMERICS: the fold order differs from the tile walk (per-warp partial
// softmaxes merged pairwise, rather than one sequential tile chain), so this
// is not bit-exact against it -- it is its own numeric class, elected per
// model like the PS flag. The sink enters the merge as an extra term with
// m = sinks[h] and weight 1, exactly the (m=sink, l=1) init the walk starts
// from.
//
// DPL = head_dim / 32 and must be a multiple of 4 (the vector load width), so
// head_dim 128 and 256 are served; other head_dims stay on the tile walk.
//
// PAGED (slot 689, and 690 for the split form): K/V in a block pool
// [n_blocks, 16, kv_dim], key t of the slot at row
// block_tables[slot*bps + t/16]*16 + t%16 - nothing else changes, so the
// paged launch is bit-identical to the dense one over the same keys (each
// warp still folds its keys one by one, in order). See pd_fmha_walk for how
// the table is read.
//
// A warp's keys t_first, t_first + step, ... in order, each handed to `fold`
// as its cache row. Dense: the row is the key. Paged: the walk goes in
// windows of 32 keys; lane j holds the pool block of the window's j-th key
// and each key takes it by a broadcast shuffle, so the loop over a window
// has no branch and no table load and unrolls like the dense one (a per-key
// branch cost the paged walk the compiler's 2x unroll: +56% at c1 32K,
// bench/paged_modes_gb10_bench.cu). The next window's blocks are fetched
// one window ahead, and unconditionally: a lane past the live keys re-reads
// the last live key's entry, because a select on a loaded value stalls the
// warp at the select (in-order issue) instead of where the value is used.
// Entries past the slot's live keys are never read.
template<bool PAGED, typename F>
__device__ __forceinline__ void pd_fmha_walk(F&& fold, const uint32_t* __restrict__ bt,
                                             uint32_t t_first, uint32_t end_pos, uint32_t step,
                                             uint32_t lane) {
    if constexpr (PAGED) {
        const uint32_t win = 32u * step, last = end_pos - 1u;
        uint32_t tn = t_first + lane * step;
        uint32_t pg_next = bt[min(tn, last) >> 4];
        for (uint32_t t0 = t_first; t0 < end_pos; t0 += win) {
            const uint32_t pg = pg_next;
            tn += win;
            pg_next = bt[min(tn, last) >> 4];
            const uint32_t t1 = min(end_pos, t0 + win);
            uint32_t j = 0;
            #pragma unroll 2
            for (uint32_t t = t0; t < t1; t += step, ++j)
                fold((size_t)__shfl_sync(0xffffffffu, pg, j) * 16u + (t & 15u));
        }
    } else {
        for (uint32_t t = t_first; t < end_pos; t += step) fold((size_t)t);
    }
}

template<typename KV, uint32_t DPL, bool PAGED = false>
__global__ void pd_attn_decode_fmha_kernel(
    const float* __restrict__ q, const KV* __restrict__ kc,
    const KV* __restrict__ vc, const float* __restrict__ sinks,
    float* __restrict__ out, const unsigned int* __restrict__ positions,
    const unsigned int* __restrict__ slots,
    uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
    uint32_t max_ctx, uint32_t kv_dim, uint32_t swa_window, float scale,
    const uint32_t* __restrict__ block_tables = nullptr, uint32_t bps = 0) {
    const uint32_t h = blockIdx.x, b = blockIdx.y;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t nw = blockDim.x >> 5;
    const uint32_t kvh = h / (n_heads / n_kv_heads);

    const uint32_t slot = slots ? slots[b] : b;
    const uint32_t pos = positions[b];
    const uint32_t first_pos =
        (swa_window > 0 && pos + 1 > swa_window) ? (pos + 1 - swa_window) : 0;
    const uint32_t end_pos = pos + 1;

    const float* qb = q + (size_t)b * n_heads * head_dim + (size_t)h * head_dim;
    // dense: the slot's strip; paged: the pool, rows from the slot's table
    const size_t slot0 = PAGED ? 0 : (size_t)slot * max_ctx * kv_dim;
    const KV* kcb = kc + slot0 + (size_t)kvh * head_dim;
    const KV* vcb = vc + slot0 + (size_t)kvh * head_dim;
    const uint32_t* bt = PAGED ? block_tables + (size_t)slot * bps : nullptr;

    const uint32_t d0 = lane * DPL;
    float qr[DPL], acc[DPL];
    #pragma unroll
    for (uint32_t i = 0; i < DPL; ++i) { qr[i] = qb[d0 + i]; acc[i] = 0.0f; }
    float m = -INFINITY, l = 0.0f;

    // one key's fold into (m, l, acc)
    auto fold = [&](size_t row) {
        const KV* kr = kcb + row * kv_dim;
        const KV* vr = vcb + row * kv_dim;
        float kk[DPL], vv[DPL];
        #pragma unroll
        for (uint32_t i = 0; i < DPL; i += 4) {
            float4 kf = pd_kv_load4(kr + d0 + i);
            float4 vf = pd_kv_load4(vr + d0 + i);
            kk[i] = kf.x; kk[i + 1] = kf.y; kk[i + 2] = kf.z; kk[i + 3] = kf.w;
            vv[i] = vf.x; vv[i + 1] = vf.y; vv[i + 2] = vf.z; vv[i + 3] = vf.w;
        }
        float sc = 0.0f;
        #pragma unroll
        for (uint32_t i = 0; i < DPL; ++i) sc += qr[i] * kk[i];
        // butterfly: every lane leaves with the full head dot product
        #pragma unroll
        for (uint32_t o = 16u; o >= 1u; o >>= 1)
            sc += __shfl_xor_sync(0xffffffffu, sc, o, 32);
        sc *= scale;
        const float m_new = fmaxf(m, sc);
        const float corr = __expf(m - m_new);
        const float w = __expf(sc - m_new);
        #pragma unroll
        for (uint32_t i = 0; i < DPL; ++i) acc[i] = acc[i] * corr + w * vv[i];
        l = l * corr + w;
        m = m_new;
    };
    pd_fmha_walk<PAGED>(fold, bt, first_pos + warp, end_pos, nw, lane);

    extern __shared__ float smem[];
    float* s_acc = smem;                    // [nw][head_dim]
    float* s_m = s_acc + nw * head_dim;     // [nw]
    float* s_l = s_m + nw;                  // [nw]
    #pragma unroll
    for (uint32_t i = 0; i < DPL; ++i) s_acc[warp * head_dim + d0 + i] = acc[i];
    if (lane == 0) { s_m[warp] = m; s_l[warp] = l; }
    __syncthreads();

    if (warp == 0) {
        const float sink = sinks[h];
        float mg = sink;
        for (uint32_t w = 0; w < nw; ++w) mg = fmaxf(mg, s_m[w]);
        // the sink is the walk's (m = sinks[h], l = 1) starting state
        float lg = __expf(sink - mg);
        float o[DPL];
        #pragma unroll
        for (uint32_t i = 0; i < DPL; ++i) o[i] = 0.0f;
        for (uint32_t w = 0; w < nw; ++w) {
            // an empty warp carries m = -inf, l = 0: this weight is 0
            const float sw = __expf(s_m[w] - mg);
            lg += s_l[w] * sw;
            #pragma unroll
            for (uint32_t i = 0; i < DPL; ++i)
                o[i] += s_acc[w * head_dim + d0 + i] * sw;
        }
        #pragma unroll
        for (uint32_t i = 0; i < DPL; ++i)
            out[(size_t)b * n_heads * head_dim + (size_t)h * head_dim + d0 + i] = o[i] / lg;
    }
}

// SPLIT-KV decode attention, pass 1 (slot 545): grid (n_heads, batch, S).
// CTA (h, b, z) runs the same striped walk as pd_attn_decode_fmha_kernel but
// over the global warp stripe z*nw + warp with step S*nw, and writes its RAW
// merged partial (m, l, acc[head_dim] - Not divided, no sink) to
// part[((b*n_heads + h)*S + z)*(head_dim+2)]. The sink enters once, in the
// merge pass. At c1 the un-split form is 24 CTAs on 148 SMs and 39 us/layer
// vs the rival's 9.1 - the die is empty and the KV stream is the wall.
template<typename KV, uint32_t DPL, bool PAGED = false>
__global__ void pd_attn_decode_fmha_sp_kernel(
    const float* __restrict__ q, const KV* __restrict__ kc,
    const KV* __restrict__ vc, float* __restrict__ part,
    const unsigned int* __restrict__ positions,
    const unsigned int* __restrict__ slots,
    uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
    uint32_t max_ctx, uint32_t kv_dim, uint32_t swa_window, float scale,
    const uint32_t* __restrict__ block_tables = nullptr, uint32_t bps = 0) {
    const uint32_t h = blockIdx.x, b = blockIdx.y, z = blockIdx.z;
    const uint32_t S = gridDim.z;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t nw = blockDim.x >> 5;
    const uint32_t kvh = h / (n_heads / n_kv_heads);
    const uint32_t slot = slots ? slots[b] : b;
    const uint32_t pos = positions[b];
    const uint32_t first_pos =
        (swa_window > 0 && pos + 1 > swa_window) ? (pos + 1 - swa_window) : 0;
    const uint32_t end_pos = pos + 1;
    const float* qb = q + (size_t)b * n_heads * head_dim + (size_t)h * head_dim;
    const size_t slot0 = PAGED ? 0 : (size_t)slot * max_ctx * kv_dim;
    const KV* kcb = kc + slot0 + (size_t)kvh * head_dim;
    const KV* vcb = vc + slot0 + (size_t)kvh * head_dim;
    const uint32_t* bt = PAGED ? block_tables + (size_t)slot * bps : nullptr;
    const uint32_t d0 = lane * DPL;
    float qr[DPL], acc[DPL];
    #pragma unroll
    for (uint32_t i = 0; i < DPL; ++i) { qr[i] = qb[d0 + i]; acc[i] = 0.0f; }
    float m = -INFINITY, l = 0.0f;
    auto fold = [&](size_t row) {
        const KV* kr = kcb + row * kv_dim;
        const KV* vr = vcb + row * kv_dim;
        float kk[DPL], vv[DPL];
        #pragma unroll
        for (uint32_t i = 0; i < DPL; i += 4) {
            float4 kf = pd_kv_load4(kr + d0 + i);
            float4 vf = pd_kv_load4(vr + d0 + i);
            kk[i] = kf.x; kk[i + 1] = kf.y; kk[i + 2] = kf.z; kk[i + 3] = kf.w;
            vv[i] = vf.x; vv[i + 1] = vf.y; vv[i + 2] = vf.z; vv[i + 3] = vf.w;
        }
        float sc = 0.0f;
        #pragma unroll
        for (uint32_t i = 0; i < DPL; ++i) sc += qr[i] * kk[i];
        #pragma unroll
        for (uint32_t o = 16u; o >= 1u; o >>= 1)
            sc += __shfl_xor_sync(0xffffffffu, sc, o, 32);
        sc *= scale;
        const float m_new = fmaxf(m, sc);
        const float corr = __expf(m - m_new);
        const float w = __expf(sc - m_new);
        #pragma unroll
        for (uint32_t i = 0; i < DPL; ++i) acc[i] = acc[i] * corr + w * vv[i];
        l = l * corr + w;
        m = m_new;
    };
    pd_fmha_walk<PAGED>(fold, bt, first_pos + z * nw + warp, end_pos, S * nw, lane);
    extern __shared__ float smem[];
    float* s_acc = smem;
    float* s_m = s_acc + nw * head_dim;
    float* s_l = s_m + nw;
    #pragma unroll
    for (uint32_t i = 0; i < DPL; ++i) s_acc[warp * head_dim + d0 + i] = acc[i];
    if (lane == 0) { s_m[warp] = m; s_l[warp] = l; }
    __syncthreads();
    if (warp == 0) {
        float mg = -INFINITY;
        for (uint32_t w = 0; w < nw; ++w) mg = fmaxf(mg, s_m[w]);
        float lg = 0.0f;
        float o[DPL];
        #pragma unroll
        for (uint32_t i = 0; i < DPL; ++i) o[i] = 0.0f;
        if (mg > -INFINITY) {
            for (uint32_t w = 0; w < nw; ++w) {
                const float sw = __expf(s_m[w] - mg);
                lg += s_l[w] * sw;
                #pragma unroll
                for (uint32_t i = 0; i < DPL; ++i)
                    o[i] += s_acc[w * head_dim + d0 + i] * sw;
            }
        }
        float* pr = part + ((size_t)(b * n_heads + h) * S + z) * (head_dim + 2u);
        #pragma unroll
        for (uint32_t i = 0; i < DPL; ++i) pr[d0 + i] = o[i];
        if (lane == 0) { pr[head_dim] = mg; pr[head_dim + 1u] = lg; }
    }
}

// SPLIT-KV pass 2: grid (n_heads, batch), head_dim threads. Seeds from the
// sink exactly like the walk (m = sinks[h], l = 1), folds the S raw partials
// in FIXED z order, divides once, stores. An empty slice carries m = -inf,
// l = 0 and weighs nothing.
__global__ void pd_attn_fmha_merge_kernel(
    const float* __restrict__ part, const float* __restrict__ sinks,
    float* __restrict__ out, uint32_t n_heads, uint32_t head_dim, uint32_t S) {
    const uint32_t h = blockIdx.x, b = blockIdx.y, d = threadIdx.x;
    const float* pb = part + (size_t)(b * n_heads + h) * S * (head_dim + 2u);
    const float sink = sinks[h];
    float mg = sink;
    for (uint32_t z = 0; z < S; ++z)
        mg = fmaxf(mg, pb[z * (head_dim + 2u) + head_dim]);
    float lg = __expf(sink - mg);
    float o = 0.0f;
    for (uint32_t z = 0; z < S; ++z) {
        const float* pr = pb + z * (head_dim + 2u);
        const float mz = pr[head_dim];
        if (mz > -INFINITY) {
            const float sw = __expf(mz - mg);
            lg += pr[head_dim + 1u] * sw;
            o += pr[d] * sw;
        }
    }
    out[(size_t)b * n_heads * head_dim + (size_t)h * head_dim + d] = o / lg;
}

// Batched decode attention: grid (n_heads, batch). Block (h, b) runs online
// softmax for sequence b's query head h against sequence b's own KV cache up to
// its own position - the per-sequence attention that continuous batching needs.
// Per-sequence KV caches are contiguous [batch][max_ctx][kv_dim]; positions[b]
// is each sequence's current decode position; swa_window=0 means full attention.
// At batch B this launches n_heads*B blocks - better GPU fill than the batch-1
// single-block path, so attention occupancy improves rather than degrades.
// KV walk: pd_attn_tile_walk (per-32-key-tile numeric class).
template<typename KV, bool PS = false>
__global__ void pd_attn_decode_batch_kernel(
    const float* __restrict__ q, const KV* __restrict__ kc,
    const KV* __restrict__ vc, const float* __restrict__ sinks,
    float* __restrict__ out, const unsigned int* __restrict__ positions,
    const unsigned int* __restrict__ slots,
    uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
    uint32_t max_ctx, uint32_t kv_dim, uint32_t swa_window, float scale) {
    uint32_t h = blockIdx.x, b = blockIdx.y;
    uint32_t d = threadIdx.x;
    uint32_t group = n_heads / n_kv_heads;
    uint32_t kvh = h / group;

    // query row is b; the KV cache it reads may live in a different slot (prefill
    // maps all rows to one slot; decode uses slots==null -> slot b).
    uint32_t slot = slots ? slots[b] : b;
    uint32_t pos = positions[b];
    uint32_t first_pos = (swa_window > 0 && pos + 1 > swa_window) ? (pos + 1 - swa_window) : 0;
    uint32_t n_pos = pos + 1 - first_pos;

    extern __shared__ float smem[];
    __shared__ float s_m, s_l;

    const float* qb = q + (size_t)b * n_heads * head_dim;
    const KV* kcb = kc + (size_t)slot * max_ctx * kv_dim;
    const KV* vcb = vc + (size_t)slot * max_ctx * kv_dim;

    if (d < head_dim) smem[d] = qb[(size_t)h * head_dim + d];
    float acc = 0.0f;
    if (d == 0) { s_m = sinks[h]; s_l = 1.0f; }
    // pd_attn_tile_walk's leading __syncthreads() orders the q stage + m/l init

    if (head_dim > 128u)
        pd_attn_tile_walk<KV, PD_ATTN_TILE_HD256, PS>(kcb, vcb, first_pos, 0, n_pos, kv_dim,
                                                      kvh, head_dim, scale, smem, &s_m, &s_l, acc);
    else
        pd_attn_tile_walk<KV, PD_ATTN_TILE, PS>(kcb, vcb, first_pos, 0, n_pos, kv_dim, kvh,
                                                head_dim, scale, smem, &s_m, &s_l, acc);
    __syncthreads();
    if (d < head_dim)
        out[(size_t)b * n_heads * head_dim + (size_t)h * head_dim + d] = acc / s_l;
}

// Paged twin of pd_attn_decode_batch_kernel: K/V read from a shared block pool
// [n_blocks, 16, kv_dim] via each slot's block table (block_tables +
// slot*blocks_per_slot) instead of a dense [batch, max_ctx, kv_dim] region.
// No max_ctx - capacity is the pool, not a per-slot reservation. Everything
// else (grid, online softmax, GQA/swa/sinks) matches the dense kernel.
// Attention-stream shared load helper: 4 consecutive elements as float4 - the f32
// form is the exact vector load the quantize family always did; the __half
// form is one 8-byte load + exact expands (used by the a16 f16-plane arms).
template <typename T>
__device__ __forceinline__ float4 pd_ld4f(const T* p) {
    return *reinterpret_cast<const float4*>(p);
}
template <>
__device__ __forceinline__ float4 pd_ld4f<__half>(const __half* p) {
    const __half2 a = *reinterpret_cast<const __half2*>(p);
    const __half2 b = *reinterpret_cast<const __half2*>(p + 2);
    const float2 fa = __half22float2(a), fb = __half22float2(b);
    return make_float4(fa.x, fa.y, fb.x, fb.y);
}

// TQ/TO (attention streams): f16 q/out planes for the a16 route. PS = the
// parallel-score walk (slot 691, the paged twin of slot 536).
template<typename KV, typename TQ = float, typename TO = float, bool PS = false>
__global__ void pd_attn_decode_batch_paged_kernel(
    const TQ* __restrict__ q, const KV* __restrict__ pool_k,
    const KV* __restrict__ pool_v, const float* __restrict__ sinks,
    TO* __restrict__ out, const unsigned int* __restrict__ positions,
    const unsigned int* __restrict__ slots,
    const uint32_t* __restrict__ block_tables, uint32_t blocks_per_slot,
    uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
    uint32_t kv_dim, uint32_t swa_window, float scale) {
    uint32_t h = blockIdx.x, b = blockIdx.y;
    uint32_t d = threadIdx.x;
    uint32_t group = n_heads / n_kv_heads;
    uint32_t kvh = h / group;

    // query row is b; the KV cache it reads may live in a different slot (prefill
    // maps all rows to one slot; decode uses slots==null -> slot b).
    uint32_t slot = slots ? slots[b] : b;
    uint32_t pos = positions[b];
    uint32_t first_pos = (swa_window > 0 && pos + 1 > swa_window) ? (pos + 1 - swa_window) : 0;
    uint32_t n_pos = pos + 1 - first_pos;

    extern __shared__ float smem[];
    __shared__ float s_m, s_l;

    const TQ* qb = q + (size_t)b * n_heads * head_dim;
    const uint32_t* bt = block_tables + (size_t)slot * blocks_per_slot;

    if (d < head_dim) smem[d] = (float)qb[(size_t)h * head_dim + d];
    float acc = 0.0f;
    if (d == 0) { s_m = sinks[h]; s_l = 1.0f; }
    // pd_attn_tile_walk_paged's leading __syncthreads() orders the q stage + m/l init

    if (head_dim > 128u)
        pd_attn_tile_walk_paged<KV, PD_ATTN_TILE_HD256, PS>(pool_k, pool_v, bt, first_pos, 0, n_pos,
                                                            kv_dim, kvh, head_dim, scale, smem,
                                                            &s_m, &s_l, acc);
    else
        pd_attn_tile_walk_paged<KV, PD_ATTN_TILE, PS>(pool_k, pool_v, bt, first_pos, 0, n_pos,
                                                      kv_dim, kvh, head_dim, scale, smem, &s_m,
                                                      &s_l, acc);
    __syncthreads();
    if (d < head_dim)
        out[(size_t)b * n_heads * head_dim + (size_t)h * head_dim + d] = (TO)(acc / s_l);
}

// Batched rmsnorm: grid `batch`, one block normalizes one row of x[batch, n].
// The norm weight is shared across the batch (same layer weight). Warp-shuffle
// reduction (one cross-warp combine, not an 8-step shared tree) + float4 vectorized
// load/store when n%4==0 - this op is latency-bound (1.7% DRAM, single block),
// so cutting the reduction's sync chain and widening the loads is the lever. Keeps
// the exact 1/sqrtf (not approximate rsqrtf) so greedy parity holds.
// ACC selects the sumsq accumulator (PD_ACC_DF / _F64 / _F32) - see
// pd_norm_acc_mode. Templated rather than branched so no variant pays a
// runtime test in the inner loop.
template <int ACC> struct pd_acc_of      { using type = float; };
template <>        struct pd_acc_of<PD_ACC_F64> { using type = double; };
template <>        struct pd_acc_of<PD_ACC_DF>  { using type = pd_df; };

template <int ACC>
__global__ void pd_rmsnorm_batch_kernel_t(const float* __restrict__ x, const float* __restrict__ w,
                                        float* __restrict__ out, uint32_t n, float eps) {
    using A = typename pd_acc_of<ACC>::type;
    PD_PDL_ARM();  // cascade (granite chain)
    uint32_t b = blockIdx.x;
    const float* xb = x + (size_t)b * n;
    float* ob = out + (size_t)b * n;
    uint32_t tid = threadIdx.x, nth = blockDim.x;
    // width-stable sumsq (f32 products) - see pd_norm_wide_nth_ws.
    __shared__ A wsum[32];
    __shared__ float s_inv;
    A acc;
    if constexpr (ACC == PD_ACC_DF) { acc.hi = 0.0f; acc.lo = 0.0f; } else { acc = (A)0; }
    bool vec = (n & 3u) == 0;
    if (vec) {
        uint32_t n4 = n >> 2;
        const float4* x4 = reinterpret_cast<const float4*>(xb);
        for (uint32_t i = tid; i < n4; i += nth) {
            float4 v = x4[i];
            // products stay f32 in every mode - only the ACCUMULATE differs
            if constexpr (ACC == PD_ACC_DF) {
                pd_df_add(acc, v.x * v.x);
                pd_df_add(acc, v.y * v.y);
                pd_df_add(acc, v.z * v.z);
                pd_df_add(acc, v.w * v.w);
            } else {
                acc += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
            }
        }
    } else {
        for (uint32_t i = tid; i < n; i += nth) {
            if constexpr (ACC == PD_ACC_DF) pd_df_add(acc, xb[i] * xb[i]);
            else acc += xb[i] * xb[i];
        }
    }
    for (uint32_t s = 16; s > 0; s >>= 1) {
        if constexpr (ACC == PD_ACC_DF) {
            pd_df o;
            o.hi = __shfl_down_sync(0xffffffffu, acc.hi, s);
            o.lo = __shfl_down_sync(0xffffffffu, acc.lo, s);
            acc = pd_df_merge(acc, o);
        } else {
            acc += __shfl_down_sync(0xffffffffu, acc, s);
        }
    }
    uint32_t warp = tid >> 5, lane = tid & 31u;
    if (lane == 0) wsum[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        uint32_t nwarps = (nth + 31u) >> 5;
        double total;
        if constexpr (ACC == PD_ACC_DF) {
            pd_df sum; sum.hi = 0.0f; sum.lo = 0.0f;
            for (uint32_t wi = 0; wi < nwarps; ++wi) sum = pd_df_merge(sum, wsum[wi]);
            // One scalar f64 add/divide on one thread per block - the f64
            // cost is reduction DEPTH, not a single op, so this stays exact
            total = (double)sum.hi + (double)sum.lo;
        } else {
            A sum = (A)0;
            for (uint32_t wi = 0; wi < nwarps; ++wi) sum += wsum[wi];
            total = (double)sum;
        }
        s_inv = 1.0f / sqrtf((float)(total / (double)n) + eps);
    }
    __syncthreads();
    float inv = s_inv;
    if (vec) {
        uint32_t n4 = n >> 2;
        const float4* x4 = reinterpret_cast<const float4*>(xb);
        const float4* w4 = reinterpret_cast<const float4*>(w);
        float4* o4 = reinterpret_cast<float4*>(ob);
        for (uint32_t i = tid; i < n4; i += nth) {
            float4 v = x4[i], wv = w4[i], r;
            r.x = v.x * inv * wv.x;
            r.y = v.y * inv * wv.y;
            r.z = v.z * inv * wv.z;
            r.w = v.w * inv * wv.w;
            o4[i] = r;
        }
    } else {
        for (uint32_t i = tid; i < n; i += nth) ob[i] = xb[i] * inv * w[i];
    }
}

// Batched YaRN rope: one thread per (sequence, head), each rotating at its
// sequence's own position. x is [batch, n_heads*head_dim]; positions[b].
// Warp-per-head, lane-per-pair shape (the old one-thread-per-head serial loop
// was a 64-thread latency-bound launch at B=1: 8.8 µs × 48/token = 0.42 ms).
// BIT-EXACT to the serial version: theta_k comes from the identical sequence of
// k successive `theta *= theta_scale` roundings - lane k just runs its own copy
// of that chain (and 32 more steps per stride when half > 32).
// NEOX=true rotates half-split pairs (k, k+half) - qwen35/gemma4/laguna/gpt-oss.
// NEOX=false is llama.cpp's ROPE_TYPE_NORM: interleaved pairs (2k, 2k+1), which
// granite (and the whole llama-arch lineage) uses. Same theta chain either way -
// only the pair indexing differs, so this is a compile-time template exactly
// like llama.cpp's separate rope_norm/rope_neox kernels and vLLM's IS_NEOX
// template arg. The NEOX instantiation compiles to what it always did; there is
// no runtime branch and no cost to the families already on this path.
template <bool NEOX>
__global__ void pd_rope_yarn_batch_kernel(float* __restrict__ x, const unsigned int* __restrict__ positions,
                                          uint32_t n_heads, uint32_t head_dim, float theta_scale,
                                          float freq_scale, float corr_low, float corr_high,
                                          float ext_factor, float mscale, uint32_t batch) {
    uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    uint32_t idx = blockIdx.x * (blockDim.x >> 5) + warp;
    if (idx >= batch * n_heads) return;
    uint32_t b = idx / n_heads, h = idx % n_heads;
    float* head = x + (size_t)b * n_heads * head_dim + (size_t)h * head_dim;
    uint32_t half = head_dim / 2;
    float theta = (float)positions[b];
    for (uint32_t i = 0; i < lane && i < half; ++i) theta *= theta_scale;
    for (uint32_t k = lane; k < half; k += 32) {
        float y = ((float)k - corr_low) / fmaxf(0.001f, corr_high - corr_low);
        float ramp = (1.0f - fminf(1.0f, fmaxf(0.0f, y))) * ext_factor;
        float angle = (freq_scale * theta) * (1.0f - ramp) + theta * ramp;
        float s = sinf(angle) * mscale;
        float c = cosf(angle) * mscale;
        // pair k lives at (k, k+half) under NEOX, (2k, 2k+1) under NORM
        uint32_t i0 = NEOX ? k : 2u * k;
        uint32_t i1 = NEOX ? k + half : 2u * k + 1u;
        float a = head[i0];
        float bb = head[i1];
        head[i0] = a * c - bb * s;
        head[i1] = a * s + bb * c;
        for (uint32_t i = 0; i < 32 && k + i < half; ++i) theta *= theta_scale;
    }
}

// pd_rope_yarn_batch with per-pair frequency divisors (ggml `freq_factors`):
// theta for pair k becomes pos * theta_scale^k / factors[k] before the yarn
// ramp, exactly ggml_rope's `theta_base / freq_factors[i0/2]`. factors may be
// null (all-1.0 = plain yarn rope). gemma4 global layers pass rope_freqs whose
// 1e30 entries collapse the angle to ~0 - those pairs ride through (partial
// rotary), matching llama.cpp b10058 bit for bit since it computes the same
// tiny-angle sin/cos in float.
// NEOX per pd_rope_yarn_batch_kernel's note: half-split (k, k+half) pairs vs
// ROPE_TYPE_NORM's interleaved (2k, 2k+1). muse-glimmer is NORM (llama.cpp
// llama-model.cpp puts LLM_ARCH_MUSE_GLIMMER in the NORM bucket alongside
// granite) while gemma4 - whose graph it shares here - is NEOX, so this
// carrier had to grow the same compile-time split the plain yarn rope has.
template <bool NEOX>
__global__ void pd_rope_factors_batch_kernel(float* __restrict__ x,
                                             const unsigned int* __restrict__ positions,
                                             const float* __restrict__ factors,
                                             uint32_t n_heads, uint32_t head_dim,
                                             float theta_scale, float freq_scale,
                                             float corr_low, float corr_high,
                                             float ext_factor, float mscale, uint32_t batch) {
    uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    uint32_t idx = blockIdx.x * (blockDim.x >> 5) + warp;
    if (idx >= batch * n_heads) return;
    uint32_t b = idx / n_heads, h = idx % n_heads;
    float* head = x + (size_t)b * n_heads * head_dim + (size_t)h * head_dim;
    uint32_t half = head_dim / 2;
    float theta = (float)positions[b];
    for (uint32_t i = 0; i < lane && i < half; ++i) theta *= theta_scale;
    for (uint32_t k = lane; k < half; k += 32) {
        float t = factors ? theta / factors[k] : theta;
        float y = ((float)k - corr_low) / fmaxf(0.001f, corr_high - corr_low);
        float ramp = (1.0f - fminf(1.0f, fmaxf(0.0f, y))) * ext_factor;
        float angle = (freq_scale * t) * (1.0f - ramp) + t * ramp;
        float s = sinf(angle) * mscale;
        float c = cosf(angle) * mscale;
        const uint32_t i0 = NEOX ? k : 2u * k;
        const uint32_t i1 = NEOX ? k + half : 2u * k + 1u;
        float a = head[i0];
        float bb = head[i1];
        head[i0] = a * c - bb * s;
        head[i1] = a * s + bb * c;
        for (uint32_t i = 0; i < 32 && k + i < half; ++i) theta *= theta_scale;
    }
}

// Fused prefill QKV epilogue norms: per-head RMS norm (q,k learned,
// V weightless) + NEOX rope on q/k - the prefill band ran this as FIVE
// launches per layer (3x rmsnorm_batch + 2x rope_factors_batch, ~7.6% of
// the pf8 GPU) with a full qn round-trip between norm and rope. One block
// per (row, head-slot): slots 0..n_head = q, then n_kv k, then n_kv v.
// Norm math clones pd_rmsnorm_batch_kernel at 256 threads exactly; the rope
// phase runs on warp 0 with pd_rope_factors_batch_kernel's per-warp theta
// chain verbatim - outputs bit-identical to the five-kernel chain. Appends
// stay separate (the SWA ring-shrink contract needs sub-span appends).
template <bool NEOX = true>
__global__ void pd_qkv_norm_rope_batch_kernel(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ qw,
    const float* __restrict__ kw, float* __restrict__ qn,
    float* __restrict__ kn, float* __restrict__ vn,
    const unsigned int* __restrict__ positions,
    const float* __restrict__ factors, uint32_t n_head, uint32_t n_kv,
    uint32_t head_dim, float eps, float theta_scale, float freq_scale,
    float corr_low, float corr_high, float ext_factor, float mscale,
    bool vnorm) {
    const uint32_t slot = blockIdx.x, b = blockIdx.y;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    const bool is_q = slot < n_head;
    const bool is_v = slot >= n_head + n_kv;
    const uint32_t h = is_q ? slot : (is_v ? slot - n_head - n_kv : slot - n_head);
    const float* src = is_q ? q + ((size_t)b * n_head + h) * head_dim
                     : (is_v ? v : k) + ((size_t)b * n_kv + h) * head_dim;
    float* dst = is_q ? qn + ((size_t)b * n_head + h) * head_dim
                : (is_v ? vn : kn) + ((size_t)b * n_kv + h) * head_dim;
    const float* w = is_q ? qw : kw;   // V: weightless (x*inv*1.0 == x*inv)

    // vnorm=false: V is a straight copy. gemma4 RMS-norms V weightlessly
    // (gemma4.cpp: `Vcur = ggml_rms_norm(ctx0, Vcur, f_norm_rms_eps)`);
    // muse-glimmer does not touch V at all (muse-glimmer.cpp hands the raw
    // Vcur to build_attn). Arch constant, not a tuning knob - and the whole
    // sum-of-squares pass below is dead work in that case, so bail early.
    // Warp-uniform: the slot is the block index, so no divergence.
    if (is_v && !vnorm) {
        for (uint32_t i = tid; i < head_dim; i += nth) dst[i] = src[i];
        return;
    }

    __shared__ float wsum[32];
    __shared__ float s_inv;
    float acc = 0.0f;
    for (uint32_t i = tid; i < head_dim; i += nth) {
        const float x = src[i];
        acc += x * x;
    }
    for (uint32_t sh = 16; sh > 0; sh >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, sh);
    const uint32_t warp = tid >> 5, lane = tid & 31u;
    if (lane == 0) wsum[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        float sum = 0.0f;
        const uint32_t nwarps = (nth + 31u) >> 5;
        for (uint32_t wi = 0; wi < nwarps; ++wi) sum += wsum[wi];
        s_inv = 1.0f / sqrtf(sum / (float)head_dim + eps);
    }
    __syncthreads();
    const float inv = s_inv;
    for (uint32_t i = tid; i < head_dim; i += nth)
        dst[i] = is_v ? src[i] * inv : src[i] * inv * w[i];
    if (is_v) return;
    __syncthreads();
    // rope on warp 0: pd_rope_factors_batch_kernel's chain verbatim
    if (warp == 0) {
        float* head = dst;
        const uint32_t half = head_dim / 2;
        float theta = (float)positions[b];
        for (uint32_t i = 0; i < lane && i < half; ++i) theta *= theta_scale;
        for (uint32_t kk = lane; kk < half; kk += 32) {
            float t = factors ? theta / factors[kk] : theta;
            float y = ((float)kk - corr_low) / fmaxf(0.001f, corr_high - corr_low);
            float ramp = (1.0f - fminf(1.0f, fmaxf(0.0f, y))) * ext_factor;
            float angle = (freq_scale * t) * (1.0f - ramp) + t * ramp;
            float s = sinf(angle) * mscale;
            float c = cosf(angle) * mscale;
            const uint32_t i0 = NEOX ? kk : 2u * kk;
            const uint32_t i1 = NEOX ? kk + half : 2u * kk + 1u;
            float a = head[i0];
            float bb = head[i1];
            head[i0] = a * c - bb * s;
            head[i1] = a * s + bb * c;
            for (uint32_t i = 0; i < 32 && kk + i < half; ++i) theta *= theta_scale;
        }
    }
}


// v2: warp-per-slot rewrite of the fused norm+rope. The v1 grid
// spent a 256-thread CTA per (row, head-slot) - 98K tiny CTAs at r=2048,
// 0.84 TB/s (10x off the roundtrip floor), rope serialized on warp 0 with
// 7 warps parked. Here each WARP owns one (row, slot): 8 slots per CTA,
// 12x fewer CTAs, no smem, no syncthreads. BIT-IDENTICAL to v1: the
// square-sum keeps v1's exact associativity (per-32-block shfl trees,
// blocks then summed in order on lane 0), the store math is the same
// scalar expression, and the rope phase was warp-scope already (verbatim).
template <bool TBL, typename TI = float, bool NEOX = true>
__global__ void pd_qkv_norm_rope_batch_v2_kernel(
    const TI* __restrict__ q, const TI* __restrict__ k,
    const TI* __restrict__ v, const float* __restrict__ qw,
    const float* __restrict__ kw, float* __restrict__ qn,
    float* __restrict__ kn, float* __restrict__ vn,
    const unsigned int* __restrict__ positions,
    const float* __restrict__ factors, uint32_t n_head, uint32_t n_kv,
    uint32_t head_dim, float eps, float theta_scale, float freq_scale,
    float corr_low, float corr_high, float ext_factor, float mscale,
    uint32_t rows, bool vnorm) {
    const uint32_t nslots = n_head + 2u * n_kv;
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t idx = blockIdx.x * (blockDim.x >> 5) + warp;
    if (idx >= rows * nslots) return;
    const uint32_t b = idx / nslots, slot = idx % nslots;
    const bool is_q = slot < n_head;
    const bool is_v = slot >= n_head + n_kv;
    const uint32_t h = is_q ? slot : (is_v ? slot - n_head - n_kv : slot - n_head);
    const TI* src = is_q ? q + ((size_t)b * n_head + h) * head_dim
                  : (is_v ? v : k) + ((size_t)b * n_kv + h) * head_dim;
    float* dst = is_q ? qn + ((size_t)b * n_head + h) * head_dim
                : (is_v ? vn : kn) + ((size_t)b * n_kv + h) * head_dim;
    const float* w = is_q ? qw : kw;

    // see v1: vnorm=false leaves V untouched (muse-glimmer). idx is a
    // per-WARP index, so is_v is warp-uniform and this bail is divergence-free.
    if (is_v && !vnorm) {
        for (uint32_t i = lane; i < head_dim; i += 32u) dst[i] = (float)src[i];
        return;
    }

    // v1's reduction shape: 32-element blocks each shfl-tree-reduced, block
    // results summed in ascending order - reproduced exactly so inv matches
    // bit-for-bit
    float sum = 0.0f;
    #pragma unroll
    for (uint32_t w2 = 0; w2 < 8u; ++w2) {
        float acc = 0.0f;
        for (uint32_t i = w2 * 32u + lane; i < head_dim; i += 256u) {
            const float x = (float)src[i];
            acc += x * x;
        }
        for (uint32_t sh = 16; sh > 0; sh >>= 1)
            acc += __shfl_down_sync(0xffffffffu, acc, sh);
        acc = __shfl_sync(0xffffffffu, acc, 0);
        sum += acc;
    }
    const float inv = 1.0f / sqrtf(sum / (float)head_dim + eps);
    for (uint32_t i = lane; i < head_dim; i += 32u)
        dst[i] = is_v ? (float)src[i] * inv : (float)src[i] * inv * w[i];
    if (is_v) return;
    __syncwarp();
    // rope: pd_rope_factors_batch_kernel's warp chain verbatim (v1 ran this
    // exact code on its warp 0)
    {
        float* head = dst;
        const uint32_t half = head_dim / 2;
        // TBL: closed-form theta = pos * 2^(kk*log2(ts)) - kills the ~160
        // serial dependent fmuls of the repeated-multiply chain (the pf8
        // profile's residual). One rounding vs kk roundings: a CLASS change,
        // serving gates arbitrate.
        const float l2ts = TBL ? log2f(theta_scale) : 0.0f;
        const float pos_f = (float)positions[b];
        float theta = pos_f;
        if (!TBL)
            for (uint32_t i = 0; i < lane && i < half; ++i) theta *= theta_scale;
        for (uint32_t kk = lane; kk < half; kk += 32) {
            if (TBL) theta = pos_f * exp2f((float)kk * l2ts);
            float t = factors ? theta / factors[kk] : theta;
            float y = ((float)kk - corr_low) / fmaxf(0.001f, corr_high - corr_low);
            float ramp = (1.0f - fminf(1.0f, fmaxf(0.0f, y))) * ext_factor;
            float angle = (freq_scale * t) * (1.0f - ramp) + t * ramp;
            float s = sinf(angle) * mscale;
            float c = cosf(angle) * mscale;
            const uint32_t i0 = NEOX ? kk : 2u * kk;
            const uint32_t i1 = NEOX ? kk + half : 2u * kk + 1u;
            float a = head[i0];
            float bb = head[i1];
            head[i0] = a * c - bb * s;
            head[i1] = a * s + bb * c;
            if (!TBL)
                for (uint32_t i = 0; i < 32 && kk + i < half; ++i) theta *= theta_scale;
        }
    }
}

// v3 (the glue rung): register-resident twin of v2. At head_dim
// <= 256 each lane already holds the whole head across <=8 registers
// (element w2*32+lane = reg w2), and the rope pair (kk, kk+half) maps to
// regs (t, t+nb/2) of the same lane - so v2's write dst / read dst back /
// write dst again collapses to one store pass. BIT-IDENTICAL to v2: the
// sumsq keeps the per-32-block shfl trees summed in ascending order
// (empty high blocks still contribute +0.0f, which can't flip a sign
// bit on a sum of squares), the norm scale is the same scalar expression
// over the same register value, and the rope math is verbatim - only the
// a/bb source moves from a dst read-back to the register that produced
// it. Gate: head_dim %64 == 0 (the pair map needs whole 32-blocks on
// each side of half) and <= 256; the launcher falls back to v2 outside.
// To (attention streams): f16 output plane. v3 only - the rope
// runs on f32 registers and rounds once at the store, so the f16 plane is
// exactly __float2half(the f32 plane) elementwise (v2 would read back its
// own rounded norm store and rope over that - a different class).
template <bool TBL, typename TI = float, typename TO = float, bool NEOX = true>
__global__ void pd_qkv_norm_rope_batch_v3_kernel(
    const TI* __restrict__ q, const TI* __restrict__ k,
    const TI* __restrict__ v, const float* __restrict__ qw,
    const float* __restrict__ kw, TO* __restrict__ qn,
    TO* __restrict__ kn, TO* __restrict__ vn,
    const unsigned int* __restrict__ positions,
    const float* __restrict__ factors, uint32_t n_head, uint32_t n_kv,
    uint32_t head_dim, float eps, float theta_scale, float freq_scale,
    float corr_low, float corr_high, float ext_factor, float mscale,
    uint32_t rows, bool vnorm) {
    const uint32_t nslots = n_head + 2u * n_kv;
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t idx = blockIdx.x * (blockDim.x >> 5) + warp;
    if (idx >= rows * nslots) return;
    const uint32_t b = idx / nslots, slot = idx % nslots;
    const bool is_q = slot < n_head;
    const bool is_v = slot >= n_head + n_kv;
    const uint32_t h = is_q ? slot : (is_v ? slot - n_head - n_kv : slot - n_head);
    const TI* src = is_q ? q + ((size_t)b * n_head + h) * head_dim
                  : (is_v ? v : k) + ((size_t)b * n_kv + h) * head_dim;
    TO* dst = is_q ? qn + ((size_t)b * n_head + h) * head_dim
            : (is_v ? vn : kn) + ((size_t)b * n_kv + h) * head_dim;
    const float* w = is_q ? qw : kw;

    const uint32_t nb = head_dim >> 5;  // <= 8 (launcher gate)
    float rv[8];
    #pragma unroll
    for (uint32_t w2 = 0; w2 < 8u; ++w2)
        rv[w2] = w2 < nb ? (float)src[w2 * 32u + lane] : 0.0f;
    // see v1: vnorm=false leaves V untouched (muse-glimmer). Staged already,
    // so this is a store pass over the registers and the sumsq below is skipped.
    if (is_v && !vnorm) {
        #pragma unroll
        for (uint32_t j = 0; j < 8u; ++j)
            if (j < nb) dst[j * 32u + lane] = (TO)rv[j];
        return;
    }
    float sum = 0.0f;
    #pragma unroll
    for (uint32_t w2 = 0; w2 < 8u; ++w2) {
        // empty high blocks contributed a full shfl tree of exact +0.0f -
        // skipping them is bit-identical and halves the tree at hd128
        if (w2 >= nb) break;
        float acc = rv[w2] * rv[w2];
        for (uint32_t sh = 16; sh > 0; sh >>= 1)
            acc += __shfl_down_sync(0xffffffffu, acc, sh);
        acc = __shfl_sync(0xffffffffu, acc, 0);
        sum += acc;
    }
    const float inv = 1.0f / sqrtf(sum / (float)head_dim + eps);
    #pragma unroll
    for (uint32_t j = 0; j < 8u; ++j)
        if (j < nb)
            rv[j] = is_v ? rv[j] * inv : rv[j] * inv * w[j * 32u + lane];
    if (is_v) {
        #pragma unroll
        for (uint32_t j = 0; j < 8u; ++j)
            if (j < nb) dst[j * 32u + lane] = (TO)rv[j];
        return;
    }
    // rope: v2's warp chain verbatim, a/bb sourced from registers
    if (NEOX) {
        const uint32_t half = head_dim >> 1, hb = nb >> 1;
        const float l2ts = TBL ? log2f(theta_scale) : 0.0f;
        const float pos_f = (float)positions[b];
        float theta = pos_f;
        if (!TBL)
            for (uint32_t i = 0; i < lane && i < half; ++i) theta *= theta_scale;
        for (uint32_t t2 = 0; t2 < hb; ++t2) {
            const uint32_t kk = t2 * 32u + lane;
            if (TBL) theta = pos_f * exp2f((float)kk * l2ts);
            float t = factors ? theta / factors[kk] : theta;
            float y = ((float)kk - corr_low) / fmaxf(0.001f, corr_high - corr_low);
            float ramp = (1.0f - fminf(1.0f, fmaxf(0.0f, y))) * ext_factor;
            float angle = (freq_scale * t) * (1.0f - ramp) + t * ramp;
            float s = sinf(angle) * mscale;
            float c = cosf(angle) * mscale;
            float a = rv[t2];
            float bb = rv[t2 + hb];
            dst[kk] = (TO)(a * c - bb * s);
            dst[kk + half] = (TO)(a * s + bb * c);
            if (!TBL)
                for (uint32_t i = 0; i < 32 && kk + i < half; ++i) theta *= theta_scale;
        }
    } else {
        // ROPE_TYPE_NORM on the register-resident form. The NEOX arm above is
        // register-local because its partner element (kk+half) sits in the
        // same lane's register t2+hb. NORM's partner is the ADJACENT ELEMENT,
        // and in this lane-strided staging (rv[j] holds element j*32+lane)
        // that lives in the adjacent LANE of the same register - so the pair
        // is closed with one shfl_xor(...,1) and each lane writes its own
        // half of the rotation.
        //
        // Angle dedup: the two lanes of a pair used to compute the
        // same theta/sinf/cosf (kk = e>>1 collapses adjacent lanes), and this
        // arm was 90% SM-bound at the muse c32 wave - sinf without fast-math
        // is a ~30-instruction software sequence that issues for the whole
        // warp regardless of the active mask, so predicating duplicates off
        // buys nothing. Two consecutive register blocks hold exactly 32
        // rotation pairs, so each lane computes one distinct pair per block-
        // PAIR and consumers fetch by shfl: the same expression over the same
        // inputs, evaluated once instead of twice - every distributed s/c is
        // bit-identical to what the consuming lane computed before. nb is
        // even (the launcher gates head_dim % 64 == 0).
        const float l2ts = TBL ? log2f(theta_scale) : 0.0f;
        const float pos_f = (float)positions[b];
        const bool odd = (lane & 1u) != 0u;
        #pragma unroll
        for (uint32_t j = 0; j < 8u; j += 2u) {
            if (j >= nb) break;
            const uint32_t kk = j * 16u + lane;  // this lane's own pair
            // theta for pair kk. TBL is the closed form both other kernels
            // use; the !TBL arm rebuilds the same multiply chain from scratch
            // (kk steps), which is what v1/v2 land on for this pair - the
            // lane-strided walk can't inherit a running chain, and this path
            // only runs under PADDOCK_NO_RTBL.
            float theta = pos_f;
            if (TBL) {
                theta = pos_f * exp2f((float)kk * l2ts);
            } else {
                for (uint32_t i = 0; i < kk; ++i) theta *= theta_scale;
            }
            const float t = factors ? theta / factors[kk] : theta;
            const float y = ((float)kk - corr_low) / fmaxf(0.001f, corr_high - corr_low);
            const float ramp = (1.0f - fminf(1.0f, fmaxf(0.0f, y))) * ext_factor;
            const float angle = (freq_scale * t) * (1.0f - ramp) + t * ramp;
            const float ps = sinf(angle) * mscale;
            const float pc = cosf(angle) * mscale;
            #pragma unroll
            for (uint32_t d2 = 0; d2 < 2u; ++d2) {
                const uint32_t jj = j + d2;
                // element jj*32+lane belongs to pair jj*16 + (lane>>1); its
                // s/c live on lane (d2*16 + lane>>1) of this block-pair walk
                const uint32_t src = d2 * 16u + (lane >> 1);
                const float s = __shfl_sync(0xffffffffu, ps, src);
                const float c = __shfl_sync(0xffffffffu, pc, src);
                const float mine = rv[jj];
                const float other = __shfl_xor_sync(0xffffffffu, mine, 1u);
                const float a = odd ? other : mine;   // even element of the pair
                const float bb = odd ? mine : other;  // odd element of the pair
                dst[jj * 32u + lane] = (TO)(odd ? (a * s + bb * c) : (a * c - bb * s));
            }
        }
    }
}

static int pd_qkv_norm_rope_batch_impl(
    const void* q, const void* k, const void* v, const void* qw,
    const void* kw, void* qn, void* kn, void* vn, const void* positions,
    const void* factors, uint32_t n_head, uint32_t n_kv, uint32_t head_dim,
    float eps, float theta_scale, float freq_scale, float corr_low,
    float corr_high, float ext_factor, float mscale, uint32_t rows,
    uint32_t i16, uint32_t o16, bool neox, bool vnorm, void* stream) {
    if (rows == 0 || head_dim == 0) return 0;
    // v2 warp-per-slot grid (bit-identical; see kernel note). Kill:
    // PADDOCK_NO_NRV2 restores the CTA-per-slot v1.
    static int no_v2 = -1;
    if (no_v2 < 0) no_v2 = pd_env("PADDOCK_NO_NRV2") ? 1 : 0;
    // rope theta mode: closed-form table by default (class change vs the
    // serial chain; kill PADDOCK_NO_RTBL restores the chain)
    static int no_tbl = -1;
    if (no_tbl < 0) no_tbl = pd_env("PADDOCK_NO_RTBL") ? 1 : 0;
    // v3 register-resident twin (bit-identical; see kernel note).
    // Kill: PADDOCK_NO_NRV3 restores the v2 smem-free warp-per-slot form.
    static int no_v3 = -1;
    if (no_v3 < 0) no_v3 = pd_env("PADDOCK_NO_NRV3") ? 1 : 0;
    const uint32_t total = rows * (n_head + 2u * n_kv);
    const bool v3ok = !no_v3 && head_dim <= 256u && (head_dim & 63u) == 0u;
    if (o16) {
        // f16 output plane (attention streams): v3 register form
        // only - the geometry gate is a hard requirement, not a fallback
        // seam (v2 would rope over its own rounded store, a different
        // class), and PADDOCK_NO_NRV3 deliberately does not reach here.
        if (head_dim > 256u || (head_dim & 63u)) return cudaErrorInvalidValue;
        if (i16) {
            auto kfn = neox
                ? (no_tbl
                   ? pd_qkv_norm_rope_batch_v3_kernel<false, __nv_bfloat16, __half, true>
                   : pd_qkv_norm_rope_batch_v3_kernel<true, __nv_bfloat16, __half, true>)
                : (no_tbl
                   ? pd_qkv_norm_rope_batch_v3_kernel<false, __nv_bfloat16, __half, false>
                   : pd_qkv_norm_rope_batch_v3_kernel<true, __nv_bfloat16, __half, false>);
            kfn<<<(total + 7u) / 8u, 256, 0, (cudaStream_t)stream>>>(
                (const __nv_bfloat16*)q, (const __nv_bfloat16*)k,
                (const __nv_bfloat16*)v, (const float*)qw, (const float*)kw,
                (__half*)qn, (__half*)kn, (__half*)vn,
                (const unsigned int*)positions, (const float*)factors, n_head,
                n_kv, head_dim, eps, theta_scale, freq_scale, corr_low,
                corr_high, ext_factor, mscale, rows, vnorm);
            return (int)cudaGetLastError();
        }
        auto kfn = neox
            ? (no_tbl ? pd_qkv_norm_rope_batch_v3_kernel<false, float, __half, true>
                      : pd_qkv_norm_rope_batch_v3_kernel<true, float, __half, true>)
            : (no_tbl ? pd_qkv_norm_rope_batch_v3_kernel<false, float, __half, false>
                      : pd_qkv_norm_rope_batch_v3_kernel<true, float, __half, false>);
        kfn<<<(total + 7u) / 8u, 256, 0, (cudaStream_t)stream>>>(
            (const float*)q, (const float*)k, (const float*)v, (const float*)qw,
            (const float*)kw, (__half*)qn, (__half*)kn, (__half*)vn,
            (const unsigned int*)positions, (const float*)factors, n_head, n_kv,
            head_dim, eps, theta_scale, freq_scale, corr_low, corr_high,
            ext_factor, mscale, rows, vnorm);
        return (int)cudaGetLastError();
    }
    if (i16) {
        // bf16 inputs (the chunk-band o16 GEMM stream): v2/v3 TI arms only -
        // v1 has no 16-bit form, so no_v2 arbitrates v2-vs-v3 but not the
        // input class
        auto kfn = v3ok
            ? (neox
               ? (no_tbl ? pd_qkv_norm_rope_batch_v3_kernel<false, __nv_bfloat16, float, true>
                         : pd_qkv_norm_rope_batch_v3_kernel<true, __nv_bfloat16, float, true>)
               : (no_tbl ? pd_qkv_norm_rope_batch_v3_kernel<false, __nv_bfloat16, float, false>
                         : pd_qkv_norm_rope_batch_v3_kernel<true, __nv_bfloat16, float, false>))
            : (neox
               ? (no_tbl ? pd_qkv_norm_rope_batch_v2_kernel<false, __nv_bfloat16, true>
                         : pd_qkv_norm_rope_batch_v2_kernel<true, __nv_bfloat16, true>)
               : (no_tbl ? pd_qkv_norm_rope_batch_v2_kernel<false, __nv_bfloat16, false>
                         : pd_qkv_norm_rope_batch_v2_kernel<true, __nv_bfloat16, false>));
        kfn<<<(total + 7u) / 8u, 256, 0, (cudaStream_t)stream>>>(
            (const __nv_bfloat16*)q, (const __nv_bfloat16*)k,
            (const __nv_bfloat16*)v, (const float*)qw, (const float*)kw,
            (float*)qn, (float*)kn, (float*)vn,
            (const unsigned int*)positions, (const float*)factors, n_head,
            n_kv, head_dim, eps, theta_scale, freq_scale, corr_low, corr_high,
            ext_factor, mscale, rows, vnorm);
        return (int)cudaGetLastError();
    }
    if (!no_v2) {
        auto kfn = v3ok
            ? (neox
               ? (no_tbl ? pd_qkv_norm_rope_batch_v3_kernel<false, float, float, true>
                         : pd_qkv_norm_rope_batch_v3_kernel<true, float, float, true>)
               : (no_tbl ? pd_qkv_norm_rope_batch_v3_kernel<false, float, float, false>
                         : pd_qkv_norm_rope_batch_v3_kernel<true, float, float, false>))
            : (neox
               ? (no_tbl ? pd_qkv_norm_rope_batch_v2_kernel<false, float, true>
                         : pd_qkv_norm_rope_batch_v2_kernel<true, float, true>)
               : (no_tbl ? pd_qkv_norm_rope_batch_v2_kernel<false, float, false>
                         : pd_qkv_norm_rope_batch_v2_kernel<true, float, false>));
        kfn<<<(total + 7u) / 8u, 256, 0, (cudaStream_t)stream>>>(
            (const float*)q, (const float*)k, (const float*)v, (const float*)qw,
            (const float*)kw, (float*)qn, (float*)kn, (float*)vn,
            (const unsigned int*)positions, (const float*)factors, n_head, n_kv,
            head_dim, eps, theta_scale, freq_scale, corr_low, corr_high,
            ext_factor, mscale, rows, vnorm);
        return (int)cudaGetLastError();
    }
    dim3 grid(n_head + 2u * n_kv, rows);
    auto kv1 = neox ? pd_qkv_norm_rope_batch_kernel<true>
                    : pd_qkv_norm_rope_batch_kernel<false>;
    kv1<<<grid, 256, 0, (cudaStream_t)stream>>>(
        (const float*)q, (const float*)k, (const float*)v, (const float*)qw,
        (const float*)kw, (float*)qn, (float*)kn, (float*)vn,
        (const unsigned int*)positions, (const float*)factors, n_head, n_kv,
        head_dim, eps, theta_scale, freq_scale, corr_low, corr_high,
        ext_factor, mscale, vnorm);
    return (int)cudaGetLastError();
}

PD_EXPORT
int pd_qkv_norm_rope_batch(const void* q, const void* k, const void* v,
                           const void* qw, const void* kw, void* qn, void* kn,
                           void* vn, const void* positions, const void* factors,
                           uint32_t n_head, uint32_t n_kv, uint32_t head_dim,
                           float eps, float theta_scale, float freq_scale,
                           float corr_low, float corr_high, float ext_factor,
                           float mscale, uint32_t rows, void* stream) {
    return pd_qkv_norm_rope_batch_impl(q, k, v, qw, kw, qn, kn, vn, positions,
                                       factors, n_head, n_kv, head_dim, eps,
                                       theta_scale, freq_scale, corr_low,
                                       corr_high, ext_factor, mscale, rows,
                                       0u, 0u, true, true, stream);
}

// i16 twin: q/k/v are bf16 (the o16 GEMM epilogue's stream);
// outputs stay f32. Appended as its own export per the ABI growth rule.
PD_EXPORT
int pd_qkv_norm_rope_batch2(const void* q, const void* k, const void* v,
                            const void* qw, const void* kw, void* qn, void* kn,
                            void* vn, const void* positions, const void* factors,
                            uint32_t n_head, uint32_t n_kv, uint32_t head_dim,
                            float eps, float theta_scale, float freq_scale,
                            float corr_low, float corr_high, float ext_factor,
                            float mscale, uint32_t rows, uint32_t i16,
                            void* stream) {
    return pd_qkv_norm_rope_batch_impl(q, k, v, qw, kw, qn, kn, vn, positions,
                                       factors, n_head, n_kv, head_dim, eps,
                                       theta_scale, freq_scale, corr_low,
                                       corr_high, ext_factor, mscale, rows,
                                       i16, 0u, true, true, stream);
}

// a16 twin (attention streams): o16 selects the f16 output plane
// (v3 register form only - one rounding at the store, so the plane is
// exactly __float2half of the f32 plane). i16 keeps its bf16-in
// meaning. Appended as its own export per the ABI growth rule.
PD_EXPORT
int pd_qkv_norm_rope_batch3(const void* q, const void* k, const void* v,
                            const void* qw, const void* kw, void* qn, void* kn,
                            void* vn, const void* positions, const void* factors,
                            uint32_t n_head, uint32_t n_kv, uint32_t head_dim,
                            float eps, float theta_scale, float freq_scale,
                            float corr_low, float corr_high, float ext_factor,
                            float mscale, uint32_t rows, uint32_t i16,
                            uint32_t o16, void* stream) {
    return pd_qkv_norm_rope_batch_impl(q, k, v, qw, kw, qn, kn, vn, positions,
                                       factors, n_head, n_kv, head_dim, eps,
                                       theta_scale, freq_scale, corr_low,
                                       corr_high, ext_factor, mscale, rows,
                                       i16, o16, true, true, stream);
}

// rope-convention twin: `neox` picks the pair layout - 1 = the
// half-split (k, k+half) every earlier consumer of this family assumes,
// 0 = llama.cpp's ROPE_TYPE_NORM interleaved (2k, 2k+1). One superset export
// rather than three `_norm` twins, matching how batch2/batch3 grew: the flag
// joins i16/o16 as another shape bit, and the three older entries keep their
// exact signatures and their exact SASS.
PD_EXPORT
int pd_qkv_norm_rope_batch4(const void* q, const void* k, const void* v,
                            const void* qw, const void* kw, void* qn, void* kn,
                            void* vn, const void* positions, const void* factors,
                            uint32_t n_head, uint32_t n_kv, uint32_t head_dim,
                            float eps, float theta_scale, float freq_scale,
                            float corr_low, float corr_high, float ext_factor,
                            float mscale, uint32_t rows, uint32_t i16,
                            uint32_t o16, uint32_t neox, void* stream) {
    return pd_qkv_norm_rope_batch_impl(q, k, v, qw, kw, qn, kn, vn, positions,
                                       factors, n_head, n_kv, head_dim, eps,
                                       theta_scale, freq_scale, corr_low,
                                       corr_high, ext_factor, mscale, rows,
                                       i16, o16, neox != 0u, true, stream);
}

// V-epilogue twin: `vnorm` picks whether the V slots get the
// weightless per-head RMS norm. gemma4 normalizes V (gemma4.cpp:
// `Vcur = ggml_rms_norm(ctx0, Vcur, f_norm_rms_eps)`); muse-glimmer hands
// the RAW Vcur to build_attn and must not. Like `neox` this is carried by
// the architecture, not by any metadata key, and like `neox` it rides the
// superset export so batch..batch4 keep their exact signatures.
PD_EXPORT
int pd_qkv_norm_rope_batch5(const void* q, const void* k, const void* v,
                            const void* qw, const void* kw, void* qn, void* kn,
                            void* vn, const void* positions, const void* factors,
                            uint32_t n_head, uint32_t n_kv, uint32_t head_dim,
                            float eps, float theta_scale, float freq_scale,
                            float corr_low, float corr_high, float ext_factor,
                            float mscale, uint32_t rows, uint32_t i16,
                            uint32_t o16, uint32_t neox, uint32_t vnorm,
                            void* stream) {
    return pd_qkv_norm_rope_batch_impl(q, k, v, qw, kw, qn, kn, vn, positions,
                                       factors, n_head, n_kv, head_dim, eps,
                                       theta_scale, freq_scale, corr_low,
                                       corr_high, ext_factor, mscale, rows,
                                       i16, o16, neox != 0u, vnorm != 0u,
                                       stream);
}

// bf16-addend twin (the o16 down-GEMM epilogue's residual consumer):
// x (f32) += y (bf16), n elements.
__global__ void pd_add_inplace_b16_kernel(float* __restrict__ x,
                                          const __nv_bfloat16* __restrict__ y,
                                          uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] += __bfloat162float(y[i]);
}


// Batched KV append: scatter each sequence's kv row [kv_dim] into its own cache
// [batch, max_ctx, kv_dim] at its own position. grid (ceil(kv_dim/256), batch).
template<typename KV>
__global__ void pd_kv_append_batch_kernel(const float* __restrict__ kv, KV* __restrict__ cache,
                                          const unsigned int* __restrict__ positions,
                                          const unsigned int* __restrict__ slots,
                                          uint32_t kv_dim, uint32_t max_ctx, uint32_t batch) {
    PD_PDL_ARM();  // consumer-safe for early PDL launches (2026-08-31)
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t b = blockIdx.y;
    if (i >= kv_dim || b >= batch) return;
    uint32_t slot = slots ? slots[b] : b;   // prefill: many rows -> one slot
    uint32_t pos = positions[b];
    pd_kv_store(&cache[(size_t)slot * max_ctx * kv_dim + (size_t)pos * kv_dim + i],
                kv[(size_t)b * kv_dim + i]);
}

// Paged twin of pd_kv_append_batch_kernel: scatter each row's kv into the block
// pool [n_blocks, 16, kv_dim] at block_tables[slot*blocks_per_slot + pos/16],
// intra-block row pos%16. Same pd_kv_store math -> bit-exact write.
template<typename KV>
__global__ void pd_kv_append_batch_paged_kernel(const float* __restrict__ kv, KV* __restrict__ pool,
                                                const unsigned int* __restrict__ positions,
                                                const unsigned int* __restrict__ slots,
                                                const uint32_t* __restrict__ block_tables,
                                                uint32_t blocks_per_slot,
                                                uint32_t kv_dim, uint32_t batch) {
    PD_PDL_ARM_WAIT();   // programmatic dependent (pd_pdl_go): full predecessor completion
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t b = blockIdx.y;
    if (i >= kv_dim || b >= batch) return;
    uint32_t slot = slots ? slots[b] : b;   // prefill: many rows -> one slot
    uint32_t pos = positions[b];
    uint32_t blk = block_tables[(size_t)slot * blocks_per_slot + (pos >> 4)];
    uint32_t within = pos & 15u;
    pd_kv_store(&pool[(size_t)blk * 16u * kv_dim + (size_t)within * kv_dim + i],
                kv[(size_t)b * kv_dim + i]);
}

// DFlash drafter-KV conditioning fold (rung C): the block drafter's ring
// append ran k-norm + rope +
// paged K/V append as 2 + 2*cuts launches per DRAFTER LAYER per round -
// ~340 eager launches/round at 32 live (one kv_append per (cut, layer,
// k/v)), where a constant ~14 launches is achievable. This is
// the append that norms for that path: per written row it reads the RAW
// wk/wv GEMM planes, runs pd_rmsnorm_batch_kernel's k-norm VERBATIM (f64
// sumsq, float4 loads, warp + cross-warp reduce - the launcher elects the
// same nth the rmsnorm launcher would for norm_batch rows, so the
// reduction grouping is identical), stages the normed head in shared,
// ropes it on warp 0 with pd_rope_yarn_batch_kernel<true>'s per-warp theta
// chain verbatim, and stores through pd_kv_store with
// pd_kv_append_batch_paged_kernel's addressing - pool bytes BIT-IDENTICAL
// to the norm -> rope -> append chain. Bonus vs the chain: only the
// written rows (rows_w, the flattened cut windows) are normed/roped - the
// chain normed all r rows and appended a subset. The drafter ring is
// always f16, so there is no fp8 arm. grid (nw, n_kv, 2): z=0 k
// (norm+rope+store), z=1 v (plain store, kv_append's cast).
__global__ void pd_dflash_cond_append_kernel(
    const float* __restrict__ fk, const float* __restrict__ fv,
    const float* __restrict__ kw,
    __half* __restrict__ pool_k, __half* __restrict__ pool_v,
    const uint32_t* __restrict__ rows_w,
    const unsigned int* __restrict__ positions,
    const unsigned int* __restrict__ slots,
    const uint32_t* __restrict__ block_tables, uint32_t blocks_per_slot,
    uint32_t n_kv, uint32_t head_dim, float eps,
    float theta_scale, float freq_scale, float corr_low, float corr_high,
    float ext_factor, float mscale) {
    const uint32_t b = rows_w[blockIdx.x];
    const uint32_t h = blockIdx.y;
    const uint32_t kv_dim = n_kv * head_dim;
    const uint32_t pos = positions[b];
    const uint32_t slot = slots[b];
    const uint32_t blk = block_tables[(size_t)slot * blocks_per_slot + (pos >> 4)];
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    __half* dst = (blockIdx.z ? pool_v : pool_k) + (size_t)blk * 16u * kv_dim
                + (size_t)(pos & 15u) * kv_dim + (size_t)h * head_dim;
    const float* src = (blockIdx.z ? fv : fk) + (size_t)b * kv_dim + (size_t)h * head_dim;
    if (blockIdx.z) {
        for (uint32_t i = tid; i < head_dim; i += nth) pd_kv_store(&dst[i], src[i]);
        return;
    }
    // k: pd_rmsnorm_batch_kernel's body with xb=src, n=head_dim, out=shared
    __shared__ __align__(16) float s_head[256];
    __shared__ double wsum[32];
    __shared__ float s_inv;
    double acc = 0.0;
    const bool vec = (head_dim & 3u) == 0;
    if (vec) {
        const uint32_t n4 = head_dim >> 2;
        const float4* x4 = reinterpret_cast<const float4*>(src);
        for (uint32_t i = tid; i < n4; i += nth) {
            float4 v = x4[i];
            acc += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
    } else {
        for (uint32_t i = tid; i < head_dim; i += nth) acc += src[i] * src[i];
    }
    for (uint32_t s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, s);
    const uint32_t warp = tid >> 5, lane = tid & 31u;
    if (lane == 0) wsum[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        double sum = 0.0;
        const uint32_t nwarps = (nth + 31u) >> 5;
        for (uint32_t wi = 0; wi < nwarps; ++wi) sum += wsum[wi];
        s_inv = 1.0f / sqrtf((float)(sum / (double)head_dim) + eps);
    }
    __syncthreads();
    const float inv = s_inv;
    if (vec) {
        const uint32_t n4 = head_dim >> 2;
        const float4* x4 = reinterpret_cast<const float4*>(src);
        const float4* w4 = reinterpret_cast<const float4*>(kw);
        float4* o4 = reinterpret_cast<float4*>(s_head);
        for (uint32_t i = tid; i < n4; i += nth) {
            float4 v = x4[i], wv = w4[i], r;
            r.x = v.x * inv * wv.x;
            r.y = v.y * inv * wv.y;
            r.z = v.z * inv * wv.z;
            r.w = v.w * inv * wv.w;
            o4[i] = r;
        }
    } else {
        for (uint32_t i = tid; i < head_dim; i += nth) s_head[i] = src[i] * inv * kw[i];
    }
    __syncthreads();
    // rope on warp 0: pd_rope_yarn_batch_kernel<true>'s chain verbatim
    if (warp == 0) {
        const uint32_t half = head_dim / 2;
        float theta = (float)pos;
        for (uint32_t i = 0; i < lane && i < half; ++i) theta *= theta_scale;
        for (uint32_t k = lane; k < half; k += 32) {
            float y = ((float)k - corr_low) / fmaxf(0.001f, corr_high - corr_low);
            float ramp = (1.0f - fminf(1.0f, fmaxf(0.0f, y))) * ext_factor;
            float angle = (freq_scale * theta) * (1.0f - ramp) + theta * ramp;
            float s = sinf(angle) * mscale;
            float c = cosf(angle) * mscale;
            uint32_t i0 = k;
            uint32_t i1 = k + half;
            float a = s_head[i0];
            float bb = s_head[i1];
            s_head[i0] = a * c - bb * s;
            s_head[i1] = a * s + bb * c;
            for (uint32_t i = 0; i < 32 && k + i < half; ++i) theta *= theta_scale;
        }
    }
    __syncthreads();
    for (uint32_t i = tid; i < head_dim; i += nth) pd_kv_store(&dst[i], s_head[i]);
}

// Fused NORM-rope(q) + NORM-rope(k) + paged K/V append (granite decode-chain
// fold): the granite layer ran this band as four launches - rope q,
// rope k, append k, append v (9.8 us/layer of the c1 decode tick, all
// latency-bound at 1.3-3.4 us each). One warp per (row, slot): slots
// [0,n_heads) rope q in place; [n_heads, n_heads+n_kv) rope k and store the
// rotated pair STRAIGHT into the paged pool (sc.k is never read after the
// append - attention consumes the pool - so the roped plane never lands);
// the last n_kv slots append v unrotated. Rope math is
// pd_rope_yarn_batch_kernel's per-warp theta chain verbatim and the
// pool store is pd_kv_store on the identical f32 values, so cache bytes and
// q bytes are bit-identical to the four-kernel chain. Granite has no SWA
// ring (window 0 on every layer), so the append fold is safe here - the SWA
// sub-span contract that keeps gemma4's appends separate does not apply.
//
// deepseek-ocr ring arm: NEOX picks the rope pair layout (the
// same compile-time split the plain rope kernels carry), and `wpos` is the
// R-SWA ring's WRITE stream - rope always turns by the true position
// (positions[b], the reference's absolute-forever rule) while the appends
// land at the ring-mapped write slot (wpos[b]). null wpos = write at pos,
// exactly the granite behavior; the granite export pins (NEOX=false, null),
// so its bytes are untouched.
template <typename KV, bool NEOX = false>
__global__ void pd_rope_norm_qk_append_paged_kernel(
    float* __restrict__ q, float* __restrict__ k, const float* __restrict__ v,
    KV* __restrict__ k_pool, KV* __restrict__ v_pool,
    const unsigned int* __restrict__ positions, const unsigned int* __restrict__ wpos,
    const unsigned int* __restrict__ slots,
    const uint32_t* __restrict__ block_tables, uint32_t blocks_per_slot,
    uint32_t n_heads, uint32_t n_kv, uint32_t head_dim,
    float theta_scale, float freq_scale, float corr_low, float corr_high,
    float ext_factor, float mscale, uint32_t batch) {
    PD_PDL_ARM();  // cascade (granite chain)
    uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    uint32_t nslots = n_heads + 2u * n_kv;
    uint32_t idx = blockIdx.x * (blockDim.x >> 5) + warp;
    if (idx >= batch * nslots) return;
    uint32_t b = idx / nslots, si = idx % nslots;
    uint32_t pos = positions[b];
    uint32_t wp = wpos ? wpos[b] : pos;
    uint32_t kv_dim = n_kv * head_dim;
    if (si >= n_heads + n_kv) {
        // v: plain paged append - pd_kv_append_batch_paged_kernel's
        // addressing + cast, lanes striding the head
        uint32_t h = si - n_heads - n_kv;
        uint32_t slot = slots ? slots[b] : b;
        uint32_t blk = block_tables[(size_t)slot * blocks_per_slot + (wp >> 4)];
        KV* dst = v_pool + (size_t)blk * 16u * kv_dim + (size_t)(wp & 15u) * kv_dim
                + (size_t)h * head_dim;
        const float* src = v + (size_t)b * kv_dim + (size_t)h * head_dim;
        for (uint32_t i = lane; i < head_dim; i += 32u) pd_kv_store(&dst[i], src[i]);
        return;
    }
    bool is_k = si >= n_heads;
    uint32_t h = is_k ? si - n_heads : si;
    float* head = (is_k ? k + (size_t)b * kv_dim : q + (size_t)b * n_heads * head_dim)
                + (size_t)h * head_dim;
    KV* kdst = nullptr;
    if (is_k) {
        uint32_t slot = slots ? slots[b] : b;
        uint32_t blk = block_tables[(size_t)slot * blocks_per_slot + (wp >> 4)];
        kdst = k_pool + (size_t)blk * 16u * kv_dim + (size_t)(wp & 15u) * kv_dim
             + (size_t)h * head_dim;
    }
    uint32_t half = head_dim / 2;
    float theta = (float)pos;
    for (uint32_t i = 0; i < lane && i < half; ++i) theta *= theta_scale;
    for (uint32_t kk = lane; kk < half; kk += 32u) {
        float y = ((float)kk - corr_low) / fmaxf(0.001f, corr_high - corr_low);
        float ramp = (1.0f - fminf(1.0f, fmaxf(0.0f, y))) * ext_factor;
        float angle = (freq_scale * theta) * (1.0f - ramp) + theta * ramp;
        float s = sinf(angle) * mscale;
        float c = cosf(angle) * mscale;
        // pair layout per the plain rope kernels: NEOX (k, k+half), NORM
        // (2k, 2k+1) - the granite export compiles the <false> arm it always had
        uint32_t i0 = NEOX ? kk : 2u * kk;
        uint32_t i1 = NEOX ? kk + half : 2u * kk + 1u;
        float a = head[i0];
        float bb = head[i1];
        float r0 = a * c - bb * s;
        float r1 = a * s + bb * c;
        if (is_k) {
            pd_kv_store(&kdst[i0], r0);
            pd_kv_store(&kdst[i1], r1);
        } else {
            head[i0] = r0;
            head[i1] = r1;
        }
        for (uint32_t i = 0; i < 32 && kk + i < half; ++i) theta *= theta_scale;
    }
}

// Fused K/V norm+rope+append (kv-epilogue fold): the chunk band
// materialized normed K/V planes for nothing - qkv_norm_rope_batch wrote
// kn/vn (134 MB/layer at the 2048-row SWA shape) and the paged append read
// them straight back into the fp8 cache. This kernel is the append that
// norms: it reads the RAW k/v GEMM planes once, runs the v2 norm+rope math
// in registers, and stores through pd_kv_store - cache bytes BIT-IDENTICAL
// to the qkv_norm_rope_batch_v2 -> kv_append_batch_paged chain, kn/vn never
// land. (The in-GEMM epilogue fold was scoped first and dropped: gemma4
// heads are 256/512 wide, spanning 2-4 of the lin GEMM's 128-row tiles, so
// the head-wide RMS sum can't be CTA-local there.) V-less layers pass
// vp == kp: the v output is the weightless norm of the RAW k values -
// exactly what the copy-k-then-norm chain produced, sans the copy.
// Warp per (row, slot): slots 0..n_kv = k (learned norm + rope), then n_kv
// v slots (weightless norm, no rope). HD is a template param so the normed
// head stays in registers with unroll-constant indices - the rope pair
// (kk, kk+half) is lane-local because kk ≡ lane (mod 32) and half % 32 == 0.
template <bool TBL, uint32_t HD, typename KV, typename TI = float,
          bool NEOX = true>
__global__ void pd_kv_nra_rows_kernel(
    const TI* __restrict__ kp, const TI* __restrict__ vp,
    const float* __restrict__ kw, KV* __restrict__ k_pool,
    KV* __restrict__ v_pool, const unsigned int* __restrict__ positions,
    const unsigned int* __restrict__ slots, const float* __restrict__ factors,
    const uint32_t* __restrict__ block_tables, uint32_t blocks_per_slot,
    uint32_t n_kv, float eps, float theta_scale, float freq_scale,
    float corr_low, float corr_high, float ext_factor, float mscale,
    uint32_t rows, bool vnorm, uint8_t* __restrict__ vdim = nullptr) {
    const uint32_t nslots = 2u * n_kv;
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t idx = blockIdx.x * (blockDim.x >> 5) + warp;
    if (idx >= rows * nslots) return;
    const uint32_t b = idx / nslots, si = idx % nslots;
    const bool is_v = si >= n_kv;
    const uint32_t h = is_v ? si - n_kv : si;
    const uint32_t kv_dim = n_kv * HD;
    const TI* src = (is_v ? vp : kp) + (size_t)b * kv_dim + (size_t)h * HD;

    // vnorm=false (muse-glimmer): V rides through un-normed, so the whole
    // sum-of-squares pass is dead work on those slots. `idx` is a per-WARP
    // index, so is_v is warp-uniform and this branch never diverges.
    const bool v_raw = is_v && !vnorm;
    float inv = 1.0f;
    if (!v_raw) {
        // v2's reduction shape verbatim (32-element blocks each shfl-tree
        // reduced, block results summed ascending) - inv matches
        // pd_qkv_norm_rope_batch_v2 bit-for-bit at every HD
        float sum = 0.0f;
        #pragma unroll
        for (uint32_t w2 = 0; w2 < 8u; ++w2) {
            float acc = 0.0f;
            for (uint32_t i = w2 * 32u + lane; i < HD; i += 256u) {
                const float x = (float)src[i];
                acc += x * x;
            }
            for (uint32_t sh = 16; sh > 0; sh >>= 1)
                acc += __shfl_down_sync(0xffffffffu, acc, sh);
            acc = __shfl_sync(0xffffffffu, acc, 0);
            sum += acc;
        }
        inv = 1.0f / sqrtf(sum / (float)HD + eps);
    }

    // normed head in registers, v2's store expression exactly (k learned
    // weight, v weightless)
    constexpr uint32_t NV = HD / 32u;
    float vals[NV];
    #pragma unroll
    for (uint32_t j = 0; j < NV; ++j) {
        const uint32_t i = j * 32u + lane;
        vals[j] = is_v ? (float)src[i] * inv : (float)src[i] * inv * kw[i];
        // (v_raw leaves inv == 1.0f, so the V expression is a plain load)
    }

    if (!is_v && NEOX) {
        // rope: pd_qkv_norm_rope_batch_v2's warp chain verbatim; kk = lane
        // + m*32 makes vals[kk>>5] = vals[m] an unroll constant
        constexpr uint32_t half = HD / 2u;
        const float l2ts = TBL ? log2f(theta_scale) : 0.0f;
        const float pos_f = (float)positions[b];
        float theta = pos_f;
        if (!TBL)
            for (uint32_t i = 0; i < lane && i < half; ++i) theta *= theta_scale;
        #pragma unroll
        for (uint32_t m = 0; m < half / 32u; ++m) {
            const uint32_t kk = m * 32u + lane;
            if (TBL) theta = pos_f * exp2f((float)kk * l2ts);
            float t = factors ? theta / factors[kk] : theta;
            float y = ((float)kk - corr_low) / fmaxf(0.001f, corr_high - corr_low);
            float ramp = (1.0f - fminf(1.0f, fmaxf(0.0f, y))) * ext_factor;
            float angle = (freq_scale * t) * (1.0f - ramp) + t * ramp;
            float s = sinf(angle) * mscale;
            float c = cosf(angle) * mscale;
            float a = vals[m];
            float bb = vals[m + half / 32u];
            vals[m] = a * c - bb * s;
            vals[m + half / 32u] = a * s + bb * c;
            if (!TBL)
                for (uint32_t i = 0; i < 32 && kk + i < half; ++i) theta *= theta_scale;
        }
    } else if (!is_v) {
        // ROPE_TYPE_NORM (muse-glimmer). Same shape as
        // pd_qkv_norm_rope_batch_v3's NORM arm and for the same reason: the
        // NEOX partner (kk+half) is register-local because half % 32 == 0,
        // but NORM's partner is the ADJACENT ELEMENT, which in this
        // lane-strided staging (vals[j] holds element j*32+lane) lives in
        // the ADJACENT LANE of the same register. One shfl_xor(...,1) closes
        // the pair and each lane keeps its own half of the rotation. The
        // whole warp is a K slot here, so the full mask is honest.
        const float l2ts = TBL ? log2f(theta_scale) : 0.0f;
        const float pos_f = (float)positions[b];
        const bool odd = (lane & 1u) != 0u;
        #pragma unroll
        for (uint32_t j = 0; j < NV; ++j) {
            const uint32_t e = j * 32u + lane;   // this lane's element
            const uint32_t kk = e >> 1;          // its rotation-pair index
            float theta = pos_f;
            if (TBL) {
                theta = pos_f * exp2f((float)kk * l2ts);
            } else {
                for (uint32_t i = 0; i < kk; ++i) theta *= theta_scale;
            }
            const float t = factors ? theta / factors[kk] : theta;
            const float y = ((float)kk - corr_low) / fmaxf(0.001f, corr_high - corr_low);
            const float ramp = (1.0f - fminf(1.0f, fmaxf(0.0f, y))) * ext_factor;
            const float angle = (freq_scale * t) * (1.0f - ramp) + t * ramp;
            const float s = sinf(angle) * mscale;
            const float c = cosf(angle) * mscale;
            const float mine = vals[j];
            const float other = __shfl_xor_sync(0xffffffffu, mine, 1u);
            const float a = odd ? other : mine;   // even element of the pair
            const float bb = odd ? mine : other;  // odd element of the pair
            vals[j] = odd ? (a * s + bb * c) : (a * c - bb * s);
        }
    }

    // paged append: pd_kv_append_batch_paged_kernel's addressing + cast
    const uint32_t slot = slots ? slots[b] : b;
    const uint32_t pos = positions[b];
    const uint32_t blk = block_tables[(size_t)slot * blocks_per_slot + (pos >> 4)];
    KV* dst = (is_v ? v_pool : k_pool) + (size_t)blk * 16u * kv_dim
        + (size_t)(pos & 15u) * kv_dim + (size_t)h * HD;
    #pragma unroll
    for (uint32_t j = 0; j < NV; ++j) pd_kv_store(&dst[j * 32u + lane], vals[j]);
    //  writer-fused double-store: re-read this warp's own just-written
    // bytes so the twin is bit-identical to the pool (fp8 paged only)
    if (is_v && vdim && sizeof(KV) == 1) {
        uint8_t* vd = vdim + ((size_t)blk * kv_dim + (size_t)h * HD) * 16u
                    + (pos & 15u);
        #pragma unroll
        for (uint32_t j = 0; j < NV; ++j) {
            const uint32_t i = j * 32u + lane;
            vd[(size_t)i * 16u] = ((const uint8_t*)dst)[i];
        }
    }
}

static int pd_kv_nra_rows_impl(const void* kp, const void* vp, const void* kw,
                               void* k_pool, void* v_pool, const void* positions,
                               const void* slots, const void* factors,
                               const void* block_tables, uint32_t blocks_per_slot,
                               uint32_t n_kv, uint32_t head_dim, float eps,
                               float theta_scale, float freq_scale, float corr_low,
                               float corr_high, float ext_factor, float mscale,
                               uint32_t rows, uint32_t kv_dtype, uint32_t i16,
                               bool neox, bool vnorm, void* stream) {
    if (rows == 0 || n_kv == 0) return 0;
    if (head_dim != 128u && head_dim != 256u && head_dim != 512u)
        return cudaErrorInvalidValue;
    // rope theta mode must match the q-side v2 launcher's latch (same env)
    // or the k cache would ride a different rounding class than q
    static int no_tbl = -1;
    if (no_tbl < 0) no_tbl = pd_env("PADDOCK_NO_RTBL") ? 1 : 0;
    const uint32_t total = rows * 2u * n_kv;
    const uint32_t blocks = (total + 7u) / 8u;
    auto st = (cudaStream_t)stream;
    #define PD_KVNRA_GO(TBL_, HD_, KV_, TI_, NEOX_)                            \
        pd_kv_nra_rows_kernel<TBL_, HD_, KV_, TI_, NEOX_>                       \
            <<<blocks, 256, 0, st>>>(                                          \
            (const TI_*)kp, (const TI_*)vp, (const float*)kw,                  \
            (KV_*)k_pool, (KV_*)v_pool, (const unsigned int*)positions,        \
            (const unsigned int*)slots, (const float*)factors,                 \
            (const uint32_t*)block_tables, blocks_per_slot, n_kv, eps,         \
            theta_scale, freq_scale, corr_low, corr_high, ext_factor, mscale,  \
            rows, vnorm, (uint8_t*)pd_vdim_base)
    #define PD_KVNRA_NX(TBL_, HD_, KV_, TI_)                                   \
        do {                                                                   \
            if (neox) PD_KVNRA_GO(TBL_, HD_, KV_, TI_, true);                  \
            else PD_KVNRA_GO(TBL_, HD_, KV_, TI_, false);                      \
        } while (0)
    #define PD_KVNRA_HD(TBL_, KV_, TI_)                                        \
        do {                                                                   \
            if (head_dim == 128u) PD_KVNRA_NX(TBL_, 128u, KV_, TI_);           \
            else if (head_dim == 256u) PD_KVNRA_NX(TBL_, 256u, KV_, TI_);      \
            else PD_KVNRA_NX(TBL_, 512u, KV_, TI_);                            \
        } while (0)
    #define PD_KVNRA_TI(TBL_, KV_)                                             \
        do {                                                                   \
            if (i16) PD_KVNRA_HD(TBL_, KV_, __nv_bfloat16);                    \
            else PD_KVNRA_HD(TBL_, KV_, float);                                \
        } while (0)
    if (kv_dtype == PD_KV_FP8_E4M3) {
        if (no_tbl) PD_KVNRA_TI(false, __nv_fp8_e4m3);
        else PD_KVNRA_TI(true, __nv_fp8_e4m3);
    } else {
        if (no_tbl) PD_KVNRA_TI(false, __half);
        else PD_KVNRA_TI(true, __half);
    }
    #undef PD_KVNRA_TI
    #undef PD_KVNRA_HD
    #undef PD_KVNRA_NX
    #undef PD_KVNRA_GO
    return (int)cudaGetLastError();
}

PD_EXPORT
int pd_kv_nra_rows(const void* kp, const void* vp, const void* kw,
                   void* k_pool, void* v_pool, const void* positions,
                   const void* slots, const void* factors,
                   const void* block_tables, uint32_t blocks_per_slot,
                   uint32_t n_kv, uint32_t head_dim, float eps,
                   float theta_scale, float freq_scale, float corr_low,
                   float corr_high, float ext_factor, float mscale,
                   uint32_t rows, uint32_t kv_dtype, void* stream) {
    return pd_kv_nra_rows_impl(kp, vp, kw, k_pool, v_pool, positions, slots,
                               factors, block_tables, blocks_per_slot, n_kv,
                               head_dim, eps, theta_scale, freq_scale,
                               corr_low, corr_high, ext_factor, mscale, rows,
                               kv_dtype, 0u, true, true, stream);
}

// i16 twin: the raw k/v GEMM planes are bf16 (o16 epilogue
// stream). Appended as its own export per the ABI growth rule.
PD_EXPORT
int pd_kv_nra_rows2(const void* kp, const void* vp, const void* kw,
                    void* k_pool, void* v_pool, const void* positions,
                    const void* slots, const void* factors,
                    const void* block_tables, uint32_t blocks_per_slot,
                    uint32_t n_kv, uint32_t head_dim, float eps,
                    float theta_scale, float freq_scale, float corr_low,
                    float corr_high, float ext_factor, float mscale,
                    uint32_t rows, uint32_t kv_dtype, uint32_t i16,
                    void* stream) {
    return pd_kv_nra_rows_impl(kp, vp, kw, k_pool, v_pool, positions, slots,
                               factors, block_tables, blocks_per_slot, n_kv,
                               head_dim, eps, theta_scale, freq_scale,
                               corr_low, corr_high, ext_factor, mscale, rows,
                               kv_dtype, i16, true, true, stream);
}

// arch-constant twin: `neox` picks the rope pair layout and
// `vnorm` whether the V slots get the weightless per-head RMS norm - the
// same two constants pd_qkv_norm_rope_batch5 carries, because this kernel
// is that one's K/V half folded into the paged append. gemma4 is (1, 1);
// muse-glimmer is (0, 0). Appended per the ABI growth rule; kv_nra_rows and
// kv_nra_rows2 keep their exact signatures.
PD_EXPORT
int pd_kv_nra_rows3(const void* kp, const void* vp, const void* kw,
                    void* k_pool, void* v_pool, const void* positions,
                    const void* slots, const void* factors,
                    const void* block_tables, uint32_t blocks_per_slot,
                    uint32_t n_kv, uint32_t head_dim, float eps,
                    float theta_scale, float freq_scale, float corr_low,
                    float corr_high, float ext_factor, float mscale,
                    uint32_t rows, uint32_t kv_dtype, uint32_t i16,
                    uint32_t neox, uint32_t vnorm, void* stream) {
    return pd_kv_nra_rows_impl(kp, vp, kw, k_pool, v_pool, positions, slots,
                               factors, block_tables, blocks_per_slot, n_kv,
                               head_dim, eps, theta_scale, freq_scale,
                               corr_low, corr_high, ext_factor, mscale, rows,
                               kv_dtype, i16, neox != 0u, vnorm != 0u, stream);
}

// Laguna decode-tick epilogue fold: one launch replaces the
// six-kernel chain q-norm + k-norm + rope(q) + rope(k) + append(k) +
// append(v) that ran per layer per tick (~6 launches x ~2 µs of tiny-grid
// latency x 40 layers). Warp per (row, slot): slots 0..n_head = q (learned
// norm + rope -> q_out), then n_kv k slots (learned norm + rope + paged
// append), then n_kv v slots (PLAIN paged append - laguna has no v-norm).
// Norm = pd_qkv_norm_rope_batch_v2's reduction + store expression verbatim;
// yarn rope = pd_kv_nra_rows' non-TBL chain verbatim (factors-null form);
// mrope (MR=true, full layers) = pd_mrope_kernel's per-pair math verbatim
// (partial rotary: pairs (p, p+NROT/2) for p < NROT/2, tail passes through
// normed-only). Appends = pd_kv_append_batch_paged_kernel's addressing +
// pd_kv_store cast. Everything downstream is BIT-IDENTICAL to the chain.
// NROT is a template param so the mrope pair regs index unroll-constant
// (laguna: HD 128, NROT 64 - both rope arms lane-local, no shuffles).
// q/k may share a fused GEMV plane (r==1 [q|k|gate] row): q_off/k_off pick
// the segments; separate planes pass k_src with k_off 0.
// QG arm (qwen3.5 family): q heads sit [q(HD)|gate(HD)]
// interleaved per head in the fused plane (head h at q_off + h*2*HD); the
// same warp also copies the RAW gate half to gate_out - split_qg never runs.
// HD 256 mirrors pd_rmsnorm_batch's n=256 reduction (threads 0..63 each dot
// one float4, warp trees over f4 0..31 and 32..63, block sum w0+w1 - upper
// warps contribute exact zeros at every launch width, so inv matches the
// chain bit-for-bit at any batch).
template <uint32_t HD, uint32_t NROT, typename KV, bool MR, bool QG = false>
__global__ void pd_lag_qk_nra_rows_kernel(
    const float* __restrict__ q_src, uint32_t q_off, uint32_t q_stride,
    const float* __restrict__ k_src, uint32_t k_off, uint32_t k_stride,
    const float* __restrict__ v_src, uint32_t v_stride,
    const float* __restrict__ qw, const float* __restrict__ kw,
    float* __restrict__ q_out, float* __restrict__ gate_out,
    KV* __restrict__ k_pool, KV* __restrict__ v_pool,
    const unsigned int* __restrict__ positions, const unsigned int* __restrict__ slots,
    const unsigned int* __restrict__ mpos, const uint32_t* __restrict__ block_tables,
    uint32_t blocks_per_slot, uint32_t n_head, uint32_t n_kv, float eps,
    float theta_scale, float freq_scale, float corr_low, float corr_high,
    float ext_factor, float mscale, uint32_t s0, uint32_t s1, uint32_t s2,
    uint32_t s3, uint32_t rows, int acc_mode) {
    // cascade: q/k/v planes come straight from the predecessor GEMV
    PD_PDL_ARM();
    const uint32_t nslots = n_head + 2u * n_kv;
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t idx = blockIdx.x * (blockDim.x >> 5) + warp;
    if (idx >= rows * nslots) return;
    const uint32_t b = idx / nslots, si = idx % nslots;
    const bool is_q = si < n_head;
    const bool is_v = si >= n_head + n_kv;
    const uint32_t h = is_q ? si : (is_v ? si - n_head - n_kv : si - n_head);
    const uint32_t kv_dim = n_kv * HD;
    const float* src = is_v
        ? v_src + (size_t)b * v_stride + (size_t)h * HD
        : (is_q ? q_src + (size_t)b * q_stride + q_off
                      + (size_t)h * (QG ? 2u : 1u) * HD
                : k_src + (size_t)b * k_stride + k_off + (size_t)h * HD);

    constexpr uint32_t NV = HD / 32u;
    float vals[NV];
    if (is_v) {
        // laguna v: no norm, no rope - straight through
        #pragma unroll
        for (uint32_t j = 0; j < NV; ++j) vals[j] = src[j * 32u + lane];
    } else {
        // the decode norms run through pd_rmsnorm_batch_kernel - replicate
        // its reduction (not v2's strided-block shape) so inv matches the
        // chain bit-for-bit. hd128: lanes 0..31 each sum one quad, one warp
        // tree, upper warps add exact zeros. hd256: threads 0..63 each dot
        // one float4 -> two warp trees (f4 0..31, 32..63), block sum w0+w1
        // ascending - identical bits at every rmsnorm launch width.
        static_assert(HD == 128u || HD == 256u,
                      "reduction mirrors rmsnorm_batch at hd128/hd256");
        // acc_mode = pd_norm_acc_mode(), the chain's accumulator election
        // (pd_rmsnorm_batch_kernel_t<ACC>): f32 products everywhere; DF
        // (default) accumulates each thread's float4 in double-float, DF warp
        // trees, the warp partials merged in order, one f64 divide; F64 the
        // same in double. Warp (f4 0..31) then (32..63) are the chain's warps
        // 0 and 1; its further warps hold exact zeros.
        float inv;
        if (acc_mode != PD_ACC_F32) {
            const float4 va = *(const float4*)(src + 4u * lane);
            const float4 vb = HD == 256u ? *(const float4*)(src + 4u * (32u + lane)) : va;
            double total;
            if (acc_mode == PD_ACC_DF) {
                auto quad = [](const float4 v) {
                    pd_df a;
                    a.hi = 0.0f; a.lo = 0.0f;
                    pd_df_add(a, v.x * v.x);
                    pd_df_add(a, v.y * v.y);
                    pd_df_add(a, v.z * v.z);
                    pd_df_add(a, v.w * v.w);
                    return a;
                };
                auto tree = [](pd_df a) {
                    for (uint32_t sh = 16; sh > 0; sh >>= 1) {
                        pd_df o;
                        o.hi = __shfl_down_sync(0xffffffffu, a.hi, sh);
                        o.lo = __shfl_down_sync(0xffffffffu, a.lo, sh);
                        a = pd_df_merge(a, o);
                    }
                    a.hi = __shfl_sync(0xffffffffu, a.hi, 0);
                    a.lo = __shfl_sync(0xffffffffu, a.lo, 0);
                    return a;
                };
                pd_df sm;
                sm.hi = 0.0f; sm.lo = 0.0f;
                sm = pd_df_merge(sm, tree(quad(va)));
                if (HD == 256u) sm = pd_df_merge(sm, tree(quad(vb)));
                total = (double)sm.hi + (double)sm.lo;
            } else {
                auto tree = [](double a) {
                    for (uint32_t sh = 16; sh > 0; sh >>= 1) a += __shfl_down_sync(0xffffffffu, a, sh);
                    return __shfl_sync(0xffffffffu, a, 0);
                };
                double a = 0.0, b2 = 0.0;
                a += va.x * va.x + va.y * va.y + va.z * va.z + va.w * va.w;
                b2 += vb.x * vb.x + vb.y * vb.y + vb.z * vb.z + vb.w * vb.w;
                double sm = 0.0;
                sm += tree(a);
                if (HD == 256u) sm += tree(b2);
                total = sm;
            }
            inv = 1.0f / sqrtf((float)(total / (double)HD) + eps);
        } else {
        float sum;
        if constexpr (HD == 128u) {
            float acc = 0.0f;
            {
                const float4 v4 = *(const float4*)(src + 4u * lane);
                acc += v4.x * v4.x + v4.y * v4.y + v4.z * v4.z + v4.w * v4.w;
            }
            for (uint32_t sh = 16; sh > 0; sh >>= 1)
                acc += __shfl_down_sync(0xffffffffu, acc, sh);
            sum = __shfl_sync(0xffffffffu, acc, 0);
        } else {
            const float4 va = *(const float4*)(src + 4u * lane);
            const float4 vb = *(const float4*)(src + 4u * (32u + lane));
            float aa = va.x * va.x + va.y * va.y + va.z * va.z + va.w * va.w;
            float ab = vb.x * vb.x + vb.y * vb.y + vb.z * vb.z + vb.w * vb.w;
            for (uint32_t sh = 16; sh > 0; sh >>= 1) {
                aa += __shfl_down_sync(0xffffffffu, aa, sh);
                ab += __shfl_down_sync(0xffffffffu, ab, sh);
            }
            const float w0 = __shfl_sync(0xffffffffu, aa, 0);
            const float w1 = __shfl_sync(0xffffffffu, ab, 0);
            sum = w0 + w1;
        }
        inv = 1.0f / sqrtf(sum / (float)HD + eps);
        }
        const float* w = is_q ? qw : kw;
        #pragma unroll
        for (uint32_t j = 0; j < NV; ++j) {
            const uint32_t i = j * 32u + lane;
            vals[j] = src[i] * inv * w[i];
        }
        if (!MR) {
            // full-width yarn: pd_kv_nra_rows' non-TBL chain verbatim
            constexpr uint32_t half = HD / 2u;
            float theta = (float)positions[b];
            for (uint32_t i = 0; i < lane && i < half; ++i) theta *= theta_scale;
            #pragma unroll
            for (uint32_t m = 0; m < half / 32u; ++m) {
                const uint32_t kk = m * 32u + lane;
                float y = ((float)kk - corr_low) / fmaxf(0.001f, corr_high - corr_low);
                float ramp = (1.0f - fminf(1.0f, fmaxf(0.0f, y))) * ext_factor;
                float angle = (freq_scale * theta) * (1.0f - ramp) + theta * ramp;
                float s = sinf(angle) * mscale;
                float c = cosf(angle) * mscale;
                float a = vals[m];
                float bb = vals[m + half / 32u];
                vals[m] = a * c - bb * s;
                vals[m + half / 32u] = a * s + bb * c;
                for (uint32_t i = 0; i < 32 && kk + i < half; ++i) theta *= theta_scale;
            }
        } else {
            // partial sectioned mrope: pd_mrope_kernel per-pair math verbatim;
            // pair (p, p+NROT/2) lives at (vals[m], vals[m + NROT/64]) since
            // p ≡ lane (mod 32) and NROT/2 % 32 == 0
            constexpr uint32_t half_r = NROT / 2u;
            const uint32_t sect = s0 + s1 + s2 + s3;
            const uint32_t sec_h = s0, sec_w = s0 + s1, sec_e = s0 + s1 + s2;
            #pragma unroll
            for (uint32_t m = 0; m < half_r / 32u; ++m) {
                const uint32_t p = m * 32u + lane;
                const uint32_t sector = p % sect;
                float base;
                if (sector < sec_h) base = (float)mpos[b];
                else if (sector < sec_w) base = (float)mpos[(size_t)rows + b];
                else if (sector < sec_e) base = (float)mpos[(size_t)2 * rows + b];
                else base = (float)mpos[(size_t)3 * rows + b];
                for (uint32_t i = 0; i < p; ++i) base *= theta_scale;
                float y = ((float)p - corr_low) / fmaxf(0.001f, corr_high - corr_low);
                float ramp = (1.0f - fminf(1.0f, fmaxf(0.0f, y))) * ext_factor;
                float angle = (freq_scale * base) * (1.0f - ramp) + base * ramp;
                float sn = sinf(angle) * mscale;
                float cs = cosf(angle) * mscale;
                float a = vals[m];
                float bb = vals[m + half_r / 32u];
                vals[m] = a * cs - bb * sn;
                vals[m + half_r / 32u] = a * sn + bb * cs;
            }
        }
    }

    if (is_q) {
        float* dst = q_out + ((size_t)b * n_head + h) * HD;
        #pragma unroll
        for (uint32_t j = 0; j < NV; ++j) dst[j * 32u + lane] = vals[j];
        if constexpr (QG) {
            // raw gate half rides the q warp: src+HD is this head's gate -
            // pd_split_qg's exact bytes (no norm, no rope), plane layout
            // [rows, n_head*HD] like q_out
            float* gd = gate_out + ((size_t)b * n_head + h) * HD;
            #pragma unroll
            for (uint32_t j = 0; j < NV; ++j)
                gd[j * 32u + lane] = src[HD + j * 32u + lane];
        }
    } else {
        // paged append: pd_kv_append_batch_paged_kernel's addressing + cast
        const uint32_t slot = slots ? slots[b] : b;
        const uint32_t pos = positions[b];
        const uint32_t blk = block_tables[(size_t)slot * blocks_per_slot + (pos >> 4)];
        KV* dst = (is_v ? v_pool : k_pool) + (size_t)blk * 16u * kv_dim
            + (size_t)(pos & 15u) * kv_dim + (size_t)h * HD;
        #pragma unroll
        for (uint32_t j = 0; j < NV; ++j) pd_kv_store(&dst[j * 32u + lane], vals[j]);
    }
}

PD_EXPORT
int pd_lag_qk_nra_rows(const void* q_src, uint32_t q_off, uint32_t q_stride,
                       const void* k_src, uint32_t k_off, uint32_t k_stride,
                       const void* v_src, uint32_t v_stride, const void* qw,
                       const void* kw, void* q_out, void* k_pool, void* v_pool,
                       const void* positions, const void* slots, const void* mpos,
                       const void* block_tables, uint32_t blocks_per_slot,
                       uint32_t n_head, uint32_t n_kv, uint32_t head_dim,
                       uint32_t n_rot, float eps, float theta_scale,
                       float freq_scale, float corr_low, float corr_high,
                       float ext_factor, float mscale, uint32_t s0, uint32_t s1,
                       uint32_t s2, uint32_t s3, uint32_t rows,
                       uint32_t kv_dtype, void* stream) {
    if (rows == 0 || n_head == 0 || n_kv == 0) return 0;
    // instantiated for the laguna shape only: hd 128; mrope needs n_rot 64
    // (the lane-local pair map) - anything else falls back to the chain
    if (head_dim != 128u) return cudaErrorInvalidValue;
    if (mpos != nullptr && n_rot != 64u) return cudaErrorInvalidValue;
    const uint32_t total = rows * (n_head + 2u * n_kv);
    const uint32_t blocks = (total + 7u) / 8u;
    auto st = (cudaStream_t)stream;
    #define PD_LQKNRA_GO(KV_, MR_)                                             \
        pd_pdl_go(pd_lag_qk_nra_rows_kernel<128u, 64u, KV_, MR_>, blocks, 256, 0u, st, \
            (const float*)q_src, q_off, q_stride, (const float*)k_src, k_off,  \
            k_stride, (const float*)v_src, v_stride, (const float*)qw,         \
            (const float*)kw, (float*)q_out, nullptr, (KV_*)k_pool, (KV_*)v_pool, \
            (const unsigned int*)positions, (const unsigned int*)slots,        \
            (const unsigned int*)mpos, (const uint32_t*)block_tables,          \
            blocks_per_slot, n_head, n_kv, eps, theta_scale, freq_scale,       \
            corr_low, corr_high, ext_factor, mscale, s0, s1, s2, s3, rows,     \
            pd_norm_acc_mode())
    if (kv_dtype == PD_KV_FP8_E4M3) {
        if (mpos) PD_LQKNRA_GO(__nv_fp8_e4m3, true);
        else PD_LQKNRA_GO(__nv_fp8_e4m3, false);
    } else {
        if (mpos) PD_LQKNRA_GO(__half, true);
        else PD_LQKNRA_GO(__half, false);
    }
    #undef PD_LQKNRA_GO
    return (int)cudaGetLastError();
}

// Qwen3.5-family fused-plane prefill consumer: the mixed-tick /
// batched-prefill chain split_qg + rmsnorm(q) + rmsnorm(k) + mrope(q) +
// mrope(k) + append(k) + append(v) - 7 launches over planes the one-GEMM
// qkv output already holds - becomes this one launch. Same template as the
// laguna fold with the QG arm on: q heads are [q(HD)|gate(HD)] interleaved
// in the fused plane (head h at q_off + h*2*HD), the raw gate half lands in
// gate_out, k/v append paged through pd_kv_store. hd 256 / n_rot 64 (qwen3.6
// partial rotary 0.25); mpos is required (4-axis sectioned mrope - the
// qwen3.5 rope class). Everything downstream BIT-IDENTICAL to the chain
// (checksum-gated). Slice-merge note: the earlier
// falsification priced split COPIES after a merged GEMM; this is its escape
// clause - a strided consumer, no copies ever land.
PD_EXPORT
int pd_q36_qkg_nra_rows(const void* qkg, uint32_t q_off, uint32_t row_stride,
                        uint32_t k_off, uint32_t v_off, const void* qw,
                        const void* kw, void* q_out, void* gate_out,
                        void* k_pool, void* v_pool, const void* positions,
                        const void* slots, const void* mpos,
                        const void* block_tables, uint32_t blocks_per_slot,
                        uint32_t n_head, uint32_t n_kv, uint32_t head_dim,
                        uint32_t n_rot, float eps, float theta_scale,
                        float freq_scale, float corr_low, float corr_high,
                        float ext_factor, float mscale, uint32_t s0, uint32_t s1,
                        uint32_t s2, uint32_t s3, uint32_t rows,
                        uint32_t kv_dtype, void* stream) {
    if (rows == 0 || n_head == 0 || n_kv == 0) return 0;
    // instantiated for the qwen3.6-27b shape: hd 256, sectioned-mrope n_rot
    // 64 (lane-local pair map); anything else falls back to the chain
    if (head_dim != 256u || n_rot != 64u || mpos == nullptr)
        return cudaErrorInvalidValue;
    const uint32_t total = rows * (n_head + 2u * n_kv);
    const uint32_t blocks = (total + 7u) / 8u;
    auto st = (cudaStream_t)stream;
    #define PD_Q36NRA_GO(KV_)                                                  \
        pd_pdl_go(pd_lag_qk_nra_rows_kernel<256u, 64u, KV_, true, true>,       \
            blocks, 256, 0u, st,                                               \
            (const float*)qkg, q_off, row_stride, (const float*)qkg, k_off,    \
            row_stride, (const float*)qkg + v_off, row_stride,                 \
            (const float*)qw, (const float*)kw, (float*)q_out,                 \
            (float*)gate_out, (KV_*)k_pool, (KV_*)v_pool,                      \
            (const unsigned int*)positions, (const unsigned int*)slots,        \
            (const unsigned int*)mpos, (const uint32_t*)block_tables,          \
            blocks_per_slot, n_head, n_kv, eps, theta_scale, freq_scale,       \
            corr_low, corr_high, ext_factor, mscale, s0, s1, s2, s3, rows,     \
            pd_norm_acc_mode())
    if (kv_dtype == PD_KV_FP8_E4M3) PD_Q36NRA_GO(__nv_fp8_e4m3);
    else PD_Q36NRA_GO(__half);
    #undef PD_Q36NRA_GO
    return (int)cudaGetLastError();
}

// Fused QKV consumer for gpt-oss (no q/k norms): one launch replaces
// rope_yarn_batch(q) + rope_yarn_batch(k) + kv_append(k) + kv_append(v) on
// the fused-QKV GEMM output [batch, qdim + 2*kvdim]. Rope math is copied
// VERBATIM from pd_rope_yarn_batch_kernel (identical theta chain -> the
// rotated values are bit-identical; they just come from the fused row and
// land directly in d_q / the caches). Warp ranges: q heads, then k heads
// (rope + store), then v heads (plain store).
template<typename KV>
__global__ void pd_qkv_rope_append_batch_kernel(
    const float* __restrict__ qkv, float* __restrict__ q_out, KV* __restrict__ k_cache,
    KV* __restrict__ v_cache, const unsigned int* __restrict__ positions,
    const unsigned int* __restrict__ slots, uint32_t n_heads, uint32_t n_kv_heads,
    uint32_t head_dim, uint32_t max_ctx, float theta_scale, float freq_scale,
    float corr_low, float corr_high, float ext_factor, float mscale, uint32_t batch) {
    const uint32_t qdim = n_heads * head_dim, kvdim = n_kv_heads * head_dim;
    const uint32_t rowd = qdim + 2u * kvdim;
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t idx = blockIdx.x * (blockDim.x >> 5) + warp;
    const uint32_t nq = batch * n_heads, nk = batch * n_kv_heads;
    const uint32_t half = head_dim / 2u;

    if (idx < nq + nk) {
        // rope a q or k head (verbatim pd_rope_yarn_batch_kernel math)
        const bool is_q = idx < nq;
        const uint32_t hidx = is_q ? idx : idx - nq;
        const uint32_t nh = is_q ? n_heads : n_kv_heads;
        const uint32_t b = hidx / nh, h = hidx % nh;
        const float* src = qkv + (size_t)b * rowd + (is_q ? 0u : qdim) + (size_t)h * head_dim;
        float theta = (float)positions[b];
        for (uint32_t i = 0; i < lane && i < half; ++i) theta *= theta_scale;
        const uint32_t slot = slots ? slots[b] : b;
        const uint32_t pos = positions[b];
        for (uint32_t k = lane; k < half; k += 32) {
            float y = ((float)k - corr_low) / fmaxf(0.001f, corr_high - corr_low);
            float ramp = (1.0f - fminf(1.0f, fmaxf(0.0f, y))) * ext_factor;
            float angle = (freq_scale * theta) * (1.0f - ramp) + theta * ramp;
            float sn = sinf(angle) * mscale;
            float cs = cosf(angle) * mscale;
            float a = src[k];
            float bb = src[k + half];
            float r0 = a * cs - bb * sn;
            float r1 = a * sn + bb * cs;
            if (is_q) {
                float* dst = q_out + (size_t)b * qdim + (size_t)h * head_dim;
                dst[k] = r0;
                dst[k + half] = r1;
            } else {
                KV* dst = k_cache + (size_t)slot * max_ctx * kvdim + (size_t)pos * kvdim +
                          (size_t)h * head_dim;
                pd_kv_store(&dst[k], r0);
                pd_kv_store(&dst[k + half], r1);
            }
            for (uint32_t i = 0; i < 32 && k + i < half; ++i) theta *= theta_scale;
        }
    } else if (idx < nq + nk + nk) {
        // v head: straight store into the cache
        const uint32_t hidx = idx - nq - nk;
        const uint32_t b = hidx / n_kv_heads, h = hidx % n_kv_heads;
        const uint32_t slot = slots ? slots[b] : b;
        const uint32_t pos = positions[b];
        const float* src = qkv + (size_t)b * rowd + qdim + kvdim + (size_t)h * head_dim;
        KV* dst = v_cache + (size_t)slot * max_ctx * kvdim + (size_t)pos * kvdim +
                  (size_t)h * head_dim;
        for (uint32_t i = lane; i < head_dim; i += 32) pd_kv_store(&dst[i], src[i]);
    }
}

// Paged twin of pd_qkv_rope_append_batch_kernel (gpt-oss b>64 mixed/prefill
// append). Byte-for-byte the same rope + store math; only the K/V cache base
// swaps the dense slot*max_ctx*kvdim stride for a block-table lookup into the
// [n_blocks, 16, kvdim] pool (blk = block_tables[slot*bps + pos/16], intra-block
// row pos%16), preserving the per-head + h*head_dim term. max_ctx drops (its
// only use was the dense base). Bit-exact vs dense under an identity table.
template<typename KV>
__global__ void pd_qkv_rope_append_batch_paged_kernel(
    const float* __restrict__ qkv, float* __restrict__ q_out, KV* __restrict__ k_cache,
    KV* __restrict__ v_cache, const unsigned int* __restrict__ positions,
    const unsigned int* __restrict__ slots, uint32_t n_heads, uint32_t n_kv_heads,
    uint32_t head_dim, float theta_scale, float freq_scale,
    float corr_low, float corr_high, float ext_factor, float mscale, uint32_t batch,
    const uint32_t* __restrict__ block_tables, uint32_t blocks_per_slot) {
    const uint32_t qdim = n_heads * head_dim, kvdim = n_kv_heads * head_dim;
    const uint32_t rowd = qdim + 2u * kvdim;
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t idx = blockIdx.x * (blockDim.x >> 5) + warp;
    const uint32_t nq = batch * n_heads, nk = batch * n_kv_heads;
    const uint32_t half = head_dim / 2u;

    if (idx < nq + nk) {
        const bool is_q = idx < nq;
        const uint32_t hidx = is_q ? idx : idx - nq;
        const uint32_t nh = is_q ? n_heads : n_kv_heads;
        const uint32_t b = hidx / nh, h = hidx % nh;
        const float* src = qkv + (size_t)b * rowd + (is_q ? 0u : qdim) + (size_t)h * head_dim;
        float theta = (float)positions[b];
        for (uint32_t i = 0; i < lane && i < half; ++i) theta *= theta_scale;
        const uint32_t slot = slots ? slots[b] : b;
        const uint32_t pos = positions[b];
        for (uint32_t k = lane; k < half; k += 32) {
            float y = ((float)k - corr_low) / fmaxf(0.001f, corr_high - corr_low);
            float ramp = (1.0f - fminf(1.0f, fmaxf(0.0f, y))) * ext_factor;
            float angle = (freq_scale * theta) * (1.0f - ramp) + theta * ramp;
            float sn = sinf(angle) * mscale;
            float cs = cosf(angle) * mscale;
            float a = src[k];
            float bb = src[k + half];
            float r0 = a * cs - bb * sn;
            float r1 = a * sn + bb * cs;
            if (is_q) {
                float* dst = q_out + (size_t)b * qdim + (size_t)h * head_dim;
                dst[k] = r0;
                dst[k + half] = r1;
            } else {
                uint32_t blk = block_tables[(size_t)slot * blocks_per_slot + (pos >> 4)];
                uint32_t within = pos & 15u;
                KV* dst = k_cache + (size_t)blk * 16u * kvdim + (size_t)within * kvdim +
                          (size_t)h * head_dim;
                pd_kv_store(&dst[k], r0);
                pd_kv_store(&dst[k + half], r1);
            }
            for (uint32_t i = 0; i < 32 && k + i < half; ++i) theta *= theta_scale;
        }
    } else if (idx < nq + nk + nk) {
        const uint32_t hidx = idx - nq - nk;
        const uint32_t b = hidx / n_kv_heads, h = hidx % n_kv_heads;
        const uint32_t slot = slots ? slots[b] : b;
        const uint32_t pos = positions[b];
        const float* src = qkv + (size_t)b * rowd + qdim + kvdim + (size_t)h * head_dim;
        uint32_t blk = block_tables[(size_t)slot * blocks_per_slot + (pos >> 4)];
        uint32_t within = pos & 15u;
        KV* dst = v_cache + (size_t)blk * 16u * kvdim + (size_t)within * kvdim +
                  (size_t)h * head_dim;
        for (uint32_t i = lane; i < head_dim; i += 32) pd_kv_store(&dst[i], src[i]);
    }
}


// K-split-combine + rope + append FUSED (glue round 3): the wqkv GEMM's
// partial planes feed the rope directly - each lane sums the nz z-planes in
// FIXED ascending order and adds bias (exactly pd_q8_0_gemm_mma_ks_combine's
// math), then applies the VERBATIM yarn rope chain from
// pd_qkv_rope_append_batch_kernel above. Kills the combine launch, the rope
// launch, and the [b, qkv] f32 round trip. Bit-identical outputs to the
// combine_b -> qkv_rope_append sequence.
template<typename KV>
__global__ void pd_ks_qkv_rope_append_kernel(
    const float* __restrict__ part, const float* __restrict__ bias,
    float* __restrict__ q_out, KV* __restrict__ k_cache, KV* __restrict__ v_cache,
    const unsigned int* __restrict__ positions, const unsigned int* __restrict__ slots,
    uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim, uint32_t max_ctx,
    float theta_scale, float freq_scale, float corr_low, float corr_high,
    float ext_factor, float mscale, uint32_t batch, uint32_t nz) {
    const uint32_t qdim = n_heads * head_dim, kvdim = n_kv_heads * head_dim;
    const uint32_t rowd = qdim + 2u * kvdim;
    const uint32_t npl = rowd * batch;
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t idx = blockIdx.x * (blockDim.x >> 5) + warp;
    const uint32_t nq = batch * n_heads, nk = batch * n_kv_heads;
    const uint32_t half = head_dim / 2u;

    if (idx < nq + nk) {
        const bool is_q = idx < nq;
        const uint32_t hidx = is_q ? idx : idx - nq;
        const uint32_t nh = is_q ? n_heads : n_kv_heads;
        const uint32_t b = hidx / nh, h = hidx % nh;
        const uint32_t bcol = (is_q ? 0u : qdim) + h * head_dim;
        const uint32_t orow = b * rowd;
        float theta = (float)positions[b];
        for (uint32_t i = 0; i < lane && i < half; ++i) theta *= theta_scale;
        const uint32_t slot = slots ? slots[b] : b;
        const uint32_t pos = positions[b];
        for (uint32_t k = lane; k < half; k += 32) {
            float y = ((float)k - corr_low) / fmaxf(0.001f, corr_high - corr_low);
            float ramp = (1.0f - fminf(1.0f, fmaxf(0.0f, y))) * ext_factor;
            float angle = (freq_scale * theta) * (1.0f - ramp) + theta * ramp;
            float sn = sinf(angle) * mscale;
            float cs = cosf(angle) * mscale;
            float a = 0.0f, bb = 0.0f;
            for (uint32_t z = 0; z < nz; ++z) {
                a += part[(size_t)z * npl + orow + bcol + k];
                bb += part[(size_t)z * npl + orow + bcol + k + half];
            }
            a += bias[bcol + k];
            bb += bias[bcol + k + half];
            float r0 = a * cs - bb * sn;
            float r1 = a * sn + bb * cs;
            if (is_q) {
                float* dst = q_out + (size_t)b * qdim + (size_t)h * head_dim;
                dst[k] = r0;
                dst[k + half] = r1;
            } else {
                KV* dst = k_cache + (size_t)slot * max_ctx * kvdim + (size_t)pos * kvdim +
                          (size_t)h * head_dim;
                pd_kv_store(&dst[k], r0);
                pd_kv_store(&dst[k + half], r1);
            }
            for (uint32_t i = 0; i < 32 && k + i < half; ++i) theta *= theta_scale;
        }
    } else if (idx < nq + nk + nk) {
        const uint32_t hidx = idx - nq - nk;
        const uint32_t b = hidx / n_kv_heads, h = hidx % n_kv_heads;
        const uint32_t slot = slots ? slots[b] : b;
        const uint32_t pos = positions[b];
        const uint32_t bcol = qdim + kvdim + h * head_dim;
        const uint32_t orow = b * rowd;
        KV* dst = v_cache + (size_t)slot * max_ctx * kvdim + (size_t)pos * kvdim +
                  (size_t)h * head_dim;
        for (uint32_t i = lane; i < head_dim; i += 32) {
            float v = 0.0f;
            for (uint32_t z = 0; z < nz; ++z)
                v += part[(size_t)z * npl + orow + bcol + i];
            v += bias[bcol + i];
            pd_kv_store(&dst[i], v);
        }
    }
}

// Paged twin of pd_ks_qkv_rope_append_kernel (gpt-oss b<=64 fused decode
// GEMM-combine + rope + append). Same fixed-order z-plane sum, bias, and yarn
// rope; only the K/V cache base swaps the dense slot*max_ctx*kvdim stride for a
// block-table lookup into the [n_blocks, 16, kvdim] pool. max_ctx drops. Bit-
// exact vs dense under an identity table.
template<typename KV>
__global__ void pd_ks_qkv_rope_append_paged_kernel(
    const float* __restrict__ part, const float* __restrict__ bias,
    float* __restrict__ q_out, KV* __restrict__ k_cache, KV* __restrict__ v_cache,
    const unsigned int* __restrict__ positions, const unsigned int* __restrict__ slots,
    uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
    float theta_scale, float freq_scale, float corr_low, float corr_high,
    float ext_factor, float mscale, uint32_t batch, uint32_t nz,
    const uint32_t* __restrict__ block_tables, uint32_t blocks_per_slot) {
    const uint32_t qdim = n_heads * head_dim, kvdim = n_kv_heads * head_dim;
    const uint32_t rowd = qdim + 2u * kvdim;
    const uint32_t npl = rowd * batch;
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t idx = blockIdx.x * (blockDim.x >> 5) + warp;
    const uint32_t nq = batch * n_heads, nk = batch * n_kv_heads;
    const uint32_t half = head_dim / 2u;

    if (idx < nq + nk) {
        const bool is_q = idx < nq;
        const uint32_t hidx = is_q ? idx : idx - nq;
        const uint32_t nh = is_q ? n_heads : n_kv_heads;
        const uint32_t b = hidx / nh, h = hidx % nh;
        const uint32_t bcol = (is_q ? 0u : qdim) + h * head_dim;
        const uint32_t orow = b * rowd;
        float theta = (float)positions[b];
        for (uint32_t i = 0; i < lane && i < half; ++i) theta *= theta_scale;
        const uint32_t slot = slots ? slots[b] : b;
        const uint32_t pos = positions[b];
        for (uint32_t k = lane; k < half; k += 32) {
            float y = ((float)k - corr_low) / fmaxf(0.001f, corr_high - corr_low);
            float ramp = (1.0f - fminf(1.0f, fmaxf(0.0f, y))) * ext_factor;
            float angle = (freq_scale * theta) * (1.0f - ramp) + theta * ramp;
            float sn = sinf(angle) * mscale;
            float cs = cosf(angle) * mscale;
            float a = 0.0f, bb = 0.0f;
            for (uint32_t z = 0; z < nz; ++z) {
                a += part[(size_t)z * npl + orow + bcol + k];
                bb += part[(size_t)z * npl + orow + bcol + k + half];
            }
            a += bias[bcol + k];
            bb += bias[bcol + k + half];
            float r0 = a * cs - bb * sn;
            float r1 = a * sn + bb * cs;
            if (is_q) {
                float* dst = q_out + (size_t)b * qdim + (size_t)h * head_dim;
                dst[k] = r0;
                dst[k + half] = r1;
            } else {
                uint32_t blk = block_tables[(size_t)slot * blocks_per_slot + (pos >> 4)];
                uint32_t within = pos & 15u;
                KV* dst = k_cache + (size_t)blk * 16u * kvdim + (size_t)within * kvdim +
                          (size_t)h * head_dim;
                pd_kv_store(&dst[k], r0);
                pd_kv_store(&dst[k + half], r1);
            }
            for (uint32_t i = 0; i < 32 && k + i < half; ++i) theta *= theta_scale;
        }
    } else if (idx < nq + nk + nk) {
        const uint32_t hidx = idx - nq - nk;
        const uint32_t b = hidx / n_kv_heads, h = hidx % n_kv_heads;
        const uint32_t slot = slots ? slots[b] : b;
        const uint32_t pos = positions[b];
        const uint32_t bcol = qdim + kvdim + h * head_dim;
        const uint32_t orow = b * rowd;
        uint32_t blk = block_tables[(size_t)slot * blocks_per_slot + (pos >> 4)];
        uint32_t within = pos & 15u;
        KV* dst = v_cache + (size_t)blk * 16u * kvdim + (size_t)within * kvdim +
                  (size_t)h * head_dim;
        for (uint32_t i = lane; i < head_dim; i += 32) {
            float v = 0.0f;
            for (uint32_t z = 0; z < nz; ++z)
                v += part[(size_t)z * npl + orow + bcol + i];
            v += bias[bcol + i];
            pd_kv_store(&dst[i], v);
        }
    }
}

// Convert an f32 buffer to f16 (used for the single-stream KV write, which stores
// one post-rope K/V row into the fp16 cache). grid ceil(n/256).
__global__ void pd_convert_f32_f16_kernel(const float* __restrict__ src, __half* __restrict__ dst,
                                          uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}


// f32 -> bf16 twin of the convert above (slot 548): stages activations for
// the TGV bf16 decode GEMM (slot 547). grid ceil(n/256).
__global__ void pd_convert_f32_bf16_kernel(const float* __restrict__ src,
                                           __nv_bfloat16* __restrict__ dst,
                                           uint64_t n) {
    PD_PDL_ARM();
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2bfloat16(src[i]);
}

// bf16 -> f32, the twin of the cast above. The vendored low-M dense GEMM
// (slot 566) emits bf16 while this engine's activations are f32.
__global__ void pd_convert_bf16_f32_kernel(const __nv_bfloat16* __restrict__ src,
                                           float* __restrict__ dst, uint64_t n) {
    // 8 elements a thread: one 16 B load, two float4 stores. The scalar form
    // cost more than the GEMM it feeds saved (a [8 x 10240] plane is 82 K
    // elements a launch, ~48 launches a tick).
    const uint64_t i8 = ((uint64_t)blockIdx.x * blockDim.x + threadIdx.x) * 8ull;
    // the vector path needs both ends 16 B aligned: callers hand us `y` slices
    // that start at an arbitrary element offset, and a float4 store into one
    // of those raises CUDA_ERROR_MISALIGNED_ADDRESS mid-generation.
    const bool vec_ok = ((reinterpret_cast<uintptr_t>(src) | reinterpret_cast<uintptr_t>(dst)) & 15u) == 0u;
    if (vec_ok && i8 + 8ull <= n) {
        const uint4 raw = *reinterpret_cast<const uint4*>(src + i8);
        const __nv_bfloat16* v = reinterpret_cast<const __nv_bfloat16*>(&raw);
        float4 a, b;
        a.x = __bfloat162float(v[0]); a.y = __bfloat162float(v[1]);
        a.z = __bfloat162float(v[2]); a.w = __bfloat162float(v[3]);
        b.x = __bfloat162float(v[4]); b.y = __bfloat162float(v[5]);
        b.z = __bfloat162float(v[6]); b.w = __bfloat162float(v[7]);
        *reinterpret_cast<float4*>(dst + i8) = a;
        *reinterpret_cast<float4*>(dst + i8 + 4) = b;
        return;
    }
    for (uint64_t i = i8; i < n && i < i8 + 8ull; ++i) dst[i] = __bfloat162float(src[i]);
}

PD_EXPORT
int pd_convert_bf16_f32(const void* src, void* dst, uint64_t n, void* stream) {
    if (n == 0) return 0;
    const uint32_t thr = 256u;
    const uint64_t n8 = (n + 7ull) / 8ull;
    pd_convert_bf16_f32_kernel<<<(uint32_t)((n8 + thr - 1) / thr), thr, 0,
                                 (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)src, (float*)dst, n);
    return (int)cudaGetLastError();
}

// Strided twin: copy `cols` of each row out of a wider bf16 plane. The low-M
// dense GEMM needs its N padded to the MMA tile, and the consumers want the
// natural width back - this unpads in the cast rather than teaching every
// consumer a second row stride.
__global__ void pd_convert_bf16_f32_rows_kernel(const __nv_bfloat16* __restrict__ src,
                                                float* __restrict__ dst,
                                                uint32_t cols, uint32_t src_rs,
                                                uint32_t dst_rs) {
    const uint32_t r = blockIdx.y;
    const uint32_t c4 = (blockIdx.x * blockDim.x + threadIdx.x) * 4u;
    if (c4 >= cols) return;
    const __nv_bfloat16* sp = src + (size_t)r * src_rs + c4;
    float* dp = dst + (size_t)r * dst_rs + c4;
    const bool vec_ok = ((reinterpret_cast<uintptr_t>(sp) & 7u) == 0u)
                     && ((reinterpret_cast<uintptr_t>(dp) & 15u) == 0u);
    if (vec_ok && c4 + 4u <= cols) {
        const uint2 raw = *reinterpret_cast<const uint2*>(sp);
        const __nv_bfloat16* v = reinterpret_cast<const __nv_bfloat16*>(&raw);
        float4 a;
        a.x = __bfloat162float(v[0]); a.y = __bfloat162float(v[1]);
        a.z = __bfloat162float(v[2]); a.w = __bfloat162float(v[3]);
        *reinterpret_cast<float4*>(dp) = a;
        return;
    }
    for (uint32_t i = 0; i < 4u && c4 + i < cols; ++i)
        dp[i] = __bfloat162float(sp[i]);
}

PD_EXPORT
int pd_convert_bf16_f32_rows(const void* src, void* dst, uint32_t rows,
                             uint32_t cols, uint32_t src_rs, uint32_t dst_rs,
                             void* stream) {
    if (rows == 0 || cols == 0) return 0;
    dim3 grid((((cols + 3u) / 4u) + 255u) / 256u, rows);
    pd_convert_bf16_f32_rows_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)src, (float*)dst, cols, src_rs, dst_rs);
    return (int)cudaGetLastError();
}

PD_EXPORT
int pd_convert_f32_bf16(const void* src, void* dst, uint64_t n, void* stream) {
    if (n == 0) return 0;
    pd_pdl_go(pd_convert_f32_bf16_kernel, (unsigned)((n + 255u) / 256u), 256, 0,
              (cudaStream_t)stream, (const float*)src, (__nv_bfloat16*)dst, n);
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -2;
}

// device-side u32 += k (MTP chain-step rope-pos advance; graph-captured so
// draft chains replay back-to-back with no host copies between steps)
__global__ void pd_u32_addk_kernel(uint32_t* __restrict__ buf, uint32_t n,
                                   uint32_t k) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] += k;
}

PD_EXPORT
int pd_u32_addk(void* buf, uint32_t n, uint32_t k, void* stream) {
    if (n == 0) return 0;
    pd_u32_addk_kernel<<<(n + 255u) / 256u, 256, 0, (cudaStream_t)stream>>>(
        (uint32_t*)buf, n, k);
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -2;
}

// Async spec round: assemble the verify tick's token rows on
// device from the drafter chain's step-major output plane, so the host
// never reads drafts back before launching verify - the chain->verify
// boundary becomes a fully queued stream sequence. Per slot s (of n):
//   dst[base[s] + 0]           = pend[s]
//   dst[base[s] + 1 + j]       = drafts[j * rr + srcrow[s]]   j < ndr[s]
//   dst[base[s] + 1 + j..clen) = last real token (the verify pad rule:
//                                 chunk.get(i).unwrap_or(last))
// meta = [pend | srcrow | ndr | clen | base], 5*n u32 in one upload (one
// pageable H2D instead of five - each is an implicit-sync risk). Cold
// slots (ndr 0) pad with pend. srcrow indexes the chain's kept rows.
__global__ void pd_spec_toks_kernel(const uint32_t* __restrict__ meta,
                                    const uint32_t* __restrict__ drafts,
                                    uint32_t* __restrict__ dst, uint32_t n,
                                    uint32_t cmax, uint32_t rr) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t s = i / cmax, j = i % cmax;
    if (s >= n) return;
    const uint32_t pend = meta[s], srow = meta[n + s], nd = meta[2u * n + s];
    const uint32_t clen = meta[3u * n + s], base = meta[4u * n + s];
    if (j >= clen) return;
    uint32_t t;
    if (j == 0u) t = pend;
    else if (j <= nd) t = drafts[(j - 1u) * rr + srow];
    else t = nd > 0u ? drafts[(nd - 1u) * rr + srow] : pend;
    dst[base + j] = t;
}

PD_EXPORT
int pd_spec_toks(const void* meta, const void* drafts, void* dst, uint32_t n,
                 uint32_t cmax, uint32_t rr, void* stream) {
    if (n == 0 || cmax == 0) return 0;
    const uint32_t total = n * cmax;
    pd_spec_toks_kernel<<<(total + 255u) / 256u, 256, 0, (cudaStream_t)stream>>>(
        (const uint32_t*)meta, (const uint32_t*)drafts, (uint32_t*)dst, n,
        cmax, rr);
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -2;
}

// Rung B1: Device-side accept for the async spec round. Runs
// right after the verify tick on the same stream and emits one compact
// per-slot strip - the host then reads the strip instead of picks +
// drafts and replays nothing (the accept-while-match walk happens here,
// once). Per slot s of n, using the same meta the token assembly used:
//   bound  = 1 + ndr[s]                  (the real chunk length)
//   draft(j) = drafts[(j-1)*rr + srow]   (exactly what verify saw)
//   a: while a+1 < bound && draft(a+1) == sampled[base+a]: ++a
// strip[s*stride ..] = { accepted = a+1, p_final = pos[base] + a,
//                        final_row = base + a, new_pending =
//                        sampled[base+a], tok_0..tok_a (the emitted
//                        tokens = sampled[base..base+a]) }
// One thread per slot (n <= 64, walk <= 16): trivially latency-bound and
// ~2us - its value is the deleted host work, not its own speed.
__global__ void pd_spec_accept_kernel(const uint32_t* __restrict__ sampled,
                                      const uint32_t* __restrict__ drafts,
                                      const uint32_t* __restrict__ meta,
                                      const uint32_t* __restrict__ pos,
                                      uint32_t* __restrict__ strip, uint32_t n,
                                      uint32_t rr, uint32_t stride) {
    const uint32_t s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n) return;
    const uint32_t srow = meta[n + s], ndr = meta[2u * n + s];
    const uint32_t base = meta[4u * n + s];
    uint32_t a = 0;
    while (a + 1u < 1u + ndr
           && drafts[a * rr + srow] == sampled[base + a])
        ++a;
    uint32_t* o = strip + (size_t)s * stride;
    o[0] = a + 1u;
    o[1] = pos[base] + a;
    o[2] = base + a;
    o[3] = sampled[base + a];
    for (uint32_t j = 0; j <= a && 4u + j < stride; ++j)
        o[4u + j] = sampled[base + j];
}

PD_EXPORT
int pd_spec_accept(const void* sampled, const void* drafts, const void* meta,
                   const void* pos, void* strip, uint32_t n, uint32_t rr,
                   uint32_t stride, void* stream) {
    if (n == 0) return 0;
    if (stride < 5u) return cudaErrorInvalidValue;
    pd_spec_accept_kernel<<<(n + 63u) / 64u, 64, 0, (cudaStream_t)stream>>>(
        (const uint32_t*)sampled, (const uint32_t*)drafts,
        (const uint32_t*)meta, (const uint32_t*)pos, (uint32_t*)strip, n, rr,
        stride);
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -2;
}

// Rung B2: accept + NEXT-ROUND device prep in one kernel - the
// one-ahead spec pipeline's heart. Runs pd_spec_accept's walk, then (per
// slot, which the caller guarantees is a kept chain row: steady-state
// pipeline entry requires all-warm/all-kept, n == rr) writes everything
// round N+1's chain and verify need, so the next round launches with no
// host uploads:
//   m_tok[ci]   = new pending          (chain step-0 token)
//   m_pos[ci]   = p_final + (hold2 ? 0 : 1)   (chain rope start; mode-0
//                 graphs carry their own +1 tail, baked at capture)
//   m_attn[ci]  = p_final              (attention bound clamp, per round)
//   meta[s]     = new pending          (the assembly's pend lane)
//   pf_pos[base+j] = p_final + 1 + j   for j < clen (next verify's rows -
//                 each thread owns its slot's rows, no cross-thread hazard;
//                 pos[base] is read before any thread writes row 0 because
//                 the read happens in this thread's prologue)
// strip is emitted exactly as pd_spec_accept (the host trails on it).
__global__ void pd_spec_prep_kernel(const uint32_t* __restrict__ sampled,
                                    const uint32_t* __restrict__ drafts,
                                    uint32_t* __restrict__ meta,
                                    uint32_t* __restrict__ pos,
                                    uint32_t* __restrict__ strip,
                                    uint32_t* __restrict__ m_tok,
                                    uint32_t* __restrict__ m_pos,
                                    uint32_t* __restrict__ m_attn, uint32_t n,
                                    uint32_t rr, uint32_t stride,
                                    uint32_t hold2) {
    const uint32_t s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n) return;
    const uint32_t srow = meta[n + s], ndr = meta[2u * n + s];
    const uint32_t clen = meta[3u * n + s], base = meta[4u * n + s];
    const uint32_t p_start = pos[base];
    uint32_t a = 0;
    while (a + 1u < 1u + ndr
           && drafts[a * rr + srow] == sampled[base + a])
        ++a;
    const uint32_t pend = sampled[base + a];
    const uint32_t p_final = p_start + a;
    uint32_t* o = strip + (size_t)s * stride;
    o[0] = a + 1u;
    o[1] = p_final;
    o[2] = base + a;
    o[3] = pend;
    for (uint32_t j = 0; j <= a && 4u + j < stride; ++j)
        o[4u + j] = sampled[base + j];
    // next-round device state
    m_tok[srow] = pend;
    m_pos[srow] = p_final + (hold2 ? 0u : 1u);
    m_attn[srow] = p_final;
    meta[s] = pend;
    for (uint32_t j = 0; j < clen; ++j)
        pos[base + j] = p_final + 1u + j;
}

PD_EXPORT
int pd_spec_prep(const void* sampled, const void* drafts, void* meta,
                 void* pos, void* strip, void* m_tok, void* m_pos,
                 void* m_attn, uint32_t n, uint32_t rr, uint32_t stride,
                 uint32_t hold2, void* stream) {
    if (n == 0) return 0;
    if (stride < 5u) return cudaErrorInvalidValue;
    pd_spec_prep_kernel<<<(n + 63u) / 64u, 64, 0, (cudaStream_t)stream>>>(
        (const uint32_t*)sampled, (const uint32_t*)drafts, (uint32_t*)meta,
        (uint32_t*)pos, (uint32_t*)strip, (uint32_t*)m_tok, (uint32_t*)m_pos,
        (uint32_t*)m_attn, n, rr, stride, hold2);
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -2;
}

// Rung B2: gather the accepted-final verify rows' hiddens into the chain's
// h input - replaces the host copy_region loop for pipelined rounds. Reads
// each slot's final_row from the strip (already computed by pd_spec_prep on
// the same stream) and its chain row from meta's srcrow lane. Must run
// before the next verify overwrites pf_normed - the pipeline enqueues it
// right after pd_spec_prep.
__global__ void pd_spec_hgather_kernel(const float* __restrict__ normed,
                                       const uint32_t* __restrict__ strip,
                                       const uint32_t* __restrict__ meta,
                                       float* __restrict__ h, uint32_t n,
                                       uint32_t n_main, uint32_t stride) {
    const uint32_t s = blockIdx.x;
    if (s >= n) return;
    const uint32_t srow = meta[n + s];
    const uint32_t row = strip[(size_t)s * stride + 2u];
    const float* src = normed + (size_t)row * n_main;
    float* dst = h + (size_t)srow * n_main;
    for (uint32_t i = threadIdx.x; i < n_main; i += blockDim.x)
        dst[i] = src[i];
}

PD_EXPORT
int pd_spec_hgather(const void* normed, const void* strip, const void* meta,
                    void* h, uint32_t n, uint32_t n_main, uint32_t stride,
                    void* stream) {
    if (n == 0) return 0;
    pd_spec_hgather_kernel<<<n, 256, 0, (cudaStream_t)stream>>>(
        (const float*)normed, (const uint32_t*)strip, (const uint32_t*)meta,
        (float*)h, n, n_main, stride);
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -2;
}

// Host-upload consolidation: the drafter round issued
// 224 single-row 21KB cudaMemcpyDtoD calls (2 per row per mtp step for the
// xh stitch + 1 per slot for the chain-h gather) - 0.9us each on-GPU but
// ~5-8us of host issue time apiece, ~1.6ms of the ~46ms wide round. These
// two kernels replace the loops with one launch each; the data movement is
// bit-identical.

// xh stitch: xh[i] = [ emb[i] | h[i] ] for i in 0..r - both sources are
// contiguous [r][n_main] f32, dst rows are 2*n_main wide.
__global__ void pd_spec_xh_stitch_kernel(const float* __restrict__ emb,
                                         const float* __restrict__ h,
                                         float* __restrict__ xh,
                                         uint32_t r, uint32_t n_main) {
    const size_t total = (size_t)r * 2u * n_main;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (size_t)gridDim.x * blockDim.x) {
        const uint32_t row = (uint32_t)(i / (2u * n_main));
        const uint32_t e = (uint32_t)(i - (size_t)row * 2u * n_main);
        const float* src = e < n_main ? emb : h;
        xh[i] = src[(size_t)row * n_main + (e < n_main ? e : e - n_main)];
    }
}

PD_EXPORT
int pd_spec_xh_stitch(const void* emb, const void* h, void* xh, uint32_t r,
                      uint32_t n_main, void* stream) {
    if (r == 0) return 0;
    const uint32_t blocks = min(2048u, (r * 2u * n_main + 255u) / 256u);
    pd_spec_xh_stitch_kernel<<<blocks, 256, 0, (cudaStream_t)stream>>>(
        (const float*)emb, (const float*)h, (float*)xh, r, n_main);
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -2;
}

// host-indexed row gather: dst[i] = src[idx[i]] for i in 0..n (f32 rows,
// n_main wide). The index plane is a tiny device u32 array the caller
// uploads with its other per-round strips (the pipelined variant with
// device-computed rows stays pd_spec_hgather).
__global__ void pd_hrow_gather_kernel(const float* __restrict__ src,
                                      const uint32_t* __restrict__ idx,
                                      float* __restrict__ dst, uint32_t n,
                                      uint32_t n_main) {
    const uint32_t s = blockIdx.x;
    if (s >= n) return;
    const float* sp = src + (size_t)idx[s] * n_main;
    float* dp = dst + (size_t)s * n_main;
    for (uint32_t i = threadIdx.x; i < n_main; i += blockDim.x)
        dp[i] = sp[i];
}

PD_EXPORT
int pd_hrow_gather(const void* src, const void* idx, void* dst, uint32_t n,
                   uint32_t n_main, void* stream) {
    if (n == 0) return 0;
    pd_hrow_gather_kernel<<<n, 256, 0, (cudaStream_t)stream>>>(
        (const float*)src, (const uint32_t*)idx, (float*)dst, n, n_main);
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -2;
}

// ---- Laguna per-head softplus output gate -----------------
// x[r, h, d] *= softplus(gate[r, h]) - the gate comes from a separate
// [embd, n_heads] projection and broadcasts over head_dim. softplus in f32
// via the overflow-safe identity max(v,0) + log1p(exp(-|v|)) - identical to
// torch F.softplus (the HF reference computes the gate in f32). Accurate
// expf deliberately: one transcendental per WEIGHT-broadcast element, cost is
// noise, and this feeds the greedy-parity gate.
__global__ void pd_mul_softplus_head_kernel(float* __restrict__ x,
                                            const float* __restrict__ gate,
                                            uint32_t n_heads, uint32_t head_dim,
                                            uint32_t rows) {
    // cascade: x is the attention combine's output, gate the band GEMV's
    PD_PDL_ARM();
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = rows * n_heads * head_dim;
    if (i >= total) return;
    const uint32_t hd = n_heads * head_dim;
    const uint32_t r = i / hd;
    const uint32_t h = (i - r * hd) / head_dim;
    const float v = gate[r * n_heads + h];
    x[i] *= fmaxf(v, 0.0f) + log1pf(expf(-fabsf(v)));
}

PD_EXPORT
int pd_mul_softplus_head(void* x, const void* gate, uint32_t n_heads,
                         uint32_t head_dim, uint32_t rows, void* stream) {
    const uint32_t total = rows * n_heads * head_dim;
    if (total == 0) return 0;
    const uint32_t threads = 256;
    pd_pdl_go(pd_mul_softplus_head_kernel, (total + threads - 1) / threads,
              threads, 0u, (cudaStream_t)stream,
        (float*)x, (const float*)gate, n_heads, head_dim, rows);
    // pd_launch_status() lives in a later segment - same expression inline
    return (int)cudaGetLastError();
}
