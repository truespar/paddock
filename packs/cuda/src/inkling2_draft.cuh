// inkling2_draft.cuh - Inkling 2 Mini's DSpark draft (the release's
// mtp.safetensors): slots 994-998.
// Textually-included segment of the single pack translation unit.
// Not standalone-compilable: include order is defined by ../pack.cu.
// Needs inkling2.cuh's pd_ik2_block_sum and attn/inkling2_attn.cuh's K / V
// layout (PdIk2Kv, pd_ik2_kv_row) and split fold.
//
// The draft (a study of the reference implementation's and vLLM's
// `inkling2` DSpark): target features at four layer visits are mixed per
// draft layer (softmax weights over the taps, then RMSNorm) into context K
// / V a committed token; a block of the anchor token and seven masks runs
// three dense layers over that context (bidirectional inside the block,
// RoPE on q / k, the local layers windowed), then a rank-256 recurrent
// head corrects the LM head's logits step by step, each step fed the
// previous draft token. The reference rounds to bf16 at every op; the
// kernels here round where a value is handed on (stores, the recurrent
// state), and keep f32 inside an op.

__device__ __forceinline__ float pd_ik2_bf16r(float x) {
    return __bfloat162float(__float2bfloat16_rn(x));
}

// slot 994: out[r] = bf16(rmsnorm(sum_t alpha[t] * feat[r][t]) * weight).
// feat rows of `taps` planes of `width` (row stride fstride); one CTA a row.
__global__ void __launch_bounds__(256) pd_inkling2_mix_norm_kernel(
        const float* feat, uint32_t fstride, uint32_t taps, const float* __restrict__ alpha,
        const __nv_bfloat16* __restrict__ weight, float* __restrict__ out, uint32_t width,
        float eps) {
    constexpr uint32_t V = 9u;
    PD_PDL_ARM();
    __shared__ float red[8];
    const uint32_t r = blockIdx.x, tid = threadIdx.x;
    const float* fr = pd_pdl_after(feat) + (size_t)r * fstride;
    float m[V];
    float ss[1] = {0.0f};
    #pragma unroll
    for (uint32_t k = 0; k < V; ++k) {
        const uint32_t d = tid + k * 256u;
        m[k] = 0.0f;
        if (d < width) {
            for (uint32_t t = 0; t < taps; ++t) m[k] += fr[(size_t)t * width + d] * alpha[t];
            ss[0] = fmaf(m[k], m[k], ss[0]);
        }
    }
    pd_ik2_block_sum<1>(ss, 1, red);
    const float inv = rsqrtf(ss[0] / (float)width + eps);
    #pragma unroll
    for (uint32_t k = 0; k < V; ++k) {
        const uint32_t d = tid + k * 256u;
        if (d < width) out[(size_t)r * width + d] = pd_ik2_bf16r(m[k] * inv * __bfloat162float(weight[d]));
    }
}

PD_EXPORT
int pd_inkling2_mix_norm(const void* feat, uint32_t fstride, uint32_t taps, const void* alpha,
                         const void* weight, void* out, uint32_t rows, uint32_t width, float eps,
                         void* stream) {
    if (rows == 0) return 0;
    if (width > 256u * 9u || taps == 0) return cudaErrorInvalidValue;
    pd_pdl_go(pd_inkling2_mix_norm_kernel, rows, 256, 0u, (cudaStream_t)stream,
              (const float*)feat, fstride, taps, (const float*)alpha,
              (const __nv_bfloat16*)weight, (float*)out, width, eps);
    return pd_launch_status();
}

// slot 995: per (row, head) of x (row stride xs, heads of 128 from col0), in
// place: with `weight` an RMSNorm (eps) times the weight, rounded to bf16;
// then neox RoPE at the row's position (angle pos * inv_freq[i], i < 64,
// pairing i with i + 64), rounded to bf16. One CTA of 64 a (row, head).
__global__ void __launch_bounds__(64) pd_inkling2_qk_rope_kernel(
        float* x, uint32_t xs, uint32_t col0, const __nv_bfloat16* __restrict__ weight,
        const uint32_t* pos, const float* __restrict__ inv_freq, float eps) {
    PD_PDL_ARM();
    __shared__ float red[2];
    const uint32_t r = blockIdx.x, h = blockIdx.y, i = threadIdx.x;
    float* xr = x + (size_t)r * xs + col0 + (size_t)h * 128u;
    float x1 = xr[i], x2 = xr[i + 64u];
    if (weight) {
        float ss = x1 * x1 + x2 * x2;
        for (uint32_t sh = 16; sh > 0; sh >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, sh);
        if ((i & 31u) == 0) red[i >> 5] = ss;
        __syncthreads();
        const float inv = 1.0f / sqrtf((red[0] + red[1]) / 128.0f + eps);
        x1 = pd_ik2_bf16r(x1 * inv * __bfloat162float(weight[i]));
        x2 = pd_ik2_bf16r(x2 * inv * __bfloat162float(weight[i + 64u]));
    }
    const float ang = (float)pd_pdl_after(pos)[r] * inv_freq[i];
    const float c = cosf(ang), s = sinf(ang);
    xr[i] = pd_ik2_bf16r(x1 * c - x2 * s);
    xr[i + 64u] = pd_ik2_bf16r(x2 * c + x1 * s);
}

PD_EXPORT
int pd_inkling2_qk_rope(void* x, uint32_t xs, uint32_t col0, uint32_t rows, uint32_t heads,
                        const void* weight, const void* pos, const void* inv_freq, float eps,
                        void* stream) {
    if (rows == 0 || heads == 0) return 0;
    pd_pdl_go(pd_inkling2_qk_rope_kernel, dim3(rows, heads), 64, 0u, (cudaStream_t)stream,
              (float*)x, xs, col0, (const __nv_bfloat16*)weight, (const uint32_t*)pos,
              (const float*)inv_freq, eps);
    return pd_launch_status();
}

// slot 996: the draft block's attention. Block b is 8 query rows (rows
// 8b..8b+7 of q, row stride qs) of slot blk[b * 2] at positions
// blk[b * 2 + 1] + 0..7, out at row stride os; each row sees every key of its slot up to the
// block's last (non-causal inside the block), the last `window + 8` of them
// with a window. A CTA: one block x one K / V head x one key split - the G
// heads' 8 rows each (8G query vectors) against 32-key tiles of the store,
// scores and the PV walk on the CUDA cores. With splits > 1 the partials
// land in part_o / part_ml for slot 990's fold (sink -inf: no sink).
template <uint32_t G>
__global__ void __launch_bounds__(128) pd_inkling2_draft_attn_kernel(
        const float* q, uint32_t qs, const uint32_t* blk, PdIk2Kv kk, PdIk2Kv vv,
        float* __restrict__ out, uint32_t os, float* __restrict__ part_o,
        float* __restrict__ part_ml, uint32_t heads, uint32_t kv_heads, uint32_t window,
        float scale) {
    constexpr uint32_t HD = 128u, B = 8u, NQ = 8u * G, T = 32u;
    PD_PDL_ARM();
    __shared__ float sq[NQ][HD + 1];
    __shared__ float sk[T][HD + 1];
    __shared__ float sv[T][HD];
    __shared__ float sp[NQ][T];
    __shared__ float sm[NQ], sl[NQ], sfix[NQ];
    __shared__ uint32_t srow[T];
    const uint32_t b = blockIdx.x, kh = blockIdx.y, split = blockIdx.z, splits = gridDim.z;
    const uint32_t tid = threadIdx.x;
    const uint32_t* bd = pd_pdl_after(blk) + (size_t)b * 2u;
    const uint32_t slot = bd[0], p0 = bd[1];
    const uint32_t kv_dim = kv_heads * HD;
    const float* qp = pd_pdl_after(q);
    // query vector n = (row j = n % 8, head kh * G + n / 8)
    for (uint32_t u = tid; u < NQ * HD; u += 128u) {
        const uint32_t n = u / HD, d = u % HD;
        sq[n][d] = qp[(size_t)(b * B + n % B) * qs + (kh * G + n / B) * HD + d];
    }
    if (tid < NQ) {
        sm[tid] = -INFINITY;
        sl[tid] = 0.0f;
    }
    const uint32_t hi = p0 + B;
    const uint32_t lo = window ? (hi > window + B ? hi - window - B : 0u) : 0u;
    const uint32_t n = hi - lo, per = (n + splits - 1u) / splits;
    const uint32_t k0s = lo + min(n, split * per), k1s = lo + min(n, (split + 1u) * per);
    float acc[NQ];
    #pragma unroll
    for (uint32_t i = 0; i < NQ; ++i) acc[i] = 0.0f;
    __syncthreads();
    for (uint32_t t0 = k0s; t0 < k1s; t0 += T) {
        const uint32_t cnt = min(T, k1s - t0);
        if (tid < T) srow[tid] = tid < cnt ? (uint32_t)pd_ik2_kv_row(kk, slot, t0 + tid) : 0u;
        __syncthreads();
        for (uint32_t u = tid; u < T * HD; u += 128u) {
            const uint32_t kx = u / HD, d = u % HD;
            const size_t e = (size_t)srow[kx] * kv_dim + kh * HD + d;
            sk[kx][d] = kx < cnt ? pd_ik2_kv_ld1(kk, e) : 0.0f;
            sv[kx][d] = kx < cnt ? pd_ik2_kv_ld1(vv, e) : 0.0f;
        }
        __syncthreads();
        for (uint32_t u = tid; u < NQ * T; u += 128u) {
            const uint32_t qn = u / T, kx = u % T;
            float d = 0.0f;
            #pragma unroll 8
            for (uint32_t i = 0; i < HD; ++i) d = fmaf(sq[qn][i], sk[kx][i], d);
            sp[qn][kx] = kx < cnt ? d * scale : -INFINITY;
        }
        __syncthreads();
        if (tid < NQ) {
            float m = sm[tid];
            for (uint32_t kx = 0; kx < T; ++kx) m = fmaxf(m, sp[tid][kx]);
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
        // thread = one dim, every query vector
        #pragma unroll
        for (uint32_t i = 0; i < NQ; ++i) acc[i] *= sfix[i];
        for (uint32_t kx = 0; kx < cnt; ++kx) {
            const float vd = sv[kx][tid];
            #pragma unroll
            for (uint32_t i = 0; i < NQ; ++i) acc[i] = fmaf(sp[i][kx], vd, acc[i]);
        }
        __syncthreads();
    }
    #pragma unroll
    for (uint32_t i = 0; i < NQ; ++i) {
        const uint32_t row = b * B + i % B, h = kh * G + i / B;
        if (splits == 1u) {
            out[(size_t)row * os + h * HD + tid] = sl[i] > 0.0f ? acc[i] / sl[i] : 0.0f;
        } else {
            const size_t o = ((size_t)row * heads + h) * splits + split;
            part_o[o * HD + tid] = acc[i];
            if (tid == 0) {
                part_ml[o * 2u] = sm[i];
                part_ml[o * 2u + 1u] = sl[i];
            }
        }
    }
}

// the fold of slot 990 over a [rows][heads][128] plane at row stride `os`
// (the draft's q-shaped output) with no sink
__global__ void __launch_bounds__(128) pd_inkling2_draft_fold_kernel(
        const float* part_o, const float* part_ml, float* __restrict__ out, uint32_t os,
        uint32_t heads, uint32_t splits) {
    constexpr uint32_t HD = 128u;
    PD_PDL_ARM();
    const uint32_t r = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
    const float* ml = pd_pdl_after(part_ml) + ((size_t)r * heads + h) * splits * 2u;
    const float* po = pd_pdl_after(part_o) + ((size_t)r * heads + h) * splits * HD;
    float M = -INFINITY;
    for (uint32_t s = 0; s < splits; ++s) M = fmaxf(M, ml[2u * s]);
    float den = 0.0f, a = 0.0f;
    for (uint32_t s = 0; s < splits; ++s) {
        const float m = ml[2u * s];
        if (m == -INFINITY) continue;
        const float f = expf(m - M);
        den = fmaf(ml[2u * s + 1u], f, den);
        a = fmaf(po[(size_t)s * HD + tid], f, a);
    }
    out[(size_t)r * os + h * HD + tid] = den > 0.0f ? a / den : 0.0f;
}

PD_EXPORT
int pd_inkling2_draft_attn(const void* q, uint32_t qs, const void* blk, uint32_t n_blocks,
                           void* kbase, void* vbase, const void* bt, uint32_t bps, uint32_t ring,
                           uint32_t dtype, void* out, uint32_t os, void* part_o, void* part_ml,
                           uint32_t heads,
                           uint32_t kv_heads, uint32_t window, uint32_t splits, float scale,
                           void* stream) {
    if (n_blocks == 0) return 0;
    if (kv_heads == 0 || heads % kv_heads || dtype > 3u || splits == 0 ||
        (splits > 1u && (!part_o || !part_ml)) ||
        (bt == nullptr && (ring == 0 || (ring & (ring - 1u)))))
        return cudaErrorInvalidValue;
    const PdIk2Kv kk{kbase, (const uint32_t*)bt, bps, ring, dtype};
    const PdIk2Kv vv{vbase, (const uint32_t*)bt, bps, ring, dtype};
    const dim3 grid(n_blocks, kv_heads, splits);
    #define PD_IK2_DRAFT_ATTN(GG)                                                                \
        pd_pdl_go(pd_inkling2_draft_attn_kernel<GG>, grid, 128, 0u, (cudaStream_t)stream,       \
                  (const float*)q, qs, (const uint32_t*)blk, kk, vv, (float*)out, os,           \
                  (float*)part_o, (float*)part_ml, heads, kv_heads, window, scale)
    switch (heads / kv_heads) {
    case 1: PD_IK2_DRAFT_ATTN(1u); break;
    case 2: PD_IK2_DRAFT_ATTN(2u); break;
    case 3: PD_IK2_DRAFT_ATTN(3u); break;
    default: return cudaErrorInvalidValue;
    }
    #undef PD_IK2_DRAFT_ATTN
    if (splits > 1u)
        pd_pdl_go(pd_inkling2_draft_fold_kernel, dim3(n_blocks * 8u, heads), 128, 0u,
                  (cudaStream_t)stream, (const float*)part_o, (const float*)part_ml, (float*)out,
                  os, heads, splits);
    return pd_launch_status();
}

// slot 1010: slot 996 on the tensor cores - the same blocks, masks and
// outputs. A CTA (block, K / V head, key split): two warps of m16 rows, the
// G = 3 heads' 8 block rows each (vector 16w + r = head 2w + r / 8, row
// r % 8; warp 1's upper 8 rows are padding), against 64-key tiles of the
// bf16 store widened to f16 (exact) for m16n8k16 with f32 accumulators -
// slot 993's class: the queries and the probabilities round to f16 on the
// way in. CUDA cores walked each score as a 128-long FMA chain out of smem
// (GB10, a full draft layer at a 22K context: 2.7 ms a call).
__device__ __forceinline__ uint4 pd_ik2_bf8_h(uint4 b) {
    uint4 h4;
    const __nv_bfloat162* s = (const __nv_bfloat162*)&b;
    __half2* h = (__half2*)&h4;
    #pragma unroll
    for (uint32_t i = 0; i < 4u; ++i) h[i] = __float22half2_rn(__bfloat1622float2(s[i]));
    return h4;
}

__global__ void __launch_bounds__(64) pd_inkling2_draft_attn_tc_kernel(
        const float* q, uint32_t qs, const uint32_t* blk, PdIk2Kv kk, PdIk2Kv vv,
        float* __restrict__ out, uint32_t os, float* __restrict__ part_o,
        float* __restrict__ part_ml, uint32_t heads, uint32_t kv_heads, uint32_t window,
        float scale) {
    constexpr uint32_t HD = 128u, B = 8u, G = 3u, BN = 64u, LD = PD_IK2T_LD;
    PD_PDL_ARM();
    __shared__ __align__(16) __half ks[BN * LD];
    __shared__ __align__(16) __half vs[BN * LD];
    __shared__ uint32_t krow[BN];
    const uint32_t b = blockIdx.x, kh = blockIdx.y, split = blockIdx.z, splits = gridDim.z;
    const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
    const uint32_t g = lane >> 2, tq = lane & 3u;
    const uint32_t* bd = pd_pdl_after(blk) + (size_t)b * 2u;
    const uint32_t slot = bd[0], p0 = bd[1];
    const uint32_t kv_dim = kv_heads * HD;
    // this thread's two vectors: block row g of heads 2w (a) and 2w + 1 (c)
    const uint32_t ha = kh * G + 2u * warp, hc = ha + 1u;
    const bool vc = 2u * warp + 1u < G;
    const float* qp = pd_pdl_after(q);
    uint32_t qf[8][4];
    #pragma unroll
    for (uint32_t k16 = 0; k16 < 8u; ++k16) {
        #pragma unroll
        for (uint32_t i = 0; i < 4u; ++i) {
            const bool up = i & 1u;
            const uint32_t col = k16 * 16u + 2u * tq + (i >> 1) * 8u;
            float x0 = 0.0f, x1 = 0.0f;
            if (!up || vc) {
                const float* qr = qp + (size_t)(b * B + g) * qs + (up ? hc : ha) * HD + col;
                x0 = qr[0];
                x1 = qr[1];
            }
            qf[k16][i] = pd_ik2_h2(x0, x1);
        }
    }
    float ma = -INFINITY, mc = -INFINITY, la = 0.0f, lc = 0.0f;
    float o[16][4];
    #pragma unroll
    for (uint32_t d = 0; d < 16u; ++d) o[d][0] = o[d][1] = o[d][2] = o[d][3] = 0.0f;
    // the keys: the slot's up to the block's last, the last window + 8 with
    // a window; this split's BN-aligned share
    const uint32_t hi0 = p0 + B;
    const uint32_t lo0 = window ? (hi0 > window + B ? hi0 - window - B : 0u) : 0u;
    const uint32_t per = ((hi0 - lo0 + splits - 1u) / splits + BN - 1u) / BN * BN;
    const uint32_t lo = min(hi0, lo0 + split * per), hi = min(hi0, lo + per);
    for (uint32_t k0 = lo; k0 < hi; k0 += BN) {
        if (tid < BN) krow[tid] = k0 + tid < hi ? (uint32_t)pd_ik2_kv_row(kk, slot, k0 + tid) : 0u;
        __syncthreads();
        for (uint32_t u = tid; u < BN * 16u; u += 64u) {
            const uint32_t key = u >> 4, c8 = (u & 15u) * 8u;
            uint4 kx = make_uint4(0, 0, 0, 0), vx = kx;
            if (k0 + key < hi) {
                const size_t e = (size_t)krow[key] * kv_dim + kh * HD + c8;
                kx = pd_ik2_bf8_h(*(const uint4*)((const __nv_bfloat16*)kk.base + e));
                vx = pd_ik2_bf8_h(*(const uint4*)((const __nv_bfloat16*)vv.base + e));
            }
            *(uint4*)(ks + key * LD + c8) = kx;
            *(uint4*)(vs + key * LD + c8) = vx;
        }
        __syncthreads();
        float s[8][4];
        #pragma unroll
        for (uint32_t nt = 0; nt < 8u; ++nt) {
            s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.0f;
            #pragma unroll
            for (uint32_t kp = 0; kp < 4u; ++kp) {
                uint32_t bq[4];
                pd_ik2_ldm4(bq, ks + (nt * 8u + (lane & 7u)) * LD + kp * 32u + (lane >> 3) * 8u);
                pd_ik2_mma16(s[nt], qf[2u * kp], bq[0], bq[1]);
                pd_ik2_mma16(s[nt], qf[2u * kp + 1u], bq[2], bq[3]);
            }
        }
        float ta = -INFINITY, tc = -INFINITY;
        #pragma unroll
        for (uint32_t nt = 0; nt < 8u; ++nt) {
            #pragma unroll
            for (uint32_t i = 0; i < 4u; ++i) {
                const bool up = i >= 2u;
                const uint32_t key = k0 + nt * 8u + 2u * tq + (i & 1u);
                const float x = key < hi && (!up || vc) ? s[nt][i] * scale : -INFINITY;
                s[nt][i] = x;
                if (up) tc = fmaxf(tc, x);
                else ta = fmaxf(ta, x);
            }
        }
        ta = fmaxf(ta, __shfl_xor_sync(0xffffffffu, ta, 1));
        ta = fmaxf(ta, __shfl_xor_sync(0xffffffffu, ta, 2));
        tc = fmaxf(tc, __shfl_xor_sync(0xffffffffu, tc, 1));
        tc = fmaxf(tc, __shfl_xor_sync(0xffffffffu, tc, 2));
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
        #pragma unroll
        for (uint32_t j = 0; j < 4u; ++j) {
            const uint32_t a[4] = {pd_ik2_h2(s[2u * j][0], s[2u * j][1]),
                                   pd_ik2_h2(s[2u * j][2], s[2u * j][3]),
                                   pd_ik2_h2(s[2u * j + 1u][0], s[2u * j + 1u][1]),
                                   pd_ik2_h2(s[2u * j + 1u][2], s[2u * j + 1u][3])};
            #pragma unroll
            for (uint32_t dt = 0; dt < 16u; dt += 2u) {
                uint32_t bv[4];
                pd_ik2_ldm4t(bv, vs + (j * 16u + (lane & 7u) + ((lane >> 3) & 1u) * 8u) * LD +
                                     dt * 8u + (lane >> 4) * 8u);
                pd_ik2_mma16(o[dt], a, bv[0], bv[1]);
                pd_ik2_mma16(o[dt + 1u], a, bv[2], bv[3]);
            }
        }
        __syncthreads();
    }
    la += __shfl_xor_sync(0xffffffffu, la, 1);
    la += __shfl_xor_sync(0xffffffffu, la, 2);
    lc += __shfl_xor_sync(0xffffffffu, lc, 1);
    lc += __shfl_xor_sync(0xffffffffu, lc, 2);
    const uint32_t row = b * B + g;
    auto put = [&](uint32_t h, float m, float l, uint32_t c, float x0, float x1) {
        if (splits == 1u) {
            float* op = out + (size_t)row * os + h * HD + c;
            op[0] = l > 0.0f ? x0 / l : 0.0f;
            op[1] = l > 0.0f ? x1 / l : 0.0f;
            return;
        }
        const size_t at = ((size_t)row * heads + h) * splits + split;
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
        put(ha, ma, la, col, o[d][0], o[d][1]);
        if (vc) put(hc, mc, lc, col, o[d][2], o[d][3]);
    }
}

PD_EXPORT
int pd_inkling2_draft_attn_tc(const void* q, uint32_t qs, const void* blk, uint32_t n_blocks,
                              void* kbase, void* vbase, const void* bt, uint32_t bps,
                              uint32_t ring, uint32_t dtype, void* out, uint32_t os, void* part_o,
                              void* part_ml, uint32_t heads, uint32_t kv_heads, uint32_t window,
                              uint32_t splits, float scale, void* stream) {
    if (n_blocks == 0) return 0;
    // the bf16 draft stores, three query heads a K / V head (Inkling's)
    if (kv_heads == 0 || heads != 3u * kv_heads || dtype != 3u || splits == 0 ||
        (splits > 1u && (!part_o || !part_ml)) ||
        (bt == nullptr && (ring == 0 || (ring & (ring - 1u)))))
        return cudaErrorNotSupported;
    const PdIk2Kv kk{kbase, (const uint32_t*)bt, bps, ring, dtype};
    const PdIk2Kv vv{vbase, (const uint32_t*)bt, bps, ring, dtype};
    pd_pdl_go(pd_inkling2_draft_attn_tc_kernel, dim3(n_blocks, kv_heads, splits), 64, 0u,
              (cudaStream_t)stream, (const float*)q, qs, (const uint32_t*)blk, kk, vv, (float*)out,
              os, (float*)part_o, (float*)part_ml, heads, kv_heads, window, scale);
    if (splits > 1u)
        pd_pdl_go(pd_inkling2_draft_fold_kernel, dim3(n_blocks * 8u, heads), 128, 0u,
                  (cudaStream_t)stream, (const float*)part_o, (const float*)part_ml, (float*)out,
                  os, heads, splits);
    return pd_launch_status();
}

// slot 997: the recurrent head's gate, the reference's bf16 ops each
// rounded: the f32 projection raw + ctx ([rows][3r]: gate | candidate |
// out, the step's input term and its context term), state [rows][r] in
// place (row stride ss); latent = bf16(tanh(out)).
__global__ void __launch_bounds__(256) pd_inkling2_markov_gate_kernel(
        const float* raw, const float* ctx, float* state, uint32_t ss, float* __restrict__ latent,
        uint32_t rank) {
    PD_PDL_ARM();
    const uint32_t r = blockIdx.y, c = blockIdx.x * 256u + threadIdx.x;
    if (c >= rank) return;
    const float* rr = pd_pdl_after(raw) + (size_t)r * 3u * rank;
    const float* cr = pd_pdl_after(ctx) + (size_t)r * 3u * rank;
    float* st = state + (size_t)r * ss;
    float g = pd_ik2_bf16r(rr[c] + cr[c]);
    const float cand_raw = pd_ik2_bf16r(rr[rank + c] + cr[rank + c]);
    const float o = pd_ik2_bf16r(rr[2u * rank + c] + cr[2u * rank + c]);
    g = pd_ik2_bf16r(__fdiv_rn(1.0f, 1.0f + expf(-g)));
    const float cand = pd_ik2_bf16r(tanhf(cand_raw));
    const float keep = pd_ik2_bf16r(g * st[c]);
    const float upd = pd_ik2_bf16r(pd_ik2_bf16r(1.0f - g) * cand);
    st[c] = pd_ik2_bf16r(keep + upd);
    latent[(size_t)r * rank + c] = pd_ik2_bf16r(tanhf(o));
}

PD_EXPORT
int pd_inkling2_markov_gate(const void* raw, const void* ctx, void* state, uint32_t ss,
                            void* latent, uint32_t rows, uint32_t rank, void* stream) {
    if (rows == 0) return 0;
    pd_pdl_go(pd_inkling2_markov_gate_kernel, dim3((rank + 255u) / 256u, rows), 256, 0u,
              (cudaStream_t)stream, (const float*)raw, (const float*)ctx, (float*)state, ss,
              (float*)latent, rank);
    return pd_launch_status();
}

// slot 998: a draft step's pick. Per row: bias = bf16(gemm + fold_bias),
// x = bf16(bf16(base) + bias), logit = bf16(cap * tanh(x / cap)), and the
// FIRST index of the largest logit (NaN as -inf) into ids[row] (best[] must
// start zeroed: the (order key, ~index) pairs meet in one 64-bit atomicMax).
__device__ __forceinline__ uint32_t pd_ik2_order_key(float v) {
    if (v != v) return 0u;  // NaN as -inf
    const uint32_t b = __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

__global__ void __launch_bounds__(256) pd_inkling2_markov_argmax_kernel(
        const float* base, const float* gemm, const __nv_bfloat16* __restrict__ fold_bias,
        uint32_t vocab, uint32_t stride, float cap, unsigned long long* best) {
    PD_PDL_ARM();
    const uint32_t r = blockIdx.y;
    const float* br = pd_pdl_after(base) + (size_t)r * stride;
    const float* gr = pd_pdl_after(gemm) + (size_t)r * stride;
    unsigned long long top = 0ull;
    for (uint32_t i = blockIdx.x * 256u + threadIdx.x; i < vocab; i += gridDim.x * 256u) {
        const float bias = pd_ik2_bf16r(gr[i] + __bfloat162float(fold_bias[i]));
        const float x = pd_ik2_bf16r(pd_ik2_bf16r(br[i]) + bias);
        const float l = pd_ik2_bf16r(tanhf(x / cap) * cap);
        const unsigned long long k =
            ((unsigned long long)pd_ik2_order_key(l) << 32) | (unsigned long long)(0xFFFFFFFFu - i);
        top = k > top ? k : top;
    }
    for (uint32_t sh = 16; sh > 0; sh >>= 1) {
        const unsigned long long o = __shfl_xor_sync(0xffffffffu, top, sh);
        top = o > top ? o : top;
    }
    if ((threadIdx.x & 31u) == 0) atomicMax(best + r, top);
}

__global__ void pd_inkling2_markov_pick_kernel(const unsigned long long* best, uint32_t* ids,
                                               uint32_t rows) {
    PD_PDL_ARM();
    const uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows) ids[r] = 0xFFFFFFFFu - (uint32_t)(pd_pdl_after(best)[r] & 0xFFFFFFFFull);
}

PD_EXPORT
int pd_inkling2_markov_argmax(const void* base, const void* gemm, const void* fold_bias,
                              void* best, void* ids, uint32_t rows, uint32_t vocab,
                              uint32_t stride, float cap, void* stream) {
    if (rows == 0) return 0;
    cudaError_t e = cudaMemsetAsync(best, 0, (size_t)rows * 8u, (cudaStream_t)stream);
    if (e != cudaSuccess) return e;
    pd_pdl_go(pd_inkling2_markov_argmax_kernel, dim3(64u, rows), 256, 0u, (cudaStream_t)stream,
              (const float*)base, (const float*)gemm, (const __nv_bfloat16*)fold_bias, vocab,
              stride, cap, (unsigned long long*)best);
    pd_pdl_go(pd_inkling2_markov_pick_kernel, (rows + 63u) / 64u, 64, 0u, (cudaStream_t)stream,
              (const unsigned long long*)best, (uint32_t*)ids, rows);
    return pd_launch_status();
}

// slot 999: dst[dst_off + r * ds + c] = src[src_off + r * ss + c] for c <
// width, r < rows (ss 0 broadcasts one row); with `bf16` the destination is
// bf16 (rounded), else f32. The draft's concatenated GEMM inputs and the
// recurrent head's folded planes.
__global__ void __launch_bounds__(256) pd_inkling2_copy2d_kernel(
        const float* src, size_t src_off, uint32_t ss, void* dst, size_t dst_off, uint32_t ds,
        uint32_t width, uint32_t bf16) {
    PD_PDL_ARM();
    const uint32_t r = blockIdx.y;
    const float* s = pd_pdl_after(src) + src_off + (size_t)r * ss;
    for (uint32_t c = blockIdx.x * 256u + threadIdx.x; c < width; c += gridDim.x * 256u) {
        const size_t o = dst_off + (size_t)r * ds + c;
        if (bf16) ((__nv_bfloat16*)dst)[o] = __float2bfloat16_rn(s[c]);
        else ((float*)dst)[o] = s[c];
    }
}

PD_EXPORT
int pd_inkling2_copy2d(const void* src, uint64_t src_off, uint32_t ss, void* dst, uint64_t dst_off,
                       uint32_t ds, uint32_t width, uint32_t rows, uint32_t bf16, void* stream) {
    if (rows == 0 || width == 0) return 0;
    pd_pdl_go(pd_inkling2_copy2d_kernel, dim3(min((width + 255u) / 256u, 16u), rows), 256, 0u,
              (cudaStream_t)stream, (const float*)src, (size_t)src_off, ss, dst, (size_t)dst_off,
              ds, width, bf16);
    return pd_launch_status();
}

// ---- speculative rounds: the short convs' rollback --------------------------
// A short conv carries its last 3 inputs, and a verify run advances them by
// the whole run; when only a prefix is accepted the carried rows must be
// those after that prefix. Slot 1000, launched before a verify pass's conv,
// keeps each run's [3 carried rows ++ its (<= 8) input rows] a slot; slot
// 1001 sets a slot's carried rows from it once `keep` rows were accepted
// (the last 3 of the first 3 + keep).
#define PD_IK2_STASH_ROWS 11u

__global__ void __launch_bounds__(256) pd_inkling2_conv_stash_kernel(
        const float* x, uint32_t xs, uint32_t C, const float* state, const uint32_t* run_off,
        const uint32_t* run_slot, float* __restrict__ stash) {
    PD_PDL_ARM();
    const uint32_t c = blockIdx.x * 256u + threadIdx.x, run = blockIdx.y;
    if (c >= C) return;
    const uint32_t* ro = pd_pdl_after(run_off);
    const uint32_t r0 = ro[run], len = min(ro[run + 1u] - r0, PD_IK2_STASH_ROWS - 3u);
    const uint32_t slot = pd_pdl_after(run_slot)[run];
    const float* st = pd_pdl_after(state) + (size_t)slot * 3u * C;
    const float* xr = pd_pdl_after(x);
    float* out = stash + (size_t)slot * PD_IK2_STASH_ROWS * C;
    #pragma unroll
    for (uint32_t j = 0; j < 3u; ++j) out[(size_t)j * C + c] = st[(size_t)j * C + c];
    for (uint32_t j = 0; j < len; ++j) out[(size_t)(3u + j) * C + c] = xr[(size_t)(r0 + j) * xs + c];
}

PD_EXPORT
int pd_inkling2_conv_stash(const void* x, uint32_t xs, uint32_t C, const void* state,
                           const void* run_off, const void* run_slot, uint32_t n_runs, void* stash,
                           void* stream) {
    if (n_runs == 0) return 0;
    pd_pdl_go(pd_inkling2_conv_stash_kernel, dim3((C + 255u) / 256u, n_runs), 256, 0u,
              (cudaStream_t)stream, (const float*)x, xs, C, (const float*)state,
              (const uint32_t*)run_off, (const uint32_t*)run_slot, (float*)stash);
    return pd_launch_status();
}

// items [n][2] = (slot, keep), keep >= 1 the run's accepted rows
__global__ void __launch_bounds__(256) pd_inkling2_conv_restore_kernel(
        const float* stash, float* state, uint32_t C, const uint32_t* items) {
    PD_PDL_ARM();
    const uint32_t c = blockIdx.x * 256u + threadIdx.x, it = blockIdx.y;
    if (c >= C) return;
    const uint32_t* t = pd_pdl_after(items) + (size_t)it * 2u;
    const uint32_t slot = t[0], keep = t[1];
    const float* s = pd_pdl_after(stash) + (size_t)slot * PD_IK2_STASH_ROWS * C;
    float* st = state + (size_t)slot * 3u * C;
    #pragma unroll
    for (uint32_t j = 0; j < 3u; ++j) st[(size_t)j * C + c] = s[(size_t)(keep + j) * C + c];
}

PD_EXPORT
int pd_inkling2_conv_restore(const void* stash, void* state, uint32_t C, const void* items,
                             uint32_t n_items, void* stream) {
    if (n_items == 0) return 0;
    pd_pdl_go(pd_inkling2_conv_restore_kernel, dim3((C + 255u) / 256u, n_items), 256, 0u,
              (cudaStream_t)stream, (const float*)stash, (float*)state, C, (const uint32_t*)items);
    return pd_launch_status();
}
