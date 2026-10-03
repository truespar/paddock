// clef.cuh - Clef's joint schema head (Cloudflare/clef-flash): the passes the
// diarization GEMM (722, F32 class on the stored BF16 weights) does not cover.
// Textually-included segment of the single pack translation unit.
// Not standalone-compilable: include order is defined by ../pack.cu.
//
// The head reads the backbone's final hidden rows and scores every option of
// every question of every request in a pass: LayerNorm at any width (724),
// means of hidden rows over token spans (725) and of lm_head rows over a
// span's token ids (726), multi-head attention whose every query has its own
// key range (727 - an option's or a field's request's memory, or a field's
// request's fields), the per-question option routing (728), row gathers
// (729), the scorer's feature rows (730) and the final logit (731).
//
// All F32 on the CUDA cores, in the Transformers reference's operation
// order; every reduction walks a fixed order that depends only on the row's
// own inputs, so a request's logits are the same bits alone or packed. The
// work is small beside the backbone (a few hundred queries, widths of 1024 /
// 4096); precision, not throughput, sets the design.

// ------------------------------------------------------------------ 724 norm
// torch.nn.LayerNorm over the last `d` channels: mean, then the biased
// variance of the centered values, eps inside the root. One CTA a row; the
// block reduction folds warps in warp order.
__device__ __forceinline__ float pd_clef_block_sum(float v, float* red) {
#pragma unroll
    for (uint32_t o = 16; o > 0; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, o));
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float s = 0.f;
    for (uint32_t w = 0; w < nw; ++w) s = __fadd_rn(s, red[w]);
    return s;
}

__global__ void __launch_bounds__(256) pd_clef_norm_kernel(const float* __restrict__ x,
                                                           const float* __restrict__ w,
                                                           const float* __restrict__ b,
                                                           float* __restrict__ y, uint32_t d,
                                                           float eps) {
    __shared__ float red[8];
    PD_PDL_ARM();
    const float* px = x + (size_t)blockIdx.x * d;
    float* py = y + (size_t)blockIdx.x * d;
    float s = 0.f;
    for (uint32_t c = threadIdx.x; c < d; c += blockDim.x) s = __fadd_rn(s, px[c]);
    const float mean = __fdiv_rn(pd_clef_block_sum(s, red), (float)d);
    float q = 0.f;
    for (uint32_t c = threadIdx.x; c < d; c += blockDim.x) {
        const float t = __fsub_rn(px[c], mean);
        q = fmaf(t, t, q);
    }
    const float rstd = __frsqrt_rn(__fadd_rn(__fdiv_rn(pd_clef_block_sum(q, red), (float)d), eps));
    for (uint32_t c = threadIdx.x; c < d; c += blockDim.x)
        py[c] = fmaf(__fmul_rn(__fsub_rn(px[c], mean), rstd), w[c], b[c]);
}

PD_EXPORT
int pd_clef_norm(const void* x, const void* w, const void* b, void* y, uint32_t d, uint32_t rows,
                 float eps, void* stream) {
    if (rows == 0) return 0;
    if (d == 0) return (int)cudaErrorInvalidValue;
    pd_pdl_go(pd_clef_norm_kernel, dim3(rows), dim3(256), 0, (cudaStream_t)stream, (const float*)x,
              (const float*)w, (const float*)b, (float*)y, d, eps);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 725 span mean
// out[s][c] = mean over rows [start, end) of x[r][c] (row stride `ld`):
// one CTA a span, each thread its columns, the rows summed in order.
__global__ void pd_clef_span_mean_kernel(const float* __restrict__ x, uint32_t ld, uint32_t d,
                                         const uint32_t* __restrict__ spans, float* __restrict__ out) {
    PD_PDL_ARM();
    const uint32_t s0 = spans[2u * blockIdx.x], s1 = spans[2u * blockIdx.x + 1u];
    float* po = out + (size_t)blockIdx.x * d;
    const float n = (float)(s1 - s0);
    for (uint32_t c = threadIdx.x; c < d; c += blockDim.x) {
        float s = 0.f;
        for (uint32_t r = s0; r < s1; ++r) s = __fadd_rn(s, x[(size_t)r * ld + c]);
        po[c] = __fdiv_rn(s, n);
    }
}

PD_EXPORT
int pd_clef_span_mean(const void* x, uint32_t ld, uint32_t d, const void* spans, uint32_t n,
                      void* out, void* stream) {
    if (n == 0) return 0;
    if (d == 0 || ld < d) return (int)cudaErrorInvalidValue;
    pd_pdl_go(pd_clef_span_mean_kernel, dim3(n), dim3(256), 0, (cudaStream_t)stream, (const float*)x,
              ld, d, (const uint32_t*)spans, (float*)out);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 726 lexical mean
// out[s][c] = mean over i in [start, end) of table[ids[i]][c], the table BF16
// rows `d` wide (the output embedding): widened exactly, summed in order.
__global__ void pd_clef_lex_mean_kernel(const __nv_bfloat16* __restrict__ table, uint32_t d,
                                        const uint32_t* __restrict__ ids,
                                        const uint32_t* __restrict__ spans, float* __restrict__ out) {
    PD_PDL_ARM();
    const uint32_t s0 = spans[2u * blockIdx.x], s1 = spans[2u * blockIdx.x + 1u];
    float* po = out + (size_t)blockIdx.x * d;
    const float n = (float)(s1 - s0);
    for (uint32_t c = threadIdx.x; c < d; c += blockDim.x) {
        float s = 0.f;
        for (uint32_t i = s0; i < s1; ++i)
            s = __fadd_rn(s, __bfloat162float(table[(size_t)ids[i] * d + c]));
        po[c] = __fdiv_rn(s, n);
    }
}

PD_EXPORT
int pd_clef_lex_mean(const void* table, uint32_t d, const void* ids, const void* spans, uint32_t n,
                     void* out, void* stream) {
    if (n == 0) return 0;
    if (d == 0) return (int)cudaErrorInvalidValue;
    pd_pdl_go(pd_clef_lex_mean_kernel, dim3(n), dim3(256), 0, (cudaStream_t)stream,
              (const __nv_bfloat16*)table, d, (const uint32_t*)ids, (const uint32_t*)spans,
              (float*)out);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 727 attention
// softmax(q k^T * scale) v per head of 64, every query row over its own key
// rows [ranges[2i], ranges[2i] + ranges[2i + 1]). One warp a (query, head):
// lane l takes keys l, l + 32, ... with its own online softmax and its own
// 64-wide accumulator, then the lanes fold in a fixed butterfly. Keys are
// whole rows; q, k, v and out carry their own row strides.
__global__ void __launch_bounds__(128) pd_clef_attn_kernel(
        const float* __restrict__ q, uint32_t ldq, const float* __restrict__ k, uint32_t ldk,
        const float* __restrict__ v, uint32_t ldv, float* __restrict__ out, uint32_t ldo,
        const uint32_t* __restrict__ ranges, uint32_t nq, uint32_t heads, float scale) {
    constexpr uint32_t HD = 64u;
    PD_PDL_ARM();
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t item = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (item >= nq * heads) return;
    const uint32_t qi = item / heads, h = item % heads;
    const uint32_t k0 = ranges[2u * qi], kn = ranges[2u * qi + 1u];
    float qv[HD];
    const float* pq = q + (size_t)qi * ldq + h * HD;
#pragma unroll
    for (uint32_t c = 0; c < HD; c += 4u) {
        const float4 t = *reinterpret_cast<const float4*>(pq + c);
        qv[c] = t.x;
        qv[c + 1u] = t.y;
        qv[c + 2u] = t.z;
        qv[c + 3u] = t.w;
    }
    float m = -INFINITY, l = 0.f, acc[HD];
#pragma unroll
    for (uint32_t c = 0; c < HD; ++c) acc[c] = 0.f;
    for (uint32_t j = lane; j < kn; j += 32u) {
        const float* pk = k + (size_t)(k0 + j) * ldk + h * HD;
        float s = 0.f;
#pragma unroll
        for (uint32_t c = 0; c < HD; c += 4u) {
            const float4 t = *reinterpret_cast<const float4*>(pk + c);
            s = fmaf(qv[c], t.x, s);
            s = fmaf(qv[c + 1u], t.y, s);
            s = fmaf(qv[c + 2u], t.z, s);
            s = fmaf(qv[c + 3u], t.w, s);
        }
        s = __fmul_rn(s, scale);
        const float mn = fmaxf(m, s);
        const float corr = expf(m - mn), p = expf(s - mn);
        l = fmaf(l, corr, p);
        const float* pv = v + (size_t)(k0 + j) * ldv + h * HD;
#pragma unroll
        for (uint32_t c = 0; c < HD; c += 4u) {
            const float4 t = *reinterpret_cast<const float4*>(pv + c);
            acc[c] = fmaf(acc[c], corr, __fmul_rn(p, t.x));
            acc[c + 1u] = fmaf(acc[c + 1u], corr, __fmul_rn(p, t.y));
            acc[c + 2u] = fmaf(acc[c + 2u], corr, __fmul_rn(p, t.z));
            acc[c + 3u] = fmaf(acc[c + 3u], corr, __fmul_rn(p, t.w));
        }
        m = mn;
    }
    // fold the lanes: the shared max, every lane's sums rescaled to it, then
    // a fixed butterfly (lanes with no key carry l = 0 and a zero accumulator)
    float mall = m;
#pragma unroll
    for (uint32_t o = 16; o > 0; o >>= 1) mall = fmaxf(mall, __shfl_xor_sync(0xffffffffu, mall, o));
    const float f = m == -INFINITY ? 0.f : expf(m - mall);
    l = __fmul_rn(l, f);
#pragma unroll
    for (uint32_t o = 16; o > 0; o >>= 1) l = __fadd_rn(l, __shfl_xor_sync(0xffffffffu, l, o));
    float* po = out + (size_t)qi * ldo + h * HD;
#pragma unroll
    for (uint32_t c = 0; c < HD; ++c) {
        float a = __fmul_rn(acc[c], f);
#pragma unroll
        for (uint32_t o = 16; o > 0; o >>= 1) a = __fadd_rn(a, __shfl_xor_sync(0xffffffffu, a, o));
        if (lane == (c & 31u)) po[c] = __fdiv_rn(a, l);
    }
}

PD_EXPORT
int pd_clef_attention(const void* q, uint32_t ldq, const void* k, uint32_t ldk, const void* v,
                      uint32_t ldv, void* out, uint32_t ldo, const void* ranges, uint32_t nq,
                      uint32_t heads, float scale, void* stream) {
    if (nq == 0) return 0;
    if (heads == 0 || ldq < heads * 64u || ldk < heads * 64u || ldv < heads * 64u ||
        ldo < heads * 64u || ((ldq | ldk | ldv) & 3u) ||
        (((uintptr_t)q | (uintptr_t)k | (uintptr_t)v) & 15u))
        return (int)cudaErrorInvalidValue;
    const uint32_t items = nq * heads;
    pd_pdl_go(pd_clef_attn_kernel, dim3((items + 3u) / 4u), dim3(128), 0, (cudaStream_t)stream,
              (const float*)q, ldq, (const float*)k, ldk, (const float*)v, ldv, (float*)out, ldo,
              (const uint32_t*)ranges, nq, heads, scale);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 728 routing
// Per question: weights = softmax over its options of (option . field) /
// sqrt(d), summary = sum of weights x options. `qopts` holds each
// question's (first option, count). One CTA a question.
__global__ void __launch_bounds__(256) pd_clef_route_kernel(const float* __restrict__ opts,
                                                            const float* __restrict__ fields,
                                                            const uint32_t* __restrict__ qopts,
                                                            uint32_t d, float* __restrict__ out) {
    extern __shared__ float clef_route_sm[];   // [count] weights, then [8] reduction
    PD_PDL_ARM();
    const uint32_t o0 = qopts[2u * blockIdx.x], n = qopts[2u * blockIdx.x + 1u];
    float* wts = clef_route_sm;
    float* red = clef_route_sm + n;
    const float* f = fields + (size_t)blockIdx.x * d;
    const float inv = __frsqrt_rn((float)d);
    for (uint32_t i = 0; i < n; ++i) {
        const float* o = opts + (size_t)(o0 + i) * d;
        float s = 0.f;
        for (uint32_t c = threadIdx.x; c < d; c += blockDim.x) s = fmaf(o[c], f[c], s);
        s = pd_clef_block_sum(s, red);
        if (threadIdx.x == 0) wts[i] = __fmul_rn(s, inv);
    }
    __syncthreads();
    float m = -INFINITY;
    for (uint32_t i = 0; i < n; ++i) m = fmaxf(m, wts[i]);
    float z = 0.f;
    for (uint32_t i = 0; i < n; ++i) z = __fadd_rn(z, expf(wts[i] - m));
    float* po = out + (size_t)blockIdx.x * d;
    for (uint32_t c = threadIdx.x; c < d; c += blockDim.x) {
        float s = 0.f;
        for (uint32_t i = 0; i < n; ++i)
            s = fmaf(__fdiv_rn(expf(wts[i] - m), z), opts[(size_t)(o0 + i) * d + c], s);
        po[c] = s;
    }
}

PD_EXPORT
int pd_clef_route(const void* opts, const void* fields, const void* qopts, uint32_t nq, uint32_t d,
                  uint32_t max_opts, void* out, void* stream) {
    if (nq == 0) return 0;
    if (d == 0 || max_opts == 0 || max_opts > 8192u) return (int)cudaErrorInvalidValue;
    pd_pdl_go(pd_clef_route_kernel, dim3(nq), dim3(256), (max_opts + 8u) * 4u, (cudaStream_t)stream,
              (const float*)opts, (const float*)fields, (const uint32_t*)qopts, d, (float*)out);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 729 gather add
// y[r][c] += src[idx[r]][c] - a question's projection onto its options, a
// request's global projection onto its fields, a type embedding.
__global__ void pd_clef_gather_add_kernel(float* __restrict__ y, const float* __restrict__ src,
                                          const uint32_t* __restrict__ idx, uint32_t d) {
    PD_PDL_ARM();
    const float* ps = src + (size_t)idx[blockIdx.x] * d;
    float* py = y + (size_t)blockIdx.x * d;
    for (uint32_t c = threadIdx.x; c < d; c += blockDim.x) py[c] = __fadd_rn(py[c], ps[c]);
}

PD_EXPORT
int pd_clef_gather_add(void* y, const void* src, const void* idx, uint32_t rows, uint32_t d,
                       void* stream) {
    if (rows == 0) return 0;
    pd_pdl_go(pd_clef_gather_add_kernel, dim3(rows), dim3(256), 0, (cudaStream_t)stream, (float*)y,
              (const float*)src, (const uint32_t*)idx, d);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 730 features
// out[o] = [f, x, f * x, |f - x|] with f = fields[qof[o]], x = opts[o].
__global__ void pd_clef_features_kernel(const float* __restrict__ fields,
                                        const float* __restrict__ opts,
                                        const uint32_t* __restrict__ qof, uint32_t d,
                                        float* __restrict__ out) {
    PD_PDL_ARM();
    const float* f = fields + (size_t)qof[blockIdx.x] * d;
    const float* x = opts + (size_t)blockIdx.x * d;
    float* po = out + (size_t)blockIdx.x * 4u * d;
    for (uint32_t c = threadIdx.x; c < d; c += blockDim.x) {
        const float a = f[c], b = x[c];
        po[c] = a;
        po[d + c] = b;
        po[2u * d + c] = __fmul_rn(a, b);
        po[3u * d + c] = fabsf(__fsub_rn(a, b));
    }
}

PD_EXPORT
int pd_clef_features(const void* fields, const void* opts, const void* qof, uint32_t nopt,
                     uint32_t d, void* out, void* stream) {
    if (nopt == 0) return 0;
    pd_pdl_go(pd_clef_features_kernel, dim3(nopt), dim3(256), 0, (cudaStream_t)stream,
              (const float*)fields, (const float*)opts, (const uint32_t*)qof, d, (float*)out);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 731 score
// One option's logit: prior + sigmoid(gate) * (joint * cos + residual), with
//   prior    = ps * normalize(lex[o]) . normalize(qvec[q] + glob[r])  (D wide)
//   cos      = normalize(field[q]) . normalize(opt[o])  (eps 1e-8, W wide)
//   residual = hid[o] . w3 + b3  (W wide; hid is the scorer's GELU layer)
// q = qof[o], r = rof[q]; ps, joint, gate are the head's three scalars
// already through their clamp/exp/sigmoid. F.normalize divides by
// max(||x||, 1e-12), cosine_similarity each side by max(||x||, 1e-8).
__global__ void __launch_bounds__(256) pd_clef_score_kernel(
        const float* __restrict__ lex, const float* __restrict__ qvec, const float* __restrict__ glob,
        const uint32_t* __restrict__ qof, const uint32_t* __restrict__ rof,
        const float* __restrict__ fields, const float* __restrict__ opts,
        const float* __restrict__ hid, const float* __restrict__ w3, float b3, uint32_t dd,
        uint32_t w, float ps, float joint, float gate, float* __restrict__ logits) {
    __shared__ float red[8];
    PD_PDL_ARM();
    const uint32_t o = blockIdx.x, q = qof[o], r = rof[q];
    const float* lx = lex + (size_t)o * dd;
    const float* qv = qvec + (size_t)q * dd;
    const float* gv = glob + (size_t)r * dd;
    float sla = 0.f, sll = 0.f, saa = 0.f;
    for (uint32_t c = threadIdx.x; c < dd; c += blockDim.x) {
        const float a = __fadd_rn(qv[c], gv[c]), x = lx[c];
        sla = fmaf(x, a, sla);
        sll = fmaf(x, x, sll);
        saa = fmaf(a, a, saa);
    }
    sla = pd_clef_block_sum(sla, red);
    sll = pd_clef_block_sum(sll, red);
    saa = pd_clef_block_sum(saa, red);
    const float* f = fields + (size_t)q * w;
    const float* x = opts + (size_t)o * w;
    const float* hv = hid + (size_t)o * w;
    float sfx = 0.f, sff = 0.f, sxx = 0.f, sr = 0.f;
    for (uint32_t c = threadIdx.x; c < w; c += blockDim.x) {
        sfx = fmaf(f[c], x[c], sfx);
        sff = fmaf(f[c], f[c], sff);
        sxx = fmaf(x[c], x[c], sxx);
        sr = fmaf(hv[c], w3[c], sr);
    }
    sfx = pd_clef_block_sum(sfx, red);
    sff = pd_clef_block_sum(sff, red);
    sxx = pd_clef_block_sum(sxx, red);
    sr = pd_clef_block_sum(sr, red);
    if (threadIdx.x == 0) {
        const float prior =
            __fmul_rn(ps, __fdiv_rn(__fdiv_rn(sla, fmaxf(sqrtf(sll), 1e-12f)), fmaxf(sqrtf(saa), 1e-12f)));
        const float cosv =
            __fdiv_rn(__fdiv_rn(sfx, fmaxf(sqrtf(sff), 1e-8f)), fmaxf(sqrtf(sxx), 1e-8f));
        const float resid = __fadd_rn(sr, b3);
        logits[o] = __fadd_rn(prior, __fmul_rn(gate, fmaf(joint, cosv, resid)));
    }
}

PD_EXPORT
int pd_clef_score(const void* lex, const void* qvec, const void* glob, const void* qof,
                  const void* rof, const void* fields, const void* opts, const void* hid,
                  const void* w3, float b3, uint32_t dd, uint32_t w, float ps, float joint,
                  float gate, uint32_t nopt, void* logits, void* stream) {
    if (nopt == 0) return 0;
    if (dd == 0 || w == 0) return (int)cudaErrorInvalidValue;
    pd_pdl_go(pd_clef_score_kernel, dim3(nopt), dim3(256), 0, (cudaStream_t)stream,
              (const float*)lex, (const float*)qvec, (const float*)glob, (const uint32_t*)qof,
              (const uint32_t*)rof, (const float*)fields, (const float*)opts, (const float*)hid,
              (const float*)w3, b3, dd, w, ps, joint, gate, (float*)logits);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 732 backbone attention
// Causal softmax(q k^T * scale) v for one request's rows, head_dim 256,
// grouped (q head h reads kv head h / (heads / kv_heads)), F32 class on the
// tensor cores: S = Q K^T and O += P V on m16n8k8 as 3xTF32 with a
// round-nearest drain every 32 channels / 16 keys - the diarization lane's
// FlashAttention-2 form (slot 723) at four times its head width.
//
// A 16-row x 256-wide F32 output is 128 accumulators a lane before its
// drain buffer, so a head is split across a warp PAIR: warp (rg, dh) owns
// query rows rg*16.. and output dims dh*128..+128. Each computes the
// partial Q K^T over its 128 channels, the pair trades partials through
// shared memory and adds them in one fixed order (low half + high half), so
// both hold the same bits of S; the online softmax runs in both, P V on each
// warp's own dims. A CTA is 32 query rows of one head (two pairs) over
// 16-key tiles of K and V staged as read - F32 rows straight out of the
// projection planes, no narrowing. Keys walk from the first in fixed tiles
// whatever tile holds a row: row-invariant, deterministic.
//
// Elected 32 rows x 16-key tiles, 4 warps (bench/clef_attn_gb10_bench.cu,
// GB10): 9.5-9.9 TF/s useful from 1092 rows up (the scalar F32 prefill
// tile it replaces ran ~2.5); the K/V tile loads pipelined against the
// softmax and P V took it from 7.6. Measured and not taken: 64 rows x 8
// keys over 8 warps (half the K/V traffic a row, but one S chain a warp:
// 7.1-9.4 TF/s).
#define PD_CLEF_TC_HD 256u
#define PD_CLEF_TC_SK (PD_CLEF_TC_HD + 4u)

// one 16-byte global -> shared copy, zero-filled when `ok` is false
__device__ __forceinline__ void pd_clef_cpa16(float* dst, const float* src, bool ok) {
    const unsigned sm = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(sm), "l"(src), "r"(ok ? 16u : 0u));
}

template <uint32_t BQ, uint32_t BKV>
__global__ void __launch_bounds__(BQ / 16u * 64u) pd_clef_attn_tc_kernel(
        const float* __restrict__ q, uint32_t ldq, const float* __restrict__ k, uint32_t ldk,
        const float* __restrict__ v, uint32_t ldv, float* __restrict__ out, uint32_t ldo, uint32_t n,
        uint32_t group, float scale) {
    constexpr uint32_t HD = PD_CLEF_TC_HD, HH = HD / 2u, SK = PD_CLEF_TC_SK, NTK = BKV / 8u,
                       KT8 = HH / 8u, NTD = HH / 8u, NTH = BQ / 16u * 64u;
    extern __shared__ __align__(16) float clef_tc_sm[];
    float* Qs = clef_tc_sm;          // [BQ][SK]
    float* Ks = Qs + BQ * SK;        // [BKV][SK]
    float* Vs = Ks + BKV * SK;       // [BKV][SK]
    float* Xs = Vs + BKV * SK;       // [warps][16][BKV] partial S
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5, gr = lane >> 2, t4 = lane & 3u;
    const uint32_t rg = warp >> 1, dh = warp & 1u;
    const uint32_t head = blockIdx.y, kvh = head / group;
    const uint32_t q0 = blockIdx.x * BQ, count = min(BQ, n - q0);
    const uint32_t r0 = rg * 16u + gr;              // this lane's rows r0 and r0 + 8
    const uint32_t qa = q0 + r0, qb = qa + 8u;      // their positions
    const uint32_t kend = q0 + count;                // keys 0 .. kend - 1 reach this tile
    const float* kb = k + kvh * HD;
    const float* vb = v + kvh * HD;
    // One K tile and one V tile, each loaded while the other is read: tile
    // t + 1's K goes out once tile t's S is done and lands under its softmax
    // and P V; its V goes out once that P V is done and lands under tile
    // t + 1's S. Every step commits a group (empty past the last tile), so
    // "one group pending" always means the older load has landed.
    auto stage = [&](float* dst, const float* src, uint32_t ld, uint32_t base) {
#pragma unroll
        for (uint32_t i = 0; i < BKV * HD / 4u / NTH; ++i) {
            const uint32_t f = tid + i * NTH, kj = f / (HD / 4u), d4 = (f % (HD / 4u)) * 4u;
            const bool ok = base + kj < kend;
            pd_clef_cpa16(&dst[kj * SK + d4], ok ? src + (size_t)(base + kj) * ld + d4 : src, ok);
        }
    };
    PD_PDL_ARM();
#pragma unroll
    for (uint32_t i = 0; i < BQ * HD / 4u / NTH; ++i) {
        const uint32_t f = tid + i * NTH, r = f / (HD / 4u), d4 = (f % (HD / 4u)) * 4u;
        const bool ok = r < count;
        pd_clef_cpa16(&Qs[r * SK + d4], ok ? q + (size_t)(q0 + r) * ldq + head * HD + d4 : q, ok);
    }
    stage(Ks, kb, ldk, 0);
    asm volatile("cp.async.commit_group;" ::: "memory");
    stage(Vs, vb, ldv, 0);
    asm volatile("cp.async.commit_group;" ::: "memory");
    asm volatile("cp.async.wait_group 1;" ::: "memory");
    __syncthreads();
    float o[NTD][4], m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
#pragma unroll
    for (uint32_t i = 0; i < NTD; ++i)
#pragma unroll
        for (uint32_t e = 0; e < 4u; ++e) o[i][e] = 0.f;
    float* xs_own = Xs + warp * 16u * BKV;
    const float* xs_lo = Xs + (warp & ~1u) * 16u * BKV;
    const float* xs_hi = xs_lo + 16u * BKV;
    for (uint32_t base = 0; base < kend; base += BKV) {
        const bool next = base + BKV < kend;
        // partial S over this warp's 128 channels: three mma a k8, a drain
        // every 32 channels
        float s[NTK][4], acc[NTK][4];
#pragma unroll
        for (uint32_t nt = 0; nt < NTK; ++nt)
#pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) s[nt][e] = acc[nt][e] = 0.f;
#pragma unroll
        for (uint32_t k8 = 0; k8 < KT8; ++k8) {
            const uint32_t d0 = dh * HH + k8 * 8u;
            const float* qr = Qs + r0 * SK + d0 + t4;
            const float a[4] = {qr[0], qr[8u * SK], qr[4u], qr[8u * SK + 4u]};
            uint32_t ab[4], as[4];
#pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) {
                ab[e] = pd_kumo_tf32(a[e]);
                as[e] = pd_kumo_tf32(a[e] - __uint_as_float(ab[e]));
            }
#pragma unroll
            for (uint32_t nt = 0; nt < NTK; ++nt) {
                const float* br = Ks + (nt * 8u + gr) * SK + d0 + t4;
                const float b0 = br[0], b1 = br[4u];
                const uint32_t bb0 = pd_kumo_tf32(b0), bb1 = pd_kumo_tf32(b1);
                const uint32_t bs0 = pd_kumo_tf32(b0 - __uint_as_float(bb0));
                const uint32_t bs1 = pd_kumo_tf32(b1 - __uint_as_float(bb1));
                pd_kumo_mma(acc[nt], ab, bb0, bb1);
                pd_kumo_mma(acc[nt], ab, bs0, bs1);
                pd_kumo_mma(acc[nt], as, bb0, bb1);
            }
            if (k8 % 4u == 3u) {
#pragma unroll
                for (uint32_t nt = 0; nt < NTK; ++nt)
#pragma unroll
                    for (uint32_t e = 0; e < 4u; ++e) {
                        s[nt][e] = __fadd_rn(s[nt][e], acc[nt][e]);
                        acc[nt][e] = 0.f;
                    }
            }
        }
        // trade partials: both warps of the pair add low + high in that order
#pragma unroll
        for (uint32_t nt = 0; nt < NTK; ++nt)
#pragma unroll
            for (uint32_t e = 0; e < 4u; ++e)
                xs_own[(gr + (e >= 2u ? 8u : 0u)) * BKV + nt * 8u + 2u * t4 + (e & 1u)] = s[nt][e];
        __syncthreads();  // partials visible; every warp is done with this K tile
        if (next) stage(Ks, kb, ldk, base + BKV);
        asm volatile("cp.async.commit_group;" ::: "memory");
#pragma unroll
        for (uint32_t nt = 0; nt < NTK; ++nt)
#pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) {
                const uint32_t at = (gr + (e >= 2u ? 8u : 0u)) * BKV + nt * 8u + 2u * t4 + (e & 1u);
                s[nt][e] = __fadd_rn(xs_lo[at], xs_hi[at]);
            }
        // the online softmax, rows r0 (h = 0) and r0 + 8 (h = 1): a lane holds
        // keys nt * 8 + 2 t4 + {0, 1}; a row's four lanes reduce by xor 1, 2
        float corr[2];
#pragma unroll
        for (uint32_t h = 0; h < 2u; ++h) {
            const uint32_t qi = h ? qb : qa;
            float hi = m[h];
#pragma unroll
            for (uint32_t nt = 0; nt < NTK; ++nt)
#pragma unroll
                for (uint32_t c = 0; c < 2u; ++c) {
                    float& z = s[nt][2u * h + c];
                    const uint32_t kj = base + nt * 8u + 2u * t4 + c;
                    z = kj <= qi && kj < kend ? __fmul_rn(z, scale) : -INFINITY;
                    hi = fmaxf(hi, z);
                }
            hi = fmaxf(hi, __shfl_xor_sync(0xffffffffu, hi, 1));
            hi = fmaxf(hi, __shfl_xor_sync(0xffffffffu, hi, 2));
            corr[h] = m[h] == -INFINITY ? 0.f : expf(__fsub_rn(m[h], hi));
            float sum = 0.f;
#pragma unroll
            for (uint32_t nt = 0; nt < NTK; ++nt)
#pragma unroll
                for (uint32_t c = 0; c < 2u; ++c) {
                    float& z = s[nt][2u * h + c];
                    z = z == -INFINITY ? 0.f : expf(__fsub_rn(z, hi));
                    sum = __fadd_rn(sum, z);
                }
            sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, 1));
            sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, 2));
            l[h] = fmaf(l[h], corr[h], sum);
            m[h] = hi;
        }
        asm volatile("cp.async.wait_group 1;" ::: "memory");  // this tile's V
        __syncthreads();
        // O = O * corr + P V over this warp's dims, P from S's accumulator
        // (within an 8-key block slot t4 <- key 2 t4, slot t4 + 4 <- key
        // 2 t4 + 1, V read with the same permutation), one drain a tile
        float pv[NTD][4];
#pragma unroll
        for (uint32_t nd = 0; nd < NTD; ++nd)
#pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) pv[nd][e] = 0.f;
#pragma unroll
        for (uint32_t kt = 0; kt < NTK; ++kt) {
            const float a[4] = {s[kt][0], s[kt][2], s[kt][1], s[kt][3]};
            uint32_t pb[4], ps[4];
#pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) {
                pb[e] = pd_kumo_tf32(a[e]);
                ps[e] = pd_kumo_tf32(a[e] - __uint_as_float(pb[e]));
            }
            const float* v0 = Vs + (kt * 8u + 2u * t4) * SK + dh * HH + gr;
#pragma unroll
            for (uint32_t nd = 0; nd < NTD; ++nd) {
                const float b0 = v0[nd * 8u], b1 = v0[SK + nd * 8u];
                const uint32_t bb0 = pd_kumo_tf32(b0), bb1 = pd_kumo_tf32(b1);
                const uint32_t bs0 = pd_kumo_tf32(b0 - __uint_as_float(bb0));
                const uint32_t bs1 = pd_kumo_tf32(b1 - __uint_as_float(bb1));
                pd_kumo_mma(pv[nd], pb, bb0, bb1);
                pd_kumo_mma(pv[nd], pb, bs0, bs1);
                pd_kumo_mma(pv[nd], ps, bb0, bb1);
            }
        }
#pragma unroll
        for (uint32_t nd = 0; nd < NTD; ++nd)
#pragma unroll
            for (uint32_t e = 0; e < 4u; ++e)
                o[nd][e] = __fadd_rn(__fmul_rn(o[nd][e], corr[e >> 1]), pv[nd][e]);
        __syncthreads();  // every warp is done with this V tile
        if (next) stage(Vs, vb, ldv, base + BKV);
        asm volatile("cp.async.commit_group;" ::: "memory");
        asm volatile("cp.async.wait_group 1;" ::: "memory");  // the next tile's K
        __syncthreads();
    }
#pragma unroll
    for (uint32_t h = 0; h < 2u; ++h) {
        const uint32_t r = r0 + 8u * h;
        if (r >= count) continue;
        float* po = out + (size_t)(q0 + r) * ldo + head * HD + dh * HH + 2u * t4;
#pragma unroll
        for (uint32_t nd = 0; nd < NTD; ++nd)
            *reinterpret_cast<float2*>(po + nd * 8u) =
                make_float2(__fdiv_rn(o[nd][2u * h], l[h]), __fdiv_rn(o[nd][2u * h + 1u], l[h]));
    }
}

template <uint32_t BQ, uint32_t BKV>
static void pd_clef_attn_tc_go(const void* q, uint32_t ldq, const void* k, uint32_t ldk, const void* v,
                               uint32_t ldv, void* out, uint32_t ldo, uint32_t n, uint32_t heads,
                               uint32_t group, float scale, cudaStream_t s) {
    constexpr uint32_t smem =
        (BQ * PD_CLEF_TC_SK + 2u * BKV * PD_CLEF_TC_SK + BQ / 16u * 2u * 16u * BKV) * 4u;
    static bool set = false;
    if (!set) {
        cudaFuncSetAttribute(pd_clef_attn_tc_kernel<BQ, BKV>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem);
        set = true;
    }
    pd_pdl_go(pd_clef_attn_tc_kernel<BQ, BKV>, dim3((n + BQ - 1u) / BQ, heads), dim3(BQ / 16u * 64u), smem, s,
              (const float*)q, ldq, (const float*)k, ldk, (const float*)v, ldv, (float*)out, ldo, n, group,
              scale);
}

PD_EXPORT
int pd_clef_attn_tc(const void* q, uint32_t ldq, const void* k, uint32_t ldk, const void* v,
                    uint32_t ldv, void* out, uint32_t ldo, uint32_t n, uint32_t heads,
                    uint32_t kv_heads, float scale, void* stream) {
    if (n == 0) return 0;
    if (kv_heads == 0 || heads % kv_heads || ldq < heads * PD_CLEF_TC_HD ||
        ldo < heads * PD_CLEF_TC_HD || ldk < kv_heads * PD_CLEF_TC_HD ||
        ldv < kv_heads * PD_CLEF_TC_HD || ((ldq | ldk | ldv | ldo) & 3u) ||
        (((uintptr_t)q | (uintptr_t)k | (uintptr_t)v | (uintptr_t)out) & 15u))
        return (int)cudaErrorInvalidValue;
    pd_clef_attn_tc_go<32u, 16u>(q, ldq, k, ldk, v, ldv, out, ldo, n, heads, heads / kv_heads, scale,
                                 (cudaStream_t)stream);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 733 backbone GEMM
// The backbone's projections: the stored-weight GEMM of slot 722 with the
// F32 activation split TWO ways in bf16 (hi + mid, 16 significant bits)
// against the exact bf16 weight - two mma a k16 where the F32 class issues
// three. Elected by Clef's gate against the reference's F32 evaluation: with
// the iterated rope (both classes) every fixture's logits stayed within
// 5.7e-5 and probabilities 1.3e-5 here against the F32 class's 6.1e-5 /
// 7.1e-6 - the long fixtures' error was the rope's either way; with the
// tabled rope (734) 3.7e-5 / 8.3e-6, still ~1000x inside the vendor's own
// BF16 (0.059 / 0.0118). The head keeps the F32 class (722); it is a few
// percent of a pass.
//
// Tiles (bench/clef_gemm_gb10_bench.cu, GB10): 64 x 128 over a 2 x 2 warp
// grid holds 31-34 TF/s useful on the 4096-wide planes and the 12288-deep
// down (the F32 class: 26-27); a 64 x 64 grid short of two CTAs an SM takes
// 64 x 64 / 4 warps on the 3-slot ring (k/v at 300 rows 89 us). Fixed k walk,
// no K split: a row's bits are its own whatever shares its pass.
PD_EXPORT
int pd_clef_gemm(const void* x, const void* w, const void* bias, void* y, uint32_t K, uint32_t N,
                 uint32_t M, uint32_t mode, void* stream) {
    if (M == 0 || N == 0) return 0;
    if (K == 0 || K % 32u || N % 4u || mode > PD_DIAR_GELU_TANH || mode == PD_DIAR_ROPE ||
        ((uintptr_t)x & 15u) || ((uintptr_t)w & 15u) || (M + 15u) / 16u > 65535u)
        return (int)cudaErrorInvalidValue;
    static int nsm = 0;
    if (nsm == 0) {
        int d = 0;
        cudaGetDevice(&d);
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, d);
        if (nsm <= 0) nsm = 48;
    }
    const cudaStream_t s = (cudaStream_t)stream;
    const uint64_t g64 = (uint64_t)((M + 63u) / 64u) * ((N + 63u) / 64u);
    if (g64 < 2u * (uint64_t)nsm)
        pd_diar_gemm_go<64u, 64u, 4u, 1u, 3u, PD_DIAR_W_BF16, 2u>((const float*)x, (const uint8_t*)w, nullptr,
                                                               (const float*)bias, (float*)y, nullptr, nullptr,
                                                               K, N, M, mode, s);
    else
        pd_diar_gemm_go<64u, 128u, 2u, 2u, 2u, PD_DIAR_W_BF16, 2u>((const float*)x, (const uint8_t*)w, nullptr,
                                                                (const float*)bias, (float*)y, nullptr,
                                                                nullptr, K, N, M, mode, s);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 734 rope
// The backbone's rotary embedding from a table: the first 64 of every
// 256-wide head rotate as 32 pairs (i, i + 32) - rotate_half - in the
// reference's operation order, q * cos + rotate_half(q) * sin, F32. Pair k's
// angle is position * inv_freq[k]; the engine tabulates (cos, sin) per
// (position, k) once, the angle formed as the reference forms it (one F32
// product of the position and the model's own F32 inv_freq) and cos / sin
// evaluated in F64 and rounded once - no angle is ever iterated, so the
// rotation does not lose bits with depth. Interleaved mrope: pair k reads the
// position of axis h where `hmask` has bit k, w where `wmask` has it, t
// otherwise (`pos` is [3][rows]); a text row carries one position on every
// axis.
__global__ void pd_clef_rope_kernel(float* __restrict__ x, uint32_t heads, uint32_t rows,
                                    const uint32_t* __restrict__ pos, const float2* __restrict__ table,
                                    uint32_t max_pos, uint32_t hmask, uint32_t wmask) {
    PD_PDL_ARM();
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * heads * 32u) return;
    const uint32_t k = i & 31u, head = (i >> 5) % heads, row = (i >> 5) / heads;
    const uint32_t axis = (hmask >> k) & 1u ? 1u : ((wmask >> k) & 1u ? 2u : 0u);
    const uint32_t p = min(pos[axis * rows + row], max_pos - 1u);
    const float2 cs = table[(size_t)p * 32u + k];
    float* h = x + ((size_t)row * heads + head) * 256u;
    const float x1 = h[k], x2 = h[k + 32u];
    h[k] = __fadd_rn(__fmul_rn(x1, cs.x), __fmul_rn(-x2, cs.y));
    h[k + 32u] = __fadd_rn(__fmul_rn(x2, cs.x), __fmul_rn(x1, cs.y));
}

PD_EXPORT
int pd_clef_rope(void* x, uint32_t heads, uint32_t rows, const void* pos, const void* table, uint32_t max_pos,
                 uint32_t hmask, uint32_t wmask, void* stream) {
    if (rows == 0 || heads == 0) return 0;
    if (max_pos == 0) return (int)cudaErrorInvalidValue;
    const uint32_t n = rows * heads * 32u;
    pd_pdl_go(pd_clef_rope_kernel, dim3((n + 255u) / 256u), dim3(256), 0, (cudaStream_t)stream, (float*)x,
              heads, rows, (const uint32_t*)pos, (const float2*)table, max_pos, hmask, wmask);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 741 backbone GEMM, Q8_0
// The GGUF's backbone projections (ggml-org's Clef conversions, Q8_0): 733's
// two-part split against the file's own weights - int8 rows [N][K] and its
// f16 block scales [K/32][N] (PD_DIAR_W_Q8H), repacked from the 34-byte
// blocks at load, 8.5 bits a weight resident as on disk. |q| <= 127 is an
// exact bf16 operand, so hi.q and mid.q are 733's two products; a 32-deep
// tile IS a Q8_0 block, so its integer-weight sum is scaled once at the
// drain, fma(d, acc, fac) - the dequantized product with one rounding fewer.
//
// Tiles (bench/clef_gemm_q8_gb10_bench.cu, GB10, every arm bit-identical):
// 64 x 128 over a 2 x 2 warp grid on a 3-slot ring holds 31-37 TF/s at 300
// and 1092 rows on the 9B and 27B planes - level with 733 on BF16 there and
// ahead where the weight stream binds (the MLP down 12288 -> 4096 at 300
// rows 1184 against 1527 us); past K 12288 (the 27B's down, 17408 deep) the
// 2-slot ring (300 rows 2620, 1092 8531, 4096 29471 us against 2558 / 10405
// / 49315 on three). A 64 x 64 grid short of two CTAs an SM takes 32 x 128
// over four warps along N (k/v at 300 rows 91 us), 64 x 64 / 4x1 on three
// slots for the 64-wide [a | b] plane. At 4096 rows the int8 -> bf16
// conversion of the B fragments shows: 35-36 TF/s against BF16's 38-41.
PD_EXPORT
int pd_clef_gemm_q8(const void* x, const void* w, const void* scale, const void* bias, void* y, uint32_t K,
                    uint32_t N, uint32_t M, uint32_t mode, void* stream) {
    if (M == 0 || N == 0) return 0;
    // 16 B cp.async rows: K a multiple of the block, N of a scale chunk's 8
    // halves, operands 16 B aligned
    if (K == 0 || K % 32u || N % 8u || mode > PD_DIAR_GELU_TANH || mode == PD_DIAR_ROPE || !scale ||
        ((uintptr_t)x & 15u) || ((uintptr_t)w & 15u) || ((uintptr_t)scale & 15u) || (M + 15u) / 16u > 65535u)
        return (int)cudaErrorInvalidValue;
    static int nsm = 0;
    if (nsm == 0) {
        int d = 0;
        cudaGetDevice(&d);
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, d);
        if (nsm <= 0) nsm = 48;
    }
    const cudaStream_t s = (cudaStream_t)stream;
    const float* xf = (const float*)x;
    const uint8_t* wq = (const uint8_t*)w;
    const float* b = (const float*)bias;
    float* yf = (float*)y;
    const uint64_t g64 = (uint64_t)((M + 63u) / 64u) * ((N + 63u) / 64u);
    if (g64 < 2u * (uint64_t)nsm) {
        if (N <= 64u)
            pd_diar_gemm_go<64u, 64u, 4u, 1u, 3u, PD_DIAR_W_Q8H, 2u>(xf, wq, scale, b, yf, nullptr, nullptr, K, N,
                                                                   M, mode, s);
        else
            pd_diar_gemm_go<32u, 128u, 1u, 4u, 3u, PD_DIAR_W_Q8H, 2u>(xf, wq, scale, b, yf, nullptr, nullptr, K,
                                                                    N, M, mode, s);
    } else if (K > 12288u) {
        pd_diar_gemm_go<64u, 128u, 2u, 2u, 2u, PD_DIAR_W_Q8H, 2u>(xf, wq, scale, b, yf, nullptr, nullptr, K, N, M,
                                                                mode, s);
    } else {
        pd_diar_gemm_go<64u, 128u, 2u, 2u, 3u, PD_DIAR_W_Q8H, 2u>(xf, wq, scale, b, yf, nullptr, nullptr, K, N, M,
                                                                mode, s);
    }
    return pd_launch_status();
}

// ------------------------------------------------------------------ 742 lexical mean, Q8_0
// 726 over the GGUF's output embedding as stored: Q8_0 rows of `d` / 32
// blocks (an f16 scale, 32 int8). A value widens exactly - (float)q * d is
// at most 18 significant bits - and the span sums in order, so the mean is
// 726's arithmetic on the file's values.
__global__ void pd_clef_lex_mean_q8_kernel(const uint8_t* __restrict__ table, uint32_t d,
                                           const uint32_t* __restrict__ ids,
                                           const uint32_t* __restrict__ spans, float* __restrict__ out) {
    PD_PDL_ARM();
    const uint32_t s0 = spans[2u * blockIdx.x], s1 = spans[2u * blockIdx.x + 1u];
    float* po = out + (size_t)blockIdx.x * d;
    const float n = (float)(s1 - s0);
    const size_t row = (size_t)(d / 32u) * 34u;
    for (uint32_t c = threadIdx.x; c < d; c += blockDim.x) {
        float s = 0.f;
        for (uint32_t i = s0; i < s1; ++i) {
            const uint8_t* blk = table + (size_t)ids[i] * row + (c >> 5) * 34u;
            const float scale = __half2float(*reinterpret_cast<const __half*>(blk));
            s = __fadd_rn(s, __fmul_rn((float)(int8_t)blk[2u + (c & 31u)], scale));
        }
        po[c] = __fdiv_rn(s, n);
    }
}

PD_EXPORT
int pd_clef_lex_mean_q8(const void* table, uint32_t d, const void* ids, const void* spans, uint32_t n,
                        void* out, void* stream) {
    if (n == 0) return 0;
    if (d == 0 || d % 32u) return (int)cudaErrorInvalidValue;
    pd_pdl_go(pd_clef_lex_mean_q8_kernel, dim3(n), dim3(256), 0, (cudaStream_t)stream, (const uint8_t*)table, d,
              (const uint32_t*)ids, (const uint32_t*)spans, (float*)out);
    return pd_launch_status();
}
