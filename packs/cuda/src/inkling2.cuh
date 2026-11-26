// inkling2.cuh - Thinking Machines' Inkling 2 Mini glue: the block attention
// residuals, the loop controllers' message mix, the attention's per-head
// prep (scale, L2 clip, unit RMS, log scale), the short convolutions (static
// and per-token dynamic taps, carried state) and the softcapped SwiGLU.
// f32 throughout; their consumers are the bf16 / NVFP4 / MXFP8 linears
// around them (abi.cuh helpers only). Every producer plane is read through
// pd_pdl_after - these all launch as PDL dependents.

// Block-reduce `n` (<= 32) per-thread partials over a 256-thread CTA; every
// thread gets the sums back in `v`.
template <uint32_t N>
__device__ __forceinline__ void pd_ik2_block_sum(float (&v)[N], uint32_t n, float* red) {
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    for (uint32_t i = 0; i < n; ++i) {
        float a = v[i];
        for (uint32_t sh = 16; sh > 0; sh >>= 1) a += __shfl_xor_sync(0xffffffffu, a, sh);
        if (lane == 0) red[warp * N + i] = a;
    }
    __syncthreads();
    for (uint32_t i = 0; i < n; ++i) {
        float a = 0.0f;
        for (uint32_t w = 0; w < (blockDim.x >> 5); ++w) a += red[w * N + i];
        v[i] = a;
    }
    __syncthreads();
}

struct PdIk2States {
    const float* s[6];
};

// slot 977: block attention residual. Per row, the states are the `ns`
// block states plus the current partial sum, each taken to unit RMS (eps);
// query q mixes them by softmax(<Q_q, n_i> / sqrt(width)):
// out[q][r] = sum_i p_qi * n_i. nq queries (1 for an MLP input, 3 for a
// full layer's q/k/v inputs, 4 for the loop's messages), out row q at
// out + q * qstride + r * width. width <= 256 * 9. Instantiated per (S = ns
// + 1, Q = nq): a decode row is one CTA, and its block reduction of S + S Q
// sums is the kernel's whole cost (each sum reduced on its own - the count
// changes nothing else).
template <uint32_t S, uint32_t Q>
__global__ void __launch_bounds__(256) pd_inkling2_attn_res_kernel(
        PdIk2States st, const float* partial, const float* __restrict__ query,
        float* __restrict__ out, uint32_t width, uint32_t qstride, float eps) {
    constexpr uint32_t V = 9u, N = S + S * Q;
    PD_PDL_ARM();
    __shared__ float red[8 * N];
    const uint32_t r = blockIdx.x, tid = threadIdx.x;
    float x[S][V];
    #pragma unroll
    for (uint32_t i = 0; i < S; ++i) {
        const float* src = i + 1u < S ? pd_pdl_after(st.s[i]) : pd_pdl_after(partial);
        #pragma unroll
        for (uint32_t k = 0; k < V; ++k) {
            const uint32_t d = tid + k * 256u;
            x[i][k] = d < width ? src[(size_t)r * width + d] : 0.0f;
        }
    }
    // [0, S): sums of squares; [S + q * S + i]: <Q_q, x_i>
    float acc[N];
    #pragma unroll
    for (uint32_t i = 0; i < N; ++i) acc[i] = 0.0f;
    #pragma unroll
    for (uint32_t k = 0; k < V; ++k) {
        const uint32_t d = tid + k * 256u;
        if (d >= width) continue;
        float qv[Q];
        #pragma unroll
        for (uint32_t q = 0; q < Q; ++q) qv[q] = query[q * width + d];
        #pragma unroll
        for (uint32_t i = 0; i < S; ++i) {
            acc[i] = fmaf(x[i][k], x[i][k], acc[i]);
            #pragma unroll
            for (uint32_t q = 0; q < Q; ++q) acc[S + q * S + i] = fmaf(qv[q], x[i][k], acc[S + q * S + i]);
        }
    }
    pd_ik2_block_sum<N>(acc, N, red);
    float inv[S];
    #pragma unroll
    for (uint32_t i = 0; i < S; ++i) inv[i] = rsqrtf(acc[i] / (float)width + eps);
    const float sc = rsqrtf((float)width);
    #pragma unroll
    for (uint32_t q = 0; q < Q; ++q) {
        float lg[S], m = -INFINITY;
        #pragma unroll
        for (uint32_t i = 0; i < S; ++i) {
            lg[i] = acc[S + q * S + i] * inv[i] * sc;
            m = fmaxf(m, lg[i]);
        }
        float den = 0.0f;
        #pragma unroll
        for (uint32_t i = 0; i < S; ++i) {
            lg[i] = expf(lg[i] - m);
            den += lg[i];
        }
        #pragma unroll
        for (uint32_t i = 0; i < S; ++i) lg[i] = lg[i] / den * inv[i];
        float* o = out + (size_t)q * qstride + (size_t)r * width;
        #pragma unroll
        for (uint32_t k = 0; k < V; ++k) {
            const uint32_t d = tid + k * 256u;
            if (d >= width) continue;
            float a = 0.0f;
            #pragma unroll
            for (uint32_t i = 0; i < S; ++i) a = fmaf(lg[i], x[i][k], a);
            o[d] = a;
        }
    }
}

PD_EXPORT
int pd_inkling2_attn_res(const void* const* states, uint32_t ns, const void* partial,
                         const void* query, uint32_t nq, void* out, uint32_t rows,
                         uint32_t width, uint32_t qstride, float eps, void* stream) {
    if (rows == 0) return 0;
    if (ns > 5u || nq == 0 || nq > 4u || width > 256u * 9u) return cudaErrorInvalidValue;
    PdIk2States st{};
    for (uint32_t i = 0; i < ns; ++i) st.s[i] = (const float*)states[i];
    #define PD_IK2_RES(SS, QQ)                                                                  \
        pd_pdl_go(pd_inkling2_attn_res_kernel<SS, QQ>, rows, 256, 0u, (cudaStream_t)stream, st, \
                  (const float*)partial, (const float*)query, (float*)out, width, qstride, eps)
    #define PD_IK2_RES_Q(SS)                    \
        switch (nq) {                           \
        case 1: PD_IK2_RES(SS, 1u); break;      \
        case 2: PD_IK2_RES(SS, 2u); break;      \
        case 3: PD_IK2_RES(SS, 3u); break;      \
        default: PD_IK2_RES(SS, 4u); break;     \
        }
    switch (ns + 1u) {
    case 1: PD_IK2_RES_Q(1u); break;
    case 2: PD_IK2_RES_Q(2u); break;
    case 3: PD_IK2_RES_Q(3u); break;
    case 4: PD_IK2_RES_Q(4u); break;
    case 5: PD_IK2_RES_Q(5u); break;
    default: PD_IK2_RES_Q(6u); break;
    }
    #undef PD_IK2_RES_Q
    #undef PD_IK2_RES
    return pd_launch_status();
}

// slot 978: the loop controller's gate input, cat(rms(h), rms(m)) rounded to
// bf16 (the reference casts the concatenation to the down projection's
// dtype): out[r] = [bf16(h_r / rms(h_r)), bf16(m_r / rms(m_r))], 2 * width.
__global__ void __launch_bounds__(256) pd_inkling2_msg_cat_kernel(
        const float* h, const float* m, float* __restrict__ out, uint32_t width, float eps) {
    constexpr uint32_t V = 9u;
    PD_PDL_ARM();
    __shared__ float red[8 * 2];
    const uint32_t r = blockIdx.x, tid = threadIdx.x;
    const float* hr = pd_pdl_after(h) + (size_t)r * width;
    const float* mr = pd_pdl_after(m) + (size_t)r * width;
    float a[V], b[V], ss[2] = {0.0f, 0.0f};
    #pragma unroll
    for (uint32_t k = 0; k < V; ++k) {
        const uint32_t d = tid + k * 256u;
        a[k] = d < width ? hr[d] : 0.0f;
        b[k] = d < width ? mr[d] : 0.0f;
        ss[0] = fmaf(a[k], a[k], ss[0]);
        ss[1] = fmaf(b[k], b[k], ss[1]);
    }
    pd_ik2_block_sum<2>(ss, 2, red);
    const float ia = rsqrtf(ss[0] / (float)width + eps), ib = rsqrtf(ss[1] / (float)width + eps);
    float* o = out + (size_t)r * 2u * width;
    #pragma unroll
    for (uint32_t k = 0; k < V; ++k) {
        const uint32_t d = tid + k * 256u;
        if (d >= width) continue;
        o[d] = __bfloat162float(__float2bfloat16_rn(a[k] * ia));
        o[width + d] = __bfloat162float(__float2bfloat16_rn(b[k] * ib));
    }
}

PD_EXPORT
int pd_inkling2_msg_cat(const void* h, const void* m, void* out, uint32_t rows, uint32_t width,
                        float eps, void* stream) {
    if (rows == 0) return 0;
    if (width > 256u * 9u) return cudaErrorInvalidValue;
    pd_pdl_go(pd_inkling2_msg_cat_kernel, rows, 256, 0u, (cudaStream_t)stream, (const float*)h,
              (const float*)m, (float*)out, width, eps);
    return pd_launch_status();
}

// slot 979: the loop controller's mix. z = bf16(D cat) (the down projection's
// bf16 output, [rows, rank]); f = F z, g = G z in f32 over the bf16 F / G
// planes ([width, rank] each, the reference's fp32 modules hold exactly those
// values); out = (1 - tanh f) * h + tanh g * m. rank <= 128, a multiple of 8.
// A CTA: 256 columns (64 at decode widths) x up to MIX_ROWS rows - each thread reads its two
// weight rows once (16-byte loads, k ascending) for all of the tile's rows,
// so a prompt chunk no longer re-reads the planes a row; every (row,
// column)'s FMA chain is the same k-ascending one as a row at a time.
#define PD_IK2_MIX_ROWS 16u
__global__ void __launch_bounds__(256) pd_inkling2_msg_mix_kernel(
        const float* z, const __nv_bfloat16* __restrict__ fw, const __nv_bfloat16* __restrict__ gw,
        const float* h, const float* m, float* __restrict__ out, uint32_t rows, uint32_t width,
        uint32_t rank) {
    constexpr uint32_t R = PD_IK2_MIX_ROWS;
    PD_PDL_ARM();
    __shared__ float zs[R][128];
    const uint32_t r0 = blockIdx.y * R, j = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t nr = min(R, rows - r0);
    const float* zr = pd_pdl_after(z);
    for (uint32_t i = threadIdx.x; i < R * rank; i += blockDim.x) {
        const uint32_t rr = i / rank, k = i - rr * rank;
        zs[rr][k] = rr < nr ? __bfloat162float(__float2bfloat16_rn(zr[(size_t)(r0 + rr) * rank + k])) : 0.0f;
    }
    __syncthreads();
    if (j >= width) return;
    const uint4* fr = (const uint4*)(fw + (size_t)j * rank);
    const uint4* gr = (const uint4*)(gw + (size_t)j * rank);
    float f[R], g[R];
    #pragma unroll
    for (uint32_t rr = 0; rr < R; ++rr) f[rr] = g[rr] = 0.0f;
    for (uint32_t k8 = 0; k8 < rank / 8u; ++k8) {
        const uint4 fv = fr[k8], gv = gr[k8];
        const __nv_bfloat16* fb = (const __nv_bfloat16*)&fv;
        const __nv_bfloat16* gb = (const __nv_bfloat16*)&gv;
        #pragma unroll
        for (uint32_t q = 0; q < 8u; ++q) {
            const float wf = __bfloat162float(fb[q]), wg = __bfloat162float(gb[q]);
            #pragma unroll
            for (uint32_t rr = 0; rr < R; ++rr) {
                const float zk = zs[rr][k8 * 8u + q];
                f[rr] = fmaf(wf, zk, f[rr]);
                g[rr] = fmaf(wg, zk, g[rr]);
            }
        }
    }
    const float* hr = pd_pdl_after(h);
    const float* mr = pd_pdl_after(m);
    #pragma unroll
    for (uint32_t rr = 0; rr < R; ++rr) {
        if (rr >= nr) break;
        const size_t o = (size_t)(r0 + rr) * width + j;
        out[o] = (1.0f - tanhf(f[rr])) * hr[o] + tanhf(g[rr]) * mr[o];
    }
}

PD_EXPORT
int pd_inkling2_msg_mix(const void* z, const void* fw, const void* gw, const void* h,
                        const void* m, void* out, uint32_t rows, uint32_t width, uint32_t rank,
                        void* stream) {
    if (rows == 0) return 0;
    if (rank > 128u || rank % 8u) return cudaErrorInvalidValue;
    // decode rows: narrower CTAs, so one row's 2304 columns spread over the die
    const uint32_t nt = rows <= 4u ? 64u : 256u;
    pd_pdl_go(pd_inkling2_msg_mix_kernel,
              dim3((width + nt - 1u) / nt, (rows + PD_IK2_MIX_ROWS - 1u) / PD_IK2_MIX_ROWS), nt, 0u,
              (cudaStream_t)stream, (const float*)z, (const __nv_bfloat16*)fw,
              (const __nv_bfloat16*)gw, (const float*)h, (const float*)m, (float*)out, rows, width,
              rank);
    return pd_launch_status();
}

// slot 1008: a loop controller's down projection off its two inputs in one
// pass. msg_cat's rows (h and m each at unit RMS, rounded to bf16) feed the
// tensor cores straight from registers instead of a [rows][2 width] f32
// plane the next GEMM reads back: a CTA takes 16 rows, finds each row's two
// inverse RMS with msg_cat's reduction (256 threads, the same order: the
// same values), then streams K = 2 width in 64-column chunks - the bf16 rows
// and the down plane D [rank <= 64][2 width] through smem - into m16n8k16
// (bf16, f32 accumulate), a warp an 8-column slice of z. z's sum runs one
// k-ascending chain a column (the ladder's K-split regroups it; z is read
// as bf16 after).
#define PD_IK2_MD_R 16u
#define PD_IK2_MD_KC 64u
#define PD_IK2_MD_LD 72u  // bf16 row stride (64 + 8): conflict-free ldmatrix
__global__ void __launch_bounds__(256) pd_inkling2_msg_down_kernel(
        const float* h, const float* m, const __nv_bfloat16* __restrict__ dw,
        float* __restrict__ z, uint32_t rows, uint32_t width, uint32_t rank, float eps) {
    constexpr uint32_t R = PD_IK2_MD_R, KC = PD_IK2_MD_KC, LD = PD_IK2_MD_LD, V = 9u;
    PD_PDL_ARM();
    __shared__ float red[8 * 2];
    __shared__ float sia[R], sib[R];
    __shared__ __align__(16) __nv_bfloat16 as[R * LD];
    __shared__ __align__(16) __nv_bfloat16 bs[64u * LD];
    const uint32_t r0 = blockIdx.x * R, tid = threadIdx.x;
    const float* hp = pd_pdl_after(h);
    const float* mp = pd_pdl_after(m);
    for (uint32_t rr = 0; rr < R; ++rr) {
        const uint32_t r = r0 + rr;
        float ss[2] = {0.0f, 0.0f};
        if (r < rows) {
            const float* hr = hp + (size_t)r * width;
            const float* mr = mp + (size_t)r * width;
            #pragma unroll
            for (uint32_t k = 0; k < V; ++k) {
                const uint32_t d = tid + k * 256u;
                const float a = d < width ? hr[d] : 0.0f;
                const float b = d < width ? mr[d] : 0.0f;
                ss[0] = fmaf(a, a, ss[0]);
                ss[1] = fmaf(b, b, ss[1]);
            }
        }
        pd_ik2_block_sum<2>(ss, 2, red);
        if (tid == 0) {
            sia[rr] = rsqrtf(ss[0] / (float)width + eps);
            sib[rr] = rsqrtf(ss[1] / (float)width + eps);
        }
    }
    __syncthreads();
    const uint32_t lane = tid & 31u, warp = tid >> 5, g = lane >> 2, tq = lane & 3u;
    const uint32_t k2 = 2u * width;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t k0 = 0; k0 < k2; k0 += KC) {
        // the chunk's bf16 rows, msg_cat's values
        for (uint32_t u = tid; u < R * KC; u += 256u) {
            const uint32_t rr = u / KC, kk = u % KC, k = k0 + kk, r = r0 + rr;
            float v = 0.0f;
            if (r < rows && k < k2)
                v = k < width ? hp[(size_t)r * width + k] * sia[rr]
                              : mp[(size_t)r * width + (k - width)] * sib[rr];
            as[rr * LD + kk] = __float2bfloat16_rn(v);
        }
        for (uint32_t u = tid; u < 64u * (KC / 8u); u += 256u) {
            const uint32_t j = u / (KC / 8u), c = (u % (KC / 8u)) * 8u;
            uint4 val = make_uint4(0, 0, 0, 0);
            if (j < rank && k0 + c < k2) val = *(const uint4*)(dw + (size_t)j * k2 + k0 + c);
            *(uint4*)(bs + j * LD + c) = val;
        }
        __syncthreads();
        #pragma unroll
        for (uint32_t kp = 0; kp < KC / 32u; ++kp) {
            uint32_t b[4];
            pd_ldm_x4(b, (const unsigned char*)(bs + (warp * 8u + (lane & 7u)) * LD + kp * 32u +
                                                (lane >> 3) * 8u));
            #pragma unroll
            for (uint32_t hh = 0; hh < 2u; ++hh) {
                uint32_t a[4];
                pd_ldm_x4(a, (const unsigned char*)(as + (lane & 15u) * LD + kp * 32u + hh * 16u +
                                                    (lane >> 4) * 8u));
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[2u * hh]),
                      "r"(b[2u * hh + 1u]));
            }
        }
        __syncthreads();
    }
    // acc: (row g, cols 2tq, 2tq + 1), (row g + 8, ...) of the warp's slice
    const uint32_t col = warp * 8u + 2u * tq;
    #pragma unroll
    for (uint32_t hh = 0; hh < 2u; ++hh) {
        const uint32_t r = r0 + g + 8u * hh;
        if (r >= rows) continue;
        if (col < rank) z[(size_t)r * rank + col] = acc[2u * hh];
        if (col + 1u < rank) z[(size_t)r * rank + col + 1u] = acc[2u * hh + 1u];
    }
}

PD_EXPORT
int pd_inkling2_msg_down(const void* h, const void* m, const void* dw, void* z, uint32_t rows,
                         uint32_t width, uint32_t rank, float eps, void* stream) {
    if (rows == 0) return 0;
    // msg_cat's reduction (9 a thread) and an 8-column slice a warp
    if (width > 256u * 9u || width % 32u || rank > 64u || rank % 8u) return cudaErrorInvalidValue;
    pd_pdl_go(pd_inkling2_msg_down_kernel, (rows + PD_IK2_MD_R - 1u) / PD_IK2_MD_R, 256, 0u,
              (cudaStream_t)stream, (const float*)h, (const float*)m, (const __nv_bfloat16*)dw,
              (float*)z, rows, width, rank, eps);
    return pd_launch_status();
}

// slot 980: the attention's per-head prep, in place over x [rows][heads][d]
// (row r at x + r * stride): optionally a learned per-(head, dim) scale
// (bf16), then either an L2 clip to `clip` (x *= min(clip / (|x| + 1e-6),
// 1)) or, with `rms`, unit RMS (eps); then, with `pos`, the full layers' log
// scale 1 + alpha * ln(max((pos + 1) / floor, 1)). One warp a (row, head).
// Queries: scale + clip (+ log); relative states: the same at their width;
// keys: clip alone; values: unit RMS.
__global__ void __launch_bounds__(256) pd_inkling2_head_prep_kernel(
        float* x, uint32_t rows, uint32_t heads, uint32_t d, uint32_t stride,
        const __nv_bfloat16* __restrict__ scale, float clip, uint32_t rms, float eps,
        const uint32_t* pos, float alpha, float floor_) {
    PD_PDL_ARM();
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t w = blockIdx.x * 8u + (threadIdx.x >> 5);
    if (w >= rows * heads) return;
    const uint32_t r = w / heads, h = w - r * heads;
    // x is read and written here, a plain (non-restrict) plane: its loads
    // are ordinary LDG, which the wait does order
    float* xr = x + (size_t)r * stride + (size_t)h * d;
    float v[4];
    float ss = 0.0f;
    #pragma unroll
    for (uint32_t k = 0; k < 4; ++k) {
        const uint32_t i = lane + k * 32u;
        v[k] = 0.0f;
        if (i < d) {
            float a = xr[i];
            if (scale) a *= __bfloat162float(scale[h * d + i]);
            v[k] = a;
            ss = fmaf(a, a, ss);
        }
    }
    for (uint32_t sh = 16; sh > 0; sh >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, sh);
    float f;
    if (rms) {
        f = rsqrtf(ss / (float)d + eps);
    } else if (clip > 0.0f) {
        f = fminf(clip / (sqrtf(ss) + 1e-6f), 1.0f);
    } else {
        f = 1.0f;
    }
    if (pos) {
        const float p = ((float)pd_pdl_after(pos)[r] + 1.0f) / floor_;
        f *= 1.0f + alpha * logf(fmaxf(p, 1.0f));
    }
    #pragma unroll
    for (uint32_t k = 0; k < 4; ++k) {
        const uint32_t i = lane + k * 32u;
        if (i < d) xr[i] = v[k] * f;
    }
}

PD_EXPORT
int pd_inkling2_head_prep(void* x, uint32_t rows, uint32_t heads, uint32_t d, uint32_t stride,
                          const void* scale, float clip, uint32_t rms, float eps, const void* pos,
                          float alpha, float floor_, void* stream) {
    if (rows == 0) return 0;
    if (d > 128u || stride < heads * d) return cudaErrorInvalidValue;
    const uint32_t warps = rows * heads;
    pd_pdl_go(pd_inkling2_head_prep_kernel, (warps + 7u) / 8u, 256, 0u, (cudaStream_t)stream,
              (float*)x, rows, heads, d, stride, (const __nv_bfloat16*)scale, clip, rms, eps,
              (const uint32_t*)pos, alpha, floor_);
    return pd_launch_status();
}

// slot 981: the short causal convolution (k = 4 taps) with carried state,
// over runs of consecutive tokens of one sequence each: run i covers rows
// [run_off[i], run_off[i + 1]) and owns state block state + run_slot[i] * 3
// * C (its last three inputs, oldest first, f32). Static taps (dyn null):
// conv1d weight w [C][4], y[t] = sum_j w[j] x[t - 3 + j], out = (x + y) /
// sqrt(5). Dynamic taps (dyn non-null, the full layers' K / V): w [4][C] the
// static taps by lag (w[0] weights the current token), plus per token and
// head the taps dyn[t][h * 4 + lag] (h = channel / (C / heads)), out = (x +
// y) / 3. Each thread: one channel over a 64-token chunk; the chunk-0 thread
// of a (run, channel) reads the old state and writes the new one.
template <bool DYN>
__global__ void __launch_bounds__(256) pd_inkling2_sconv_kernel(
        const float* x, uint32_t xstride, float* __restrict__ out, uint32_t ostride, uint32_t C,
        const __nv_bfloat16* __restrict__ w, const float* dyn, uint32_t dstride, uint32_t heads,
        float* state, const uint32_t* run_off, const uint32_t* run_slot) {
    constexpr uint32_t T = 64u;
    PD_PDL_ARM();
    const uint32_t c = blockIdx.x * 256u + threadIdx.x;
    const uint32_t run = blockIdx.z, chunk = blockIdx.y;
    if (c >= C) return;
    const uint32_t* ro = pd_pdl_after(run_off);
    const uint32_t r0 = ro[run], r1 = ro[run + 1u];
    const uint32_t t0 = r0 + chunk * T;
    if (t0 >= r1) return;
    const uint32_t t1 = min(r1, t0 + T);
    const float* xs = pd_pdl_after(x);
    float* st = state + (size_t)pd_pdl_after(run_slot)[run] * 3u * C;
    // the three inputs before t0: the run's earlier rows, else the state
    float p[3];
    #pragma unroll
    for (uint32_t j = 0; j < 3; ++j) {
        const int t = (int)t0 - 3 + (int)j;  // oldest first
        p[j] = t >= (int)r0 ? xs[(size_t)t * xstride + c] : st[(size_t)(t - (int)r0 + 3) * C + c];
    }
    float wt[4];
    #pragma unroll
    for (uint32_t j = 0; j < 4; ++j) wt[j] = __bfloat162float(DYN ? w[(size_t)j * C + c] : w[(size_t)c * 4u + j]);
    const uint32_t hh = DYN ? c / (C / heads) : 0u;
    const float* dy = DYN ? pd_pdl_after(dyn) : nullptr;
    for (uint32_t t = t0; t < t1; ++t) {
        const float cur = xs[(size_t)t * xstride + c];
        float y;
        if (DYN) {
            const float* dt = dy + (size_t)t * dstride + hh * 4u;
            // lag 0 = the current token, lag 3 = the oldest
            y = (wt[0] + dt[0]) * cur;
            y = fmaf(wt[1] + dt[1], p[2], y);
            y = fmaf(wt[2] + dt[2], p[1], y);
            y = fmaf(wt[3] + dt[3], p[0], y);
            out[(size_t)t * ostride + c] = (y + cur) * (1.0f / 3.0f);
        } else {
            y = wt[0] * p[0];
            y = fmaf(wt[1], p[1], y);
            y = fmaf(wt[2], p[2], y);
            y = fmaf(wt[3], cur, y);
            out[(size_t)t * ostride + c] = (y + cur) * 0.44721359549995793f;  // 1 / sqrt(5)
        }
        p[0] = p[1];
        p[1] = p[2];
        p[2] = cur;
    }
    if (chunk == 0) {
        // the run's last three inputs (the old state's tail where the run is
        // shorter than three)
        const uint32_t n = r1 - r0;
        float ns[3];
        #pragma unroll
        for (uint32_t j = 0; j < 3; ++j) {
            const int t = (int)n - 3 + (int)j;
            ns[j] = t >= 0 ? xs[(size_t)(r0 + (uint32_t)t) * xstride + c] : st[(size_t)(t + 3) * C + c];
        }
        #pragma unroll
        for (uint32_t j = 0; j < 3; ++j) st[(size_t)j * C + c] = ns[j];
    }
}

PD_EXPORT
int pd_inkling2_sconv(const void* x, uint32_t xstride, void* out, uint32_t ostride, uint32_t C,
                      const void* w, const void* dyn, uint32_t dstride, uint32_t heads,
                      void* state, const void* run_off, const void* run_slot, uint32_t n_runs,
                      uint32_t max_run, void* stream) {
    if (n_runs == 0 || max_run == 0) return 0;
    if (dyn && (heads == 0 || C % heads)) return cudaErrorInvalidValue;
    if (x == out) return cudaErrorInvalidValue;  // the chunks read rows other chunks write
    const dim3 grid((C + 255u) / 256u, (max_run + 63u) / 64u, n_runs);
    if (dyn)
        pd_pdl_go(pd_inkling2_sconv_kernel<true>, grid, 256, 0u, (cudaStream_t)stream,
                  (const float*)x, xstride, (float*)out, ostride, C, (const __nv_bfloat16*)w,
                  (const float*)dyn, dstride, heads, (float*)state, (const uint32_t*)run_off,
                  (const uint32_t*)run_slot);
    else
        pd_pdl_go(pd_inkling2_sconv_kernel<false>, grid, 256, 0u, (cudaStream_t)stream,
                  (const float*)x, xstride, (float*)out, ostride, C, (const __nv_bfloat16*)w,
                  (const float*)nullptr, 0u, 0u, (float*)state, (const uint32_t*)run_off,
                  (const uint32_t*)run_slot);
    return pd_launch_status();
}

// slot 1004: slot 981's static-tap conv over the sum of K partial rows, onto
// a residual - the branch tails in one pass. x[t] = in[t K] + ... + in[t K +
// K - 1] folded from 0.0f in slot order (slot_combine's fold; K = 1 reads the
// row as is), y the 4-tap conv with carried state, out[t] = res[t] + (y +
// x[t]) / sqrt(5) (res null: the conv alone). Every value in the order the
// combine -> conv -> copy -> add chain it replaces computes it: bit-identical.
// A run's rows before a chunk and its last three (the new state) re-fold
// their partials.
template <uint32_t K>
__global__ void __launch_bounds__(256) pd_inkling2_sconv_sum_kernel(
        const float* in, const float* res, float* out, uint32_t C,
        const __nv_bfloat16* __restrict__ w, float* state, const uint32_t* run_off,
        const uint32_t* run_slot) {
    constexpr uint32_t T = 64u;
    PD_PDL_ARM();
    const uint32_t c = blockIdx.x * 256u + threadIdx.x;
    const uint32_t run = blockIdx.z, chunk = blockIdx.y;
    if (c >= C) return;
    const uint32_t* ro = pd_pdl_after(run_off);
    const uint32_t r0 = ro[run], r1 = ro[run + 1u];
    const uint32_t t0 = r0 + chunk * T;
    if (t0 >= r1) return;
    const uint32_t t1 = min(r1, t0 + T);
    const float* xs = pd_pdl_after(in);
    const float* rs = res ? pd_pdl_after(res) : nullptr;
    float* st = state + (size_t)pd_pdl_after(run_slot)[run] * 3u * C;
    auto xin = [&](uint32_t t) {
        const float* p = xs + (size_t)t * K * C + c;
        float acc = K == 1u ? p[0] : 0.0f;
        #pragma unroll
        for (uint32_t k = K == 1u ? 1u : 0u; k < K; ++k) acc += p[(size_t)k * C];
        return acc;
    };
    float p[3];
    #pragma unroll
    for (uint32_t j = 0; j < 3; ++j) {
        const int t = (int)t0 - 3 + (int)j;  // oldest first
        p[j] = t >= (int)r0 ? xin((uint32_t)t) : st[(size_t)(t - (int)r0 + 3) * C + c];
    }
    float wt[4];
    #pragma unroll
    for (uint32_t j = 0; j < 4; ++j) wt[j] = __bfloat162float(w[(size_t)c * 4u + j]);
    for (uint32_t t = t0; t < t1; ++t) {
        const float cur = xin(t);
        float y = wt[0] * p[0];
        y = fmaf(wt[1], p[1], y);
        y = fmaf(wt[2], p[2], y);
        y = fmaf(wt[3], cur, y);
        const float yc = (y + cur) * 0.44721359549995793f;  // 1 / sqrt(5)
        out[(size_t)t * C + c] = rs ? rs[(size_t)t * C + c] + yc : yc;
        p[0] = p[1];
        p[1] = p[2];
        p[2] = cur;
    }
    if (chunk == 0) {
        const uint32_t n = r1 - r0;
        float ns[3];
        #pragma unroll
        for (uint32_t j = 0; j < 3; ++j) {
            const int t = (int)n - 3 + (int)j;
            ns[j] = t >= 0 ? xin(r0 + (uint32_t)t) : st[(size_t)(t + 3) * C + c];
        }
        #pragma unroll
        for (uint32_t j = 0; j < 3; ++j) st[(size_t)j * C + c] = ns[j];
    }
}

PD_EXPORT
int pd_inkling2_sconv_sum(const void* in, uint32_t k, const void* res, void* out, uint32_t C,
                          const void* w, void* state, const void* run_off, const void* run_slot,
                          uint32_t n_runs, uint32_t max_run, void* stream) {
    if (n_runs == 0 || max_run == 0) return 0;
    // the chunks read rows other chunks write; res may be out (read then
    // written by the same thread)
    if (in == out || (k != 1u && k != 8u)) return cudaErrorInvalidValue;
    const dim3 grid((C + 255u) / 256u, (max_run + 63u) / 64u, n_runs);
    #define PD_IK2_SCONV_SUM(KK)                                                                   \
        pd_pdl_go(pd_inkling2_sconv_sum_kernel<KK>, grid, 256, 0u, (cudaStream_t)stream,          \
                  (const float*)in, (const float*)res, (float*)out, C, (const __nv_bfloat16*)w,    \
                  (float*)state, (const uint32_t*)run_off, (const uint32_t*)run_slot)
    if (k == 1u) PD_IK2_SCONV_SUM(1u);
    else PD_IK2_SCONV_SUM(8u);
    #undef PD_IK2_SCONV_SUM
    return pd_launch_status();
}

// slot 1014: a short conv's carried state at a cut INSIDE a run - what slot
// 981 / 1004 would have stored had the run ended there (the prefix cache's
// checkpoints land mid-chunk, the conv state is live per-slot state). Item i
// = {r0 (the run's first row in the pass), kk (rows of the run before the
// cut), state slot, out offset}: out[off + dst + j * C + c], j < 3, is the
// input at run row kk - 3 + j - folded from K partials exactly as 1004 does
// (K = 1 reads the row as is; xstride = C for partials) - or the run's OLD
// carried row where that is before the run. Launched BEFORE the conv that
// consumes x (it overwrites the old state).
template <uint32_t K>
__global__ void __launch_bounds__(256) pd_inkling2_conv_capture_kernel(
        const float* x, uint32_t xstride, uint32_t C, const float* state, const uint32_t* items,
        float* __restrict__ out, uint32_t dst) {
    PD_PDL_ARM();
    const uint32_t c = blockIdx.x * 256u + threadIdx.x;
    if (c >= C) return;
    const uint32_t* it = pd_pdl_after(items) + (size_t)blockIdx.y * 4u;
    const uint32_t r0 = it[0], kk = it[1], slot = it[2], off = it[3];
    const float* xs = pd_pdl_after(x);
    const float* st = pd_pdl_after(state) + (size_t)slot * 3u * C;
    #pragma unroll
    for (uint32_t j = 0; j < 3; ++j) {
        const int t = (int)kk - 3 + (int)j;
        float v;
        if (t >= 0) {
            const float* p = xs + (size_t)(r0 + (uint32_t)t) * K * xstride + c;
            v = K == 1u ? p[0] : 0.0f;
            #pragma unroll
            for (uint32_t q = K == 1u ? 1u : 0u; q < K; ++q) v += p[(size_t)q * C];
        } else {
            v = st[(size_t)(t + 3) * C + c];
        }
        out[(size_t)off + dst + (size_t)j * C + c] = v;
    }
}

PD_EXPORT
int pd_inkling2_conv_capture(const void* x, uint32_t k, uint32_t xstride, uint32_t C,
                             const void* state, const void* items, uint32_t n_items, void* out,
                             uint32_t dst, void* stream) {
    if (n_items == 0) return 0;
    if ((k != 1u && k != 8u) || (k == 8u && xstride != C) || xstride < C || n_items > 65535u)
        return cudaErrorInvalidValue;
    const dim3 grid((C + 255u) / 256u, n_items);
    if (k == 1u)
        pd_pdl_go(pd_inkling2_conv_capture_kernel<1u>, grid, 256, 0u, (cudaStream_t)stream,
                  (const float*)x, xstride, C, (const float*)state, (const uint32_t*)items,
                  (float*)out, dst);
    else
        pd_pdl_go(pd_inkling2_conv_capture_kernel<8u>, grid, 256, 0u, (cudaStream_t)stream,
                  (const float*)x, xstride, C, (const float*)state, (const uint32_t*)items,
                  (float*)out, dst);
    return pd_launch_status();
}

// slot 982: the experts' softcapped SwiGLU, out = 4 tanh(g / 4) * sigmoid(g)
// * 25 tanh(u / 25) (the sigmoid sees the raw gate; the caps bound the linear
// gate factor and up), caps as given.
__global__ void pd_inkling2_swiglu_cap_kernel(const float* g, const float* u,
                                              float* __restrict__ out, uint32_t n, float gc,
                                              float uc) {
    PD_PDL_ARM();
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float a = pd_pdl_after(g)[i], b = pd_pdl_after(u)[i];
    out[i] = gc * tanhf(a / gc) * (1.0f / (1.0f + expf(-a))) * (uc * tanhf(b / uc));
}

PD_EXPORT
int pd_inkling2_swiglu_cap(const void* g, const void* u, void* out, uint32_t n, float gc, float uc,
                           void* stream) {
    if (n == 0) return 0;
    pd_pdl_go(pd_inkling2_swiglu_cap_kernel, (n + 255u) / 256u, 256, 0u, (cudaStream_t)stream,
              (const float*)g, (const float*)u, (float*)out, n, gc, uc);
    return pd_launch_status();
}
