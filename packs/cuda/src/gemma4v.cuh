// gemma4v.cuh - Gemma 4's vision tower (gemma4v, the E2B/E4B and 26B/31B
// geometries; EmbeddingGemma 2's picture tower) as fused glue between its
// f16 GEMMs (slots 820-825). The unfused chain ran a row norm, an f32 -> f16
// convert, an add or a GEGLU as their own passes - a third of a picture's
// kernel time on GB10 - and its patchify, position add and pooled tail on the
// host. Here:
//
//   820 patchify   the resized u8 picture -> f16 im2row patches (the conv
//                  GEMM's input), on the device
//   821 pos_norm   x += pos_x[cx] + pos_y[cy], then ln1(x) as f16
//   822 heads      per (row, head): q, k, v RMS norms (q / k weighted, v
//                  not), the 2-D NEOX rope on q and k, f16 for the half
//                  attention entry (620)
//   827 rope table the rope's (cos, sin) per (row, pair), once a picture
//   823 post       x += rmsnorm(proj) * w_post, then the next norm as f16
//   824 geglu      gelu_tanh(gate) * up as f16 (the down GEMM's input)
//   825 pool       3x3 average pool * sqrt(embd), optional standardize,
//                  weightless RMS norm, f16 (the projection's input)
//
// BIT-IDENTICAL to the unfused chain by construction - so the Gemma 4 chat
// models' image prefills do not move: every norm is pd_rmsnorm_batch_kernel
// _t's double-float (width-stable) vectorized branch verbatim at the block
// width that kernel elects; the GEGLU and the rope are pd_geglu_kernel's and
// pd_rope2d_kernel's expressions verbatim; the f16 rounding is
// convert_f32_f16's (__float2half); and where the chain wrote an f32 result
// and a later pass added it, the add is __fadd_rn on that exact value so
// nvcc cannot contract it into an FMA. The host steps (patchify, pool) are
// replayed with explicit _rn ops in the host's order. Gated by the engine
// test that runs both paths on the same pictures.
//
// Needs abi.cuh's pd_df / pd_df_add / pd_df_merge and elementwise.cuh's
// PD_ACC_* (the double-float accumulate), so it sits after both in pack.cu.

#define PD_G4V_MAX_HD 128u   // a head is one warp's worth of float4s

// pd_rmsnorm_batch_kernel_t<PD_ACC_DF>'s vectorized reduction, verbatim:
// thread `tid` of an `nth`-wide block sums the squares of float4s tid,
// tid + nth, ... into a double-float, a shuffle-down tree folds the warp, and
// thread 0 merges the warp sums in order. Returns the row's 1 / rms.
__device__ __forceinline__ float pd_g4v_row_inv(const float4* __restrict__ xb, uint32_t n4,
                                                float eps, pd_df* wsum, float* s_inv) {
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    pd_df acc;
    acc.hi = 0.0f;
    acc.lo = 0.0f;
    for (uint32_t i = tid; i < n4; i += nth) {
        float4 v = xb[i];
        pd_df_add(acc, v.x * v.x);
        pd_df_add(acc, v.y * v.y);
        pd_df_add(acc, v.z * v.z);
        pd_df_add(acc, v.w * v.w);
    }
    for (uint32_t s = 16; s > 0; s >>= 1) {
        pd_df o;
        o.hi = __shfl_down_sync(0xffffffffu, acc.hi, s);
        o.lo = __shfl_down_sync(0xffffffffu, acc.lo, s);
        acc = pd_df_merge(acc, o);
    }
    const uint32_t warp = tid >> 5, lane = tid & 31u;
    __syncthreads();   // wsum / s_inv may still be read from a previous row norm
    if (lane == 0) wsum[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        const uint32_t nwarps = (nth + 31u) >> 5;
        pd_df sum;
        sum.hi = 0.0f;
        sum.lo = 0.0f;
        for (uint32_t wi = 0; wi < nwarps; ++wi) sum = pd_df_merge(sum, wsum[wi]);
        const double total = (double)sum.hi + (double)sum.lo;
        *s_inv = 1.0f / sqrtf((float)(total / (double)(n4 * 4u)) + eps);
    }
    __syncthreads();
    return *s_inv;
}

// ------------------------------------------------------------------ 820 patchify
// One patch a block: patch (px, py) of a [th][tw][3] u8 picture into the
// conv GEMM's row [c][ky][kx] (patches_from_rgb's order), each value
// f16(2 * (u8 / 255) - 1) - the host's division and two roundings.
__global__ void pd_g4v_patchify_kernel(const uint8_t* __restrict__ rgb, __half* __restrict__ out,
                                       uint32_t tw, uint32_t gw, uint32_t p) {
    const uint32_t patch = blockIdx.x, py = patch / gw, px = patch % gw, pp = p * p;
    __half* dst = out + (size_t)patch * 3u * pp;
    for (uint32_t e = threadIdx.x; e < 3u * pp; e += blockDim.x) {
        const uint32_t c = e / pp, r = e % pp, ky = r / p, kx = r % p;
        const uint32_t sy = py * p + ky, sx = px * p + kx;
        const float v = __fdiv_rn((float)rgb[((size_t)sy * tw + sx) * 3u + c], 255.0f);
        dst[e] = __float2half(__fsub_rn(__fmul_rn(2.0f, v), 1.0f));
    }
}

PD_EXPORT
int pd_g4v_patchify(const void* rgb, void* out, uint32_t tw, uint32_t th, uint32_t p,
                    void* stream) {
    if (p == 0 || tw % p || th % p) return (int)cudaErrorInvalidValue;
    const uint32_t gw = tw / p, gh = th / p;
    if (gw * gh == 0) return 0;
    pd_g4v_patchify_kernel<<<gw * gh, 256, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)rgb, (__half*)out, tw, gw, p);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 821 pos + ln1
// One row a block: x += (pos_x[cx] + pos_y[cy]) - the host's gathered sum,
// then the device add - then ln1(x) as f16. `pos` holds the picture's own
// rows of the table: [gw] x-rows then [gh] y-rows.
__global__ void pd_g4v_pos_norm_kernel(float* __restrict__ x, const float* __restrict__ pos,
                                       const float* __restrict__ w, __half* __restrict__ out16,
                                       uint32_t gw, uint32_t n4, float eps) {
    __shared__ pd_df wsum[32];
    __shared__ float s_inv;
    const uint32_t row = blockIdx.x, cx = row % gw, cy = row / gw, tid = threadIdx.x;
    float4* xb = reinterpret_cast<float4*>(x) + (size_t)row * n4;
    const float4* tx = reinterpret_cast<const float4*>(pos) + (size_t)cx * n4;
    const float4* ty = reinterpret_cast<const float4*>(pos) + (size_t)(gw + cy) * n4;
    for (uint32_t i = tid; i < n4; i += blockDim.x) {
        float4 v = xb[i];
        const float4 a = tx[i], b = ty[i];
        v.x = __fadd_rn(v.x, __fadd_rn(a.x, b.x));
        v.y = __fadd_rn(v.y, __fadd_rn(a.y, b.y));
        v.z = __fadd_rn(v.z, __fadd_rn(a.z, b.z));
        v.w = __fadd_rn(v.w, __fadd_rn(a.w, b.w));
        xb[i] = v;
    }
    const float inv = pd_g4v_row_inv(xb, n4, eps, wsum, &s_inv);
    const float4* w4 = reinterpret_cast<const float4*>(w);
    __half2* o = reinterpret_cast<__half2*>(out16 + (size_t)row * n4 * 4u);
    for (uint32_t i = tid; i < n4; i += blockDim.x) {
        const float4 v = xb[i], wv = w4[i];
        o[2u * i] = __halves2half2(__float2half(v.x * inv * wv.x), __float2half(v.y * inv * wv.y));
        o[2u * i + 1u] =
            __halves2half2(__float2half(v.z * inv * wv.z), __float2half(v.w * inv * wv.w));
    }
}

PD_EXPORT
int pd_g4v_pos_norm(void* x, const void* pos, const void* w, void* out16, uint32_t gw,
                    uint32_t rows, uint32_t n, float eps, void* stream) {
    if (rows == 0) return 0;
    if (gw == 0 || (n & 3u) || pd_norm_acc_mode() != PD_ACC_DF) return -2;
    const uint32_t nth = rows >= 64u ? pd_norm_wide_nth_ws(rows) : pd_norm_decode_nth();
    pd_g4v_pos_norm_kernel<<<rows, nth, 0, (cudaStream_t)stream>>>(
        (float*)x, (const float*)pos, (const float*)w, (__half*)out16, gw, n >> 2, eps);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 822 heads
// Warp lanes a (row, head). The unfused chain normed each head as its own row
// of a `nth`-wide block - with hd <= 128 every float4 lands in warp 0, lane
// = thread, so these lanes ARE that block's warp 0; the other warps' zero
// sums are merged in as that block's thread 0 merged them. Then q and k take
// the rope (pd_rope2d_kernel<true>, angles from the 827 table), and all three
// are rounded to f16 in [row][head][dim]. q, k, v are row-strided by `ld`
// (three planes or one fused landing).
//
// HW = lanes a head: 32, or 16 when hd <= 64 - two heads a warp. The
// reference tree's first step (shfl_down 16) folds lanes 16-31, which hold
// no float4 at hd <= 64, and pd_df_merge(a, 0) is a (s = a.hi, e = 0, lo =
// a.lo + 0 + 0: at most a -0 lo turns +0, which no later sum can see), so the
// tree from step 8 down over a half warp lands the same sum. GB10, 2394 rows
// x 12 heads x hd 64 (EmbeddingGemma 2's tower): 176 -> 125-132 us a layer
// (~33 MB moved: the DRAM roof), the 1152-wide tower's hd 72 (the table
// alone) 255 -> 230, both byte-identical to the one-head form
// (bench/g4v_heads_bench.cu).
template <uint32_t HW>
__global__ void pd_g4v_heads_kernel(const float* __restrict__ q, const float* __restrict__ k,
                                    const float* __restrict__ v, uint32_t ld,
                                    const float* __restrict__ qw, const float* __restrict__ kw,
                                    const float2* __restrict__ tab, __half* __restrict__ q16,
                                    __half* __restrict__ k16, __half* __restrict__ v16,
                                    uint32_t rows, uint32_t heads, uint32_t hd,
                                    uint32_t ref_warps, float eps) {
    constexpr uint32_t PER = 32u / HW;   // heads a warp
    constexpr uint32_t NP = 2u;          // rope pairs a lane: hd / 2 <= 2 HW
    __shared__ float4 buf[8][PER][PD_G4V_MAX_HD / 4u];
    const uint32_t wib = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t sub = lane / HW, hl = lane % HW;
    const uint32_t first = (blockIdx.x * 8u + wib) * PER;
    if (first >= rows * heads) return;   // the whole warp: the shuffles stay full
    const uint32_t task = first + sub;
    const bool on = task < rows * heads;
    const uint32_t t = on ? task / heads : 0u, h = on ? task % heads : 0u, n4 = hd >> 2;
    const bool live = on && hl < n4;
    float4* hb = buf[wib][sub];
    float* hf = reinterpret_cast<float*>(hb);
    const float* src[3] = {q, k, v};
    const float* wts[3] = {qw, kw, nullptr};
    __half* dst[3] = {q16, k16, v16};
    float4 val[3];
#pragma unroll
    for (uint32_t which = 0; which < 3u; ++which) {
        val[which] = make_float4(0.f, 0.f, 0.f, 0.f);
        if (live)
            val[which] = reinterpret_cast<const float4*>(src[which] + (size_t)t * ld
                                                         + (size_t)h * hd)[hl];
    }
    const uint32_t pairs = hd / 2u, quarter = hd / 4u;
    float2 ang[NP];
#pragma unroll
    for (uint32_t u = 0; u < NP; ++u) {
        const uint32_t j = hl + u * HW;
        ang[u] = (on && j < pairs) ? tab[(size_t)t * pairs + j] : make_float2(0.f, 0.f);
    }
#pragma unroll
    for (uint32_t which = 0; which < 3u; ++which) {
        const float4 x = val[which];
        pd_df acc;
        acc.hi = 0.0f;
        acc.lo = 0.0f;
        if (live) {
            pd_df_add(acc, x.x * x.x);
            pd_df_add(acc, x.y * x.y);
            pd_df_add(acc, x.z * x.z);
            pd_df_add(acc, x.w * x.w);
        }
        // a warp of the reference block that held no float4: its lanes start
        // at zero and fold through the same tree
        pd_df zero;
        zero.hi = 0.0f;
        zero.lo = 0.0f;
        for (uint32_t s = 16; s > HW / 2u; s >>= 1) zero = pd_df_merge(zero, zero);
        for (uint32_t s = HW / 2u; s > 0; s >>= 1) {
            pd_df o;
            o.hi = __shfl_down_sync(0xffffffffu, acc.hi, s, HW);
            o.lo = __shfl_down_sync(0xffffffffu, acc.lo, s, HW);
            acc = pd_df_merge(acc, o);
            zero = pd_df_merge(zero, zero);
        }
        float inv = 0.f;
        if (hl == 0u) {
            pd_df sum;
            sum.hi = 0.0f;
            sum.lo = 0.0f;
            sum = pd_df_merge(sum, acc);
            for (uint32_t wi = 1; wi < ref_warps; ++wi) sum = pd_df_merge(sum, zero);
            const double total = (double)sum.hi + (double)sum.lo;
            inv = 1.0f / sqrtf((float)(total / (double)hd) + eps);
        }
        inv = __shfl_sync(0xffffffffu, inv, lane & ~(HW - 1u));
        if (live) {
            if (wts[which]) {
                const float4 wv = reinterpret_cast<const float4*>(wts[which])[hl];
                hb[hl] = make_float4(x.x * inv * wv.x, x.y * inv * wv.y, x.z * inv * wv.z,
                                     x.w * inv * wv.w);
            } else {
                // v's norm weight is the ones row: a multiply by 1.0 is exact
                hb[hl] = make_float4(x.x * inv * 1.0f, x.y * inv * 1.0f, x.z * inv * 1.0f,
                                     x.w * inv * 1.0f);
            }
        }
        __syncwarp();
        if (which < 2u && on) {
            // pd_rope2d_kernel<true> on pair j: the x block (dims [0, hd/2))
            // or the y block. The rotation is pinned to the FMAs that kernel
            // compiles to (a * cs - b * sn -> fma(a, cs, -(b * sn)), a * sn +
            // b * cs -> fma(a, sn, b * cs)): left to nvcc, the same source
            // contracts the other way round in another surrounding
#pragma unroll
            for (uint32_t u = 0; u < NP; ++u) {
                const uint32_t j = hl + u * HW;
                if (j >= pairs) continue;
                const uint32_t i = (j < quarter) ? j : (j - quarter);
                const uint32_t base = (j < quarter) ? 0u : (hd / 2u);
                const uint32_t e0 = base + i, e1 = base + i + quarter;
                const float a = hf[e0], b = hf[e1], cs = ang[u].x, sn = ang[u].y;
                hf[e0] = __fmaf_rn(a, cs, -__fmul_rn(b, sn));
                hf[e1] = __fmaf_rn(a, sn, __fmul_rn(b, cs));
            }
        }
        __syncwarp();
        if (live) {
            const float4 r = hb[hl];
            const __half2 lo = __floats2half2_rn(r.x, r.y), hi = __floats2half2_rn(r.z, r.w);
            uint2 pk;
            pk.x = *reinterpret_cast<const uint32_t*>(&lo);
            pk.y = *reinterpret_cast<const uint32_t*>(&hi);
            reinterpret_cast<uint2*>(dst[which] + ((size_t)t * heads + h) * hd)[hl] = pk;
        }
        __syncwarp();
    }
}

PD_EXPORT
int pd_g4v_heads(const void* q, const void* k, const void* v, uint32_t ld, const void* qw,
                 const void* kw, const void* tab, void* q16, void* k16, void* v16,
                 uint32_t rows, uint32_t heads, uint32_t hd, float eps, void* stream) {
    if (rows == 0 || heads == 0) return 0;
    if (hd == 0 || (hd & 3u) || hd > PD_G4V_MAX_HD || pd_norm_acc_mode() != PD_ACC_DF)
        return -2;
    // the width the unfused norm elected for these rows - its zero warps
    const uint32_t norm_rows = rows * heads;
    const uint32_t nth =
        norm_rows >= 64u ? pd_norm_wide_nth_ws(norm_rows) : pd_norm_decode_nth();
    const uint32_t tasks = rows * heads, rw = (nth + 31u) >> 5;
    if (hd <= 64u) {
        pd_g4v_heads_kernel<16u><<<(tasks + 15u) / 16u, 256, 0, (cudaStream_t)stream>>>(
            (const float*)q, (const float*)k, (const float*)v, ld, (const float*)qw,
            (const float*)kw, (const float2*)tab, (__half*)q16, (__half*)k16, (__half*)v16,
            rows, heads, hd, rw, eps);
    } else {
        pd_g4v_heads_kernel<32u><<<(tasks + 7u) / 8u, 256, 0, (cudaStream_t)stream>>>(
            (const float*)q, (const float*)k, (const float*)v, ld, (const float*)qw,
            (const float*)kw, (const float2*)tab, (__half*)q16, (__half*)k16, (__half*)v16,
            rows, heads, hd, rw, eps);
    }
    return pd_launch_status();
}

// ------------------------------------------------------------------ 827 rope table
// pd_rope2d_kernel<true>'s angle for pair j of row t - the x block (j < hd/4,
// pos_x) or the y block (pos_y), theta by its serial multiply chain - as
// (cos, sin) in a [rows][hd/2] table. It depends on the row and the pair
// alone, so a picture builds it once (~4 us) where the chain recomputed it
// for every head of every layer: 16 x 12 x q and k on EmbeddingGemma 2's
// tower, 384 times each.
__global__ void pd_g4v_rope_table_kernel(const uint32_t* __restrict__ pos_x,
                                         const uint32_t* __restrict__ pos_y,
                                         float2* __restrict__ tab, uint32_t rows, uint32_t hd,
                                         float theta_scale) {
    const uint32_t pairs = hd / 2u, quarter = hd / 4u;
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * pairs) return;
    const uint32_t t = idx / pairs, j = idx % pairs;
    const uint32_t i = (j < quarter) ? j : (j - quarter);
    float theta = (j < quarter) ? (float)pos_x[t] : (float)pos_y[t];
    for (uint32_t kk = 0; kk < i; ++kk) theta *= theta_scale;
    const float sn = sinf(theta), cs = cosf(theta);
    tab[idx] = make_float2(cs, sn);
}

PD_EXPORT
int pd_g4v_rope_table(const void* pos_x, const void* pos_y, void* tab, uint32_t rows,
                      uint32_t hd, float theta_scale, void* stream) {
    if (rows == 0) return 0;
    if (hd == 0 || (hd & 3u) || hd > PD_G4V_MAX_HD) return (int)cudaErrorInvalidValue;
    const uint32_t n = rows * (hd / 2u);
    pd_g4v_rope_table_kernel<<<(n + 255u) / 256u, 256, 0, (cudaStream_t)stream>>>(
        (const uint32_t*)pos_x, (const uint32_t*)pos_y, (float2*)tab, rows, hd, theta_scale);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 823 post + next norm
// One row a block: x += rmsnorm(proj) * w_post (the norm's f32 result, then
// the add pass's add), then, with `wnext`, the next norm of x as f16.
__global__ void pd_g4v_post_kernel(float* __restrict__ x, const float* __restrict__ proj,
                                   const float* __restrict__ wpost,
                                   const float* __restrict__ wnext, __half* __restrict__ out16,
                                   uint32_t n4, float eps) {
    __shared__ pd_df wsum[32];
    __shared__ float s_inv;
    const uint32_t row = blockIdx.x, tid = threadIdx.x;
    const float4* pb = reinterpret_cast<const float4*>(proj) + (size_t)row * n4;
    float4* xb = reinterpret_cast<float4*>(x) + (size_t)row * n4;
    const float pinv = pd_g4v_row_inv(pb, n4, eps, wsum, &s_inv);
    const float4* wp = reinterpret_cast<const float4*>(wpost);
    for (uint32_t i = tid; i < n4; i += blockDim.x) {
        const float4 v = pb[i], wv = wp[i];
        float4 xv = xb[i];
        xv.x = __fadd_rn(xv.x, v.x * pinv * wv.x);
        xv.y = __fadd_rn(xv.y, v.y * pinv * wv.y);
        xv.z = __fadd_rn(xv.z, v.z * pinv * wv.z);
        xv.w = __fadd_rn(xv.w, v.w * pinv * wv.w);
        xb[i] = xv;
    }
    if (!wnext) return;
    const float inv = pd_g4v_row_inv(xb, n4, eps, wsum, &s_inv);
    const float4* wn = reinterpret_cast<const float4*>(wnext);
    __half2* o = reinterpret_cast<__half2*>(out16 + (size_t)row * n4 * 4u);
    for (uint32_t i = tid; i < n4; i += blockDim.x) {
        const float4 v = xb[i], wv = wn[i];
        o[2u * i] = __halves2half2(__float2half(v.x * inv * wv.x), __float2half(v.y * inv * wv.y));
        o[2u * i + 1u] =
            __halves2half2(__float2half(v.z * inv * wv.z), __float2half(v.w * inv * wv.w));
    }
}

PD_EXPORT
int pd_g4v_post(void* x, const void* proj, const void* wpost, const void* wnext, void* out16,
                uint32_t rows, uint32_t n, float eps, void* stream) {
    if (rows == 0) return 0;
    if ((n & 3u) || pd_norm_acc_mode() != PD_ACC_DF || (wnext && !out16)) return -2;
    const uint32_t nth = rows >= 64u ? pd_norm_wide_nth_ws(rows) : pd_norm_decode_nth();
    pd_g4v_post_kernel<<<rows, nth, 0, (cudaStream_t)stream>>>(
        (float*)x, (const float*)proj, (const float*)wpost, (const float*)wnext,
        (__half*)out16, n >> 2, eps);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 824 GEGLU
// gelu_tanh(gate) * up -> f16, pd_geglu_kernel's expression verbatim. gate
// and up are row-strided by `ld` (two planes, or the halves of one landing);
// with `relaid`, `gate` is one landing of a weight re-laid in 16-row blocks
// (gate features 8b..8b+7 at columns 16b.., their ups 8 later - slot 826's
// layout) and `up` is ignored: the reference the 826 epilogue is held to.
__global__ void pd_g4v_geglu_kernel(const float* __restrict__ gate, const float* __restrict__ up,
                                    uint32_t ld, __half* __restrict__ out, uint32_t ffn,
                                    uint64_t total, uint32_t relaid) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const uint64_t r = i / ffn, j = i % ffn;
    const uint64_t gc = relaid ? (j >> 3) * 16u + (j & 7u) : j;
    const float* upp = relaid ? gate + 8u : up;
    float g = gate[r * ld + gc];
    float gelu = 0.5f * g * (1.0f + tanhf(0.79788456080286535587989211986876f * g
                                          * (1.0f + 0.044715f * g * g)));
    out[i] = __float2half(gelu * upp[r * ld + gc]);
}

PD_EXPORT
int pd_g4v_geglu(const void* gate, const void* up, uint32_t ld, void* out, uint32_t ffn,
                 uint32_t rows, uint32_t relaid, void* stream) {
    const uint64_t total = (uint64_t)rows * ffn;
    if (total == 0) return 0;
    if (relaid && (ffn & 7u)) return (int)cudaErrorInvalidValue;
    pd_g4v_geglu_kernel<<<(uint32_t)((total + 255u) / 256u), 256, 0, (cudaStream_t)stream>>>(
        (const float*)gate, (const float*)up, ld, (__half*)out, ffn, total, relaid);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 825 pool
// One output token a block - the host tail replayed in its order: a 3x3 sum
// from 0 (row-major over the window), times `inv` (sqrt(embd) / 9, the
// caller's f32), optionally (h - bias) * scale, then a weightless RMS norm
// whose sum of squares is the host's serial left fold - one thread, element
// order - and f16 for the projection. Every op explicitly rounded: the host
// had no FMA.
__global__ void pd_g4v_pool_kernel(const float* __restrict__ x, const float* __restrict__ bias,
                                   const float* __restrict__ scale, __half* __restrict__ out,
                                   uint32_t gw, uint32_t ow, uint32_t embd, float inv, float eps) {
    extern __shared__ float pd_g4v_d[];
    __shared__ float s_r;
    const uint32_t tok = blockIdx.x, oy = tok / ow, ox = tok % ow;
    for (uint32_t e = threadIdx.x; e < embd; e += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t ky = 0; ky < 3u; ++ky)
            for (uint32_t kx = 0; kx < 3u; ++kx)
                acc = __fadd_rn(acc, x[((size_t)(oy * 3u + ky) * gw + (ox * 3u + kx)) * embd + e]);
        acc = __fmul_rn(acc, inv);
        if (bias) acc = __fmul_rn(__fsub_rn(acc, bias[e]), scale[e]);
        pd_g4v_d[e] = acc;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float ms = 0.0f;
        for (uint32_t e = 0; e < embd; ++e) ms = __fadd_rn(ms, __fmul_rn(pd_g4v_d[e], pd_g4v_d[e]));
        ms = __fdiv_rn(ms, (float)embd);
        s_r = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(ms, eps)));
    }
    __syncthreads();
    const float r = s_r;
    for (uint32_t e = threadIdx.x; e < embd; e += blockDim.x)
        out[(size_t)tok * embd + e] = __float2half(__fmul_rn(pd_g4v_d[e], r));
}

PD_EXPORT
int pd_g4v_pool(const void* x, const void* bias, const void* scale, void* out, uint32_t gw,
                uint32_t gh, uint32_t embd, float inv, float eps, void* stream) {
    const uint32_t ow = gw / 3u, oh = gh / 3u;
    if (ow * oh == 0) return 0;
    if (gw % 3u || gh % 3u || (bias == nullptr) != (scale == nullptr) ||
        (size_t)embd * sizeof(float) > 48u * 1024u)
        return (int)cudaErrorInvalidValue;
    pd_g4v_pool_kernel<<<ow * oh, 256, embd * sizeof(float), (cudaStream_t)stream>>>(
        (const float*)x, (const float*)bias, (const float*)scale, (__half*)out, gw, ow, embd,
        inv, eps);
    return pd_launch_status();
}
