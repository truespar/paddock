// sam3/detector.cuh - SAM 3's detector glue: residual/LayerNorm seams, the box-biased decoder attention and its tables, box embeddings, refinement, RoIAlign, the scorer
// Textually-included segment of the single pack translation unit.
// Not standalone-compilable: include order is defined by ../pack.cu.
// ------------------------------------------------------------ sam3 detector
// The detector (Meta's geometry encoder, fusion encoder, DETR decoder and
// dot-product scorer) is d = 256 everywhere, 8 heads of 32. Its GEMMs are the
// f16 tensor-core ones and its plain attention is pd_vision_attn_h (prompt
// keys compacted to the valid ones, which is what a key-padding mask computes).
// What is here is everything between those that no other tower needed.
//
// Precision is the image encoder's class: f32 residual, norms and box math,
// f16 GEMM operands. Measured on Meta's fp32 run, the largest activation any
// detector module lands is 139 (the box-bias MLP), so f16 is never the limit.
// Meta runs the decoder FFN with autocast off - but with TF32 allowed, so its
// inputs keep 10 mantissa bits; the f16 class keeps 11.
//
// Needs dense_pred.cuh (the GroupNorm reduce/fold kernels) and f32_qkv.cuh's
// pd_launch_status.

// One block reduces a row of n <= 1024 floats in a fixed order: per-thread
// strided partials, then a shared tree. Two passes (mean, centred squares),
// like every norm in this pack.
__device__ __forceinline__ float pd_sam3_block_sum(float v, float* red) {
    const uint32_t t = threadIdx.x;
    red[t] = v;
    __syncthreads();
    for (uint32_t s = blockDim.x >> 1; s > 0u; s >>= 1) {
        if (t < s) red[t] += red[t + s];
        __syncthreads();
    }
    const float r = red[0];
    __syncthreads();
    return r;
}

#define PD_SAM3_SEAM_POST 1u   // x = LN(x + proj + bias), else x += proj + bias; y = LN(x)
#define PD_SAM3_SEAM_NOLN 2u   // no norm at all: y = x (+ proj + bias)

// 760: every residual seam of the detector in one launch a row:
//   x' = x + proj + bias           (proj f16, bias f32, both optional)
//   pre-norm:  x <- x';  y = LN(x')        post-norm: x <- y = LN(x')
//   out = f16(y);  outq = f16(y + pos[t]) for the row's position t
// `pos` is a [npos][n] f32 table read at t = row % period; rows with
// t >= npos get no position (the decoder's presence token, which rides as
// its last row). n <= 1024, one block a row, blockDim a power of two >= n.
__global__ void pd_sam3_seam_h_kernel(float* __restrict__ x, const __half* __restrict__ proj,
                                      const float* __restrict__ bias,
                                      const float* __restrict__ w, const float* __restrict__ b,
                                      const float* __restrict__ pos, __half* __restrict__ out,
                                      __half* __restrict__ outq, uint32_t n, uint32_t period,
                                      uint32_t npos, float eps, uint32_t flags) {
    __shared__ float red[1024];
    const uint32_t r = blockIdx.x, i = threadIdx.x;
    const bool on = i < n;
    float* xr = x + (size_t)r * n;
    float v = 0.0f;
    if (on) {
        v = xr[i];
        if (proj != nullptr) v += __half2float(proj[(size_t)r * n + i]);
        if (bias != nullptr) v += bias[i];
    }
    float y = v;
    if (!(flags & PD_SAM3_SEAM_NOLN)) {
        const float mean = pd_sam3_block_sum(on ? v : 0.0f, red) / (float)n;
        const float dv = on ? v - mean : 0.0f;
        const float var = pd_sam3_block_sum(dv * dv, red) / (float)n;
        y = on ? dv * rsqrtf(var + eps) * w[i] + b[i] : 0.0f;
    }
    if (!on) return;
    xr[i] = (flags & PD_SAM3_SEAM_POST) ? y : v;
    if (out != nullptr) out[(size_t)r * n + i] = __float2half(y);
    if (outq != nullptr) {
        const uint32_t t = r % period;
        const float p = (pos != nullptr && t < npos) ? pos[(size_t)t * n + i] : 0.0f;
        outq[(size_t)r * n + i] = __float2half(y + p);
    }
}

PD_EXPORT
int pd_sam3_seam_h(void* x, const void* proj, const void* bias, const void* w, const void* b,
                   const void* pos, void* out, void* outq, uint32_t rows, uint32_t n,
                   uint32_t period, uint32_t npos, float eps, uint32_t flags, void* stream) {
    if (rows == 0 || n == 0) return 0;
    if (n > 1024u || period == 0) return cudaErrorInvalidValue;
    if (!(flags & PD_SAM3_SEAM_NOLN) && (w == nullptr || b == nullptr)) return cudaErrorInvalidValue;
    uint32_t nth = 32u;
    while (nth < n) nth <<= 1;
    pd_sam3_seam_h_kernel<<<rows, nth, 0, (cudaStream_t)stream>>>(
        (float*)x, (const __half*)proj, (const float*)bias, (const float*)w, (const float*)b,
        (const float*)pos, (__half*)out, (__half*)outq, n, period, npos, eps, flags);
    return pd_launch_status();
}

// 761: the decoder's image cross-attention, with Meta's box relative-position
// bias ("boxRPB", log mode) added to the scaled scores:
//   s[q][h][y*gw + x] = q.k + by[q][y][h] + bx[q][x][h]
// The 8 x 200 x 5184 bias Meta materializes is a sum of two per-axis tables,
// so it is never built here - each score reads its two table entries. Only
// the first `nbias` query rows are biased; the presence token rides as the
// last row and gets none, as Meta pads its row with zeros. q is PRE-SCALED (the loader folds 1/sqrt(hd) into the q
// projection), all planes f16 [rows][H][hd].
//
// One block a (query, head): scores into shared memory, a fixed-order block
// max and sum, then the weighted V in 8 key lanes x hd dims, folded in order.
// The interim form - SIMT, K and V re-read from L2 per query - is ~200 x 5184
// x 8 heads x 32 a layer, under a millisecond on this die; the tensor-core
// tile with the bias added to its score fragments is the target once the
// decoder is a measurable share of a frame.
#define PD_SAM3_BOX_MAXK 5184u
#define PD_SAM3_BOX_MAXD 64u
__global__ void __launch_bounds__(256) pd_sam3_box_attn_h_kernel(
    const __half* __restrict__ q, const __half* __restrict__ k, const __half* __restrict__ v,
    const float* __restrict__ bx, const float* __restrict__ by, __half* __restrict__ out,
    uint32_t nk, uint32_t H, uint32_t hd, uint32_t gh, uint32_t gw, uint32_t nbias) {
    __shared__ float sc[PD_SAM3_BOX_MAXK];
    __shared__ float sq[PD_SAM3_BOX_MAXD];
    __shared__ float red[256];
    __shared__ float acc8[8 * PD_SAM3_BOX_MAXD];
    const uint32_t qi = blockIdx.x, h = blockIdx.y, t = threadIdx.x;
    const size_t rs = (size_t)H * hd;
    for (uint32_t e = t; e < hd; e += blockDim.x) sq[e] = __half2float(q[qi * rs + h * hd + e]);
    __syncthreads();
    const bool biased = qi < nbias;
    const float* bxq = biased ? bx + (size_t)qi * gw * H : nullptr;
    const float* byq = biased ? by + (size_t)qi * gh * H : nullptr;
    float m = -INFINITY;
    for (uint32_t j = t; j < nk; j += blockDim.x) {
        const __half* kr = k + j * rs + h * hd;
        float s = 0.0f;
        for (uint32_t e = 0; e < hd; ++e) s += sq[e] * __half2float(kr[e]);
        if (biased) {
            const uint32_t yy = j / gw, xx = j - yy * gw;
            s += byq[yy * H + h] + bxq[xx * H + h];
        }
        sc[j] = s;
        m = fmaxf(m, s);
    }
    red[t] = m;
    __syncthreads();
    for (uint32_t s = 128u; s > 0u; s >>= 1) {
        if (t < s) red[t] = fmaxf(red[t], red[t + s]);
        __syncthreads();
    }
    m = red[0];
    __syncthreads();
    float l = 0.0f;
    for (uint32_t j = t; j < nk; j += blockDim.x) {
        const float p = expf(sc[j] - m);
        sc[j] = p;
        l += p;
    }
    l = pd_sam3_block_sum(l, red);
    // 8 key lanes x hd dims: lane g sums keys g, g+8, ... for dim e, in order
    const uint32_t g = t / hd, e = t - g * hd;
    if (t < 8u * hd) {
        float a = 0.0f;
        for (uint32_t j = g; j < nk; j += 8u) a += sc[j] * __half2float(v[j * rs + h * hd + e]);
        acc8[g * hd + e] = a;
    }
    __syncthreads();
    if (t < hd) {
        float a = 0.0f;
        for (uint32_t gg = 0; gg < 8u; ++gg) a += acc8[gg * hd + t];
        out[qi * rs + h * hd + t] = __float2half(a / l);
    }
}

PD_EXPORT
int pd_sam3_box_attn_h(const void* q, const void* k, const void* v, const void* bx,
                       const void* by, void* out, uint32_t nq, uint32_t nk, uint32_t H,
                       uint32_t hd, uint32_t gh, uint32_t gw, uint32_t nbias, void* stream) {
    if (nq == 0 || H == 0) return 0;
    if (nk == 0 || nk > PD_SAM3_BOX_MAXK || nk != gh * gw || hd == 0 || hd > PD_SAM3_BOX_MAXD ||
        8u * hd > 256u)
        return cudaErrorInvalidValue;
    dim3 grid(nq, H);
    pd_sam3_box_attn_h_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(
        (const __half*)q, (const __half*)k, (const __half*)v, (const float*)bx,
        (const float*)by, (__half*)out, nk, H, hd, gh, gw, nbias);
    return pd_launch_status();
}

// 774: slot 761 on the tensor cores - the same scores, bias and softmax.
// The tile is vision_attn_mma's (16 query rows a warp, K/V tiles through
// ldmatrix, m16n8k16 f16 -> f32, P rounded to f16 for the PV product) with
// the box bias added to the score fragments. A key tile is TWO grid rows
// (2 * gw keys), so every tile puts a lane's columns at the same x: its bx
// entries load into registers once, and only its two by entries change from
// tile to tile. One (query block, head) on its own is 36 tiles of serial work
// on a 32-block grid, so the row pairs are split across grid.z and a second
// pass folds the splits in a fixed order.
//   part: [nsplit][nq][H][hd + 2] f32 - per split (max, sum, unnormalized o)
// With one split there is nothing to fold: the kernel normalizes and stores
// the halves itself (part is null). Rows past nq compute on zeros and store
// nothing. With nbias = 0 it is plain cross- or self-attention over the
// memory, which is how the geometry and fusion encoders use it.
#define PD_SAM3_BXM_QW 4u
template <uint32_t DP, uint32_t GW>
__global__ void __launch_bounds__(32u * PD_SAM3_BXM_QW) pd_sam3_box_attn_mma_kernel(
    const __half* __restrict__ q, const __half* __restrict__ k, const __half* __restrict__ v,
    const float* __restrict__ bx, const float* __restrict__ by, float* __restrict__ part,
    __half* __restrict__ out, uint32_t nq, uint32_t gh, uint32_t H, uint32_t nbias,
    uint32_t per) {
#if PD_FA_OK
    constexpr uint32_t QW = PD_SAM3_BXM_QW, KT = 2u * GW, NX = GW / 8u, DPD = DP + 8u;
    constexpr uint32_t NT = 32u * QW, SPAN = KT * (DP / 8u), REC = DP + 2u;
    static_assert(GW % 8u == 0u && DP % 16u == 0u, "a grid row is whole n-tiles");
    const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
    const uint32_t g8 = lane >> 2, t4 = lane & 3u, lg = lane >> 3;
    const uint32_t h = blockIdx.y, sp = blockIdx.z;
    const uint32_t row0 = blockIdx.x * (QW * 16u) + warp * 16u;
    const size_t rs = (size_t)H * DP;
    // +8 halves of row pad, as vision_attn_mma: an ldmatrix's 8 rows land in
    // distinct bank groups
    __shared__ __align__(16) __half sh_k[KT * DPD];
    __shared__ __align__(16) __half sh_v[KT * DPD];

    // the lane's two rows (mma layout: lane/4 and +8), their q fragments and
    // their bx entries at the lane's columns x = 8 xt + 2 t4 + {0, 1}
    const uint32_t rw[2] = {row0 + g8, row0 + g8 + 8u};
    bool biased[2];
    uint32_t qa[DP / 16u][4];
    float bxr[NX][4];
#pragma unroll
    for (uint32_t r = 0; r < 2u; ++r) {
        const uint32_t b = rw[r];
        const bool ok = b < nq;
        biased[r] = ok && b < nbias;
        const __half* qp = q + (size_t)(ok ? b : nq - 1u) * rs + (size_t)h * DP;
#pragma unroll
        for (uint32_t d0 = 0; d0 < DP / 16u; ++d0) {
            const uint32_t c0 = d0 * 16u + 2u * t4;
            const uint32_t lo = *reinterpret_cast<const uint32_t*>(qp + c0);
            const uint32_t hi = *reinterpret_cast<const uint32_t*>(qp + c0 + 8u);
            qa[d0][r] = ok ? lo : 0u;
            qa[d0][r + 2u] = ok ? hi : 0u;
        }
        const float* bxq = bx + (size_t)(biased[r] ? b : 0u) * GW * H + h;
#pragma unroll
        for (uint32_t xt = 0; xt < NX; ++xt)
#pragma unroll
            for (uint32_t e1 = 0; e1 < 2u; ++e1)
                bxr[xt][r * 2u + e1] =
                    biased[r] ? bxq[(size_t)(xt * 8u + 2u * t4 + e1) * H] : 0.f;
    }

    float m_st[2] = {-1e30f, -1e30f}, l_st[2] = {0.f, 0.f};
    float o_acc[DP / 8u][4];
#pragma unroll
    for (uint32_t nt = 0; nt < DP / 8u; ++nt)
#pragma unroll
        for (uint32_t e = 0; e < 4u; ++e) o_acc[nt][e] = 0.f;

    const uint32_t pairs = gh / 2u;
    const uint32_t tb = sp * per, te = tb + per < pairs ? tb + per : pairs;
    for (uint32_t tp = tb; tp < te; ++tp) {
        // K then V for grid rows 2 tp and 2 tp + 1 - always whole, nk is gh * gw
        const size_t k0 = (size_t)tp * KT;
        for (uint32_t u = tid; u < 2u * SPAN; u += NT) {
            const bool isv = u >= SPAN;
            const uint32_t ur = isv ? u - SPAN : u;
            const uint32_t kk = ur / (DP / 8u), d8 = (ur % (DP / 8u)) * 8u;
            const __half* src = (isv ? v : k) + (k0 + kk) * rs + (size_t)h * DP + d8;
            *reinterpret_cast<uint4*>((isv ? sh_v : sh_k) + kk * DPD + d8) =
                *reinterpret_cast<const uint4*>(src);
        }
        float byr[2][2];
#pragma unroll
        for (uint32_t r = 0; r < 2u; ++r)
#pragma unroll
            for (uint32_t yy = 0; yy < 2u; ++yy)
                byr[r][yy] = biased[r]
                                 ? by[((size_t)rw[r] * gh + 2u * tp + yy) * H + h]
                                 : 0.f;
        __syncthreads();

        float s_acc[KT / 8u][4];
#pragma unroll
        for (uint32_t nt = 0; nt < KT / 8u; ++nt)
#pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) s_acc[nt][e] = 0.f;
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
                pd_fa_mma16(s_acc[np * 2u], qa[d0][0], qa[d0][1], qa[d0][2], qa[d0][3],
                            kb4[0], kb4[1]);
                pd_fa_mma16(s_acc[np * 2u + 1u], qa[d0][0], qa[d0][1], qa[d0][2], qa[d0][3],
                            kb4[2], kb4[3]);
            }
        }
        // the bias, as 761 adds it (s + (by + bx)), then the online softmax:
        // a C-fragment row sits across the four lanes of a lane/4 group
        float mn[2] = {m_st[0], m_st[1]};
#pragma unroll
        for (uint32_t nt = 0; nt < KT / 8u; ++nt) {
            const uint32_t yy = nt / NX, xt = nt - yy * NX;
#pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) {
                s_acc[nt][e] += byr[e >> 1][yy] + bxr[xt][e];
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
        // PV: A = the weights straight out of the score fragments, B = V as
        // [k=key][n=dim] through ldmatrix.trans
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
        __syncthreads();  // tiles read before the next stage overwrites them
    }

    if (part == nullptr) {
        // the one split: normalize and store
#pragma unroll
        for (uint32_t r = 0; r < 2u; ++r) {
            if (rw[r] >= nq) continue;
            const float inv = l_st[r] > 0.f ? 1.f / l_st[r] : 0.f;
            __half* op = out + (size_t)rw[r] * rs + (size_t)h * DP;
#pragma unroll
            for (uint32_t nt = 0; nt < DP / 8u; ++nt)
                *reinterpret_cast<__half2*>(op + nt * 8u + 2u * t4) =
                    __floats2half2_rn(o_acc[nt][2u * r] * inv, o_acc[nt][2u * r + 1u] * inv);
        }
        return;
    }
    // this split's (max, sum, unnormalized o); an empty split says (-1e30, 0, 0)
#pragma unroll
    for (uint32_t r = 0; r < 2u; ++r) {
        if (rw[r] >= nq) continue;
        float* rec = part + (((size_t)sp * nq + rw[r]) * H + h) * REC;
        if (t4 == 0u) {
            rec[0] = m_st[r];
            rec[1] = l_st[r];
        }
#pragma unroll
        for (uint32_t nt = 0; nt < DP / 8u; ++nt) {
            const uint32_t c = nt * 8u + 2u * t4;
            rec[2u + c] = o_acc[nt][2u * r];
            rec[3u + c] = o_acc[nt][2u * r + 1u];
        }
    }
#else
    (void)q; (void)k; (void)v; (void)bx; (void)by; (void)part; (void)out; (void)nq; (void)gh;
    (void)H; (void)nbias; (void)per;
#endif
}

// the splits folded in order: out = sum_s o_s e^(m_s - M) / sum_s l_s e^(m_s - M)
__global__ void pd_sam3_box_fold_kernel(const float* __restrict__ part, __half* __restrict__ out,
                                        uint32_t nq, uint32_t H, uint32_t hd, uint32_t nsplit) {
    const uint32_t b = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const uint32_t rec = hd + 2u;
    const size_t stride = (size_t)nq * H * rec;
    const float* r0 = part + ((size_t)b * H + h) * rec;
    float M = -1e30f;
    for (uint32_t s = 0; s < nsplit; ++s) M = fmaxf(M, r0[s * stride]);
    float L = 0.f, O = 0.f;
    for (uint32_t s = 0; s < nsplit; ++s) {
        const float* r = r0 + s * stride;
        const float w = expf(r[0] - M);
        L += r[1] * w;
        O += r[2u + d] * w;
    }
    out[((size_t)b * H + h) * hd + d] = __float2half(L > 0.f ? O / L : 0.f);
}

PD_EXPORT
int pd_sam3_box_attn_mma(const void* q, const void* k, const void* v, const void* bx,
                         const void* by, void* part, void* out, uint32_t nq, uint32_t H,
                         uint32_t hd, uint32_t gh, uint32_t gw, uint32_t nbias, uint32_t nsplit,
                         void* stream) {
    if (nq == 0 || H == 0) return 0;
    // the one instantiated shape: SAM 3's 72 x 72 memory, 32-wide heads
    if (hd != 32u || gw != 72u || gh == 0u || (gh & 1u) != 0u || nsplit == 0u ||
        nsplit > gh / 2u)
        return cudaErrorInvalidValue;
    int dev = 0, cc = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&cc, cudaDevAttrComputeCapabilityMajor, dev);
    if (cc < 8) return cudaErrorInvalidValue;
    const cudaStream_t st = (cudaStream_t)stream;
    const uint32_t per = (gh / 2u + nsplit - 1u) / nsplit;
    const uint32_t qt = PD_SAM3_BXM_QW * 16u;
    pd_sam3_box_attn_mma_kernel<32u, 72u><<<dim3((nq + qt - 1u) / qt, H, nsplit),
                                            32u * PD_SAM3_BXM_QW, 0, st>>>(
        (const __half*)q, (const __half*)k, (const __half*)v, (const float*)bx,
        (const float*)by, nsplit == 1u ? nullptr : (float*)part, (__half*)out, nq, gh, H, nbias,
        per);
    if (nsplit > 1u)
        pd_sam3_box_fold_kernel<<<dim3(nq, H), hd, 0, st>>>((const float*)part, (__half*)out, nq,
                                                            H, hd, nsplit);
    return pd_launch_status();
}

// 762: the two box-bias tables for the current reference boxes, Meta's
// `_get_rpb_matrix` in "log" mode, both axes in one launch. For query q with
// box (cx, cy, w, h) -> xyxy and grid coordinate c = i / n (top-left corners,
// torch.arange(0, n) / n in f32):
//   d = (c - lo, c - hi);  d' = sign(8d) * log2(|8d| + 1) / 3
//   table[q][i][:] = W2 relu(W1 d' + b1) + b2       (2 -> hidden -> heads)
// One thread a (query, i), grid.y the axis; the MLP runs in f32 in a fixed
// order (Meta's runs under bf16 autocast). Weights are [out][in] f32 as Linear
// stores them; the block stages its axis's MLP in shared memory first, W2
// transposed so a hidden unit's H outputs sit together - every read in the
// loop is then a broadcast. Reading them from global through L1 instead cost
// ~200 us a layer on the A6000, more than the attention it feeds.
__global__ void pd_sam3_rpb_tables_kernel(const float* __restrict__ ref,
                                          const float* __restrict__ w1x,
                                          const float* __restrict__ b1x,
                                          const float* __restrict__ w2x,
                                          const float* __restrict__ b2x,
                                          const float* __restrict__ w1y,
                                          const float* __restrict__ b1y,
                                          const float* __restrict__ w2y,
                                          const float* __restrict__ b2y, float* __restrict__ tx,
                                          float* __restrict__ ty, uint32_t nq, uint32_t gh,
                                          uint32_t gw, uint32_t hid, uint32_t H) {
    extern __shared__ float rpb_sh[];
    const bool ax = blockIdx.y == 0;  // x axis first
    const float* w1 = ax ? w1x : w1y;
    const float* b1 = ax ? b1x : b1y;
    const float* w2 = ax ? w2x : w2y;
    const float* b2 = ax ? b2x : b2y;
    float* s_w1 = rpb_sh;              // [hid][2]
    float* s_b1 = s_w1 + 2u * hid;     // [hid]
    float* s_w2 = s_b1 + hid;          // [hid][H], transposed
    for (uint32_t i = threadIdx.x; i < 2u * hid; i += blockDim.x) s_w1[i] = w1[i];
    for (uint32_t i = threadIdx.x; i < hid; i += blockDim.x) s_b1[i] = b1[i];
    for (uint32_t i = threadIdx.x; i < hid * H; i += blockDim.x) {
        const uint32_t hh = i / hid, j = i - hh * hid;
        s_w2[j * H + hh] = w2[i];
    }
    __syncthreads();
    const uint32_t n = ax ? gw : gh;
    const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= nq * n) return;
    const uint32_t qi = id / n, i = id - qi * n;
    const float* bxr = ref + (size_t)qi * 4u;
    const float c = ax ? bxr[0] : bxr[1], s = ax ? bxr[2] : bxr[3];
    const float lo = c - 0.5f * s, hi = c + 0.5f * s;
    const float coord = (float)i / (float)n;
    float d[2] = {coord - lo, coord - hi};
    for (uint32_t u = 0; u < 2u; ++u) {
        const float e = d[u] * 8.0f;
        const float sg = e > 0.0f ? 1.0f : (e < 0.0f ? -1.0f : 0.0f);
        d[u] = sg * log2f(fabsf(e) + 1.0f) / 3.0f;
    }
    float* o = (ax ? tx + ((size_t)qi * gw + i) * H : ty + ((size_t)qi * gh + i) * H);
    // the head loops run to the 16 the launcher allows, guarded, so acc stays
    // in registers - a runtime bound put it on the stack (64 B of local
    // memory, every FMA a load and a store)
    float acc[16];
#pragma unroll
    for (uint32_t hh = 0; hh < 16u; ++hh) acc[hh] = hh < H ? b2[hh] : 0.0f;
    for (uint32_t j = 0; j < hid; ++j) {
        float a = s_w1[j * 2u] * d[0] + s_w1[j * 2u + 1u] * d[1] + s_b1[j];
        a = fmaxf(a, 0.0f);
        const float* wr = s_w2 + j * H;
#pragma unroll
        for (uint32_t hh = 0; hh < 16u; ++hh)
            if (hh < H) acc[hh] += wr[hh] * a;
    }
#pragma unroll
    for (uint32_t hh = 0; hh < 16u; ++hh)
        if (hh < H) o[hh] = acc[hh];
}

PD_EXPORT
int pd_sam3_rpb_tables(const void* ref, const void* w1x, const void* b1x, const void* w2x,
                       const void* b2x, const void* w1y, const void* b1y, const void* w2y,
                       const void* b2y, void* tx, void* ty, uint32_t nq, uint32_t gh,
                       uint32_t gw, uint32_t hid, uint32_t H, void* stream) {
    if (nq == 0) return 0;
    if (H == 0 || H > 16u || hid == 0) return cudaErrorInvalidValue;
    const size_t smem = (size_t)hid * (3u + H) * sizeof(float);
    if (smem > 48u * 1024u) return cudaErrorInvalidValue;
    const uint32_t n = nq * (gw > gh ? gw : gh);
    pd_sam3_rpb_tables_kernel<<<dim3((n + 127u) / 128u, 2), 128, smem, (cudaStream_t)stream>>>(
        (const float*)ref, (const float*)w1x, (const float*)b1x, (const float*)w2x,
        (const float*)b2x, (const float*)w1y, (const float*)b1y, (const float*)w2y,
        (const float*)b2y, (float*)tx, (float*)ty, nq, gh, gw, hid, H);
    return pd_launch_status();
}

// 763: sine embeddings of boxes, f16 out (they feed a GEMM). Per coordinate
// value u, `npf` features: u * 2pi / T^(2 floor(k/2) / npf), sin on even k,
// cos on odd - Meta's `gen_sineembed_for_position` and
// `PositionEmbeddingSine._encode_xy`, f32 throughout.
//   mode 0 (decoder query position): boxes cxcywh -> [y | x | w | h] sines, 4*npf
//   mode 1 (geometry box encoding):  [y | x] sines then the raw h, w: 2*npf + 2
// Row stride `ld` >= the width, the tail zero (the GEMM's 8-wide K pad).
__global__ void pd_sam3_box_sine_kernel(const float* __restrict__ boxes, __half* __restrict__ out,
                                        uint32_t n, uint32_t npf, uint32_t mode, uint32_t ld,
                                        float temperature) {
    const uint32_t r = blockIdx.x;
    if (r >= n) return;
    const float* bxr = boxes + (size_t)r * 4u;
    // Meta's order: y first, then x, then (mode 0) w, h
    const float val[4] = {bxr[1], bxr[0], bxr[2], bxr[3]};
    const uint32_t nsin = mode == 0u ? 4u : 2u;
    const float scale = 6.283185307179586f;  // 2 * math.pi, in f32
    for (uint32_t c = threadIdx.x; c < ld; c += blockDim.x) {
        float o = 0.0f;
        if (c < nsin * npf) {
            const uint32_t a = c / npf, kk = c - a * npf;
            const float dim_t = powf(temperature, (float)(2u * (kk / 2u)) / (float)npf);
            const float u = (val[a] * scale) / dim_t;
            o = (kk & 1u) ? cosf(u) : sinf(u);
        } else if (mode == 1u && c == 2u * npf) {
            o = bxr[3];  // h
        } else if (mode == 1u && c == 2u * npf + 1u) {
            o = bxr[2];  // w
        }
        out[(size_t)r * ld + c] = __float2half(o);
    }
}

PD_EXPORT
int pd_sam3_box_sine(const void* boxes, void* out, uint32_t n, uint32_t npf, uint32_t mode,
                     uint32_t ld, float temperature, void* stream) {
    if (n == 0) return 0;
    const uint32_t width = mode == 0u ? 4u * npf : 2u * npf + 2u;
    if (npf == 0 || mode > 1u || ld < width) return cudaErrorInvalidValue;
    pd_sam3_box_sine_kernel<<<n, 256, 0, (cudaStream_t)stream>>>(
        (const float*)boxes, (__half*)out, n, npf, mode, ld, temperature);
    return pd_launch_status();
}

// 764: box refinement, in place: ref = sigmoid(delta + bias + inverse_sigmoid(ref))
// with Meta's inverse_sigmoid (clamp to [0, 1], then log(max(x, 1e-3) /
// max(1 - x, 1e-3))). `delta` rows are `ld` floats apart (the box head's
// landing), its first 4 the cxcywh offsets.
__global__ void pd_sam3_box_refine_kernel(float* __restrict__ ref, const float* __restrict__ delta,
                                          const float* __restrict__ bias, uint32_t n,
                                          uint32_t ld) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * 4u) return;
    const uint32_t r = i >> 2, c = i & 3u;
    float x = fminf(fmaxf(ref[i], 0.0f), 1.0f);
    const float x1 = fmaxf(x, 1e-3f), x2 = fmaxf(1.0f - x, 1e-3f);
    const float inv = logf(x1 / x2);
    const float u = delta[(size_t)r * ld + c] + (bias != nullptr ? bias[c] : 0.0f) + inv;
    ref[i] = 1.0f / (1.0f + expf(-u));
}

PD_EXPORT
int pd_sam3_box_refine(void* ref, const void* delta, const void* bias, uint32_t n, uint32_t ld,
                       void* stream) {
    if (n == 0) return 0;
    if (ld < 4u) return cudaErrorInvalidValue;
    pd_sam3_box_refine_kernel<<<(n * 4u + 127u) / 128u, 128, 0, (cudaStream_t)stream>>>(
        (float*)ref, (const float*)delta, (const float*)bias, n, ld);
    return pd_launch_status();
}

// 765: torchvision.ops.roi_align as SAM 3's geometry encoder calls it -
// aligned=False (the legacy half-pixel convention), spatial_scale 1,
// sampling_ratio -1 (adaptive: ceil(bin size) samples a bin side), output
// `pooled` x `pooled`. Features are an NHWC f32 raster [H][W][C]; boxes xyxy
// in feature pixels; the output [n][C][pooled][pooled] f16 is the box conv's
// GEMM input (its weight [out][c][ky][kx] flattens to the same order).
// The sample walk, the bilinear weights and their order are torchvision's
// CUDA kernel's, written from its published algorithm.
__device__ __forceinline__ float pd_sam3_roi_bilinear(const float* __restrict__ f, uint32_t H,
                                                      uint32_t W, uint32_t C, uint32_t c,
                                                      float y, float x) {
    if (y < -1.0f || y > (float)H || x < -1.0f || x > (float)W) return 0.0f;
    if (y <= 0.0f) y = 0.0f;
    if (x <= 0.0f) x = 0.0f;
    uint32_t yl = (uint32_t)y, xl = (uint32_t)x, yh, xh;
    if (yl >= H - 1u) {
        yh = yl = H - 1u;
        y = (float)yl;
    } else {
        yh = yl + 1u;
    }
    if (xl >= W - 1u) {
        xh = xl = W - 1u;
        x = (float)xl;
    } else {
        xh = xl + 1u;
    }
    const float ly = y - (float)yl, lx = x - (float)xl, hy = 1.0f - ly, hx = 1.0f - lx;
    const float v1 = f[((size_t)yl * W + xl) * C + c], v2 = f[((size_t)yl * W + xh) * C + c];
    const float v3 = f[((size_t)yh * W + xl) * C + c], v4 = f[((size_t)yh * W + xh) * C + c];
    return hy * hx * v1 + hy * lx * v2 + ly * hx * v3 + ly * lx * v4;
}

__global__ void pd_sam3_roi_align_kernel(const float* __restrict__ feat,
                                         const float* __restrict__ boxes, __half* __restrict__ out,
                                         uint32_t H, uint32_t W, uint32_t C, uint32_t P) {
    const uint32_t bi = blockIdx.x, bin = blockIdx.y;  // bin = ph * P + pw
    const uint32_t ph = bin / P, pw = bin - ph * P;
    const float* bxr = boxes + (size_t)bi * 4u;
    const float x0 = bxr[0], y0 = bxr[1];
    const float rw = fmaxf(bxr[2] - x0, 1.0f), rh = fmaxf(bxr[3] - y0, 1.0f);
    const float bw = rw / (float)P, bh = rh / (float)P;
    const uint32_t gh = (uint32_t)ceilf(rh / (float)P), gw = (uint32_t)ceilf(rw / (float)P);
    const float count = (float)(gh * gw > 0u ? gh * gw : 1u);
    for (uint32_t c = threadIdx.x; c < C; c += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t iy = 0; iy < gh; ++iy) {
            const float y = y0 + ph * bh + ((float)iy + 0.5f) * bh / (float)gh;
            for (uint32_t ix = 0; ix < gw; ++ix) {
                const float x = x0 + pw * bw + ((float)ix + 0.5f) * bw / (float)gw;
                acc += pd_sam3_roi_bilinear(feat, H, W, C, c, y, x);
            }
        }
        out[((size_t)bi * C + c) * P * P + bin] = __float2half(acc / count);
    }
}

PD_EXPORT
int pd_sam3_roi_align(const void* feat, const void* boxes, void* out, uint32_t n, uint32_t H,
                      uint32_t W, uint32_t C, uint32_t pooled, void* stream) {
    if (n == 0) return 0;
    if (H == 0 || W == 0 || C == 0 || pooled == 0) return cudaErrorInvalidValue;
    dim3 grid(n, pooled * pooled);
    pd_sam3_roi_align_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(
        (const float*)feat, (const float*)boxes, (__half*)out, H, W, C, pooled);
    return pd_launch_status();
}

// 768: the dot-product scorer's last step and the processor's score, for the
// last decoder layer: logit[q] = clamp(scale * hp[q] . pp, +-clamp) and
// prob[q] = sigmoid(logit[q]) * sigmoid(presence). One warp a query, fixed
// lane order.
__global__ void pd_sam3_score_kernel(const float* __restrict__ hp, const float* __restrict__ pp,
                                     const float* __restrict__ presence,
                                     float* __restrict__ logit, float* __restrict__ prob,
                                     uint32_t nq, uint32_t d, float scale, float clampv) {
    const uint32_t warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31u;
    if (warp >= nq) return;
    float a = 0.0f;
    for (uint32_t i = lane; i < d; i += 32u) a += hp[(size_t)warp * d + i] * pp[i];
    for (uint32_t o = 16u; o > 0u; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
    if (lane == 0u) {
        float s = a * scale;
        s = fminf(fmaxf(s, -clampv), clampv);
        logit[warp] = s;
        const float pr = 1.0f / (1.0f + expf(-presence[0]));
        prob[warp] = (1.0f / (1.0f + expf(-s))) * pr;
    }
}

PD_EXPORT
int pd_sam3_score(const void* hp, const void* pp, const void* presence, void* logit, void* prob,
                  uint32_t nq, uint32_t d, float scale, float clampv, void* stream) {
    if (nq == 0) return 0;
    pd_sam3_score_kernel<<<(nq * 32u + 255u) / 256u, 256, 0, (cudaStream_t)stream>>>(
        (const float*)hp, (const float*)pp, (const float*)presence, (float*)logit, (float*)prob,
        nq, d, scale, clampv);
    return pd_launch_status();
}
