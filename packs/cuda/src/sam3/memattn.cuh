// sam3/memattn.cuh - SAM 3's memory attention: the axial rope landing of its 256-wide q and k, and one-head attention with a 256-wide score and a 64- or 128-wide value
// Textually-included segment of the single pack translation unit.
// Not standalone-compilable: include order is defined by ../pack.cu.
// ------------------------------------------------------------ sam3 memory attention
// The tracker conditions every frame's 72^2 feature on the object's memory:
// four pre-norm layers of self-attention (5184 x 5184) and cross-attention
// over the memory bank (7-8 frames x 5184 tokens of 64 channels, plus 4
// tokens a past object pointer), ONE head of 256, axial rope on q and on the
// memory tokens' keys (the pointer tokens are not rotated).
//
// Two facts set the kernel's shape. The keys are rotated after their
// projection, so the score is a full 256 wide. But the cross-attention's
// values are a projection of the 64-channel memory and the weights of a row
// sum to one, so o_proj(sum p v_proj(m)) = (W_o W_v) (sum p m) + W_o b_v +
// b_o - the value side runs at 64 and the two projections fold into one
// 64 -> 256 GEMM behind it (exact algebra; the engine folds them at load).
// The self-attention's values are a full 256; it runs as two 128-wide value
// halves over the same scores. A flash-attention-2 walk (online softmax over
// 32-key tiles, q fragments in registers) - the vision tower's
// pd_vision_attn_mma_kernel at a 256 score and a separate value width.
//
// Needs attn/decode_spec.cuh's pd_fa_mma16 / PD_FA_OK and f32_qkv.cuh's
// pd_launch_status.

// 788: the rope landing. in is f32 rows [rows][in_stride] (a GEMM's f32
// landing, bias already in) read from column in_off, d columns; out f16
// [rows][d]. Row r is row r % group of its group; it is rotated when that is
// < nrope, by table row (r % group) % period, rotate-half pairs (j, j + d/2)
// (the projection's rows were permuted at load so Meta's interleaved complex
// pairs land there); cs / sn are [period][d/2]. Then * scale (q's 1/sqrt d).
// cs null = no rotation at all (a value plane's convert). One block a row, a
// thread a pair.
__global__ void pd_sam3_rope_rows_h_kernel(const float* __restrict__ in, __half* __restrict__ out,
                                           const float* __restrict__ cs,
                                           const float* __restrict__ sn, uint32_t in_stride,
                                           uint32_t in_off, uint32_t d, uint32_t group,
                                           uint32_t nrope, uint32_t period, float scale) {
    const uint32_t r = blockIdx.x, j = threadIdx.x, h = d / 2u;
    const float* src = in + (size_t)r * in_stride + in_off;
    float a = src[j], b = src[j + h];
    const uint32_t rg = r % group;
    if (cs != nullptr && rg < nrope) {
        const uint32_t t = rg % period;
        const float c = cs[(size_t)t * h + j], s = sn[(size_t)t * h + j];
        const float ra = a * c - b * s, rb = a * s + b * c;
        a = ra;
        b = rb;
    }
    __half* dst = out + (size_t)r * d;
    dst[j] = __float2half_rn(a * scale);
    dst[j + h] = __float2half_rn(b * scale);
}

PD_EXPORT
int pd_sam3_rope_rows_h(const void* in, void* out, const void* cs, const void* sn, uint32_t rows,
                        uint32_t in_stride, uint32_t in_off, uint32_t d, uint32_t group,
                        uint32_t nrope, uint32_t period, float scale, void* stream) {
    if (rows == 0u) return 0;
    if (d == 0u || (d & 1u) != 0u || d / 2u > 1024u || in_off + d > in_stride || group == 0u ||
        period == 0u || rows > 0x7fffffffu || ((cs == nullptr) != (sn == nullptr)))
        return cudaErrorInvalidValue;
    pd_sam3_rope_rows_h_kernel<<<rows, d / 2u, 0, (cudaStream_t)stream>>>(
        (const float*)in, (__half*)out, (const float*)cs, (const float*)sn, in_stride, in_off, d,
        group, nrope, period, scale);
    return pd_launch_status();
}

// 789: attention, one head, for `ngroups` independent groups (objects): q
// [g][nq][256] halves PRE-SCALED by 1/16, k [g][nk][256], v rows of DV halves
// at stride ldv ([g][nk] rows, a group sv = nk * ldv apart), out rows of DV
// halves at stride ldo ([g][nq], so = nq * ldo apart). The caller offsets v
// and out to take one value half of a wider plane. 32-key tiles, QW warps of
// 16 query rows a block; the scores and the softmax are the vision kernel's
// (f32 accumulators, online max / sum over a row's four lanes).
template <uint32_t DV, uint32_t QW>
__global__ void __launch_bounds__(32u * QW)
    pd_sam3_mem_attn_kernel(const __half* __restrict__ q, const __half* __restrict__ k,
                            const __half* __restrict__ v, __half* __restrict__ out, uint32_t nq,
                            uint32_t nk, uint32_t ldv, uint32_t ldo) {
#if PD_FA_OK
    constexpr uint32_t DQ = 256u, KT = 32u, DQP = DQ + 8u, DVP = DV + 8u, NT = 32u * QW;
    __shared__ __align__(16) __half sh_k[KT * DQP];
    __shared__ __align__(16) __half sh_v[KT * DVP];
    const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
    const uint32_t g8 = lane >> 2, t4 = lane & 3u, lg = lane >> 3;
    const uint32_t row0 = blockIdx.x * (QW * 16u) + warp * 16u;
    const size_t g = blockIdx.z;
    const __half* qg = q + g * nq * DQ;
    const __half* kg = k + g * nk * DQ;
    const __half* vg = v + g * (size_t)nk * ldv;
    __half* og = out + g * (size_t)nq * ldo;

    // A-fragment rows this lane owns (mma layout: row = lane/4, and +8)
    const uint32_t jr[2] = {g8, g8 + 8u};
    uint32_t qa[DQ / 16u][4];
#pragma unroll
    for (uint32_t d0 = 0; d0 < DQ / 16u; ++d0) {
#pragma unroll
        for (uint32_t e = 0; e < 2u; ++e) {
            const uint32_t b = row0 + jr[e];
            const bool rowok = b < nq;
            const __half* qp = qg + (size_t)(rowok ? b : nq - 1u) * DQ;
            const uint32_t c0 = d0 * 16u + 2u * t4, c1 = c0 + 8u;
            const uint32_t w0 = *reinterpret_cast<const uint32_t*>(qp + c0);
            const uint32_t w1 = *reinterpret_cast<const uint32_t*>(qp + c1);
            qa[d0][e] = rowok ? w0 : 0u;
            qa[d0][e + 2u] = rowok ? w1 : 0u;
        }
    }

    float m_st[2] = {-1e30f, -1e30f}, l_st[2] = {0.f, 0.f};
    float o_acc[DV / 8u][4];
#pragma unroll
    for (uint32_t nt = 0; nt < DV / 8u; ++nt)
#pragma unroll
        for (uint32_t e = 0; e < 4u; ++e) o_acc[nt][e] = 0.f;

    for (uint32_t t0 = 0; t0 < nk; t0 += KT) {
        // K then V, 8 halves a 16 B copy; rows past nk store zeros so the
        // fragments are always defined (their scores are masked below)
        constexpr uint32_t KCH = KT * (DQ / 8u), VCH = KT * (DV / 8u);
        for (uint32_t u = tid; u < KCH + VCH; u += NT) {
            const bool isv = u >= KCH;
            const uint32_t ur = isv ? u - KCH : u, per = isv ? DV / 8u : DQ / 8u;
            const uint32_t kk = ur / per, d8 = (ur % per) * 8u, ks = t0 + kk;
            __half* dst = isv ? sh_v + (size_t)kk * DVP + d8 : sh_k + (size_t)kk * DQP + d8;
            if (ks < nk) {
                const __half* src = isv ? vg + (size_t)ks * ldv + d8 : kg + (size_t)ks * DQ + d8;
                *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
            } else {
                *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        __syncthreads();

        float s_acc[KT / 8u][4];
#pragma unroll
        for (uint32_t nt = 0; nt < KT / 8u; ++nt)
#pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) s_acc[nt][e] = 0.f;
#pragma unroll
        for (uint32_t d0 = 0; d0 < DQ / 16u; ++d0) {
#pragma unroll
            for (uint32_t np = 0; np < KT / 16u; ++np) {
                const __half* kp = sh_k + (size_t)(np * 16u + (lg >> 1) * 8u + (lane & 7u)) * DQP +
                                   d0 * 16u + (lg & 1u) * 8u;
                uint32_t kb4[4];
                const uint32_t ka = (uint32_t)__cvta_generic_to_shared(kp);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(kb4[0]), "=r"(kb4[1]), "=r"(kb4[2]), "=r"(kb4[3])
                             : "r"(ka));
                pd_fa_mma16(s_acc[np * 2u], qa[d0][0], qa[d0][1], qa[d0][2], qa[d0][3], kb4[0],
                            kb4[1]);
                pd_fa_mma16(s_acc[np * 2u + 1u], qa[d0][0], qa[d0][1], qa[d0][2], qa[d0][3],
                            kb4[2], kb4[3]);
            }
        }
        // online softmax over the tile; a row's 4 lanes share lane/4
        float mn[2] = {m_st[0], m_st[1]};
#pragma unroll
        for (uint32_t nt = 0; nt < KT / 8u; ++nt) {
            const uint32_t kb = t0 + nt * 8u + 2u * t4;
#pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) {
                if (kb + (e & 1u) >= nk) s_acc[nt][e] = -1e30f;
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
                const float w = __expf(s_acc[nt][e] - mn[e >> 1]);
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
            corr[r] = __expf(m_st[r] - mn[r]);
            l_st[r] = l_st[r] * corr[r] + ws[r];
            m_st[r] = mn[r];
        }
#pragma unroll
        for (uint32_t nt = 0; nt < DV / 8u; ++nt) {
            o_acc[nt][0] *= corr[0];
            o_acc[nt][1] *= corr[0];
            o_acc[nt][2] *= corr[1];
            o_acc[nt][3] *= corr[1];
        }
        // PV: A = the tile's weights straight out of the score C-fragment,
        // B = V read [k=key][n=dim] through ldmatrix.trans
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
            const __half* vp = sh_v + (size_t)vr * DVP + (lg >> 1) * 8u;
#pragma unroll
            for (uint32_t nt = 0; nt < DV / 8u; nt += 2u) {
                uint32_t vb4[4];
                const uint32_t va = (uint32_t)__cvta_generic_to_shared(vp + nt * 8u);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(vb4[0]), "=r"(vb4[1]), "=r"(vb4[2]), "=r"(vb4[3])
                             : "r"(va));
                pd_fa_mma16(o_acc[nt], pa0, pa1, pa2, pa3, vb4[0], vb4[1]);
                pd_fa_mma16(o_acc[nt + 1u], pa0, pa1, pa2, pa3, vb4[2], vb4[3]);
            }
        }
        __syncthreads();  // tiles read before the next stage overwrites them
    }

    const float nrm[2] = {l_st[0] > 0.f ? 1.f / l_st[0] : 0.f, l_st[1] > 0.f ? 1.f / l_st[1] : 0.f};
#pragma unroll
    for (uint32_t nt = 0; nt < DV / 8u; ++nt) {
        const uint32_t dcol = nt * 8u + 2u * t4;
#pragma unroll
        for (uint32_t r = 0; r < 2u; ++r) {
            const uint32_t b = row0 + jr[r];
            if (b >= nq) continue;
            *reinterpret_cast<__half2*>(og + (size_t)b * ldo + dcol) =
                __floats2half2_rn(o_acc[nt][2u * r] * nrm[r], o_acc[nt][2u * r + 1u] * nrm[r]);
        }
    }
#else
    (void)q; (void)k; (void)v; (void)out; (void)nq; (void)nk; (void)ldv; (void)ldo;
#endif
}

PD_EXPORT
int pd_sam3_mem_attn_h(const void* q, const void* k, const void* v, void* out, uint32_t nq,
                       uint32_t nk, uint32_t ngroups, uint32_t dv, uint32_t ldv, uint32_t ldo,
                       void* stream) {
    if (nq == 0u || nk == 0u || ngroups == 0u) return 0;
    if ((dv != 64u && dv != 128u) || ldv < dv || ldo < dv || (ldv & 7u) != 0u || (ldo & 7u) != 0u)
        return cudaErrorInvalidValue;
    int dev = 0, cc = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&cc, cudaDevAttrComputeCapabilityMajor, dev);
    if (cc < 8) return cudaErrorInvalidValue;
    constexpr uint32_t QW = 4u;
    dim3 grid((nq + QW * 16u - 1u) / (QW * 16u), 1u, ngroups);
    cudaStream_t st = (cudaStream_t)stream;
    if (dv == 64u)
        pd_sam3_mem_attn_kernel<64u, QW><<<grid, 32u * QW, 0, st>>>(
            (const __half*)q, (const __half*)k, (const __half*)v, (__half*)out, nq, nk, ldv, ldo);
    else
        pd_sam3_mem_attn_kernel<128u, QW><<<grid, 32u * QW, 0, st>>>(
            (const __half*)q, (const __half*)k, (const __half*)v, (__half*)out, nq, nk, ldv, ldo);
    return pd_launch_status();
}
