// moe/inkling2_experts_tc.cuh - Inkling 2 Mini's experts at prompt widths on
// the block-scale tensor cores (slots 991-992), both declared classes.
// Textually-included segment of the single pack translation unit.
// Not standalone-compilable: include order is defined by ../pack.cu.
//
// Needs moe/nvf4_sorted.cuh (PD_NV4M_* tile geometry, pd_nv4t_arm),
// quant/nvf4.cuh (pd_nv4_mma), mmq.cuh's pd_cp_async16 and
// decode_spec.cuh's pd_ldm_x4.
//
// The CUDA-core grouped arms (slots 987-988) decode both operands to f32
// and FMA them; here the packed e2m1 codes go straight into
// mma.sync kind::mxf4nvf4 m16n8k64 with their ue4m3 block scales, the
// skeleton of the NVFP4 sorted MoE pair (cp.async double buffer, 128 weight
// rows x 32 sorted pairs a CTA, the pairs gathered through moe_align's
// layout). What differs is Inkling's: the activations carry a per-row outer
// scale (the four-over-six rule's amax / 1536) and the weights a per-expert
// factor - both multiply the accumulator in the epilogue, exactly where the
// CUDA-core arms apply them - and w13's rows interleave gate (2i) and up
// (2i + 1). e2m1 x e2m1 products and their block-scale products are exact,
// so only the f32 accumulation order differs from slots 987-988.
//
// The MXFP8 class (layers 20-23) rides the same tiles: e4m3 x e4m3 on
// kind::mxf8f6f4 m16n8k32 with one ue8m0 scale a 32-element block. A
// 32-byte smem step is one k64 of packed fp4 or one k32 of fp8, so the
// fragment walk is the same; the scale is one byte a step instead of four.
// FMT 0 = NVFP4 (2 elements a byte, e4m3 a 16), 1 = MXFP8 (1 a byte, ue8m0 a 32).
#define PD_IK2TC_RB(FMT, K) ((FMT) == 0u ? (K) >> 1 : (K))
#define PD_IK2TC_NS(FMT, K) ((FMT) == 0u ? (K) >> 4 : (K) >> 5)
#define PD_IK2TC_SPC(FMT, KB) ((FMT) == 0u ? (KB) * 2u : (KB) / 2u)

#if PD_BS_OK
// e4m3 x e4m3 with ue8m0 block scales, one a k32 (scale_vec::1X)
__device__ __forceinline__ void pd_ik2_mx8_mma(float d[4], uint32_t a0, uint32_t a1, uint32_t a2,
                                               uint32_t a3, uint32_t b0, uint32_t b1,
                                               uint32_t sfa, uint32_t sfb) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X"
        ".f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
        "{%10}, {0, 0}, {%11}, {0, 0};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "r"(sfb));
}
#endif

#define PD_IK2TC_GU_SMEM(KB) (4u * 128u * PD_NV4M_WROW(KB) + 2u * PD_NV4M_BM * PD_NV4M_YROW(KB))
#define PD_IK2TC_DN_SMEM(KB) (2u * 128u * PD_NV4M_WROW(KB) + 2u * PD_NV4M_BM * PD_NV4M_YROW(KB))

// One K-chunk's weight scales (plain loads after the data lands, as the
// sorted pair does) for 128 rows of plane `src` whose row r is `wrow(r)`.
#define PD_IK2TC_W_SCALES(dst, src, WROW_OF, kt)                                    \
    for (uint32_t u = tid; u < 128u * SPC; u += 256u) {                              \
        const uint32_t row = u / SPC, kb16 = u % SPC;                                \
        const bool ok = (row_base + row) < rows_n && (kt) * SPC + kb16 < n_sc;       \
        (dst)[row * WROW + KB * 16u + kb16] =                                        \
            ok ? (src)[(size_t)(WROW_OF(row)) * n_sc + (kt) * SPC + kb16] : 0u;      \
    }
#define PD_IK2TC_W_DATA(dst, src, WROW_OF, kt)                                       \
    for (uint32_t u = tid; u < 128u * KB; u += 256u) {                               \
        const uint32_t row = u / KB, seg = u % KB;                                   \
        const bool ok = (row_base + row) < rows_n && (kt) * KB + seg < n_kb;         \
        pd_cp_async16((int*)((dst) + row * WROW + seg * 16u),                        \
                      (src) + (size_t)(WROW_OF(ok ? row : 0u)) * RB +                \
                          (kt) * (KB * 16u) + seg * 16u,                             \
                      ok);                                                           \
    }
// the gathered activation rows (`yrow[col]`, PAD -> zero-filled) of chunk kt
#define PD_IK2TC_Y_DATA(dst, kt)                                                     \
    for (uint32_t u = tid; u < PD_NV4M_BM * KB; u += 256u) {                         \
        const uint32_t col = u / KB, seg = u % KB;                                   \
        const uint32_t r = yrow[col];                                                \
        const bool ok = r != PD_MOE_PAD && (kt) * KB + seg < n_kb;                   \
        pd_cp_async16((int*)((dst) + col * YROW + 16u + seg * 16u),                  \
                      yq + (size_t)(ok ? r : 0u) * RB + (kt) * (KB * 16u) + seg * 16u, \
                      ok);                                                           \
    }
#define PD_IK2TC_Y_SCALES(dst, kt)                                                   \
    for (uint32_t u = tid; u < PD_NV4M_BM * SPC; u += 256u) {                        \
        const uint32_t col = u / SPC, kb16 = u % SPC;                                \
        const uint32_t r = yrow[col];                                                \
        const bool ok = r != PD_MOE_PAD && (kt) * SPC + kb16 < n_sc;                 \
        (dst)[col * YROW + kb16] = ok ? ys[(size_t)r * n_sc + (kt) * SPC + kb16] : 0u; \
    }

// A warp's 32 weight rows x 16 pairs of one K-chunk into acc (the sorted
// pair's fragment walk: ldmatrix the packed rows, the scale words straight
// from smem).
template <uint32_t KB, uint32_t FMT>
__device__ __forceinline__ void pd_ik2tc_chunk(const unsigned char* tw, const unsigned char* ty,
                                               uint32_t i0, uint32_t joff, float (&acc)[4][4]) {
    constexpr uint32_t WROW = PD_NV4M_WROW(KB);
    constexpr uint32_t YROW = PD_NV4M_YROW(KB);
    const uint32_t lane = threadIdx.x & 31u, g = lane >> 2, tq = lane & 3u;
    uint32_t am[2][KB / 2u][4], sa[2][KB / 2u];
    #pragma unroll
    for (uint32_t n = 0; n < 2u; ++n) {
        const uint32_t r0 = i0 + n * 16u + g;
        const uint32_t rs = (tq & 1u) ? r0 + 8u : r0;
        const uint32_t wo = (i0 + n * 16u + ((lane >> 3) & 1u) * 8u + (lane & 7u)) * WROW;
        #pragma unroll
        for (uint32_t k64 = 0; k64 < KB / 2u; ++k64) {
            pd_ldm_x4(am[n][k64], tw + wo + k64 * 32u + (lane >> 4) * 16u);
            sa[n][k64] = FMT == 0u ? *(const uint32_t*)(tw + rs * WROW + KB * 16u + k64 * 4u)
                                   : (uint32_t)tw[rs * WROW + KB * 16u + k64];
        }
    }
    #pragma unroll
    for (uint32_t j0 = 0; j0 < PD_NV4M_BM; j0 += 16u) {
        uint32_t bm[2u * (KB / 2u)];
        #pragma unroll
        for (uint32_t q = 0; q < KB / 4u; ++q)
            pd_ldm_x4(bm + q * 4u,
                      ty + (j0 + joff + (lane & 7u)) * YROW + 16u + q * 64u + (lane >> 3) * 16u);
        const unsigned char* ysr = ty + (j0 + joff + g) * YROW;
        #pragma unroll
        for (uint32_t k64 = 0; k64 < KB / 2u; ++k64) {
            const uint32_t sb = FMT == 0u ? *(const uint32_t*)(ysr + k64 * 4u) : (uint32_t)ysr[k64];
            #pragma unroll
            for (uint32_t n = 0; n < 2u; ++n) {
                if (FMT == 0u)
                    pd_nv4_mma(acc[(j0 >> 3) + n], am[n][k64][0], am[n][k64][1], am[n][k64][2],
                               am[n][k64][3], bm[k64 * 2u], bm[k64 * 2u + 1u], sa[n][k64], sb);
                else
                    pd_ik2_mx8_mma(acc[(j0 >> 3) + n], am[n][k64][0], am[n][k64][1],
                                   am[n][k64][2], am[n][k64][3], bm[k64 * 2u],
                                   bm[k64 * 2u + 1u], sa[n][k64], sb);
            }
        }
    }
}

// slot 991: gate|up of a sorted wave on the block-scale tensor cores. grid
// (moe_align blocks, ff / 128); a CTA: 128 ff columns (w13 rows 2i / 2i + 1
// as the gate / up planes) x the block's 32 pairs. act[p][i] for pair p =
// token * topk + slot; epilogue as slot 987's (outer x factor, bf16, the
// softcapped SwiGLU).
template <uint32_t KB, uint32_t FMT>
__global__ void __launch_bounds__(256, 2) pd_inkling2_moe_gu_tc_kernel(
        const uint8_t* __restrict__ w, const uint8_t* __restrict__ wsc,
        const float* __restrict__ wfac, const uint8_t* xq, const uint8_t* xsc, const float* xo,
        const uint32_t* sorted_row, const uint32_t* sorted_slot, const uint32_t* block_expert,
        uint32_t topk, float* __restrict__ act, uint32_t K, uint32_t ff, float gcap, float ucap) {
#if PD_BS_OK
    constexpr uint32_t WROW = PD_NV4M_WROW(KB);
    constexpr uint32_t YROW = PD_NV4M_YROW(KB);
    PD_PDL_ARM();
    const uint32_t blk = blockIdx.x;
    const uint32_t e = pd_pdl_after(block_expert)[blk];
    if (e == PD_MOE_PAD) return;
    const uint32_t row_base = blockIdx.y * 128u, rows_n = ff;
    extern __shared__ unsigned char pd_bs_sh[];
    unsigned char* wg0 = pd_bs_sh;
    unsigned char* wg1 = wg0 + 128u * WROW;
    unsigned char* wu0 = wg1 + 128u * WROW;
    unsigned char* wu1 = wu0 + 128u * WROW;
    unsigned char* yb0 = wu1 + 128u * WROW;
    unsigned char* yb1 = yb0 + PD_NV4M_BM * YROW;
    __shared__ uint32_t yrow[PD_NV4M_BM], yslot[PD_NV4M_BM];
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, tq = lane & 3u;
    const uint32_t i0 = (warp >> 1) * 32u, joff = (warp & 1u) * 8u;
    constexpr uint32_t SPC = PD_IK2TC_SPC(FMT, KB);
    const uint32_t RB = PD_IK2TC_RB(FMT, K), n_kb = RB >> 4, n_sc = PD_IK2TC_NS(FMT, K);
    const uint32_t nk = (n_kb + KB - 1u) / KB;
    const uint8_t* yq = pd_pdl_after(xq);
    const uint8_t* ys = pd_pdl_after(xsc);
    if (tid < PD_NV4M_BM) {
        yrow[tid] = pd_pdl_after(sorted_row)[(size_t)blk * PD_NV4M_BM + tid];
        yslot[tid] = pd_pdl_after(sorted_slot)[(size_t)blk * PD_NV4M_BM + tid];
    }
    __syncthreads();
    // w13 row of ff column (row_base + r): the gate plane's, the up plane's
    #define PD_IK2TC_GATE(r) (2u * ((size_t)e * ff + row_base + (r)))
    #define PD_IK2TC_UP(r) (2u * ((size_t)e * ff + row_base + (r)) + 1u)
    float accg[4][4] = {}, accu[4][4] = {};
    PD_IK2TC_W_DATA(wg0, w, PD_IK2TC_GATE, 0u)
    PD_IK2TC_W_DATA(wu0, w, PD_IK2TC_UP, 0u)
    PD_IK2TC_Y_DATA(yb0, 0u)
    asm volatile("cp.async.commit_group;");
    for (uint32_t kt = 0; kt < nk; ++kt) {
        unsigned char* tg = (kt & 1u) ? wg1 : wg0;
        unsigned char* tu = (kt & 1u) ? wu1 : wu0;
        unsigned char* ty = (kt & 1u) ? yb1 : yb0;
        if (kt + 1u < nk) {
            PD_IK2TC_W_DATA((kt & 1u) ? wg0 : wg1, w, PD_IK2TC_GATE, kt + 1u)
            PD_IK2TC_W_DATA((kt & 1u) ? wu0 : wu1, w, PD_IK2TC_UP, kt + 1u)
            PD_IK2TC_Y_DATA((kt & 1u) ? yb0 : yb1, kt + 1u)
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        PD_IK2TC_W_SCALES(tg, wsc, PD_IK2TC_GATE, kt)
        PD_IK2TC_W_SCALES(tu, wsc, PD_IK2TC_UP, kt)
        PD_IK2TC_Y_SCALES(ty, kt)
        __syncthreads();
        pd_ik2tc_chunk<KB, FMT>(tg, ty, i0, joff, accg);
        pd_ik2tc_chunk<KB, FMT>(tu, ty, i0, joff, accu);
        __syncthreads();
    }
    #undef PD_IK2TC_GATE
    #undef PD_IK2TC_UP
    // acc[(j0 >> 3) + n][qc + 2h]: ff column row_base + i0 + 16n + g + 8h,
    // pair column joff + j0 + 2tq + qc
    // the NVFP4 class's outer scale x expert factor; MXFP8 carries neither
    const float fe = FMT == 0u ? wfac[e] : 1.0f;
    const float* xor_ = FMT == 0u ? pd_pdl_after(xo) : nullptr;
    #pragma unroll
    for (uint32_t j0 = 0; j0 < PD_NV4M_BM; j0 += 16u) {
        #pragma unroll
        for (uint32_t qc = 0; qc < 2u; ++qc) {
            const uint32_t c = joff + j0 + 2u * tq + qc;
            const uint32_t t = yrow[c];
            if (t == PD_MOE_PAD) continue;
            const float f = FMT == 0u ? xor_[t] * fe : 1.0f;
            float* dst = act + ((size_t)t * topk + yslot[c]) * ff;
            #pragma unroll
            for (uint32_t n = 0; n < 2u; ++n) {
                #pragma unroll
                for (uint32_t h = 0; h < 2u; ++h) {
                    const uint32_t i = row_base + i0 + n * 16u + g + 8u * h;
                    if (i >= ff) continue;
                    const float gv = __bfloat162float(__float2bfloat16_rn(accg[(j0 >> 3) + n][qc + 2u * h] * f));
                    const float uv = __bfloat162float(__float2bfloat16_rn(accu[(j0 >> 3) + n][qc + 2u * h] * f));
                    const float sig = 1.0f / (1.0f + expf(-gv));
                    dst[i] = ((gcap * tanhf(gv / gcap)) * sig) * (ucap * tanhf(uv / ucap));
                }
            }
        }
    }
#else
    (void)w; (void)wsc; (void)wfac; (void)xq; (void)xsc; (void)xo; (void)sorted_row;
    (void)sorted_slot; (void)block_expert; (void)topk; (void)act; (void)K; (void)ff;
    (void)gcap; (void)ucap;
#endif
}

// slot 992: down of a sorted wave on the block-scale tensor cores. grid
// (moe_align blocks, hidden / 128); a CTA: 128 output columns x the
// block's 32 pairs, part[p][o] = acc * outer[p] * factor[e] (slot 988's).
template <uint32_t KB, uint32_t FMT>
__global__ void __launch_bounds__(256, 2) pd_inkling2_moe_dn_tc_kernel(
        const uint8_t* __restrict__ w, const uint8_t* __restrict__ wsc,
        const float* __restrict__ wfac, const uint8_t* aq, const uint8_t* asc, const float* ao,
        const uint32_t* sorted_row, const uint32_t* sorted_slot, const uint32_t* block_expert,
        uint32_t topk, float* __restrict__ part, uint32_t K, uint32_t hidden) {
#if PD_BS_OK
    constexpr uint32_t WROW = PD_NV4M_WROW(KB);
    constexpr uint32_t YROW = PD_NV4M_YROW(KB);
    PD_PDL_ARM();
    const uint32_t blk = blockIdx.x;
    const uint32_t e = pd_pdl_after(block_expert)[blk];
    if (e == PD_MOE_PAD) return;
    const uint32_t row_base = blockIdx.y * 128u, rows_n = hidden;
    extern __shared__ unsigned char pd_bs_sh[];
    unsigned char* w0 = pd_bs_sh;
    unsigned char* w1 = w0 + 128u * WROW;
    unsigned char* yb0 = w1 + 128u * WROW;
    unsigned char* yb1 = yb0 + PD_NV4M_BM * YROW;
    __shared__ uint32_t yrow[PD_NV4M_BM];
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, tq = lane & 3u;
    const uint32_t i0 = (warp >> 1) * 32u, joff = (warp & 1u) * 8u;
    constexpr uint32_t SPC = PD_IK2TC_SPC(FMT, KB);
    const uint32_t RB = PD_IK2TC_RB(FMT, K), n_kb = RB >> 4, n_sc = PD_IK2TC_NS(FMT, K);
    const uint32_t nk = (n_kb + KB - 1u) / KB;
    const uint8_t* yq = pd_pdl_after(aq);
    const uint8_t* ys = pd_pdl_after(asc);
    if (tid < PD_NV4M_BM) {
        // the pair's row of the down input
        const uint32_t t = pd_pdl_after(sorted_row)[(size_t)blk * PD_NV4M_BM + tid];
        yrow[tid] = t == PD_MOE_PAD
                        ? PD_MOE_PAD
                        : t * topk + pd_pdl_after(sorted_slot)[(size_t)blk * PD_NV4M_BM + tid];
    }
    __syncthreads();
    #define PD_IK2TC_ROW(r) ((size_t)e * hidden + row_base + (r))
    float acc[4][4] = {};
    PD_IK2TC_W_DATA(w0, w, PD_IK2TC_ROW, 0u)
    PD_IK2TC_Y_DATA(yb0, 0u)
    asm volatile("cp.async.commit_group;");
    for (uint32_t kt = 0; kt < nk; ++kt) {
        unsigned char* tw = (kt & 1u) ? w1 : w0;
        unsigned char* ty = (kt & 1u) ? yb1 : yb0;
        if (kt + 1u < nk) {
            PD_IK2TC_W_DATA((kt & 1u) ? w0 : w1, w, PD_IK2TC_ROW, kt + 1u)
            PD_IK2TC_Y_DATA((kt & 1u) ? yb0 : yb1, kt + 1u)
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        PD_IK2TC_W_SCALES(tw, wsc, PD_IK2TC_ROW, kt)
        PD_IK2TC_Y_SCALES(ty, kt)
        __syncthreads();
        pd_ik2tc_chunk<KB, FMT>(tw, ty, i0, joff, acc);
        __syncthreads();
    }
    #undef PD_IK2TC_ROW
    const float fe = FMT == 0u ? wfac[e] : 1.0f;
    const float* aor = FMT == 0u ? pd_pdl_after(ao) : nullptr;
    #pragma unroll
    for (uint32_t j0 = 0; j0 < PD_NV4M_BM; j0 += 16u) {
        #pragma unroll
        for (uint32_t qc = 0; qc < 2u; ++qc) {
            const uint32_t p = yrow[joff + j0 + 2u * tq + qc];
            if (p == PD_MOE_PAD) continue;
            const float f = FMT == 0u ? aor[p] * fe : 1.0f;
            #pragma unroll
            for (uint32_t n = 0; n < 2u; ++n) {
                #pragma unroll
                for (uint32_t h = 0; h < 2u; ++h) {
                    const uint32_t o = row_base + i0 + n * 16u + g + 8u * h;
                    if (o < hidden) part[(size_t)p * hidden + o] = acc[(j0 >> 3) + n][qc + 2u * h] * f;
                }
            }
        }
    }
#else
    (void)w; (void)wsc; (void)wfac; (void)aq; (void)asc; (void)ao; (void)sorted_row;
    (void)sorted_slot; (void)block_expert; (void)topk; (void)part; (void)K; (void)hidden;
#endif
}

#undef PD_IK2TC_W_SCALES
#undef PD_IK2TC_W_DATA
#undef PD_IK2TC_Y_DATA
#undef PD_IK2TC_Y_SCALES

PD_EXPORT
int pd_inkling2_moe_gu_tc(const void* w, const void* wsc, const void* wfac, const void* xq,
                          const void* xsc, const void* xo, const void* sorted_row,
                          const void* sorted_slot, const void* block_expert, uint32_t max_blocks,
                          uint32_t topk, void* act, uint32_t K, uint32_t ff, uint32_t fmt,
                          float gcap, float ucap, void* stream) {
#ifndef PD_BS_HOST
    (void)w; (void)wsc; (void)wfac; (void)xq; (void)xsc; (void)xo; (void)sorted_row;
    (void)sorted_slot; (void)block_expert; (void)max_blocks; (void)topk; (void)act; (void)K;
    (void)ff; (void)fmt; (void)gcap; (void)ucap; (void)stream;
    return cudaErrorNotSupported;
#else
    if (max_blocks == 0) return 0;
    // the packed-fp4 tensor core only: a die without one keeps slot 987
    if (pd_nv4t_arm()) return cudaErrorNotSupported;
    // whole 32-element K blocks (a partial last chunk is zero-filled)
    if ((K % 32u) || (ff % 16u) || fmt > 1u || (fmt == 0u && (!wfac || !xo)))
        return cudaErrorInvalidValue;
    constexpr uint32_t KB = 4u;  // two weight planes: KB 8 would halve occupancy
    const uint32_t smem = PD_IK2TC_GU_SMEM(KB);
    // once: decode ticks wider than the GEMVs take this arm inside a capture
    static const bool attr = (cudaFuncSetAttribute(pd_inkling2_moe_gu_tc_kernel<KB, 0u>,
                                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                   (int)smem),
                              cudaFuncSetAttribute(pd_inkling2_moe_gu_tc_kernel<KB, 1u>,
                                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                   (int)smem),
                              true);
    (void)attr;
    const dim3 grid(max_blocks, (ff + 127u) / 128u);
    #define PD_IK2TC_GU(F)                                                                       \
        pd_pdl_go(pd_inkling2_moe_gu_tc_kernel<KB, F>, grid, 256, smem, (cudaStream_t)stream,   \
                  (const uint8_t*)w, (const uint8_t*)wsc, (const float*)wfac, (const uint8_t*)xq, \
                  (const uint8_t*)xsc, (const float*)xo, (const uint32_t*)sorted_row,           \
                  (const uint32_t*)sorted_slot, (const uint32_t*)block_expert, topk, (float*)act, \
                  K, ff, gcap, ucap)
    if (fmt == 0u) PD_IK2TC_GU(0u);
    else PD_IK2TC_GU(1u);
    #undef PD_IK2TC_GU
    return pd_launch_status();
#endif
}

PD_EXPORT
int pd_inkling2_moe_dn_tc(const void* w, const void* wsc, const void* wfac, const void* aq,
                          const void* asc, const void* ao, const void* sorted_row,
                          const void* sorted_slot, const void* block_expert, uint32_t max_blocks,
                          uint32_t topk, void* part, uint32_t ff, uint32_t hidden, uint32_t fmt,
                          void* stream) {
#ifndef PD_BS_HOST
    (void)w; (void)wsc; (void)wfac; (void)aq; (void)asc; (void)ao; (void)sorted_row;
    (void)sorted_slot; (void)block_expert; (void)max_blocks; (void)topk; (void)part; (void)ff;
    (void)hidden; (void)fmt; (void)stream;
    return cudaErrorNotSupported;
#else
    if (max_blocks == 0) return 0;
    if (pd_nv4t_arm()) return cudaErrorNotSupported;
    if ((ff % 32u) || (hidden % 16u) || fmt > 1u || (fmt == 0u && (!wfac || !ao)))
        return cudaErrorInvalidValue;
    constexpr uint32_t KB = 8u;
    const uint32_t smem = PD_IK2TC_DN_SMEM(KB);
    static const bool attr = (cudaFuncSetAttribute(pd_inkling2_moe_dn_tc_kernel<KB, 0u>,
                                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                   (int)smem),
                              cudaFuncSetAttribute(pd_inkling2_moe_dn_tc_kernel<KB, 1u>,
                                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                   (int)smem),
                              true);
    (void)attr;
    const dim3 grid(max_blocks, (hidden + 127u) / 128u);
    #define PD_IK2TC_DN(F)                                                                       \
        pd_pdl_go(pd_inkling2_moe_dn_tc_kernel<KB, F>, grid, 256, smem, (cudaStream_t)stream,   \
                  (const uint8_t*)w, (const uint8_t*)wsc, (const float*)wfac, (const uint8_t*)aq, \
                  (const uint8_t*)asc, (const float*)ao, (const uint32_t*)sorted_row,           \
                  (const uint32_t*)sorted_slot, (const uint32_t*)block_expert, topk, (float*)part, \
                  ff, hidden)
    if (fmt == 0u) PD_IK2TC_DN(0u);
    else PD_IK2TC_DN(1u);
    #undef PD_IK2TC_DN
    return pd_launch_status();
#endif
}

// ---- slots 1002-1003: the NVFP4 class on 64-pair blocks -------------------
// Slots 991-992 take moe_align's 32-pair blocks, so at prompt widths every
// expert's planes are re-read a block (an expert of a 2048-row chunk holds
// ~128 pairs: four reads) through a 2-deep pipeline. These take 64-pair
// blocks on the multi-stage skeleton of the NVFP4 sorted MoE (an ST-deep
// cp.async ring with one barrier a chunk, the CTA's weight strip prefetched
// to L2 ahead of the ring, a block's activations resident for the down
// half), Inkling's planes and epilogues as slots 991-992's. Each
// accumulator sees the same mma sequence (k64 ascending), so the act and
// part rows are slot 991-992's bit for bit. Measured ahead of them at every
// width past the decode GEMVs, 8 tokens to 2048 (the ring and the L2
// prefetch pay on the bandwidth-bound narrow waves too). NVFP4 only (the
// MXFP8 layers keep slots 991-992: their ue8m0 scale slot would be 2 bytes
// a chunk). K % 128 == 0.
#define PD_IK2MS_KB 4u
#define PD_IK2MS_GU_SMEM(BT, ST) \
    ((ST) * (2u * 128u * PD_NV4M_WROW(PD_IK2MS_KB) + (BT) * PD_NV4M_YROW(PD_IK2MS_KB)))

// A16 (slot 1006): act lands as bf16(act x gamma[pair]) - the value the
// quantizer forms from an f32 act and the router weight (slot 983's gamma +
// bf16 rounding), so its codes are the same and the plane half the bytes.
template <uint32_t BT, uint32_t WN, uint32_t ST, bool A16>
__global__ void __launch_bounds__(4u * BT / WN * 32u, 1) pd_inkling2_moe_gu_ms_kernel(
        const uint8_t* __restrict__ w, const uint8_t* __restrict__ wsc,
        const float* __restrict__ wfac, const uint8_t* xq, const uint8_t* xsc, const float* xo,
        const uint32_t* sorted_row, const uint32_t* sorted_slot, const uint32_t* block_expert,
        uint32_t topk, void* __restrict__ act, const float* gamma, uint32_t K, uint32_t ff,
        float gcap, float ucap) {
#if PD_BS_OK
    constexpr uint32_t KB = PD_IK2MS_KB;
    constexpr uint32_t WROW = PD_NV4M_WROW(KB);
    constexpr uint32_t YROW = PD_NV4M_YROW(KB);
    constexpr uint32_t NT = 4u * BT / WN * 32u;
    constexpr uint32_t SS = 2u * 128u * WROW + BT * YROW;
    PD_PDL_ARM();
    // 1-D grid, row tile fastest: a token block's tiles run side by side
    const uint32_t n_rt = (ff + 127u) / 128u;
    const uint32_t blk = blockIdx.x / n_rt;
    const uint32_t e = pd_pdl_after(block_expert)[blk];
    if (e == PD_MOE_PAD) return;
    const uint32_t row_base = (blockIdx.x % n_rt) * 128u;
    extern __shared__ unsigned char pd_bs_sh[];
    __shared__ uint32_t tok[BT], slt[BT];
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, tq = lane & 3u;
    const uint32_t i0 = ((warp >> 1) & 3u) * 32u;
    const uint32_t joff = (warp & 1u) * 8u;
    const uint32_t tb = (warp >> 3) * (2u * WN);
    const uint32_t n_kb = K >> 5, n_k16 = K >> 4, rowb = K >> 1;
    const uint32_t nk = K / (KB * 32u);
    const uint8_t* xqr = pd_pdl_after(xq);
    const uint8_t* xsr = pd_pdl_after(xsc);
    if (tid < BT) {
        tok[tid] = pd_pdl_after(sorted_row)[(size_t)blk * BT + tid];
        slt[tid] = pd_pdl_after(sorted_slot)[(size_t)blk * BT + tid];
    }
    // w13 row of ff column (row_base + r): gate 2i, up 2i + 1 - a tile's
    // gate|up rows are one contiguous span of 2 x nrow rows
    const size_t grow0 = 2u * ((size_t)e * ff + row_base);
    const uint32_t nrow = ff - row_base < 128u ? ff - row_base : 128u;
    // DRAM sees what is prefetched, not what the ring copies (the sorted
    // pair's measurement): the scale span goes to L2 whole now, each weight
    // row a 128-byte window at a time, two windows ahead of the ring
    {
        const uint8_t* s0 = wsc + grow0 * n_k16;
        const uint32_t bytes = 2u * nrow * n_k16;
        for (uint32_t o = tid * 128u; o < bytes + 127u; o += NT * 128u)
            asm volatile("prefetch.global.L2 [%0];" ::"l"(s0 + (o < bytes ? o : bytes - 1u)));
    }
    auto prefetch_w = [&](uint32_t byte0) {
        for (uint32_t q = tid; q < 2u * nrow; q += NT)
            asm volatile("prefetch.global.L2 [%0];" ::"l"(w + (grow0 + q) * rowb + byte0));
    };
    prefetch_w(0u);
    if (128u < rowb) prefetch_w(128u);
    __syncthreads();
    constexpr uint32_t WI = 128u * KB / NT, YI = (BT * KB + NT - 1u) / NT;
    static_assert(WI >= 1u, "a weight segment a thread at least");
    size_t woff[WI];
    uint32_t wdst[WI], wseg[WI];
    bool wrow_ok[WI];
    #pragma unroll
    for (uint32_t i = 0; i < WI; ++i) {
        const uint32_t u = tid + i * NT, row = u / KB;
        wseg[i] = u % KB;
        wdst[i] = row * WROW + wseg[i] * 16u;
        woff[i] = (grow0 + 2u * row) * (size_t)rowb + wseg[i] * 16u;
        wrow_ok[i] = row_base + row < ff;
    }
    const bool ws_on = tid < 128u && row_base + tid < ff;
    const size_t wsoff = (grow0 + 2u * (tid < 128u ? tid : 0u)) * (size_t)n_k16;
    const uint32_t ysrow = tid - 128u;
    const bool ys_on = tid >= 128u && ysrow < BT && tok[ysrow < BT ? ysrow : 0u] != PD_MOE_PAD;
    const size_t ysoff = ys_on ? (size_t)tok[ysrow] * n_k16 : 0u;
    size_t yoff[YI];
    uint32_t ydst[YI], yseg[YI];
    bool y_on[YI];
    #pragma unroll
    for (uint32_t i = 0; i < YI; ++i) {
        const uint32_t u = (tid + NT / 2u) % NT + i * NT, col = u / KB;
        const uint32_t r = u < BT * KB ? tok[col] : PD_MOE_PAD;
        yseg[i] = u % KB;
        ydst[i] = u < BT * KB ? col * YROW + 16u + yseg[i] * 16u : ~0u;
        y_on[i] = r != PD_MOE_PAD;
        yoff[i] = y_on[i] ? (size_t)r * rowb + yseg[i] * 16u : 0u;
    }
    bool tlive[2u * WN / 16u];
    #pragma unroll
    for (uint32_t j = 0; j < 2u * WN / 16u; ++j) tlive[j] = tok[tb + j * 16u + joff] != PD_MOE_PAD;
    auto issue = [&](uint32_t kt) {
        unsigned char* wg = pd_bs_sh + (kt % ST) * SS;
        unsigned char* wu = wg + 128u * WROW;
        unsigned char* yb = wu + 128u * WROW;
        #pragma unroll
        for (uint32_t i = 0; i < WI; ++i) {
            const bool ok = wrow_ok[i] && kt * KB + wseg[i] < n_kb;
            const size_t o = woff[i] + kt * (KB * 16u);
            pd_cp_async16((int*)(wg + wdst[i]), w + o, ok);
            pd_cp_async16((int*)(wu + wdst[i]), w + o + rowb, ok);
        }
        if (tid < 128u) {
            const size_t o = wsoff + kt * (KB * 2u);
            pd_cpa8p(wg + tid * WROW + KB * 16u, wsc + o, ws_on);
            pd_cpa8p(wu + tid * WROW + KB * 16u, wsc + o + n_k16, ws_on);
        } else if (ysrow < BT) {
            pd_cpa8p(yb + ysrow * YROW, xsr + ysoff + kt * (KB * 2u), ys_on);
        }
        #pragma unroll
        for (uint32_t i = 0; i < YI; ++i) {
            if (ydst[i] == ~0u) continue;
            const bool ok = y_on[i] && kt * KB + yseg[i] < n_kb;
            pd_cp_async16((int*)(yb + ydst[i]), xqr + yoff[i] + kt * (KB * 16u), ok);
        }
    };
    float accg[WN / 4u][4] = {};
    float accu[WN / 4u][4] = {};
    #pragma unroll
    for (uint32_t s = 0; s + 1u < ST; ++s) {
        if (s < nk) issue(s);
        asm volatile("cp.async.commit_group;");
    }
    for (uint32_t kt = 0; kt < nk; ++kt) {
        asm volatile("cp.async.wait_group %0;" ::"n"(ST - 2u));
        __syncthreads();
        {
            const uint32_t b0 = kt * (KB * 16u);
            if (b0 % 128u == 0u && b0 + 256u < rowb) prefetch_w(b0 + 256u);
        }
        if (kt + ST - 1u < nk) issue(kt + ST - 1u);
        asm volatile("cp.async.commit_group;");
        const unsigned char* tg = pd_bs_sh + (kt % ST) * SS;
        const unsigned char* tu = tg + 128u * WROW;
        const unsigned char* ty = tu + 128u * WROW;
        #pragma unroll
        for (uint32_t kp = 0; kp < KB / 4u; ++kp) {
            uint32_t amg[2][2][4], amu[2][2][4];
            uint2 sag[2], sau[2];
            #pragma unroll
            for (uint32_t n = 0; n < 2u; ++n) {
                const uint32_t r0 = i0 + n * 16u + g;
                const uint32_t rs = (tq & 1u) ? r0 + 8u : r0;
                const uint32_t wo = (i0 + n * 16u + ((lane >> 3) & 1u) * 8u + (lane & 7u)) * WROW;
                #pragma unroll
                for (uint32_t h = 0; h < 2u; ++h) {
                    const uint32_t k64 = kp * 2u + h;
                    pd_ldm_x4(amg[n][h], tg + wo + k64 * 32u + (lane >> 4) * 16u);
                    pd_ldm_x4(amu[n][h], tu + wo + k64 * 32u + (lane >> 4) * 16u);
                }
                sag[n] = *(const uint2*)(tg + rs * WROW + KB * 16u + kp * 8u);
                sau[n] = *(const uint2*)(tu + rs * WROW + KB * 16u + kp * 8u);
            }
            #pragma unroll
            for (uint32_t j0 = 0; j0 < 2u * WN; j0 += 16u) {
                if (!tlive[j0 / 16u]) continue;
                uint32_t bm[4];
                pd_ldm_x4(bm, ty + (tb + j0 + joff + (lane & 7u)) * YROW + 16u + kp * 64u +
                                  (lane >> 3) * 16u);
                const uint2 sbp = *(const uint2*)(ty + (tb + j0 + joff + g) * YROW + kp * 8u);
                #pragma unroll
                for (uint32_t h = 0; h < 2u; ++h) {
                    const uint32_t sb = h ? sbp.y : sbp.x;
                    #pragma unroll
                    for (uint32_t n = 0; n < 2u; ++n) {
                        pd_nv4_mma(accg[(j0 >> 3) + n], amg[n][h][0], amg[n][h][1], amg[n][h][2],
                                   amg[n][h][3], bm[h * 2u], bm[h * 2u + 1u],
                                   h ? sag[n].y : sag[n].x, sb);
                        pd_nv4_mma(accu[(j0 >> 3) + n], amu[n][h][0], amu[n][h][1], amu[n][h][2],
                                   amu[n][h][3], bm[h * 2u], bm[h * 2u + 1u],
                                   h ? sau[n].y : sau[n].x, sb);
                    }
                }
            }
        }
    }
    // acc[(j0 >> 3) + n][qc + 2h]: ff column row_base + i0 + 16n + g + 8h,
    // block column tb + joff + j0 + 2tq + qc - slot 991's epilogue
    const float fe = wfac[e];
    const float* xor_ = pd_pdl_after(xo);
    #pragma unroll
    for (uint32_t j0 = 0; j0 < 2u * WN; j0 += 16u) {
        #pragma unroll
        for (uint32_t qc = 0; qc < 2u; ++qc) {
            const uint32_t c = tb + joff + j0 + 2u * tq + qc;
            const uint32_t t = tok[c];
            if (t == PD_MOE_PAD) continue;
            const float f = xor_[t] * fe;
            const size_t pair = (size_t)t * topk + slt[c];
            const float gw = A16 ? pd_pdl_after(gamma)[pair] : 1.0f;
            #pragma unroll
            for (uint32_t n = 0; n < 2u; ++n) {
                #pragma unroll
                for (uint32_t h = 0; h < 2u; ++h) {
                    const uint32_t i = row_base + i0 + n * 16u + g + 8u * h;
                    if (i >= ff) continue;
                    const float gv = __bfloat162float(__float2bfloat16_rn(accg[(j0 >> 3) + n][qc + 2u * h] * f));
                    const float uv = __bfloat162float(__float2bfloat16_rn(accu[(j0 >> 3) + n][qc + 2u * h] * f));
                    const float sig = 1.0f / (1.0f + expf(-gv));
                    float v = ((gcap * tanhf(gv / gcap)) * sig) * (ucap * tanhf(uv / ucap));
                    if constexpr (A16) {
                        v *= gw;
                        ((__nv_bfloat16*)act)[pair * ff + i] = __float2bfloat16_rn(v);
                    } else {
                        ((float*)act)[pair * ff + i] = v;
                    }
                }
            }
        }
    }
#else
    (void)w; (void)wsc; (void)wfac; (void)xq; (void)xsc; (void)xo; (void)sorted_row;
    (void)sorted_slot; (void)block_expert; (void)topk; (void)act; (void)gamma; (void)K; (void)ff;
    (void)gcap; (void)ucap;
#endif
}

// the down half: a block's activations (its pairs' rows, whole K) resident,
// RPC 128-row tiles of the expert's down plane streamed through the ring
#define PD_IK2MS_DN_SMEM(BT, ST, ff) \
    ((ST) * 128u * PD_NV4M_WROW(PD_IK2MS_KB) + ((ff) / (PD_IK2MS_KB * 32u)) * (BT) * PD_NV4M_YROW(PD_IK2MS_KB))

template <uint32_t BT, uint32_t WN, uint32_t ST, uint32_t RPC>
__global__ void __launch_bounds__(4u * BT / WN * 32u, 1) pd_inkling2_moe_dn_ms_kernel(
        const uint8_t* __restrict__ w, const uint8_t* __restrict__ wsc,
        const float* __restrict__ wfac, const uint8_t* aq, const uint8_t* asc, const float* ao,
        const uint32_t* sorted_row, const uint32_t* sorted_slot, const uint32_t* block_expert,
        uint32_t topk, float* __restrict__ part, uint32_t ff, uint32_t hidden) {
#if PD_BS_OK
    constexpr uint32_t KB = PD_IK2MS_KB;
    constexpr uint32_t WROW = PD_NV4M_WROW(KB);
    constexpr uint32_t YROW = PD_NV4M_YROW(KB);
    constexpr uint32_t NT = 4u * BT / WN * 32u;
    PD_PDL_ARM();
    const uint32_t n_rt = (hidden + 127u) / 128u;
    const uint32_t n_grp = (n_rt + RPC - 1u) / RPC;
    const uint32_t blk = blockIdx.x / n_grp;
    const uint32_t e = pd_pdl_after(block_expert)[blk];
    if (e == PD_MOE_PAD) return;
    const uint32_t rt0 = (blockIdx.x % n_grp) * RPC;
    const uint32_t nrt = n_rt - rt0 < RPC ? n_rt - rt0 : RPC;
    const uint32_t nk = ff / (KB * 32u);
    const uint32_t steps = nrt * nk;
    extern __shared__ unsigned char pd_bs_sh[];
    unsigned char* yres = pd_bs_sh + ST * 128u * WROW;  // [nk][BT][YROW]
    __shared__ float colf[BT];
    __shared__ uint32_t colp[BT];
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, tq = lane & 3u;
    const uint32_t i0 = ((warp >> 1) & 3u) * 32u;
    const uint32_t joff = (warp & 1u) * 8u;
    const uint32_t tb = (warp >> 3) * (2u * WN);
    const uint32_t n_k16 = ff >> 4, rowb = ff >> 1;
    const uint8_t* aqr = pd_pdl_after(aq);
    const uint8_t* asr = pd_pdl_after(asc);
    if (tid < BT) {
        const uint32_t t = pd_pdl_after(sorted_row)[(size_t)blk * BT + tid];
        if (t == PD_MOE_PAD) {
            colp[tid] = PD_MOE_PAD;
            colf[tid] = 0.0f;
        } else {
            const uint32_t p = t * topk + pd_pdl_after(sorted_slot)[(size_t)blk * BT + tid];
            colp[tid] = p;
            colf[tid] = pd_pdl_after(ao)[p] * wfac[e];
        }
    }
    __syncthreads();
    // the block's pair rows, whole K: chunk kt of column c at yres + (kt * BT + c) * YROW
    for (uint32_t u = tid; u < BT * nk * KB; u += NT) {
        const uint32_t c = u / (nk * KB), r = u % (nk * KB), kt = r / KB, seg = r % KB;
        const uint32_t p = colp[c];
        const bool ok = p != PD_MOE_PAD;
        pd_cp_async16((int*)(yres + (kt * BT + c) * YROW + 16u + seg * 16u),
                      aqr + (size_t)(ok ? p : 0u) * rowb + kt * (KB * 16u) + seg * 16u, ok);
    }
    for (uint32_t u = tid; u < BT * nk; u += NT) {
        const uint32_t c = u / nk, kt = u % nk;
        const uint32_t p = colp[c];
        const bool ok = p != PD_MOE_PAD;
        pd_cpa8p(yres + (kt * BT + c) * YROW, asr + (size_t)(ok ? p : 0u) * n_k16 + kt * (KB * 2u), ok);
    }
    auto issue = [&](uint32_t st) {
        unsigned char* wb = pd_bs_sh + (st % ST) * (128u * WROW);
        const uint32_t rb = (rt0 + st / nk) * 128u, kt = st % nk;
        const size_t wrow = (size_t)e * hidden + rb;
        for (uint32_t u = tid; u < 128u * KB; u += NT) {
            const uint32_t row = u / KB, seg = u % KB;
            const bool ok = rb + row < hidden;
            pd_cp_async16((int*)(wb + row * WROW + seg * 16u),
                          w + (wrow + (ok ? row : 0u)) * (size_t)rowb + kt * (KB * 16u) + seg * 16u, ok);
        }
        if (tid < 128u) {
            const bool ok = rb + tid < hidden;
            pd_cpa8p(wb + tid * WROW + KB * 16u,
                     wsc + (wrow + (ok ? tid : 0u)) * (size_t)n_k16 + kt * (KB * 2u), ok);
        }
    };
    auto prefetch_rt = [&](uint32_t rt) {
        const size_t wrow = (size_t)e * hidden + rt * 128u;
        const uint32_t nrow = hidden - rt * 128u < 128u ? hidden - rt * 128u : 128u;
        const uint32_t db = nrow * rowb, sb = nrow * n_k16;
        for (uint32_t o = tid * 128u; o < db + sb; o += NT * 128u) {
            const uint8_t* q = o < db ? w + wrow * rowb + o : wsc + wrow * n_k16 + (o - db);
            asm volatile("prefetch.global.L2 [%0];" ::"l"(q));
        }
    };
    if (nrt > 1u) prefetch_rt(rt0 + 1u);
    #pragma unroll
    for (uint32_t s = 0; s + 1u < ST; ++s) {
        if (s < steps) issue(s);
        asm volatile("cp.async.commit_group;");
    }
    bool tlive[2u * WN / 16u];
    #pragma unroll
    for (uint32_t j = 0; j < 2u * WN / 16u; ++j) tlive[j] = colp[tb + j * 16u + joff] != PD_MOE_PAD;
    float acc[WN / 4u][4] = {};
    for (uint32_t st = 0; st < steps; ++st) {
        asm volatile("cp.async.wait_group %0;" ::"n"(ST - 2u));
        __syncthreads();
        const uint32_t kt = st % nk, rt = rt0 + st / nk;
        if (kt == 0u && st / nk + 2u < nrt) prefetch_rt(rt + 2u);
        if (st + ST - 1u < steps) issue(st + ST - 1u);
        asm volatile("cp.async.commit_group;");
        const unsigned char* tw = pd_bs_sh + (st % ST) * (128u * WROW);
        const unsigned char* ty = yres + kt * BT * YROW;
        #pragma unroll
        for (uint32_t kp = 0; kp < KB / 4u; ++kp) {
            uint32_t am[2][2][4];
            uint2 sa[2];
            #pragma unroll
            for (uint32_t n = 0; n < 2u; ++n) {
                const uint32_t r0 = i0 + n * 16u + g;
                const uint32_t rs = (tq & 1u) ? r0 + 8u : r0;
                const uint32_t wo = (i0 + n * 16u + ((lane >> 3) & 1u) * 8u + (lane & 7u)) * WROW;
                #pragma unroll
                for (uint32_t h = 0; h < 2u; ++h)
                    pd_ldm_x4(am[n][h], tw + wo + (kp * 2u + h) * 32u + (lane >> 4) * 16u);
                sa[n] = *(const uint2*)(tw + rs * WROW + KB * 16u + kp * 8u);
            }
            #pragma unroll
            for (uint32_t j0 = 0; j0 < 2u * WN; j0 += 16u) {
                if (!tlive[j0 / 16u]) continue;
                uint32_t bm[4];
                pd_ldm_x4(bm, ty + (tb + j0 + joff + (lane & 7u)) * YROW + 16u + kp * 64u +
                                  (lane >> 3) * 16u);
                const uint2 sbp = *(const uint2*)(ty + (tb + j0 + joff + g) * YROW + kp * 8u);
                #pragma unroll
                for (uint32_t h = 0; h < 2u; ++h) {
                    #pragma unroll
                    for (uint32_t n = 0; n < 2u; ++n)
                        pd_nv4_mma(acc[(j0 >> 3) + n], am[n][h][0], am[n][h][1], am[n][h][2],
                                   am[n][h][3], bm[h * 2u], bm[h * 2u + 1u],
                                   h ? sa[n].y : sa[n].x, h ? sbp.y : sbp.x);
                }
            }
        }
        if (kt + 1u == nk) {
            // the row tile is whole: slot 992's epilogue
            #pragma unroll
            for (uint32_t j0 = 0; j0 < 2u * WN; j0 += 16u) {
                #pragma unroll
                for (uint32_t qc = 0; qc < 2u; ++qc) {
                    const uint32_t c = tb + j0 + joff + 2u * tq + qc;
                    const uint32_t p = colp[c];
                    if (p == PD_MOE_PAD) continue;
                    const float f = colf[c];
                    float* dst = part + (size_t)p * hidden;
                    #pragma unroll
                    for (uint32_t n = 0; n < 2u; ++n) {
                        #pragma unroll
                        for (uint32_t h = 0; h < 2u; ++h) {
                            const uint32_t o = rt * 128u + i0 + n * 16u + g + 8u * h;
                            if (o < hidden) dst[o] = acc[(j0 >> 3) + n][qc + 2u * h] * f;
                        }
                    }
                }
            }
            #pragma unroll
            for (uint32_t j = 0; j < WN / 4u; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;
        }
    }
#else
    (void)w; (void)wsc; (void)wfac; (void)aq; (void)asc; (void)ao; (void)sorted_row;
    (void)sorted_slot; (void)block_expert; (void)topk; (void)part; (void)ff; (void)hidden;
#endif
}

template <bool A16>
static int pd_inkling2_moe_gu_ms_impl(const void* w, const void* wsc, const void* wfac,
                                      const void* xq, const void* xsc, const void* xo,
                                      const void* sorted_row, const void* sorted_slot,
                                      const void* block_expert, uint32_t max_blocks, uint32_t topk,
                                      void* act, const void* gamma, uint32_t K, uint32_t ff,
                                      float gcap, float ucap, void* stream) {
#ifndef PD_BS_HOST
    (void)w; (void)wsc; (void)wfac; (void)xq; (void)xsc; (void)xo; (void)sorted_row;
    (void)sorted_slot; (void)block_expert; (void)max_blocks; (void)topk; (void)act; (void)gamma;
    (void)K; (void)ff; (void)gcap; (void)ucap; (void)stream;
    return cudaErrorNotSupported;
#else
    if (max_blocks == 0) return 0;
    if (pd_nv4t_arm()) return cudaErrorNotSupported;
    if ((K % (PD_IK2MS_KB * 32u)) || (ff % 16u) || !wfac || !xo || (A16 && !gamma))
        return cudaErrorInvalidValue;
    constexpr uint32_t BT = 64u, WN = 16u, ST = 3u;
    auto k = pd_inkling2_moe_gu_ms_kernel<BT, WN, ST, A16>;
    constexpr uint32_t smem = PD_IK2MS_GU_SMEM(BT, ST);
    static const bool attr =
        cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) ==
            cudaSuccess ||
        (cudaGetLastError(), false);
    if (!attr) return cudaErrorInvalidConfiguration;
    pd_pdl_go(k, max_blocks * ((ff + 127u) / 128u), 4u * BT / WN * 32u, smem, (cudaStream_t)stream,
              (const uint8_t*)w, (const uint8_t*)wsc, (const float*)wfac, (const uint8_t*)xq,
              (const uint8_t*)xsc, (const float*)xo, (const uint32_t*)sorted_row,
              (const uint32_t*)sorted_slot, (const uint32_t*)block_expert, topk, act,
              (const float*)gamma, K, ff, gcap, ucap);
    return pd_launch_status();
#endif
}

PD_EXPORT
int pd_inkling2_moe_gu_ms(const void* w, const void* wsc, const void* wfac, const void* xq,
                          const void* xsc, const void* xo, const void* sorted_row,
                          const void* sorted_slot, const void* block_expert, uint32_t max_blocks,
                          uint32_t topk, void* act, uint32_t K, uint32_t ff, float gcap, float ucap,
                          void* stream) {
    return pd_inkling2_moe_gu_ms_impl<false>(w, wsc, wfac, xq, xsc, xo, sorted_row, sorted_slot,
                                             block_expert, max_blocks, topk, act, nullptr, K, ff,
                                             gcap, ucap, stream);
}

// slot 1006: slot 1002 with act as bf16(act x gamma[pair]) (gamma the
// router weights, [pairs])
PD_EXPORT
int pd_inkling2_moe_gu_ms16(const void* w, const void* wsc, const void* wfac, const void* xq,
                            const void* xsc, const void* xo, const void* sorted_row,
                            const void* sorted_slot, const void* block_expert, uint32_t max_blocks,
                            uint32_t topk, void* act16, const void* gamma, uint32_t K, uint32_t ff,
                            float gcap, float ucap, void* stream) {
    return pd_inkling2_moe_gu_ms_impl<true>(w, wsc, wfac, xq, xsc, xo, sorted_row, sorted_slot,
                                            block_expert, max_blocks, topk, act16, gamma, K, ff,
                                            gcap, ucap, stream);
}

PD_EXPORT
int pd_inkling2_moe_dn_ms(const void* w, const void* wsc, const void* wfac, const void* aq,
                          const void* asc, const void* ao, const void* sorted_row,
                          const void* sorted_slot, const void* block_expert, uint32_t max_blocks,
                          uint32_t topk, void* part, uint32_t ff, uint32_t hidden, void* stream) {
#ifndef PD_BS_HOST
    (void)w; (void)wsc; (void)wfac; (void)aq; (void)asc; (void)ao; (void)sorted_row;
    (void)sorted_slot; (void)block_expert; (void)max_blocks; (void)topk; (void)part; (void)ff;
    (void)hidden; (void)stream;
    return cudaErrorNotSupported;
#else
    if (max_blocks == 0) return 0;
    if (pd_nv4t_arm()) return cudaErrorNotSupported;
    if ((ff % (PD_IK2MS_KB * 32u)) || (hidden % 16u) || !wfac || !ao) return cudaErrorInvalidValue;
    constexpr uint32_t BT = 64u, WN = 16u, ST = 3u, RPC = 6u;
    auto k = pd_inkling2_moe_dn_ms_kernel<BT, WN, ST, RPC>;
    // the resident columns grow with ff: past 1664 (the opt-in less the
    // static bytes and the system's reserved 1 KB) the shape is not this
    // slot's. The attribute once, at the most any ff may take (decode ticks
    // wider than the GEMVs launch this inside a capture); a refused set
    // clears its error so the next launch does not report it
    constexpr uint32_t dyn_max = PD_IK2MS_DN_SMEM(BT, ST, 1664u);
    const uint32_t smem = PD_IK2MS_DN_SMEM(BT, ST, ff);
    if (smem > dyn_max) return cudaErrorNotSupported;
    static const bool attr =
        cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn_max) ==
            cudaSuccess ||
        (cudaGetLastError(), false);
    if (!attr) return cudaErrorInvalidConfiguration;
    const uint32_t n_grp = ((hidden + 127u) / 128u + RPC - 1u) / RPC;
    pd_pdl_go(k, max_blocks * n_grp, 4u * BT / WN * 32u, smem, (cudaStream_t)stream,
              (const uint8_t*)w, (const uint8_t*)wsc, (const float*)wfac, (const uint8_t*)aq,
              (const uint8_t*)asc, (const float*)ao, (const uint32_t*)sorted_row,
              (const uint32_t*)sorted_slot, (const uint32_t*)block_expert, topk, (float*)part, ff,
              hidden);
    return pd_launch_status();
#endif
}
