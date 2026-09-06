#include <cuda_bf16.h>
// moe/f8.cuh (formerly 21_moe_f8.cuh) - tcgen05 e4m3 grouped/sorted MoE GEMMs (gemma4-A4B expert
// class, sm_100a). Textually-included segment of the single pack translation
// unit; not standalone-compilable (needs 03's PD_MOE_PAD, 12's e4m3 quant
// helpers, 14's tc5 descriptors + tmap builders - pack.cu include order).
//
// Why: the s8 `mma.sync` sorted pair is
// the sm_80/86/89 class - on sm_100a it measured ~10-20% of tensor
// throughput (31% of pf8 GPU time). This family reruns the experts on the
// die's real pipe: the same tc5bs slab ring as the dense f8w prefill GEMM
// (async-SF v2, hardware ue8m0 fold), with two deltas:
//   - the W row window comes from the sorted layout's per-block expert id
//     (block_expert) into one flat weight stream (rows at e*rows_per_e + r),
//     so a single 2D tensor map serves all 128 experts;
//   - the activation side is PRE-GATHERED into the sorted row order
//     (pd_moe_gather_e4m3), which makes the Y side a plain dense tmap -
//     no gather inside the GEMM.
// Layout choices that make the tiles land exactly:
//   - gate/up stay FUSED per expert ([gate|up] = 1408 rows = 11 x 128 tiles
//     - the split-planes' 704 would be ragged);
//   - the down K pads 704 -> 768 at CONVERSION (pd_q8_0_to_f8w_pad, zero
//     blocks: 0 x anything accumulates exactly 0);
//   - sorted blocks are BM=128 (moe_align_bm 128) so a block is a Y tile.
// e4m3 expert weights are a precision-class change vs the Q8 originals -
// the coherence/greedy-basin gates arbitrate, same as the dense f8t default.

// ---- Q8_0 -> per-32 e4m3 planes with K-padding ----------------------------
// Same scale pick + RN-even encode as pd_q8_0_to_f8w; row r's blocks land at
// r*bpr_pad, tail blocks [bpr, bpr_pad) written as zeros (data AND scale).
__global__ void pd_q8_0_to_f8w_pad_kernel(const int8_t* __restrict__ q8,
                                          const __half* __restrict__ s8,
                                          unsigned char* __restrict__ data,
                                          unsigned char* __restrict__ scale,
                                          uint64_t rows, uint32_t bpr,
                                          uint32_t bpr_pad) {
    const uint64_t blk = blockIdx.x;         // padded block index
    const uint32_t d = threadIdx.x;
    if (blk >= rows * bpr_pad) return;
    const uint64_t r = blk / bpr_pad;
    const uint32_t j = (uint32_t)(blk - r * bpr_pad);
    if (j >= bpr) {
        data[blk * 32u + d] = 0u;
        if (d == 0) scale[blk] = 0u;
        return;
    }
    const uint64_t src = r * bpr + j;
    float v = (float)q8[src * 32u + d] * __half2float(s8[src]);
    float a = fabsf(v);
    for (uint32_t off = 16; off > 0; off >>= 1)
        a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, off));
    int e = 0;
    if (a > 0.0f) {
        int ex;
        float m = frexpf(a, &ex);
        e = ex - 9 + (m > 0.875f ? 1 : 0);
    }
    data[blk * 32u + d] = __nv_fp8_e4m3(v * ldexpf(1.0f, -e)).__x;
    if (d == 0) scale[blk] = (unsigned char)(e + 127);
}

PD_EXPORT
int pd_q8_0_to_f8w_pad(const void* q8_data, const void* q8_scale, void* f8_data,
                       void* f8_scale, uint64_t rows, uint32_t bpr,
                       uint32_t bpr_pad, void* stream) {
#ifndef PD_BS_HOST
    (void)q8_data; (void)q8_scale; (void)f8_data; (void)f8_scale; (void)rows;
    (void)bpr; (void)bpr_pad; (void)stream;
    return cudaErrorNotSupported;
#else
    if (rows == 0) return 0;
    if (bpr_pad < bpr) return cudaErrorInvalidValue;
    pd_q8_0_to_f8w_pad_kernel<<<(uint32_t)(rows * bpr_pad), 32, 0,
                                (cudaStream_t)stream>>>(
        (const int8_t*)q8_data, (const __half*)q8_scale, (unsigned char*)f8_data,
        (unsigned char*)f8_scale, rows, bpr, bpr_pad);
    return pd_launch_status();
#endif
}

// ---- sorted gather of e4m3 activations + ue8m0 scales ---------------------
// xg[i] = xq[srow[i]] (PAD -> zeros), 128 sorted rows per block row-tile.
// One CTA per sorted row; data as u32 words (in_dim % 4 == 0), scales bytes.
__global__ void pd_moe_gather_e4m3_kernel(const unsigned char* __restrict__ xq,
                                          const unsigned char* __restrict__ xs,
                                          const unsigned int* __restrict__ srow,
                                          unsigned char* __restrict__ xg,
                                          unsigned char* __restrict__ sg,
                                          uint32_t in_dim) {
    const uint32_t i = blockIdx.x;
    const unsigned int r = srow[i];
    const bool live = r != PD_MOE_PAD;
    const uint32_t nw = in_dim >> 2, nb = in_dim >> 5;
    const uint32_t* src = (const uint32_t*)(xq + (size_t)(live ? r : 0u) * in_dim);
    uint32_t* dst = (uint32_t*)(xg + (size_t)i * in_dim);
    for (uint32_t w = threadIdx.x; w < nw; w += blockDim.x)
        dst[w] = live ? src[w] : 0u;
    for (uint32_t b = threadIdx.x; b < nb; b += blockDim.x)
        sg[(size_t)i * nb + b] = live ? xs[(size_t)r * nb + b] : 0u;
}

PD_EXPORT
int pd_moe_gather_e4m3(const void* xq, const void* xs, const void* srow, void* xg,
                       void* sg, uint32_t in_dim, uint32_t srows, void* stream) {
    if (srows == 0) return 0;
    if ((in_dim & 31u) != 0) return cudaErrorInvalidValue;
    pd_moe_gather_e4m3_kernel<<<srows, 128, 0, (cudaStream_t)stream>>>(
        (const unsigned char*)xq, (const unsigned char*)xs, (const unsigned int*)srow,
        (unsigned char*)xg, (unsigned char*)sg, in_dim);
    return pd_launch_status();
}

// ---- fused-plane GEGLU quantize with padded output stride -----------------
// gu rows are [gate(n_ff) | up(n_ff)]; output rows are n_ff_pad wide with
// only [0, n_ff) written - the K-pad tail stays whatever the buffer holds
// (the caller allocs zeros once and nothing else writes the plane, so the
// zero tail is standing). Same formula + pd_e4m3_quant4 as
// pd_quantize_e4m3_geglu2 -> identical values on the live region.
__global__ void pd_quantize_e4m3_geglu2_pad_kernel(const float* __restrict__ gu,
                                                   unsigned char* __restrict__ q,
                                                   unsigned char* __restrict__ scale,
                                                   uint32_t n_ff, uint32_t n_ff_pad,
                                                   uint32_t n) {
    const uint32_t i = (blockIdx.x * 256u + threadIdx.x) * 4u;
    if (i >= n) return;
    const uint32_t row = i / n_ff, col = i - row * n_ff;
    const float* base = gu + (size_t)row * 2u * n_ff + col;
    const float4 g = *(const float4*)base;
    const float4 u = *(const float4*)(base + n_ff);
    float4 v;
    v.x = 0.5f * g.x * (1.0f + tanhf(0.79788456080286535587989211986876f * g.x
                                     * (1.0f + 0.044715f * g.x * g.x))) * u.x;
    v.y = 0.5f * g.y * (1.0f + tanhf(0.79788456080286535587989211986876f * g.y
                                     * (1.0f + 0.044715f * g.y * g.y))) * u.y;
    v.z = 0.5f * g.z * (1.0f + tanhf(0.79788456080286535587989211986876f * g.z
                                     * (1.0f + 0.044715f * g.z * g.z))) * u.z;
    v.w = 0.5f * g.w * (1.0f + tanhf(0.79788456080286535587989211986876f * g.w
                                     * (1.0f + 0.044715f * g.w * g.w))) * u.w;
    pd_e4m3_quant4(v, threadIdx.x & 7u, q, scale, row * n_ff_pad + col);
}

PD_EXPORT
int pd_quantize_e4m3_geglu2_pad(const void* gu, void* q, void* scale, uint32_t n_ff,
                                uint32_t n_ff_pad, uint32_t rows, void* stream) {
    const uint32_t n = n_ff * rows;
    if (n == 0) return 0;
    if ((n_ff & 31u) != 0 || (n_ff_pad & 31u) != 0) return cudaErrorInvalidValue;
    pd_quantize_e4m3_geglu2_pad_kernel<<<(n / 4u + 255u) / 256u, 256u, 0,
                                         (cudaStream_t)stream>>>(
        (const float*)gu, (unsigned char*)q, (unsigned char*)scale, n_ff, n_ff_pad, n);
    return pd_launch_status();
}

// ---- grouped tc5bs GEMMs ---------------------------------------------------
// Clones of pd_f8bs_gemm_tc5_kt (same slab ring, async-SF v2, hw ue8m0 fold;
// see its header note) with the expert indirection. DN adds the scattered
// weighted epilogue (part[(token,slot)] = topk_w x val) so the down output
// never lands as a dense sorted plane.
// NT = W out-tiles per CTA sharing one Y stage (the grouped twin of the
// dense tc5w widening, on the other axis: an expert's 11/22 out-tiles all
// consume the same 128 sorted tokens, so sharing the Y stage cuts its
// re-read traffic /NT - Y was ~2/3 of the gu bytes). NT=1 keeps the
// original single-tile schedule bit-for-bit.
// BN = sorted rows per block (the mma N). 128 is the prefill-band shape
// (block is a full Y tile); 32 is the decode-band shape (BM=32
// blocks - 4KB Y stages, 64-col tmem alloc, quarter the PAD flops at
// r*k < 512 where 128-row blocks are mostly padding). All BN uses fold at
// compile time, so BN=128 instantiations keep the shipped PTX.
template <uint32_t S, bool DN, uint32_t NT = 1u, uint32_t BN = 128u>
__global__ void __launch_bounds__(128) pd_f8bs_moe_tc5_kt(
    const __grid_constant__ PdTmap wmap, const __grid_constant__ PdTmap ymap,
    const unsigned char* __restrict__ wsc, const unsigned char* __restrict__ xsc,
    const unsigned int* __restrict__ bexp, const unsigned int* __restrict__ srow,
    const unsigned int* __restrict__ sslot, const float* __restrict__ topk_w,
    float* __restrict__ y, uint32_t in_dim, uint32_t rows_per_e, uint32_t srows_pad,
    uint32_t w_rows, uint32_t n_active) {
#if PD_TC5_OK
    const uint32_t not_ = ((rows_per_e >> 7) + NT - 1u) / NT; // ceil tile groups
    const uint32_t blk = blockIdx.x / not_;
    const uint32_t ot = (blockIdx.x - blk * not_) * NT;
    const unsigned int e = bexp[blk];
    if (e == PD_MOE_PAD) return;   // uniform per CTA - before any alloc

    extern __shared__ __align__(1024) unsigned char pd_m8_sh[];
    constexpr uint32_t YB = BN * 128u;                   // Y stage bytes
    unsigned char* wt = pd_m8_sh;                        // S x NT x 16KB
    unsigned char* yt = pd_m8_sh + S * NT * 16384u;      // S x YB
    unsigned char* sfs = pd_m8_sh + S * (NT * 16384u + YB); // S x (NT*512 SFA | 512 SFB)
    uint64_t* bfull = (uint64_t*)(sfs + S * (NT + 1u) * 512u);
    uint64_t* bdone = bfull + S;
    __shared__ uint32_t tmem_slot[1];

    const uint32_t tid = threadIdx.x;
    const uint32_t nk = (in_dim + 127u) / 128u;
    const uint32_t n_kb = in_dim >> 5;
    const uint32_t row_base = e * rows_per_e + ot * 128u;   // W rows (flat stream)
    const uint32_t col_base = blk * BN;                      // sorted Y rows

    if (tid == 0) {
        #pragma unroll
        for (uint32_t s = 0; s < S; ++s) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 129;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bfull[s])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bdone[s])));
        }
    }
    __syncthreads();
    // tmem: NT x BN D cols + SF ring, rounded to the allocation granule
    // (power of 2, >= 32). NT=1/BN=128 must stay at 256 - a blanket 512
    // halves tcgen05 CTA co-residency and collapsed the base path (
    // regression, caught by the A/B baseline: c8 1012 -> 515). BN=32 fits
    // D(32) + SF ring(<=24) in 64 - the decode shape's co-residency edge.
    constexpr uint32_t TCOLS = (NT == 1u) ? (BN == 32u ? 64u : 256u) : 512u;
    if (tid < 32)
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                     ::"r"((uint32_t)__cvta_generic_to_shared(tmem_slot)), "r"(TCOLS));
    __syncthreads();
    const uint32_t tmem = tmem_slot[0];              // D: NT x BN cols
    const uint32_t sf_base = tmem + NT * BN;

    auto bar_wait = [&](uint64_t* bar, uint32_t parity) {
        const uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
        asm volatile("{\n\t.reg .pred p;\nW%=:\n\t"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                     "@!p bra W%=;\n\t}" ::"r"(a), "r"(parity));
    };
    auto tma_stage = [&](uint32_t kt, uint32_t s) {
        const uint32_t m = (uint32_t)__cvta_generic_to_shared(&bfull[s]);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                     ::"r"(m), "r"(NT * 16384u + YB));
        const int ck = (int)(kt * 128u);
        #pragma unroll
        for (uint32_t t = 0; t < NT; ++t) {
            const uint32_t wd =
                (uint32_t)__cvta_generic_to_shared(wt + (s * NT + t) * 16384u);
            asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                         " [%0], [%1, {%2, %3}], [%4];" ::"r"(wd), "l"(&wmap), "r"(ck),
                         "r"((int)(row_base + t * 128u)), "r"(m) : "memory");
        }
        const uint32_t yd = (uint32_t)__cvta_generic_to_shared(yt + s * YB);
        asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                     " [%0], [%1, {%2, %3}], [%4];" ::"r"(yd), "l"(&ymap), "r"(ck),
                     "r"((int)col_base), "r"(m) : "memory");
    };
    auto sf_stage = [&](uint32_t kt, uint32_t s) {
        const uint32_t kb0 = kt * 4u;
        unsigned char* base = sfs + s * (NT + 1u) * 512u;
        const uint32_t off = (tid % 32u) * 16u + (tid / 32u) * 4u;
        const uint32_t rc = col_base + tid;
        #pragma unroll
        for (uint32_t t = 0; t < NT; ++t) {
            const uint32_t rw = row_base + t * 128u + tid;
            pd_mma_cpa4p(base + t * 512u + off, wsc + (size_t)rw * n_kb + kb0,
                         rw < w_rows && kb0 + 4u <= n_kb);
        }
        pd_mma_cpa4p(base + NT * 512u + off, xsc + (size_t)rc * n_kb + kb0,
                     rc < srows_pad && kb0 + 4u <= n_kb);
        const uint32_t m = (uint32_t)__cvta_generic_to_shared(&bfull[s]);
        asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(m) : "memory");
    };

    #pragma unroll
    for (uint32_t s = 0; s < S; ++s) {
        if (s < nk) {
            sf_stage(s, s);
            if (tid == 0) tma_stage(s, s);
        }
    }
    uint32_t fph = 0, dph = 0;
    for (uint32_t kt = 0; kt < nk; ++kt) {
        const uint32_t s = kt % S;
        if (tid == 0) {
            bar_wait(&bfull[s], (fph >> s) & 1u);
            fph ^= 1u << s;
            const uint32_t v =
                (uint32_t)__cvta_generic_to_shared(sfs + s * (NT + 1u) * 512u) >> 4;
            const uint64_t db = ((uint64_t)((v + NT * 32u) & 0x3FFFu))
                              | ((uint64_t)1u << 16) | ((uint64_t)8u << 32);
            const uint32_t sfa0 = sf_base + s * 4u * (NT + 1u);
            const uint32_t sfb_t = sfa0 + NT * 4u;
            #pragma unroll
            for (uint32_t t = 0; t < NT; ++t) {
                const uint64_t da = ((uint64_t)((v + t * 32u) & 0x3FFFu))
                                  | ((uint64_t)1u << 16) | ((uint64_t)8u << 32);
                asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                             ::"r"(sfa0 + t * 4u), "l"(da) : "memory");
            }
            asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                         ::"r"(sfb_t), "l"(db) : "memory");
            const uint32_t y16 = (uint32_t)__cvta_generic_to_shared(yt + s * YB) >> 4;
            #pragma unroll
            for (uint32_t t = 0; t < NT; ++t) {
                const uint32_t w16 =
                    (uint32_t)__cvta_generic_to_shared(wt + (s * NT + t) * 16384u) >> 4;
                #pragma unroll
                for (uint32_t kb = 0; kb < 4u; ++kb) {
                    const uint64_t ad = pd_tc5_sdesc(w16 + kb * 2u);
                    const uint64_t bd = pd_tc5_sdesc(y16 + kb * 2u);
                    const uint32_t en = (kt > 0 || kb > 0) ? 1u : 0u;
                    asm volatile(
                        "{\n\t.reg .pred p;\n\t"
                        "setp.ne.b32 p, %6, 0;\n\t"
                        "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X"
                        " [%0], %1, %2, %3, [%4], [%5], p;\n\t}"
                        ::"r"(tmem + t * BN), "l"(ad), "l"(bd),
                          "r"(pd_tc5_bs_idesc_bn(kb, BN)),
                          "r"(sfa0 + t * 4u), "r"(sfb_t), "r"(en));
                }
            }
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                ::"r"((uint32_t)__cvta_generic_to_shared(&bdone[s])));
        }
        bar_wait(&bdone[s], (dph >> s) & 1u);
        dph ^= 1u << s;
        const uint32_t pf = kt + S;
        if (pf < nk) {
            sf_stage(pf, s);
            if (tid == 0) tma_stage(pf, s);
        }
    }
    __syncthreads();
    #pragma unroll
    for (uint32_t t = 0; t < NT; ++t)
    #pragma unroll
    for (uint32_t cc = 0; cc < BN / 32u; ++cc) {
        uint32_t r[32];
        const uint32_t warp = tid >> 5, lane = tid & 31u;
        const uint32_t taddr = tmem + t * BN + ((warp * 32u) << 16) + cc * 32u;
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
            "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
            : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
              "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),"=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15]),
              "=r"(r[16]),"=r"(r[17]),"=r"(r[18]),"=r"(r[19]),"=r"(r[20]),"=r"(r[21]),"=r"(r[22]),"=r"(r[23]),
              "=r"(r[24]),"=r"(r[25]),"=r"(r[26]),"=r"(r[27]),"=r"(r[28]),"=r"(r[29]),"=r"(r[30]),"=r"(r[31])
            : "r"(taddr));
        asm volatile("tcgen05.wait::ld.sync.aligned;");
        const uint32_t local = (ot + t) * 128u + warp * 32u + lane; // row within expert
        if (local >= rows_per_e) continue;   // NT remainder: phantom tile, no store
        #pragma unroll
        for (uint32_t j = 0; j < 32u; ++j) {
            const uint32_t col = col_base + cc * 32u + j;       // sorted row
            if (DN) {
                const unsigned int tok = srow[col];
                if (tok == PD_MOE_PAD) continue;
                const unsigned int sl = sslot[col];
                const float w = topk_w[(size_t)tok * n_active + sl];
                y[((size_t)tok * n_active + sl) * rows_per_e + local] =
                    w * __uint_as_float(r[j]);
            } else {
                y[(size_t)col * rows_per_e + local] = __uint_as_float(r[j]);
            }
        }
    }
    __syncthreads();
    if (tid < 32)
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem),
                     "r"(TCOLS));
#else
    (void)wmap; (void)ymap; (void)wsc; (void)xsc; (void)bexp; (void)srow;
    (void)sslot; (void)topk_w; (void)y; (void)in_dim; (void)rows_per_e;
    (void)srows_pad; (void)w_rows; (void)n_active;
#endif  // PD_TC5_OK
}

// One launcher serves both halves (DN switches the epilogue). Requires the
// tc5 route (cc 10 + driver tmap encode) - callers fall back to the s8-mma
// sorted pair when this returns cudaErrorNotSupported.
static int pd_f8bs_moe_launch(bool dn, const void* wdata, const void* wsc,
                              const void* xg, const void* sg, const void* bexp,
                              const void* srow, const void* sslot, const void* topk_w,
                              void* y, uint32_t in_dim, uint32_t rows_per_e,
                              uint32_t n_expert, uint32_t srows_pad,
                              uint32_t max_blocks, uint32_t n_active, void* stream) {
#if !defined(PD_BS_HOST) || !defined(PD_TC5_HOST)
    (void)dn; (void)wdata; (void)wsc; (void)xg; (void)sg; (void)bexp; (void)srow;
    (void)sslot; (void)topk_w; (void)y; (void)in_dim; (void)rows_per_e;
    (void)n_expert; (void)srows_pad; (void)max_blocks; (void)n_active; (void)stream;
    return cudaErrorNotSupported;
#else
    static const bool tc5 = [] {
        int dev = 0, cc = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&cc, cudaDevAttrComputeCapabilityMajor, dev);
        return cc == 10 && pd_tmap_encode() != nullptr
            && pd_env("PADDOCK_NO_TC5") == nullptr;
    }();
    if (!tc5) return cudaErrorNotSupported;
    if ((in_dim & 127u) != 0 || (rows_per_e & 127u) != 0) return cudaErrorInvalidValue;
    if (max_blocks == 0) return 0;
    CUtensorMap wm, ym;
    if (!pd_tmap_2d(&wm, wdata, in_dim, n_expert * rows_per_e) ||
        !pd_tmap_2d(&ym, xg, in_dim, srows_pad))
        return cudaErrorNotSupported;
    constexpr uint32_t S = 3u;
    // NT=2 share-Y widening - OPT-IN only (PADDOCK_MOE_TC5W=1). Board
    // FALSIFIED the S=3/NT=2 geometry: pf8 -6.3% (589.4 ->
    // 552.4), c32/c8 ~-1%. The smem doubling (99 -> 148KB) halves CTAs/SM
    // and the occupancy loss beats the Y-traffic savings (~1/3 of gu
    // bytes). Output-IDENTICAL to NT=1, so the kernel stays for an S=2
    // retune (98KB keeps 2 CTA/SM) - measure before any default flip.
    static const bool wide_ok = pd_env("PADDOCK_MOE_TC5W") != nullptr;
    const uint32_t tiles = rows_per_e >> 7;
    const uint32_t nt = (wide_ok && tiles >= 2u) ? 2u : 1u;
    const uint32_t smem = (nt + 1u) * S * 16384u + S * (nt + 1u) * 512u + 2u * S * 8u;
    static bool at[2][2] = {{false, false}, {false, false}};
    const uint32_t ai = nt - 1u;
    if (!at[dn ? 1 : 0][ai]) {
        const void* f = dn ? (nt == 2u ? (const void*)pd_f8bs_moe_tc5_kt<S, true, 2u>
                                       : (const void*)pd_f8bs_moe_tc5_kt<S, true, 1u>)
                           : (nt == 2u ? (const void*)pd_f8bs_moe_tc5_kt<S, false, 2u>
                                       : (const void*)pd_f8bs_moe_tc5_kt<S, false, 1u>);
        cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        at[dn ? 1 : 0][ai] = true;
    }
    const uint32_t grid = max_blocks * ((tiles + nt - 1u) / nt);
    const unsigned char* wscp = (const unsigned char*)wsc;
    const unsigned char* sgp = (const unsigned char*)sg;
    const unsigned int* bep = (const unsigned int*)bexp;
    const unsigned int* srp = (const unsigned int*)srow;
    const unsigned int* slp = (const unsigned int*)sslot;
    const float* twp = (const float*)topk_w;
    if (dn) {
        if (nt == 2u)
            pd_f8bs_moe_tc5_kt<S, true, 2u><<<grid, 128, smem, (cudaStream_t)stream>>>(
                wm, ym, wscp, sgp, bep, srp, slp, twp, (float*)y, in_dim, rows_per_e,
                srows_pad, n_expert * rows_per_e, n_active);
        else
            pd_f8bs_moe_tc5_kt<S, true, 1u><<<grid, 128, smem, (cudaStream_t)stream>>>(
                wm, ym, wscp, sgp, bep, srp, slp, twp, (float*)y, in_dim, rows_per_e,
                srows_pad, n_expert * rows_per_e, n_active);
    } else {
        if (nt == 2u)
            pd_f8bs_moe_tc5_kt<S, false, 2u><<<grid, 128, smem, (cudaStream_t)stream>>>(
                wm, ym, wscp, sgp, bep, srp, slp, twp, (float*)y, in_dim, rows_per_e,
                srows_pad, n_expert * rows_per_e, n_active);
        else
            pd_f8bs_moe_tc5_kt<S, false, 1u><<<grid, 128, smem, (cudaStream_t)stream>>>(
                wm, ym, wscp, sgp, bep, srp, slp, twp, (float*)y, in_dim, rows_per_e,
                srows_pad, n_expert * rows_per_e, n_active);
    }
    return pd_launch_status();
#endif
}

PD_EXPORT
int pd_f8bs_moe_gemm_gu(const void* wdata, const void* wsc, const void* xg,
                        const void* sg, const void* bexp, void* y, uint32_t in_dim,
                        uint32_t rows_per_e, uint32_t n_expert, uint32_t srows_pad,
                        uint32_t max_blocks, void* stream) {
    return pd_f8bs_moe_launch(false, wdata, wsc, xg, sg, bexp, nullptr, nullptr,
                              nullptr, y, in_dim, rows_per_e, n_expert, srows_pad,
                              max_blocks, 0, stream);
}

PD_EXPORT
int pd_f8bs_moe_gemm_dn(const void* wdata, const void* wsc, const void* xg,
                        const void* sg, const void* bexp, const void* srow,
                        const void* sslot, const void* topk_w, void* part,
                        uint32_t in_dim, uint32_t rows_per_e, uint32_t n_expert,
                        uint32_t srows_pad, uint32_t max_blocks, uint32_t n_active,
                        void* stream) {
    return pd_f8bs_moe_launch(true, wdata, wsc, xg, sg, bexp, srow, sslot, topk_w,
                              part, in_dim, rows_per_e, n_expert, srows_pad,
                              max_blocks, n_active, stream);
}

// ---- decode-band shapes  ------------------------------------------
// The f8_min=64 attribution (c8, real routing): the M=128/N=128 gu
// already beat dec2 at decode (55.3 vs 71.3us median) - the chain lost on
// (a) the dn geometry (81us: 22 out-tiles x only 6 k-stages = prelude-
// dominated CTAs) and (b) interstitials scaled by the worst-case srp
// (gather 15.8 + geglu_pad 25.0us over 16384 mostly-PAD rows). The decode
// band therefore keeps the tc5 pipe and fixes the shapes: BM=32 sorted
// blocks (BN=32 mma tiles, 4x less PAD, srp/4), a Y-RESIDENT dn (the whole
// 32x768 fq tile lives in smem, one CTA walks OTL out-tiles streaming only
// W - prelude amortized OTL-fold, Y read once instead of per tile), and
// PAD-block-aware interstitials. No evict_first anywhere: the dec3 lesson -
// skewed routing makes hot slabs L2-resident, and this grid's consecutive
// same-expert blocks (align lays them adjacent) lean into that.

// gu at BM=32: the BN-templated grouped kernel, h32 Y boxes, 64-col tmem.
PD_EXPORT
int pd_f8bs_moe_gemm_gu_d32(const void* wdata, const void* wsc, const void* xg,
                            const void* sg, const void* bexp, void* y,
                            uint32_t in_dim, uint32_t rows_per_e, uint32_t n_expert,
                            uint32_t srows_pad, uint32_t max_blocks, void* stream) {
#if !defined(PD_BS_HOST) || !defined(PD_TC5_HOST)
    (void)wdata; (void)wsc; (void)xg; (void)sg; (void)bexp; (void)y; (void)in_dim;
    (void)rows_per_e; (void)n_expert; (void)srows_pad; (void)max_blocks; (void)stream;
    return cudaErrorNotSupported;
#else
    static const bool tc5 = [] {
        int dev = 0, cc = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&cc, cudaDevAttrComputeCapabilityMajor, dev);
        return cc == 10 && pd_tmap_encode() != nullptr
            && pd_env("PADDOCK_NO_TC5") == nullptr;
    }();
    if (!tc5) return cudaErrorNotSupported;
    if ((in_dim & 127u) != 0 || (rows_per_e & 127u) != 0 || (srows_pad & 31u) != 0)
        return cudaErrorInvalidValue;
    if (max_blocks == 0) return 0;
    CUtensorMap wm, ym;
    if (!pd_tmap_2d(&wm, wdata, in_dim, n_expert * rows_per_e) ||
        !pd_tmap_2d_h32(&ym, xg, in_dim, srows_pad))
        return cudaErrorNotSupported;
    // ring depth: S=2's occupancy gain (43KB -> 5 CTAs/SM, one wave at the
    // 704-CTA uniform-r=8 grid) was MEASURED worse than S=3 (gu 90.2 vs
    // 74.8us r=8, 173 vs 143 r=32) - the depth-2 ring stalls the consumer
    // harder than the extra wave costs. Same verdict as dec3's S sweep on
    // the dp4a consumer. S=3 default; env re-pins for sweeps.
    static const uint32_t sd = [] {
        const char* v = pd_env("PADDOCK_MOE_F8D_S");
        const int x = v ? atoi(v) : 3;
        return (uint32_t)(x < 2 ? 2 : (x > 3 ? 3 : x));
    }();
    const uint32_t smem = sd * 16384u + sd * 4096u + sd * 2u * 512u + 2u * sd * 8u;
    static bool attr[2] = {false, false};
    if (!attr[sd - 2u]) {
        const void* f = sd == 2u ? (const void*)pd_f8bs_moe_tc5_kt<2u, false, 1u, 32u>
                                 : (const void*)pd_f8bs_moe_tc5_kt<3u, false, 1u, 32u>;
        cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        attr[sd - 2u] = true;
    }
    const uint32_t grid = max_blocks * (rows_per_e >> 7);
    if (sd == 2u)
        pd_f8bs_moe_tc5_kt<2u, false, 1u, 32u><<<grid, 128, smem, (cudaStream_t)stream>>>(
            wm, ym, (const unsigned char*)wsc, (const unsigned char*)sg,
            (const unsigned int*)bexp, nullptr, nullptr, nullptr, (float*)y, in_dim,
            rows_per_e, srows_pad, n_expert * rows_per_e, 0);
    else
        pd_f8bs_moe_tc5_kt<3u, false, 1u, 32u><<<grid, 128, smem, (cudaStream_t)stream>>>(
            wm, ym, (const unsigned char*)wsc, (const unsigned char*)sg,
            (const unsigned int*)bexp, nullptr, nullptr, nullptr, (float*)y, in_dim,
            rows_per_e, srows_pad, n_expert * rows_per_e, 0);
    return pd_launch_status();
#endif
}

// dn at BM=32, Y-resident: the block's full fq tile (32 rows x in_dim,
// in_dim=768 -> 24KB + 6 SFB tiles) is staged once; the CTA then walks OTL
// out-tiles, streaming only the expert's W k-tiles through an S-deep ring.
// Per-out-tile mma order (k-tile 0..nkd-1 x kb 0..3) is identical to the
// M=128 dn kernel, so live outputs match it bitwise. D double-buffers in
// tmem (2 x 32 cols); each drain ends in __syncthreads so a lagging warp
// can never overlap a reused D buffer (v1 keeps the drain inline - pipeline
// it only if kbench shows the stall).
template <uint32_t S>
__global__ void __launch_bounds__(128) pd_f8bs_moe_dn_d32_kt(
    const __grid_constant__ PdTmap wmap, const __grid_constant__ PdTmap ymap,
    const unsigned char* __restrict__ wsc, const unsigned char* __restrict__ xsc,
    const unsigned int* __restrict__ bexp, const unsigned int* __restrict__ srow,
    const unsigned int* __restrict__ sslot, const float* __restrict__ topk_w,
    float* __restrict__ y, uint32_t in_dim, uint32_t rows_per_e, uint32_t srows_pad,
    uint32_t w_rows, uint32_t n_active, uint32_t otl) {
#if PD_TC5_OK
    const uint32_t tiles = rows_per_e >> 7;
    const uint32_t ntg = (tiles + otl - 1u) / otl;
    const uint32_t blk = blockIdx.x / ntg;
    const uint32_t ot0 = (blockIdx.x - blk * ntg) * otl;
    const unsigned int e = bexp[blk];
    if (e == PD_MOE_PAD) return;   // uniform per CTA - before any alloc
    const uint32_t otn = min(otl, tiles - ot0);
    const uint32_t nkd = in_dim >> 7;          // k-tiles (6 at in_dim=768)
    const uint32_t n_kb = in_dim >> 5;         // scale bytes per row
    const uint32_t col_base = blk * 32u;

    extern __shared__ __align__(1024) unsigned char pd_d32_sh[];
    unsigned char* yt = pd_d32_sh;                        // nkd x 4KB, resident
    unsigned char* wt = pd_d32_sh + nkd * 4096u;          // S x 16KB ring
    unsigned char* sfy = wt + S * 16384u;                 // nkd x 512B SFB tiles
    unsigned char* sfw = sfy + nkd * 512u;                // S x 512B SFA ring
    uint64_t* yfull = (uint64_t*)(sfw + S * 512u);
    uint64_t* wfull = yfull + 1;                          // [S]
    uint64_t* wdone = wfull + S;                          // [S]
    __shared__ uint32_t tmem_slot[1];

    const uint32_t tid = threadIdx.x;
    if (tid == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 129;"
                     ::"r"((uint32_t)__cvta_generic_to_shared(yfull)));
        for (uint32_t s = 0; s < S; ++s) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 129;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&wfull[s])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&wdone[s])));
        }
    }
    __syncthreads();
    // tmem: D 2x32 (double buffer) + SFB nkd x 4 + SFA ring S x 4 -> <=112
    constexpr uint32_t TCOLS = 128u;
    if (tid < 32)
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                     ::"r"((uint32_t)__cvta_generic_to_shared(tmem_slot)), "r"(TCOLS));
    __syncthreads();
    const uint32_t tmem = tmem_slot[0];
    const uint32_t sfb0 = tmem + 64u;            // nkd x 4 cols
    const uint32_t sfa0 = sfb0 + nkd * 4u;       // S x 4 cols

    auto bar_wait = [&](uint64_t* bar, uint32_t parity) {
        const uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
        asm volatile("{\n\t.reg .pred p;\nW%=:\n\t"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                     "@!p bra W%=;\n\t}" ::"r"(a), "r"(parity));
    };
    // one ring step = (ot, kt); stage s = it % S
    auto w_stage = [&](uint32_t it, uint32_t s) {
        const uint32_t ot = ot0 + it / nkd, kt = it - (it / nkd) * nkd;
        const uint32_t rw = e * rows_per_e + ot * 128u + tid;
        const uint32_t off = (tid % 32u) * 16u + (tid / 32u) * 4u;
        pd_mma_cpa4p(sfw + s * 512u + off, wsc + (size_t)rw * n_kb + kt * 4u,
                     rw < w_rows);
        const uint32_t m = (uint32_t)__cvta_generic_to_shared(&wfull[s]);
        asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(m)
                     : "memory");
        if (tid == 0) {
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                         ::"r"(m), "r"(16384u));
            const uint32_t wd = (uint32_t)__cvta_generic_to_shared(wt + s * 16384u);
            asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                         " [%0], [%1, {%2, %3}], [%4];" ::"r"(wd), "l"(&wmap),
                         "r"((int)(kt * 128u)), "r"((int)(e * rows_per_e + ot * 128u)),
                         "r"(m) : "memory");
        }
    };

    // Y prologue: all k-tiles of the block's fq rows + their SFB quads, once
    {
        const uint32_t m = (uint32_t)__cvta_generic_to_shared(yfull);
        const uint32_t off = (tid % 32u) * 16u + (tid / 32u) * 4u;
        const uint32_t rc = col_base + tid;   // rows 32..127 guarded garbage
        for (uint32_t kt = 0; kt < nkd; ++kt)
            pd_mma_cpa4p(sfy + kt * 512u + off, xsc + (size_t)rc * n_kb + kt * 4u,
                         rc < srows_pad);
        asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(m)
                     : "memory");
        if (tid == 0) {
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                         ::"r"(m), "r"(nkd * 4096u));
            for (uint32_t kt = 0; kt < nkd; ++kt) {
                const uint32_t yd =
                    (uint32_t)__cvta_generic_to_shared(yt + kt * 4096u);
                asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                             " [%0], [%1, {%2, %3}], [%4];" ::"r"(yd), "l"(&ymap),
                             "r"((int)(kt * 128u)), "r"((int)col_base), "r"(m)
                             : "memory");
            }
        }
    }

    const uint32_t ni = otn * nkd;
    for (uint32_t it = 0; it < S && it < ni; ++it) w_stage(it, it);
    bar_wait(yfull, 0);
    if (tid == 0) {
        // SFB tiles to tmem once - cp and mma form the in-order tcgen05 pipe
        for (uint32_t kt = 0; kt < nkd; ++kt) {
            const uint32_t v =
                (uint32_t)__cvta_generic_to_shared(sfy + kt * 512u) >> 4;
            const uint64_t db = ((uint64_t)(v & 0x3FFFu)) | ((uint64_t)1u << 16)
                              | ((uint64_t)8u << 32);
            asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                         ::"r"(sfb0 + kt * 4u), "l"(db) : "memory");
        }
    }

    uint32_t fph = 0, dph = 0;
    for (uint32_t it = 0; it < ni; ++it) {
        const uint32_t s = it % S;
        const uint32_t kt = it - (it / nkd) * nkd;
        const uint32_t dbuf = tmem + ((it / nkd) & 1u) * 32u;
        if (tid == 0) {
            bar_wait(&wfull[s], (fph >> s) & 1u);
            fph ^= 1u << s;
            const uint32_t v =
                (uint32_t)__cvta_generic_to_shared(sfw + s * 512u) >> 4;
            const uint64_t da = ((uint64_t)(v & 0x3FFFu)) | ((uint64_t)1u << 16)
                              | ((uint64_t)8u << 32);
            asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                         ::"r"(sfa0 + s * 4u), "l"(da) : "memory");
            const uint32_t w16 = (uint32_t)__cvta_generic_to_shared(wt + s * 16384u) >> 4;
            const uint32_t y16 = (uint32_t)__cvta_generic_to_shared(yt + kt * 4096u) >> 4;
            #pragma unroll
            for (uint32_t kb = 0; kb < 4u; ++kb) {
                const uint64_t ad = pd_tc5_sdesc(w16 + kb * 2u);
                const uint64_t bd = pd_tc5_sdesc(y16 + kb * 2u);
                const uint32_t en = (kt > 0 || kb > 0) ? 1u : 0u;
                asm volatile(
                    "{\n\t.reg .pred p;\n\t"
                    "setp.ne.b32 p, %6, 0;\n\t"
                    "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X"
                    " [%0], %1, %2, %3, [%4], [%5], p;\n\t}"
                    ::"r"(dbuf), "l"(ad), "l"(bd), "r"(pd_tc5_bs_idesc_bn(kb, 32u)),
                      "r"(sfa0 + s * 4u), "r"(sfb0 + kt * 4u), "r"(en));
            }
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                ::"r"((uint32_t)__cvta_generic_to_shared(&wdone[s])));
        }
        bar_wait(&wdone[s], (dph >> s) & 1u);
        dph ^= 1u << s;
        if (it + S < ni) w_stage(it + S, s);
        if (kt == nkd - 1u) {
            // drain this out-tile: D rows = out features, cols = tokens
            const uint32_t ot = ot0 + it / nkd;
            uint32_t r[32];
            const uint32_t warp = tid >> 5, lane = tid & 31u;
            const uint32_t taddr = dbuf + ((warp * 32u) << 16);
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
                "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
                "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
                : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
                  "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),"=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15]),
                  "=r"(r[16]),"=r"(r[17]),"=r"(r[18]),"=r"(r[19]),"=r"(r[20]),"=r"(r[21]),"=r"(r[22]),"=r"(r[23]),
                  "=r"(r[24]),"=r"(r[25]),"=r"(r[26]),"=r"(r[27]),"=r"(r[28]),"=r"(r[29]),"=r"(r[30]),"=r"(r[31])
                : "r"(taddr));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            const uint32_t local = ot * 128u + warp * 32u + lane;
            if (local < rows_per_e) {
                #pragma unroll
                for (uint32_t j = 0; j < 32u; ++j) {
                    const unsigned int tok = srow[col_base + j];
                    if (tok == PD_MOE_PAD) continue;
                    const unsigned int sl = sslot[col_base + j];
                    const float w = topk_w[(size_t)tok * n_active + sl];
                    y[((size_t)tok * n_active + sl) * rows_per_e + local] =
                        w * __uint_as_float(r[j]);
                }
            }
            __syncthreads();   // no warp may lag into this D buffer's reuse
        }
    }
    if (tid < 32)
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem),
                     "r"(TCOLS));
#else
    (void)wmap; (void)ymap; (void)wsc; (void)xsc; (void)bexp; (void)srow;
    (void)sslot; (void)topk_w; (void)y; (void)in_dim; (void)rows_per_e;
    (void)srows_pad; (void)w_rows; (void)n_active; (void)otl;
#endif  // PD_TC5_OK
}

PD_EXPORT
int pd_f8bs_moe_gemm_dn_d32(const void* wdata, const void* wsc, const void* xg,
                            const void* sg, const void* bexp, const void* srow,
                            const void* sslot, const void* topk_w, void* part,
                            uint32_t in_dim, uint32_t rows_per_e, uint32_t n_expert,
                            uint32_t srows_pad, uint32_t max_blocks, uint32_t n_active,
                            void* stream) {
#if !defined(PD_BS_HOST) || !defined(PD_TC5_HOST)
    (void)wdata; (void)wsc; (void)xg; (void)sg; (void)bexp; (void)srow; (void)sslot;
    (void)topk_w; (void)part; (void)in_dim; (void)rows_per_e; (void)n_expert;
    (void)srows_pad; (void)max_blocks; (void)n_active; (void)stream;
    return cudaErrorNotSupported;
#else
    static const bool tc5 = [] {
        int dev = 0, cc = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&cc, cudaDevAttrComputeCapabilityMajor, dev);
        return cc == 10 && pd_tmap_encode() != nullptr
            && pd_env("PADDOCK_NO_TC5") == nullptr;
    }();
    if (!tc5) return cudaErrorNotSupported;
    if ((in_dim & 127u) != 0 || (rows_per_e & 127u) != 0 || (srows_pad & 31u) != 0)
        return cudaErrorInvalidValue;
    const uint32_t nkd = in_dim >> 7;
    if (nkd == 0 || nkd > 8u) return cudaErrorInvalidValue;  // smem sizing bound
    if (max_blocks == 0) return 0;
    CUtensorMap wm, ym;
    if (!pd_tmap_2d(&wm, wdata, in_dim, n_expert * rows_per_e) ||
        !pd_tmap_2d_h32(&ym, xg, in_dim, srows_pad))
        return cudaErrorNotSupported;
    constexpr uint32_t S = 3u;
    // lab comparator: the plain BN=32 clone of the M=128 dn (no Y residency,
    // per-tile CTAs) - isolates the Y-resident restructure's contribution
    static const bool naive = pd_env("PADDOCK_MOE_F8D_DN_NAIVE") != nullptr;
    if (naive) {
        const uint32_t smem = S * 16384u + S * 4096u + S * 2u * 512u + 2u * S * 8u;
        static bool nat = false;
        if (!nat) {
            cudaFuncSetAttribute((const void*)pd_f8bs_moe_tc5_kt<S, true, 1u, 32u>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
            nat = true;
        }
        const uint32_t grid = max_blocks * (rows_per_e >> 7);
        pd_f8bs_moe_tc5_kt<S, true, 1u, 32u><<<grid, 128, smem, (cudaStream_t)stream>>>(
            wm, ym, (const unsigned char*)wsc, (const unsigned char*)sg,
            (const unsigned int*)bexp, (const unsigned int*)srow,
            (const unsigned int*)sslot, (const float*)topk_w, (float*)part, in_dim,
            rows_per_e, srows_pad, n_expert * rows_per_e, n_active);
        return pd_launch_status();
    }
    // OTL adapts to the block bound: target ~2 CTAs/SM on the full die
    // (~296 at 148 SMs) - a hot-routed tick with 8 blocks runs OTL=1 (176
    // CTAs) where a fixed OTL=4 left the die 2/3 idle. Shape-derived only,
    // so CUDA-graph captures stay deterministic. Env pins it for sweeps.
    static const int otl_env = [] {
        const char* v = pd_env("PADDOCK_MOE_F8D_OTL");
        return v ? atoi(v) : 0;
    }();
    const uint32_t tiles0 = rows_per_e >> 7;
    uint32_t otl;
    if (otl_env > 0) {
        otl = (uint32_t)otl_env;
    } else {
        otl = (tiles0 * max_blocks + 295u) / 296u;
    }
    otl = otl < 1u ? 1u : (otl > 8u ? 8u : otl);
    // same shallow-and-wide default as the gu launcher: S=2 -> 62KB smem ->
    // 3 CTAs/SM (one wave for the uniform-r=8 384-CTA grid) vs S=3's 78KB/2
    static const uint32_t sd = [] {
        const char* v = pd_env("PADDOCK_MOE_F8D_S");
        const int x = v ? atoi(v) : 3;
        return (uint32_t)(x < 2 ? 2 : (x > 3 ? 3 : x));
    }();
    const uint32_t smem = nkd * 4096u + sd * 16384u + nkd * 512u + sd * 512u
                        + (1u + 2u * sd) * 8u;
    static bool attr[2] = {false, false};
    if (!attr[sd - 2u]) {
        const void* f = sd == 2u ? (const void*)pd_f8bs_moe_dn_d32_kt<2u>
                                 : (const void*)pd_f8bs_moe_dn_d32_kt<3u>;
        cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        attr[sd - 2u] = true;
    }
    const uint32_t tiles = rows_per_e >> 7;
    const uint32_t grid = max_blocks * ((tiles + otl - 1u) / otl);
    if (sd == 2u)
        pd_f8bs_moe_dn_d32_kt<2u><<<grid, 128, smem, (cudaStream_t)stream>>>(
            wm, ym, (const unsigned char*)wsc, (const unsigned char*)sg,
            (const unsigned int*)bexp, (const unsigned int*)srow,
            (const unsigned int*)sslot, (const float*)topk_w, (float*)part, in_dim,
            rows_per_e, srows_pad, n_expert * rows_per_e, n_active, otl);
    else
        pd_f8bs_moe_dn_d32_kt<3u><<<grid, 128, smem, (cudaStream_t)stream>>>(
            wm, ym, (const unsigned char*)wsc, (const unsigned char*)sg,
            (const unsigned int*)bexp, (const unsigned int*)srow,
            (const unsigned int*)sslot, (const float*)topk_w, (float*)part, in_dim,
            rows_per_e, srows_pad, n_expert * rows_per_e, n_active, otl);
    return pd_launch_status();
#endif
}

// block-aware geglu+quant: identical math to pd_quantize_e4m3_geglu2_pad but
// PAD blocks (bexp == PD_MOE_PAD) retire after one load - the worst-case srp
// grid stops costing 25us of garbage traffic at decode (attribution).
__global__ void pd_quantize_e4m3_geglu2_pad_b_kernel(
    const float* __restrict__ gu, unsigned char* __restrict__ q,
    unsigned char* __restrict__ scale, const unsigned int* __restrict__ bexp,
    uint32_t n_ff, uint32_t n_ff_pad, uint32_t bm, uint32_t n) {
    const uint32_t i = (blockIdx.x * 256u + threadIdx.x) * 4u;
    if (i >= n) return;
    const uint32_t row = i / n_ff, col = i - row * n_ff;
    if (bexp[row / bm] == PD_MOE_PAD) return;
    const float* base = gu + (size_t)row * 2u * n_ff + col;
    const float4 g = *(const float4*)base;
    const float4 u = *(const float4*)(base + n_ff);
    float4 v;
    v.x = 0.5f * g.x * (1.0f + tanhf(0.79788456080286535587989211986876f * g.x
                                     * (1.0f + 0.044715f * g.x * g.x))) * u.x;
    v.y = 0.5f * g.y * (1.0f + tanhf(0.79788456080286535587989211986876f * g.y
                                     * (1.0f + 0.044715f * g.y * g.y))) * u.y;
    v.z = 0.5f * g.z * (1.0f + tanhf(0.79788456080286535587989211986876f * g.z
                                     * (1.0f + 0.044715f * g.z * g.z))) * u.z;
    v.w = 0.5f * g.w * (1.0f + tanhf(0.79788456080286535587989211986876f * g.w
                                     * (1.0f + 0.044715f * g.w * g.w))) * u.w;
    pd_e4m3_quant4(v, threadIdx.x & 7u, q, scale, row * n_ff_pad + col);
}

PD_EXPORT
int pd_quantize_e4m3_geglu2_pad_b(const void* gu, void* q, void* scale,
                                  const void* bexp, uint32_t n_ff, uint32_t n_ff_pad,
                                  uint32_t bm, uint32_t rows, void* stream) {
    const uint32_t n = n_ff * rows;
    if (n == 0) return 0;
    if ((n_ff & 31u) != 0 || (n_ff_pad & 31u) != 0) return cudaErrorInvalidValue;
    if (bm == 0 || (bm & (bm - 1u))) return cudaErrorInvalidValue;
    pd_quantize_e4m3_geglu2_pad_b_kernel<<<(n / 4u + 255u) / 256u, 256u, 0,
                                           (cudaStream_t)stream>>>(
        (const float*)gu, (unsigned char*)q, (unsigned char*)scale,
        (const unsigned int*)bexp, n_ff, n_ff_pad, bm, n);
    return pd_launch_status();
}

// ---- decode-band expert pair, intensity rebuild (lever 2) ----------
// The original token-batched dp4a pair measured 28% of HBM at r=8: one
// OUTPUT ROW per 256-thread block (176 live threads at in_dim 2816), one
// pass, no activation reuse. These twins put four output rows in a block -
// warp w owns (row o0 + w/2, gate|up), streams its full weight row in
// 512B/lane-pass chunks, and the block's x tile is read once for 8 row-dots
// (4x the arithmetic intensity, 4x fewer blocks). Reduction moves from a
// block tree to per-warp shfl - a REORDER class vs the originals (same dot
// terms, different summation tree): greedy/coherence gates arbitrate, and
// they ship as separate exports so qwen's pinned launchers keep their exact
// numerics.
template <bool GELU>
__global__ void __launch_bounds__(256) pd_q8_0_moe_gu_dec2_kernel(
    const int8_t* __restrict__ gate_data, const __half* __restrict__ gate_scale,
    const int8_t* __restrict__ up_data, const __half* __restrict__ up_scale,
    const unsigned int* __restrict__ idx, const int8_t* __restrict__ xq,
    const float* __restrict__ xs, float* __restrict__ out, uint32_t in_dim,
    uint32_t ff, uint32_t n_active) {
    const uint32_t o0 = blockIdx.x * 4u, slot = blockIdx.y, b = blockIdx.z;
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t o = o0 + (warp >> 1);          // this warp's output row
    const bool up = (warp & 1u) != 0;             // gate or up matrix
    const uint32_t n_blocks = in_dim >> 5;
    const uint32_t e = idx[(size_t)b * n_active + slot];
    const int8_t* wd = up ? up_data : gate_data;
    const __half* ws = up ? up_scale : gate_scale;
    const int8_t* row = wd + ((size_t)e * ff + (o < ff ? o : ff - 1u)) * in_dim;
    const __half* rsc = ws + ((size_t)e * ff + (o < ff ? o : ff - 1u)) * n_blocks;
    const int8_t* xrow = xq + (size_t)b * in_dim;
    const float* xsc = xs + (size_t)b * n_blocks;
    float acc = 0.0f;
    for (uint32_t base = lane * 16u; base < in_dim; base += 32u * 16u) {
        const int4 wv = __ldcs(reinterpret_cast<const int4*>(row + base));
        const int4 xv = *reinterpret_cast<const int4*>(xrow + base);
        int s = __dp4a(wv.x, xv.x, 0);
        s = __dp4a(wv.y, xv.y, s);
        s = __dp4a(wv.z, xv.z, s);
        s = __dp4a(wv.w, xv.w, s);
        acc += __half2float(__ldcs(rsc + (base >> 5))) * xsc[base >> 5] * (float)s;
    }
    for (uint32_t s2 = 16; s2 > 0; s2 >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, s2);
    __shared__ float gu[8];
    if (lane == 0) gu[warp] = acc;
    __syncthreads();
    if (threadIdx.x < 4u) {
        const uint32_t oo = o0 + threadIdx.x;
        if (oo < ff) {
            const float g = gu[threadIdx.x * 2u], u = gu[threadIdx.x * 2u + 1u];
            const float act = GELU
                ? 0.5f * g * (1.0f + tanhf(0.79788456080286535587989211986876f * g
                                           * (1.0f + 0.044715f * g * g))) * u
                : (g / (1.0f + __expf(-g))) * u;
            out[((size_t)b * n_active + slot) * ff + oo] = act;
        }
    }
}

PD_EXPORT
int pd_q8_0_moe_gu_dec2_geglu(const void* gate_data, const void* gate_scale,
                              const void* up_data, const void* up_scale, const void* idx,
                              const void* xq, const void* xs, void* out, uint32_t in_dim,
                              uint32_t ff, uint32_t n_active, uint32_t batch, void* stream) {
    if (ff == 0 || n_active == 0 || batch == 0) return 0;
    if ((in_dim & 31u) != 0) return cudaErrorInvalidValue;
    dim3 grid((ff + 3u) / 4u, n_active, batch);
    pd_q8_0_moe_gu_dec2_kernel<true><<<grid, 256, 0, (cudaStream_t)stream>>>(
        (const int8_t*)gate_data, (const __half*)gate_scale, (const int8_t*)up_data,
        (const __half*)up_scale, (const unsigned int*)idx, (const int8_t*)xq,
        (const float*)xs, (float*)out, in_dim, ff, n_active);
    return pd_launch_status();
}

// Down twin: 4 embd rows per block; warp w = (row w/2, slot-half w&1) - each
// warp dots 4 of the 8 slots' expert rows sequentially over its row, so the
// fq tile is read once per 4 output rows. Weighted slot sum in smem (fixed
// ascending slot order via the pair loop - deterministic).
__global__ void __launch_bounds__(256) pd_q8_0_moe_dn_dec2_kernel(
    const int8_t* __restrict__ down_data, const __half* __restrict__ down_scale,
    const unsigned int* __restrict__ idx, const float* __restrict__ topk_w,
    const int8_t* __restrict__ fq, const float* __restrict__ fs,
    float* __restrict__ out, uint32_t ff, uint32_t embd, uint32_t n_active) {
    const uint32_t o0 = blockIdx.x * 4u, b = blockIdx.y;
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t o = o0 + (warp >> 1);
    const uint32_t s0 = (warp & 1u) * 4u;         // this warp's 4 slots
    const uint32_t n_blocks = ff >> 5;
    const uint32_t ns = n_active < 4u ? n_active : 4u;
    float acc = 0.0f;                              // sum over the warp's slots
    if (o < embd) {
        for (uint32_t si = 0; si < ns; ++si) {
            const uint32_t slot = s0 + si;
            if (slot >= n_active) break;
            const size_t srow = (size_t)b * n_active + slot;
            const uint32_t e = idx[srow];
            const int8_t* row = down_data + ((size_t)e * embd + o) * ff;
            const __half* rsc = down_scale + ((size_t)e * embd + o) * n_blocks;
            const int8_t* xrow = fq + srow * ff;
            const float* xsc = fs + srow * n_blocks;
            float a = 0.0f;
            for (uint32_t base = lane * 16u; base < ff; base += 32u * 16u) {
                const int4 wv = __ldcs(reinterpret_cast<const int4*>(row + base));
                const int4 xv = *reinterpret_cast<const int4*>(xrow + base);
                int s = __dp4a(wv.x, xv.x, 0);
                s = __dp4a(wv.y, xv.y, s);
                s = __dp4a(wv.z, xv.z, s);
                s = __dp4a(wv.w, xv.w, s);
                a += __half2float(__ldcs(rsc + (base >> 5))) * xsc[base >> 5] * (float)s;
            }
            acc += topk_w[srow] * a;
        }
    }
    for (uint32_t s2 = 16; s2 > 0; s2 >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, s2);
    __shared__ float sh[8];
    if (lane == 0) sh[warp] = acc;
    __syncthreads();
    if (threadIdx.x < 4u) {
        const uint32_t oo = o0 + threadIdx.x;
        if (oo < embd)
            out[(size_t)b * embd + oo] = sh[threadIdx.x * 2u] + sh[threadIdx.x * 2u + 1u];
    }
}

PD_EXPORT
int pd_q8_0_moe_dn_dec2(const void* down_data, const void* down_scale, const void* idx,
                        const void* topk_w, const void* fq, const void* fs, void* out,
                        uint32_t ff, uint32_t embd, uint32_t n_active, uint32_t batch,
                        void* stream) {
    if (embd == 0 || n_active == 0 || batch == 0) return 0;
    if ((ff & 31u) != 0 || n_active > 8u) return cudaErrorInvalidValue;
    dim3 grid((embd + 3u) / 4u, batch);
    pd_q8_0_moe_dn_dec2_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(
        (const int8_t*)down_data, (const __half*)down_scale, (const unsigned int*)idx,
        (const float*)topk_w, (const int8_t*)fq, (const float*)fs, (float*)out, ff, embd,
        n_active);
    return pd_launch_status();
}

// ---- dec3: bulk-streamed decode-band expert pair (target) ----------
// dec2's per-lane __ldcs gathers cap at ~3.6/2.5 TB/s (45/32% of B200 DRAM)
// while the pack's own TMA streams run 6.4-7.2 cold (23) - the fast
// decode-band expert kernels are in that class, and their edge is the
// load machinery, not fp8 operands. dec3 keeps dec2's
// MATH - identical per-lane pass pattern, identical shfl tree, identical
// epilogue formulas - and swaps the weight loads: one CTA per (touched-
// expert block, out-row tile) streams the expert's slab rows once through a
// cp.async.bulk 1D ring (evict_first: streamed weights are never re-read)
// and applies them to the block's routed rows from smem. Expert -> rows via
// the moe_align CSR at BM=8 (bexp/srow/sslot), which also dedups weight
// reads across pairs sharing an expert (dec2 re-reads per pair - at r=32
// the unique-byte rate is half the printed one).
// Structure (final): split
// issuer - producer warp waits per-stage consumed barriers and issues
// immediately, compute warps never block-sync - plus adaptive tiling (the
// launcher sizes grid.y to fill ~2 CTA/SM from the caller's pair count),
// with the block's activation rows in a BM-row smem tile (np <= BM always).
// Tuning falsifications, all measured at the A4B decode shape (keep these;
// do not re-try without new evidence):
//   - reading x/fq from global in the dp4a loop (v3): consumer latency-
//     serializes (~5 passes of ILP cannot hide L2), ring starves - 40%
//     LOSS. Streamed grids (~300 CTAs) lack dec2's massive TLP; the
//     activation tile must be smem.
//   - deeper rings (gu S=4/S=6) and finer stages (CR=2): no gain to -40%.
//     The consumer-coupled ring saturates ~12-15 GB/s per CTA regardless
//     of outstanding bytes - unlike stream_roof's consumer-free 25 GB/s -
//     so gu S=3/CR=4 and dn S=4/CR=16 are the measured optima.
//   - an in-loop wave/restage wrapper for a smaller x-tile (NP=2): ~40%
//     loss at identical PARAMETERS from the loop-body code structure
//     alone, and pointless - np never exceeds BM.
// Where it lands: on UNIFORM routing gu beats dec2 from r=4
// (-22%) through r=32 (-53%, dedup); dn dec3 loses at every r. But real
// serving routing is SKEWED, and skew flips the economics twice over: hot
// experts made straggler CTAs (measured in serving: gu dec3 median 96us stdev 20
// vs dec2's stdev 1.1; BM=2 rebalances but re-streams), and - the deeper
// finding - dec2 on skewed routing is faster than on uniform (kbench
// `hot` r=8: 40.0us, 6.7 TB/s effective) because the hot slabs are
// L2-RESIDENT and its per-pair re-reads become L2 hits. The dedup dec3
// streams DRAM for, the cache already gives dec2 for free - and
// evict_first deliberately bypasses it. FALSIFIED as the serving default
// (c8 -11%); kept opt-in (PADDOCK_MOE_DEC3=1) as the uniform/large-uniq
// regime study. A future dec4 must be L2-AWARE (gather hot experts,
// stream cold ones) or move the band to the fp8 tensor-core class.
// Numerics: the gate_up leg is BITWISE dec2 (same dot order per row, same
// GEGLU). The down leg's cross-slot sum moves from dec2's in-warp lane
// mixing to per-pair partials + a fixed-order combine (pd_moe_combine_dec3,
// dec2's slot-half tree) - a REORDER class kept for future work;
// PADDOCK_NO_MOE_DEC3 kills the serving arm for A/B.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#define PD_DEC3_OK 1
#else
#define PD_DEC3_OK 0
#endif

#define PD_DEC3_BM 2u        // pairs per aligned block (moe_align_bm bm=2).
                             // Small on PURPOSE: real routing is skewed (the
                             // hot experts carry 4-8 pairs and
                             // a BM=8 block serialized them in one CTA chain,
                             // stdev 20us vs dec2's 1.1); at BM=2 the block
                             // count TRACKS work and the consumer stays
                             // fetch-bound at any skew. Multi-block experts
                             // re-stream their slab once per block - still
                             // >= 2x fewer weight reads than dec2's per-pair.
#define PD_DEC3_GU_CR 4u     // gate_up chunk rows (warp w owns chunk-row w)
#define PD_DEC3_GU_S 3u      // gate_up ring depth (see the tuning note in
                             // the header: 3 beat 4 and 6)
#define PD_DEC3_DN_CR 16u    // down chunk rows (warp w owns rows 4w..4w+3)
#define PD_DEC3_DN_S 4u      // down ring depth

#if PD_DEC3_OK
__device__ __forceinline__ void pd_dec3_wait(uint64_t* bar, uint32_t parity) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("{\n\t.reg .pred p;\nW%=:\n\t"
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                 "@!p bra W%=;\n\t}" ::"r"(a), "r"(parity));
}
__device__ __forceinline__ void pd_dec3_bulk(unsigned char* dst, const void* src,
                                             uint32_t bytes, uint32_t mbar,
                                             uint64_t pol) {
    asm volatile(
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
        " [%0], [%1], %2, [%3], %4;"
        ::"r"((uint32_t)__cvta_generic_to_shared(dst)), "l"(src), "r"(bytes),
          "r"(mbar), "l"(pol) : "memory");
}
#endif

// Fused gate+up+GEGLU, expert-streamed. Grid (max_blocks, tiles), 160
// threads: warps 0-3 compute (warp w owns chunk-row w), warp 4 lane 0 is
// the producer. Chunk = CR consecutive out rows of both matrices + their
// f16 block scales, staged as 4 contiguous bulk copies (rows are contiguous
// in the repacked planes). Per-(row, pair) math is dec2's exactly.
template <bool GELU>
__global__ void __launch_bounds__(160) pd_q8_0_moe_gu_dec3_kernel(
    const int8_t* __restrict__ gate_data, const __half* __restrict__ gate_scale,
    const int8_t* __restrict__ up_data, const __half* __restrict__ up_scale,
    const unsigned int* __restrict__ bexp, const unsigned int* __restrict__ srow,
    const unsigned int* __restrict__ sslot, const int8_t* __restrict__ xq,
    const float* __restrict__ xs, float* __restrict__ out, uint32_t in_dim,
    uint32_t ff, uint32_t n_active, uint32_t tile_r) {
#if PD_DEC3_OK
    constexpr uint32_t CR = 4u, S = 3u, BM = PD_DEC3_BM;
    const unsigned int e = bexp[blockIdx.x];
    if (e == PD_MOE_PAD) return;
    const uint32_t tile0 = blockIdx.y * tile_r;
    if (tile0 >= ff) return;
    const uint32_t tile_rows = (tile0 + tile_r <= ff) ? tile_r : ff - tile0;
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const uint32_t n_blocks = in_dim >> 5;
    const uint32_t scb = in_dim >> 4;
    const uint32_t stage_b = 2u * CR * (in_dim + scb);

    extern __shared__ __align__(128) unsigned char pd_d3_sh[];
    unsigned char* ring = pd_d3_sh;
    int8_t* xtile = (int8_t*)(pd_d3_sh + S * stage_b);
    float* xsc_sm = (float*)(xtile + BM * in_dim);
    __shared__ uint64_t bfull[S], bdone[S];
    __shared__ unsigned int prow[BM], pslot[BM];

    if (tid < BM) {
        prow[tid] = srow[(size_t)blockIdx.x * BM + tid];
        pslot[tid] = sslot[(size_t)blockIdx.x * BM + tid];
    }
    if (tid == 0) {
        #pragma unroll
        for (uint32_t s = 0; s < S; ++s) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bfull[s])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 4;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bdone[s])));
        }
    }
    __syncthreads();
    uint32_t np = 0;
    while (np < BM && prow[np] != PD_MOE_PAD) ++np;

    {
        const uint32_t xw = in_dim >> 4;
        for (uint32_t i = tid; i < np * xw; i += 160u) {
            const uint32_t p = i / xw, w = i - p * xw;
            ((int4*)xtile)[p * xw + w] =
                ((const int4*)(xq + (size_t)prow[p] * in_dim))[w];
        }
        for (uint32_t i = tid; i < np * n_blocks; i += 160u) {
            const uint32_t p = i / n_blocks, b = i - p * n_blocks;
            xsc_sm[p * n_blocks + b] = xs[(size_t)prow[p] * n_blocks + b];
        }
    }
    __syncthreads();

    const uint32_t nchunks = (tile_rows + CR - 1u) / CR;
    if (warp == 4u) {
        if (lane == 0) {
            auto issue = [&](uint32_t c) {
                const uint32_t s = c % S;
                const uint32_t r0 = c * CR;
                const uint32_t rows_c = (r0 + CR <= tile_rows) ? CR : tile_rows - r0;
                const uint32_t db = rows_c * in_dim, sb = rows_c * scb;
                unsigned char* st = ring + s * stage_b;
                const uint32_t m = (uint32_t)__cvta_generic_to_shared(&bfull[s]);
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                             ::"r"(m), "r"(2u * (db + sb)));
                uint64_t pol;
                asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
                const size_t wrow = (size_t)e * ff + tile0 + r0;
                pd_dec3_bulk(st, gate_data + wrow * in_dim, db, m, pol);
                pd_dec3_bulk(st + CR * in_dim, up_data + wrow * in_dim, db, m, pol);
                pd_dec3_bulk(st + 2u * CR * in_dim,
                             (const unsigned char*)gate_scale + wrow * scb, sb, m, pol);
                pd_dec3_bulk(st + 2u * CR * in_dim + CR * scb,
                             (const unsigned char*)up_scale + wrow * scb, sb, m, pol);
            };
            for (uint32_t c = 0; c < (nchunks < S ? nchunks : S); ++c) issue(c);
            uint32_t dph = 0;
            for (uint32_t c = S; c < nchunks; ++c) {
                const uint32_t s = c % S;
                pd_dec3_wait(&bdone[s], (dph >> s) & 1u);
                dph ^= 1u << s;
                issue(c);
            }
        }
        return;
    }

    uint32_t fph = 0;
    for (uint32_t c = 0; c < nchunks; ++c) {
        const uint32_t s = c % S;
        pd_dec3_wait(&bfull[s], (fph >> s) & 1u);
        fph ^= 1u << s;
        const uint32_t r = c * CR + warp;
        if (r < tile_rows) {
            const uint32_t o = tile0 + r;
            const unsigned char* st = ring + s * stage_b;
            const int8_t* grow = (const int8_t*)st + warp * in_dim;
            const int8_t* urow = (const int8_t*)(st + CR * in_dim) + warp * in_dim;
            const __half* gsc = (const __half*)(st + 2u * CR * in_dim) + warp * n_blocks;
            const __half* usc =
                (const __half*)(st + 2u * CR * in_dim + CR * scb) + warp * n_blocks;
            float accg[BM] = {}, accu[BM] = {};
            for (uint32_t base = lane * 16u; base < in_dim; base += 32u * 16u) {
                const int4 gv = *(const int4*)(grow + base);
                const int4 uv = *(const int4*)(urow + base);
                const float gs = __half2float(gsc[base >> 5]);
                const float us = __half2float(usc[base >> 5]);
                for (uint32_t p = 0; p < np; ++p) {
                    const int4 xv = *(const int4*)(xtile + (size_t)p * in_dim + base);
                    const float xsp = xsc_sm[p * n_blocks + (base >> 5)];
                    int sg = __dp4a(gv.x, xv.x, 0);
                    sg = __dp4a(gv.y, xv.y, sg);
                    sg = __dp4a(gv.z, xv.z, sg);
                    sg = __dp4a(gv.w, xv.w, sg);
                    accg[p] += gs * xsp * (float)sg;
                    int su = __dp4a(uv.x, xv.x, 0);
                    su = __dp4a(uv.y, xv.y, su);
                    su = __dp4a(uv.z, xv.z, su);
                    su = __dp4a(uv.w, xv.w, su);
                    accu[p] += us * xsp * (float)su;
                }
            }
            for (uint32_t p = 0; p < np; ++p)
                for (uint32_t s2 = 16; s2 > 0; s2 >>= 1) {
                    accg[p] += __shfl_down_sync(0xffffffffu, accg[p], s2);
                    accu[p] += __shfl_down_sync(0xffffffffu, accu[p], s2);
                }
            if (lane == 0 && o < ff)
                for (uint32_t p = 0; p < np; ++p) {
                    const float g = accg[p], u = accu[p];
                    const float act = GELU
                        ? 0.5f * g * (1.0f + tanhf(0.79788456080286535587989211986876f * g
                                                   * (1.0f + 0.044715f * g * g))) * u
                        : (g / (1.0f + __expf(-g))) * u;
                    out[((size_t)prow[p] * n_active + pslot[p]) * ff + o] = act;
                }
        }
        __syncwarp();
        if (lane == 0)
            asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bdone[s])));
    }
#else
    (void)gate_data; (void)gate_scale; (void)up_data; (void)up_scale; (void)bexp;
    (void)srow; (void)sslot; (void)xq; (void)xs; (void)out; (void)in_dim; (void)ff;
    (void)n_active; (void)tile_r;
#endif
}

// Down half, expert-streamed: grid (max_blocks, tiles), 160 threads. Warps
// 0-3 own chunk-rows 4w..4w+3, warp 4 produces; the fq tile holds the
// current NP-pair wave. Per (row, pair) the dot pattern is dec2's, the
// topk_w product is dec2's - but the per-pair value lands in the partials
// buffer for pd_moe_combine_dec3 (one writer per element, deterministic).
__global__ void __launch_bounds__(160) pd_q8_0_moe_dn_dec3_kernel(
    const int8_t* __restrict__ down_data, const __half* __restrict__ down_scale,
    const unsigned int* __restrict__ bexp, const unsigned int* __restrict__ srow,
    const unsigned int* __restrict__ sslot, const float* __restrict__ topk_w,
    const int8_t* __restrict__ fq, const float* __restrict__ fs,
    float* __restrict__ part, uint32_t ff, uint32_t embd, uint32_t n_active,
    uint32_t tile_r) {
#if PD_DEC3_OK
    constexpr uint32_t CR = PD_DEC3_DN_CR, S = PD_DEC3_DN_S, BM = PD_DEC3_BM;
    const unsigned int e = bexp[blockIdx.x];
    if (e == PD_MOE_PAD) return;
    const uint32_t tile0 = blockIdx.y * tile_r;
    if (tile0 >= embd) return;
    const uint32_t tile_rows = (tile0 + tile_r <= embd) ? tile_r : embd - tile0;
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const uint32_t n_blocks = ff >> 5;
    const uint32_t scb = ff >> 4;
    const uint32_t stage_b = CR * (ff + scb);

    extern __shared__ __align__(128) unsigned char pd_d3d_sh[];
    unsigned char* ring = pd_d3d_sh;                      // S stages
    int8_t* fqt = (int8_t*)(pd_d3d_sh + S * stage_b);     // BM x ff
    float* fsc_sm = (float*)(fqt + BM * ff);              // BM x n_blocks
    __shared__ uint64_t bfull[S], bdone[S];
    __shared__ unsigned int prow[BM], pslot[BM];
    __shared__ float tw_sm[BM];

    if (tid < BM) {
        prow[tid] = srow[(size_t)blockIdx.x * BM + tid];
        pslot[tid] = sslot[(size_t)blockIdx.x * BM + tid];
        if (prow[tid] != PD_MOE_PAD)
            tw_sm[tid] = topk_w[(size_t)prow[tid] * n_active + pslot[tid]];
    }
    if (tid == 0) {
        #pragma unroll
        for (uint32_t s = 0; s < S; ++s) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bfull[s])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 4;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bdone[s])));
        }
    }
    __syncthreads();
    uint32_t np = 0;
    while (np < BM && prow[np] != PD_MOE_PAD) ++np;

    {
        const uint32_t xw = ff >> 4;
        for (uint32_t i = tid; i < np * xw; i += 160u) {
            const uint32_t p = i / xw, w = i - p * xw;
            const size_t pr = (size_t)prow[p] * n_active + pslot[p];
            ((int4*)fqt)[p * xw + w] = ((const int4*)(fq + pr * ff))[w];
        }
        for (uint32_t i = tid; i < np * n_blocks; i += 160u) {
            const uint32_t p = i / n_blocks, b = i - p * n_blocks;
            fsc_sm[p * n_blocks + b] =
                fs[((size_t)prow[p] * n_active + pslot[p]) * n_blocks + b];
        }
    }
    __syncthreads();

    const uint32_t nchunks = (tile_rows + CR - 1u) / CR;
    if (warp == 4u) {
        if (lane == 0) {
            auto issue = [&](uint32_t c) {
                const uint32_t s = c % S;
                const uint32_t r0 = c * CR;
                const uint32_t rows_c = (r0 + CR <= tile_rows) ? CR : tile_rows - r0;
                const uint32_t db = rows_c * ff, sb = rows_c * scb;
                unsigned char* st = ring + s * stage_b;
                const uint32_t m = (uint32_t)__cvta_generic_to_shared(&bfull[s]);
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                             ::"r"(m), "r"(db + sb));
                uint64_t pol;
                asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
                const size_t wrow = (size_t)e * embd + tile0 + r0;
                pd_dec3_bulk(st, down_data + wrow * ff, db, m, pol);
                pd_dec3_bulk(st + CR * ff,
                             (const unsigned char*)down_scale + wrow * scb, sb, m, pol);
            };
            for (uint32_t c = 0; c < (nchunks < S ? nchunks : S); ++c) issue(c);
            uint32_t dph = 0;
            for (uint32_t c = S; c < nchunks; ++c) {
                const uint32_t s = c % S;
                pd_dec3_wait(&bdone[s], (dph >> s) & 1u);
                dph ^= 1u << s;
                issue(c);
            }
        }
        return;
    }

    uint32_t fph = 0;
    for (uint32_t c = 0; c < nchunks; ++c) {
        const uint32_t s = c % S;
        pd_dec3_wait(&bfull[s], (fph >> s) & 1u);
        fph ^= 1u << s;
        const unsigned char* st = ring + s * stage_b;
        for (uint32_t rr = warp * 4u; rr < warp * 4u + 4u; ++rr) {
            const uint32_t r = c * CR + rr;
            if (r >= tile_rows) break;
            const uint32_t o = tile0 + r;
            const int8_t* row = (const int8_t*)st + rr * ff;
            const __half* rsc = (const __half*)(st + CR * ff) + rr * n_blocks;
            for (uint32_t p = 0; p < np; ++p) {
                float a = 0.0f;
                for (uint32_t base = lane * 16u; base < ff; base += 32u * 16u) {
                    const int4 wv4 = *(const int4*)(row + base);
                    const int4 xv = *(const int4*)(fqt + (size_t)p * ff + base);
                    int s4 = __dp4a(wv4.x, xv.x, 0);
                    s4 = __dp4a(wv4.y, xv.y, s4);
                    s4 = __dp4a(wv4.z, xv.z, s4);
                    s4 = __dp4a(wv4.w, xv.w, s4);
                    a += __half2float(rsc[base >> 5])
                         * fsc_sm[p * n_blocks + (base >> 5)] * (float)s4;
                }
                for (uint32_t s2 = 16; s2 > 0; s2 >>= 1)
                    a += __shfl_down_sync(0xffffffffu, a, s2);
                if (lane == 0)
                    part[((size_t)prow[p] * n_active + pslot[p]) * embd + o] =
                        tw_sm[p] * a;
            }
        }
        __syncwarp();
        if (lane == 0)
            asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bdone[s])));
    }
#else
    (void)down_data; (void)down_scale; (void)bexp; (void)srow; (void)sslot;
    (void)topk_w; (void)fq; (void)fs; (void)part; (void)ff; (void)embd;
    (void)n_active; (void)tile_r;
#endif
}

// Fixed-order slot combine for the dec3 down partials: dec2's slot-half tree
// (slots 0..3 summed ascending, 4..7 summed ascending, halves added last).
// Plain write - no memset, no accumulate. One thread per (token, element).
__global__ void pd_moe_combine_dec3_kernel(const float* __restrict__ part,
                                           float* __restrict__ out, uint32_t n,
                                           uint32_t n_active) {
    const uint32_t o = blockIdx.x * 256u + threadIdx.x, b = blockIdx.y;
    if (o >= n) return;
    const float* pb = part + (size_t)b * n_active * n + o;
    const uint32_t h = n_active < 4u ? n_active : 4u;
    float v0 = 0.0f, v1 = 0.0f;
    for (uint32_t s = 0; s < h; ++s) v0 += pb[(size_t)s * n];
    for (uint32_t s = 4u; s < n_active; ++s) v1 += pb[(size_t)s * n];
    out[(size_t)b * n + o] = v0 + v1;
}

#ifdef PD_BS_HOST
static bool pd_dec3_dev_ok() {
    static const bool ok = [] {
        int dev = 0, cc = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&cc, cudaDevAttrComputeCapabilityMajor, dev);
        return cc >= 9;   // cp.async.bulk + mbarrier tx tracking
    }();
    return ok;
}
// Adaptive tile rows: split each expert's out_rows so ~`target` CTAs go
// live from `pairs` routed pairs (each pair is at most one live block at
// decode counts), keeping the die fed at r=1..8. Tile rows round up to the
// chunk size; the kernel gets the result verbatim.
static uint32_t pd_dec3_tile(uint32_t out_rows, uint32_t pairs, uint32_t target,
                             uint32_t cr) {
    uint32_t live = pairs ? pairs : 1u;
    uint32_t tiles = (target + live - 1u) / live;
    const uint32_t max_tiles = (out_rows + cr - 1u) / cr;
    if (tiles > max_tiles) tiles = max_tiles;
    if (tiles == 0) tiles = 1u;
    uint32_t t = (out_rows + tiles - 1u) / tiles;
    return (t + cr - 1u) / cr * cr;
}
#endif

PD_EXPORT
int pd_q8_0_moe_gu_dec3_geglu(const void* gate_data, const void* gate_scale,
                              const void* up_data, const void* up_scale,
                              const void* bexp, const void* srow, const void* sslot,
                              const void* xq, const void* xs, void* out,
                              uint32_t in_dim, uint32_t ff, uint32_t n_active,
                              uint32_t max_blocks, uint32_t pairs, void* stream) {
#ifndef PD_BS_HOST
    (void)gate_data; (void)gate_scale; (void)up_data; (void)up_scale; (void)bexp;
    (void)srow; (void)sslot; (void)xq; (void)xs; (void)out; (void)in_dim; (void)ff;
    (void)n_active; (void)max_blocks; (void)pairs; (void)stream;
    return cudaErrorNotSupported;
#else
    if (!pd_dec3_dev_ok()) return cudaErrorNotSupported;
    if (ff == 0 || max_blocks == 0) return 0;
    // in_dim%256 keeps the scale-chunk bulk copies 16B-sized AND -aligned at
    // any chunk row (scb = in_dim/16 must be a 16-multiple)
    if ((in_dim & 255u) != 0 || n_active > 8u) return cudaErrorInvalidValue;
    const uint32_t stage_b = 2u * PD_DEC3_GU_CR * (in_dim + (in_dim >> 4));
    const uint32_t smem = PD_DEC3_GU_S * stage_b + PD_DEC3_BM * in_dim
                        + PD_DEC3_BM * (in_dim >> 5) * 4u;
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute((const void*)pd_q8_0_moe_gu_dec3_kernel<true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        attr = true;
    }
    // ~2 CTA/SM live on the ~148-SM class die
    const uint32_t tile_r = pd_dec3_tile(ff, pairs, 288u, PD_DEC3_GU_CR);
    dim3 grid(max_blocks, (ff + tile_r - 1u) / tile_r);
    pd_q8_0_moe_gu_dec3_kernel<true><<<grid, 160, smem, (cudaStream_t)stream>>>(
        (const int8_t*)gate_data, (const __half*)gate_scale, (const int8_t*)up_data,
        (const __half*)up_scale, (const unsigned int*)bexp, (const unsigned int*)srow,
        (const unsigned int*)sslot, (const int8_t*)xq, (const float*)xs, (float*)out,
        in_dim, ff, n_active, tile_r);
    return pd_launch_status();
#endif
}

PD_EXPORT
int pd_q8_0_moe_dn_dec3(const void* down_data, const void* down_scale,
                        const void* bexp, const void* srow, const void* sslot,
                        const void* topk_w, const void* fq, const void* fs,
                        void* part, uint32_t ff, uint32_t embd, uint32_t n_active,
                        uint32_t max_blocks, uint32_t pairs, void* stream) {
#ifndef PD_BS_HOST
    (void)down_data; (void)down_scale; (void)bexp; (void)srow; (void)sslot;
    (void)topk_w; (void)fq; (void)fs; (void)part; (void)ff; (void)embd;
    (void)n_active; (void)max_blocks; (void)pairs; (void)stream;
    return cudaErrorNotSupported;
#else
    if (!pd_dec3_dev_ok()) return cudaErrorNotSupported;
    if (embd == 0 || max_blocks == 0) return 0;
    // ff%32 = dec2's block-granularity check; embd%16 keeps every chunk's
    // scale bulk copy 16B-aligned (chunk row starts stay 16-multiples)
    if ((ff & 31u) != 0 || (embd & 15u) != 0 || n_active > 8u)
        return cudaErrorInvalidValue;
    const uint32_t stage_b = PD_DEC3_DN_CR * (ff + (ff >> 4));
    const uint32_t smem = PD_DEC3_DN_S * stage_b + PD_DEC3_BM * ff
                        + PD_DEC3_BM * (ff >> 5) * 4u;
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute((const void*)pd_q8_0_moe_dn_dec3_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        attr = true;
    }
    // ~3 CTA/SM live (small stages want more streams)
    const uint32_t tile_r = pd_dec3_tile(embd, pairs, 448u, PD_DEC3_DN_CR);
    dim3 grid(max_blocks, (embd + tile_r - 1u) / tile_r);
    pd_q8_0_moe_dn_dec3_kernel<<<grid, 160, smem, (cudaStream_t)stream>>>(
        (const int8_t*)down_data, (const __half*)down_scale, (const unsigned int*)bexp,
        (const unsigned int*)srow, (const unsigned int*)sslot, (const float*)topk_w,
        (const int8_t*)fq, (const float*)fs, (float*)part, ff, embd, n_active, tile_r);
    return pd_launch_status();
#endif
}

PD_EXPORT
int pd_moe_combine_dec3(const void* part, void* out, uint32_t n, uint32_t n_active,
                        uint32_t batch, void* stream) {
    if (n == 0 || batch == 0) return 0;
    if (n_active > 8u) return cudaErrorInvalidValue;
    dim3 grid((n + 255u) / 256u, batch);
    pd_moe_combine_dec3_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(
        (const float*)part, (float*)out, n, n_active);
    return pd_launch_status();
}

// ---- MoE tail fusions (serial-chain DEPTH cuts) -------------------
// The A4B decode wall is the on-device serial chain (~13 moe nodes/layer,
// where 6-7 is achievable). These three fusions cut the head 3->1, the router
// 2->1 and the combine trailer 4->1. Norm math follows the
// pd_rmsnorm_quant_q8_batch conventions (width-by-batch, warp-per-32 q8
// blocks); different block geometry than the standalone rmsnorm_batch =
// REORDER class - greedy-basin + coherence gates arbitrate.

// head: one sumsq of x serves both weighted norms (router gamma + pre_ffw_2).
// Outputs: rn (f32 router-normed), pn (f32 pre2-normed - the f8 lane's
// quantize input), q/qs (q8 of the pre2-normed rows for the int8 expert
// lanes).
__global__ void pd_moe_head_kernel(const float* __restrict__ x,
                                   const float* __restrict__ gamma,
                                   const float* __restrict__ pre2,
                                   float* __restrict__ rn, float* __restrict__ pn,
                                   signed char* __restrict__ q, float* __restrict__ qs,
                                   uint32_t n, float eps) {
    const uint32_t b = blockIdx.x;
    const float* xb = x + (size_t)b * n;
    float* rb = rn + (size_t)b * n;
    float* pb = pn + (size_t)b * n;
    signed char* qb = q + (size_t)b * n;
    float* sb = qs + (size_t)b * (n >> 5);
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    __shared__ float wsum[32];
    __shared__ float s_inv;
    float acc = 0.0f;
    for (uint32_t i = tid; i < n; i += nth) {
        const float v = xb[i];
        acc += v * v;
    }
    for (uint32_t sh = 16; sh > 0; sh >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, sh);
    const uint32_t warp = tid >> 5, lane = tid & 31u;
    if (lane == 0) wsum[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        float sum = 0.0f;
        const uint32_t nwarps = (nth + 31u) >> 5;
        for (uint32_t wi = 0; wi < nwarps; ++wi) sum += wsum[wi];
        s_inv = 1.0f / sqrtf(sum / (float)n + eps);
    }
    __syncthreads();
    const float inv = s_inv;
    for (uint32_t i = tid; i < n; i += nth) {
        const float xh = xb[i] * inv;
        rb[i] = xh * gamma[i];
        const float v = xh * pre2[i];
        pb[i] = v;
        float a = fabsf(v);
        for (uint32_t sh = 16; sh > 0; sh >>= 1)
            a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, sh));
        const float scl = a * (1.0f / 127.0f);
        if (lane == 0) sb[i >> 5] = scl;
        const float qinv = scl > 0.0f ? 1.0f / scl : 0.0f;
        int qi = __float2int_rn(v * qinv);
        qi = qi < -127 ? -127 : (qi > 127 ? 127 : qi);
        qb[i] = (signed char)qi;
    }
}

PD_EXPORT
// P1-2 producer twin (hibatch path 1): identical dual-weight head, but the
// pn->q8 quantize uses PER-128 scale groups (measured 1.27-1.35x the per-32
// error on real serving rows; per-row FAILED). Two passes over xb - the
// second re-read is L2-hot (first-touch law). qs is written at n/128 stride.
// Fixed 512 threads: uniform groups-per-iter; s_inv tree differs from the
// per-32 head (precision-class lane, gates arbitrate).
__global__ void __launch_bounds__(512) pd_moe_head_xg_kernel(
    const float* __restrict__ x, const float* __restrict__ gamma,
    const float* __restrict__ pre2, float* __restrict__ rn, float* __restrict__ pn,
    signed char* __restrict__ q, float* __restrict__ qs, uint32_t n, float eps) {
    const uint32_t b = blockIdx.x;
    const float* xb = x + (size_t)b * n;
    float* rb = rn + (size_t)b * n;
    float* pb = pn + (size_t)b * n;
    signed char* qb = q + (size_t)b * n;
    float* sb = qs + (size_t)b * (n >> 7);
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    __shared__ float wsum[32];
    __shared__ float s_inv;
    __shared__ unsigned int gmax[32];          // n/128 <= 32 (n <= 4096)
    for (uint32_t i = tid; i < (n >> 7); i += nth) gmax[i] = 0u;
    float acc = 0.0f;
    for (uint32_t i = tid; i < n; i += nth) {
        const float v = xb[i];
        acc += v * v;
    }
    for (uint32_t sh = 16; sh > 0; sh >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, sh);
    const uint32_t warp = tid >> 5, lane = tid & 31u;
    if (lane == 0) wsum[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        float sum = 0.0f;
        const uint32_t nwarps = (nth + 31u) >> 5;
        for (uint32_t wi = 0; wi < nwarps; ++wi) sum += wsum[wi];
        s_inv = 1.0f / sqrtf(sum / (float)n + eps);
    }
    __syncthreads();
    const float inv = s_inv;
    // pass A: rn/pn writes + per-128 amax (atomicMax on non-negative floats)
    for (uint32_t i = tid; i < n; i += nth) {
        const float xh = xb[i] * inv;
        rb[i] = xh * gamma[i];
        const float v = xh * pre2[i];
        pb[i] = v;
        atomicMax(&gmax[i >> 7], __float_as_uint(fabsf(v)));
    }
    __syncthreads();
    // pass B: quantize with the group scale (xb re-read: L2-hot)
    for (uint32_t i = tid; i < n; i += nth) {
        const float a = __uint_as_float(gmax[i >> 7]);
        const float scl = a * (1.0f / 127.0f);
        if ((i & 127u) == 0u) sb[i >> 7] = scl;
        const float qinv = scl > 0.0f ? 1.0f / scl : 0.0f;
        const float v = (xb[i] * inv) * pre2[i];
        int qi = __float2int_rn(v * qinv);
        qi = qi < -127 ? -127 : (qi > 127 ? 127 : qi);
        qb[i] = (signed char)qi;
    }
}

PD_EXPORT
int pd_moe_head_xg(const void* x, const void* gamma, const void* pre2, void* rn,
                   void* pn, void* q, void* qs, uint32_t n, float eps,
                   uint32_t batch, void* stream) {
    if (n == 0 || batch == 0) return 0;
    if ((n & 127u) || n > 4096u) return cudaErrorInvalidValue;
    pd_moe_head_xg_kernel<<<batch, 512, 0, (cudaStream_t)stream>>>(
        (const float*)x, (const float*)gamma, (const float*)pre2, (float*)rn,
        (float*)pn, (signed char*)q, (float*)qs, n, eps);
    return pd_launch_status();
}

int pd_moe_head(const void* x, const void* gamma, const void* pre2, void* rn, void* pn,
                void* q, void* qs, uint32_t n, float eps, uint32_t batch, void* stream) {
    if (n == 0 || batch == 0) return 0;
    if (n & 31u) return cudaErrorInvalidValue;
    pd_moe_head_kernel<<<batch, (batch >= 64u ? pd_norm_wide_nth(batch) : 1024u), 0, (cudaStream_t)stream>>>(
        (const float*)x, (const float*)gamma, (const float*)pre2, (float*)rn, (float*)pn,
        (signed char*)q, (float*)qs, n, eps);
    return pd_launch_status();
}

// hibatch-lane twin of head_router (M1): grid =
// ceil(batch/8) blocks x 256 thr; warp t owns token b0+t. The router-normed
// rows live in smem as BF16 (precision-class, lane gates arbitrate) and the
// router weight plane is read once per BLOCK for 8 tokens - the exact fix
// for headr's fatal per-token plane re-read (180MB/tick at r=128). Phase A
// norm/quantize is warp-local (per-token f32 walk, same 32-group amax tree);
// phase B accumulates all 8 token dots per weight read, FMA-contracted,
// ascending-i per (o,t) with the tile matvec's lane-stride+shfl tree; phase
// C is pd_moe_topk_warp + dscale fold per token, verbatim.
__global__ void pd_moe_head_router_hb_kernel(
    const float* __restrict__ x, const float* __restrict__ gamma,
    const float* __restrict__ pre2, const float* __restrict__ rw,
    const float* __restrict__ dscale, float* __restrict__ pn,
    signed char* __restrict__ q, float* __restrict__ qs,
    unsigned int* __restrict__ out_idx, float* __restrict__ out_w,
    uint32_t n, uint32_t n_expert, uint32_t k, float eps, uint32_t batch) {
    extern __shared__ unsigned char hb_sh[];
    __nv_bfloat16* srn = (__nv_bfloat16*)hb_sh;                 // [8][n]
    float* slog = (float*)(hb_sh + 8u * (size_t)n * 2u);        // [8][n_expert]
    const uint32_t b0 = blockIdx.x * 8u;
    const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
    const uint32_t nlive = (batch > b0) ? min(8u, batch - b0) : 0u;
    if (nlive == 0) return;
    const uint32_t t = warp;
    if (t < nlive) {
        const uint32_t b = b0 + t;
        const float* xb = x + (size_t)b * n;
        float* pb = pn + (size_t)b * n;
        signed char* qb = q + (size_t)b * n;
        float* sb = qs + (size_t)b * (n >> 5);
        float acc = 0.0f;
        for (uint32_t i = lane; i < n; i += 32u) {
            const float v = xb[i];
            acc += v * v;
        }
        for (uint32_t sh = 16; sh > 0; sh >>= 1)
            acc += __shfl_down_sync(0xffffffffu, acc, sh);
        acc = __shfl_sync(0xffffffffu, acc, 0);
        const float inv = 1.0f / sqrtf(acc / (float)n + eps);
        for (uint32_t i = lane; i < n; i += 32u) {
            const float xh = xb[i] * inv;
            srn[(size_t)t * n + i] = __float2bfloat16(xh * gamma[i]);
            const float v = xh * pre2[i];
            pb[i] = v;
            float a = fabsf(v);
            for (uint32_t sh = 16; sh > 0; sh >>= 1)
                a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, sh));
            const float scl = a * (1.0f / 127.0f);
            if (lane == 0) sb[i >> 5] = scl;
            const float qinv = scl > 0.0f ? 1.0f / scl : 0.0f;
            int qi = __float2int_rn(v * qinv);
            qi = qi < -127 ? -127 : (qi > 127 ? 127 : qi);
            qb[i] = (signed char)qi;
        }
    }
    __syncthreads();
    // phase B: each warp walks experts o = warp, warp+8, ... - the rw row is
    // read once and feeds all 8 token accumulators.
    for (uint32_t o = warp; o < n_expert; o += 8u) {
        const float* wr = rw + (size_t)o * n;
        float v[8] = {};
        for (uint32_t i = lane; i < n; i += 32u) {
            const float wv = wr[i];
            #pragma unroll
            for (uint32_t tt = 0; tt < 8u; ++tt)
                v[tt] = __fmaf_rn(wv, __bfloat162float(srn[(size_t)tt * n + i]), v[tt]);
        }
        #pragma unroll
        for (uint32_t tt = 0; tt < 8u; ++tt) {
            float vv = v[tt];
            for (uint32_t sh = 16; sh > 0; sh >>= 1)
                vv += __shfl_down_sync(0xffffffffu, vv, sh);
            if (lane == 0 && tt < nlive) slog[(size_t)tt * n_expert + o] = vv;
        }
    }
    __syncthreads();
    if (t < nlive) {
        const uint32_t b = b0 + t;
        unsigned int* oi = out_idx + (size_t)b * k;
        float* ow = out_w + (size_t)b * k;
        pd_moe_topk_warp(slog + (size_t)t * n_expert, (const float*)0, n_expert, k, oi, ow);
        if (lane == 0)
            for (uint32_t s2 = 0; s2 < k; ++s2) ow[s2] *= dscale[oi[s2]];
    }
}

PD_EXPORT
int pd_moe_head_router_hb(const void* x, const void* gamma, const void* pre2,
                          const void* rw, const void* dscale, void* pn, void* q,
                          void* qs, void* out_idx, void* out_w, uint32_t n,
                          uint32_t n_expert, uint32_t k, float eps,
                          uint32_t batch, void* stream) {
    if (n == 0 || batch == 0) return 0;
    if ((n & 31u) || n_expert > 256u || k > 16u) return cudaErrorInvalidValue;
    const uint32_t smem = 8u * n * 2u + 8u * n_expert * 4u;
    static bool at = false;
    if (!at) {
        cudaFuncSetAttribute((const void*)pd_moe_head_router_hb_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)(160u * 1024u));
        at = true;
    }
    pd_moe_head_router_hb_kernel<<<(batch + 7u) / 8u, 256, smem, (cudaStream_t)stream>>>(
        (const float*)x, (const float*)gamma, (const float*)pre2, (const float*)rw,
        (const float*)dscale, (float*)pn, (signed char*)q, (float*)qs,
        (unsigned int*)out_idx, (float*)out_w, n, n_expert, k, eps, batch);
    return pd_launch_status();
}

// topk + per-expert scale fold: pd_moe_topk_warp then w[s] *= scale[idx[s]]
// - element-for-element the ops the separate pd_moe_scale_w launch did, so
// the fused result is bit-identical to the pair.
__global__ void pd_moe_topk_scaled_kernel(const float* __restrict__ logits,
                                          const float* __restrict__ scale,
                                          uint32_t n_expert, uint32_t k,
                                          unsigned int* __restrict__ out_idx,
                                          float* __restrict__ out_w) {
    const uint32_t b = blockIdx.x;
    unsigned int* oi = out_idx + (size_t)b * k;
    float* ow = out_w + (size_t)b * k;
    pd_moe_topk_warp(logits + (size_t)b * n_expert, (const float*)0, n_expert, k, oi, ow);
    if ((threadIdx.x & 31u) == 0)
        for (uint32_t s = 0; s < k; ++s) ow[s] *= scale[oi[s]];
}

PD_EXPORT
int pd_moe_topk_scaled(const void* logits, const void* scale, uint32_t n_expert,
                       uint32_t k, void* out_idx, void* out_w, uint32_t batch,
                       void* stream) {
    if (batch == 0) return 0;
    if (n_expert > 256u || k > 16u) return cudaErrorInvalidValue;
    pd_moe_topk_scaled_kernel<<<batch, 32, 0, (cudaStream_t)stream>>>(
        (const float*)logits, (const float*)scale, n_expert, k,
        (unsigned int*)out_idx, (float*)out_w);
    return pd_launch_status();
}

// ---- prefill dn hybrid (slots 489/490): f8s-gu + v2-dn ---------------------
// At every measured width the BM=128 tc5 down loses to the v2 BM=32 down
// (uni:1024 8192p: 424 vs 213us) while the BM=128 gu wins full tiles. The
// hybrid keeps f8s-gu's f32 output and hands the down half to v2: a pair
// map converts the bm128 CSR row order into the bm32 one, and the GEGLU
// quantize writes q8 fq/fs STRAIGHT into the bm32 rows the v2 down reads.
// fq moves from the e4m3 class to q8 (finer); gates: greedy/coherence (the
// prefill tokens' KV feeds every later output) + the serve cells.

// map[token*n_active + slot] = bm32 row (PD_MOE_PAD rows skipped)
__global__ void pd_moe_pair_map_kernel(const unsigned int* __restrict__ srow32,
                                       const unsigned int* __restrict__ sslot32,
                                       unsigned int* __restrict__ map,
                                       uint32_t n_active, uint32_t srp32) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= srp32) return;
    const unsigned int t = srow32[i];
    if (t == PD_MOE_PAD) return;
    map[(size_t)t * n_active + sslot32[i]] = i;
}

PD_EXPORT
int pd_moe_pair_map(const void* srow32, const void* sslot32, void* map,
                    uint32_t n_active, uint32_t srp32, void* stream) {
    if (srp32 == 0) return 0;
    pd_moe_pair_map_kernel<<<(srp32 + 255u) / 256u, 256, 0, (cudaStream_t)stream>>>(
        (const unsigned int*)srow32, (const unsigned int*)sslot32,
        (unsigned int*)map, n_active, srp32);
    return pd_launch_status();
}

// gu rows are [gate(n_ff) | up(n_ff)] f32 in bm128 order; out fq/fs are q8
// per-32 rows in bm32 order (via map). Same GEGLU/SiLU expressions and the
// same per-32 amax/quantize math as the q8 pair epilogues. One CTA per
// live bm128 row; PAD rows exit (their bm32 twins are written by nothing -
// dead columns feed dead accumulators only, the v2 contract).
__global__ void pd_quantize_q8_geglu_remap_kernel(
    const float* __restrict__ gu, const unsigned int* __restrict__ srow128,
    const unsigned int* __restrict__ sslot128, const unsigned int* __restrict__ map,
    int8_t* __restrict__ fq, float* __restrict__ fs, uint32_t n_ff,
    uint32_t n_active, uint32_t srp128, uint32_t act) {
    // B1: 8 bm128 rows per CTA, 128 lanes each - same
    // per-element math, same per-32 amax groups (stride-128 keeps each warp
    // on one aligned 32-group per iteration; max is order-free-exact).
    const uint32_t sub = threadIdx.x / 128u;
    const uint32_t j = threadIdx.x & 127u;
    const uint32_t r128 = blockIdx.x * 8u + sub;
    if (r128 >= srp128) return;
    const unsigned int tok = srow128[r128];
    if (tok == PD_MOE_PAD) return;
    const size_t out_row = map[(size_t)tok * n_active + sslot128[r128]];
    const float* g = gu + (size_t)r128 * 2u * n_ff;
    const float* u = g + n_ff;
    int8_t* qr = fq + out_row * n_ff;
    float* sr = fs + out_row * (n_ff >> 5);
    const uint32_t lane = threadIdx.x & 31u;
    for (uint32_t i = j; i < n_ff; i += 128u) {
        const float gv = g[i];
        const float uv = u[i];
        const float v = act
            ? (gv / (1.0f + __expf(-gv))) * uv
            : 0.5f * gv
                  * (1.0f
                     + tanhf(0.79788456080286535587989211986876f * gv
                             * (1.0f + 0.044715f * gv * gv)))
                  * uv;
        float a = fabsf(v);
        for (uint32_t sh = 16; sh > 0; sh >>= 1)
            a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, sh));
        const float scl = a * (1.0f / 127.0f);
        if (lane == 0) sr[i >> 5] = scl;
        const float qinv = scl > 0.0f ? 1.0f / scl : 0.0f;
        int qi = __float2int_rn(v * qinv);
        qi = qi < -127 ? -127 : (qi > 127 ? 127 : qi);
        qr[i] = (int8_t)qi;
    }
}

PD_EXPORT
int pd_quantize_q8_geglu_remap(const void* gu, const void* srow128,
                               const void* sslot128, const void* map, void* fq,
                               void* fs, uint32_t n_ff, uint32_t n_active,
                               uint32_t srp128, uint32_t act, void* stream) {
    if (srp128 == 0 || n_ff == 0) return 0;
    if (n_ff & 31u) return cudaErrorInvalidValue;
    pd_quantize_q8_geglu_remap_kernel<<<(srp128 + 7u) / 8u, 1024, 0, (cudaStream_t)stream>>>(
        (const float*)gu, (const unsigned int*)srow128,
        (const unsigned int*)sslot128, (const unsigned int*)map, (int8_t*)fq,
        (float*)fs, n_ff, n_active, srp128, act);
    return pd_launch_status();
}

// head+router+topk in one launch (slot 487; g26a4b act): the head,
// the router GEMV and the scaled top-k were three serial nodes (4.7 + 15.8
// + 6.2us + joints per layer-tick at c32). The router-normed row never
// leaves smem, and each logit is computed by one warp with the exact tile
// matvec walk (i = lane; i += 32; shfl_down tree) - logits, top-k and the
// scale fold are all BIT-IDENTICAL to the three-launch chain. rn is not
// written to gmem (its only consumer was the matvec).
__global__ void pd_moe_head_router_kernel(
    const float* __restrict__ x, const float* __restrict__ gamma,
    const float* __restrict__ pre2, const float* __restrict__ rw,
    const float* __restrict__ dscale, float* __restrict__ pn,
    signed char* __restrict__ q, float* __restrict__ qs,
    unsigned int* __restrict__ out_idx, float* __restrict__ out_w,
    uint32_t n, uint32_t n_expert, uint32_t k, float eps) {
    extern __shared__ float hr_sh[];       // [n] router-normed row | [n_expert] logits
    float* srn = hr_sh;
    float* slog = hr_sh + n;
    const uint32_t b = blockIdx.x;
    const float* xb = x + (size_t)b * n;
    float* pb = pn + (size_t)b * n;
    signed char* qb = q + (size_t)b * n;
    float* sb = qs + (size_t)b * (n >> 5);
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    __shared__ float wsum[32];
    __shared__ float s_inv;
    float acc = 0.0f;
    for (uint32_t i = tid; i < n; i += nth) {
        const float v = xb[i];
        acc += v * v;
    }
    for (uint32_t sh = 16; sh > 0; sh >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, sh);
    const uint32_t warp = tid >> 5, lane = tid & 31u;
    if (lane == 0) wsum[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        float sum = 0.0f;
        const uint32_t nwarps = (nth + 31u) >> 5;
        for (uint32_t wi = 0; wi < nwarps; ++wi) sum += wsum[wi];
        s_inv = 1.0f / sqrtf(sum / (float)n + eps);
    }
    __syncthreads();
    const float inv = s_inv;
    for (uint32_t i = tid; i < n; i += nth) {
        const float xh = xb[i] * inv;
        srn[i] = xh * gamma[i];
        const float v = xh * pre2[i];
        pb[i] = v;
        float a = fabsf(v);
        for (uint32_t sh = 16; sh > 0; sh >>= 1)
            a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, sh));
        const float scl = a * (1.0f / 127.0f);
        if (lane == 0) sb[i >> 5] = scl;
        const float qinv = scl > 0.0f ? 1.0f / scl : 0.0f;
        int qi = __float2int_rn(v * qinv);
        qi = qi < -127 ? -127 : (qi > 127 ? 127 : qi);
        qb[i] = (signed char)qi;
    }
    __syncthreads();
    // router: warp per logit, the tile matvec walk verbatim (bit-identical
    // partial sums per lane, same shfl_down tree)
    const uint32_t nwarps = nth >> 5;
    for (uint32_t o = warp; o < n_expert; o += nwarps) {
        const float* wr = rw + (size_t)o * n;
        float v = 0.0f;
        // explicit fused mul-add: the tile matvec's `acc += wv * x` contracts
        // to FMA; leaving this to the scheduler produced a double-rounded dot
        // and a CONSISTENT +0.049-nat PPL shift (three windows) - noisier
        // logits systematically degrade near-tie routing.
        for (uint32_t i = lane; i < n; i += 32u) v = __fmaf_rn(wr[i], srn[i], v);
        for (uint32_t sh = 16; sh > 0; sh >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, sh);
        if (lane == 0) slog[o] = v;
    }
    __syncthreads();
    if (warp == 0) {
        unsigned int* oi = out_idx + (size_t)b * k;
        float* ow = out_w + (size_t)b * k;
        pd_moe_topk_warp(slog, (const float*)0, n_expert, k, oi, ow);
        if (lane == 0)
            for (uint32_t s2 = 0; s2 < k; ++s2) ow[s2] *= dscale[oi[s2]];
    }
}

PD_EXPORT
int pd_moe_head_router(const void* x, const void* gamma, const void* pre2,
                       const void* rw, const void* dscale, void* pn, void* q,
                       void* qs, void* out_idx, void* out_w, uint32_t n,
                       uint32_t n_expert, uint32_t k, float eps, uint32_t batch,
                       void* stream) {
    if (n == 0 || batch == 0) return 0;
    if ((n & 31u) || n_expert > 256u || k > 16u) return cudaErrorInvalidValue;
    const uint32_t smem = (n + n_expert) * 4u;
    pd_moe_head_router_kernel<<<batch, (batch >= 64u ? pd_norm_wide_nth(batch) : 1024u),
                                smem, (cudaStream_t)stream>>>(
        (const float*)x, (const float*)gamma, (const float*)pre2, (const float*)rw,
        (const float*)dscale, (float*)pn, (signed char*)q, (float*)qs,
        (unsigned int*)out_idx, (float*)out_w, n, n_expert, k, eps);
    return pd_launch_status();
}

//  routing diagnostic: uniq-experts-per-(tick,layer) histogram. One
// tiny block ORs the routed pair ids into a 128-bit presence bitmap and
// bumps a persistent device accumulator: hist[uniq]++, pairs_sum[uniq] +=
// pairs, plus invocation/pairs totals - the number that prices a
// decode-band expert kernel's true weight bytes (uniform kbench cells
// assume uniq = min(pairs, n_expert); real routing overlaps). Reads only;
// the engine arms it off PADDOCK_MOE_UNIQ, never in the default path.
// DESIGNED FOR GRAPH CAPTURE: serving decode ticks replay captured
// (r, k1) graphs, so this launch gets baked in and re-reads the moe_idx
// the captured topk refreshes - accumulation stays live on every replay
// (pairs is baked per graph, correct because r is the graph key). The
// accumulator must be a NON-POOL allocation (a first-decode
// sweep overwrites persistent passive pool buffers). Four 260-u32 regions
// banded by pairs (<=64 / <=256 / <=1024 / >1024 - banding by pairs, not
// the engine's pf flag, because pf picks a buffer lane and batched decode
// rides the pf lane too). Region layout: [0,129) hist by uniq, [129,258)
// pairs sums by uniq, [258] invocations, [259] pairs.
__global__ void pd_moe_uniq_hist_kernel(const unsigned int* __restrict__ idx,
                                        int pairs,
                                        unsigned int* __restrict__ out) {
    __shared__ unsigned int pres[4]; // 128-bit presence bitmap
    if (threadIdx.x < 4) pres[threadIdx.x] = 0u;
    __syncthreads();
    for (int i = threadIdx.x; i < pairs; i += blockDim.x) {
        unsigned int e = idx[i];
        // ids >= 128 are shared-expert pseudo-picks (the nemotron _sh topk
        // appends them past n_expert): a constant fetch per launch, not
        // routing - skip them so uniq stays ROUTED uniq (and the bitmap
        // stays in bounds; unguarded this was an OOB shared write)
        if (e < 128u) atomicOr(&pres[e >> 5], 1u << (e & 31u));
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int band = pairs <= 64 ? 0 : pairs <= 256 ? 1 : pairs <= 1024 ? 2 : 3;
        unsigned int* o = out + band * 260;
        int u = __popc(pres[0]) + __popc(pres[1]) + __popc(pres[2]) +
                __popc(pres[3]);
        atomicAdd(o + u, 1u);
        atomicAdd(o + 129 + u, (unsigned int)pairs);
        atomicAdd(o + 258, 1u);
        atomicAdd(o + 259, (unsigned int)pairs);
    }
}

PD_EXPORT
int pd_moe_uniq_hist(const void* idx, uint32_t pairs, uint32_t n_expert,
                     void* out, void* stream) {
    if (pairs == 0) return 0;
    if (n_expert > 128u) return cudaErrorInvalidValue;
    pd_moe_uniq_hist_kernel<<<1, 128, 0, (cudaStream_t)stream>>>(
        (const unsigned int*)idx, (int)pairs, (unsigned int*)out);
    return pd_launch_status();
}

// combine trailer: x = (x + rmsnorm(rmsnorm(proj)*pn1 + rmsnorm(dn)*pn2)
//                          * postw) * os   - 4 launches folded to 1.
// The branch sum s is staged in smem (n <= 12288 floats) between the second
// and third normalizations; three block reductions, same shfl tree as the
// other norm fusions.
__global__ void pd_moe_tail_kernel(float* __restrict__ x, const float* __restrict__ proj,
                                   const float* __restrict__ dn,
                                   const float* __restrict__ pn1,
                                   const float* __restrict__ pn2,
                                   const float* __restrict__ postw, uint32_t n,
                                   float eps, float os) {
    extern __shared__ float pd_mt_s[];   // s row [n] + 32 reduction slots
    float* srow = pd_mt_s;
    float* wsum = pd_mt_s + n;
    __shared__ float s_va, s_vb;
    const uint32_t b = blockIdx.x;
    float* xb = x + (size_t)b * n;
    const float* pr = proj + (size_t)b * n;
    const float* db = dn + (size_t)b * n;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    const uint32_t warp = tid >> 5, lane = tid & 31u, nwarps = (nth + 31u) >> 5;
    float ap = 0.0f, ad = 0.0f;
    for (uint32_t i = tid; i < n; i += nth) {
        const float p = pr[i], d = db[i];
        ap += p * p;
        ad += d * d;
    }
    for (uint32_t sh = 16; sh > 0; sh >>= 1) {
        ap += __shfl_down_sync(0xffffffffu, ap, sh);
        ad += __shfl_down_sync(0xffffffffu, ad, sh);
    }
    if (lane == 0) { wsum[warp] = ap; wsum[warp + 32u] = ad; }
    __syncthreads();
    if (tid == 0) {
        float sp = 0.0f, sd = 0.0f;
        for (uint32_t wi = 0; wi < nwarps; ++wi) { sp += wsum[wi]; sd += wsum[wi + 32u]; }
        s_va = 1.0f / sqrtf(sp / (float)n + eps);
        s_vb = 1.0f / sqrtf(sd / (float)n + eps);
    }
    __syncthreads();
    const float ip = s_va, id_ = s_vb;
    float as = 0.0f;
    for (uint32_t i = tid; i < n; i += nth) {
        const float s = pr[i] * ip * pn1[i] + db[i] * id_ * pn2[i];
        srow[i] = s;
        as += s * s;
    }
    for (uint32_t sh = 16; sh > 0; sh >>= 1) as += __shfl_down_sync(0xffffffffu, as, sh);
    if (lane == 0) wsum[warp] = as;
    __syncthreads();
    if (tid == 0) {
        float ss = 0.0f;
        for (uint32_t wi = 0; wi < nwarps; ++wi) ss += wsum[wi];
        s_va = 1.0f / sqrtf(ss / (float)n + eps);
    }
    __syncthreads();
    const float is = s_va;
    for (uint32_t i = tid; i < n; i += nth)
        xb[i] = (xb[i] + srow[i] * is * postw[i]) * os;
}

// tail+combine fold (slot 491): the slot-combine's ascending-k sum happens
// at the tail's two dn reads instead of through a moe_xn round trip - one
// launch and one 360KB write+read pass gone per layer-tick. d's fold order
// is exactly pd_moe_slot_combine_init's, so the result is BITWISE the
// combine_init -> moe_tail chain.
__global__ void pd_moe_tail_combine_kernel(
    float* __restrict__ x, const float* __restrict__ proj,
    const float* __restrict__ part, const float* __restrict__ pn1,
    const float* __restrict__ pn2, const float* __restrict__ postw, uint32_t n,
    uint32_t n_active, float eps, float os) {
    extern __shared__ float pd_mtc_s[];
    float* srow = pd_mtc_s;
    float* wsum = pd_mtc_s + n;
    __shared__ float s_va, s_vb;
    const uint32_t b = blockIdx.x;
    float* xb = x + (size_t)b * n;
    const float* pr = proj + (size_t)b * n;
    const float* pb = part + (size_t)b * n_active * n;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    const uint32_t warp = tid >> 5, lane = tid & 31u, nwarps = (nth + 31u) >> 5;
    float ap = 0.0f, ad = 0.0f;
    for (uint32_t i = tid; i < n; i += nth) {
        const float p = pr[i];
        float d = 0.0f;
        for (uint32_t kk = 0; kk < n_active; ++kk) d += pb[(size_t)kk * n + i];
        ap += p * p;
        ad += d * d;
    }
    for (uint32_t sh = 16; sh > 0; sh >>= 1) {
        ap += __shfl_down_sync(0xffffffffu, ap, sh);
        ad += __shfl_down_sync(0xffffffffu, ad, sh);
    }
    if (lane == 0) { wsum[warp] = ap; wsum[warp + 32u] = ad; }
    __syncthreads();
    if (tid == 0) {
        float sp = 0.0f, sd = 0.0f;
        for (uint32_t wi = 0; wi < nwarps; ++wi) { sp += wsum[wi]; sd += wsum[wi + 32u]; }
        s_va = 1.0f / sqrtf(sp / (float)n + eps);
        s_vb = 1.0f / sqrtf(sd / (float)n + eps);
    }
    __syncthreads();
    const float ip = s_va, id_ = s_vb;
    float as = 0.0f;
    for (uint32_t i = tid; i < n; i += nth) {
        float d = 0.0f;
        for (uint32_t kk = 0; kk < n_active; ++kk) d += pb[(size_t)kk * n + i];
        const float s = pr[i] * ip * pn1[i] + d * id_ * pn2[i];
        srow[i] = s;
        as += s * s;
    }
    for (uint32_t sh = 16; sh > 0; sh >>= 1) as += __shfl_down_sync(0xffffffffu, as, sh);
    if (lane == 0) wsum[warp] = as;
    __syncthreads();
    if (tid == 0) {
        float ss = 0.0f;
        for (uint32_t wi = 0; wi < nwarps; ++wi) ss += wsum[wi];
        s_va = 1.0f / sqrtf(ss / (float)n + eps);
    }
    __syncthreads();
    const float is = s_va;
    for (uint32_t i = tid; i < n; i += nth)
        xb[i] = (xb[i] + srow[i] * is * postw[i]) * os;
}

// hibatch P1-1 twin: tail+combine over BF16 partials (down stored bf16); the
// slot sum stays f32 in the same ascending-k order - precision-class is the
// bf16 round at the store only. Body = pd_moe_tail_combine_kernel with the
// part loads converted.
__global__ void pd_moe_tail_combine_bf16_kernel(
    float* __restrict__ x, const float* __restrict__ proj,
    const __nv_bfloat16* __restrict__ part, const float* __restrict__ pn1,
    const float* __restrict__ pn2, const float* __restrict__ postw, uint32_t n,
    uint32_t n_active, float eps, float os) {
    extern __shared__ float pd_mtcb_s[];
    float* srow = pd_mtcb_s;
    float* wsum = pd_mtcb_s + n;
    __shared__ float s_va, s_vb;
    const uint32_t b = blockIdx.x;
    float* xb = x + (size_t)b * n;
    const float* pr = proj + (size_t)b * n;
    const __nv_bfloat16* pb = part + (size_t)b * n_active * n;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    const uint32_t warp = tid >> 5, lane = tid & 31u, nwarps = (nth + 31u) >> 5;
    float ap = 0.0f, ad = 0.0f;
    for (uint32_t i = tid; i < n; i += nth) {
        const float p = pr[i];
        float d = 0.0f;
        for (uint32_t kk = 0; kk < n_active; ++kk)
            d += __bfloat162float(pb[(size_t)kk * n + i]);
        ap += p * p;
        ad += d * d;
    }
    for (uint32_t sh = 16; sh > 0; sh >>= 1) {
        ap += __shfl_down_sync(0xffffffffu, ap, sh);
        ad += __shfl_down_sync(0xffffffffu, ad, sh);
    }
    if (lane == 0) { wsum[warp] = ap; wsum[warp + 32u] = ad; }
    __syncthreads();
    if (tid == 0) {
        float sp = 0.0f, sd = 0.0f;
        for (uint32_t wi = 0; wi < nwarps; ++wi) { sp += wsum[wi]; sd += wsum[wi + 32u]; }
        s_va = 1.0f / sqrtf(sp / (float)n + eps);
        s_vb = 1.0f / sqrtf(sd / (float)n + eps);
    }
    __syncthreads();
    const float ip = s_va, id_ = s_vb;
    float as = 0.0f;
    for (uint32_t i = tid; i < n; i += nth) {
        float d = 0.0f;
        for (uint32_t kk = 0; kk < n_active; ++kk)
            d += __bfloat162float(pb[(size_t)kk * n + i]);
        const float s = pr[i] * ip * pn1[i] + d * id_ * pn2[i];
        srow[i] = s;
        as += s * s;
    }
    for (uint32_t sh = 16; sh > 0; sh >>= 1) as += __shfl_down_sync(0xffffffffu, as, sh);
    if (lane == 0) wsum[warp] = as;
    __syncthreads();
    if (tid == 0) {
        float ss = 0.0f;
        for (uint32_t wi = 0; wi < nwarps; ++wi) ss += wsum[wi];
        s_va = 1.0f / sqrtf(ss / (float)n + eps);
    }
    __syncthreads();
    const float is = s_va;
    for (uint32_t i = tid; i < n; i += nth)
        xb[i] = (xb[i] + srow[i] * is * postw[i]) * os;
}

PD_EXPORT
int pd_moe_tail_combine_bf16(void* x, const void* proj, const void* part,
                             const void* pn1, const void* pn2, const void* postw,
                             uint32_t n, uint32_t n_active, float eps, float os,
                             uint32_t batch, void* stream) {
    if (n == 0 || batch == 0) return 0;
    if (n & 31u) return cudaErrorInvalidValue;
    const uint32_t smem = (n + 64u) * 4u;
    if (smem > 96u * 1024u) return cudaErrorInvalidValue;
    static bool at = false;
    if (!at) {
        cudaFuncSetAttribute((const void*)pd_moe_tail_combine_bf16_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(96u * 1024u));
        at = true;
    }
    pd_moe_tail_combine_bf16_kernel<<<batch, (batch >= 64u ? pd_norm_wide_nth(batch) : 1024u),
                                      smem, (cudaStream_t)stream>>>(
        (float*)x, (const float*)proj, (const __nv_bfloat16*)part, (const float*)pn1,
        (const float*)pn2, (const float*)postw, n, n_active, eps, os);
    return pd_launch_status();
}

PD_EXPORT
int pd_moe_tail_combine(void* x, const void* proj, const void* part,
                        const void* pn1, const void* pn2, const void* postw,
                        uint32_t n, uint32_t n_active, float eps, float os,
                        uint32_t batch, void* stream) {
    if (n == 0 || batch == 0) return 0;
    if (n & 31u) return cudaErrorInvalidValue;
    const uint32_t smem = (n + 64u) * 4u;
    if (smem > 96u * 1024u) return cudaErrorInvalidValue;
    static bool at = false;
    if (!at) {
        cudaFuncSetAttribute((const void*)pd_moe_tail_combine_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(96u * 1024u));
        at = true;
    }
    pd_moe_tail_combine_kernel<<<batch, (batch >= 64u ? pd_norm_wide_nth(batch) : 1024u),
                                 smem, (cudaStream_t)stream>>>(
        (float*)x, (const float*)proj, (const float*)part, (const float*)pn1,
        (const float*)pn2, (const float*)postw, n, n_active, eps, os);
    return pd_launch_status();
}

PD_EXPORT
int pd_moe_tail(void* x, const void* proj, const void* dn, const void* pn1,
                const void* pn2, const void* postw, uint32_t n, float eps, float os,
                uint32_t batch, void* stream) {
    if (n == 0 || batch == 0) return 0;
    if (n & 31u) return cudaErrorInvalidValue;
    const uint32_t smem = (n + 64u) * 4u;
    if (smem > 96u * 1024u) return cudaErrorInvalidValue;
    static bool at = false;
    if (!at) {
        cudaFuncSetAttribute((const void*)pd_moe_tail_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(96u * 1024u));
        at = true;
    }
    pd_moe_tail_kernel<<<batch, (batch >= 64u ? pd_norm_wide_nth(batch) : 1024u), smem, (cudaStream_t)stream>>>(
        (float*)x, (const float*)proj, (const float*)dn, (const float*)pn1,
        (const float*)pn2, (const float*)postw, n, eps, os);
    return pd_launch_status();
}
