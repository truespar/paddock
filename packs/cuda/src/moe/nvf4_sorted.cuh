// moe/nvf4_sorted.cuh - NVFP4 MoE over the SORTED layout: the sorted expert
// GEMMs, the sm_100 sorted-tile weight-only bf16 arm, and the decode
// multi-task expert GEMVs.
// Textually-included segment of the single pack translation unit.
// Not standalone-compilable: include order is defined by ../pack.cu.
//
// Split out of quant/nvf4.cuh (see moe/nvf4_expert.cuh for the
// split and the domain reasoning).
//
// Include after moe/nvf4_expert.cuh.
//
// The sm_100 sorted-tile arm here is arch-gated, so it compiles to empty
// bodies on sm_120 - changes in this file need an sm_100 build to be verified
// at all.
// ---- NVFP4 MoE expert GEMMs over the sorted layout  ------
// The GEMV pair above is warp-per-(slot, row) with no cross-token tiling -
// profiled at 91% of nemotron's bulk-prefill GPU time (nemo-prefill3). These
// two kernels are the prefill class: moe_align groups the routed (token,
// pick) pairs into BM=32 sorted blocks per expert, and each CTA runs a
// 128-output-row x 32-token block-scale MMA tile that reads the expert's
// weight strip once per block instead of once per slot. Skeleton is
// pd_mxfp4_gemm_nv4_kernel verbatim (cp.async commit-group double buffer,
// KC=128, 2 CTAs/SM, kind::mxf4nvf4 m16n8k64) with the token side narrowed
// 128 -> 32 and gathered/scattered through the moe_align layout - per-acc
// K-accumulation order is identical to the dense kernel (kt ascending, k64
// ascending), which is what the bit-exact unit gates lean on. NUMERIC CLASS:
// activations ride nvf4 (W4A4) instead of the GEMV pair's f32 (W4A16) -
// quality-gated at the model level; decode stays on the
// GEMV pair (b=1 is bandwidth-bound, kernel shape not weight class).
// The warp-specialized mbarrier form (gate_up_bs precedent) is the follow-up
// if profiling shows the per-chunk barrier turnaround binding at 2 CTAs/SM.
//
// Rung 7: both kernels are templated on KB = K-chunk depth in
// 32-element blocks (KC = KB*32). KB=4 is the original KC=128 geometry;
// KB=8 (KC=256) doubles the per-row read granularity - the stream-pattern
// facts (block_scale_quant.cuh): 64 B reads at ~1.4 KB row
// stride run 940 GB/s, 128 B reads 1580. Unlike A4B, the doubled tile still
// fits two CTAs/SM here (46080 B dynamic smem at KB=8, under the ~50 KB/
// block sm_120a ceiling), so the full-CTA-barrier double buffer keeps its
// co-resident turnaround hiding and no warp-specialized rewrite is needed.
// Flattened (kt, k64) accumulate order is identical across KB - the
// bit-exact unit gates hold for both instantiations.
// PADDOCK_NV4M_KC128=1 pins the KB=4 arm (A/B kill switch).
#define PD_NV4M_BM 32u
// KB*16 B packed fp4 + KB*2 e4m3 scale bytes, padded to 16 B (KB=4 -> 80,
// the original PD_F4_WROW; KB=8 -> 144)
#define PD_NV4M_WROW(KB) ((((KB) * 18u) + 15u) & ~15u)
// 16 B scale header (KB*2 bytes used) + KB*16 B packed fp4
#define PD_NV4M_YROW(KB) (16u + (KB) * 16u)
#define PD_NV4M_SMEM(KB) \
    (2u * 128u * PD_NV4M_WROW(KB) + 2u * PD_NV4M_BM * PD_NV4M_YROW(KB))

static inline uint32_t pd_nv4m_kb() {
    static const uint32_t kb =
        (pd_env("PADDOCK_NV4M_KC128") != nullptr) ? 4u : 8u;
    return kb;
}
// NOTE: the SFOLD arm below is COMPILE-TIME only and every production launch
// passes false. It is not env-selectable deliberately - it wins only where the
// die is under-filled (see the entry), and a switch that picks a
// regime is a missing structure, not a fix. An out-of-tree bench is the one
// caller that instantiates it true.

// Sorted-tile expert up + squared-relu, re-quantized to nvf4 in REGISTERS:
// fq/fs are sorted-position indexed ([nb*32, ff/2] + [nb*32, ff/16]) - the
// down kernel's direct B input, no gather. B rows gather token activations
// (xq/xs, pd_quantize_nvf4 planes) via sorted_row; PAD columns compute on
// zero-filled tiles and emit exact-zero blocks (scale byte 0). Epilogue
// order matches the GEMV: scale2[e] first, then relu, then square; the
// per-16-along-ff quantize is the bs_gu epilogue's shuffle scheme with the
// nvf4 scale pick (amax/6 RN-e4m3, exactly pd_nvf4_quant8's math).
// SFOLD folds the e4m3 scale planes into the same cp.async pipeline as the
// packed data (Q8_0's mmq family already stages its scales that way, via
// pd_cpa4p). Without it the scale bytes are plain global loads issued after
// cp.async.wait_group and before __syncthreads - one full memory latency per
// K-chunk that nothing prefetches, covered only by the other resident CTA.
// 4-byte grain, because KB*2=16 does not divide the scale row (in_dim 2688 ->
// n_k16 168, 1856 -> 116) while 4 always does; src_ok=false zero-fills,
// matching the synchronous path's `: 0u` exactly, so the staged bytes - and
// therefore the result - are identical either way.
template <uint32_t KB, bool SFOLD>
__global__ void __launch_bounds__(256, 2) pd_nvf4_moe_up_relu2_bs_kernel(
    const uint8_t* __restrict__ data, const uint8_t* __restrict__ scale,
    const float* __restrict__ scale2, const uint32_t* __restrict__ sorted_row,
    const uint32_t* __restrict__ block_expert, const uint8_t* __restrict__ xq,
    const uint8_t* __restrict__ xs, uint8_t* __restrict__ fq,
    uint8_t* __restrict__ fs, uint32_t in_dim, uint32_t ff) {
#if PD_BS_OK
    constexpr uint32_t WROW = PD_NV4M_WROW(KB);
    constexpr uint32_t YROW = PD_NV4M_YROW(KB);
    const uint32_t blk = blockIdx.x;
    const uint32_t e = block_expert[blk];
    if (e == PD_MOE_PAD) return;
    const uint32_t row_base = blockIdx.y * 128u;

    extern __shared__ unsigned char pd_bs_sh[];
    unsigned char* wb0 = pd_bs_sh;
    unsigned char* wb1 = wb0 + 128u * WROW;
    unsigned char* yb0 = wb1 + 128u * WROW;
    unsigned char* yb1 = yb0 + PD_NV4M_BM * YROW;
    __shared__ uint32_t tok[PD_NV4M_BM];

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, tq = lane & 3u;
    const uint32_t i0 = (warp >> 1) * 32u;
    const uint32_t joff = (warp & 1u) * 8u;
    const uint32_t n_kb = in_dim >> 5;
    const uint32_t n_k16 = in_dim >> 4;
    const uint32_t nk = (in_dim + KB * 32u - 1u) / (KB * 32u);
    const size_t wrow0 = (size_t)e * ff + row_base;
    const bool sfold = SFOLD && (n_k16 & 3u) == 0u;

    if (tid < PD_NV4M_BM) tok[tid] = sorted_row[(size_t)blk * PD_NV4M_BM + tid];
    __syncthreads();

    float acc[4][4] = {};

    #define PD_NV4M_ISSUE_W(dst, kt)                                                  \
        for (uint32_t u = tid; u < 128u * KB; u += 256u) {                            \
            const uint32_t row = u / KB, seg = u % KB;                                \
            const bool ok = (row_base + row) < ff && (kt) * KB + seg < n_kb;          \
            pd_cp_async16((int*)((dst) + row * WROW + seg * 16u),                     \
                          data + (wrow0 + row) * (size_t)(in_dim >> 1) +              \
                              (kt) * (KB * 16u) + seg * 16u,                          \
                          ok);                                                        \
        }
    #define PD_NV4M_ISSUE_Y(dst, kt)                                                  \
        for (uint32_t u = tid; u < PD_NV4M_BM * KB; u += 256u) {                      \
            const uint32_t col = u / KB, seg = u % KB;                                \
            const uint32_t r = tok[col];                                              \
            const bool ok = r != PD_MOE_PAD && (kt) * KB + seg < n_kb;                \
            pd_cp_async16((int*)((dst) + col * YROW + 16u + seg * 16u),               \
                          xq + ((size_t)(ok ? r : 0u) * in_dim >> 1) +                \
                              (kt) * (KB * 16u) + seg * 16u,                          \
                          ok);                                                        \
        }
    #define PD_NV4M_ISSUE_WS(dst, kt)                                                 \
        for (uint32_t u = tid; u < 128u * (KB / 2u); u += 256u) {                     \
            const uint32_t row = u / (KB / 2u), q = u % (KB / 2u);                    \
            const bool ok = (row_base + row) < ff &&                                  \
                            (kt) * (KB * 2u) + q * 4u + 4u <= n_k16;                  \
            pd_cpa4p((dst) + row * WROW + KB * 16u + q * 4u,                          \
                     scale + (wrow0 + row) * (size_t)n_k16 + (kt) * (KB * 2u) +       \
                         q * 4u,                                                      \
                     ok);                                                             \
        }
    #define PD_NV4M_ISSUE_YS(dst, kt)                                                 \
        for (uint32_t u = tid; u < PD_NV4M_BM * (KB / 2u); u += 256u) {               \
            const uint32_t col = u / (KB / 2u), q = u % (KB / 2u);                    \
            const uint32_t r = tok[col];                                              \
            const bool ok = r != PD_MOE_PAD &&                                        \
                            (kt) * (KB * 2u) + q * 4u + 4u <= n_k16;                  \
            pd_cpa4p((dst) + col * YROW + q * 4u,                                     \
                     xs + (size_t)(ok ? r : 0u) * n_k16 + (kt) * (KB * 2u) + q * 4u,  \
                     ok);                                                             \
        }

    PD_NV4M_ISSUE_W(wb0, 0u)
    PD_NV4M_ISSUE_Y(yb0, 0u)
    if (sfold) { PD_NV4M_ISSUE_WS(wb0, 0u) PD_NV4M_ISSUE_YS(yb0, 0u) }
    asm volatile("cp.async.commit_group;");
    for (uint32_t kt = 0; kt < nk; ++kt) {
        unsigned char* tw = (kt & 1u) ? wb1 : wb0;
        unsigned char* ty = (kt & 1u) ? yb1 : yb0;
        if (kt + 1u < nk) {
            PD_NV4M_ISSUE_W((kt & 1u) ? wb0 : wb1, kt + 1u)
            PD_NV4M_ISSUE_Y((kt & 1u) ? yb0 : yb1, kt + 1u)
            if (sfold) {
                PD_NV4M_ISSUE_WS((kt & 1u) ? wb0 : wb1, kt + 1u)
                PD_NV4M_ISSUE_YS((kt & 1u) ? yb0 : yb1, kt + 1u)
            }
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        if (!sfold) {   // e4m3 scale planes: W KB*2 bytes/row, Y 32 rows x KB*2
            for (uint32_t u = tid; u < 128u * KB * 2u; u += 256u) {
                const uint32_t row = u / (KB * 2u), kb16 = u % (KB * 2u);
                const bool wok = (row_base + row) < ff && kt * (KB * 2u) + kb16 < n_k16;
                tw[row * WROW + KB * 16u + kb16] =
                    wok ? scale[(wrow0 + row) * (size_t)n_k16 + kt * (KB * 2u) + kb16]
                        : 0u;
            }
            for (uint32_t u = tid; u < PD_NV4M_BM * KB * 2u; u += 256u) {
                const uint32_t row = u / (KB * 2u), kb16 = u % (KB * 2u);
                const uint32_t r = tok[row];
                const bool yok = r != PD_MOE_PAD && kt * (KB * 2u) + kb16 < n_k16;
                ty[row * YROW + kb16] =
                    yok ? xs[(size_t)r * n_k16 + kt * (KB * 2u) + kb16] : 0u;
            }
        }
        __syncthreads();

        uint32_t am[2][KB / 2u][4], sa[2][KB / 2u];
        #pragma unroll
        for (uint32_t n = 0; n < 2u; ++n) {
            const uint32_t r0 = i0 + n * 16u + g;
            const uint32_t rs = (tq & 1u) ? r0 + 8u : r0;
            #pragma unroll
            for (uint32_t k64 = 0; k64 < KB / 2u; ++k64) {
                pd_ldm_x4(am[n][k64],
                          tw + (i0 + n * 16u + ((lane >> 3) & 1u) * 8u + (lane & 7u)) *
                                  WROW +
                              k64 * 32u + (lane >> 4) * 16u);
                sa[n][k64] =
                    *(const uint32_t*)(tw + rs * WROW + KB * 16u + k64 * 4u);
            }
        }
        #pragma unroll
        for (uint32_t j0 = 0; j0 < PD_NV4M_BM; j0 += 16u) {
            uint32_t bm[2u * (KB / 2u)];
            #pragma unroll
            for (uint32_t q = 0; q < KB / 4u; ++q)
                pd_ldm_x4(bm + q * 4u, ty + (j0 + joff + (lane & 7u)) * YROW + 16u +
                                           q * 64u + (lane >> 3) * 16u);
            const unsigned char* ysr = ty + (j0 + joff + g) * YROW;
            #pragma unroll
            for (uint32_t k64 = 0; k64 < KB / 2u; ++k64) {
                const uint32_t sb = *(const uint32_t*)(ysr + k64 * 4u);
                #pragma unroll
                for (uint32_t n = 0; n < 2u; ++n)
                    pd_nv4_mma(acc[(j0 >> 3) + n], am[n][k64][0], am[n][k64][1],
                               am[n][k64][2], am[n][k64][3], bm[k64 * 2u],
                               bm[k64 * 2u + 1u], sa[n][k64], sb);
            }
        }
        __syncthreads();
    }
    #undef PD_NV4M_ISSUE_W
    #undef PD_NV4M_ISSUE_Y
    #undef PD_NV4M_ISSUE_WS
    #undef PD_NV4M_ISSUE_YS

    // epilogue: v = relu(acc * scale2[e])^2, then nvf4 quantize per 16 ALONG
    // ff (the bs_gu shuffle scheme): at fixed token column the 16-block's
    // values live on this tq's 8 lanes (rows g and g+8); lane g assembles
    // byte g from its neighbours' e2m1 codes. PAD columns emit exact zeros.
    const float s2 = scale2[e];
    const uint32_t tmask = 0x11111111u << tq;  // the 8 lanes of this tq column
    #pragma unroll
    for (uint32_t j0 = 0; j0 < PD_NV4M_BM; j0 += 16u) {
        #pragma unroll
        for (uint32_t n = 0; n < 2u; ++n) {
            const uint32_t rb = row_base + i0 + n * 16u;  // 16-block base row
            #pragma unroll
            for (uint32_t qc = 0; qc < 2u; ++qc) {
                const uint32_t c = j0 + joff + 2u * tq + qc;
                const bool pad = tok[c] == PD_MOE_PAD;
                const float a0 = acc[(j0 >> 3) + n][qc] * s2;
                const float a1 = acc[(j0 >> 3) + n][qc + 2u] * s2;
                const float r0v = fmaxf(a0, 0.0f);
                const float r1v = fmaxf(a1, 0.0f);
                const float v0 = pad ? 0.0f : r0v * r0v;  // row rb + g
                const float v1 = pad ? 0.0f : r1v * r1v;  // row rb + 8 + g
                float a = fmaxf(v0, v1);
                a = fmaxf(a, __shfl_xor_sync(tmask, a, 4));
                a = fmaxf(a, __shfl_xor_sync(tmask, a, 8));
                a = fmaxf(a, __shfl_xor_sync(tmask, a, 16));
                float inv;
                const unsigned sbyte = pd_nvf4_scale(a, &inv);
                const uint32_t n0 = pd_e2m1_rn(v0 * inv);
                const uint32_t n1 = pd_e2m1_rn(v1 * inv);
                // lane g assembles byte g of the block: elems (2g, 2g+1) -
                // rows rb+2g,+2g+1 for g<4 (the n0 plane), rb+8+.. for g>=4.
                // Shuffle both planes and select locally (a shfl source lane
                // contributes its evaluation of the operand).
                const uint32_t m = (g & 3u) * 2u;
                const uint32_t lo0 = __shfl_sync(0xffffffffu, n0, m * 4u + tq);
                const uint32_t hi0 = __shfl_sync(0xffffffffu, n0, (m + 1u) * 4u + tq);
                const uint32_t lo1 = __shfl_sync(0xffffffffu, n1, m * 4u + tq);
                const uint32_t hi1 = __shfl_sync(0xffffffffu, n1, (m + 1u) * 4u + tq);
                const uint32_t lo = (g < 4u) ? lo0 : lo1;
                const uint32_t hi = (g < 4u) ? hi0 : hi1;
                if (rb < ff) {
                    const size_t srow = (size_t)blk * PD_NV4M_BM + c;
                    fq[srow * (ff >> 1) + (rb >> 1) + g] =
                        (unsigned char)(lo | (hi << 4));
                    if (g == 0)
                        fs[srow * (ff >> 4) + (rb >> 4)] = (unsigned char)sbyte;
                }
            }
        }
    }
#else
    (void)data; (void)scale; (void)scale2; (void)sorted_row; (void)block_expert;
    (void)xq; (void)xs; (void)fq; (void)fs; (void)in_dim; (void)ff;
#endif
}

// ---- sm_100 sorted-tile MoE: WEIGHT-ONLY bf16 tensor cores ----------------
// The datacenter-Blackwell arm of the *_bs pair.
//
// CORRECTED -- READ this before REUSING the old CLAIM. This block
// used to say "B200 has no NATIVE PACKED-FP4 TENSOR CORE", citing a raw-PTX
// probe that got illegal-instruction from
// `tcgen05.mma kind::mxf4nvf4`, and citing engines falling back to Marlin.
// Both halves are false:
//   - FlashInfer ships a CuteDSL NVFP4 GEMM for this die
//     (Sm100BlockScaledPersistentDenseGemmKernel), and that is what a tuned
//     NVFP4 decode path actually runs.
//   - The instruction runs here. A probe builds CUTLASS 4.5's
//     `KernelTmaWarpSpecialized1SmNvf4Sm100` for sm_100a and measures it on
//     this die: gate|up (n=34816,k=5120,m=32) 26.83 us, down
//     (n=5120,k=17408) 22.67 us, L2-honest 4-clone rotation.
// So the old probe was wrong about the hardware, not the hardware about the
// probe -- the likely cause is raw-PTX operand/scale-descriptor layout, which
// CUTLASS gets right. The weight-only bf16 arm below stays (it is still the
// right arm for a die with no fp4 MMA, and it is what the MoE lane ships),
// but "B200 cannot do packed fp4" must not be quoted from here again.
//
// Class of the arm below: weight-only, exactly like Marlin's.
//
// Class: W4A16-in-bf16. Both operands are dequantized nvf4 -- weights from
// (data, scale, scale2[e]) and activations from the caller's own
// (xq, xs) nvf4 planes -- folded to bf16 and fed to ordinary m16n8k16
// bf16 mma with f32 accumulate. The e2m1 x e4m3 product is exactly
// representable in bf16 (e2m1 carries <=3 significant bits, e4m3 4, and bf16
// has 8), so the dequant itself is lossless; only the accumulation order
// differs from the sm_120a hardware path, which is why this arm is
// tolerance-gated against the dense reference rather than bit-exact against
// the *_bs pair.
//
// ABI is the *_bs ABI verbatim, deliberately: has_nvf4_moe_bs() then reports
// true on this die, nemotron's class_ok passes, and forward.rs's prefill
// takes its `moe_bs` arm with no call-site change. That arm is the whole
// point -- the false arm runs nvf4_moe_up_relu2/down_acc, which are
// DECODE-class scalar FFMA kernels, over every prompt token, and that is
// where nemotron's 465 ms c1 TTFT goes.
//
// Geometry mirrors the sm_120a twin so the epilogue is reusable UNCHANGED:
// 256 threads, 128 weight rows x PD_NV4M_BM(32) sorted tokens per CTA, warp
// grid i0=(warp>>1)*32 / joff=(warp&1)*8. The m16n8 f32 accumulator layout is
// identical for every mma.sync K-size and input format -- c0/c1 are row g,
// c2/c3 row g+8, cols 2*tq+qc -- so acc[2*jg + rg] indexes exactly what the
// block-scale kernel's acc[(j0>>3)+n] did.
#if PD_NV4_OK && PD_BF16MMA_OK
#define PD_NV4T_OK 1
#else
#define PD_NV4T_OK 0
#endif

// smem: ST buffers of (128 weight rows + 32 token cols) x KPAD bf16
#define PD_NV4T_SMEM(ST, KT) \
    ((ST) * (128u + PD_NV4M_BM) * ((KT) + 8u) * (uint32_t)sizeof(__nv_bfloat16))

#ifdef PD_BS_HOST
// True when this die must take the weight-only bf16 tile arm: the block-scale
// *_bs kernels are sm_120a SASS, and no other die implements packed-fp4 MMA.
static bool pd_nv4t_arm() {
    static const bool v = [] {
        return !pd_dev_bs_sass() && pd_env("PADDOCK_NO_NV4TILE") == nullptr;
    }();
    return v;
}
// K-tile for the bf16 tile arm. KT sets how many contiguous BYTES of a weight
// row one k-step reads: KT nvf4 elements = KT/2 bytes. At KT=64 that is 32 B,
// so a warp's 32 threads touch EIGHT different rows as eight separate sectors
// - a c8 decode attribution had the up kernel at 171 us for ~100 MB of
// expert weights, i.e. 7% of the card's bandwidth. Wider KT trades smem (and so
// CTAs/SM) for coalescing and for fewer k-steps. Elected by measurement, not
// assumption; PADDOCK_NV4T_KT overrides for the A/B.
static uint32_t pd_nv4t_kt() {
    static const uint32_t v = [] {
        const char* e = pd_env("PADDOCK_NV4T_KT");
        if (e) { const uint32_t k = (uint32_t)atoi(e); if (k == 64u || k == 128u) return k; }
        return 128u;
    }();
    return v;
}
#endif

#if PD_NV4T_OK
// 16 nvf4 elements (one scale block) -> 16 bf16, s folded. The nibble decode
// is pd_nvf4_gemv's prmt pair verbatim (T0/T1), so this arm and the scalar
// lane dequantize bit-identically.
__device__ __forceinline__ void pd_nv4t_deq16(uint64_t wq, float s,
                                              __nv_bfloat16* out) {
    constexpr uint32_t T0 = 0x3C383000u, T1 = 0x4C484440u;
    #pragma unroll
    for (uint32_t q = 0; q < 4u; ++q) {
        const uint32_t wb = (uint32_t)(wq >> (16u * q)) & 0xFFFFu;
        const uint32_t v = (wb & 0xFu) | ((wb & 0xF0u) << 4)
                         | ((wb & 0xF00u) << 8) | ((wb & 0xF000u) << 12);
        const uint32_t mag = v & 0x07070707u;
        const uint32_t tt = (mag | (mag >> 4)) & 0x00FF00FFu;
        const uint32_t e4 = __byte_perm(T0, T1, (tt | (tt >> 8)) & 0xFFFFu)
                          | ((v & 0x08080808u) << 4);
        const __nv_fp8_e4m3* eb = reinterpret_cast<const __nv_fp8_e4m3*>(&e4);
        #pragma unroll
        for (uint32_t e = 0; e < 4u; ++e)
            out[q * 4u + e] = __float2bfloat16((float)eb[e] * s);
    }
}
#endif

template <uint32_t ST, uint32_t KT>
__global__ void __launch_bounds__(256) pd_nv4t_moe_up_relu2_kernel(
    const uint8_t* __restrict__ data, const uint8_t* __restrict__ scale,
    const float* __restrict__ scale2, const uint32_t* __restrict__ sorted_row,
    const uint32_t* __restrict__ block_expert, const uint8_t* __restrict__ xq,
    const uint8_t* __restrict__ xs, uint8_t* __restrict__ fq,
    uint8_t* __restrict__ fs, uint32_t in_dim, uint32_t ff) {
#if PD_NV4T_OK
    constexpr uint32_t KPAD = KT + 8u;
    constexpr uint32_t GPR = KT / 16u;                 // scale blocks per row
    constexpr uint32_t NTH = 256u;
    constexpr uint32_t WROWS = 128u;                   // weight rows per CTA
    constexpr uint32_t AIT = (WROWS * GPR + NTH - 1u) / NTH;
    constexpr uint32_t BIT = (PD_NV4M_BM * GPR + NTH - 1u) / NTH;
    static_assert(KT % 16u == 0u, "KT k16-multiple");

    const uint32_t blk = blockIdx.x;
    const uint32_t e = block_expert[blk];
    if (e == PD_MOE_PAD) return;
    const uint32_t row_base = blockIdx.y * WROWS;

    extern __shared__ __align__(16) __nv_bfloat16 pd_nv4t_sh[];
    auto sh_w = reinterpret_cast<__nv_bfloat16(*)[WROWS * KPAD]>(pd_nv4t_sh);
    auto sh_y = reinterpret_cast<__nv_bfloat16(*)[PD_NV4M_BM * KPAD]>(
        pd_nv4t_sh + ST * WROWS * KPAD);
    __shared__ uint32_t tok[PD_NV4M_BM];

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, tq = lane & 3u;
    const uint32_t i0 = (warp >> 1) * 32u;
    const uint32_t joff = (warp & 1u) * 8u;
    const uint32_t n_kh = in_dim >> 1;                 // packed bytes per row
    const uint32_t n_k16 = in_dim >> 4;                // scale bytes per row
    const size_t wrow0 = (size_t)e * ff + row_base;
    const __nv_bfloat16 bzero = __float2bfloat16(0.0f);

    __shared__ uint32_t livemask;
    if (tid < PD_NV4M_BM) tok[tid] = sorted_row[(size_t)blk * PD_NV4M_BM + tid];
    __syncthreads();
    // OCCUPANCY GUARD. moe_align gives almost every expert its
    // own block, so at DECODE shapes the sorted plane is nearly empty: at
    // r=8, top-k 6 over 128 experts the c8 trace showed grid=(260,15) = 3900
    // CTAs carrying ~56 live rows across 260x32 = 8320 slots, i.e. 0.67%.
    // Blocks whose 32 slots are all pad were still running a full
    // 128x32xin_dim tile and writing zeros. Skipping them is safe because the
    // down half skips the same blocks (same tok[]), so nobody reads the fq/fs
    // this leaves unwritten.
    if (tid == 0) {
        uint32_t m = 0;
        for (uint32_t i = 0; i < PD_NV4M_BM; ++i)
            if (tok[i] != PD_MOE_PAD) m |= 1u << i;
        livemask = m;
    }
    __syncthreads();
    if (livemask == 0u) return;
    // per-8-column-group liveness: a warp's two n8 fragments. A group that is
    // all pad contributes only zeros, which the epilogue writes anyway from a
    // zero accumulator - so its B load and mma are pure waste.
    const bool jg_live[2] = {
        ((livemask >> (0u * 16u + joff)) & 0xFFu) != 0u,
        ((livemask >> (1u * 16u + joff)) & 0xFFu) != 0u};

    // split load/store so every long-latency global read of stage k+ST-1 is
    // in flight across compute(k) - the gemm_tc staging pattern
    uint64_t a_wq[AIT]; float a_s[AIT]; bool a_ok[AIT];
    uint64_t b_wq[BIT]; float b_s[BIT]; bool b_ok[BIT];
    auto stage_load = [&](uint32_t k0) {
        #pragma unroll
        for (uint32_t it = 0; it < AIT; ++it) {
            const uint32_t i = tid + it * NTH;
            const uint32_t row = i / GPR, g16 = (i % GPR) * 16u;
            const uint32_t gk = k0 + g16;
            const bool ok = i < WROWS * GPR && (row_base + row) < ff && gk < in_dim;
            a_ok[it] = ok;
            a_wq[it] = ok ? *reinterpret_cast<const uint64_t*>(
                                data + (wrow0 + row) * (size_t)n_kh + (gk >> 1))
                          : 0ull;
            a_s[it] = ok ? (float)reinterpret_cast<const __nv_fp8_e4m3&>(
                               scale[(wrow0 + row) * (size_t)n_k16 + (gk >> 4)])
                         : 0.0f;
        }
        #pragma unroll
        for (uint32_t it = 0; it < BIT; ++it) {
            const uint32_t i = tid + it * NTH;
            const uint32_t col = i / GPR, g16 = (i % GPR) * 16u;
            const uint32_t gk = k0 + g16;
            const uint32_t r = (i < PD_NV4M_BM * GPR) ? tok[col] : PD_MOE_PAD;
            const bool ok = i < PD_NV4M_BM * GPR && r != PD_MOE_PAD && gk < in_dim;
            b_ok[it] = ok;
            b_wq[it] = ok ? *reinterpret_cast<const uint64_t*>(
                                xq + (size_t)r * n_kh + (gk >> 1))
                          : 0ull;
            b_s[it] = ok ? (float)reinterpret_cast<const __nv_fp8_e4m3&>(
                               xs[(size_t)r * n_k16 + (gk >> 4)])
                         : 0.0f;
        }
    };
    auto stage_store = [&](uint32_t buf) {
        #pragma unroll
        for (uint32_t it = 0; it < AIT; ++it) {
            const uint32_t i = tid + it * NTH;
            if (i >= WROWS * GPR) continue;
            const uint32_t row = i / GPR, g16 = (i % GPR) * 16u;
            __nv_bfloat16 tmp[16];
            if (a_ok[it]) pd_nv4t_deq16(a_wq[it], a_s[it], tmp);
            else {
                #pragma unroll
                for (uint32_t z = 0; z < 16u; ++z) tmp[z] = bzero;
            }
            __nv_bfloat16* dst = &sh_w[buf][row * KPAD + g16];
            reinterpret_cast<int4*>(dst)[0] = reinterpret_cast<const int4*>(tmp)[0];
            reinterpret_cast<int4*>(dst)[1] = reinterpret_cast<const int4*>(tmp)[1];
        }
        #pragma unroll
        for (uint32_t it = 0; it < BIT; ++it) {
            const uint32_t i = tid + it * NTH;
            if (i >= PD_NV4M_BM * GPR) continue;
            const uint32_t col = i / GPR, g16 = (i % GPR) * 16u;
            __nv_bfloat16 tmp[16];
            if (b_ok[it]) pd_nv4t_deq16(b_wq[it], b_s[it], tmp);
            else {
                #pragma unroll
                for (uint32_t z = 0; z < 16u; ++z) tmp[z] = bzero;
            }
            __nv_bfloat16* dst = &sh_y[buf][col * KPAD + g16];
            reinterpret_cast<int4*>(dst)[0] = reinterpret_cast<const int4*>(tmp)[0];
            reinterpret_cast<int4*>(dst)[1] = reinterpret_cast<const int4*>(tmp)[1];
        }
    };

    const uint32_t l7 = lane & 7u;
    const uint32_t a_roff = ((lane & 8u) ? 8u : 0u) + l7;
    const uint32_t a_kof = (lane & 16u) ? 8u : 0u;
    const uint32_t b_kof = (lane & 8u) ? 8u : 0u;
    // acc[2*jg + rg]: jg = token 16-block, rg = row half. This is exactly the
    // sm_120a kernel's acc[(j0>>3)+n], which is why its epilogue drops in.
    float acc[4][4] = {};
    auto compute = [&](uint32_t buf) {
        #pragma unroll
        for (uint32_t sk = 0; sk < KT / 16u; ++sk) {
            const uint32_t ko = sk * 16u;
            uint32_t a[2][4];
            #pragma unroll
            for (uint32_t rg = 0; rg < 2u; ++rg)
                pd_bf16m_ldm_x4(&sh_w[buf][(i0 + rg * 16u + a_roff) * KPAD + ko + a_kof],
                                a[rg][0], a[rg][1], a[rg][2], a[rg][3]);
            #pragma unroll
            for (uint32_t jg = 0; jg < 2u; ++jg) {
                if (!jg_live[jg]) continue;      // all-pad n8 group
                uint32_t b[2];
                pd_bf16m_ldm_x2(&sh_y[buf][(joff + jg * 16u + l7) * KPAD + ko + b_kof],
                                b[0], b[1]);
                #pragma unroll
                for (uint32_t rg = 0; rg < 2u; ++rg)
                    pd_bf16m_mma(acc[2u * jg + rg], a[rg], b);
            }
        }
    };

    #pragma unroll
    for (uint32_t s = 0; s < ST - 1u; ++s) {
        const uint32_t k0 = s * KT;
        if (k0 < in_dim) { stage_load(k0); stage_store(s); }
    }
    uint32_t p = 0;
    for (uint32_t k0 = 0; k0 < in_dim; k0 += KT) {
        const uint32_t pre = k0 + (ST - 1u) * KT;
        __syncthreads();
        if (pre < in_dim) stage_load(pre);
        compute(p);
        if (pre < in_dim) stage_store((p + ST - 1u) % ST);
        p = (p + 1u) % ST;
    }

    // epilogue: identical to the sm_120a *_bs kernel's (same fragment layout,
    // same shuffle scheme, same scale pick) so both arms emit byte-compatible
    // fq/fs planes for the down half.
    const float s2 = scale2[e];
    const uint32_t tmask = 0x11111111u << tq;
    #pragma unroll
    for (uint32_t jg = 0; jg < 2u; ++jg) {
        const uint32_t j0 = jg * 16u;
        #pragma unroll
        for (uint32_t n = 0; n < 2u; ++n) {
            const uint32_t rb = row_base + i0 + n * 16u;
            #pragma unroll
            for (uint32_t qc = 0; qc < 2u; ++qc) {
                const uint32_t c = j0 + joff + 2u * tq + qc;
                const bool pad = tok[c] == PD_MOE_PAD;
                const float a0 = acc[2u * jg + n][qc] * s2;
                const float a1 = acc[2u * jg + n][qc + 2u] * s2;
                const float r0v = fmaxf(a0, 0.0f);
                const float r1v = fmaxf(a1, 0.0f);
                const float v0 = pad ? 0.0f : r0v * r0v;
                const float v1 = pad ? 0.0f : r1v * r1v;
                float a = fmaxf(v0, v1);
                a = fmaxf(a, __shfl_xor_sync(tmask, a, 4));
                a = fmaxf(a, __shfl_xor_sync(tmask, a, 8));
                a = fmaxf(a, __shfl_xor_sync(tmask, a, 16));
                float inv;
                const unsigned sbyte = pd_nvf4_scale(a, &inv);
                const uint32_t n0 = pd_e2m1_rn(v0 * inv);
                const uint32_t n1 = pd_e2m1_rn(v1 * inv);
                const uint32_t m = (g & 3u) * 2u;
                const uint32_t lo0 = __shfl_sync(0xffffffffu, n0, m * 4u + tq);
                const uint32_t hi0 = __shfl_sync(0xffffffffu, n0, (m + 1u) * 4u + tq);
                const uint32_t lo1 = __shfl_sync(0xffffffffu, n1, m * 4u + tq);
                const uint32_t hi1 = __shfl_sync(0xffffffffu, n1, (m + 1u) * 4u + tq);
                const uint32_t lo = (g < 4u) ? lo0 : lo1;
                const uint32_t hi = (g < 4u) ? hi0 : hi1;
                if (rb < ff) {
                    const size_t srw = (size_t)blk * PD_NV4M_BM + c;
                    fq[srw * (ff >> 1) + (rb >> 1) + g] =
                        (unsigned char)(lo | (hi << 4));
                    if (g == 0)
                        fs[srw * (ff >> 4) + (rb >> 4)] = (unsigned char)sbyte;
                }
            }
        }
    }
#else
    (void)data; (void)scale; (void)scale2; (void)sorted_row; (void)block_expert;
    (void)xq; (void)xs; (void)fq; (void)fs; (void)in_dim; (void)ff;
#endif
}

PD_EXPORT
int pd_nvf4_moe_up_relu2_bs(const void* data, const void* scale,
                            const void* scale2, const void* sorted_row,
                            const void* block_expert, const void* xq,
                            const void* xs, void* fq, void* fs, uint32_t in_dim,
                            uint32_t ff, uint32_t nb, void* stream) {
#ifndef PD_BS_HOST
    (void)data; (void)scale; (void)scale2; (void)sorted_row; (void)block_expert;
    (void)xq; (void)xs; (void)fq; (void)fs; (void)in_dim; (void)ff; (void)nb;
    (void)stream;
    return cudaErrorNotSupported;
#else
    if (ff == 0 || nb == 0) return 0;
    if ((in_dim & 31u) != 0 || (ff & 15u) != 0) return cudaErrorInvalidValue;
    dim3 grid(nb, (ff + 127u) / 128u);
    // ARM ELECTION. The block-scale kernels below are sm_120a SASS; any other
    // die (B200 included) has no packed-fp4 tensor core at all - kind::mxf4
    // and kind::mxf4nvf4 raise illegal-instruction there, measured with
    // with a raw-PTX probe. Those dies take the weight-only bf16
    // tile arm instead - the same class other engines fall back to (Marlin).
    // PADDOCK_NO_NV4TILE=1 pins the old behaviour for an A/B.
    if (pd_nv4t_arm()) {
        #define PD_NV4T_UP(KTV)                                                    \
            pd_nv4t_moe_up_relu2_kernel<2u, KTV><<<grid, 256,                      \
                                                   PD_NV4T_SMEM(2u, KTV),          \
                                                   (cudaStream_t)stream>>>(        \
                (const uint8_t*)data, (const uint8_t*)scale, (const float*)scale2, \
                (const uint32_t*)sorted_row, (const uint32_t*)block_expert,        \
                (const uint8_t*)xq, (const uint8_t*)xs, (uint8_t*)fq,              \
                (uint8_t*)fs, in_dim, ff)
        if (pd_nv4t_kt() == 128u && (in_dim & 127u) == 0u) {
            static bool at = false;
            if (!at) { cudaFuncSetAttribute(
                (const void*)pd_nv4t_moe_up_relu2_kernel<2u, 128u>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, PD_NV4T_SMEM(2u, 128u));
                at = true; }
            PD_NV4T_UP(128u);
        } else PD_NV4T_UP(64u);
        #undef PD_NV4T_UP
        return pd_launch_status();
    }
    if (pd_nv4m_kb() == 8u)
        pd_nvf4_moe_up_relu2_bs_kernel<8u, false><<<grid, 256, PD_NV4M_SMEM(8u),
                                                    (cudaStream_t)stream>>>(
            (const uint8_t*)data, (const uint8_t*)scale, (const float*)scale2,
            (const uint32_t*)sorted_row, (const uint32_t*)block_expert,
            (const uint8_t*)xq, (const uint8_t*)xs, (uint8_t*)fq, (uint8_t*)fs,
            in_dim, ff);
    else
        pd_nvf4_moe_up_relu2_bs_kernel<4u, false><<<grid, 256, PD_NV4M_SMEM(4u),
                                                    (cudaStream_t)stream>>>(
            (const uint8_t*)data, (const uint8_t*)scale, (const float*)scale2,
            (const uint32_t*)sorted_row, (const uint32_t*)block_expert,
            (const uint8_t*)xq, (const uint8_t*)xs, (uint8_t*)fq, (uint8_t*)fs,
            in_dim, ff);
    return pd_launch_status();
#endif
}

// Sorted-tile expert down -> weighted per-(token, slot) f32 partials. B is
// the up kernel's fq/fs output (sorted-position indexed, no gather); the
// epilogue scatters acc * topk_w[tok*kw + slt] * scale2[e] to
// part[(tok*np + slt + slot_off) * embd + r] - pd_moe_slot_combine folds the
// np partials per token in fixed slot order (deterministic, the down_mmq
// contract; no atomics per the numerics doctrine). topk_w NULL means 1.0
// (the shared-expert pass: its own trivial align, slot_off past the routed
// picks). PAD columns are skipped at scatter, so their garbage accs never
// land.
template <uint32_t KB, bool SFOLD>
__global__ void __launch_bounds__(256, 2) pd_nvf4_moe_down_bs_kernel(  // SFOLD: see the up kernel
    const uint8_t* __restrict__ data, const uint8_t* __restrict__ scale,
    const float* __restrict__ scale2, const uint32_t* __restrict__ sorted_row,
    const uint32_t* __restrict__ sorted_slot,
    const uint32_t* __restrict__ block_expert, const float* __restrict__ topk_w,
    const uint8_t* __restrict__ fq, const uint8_t* __restrict__ fs,
    float* __restrict__ part, uint32_t ff, uint32_t embd, uint32_t kw,
    uint32_t np, uint32_t slot_off) {
#if PD_BS_OK
    constexpr uint32_t WROW = PD_NV4M_WROW(KB);
    constexpr uint32_t YROW = PD_NV4M_YROW(KB);
    const uint32_t blk = blockIdx.x;
    const uint32_t e = block_expert[blk];
    if (e == PD_MOE_PAD) return;
    const uint32_t row_base = blockIdx.y * 128u;

    extern __shared__ unsigned char pd_bs_sh[];
    unsigned char* wb0 = pd_bs_sh;
    unsigned char* wb1 = wb0 + 128u * WROW;
    unsigned char* yb0 = wb1 + 128u * WROW;
    unsigned char* yb1 = yb0 + PD_NV4M_BM * YROW;

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, tq = lane & 3u;
    const uint32_t i0 = (warp >> 1) * 32u;
    const uint32_t joff = (warp & 1u) * 8u;
    const uint32_t n_kb = ff >> 5;
    const uint32_t n_k16 = ff >> 4;
    const uint32_t nk = (ff + KB * 32u - 1u) / (KB * 32u);
    const size_t wrow0 = (size_t)e * embd + row_base;
    const bool sfold = SFOLD && (n_k16 & 3u) == 0u;

    float acc[4][4] = {};

    #define PD_NV4M_ISSUE_W(dst, kt)                                                  \
        for (uint32_t u = tid; u < 128u * KB; u += 256u) {                            \
            const uint32_t row = u / KB, seg = u % KB;                                \
            const bool ok = (row_base + row) < embd && (kt) * KB + seg < n_kb;        \
            pd_cp_async16((int*)((dst) + row * WROW + seg * 16u),                     \
                          data + (wrow0 + row) * (size_t)(ff >> 1) +                  \
                              (kt) * (KB * 16u) + seg * 16u,                          \
                          ok);                                                        \
        }
    #define PD_NV4M_ISSUE_Y(dst, kt)                                                  \
        for (uint32_t u = tid; u < PD_NV4M_BM * KB; u += 256u) {                      \
            const uint32_t col = u / KB, seg = u % KB;                                \
            const bool ok = (kt) * KB + seg < n_kb;                                   \
            pd_cp_async16((int*)((dst) + col * YROW + 16u + seg * 16u),               \
                          fq + ((size_t)blk * PD_NV4M_BM + col) * (size_t)(ff >> 1) + \
                              (kt) * (KB * 16u) + seg * 16u,                          \
                          ok);                                                        \
        }
    #define PD_NV4M_ISSUE_WS(dst, kt)                                                 \
        for (uint32_t u = tid; u < 128u * (KB / 2u); u += 256u) {                     \
            const uint32_t row = u / (KB / 2u), q = u % (KB / 2u);                    \
            const bool ok = (row_base + row) < embd &&                                \
                            (kt) * (KB * 2u) + q * 4u + 4u <= n_k16;                  \
            pd_cpa4p((dst) + row * WROW + KB * 16u + q * 4u,                          \
                     scale + (wrow0 + row) * (size_t)n_k16 + (kt) * (KB * 2u) +       \
                         q * 4u,                                                      \
                     ok);                                                             \
        }
    #define PD_NV4M_ISSUE_YS(dst, kt)                                                 \
        for (uint32_t u = tid; u < PD_NV4M_BM * (KB / 2u); u += 256u) {               \
            const uint32_t row = u / (KB / 2u), q = u % (KB / 2u);                    \
            const bool ok = (kt) * (KB * 2u) + q * 4u + 4u <= n_k16;                  \
            pd_cpa4p((dst) + row * YROW + q * 4u,                                     \
                     fs + ((size_t)blk * PD_NV4M_BM + row) * n_k16 +                  \
                         (kt) * (KB * 2u) + q * 4u,                                   \
                     ok);                                                             \
        }

    PD_NV4M_ISSUE_W(wb0, 0u)
    PD_NV4M_ISSUE_Y(yb0, 0u)
    if (sfold) { PD_NV4M_ISSUE_WS(wb0, 0u) PD_NV4M_ISSUE_YS(yb0, 0u) }
    asm volatile("cp.async.commit_group;");
    for (uint32_t kt = 0; kt < nk; ++kt) {
        unsigned char* tw = (kt & 1u) ? wb1 : wb0;
        unsigned char* ty = (kt & 1u) ? yb1 : yb0;
        if (kt + 1u < nk) {
            PD_NV4M_ISSUE_W((kt & 1u) ? wb0 : wb1, kt + 1u)
            PD_NV4M_ISSUE_Y((kt & 1u) ? yb0 : yb1, kt + 1u)
            if (sfold) {
                PD_NV4M_ISSUE_WS((kt & 1u) ? wb0 : wb1, kt + 1u)
                PD_NV4M_ISSUE_YS((kt & 1u) ? yb0 : yb1, kt + 1u)
            }
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        if (!sfold) {   // e4m3 scale planes: W KB*2 bytes/row, Y 32 rows x KB*2
            for (uint32_t u = tid; u < 128u * KB * 2u; u += 256u) {
                const uint32_t row = u / (KB * 2u), kb16 = u % (KB * 2u);
                const bool wok = (row_base + row) < embd && kt * (KB * 2u) + kb16 < n_k16;
                tw[row * WROW + KB * 16u + kb16] =
                    wok ? scale[(wrow0 + row) * (size_t)n_k16 + kt * (KB * 2u) + kb16]
                        : 0u;
            }
            for (uint32_t u = tid; u < PD_NV4M_BM * KB * 2u; u += 256u) {
                const uint32_t row = u / (KB * 2u), kb16 = u % (KB * 2u);
                const bool yok = kt * (KB * 2u) + kb16 < n_k16;
                ty[row * YROW + kb16] =
                    yok ? fs[((size_t)blk * PD_NV4M_BM + row) * n_k16 +
                             kt * (KB * 2u) + kb16]
                        : 0u;
            }
        }
        __syncthreads();

        uint32_t am[2][KB / 2u][4], sa[2][KB / 2u];
        #pragma unroll
        for (uint32_t n = 0; n < 2u; ++n) {
            const uint32_t r0 = i0 + n * 16u + g;
            const uint32_t rs = (tq & 1u) ? r0 + 8u : r0;
            #pragma unroll
            for (uint32_t k64 = 0; k64 < KB / 2u; ++k64) {
                pd_ldm_x4(am[n][k64],
                          tw + (i0 + n * 16u + ((lane >> 3) & 1u) * 8u + (lane & 7u)) *
                                  WROW +
                              k64 * 32u + (lane >> 4) * 16u);
                sa[n][k64] =
                    *(const uint32_t*)(tw + rs * WROW + KB * 16u + k64 * 4u);
            }
        }
        #pragma unroll
        for (uint32_t j0 = 0; j0 < PD_NV4M_BM; j0 += 16u) {
            uint32_t bm[2u * (KB / 2u)];
            #pragma unroll
            for (uint32_t q = 0; q < KB / 4u; ++q)
                pd_ldm_x4(bm + q * 4u, ty + (j0 + joff + (lane & 7u)) * YROW + 16u +
                                           q * 64u + (lane >> 3) * 16u);
            const unsigned char* ysr = ty + (j0 + joff + g) * YROW;
            #pragma unroll
            for (uint32_t k64 = 0; k64 < KB / 2u; ++k64) {
                const uint32_t sb = *(const uint32_t*)(ysr + k64 * 4u);
                #pragma unroll
                for (uint32_t n = 0; n < 2u; ++n)
                    pd_nv4_mma(acc[(j0 >> 3) + n], am[n][k64][0], am[n][k64][1],
                               am[n][k64][2], am[n][k64][3], bm[k64 * 2u],
                               bm[k64 * 2u + 1u], sa[n][k64], sb);
            }
        }
        __syncthreads();
    }
    #undef PD_NV4M_ISSUE_W
    #undef PD_NV4M_ISSUE_Y
    #undef PD_NV4M_ISSUE_WS
    #undef PD_NV4M_ISSUE_YS

    // epilogue: weighted scatter to the per-(token, slot) partial rows -
    // the dense kernel's per-column writes with the sorted-layout target.
    const float s2 = scale2[e];
    #pragma unroll
    for (uint32_t j0 = 0; j0 < PD_NV4M_BM; j0 += 16u) {
        #pragma unroll
        for (uint32_t qc = 0; qc < 2u; ++qc) {
            const uint32_t c = j0 + joff + 2u * tq + qc;
            const uint32_t t = sorted_row[(size_t)blk * PD_NV4M_BM + c];
            if (t == PD_MOE_PAD) continue;
            const uint32_t slt = sorted_slot[(size_t)blk * PD_NV4M_BM + c];
            const float w =
                (topk_w ? topk_w[(size_t)t * kw + slt] : 1.0f) * s2;
            float* prow = part + ((size_t)t * np + slt + slot_off) * embd;
            #pragma unroll
            for (uint32_t n = 0; n < 2u; ++n) {
                const uint32_t r0 = row_base + i0 + n * 16u + g;
                const uint32_t r8 = r0 + 8u;
                if (r0 < embd) prow[r0] = acc[(j0 >> 3) + n][qc] * w;
                if (r8 < embd) prow[r8] = acc[(j0 >> 3) + n][qc + 2u] * w;
            }
        }
    }
#else
    (void)data; (void)scale; (void)scale2; (void)sorted_row; (void)sorted_slot;
    (void)block_expert; (void)topk_w; (void)fq; (void)fs; (void)part; (void)ff;
    (void)embd; (void)kw; (void)np; (void)slot_off;
#endif
}

// sm_100 weight-only twin of the down half - see the up kernel's header for
// why this die gets a bf16 arm at all. Same tile geometry and same acc[4][4]
// fragment layout; the differences are only which planes feed the operands:
//   A = down weights for expert e, `embd` rows, K = ff
//   B = the up half's fq/fs, indexed by SORTED POSITION with no gather (that
//       is the *_bs contract: fq is [nb*PD_NV4M_BM, ff/2])
// and the epilogue, which is the block-scale kernel's weighted scatter
// verbatim rather than a requantize.
template <uint32_t ST, uint32_t KT>
__global__ void __launch_bounds__(256) pd_nv4t_moe_down_kernel(
    const uint8_t* __restrict__ data, const uint8_t* __restrict__ scale,
    const float* __restrict__ scale2, const uint32_t* __restrict__ sorted_row,
    const uint32_t* __restrict__ sorted_slot,
    const uint32_t* __restrict__ block_expert, const float* __restrict__ topk_w,
    const uint8_t* __restrict__ fq, const uint8_t* __restrict__ fs,
    float* __restrict__ part, uint32_t ff, uint32_t embd, uint32_t kw,
    uint32_t np, uint32_t slot_off) {
#if PD_NV4T_OK
    constexpr uint32_t KPAD = KT + 8u;
    constexpr uint32_t GPR = KT / 16u;
    constexpr uint32_t NTH = 256u;
    constexpr uint32_t WROWS = 128u;
    constexpr uint32_t AIT = (WROWS * GPR + NTH - 1u) / NTH;
    constexpr uint32_t BIT = (PD_NV4M_BM * GPR + NTH - 1u) / NTH;

    const uint32_t blk = blockIdx.x;
    const uint32_t e = block_expert[blk];
    if (e == PD_MOE_PAD) return;
    const uint32_t row_base = blockIdx.y * WROWS;

    extern __shared__ __align__(16) __nv_bfloat16 pd_nv4t_sh[];
    auto sh_w = reinterpret_cast<__nv_bfloat16(*)[WROWS * KPAD]>(pd_nv4t_sh);
    auto sh_y = reinterpret_cast<__nv_bfloat16(*)[PD_NV4M_BM * KPAD]>(
        pd_nv4t_sh + ST * WROWS * KPAD);

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, tq = lane & 3u;
    const uint32_t i0 = (warp >> 1) * 32u;
    const uint32_t joff = (warp & 1u) * 8u;
    const uint32_t n_kh = ff >> 1;
    const uint32_t n_k16 = ff >> 4;
    const size_t wrow0 = (size_t)e * embd + row_base;
    const __nv_bfloat16 bzero = __float2bfloat16(0.0f);

    // same occupancy guard as the up half - see its note. This half writes
    // only for non-pad tokens, so an all-pad block writes nothing at all and
    // skipping it is a pure saving.
    __shared__ uint32_t tokd[PD_NV4M_BM];
    __shared__ uint32_t livemask;
    if (tid < PD_NV4M_BM) tokd[tid] = sorted_row[(size_t)blk * PD_NV4M_BM + tid];
    __syncthreads();
    if (tid == 0) {
        uint32_t m = 0;
        for (uint32_t i = 0; i < PD_NV4M_BM; ++i)
            if (tokd[i] != PD_MOE_PAD) m |= 1u << i;
        livemask = m;
    }
    __syncthreads();
    if (livemask == 0u) return;
    const bool jg_live[2] = {
        ((livemask >> (0u * 16u + joff)) & 0xFFu) != 0u,
        ((livemask >> (1u * 16u + joff)) & 0xFFu) != 0u};

    uint64_t a_wq[AIT]; float a_s[AIT]; bool a_ok[AIT];
    uint64_t b_wq[BIT]; float b_s[BIT]; bool b_ok[BIT];
    auto stage_load = [&](uint32_t k0) {
        #pragma unroll
        for (uint32_t it = 0; it < AIT; ++it) {
            const uint32_t i = tid + it * NTH;
            const uint32_t row = i / GPR, g16 = (i % GPR) * 16u;
            const uint32_t gk = k0 + g16;
            const bool ok = i < WROWS * GPR && (row_base + row) < embd && gk < ff;
            a_ok[it] = ok;
            a_wq[it] = ok ? *reinterpret_cast<const uint64_t*>(
                                data + (wrow0 + row) * (size_t)n_kh + (gk >> 1))
                          : 0ull;
            a_s[it] = ok ? (float)reinterpret_cast<const __nv_fp8_e4m3&>(
                               scale[(wrow0 + row) * (size_t)n_k16 + (gk >> 4)])
                         : 0.0f;
        }
        #pragma unroll
        for (uint32_t it = 0; it < BIT; ++it) {
            const uint32_t i = tid + it * NTH;
            const uint32_t col = i / GPR, g16 = (i % GPR) * 16u;
            const uint32_t gk = k0 + g16;
            const bool ok = i < PD_NV4M_BM * GPR && gk < ff;
            const size_t srw = (size_t)blk * PD_NV4M_BM + col;
            b_ok[it] = ok;
            b_wq[it] = ok ? *reinterpret_cast<const uint64_t*>(
                                fq + srw * n_kh + (gk >> 1))
                          : 0ull;
            b_s[it] = ok ? (float)reinterpret_cast<const __nv_fp8_e4m3&>(
                               fs[srw * n_k16 + (gk >> 4)])
                         : 0.0f;
        }
    };
    auto stage_store = [&](uint32_t buf) {
        #pragma unroll
        for (uint32_t it = 0; it < AIT; ++it) {
            const uint32_t i = tid + it * NTH;
            if (i >= WROWS * GPR) continue;
            const uint32_t row = i / GPR, g16 = (i % GPR) * 16u;
            __nv_bfloat16 tmp[16];
            if (a_ok[it]) pd_nv4t_deq16(a_wq[it], a_s[it], tmp);
            else {
                #pragma unroll
                for (uint32_t z = 0; z < 16u; ++z) tmp[z] = bzero;
            }
            __nv_bfloat16* dst = &sh_w[buf][row * KPAD + g16];
            reinterpret_cast<int4*>(dst)[0] = reinterpret_cast<const int4*>(tmp)[0];
            reinterpret_cast<int4*>(dst)[1] = reinterpret_cast<const int4*>(tmp)[1];
        }
        #pragma unroll
        for (uint32_t it = 0; it < BIT; ++it) {
            const uint32_t i = tid + it * NTH;
            if (i >= PD_NV4M_BM * GPR) continue;
            const uint32_t col = i / GPR, g16 = (i % GPR) * 16u;
            __nv_bfloat16 tmp[16];
            if (b_ok[it]) pd_nv4t_deq16(b_wq[it], b_s[it], tmp);
            else {
                #pragma unroll
                for (uint32_t z = 0; z < 16u; ++z) tmp[z] = bzero;
            }
            __nv_bfloat16* dst = &sh_y[buf][col * KPAD + g16];
            reinterpret_cast<int4*>(dst)[0] = reinterpret_cast<const int4*>(tmp)[0];
            reinterpret_cast<int4*>(dst)[1] = reinterpret_cast<const int4*>(tmp)[1];
        }
    };

    const uint32_t l7 = lane & 7u;
    const uint32_t a_roff = ((lane & 8u) ? 8u : 0u) + l7;
    const uint32_t a_kof = (lane & 16u) ? 8u : 0u;
    const uint32_t b_kof = (lane & 8u) ? 8u : 0u;
    float acc[4][4] = {};
    auto compute = [&](uint32_t buf) {
        #pragma unroll
        for (uint32_t sk = 0; sk < KT / 16u; ++sk) {
            const uint32_t ko = sk * 16u;
            uint32_t a[2][4];
            #pragma unroll
            for (uint32_t rg = 0; rg < 2u; ++rg)
                pd_bf16m_ldm_x4(&sh_w[buf][(i0 + rg * 16u + a_roff) * KPAD + ko + a_kof],
                                a[rg][0], a[rg][1], a[rg][2], a[rg][3]);
            #pragma unroll
            for (uint32_t jg = 0; jg < 2u; ++jg) {
                if (!jg_live[jg]) continue;      // all-pad n8 group
                uint32_t b[2];
                pd_bf16m_ldm_x2(&sh_y[buf][(joff + jg * 16u + l7) * KPAD + ko + b_kof],
                                b[0], b[1]);
                #pragma unroll
                for (uint32_t rg = 0; rg < 2u; ++rg)
                    pd_bf16m_mma(acc[2u * jg + rg], a[rg], b);
            }
        }
    };

    #pragma unroll
    for (uint32_t s = 0; s < ST - 1u; ++s) {
        const uint32_t k0 = s * KT;
        if (k0 < ff) { stage_load(k0); stage_store(s); }
    }
    uint32_t p = 0;
    for (uint32_t k0 = 0; k0 < ff; k0 += KT) {
        const uint32_t pre = k0 + (ST - 1u) * KT;
        __syncthreads();
        if (pre < ff) stage_load(pre);
        compute(p);
        if (pre < ff) stage_store((p + ST - 1u) % ST);
        p = (p + 1u) % ST;
    }

    // epilogue: the block-scale kernel's weighted scatter, verbatim
    const float s2 = scale2[e];
    #pragma unroll
    for (uint32_t jg = 0; jg < 2u; ++jg) {
        const uint32_t j0 = jg * 16u;
        #pragma unroll
        for (uint32_t qc = 0; qc < 2u; ++qc) {
            const uint32_t c = j0 + joff + 2u * tq + qc;
            const uint32_t t = sorted_row[(size_t)blk * PD_NV4M_BM + c];
            if (t == PD_MOE_PAD) continue;
            const uint32_t slt = sorted_slot[(size_t)blk * PD_NV4M_BM + c];
            const float w = (topk_w ? topk_w[(size_t)t * kw + slt] : 1.0f) * s2;
            float* prow = part + ((size_t)t * np + slt + slot_off) * embd;
            #pragma unroll
            for (uint32_t n = 0; n < 2u; ++n) {
                const uint32_t r0 = row_base + i0 + n * 16u + g;
                const uint32_t r8 = r0 + 8u;
                if (r0 < embd) prow[r0] = acc[2u * jg + n][qc] * w;
                if (r8 < embd) prow[r8] = acc[2u * jg + n][qc + 2u] * w;
            }
        }
    }
#else
    (void)data; (void)scale; (void)scale2; (void)sorted_row; (void)sorted_slot;
    (void)block_expert; (void)topk_w; (void)fq; (void)fs; (void)part; (void)ff;
    (void)embd; (void)kw; (void)np; (void)slot_off;
#endif
}

PD_EXPORT
int pd_nvf4_moe_down_bs(const void* data, const void* scale, const void* scale2,
                        const void* sorted_row, const void* sorted_slot,
                        const void* block_expert, const void* topk_w,
                        const void* fq, const void* fs, void* part, uint32_t ff,
                        uint32_t embd, uint32_t kw, uint32_t np,
                        uint32_t slot_off, uint32_t nb, void* stream) {
#ifndef PD_BS_HOST
    (void)data; (void)scale; (void)scale2; (void)sorted_row; (void)sorted_slot;
    (void)block_expert; (void)topk_w; (void)fq; (void)fs; (void)part; (void)ff;
    (void)embd; (void)kw; (void)np; (void)slot_off; (void)nb; (void)stream;
    return cudaErrorNotSupported;
#else
    if (embd == 0 || nb == 0) return 0;
    if ((ff & 31u) != 0 || np == 0) return cudaErrorInvalidValue;
    dim3 grid(nb, (embd + 127u) / 128u);
    if (pd_nv4t_arm()) {   // see the up half's election note
        #define PD_NV4T_DN(KTV)                                                    \
            pd_nv4t_moe_down_kernel<2u, KTV><<<grid, 256,                          \
                                               PD_NV4T_SMEM(2u, KTV),              \
                                               (cudaStream_t)stream>>>(            \
                (const uint8_t*)data, (const uint8_t*)scale, (const float*)scale2, \
                (const uint32_t*)sorted_row, (const uint32_t*)sorted_slot,         \
                (const uint32_t*)block_expert, (const float*)topk_w,               \
                (const uint8_t*)fq, (const uint8_t*)fs, (float*)part, ff, embd,    \
                kw, np, slot_off)
        if (pd_nv4t_kt() == 128u && (ff & 127u) == 0u) {
            static bool at = false;
            if (!at) { cudaFuncSetAttribute(
                (const void*)pd_nv4t_moe_down_kernel<2u, 128u>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, PD_NV4T_SMEM(2u, 128u));
                at = true; }
            PD_NV4T_DN(128u);
        } else PD_NV4T_DN(64u);
        #undef PD_NV4T_DN
        return pd_launch_status();
    }
    if (pd_nv4m_kb() == 8u)
        pd_nvf4_moe_down_bs_kernel<8u, false><<<grid, 256, PD_NV4M_SMEM(8u),
                                                (cudaStream_t)stream>>>(
            (const uint8_t*)data, (const uint8_t*)scale, (const float*)scale2,
            (const uint32_t*)sorted_row, (const uint32_t*)sorted_slot,
            (const uint32_t*)block_expert, (const float*)topk_w,
            (const uint8_t*)fq, (const uint8_t*)fs, (float*)part, ff, embd, kw,
            np, slot_off);
    else
        pd_nvf4_moe_down_bs_kernel<4u, false><<<grid, 256, PD_NV4M_SMEM(4u),
                                                (cudaStream_t)stream>>>(
            (const uint8_t*)data, (const uint8_t*)scale, (const float*)scale2,
            (const uint32_t*)sorted_row, (const uint32_t*)sorted_slot,
            (const uint32_t*)block_expert, (const float*)topk_w,
            (const uint8_t*)fq, (const uint8_t*)fs, (float*)part, ff, embd, kw,
            np, slot_off);
    return pd_launch_status();
#endif
}

// ---- decode multi-task NVFP4 MoE expert GEMVs (decode rung) ------
// The GEMV pair above launches each MoE layer as four small grids at decode
// (routed up 1392 CTAs, shared up 464, the two down legs 336 each with the
// k picks walked SERIALLY inside one warp) - measured
// at 337-688 GB/s effective against the same dot4 walk's 1076 GB/s on
// the wave-dense lm_head launch: the walk is fine, the grids are starved.
// These two kernels fuse each layer to two wave-dense launches:
//   up_mt:     one task per output row across all k routed slots AND the
//              shared expert (task space k*ff_r + ff_s).
//   down_part: one task per (slot, out row) - the k-serial in-warp fold is
//              split across slots, each writing its PRE-WEIGHTED partial to
//              part[slot*embd + r]; pd_moe_slot_combine then folds the np
//              slot planes in fixed ascending order into the residual.
// Rung-4b shape: CTA per TASK, the 4 warps split the K walk (warp w takes
// k0 = w*128, +512, ...) with a fixed ascending-warp combine through shared
// memory. The warp-per-task form measured 848/700 GB/s = 82%/70% of the dot4
// walk's 1076 ceiling, and both numbers are exactly waves/ceil(waves) of the
// 9024-warp capacity (1.65 and 2.09 waves) - pure wave tail. x4 the warps
// puts the tails at 6.6/8.3 waves (94%/93%). Deterministic fixed-order
// summation throughout, but the grouping differs from the warp-per-task form
// (and from the GEMV pair), so the gates are rel-to-rms + the token battery,
// with determinism still bit-gated (the rung-4a precedent class).
// 128-thread CTAs (4 warps): finer wave granularity at decode grids, per the
// Q8_0 GEMV block-width result.

__global__ void pd_nvf4_moe_up_relu2_mt_kernel(
    const uint8_t* __restrict__ rdata, const uint8_t* __restrict__ rscale,
    const float* __restrict__ rscale2, const uint8_t* __restrict__ sdata,
    const uint8_t* __restrict__ sscale, const float* __restrict__ sscale2,
    const uint32_t* __restrict__ idx, const float* __restrict__ x,
    float* __restrict__ act, uint32_t in_dim, uint32_t ff_r, uint32_t ff_s,
    uint32_t k) {
#if PD_NV4_OK
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t task = blockIdx.x;
    const uint32_t ntask = k * ff_r + ff_s;
    if (task >= ntask) return;
    const uint8_t* row;
    const uint8_t* srow;
    float s2;
    if (task < k * ff_r) {
        const uint32_t slot = task / ff_r, r = task - slot * ff_r;
        const uint32_t e = idx[slot];
        row = rdata + ((size_t)e * ff_r + r) * (in_dim >> 1);
        srow = rscale + ((size_t)e * ff_r + r) * (in_dim >> 4);
        s2 = rscale2[e];
    } else {
        const uint32_t r = task - k * ff_r;
        row = sdata + (size_t)r * (in_dim >> 1);
        srow = sscale + (size_t)r * (in_dim >> 4);
        s2 = sscale2[0];
    }
    __shared__ float psum[4];
    float acc = 0.0f;
    // full 128-steps only in the pipelined loop; the (at most one) partial
    // step lands on exactly one warp and runs guarded after - same per-lane
    // element sequence as the old in-loop break, without its 24% pipelining
    // tax (see pd_nvf4_gemv_kernel)
    const uint32_t full = in_dim & ~127u;
    uint32_t k0 = warp * 128u;
    #pragma unroll 4
    for (; k0 < full; k0 += 512u) acc += pd_nvf4_dot4(row, srow, x, k0 + lane * 4u);
    if (k0 < in_dim) {
        const uint32_t el = k0 + lane * 4u;
        if (el < in_dim) acc += pd_nvf4_dot4(row, srow, x, el);
    }
    for (uint32_t s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, s);
    if (lane == 0) psum[warp] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        const float total = ((psum[0] + psum[1]) + psum[2]) + psum[3];
        const float v = fmaxf(total * s2, 0.0f);
        act[task] = v * v;
    }
#else
    (void)rdata; (void)rscale; (void)rscale2; (void)sdata; (void)sscale;
    (void)sscale2; (void)idx; (void)x; (void)act; (void)in_dim; (void)ff_r;
    (void)ff_s; (void)k;
#endif
}

PD_EXPORT
int pd_nvf4_moe_up_relu2_mt(const void* rdata, const void* rscale,
                            const void* rscale2, const void* sdata,
                            const void* sscale, const void* sscale2,
                            const void* idx, const void* x, void* act,
                            uint32_t in_dim, uint32_t ff_r, uint32_t ff_s,
                            uint32_t k, void* stream) {
#ifndef PD_BS_HOST
    (void)rdata; (void)rscale; (void)rscale2; (void)sdata; (void)sscale;
    (void)sscale2; (void)idx; (void)x; (void)act; (void)in_dim; (void)ff_r;
    (void)ff_s; (void)k; (void)stream;
    return cudaErrorNotSupported;
#else
    if (ff_r == 0 || k == 0) return 0;
    if ((in_dim & 31u) != 0) return cudaErrorInvalidValue;
    // CTA per task, 4 warps splitting K
    const uint32_t grid = k * ff_r + ff_s;
    pd_nvf4_moe_up_relu2_mt_kernel<<<grid, 128u, 0,
                                     (cudaStream_t)stream>>>(
        (const uint8_t*)rdata, (const uint8_t*)rscale, (const float*)rscale2,
        (const uint8_t*)sdata, (const uint8_t*)sscale, (const float*)sscale2,
        (const uint32_t*)idx, (const float*)x, (float*)act, in_dim, ff_r,
        ff_s, k);
    return pd_launch_status();
#endif
}

__global__ void pd_nvf4_moe_down_part_kernel(
    const uint8_t* __restrict__ rdata, const uint8_t* __restrict__ rscale,
    const float* __restrict__ rscale2, const uint8_t* __restrict__ sdata,
    const uint8_t* __restrict__ sscale, const float* __restrict__ sscale2,
    const uint32_t* __restrict__ idx, const float* __restrict__ topk_w,
    const float* __restrict__ act, float* __restrict__ part, uint32_t ff_r,
    uint32_t ff_s, uint32_t embd, uint32_t k) {
#if PD_NV4_OK
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t task = blockIdx.x;
    if (task >= (k + 1u) * embd) return;
    const uint32_t slot = task / embd, r = task - slot * embd;
    const uint8_t* row;
    const uint8_t* srow;
    const float* xrow;
    float w;
    uint32_t kk;
    if (slot < k) {
        const uint32_t e = idx[slot];
        w = topk_w[slot] * rscale2[e];
        row = rdata + ((size_t)e * embd + r) * (ff_r >> 1);
        srow = rscale + ((size_t)e * embd + r) * (ff_r >> 4);
        xrow = act + (size_t)slot * ff_r;
        kk = ff_r;
    } else {
        w = sscale2[0];
        row = sdata + (size_t)r * (ff_s >> 1);
        srow = sscale + (size_t)r * (ff_s >> 4);
        xrow = act + (size_t)k * ff_r;
        kk = ff_s;
    }
    // the down_acc fold shape per warp (walk, then w per lane, then reduce)
    // over the warp's K quarter; fixed ascending-warp combine, then the
    // 0.0f + acc write
    __shared__ float psum[4];
    float p = 0.0f;
    // same tail hoist as up_mt above (kk = ff_r 1856 does have a live
    // 64-wide tail here - it lands on one warp's guarded step)
    const uint32_t full = kk & ~127u;
    uint32_t k0 = warp * 128u;
    #pragma unroll 4
    for (; k0 < full; k0 += 512u) p += pd_nvf4_dot4(row, srow, xrow, k0 + lane * 4u);
    if (k0 < kk) {
        const uint32_t el = k0 + lane * 4u;
        if (el < kk) p += pd_nvf4_dot4(row, srow, xrow, el);
    }
    float acc = w * p;
    for (uint32_t s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, s);
    if (lane == 0) psum[warp] = acc;
    __syncthreads();
    if (threadIdx.x == 0)
        part[task] = 0.0f + (((psum[0] + psum[1]) + psum[2]) + psum[3]);
#else
    (void)rdata; (void)rscale; (void)rscale2; (void)sdata; (void)sscale;
    (void)sscale2; (void)idx; (void)topk_w; (void)act; (void)part; (void)ff_r;
    (void)ff_s; (void)embd; (void)k;
#endif
}

PD_EXPORT
int pd_nvf4_moe_down_part(const void* rdata, const void* rscale,
                          const void* rscale2, const void* sdata,
                          const void* sscale, const void* sscale2,
                          const void* idx, const void* topk_w, const void* act,
                          void* part, uint32_t ff_r, uint32_t ff_s,
                          uint32_t embd, uint32_t k, void* stream) {
#ifndef PD_BS_HOST
    (void)rdata; (void)rscale; (void)rscale2; (void)sdata; (void)sscale;
    (void)sscale2; (void)idx; (void)topk_w; (void)act; (void)part; (void)ff_r;
    (void)ff_s; (void)embd; (void)k; (void)stream;
    return cudaErrorNotSupported;
#else
    if (embd == 0 || k == 0) return 0;
    if ((ff_r & 31u) != 0 || (ff_s & 31u) != 0) return cudaErrorInvalidValue;
    // CTA per task, 4 warps splitting K
    const uint32_t grid = (k + 1u) * embd;
    pd_nvf4_moe_down_part_kernel<<<grid, 128u, 0,
                                   (cudaStream_t)stream>>>(
        (const uint8_t*)rdata, (const uint8_t*)rscale, (const float*)rscale2,
        (const uint8_t*)sdata, (const uint8_t*)sscale, (const float*)sscale2,
        (const uint32_t*)idx, (const float*)topk_w, (const float*)act,
        (float*)part, ff_r, ff_s, embd, k);
    return pd_launch_status();
#endif
}
