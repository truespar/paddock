// The diarization encoder's projections in the BF16 class (slot 965) - how
// Transcribe's speaker branch is served (vLLM's NeMo plugin runs the
// perception module in bf16). Slot 722 is this encoder's F32 class: the f32
// activation split into three bf16 parts in registers against the exact
// bf16 weight. That kernel is bound by staging and splitting its f32
// operand, not by its mma (one part instead of three, slot 964, buys 0-7%
// at a window's 500-828 rows), so the bf16 class here is a different tile:
// both operands bf16 on the cp.async ring - the activation landed as bf16
// by whatever produced it (the LayerNorm 966, the attention 967, the FFN
// up's GELU below) - on the f16 blocked tile's ring (gemm/f16_dense.cuh:
// three swizzled stages of 32 elements a K row, one barrier a K step, the
// same ldmatrix addresses) with the bf16 mma. Every tile shape walks a
// row's K in the same order, so the election is shape-only and never moves
// a bit, and a row's outputs do not depend on the other rows in the pass.
//
// The output tile is BM features x BN rows (W `[feat][K]` the checkpoint's
// rows as stored, X `[rows][K]`), landed `[rows][feat]` as slot 722 lands
// it, with that kernel's epilogues on `acc + bias` (bias optional):
//   0 store f32;  1 GELU (torch.nn.GELU, erf) landed as bf16 - the down
//   projection's operand;  2 accumulate into y (the residual, bias added
//   first);  3 the qkv projection (features 1536 = q | k | v, 8 heads of 64
//   each): split-half rope at position row on q and k from the (cos, sin)
//   table `[rows][32]`, q into its own `[rows][512]` plane y2, k roped in
//   place, v as computed - a warp's 64 feature rows are one head, so dims j
//   and j + 32 are one lane's m-groups rg and rg + 2.

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define PD_DB16_OK 1
#else
#define PD_DB16_OK 0
#endif

#define PD_DB16_STORE 0u
#define PD_DB16_GELU 1u
#define PD_DB16_RESID 2u
#define PD_DB16_ROPE 3u

#if PD_DB16_OK
// D = A*B + D, m16n8k16 bf16 x bf16 -> f32: the f16 mma's fragments
__device__ __forceinline__ void pd_db16_mma(float d[4], const uint32_t a[4], const uint32_t b[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
#endif

template <uint32_t BM, uint32_t BN, uint32_t NW, uint32_t RG, uint32_t CG, uint32_t EPI>
__global__ void __launch_bounds__(NW * 32u) pd_diar_b16_kernel(
        const __nv_bfloat16* __restrict__ W, const __nv_bfloat16* __restrict__ X,
        void* __restrict__ Y, float* __restrict__ Y2, const float* __restrict__ bias,
        const float2* __restrict__ rope, uint32_t K, uint32_t F, uint32_t R) {
#if PD_DB16_OK
    constexpr uint32_t NTH = NW * 32u, ST = 3u, KT = 32u, NSUBK = KT / 16u, H8PR = KT / 8u;
    constexpr uint32_t WM = RG * 16u, WN = CG * 8u, WR = BM / WM;
    static_assert(WR * (BN / WN) == NW && WM * WR == BM, "warp grid");
    static_assert(EPI != PD_DB16_ROPE || RG == 4u, "the rope pairs m-groups rg, rg + 2 of a 64-row head");
    extern __shared__ __align__(16) __half pd_db16_dyn[];
    auto sh_a = reinterpret_cast<__half(*)[BM * KT]>(pd_db16_dyn);
    auto sh_b = reinterpret_cast<__half(*)[BN * KT]>(pd_db16_dyn + ST * BM * KT);
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, t = lane & 3u;
    const uint32_t wr = (warp % WR) * WM, wc = (warp / WR) * WN;
    const uint32_t f0 = blockIdx.x * BM, r0 = blockIdx.y * BN;

    // K % 8 == 0 (the host checks): a 16 B chunk is whole or past K
    auto stage = [&](uint32_t k0, uint32_t buf) {
        #pragma unroll
        for (uint32_t i = tid; i < BM * H8PR; i += NTH) {
            const uint32_t row = i / H8PR, h8 = (i % H8PR) * 8u, gk = k0 + h8;
            const bool ok = f0 + row < F && gk < K;
            pd_f16_cpa16(&sh_a[buf][pd_f16_sidx<KT, true>(row, h8)],
                         W + (size_t)(ok ? f0 + row : 0u) * K + (ok ? gk : 0u), ok);
        }
        #pragma unroll
        for (uint32_t i = tid; i < BN * H8PR; i += NTH) {
            const uint32_t col = i / H8PR, h8 = (i % H8PR) * 8u, gk = k0 + h8;
            const bool ok = r0 + col < R && gk < K;
            pd_f16_cpa16(&sh_b[buf][pd_f16_sidx<KT, true>(col, h8)],
                         X + (size_t)(ok ? r0 + col : 0u) * K + (ok ? gk : 0u), ok);
        }
    };
    const uint32_t l7 = lane & 7u;
    const uint32_t a_roff = ((lane & 8u) ? 8u : 0u) + l7;
    const uint32_t a_kof = (lane & 16u) ? 8u : 0u;
    const uint32_t b_kof = (lane & 8u) ? 8u : 0u;
    float acc[RG][CG][4] = {};
    auto compute = [&](uint32_t buf) {
        #pragma unroll
        for (uint32_t sk = 0; sk < NSUBK; ++sk) {
            const uint32_t ko = sk * 16u;
            uint32_t a[RG][4], b[CG][2];
            #pragma unroll
            for (uint32_t rg = 0; rg < RG; ++rg)
                pd_f16_ldm_x4(&sh_a[buf][pd_f16_sidx<KT, true>(wr + rg * 16u + a_roff, ko + a_kof)],
                              a[rg][0], a[rg][1], a[rg][2], a[rg][3]);
            #pragma unroll
            for (uint32_t cg = 0; cg < CG; ++cg)
                pd_f16_ldm_x2(&sh_b[buf][pd_f16_sidx<KT, true>(wc + cg * 8u + l7, ko + b_kof)],
                              b[cg][0], b[cg][1]);
            #pragma unroll
            for (uint32_t rg = 0; rg < RG; ++rg)
                #pragma unroll
                for (uint32_t cg = 0; cg < CG; ++cg) pd_db16_mma(acc[rg][cg], a[rg], b[cg]);
        }
    };
    #pragma unroll
    for (uint32_t s = 0; s < ST - 1u; ++s) {
        if (s * KT < K) stage(s * KT, s);
        pd_f16_cpa_commit();
    }
    uint32_t p = 0;
    for (uint32_t k0 = 0; k0 < K; k0 += KT) {
        pd_f16_cpa_waitN<(int)ST - 2>();
        __syncthreads();
        const uint32_t pre = k0 + (ST - 1u) * KT;
        if (pre < K) stage(pre, (p + ST - 1u) % ST);
        pd_f16_cpa_commit();
        compute(p);
        p = (p + 1u) % ST;
    }

    // element (feature f, row r): d0 = (g, 2t), d1 = (g, 2t+1), d2 = (g+8,
    // 2t), d3 = (g+8, 2t+1)
    if constexpr (EPI == PD_DB16_ROPE) {
        // this warp's 64 feature rows are head (f0 + wr) / 64: q 0-7, k 8-15,
        // v 16-23
        const uint32_t head = (f0 + wr) / 64u;
        float* yf = reinterpret_cast<float*>(Y);
        #pragma unroll
        for (uint32_t rg = 0; rg < 2u; ++rg)
            #pragma unroll
            for (uint32_t h = 0; h < 2u; ++h) {
                const uint32_t j = rg * 16u + g + 8u * h;  // dim j, partner j + 32
                #pragma unroll
                for (uint32_t cg = 0; cg < CG; ++cg)
                    #pragma unroll
                    for (uint32_t e = 0; e < 2u; ++e) {
                        const uint32_t r = r0 + wc + cg * 8u + 2u * t + e;
                        if (r >= R || head >= 24u) continue;
                        float a = acc[rg][cg][2u * h + e], c = acc[rg + 2u][cg][2u * h + e];
                        if (bias) {
                            a = __fadd_rn(a, bias[head * 64u + j]);
                            c = __fadd_rn(c, bias[head * 64u + j + 32u]);
                        }
                        if (head < 16u) {
                            const float2 cs = rope[(size_t)r * 32u + j];
                            const float x1 = a;
                            a = __fadd_rn(__fmul_rn(x1, cs.x), __fmul_rn(-c, cs.y));
                            c = __fadd_rn(__fmul_rn(c, cs.x), __fmul_rn(x1, cs.y));
                        }
                        float* o = head < 8u ? Y2 + (size_t)r * 512u + head * 64u
                                             : yf + (size_t)r * F + head * 64u;
                        o[j] = a;
                        o[j + 32u] = c;
                    }
            }
        return;
    }
    #pragma unroll
    for (uint32_t rg = 0; rg < RG; ++rg)
        #pragma unroll
        for (uint32_t h = 0; h < 2u; ++h) {
            const uint32_t f = f0 + wr + rg * 16u + g + 8u * h;
            if (f >= F) continue;
            const float bf = bias ? bias[f] : 0.0f;
            #pragma unroll
            for (uint32_t cg = 0; cg < CG; ++cg)
                #pragma unroll
                for (uint32_t e = 0; e < 2u; ++e) {
                    const uint32_t r = r0 + wc + cg * 8u + 2u * t + e;
                    if (r >= R) continue;
                    float z = acc[rg][cg][2u * h + e];
                    if (bias) z = __fadd_rn(z, bf);
                    const size_t o = (size_t)r * F + f;
                    if (EPI == PD_DB16_GELU)
                        reinterpret_cast<__nv_bfloat16*>(Y)[o] = __float2bfloat16_rn(pd_kumo_gelu(z));
                    else if (EPI == PD_DB16_RESID)
                        reinterpret_cast<float*>(Y)[o] = __fadd_rn(reinterpret_cast<float*>(Y)[o], z);
                    else
                        reinterpret_cast<float*>(Y)[o] = z;
                }
        }
#else
    (void)W; (void)X; (void)Y; (void)Y2; (void)bias; (void)rope; (void)K; (void)F; (void)R;
#endif
}

template <uint32_t BM, uint32_t BN, uint32_t NW, uint32_t RG, uint32_t CG, uint32_t EPI>
static void pd_diar_b16_go(const __nv_bfloat16* w, const __nv_bfloat16* x, void* y, float* y2,
                           const float* bias, const float2* rope, uint32_t K, uint32_t F,
                           uint32_t R, cudaStream_t s) {
    constexpr unsigned smem = 3u * (BM + BN) * 32u * 2u;
    const dim3 grid((F + BM - 1u) / BM, (R + BN - 1u) / BN);
    pd_diar_b16_kernel<BM, BN, NW, RG, CG, EPI><<<grid, NW * 32u, smem, s>>>(w, x, y, y2, bias, rope, K, F, R);
}

// Shape-only election by grid fill (every arm the same bits): the 128 x 128
// tile (eight warps of 64 x 32) while its grid gives every SM a block, then
// 64 x 64 (four warps of 64 x 16) while it gives two, then 64 x 32 (two
// warps); below that the
// planes without the rope (a 64-row head a warp) take 32 x 32 and 16 x 32,
// so a short pass still keeps enough blocks in flight to stream the weights.
template <uint32_t EPI>
static void pd_diar_b16_elect(const __nv_bfloat16* w, const __nv_bfloat16* x, void* y, float* y2,
                              const float* bias, const float2* rope, uint32_t K, uint32_t F, uint32_t R,
                              cudaStream_t s) {
    static int nsm = 0;
    if (nsm == 0) {
        int d = 0;
        cudaGetDevice(&d);
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, d);
        if (nsm <= 0) nsm = 48;
    }
    const uint64_t big = (uint64_t)((F + 127u) / 128u) * ((R + 127u) / 128u);
    const uint64_t mid = (uint64_t)((F + 63u) / 64u) * ((R + 63u) / 64u);
    // not `small`: the Windows SDK's rpcndr.h defines it as `char`
    const uint64_t narrow = (uint64_t)((F + 63u) / 64u) * ((R + 31u) / 32u);
    const uint64_t s32 = (uint64_t)((F + 31u) / 32u) * ((R + 31u) / 32u);
    if (big >= (uint64_t)nsm)
        pd_diar_b16_go<128u, 128u, 8u, 4u, 4u, EPI>(w, x, y, y2, bias, rope, K, F, R, s);
    else if (mid >= 2u * (uint64_t)nsm)
        pd_diar_b16_go<64u, 64u, 4u, 4u, 2u, EPI>(w, x, y, y2, bias, rope, K, F, R, s);
    else if constexpr (EPI == PD_DB16_ROPE)
        pd_diar_b16_go<64u, 32u, 2u, 4u, 2u, EPI>(w, x, y, y2, bias, rope, K, F, R, s);
    else if (narrow >= (uint64_t)nsm)
        pd_diar_b16_go<64u, 32u, 2u, 4u, 2u, EPI>(w, x, y, y2, bias, rope, K, F, R, s);
    else if (s32 >= (uint64_t)nsm)
        pd_diar_b16_go<32u, 32u, 2u, 2u, 2u, EPI>(w, x, y, y2, bias, rope, K, F, R, s);
    else
        pd_diar_b16_go<16u, 32u, 2u, 1u, 2u, EPI>(w, x, y, y2, bias, rope, K, F, R, s);
}

// 965: y = epi(x16 . W + bias) over `rows` bf16 rows of K, W the stored BF16
// rows `[F][K]`; epi as above (1 lands bf16 `[rows][F]`, 3 takes F 1536, the
// q plane y2 and the (cos, sin) table `[rows][32]`).
PD_EXPORT
int pd_diar_b16_gemm(const void* x16, const void* w, const void* bias, void* y, void* y2,
                     const void* rope, uint32_t K, uint32_t F, uint32_t rows, uint32_t epi,
                     void* stream) {
    if (rows == 0 || F == 0) return 0;
    if (K == 0 || K % 8u || F % 64u || epi > PD_DB16_ROPE ||
        (epi == PD_DB16_ROPE && (F != 1536u || !y2 || !rope)) ||
        (((uintptr_t)x16 | (uintptr_t)w) & 15u) || (rows + 31u) / 32u > 65535u)
        return (int)cudaErrorInvalidValue;
    const auto* wp = (const __nv_bfloat16*)w;
    const auto* xp = (const __nv_bfloat16*)x16;
    const auto* bp = (const float*)bias;
    const auto* tp = (const float2*)rope;
    const cudaStream_t s = (cudaStream_t)stream;
    switch (epi) {
    case PD_DB16_STORE: pd_diar_b16_elect<PD_DB16_STORE>(wp, xp, y, (float*)y2, bp, tp, K, F, rows, s); break;
    case PD_DB16_GELU: pd_diar_b16_elect<PD_DB16_GELU>(wp, xp, y, (float*)y2, bp, tp, K, F, rows, s); break;
    case PD_DB16_RESID: pd_diar_b16_elect<PD_DB16_RESID>(wp, xp, y, (float*)y2, bp, tp, K, F, rows, s); break;
    default: pd_diar_b16_elect<PD_DB16_ROPE>(wp, xp, y, (float*)y2, bp, tp, K, F, rows, s); break;
    }
    return pd_launch_status();
}
