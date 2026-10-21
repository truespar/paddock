// gemm/f32_tf32.cuh - the tf32 tensor-core tiles of the f32 NT GEMM family: the 32x32 cp.async
// ring tile (the DeltaNet alpha|beta plane, the ba rung) and the WIDE 128x128 3xTF32 tile.
// Included in place by gemm/f32_qkv.cuh (needs its PD_RGEMM_BK; its launchers elect these);
// moved out of it unchanged.

// tf32 tensor-core twin of the nt/nt_ks pair (ba rung): at the
// c16 wave the ba plane (out=96, batch~4k, K=5120) ran the SIMT kernel at
// its own LDS-bound ceiling (204.8 us/layer, ~20 TF - the 4x2 micro-tile
// pays 6 shared loads per 8 FMAs) where a bf16 nvjet runs the same plane in
// 35.1 us. 3xTF32 (big+small correction, ascending-k in k8
// groups) is STRICTLY finer than that bf16 class, and the earlier
// 3xTF32 arm's +0.58% PPL was already shown to be chaos-band RESAMPLING
// (in-band vs the known-benign perturbation), not bias - f32 shipped then
// only because the grid, not the rate, was the problem. No K-split here:
// the route gates on (batch/64)x(out/32) >= 148 CTAs, so the wave fills
// without partials (the K-split existed for the compute-starved SIMT
// form). PREC=1 (single tf32) kept as a probe arm.
static __device__ __forceinline__ uint32_t pd_bnt_tf32(float v) {
    uint32_t r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(v));
    return r;
}
// NS: cp.async ring depth. 2 (the original) kept one stage in flight, so every
// k-step paid an L2 round trip (~0.6 us a step, 196 us for the 96-out ba plane at
// ~3000 rows on B200); deeper rings keep NS-1 in flight. The per-element k order
// is NS-independent - bit-identical at every depth (bench/tf32_ns_probe.cu).
// Deeper than 4 (the ring in dynamic smem) falsified 2026-10-21: bit-identical but
// slower alone (1760 rows: NS4 57.4, NS6 72.0, NS8 66.2 us). The serve's ~138 us a
// call is the decode lane sharing the SMs, not the ring.
template <uint32_t PREC, uint32_t NS = 4u>
__global__ void __launch_bounds__(128)
pd_gemm_tf32_nt_kernel(const float* __restrict__ w, const float* __restrict__ x,
                       float* __restrict__ out, uint32_t in_dim,
                       uint32_t out_dim, uint32_t batch) {
    // ROW-major tiles + cp.async 2-slot ring: the SIMT family's k-major
    // transpose scatter costs 12 scalar STS/thread/step (~384 STS per
    // warp-step - the family's real wall; the probe read both the mma
    // rewrite and a register double-buffer as ~x1.2). Row-major stages as
    // three 16B cp.async per thread and the tf32 fragments read it
    // conflict-free (stride 36 floats: bank = 4*row + col mod 32).
    constexpr uint32_t BM = 32u, BN = 32u, BK = PD_RGEMM_BK, SK = BK + 4u;
    static_assert(NS >= 2u && NS <= 4u, "ring depth");
    __shared__ float xs[NS][BM][SK];
    __shared__ float ws[NS][BN][SK];
    const uint32_t m0 = blockIdx.x * BM, n0 = blockIdx.y * BN;
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const uint32_t gr = lane >> 2, t4 = lane & 3u;
    const uint32_t mt = warp & 1u, nh = warp >> 1;  // m16 tile / n16 half
    const uint32_t xr = tid / (BK / 4u), xk = (tid % (BK / 4u)) * 4u;
    // Two-level accumulation. The DPU behind mma.f32.tf32
    // accumulates the C chain with TRUNCATION (RZ) and a few guard bits
    // (Fasi/Higham/Mikaitis/Pranesh 2021 - study ref), so a running K-sum
    // passed through C compounds a ~2^-27-per-pass BIAS: measured K*2^-27
    // exactly on the kquant parity rung (K=4096 -> 3.83e-5, K=12288 ->
    // 1.15e-4 - LINEAR in K, the truncation-bias signature), drowning the
    // 3xTF32 split's ~1e-6 operand class. acc[] therefore chains only the
    // 12 mma of one BK tile; fac[] carries the cross-tile sum in CUDA-core
    // FADD (round-nearest), turning the linear bias into a per-tile random
    // walk. Same two-level technique as CUTLASS/FA3 fp8 accumulation
    // promotion (inspiration only - original implementation). The extra 8
    // FADDs per tile ride free: the kernel is staging-bound (rung
    // 9 measured 3x the mma work at 1.00x wall).
    float acc[2][4] = {};
    float fac[2][4] = {};
    auto cp16 = [](float* dst, const float* src, uint32_t bytes) {
        const unsigned sm = (unsigned)__cvta_generic_to_shared(dst);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(sm),
                     "l"(src), "r"(bytes));
    };
    auto stage = [&](uint32_t slot, uint32_t k0) {
        cp16(&xs[slot][xr][xk], x + (size_t)(m0 + xr) * in_dim + k0 + xk,
             m0 + xr < batch ? 16u : 0u);
        cp16(&xs[slot][xr + 16u][xk],
             x + (size_t)(m0 + xr + 16u) * in_dim + k0 + xk,
             m0 + xr + 16u < batch ? 16u : 0u);
        cp16(&ws[slot][xr][xk], w + (size_t)(n0 + xr) * in_dim + k0 + xk, 16u);
        cp16(&ws[slot][xr + 16u][xk],
             w + (size_t)(n0 + xr + 16u) * in_dim + k0 + xk, 16u);
        asm volatile("cp.async.commit_group;" ::: "memory");
    };
    const uint32_t nkt = in_dim / BK;
    #pragma unroll
    for (uint32_t i = 0; i + 1u < NS; ++i) {
        if (i < nkt) stage(i, i * BK);
        else asm volatile("cp.async.commit_group;" ::: "memory");
    }
    for (uint32_t k0 = 0; k0 < in_dim; k0 += BK) {
        const uint32_t kt = k0 / BK, slot = kt % NS;
        // tile kt has landed once at most NS-2 newer groups are pending
        asm volatile("cp.async.wait_group %0;" ::"n"(NS - 2u) : "memory");
        __syncthreads();
        // every thread is past tile kt-1: its slot takes tile kt+NS-1
        if (kt + NS - 1u < nkt) stage((kt + NS - 1u) % NS, (kt + NS - 1u) * BK);
        else asm volatile("cp.async.commit_group;" ::: "memory");
        #pragma unroll
        for (uint32_t k8 = 0; k8 < BK; k8 += 8u) {
            // A m16k8, converted once for both n8 tiles
            uint32_t ab[4], as[4];
            {
                const float a0 = xs[slot][mt * 16u + gr][k8 + t4];
                const float a1 = xs[slot][mt * 16u + gr + 8u][k8 + t4];
                const float a2 = xs[slot][mt * 16u + gr][k8 + t4 + 4u];
                const float a3 = xs[slot][mt * 16u + gr + 8u][k8 + t4 + 4u];
                ab[0] = pd_bnt_tf32(a0); ab[1] = pd_bnt_tf32(a1);
                ab[2] = pd_bnt_tf32(a2); ab[3] = pd_bnt_tf32(a3);
                if (PREC == 3u) {
                    as[0] = pd_bnt_tf32(a0 - __uint_as_float(ab[0]));
                    as[1] = pd_bnt_tf32(a1 - __uint_as_float(ab[1]));
                    as[2] = pd_bnt_tf32(a2 - __uint_as_float(ab[2]));
                    as[3] = pd_bnt_tf32(a3 - __uint_as_float(ab[3]));
                }
            }
            #pragma unroll
            for (uint32_t nt = 0; nt < 2u; ++nt) {
                const float b0 = ws[slot][nh * 16u + nt * 8u + gr][k8 + t4];
                const float b1 = ws[slot][nh * 16u + nt * 8u + gr][k8 + t4 + 4u];
                uint32_t bb[2] = {pd_bnt_tf32(b0), pd_bnt_tf32(b1)};
                asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+f"(acc[nt][0]), "+f"(acc[nt][1]), "+f"(acc[nt][2]),
                      "+f"(acc[nt][3])
                    : "r"(ab[0]), "r"(ab[1]), "r"(ab[2]), "r"(ab[3]),
                      "r"(bb[0]), "r"(bb[1]));
                if (PREC == 3u) {
                    uint32_t bs[2] = {pd_bnt_tf32(b0 - __uint_as_float(bb[0])),
                                      pd_bnt_tf32(b1 - __uint_as_float(bb[1]))};
                    asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+f"(acc[nt][0]), "+f"(acc[nt][1]), "+f"(acc[nt][2]),
                          "+f"(acc[nt][3])
                        : "r"(ab[0]), "r"(ab[1]), "r"(ab[2]), "r"(ab[3]),
                          "r"(bs[0]), "r"(bs[1]));
                    asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+f"(acc[nt][0]), "+f"(acc[nt][1]), "+f"(acc[nt][2]),
                          "+f"(acc[nt][3])
                        : "r"(as[0]), "r"(as[1]), "r"(as[2]), "r"(as[3]),
                          "r"(bb[0]), "r"(bb[1]));
                }
            }
        }
        // drain the tile's mma chain into the RN accumulator (see fac[])
        #pragma unroll
        for (uint32_t nt = 0; nt < 2u; ++nt)
            #pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) {
                fac[nt][e] += acc[nt][e];
                acc[nt][e] = 0.f;
            }
        __syncthreads();
    }
    #pragma unroll
    for (uint32_t nt = 0; nt < 2u; ++nt)
        #pragma unroll
        for (uint32_t e = 0; e < 4u; ++e) {
            const uint32_t m = m0 + mt * 16u + gr + (e >= 2u ? 8u : 0u);
            const uint32_t n = n0 + nh * 16u + nt * 8u + 2u * t4 + (e & 1u);
            if (m < batch) out[(size_t)m * out_dim + n] = fac[nt][e];
        }
}

// The WIDE 3xTF32 tile (GB10, 2026-09-11): the
// 32x32 tile above is staging-bound at ~10 TF/s useful on the 48-SM die -
// PREC 1 and PREC 3 time the same there too (bench/tf32_gb10_bench.cu: q
// 2688->4096 at 1024 rows 2313 vs 2340 us), so the three mma per k8 are
// free and what binds is the per-fragment work around them: each converted
// A fragment feeds two n8 tiles and each B fragment one m16 tile, so every
// mma pays ~4 cvt/fsub plus a scalar smem load. This tile is 128 x 128 per
// CTA, 8 warps as 2 (m) x 4 (n), a 64 x 32 warp tile (4 m16 x 4 n8): a
// converted A fragment serves four n8 tiles and a B fragment four m16
// tiles, ~1.5 conversions per mma. BIT-EXACT vs the tile above by
// construction - the per-element k sequence is unchanged: the same
// cvt.rna split of A and B, the same three mma per k8 in the same order
// (big.big, big.small, small.big), k8 ascending within the BK = 32 tile,
// the same drain of the tile's mma chain into the RN accumulator (fac +=
// acc; acc = 0) per BK tile, tiles ascending. Only the ownership of the
// (m, n) pairs moves. Same smem geometry per row (stride SK = 36 floats,
// conflict-free scalar fragment loads), 2 slots of 128 + 128 rows = 72 KB
// dynamic smem, one CTA per SM. Elected for planes whose grid fills the
// die (see pd_gemm_f32); k/v (out 256) stay on the 32x32 tile.
template <uint32_t PREC>
__global__ void __launch_bounds__(256, 1)
pd_gemm_tf32_nt_wide_kernel(const float* __restrict__ w, const float* __restrict__ x,
                            float* __restrict__ out, uint32_t in_dim,
                            uint32_t out_dim, uint32_t batch) {
    constexpr uint32_t BM = 128u, BN = 128u, BK = PD_RGEMM_BK, SK = BK + 4u;
    constexpr uint32_t MT = 4u, NT = 4u;   // m16 tiles x n8 tiles per warp
    extern __shared__ float pd_tfw_sh[];
    float* xs = pd_tfw_sh;                      // [2][BM][SK]
    float* ws = pd_tfw_sh + 2u * BM * SK;       // [2][BN][SK]
    // Grouped raster (bit-exact: only which CTA owns which tile changes).
    // The grid is 1-D; consecutive CTAs walk the n tiles of a GROUP of 8 m
    // tiles before the next group, so a group's 8 x tiles (8 x 128 rows x
    // K x 4 B = 11 MB at K 2688) stay in L2 while the w plane streams once
    // per group. With the plain (m fastest) raster the x panel was
    // re-streamed once per n tile - 32 x 81 MB at 7556 rows on the q plane,
    // 14.5 ms against 3.0 at 2048 rows where the panel still fit (GB10
    // 2026-09-11). The last group is the ragged tail.
    const uint32_t mtiles = (batch + BM - 1u) / BM, ntiles = out_dim / BN;
    constexpr uint32_t GROUP = 8u;
    const uint32_t bid = blockIdx.x;
    const uint32_t grp = bid / (GROUP * ntiles);
    const uint32_t gm0 = grp * GROUP;
    const uint32_t gsz = (gm0 + GROUP <= mtiles) ? GROUP : (mtiles - gm0);
    const uint32_t rem = bid - grp * (GROUP * ntiles);
    const uint32_t m0 = (gm0 + rem % gsz) * BM, n0 = (rem / gsz) * BN;
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const uint32_t gr = lane >> 2, t4 = lane & 3u;
    const uint32_t wm = warp >> 2, wn = warp & 3u;   // warp tile origin: (wm*64, wn*32)
    const uint32_t xr = tid >> 3, xk = (tid & 7u) * 4u;   // staging: row xr + 32j, 16 B at xk
    float acc[MT][NT][4] = {};
    float fac[MT][NT][4] = {};
    auto cp16 = [](float* dst, const float* src, uint32_t bytes) {
        const unsigned sm = (unsigned)__cvta_generic_to_shared(dst);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(sm),
                     "l"(src), "r"(bytes));
    };
    auto stage = [&](uint32_t slot, uint32_t k0) {
        float* xsl = xs + slot * BM * SK;
        float* wsl = ws + slot * BN * SK;
        #pragma unroll
        for (uint32_t j = 0; j < BM / 32u; ++j) {
            const uint32_t r = xr + 32u * j;
            cp16(xsl + r * SK + xk, x + (size_t)(m0 + r) * in_dim + k0 + xk,
                 m0 + r < batch ? 16u : 0u);
        }
        #pragma unroll
        for (uint32_t j = 0; j < BN / 32u; ++j) {
            const uint32_t r = xr + 32u * j;
            cp16(wsl + r * SK + xk, w + (size_t)(n0 + r) * in_dim + k0 + xk, 16u);
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    };
    stage(0u, 0u);
    for (uint32_t k0 = 0; k0 < in_dim; k0 += BK) {
        const uint32_t slot = (k0 / BK) & 1u;
        const bool more = k0 + BK < in_dim;
        if (more) stage(slot ^ 1u, k0 + BK);
        if (more) asm volatile("cp.async.wait_group 1;" ::: "memory");
        else asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        const float* xsl = xs + slot * BM * SK;
        const float* wsl = ws + slot * BN * SK;
        #pragma unroll
        for (uint32_t k8 = 0; k8 < BK; k8 += 8u) {
            uint32_t ab[MT][4], as[MT][4];
            #pragma unroll
            for (uint32_t mt = 0; mt < MT; ++mt) {
                const float* ar = xsl + (wm * 64u + mt * 16u + gr) * SK + k8 + t4;
                const float a0 = ar[0];
                const float a1 = ar[8u * SK];
                const float a2 = ar[4u];
                const float a3 = ar[8u * SK + 4u];
                ab[mt][0] = pd_bnt_tf32(a0); ab[mt][1] = pd_bnt_tf32(a1);
                ab[mt][2] = pd_bnt_tf32(a2); ab[mt][3] = pd_bnt_tf32(a3);
                if (PREC == 3u) {
                    as[mt][0] = pd_bnt_tf32(a0 - __uint_as_float(ab[mt][0]));
                    as[mt][1] = pd_bnt_tf32(a1 - __uint_as_float(ab[mt][1]));
                    as[mt][2] = pd_bnt_tf32(a2 - __uint_as_float(ab[mt][2]));
                    as[mt][3] = pd_bnt_tf32(a3 - __uint_as_float(ab[mt][3]));
                }
            }
            #pragma unroll
            for (uint32_t nt = 0; nt < NT; ++nt) {
                const float* br = wsl + (wn * 32u + nt * 8u + gr) * SK + k8 + t4;
                const float b0 = br[0];
                const float b1 = br[4u];
                uint32_t bb[2] = {pd_bnt_tf32(b0), pd_bnt_tf32(b1)};
                uint32_t bs[2] = {0u, 0u};
                if (PREC == 3u) {
                    bs[0] = pd_bnt_tf32(b0 - __uint_as_float(bb[0]));
                    bs[1] = pd_bnt_tf32(b1 - __uint_as_float(bb[1]));
                }
                #pragma unroll
                for (uint32_t mt = 0; mt < MT; ++mt) {
                    asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+f"(acc[mt][nt][0]), "+f"(acc[mt][nt][1]),
                          "+f"(acc[mt][nt][2]), "+f"(acc[mt][nt][3])
                        : "r"(ab[mt][0]), "r"(ab[mt][1]), "r"(ab[mt][2]), "r"(ab[mt][3]),
                          "r"(bb[0]), "r"(bb[1]));
                    if (PREC == 3u) {
                        asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                            : "+f"(acc[mt][nt][0]), "+f"(acc[mt][nt][1]),
                              "+f"(acc[mt][nt][2]), "+f"(acc[mt][nt][3])
                            : "r"(ab[mt][0]), "r"(ab[mt][1]), "r"(ab[mt][2]), "r"(ab[mt][3]),
                              "r"(bs[0]), "r"(bs[1]));
                        asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                            : "+f"(acc[mt][nt][0]), "+f"(acc[mt][nt][1]),
                              "+f"(acc[mt][nt][2]), "+f"(acc[mt][nt][3])
                            : "r"(as[mt][0]), "r"(as[mt][1]), "r"(as[mt][2]), "r"(as[mt][3]),
                              "r"(bb[0]), "r"(bb[1]));
                    }
                }
            }
        }
        // drain the tile's mma chain into the RN accumulator (the 32x32
        // tile's two-level accumulation, per BK tile, verbatim)
        #pragma unroll
        for (uint32_t mt = 0; mt < MT; ++mt)
            #pragma unroll
            for (uint32_t nt = 0; nt < NT; ++nt)
                #pragma unroll
                for (uint32_t e = 0; e < 4u; ++e) {
                    fac[mt][nt][e] += acc[mt][nt][e];
                    acc[mt][nt][e] = 0.f;
                }
        __syncthreads();
    }
    #pragma unroll
    for (uint32_t mt = 0; mt < MT; ++mt)
        #pragma unroll
        for (uint32_t nt = 0; nt < NT; ++nt)
            #pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) {
                const uint32_t m = m0 + wm * 64u + mt * 16u + gr + (e >= 2u ? 8u : 0u);
                const uint32_t n = n0 + wn * 32u + nt * 8u + 2u * t4 + (e & 1u);
                if (m < batch) out[(size_t)m * out_dim + n] = fac[mt][nt][e];
            }
}
#define PD_TFW_SMEM (2u * 2u * 128u * (PD_RGEMM_BK + 4u) * 4u)
