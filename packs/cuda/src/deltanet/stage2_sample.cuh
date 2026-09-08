// deltanet/stage2_sample.cuh (formerly 08_dnc_stage2_sample.cuh) - DeltaNet chunked stage-2 tensor-core rebuild, two-level scan, argmax/sampling
// Textually-included segment of the single pack translation unit.
// Not standalone-compilable: include order is defined by ../pack.cu.
// ---- stage 2, TENSOR-CORE rebuild. The scalar stage2 above is
// the single biggest kernel on the losing gensweep configs (15.7% of pf8 /
// 12.2% of c32 GPU time) and runs at ~9 TF effective: 128 blocks each walking
// every chunk serially, all f32 scalar FMA, with the dw/q/k rows re-read 8x
// across the threads sharing a row-pair. This variant keeps the exact same
// math shape and grid but runs the three inner contractions per chunk as
// warp-cooperative mma.sync m16n8k8 tf32 passes over smem-staged panes:
//   pass 1: [dw; q] stacked (2C x D) x S0 (D x G)  -> delta pre-terms + o1
//           (both products share the same B operand = the resident state)
//   pass 2: coef (C x C) x deltaT (C x G), accumulated into the gam-scaled
//           o1 fragments (same fragment coordinates) -> out rows
//   pass 3: (w-scaled k)^T (D x C) x deltaT (C x G) into gall-scaled S -> hop
// Panes are +4-padded (stride % 32 == 4 -> the g8/t4 fragment access pattern
// is bank-conflict-free); zero-padding past cl reproduces the partial-chunk
// semantics (garbage-free: padded rows contribute exact 0s).
// PRECISION: template PREC. 1 = plain tf32 inputs (10-bit mantissa - finer
// than the bf16 vLLM's FLA kernels use for these same products). 3 = 3xTF32
// (CUTLASS f32 emulation: a = big + small, big*big + big*small + small*big;
// ~1e-6 relative, near-f32) for the 2e-5-class parity gate. Panes hold plain
// f32; conversion happens at fragment load, so both share one body.
// Not bit-identical to the scalar stage2 (different summation grouping) -
// env-gated PADDOCK_DNC_MMA (0=off, 1=tf32, 3=3xtf32).
#define PD_DNM_KS 64u
// (tf32 cvt + pd_dnm_mma helpers now live in deltanet/split.cuh, included
// just before this segment - shared with the split walk/o-pass kernels)

// The chunk walk is RANGE-parameterized so the two-level scan reuses this
// exact body per partition: `entry` is a [gridDim.z][n_heads][D][D] array of
// partition entry states ([v][a] row-major), block z walks chunks
// [z*mchunks, min((z+1)*mchunks, nc)), and only the last partition writes the
// final state back. The direct (non-scan) call passes entry=state,
// mchunks=nc, gridDim.z=1 - identical behavior to the un-parameterized form.
template <uint32_t PREC, uint32_t G, typename ST = float>
__global__ void __launch_bounds__(256)
pd_dnc_stage2_mma_kernel(const float* __restrict__ q, const float* __restrict__ k,
                         ST* __restrict__ state, const float* __restrict__ dw,
                         const float* __restrict__ du, const double* __restrict__ cg,
                         const float* __restrict__ coef, float* __restrict__ out,
                         uint32_t n_tokens, uint32_t n_heads,
                         const ST* __restrict__ entry, uint32_t mchunks) {
    // G=32 -> 128 blocks (0.7 waves, the measured wall); G=16 -> 256 blocks
    // (1.36 waves). NT = n-tiles per block. Unlike the SCALAR G-split (which
    // lost: per-thread full-row loads replicate per block), the mma version's
    // row loads are smem-staged/coalesced and L2 sits at 13% - the extra
    // block-level staging duplication is cheap next to the wave-fill win.
    constexpr uint32_t D = PD_DNC_D, C = PD_DNC_C;
    constexpr uint32_t NT = G / 8u;
    constexpr uint32_t KS = PD_DNM_KS;
    constexpr uint32_t SD = D + 4u, SC = C + 4u, SK = KS + 4u;
    const uint32_t h = blockIdx.x, col0 = blockIdx.y * G;
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const uint32_t g8 = lane >> 2, t4 = lane & 3u;
    const uint32_t nc = (n_tokens + C - 1u) / C;
    const uint32_t part = blockIdx.z;
    const uint32_t ch_lo = part * mchunks;
    const uint32_t ch_hi = min(ch_lo + mchunks, nc);

    extern __shared__ float shm[];
    float* sh_s = shm;                    // [G][SD] resident state columns
    float* sh_a = sh_s + G * SD;          // [2C][SK] A slab; reused as wkT [D][SC]
    float* sh_dT = sh_a + 2u * C * SK;    // [G][SC] deltas, transposed (c-major)
    float* sh_cf = sh_dT + G * SC;        // [C][SC] coef rows
    __shared__ float sh_w[C], sh_gam[C];
    __shared__ float sh_gall;

    const ST* e_head = entry + ((size_t)part * n_heads + h) * D * D;
    ST* s_head = state + (size_t)h * D * D;
    for (uint32_t idx = tid; idx < G * (D / 4u); idx += 256u) {
        const uint32_t c = idx / (D / 4u), a4 = (idx % (D / 4u)) * 4u;
        *reinterpret_cast<float4*>(&sh_s[c * SD + a4]) =
            pd_dns_ld4(e_head + (size_t)(col0 + c) * D + a4);
    }
    __syncthreads();

    for (uint32_t ch = ch_lo; ch < ch_hi; ++ch) {
        const uint32_t c0 = ch * C;
        const uint32_t cl = min(C, n_tokens - c0);
        const size_t tb = (size_t)ch * n_heads + h;

        if (tid < C) {
            sh_w[tid] = tid < cl
                ? expf((float)(cg[tb * C + cl - 1u] - cg[tb * C + tid]))
                : 0.0f;
            sh_gam[tid] = tid < cl ? expf((float)cg[tb * C + tid]) : 0.0f;
        }
        if (tid == 0) sh_gall = expf((float)cg[tb * C + cl - 1u]);
        // coef pane (zeros past cl in both dims - partial-chunk rows/cols
        // contribute exact 0s downstream)
        for (uint32_t u = tid; u < C * (C / 4u); u += 256u) {
            const uint32_t r = u / (C / 4u), q4 = (u % (C / 4u)) * 4u;
            float4 v4 = make_float4(0.f, 0.f, 0.f, 0.f);
            if (r < cl) {
                v4 = *reinterpret_cast<const float4*>(coef + (tb * C + r) * C + q4);
                if (q4 + 0u >= cl) v4.x = 0.f;
                if (q4 + 1u >= cl) v4.y = 0.f;
                if (q4 + 2u >= cl) v4.z = 0.f;
                if (q4 + 3u >= cl) v4.w = 0.f;
            }
            sh_cf[r * SC + q4 + 0u] = v4.x; sh_cf[r * SC + q4 + 1u] = v4.y;
            sh_cf[r * SC + q4 + 2u] = v4.z; sh_cf[r * SC + q4 + 3u] = v4.w;
        }
        __syncthreads();

        // ---- pass 1: [dw; q] (2C x D) x S0 (D x G). Warp w owns m-tile w
        // (rows 16w..16w+16) across all 4 n-tiles; K streams in two 64-slabs.
        float acc1[NT][4];
#pragma unroll
        for (uint32_t nt = 0; nt < NT; ++nt)
#pragma unroll
            for (uint32_t e = 0; e < 4; ++e) acc1[nt][e] = 0.f;
        for (uint32_t slab = 0; slab < 2; ++slab) {
            const uint32_t k0s = slab * KS;
            for (uint32_t u = tid; u < 2u * C * (KS / 4u); u += 256u) {
                const uint32_t r = u / (KS / 4u), q4 = (u % (KS / 4u)) * 4u;
                float4 v4 = make_float4(0.f, 0.f, 0.f, 0.f);
                if (r < C) {
                    if (r < cl)
                        v4 = *reinterpret_cast<const float4*>(
                            dw + (tb * C + r) * D + k0s + q4);
                } else {
                    const uint32_t i = r - C;
                    if (i < cl)
                        v4 = *reinterpret_cast<const float4*>(
                            q + ((size_t)(c0 + i) * n_heads + h) * D + k0s + q4);
                }
                sh_a[r * SK + q4 + 0u] = v4.x; sh_a[r * SK + q4 + 1u] = v4.y;
                sh_a[r * SK + q4 + 2u] = v4.z; sh_a[r * SK + q4 + 3u] = v4.w;
            }
            __syncthreads();
#pragma unroll
            for (uint32_t kk = 0; kk < KS; kk += 8u) {
                float ar[4];
                ar[0] = sh_a[(warp * 16u + g8) * SK + kk + t4];
                ar[1] = sh_a[(warp * 16u + g8 + 8u) * SK + kk + t4];
                ar[2] = sh_a[(warp * 16u + g8) * SK + kk + t4 + 4u];
                ar[3] = sh_a[(warp * 16u + g8 + 8u) * SK + kk + t4 + 4u];
#pragma unroll
                for (uint32_t nt = 0; nt < NT; ++nt) {
                    float br[2];
                    br[0] = sh_s[(nt * 8u + g8) * SD + k0s + kk + t4];
                    br[1] = sh_s[(nt * 8u + g8) * SD + k0s + kk + t4 + 4u];
                    pd_dnm_mma<PREC>(acc1[nt], ar, br);
                }
            }
            __syncthreads();
        }

        // ---- split: warps 0..3 resolve deltas (rows 0..C) into sh_dT;
        // warps 4..7 gam-scale their o1 fragments (rows C..2C) in place.
        if (warp < 4) {
#pragma unroll
            for (uint32_t nt = 0; nt < NT; ++nt) {
#pragma unroll
                for (uint32_t e = 0; e < 4; ++e) {
                    const uint32_t i = warp * 16u + g8 + (e >= 2 ? 8u : 0u);
                    const uint32_t c = nt * 8u + 2u * t4 + (e & 1u);
                    float dlt = 0.f;
                    if (i < cl)
                        dlt = du[(tb * C + i) * D + col0 + c] - acc1[nt][e];
                    sh_dT[c * SC + i] = dlt;
                }
            }
        } else {
#pragma unroll
            for (uint32_t nt = 0; nt < NT; ++nt) {
#pragma unroll
                for (uint32_t e = 0; e < 4; ++e) {
                    const uint32_t i = (warp - 4u) * 16u + g8 + (e >= 2 ? 8u : 0u);
                    acc1[nt][e] *= sh_gam[min(i, cl - 1u)];
                }
            }
        }
        __syncthreads();

        // ---- pass 2 (warps 4..7): out = gam*o1 + coef x deltaT, m-tile
        // (warp-4); meanwhile warps 0..3 stage wkT = (w_j k_j)^T into sh_a
        // ([a][j], j-contiguous - pass 3's row-major A operand).
        if (warp >= 4) {
            const uint32_t mt = warp - 4u;
#pragma unroll
            for (uint32_t kk = 0; kk < C; kk += 8u) {
                float ar[4];
                ar[0] = sh_cf[(mt * 16u + g8) * SC + kk + t4];
                ar[1] = sh_cf[(mt * 16u + g8 + 8u) * SC + kk + t4];
                ar[2] = sh_cf[(mt * 16u + g8) * SC + kk + t4 + 4u];
                ar[3] = sh_cf[(mt * 16u + g8 + 8u) * SC + kk + t4 + 4u];
#pragma unroll
                for (uint32_t nt = 0; nt < NT; ++nt) {
                    float br[2];
                    br[0] = sh_dT[(nt * 8u + g8) * SC + kk + t4];
                    br[1] = sh_dT[(nt * 8u + g8) * SC + kk + t4 + 4u];
                    pd_dnm_mma<PREC>(acc1[nt], ar, br);
                }
            }
#pragma unroll
            for (uint32_t nt = 0; nt < NT; ++nt) {
#pragma unroll
                for (uint32_t e = 0; e < 4; ++e) {
                    const uint32_t i = mt * 16u + g8 + (e >= 2 ? 8u : 0u);
                    const uint32_t c = nt * 8u + 2u * t4 + (e & 1u);
                    if (i < cl)
                        out[((size_t)(c0 + i) * n_heads + h) * D + col0 + c] =
                            acc1[nt][e];
                }
            }
        } else {
            // 128 threads: thread pair (j, half) - coalesced k row reads,
            // strided single-float wkT writes (pad keeps banks spread)
            for (uint32_t u = tid; u < C * (D / 4u); u += 128u) {
                const uint32_t j = u / (D / 4u), a4 = (u % (D / 4u)) * 4u;
                float4 kv = make_float4(0.f, 0.f, 0.f, 0.f);
                if (j < cl)
                    kv = *reinterpret_cast<const float4*>(
                        k + ((size_t)(c0 + j) * n_heads + h) * D + a4);
                const float wj = sh_w[j];
                sh_a[(a4 + 0u) * SC + j] = wj * kv.x;
                sh_a[(a4 + 1u) * SC + j] = wj * kv.y;
                sh_a[(a4 + 2u) * SC + j] = wj * kv.z;
                sh_a[(a4 + 3u) * SC + j] = wj * kv.w;
            }
        }
        __syncthreads();

        // ---- pass 3 (all warps): S = gall*S + wkT (D x C) x deltaT (C x G).
        // Warp w owns a-rows 16w..16w+16; init/writeback regions are disjoint
        // per warp, so no barrier is needed between a warp's init reads and
        // another's writes.
        {
            float acc3[NT][4];
#pragma unroll
            for (uint32_t nt = 0; nt < NT; ++nt) {
#pragma unroll
                for (uint32_t e = 0; e < 4; ++e) {
                    const uint32_t a = warp * 16u + g8 + (e >= 2 ? 8u : 0u);
                    const uint32_t c = nt * 8u + 2u * t4 + (e & 1u);
                    acc3[nt][e] = sh_gall * sh_s[c * SD + a];
                }
            }
#pragma unroll
            for (uint32_t kk = 0; kk < C; kk += 8u) {
                float ar[4];
                ar[0] = sh_a[(warp * 16u + g8) * SC + kk + t4];
                ar[1] = sh_a[(warp * 16u + g8 + 8u) * SC + kk + t4];
                ar[2] = sh_a[(warp * 16u + g8) * SC + kk + t4 + 4u];
                ar[3] = sh_a[(warp * 16u + g8 + 8u) * SC + kk + t4 + 4u];
#pragma unroll
                for (uint32_t nt = 0; nt < NT; ++nt) {
                    float br[2];
                    br[0] = sh_dT[(nt * 8u + g8) * SC + kk + t4];
                    br[1] = sh_dT[(nt * 8u + g8) * SC + kk + t4 + 4u];
                    pd_dnm_mma<PREC>(acc3[nt], ar, br);
                }
            }
#pragma unroll
            for (uint32_t nt = 0; nt < NT; ++nt) {
#pragma unroll
                for (uint32_t e = 0; e < 4; ++e) {
                    const uint32_t a = warp * 16u + g8 + (e >= 2 ? 8u : 0u);
                    const uint32_t c = nt * 8u + 2u * t4 + (e & 1u);
                    sh_s[c * SD + a] = acc3[nt][e];
                }
            }
        }
        __syncthreads();
    }

    if (part == gridDim.z - 1u) {
        for (uint32_t idx = tid; idx < G * (D / 4u); idx += 256u) {
            const uint32_t c = idx / (D / 4u), a4 = (idx % (D / 4u)) * 4u;
            pd_dns_st4(s_head + (size_t)(col0 + c) * D + a4,
                       *reinterpret_cast<const float4*>(&sh_s[c * SD + a4]));
        }
    }
}

#define PD_DNM_SMEM(G)                                                         \
    (((G) * (PD_DNC_D + 4u) + 2u * PD_DNC_C * (PD_DNM_KS + 4u) +               \
      (G) * (PD_DNC_C + 4u) + PD_DNC_C * (PD_DNC_C + 4u)) *                    \
     4u)

// ---- v2 walk: same math, same per-element summation order as
// the v1 walk above (bit-exact by construction), but every state-independent
// input streams through a 4-slot cp.async ring prefetched 4 items ahead.
// Profiling v1 at the 2048-token span (G=32): 2.0 active warps/scheduler,
// issue 0.22/cyc, long-scoreboard = 36% of stall time - the walk is
// naked-latency-bound, not mma-bound (3xTF32 costs only +11% over plain
// tf32 end-to-end). v2 shrinks the per-chunk critical path to mma + barriers.
//
// Ring slots are 4608 floats (18 KB x4 = 74 KB; sm_120 caps dynamic smem at
// 99 KB, which killed every fatter double-buffer plan). Per chunk the item
// sequence is A0..A3 ([dw;q] pass-1 K-slabs at KS2=32), DU (this block's G-col
// du strip), CF (coef pane), K0/K1 (raw k rows, a-column halves) - consumed
// strictly in issue order, one wait_group<3> + barrier per item, re-issuing
// slot i&3 for item i+4 right after consumption. Partial-chunk tails ride the
// cp.async src-size zero-fill.
//
// Pass 3 is FLIPPED vs v1: A = deltaT (G x C, c-major = sh_dT as-is), B = the
// RAW k rows ([j][a], the natural n-major col-major operand), w_j folded at
// the fragment load (same two f32 operands, same multiply as v1's staged
// wkT). Identical k-groups per mma instruction -> identical hardware
// accumulation order -> bit-exact; and raw k needs no transpose staging, so
// v1's pass-2 wkT staging phase (warps 0-3) disappears.
#define PD_DNM2_KS 32u
#define PD_DNM2_SLOT (2u * PD_DNC_C * (PD_DNM2_KS + 4u))
#define PD_DNM2_SMEM(G)                                                        \
    (((G) * (PD_DNC_D + 4u) + (G) * (PD_DNC_C + 4u) + 4u * PD_DNM2_SLOT) * 4u)

// (pd_dnm_mma_sw + pd_dnc_cpa16 moved to deltanet/split.cuh - see the note
// at PD_DNM_KS)

// DO_OUT/DO_STATE (fla-class split, iteration 2): <true,true> is
// the classic single-kernel walk (8-item ring schedule). <false,true> is the
// WALK half on a SHRUNK 7-item schedule: A-slabs stage dw-only (C rows, pass
// 1 runs on warps 0..3 alone), the coef item is gone, pass 2 is gone, and
// each chunk stashes both its entry-state slice and its resolved deltaT
// slice. <true,false> is the parallel O half on a 5-item schedule (q-only
// A-slabs + coef): gridDim.z = nc, mchunks = 1, entry = the S-stash, deltaT
// loaded straight from the dT-stash - no dw/du staging, no delta solve, no
// pass 3, no state write. Every surviving mma sees fragments bit-identical
// to its classic twin (dw rows at pane base for the walk, q rows at pane
// base for the replay, deltaT round-tripped through fp32 exactly) ->
// bit-exact composition vs the classic walk.
template <uint32_t PREC, uint32_t G, typename ST = float, bool DO_OUT = true,
          bool DO_STATE = true, typename DWT = float>
__global__ void __launch_bounds__(256)
pd_dnc_stage2_mma_v2_kernel(const float* __restrict__ q, const float* __restrict__ k,
                            ST* __restrict__ state, const DWT* __restrict__ dw,
                            const DWT* __restrict__ du, const double* __restrict__ cg,
                            const float* __restrict__ coef, float* __restrict__ out,
                            uint32_t n_tokens, uint32_t n_heads,
                            const ST* __restrict__ entry, uint32_t mchunks,
                            float* __restrict__ stash = nullptr) {
    constexpr uint32_t D = PD_DNC_D, C = PD_DNC_C;
    constexpr uint32_t NT = G / 8u;
    constexpr uint32_t KS2 = PD_DNM2_KS;
    constexpr bool CLASSIC = DO_OUT && DO_STATE;
    // ring items per chunk: classic 8, walk 7 (no coef), replay 5 (no du/k)
    constexpr uint32_t IPC = CLASSIC ? 8u : (DO_STATE ? 7u : 5u);
    constexpr uint32_t SD = D + 4u, SC = C + 4u, SK2 = KS2 + 4u, SGP = G + 4u;
    // dw/du operand dtype (DWT): pipeline-internal buffers, so the dtype is
    // decided per call with no cross-path hazard. bf16 halves their staged
    // bytes; in the classic A-slab the dw half packs as DWT (stride SKW,
    // padded so cp.async rows stay 16B-aligned) with the f32 q half at a
    // fixed byte offset behind it. q/k/coef stay f32.
    constexpr uint32_t SKW = sizeof(DWT) == 2 ? 40u : 36u;
    constexpr uint32_t QOFF_B = PD_DNC_C * SKW * (uint32_t)sizeof(DWT);
    constexpr uint32_t SGW = sizeof(DWT) == 2 ? 40u : SGP;
    const uint32_t h = blockIdx.x, col0 = blockIdx.y * G;
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const uint32_t g8 = lane >> 2, t4 = lane & 3u;
    const uint32_t nc = (n_tokens + C - 1u) / C;
    const uint32_t part = blockIdx.z;
    const uint32_t ch_lo = part * mchunks;
    const uint32_t ch_hi = min(ch_lo + mchunks, nc);

    extern __shared__ float shm[];
    float* sh_s = shm;                    // [G][SD] resident state columns
    float* sh_dT = sh_s + G * SD;         // [G][SC] deltas, transposed (c-major)
    float* sh_ring = sh_dT + G * SC;      // 4 x PD_DNM2_SLOT item ring
    __shared__ float sh_w[C], sh_gam[C];
    __shared__ float sh_gall;

    // item ty (positional: it -> chunk it/IPC, slot it%IPC):
    //   classic: 0..3 = A-slab ([dw;q], K cols ty*32..+32), 4 = du, 5 = coef,
    //            6/7 = raw k a-halves
    //   walk:    0..3 = A-slab (dw-only), 4 = du, 5/6 = raw k a-halves
    //   replay:  0..3 = A-slab (q-only), 4 = coef
    // every thread commits a group even when the chunk is past ch_hi (empty
    // group) so wait_group counts stay aligned.
    auto issue_item = [&](uint32_t it) {
        const uint32_t ch = ch_lo + it / IPC;
        if (ch < ch_hi) {
            const uint32_t ty = it % IPC;
            const uint32_t c0i = ch * C;
            const uint32_t cli = min(C, n_tokens - c0i);
            const size_t tbi = (size_t)ch * n_heads + h;
            float* pane = sh_ring + (size_t)(it & 3u) * PD_DNM2_SLOT;
            if (ty < 4u) {
                const uint32_t k0s = ty * KS2;
                constexpr uint32_t EPW = 16u / (uint32_t)sizeof(DWT);
                constexpr uint32_t WCH = KS2 / EPW;  // dw cp.async chunks/row
                if (DO_STATE) {
                    // dw half: DWT elements at the pane base
                    DWT* wp = (DWT*)pane;
                    for (uint32_t u = tid; u < C * WCH; u += 256u) {
                        const uint32_t r = u / WCH, ce = (u % WCH) * EPW;
                        pd_dnc_cpa16(wp + r * SKW + ce,
                                     dw + (tbi * C + r) * D + k0s + ce,
                                     r < cli ? 16u : 0u);
                    }
                }
                if (CLASSIC || !DO_STATE) {
                    // q half: f32 rows behind the dw half (classic) or at the
                    // pane base (replay)
                    float* qp = CLASSIC ? (float*)((char*)pane + QOFF_B)
                                        : (float*)pane;
                    for (uint32_t u = tid; u < C * (KS2 / 4u); u += 256u) {
                        const uint32_t i = u / (KS2 / 4u), c4 = (u % (KS2 / 4u)) * 4u;
                        pd_dnc_cpa16(qp + i * SK2 + c4,
                                     q + ((size_t)(c0i + i) * n_heads + h) * D + k0s + c4,
                                     i < cli ? 16u : 0u);
                    }
                }
            } else if (DO_STATE && ty == 4u) {
                constexpr uint32_t EPW = 16u / (uint32_t)sizeof(DWT);
                DWT* up = (DWT*)pane;
                for (uint32_t u = tid; u < C * (G / EPW); u += 256u) {
                    const uint32_t i = u / (G / EPW), ce = (u % (G / EPW)) * EPW;
                    pd_dnc_cpa16(up + i * SGW + ce,
                                 du + (tbi * C + i) * D + col0 + ce,
                                 i < cli ? 16u : 0u);
                }
            } else if (CLASSIC ? (ty == 5u) : (ty == 4u)) {
                // coef pane - classic slot 5, replay slot 4; the walk never
                // stages coef (its ty 5/6 fall through to the k halves)
                for (uint32_t u = tid; u < C * (C / 4u); u += 256u) {
                    const uint32_t r = u / (C / 4u), c4 = (u % (C / 4u)) * 4u;
                    uint32_t by = 0u;
                    if (r < cli && c4 < cli) by = min(16u, (cli - c4) * 4u);
                    pd_dnc_cpa16(pane + r * SC + c4, coef + (tbi * C + r) * C + c4, by);
                }
            } else {
                const uint32_t a0 = (ty - (CLASSIC ? 6u : 5u)) * 64u;
                for (uint32_t u = tid; u < C * 16u; u += 256u) {
                    const uint32_t j = u / 16u, c4 = (u % 16u) * 4u;
                    pd_dnc_cpa16(pane + j * SC + c4,
                                 k + ((size_t)(c0i + j) * n_heads + h) * D + a0 + c4,
                                 j < cli ? 16u : 0u);
                }
            }
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    };
    // item `it` is staged: 4 groups in flight -> oldest (ours) must retire
    auto ring_wait = [&] {
        asm volatile("cp.async.wait_group 3;" ::: "memory");
        __syncthreads();
    };

    const ST* e_head = entry + ((size_t)part * n_heads + h) * D * D;
    ST* s_head = state + (size_t)h * D * D;
    for (uint32_t idx = tid; idx < G * (D / 4u); idx += 256u) {
        const uint32_t c = idx / (D / 4u), a4 = (idx % (D / 4u)) * 4u;
        *reinterpret_cast<float4*>(&sh_s[c * SD + a4]) =
            pd_dns_ld4(e_head + (size_t)(col0 + c) * D + a4);
    }
    issue_item(0); issue_item(1); issue_item(2); issue_item(3);
    __syncthreads();

    for (uint32_t ch = ch_lo; ch < ch_hi; ++ch) {
        const uint32_t c0 = ch * C;
        const uint32_t cl = min(C, n_tokens - c0);
        const size_t tb = (size_t)ch * n_heads + h;
        const uint32_t it0 = IPC * (ch - ch_lo);

        // decay vectors - issued early, first consumed at the split (sh_gam)
        // with several barriers in between; sh_w/gam single-buffered is safe
        // because chunk c's last read precedes the pass3-end barrier and
        // chunk c+1 writes after it.
        if (tid < C) {
            sh_w[tid] = tid < cl
                ? expf((float)(cg[tb * C + cl - 1u] - cg[tb * C + tid]))
                : 0.0f;
            sh_gam[tid] = tid < cl ? expf((float)cg[tb * C + tid]) : 0.0f;
        }
        if (tid == 0) sh_gall = expf((float)cg[tb * C + cl - 1u]);
        if (DO_STATE && stash != nullptr) {
            // fla split: stage3o replays this chunk from its ENTRY state
            float* dst = stash + ((size_t)ch * n_heads + h) * (size_t)(D * D);
            for (uint32_t idx = tid; idx < G * (D / 4u); idx += 256u) {
                const uint32_t c = idx / (D / 4u), a4 = (idx % (D / 4u)) * 4u;
                *reinterpret_cast<float4*>(&dst[(size_t)(col0 + c) * D + a4]) =
                    *reinterpret_cast<const float4*>(&sh_s[c * SD + a4]);
            }
        }

        // ---- pass 1: [dw; q] (2C x D) x S0 (D x G), K streamed in 4 ring slabs
        float acc1[NT][4];
#pragma unroll
        for (uint32_t nt = 0; nt < NT; ++nt)
#pragma unroll
            for (uint32_t e = 0; e < 4; ++e) acc1[nt][e] = 0.f;
        // pass-1 owners: classic = all 8 warps over [dw;q]; walk = warps 0..3
        // over the dw-only pane; replay = warps 4..7 over the q-only pane.
        // Both split panes put their rows at the pane BASE, so the surviving
        // warps read the same bits (hence same fragments) as their classic
        // twins.
        const bool p1w = CLASSIC || (DO_STATE ? (warp < 4u) : (warp >= 4u));
        const uint32_t p1r = (DO_STATE ? warp : warp - 4u) * 16u;
        for (uint32_t slab = 0; slab < 4; ++slab) {
            const uint32_t k0s = slab * KS2;
            ring_wait();
            const float* pane = sh_ring + (size_t)((it0 + slab) & 3u) * PD_DNM2_SLOT;
            // dw rows read DWT at the pane base; q rows read f32 behind them
            // (classic warps 4-7) or at the base (replay). p1r for the q half
            // is the in-half row (p1r - C never occurs: DO_STATE warps use
            // p1r < C for dw, and the q owners recompute their base below).
            const DWT* wf = (const DWT*)pane;
            const float* qf = CLASSIC ? (const float*)((const char*)pane + QOFF_B)
                                      : pane;
            const bool dwrow = DO_STATE && warp < 4u;
            const uint32_t qr = CLASSIC ? (warp - 4u) * 16u : p1r;
            if (p1w) {
#pragma unroll
                for (uint32_t kk = 0; kk < KS2; kk += 8u) {
                    float ar[4];
                    if (dwrow) {
                        ar[0] = (float)wf[(p1r + g8) * SKW + kk + t4];
                        ar[1] = (float)wf[(p1r + g8 + 8u) * SKW + kk + t4];
                        ar[2] = (float)wf[(p1r + g8) * SKW + kk + t4 + 4u];
                        ar[3] = (float)wf[(p1r + g8 + 8u) * SKW + kk + t4 + 4u];
                    } else {
                        ar[0] = qf[(qr + g8) * SK2 + kk + t4];
                        ar[1] = qf[(qr + g8 + 8u) * SK2 + kk + t4];
                        ar[2] = qf[(qr + g8) * SK2 + kk + t4 + 4u];
                        ar[3] = qf[(qr + g8 + 8u) * SK2 + kk + t4 + 4u];
                    }
#pragma unroll
                    for (uint32_t nt = 0; nt < NT; ++nt) {
                        float br[2];
                        br[0] = sh_s[(nt * 8u + g8) * SD + k0s + kk + t4];
                        br[1] = sh_s[(nt * 8u + g8) * SD + k0s + kk + t4 + 4u];
                        pd_dnm_mma<PREC>(acc1[nt], ar, br);
                    }
                }
            }
            __syncthreads();
            issue_item(it0 + slab + 4u);
        }

        // ---- split. classic/walk: warps 0..3 resolve deltas (du from the
        // ring strip) into sh_dT - and the walk stashes the resolved slice
        // for the replay. classic warps 4..7 gam-scale their o1 fragments in
        // place. replay: no du item at all - deltaT comes straight from the
        // walk's stash (exact fp32 round-trip), warps 4..7 gam-scale.
        if (!DO_STATE) {
            const float* dsrc = stash + (size_t)nc * n_heads * (D * D) +
                                ((size_t)ch * n_heads + h) * (size_t)(C * D) +
                                (size_t)col0 * C;
            for (uint32_t idx = tid; idx < G * (C / 4u); idx += 256u) {
                const uint32_t c = idx / (C / 4u), i4 = (idx % (C / 4u)) * 4u;
                *reinterpret_cast<float4*>(&sh_dT[c * SC + i4]) =
                    *reinterpret_cast<const float4*>(&dsrc[(size_t)c * C + i4]);
            }
            if (warp >= 4) {
#pragma unroll
                for (uint32_t nt = 0; nt < NT; ++nt) {
#pragma unroll
                    for (uint32_t e = 0; e < 4; ++e) {
                        const uint32_t i = (warp - 4u) * 16u + g8 + (e >= 2 ? 8u : 0u);
                        acc1[nt][e] *= sh_gam[min(i, cl - 1u)];
                    }
                }
            }
            __syncthreads();
        } else {
            ring_wait();  // du item (it0+4)
            const DWT* pane = (const DWT*)(sh_ring +
                                           (size_t)((it0 + 4u) & 3u) * PD_DNM2_SLOT);
            if (warp < 4) {
#pragma unroll
                for (uint32_t nt = 0; nt < NT; ++nt) {
#pragma unroll
                    for (uint32_t e = 0; e < 4; ++e) {
                        const uint32_t i = warp * 16u + g8 + (e >= 2 ? 8u : 0u);
                        const uint32_t c = nt * 8u + 2u * t4 + (e & 1u);
                        float dlt = 0.f;
                        if (i < cl) dlt = (float)pane[i * SGW + c] - acc1[nt][e];
                        sh_dT[c * SC + i] = dlt;
                    }
                }
            } else if (DO_OUT) {
#pragma unroll
                for (uint32_t nt = 0; nt < NT; ++nt) {
#pragma unroll
                    for (uint32_t e = 0; e < 4; ++e) {
                        const uint32_t i = (warp - 4u) * 16u + g8 + (e >= 2 ? 8u : 0u);
                        acc1[nt][e] *= sh_gam[min(i, cl - 1u)];
                    }
                }
            }
            __syncthreads();
            issue_item(it0 + 8u);
            if (!DO_OUT && stash != nullptr) {
                // walk: stash this chunk's resolved deltaT slice (the barrier
                // above makes warps 0..3's sh_dT writes visible to all)
                float* dst = stash + (size_t)nc * n_heads * (D * D) +
                             ((size_t)ch * n_heads + h) * (size_t)(C * D) +
                             (size_t)col0 * C;
                for (uint32_t idx = tid; idx < G * (C / 4u); idx += 256u) {
                    const uint32_t c = idx / (C / 4u), i4 = (idx % (C / 4u)) * 4u;
                    *reinterpret_cast<float4*>(&dst[(size_t)c * C + i4]) =
                        *reinterpret_cast<const float4*>(&sh_dT[c * SC + i4]);
                }
            }
        }

        // ---- pass 2 (warps 4..7): out = gam*o1 + coef x deltaT. Warps 0..3
        // idle here - v1's wkT staging job is gone (pass 3 eats raw k).
        // The walk has no coef item and no pass 2 at all.
        if (DO_OUT) {
            const uint32_t cfit = CLASSIC ? it0 + 5u : it0 + 4u;  // coef slot
            ring_wait();
            const float* pane = sh_ring + (size_t)(cfit & 3u) * PD_DNM2_SLOT;
            if (warp >= 4) {
                const uint32_t mt = warp - 4u;
#pragma unroll
                for (uint32_t kk = 0; kk < C; kk += 8u) {
                    float ar[4];
                    ar[0] = pane[(mt * 16u + g8) * SC + kk + t4];
                    ar[1] = pane[(mt * 16u + g8 + 8u) * SC + kk + t4];
                    ar[2] = pane[(mt * 16u + g8) * SC + kk + t4 + 4u];
                    ar[3] = pane[(mt * 16u + g8 + 8u) * SC + kk + t4 + 4u];
#pragma unroll
                    for (uint32_t nt = 0; nt < NT; ++nt) {
                        float br[2];
                        br[0] = sh_dT[(nt * 8u + g8) * SC + kk + t4];
                        br[1] = sh_dT[(nt * 8u + g8) * SC + kk + t4 + 4u];
                        pd_dnm_mma<PREC>(acc1[nt], ar, br);
                    }
                }
#pragma unroll
                for (uint32_t nt = 0; nt < NT; ++nt) {
#pragma unroll
                    for (uint32_t e = 0; e < 4; ++e) {
                        const uint32_t i = mt * 16u + g8 + (e >= 2 ? 8u : 0u);
                        const uint32_t c = nt * 8u + 2u * t4 + (e & 1u);
                        if (i < cl)
                            out[((size_t)(c0 + i) * n_heads + h) * D + col0 + c] =
                                acc1[nt][e];
                    }
                }
            }
            __syncthreads();
            issue_item(cfit + 4u);
        }

        // ---- pass 3, FLIPPED: S(c-rows) = gall*S + deltaT (G x C) x wk
        // (C x D, raw k pane x w_j at the B-fragment load), one 64-col a-half
        // per ring item. Init reads and writebacks are per-warp disjoint.
        // The replay has no k items and no pass 3 (its schedule ends at coef).
        if (DO_STATE) for (uint32_t half = 0; half < 2; ++half) {
            const uint32_t kit = CLASSIC ? it0 + 6u + half : it0 + 5u + half;
            ring_wait();  // k half item
            const float* pane =
                sh_ring + (size_t)(kit & 3u) * PD_DNM2_SLOT;
            // warp -> (m-tile, n-tile pair) over 2x8 (G=32) or 1x8 (G=16) tiles
            const uint32_t mt = (G == 32u) ? (warp & 1u) : 0u;
            const uint32_t nt0 = (G == 32u) ? (warp >> 1) * 2u : warp;
            const uint32_t ntn = (G == 32u) ? 2u : 1u;
            float acc3[2][4];
#pragma unroll
            for (uint32_t nn = 0; nn < 2; ++nn) {
                if (nn >= ntn) break;
#pragma unroll
                for (uint32_t e = 0; e < 4; ++e) {
                    const uint32_t c = mt * 16u + g8 + (e >= 2 ? 8u : 0u);
                    const uint32_t a = half * 64u + (nt0 + nn) * 8u + 2u * t4 + (e & 1u);
                    acc3[nn][e] = sh_gall * sh_s[c * SD + a];
                }
            }
#pragma unroll
            for (uint32_t kk = 0; kk < C; kk += 8u) {
                float ar[4];
                ar[0] = sh_dT[(mt * 16u + g8) * SC + kk + t4];
                ar[1] = sh_dT[(mt * 16u + g8 + 8u) * SC + kk + t4];
                ar[2] = sh_dT[(mt * 16u + g8) * SC + kk + t4 + 4u];
                ar[3] = sh_dT[(mt * 16u + g8 + 8u) * SC + kk + t4 + 4u];
                const float w0 = sh_w[kk + t4], w4 = sh_w[kk + t4 + 4u];
#pragma unroll
                for (uint32_t nn = 0; nn < 2; ++nn) {
                    if (nn >= ntn) break;
                    float br[2];
                    br[0] = w0 * pane[(kk + t4) * SC + (nt0 + nn) * 8u + g8];
                    br[1] = w4 * pane[(kk + t4 + 4u) * SC + (nt0 + nn) * 8u + g8];
                    pd_dnm_mma_sw<PREC>(acc3[nn], ar, br);
                }
            }
#pragma unroll
            for (uint32_t nn = 0; nn < 2; ++nn) {
                if (nn >= ntn) break;
#pragma unroll
                for (uint32_t e = 0; e < 4; ++e) {
                    const uint32_t c = mt * 16u + g8 + (e >= 2 ? 8u : 0u);
                    const uint32_t a = half * 64u + (nt0 + nn) * 8u + 2u * t4 + (e & 1u);
                    sh_s[c * SD + a] = acc3[nn][e];
                }
            }
            __syncthreads();
            issue_item(kit + 4u);
        }
    }

    if (DO_STATE && part == gridDim.z - 1u) {
        for (uint32_t idx = tid; idx < G * (D / 4u); idx += 256u) {
            const uint32_t c = idx / (D / 4u), a4 = (idx % (D / 4u)) * 4u;
            pd_dns_st4(s_head + (size_t)(col0 + c) * D + a4,
                       *reinterpret_cast<const float4*>(&sh_s[c * SD + a4]));
        }
    }
}

// ---- TWO-LEVEL SCAN: sequence parallelism for the stage-2
// chunk walk. The inter-chunk recurrence is an AFFINE map on the state:
//     S' = S · A_c + B_c,   A_c = gall_c·I - dw_cᵀ·wk_c,  B_c = du_cᵀ·wk_c
// (wk = decay-weighted k rows; derivation: substitute delta = du - dw·S into
// the hop). Affine maps compose associatively, so the serial nc-chunk chain
// (the measured wall: 128 blocks, 0.7 waves, every block walking every
// chunk) becomes:
//   A1: materialize Mᵀ_c = (dwᵀwk)ᵀ and B_c per chunk       (nc×H blocks)
//   A2: per partition p, fold T̄=[Ā;B̄] over its m chunks     (P×H×4 blocks)
//   B : compose the P transfers sequentially -> entry states (tiny)
//   C : the EXISTING range-parameterized walk, one partition
//       per grid-z, from its true entry state               (H×(D/G)×P blocks)
// ~2.2x the FLOP of the direct walk for P-way chain shortening + a wave-
// filling grid. Same tf32/3xTF32 numeric class as the mma walk (compose
// order differs from the sequential walk - the 2e-5 CPU-oracle parity gate
// judges). Scratch (M/B/transfer/entry stash) is stream-ordered
// cudaMallocAsync in the launcher.

// A1: per (chunk, head) build Mst[a][b] = Σ_j dw[j][b]·wk[j][a]  (M stored
// TRANSPOSED = a-major, exactly the col-major B-operand A2 wants) and
// Bst[v][a] = Σ_j du[j][v]·wk[j][a] (row-major, A2's additive inject +
// kernel B's layout). Two sub-phases share the wkT pane; dwT is overwritten
// by duT between them. Zero-padding past cl keeps partial chunks exact.
template <uint32_t PREC>
__global__ void __launch_bounds__(256)
pd_dnc_scan_a1_kernel(const float* __restrict__ k, const float* __restrict__ dw,
                      const float* __restrict__ du, const double* __restrict__ cg,
                      float* __restrict__ mst, float* __restrict__ bst,
                      uint32_t n_tokens, uint32_t n_heads) {
    constexpr uint32_t D = PD_DNC_D, C = PD_DNC_C, SC = C + 4u;
    const uint32_t ch = blockIdx.x, h = blockIdx.y;
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const uint32_t g8 = lane >> 2, t4 = lane & 3u;
    const uint32_t c0 = ch * C;
    const uint32_t cl = min(C, n_tokens - c0);
    const size_t tb = (size_t)ch * n_heads + h;

    extern __shared__ float shm[];
    float* sh_x = shm;             // [D][SC]: dwT, then duT
    float* sh_wk = shm + D * SC;   // [D][SC]: wkT (w_j folded)
    __shared__ float sh_w[C];

    if (tid < C)
        sh_w[tid] = tid < cl
            ? expf((float)(cg[tb * C + cl - 1u] - cg[tb * C + tid]))
            : 0.0f;
    __syncthreads();
    // stage dwT[b][j] and wkT[a][j] (coalesced row reads, strided writes)
    for (uint32_t u = tid; u < C * (D / 4u); u += 256u) {
        const uint32_t j = u / (D / 4u), a4 = (u % (D / 4u)) * 4u;
        float4 dv = make_float4(0.f, 0.f, 0.f, 0.f);
        float4 kv = make_float4(0.f, 0.f, 0.f, 0.f);
        if (j < cl) {
            dv = *reinterpret_cast<const float4*>(dw + (tb * C + j) * D + a4);
            kv = *reinterpret_cast<const float4*>(
                k + ((size_t)(c0 + j) * n_heads + h) * D + a4);
        }
        const float wj = sh_w[j];
        sh_x[(a4 + 0u) * SC + j] = dv.x; sh_x[(a4 + 1u) * SC + j] = dv.y;
        sh_x[(a4 + 2u) * SC + j] = dv.z; sh_x[(a4 + 3u) * SC + j] = dv.w;
        sh_wk[(a4 + 0u) * SC + j] = wj * kv.x; sh_wk[(a4 + 1u) * SC + j] = wj * kv.y;
        sh_wk[(a4 + 2u) * SC + j] = wj * kv.z; sh_wk[(a4 + 3u) * SC + j] = wj * kv.w;
    }
    __syncthreads();
    // Mst = wkT-rows x dwT-cols: rows a (8 m-tiles, warp-owned), cols b
    // (16 n-tiles in two halves of 8 -> 32 acc regs), K = j (8 steps)
    for (uint32_t half = 0; half < 2; ++half) {
        float acc[8][4];
#pragma unroll
        for (uint32_t nt = 0; nt < 8; ++nt)
#pragma unroll
            for (uint32_t e = 0; e < 4; ++e) acc[nt][e] = 0.f;
#pragma unroll
        for (uint32_t kk = 0; kk < C; kk += 8u) {
            float ar[4];
            ar[0] = sh_wk[(warp * 16u + g8) * SC + kk + t4];
            ar[1] = sh_wk[(warp * 16u + g8 + 8u) * SC + kk + t4];
            ar[2] = sh_wk[(warp * 16u + g8) * SC + kk + t4 + 4u];
            ar[3] = sh_wk[(warp * 16u + g8 + 8u) * SC + kk + t4 + 4u];
#pragma unroll
            for (uint32_t nt = 0; nt < 8; ++nt) {
                float br[2];
                br[0] = sh_x[(half * 64u + nt * 8u + g8) * SC + kk + t4];
                br[1] = sh_x[(half * 64u + nt * 8u + g8) * SC + kk + t4 + 4u];
                pd_dnm_mma<PREC>(acc[nt], ar, br);
            }
        }
#pragma unroll
        for (uint32_t nt = 0; nt < 8; ++nt)
#pragma unroll
            for (uint32_t e = 0; e < 4; ++e) {
                const uint32_t a = warp * 16u + g8 + (e >= 2 ? 8u : 0u);
                const uint32_t b = half * 64u + nt * 8u + 2u * t4 + (e & 1u);
                mst[tb * (size_t)(D * D) + (size_t)a * D + b] = acc[nt][e];
            }
    }
    __syncthreads();
    // duT overwrites the dwT pane
    for (uint32_t u = tid; u < C * (D / 4u); u += 256u) {
        const uint32_t j = u / (D / 4u), a4 = (u % (D / 4u)) * 4u;
        float4 dv = make_float4(0.f, 0.f, 0.f, 0.f);
        if (j < cl)
            dv = *reinterpret_cast<const float4*>(du + (tb * C + j) * D + a4);
        sh_x[(a4 + 0u) * SC + j] = dv.x; sh_x[(a4 + 1u) * SC + j] = dv.y;
        sh_x[(a4 + 2u) * SC + j] = dv.z; sh_x[(a4 + 3u) * SC + j] = dv.w;
    }
    __syncthreads();
    // Bst[v][a] = duT-rows x wkT-cols
    for (uint32_t half = 0; half < 2; ++half) {
        float acc[8][4];
#pragma unroll
        for (uint32_t nt = 0; nt < 8; ++nt)
#pragma unroll
            for (uint32_t e = 0; e < 4; ++e) acc[nt][e] = 0.f;
#pragma unroll
        for (uint32_t kk = 0; kk < C; kk += 8u) {
            float ar[4];
            ar[0] = sh_x[(warp * 16u + g8) * SC + kk + t4];
            ar[1] = sh_x[(warp * 16u + g8 + 8u) * SC + kk + t4];
            ar[2] = sh_x[(warp * 16u + g8) * SC + kk + t4 + 4u];
            ar[3] = sh_x[(warp * 16u + g8 + 8u) * SC + kk + t4 + 4u];
#pragma unroll
            for (uint32_t nt = 0; nt < 8; ++nt) {
                float br[2];
                br[0] = sh_wk[(half * 64u + nt * 8u + g8) * SC + kk + t4];
                br[1] = sh_wk[(half * 64u + nt * 8u + g8) * SC + kk + t4 + 4u];
                pd_dnm_mma<PREC>(acc[nt], ar, br);
            }
        }
#pragma unroll
        for (uint32_t nt = 0; nt < 8; ++nt)
#pragma unroll
            for (uint32_t e = 0; e < 4; ++e) {
                const uint32_t v = warp * 16u + g8 + (e >= 2 ? 8u : 0u);
                const uint32_t a = half * 64u + nt * 8u + 2u * t4 + (e & 1u);
                bst[tb * (size_t)(D * D) + (size_t)v * D + a] = acc[nt][e];
            }
    }
}

// A2: fold the partition transfer. T̄ = [Ā; B̄] (256 rows x D cols) split in
// 4 row-bands of 64 (grid z); per chunk: T̄ <- gall·T̄ - T̄·M (+ B_c into the
// B̄ rows). Double-buffered band panes (the full T̄ row is read for every
// output column, so in-place update is impossible). M streamed from global
// as B-operand fragments (Mst is a-major = col-major-compatible; each element
// is read once per band-block - L2-resident).
template <uint32_t PREC>
__global__ void __launch_bounds__(256)
pd_dnc_scan_a2_kernel(const double* __restrict__ cg, const float* __restrict__ mst,
                      const float* __restrict__ bst, float* __restrict__ att,
                      float* __restrict__ bbar, uint32_t n_tokens, uint32_t n_heads,
                      uint32_t mchunks) {
    constexpr uint32_t D = PD_DNC_D, C = PD_DNC_C, RB = 64u, SD = D + 4u;
    const uint32_t part = blockIdx.x, h = blockIdx.y, r0 = blockIdx.z * RB;
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const uint32_t g8 = lane >> 2, t4 = lane & 3u;
    const uint32_t nc = (n_tokens + C - 1u) / C;
    const uint32_t ch_lo = part * mchunks, ch_hi = min(ch_lo + mchunks, nc);

    extern __shared__ float shm[];
    float* pane[2] = {shm, shm + RB * SD};
    __shared__ float sh_gall;

    // init: Ā rows (r0 < 2C? rows 0..127) = I; B̄ rows (128..255) = 0
    for (uint32_t u = tid; u < RB * D; u += 256u) {
        const uint32_t r = u / D, a = u % D;
        pane[0][r * SD + a] = (r0 + r == a) ? 1.0f : 0.0f;  // B̄ bands: r0+r >= 128 != a<128 -> 0
    }
    __syncthreads();

    uint32_t cur = 0;
    for (uint32_t ch = ch_lo; ch < ch_hi; ++ch) {
        const uint32_t cl = min(C, n_tokens - ch * C);
        const size_t tb = (size_t)ch * n_heads + h;
        if (tid == 0) sh_gall = expf((float)cg[tb * C + cl - 1u]);
        __syncthreads();
        const float gall = sh_gall;
        const float* m_c = mst + tb * (size_t)(D * D);
        // 4 m-tiles x 16 n-tiles; warp w: m-tile w>>1, n-half w&1 (8 tiles)
        const uint32_t mt = warp >> 1, nh = (warp & 1u) * 8u;
        float acc[8][4];
#pragma unroll
        for (uint32_t nt = 0; nt < 8; ++nt)
#pragma unroll
            for (uint32_t e = 0; e < 4; ++e) acc[nt][e] = 0.f;
#pragma unroll
        for (uint32_t kk = 0; kk < D; kk += 8u) {
            float ar[4];
            ar[0] = pane[cur][(mt * 16u + g8) * SD + kk + t4];
            ar[1] = pane[cur][(mt * 16u + g8 + 8u) * SD + kk + t4];
            ar[2] = pane[cur][(mt * 16u + g8) * SD + kk + t4 + 4u];
            ar[3] = pane[cur][(mt * 16u + g8 + 8u) * SD + kk + t4 + 4u];
#pragma unroll
            for (uint32_t nt = 0; nt < 8; ++nt) {
                float br[2];
                br[0] = m_c[(size_t)((nh + nt) * 8u + g8) * D + kk + t4];
                br[1] = m_c[(size_t)((nh + nt) * 8u + g8) * D + kk + t4 + 4u];
                pd_dnm_mma<PREC>(acc[nt], ar, br);
            }
        }
        const float* b_c = bst + tb * (size_t)(D * D);
#pragma unroll
        for (uint32_t nt = 0; nt < 8; ++nt)
#pragma unroll
            for (uint32_t e = 0; e < 4; ++e) {
                const uint32_t r = mt * 16u + g8 + (e >= 2 ? 8u : 0u);
                const uint32_t a = (nh + nt) * 8u + 2u * t4 + (e & 1u);
                float v = gall * pane[cur][r * SD + a] - acc[nt][e];
                if (r0 + r >= D)  // B̄ rows: T̄ = [Ā (D rows); B̄ (D rows)]
                    v += b_c[(size_t)(r0 + r - D) * D + a];
                pane[1u - cur][r * SD + a] = v;
            }
        __syncthreads();
        cur = 1u - cur;
    }
    // store: Ā TRANSPOSED to att[p][h][a][b] (kernel B's col-major operand);
    // B̄ row-major to bbar[p][h][v][a]
    const size_t base = ((size_t)part * n_heads + h) * (size_t)(D * D);
    for (uint32_t u = tid; u < RB * D; u += 256u) {
        const uint32_t r = u / D, a = u % D;
        const float v = pane[cur][r * SD + a];
        if (r0 + r < D)
            att[base + (size_t)a * D + (r0 + r)] = v;  // Ā transposed -> [a][b]
        else
            bbar[base + (size_t)(r0 + r - D) * D + a] = v;  // B̄ row-major [v][a]
    }
}

// B: sequential compose of the P partition transfers. Entry states:
// stash[0] = incoming state; stash[p+1] = stash[p]·Ā_p + B̄_p. Row-band
// blocks (grid (H, 2)), double-buffered like A2; Ā read straight from
// global (att is a-major = the col-major B operand).
template <uint32_t PREC>
__global__ void __launch_bounds__(256)
pd_dnc_scan_b_kernel(const float* __restrict__ state, const float* __restrict__ att,
                     const float* __restrict__ bbar, float* __restrict__ stash,
                     uint32_t n_heads, uint32_t nparts) {
    constexpr uint32_t D = PD_DNC_D, RB = 64u, SD = D + 4u;
    const uint32_t h = blockIdx.x, v0 = blockIdx.y * RB;
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const uint32_t g8 = lane >> 2, t4 = lane & 3u;

    extern __shared__ float shm[];
    float* pane[2] = {shm, shm + RB * SD};

    const float* s_head = state + (size_t)h * D * D;
    for (uint32_t u = tid; u < RB * D; u += 256u) {
        const uint32_t r = u / D, a = u % D;
        pane[0][r * SD + a] = s_head[(size_t)(v0 + r) * D + a];
    }
    __syncthreads();
    uint32_t cur = 0;
    for (uint32_t p = 0; p < nparts; ++p) {
        // stash the ENTRY state of partition p
        float* dst = stash + ((size_t)p * n_heads + h) * (size_t)(D * D);
        for (uint32_t u = tid; u < RB * D; u += 256u) {
            const uint32_t r = u / D, a = u % D;
            dst[(size_t)(v0 + r) * D + a] = pane[cur][r * SD + a];
        }
        if (p + 1u == nparts) break;
        const size_t base = ((size_t)p * n_heads + h) * (size_t)(D * D);
        const float* a_p = att + base;
        const float* b_p = bbar + base;
        const uint32_t mt = warp >> 1, nh = (warp & 1u) * 8u;
        float acc[8][4];
#pragma unroll
        for (uint32_t nt = 0; nt < 8; ++nt)
#pragma unroll
            for (uint32_t e = 0; e < 4; ++e) acc[nt][e] = 0.f;
#pragma unroll
        for (uint32_t kk = 0; kk < D; kk += 8u) {
            float ar[4];
            ar[0] = pane[cur][(mt * 16u + g8) * SD + kk + t4];
            ar[1] = pane[cur][(mt * 16u + g8 + 8u) * SD + kk + t4];
            ar[2] = pane[cur][(mt * 16u + g8) * SD + kk + t4 + 4u];
            ar[3] = pane[cur][(mt * 16u + g8 + 8u) * SD + kk + t4 + 4u];
#pragma unroll
            for (uint32_t nt = 0; nt < 8; ++nt) {
                float br[2];
                br[0] = a_p[(size_t)((nh + nt) * 8u + g8) * D + kk + t4];
                br[1] = a_p[(size_t)((nh + nt) * 8u + g8) * D + kk + t4 + 4u];
                pd_dnm_mma<PREC>(acc[nt], ar, br);
            }
        }
        __syncthreads();  // pane[cur] fully read by every warp before overwrite of the other pane is irrelevant; this fences the stash reads above
#pragma unroll
        for (uint32_t nt = 0; nt < 8; ++nt)
#pragma unroll
            for (uint32_t e = 0; e < 4; ++e) {
                const uint32_t r = mt * 16u + g8 + (e >= 2 ? 8u : 0u);
                const uint32_t a = (nh + nt) * 8u + 2u * t4 + (e & 1u);
                pane[1u - cur][r * SD + a] =
                    acc[nt][e] + b_p[(size_t)(v0 + r) * D + a];
            }
        __syncthreads();
        cur = 1u - cur;
    }
}

// stage1 RS rebuild election (PADDOCK_DNC_S1RS): shared by the
// per-span RS route and the vl entry. Class change - see the route comment.
static inline bool pd_dnc_s1rs_on() {
    static const bool on = [] {
        const char* e = pd_env("PADDOCK_DNC_S1RS");
        return e && atoi(e) != 0;
    }();
    return on;
}

static int pd_gated_delta_chunked_go(bool vb16, const void* q, const void* k,
                                     const void* v, const void* g,
                           const void* beta, void* state, void* out, void* dw, void* du,
                           void* aqk, void* cg, uint32_t n_tokens, uint32_t n_heads,
                           uint32_t head_dim, void* stream) {
    if (n_tokens == 0 || n_heads == 0) return 0;
    if (head_dim != PD_DNC_D) return cudaErrorInvalidValue;
    const uint32_t nc = (n_tokens + PD_DNC_C - 1u) / PD_DNC_C;
    static bool attr_done = false;
    if (!attr_done) {
        cudaFuncSetAttribute((const void*)pd_dnc_stage1_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, PD_DNC_S1_SMEM);
        pd_prefer_max_shared(pd_dnc_stage1_kernel);
        pd_prefer_max_shared(pd_dnc_stage2_kernel<PD_DNC_G>);
        pd_prefer_max_shared(pd_dnc_stage2_kernel<16u>);
        pd_prefer_max_shared(pd_dnc_stage2_kernel<8u>);
        cudaFuncSetAttribute((const void*)pd_dnc_stage2_mma_kernel<1u, 32u>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, PD_DNM_SMEM(32u));
        cudaFuncSetAttribute((const void*)pd_dnc_stage2_mma_kernel<3u, 32u>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, PD_DNM_SMEM(32u));
        cudaFuncSetAttribute((const void*)pd_dnc_stage2_mma_kernel<1u, 16u>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, PD_DNM_SMEM(16u));
        cudaFuncSetAttribute((const void*)pd_dnc_stage2_mma_kernel<3u, 16u>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, PD_DNM_SMEM(16u));
        cudaFuncSetAttribute((const void*)pd_dnc_stage2_mma_v2_kernel<1u, 32u>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, PD_DNM2_SMEM(32u));
        cudaFuncSetAttribute((const void*)pd_dnc_stage2_mma_v2_kernel<3u, 32u>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, PD_DNM2_SMEM(32u));
        cudaFuncSetAttribute((const void*)pd_dnc_stage2_mma_v2_kernel<1u, 16u>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, PD_DNM2_SMEM(16u));
        cudaFuncSetAttribute((const void*)pd_dnc_stage2_mma_v2_kernel<3u, 16u>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, PD_DNM2_SMEM(16u));
        cudaFuncSetAttribute(
            (const void*)pd_dnc_stage2_mma_kernel<3u, 32u, __nv_bfloat16>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, PD_DNM_SMEM(32u));
        cudaFuncSetAttribute(
            (const void*)pd_dnc_stage2_mma_kernel<1u, 32u, __nv_bfloat16>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, PD_DNM_SMEM(32u));
        cudaFuncSetAttribute(
            (const void*)pd_dnc_stage2_mma_kernel<3u, 16u, __nv_bfloat16>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, PD_DNM_SMEM(16u));
        cudaFuncSetAttribute(
            (const void*)pd_dnc_stage2_mma_kernel<1u, 16u, __nv_bfloat16>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, PD_DNM_SMEM(16u));
        cudaFuncSetAttribute(
            (const void*)pd_dnc_stage2_mma_v2_kernel<3u, 32u, __nv_bfloat16>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, PD_DNM2_SMEM(32u));
        cudaFuncSetAttribute(
            (const void*)pd_dnc_stage2_mma_v2_kernel<1u, 32u, __nv_bfloat16>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, PD_DNM2_SMEM(32u));
        cudaFuncSetAttribute(
            (const void*)pd_dnc_stage2_mma_v2_kernel<3u, 16u, __nv_bfloat16>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, PD_DNM2_SMEM(16u));
        cudaFuncSetAttribute(
            (const void*)pd_dnc_stage2_mma_v2_kernel<1u, 16u, __nv_bfloat16>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, PD_DNM2_SMEM(16u));
        // scan kernels: 67-70 KB dynamic smem, over the 48 KB default
        const uint32_t a1sz = 2u * PD_DNC_D * (PD_DNC_C + 4u) * 4u;
        const uint32_t a2sz = 2u * 64u * (PD_DNC_D + 4u) * 4u;
        cudaFuncSetAttribute((const void*)pd_dnc_scan_a1_kernel<1u>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, a1sz);
        cudaFuncSetAttribute((const void*)pd_dnc_scan_a1_kernel<3u>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, a1sz);
        cudaFuncSetAttribute((const void*)pd_dnc_scan_a2_kernel<1u>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, a2sz);
        cudaFuncSetAttribute((const void*)pd_dnc_scan_a2_kernel<3u>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, a2sz);
        cudaFuncSetAttribute((const void*)pd_dnc_scan_b_kernel<1u>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, a2sz);
        cudaFuncSetAttribute((const void*)pd_dnc_scan_b_kernel<3u>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, a2sz);
        attr_done = true;
    }
    const cudaStream_t s = (cudaStream_t)stream;
    dim3 g1(nc, n_heads);
    // stage1 v2 (PADDOCK_DNC_S1MMA=1): mma dots + hierarchical explicit
    // (I+M)^-1 + mma dw/du - deltanet/split.cuh. Near-f32 (3xTF32) but not
    // bit-exact vs the scalar stage1 (dot/solve order) - oracle+PPL gated.
    static const bool s1mma_env = [] {
        const char* e = pd_env("PADDOCK_DNC_S1MMA");
        return e && atoi(e) != 0;
    }();
    // dw/du-bf16 route (PADDOCK_DNC_DWB16): stage1 emits bf16 dw/du and the
    // classic walk stages/reads them as bf16 - call-internal buffers, so a
    // process-wide latch is safe. Restricted to the plain classic-v2 dispatch
    // at G=32 (scan/fla/split/v1/scalar all read f32 dw/du) and requires the
    // mma stage1 (the bf16 writer).
    static const bool dwb16 = [] {
        const char* e = pd_env("PADDOCK_DNC_DWB16");
        return e && atoi(e) != 0;
    }();
    // the walk-side reader only exists for the mma-v2 G=32 walk - mirror
    // the mma_env/mma_g resolution here (they latch later in this function)
    static const bool dwb16_walk_ok = [] {
        const char* me = pd_env("PADDOCK_DNC_MMA");
        const char* mg = pd_env("PADDOCK_DNC_MMA_G");
        const uint32_t v = me ? (uint32_t)atoi(me) : 0u;
        return (v == 1u || v == 3u) && mg && atoi(mg) == 32;
    }();
    const bool dwb16_active = dwb16 && s1mma_env && dwb16_walk_ok && !vb16 &&
        !pd_env("PADDOCK_NO_DNC_MMA_V2") && !pd_env("PADDOCK_DNC_SCAN") &&
        !pd_env("PADDOCK_DNC_FLA") && !pd_env("PADDOCK_DNC_SPLIT");
    // stage1 RS rebuild election (PADDOCK_DNC_S1RS): the
    // bf16-operand stage1 (deltanet/walk_rs.cuh tail) replaces stage1_v2 on
    // the RS routes. Numeric class change (bf16-rounded dots/dw/du
    // products - the reference's own operand class) - opt-in until it passes
    // the walk-election gate set (proto band, PPL, greedy, suite, serve A/B).
    // REGISTER-STATE bf16-operand walk route (PADDOCK_DNC_RS,
    // - deltanet/walk_rs.cuh). Replaces the classic stage1-f32 +
    // v2-walk pair: stage1 runs its OT/AT=bf16 arm (bf16 dw/du/coef ride
    // the f32-sized call-internal buffers - the DWB16 precedent - plus
    // bf16 q/k copies from the tail epilogue), then one 128-thread
    // register-state walk. f32 STATE preserved (bf16 state stays falsified);
    // proto -61% vs classic at T=2048. Read per call so a process can A/B.
    // Excluded: bf16 state, dwb16, scan/fla/split arms, NO_DNC_MMA_V2.
    {
        const char* rse = pd_env("PADDOCK_DNC_RS");
        const bool rs_on = rse && atoi(rse) != 0;
        // f16 state rides the ST walk (PPL-gated +0.09%); bf16 stays
        // falsified and falls through to the DNM walk arm below.
        if (rs_on && s1mma_env && !dwb16_active && pd_dns_state_class() != 1 &&
            !pd_env("PADDOCK_NO_DNC_MMA_V2") && !pd_env("PADDOCK_DNC_SCAN") &&
            !pd_env("PADDOCK_DNC_FLA") && !pd_env("PADDOCK_DNC_SPLIT")) {
            static const uint32_t rs_s1prec = [] {
                const char* e = pd_env("PADDOCK_DNC_S1PREC");
                const uint32_t v = e ? (uint32_t)atoi(e) : 3u;
                return v == 1u ? 1u : 3u;
            }();
            static bool rs_attr = false;
            if (!rs_attr) {
#define PD_RS_S1_ATTR(P, VTT)                                                  \
    cudaFuncSetAttribute(                                                      \
        (const void*)pd_dnc_stage1_v2_kernel<P, float, __nv_bfloat16, VTT,     \
                                             __nv_bfloat16>,                   \
        cudaFuncAttributeMaxDynamicSharedMemorySize, PD_DNS1_SMEM)
                PD_RS_S1_ATTR(1u, float);
                PD_RS_S1_ATTR(3u, float);
                PD_RS_S1_ATTR(1u, __nv_bfloat16);
                PD_RS_S1_ATTR(3u, __nv_bfloat16);
#undef PD_RS_S1_ATTR
                cudaFuncSetAttribute(
                    (const void*)pd_dnc_stage1_rs_kernel<float>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    PD_DNS1RS_SMEM);
                cudaFuncSetAttribute(
                    (const void*)pd_dnc_stage1_rs_kernel<__nv_bfloat16>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    PD_DNS1RS_SMEM);
                // depth-8 ring (80 KB) crossed the 48 KB default cap
                cudaFuncSetAttribute((const void*)pd_dnc_walk_rs_kernel<float>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, PD_DNRS_SMEM);
                cudaFuncSetAttribute((const void*)pd_dnc_walk_rs_kernel<__half>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, PD_DNRS_SMEM);
                cudaFuncSetAttribute(
                    (const void*)pd_dnc_walk_rs_kernel<__nv_fp8_e4m3>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, PD_DNRS_SMEM);
                rs_attr = true;
            }
            // bf16 q/k copies live in the UPPER halves of the f32-sized
            // dw/du call-internal buffers: the RS route's bf16 dw/du
            // payloads occupy only the lower nc*H*C*D*2 bytes, and
            // n_tokens <= nc*C makes the freed upper half always big
            // enough. No allocation - the engine's KV pool owns ~all VRAM,
            // so any mid-serve cudaMalloc(-Async) in a launcher is a
            // serve-killer (measured: error 2 under the PPL harness).
            const size_t rs_half = (size_t)nc * PD_DNC_C * n_heads * PD_DNC_D;
            __nv_bfloat16* rs_qb = (__nv_bfloat16*)dw + rs_half;
            __nv_bfloat16* rs_kb = (__nv_bfloat16*)du + rs_half;
            // gate vectors ride the free upper half of the aqk buffer (the
            // bf16 coef payload fills only the lower nc*H*C*C*2 bytes);
            // 2C floats per chunk-head needs nc*H*512 B of the nc*H*8192 B
            // upper region - same no-allocation law as rs_qb/rs_kb.
            float* rs_gsh = (float*)((char*)aqk
                + (size_t)nc * PD_DNC_C * PD_DNC_C * n_heads * 2u);
#define PD_RS_S1_GO(P, VTT)                                                    \
    pd_dnc_stage1_v2_kernel<P, float, __nv_bfloat16, VTT, __nv_bfloat16>       \
        <<<g1, 256, PD_DNS1_SMEM, s>>>(                                        \
            (const float*)q, (const float*)k, (const VTT*)v, (const float*)g,  \
            (const float*)beta, (__nv_bfloat16*)dw, (__nv_bfloat16*)du,        \
            (__nv_bfloat16*)aqk, (double*)cg, n_tokens, n_heads, rs_qb,        \
            rs_kb, rs_gsh)
#define PD_RS_S1RS_GO(VTT)                                                     \
    pd_dnc_stage1_rs_kernel<VTT><<<g1, 256, PD_DNS1RS_SMEM, s>>>(              \
        (const float*)q, (const float*)k, (const VTT*)v, (const float*)g,      \
        (const float*)beta, (__nv_bfloat16*)dw, (__nv_bfloat16*)du,            \
        (__nv_bfloat16*)aqk, (double*)cg, n_tokens, n_heads, rs_qb, rs_kb,     \
        rs_gsh)
// PD_DNC_PROBE (bench-only, default 0): 1 = stage1 only (no walk), 2 = walk
// only (on whatever the scratch holds) - the GB10 phase split
// (bench/dnc_gb10_bench.cu, 2026-09-08). Never set by the pack build.
#ifndef PD_DNC_PROBE
#define PD_DNC_PROBE 0
#endif
#if PD_DNC_PROBE != 2
            if (pd_dnc_s1rs_on()) {
                if (vb16) PD_RS_S1RS_GO(__nv_bfloat16);
                else PD_RS_S1RS_GO(float);
            } else if (vb16) {
                if (rs_s1prec == 1u) PD_RS_S1_GO(1u, __nv_bfloat16);
                else PD_RS_S1_GO(3u, __nv_bfloat16);
            } else {
                if (rs_s1prec == 1u) PD_RS_S1_GO(1u, float);
                else PD_RS_S1_GO(3u, float);
            }
#endif
#undef PD_RS_S1RS_GO
#undef PD_RS_S1_GO
            // column groups FASTEST (2026-09-08, GB10): a head's four CTAs
            // share dw/q/k/coef, and on a 48-SM die with a 24 MB L2 they
            // must sit in the same wave to share them through L2 - head-
            // fastest order put them two waves apart (walk_rs.cuh note).
            dim3 gw(PD_DNC_D / PD_DNC_G, n_heads);
            const int rs_cls = pd_dns_state_class();
#if PD_DNC_PROBE == 1
            (void)gw; (void)rs_cls;
            return pd_launch_status();
#endif
            if (rs_cls == 3)
                pd_dnrs_walk_go<__nv_fp8_e4m3, false>(gw, s, rs_qb, rs_kb, (__nv_fp8_e4m3*)state, (const __nv_bfloat16*)dw, (const __nv_bfloat16*)du, rs_gsh, (const __nv_bfloat16*)aqk, (float*)out, n_tokens, n_heads, nullptr, nullptr, nullptr, 0u);
            else if (rs_cls == 2)
                pd_dnrs_walk_go<__half, false>(gw, s, rs_qb, rs_kb, (__half*)state, (const __nv_bfloat16*)dw, (const __nv_bfloat16*)du, rs_gsh, (const __nv_bfloat16*)aqk, (float*)out, n_tokens, n_heads, nullptr, nullptr, nullptr, 0u);
            else
                pd_dnrs_walk_go<float, false>(gw, s, rs_qb, rs_kb, (float*)state, (const __nv_bfloat16*)dw, (const __nv_bfloat16*)du, rs_gsh, (const __nv_bfloat16*)aqk, (float*)out, n_tokens, n_heads, nullptr, nullptr, nullptr, 0u);
            return pd_launch_status();
        }
    }
    if (s1mma_env) {
        // v-bf16 route (the vb16 export): v arrives bf16 from
        // conv_qkv_b16; q/k stay f32 (the walk reads both, the dots keep
        // f32 fragments) and dw/du stay f32 - the walk is untouched. Routed
        // per CALL by the export used, so f32-v producers (resume/mixed
        // paths) can never hit the bf16 read.
        const bool bf16ops = vb16;
        // stage1 PREC election (PADDOCK_DNC_S1PREC): 1 = plain
        // tf32 (the walk's own elected class), 3 = 3xTF32 (the shipped
        // default until the tf32 arm passes its gates). Same knob semantics
        // as PADDOCK_DNC_MMA; not bit-exact across values - oracle/PPL/
        // greedy gated like every class change.
        static const uint32_t s1prec = [] {
            const char* e = pd_env("PADDOCK_DNC_S1PREC");
            const uint32_t v = e ? (uint32_t)atoi(e) : 3u;
            return v == 1u ? 1u : 3u;
        }();
        static bool s1_attr = false;
        if (!s1_attr) {
            cudaFuncSetAttribute(
                (const void*)pd_dnc_stage1_v2_kernel<3u, float, float>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, PD_DNS1_SMEM);
            cudaFuncSetAttribute(
                (const void*)pd_dnc_stage1_v2_kernel<1u, float, float>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, PD_DNS1_SMEM);
            cudaFuncSetAttribute(
                (const void*)pd_dnc_stage1_v2_kernel<3u, float, float, __nv_bfloat16>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, PD_DNS1_SMEM);
            cudaFuncSetAttribute(
                (const void*)pd_dnc_stage1_v2_kernel<1u, float, float, __nv_bfloat16>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, PD_DNS1_SMEM);
            s1_attr = true;
        }
        if (dwb16_active) {
            static bool s1b_attr = false;
            if (!s1b_attr) {
                cudaFuncSetAttribute(
                    (const void*)pd_dnc_stage1_v2_kernel<3u, float, __nv_bfloat16, float>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, PD_DNS1_SMEM);
                cudaFuncSetAttribute(
                    (const void*)pd_dnc_stage1_v2_kernel<1u, float, __nv_bfloat16, float>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, PD_DNS1_SMEM);
                s1b_attr = true;
            }
            if (s1prec == 1u)
                pd_dnc_stage1_v2_kernel<1u, float, __nv_bfloat16, float>
                    <<<g1, 256, PD_DNS1_SMEM, s>>>(
                        (const float*)q, (const float*)k, (const float*)v,
                        (const float*)g, (const float*)beta, (__nv_bfloat16*)dw,
                        (__nv_bfloat16*)du, (float*)aqk, (double*)cg, n_tokens,
                        n_heads);
            else
                pd_dnc_stage1_v2_kernel<3u, float, __nv_bfloat16, float>
                    <<<g1, 256, PD_DNS1_SMEM, s>>>(
                        (const float*)q, (const float*)k, (const float*)v,
                        (const float*)g, (const float*)beta, (__nv_bfloat16*)dw,
                        (__nv_bfloat16*)du, (float*)aqk, (double*)cg, n_tokens,
                        n_heads);
        } else if (bf16ops) {
            if (s1prec == 1u)
                pd_dnc_stage1_v2_kernel<1u, float, float, __nv_bfloat16>
                    <<<g1, 256, PD_DNS1_SMEM, s>>>(
                        (const float*)q, (const float*)k,
                        (const __nv_bfloat16*)v, (const float*)g, (const float*)beta,
                        (float*)dw, (float*)du, (float*)aqk,
                        (double*)cg, n_tokens, n_heads);
            else
                pd_dnc_stage1_v2_kernel<3u, float, float, __nv_bfloat16>
                    <<<g1, 256, PD_DNS1_SMEM, s>>>(
                        (const float*)q, (const float*)k,
                        (const __nv_bfloat16*)v, (const float*)g, (const float*)beta,
                        (float*)dw, (float*)du, (float*)aqk,
                        (double*)cg, n_tokens, n_heads);
        } else if (s1prec == 1u)
        pd_dnc_stage1_v2_kernel<1u, float, float><<<g1, 256, PD_DNS1_SMEM, s>>>(
            (const float*)q, (const float*)k, (const float*)v, (const float*)g,
            (const float*)beta, (float*)dw, (float*)du, (float*)aqk, (double*)cg,
            n_tokens, n_heads);
        else
        pd_dnc_stage1_v2_kernel<3u, float, float><<<g1, 256, PD_DNS1_SMEM, s>>>(
            (const float*)q, (const float*)k, (const float*)v, (const float*)g,
            (const float*)beta, (float*)dw, (float*)du, (float*)aqk, (double*)cg,
            n_tokens, n_heads);
    } else
    pd_dnc_stage1_kernel<<<g1, 256, PD_DNC_S1_SMEM, s>>>(
        (const float*)q, (const float*)k, (const float*)v, (const float*)g,
        (const float*)beta, (float*)dw, (float*)du, (float*)aqk, (double*)cg,
        n_tokens, n_heads);
    // Tensor-core stage 2: PADDOCK_DNC_MMA=1 (plain tf32) or 3 (3xTF32,
    // near-f32). PADDOCK_DNC_MMA_G in {16,32} picks the column-slice width
    // (16 -> 256 blocks = 1.36 waves, the profiled grid fix; default).
    // Default off (scalar stage2) until gated.
    static const uint32_t mma_env = [] {
        const char* e = pd_env("PADDOCK_DNC_MMA");
        const uint32_t v = e ? (uint32_t)atoi(e) : 0u;
        return (v == 1u || v == 3u) ? v : 0u;
    }();
    static const uint32_t mma_g = [] {
        const char* e = pd_env("PADDOCK_DNC_MMA_G");
        const uint32_t v = e ? (uint32_t)atoi(e) : 16u;
        return (v == 32u) ? 32u : 16u;
    }();
    if (mma_env != 0u) {
        const float* qf = (const float*)q;
        const float* kf = (const float*)k;
        // Two-level scan (PADDOCK_DNC_SCAN=1, needs enough chunks to split):
        // A1 -> A2 -> B produce per-partition ENTRY states; the walk (C) then
        // runs one partition per grid-z from its true entry state. Scratch is
        // stream-ordered mallocAsync (pooled after warmup).
        static const bool scan_env = [] {
            const char* e = pd_env("PADDOCK_DNC_SCAN");
            return e && atoi(e) != 0;
        }();
        constexpr uint32_t DD = PD_DNC_D * PD_DNC_D;
        // scan scratch is f32 mallocAsync; the walk under bf16 state skips
        // scan (the walk itself handles bf16 entry/state).
        const int dns_cls = pd_dns_state_class();
        const bool dns_bf16 = dns_cls == 1;
        const bool dns_f16 = dns_cls == 2;
        const bool dns_f8 = dns_cls == 3;
        if (scan_env && dns_cls == 0 && nc >= 4u) {
            const uint32_t m = (nc + 7u) / 8u < 2u ? 2u : (nc + 7u) / 8u;
            const uint32_t P = (nc + m - 1u) / m;
            float *d_mst, *d_bst, *d_att, *d_bbar, *d_stash;
            if (cudaMallocAsync(&d_mst, (size_t)nc * n_heads * DD * 4u, s) ||
                cudaMallocAsync(&d_bst, (size_t)nc * n_heads * DD * 4u, s) ||
                cudaMallocAsync(&d_att, (size_t)P * n_heads * DD * 4u, s) ||
                cudaMallocAsync(&d_bbar, (size_t)P * n_heads * DD * 4u, s) ||
                cudaMallocAsync(&d_stash, (size_t)P * n_heads * DD * 4u, s))
                return cudaErrorMemoryAllocation;
            dim3 ga1(nc, n_heads);
            dim3 ga2(P, n_heads, 4);
            dim3 gb(n_heads, 2);
            dim3 gc(n_heads, PD_DNC_D / mma_g, P);
            const uint32_t a1s = 2u * PD_DNC_D * (PD_DNC_C + 4u) * 4u;
            const uint32_t a2s = 2u * 64u * (PD_DNC_D + 4u) * 4u;
            if (mma_env == 3u) {
                pd_dnc_scan_a1_kernel<3u><<<ga1, 256, a1s, s>>>(
                    kf, (const float*)dw, (const float*)du, (const double*)cg,
                    d_mst, d_bst, n_tokens, n_heads);
                pd_dnc_scan_a2_kernel<3u><<<ga2, 256, a2s, s>>>(
                    (const double*)cg, d_mst, d_bst, d_att, d_bbar, n_tokens,
                    n_heads, m);
                pd_dnc_scan_b_kernel<3u><<<gb, 256, a2s, s>>>(
                    (const float*)state, d_att, d_bbar, d_stash, n_heads, P);
                if (mma_g == 16u)
                    pd_dnc_stage2_mma_kernel<3u, 16u><<<gc, 256, PD_DNM_SMEM(16u), s>>>(
                        qf, kf, (float*)state, (const float*)dw, (const float*)du,
                        (const double*)cg, (const float*)aqk, (float*)out,
                        n_tokens, n_heads, d_stash, m);
                else
                    pd_dnc_stage2_mma_kernel<3u, 32u><<<gc, 256, PD_DNM_SMEM(32u), s>>>(
                        qf, kf, (float*)state, (const float*)dw, (const float*)du,
                        (const double*)cg, (const float*)aqk, (float*)out,
                        n_tokens, n_heads, d_stash, m);
            } else {
                pd_dnc_scan_a1_kernel<1u><<<ga1, 256, a1s, s>>>(
                    kf, (const float*)dw, (const float*)du, (const double*)cg,
                    d_mst, d_bst, n_tokens, n_heads);
                pd_dnc_scan_a2_kernel<1u><<<ga2, 256, a2s, s>>>(
                    (const double*)cg, d_mst, d_bst, d_att, d_bbar, n_tokens,
                    n_heads, m);
                pd_dnc_scan_b_kernel<1u><<<gb, 256, a2s, s>>>(
                    (const float*)state, d_att, d_bbar, d_stash, n_heads, P);
                if (mma_g == 16u)
                    pd_dnc_stage2_mma_kernel<1u, 16u><<<gc, 256, PD_DNM_SMEM(16u), s>>>(
                        qf, kf, (float*)state, (const float*)dw, (const float*)du,
                        (const double*)cg, (const float*)aqk, (float*)out,
                        n_tokens, n_heads, d_stash, m);
                else
                    pd_dnc_stage2_mma_kernel<1u, 32u><<<gc, 256, PD_DNM_SMEM(32u), s>>>(
                        qf, kf, (float*)state, (const float*)dw, (const float*)du,
                        (const double*)cg, (const float*)aqk, (float*)out,
                        n_tokens, n_heads, d_stash, m);
            }
            cudaFreeAsync(d_mst, s); cudaFreeAsync(d_bst, s);
            cudaFreeAsync(d_att, s); cudaFreeAsync(d_bbar, s);
            cudaFreeAsync(d_stash, s);
            return pd_launch_status();
        }
        dim3 gm(n_heads, PD_DNC_D / mma_g, 1);
        // v2 cp.async-ring walk - DEFAULT on (bit-exact vs v1 by construction,
        // verified 0/8.9M words at the pf8 span + memcheck-clean incl. partial
        // tails; serving A/B: pf8 366.6->375.6, c32 band-top
        // 1587.4). Kill switch PADDOCK_NO_DNC_MMA_V2 pins the v1 walk. Read
        // per call, not latched, so a single process can A/B both walks.
        // one dtype-branching launch per (walk, PREC, G): bf16 state rides the
        // same kernels via the ST template (PADDOCK_DN_STATE_BF16)
#define PD_DNM_GO(KER, P, GG, SMEM)                                            \
        do {                                                                    \
            if (dns_f8)                                                         \
                KER<P, GG, __nv_fp8_e4m3><<<gm, 256, SMEM, s>>>(                \
                    qf, kf, (__nv_fp8_e4m3*)state, (const float*)dw,            \
                    (const float*)du, (const double*)cg, (const float*)aqk,     \
                    (float*)out, n_tokens, n_heads,                             \
                    (const __nv_fp8_e4m3*)state, nc);                           \
            else if (dns_f16)                                                   \
                KER<P, GG, __half><<<gm, 256, SMEM, s>>>(                       \
                    qf, kf, (__half*)state, (const float*)dw,                   \
                    (const float*)du, (const double*)cg, (const float*)aqk,     \
                    (float*)out, n_tokens, n_heads,                             \
                    (const __half*)state, nc);                                  \
            else if (dns_bf16)                                                  \
                KER<P, GG, __nv_bfloat16><<<gm, 256, SMEM, s>>>(                \
                    qf, kf, (__nv_bfloat16*)state, (const float*)dw,            \
                    (const float*)du, (const double*)cg, (const float*)aqk,     \
                    (float*)out, n_tokens, n_heads,                             \
                    (const __nv_bfloat16*)state, nc);                           \
            else                                                                \
                KER<P, GG><<<gm, 256, SMEM, s>>>(                               \
                    qf, kf, (float*)state, (const float*)dw, (const float*)du,  \
                    (const double*)cg, (const float*)aqk, (float*)out,          \
                    n_tokens, n_heads, (const float*)state, nc);                \
        } while (0)
        // SPLIT walk route (PADDOCK_DNC_SPLIT=1, nc >= 2, G=32; bf16 state
        // supported - the walk is ST-templated, both stashes stay f32): the
        // minimal serial walk (deltanet/split.cuh) + the chunk-parallel
        // o-pass. Bit-exact vs the classic v2 walk by construction (classic
        // k-grouping and operand bits preserved end to end - see split.cuh).
        {
            static const bool split_env = [] {
                const char* e = pd_env("PADDOCK_DNC_SPLIT");
                return e && atoi(e) != 0;
            }();
            const uint32_t nc3 = (n_tokens + PD_DNC_C - 1u) / PD_DNC_C;
            if (split_env && nc3 >= 2u && mma_g == 32u &&
                !pd_env("PADDOCK_NO_DNC_MMA_V2")) {
                static bool split_attr = false;
                if (!split_attr) {
#define PD_SPLIT_ATTR(...)                                                     \
    cudaFuncSetAttribute((const void*)(__VA_ARGS__),                           \
                         cudaFuncAttributeMaxDynamicSharedMemorySize,          \
                         PD_DNS3_WALK_SMEM)
                    PD_SPLIT_ATTR(pd_dnc_walk3_kernel<1u, float>);
                    PD_SPLIT_ATTR(pd_dnc_walk3_kernel<3u, float>);
                    PD_SPLIT_ATTR(pd_dnc_walk3_kernel<1u, __nv_bfloat16>);
                    PD_SPLIT_ATTR(pd_dnc_walk3_kernel<3u, __nv_bfloat16>);
                    PD_SPLIT_ATTR(pd_dnc_walk3_kernel<1u, __half>);
                    PD_SPLIT_ATTR(pd_dnc_walk3_kernel<3u, __half>);
                    PD_SPLIT_ATTR(pd_dnc_walk3_kernel<1u, __nv_fp8_e4m3>);
                    PD_SPLIT_ATTR(pd_dnc_walk3_kernel<3u, __nv_fp8_e4m3>);
#undef PD_SPLIT_ATTR
                    cudaFuncSetAttribute((const void*)pd_dnc_opass_kernel<1u>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         PD_DNS3_OPASS_SMEM);
                    cudaFuncSetAttribute((const void*)pd_dnc_opass_kernel<3u>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         PD_DNS3_OPASS_SMEM);
                    split_attr = true;
                }
                constexpr uint32_t CD3 = PD_DNC_C * PD_DNC_D;
                float* d_dt;
                if (cudaMallocAsync(&d_dt, (size_t)nc3 * n_heads * CD3 * 4u, s))
                    return cudaErrorMemoryAllocation;
                // column groups FASTEST (2026-09-08, GB10): a head's four CTAs
            // share dw/q/k/coef, and on a 48-SM die with a 24 MB L2 they
            // must sit in the same wave to share them through L2 - head-
            // fastest order put them two waves apart (walk_rs.cuh note).
            dim3 gw(n_heads, PD_DNC_D / PD_DNC_G);   // walk3 decodes the head from blockIdx.x (the rs walk's swap does not apply here)
                dim3 go3(n_heads, PD_DNC_D / PD_DNC_G, nc3);
                const float* qf3 = (const float*)q;
                const float* kf3 = (const float*)k;
#define PD_SPLIT_GO(P)                                                         \
    do {                                                                       \
        if (dns_f8)                                                            \
            pd_dnc_walk3_kernel<P, __nv_fp8_e4m3>                              \
                <<<gw, 256, PD_DNS3_WALK_SMEM, s>>>(                           \
                    qf3, kf3, (__nv_fp8_e4m3*)state, (const float*)dw,         \
                    (const float*)du, (const double*)cg, (float*)out, d_dt,    \
                    n_tokens, n_heads);                                        \
        else if (dns_f16)                                                      \
            pd_dnc_walk3_kernel<P, __half>                                     \
                <<<gw, 256, PD_DNS3_WALK_SMEM, s>>>(                           \
                    qf3, kf3, (__half*)state, (const float*)dw,                \
                    (const float*)du, (const double*)cg, (float*)out, d_dt,    \
                    n_tokens, n_heads);                                        \
        else if (dns_bf16)                                                     \
            pd_dnc_walk3_kernel<P, __nv_bfloat16>                              \
                <<<gw, 256, PD_DNS3_WALK_SMEM, s>>>(                           \
                    qf3, kf3, (__nv_bfloat16*)state, (const float*)dw,         \
                    (const float*)du, (const double*)cg, (float*)out, d_dt,    \
                    n_tokens, n_heads);                                        \
        else                                                                   \
            pd_dnc_walk3_kernel<P, float><<<gw, 256, PD_DNS3_WALK_SMEM, s>>>(  \
                qf3, kf3, (float*)state, (const float*)dw, (const float*)du,   \
                (const double*)cg, (float*)out, d_dt, n_tokens, n_heads);      \
        pd_dnc_opass_kernel<P><<<go3, 256, PD_DNS3_OPASS_SMEM, s>>>(           \
            (const float*)aqk, d_dt, (float*)out, n_tokens, n_heads);          \
    } while (0)
                if (mma_env == 3u) PD_SPLIT_GO(3u);
                else PD_SPLIT_GO(1u);
#undef PD_SPLIT_GO
                cudaFreeAsync(d_dt, s);
                return pd_launch_status();
            }
        }
        // fla-class split route (PADDOCK_DNC_FLA=1, f32 state, nc >= 2),
        // iteration 2: the serial walk runs a state-only 7-item schedule
        // (dw-only A-slabs, no coef, no pass 2) stashing each chunk's entry
        // state AND resolved deltaT; a fully chunk-parallel launch then
        // assembles outputs on a 5-item schedule (q-only A-slabs + coef)
        // straight from the stashes - no solve recompute. Bit-exact
        // composition vs the single kernel (identical fragment math).
        // Stash: nc*H*(D*D entry-S + C*D deltaT) f32, stream-ordered like
        // the scan scratch.
        {
            static const bool fla_env = [] {
                const char* e = pd_env("PADDOCK_DNC_FLA");
                return e && atoi(e) != 0;
            }();
            const uint32_t nc_ = (n_tokens + PD_DNC_C - 1u) / PD_DNC_C;
            if (fla_env && dns_cls == 0 && nc_ >= 2u && !pd_env("PADDOCK_NO_DNC_MMA_V2")) {
                static bool fla_attr = false;
                if (!fla_attr) {
#define PD_FLA_ATTR(P, GG)                                                         cudaFuncSetAttribute(                                                              (const void*)pd_dnc_stage2_mma_v2_kernel<P, GG, float, false, true>,           cudaFuncAttributeMaxDynamicSharedMemorySize, PD_DNM2_SMEM(GG));            cudaFuncSetAttribute(                                                              (const void*)pd_dnc_stage2_mma_v2_kernel<P, GG, float, true, false>,           cudaFuncAttributeMaxDynamicSharedMemorySize, PD_DNM2_SMEM(GG))
                    PD_FLA_ATTR(1u, 16u); PD_FLA_ATTR(1u, 32u);
                    PD_FLA_ATTR(3u, 16u); PD_FLA_ATTR(3u, 32u);
#undef PD_FLA_ATTR
                    fla_attr = true;
                }
                float* d_stash = nullptr;
                if (cudaMallocAsync(&d_stash,
                        (size_t)nc_ * n_heads *
                            (PD_DNC_D * PD_DNC_D + PD_DNC_C * PD_DNC_D) * 4u, s))
                    return cudaErrorMemoryAllocation;
                dim3 go(n_heads, PD_DNC_D / mma_g, nc_);
#define PD_FLA_GO(P, GG, SMEM)                                                     do {                                                                               pd_dnc_stage2_mma_v2_kernel<P, GG, float, false, true>                             <<<gm, 256, SMEM, s>>>(qf, kf, (float*)state, (const float*)dw,                    (const float*)du, (const double*)cg, (const float*)aqk,                        (float*)out, n_tokens, n_heads, (const float*)state, nc_,                      d_stash);                                                              pd_dnc_stage2_mma_v2_kernel<P, GG, float, true, false>                             <<<go, 256, SMEM, s>>>(qf, kf, (float*)state, (const float*)dw,                    (const float*)du, (const double*)cg, (const float*)aqk,                        (float*)out, n_tokens, n_heads, (const float*)d_stash, 1u,                     d_stash);                                                          } while (0)
                if (mma_env == 3u && mma_g == 16u)
                    PD_FLA_GO(3u, 16u, PD_DNM2_SMEM(16u));
                else if (mma_env == 3u)
                    PD_FLA_GO(3u, 32u, PD_DNM2_SMEM(32u));
                else if (mma_g == 16u)
                    PD_FLA_GO(1u, 16u, PD_DNM2_SMEM(16u));
                else
                    PD_FLA_GO(1u, 32u, PD_DNM2_SMEM(32u));
#undef PD_FLA_GO
                cudaFreeAsync(d_stash, s);
                return pd_launch_status();
            }
        }
        {
            if (!pd_env("PADDOCK_NO_DNC_MMA_V2")) {
                if (dwb16_active) {
                    static bool dwb_attr = false;
                    if (!dwb_attr) {
#define PD_DWB_ATTR(...)                                                       \
    cudaFuncSetAttribute((const void*)(__VA_ARGS__),                           \
                         cudaFuncAttributeMaxDynamicSharedMemorySize,          \
                         PD_DNM2_SMEM(32u))
                        PD_DWB_ATTR(pd_dnc_stage2_mma_v2_kernel<1u, 32u, float, true, true, __nv_bfloat16>);
                        PD_DWB_ATTR(pd_dnc_stage2_mma_v2_kernel<3u, 32u, float, true, true, __nv_bfloat16>);
                        PD_DWB_ATTR(pd_dnc_stage2_mma_v2_kernel<1u, 32u, __nv_bfloat16, true, true, __nv_bfloat16>);
                        PD_DWB_ATTR(pd_dnc_stage2_mma_v2_kernel<3u, 32u, __nv_bfloat16, true, true, __nv_bfloat16>);
#undef PD_DWB_ATTR
                        dwb_attr = true;
                    }
                    // dwb16 serves the G=32 serving shape; other G falls back
                    if (mma_g == 32u) {
#define PD_DWB_GO(P)                                                           \
    do {                                                                       \
        if (dns_f8)                                                            \
            pd_dnc_stage2_mma_v2_kernel<P, 32u, __nv_fp8_e4m3, true, true,     \
                                        __nv_bfloat16>                         \
                <<<gm, 256, PD_DNM2_SMEM(32u), s>>>(                           \
                    qf, kf, (__nv_fp8_e4m3*)state, (const __nv_bfloat16*)dw,   \
                    (const __nv_bfloat16*)du, (const double*)cg,               \
                    (const float*)aqk, (float*)out, n_tokens, n_heads,         \
                    (const __nv_fp8_e4m3*)state, nc);                          \
        else if (dns_f16)                                                      \
            pd_dnc_stage2_mma_v2_kernel<P, 32u, __half, true, true,            \
                                        __nv_bfloat16>                         \
                <<<gm, 256, PD_DNM2_SMEM(32u), s>>>(                           \
                    qf, kf, (__half*)state, (const __nv_bfloat16*)dw,          \
                    (const __nv_bfloat16*)du, (const double*)cg,               \
                    (const float*)aqk, (float*)out, n_tokens, n_heads,         \
                    (const __half*)state, nc);                                 \
        else if (dns_bf16)                                                     \
            pd_dnc_stage2_mma_v2_kernel<P, 32u, __nv_bfloat16, true, true,     \
                                        __nv_bfloat16>                         \
                <<<gm, 256, PD_DNM2_SMEM(32u), s>>>(                           \
                    qf, kf, (__nv_bfloat16*)state, (const __nv_bfloat16*)dw,   \
                    (const __nv_bfloat16*)du, (const double*)cg,               \
                    (const float*)aqk, (float*)out, n_tokens, n_heads,         \
                    (const __nv_bfloat16*)state, nc);                          \
        else                                                                   \
            pd_dnc_stage2_mma_v2_kernel<P, 32u, float, true, true,             \
                                        __nv_bfloat16>                         \
                <<<gm, 256, PD_DNM2_SMEM(32u), s>>>(                           \
                    qf, kf, (float*)state, (const __nv_bfloat16*)dw,           \
                    (const __nv_bfloat16*)du, (const double*)cg,               \
                    (const float*)aqk, (float*)out, n_tokens, n_heads,         \
                    (const float*)state, nc);                                  \
    } while (0)
                        if (mma_env == 3u) PD_DWB_GO(3u);
                        else PD_DWB_GO(1u);
#undef PD_DWB_GO
                        return pd_launch_status();
                    }
                }
                if (mma_env == 3u && mma_g == 16u)
                    PD_DNM_GO(pd_dnc_stage2_mma_v2_kernel, 3u, 16u, PD_DNM2_SMEM(16u));
                else if (mma_env == 3u)
                    PD_DNM_GO(pd_dnc_stage2_mma_v2_kernel, 3u, 32u, PD_DNM2_SMEM(32u));
                else if (mma_g == 16u)
                    PD_DNM_GO(pd_dnc_stage2_mma_v2_kernel, 1u, 16u, PD_DNM2_SMEM(16u));
                else
                    PD_DNM_GO(pd_dnc_stage2_mma_v2_kernel, 1u, 32u, PD_DNM2_SMEM(32u));
                return pd_launch_status();
            }
        }
        if (mma_env == 3u && mma_g == 16u)
            PD_DNM_GO(pd_dnc_stage2_mma_kernel, 3u, 16u, PD_DNM_SMEM(16u));
        else if (mma_env == 3u)
            PD_DNM_GO(pd_dnc_stage2_mma_kernel, 3u, 32u, PD_DNM_SMEM(32u));
        else if (mma_g == 16u)
            PD_DNM_GO(pd_dnc_stage2_mma_kernel, 1u, 16u, PD_DNM_SMEM(16u));
        else
            PD_DNM_GO(pd_dnc_stage2_mma_kernel, 1u, 32u, PD_DNM_SMEM(32u));
#undef PD_DNM_GO
        return pd_launch_status();
    }
    // narrow state (bf16/f16) requires the mma walks (the scalar stage2 below
    // and the scan path read f32 state) - fail loud rather than corrupt.
    if (pd_dns_nonf32_env()) return cudaErrorInvalidValue;
    // Column-slice width: G=16 doubles the grid (256 blocks) bit-exactly -
    // see the kernel comment. Env PADDOCK_DNC_G in {8,16,32}, default 32
    // (the original) until the serving A/B picks the winner.
    static const uint32_t g_env = [] {
        const char* e = pd_env("PADDOCK_DNC_G");
        const uint32_t v = e ? (uint32_t)atoi(e) : PD_DNC_G;
        return (v == 8u || v == 16u) ? v : PD_DNC_G;
    }();
    dim3 g2(n_heads, PD_DNC_D / g_env);
    if (g_env == 8u)
        pd_dnc_stage2_kernel<8u><<<g2, 256, 0, s>>>(
            (const float*)q, (const float*)k, (float*)state, (const float*)dw,
            (const float*)du, (const double*)cg, (const float*)aqk, (float*)out,
            n_tokens, n_heads);
    else if (g_env == 16u)
        pd_dnc_stage2_kernel<16u><<<g2, 256, 0, s>>>(
            (const float*)q, (const float*)k, (float*)state, (const float*)dw,
            (const float*)du, (const double*)cg, (const float*)aqk, (float*)out,
            n_tokens, n_heads);
    else
        pd_dnc_stage2_kernel<PD_DNC_G><<<g2, 256, 0, s>>>(
            (const float*)q, (const float*)k, (float*)state, (const float*)dw,
            (const float*)du, (const double*)cg, (const float*)aqk, (float*)out,
            n_tokens, n_heads);
    return pd_launch_status();
}

PD_EXPORT
int pd_gated_delta_chunked(const void* q, const void* k, const void* v,
                           const void* g, const void* beta, void* state,
                           void* out, void* dw, void* du, void* aqk, void* cg,
                           uint32_t n_tokens, uint32_t n_heads,
                           uint32_t head_dim, void* stream) {
    return pd_gated_delta_chunked_go(false, q, k, v, g, beta, state, out, dw,
                                     du, aqk, cg, n_tokens, n_heads, head_dim,
                                     stream);
}

// 264: v-bf16 twin - v is bf16 (conv_qkv_b16 upstream), q/k/dw/du stay f32.
PD_EXPORT
int pd_gated_delta_chunked_vb16(const void* q, const void* k, const void* v,
                                const void* g, const void* beta, void* state,
                                void* out, void* dw, void* du, void* aqk,
                                void* cg, uint32_t n_tokens, uint32_t n_heads,
                                uint32_t head_dim, void* stream) {
    return pd_gated_delta_chunked_go(true, q, k, v, g, beta, state, out, dw,
                                     du, aqk, cg, n_tokens, n_heads, head_dim,
                                     stream);
}

// 323: varlen chunked-GDN (GDN formulation band): one
// stage1 + walk launch pair covers every eligible span of the tick - the
// reference varlen class (fla chunk kernels batch all spans and decode
// rows via chunk_indices; we batch the spans and keep decode rows on the
// recurrent path). chunk_items = (global row0, chunk len) u32 pairs per
// launch chunk; span_items = (first launch chunk, span rows, state f32
// offset, out row0) u32 quads per span. Per-span math is identical to the
// per-span RS launches (scratch stays launch-chunk-indexed; the walk
// re-bases every operand off its span item) - only the grid packing
// changes. RS-route only: mirrors the per-span dispatch's env gates and
// returns cudaErrorNotSupported when any other arm is elected, so the
// engine falls back per-span loudly rather than silently misrouting.
// f32-v only (the mixed-tick producer class this serves).
static int pd_gdc_rs_vl_go(
    bool qkc, uint32_t n_k_heads, const void* q, const void* k, const void* v,
    const void* g, const void* beta, void* state, void* out, void* dw,
    void* du, void* aqk, void* cg, const void* chunk_items, uint32_t n_chunks,
    const void* span_items, uint32_t n_spans, uint32_t n_tokens,
    uint32_t n_heads, uint32_t head_dim, void* stream) {
    if (n_chunks == 0 || n_spans == 0 || n_heads == 0) return 0;
    if (head_dim != PD_DNC_D) return cudaErrorInvalidValue;
    static const bool rs_ok = [] {
        const char* rse = pd_env("PADDOCK_DNC_RS");
        const char* s1e = pd_env("PADDOCK_DNC_S1MMA");
        const char* db = pd_env("PADDOCK_DNC_DWB16");
        return rse && atoi(rse) != 0 && s1e && atoi(s1e) != 0 &&
               !(db && atoi(db) != 0) && pd_dns_state_class() != 1 &&
               !pd_env("PADDOCK_NO_DNC_MMA_V2") && !pd_env("PADDOCK_DNC_SCAN") &&
               !pd_env("PADDOCK_DNC_FLA") && !pd_env("PADDOCK_DNC_SPLIT");
    }();
    if (!rs_ok) return cudaErrorNotSupported;
    static const uint32_t rs_s1prec = [] {
        const char* e = pd_env("PADDOCK_DNC_S1PREC");
        const uint32_t v = e ? (uint32_t)atoi(e) : 3u;
        return v == 1u ? 1u : 3u;
    }();
    static bool vl_attr = false;
    if (!vl_attr) {
        cudaFuncSetAttribute(
            (const void*)pd_dnc_stage1_v2_kernel<1u, float, __nv_bfloat16,
                                                 float, __nv_bfloat16>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, PD_DNS1_SMEM);
        cudaFuncSetAttribute(
            (const void*)pd_dnc_stage1_v2_kernel<3u, float, __nv_bfloat16,
                                                 float, __nv_bfloat16>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, PD_DNS1_SMEM);
        cudaFuncSetAttribute((const void*)pd_dnc_walk_rs_kernel<float>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             PD_DNRS_SMEM);
        cudaFuncSetAttribute((const void*)pd_dnc_walk_rs_kernel<__half>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             PD_DNRS_SMEM);
        cudaFuncSetAttribute((const void*)pd_dnc_walk_rs_kernel<__nv_fp8_e4m3>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             PD_DNRS_SMEM);
        cudaFuncSetAttribute((const void*)pd_dnc_stage1_rs_kernel<float>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             PD_DNS1RS_SMEM);
        cudaFuncSetAttribute(
            (const void*)pd_dnc_stage1_rs_kernel<float, 0, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, PD_DNS1RS_SMEM);
        cudaFuncSetAttribute((const void*)pd_dnc_walk_rs_kernel<float, true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             PD_DNRS_SMEM);
        cudaFuncSetAttribute((const void*)pd_dnc_walk_rs_kernel<__half, true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             PD_DNRS_SMEM);
        cudaFuncSetAttribute(
            (const void*)pd_dnc_walk_rs_kernel<__nv_fp8_e4m3, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, PD_DNRS_SMEM);
        ::fprintf(stderr, "[dnc-vl] ENGAGED (varlen chunked-GDN)\n");
        vl_attr = true;
    }
    const cudaStream_t s = (cudaStream_t)stream;
    const uint32_t nc = n_chunks;
    // scratch carving: identical formulas to the per-span RS block, with nc
    // = the TICK's total launch chunks (buffers sized for it engine-side)
    const size_t rs_half = (size_t)nc * PD_DNC_C * n_heads * PD_DNC_D;
    __nv_bfloat16* rs_qb = (__nv_bfloat16*)dw + rs_half;
    __nv_bfloat16* rs_kb = (__nv_bfloat16*)du + rs_half;
    float* rs_gsh = (float*)((char*)aqk
        + (size_t)nc * PD_DNC_C * PD_DNC_C * n_heads * 2u);
    dim3 g1(nc, n_heads);
#define PD_RS_VL_S1(P)                                                         \
    pd_dnc_stage1_v2_kernel<P, float, __nv_bfloat16, float, __nv_bfloat16>     \
        <<<g1, 256, PD_DNS1_SMEM, s>>>(                                        \
            (const float*)q, (const float*)k, (const float*)v,                 \
            (const float*)g, (const float*)beta, (__nv_bfloat16*)dw,           \
            (__nv_bfloat16*)du, (__nv_bfloat16*)aqk, (double*)cg, n_tokens,    \
            n_heads, rs_qb, rs_kb, rs_gsh, (const uint32_t*)chunk_items)
    // QKD (2026-09-08, GB10): on the qkc route the walk reads the compact
    // plane itself (walk_rs.cuh), so stage1 skips its 25 MB emission
    // (qb16/kb16 = NULL). PADDOCK_NO_DNC_QKD=1 keeps the copies for A/B.
    static const bool qkd_off = pd_env("PADDOCK_NO_DNC_QKD") != nullptr;
    const bool qkd = qkc && !qkd_off;
#if PD_DNC_PROBE != 2
    if (qkc) {
        // compact pair (slot 447): the ENGINE guarantees conv wrote the
        // compact bf16 planes and that the rs stage1 route is live; a
        // mispaired call must fail loud, not read garbage.
        if (!pd_dnc_s1rs_on() || n_k_heads == 0u ||
            (n_heads % n_k_heads) != 0u)
            return cudaErrorNotSupported;
        pd_dnc_stage1_rs_kernel<float, 0, true><<<g1, 256, PD_DNS1RS_SMEM, s>>>(
            (const float*)q, (const float*)k, (const float*)v, (const float*)g,
            (const float*)beta, (__nv_bfloat16*)dw, (__nv_bfloat16*)du,
            (__nv_bfloat16*)aqk, (double*)cg, n_tokens, n_heads,
            qkd ? nullptr : rs_qb, qkd ? nullptr : rs_kb,
            rs_gsh, (const uint32_t*)chunk_items, n_k_heads);
    } else if (pd_dnc_s1rs_on())
        pd_dnc_stage1_rs_kernel<float><<<g1, 256, PD_DNS1RS_SMEM, s>>>(
            (const float*)q, (const float*)k, (const float*)v, (const float*)g,
            (const float*)beta, (__nv_bfloat16*)dw, (__nv_bfloat16*)du,
            (__nv_bfloat16*)aqk, (double*)cg, n_tokens, n_heads, rs_qb, rs_kb,
            rs_gsh, (const uint32_t*)chunk_items);
    else if (rs_s1prec == 1u) PD_RS_VL_S1(1u);
    else PD_RS_VL_S1(3u);
#endif
#undef PD_RS_VL_S1
    dim3 gw(PD_DNC_D / PD_DNC_G, n_heads, n_spans);   // column groups fastest (walk_rs.cuh note)
#if PD_DNC_PROBE == 1
    (void)gw; return pd_launch_status();
#endif
    const int vl_cls = pd_dns_state_class();
    if (qkd) {
        const __nv_bfloat16* qc = (const __nv_bfloat16*)q;
        const __nv_bfloat16* kc = (const __nv_bfloat16*)k;
        if (vl_cls == 3)
            pd_dnrs_walk_go<__nv_fp8_e4m3, true>(gw, s, rs_qb, rs_kb, (__nv_fp8_e4m3*)state, (const __nv_bfloat16*)dw, (const __nv_bfloat16*)du, rs_gsh, (const __nv_bfloat16*)aqk, (float*)out, n_tokens, n_heads, (const uint32_t*)span_items, qc, kc, n_k_heads);
        else if (vl_cls == 2)
            pd_dnrs_walk_go<__half, true>(gw, s, rs_qb, rs_kb, (__half*)state, (const __nv_bfloat16*)dw, (const __nv_bfloat16*)du, rs_gsh, (const __nv_bfloat16*)aqk, (float*)out, n_tokens, n_heads, (const uint32_t*)span_items, qc, kc, n_k_heads);
        else
            pd_dnrs_walk_go<float, true>(gw, s, rs_qb, rs_kb, (float*)state, (const __nv_bfloat16*)dw, (const __nv_bfloat16*)du, rs_gsh, (const __nv_bfloat16*)aqk, (float*)out, n_tokens, n_heads, (const uint32_t*)span_items, qc, kc, n_k_heads);
        return pd_launch_status();
    }
    if (vl_cls == 3)
        pd_dnrs_walk_go<__nv_fp8_e4m3, false>(gw, s, rs_qb, rs_kb, (__nv_fp8_e4m3*)state, (const __nv_bfloat16*)dw, (const __nv_bfloat16*)du, rs_gsh, (const __nv_bfloat16*)aqk, (float*)out, n_tokens, n_heads, (const uint32_t*)span_items, nullptr, nullptr, 0u);
    else if (vl_cls == 2)
        pd_dnrs_walk_go<__half, false>(gw, s, rs_qb, rs_kb, (__half*)state, (const __nv_bfloat16*)dw, (const __nv_bfloat16*)du, rs_gsh, (const __nv_bfloat16*)aqk, (float*)out, n_tokens, n_heads, (const uint32_t*)span_items, nullptr, nullptr, 0u);
    else
        pd_dnrs_walk_go<float, false>(gw, s, rs_qb, rs_kb, (float*)state, (const __nv_bfloat16*)dw, (const __nv_bfloat16*)du, rs_gsh, (const __nv_bfloat16*)aqk, (float*)out, n_tokens, n_heads, (const uint32_t*)span_items, nullptr, nullptr, 0u);
    return pd_launch_status();
}

PD_EXPORT
int pd_gated_delta_chunked_rs_vl(
    const void* q, const void* k, const void* v, const void* g,
    const void* beta, void* state, void* out, void* dw, void* du, void* aqk,
    void* cg, const void* chunk_items, uint32_t n_chunks,
    const void* span_items, uint32_t n_spans, uint32_t n_tokens,
    uint32_t n_heads, uint32_t head_dim, void* stream) {
    return pd_gdc_rs_vl_go(false, 0u, q, k, v, g, beta, state, out, dw, du,
                           aqk, cg, chunk_items, n_chunks, span_items, n_spans,
                           n_tokens, n_heads, head_dim, stream);
}

// QKC pair entry (slot 447): q/k are the conv qkc twin's COMPACT bf16
// planes; everything from the stage1 panes on is bit-identical to the
// expanded pair (see the kernels' comments).
PD_EXPORT
int pd_gated_delta_chunked_rs_vl_qkc(
    const void* q, const void* k, const void* v, const void* g,
    const void* beta, void* state, void* out, void* dw, void* du, void* aqk,
    void* cg, const void* chunk_items, uint32_t n_chunks,
    const void* span_items, uint32_t n_spans, uint32_t n_tokens,
    uint32_t n_heads, uint32_t n_k_heads, uint32_t head_dim, void* stream) {
    return pd_gdc_rs_vl_go(true, n_k_heads, q, k, v, g, beta, state, out, dw,
                           du, aqk, cg, chunk_items, n_chunks, span_items,
                           n_spans, n_tokens, n_heads, head_dim, stream);
}

// Row-wise argmax: out[row] = index of the max logit in row (LOWEST index wins
// ties - matches the host argmax's strict-greater ascending scan). One block per
// row; used by batched speculative decoding so per-row token picks stay on
// device instead of reading back [rows, vocab] logits.
__global__ void pd_argmax_rows_kernel(const float* __restrict__ logits,
                                      unsigned int* __restrict__ out, uint32_t n) {
    const uint32_t row = blockIdx.x, tid = threadIdx.x, nth = blockDim.x;
    const float* x = logits + (size_t)row * n;
    float bv = -3.402823466e+38f;
    uint32_t bi = 0;
    for (uint32_t i = tid; i < n; i += nth) {
        float v = x[i];
        if (v > bv) { bv = v; bi = i; }
    }
#pragma unroll
    for (uint32_t off = 16; off > 0; off >>= 1) {
        float ov = __shfl_down_sync(0xffffffffu, bv, off);
        uint32_t oi = __shfl_down_sync(0xffffffffu, bi, off);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    __shared__ float sv[8];
    __shared__ uint32_t si[8];
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    if (lane == 0) { sv[warp] = bv; si[warp] = bi; }
    __syncthreads();
    if (tid == 0) {
        for (uint32_t w = 1; w < ((nth + 31u) >> 5); ++w)
            if (sv[w] > bv || (sv[w] == bv && si[w] < bi)) { bv = sv[w]; bi = si[w]; }
        out[row] = bi;
    }
}

PD_EXPORT
int pd_argmax_rows(const void* logits, void* out, uint32_t rows, uint32_t n,
                   void* stream) {
    if (rows == 0 || n == 0) return 0;
    pd_argmax_rows_kernel<<<rows, 256, 0, (cudaStream_t)stream>>>(
        (const float*)logits, (unsigned int*)out, n);
    return pd_launch_status();
}

// Same greedy pick, plus everything a transcript needs to say how sure the
// model was - the RUNNER-UP included, which is what turns "confidence" from a
// number into something a reader can act on.
//
// Four readouts, all from one pass over the row:
//   log p(top1)  how much mass the chosen token took
//   log p(top2)  the road not taken. `p1 - p2` is the MARGIN, and margin is
//                the signal that separates "torn between two words" (where ASR
//                errors actually live) from "merely diffuse". The runner-up's
//                ID rides out alongside so the UI can name the alternative.
//   p(probe)     one nominated token's probability - whisper's `<|nospeech|>`,
//                which OpenAI defines at the first decode step. Reading it
//                host-side would mean copying back a 51866-float row per
//                window just to index one element of it.
//   H2           Renyi-2 (collision) entropy, -log sum(p^2), in nats.
//
// Why H2 and not Shannon: over a 51866-token vocabulary the Shannon sum is
// dominated by the TAIL - tens of thousands of near-zero terms - so it ranks a
// decisive row as more uncertain than a genuine two-way tie, which is
// backwards for the one job it has. The ASR confidence literature's answer is a
// tail-suppressed entropy (Tsallis/Renyi with alpha > 1; Laptev & Ginsburg,
// "Fast Entropy-Based Methods of Word-Level Confidence Estimation for E2E ASR",
// SLT 2022 - technique studied, implementation ours). alpha = 2 specifically,
// because sum(p^2) is the one alpha that falls out of an online softmax for
// free: square the same exponential the sum already computed. Any other alpha
// costs a second pass over the row plus a powf per element, which this kernel
// sits inside a graph-captured decode tick to avoid.
//
// This is `pd_argmax_rows` walking an online softmax and a top-2 alongside the
// max, so it stays one read of [rows, vocab] rather than three.
//
// Tie rule is the argmax kernel's exactly (lowest index wins) at both ranks, so
// the token this returns is bit-identical to what `pd_argmax_rows` would have
// picked - asking for confidence must never move the transcript.
#define PD_A2_BETTER(AM, AI, BM, BI) ((AM) > (BM) || ((AM) == (BM) && (AI) < (BI)))
// "no runner-up yet". Loses every comparison on value alone, so it also works
// as the initial champion for a thread that owns no elements at all.
#define PD_A2_NONE 0xffffffffu

__global__ void pd_argmax_top2_rows_kernel(const float* __restrict__ logits,
                                           unsigned int* __restrict__ out,
                                           unsigned int* __restrict__ alt,
                                           float* __restrict__ stats,
                                           uint32_t probe, uint32_t n) {
    const uint32_t row = blockIdx.x, tid = threadIdx.x, nth = blockDim.x;
    const float* x = logits + (size_t)row * n;
    // -FLT_MAX rather than -inf: the merges below take differences of two
    // running maxima, and (-inf) - (-inf) is a nan that would poison the sum.
    float m1 = -3.402823466e+38f, m2 = -3.402823466e+38f;
    float s = 0.f, s2 = 0.f;
    uint32_t i1 = PD_A2_NONE, i2 = PD_A2_NONE;
    for (uint32_t i = tid; i < n; i += nth) {
        const float v = x[i];
        if (v > m1) {
            // the outgoing champion is by construction the new runner-up: it
            // beat everything this thread has seen except the value replacing
            // it. Strict `>` on an ascending walk keeps the LOWEST index at
            // both ranks without a tie test.
            const float d = __expf(m1 - v);
            s = s * d + 1.f;
            s2 = s2 * d * d + 1.f;
            m2 = m1;
            i2 = i1;
            m1 = v;
            i1 = i;
        } else {
            const float e = __expf(v - m1);
            s += e;
            s2 += e * e;
            if (v > m2) {
                m2 = v;
                i2 = i;
            }
        }
    }
    // Merge two (top-2, sum, sum-of-squares) states: rescale the loser's sums
    // onto the winner's max. Each side is already sorted internally, so the
    // merged runner-up is one comparison, never a re-sort of four candidates.
    // Fixed lane order, so the result is run-to-run stable.
#define PD_A2_MERGE(OM1, OI1, OM2, OI2, OS, OS2)                              \
    do {                                                                      \
        const float om1_ = (OM1), om2_ = (OM2), os_ = (OS), os2_ = (OS2);     \
        const uint32_t oi1_ = (OI1), oi2_ = (OI2);                            \
        if (PD_A2_BETTER(om1_, oi1_, m1, i1)) {                               \
            /* their champion takes the row; ours drops into the runner-up   \
               race against theirs (our own m2 lost to our m1 already) */     \
            if (PD_A2_BETTER(m1, i1, om2_, oi2_)) {                           \
                m2 = m1;                                                      \
                i2 = i1;                                                      \
            } else {                                                          \
                m2 = om2_;                                                    \
                i2 = oi2_;                                                    \
            }                                                                 \
            const float d_ = __expf(m1 - om1_);                               \
            s = s * d_ + os_;                                                 \
            s2 = s2 * d_ * d_ + os2_;                                         \
            m1 = om1_;                                                        \
            i1 = oi1_;                                                        \
        } else {                                                              \
            /* ours holds; only their champion can outrank our runner-up,    \
               since their m2 sits below their m1 */                          \
            if (PD_A2_BETTER(om1_, oi1_, m2, i2)) {                           \
                m2 = om1_;                                                    \
                i2 = oi1_;                                                    \
            }                                                                 \
            const float d_ = __expf(om1_ - m1);                               \
            s += os_ * d_;                                                    \
            s2 += os2_ * d_ * d_;                                             \
        }                                                                     \
    } while (0)
#pragma unroll
    for (uint32_t off = 16; off > 0; off >>= 1) {
        PD_A2_MERGE(__shfl_down_sync(0xffffffffu, m1, off),
                    __shfl_down_sync(0xffffffffu, i1, off),
                    __shfl_down_sync(0xffffffffu, m2, off),
                    __shfl_down_sync(0xffffffffu, i2, off),
                    __shfl_down_sync(0xffffffffu, s, off),
                    __shfl_down_sync(0xffffffffu, s2, off));
    }
    __shared__ float sm1[8], sm2[8], ss[8], ss2[8];
    __shared__ uint32_t si1[8], si2[8];
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    if (lane == 0) {
        sm1[warp] = m1;
        sm2[warp] = m2;
        ss[warp] = s;
        ss2[warp] = s2;
        si1[warp] = i1;
        si2[warp] = i2;
    }
    __syncthreads();
    if (tid != 0) return;
    for (uint32_t w = 1; w < ((nth + 31u) >> 5); ++w)
        PD_A2_MERGE(sm1[w], si1[w], sm2[w], si2[w], ss[w], ss2[w]);
#undef PD_A2_MERGE
    // A row whose every logit is exactly -FLT_MAX never promotes anything, so
    // index 0 stands in - which is what a host argmax over an all-equal row
    // returns anyway. Never let the sentinel out as a token id.
    out[row] = (i1 == PD_A2_NONE) ? 0u : i1;
    // n means "there wasn't one" - the same out-of-range convention `probe`
    // uses, so a caller has one rule to remember. Only a 1-token row can hit it.
    if (alt) alt[row] = (i2 == PD_A2_NONE) ? n : i2;
    if (!stats) return;
    // s is already sum(exp(x - m1)), so log p(v) = (v - m1) - log s.
    // logf, not __logf: it runs once per row and the values are user-visible.
    const float lse = (s > 0.f) ? logf(s) : 0.f;
    float* o = stats + (size_t)row * 4u;
    o[0] = (s > 0.f) ? -lse : 0.f;
    o[1] = (probe < n && s > 0.f) ? (__expf(x[probe] - m1) / s) : 0.f;
    o[2] = (i2 != PD_A2_NONE && s > 0.f) ? (m2 - m1 - lse) : 0.f;
    // sum(p^2) = s2 / s^2, so H2 = -log(s2/s^2) = 2 log s - log s2. s2 >= 1
    // always (the max's own term is exactly 1), so the log is safe and H2
    // lands in [0, log n]: 0 for a one-token row, log n for a uniform one.
    o[3] = (s2 > 0.f) ? (2.f * lse - logf(s2)) : 0.f;
}
#undef PD_A2_BETTER
#undef PD_A2_NONE

// 343: greedy pick + the runner-up + the row's confidence readouts, one pass.
//   logits f32 [rows, n]
//   out    u32 [rows]      argmax (same tie rule as pd_argmax_rows)
//   alt    u32 [rows]      the runner-up's id, or n for "none"; NULL to skip
//   stats  f32 [rows, 4]   {log p(top1), p(probe), log p(top2), H2}; NULL to skip
//   probe  token id to report, or >= n for "none" (writes 0)
PD_EXPORT
int pd_argmax_top2_rows(const void* logits, void* out, void* alt, void* stats,
                        uint32_t probe, uint32_t rows, uint32_t n, void* stream) {
    if (rows == 0 || n == 0) return 0;
    pd_argmax_top2_rows_kernel<<<rows, 256, 0, (cudaStream_t)stream>>>(
        (const float*)logits, (unsigned int*)out, (unsigned int*)alt, (float*)stats,
        probe, n);
    return pd_launch_status();
}

// Fused decode-step sampler: one block per logits row, so a batched step reads
// back rows u32 token ids instead of [rows, vocab] f32 logits (25.7 MB/step at
// B=32 on a 201k vocab -- the readback plus host sampling was ~10% of the
// step). Per-row mode: 0 = skip (hole, or a row the host will sample from its
// own readback), 1 = greedy argmax (LOWEST index on ties, matching the host
// argmax's strict-greater ascending scan), 2 = temperature-only categorical:
// p ~ softmax(logits * inv_t), pick the u-quantile walking the vocab in index
// order. Mode 2 mirrors the host sampler's sample_all() including its
// degenerate fallbacks (non-finite max or zero total mass -> argmax); only the
// summation ORDER differs (per-thread chunk sums vs one serial pass), which
// can shift the drawn token only when u lands within float eps of a cumsum
// boundary -- identical distribution, and the seed->token mapping was never a
// stable contract.
typedef struct {
    float inv_t;       // 1 / temperature (mode 2 only)
    float u;           // uniform in [0,1) (mode 2 only)
    unsigned int mode; // 0 skip, 1 greedy, 2 categorical
    unsigned int _pad;
} PdSampleRow;

#define PD_SAMPLE_TPB 256u

__global__ void pd_sample_rows_kernel(const float* __restrict__ logits,
                                      const PdSampleRow* __restrict__ ps,
                                      unsigned int* __restrict__ out, uint32_t n) {
    const uint32_t row = blockIdx.x, tid = threadIdx.x;
    const uint32_t mode = ps[row].mode;
    if (mode != 1u && mode != 2u) return;
    const float* x = logits + (size_t)row * n;

    // pass 1 (both modes): block max, tracking the lowest argmax index --
    // greedy's answer, categorical's softmax shift AND its degenerate fallback
    float bv = -3.402823466e+38f;
    uint32_t bi = 0;
    for (uint32_t i = tid; i < n; i += PD_SAMPLE_TPB) {
        float v = x[i];
        if (v > bv) { bv = v; bi = i; }
    }
#pragma unroll
    for (uint32_t off = 16; off > 0; off >>= 1) {
        float ov = __shfl_down_sync(0xffffffffu, bv, off);
        uint32_t oi = __shfl_down_sync(0xffffffffu, bi, off);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    __shared__ float sv[PD_SAMPLE_TPB / 32u];
    __shared__ uint32_t si[PD_SAMPLE_TPB / 32u];
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    if (lane == 0) { sv[warp] = bv; si[warp] = bi; }
    __syncthreads();
    __shared__ float s_max;
    __shared__ uint32_t s_arg;
    if (tid == 0) {
        for (uint32_t w = 1; w < PD_SAMPLE_TPB / 32u; ++w)
            if (sv[w] > bv || (sv[w] == bv && si[w] < bi)) { bv = sv[w]; bi = si[w]; }
        s_max = bv;
        s_arg = bi;
    }
    __syncthreads();
    if (mode == 1u) {
        if (tid == 0) out[row] = s_arg;
        return;
    }

    const float inv_t = ps[row].inv_t;
    const float m = s_max * inv_t;
    if (!isfinite(m)) { // all -inf (or nan) logits: host falls back to argmax
        if (tid == 0) out[row] = s_arg;
        return;
    }
    // pass 2: per-thread CONTIGUOUS chunk of exp mass. Chunked, not strided,
    // so each partial is a contiguous cumsum segment: the quantile walk below
    // re-accumulates the owner chunk in the same element order.
    const uint32_t chunk = (n + PD_SAMPLE_TPB - 1u) / PD_SAMPLE_TPB;
    const uint32_t lo = tid * chunk;
    const uint32_t hi = min(lo + chunk, n);
    float csum = 0.0f;
    for (uint32_t i = lo; i < hi; ++i) csum += expf(x[i] * inv_t - m);
    __shared__ float ssum[PD_SAMPLE_TPB];
    ssum[tid] = csum;
    __syncthreads();

    // serial scan by thread 0: 256 adds against two O(n) passes -- noise.
    // Owner = first chunk with mass whose inclusive prefix reaches r.
    __shared__ float s_rr;
    __shared__ uint32_t s_owner;
    if (tid == 0) {
        float total = 0.0f;
        for (uint32_t t = 0; t < PD_SAMPLE_TPB; ++t) total += ssum[t];
        if (!(total > 0.0f)) { // all mass underflowed: argmax, like the host
            out[row] = s_arg;
            s_owner = PD_SAMPLE_TPB;
        } else {
            const float r = ps[row].u * total;
            float pre = 0.0f;
            uint32_t own = PD_SAMPLE_TPB;
            float rr = 0.0f;
            for (uint32_t t = 0; t < PD_SAMPLE_TPB; ++t) {
                const float c = ssum[t];
                if (c > 0.0f && pre + c >= r) { own = t; rr = r - pre; break; }
                pre += c;
            }
            if (own == PD_SAMPLE_TPB) { // fp round-off tail: last chunk with mass
                for (uint32_t t = PD_SAMPLE_TPB; t-- > 0u;)
                    if (ssum[t] > 0.0f) { own = t; rr = ssum[t]; break; }
            }
            s_owner = own;
            s_rr = rr;
        }
    }
    __syncthreads();
    if (tid == s_owner) {
        // walk the owner chunk to the quantile: same expf sequence as pass 2,
        // so rr provably crosses zero at or before the chunk's last massive
        // element (up to the subtract-vs-add rounding split, hence the
        // last-with-mass backstop -- exactly the host walk's tail behavior)
        float rr = s_rr;
        uint32_t last = 0xFFFFFFFFu;
        for (uint32_t i = lo; i < hi; ++i) {
            const float e = expf(x[i] * inv_t - m);
            if (e > 0.0f) {
                last = i;
                rr -= e;
                if (rr <= 0.0f) break;
            }
        }
        out[row] = (last != 0xFFFFFFFFu) ? last : s_arg;
    }
}

// Split-phase sampler scratch: the single-block-per-row kernel walks a 201k
// vocab 2-3x alone (141 us/tick for 8 rows in the c8 profile - pure latency
// on a 188-SM die). Phase A fans the max/argmax out over PD_SR_C chunk
// blocks per row; B combines and exp-sums each chunk against the global max;
// C picks the owner chunk from the PD_SR_C partials (ascending chunk order =
// the lowest-index tie rule) and re-walks only that chunk for the quantile.
// Categorical sums regroup (documented non-contract, same distribution);
// greedy keeps the exact lowest-index-tie answer. Static device scratch
// keeps the export ABI unchanged (single stream, no cross-call races).
// 128 chunks (not 32): C's warp-scan walk is latency-bound on its single
// warp, so the owner chunk length (vocab/PD_SR_C) is the tick cost - 201k/128
// = 1.6k elements ~= 50 dependent scan steps ~= 18 us. A/B just get more
// (tiny) blocks, and B's serial 128-partial combine on tid0 stays trivial.
#define PD_SR_C 32u
#define PD_SR_MAXROWS 256u
__device__ float pd_sr_max[PD_SR_MAXROWS * PD_SR_C];
__device__ unsigned int pd_sr_arg[PD_SR_MAXROWS * PD_SR_C];
__device__ float pd_sr_sum[PD_SR_MAXROWS * PD_SR_C];
__device__ float pd_sr_gmax[PD_SR_MAXROWS];
__device__ unsigned int pd_sr_garg[PD_SR_MAXROWS];

__global__ void pd_sample_rows_a_kernel(const float* __restrict__ logits,
                                        const PdSampleRow* __restrict__ ps, uint32_t n) {
    const uint32_t row = blockIdx.x, c = blockIdx.y, tid = threadIdx.x;
    const uint32_t mode = ps[row].mode;
    if (mode != 1u && mode != 2u) return;
    const float* x = logits + (size_t)row * n;
    const uint32_t clen = (n + PD_SR_C - 1u) / PD_SR_C;
    const uint32_t lo = c * clen, hi = min(lo + clen, n);
    float bv = -3.402823466e+38f;
    uint32_t bi = lo;
    for (uint32_t i = lo + tid; i < hi; i += PD_SAMPLE_TPB) {
        float v = x[i];
        if (v > bv) { bv = v; bi = i; }
    }
#pragma unroll
    for (uint32_t off = 16; off > 0; off >>= 1) {
        float ov = __shfl_down_sync(0xffffffffu, bv, off);
        uint32_t oi = __shfl_down_sync(0xffffffffu, bi, off);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    __shared__ float sv[PD_SAMPLE_TPB / 32u];
    __shared__ uint32_t si[PD_SAMPLE_TPB / 32u];
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    if (lane == 0) { sv[warp] = bv; si[warp] = bi; }
    __syncthreads();
    if (tid == 0) {
        for (uint32_t w = 1; w < PD_SAMPLE_TPB / 32u; ++w)
            if (sv[w] > bv || (sv[w] == bv && si[w] < bi)) { bv = sv[w]; bi = si[w]; }
        pd_sr_max[row * PD_SR_C + c] = bv;
        pd_sr_arg[row * PD_SR_C + c] = bi;
    }
}

__global__ void pd_sample_rows_b_kernel(const float* __restrict__ logits,
                                        const PdSampleRow* __restrict__ ps,
                                        unsigned int* __restrict__ out, uint32_t n) {
    const uint32_t row = blockIdx.x, c = blockIdx.y, tid = threadIdx.x;
    const uint32_t mode = ps[row].mode;
    if (mode != 1u && mode != 2u) return;
    __shared__ float s_m;
    if (tid == 0) {
        // combine the chunk maxima in ascending order (lowest-index ties)
        float bv = pd_sr_max[row * PD_SR_C];
        uint32_t bi = pd_sr_arg[row * PD_SR_C];
        for (uint32_t k = 1; k < PD_SR_C; ++k) {
            const float v = pd_sr_max[row * PD_SR_C + k];
            if (v > bv) { bv = v; bi = pd_sr_arg[row * PD_SR_C + k]; }
        }
        if (c == 0) {
            pd_sr_gmax[row] = bv;
            pd_sr_garg[row] = bi;
            if (mode == 1u) out[row] = bi;
        }
        s_m = bv * ps[row].inv_t;
    }
    __syncthreads();
    if (mode == 1u) return;
    const float m = s_m;
    const float inv_t = ps[row].inv_t;
    const float* x = logits + (size_t)row * n;
    const uint32_t clen = (n + PD_SR_C - 1u) / PD_SR_C;
    const uint32_t lo = c * clen, hi = min(lo + clen, n);
    float acc = 0.0f;
    if (isfinite(m))
        for (uint32_t i = lo + tid; i < hi; i += PD_SAMPLE_TPB)
            acc += expf(x[i] * inv_t - m);
#pragma unroll
    for (uint32_t off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    __shared__ float sw[PD_SAMPLE_TPB / 32u];
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    if (lane == 0) sw[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        float v = 0.0f;
        for (uint32_t w = 0; w < PD_SAMPLE_TPB / 32u; ++w) v += sw[w];
        pd_sr_sum[row * PD_SR_C + c] = v;
    }
}

__global__ void pd_sample_rows_c_kernel(const float* __restrict__ logits,
                                        const PdSampleRow* __restrict__ ps,
                                        unsigned int* __restrict__ out, uint32_t n) {
    const uint32_t row = blockIdx.x, tid = threadIdx.x;
    if (ps[row].mode != 2u) return;
    const float inv_t = ps[row].inv_t;
    const float m = pd_sr_gmax[row] * inv_t;
    if (!isfinite(m)) { // degenerate: host-rule argmax fallback
        if (tid == 0) out[row] = pd_sr_garg[row];
        return;
    }
    __shared__ float s_r;
    __shared__ uint32_t s_lo, s_hi;
    if (tid == 0) {
        float total = 0.0f;
        for (uint32_t k = 0; k < PD_SR_C; ++k) total += pd_sr_sum[row * PD_SR_C + k];
        float r = ps[row].u * total;
        if (!(total > 0.0f)) {
            out[row] = pd_sr_garg[row];
            s_lo = 0xFFFFFFFFu;
        } else {
            const uint32_t clen = (n + PD_SR_C - 1u) / PD_SR_C;
            uint32_t k = 0;
            for (; k < PD_SR_C - 1u; ++k) {
                const float cs = pd_sr_sum[row * PD_SR_C + k];
                if (r < cs) break;
                r -= cs;
            }
            s_r = r;
            s_lo = k * clen;
            s_hi = min(s_lo + clen, n);
        }
    }
    __syncthreads();
    if (s_lo == 0xFFFFFFFFu) return;
    // Block-parallel ascending walk of the owner chunk (<= vocab/PD_SR_C
    // elements): 256-wide tiles, warp shuffle scan + cross-warp prefix in
    // shared. Same quantile semantics as the serial walk this replaced -
    // first index whose cumulative mass reaches r, last-positive fallback -
    // the ~6.3k serial expf's (420 us per TICK on the 201k gpt-oss vocab, 3%
    // of the c8 GPU; still 69 us single-warp) become ~25 scan tiles. The
    // scan regroups float adds, fine on the stochastic path (greedy mode
    // never comes here).
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    constexpr uint32_t NW = PD_SAMPLE_TPB / 32u;
    __shared__ float s_wsum[NW];
    __shared__ uint32_t s_pmask[NW]; // per-warp p>0 lane mask
    __shared__ uint32_t s_hit[NW];   // per-warp lowest lane with cum >= r
    const float* x = logits + (size_t)row * n;
    float r = s_r;
    const uint32_t lo = s_lo, hi = s_hi;
    __shared__ uint32_t s_last; // last index with p > 0 seen in prior tiles
    if (tid == 0) s_last = 0xFFFFFFFFu;
    __syncthreads();
    for (uint32_t base = lo; base < hi; base += PD_SAMPLE_TPB) {
        const uint32_t i = base + tid;
        const float p = (i < hi) ? expf(x[i] * inv_t - m) : 0.0f;
        float cum = p;
#pragma unroll
        for (uint32_t off = 1; off < 32; off <<= 1) {
            const float v = __shfl_up_sync(0xffffffffu, cum, off);
            if (lane >= off) cum += v;
        }
        if (lane == 31u) s_wsum[warp] = cum;
        s_pmask[warp] = __ballot_sync(0xffffffffu, p > 0.0f);
        __syncthreads();
        float pre = 0.0f, tile = 0.0f;
#pragma unroll
        for (uint32_t w = 0; w < NW; ++w) {
            if (w < warp) pre += s_wsum[w];
            tile += s_wsum[w];
        }
        cum += pre; // block-inclusive cumulative mass at this thread
        if (r <= tile && tile > 0.0f) {
            // stop tile: lowest tid whose cumulative reaches r; its p is > 0
            // (cum strictly jumps there) except at the r <= 0 edge, where the
            // last positive index before it (or a prior tile's, or the
            // argmax fallback) stands in.
            const uint32_t hitw = __ballot_sync(0xffffffffu, cum >= r);
            s_hit[warp] = hitw ? (uint32_t)__ffs(hitw) - 1u : 32u;
            __syncthreads();
            if (tid == 0) {
                uint32_t hw = NW, hl = 32u;
                for (uint32_t w = 0; w < NW; ++w)
                    if (s_hit[w] < 32u) { hw = w; hl = s_hit[w]; break; }
                if (hw == NW) { hw = NW - 1u; hl = 31u; } // unreachable guard
                uint32_t last = s_last;
                for (int32_t w = (int32_t)hw; w >= 0; --w) {
                    uint32_t msk = s_pmask[w];
                    if ((uint32_t)w == hw)
                        msk &= (hl == 31u) ? 0xffffffffu : ((2u << hl) - 1u);
                    if (msk) { last = base + (uint32_t)w * 32u + (31u - __clz(msk)); break; }
                }
                out[row] = (last != 0xFFFFFFFFu) ? last : pd_sr_garg[row];
            }
            return;
        }
        r -= tile;
        if (tid == 0) {
            for (int32_t w = NW - 1; w >= 0; --w)
                if (s_pmask[w]) { s_last = base + (uint32_t)w * 32u + (31u - __clz(s_pmask[w])); break; }
        }
        __syncthreads(); // s_wsum/s_pmask reuse + s_last visibility
    }
    if (tid == 0) out[row] = (s_last != 0xFFFFFFFFu) ? s_last : pd_sr_garg[row];
}

PD_EXPORT
int pd_sample_rows(const void* logits, const void* params, void* out,
                   uint32_t rows, uint32_t n, void* stream) {
    if (rows == 0 || n == 0) return 0;
    auto st = (cudaStream_t)stream;
    if (rows <= PD_SR_MAXROWS && n >= 32768u) {
        dim3 gab(rows, PD_SR_C);
        pd_sample_rows_a_kernel<<<gab, PD_SAMPLE_TPB, 0, st>>>(
            (const float*)logits, (const PdSampleRow*)params, n);
        pd_sample_rows_b_kernel<<<gab, PD_SAMPLE_TPB, 0, st>>>(
            (const float*)logits, (const PdSampleRow*)params, (unsigned int*)out, n);
        pd_sample_rows_c_kernel<<<rows, PD_SAMPLE_TPB, 0, st>>>(
            (const float*)logits, (const PdSampleRow*)params, (unsigned int*)out, n);
        return pd_launch_status();
    }
    pd_sample_rows_kernel<<<rows, PD_SAMPLE_TPB, 0, st>>>(
        (const float*)logits, (const PdSampleRow*)params, (unsigned int*)out, n);
    return pd_launch_status();
}

// ── device top-K prefilter for HOST-HEAD sampling rows ───────────────────
// A qwen3.8 B200 attribution found 54% of the c32 wall in one
// seam: every row whose sampler carries a truncation filter (top-k/top-p -
// qwen3.x's PUBLISHED defaults) fails is_device_plannable, so the host read
// back the full 993KB logits row per slot per round (9.6 GB / 20 s at c32)
// and ran nucleus sampling on the CPU - 21.3 ms of GPU idle per round.
// This kernel runs the SELECTION on device instead: rows marked mode 4 in
// the shared PdSampleRow plane (the sample_rows family skips any mode it
// doesn't know) get their top-K (id, raw logit) pairs written compactly;
// the host then runs its EXISTING nucleus pipeline (sampler.rs
// build_nucleus semantics, unit-tested) over K entries instead of the
// vocab. Exact by construction for penalty-free rows with top_k <= K: the
// K-head is a superset of any top_k head, selection order is f32 total
// order (matches the host's total_cmp on scaled logits - inv_t > 0 is
// monotonic), and k-boundary TIE CHOICE was never contractual
// (select_nth_unstable is arbitrary there too; here ties resolve by
// ascending index, capped at PD_TOPK_CAP collected ties).
#define PD_TOPK_HEAD 64u
#define PD_TOPK_CAP 128u

__device__ __forceinline__ uint32_t pd_okey(float f) {
    // total-order key: monotonic uint mapping of f32 total order
    uint32_t b = __float_as_uint(f);
    return b ^ ((b & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u);
}

__global__ void pd_topk_rows_kernel(const float* __restrict__ logits,
                                    const PdSampleRow* __restrict__ ps,
                                    unsigned int* __restrict__ out,
                                    uint32_t n, uint32_t k) {
    const uint32_t row = blockIdx.x, tid = threadIdx.x;
    if (ps[row].mode != 4u) return;
    const float* x = logits + (size_t)row * n;
    const uint32_t K = min(k, n);
    // Two-level 11-bit histogram threshold (rewrite): the 32-pass
    // binary search read 33 x 993KB per row and measured 1.64 ms/round at
    // c32 - 12.8% of the whole decode window. This form does three passes:
    // (A) 2048-bin histogram of the top 11 okey bits -> boundary bin b1
    // with count-above < K; (B) 2048-bin histogram of the next 11 bits
    // within b1 -> boundary sub-bin b2; (C) gather. Strict set G = keys
    // whose top-22 bits exceed (b1,b2): |G| < K <= 64 by construction.
    // Boundary set E = the (b1,b2) bucket, gathered with values and
    // value-sorted before filling K - |G| - so the emitted head is top-K
    // by VALUE exactly (a coarse bucket cannot displace a larger value),
    // and exact-equal ties keep the arbitrary-choice class the host's
    // select_nth_unstable has. E is capped at PD_TOPK_ECAP; >ECAP values
    // sharing 22 top bits at the boundary is a degenerate distribution
    // (ties-class, documented).
    __shared__ uint32_t s_hist[2048];
    __shared__ uint32_t s_b1, s_b2, s_cg;
    for (uint32_t i = tid; i < 2048u; i += blockDim.x) s_hist[i] = 0;
    __syncthreads();
    for (uint32_t i = tid; i < n; i += blockDim.x)
        atomicAdd(&s_hist[pd_okey(x[i]) >> 21], 1u);
    __syncthreads();
    if (tid == 0) {
        uint32_t above = 0, b = 2047;
        for (;; --b) {
            if (above + s_hist[b] >= K || b == 0) break;
            above += s_hist[b];
        }
        s_b1 = b;
        s_cg = above; // strict count from level 1
    }
    __syncthreads();
    const uint32_t b1 = s_b1;
    const uint32_t cg1 = s_cg;
    for (uint32_t i = tid; i < 2048u; i += blockDim.x) s_hist[i] = 0;
    __syncthreads();
    for (uint32_t i = tid; i < n; i += blockDim.x) {
        const uint32_t key = pd_okey(x[i]);
        if ((key >> 21) == b1) atomicAdd(&s_hist[(key >> 10) & 0x7FFu], 1u);
    }
    __syncthreads();
    if (tid == 0) {
        uint32_t above = cg1, b = 2047;
        for (;; --b) {
            if (above + s_hist[b] >= K || b == 0) break;
            above += s_hist[b];
        }
        s_b2 = b;
        s_cg = above; // strict count at 22-bit granularity, < K
    }
    __syncthreads();
    const uint32_t bfloor = (b1 << 11) | s_b2; // top-22-bit boundary bucket
#define PD_TOPK_ECAP 256u
    __shared__ uint32_t s_gids[PD_TOPK_HEAD];
    __shared__ uint32_t s_eids[PD_TOPK_ECAP];
    __shared__ uint32_t s_evals[PD_TOPK_ECAP];
    __shared__ uint32_t s_gn, s_en;
    if (tid == 0) { s_gn = 0; s_en = 0; }
    __syncthreads();
    for (uint32_t i = tid; i < n; i += blockDim.x) {
        const uint32_t key = pd_okey(x[i]);
        const uint32_t top22 = key >> 10;
        if (top22 > bfloor) {
            const uint32_t p = atomicAdd(&s_gn, 1u);
            if (p < PD_TOPK_HEAD) s_gids[p] = i; // |G| < K <= 64 guaranteed
        } else if (top22 == bfloor) {
            const uint32_t p = atomicAdd(&s_en, 1u);
            if (p < PD_TOPK_ECAP) {
                s_eids[p] = i;
                s_evals[p] = key;
            }
        }
    }
    __syncthreads();
    if (tid != 0) return;
    const uint32_t gn = min(s_gn, PD_TOPK_HEAD);
    const uint32_t en = min(s_en, PD_TOPK_ECAP);
    uint32_t cnt = 0;
    for (uint32_t a = 0; a < gn && cnt < K; ++a, ++cnt) {
        out[((size_t)row * k + cnt) * 2u] = s_gids[a];
        out[((size_t)row * k + cnt) * 2u + 1u] = __float_as_uint(x[s_gids[a]]);
    }
    // boundary bucket: selection by (okey desc, index asc) - top-by-value
    // exact; equal keys resolve to the lowest index (deterministic)
    uint32_t taken = 0;
    while (cnt < K && taken < en) {
        // consumed marker lives in s_eids (a real id is < n << 2^32-1;
        // an okey can be 0xFFFFFFFF - NaN logits - so values can't mark)
        uint32_t best = 0xFFFFFFFFu, bkey = 0, bid = 0xFFFFFFFFu;
        for (uint32_t a = 0; a < en; ++a) {
            if (s_eids[a] == 0xFFFFFFFFu) continue;
            if (best == 0xFFFFFFFFu || s_evals[a] > bkey
                || (s_evals[a] == bkey && s_eids[a] < bid)) {
                best = a;
                bkey = s_evals[a];
                bid = s_eids[a];
            }
        }
        if (best == 0xFFFFFFFFu) break;
        s_eids[best] = 0xFFFFFFFFu;
        out[((size_t)row * k + cnt) * 2u] = bid;
        out[((size_t)row * k + cnt) * 2u + 1u] = __float_as_uint(x[bid]);
        ++cnt;
        ++taken;
    }
    for (; cnt < k; ++cnt) { // n < k edge / truncated-bucket pad
        out[((size_t)row * k + cnt) * 2u] = 0xFFFFFFFFu;
        out[((size_t)row * k + cnt) * 2u + 1u] = __float_as_uint(-INFINITY);
    }
}

// pd_topk_rows PD_EXPORT lives below the P71b multi-block machinery (it
// dispatches to it; C++ needs the definitions first).

// ── FULL-DEVICE truncation sampling, mode 5 ──────────────────────────────
// Same two-level histogram head build as pd_topk_rows, then thread 0
// finishes the draw on device - a verbatim port of the host
// sample_trunc_head pipeline (sampler.rs): scale by inv_t, sort the head
// desc by scaled value (f32 total order; equal values -> lowest index),
// truncate to k, m = head max (the row argmax is in the head), exp(c-m),
// head-sum denominator (top_k>0 semantics), min_p take-while, top_p
// INCLUSIVE cum walk, then the renormalized u·total walk. Token lands in
// out[row] exactly like modes 1/2 - no head readback, no host tail, and
// the row becomes zero-host (pipe/overlap admissible). Distribution class:
// identical pipeline; expf vs the host's f32::exp can differ by 1 ulp so a
// u landing within eps of a cum boundary may pick the adjacent survivor -
// the documented mode-2 class ("identical distribution; the seed->token
// mapping is not a contract").
// Per-row trunc params ride a side plane pt[row] = {k, top_p bits,
// min_p bits, pad} - f32-exact (not f16-packed: a 3e-4 top_p shift would
// be a systematic distribution change, the wide-nth refusal class).
typedef struct {
    unsigned int k;
    float top_p;
    float min_p;
    unsigned int _pad;
} PdSampleTrunc;

// __launch_bounds__ is LOAD-BEARING, not a hint: the launcher below runs this
// at 1024 threads, and an SM has 65536 registers, so anything over 64 regs per
// thread makes the block unlaunchable and the driver answers 701
// (LAUNCH_OUT_OF_RESOURCES). Without the bound nvcc allocated 80 on sm_86 and
// the mode-5 fallback arm could not run at all  -- and since the
// 65536-registers-per-SM budget is the same on every arch we ship, nothing
// protected the other five either. Every other 1024-thread kernel in this file
// carries the same annotation; this one was the omission.
__global__ void __launch_bounds__(1024)
pd_sample_rows_t_kernel(const float* __restrict__ logits,
                        const PdSampleRow* __restrict__ ps,
                        const PdSampleTrunc* __restrict__ pt,
                        unsigned int* __restrict__ out,
                        uint32_t n) {
    const uint32_t row = blockIdx.x, tid = threadIdx.x;
    if (ps[row].mode != 5u) return;
    const float* x = logits + (size_t)row * n;
    const uint32_t k = min(max(pt[row].k, 1u), PD_TOPK_HEAD);
    const uint32_t K = min(k, n);

    // head build: identical structure to pd_topk_rows_kernel
    __shared__ uint32_t s_hist[2048];
    __shared__ uint32_t s_b1, s_b2, s_cg;
    for (uint32_t i = tid; i < 2048u; i += blockDim.x) s_hist[i] = 0;
    __syncthreads();
    for (uint32_t i = tid; i < n; i += blockDim.x)
        atomicAdd(&s_hist[pd_okey(x[i]) >> 21], 1u);
    __syncthreads();
    if (tid == 0) {
        uint32_t above = 0, b = 2047;
        for (;; --b) {
            if (above + s_hist[b] >= K || b == 0) break;
            above += s_hist[b];
        }
        s_b1 = b;
        s_cg = above;
    }
    __syncthreads();
    const uint32_t b1 = s_b1;
    const uint32_t cg1 = s_cg;
    for (uint32_t i = tid; i < 2048u; i += blockDim.x) s_hist[i] = 0;
    __syncthreads();
    for (uint32_t i = tid; i < n; i += blockDim.x) {
        const uint32_t key = pd_okey(x[i]);
        if ((key >> 21) == b1) atomicAdd(&s_hist[(key >> 10) & 0x7FFu], 1u);
    }
    __syncthreads();
    if (tid == 0) {
        uint32_t above = cg1, b = 2047;
        for (;; --b) {
            if (above + s_hist[b] >= K || b == 0) break;
            above += s_hist[b];
        }
        s_b2 = b;
    }
    __syncthreads();
    const uint32_t bfloor = (b1 << 11) | s_b2;
    __shared__ uint32_t s_ids[PD_TOPK_HEAD + PD_TOPK_ECAP];
    __shared__ uint32_t s_keys[PD_TOPK_HEAD + PD_TOPK_ECAP];
    __shared__ uint32_t s_gn, s_en;
    if (tid == 0) { s_gn = 0; s_en = 0; }
    __syncthreads();
    for (uint32_t i = tid; i < n; i += blockDim.x) {
        const uint32_t key = pd_okey(x[i]);
        const uint32_t top22 = key >> 10;
        if (top22 > bfloor) {
            const uint32_t p = atomicAdd(&s_gn, 1u);
            if (p < PD_TOPK_HEAD) { s_ids[p] = i; s_keys[p] = key; }
        } else if (top22 == bfloor) {
            const uint32_t p = atomicAdd(&s_en, 1u);
            if (p < PD_TOPK_ECAP) {
                s_ids[PD_TOPK_HEAD + p] = i;
                s_keys[PD_TOPK_HEAD + p] = key;
            }
        }
    }
    __syncthreads();
    if (tid != 0) return;

    // assemble candidate list: G (all of it, < K) + boundary bucket, then
    // one selection order: (okey desc, index asc) - okey order == scaled-
    // value total order (inv_t > 0), matching the host sort + tie choice
    const uint32_t gn = min(s_gn, (uint32_t)PD_TOPK_HEAD);
    const uint32_t en = min(s_en, (uint32_t)PD_TOPK_ECAP);
    uint32_t cid[PD_TOPK_HEAD];
    float cval[PD_TOPK_HEAD];
    uint32_t cn = 0;
    // selection loop over the union (gn + en <= 64 + 256): pick top K
    uint32_t used_mark = 0xFFFFFFFFu;
    for (; cn < K; ++cn) {
        uint32_t best = used_mark, bkey = 0, bid = used_mark;
        for (uint32_t a = 0; a < gn + en; ++a) {
            const uint32_t idx = a < gn ? a : PD_TOPK_HEAD + (a - gn);
            if (s_ids[idx] == used_mark) continue;
            if (best == used_mark || s_keys[idx] > bkey
                || (s_keys[idx] == bkey && s_ids[idx] < bid)) {
                best = idx;
                bkey = s_keys[idx];
                bid = s_ids[idx];
            }
        }
        if (best == used_mark) break; // n < K edge
        s_ids[best] = used_mark;
        cid[cn] = bid;
        cval[cn] = x[bid];
    }
    if (cn == 0) { out[row] = 0; return; }

    // host sample_trunc_head, verbatim: scaled desc order is cval order
    const float inv_t = ps[row].inv_t;
    const float m = cval[0] * inv_t;
    if (!isfinite(m)) { out[row] = cid[0]; return; }
    float p[PD_TOPK_HEAD];
    float head_sum = 0.0f;
    for (uint32_t a = 0; a < cn; ++a) {
        p[a] = expf(cval[a] * inv_t - m);
        head_sum += p[a];
    }
    if (!(head_sum > 0.0f)) { out[row] = cid[0]; return; }
    for (uint32_t a = 0; a < cn; ++a) p[a] /= head_sum;
    uint32_t keep = cn;
    const float min_p = pt[row].min_p;
    if (min_p > 0.0f) {
        const float thresh = min_p * p[0];
        uint32_t s = 0;
        while (s < cn && p[s] >= thresh) ++s;
        keep = s;
    }
    const float top_p = pt[row].top_p;
    if (top_p < 1.0f) {
        float cum = 0.0f;
        uint32_t kp = keep;
        for (uint32_t a = 0; a < keep; ++a) {
            cum += p[a];
            if (cum >= top_p) { kp = a + 1u; break; }
        }
        keep = kp;
    }
    if (keep == 0u) keep = 1u;
    float total = 0.0f;
    for (uint32_t a = 0; a < keep; ++a) total += p[a];
    float r = ps[row].u * total;
    uint32_t pick = cid[keep - 1u];
    for (uint32_t a = 0; a < keep; ++a) {
        r -= p[a];
        if (r <= 0.0f) { pick = cid[a]; break; }
    }
    out[row] = pick;
}

// ── mode-5 chunked build (the c8 fix) ────────────────────────────────────
// The one-block-per-row head build above is DRAM-latency-bound: at c8 it is
// 8 blocks on a 188-SM die, each streaming its full 604 KB logits row at
// single-block bandwidth - 226 us/launch in the c8 profile, 0.9% of the
// step, and it dominated the qwen3.8-27b c8 cell. Same
// disease the categorical chain already cured with its a/b/c split. Cure is
// the same shape: fan the row scan out over PD_T2_C chunk blocks, each
// finding its chunk-local top-64 exactly (the global top-K of K <= 64 is a
// subset of the union of chunk-local top-64s), then one small finisher
// block per row merges 2048 candidates and runs the verbatim trunc tail.
// Static device scratch keeps the export ABI unchanged (single stream, no
// cross-call races - the pd_sr_* precedent above). Candidates travel as
// (okey << 32 | ~id) composites: u64 desc order == (value desc, index asc),
// the host tie rule, so the merge needs no tie logic at all. Per-chunk
// boundary buckets keep the ECAP ties-class of the one-block kernel
// (degenerate distributions only, documented there).
#define PD_T2_C 32u
#define PD_T2_ECAP 256u
__device__ unsigned long long pd_t2_cand[PD_SR_MAXROWS * PD_T2_C * PD_TOPK_HEAD];

__device__ __forceinline__ void pd_t2_bitonic(unsigned long long* v,
                                              uint32_t len, uint32_t tid,
                                              uint32_t nthreads) {
    // descending bitonic sort, len a power of two, one CE per thread per step
    for (uint32_t k = 2; k <= len; k <<= 1) {
        for (uint32_t j = k >> 1; j > 0; j >>= 1) {
            for (uint32_t i = tid; i < len; i += nthreads) {
                const uint32_t p = i ^ j;
                if (p > i) {
                    const bool up = (i & k) == 0u; // descending region
                    unsigned long long a = v[i], b = v[p];
                    if ((a < b) == up) { v[i] = b; v[p] = a; }
                }
            }
            __syncthreads();
        }
    }
}

__global__ void __launch_bounds__(256) pd_sample_rows_t2a_kernel(
    const float* __restrict__ logits, const PdSampleRow* __restrict__ ps,
    uint32_t n) {
    const uint32_t row = blockIdx.x, c = blockIdx.y, tid = threadIdx.x;
    if (ps[row].mode != 5u) return;
    const float* x = logits + (size_t)row * n;
    const uint32_t clen = (n + PD_T2_C - 1u) / PD_T2_C;
    const uint32_t lo = c * clen, hi = min(lo + clen, n);
    unsigned long long* cand =
        pd_t2_cand + ((size_t)row * PD_T2_C + c) * PD_TOPK_HEAD;
    if (lo >= hi) { // short-tail empty chunk: all-sentinel, whole block exits
        for (uint32_t j = tid; j < PD_TOPK_HEAD; j += blockDim.x) cand[j] = 0ull;
        return;
    }
    // chunk-local head, same two-level 11-bit histogram as the one-block
    // kernel, K' = 64 regardless of the row's k (superset for any K <= 64)
    __shared__ uint32_t s_hist[2048];
    __shared__ uint32_t s_b1, s_b2, s_cg;
    for (uint32_t i = tid; i < 2048u; i += blockDim.x) s_hist[i] = 0;
    __syncthreads();
    for (uint32_t i = lo + tid; i < hi; i += blockDim.x)
        atomicAdd(&s_hist[pd_okey(x[i]) >> 21], 1u);
    __syncthreads();
    if (tid == 0) {
        uint32_t above = 0, b = 2047;
        for (;; --b) {
            if (above + s_hist[b] >= PD_TOPK_HEAD || b == 0) break;
            above += s_hist[b];
        }
        s_b1 = b;
        s_cg = above;
    }
    __syncthreads();
    const uint32_t b1 = s_b1;
    const uint32_t cg1 = s_cg;
    for (uint32_t i = tid; i < 2048u; i += blockDim.x) s_hist[i] = 0;
    __syncthreads();
    for (uint32_t i = lo + tid; i < hi; i += blockDim.x) {
        const uint32_t key = pd_okey(x[i]);
        if ((key >> 21) == b1) atomicAdd(&s_hist[(key >> 10) & 0x7FFu], 1u);
    }
    __syncthreads();
    if (tid == 0) {
        uint32_t above = cg1, b = 2047;
        for (;; --b) {
            if (above + s_hist[b] >= PD_TOPK_HEAD || b == 0) break;
            above += s_hist[b];
        }
        s_b2 = b;
    }
    __syncthreads();
    const uint32_t bfloor = (b1 << 11) | s_b2;
    // gather strict G (< 64 by construction) + boundary bucket E (ECAP) as
    // composites, zero-padded, sort desc, emit the chunk-local top-64
    __shared__ unsigned long long s_comp[PD_TOPK_HEAD + PD_T2_ECAP + 192u]; // 512
    __shared__ uint32_t s_gn, s_en;
    for (uint32_t i = tid; i < 512u; i += blockDim.x) s_comp[i] = 0ull;
    if (tid == 0) { s_gn = 0; s_en = 0; }
    __syncthreads();
    for (uint32_t i = lo + tid; i < hi; i += blockDim.x) {
        const uint32_t key = pd_okey(x[i]);
        const uint32_t top22 = key >> 10;
        if (top22 > bfloor) {
            const uint32_t p = atomicAdd(&s_gn, 1u);
            if (p < PD_TOPK_HEAD)
                s_comp[p] = ((unsigned long long)key << 32) | (uint32_t)~i;
        } else if (top22 == bfloor) {
            const uint32_t p = atomicAdd(&s_en, 1u);
            if (p < PD_T2_ECAP)
                s_comp[PD_TOPK_HEAD + p] =
                    ((unsigned long long)key << 32) | (uint32_t)~i;
        }
    }
    __syncthreads();
    pd_t2_bitonic(s_comp, 512u, tid, blockDim.x);
    for (uint32_t j = tid; j < PD_TOPK_HEAD; j += blockDim.x) cand[j] = s_comp[j];
}

__global__ void __launch_bounds__(1024) pd_sample_rows_t2f_kernel(
    const float* __restrict__ logits, const PdSampleRow* __restrict__ ps,
    const PdSampleTrunc* __restrict__ pt, unsigned int* __restrict__ out,
    uint32_t n) {
    const uint32_t row = blockIdx.x, tid = threadIdx.x;
    if (ps[row].mode != 5u) return;
    const float* x = logits + (size_t)row * n;
    const uint32_t k = min(max(pt[row].k, 1u), PD_TOPK_HEAD);
    const uint32_t K = min(k, n);
    __shared__ unsigned long long s_comp[PD_T2_C * PD_TOPK_HEAD]; // 2048
    const unsigned long long* cand =
        pd_t2_cand + (size_t)row * PD_T2_C * PD_TOPK_HEAD;
    for (uint32_t i = tid; i < PD_T2_C * PD_TOPK_HEAD; i += blockDim.x)
        s_comp[i] = cand[i];
    __syncthreads();
    pd_t2_bitonic(s_comp, PD_T2_C * PD_TOPK_HEAD, tid, blockDim.x);
    if (tid != 0) return;
    // top-K head is the first K non-sentinel composites; tail below is the
    // one-block kernel's verbatim (host sample_trunc_head port)
    uint32_t cid[PD_TOPK_HEAD];
    float cval[PD_TOPK_HEAD];
    uint32_t cn = 0;
    for (; cn < K; ++cn) {
        const unsigned long long comp = s_comp[cn];
        if (comp == 0ull) break;
        cid[cn] = (uint32_t)~(uint32_t)(comp & 0xFFFFFFFFull);
        cval[cn] = x[cid[cn]];
    }
    if (cn == 0) { out[row] = 0; return; }
    const float inv_t = ps[row].inv_t;
    const float m = cval[0] * inv_t;
    if (!isfinite(m)) { out[row] = cid[0]; return; }
    float p[PD_TOPK_HEAD];
    float head_sum = 0.0f;
    for (uint32_t a = 0; a < cn; ++a) {
        p[a] = expf(cval[a] * inv_t - m);
        head_sum += p[a];
    }
    if (!(head_sum > 0.0f)) { out[row] = cid[0]; return; }
    for (uint32_t a = 0; a < cn; ++a) p[a] /= head_sum;
    uint32_t keep = cn;
    const float min_p = pt[row].min_p;
    if (min_p > 0.0f) {
        const float thresh = min_p * p[0];
        uint32_t s = 0;
        while (s < cn && p[s] >= thresh) ++s;
        keep = s;
    }
    const float top_p = pt[row].top_p;
    if (top_p < 1.0f) {
        float cum = 0.0f;
        uint32_t kp = keep;
        for (uint32_t a = 0; a < keep; ++a) {
            cum += p[a];
            if (cum >= top_p) { kp = a + 1u; break; }
        }
        keep = kp;
    }
    if (keep == 0u) keep = 1u;
    float total = 0.0f;
    for (uint32_t a = 0; a < keep; ++a) total += p[a];
    float r = ps[row].u * total;
    uint32_t pick = cid[keep - 1u];
    for (uint32_t a = 0; a < keep; ++a) {
        r -= p[a];
        if (r <= 0.0f) { pick = cid[a]; break; }
    }
    out[row] = pick;
}


// ── P71b: multi-block head build (modes 4/5), v2 two-level ───────────────
// The one-block-per-row head build is a per-CTA DRAM-latency wall: 162 us
// at rows=1 rising only to 219 at rows=32 (three full 993KB scans from
// 8-32 CTAs on a 148-SM die). Split each row's scans across SEG blocks:
// K1 merges per-segment level-1 (top-11-bit) histograms; K1b merges the
// level-2 histogram within the boundary bin b1; K2 re-derives (b1, b2)
// per block (parallel suffix scans, ~2us each) and gathers the strict-
// above set G plus the 22-BIT boundary bucket E into global lists; K3
// bitonic-sorts both into canonical (okey desc, id asc) order and emits
// the head (mode 4) or runs the draw tail (mode 5, verbatim port of the
// one-block tail). No G/E merge is needed: every G key's top-22 bits
// exceed bfloor and every E key's equal it.
//
// v2 lesson (c32 regression): the v1 single-level gather
// bounded E at 11-bit granularity - Real decode rows include flat
// (high-entropy) distributions whose boundary bucket holds thousands, so
// v1's in-block full-row fallback (~170us serial while 31 blocks idle)
// fired per ROW and c32 paid ~2 rows/tick (+0.33 ms itl, +99 ms ttft
// bisect). At 22-bit granularity a flat row's bucket is ~n/4M-dense
// (~121 avg at the serve vocab) and E-overflow collapses to the legacy
// ties class - handled by TRUNCATION exactly like the one-block kernel's
// ECAP (ours 512 vs legacy 256, strictly roomier), so the fallback and
// its cliff are DELETED, at every width.
//
// Head is top-K by VALUE exactly while |E22| <= 512. Scratch is
// self-cleaning (K3 zeroes its row's histograms + counters), so graph
// replays never need a memset; it is allocated at f8-plane repack (load
// time) or lazily by the wrapper when the stream is not capturing
// (cudaMalloc invalidates capture). Falls back to the one-block kernels
// when scratch is unavailable. Kill: PADDOCK_NO_TOPK_MB.
#define PD_TOPK_MB_SEG 32u
#define PD_TOPK_MB_ECAP 512u
#define PD_TOPK_MB_MAXR 64u
// per-row u32 scratch layout:
//   [0..2047] hist1 | [2048..4095] hist2 | [4096] gcnt | [4097] ecnt |
//   [4098..4161] gids | [4162..4225] gkeys | [4226..4737] eids |
//   [4738..5249] ekeys
#define PD_TOPK_MB_H2 2048u
#define PD_TOPK_MB_GC 4096u
#define PD_TOPK_MB_EC 4097u
#define PD_TOPK_MB_GI 4098u
#define PD_TOPK_MB_GK 4162u
#define PD_TOPK_MB_EI 4226u
#define PD_TOPK_MB_EK 4738u
#define PD_TOPK_MB_STRIDE 5312u

static uint32_t* pd_smp_scr(bool make = false) {
    static uint32_t* scr = nullptr;
    if (make && !scr) {
        const size_t bytes = (size_t)PD_TOPK_MB_MAXR * PD_TOPK_MB_STRIDE * 4u;
        if (cudaMalloc(&scr, bytes) != cudaSuccess) scr = nullptr;
        else cudaMemset(scr, 0, bytes);
    }
    return scr;
}

template <int MODE>
__global__ void __launch_bounds__(1024) pd_topk_mb_hist_kernel(
    const float* __restrict__ logits, const PdSampleRow* __restrict__ ps,
    uint32_t* __restrict__ scr, uint32_t n) {
    const uint32_t row = blockIdx.x, seg = blockIdx.y, tid = threadIdx.x;
    if (ps[row].mode != (uint32_t)MODE) return;
    const float* x = logits + (size_t)row * n;
    const uint32_t seglen = (n + gridDim.y - 1u) / gridDim.y;
    const uint32_t i0 = seg * seglen, i1 = min(n, i0 + seglen);
    __shared__ uint32_t s_hist[2048];
    for (uint32_t i = tid; i < 2048u; i += blockDim.x) s_hist[i] = 0;
    __syncthreads();
    for (uint32_t i = i0 + tid; i < i1; i += blockDim.x)
        atomicAdd(&s_hist[pd_okey(x[i]) >> 21], 1u);
    __syncthreads();
    uint32_t* h = scr + (size_t)row * PD_TOPK_MB_STRIDE;
    for (uint32_t i = tid; i < 2048u; i += blockDim.x)
        if (s_hist[i]) atomicAdd(&h[i], s_hist[i]);
}

// parallel boundary-bin from a merged 2048-bin histogram: T[b] = suffix-
// inclusive counts (Hillis-Steele scan into s_t); returns max{b : T[b] >=
// thresh} - exactly the serial top-down walk's stop (0 when never met,
// the b==0 stop). s_t holds T[] on return (caller may read, then must
// barrier before any reuse).
__device__ __forceinline__ uint32_t pd_topk_mb_bsel(const uint32_t* h,
                                                    uint32_t* s_t,
                                                    uint32_t* s_b,
                                                    uint32_t thresh,
                                                    uint32_t tid) {
    for (uint32_t i = tid; i < 2048u; i += 1024u) s_t[i] = h[i];
    if (tid == 0) *s_b = 0u;
    __syncthreads();
    for (uint32_t off = 1u; off < 2048u; off <<= 1) {
        const uint32_t a0 = tid, a1 = tid + 1024u;
        uint32_t v0 = 0, v1 = 0;
        if (a0 + off < 2048u) v0 = s_t[a0 + off];
        if (a1 + off < 2048u) v1 = s_t[a1 + off];
        __syncthreads();
        s_t[a0] += v0;
        s_t[a1] += v1;
        __syncthreads();
    }
    for (uint32_t i = tid; i < 2048u; i += 1024u)
        if (s_t[i] >= thresh) atomicMax(s_b, i);
    __syncthreads();
    return *s_b;
}

// K1b: level-2 histogram of the boundary bin b1 (next 11 okey bits).
template <int MODE>
__global__ void __launch_bounds__(1024) pd_topk_mb_hist2_kernel(
    const float* __restrict__ logits, const PdSampleRow* __restrict__ ps,
    const PdSampleTrunc* __restrict__ pt, uint32_t* __restrict__ scr,
    uint32_t n, uint32_t k) {
    const uint32_t row = blockIdx.x, seg = blockIdx.y, tid = threadIdx.x;
    if (ps[row].mode != (uint32_t)MODE) return;
    uint32_t kk = k;
    if (MODE == 5) kk = min(max(pt[row].k, 1u), (uint32_t)PD_TOPK_HEAD);
    const uint32_t K = min(kk, n);
    uint32_t* h = scr + (size_t)row * PD_TOPK_MB_STRIDE;
    __shared__ uint32_t s_t[2048];
    __shared__ uint32_t s_b1;
    const uint32_t b1 = pd_topk_mb_bsel(h, s_t, &s_b1, K, tid);
    __syncthreads(); // T[] reads done; s_t reused as the local histogram
    for (uint32_t i = tid; i < 2048u; i += blockDim.x) s_t[i] = 0;
    __syncthreads();
    const uint32_t seglen = (n + gridDim.y - 1u) / gridDim.y;
    const uint32_t i0 = seg * seglen, i1 = min(n, i0 + seglen);
    const float* x = logits + (size_t)row * n;
    for (uint32_t i = i0 + tid; i < i1; i += blockDim.x) {
        const uint32_t key = pd_okey(x[i]);
        if ((key >> 21) == b1) atomicAdd(&s_t[(key >> 10) & 0x7FFu], 1u);
    }
    __syncthreads();
    for (uint32_t i = tid; i < 2048u; i += blockDim.x)
        if (s_t[i]) atomicAdd(&h[PD_TOPK_MB_H2 + i], s_t[i]);
}

template <int MODE>
__global__ void __launch_bounds__(1024) pd_topk_mb_gather_kernel(
    const float* __restrict__ logits, const PdSampleRow* __restrict__ ps,
    const PdSampleTrunc* __restrict__ pt, uint32_t* __restrict__ scr,
    uint32_t n, uint32_t k) {
    const uint32_t row = blockIdx.x, seg = blockIdx.y, tid = threadIdx.x;
    if (ps[row].mode != (uint32_t)MODE) return;
    uint32_t kk = k;
    if (MODE == 5) kk = min(max(pt[row].k, 1u), (uint32_t)PD_TOPK_HEAD);
    const uint32_t K = min(kk, n);
    uint32_t* h = scr + (size_t)row * PD_TOPK_MB_STRIDE;
    __shared__ uint32_t s_t[2048];
    __shared__ uint32_t s_b;
    const uint32_t b1 = pd_topk_mb_bsel(h, s_t, &s_b, K, tid);
    // cg1 = strict-above count at 11 bits (T[b1+1]); every thread reads
    // the same value, then the helper's next scan may reuse s_t
    const uint32_t cg1 = b1 < 2047u ? s_t[b1 + 1u] : 0u;
    __syncthreads();
    // b2: the serial walk continues with above = cg1, so the threshold on
    // hist2's suffix counts is K - cg1 (cg1 < K by b1's maximality)
    const uint32_t b2 =
        pd_topk_mb_bsel(h + PD_TOPK_MB_H2, s_t, &s_b, K - cg1, tid);
    const uint32_t bfloor = (b1 << 11) | b2;
    // gather this segment at 22-bit granularity: G is bounded by
    // construction (strict-above count < K <= 64); E counts past its cap
    // so K3 can apply the ties-class truncation knowingly
    const uint32_t seglen = (n + gridDim.y - 1u) / gridDim.y;
    const uint32_t i0 = seg * seglen, i1 = min(n, i0 + seglen);
    const float* x = logits + (size_t)row * n;
    for (uint32_t i = i0 + tid; i < i1; i += blockDim.x) {
        const uint32_t key = pd_okey(x[i]);
        const uint32_t top22 = key >> 10;
        if (top22 > bfloor) {
            const uint32_t p = atomicAdd(&h[PD_TOPK_MB_GC], 1u);
            if (p < PD_TOPK_HEAD) {
                h[PD_TOPK_MB_GI + p] = i;
                h[PD_TOPK_MB_GK + p] = key;
            }
        } else if (top22 == bfloor) {
            const uint32_t p = atomicAdd(&h[PD_TOPK_MB_EC], 1u);
            if (p < PD_TOPK_MB_ECAP) {
                h[PD_TOPK_MB_EI + p] = i;
                h[PD_TOPK_MB_EK + p] = key;
            }
        }
    }
}

// canonical (okey desc, id asc) bitonic over u64 composites (key<<32 | ~id);
// pad 0ull sorts last (a real entry's low word is ~id with id < n). len must
// be a power of two.
__device__ __forceinline__ void pd_topk_mb_sortd(unsigned long long* a,
                                                 uint32_t len, uint32_t tid) {
    for (uint32_t ksz = 2; ksz <= len; ksz <<= 1)
        for (uint32_t j = ksz >> 1; j > 0; j >>= 1) {
            for (uint32_t i = tid; i < len; i += 1024u) {
                const uint32_t ixj = i ^ j;
                if (ixj > i) {
                    const bool up = (i & ksz) == 0u; // DESC overall
                    const unsigned long long x0 = a[i], x1 = a[ixj];
                    if (up ? (x0 < x1) : (x0 > x1)) { a[i] = x1; a[ixj] = x0; }
                }
            }
            __syncthreads();
        }
}

template <int MODE>
__global__ void __launch_bounds__(1024) pd_topk_mb_fin_kernel(
    const float* __restrict__ logits, const PdSampleRow* __restrict__ ps,
    const PdSampleTrunc* __restrict__ pt, unsigned int* __restrict__ out,
    uint32_t* __restrict__ scr, uint32_t n, uint32_t k) {
    const uint32_t row = blockIdx.x, tid = threadIdx.x;
    if (ps[row].mode != (uint32_t)MODE) return;
    uint32_t kk = k;
    if (MODE == 5) kk = min(max(pt[row].k, 1u), (uint32_t)PD_TOPK_HEAD);
    const uint32_t K = min(kk, n);
    uint32_t* h = scr + (size_t)row * PD_TOPK_MB_STRIDE;
    const float* x = logits + (size_t)row * n;
    const uint32_t gn = min(h[PD_TOPK_MB_GC], (uint32_t)PD_TOPK_HEAD);
    // >ECAP members sharing 22 top bits is the legacy ties class: the
    // gathered 512 are an arbitrary subset exactly like the one-block
    // kernel's first-256 (ours roomier) - truncate and proceed
    const uint32_t en = min(h[PD_TOPK_MB_EC], (uint32_t)PD_TOPK_MB_ECAP);
    __shared__ unsigned long long s_g[PD_TOPK_HEAD];
    __shared__ unsigned long long s_e[PD_TOPK_MB_ECAP];
    __shared__ uint32_t s_hid[PD_TOPK_HEAD];
    __shared__ float s_hval[PD_TOPK_HEAD];
    __shared__ uint32_t s_hn;
    for (uint32_t i = tid; i < PD_TOPK_HEAD; i += blockDim.x)
        s_g[i] = i < gn
            ? (((unsigned long long)h[PD_TOPK_MB_GK + i] << 32)
               | (uint32_t)(h[PD_TOPK_MB_GI + i] ^ 0xFFFFFFFFu))
            : 0ull;
    for (uint32_t i = tid; i < PD_TOPK_MB_ECAP; i += blockDim.x)
        s_e[i] = i < en
            ? (((unsigned long long)h[PD_TOPK_MB_EK + i] << 32)
               | (uint32_t)(h[PD_TOPK_MB_EI + i] ^ 0xFFFFFFFFu))
            : 0ull;
    __syncthreads();
    // every G key exceeds every E key (strictly-above vs boundary bucket),
    // so the head is sorted-G ++ sorted-E - no merge
    pd_topk_mb_sortd(s_g, PD_TOPK_HEAD, tid);
    pd_topk_mb_sortd(s_e, PD_TOPK_MB_ECAP, tid);
    if (tid == 0) s_hn = min(K, gn + en);
    __syncthreads();
    const uint32_t hn = s_hn;
    for (uint32_t i = tid; i < hn; i += blockDim.x) {
        const unsigned long long c = i < gn ? s_g[i] : s_e[i - gn];
        const uint32_t id = (uint32_t)(c & 0xFFFFFFFFull) ^ 0xFFFFFFFFu;
        s_hid[i] = id;
        s_hval[i] = x[id]; // parallel value fetch (the tails are serial)
    }
    __syncthreads();
    if (MODE == 4) {
        for (uint32_t i = tid; i < k; i += blockDim.x) {
            if (i < hn && i < K) {
                out[((size_t)row * k + i) * 2u] = s_hid[i];
                out[((size_t)row * k + i) * 2u + 1u] =
                    __float_as_uint(s_hval[i]);
            } else {
                out[((size_t)row * k + i) * 2u] = 0xFFFFFFFFu;
                out[((size_t)row * k + i) * 2u + 1u] =
                    __float_as_uint(-INFINITY);
            }
        }
    } else if (tid == 0) {
        // mode 5 draw tail - verbatim pd_sample_rows_t_kernel semantics
        const uint32_t cn = min(hn, K);
        if (cn == 0) {
            out[row] = 0;
        } else {
            const float inv_t = ps[row].inv_t;
            const float m = s_hval[0] * inv_t;
            if (!isfinite(m)) {
                out[row] = s_hid[0];
            } else {
                float p[PD_TOPK_HEAD];
                float head_sum = 0.0f;
                for (uint32_t a = 0; a < cn; ++a) {
                    p[a] = expf(s_hval[a] * inv_t - m);
                    head_sum += p[a];
                }
                if (!(head_sum > 0.0f)) {
                    out[row] = s_hid[0];
                } else {
                    for (uint32_t a = 0; a < cn; ++a) p[a] /= head_sum;
                    uint32_t keep = cn;
                    const float min_p = pt[row].min_p;
                    if (min_p > 0.0f) {
                        const float thresh = min_p * p[0];
                        uint32_t s = 0;
                        while (s < cn && p[s] >= thresh) ++s;
                        keep = s;
                    }
                    const float top_p = pt[row].top_p;
                    if (top_p < 1.0f) {
                        float cum = 0.0f;
                        uint32_t kp = keep;
                        for (uint32_t a = 0; a < keep; ++a) {
                            cum += p[a];
                            if (cum >= top_p) { kp = a + 1u; break; }
                        }
                        keep = kp;
                    }
                    if (keep == 0u) keep = 1u;
                    float total = 0.0f;
                    for (uint32_t a = 0; a < keep; ++a) total += p[a];
                    float r = ps[row].u * total;
                    uint32_t pick = s_hid[keep - 1u];
                    for (uint32_t a = 0; a < keep; ++a) {
                        r -= p[a];
                        if (r <= 0.0f) { pick = s_hid[a]; break; }
                    }
                    out[row] = pick;
                }
            }
        }
    }
    __syncthreads();
    // self-clean: next launch's K1/K1b atomics need zero histograms
    for (uint32_t i = tid; i < 4096u; i += blockDim.x) h[i] = 0;
    if (tid == 0) { h[PD_TOPK_MB_GC] = 0; h[PD_TOPK_MB_EC] = 0; }
}

// mb dispatch: rows 1..MAXR with real vocab widths; scratch comes from the
// load-time repack hook or a lazy alloc when the stream is not capturing
// (cudaMalloc invalidates capture). Falls back to the one-block kernels.
// FLOOR HISTORY: the floor was rows < 3 - an
// UNMEASURED boundary, not a verdict (the sm_100 reconciliation note below
// says mb "never dispatched" at rows 1-2). A rows-small one-CTA fused arm
// (t2s: global two-level hist + inline tail, one launch) was built first
// and FALSIFIED by a serve capture at rows=1: 232us/launch vs the t2
// pair's 122 - the one-CTA full-row scan is the same per-CTA DRAM-latency
// wall the mb comment already records at 162us; consolidation cannot beat
// fan-out here. Dead: rows-small single-launch head build (any shape whose
// row scan is one CTA). The floor drop below is the surviving experiment
// and it PAYS, measured on the same serve protocol (A3B fp8 c1, sm_120a):
// mb at rows=1 is 31.7us of kernel time (hist 5.3 + hist2 6.9 + gather 9.0
// + fin 10.5) vs the t2 pair's 122.5 (98.9 + 23.6), and e2e the elected-
// sampling gap vs the top_k=0/top_p=1 control halves, +3.05% -> +1.54%
// (ctl invariant at 179.1-179.3 across all three serves = clean control).
// The residual ~1.5% is four launch slots + PDL waits on the serial pipe's
// critical path - recorded, not chased.
static uint32_t* pd_topk_mb_scr(uint32_t rows, uint32_t n, void* stream) {
    static int no_mb = -1;
    if (no_mb < 0) no_mb = pd_env("PADDOCK_NO_TOPK_MB") ? 1 : 0;
    if (no_mb || rows == 0u || rows > PD_TOPK_MB_MAXR || n < 4096u)
        return nullptr;
    uint32_t* scr = pd_smp_scr();
    if (!scr) {
        cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing((cudaStream_t)stream, &cs);
        if (cs == cudaStreamCaptureStatusNone) scr = pd_smp_scr(true);
    }
    static bool said = false;
    if (scr && !said) {
        said = true;
        puts("[topk-mb] engaged (multi-block head build, two-level)");
        fflush(stdout); // serve logs are block-buffered; don't lose the witness
    }
    return scr;
}

PD_EXPORT
int pd_topk_rows(const void* logits, const void* params, void* out,
                 uint32_t rows, uint32_t n, uint32_t k, void* stream) {
    if (rows == 0 || n == 0) return 0;
    if (k == 0 || k > PD_TOPK_HEAD) return -2;
    uint32_t* scr = pd_topk_mb_scr(rows, n, stream);
    if (scr) {
        const dim3 g(rows, PD_TOPK_MB_SEG);
        cudaStream_t st = (cudaStream_t)stream;
        pd_topk_mb_hist_kernel<4><<<g, 1024u, 0, st>>>(
            (const float*)logits, (const PdSampleRow*)params, scr, n);
        pd_topk_mb_hist2_kernel<4><<<g, 1024u, 0, st>>>(
            (const float*)logits, (const PdSampleRow*)params,
            (const PdSampleTrunc*)nullptr, scr, n, k);
        pd_topk_mb_gather_kernel<4><<<g, 1024u, 0, st>>>(
            (const float*)logits, (const PdSampleRow*)params,
            (const PdSampleTrunc*)nullptr, scr, n, k);
        pd_topk_mb_fin_kernel<4><<<rows, 1024u, 0, st>>>(
            (const float*)logits, (const PdSampleRow*)params,
            (const PdSampleTrunc*)nullptr, (unsigned int*)out, scr, n, k);
        return pd_launch_status();
    }
    // 1024 threads: the one-block-per-row build is DRAM-latency-bound at
    // 256 (338us at rows=32); 4x the load streams per row
    pd_topk_rows_kernel<<<rows, 1024u, 0, (cudaStream_t)stream>>>(
        (const float*)logits, (const PdSampleRow*)params, (unsigned int*)out,
        n, k);
    return pd_launch_status();
}

PD_EXPORT
int pd_sample_rows_t(const void* logits, const void* params,
                     const void* trunc_params, void* out, uint32_t rows,
                     uint32_t n, void* stream) {
    if (rows == 0 || n == 0) return 0;
    // reconciliation: two independent cures for the one-block
    // head build landed the same day - the sm_120 twin's chunked t2a/t2f
    // (subset property, 2 kernels, static scratch) and the P71b two-level
    // multi-block pipeline below (4 kernels, also mode 4). sm_100 probe
    // with tie+flat rows planted: mb 43us/89us at rows 8/32 vs t2 123/143
    // - but t2 86us at rows 1-2 where mb never dispatched (legacy 219).
    // Election: rows <= 2 -> t2; rows >= 3 -> mb first. The sm_120 box
    // measured the OPPOSITE shape for its die - PADDOCK_SR_T2_FIRST=1
    // flips t2 ahead everywhere (a tuned per-die default candidate).
    // Kills: PADDOCK_NO_SR_T2 (t2), PADDOCK_NO_TOPK_MB (the mb arm).
    static int no_t2 = -1;
    if (no_t2 < 0) no_t2 = pd_env("PADDOCK_NO_SR_T2") ? 1 : 0;
    static int t2_first = -1;
    if (t2_first < 0) t2_first = pd_env("PADDOCK_SR_T2_FIRST") ? 1 : 0;
    uint32_t* scr = pd_topk_mb_scr(rows, n, stream); // null = mb ineligible
    if (!no_t2 && n >= PD_T2_C * 2u * PD_TOPK_HEAD && rows <= PD_SR_MAXROWS
        && (t2_first || !scr)) {
        static bool said_t2 = false;
        if (!said_t2) {
            said_t2 = true;
            puts("[sr-t2] engaged (chunked mode-5 head build)");
            fflush(stdout);
        }
        cudaStream_t st = (cudaStream_t)stream;
        dim3 grid(rows, PD_T2_C);
        pd_sample_rows_t2a_kernel<<<grid, 256u, 0, st>>>(
            (const float*)logits, (const PdSampleRow*)params, n);
        pd_sample_rows_t2f_kernel<<<rows, 1024u, 0, st>>>(
            (const float*)logits, (const PdSampleRow*)params,
            (const PdSampleTrunc*)trunc_params, (unsigned int*)out, n);
        return pd_launch_status();
    }
    if (scr) {
        const dim3 g(rows, PD_TOPK_MB_SEG);
        cudaStream_t st = (cudaStream_t)stream;
        pd_topk_mb_hist_kernel<5><<<g, 1024u, 0, st>>>(
            (const float*)logits, (const PdSampleRow*)params, scr, n);
        pd_topk_mb_hist2_kernel<5><<<g, 1024u, 0, st>>>(
            (const float*)logits, (const PdSampleRow*)params,
            (const PdSampleTrunc*)trunc_params, scr, n, PD_TOPK_HEAD);
        pd_topk_mb_gather_kernel<5><<<g, 1024u, 0, st>>>(
            (const float*)logits, (const PdSampleRow*)params,
            (const PdSampleTrunc*)trunc_params, scr, n, PD_TOPK_HEAD);
        pd_topk_mb_fin_kernel<5><<<rows, 1024u, 0, st>>>(
            (const float*)logits, (const PdSampleRow*)params,
            (const PdSampleTrunc*)trunc_params, (unsigned int*)out, scr, n,
            PD_TOPK_HEAD);
        return pd_launch_status();
    }
    pd_sample_rows_t_kernel<<<rows, 1024u, 0, (cudaStream_t)stream>>>(
        (const float*)logits, (const PdSampleRow*)params,
        (const PdSampleTrunc*)trunc_params, (unsigned int*)out, n);
    return pd_launch_status();
}


// ── truncation stage (c): GENERAL truncation sampling, mode 6 ──────────────────
// Mode 5 covers top_k 1..=64 (a 64-head bounds the whole nucleus). Mode 6
// covers the k-less half of the truncation space exactly - top-p only
// (nemotron's published profile is temperature 1.0 / top_p 0.95 with no
// top_k), min-p only, and their combination - via a histogram-of-masses
// QUANTILE WALK: no vocab sort, no rejection loop, a fixed straight-line
// sequence of cooperative passes (static launch geometry, so the captured
// decode graphs can carry it). top_k in 65..n-1 stays on the host pipeline:
// no elected profile uses it (OpenAI's API does not even expose top_k) and
// the head-partial machinery it needs is not worth the code until a
// measurement says otherwise.
//
// Host semantics (sampler.rs build_nucleus, top_k == 0 branch) verbatim:
//   D = full-vocab exp-mass; min-p prunes candidates with e < min_p
//   (probs[0] = 1/D - the global argmax is always a candidate, so the
//   host's min_p·probs[0] threshold is min_p on raw exp-masses, compared
//   per element); top-p keeps the shortest desc-order prefix of survivors
//   with cum/D >= top_p (all survivors when never reached); the draw walks
//   the same desc order at quantile t = u · M_nucleus.
//
// Structure: 1024-bucket histogram over the top-10 okey bits accumulates
// (count, exp-mass, min_p-passing exp-mass); the two boundaries the walk
// meets - the top-p cut and the quantile bucket - each refine one bucket a
// second 10-bit level, gather that sub-bucket (bitonic sort by okey desc,
// index asc), and finish element-wise in thread 0. Skipped phases run with
// a match-nothing sentinel prefix so every thread crosses the same
// barriers. Distribution class: expf ulps at cum boundaries (the mode-2
// doctrine); key-equal ties beyond the gather cap resolve by ascending
// index (boundary-tie choice was never contractual). Rivals sample
// arbitrary top-p on device via rejection loops (FlashInfer class) -
// studied as inspiration, this deterministic-pass design is original.
#define PD_TOPP_BUCKETS 1024u
#define PD_TOPP_GCAP 1024u
#define PD_TOPP_NONE 0xFFFFFFFFu

__global__ void __launch_bounds__(1024) pd_sample_rows_p_kernel(
    const float* __restrict__ logits, const PdSampleRow* __restrict__ ps,
    const PdSampleTrunc* __restrict__ pt, unsigned int* __restrict__ out,
    uint32_t n) {
    const uint32_t row = blockIdx.x, tid = threadIdx.x;
    if (ps[row].mode != 6u) return;
    const float* x = logits + (size_t)row * n;
    const float inv_t = ps[row].inv_t;
    const float top_p = pt[row].top_p;
    const float min_p = pt[row].min_p;

    __shared__ uint32_t s_cnt[PD_TOPP_BUCKETS];
    __shared__ float s_mass[PD_TOPP_BUCKETS];  // exp-mass, all elements
    __shared__ float s_massp[PD_TOPP_BUCKETS]; // exp-mass, min_p survivors
    __shared__ float s_m1[PD_TOPP_BUCKETS];    // level-1 survivor mass
    __shared__ unsigned long long s_g[PD_TOPP_GCAP];
    __shared__ unsigned long long s_amax; // (okey << 32 | ~id) argmax
    __shared__ uint32_t s_gn;
    __shared__ float s_red[32];
    __shared__ float s_m;
    __shared__ uint32_t s_req[2]; // (10-bit L1 prefix request, 20-bit gather prefix)
    __shared__ float s_scal[4];   // thread-0 scalars carried across barriers

    // Phase A: global max of scaled logits + exact argmax (host tie rule:
    // lowest index - ~id makes the composite prefer it)
    if (tid == 0) s_amax = 0ull;
    {
        float mx = -INFINITY;
        for (uint32_t i = tid; i < n; i += blockDim.x)
            mx = fmaxf(mx, x[i] * inv_t);
        for (uint32_t sh = 16; sh > 0; sh >>= 1)
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, sh));
        if ((tid & 31u) == 0u) s_red[tid >> 5] = mx;
        __syncthreads();
        if (tid == 0) {
            float m = -INFINITY;
            for (uint32_t w = 0; w < (blockDim.x + 31u) >> 5; ++w)
                m = fmaxf(m, s_red[w]);
            s_m = m;
        }
        __syncthreads();
    }
    const float m = s_m;

    // Phase B: level-0 histogram (top 10 okey bits) + argmax composite
    for (uint32_t b = tid; b < PD_TOPP_BUCKETS; b += blockDim.x) {
        s_cnt[b] = 0; s_mass[b] = 0.0f; s_massp[b] = 0.0f;
    }
    __syncthreads();
    for (uint32_t i = tid; i < n; i += blockDim.x) {
        const float sv = x[i] * inv_t;
        const uint32_t key = pd_okey(sv);
        const float e = expf(sv - m);
        const uint32_t b = key >> 22;
        atomicAdd(&s_cnt[b], 1u);
        atomicAdd(&s_mass[b], e);
        if (!(min_p > 0.0f) || e >= min_p) atomicAdd(&s_massp[b], e);
        atomicMax(&s_amax, ((unsigned long long)key << 32)
                               | (uint32_t)(i ^ 0xFFFFFFFFu));
    }
    __syncthreads();

    // Phase B-tail (thread 0): D, survivor mass S, and the top-p cut walk
    // over full level-0 buckets. s_req[0] = L1 refine prefix (or none),
    // s_scal = {D, cum entering the cut bucket, S, unused}.
    if (tid == 0) {
        float dsum = 0.0f, ssum = 0.0f;
        for (uint32_t b = 0; b < PD_TOPP_BUCKETS; ++b) {
            dsum += s_mass[b];
            ssum += s_massp[b];
        }
        const float d = dsum;
        s_req[0] = PD_TOPP_NONE;
        s_scal[0] = d;
        s_scal[2] = ssum;
        if (d > 0.0f && ssum > 0.0f && top_p < 1.0f) {
            float cum = 0.0f;
            for (uint32_t b = PD_TOPP_BUCKETS - 1u;; --b) {
                const float bm = s_massp[b];
                if (bm > 0.0f) {
                    if ((cum + bm) / d >= top_p) {
                        s_req[0] = b; // cut lands inside this bucket
                        s_scal[1] = cum;
                        break;
                    }
                    cum += bm;
                }
                if (b == 0u) break;
            }
        }
    }
    __syncthreads();
    const float D = s_scal[0];
    const float S = s_scal[2];

    // Degenerate row (softcap'd -inf floods, min_p ate everything): the
    // host keeps the argmax (keep.max(1)) - bit-matching tie rule
    if (!(D > 0.0f) || !(S > 0.0f)) {
        if (tid == 0) out[row] = (uint32_t)(s_amax & 0xFFFFFFFFu) ^ 0xFFFFFFFFu;
        return;
    }

    // Phase C: level-1 histogram of the cut bucket (survivor mass only).
    // Sentinel prefix -> pure no-op pass, same barriers for everyone.
    const uint32_t pb = s_req[0];
    for (uint32_t b = tid; b < PD_TOPP_BUCKETS; b += blockDim.x) s_m1[b] = 0.0f;
    __syncthreads();
    if (pb != PD_TOPP_NONE) {
        for (uint32_t i = tid; i < n; i += blockDim.x) {
            const float sv = x[i] * inv_t;
            const uint32_t key = pd_okey(sv);
            if ((key >> 22) != pb) continue;
            const float e = expf(sv - m);
            if (min_p > 0.0f && e < min_p) continue;
            atomicAdd(&s_m1[(key >> 12) & (PD_TOPP_BUCKETS - 1u)], e);
        }
    }
    __syncthreads();
    // C-tail (thread 0): sub-bucket scan -> gather request (20-bit prefix)
    if (tid == 0) {
        s_req[1] = PD_TOPP_NONE;
        if (pb != PD_TOPP_NONE) {
            float cum = s_scal[1];
            const float target = top_p * D;
            for (uint32_t b = PD_TOPP_BUCKETS - 1u;; --b) {
                const float bm = s_m1[b];
                if (bm > 0.0f) {
                    if (cum + bm >= target) {
                        s_req[1] = (pb << 10) | b;
                        s_scal[1] = cum; // cum entering the sub-bucket
                        break;
                    }
                    cum += bm;
                }
                if (b == 0u) break;
            }
            if (s_req[1] == PD_TOPP_NONE) s_scal[3] = cum; // fp: never crossed
        }
    }
    __syncthreads();

    // Phase D: gather + bitonic sort of the cut sub-bucket (or a no-op)
    const uint32_t gp = s_req[1];
    if (tid == 0) s_gn = 0;
    __syncthreads();
    if (gp != PD_TOPP_NONE) {
        for (uint32_t i = tid; i < n; i += blockDim.x) {
            const uint32_t key = pd_okey(x[i] * inv_t);
            if ((key >> 12) == gp) {
                const uint32_t p2i = atomicAdd(&s_gn, 1u);
                if (p2i < PD_TOPP_GCAP)
                    s_g[p2i] = ((unsigned long long)key << 32)
                                   | (uint32_t)(i ^ 0xFFFFFFFFu);
            }
        }
    }
    __syncthreads();
    {
        const uint32_t len = min(s_gn, PD_TOPP_GCAP);
        uint32_t p2 = 1;
        while (p2 < len) p2 <<= 1;
        for (uint32_t i = tid; i < p2; i += blockDim.x)
            if (i >= len) s_g[i] = 0ull;
        __syncthreads();
        for (uint32_t ksz = 2; ksz <= p2; ksz <<= 1)
            for (uint32_t j = ksz >> 1; j > 0; j >>= 1) {
                for (uint32_t i = tid; i < p2; i += blockDim.x) {
                    const uint32_t ixj = i ^ j;
                    if (ixj > i && ixj < p2) {
                        const bool up = (i & ksz) == 0u; // DESC overall
                        const unsigned long long a = s_g[i], b2 = s_g[ixj];
                        if (up ? (a < b2) : (a > b2)) { s_g[i] = b2; s_g[ixj] = a; }
                    }
                }
                __syncthreads();
            }
    }
    // D-tail (thread 0): finish the cut element-wise -> M = nucleus mass,
    // and the cut's last element id (the draw's fp-tail fallback)
    if (tid == 0) {
        float mnuc;
        uint32_t cut_last = (uint32_t)(s_amax & 0xFFFFFFFFu) ^ 0xFFFFFFFFu;
        if (pb == PD_TOPP_NONE) {
            mnuc = S; // top_p off or never reached at level 0
        } else if (gp == PD_TOPP_NONE) {
            mnuc = s_scal[3]; // fp: crossed at L0 but not at L1 - all of pb
        } else {
            float cum = s_scal[1];
            const float target = top_p * D;
            mnuc = cum;
            const uint32_t len = min(s_gn, PD_TOPP_GCAP);
            for (uint32_t a = 0; a < len; ++a) {
                const uint32_t id = (uint32_t)(s_g[a] & 0xFFFFFFFFu) ^ 0xFFFFFFFFu;
                const float sv = x[id] * inv_t;
                const float e = expf(sv - m);
                if (min_p > 0.0f && e < min_p) continue;
                cum += e;
                cut_last = id;
                if (cum >= target) break;
            }
            mnuc = cum;
        }
        s_scal[1] = mnuc;
        s_scal[3] = __uint_as_float(cut_last);
    }
    __syncthreads();
    const float M = s_scal[1];

    // Phase E (thread 0): quantile walk t = u·M over the same desc order -
    // full buckets by survivor mass; the crossing bucket refines next
    if (tid == 0) {
        s_req[0] = PD_TOPP_NONE;
        const float t = ps[row].u * M;
        float cum = 0.0f;
        for (uint32_t b = PD_TOPP_BUCKETS - 1u;; --b) {
            const float bm = s_massp[b];
            if (bm > 0.0f) {
                if (cum + bm >= t) {
                    s_req[0] = b;
                    s_scal[0] = cum;
                    s_scal[2] = t;
                    break;
                }
                cum += bm;
            }
            if (b == 0u) break;
        }
    }
    __syncthreads();

    // Phase F: level-1 histogram of the quantile bucket (reuse if == pb)
    const uint32_t qb = s_req[0];
    const bool reuse_l1 = qb != PD_TOPP_NONE && qb == pb;
    if (!reuse_l1) {
        for (uint32_t b = tid; b < PD_TOPP_BUCKETS; b += blockDim.x) s_m1[b] = 0.0f;
        __syncthreads();
        if (qb != PD_TOPP_NONE) {
            for (uint32_t i = tid; i < n; i += blockDim.x) {
                const float sv = x[i] * inv_t;
                const uint32_t key = pd_okey(sv);
                if ((key >> 22) != qb) continue;
                const float e = expf(sv - m);
                if (min_p > 0.0f && e < min_p) continue;
                atomicAdd(&s_m1[(key >> 12) & (PD_TOPP_BUCKETS - 1u)], e);
            }
        }
    }
    __syncthreads();
    // F-tail (thread 0): sub-bucket scan for the quantile
    if (tid == 0) {
        s_req[1] = PD_TOPP_NONE;
        if (qb != PD_TOPP_NONE) {
            float cum = s_scal[0];
            const float t = s_scal[2];
            for (uint32_t b = PD_TOPP_BUCKETS - 1u;; --b) {
                const float bm = s_m1[b];
                if (bm > 0.0f) {
                    if (cum + bm >= t) {
                        s_req[1] = (qb << 10) | b;
                        s_scal[0] = cum;
                        break;
                    }
                    cum += bm;
                }
                if (b == 0u) break;
            }
        }
    }
    __syncthreads();

    // Phase G: gather + sort the quantile sub-bucket (reuse the cut's
    // gather when it is the same 20-bit prefix), final element walk
    const uint32_t gq = s_req[1];
    const bool reuse_g = gq != PD_TOPP_NONE && gq == gp;
    if (!reuse_g) {
        if (tid == 0) s_gn = 0;
        __syncthreads();
        if (gq != PD_TOPP_NONE) {
            for (uint32_t i = tid; i < n; i += blockDim.x) {
                const uint32_t key = pd_okey(x[i] * inv_t);
                if ((key >> 12) == gq) {
                    const uint32_t p2i = atomicAdd(&s_gn, 1u);
                    if (p2i < PD_TOPP_GCAP)
                        s_g[p2i] = ((unsigned long long)key << 32)
                                       | (uint32_t)(i ^ 0xFFFFFFFFu);
                }
            }
        }
        __syncthreads();
        const uint32_t len = min(s_gn, PD_TOPP_GCAP);
        uint32_t p2 = 1;
        while (p2 < len) p2 <<= 1;
        for (uint32_t i = tid; i < p2; i += blockDim.x)
            if (i >= len) s_g[i] = 0ull;
        __syncthreads();
        for (uint32_t ksz = 2; ksz <= p2; ksz <<= 1)
            for (uint32_t j = ksz >> 1; j > 0; j >>= 1) {
                for (uint32_t i = tid; i < p2; i += blockDim.x) {
                    const uint32_t ixj = i ^ j;
                    if (ixj > i && ixj < p2) {
                        const bool up = (i & ksz) == 0u;
                        const unsigned long long a = s_g[i], b2 = s_g[ixj];
                        if (up ? (a < b2) : (a > b2)) { s_g[i] = b2; s_g[ixj] = a; }
                    }
                }
                __syncthreads();
            }
    } else {
        __syncthreads(); // mirror the barrier count of the gather arm
        __syncthreads();
    }
    if (tid == 0) {
        uint32_t pick = __float_as_uint(s_scal[3]); // cut_last fallback
        if (gq != PD_TOPP_NONE) {
            float cum = s_scal[0];
            const float t = s_scal[2];
            const uint32_t len = min(s_gn, PD_TOPP_GCAP);
            for (uint32_t a = 0; a < len; ++a) {
                const uint32_t id = (uint32_t)(s_g[a] & 0xFFFFFFFFu) ^ 0xFFFFFFFFu;
                const float sv = x[id] * inv_t;
                const float e = expf(sv - m);
                if (min_p > 0.0f && e < min_p) continue;
                cum += e;
                pick = id;
                if (cum >= t) break;
            }
        }
        out[row] = pick;
    }
}

// ── mode-6 MULTI-BLOCK chain (nemotron c8 attribution) ───────
// The one-block kernel above is the c8/c1 disease shape re-measured on
// nemotron's k-less election: up to 6 serial full-vocab passes from one
// block per row (~1.2 ms/step at rows=8 - a 12.7% c8 serve A/B, categorical
// 959 vs elected-trunc 837 out_tok/s). This chain is the same decision
// sequence phase-for-phase - identical serial walk orders, tie rules, and
// min-p semantics - with every full-vocab pass wave-dense (rows x SEG
// grid, global-atomic merge) and the walks as one-block-per-row kernels
// over the 1024-bucket histograms only. Fixed 11-launch sequence, static
// geometry, data-dependent phases no-op via flag-encoded requests (rest
// state 0 = none, so the self-clean is the sentinel reset). Distribution
// class unchanged: global float-atomic merges reassociate exp-mass adds
// (the mode-2 doctrine ulps class; the host-nucleus gate bar is >= 99%
// with adjacency). Scratch mirrors pd_smp_scr: load-time alloc from the
// f8 repack hook or lazy when not capturing; fin self-cleans the row.
// Kill: PADDOCK_NO_TOPP_MB (one-block fallback).
#define PD_TOPP_MB_SEG 32u
#define PD_TOPP_MB_MAXR 64u
// per-row u32 scratch layout (rest state all-zero):
//   [0..1] amax composite (okey<<32 | ~id)  [2] reqc  [3] reqg  [4] reqd
//   [5] reqq  [6] done  [7] gcnt  [8] D  [9] S  [10] cum_cut  [11] M
//   [12] t  [13] cum_draw  [14] cut_last  [15] fp-mass (pb never crossed)
//   [16..1039] mass f32[1024] | [1040..2063] massp f32[1024]
//   [2064..3087] m1 f32[1024] | [3088..5135] g u64[1024]
#define PD_TOPP_MB_REQC 2u
#define PD_TOPP_MB_REQG 3u
#define PD_TOPP_MB_REQD 4u
#define PD_TOPP_MB_REQQ 5u
#define PD_TOPP_MB_DONE 6u
#define PD_TOPP_MB_GC 7u
#define PD_TOPP_MB_D 8u
#define PD_TOPP_MB_S 9u
#define PD_TOPP_MB_CUMC 10u
#define PD_TOPP_MB_M 11u
#define PD_TOPP_MB_T 12u
#define PD_TOPP_MB_CUMD 13u
#define PD_TOPP_MB_LAST 14u
#define PD_TOPP_MB_FP 15u
#define PD_TOPP_MB_MASS 16u
#define PD_TOPP_MB_MASSP 1040u
#define PD_TOPP_MB_M1 2064u
#define PD_TOPP_MB_G 3088u
#define PD_TOPP_MB_STRIDE 5136u
#define PD_TOPP_REQ_SET 0x80000000u

static uint32_t* pd_topp_scr(bool make = false) {
    static uint32_t* scr = nullptr;
    if (make && !scr) {
        const size_t bytes = (size_t)PD_TOPP_MB_MAXR * PD_TOPP_MB_STRIDE * 4u;
        if (cudaMalloc(&scr, bytes) != cudaSuccess) scr = nullptr;
        else cudaMemset(scr, 0, bytes);
    }
    return scr;
}

__device__ __forceinline__ float pd_okey_inv(uint32_t key) {
    // exact inverse of pd_okey (monotone f32 total-order map)
    const uint32_t b = (key & 0x80000000u) ? (key ^ 0x80000000u) : ~key;
    return __uint_as_float(b);
}

// P1: composite max - max scaled value AND its lowest-index argmax in one
// wave-dense pass (the composite's ~id makes ties prefer the lowest index,
// the host rule the one-block kernel's phase A + s_amax pair encodes).
__global__ void __launch_bounds__(1024) pd_topp_mb_max_kernel(
    const float* __restrict__ logits, const PdSampleRow* __restrict__ ps,
    uint32_t* __restrict__ scr, uint32_t n) {
    const uint32_t row = blockIdx.x, seg = blockIdx.y, tid = threadIdx.x;
    if (ps[row].mode != 6u) return;
    const float inv_t = ps[row].inv_t;
    const float* x = logits + (size_t)row * n;
    const uint32_t seglen = (n + gridDim.y - 1u) / gridDim.y;
    const uint32_t i0 = seg * seglen, i1 = min(n, i0 + seglen);
    unsigned long long best = 0ull;
    for (uint32_t i = i0 + tid; i < i1; i += blockDim.x) {
        const uint32_t key = pd_okey(x[i] * inv_t);
        const unsigned long long c =
            ((unsigned long long)key << 32) | (uint32_t)(i ^ 0xFFFFFFFFu);
        if (c > best) best = c;
    }
    for (uint32_t sh = 16; sh > 0; sh >>= 1) {
        const unsigned long long o = __shfl_xor_sync(0xffffffffu, best, sh);
        if (o > best) best = o;
    }
    __shared__ unsigned long long s_r[32];
    if ((tid & 31u) == 0u) s_r[tid >> 5] = best;
    __syncthreads();
    if (tid == 0) {
        for (uint32_t w = 1; w < (blockDim.x + 31u) >> 5; ++w)
            if (s_r[w] > best) best = s_r[w];
        atomicMax((unsigned long long*)(scr + (size_t)row * PD_TOPP_MB_STRIDE),
                  best);
    }
}

// P2: level-0 (top-10 okey bits) exp-mass histograms, shared-staged then
// global-atomic merged. massp = min_p survivors (the walks' plane).
__global__ void __launch_bounds__(1024) pd_topp_mb_hist_kernel(
    const float* __restrict__ logits, const PdSampleRow* __restrict__ ps,
    const PdSampleTrunc* __restrict__ pt, uint32_t* __restrict__ scr,
    uint32_t n) {
    const uint32_t row = blockIdx.x, seg = blockIdx.y, tid = threadIdx.x;
    if (ps[row].mode != 6u) return;
    const float inv_t = ps[row].inv_t;
    const float min_p = pt[row].min_p;
    uint32_t* h = scr + (size_t)row * PD_TOPP_MB_STRIDE;
    const float m =
        pd_okey_inv((uint32_t)(*(const unsigned long long*)h >> 32));
    const float* x = logits + (size_t)row * n;
    const uint32_t seglen = (n + gridDim.y - 1u) / gridDim.y;
    const uint32_t i0 = seg * seglen, i1 = min(n, i0 + seglen);
    __shared__ float s_mass[PD_TOPP_BUCKETS];
    __shared__ float s_massp[PD_TOPP_BUCKETS];
    for (uint32_t b = tid; b < PD_TOPP_BUCKETS; b += blockDim.x) {
        s_mass[b] = 0.0f; s_massp[b] = 0.0f;
    }
    __syncthreads();
    // Warp-aggregated bucket adds (micro-rung): real
    // logit rows concentrate okey mass in a handful of buckets, so the
    // per-element shared float atomics serialized (131 us/launch at c32).
    // match_any groups a warp's equal-bucket lanes; the group leader sums
    // its peers via shfl and issues one atomic per (warp, bucket). Same
    // arithmetic class: shared float-atomic order was already hardware-
    // nondeterministic, and the chain's gates are tolerance-class.
    const uint32_t lane = tid & 31u;
    for (uint32_t i = i0 + tid; i < i1; i += blockDim.x) {
        const float sv = x[i] * inv_t;
        const uint32_t b = pd_okey(sv) >> 22;
        const float e = expf(sv - m);
        const float ep = (!(min_p > 0.0f) || e >= min_p) ? e : 0.0f;
        const uint32_t peers = __match_any_sync(__activemask(), b);
        const uint32_t leader = __ffs(peers) - 1u;
        float sum = e, sump = ep;
        uint32_t rest = peers & ~(1u << leader);
        while (rest) {
            const uint32_t srcl = __ffs(rest) - 1u;
            sum += __shfl_sync(peers, e, srcl);
            sump += __shfl_sync(peers, ep, srcl);
            rest &= rest - 1u;
        }
        if (lane == leader) {
            atomicAdd(&s_mass[b], sum);
            if (sump != 0.0f) atomicAdd(&s_massp[b], sump);
        }
    }
    __syncthreads();
    float* gm = (float*)(h + PD_TOPP_MB_MASS);
    float* gp = (float*)(h + PD_TOPP_MB_MASSP);
    for (uint32_t b = tid; b < PD_TOPP_BUCKETS; b += blockDim.x) {
        if (s_mass[b] != 0.0f) atomicAdd(&gm[b], s_mass[b]);
        if (s_massp[b] != 0.0f) atomicAdd(&gp[b], s_massp[b]);
    }
}

// P3: B-tail - D/S sums + degenerate fallback + the level-0 top-p cut walk.
// Histograms are staged to shared so thread 0's serial order (the one-block
// kernel's exact arithmetic order) reads at shared latency.
__global__ void __launch_bounds__(1024) pd_topp_mb_walk1_kernel(
    const PdSampleRow* __restrict__ ps, const PdSampleTrunc* __restrict__ pt,
    unsigned int* __restrict__ out, uint32_t* __restrict__ scr) {
    const uint32_t row = blockIdx.x, tid = threadIdx.x;
    if (ps[row].mode != 6u) return;
    uint32_t* h = scr + (size_t)row * PD_TOPP_MB_STRIDE;
    __shared__ float s_mass[PD_TOPP_BUCKETS];
    __shared__ float s_massp[PD_TOPP_BUCKETS];
    for (uint32_t b = tid; b < PD_TOPP_BUCKETS; b += blockDim.x) {
        s_mass[b] = ((const float*)(h + PD_TOPP_MB_MASS))[b];
        s_massp[b] = ((const float*)(h + PD_TOPP_MB_MASSP))[b];
    }
    __syncthreads();
    if (tid != 0) return;
    const float top_p = pt[row].top_p;
    float dsum = 0.0f, ssum = 0.0f;
    for (uint32_t b = 0; b < PD_TOPP_BUCKETS; ++b) {
        dsum += s_mass[b];
        ssum += s_massp[b];
    }
    ((float*)h)[PD_TOPP_MB_D] = dsum;
    ((float*)h)[PD_TOPP_MB_S] = ssum;
    if (!(dsum > 0.0f) || !(ssum > 0.0f)) {
        const unsigned long long amax = *(const unsigned long long*)h;
        out[row] = (uint32_t)(amax & 0xFFFFFFFFu) ^ 0xFFFFFFFFu;
        h[PD_TOPP_MB_DONE] = 1u;
        return;
    }
    if (top_p < 1.0f) {
        float cum = 0.0f;
        for (uint32_t b = PD_TOPP_BUCKETS - 1u;; --b) {
            const float bm = s_massp[b];
            if (bm > 0.0f) {
                if ((cum + bm) / dsum >= top_p) {
                    h[PD_TOPP_MB_REQC] = PD_TOPP_REQ_SET | b;
                    ((float*)h)[PD_TOPP_MB_CUMC] = cum;
                    break;
                }
                cum += bm;
            }
            if (b == 0u) break;
        }
    }
}

// P4/P8: level-1 survivor-mass histogram of a requested level-0 bucket
// (cut when phase==0, draw when phase==1), wave-dense.
__global__ void __launch_bounds__(1024) pd_topp_mb_ref_kernel(
    const float* __restrict__ logits, const PdSampleRow* __restrict__ ps,
    const PdSampleTrunc* __restrict__ pt, uint32_t* __restrict__ scr,
    uint32_t n, uint32_t phase) {
    const uint32_t row = blockIdx.x, seg = blockIdx.y, tid = threadIdx.x;
    if (ps[row].mode != 6u) return;
    uint32_t* h = scr + (size_t)row * PD_TOPP_MB_STRIDE;
    if (h[PD_TOPP_MB_DONE]) return;
    const uint32_t req = h[phase == 0u ? PD_TOPP_MB_REQC : PD_TOPP_MB_REQD];
    if (!(req & PD_TOPP_REQ_SET)) return;
    const uint32_t pb = req & (PD_TOPP_BUCKETS - 1u);
    const float inv_t = ps[row].inv_t;
    const float min_p = pt[row].min_p;
    const float m =
        pd_okey_inv((uint32_t)(*(const unsigned long long*)h >> 32));
    const float* x = logits + (size_t)row * n;
    const uint32_t seglen = (n + gridDim.y - 1u) / gridDim.y;
    const uint32_t i0 = seg * seglen, i1 = min(n, i0 + seglen);
    __shared__ float s_m1[PD_TOPP_BUCKETS];
    for (uint32_t b = tid; b < PD_TOPP_BUCKETS; b += blockDim.x)
        s_m1[b] = 0.0f;
    __syncthreads();
    for (uint32_t i = i0 + tid; i < i1; i += blockDim.x) {
        const float sv = x[i] * inv_t;
        const uint32_t key = pd_okey(sv);
        if ((key >> 22) != pb) continue;
        const float e = expf(sv - m);
        if (min_p > 0.0f && e < min_p) continue;
        atomicAdd(&s_m1[(key >> 12) & (PD_TOPP_BUCKETS - 1u)], e);
    }
    __syncthreads();
    float* g1 = (float*)(h + PD_TOPP_MB_M1);
    for (uint32_t b = tid; b < PD_TOPP_BUCKETS; b += blockDim.x)
        if (s_m1[b] != 0.0f) atomicAdd(&g1[b], s_m1[b]);
}

// P5: C-tail - the cut's level-1 sub-bucket walk; zeroes m1 afterwards so
// the draw refine (P8) starts clean.
__global__ void __launch_bounds__(1024) pd_topp_mb_walk2_kernel(
    const PdSampleRow* __restrict__ ps, const PdSampleTrunc* __restrict__ pt,
    uint32_t* __restrict__ scr) {
    const uint32_t row = blockIdx.x, tid = threadIdx.x;
    if (ps[row].mode != 6u) return;
    uint32_t* h = scr + (size_t)row * PD_TOPP_MB_STRIDE;
    if (h[PD_TOPP_MB_DONE]) return;
    __shared__ float s_m1[PD_TOPP_BUCKETS];
    float* g1 = (float*)(h + PD_TOPP_MB_M1);
    for (uint32_t b = tid; b < PD_TOPP_BUCKETS; b += blockDim.x)
        s_m1[b] = g1[b];
    __syncthreads();
    const uint32_t reqc = h[PD_TOPP_MB_REQC];
    if (tid == 0 && (reqc & PD_TOPP_REQ_SET)) {
        const uint32_t pb = reqc & (PD_TOPP_BUCKETS - 1u);
        float cum = ((const float*)h)[PD_TOPP_MB_CUMC];
        const float target = pt[row].top_p * ((const float*)h)[PD_TOPP_MB_D];
        bool crossed = false;
        for (uint32_t b = PD_TOPP_BUCKETS - 1u;; --b) {
            const float bm = s_m1[b];
            if (bm > 0.0f) {
                if (cum + bm >= target) {
                    h[PD_TOPP_MB_REQG] = PD_TOPP_REQ_SET | ((pb << 10) | b);
                    ((float*)h)[PD_TOPP_MB_CUMC] = cum;
                    crossed = true;
                    break;
                }
                cum += bm;
            }
            if (b == 0u) break;
        }
        if (!crossed) ((float*)h)[PD_TOPP_MB_FP] = cum;
    }
    __syncthreads();
    for (uint32_t b = tid; b < PD_TOPP_BUCKETS; b += blockDim.x) g1[b] = 0.0f;
}

// P6/P10: gather a requested 20-bit prefix into the row's global list
// (cap 1024 - the ties-class truncation, exactly the one-block GCAP).
__global__ void __launch_bounds__(1024) pd_topp_mb_gath_kernel(
    const float* __restrict__ logits, const PdSampleRow* __restrict__ ps,
    uint32_t* __restrict__ scr, uint32_t n, uint32_t phase) {
    const uint32_t row = blockIdx.x, seg = blockIdx.y, tid = threadIdx.x;
    if (ps[row].mode != 6u) return;
    uint32_t* h = scr + (size_t)row * PD_TOPP_MB_STRIDE;
    if (h[PD_TOPP_MB_DONE]) return;
    const uint32_t req = h[phase == 0u ? PD_TOPP_MB_REQG : PD_TOPP_MB_REQQ];
    if (!(req & PD_TOPP_REQ_SET)) return;
    const uint32_t gp = req & 0xFFFFFu;
    const float inv_t = ps[row].inv_t;
    const float* x = logits + (size_t)row * n;
    const uint32_t seglen = (n + gridDim.y - 1u) / gridDim.y;
    const uint32_t i0 = seg * seglen, i1 = min(n, i0 + seglen);
    unsigned long long* g = (unsigned long long*)(h + PD_TOPP_MB_G);
    for (uint32_t i = i0 + tid; i < i1; i += blockDim.x) {
        const uint32_t key = pd_okey(x[i] * inv_t);
        if ((key >> 12) == gp) {
            const uint32_t p2i = atomicAdd(&h[PD_TOPP_MB_GC], 1u);
            if (p2i < PD_TOPP_GCAP)
                g[p2i] = ((unsigned long long)key << 32)
                             | (uint32_t)(i ^ 0xFFFFFFFFu);
        }
    }
}

// P7: D-tail (finish the cut element-wise -> M, cut_last) + phase E (the
// draw's level-0 quantile walk). Resets gcnt for the draw gather.
__global__ void __launch_bounds__(1024) pd_topp_mb_fincut_kernel(
    const float* __restrict__ logits, const PdSampleRow* __restrict__ ps,
    const PdSampleTrunc* __restrict__ pt, uint32_t* __restrict__ scr,
    uint32_t n) {
    const uint32_t row = blockIdx.x, tid = threadIdx.x;
    if (ps[row].mode != 6u) return;
    uint32_t* h = scr + (size_t)row * PD_TOPP_MB_STRIDE;
    if (h[PD_TOPP_MB_DONE]) return;
    __shared__ unsigned long long s_g[PD_TOPP_GCAP];
    __shared__ float s_massp[PD_TOPP_BUCKETS];
    const unsigned long long* g = (const unsigned long long*)(h + PD_TOPP_MB_G);
    const uint32_t len = min(h[PD_TOPP_MB_GC], PD_TOPP_GCAP);
    for (uint32_t i = tid; i < PD_TOPP_GCAP; i += blockDim.x)
        s_g[i] = i < len ? g[i] : 0ull;
    for (uint32_t b = tid; b < PD_TOPP_BUCKETS; b += blockDim.x)
        s_massp[b] = ((const float*)(h + PD_TOPP_MB_MASSP))[b];
    __syncthreads();
    pd_topk_mb_sortd(s_g, PD_TOPP_GCAP, tid);
    if (tid != 0) return;
    const float inv_t = ps[row].inv_t;
    const float min_p = pt[row].min_p;
    const float m =
        pd_okey_inv((uint32_t)(*(const unsigned long long*)h >> 32));
    const float* x = logits + (size_t)row * n;
    const unsigned long long amax = *(const unsigned long long*)h;
    uint32_t cut_last = (uint32_t)(amax & 0xFFFFFFFFu) ^ 0xFFFFFFFFu;
    const uint32_t reqc = h[PD_TOPP_MB_REQC], reqg = h[PD_TOPP_MB_REQG];
    float mnuc;
    if (!(reqc & PD_TOPP_REQ_SET)) {
        mnuc = ((const float*)h)[PD_TOPP_MB_S];
    } else if (!(reqg & PD_TOPP_REQ_SET)) {
        mnuc = ((const float*)h)[PD_TOPP_MB_FP];
    } else {
        float cum = ((const float*)h)[PD_TOPP_MB_CUMC];
        const float target = pt[row].top_p * ((const float*)h)[PD_TOPP_MB_D];
        for (uint32_t a = 0; a < len; ++a) {
            const uint32_t id = (uint32_t)(s_g[a] & 0xFFFFFFFFu) ^ 0xFFFFFFFFu;
            const float sv = x[id] * inv_t;
            const float e = expf(sv - m);
            if (min_p > 0.0f && e < min_p) continue;
            cum += e;
            cut_last = id;
            if (cum >= target) break;
        }
        mnuc = cum;
    }
    ((float*)h)[PD_TOPP_MB_M] = mnuc;
    h[PD_TOPP_MB_LAST] = cut_last;
    h[PD_TOPP_MB_GC] = 0u; // draw gather starts a fresh list
    // phase E: draw quantile walk over the level-0 survivor masses
    const float t = ps[row].u * mnuc;
    float cum = 0.0f;
    for (uint32_t b = PD_TOPP_BUCKETS - 1u;; --b) {
        const float bm = s_massp[b];
        if (bm > 0.0f) {
            if (cum + bm >= t) {
                h[PD_TOPP_MB_REQD] = PD_TOPP_REQ_SET | b;
                ((float*)h)[PD_TOPP_MB_CUMD] = cum;
                ((float*)h)[PD_TOPP_MB_T] = t;
                break;
            }
            cum += bm;
        }
        if (b == 0u) break;
    }
}

// P9: F-tail - the draw's level-1 sub-bucket walk.
__global__ void __launch_bounds__(1024) pd_topp_mb_walk3_kernel(
    const PdSampleRow* __restrict__ ps, uint32_t* __restrict__ scr) {
    const uint32_t row = blockIdx.x, tid = threadIdx.x;
    if (ps[row].mode != 6u) return;
    uint32_t* h = scr + (size_t)row * PD_TOPP_MB_STRIDE;
    if (h[PD_TOPP_MB_DONE]) return;
    const uint32_t reqd = h[PD_TOPP_MB_REQD];
    if (!(reqd & PD_TOPP_REQ_SET)) return;
    __shared__ float s_m1[PD_TOPP_BUCKETS];
    const float* g1 = (const float*)(h + PD_TOPP_MB_M1);
    for (uint32_t b = tid; b < PD_TOPP_BUCKETS; b += blockDim.x)
        s_m1[b] = g1[b];
    __syncthreads();
    if (tid != 0) return;
    const uint32_t qb = reqd & (PD_TOPP_BUCKETS - 1u);
    float cum = ((const float*)h)[PD_TOPP_MB_CUMD];
    const float t = ((const float*)h)[PD_TOPP_MB_T];
    for (uint32_t b = PD_TOPP_BUCKETS - 1u;; --b) {
        const float bm = s_m1[b];
        if (bm > 0.0f) {
            if (cum + bm >= t) {
                h[PD_TOPP_MB_REQQ] = PD_TOPP_REQ_SET | ((qb << 10) | b);
                ((float*)h)[PD_TOPP_MB_CUMD] = cum;
                break;
            }
            cum += bm;
        }
        if (b == 0u) break;
    }
}

// P11: G-tail - final element walk over the sorted draw sub-bucket, then
// self-clean the whole row stride (the chain's rest-state contract).
__global__ void __launch_bounds__(1024) pd_topp_mb_fin_kernel(
    const float* __restrict__ logits, const PdSampleRow* __restrict__ ps,
    const PdSampleTrunc* __restrict__ pt, unsigned int* __restrict__ out,
    uint32_t* __restrict__ scr, uint32_t n) {
    const uint32_t row = blockIdx.x, tid = threadIdx.x;
    if (ps[row].mode != 6u) return;
    uint32_t* h = scr + (size_t)row * PD_TOPP_MB_STRIDE;
    if (!h[PD_TOPP_MB_DONE]) {
        __shared__ unsigned long long s_g[PD_TOPP_GCAP];
        const unsigned long long* g =
            (const unsigned long long*)(h + PD_TOPP_MB_G);
        const uint32_t len = min(h[PD_TOPP_MB_GC], PD_TOPP_GCAP);
        for (uint32_t i = tid; i < PD_TOPP_GCAP; i += blockDim.x)
            s_g[i] = i < len ? g[i] : 0ull;
        __syncthreads();
        pd_topk_mb_sortd(s_g, PD_TOPP_GCAP, tid);
        if (tid == 0) {
            uint32_t pick = h[PD_TOPP_MB_LAST];
            const uint32_t reqq = h[PD_TOPP_MB_REQQ];
            if (reqq & PD_TOPP_REQ_SET) {
                const float inv_t = ps[row].inv_t;
                const float min_p = pt[row].min_p;
                const float m = pd_okey_inv(
                    (uint32_t)(*(const unsigned long long*)h >> 32));
                const float* x = logits + (size_t)row * n;
                float cum = ((const float*)h)[PD_TOPP_MB_CUMD];
                const float t = ((const float*)h)[PD_TOPP_MB_T];
                for (uint32_t a = 0; a < len; ++a) {
                    const uint32_t id =
                        (uint32_t)(s_g[a] & 0xFFFFFFFFu) ^ 0xFFFFFFFFu;
                    const float sv = x[id] * inv_t;
                    const float e = expf(sv - m);
                    if (min_p > 0.0f && e < min_p) continue;
                    cum += e;
                    pick = id;
                    if (cum >= t) break;
                }
            }
            out[row] = pick;
        }
        __syncthreads();
    }
    for (uint32_t i = tid; i < PD_TOPP_MB_STRIDE; i += blockDim.x) h[i] = 0u;
}

// mb dispatch: any row count up to MAXR (rows=1 is the worst one-block
// case - the c1 pipe fallback lane), real vocab widths only.
static uint32_t* pd_topp_mb_scr(uint32_t rows, uint32_t n, void* stream) {
    static int no_mb = -1;
    if (no_mb < 0) no_mb = pd_env("PADDOCK_NO_TOPP_MB") ? 1 : 0;
    if (no_mb || rows > PD_TOPP_MB_MAXR || n < 4096u) return nullptr;
    uint32_t* scr = pd_topp_scr();
    if (!scr) {
        cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing((cudaStream_t)stream, &cs);
        if (cs == cudaStreamCaptureStatusNone) scr = pd_topp_scr(true);
    }
    static bool said = false;
    if (scr && !said) {
        said = true;
        puts("[topp-mb] engaged (multi-block k-less truncation chain)");
        fflush(stdout); // serve logs are block-buffered; don't lose the witness
    }
    return scr;
}

PD_EXPORT
int pd_sample_rows_p(const void* logits, const void* params,
                     const void* trunc_params, void* out, uint32_t rows,
                     uint32_t n, void* stream) {
    if (rows == 0 || n == 0) return 0;
    uint32_t* scr = pd_topp_mb_scr(rows, n, stream);
    if (scr) {
        const dim3 g(rows, PD_TOPP_MB_SEG);
        cudaStream_t st = (cudaStream_t)stream;
        const float* l = (const float*)logits;
        const PdSampleRow* ps = (const PdSampleRow*)params;
        const PdSampleTrunc* pt = (const PdSampleTrunc*)trunc_params;
        unsigned int* o = (unsigned int*)out;
        pd_topp_mb_max_kernel<<<g, 1024u, 0, st>>>(l, ps, scr, n);
        pd_topp_mb_hist_kernel<<<g, 1024u, 0, st>>>(l, ps, pt, scr, n);
        pd_topp_mb_walk1_kernel<<<rows, 1024u, 0, st>>>(ps, pt, o, scr);
        pd_topp_mb_ref_kernel<<<g, 1024u, 0, st>>>(l, ps, pt, scr, n, 0u);
        pd_topp_mb_walk2_kernel<<<rows, 1024u, 0, st>>>(ps, pt, scr);
        pd_topp_mb_gath_kernel<<<g, 1024u, 0, st>>>(l, ps, scr, n, 0u);
        pd_topp_mb_fincut_kernel<<<rows, 1024u, 0, st>>>(l, ps, pt, scr, n);
        pd_topp_mb_ref_kernel<<<g, 1024u, 0, st>>>(l, ps, pt, scr, n, 1u);
        pd_topp_mb_walk3_kernel<<<rows, 1024u, 0, st>>>(ps, scr);
        pd_topp_mb_gath_kernel<<<g, 1024u, 0, st>>>(l, ps, scr, n, 1u);
        pd_topp_mb_fin_kernel<<<rows, 1024u, 0, st>>>(l, ps, pt, o, scr, n);
        return pd_launch_status();
    }
    pd_sample_rows_p_kernel<<<rows, 1024u, 0, (cudaStream_t)stream>>>(
        (const float*)logits, (const PdSampleRow*)params,
        (const PdSampleTrunc*)trunc_params, (unsigned int*)out, n);
    return pd_launch_status();
}

// Pipelined-decode tick advance: the sampled tokens of tick N become tick
// N+1's model inputs without a host round trip - tokens[i] = out[i] and the
// per-row position bumps by one. Runs between tick N's sampler and tick N+1's
// step-graph replay so the serving loop can enqueue N+1 before N's ids reach
// the host. Hole rows advance too (their tokens/positions are ignored and the
// pipe's lifetime keeps them under max_ctx).
__global__ void pd_pipe_advance_kernel(const unsigned int* __restrict__ out,
                                       unsigned int* __restrict__ tokens,
                                       unsigned int* __restrict__ positions,
                                       uint32_t rows) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows) {
        tokens[i] = out[i];
        positions[i] += 1u;
    }
}

PD_EXPORT
int pd_pipe_advance(const void* out, void* tokens, void* positions,
                    uint32_t rows, void* stream) {
    if (rows == 0) return 0;
    uint32_t threads = 256;
    uint32_t blocks = (rows + threads - 1) / threads;
    pd_pipe_advance_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        (const unsigned int*)out, (unsigned int*)tokens, (unsigned int*)positions, rows);
    return pd_launch_status();
}

// Batched-spec conv-ext staging: ext[b] = [slot b's persistent window rows] ++
// [slot b's r chunk rows of mixed]. One launch replaces 2*B copy_regions per
// DeltaNet layer per verify round. wins [n_slots, km1, conv_dim]; mixed
// [B, r, conv_dim]; ext [B, km1 + r, conv_dim].
__global__ void pd_conv_ext_build_slots_kernel(
        const float* __restrict__ wins, const unsigned int* __restrict__ slots,
        const float* __restrict__ mixed, float* __restrict__ ext,
        uint32_t km1, uint32_t r, uint32_t conv_dim) {
    const uint32_t b = blockIdx.y;
    const uint32_t seg = km1 + r;
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (uint64_t)seg * conv_dim) return;
    const uint32_t row = (uint32_t)(idx / conv_dim);
    const uint32_t c = (uint32_t)(idx % conv_dim);
    float v;
    if (row < km1) {
        v = wins[((size_t)slots[b] * km1 + row) * conv_dim + c];
    } else {
        v = mixed[((size_t)b * r + (row - km1)) * conv_dim + c];
    }
    ext[((size_t)b * seg + row) * conv_dim + c] = v;
}

PD_EXPORT
int pd_conv_ext_build_slots(const void* wins, const void* slots, const void* mixed,
                            void* ext, uint32_t batch, uint32_t km1, uint32_t r,
                            uint32_t conv_dim, void* stream) {
    if (batch == 0 || conv_dim == 0 || km1 + r == 0) return 0;
    uint64_t per_seg = (uint64_t)(km1 + r) * conv_dim;
    uint32_t threads = 256;
    dim3 grid((uint32_t)((per_seg + threads - 1) / threads), batch);
    pd_conv_ext_build_slots_kernel<<<grid, threads, 0, (cudaStream_t)stream>>>(
        (const float*)wins, (const unsigned int*)slots, (const float*)mixed,
        (float*)ext, km1, r, conv_dim);
    return pd_launch_status();
}

// Depthwise causal conv1d + SiLU over per-slot EXTENDED segments: for each slot
// segment ext[b] = [km1 window rows ++ r chunk rows], emit only the r real
// rows: out[b][t] = conv(ext[b][t .. t+k)). Replaces conv-over-ext + copy-back
// in the batched verify pass. w layout w[c*k+kk] as in pd_causal_conv1d_silu.
__global__ void pd_conv_chunk_ext_kernel(
        const float* __restrict__ ext, const float* __restrict__ w,
        float* __restrict__ out, uint32_t km1, uint32_t r, uint32_t conv_dim,
        uint32_t k) {
    const uint32_t b = blockIdx.y;
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (uint64_t)r * conv_dim) return;
    const uint32_t t = (uint32_t)(idx / conv_dim);
    const uint32_t c = (uint32_t)(idx % conv_dim);
    const uint32_t seg = km1 + r;
    const float* xrow = ext + ((size_t)b * seg + t) * conv_dim + c;
    float acc = 0.0f;
    for (uint32_t kk = 0; kk < k; ++kk) acc += w[(size_t)c * k + kk] * xrow[(size_t)kk * conv_dim];
    out[((size_t)b * r + t) * conv_dim + c] = acc / (1.0f + expf(-acc));
}

PD_EXPORT
int pd_conv_chunk_ext(const void* ext, const void* w, void* out, uint32_t batch,
                      uint32_t km1, uint32_t r, uint32_t conv_dim, uint32_t k,
                      void* stream) {
    if (batch == 0 || r == 0 || conv_dim == 0) return 0;
    if (k != km1 + 1u) return cudaErrorInvalidValue;
    uint64_t per_seg = (uint64_t)r * conv_dim;
    uint32_t threads = 256;
    dim3 grid((uint32_t)((per_seg + threads - 1) / threads), batch);
    pd_conv_chunk_ext_kernel<<<grid, threads, 0, (cudaStream_t)stream>>>(
        (const float*)ext, (const float*)w, (float*)out, km1, r, conv_dim, k);
    return pd_launch_status();
}

// Ragged per-slot spec commit, state half: for every seq b whose accepted count
// is short of the chunk (committed[b] < r), roll slot slots[b]'s recurrent state
// back to snapshot row committed[b]-1. Snapshot layout matches the v2 kernel:
// [B, r, n_heads, D, D] t-major, transposed (column-contiguous) tiles - the copy
// is tile-for-tile so the transposition is invisible here. One launch replaces
// up to B*n_layers copy_regions per round.
template <typename ST = float>
__global__ void pd_state_restore_slots_kernel(
        ST* __restrict__ states, const ST* __restrict__ snap,
        const unsigned int* __restrict__ slots, const unsigned int* __restrict__ committed,
        uint32_t r, uint32_t n_heads, uint32_t D) {
    const uint32_t h = blockIdx.x;
    const uint32_t b = blockIdx.y;
    const uint32_t c = committed[b];
    if (c >= r) return;                          // whole chunk stood: state is current
    // tile-for-tile raw copy; dtype only changes the byte count
    const uint4* src = reinterpret_cast<const uint4*>(
        snap + (((size_t)b * r + (c - 1)) * n_heads + h) * (size_t)D * D);
    uint4* dst = reinterpret_cast<uint4*>(
        states + ((size_t)slots[b] * n_heads + h) * (size_t)D * D);
    const uint32_t n16 = (uint32_t)(((size_t)D * D * sizeof(ST)) >> 4);
    for (uint32_t i = threadIdx.x; i < n16; i += blockDim.x) dst[i] = src[i];
}

PD_EXPORT
int pd_state_restore_slots(void* states, const void* snap, const void* slots,
                           const void* committed, uint32_t batch, uint32_t r,
                           uint32_t n_heads, uint32_t head_dim, void* stream) {
    if (batch == 0 || n_heads == 0) return 0;
    if (head_dim & 3u) return cudaErrorInvalidValue;
    dim3 grid(n_heads, batch);
    const int dns_cls = pd_dns_state_class();
    if (dns_cls == 3)
        pd_state_restore_slots_kernel<__nv_fp8_e4m3><<<grid, 256, 0,
                                                       (cudaStream_t)stream>>>(
            (__nv_fp8_e4m3*)states, (const __nv_fp8_e4m3*)snap,
            (const unsigned int*)slots, (const unsigned int*)committed, r, n_heads,
            head_dim);
    else if (dns_cls == 2)
        pd_state_restore_slots_kernel<__half><<<grid, 256, 0,
                                                (cudaStream_t)stream>>>(
            (__half*)states, (const __half*)snap,
            (const unsigned int*)slots, (const unsigned int*)committed, r, n_heads,
            head_dim);
    else if (dns_cls == 1)
        pd_state_restore_slots_kernel<__nv_bfloat16><<<grid, 256, 0,
                                                       (cudaStream_t)stream>>>(
            (__nv_bfloat16*)states, (const __nv_bfloat16*)snap,
            (const unsigned int*)slots, (const unsigned int*)committed, r, n_heads,
            head_dim);
    else
        pd_state_restore_slots_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(
            (float*)states, (const float*)snap, (const unsigned int*)slots,
            (const unsigned int*)committed, r, n_heads, head_dim);
    return pd_launch_status();
}

// Ragged per-slot spec commit, conv half: slot slots[b]'s persistent window
// becomes ext[b] rows [committed[b], committed[b] + km1) - the km1 rows ending
// at the last committed token.
__global__ void pd_conv_commit_slots_kernel(
        const float* __restrict__ ext, float* __restrict__ wins,
        const unsigned int* __restrict__ slots, const unsigned int* __restrict__ committed,
        uint32_t km1, uint32_t r, uint32_t conv_dim) {
    const uint32_t b = blockIdx.y;
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (uint64_t)km1 * conv_dim) return;
    const uint32_t j = (uint32_t)(idx / conv_dim);
    const uint32_t c = (uint32_t)(idx % conv_dim);
    const uint32_t seg = km1 + r;
    wins[((size_t)slots[b] * km1 + j) * conv_dim + c] =
        ext[((size_t)b * seg + committed[b] + j) * conv_dim + c];
}

PD_EXPORT
int pd_conv_commit_slots(const void* ext, void* wins, const void* slots,
                         const void* committed, uint32_t batch, uint32_t km1,
                         uint32_t r, uint32_t conv_dim, void* stream) {
    if (batch == 0 || km1 == 0 || conv_dim == 0) return 0;
    uint64_t per = (uint64_t)km1 * conv_dim;
    uint32_t threads = 256;
    dim3 grid((uint32_t)((per + threads - 1) / threads), batch);
    pd_conv_commit_slots_kernel<<<grid, threads, 0, (cudaStream_t)stream>>>(
        (const float*)ext, (float*)wins, (const unsigned int*)slots,
        (const unsigned int*)committed, km1, r, conv_dim);
    return pd_launch_status();
}

// Slot-indexed gated delta recurrence - the continuous-batching decode step: B
// sequences each advance their own recurrent state by one token. grid (n_heads,
// B); block (h, b) reads q/k/v row b ([B, n_heads, D]) and read-modify-writes
// states[slots[b]][h] ([n_slots, n_heads, D, D]). Same math/order as the
// single-sequence kernel at n_tokens=1.
template <typename ST = float>
__global__ void pd_gated_delta_recurrent_slots_kernel(
        const float* __restrict__ q, const float* __restrict__ k,
        const float* __restrict__ v, const float* __restrict__ g,
        const float* __restrict__ beta, const unsigned int* __restrict__ slots,
        ST* __restrict__ states, float* __restrict__ out,
        uint32_t n_heads, uint32_t D) {
    PD_PDL_ARM();  // consumer-safe for early PDL launches (2026-08-31)
    const uint32_t h = blockIdx.x;
    const uint32_t b = blockIdx.y;
    const uint32_t j = threadIdx.x;
    if (h >= n_heads || j >= D) return;

    extern __shared__ float smem[];
    float* q_sh = smem;
    float* k_sh = smem + D;
    float* red  = smem + 2 * D;
    const float scale = rsqrtf((float)D);

    ST* s_head = states + ((size_t)slots[b] * n_heads + h) * (size_t)D * D;
    const size_t base = ((size_t)b * n_heads + h) * (size_t)D;
    const float qj = q[base + j];
    const float kj = k[base + j];
    const float vj = v[base + j];

    q_sh[j] = qj * qj;
    k_sh[j] = kj * kj;
    __syncthreads();
    for (uint32_t s = D >> 1; s > 0; s >>= 1) {
        if (j < s) { q_sh[j] += q_sh[j + s]; k_sh[j] += k_sh[j + s]; }
        __syncthreads();
    }
    if (j == 0) { red[0] = rsqrtf(q_sh[0] + 1e-6f); red[1] = rsqrtf(k_sh[0] + 1e-6f); }
    __syncthreads();
    q_sh[j] = qj * red[0] * scale;
    k_sh[j] = kj * red[1];
    __syncthreads();

    const float g_t = expf(g[(size_t)b * n_heads + h]);
    const float beta_t = beta[(size_t)b * n_heads + h];

    // this thread owns state column j; the column is small (D) - load, update, store
    float u = 0.0f;
    float col[PD_DN_MAX_D];
    for (uint32_t i = 0; i < D; ++i) {
        col[i] = pd_dns_ld(s_head + (size_t)i * D + j) * g_t;
        u += col[i] * k_sh[i];
    }
    const float delta = beta_t * (vj - u);
    float o = 0.0f;
    for (uint32_t i = 0; i < D; ++i) {
        col[i] += k_sh[i] * delta;
        o += col[i] * q_sh[i];
        pd_dns_st(s_head + (size_t)i * D + j, col[i]);
    }
    out[base + j] = o;
}

// Compile-time-D twin (rung 5: the GDN spill). The runtime-D kernel above
// declares `float col[PD_DN_MAX_D]` = 512 B per thread; ncu measures 32
// registers/thread, so it cannot be in registers - it spills to local memory,
// which is DRAM-backed. Measured on the live c32 tick:
//   DRAM read 202.6 MB + write 243.1 MB = 445.7 MB per launch
//   the algorithm's minimum (state 100.7 MB, read once + written once) = 201.3
//   spill accounts for the rest: 128 floats x 4 B x (w+r) x 128 thr x 1536 blk
//                                = 201.3 MB, predicted total 402.7 vs 445.7
// It is not coalescing (load sectors/request 3.93 of 4.00) and not occupancy
// (58.8% achieved of a 100% theoretical); the loop already touches the state
// exactly once. It is one local array.
//
// For scale: the rival's fused_recurrent_gated_delta_rule_packed_decode moves
// the same 201 MB in 35.1 us = 5.74 TB/s = 72% of roof; we move 445.7 MB at
// 29%, which is the whole 3.11 vs 1.25 ms/step band gap.
//
// With D a template parameter the array is indexable at compile time and stays
// in registers. That costs occupancy - the single-sequence twin
// (pd_gated_delta_recurrent_kernel_t) lands at 255 reg/thread and 2 blocks/SM -
// so this is elected by MEASUREMENT, not by the byte argument alone:
// PADDOCK_DN_SLOTS_GENERIC=1 reverts to the runtime-D kernel for an A/B.
template <typename ST, uint32_t D>
__global__ __launch_bounds__(D) void pd_gated_delta_recurrent_slots_kernel_t(
        const float* __restrict__ q, const float* __restrict__ k,
        const float* __restrict__ v, const float* __restrict__ g,
        const float* __restrict__ beta, const unsigned int* __restrict__ slots,
        ST* __restrict__ states, float* __restrict__ out, uint32_t n_heads,
        // slot 564: the qwen4_exp gated norm fused into this epilogue. The
        // norm's row is this block's head output (grid (heads, batch), D
        // threads), so its reduction is block-local. Same tree, same
        // 1/sqrt(sum/D + eps), same w[j]*(x*inv)*sigmoid(z) => bit-identical.
        // gn_w == nullptr keeps the plain `out[base+j] = o` store.
        const float* __restrict__ gn_z = nullptr,
        const float* __restrict__ gn_w = nullptr,
        __nv_bfloat16* __restrict__ gn_out16 = nullptr,
        float gn_eps = 0.0f) {
    const uint32_t h = blockIdx.x;
    const uint32_t b = blockIdx.y;
    const uint32_t j = threadIdx.x;
    if (h >= n_heads) return;

    extern __shared__ float smem[];
    float* q_sh = smem;
    float* k_sh = smem + D;
    float* red  = smem + 2 * D;
    const float scale = rsqrtf((float)D);

    ST* s_head = states + ((size_t)slots[b] * n_heads + h) * (size_t)D * D;
    const size_t base = ((size_t)b * n_heads + h) * (size_t)D;
    const float qj = q[base + j];
    const float kj = k[base + j];
    const float vj = v[base + j];

    q_sh[j] = qj * qj;
    k_sh[j] = kj * kj;
    __syncthreads();
    #pragma unroll
    for (uint32_t s = D >> 1; s > 0; s >>= 1) {
        if (j < s) { q_sh[j] += q_sh[j + s]; k_sh[j] += k_sh[j + s]; }
        __syncthreads();
    }
    if (j == 0) { red[0] = rsqrtf(q_sh[0] + 1e-6f); red[1] = rsqrtf(k_sh[0] + 1e-6f); }
    __syncthreads();
    q_sh[j] = qj * red[0] * scale;
    k_sh[j] = kj * red[1];
    __syncthreads();

    const float g_t = expf(g[(size_t)b * n_heads + h]);
    const float beta_t = beta[(size_t)b * n_heads + h];

    // identical arithmetic to the runtime-D kernel, in the identical order -
    // only col[]'s storage class changes, so this is bit-exact by construction
    float u = 0.0f;
    float col[D];
    #pragma unroll
    for (uint32_t i = 0; i < D; ++i) {
        col[i] = pd_dns_ld(s_head + (size_t)i * D + j) * g_t;
        u += col[i] * k_sh[i];
    }
    const float delta = beta_t * (vj - u);
    float o = 0.0f;
    #pragma unroll
    for (uint32_t i = 0; i < D; ++i) {
        col[i] += k_sh[i] * delta;
        o += col[i] * q_sh[i];
        pd_dns_st(s_head + (size_t)i * D + j, col[i]);
    }
    if (gn_w == nullptr) {
        out[base + j] = o;
        return;
    }
    __syncthreads();
    q_sh[j] = o * o;
    __syncthreads();
    #pragma unroll
    for (uint32_t s = D >> 1; s > 0; s >>= 1) {
        if (j < s) q_sh[j] += q_sh[j + s];
        __syncthreads();
    }
    const float gn_inv = 1.0f / sqrtf(q_sh[0] / (float)D + gn_eps);
    const float y = gn_w[j] * (o * gn_inv) *
                    (1.0f / (1.0f + expf(-gn_z[base + j])));
    out[base + j] = y;
    if (gn_out16) gn_out16[base + j] = __float2bfloat16(y);
}

static bool pd_dn_slots_generic_env() {
    static const bool v = pd_env("PADDOCK_DN_SLOTS_GENERIC") != nullptr;
    return v;
}

static int pd_gdrs_impl(const void* q, const void* k, const void* v,
                                   const void* g, const void* beta, const void* slots,
                                   void* states, void* out, uint32_t batch,
                                   uint32_t n_heads, uint32_t head_dim, const void* gn_z, const void* gn_w,
                        void* gn_out16, float gn_eps, void* stream) {
    if (batch == 0 || n_heads == 0 || head_dim == 0) return 0;
    if (head_dim > PD_DN_MAX_D) return cudaErrorInvalidValue;
    size_t shmem = ((size_t)2 * head_dim + 2) * sizeof(float);
    dim3 grid(n_heads, batch);
    const int dns_cls = pd_dns_state_class();
    // compile-time-D arm first: same arithmetic in the same order, only col[]'s
    // storage class changes (local-memory spill -> registers), which is worth
    // 2.21x the kernel's DRAM traffic. See the kernel's header note.
#define PD_DNS_T(TY, CAST, DD)                                                 \
    do {                                                                       \
        pd_gated_delta_recurrent_slots_kernel_t<TY, (DD)>                      \
            <<<grid, (DD), shmem, (cudaStream_t)stream>>>(                     \
                (const float*)q, (const float*)k, (const float*)v,             \
                (const float*)g, (const float*)beta,                           \
                (const unsigned int*)slots, (CAST*)states, (float*)out,        \
                n_heads, (const float*)gn_z, (const float*)gn_w,               \
                (__nv_bfloat16*)gn_out16, gn_eps);                             \
        return pd_launch_status();                                             \
    } while (0)
    if (!pd_dn_slots_generic_env() && (head_dim == 128u || head_dim == 64u)) {
        if (head_dim == 128u) {
            if (dns_cls == 3) PD_DNS_T(__nv_fp8_e4m3, __nv_fp8_e4m3, 128u);
            if (dns_cls == 2) PD_DNS_T(__half, __half, 128u);
            if (dns_cls == 1) PD_DNS_T(__nv_bfloat16, __nv_bfloat16, 128u);
            PD_DNS_T(float, float, 128u);
        } else {
            if (dns_cls == 3) PD_DNS_T(__nv_fp8_e4m3, __nv_fp8_e4m3, 64u);
            if (dns_cls == 2) PD_DNS_T(__half, __half, 64u);
            if (dns_cls == 1) PD_DNS_T(__nv_bfloat16, __nv_bfloat16, 64u);
            PD_DNS_T(float, float, 64u);
        }
    }
#undef PD_DNS_T
    if (dns_cls == 3)
        pd_gated_delta_recurrent_slots_kernel<__nv_fp8_e4m3><<<grid, head_dim, shmem, (cudaStream_t)stream>>>(
        (const float*)q, (const float*)k, (const float*)v, (const float*)g,
        (const float*)beta, (const unsigned int*)slots, (__nv_fp8_e4m3*)states, (float*)out,
        n_heads, head_dim);
    else if (dns_cls == 2)
        pd_gated_delta_recurrent_slots_kernel<__half><<<grid, head_dim, shmem, (cudaStream_t)stream>>>(
        (const float*)q, (const float*)k, (const float*)v, (const float*)g,
        (const float*)beta, (const unsigned int*)slots, (__half*)states, (float*)out,
        n_heads, head_dim);
    else if (dns_cls == 1)
        pd_gated_delta_recurrent_slots_kernel<__nv_bfloat16><<<grid, head_dim, shmem, (cudaStream_t)stream>>>(
        (const float*)q, (const float*)k, (const float*)v, (const float*)g,
        (const float*)beta, (const unsigned int*)slots, (__nv_bfloat16*)states, (float*)out,
        n_heads, head_dim);
    else
        pd_gated_delta_recurrent_slots_kernel<<<grid, head_dim, shmem, (cudaStream_t)stream>>>(
        (const float*)q, (const float*)k, (const float*)v, (const float*)g,
        (const float*)beta, (const unsigned int*)slots, (float*)states, (float*)out,
        n_heads, head_dim);
    return pd_launch_status();
}

PD_EXPORT
int pd_gated_delta_recurrent_slots(const void* q, const void* k, const void* v,
                                   const void* g, const void* beta, const void* slots,
                                   void* states, void* out, uint32_t batch,
                                   uint32_t n_heads, uint32_t head_dim, void* stream) {
    return pd_gdrs_impl(q, k, v, g, beta, slots, states, out, batch, n_heads,
                        head_dim, nullptr, nullptr, nullptr, 0.0f, stream);
}

// slot 564: the same recurrence with qwen4_exp's gated norm folded into its
// epilogue (2.2 us a layer on the critical branch of 36 of 48 layers). Only
// the compile-time-D arm carries the fold, so an unsupported geometry declines
// with -1 and the caller runs the pair.
PD_EXPORT
int pd_gated_delta_recurrent_slots_gn(const void* q, const void* k, const void* v,
                                      const void* g, const void* beta,
                                      const void* slots, void* states, void* out,
                                      const void* gn_z, const void* gn_w,
                                      void* gn_out16, float gn_eps, uint32_t batch,
                                      uint32_t n_heads, uint32_t head_dim,
                                      void* stream) {
    static const bool off = [] {
        const char* e = pd_env("PADDOCK_Q38FN_GDN_GN");
        return e && e[0] == '0';
    }();
    if (off || gn_w == nullptr || pd_dn_slots_generic_env() ||
        !(head_dim == 128u || head_dim == 64u)) {
        return -1;
    }
    return pd_gdrs_impl(q, k, v, g, beta, slots, states, out, batch, n_heads,
                        head_dim, gn_z, gn_w, gn_out16, gn_eps, stream);
}

// Slot-indexed single-token conv+silu: B sequences advance their own persistent
// window. wins [n_slots, (k-1), conv_dim]; x_new/out [B, conv_dim].
__global__ void pd_conv_step_slots_kernel(float* __restrict__ wins,
                                          const float* __restrict__ x_new,
                                          const float* __restrict__ w,
                                          float* __restrict__ out,
                                          const unsigned int* __restrict__ slots,
                                          uint32_t conv_dim, uint32_t k,
                                          uint32_t x_stride) {
    PD_PDL_ARM();  // consumer-safe for early PDL launches (2026-08-31)
    uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t b = blockIdx.y;
    if (c >= conv_dim) return;
    uint32_t km1 = k - 1u;
    float* win = wins + (size_t)slots[b] * km1 * conv_dim;
    const float* xb = x_new + (size_t)b * x_stride;
    float vals[PD_CONV_K_MAX];
    for (uint32_t j = 0; j < km1; ++j) vals[j] = win[(size_t)j * conv_dim + c];
    vals[km1] = xb[c];
    float acc = 0.0f;
    for (uint32_t j = 0; j < k; ++j) acc += w[(size_t)c * k + j] * vals[j];
    out[(size_t)b * conv_dim + c] = acc / (1.0f + expf(-acc));
    for (uint32_t j = 1; j < km1; ++j) win[(size_t)(j - 1) * conv_dim + c] = vals[j];
    if (km1 >= 1u) win[(size_t)(km1 - 1) * conv_dim + c] = vals[km1];
}

// k=4 float4 twin: 4 channels/thread, one float4 per window row / x / out and
// one per channel of w ([c,4] is float4-shaped). Same loads, same stores, FMA
// chain in the scalar kernel's order - bit-identical output and window. 8.83 ->
// 6.14 us at the c32 shape (b=32, conv_dim=10240, L2-cold windows); what is
// left is the k-1 shift writes, which only a ring window (persistent
// per-slot phase - wide state surface: spec stash, prefix save, multimodal)
// would remove. Kill: PADDOCK_NO_CONV_V4.
__global__ void pd_conv_step_slots_k4v4_kernel(float* __restrict__ wins,
                                               const float* __restrict__ x_new,
                                               const float* __restrict__ w,
                                               float* __restrict__ out,
                                               const unsigned int* __restrict__ slots,
                                               uint32_t n4, uint32_t xs4) {
    uint32_t c4 = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t b = blockIdx.y;
    if (c4 >= n4) return;
    float4* win = (float4*)wins + (size_t)slots[b] * 3u * n4;
    float4 v0 = win[c4];
    float4 v1 = win[n4 + c4];
    float4 v2 = win[2u * n4 + c4];
    float4 v3 = ((const float4*)x_new + (size_t)b * xs4)[c4];
    const float4* w4 = (const float4*)w;
    float4 w0 = w4[c4 * 4 + 0], w1 = w4[c4 * 4 + 1], w2 = w4[c4 * 4 + 2], w3 = w4[c4 * 4 + 3];
    float4 o;
    o.x = __fmaf_rn(w0.w, v3.x, __fmaf_rn(w0.z, v2.x, __fmaf_rn(w0.y, v1.x, w0.x * v0.x)));
    o.y = __fmaf_rn(w1.w, v3.y, __fmaf_rn(w1.z, v2.y, __fmaf_rn(w1.y, v1.y, w1.x * v0.y)));
    o.z = __fmaf_rn(w2.w, v3.z, __fmaf_rn(w2.z, v2.z, __fmaf_rn(w2.y, v1.z, w2.x * v0.z)));
    o.w = __fmaf_rn(w3.w, v3.w, __fmaf_rn(w3.z, v2.w, __fmaf_rn(w3.y, v1.w, w3.x * v0.w)));
    o.x = o.x / (1.0f + expf(-o.x));
    o.y = o.y / (1.0f + expf(-o.y));
    o.z = o.z / (1.0f + expf(-o.z));
    o.w = o.w / (1.0f + expf(-o.w));
    ((float4*)out + (size_t)b * n4)[c4] = o;
    win[c4] = v1;
    win[n4 + c4] = v2;
    win[2u * n4 + c4] = v3;
}

// SPLIT-FUSED twin of the k=4 float4 kernel (slot 563). The GDN conv output is
// consumed by exactly one kernel -- q4x_gdn_split_widen, which re-reads the
// whole row and scatters it into q, k (repeat-interleave widened from hk key
// heads to hv value heads) and v. Writing those destinations straight from the
// conv epilogue retires that launch (2.7 us on the CRITICAL branch of every
// GDN layer, 36 of 48 layers) and the conv row's 40 KB round-trip. Values are
// the k4v4 kernel's verbatim, so q/k/v are bit-identical.
//
// Region map (channel base cb of a float4, all boundaries 4-aligned by the
// launcher's guard): cb < kdim -> q head cb/kd, replicated to the R = hv/hk
// value heads it serves; cb < 2*kdim -> k, same replication; else -> v, 1:1.
__global__ void pd_conv_step_slots_k4v4_split_kernel(
        float* __restrict__ wins, const float* __restrict__ x_new,
        const float* __restrict__ w, float* __restrict__ q,
        float* __restrict__ kk, float* __restrict__ v,
        const unsigned int* __restrict__ slots, uint32_t n4, uint32_t xs4,
        uint32_t kdim, uint32_t hk, uint32_t hv, uint32_t kd, uint32_t vd) {
    PD_PDL_ARM();
    uint32_t c4 = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t b = blockIdx.y;
    if (c4 >= n4) return;
    float4* win = (float4*)wins + (size_t)slots[b] * 3u * n4;
    float4 v0 = win[c4];
    float4 v1 = win[n4 + c4];
    float4 v2 = win[2u * n4 + c4];
    float4 v3 = ((const float4*)x_new + (size_t)b * xs4)[c4];
    const float4* w4 = (const float4*)w;
    float4 w0 = w4[c4 * 4 + 0], w1 = w4[c4 * 4 + 1], w2 = w4[c4 * 4 + 2], w3 = w4[c4 * 4 + 3];
    float4 o;
    o.x = __fmaf_rn(w0.w, v3.x, __fmaf_rn(w0.z, v2.x, __fmaf_rn(w0.y, v1.x, w0.x * v0.x)));
    o.y = __fmaf_rn(w1.w, v3.y, __fmaf_rn(w1.z, v2.y, __fmaf_rn(w1.y, v1.y, w1.x * v0.y)));
    o.z = __fmaf_rn(w2.w, v3.z, __fmaf_rn(w2.z, v2.z, __fmaf_rn(w2.y, v1.z, w2.x * v0.z)));
    o.w = __fmaf_rn(w3.w, v3.w, __fmaf_rn(w3.z, v2.w, __fmaf_rn(w3.y, v1.w, w3.x * v0.w)));
    o.x = o.x / (1.0f + expf(-o.x));
    o.y = o.y / (1.0f + expf(-o.y));
    o.z = o.z / (1.0f + expf(-o.z));
    o.w = o.w / (1.0f + expf(-o.w));
    win[c4] = v1;
    win[n4 + c4] = v2;
    win[2u * n4 + c4] = v3;
    const uint32_t cb = c4 * 4u;
    const uint32_t R = hv / hk;
    if (cb < 2u * kdim) {
        float* dst = (cb < kdim) ? q : kk;
        const uint32_t base = (cb < kdim) ? cb : cb - kdim;
        const uint32_t kh = base / kd, off = base - kh * kd;
        for (uint32_t j = 0; j < R; ++j) {
            const uint32_t vh = kh * R + j;
            *(float4*)(dst + ((size_t)b * hv + vh) * kd + off) = o;
        }
    } else {
        const uint32_t off = cb - 2u * kdim;
        *(float4*)(v + (size_t)b * hv * vd + off) = o;
    }
}

PD_EXPORT
int pd_conv_step_slots_split(void* wins, const void* x_new, const void* w,
                             void* q, void* k_out, void* v_out, const void* slots,
                             uint32_t batch, uint32_t conv_dim, uint32_t kconv,
                             uint32_t x_stride, uint32_t hk, uint32_t hv,
                             uint32_t kd, uint32_t vd, void* stream) {
    if (batch == 0 || conv_dim == 0) return 0;
    static const bool off = [] { return pd_env("PADDOCK_Q38FN_CONV_SPLIT") &&
                                        pd_env("PADDOCK_Q38FN_CONV_SPLIT")[0] == '0'; }();
    // geometry guard: float4 units must stay inside one region and one head
    if (off || kconv != 4u || hk == 0u || hv % hk != 0u || (kd & 3u) != 0u ||
        (vd & 3u) != 0u || (conv_dim & 3u) != 0u || (x_stride & 3u) != 0u ||
        conv_dim != 2u * hk * kd + hv * vd) {
        return -1;
    }
    const uint32_t threads = 256u, n4 = conv_dim >> 2;
    dim3 grid((n4 + threads - 1u) / threads, batch);
    pd_conv_step_slots_k4v4_split_kernel<<<grid, threads, 0, (cudaStream_t)stream>>>(
        (float*)wins, (const float*)x_new, (const float*)w, (float*)q,
        (float*)k_out, (float*)v_out, (const unsigned int*)slots, n4,
        x_stride >> 2, hk * kd, hk, hv, kd, vd);
    return pd_launch_status();
}

static int pd_conv_step_slots_impl(void* wins, const void* x_new, const void* w,
                                   void* out, const void* slots, uint32_t batch,
                                   uint32_t conv_dim, uint32_t k,
                                   uint32_t x_stride, void* stream) {
    if (batch == 0 || conv_dim == 0) return 0;
    if (k > PD_CONV_K_MAX) return cudaErrorInvalidValue;
    uint32_t threads = 256;
    static const bool v4_ok = [] { return !pd_env("PADDOCK_NO_CONV_V4"); }();
    if (v4_ok && k == 4 && (conv_dim & 3u) == 0 && (x_stride & 3u) == 0) {
        uint32_t n4 = conv_dim >> 2;
        dim3 grid((n4 + threads - 1) / threads, batch);
        pd_conv_step_slots_k4v4_kernel<<<grid, threads, 0, (cudaStream_t)stream>>>(
            (float*)wins, (const float*)x_new, (const float*)w, (float*)out,
            (const unsigned int*)slots, n4, x_stride >> 2);
        return pd_launch_status();
    }
    dim3 grid((conv_dim + threads - 1) / threads, batch);
    pd_conv_step_slots_kernel<<<grid, threads, 0, (cudaStream_t)stream>>>(
        (float*)wins, (const float*)x_new, (const float*)w, (float*)out,
        (const unsigned int*)slots, conv_dim, k, x_stride);
    return pd_launch_status();
}

PD_EXPORT
int pd_conv_step_slots(void* wins, const void* x_new, const void* w, void* out,
                       const void* slots, uint32_t batch, uint32_t conv_dim, uint32_t k,
                       void* stream) {
    return pd_conv_step_slots_impl(wins, x_new, w, out, slots, batch, conv_dim,
                                   k, conv_dim, stream);
}

// x_new read STRIDED out of a wider fused plane (the DN in-proj
// landing) - kills the row_slice copy of the conv half. Same loads by
// value, so output and window are bit-identical to slice-then-conv.
PD_EXPORT
int pd_conv_step_slots_s(void* wins, const void* x_new, const void* w, void* out,
                         const void* slots, uint32_t batch, uint32_t conv_dim,
                         uint32_t k, uint32_t x_stride, void* stream) {
    if (x_stride < conv_dim) return cudaErrorInvalidValue;
    return pd_conv_step_slots_impl(wins, x_new, w, out, slots, batch, conv_dim,
                                   k, x_stride, stream);
}

// Dequant a repacked Q8_0 weight into a dense f16 matrix (prefill staging for the
// cuBLAS tensor-core GEMM - llama.cpp's own large-batch route). out[i] =
// (f16)(data[i] * scale[i/32]); product computed in f32 then rounded once.
__global__ void pd_q8_0_repacked_to_f16_kernel(const int8_t* __restrict__ data,
                                               const __half* __restrict__ scale,
                                               __half* __restrict__ out, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __float2half((float)data[i] * __half2float(scale[i >> 5]));
}

PD_EXPORT
int pd_q8_0_repacked_to_f16(const void* data, const void* scale, void* out, uint64_t n,
                            void* stream) {
    if (n == 0) return 0;
    uint32_t threads = 256;
    uint64_t blocks = (n + threads - 1) / threads;
    pd_q8_0_repacked_to_f16_kernel<<<(uint32_t)blocks, threads, 0, (cudaStream_t)stream>>>(
        (const int8_t*)data, (const __half*)scale, (__half*)out, n);
    return pd_launch_status();
}

// Batched embedding gather: out[t] = table[tokens[t]] for t in [0, n_tokens).
// grid (ceil(embd/256), n_tokens) - the prefill analog of pd_embed_gather.
__global__ void pd_embed_gather_batch_kernel(const float* __restrict__ table,
                                             const uint32_t* __restrict__ tokens,
                                             float* __restrict__ out, uint32_t embd) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t t = blockIdx.y;
    if (i >= embd) return;
    out[(size_t)t * embd + i] = table[(size_t)tokens[t] * embd + i];
}

PD_EXPORT
int pd_embed_gather_batch(const void* table, const void* tokens, void* out, uint32_t embd,
                          uint32_t n_tokens, void* stream) {
    if (embd == 0 || n_tokens == 0) return 0;
    uint32_t threads = 256;
    dim3 grid((embd + threads - 1) / threads, n_tokens);
    pd_embed_gather_batch_kernel<<<grid, threads, 0, (cudaStream_t)stream>>>(
        (const float*)table, (const uint32_t*)tokens, (float*)out, embd);
    return pd_launch_status();
}

// Q8_0 twin of pd_embed_gather_batch with a fused output scale: gathers each
// device-selected token's row STRAIGHT from the Q8_0 embedding (4x fewer
// bytes than an f32 table, no 5+ GB dequant copy) and multiplies by `scale`
// (gemma4 bakes sqrt(n_embd) here). Graph-capturable: token ids come from
// device memory.
__global__ void pd_embed_gather_q8_kernel(const uint8_t* __restrict__ table,
                                          const uint32_t* __restrict__ tokens,
                                          float* __restrict__ out, uint32_t embd,
                                          float scale) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t t = blockIdx.y;
    if (i >= embd) return;
    // Q8_0: 34-byte blocks of 32 elems (f16 scale + 32 int8)
    const uint8_t* row = table + (size_t)tokens[t] * (embd / 32) * 34;
    const uint8_t* blk = row + (i >> 5) * 34;
    __half h;
    memcpy(&h, blk, sizeof(h));
    float d = __half2float(h);
    out[(size_t)t * embd + i] = (float)((int8_t)blk[2 + (i & 31u)]) * d * scale;
}

PD_EXPORT
int pd_embed_gather_q8(const void* table, const void* tokens, void* out, uint32_t embd,
                       uint32_t n_tokens, float scale, void* stream) {
    if (embd == 0 || n_tokens == 0) return 0;
    uint32_t threads = 256;
    dim3 grid((embd + threads - 1) / threads, n_tokens);
    pd_embed_gather_q8_kernel<<<grid, threads, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)table, (const uint32_t*)tokens, (float*)out, embd, scale);
    return pd_launch_status();
}

// Gather one embedding row selected by a device token id (graph-capturable).
__global__ void pd_embed_gather_kernel(const float* __restrict__ table,
                                       const uint32_t* __restrict__ token,
                                       float* __restrict__ out, uint32_t embd) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= embd) return;
    out[i] = table[(size_t)token[0] * embd + i];
}

PD_EXPORT
int pd_embed_gather(const void* table, const void* token, void* out, uint32_t embd,
                    void* stream) {
    if (embd == 0) return 0;
    uint32_t threads = 256;
    uint32_t blocks = (embd + threads - 1) / threads;
    pd_embed_gather_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        (const float*)table, (const uint32_t*)token, (float*)out, embd);
    return pd_launch_status();
}

// Greedy argmax, pass 1: parallel partial reduction. grid = n_parts blocks; block b
// scans its contiguous slice of `logits` and writes its (max, first-index) to
// pmax[b]/pidx[b]. Splitting the 248k-wide scan across many blocks makes it
// memory-bound (~1.4us) instead of a single-block, single-SM serial crawl.
__global__ void pd_argmax_partial_kernel(
    const float* __restrict__ logits, uint32_t vocab,
    float* __restrict__ pmax, uint32_t* __restrict__ pidx) {
    uint32_t tid = threadIdx.x, nth = blockDim.x, blk = blockIdx.x, nblk = gridDim.x;
    __shared__ float sval[256];
    __shared__ uint32_t sidx[256];
    float best = -3.402823e38f;
    uint32_t bi = 0;
    // block-strided: block blk owns elements blk, blk+nblk, ... (coalesced within warp)
    for (uint64_t i = (uint64_t)blk * nth + tid; i < vocab; i += (uint64_t)nblk * nth) {
        float v = logits[i];
        if (v > best) { best = v; bi = (uint32_t)i; }
    }
    sval[tid] = best;
    sidx[tid] = bi;
    __syncthreads();
    for (uint32_t s = nth >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            float o = sval[tid + s];
            uint32_t oi = sidx[tid + s];
            if (o > sval[tid] || (o == sval[tid] && oi < sidx[tid])) {
                sval[tid] = o;
                sidx[tid] = oi;
            }
        }
        __syncthreads();
    }
    if (tid == 0) { pmax[blk] = sval[0]; pidx[blk] = sidx[0]; }
}

// Greedy argmax, pass 2 + advance: one block reduces the n_parts partials to the
// winning token (ties -> lowest index, matching a host first-max scan), then advances
// all per-token decode state on-device so a captured graph replays with no host
// round-trip: write the id into `token`, append to out_ids[step], bump step, set
// pos/mrope to pos+1.
__global__ void pd_argmax_advance_kernel(
    const float* __restrict__ pmax, const uint32_t* __restrict__ pidx, uint32_t n_parts,
    uint32_t* __restrict__ token, uint32_t* __restrict__ pos,
    uint32_t* __restrict__ mrope, uint32_t* __restrict__ out_ids,
    uint32_t* __restrict__ step) {
    uint32_t tid = threadIdx.x, nth = blockDim.x;
    __shared__ float sval[1024];
    __shared__ uint32_t sidx[1024];
    float best = -3.402823e38f;
    uint32_t bi = 0;
    for (uint32_t i = tid; i < n_parts; i += nth) {
        float v = pmax[i];
        uint32_t vi = pidx[i];
        if (v > best || (v == best && vi < bi)) { best = v; bi = vi; }
    }
    sval[tid] = best;
    sidx[tid] = bi;
    __syncthreads();
    for (uint32_t s = nth >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            float o = sval[tid + s];
            uint32_t oi = sidx[tid + s];
            if (o > sval[tid] || (o == sval[tid] && oi < sidx[tid])) {
                sval[tid] = o;
                sidx[tid] = oi;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        uint32_t id = sidx[0];
        token[0] = id;
        uint32_t k = step[0];
        out_ids[k] = id;
        step[0] = k + 1;
        pos[0] = pos[0] + 1;
        // mrope advances INDEPENDENTLY of the kv row position: after an image
        // chunk the llama-position (all four axes equal for text) is offset from
        // the row index by max(grid_x, grid_y) - n_image_rows.
        mrope[0] += 1;
        mrope[1] += 1;
        mrope[2] += 1;
        mrope[3] += 1;
    }
}

// Full greedy epilogue: parallel argmax (pass 1 over `n_parts` blocks into the
// pmax/pidx scratch) + advance (pass 2). Both launches are captured into the decode
// graph. `pmax`/`pidx` must hold n_parts entries.
PD_EXPORT
int pd_argmax_advance(const void* logits, uint32_t vocab, void* pmax, void* pidx,
                      uint32_t n_parts, void* token, void* pos, void* mrope,
                      void* out_ids, void* step, void* stream) {
    if (vocab == 0) return 0;
    pd_argmax_partial_kernel<<<n_parts, 256, 0, (cudaStream_t)stream>>>(
        (const float*)logits, vocab, (float*)pmax, (uint32_t*)pidx);
    uint32_t fth = n_parts > 1024 ? 1024 : n_parts;   // pass-2 block covers all parts
    pd_argmax_advance_kernel<<<1, fth, 0, (cudaStream_t)stream>>>(
        (const float*)pmax, (const uint32_t*)pidx, n_parts, (uint32_t*)token,
        (uint32_t*)pos, (uint32_t*)mrope, (uint32_t*)out_ids, (uint32_t*)step);
    return pd_launch_status();
}

// Advance the staged per-row position inputs by one on device: pos[r] += 1 and
// mrope[4*r] += 1 (all four axes equal for text). Captured between the unrolled
// draft steps so the whole K-step MTP draft loop replays as one CUDA graph.
__global__ void pd_bump_rows_u32_kernel(unsigned int* __restrict__ pos,
                                        unsigned int* __restrict__ mrope,
                                        uint32_t r) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < r) pos[i] += 1u;
    if (i < 4u * r) mrope[i] += 1u;
}

PD_EXPORT
int pd_bump_rows_u32(void* pos, void* mrope, uint32_t r, void* stream) {
    if (r == 0) return 0;
    uint32_t threads = 256;
    uint32_t blocks = (4u * r + threads - 1) / threads;
    pd_bump_rows_u32_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        (unsigned int*)pos, (unsigned int*)mrope, r);
    return pd_launch_status();
}

