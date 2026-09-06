// gemm/int8_mma.cuh (formerly 07_int8_mma_gemm.cuh) - int8 MMA GEMM family + DeltaNet snap/split-GQA
// Textually-included segment of the single pack translation unit.
// Not standalone-compilable: include order is defined by ../pack.cu.
// ---------------------------------------------------------------- int8 MMA GEMM
// Tensor-core Q8_0 GEMM: same numeric class as the dp4a MT kernel (per-32-block
// int8 dot, exact int32, then f32 per-block scale accumulate) but the dot runs
// on the s8 tensor cores (mma.sync m16n8k32) instead of the dp4a INT pipe. The
// A/B/D fragment thread->element maps are llama's tested mma.cuh layout for the
// .s8 m16n8k32 tile (verified there, cross-checked here by a rel-err parity test
// vs the dp4a kernel). Not bit-identical to dp4a: the f32 partials sum in a
// different grouping (per (row,col) across blocks vs dp4a's lane-partial then
// warp reduce) -> same ~1e-6 class the token gates already arbitrate.
//
// Shared-tiled MMA GEMM, templated on the output tile (BM out-rows x BN tokens)
// and warp count so one body serves the small-batch serving tile (64x64, 8
// warps -> weight read once for B<=64) and the wide prefill tile (128x128, 8
// warps -> weight re-read only ceil(B/128)x, vs the naive per-8-tile reread).
// K is staged PD_MMA_KT-wide per barrier (4 Q8_0 blocks) so the sync amortizes
// over 4 MMAs. Shared rows are PADDED by 4 bytes (PD_MMA_KPAD) so the 4-byte
// fragment reads don't 8-way bank-conflict on the 128-wide rows. Scale reapplies
// per 32-block (int32 dots can't accumulate across blocks with different scales).
// Canonical PTX m16n8k32.s8 fragment order: A a0=(m,k=4t) a1=(m+8,4t)
// a2=(m,16+4t) a3=(m+8,16+4t); B b0=(n,4t) b1=(n,16+4t); D d0=(m,2t) d1=(m,2t+1)
// d2=(m+8,2t) d3=(m+8,2t+1); g=lane/4, t=lane%4.
#if defined(__CUDACC__)
#define PD_MMA_OK (__CUDA_ARCH__ >= 800)
#else
#define PD_MMA_OK 0
#endif

#define PD_MMA_KT 128                 // K staged per barrier (bytes = 4 blocks)
#define PD_MMA_NSUBK 4                // = PD_MMA_KT / 32
// Padded shared K-stride: +16 bytes (=4 int32) makes the row stride 36 int32
// (odd multiple of 4) so the 4-byte fragment reads don't 8-way bank-conflict,
// while staying 16-byte aligned for the int4 staging stores.
#define PD_MMA_KPAD (PD_MMA_KT + 16)

// Predicated cp.async copies for the MMA stager (the mmq_pipe helpers live
// further down the file - nvcc needs these defined before use). src-size 0
// zero-fills the shared destination without touching global memory.
__device__ __forceinline__ void pd_mma_cpa16p(void* smem, const void* gmem, bool ok) {
#if PD_MMA_OK
    const unsigned sm = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(sm), "l"(gmem),
                 "r"(ok ? 16u : 0u));
#endif
}
__device__ __forceinline__ void pd_mma_cpa4p(void* smem, const void* gmem, bool ok) {
#if PD_MMA_OK
    const unsigned sm = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(sm), "l"(gmem),
                 "r"(ok ? 4u : 0u));
#endif
}
// wait until at most N cp.async groups are outstanding (immediate operand -
// the generic sibling of pd_attn_cpa_wait0/wait1 for the ST-deep MMA ring)
template <int N>
__device__ __forceinline__ void pd_mma_cpa_waitN() {
#if PD_MMA_OK
    asm volatile("cp.async.wait_group %0;" ::"n"(N));
#endif
}

// ldmatrix fragment loads (sm_75+; the pack floor is sm_80 so unconditional
// under PD_MMA_OK). One x4 replaces the four scalar LDS.32 of an s8 m16n8k32
// A-fragment, one x2 the two of a B-fragment - the b16 8x8 tile mapping lands
// each lane's registers on exactly the bytes the scalar reads fetched, so the
// conversion is bit-identical. Motivation: profiling the 27B decode band
// (b=32, GB202) put 43% of warp stalls on the MIO queue with DRAM at 81%
// - the scalar fragment reads were throttling the cp.async producers.
__device__ __forceinline__ void pd_mma_ldm_x4(const void* p, int& r0, int& r1,
                                              int& r2, int& r3) {
#if PD_MMA_OK
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(a));
#endif
}
__device__ __forceinline__ void pd_mma_ldm_x2(const void* p, int& r0, int& r1) {
#if PD_MMA_OK
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                 : "=r"(r0), "=r"(r1)
                 : "r"(a));
#endif
}

// ST = pipeline stages. ST=2 double-buffers all four staged planes via
// cp.async so the next K-stage streams in while the current one MMAs - the
// serving-rung fix for the profile (long-scoreboard 14.9/issue, issue
// slots 15% active: every stage stalled the whole block on DRAM latency).
// ST=1 keeps the synchronous single-buffer walk (prefill's 128x128 tile -
// doubled it would blow the 48KB static-smem limit; it's bandwidth-saturated
// anyway). Data movement only: the mma/acc order is untouched, so ST=2 is
// BIT-IDENTICAL to ST=1. Scales stage as 4B cp.async pairs (their global
// stride is n_blocks - 8/16B copies would be misaligned for odd rows), which
// needs an even n_blocks (in_dim % 64 == 0); the launcher enforces it.
template <uint32_t BM, uint32_t BN, uint32_t NWARP, uint32_t ST = 1u,
          uint32_t KT = PD_MMA_KT>
__global__ void __launch_bounds__(NWARP * 32) pd_q8_0_gemm_mma_kernel(
        const int8_t* __restrict__ data, const __half* __restrict__ scale,
        const int8_t* __restrict__ xq, const float* __restrict__ xs,
        const float* __restrict__ bias, float* __restrict__ y, uint32_t in_dim,
        uint32_t out_dim, uint32_t batch) {
#if PD_MMA_OK
    constexpr uint32_t NTH = NWARP * 32u;
    constexpr uint32_t WR = BM / 16u;         // warp rows (each warp owns 16 rows)
    constexpr uint32_t WC = NWARP / WR;       // warp cols
    constexpr uint32_t CPW = BN / WC;         // cols per warp
    constexpr uint32_t NSUB = CPW / 8u;       // 16x8 col sub-tiles per warp
    // KT = K bytes staged per barrier. 128 is the historical (GB202-tuned)
    // depth; sm_100 (B200) wants wider stages for bytes-in-flight.
    // K-ascending fold order is KT-invariant, so same-nz results
    // stay bit-identical across KT (nz>1 z-slice boundaries do shift).
    constexpr uint32_t NSUBK = KT / 32u;      // k32 blocks per stage
    constexpr uint32_t KPAD = KT + 16u;       // padded shared K-stride
    constexpr uint32_t I4PR = KT / 16u;       // int4 loads per staged row
    static_assert(WR * WC == NWARP, "warp grid");
    static_assert(NSUB * 8u * WC == BN, "col cover");
    // ST > 2 is the sm_100 (B200) depth: 54 GB/s/SM of HBM3e wants ~4x the
    // bytes in flight that the GB202-era double-buffer keeps. Ring math
    // below is ST-generic; numerics are ST-invariant.
    static_assert(ST >= 1u && ST <= 4u, "stage count");

    // scales live as raw __half / packed f32 quads per row (async-copyable);
    // conversion moved to the read - same values, bit-identical results
    __shared__ __align__(16) int8_t sh_a[ST][BM * KPAD];
    __shared__ __align__(16) int8_t sh_b[ST][BN * KPAD];
    __shared__ __align__(16) __half sh_ws[ST][BM][NSUBK];
    __shared__ __align__(16) float  sh_xs[ST][BN][NSUBK];

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, t = lane & 3u;
    const uint32_t wr = (warp % WR) * 16u;    // warp's row base within tile
    const uint32_t wc = (warp / WR) * CPW;    // warp's col base within tile
    const uint32_t row_base = blockIdx.x * BM;
    const uint32_t col_base = blockIdx.y * BN;
    const uint32_t n_blocks = in_dim >> 5;

    // K-split (grid.z > 1): each z-block walks its own K range and stores an
    // UNBIASED partial plane; pd_q8_0_gemm_mma_ks_combine sums the planes in
    // fixed z order (deterministic). grid.z == 1 keeps the historical walk
    // bit-for-bit. This is the many-SM fix for the 33..64 serving rung: at
    // B<=64 the grid is out/64 blocks (wq 64, wk/wv 8) and idles a 188-SM die.
    const uint32_t nz = gridDim.z;
    uint32_t kt_lo = 0, kt_hi = n_blocks;
    if (nz > 1u) {
        const uint32_t per = ((n_blocks + nz - 1u) / nz + NSUBK - 1u) /
                             NSUBK * NSUBK;
        kt_lo = blockIdx.z * per;
        kt_hi = kt_lo + per < n_blocks ? kt_lo + per : n_blocks;
        y += (size_t)blockIdx.z * out_dim * batch;
    }

    // stage kt's four planes into buffer `buf`: async when ST>=2 (commit at
    // the call site), synchronous stores when ST=1 (original walk)
    auto stage = [&](uint32_t kt, uint32_t buf) {
        #pragma unroll
        for (uint32_t i = tid; i < BM * I4PR; i += NTH) {
            uint32_t row = i / I4PR, k16 = (i % I4PR) * 16u, gk = kt * 32u + k16;
            const bool ok = gk < in_dim && (row_base + row) < out_dim;
            const int8_t* src = data + (size_t)(row_base + row) * in_dim + gk;
            if (ST >= 2u) {
                pd_mma_cpa16p(&sh_a[buf][row * KPAD + k16], src, ok);
            } else {
                *reinterpret_cast<int4*>(&sh_a[buf][row * KPAD + k16]) =
                    ok ? *reinterpret_cast<const int4*>(src) : make_int4(0, 0, 0, 0);
            }
        }
        #pragma unroll
        for (uint32_t i = tid; i < BN * I4PR; i += NTH) {
            uint32_t col = i / I4PR, k16 = (i % I4PR) * 16u, gk = kt * 32u + k16;
            const bool ok = gk < in_dim && (col_base + col) < batch;
            const int8_t* src = xq + (size_t)(col_base + col) * in_dim + gk;
            if (ST >= 2u) {
                pd_mma_cpa16p(&sh_b[buf][col * KPAD + k16], src, ok);
            } else {
                *reinterpret_cast<int4*>(&sh_b[buf][col * KPAD + k16]) =
                    ok ? *reinterpret_cast<const int4*>(src) : make_int4(0, 0, 0, 0);
            }
        }
        if (ST >= 2u) {
            // 4B copies: 2 halves (weight scales) / 1 f32 (activation scales)
            // per copy. Zero-filled tails/OOB rows are only ever multiplied
            // by the zero dot products of their zero-filled data tiles.
            for (uint32_t i = tid; i < BM * (NSUBK / 2u); i += NTH) {
                uint32_t row = i / (NSUBK / 2u), j = (i % (NSUBK / 2u)) * 2u;
                const bool ok = (row_base + row) < out_dim && (kt + j) < n_blocks;
                pd_mma_cpa4p(&sh_ws[buf][row][j],
                             scale + (size_t)(row_base + row) * n_blocks + kt + j, ok);
            }
            for (uint32_t i = tid; i < BN * NSUBK; i += NTH) {
                uint32_t col = i / NSUBK, sb = i % NSUBK;
                const bool ok = (col_base + col) < batch && (kt + sb) < n_blocks;
                pd_mma_cpa4p(&sh_xs[buf][col][sb],
                             xs + (size_t)(col_base + col) * n_blocks + kt + sb, ok);
            }
        } else {
            for (uint32_t i = tid; i < NSUBK * BM; i += NTH) {
                uint32_t sb = i / BM, row = i % BM;
                sh_ws[buf][row][sb] = ((kt + sb) < n_blocks && (row_base + row) < out_dim)
                    ? scale[(size_t)(row_base + row) * n_blocks + kt + sb] : __half(0.f);
            }
            for (uint32_t i = tid; i < NSUBK * BN; i += NTH) {
                uint32_t sb = i / BN, col = i % BN;
                sh_xs[buf][col][sb] = ((kt + sb) < n_blocks && (col_base + col) < batch)
                    ? xs[(size_t)(col_base + col) * n_blocks + kt + sb] : 0.f;
            }
        }
    };

    float acc[NSUB][4] = {};
    // ldmatrix per-lane source rows: A x4 tiles are (rows wr..wr+7 @ko,
    // wr+8..15 @ko, wr..7 @ko+16, wr+8..15 @ko+16) across lane octets; B x2
    // uses the first two octets (rows csub..csub+7 @ko, @ko+16) - lanes 16+
    // pass a benign in-range address (their operand is unused by .x2).
    const uint32_t ldm_l7 = lane & 7u;
    const uint32_t ldm_arow = wr + ((lane & 8u) ? 8u : 0u) + ldm_l7;
    const uint32_t ldm_akof = (lane & 16u) ? 16u : 0u;
    const uint32_t ldm_bkof = (lane & 8u) ? 16u : 0u;
    // inner: the NSUBK staged blocks, no barrier between (all in shared)
    auto compute = [&](uint32_t buf) {
        #pragma unroll
        for (uint32_t sb = 0; sb < NSUBK; ++sb) {
            const uint32_t ko = sb * 32u;
            int a0, a1, a2, a3;
            pd_mma_ldm_x4(&sh_a[buf][ldm_arow * KPAD + ko + ldm_akof], a0, a1, a2, a3);
            float ws0 = __half2float(sh_ws[buf][wr + g][sb]);
            float ws8 = __half2float(sh_ws[buf][wr + 8u + g][sb]);
            #pragma unroll
            for (uint32_t sub = 0; sub < NSUB; ++sub) {
                const uint32_t csub = wc + sub * 8u;
                int b0, b1;
                pd_mma_ldm_x2(&sh_b[buf][(csub + ldm_l7) * KPAD + ko + ldm_bkof], b0, b1);
                int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
                float xc0 = sh_xs[buf][csub + 2u * t][sb], xc1 = sh_xs[buf][csub + 2u * t + 1u][sb];
                acc[sub][0] += ws0 * xc0 * (float)d0;
                acc[sub][1] += ws0 * xc1 * (float)d1;
                acc[sub][2] += ws8 * xc0 * (float)d2;
                acc[sub][3] += ws8 * xc1 * (float)d3;
            }
        }
    };

    if (ST >= 2u && kt_lo < kt_hi) {
        // ST-deep ring: buffer p computes while up to ST-1 stages stream in.
        // One commit group per iteration always (empty groups are legal PTX
        // and complete immediately) so the wait immediate stays uniform:
        // after the per-iter commit at most ST groups are outstanding and
        // wait_group(ST-1) retires exactly the buffer about to be computed.
        // The end-of-compute barrier doubles as the write-hazard fence for
        // the next issue into the just-read buffer (same as the old ST=2
        // special case, which this generalizes byte-for-byte at ST=2).
        #pragma unroll
        for (int s = 0; s < (int)ST - 1; ++s) {
            const uint32_t kt = kt_lo + (uint32_t)s * NSUBK;
            if (kt < kt_hi) stage(kt, s);
            pd_attn_cpa_commit(); // possibly empty - keeps group count fixed
        }
        uint32_t p = 0;
        for (uint32_t kt = kt_lo; kt < kt_hi; kt += NSUBK) {
            const uint32_t pre = kt + (ST - 1u) * NSUBK;
            // prefetch lands in the buffer computed last iteration; the
            // trailing barrier below has already fenced every reader
            if (pre < kt_hi) stage(pre, (p + ST - 1u) % ST);
            pd_attn_cpa_commit();
            pd_mma_cpa_waitN<(int)ST - 1>(); // buffer p landed; rest in flight
            __syncthreads();
            compute(p);
            __syncthreads();
            p = (p + 1u) % ST;
        }
    } else {
        for (uint32_t kt = kt_lo; kt < kt_hi; kt += NSUBK) {
            stage(kt, 0);
            __syncthreads();
            compute(0);
            __syncthreads();
        }
    }

    // store: element (row, tok) -> y[tok*out_dim + row]. bias fold only when
    // this kernel is the final writer (the ks launcher passes nullptr for the
    // nz>1 partial planes - the combine adds bias there); the single f32 bias
    // add on the completed sum matches store-then-bias_add rounding exactly.
    const uint32_t r0 = row_base + wr + g, r8 = row_base + wr + 8u + g;
    const float b0f = (bias && r0 < out_dim) ? bias[r0] : 0.0f;
    const float b8f = (bias && r8 < out_dim) ? bias[r8] : 0.0f;
    #pragma unroll
    for (uint32_t sub = 0; sub < NSUB; ++sub) {
        const uint32_t c0 = col_base + wc + sub * 8u + 2u * t;
        const uint32_t c1 = c0 + 1u;
        if (r0 < out_dim) {
            if (c0 < batch) y[(size_t)c0 * out_dim + r0] = bias ? acc[sub][0] + b0f : acc[sub][0];
            if (c1 < batch) y[(size_t)c1 * out_dim + r0] = bias ? acc[sub][1] + b0f : acc[sub][1];
        }
        if (r8 < out_dim) {
            if (c0 < batch) y[(size_t)c0 * out_dim + r8] = bias ? acc[sub][2] + b8f : acc[sub][2];
            if (c1 < batch) y[(size_t)c1 * out_dim + r8] = bias ? acc[sub][3] + b8f : acc[sub][3];
        }
    }
#else
    (void)data; (void)scale; (void)xq; (void)xs; (void)y;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}

PD_EXPORT
int pd_q8_0_gemm_mma(const void* data, const void* scale, const void* xq,
                     const void* xs, void* y, uint32_t in_dim, uint32_t out_dim,
                     uint32_t batch, void* stream) {
    if (out_dim == 0 || batch == 0) return 0;
    // in_dim must be Q8_0-block-aligned (format invariant: every block is
    // exactly 32 elements). out_dim has no such requirement - the tile
    // staging zero-pads rows past out_dim and the writeback bounds-checks
    // every row individually (row_base+r < out_dim), so a ragged last tile
    // is already handled; the historical `out_dim & 15u` reject was
    // over-conservative (see the verify in tests/gemm_ragged_out_dim.rs -
    // laguna S-2.1's g_proj is out_dim=72, not 16-aligned).
    if (in_dim & 31u) return cudaErrorInvalidValue;
    // small batch -> 64x64 (weight read once, no wasted cols); large batch ->
    // 128x128 (fewer weight re-reads, tensor cores saturated -> prefill)
    const auto* d = (const int8_t*)data;
    const auto* sc = (const __half*)scale;
    const auto* q = (const int8_t*)xq;
    const auto* s = (const float*)xs;
    auto* o = (float*)y;
    auto st = (cudaStream_t)stream;
    if (batch <= 64) {
        dim3 grid((out_dim + 63u) / 64u, (batch + 63u) / 64u);
        // No ST=2 here: the BN64 tile's doubled smem (40KB) drops it to
        // 1 block/SM, which measured ~3x slower at b=8 and 1.54x floor at
        // b=32 across the 3142-block lm_head grid. The pipelined serving
        // shapes live in the ks rungs (BN16/BN32 fit 2 blocks/SM).
        pd_q8_0_gemm_mma_kernel<64u, 64u, 8u><<<grid, 256, 0, st>>>(d, sc, q, s, nullptr, o, in_dim, out_dim, batch);
    } else {
        dim3 grid((out_dim + 127u) / 128u, (batch + 127u) / 128u);
        pd_q8_0_gemm_mma_kernel<128u, 128u, 8u><<<grid, 256, 0, st>>>(d, sc, q, s, nullptr, o, in_dim, out_dim, batch);
    }
    return pd_launch_status();
}

// ---- M-col mma (wide-spec verify GEMM class, 65..192 tokens) --------------
// The ks rungs read weights once only up to their col-tile (BN<=64 tokens);
// wider batches fell to the mmq M-tile ladder = one full weight pass per
// 64-row tile (3x at 160 verify rows - the c32 wide-spec k=4 blocker).
// This variant keeps the BN64 kernel's A-side exactly (weights + scales
// cp.async double-buffered per K-stage) and loops NCOL 64-token col-tiles
// per stage with B AND activation scales read DIRECT from L2 (activations
// are ~100KB/GEMM and re-read NCOL x <=42-stage times - noise next to the
// weight stream). No B smem at all: ~20KB total -> the col loop is pure
// register cost (16 f32 acc per col-tile per thread; NCOL<=3 = 48).
// Numerics: each col-tile's mma/acc sequence is identical to the BN64
// kernel run on that 64-token slice (same kt walk, same z-slicing) ->
// bit-equal per slice; the K-split combine is unchanged.
template <uint32_t BM, uint32_t NCOL, uint32_t NWARP>
__global__ void __launch_bounds__(NWARP * 32) pd_q8_0_gemm_mma_mcol_kernel(
        const int8_t* __restrict__ data, const __half* __restrict__ scale,
        const int8_t* __restrict__ xq, const float* __restrict__ xs,
        const float* __restrict__ bias, float* __restrict__ y, uint32_t in_dim,
        uint32_t out_dim, uint32_t batch) {
#if PD_MMA_OK
    constexpr uint32_t BN = 64u;
    constexpr uint32_t NTH = NWARP * 32u;
    constexpr uint32_t WR = BM / 16u;
    constexpr uint32_t WC = NWARP / WR;
    constexpr uint32_t CPW = BN / WC;
    constexpr uint32_t NSUB = CPW / 8u;
    constexpr uint32_t I4PR = PD_MMA_KT / 16u;
    static_assert(WR * WC == NWARP, "warp grid");

    // dynamic smem: A + B (all NCOL col-tiles) + both scale planes, all
    // cp.async double-buffered - an earlier cut read B direct from L2 and
    // stalled the mma issue chain (213 vs 91 ms; the loads sat in the mma
    // dependency path). Layout: [2][A | ws | B | xs], 16B-aligned segments.
    extern __shared__ __align__(16) unsigned char mcol_sh[];
    constexpr uint32_t A_BYTES = BM * PD_MMA_KPAD;
    constexpr uint32_t WS_BYTES = BM * PD_MMA_NSUBK * 2u;
    constexpr uint32_t B_BYTES = NCOL * 64u * PD_MMA_KPAD;
    constexpr uint32_t XS_BYTES = NCOL * 64u * PD_MMA_NSUBK * 4u;
    constexpr uint32_t BUF_BYTES = (A_BYTES + WS_BYTES + B_BYTES + XS_BYTES + 15u) & ~15u;
    auto sh_a = [&](uint32_t buf) { return (int8_t*)(mcol_sh + (size_t)buf * BUF_BYTES); };
    auto sh_ws = [&](uint32_t buf) {
        return (__half*)(mcol_sh + (size_t)buf * BUF_BYTES + A_BYTES);
    };
    auto sh_b = [&](uint32_t buf) {
        return (int8_t*)(mcol_sh + (size_t)buf * BUF_BYTES + A_BYTES + WS_BYTES);
    };
    auto sh_xs = [&](uint32_t buf) {
        return (float*)(mcol_sh + (size_t)buf * BUF_BYTES + A_BYTES + WS_BYTES + B_BYTES);
    };

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, t = lane & 3u;
    const uint32_t wr = (warp % WR) * 16u;
    const uint32_t wc = (warp / WR) * CPW;
    const uint32_t row_base = blockIdx.x * BM;
    const uint32_t n_blocks = in_dim >> 5;

    const uint32_t nz = gridDim.z;
    uint32_t kt_lo = 0, kt_hi = n_blocks;
    if (nz > 1u) {
        const uint32_t per = ((n_blocks + nz - 1u) / nz + PD_MMA_NSUBK - 1u) /
                             PD_MMA_NSUBK * PD_MMA_NSUBK;
        kt_lo = blockIdx.z * per;
        kt_hi = kt_lo + per < n_blocks ? kt_lo + per : n_blocks;
        y += (size_t)blockIdx.z * out_dim * batch;
    }

    auto stage_a = [&](uint32_t kt, uint32_t buf) {
        int8_t* a = sh_a(buf);
        __half* ws = sh_ws(buf);
        int8_t* b = sh_b(buf);
        float* xsv = sh_xs(buf);
        #pragma unroll
        for (uint32_t i = tid; i < BM * I4PR; i += NTH) {
            uint32_t row = i / I4PR, k16 = (i % I4PR) * 16u, gk = kt * 32u + k16;
            const bool ok = gk < in_dim && (row_base + row) < out_dim;
            pd_mma_cpa16p(&a[row * PD_MMA_KPAD + k16],
                          data + (size_t)(row_base + row) * in_dim + gk, ok);
        }
        for (uint32_t i = tid; i < BM * 2u; i += NTH) {
            uint32_t row = i >> 1, j = (i & 1u) * 2u;
            const bool ok = (row_base + row) < out_dim && (kt + j) < n_blocks;
            pd_mma_cpa4p(&ws[row * PD_MMA_NSUBK + j],
                         scale + (size_t)(row_base + row) * n_blocks + kt + j, ok);
        }
        // B: NCOL 64-token col-tiles, same masked 16B copies as the BN64
        // kernel (zero-fill past batch/K)
        #pragma unroll
        for (uint32_t i = tid; i < NCOL * 64u * I4PR; i += NTH) {
            uint32_t col = i / I4PR, k16 = (i % I4PR) * 16u, gk = kt * 32u + k16;
            const bool ok = gk < in_dim && col < batch;
            pd_mma_cpa16p(&b[col * PD_MMA_KPAD + k16],
                          xq + (size_t)col * in_dim + gk, ok);
        }
        for (uint32_t i = tid; i < NCOL * 64u * PD_MMA_NSUBK; i += NTH) {
            uint32_t col = i >> 2, sb = i & 3u;
            const bool ok = col < batch && (kt + sb) < n_blocks;
            pd_mma_cpa4p(&xsv[col * PD_MMA_NSUBK + sb],
                         xs + (size_t)col * n_blocks + kt + sb, ok);
        }
    };

    float acc[NCOL][NSUB][4] = {};

    auto compute = [&](uint32_t buf) {
        const int8_t* a = sh_a(buf);
        const __half* ws = sh_ws(buf);
        const int8_t* b = sh_b(buf);
        const float* xsv = sh_xs(buf);
        #pragma unroll
        for (uint32_t sb = 0; sb < PD_MMA_NSUBK; ++sb) {
            const uint32_t ko = sb * 32u;
            int a0 = *reinterpret_cast<const int*>(&a[(wr + g) * PD_MMA_KPAD + ko + t * 4u]);
            int a1 = *reinterpret_cast<const int*>(&a[(wr + 8u + g) * PD_MMA_KPAD + ko + t * 4u]);
            int a2 = *reinterpret_cast<const int*>(&a[(wr + g) * PD_MMA_KPAD + ko + 16u + t * 4u]);
            int a3 = *reinterpret_cast<const int*>(&a[(wr + 8u + g) * PD_MMA_KPAD + ko + 16u + t * 4u]);
            float ws0 = __half2float(ws[(wr + g) * PD_MMA_NSUBK + sb]);
            float ws8 = __half2float(ws[(wr + 8u + g) * PD_MMA_NSUBK + sb]);
            #pragma unroll
            for (uint32_t ct = 0; ct < NCOL; ++ct) {
                const uint32_t col_base = ct * 64u;
                #pragma unroll
                for (uint32_t sub = 0; sub < NSUB; ++sub) {
                    const uint32_t cn = col_base + wc + sub * 8u + g;
                    int b0 = *reinterpret_cast<const int*>(&b[cn * PD_MMA_KPAD + ko + t * 4u]);
                    int b1 = *reinterpret_cast<const int*>(&b[cn * PD_MMA_KPAD + ko + 16u + t * 4u]);
                    int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
                    const uint32_t c0 = col_base + wc + sub * 8u + 2u * t;
                    const float xc0 = xsv[c0 * PD_MMA_NSUBK + sb];
                    const float xc1 = xsv[(c0 + 1u) * PD_MMA_NSUBK + sb];
                    acc[ct][sub][0] += ws0 * xc0 * (float)d0;
                    acc[ct][sub][1] += ws0 * xc1 * (float)d1;
                    acc[ct][sub][2] += ws8 * xc0 * (float)d2;
                    acc[ct][sub][3] += ws8 * xc1 * (float)d3;
                }
            }
        }
    };

    if (kt_lo < kt_hi) {
        stage_a(kt_lo, 0);
        pd_attn_cpa_commit();
        uint32_t p = 0;
        for (uint32_t kt = kt_lo; kt < kt_hi; kt += PD_MMA_NSUBK) {
            const uint32_t nxt = kt + PD_MMA_NSUBK;
            if (nxt < kt_hi) {
                stage_a(nxt, p ^ 1u);
                pd_attn_cpa_commit();
                pd_attn_cpa_wait1();
            } else {
                pd_attn_cpa_wait0();
            }
            __syncthreads();
            compute(p);
            __syncthreads();
            p ^= 1u;
        }
    }

    const uint32_t r0 = row_base + wr + g, r8 = row_base + wr + 8u + g;
    const float b0f = (bias && r0 < out_dim) ? bias[r0] : 0.0f;
    const float b8f = (bias && r8 < out_dim) ? bias[r8] : 0.0f;
    #pragma unroll
    for (uint32_t ct = 0; ct < NCOL; ++ct) {
        #pragma unroll
        for (uint32_t sub = 0; sub < NSUB; ++sub) {
            const uint32_t c0 = ct * BN + wc + sub * 8u + 2u * t;
            const uint32_t c1 = c0 + 1u;
            if (r0 < out_dim) {
                if (c0 < batch) y[(size_t)c0 * out_dim + r0] = bias ? acc[ct][sub][0] + b0f : acc[ct][sub][0];
                if (c1 < batch) y[(size_t)c1 * out_dim + r0] = bias ? acc[ct][sub][1] + b0f : acc[ct][sub][1];
            }
            if (r8 < out_dim) {
                if (c0 < batch) y[(size_t)c0 * out_dim + r8] = bias ? acc[ct][sub][2] + b8f : acc[ct][sub][2];
                if (c1 < batch) y[(size_t)c1 * out_dim + r8] = bias ? acc[ct][sub][3] + b8f : acc[ct][sub][3];
            }
        }
    }
#else
    (void)data; (void)scale; (void)xq; (void)xs; (void)bias; (void)y;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}


// ---- warp-specialized persistent K-split GEMM (the sm_100 decode rung) ----
// The ST-ring kernel tops out ~3.2 TB/s on B200 no matter ST/KT/nz: every
// K-stage is a whole-CTA rendezvous, so a CTA holds at most one stage's
// bytes in flight while computing, and 2-4 co-resident CTAs cannot bridge
// the ~6x per-SM bandwidth step from GB202 (9.5 GB/s/SM) to B200 HBM3e
// (54 GB/s/SM) - a raw cp.async stream does 5.4+ TB/s at the same access
// pattern. Cure =
// the moe/decode_block_scale.cuh producer/consumer idiom (original implementation for
// the dense q8 rung): PW producer warps own all staging through an S-deep
// smem ring with split-phase mbarriers, NWARP consumer warps run the ring
// kernel's UNCHANGED fragment/mma/fold sequence, and the grid is persistent
// ((tile, z) items strided across ~2*nsm CTAs) so the 84-336-tile decode
// shapes stop paying wave-quantization tails. Numerics: per (tile, z) item
// the k-ascending fold order is exactly the ring kernel's -> bit-equal per
// (nz, KT) config; the epilogue math is verbatim.
// mbarrier plumbing lives here (not reused from moe/block_scale_quant.cuh - that
// segment includes after this one and is PD_BS_OK-gated).
__device__ __forceinline__ void pd_mma_bar_init(uint64_t* bar, uint32_t count) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(a), "r"(count));
}
__device__ __forceinline__ void pd_mma_bar_arrive(uint64_t* bar) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(a) : "memory");
}
// async arrival: fires when this thread's outstanding cp.asyncs complete.
// .noinc = the arrival counts against the barrier's init count.
__device__ __forceinline__ void pd_mma_cp_arrive_noinc(uint64_t* bar) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(a) : "memory");
}
__device__ __forceinline__ uint32_t pd_mma_bar_try_wait(uint64_t* bar, uint32_t parity) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    uint32_t done;
    asm volatile("{\n\t.reg .pred p;\n\t"
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\t"
                 "selp.b32 %0, 1, 0, p;\n\t}"
                 : "=r"(done) : "r"(a), "r"(parity) : "memory");
    return done;
}
__device__ __forceinline__ void pd_mma_bar_wait(uint64_t* bar, uint32_t parity) {
    // backoff on miss: hundreds of threads spinning raw try_wait on one smem
    // address contend on the very LSU pipe the producers need
    while (!pd_mma_bar_try_wait(bar, parity)) { __nanosleep(32); }
}

template <uint32_t BM, uint32_t BN, uint32_t NWARP, uint32_t PW, uint32_t S,
          uint32_t KT = PD_MMA_KT>
__global__ void __launch_bounds__((NWARP + PW) * 32)
pd_q8_0_gemm_mma_ws_kernel(const int8_t* __restrict__ data,
                           const __half* __restrict__ scale,
                           const int8_t* __restrict__ xq,
                           const float* __restrict__ xs,
                           const float* __restrict__ bias, float* __restrict__ y,
                           uint32_t in_dim, uint32_t out_dim, uint32_t batch,
                           uint32_t nz) {
#if PD_MMA_OK && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    constexpr uint32_t NSUBK = KT / 32u, KPAD = KT + 16u, I4PR = KT / 16u;
    constexpr uint32_t WR = BM / 16u, WC = NWARP / WR, CPW = BN / WC;
    constexpr uint32_t NSUB = CPW / 8u;
    static_assert(WR * WC == NWARP && NSUB * 8u * WC == BN, "warp grid");
    // ks rungs keep batch <= BN (the 65..192 band is mcol territory), so the
    // whole batch is this kernel's single column tile: col_base == 0.
    const uint32_t n_blocks = in_dim >> 5;
    const uint32_t tiles = (out_dim + BM - 1u) / BM;
    const uint32_t per = ((n_blocks + nz - 1u) / nz + NSUBK - 1u) / NSUBK * NSUBK;
    const uint32_t nit = tiles * nz;

    extern __shared__ unsigned char pd_ws_sh[];
    uint64_t* bfull = (uint64_t*)pd_ws_sh;                       // [S]
    uint64_t* bempty = bfull + S;                                // [S]
    unsigned char* ring = pd_ws_sh + ((2u * S * 8u + 15u) & ~15u);
    constexpr uint32_t ASZ = BM * KPAD, BSZ = BN * KPAD;
    constexpr uint32_t WSSZ = BM * NSUBK * 2u, XSSZ = BN * NSUBK * 4u;
    constexpr uint32_t STAGE = (ASZ + BSZ + WSSZ + XSSZ + 15u) & ~15u;

    const uint32_t tid = threadIdx.x;
    if (tid == 0) {
        #pragma unroll
        for (uint32_t s = 0; s < S; ++s) {
            pd_mma_bar_init(&bfull[s], PW * 32u);
            pd_mma_bar_init(&bempty[s], NWARP * 32u);
        }
    }
    __syncthreads();  // the only full-CTA barrier in the kernel

    const uint32_t warp = tid >> 5;
    if (warp >= NWARP) {
        // ---------------- producers: PW warps own all staging ----------------
        const uint32_t ptid = tid - NWARP * 32u, pth = PW * 32u;
        uint32_t eph[S] = {};  // per-slot empty parity (round-robin reuse)
        uint32_t gkt = 0;
        for (uint32_t it = blockIdx.x; it < nit; it += gridDim.x) {
            const uint32_t tile = it % tiles, z = it / tiles;
            const uint32_t row_base = tile * BM;
            const uint32_t kt_lo = z * per;
            const uint32_t kt_hi = kt_lo + per < n_blocks ? kt_lo + per : n_blocks;
            for (uint32_t kt = kt_lo; kt < kt_hi; kt += NSUBK) {
                const uint32_t s = gkt % S;
                if (gkt >= S) { pd_mma_bar_wait(&bempty[s], eph[s]); eph[s] ^= 1u; }
                ++gkt;
                unsigned char* sa = ring + s * STAGE;
                unsigned char* sb = sa + ASZ;
                __half* sws = (__half*)(sb + BSZ);
                float* sxs = (float*)((unsigned char*)sws + WSSZ);
                for (uint32_t i = ptid; i < BM * I4PR; i += pth) {
                    uint32_t row = i / I4PR, k16 = (i % I4PR) * 16u, gk = kt * 32u + k16;
                    const bool ok = gk < in_dim && (row_base + row) < out_dim;
                    pd_mma_cpa16p(sa + row * KPAD + k16,
                                  data + (size_t)(row_base + row) * in_dim + gk, ok);
                }
                for (uint32_t i = ptid; i < BN * I4PR; i += pth) {
                    uint32_t col = i / I4PR, k16 = (i % I4PR) * 16u, gk = kt * 32u + k16;
                    const bool ok = gk < in_dim && col < batch;
                    pd_mma_cpa16p(sb + col * KPAD + k16,
                                  xq + (size_t)col * in_dim + gk, ok);
                }
                for (uint32_t i = ptid; i < BM * (NSUBK / 2u); i += pth) {
                    uint32_t row = i / (NSUBK / 2u), j = (i % (NSUBK / 2u)) * 2u;
                    const bool ok = (row_base + row) < out_dim && (kt + j) < n_blocks;
                    pd_mma_cpa4p(&sws[row * NSUBK + j],
                                 scale + (size_t)(row_base + row) * n_blocks + kt + j, ok);
                }
                for (uint32_t i = ptid; i < BN * NSUBK; i += pth) {
                    uint32_t col = i / NSUBK, sbk = i % NSUBK;
                    const bool ok = col < batch && (kt + sbk) < n_blocks;
                    pd_mma_cpa4p(&sxs[col * NSUBK + sbk],
                                 xs + (size_t)col * n_blocks + kt + sbk, ok);
                }
                pd_mma_cp_arrive_noinc(&bfull[s]);
            }
        }
    } else {
        // -------- consumers: the ring kernel's fragment/mma/fold, verbatim --------
        const uint32_t lane = tid & 31u, g = lane >> 2, t = lane & 3u;
        const uint32_t wr = (warp % WR) * 16u, wc = (warp / WR) * CPW;
        uint32_t fph[S] = {};  // per-slot full parity
        uint32_t gkt = 0;
        for (uint32_t it = blockIdx.x; it < nit; it += gridDim.x) {
            const uint32_t tile = it % tiles, z = it / tiles;
            const uint32_t row_base = tile * BM;
            const uint32_t kt_lo = z * per;
            const uint32_t kt_hi = kt_lo + per < n_blocks ? kt_lo + per : n_blocks;
            float acc[NSUB][4] = {};
            for (uint32_t kt = kt_lo; kt < kt_hi; kt += NSUBK) {
                const uint32_t s = gkt % S;
                pd_mma_bar_wait(&bfull[s], fph[s]); fph[s] ^= 1u;
                ++gkt;
                const unsigned char* sa = ring + s * STAGE;
                const unsigned char* sb = sa + ASZ;
                const __half* sws = (const __half*)(sb + BSZ);
                const float* sxs = (const float*)((const unsigned char*)sws + WSSZ);
                #pragma unroll
                for (uint32_t sbk = 0; sbk < NSUBK; ++sbk) {
                    const uint32_t ko = sbk * 32u;
                    int a0 = *reinterpret_cast<const int*>(sa + (wr + g) * KPAD + ko + t * 4u);
                    int a1 = *reinterpret_cast<const int*>(sa + (wr + 8u + g) * KPAD + ko + t * 4u);
                    int a2 = *reinterpret_cast<const int*>(sa + (wr + g) * KPAD + ko + 16u + t * 4u);
                    int a3 = *reinterpret_cast<const int*>(sa + (wr + 8u + g) * KPAD + ko + 16u + t * 4u);
                    float ws0 = __half2float(sws[(wr + g) * NSUBK + sbk]);
                    float ws8 = __half2float(sws[(wr + 8u + g) * NSUBK + sbk]);
                    #pragma unroll
                    for (uint32_t sub = 0; sub < NSUB; ++sub) {
                        const uint32_t csub = wc + sub * 8u;
                        int b0 = *reinterpret_cast<const int*>(sb + (csub + g) * KPAD + ko + t * 4u);
                        int b1 = *reinterpret_cast<const int*>(sb + (csub + g) * KPAD + ko + 16u + t * 4u);
                        int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                        asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                            : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
                        float xc0 = sxs[(csub + 2u * t) * NSUBK + sbk];
                        float xc1 = sxs[(csub + 2u * t + 1u) * NSUBK + sbk];
                        acc[sub][0] += ws0 * xc0 * (float)d0;
                        acc[sub][1] += ws0 * xc1 * (float)d1;
                        acc[sub][2] += ws8 * xc0 * (float)d2;
                        acc[sub][3] += ws8 * xc1 * (float)d3;
                    }
                }
                pd_mma_bar_arrive(&bempty[s]);
            }
            // epilogue - ring kernel's store, z-plane offset per item
            float* dst = nz > 1u ? y + (size_t)z * out_dim * batch : y;
            const uint32_t r0 = row_base + wr + g, r8 = row_base + wr + 8u + g;
            const float b0f = (bias && r0 < out_dim) ? bias[r0] : 0.0f;
            const float b8f = (bias && r8 < out_dim) ? bias[r8] : 0.0f;
            #pragma unroll
            for (uint32_t sub = 0; sub < NSUB; ++sub) {
                const uint32_t c0 = wc + sub * 8u + 2u * t;
                const uint32_t c1 = c0 + 1u;
                if (r0 < out_dim) {
                    if (c0 < batch) dst[(size_t)c0 * out_dim + r0] = bias ? acc[sub][0] + b0f : acc[sub][0];
                    if (c1 < batch) dst[(size_t)c1 * out_dim + r0] = bias ? acc[sub][1] + b0f : acc[sub][1];
                }
                if (r8 < out_dim) {
                    if (c0 < batch) dst[(size_t)c0 * out_dim + r8] = bias ? acc[sub][2] + b8f : acc[sub][2];
                    if (c1 < batch) dst[(size_t)c1 * out_dim + r8] = bias ? acc[sub][3] + b8f : acc[sub][3];
                }
            }
        }
    }
#else
    (void)data; (void)scale; (void)xq; (void)xs; (void)bias; (void)y;
    (void)in_dim; (void)out_dim; (void)batch; (void)nz;
#endif
}


// ---- persistent-chained ring (the 3.2 TB/s wall experiment) ---------------
// Evidence: the same ring kernel hits 4.7 TB/s on lm_head (4096 tiles, 42
// serial K-stages per CTA) and 3.2 on the decode shapes (84-336 tiles) at
// any nz - deep waves alone don't help because K-splitting shortens each
// CTA's serial run below pipeline steady state. This variant strides (tile,
// z) items across a persistent grid and carries the 2-stage cp.async ring
// across item boundaries: the next item's first stage prefetches while the
// current item's last stage computes, so per-CTA runs get lm_head-deep at
// any shape. Fold order per item is untouched -> bit-exact vs the plain
// ring at the same (nz, KT). Serving rungs only (col_base == 0).
template <uint32_t BM, uint32_t BN, uint32_t NWARP, uint32_t KT = PD_MMA_KT>
__global__ void __launch_bounds__(NWARP * 32, 4) pd_q8_0_gemm_mma_pc_kernel(
        const int8_t* __restrict__ data, const __half* __restrict__ scale,
        const int8_t* __restrict__ xq, const float* __restrict__ xs,
        const float* __restrict__ bias, float* __restrict__ y, uint32_t in_dim,
        uint32_t out_dim, uint32_t batch, uint32_t nz) {
#if PD_MMA_OK
    constexpr uint32_t NTH = NWARP * 32u;
    constexpr uint32_t WR = BM / 16u;
    constexpr uint32_t WC = NWARP / WR;
    constexpr uint32_t CPW = BN / WC;
    constexpr uint32_t NSUB = CPW / 8u;
    constexpr uint32_t NSUBK = KT / 32u;
    constexpr uint32_t KPAD = KT + 16u;
    constexpr uint32_t I4PR = KT / 16u;
    static_assert(WR * WC == NWARP && NSUB * 8u * WC == BN, "warp grid");

    __shared__ __align__(16) int8_t sh_a[2][BM * KPAD];
    __shared__ __align__(16) int8_t sh_b[2][BN * KPAD];
    __shared__ __align__(16) __half sh_ws[2][BM][NSUBK];
    __shared__ __align__(16) float  sh_xs[2][BN][NSUBK];

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, t = lane & 3u;
    const uint32_t wr = (warp % WR) * 16u;
    const uint32_t wc = (warp / WR) * CPW;
    const uint32_t n_blocks = in_dim >> 5;
    const uint32_t tiles = (out_dim + BM - 1u) / BM;
    const uint32_t per = ((n_blocks + nz - 1u) / nz + NSUBK - 1u) / NSUBK * NSUBK;
    const uint32_t nit = tiles * nz;

    // stage count of item `it` (z tails may be short or empty)
    auto scnt = [&](uint32_t it) -> uint32_t {
        const uint32_t lo = (it / tiles) * per;
        const uint32_t hi = lo + per < n_blocks ? lo + per : n_blocks;
        return lo < hi ? (hi - lo + NSUBK - 1u) / NSUBK : 0u;
    };
    auto stage = [&](uint32_t it, uint32_t st, uint32_t buf) {
        const uint32_t row_base = (it % tiles) * BM;
        const uint32_t kt = (it / tiles) * per + st * NSUBK;
        #pragma unroll
        for (uint32_t i = tid; i < BM * I4PR; i += NTH) {
            uint32_t row = i / I4PR, k16 = (i % I4PR) * 16u, gk = kt * 32u + k16;
            const bool ok = gk < in_dim && (row_base + row) < out_dim;
            pd_mma_cpa16p(&sh_a[buf][row * KPAD + k16],
                          data + (size_t)(row_base + row) * in_dim + gk, ok);
        }
        #pragma unroll
        for (uint32_t i = tid; i < BN * I4PR; i += NTH) {
            uint32_t col = i / I4PR, k16 = (i % I4PR) * 16u, gk = kt * 32u + k16;
            const bool ok = gk < in_dim && col < batch;
            pd_mma_cpa16p(&sh_b[buf][col * KPAD + k16],
                          xq + (size_t)col * in_dim + gk, ok);
        }
        for (uint32_t i = tid; i < BM * (NSUBK / 2u); i += NTH) {
            uint32_t row = i / (NSUBK / 2u), j = (i % (NSUBK / 2u)) * 2u;
            const bool ok = (row_base + row) < out_dim && (kt + j) < n_blocks;
            pd_mma_cpa4p(&sh_ws[buf][row][j],
                         scale + (size_t)(row_base + row) * n_blocks + kt + j, ok);
        }
        for (uint32_t i = tid; i < BN * NSUBK; i += NTH) {
            uint32_t col = i / NSUBK, sb = i % NSUBK;
            const bool ok = col < batch && (kt + sb) < n_blocks;
            pd_mma_cpa4p(&sh_xs[buf][col][sb],
                         xs + (size_t)col * n_blocks + kt + sb, ok);
        }
    };

    float acc[NSUB][4] = {};
    auto compute = [&](uint32_t buf) {
        #pragma unroll
        for (uint32_t sb = 0; sb < NSUBK; ++sb) {
            const uint32_t ko = sb * 32u;
            int a0 = *reinterpret_cast<const int*>(&sh_a[buf][(wr + g) * KPAD + ko + t * 4u]);
            int a1 = *reinterpret_cast<const int*>(&sh_a[buf][(wr + 8u + g) * KPAD + ko + t * 4u]);
            int a2 = *reinterpret_cast<const int*>(&sh_a[buf][(wr + g) * KPAD + ko + 16u + t * 4u]);
            int a3 = *reinterpret_cast<const int*>(&sh_a[buf][(wr + 8u + g) * KPAD + ko + 16u + t * 4u]);
            float ws0 = __half2float(sh_ws[buf][wr + g][sb]);
            float ws8 = __half2float(sh_ws[buf][wr + 8u + g][sb]);
            #pragma unroll
            for (uint32_t sub = 0; sub < NSUB; ++sub) {
                const uint32_t csub = wc + sub * 8u;
                int b0 = *reinterpret_cast<const int*>(&sh_b[buf][(csub + g) * KPAD + ko + t * 4u]);
                int b1 = *reinterpret_cast<const int*>(&sh_b[buf][(csub + g) * KPAD + ko + 16u + t * 4u]);
                int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
                float xc0 = sh_xs[buf][csub + 2u * t][sb], xc1 = sh_xs[buf][csub + 2u * t + 1u][sb];
                acc[sub][0] += ws0 * xc0 * (float)d0;
                acc[sub][1] += ws0 * xc1 * (float)d1;
                acc[sub][2] += ws8 * xc0 * (float)d2;
                acc[sub][3] += ws8 * xc1 * (float)d3;
            }
        }
    };
    auto store_item = [&](uint32_t it) {
        const uint32_t row_base = (it % tiles) * BM;
        float* dst = nz > 1u ? y + (size_t)(it / tiles) * out_dim * batch : y;
        const uint32_t r0 = row_base + wr + g, r8 = row_base + wr + 8u + g;
        const float b0f = (bias && r0 < out_dim) ? bias[r0] : 0.0f;
        const float b8f = (bias && r8 < out_dim) ? bias[r8] : 0.0f;
        #pragma unroll
        for (uint32_t sub = 0; sub < NSUB; ++sub) {
            const uint32_t c0 = wc + sub * 8u + 2u * t;
            const uint32_t c1 = c0 + 1u;
            if (r0 < out_dim) {
                if (c0 < batch) dst[(size_t)c0 * out_dim + r0] = bias ? acc[sub][0] + b0f : acc[sub][0];
                if (c1 < batch) dst[(size_t)c1 * out_dim + r0] = bias ? acc[sub][1] + b0f : acc[sub][1];
            }
            if (r8 < out_dim) {
                if (c0 < batch) dst[(size_t)c0 * out_dim + r8] = bias ? acc[sub][2] + b8f : acc[sub][2];
                if (c1 < batch) dst[(size_t)c1 * out_dim + r8] = bias ? acc[sub][3] + b8f : acc[sub][3];
            }
            acc[sub][0] = acc[sub][1] = acc[sub][2] = acc[sub][3] = 0.0f;
        }
    };

    // chained cursors over this CTA's item stripe (empty items store zeros)
    uint32_t cit = blockIdx.x, cst = 0;
    while (cit < nit && scnt(cit) == 0u) { store_item(cit); cit += gridDim.x; }
    if (cit >= nit) return;
    uint32_t pit = cit, pst = 0;  // prefetch cursor
    stage(pit, pst, 0);
    pd_attn_cpa_commit();
    // advance prefetch to the next existing stage (skips empty items)
    auto padv = [&]() {
        if (pit >= nit) return;
        if (++pst >= scnt(pit)) {
            pst = 0;
            do { pit += gridDim.x; } while (pit < nit && scnt(pit) == 0u);
        }
    };
    padv();
    uint32_t p = 0;
    while (cit < nit) {
        if (pit < nit) {
            stage(pit, pst, p ^ 1u);
            pd_attn_cpa_commit();
            pd_attn_cpa_wait1();
            padv();
        } else {
            pd_attn_cpa_wait0();
        }
        __syncthreads();
        compute(p);
        __syncthreads();
        p ^= 1u;
        if (++cst >= scnt(cit)) {
            store_item(cit);
            cst = 0;
            do { cit += gridDim.x; } while (cit < nit && scnt(cit) == 0u ? (store_item(cit), true) : false);
        }
    }
#else
    (void)data; (void)scale; (void)xq; (void)xs; (void)bias; (void)y;
    (void)in_dim; (void)out_dim; (void)batch; (void)nz;
#endif
}

// K-split partial combine for pd_q8_0_gemm_mma_ks: y[i] = sum_z part[z][i],
// fixed z order (deterministic). Optional bias fold (y layout is
// [tok][out_dim], so bias index is i % out_dim): the bias adds after the
// completed z-sum - identical rounding to the separate pd_bias_add pass.
__global__ void pd_q8_0_gemm_mma_ks_combine_kernel(const float* __restrict__ part,
                                                   const float* __restrict__ bias,
                                                   float* __restrict__ y, uint32_t n,
                                                   uint32_t nz, uint32_t out_dim) {
    PD_PDL_ARM();  // fp8-native chain cascade; no-op under plain launches
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = 0.0f;
    for (uint32_t z = 0; z < nz; ++z) acc += part[(size_t)z * n + i];
    if (bias) acc += bias[i % out_dim];
    y[i] = acc;
}

// K-split mma GEMM: the 64x64-tile kernel with grid.z K-ranges writing partial
// planes into `part` (>= nz * out_dim * batch f32; the stream-k fixup scratch
// is big enough for every dense shape), then a fixed-order combine into y.
// Same numeric class as pd_q8_0_gemm_mma (f32 partial regroup only). Picks nz
// to fill the device; nz == 1 collapses to the plain kernel writing y direct.
static int pd_q8_0_gemm_mma_ks_impl(const void* data, const void* scale, const void* xq,
                                    const void* xs, const void* bias, void* part, void* y,
                                    uint32_t in_dim, uint32_t out_dim, uint32_t batch,
                                    void* stream) {
    if (out_dim == 0 || batch == 0) return 0;
    // out_dim needs no alignment: this dispatches into pd_q8_0_gemm_mma_kernel
    // (batch<=32) or pd_q8_0_gemm_mma_mcol_kernel (batch>32), both of which
    // zero-pad rows past out_dim during staging and bounds-check every row
    // at writeback - see pd_q8_0_gemm_mma's guard removal note (verified by
    // tests/gemm_ragged_out_dim.rs). This is the same
    // kernel family reached by a second wrapper (the decode-tick mmq_pre
    // path, laguna S-2.1's g_proj at batch=32 during CUDA-graph-captured
    // decode) - it has its own copy of the historical guard, missed the
    // first time around.
    if (in_dim & 31u) return cudaErrorInvalidValue;
    if (batch > 192u) return cudaErrorInvalidValue;
    auto st = (cudaStream_t)stream;
    static int nsm = 0;
    if (nsm == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
        if (nsm <= 0) nsm = 128;
    }
    const uint32_t tiles = (out_dim + 63u) / 64u;
    const uint32_t n_blocks = in_dim >> 5;
    // K-split target: CTAs ~= mul x SMs. Shape-aware default from the GB202
    // cold-stream ladder (b=32, 27B shapes): small-tile shapes want ~3.5x
    // SM of CTAs for DRAM in-flight
    // (down 84.2 -> 75.6 us, wq|gate 63.3 -> 54.2, gate_w 33.3 -> 30.8),
    // while the 272-tile gate/up is best at the historical 2x (816 CTAs
    // regressed it 2%). PD_KS_NZ_MUL overrides both (x2 fixed = the old
    // behavior).
    static uint32_t nz_mul2x = 0; // fixed-point x2: 4 = 2.0, 7 = 3.5
    if (nz_mul2x == 0) {
        const char* e = pd_env("PD_KS_NZ_MUL");
        int v = e ? atoi(e) : 0;
        nz_mul2x = (v >= 1 && v <= 8) ? (uint32_t)(2 * v) : 0xffu;
    }
    const uint32_t mul2x = nz_mul2x != 0xffu ? nz_mul2x : (tiles >= 250u ? 4u : 7u);
    uint32_t nz = ((uint32_t)nsm * mul2x / 2u + tiles - 1u) / tiles;
    // A tile count that alone fills ~1.3x the die needs no K-split at all -
    // nz=1 writes y direct (no partial planes, no combine). The fused 256-tile
    // DN merge sat at nz=2 paying an 8.4 MB partial round trip for nothing
    // (default path unaffected: 250..1.3x-SM shapes barely exist unfused).
    if (nz_mul2x == 0xffu && tiles * 10u >= (uint32_t)nsm * 13u) nz = 1u;
    const uint32_t max_nz = (n_blocks + 3u) / 4u; // >= 1 K-stage per slice
    if (nz > 8u) nz = 8u;
    if (nz > max_nz) nz = max_nz;
    if (nz < 1u) nz = 1u;
    // Narrow-batch tiles: the 64-token column tile at spec-verify row counts
    // (r = 5..16) is >75% padding compute - the small-r cost that gave back
    // 14% of B=1 spec when the verify ladder moved to ks. Per-element K-fold
    // order is BN-independent (same kt sequence, same z slicing - nz derives
    // from out-tiles only), so the BN16/BN32 rungs are BIT-EQUAL to BN64 and
    // the r-class rules hold; the exact-match spec gates arbitrate.
    float* dst = nz > 1u ? (float*)part : (float*)y;
    // bias rides the final writer: the combine when K is split, the GEMM
    // epilogue when nz collapsed to 1 (partial planes stay unbiased).
    const float* kbias = nz > 1u ? nullptr : (const float*)bias;
    // 2-stage cp.async pipeline on the serving rungs (the single-buffer
    // walk parked every warp on DRAM latency - long_sb 14.9/issue). The 4B
    // scale copies need an even n_blocks; odd keeps the synchronous walk.
    const bool pipe2 = ((in_dim >> 5) & 1u) == 0u;
    if (batch > 64u) {
        // 65..192 (wide-spec verify rows): M-col rung - weights staged once,
        // 64-token col-tiles looped in-register, B/xs direct from L2. Same
        // even-n_blocks guard as the pipelined rungs (A-side 4B copies).
        if (!pipe2) return cudaErrorInvalidValue; // callers fall to mmq
        // dynamic smem: 2 x (A 9216 + ws 512 + B NCOL*9216 + xs NCOL*1024)
        auto mcol_smem = [](uint32_t ncol) {
            const uint32_t one = (9216u + 512u + ncol * 9216u + ncol * 1024u + 15u) & ~15u;
            return 2u * one;
        };
        static uint32_t mcol_set = 0;
        if (mcol_set == 0) {
            cudaFuncSetAttribute((const void*)pd_q8_0_gemm_mma_mcol_kernel<64u, 2u, 8u>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, mcol_smem(2u));
            cudaFuncSetAttribute((const void*)pd_q8_0_gemm_mma_mcol_kernel<64u, 3u, 8u>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, mcol_smem(3u));
            mcol_set = 1;
        }
        dim3 grid(tiles, 1u, nz);
        if (batch <= 128u)
            pd_q8_0_gemm_mma_mcol_kernel<64u, 2u, 8u><<<grid, 256, mcol_smem(2u), st>>>(
                (const int8_t*)data, (const __half*)scale, (const int8_t*)xq,
                (const float*)xs, kbias, dst, in_dim, out_dim, batch);
        else
            pd_q8_0_gemm_mma_mcol_kernel<64u, 3u, 8u><<<grid, 256, mcol_smem(3u), st>>>(
                (const int8_t*)data, (const __half*)scale, (const int8_t*)xq,
                (const float*)xs, kbias, dst, in_dim, out_dim, batch);
        if (nz > 1u) {
            uint32_t n = out_dim * batch;
            pd_q8_0_gemm_mma_ks_combine_kernel<<<(n + 255u) / 256u, 256, 0, st>>>(
                (const float*)part, (const float*)bias, (float*)y, n, nz, out_dim);
        }
        return pd_launch_status();
    }
    if (batch <= 16u) {
        dim3 grid(tiles, 1u, nz);
        if (pipe2)
            pd_q8_0_gemm_mma_kernel<64u, 16u, 8u, 2u><<<grid, 256, 0, st>>>(
                (const int8_t*)data, (const __half*)scale, (const int8_t*)xq, (const float*)xs,
                kbias, dst, in_dim, out_dim, batch);
        else
            pd_q8_0_gemm_mma_kernel<64u, 16u, 8u><<<grid, 256, 0, st>>>(
                (const int8_t*)data, (const __half*)scale, (const int8_t*)xq, (const float*)xs,
                kbias, dst, in_dim, out_dim, batch);
    } else if (batch <= 32u) {
        dim3 grid(tiles, 1u, nz);
        if (pipe2)
            pd_q8_0_gemm_mma_kernel<64u, 32u, 8u, 2u><<<grid, 256, 0, st>>>(
                (const int8_t*)data, (const __half*)scale, (const int8_t*)xq, (const float*)xs,
                kbias, dst, in_dim, out_dim, batch);
        else
            pd_q8_0_gemm_mma_kernel<64u, 32u, 8u><<<grid, 256, 0, st>>>(
                (const int8_t*)data, (const __half*)scale, (const int8_t*)xq, (const float*)xs,
                kbias, dst, in_dim, out_dim, batch);
    } else {
        // 33..64: ST=2 after all - the old "1.54x floor" measurement was the
        // plain-mma lm_head grid (3142 tiles, occupancy-bound); the ks rung's
        // K-split grids are latency-bound like BN16/32, and the WIDE-SPEC
        // verify (r=64: 32 slots x pending+draft) lives exactly here.
        // Single-buffer stays for odd n_blocks (4B scale copy alignment).
        dim3 grid(tiles, (batch + 63u) / 64u, nz);
        if (pipe2)
            pd_q8_0_gemm_mma_kernel<64u, 64u, 8u, 2u><<<grid, 256, 0, st>>>(
                (const int8_t*)data, (const __half*)scale, (const int8_t*)xq, (const float*)xs,
                kbias, dst, in_dim, out_dim, batch);
        else
            pd_q8_0_gemm_mma_kernel<64u, 64u, 8u><<<grid, 256, 0, st>>>(
                (const int8_t*)data, (const __half*)scale, (const int8_t*)xq, (const float*)xs,
                kbias, dst, in_dim, out_dim, batch);
    }
    if (nz > 1u) {
        uint32_t n = out_dim * batch;
        pd_q8_0_gemm_mma_ks_combine_kernel<<<(n + 255u) / 256u, 256, 0, st>>>(
            (const float*)part, (const float*)bias, (float*)y, n, nz, out_dim);
    }
    return pd_launch_status();
}

// ---- e4m3 decode-band twin of the ST=2 serving rung (native-fp8 lane) ----
// Same tile walk, same ldmatrix fragments (8-bit layouts are type-blind), but
// operands are e4m3 with per-32-block e8m0 scale BYTES on both sides (f8w
// weights via pd_q8_0_to_f8w; activations via pd_quantize_e4m3). The mma is
// m16n8k32.f32.e4m3.e4m3.f32 (fresh accumulators per block - cross-block
// scales differ) and the fold is a pure exponent add: ldexpf(d, ws+xs-254).
// Weight stream is 1.031 B/param vs Q8_0's 1.0625 (-3%). PRECISION CLASS:
// e4m3 operands, PPL-gated like the W8 prefill planes - never default-on
// without the gate. sm_89+ (e4m3 mma).
template <uint32_t BN>
__global__ void __launch_bounds__(256) pd_f8_gemm_mma_kernel(
        const unsigned char* __restrict__ data, const unsigned char* __restrict__ scale,
        const unsigned char* __restrict__ xq, const unsigned char* __restrict__ xs,
        float* __restrict__ y, uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
#if PD_MMA_OK && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    constexpr uint32_t BM = 64u, NWARP = 8u, NTH = 256u;
    constexpr uint32_t WR = BM / 16u, WC = NWARP / WR, CPW = BN / WC;
    constexpr uint32_t NSUB = CPW / 8u;
    constexpr uint32_t KT = PD_MMA_KT, NSUBK = KT / 32u, KPAD = KT + 16u;
    constexpr uint32_t I4PR = KT / 16u;
    __shared__ __align__(16) unsigned char sh_a[2][BM * KPAD];
    __shared__ __align__(16) unsigned char sh_b[2][BN * KPAD];
    __shared__ __align__(16) unsigned char sh_ws[2][BM][NSUBK];
    __shared__ __align__(16) unsigned char sh_xs[2][BN][NSUBK];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, t = lane & 3u;
    const uint32_t wr = (warp % WR) * 16u, wc = (warp / WR) * CPW;
    // Grid roles (dflash rung D): blockIdx.x walks the BATCH
    // tiles and blockIdx.y the weight (out_dim) tiles - the reverse of the
    // Q8 rung and of this kernel's f8r twin. Hardware schedules x fastest,
    // so at batch > 64 the blocks that share one weight tile run back-to-
    // back and the 64-row weight tile streams from DRAM once per k-step
    // instead of once per batch tile (the lm_head at 256 verify rows:
    // 1.27 GB read once, not 4x). The activation tile re-read instead is
    // batch x in_dim bytes - L2-resident at any serving width. Per-element
    // arithmetic is unchanged (same tiles, same k-split, same combine
    // order): bit-identical output at every batch, and at batch <= 64 the
    // block sequence is identical too (one batch tile).
    const uint32_t row_base = blockIdx.y * BM, col_base = blockIdx.x * BN;
    const uint32_t n_blocks = in_dim >> 5;
    const uint32_t nz = gridDim.z;
    uint32_t kt_lo = 0, kt_hi = n_blocks;
    if (nz > 1u) {
        const uint32_t per = ((n_blocks + nz - 1u) / nz + NSUBK - 1u) / NSUBK * NSUBK;
        kt_lo = blockIdx.z * per;
        kt_hi = kt_lo + per < n_blocks ? kt_lo + per : n_blocks;
        y += (size_t)blockIdx.z * out_dim * batch;
    }
    auto stage = [&](uint32_t kt, uint32_t buf) {
        #pragma unroll
        for (uint32_t i = tid; i < BM * I4PR; i += NTH) {
            uint32_t row = i / I4PR, k16 = (i % I4PR) * 16u, gk = kt * 32u + k16;
            const bool ok = gk < in_dim && (row_base + row) < out_dim;
            pd_mma_cpa16p(&sh_a[buf][row * KPAD + k16],
                          data + (size_t)(row_base + row) * in_dim + gk, ok);
        }
        #pragma unroll
        for (uint32_t i = tid; i < BN * I4PR; i += NTH) {
            uint32_t col = i / I4PR, k16 = (i % I4PR) * 16u, gk = kt * 32u + k16;
            const bool ok = gk < in_dim && (col_base + col) < batch;
            pd_mma_cpa16p(&sh_b[buf][col * KPAD + k16],
                          xq + (size_t)(col_base + col) * in_dim + gk, ok);
        }
        // scale BYTES: NSUBK=4 per staged row = one 4B copy each
        for (uint32_t i = tid; i < BM; i += NTH) {
            const bool ok = (row_base + i) < out_dim && kt < n_blocks;
            pd_mma_cpa4p(&sh_ws[buf][i][0],
                         scale + (size_t)(row_base + i) * n_blocks + kt, ok);
        }
        for (uint32_t i = tid; i < BN; i += NTH) {
            const bool ok = (col_base + i) < batch && kt < n_blocks;
            pd_mma_cpa4p(&sh_xs[buf][i][0],
                         xs + (size_t)(col_base + i) * n_blocks + kt, ok);
        }
    };
    const uint32_t ldm_l7 = lane & 7u;
    const uint32_t ldm_arow = wr + ((lane & 8u) ? 8u : 0u) + ldm_l7;
    const uint32_t ldm_akof = (lane & 16u) ? 16u : 0u;
    const uint32_t ldm_bkof = (lane & 8u) ? 16u : 0u;
    float acc[NSUB][4] = {};
    auto compute = [&](uint32_t buf) {
        #pragma unroll
        for (uint32_t sb = 0; sb < NSUBK; ++sb) {
            const uint32_t ko = sb * 32u;
            int a0, a1, a2, a3;
            pd_mma_ldm_x4(&sh_a[buf][ldm_arow * KPAD + ko + ldm_akof], a0, a1, a2, a3);
            const int ws0 = (int)sh_ws[buf][wr + g][sb];
            const int ws8 = (int)sh_ws[buf][wr + 8u + g][sb];
            #pragma unroll
            for (uint32_t sub = 0; sub < NSUB; ++sub) {
                const uint32_t csub = wc + sub * 8u;
                int b0, b1;
                pd_mma_ldm_x2(&sh_b[buf][(csub + ldm_l7) * KPAD + ko + ldm_bkof], b0, b1);
                float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
                asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
                const int xc0 = (int)sh_xs[buf][csub + 2u * t][sb];
                const int xc1 = (int)sh_xs[buf][csub + 2u * t + 1u][sb];
                // 2^(ws+xs-254) as a bit-cast float (exponent sums stay well
                // inside f32 range for e4m3-scaled operands) - ldexpf here
                // measured as a slow-path drag on the mma loop
                const float f00 = __uint_as_float((uint32_t)(ws0 + xc0 - 127) << 23);
                const float f01 = __uint_as_float((uint32_t)(ws0 + xc1 - 127) << 23);
                const float f80 = __uint_as_float((uint32_t)(ws8 + xc0 - 127) << 23);
                const float f81 = __uint_as_float((uint32_t)(ws8 + xc1 - 127) << 23);
                acc[sub][0] += f00 * d0;
                acc[sub][1] += f01 * d1;
                acc[sub][2] += f80 * d2;
                acc[sub][3] += f81 * d3;
            }
        }
    };
    // one-ahead double-buffer ring, exactly the q8 rung's scheme: prologue
    // stages buf0; each iteration prefetches kt+NSUBK into p^1 (the buffer
    // computed last iteration, fenced by the trailing barrier), commits, and
    // wait_group(1) retires the oldest group = the buffer about to compute.
    if (kt_lo < kt_hi) stage(kt_lo, 0);
    pd_attn_cpa_commit();
    uint32_t p = 0;
    for (uint32_t kt = kt_lo; kt < kt_hi; kt += NSUBK) {
        const uint32_t pre = kt + NSUBK;
        if (pre < kt_hi) stage(pre, p ^ 1u);
        pd_attn_cpa_commit();
        pd_mma_cpa_waitN<1>();
        __syncthreads();
        compute(p);
        __syncthreads();
        p ^= 1u;
    }
    const uint32_t r0 = row_base + wr + g, r8 = row_base + wr + 8u + g;
    #pragma unroll
    for (uint32_t sub = 0; sub < NSUB; ++sub) {
        const uint32_t c0 = col_base + wc + sub * 8u + 2u * t, c1 = c0 + 1u;
        if (r0 < out_dim) {
            if (c0 < batch) y[(size_t)c0 * out_dim + r0] = acc[sub][0];
            if (c1 < batch) y[(size_t)c1 * out_dim + r0] = acc[sub][1];
        }
        if (r8 < out_dim) {
            if (c0 < batch) y[(size_t)c0 * out_dim + r8] = acc[sub][2];
            if (c1 < batch) y[(size_t)c1 * out_dim + r8] = acc[sub][3];
        }
    }
#else
    (void)data; (void)scale; (void)xq; (void)xs; (void)y;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}

// Per-row-scale twin of pd_f8_gemm_mma_kernel: weight scale is one e8m0
// byte per output row (the scale-free stream - 1.0 B/param, ~3% fewer bytes
// than the per-32 f8w plane; vLLM fp8's granularity). The row scales stage
// once per tile (BM bytes, not per k-stage) and the fold hoists ws out of
// the k loop. Activations keep the per-32 e4m3 class. sm_89+.
template <uint32_t BN>
__global__ void __launch_bounds__(256) pd_f8r_gemm_mma_kernel(
        const unsigned char* __restrict__ data, const unsigned char* __restrict__ scale,
        const unsigned char* __restrict__ xq, const unsigned char* __restrict__ xs,
        float* __restrict__ y, uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
#if PD_MMA_OK && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    constexpr uint32_t BM = 64u, NWARP = 8u, NTH = 256u;
    constexpr uint32_t WR = BM / 16u, WC = NWARP / WR, CPW = BN / WC;
    constexpr uint32_t NSUB = CPW / 8u;
    constexpr uint32_t KT = PD_MMA_KT, NSUBK = KT / 32u, KPAD = KT + 16u;
    constexpr uint32_t I4PR = KT / 16u;
    __shared__ __align__(16) unsigned char sh_a[2][BM * KPAD];
    __shared__ __align__(16) unsigned char sh_b[2][BN * KPAD];
    __shared__ unsigned char sh_wr[BM];
    __shared__ __align__(16) unsigned char sh_xs[2][BN][NSUBK];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, t = lane & 3u;
    const uint32_t wr = (warp % WR) * 16u, wc = (warp / WR) * CPW;
    const uint32_t row_base = blockIdx.x * BM, col_base = blockIdx.y * BN;
    const uint32_t n_blocks = in_dim >> 5;
    const uint32_t nz = gridDim.z;
    uint32_t kt_lo = 0, kt_hi = n_blocks;
    if (nz > 1u) {
        const uint32_t per = ((n_blocks + nz - 1u) / nz + NSUBK - 1u) / NSUBK * NSUBK;
        kt_lo = blockIdx.z * per;
        kt_hi = kt_lo + per < n_blocks ? kt_lo + per : n_blocks;
        y += (size_t)blockIdx.z * out_dim * batch;
    }
    // row scales: once per tile
    if (tid < BM)
        sh_wr[tid] = (row_base + tid) < out_dim ? scale[row_base + tid] : 127u;
    auto stage = [&](uint32_t kt, uint32_t buf) {
        #pragma unroll
        for (uint32_t i = tid; i < BM * I4PR; i += NTH) {
            uint32_t row = i / I4PR, k16 = (i % I4PR) * 16u, gk = kt * 32u + k16;
            const bool ok = gk < in_dim && (row_base + row) < out_dim;
            pd_mma_cpa16p(&sh_a[buf][row * KPAD + k16],
                          data + (size_t)(row_base + row) * in_dim + gk, ok);
        }
        #pragma unroll
        for (uint32_t i = tid; i < BN * I4PR; i += NTH) {
            uint32_t col = i / I4PR, k16 = (i % I4PR) * 16u, gk = kt * 32u + k16;
            const bool ok = gk < in_dim && (col_base + col) < batch;
            pd_mma_cpa16p(&sh_b[buf][col * KPAD + k16],
                          xq + (size_t)(col_base + col) * in_dim + gk, ok);
        }
        for (uint32_t i = tid; i < BN; i += NTH) {
            const bool ok = (col_base + i) < batch && kt < n_blocks;
            pd_mma_cpa4p(&sh_xs[buf][i][0],
                         xs + (size_t)(col_base + i) * n_blocks + kt, ok);
        }
    };
    const uint32_t ldm_l7 = lane & 7u;
    const uint32_t ldm_arow = wr + ((lane & 8u) ? 8u : 0u) + ldm_l7;
    const uint32_t ldm_akof = (lane & 16u) ? 16u : 0u;
    const uint32_t ldm_bkof = (lane & 8u) ? 16u : 0u;
    float acc[NSUB][4] = {};
    auto compute = [&](uint32_t buf) {
        const int ws0 = (int)sh_wr[wr + g];
        const int ws8 = (int)sh_wr[wr + 8u + g];
        #pragma unroll
        for (uint32_t sb = 0; sb < NSUBK; ++sb) {
            const uint32_t ko = sb * 32u;
            int a0, a1, a2, a3;
            pd_mma_ldm_x4(&sh_a[buf][ldm_arow * KPAD + ko + ldm_akof], a0, a1, a2, a3);
            #pragma unroll
            for (uint32_t sub = 0; sub < NSUB; ++sub) {
                const uint32_t csub = wc + sub * 8u;
                int b0, b1;
                pd_mma_ldm_x2(&sh_b[buf][(csub + ldm_l7) * KPAD + ko + ldm_bkof], b0, b1);
                float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
                asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
                const int xc0 = (int)sh_xs[buf][csub + 2u * t][sb];
                const int xc1 = (int)sh_xs[buf][csub + 2u * t + 1u][sb];
                acc[sub][0] += __uint_as_float((uint32_t)(ws0 + xc0 - 127) << 23) * d0;
                acc[sub][1] += __uint_as_float((uint32_t)(ws0 + xc1 - 127) << 23) * d1;
                acc[sub][2] += __uint_as_float((uint32_t)(ws8 + xc0 - 127) << 23) * d2;
                acc[sub][3] += __uint_as_float((uint32_t)(ws8 + xc1 - 127) << 23) * d3;
            }
        }
    };
    if (kt_lo < kt_hi) stage(kt_lo, 0);
    pd_attn_cpa_commit();
    uint32_t p = 0;
    for (uint32_t kt = kt_lo; kt < kt_hi; kt += NSUBK) {
        const uint32_t pre = kt + NSUBK;
        if (pre < kt_hi) stage(pre, p ^ 1u);
        pd_attn_cpa_commit();
        pd_mma_cpa_waitN<1>();
        __syncthreads();
        compute(p);
        __syncthreads();
        p ^= 1u;
    }
    const uint32_t r0 = row_base + wr + g, r8 = row_base + wr + 8u + g;
    #pragma unroll
    for (uint32_t sub = 0; sub < NSUB; ++sub) {
        const uint32_t c0 = col_base + wc + sub * 8u + 2u * t, c1 = c0 + 1u;
        if (r0 < out_dim) {
            if (c0 < batch) y[(size_t)c0 * out_dim + r0] = acc[sub][0];
            if (c1 < batch) y[(size_t)c1 * out_dim + r0] = acc[sub][1];
        }
        if (r8 < out_dim) {
            if (c0 < batch) y[(size_t)c0 * out_dim + r8] = acc[sub][2];
            if (c1 < batch) y[(size_t)c1 * out_dim + r8] = acc[sub][3];
        }
    }
#else
    (void)data; (void)scale; (void)xq; (void)xs; (void)y;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}

// Fold-free f32-row-scale twin of pd_f8r_gemm_mma_kernel (mamba
// proj rung): the f8row scale class - f32 scales on both sides,
// applied as a pure per-(r,c) epilogue - on the decode-shaped BM=64 tile.
// The parent's e8m0 fold machinery (sh_wr/sh_xs staging + the per-mma
// exponent add) is DELETED: the mma accumulates raw in place, which is
// exactly pd_f8row_gemm_kt's arithmetic on a tile that fits the batch
// (the 128x128 tile at batch 32 threw away 3/4 of every mma and
// its 64-float accumulator pinned residency at 2 blocks/SM). Same
// ascending m16n8k32 chain as the kt kernel, so an nz=1 launch is
// BIT-IDENTICAL to it (the unit gate); K-split slabs (grid.z, 128-elem
// granularity from the KT=128 stage) write per-slab planes for the same
// fixed-order ks_combine - per-slab scaled partials sum exactly because
// the scales are per-(r,c) constants.
// Body shared by the 1-segment kernel below and the 2-segment gate|up twin
// (pd_f8row_gemm_mma2_kernel): tile coordinates come in as parameters so both
// wrappers execute the identical instruction stream per tile (bit-exact by
// construction). Two-segment rung: one grid covers two planes,
// CTA picks its plane by row-tile -- the memory-neutral form of vLLM's fused
// gate_up (a concat plane would duplicate 4.2 GB on 8b).
template <uint32_t BN>
__device__ __forceinline__ void pd_f8row_gemm_mma_body(
        const unsigned char* __restrict__ data, const float* __restrict__ wrs,
        const unsigned char* __restrict__ xq, const float* __restrict__ xrs,
        float* __restrict__ y, uint32_t in_dim, uint32_t out_dim, uint32_t batch,
        uint32_t bx, uint32_t by, uint32_t bz, uint32_t nz) {
#if PD_MMA_OK && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    constexpr uint32_t BM = 64u, NWARP = 8u, NTH = 256u;
    constexpr uint32_t WR = BM / 16u, WC = NWARP / WR, CPW = BN / WC;
    constexpr uint32_t NSUB = CPW / 8u;
    constexpr uint32_t KT = PD_MMA_KT, NSUBK = KT / 32u, KPAD = KT + 16u;
    constexpr uint32_t I4PR = KT / 16u;
    __shared__ __align__(16) unsigned char sh_a[2][BM * KPAD];
    __shared__ __align__(16) unsigned char sh_b[2][BN * KPAD];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, t = lane & 3u;
    const uint32_t wr = (warp % WR) * 16u, wc = (warp / WR) * CPW;
    const uint32_t row_base = bx * BM, col_base = by * BN;
    const uint32_t n_blocks = in_dim >> 5;
    uint32_t kt_lo = 0, kt_hi = n_blocks;
    if (nz > 1u) {
        const uint32_t per = ((n_blocks + nz - 1u) / nz + NSUBK - 1u) / NSUBK * NSUBK;
        kt_lo = bz * per;
        kt_hi = kt_lo + per < n_blocks ? kt_lo + per : n_blocks;
        y += (size_t)bz * out_dim * batch;
    }
    auto stage = [&](uint32_t kt, uint32_t buf) {
        #pragma unroll
        for (uint32_t i = tid; i < BM * I4PR; i += NTH) {
            uint32_t row = i / I4PR, k16 = (i % I4PR) * 16u, gk = kt * 32u + k16;
            const bool ok = gk < in_dim && (row_base + row) < out_dim;
            pd_mma_cpa16p(&sh_a[buf][row * KPAD + k16],
                          data + (size_t)(row_base + row) * in_dim + gk, ok);
        }
        #pragma unroll
        for (uint32_t i = tid; i < BN * I4PR; i += NTH) {
            uint32_t col = i / I4PR, k16 = (i % I4PR) * 16u, gk = kt * 32u + k16;
            const bool ok = gk < in_dim && (col_base + col) < batch;
            pd_mma_cpa16p(&sh_b[buf][col * KPAD + k16],
                          xq + (size_t)(col_base + col) * in_dim + gk, ok);
        }
    };
    const uint32_t ldm_l7 = lane & 7u;
    const uint32_t ldm_arow = wr + ((lane & 8u) ? 8u : 0u) + ldm_l7;
    const uint32_t ldm_akof = (lane & 16u) ? 16u : 0u;
    const uint32_t ldm_bkof = (lane & 8u) ? 16u : 0u;
    float acc[NSUB][4] = {};
    auto compute = [&](uint32_t buf) {
        #pragma unroll
        for (uint32_t sb = 0; sb < NSUBK; ++sb) {
            const uint32_t ko = sb * 32u;
            int a0, a1, a2, a3;
            pd_mma_ldm_x4(&sh_a[buf][ldm_arow * KPAD + ko + ldm_akof], a0, a1, a2, a3);
            #pragma unroll
            for (uint32_t sub = 0; sub < NSUB; ++sub) {
                const uint32_t csub = wc + sub * 8u;
                int b0, b1;
                pd_mma_ldm_x2(&sh_b[buf][(csub + ldm_l7) * KPAD + ko + ldm_bkof], b0, b1);
                asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+f"(acc[sub][0]), "+f"(acc[sub][1]), "+f"(acc[sub][2]),
                      "+f"(acc[sub][3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            }
        }
    };
    if (kt_lo < kt_hi) stage(kt_lo, 0);
    pd_attn_cpa_commit();
    uint32_t p = 0;
    for (uint32_t kt = kt_lo; kt < kt_hi; kt += NSUBK) {
        const uint32_t pre = kt + NSUBK;
        if (pre < kt_hi) stage(pre, p ^ 1u);
        pd_attn_cpa_commit();
        pd_mma_cpa_waitN<1>();
        __syncthreads();
        compute(p);
        __syncthreads();
        p ^= 1u;
    }
    // epilogue: the only place scales exist. y[c*out+r] = acc * wrs[r] * xrs[c]
    const uint32_t r0 = row_base + wr + g, r8 = row_base + wr + 8u + g;
    const float w0 = r0 < out_dim ? wrs[r0] : 0.0f;
    const float w8 = r8 < out_dim ? wrs[r8] : 0.0f;
    #pragma unroll
    for (uint32_t sub = 0; sub < NSUB; ++sub) {
        const uint32_t c0 = col_base + wc + sub * 8u + 2u * t, c1 = c0 + 1u;
        const float x0 = c0 < batch ? xrs[c0] : 0.0f;
        const float x1 = c1 < batch ? xrs[c1] : 0.0f;
        if (r0 < out_dim) {
            if (c0 < batch) y[(size_t)c0 * out_dim + r0] = acc[sub][0] * w0 * x0;
            if (c1 < batch) y[(size_t)c1 * out_dim + r0] = acc[sub][1] * w0 * x1;
        }
        if (r8 < out_dim) {
            if (c0 < batch) y[(size_t)c0 * out_dim + r8] = acc[sub][2] * w8 * x0;
            if (c1 < batch) y[(size_t)c1 * out_dim + r8] = acc[sub][3] * w8 * x1;
        }
    }
#else
    (void)data; (void)wrs; (void)xq; (void)xrs; (void)y;
    (void)in_dim; (void)out_dim; (void)batch; (void)bx; (void)by; (void)bz; (void)nz;
#endif
}

template <uint32_t BN>
__global__ void __launch_bounds__(256) pd_f8row_gemm_mma_kernel(
        const unsigned char* __restrict__ data, const float* __restrict__ wrs,
        const unsigned char* __restrict__ xq, const float* __restrict__ xrs,
        float* __restrict__ y, uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
    PD_PDL_ARM();  // fp8-native chain cascade; no-op under plain launches
    pd_f8row_gemm_mma_body<BN>(data, wrs, xq, xrs, y, in_dim, out_dim, batch,
                               blockIdx.x, blockIdx.y, blockIdx.z, gridDim.z);
}
// two-segment twin: grid.x = mt0 + mt1 row tiles; CTAs [0, mt0) run plane 0
// into y0, the rest plane 1 into y1. Unsplit only (nz == 1): the segments
// would otherwise need two partial planes and two combines.
template <uint32_t BN>
__global__ void __launch_bounds__(256) pd_f8row_gemm_mma2_kernel(
        const unsigned char* __restrict__ d0, const float* __restrict__ w0,
        float* __restrict__ y0, uint32_t mt0,
        const unsigned char* __restrict__ d1, const float* __restrict__ w1,
        float* __restrict__ y1,
        const unsigned char* __restrict__ xq, const float* __restrict__ xrs,
        uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
    PD_PDL_ARM();
    const bool seg1 = blockIdx.x >= mt0;
    pd_f8row_gemm_mma_body<BN>(seg1 ? d1 : d0, seg1 ? w1 : w0, xq, xrs, seg1 ? y1 : y0,
                               in_dim, out_dim, batch,
                               seg1 ? blockIdx.x - mt0 : blockIdx.x, blockIdx.y, 0u, 1u);
}


// Fold-free wide-row mcol on the f8row class: NCOL 64-col B tiles staged in
// dynamic smem next to the A tile (double-buffered cp.async), weights read
// once per K-slab for up to 192 rows, scales a pure (r,c) epilogue. Per
// 64-col slice the k-walk is the BN64 parent's - BIT-EQUAL vs its grid.y
// col-tile launch (max|d| = 0 on every shape/width). Cold 12-plane
// measurement against the kt tail this replaces at 65..192: wq 683-744 vs
// 289-293 GB/s-wt (2.4-2.6x), NCOL3 at 160-192 still 1.7x; the compute roof at
// f32-acc is 451e12/(2*rows) - the honest ceiling this band lives under
// (and why the SCHEDULER caps fused-tick chunks near this window: vLLM's
// token budget holds their mixed steps at ~160 rows for the same reason).
template <uint32_t NCOL>
__global__ void __launch_bounds__(256) pd_f8row_gemm_mcol_kernel(
    const unsigned char* __restrict__ data, const float* __restrict__ wrs,
    const unsigned char* __restrict__ xq, const float* __restrict__ xrs,
    float* __restrict__ y, uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
#if PD_MMA_OK && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    constexpr uint32_t BM = 64u, BN = 64u, NWARP = 8u, NTH = 256u;
    constexpr uint32_t WR = BM / 16u, WC = NWARP / WR, CPW = BN / WC;
    constexpr uint32_t NSUB = CPW / 8u;
    constexpr uint32_t KT = PD_MMA_KT, NSUBK = KT / 32u, KPAD = KT + 16u;
    constexpr uint32_t I4PR = KT / 16u;
    constexpr uint32_t A_BYTES = BM * KPAD;
    constexpr uint32_t B_BYTES = NCOL * BN * KPAD;
    constexpr uint32_t BUF = (A_BYTES + B_BYTES + 15u) & ~15u;
    extern __shared__ __align__(16) unsigned char f8mc_sh[];
    auto sh_a = [&](uint32_t b) { return f8mc_sh + (size_t)b * BUF; };
    auto sh_b = [&](uint32_t b) { return f8mc_sh + (size_t)b * BUF + A_BYTES; };
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, t = lane & 3u;
    const uint32_t wr = (warp % WR) * 16u, wc = (warp / WR) * CPW;
    const uint32_t row_base = blockIdx.x * BM;
    const uint32_t n_blocks = in_dim >> 5;
    const uint32_t nz = gridDim.z;
    uint32_t kt_lo = 0, kt_hi = n_blocks;
    if (nz > 1u) {
        const uint32_t per = ((n_blocks + nz - 1u) / nz + NSUBK - 1u) / NSUBK * NSUBK;
        kt_lo = blockIdx.z * per;
        kt_hi = kt_lo + per < n_blocks ? kt_lo + per : n_blocks;
        y += (size_t)blockIdx.z * out_dim * batch;
    }
    auto stage = [&](uint32_t kt, uint32_t buf) {
        unsigned char* a = sh_a(buf);
        unsigned char* b = sh_b(buf);
        #pragma unroll
        for (uint32_t i = tid; i < BM * I4PR; i += NTH) {
            uint32_t row = i / I4PR, k16 = (i % I4PR) * 16u, gk = kt * 32u + k16;
            const bool ok = gk < in_dim && (row_base + row) < out_dim;
            pd_mma_cpa16p(&a[row * KPAD + k16],
                          data + (size_t)(row_base + row) * in_dim + gk, ok);
        }
        #pragma unroll
        for (uint32_t i = tid; i < NCOL * BN * I4PR; i += NTH) {
            uint32_t col = i / I4PR, k16 = (i % I4PR) * 16u, gk = kt * 32u + k16;
            const bool ok = gk < in_dim && col < batch;
            pd_mma_cpa16p(&b[col * KPAD + k16],
                          xq + (size_t)col * in_dim + gk, ok);
        }
    };
    const uint32_t ldm_l7 = lane & 7u;
    const uint32_t ldm_arow = wr + ((lane & 8u) ? 8u : 0u) + ldm_l7;
    const uint32_t ldm_akof = (lane & 16u) ? 16u : 0u;
    const uint32_t ldm_bkof = (lane & 8u) ? 16u : 0u;
    float acc[NCOL][NSUB][4] = {};
    auto compute = [&](uint32_t buf) {
        const unsigned char* a = sh_a(buf);
        const unsigned char* b = sh_b(buf);
        #pragma unroll
        for (uint32_t sb = 0; sb < NSUBK; ++sb) {
            const uint32_t ko = sb * 32u;
            int a0, a1, a2, a3;
            pd_mma_ldm_x4(&a[ldm_arow * KPAD + ko + ldm_akof], a0, a1, a2, a3);
            #pragma unroll
            for (uint32_t ct = 0; ct < NCOL; ++ct) {
                #pragma unroll
                for (uint32_t sub = 0; sub < NSUB; ++sub) {
                    const uint32_t csub = ct * BN + wc + sub * 8u;
                    int b0, b1;
                    pd_mma_ldm_x2(&b[(csub + ldm_l7) * KPAD + ko + ldm_bkof], b0, b1);
                    asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+f"(acc[ct][sub][0]), "+f"(acc[ct][sub][1]),
                          "+f"(acc[ct][sub][2]), "+f"(acc[ct][sub][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
                }
            }
        }
    };
    if (kt_lo < kt_hi) stage(kt_lo, 0);
    pd_attn_cpa_commit();
    uint32_t p = 0;
    for (uint32_t kt = kt_lo; kt < kt_hi; kt += NSUBK) {
        const uint32_t pre = kt + NSUBK;
        if (pre < kt_hi) stage(pre, p ^ 1u);
        pd_attn_cpa_commit();
        pd_mma_cpa_waitN<1>();
        __syncthreads();
        compute(p);
        __syncthreads();
        p ^= 1u;
    }
    const uint32_t r0 = row_base + wr + g, r8 = row_base + wr + 8u + g;
    const float w0 = r0 < out_dim ? wrs[r0] : 0.0f;
    const float w8 = r8 < out_dim ? wrs[r8] : 0.0f;
    #pragma unroll
    for (uint32_t ct = 0; ct < NCOL; ++ct) {
        #pragma unroll
        for (uint32_t sub = 0; sub < NSUB; ++sub) {
            const uint32_t c0 = ct * BN + wc + sub * 8u + 2u * t, c1 = c0 + 1u;
            const float x0 = c0 < batch ? xrs[c0] : 0.0f;
            const float x1 = c1 < batch ? xrs[c1] : 0.0f;
            if (r0 < out_dim) {
                if (c0 < batch) y[(size_t)c0 * out_dim + r0] = acc[ct][sub][0] * w0 * x0;
                if (c1 < batch) y[(size_t)c1 * out_dim + r0] = acc[ct][sub][1] * w0 * x1;
            }
            if (r8 < out_dim) {
                if (c0 < batch) y[(size_t)c0 * out_dim + r8] = acc[ct][sub][2] * w8 * x0;
                if (c1 < batch) y[(size_t)c1 * out_dim + r8] = acc[ct][sub][3] * w8 * x1;
            }
        }
    }
#else
    (void)data; (void)wrs; (void)xq; (void)xrs; (void)y;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}


// PD_KS_NZ_MUL on the f8 lanes mirrors the Q8 rung: forces the CTA
// multiplier AND skips the die-filling nz=1 collapse. The f8 stream is
// 1 B/elem (half Q8's bytes per K-step), so the Q8-tuned collapse has to be
// re-measured on this lane, not assumed.
static uint32_t pd_f8_nz_mul2x() {
    static uint32_t m = 0;
    if (m == 0) {
        const char* e = pd_env("PD_KS_NZ_MUL");
        int v = e ? atoi(e) : 0;
        m = (v >= 1 && v <= 8) ? (uint32_t)(2 * v) : 0xffu;
    }
    return m;
}
PD_EXPORT
int pd_f8r_gemm_mma_ks(const void* data, const void* scale, const void* xq,
                       const void* xs, void* part, void* y, uint32_t in_dim,
                       uint32_t out_dim, uint32_t batch, void* stream) {
    if (out_dim == 0 || batch == 0) return 0;
    if ((out_dim & 15u) || (in_dim & 31u) || ((in_dim >> 5) & 1u)) return cudaErrorInvalidValue;
    if (batch > 64u) return cudaErrorInvalidValue;
    auto st = (cudaStream_t)stream;
    static int nsm = 0;
    if (nsm == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
        if (nsm <= 0) nsm = 128;
    }
    const uint32_t tiles = (out_dim + 63u) / 64u;
    const uint32_t n_blocks = in_dim >> 5;
    const uint32_t env2x = pd_f8_nz_mul2x();
    const uint32_t mul2x = env2x != 0xffu ? env2x : (tiles >= 250u ? 4u : 7u);
    uint32_t nz = ((uint32_t)nsm * mul2x / 2u + tiles - 1u) / tiles;
    const uint32_t max_nz = (n_blocks + 3u) / 4u;
    if (nz > 8u) nz = 8u;
    if (nz > max_nz) nz = max_nz;
    if (nz < 1u) nz = 1u;
    if (env2x == 0xffu && tiles * 10u >= (uint32_t)nsm * 13u) nz = 1u;
    float* dst = nz > 1u ? (float*)part : (float*)y;
    dim3 grid(tiles, (batch + 63u) / 64u, nz);
    if (batch <= 16u)
        pd_f8r_gemm_mma_kernel<16u><<<grid, 256, 0, st>>>(
            (const unsigned char*)data, (const unsigned char*)scale,
            (const unsigned char*)xq, (const unsigned char*)xs, dst, in_dim, out_dim, batch);
    else if (batch <= 32u)
        pd_f8r_gemm_mma_kernel<32u><<<grid, 256, 0, st>>>(
            (const unsigned char*)data, (const unsigned char*)scale,
            (const unsigned char*)xq, (const unsigned char*)xs, dst, in_dim, out_dim, batch);
    else
        pd_f8r_gemm_mma_kernel<64u><<<grid, 256, 0, st>>>(
            (const unsigned char*)data, (const unsigned char*)scale,
            (const unsigned char*)xq, (const unsigned char*)xs, dst, in_dim, out_dim, batch);
    if (nz > 1u) {
        uint32_t n = out_dim * batch;
        pd_q8_0_gemm_mma_ks_combine_kernel<<<(n + 255u) / 256u, 256, 0, st>>>(
            (const float*)part, nullptr, (float*)y, n, nz, out_dim);
    }
    return pd_launch_status();
}

// e4m3 ks launcher: same nz policy as the Q8 rung (shape-aware target, nz=1
// for die-filling tile counts), same combine. Any batch (dflash rung D,
// the old `b <= 64` refusal was the 64-row wall under every
// qwen35 spec round deeper than k=1 at 32 live - the verify walk and the
// block drafter's head both call this at batch x k1 rows, and the kernel
// already tiles the batch (grid.x, 64 per tile; see the grid-role note on
// the kernel). Callers size `part` for nz x batch x out_dim.
PD_EXPORT
int pd_f8d_gemm_mma_ks(const void* data, const void* scale, const void* xq,
                      const void* xs, void* part, void* y, uint32_t in_dim,
                      uint32_t out_dim, uint32_t batch, void* stream) {
    if (out_dim == 0 || batch == 0) return 0;
    if ((out_dim & 15u) || (in_dim & 31u) || ((in_dim >> 5) & 1u)) return cudaErrorInvalidValue;
    auto st = (cudaStream_t)stream;
    static int nsm = 0;
    if (nsm == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
        if (nsm <= 0) nsm = 128;
    }
    const uint32_t tiles = (out_dim + 63u) / 64u;
    const uint32_t n_blocks = in_dim >> 5;
    const uint32_t env2x = pd_f8_nz_mul2x();
    const uint32_t mul2x = env2x != 0xffu ? env2x : (tiles >= 250u ? 4u : 7u);
    uint32_t nz = ((uint32_t)nsm * mul2x / 2u + tiles - 1u) / tiles;
    const uint32_t max_nz = (n_blocks + 3u) / 4u;
    if (nz > 8u) nz = 8u;
    if (nz > max_nz) nz = max_nz;
    if (nz < 1u) nz = 1u;
    if (env2x == 0xffu && tiles * 10u >= (uint32_t)nsm * 13u) nz = 1u;
    float* dst = nz > 1u ? (float*)part : (float*)y;
    // grid.x = batch tiles, grid.y = weight tiles (see the kernel's grid-role note)
    dim3 grid((batch + 63u) / 64u, tiles, nz);
    if (batch <= 16u)
        pd_f8_gemm_mma_kernel<16u><<<grid, 256, 0, st>>>(
            (const unsigned char*)data, (const unsigned char*)scale,
            (const unsigned char*)xq, (const unsigned char*)xs, dst, in_dim, out_dim, batch);
    else if (batch <= 32u)
        pd_f8_gemm_mma_kernel<32u><<<grid, 256, 0, st>>>(
            (const unsigned char*)data, (const unsigned char*)scale,
            (const unsigned char*)xq, (const unsigned char*)xs, dst, in_dim, out_dim, batch);
    else
        pd_f8_gemm_mma_kernel<64u><<<grid, 256, 0, st>>>(
            (const unsigned char*)data, (const unsigned char*)scale,
            (const unsigned char*)xq, (const unsigned char*)xs, dst, in_dim, out_dim, batch);
    if (nz > 1u) {
        uint32_t n = out_dim * batch;
        pd_q8_0_gemm_mma_ks_combine_kernel<<<(n + 255u) / 256u, 256, 0, st>>>(
            (const float*)part, nullptr, (float*)y, n, nz, out_dim);
    }
    return pd_launch_status();
}

PD_EXPORT
int pd_q8_0_gemm_mma_ks(const void* data, const void* scale, const void* xq,
                        const void* xs, void* part, void* y, uint32_t in_dim,
                        uint32_t out_dim, uint32_t batch, void* stream) {
    return pd_q8_0_gemm_mma_ks_impl(data, scale, xq, xs, nullptr, part, y, in_dim,
                                    out_dim, batch, stream);
}

// Bias-carrying variant for the serving 9..=64 dense rung (gpt-oss projections
// are biased): the fold is bit-exact vs GEMM + pd_bias_add - see the combine
// kernel note.
PD_EXPORT
int pd_q8_0_gemm_mma_ks_b(const void* data, const void* scale, const void* xq,
                          const void* xs, const void* bias, void* part, void* y,
                          uint32_t in_dim, uint32_t out_dim, uint32_t batch,
                          void* stream) {
    return pd_q8_0_gemm_mma_ks_impl(data, scale, xq, xs, bias, part, y, in_dim,
                                    out_dim, batch, stream);
}

// wqkv all-in-one: the ks GEMM always writes the partial planes, then the
// fused combine+rope+append kernel (above, near the plain rope kernel)
// consumes them - no y materialization, no separate combine/rope launches.
// Bit-identical to mma_ks_b -> qkv_rope_append_batch.
PD_EXPORT
int pd_q8_0_gemm_mma_ks_qkv_rope(
    const void* data, const void* scale, const void* xq, const void* xs,
    const void* bias, void* part, void* q_out, void* k_cache, void* v_cache,
    const void* positions, const void* slots, uint32_t in_dim, uint32_t n_heads,
    uint32_t n_kv_heads, uint32_t head_dim, uint32_t max_ctx, float theta_scale,
    float freq_scale, float corr_low, float corr_high, float ext_factor,
    float mscale, uint32_t batch, uint32_t kv_dtype, void* stream) {
    const uint32_t out_dim = (n_heads + 2u * n_kv_heads) * head_dim;
    if (out_dim == 0 || batch == 0) return 0;
    if ((out_dim & 15u) || (in_dim & 31u)) return cudaErrorInvalidValue;
    if (batch > 64u) return cudaErrorInvalidValue;
    auto st = (cudaStream_t)stream;
    static int nsm = 0;
    if (nsm == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
        if (nsm <= 0) nsm = 128;
    }
    const uint32_t tiles = (out_dim + 63u) / 64u;
    const uint32_t n_blocks = in_dim >> 5;
    uint32_t nz = ((uint32_t)nsm * 2u + tiles - 1u) / tiles;
    const uint32_t max_nz = (n_blocks + 3u) / 4u;
    if (nz > 8u) nz = 8u;
    if (nz > max_nz) nz = max_nz;
    if (nz < 1u) nz = 1u;
    const bool pipe2 = ((in_dim >> 5) & 1u) == 0u;
    float* dst = (float*)part;  // Always partials - the rope kernel combines
    if (batch <= 16u) {
        dim3 grid(tiles, 1u, nz);
        if (pipe2)
            pd_q8_0_gemm_mma_kernel<64u, 16u, 8u, 2u><<<grid, 256, 0, st>>>(
                (const int8_t*)data, (const __half*)scale, (const int8_t*)xq,
                (const float*)xs, nullptr, dst, in_dim, out_dim, batch);
        else
            pd_q8_0_gemm_mma_kernel<64u, 16u, 8u><<<grid, 256, 0, st>>>(
                (const int8_t*)data, (const __half*)scale, (const int8_t*)xq,
                (const float*)xs, nullptr, dst, in_dim, out_dim, batch);
    } else if (batch <= 32u) {
        dim3 grid(tiles, 1u, nz);
        if (pipe2)
            pd_q8_0_gemm_mma_kernel<64u, 32u, 8u, 2u><<<grid, 256, 0, st>>>(
                (const int8_t*)data, (const __half*)scale, (const int8_t*)xq,
                (const float*)xs, nullptr, dst, in_dim, out_dim, batch);
        else
            pd_q8_0_gemm_mma_kernel<64u, 32u, 8u><<<grid, 256, 0, st>>>(
                (const int8_t*)data, (const __half*)scale, (const int8_t*)xq,
                (const float*)xs, nullptr, dst, in_dim, out_dim, batch);
    } else {
        dim3 grid(tiles, (batch + 63u) / 64u, nz);
        pd_q8_0_gemm_mma_kernel<64u, 64u, 8u><<<grid, 256, 0, st>>>(
            (const int8_t*)data, (const __half*)scale, (const int8_t*)xq,
            (const float*)xs, nullptr, dst, in_dim, out_dim, batch);
    }
    const uint32_t warps = batch * (n_heads + 2u * n_kv_heads);
    const uint32_t blocks = (warps + 7u) / 8u;
    if (kv_dtype == PD_KV_FP8_E4M3) {
        pd_ks_qkv_rope_append_kernel<__nv_fp8_e4m3><<<blocks, 256, 0, st>>>(
            (const float*)part, (const float*)bias, (float*)q_out,
            (__nv_fp8_e4m3*)k_cache, (__nv_fp8_e4m3*)v_cache,
            (const unsigned int*)positions, (const unsigned int*)slots, n_heads,
            n_kv_heads, head_dim, max_ctx, theta_scale, freq_scale, corr_low,
            corr_high, ext_factor, mscale, batch, nz);
    } else {
        pd_ks_qkv_rope_append_kernel<__half><<<blocks, 256, 0, st>>>(
            (const float*)part, (const float*)bias, (float*)q_out,
            (__half*)k_cache, (__half*)v_cache, (const unsigned int*)positions,
            (const unsigned int*)slots, n_heads, n_kv_heads, head_dim, max_ctx,
            theta_scale, freq_scale, corr_low, corr_high, ext_factor, mscale,
            batch, nz);
    }
    return pd_launch_status();
}

// Paged twin of pd_q8_0_gemm_mma_ks_qkv_rope: identical GEMM into the partial
// planes, then the PAGED combine+rope+append kernel (block-table K/V store).
// max_ctx dropped; block_tables + blocks_per_slot appended. Bit-identical to the
// dense launcher under an identity table.
PD_EXPORT
int pd_q8_0_gemm_mma_ks_qkv_rope_paged(
    const void* data, const void* scale, const void* xq, const void* xs,
    const void* bias, void* part, void* q_out, void* k_cache, void* v_cache,
    const void* positions, const void* slots, uint32_t in_dim, uint32_t n_heads,
    uint32_t n_kv_heads, uint32_t head_dim, float theta_scale,
    float freq_scale, float corr_low, float corr_high, float ext_factor,
    float mscale, uint32_t batch, const void* block_tables, uint32_t blocks_per_slot,
    uint32_t kv_dtype, void* stream) {
    const uint32_t out_dim = (n_heads + 2u * n_kv_heads) * head_dim;
    if (out_dim == 0 || batch == 0) return 0;
    if ((out_dim & 15u) || (in_dim & 31u)) return cudaErrorInvalidValue;
    if (batch > 64u) return cudaErrorInvalidValue;
    auto st = (cudaStream_t)stream;
    static int nsm = 0;
    if (nsm == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
        if (nsm <= 0) nsm = 128;
    }
    const uint32_t tiles = (out_dim + 63u) / 64u;
    const uint32_t n_blocks = in_dim >> 5;
    uint32_t nz = ((uint32_t)nsm * 2u + tiles - 1u) / tiles;
    const uint32_t max_nz = (n_blocks + 3u) / 4u;
    if (nz > 8u) nz = 8u;
    if (nz > max_nz) nz = max_nz;
    if (nz < 1u) nz = 1u;
    const bool pipe2 = ((in_dim >> 5) & 1u) == 0u;
    float* dst = (float*)part;  // Always partials - the rope kernel combines
    if (batch <= 16u) {
        dim3 grid(tiles, 1u, nz);
        if (pipe2)
            pd_q8_0_gemm_mma_kernel<64u, 16u, 8u, 2u><<<grid, 256, 0, st>>>(
                (const int8_t*)data, (const __half*)scale, (const int8_t*)xq,
                (const float*)xs, nullptr, dst, in_dim, out_dim, batch);
        else
            pd_q8_0_gemm_mma_kernel<64u, 16u, 8u><<<grid, 256, 0, st>>>(
                (const int8_t*)data, (const __half*)scale, (const int8_t*)xq,
                (const float*)xs, nullptr, dst, in_dim, out_dim, batch);
    } else if (batch <= 32u) {
        dim3 grid(tiles, 1u, nz);
        if (pipe2)
            pd_q8_0_gemm_mma_kernel<64u, 32u, 8u, 2u><<<grid, 256, 0, st>>>(
                (const int8_t*)data, (const __half*)scale, (const int8_t*)xq,
                (const float*)xs, nullptr, dst, in_dim, out_dim, batch);
        else
            pd_q8_0_gemm_mma_kernel<64u, 32u, 8u><<<grid, 256, 0, st>>>(
                (const int8_t*)data, (const __half*)scale, (const int8_t*)xq,
                (const float*)xs, nullptr, dst, in_dim, out_dim, batch);
    } else {
        dim3 grid(tiles, (batch + 63u) / 64u, nz);
        pd_q8_0_gemm_mma_kernel<64u, 64u, 8u><<<grid, 256, 0, st>>>(
            (const int8_t*)data, (const __half*)scale, (const int8_t*)xq,
            (const float*)xs, nullptr, dst, in_dim, out_dim, batch);
    }
    const uint32_t warps = batch * (n_heads + 2u * n_kv_heads);
    const uint32_t blocks = (warps + 7u) / 8u;
    if (kv_dtype == PD_KV_FP8_E4M3) {
        pd_ks_qkv_rope_append_paged_kernel<__nv_fp8_e4m3><<<blocks, 256, 0, st>>>(
            (const float*)part, (const float*)bias, (float*)q_out,
            (__nv_fp8_e4m3*)k_cache, (__nv_fp8_e4m3*)v_cache,
            (const unsigned int*)positions, (const unsigned int*)slots, n_heads,
            n_kv_heads, head_dim, theta_scale, freq_scale, corr_low,
            corr_high, ext_factor, mscale, batch, nz,
            (const uint32_t*)block_tables, blocks_per_slot);
    } else {
        pd_ks_qkv_rope_append_paged_kernel<__half><<<blocks, 256, 0, st>>>(
            (const float*)part, (const float*)bias, (float*)q_out,
            (__half*)k_cache, (__half*)v_cache, (const unsigned int*)positions,
            (const unsigned int*)slots, n_heads, n_kv_heads, head_dim,
            theta_scale, freq_scale, corr_low, corr_high, ext_factor, mscale,
            batch, nz, (const uint32_t*)block_tables, blocks_per_slot);
    }
    return pd_launch_status();
}

// Gated delta recurrence with per-token state snapshots - the speculative-decode
// verify pass. Identical math/order to pd_gated_delta_recurrent_kernel (state
// columns live in registers across the token loop; f32 exact), but after each
// token t the thread also writes its state column into snap[t] - so a partial
// draft acceptance can roll the recurrent state back to any position with one
// memcpy instead of a re-forward. snap is [n_tokens, n_heads, D, D].
template <typename ST = float>
__global__ void pd_gated_delta_recurrent_snap_kernel(
        const float* __restrict__ q, const float* __restrict__ k,
        const float* __restrict__ v, const float* __restrict__ g,
        const float* __restrict__ beta, ST* __restrict__ state,
        float* __restrict__ out, ST* __restrict__ snap,
        uint32_t n_tokens, uint32_t n_heads, uint32_t D) {
    const uint32_t h = blockIdx.x;
    const uint32_t j = threadIdx.x;
    if (h >= n_heads || j >= D) return;

    extern __shared__ float smem[];
    float* q_sh = smem;
    float* k_sh = smem + D;
    float* red  = smem + 2 * D;
    const float scale = rsqrtf((float)D);

    float col[PD_DN_MAX_D];
    ST* s_head = state + (size_t)h * D * D;
    for (uint32_t i = 0; i < D; ++i) col[i] = pd_dns_ld(s_head + (size_t)i * D + j);

    for (uint32_t t = 0; t < n_tokens; ++t) {
        const size_t base = ((size_t)t * n_heads + h) * (size_t)D;
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

        const float g_t = expf(g[(size_t)t * n_heads + h]);
        const float beta_t = beta[(size_t)t * n_heads + h];

        float u = 0.0f;
        for (uint32_t i = 0; i < D; ++i) { col[i] *= g_t; u += col[i] * k_sh[i]; }
        const float delta = beta_t * (vj - u);
        float o = 0.0f;
        for (uint32_t i = 0; i < D; ++i) { col[i] += k_sh[i] * delta; o += col[i] * q_sh[i]; }
        out[base + j] = o;

        // per-token snapshot of this thread's state column
        ST* sn = snap + (((size_t)t * n_heads + h) * (size_t)D) * D;
        for (uint32_t i = 0; i < D; ++i) pd_dns_st(sn + (size_t)i * D + j, col[i]);
        __syncthreads();
    }

    for (uint32_t i = 0; i < D; ++i) pd_dns_st(s_head + (size_t)i * D + j, col[i]);
}

PD_EXPORT
int pd_gated_delta_recurrent_snap(const void* q, const void* k, const void* v, const void* g,
                                  const void* beta, void* state, void* out, void* snap,
                                  uint32_t n_tokens, uint32_t n_heads, uint32_t head_dim,
                                  void* stream) {
    if (n_tokens == 0 || n_heads == 0 || head_dim == 0) return 0;
    if (head_dim > PD_DN_MAX_D) return cudaErrorInvalidValue;
    size_t shmem = ((size_t)2 * head_dim + 2) * sizeof(float);
    const int dns_cls = pd_dns_state_class();
    if (dns_cls == 3)
        pd_gated_delta_recurrent_snap_kernel<__nv_fp8_e4m3>
            <<<n_heads, head_dim, shmem, (cudaStream_t)stream>>>(
                (const float*)q, (const float*)k, (const float*)v, (const float*)g,
                (const float*)beta, (__nv_fp8_e4m3*)state, (float*)out,
                (__nv_fp8_e4m3*)snap, n_tokens, n_heads, head_dim);
    else if (dns_cls == 2)
        pd_gated_delta_recurrent_snap_kernel<__half>
            <<<n_heads, head_dim, shmem, (cudaStream_t)stream>>>(
                (const float*)q, (const float*)k, (const float*)v, (const float*)g,
                (const float*)beta, (__half*)state, (float*)out,
                (__half*)snap, n_tokens, n_heads, head_dim);
    else if (dns_cls == 1)
        pd_gated_delta_recurrent_snap_kernel<__nv_bfloat16>
            <<<n_heads, head_dim, shmem, (cudaStream_t)stream>>>(
                (const float*)q, (const float*)k, (const float*)v, (const float*)g,
                (const float*)beta, (__nv_bfloat16*)state, (float*)out,
                (__nv_bfloat16*)snap, n_tokens, n_heads, head_dim);
    else
        pd_gated_delta_recurrent_snap_kernel<<<n_heads, head_dim, shmem,
                                               (cudaStream_t)stream>>>(
            (const float*)q, (const float*)k, (const float*)v, (const float*)g,
            (const float*)beta, (float*)state, (float*)out, (float*)snap,
            n_tokens, n_heads, head_dim);
    return pd_launch_status();
}

// Small-batch tiled GEMM over the repacked Q8_0 layout - the speculative-decode
// verify matmul (2..12 rows). The plain per-row kernels re-read the activation
// tile from L2 once per OUTPUT row (out_dim x r x in_dim floats - at r=5 on the
// 27B that's ~0.5 TB/round and the whole spec speedup drowns); here each block
// stages the x chunk in shared once and 8 warps x 2 output rows share it, cutting
// activation traffic 16x while the weight is still read exactly once. f32
// accumulate, scale factored per half-block - same dequant math as the GEMV.
__global__ void __launch_bounds__(256) pd_q8_0_gemm_repacked_mt_kernel(
    const int8_t* __restrict__ data, const __half* __restrict__ scale,
    const float* __restrict__ bias, const float* __restrict__ x, float* __restrict__ y,
    uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
    uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    uint32_t o0 = blockIdx.x * PD_MT_BM + warp * 2;   // this warp's two output rows
    uint32_t n_blocks = in_dim >> 5;
    __shared__ float xsh[PD_MT_ROWS * PD_MT_CHUNK];

    // accumulators must index with compile-time t (unrolled loops below) or the
    // arrays spill to local memory and the cost scales linearly with batch (one
    // full weight-pass-equivalent per row - the exact failure this kernel avoids).
    float acc0[PD_MT_ROWS], acc1[PD_MT_ROWS];
#pragma unroll
    for (uint32_t t = 0; t < PD_MT_ROWS; ++t) { acc0[t] = 0.0f; acc1[t] = 0.0f; }

    const int8_t* row0 = data + (size_t)o0 * in_dim;
    const int8_t* row1 = data + (size_t)(o0 + 1) * in_dim;
    const __half* sc0 = scale + (size_t)o0 * n_blocks;
    const __half* sc1 = scale + (size_t)(o0 + 1) * n_blocks;

    for (uint32_t c0 = 0; c0 < in_dim; c0 += PD_MT_CHUNK) {
        uint32_t clen = in_dim - c0 < PD_MT_CHUNK ? in_dim - c0 : PD_MT_CHUNK;
        // cooperative stage: batch x chunk floats
        for (uint32_t i = tid; i < batch * PD_MT_CHUNK; i += blockDim.x) {
            uint32_t t = i / PD_MT_CHUNK, c = i % PD_MT_CHUNK;
            xsh[i] = (c < clen) ? x[(size_t)t * in_dim + c0 + c] : 0.0f;
        }
        __syncthreads();
        // each lane owns one aligned 16-elem subchunk; streaming weight loads so
        // the one-pass weight flow doesn't evict the x tile from L2
        uint32_t base = c0 + lane * 16u;
        if (o0 < out_dim && base < in_dim) {
            int4 w0 = __ldcs(reinterpret_cast<const int4*>(row0 + base));
            int4 w1 = __ldcs(reinterpret_cast<const int4*>(row1 + base));
            const int8_t* b0 = reinterpret_cast<const int8_t*>(&w0);
            const int8_t* b1 = reinterpret_cast<const int8_t*>(&w1);
            float s0 = __half2float(__ldcs(sc0 + (base >> 5)));
            float s1 = __half2float(__ldcs(sc1 + (base >> 5)));
            const float* xs = xsh + (lane * 16u);
#pragma unroll
            for (uint32_t t = 0; t < PD_MT_ROWS; ++t) {
                if (t >= batch) break;
                const float* xt = xs + (size_t)t * PD_MT_CHUNK;
                float d0 = 0.0f, d1 = 0.0f;
#pragma unroll
                for (uint32_t j = 0; j < 16; ++j) {
                    float xv = xt[j];
                    d0 += (float)b0[j] * xv;
                    d1 += (float)b1[j] * xv;
                }
                acc0[t] += s0 * d0;
                acc1[t] += s1 * d1;
            }
        }
        __syncthreads();
    }
    if (o0 >= out_dim) return;
#pragma unroll
    for (uint32_t t = 0; t < PD_MT_ROWS; ++t) {
        if (t >= batch) break;
        float a0 = acc0[t], a1 = acc1[t];
        for (uint32_t s = 16; s > 0; s >>= 1) {
            a0 += __shfl_down_sync(0xffffffffu, a0, s);
            a1 += __shfl_down_sync(0xffffffffu, a1, s);
        }
        if (lane == 0) {
            y[(size_t)t * out_dim + o0] = a0 + (bias ? bias[o0] : 0.0f);
            if (o0 + 1 < out_dim) {
                y[(size_t)t * out_dim + o0 + 1] = a1 + (bias ? bias[o0 + 1] : 0.0f);
            }
        }
    }
}

PD_EXPORT
int pd_q8_0_gemm_repacked_mt(const void* data, const void* scale, const void* bias,
                             const void* x, void* y, uint32_t in_dim, uint32_t out_dim,
                             uint32_t batch, void* stream) {
    if (out_dim == 0 || batch == 0) return 0;
    if (batch > PD_MT_ROWS) return cudaErrorInvalidValue;
    uint32_t blocks = (out_dim + PD_MT_BM - 1) / PD_MT_BM;
    pd_q8_0_gemm_repacked_mt_kernel<<<blocks, 256, 0, (cudaStream_t)stream>>>(
        (const int8_t*)data, (const __half*)scale, (const float*)bias, (const float*)x,
        (float*)y, in_dim, out_dim, batch);
    return pd_launch_status();
}

// Wide-batch dp4a GEMM: 32 batch rows per weight pass (vs 16 in the MT kernel),
// for B >= 17 where z-tile weight re-reads start to dominate. One output row per
// warp keeps the accumulator file at 32 registers (compile-time indexed - no
// spill); 8 warps/block, shared int8 x staging is 16 KB.
#define PD_MTW_ROWS 32
#define PD_MTW_BM 8
__global__ void __launch_bounds__(256) pd_q8_0_gemm_mt_dp4a_wide_kernel(
    const int8_t* __restrict__ data, const __half* __restrict__ scale,
    const int8_t* __restrict__ xq, const float* __restrict__ xs,
    float* __restrict__ y, uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
    uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    uint32_t o = blockIdx.x * PD_MTW_BM + warp;   // one output row per warp
    uint32_t n_blocks = in_dim >> 5;
    {
        uint32_t b0 = blockIdx.z * PD_MTW_ROWS;
        xq += (size_t)b0 * in_dim;
        xs += (size_t)b0 * n_blocks;
        y += (size_t)b0 * out_dim;
        batch = (batch - b0 < PD_MTW_ROWS) ? (batch - b0) : PD_MTW_ROWS;
    }
    __shared__ int4 xqs[PD_MTW_ROWS * (PD_MT_CHUNK / 16)];
    __shared__ float xss[PD_MTW_ROWS * (PD_MT_CHUNK / 32)];

    float acc[PD_MTW_ROWS];
#pragma unroll
    for (uint32_t t = 0; t < PD_MTW_ROWS; ++t) acc[t] = 0.0f;

    const int8_t* row = data + (size_t)o * in_dim;
    const __half* srow = scale + (size_t)o * n_blocks;

    for (uint32_t c0 = 0; c0 < in_dim; c0 += PD_MT_CHUNK) {
        for (uint32_t i = tid; i < batch * (PD_MT_CHUNK / 16); i += blockDim.x) {
            uint32_t t = i / (PD_MT_CHUNK / 16), kk = i % (PD_MT_CHUNK / 16);
            uint32_t sc0 = c0 + kk * 16u;
            xqs[i] = (sc0 < in_dim)
                ? *reinterpret_cast<const int4*>(xq + (size_t)t * in_dim + sc0)
                : make_int4(0, 0, 0, 0);
        }
        for (uint32_t i = tid; i < batch * (PD_MT_CHUNK / 32); i += blockDim.x) {
            uint32_t t = i / (PD_MT_CHUNK / 32), bb = i % (PD_MT_CHUNK / 32);
            uint32_t blk = (c0 >> 5) + bb;
            xss[i] = (blk < n_blocks) ? xs[(size_t)t * n_blocks + blk] : 0.0f;
        }
        __syncthreads();
        uint32_t base = c0 + lane * 16u;
        if (o < out_dim && base < in_dim) {
            int4 wv = __ldcs(reinterpret_cast<const int4*>(row + base));
            float ws = __half2float(__ldcs(srow + (base >> 5)));
#pragma unroll
            for (uint32_t t = 0; t < PD_MTW_ROWS; ++t) {
                if (t >= batch) break;
                int4 xv = xqs[t * (PD_MT_CHUNK / 16) + lane];
                int s = __dp4a(wv.x, xv.x, 0);
                s = __dp4a(wv.y, xv.y, s);
                s = __dp4a(wv.z, xv.z, s);
                s = __dp4a(wv.w, xv.w, s);
                acc[t] += ws * xss[t * (PD_MT_CHUNK / 32) + ((lane * 16u) >> 5)] * (float)s;
            }
        }
        __syncthreads();
    }
    if (o >= out_dim) return;
#pragma unroll
    for (uint32_t t = 0; t < PD_MTW_ROWS; ++t) {
        if (t >= batch) break;
        float a = acc[t];
        for (uint32_t s2 = 16; s2 > 0; s2 >>= 1) a += __shfl_down_sync(0xffffffffu, a, s2);
        if (lane == 0) y[(size_t)t * out_dim + o] = a;
    }
}

PD_EXPORT
int pd_q8_0_gemm_mt_dp4a_wide(const void* data, const void* scale, const void* xq,
                              const void* xs, void* y, uint32_t in_dim, uint32_t out_dim,
                              uint32_t batch, void* stream) {
    if (out_dim == 0 || batch == 0) return 0;
    dim3 grid((out_dim + PD_MTW_BM - 1) / PD_MTW_BM, 1,
              (batch + PD_MTW_ROWS - 1) / PD_MTW_ROWS);
    pd_q8_0_gemm_mt_dp4a_wide_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(
        (const int8_t*)data, (const __half*)scale, (const int8_t*)xq, (const float*)xs,
        (float*)y, in_dim, out_dim, batch);
    return pd_launch_status();
}

// DeltaNet split+GQA with fused q/k L2-normalization - feeds the v2 recurrence,
// which takes q,k PRE-normalized (llama.cpp does the same: ggml_l2_norm sits
// upstream of its fused gated-delta-net kernel). One block per output row
// (row, hv), s threads; block-reduces sum(q^2)/sum(k^2) with warp shuffles and
// writes q_hat * (1/sqrt s), k_hat, v. Same GQA tiling as pd_deltanet_split_gqa
// (v-head hv reads key head hv % n_k_heads, llama ggml_repeat_4d convention).
__global__ void pd_deltanet_split_gqa_norm_kernel(
        const float* __restrict__ conv, float* __restrict__ q_out,
        float* __restrict__ k_out, float* __restrict__ v_out,
        uint32_t n_k_heads, uint32_t n_v_heads, uint32_t s) {
    const uint32_t hv = blockIdx.x;
    const uint32_t row = blockIdx.y;
    const uint32_t j = threadIdx.x;
    if (j >= s) return;
    const uint32_t hk = hv % n_k_heads;
    const uint32_t key_dim = s * n_k_heads;
    const uint32_t conv_dim = 2u * key_dim + s * n_v_heads;
    const size_t base = (size_t)row * conv_dim;
    const float qj = conv[base + (size_t)hk * s + j];
    const float kj = conv[base + key_dim + (size_t)hk * s + j];
    const float vj = conv[base + 2u * key_dim + (size_t)hv * s + j];

    float q2 = qj * qj, k2 = kj * kj;
#pragma unroll
    for (uint32_t off = 16; off > 0; off >>= 1) {
        q2 += __shfl_xor_sync(0xffffffffu, q2, off);
        k2 += __shfl_xor_sync(0xffffffffu, k2, off);
    }
    __shared__ float sh[8];                       // up to 4 warps x {q2, k2}
    const uint32_t lane = j & 31u, warp = j >> 5, nwarps = (s + 31u) >> 5;
    if (lane == 0) { sh[warp] = q2; sh[4 + warp] = k2; }
    __syncthreads();
    float qs = 0.0f, ks = 0.0f;
    for (uint32_t w = 0; w < nwarps; ++w) { qs += sh[w]; ks += sh[4 + w]; }

    const size_t oidx = ((size_t)row * n_v_heads + hv) * s + j;
    q_out[oidx] = qj * rsqrtf(qs + 1e-6f) * rsqrtf((float)s);
    k_out[oidx] = kj * rsqrtf(ks + 1e-6f);
    v_out[oidx] = vj;
}

PD_EXPORT
int pd_deltanet_split_gqa_norm(const void* conv, void* q_out, void* k_out, void* v_out,
                               uint32_t n_rows, uint32_t n_k_heads, uint32_t n_v_heads,
                               uint32_t s, void* stream) {
    if (n_rows == 0 || n_v_heads == 0 || s == 0) return 0;
    if (s > 128 || (s & 31u)) return cudaErrorInvalidValue;
    dim3 grid(n_v_heads, n_rows);
    pd_deltanet_split_gqa_norm_kernel<<<grid, s, 0, (cudaStream_t)stream>>>(
        (const float*)conv, (float*)q_out, (float*)k_out, (float*)v_out,
        n_k_heads, n_v_heads, s);
    return pd_launch_status();
}

// Gated delta recurrence v2 - llama-shape rewrite (cf. llama.cpp's gated_delta_net.cu):
// warp-per-STATE-COLUMN instead of block-per-head. grid (H, B, D/4), block
// (32, 4); each warp owns column `col`, each lane holds a float4 shard of it
// (S[i][col] for i = 4*lane..4*lane+3) in REGISTERS for the whole token loop -
// no local-memory spill (compile-time D), no block-wide syncs (warp shuffles
// only), and 32x the block count of the v1 kernels. State is stored TRANSPOSED
// relative to v1: column col is contiguous (s_head[col*D + i]) so lane loads
// coalesce; snapshots use the same transposed tile layout and rollback copies
// whole tiles, so callers are unaffected. One body serves all four uses:
//   slots == NULL: seq b uses state slot b        (single decode / prefill)
//   slots != NULL: seq b uses states[slots[b]]    (continuous batching)
//   snap  != NULL: per-token state snapshots, t-major (speculative rollback)
//   n_tokens > 1:  chunk loop, state stays in registers across the chunk.
// q,k arrive PRE-normalized (pd_deltanet_split_gqa_norm; q carries 1/sqrt(D)).
// q/k/v/out are [B, T, H, D]; g/beta [B, T, H]; g is log-decay (exp'd here).
// Math order matches v1/the CPU reference: decay, dot k, rank-1 update, dot q.
#define PD_DN2_D 128
#define PD_DN2_WARPS 4
// Occupancy hint for the v2 recurrence family. Profiled on B200 (live
// serve, b=32): DRAM throughput 18.25%, Mem Busy 68.21%, L1 hit 63.09% - the
// kernel is not DRAM-bound here, it is SM-memory-pipeline bound, and achieved
// occupancy is 64.47% against a theoretical 75% with "Block Limit Registers"
// = 12 at 40 registers/thread. (The pack's older "state band ~1.67 TB/s, i.e.
// at roof" note is the TWIN's GDDR7 die and does not transfer to HBM3e.)
// A minBlocksPerMultiprocessor hint of 16 forces R <= 32 and lifts the
// register block limit to 16 (theoretical occupancy 75% -> 100%).
//
// SHIPPED 16. cuobjdump on the serving instantiation
// (<__half, false>): REG 40 -> 32, STACK 0, LOCAL 0 - the compiler hits the
// budget with no spill. The change is bit-identical by construction: a launch
// bound steers register allocation, never a value.
//
// A serve A/B (alternating boots per arm, both arms built from the same
// source with only this define differing) measured a small but consistent
// throughput win, which is what elected 16.
// PD_DEFS=-DPD_DN2_MINB=0 reverts to the old codegen.
//
// The 16 is a B200 number: 16 blocks x 128 threads is exactly that die's
// 2048 resident threads. On a 1536-thread SM (sm_86/89/110, every 12.x die)
// the same 16 asks for 2048 and ptxas does not clamp it, it DROPS the hint
// (".minnctapersm will be ignored", 22 instantiations x 6 targets) - so
// those dies never had a hint at all. Hint only where it was elected:
// PD_SM_MAX_THREADS >= 2048 keeps the 16 on B200/H100/A100 and passes no
// minBlocks on the 1536-thread dies, which is byte-identical SASS to what
// they ran before (per-kernel hash, 2026-09-06) with the warning gone. The
// honest 1536-die hint would be 12 (R <= 42; the family sits at 40) - built
// and hashed the same day: REG stays 40 but ptxas reschedules (v2<half>
// 912 -> 896 instructions on sm_120a). Dropping the hint reschedules too
// (18 of the 22 instantiations differ from the dropped-16 build), so all
// three forms were served back to back on the RTX PRO 6000, qwen3.8-27b Q8,
// 3 reps: dropped-16 c1 46.77 / c32 1016.85, none 46.78 / 1016.99, 12 46.79
// / 1015.79 tok/s - a three-way tie inside 0.1%, i.e. the hint is inert on
// this die (the kernel is SM-pipeline bound at 12 blocks either way). None
// is the shipped form: no hint to be wrong about; 12 stays a probe arm.
#ifndef PD_DN2_MINB
#define PD_DN2_MINB (PD_SM_MAX_THREADS >= 2048 ? 16 : 0)
#endif
#if PD_DN2_MINB > 0
#define PD_DN2_LB __launch_bounds__(32 * PD_DN2_WARPS, PD_DN2_MINB)
#else
#define PD_DN2_LB __launch_bounds__(32 * PD_DN2_WARPS)
#endif
// CS: streaming (evict-first) state I/O - see pd_dns_ld4_cs (abi.cuh). The
// decode tick's state walk dirtied ~100MB/layer of L2 whose writebacks
// throttled the FOLLOWING GEMMs' reads (gu/dnout inflate +21/+34% in-tick,
// dnig before the recurrence at parity). Values identical; cache-op only.
// PADDOCK_NO_DNS_CS reverts at launch.
template <typename ST = float, bool CS = false>
__global__ void PD_DN2_LB
pd_gated_delta_recurrent_v2_kernel(
        const float* __restrict__ q, const float* __restrict__ k,
        const float* __restrict__ v, const float* __restrict__ g,
        const float* __restrict__ beta, const unsigned int* __restrict__ slots,
        ST* __restrict__ states, ST* __restrict__ snap,
        float* __restrict__ out, uint32_t batch, uint32_t n_tokens,
        uint32_t n_heads) {
    constexpr uint32_t D = PD_DN2_D;
    const uint32_t h = blockIdx.x;
    const uint32_t b = blockIdx.y;
    const uint32_t lane = threadIdx.x;
    // Two adjacent columns per warp (P6h experiment): interleaved u/o
    // reduction chains + amortized k/q/g/beta loads and expf.
    const uint32_t col = (blockIdx.z * PD_DN2_WARPS + threadIdx.y) * 2u;

    const uint32_t slot = slots ? slots[b] : b;
    ST* s_head = states + ((size_t)slot * n_heads + h) * (size_t)D * D
                        + (size_t)col * D;

    float4 sa = CS ? pd_dns_ld4_cs(s_head + lane * 4u)
                   : pd_dns_ld4(s_head + lane * 4u);
    float4 sb = CS ? pd_dns_ld4_cs(s_head + D + lane * 4u)
                   : pd_dns_ld4(s_head + D + lane * 4u);

    for (uint32_t t = 0; t < n_tokens; ++t) {
        const size_t base = (((size_t)b * n_tokens + t) * n_heads + h) * (size_t)D;
        const float4 k4 = *reinterpret_cast<const float4*>(k + base + lane * 4u);
        const float4 q4 = *reinterpret_cast<const float4*>(q + base + lane * 4u);
        const float va = v[base + col];
        const float vb = v[base + col + 1u];
        const size_t gb = ((size_t)b * n_tokens + t) * n_heads + h;
        const float g_t = expf(g[gb]);
        const float beta_t = beta[gb];

        // decay the column shards, dot with k_hat (v1 / CPU-reference order)
        sa.x *= g_t; sa.y *= g_t; sa.z *= g_t; sa.w *= g_t;
        sb.x *= g_t; sb.y *= g_t; sb.z *= g_t; sb.w *= g_t;
        float ua = sa.x * k4.x + sa.y * k4.y + sa.z * k4.z + sa.w * k4.w;
        float ub = sb.x * k4.x + sb.y * k4.y + sb.z * k4.z + sb.w * k4.w;
#pragma unroll
        for (uint32_t off = 16; off > 0; off >>= 1) {
            ua += __shfl_xor_sync(0xffffffffu, ua, off);
            ub += __shfl_xor_sync(0xffffffffu, ub, off);
        }
        const float da = beta_t * (va - ua);
        const float db = beta_t * (vb - ub);

        // rank-1 update, then dot with q_hat (already carries 1/sqrt D)
        sa.x += k4.x * da; sa.y += k4.y * da; sa.z += k4.z * da; sa.w += k4.w * da;
        sb.x += k4.x * db; sb.y += k4.y * db; sb.z += k4.z * db; sb.w += k4.w * db;
        float oa = sa.x * q4.x + sa.y * q4.y + sa.z * q4.z + sa.w * q4.w;
        float ob = sb.x * q4.x + sb.y * q4.y + sb.z * q4.z + sb.w * q4.w;
#pragma unroll
        for (uint32_t off = 16; off > 0; off >>= 1) {
            oa += __shfl_xor_sync(0xffffffffu, oa, off);
            ob += __shfl_xor_sync(0xffffffffu, ob, off);
        }
        if (lane == 0) {
            out[base + col] = oa;
            out[base + col + 1u] = ob;
        }

        if (snap) {
            ST* sn = snap + (((size_t)b * n_tokens + t) * n_heads + h)
                          * (size_t)D * D + (size_t)col * D;
            pd_dns_st4(sn + lane * 4u, sa);
            pd_dns_st4(sn + D + lane * 4u, sb);
        }
    }

    if (CS) {
        pd_dns_st4_cs(s_head + lane * 4u, sa);
        pd_dns_st4_cs(s_head + D + lane * 4u, sb);
    } else {
        pd_dns_st4(s_head + lane * 4u, sa);
        pd_dns_st4(s_head + D + lane * 4u, sb);
    }
}


// ── v2w: widened recurrence probe ────────────────────────────────────────
// 16 lanes per state column, four columns per warp, grid.z halved vs v2.
// Why: v2 profiles as SM-memory-pipeline bound (Mem Busy 68%, DRAM 18%),
// and v2's in-loop MIO traffic per 4 columns per token is 40 shfl + 4
// state-entry 8B lds + 4 uniform v lds + 2x(g,beta,expf). v2w makes the
// state entry/exit 16B transactions that each span a column PAIR
// (transposed layout: columns are contiguous), cuts the shuffle tree to
// 4 levels over 16-lane groups (16 shfl per 4 cols per token), and
// amortizes v/g/beta/expf over 4 columns. k/q issue slots are a wash
// (2x ld.128 per lane vs 1, but half the warps).
// Not bit-identical to v2: the dot's summation grouping changes
// (8-element serial + 4-level tree vs 4-element serial + 5-level tree).
// Same f32 ops, different rounding order - gate like a numerics change
// (maxrel vs an f64 reference + interleaved serve legs), never
// by a bit-gate. Opt-in PADDOCK_DN_V2W=1; f16 states (dns_cls 2) only.
__device__ __forceinline__ void pd_dn2w_ld8(const __half* p, float4& lo, float4& hi) {
    const uint4 r = *reinterpret_cast<const uint4*>(p);
    const __half2 a = *reinterpret_cast<const __half2*>(&r.x);
    const __half2 b = *reinterpret_cast<const __half2*>(&r.y);
    const __half2 c = *reinterpret_cast<const __half2*>(&r.z);
    const __half2 d = *reinterpret_cast<const __half2*>(&r.w);
    lo = make_float4(__half2float(a.x), __half2float(a.y), __half2float(b.x), __half2float(b.y));
    hi = make_float4(__half2float(c.x), __half2float(c.y), __half2float(d.x), __half2float(d.y));
}
__device__ __forceinline__ void pd_dn2w_ld8_cs(const __half* p, float4& lo, float4& hi) {
    uint4 r;
    asm volatile("ld.global.cs.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    const __half2 a = *reinterpret_cast<const __half2*>(&r.x);
    const __half2 b = *reinterpret_cast<const __half2*>(&r.y);
    const __half2 c = *reinterpret_cast<const __half2*>(&r.z);
    const __half2 d = *reinterpret_cast<const __half2*>(&r.w);
    lo = make_float4(__half2float(a.x), __half2float(a.y), __half2float(b.x), __half2float(b.y));
    hi = make_float4(__half2float(c.x), __half2float(c.y), __half2float(d.x), __half2float(d.y));
}
__device__ __forceinline__ void pd_dn2w_st8(__half* p, float4 lo, float4 hi) {
    uint4 r;
    *reinterpret_cast<__half2*>(&r.x) = __floats2half2_rn(lo.x, lo.y);
    *reinterpret_cast<__half2*>(&r.y) = __floats2half2_rn(lo.z, lo.w);
    *reinterpret_cast<__half2*>(&r.z) = __floats2half2_rn(hi.x, hi.y);
    *reinterpret_cast<__half2*>(&r.w) = __floats2half2_rn(hi.z, hi.w);
    *reinterpret_cast<uint4*>(p) = r;
}
__device__ __forceinline__ void pd_dn2w_st8_cs(__half* p, float4 lo, float4 hi) {
    uint4 r;
    *reinterpret_cast<__half2*>(&r.x) = __floats2half2_rn(lo.x, lo.y);
    *reinterpret_cast<__half2*>(&r.y) = __floats2half2_rn(lo.z, lo.w);
    *reinterpret_cast<__half2*>(&r.z) = __floats2half2_rn(hi.x, hi.y);
    *reinterpret_cast<__half2*>(&r.w) = __floats2half2_rn(hi.z, hi.w);
    asm volatile("st.global.cs.v4.b32 [%0], {%1,%2,%3,%4};"
                 :: "l"(p), "r"(r.x), "r"(r.y), "r"(r.z), "r"(r.w) : "memory");
}
// v2w register budget: 16 f32 state regs + 16 k/q temporaries land at 48
// regs; forcing occupancy SPILLS the STATE and loses (bench sweep, B=32 T=1:
// default/48reg 20.51 us; MINB=10/48reg 20.48; MINB=12/40reg+16B stack
// 22.53; MINB=14/32reg+80B stack 32.83). The v2 launch-bounds lesson
// INVERTS here - occupancy is not the constraint the widened form trades
// against. Default 0 = compiler's choice, which is the measured optimum.
#ifndef PD_DN2W_MINB
#define PD_DN2W_MINB 0
#endif
#if PD_DN2W_MINB > 0
#define PD_DN2W_LB __launch_bounds__(32 * PD_DN2_WARPS, PD_DN2W_MINB)
#else
#define PD_DN2W_LB __launch_bounds__(32 * PD_DN2_WARPS)
#endif
template <bool CS = false>
__global__ void PD_DN2W_LB
pd_gated_delta_recurrent_v2w_kernel(
        const float* __restrict__ q, const float* __restrict__ k,
        const float* __restrict__ v, const float* __restrict__ g,
        const float* __restrict__ beta, const unsigned int* __restrict__ slots,
        __half* __restrict__ states, __half* __restrict__ snap,
        float* __restrict__ out, uint32_t batch, uint32_t n_tokens,
        uint32_t n_heads) {
    constexpr uint32_t D = PD_DN2_D;
    const uint32_t h = blockIdx.x;
    const uint32_t b = blockIdx.y;
    const uint32_t lane = threadIdx.x;
    const uint32_t colq = (blockIdx.z * PD_DN2_WARPS + threadIdx.y) * 4u;
    const uint32_t l16 = lane & 15u;   // row shard: rows 8*l16 .. 8*l16+7
    const uint32_t hw = lane >> 4;     // 0: cols colq+0/+2   1: cols colq+1/+3

    const uint32_t slot = slots ? slots[b] : b;
    __half* s_head = states + ((size_t)slot * n_heads + h) * (size_t)D * D
                            + (size_t)colq * D;

    // sa = my shard of column colq+hw, sb = of column colq+2+hw
    float4 sa0, sa1, sb0, sb1;
    if (CS) { pd_dn2w_ld8_cs(s_head + lane * 8u, sa0, sa1);
              pd_dn2w_ld8_cs(s_head + 2u * D + lane * 8u, sb0, sb1); }
    else    { pd_dn2w_ld8(s_head + lane * 8u, sa0, sa1);
              pd_dn2w_ld8(s_head + 2u * D + lane * 8u, sb0, sb1); }

    for (uint32_t t = 0; t < n_tokens; ++t) {
        const size_t base = (((size_t)b * n_tokens + t) * n_heads + h) * (size_t)D;
        const float4 k0 = *reinterpret_cast<const float4*>(k + base + l16 * 8u);
        const float4 k1 = *reinterpret_cast<const float4*>(k + base + l16 * 8u + 4u);
        const float4 q0 = *reinterpret_cast<const float4*>(q + base + l16 * 8u);
        const float4 q1 = *reinterpret_cast<const float4*>(q + base + l16 * 8u + 4u);
        const float4 vq = *reinterpret_cast<const float4*>(v + base + colq);
        const float va = hw ? vq.y : vq.x;
        const float vb = hw ? vq.w : vq.z;
        const size_t gb = ((size_t)b * n_tokens + t) * n_heads + h;
        const float g_t = expf(g[gb]);
        const float beta_t = beta[gb];

        sa0.x *= g_t; sa0.y *= g_t; sa0.z *= g_t; sa0.w *= g_t;
        sa1.x *= g_t; sa1.y *= g_t; sa1.z *= g_t; sa1.w *= g_t;
        sb0.x *= g_t; sb0.y *= g_t; sb0.z *= g_t; sb0.w *= g_t;
        sb1.x *= g_t; sb1.y *= g_t; sb1.z *= g_t; sb1.w *= g_t;
        float ua = sa0.x * k0.x + sa0.y * k0.y + sa0.z * k0.z + sa0.w * k0.w
                 + sa1.x * k1.x + sa1.y * k1.y + sa1.z * k1.z + sa1.w * k1.w;
        float ub = sb0.x * k0.x + sb0.y * k0.y + sb0.z * k0.z + sb0.w * k0.w
                 + sb1.x * k1.x + sb1.y * k1.y + sb1.z * k1.z + sb1.w * k1.w;
#pragma unroll
        for (uint32_t off = 8; off > 0; off >>= 1) {
            ua += __shfl_xor_sync(0xffffffffu, ua, off);
            ub += __shfl_xor_sync(0xffffffffu, ub, off);
        }
        const float da = beta_t * (va - ua);
        const float db = beta_t * (vb - ub);

        sa0.x += k0.x * da; sa0.y += k0.y * da; sa0.z += k0.z * da; sa0.w += k0.w * da;
        sa1.x += k1.x * da; sa1.y += k1.y * da; sa1.z += k1.z * da; sa1.w += k1.w * da;
        sb0.x += k0.x * db; sb0.y += k0.y * db; sb0.z += k0.z * db; sb0.w += k0.w * db;
        sb1.x += k1.x * db; sb1.y += k1.y * db; sb1.z += k1.z * db; sb1.w += k1.w * db;
        float oa = sa0.x * q0.x + sa0.y * q0.y + sa0.z * q0.z + sa0.w * q0.w
                 + sa1.x * q1.x + sa1.y * q1.y + sa1.z * q1.z + sa1.w * q1.w;
        float ob = sb0.x * q0.x + sb0.y * q0.y + sb0.z * q0.z + sb0.w * q0.w
                 + sb1.x * q1.x + sb1.y * q1.y + sb1.z * q1.z + sb1.w * q1.w;
#pragma unroll
        for (uint32_t off = 8; off > 0; off >>= 1) {
            oa += __shfl_xor_sync(0xffffffffu, oa, off);
            ob += __shfl_xor_sync(0xffffffffu, ob, off);
        }
        if (l16 == 0) {
            out[base + colq + hw] = oa;
            out[base + colq + 2u + hw] = ob;
        }

        if (snap) {
            __half* sn = snap + (((size_t)b * n_tokens + t) * n_heads + h)
                              * (size_t)D * D + (size_t)colq * D;
            pd_dn2w_st8(sn + lane * 8u, sa0, sa1);
            pd_dn2w_st8(sn + 2u * D + lane * 8u, sb0, sb1);
        }
    }

    if (CS) {
        pd_dn2w_st8_cs(s_head + lane * 8u, sa0, sa1);
        pd_dn2w_st8_cs(s_head + 2u * D + lane * 8u, sb0, sb1);
    } else {
        pd_dn2w_st8(s_head + lane * 8u, sa0, sa1);
        pd_dn2w_st8(s_head + 2u * D + lane * 8u, sb0, sb1);
    }
}

PD_EXPORT
int pd_gated_delta_recurrent_v2(const void* q, const void* k, const void* v,
                                const void* g, const void* beta, const void* slots,
                                void* states, void* snap, void* out, uint32_t batch,
                                uint32_t n_tokens, uint32_t n_heads, uint32_t head_dim,
                                void* stream) {
    if (batch == 0 || n_tokens == 0 || n_heads == 0) return 0;
    if (head_dim != PD_DN2_D) return cudaErrorInvalidValue;
    dim3 grid(n_heads, batch, PD_DN2_D / (2u * PD_DN2_WARPS));  // 2 cols/warp
    dim3 block(32, PD_DN2_WARPS);
    // streaming-state arm: FALSIFIED at serve - indistinguishable from
    // control, in-noise on throughput and ITL alike. The state writes
    // must reach DRAM either way; the followers' in-tick inflation is the
    // recurrence's writeback DRAIN attributed to them, not recoverable
    // waste - the state band runs ~1.67 TB/s including the absorbed drain,
    // i.e. at roof. Kept opt-in (PADDOCK_DNS_CS=1) for re-probes.
    static const bool cs = pd_env("PADDOCK_DNS_CS") != nullptr;
    const int dns_cls = pd_dns_state_class();
    if (dns_cls == 3) {
        pd_gated_delta_recurrent_v2_kernel<__nv_fp8_e4m3>
            <<<grid, block, 0, (cudaStream_t)stream>>>(
                (const float*)q, (const float*)k, (const float*)v, (const float*)g,
                (const float*)beta, (const unsigned int*)slots, (__nv_fp8_e4m3*)states,
                (__nv_fp8_e4m3*)snap, (float*)out, batch, n_tokens, n_heads);
    } else if (dns_cls == 2) {
        // v2w election: DEFAULT on on sm_100, where a serve A/B (alternating
        // boots per arm, same pack) came out ahead with non-overlapping reps.
        // Kernel scope: 22.53 -> 20.51 us at B=32 T=1, correctness in the
        // same f16-rounding band as v2 against an f64 reference. sm_120 stays
        // on v2 by default - v2w is UNMEASURED on that die and rung elections
        // do not port; force with PADDOCK_DN_V2W, kill with PADDOCK_NO_DN_V2W.
        static const bool v2w = [] {
            if (pd_env("PADDOCK_NO_DN_V2W")) return false;
            if (pd_env("PADDOCK_DN_V2W")) return true;
            int dev = 0, major = 0;
            cudaGetDevice(&dev);
            cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
            return major == 10;
        }();
        if (v2w) {
            dim3 gridw(n_heads, batch, PD_DN2_D / (4u * PD_DN2_WARPS));
            if (cs)
                pd_gated_delta_recurrent_v2w_kernel<true>
                    <<<gridw, block, 0, (cudaStream_t)stream>>>(
                        (const float*)q, (const float*)k, (const float*)v, (const float*)g,
                        (const float*)beta, (const unsigned int*)slots, (__half*)states,
                        (__half*)snap, (float*)out, batch, n_tokens, n_heads);
            else
                pd_gated_delta_recurrent_v2w_kernel<false>
                    <<<gridw, block, 0, (cudaStream_t)stream>>>(
                        (const float*)q, (const float*)k, (const float*)v, (const float*)g,
                        (const float*)beta, (const unsigned int*)slots, (__half*)states,
                        (__half*)snap, (float*)out, batch, n_tokens, n_heads);
        } else if (cs)
            pd_gated_delta_recurrent_v2_kernel<__half, true>
                <<<grid, block, 0, (cudaStream_t)stream>>>(
                    (const float*)q, (const float*)k, (const float*)v, (const float*)g,
                    (const float*)beta, (const unsigned int*)slots, (__half*)states,
                    (__half*)snap, (float*)out, batch, n_tokens, n_heads);
        else
            pd_gated_delta_recurrent_v2_kernel<__half>
                <<<grid, block, 0, (cudaStream_t)stream>>>(
                    (const float*)q, (const float*)k, (const float*)v, (const float*)g,
                    (const float*)beta, (const unsigned int*)slots, (__half*)states,
                    (__half*)snap, (float*)out, batch, n_tokens, n_heads);
    } else if (dns_cls == 1) {
        if (cs)
            pd_gated_delta_recurrent_v2_kernel<__nv_bfloat16, true>
                <<<grid, block, 0, (cudaStream_t)stream>>>(
                    (const float*)q, (const float*)k, (const float*)v, (const float*)g,
                    (const float*)beta, (const unsigned int*)slots, (__nv_bfloat16*)states,
                    (__nv_bfloat16*)snap, (float*)out, batch, n_tokens, n_heads);
        else
            pd_gated_delta_recurrent_v2_kernel<__nv_bfloat16>
                <<<grid, block, 0, (cudaStream_t)stream>>>(
                    (const float*)q, (const float*)k, (const float*)v, (const float*)g,
                    (const float*)beta, (const unsigned int*)slots, (__nv_bfloat16*)states,
                    (__nv_bfloat16*)snap, (float*)out, batch, n_tokens, n_heads);
    } else if (cs) {
        pd_gated_delta_recurrent_v2_kernel<float, true>
            <<<grid, block, 0, (cudaStream_t)stream>>>(
            (const float*)q, (const float*)k, (const float*)v, (const float*)g,
            (const float*)beta, (const unsigned int*)slots, (float*)states, (float*)snap,
            (float*)out, batch, n_tokens, n_heads);
    } else {
        pd_gated_delta_recurrent_v2_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
            (const float*)q, (const float*)k, (const float*)v, (const float*)g,
            (const float*)beta, (const unsigned int*)slots, (float*)states, (float*)snap,
            (float*)out, batch, n_tokens, n_heads);
    }
    return pd_launch_status();
}

// ── Snapshot-free spec verify ──────────────────────────────────────
// The spec verify used v2 in snap mode: advance the live state through all
// k1 drafts, write a per-token state snapshot (b x k1 x H x D x D), and let
// the commit ROLL BACK to the accepted position's snapshot
// (pd_state_restore_slots). That is the snapshot-per-step shape whose write
// traffic swamps the tick at c32, and the b x k1 snapshot allocation is ~87%
// of the engine's ~1.15 GiB/spec-row draft state - the hard cap at 14 spec
// rows on the 96 GB card. The replacement pair:
//   v2h  - the verify twin: v2 without the final state writeback (state
//          stays at round-start in the live buffer) and without snapshots.
//          Same math, same out[] values, bit-identical per-token results.
//   commit_walk - at commit, re-run the recurrence from the round-start
//          state over each row's ACCEPTED prefix only (committed[b] from
//          the device-staged commit buffer - capture-safe), then write the
//          state back once. Same fixed op order as v2 on the same stashed
//          split/gate planes, so the final state is bit-exact vs the
//          snapshot the old path would have restored. q is not read (it
//          only ever fed out[]); committed[b] == 0 writes nothing.
// State traffic per row per round: O(1) read + O(1) write vs O(k1) writes.
template <typename ST = float, bool CS = false>
__global__ void PD_DN2_LB
pd_gated_delta_verify_hold_kernel(
        const float* __restrict__ q, const float* __restrict__ k,
        const float* __restrict__ v, const float* __restrict__ g,
        const float* __restrict__ beta, const unsigned int* __restrict__ slots,
        const ST* __restrict__ states, float* __restrict__ out, uint32_t batch,
        uint32_t n_tokens, uint32_t n_heads) {
    constexpr uint32_t D = PD_DN2_D;
    const uint32_t h = blockIdx.x;
    const uint32_t b = blockIdx.y;
    const uint32_t lane = threadIdx.x;
    const uint32_t col = (blockIdx.z * PD_DN2_WARPS + threadIdx.y) * 2u;

    const uint32_t slot = slots ? slots[b] : b;
    const ST* s_head = states + ((size_t)slot * n_heads + h) * (size_t)D * D
                              + (size_t)col * D;

    float4 sa = CS ? pd_dns_ld4_cs(s_head + lane * 4u)
                   : pd_dns_ld4(s_head + lane * 4u);
    float4 sb = CS ? pd_dns_ld4_cs(s_head + D + lane * 4u)
                   : pd_dns_ld4(s_head + D + lane * 4u);

    for (uint32_t t = 0; t < n_tokens; ++t) {
        const size_t base = (((size_t)b * n_tokens + t) * n_heads + h) * (size_t)D;
        const float4 k4 = *reinterpret_cast<const float4*>(k + base + lane * 4u);
        const float4 q4 = *reinterpret_cast<const float4*>(q + base + lane * 4u);
        const float va = v[base + col];
        const float vb = v[base + col + 1u];
        const size_t gb = ((size_t)b * n_tokens + t) * n_heads + h;
        const float g_t = expf(g[gb]);
        const float beta_t = beta[gb];

        sa.x *= g_t; sa.y *= g_t; sa.z *= g_t; sa.w *= g_t;
        sb.x *= g_t; sb.y *= g_t; sb.z *= g_t; sb.w *= g_t;
        float ua = sa.x * k4.x + sa.y * k4.y + sa.z * k4.z + sa.w * k4.w;
        float ub = sb.x * k4.x + sb.y * k4.y + sb.z * k4.z + sb.w * k4.w;
#pragma unroll
        for (uint32_t off = 16; off > 0; off >>= 1) {
            ua += __shfl_xor_sync(0xffffffffu, ua, off);
            ub += __shfl_xor_sync(0xffffffffu, ub, off);
        }
        const float da = beta_t * (va - ua);
        const float db = beta_t * (vb - ub);

        sa.x += k4.x * da; sa.y += k4.y * da; sa.z += k4.z * da; sa.w += k4.w * da;
        sb.x += k4.x * db; sb.y += k4.y * db; sb.z += k4.z * db; sb.w += k4.w * db;
        float oa = sa.x * q4.x + sa.y * q4.y + sa.z * q4.z + sa.w * q4.w;
        float ob = sb.x * q4.x + sb.y * q4.y + sb.z * q4.z + sb.w * q4.w;
#pragma unroll
        for (uint32_t off = 16; off > 0; off >>= 1) {
            oa += __shfl_xor_sync(0xffffffffu, oa, off);
            ob += __shfl_xor_sync(0xffffffffu, ob, off);
        }
        if (lane == 0) {
            out[base + col] = oa;
            out[base + col + 1u] = ob;
        }
    }
    // no state writeback, no snapshots - the live state stays at round-start
}

template <typename ST = float>
__global__ void PD_DN2_LB
pd_gated_delta_commit_walk_kernel(
        const float* __restrict__ k, const float* __restrict__ v,
        const float* __restrict__ g, const float* __restrict__ beta,
        const unsigned int* __restrict__ slots,
        const unsigned int* __restrict__ committed, ST* __restrict__ states,
        uint32_t n_tokens, uint32_t n_heads) {
    constexpr uint32_t D = PD_DN2_D;
    const uint32_t h = blockIdx.x;
    const uint32_t b = blockIdx.y;
    const uint32_t lane = threadIdx.x;
    const uint32_t col = (blockIdx.z * PD_DN2_WARPS + threadIdx.y) * 2u;

    const uint32_t take = min(committed[b], n_tokens);
    if (take == 0) return;                        // round declined: state untouched
    const uint32_t slot = slots ? slots[b] : b;
    ST* s_head = states + ((size_t)slot * n_heads + h) * (size_t)D * D
                        + (size_t)col * D;

    float4 sa = pd_dns_ld4(s_head + lane * 4u);
    float4 sb = pd_dns_ld4(s_head + D + lane * 4u);

    for (uint32_t t = 0; t < take; ++t) {
        const size_t base = (((size_t)b * n_tokens + t) * n_heads + h) * (size_t)D;
        const float4 k4 = *reinterpret_cast<const float4*>(k + base + lane * 4u);
        const float va = v[base + col];
        const float vb = v[base + col + 1u];
        const size_t gb = ((size_t)b * n_tokens + t) * n_heads + h;
        const float g_t = expf(g[gb]);
        const float beta_t = beta[gb];

        sa.x *= g_t; sa.y *= g_t; sa.z *= g_t; sa.w *= g_t;
        sb.x *= g_t; sb.y *= g_t; sb.z *= g_t; sb.w *= g_t;
        float ua = sa.x * k4.x + sa.y * k4.y + sa.z * k4.z + sa.w * k4.w;
        float ub = sb.x * k4.x + sb.y * k4.y + sb.z * k4.z + sb.w * k4.w;
#pragma unroll
        for (uint32_t off = 16; off > 0; off >>= 1) {
            ua += __shfl_xor_sync(0xffffffffu, ua, off);
            ub += __shfl_xor_sync(0xffffffffu, ub, off);
        }
        const float da = beta_t * (va - ua);
        const float db = beta_t * (vb - ub);
        sa.x += k4.x * da; sa.y += k4.y * da; sa.z += k4.z * da; sa.w += k4.w * da;
        sb.x += k4.x * db; sb.y += k4.y * db; sb.z += k4.z * db; sb.w += k4.w * db;
    }

    pd_dns_st4(s_head + lane * 4u, sa);
    pd_dns_st4(s_head + D + lane * 4u, sb);
}

PD_EXPORT
int pd_gated_delta_verify_hold(const void* q, const void* k, const void* v,
                               const void* g, const void* beta, const void* slots,
                               const void* states, void* out, uint32_t batch,
                               uint32_t n_tokens, uint32_t n_heads,
                               uint32_t head_dim, void* stream) {
    if (batch == 0 || n_tokens == 0 || n_heads == 0) return 0;
    if (head_dim != PD_DN2_D) return cudaErrorInvalidValue;
    dim3 grid(n_heads, batch, PD_DN2_D / (2u * PD_DN2_WARPS));
    dim3 block(32, PD_DN2_WARPS);
    const int dns_cls = pd_dns_state_class();
    if (dns_cls == 3) {
        pd_gated_delta_verify_hold_kernel<__nv_fp8_e4m3>
            <<<grid, block, 0, (cudaStream_t)stream>>>(
                (const float*)q, (const float*)k, (const float*)v, (const float*)g,
                (const float*)beta, (const unsigned int*)slots,
                (const __nv_fp8_e4m3*)states, (float*)out, batch, n_tokens, n_heads);
    } else if (dns_cls == 2) {
        pd_gated_delta_verify_hold_kernel<__half>
            <<<grid, block, 0, (cudaStream_t)stream>>>(
                (const float*)q, (const float*)k, (const float*)v, (const float*)g,
                (const float*)beta, (const unsigned int*)slots,
                (const __half*)states, (float*)out, batch, n_tokens, n_heads);
    } else if (dns_cls == 1) {
        pd_gated_delta_verify_hold_kernel<__nv_bfloat16>
            <<<grid, block, 0, (cudaStream_t)stream>>>(
                (const float*)q, (const float*)k, (const float*)v, (const float*)g,
                (const float*)beta, (const unsigned int*)slots,
                (const __nv_bfloat16*)states, (float*)out, batch, n_tokens, n_heads);
    } else {
        pd_gated_delta_verify_hold_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
            (const float*)q, (const float*)k, (const float*)v, (const float*)g,
            (const float*)beta, (const unsigned int*)slots, (const float*)states,
            (float*)out, batch, n_tokens, n_heads);
    }
    return pd_launch_status();
}

// Chain-layout pick copy (dflash async round, slot 464): the block-draft
// graph writes its argmax picks row-major into d_out ([n, rows], pick j of
// block b at [b*rows + 1 + j]); the armed-chain verify assembles its tokens
// from d_draft in the MTP chain's i-major layout (d_draft[i*n + b]). One
// launch moves the picks device-side so the round never round-trips the
// host (the readback happens post-verify via the chain peek, when the
// stream has already synced).
__global__ void pd_dflash_chain_picks_kernel(const unsigned int* __restrict__ out,
                                             unsigned int* __restrict__ draft,
                                             uint32_t n, uint32_t rows,
                                             uint32_t k_use) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * k_use) return;
    const uint32_t b = idx % n;
    const uint32_t i = idx / n;
    draft[(size_t)i * n + b] = out[(size_t)b * rows + 1 + i];
}

PD_EXPORT
int pd_dflash_chain_picks(const void* out, void* draft, uint32_t n,
                          uint32_t rows, uint32_t k_use, void* stream) {
    if (n == 0 || k_use == 0) return 0;
    if (k_use + 1 > rows) return cudaErrorInvalidValue;
    const uint32_t total = n * k_use;
    const uint32_t block = 128;
    pd_dflash_chain_picks_kernel<<<(total + block - 1) / block, block, 0,
                                   (cudaStream_t)stream>>>(
        (const unsigned int*)out, (unsigned int*)draft, n, rows, k_use);
    return pd_launch_status();
}

PD_EXPORT
int pd_gated_delta_commit_walk(const void* k, const void* v, const void* g,
                               const void* beta, const void* slots,
                               const void* committed, void* states,
                               uint32_t batch, uint32_t n_tokens,
                               uint32_t n_heads, uint32_t head_dim, void* stream) {
    if (batch == 0 || n_tokens == 0 || n_heads == 0) return 0;
    if (head_dim != PD_DN2_D) return cudaErrorInvalidValue;
    dim3 grid(n_heads, batch, PD_DN2_D / (2u * PD_DN2_WARPS));
    dim3 block(32, PD_DN2_WARPS);
    const int dns_cls = pd_dns_state_class();
    if (dns_cls == 3) {
        pd_gated_delta_commit_walk_kernel<__nv_fp8_e4m3>
            <<<grid, block, 0, (cudaStream_t)stream>>>(
                (const float*)k, (const float*)v, (const float*)g,
                (const float*)beta, (const unsigned int*)slots,
                (const unsigned int*)committed, (__nv_fp8_e4m3*)states,
                n_tokens, n_heads);
    } else if (dns_cls == 2) {
        pd_gated_delta_commit_walk_kernel<__half>
            <<<grid, block, 0, (cudaStream_t)stream>>>(
                (const float*)k, (const float*)v, (const float*)g,
                (const float*)beta, (const unsigned int*)slots,
                (const unsigned int*)committed, (__half*)states,
                n_tokens, n_heads);
    } else if (dns_cls == 1) {
        pd_gated_delta_commit_walk_kernel<__nv_bfloat16>
            <<<grid, block, 0, (cudaStream_t)stream>>>(
                (const float*)k, (const float*)v, (const float*)g,
                (const float*)beta, (const unsigned int*)slots,
                (const unsigned int*)committed, (__nv_bfloat16*)states,
                n_tokens, n_heads);
    } else {
        pd_gated_delta_commit_walk_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
            (const float*)k, (const float*)v, (const float*)g,
            (const float*)beta, (const unsigned int*)slots,
            (const unsigned int*)committed, (float*)states, n_tokens, n_heads);
    }
    return pd_launch_status();
}

// ── P70: v2f - the DECODE recurrence with split+l2norm fused in ─────────
// The decode chain ran conv -> pd_deltanet_split_gqa_norm (write dq/dk/dv
// planes) -> v2 (read them back): one whole kernel plus a 3-plane round
// trip per GDN layer per round, for values the recurrence could compute
// itself (sglang's fused_recurrent_gated_delta_rule does exactly this).
// v2f reads the CONV plane directly: each (h,b,z)-block recomputes the
// q/k L2 norms - BIT-IDENTICAL to the split kernel: the same flat j =
// warp*32+lane element mapping, the same per-warp shfl_xor 16..1 chain,
// the same 4-slot smem cross-warp sum order, the same rsqrtf(sum+1e-6)
// (+ q's 1/sqrt D) - then stages q_hat/k_hat in smem and runs the v2 body
// verbatim (f32 plane round-trips were value-lossless, so outputs match
// the two-kernel chain byte-for-byte; the z-redundant recompute is L1-hot
// and costs less than the dead plane traffic). DECODE only: n_tokens = 1,
// no snap (spec rollback keeps the classic chain).
template <typename ST = float, bool CS = false>
__global__ void __launch_bounds__(128)
pd_gated_delta_recurrent_v2f_kernel(
        const float* __restrict__ conv, const float* __restrict__ g,
        const float* __restrict__ beta, const unsigned int* __restrict__ slots,
        ST* __restrict__ states, float* __restrict__ out, uint32_t n_k_heads,
        uint32_t n_heads, const float* __restrict__ ab, uint32_t ab_stride,
        const float* __restrict__ ssm_a, const float* __restrict__ dt_bias) {
    constexpr uint32_t D = PD_DN2_D;
    const uint32_t h = blockIdx.x;
    const uint32_t b = blockIdx.y;
    const uint32_t lane = threadIdx.x;
    const uint32_t col = (blockIdx.z * PD_DN2_WARPS + threadIdx.y) * 2u;

    // fused split + L2 norm (pd_deltanet_split_gqa_norm, verbatim order)
    const uint32_t hk = h % n_k_heads;
    const uint32_t key_dim = D * n_k_heads;
    const uint32_t conv_dim = 2u * key_dim + D * n_heads;
    const float* crow = conv + (size_t)b * conv_dim;
    const uint32_t j = threadIdx.y * 32u + lane; // flat 0..127 == old tid.x
    const float qj = crow[(size_t)hk * D + j];
    const float kj = crow[key_dim + (size_t)hk * D + j];
    float q2 = qj * qj, k2 = kj * kj;
#pragma unroll
    for (uint32_t off = 16; off > 0; off >>= 1) {
        q2 += __shfl_xor_sync(0xffffffffu, q2, off);
        k2 += __shfl_xor_sync(0xffffffffu, k2, off);
    }
    __shared__ float sh[8];
    if (lane == 0) { sh[threadIdx.y] = q2; sh[4 + threadIdx.y] = k2; }
    __syncthreads();
    float qs = 0.0f, ks = 0.0f;
#pragma unroll
    for (uint32_t w = 0; w < 4; ++w) { qs += sh[w]; ks += sh[4 + w]; }
    __shared__ float s_q[D], s_k[D];
    s_q[j] = qj * rsqrtf(qs + 1e-6f) * rsqrtf((float)D);
    s_k[j] = kj * rsqrtf(ks + 1e-6f);
    __syncthreads();

    // v2 body, n_tokens = 1, q/k from smem, v from the conv plane
    const uint32_t slot = slots ? slots[b] : b;
    ST* s_head = states + ((size_t)slot * n_heads + h) * (size_t)D * D
                        + (size_t)col * D;
    float4 sa = CS ? pd_dns_ld4_cs(s_head + lane * 4u)
                   : pd_dns_ld4(s_head + lane * 4u);
    float4 sb = CS ? pd_dns_ld4_cs(s_head + D + lane * 4u)
                   : pd_dns_ld4(s_head + D + lane * 4u);
    const float4 k4 = *reinterpret_cast<const float4*>(s_k + lane * 4u);
    const float4 q4 = *reinterpret_cast<const float4*>(s_q + lane * 4u);
    const float va = crow[2u * key_dim + (size_t)h * D + col];
    const float vb = crow[2u * key_dim + (size_t)h * D + col + 1u];
    const size_t gb = (size_t)b * n_heads + h;
    // P71-R2 gate-inline: with `ab` set, g/beta come straight off the DN
    // in-proj fused plane - pd_row_slice2_gate's expressions VERBATIM
    // (then the same expf), so the values are bit-identical while the
    // slice launch and the g/beta planes disappear. Uniform branch.
    float g_in, beta_in;
    if (ab) {
        const float* abrow = ab + (size_t)b * ab_stride;
        const float bx = abrow[n_heads + h];
        beta_in = 1.0f / (1.0f + expf(-bx));
        const float ax = abrow[h] + dt_bias[h];
        const float sp = fmaxf(ax, 0.0f) + log1pf(expf(-fabsf(ax)));
        g_in = ssm_a[h] * sp;
    } else {
        g_in = g[gb];
        beta_in = beta[gb];
    }
    const float g_t = expf(g_in);
    const float beta_t = beta_in;

    sa.x *= g_t; sa.y *= g_t; sa.z *= g_t; sa.w *= g_t;
    sb.x *= g_t; sb.y *= g_t; sb.z *= g_t; sb.w *= g_t;
    // plain v2 source expressions: the compiler contracts ua/ub exactly as
    // v2 (state stays BYTE-identical - probed); the oa/ob READOUT contracts
    // differently in this kernel body (1-ulp diffs, max_abs ~1e-8, probed).
    // Reassociation class, same as the mode-2 sampler's documented order
    // difference: the readout does not feed state, so nothing compounds -
    // state evolution is exact.
    float ua = sa.x * k4.x + sa.y * k4.y + sa.z * k4.z + sa.w * k4.w;
    float ub = sb.x * k4.x + sb.y * k4.y + sb.z * k4.z + sb.w * k4.w;
#pragma unroll
    for (uint32_t off = 16; off > 0; off >>= 1) {
        ua += __shfl_xor_sync(0xffffffffu, ua, off);
        ub += __shfl_xor_sync(0xffffffffu, ub, off);
    }
    const float da = beta_t * (va - ua);
    const float db = beta_t * (vb - ub);
    sa.x += k4.x * da; sa.y += k4.y * da; sa.z += k4.z * da; sa.w += k4.w * da;
    sb.x += k4.x * db; sb.y += k4.y * db; sb.z += k4.z * db; sb.w += k4.w * db;
    float oa = sa.x * q4.x + sa.y * q4.y + sa.z * q4.z + sa.w * q4.w;
    float ob = sb.x * q4.x + sb.y * q4.y + sb.z * q4.z + sb.w * q4.w;
#pragma unroll
    for (uint32_t off = 16; off > 0; off >>= 1) {
        oa += __shfl_xor_sync(0xffffffffu, oa, off);
        ob += __shfl_xor_sync(0xffffffffu, ob, off);
    }
    const size_t obase = ((size_t)b * n_heads + h) * (size_t)D;
    if (lane == 0) {
        out[obase + col] = oa;
        out[obase + col + 1u] = ob;
    }
    if (CS) {
        pd_dns_st4_cs(s_head + lane * 4u, sa);
        pd_dns_st4_cs(s_head + D + lane * 4u, sb);
    } else {
        pd_dns_st4(s_head + lane * 4u, sa);
        pd_dns_st4(s_head + D + lane * 4u, sb);
    }
}

static int pd_dn_v2f_launch(const void* conv, const void* g,
                            const void* beta, const void* slots,
                            void* states, void* out, uint32_t batch,
                            uint32_t n_k_heads, uint32_t n_heads,
                            uint32_t head_dim, const float* ab,
                            uint32_t ab_stride, const float* ssm_a,
                            const float* dt_bias, void* stream) {
    if (batch == 0 || n_heads == 0) return 0;
    if (head_dim != PD_DN2_D) return cudaErrorInvalidValue;
    dim3 grid(n_heads, batch, PD_DN2_D / (2u * PD_DN2_WARPS));
    dim3 block(32, PD_DN2_WARPS);
    static const bool cs = pd_env("PADDOCK_DNS_CS") != nullptr;
    const int dns_cls = pd_dns_state_class();
    if (dns_cls == 3) {
        pd_gated_delta_recurrent_v2f_kernel<__nv_fp8_e4m3>
            <<<grid, block, 0, (cudaStream_t)stream>>>(
                (const float*)conv, (const float*)g, (const float*)beta,
                (const unsigned int*)slots, (__nv_fp8_e4m3*)states,
                (float*)out, n_k_heads, n_heads, ab, ab_stride, ssm_a, dt_bias);
    } else if (dns_cls == 2) {
        if (cs)
            pd_gated_delta_recurrent_v2f_kernel<__half, true>
                <<<grid, block, 0, (cudaStream_t)stream>>>(
                    (const float*)conv, (const float*)g, (const float*)beta,
                    (const unsigned int*)slots, (__half*)states,
                    (float*)out, n_k_heads, n_heads, ab, ab_stride, ssm_a, dt_bias);
        else
            pd_gated_delta_recurrent_v2f_kernel<__half>
                <<<grid, block, 0, (cudaStream_t)stream>>>(
                    (const float*)conv, (const float*)g, (const float*)beta,
                    (const unsigned int*)slots, (__half*)states,
                    (float*)out, n_k_heads, n_heads, ab, ab_stride, ssm_a, dt_bias);
    } else if (dns_cls == 1) {
        if (cs)
            pd_gated_delta_recurrent_v2f_kernel<__nv_bfloat16, true>
                <<<grid, block, 0, (cudaStream_t)stream>>>(
                    (const float*)conv, (const float*)g, (const float*)beta,
                    (const unsigned int*)slots, (__nv_bfloat16*)states,
                    (float*)out, n_k_heads, n_heads, ab, ab_stride, ssm_a, dt_bias);
        else
            pd_gated_delta_recurrent_v2f_kernel<__nv_bfloat16>
                <<<grid, block, 0, (cudaStream_t)stream>>>(
                    (const float*)conv, (const float*)g, (const float*)beta,
                    (const unsigned int*)slots, (__nv_bfloat16*)states,
                    (float*)out, n_k_heads, n_heads, ab, ab_stride, ssm_a, dt_bias);
    } else if (cs) {
        pd_gated_delta_recurrent_v2f_kernel<float, true>
            <<<grid, block, 0, (cudaStream_t)stream>>>(
                (const float*)conv, (const float*)g, (const float*)beta,
                (const unsigned int*)slots, (float*)states, (float*)out,
                n_k_heads, n_heads, ab, ab_stride, ssm_a, dt_bias);
    } else {
        pd_gated_delta_recurrent_v2f_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
            (const float*)conv, (const float*)g, (const float*)beta,
            (const unsigned int*)slots, (float*)states, (float*)out,
            n_k_heads, n_heads, ab, ab_stride, ssm_a, dt_bias);
    }
    return pd_launch_status();
}

PD_EXPORT
int pd_gated_delta_recurrent_v2f(const void* conv, const void* g,
                                 const void* beta, const void* slots,
                                 void* states, void* out, uint32_t batch,
                                 uint32_t n_k_heads, uint32_t n_heads,
                                 uint32_t head_dim, void* stream) {
    return pd_dn_v2f_launch(conv, g, beta, slots, states, out, batch,
                            n_k_heads, n_heads, head_dim, nullptr, 0u,
                            nullptr, nullptr, stream);
}

// P71-R2 gate-inline twin: g/beta computed in-kernel from the DN in-proj
// fused plane (row_slice2_gate's expressions verbatim) - the slice launch
// and the g/beta planes disappear; values bit-identical.
PD_EXPORT
int pd_gated_delta_recurrent_v2f_g(const void* conv, const void* fused,
                                   uint32_t ab_off, uint32_t fused_stride,
                                   const void* ssm_a, const void* dt_bias,
                                   const void* slots, void* states, void* out,
                                   uint32_t batch, uint32_t n_k_heads,
                                   uint32_t n_heads, uint32_t head_dim,
                                   void* stream) {
    if (!fused || !ssm_a || !dt_bias) return cudaErrorInvalidValue;
    return pd_dn_v2f_launch(conv, nullptr, nullptr, slots, states, out, batch,
                            n_k_heads, n_heads, head_dim,
                            (const float*)fused + ab_off, fused_stride,
                            (const float*)ssm_a, (const float*)dt_bias,
                            stream);
}

// Packed multi-span serial recurrence: the mixed tick used to run the b-row
// decode step (one slots launch) and then each sub-chunk_min prefill span as
// its own b=1 v2_at launch - tens of thousands of extra launches on a
// 2048x128 c32 workload (~1.28s of pure launch serialization). Nearly all
// of those spans are FUSED CKPT TAIL chains (a chunked leader share cut at
// the DN checkpoint boundary + 1-2 short same-slot tail shares, with the
// boundary state copy_region'd to a stage blob between them), so packing
// needs chain support, not just independent items. This kernel takes u32
// descriptors of STRIDE 8 - (row0, len, slot, snapA_t, snapA_sel, snapB_t,
// snapB_sel, pad) - and walks every item in one launch: decode rows ride as
// len-1 items, an entire same-slot serial chain rides as one item (its
// shares' rows are contiguous in the tick buffers), and each internal seam
// that used to be a stage copy becomes an IN-KERNEL snapshot: after row
// snapX_t of the item, the block writes its register-resident state slice
// to snap0/snap1 (selected by snapX_sel - the per-layer pre-offset stage
// blob state regions, same transposed tile layout as `states`). snapX_t==0
// means no snapshot. The walk body is the v2 kernel's verbatim (same math,
// same order): a chain item is bit-exact vs the per-share _at walks + the
// stage copies it replaces, because the state is continuous across the seam
// either way. q/k/v/out are addressed by ABSOLUTE row (row0+t). Items must
// touch DISTINCT slots; the caller launches this after the chunked span
// loop so chain leaders (chunked class) have already advanced the state.
// No per-token snap array: speculative paths keep the v2 call.
template <typename ST = float, bool CS = false>
__global__ void PD_DN2_LB
pd_gated_delta_recurrent_v2_packed_kernel(
        const float* __restrict__ q, const float* __restrict__ k,
        const float* __restrict__ v, const float* __restrict__ g,
        const float* __restrict__ beta, const uint32_t* __restrict__ items,
        ST* __restrict__ states, float* __restrict__ out,
        ST* __restrict__ snap0, ST* __restrict__ snap1, uint32_t n_heads) {
    constexpr uint32_t D = PD_DN2_D;
    const uint32_t h = blockIdx.x;
    const uint32_t it = blockIdx.y;
    const uint32_t lane = threadIdx.x;
    const uint32_t col = (blockIdx.z * PD_DN2_WARPS + threadIdx.y) * 2u;
    const uint32_t row0 = items[it * 8u];
    const uint32_t len = items[it * 8u + 1u];
    const uint32_t slot = items[it * 8u + 2u];
    const uint32_t sa_t = items[it * 8u + 3u];
    const uint32_t sa_sel = items[it * 8u + 4u];
    const uint32_t sb_t = items[it * 8u + 5u];
    const uint32_t sb_sel = items[it * 8u + 6u];

    ST* s_head = states + ((size_t)slot * n_heads + h) * (size_t)D * D
                        + (size_t)col * D;
    float4 sa = CS ? pd_dns_ld4_cs(s_head + lane * 4u)
                   : pd_dns_ld4(s_head + lane * 4u);
    float4 sb = CS ? pd_dns_ld4_cs(s_head + D + lane * 4u)
                   : pd_dns_ld4(s_head + D + lane * 4u);

    for (uint32_t t = 0; t < len; ++t) {
        const size_t base = ((size_t)(row0 + t) * n_heads + h) * (size_t)D;
        const float4 k4 = *reinterpret_cast<const float4*>(k + base + lane * 4u);
        const float4 q4 = *reinterpret_cast<const float4*>(q + base + lane * 4u);
        const float va = v[base + col];
        const float vb = v[base + col + 1u];
        const size_t gb = (size_t)(row0 + t) * n_heads + h;
        const float g_t = expf(g[gb]);
        const float beta_t = beta[gb];

        sa.x *= g_t; sa.y *= g_t; sa.z *= g_t; sa.w *= g_t;
        sb.x *= g_t; sb.y *= g_t; sb.z *= g_t; sb.w *= g_t;
        float ua = sa.x * k4.x + sa.y * k4.y + sa.z * k4.z + sa.w * k4.w;
        float ub = sb.x * k4.x + sb.y * k4.y + sb.z * k4.z + sb.w * k4.w;
#pragma unroll
        for (uint32_t off = 16; off > 0; off >>= 1) {
            ua += __shfl_xor_sync(0xffffffffu, ua, off);
            ub += __shfl_xor_sync(0xffffffffu, ub, off);
        }
        const float da = beta_t * (va - ua);
        const float db = beta_t * (vb - ub);

        sa.x += k4.x * da; sa.y += k4.y * da; sa.z += k4.z * da; sa.w += k4.w * da;
        sb.x += k4.x * db; sb.y += k4.y * db; sb.z += k4.z * db; sb.w += k4.w * db;
        float oa = sa.x * q4.x + sa.y * q4.y + sa.z * q4.z + sa.w * q4.w;
        float ob = sb.x * q4.x + sb.y * q4.y + sb.z * q4.z + sb.w * q4.w;
#pragma unroll
        for (uint32_t off = 16; off > 0; off >>= 1) {
            oa += __shfl_xor_sync(0xffffffffu, oa, off);
            ob += __shfl_xor_sync(0xffffffffu, ob, off);
        }
        if (lane == 0) {
            out[base + col] = oa;
            out[base + col + 1u] = ob;
        }

        // in-kernel seam snapshot: the state after row t+1 of the chain is
        // exactly what the replaced copy_region staged between the shares
        if (t + 1u == sa_t || t + 1u == sb_t) {
            ST* sn = ((t + 1u == sa_t ? sa_sel : sb_sel) ? snap1 : snap0)
                   + (size_t)h * D * D + (size_t)col * D;
            pd_dns_st4(sn + lane * 4u, sa);
            pd_dns_st4(sn + D + lane * 4u, sb);
        }
    }

    if (CS) {
        pd_dns_st4_cs(s_head + lane * 4u, sa);
        pd_dns_st4_cs(s_head + D + lane * 4u, sb);
    } else {
        pd_dns_st4(s_head + lane * 4u, sa);
        pd_dns_st4(s_head + D + lane * 4u, sb);
    }
}

PD_EXPORT
int pd_gated_delta_recurrent_v2_packed(const void* q, const void* k,
                                       const void* v, const void* g,
                                       const void* beta, const void* items,
                                       void* states, void* out, void* snap0,
                                       void* snap1, uint32_t n_items,
                                       uint32_t n_heads, uint32_t head_dim,
                                       void* stream) {
    if (n_items == 0 || n_heads == 0) return 0;
    if (head_dim != PD_DN2_D) return cudaErrorInvalidValue;
    dim3 grid(n_heads, n_items, PD_DN2_D / (2u * PD_DN2_WARPS));
    dim3 block(32, PD_DN2_WARPS);
    static const bool cs = pd_env("PADDOCK_DNS_CS") != nullptr;
    const int dns_cls = pd_dns_state_class();
    if (dns_cls == 3) {
        pd_gated_delta_recurrent_v2_packed_kernel<__nv_fp8_e4m3>
            <<<grid, block, 0, (cudaStream_t)stream>>>(
                (const float*)q, (const float*)k, (const float*)v, (const float*)g,
                (const float*)beta, (const uint32_t*)items,
                (__nv_fp8_e4m3*)states, (float*)out, (__nv_fp8_e4m3*)snap0,
                (__nv_fp8_e4m3*)snap1, n_heads);
    } else if (dns_cls == 2) {
        if (cs)
            pd_gated_delta_recurrent_v2_packed_kernel<__half, true>
                <<<grid, block, 0, (cudaStream_t)stream>>>(
                    (const float*)q, (const float*)k, (const float*)v, (const float*)g,
                    (const float*)beta, (const uint32_t*)items,
                    (__half*)states, (float*)out, (__half*)snap0,
                    (__half*)snap1, n_heads);
        else
            pd_gated_delta_recurrent_v2_packed_kernel<__half>
                <<<grid, block, 0, (cudaStream_t)stream>>>(
                    (const float*)q, (const float*)k, (const float*)v, (const float*)g,
                    (const float*)beta, (const uint32_t*)items,
                    (__half*)states, (float*)out, (__half*)snap0,
                    (__half*)snap1, n_heads);
    } else if (dns_cls == 1) {
        if (cs)
            pd_gated_delta_recurrent_v2_packed_kernel<__nv_bfloat16, true>
                <<<grid, block, 0, (cudaStream_t)stream>>>(
                    (const float*)q, (const float*)k, (const float*)v, (const float*)g,
                    (const float*)beta, (const uint32_t*)items,
                    (__nv_bfloat16*)states, (float*)out, (__nv_bfloat16*)snap0,
                    (__nv_bfloat16*)snap1, n_heads);
        else
            pd_gated_delta_recurrent_v2_packed_kernel<__nv_bfloat16>
                <<<grid, block, 0, (cudaStream_t)stream>>>(
                    (const float*)q, (const float*)k, (const float*)v, (const float*)g,
                    (const float*)beta, (const uint32_t*)items,
                    (__nv_bfloat16*)states, (float*)out, (__nv_bfloat16*)snap0,
                    (__nv_bfloat16*)snap1, n_heads);
    } else if (cs) {
        pd_gated_delta_recurrent_v2_packed_kernel<float, true>
            <<<grid, block, 0, (cudaStream_t)stream>>>(
                (const float*)q, (const float*)k, (const float*)v, (const float*)g,
                (const float*)beta, (const uint32_t*)items, (float*)states,
                (float*)out, (float*)snap0, (float*)snap1, n_heads);
    } else {
        pd_gated_delta_recurrent_v2_packed_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
            (const float*)q, (const float*)k, (const float*)v, (const float*)g,
            (const float*)beta, (const uint32_t*)items, (float*)states,
            (float*)out, (float*)snap0, (float*)snap1, n_heads);
    }
    return pd_launch_status();
}

// Chunked gated delta rule (prefill) -- the recurrence above restructured so
// only n_tokens/64 state hops are sequential; everything inside a chunk is
// dense parallel work. Math, with chunk-local cumulative log-decay cg_i (all
// ratios exp(cg_i - cg_j), j <= i, bounded by 1 since g <= 0):
//   (I + M) Delta = diag(beta) (V - diag(exp cg) K S0)
//        M[i][j] = beta_i exp(cg_i - cg_j) (k_i . k_j),   j < i
//   o_i = exp(cg_i) (q_i^T S0) + sum_{j<=i} exp(cg_i - cg_j)(q_i . k_j) delta_j
//   S  <- exp(cg_last) S0 + sum_j exp(cg_last - cg_j) k_j (x) delta_j
// (I + M) is unit lower triangular. S0 (chunk-start state) is produced by the
// previous chunk, so stage 1 forward-substitutes T = (I+M)^-1 into two S0-free
// right-hand sides -- du = T diag(beta) V and dw = T diag(beta exp cg) K -- and
// the state-dependent deltas resolve later as Delta = du - dw S0. The
// inter-chunk recursion is independent per VALUE COLUMN (the property the v2
// kernel exploits), so stage 2 is one launch: each block owns (head, 16-column
// slice), walks the chunks with its state slice in shared, and stashes each
// chunk-start slice for stage 3's output assembly. Numeric recipe matches the
// CPU oracle (reference gated_delta_chunked): f32 FMA everywhere; cumulative
// log-decay carried in f64 with ratios taken as expf of the f64 difference
// (f32 rounding of |cg| ~ 10s costs ~1e-5 relative after the exp).
// q/k/v/out are [T, H, D] with q,k PRE-normalized (q carries 1/sqrt D);
// g/beta [T, H]; state [H, D, D] TRANSPOSED column-contiguous like v2.
// Not bit-identical to the sequential recurrence (different accumulation
// structure): prefill-only -- decode and speculative paths stay on v2.
#define PD_DNC_D 128
#define PD_DNC_C 64
#define PD_DNC_G 32
// stage 1 dynamic shared: M [C][C] | cg [C] f64 | sub [C][D] (kT/qT tiles,
// then dw, then du) -- one [C][D] pane substituted twice keeps the block at
// ~48.5 KB so two blocks share an SM (the 80 KB single-pass variant ran at
// 8 warps/SM, issue 0.22)
#define PD_DNC_S1_SMEM ((PD_DNC_C * PD_DNC_C + 2 * PD_DNC_C + PD_DNC_C * PD_DNC_D) * 4)

// Stage 1, grid (n_chunks, H) x 256: per chunk-head, the dot matrices, M, and
// forward substitution applied to both right-hand sides. The dots run as a
// register-tile matmul over 16-wide TRANSPOSED a-slices of k/q staged into the
// (still unused) substitution pane -- thread (i-quad, j-quad) accumulates 4x4
// akk/aqk tiles from three float4 loads per a-step, no shuffles (the
// warp-reduce version spent 2/3 of its issue on the reductions). The
// substitution is column-independent (thread owns a column, no syncs across
// the C steps); it runs once over the dw pane (the k rows scaled at load) and
// once over du.
__global__ void __launch_bounds__(256)
pd_dnc_stage1_kernel(const float* __restrict__ q, const float* __restrict__ k,
                     const float* __restrict__ v, const float* __restrict__ g,
                     const float* __restrict__ beta, float* __restrict__ dw,
                     float* __restrict__ du, float* __restrict__ aqk,
                     double* __restrict__ cg, uint32_t n_tokens, uint32_t n_heads) {
    constexpr uint32_t D = PD_DNC_D, C = PD_DNC_C;
    const uint32_t ch = blockIdx.x, h = blockIdx.y;
    const uint32_t c0 = ch * C;
    const uint32_t cl = min(C, n_tokens - c0);
    const uint32_t tid = threadIdx.x;

    extern __shared__ float sh[];
    float* sh_m = sh;                       // [C][C], j < i
    double* sh_cg = (double*)(sh + C * C);  // [C]
    float* sh_sub = sh + C * C + 2 * C;     // [C][D]: kT/qT tiles, then dw, du
    __shared__ float sh_b[PD_DNC_C], sh_bg[PD_DNC_C];

    // stage g values in parallel, then one thread runs the f64 cumsum
    if (tid < cl) sh_b[tid] = g[(size_t)(c0 + tid) * n_heads + h];
    __syncthreads();
    if (tid == 0) {
        double run = 0.0;
        for (uint32_t i = 0; i < cl; ++i) {
            run += (double)sh_b[i];
            sh_cg[i] = run;
            cg[((size_t)ch * n_heads + h) * C + i] = run;
        }
    }
    __syncthreads();
    if (tid < cl) {
        sh_b[tid] = beta[(size_t)(c0 + tid) * n_heads + h];
        sh_bg[tid] = sh_b[tid] * expf((float)sh_cg[tid]);
    }

    // dot matrices through transposed a-tiles staged in the pane region
    float* sh_kt = sh_sub;                  // [16][C + 4]
    float* sh_qt = sh_sub + 16u * (C + 4u); // [16][C + 4]
    const uint32_t i0 = (tid >> 4) * 4u;    // output tile rows
    const uint32_t j0 = (tid & 15u) * 4u;   // output tile cols
    float akk_r[4][4] = {}, aqk_r[4][4] = {};
    for (uint32_t a0 = 0; a0 < D; a0 += 16u) {
        __syncthreads();
        for (uint32_t idx = tid; idx < cl * 16u; idx += 256) {
            const uint32_t i = idx / 16u, aa = idx % 16u;
            const size_t rbase = ((size_t)(c0 + i) * n_heads + h) * D + a0 + aa;
            sh_kt[aa * (C + 4u) + i] = k[rbase];
            sh_qt[aa * (C + 4u) + i] = q[rbase];
        }
        __syncthreads();
#pragma unroll
        for (uint32_t aa = 0; aa < 16u; ++aa) {
            const float4 kj =
                *reinterpret_cast<const float4*>(&sh_kt[aa * (C + 4u) + j0]);
            const float4 ki =
                *reinterpret_cast<const float4*>(&sh_kt[aa * (C + 4u) + i0]);
            const float4 qi =
                *reinterpret_cast<const float4*>(&sh_qt[aa * (C + 4u) + i0]);
            const float kiv[4] = {ki.x, ki.y, ki.z, ki.w};
            const float qiv[4] = {qi.x, qi.y, qi.z, qi.w};
#pragma unroll
            for (uint32_t ii = 0; ii < 4; ++ii) {
                akk_r[ii][0] = fmaf(kiv[ii], kj.x, akk_r[ii][0]);
                akk_r[ii][1] = fmaf(kiv[ii], kj.y, akk_r[ii][1]);
                akk_r[ii][2] = fmaf(kiv[ii], kj.z, akk_r[ii][2]);
                akk_r[ii][3] = fmaf(kiv[ii], kj.w, akk_r[ii][3]);
                aqk_r[ii][0] = fmaf(qiv[ii], kj.x, aqk_r[ii][0]);
                aqk_r[ii][1] = fmaf(qiv[ii], kj.y, aqk_r[ii][1]);
                aqk_r[ii][2] = fmaf(qiv[ii], kj.z, aqk_r[ii][2]);
                aqk_r[ii][3] = fmaf(qiv[ii], kj.w, aqk_r[ii][3]);
            }
        }
    }
    // emit: the READY output-coefficient matrix coef[i][j] = e^{cg_i-cg_j} *
    // (q_i . k_j), zeroed above the diagonal, straight to global (stage 2
    // streams it; rows past cl are dead scratch), and M to shared for the
    // substitution (j < i only; dead rows never read)
    {
        const size_t ab = ((size_t)ch * n_heads + h) * C * C;
#pragma unroll
        for (uint32_t ii = 0; ii < 4; ++ii) {
            const uint32_t i = i0 + ii;
            float cf[4];
#pragma unroll
            for (uint32_t jj = 0; jj < 4; ++jj) {
                const uint32_t j = j0 + jj;
                const float ratio = j <= i ? expf((float)(sh_cg[i] - sh_cg[j])) : 0.f;
                cf[jj] = ratio * aqk_r[ii][jj];
                if (j < i) sh_m[i * C + j] = sh_b[i] * ratio * akk_r[ii][jj];
            }
            *reinterpret_cast<float4*>(&aqk[ab + i * C + j0]) =
                make_float4(cf[0], cf[1], cf[2], cf[3]);
        }
    }
    __syncthreads();

    // pass 1: dw = bg * k rows into the pane, substitute, write out
    for (uint32_t idx = tid; idx < cl * (D / 4u); idx += 256) {
        const uint32_t i = idx / (D / 4u), a4 = idx % (D / 4u);
        const float4 kv = *reinterpret_cast<const float4*>(
            k + ((size_t)(c0 + i) * n_heads + h) * D + a4 * 4u);
        const float s = sh_bg[i];
        sh_sub[i * D + a4 * 4u] = s * kv.x;
        sh_sub[i * D + a4 * 4u + 1u] = s * kv.y;
        sh_sub[i * D + a4 * 4u + 2u] = s * kv.z;
        sh_sub[i * D + a4 * 4u + 3u] = s * kv.w;
    }
    __syncthreads();
    if (tid < D) {
        for (uint32_t i = 1; i < cl; ++i) {
            float acc = sh_sub[i * D + tid];
            for (uint32_t j = 0; j < i; ++j)
                acc = fmaf(-sh_m[i * C + j], sh_sub[j * D + tid], acc);
            sh_sub[i * D + tid] = acc;
        }
    }
    __syncthreads();
    for (uint32_t idx = tid; idx < cl * (D / 4u); idx += 256) {
        const uint32_t i = idx / (D / 4u), a4 = idx % (D / 4u);
        *reinterpret_cast<float4*>(
            &dw[(((size_t)ch * n_heads + h) * C + i) * D + a4 * 4u]) =
            *reinterpret_cast<const float4*>(&sh_sub[i * D + a4 * 4u]);
    }
    __syncthreads();

    // pass 2: du = beta * v, substitutes, writes out
    for (uint32_t idx = tid; idx < cl * D; idx += 256) {
        const uint32_t i = idx / D, a = idx % D;
        sh_sub[idx] = sh_b[i] * v[((size_t)(c0 + i) * n_heads + h) * D + a];
    }
    __syncthreads();
    if (tid < D) {
        for (uint32_t i = 1; i < cl; ++i) {
            float acc = sh_sub[i * D + tid];
            for (uint32_t j = 0; j < i; ++j)
                acc = fmaf(-sh_m[i * C + j], sh_sub[j * D + tid], acc);
            sh_sub[i * D + tid] = acc;
        }
    }
    __syncthreads();
    for (uint32_t idx = tid; idx < cl * (D / 4u); idx += 256) {
        const uint32_t i = idx / (D / 4u), a4 = idx % (D / 4u);
        *reinterpret_cast<float4*>(
            &du[(((size_t)ch * n_heads + h) * C + i) * D + a4 * 4u]) =
            *reinterpret_cast<const float4*>(&sh_sub[i * D + a4 * 4u]);
    }
}

// Stage 2, grid (H, D/G) x 256, G = 32: the sequential chunk walk AND the
// whole output assembly. Each block owns G state columns of one head, kept in
// shared across the walk. Per chunk: resolve Delta = du - dw S0 on its
// columns together with the readout term gam_i (q_i . S0[:,dv]), write the
// finished outputs directly --
//   out[i][dv] = gam_i (q_i . S0[:,dv]) + sum_j coef[i][j] delta_j[dv]
// with coef streamed from global (stage 1 pre-folds the decay ratios and the
// triangular mask) -- then hop the state with the decay weights applied to
// the k rows. Owning the state columns makes the readout term free of the
// 16 MB chunk-state stash a separate output stage would need; folding the
// coef matmul here kills the delta writeback round-trip and a third kernel.
// The dw/q/k operand rows stream straight from global through L1 (each row is
// read once per warp, coalesced, and reused by the register tiles) -- staging
// them through a shared pane cost ~12 barriers per chunk and capped the block
// at half occupancy; this shape needs two barriers per chunk.
template <uint32_t G>
__global__ void __launch_bounds__(256)
pd_dnc_stage2_kernel(const float* __restrict__ q, const float* __restrict__ k,
                     float* __restrict__ state, const float* __restrict__ dw,
                     const float* __restrict__ du, const double* __restrict__ cg,
                     const float* __restrict__ coef, float* __restrict__ out,
                     uint32_t n_tokens, uint32_t n_heads) {
    constexpr uint32_t D = PD_DNC_D, C = PD_DNC_C;
    // NCC column-groups of 8 per thread cover the block's G state columns.
    // G is a pure work REDISTRIBUTION: every output/state column's sums run
    // in the identical (a4 ascending, j ascending) order at any G, so
    // G=32/16/8 are bit-exact to each other - smaller G only buys GRID (the
    // (32 heads x D/G) launch is 128 blocks at G=32 on a 188-SM die = 0.68
    // waves with every block walking all chunks serially; G=16 doubles it).
    constexpr uint32_t NCC = G / 8u;
    const uint32_t h = blockIdx.x;
    const uint32_t col0 = blockIdx.y * G;
    const uint32_t tid = threadIdx.x;
    const uint32_t nc = (n_tokens + C - 1u) / C;

    __shared__ float sh_s[G][D + 4];      // owned state columns
    __shared__ float sh_delta[C][G + 4];  // resolved deltas, owned columns
    __shared__ float sh_w[C], sh_gam[C];
    __shared__ float sh_gall;

    float* s_head = state + (size_t)h * D * D;
    for (uint32_t idx = tid; idx < G * (D / 4u); idx += 256) {
        const uint32_t c = idx / (D / 4u), a4 = idx % (D / 4u);
        *reinterpret_cast<float4*>(&sh_s[c][a4 * 4u]) =
            *reinterpret_cast<const float4*>(s_head + (size_t)(col0 + c) * D + a4 * 4u);
    }
    __syncthreads();

    // delta/o1/out tiles: rows (2p, 2p+1) x cols {q, q+8, q+16, q+24}
    const uint32_t dp = tid >> 3, dq = tid & 7u;
    // hop tile: cols {cp, cp+8, cp+16, cp+24} x a-quad
    const uint32_t cp = tid >> 5, a0 = (tid & 31u) * 4u;

    for (uint32_t ch = 0; ch < nc; ++ch) {
        const uint32_t c0 = ch * C;
        const uint32_t cl = min(C, n_tokens - c0);
        const size_t tb = (size_t)ch * n_heads + h;

        if (tid < cl) {
            sh_w[tid] = expf((float)(cg[tb * C + cl - 1u] - cg[tb * C + tid]));
            sh_gam[tid] = expf((float)cg[tb * C + tid]);
        }
        if (tid == 0) sh_gall = expf((float)cg[tb * C + cl - 1u]);

        // Delta = du - dw S0 and the o1 readout, dw/q rows straight from L1
        // (rows past cl clamp to a valid token; their results are discarded)
        const uint32_t i0 = dp * 2u, i1 = i0 + 1u;
        const float* dwr0 = dw + (tb * C + i0) * D;
        const float* dwr1 = dwr0 + D;
        const float* qr0 =
            q + ((size_t)min(c0 + i0, n_tokens - 1u) * n_heads + h) * D;
        const float* qr1 =
            q + ((size_t)min(c0 + i1, n_tokens - 1u) * n_heads + h) * D;
        float dacc[2][NCC] = {}, oacc[2][NCC] = {};
#pragma unroll 4
        for (uint32_t a4 = 0; a4 < D; a4 += 4u) {
            const float4 w0 = *reinterpret_cast<const float4*>(dwr0 + a4);
            const float4 w1 = *reinterpret_cast<const float4*>(dwr1 + a4);
            const float4 q0 = *reinterpret_cast<const float4*>(qr0 + a4);
            const float4 q1 = *reinterpret_cast<const float4*>(qr1 + a4);
#pragma unroll
            for (uint32_t cc = 0; cc < NCC; ++cc) {
                const float4 sv =
                    *reinterpret_cast<const float4*>(&sh_s[dq + cc * 8u][a4]);
                dacc[0][cc] += w0.x * sv.x + w0.y * sv.y + w0.z * sv.z + w0.w * sv.w;
                dacc[1][cc] += w1.x * sv.x + w1.y * sv.y + w1.z * sv.z + w1.w * sv.w;
                oacc[0][cc] += q0.x * sv.x + q0.y * sv.y + q0.z * sv.z + q0.w * sv.w;
                oacc[1][cc] += q1.x * sv.x + q1.y * sv.y + q1.z * sv.z + q1.w * sv.w;
            }
        }
#pragma unroll
        for (uint32_t ii = 0; ii < 2; ++ii) {
            const uint32_t i = i0 + ii;
            if (i < cl) {
#pragma unroll
                for (uint32_t cc = 0; cc < NCC; ++cc) {
                    const uint32_t c = dq + cc * 8u;
                    sh_delta[i][c] = du[(tb * C + i) * D + col0 + c] - dacc[ii][cc];
                }
            }
        }
        __syncthreads();

        // fused output: out = gam (q . S0) + sum_{j<=i} coef[i][j] delta_j
        {
            float outa[2][NCC];
#pragma unroll
            for (uint32_t ii = 0; ii < 2; ++ii)
#pragma unroll
                for (uint32_t cc = 0; cc < NCC; ++cc)
                    outa[ii][cc] = sh_gam[min(i0 + ii, cl - 1u)] * oacc[ii][cc];
            const float* cf0 = coef + (tb * C + i0) * C;
            for (uint32_t j = 0; j < cl; ++j) {
                const float c00 = cf0[j];
                const float c10 = cf0[C + j];
#pragma unroll
                for (uint32_t cc = 0; cc < NCC; ++cc) {
                    const float dlt = sh_delta[j][dq + cc * 8u];
                    outa[0][cc] = fmaf(c00, dlt, outa[0][cc]);
                    outa[1][cc] = fmaf(c10, dlt, outa[1][cc]);
                }
            }
#pragma unroll
            for (uint32_t ii = 0; ii < 2; ++ii) {
                const uint32_t i = i0 + ii;
                if (i < cl) {
#pragma unroll
                    for (uint32_t cc = 0; cc < NCC; ++cc)
                        out[((size_t)(c0 + i) * n_heads + h) * D + col0 + dq + cc * 8u] =
                            outa[ii][cc];
                }
            }
        }

        // state hop: S[c][a] = gall S[c][a] + sum_j (w_j k_j[a]) delta_j[c],
        // k rows from L1 with the decay weight applied on the k side (the
        // (w k) . delta order matches the CPU oracle)
        {
            float4 acc[NCC];
#pragma unroll
            for (uint32_t cc = 0; cc < NCC; ++cc) {
                acc[cc] = *reinterpret_cast<const float4*>(&sh_s[cp + cc * 8u][a0]);
                acc[cc].x *= sh_gall; acc[cc].y *= sh_gall;
                acc[cc].z *= sh_gall; acc[cc].w *= sh_gall;
            }
            for (uint32_t j = 0; j < cl; ++j) {
                const float4 kj = *reinterpret_cast<const float4*>(
                    k + ((size_t)(c0 + j) * n_heads + h) * D + a0);
                const float wj = sh_w[j];
                const float4 kw =
                    make_float4(wj * kj.x, wj * kj.y, wj * kj.z, wj * kj.w);
#pragma unroll
                for (uint32_t cc = 0; cc < NCC; ++cc) {
                    const float dlt = sh_delta[j][cp + cc * 8u];
                    acc[cc].x = fmaf(kw.x, dlt, acc[cc].x);
                    acc[cc].y = fmaf(kw.y, dlt, acc[cc].y);
                    acc[cc].z = fmaf(kw.z, dlt, acc[cc].z);
                    acc[cc].w = fmaf(kw.w, dlt, acc[cc].w);
                }
            }
#pragma unroll
            for (uint32_t cc = 0; cc < NCC; ++cc)
                *reinterpret_cast<float4*>(&sh_s[cp + cc * 8u][a0]) = acc[cc];
        }
        __syncthreads();
    }

    for (uint32_t idx = tid; idx < G * (D / 4u); idx += 256) {
        const uint32_t c = idx / (D / 4u), a4 = idx % (D / 4u);
        *reinterpret_cast<float4*>(s_head + (size_t)(col0 + c) * D + a4 * 4u) =
            *reinterpret_cast<const float4*>(&sh_s[c][a4 * 4u]);
    }
}

