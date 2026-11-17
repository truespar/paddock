// NVFP4 MoE consumers over the TILED expert-plane layout (the skinny-tile
// pair). A mechanism probe left one candidate for marlin's per-byte edge on
// the c32 decode MoE band: the tile SHAPE.
// Pricing it out left every arm BIT-EXACT vs the
// shipped bs pair: the winning axis is the LAYOUT (contiguous per-stage
// spans -> 512 B-class warp loads; the shipped row-major pair reads 128 B
// rows at ~1.4 KB stride), with the skinny BM=8 tile second and ring depth
// a wash (ST 2/3/4 within +-0.5%). At the c32-realistic uniq 96 the pair
// runs 1461-1474 GB/s-wt (92-93% of the 1.58 TB/s practical roof) vs the
// shipped 1280/1309 (81/83%) - pair time -11.7%..-17.3% across uniq 24-128,
// ~= the full ~1.1 ms/tick prize at marlin's per-byte rate. Persistent
// grids and deep 1-CTA rings stay dead; these are ordinary
// multi-wave grids, 2-stage ring, 2 CTAs/SM.
//
// TILED PLANE LAYOUT (the engine's nvf4_moe_upload_tiled contract; both
// nemotron planes tile exactly - 1856 = 29*64 rows, 2688 = 42*64 - so no
// pad bytes exist and DRAM traffic is identical to row-major):
//   data : [e][rt][ks][piece 2][row 64][16 B]   (2048 B per (e,rt,ks) block)
//   scale: [e][rt][ks][row 64][4 B]             (256 B per block)
// rt = 64-row output tile, ks = 64-element K block (piece = its 32-element
// half). A K-chunk fetch for one row tile is one contiguous span (data) plus
// one for scales; ldmatrix reads rows at 16 B stride (4-bank steps, conflict
// free, better than the row-major WROW=144 stride).
//
// Three consumer classes, one layout (a tiled plane must never exist without
// every class able to read it - the lm_head has_nvf4_tm law):
//   - _st  (BM=8, 64 rows, 128 thr): the DECODE pair. pairs/uniq at c32 is
//     ~2.4, so 32-wide blocks are ~7.5% live; 8-wide quarters the Y/fq
//     staging and the mma count. Fed by pd_moe_align_bm(bm=8).
//   - _stw (BM=32, 128 rows, 256 thr): the PREFILL pair, same geometry class
//     as the shipped bs pair (full 32-token blocks; BM=8 there would re-read
//     each expert strip per 8 tokens). Same align as today.
//   - _mtt: the r=1 serial-decode GEMV twins (W4A16 f32 class, same numeric
//     class as the mt pair). CTA per 16-row group, K split by warp, lane =
//     (row, piece) so a warp's ks-block read is 512 B contiguous. The
//     grouping (not the math) differs from mt, so its gates are the mt
//     class: rel-to-rms vs the row-major twin + determinism bit-gated.
// Accumulate order in _st/_stw is the shipped pair's (kt asc, k64 asc)
// verbatim -> both are BIT-EXACT vs the row-major bs pair on identical
// routing; the unit gates lean on that.
// All six launchers are cc12-only (block-scale mma / e4m3 decode on the
// tiled layout); every other die keeps row-major planes and the shipped
// consumers, which is what the engine's layout election checks.

#define PD_STT_KB 8u                    // KC = 256 elements per ring stage
#define PD_STT_K64S (PD_STT_KB / 2u)    // ks blocks per stage
#define PD_STT_TDATA (PD_STT_K64S * 2048u)   // per-64-row-tile stage bytes
#define PD_STT_TSCL (PD_STT_K64S * 256u)
#define PD_STT_YROW (16u + PD_STT_KB * 16u)
#define PD_STT_STAGE(BM, ROWS) \
    ((ROWS / 64u) * (PD_STT_TDATA + PD_STT_TSCL) + (BM) * PD_STT_YROW)
#define PD_STT_SMEM(BM, ROWS) (2u * PD_STT_STAGE(BM, ROWS))

// Sorted-tile expert up + squared-relu + nvf4 requantize over the tiled
// plane. Geometry: CTA = ROWS output rows x BM token columns, ROWS*2
// threads, each warp owns one 16-row m-tile across all BM columns. 2-stage
// cp.async ring with the scale bytes folded into the ring (pd_cpa4p) - the
// stage is contiguous so there is no strided-scale tax to defer.
template <uint32_t BM, uint32_t ROWS, uint32_t PF = 0u, bool YB = false>
__global__ void __launch_bounds__(ROWS * 2u, (BM >= 128u ? 1 : 2)) pd_nv4st_up_kernel(
    const uint8_t* __restrict__ data, const uint8_t* __restrict__ scale,
    const float* __restrict__ scale2, const uint32_t* __restrict__ sorted_row,
    const uint32_t* __restrict__ block_expert, const uint8_t* __restrict__ xq,
    const uint8_t* __restrict__ xs, uint8_t* __restrict__ fq,
    uint8_t* __restrict__ fs, uint32_t in_dim, uint32_t ff) {
#if PD_BS_OK
    constexpr uint32_t THR = ROWS * 2u;
    constexpr uint32_t YROW = PD_STT_YROW;
    constexpr uint32_t K64S = PD_STT_K64S;
    constexpr uint32_t STAGE = PD_STT_STAGE(BM, ROWS);
    // YB: the expert block on gridDim.y (row tiles fastest), so a PAD-padded
    // block list's empty blocks are whole trailing y-slices instead of being
    // interleaved into every row tile's x-run (bench/nv4st_gb10_bench.cu).
    const uint32_t blk = YB ? blockIdx.y : blockIdx.x;
    const uint32_t e = block_expert[blk];
    if (e == PD_MOE_PAD) return;
    const uint32_t row_base = (YB ? blockIdx.x : blockIdx.y) * ROWS;

    extern __shared__ unsigned char pd_stt_sh[];
    __shared__ uint32_t tok[BM];

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, tq = lane & 3u;
    const uint32_t i0 = warp * 16u;
    const uint32_t n_kb = in_dim >> 5;
    const uint32_t n_k16 = in_dim >> 4;
    const uint32_t nks = in_dim >> 6;
    const uint32_t nrt = ff >> 6;   // exact: the layout requires ff % 64 == 0
    const uint32_t nk = (in_dim + PD_STT_KB * 32u - 1u) / (PD_STT_KB * 32u);
    const size_t trt0 = ((size_t)e * nrt + (row_base >> 6)) * nks;

    if (tid < BM) tok[tid] = sorted_row[(size_t)blk * BM + tid];
    __syncthreads();

    float acc[BM / 8u][4] = {};

    #define PD_STT_ISSUE_W(dst, kt)                                                   \
        for (uint32_t u = tid; u < (ROWS / 64u) * (PD_STT_TDATA / 16u); u += THR) {   \
            const uint32_t h = u / (PD_STT_TDATA / 16u), v = u % (PD_STT_TDATA / 16u);\
            const uint32_t ks = (kt) * K64S + v / 128u;                               \
            const bool ok = ks < nks && (row_base >> 6) + h < nrt;                    \
            pd_cp_async16(                                                            \
                (int*)((dst) + h * (PD_STT_TDATA + PD_STT_TSCL) + v * 16u),           \
                data + (trt0 + h * (size_t)nks + (kt) * K64S) * 2048u + v * 16u,      \
                ok);                                                                  \
        }                                                                             \
        for (uint32_t u = tid; u < (ROWS / 64u) * (PD_STT_TSCL / 4u); u += THR) {     \
            const uint32_t h = u / (PD_STT_TSCL / 4u), v = u % (PD_STT_TSCL / 4u);    \
            const uint32_t ks = (kt) * K64S + v / 64u;                                \
            const bool ok = ks < nks && (row_base >> 6) + h < nrt;                    \
            pd_cpa4p((dst) + h * (PD_STT_TDATA + PD_STT_TSCL) + PD_STT_TDATA + v * 4u,\
                     scale + (trt0 + h * (size_t)nks + (kt) * K64S) * 256u + v * 4u,  \
                     ok);                                                             \
        }
    #define PD_STT_ISSUE_Y(dst, kt)                                                   \
        for (uint32_t u = tid; u < BM * PD_STT_KB; u += THR) {                        \
            const uint32_t col = u / PD_STT_KB, seg = u % PD_STT_KB;                  \
            const uint32_t r = tok[col];                                              \
            const bool ok = r != PD_MOE_PAD && (kt) * PD_STT_KB + seg < n_kb;         \
            pd_cp_async16((int*)((dst) + col * YROW + 16u + seg * 16u),               \
                          xq + ((size_t)(ok ? r : 0u) * in_dim >> 1) +                \
                              (kt) * (PD_STT_KB * 16u) + seg * 16u,                   \
                          ok);                                                        \
        }                                                                             \
        for (uint32_t u = tid; u < BM * (PD_STT_KB / 2u); u += THR) {                 \
            const uint32_t col = u / (PD_STT_KB / 2u), q = u % (PD_STT_KB / 2u);      \
            const uint32_t r = tok[col];                                              \
            const bool ok = r != PD_MOE_PAD &&                                        \
                            (kt) * (PD_STT_KB * 2u) + q * 4u + 4u <= n_k16;           \
            pd_cpa4p((dst) + col * YROW + q * 4u,                                     \
                     xs + (size_t)(ok ? r : 0u) * n_k16 +                             \
                         (kt) * (PD_STT_KB * 2u) + q * 4u,                            \
                     ok);                                                             \
        }
    #define PD_STT_WBUF(s) (pd_stt_sh + ((s) & 1u) * STAGE)
    #define PD_STT_YBUF(s) (PD_STT_WBUF(s) + (ROWS / 64u) * (PD_STT_TDATA + PD_STT_TSCL))
    // PF > 0: one thread presents ring stage `kp` (this CTA's contiguous
    // data + scale spans, one 8 KB and one 1 KB per 64-row tile) to L2 PF
    // stages ahead of the cp.async issue - the f4tn move on the skinny
    // pair. A hint: no data dependency, bit-exact by construction.
    #define PD_STT_PREFETCH(kp)                                                       \
        if constexpr (PF > 0u) {                                                      \
            if (tid == 0u && (kp) < nk) {                                             \
                const uint32_t nkl = nks - (kp) * K64S;                               \
                const uint32_t nks_pf = nkl < K64S ? nkl : K64S;                      \
                for (uint32_t h = 0; h < ROWS / 64u; ++h) {                           \
                    if ((row_base >> 6) + h < nrt) {                                  \
                        const size_t b = trt0 + h * (size_t)nks + (kp) * K64S;        \
                        pd_l2_prefetch_bulk(data + b * 2048u, nks_pf * 2048u);       \
                        pd_l2_prefetch_bulk(scale + b * 256u, nks_pf * 256u);        \
                    }                                                                 \
                }                                                                     \
            }                                                                         \
        }

    PD_STT_ISSUE_W(PD_STT_WBUF(0), 0u)
    PD_STT_ISSUE_Y(PD_STT_YBUF(0), 0u)
    asm volatile("cp.async.commit_group;");
    if constexpr (PF > 0u) {
        for (uint32_t p = 1u; p <= PF; ++p) PD_STT_PREFETCH(p)
    }
    for (uint32_t kt = 0; kt < nk; ++kt) {
        unsigned char* tw = PD_STT_WBUF(kt);
        unsigned char* ty = PD_STT_YBUF(kt);
        if (kt + 1u < nk) {
            PD_STT_ISSUE_W(PD_STT_WBUF(kt + 1u), kt + 1u)
            PD_STT_ISSUE_Y(PD_STT_YBUF(kt + 1u), kt + 1u)
            PD_STT_PREFETCH(kt + 1u + PF)
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();

        uint32_t am[K64S][4], sa[K64S];
        const uint32_t rl = ((lane >> 3) & 1u) * 8u + (lane & 7u);
        const uint32_t pl = lane >> 4;
        const uint32_t rs = (tq & 1u) ? (i0 & 63u) + g + 8u : (i0 & 63u) + g;
        const uint32_t h = i0 >> 6;
        #pragma unroll
        for (uint32_t k64 = 0; k64 < K64S; ++k64) {
            pd_ldm_x4(am[k64], tw + h * (PD_STT_TDATA + PD_STT_TSCL) +
                                   (k64 * 2u + pl) * 1024u +
                                   ((i0 & 63u) + rl) * 16u);
            sa[k64] = *(const uint32_t*)(tw + h * (PD_STT_TDATA + PD_STT_TSCL) +
                                         PD_STT_TDATA + k64 * 256u + rs * 4u);
        }
        #pragma unroll
        for (uint32_t j0 = 0; j0 < BM; j0 += 8u) {
            uint32_t bm[2u * K64S];
            #pragma unroll
            for (uint32_t q = 0; q < PD_STT_KB / 4u; ++q)
                pd_ldm_x4(bm + q * 4u, ty + (j0 + (lane & 7u)) * YROW + 16u +
                                           q * 64u + (lane >> 3) * 16u);
            const unsigned char* ysr = ty + (j0 + g) * YROW;
            #pragma unroll
            for (uint32_t k64 = 0; k64 < K64S; ++k64) {
                const uint32_t sb = *(const uint32_t*)(ysr + k64 * 4u);
                pd_nv4_mma(acc[j0 >> 3], am[k64][0], am[k64][1], am[k64][2],
                           am[k64][3], bm[k64 * 2u], bm[k64 * 2u + 1u],
                           sa[k64], sb);
            }
        }
        __syncthreads();
    }
    #undef PD_STT_ISSUE_W
    #undef PD_STT_ISSUE_Y
    #undef PD_STT_WBUF
    #undef PD_STT_YBUF
    #undef PD_STT_PREFETCH

    // epilogue: the shipped bs kernel's per-16-along-ff quantize, verbatim
    // math - one 16-row block per warp, 8 token columns per j0 group.
    const float s2 = scale2[e];
    const uint32_t tmask = 0x11111111u << tq;
    const uint32_t rb = row_base + i0;
    #pragma unroll
    for (uint32_t j0 = 0; j0 < BM; j0 += 8u) {
        #pragma unroll
        for (uint32_t qc = 0; qc < 2u; ++qc) {
            const uint32_t c = j0 + 2u * tq + qc;
            const bool pad = tok[c] == PD_MOE_PAD;
            const float a0 = acc[j0 >> 3][qc] * s2;
            const float a1 = acc[j0 >> 3][qc + 2u] * s2;
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
                const size_t srow = (size_t)blk * BM + c;
                fq[srow * (ff >> 1) + (rb >> 1) + g] =
                    (unsigned char)(lo | (hi << 4));
                if (g == 0)
                    fs[srow * (ff >> 4) + (rb >> 4)] = (unsigned char)sbyte;
            }
        }
    }
#else
    (void)data; (void)scale; (void)scale2; (void)sorted_row; (void)block_expert;
    (void)xq; (void)xs; (void)fq; (void)fs; (void)in_dim; (void)ff;
#endif
}

// Down + weighted scatter over the tiled plane. B = fq/fs by sorted position
// ([nb*BM, ff/2], the bs contract at this BM); W = down plane, K = ff.
template <uint32_t BM, uint32_t ROWS, uint32_t PF = 0u, bool YB = false>
__global__ void __launch_bounds__(ROWS * 2u, (BM >= 128u ? 1 : 2)) pd_nv4st_dn_kernel(
    const uint8_t* __restrict__ data, const uint8_t* __restrict__ scale,
    const float* __restrict__ scale2, const uint32_t* __restrict__ sorted_row,
    const uint32_t* __restrict__ sorted_slot,
    const uint32_t* __restrict__ block_expert, const float* __restrict__ topk_w,
    const uint8_t* __restrict__ fq, const uint8_t* __restrict__ fs,
    float* __restrict__ part, uint32_t ff, uint32_t embd, uint32_t kw,
    uint32_t np, uint32_t slot_off, uint32_t vf = 0xffffffffu) {
#if PD_BS_OK
    constexpr uint32_t THR = ROWS * 2u;
    constexpr uint32_t YROW = PD_STT_YROW;
    constexpr uint32_t K64S = PD_STT_K64S;
    constexpr uint32_t STAGE = PD_STT_STAGE(BM, ROWS);
    // YB: the expert block on gridDim.y (row tiles fastest), so a PAD-padded
    // block list's empty blocks are whole trailing y-slices instead of being
    // interleaved into every row tile's x-run (bench/nv4st_gb10_bench.cu).
    const uint32_t blk = YB ? blockIdx.y : blockIdx.x;
    const uint32_t e = block_expert[blk];
    if (e == PD_MOE_PAD) return;
    const uint32_t row_base = (YB ? blockIdx.x : blockIdx.y) * ROWS;

    extern __shared__ unsigned char pd_stt_sh[];

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, tq = lane & 3u;
    const uint32_t i0 = warp * 16u;
    const uint32_t n_kb = ff >> 5;
    const uint32_t n_k16 = ff >> 4;
    const uint32_t nks = ff >> 6;
    const uint32_t nrt = embd >> 6;
    const uint32_t nk = (ff + PD_STT_KB * 32u - 1u) / (PD_STT_KB * 32u);
    const size_t trt0 = ((size_t)e * nrt + (row_base >> 6)) * nks;

    float acc[BM / 8u][4] = {};

    #define PD_STT_ISSUE_W(dst, kt)                                                   \
        for (uint32_t u = tid; u < (ROWS / 64u) * (PD_STT_TDATA / 16u); u += THR) {   \
            const uint32_t h = u / (PD_STT_TDATA / 16u), v = u % (PD_STT_TDATA / 16u);\
            const uint32_t ks = (kt) * K64S + v / 128u;                               \
            const bool ok = ks < nks && (row_base >> 6) + h < nrt;                    \
            pd_cp_async16(                                                            \
                (int*)((dst) + h * (PD_STT_TDATA + PD_STT_TSCL) + v * 16u),           \
                data + (trt0 + h * (size_t)nks + (kt) * K64S) * 2048u + v * 16u,      \
                ok);                                                                  \
        }                                                                             \
        for (uint32_t u = tid; u < (ROWS / 64u) * (PD_STT_TSCL / 4u); u += THR) {     \
            const uint32_t h = u / (PD_STT_TSCL / 4u), v = u % (PD_STT_TSCL / 4u);    \
            const uint32_t ks = (kt) * K64S + v / 64u;                                \
            const bool ok = ks < nks && (row_base >> 6) + h < nrt;                    \
            pd_cpa4p((dst) + h * (PD_STT_TDATA + PD_STT_TSCL) + PD_STT_TDATA + v * 4u,\
                     scale + (trt0 + h * (size_t)nks + (kt) * K64S) * 256u + v * 4u,  \
                     ok);                                                             \
        }
    #define PD_STT_ISSUE_Y(dst, kt)                                                   \
        for (uint32_t u = tid; u < BM * PD_STT_KB; u += THR) {                        \
            const uint32_t col = u / PD_STT_KB, seg = u % PD_STT_KB;                  \
            const bool ok = (kt) * PD_STT_KB + seg < n_kb;                            \
            pd_cp_async16((int*)((dst) + col * YROW + 16u + seg * 16u),               \
                          fq + ((size_t)blk * BM + col) * (size_t)(ff >> 1) +         \
                              (kt) * (PD_STT_KB * 16u) + seg * 16u,                   \
                          ok);                                                        \
        }                                                                             \
        for (uint32_t u = tid; u < BM * (PD_STT_KB / 2u); u += THR) {                 \
            const uint32_t col = u / (PD_STT_KB / 2u), q = u % (PD_STT_KB / 2u);      \
            const bool ok = (kt) * (PD_STT_KB * 2u) + q * 4u + 4u <= n_k16;           \
            pd_cpa4p((dst) + col * YROW + q * 4u,                                     \
                     fs + ((size_t)blk * BM + col) * n_k16 +                          \
                         (kt) * (PD_STT_KB * 2u) + q * 4u,                            \
                     ok);                                                             \
        }
    #define PD_STT_WBUF(s) (pd_stt_sh + ((s) & 1u) * STAGE)
    #define PD_STT_YBUF(s) (PD_STT_WBUF(s) + (ROWS / 64u) * (PD_STT_TDATA + PD_STT_TSCL))
    // PF > 0: one thread presents ring stage `kp` (this CTA's contiguous
    // data + scale spans, one 8 KB and one 1 KB per 64-row tile) to L2 PF
    // stages ahead of the cp.async issue - the f4tn move on the skinny
    // pair. A hint: no data dependency, bit-exact by construction.
    #define PD_STT_PREFETCH(kp)                                                       \
        if constexpr (PF > 0u) {                                                      \
            if (tid == 0u && (kp) < nk) {                                             \
                const uint32_t nkl = nks - (kp) * K64S;                               \
                const uint32_t nks_pf = nkl < K64S ? nkl : K64S;                      \
                for (uint32_t h = 0; h < ROWS / 64u; ++h) {                           \
                    if ((row_base >> 6) + h < nrt) {                                  \
                        const size_t b = trt0 + h * (size_t)nks + (kp) * K64S;        \
                        pd_l2_prefetch_bulk(data + b * 2048u, nks_pf * 2048u);       \
                        pd_l2_prefetch_bulk(scale + b * 256u, nks_pf * 256u);        \
                    }                                                                 \
                }                                                                     \
            }                                                                         \
        }

    PD_STT_ISSUE_W(PD_STT_WBUF(0), 0u)
    PD_STT_ISSUE_Y(PD_STT_YBUF(0), 0u)
    asm volatile("cp.async.commit_group;");
    if constexpr (PF > 0u) {
        for (uint32_t p = 1u; p <= PF; ++p) PD_STT_PREFETCH(p)
    }
    for (uint32_t kt = 0; kt < nk; ++kt) {
        unsigned char* tw = PD_STT_WBUF(kt);
        unsigned char* ty = PD_STT_YBUF(kt);
        if (kt + 1u < nk) {
            PD_STT_ISSUE_W(PD_STT_WBUF(kt + 1u), kt + 1u)
            PD_STT_ISSUE_Y(PD_STT_YBUF(kt + 1u), kt + 1u)
            PD_STT_PREFETCH(kt + 1u + PF)
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();

        uint32_t am[K64S][4], sa[K64S];
        const uint32_t rl = ((lane >> 3) & 1u) * 8u + (lane & 7u);
        const uint32_t pl = lane >> 4;
        const uint32_t rs = (tq & 1u) ? (i0 & 63u) + g + 8u : (i0 & 63u) + g;
        const uint32_t h = i0 >> 6;
        #pragma unroll
        for (uint32_t k64 = 0; k64 < K64S; ++k64) {
            pd_ldm_x4(am[k64], tw + h * (PD_STT_TDATA + PD_STT_TSCL) +
                                   (k64 * 2u + pl) * 1024u +
                                   ((i0 & 63u) + rl) * 16u);
            sa[k64] = *(const uint32_t*)(tw + h * (PD_STT_TDATA + PD_STT_TSCL) +
                                         PD_STT_TDATA + k64 * 256u + rs * 4u);
        }
        #pragma unroll
        for (uint32_t j0 = 0; j0 < BM; j0 += 8u) {
            uint32_t bm[2u * K64S];
            #pragma unroll
            for (uint32_t q = 0; q < PD_STT_KB / 4u; ++q)
                pd_ldm_x4(bm + q * 4u, ty + (j0 + (lane & 7u)) * YROW + 16u +
                                           q * 64u + (lane >> 3) * 16u);
            const unsigned char* ysr = ty + (j0 + g) * YROW;
            #pragma unroll
            for (uint32_t k64 = 0; k64 < K64S; ++k64) {
                const uint32_t sb = *(const uint32_t*)(ysr + k64 * 4u);
                pd_nv4_mma(acc[j0 >> 3], am[k64][0], am[k64][1], am[k64][2],
                           am[k64][3], bm[k64 * 2u], bm[k64 * 2u + 1u],
                           sa[k64], sb);
            }
        }
        __syncthreads();
    }
    #undef PD_STT_ISSUE_W
    #undef PD_STT_ISSUE_Y
    #undef PD_STT_WBUF
    #undef PD_STT_YBUF
    #undef PD_STT_PREFETCH

    // weighted scatter to the per-(token, slot) partial rows - the shipped
    // down epilogue with the j0 groups walked 8 columns at a time. Token rows
    // from `vf` on (slots 956-957: a pass's prompt rows) round their partials
    // to bf16 at the store, into a bf16 plane laid from the f32 plane's row
    // vf on - half the bytes it writes and the combine reads back; the rows
    // before vf (a round's verify rows) keep the f32 partials.
    const float s2 = scale2[e];
    #pragma unroll
    for (uint32_t j0 = 0; j0 < BM; j0 += 8u) {
        #pragma unroll
        for (uint32_t qc = 0; qc < 2u; ++qc) {
            const uint32_t c = j0 + 2u * tq + qc;
            const uint32_t t = sorted_row[(size_t)blk * BM + c];
            if (t == PD_MOE_PAD) continue;
            const uint32_t slt = sorted_slot[(size_t)blk * BM + c];
            const float w = (topk_w ? topk_w[(size_t)t * kw + slt] : 1.0f) * s2;
            const uint32_t r0 = row_base + i0 + g;
            const uint32_t r8 = r0 + 8u;
            if (t >= vf) {
                __nv_bfloat16* prow =
                    reinterpret_cast<__nv_bfloat16*>(part + (size_t)vf * np * embd) +
                    ((size_t)(t - vf) * np + slt + slot_off) * embd;
                if (r0 < embd) prow[r0] = __float2bfloat16_rn(acc[j0 >> 3][qc] * w);
                if (r8 < embd) prow[r8] = __float2bfloat16_rn(acc[j0 >> 3][qc + 2u] * w);
            } else {
                float* prow = part + ((size_t)t * np + slt + slot_off) * embd;
                if (r0 < embd) prow[r0] = acc[j0 >> 3][qc] * w;
                if (r8 < embd) prow[r8] = acc[j0 >> 3][qc + 2u] * w;
            }
        }
    }
#else
    (void)data; (void)scale; (void)scale2; (void)sorted_row; (void)sorted_slot;
    (void)block_expert; (void)topk_w; (void)fq; (void)fs; (void)part; (void)ff;
    (void)embd; (void)kw; (void)np; (void)slot_off; (void)vf;
#endif
}

// ---- r=1 serial-decode twins over the tiled plane (the mt class) ----------
// Same numeric class as pd_nvf4_moe_up_relu2_mt / down_part (W4A16: f32
// activations, pd_nvf4_dot4w per element quad, relu^2 / weighted partial
// epilogues). The GROUPING differs: CTA per 16-row group (a task-per-row
// GEMV on the tiled layout would read 16 B at 1 KB stride - the reason
// gemv_batch_tf regrouped the lm_head). 128 threads; warp w owns ks blocks
// w, w+4, ...; lane = (row = lane>>1, piece = lane&1) so each warp's
// ks-block read is 512 B contiguous. Per-lane K order ascends; combine is
// fixed-order (piece pair -> ascending warps), deterministic.
__global__ void pd_nv4st_mt_up_kernel(
    const uint8_t* __restrict__ rdata, const uint8_t* __restrict__ rscale,
    const float* __restrict__ rscale2, const uint8_t* __restrict__ sdata,
    const uint8_t* __restrict__ sscale, const float* __restrict__ sscale2,
    const uint32_t* __restrict__ idx, const float* __restrict__ x,
    float* __restrict__ act, uint32_t in_dim, uint32_t ff_r, uint32_t ff_s,
    uint32_t k) {
#if PD_NV4_OK
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t gr_r = ff_r >> 4, gr_s = ff_s >> 4;
    const uint32_t task = blockIdx.x;
    if (task >= k * gr_r + gr_s) return;
    const uint8_t* base;
    const uint8_t* sbase;
    float s2;
    uint32_t rg, nrt, aoff;
    if (task < k * gr_r) {
        const uint32_t slot = task / gr_r;
        rg = task - slot * gr_r;
        const uint32_t e = idx[slot];
        nrt = ff_r >> 6;
        const uint32_t nks = in_dim >> 6;
        base = rdata + (size_t)e * nrt * nks * 2048u;
        sbase = rscale + (size_t)e * nrt * nks * 256u;
        s2 = rscale2[e];
        aoff = slot * ff_r;
    } else {
        rg = task - k * gr_r;
        nrt = ff_s >> 6;
        base = sdata;
        sbase = sscale;
        s2 = sscale2[0];
        aoff = k * ff_r;
    }
    const uint32_t nks = in_dim >> 6;
    const uint32_t rt = rg >> 2, r0 = (rg & 3u) * 16u;
    const uint32_t r6 = r0 + (lane >> 1), p = lane & 1u;

    float acc = 0.0f;
    for (uint32_t ks = warp; ks < nks; ks += 4u) {
        const size_t blk = (size_t)rt * nks + ks;
        const uint4 wv = *reinterpret_cast<const uint4*>(
            base + blk * 2048u + p * 1024u + r6 * 16u);
        const uint32_t sw = *reinterpret_cast<const uint32_t*>(
            sbase + blk * 256u + r6 * 4u);
        const uint32_t s0 = (sw >> (p * 16u)) & 0xFFu;
        const uint32_t s1 = (sw >> (p * 16u + 8u)) & 0xFFu;
        const uint32_t e0 = ks * 64u + p * 32u;
        acc += pd_nvf4_dot4w(wv.x & 0xFFFFu, s0, x, e0);
        acc += pd_nvf4_dot4w(wv.x >> 16, s0, x, e0 + 4u);
        acc += pd_nvf4_dot4w(wv.y & 0xFFFFu, s0, x, e0 + 8u);
        acc += pd_nvf4_dot4w(wv.y >> 16, s0, x, e0 + 12u);
        acc += pd_nvf4_dot4w(wv.z & 0xFFFFu, s1, x, e0 + 16u);
        acc += pd_nvf4_dot4w(wv.z >> 16, s1, x, e0 + 20u);
        acc += pd_nvf4_dot4w(wv.w & 0xFFFFu, s1, x, e0 + 24u);
        acc += pd_nvf4_dot4w(wv.w >> 16, s1, x, e0 + 28u);
    }
    // piece pair first (lane, lane^1), then ascending warps through shared -
    // fixed summation order per output row.
    acc += __shfl_xor_sync(0xffffffffu, acc, 1);
    __shared__ float psum[4][16];
    if (p == 0) psum[warp][lane >> 1] = acc;
    __syncthreads();
    if (threadIdx.x < 16u) {
        const float total = ((psum[0][threadIdx.x] + psum[1][threadIdx.x]) +
                             psum[2][threadIdx.x]) + psum[3][threadIdx.x];
        const float v = fmaxf(total * s2, 0.0f);
        act[aoff + rg * 16u + threadIdx.x] = v * v;
    }
#else
    (void)rdata; (void)rscale; (void)rscale2; (void)sdata; (void)sscale;
    (void)sscale2; (void)idx; (void)x; (void)act; (void)in_dim; (void)ff_r;
    (void)ff_s; (void)k;
#endif
}

__global__ void pd_nv4st_mt_dn_kernel(
    const uint8_t* __restrict__ rdata, const uint8_t* __restrict__ rscale,
    const float* __restrict__ rscale2, const uint8_t* __restrict__ sdata,
    const uint8_t* __restrict__ sscale, const float* __restrict__ sscale2,
    const uint32_t* __restrict__ idx, const float* __restrict__ topk_w,
    const float* __restrict__ act, float* __restrict__ part, uint32_t ff_r,
    uint32_t ff_s, uint32_t embd, uint32_t k) {
#if PD_NV4_OK
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t gr_e = embd >> 4;
    const uint32_t task = blockIdx.x;
    if (task >= (k + 1u) * gr_e) return;
    const uint32_t slot = task / gr_e, rg = task - slot * gr_e;
    const uint8_t* base;
    const uint8_t* sbase;
    const float* xrow;
    float w;
    uint32_t kk;
    const uint32_t nrt = embd >> 6;
    if (slot < k) {
        const uint32_t e = idx[slot];
        w = topk_w[slot] * rscale2[e];
        kk = ff_r;
        const uint32_t nks = kk >> 6;
        base = rdata + (size_t)e * nrt * nks * 2048u;
        sbase = rscale + (size_t)e * nrt * nks * 256u;
        xrow = act + (size_t)slot * ff_r;
    } else {
        w = sscale2[0];
        kk = ff_s;
        base = sdata;
        sbase = sscale;
        xrow = act + (size_t)k * ff_r;
    }
    const uint32_t nks = kk >> 6;
    const uint32_t rt = rg >> 2, r0 = (rg & 3u) * 16u;
    const uint32_t r6 = r0 + (lane >> 1), p = lane & 1u;

    float psm = 0.0f;
    for (uint32_t ks = warp; ks < nks; ks += 4u) {
        const size_t blk = (size_t)rt * nks + ks;
        const uint4 wv = *reinterpret_cast<const uint4*>(
            base + blk * 2048u + p * 1024u + r6 * 16u);
        const uint32_t sw = *reinterpret_cast<const uint32_t*>(
            sbase + blk * 256u + r6 * 4u);
        const uint32_t s0 = (sw >> (p * 16u)) & 0xFFu;
        const uint32_t s1 = (sw >> (p * 16u + 8u)) & 0xFFu;
        const uint32_t e0 = ks * 64u + p * 32u;
        psm += pd_nvf4_dot4w(wv.x & 0xFFFFu, s0, xrow, e0);
        psm += pd_nvf4_dot4w(wv.x >> 16, s0, xrow, e0 + 4u);
        psm += pd_nvf4_dot4w(wv.y & 0xFFFFu, s0, xrow, e0 + 8u);
        psm += pd_nvf4_dot4w(wv.y >> 16, s0, xrow, e0 + 12u);
        psm += pd_nvf4_dot4w(wv.z & 0xFFFFu, s1, xrow, e0 + 16u);
        psm += pd_nvf4_dot4w(wv.z >> 16, s1, xrow, e0 + 20u);
        psm += pd_nvf4_dot4w(wv.w & 0xFFFFu, s1, xrow, e0 + 24u);
        psm += pd_nvf4_dot4w(wv.w >> 16, s1, xrow, e0 + 28u);
    }
    // the down_part fold shape: w per lane before the combine
    float acc = w * psm;
    acc += __shfl_xor_sync(0xffffffffu, acc, 1);
    __shared__ float psum[4][16];
    if (p == 0) psum[warp][lane >> 1] = acc;
    __syncthreads();
    if (threadIdx.x < 16u)
        part[(size_t)slot * embd + rg * 16u + threadIdx.x] =
            0.0f + (((psum[0][threadIdx.x] + psum[1][threadIdx.x]) +
                     psum[2][threadIdx.x]) + psum[3][threadIdx.x]);
#else
    (void)rdata; (void)rscale; (void)rscale2; (void)sdata; (void)sscale;
    (void)sscale2; (void)idx; (void)topk_w; (void)act; (void)part; (void)ff_r;
    (void)ff_s; (void)embd; (void)k;
#endif
}

// ---- launchers (ABI 472-477; cc12-only, see exports.cuh) -------------------

// Skinny decode pair: BM=8 blocks from pd_moe_align_bm(bm=8). Same argument
// contract as the bs pair at BM=8 strides (sorted_row/sorted_slot are
// [nb*8], fq/fs are [nb*8, ff/16th]).
// ---- the PREFILL twins on a bulk-async weight stream (the f4t recipe) -------
// pd_nv4st_{up,dn}_kernel<32,128> stream each stage's expert bytes through
// 256 threads x 16 B cp.async (4.5 issues per thread per stage) and a 2-deep
// ring; on GB10 that binds them at 70% of the byte roof and 32 TF/s
// (bench/nv4st_gb10_bench.cu 1024, uniq 114: pair 3.80 ms per layer) - the
// dense f4t lane's diagnosis exactly (the load path,
// not the layout, not the tensor pipe). The tiled plane already lays every
// stage out as one contiguous span per 64-row tile (8 KB of nibbles + 1 KB
// of scales), so the fix needs no tensor map at all: one thread issues four
// `cp.async.bulk` copies per stage into an mbarrier ring with expect_tx, the
// TMA unit moves the bytes, and the 255 other threads have nothing to issue
// but the 32-token activation rows (whose scale rows are 168 / 116 B apart,
// not 16-B aligned, so those stay on cp.async as before). The stage bytes
// land at the same offsets the cp.async loop put them, the consumer
// fragment / mma code below is the shipped one verbatim, so the outputs are
// BIT-EXACT vs pd_nv4st_{up,dn}_kernel<32,128> (the bench byte-compares fq,
// fs and part). ST-deep ring (2 = two CTAs per SM at 46 KB, 3/4 = one CTA);
// the waits are executed by every thread (no lone lane spins - session 22's
// divergence trap), the issuing lane rejoins at the __syncthreads before
// the collective loads.
#define PD_STB_STAGE PD_STT_STAGE(32u, 128u)
#define PD_STB_SMEM(ST) ((ST) * PD_STB_STAGE + 64u)

template <uint32_t ST, uint32_t PROBE = 0u>
__global__ void __launch_bounds__(256, (ST <= 2u ? 2 : 1)) pd_nv4stb_up_kernel(
    const uint8_t* __restrict__ data, const uint8_t* __restrict__ scale,
    const float* __restrict__ scale2, const uint32_t* __restrict__ sorted_row,
    const uint32_t* __restrict__ block_expert, const uint8_t* __restrict__ xq,
    const uint8_t* __restrict__ xs, uint8_t* __restrict__ fq,
    uint8_t* __restrict__ fs, uint32_t in_dim, uint32_t ff) {
#if PD_BS_OK
    constexpr uint32_t BM = 32u, ROWS = 128u, THR = 256u;
    constexpr uint32_t YROW = PD_STT_YROW;
    constexpr uint32_t K64S = PD_STT_K64S;
    constexpr uint32_t STAGE = PD_STB_STAGE;
    constexpr uint32_t HSTRIDE = PD_STT_TDATA + PD_STT_TSCL;
    const uint32_t blk = blockIdx.x;
    const uint32_t e = block_expert[blk];
    if (e == PD_MOE_PAD) return;
    const uint32_t row_base = blockIdx.y * ROWS;

    extern __shared__ __align__(128) unsigned char pd_stb_sh[];
    __shared__ uint32_t tok[BM];
    uint64_t* mbar = (uint64_t*)(pd_stb_sh + ST * STAGE);

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, tq = lane & 3u;
    const uint32_t i0 = warp * 16u;
    const uint32_t n_kb = in_dim >> 5;
    const uint32_t n_k16 = in_dim >> 4;
    const uint32_t nks = in_dim >> 6;
    const uint32_t nrt = ff >> 6;
    const uint32_t nk = (in_dim + PD_STT_KB * 32u - 1u) / (PD_STT_KB * 32u);
    const size_t trt0 = ((size_t)e * nrt + (row_base >> 6)) * nks;
    const uint32_t m0 = (uint32_t)__cvta_generic_to_shared(mbar);

    if (tid < BM) tok[tid] = sorted_row[(size_t)blk * BM + tid];
    if (tid == 0u) {
        #pragma unroll
        for (uint32_t s = 0; s < ST; ++s)
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(m0 + s * 8u));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();

    float acc[BM / 8u][4] = {};

    // W stage kt into slot kt % ST: expect the stage's bytes, then one bulk
    // copy per span (data, scales) per valid 64-row tile. Called by tid 0.
    auto issue_w = [&](uint32_t kt) {
        const uint32_t s = kt % ST;
        const uint32_t m = m0 + s * 8u;
        const uint32_t nkl = nks - kt * K64S;
        const uint32_t nkc = nkl < K64S ? nkl : K64S;
        uint32_t tx = 0u;
        #pragma unroll
        for (uint32_t h = 0; h < ROWS / 64u; ++h)
            if ((row_base >> 6) + h < nrt) tx += nkc * (2048u + 256u);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(m), "r"(tx));
        #pragma unroll
        for (uint32_t h = 0; h < ROWS / 64u; ++h) {
            if ((row_base >> 6) + h >= nrt) continue;
            const size_t b = trt0 + h * (size_t)nks + kt * K64S;
            const uint32_t wd = (uint32_t)__cvta_generic_to_shared(pd_stb_sh + s * STAGE + h * HSTRIDE);
            asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                         " [%0], [%1], %2, [%3];" ::"r"(wd), "l"(data + b * 2048u),
                         "r"(nkc * 2048u), "r"(m) : "memory");
            asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                         " [%0], [%1], %2, [%3];" ::"r"(wd + PD_STT_TDATA), "l"(scale + b * 256u),
                         "r"(nkc * 256u), "r"(m) : "memory");
        }
    };
    #define PD_STB_ISSUE_Y(dst, kt)                                                   \
        for (uint32_t u = tid; u < BM * PD_STT_KB; u += THR) {                        \
            const uint32_t col = u / PD_STT_KB, seg = u % PD_STT_KB;                  \
            const uint32_t r = tok[col];                                              \
            const bool ok = r != PD_MOE_PAD && (kt) * PD_STT_KB + seg < n_kb;         \
            pd_cp_async16((int*)((dst) + col * YROW + 16u + seg * 16u),               \
                          xq + ((size_t)(ok ? r : 0u) * in_dim >> 1) +                \
                              (kt) * (PD_STT_KB * 16u) + seg * 16u,                   \
                          ok);                                                        \
        }                                                                             \
        for (uint32_t u = tid; u < BM * (PD_STT_KB / 2u); u += THR) {                 \
            const uint32_t col = u / (PD_STT_KB / 2u), q = u % (PD_STT_KB / 2u);      \
            const uint32_t r = tok[col];                                              \
            const bool ok = r != PD_MOE_PAD &&                                        \
                            (kt) * (PD_STT_KB * 2u) + q * 4u + 4u <= n_k16;           \
            pd_cpa4p((dst) + col * YROW + q * 4u,                                     \
                     xs + (size_t)(ok ? r : 0u) * n_k16 +                             \
                         (kt) * (PD_STT_KB * 2u) + q * 4u,                            \
                     ok);                                                             \
        }
    #define PD_STB_WBUF(s) (pd_stb_sh + (s) * STAGE)
    #define PD_STB_YBUF(s) (PD_STB_WBUF(s) + (ROWS / 64u) * HSTRIDE)

    // prologue: stages 0 .. ST-2 (one cp.async group per stage, empty groups
    // included so the wait_group count stays one per stage)
    #pragma unroll
    for (uint32_t p = 0; p + 1u < ST; ++p) {
        if (p < nk) {
            if (tid == 0u) issue_w(p);
            PD_STB_ISSUE_Y(PD_STB_YBUF(p), p)
        }
        asm volatile("cp.async.commit_group;");
    }
    uint32_t fph = 0u;   // full parities, bit s
    for (uint32_t kt = 0; kt < nk; ++kt) {
        const uint32_t s = kt % ST;
        {
            const uint32_t kn = kt + ST - 1u;
            if (kn < nk) {
                if (tid == 0u) issue_w(kn);
                PD_STB_ISSUE_Y(PD_STB_YBUF(kn % ST), kn)
            }
            asm volatile("cp.async.commit_group;");
        }
        asm volatile("cp.async.wait_group %0;" ::"n"(ST - 1u));
        asm volatile(
            "{\n\t.reg .pred P;\n"
            "PD_STB_WAIT_%=:\n\t"
            "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
            "@!P bra PD_STB_WAIT_%=;\n\t}" ::"r"(m0 + s * 8u), "r"((fph >> s) & 1u) : "memory");
        fph ^= 1u << s;
        __syncthreads();

        unsigned char* tw = PD_STB_WBUF(s);
        unsigned char* ty = PD_STB_YBUF(s);
        uint32_t am[K64S][4], sa[K64S];
        const uint32_t rl = ((lane >> 3) & 1u) * 8u + (lane & 7u);
        const uint32_t pl = lane >> 4;
        const uint32_t rs = (tq & 1u) ? (i0 & 63u) + g + 8u : (i0 & 63u) + g;
        const uint32_t h = i0 >> 6;
        #pragma unroll
        for (uint32_t k64 = 0; k64 < K64S; ++k64) {
            pd_ldm_x4(am[k64], tw + h * HSTRIDE + (k64 * 2u + pl) * 1024u +
                                   ((i0 & 63u) + rl) * 16u);
            sa[k64] = *(const uint32_t*)(tw + h * HSTRIDE + PD_STT_TDATA + k64 * 256u + rs * 4u);
        }
        #pragma unroll
        for (uint32_t j0 = 0; j0 < BM; j0 += 8u) {
            uint32_t bm[2u * K64S];
            #pragma unroll
            for (uint32_t q = 0; q < PD_STT_KB / 4u; ++q)
                pd_ldm_x4(bm + q * 4u, ty + (j0 + (lane & 7u)) * YROW + 16u +
                                           q * 64u + (lane >> 3) * 16u);
            const unsigned char* ysr = ty + (j0 + g) * YROW;
            #pragma unroll
            for (uint32_t k64 = 0; k64 < K64S; ++k64) {
                const uint32_t sb = *(const uint32_t*)(ysr + k64 * 4u);
                if constexpr (PROBE != 2u)
                    pd_nv4_mma(acc[j0 >> 3], am[k64][0], am[k64][1], am[k64][2],
                               am[k64][3], bm[k64 * 2u], bm[k64 * 2u + 1u],
                               sa[k64], sb);
                else
                    acc[j0 >> 3][0] += __uint_as_float(sb ^ sa[k64] ^ am[k64][0] ^ bm[k64 * 2u]);
            }
        }
        __syncthreads();
    }
    #undef PD_STB_ISSUE_Y
    #undef PD_STB_WBUF
    #undef PD_STB_YBUF

    // epilogue: pd_nv4st_up_kernel's quantize, verbatim - but LANDED through
    // smem: the shipped twin writes 8 bytes of fq per token per warp-store
    // (a 32-B sector a quarter full, 16 such per token per CTA) where the
    // CTA's whole contribution to a token is 64 contiguous bytes of fq and 8
    // of fs. The no-epilogue probe put the pair's epilogues at 25% of its
    // time on GB10 (bench/nv4st_gb10_bench.cu 1024: 3.82 -> 2.89 ms), so
    // the tile is staged and each token's run goes out as four 16-B stores.
    if constexpr (PROBE == 1u) {
        if (acc[0][0] == 12345.678f) fs[0] = 1u;   // keep the mainloop live
        return;
    }
    const float s2 = scale2[e];
    const uint32_t tmask = 0x11111111u << tq;
    const uint32_t rb = row_base + i0;
    unsigned char* stq = pd_stb_sh;                 // [BM][64 B]  (the ring is free)
    unsigned char* sts = pd_stb_sh + BM * 64u;      // [BM][8 B]
    #pragma unroll
    for (uint32_t j0 = 0; j0 < BM; j0 += 8u) {
        #pragma unroll
        for (uint32_t qc = 0; qc < 2u; ++qc) {
            const uint32_t c = j0 + 2u * tq + qc;
            const bool pad = tok[c] == PD_MOE_PAD;
            const float a0 = acc[j0 >> 3][qc] * s2;
            const float a1 = acc[j0 >> 3][qc + 2u] * s2;
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
            const uint32_t mm = (g & 3u) * 2u;
            const uint32_t lo0 = __shfl_sync(0xffffffffu, n0, mm * 4u + tq);
            const uint32_t hi0 = __shfl_sync(0xffffffffu, n0, (mm + 1u) * 4u + tq);
            const uint32_t lo1 = __shfl_sync(0xffffffffu, n1, mm * 4u + tq);
            const uint32_t hi1 = __shfl_sync(0xffffffffu, n1, (mm + 1u) * 4u + tq);
            const uint32_t lo = (g < 4u) ? lo0 : lo1;
            const uint32_t hi = (g < 4u) ? hi0 : hi1;
            stq[c * 64u + (i0 >> 1) + g] = (unsigned char)(lo | (hi << 4));
            if (g == 0) sts[c * 8u + (i0 >> 4)] = (unsigned char)sbyte;
        }
    }
    __syncthreads();
    // the landing: token c's 64 B of fq as four 16-B stores (rows past ff are
    // whole 64-row tiles - the second tile of the last CTA - so mask by tile)
    const uint32_t nvalid = (row_base + ROWS <= ff) ? ROWS : (ff > row_base ? ff - row_base : 0u);
    for (uint32_t u = tid; u < BM * 4u; u += THR) {
        const uint32_t c = u >> 2, q = u & 3u;
        if (tok[c] == PD_MOE_PAD || q * 32u >= nvalid) continue;
        const size_t srow = (size_t)blk * BM + c;
        *(uint4*)(fq + srow * (ff >> 1) + (row_base >> 1) + q * 16u) =
            *(const uint4*)(stq + c * 64u + q * 16u);
    }
    for (uint32_t u = tid; u < BM * 2u; u += THR) {
        const uint32_t c = u >> 1, q = u & 1u;
        if (tok[c] == PD_MOE_PAD || q * 64u >= nvalid) continue;
        const size_t srow = (size_t)blk * BM + c;
        *(uint32_t*)(fs + srow * (ff >> 4) + (row_base >> 4) + q * 4u) =
            *(const uint32_t*)(sts + c * 8u + q * 4u);
    }
#else
    (void)data; (void)scale; (void)scale2; (void)sorted_row; (void)block_expert;
    (void)xq; (void)xs; (void)fq; (void)fs; (void)in_dim; (void)ff;
#endif
}

template <uint32_t ST, uint32_t PROBE = 0u>
__global__ void __launch_bounds__(256, (ST <= 2u ? 2 : 1)) pd_nv4stb_dn_kernel(
    const uint8_t* __restrict__ data, const uint8_t* __restrict__ scale,
    const float* __restrict__ scale2, const uint32_t* __restrict__ sorted_row,
    const uint32_t* __restrict__ sorted_slot,
    const uint32_t* __restrict__ block_expert, const float* __restrict__ topk_w,
    const uint8_t* __restrict__ fq, const uint8_t* __restrict__ fs,
    float* __restrict__ part, uint32_t ff, uint32_t embd, uint32_t kw,
    uint32_t np, uint32_t slot_off) {
#if PD_BS_OK
    constexpr uint32_t BM = 32u, ROWS = 128u, THR = 256u;
    constexpr uint32_t YROW = PD_STT_YROW;
    constexpr uint32_t K64S = PD_STT_K64S;
    constexpr uint32_t STAGE = PD_STB_STAGE;
    constexpr uint32_t HSTRIDE = PD_STT_TDATA + PD_STT_TSCL;
    const uint32_t blk = blockIdx.x;
    const uint32_t e = block_expert[blk];
    if (e == PD_MOE_PAD) return;
    const uint32_t row_base = blockIdx.y * ROWS;

    extern __shared__ __align__(128) unsigned char pd_stb_sh[];
    __shared__ uint32_t tok[BM], tsl[BM];
    __shared__ float tw_s[BM];
    uint64_t* mbar = (uint64_t*)(pd_stb_sh + ST * STAGE);

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, tq = lane & 3u;
    const uint32_t i0 = warp * 16u;
    const uint32_t n_kb = ff >> 5;
    const uint32_t n_k16 = ff >> 4;
    const uint32_t nks = ff >> 6;
    const uint32_t nrt = embd >> 6;
    const uint32_t nk = (ff + PD_STT_KB * 32u - 1u) / (PD_STT_KB * 32u);
    const size_t trt0 = ((size_t)e * nrt + (row_base >> 6)) * nks;
    const uint32_t m0 = (uint32_t)__cvta_generic_to_shared(mbar);

    if (tid < BM) {
        const uint32_t t = sorted_row[(size_t)blk * BM + tid];
        const uint32_t slt = sorted_slot[(size_t)blk * BM + tid];
        tok[tid] = t;
        tsl[tid] = slt;
        tw_s[tid] = (t != PD_MOE_PAD && topk_w) ? topk_w[(size_t)t * kw + slt] : 1.0f;
    }
    if (tid == 0u) {
        #pragma unroll
        for (uint32_t s = 0; s < ST; ++s)
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(m0 + s * 8u));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();

    float acc[BM / 8u][4] = {};

    auto issue_w = [&](uint32_t kt) {
        const uint32_t s = kt % ST;
        const uint32_t m = m0 + s * 8u;
        const uint32_t nkl = nks - kt * K64S;
        const uint32_t nkc = nkl < K64S ? nkl : K64S;
        uint32_t tx = 0u;
        #pragma unroll
        for (uint32_t h = 0; h < ROWS / 64u; ++h)
            if ((row_base >> 6) + h < nrt) tx += nkc * (2048u + 256u);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(m), "r"(tx));
        #pragma unroll
        for (uint32_t h = 0; h < ROWS / 64u; ++h) {
            if ((row_base >> 6) + h >= nrt) continue;
            const size_t b = trt0 + h * (size_t)nks + kt * K64S;
            const uint32_t wd = (uint32_t)__cvta_generic_to_shared(pd_stb_sh + s * STAGE + h * HSTRIDE);
            asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                         " [%0], [%1], %2, [%3];" ::"r"(wd), "l"(data + b * 2048u),
                         "r"(nkc * 2048u), "r"(m) : "memory");
            asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                         " [%0], [%1], %2, [%3];" ::"r"(wd + PD_STT_TDATA), "l"(scale + b * 256u),
                         "r"(nkc * 256u), "r"(m) : "memory");
        }
    };
    #define PD_STB_ISSUE_Y(dst, kt)                                                   \
        for (uint32_t u = tid; u < BM * PD_STT_KB; u += THR) {                        \
            const uint32_t col = u / PD_STT_KB, seg = u % PD_STT_KB;                  \
            const bool ok = (kt) * PD_STT_KB + seg < n_kb;                            \
            pd_cp_async16((int*)((dst) + col * YROW + 16u + seg * 16u),               \
                          fq + ((size_t)blk * BM + col) * (size_t)(ff >> 1) +         \
                              (kt) * (PD_STT_KB * 16u) + seg * 16u,                   \
                          ok);                                                        \
        }                                                                             \
        for (uint32_t u = tid; u < BM * (PD_STT_KB / 2u); u += THR) {                 \
            const uint32_t col = u / (PD_STT_KB / 2u), q = u % (PD_STT_KB / 2u);      \
            const bool ok = (kt) * (PD_STT_KB * 2u) + q * 4u + 4u <= n_k16;           \
            pd_cpa4p((dst) + col * YROW + q * 4u,                                     \
                     fs + ((size_t)blk * BM + col) * n_k16 +                          \
                         (kt) * (PD_STT_KB * 2u) + q * 4u,                            \
                     ok);                                                             \
        }
    #define PD_STB_WBUF(s) (pd_stb_sh + (s) * STAGE)
    #define PD_STB_YBUF(s) (PD_STB_WBUF(s) + (ROWS / 64u) * HSTRIDE)

    #pragma unroll
    for (uint32_t p = 0; p + 1u < ST; ++p) {
        if (p < nk) {
            if (tid == 0u) issue_w(p);
            PD_STB_ISSUE_Y(PD_STB_YBUF(p), p)
        }
        asm volatile("cp.async.commit_group;");
    }
    uint32_t fph = 0u;
    for (uint32_t kt = 0; kt < nk; ++kt) {
        const uint32_t s = kt % ST;
        {
            const uint32_t kn = kt + ST - 1u;
            if (kn < nk) {
                if (tid == 0u) issue_w(kn);
                PD_STB_ISSUE_Y(PD_STB_YBUF(kn % ST), kn)
            }
            asm volatile("cp.async.commit_group;");
        }
        asm volatile("cp.async.wait_group %0;" ::"n"(ST - 1u));
        asm volatile(
            "{\n\t.reg .pred P;\n"
            "PD_STB_WAIT_%=:\n\t"
            "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
            "@!P bra PD_STB_WAIT_%=;\n\t}" ::"r"(m0 + s * 8u), "r"((fph >> s) & 1u) : "memory");
        fph ^= 1u << s;
        __syncthreads();

        unsigned char* tw = PD_STB_WBUF(s);
        unsigned char* ty = PD_STB_YBUF(s);
        uint32_t am[K64S][4], sa[K64S];
        const uint32_t rl = ((lane >> 3) & 1u) * 8u + (lane & 7u);
        const uint32_t pl = lane >> 4;
        const uint32_t rs = (tq & 1u) ? (i0 & 63u) + g + 8u : (i0 & 63u) + g;
        const uint32_t h = i0 >> 6;
        #pragma unroll
        for (uint32_t k64 = 0; k64 < K64S; ++k64) {
            pd_ldm_x4(am[k64], tw + h * HSTRIDE + (k64 * 2u + pl) * 1024u +
                                   ((i0 & 63u) + rl) * 16u);
            sa[k64] = *(const uint32_t*)(tw + h * HSTRIDE + PD_STT_TDATA + k64 * 256u + rs * 4u);
        }
        #pragma unroll
        for (uint32_t j0 = 0; j0 < BM; j0 += 8u) {
            uint32_t bm[2u * K64S];
            #pragma unroll
            for (uint32_t q = 0; q < PD_STT_KB / 4u; ++q)
                pd_ldm_x4(bm + q * 4u, ty + (j0 + (lane & 7u)) * YROW + 16u +
                                           q * 64u + (lane >> 3) * 16u);
            const unsigned char* ysr = ty + (j0 + g) * YROW;
            #pragma unroll
            for (uint32_t k64 = 0; k64 < K64S; ++k64) {
                const uint32_t sb = *(const uint32_t*)(ysr + k64 * 4u);
                if constexpr (PROBE != 2u)
                    pd_nv4_mma(acc[j0 >> 3], am[k64][0], am[k64][1], am[k64][2],
                               am[k64][3], bm[k64 * 2u], bm[k64 * 2u + 1u],
                               sa[k64], sb);
                else
                    acc[j0 >> 3][0] += __uint_as_float(sb ^ sa[k64] ^ am[k64][0] ^ bm[k64 * 2u]);
            }
        }
        __syncthreads();
    }
    #undef PD_STB_ISSUE_Y
    #undef PD_STB_WBUF
    #undef PD_STB_YBUF

    // epilogue: pd_nv4st_dn_kernel's math, verbatim - but the per-token row,
    // slot and weight come from the smem table filled at the prologue (the
    // shipped twin issues three dependent global loads per lane per column
    // here, 48 per lane, each a DRAM round trip the mainloop no longer hides).
    if constexpr (PROBE == 1u) {
        if (acc[0][0] == 12345.678f) part[0] = 1.0f;
        return;
    }
    // Landed through smem: the shipped twin's per-lane scatter writes each
    // token's 512-B partial run as sixteen 32-B pieces (the f32 partial
    // planes are 66 MB per layer at prefill width, and the write roof of
    // LPDDR5X is the lower one); staged, each token row leaves as 32 x 16 B.
    const float s2 = scale2[e];
    float* stile = (float*)pd_stb_sh;   // [BM][ROWS + 4] f32 (the ring is free)
    constexpr uint32_t SROW = ROWS + 4u;
    #pragma unroll
    for (uint32_t j0 = 0; j0 < BM; j0 += 8u) {
        #pragma unroll
        for (uint32_t qc = 0; qc < 2u; ++qc) {
            const uint32_t c = j0 + 2u * tq + qc;
            const float w = tw_s[c] * s2;
            stile[c * SROW + i0 + g] = acc[j0 >> 3][qc] * w;
            stile[c * SROW + i0 + g + 8u] = acc[j0 >> 3][qc + 2u] * w;
        }
    }
    __syncthreads();
    const uint32_t nvalid = (row_base + ROWS <= embd) ? ROWS : (embd > row_base ? embd - row_base : 0u);
    for (uint32_t u = tid; u < BM * (ROWS / 4u); u += THR) {
        const uint32_t c = u / (ROWS / 4u), q = u % (ROWS / 4u);
        const uint32_t t = tok[c];
        if (t == PD_MOE_PAD || q * 4u >= nvalid) continue;
        float* prow = part + ((size_t)t * np + tsl[c] + slot_off) * embd;
        *(float4*)(prow + row_base + q * 4u) = *(const float4*)(stile + c * SROW + q * 4u);
    }
#else
    (void)data; (void)scale; (void)scale2; (void)sorted_row; (void)sorted_slot;
    (void)block_expert; (void)topk_w; (void)fq; (void)fs; (void)part; (void)ff;
    (void)embd; (void)kw; (void)np; (void)slot_off;
#endif
}

// Ring depth for the bulk-async prefill twins: `PADDOCK_NVF4_STB_ST` = 2, 3
// or 4 arms them (dev-only probe); unset or 0 = the shipped cp.async pair on
// every die. NEUTRAL on GB10 (2026-09-11): with
// the load path on bulk copies and the landings staged, the pair sits at
// the same 3.78 ms per layer at 1024 rows / uniq 114 as the cp.async twins
// (bit-exact, every arm byte-compared), because the twins' binder is not
// the loads - the no-epilogue probe runs the mainloop at 92% of the roof -
// but the down twin's f32 partial-plane writes (66 MB per layer at prefill
// width, at the write rate of LPDDR5X) plus the up twin's quantize tail.
// Kept as the probe vehicle (PROBE = 1 no epilogue, 2 no mma) and for the
// day the partial planes change class. One attribute call per
// instantiation for the > 48 KB rings.
static uint32_t pd_stb_st(void) {
    static const uint32_t st = [] {
        const char* v = pd_env("PADDOCK_NVF4_STB_ST");
        if (!v) return 0u;
        const int n = atoi(v);
        return n <= 0 ? 0u : (n >= 4 ? 4u : (n >= 3 ? 3u : 2u));
    }();
    return st;
}
template <uint32_t ST>
static int pd_stb_up_go(const uint8_t* data, const uint8_t* scale, const float* scale2,
                        const uint32_t* sr, const uint32_t* be, const uint8_t* xq,
                        const uint8_t* xs, uint8_t* fq, uint8_t* fs, uint32_t in_dim,
                        uint32_t ff, uint32_t nb, cudaStream_t st) {
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute(pd_nv4stb_up_kernel<ST>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)PD_STB_SMEM(ST));
        attr = true;
    }
    dim3 grid(nb, (ff + 127u) >> 7);
    pd_nv4stb_up_kernel<ST><<<grid, 256u, PD_STB_SMEM(ST), st>>>(
        data, scale, scale2, sr, be, xq, xs, fq, fs, in_dim, ff);
    return pd_launch_status();
}
template <uint32_t ST>
static int pd_stb_dn_go(const uint8_t* data, const uint8_t* scale, const float* scale2,
                        const uint32_t* sr, const uint32_t* ss, const uint32_t* be,
                        const float* tkw, const uint8_t* fq, const uint8_t* fs,
                        float* part, uint32_t ff, uint32_t embd, uint32_t kw, uint32_t np,
                        uint32_t slot_off, uint32_t nb, cudaStream_t st) {
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute(pd_nv4stb_dn_kernel<ST>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)PD_STB_SMEM(ST));
        attr = true;
    }
    dim3 grid(nb, (embd + 127u) >> 7);
    pd_nv4stb_dn_kernel<ST><<<grid, 256u, PD_STB_SMEM(ST), st>>>(
        data, scale, scale2, sr, ss, be, tkw, fq, fs, part, ff, embd, kw, np, slot_off);
    return pd_launch_status();
}

// L2-prefetch depth for the decode skinny pair (PF template stages ahead of
// the cp.async ring), a dev-only probe: `PADDOCK_NVF4_ST_PF` = 2 or 4, else
// 0 on every die. FALSIFIED as a lever on GB10 (2026-09-11):
// bench/nv4st_gb10_bench.cu has the shipped pair at 93-97% of
// the die's 240 GB/s at every unique-expert count from 6 to 64 (DRAM-cold
// expert rotation), and PF 2/4 within noise to -3%. The serve census's
// 155-170 GB/s reading was the cohort's routing (unique experts per
// layer-tick), not the kernel; the engine's PAD-padded launch extent
// (moe_live_blocks_bm8) costs nothing either (the PADx/PADy arms). Kept as
// a probe because it is bit-exact by construction and the bench
// byte-compares every arm.
static uint32_t pd_stt_pf(void) {
    static const uint32_t pf = [] {
        const char* v = pd_env("PADDOCK_NVF4_ST_PF");
        if (!v) return 0u;
        const int n = atoi(v);
        return n >= 4 ? 4u : (n >= 2 ? 2u : 0u);
    }();
    return pf;
}

PD_EXPORT
int pd_nvf4_moe_up_relu2_st(const void* data, const void* scale,
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
    if (nb == 0) return 0;
    if ((in_dim & 63u) != 0 || (ff & 63u) != 0) return cudaErrorInvalidValue;
    dim3 grid(nb, ff >> 6);
    switch (pd_stt_pf()) {
    case 2u:
        pd_nv4st_up_kernel<8u, 64u, 2u>
            <<<grid, 128u, PD_STT_SMEM(8u, 64u), (cudaStream_t)stream>>>(
                (const uint8_t*)data, (const uint8_t*)scale, (const float*)scale2,
                (const uint32_t*)sorted_row, (const uint32_t*)block_expert,
                (const uint8_t*)xq, (const uint8_t*)xs, (uint8_t*)fq, (uint8_t*)fs,
                in_dim, ff);
        break;
    case 4u:
        pd_nv4st_up_kernel<8u, 64u, 4u>
            <<<grid, 128u, PD_STT_SMEM(8u, 64u), (cudaStream_t)stream>>>(
                (const uint8_t*)data, (const uint8_t*)scale, (const float*)scale2,
                (const uint32_t*)sorted_row, (const uint32_t*)block_expert,
                (const uint8_t*)xq, (const uint8_t*)xs, (uint8_t*)fq, (uint8_t*)fs,
                in_dim, ff);
        break;
    default:
        pd_nv4st_up_kernel<8u, 64u>
            <<<grid, 128u, PD_STT_SMEM(8u, 64u), (cudaStream_t)stream>>>(
                (const uint8_t*)data, (const uint8_t*)scale, (const float*)scale2,
                (const uint32_t*)sorted_row, (const uint32_t*)block_expert,
                (const uint8_t*)xq, (const uint8_t*)xs, (uint8_t*)fq, (uint8_t*)fs,
                in_dim, ff);
    }
    return pd_launch_status();
#endif
}

PD_EXPORT
int pd_nvf4_moe_down_st(const void* data, const void* scale, const void* scale2,
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
    if (nb == 0) return 0;
    if ((ff & 63u) != 0 || (embd & 63u) != 0) return cudaErrorInvalidValue;
    dim3 grid(nb, embd >> 6);
    switch (pd_stt_pf()) {
    case 2u:
        pd_nv4st_dn_kernel<8u, 64u, 2u>
            <<<grid, 128u, PD_STT_SMEM(8u, 64u), (cudaStream_t)stream>>>(
                (const uint8_t*)data, (const uint8_t*)scale, (const float*)scale2,
                (const uint32_t*)sorted_row, (const uint32_t*)sorted_slot,
                (const uint32_t*)block_expert, (const float*)topk_w,
                (const uint8_t*)fq, (const uint8_t*)fs, (float*)part, ff, embd, kw,
                np, slot_off);
        break;
    case 4u:
        pd_nv4st_dn_kernel<8u, 64u, 4u>
            <<<grid, 128u, PD_STT_SMEM(8u, 64u), (cudaStream_t)stream>>>(
                (const uint8_t*)data, (const uint8_t*)scale, (const float*)scale2,
                (const uint32_t*)sorted_row, (const uint32_t*)sorted_slot,
                (const uint32_t*)block_expert, (const float*)topk_w,
                (const uint8_t*)fq, (const uint8_t*)fs, (float*)part, ff, embd, kw,
                np, slot_off);
        break;
    default:
        pd_nv4st_dn_kernel<8u, 64u>
            <<<grid, 128u, PD_STT_SMEM(8u, 64u), (cudaStream_t)stream>>>(
                (const uint8_t*)data, (const uint8_t*)scale, (const float*)scale2,
                (const uint32_t*)sorted_row, (const uint32_t*)sorted_slot,
                (const uint32_t*)block_expert, (const float*)topk_w,
                (const uint8_t*)fq, (const uint8_t*)fs, (float*)part, ff, embd, kw,
                np, slot_off);
    }
    return pd_launch_status();
#endif
}

// 956-957: pd_nvf4_moe_down_st / _stw with token rows from `vf` on folding
// bf16 partials (see the kernel's epilogue; pd_moe_slot_combine_bf16 folds
// that plane, pd_moe_slot_combine the f32 rows before it). The skinny and
// wide twins both take it, so a prompt class keeps one partial type a row
// at every width.
PD_EXPORT
int pd_nvf4_moe_down_st_bf16(const void* data, const void* scale, const void* scale2,
                             const void* sorted_row, const void* sorted_slot,
                             const void* block_expert, const void* topk_w, const void* fq,
                             const void* fs, void* part, uint32_t ff, uint32_t embd, uint32_t kw,
                             uint32_t np, uint32_t slot_off, uint32_t nb, uint32_t vf,
                             void* stream) {
#ifndef PD_BS_HOST
    (void)data; (void)scale; (void)scale2; (void)sorted_row; (void)sorted_slot;
    (void)block_expert; (void)topk_w; (void)fq; (void)fs; (void)part; (void)ff;
    (void)embd; (void)kw; (void)np; (void)slot_off; (void)nb; (void)vf; (void)stream;
    return cudaErrorNotSupported;
#else
    if (nb == 0) return 0;
    if ((ff & 63u) != 0 || (embd & 63u) != 0) return cudaErrorInvalidValue;
    dim3 grid(nb, embd >> 6);
    pd_nv4st_dn_kernel<8u, 64u>
        <<<grid, 128u, PD_STT_SMEM(8u, 64u), (cudaStream_t)stream>>>(
            (const uint8_t*)data, (const uint8_t*)scale, (const float*)scale2,
            (const uint32_t*)sorted_row, (const uint32_t*)sorted_slot,
            (const uint32_t*)block_expert, (const float*)topk_w, (const uint8_t*)fq,
            (const uint8_t*)fs, (float*)part, ff, embd, kw, np, slot_off, vf);
    return pd_launch_status();
#endif
}

PD_EXPORT
int pd_nvf4_moe_down_stw_bf16(const void* data, const void* scale, const void* scale2,
                              const void* sorted_row, const void* sorted_slot,
                              const void* block_expert, const void* topk_w, const void* fq,
                              const void* fs, void* part, uint32_t ff, uint32_t embd,
                              uint32_t kw, uint32_t np, uint32_t slot_off, uint32_t nb,
                              uint32_t vf, void* stream) {
#ifndef PD_BS_HOST
    (void)data; (void)scale; (void)scale2; (void)sorted_row; (void)sorted_slot;
    (void)block_expert; (void)topk_w; (void)fq; (void)fs; (void)part; (void)ff;
    (void)embd; (void)kw; (void)np; (void)slot_off; (void)nb; (void)vf; (void)stream;
    return cudaErrorNotSupported;
#else
    if (nb == 0) return 0;
    if ((ff & 63u) != 0 || (embd & 63u) != 0 || nb > 65535u) return cudaErrorInvalidValue;
    dim3 grid((embd + 127u) >> 7, nb);
    pd_nv4st_dn_kernel<32u, 128u, 0u, true>
        <<<grid, 256u, PD_STT_SMEM(32u, 128u), (cudaStream_t)stream>>>(
            (const uint8_t*)data, (const uint8_t*)scale, (const float*)scale2,
            (const uint32_t*)sorted_row, (const uint32_t*)sorted_slot,
            (const uint32_t*)block_expert, (const float*)topk_w, (const uint8_t*)fq,
            (const uint8_t*)fs, (float*)part, ff, embd, kw, np, slot_off, vf);
    return pd_launch_status();
#endif
}

// Wide prefill pair: BM=32 blocks (the shipped align), 128-row CTAs over
// the tiled plane. Bit-exact vs the row-major bs pair on identical routing.
PD_EXPORT
int pd_nvf4_moe_up_relu2_stw(const void* data, const void* scale,
                             const void* scale2, const void* sorted_row,
                             const void* block_expert, const void* xq,
                             const void* xs, void* fq, void* fs,
                             uint32_t in_dim, uint32_t ff, uint32_t nb,
                             void* stream) {
#ifndef PD_BS_HOST
    (void)data; (void)scale; (void)scale2; (void)sorted_row; (void)block_expert;
    (void)xq; (void)xs; (void)fq; (void)fs; (void)in_dim; (void)ff; (void)nb;
    (void)stream;
    return cudaErrorNotSupported;
#else
    if (nb == 0) return 0;
    if ((in_dim & 63u) != 0 || (ff & 63u) != 0) return cudaErrorInvalidValue;
    switch (pd_stb_st()) {
    case 2u: return pd_stb_up_go<2u>((const uint8_t*)data, (const uint8_t*)scale, (const float*)scale2,
                                     (const uint32_t*)sorted_row, (const uint32_t*)block_expert,
                                     (const uint8_t*)xq, (const uint8_t*)xs, (uint8_t*)fq, (uint8_t*)fs,
                                     in_dim, ff, nb, (cudaStream_t)stream);
    case 3u: return pd_stb_up_go<3u>((const uint8_t*)data, (const uint8_t*)scale, (const float*)scale2,
                                     (const uint32_t*)sorted_row, (const uint32_t*)block_expert,
                                     (const uint8_t*)xq, (const uint8_t*)xs, (uint8_t*)fq, (uint8_t*)fs,
                                     in_dim, ff, nb, (cudaStream_t)stream);
    case 4u: return pd_stb_up_go<4u>((const uint8_t*)data, (const uint8_t*)scale, (const float*)scale2,
                                     (const uint32_t*)sorted_row, (const uint32_t*)block_expert,
                                     (const uint8_t*)xq, (const uint8_t*)xs, (uint8_t*)fq, (uint8_t*)fs,
                                     in_dim, ff, nb, (cudaStream_t)stream);
    default: break;
    }
    // row tiles fastest (YB): a block's CTAs run back to back, so its
    // gathered activation rows and its expert's plane stay in L2 across
    // them instead of a full sweep of every block per row tile -
    // bit-identical (scheduling order only), -5..-16% on this twin at
    // 200..2400 rows (GB10)
    if (nb > 65535u) return cudaErrorInvalidValue;
    dim3 grid((ff + 127u) >> 7, nb);
    pd_nv4st_up_kernel<32u, 128u, 0u, true>
        <<<grid, 256u, PD_STT_SMEM(32u, 128u), (cudaStream_t)stream>>>(
            (const uint8_t*)data, (const uint8_t*)scale, (const float*)scale2,
            (const uint32_t*)sorted_row, (const uint32_t*)block_expert,
            (const uint8_t*)xq, (const uint8_t*)xs, (uint8_t*)fq, (uint8_t*)fs,
            in_dim, ff);
    return pd_launch_status();
#endif
}

PD_EXPORT
int pd_nvf4_moe_down_stw(const void* data, const void* scale,
                         const void* scale2, const void* sorted_row,
                         const void* sorted_slot, const void* block_expert,
                         const void* topk_w, const void* fq, const void* fs,
                         void* part, uint32_t ff, uint32_t embd, uint32_t kw,
                         uint32_t np, uint32_t slot_off, uint32_t nb,
                         void* stream) {
#ifndef PD_BS_HOST
    (void)data; (void)scale; (void)scale2; (void)sorted_row; (void)sorted_slot;
    (void)block_expert; (void)topk_w; (void)fq; (void)fs; (void)part; (void)ff;
    (void)embd; (void)kw; (void)np; (void)slot_off; (void)nb; (void)stream;
    return cudaErrorNotSupported;
#else
    if (nb == 0) return 0;
    if ((ff & 63u) != 0 || (embd & 63u) != 0) return cudaErrorInvalidValue;
    switch (pd_stb_st()) {
    case 2u: return pd_stb_dn_go<2u>((const uint8_t*)data, (const uint8_t*)scale, (const float*)scale2,
                                     (const uint32_t*)sorted_row, (const uint32_t*)sorted_slot,
                                     (const uint32_t*)block_expert, (const float*)topk_w,
                                     (const uint8_t*)fq, (const uint8_t*)fs, (float*)part, ff, embd,
                                     kw, np, slot_off, nb, (cudaStream_t)stream);
    case 3u: return pd_stb_dn_go<3u>((const uint8_t*)data, (const uint8_t*)scale, (const float*)scale2,
                                     (const uint32_t*)sorted_row, (const uint32_t*)sorted_slot,
                                     (const uint32_t*)block_expert, (const float*)topk_w,
                                     (const uint8_t*)fq, (const uint8_t*)fs, (float*)part, ff, embd,
                                     kw, np, slot_off, nb, (cudaStream_t)stream);
    case 4u: return pd_stb_dn_go<4u>((const uint8_t*)data, (const uint8_t*)scale, (const float*)scale2,
                                     (const uint32_t*)sorted_row, (const uint32_t*)sorted_slot,
                                     (const uint32_t*)block_expert, (const float*)topk_w,
                                     (const uint8_t*)fq, (const uint8_t*)fs, (float*)part, ff, embd,
                                     kw, np, slot_off, nb, (cudaStream_t)stream);
    default: break;
    }
    // row tiles fastest (YB): the down twin's input is per PAIR (six rows a
    // token, ~15 MB of fq at a 2.4K-row pass), and with blocks fastest
    // every one of the 21 row-tile sweeps re-streamed all of it past the
    // 24 MB L2 beside the expert planes - the twin ran at twice the up
    // twin's time with its mma removed. Back to back, a block's 30 KB of
    // fq serves its 21 tiles from L2: dn 4.45 -> 2.60 ms at 2400 rows,
    // 2.76 -> 1.92 at 1024, 1.79 -> 1.63 at 200 (GB10, uniq 128),
    // bit-identical (scheduling order only)
    if (nb > 65535u) return cudaErrorInvalidValue;
    dim3 grid((embd + 127u) >> 7, nb);
    pd_nv4st_dn_kernel<32u, 128u, 0u, true>
        <<<grid, 256u, PD_STT_SMEM(32u, 128u), (cudaStream_t)stream>>>(
            (const uint8_t*)data, (const uint8_t*)scale, (const float*)scale2,
            (const uint32_t*)sorted_row, (const uint32_t*)sorted_slot,
            (const uint32_t*)block_expert, (const float*)topk_w,
            (const uint8_t*)fq, (const uint8_t*)fs, (float*)part, ff, embd, kw,
            np, slot_off);
    return pd_launch_status();
#endif
}

// r=1 twins: same argument contract as the mt pair, tiled planes.
PD_EXPORT
int pd_nvf4_moe_up_relu2_mtt(const void* rdata, const void* rscale,
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
    if ((in_dim & 63u) != 0 || (ff_r & 63u) != 0 || (ff_s & 63u) != 0)
        return cudaErrorInvalidValue;
    const uint32_t grid = k * (ff_r >> 4) + (ff_s >> 4);
    pd_nv4st_mt_up_kernel<<<grid, 128u, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)rdata, (const uint8_t*)rscale, (const float*)rscale2,
        (const uint8_t*)sdata, (const uint8_t*)sscale, (const float*)sscale2,
        (const uint32_t*)idx, (const float*)x, (float*)act, in_dim, ff_r, ff_s,
        k);
    return pd_launch_status();
#endif
}

PD_EXPORT
int pd_nvf4_moe_down_part_tt(const void* rdata, const void* rscale,
                             const void* rscale2, const void* sdata,
                             const void* sscale, const void* sscale2,
                             const void* idx, const void* topk_w,
                             const void* act, void* part, uint32_t ff_r,
                             uint32_t ff_s, uint32_t embd, uint32_t k,
                             void* stream) {
#ifndef PD_BS_HOST
    (void)rdata; (void)rscale; (void)rscale2; (void)sdata; (void)sscale;
    (void)sscale2; (void)idx; (void)topk_w; (void)act; (void)part; (void)ff_r;
    (void)ff_s; (void)embd; (void)k; (void)stream;
    return cudaErrorNotSupported;
#else
    if (embd == 0 || k == 0) return 0;
    if ((ff_r & 63u) != 0 || (ff_s & 63u) != 0 || (embd & 63u) != 0)
        return cudaErrorInvalidValue;
    const uint32_t grid = (k + 1u) * (embd >> 4);
    pd_nv4st_mt_dn_kernel<<<grid, 128u, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)rdata, (const uint8_t*)rscale, (const float*)rscale2,
        (const uint8_t*)sdata, (const uint8_t*)sscale, (const float*)sscale2,
        (const uint32_t*)idx, (const float*)topk_w, (const float*)act,
        (float*)part, ff_r, ff_s, embd, k);
    return pd_launch_status();
#endif
}
