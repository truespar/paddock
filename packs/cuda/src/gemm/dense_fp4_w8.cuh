// gemm/dense_fp4_w8.cuh (formerly 14_dense_fp4_w8.cuh) - dense block-scale FP4 GEMM (E1f) + W8A8-FP8 dense GEMM
// Textually-included segment of the single pack translation unit.
// Not standalone-compilable: include order is defined by ../pack.cu.
//
// NOTE: this file used to be 10962 lines and carried the tcgen05
// families and the e4m3 decode/GEMV lane as well. Those are now
// gemm/dense_tc5.cuh and gemm/dense_f8_decode.cuh, included immediately after
// this one - pack.cu's list is the order, and it is load-bearing here (the
// decode lane consumes symbols the tcgen05 segment defines).
// ---- dense block-scale FP4 GEMM (E1f: the Blackwell encoder-prefill path) --
// Re-quantize a repacked Q8_0 weight (int8 rows + f16 per-32 scales) to mxfp4
// planes (packed e2m1 nibbles + ue8m0 per-32) for the block-scale MMA: the
// Q8_0 mmq GEMMs fold dA*dB on CUDA cores per k32 block, which caps them at a
// fraction of the Blackwell tensor pipe; the mxf8f6f4 MMA takes both scales
// as native operands. Nibbles pack in the same split order GGUF/repack use
// (low nibble of byte j = element j, high = element j+16) so all on-device
// mxfp4 shares one byte convention (consumed via pd_bs_afrag_split). Run
// once at model load on sm_120; lossy (4-bit weights) - model paths gate it
// on retrieval quality, never on greedy exactness.

// f32 -> e2m1 nibble, round-to-nearest-even on the e2m1 grid
// {0, .5, 1, 1.5, 2, 3, 4, 6} (ties resolve toward even mantissa, matching
// IEEE RN semantics). Caller guarantees |v| <= 6 via the block scale.
__device__ __forceinline__ unsigned pd_e2m1_rn(float v) {
    unsigned s = v < 0.0f ? 8u : 0u;
    float a = fabsf(v);
    unsigned m;
    if (a <= 0.25f) m = 0u;            // .25 tie -> 0   (mantissa 0 even)
    else if (a < 0.75f) m = 1u;        // .75 tie -> 1.0 (mantissa 0)
    else if (a <= 1.25f) m = 2u;       // 1.25 tie -> 1.0
    else if (a < 1.75f) m = 3u;        // 1.75 tie -> 2.0
    else if (a <= 2.5f) m = 4u;        // 2.5 tie -> 2.0
    else if (a < 3.5f) m = 5u;         // 3.5 tie -> 4.0
    else if (a <= 5.0f) m = 6u;        // 5.0 tie -> 4.0
    else m = 7u;
    return s | m;
}

// One warp per 32-element block: dequant Q8_0, pick the smallest power-of-2
// scale with amax/2^e <= 6 (the e2m1 max, same construction as
// pd_quantize_e4m3's 448 bound), quantize RN-even, pack split-order nibbles.
__global__ void pd_q8_0_to_mxfp4_kernel(const int8_t* __restrict__ q8,
                                        const __half* __restrict__ s8,
                                        unsigned char* __restrict__ data,
                                        unsigned char* __restrict__ scale,
                                        uint64_t n_blocks) {
    uint64_t blk = blockIdx.x;
    uint32_t d = threadIdx.x;
    if (blk >= n_blocks) return;
    float v = (float)q8[blk * 32u + d] * __half2float(s8[blk]);
    float a = fabsf(v);
    for (uint32_t off = 16; off > 0; off >>= 1)
        a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, off));
    int e = 0;
    if (a > 0.0f) {
        // smallest e with a / 2^e <= 6  (6 = 0.75 * 2^3)
        int ex;
        float m = frexpf(a, &ex);
        e = ex - 3 + (m > 0.75f ? 1 : 0);
    }
    unsigned nib = pd_e2m1_rn(v * ldexpf(1.0f, -e));
    unsigned hi = __shfl_sync(0xffffffffu, nib, d + 16u);  // element d+16's nibble
    if (d < 16u) data[blk * 16u + d] = (unsigned char)(nib | (hi << 4));
    if (d == 0) scale[blk] = (unsigned char)(e + 127);
}

PD_EXPORT
int pd_q8_0_to_mxfp4(const void* q8_data, const void* q8_scale, void* mx_data,
                     void* mx_scale, uint64_t n_blocks, void* stream) {
#ifndef PD_BS_HOST
    (void)q8_data; (void)q8_scale; (void)mx_data; (void)mx_scale; (void)n_blocks;
    (void)stream;
    return cudaErrorNotSupported;
#else
    if (n_blocks == 0) return 0;
    pd_q8_0_to_mxfp4_kernel<<<(uint32_t)n_blocks, 32, 0, (cudaStream_t)stream>>>(
        (const int8_t*)q8_data, (const __half*)q8_scale, (unsigned char*)mx_data,
        (unsigned char*)mx_scale, n_blocks);
    return pd_launch_status();
#endif
}

// Dense block-scale GEMM: y[token][r] = sum_k W[r][k] * x[token][k] with
// W = mxfp4 (split-order e2m1 + ue8m0/32) and x = e4m3 + ue8m0/32 (the
// pd_quantize_e4m3 layout), both scales riding the MMA as native operands.
// Skeleton = pd_q8_0_gemm_mmq_pipe (128x128 tile, 2-stage cp.async pipe,
// batch tiles on the fast grid axis for weight L2 reuse) with the int8 MMA +
// CUDA-core scale fold swapped for the block-scale mxf8f6f4 MMA. K stages
// 64-deep: the fp4+e4m3 tiles are half the int8 mmq tile, so the double
// buffer fits 2 blocks/SM (mmq had to choose pipe OR occupancy). Scale bytes
// stage synchronously after the data wait (2 B/row/chunk, unaligned in their
// planes). No batch padding requirement: ragged tails are masked (zero-fill
// staging, guarded stores).
__global__ void __launch_bounds__(256, 2) pd_mxfp4_gemm_bs_kernel(
    const unsigned char* __restrict__ data, const unsigned char* __restrict__ scale,
    const unsigned char* __restrict__ xq, const unsigned char* __restrict__ xs,
    float* __restrict__ y, uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
#if PD_BS_OK
    extern __shared__ unsigned char pd_bs_sh[];
    unsigned char* wb0 = pd_bs_sh;
    unsigned char* wb1 = wb0 + 128u * PD_BS_P_WROW;
    unsigned char* yb0 = wb1 + 128u * PD_BS_P_WROW;
    unsigned char* yb1 = yb0 + 128u * PD_BS_P_YROW;

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, tq = lane & 3u;
    // warp tile 64 rows x 32 cols (2 row-groups x 4 col-groups): each B
    // fragment is re-read by two warps instead of four - the kernel is
    // LDS-bandwidth-bound and B is the bigger tile side (e4m3 bytes vs
    // packed fp4), so trading A re-reads (2x -> 4x, half the bytes) for B
    // re-reads (4x -> 2x) cuts total LDS traffic ~20%.
    const uint32_t i0 = (warp & 1u) * 64u;
    const uint32_t c0w = (warp >> 1) * 32u;
    const uint32_t n_kb = in_dim >> 5;
    const uint32_t nk = (in_dim + 63u) / 64u;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t nct = batch_pad >> 7;
    const uint32_t tile = blockIdx.x;
    const uint32_t row_base = (tile / nct) * 128u;
    const uint32_t col_base = (tile % nct) * 128u;

    float acc[16][4] = {};

    // W: 128 rows x 32B packed fp4 (2 16B segs); Y: 128 cols x 64B e4m3
    // (4 segs) at +16 in the row. Ragged K/row/col tails zero-fill.
    #define PD_BSP_ISSUE_W(dst, kt)                                                   \
        {                                                                             \
            const uint32_t row = tid >> 1, seg = tid & 1u;                            \
            const bool ok = (row_base + row) < out_dim && (kt) * 2u + seg < n_kb;     \
            pd_cp_async16((int*)((dst) + row * PD_BS_P_WROW + seg * 16u),             \
                          data + (size_t)(row_base + row) * (in_dim >> 1) +           \
                              (kt) * 32u + seg * 16u,                                 \
                          ok);                                                        \
        }
    #define PD_BSP_ISSUE_Y(dst, kt)                                                   \
        for (uint32_t u = tid; u < 512u; u += 256u) {                                 \
            const uint32_t col = u >> 2, seg = u & 3u;                                \
            const bool ok =                                                           \
                (col_base + col) < batch && ((kt) * 4u + seg) * 16u < in_dim;         \
            pd_cp_async16((int*)((dst) + col * PD_BS_P_YROW + 16u + seg * 16u),       \
                          xq + (size_t)(ok ? col_base + col : 0u) * in_dim +          \
                              (kt) * 64u + seg * 16u,                                 \
                          ok);                                                        \
        }

    PD_BSP_ISSUE_W(wb0, 0u)
    PD_BSP_ISSUE_Y(yb0, 0u)
    asm volatile("cp.async.commit_group;");
    for (uint32_t kt = 0; kt < nk; ++kt) {
        unsigned char* tw = (kt & 1u) ? wb1 : wb0;
        unsigned char* ty = (kt & 1u) ? yb1 : yb0;
        if (kt + 1u < nk) {  // prefetch chunk kt+1 into the other buffers
            PD_BSP_ISSUE_W((kt & 1u) ? wb0 : wb1, kt + 1u)
            PD_BSP_ISSUE_Y((kt & 1u) ? yb0 : yb1, kt + 1u)
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 1;");  // kt done; kt+1 in flight
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        // ue8m0 planes for this chunk (sync byte loads into the row tails)
        {
            const uint32_t row = tid >> 1, kb = tid & 1u;
            const bool wok = (row_base + row) < out_dim && kt * 2u + kb < n_kb;
            tw[row * PD_BS_P_WROW + 32u + kb] =
                wok ? scale[(size_t)(row_base + row) * n_kb + kt * 2u + kb] : 0u;
            const bool yok = (col_base + row) < batch && kt * 2u + kb < n_kb;
            ty[row * PD_BS_P_YROW + kb] =
                yok ? xs[(size_t)(yok ? col_base + row : 0u) * n_kb + kt * 2u + kb] : 0u;
        }
        __syncthreads();

        // A fragments for the whole chunk via 4 ldmatrix.x4 (both k32 blocks
        // of the warp's four 16-row groups): lane part = l>>3 picks {rows+0
        // kb0, rows+8 kb0, rows+0 kb1, rows+8 kb1}; the split-order nibble
        // spread (afrag_split's masks) applies in registers. Scale u16 =
        // both kbs.
        uint32_t am[4][2][4];  // [s][kb][a-frag]
        uint32_t sa[4];
        #pragma unroll
        for (uint32_t s = 0; s < 4u; ++s) {
            const uint32_t r0 = i0 + s * 16u + g;
            uint32_t raw[4];
            pd_ldm_x4(raw, tw + (i0 + s * 16u + ((lane >> 3) & 1u) * 8u + (lane & 7u)) *
                                   PD_BS_P_WROW +
                               (lane >> 4) * 16u);
            #pragma unroll
            for (uint32_t kb = 0; kb < 2u; ++kb) {
                am[s][kb][0] = (raw[kb * 2u] & 0x0F0F0F0Fu) << 2;
                am[s][kb][1] = (raw[kb * 2u + 1u] & 0x0F0F0F0Fu) << 2;
                am[s][kb][2] = (raw[kb * 2u] & 0xF0F0F0F0u) >> 2;
                am[s][kb][3] = (raw[kb * 2u + 1u] & 0xF0F0F0F0u) >> 2;
            }
            const uint32_t rs = (tq & 1u) ? r0 + 8u : r0;
            sa[s] = *(const unsigned short*)(tw + rs * PD_BS_P_WROW + 32u);
        }
        #pragma unroll
        for (uint32_t j = 0; j < 4u; ++j) {
            // B fragments for 8 cols x both kbs in one ldmatrix.x4 (part =
            // l>>3 walks the four 16 B slices b0/b1 of kb0/kb1); scale u16.
            // Each load feeds all 8 of the warp's MMAs on this col octet.
            uint32_t bm[4];
            pd_ldm_x4(bm, ty + (c0w + j * 8u + (lane & 7u)) * PD_BS_P_YROW + 16u +
                              (lane >> 3) * 16u);
            const uint32_t sb =
                *(const unsigned short*)(ty + (c0w + j * 8u + g) * PD_BS_P_YROW);
            #pragma unroll
            for (uint32_t s = 0; s < 4u; ++s) {
                pd_bs_mma_kb<0>(acc[s * 4u + j], am[s][0][0], am[s][0][1],
                                am[s][0][2], am[s][0][3], bm[0], bm[1], sa[s], sb);
                pd_bs_mma_kb<1>(acc[s * 4u + j], am[s][1][0], am[s][1][1],
                                am[s][1][2], am[s][1][3], bm[2], bm[3], sa[s], sb);
            }
        }
        __syncthreads();  // buffer free before chunk kt+2 prefetches into it
    }
    #undef PD_BSP_ISSUE_W
    #undef PD_BSP_ISSUE_Y

    #pragma unroll
    for (uint32_t j = 0; j < 4u; ++j) {
        const uint32_t c0 = col_base + c0w + j * 8u + 2u * tq;
        #pragma unroll
        for (uint32_t s = 0; s < 4u; ++s) {
            const uint32_t r0 = row_base + i0 + s * 16u + g;
            const uint32_t r8 = r0 + 8u;
            if (r0 < out_dim) {
                if (c0 < batch) y[(size_t)c0 * out_dim + r0] = acc[s * 4u + j][0];
                if (c0 + 1u < batch) y[(size_t)(c0 + 1u) * out_dim + r0] = acc[s * 4u + j][1];
            }
            if (r8 < out_dim) {
                if (c0 < batch) y[(size_t)c0 * out_dim + r8] = acc[s * 4u + j][2];
                if (c0 + 1u < batch) y[(size_t)(c0 + 1u) * out_dim + r8] = acc[s * 4u + j][3];
            }
        }
    }
#else
    (void)data; (void)scale; (void)xq; (void)xs; (void)y;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}

// ---- fp4-TMA dense GEMM ---------------------------------------------------
// pd_f8_gemm_w8_tma_kt's warp-specialized TMA structure with e2m1 weights:
// the 128B x 128-row SWIZZLE_128B W box covers K-256 PACKED elements = two
// K-128 pairs per fetch, so W streams half the bytes of the e4m3 kernel at
// the same proven box geometry. W stage ring [2][16KB], fetched on even
// pairs; the even pair's mbarrier counts W+Y tx (32KB), odd pairs Y only
// (16KB). W overwrite needs no extra barrier: the producer's Y-EMPTY(b)
// wait (pair sp-2 done) already implies the W stage's last reader (pair
// sp-3) finished. Consumer fragments ride the MoE mxfp4 pattern: ldmatrix
// raw over packed bytes + nibble expand (e2m1 at bits 5:2, split order),
// pd_bs_mma_kb e2m1.e4m3 block-scale. Scale staging identical (ue8m0/32).
__global__ void __launch_bounds__(384, 1) pd_fp4_gemm_w8_tma_kt(
    const __grid_constant__ PdTmap wmap, const __grid_constant__ PdTmap ymap,
    const unsigned char* __restrict__ scale, const unsigned char* __restrict__ xs,
    float* __restrict__ y, uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
#if PD_BS_OK
    constexpr uint32_t PAIR16 = 16384u;
    extern __shared__ __align__(128) unsigned char pd_fp4t_sh[];
    unsigned char* wdat = pd_fp4t_sh;              // 2 stages x 16KB (K-256 each)
    unsigned char* ydat = pd_fp4t_sh + 32768u;     // 2 pairs x 16KB
    unsigned char* wsc = pd_fp4t_sh + 65536u;      // 2 pairs x 2 slabs x 256B
    unsigned char* ysc = pd_fp4t_sh + 66560u;
    unsigned long long* mb = (unsigned long long*)(pd_fp4t_sh + 67584u);

    const uint32_t tid = threadIdx.x;
    const uint32_t n_kb = in_dim >> 5;
    const uint32_t nk = (in_dim + 63u) / 64u;
    const uint32_t nsp = (nk + 1u) >> 1;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t nct = batch_pad >> 7;
    const uint32_t tile = blockIdx.x;
    const uint32_t row_base = (tile / nct) * 128u;
    const uint32_t col_base = (tile % nct) * 128u;

    if (tid == 0u) {
        const uint32_t m0 = (uint32_t)__cvta_generic_to_shared(mb);
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 128;" ::"r"(m0));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 128;" ::"r"(m0 + 8u));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    asm volatile("bar.sync 0, 384;");

    if (tid >= 256u) {
        // ---------------- producer warps 8-11 ----------------
        const uint32_t ptid = tid - 256u;
        unsigned char swr[2][2], syr[2][2];
        for (uint32_t sp = 0; sp < nsp; ++sp) {
            const uint32_t b = sp % 2u;
            #pragma unroll
            for (uint32_t h = 0; h < 2u; ++h) {
                #pragma unroll
                for (uint32_t kb = 0; kb < 2u; ++kb) {
                    const uint32_t kt = sp * 2u + h;
                    const bool wok = (row_base + ptid) < out_dim && kt * 2u + kb < n_kb;
                    swr[h][kb] = wok ? scale[(size_t)(row_base + ptid) * n_kb + kt * 2u + kb] : 0u;
                    const bool yok = (col_base + ptid) < batch && kt * 2u + kb < n_kb;
                    syr[h][kb] = yok ? xs[(size_t)(col_base + ptid) * n_kb + kt * 2u + kb] : 0u;
                }
            }
            if (sp >= 2u) asm volatile("bar.sync %0, 384;" ::"r"(1u + b));
            #pragma unroll
            for (uint32_t h = 0; h < 2u; ++h) {
                #pragma unroll
                for (uint32_t kb = 0; kb < 2u; ++kb) {
                    wsc[b * 512u + h * 256u + ptid * 2u + kb] = swr[h][kb];
                    ysc[b * 512u + h * 256u + ptid * 2u + kb] = syr[h][kb];
                }
            }
            const uint32_t m = (uint32_t)__cvta_generic_to_shared(mb) + b * 8u;
            if (ptid == 0u) {
                const bool wfetch = (sp & 1u) == 0u;
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(m),
                             "r"(wfetch ? 32768u : 16384u));
                if (wfetch) {
                    const uint32_t wd = (uint32_t)__cvta_generic_to_shared(
                        wdat + ((sp >> 1) & 1u) * PAIR16);
                    // W row = in_dim/2 bytes; K-256 span = 128 bytes at ck
                    const int ckw = (int)((sp >> 1) * 128u);
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                        " [%0], [%1, {%2, %3}], [%4];" ::"r"(wd),
                        "l"(&wmap), "r"(ckw), "r"((int)row_base), "r"(m)
                        : "memory");
                }
                const uint32_t yd = (uint32_t)__cvta_generic_to_shared(ydat + b * PAIR16);
                const int ck = (int)(sp * 128u);
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3}], [%4];" ::"r"(yd),
                    "l"(&ymap), "r"(ck), "r"((int)col_base), "r"(m)
                    : "memory");
            } else {
                asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(m));
            }
        }
        return;
    }

    // ---------------- consumer warps 0-7 ----------------
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, tq = lane & 3u;
    const uint32_t i0 = (warp & 1u) * 64u;
    const uint32_t c0w = (warp >> 1) * 32u;

    float acc[16][4] = {};
    uint32_t ph0 = 0u, ph1 = 0u;

    for (uint32_t sp = 0; sp < nsp; ++sp) {
        const uint32_t b = sp % 2u;
        const uint32_t m = (uint32_t)__cvta_generic_to_shared(mb) + b * 8u;
        const uint32_t ph = (b == 0u) ? ph0 : ph1;
        asm volatile(
            "{\n\t.reg .pred P;\n"
            "PD_FP4T_WAIT_%=:\n\t"
            "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
            "@!P bra PD_FP4T_WAIT_%=;\n\t}" ::"r"(m), "r"(ph));
        if (b == 0u) ph0 ^= 1u; else ph1 ^= 1u;

        const unsigned char* wp = wdat + ((sp >> 1) & 1u) * PAIR16;
        const uint32_t po = sp & 1u;  // which K-128 half of the W K-256 span
        const unsigned char* yp = ydat + b * PAIR16;
        #pragma unroll
        for (uint32_t h = 0; h < 2u; ++h) {
            const uint32_t kt = sp * 2u + h;
            if (kt >= nk) break;

            uint32_t am[4][2][4];
            uint32_t sa[4];
            #pragma unroll
            for (uint32_t s = 0; s < 4u; ++s) {
                const uint32_t rr = i0 + s * 16u + ((lane >> 3) & 1u) * 8u + (lane & 7u);
                // packed K-64 for this half = 32B = chunks {c0, c0+1} of the
                // swizzled 128B row; raw x4 covers both kbs, then expand
                const uint32_t c0 = po * 4u + h * 2u + (lane >> 4);
                uint32_t raw[4];
                pd_ldm_x4(raw, wp + rr * 128u + ((c0 ^ (rr & 7u)) * 16u));
                #pragma unroll
                for (uint32_t kb = 0; kb < 2u; ++kb) {
                    am[s][kb][0] = (raw[kb * 2u] & 0x0F0F0F0Fu) << 2;
                    am[s][kb][1] = (raw[kb * 2u + 1u] & 0x0F0F0F0Fu) << 2;
                    am[s][kb][2] = (raw[kb * 2u] & 0xF0F0F0F0u) >> 2;
                    am[s][kb][3] = (raw[kb * 2u + 1u] & 0xF0F0F0F0u) >> 2;
                }
                const uint32_t r0 = i0 + s * 16u + g;
                const uint32_t rs = (tq & 1u) ? r0 + 8u : r0;
                sa[s] = *(const unsigned short*)(wsc + b * 512u + h * 256u + rs * 2u);
            }
            uint32_t bmj[4][4];
            uint32_t sbj[4];
            #pragma unroll
            for (uint32_t j = 0; j < 4u; ++j) {
                const uint32_t col = c0w + j * 8u + (lane & 7u);
                const uint32_t c = h * 4u + (lane >> 3);
                pd_ldm_x4(bmj[j], yp + col * 128u + ((c ^ (col & 7u)) * 16u));
                sbj[j] = *(const unsigned short*)(ysc + b * 512u + h * 256u + (c0w + j * 8u + g) * 2u);
            }
            #pragma unroll
            for (uint32_t j = 0; j < 4u; ++j)
                #pragma unroll
                for (uint32_t s = 0; s < 4u; ++s)
                    pd_bs_mma_kb<0>(acc[s * 4u + j], am[s][0][0], am[s][0][1],
                                    am[s][0][2], am[s][0][3], bmj[j][0], bmj[j][1],
                                    sa[s], sbj[j]);
            #pragma unroll
            for (uint32_t j = 0; j < 4u; ++j)
                #pragma unroll
                for (uint32_t s = 0; s < 4u; ++s)
                    pd_bs_mma_kb<1>(acc[s * 4u + j], am[s][1][0], am[s][1][1],
                                    am[s][1][2], am[s][1][3], bmj[j][2], bmj[j][3],
                                    sa[s], sbj[j]);
        }
        asm volatile("bar.arrive %0, 384;" ::"r"(1u + b));
    }

    #pragma unroll
    for (uint32_t j = 0; j < 4u; ++j) {
        const uint32_t c0 = col_base + c0w + j * 8u + 2u * tq;
        #pragma unroll
        for (uint32_t s = 0; s < 4u; ++s) {
            const uint32_t r0 = row_base + i0 + s * 16u + g;
            const uint32_t r8 = r0 + 8u;
            if (r0 < out_dim) {
                if (c0 < batch) y[(size_t)c0 * out_dim + r0] = acc[s * 4u + j][0];
                if (c0 + 1u < batch) y[(size_t)(c0 + 1u) * out_dim + r0] = acc[s * 4u + j][1];
            }
            if (r8 < out_dim) {
                if (c0 < batch) y[(size_t)c0 * out_dim + r8] = acc[s * 4u + j][2];
                if (c0 + 1u < batch) y[(size_t)(c0 + 1u) * out_dim + r8] = acc[s * 4u + j][3];
            }
        }
    }
#else
    (void)wmap; (void)ymap; (void)scale; (void)xs; (void)y;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}

PD_EXPORT
int pd_mxfp4_gemm_bs(const void* data, const void* scale, const void* xq,
                     const void* xs, void* y, uint32_t in_dim, uint32_t out_dim,
                     uint32_t batch, void* stream) {
#ifndef PD_BS_HOST
    (void)data; (void)scale; (void)xq; (void)xs; (void)y; (void)in_dim;
    (void)out_dim; (void)batch; (void)stream;
    return cudaErrorNotSupported;
#else
    if (out_dim == 0 || batch == 0) return 0;
    if ((in_dim & 31u) != 0) return cudaErrorInvalidValue;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t ntiles = ((out_dim + 127u) / 128u) * (batch_pad >> 7);
    // fp4-TMA route (PADDOCK_FP4_TMA=1): the warp-specialized TMA kernel
    // with e2m1 weights - half the W stream at the proven box geometry.
    // K must fill whole 256-element W spans (in_dim % 256 == 0; gemma/qwen
    // dims all qualify) so every fetched span is fully consumed.
    static const bool fp4tma = pd_env("PADDOCK_FP4_TMA") != nullptr;
    if (fp4tma && (in_dim & 255u) == 0u) {
        const uint32_t smem = 67600u;
        static bool a4t = false;
        if (!a4t) {
            cudaFuncSetAttribute((const void*)pd_fp4_gemm_w8_tma_kt,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
            a4t = true;
        }
        CUtensorMap wm, ym;
        if (pd_tmap_2d(&wm, data, in_dim / 2u, out_dim) &&
            pd_tmap_2d(&ym, xq, in_dim, batch)) {
            pd_fp4_gemm_w8_tma_kt<<<ntiles, 384, smem, (cudaStream_t)stream>>>(
                wm, ym, (const unsigned char*)scale, (const unsigned char*)xs,
                (float*)y, in_dim, out_dim, batch);
            return pd_launch_status();
        }
    }
    pd_mxfp4_gemm_bs_kernel<<<ntiles, 256, PD_BS_P_SMEM, (cudaStream_t)stream>>>(
        (const unsigned char*)data, (const unsigned char*)scale,
        (const unsigned char*)xq, (const unsigned char*)xs, (float*)y, in_dim,
        out_dim, batch);
    return pd_launch_status();
#endif
}

// ---------------- W8A8-FP8 dense GEMM (e4m3 weights x e4m3 activations) ----
// The quality rung between q8_0 and the fp4 classes: weights requantized
// Q8_0 -> e4m3 (8-bit container, 3 mantissa bits, ue8m0/32 scale - ~4x finer
// than the e2m1 weights that fail small-model rerank gates) while the GEMM
// still rides the block-scale mxf8f6f4 MMA at the full int8-class issue rate
// (~273 TFLOPs GB202) instead of mmq's ~100 effective. Same 128x128 pipe
// skeleton as pd_mxfp4_gemm_bs; the weight tile doubles to e4m3 bytes and
// stages exactly like the activation tile (scales at row+0, data at +16).

// mxf8f6f4 MMA with both operands e4m3 (weights A, activations B). sm_120a
// only - the ue8m0 scales ride as native LANE-DISTRIBUTED operands (the hw
// routes each lane's scale word to the fragment rows/cols that need it).
// Template => only instantiated by PD_BS_OK-gated callers.
template <int KB>
__device__ __forceinline__ void pd_bs_mma_w8_kb(float d[4], uint32_t a0, uint32_t a1,
                                                uint32_t a2, uint32_t a3, uint32_t b0,
                                                uint32_t b1, uint32_t sfa, uint32_t sfb) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X"
        ".f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
        "{%10}, {%12, 0}, {%11}, {%12, 0};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "r"(sfb),
          "n"(KB));
}

// 2^(sa-127) * 2^(sb-127) = 2^(e-127) for e = sa+sb-127: exact exponent-add
// in the f32 bit pattern; the rare underflow (dead-row x dead-activation)
// drops to ldexpf's graceful denorm/zero - hardware ue8m0 flushes the same.
__device__ __forceinline__ float pd_ue8m0_mul(uint32_t sa, uint32_t sb) {
    const int e = (int)(sa + sb) - 127;
    return e > 0 ? __int_as_float((uint32_t)e << 23) : ldexpf(1.0f, e - 127);
}

// Software twin of pd_bs_mma_w8_kb for the other FP8-MMA arches (sm_89/90/
// 100 - PD_F8W8_OK without PD_BS_OK): plain e4m3 mma into a ZEROED
// accumulator, then the four per-quad ue8m0 scale products folded on CUDA
// cores. The m16n8k32 accumulator quad spans rows (r, r+8) x cols (c0, c1),
// i.e. Four distinct scale combinations - which is why this takes the four
// scale bytes explicitly, resolved by the caller from shared memory, instead
// of aping the hw call's lane-distributed words (doing that folded one
// combined scale over all four quads and shipped gibberish - caught by the
// g4 coherence gate + the f8 unit tests). Numeric class vs hw: the
// fold FMA's rounding point per k32 - same family as the q8 mmq fold; the
// per-arch parity gates arbitrate.
__device__ __forceinline__ void pd_f8_mma_sw(float d[4], uint32_t a0, uint32_t a1,
                                             uint32_t a2, uint32_t a3, uint32_t b0,
                                             uint32_t b1, uint32_t sar, uint32_t sar8,
                                             uint32_t sbc0, uint32_t sbc1) {
    float t0 = 0.f, t1 = 0.f, t2 = 0.f, t3 = 0.f;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(t0), "+f"(t1), "+f"(t2), "+f"(t3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    d[0] += pd_ue8m0_mul(sar, sbc0) * t0;
    d[1] += pd_ue8m0_mul(sar, sbc1) * t1;
    d[2] += pd_ue8m0_mul(sar8, sbc0) * t2;
    d[3] += pd_ue8m0_mul(sar8, sbc1) * t3;
}

// Q8_0 -> e4m3 planes: same per-32 ue8m0 scale pick as pd_quantize_e4m3
// (smallest e with amax/2^e <= 448), RN-even encode via the hardware cast.
__global__ void pd_q8_0_to_f8w_kernel(const int8_t* __restrict__ q8,
                                      const __half* __restrict__ s8,
                                      unsigned char* __restrict__ data,
                                      unsigned char* __restrict__ scale,
                                      uint64_t n_blocks) {
    uint64_t blk = blockIdx.x;
    uint32_t d = threadIdx.x;
    if (blk >= n_blocks) return;
    float v = (float)q8[blk * 32u + d] * __half2float(s8[blk]);
    float a = fabsf(v);
    for (uint32_t off = 16; off > 0; off >>= 1)
        a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, off));
    int e = 0;
    if (a > 0.0f) {
        // smallest e with a / 2^e <= 448  (448 = 0.875 * 2^9)
        int ex;
        float m = frexpf(a, &ex);
        e = ex - 9 + (m > 0.875f ? 1 : 0);
    }
    data[blk * 32u + d] = __nv_fp8_e4m3(v * ldexpf(1.0f, -e)).__x;
    if (d == 0) scale[blk] = (unsigned char)(e + 127);
}

// Native-bf16 -> f8w conversion (the fp8-native ingestion lane): same
// per-32-block e8m0 scale pick + RN-even e4m3 encode as pd_q8_0_to_f8w, but
// sourced from bf16 checkpoint bytes directly - no Q8_0 double quantization.
__global__ void pd_bf16_to_f8w_kernel(const __nv_bfloat16* __restrict__ src,
                                      unsigned char* __restrict__ data,
                                      unsigned char* __restrict__ scale,
                                      uint64_t n_blocks) {
    uint64_t blk = blockIdx.x;
    uint32_t d = threadIdx.x;
    if (blk >= n_blocks) return;
    float v = __bfloat162float(src[blk * 32u + d]);
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

// Native-bf16 -> f8r: e4m3 data + one e8m0 scale byte per output ROW - the
// scale-free-class weight stream (1.0 B/param + 1B/row; vLLM fp8's scale
// granularity). One block per row; 256-thread absmax reduce, then encode.
__global__ void pd_bf16_to_f8r_kernel(const __nv_bfloat16* __restrict__ src,
                                      unsigned char* __restrict__ data,
                                      unsigned char* __restrict__ scale,
                                      uint32_t in_dim) {
    const uint32_t row = blockIdx.x, tid = threadIdx.x;
    const __nv_bfloat16* r = src + (size_t)row * in_dim;
    __shared__ float wmax[8];
    float a = 0.0f;
    for (uint32_t i = tid; i < in_dim; i += 256u)
        a = fmaxf(a, fabsf(__bfloat162float(r[i])));
    for (uint32_t off = 16; off > 0; off >>= 1)
        a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, off));
    if ((tid & 31u) == 0) wmax[tid >> 5] = a;
    __syncthreads();
    if (tid < 8) {
        float m = wmax[tid];
        for (uint32_t off = 4; off > 0; off >>= 1)
            m = fmaxf(m, __shfl_xor_sync(0xffu, m, off));
        if (tid == 0) {
            int e = 0;
            if (m > 0.0f) {
                int ex;
                float f = frexpf(m, &ex);
                e = ex - 9 + (f > 0.875f ? 1 : 0);
            }
            wmax[0] = ldexpf(1.0f, -e);
            scale[row] = (unsigned char)(e + 127);
        }
    }
    __syncthreads();
    const float inv = wmax[0];
    for (uint32_t i = tid; i < in_dim; i += 256u)
        data[(size_t)row * in_dim + i] = __nv_fp8_e4m3(__bfloat162float(r[i]) * inv).__x;
}

// Native-bf16 -> f8ROW: e4m3 data + one F32 scale per output row. The f32
// twin of pd_bf16_to_f8r_kernel above, and the bf16 twin of
// pd_q8_0_to_f8row_kernel below - same absmax, same exponent pick, same
// `rscale[row] = 2^e` convention, so the plane it emits is byte-for-byte the
// shape f8_repack_tiles/f8t_gemm already consume.
//
// Why both scale encodings exist: the e8m0 BYTE form feeds the scale-free
// weight-stream class (1 B/param + 1 B/row), while F8RowPlane's tile route
// wants an f32 scale it can fold into the epilogue without an exp2 per tile.
// A bf16 lm_head (muse-glimmer) had no path to the tile route at all because
// the only f32-scale producer wanted a Q8 source; this is that missing edge.
__global__ void pd_bf16_to_f8row_kernel(const __nv_bfloat16* __restrict__ src,
                                        unsigned char* __restrict__ data,
                                        float* __restrict__ rscale,
                                        uint32_t in_dim) {
    const uint32_t row = blockIdx.x, tid = threadIdx.x;
    const __nv_bfloat16* r = src + (size_t)row * in_dim;
    __shared__ float wmax[8];
    __shared__ float s_inv;
    float a = 0.0f;
    for (uint32_t i = tid; i < in_dim; i += 256u)
        a = fmaxf(a, fabsf(__bfloat162float(r[i])));
    for (uint32_t off = 16; off > 0; off >>= 1)
        a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, off));
    if ((tid & 31u) == 0) wmax[tid >> 5] = a;
    __syncthreads();
    if (tid < 8) {
        float m = wmax[tid];
        for (uint32_t off = 4; off > 0; off >>= 1)
            m = fmaxf(m, __shfl_xor_sync(0xffu, m, off));
        if (tid == 0) {
            int e = 0;
            if (m > 0.0f) {
                int ex;
                float f = frexpf(m, &ex);
                e = ex - 9 + (f > 0.875f ? 1 : 0);  // amax/2^e <= 448 = 0.875*2^9
            }
            s_inv = ldexpf(1.0f, -e);
            rscale[row] = ldexpf(1.0f, e);
        }
    }
    __syncthreads();
    const float inv = s_inv;
    for (uint32_t i = tid; i < in_dim; i += 256u)
        data[(size_t)row * in_dim + i] = __nv_fp8_e4m3(__bfloat162float(r[i]) * inv).__x;
}

PD_EXPORT
int pd_bf16_to_f8row(const void* bf16, void* f8_data, void* row_scale,
                     uint32_t in_dim, uint32_t out_dim, void* stream) {
#ifndef PD_BS_HOST
    (void)bf16; (void)f8_data; (void)row_scale; (void)in_dim; (void)out_dim; (void)stream;
    return cudaErrorNotSupported;
#else
    if (in_dim == 0 || out_dim == 0) return 0;
    pd_bf16_to_f8row_kernel<<<out_dim, 256, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)bf16, (unsigned char*)f8_data,
        (float*)row_scale, in_dim);
    return pd_launch_status();
#endif
}

PD_EXPORT
int pd_bf16_to_f8r(const void* bf16, void* f8_data, void* f8_scale,
                   uint32_t in_dim, uint32_t out_dim, void* stream) {
#ifndef PD_BS_HOST
    (void)bf16; (void)f8_data; (void)f8_scale; (void)in_dim; (void)out_dim; (void)stream;
    return cudaErrorNotSupported;
#else
    if (in_dim == 0 || out_dim == 0) return 0;
    pd_bf16_to_f8r_kernel<<<out_dim, 256, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)bf16, (unsigned char*)f8_data,
        (unsigned char*)f8_scale, in_dim);
    return pd_launch_status();
#endif
}

PD_EXPORT
int pd_bf16_to_f8w(const void* bf16, void* f8_data, void* f8_scale,
                   uint64_t n_blocks, void* stream) {
#ifndef PD_BS_HOST
    (void)bf16; (void)f8_data; (void)f8_scale; (void)n_blocks; (void)stream;
    return cudaErrorNotSupported;
#else
    if (n_blocks == 0) return 0;
    pd_bf16_to_f8w_kernel<<<(uint32_t)n_blocks, 32, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)bf16, (unsigned char*)f8_data,
        (unsigned char*)f8_scale, n_blocks);
    return pd_launch_status();
#endif
}

PD_EXPORT
int pd_q8_0_to_f8w(const void* q8_data, const void* q8_scale, void* f8_data,
                   void* f8_scale, uint64_t n_blocks, void* stream) {
#ifndef PD_BS_HOST
    (void)q8_data; (void)q8_scale; (void)f8_data; (void)f8_scale; (void)n_blocks;
    (void)stream;
    return cudaErrorNotSupported;
#else
    if (n_blocks == 0) return 0;
    pd_q8_0_to_f8w_kernel<<<(uint32_t)n_blocks, 32, 0, (cudaStream_t)stream>>>(
        (const int8_t*)q8_data, (const __half*)q8_scale, (unsigned char*)f8_data,
        (unsigned char*)f8_scale, n_blocks);
    return pd_launch_status();
#endif
}

#define PD_BS_W8_ROW 80u  // 2 scale bytes, data at +16 (same shape as YROW)
#define PD_BS_W8_SMEM (4u * 128u * PD_BS_W8_ROW)
// Multistage (STAGES-deep cp.async) block-scale fp8 W8A8 GEMM. STAGES=2 is the
// original double-buffer (40 KB, 2 CTA/SM); STAGES=3/4 (60/80 KB, 1 CTA/SM) keep
// more K-chunks in flight to hide load latency on the SYNCHRONOUS mma.sync pipe
// (the kernel is latency-bound at ~40% SOL, not fold-bound - the mxf8f6f4
// block-scale MMA applies the per-32 ue8m0 scales in hardware at full rate).
// BIT-IDENTICAL across STAGES: same k-chunk iteration + accumulation order, only
// the prefetch depth differs. The tail keeps a UNIFORM group count by emitting an
// empty commit_group once prefetch runs out, so `cp.async.wait_group STAGES-1`
// (immediate) is always correct.  Ring: chunk kt lives in buffer kt%STAGES; the
// prefetch of kt+STAGES-1 targets buffer (kt-1)%STAGES, freed by kt-1's trailing
// barrier.  Smem = 2*STAGES*128*ROW (W ring then Y ring).
template <uint32_t STAGES>
__global__ void __launch_bounds__(256, (STAGES <= 2u) ? 2 : 1) pd_f8_gemm_w8_kt(
    const unsigned char* __restrict__ data, const unsigned char* __restrict__ scale,
    const unsigned char* __restrict__ xq, const unsigned char* __restrict__ xs,
    float* __restrict__ y, uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
#if PD_F8W8_OK
    constexpr uint32_t ROW = PD_BS_W8_ROW;
    extern __shared__ unsigned char pd_bsw8_sh[];
    unsigned char* wring = pd_bsw8_sh;                       // STAGES x 128 x ROW
    unsigned char* yring = wring + STAGES * 128u * ROW;      // STAGES x 128 x ROW
    #define PD_W8_WBUF(s) (wring + ((s) % STAGES) * 128u * ROW)
    #define PD_W8_YBUF(s) (yring + ((s) % STAGES) * 128u * ROW)

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, tq = lane & 3u;
    const uint32_t i0 = (warp & 1u) * 64u;
    const uint32_t c0w = (warp >> 1) * 32u;
    const uint32_t n_kb = in_dim >> 5;
    const uint32_t nk = (in_dim + 63u) / 64u;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t nct = batch_pad >> 7;
    const uint32_t tile = blockIdx.x;
    const uint32_t row_base = (tile / nct) * 128u;
    const uint32_t col_base = (tile % nct) * 128u;

    float acc[16][4] = {};

    // W and Y both: 128 rows x 64B e4m3 (4 16B segs) at +16.
    #define PD_W8_ISSUE_W(dst, kt)                                                    \
        for (uint32_t u = tid; u < 512u; u += 256u) {                                 \
            const uint32_t r = u >> 2, seg = u & 3u;                                  \
            const bool ok = (row_base + r) < out_dim && ((kt) * 4u + seg) * 16u < in_dim; \
            pd_cp_async16((int*)((dst) + r * ROW + 16u + seg * 16u),                  \
                          data + (size_t)(ok ? row_base + r : 0u) * in_dim +          \
                              (kt) * 64u + seg * 16u,                                 \
                          ok);                                                        \
        }
    #define PD_W8_ISSUE_Y(dst, kt)                                                    \
        for (uint32_t u = tid; u < 512u; u += 256u) {                                 \
            const uint32_t col = u >> 2, seg = u & 3u;                                \
            const bool ok = (col_base + col) < batch && ((kt) * 4u + seg) * 16u < in_dim; \
            pd_cp_async16((int*)((dst) + col * ROW + 16u + seg * 16u),                \
                          xq + (size_t)(ok ? col_base + col : 0u) * in_dim +          \
                              (kt) * 64u + seg * 16u,                                 \
                          ok);                                                        \
        }

    // prologue: fill STAGES-1 stages (one committed group each)
    #pragma unroll
    for (uint32_t s = 0; s < STAGES - 1u; ++s) {
        if (s < nk) { PD_W8_ISSUE_W(PD_W8_WBUF(s), s) PD_W8_ISSUE_Y(PD_W8_YBUF(s), s) }
        asm volatile("cp.async.commit_group;");
    }
    for (uint32_t kt = 0; kt < nk; ++kt) {
        unsigned char* tw = PD_W8_WBUF(kt);
        unsigned char* ty = PD_W8_YBUF(kt);
        const uint32_t pf = kt + STAGES - 1u;  // chunk to prefetch this iter
        if (pf < nk) { PD_W8_ISSUE_W(PD_W8_WBUF(pf), pf) PD_W8_ISSUE_Y(PD_W8_YBUF(pf), pf) }
        asm volatile("cp.async.commit_group;");  // empty group when pf>=nk -> uniform count
        asm volatile("cp.async.wait_group %0;" ::"n"(STAGES - 1u));
        {
            const uint32_t row = tid >> 1, kb = tid & 1u;
            const bool wok = (row_base + row) < out_dim && kt * 2u + kb < n_kb;
            tw[row * ROW + kb] =
                wok ? scale[(size_t)(row_base + row) * n_kb + kt * 2u + kb] : 0u;
            const bool yok = (col_base + row) < batch && kt * 2u + kb < n_kb;
            ty[row * ROW + kb] =
                yok ? xs[(size_t)(yok ? col_base + row : 0u) * n_kb + kt * 2u + kb] : 0u;
        }
        __syncthreads();

        // A fragments: e4m3 bytes land straight off ldmatrix.x4 (lanes 0-7 /
        // 8-15 give rows +0/+8 of the 16B k-half picked by lane>>4) - the
        // four regs are {a0,a1,a2,a3}, no nibble spread.
        uint32_t am[4][2][4];
#if PD_BS_OK
        uint32_t sa[4];
#endif
        #pragma unroll
        for (uint32_t s = 0; s < 4u; ++s) {
            const uint32_t rr = i0 + s * 16u + ((lane >> 3) & 1u) * 8u + (lane & 7u);
            #pragma unroll
            for (uint32_t kb = 0; kb < 2u; ++kb)
                pd_ldm_x4(am[s][kb], tw + rr * ROW + 16u + kb * 32u + (lane >> 4) * 16u);
#if PD_BS_OK
            const uint32_t r0 = i0 + s * 16u + g;
            const uint32_t rs = (tq & 1u) ? r0 + 8u : r0;
            sa[s] = *(const unsigned short*)(tw + rs * ROW);
#endif
        }
        #pragma unroll
        for (uint32_t j = 0; j < 4u; ++j) {
            uint32_t bm[4];
            pd_ldm_x4(bm, ty + (c0w + j * 8u + (lane & 7u)) * ROW + 16u +
                              (lane >> 3) * 16u);
#if PD_BS_OK
            const uint32_t sb = *(const unsigned short*)(ty + (c0w + j * 8u + g) * ROW);
            #pragma unroll
            for (uint32_t s = 0; s < 4u; ++s) {
                pd_bs_mma_w8_kb<0>(acc[s * 4u + j], am[s][0][0], am[s][0][1],
                                   am[s][0][2], am[s][0][3], bm[0], bm[1], sa[s], sb);
                pd_bs_mma_w8_kb<1>(acc[s * 4u + j], am[s][1][0], am[s][1][1],
                                   am[s][1][2], am[s][1][3], bm[2], bm[3], sa[s], sb);
            }
#else
            // sw fold path: the quad's four scale bytes read straight off the
            // staged rows/cols (this thread's own quad - no hw lane routing)
            const uint32_t c0 = c0w + j * 8u + 2u * tq;
            #pragma unroll
            for (uint32_t s = 0; s < 4u; ++s) {
                const uint32_t r0 = i0 + s * 16u + g;
                #pragma unroll
                for (uint32_t kb = 0; kb < 2u; ++kb) {
                    const uint32_t sar = tw[r0 * ROW + kb];
                    const uint32_t sar8 = tw[(r0 + 8u) * ROW + kb];
                    const uint32_t sbc0 = ty[c0 * ROW + kb];
                    const uint32_t sbc1 = ty[(c0 + 1u) * ROW + kb];
                    pd_f8_mma_sw(acc[s * 4u + j], am[s][kb][0], am[s][kb][1],
                                 am[s][kb][2], am[s][kb][3], bm[kb * 2u],
                                 bm[kb * 2u + 1u], sar, sar8, sbc0, sbc1);
                }
            }
#endif
        }
        __syncthreads();
    }
    #undef PD_W8_ISSUE_W
    #undef PD_W8_ISSUE_Y
    #undef PD_W8_WBUF
    #undef PD_W8_YBUF

    #pragma unroll
    for (uint32_t j = 0; j < 4u; ++j) {
        const uint32_t c0 = col_base + c0w + j * 8u + 2u * tq;
        #pragma unroll
        for (uint32_t s = 0; s < 4u; ++s) {
            const uint32_t r0 = row_base + i0 + s * 16u + g;
            const uint32_t r8 = r0 + 8u;
            if (r0 < out_dim) {
                if (c0 < batch) y[(size_t)c0 * out_dim + r0] = acc[s * 4u + j][0];
                if (c0 + 1u < batch) y[(size_t)(c0 + 1u) * out_dim + r0] = acc[s * 4u + j][1];
            }
            if (r8 < out_dim) {
                if (c0 < batch) y[(size_t)c0 * out_dim + r8] = acc[s * 4u + j][2];
                if (c0 + 1u < batch) y[(size_t)(c0 + 1u) * out_dim + r8] = acc[s * 4u + j][3];
            }
        }
    }
#else
    (void)data; (void)scale; (void)xq; (void)xs; (void)y;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}

// N2 reuse variant: each block owns 128 out-rows x 256 batch-cols (two col-tiles),
// loading the weight K-chunk (and its ldmatrix fragments) once and MMAing it against
// both activation col-strips. Halves the weight L2 re-reads (the L2-bandwidth wall at
// 128x128: 85% L2 / 45% compute). Cost: 2x accumulators (128 f32/thread) -> 1
// CTA/SM. BIT-IDENTICAL per output element to pd_f8_gemm_w8_kt (same per-tile math).
// Smem = STAGES*(128 W + 256 Y)*ROW.
template <uint32_t STAGES>
__global__ void __launch_bounds__(256, 1) pd_f8_gemm_w8_n2_kt(
    const unsigned char* __restrict__ data, const unsigned char* __restrict__ scale,
    const unsigned char* __restrict__ xq, const unsigned char* __restrict__ xs,
    float* __restrict__ y, uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
#if PD_BS_OK
    constexpr uint32_t ROW = PD_BS_W8_ROW;
    extern __shared__ unsigned char pd_bsw8_sh[];
    unsigned char* wring = pd_bsw8_sh;                    // STAGES x 128 x ROW
    unsigned char* yring = wring + STAGES * 128u * ROW;   // STAGES x 256 x ROW
    #define PD_N2_WBUF(s) (wring + ((s) % STAGES) * 128u * ROW)
    #define PD_N2_YBUF(s) (yring + ((s) % STAGES) * 256u * ROW)

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, tq = lane & 3u;
    const uint32_t i0 = (warp & 1u) * 64u;
    const uint32_t c0w = (warp >> 1) * 32u;
    const uint32_t n_kb = in_dim >> 5;
    const uint32_t nk = (in_dim + 63u) / 64u;
    const uint32_t batch_pad = (batch + 255u) & ~255u;
    const uint32_t nct = batch_pad >> 8;                  // 256-col tiles
    const uint32_t tile = blockIdx.x;
    const uint32_t row_base = (tile / nct) * 128u;
    const uint32_t col_base = (tile % nct) * 256u;

    float acc[2][16][4] = {};

    #define PD_N2_ISSUE_W(dst, kt)                                                    \
        for (uint32_t u = tid; u < 512u; u += 256u) {                                 \
            const uint32_t r = u >> 2, seg = u & 3u;                                  \
            const bool ok = (row_base + r) < out_dim && ((kt) * 4u + seg) * 16u < in_dim; \
            pd_cp_async16((int*)((dst) + r * ROW + 16u + seg * 16u),                  \
                          data + (size_t)(ok ? row_base + r : 0u) * in_dim +          \
                              (kt) * 64u + seg * 16u, ok);                            \
        }
    #define PD_N2_ISSUE_Y(dst, kt)                                                    \
        for (uint32_t u = tid; u < 1024u; u += 256u) {                                \
            const uint32_t col = u >> 2, seg = u & 3u;                                \
            const bool ok = (col_base + col) < batch && ((kt) * 4u + seg) * 16u < in_dim; \
            pd_cp_async16((int*)((dst) + col * ROW + 16u + seg * 16u),                \
                          xq + (size_t)(ok ? col_base + col : 0u) * in_dim +          \
                              (kt) * 64u + seg * 16u, ok);                            \
        }

    #pragma unroll
    for (uint32_t s = 0; s < STAGES - 1u; ++s) {
        if (s < nk) { PD_N2_ISSUE_W(PD_N2_WBUF(s), s) PD_N2_ISSUE_Y(PD_N2_YBUF(s), s) }
        asm volatile("cp.async.commit_group;");
    }
    for (uint32_t kt = 0; kt < nk; ++kt) {
        unsigned char* tw = PD_N2_WBUF(kt);
        unsigned char* ty = PD_N2_YBUF(kt);
        const uint32_t pf = kt + STAGES - 1u;
        if (pf < nk) { PD_N2_ISSUE_W(PD_N2_WBUF(pf), pf) PD_N2_ISSUE_Y(PD_N2_YBUF(pf), pf) }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group %0;" ::"n"(STAGES - 1u));
        {
            const uint32_t row = tid >> 1, kb = tid & 1u;
            const bool wok = (row_base + row) < out_dim && kt * 2u + kb < n_kb;
            tw[row * ROW + kb] =
                wok ? scale[(size_t)(row_base + row) * n_kb + kt * 2u + kb] : 0u;
            // two 128-col activation strips share the ty buffer (rows 0..255)
            #pragma unroll
            for (uint32_t st = 0; st < 2u; ++st) {
                const uint32_t crow = st * 128u + row;
                const bool yok = (col_base + crow) < batch && kt * 2u + kb < n_kb;
                ty[crow * ROW + kb] =
                    yok ? xs[(size_t)(col_base + crow) * n_kb + kt * 2u + kb] : 0u;
            }
        }
        __syncthreads();

        // weight fragments loaded once, reused across both strips
        uint32_t am[4][2][4];
        uint32_t sa[4];
        #pragma unroll
        for (uint32_t s = 0; s < 4u; ++s) {
            const uint32_t r0 = i0 + s * 16u + g;
            const uint32_t rr = i0 + s * 16u + ((lane >> 3) & 1u) * 8u + (lane & 7u);
            #pragma unroll
            for (uint32_t kb = 0; kb < 2u; ++kb)
                pd_ldm_x4(am[s][kb], tw + rr * ROW + 16u + kb * 32u + (lane >> 4) * 16u);
            const uint32_t rs = (tq & 1u) ? r0 + 8u : r0;
            sa[s] = *(const unsigned short*)(tw + rs * ROW);
        }
        #pragma unroll
        for (uint32_t st = 0; st < 2u; ++st) {
            unsigned char* tys = ty + st * 128u * ROW;
            #pragma unroll
            for (uint32_t j = 0; j < 4u; ++j) {
                uint32_t bm[4];
                pd_ldm_x4(bm, tys + (c0w + j * 8u + (lane & 7u)) * ROW + 16u +
                                  (lane >> 3) * 16u);
                const uint32_t sb = *(const unsigned short*)(tys + (c0w + j * 8u + g) * ROW);
                #pragma unroll
                for (uint32_t s = 0; s < 4u; ++s) {
                    pd_bs_mma_w8_kb<0>(acc[st][s * 4u + j], am[s][0][0], am[s][0][1],
                                       am[s][0][2], am[s][0][3], bm[0], bm[1], sa[s], sb);
                    pd_bs_mma_w8_kb<1>(acc[st][s * 4u + j], am[s][1][0], am[s][1][1],
                                       am[s][1][2], am[s][1][3], bm[2], bm[3], sa[s], sb);
                }
            }
        }
        __syncthreads();
    }
    #undef PD_N2_ISSUE_W
    #undef PD_N2_ISSUE_Y
    #undef PD_N2_WBUF
    #undef PD_N2_YBUF

    #pragma unroll
    for (uint32_t st = 0; st < 2u; ++st) {
        #pragma unroll
        for (uint32_t j = 0; j < 4u; ++j) {
            const uint32_t c0 = col_base + st * 128u + c0w + j * 8u + 2u * tq;
            #pragma unroll
            for (uint32_t s = 0; s < 4u; ++s) {
                const uint32_t r0 = row_base + i0 + s * 16u + g;
                const uint32_t r8 = r0 + 8u;
                if (r0 < out_dim) {
                    if (c0 < batch) y[(size_t)c0 * out_dim + r0] = acc[st][s * 4u + j][0];
                    if (c0 + 1u < batch) y[(size_t)(c0 + 1u) * out_dim + r0] = acc[st][s * 4u + j][1];
                }
                if (r8 < out_dim) {
                    if (c0 < batch) y[(size_t)c0 * out_dim + r8] = acc[st][s * 4u + j][2];
                    if (c0 + 1u < batch) y[(size_t)(c0 + 1u) * out_dim + r8] = acc[st][s * 4u + j][3];
                }
            }
        }
    }
#else
    (void)data; (void)scale; (void)xq; (void)xs; (void)y;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}

// Warp-specialized (CUTLASS-cooperative-style) variant: 384 threads = 8
// CONSUMER warps (identical per-tile math + accumulation order to
// pd_f8_gemm_w8_kt -> bit-identical output) + 4 PRODUCER warps that own all
// global->shared staging over a STAGES-deep ring. Why (measured): a
// CUTLASS-style sm120 blockwise kernel hits 76.7% tensor SOL at the same
// 1 CTA/SM and 128x128 tile by splitting load from MMA - our uniform-warp
// kernels stall at ~45% because every warp also issues cp.async and
// crosses a full-CTA
// __syncthreads twice per K-chunk. Handoff per stage b uses named barriers,
// both with count 384 (a barrier completes when sync+arrive arrivals reach
// the count): full(b)=1+b, producers bar.arrive after their cp.async
// wait_group lands chunk kt (bar gives the cross-warp smem ordering),
// consumers bar.sync; empty(b)=1+STAGES+b, consumers bar.arrive when done
// MMAing the stage, producers bar.sync before overwriting it. Producer flow
// control: at iter kt the ring holds chunks kt..kt+S-1 committed, so
// wait_group S-1 == "chunk kt's data landed". Consumers never touch cp.async
// state; producers never touch accumulators - consumer register pressure
// drops and the tensor pipe issues back-to-back.
// v2 geometry (v1 profiled at 47% of warp time in barrier stall, tensor 33%):
// the handoff unit is a K-128 SUPER-CHUNK (two K64 slabs, one FULL/EMPTY pair
// per 128 K = half v1's barrier crossings, 2x the MMA work amortizing each
// stall), and the producer loads the super-chunk's scale bytes into REGISTERS
// at iteration start so their ~500-cycle global latency hides behind the
// empty-wait + issue + wait_group instead of sitting between wait_group and
// the full arrive (v1 put them on the handoff critical path). Ring = 2 pairs
// = 4 K64 slabs = 80 KB.
__global__ void __launch_bounds__(384, 1) pd_f8_gemm_w8_ws_kt(
    const unsigned char* __restrict__ data, const unsigned char* __restrict__ scale,
    const unsigned char* __restrict__ xq, const unsigned char* __restrict__ xs,
    float* __restrict__ y, uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
#if PD_BS_OK
    constexpr uint32_t ROW = PD_BS_W8_ROW;
    constexpr uint32_t PAIRS = 2u;
    extern __shared__ unsigned char pd_bsw8_sh[];
    unsigned char* wring = pd_bsw8_sh;                        // PAIRS*2 x 128 x ROW
    unsigned char* yring = wring + PAIRS * 2u * 128u * ROW;   // PAIRS*2 x 128 x ROW

    const uint32_t tid = threadIdx.x;
    const uint32_t n_kb = in_dim >> 5;
    const uint32_t nk = (in_dim + 63u) / 64u;
    const uint32_t nsp = (nk + 1u) >> 1;  // K-128 super-chunks
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t nct = batch_pad >> 7;
    const uint32_t tile = blockIdx.x;
    const uint32_t row_base = (tile / nct) * 128u;
    const uint32_t col_base = (tile % nct) * 128u;

    if (tid >= 256u) {
        // ---------------- producer warps 8-11: stage the ring ----------------
        const uint32_t ptid = tid - 256u;
        #define PD_WS_ISSUE_W(dst, kt)                                                \
            for (uint32_t u = ptid; u < 512u; u += 128u) {                            \
                const uint32_t r = u >> 2, seg = u & 3u;                              \
                const bool ok = (row_base + r) < out_dim && ((kt) * 4u + seg) * 16u < in_dim; \
                pd_cp_async16((int*)((dst) + r * ROW + 16u + seg * 16u),              \
                              data + (size_t)(ok ? row_base + r : 0u) * in_dim +      \
                                  (kt) * 64u + seg * 16u, ok);                        \
            }
        #define PD_WS_ISSUE_Y(dst, kt)                                                \
            for (uint32_t u = ptid; u < 512u; u += 128u) {                            \
                const uint32_t col = u >> 2, seg = u & 3u;                            \
                const bool ok = (col_base + col) < batch && ((kt) * 4u + seg) * 16u < in_dim; \
                pd_cp_async16((int*)((dst) + col * ROW + 16u + seg * 16u),            \
                              xq + (size_t)(ok ? col_base + col : 0u) * in_dim +      \
                                  (kt) * 64u + seg * 16u, ok);                        \
            }
        // Loop is ROTATED so full(pair sp) is signalled the moment sp's data
        // lands, before any work for pair sp+1 (v2 ordered it after the next
        // empty-wait, which chained the consumer's full behind its own empty
        // release = a hidden serialization). Scale bytes for pair sp+1 preload
        // into regs right after the full arrive, so their global latency hides
        // behind the consumer's chew of pair sp.
        #define PD_WS_SCALES(sp_, sw_, sy_)                                           \
            _Pragma("unroll")                                                         \
            for (uint32_t h = 0; h < 2u; ++h) {                                       \
                _Pragma("unroll")                                                     \
                for (uint32_t kb = 0; kb < 2u; ++kb) {                                \
                    const uint32_t kt = (sp_) * 2u + h;                               \
                    const bool wok = (row_base + ptid) < out_dim && kt * 2u + kb < n_kb; \
                    sw_[h][kb] = wok ? scale[(size_t)(row_base + ptid) * n_kb + kt * 2u + kb] : 0u; \
                    const bool yok = (col_base + ptid) < batch && kt * 2u + kb < n_kb; \
                    sy_[h][kb] = yok ? xs[(size_t)(col_base + ptid) * n_kb + kt * 2u + kb] : 0u; \
                }                                                                     \
            }
        #define PD_WS_ISSUE_PAIR(p_, buf_)                                            \
            _Pragma("unroll")                                                         \
            for (uint32_t h = 0; h < 2u; ++h) {                                       \
                const uint32_t kt = (p_) * 2u + h;                                    \
                if (kt < nk) {                                                        \
                    PD_WS_ISSUE_W(wring + ((buf_) * 2u + h) * 128u * ROW, kt)         \
                    PD_WS_ISSUE_Y(yring + ((buf_) * 2u + h) * 128u * ROW, kt)         \
                }                                                                     \
            }
        unsigned char sw[2][2], sy[2][2];
        PD_WS_SCALES(0u, sw, sy)
        PD_WS_ISSUE_PAIR(0u, 0u)
        asm volatile("cp.async.commit_group;");
        for (uint32_t sp = 0; sp < nsp; ++sp) {
            const uint32_t b = sp % PAIRS;
            // pair sp is the only outstanding group; drain it, hand it over
            asm volatile("cp.async.wait_group 0;");
            #pragma unroll
            for (uint32_t h = 0; h < 2u; ++h) {
                unsigned char* tw = wring + (b * 2u + h) * 128u * ROW;
                unsigned char* ty = yring + (b * 2u + h) * 128u * ROW;
                #pragma unroll
                for (uint32_t kb = 0; kb < 2u; ++kb) {
                    tw[ptid * ROW + kb] = sw[h][kb];
                    ty[ptid * ROW + kb] = sy[h][kb];
                }
            }
            asm volatile("bar.arrive %0, 384;" ::"r"(1u + b));
            if (sp + 1u < nsp) {
                PD_WS_SCALES(sp + 1u, sw, sy)
                const uint32_t bp = (sp + 1u) % PAIRS;
                // buffer bp last held pair sp-1; consumers released it before
                // starting pair sp, so this rarely blocks
                if (sp > 0u) asm volatile("bar.sync %0, 384;" ::"r"(1u + PAIRS + bp));
                PD_WS_ISSUE_PAIR(sp + 1u, bp)
                asm volatile("cp.async.commit_group;");
            }
        }
        #undef PD_WS_ISSUE_PAIR
        #undef PD_WS_SCALES
        #undef PD_WS_ISSUE_W
        #undef PD_WS_ISSUE_Y
        return;
    }

    // ---------------- consumer warps 0-7: pure ldmatrix + MMA ----------------
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, tq = lane & 3u;
    const uint32_t i0 = (warp & 1u) * 64u;
    const uint32_t c0w = (warp >> 1) * 32u;

    float acc[16][4] = {};

    for (uint32_t sp = 0; sp < nsp; ++sp) {
        const uint32_t b = sp % PAIRS;
        asm volatile("bar.sync %0, 384;" ::"r"(1u + b));
        #pragma unroll
        for (uint32_t h = 0; h < 2u; ++h) {
            const uint32_t kt = sp * 2u + h;
            if (kt >= nk) break;
            const unsigned char* tw = wring + (b * 2u + h) * 128u * ROW;
            const unsigned char* ty = yring + (b * 2u + h) * 128u * ROW;

            uint32_t am[4][2][4];
            uint32_t sa[4];
            #pragma unroll
            for (uint32_t s = 0; s < 4u; ++s) {
                const uint32_t r0 = i0 + s * 16u + g;
                const uint32_t rr = i0 + s * 16u + ((lane >> 3) & 1u) * 8u + (lane & 7u);
                #pragma unroll
                for (uint32_t kb = 0; kb < 2u; ++kb)
                    pd_ldm_x4(am[s][kb], tw + rr * ROW + 16u + kb * 32u + (lane >> 4) * 16u);
                const uint32_t rs = (tq & 1u) ? r0 + 8u : r0;
                sa[s] = *(const unsigned short*)(tw + rs * ROW);
            }
            // preload all B fragments + scales for the slab, then run the 32
            // MMAs as one unbroken stream (v3 interleaved a ldmatrix + u16
            // load into every j step -> wait 3.7 + mio_throttle 2.3 of
            // 12.8 cycles/instr; MMA ORDER is unchanged so output stays
            // bit-identical)
            uint32_t bmj[4][4], sbj[4];
            #pragma unroll
            for (uint32_t j = 0; j < 4u; ++j) {
                pd_ldm_x4(bmj[j], ty + (c0w + j * 8u + (lane & 7u)) * ROW + 16u +
                                      (lane >> 3) * 16u);
                sbj[j] = *(const unsigned short*)(ty + (c0w + j * 8u + g) * ROW);
            }
            // kb passes SPLIT: all 16 kb0 MMAs then all 16 kb1. v4 issued
            // kb0->kb1 back-to-back on the same accumulator (hard register
            // dependency every 2nd MMA = the `wait` stall); per-acc order
            // is still kb0-then-kb1 so the sum stays bit-identical.
            #pragma unroll
            for (uint32_t j = 0; j < 4u; ++j)
                #pragma unroll
                for (uint32_t s = 0; s < 4u; ++s)
                    pd_bs_mma_w8_kb<0>(acc[s * 4u + j], am[s][0][0], am[s][0][1],
                                       am[s][0][2], am[s][0][3], bmj[j][0], bmj[j][1],
                                       sa[s], sbj[j]);
            #pragma unroll
            for (uint32_t j = 0; j < 4u; ++j)
                #pragma unroll
                for (uint32_t s = 0; s < 4u; ++s)
                    pd_bs_mma_w8_kb<1>(acc[s * 4u + j], am[s][1][0], am[s][1][1],
                                       am[s][1][2], am[s][1][3], bmj[j][2], bmj[j][3],
                                       sa[s], sbj[j]);
        }
        asm volatile("bar.arrive %0, 384;" ::"r"(1u + PAIRS + b));
    }

    #pragma unroll
    for (uint32_t j = 0; j < 4u; ++j) {
        const uint32_t c0 = col_base + c0w + j * 8u + 2u * tq;
        #pragma unroll
        for (uint32_t s = 0; s < 4u; ++s) {
            const uint32_t r0 = row_base + i0 + s * 16u + g;
            const uint32_t r8 = r0 + 8u;
            if (r0 < out_dim) {
                if (c0 < batch) y[(size_t)c0 * out_dim + r0] = acc[s * 4u + j][0];
                if (c0 + 1u < batch) y[(size_t)(c0 + 1u) * out_dim + r0] = acc[s * 4u + j][1];
            }
            if (r8 < out_dim) {
                if (c0 < batch) y[(size_t)c0 * out_dim + r8] = acc[s * 4u + j][2];
                if (c0 + 1u < batch) y[(size_t)(c0 + 1u) * out_dim + r8] = acc[s * 4u + j][3];
            }
        }
    }
#else
    (void)data; (void)scale; (void)xq; (void)xs; (void)y;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}

// TMA variant of the warp-specialized kernel (profiling ws v5:
// lts__t_requests 76% vs lts__t_sectors 59% = the L2 is REQUEST-RATE-bound on
// our 16B cp.asyncs, exactly the sub-throughput CUTLASS avoids with
// bulk-tensor loads). One 128x128-BYTE box per plane per K-128 pair - 2 TMA
// requests replace ~2048 cp.asyncs - written SWIZZLE_128B so the packed
// 128B rows stay ldmatrix-bank-conflict-free (16B chunk c of row r lands at
// c^(r&7)). Completion rides the mbarrier (complete_tx): producers never
// wait for data at all - they store the pair's scale bytes, arrive (elected
// thread arrives with expect_tx then issues the two boxes), and move on;
// the mbarrier phase completes when 128 arrivals + 32 KB of tx have landed.
// Empty stays a named barrier. Consumer math/order identical to the uniform
// kernels -> bit-identical output (OOB box regions zero-fill, same as the
// guarded-zero cp.async predicates).
// Win (write-windowing rung, MEASURED NEGATIVE - kept opt-in): on the
// hw-block-scale path the consumer's last smem read of a buffer is the h=1
// fragment/scale block, so empty can be released there instead of after the
// mma phase, aiming the producer's TMA refill at this pair's mma window
// instead of the next pair's ldmatrix burst (the clock64-measured ~220c/pair
// smem write-vs-read contention). Locked-clock A/B gate/2048 x5: 407.8 vs
// 420.2 TF (-3%), bit-identical. TMA delivery spans the whole pair cadence,
// so early release only reshuffles the same overlap - and the mid-loop
// bar.arrive likely fences the loads/mma interleave on top.
template <bool WIN = false, bool O16 = false>
__global__ void __launch_bounds__(384, 1) pd_f8_gemm_w8_tma_kt(
    const __grid_constant__ PdTmap wmap, const __grid_constant__ PdTmap ymap,
    const unsigned char* __restrict__ scale, const unsigned char* __restrict__ xs,
    float* __restrict__ y, uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
#if PD_F8W8_TMA_OK
    constexpr uint32_t PAIR16 = 16384u;  // one 128x128B data plane
    extern __shared__ __align__(128) unsigned char pd_bsw8t_sh[];
    unsigned char* wdat = pd_bsw8t_sh;             // 2 pairs x 16 KB
    unsigned char* ydat = pd_bsw8t_sh + 32768u;    // 2 pairs x 16 KB
    unsigned char* wsc = pd_bsw8t_sh + 65536u;     // 2 pairs x 2 slabs x 256 B
    unsigned char* ysc = pd_bsw8t_sh + 66560u;
    unsigned long long* mb = (unsigned long long*)(pd_bsw8t_sh + 67584u);

    const uint32_t tid = threadIdx.x;
    const uint32_t n_kb = in_dim >> 5;
    const uint32_t nk = (in_dim + 63u) / 64u;
    const uint32_t nsp = (nk + 1u) >> 1;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t nct = batch_pad >> 7;
    const uint32_t tile = blockIdx.x;
    const uint32_t row_base = (tile / nct) * 128u;
    const uint32_t col_base = (tile % nct) * 128u;

    if (tid == 0u) {
        const uint32_t m0 = (uint32_t)__cvta_generic_to_shared(mb);
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 128;" ::"r"(m0));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 128;" ::"r"(m0 + 8u));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    asm volatile("bar.sync 0, 384;");

    if (tid >= 256u) {
        // ---------------- producer warps 8-11 ----------------
        const uint32_t ptid = tid - 256u;
        unsigned char swr[2][2], syr[2][2];
        for (uint32_t sp = 0; sp < nsp; ++sp) {
            const uint32_t b = sp % 2u;
            // scale bytes for pair sp -> regs (guarded-zero, same math as
            // the uniform kernels)
            #pragma unroll
            for (uint32_t h = 0; h < 2u; ++h) {
                #pragma unroll
                for (uint32_t kb = 0; kb < 2u; ++kb) {
                    const uint32_t kt = sp * 2u + h;
                    const bool wok = (row_base + ptid) < out_dim && kt * 2u + kb < n_kb;
                    swr[h][kb] = wok ? scale[(size_t)(row_base + ptid) * n_kb + kt * 2u + kb] : 0u;
                    const bool yok = (col_base + ptid) < batch && kt * 2u + kb < n_kb;
                    syr[h][kb] = yok ? xs[(size_t)(col_base + ptid) * n_kb + kt * 2u + kb] : 0u;
                }
            }
            // buffer b last held pair sp-2; consumers released it via empty
            if (sp >= 2u) asm volatile("bar.sync %0, 384;" ::"r"(1u + b));
            #pragma unroll
            for (uint32_t h = 0; h < 2u; ++h) {
                #pragma unroll
                for (uint32_t kb = 0; kb < 2u; ++kb) {
                    wsc[b * 512u + h * 256u + ptid * 2u + kb] = swr[h][kb];
                    ysc[b * 512u + h * 256u + ptid * 2u + kb] = syr[h][kb];
                }
            }
            const uint32_t m = (uint32_t)__cvta_generic_to_shared(mb) + b * 8u;
            if (ptid == 0u) {
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 32768;" ::"r"(m));
                const uint32_t wd = (uint32_t)__cvta_generic_to_shared(wdat + b * PAIR16);
                const uint32_t yd = (uint32_t)__cvta_generic_to_shared(ydat + b * PAIR16);
                const int ck = (int)(sp * 128u);
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3}], [%4];" ::"r"(wd),
                    "l"(&wmap), "r"(ck), "r"((int)row_base), "r"(m)
                    : "memory");
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3}], [%4];" ::"r"(yd),
                    "l"(&ymap), "r"(ck), "r"((int)col_base), "r"(m)
                    : "memory");
            } else {
                asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(m));
            }
        }
        return;
    }

    // ---------------- consumer warps 0-7 ----------------
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, tq = lane & 3u;
    const uint32_t i0 = (warp & 1u) * 64u;
    const uint32_t c0w = (warp >> 1) * 32u;

    float acc[16][4] = {};
    uint32_t ph0 = 0u, ph1 = 0u;

    for (uint32_t sp = 0; sp < nsp; ++sp) {
        const uint32_t b = sp % 2u;
        const uint32_t m = (uint32_t)__cvta_generic_to_shared(mb) + b * 8u;
        const uint32_t ph = (b == 0u) ? ph0 : ph1;
        asm volatile(
            "{\n\t.reg .pred P;\n"
            "PD_TMA_WAIT_%=:\n\t"
            "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
            "@!P bra PD_TMA_WAIT_%=;\n\t}" ::"r"(m), "r"(ph));
        if (b == 0u) ph0 ^= 1u; else ph1 ^= 1u;

        const unsigned char* wp = wdat + b * PAIR16;
        const unsigned char* yp = ydat + b * PAIR16;
        #pragma unroll
        for (uint32_t h = 0; h < 2u; ++h) {
            const uint32_t kt = sp * 2u + h;
            if (kt >= nk) break;

            uint32_t am[4][2][4];
#if PD_BS_OK
            uint32_t sa[4];
#endif
            #pragma unroll
            for (uint32_t s = 0; s < 4u; ++s) {
                const uint32_t rr = i0 + s * 16u + ((lane >> 3) & 1u) * 8u + (lane & 7u);
                #pragma unroll
                for (uint32_t kb = 0; kb < 2u; ++kb) {
                    const uint32_t c = h * 4u + kb * 2u + (lane >> 4);
                    pd_ldm_x4(am[s][kb], wp + rr * 128u + ((c ^ (rr & 7u)) * 16u));
                }
#if PD_BS_OK
                const uint32_t r0 = i0 + s * 16u + g;
                const uint32_t rs = (tq & 1u) ? r0 + 8u : r0;
                sa[s] = *(const unsigned short*)(wsc + b * 512u + h * 256u + rs * 2u);
#endif
            }
            uint32_t bmj[4][4];
#if PD_BS_OK
            uint32_t sbj[4];
#endif
            #pragma unroll
            for (uint32_t j = 0; j < 4u; ++j) {
                const uint32_t col = c0w + j * 8u + (lane & 7u);
                const uint32_t c = h * 4u + (lane >> 3);
                pd_ldm_x4(bmj[j], yp + col * 128u + ((c ^ (col & 7u)) * 16u));
#if PD_BS_OK
                sbj[j] = *(const unsigned short*)(ysc + b * 512u + h * 256u + (c0w + j * 8u + g) * 2u);
#endif
            }
#if PD_BS_OK
            // Win: last smem read of buffer b just retired into registers on
            // this pair's final half - release empty now so the refill TMA
            // rides the mma window. (h==1 on full pairs; h==0 on an odd-nk
            // tail pair whose h=1 never runs.) sw-fold path excluded: it
            // reads scale slabs inside the mma loop below.
            if (WIN && (h == 1u || sp * 2u + h + 1u >= nk))
                asm volatile("bar.arrive %0, 384;" ::"r"(1u + b));
            #pragma unroll
            for (uint32_t j = 0; j < 4u; ++j)
                #pragma unroll
                for (uint32_t s = 0; s < 4u; ++s)
                    pd_bs_mma_w8_kb<0>(acc[s * 4u + j], am[s][0][0], am[s][0][1],
                                       am[s][0][2], am[s][0][3], bmj[j][0], bmj[j][1],
                                       sa[s], sbj[j]);
            #pragma unroll
            for (uint32_t j = 0; j < 4u; ++j)
                #pragma unroll
                for (uint32_t s = 0; s < 4u; ++s)
                    pd_bs_mma_w8_kb<1>(acc[s * 4u + j], am[s][1][0], am[s][1][1],
                                       am[s][1][2], am[s][1][3], bmj[j][2], bmj[j][3],
                                       sa[s], sbj[j]);
#else
            // sw fold (non-120a FP8 arches): quad scale bytes read direct
            // from the staged slabs - see pd_f8_mma_sw for why the hw call's
            // lane-distributed words must not be reused here
            #pragma unroll
            for (uint32_t j = 0; j < 4u; ++j) {
                const uint32_t c0 = c0w + j * 8u + 2u * tq;
                #pragma unroll
                for (uint32_t s = 0; s < 4u; ++s) {
                    const uint32_t r0 = i0 + s * 16u + g;
                    #pragma unroll
                    for (uint32_t kb = 0; kb < 2u; ++kb) {
                        const unsigned char* wsb = wsc + b * 512u + h * 256u;
                        const unsigned char* ysb = ysc + b * 512u + h * 256u;
                        pd_f8_mma_sw(acc[s * 4u + j], am[s][kb][0], am[s][kb][1],
                                     am[s][kb][2], am[s][kb][3], bmj[j][kb * 2u],
                                     bmj[j][kb * 2u + 1u], wsb[r0 * 2u + kb],
                                     wsb[(r0 + 8u) * 2u + kb], ysb[c0 * 2u + kb],
                                     ysb[(c0 + 1u) * 2u + kb]);
                    }
                }
            }
#endif
        }
#if PD_BS_OK
        if (!WIN) asm volatile("bar.arrive %0, 384;" ::"r"(1u + b));
#else
        asm volatile("bar.arrive %0, 384;" ::"r"(1u + b));
#endif
    }

    #pragma unroll
    for (uint32_t j = 0; j < 4u; ++j) {
        const uint32_t c0 = col_base + c0w + j * 8u + 2u * tq;
        #pragma unroll
        for (uint32_t s = 0; s < 4u; ++s) {
            const uint32_t r0 = row_base + i0 + s * 16u + g;
            const uint32_t r8 = r0 + 8u;
            // O16: bf16 out - halves the epilogue write (the gate/up plane
            // otherwise stores 143 MB/layer-tick of f32 for nothing).
            // Consumers read bf16 (quantize_e4m3_swiglu_b16).
            __nv_bfloat16* yh = (__nv_bfloat16*)y;
            if (r0 < out_dim) {
                if (c0 < batch) { if (O16) yh[(size_t)c0 * out_dim + r0] = __float2bfloat16(acc[s * 4u + j][0]); else y[(size_t)c0 * out_dim + r0] = acc[s * 4u + j][0]; }
                if (c0 + 1u < batch) { if (O16) yh[(size_t)(c0 + 1u) * out_dim + r0] = __float2bfloat16(acc[s * 4u + j][1]); else y[(size_t)(c0 + 1u) * out_dim + r0] = acc[s * 4u + j][1]; }
            }
            if (r8 < out_dim) {
                if (c0 < batch) { if (O16) yh[(size_t)c0 * out_dim + r8] = __float2bfloat16(acc[s * 4u + j][2]); else y[(size_t)c0 * out_dim + r8] = acc[s * 4u + j][2]; }
                if (c0 + 1u < batch) { if (O16) yh[(size_t)(c0 + 1u) * out_dim + r8] = __float2bfloat16(acc[s * 4u + j][3]); else y[(size_t)(c0 + 1u) * out_dim + r8] = acc[s * 4u + j][3]; }
            }
        }
    }
#else
    (void)wmap; (void)ymap; (void)scale; (void)xs; (void)y;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}

// ---- JIT-B isolation rung: the only change vs
// pd_f8_gemm_w8_tma_kt is the consumer's B-side: fragments + scale load
// just-in-time per j-group inside the mma stream (MIO spread) instead of a
// front-loaded burst. Pipeline, barriers and math order untouched ->
// bit-identical. Ring-depth variants are all falsified; the first stall
// reading misattributed producer idle-at-barrier to the pipeline.
__global__ void __launch_bounds__(384, 1) pd_f8_gemm_w8_tma4_kt(
    const __grid_constant__ PdTmap wmap, const __grid_constant__ PdTmap ymap,
    const unsigned char* __restrict__ scale, const unsigned char* __restrict__ xs,
    float* __restrict__ y, uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
#if PD_BS_OK
    constexpr uint32_t PAIR16 = 16384u;  // one 128x128B data plane
    extern __shared__ __align__(128) unsigned char pd_bsw8t4_sh[];
    unsigned char* wdat = pd_bsw8t4_sh;             // 2 pairs x 16 KB
    unsigned char* ydat = pd_bsw8t4_sh + 32768u;    // 2 pairs x 16 KB
    unsigned char* wsc = pd_bsw8t4_sh + 65536u;     // 2 pairs x 2 slabs x 256 B
    unsigned char* ysc = pd_bsw8t4_sh + 66560u;
    unsigned long long* mb = (unsigned long long*)(pd_bsw8t4_sh + 67584u);

    const uint32_t tid = threadIdx.x;
    const uint32_t n_kb = in_dim >> 5;
    const uint32_t nk = (in_dim + 63u) / 64u;
    const uint32_t nsp = (nk + 1u) >> 1;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t nct = batch_pad >> 7;
    const uint32_t tile = blockIdx.x;
    const uint32_t row_base = (tile / nct) * 128u;
    const uint32_t col_base = (tile % nct) * 128u;

    if (tid == 0u) {
        const uint32_t m0 = (uint32_t)__cvta_generic_to_shared(mb);
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 128;" ::"r"(m0));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 128;" ::"r"(m0 + 8u));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    asm volatile("bar.sync 0, 384;");

    if (tid >= 256u) {
        // ---------------- producer warps 8-11 ----------------
        const uint32_t ptid = tid - 256u;
        unsigned char swr[2][2], syr[2][2];
        for (uint32_t sp = 0; sp < nsp; ++sp) {
            const uint32_t b = sp % 2u;
            // scale bytes for pair sp -> regs (guarded-zero, same math as
            // the uniform kernels)
            #pragma unroll
            for (uint32_t h = 0; h < 2u; ++h) {
                #pragma unroll
                for (uint32_t kb = 0; kb < 2u; ++kb) {
                    const uint32_t kt = sp * 2u + h;
                    const bool wok = (row_base + ptid) < out_dim && kt * 2u + kb < n_kb;
                    swr[h][kb] = wok ? scale[(size_t)(row_base + ptid) * n_kb + kt * 2u + kb] : 0u;
                    const bool yok = (col_base + ptid) < batch && kt * 2u + kb < n_kb;
                    syr[h][kb] = yok ? xs[(size_t)(col_base + ptid) * n_kb + kt * 2u + kb] : 0u;
                }
            }
            // buffer b last held pair sp-2; consumers released it via empty
            if (sp >= 2u) asm volatile("bar.sync %0, 384;" ::"r"(1u + b));
            #pragma unroll
            for (uint32_t h = 0; h < 2u; ++h) {
                #pragma unroll
                for (uint32_t kb = 0; kb < 2u; ++kb) {
                    wsc[b * 512u + h * 256u + ptid * 2u + kb] = swr[h][kb];
                    ysc[b * 512u + h * 256u + ptid * 2u + kb] = syr[h][kb];
                }
            }
            const uint32_t m = (uint32_t)__cvta_generic_to_shared(mb) + b * 8u;
            if (ptid == 0u) {
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 32768;" ::"r"(m));
                const uint32_t wd = (uint32_t)__cvta_generic_to_shared(wdat + b * PAIR16);
                const uint32_t yd = (uint32_t)__cvta_generic_to_shared(ydat + b * PAIR16);
                const int ck = (int)(sp * 128u);
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3}], [%4];" ::"r"(wd),
                    "l"(&wmap), "r"(ck), "r"((int)row_base), "r"(m)
                    : "memory");
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3}], [%4];" ::"r"(yd),
                    "l"(&ymap), "r"(ck), "r"((int)col_base), "r"(m)
                    : "memory");
            } else {
                asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(m));
            }
        }
        return;
    }

    // ---------------- consumer warps 0-7 ----------------
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, tq = lane & 3u;
    const uint32_t i0 = (warp & 1u) * 64u;
    const uint32_t c0w = (warp >> 1) * 32u;

    float acc[16][4] = {};
    uint32_t ph0 = 0u, ph1 = 0u;

    for (uint32_t sp = 0; sp < nsp; ++sp) {
        const uint32_t b = sp % 2u;
        const uint32_t m = (uint32_t)__cvta_generic_to_shared(mb) + b * 8u;
        const uint32_t ph = (b == 0u) ? ph0 : ph1;
        asm volatile(
            "{\n\t.reg .pred P;\n"
            "PD_TMA_WAIT_%=:\n\t"
            "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
            "@!P bra PD_TMA_WAIT_%=;\n\t}" ::"r"(m), "r"(ph));
        if (b == 0u) ph0 ^= 1u; else ph1 ^= 1u;

        const unsigned char* wp = wdat + b * PAIR16;
        const unsigned char* yp = ydat + b * PAIR16;
        #pragma unroll
        for (uint32_t h = 0; h < 2u; ++h) {
            const uint32_t kt = sp * 2u + h;
            if (kt >= nk) break;

            uint32_t am[4][2][4];
            uint32_t sa[4];
            #pragma unroll
            for (uint32_t s = 0; s < 4u; ++s) {
                const uint32_t r0 = i0 + s * 16u + g;
                const uint32_t rr = i0 + s * 16u + ((lane >> 3) & 1u) * 8u + (lane & 7u);
                #pragma unroll
                for (uint32_t kb = 0; kb < 2u; ++kb) {
                    const uint32_t c = h * 4u + kb * 2u + (lane >> 4);
                    pd_ldm_x4(am[s][kb], wp + rr * 128u + ((c ^ (rr & 7u)) * 16u));
                }
                const uint32_t rs = (tq & 1u) ? r0 + 8u : r0;
                sa[s] = *(const unsigned short*)(wsc + b * 512u + h * 256u + rs * 2u);
            }
            // JIT-B isolation rung: per j-group, load B then run its 8
            // mmas - spreads MIO through the mma stream; per-element
            // accumulation order unchanged (kb0 before kb1) -> bit-identical
            #pragma unroll
            for (uint32_t j = 0; j < 4u; ++j) {
                uint32_t bm[4];
                const uint32_t col = c0w + j * 8u + (lane & 7u);
                const uint32_t c = h * 4u + (lane >> 3);
                pd_ldm_x4(bm, yp + col * 128u + ((c ^ (col & 7u)) * 16u));
                const uint32_t sb = *(const unsigned short*)(ysc + b * 512u + h * 256u + (c0w + j * 8u + g) * 2u);
                #pragma unroll
                for (uint32_t s = 0; s < 4u; ++s)
                    pd_bs_mma_w8_kb<0>(acc[s * 4u + j], am[s][0][0], am[s][0][1],
                                       am[s][0][2], am[s][0][3], bm[0], bm[1], sa[s], sb);
                #pragma unroll
                for (uint32_t s = 0; s < 4u; ++s)
                    pd_bs_mma_w8_kb<1>(acc[s * 4u + j], am[s][1][0], am[s][1][1],
                                       am[s][1][2], am[s][1][3], bm[2], bm[3], sa[s], sb);
            }
        }
        asm volatile("bar.arrive %0, 384;" ::"r"(1u + b));
    }

    #pragma unroll
    for (uint32_t j = 0; j < 4u; ++j) {
        const uint32_t c0 = col_base + c0w + j * 8u + 2u * tq;
        #pragma unroll
        for (uint32_t s = 0; s < 4u; ++s) {
            const uint32_t r0 = row_base + i0 + s * 16u + g;
            const uint32_t r8 = r0 + 8u;
            if (r0 < out_dim) {
                if (c0 < batch) y[(size_t)c0 * out_dim + r0] = acc[s * 4u + j][0];
                if (c0 + 1u < batch) y[(size_t)(c0 + 1u) * out_dim + r0] = acc[s * 4u + j][1];
            }
            if (r8 < out_dim) {
                if (c0 < batch) y[(size_t)c0 * out_dim + r8] = acc[s * 4u + j][2];
                if (c0 + 1u < batch) y[(size_t)(c0 + 1u) * out_dim + r8] = acc[s * 4u + j][3];
            }
        }
    }
#else
    (void)wmap; (void)ymap; (void)scale; (void)xs; (void)y;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}

// ---- from-scratch tile geometry, candidate B (f8 inner loop) --------------
// The falsification record closed the pipeline-shape axis: the baseline is
// bounded by 2 mma-issuing warps/scheduler stalling on ldmatrix->mma latency
// with nothing co-resident to hide it. Candidate B doubles the mma-issuing
// warps without touching the L2-neutral 128x128 tile or the 2-stage TMA
// ring: 16 consumer warps of a 32x32 warp tile (acc 64 -> 32 f32/thread,
// A-fragments halve; ~103 regs fits the 65,536/576 = 113 budget) + 2
// producer warps (64 threads stage 2 scale rows each). Per-element K-chain
// identical to the baseline -> bit-identical output.
__global__ void __launch_bounds__(576, 1) pd_f8_gemm_w8_tma16_kt(
    const __grid_constant__ PdTmap wmap, const __grid_constant__ PdTmap ymap,
    const unsigned char* __restrict__ scale, const unsigned char* __restrict__ xs,
    float* __restrict__ y, uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
#if PD_BS_OK
    constexpr uint32_t PAIR16 = 16384u;
    extern __shared__ __align__(128) unsigned char pd_bsw8t16_sh[];
    unsigned char* wdat = pd_bsw8t16_sh;             // 2 pairs x 16 KB
    unsigned char* ydat = pd_bsw8t16_sh + 32768u;    // 2 pairs x 16 KB
    unsigned char* wsc = pd_bsw8t16_sh + 65536u;     // 2 pairs x 2 slabs x 256 B
    unsigned char* ysc = pd_bsw8t16_sh + 66560u;
    unsigned long long* mb = (unsigned long long*)(pd_bsw8t16_sh + 67584u);

    const uint32_t tid = threadIdx.x;
    const uint32_t n_kb = in_dim >> 5;
    const uint32_t nk = (in_dim + 63u) / 64u;
    const uint32_t nsp = (nk + 1u) >> 1;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t nct = batch_pad >> 7;
    const uint32_t tile = blockIdx.x;
    const uint32_t row_base = (tile / nct) * 128u;
    const uint32_t col_base = (tile % nct) * 128u;

    if (tid == 0u) {
        const uint32_t m0 = (uint32_t)__cvta_generic_to_shared(mb);
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 64;" ::"r"(m0));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 64;" ::"r"(m0 + 8u));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    asm volatile("bar.sync 0, 576;");

    if (tid >= 512u) {
        // ------------- producer warps 16-17: 2 scale rows/thread -------------
        const uint32_t ptid = tid - 512u;  // 0..63
        unsigned char swr[2][2][2], syr[2][2][2];  // [row-pass][h][kb]
        for (uint32_t sp = 0; sp < nsp; ++sp) {
            const uint32_t b = sp % 2u;
            #pragma unroll
            for (uint32_t rp = 0; rp < 2u; ++rp) {
                const uint32_t row = ptid + rp * 64u;
                #pragma unroll
                for (uint32_t h = 0; h < 2u; ++h) {
                    #pragma unroll
                    for (uint32_t kb = 0; kb < 2u; ++kb) {
                        const uint32_t kt = sp * 2u + h;
                        const bool wok = (row_base + row) < out_dim && kt * 2u + kb < n_kb;
                        swr[rp][h][kb] = wok ? scale[(size_t)(row_base + row) * n_kb + kt * 2u + kb] : 0u;
                        const bool yok = (col_base + row) < batch && kt * 2u + kb < n_kb;
                        syr[rp][h][kb] = yok ? xs[(size_t)(col_base + row) * n_kb + kt * 2u + kb] : 0u;
                    }
                }
            }
            if (sp >= 2u) asm volatile("bar.sync %0, 576;" ::"r"(1u + b));
            #pragma unroll
            for (uint32_t rp = 0; rp < 2u; ++rp) {
                const uint32_t row = ptid + rp * 64u;
                #pragma unroll
                for (uint32_t h = 0; h < 2u; ++h) {
                    #pragma unroll
                    for (uint32_t kb = 0; kb < 2u; ++kb) {
                        wsc[b * 512u + h * 256u + row * 2u + kb] = swr[rp][h][kb];
                        ysc[b * 512u + h * 256u + row * 2u + kb] = syr[rp][h][kb];
                    }
                }
            }
            const uint32_t m = (uint32_t)__cvta_generic_to_shared(mb) + b * 8u;
            if (ptid == 0u) {
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 32768;" ::"r"(m));
                const uint32_t wd = (uint32_t)__cvta_generic_to_shared(wdat + b * PAIR16);
                const uint32_t yd = (uint32_t)__cvta_generic_to_shared(ydat + b * PAIR16);
                const int ck = (int)(sp * 128u);
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3}], [%4];" ::"r"(wd),
                    "l"(&wmap), "r"(ck), "r"((int)row_base), "r"(m) : "memory");
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3}], [%4];" ::"r"(yd),
                    "l"(&ymap), "r"(ck), "r"((int)col_base), "r"(m) : "memory");
            } else {
                asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(m));
            }
        }
        return;
    }

    // -------- consumer warps 0-15: 32x32 warp tile (4 row x 4 col groups) ----
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, tq = lane & 3u;
    const uint32_t i0 = (warp & 3u) * 32u;
    const uint32_t c0w = (warp >> 2) * 32u;

    float acc[8][4] = {};
    uint32_t ph0 = 0u, ph1 = 0u;

    for (uint32_t sp = 0; sp < nsp; ++sp) {
        const uint32_t b = sp % 2u;
        const uint32_t m = (uint32_t)__cvta_generic_to_shared(mb) + b * 8u;
        const uint32_t ph = (b == 0u) ? ph0 : ph1;
        asm volatile(
            "{\n\t.reg .pred P;\n"
            "PD_TMA16_WAIT_%=:\n\t"
            "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
            "@!P bra PD_TMA16_WAIT_%=;\n\t}" ::"r"(m), "r"(ph));
        if (b == 0u) ph0 ^= 1u; else ph1 ^= 1u;

        const unsigned char* wp = wdat + b * PAIR16;
        const unsigned char* yp = ydat + b * PAIR16;
        #pragma unroll
        for (uint32_t h = 0; h < 2u; ++h) {
            const uint32_t kt = sp * 2u + h;
            if (kt >= nk) break;

            uint32_t am[2][2][4];
            uint32_t sa[2];
            #pragma unroll
            for (uint32_t s = 0; s < 2u; ++s) {
                const uint32_t r0 = i0 + s * 16u + g;
                const uint32_t rr = i0 + s * 16u + ((lane >> 3) & 1u) * 8u + (lane & 7u);
                #pragma unroll
                for (uint32_t kb = 0; kb < 2u; ++kb) {
                    const uint32_t c = h * 4u + kb * 2u + (lane >> 4);
                    pd_ldm_x4(am[s][kb], wp + rr * 128u + ((c ^ (rr & 7u)) * 16u));
                }
                const uint32_t rs = (tq & 1u) ? r0 + 8u : r0;
                sa[s] = *(const unsigned short*)(wsc + b * 512u + h * 256u + rs * 2u);
            }
            uint32_t bmj[4][4], sbj[4];
            #pragma unroll
            for (uint32_t j = 0; j < 4u; ++j) {
                const uint32_t col = c0w + j * 8u + (lane & 7u);
                const uint32_t c = h * 4u + (lane >> 3);
                pd_ldm_x4(bmj[j], yp + col * 128u + ((c ^ (col & 7u)) * 16u));
                sbj[j] = *(const unsigned short*)(ysc + b * 512u + h * 256u + (c0w + j * 8u + g) * 2u);
            }
            #pragma unroll
            for (uint32_t j = 0; j < 4u; ++j)
                #pragma unroll
                for (uint32_t s = 0; s < 2u; ++s)
                    pd_bs_mma_w8_kb<0>(acc[s * 4u + j], am[s][0][0], am[s][0][1],
                                       am[s][0][2], am[s][0][3], bmj[j][0], bmj[j][1],
                                       sa[s], sbj[j]);
            #pragma unroll
            for (uint32_t j = 0; j < 4u; ++j)
                #pragma unroll
                for (uint32_t s = 0; s < 2u; ++s)
                    pd_bs_mma_w8_kb<1>(acc[s * 4u + j], am[s][1][0], am[s][1][1],
                                       am[s][1][2], am[s][1][3], bmj[j][2], bmj[j][3],
                                       sa[s], sbj[j]);
        }
        asm volatile("bar.arrive %0, 576;" ::"r"(1u + b));
    }

    #pragma unroll
    for (uint32_t j = 0; j < 4u; ++j) {
        const uint32_t c0 = col_base + c0w + j * 8u + 2u * tq;
        #pragma unroll
        for (uint32_t s = 0; s < 2u; ++s) {
            const uint32_t r0 = row_base + i0 + s * 16u + g;
            const uint32_t r8 = r0 + 8u;
            if (r0 < out_dim) {
                if (c0 < batch) y[(size_t)c0 * out_dim + r0] = acc[s * 4u + j][0];
                if (c0 + 1u < batch) y[(size_t)(c0 + 1u) * out_dim + r0] = acc[s * 4u + j][1];
            }
            if (r8 < out_dim) {
                if (c0 < batch) y[(size_t)c0 * out_dim + r8] = acc[s * 4u + j][2];
                if (c0 + 1u < batch) y[(size_t)(c0 + 1u) * out_dim + r8] = acc[s * 4u + j][3];
            }
        }
    }
#else
    (void)wmap; (void)ymap; (void)scale; (void)xs; (void)y;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}

// ---- two-copy arrive-early split (f8 feed-latency rung) -----------------
// Ceiling probes exonerated the mix/scales/sync (763-780 TF smem-fed); the
// residual is TMA->L2 feed latency coupling. This is the baseline kernel
// with W and Y on separate mbarriers per stage: consumers wait W, run half
// 0's A-side burst (8 ldmatrix.x4 + scale words) under Y's in-flight tail,
// then wait Y. Two copies per pair, same as baseline (overpaid
// with 4). Per-element accumulation order unchanged -> bit-identical.
__global__ void __launch_bounds__(384, 1) pd_f8_gemm_w8_tma2s_kt(
    const __grid_constant__ PdTmap wmap, const __grid_constant__ PdTmap ymap,
    const unsigned char* __restrict__ scale, const unsigned char* __restrict__ xs,
    float* __restrict__ y, uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
#if PD_BS_OK
    constexpr uint32_t PAIR16 = 16384u;  // one 128x128B data plane
    extern __shared__ __align__(128) unsigned char pd_bsw8t2s_sh[];
    unsigned char* wdat = pd_bsw8t2s_sh;             // 2 pairs x 16 KB
    unsigned char* ydat = pd_bsw8t2s_sh + 32768u;    // 2 pairs x 16 KB
    unsigned char* wsc = pd_bsw8t2s_sh + 65536u;     // 2 pairs x 2 slabs x 256 B
    unsigned char* ysc = pd_bsw8t2s_sh + 66560u;
    unsigned long long* mb = (unsigned long long*)(pd_bsw8t2s_sh + 67584u);

    const uint32_t tid = threadIdx.x;
    const uint32_t n_kb = in_dim >> 5;
    const uint32_t nk = (in_dim + 63u) / 64u;
    const uint32_t nsp = (nk + 1u) >> 1;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t nct = batch_pad >> 7;
    const uint32_t tile = blockIdx.x;
    const uint32_t row_base = (tile / nct) * 128u;
    const uint32_t col_base = (tile % nct) * 128u;

    if (tid == 0u) {
        const uint32_t m0 = (uint32_t)__cvta_generic_to_shared(mb);
        #pragma unroll
        for (uint32_t i = 0; i < 4u; ++i)
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 128;" ::"r"(m0 + i * 8u));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    asm volatile("bar.sync 0, 384;");

    if (tid >= 256u) {
        // ---------------- producer warps 8-11 ----------------
        const uint32_t ptid = tid - 256u;
        unsigned char swr[2][2], syr[2][2];
        for (uint32_t sp = 0; sp < nsp; ++sp) {
            const uint32_t b = sp % 2u;
            // scale bytes for pair sp -> regs (guarded-zero, same math as
            // the uniform kernels)
            #pragma unroll
            for (uint32_t h = 0; h < 2u; ++h) {
                #pragma unroll
                for (uint32_t kb = 0; kb < 2u; ++kb) {
                    const uint32_t kt = sp * 2u + h;
                    const bool wok = (row_base + ptid) < out_dim && kt * 2u + kb < n_kb;
                    swr[h][kb] = wok ? scale[(size_t)(row_base + ptid) * n_kb + kt * 2u + kb] : 0u;
                    const bool yok = (col_base + ptid) < batch && kt * 2u + kb < n_kb;
                    syr[h][kb] = yok ? xs[(size_t)(col_base + ptid) * n_kb + kt * 2u + kb] : 0u;
                }
            }
            // buffer b last held pair sp-2; consumers released it via empty
            if (sp >= 2u) asm volatile("bar.sync %0, 384;" ::"r"(1u + b));
            #pragma unroll
            for (uint32_t h = 0; h < 2u; ++h) {
                #pragma unroll
                for (uint32_t kb = 0; kb < 2u; ++kb) {
                    wsc[b * 512u + h * 256u + ptid * 2u + kb] = swr[h][kb];
                    ysc[b * 512u + h * 256u + ptid * 2u + kb] = syr[h][kb];
                }
            }
            const uint32_t mw = (uint32_t)__cvta_generic_to_shared(mb) + b * 16u;
            const uint32_t my = mw + 8u;
            if (ptid == 0u) {
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 16384;" ::"r"(mw));
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 16384;" ::"r"(my));
                const uint32_t wd = (uint32_t)__cvta_generic_to_shared(wdat + b * PAIR16);
                const uint32_t yd = (uint32_t)__cvta_generic_to_shared(ydat + b * PAIR16);
                const int ck = (int)(sp * 128u);
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3}], [%4];" ::"r"(wd),
                    "l"(&wmap), "r"(ck), "r"((int)row_base), "r"(mw)
                    : "memory");
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3}], [%4];" ::"r"(yd),
                    "l"(&ymap), "r"(ck), "r"((int)col_base), "r"(my)
                    : "memory");
            } else {
                asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(mw));
                asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(my));
            }
        }
        return;
    }

    // ---------------- consumer warps 0-7 ----------------
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, tq = lane & 3u;
    const uint32_t i0 = (warp & 1u) * 64u;
    const uint32_t c0w = (warp >> 1) * 32u;

    float acc[16][4] = {};
    uint32_t ph0 = 0u, ph1 = 0u;

    for (uint32_t sp = 0; sp < nsp; ++sp) {
        const uint32_t b = sp % 2u;
        const uint32_t mw = (uint32_t)__cvta_generic_to_shared(mb) + b * 16u;
        const uint32_t my = mw + 8u;
        const uint32_t ph = (b == 0u) ? ph0 : ph1;
        asm volatile(
            "{\n\t.reg .pred P;\n"
            "PD_T2SW_%=:\n\t"
            "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
            "@!P bra PD_T2SW_%=;\n\t}" ::"r"(mw), "r"(ph));

        const unsigned char* wp = wdat + b * PAIR16;
        const unsigned char* yp = ydat + b * PAIR16;
        // half 0's A-side burst runs under Y's in-flight TMA tail
        uint32_t am0[4][2][4];
        uint32_t sa0[4];
        #pragma unroll
        for (uint32_t s = 0; s < 4u; ++s) {
            const uint32_t r0 = i0 + s * 16u + g;
            const uint32_t rr = i0 + s * 16u + ((lane >> 3) & 1u) * 8u + (lane & 7u);
            #pragma unroll
            for (uint32_t kb = 0; kb < 2u; ++kb) {
                const uint32_t c = kb * 2u + (lane >> 4);
                pd_ldm_x4(am0[s][kb], wp + rr * 128u + ((c ^ (rr & 7u)) * 16u));
            }
            const uint32_t rs = (tq & 1u) ? r0 + 8u : r0;
            sa0[s] = *(const unsigned short*)(wsc + b * 512u + rs * 2u);
        }
        asm volatile(
            "{\n\t.reg .pred P;\n"
            "PD_T2SY_%=:\n\t"
            "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
            "@!P bra PD_T2SY_%=;\n\t}" ::"r"(my), "r"(ph));
        if (b == 0u) ph0 ^= 1u; else ph1 ^= 1u;
        #pragma unroll
        for (uint32_t h = 0; h < 2u; ++h) {
            const uint32_t kt = sp * 2u + h;
            if (kt >= nk) break;

            uint32_t am[4][2][4];
            uint32_t sa[4];
            if (h == 0u) {
                #pragma unroll
                for (uint32_t s = 0; s < 4u; ++s) {
                    #pragma unroll
                    for (uint32_t kb = 0; kb < 2u; ++kb)
                        #pragma unroll
                        for (uint32_t q = 0; q < 4u; ++q) am[s][kb][q] = am0[s][kb][q];
                    sa[s] = sa0[s];
                }
            } else {
                #pragma unroll
                for (uint32_t s = 0; s < 4u; ++s) {
                    const uint32_t r0 = i0 + s * 16u + g;
                    const uint32_t rr = i0 + s * 16u + ((lane >> 3) & 1u) * 8u + (lane & 7u);
                    #pragma unroll
                    for (uint32_t kb = 0; kb < 2u; ++kb) {
                        const uint32_t c = h * 4u + kb * 2u + (lane >> 4);
                        pd_ldm_x4(am[s][kb], wp + rr * 128u + ((c ^ (rr & 7u)) * 16u));
                    }
                    const uint32_t rs = (tq & 1u) ? r0 + 8u : r0;
                    sa[s] = *(const unsigned short*)(wsc + b * 512u + h * 256u + rs * 2u);
                }
            }
            uint32_t bmj[4][4], sbj[4];
            #pragma unroll
            for (uint32_t j = 0; j < 4u; ++j) {
                const uint32_t col = c0w + j * 8u + (lane & 7u);
                const uint32_t c = h * 4u + (lane >> 3);
                pd_ldm_x4(bmj[j], yp + col * 128u + ((c ^ (col & 7u)) * 16u));
                sbj[j] = *(const unsigned short*)(ysc + b * 512u + h * 256u + (c0w + j * 8u + g) * 2u);
            }
            #pragma unroll
            for (uint32_t j = 0; j < 4u; ++j)
                #pragma unroll
                for (uint32_t s = 0; s < 4u; ++s)
                    pd_bs_mma_w8_kb<0>(acc[s * 4u + j], am[s][0][0], am[s][0][1],
                                       am[s][0][2], am[s][0][3], bmj[j][0], bmj[j][1],
                                       sa[s], sbj[j]);
            #pragma unroll
            for (uint32_t j = 0; j < 4u; ++j)
                #pragma unroll
                for (uint32_t s = 0; s < 4u; ++s)
                    pd_bs_mma_w8_kb<1>(acc[s * 4u + j], am[s][1][0], am[s][1][1],
                                       am[s][1][2], am[s][1][3], bmj[j][2], bmj[j][3],
                                       sa[s], sbj[j]);
        }
        asm volatile("bar.arrive %0, 384;" ::"r"(1u + b));
    }

    #pragma unroll
    for (uint32_t j = 0; j < 4u; ++j) {
        const uint32_t c0 = col_base + c0w + j * 8u + 2u * tq;
        #pragma unroll
        for (uint32_t s = 0; s < 4u; ++s) {
            const uint32_t r0 = row_base + i0 + s * 16u + g;
            const uint32_t r8 = r0 + 8u;
            if (r0 < out_dim) {
                if (c0 < batch) y[(size_t)c0 * out_dim + r0] = acc[s * 4u + j][0];
                if (c0 + 1u < batch) y[(size_t)(c0 + 1u) * out_dim + r0] = acc[s * 4u + j][1];
            }
            if (r8 < out_dim) {
                if (c0 < batch) y[(size_t)c0 * out_dim + r8] = acc[s * 4u + j][2];
                if (c0 + 1u < batch) y[(size_t)(c0 + 1u) * out_dim + r8] = acc[s * 4u + j][3];
            }
        }
    }
#else
    (void)wmap; (void)ymap; (void)scale; (void)xs; (void)y;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}

// ---- asymmetric W2/Y3 depth rung (f8 feed-depth hypothesis) ---------------
// Little's law on the measured feed: ~53KB must stay outstanding to saturate
// the ~26 GB/s per-SM L2 share at a ~2us round-trip; the 2-stage ring caps at
// 64KB and delivers 17. This rung deepens the Y ring to 3 (W 2 + Y 3 = 80KB
// data, 84.5KB total) with DECOUPLED producer streams - 2 warps produce W at
// pace sp+1, 2 produce Y running to sp+2 (a shared loop would re-serialize
// both behind the tighter W ring). Consumer = the arrive-early split shape
// (wait W -> half-0 A burst -> wait Y). Accumulation order unchanged ->
// bit-identical.
__global__ void __launch_bounds__(384, 1) pd_f8_gemm_w8_wy23_kt(
    const __grid_constant__ PdTmap wmap, const __grid_constant__ PdTmap ymap,
    const unsigned char* __restrict__ scale, const unsigned char* __restrict__ xs,
    float* __restrict__ y, uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
#if PD_BS_OK
    constexpr uint32_t PAIR16 = 16384u;
    extern __shared__ __align__(128) unsigned char pd_bsw8wy_sh[];
    unsigned char* wdat = pd_bsw8wy_sh;              // 2 stages x 16 KB
    unsigned char* ydat = pd_bsw8wy_sh + 32768u;     // 3 stages x 16 KB
    unsigned char* wsc = pd_bsw8wy_sh + 81920u;      // 2 x 2 halves x 256 B
    unsigned char* ysc = pd_bsw8wy_sh + 82944u;      // 3 x 2 halves x 256 B
    unsigned long long* mb = (unsigned long long*)(pd_bsw8wy_sh + 84480u); // [W0,W1,Y0,Y1,Y2]

    const uint32_t tid = threadIdx.x;
    const uint32_t n_kb = in_dim >> 5;
    const uint32_t nk = (in_dim + 63u) / 64u;
    const uint32_t nsp = (nk + 1u) >> 1;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t nct = batch_pad >> 7;
    const uint32_t tile = blockIdx.x;
    const uint32_t row_base = (tile / nct) * 128u;
    const uint32_t col_base = (tile % nct) * 128u;

    if (tid == 0u) {
        const uint32_t m0 = (uint32_t)__cvta_generic_to_shared(mb);
        #pragma unroll
        for (uint32_t i = 0; i < 5u; ++i)
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 64;" ::"r"(m0 + i * 8u));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    asm volatile("bar.sync 0, 384;");

    if (tid >= 320u) {
        // ---------- Y producers (warps 10-11): run up to 2 pairs ahead ----------
        const uint32_t ptid = tid - 320u;  // 0..63
        unsigned char syr[2][2][2];        // [row-pass][h][kb]
        for (uint32_t sp = 0; sp < nsp; ++sp) {
            const uint32_t by = sp % 3u;
            #pragma unroll
            for (uint32_t rp = 0; rp < 2u; ++rp) {
                const uint32_t row = ptid + rp * 64u;
                #pragma unroll
                for (uint32_t h = 0; h < 2u; ++h)
                    #pragma unroll
                    for (uint32_t kb = 0; kb < 2u; ++kb) {
                        const uint32_t kt = sp * 2u + h;
                        const bool yok = (col_base + row) < batch && kt * 2u + kb < n_kb;
                        syr[rp][h][kb] = yok ? xs[(size_t)(col_base + row) * n_kb + kt * 2u + kb] : 0u;
                    }
            }
            if (sp >= 3u) asm volatile("bar.sync %0, 320;" ::"r"(3u + by));
            #pragma unroll
            for (uint32_t rp = 0; rp < 2u; ++rp) {
                const uint32_t row = ptid + rp * 64u;
                #pragma unroll
                for (uint32_t h = 0; h < 2u; ++h)
                    #pragma unroll
                    for (uint32_t kb = 0; kb < 2u; ++kb)
                        ysc[by * 512u + h * 256u + row * 2u + kb] = syr[rp][h][kb];
            }
            const uint32_t my = (uint32_t)__cvta_generic_to_shared(mb) + (2u + by) * 8u;
            if (ptid == 0u) {
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 16384;" ::"r"(my));
                const uint32_t yd = (uint32_t)__cvta_generic_to_shared(ydat + by * PAIR16);
                const int ck = (int)(sp * 128u);
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3}], [%4];" ::"r"(yd),
                    "l"(&ymap), "r"(ck), "r"((int)col_base), "r"(my) : "memory");
            } else {
                asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(my));
            }
        }
        return;
    }
    if (tid >= 256u) {
        // ---------- W producers (warps 8-9) ----------
        const uint32_t ptid = tid - 256u;  // 0..63
        unsigned char swr[2][2][2];
        for (uint32_t sp = 0; sp < nsp; ++sp) {
            const uint32_t bw = sp % 2u;
            #pragma unroll
            for (uint32_t rp = 0; rp < 2u; ++rp) {
                const uint32_t row = ptid + rp * 64u;
                #pragma unroll
                for (uint32_t h = 0; h < 2u; ++h)
                    #pragma unroll
                    for (uint32_t kb = 0; kb < 2u; ++kb) {
                        const uint32_t kt = sp * 2u + h;
                        const bool wok = (row_base + row) < out_dim && kt * 2u + kb < n_kb;
                        swr[rp][h][kb] = wok ? scale[(size_t)(row_base + row) * n_kb + kt * 2u + kb] : 0u;
                    }
            }
            if (sp >= 2u) asm volatile("bar.sync %0, 320;" ::"r"(1u + bw));
            #pragma unroll
            for (uint32_t rp = 0; rp < 2u; ++rp) {
                const uint32_t row = ptid + rp * 64u;
                #pragma unroll
                for (uint32_t h = 0; h < 2u; ++h)
                    #pragma unroll
                    for (uint32_t kb = 0; kb < 2u; ++kb)
                        wsc[bw * 512u + h * 256u + row * 2u + kb] = swr[rp][h][kb];
            }
            const uint32_t mw = (uint32_t)__cvta_generic_to_shared(mb) + bw * 8u;
            if (ptid == 0u) {
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 16384;" ::"r"(mw));
                const uint32_t wd = (uint32_t)__cvta_generic_to_shared(wdat + bw * PAIR16);
                const int ck = (int)(sp * 128u);
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3}], [%4];" ::"r"(wd),
                    "l"(&wmap), "r"(ck), "r"((int)row_base), "r"(mw) : "memory");
            } else {
                asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(mw));
            }
        }
        return;
    }

    // ---------------- consumer warps 0-7 (split shape) ----------------
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, tq = lane & 3u;
    const uint32_t i0 = (warp & 1u) * 64u;
    const uint32_t c0w = (warp >> 1) * 32u;

    float acc[16][4] = {};
    uint32_t phw[2] = {}, phy[3] = {};

    for (uint32_t sp = 0; sp < nsp; ++sp) {
        const uint32_t bw = sp % 2u;
        const uint32_t by = sp % 3u;
        const uint32_t mw = (uint32_t)__cvta_generic_to_shared(mb) + bw * 8u;
        const uint32_t my = (uint32_t)__cvta_generic_to_shared(mb) + (2u + by) * 8u;
        asm volatile(
            "{\n\t.reg .pred P;\n"
            "PD_WYW_%=:\n\t"
            "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
            "@!P bra PD_WYW_%=;\n\t}" ::"r"(mw), "r"(phw[bw]));
        phw[bw] ^= 1u;

        const unsigned char* wp = wdat + bw * PAIR16;
        const unsigned char* yp = ydat + by * PAIR16;
        // half 0's A-side burst under Y's in-flight tail
        uint32_t am0[4][2][4];
        uint32_t sa0[4];
        #pragma unroll
        for (uint32_t s = 0; s < 4u; ++s) {
            const uint32_t r0 = i0 + s * 16u + g;
            const uint32_t rr = i0 + s * 16u + ((lane >> 3) & 1u) * 8u + (lane & 7u);
            #pragma unroll
            for (uint32_t kb = 0; kb < 2u; ++kb) {
                const uint32_t c = kb * 2u + (lane >> 4);
                pd_ldm_x4(am0[s][kb], wp + rr * 128u + ((c ^ (rr & 7u)) * 16u));
            }
            const uint32_t rs = (tq & 1u) ? r0 + 8u : r0;
            sa0[s] = *(const unsigned short*)(wsc + bw * 512u + rs * 2u);
        }
        asm volatile(
            "{\n\t.reg .pred P;\n"
            "PD_WYY_%=:\n\t"
            "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
            "@!P bra PD_WYY_%=;\n\t}" ::"r"(my), "r"(phy[by]));
        phy[by] ^= 1u;

        #pragma unroll
        for (uint32_t h = 0; h < 2u; ++h) {
            const uint32_t kt = sp * 2u + h;
            if (kt >= nk) break;

            uint32_t am[4][2][4];
            uint32_t sa[4];
            if (h == 0u) {
                #pragma unroll
                for (uint32_t s = 0; s < 4u; ++s) {
                    #pragma unroll
                    for (uint32_t kb = 0; kb < 2u; ++kb)
                        #pragma unroll
                        for (uint32_t q = 0; q < 4u; ++q) am[s][kb][q] = am0[s][kb][q];
                    sa[s] = sa0[s];
                }
            } else {
                #pragma unroll
                for (uint32_t s = 0; s < 4u; ++s) {
                    const uint32_t r0 = i0 + s * 16u + g;
                    const uint32_t rr = i0 + s * 16u + ((lane >> 3) & 1u) * 8u + (lane & 7u);
                    #pragma unroll
                    for (uint32_t kb = 0; kb < 2u; ++kb) {
                        const uint32_t c = h * 4u + kb * 2u + (lane >> 4);
                        pd_ldm_x4(am[s][kb], wp + rr * 128u + ((c ^ (rr & 7u)) * 16u));
                    }
                    const uint32_t rs = (tq & 1u) ? r0 + 8u : r0;
                    sa[s] = *(const unsigned short*)(wsc + bw * 512u + h * 256u + rs * 2u);
                }
            }
            uint32_t bmj[4][4], sbj[4];
            #pragma unroll
            for (uint32_t j = 0; j < 4u; ++j) {
                const uint32_t col = c0w + j * 8u + (lane & 7u);
                const uint32_t c = h * 4u + (lane >> 3);
                pd_ldm_x4(bmj[j], yp + col * 128u + ((c ^ (col & 7u)) * 16u));
                sbj[j] = *(const unsigned short*)(ysc + by * 512u + h * 256u + (c0w + j * 8u + g) * 2u);
            }
            #pragma unroll
            for (uint32_t j = 0; j < 4u; ++j)
                #pragma unroll
                for (uint32_t s = 0; s < 4u; ++s)
                    pd_bs_mma_w8_kb<0>(acc[s * 4u + j], am[s][0][0], am[s][0][1],
                                       am[s][0][2], am[s][0][3], bmj[j][0], bmj[j][1],
                                       sa[s], sbj[j]);
            #pragma unroll
            for (uint32_t j = 0; j < 4u; ++j)
                #pragma unroll
                for (uint32_t s = 0; s < 4u; ++s)
                    pd_bs_mma_w8_kb<1>(acc[s * 4u + j], am[s][1][0], am[s][1][1],
                                       am[s][1][2], am[s][1][3], bmj[j][2], bmj[j][3],
                                       sa[s], sbj[j]);
        }
        asm volatile("bar.arrive %0, 320;" ::"r"(1u + bw));
        asm volatile("bar.arrive %0, 320;" ::"r"(3u + by));
    }

    #pragma unroll
    for (uint32_t j = 0; j < 4u; ++j) {
        const uint32_t c0 = col_base + c0w + j * 8u + 2u * tq;
        #pragma unroll
        for (uint32_t s = 0; s < 4u; ++s) {
            const uint32_t r0 = row_base + i0 + s * 16u + g;
            const uint32_t r8 = r0 + 8u;
            if (r0 < out_dim) {
                if (c0 < batch) y[(size_t)c0 * out_dim + r0] = acc[s * 4u + j][0];
                if (c0 + 1u < batch) y[(size_t)(c0 + 1u) * out_dim + r0] = acc[s * 4u + j][1];
            }
            if (r8 < out_dim) {
                if (c0 < batch) y[(size_t)c0 * out_dim + r8] = acc[s * 4u + j][2];
                if (c0 + 1u < batch) y[(size_t)(c0 + 1u) * out_dim + r8] = acc[s * 4u + j][3];
            }
        }
    }
#else
    (void)wmap; (void)ymap; (void)scale; (void)xs; (void)y;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}

// ---- residency kernel (the f8 model-derived build) ------------------------
// The clock64 model: per-warp mma issue is a universal ~28-32 cycles/mma and
// SM throughput = (warps concurrently in mma phase) x that rate; the baseline
// spends 24% of each pair in a serialized MIO load burst that dilutes mma
// residency to 54% (= 417 TF exactly). This kernel hides the load bursts
// inside the mma stream via cross-half fragment double-buffering:
//   - h1's fragments load between h0's kb0 and kb1 mma blocks;
//   - the next pair's h0 fragments load during h1's mmas after a NON-BLOCKING
//     poll of its full barrier (per-warp; a miss just falls back to loading
//     at the next pair's top).
// Register math: this needs acc 64 (16 independent chains per warp - the
// tma16 trap) + ~96 fragment regs, impossible at 384 threads (168 cap) but
// fine at 320 (2 producer warps, 204-reg cap). Accumulation order unchanged
// -> bit-identical.
__global__ void __launch_bounds__(320, 1) pd_f8_gemm_w8_res_kt(
    const __grid_constant__ PdTmap wmap, const __grid_constant__ PdTmap ymap,
    const unsigned char* __restrict__ scale, const unsigned char* __restrict__ xs,
    float* __restrict__ y, uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
#if PD_BS_OK
    constexpr uint32_t PAIR16 = 16384u;
    extern __shared__ __align__(128) unsigned char pd_bsw8res_sh[];
    unsigned char* wdat = pd_bsw8res_sh;
    unsigned char* ydat = pd_bsw8res_sh + 32768u;
    unsigned char* wsc = pd_bsw8res_sh + 65536u;
    unsigned char* ysc = pd_bsw8res_sh + 66560u;
    unsigned long long* mb = (unsigned long long*)(pd_bsw8res_sh + 67584u);

    const uint32_t tid = threadIdx.x;
    const uint32_t n_kb = in_dim >> 5;
    const uint32_t nk = (in_dim + 63u) / 64u;
    const uint32_t nsp = (nk + 1u) >> 1;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t nct = batch_pad >> 7;
    const uint32_t tile = blockIdx.x;
    const uint32_t row_base = (tile / nct) * 128u;
    const uint32_t col_base = (tile % nct) * 128u;

    if (tid == 0u) {
        const uint32_t m0 = (uint32_t)__cvta_generic_to_shared(mb);
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 64;" ::"r"(m0));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 64;" ::"r"(m0 + 8u));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    asm volatile("bar.sync 0, 320;");

    if (tid >= 256u) {
        // ------------- producers (warps 8-9): 2 scale rows/thread -------------
        const uint32_t ptid = tid - 256u;
        unsigned char swr[2][2][2], syr[2][2][2];
        for (uint32_t sp = 0; sp < nsp; ++sp) {
            const uint32_t b = sp % 2u;
            #pragma unroll
            for (uint32_t rp = 0; rp < 2u; ++rp) {
                const uint32_t row = ptid + rp * 64u;
                #pragma unroll
                for (uint32_t h = 0; h < 2u; ++h)
                    #pragma unroll
                    for (uint32_t kb = 0; kb < 2u; ++kb) {
                        const uint32_t kt = sp * 2u + h;
                        const bool wok = (row_base + row) < out_dim && kt * 2u + kb < n_kb;
                        swr[rp][h][kb] = wok ? scale[(size_t)(row_base + row) * n_kb + kt * 2u + kb] : 0u;
                        const bool yok = (col_base + row) < batch && kt * 2u + kb < n_kb;
                        syr[rp][h][kb] = yok ? xs[(size_t)(col_base + row) * n_kb + kt * 2u + kb] : 0u;
                    }
            }
            if (sp >= 2u) asm volatile("bar.sync %0, 320;" ::"r"(1u + b));
            #pragma unroll
            for (uint32_t rp = 0; rp < 2u; ++rp) {
                const uint32_t row = ptid + rp * 64u;
                #pragma unroll
                for (uint32_t h = 0; h < 2u; ++h)
                    #pragma unroll
                    for (uint32_t kb = 0; kb < 2u; ++kb) {
                        wsc[b * 512u + h * 256u + row * 2u + kb] = swr[rp][h][kb];
                        ysc[b * 512u + h * 256u + row * 2u + kb] = syr[rp][h][kb];
                    }
            }
            const uint32_t m = (uint32_t)__cvta_generic_to_shared(mb) + b * 8u;
            if (ptid == 0u) {
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 32768;" ::"r"(m));
                const uint32_t wd = (uint32_t)__cvta_generic_to_shared(wdat + b * PAIR16);
                const uint32_t yd = (uint32_t)__cvta_generic_to_shared(ydat + b * PAIR16);
                const int ck = (int)(sp * 128u);
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3}], [%4];" ::"r"(wd),
                    "l"(&wmap), "r"(ck), "r"((int)row_base), "r"(m) : "memory");
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3}], [%4];" ::"r"(yd),
                    "l"(&ymap), "r"(ck), "r"((int)col_base), "r"(m) : "memory");
            } else {
                asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(m));
            }
        }
        return;
    }

    // ---------------- consumer warps 0-7 ----------------
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, tq = lane & 3u;
    const uint32_t i0 = (warp & 1u) * 64u;
    const uint32_t c0w = (warp >> 1) * 32u;

    float acc[16][4] = {};
    uint32_t ph0 = 0u, ph1 = 0u;

    // fragment sets P (current half) and Q (prefetched half)
    uint32_t amP[4][2][4], bmP[4][4], saP[4], sbP[4];
    uint32_t amQ[4][2][4], bmQ[4][4], saQ[4], sbQ[4];
    bool haveP = false;  // P holds h0 of the CURRENT pair (prefetched cross-pair)

    // fragment load helper macro over buffer bb / half hh into set S
    #define PD_RES_LOAD(S_am, S_bm, S_sa, S_sb, bb, hh)                              \
    {                                                                                \
        const unsigned char* wp_ = wdat + (bb) * PAIR16;                             \
        const unsigned char* yp_ = ydat + (bb) * PAIR16;                             \
        _Pragma("unroll")                                                            \
        for (uint32_t s = 0; s < 4u; ++s) {                                          \
            const uint32_t r0 = i0 + s * 16u + g;                                    \
            const uint32_t rr = i0 + s * 16u + ((lane >> 3) & 1u) * 8u + (lane & 7u);\
            _Pragma("unroll")                                                        \
            for (uint32_t kb = 0; kb < 2u; ++kb) {                                   \
                const uint32_t c = (hh) * 4u + kb * 2u + (lane >> 4);                \
                pd_ldm_x4(S_am[s][kb], wp_ + rr * 128u + ((c ^ (rr & 7u)) * 16u));   \
            }                                                                        \
            const uint32_t rs = (tq & 1u) ? r0 + 8u : r0;                            \
            S_sa[s] = *(const unsigned short*)(wsc + (bb) * 512u + (hh) * 256u + rs * 2u); \
        }                                                                            \
        _Pragma("unroll")                                                            \
        for (uint32_t j = 0; j < 4u; ++j) {                                          \
            const uint32_t col = c0w + j * 8u + (lane & 7u);                         \
            const uint32_t c = (hh) * 4u + (lane >> 3);                              \
            pd_ldm_x4(S_bm[j], yp_ + col * 128u + ((c ^ (col & 7u)) * 16u));         \
            S_sb[j] = *(const unsigned short*)(ysc + (bb) * 512u + (hh) * 256u + (c0w + j * 8u + g) * 2u); \
        }                                                                            \
    }
    #define PD_RES_MMA(S_am, S_bm, S_sa, S_sb, KB)                                   \
        _Pragma("unroll")                                                            \
        for (uint32_t j = 0; j < 4u; ++j)                                            \
            _Pragma("unroll")                                                        \
            for (uint32_t s = 0; s < 4u; ++s)                                        \
                pd_bs_mma_w8_kb<KB>(acc[s * 4u + j], S_am[s][KB][0], S_am[s][KB][1], \
                                    S_am[s][KB][2], S_am[s][KB][3], S_bm[j][KB * 2u],\
                                    S_bm[j][KB * 2u + 1u], S_sa[s], S_sb[j]);

    for (uint32_t sp = 0; sp < nsp; ++sp) {
        const uint32_t b = sp % 2u;
        const uint32_t m = (uint32_t)__cvta_generic_to_shared(mb) + b * 8u;
        if (!haveP) {
            const uint32_t ph = (b == 0u) ? ph0 : ph1;
            asm volatile(
                "{\n\t.reg .pred P;\n"
                "PD_RES_WAIT_%=:\n\t"
                "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
                "@!P bra PD_RES_WAIT_%=;\n\t}" ::"r"(m), "r"(ph));
            if (b == 0u) ph0 ^= 1u; else ph1 ^= 1u;
            PD_RES_LOAD(amP, bmP, saP, sbP, b, 0u)
        }
        haveP = false;
        const uint32_t kt1 = sp * 2u + 1u;

        // ---- h0: kb0 mmas, then h1 loads (hidden), then kb1 mmas ----
        PD_RES_MMA(amP, bmP, saP, sbP, 0)
        if (kt1 < nk) PD_RES_LOAD(amQ, bmQ, saQ, sbQ, b, 1u)
        PD_RES_MMA(amP, bmP, saP, sbP, 1)

        if (kt1 < nk) {
            // ---- h1: kb0, then (poll+prefetch next pair's h0), then kb1 ----
            PD_RES_MMA(amQ, bmQ, saQ, sbQ, 0)
            if (sp + 1u < nsp) {
                const uint32_t bn = (sp + 1u) % 2u;
                const uint32_t mn = (uint32_t)__cvta_generic_to_shared(mb) + bn * 8u;
                const uint32_t phn = (bn == 0u) ? ph0 : ph1;
                uint32_t ready;
                asm volatile(
                    "{\n\t.reg .pred P;\n\t"
                    "mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2;\n\t"
                    "selp.b32 %0, 1, 0, P;\n\t}"
                    : "=r"(ready) : "r"(mn), "r"(phn));
                // broadcast lane 0's verdict: a warp-split poll would feed
                // ldmatrix.x4 a partial warp
                ready = __shfl_sync(0xffffffffu, ready, 0);
                if (ready) {
                    if (bn == 0u) ph0 ^= 1u; else ph1 ^= 1u;
                    PD_RES_LOAD(amP, bmP, saP, sbP, bn, 0u)
                    haveP = true;
                }
            }
            PD_RES_MMA(amQ, bmQ, saQ, sbQ, 1)
        }
        asm volatile("bar.arrive %0, 320;" ::"r"(1u + b));
    }
    #undef PD_RES_LOAD
    #undef PD_RES_MMA

    #pragma unroll
    for (uint32_t j = 0; j < 4u; ++j) {
        const uint32_t c0 = col_base + c0w + j * 8u + 2u * tq;
        #pragma unroll
        for (uint32_t s = 0; s < 4u; ++s) {
            const uint32_t r0 = row_base + i0 + s * 16u + g;
            const uint32_t r8 = r0 + 8u;
            if (r0 < out_dim) {
                if (c0 < batch) y[(size_t)c0 * out_dim + r0] = acc[s * 4u + j][0];
                if (c0 + 1u < batch) y[(size_t)(c0 + 1u) * out_dim + r0] = acc[s * 4u + j][1];
            }
            if (r8 < out_dim) {
                if (c0 < batch) y[(size_t)c0 * out_dim + r8] = acc[s * 4u + j][2];
                if (c0 + 1u < batch) y[(size_t)(c0 + 1u) * out_dim + r8] = acc[s * 4u + j][3];
            }
        }
    }
#else
    (void)wmap; (void)ymap; (void)scale; (void)xs; (void)y;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}

