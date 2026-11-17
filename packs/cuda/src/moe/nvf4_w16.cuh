// moe/nvf4_w16.cuh - the checkpoint's W4A16 expert class on tensor cores, one
// kernel pair for every row count: one-row decode, a decode tick's rows, a
// spec verify round's.
// Textually-included segment of the single pack translation unit.
// Not standalone-compilable: include order is defined by ../pack.cu.
//
// Why: the modelopt checkpoint declares its experts W4A16_NVFP4 - 16-bit
// activations - and NVIDIA's GB10 recipe serves them that way (Marlin). The
// multi-row lanes here ran W4A4 (activations quantized to NVFP4 for the
// block-scaled MMA) while one-row decode ran W4A16 at f32 activations on
// FFMA, so a verify row scored its token in a coarser class than the decode
// tick it stands for (logits up to 1.5 apart, near-ties flipped - the DFlash
// spec loop parted from greedy), and every tick past one live row served
// another class than the first. f32-activation FFMA twins made the rows exact
// but are issue-bound from ~4 rows on GB10 (per weight element each row costs
// a float4 x load and ~6 FP ops). This pair runs bf16 activations through
// mma.m16n8k16 - the weights' dequant is paid once per weight whatever the
// row count, and a row costs only its share of an n8 column.
//
// BATCH INVARIANT by construction: an output element is a weight row (MMA M)
// against a token column (MMA N); MMA columns never mix, and every warp walks
// the full K in one fixed order (64-wide tiles ascending, four k16 steps
// each) - nothing about the walk depends on how many tokens share the block.
// A token's activations and partials are the same bits alone (r = 1) or in a
// 32-row verify block, so a verify row IS the decode step it stands for.
//
// No repack: the tiled plane (moe/nvf4_st.cuh - per 64x64 tile, two 1 KB
// pieces [row][16 B of 32 k] and 256 B of [row][4 block scales]) is read in
// place. Inside each tile, MMA k-slots map to actual k by a fixed permutation
// applied to weights and activations alike: thread quad tq's slots in step s
// are actual k 16*tq + 4*s + {0..3}. So a thread's weights for a tile are the
// 16 k of ONE block-scale group: one aligned 8-byte load per row, one scale.
// e2m1 x e4m3 dequantizes EXACTLY into bf16 (at most 6 significant bits,
// magnitudes 2^-10..2688), the scale multiplied in before the MMA; the
// expert's f32 scale2 rides the epilogue.

#define PD_W16_BM 32u   // tokens a routed block holds (moe_align's default tile)

#if PD_NV4_OK
// One weight byte (two e2m1 nibbles, k even in the low nibble) as a bf16x2
// {k even, k odd} register, times the block scale (bf16x2 {s, s}).
__device__ __forceinline__ uint32_t pd_w16_dq2(uint32_t b, uint32_t s2) {
    // |e2m1| -> bf16 bytes: 0, .5, 1, 1.5, 2, 3, 4, 6 -> hi 00 3F 3F 3F 40 40 40 40,
    // lo 00 00 80 C0 00 40 80 C0
    const uint32_t m0 = b & 7u, m1 = (b >> 4) & 7u;
    const uint32_t sel = m0 | (m1 << 4);
    const uint32_t hi = __byte_perm(0x3F3F3F00u, 0x40404040u, sel);
    const uint32_t lo = __byte_perm(0xC0800000u, 0xC0804000u, sel);
    uint32_t v = __byte_perm(lo, hi, 0x5140);             // {lo0, hi0, lo1, hi1}
    v |= ((b & 0x08u) << 12) | ((b & 0x80u) << 24);      // signs -> bits 15, 31
    uint32_t r;
#if __CUDA_ARCH__ >= 900
    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(v), "r"(s2));
#else
    // sm_89 (PD_NV4_OK's floor) has no bf16x2 multiply. An FMA with a -0
    // addend rounds the same product the same way - rn(a*b + -0) is rn(a*b),
    // signed zeros included - which is how cuda_bf16's __hmul2 lowers there.
    asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(r) : "r"(v), "r"(s2), "r"(0x80008000u));
#endif
    return r;
}

// the e4m3 block scale as bf16x2 {s, s} (exact: 3 mantissa bits)
__device__ __forceinline__ uint32_t pd_w16_scale2(uint32_t sb) {
    const float s = (float)reinterpret_cast<const __nv_fp8_e4m3&>(sb);
    const uint32_t h = __float_as_uint(s) >> 16;          // exact - no rounding
    return h | (h << 16);
}

__device__ __forceinline__ void pd_w16_mma(float d[4], const uint32_t a[4],
                                           const uint32_t b[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// One 64-wide K tile of a warp's walk: dequantize its two rows' 8-byte words
// (four k16 steps, bytes 2s / 2s+1 of each row) once, then every live n8
// column's four MMAs. `b` columns: token n*8 + g's bf16 activations, the 16 k
// of this thread's quad.
template <uint32_t NTMAX>
__device__ __forceinline__ void pd_w16_tile(float (&acc)[NTMAX][4], uint2 wg, uint2 wh,
                                            uint32_t qg, uint32_t qh, uint32_t k0,
                                            uint32_t g, uint32_t nt,
                                            const void* const* xr) {
    const uint32_t sg = pd_w16_scale2(qg), sh = pd_w16_scale2(qh);
    uint32_t a[4][4];
#pragma unroll
    for (uint32_t s = 0; s < 4u; ++s) {
        const uint32_t bg = s < 2u ? wg.x : wg.y, bh = s < 2u ? wh.x : wh.y;
        const uint32_t sh8 = (s & 1u) * 16u;
        a[s][0] = pd_w16_dq2((bg >> sh8) & 0xFFu, sg);
        a[s][1] = pd_w16_dq2((bh >> sh8) & 0xFFu, sh);
        a[s][2] = pd_w16_dq2((bg >> (sh8 + 8u)) & 0xFFu, sg);
        a[s][3] = pd_w16_dq2((bh >> (sh8 + 8u)) & 0xFFu, sh);
    }
#pragma unroll
    for (uint32_t n = 0; n < NTMAX; ++n) {
        if (n >= nt) break;
        const uint4* x = reinterpret_cast<const uint4*>(
            (const __nv_bfloat16*)xr[n * 8u + g] + k0);
        const uint4 u0 = x[0], u1 = x[1];
        const uint32_t b[8] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
#pragma unroll
        for (uint32_t s = 0; s < 4u; ++s) {
            const uint32_t bb[2] = {b[2u * s], b[2u * s + 1u]};
            pd_w16_mma(acc[n], a[s], bb);
        }
    }
}

// A warp's 16 weight rows x the block's tokens, the full K in ascending tiles.
// `tiles`/`scales` = the tiled plane at this expert's 64-row tile; warp rows
// r0..r0+15 of it. B columns `xr[j]`: bf16 token rows. acc[nt][4] = C.
// The weight words and block scales run two tiles ahead in two fixed
// register sets (tile ks + 2's loads issue before tile ks's math), so a
// warp's DRAM reads stay in flight across the dequant and MMAs; one tile deep
// streamed ~180 GB/s at one row, a ring indexed ks % 2 spilled.
template <uint32_t NTMAX>
__device__ __forceinline__ void pd_w16_walk(float (&acc)[NTMAX][4],
                                            const uint8_t* __restrict__ tiles,
                                            const uint8_t* __restrict__ scales,
                                            uint32_t nks, uint32_t r0, uint32_t g,
                                            uint32_t tq, uint32_t nt,
                                            const void* const* xr) {
    const uint32_t row_g = r0 + g, row_h = row_g + 8u;
    const uint32_t woff_g = (tq >> 1) * 1024u + row_g * 16u + 8u * (tq & 1u);
    const uint32_t woff_h = (tq >> 1) * 1024u + row_h * 16u + 8u * (tq & 1u);
    const uint32_t soff_g = row_g * 4u + tq, soff_h = row_h * 4u + tq;
    auto wword = [&](uint32_t ks, uint32_t off) {
        return ks < nks ? *reinterpret_cast<const uint2*>(tiles + (size_t)ks * 2048u + off)
                        : make_uint2(0u, 0u);
    };
    auto sbyte = [&](uint32_t ks, uint32_t off) {
        return ks < nks ? (uint32_t)scales[(size_t)ks * 256u + off] : 0u;
    };
    uint2 g0 = wword(0, woff_g), h0 = wword(0, woff_h);
    uint32_t sg0 = sbyte(0, soff_g), sh0 = sbyte(0, soff_h);
    uint2 g1 = wword(1, woff_g), h1 = wword(1, woff_h);
    uint32_t sg1 = sbyte(1, soff_g), sh1 = sbyte(1, soff_h);
    uint32_t ks = 0;
    for (; ks + 1u < nks; ks += 2u) {
        const uint2 cg0 = g0, ch0 = h0;
        const uint32_t csg0 = sg0, csh0 = sh0;
        g0 = wword(ks + 2u, woff_g); h0 = wword(ks + 2u, woff_h);
        sg0 = sbyte(ks + 2u, soff_g); sh0 = sbyte(ks + 2u, soff_h);
        pd_w16_tile(acc, cg0, ch0, csg0, csh0, ks * 64u + 16u * tq, g, nt, xr);
        const uint2 cg1 = g1, ch1 = h1;
        const uint32_t csg1 = sg1, csh1 = sh1;
        g1 = wword(ks + 3u, woff_g); h1 = wword(ks + 3u, woff_h);
        sg1 = sbyte(ks + 3u, soff_g); sh1 = sbyte(ks + 3u, soff_h);
        pd_w16_tile(acc, cg1, ch1, csg1, csh1, (ks + 1u) * 64u + 16u * tq, g, nt, xr);
    }
    if (ks < nks)
        pd_w16_tile(acc, g0, h0, sg0, sh0, ks * 64u + 16u * tq, g, nt, xr);
}
#endif

// up + relu^2 -> act, bf16 [rows][k * ff_r + ff_s] (the down's B operand).
// x: the rows' bf16 activations (convert_f32_bf16 of the normed rows - the
// same round-to-nearest cast whatever the row count).
// Tasks: routed = (sorted 32-block, 64-row tile of ff_r), then shared =
// (group of 32 token rows, 64-row tile of ff_s). 4 warps x 16 rows.
__global__ void __launch_bounds__(128) pd_nv4w16_up_kernel(
    const uint8_t* __restrict__ rdata, const uint8_t* __restrict__ rscale,
    const float* __restrict__ rscale2, const uint8_t* __restrict__ sdata,
    const uint8_t* __restrict__ sscale, const float* __restrict__ sscale2,
    const uint32_t* __restrict__ sorted_row, const uint32_t* __restrict__ sorted_slot,
    const uint32_t* __restrict__ block_expert, const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ act, uint32_t in_dim, uint32_t ff_r, uint32_t ff_s,
    uint32_t k, uint32_t n_blocks, uint32_t rows) {
#if PD_NV4_OK
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t g = lane >> 2, tq = lane & 3u;
    const uint32_t rt_r = ff_r >> 6, rt_s = ff_s >> 6;
    const uint32_t n_rt = n_blocks * rt_r;
    const uint32_t task = blockIdx.x;
    const uint32_t aw = k * ff_r + ff_s;
    const uint32_t nks = in_dim >> 6;
    __shared__ const void* xr[PD_W16_BM];
    __shared__ __nv_bfloat16* ar[PD_W16_BM];
    const uint8_t* tiles;
    const uint8_t* scales;
    float s2;
    uint32_t rt, nv = 0;
    if (task < n_rt) {
        const uint32_t blk = task / rt_r;
        rt = task - blk * rt_r;
        const uint32_t e = block_expert[blk];
        if (e == PD_MOE_PAD) return;
        tiles = rdata + ((size_t)e * rt_r + rt) * nks * 2048u;
        scales = rscale + ((size_t)e * rt_r + rt) * nks * 256u;
        s2 = rscale2[e];
        // the align packs a block's live rows first
        while (nv < PD_W16_BM && sorted_row[(size_t)blk * PD_W16_BM + nv] != PD_MOE_PAD) ++nv;
        if (threadIdx.x < PD_W16_BM) {
            const uint32_t j = threadIdx.x;
            const uint32_t t = j < nv ? sorted_row[(size_t)blk * PD_W16_BM + j] : 0u;
            const uint32_t slt = j < nv ? sorted_slot[(size_t)blk * PD_W16_BM + j] : 0u;
            // columns past the live rows read a live row and are never stored
            const uint32_t tt = j < nv ? t : sorted_row[(size_t)blk * PD_W16_BM];
            xr[j] = x + (size_t)tt * in_dim;
            ar[j] = act + (size_t)t * aw + (size_t)slt * ff_r;
        }
    } else {
        const uint32_t sg = (task - n_rt) / rt_s;
        rt = task - n_rt - sg * rt_s;
        if (sg * PD_W16_BM >= rows) return;
        tiles = sdata + (size_t)rt * nks * 2048u;
        scales = sscale + (size_t)rt * nks * 256u;
        s2 = sscale2[0];
        nv = min(PD_W16_BM, rows - sg * PD_W16_BM);
        if (threadIdx.x < PD_W16_BM) {
            const uint32_t j = threadIdx.x;
            const uint32_t t = sg * PD_W16_BM + (j < nv ? j : 0u);
            xr[j] = x + (size_t)t * in_dim;
            ar[j] = act + (size_t)t * aw + (size_t)k * ff_r;
        }
    }
    __syncthreads();
    const uint32_t nt = (nv + 7u) >> 3;
    float acc[4][4] = {};
    pd_w16_walk<4u>(acc, tiles, scales, nks, warp * 16u, g, tq, nt, xr);
    // C[g | g+8][2tq | 2tq+1] -> relu^2(c * s2), bf16, into each live token's row
    const uint32_t o = rt * 64u + warp * 16u + g;
#pragma unroll
    for (uint32_t n = 0; n < 4u; ++n) {
        if (n >= nt) break;
#pragma unroll
        for (uint32_t i = 0; i < 4u; ++i) {
            const uint32_t j = n * 8u + 2u * tq + (i & 1u);
            if (j >= nv) continue;
            const float v = fmaxf(acc[n][i] * s2, 0.0f);
            ar[j][o + (i >> 1) * 8u] = __float2bfloat16_rn(v * v);
        }
    }
#else
    (void)rdata; (void)rscale; (void)rscale2; (void)sdata; (void)sscale;
    (void)sscale2; (void)sorted_row; (void)sorted_slot; (void)block_expert;
    (void)x; (void)act; (void)in_dim; (void)ff_r; (void)ff_s; (void)k;
    (void)n_blocks; (void)rows;
#endif
}

// down + the topk-weight fold -> part f32 [rows][k + 1][embd] (slot k = the
// shared expert), for moe_slot_combine's fixed slot order.
__global__ void __launch_bounds__(128) pd_nv4w16_dn_kernel(
    const uint8_t* __restrict__ rdata, const uint8_t* __restrict__ rscale,
    const float* __restrict__ rscale2, const uint8_t* __restrict__ sdata,
    const uint8_t* __restrict__ sscale, const float* __restrict__ sscale2,
    const uint32_t* __restrict__ sorted_row, const uint32_t* __restrict__ sorted_slot,
    const uint32_t* __restrict__ block_expert, const float* __restrict__ topk_w,
    const __nv_bfloat16* __restrict__ act, float* __restrict__ part, uint32_t ff_r,
    uint32_t ff_s, uint32_t embd, uint32_t k, uint32_t n_blocks, uint32_t rows) {
#if PD_NV4_OK
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t g = lane >> 2, tq = lane & 3u;
    const uint32_t rt_e = embd >> 6;
    const uint32_t n_rt = n_blocks * rt_e;
    const uint32_t task = blockIdx.x;
    const uint32_t aw = k * ff_r + ff_s;
    __shared__ const void* xr[PD_W16_BM];
    __shared__ float* pr[PD_W16_BM];
    __shared__ float wr[PD_W16_BM];
    const uint8_t* tiles;
    const uint8_t* scales;
    uint32_t rt, nks, nv = 0;
    if (task < n_rt) {
        const uint32_t blk = task / rt_e;
        rt = task - blk * rt_e;
        const uint32_t e = block_expert[blk];
        if (e == PD_MOE_PAD) return;
        nks = ff_r >> 6;
        tiles = rdata + ((size_t)e * rt_e + rt) * nks * 2048u;
        scales = rscale + ((size_t)e * rt_e + rt) * nks * 256u;
        while (nv < PD_W16_BM && sorted_row[(size_t)blk * PD_W16_BM + nv] != PD_MOE_PAD) ++nv;
        if (threadIdx.x < PD_W16_BM) {
            const uint32_t j = threadIdx.x;
            const uint32_t jj = j < nv ? j : 0u;
            const uint32_t t = sorted_row[(size_t)blk * PD_W16_BM + jj];
            const uint32_t slt = sorted_slot[(size_t)blk * PD_W16_BM + jj];
            wr[j] = topk_w[(size_t)t * k + slt] * rscale2[e];
            xr[j] = act + (size_t)t * aw + (size_t)slt * ff_r;
            pr[j] = part + ((size_t)t * (k + 1u) + slt) * embd;
        }
    } else {
        const uint32_t sg = (task - n_rt) / rt_e;
        rt = task - n_rt - sg * rt_e;
        if (sg * PD_W16_BM >= rows) return;
        nks = ff_s >> 6;
        tiles = sdata + (size_t)rt * nks * 2048u;
        scales = sscale + (size_t)rt * nks * 256u;
        nv = min(PD_W16_BM, rows - sg * PD_W16_BM);
        if (threadIdx.x < PD_W16_BM) {
            const uint32_t j = threadIdx.x;
            const uint32_t t = sg * PD_W16_BM + (j < nv ? j : 0u);
            wr[j] = sscale2[0];
            xr[j] = act + (size_t)t * aw + (size_t)k * ff_r;
            pr[j] = part + ((size_t)t * (k + 1u) + k) * embd;
        }
    }
    __syncthreads();
    const uint32_t nt = (nv + 7u) >> 3;
    float acc[4][4] = {};
    pd_w16_walk<4u>(acc, tiles, scales, nks, warp * 16u, g, tq, nt, xr);
    const uint32_t o = rt * 64u + warp * 16u + g;
#pragma unroll
    for (uint32_t n = 0; n < 4u; ++n) {
        if (n >= nt) break;
#pragma unroll
        for (uint32_t i = 0; i < 4u; ++i) {
            const uint32_t j = n * 8u + 2u * tq + (i & 1u);
            if (j >= nv) continue;
            pr[j][o + (i >> 1) * 8u] = acc[n][i] * wr[j];
        }
    }
#else
    (void)rdata; (void)rscale; (void)rscale2; (void)sdata; (void)sscale;
    (void)sscale2; (void)sorted_row; (void)sorted_slot; (void)block_expert;
    (void)topk_w; (void)act; (void)part; (void)ff_r; (void)ff_s; (void)embd;
    (void)k; (void)n_blocks; (void)rows;
#endif
}

// ---- launchers (ABI 680-681; the tiled plane, so cc12-only like the st/mt
// family). Routed work is moe_align's 32-row sorting of k-wide topk rows
// (sorted_row / sorted_slot [n_blocks * 32], PAD past a block's live rows,
// block_expert PAD for spare blocks); the shared expert covers rows 0..rows.

PD_EXPORT
int pd_nvf4_moe_up_relu2_w16(const void* rdata, const void* rscale,
                             const void* rscale2, const void* sdata,
                             const void* sscale, const void* sscale2,
                             const void* sorted_row, const void* sorted_slot,
                             const void* block_expert, const void* x, void* act,
                             uint32_t in_dim, uint32_t ff_r, uint32_t ff_s,
                             uint32_t k, uint32_t n_blocks, uint32_t rows,
                             void* stream) {
#ifndef PD_BS_HOST
    (void)rdata; (void)rscale; (void)rscale2; (void)sdata; (void)sscale;
    (void)sscale2; (void)sorted_row; (void)sorted_slot; (void)block_expert;
    (void)x; (void)act; (void)in_dim; (void)ff_r; (void)ff_s; (void)k;
    (void)n_blocks; (void)rows; (void)stream;
    return cudaErrorNotSupported;
#else
    if (rows == 0 || k == 0) return 0;
    if ((in_dim & 63u) != 0 || (ff_r & 63u) != 0 || (ff_s & 63u) != 0)
        return cudaErrorInvalidValue;
    const uint32_t grid = n_blocks * (ff_r >> 6) +
                          ((rows + PD_W16_BM - 1u) / PD_W16_BM) * (ff_s >> 6);
    pd_nv4w16_up_kernel<<<grid, 128u, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)rdata, (const uint8_t*)rscale, (const float*)rscale2,
        (const uint8_t*)sdata, (const uint8_t*)sscale, (const float*)sscale2,
        (const uint32_t*)sorted_row, (const uint32_t*)sorted_slot,
        (const uint32_t*)block_expert, (const __nv_bfloat16*)x, (__nv_bfloat16*)act,
        in_dim, ff_r, ff_s, k, n_blocks, rows);
    return pd_launch_status();
#endif
}

PD_EXPORT
int pd_nvf4_moe_down_part_w16(const void* rdata, const void* rscale,
                              const void* rscale2, const void* sdata,
                              const void* sscale, const void* sscale2,
                              const void* sorted_row, const void* sorted_slot,
                              const void* block_expert, const void* topk_w,
                              const void* act, void* part, uint32_t ff_r,
                              uint32_t ff_s, uint32_t embd, uint32_t k,
                              uint32_t n_blocks, uint32_t rows, void* stream) {
#ifndef PD_BS_HOST
    (void)rdata; (void)rscale; (void)rscale2; (void)sdata; (void)sscale;
    (void)sscale2; (void)sorted_row; (void)sorted_slot; (void)block_expert;
    (void)topk_w; (void)act; (void)part; (void)ff_r; (void)ff_s; (void)embd;
    (void)k; (void)n_blocks; (void)rows; (void)stream;
    return cudaErrorNotSupported;
#else
    if (rows == 0 || k == 0 || embd == 0) return 0;
    if ((ff_r & 63u) != 0 || (ff_s & 63u) != 0 || (embd & 63u) != 0)
        return cudaErrorInvalidValue;
    const uint32_t rt_e = embd >> 6;
    const uint32_t grid = n_blocks * rt_e + ((rows + PD_W16_BM - 1u) / PD_W16_BM) * rt_e;
    pd_nv4w16_dn_kernel<<<grid, 128u, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)rdata, (const uint8_t*)rscale, (const float*)rscale2,
        (const uint8_t*)sdata, (const uint8_t*)sscale, (const float*)sscale2,
        (const uint32_t*)sorted_row, (const uint32_t*)sorted_slot,
        (const uint32_t*)block_expert, (const float*)topk_w,
        (const __nv_bfloat16*)act, (float*)part, ff_r, ff_s, embd, k, n_blocks, rows);
    return pd_launch_status();
#endif
}

// ---- the class's routing front (ABI 685): router matvec + sigmoid top-k +
// bf16 activations + moe_align's 32-row sorting in ONE launch.
//
// A W16 MoE layer ran matvec_f32_batch -> moe_topk_sigmoid_batch ->
// convert_f32_bf16 -> moe_align -> up -> down: four of the six launches are
// tiny, and each boundary is a full drain plus a dependent's start (GB10
// 2026-09-26, one-row decode: 7.8 + 4.7 + 0.9 + 2.4 us a layer unoverlapped,
// 23 layers). Here the matvec CTAs (grid (n_expert, ceil(rows / BT)), the
// router's decode-width body pd_matvec_f32_batch_body, BT-invariant) also
// cast their token rows to bf16 (the x-column-0 CTAs; round to nearest,
// convert_f32_bf16's bits).
//
// One summation order at every row count - and that is a class fix, not
// only a fusion: the unfused router launcher moves from this body (a 256-
// thread K stride) to a lane-strided tile kernel (a 32-lane stride) at 16
// rows, so a 40-row verify round (8 slots x depth 4) scored its router
// logits in another order than the 8-row decode tick it stands for, and a
// near-tie could pick another expert. Below 16 rows the logits are the
// unfused router's bits (gpu_nemotron_w16 gates both).
//
// Two tickets then hand the work on, PD_LAST_BLOCK_FOLD's reader
// discipline at each (release fence, ticket, acquire fence, cross-CTA data
// through L2): the last CTA of a row group runs the top-k warp routine
// (pd_moe_topk_sigmoid_warp, the unfused kernel's) over its BT rows, and the
// last row group runs moe_align's body over every row's picks. Each resets
// its ticket, so the next launch finds them zero. `tickets` is 1 + ceil(rows
// / BT) u32, zeroed once at allocation and owned by one stream's launches
// (they never overlap: a launch touches them only after its predecessor
// completed).
template <uint32_t BT>
__global__ void __launch_bounds__(256) pd_moe_route_w16_kernel(
    const float* __restrict__ rw, const float* __restrict__ x, float* __restrict__ logits,
    const float* __restrict__ bias, float routed_scale, uint32_t in_dim, uint32_t n_expert,
    uint32_t k, __nv_bfloat16* __restrict__ x16, uint32_t* __restrict__ out_idx,
    float* __restrict__ out_w, uint32_t* __restrict__ sorted_row,
    uint32_t* __restrict__ sorted_slot, uint32_t* __restrict__ block_expert,
    uint32_t max_blocks, uint32_t rows, uint32_t* __restrict__ tickets) {
    PD_PDL_ARM();
    const uint32_t t0 = blockIdx.y * BT, t1 = min(t0 + BT, rows);
    pd_matvec_f32_batch_body<BT>(rw, x, logits, in_dim, n_expert, rows, blockIdx.x, t0);
    if (blockIdx.x == 0u) {
        for (uint32_t t = t0; t < t1; ++t)
            for (uint32_t i = threadIdx.x; i < in_dim; i += blockDim.x)
                x16[(size_t)t * in_dim + i] = __float2bfloat16_rn(x[(size_t)t * in_dim + i]);
    }
    extern __shared__ float rsh[];
    __shared__ uint32_t s_last;
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t nwarp = blockDim.x >> 5;
    // the row group's last CTA out: its rows' top-k
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0u)
        s_last = atomicAdd(&tickets[1u + blockIdx.y], 1u) == gridDim.x - 1u;
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    float* sl = rsh + (size_t)warp * n_expert;
    for (uint32_t t = t0 + warp; t < t1; t += nwarp) {
        for (uint32_t i = lane; i < n_expert; i += 32u)
            sl[i] = __ldcg(logits + (size_t)t * n_expert + i);
        __syncwarp();
        pd_moe_topk_sigmoid_warp(sl, bias, routed_scale, n_expert, k, out_idx + (size_t)t * k,
                                 out_w + (size_t)t * k);
        __syncwarp();
    }
    if (threadIdx.x == 0u) tickets[1u + blockIdx.y] = 0u;
    // the last row group out: every row's picks sorted into 32-row blocks
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0u) s_last = atomicAdd(&tickets[0], 1u) == gridDim.y - 1u;
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    unsigned int* ash = reinterpret_cast<unsigned int*>(rsh + (size_t)nwarp * n_expert);
    unsigned int* sidx = ash + 3u * n_expert;
    for (uint32_t p = threadIdx.x; p < rows * k; p += blockDim.x) sidx[p] = __ldcg(out_idx + p);
    __syncthreads();
    pd_moe_align_body(sidx, sorted_row, sorted_slot, block_expert, rows, k, n_expert, PD_W16_BM,
                      max_blocks, ash);
    if (threadIdx.x == 0u) tickets[0] = 0u;
}

// ABI 685. rows <= PD_W16_ROUTE_ROWS; `tickets` >= 1 + ceil(rows / 2) u32,
// zero at the first launch (the kernel leaves them zero).
#define PD_W16_ROUTE_ROWS 128u
PD_EXPORT
int pd_moe_route_w16(const void* router_w, const void* x, void* logits, const void* bias,
                     float routed_scale, uint32_t in_dim, uint32_t n_expert, uint32_t k,
                     void* x16, void* out_idx, void* out_w, void* sorted_row, void* sorted_slot,
                     void* block_expert, uint32_t max_blocks, uint32_t rows, void* tickets,
                     void* stream) {
    if (rows == 0) return 0;
    if (n_expert == 0 || n_expert > PD_MOE_MAX_EXPERT || k == 0 || k > 16u ||
        rows > PD_W16_ROUTE_ROWS || tickets == nullptr)
        return cudaErrorInvalidValue;
    const uint32_t nth = 256u;
    // per-warp logit rows for the top-k, the align's 3 x n_expert, the picks
    const uint32_t smem = ((nth / 32u) * n_expert + 3u * n_expert + rows * k) * 4u;
    // tokens per CTA as the router launcher elects them (the sums are
    // BT-invariant): 2 fills the die at decode widths, wider rows re-read
    // the plane less
    const uint32_t bt = rows < 16u ? 2u : 8u;
    dim3 grid(n_expert, (rows + bt - 1u) / bt);
#define PD_ROUTE_GO(BT)                                                                    \
    pd_pdl_go(pd_moe_route_w16_kernel<BT>, grid, nth, smem, (cudaStream_t)stream,           \
              (const float*)router_w, (const float*)x, (float*)logits, (const float*)bias,   \
              routed_scale, in_dim, n_expert, k, (__nv_bfloat16*)x16, (uint32_t*)out_idx,     \
              (float*)out_w, (uint32_t*)sorted_row, (uint32_t*)sorted_slot,                  \
              (uint32_t*)block_expert, max_blocks, rows, (uint32_t*)tickets)
    if (bt == 2u) PD_ROUTE_GO(2u);
    else PD_ROUTE_GO(8u);
#undef PD_ROUTE_GO
    return pd_launch_status();
}
