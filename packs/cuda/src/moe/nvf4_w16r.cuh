// moe/nvf4_w16r.cuh - the W4A16 expert pair (moe/nvf4_w16.cuh) over ROW-MAJOR
// planes, with a SwiGLU gate|up.
// Textually-included segment of the single pack translation unit.
// Not standalone-compilable: include order is defined by ../pack.cu.
//
// Why: a SwiGLU checkpoint whose file marks its experts 16-bit-activation
// (Flint's NVFP4 GGUF: prec_a4 false on every tensor) needs its prefill
// experts on tensor cores without quantizing the activations to NVFP4 - the
// W4A4 block-scale pair would serve a class the file says it was not built
// for. moe/nvf4_w16.cuh is that class, but it reads the tiled plane and its
// up kernel is relu^2. Keeping the planes row-major lets the f32-activation
// decode GEMVs (q4x_moe_gu_swiglu + nvf4_moe_down_acc) keep serving decode
// on the same bytes, so one residency carries both classes.
//
// Same math as the tiled pair, only the addressing moves: a thread quad tq's
// 16 k of a 64-wide tile are one 8-byte word at row * in/2 + ks * 32 + 8 tq
// and one scale byte at row * in/16 + ks * 4 + tq - the bytes the tiled
// layout's (piece tq/2, half tq%2) holds - so pd_w16_tile's dequant, k-slot
// permutation and MMAs are reused unchanged. Every warp walks the full K in
// ascending tiles: an output element's bits do not depend on the rows that
// share its block (batch invariant, as the tiled pair).
//
// gate|up: each task walks the gate plane, then the up plane, of one 64-row
// tile, and lands silu(g) * u (g, u times their expert's f32 factor; the
// decode GEMV's expression) as bf16 - the down's B operand.

#if PD_NV4_OK
// One warp's 16 rows of a row-major plane x the block's tokens, full K in
// ascending 64-wide tiles. `data` / `scale` point at the tile's first row;
// `dpitch` = in/2 and `spitch` = in/16 bytes. Two tiles of loads in flight,
// as pd_w16_walk.
template <uint32_t NTMAX>
__device__ __forceinline__ void pd_w16r_walk(float (&acc)[NTMAX][4],
                                             const uint8_t* __restrict__ data,
                                             const uint8_t* __restrict__ scale,
                                             uint32_t dpitch, uint32_t spitch, uint32_t nks,
                                             uint32_t r0, uint32_t g, uint32_t tq,
                                             uint32_t nt, const void* const* xr) {
    const uint32_t row_g = r0 + g, row_h = row_g + 8u;
    const uint8_t* dg = data + (size_t)row_g * dpitch + 8u * tq;
    const uint8_t* dh = data + (size_t)row_h * dpitch + 8u * tq;
    const uint8_t* sg = scale + (size_t)row_g * spitch + tq;
    const uint8_t* sh = scale + (size_t)row_h * spitch + tq;
    auto wword = [&](const uint8_t* p, uint32_t ks) {
        return ks < nks ? *reinterpret_cast<const uint2*>(p + (size_t)ks * 32u)
                        : make_uint2(0u, 0u);
    };
    auto sbyte = [&](const uint8_t* p, uint32_t ks) {
        return ks < nks ? (uint32_t)p[(size_t)ks * 4u] : 0u;
    };
    uint2 g0 = wword(dg, 0), h0 = wword(dh, 0);
    uint32_t sg0 = sbyte(sg, 0), sh0 = sbyte(sh, 0);
    uint2 g1 = wword(dg, 1), h1 = wword(dh, 1);
    uint32_t sg1 = sbyte(sg, 1), sh1 = sbyte(sh, 1);
    uint32_t ks = 0;
    for (; ks + 1u < nks; ks += 2u) {
        const uint2 cg0 = g0, ch0 = h0;
        const uint32_t csg0 = sg0, csh0 = sh0;
        g0 = wword(dg, ks + 2u); h0 = wword(dh, ks + 2u);
        sg0 = sbyte(sg, ks + 2u); sh0 = sbyte(sh, ks + 2u);
        pd_w16_tile(acc, cg0, ch0, csg0, csh0, ks * 64u + 16u * tq, g, nt, xr);
        const uint2 cg1 = g1, ch1 = h1;
        const uint32_t csg1 = sg1, csh1 = sh1;
        g1 = wword(dg, ks + 3u); h1 = wword(dh, ks + 3u);
        sg1 = sbyte(sg, ks + 3u); sh1 = sbyte(sh, ks + 3u);
        pd_w16_tile(acc, cg1, ch1, csg1, csh1, (ks + 1u) * 64u + 16u * tq, g, nt, xr);
    }
    if (ks < nks)
        pd_w16_tile(acc, g0, h0, sg0, sh0, ks * 64u + 16u * tq, g, nt, xr);
}
#endif

// gate|up + SwiGLU -> act, bf16 [rows][k * ff_r + ff_s]. Tasks: routed =
// (sorted 32-block, 64-row tile of ff_r), then shared = (group of 32 token
// rows, 64-row tile of ff_s). 4 warps x 16 rows.
__global__ void __launch_bounds__(128) pd_nv4w16r_gu_kernel(
    const uint8_t* __restrict__ gdata, const uint8_t* __restrict__ gscale,
    const float* __restrict__ gscale2, const uint8_t* __restrict__ udata,
    const uint8_t* __restrict__ uscale, const float* __restrict__ uscale2,
    const uint8_t* __restrict__ sgdata, const uint8_t* __restrict__ sgscale,
    const uint8_t* __restrict__ sudata, const uint8_t* __restrict__ suscale,
    float sg2, float su2, const uint32_t* __restrict__ sorted_row,
    const uint32_t* __restrict__ sorted_slot, const uint32_t* __restrict__ block_expert,
    const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ act, uint32_t in_dim,
    uint32_t ff_r, uint32_t ff_s, uint32_t k, uint32_t n_blocks, uint32_t rows) {
#if PD_NV4_OK
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t g = lane >> 2, tq = lane & 3u;
    const uint32_t rt_r = ff_r >> 6, rt_s = ff_s >> 6;
    const uint32_t n_rt = n_blocks * rt_r;
    const uint32_t task = blockIdx.x;
    const uint32_t aw = k * ff_r + ff_s;
    const uint32_t nks = in_dim >> 6, dpitch = in_dim >> 1, spitch = in_dim >> 4;
    __shared__ const void* xr[PD_W16_BM];
    __shared__ __nv_bfloat16* ar[PD_W16_BM];
    const uint8_t *gd, *gs, *ud, *us;
    float s2g, s2u;
    uint32_t rt, nv = 0;
    if (task < n_rt) {
        const uint32_t blk = task / rt_r;
        rt = task - blk * rt_r;
        const uint32_t e = block_expert[blk];
        if (e == PD_MOE_PAD) return;
        const size_t row0 = (size_t)e * ff_r + (size_t)rt * 64u;
        gd = gdata + row0 * dpitch; gs = gscale + row0 * spitch;
        ud = udata + row0 * dpitch; us = uscale + row0 * spitch;
        s2g = gscale2[e]; s2u = uscale2[e];
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
        const uint32_t sgp = (task - n_rt) / rt_s;
        rt = task - n_rt - sgp * rt_s;
        if (sgp * PD_W16_BM >= rows) return;
        const size_t row0 = (size_t)rt * 64u;
        gd = sgdata + row0 * dpitch; gs = sgscale + row0 * spitch;
        ud = sudata + row0 * dpitch; us = suscale + row0 * spitch;
        s2g = sg2; s2u = su2;
        nv = min(PD_W16_BM, rows - sgp * PD_W16_BM);
        if (threadIdx.x < PD_W16_BM) {
            const uint32_t j = threadIdx.x;
            const uint32_t t = sgp * PD_W16_BM + (j < nv ? j : 0u);
            xr[j] = x + (size_t)t * in_dim;
            ar[j] = act + (size_t)t * aw + (size_t)k * ff_r;
        }
    }
    __syncthreads();
    const uint32_t nt = (nv + 7u) >> 3;
    float accg[4][4] = {}, accu[4][4] = {};
    pd_w16r_walk<4u>(accg, gd, gs, dpitch, spitch, nks, warp * 16u, g, tq, nt, xr);
    pd_w16r_walk<4u>(accu, ud, us, dpitch, spitch, nks, warp * 16u, g, tq, nt, xr);
    // C[g | g+8][2tq | 2tq+1] -> silu(g) * u, bf16, into each live token's row
    const uint32_t o = rt * 64u + warp * 16u + g;
#pragma unroll
    for (uint32_t n = 0; n < 4u; ++n) {
        if (n >= nt) break;
#pragma unroll
        for (uint32_t i = 0; i < 4u; ++i) {
            const uint32_t j = n * 8u + 2u * tq + (i & 1u);
            if (j >= nv) continue;
            const float gv = accg[n][i] * s2g, uv = accu[n][i] * s2u;
            ar[j][o + (i >> 1) * 8u] =
                __float2bfloat16_rn(gv * (1.0f / (1.0f + expf(-gv))) * uv);
        }
    }
#else
    (void)gdata; (void)gscale; (void)gscale2; (void)udata; (void)uscale; (void)uscale2;
    (void)sgdata; (void)sgscale; (void)sudata; (void)suscale; (void)sg2; (void)su2;
    (void)sorted_row; (void)sorted_slot; (void)block_expert; (void)x; (void)act;
    (void)in_dim; (void)ff_r; (void)ff_s; (void)k; (void)n_blocks; (void)rows;
#endif
}

// down + the topk-weight fold -> part f32 [rows][k + 1][embd] (slot k = the
// shared expert), for moe_slot_combine's fixed slot order. The tiled pair's
// down over row-major planes: routed rows of expert e at e * embd + r.
__global__ void __launch_bounds__(128) pd_nv4w16r_dn_kernel(
    const uint8_t* __restrict__ rdata, const uint8_t* __restrict__ rscale,
    const float* __restrict__ rscale2, const uint8_t* __restrict__ sdata,
    const uint8_t* __restrict__ sscale, float ss2, const uint32_t* __restrict__ sorted_row,
    const uint32_t* __restrict__ sorted_slot, const uint32_t* __restrict__ block_expert,
    const float* __restrict__ topk_w, const __nv_bfloat16* __restrict__ act,
    float* __restrict__ part, uint32_t ff_r, uint32_t ff_s, uint32_t embd, uint32_t k,
    uint32_t n_blocks, uint32_t rows) {
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
    const uint8_t *d, *s;
    uint32_t rt, kin, nv = 0;
    if (task < n_rt) {
        const uint32_t blk = task / rt_e;
        rt = task - blk * rt_e;
        const uint32_t e = block_expert[blk];
        if (e == PD_MOE_PAD) return;
        kin = ff_r;
        const size_t row0 = (size_t)e * embd + (size_t)rt * 64u;
        d = rdata + row0 * (kin >> 1);
        s = rscale + row0 * (kin >> 4);
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
        const uint32_t sgp = (task - n_rt) / rt_e;
        rt = task - n_rt - sgp * rt_e;
        if (sgp * PD_W16_BM >= rows) return;
        kin = ff_s;
        const size_t row0 = (size_t)rt * 64u;
        d = sdata + row0 * (kin >> 1);
        s = sscale + row0 * (kin >> 4);
        nv = min(PD_W16_BM, rows - sgp * PD_W16_BM);
        if (threadIdx.x < PD_W16_BM) {
            const uint32_t j = threadIdx.x;
            const uint32_t t = sgp * PD_W16_BM + (j < nv ? j : 0u);
            wr[j] = ss2;
            xr[j] = act + (size_t)t * aw + (size_t)k * ff_r;
            pr[j] = part + ((size_t)t * (k + 1u) + k) * embd;
        }
    }
    __syncthreads();
    const uint32_t nt = (nv + 7u) >> 3;
    float acc[4][4] = {};
    pd_w16r_walk<4u>(acc, d, s, kin >> 1, kin >> 4, kin >> 6, warp * 16u, g, tq, nt, xr);
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
    (void)rdata; (void)rscale; (void)rscale2; (void)sdata; (void)sscale; (void)ss2;
    (void)sorted_row; (void)sorted_slot; (void)block_expert; (void)topk_w; (void)act;
    (void)part; (void)ff_r; (void)ff_s; (void)embd; (void)k; (void)n_blocks; (void)rows;
#endif
}

// ---- launchers (ABI 918-919). Routed work is moe_align's 32-row sorting of
// k-wide topk rows (as 680-681); the shared expert covers rows 0..rows and
// is required (ff_s > 0). Row planes: routed gate/up [n_expert * ff_r][in],
// down [n_expert * embd][ff_r], shared [ff_s][in] / [embd][ff_s]; codes
// [.., in/2], e4m3 scales [.., in/16]; in, ff_r, ff_s, embd % 64.

PD_EXPORT
int pd_nvf4_moe_gu_swiglu_w16r(const void* gdata, const void* gscale, const void* gscale2,
                               const void* udata, const void* uscale, const void* uscale2,
                               const void* sgdata, const void* sgscale, const void* sudata,
                               const void* suscale, float sg2, float su2,
                               const void* sorted_row, const void* sorted_slot,
                               const void* block_expert, const void* x, void* act,
                               uint32_t in_dim, uint32_t ff_r, uint32_t ff_s, uint32_t k,
                               uint32_t n_blocks, uint32_t rows, void* stream) {
#ifndef PD_BS_HOST
    (void)gdata; (void)gscale; (void)gscale2; (void)udata; (void)uscale; (void)uscale2;
    (void)sgdata; (void)sgscale; (void)sudata; (void)suscale; (void)sg2; (void)su2;
    (void)sorted_row; (void)sorted_slot; (void)block_expert; (void)x; (void)act;
    (void)in_dim; (void)ff_r; (void)ff_s; (void)k; (void)n_blocks; (void)rows; (void)stream;
    return cudaErrorNotSupported;
#else
    if (rows == 0 || k == 0) return 0;
    if ((in_dim & 63u) != 0 || (ff_r & 63u) != 0 || (ff_s & 63u) != 0 || ff_s == 0)
        return cudaErrorInvalidValue;
    const uint32_t grid = n_blocks * (ff_r >> 6) +
                          ((rows + PD_W16_BM - 1u) / PD_W16_BM) * (ff_s >> 6);
    pd_nv4w16r_gu_kernel<<<grid, 128u, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)gdata, (const uint8_t*)gscale, (const float*)gscale2,
        (const uint8_t*)udata, (const uint8_t*)uscale, (const float*)uscale2,
        (const uint8_t*)sgdata, (const uint8_t*)sgscale, (const uint8_t*)sudata,
        (const uint8_t*)suscale, sg2, su2, (const uint32_t*)sorted_row,
        (const uint32_t*)sorted_slot, (const uint32_t*)block_expert,
        (const __nv_bfloat16*)x, (__nv_bfloat16*)act, in_dim, ff_r, ff_s, k, n_blocks, rows);
    return pd_launch_status();
#endif
}

PD_EXPORT
int pd_nvf4_moe_down_part_w16r(const void* rdata, const void* rscale, const void* rscale2,
                               const void* sdata, const void* sscale, float ss2,
                               const void* sorted_row, const void* sorted_slot,
                               const void* block_expert, const void* topk_w, const void* act,
                               void* part, uint32_t ff_r, uint32_t ff_s, uint32_t embd,
                               uint32_t k, uint32_t n_blocks, uint32_t rows, void* stream) {
#ifndef PD_BS_HOST
    (void)rdata; (void)rscale; (void)rscale2; (void)sdata; (void)sscale; (void)ss2;
    (void)sorted_row; (void)sorted_slot; (void)block_expert; (void)topk_w; (void)act;
    (void)part; (void)ff_r; (void)ff_s; (void)embd; (void)k; (void)n_blocks; (void)rows;
    (void)stream;
    return cudaErrorNotSupported;
#else
    if (rows == 0 || k == 0 || embd == 0) return 0;
    if ((ff_r & 63u) != 0 || (ff_s & 63u) != 0 || (embd & 63u) != 0 || ff_s == 0)
        return cudaErrorInvalidValue;
    const uint32_t rt_e = embd >> 6;
    const uint32_t grid = n_blocks * rt_e + ((rows + PD_W16_BM - 1u) / PD_W16_BM) * rt_e;
    pd_nv4w16r_dn_kernel<<<grid, 128u, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)rdata, (const uint8_t*)rscale, (const float*)rscale2,
        (const uint8_t*)sdata, (const uint8_t*)sscale, ss2, (const uint32_t*)sorted_row,
        (const uint32_t*)sorted_slot, (const uint32_t*)block_expert, (const float*)topk_w,
        (const __nv_bfloat16*)act, (float*)part, ff_r, ff_s, embd, k, n_blocks, rows);
    return pd_launch_status();
#endif
}
