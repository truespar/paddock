// quant/nvf4_f4tn.cuh - the f4t TMA tile at decode and verify widths (slots
// 597 / 972), split from quant/nvf4.cuh: f4t's loader and fragment plan over
// a 32- or 64-column batch tile. Needs nvf4.cuh's f4t maps, mma and ldmatrix
// helpers.

// ---- f4tn: the f4t TMA tile at DECODE width (2026-09-10, GB10) -------------
// The rival-gap audit put the whole c8 decode gap on the fp4 FFN decode GEMM:
// the cp.async decode arms (f4cn / f4cd) stream the qwen3.8 planes at 163-181
// GB/s where CUTLASS streams the same bytes at 222/207 and a D2D memcpy at
// ~236 - and f4t itself, the prefill tile with the batch padded to 128,
// already does 205/195 at B=8 (session 15 elected it at batch <= 32). What f4t
// pays at decode width is phantom work: 16 batch-column fragments of MMA per
// K-step for 8 real rows, a 16 KB activation box per stage, acc[16][4]
// pinning it at 1 CTA/SM. This twin keeps f4t's loader (TMA + mbarrier ring,
// 4 producer warps, KC=256 stages, 128B swizzle) and f4t's exact fragment /
// mma plan and per-acc K order (kt, kh, k64, n - bit-exact vs f4t / f4c) over
// a 32-column batch tile: 32-row activation boxes (4 KB + 512 B per stage),
// acc[4][4], 2 CTAs/SM under 46 KB of smem, four MMAs per warp per K-step
// instead of sixteen. Measured, that alone is not the lever (202 vs f4t's 204
// GB/s at B=8): at decode width the binder is the DRAM stream itself, and
// what moves it is the producer asking the TMA unit to PREFETCH the weight
// boxes two stages ahead into L2 (cp.async.bulk.prefetch.tensor) - the
// stream runs ahead of the smem ring without the smem: 220 / 213 GB/s on
// gate_up / down, CUTLASS's rate on the same bytes. batch <= 32, in_dim %
// 256 == 0, out_dim in 128-row tiles (masked). y is [batch, out_dim] f32;
// only rows < batch are written.
//
// YR = the batch tile's columns: 32 (slot 597), or 64 (slot 972, 2026-11-22):
// a verify round's 33..64 rows otherwise fell to f4c's 128-column tile - two
// to four times the phantom MMA work, and its cp.async stream: Ripple's
// gate / up / down at 64 rows 735 / 718 / 758 us a layer in serving. Same
// fragment plan per 16 columns and the same per-acc K order, so YR moves
// work between warps' loops, never a sum: bit-exact vs f4t / f4c / YR 32.
#define PD_F4TN_YROWS 32u
#define PD_F4TN_STAGE_YR(YR) (128u * 128u + 128u * 16u + (YR) * 128u + (YR) * 16u)
#define PD_F4TN_STAGE PD_F4TN_STAGE_YR(PD_F4TN_YROWS)  // 23040
// ST-stage ring: all weight data boxes first (ST x 16 KB, 1024-aligned), then
// the weight scales (ST x 2 KB), the activation boxes (ST x YR x 128 B,
// 1024-aligned: ST x 18432 is) and their scales (ST x YR x 16 B, 128-
// aligned), then ST mbarriers.
#define PD_F4TN_SMEM_YR(ST, YR) ((ST) * PD_F4TN_STAGE_YR(YR) + 64u)
#define PD_F4TN_SMEM(ST) PD_F4TN_SMEM_YR(ST, PD_F4TN_YROWS)

// ST = ring depth (2 = f4t's, 3 fits 1 CTA/SM), PF = stages of L2 prefetch
// issued ahead of the ring for the weight boxes (0 = none): the producer
// asks the TMA unit to pull stage kt+PF into L2 while stage kt lands in smem,
// so the DRAM stream runs PF stages ahead of the smem ring without the smem.
template <uint32_t ST, uint32_t PF, uint32_t YR = PD_F4TN_YROWS>
__global__ void __launch_bounds__(384, (ST <= 2u && YR <= 32u) ? 2 : 1) pd_nvf4_gemm_f4tn_kernel(
    const __grid_constant__ PdTmap wdm, const __grid_constant__ PdTmap wsm,
    const __grid_constant__ PdTmap ydm, const __grid_constant__ PdTmap ysm,
    float* __restrict__ y, float scale2, const float* __restrict__ bias,
    uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
#if PD_BS_OK
    extern __shared__ __align__(1024) unsigned char pd_f4tn_sh[];
    unsigned char* wdat = pd_f4tn_sh;                                  // ST x 16384
    unsigned char* wsc = pd_f4tn_sh + ST * 16384u;                     // ST x 2048
    static_assert(YR == 32u || YR == 64u, "a 32- or 64-column batch tile");
    constexpr uint32_t YB = YR * 128u, YSB = YR * 16u;                 // per stage
    unsigned char* ydat = pd_f4tn_sh + ST * 18432u;                    // ST x YB
    unsigned char* ysc = pd_f4tn_sh + ST * (18432u + YB);              // ST x YSB
    unsigned long long* mb =
        (unsigned long long*)(pd_f4tn_sh + ST * PD_F4TN_STAGE_YR(YR));

    const uint32_t tid = threadIdx.x;
    const uint32_t nk = in_dim >> 8;  // KC=256 chunks (launcher gates %256)
    const uint32_t row_base = blockIdx.x * 128u;

    if (tid == 0u) {
        const uint32_t m0 = (uint32_t)__cvta_generic_to_shared(mb);
        #pragma unroll
        for (uint32_t s = 0; s < ST; ++s)
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 128;" ::"r"(m0 + s * 8u));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    asm volatile("bar.sync 0, 384;");

    if (tid >= 256u) {
        // ---------------- producer warps 8-11 (f4t's, narrow y boxes) -------
        const uint32_t ptid = tid - 256u;
        for (uint32_t kt = 0; kt < nk; ++kt) {
            const uint32_t b = kt % ST;
            if (kt >= ST) asm volatile("bar.sync %0, 384;" ::"r"(1u + b));
            const uint32_t m = (uint32_t)__cvta_generic_to_shared(mb) + b * 8u;
            if (ptid == 0u) {
                asm volatile(
                    "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(m),
                    "r"(PD_F4TN_STAGE_YR(YR)));
                const uint32_t wd = (uint32_t)__cvta_generic_to_shared(wdat + b * 16384u);
                const uint32_t ws = (uint32_t)__cvta_generic_to_shared(wsc + b * 2048u);
                const uint32_t yd = (uint32_t)__cvta_generic_to_shared(ydat + b * YB);
                const uint32_t ys = (uint32_t)__cvta_generic_to_shared(ysc + b * YSB);
                const int ck = (int)(kt * 128u), sk16 = (int)(kt * 16u);
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3}], [%4];" ::"r"(wd), "l"(&wdm), "r"(ck),
                    "r"((int)row_base), "r"(m) : "memory");
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3}], [%4];" ::"r"(ws), "l"(&wsm), "r"(sk16),
                    "r"((int)row_base), "r"(m) : "memory");
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3}], [%4];" ::"r"(yd), "l"(&ydm), "r"(ck),
                    "r"(0), "r"(m) : "memory");
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3}], [%4];" ::"r"(ys), "l"(&ysm), "r"(sk16),
                    "r"(0), "r"(m) : "memory");
                if (PF > 0u && kt + PF < nk) {
                    const int pck = (int)((kt + PF) * 128u), psk = (int)((kt + PF) * 16u);
                    asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];"
                                 ::"l"(&wdm), "r"(pck), "r"((int)row_base) : "memory");
                    asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];"
                                 ::"l"(&wsm), "r"(psk), "r"((int)row_base) : "memory");
                }
            } else {
                asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(m));
            }
        }
        return;
    }

    // ---------------- consumer warps 0-7 (f4t's fragment plan, YR cols) -----
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, tq = lane & 3u;
    const uint32_t i0 = (warp >> 1) * 32u;
    const uint32_t joff = (warp & 1u) * 8u;

    float acc[YR / 8u][4] = {};
    uint32_t phbits = 0u;

    for (uint32_t kt = 0; kt < nk; ++kt) {
        const uint32_t b = kt % ST;
        const uint32_t m = (uint32_t)__cvta_generic_to_shared(mb) + b * 8u;
        const uint32_t ph = (phbits >> b) & 1u;
        asm volatile(
            "{\n\t.reg .pred P;\n"
            "PD_F4TN_WAIT_%=:\n\t"
            "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
            "@!P bra PD_F4TN_WAIT_%=;\n\t}" ::"r"(m), "r"(ph) : "memory");
        phbits ^= 1u << b;

        const unsigned char* tw = wdat + b * 16384u;
        const unsigned char* tws = wsc + b * 2048u;
        const unsigned char* ty = ydat + b * YB;
        const unsigned char* tys = ysc + b * YSB;

        #pragma unroll
        for (uint32_t kh = 0; kh < 2u; ++kh) {
            const uint32_t sb8 = kh * 8u;
            uint32_t am[2][2][4], sa[2][2];
            #pragma unroll
            for (uint32_t n = 0; n < 2u; ++n) {
                const uint32_t r0 = i0 + n * 16u + g;
                const uint32_t rs = (tq & 1u) ? r0 + 8u : r0;
                const uint32_t rr = i0 + n * 16u + ((lane >> 3) & 1u) * 8u + (lane & 7u);
                #pragma unroll
                for (uint32_t k64 = 0; k64 < 2u; ++k64) {
                    const uint32_t c = kh * 4u + k64 * 2u + (lane >> 4);
                    pd_ldm_x4(am[n][k64], tw + rr * 128u + ((c ^ (rr & 7u)) * 16u));
                    sa[n][k64] = *(const uint32_t*)(tws + rs * 16u + sb8 + k64 * 4u);
                }
            }
            #pragma unroll
            for (uint32_t j0 = 0; j0 < YR; j0 += 16u) {
                const uint32_t col = j0 + joff + (lane & 7u);
                const uint32_t cy = kh * 4u + (lane >> 3);
                uint32_t bm[4];
                pd_ldm_x4(bm, ty + col * 128u + ((cy ^ (col & 7u)) * 16u));
                const unsigned char* ysr = tys + (j0 + joff + g) * 16u + sb8;
                #pragma unroll
                for (uint32_t k64 = 0; k64 < 2u; ++k64) {
                    const uint32_t sbv = *(const uint32_t*)(ysr + k64 * 4u);
                    #pragma unroll
                    for (uint32_t n = 0; n < 2u; ++n)
                        pd_nv4_mma(acc[(j0 >> 3) + n], am[n][k64][0], am[n][k64][1],
                                   am[n][k64][2], am[n][k64][3], bm[k64 * 2u],
                                   bm[k64 * 2u + 1u], sa[n][k64], sbv);
                }
            }
        }
        asm volatile("bar.arrive %0, 384;" ::"r"(1u + b));
    }

    // f4t's epilogue: scale2, conditional bias, f32 y, rows < batch only
    #pragma unroll
    for (uint32_t j0 = 0; j0 < YR; j0 += 16u) {
        const uint32_t c0 = j0 + joff + 2u * tq;
        #pragma unroll
        for (uint32_t n = 0; n < 2u; ++n) {
            const uint32_t r0 = row_base + i0 + n * 16u + g;
            const uint32_t r8 = r0 + 8u;
            float v00 = acc[(j0 >> 3) + n][0], v01 = acc[(j0 >> 3) + n][1];
            float v10 = acc[(j0 >> 3) + n][2], v11 = acc[(j0 >> 3) + n][3];
            v00 *= scale2; v01 *= scale2; v10 *= scale2; v11 *= scale2;
            if (bias) {
                if (r0 < out_dim) { const float bv = bias[r0]; v00 += bv; v01 += bv; }
                if (r8 < out_dim) { const float bv = bias[r8]; v10 += bv; v11 += bv; }
            }
            if (r0 < out_dim) {
                if (c0 < batch) y[(size_t)c0 * out_dim + r0] = v00;
                if (c0 + 1u < batch) y[(size_t)(c0 + 1u) * out_dim + r0] = v01;
            }
            if (r8 < out_dim) {
                if (c0 < batch) y[(size_t)c0 * out_dim + r8] = v10;
                if (c0 + 1u < batch) y[(size_t)(c0 + 1u) * out_dim + r8] = v11;
            }
        }
    }
#else
    (void)wdm; (void)wsm; (void)ydm; (void)ysm; (void)y; (void)scale2; (void)bias;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}

#ifdef PD_BS_HOST
// narrow activation maps for f4tn: the f4t maps with YR-row boxes (data
// 128B x YR under the 128B swizzle, scales 16 B x YR plain); gdim rows =
// batch, so rows past it zero-fill. Cached on (ptr, scale ptr, rows, k) like
// the f4t activation cache and for the same reason (see the corruption note)
// - one table per box height, so a 32- and a 64-row map never alias.
static bool pd_tmap_2d_rows(CUtensorMap* map, const void* base, uint64_t inner,
                            uint64_t rows, uint32_t box_rows, bool swz) {
    pd_tmap_encode_fn enc = pd_tmap_encode();
    if (!enc || ((uintptr_t)base & 15u) || (inner & 15u)) return false;
    const cuuint64_t gdim[2] = {inner, rows};
    const cuuint64_t gstride[1] = {inner};
    const cuuint32_t box[2] = {swz ? 128u : 16u, box_rows};
    const cuuint32_t estride[2] = {1u, 1u};
    return enc(map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2u, (void*)base, gdim, gstride,
               box, estride, CU_TENSOR_MAP_INTERLEAVE_NONE,
               swz ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE,
               CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}
template <uint32_t YR>
static inline bool pd_f4tn_ymaps(const void* xq, const void* xs, uint32_t in_dim, uint32_t batch,
                                 CUtensorMap** ydm, CUtensorMap** ysm) {
    static PdF4tYEnt yc[64]; static uint32_t yn = 0;
    *ydm = nullptr; *ysm = nullptr;
    for (uint32_t i = 0; i < yn; ++i)
        if (yc[i].p == xq && yc[i].s == xs && yc[i].rows == batch && yc[i].k == in_dim) { *ydm = &yc[i].dm; *ysm = &yc[i].sm; return true; }
    PdF4tYEnt& e = yc[yn % 64u];
    if (!pd_tmap_2d_rows(&e.dm, xq, in_dim >> 1, batch, YR, true) ||
        !pd_tmap_2d_rows(&e.sm, xs, in_dim >> 4, batch, YR, false))
        return false;
    e.p = xq; e.s = xs; e.rows = batch; e.k = in_dim; *ydm = &e.dm; *ysm = &e.sm;
    yn = yn < 64u ? yn + 1u : yn;
    return true;
}
#endif

#ifdef PD_BS_HOST
template <uint32_t ST, uint32_t PF, uint32_t YR = PD_F4TN_YROWS>
static inline int pd_f4tn_launch(const void* data, const void* scale, const void* bias,
                                 const void* xq, const void* xs, void* y, float scale2,
                                 uint32_t in_dim, uint32_t out_dim, uint32_t batch,
                                 cudaStream_t stream) {
    CUtensorMap *wdm, *wsm, *ydm128, *ysm128, *ydm, *ysm;
    if (!pd_f4t_maps(data, scale, xq, xs, in_dim, out_dim, batch, &wdm, &wsm, &ydm128, &ysm128))
        return cudaErrorNotSupported;
    if (!pd_f4tn_ymaps<YR>(xq, xs, in_dim, batch, &ydm, &ysm)) return cudaErrorNotSupported;
    const uint32_t ntiles = (out_dim + 127u) / 128u;
    static int smem_set = 0;
    if (!smem_set) {
        cudaFuncSetAttribute(pd_nvf4_gemm_f4tn_kernel<ST, PF, YR>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)PD_F4TN_SMEM_YR(ST, YR));
        smem_set = 1;
    }
    pd_nvf4_gemm_f4tn_kernel<ST, PF, YR><<<ntiles, 384, PD_F4TN_SMEM_YR(ST, YR), stream>>>(
        *wdm, *wsm, *ydm, *ysm, (float*)y, scale2, (const float*)bias, in_dim, out_dim,
        batch);
    return pd_launch_status();
}
// (yr, st, pf) probe twin for the bench: a 32/64-column tile x 2/3-deep ring x
// 0/2/4 stages of L2 prefetch
static inline int pd_f4tn_variant_yr(uint32_t yr, uint32_t st, uint32_t pf, const void* data,
                                     const void* scale, const void* bias, const void* xq,
                                     const void* xs, void* y, float scale2, uint32_t in_dim,
                                     uint32_t out_dim, uint32_t batch, cudaStream_t stream) {
    if (out_dim == 0 || batch == 0) return 0;
    if ((in_dim & 255u) != 0 || batch > yr) return cudaErrorInvalidValue;
    #define PD_F4TN_CASE(Y, S, P) \
        if (yr == Y && st == S && pf == P) return pd_f4tn_launch<S, P, Y>(data, scale, bias, xq, xs, y, scale2, in_dim, out_dim, batch, stream);
    PD_F4TN_CASE(32u, 2u, 0u) PD_F4TN_CASE(32u, 2u, 2u) PD_F4TN_CASE(32u, 2u, 4u)
    PD_F4TN_CASE(32u, 3u, 0u) PD_F4TN_CASE(32u, 3u, 2u) PD_F4TN_CASE(32u, 3u, 4u)
    PD_F4TN_CASE(64u, 2u, 2u) PD_F4TN_CASE(64u, 2u, 4u)
    PD_F4TN_CASE(64u, 3u, 2u) PD_F4TN_CASE(64u, 3u, 4u)
    #undef PD_F4TN_CASE
    return cudaErrorInvalidValue;
}
static inline int pd_f4tn_variant(uint32_t st, uint32_t pf, const void* data, const void* scale,
                                  const void* bias, const void* xq, const void* xs, void* y,
                                  float scale2, uint32_t in_dim, uint32_t out_dim,
                                  uint32_t batch, cudaStream_t stream) {
    return pd_f4tn_variant_yr(PD_F4TN_YROWS, st, pf, data, scale, bias, xq, xs, y, scale2,
                              in_dim, out_dim, batch, stream);
}
#endif

// The decode-width launcher: slot 597. Weight maps come from the f4t cache
// (same plane, same boxes); the activation maps are the 32-row twins.
// batch > 32 or in_dim % 256 != 0 is refused - the caller keeps f4t / f4cn.
// The shipped (ST, PF) is the bench winner (nv4_dec_f4t_probe.cu).
PD_EXPORT
int pd_nvf4_gemm_f4tn(const void* data, const void* scale, const void* bias,
                      const void* xq, const void* xs, void* y, float scale2,
                      uint32_t in_dim, uint32_t out_dim, uint32_t batch,
                      void* stream) {
#ifndef PD_BS_HOST
    (void)data; (void)scale; (void)bias; (void)xq; (void)xs; (void)y;
    (void)scale2; (void)in_dim; (void)out_dim; (void)batch; (void)stream;
    return cudaErrorNotSupported;
#else
    // (2, 2): two-deep smem ring, two stages of L2 prefetch ahead - the bench
    // winner on GB10 (gate_up 456 us / 220 GB/s, down 236 / 213 vs 491 / 261
    // without the prefetch; CUTLASS 451 / 243). A 3-deep ring buys nothing
    // over it (456 / 233) and costs the second CTA per SM.
    return pd_f4tn_variant(2u, 2u, data, scale, bias, xq, xs, y, scale2, in_dim, out_dim,
                           batch, (cudaStream_t)stream);
#endif
}

// The verify-width twin: slot 972, batch 33..64 (any batch <= 64 is served;
// the engine keeps slot 597 at <= 32). Same maps and contract as slot 597.
PD_EXPORT
int pd_nvf4_gemm_f4tn64(const void* data, const void* scale, const void* bias,
                        const void* xq, const void* xs, void* y, float scale2,
                        uint32_t in_dim, uint32_t out_dim, uint32_t batch,
                        void* stream) {
#ifndef PD_BS_HOST
    (void)data; (void)scale; (void)bias; (void)xq; (void)xs; (void)y;
    (void)scale2; (void)in_dim; (void)out_dim; (void)batch; (void)stream;
    return cudaErrorNotSupported;
#else
    // (3, 4): three-deep ring (one CTA/SM - 83 KB of smem at 64 columns),
    // four stages of L2 prefetch. GB10, Ripple's planes, cold, us at 48 / 64
    // rows: gate|up 583 / 599 vs f4c 727 / 739, down 549 / 554 vs 709 / 732;
    // the (2, 2) ring 581 / 617 and 555 / 559. PADDOCK_F4TN64_ST / _PF pin
    // another ring for a sweep (ripple_verify_gemm_gb10_bench.cu MODE=ffn).
    static const uint32_t st = [] {
        const char* v = pd_env("PADDOCK_F4TN64_ST");
        return v ? (uint32_t)atoi(v) : 3u;
    }();
    static const uint32_t pf = [] {
        const char* v = pd_env("PADDOCK_F4TN64_PF");
        return v ? (uint32_t)atoi(v) : 4u;
    }();
    return pd_f4tn_variant_yr(64u, st, pf, data, scale, bias, xq, xs, y, scale2, in_dim,
                              out_dim, batch, (cudaStream_t)stream);
#endif
}
