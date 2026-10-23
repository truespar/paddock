// gemm/dense_w16.cuh - batch-invariant dense GEMM over rows on tensor cores:
// y[t][o] = scale[o] * sum_k w[o][k] * x[t][k], bf16 or e4m3 weights against
// 16-bit activations, one kernel for every row count - one-row decode, a decode
// tick's rows, a spec verify round's. The dense half of the W16 decode class
// (moe/nvf4_w16.cuh is the experts').
// Textually-included segment of the single pack translation unit.
// Not standalone-compilable: include order is defined by ../pack.cu.
//
// The lanes it replaces for decode-class rows changed class with the row
// count: one row read f32 activations on FFMA GEMVs (bf16_gemv, f8r_gemv) while
// two or more took the bf16 MMA tile (its tile shape, K depth and split elected
// by batch, declining one row) or quantized the activations to e4m3 per row
// (W8A8). A spec verify row therefore scored its token in another class than
// the one-row tick it stands for.
//
// BATCH INVARIANT by construction: a weight row is the MMA's M, a token its N;
// columns never mix, and every CTA walks the full K in one fixed order - chunk
// c of CW k to warp c % 8, CW / 16 k16 steps each, the 8 warps' partials
// folded in warp order - whatever the row count. How many 8-token n columns a
// CTA carries (NT = 1, 2 or 4, by the call's row count) changes which columns
// share a CTA, never a column's arithmetic. Inside each chunk the k-slots map
// to actual k by the permutation the expert pair uses (quad tq's slots in step
// s are k (CW / 4) * tq + 4 * s + {0..3}), applied to weights and activations
// alike, so a thread's A operand for a chunk is one contiguous 32-byte run of
// each of its two rows: CW = 64 for bf16, 128 for e4m3 (64 when K is not a
// multiple of 128 - a property of the plane, not of the call).
//
// Classes: bf16 weights x bf16 activations (mma f32.bf16.bf16), e4m3 weights
// widened exactly to f16 (cvt.f16x2.e4m3x2) x f16 activations (mma
// f32.f16.f16) - f16 carries three more mantissa bits than bf16, and the FP8
// planes here (the mamba in/out projections) read rms-normed inputs - and f16
// weights x f16 activations on the same mma (Nemotron 3.5 Parse's decoder,
// whose F32 masters land as f16), loaded like the bf16 plane. The f32
// activations convert in-kernel (round to nearest), so no cast launch rides
// each projection.
//
// Geometry (GB10, 2026-09-26, dense_w16_gb10_bench): the first cut (4 warps,
// 64-wide e4m3 chunks, 32 token columns of accumulators at any row count)
// streamed the FP8 planes at 190-208 GB/s against f8r_gemv's 220-240 - at
// ~100 registers only four 128-thread CTAs fit an SM, and a 16-byte e4m3 run
// per row half-filled each 128-byte line a warp touched. 8 warps, 32-byte runs
// and accumulators sized to the call's columns stream in_proj at 1.00-1.04x
// and out_proj at 0.96-1.00x the old lanes from one row to 16. Measured and
// not taken: a bulk L2 prefetch of the CTA's row block (within noise here -
// the chunked walk already keeps the block's lines in flight), two m16 tiles
// a warp, a third prefetched chunk, 256-wide e4m3 chunks (register spills).

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
#define PD_DW16_OK 1
#else
#define PD_DW16_OK 0
#endif

#define PD_DW16_WARPS 8u

// Up to three output segments over the weight rows: rows [base[s], base[s+1])
// land in y[s] at row stride stride[s] (a fused q|k|v plane writes its three
// projections in one launch). Unused segments carry base UINT32_MAX.
struct PdDw16Out {
    float* y[3];
    uint32_t base[3];
    uint32_t stride[3];
};

#if PD_DW16_OK
__device__ __forceinline__ void pd_dw16_mma_bf16(float d[4], const uint32_t a[4],
                                                 const uint32_t b[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void pd_dw16_mma_f16(float d[4], const uint32_t a[4],
                                                const uint32_t b[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// f32 pair -> 16-bit x2 {lo, hi}, round to nearest even
template <bool F16>
__device__ __forceinline__ uint32_t pd_dw16_pack(float lo, float hi) {
    uint32_t r;
    if constexpr (F16)
        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    else
        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}
// two e4m3 bytes (low byte = lower k) -> f16x2, exact
__device__ __forceinline__ uint32_t pd_dw16_e4m3x2(uint32_t two) {
    const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)two, __NV_E4M3);
    return (uint32_t)h.x | ((uint32_t)h.y << 16);
}

// A thread's 32-byte run of one weight row in chunk c (CW / 4 consecutive k).
struct PdDw16Run { uint4 u[2]; };

template <bool E4M3, uint32_t CW>
__device__ __forceinline__ PdDw16Run pd_dw16_load(const uint8_t* row, uint32_t c, uint32_t tq,
                                                  bool live) {
    constexpr uint32_t EB = E4M3 ? 1u : 2u;
    PdDw16Run r;
    if constexpr (E4M3 && CW == 64u) {
        // half run: 16 k = 16 bytes
        r.u[0] = live ? *reinterpret_cast<const uint4*>(row + (size_t)c * 64u + 16u * tq)
                      : make_uint4(0u, 0u, 0u, 0u);
        r.u[1] = make_uint4(0u, 0u, 0u, 0u);
    } else {
        const uint4* p =
            reinterpret_cast<const uint4*>(row + ((size_t)c * CW + (CW / 4u) * tq) * EB);
        r.u[0] = live ? p[0] : make_uint4(0u, 0u, 0u, 0u);
        r.u[1] = live ? p[1] : make_uint4(0u, 0u, 0u, 0u);
    }
    return r;
}

// the run as A-operand registers: w[2s] = k 4s..4s+1, w[2s+1] = k 4s+2..4s+3
template <bool E4M3, uint32_t CW>
__device__ __forceinline__ void pd_dw16_regs(const PdDw16Run& r, uint32_t (&w)[CW / 8u]) {
    if constexpr (E4M3) {
#pragma unroll
        for (uint32_t i = 0; i < CW / 64u; ++i) {
            const uint32_t v[4] = {r.u[i].x, r.u[i].y, r.u[i].z, r.u[i].w};
#pragma unroll
            for (uint32_t j = 0; j < 4u; ++j) {
                w[8u * i + 2u * j] = pd_dw16_e4m3x2(v[j] & 0xFFFFu);
                w[8u * i + 2u * j + 1u] = pd_dw16_e4m3x2(v[j] >> 16);
            }
        }
    } else {
        w[0] = r.u[0].x; w[1] = r.u[0].y; w[2] = r.u[0].z; w[3] = r.u[0].w;
        w[4] = r.u[1].x; w[5] = r.u[1].y; w[6] = r.u[1].z; w[7] = r.u[1].w;
    }
}

// one chunk against the CTA's live n8 columns: token n*8 + g (a dead column
// past the last live row reads that row and is never stored). X16: the
// activations arrive already 16-bit (the same round-to-nearest cast, done once
// for a wide call); otherwise f32, cast here. E4M3 picks the weight bytes
// (e4m3, widened in registers, or 16-bit as stored); F16 the mma class and the
// activations' 16-bit type (f16, or bf16).
template <bool E4M3, bool F16, bool X16, uint32_t CW, uint32_t NT>
__device__ __forceinline__ void pd_dw16_chunk(float (&acc)[NT][4], const PdDw16Run& rg,
                                              const PdDw16Run& rh, uint32_t k0, uint32_t g,
                                              uint32_t nt, uint32_t nv, const void* x,
                                              uint32_t in_dim) {
    constexpr uint32_t NR = CW / 8u;  // registers per row per chunk
    uint32_t wg[NR], wh[NR];
    pd_dw16_regs<E4M3, CW>(rg, wg);
    pd_dw16_regs<E4M3, CW>(rh, wh);
#pragma unroll
    for (uint32_t n = 0; n < NT; ++n) {
        if (n >= nt) break;
        const size_t xo = (size_t)min(n * 8u + g, nv - 1u) * in_dim + k0;
        uint32_t b[NR];
        if constexpr (X16) {
            const uint4* xp = reinterpret_cast<const uint4*>((const uint16_t*)x + xo);
#pragma unroll
            for (uint32_t i = 0; i < NR / 4u; ++i) {
                const uint4 u = xp[i];
                b[4u * i] = u.x; b[4u * i + 1u] = u.y; b[4u * i + 2u] = u.z; b[4u * i + 3u] = u.w;
            }
        } else {
            const float4* xp = reinterpret_cast<const float4*>((const float*)x + xo);
#pragma unroll
            for (uint32_t i = 0; i < NR / 2u; ++i) {
                const float4 f = xp[i];
                b[2u * i] = pd_dw16_pack<F16>(f.x, f.y);
                b[2u * i + 1u] = pd_dw16_pack<F16>(f.z, f.w);
            }
        }
#pragma unroll
        for (uint32_t s = 0; s < CW / 16u; ++s) {
            const uint32_t a[4] = {wg[2u * s], wh[2u * s], wg[2u * s + 1u], wh[2u * s + 1u]};
            const uint32_t bb[2] = {b[2u * s], b[2u * s + 1u]};
            if constexpr (F16) pd_dw16_mma_f16(acc[n], a, bb);
            else pd_dw16_mma_bf16(acc[n], a, bb);
        }
    }
}
#endif

// grid (ceil(out / 16), ceil(rows / (8 NT))): a CTA owns 16 output rows and up
// to 8 NT token rows; warp w walks chunks w, w + 8, ... two ahead in fixed
// register sets, and the 8 partials fold in warp order. `w` row-major
// [out][in], output per PdDw16Out. Launched as a programmatic dependent: the
// first chunks' weight loads (no predecessor's output) issue before the wait,
// the activations after it; the release fires once the walk's loads are out.
template <bool E4M3, bool F16, bool X16, uint32_t CW, uint32_t NT>
__global__ void __launch_bounds__(PD_DW16_WARPS * 32u) pd_dense_w16_kernel(
    const uint8_t* __restrict__ w, const float* __restrict__ rscale,
    const void* __restrict__ x, PdDw16Out out, uint32_t in_dim, uint32_t out_dim,
    uint32_t rows) {
#if PD_DW16_OK
    constexpr uint32_t KW = PD_DW16_WARPS;
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t g = lane >> 2, tq = lane & 3u;
    const uint32_t o0 = blockIdx.x * 16u;
    const uint32_t t0 = blockIdx.y * 8u * NT;
    const uint32_t og = o0 + g, oh = og + 8u;
    const bool lg = og < out_dim, lh = oh < out_dim;
    constexpr uint32_t EB = E4M3 ? 1u : 2u;
    const uint8_t* rg = w + (size_t)(lg ? og : 0u) * in_dim * EB;
    const uint8_t* rh = w + (size_t)(lh ? oh : 0u) * in_dim * EB;
    const uint32_t nks = in_dim / CW;
    uint32_t c = warp;
    PdDw16Run g0 = pd_dw16_load<E4M3, CW>(rg, c, tq, lg && c < nks);
    PdDw16Run h0 = pd_dw16_load<E4M3, CW>(rh, c, tq, lh && c < nks);
    PdDw16Run g1 = pd_dw16_load<E4M3, CW>(rg, c + KW, tq, lg && c + KW < nks);
    PdDw16Run h1 = pd_dw16_load<E4M3, CW>(rh, c + KW, tq, lh && c + KW < nks);
    PD_PDL_ARM_WAIT();
    const uint32_t nv = min(8u * NT, rows - t0);
    const uint32_t nt = (nv + 7u) >> 3;
    const void* xb = X16 ? (const void*)((const uint16_t*)x + (size_t)t0 * in_dim)
                         : (const void*)((const float*)x + (size_t)t0 * in_dim);
    float acc[NT][4] = {};
    for (; c + KW < nks; c += 2u * KW) {
        const PdDw16Run cg0 = g0, ch0 = h0;
        g0 = pd_dw16_load<E4M3, CW>(rg, c + 2u * KW, tq, lg && c + 2u * KW < nks);
        h0 = pd_dw16_load<E4M3, CW>(rh, c + 2u * KW, tq, lh && c + 2u * KW < nks);
        pd_dw16_chunk<E4M3, F16, X16, CW, NT>(acc, cg0, ch0, c * CW + (CW / 4u) * tq, g, nt, nv,
                                         xb, in_dim);
        const PdDw16Run cg1 = g1, ch1 = h1;
        g1 = pd_dw16_load<E4M3, CW>(rg, c + 3u * KW, tq, lg && c + 3u * KW < nks);
        h1 = pd_dw16_load<E4M3, CW>(rh, c + 3u * KW, tq, lh && c + 3u * KW < nks);
        pd_dw16_chunk<E4M3, F16, X16, CW, NT>(acc, cg1, ch1, (c + KW) * CW + (CW / 4u) * tq, g, nt,
                                         nv, xb, in_dim);
    }
    if (c < nks)
        pd_dw16_chunk<E4M3, F16, X16, CW, NT>(acc, g0, h0, c * CW + (CW / 4u) * tq, g, nt, nv, xb,
                                         in_dim);
    PD_PDL_RELEASE();
    // fold the eight K parts in warp order: ((warp 0 + 1) + 2) + ... + 7
    __shared__ float part[KW - 1u][NT][4][32];
    if (warp > 0) {
#pragma unroll
        for (uint32_t n = 0; n < NT; ++n)
#pragma unroll
            for (uint32_t i = 0; i < 4u; ++i) part[warp - 1u][n][i][lane] = acc[n][i];
    }
    __syncthreads();
    if (warp != 0 || !lg) return;
    const float sg = rscale ? rscale[og] : 1.0f, sh = rscale && lh ? rscale[oh] : 1.0f;
#pragma unroll
    for (uint32_t n = 0; n < NT; ++n) {
        if (n >= nt) break;
#pragma unroll
        for (uint32_t i = 0; i < 4u; ++i) {
            const uint32_t j = n * 8u + 2u * tq + (i & 1u);
            const bool hi = i >> 1;
            if (j >= nv || (hi && !lh)) continue;
            float v = acc[n][i];
#pragma unroll
            for (uint32_t q = 0; q + 1u < KW; ++q) v += part[q][n][i][lane];
            // the segment by select, not by indexing the parameter struct
            // (a dynamic index copies it to local memory)
            const uint32_t o = hi ? oh : og;
            const bool s2 = o >= out.base[2], s1 = !s2 && o >= out.base[1];
            float* yb = s2 ? out.y[2] : (s1 ? out.y[1] : out.y[0]);
            const uint32_t b = s2 ? out.base[2] : (s1 ? out.base[1] : 0u);
            const uint32_t st = s2 ? out.stride[2] : (s1 ? out.stride[1] : out.stride[0]);
            yb[(size_t)(t0 + j) * st + (o - b)] = v * (hi ? sh : sg);
        }
    }
#else
    (void)w; (void)rscale; (void)x; (void)out; (void)in_dim; (void)out_dim; (void)rows;
#endif
}

static int pd_dense_w16_go(const void* w, const void* rscale, const void* x,
                           const PdDw16Out& out, uint32_t in_dim, uint32_t out_dim,
                           uint32_t rows, uint32_t dtype, uint32_t x16, void* stream) {
    // columns per CTA by the call's rows - which columns share a CTA, never
    // a column's arithmetic. The f16 class (Parse's verify rounds: 2 rows a
    // slot, 64 at 32 slots) also takes 64 columns, so a 64-row call streams
    // its plane once instead of once per 32-column block; the bf16 / e4m3
    // elections stay as they were measured.
    const uint32_t nt = rows <= 8u ? 1u
                      : rows <= 16u ? 2u
                      : (rows <= 32u || dtype != 2u) ? 4u : 8u;
    dim3 grid((out_dim + 15u) / 16u, (rows + 8u * nt - 1u) / (8u * nt));
    const cudaStream_t st = (cudaStream_t)stream;
    const bool cw128 = dtype == 1u && (in_dim & 127u) == 0u;
#define PD_DW16_GO(E, F, X, CW, NT)                                                        \
    pd_pdl_go(pd_dense_w16_kernel<E, F, X, CW, NT>, grid, PD_DW16_WARPS * 32u, 0u, st,      \
              (const uint8_t*)w, (const float*)rscale, x, out, in_dim, out_dim, rows)
#define PD_DW16_NT(E, F, X, CW)                                                            \
    do {                                                                                   \
        if (nt == 1u) PD_DW16_GO(E, F, X, CW, 1u);                                         \
        else if (nt == 2u) PD_DW16_GO(E, F, X, CW, 2u);                                    \
        else PD_DW16_GO(E, F, X, CW, 4u);                                                  \
    } while (0)
#define PD_DW16_NT8(E, F, X, CW)                                                           \
    do {                                                                                   \
        if (nt == 8u) PD_DW16_GO(E, F, X, CW, 8u);                                         \
        else PD_DW16_NT(E, F, X, CW);                                                      \
    } while (0)
    if (dtype == 1u) {
        if (cw128) {
            if (x16) PD_DW16_NT(true, true, true, 128u); else PD_DW16_NT(true, true, false, 128u);
        } else {
            if (x16) PD_DW16_NT(true, true, true, 64u); else PD_DW16_NT(true, true, false, 64u);
        }
    } else if (dtype == 2u) {
        if (x16) PD_DW16_NT8(false, true, true, 64u); else PD_DW16_NT8(false, true, false, 64u);
    } else {
        if (x16) PD_DW16_NT(false, false, true, 64u); else PD_DW16_NT(false, false, false, 64u);
    }
#undef PD_DW16_NT8
#undef PD_DW16_NT
#undef PD_DW16_GO
    return pd_launch_status();
}

// ABI 682. `dtype` 0 = bf16 weights x bf16 activations, 1 = e4m3 weights
// (widened to f16) x f16 activations with a per-output-row f32 scale, 2 = f16
// weights x f16 activations (`rscale` nullable for the 16-bit planes). `x16` 0 = x is f32 [rows][in_dim] (cast in the
// kernel), 1 = x is already the class's 16-bit type (bf16 / f16 - the
// convert_f32_{bf16,f16} of the same rows, which rounds identically). K % 64
// == 0; any out_dim (rows past it neither load nor store); x rows past `rows`
// are never read. y [rows][y_stride].
PD_EXPORT
int pd_dense_w16(const void* w, const void* rscale, const void* x, void* y,
                 uint32_t in_dim, uint32_t out_dim, uint32_t rows, uint32_t y_stride,
                 uint32_t dtype, uint32_t x16, void* stream) {
    if (out_dim == 0 || rows == 0) return 0;
    if ((in_dim & 63u) != 0 || dtype > 2u || x16 > 1u || y_stride < out_dim)
        return cudaErrorInvalidValue;
    const PdDw16Out out = {{(float*)y, nullptr, nullptr},
                           {0u, UINT32_MAX, UINT32_MAX},
                           {y_stride, 0u, 0u}};
    return pd_dense_w16_go(w, rscale, x, out, in_dim, out_dim, rows, dtype, x16, stream);
}

// ABI 684. pd_dense_w16 over a plane whose rows split into up to three
// outputs, one launch: rows [0, n0) -> y0 [rows][n0], [n0, n0 + n1) -> y1
// [rows][n1], [n0 + n1, out_dim) -> y2 [rows][out_dim - n0 - n1] (a fused
// q|k|v plane; n1 = 0 and n0 + n1 = out_dim leave segments unused - their
// pointers may be null). Same class as 682: an output's bits do not depend on
// which segment it lands in.
PD_EXPORT
int pd_dense_w16_seg(const void* w, const void* rscale, const void* x, void* y0, void* y1,
                     void* y2, uint32_t n0, uint32_t n1, uint32_t in_dim, uint32_t out_dim,
                     uint32_t rows, uint32_t dtype, uint32_t x16, void* stream) {
    if (out_dim == 0 || rows == 0) return 0;
    if ((in_dim & 63u) != 0 || dtype > 2u || x16 > 1u || n0 == 0u || n0 > out_dim ||
        n1 > out_dim - n0 || y0 == nullptr || (n1 && y1 == nullptr) ||
        (n0 + n1 < out_dim && y2 == nullptr))
        return cudaErrorInvalidValue;
    const uint32_t n2 = out_dim - n0 - n1;
    const PdDw16Out out = {{(float*)y0, (float*)y1, (float*)y2},
                           {0u, n1 ? n0 : UINT32_MAX, n2 ? n0 + n1 : UINT32_MAX},
                           {n0, n1, n2}};
    return pd_dense_w16_go(w, rscale, x, out, in_dim, out_dim, rows, dtype, x16, stream);
}
