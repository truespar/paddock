// gemm/q8_sb.cuh - the Q8_0 decode-band GEMM: b <= 16 activation columns
// against a repacked Q8_0 plane streamed once (elected inside
// pd_q8_0_gemm_mma_ks for b <= 16; included by int8_mma.cuh just above it,
// after its cp.async helpers).
//
// The K-split mma tile (pd_q8_0_gemm_mma_kernel at BN16) held ~170 GB/s on
// GB10 at spec-verify widths - 70% of the ~245 GB/s a plain linear read
// reaches - on every plane of a 12B dense decode round. Probing the access
// pattern alone (no math) found the cause: what a single load instruction
// covers of each weight row. A warp instruction touching 16 rows x 32 B (the
// mma fragment's own layout: lane (g, t) reads rows g and g + 8 at 8t) reads
// at 110-175 GB/s, 64 B a row at ~200-240, 128 B a row at the roof, 256 B a
// row better again on the wide planes (head 222 -> 236) - independent of 8-
// or 16-byte lanes. The tile's 64 x 128 B cp.async stages and a register-
// direct fragment streamer both sat in the slow class.
//
// So this rung stages: one WARP owns a 16-row fragment and streams it
// through its own shared-memory ring - 16 rows x 256 B (8 k32 blocks) a
// stage, each cp.async instruction 2 rows x 256 B of 16-byte lanes, the 16
// rows' f16 scales beside them - and reads the mma fragments back with
// 8-byte LDS. No CTA barrier (the ring is warp-private, __syncwarp orders
// it). The ring's row stride is exactly 256 B: a 16-byte pad made the
// cp.async stores themselves cost ~20% of the read rate, so the 16-byte
// units are XOR-swizzled by row instead (unit u of row r at u ^ (r & 7);
// the fragment reads then see at most 2-way conflicts). Inside each 32-block
// the k order is permuted so a lane's A and B words are 8 contiguous bytes -
// lane t holds k 8t..8t+7 where the canonical fragment has 4t..4t+3 and
// 16+4t..16+4t+3; an exact integer dot does not care which lane holds which
// k, as long as A and B agree. Activations (B and its scales) are tiny and
// L1-resident; they are loaded one stage ahead in registers.
//
// BIT-IDENTICAL to the tile it replaces, by construction: the same z slices
// (nz from the out-tile count, slices rounded to 4 blocks), the same
// ascending block order, the same per-block exact s32 dot, the same fold
// expression `acc += ws * xs * (float)d`, and the same slice combine. A
// row's bits therefore stay independent of the batch width, the property
// the spec-exact gates and batched-equals-alone rely on. Two ways to cover
// the slices:
//   A  one warp walks every slice of its rows end to end (long runs per
//      row), folding each finished slice into a z-ordered total exactly as
//      the combine does (0 + p0 + p1 + ... then the bias) and writing y -
//      no partial planes, no combine launch. nz == 1 writes acc (+ bias)
//      directly, as the tile does (0 + acc would turn -0 into +0).
//   B  grid.z = nz, one slice a warp, raw partial planes for the combine -
//      the tile's own layout. Only for planes too narrow to fill the die
//      with whole-K warps (< 64 fragments: 3840 -> 512 has 32).

#define PD_Q8SB_RB 256u                                    // bytes a row a stage
#define PD_Q8SB_STAGE (16u * PD_Q8SB_RB + 16u * 16u)       // + 16 rows x 8 f16 scales
#define PD_Q8SB_DEPTH 2u                                   // stages a warp's ring holds
#define PD_Q8SB_WPC 4u                                     // warps (fragments) a CTA
#define PD_Q8SB_AMIN 64u                                   // fragments for way A

// 8-byte predicated cp.async (the scale rows; src-size 0 zero-fills)
__device__ __forceinline__ void pd_q8sb_cpa8p(void* smem, const void* gmem, bool ok) {
#if PD_MMA_OK
    const unsigned sm = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;" ::"r"(sm), "l"(gmem),
                 "r"(ok ? 8u : 0u));
#endif
}
__device__ __forceinline__ uint2 pd_q8sb_lds2(const void* p) {
    uint2 v;
    asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];"
                 : "=r"(v.x), "=r"(v.y)
                 : "r"((unsigned)__cvta_generic_to_shared(p)));
    return v;
}
// byte offset of (row r, byte b) in a stage: 16-byte unit u of row r sits at
// u ^ (r & 7)
__device__ __forceinline__ uint32_t pd_q8sb_at(uint32_t r, uint32_t b) {
    return r * PD_Q8SB_RB + ((((b >> 4) ^ (r & 7u)) << 4) | (b & 15u));
}

template <uint32_t NF>
__global__ void __launch_bounds__(PD_Q8SB_WPC * 32u) pd_q8_0_gemm_sb_kernel(
        const int8_t* __restrict__ data, const __half* __restrict__ scale,
        const int8_t* __restrict__ xq, const float* __restrict__ xs,
        const float* __restrict__ bias, float* __restrict__ y, uint32_t in_dim,
        uint32_t out_dim, uint32_t batch, uint32_t nz, uint32_t zw) {
#if PD_MMA_OK
    constexpr uint32_t C = PD_Q8SB_RB / 32u;   // k32 blocks a stage
    constexpr uint32_t D = PD_Q8SB_DEPTH;
    extern __shared__ __align__(16) uint8_t pd_q8sb_smem[];
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u;
    const uint32_t g = lane >> 2u, t = lane & 3u;
    const uint32_t row0 = (blockIdx.x * PD_Q8SB_WPC + warp) * 16u;
    if (row0 >= out_dim) return;                // warp-uniform
    uint8_t* ring = pd_q8sb_smem + warp * D * PD_Q8SB_STAGE;
    const uint32_t n_blocks = in_dim >> 5u;

    // the tile's slicing (PD_MMA_NSUBK = 4): slice z is
    // [z * per, min((z + 1) * per, n_blocks)); this warp covers zw of them
    // from z0 (zw = nz: all of K; zw = 1: slice blockIdx.z)
    const uint32_t per = nz > 1u ? ((n_blocks + nz - 1u) / nz + PD_MMA_NSUBK - 1u) /
                                       PD_MMA_NSUBK * PD_MMA_NSUBK
                                 : n_blocks;
    const uint32_t z0 = blockIdx.z * zw;
    const uint32_t kt_lo = z0 * per < n_blocks ? z0 * per : n_blocks;
    const uint32_t kt_hi = zw == nz || (z0 + zw) * per > n_blocks ? n_blocks : (z0 + zw) * per;
    if (zw == 1u && nz > 1u) y += (size_t)z0 * out_dim * batch;
    // stages; slices are 4-block multiples, so the last one may be half
    const uint32_t nst = (kt_hi - kt_lo + C - 1u) / C;

    // copy lanes: rows 2i + lane / 16, the 16 B unit lane % 16 (= block
    // (lane % 16) / 2 of the stage); scales: row lane / 2, 4 blocks at
    // (lane & 1) * 4. Rows past out_dim copy row 0 (always valid) and are
    // never stored.
    const uint32_t cr = lane >> 4u, cu = lane & 15u;
    const int8_t* cpa[8];
    #pragma unroll
    for (uint32_t i = 0; i < 8u; ++i) {
        const uint32_t r = row0 + 2u * i + cr;
        cpa[i] = data + (size_t)(r < out_dim ? r : 0u) * in_dim + cu * 16u;
    }
    const uint32_t srow = row0 + (lane >> 1u);
    const __half* cps =
        scale + (size_t)(srow < out_dim ? srow : 0u) * n_blocks + (lane & 1u) * 4u;
    auto issue = [&](uint32_t kb, uint32_t slot) {
        uint8_t* st = ring + slot * PD_Q8SB_STAGE;
        const bool live = kb + cu / 2u < kt_hi;
        #pragma unroll
        for (uint32_t i = 0; i < 8u; ++i)
            pd_mma_cpa16p(st + pd_q8sb_at(2u * i + cr, cu * 16u), cpa[i] + (size_t)kb * 32u, live);
        pd_q8sb_cpa8p(st + 16u * PD_Q8SB_RB + (lane >> 1u) * 16u + (lane & 1u) * 8u, cps + kb,
                      kb + (lane & 1u) * 4u < kt_hi);
    };

    const uint32_t r0 = row0 + g, r8 = row0 + 8u + g;
    const bool ok0 = r0 < out_dim, ok8 = r8 < out_dim;
    // B fragment f: this lane's column f*8 + g (its 8 k bytes), and the two
    // output columns f*8 + 2t, f*8 + 2t + 1 whose scales it folds with
    bool bok[NF], cok0[NF], cok1[NF];
    const int8_t* bp[NF];
    const float* x0p[NF];
    const float* x1p[NF];
    #pragma unroll
    for (uint32_t f = 0; f < NF; ++f) {
        const uint32_t cb = f * 8u + g, c0 = f * 8u + 2u * t, c1 = c0 + 1u;
        bok[f] = cb < batch;
        cok0[f] = c0 < batch;
        cok1[f] = c1 < batch;
        bp[f] = xq + (size_t)(bok[f] ? cb : 0u) * in_dim + 8u * t;
        x0p[f] = xs + (size_t)(cok0[f] ? c0 : 0u) * n_blocks;
        x1p[f] = xs + (size_t)(cok1[f] ? c1 : 0u) * n_blocks;
    }
    // one stage of activations, in registers end to end: returned and
    // rotated by value, every component picked by a compile-time index -
    // taking an address anywhere in here puts it on the stack
    struct Act {
        uint2 b[NF][C];
        float4 x0[NF][C / 4u], x1[NF][C / 4u];
    };
    auto aload = [&](uint32_t kb) {
        Act a;
        #pragma unroll
        for (uint32_t f = 0; f < NF; ++f) {
            #pragma unroll
            for (uint32_t j = 0; j < C; ++j) {
                const uint32_t blk = kb + j < kt_hi ? kb + j : kt_lo;
                a.b[f][j] = bok[f] ? __ldg(reinterpret_cast<const uint2*>(bp[f] + (size_t)blk * 32u))
                                   : make_uint2(0u, 0u);
            }
            #pragma unroll
            for (uint32_t q = 0; q < C / 4u; ++q) {
                const uint32_t blk = kb + 4u * q < kt_hi ? kb + 4u * q : kt_lo;
                a.x0[f][q] = cok0[f] ? __ldg(reinterpret_cast<const float4*>(x0p[f] + blk))
                                     : make_float4(0.f, 0.f, 0.f, 0.f);
                a.x1[f][q] = cok1[f] ? __ldg(reinterpret_cast<const float4*>(x1p[f] + blk))
                                     : make_float4(0.f, 0.f, 0.f, 0.f);
            }
        }
        return a;
    };
    // block j's f16 scale out of a packed 4-half word pair / its f32 out of a
    // float4 (j is a compile-time index under the unrolled loops)
    auto half_at = [](uint2 w, uint32_t j) -> float {
        const uint32_t word = j < 2u ? w.x : w.y;
        return __half2float(__ushort_as_half((unsigned short)((j & 1u) ? word >> 16 : word & 0xffffu)));
    };
    auto f4_at = [](float4 v, uint32_t j) -> float {
        return j == 0u ? v.x : j == 1u ? v.y : j == 2u ? v.z : v.w;
    };

    float acc[NF][4] = {}, tot[NF][4] = {};
    uint32_t zdone = 0u;
    uint32_t bound = kt_lo + per < kt_hi ? kt_lo + per : kt_hi;   // end of the slice in fold
    #pragma unroll
    for (uint32_t s = 0; s + 1u < D; ++s) {
        if (s < nst) issue(kt_lo + s * C, s);
        asm volatile("cp.async.commit_group;");
    }
    Act an = {};
    if (nst) an = aload(kt_lo);
    for (uint32_t s = 0; s < nst; ++s) {
        const uint32_t kb = kt_lo + s * C;
        if (s + D - 1u < nst) issue(kb + (D - 1u) * C, (s + D - 1u) % D);
        asm volatile("cp.async.commit_group;");
        const Act a = an;
        if (s + 1u < nst) an = aload(kb + C);
        pd_mma_cpa_waitN<(int)D - 1>();
        __syncwarp();                           // every lane's copies of stage s landed
        const uint8_t* st = ring + (s % D) * PD_Q8SB_STAGE;
        #pragma unroll
        for (uint32_t q = 0; q < C / 4u; ++q) {
            if (q > 0u && kb + 4u * q >= kt_hi) break;   // half stage (warp-uniform)
            const uint2 w0 = pd_q8sb_lds2(st + 16u * PD_Q8SB_RB + g * 16u + q * 8u);
            const uint2 w8 = pd_q8sb_lds2(st + 16u * PD_Q8SB_RB + (g + 8u) * 16u + q * 8u);
            #pragma unroll
            for (uint32_t jj = 0; jj < 4u; ++jj) {
                const uint32_t j = 4u * q + jj;
                const uint2 a0 = pd_q8sb_lds2(st + pd_q8sb_at(g, 32u * j + 8u * t));
                const uint2 a8 = pd_q8sb_lds2(st + pd_q8sb_at(g + 8u, 32u * j + 8u * t));
                const float ws0 = half_at(w0, jj), ws8 = half_at(w8, jj);
                #pragma unroll
                for (uint32_t f = 0; f < NF; ++f) {
                    int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                        : "r"(a0.x), "r"(a8.x), "r"(a0.y), "r"(a8.y), "r"(a.b[f][j].x),
                          "r"(a.b[f][j].y));
                    const float xc0 = f4_at(a.x0[f][q], jj), xc1 = f4_at(a.x1[f][q], jj);
                    // the tile's fold expression, verbatim
                    acc[f][0] += ws0 * xc0 * (float)d0;
                    acc[f][1] += ws0 * xc1 * (float)d1;
                    acc[f][2] += ws8 * xc0 * (float)d2;
                    acc[f][3] += ws8 * xc1 * (float)d3;
                }
            }
            // slices end on 4-block boundaries: a finished slice joins the
            // z-ordered total the way the combine adds partial planes
            if (zw > 1u && kb + 4u * q + 4u == bound) {
                #pragma unroll
                for (uint32_t f = 0; f < NF; ++f)
                    #pragma unroll
                    for (uint32_t e = 0; e < 4u; ++e) {
                        tot[f][e] += acc[f][e];
                        acc[f][e] = 0.0f;
                    }
                ++zdone;
                bound = bound + per < kt_hi ? bound + per : kt_hi;
            }
        }
        __syncwarp();                           // stage s is refilled next iteration
    }

    if (zw > 1u) {
        // slices that start past the plane's end are +0 partial planes in
        // the combine; add them too (-0 + +0 = +0)
        for (; zdone < nz; ++zdone)
            #pragma unroll
            for (uint32_t f = 0; f < NF; ++f)
                #pragma unroll
                for (uint32_t e = 0; e < 4u; ++e) tot[f][e] += 0.0f;
        #pragma unroll
        for (uint32_t f = 0; f < NF; ++f)
            #pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) acc[f][e] = tot[f][e];
    }
    const float b0f = (bias && ok0) ? bias[r0] : 0.0f;
    const float b8f = (bias && ok8) ? bias[r8] : 0.0f;
    #pragma unroll
    for (uint32_t f = 0; f < NF; ++f) {
        const uint32_t c0 = f * 8u + 2u * t, c1 = c0 + 1u;
        if (ok0) {
            if (c0 < batch) y[(size_t)c0 * out_dim + r0] = bias ? acc[f][0] + b0f : acc[f][0];
            if (c1 < batch) y[(size_t)c1 * out_dim + r0] = bias ? acc[f][1] + b0f : acc[f][1];
        }
        if (ok8) {
            if (c0 < batch) y[(size_t)c0 * out_dim + r8] = bias ? acc[f][2] + b8f : acc[f][2];
            if (c1 < batch) y[(size_t)c1 * out_dim + r8] = bias ? acc[f][3] + b8f : acc[f][3];
        }
    }
#else
    (void)data; (void)scale; (void)xq; (void)xs; (void)bias; (void)y;
    (void)in_dim; (void)out_dim; (void)batch; (void)nz; (void)zw;
#endif
}

// Whether the streamer takes this call: b <= 16, n_blocks % 4 == 0 (every
// dense plane served - it keeps the 8-byte scale copies and the float4
// activation-scale loads aligned), and aligned bases (16 B weights - the
// cp.async lanes - 8 B scales / quantized activations, 16 B activation
// scales; a plane carved out of a fused buffer may not be). -1 = read
// PADDOCK_NO_Q8SB once; benches set it to force either arm.
static int pd_q8sb_mode = -1;
static bool pd_q8sb_elect(const void* data, const void* scale, const void* xq, const void* xs,
                          uint32_t n_blocks, uint32_t batch) {
    if (pd_q8sb_mode < 0) pd_q8sb_mode = pd_env("PADDOCK_NO_Q8SB") ? 0 : 1;
    const bool aligned = (((uintptr_t)data | (uintptr_t)xs) & 15u) == 0u
                         && (((uintptr_t)scale | (uintptr_t)xq) & 7u) == 0u;
    return pd_q8sb_mode == 1 && batch <= 16u && (n_blocks & 3u) == 0u && n_blocks > 0u
           && aligned;
}

// Launch for the ks launcher's nz. Returns true when the kernel wrote y
// itself (bias included) - way A, or nz == 1 - and false when it left raw
// partial planes in `part` for the caller's combine (way B).
static bool pd_q8_0_gemm_sb_go(const void* data, const void* scale, const void* xq,
                               const void* xs, const float* bias, float* part, float* y,
                               uint32_t in_dim, uint32_t out_dim, uint32_t batch, uint32_t nz,
                               cudaStream_t st) {
    const uint32_t frags = (out_dim + 15u) / 16u;
    const bool whole = nz == 1u || frags >= PD_Q8SB_AMIN;
    const uint32_t zw = whole ? nz : 1u;
    dim3 grid((frags + PD_Q8SB_WPC - 1u) / PD_Q8SB_WPC, 1u, whole ? 1u : nz);
    // 4 warps x 2 stages x 4.25 KB = 34 KB: under the 48 KB default, no opt-in
    const size_t smem = (size_t)PD_Q8SB_WPC * PD_Q8SB_DEPTH * PD_Q8SB_STAGE;
    float* dst = whole ? y : part;
    const float* kbias = whole ? bias : nullptr;
    if (batch <= 8u)
        pd_q8_0_gemm_sb_kernel<1u><<<grid, PD_Q8SB_WPC * 32u, smem, st>>>(
            (const int8_t*)data, (const __half*)scale, (const int8_t*)xq, (const float*)xs,
            kbias, dst, in_dim, out_dim, batch, nz, zw);
    else
        pd_q8_0_gemm_sb_kernel<2u><<<grid, PD_Q8SB_WPC * 32u, smem, st>>>(
            (const int8_t*)data, (const __half*)scale, (const int8_t*)xq, (const float*)xs,
            kbias, dst, in_dim, out_dim, batch, nz, zw);
    return whole;
}
