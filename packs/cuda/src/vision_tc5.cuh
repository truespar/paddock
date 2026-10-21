// vision_tc5.cuh - the picture tower's bidirectional attention on tcgen05 (sm_100a).
// Textually-included segment of the single pack translation unit; needs only abi.cuh and
// tma_desc.cuh (PdTmap, pd_tmap_encode, PD_TC5_OK, pd_tc5_sdesc), so a bench probe can
// include it without the rest of the pack.
//
// Why. On B200 the qwen3.8 tower ran pd_vision_attn_mma_kernel (mma.sync m16n8k16, K/V
// staged synchronously per 64-key tile) at ~165 TFLOP/s: 1.43 ms a layer on a 1568 px page
// (6860 rows x 16 heads x hd 72), 27 layers, ~39 ms of GPU per picture - the largest single
// kernel cost on the recover load once the GEMMs moved to tcgen05. Same contract as
// pd_vision_attn_h (slot 620): q (pre-scaled), k, v, out are [batch][row][head][hd] halves.
//
// Structure (one CTA = one 128-query tile of one head of one picture; 192 threads):
//   warp 4  producer: TMA Q once, then K and V tiles (128 keys each) on 2-deep rings. The
//           maps are 4-D over [batch][row][head][hd] with a 64-half inner box and SW128, so
//           each tile lands as two [128 x 64-half] canonical images; hd past 72 is the map's
//           out-of-bounds zero fill (the second image carries hd 64..127).
//   warp 5  mma issuer: S = Q K^T into one of two tmem S buffers (kind::f16, M=128 N=128,
//           K = hd 0..79 = 5 mmas - the zero tail of the padded K is never multiplied), then,
//           once softmax has staged P, O += P V into ONE tmem accumulator with V read
//           transposed (instruction-descriptor bit 16, pf6g's probed trick): N=64 over hd
//           0..63 plus N=16 over hd 64..79. S(j+1) is issued before PV(j), so the next tile's
//           scores run while softmax works on this one.
//   warps 0-3 softmax: thread = query row. One tcgen05.ld pass of its 128 scores, row max,
//           exp2 against a REFERENCE max, P (f16) into one of two SW128 A images, then
//           p_full. The reference only moves when the row max outgrows it by 2^8 (FA4's lazy
//           rescale): then - and only then - the warp waits for the previous PV and scales
//           its O rows in tmem. P under a stale reference is <= 256, inside f16 and the f32
//           sums. So the common tile never waits on the PV it feeds, which the first cut
//           (O folded into 72 registers after every PV: 1.29 ms vs 1.43 at the page shape)
//           did 54 times a CTA.
// Numeric class: f16 operands, f32 accumulate, P rounded to f16 per tile - the class of
// pd_vision_attn_mma_kernel (which also rounds P to f16 for its PV mma); the softmax
// reduction order differs (per-tile online), so outputs match to f16 rounding, not bitwise.
//
// FALSIFIED around it (B200, page shape 6860 x 16 x 72, 2026-10-21; this kernel 376 us):
//   - ex2.approx.f16x2 for P: sm_100 lowers it to two scalar MUFU.EX2.F16 - no packed rate,
//     and the f32 round trip for l made it 459 us.
//   - P in tmem over its scores (the PV mma's A operand) instead of the smem image: passes the
//     f64 gate, 374 us - the smem store / proxy fence was not the cost.
//   - FA4's two query tiles a CTA (two softmax warpgroups, S0 S1 | PV0 S0' | PV1 S1', P in tmem,
//     setmaxnreg 40 / 232 over 384 threads): passes, 396 us. ncu has this kernel at MUFU 49% /
//     issue 35% - but the two groups run in lockstep (both tiles' S land together), so both
//     exponentiate at once on the shared MUFU (8 clk a warp ex2) and then both wait on the mmas.
//     FA4's FMA-emulated exp2 would move the wall to issue (~7 instructions an element at two
//     warps a scheduler); at hd 72 the mmas are too short to hide either.

#define PD_VTC5_SMEM (7u * 32768u + 1024u + 24u * 8u)   // Q + 2 K + 2 V + 2 P, align pad, barriers

#if PD_TC5_OK
__device__ __forceinline__ void pd_vtc5_bar_wait(uint64_t* bar, uint32_t parity) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("{\n\t.reg .pred p;\nW%=:\n\t"
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                 "@!p bra W%=;\n\t}" ::"r"(a), "r"(parity) : "memory");
}
__device__ __forceinline__ void pd_vtc5_arrive(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
                 ::"r"((uint32_t)__cvta_generic_to_shared(bar)) : "memory");
}
__device__ __forceinline__ void pd_vtc5_tma4(uint32_t dst, const PdTmap* map, int c0, int c1,
                                             int c2, int c3, uint64_t* bar) {
    asm volatile("cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes"
                 " [%0], [%1, {%2, %3, %4, %5}], [%6];"
                 ::"r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(c3),
                   "r"((uint32_t)__cvta_generic_to_shared(bar)) : "memory");
}
// tmem <-> registers, 32 lanes x 32 (or 16) columns; the caller issues tcgen05.wait::ld/st
__device__ __forceinline__ void pd_vtc5_ld32(uint32_t ta, uint32_t* r) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
        : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
          "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),"=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15]),
          "=r"(r[16]),"=r"(r[17]),"=r"(r[18]),"=r"(r[19]),"=r"(r[20]),"=r"(r[21]),"=r"(r[22]),"=r"(r[23]),
          "=r"(r[24]),"=r"(r[25]),"=r"(r[26]),"=r"(r[27]),"=r"(r[28]),"=r"(r[29]),"=r"(r[30]),"=r"(r[31])
        : "r"(ta));
}
__device__ __forceinline__ void pd_vtc5_ld16(uint32_t ta, uint32_t* r) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
        : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
          "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),"=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15])
        : "r"(ta));
}
__device__ __forceinline__ void pd_vtc5_st32(uint32_t ta, const uint32_t* r) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], "
        "{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,"
        "%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32};"
        ::"r"(ta), "r"(r[0]),"r"(r[1]),"r"(r[2]),"r"(r[3]),"r"(r[4]),"r"(r[5]),"r"(r[6]),"r"(r[7]),
          "r"(r[8]),"r"(r[9]),"r"(r[10]),"r"(r[11]),"r"(r[12]),"r"(r[13]),"r"(r[14]),"r"(r[15]),
          "r"(r[16]),"r"(r[17]),"r"(r[18]),"r"(r[19]),"r"(r[20]),"r"(r[21]),"r"(r[22]),"r"(r[23]),
          "r"(r[24]),"r"(r[25]),"r"(r[26]),"r"(r[27]),"r"(r[28]),"r"(r[29]),"r"(r[30]),"r"(r[31]));
}
__device__ __forceinline__ void pd_vtc5_st16(uint32_t ta, const uint32_t* r) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], "
        "{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};"
        ::"r"(ta), "r"(r[0]),"r"(r[1]),"r"(r[2]),"r"(r[3]),"r"(r[4]),"r"(r[5]),"r"(r[6]),"r"(r[7]),
          "r"(r[8]),"r"(r[9]),"r"(r[10]),"r"(r[11]),"r"(r[12]),"r"(r[13]),"r"(r[14]),"r"(r[15]));
}
__device__ __forceinline__ float pd_vtc5_ex2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
#endif

__global__ void __launch_bounds__(192, 1) pd_vision_attn_tc5_kernel(
    const __grid_constant__ PdTmap qmap, const __grid_constant__ PdTmap kmap,
    const __grid_constant__ PdTmap vmap, __half* __restrict__ out, uint32_t nq, uint32_t nkv,
    uint32_t n_heads) {
#if PD_TC5_OK
    constexpr uint32_t HD = 72u;   // the qwen3.8 / qwen3-vl tower; the launcher refuses others
    extern __shared__ unsigned char pd_vtc5_raw[];
    unsigned char* sh = pd_vtc5_raw
        + ((1024u - ((uint32_t)__cvta_generic_to_shared(pd_vtc5_raw) & 1023u)) & 1023u);
    unsigned char* sq = sh;                     // Q  [2 x 16 KB]
    unsigned char* sk = sq + 32768u;            // K  ring 2 x [2 x 16 KB]
    unsigned char* sv = sk + 2u * 32768u;       // V  ring 2 x [2 x 16 KB]
    unsigned char* sp = sv + 2u * 32768u;       // P  ring 2 x [2 x 16 KB]
    uint64_t* bars = (uint64_t*)(sp + 2u * 32768u);
    uint64_t* q_full = bars + 0;
    uint64_t* k_full = bars + 1;   // [2]
    uint64_t* k_empty = bars + 3;  // [2]
    uint64_t* v_full = bars + 5;   // [2]
    uint64_t* v_empty = bars + 7;  // [2]
    uint64_t* s_full = bars + 9;   // [2]
    uint64_t* s_empty = bars + 11; // [2]  one arrival per softmax warp
    uint64_t* p_full = bars + 13;  // [2]  one arrival per softmax warp
    uint64_t* pv_done = bars + 15; // [2]  PV(j) commits pv_done[j & 1]
    __shared__ uint32_t tmem_slot[1];

    const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
    const uint32_t q0 = blockIdx.x * 128u, head = blockIdx.y, bi = blockIdx.z;
    const uint32_t nt = (nkv + 127u) / 128u;

    if (tid == 0) {
        auto init = [](uint64_t* b, uint32_t n) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(b)), "r"(n));
        };
        init(q_full, 1);
        for (uint32_t i = 0; i < 2u; ++i) {
            init(&k_full[i], 1); init(&k_empty[i], 1);
            init(&v_full[i], 1); init(&v_empty[i], 1);
            init(&s_full[i], 1); init(&s_empty[i], 4);
            init(&p_full[i], 4); init(&pv_done[i], 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    if (warp == 5)
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;"
                     ::"r"((uint32_t)__cvta_generic_to_shared(tmem_slot)));
    asm volatile("tcgen05.fence::before_thread_sync;");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");
    const uint32_t tmem = tmem_slot[0];   // S: cols 0..255 (2 x 128); O: cols 256..335

    if (warp == 4) {
        // ---- producer ----
        if (lane == 0) {
            const uint32_t m = (uint32_t)__cvta_generic_to_shared(q_full);
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 32768;" ::"r"(m));
            const uint32_t d = (uint32_t)__cvta_generic_to_shared(sq);
            pd_vtc5_tma4(d, &qmap, 0, (int)head, (int)q0, (int)bi, q_full);
            pd_vtc5_tma4(d + 16384u, &qmap, 64, (int)head, (int)q0, (int)bi, q_full);
            for (uint32_t j = 0; j < nt; ++j) {
                const uint32_t s = j & 1u, ph = (j >> 1) & 1u;
                pd_vtc5_bar_wait(&k_empty[s], ph ^ 1u);
                const uint32_t mk = (uint32_t)__cvta_generic_to_shared(&k_full[s]);
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 32768;" ::"r"(mk));
                const uint32_t dk = (uint32_t)__cvta_generic_to_shared(sk + s * 32768u);
                pd_vtc5_tma4(dk, &kmap, 0, (int)head, (int)(j * 128u), (int)bi, &k_full[s]);
                pd_vtc5_tma4(dk + 16384u, &kmap, 64, (int)head, (int)(j * 128u), (int)bi, &k_full[s]);
                pd_vtc5_bar_wait(&v_empty[s], ph ^ 1u);
                const uint32_t mv = (uint32_t)__cvta_generic_to_shared(&v_full[s]);
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 32768;" ::"r"(mv));
                const uint32_t dv = (uint32_t)__cvta_generic_to_shared(sv + s * 32768u);
                pd_vtc5_tma4(dv, &vmap, 0, (int)head, (int)(j * 128u), (int)bi, &v_full[s]);
                pd_vtc5_tma4(dv + 16384u, &vmap, 64, (int)head, (int)(j * 128u), (int)bi, &v_full[s]);
            }
        }
        __syncwarp();
    } else if (warp == 5) {
        // ---- mma issuer ----
        if (lane == 0) {
            // kind::f16: f32 accumulate, f16 x f16, K-major A; M=128
            const uint32_t id_s = (1u << 4) | ((128u >> 3) << 17) | ((128u >> 4) << 24);
            const uint32_t id_v64 = (1u << 4) | (1u << 16) | ((64u >> 3) << 17) | ((128u >> 4) << 24);
            const uint32_t id_v16 = (1u << 4) | (1u << 16) | ((16u >> 3) << 17) | ((128u >> 4) << 24);
            const uint32_t q16 = (uint32_t)__cvta_generic_to_shared(sq) >> 4;
            auto mma = [](uint32_t d, uint64_t a, uint64_t b, uint32_t id, uint32_t en) {
                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
                             "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}"
                             ::"r"(d), "l"(a), "l"(b), "r"(id), "r"(en));
            };
            auto commit = [](uint64_t* b) {
                asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                             ::"r"((uint32_t)__cvta_generic_to_shared(b)) : "memory");
            };
            auto issue_s = [&](uint32_t j) {
                const uint32_t s = j & 1u, ph = (j >> 1) & 1u;
                pd_vtc5_bar_wait(&k_full[s], ph);
                pd_vtc5_bar_wait(&s_empty[s], ph ^ 1u);
                asm volatile("tcgen05.fence::after_thread_sync;");
                const uint32_t k16 = (uint32_t)__cvta_generic_to_shared(sk + s * 32768u) >> 4;
                // hd 0..79: chunk 0 (hd 0..63) k-steps 0..3, chunk 1 (hd 64..79) k-step 0
                #pragma unroll
                for (uint32_t kb = 0; kb < 5u; ++kb) {
                    const uint32_t off = (kb >> 2) * 1024u + (kb & 3u) * 2u;
                    mma(tmem + s * 128u, pd_tc5_sdesc(q16 + off), pd_tc5_sdesc(k16 + off), id_s,
                        kb > 0 ? 1u : 0u);
                }
                commit(&s_full[s]);
                commit(&k_empty[s]);
            };
            pd_vtc5_bar_wait(q_full, 0u);
            if (nt) issue_s(0);
            const uint32_t ob = tmem + 256u;
            for (uint32_t j = 0; j < nt; ++j) {
                if (j + 1u < nt) issue_s(j + 1u);
                const uint32_t s = j & 1u, ph = (j >> 1) & 1u;
                pd_vtc5_bar_wait(&p_full[s], ph);
                pd_vtc5_bar_wait(&v_full[s], ph);
                asm volatile("tcgen05.fence::after_thread_sync;");
                const uint32_t v16 = (uint32_t)__cvta_generic_to_shared(sv + s * 32768u) >> 4;
                const uint32_t p16 = (uint32_t)__cvta_generic_to_shared(sp + s * 32768u) >> 4;
                // K = 128 keys, 16 a step: P's chunk (keys/64) and in-row offset; V's 16-row step
                #pragma unroll
                for (uint32_t kb = 0; kb < 8u; ++kb) {
                    const uint64_t ad = pd_tc5_sdesc(p16 + (kb >> 2) * 1024u + (kb & 3u) * 2u);
                    const uint32_t en = (j > 0 || kb > 0) ? 1u : 0u;
                    mma(ob, ad, pd_tc5_sdesc(v16 + kb * 128u), id_v64, en);
                    mma(ob + 64u, ad, pd_tc5_sdesc(v16 + 1024u + kb * 128u), id_v16, en);
                }
                commit(&pv_done[s]);
                commit(&v_empty[s]);
            }
        }
        __syncwarp();
    } else {
        // ---- softmax (warps 0-3; thread = query row) ----
        const uint32_t row = warp * 32u + lane;
        const uint32_t tl = (warp * 32u) << 16;   // this warp's tmem lane quarter
        const uint32_t ot = tmem + tl + 256u;
        constexpr float L2E = 1.4426950408889634f;
        constexpr float LAZY = 8.0f;               // log2 of the largest P a stale reference allows
        float m_ref = -INFINITY, l_run = 0.0f;     // m_ref: log2 domain
        // PV(j) is completion j/2 of pv_done[j & 1]; every wait below is on the next one
        // outstanding, so the parity is never ambiguous
        auto wait_pv = [&](uint32_t j) { pd_vtc5_bar_wait(&pv_done[j & 1u], (j >> 1) & 1u); };
        for (uint32_t j = 0; j < nt; ++j) {
            const uint32_t s = j & 1u, ph = (j >> 1) & 1u;
            pd_vtc5_bar_wait(&s_full[s], ph);
            asm volatile("tcgen05.fence::after_thread_sync;");
            uint32_t r[128];
            #pragma unroll
            for (uint32_t c = 0; c < 4u; ++c) pd_vtc5_ld32(tmem + tl + s * 128u + c * 32u, r + c * 32u);
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            asm volatile("tcgen05.fence::before_thread_sync;");
            __syncwarp();
            if (lane == 0) pd_vtc5_arrive(&s_empty[s]);
            // keys past nkv (the last tile's TMA zero fill) take no weight
            const uint32_t valid = nkv - j * 128u;
            if (valid < 128u) {
                #pragma unroll
                for (uint32_t i = 0; i < 128u; ++i)
                    if (i >= valid) r[i] = 0xff800000u;   // -inf
            }
            // eight independent max chains (one serial chain is 127 dependent fmax on the
            // critical path of every tile)
            float m8[8];
            #pragma unroll
            for (uint32_t i = 0; i < 8u; ++i) m8[i] = __uint_as_float(r[i]);
            #pragma unroll
            for (uint32_t i = 8; i < 128u; ++i) m8[i & 7u] = fmaxf(m8[i & 7u], __uint_as_float(r[i]));
            const float mx = fmaxf(fmaxf(fmaxf(m8[0], m8[1]), fmaxf(m8[2], m8[3])),
                                   fmaxf(fmaxf(m8[4], m8[5]), fmaxf(m8[6], m8[7])));
            const float mx2 = mx * L2E;
            if (__any_sync(0xffffffffu, mx2 > m_ref + LAZY)) {
                const float nref = fmaxf(m_ref, mx2);
                const float alpha = pd_vtc5_ex2(m_ref - nref);   // 0 on the first tile
                if (j > 0) {
                    // O holds PV(0..j-1) under the old reference: let the last one land
                    wait_pv(j - 1u);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    uint32_t o[80];
                    pd_vtc5_ld32(ot, o);
                    pd_vtc5_ld32(ot + 32u, o + 32);
                    pd_vtc5_ld16(ot + 64u, o + 64);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (uint32_t d = 0; d < 80u; ++d) o[d] = __float_as_uint(__uint_as_float(o[d]) * alpha);
                    pd_vtc5_st32(ot, o);
                    pd_vtc5_st32(ot + 32u, o + 32);
                    pd_vtc5_st16(ot + 64u, o + 64);
                    asm volatile("tcgen05.wait::st.sync.aligned;");
                }
                l_run *= alpha;
                m_ref = nref;
            }
            // P ring slot s was last read by PV(j-2)
            if (j >= 2u) wait_pv(j - 2u);
            unsigned char* ps = sp + s * 32768u;
            float sum4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            #pragma unroll
            for (uint32_t u = 0; u < 16u; ++u) {
                // P as the PV mma's K-major SW128 A image: keys 8u..8u+7 of row `row` -> 16 B
                // unit u (chunk = keys / 64), stored at unit u ^ (row & 7)
                __half2 h[4];
                #pragma unroll
                for (uint32_t e = 0; e < 4u; ++e) {
                    const float p0 = pd_vtc5_ex2(fmaf(__uint_as_float(r[u * 8u + 2u * e]), L2E, -m_ref));
                    const float p1 = pd_vtc5_ex2(fmaf(__uint_as_float(r[u * 8u + 2u * e + 1u]), L2E, -m_ref));
                    sum4[e] += p0 + p1;
                    h[e] = __floats2half2_rn(p0, p1);
                }
                const uint32_t ch = u >> 3, uu = u & 7u;
                *reinterpret_cast<uint4*>(ps + ch * 16384u + row * 128u + ((uu ^ (row & 7u)) << 4)) =
                    *reinterpret_cast<const uint4*>(h);
            }
            l_run += (sum4[0] + sum4[1]) + (sum4[2] + sum4[3]);
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            __syncwarp();
            if (lane == 0) pd_vtc5_arrive(&p_full[s]);
        }
        // epilogue: the last PV, 1/l and the half store, 144 B a row (9 x 16 B, aligned: hd 72)
        if (nt) wait_pv(nt - 1u);
        asm volatile("tcgen05.fence::after_thread_sync;");
        uint32_t o[80];
        pd_vtc5_ld32(ot, o);
        pd_vtc5_ld32(ot + 32u, o + 32);
        pd_vtc5_ld16(ot + 64u, o + 64);
        asm volatile("tcgen05.wait::ld.sync.aligned;");
        const uint32_t qr = q0 + row;
        if (qr < nq) {
            const float inv = l_run > 0.0f ? 1.0f / l_run : 0.0f;
            __half* dst = out + (((size_t)bi * nq + qr) * n_heads + head) * HD;
            #pragma unroll
            for (uint32_t u = 0; u < HD / 8u; ++u) {
                __half2 h[4];
                #pragma unroll
                for (uint32_t e = 0; e < 4u; ++e)
                    h[e] = __floats2half2_rn(__uint_as_float(o[u * 8u + 2u * e]) * inv,
                                             __uint_as_float(o[u * 8u + 2u * e + 1u]) * inv);
                *reinterpret_cast<uint4*>(dst + u * 8u) = *reinterpret_cast<const uint4*>(h);
            }
        }
    }
    asm volatile("tcgen05.fence::before_thread_sync;");
    __syncthreads();
    if (warp == 5)
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" ::"r"(tmem));
#else
    (void)qmap; (void)kmap; (void)vmap; (void)out; (void)nq; (void)nkv; (void)n_heads;
#endif
}

#if defined(PD_TC5_HOST)
// 4-D map over a [batch][row][head][hd] half plane: 64-half inner box (128 B, SW128) x 1 head
// x 128 rows x 1 picture. hd past the plane's own (72) reads as zero.
static bool pd_vtc5_map(CUtensorMap* m, const void* base, uint32_t hd, uint32_t n_heads,
                        uint32_t rows, uint32_t batch) {
    pd_tmap_encode_fn enc = pd_tmap_encode();
    if (!enc || ((uintptr_t)base & 15u)) return false;
    const cuuint64_t gdim[4] = {hd, n_heads, rows, batch};
    const cuuint64_t gstride[3] = {(cuuint64_t)hd * 2u, (cuuint64_t)n_heads * hd * 2u,
                                   (cuuint64_t)rows * n_heads * hd * 2u};
    const cuuint32_t box[4] = {64u, 1u, 128u, 1u};
    const cuuint32_t es[4] = {1u, 1u, 1u, 1u};
    return enc(m, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 4u, (void*)base, gdim, gstride, box, es,
               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
               CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

// pd_vision_attn_h's contract on tcgen05: self-attention (nq == nkv) over each of `batch`
// pictures, hd 72. Declines (cudaErrorNotSupported) off cc 10.0, for other head dims or for
// cross-attention, so the caller keeps the mma.sync kernel.
static int pd_vision_attn_tc5(const void* q, const void* k, const void* v, void* out, uint32_t nq,
                              uint32_t nkv, uint32_t n_heads, uint32_t head_dim, uint32_t batch,
                              cudaStream_t st) {
    static int ok = -1;
    if (ok < 0) {
        int dev = 0, ccM = 0, ccm = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&ccM, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&ccm, cudaDevAttrComputeCapabilityMinor, dev);
        ok = (ccM == 10 && ccm == 0 && pd_tmap_encode() != nullptr
              && pd_env("PADDOCK_NO_VIS_TC5") == nullptr) ? 1 : 0;
    }
    if (!ok || head_dim != 72u || nq != nkv || nq == 0) return cudaErrorNotSupported;
    CUtensorMap qm, km, vm;
    if (!pd_vtc5_map(&qm, q, head_dim, n_heads, nq, batch) ||
        !pd_vtc5_map(&km, k, head_dim, n_heads, nkv, batch) ||
        !pd_vtc5_map(&vm, v, head_dim, n_heads, nkv, batch))
        return cudaErrorNotSupported;
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute((const void*)pd_vision_attn_tc5_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)PD_VTC5_SMEM);
        attr = true;
    }
    const dim3 grid((nq + 127u) / 128u, n_heads, batch);
    pd_vision_attn_tc5_kernel<<<grid, 192, PD_VTC5_SMEM, st>>>(qm, km, vm, (__half*)out, nq, nkv,
                                                               n_heads);
    return (int)cudaPeekAtLastError();
}
#endif
