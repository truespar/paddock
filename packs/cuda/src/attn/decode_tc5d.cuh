// attn/decode_tc5d.cuh - qwen3.8's full-attention decode (24q / 4kv / hd 256, fp8-e4m3 paged
// KV) on tcgen05 with a decoupled TMA ring. Textually-included segment of the single pack
// translation unit; follows decode_tc5.cuh (pd_attn_decode_tc5_paged routes the G6 shape here)
// and needs tma_desc's pd_tc5_sdesc + f32_qkv's pd_attn_tmap_kv_f8s maps.
//
// Why. pd_attn_decode_tc5e_kernel (the two-team ping-pong, built for gemma's short SWA cells)
// stages K/V two 192-key slots deep and refills a slot only once the wedge barriers clear, so a
// CTA had about one tick of lookahead: ~4.3 us a 96 KB tick, 363 us a layer at the B200 serve's
// depth 128 (~110 rows x ~5000 keys). Its MMA shapes were right; its pipeline was not. Two-CTA occupancy with 96-key slots did not help (198 vs
// 207 pages/min): the bytes in flight stayed the same.
//
// Same numeric class and math shapes as tc5e: Q (pre-scaled) / K / V / P in e4m3, f32 accumulate,
// S^T = K Q^T (M=128 keys x N=8 heads, swapAB) and O^T = V^T P^T (M=128 hd x N=8, V transposed
// via instruction-descriptor bit 15). Whole (sequence, kv head) cells per CTA, so a row's math is
// its own at any batch (tc5e's invariance). The tile boundary is 128 keys (tc5e: 192) and the
// softmax uses a lazy reference (moves past 2^8 only), so outputs agree to the class, not bitwise.
//
// Shape (192 threads, one CTA an SM, persistent over cells dealt longest-first - cell_of):
//   warp 4  producer: per 128-key tile, 8 blocks x 2 hd halves of K and of V by 2 KB TMA boxes
//           into a 3-deep ring (64 KB a stage); block ids read from the table per tile.
//   warp 5  mma issuer (warp-converged, elect.sync): S(g) into one of two S buffers, then
//           PV(g-1) into the O accumulator - the next tile's scores run under this tile's softmax.
//   warps 0-3 softmax, thread = key: 6 head scores per thread, per-head tile max through warp
//           shuffles + a 4-warp smem exchange, P^T (e4m3) into one of two images, per-thread l.
//           They also build each cell's Q image (double-buffered) and run its epilogue (thread =
//           hd lane: O / l to the output rows) while the issuer moves on to the next cell.

#define PD_TC5D_TK 128u                     // keys a tile (8 blocks of 16)
#define PD_TC5D_NS 3u                       // ring stages
#define PD_TC5D_STAGE (2u * 2u * 16384u)    // K + V, two 128-B hd panels each
#define PD_TC5D_SMEM (PD_TC5D_NS * PD_TC5D_STAGE + 2u * 2048u + 2u * 1024u + 1024u)
#define PD_TC5D_LPT_MAX 512u                // sequences the longest-first deal ranks in smem

template <uint32_t G>
__global__ void __launch_bounds__(192, 1) pd_attn_decode_tc5d_kernel(
    const __grid_constant__ PdTmap tmk, const __grid_constant__ PdTmap tmv,
    const float* __restrict__ q, float* __restrict__ out,
    const unsigned int* __restrict__ positions, const unsigned int* __restrict__ slots,
    const uint32_t* __restrict__ block_tables, uint32_t blocks_per_slot, uint32_t swa_window,
    float scale, uint32_t n_kv, uint32_t n_seq, uint32_t kvh_div, uint32_t lpt_on) {
#if PD_ATTN_TMA_OK && PD_TC5_OK
    constexpr uint32_t HD = 256u, TK = PD_TC5D_TK, NS = PD_TC5D_NS, STG = PD_TC5D_STAGE;
    constexpr uint32_t PANEL = 16384u;      // one hd half of a tile: 128 keys x 128 B
    static_assert(G <= 8u, "the 8-col N tile carries the group");
    extern __shared__ __align__(1024) unsigned char pd_tc5d_raw[];
    unsigned char* sh = pd_tc5d_raw
        + ((1024u - ((uint32_t)__cvta_generic_to_shared(pd_tc5d_raw) & 1023u)) & 1023u);
    unsigned char* ring = sh;                         // NS x [K p0 | K p1 | V p0 | V p1]
    unsigned char* qimg = ring + NS * STG;            // 2 x 2 KB (cell parity)
    unsigned char* pimg = qimg + 2u * 2048u;          // 2 x 1 KB (tile parity)
    __shared__ __align__(8) uint64_t k_full[NS], v_full[NS], kv_empty[NS], s_full[2], s_empty[2],
        p_full[2], pv_done[2], q_ready[2], o_free;
    __shared__ float red[2][4][8];                    // per-head partials (max, then l)
    __shared__ uint32_t tmem_slot[1];

    const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
    const uint32_t n_cells = n_seq * n_kv;
    const uint32_t n_heads = G * n_kv;

    if (tid == 0) {
        auto init = [](uint64_t* b, uint32_t n) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(b)), "r"(n));
        };
        for (uint32_t i = 0; i < NS; ++i) { init(&k_full[i], 1); init(&v_full[i], 1); init(&kv_empty[i], 1); }
        for (uint32_t i = 0; i < 2u; ++i) {
            init(&s_full[i], 1); init(&s_empty[i], 4); init(&p_full[i], 4); init(&pv_done[i], 1);
            init(&q_ready[i], 4);
        }
        init(&o_free, 4);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    // P images start zeroed so the padding head rows (G..7) stay 0. (V rows past a cell's last
    // key are zeroed per tail tile by the softmax warps, below.)
    for (uint32_t i = tid; i < 2u * 1024u / 16u; i += 192u) ((uint4*)pimg)[i] = make_uint4(0u, 0u, 0u, 0u);
    if (warp == 5)
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;"
                     ::"r"((uint32_t)__cvta_generic_to_shared(tmem_slot)));
    asm volatile("fence.proxy.async.shared::cta;");
    asm volatile("tcgen05.fence::before_thread_sync;");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");
    const uint32_t tmem = tmem_slot[0];   // S: cols 0..15 (2 x 8); O^T: cols 16..31 (2 hd halves x 8)
    asm volatile("griddepcontrol.launch_dependents;");
    asm volatile("griddepcontrol.wait;" ::: "memory");

    auto bar_wait = [&](uint64_t* bar, uint32_t parity) {
        const uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
        asm volatile("{\n\t.reg .pred p;\nW%=:\n\t"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                     "@!p bra W%=;\n\t}" ::"r"(a), "r"(parity) : "memory");
    };
    auto arrive = [&](uint64_t* bar) {
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
                     ::"r"((uint32_t)__cvta_generic_to_shared(bar)) : "memory");
    };
    // a cell's key window and block range (tc5e's rules: glo from the window, ghi = pos + 1)
    struct Cell { uint32_t b, kvh, glo, ghi, B0, nblk, nt, slot; };
    auto cell_at = [&](uint32_t c) {
        Cell x;
        x.b = c / n_kv; x.kvh = c % n_kv;
        const uint32_t pos = __ldg(positions + x.b);
        x.glo = (swa_window > 0u && pos + 1u > swa_window) ? pos + 1u - swa_window : 0u;
        x.ghi = pos + 1u;
        x.B0 = x.glo >> 4;
        x.nblk = ((x.ghi + 15u) >> 4) - x.B0;
        x.nt = (x.nblk + 7u) / 8u;
        x.slot = slots ? __ldg(slots + x.b) : x.b;
        return x;
    };

    // Cells dealt longest first, snake-wise: every CTA ranks the sequences by key tiles (ties by
    // index - the same order in every CTA), and CTA b takes ranks b, 2 grid - 1 - b, 2 grid + b, ...
    // Round-robin over unsorted cells (~3.5 a CTA at the serve's depth 128, 3-8k keys) left the
    // busiest SM at 1.23x the mean (ncu sm__cycles_active max / avg). A cell's math is its own,
    // so the outputs do not move (tc5d_probe DUMP=, both arms: byte-identical at every case).
    // Probe, uniform 3-6k / 4-6k keys: 110 rows 180.1 -> 164.5 us, 128 rows 241.0 -> 228.2. The
    // B200 serve's decode ticks timed the same (225.5 vs 225.1 us a layer): there the layer moves
    // ~80% of DRAM peak, so an early-idle SM just hands its share of bandwidth to the others.
    // lpt_on 0 (PADDOCK_NO_TC5D_LPT): c = blockIdx.x + k grid.
    // (Correction, same day: the serve's layer reads 0.62 GB of DRAM in ~225 us but hits L2 43%
    // - the recover prompts share prefix blocks - so it walks ~1.75x that in KV; ncu in the serve:
    // 48.4 M instructions vs the probe's 22.0 M at 121 rows over 300-4700 keys, IPC 1.0 both,
    // 6 warps an SM, no warp eligible 75% of cycles. Latency-bound per tile, not DRAM-bound.
    // A scattered pool 32x the probe's (tc5d_probe POOLX / SCATTER) timed the same: not the TLB.)
    __shared__ uint16_t lpt_nt[PD_TC5D_LPT_MAX], lpt_seq[PD_TC5D_LPT_MAX];
    const bool lpt = lpt_on != 0u && n_seq <= PD_TC5D_LPT_MAX;
    if (lpt) {
        for (uint32_t i = tid; i < n_seq; i += 192u) lpt_nt[i] = (uint16_t)cell_at(i * n_kv).nt;
        __syncthreads();
        for (uint32_t i = tid; i < n_seq; i += 192u) {
            const uint32_t ti = lpt_nt[i];
            uint32_t r = 0;
            for (uint32_t j = 0; j < n_seq; ++j) {
                const uint32_t tj = lpt_nt[j];
                r += (tj > ti || (tj == ti && j < i)) ? 1u : 0u;
            }
            lpt_seq[r] = (uint16_t)i;
        }
        __syncthreads();
    }
    // this CTA's k-th cell, n_cells past its last
    auto cell_of = [&](uint32_t k) -> uint32_t {
        const uint32_t r = k * gridDim.x + ((lpt && (k & 1u)) ? gridDim.x - 1u - blockIdx.x : blockIdx.x);
        if (r >= n_cells) return n_cells;
        return lpt ? (uint32_t)lpt_seq[r / n_kv] * n_kv + r % n_kv : r;
    };

    if (warp == 4) {
        // ---- producer ----
        uint32_t g = 0;
        for (uint32_t k = 0, c = cell_of(0u); c < n_cells; c = cell_of(++k)) {
            const Cell x = cell_at(c);
            const int xk = (int)((x.kvh / kvh_div) * HD);
            const uint32_t* bt = block_tables + (size_t)x.slot * blocks_per_slot + x.B0;
            for (uint32_t t = 0; t < x.nt; ++t, ++g) {
                const uint32_t s = g % NS, ph = (g / NS) & 1u;
                bar_wait(&kv_empty[s], ph ^ 1u);
                const uint32_t nbc = min(8u, x.nblk - t * 8u);
                if (lane == 0) {
                    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                                 ::"r"((uint32_t)__cvta_generic_to_shared(&k_full[s])), "r"(nbc * 2u * 2048u));
                    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                                 ::"r"((uint32_t)__cvta_generic_to_shared(&v_full[s])), "r"(nbc * 2u * 2048u));
                }
                __syncwarp();
                if (lane < nbc * 2u) {
                    const uint32_t j = lane >> 1, p = lane & 1u;
                    const int y = (int)(__ldg(bt + t * 8u + j) * 16u);
                    unsigned char* st = ring + s * STG;
                    const uint32_t dk = (uint32_t)__cvta_generic_to_shared(st + p * PANEL + j * 2048u);
                    const uint32_t dv = (uint32_t)__cvta_generic_to_shared(st + 2u * PANEL + p * PANEL + j * 2048u);
                    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                                 " [%0], [%1, {%2, %3}], [%4];"
                                 ::"r"(dk), "l"(&tmk), "r"(xk + (int)(p * 128u)), "r"(y),
                                   "r"((uint32_t)__cvta_generic_to_shared(&k_full[s])) : "memory");
                    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                                 " [%0], [%1, {%2, %3}], [%4];"
                                 ::"r"(dv), "l"(&tmv), "r"(xk + (int)(p * 128u)), "r"(y),
                                   "r"((uint32_t)__cvta_generic_to_shared(&v_full[s])) : "memory");
                }
            }
        }
    } else if (warp == 5) {
        // ---- mma issuer ----
        const uint32_t tm = __shfl_sync(0xffffffffu, tmem, 0);
        const uint32_t id_s = (1u << 4) | ((8u >> 3) << 17) | ((128u >> 4) << 24);
        const uint32_t id_o = (1u << 4) | (1u << 15) | ((8u >> 3) << 17) | ((128u >> 4) << 24);
        const uint32_t ring16 = (uint32_t)__cvta_generic_to_shared(ring) >> 4;
        const uint32_t q16 = (uint32_t)__cvta_generic_to_shared(qimg) >> 4;
        const uint32_t p16 = (uint32_t)__cvta_generic_to_shared(pimg) >> 4;
        auto mma = [](uint32_t d, uint64_t a, uint64_t b, uint32_t id, uint32_t en) {
            asm volatile("{\n\t.reg .pred p, e;\n\telect.sync _|e, 0xffffffff;\n\t"
                         "setp.ne.b32 p, %4, 0;\n\t"
                         "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t}"
                         ::"r"(d), "l"(a), "l"(b), "r"(id), "r"(en));
        };
        auto commit = [](uint64_t* b) {
            asm volatile("{\n\t.reg .pred e;\n\telect.sync _|e, 0xffffffff;\n\t"
                         "@e tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n\t}"
                         ::"r"((uint32_t)__cvta_generic_to_shared(b)) : "memory");
        };
        // PV(g) of cell ordinal ci, tile t of it
        auto issue_pv = [&](uint32_t g, uint32_t ci, uint32_t t) {
            const uint32_t s = g % NS, ph = (g / NS) & 1u;
            bar_wait(&p_full[g & 1u], (g >> 1) & 1u);
            bar_wait(&v_full[s], ph);
            if (t == 0u && ci > 0u) bar_wait(&o_free, (ci - 1u) & 1u);   // the last cell's O was read
            asm volatile("tcgen05.fence::after_thread_sync;");
            const uint32_t v16 = ring16 + (s * STG + 2u * PANEL) / 16u;
            #pragma unroll
            for (uint32_t ct = 0; ct < 2u; ++ct)
                #pragma unroll
                for (uint32_t kb = 0; kb < TK / 32u; ++kb)
                    mma(tm + 16u + ct * 8u, pd_tc5_sdesc(v16 + ct * (PANEL / 16u) + kb * 256u),
                        pd_tc5_sdesc(p16 + (g & 1u) * 64u + kb * 2u), id_o, (t > 0u || kb > 0u) ? 1u : 0u);
            commit(&pv_done[g & 1u]);
            commit(&kv_empty[s]);
        };
        uint32_t g = 0, ci = 0;
        bool have_prev = false;
        uint32_t pg = 0, pci = 0, pt = 0;
        for (uint32_t k = 0, c = cell_of(0u); c < n_cells; c = cell_of(++k), ++ci) {
            const Cell x = cell_at(c);
            bar_wait(&q_ready[ci & 1u], (ci >> 1) & 1u);
            for (uint32_t t = 0; t < x.nt; ++t, ++g) {
                const uint32_t s = g % NS, ph = (g / NS) & 1u;
                bar_wait(&k_full[s], ph);
                bar_wait(&s_empty[g & 1u], ((g >> 1) & 1u) ^ 1u);
                asm volatile("tcgen05.fence::after_thread_sync;");
                const uint32_t k16 = ring16 + (s * STG) / 16u;
                const uint32_t qb = q16 + (ci & 1u) * 128u;
                #pragma unroll
                for (uint32_t kb = 0; kb < HD / 32u; ++kb)
                    mma(tm + (g & 1u) * 8u, pd_tc5_sdesc(k16 + (kb >> 2) * (PANEL / 16u) + (kb & 3u) * 2u),
                        pd_tc5_sdesc(qb + (kb >> 2) * 64u + (kb & 3u) * 2u), id_s, kb > 0u ? 1u : 0u);
                commit(&s_full[g & 1u]);
                if (have_prev) issue_pv(pg, pci, pt);
                have_prev = true; pg = g; pci = ci; pt = t;
            }
        }
        if (have_prev) issue_pv(pg, pci, pt);
    } else {
        // ---- softmax + Q build + epilogue (warps 0-3) ----
        constexpr float L2E = 1.4426950408889634f;
// The reference moves with every max increase (LAZY 0): with e4m3 weights each stale binade is
// precision lost at the bottom of the range - LAZY 2 measured 1.11e-3 rms vs 0 at 8.92e-4,
// which is tc5e's 8.95e-4 (bench/tc5d_probe.cu, 110 rows x 3-6k keys)
#ifndef PD_TC5D_LAZY
#define PD_TC5D_LAZY 0.0f
#endif
        constexpr float LAZY = PD_TC5D_LAZY;
        // P lands in e4m3, whose subnormals start at 2^-6: weights computed against the max land
        // as subnormals for any spread-out row (most of a long context) and the rounding bias
        // shows (a 1.2x row at 6k keys). tc5e scales the max weight to 448 (exp(s - m + ln 448));
        // a reference up to LAZY stale must still not overflow 448, so the bias here is
        // log2(448) - LAZY: the max weight sits at 448 / 2^LAZY .. 448.
        constexpr float PBIAS = 8.807355f - LAZY;
        const uint32_t tl = (warp * 32u) << 16;
        auto build_q = [&](uint32_t c, uint32_t par) {
            const Cell x = cell_at(c);
            const uint32_t r = tid >> 4, c16 = tid & 15u;   // head row, 16-byte chunk
            unsigned char tmp[16];
            if (r < G) {
                const float* src = q + ((size_t)x.b * n_heads + x.kvh * G + r) * HD + c16 * 16u;
                #pragma unroll
                for (uint32_t e = 0; e < 16u; ++e) tmp[e] = __nv_fp8_e4m3(__ldg(src + e) * scale).__x;
            } else {
                #pragma unroll
                for (uint32_t e = 0; e < 16u; ++e) tmp[e] = 0u;
            }
            *(uint4*)(qimg + par * 2048u + (c16 >> 3) * 1024u + r * 128u + (((c16 & 7u) ^ r) << 4)) =
                *(const uint4*)tmp;
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncwarp();
            if (lane == 0) arrive(&q_ready[par]);
        };
        // PV(g) is completion g/2 of pv_done[g & 1]; every wait is on the next one outstanding
        auto wait_pv = [&](uint32_t g) { bar_wait(&pv_done[g & 1u], (g >> 1) & 1u); };
        uint32_t g = 0, ci = 0, rp = 0;
        if (cell_of(0u) < n_cells) build_q(cell_of(0u), 0u);
        for (uint32_t k = 0, c = cell_of(0u); c < n_cells; c = cell_of(++k), ++ci) {
            const Cell x = cell_at(c);
            float m_ref[8], l_run[8];
            #pragma unroll
            for (uint32_t h = 0; h < 8u; ++h) { m_ref[h] = -INFINITY; l_run[h] = 0.0f; }
            for (uint32_t t = 0; t < x.nt; ++t, ++g) {
                bar_wait(&s_full[g & 1u], (g >> 1) & 1u);
                asm volatile("tcgen05.fence::after_thread_sync;");
                uint32_t r8[8];
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                             : "=r"(r8[0]), "=r"(r8[1]), "=r"(r8[2]), "=r"(r8[3]), "=r"(r8[4]), "=r"(r8[5]),
                               "=r"(r8[6]), "=r"(r8[7]) : "r"(tmem + tl + (g & 1u) * 8u));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                asm volatile("tcgen05.fence::before_thread_sync;");
                __syncwarp();
                if (lane == 0) arrive(&s_empty[g & 1u]);
                const uint32_t key = (x.B0 + t * 8u) * 16u + tid;
                const bool valid = key >= x.glo && key < x.ghi;
                float s2[G], mx[G];
                #pragma unroll
                for (uint32_t h = 0; h < G; ++h) {
                    s2[h] = valid ? __uint_as_float(r8[h]) * L2E : -INFINITY;
                    float v = s2[h];
                    #pragma unroll
                    for (uint32_t o = 16u; o > 0u; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
                    mx[h] = v;
                }
                if (lane == 0) {
                    #pragma unroll
                    for (uint32_t h = 0; h < G; ++h) red[rp][warp][h] = mx[h];
                }
                asm volatile("bar.sync 1, 128;" ::: "memory");
                bool resc = false;
                float alpha[G];
                #pragma unroll
                for (uint32_t h = 0; h < G; ++h) {
                    const float tmax = fmaxf(fmaxf(red[rp][0][h], red[rp][1][h]), fmaxf(red[rp][2][h], red[rp][3][h]));
                    alpha[h] = 1.0f;
                    if (tmax > m_ref[h] + LAZY) {
                        alpha[h] = exp2f(m_ref[h] - tmax);   // 0 on the cell's first tile
                        m_ref[h] = tmax;
                        resc = true;
                    }
                }
                rp ^= 1u;
                if (resc) {
                    #pragma unroll
                    for (uint32_t h = 0; h < G; ++h) l_run[h] *= alpha[h];
                    if (t > 0u) {
                        // O^T holds PV(0..t-1) under the old references: let the last one land
                        wait_pv(g - 1u);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        uint32_t o16[16];
                        asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                                     "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                                     : "=r"(o16[0]), "=r"(o16[1]), "=r"(o16[2]), "=r"(o16[3]), "=r"(o16[4]),
                                       "=r"(o16[5]), "=r"(o16[6]), "=r"(o16[7]), "=r"(o16[8]), "=r"(o16[9]),
                                       "=r"(o16[10]), "=r"(o16[11]), "=r"(o16[12]), "=r"(o16[13]),
                                       "=r"(o16[14]), "=r"(o16[15]) : "r"(tmem + tl + 16u));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        #pragma unroll
                        for (uint32_t ct = 0; ct < 2u; ++ct)
                            #pragma unroll
                            for (uint32_t h = 0; h < G; ++h)
                                o16[ct * 8u + h] = __float_as_uint(__uint_as_float(o16[ct * 8u + h]) * alpha[h]);
                        asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], "
                                     "{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};"
                                     ::"r"(tmem + tl + 16u), "r"(o16[0]), "r"(o16[1]), "r"(o16[2]), "r"(o16[3]),
                                       "r"(o16[4]), "r"(o16[5]), "r"(o16[6]), "r"(o16[7]), "r"(o16[8]),
                                       "r"(o16[9]), "r"(o16[10]), "r"(o16[11]), "r"(o16[12]), "r"(o16[13]),
                                       "r"(o16[14]), "r"(o16[15]));
                        asm volatile("tcgen05.wait::st.sync.aligned;");
                    }
                }
                // the tail tile: V rows past the cell's last key would be multiplied by P = 0, but
                // the pool rows behind them (the rest of the last block, earlier tiles' stale rows)
                // can hold e4m3 NaN patterns and 0 x NaN poisons O - zero them once V landed
                // (tc5e zeroes the same tail)
                const uint32_t vkeys = x.ghi - (x.B0 + t * 8u) * 16u;
                if (vkeys < TK) {
                    bar_wait(&v_full[g % NS], (g / NS) & 1u);
                    unsigned char* vb = ring + (g % NS) * STG + 2u * PANEL;
                    for (uint32_t i = tid; i < (TK - vkeys) * 16u; i += 128u) {
                        const uint32_t r = vkeys + (i >> 4), c = i & 15u;   // row, 16-B chunk of 2 panels
                        *(uint4*)(vb + (c >> 3) * PANEL + r * 128u + (c & 7u) * 16u) = make_uint4(0u, 0u, 0u, 0u);
                    }
                }
                // P image g & 1 was last read by PV(g - 2)
                if (g >= 2u) wait_pv(g - 2u);
                unsigned char* pi = pimg + (g & 1u) * 1024u;
                const uint32_t kc = tid >> 4, kl = tid & 15u;
                #pragma unroll
                for (uint32_t h = 0; h < G; ++h) {
                    const float p = valid ? exp2f(s2[h] - m_ref[h] + PBIAS) : 0.0f;
                    const __nv_fp8_e4m3 pq(p);
#if defined(PD_TC5D_LROUND)
                    l_run[h] += (float)pq;   // the weight the PV mma multiplies
#else
                    l_run[h] += p;           // tc5e's: the exact weight
#endif
                    pi[h * 128u + ((kc ^ h) << 4) + kl] = pq.__x;
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                __syncwarp();
                if (lane == 0) arrive(&p_full[g & 1u]);
            }
            // the next cell's Q (the issuer's S for it overlaps this cell's last PV + epilogue)
            const uint32_t cn = cell_of(k + 1u);
            if (cn < n_cells) build_q(cn, (ci + 1u) & 1u);
            // epilogue: l per head over the 128 key threads, then O / l (thread = hd lane)
            float lsum[G];
            #pragma unroll
            for (uint32_t h = 0; h < G; ++h) {
                float v = l_run[h];
                #pragma unroll
                for (uint32_t o = 16u; o > 0u; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
                lsum[h] = v;
            }
            if (lane == 0) {
                #pragma unroll
                for (uint32_t h = 0; h < G; ++h) red[rp][warp][h] = lsum[h];
            }
            asm volatile("bar.sync 1, 128;" ::: "memory");
            float inv[G];
            #pragma unroll
            for (uint32_t h = 0; h < G; ++h) {
                const float L = (red[rp][0][h] + red[rp][1][h]) + (red[rp][2][h] + red[rp][3][h]);
                inv[h] = L > 0.0f ? 1.0f / L : 0.0f;
            }
            rp ^= 1u;
            wait_pv(g - 1u);
            asm volatile("tcgen05.fence::after_thread_sync;");
            uint32_t o16[16];
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                         "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                         : "=r"(o16[0]), "=r"(o16[1]), "=r"(o16[2]), "=r"(o16[3]), "=r"(o16[4]),
                           "=r"(o16[5]), "=r"(o16[6]), "=r"(o16[7]), "=r"(o16[8]), "=r"(o16[9]),
                           "=r"(o16[10]), "=r"(o16[11]), "=r"(o16[12]), "=r"(o16[13]),
                           "=r"(o16[14]), "=r"(o16[15]) : "r"(tmem + tl + 16u));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            asm volatile("tcgen05.fence::before_thread_sync;");
            __syncwarp();
            if (lane == 0) arrive(&o_free);
            float* ob = out + ((size_t)x.b * n_heads + x.kvh * G) * HD;
            #pragma unroll
            for (uint32_t ct = 0; ct < 2u; ++ct)
                #pragma unroll
                for (uint32_t h = 0; h < G; ++h)
                    ob[(size_t)h * HD + ct * 128u + tid] = __uint_as_float(o16[ct * 8u + h]) * inv[h];
        }
    }
    asm volatile("tcgen05.fence::before_thread_sync;");
    __syncthreads();
    if (warp == 5)
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;" ::"r"(tmem));
#else
    (void)tmk; (void)tmv; (void)q; (void)out; (void)positions; (void)slots; (void)block_tables;
    (void)blocks_per_slot; (void)swa_window; (void)scale; (void)n_kv; (void)n_seq; (void)kvh_div;
#endif
}
