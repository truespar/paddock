// attn/prefill_tc5.cuh - hd256 fp8-KV prefill attention with both GEMMs on tcgen05 (sm_100a), the
// varlen packed form (one launch per layer, every prefill span of the tick). Textually-included
// segment of the single pack translation unit; follows prefill_pf5.cuh (pd_pf_floor / pd_pf_wp
// from prefill.cuh, pd_attn_tmap_kv_f8s from f32_qkv.cuh, pd_vtc5_ld32 from vision_tc5.cuh).
//
// Why. pf7rp (the varlen arm qwen3.8's full-attention layers ran) is FA2 on mma.sync with a
// fp8->f16 repack pane: ~135 TFLOP/s on B200, ~20 ms of GPU a page at the recover load. The
// tensor pipe it never touched does 4.5 PF in e4m3.
//
// Numeric class: e4m3 Q (scaled), K, V and P, f32 accumulate - tc5d / tc5e's decode class (and
// vLLM's context kernel's), not pf7rp's f16 one: a serve-gated class change. P carries tc5e's 448
// bias (log2 448 added before the round; the e4m3 subnormal floor otherwise bends spread-out
// rows) and l sums the unrounded weights; the reference max moves with every increase (an exact
// online softmax - with e4m3 P a lazy one costs binades). Masks are pf7rp's: a row sees
// keys [pd_pf_floor(win pos), pos].
//
// Shape: one CTA = one 128-row tile of (token, head) rows of one span for one kv head (items
// stride 4: q_row0, span_rows, tile_row0, slot - the pf7 VL contract at 128-row tiles), 192
// threads:
//   warp 4   producer: K and V 128-key tiles (8 blocks x 2 hd panels each, 2 KB TMA boxes through
//            the paged maps) into a 2-deep ring.
//   warp 5   issuer (warp-converged): S(t) = Q K^T (M 128 rows, N 128 keys, 8 K-steps) into one of
//            two tmem S buffers, then O += P(t-1) V(t-1) (V read transposed, bit 16; N 128 a panel).
//   warps 0-3 thread = row: Q image, softmax, P image (two, by tile parity), O rescale on a max
//            move, the epilogue.

#define PD_PFT_TK 128u
#define PD_PFT_NS 2u
#define PD_PFT_PANEL 16384u                          // 128 rows x 128 B
#define PD_PFT_STAGE (4u * PD_PFT_PANEL)             // K p0 | K p1 | V p0 | V p1
// Q rides as TWO e4m3 images (hi + lo, below), so 2 x 2 panels
#define PD_PFT_SMEM (4u * PD_PFT_PANEL + PD_PFT_NS * PD_PFT_STAGE + 2u * PD_PFT_PANEL + 1024u)

template <uint32_t G>
__global__ void __launch_bounds__(192, 1) pd_attn_prefill_tc5_kernel(
    const __grid_constant__ PdTmap tmk, const __grid_constant__ PdTmap tmv,
    const float* __restrict__ q, const float* __restrict__ sinks, float* __restrict__ out,
    const unsigned int* __restrict__ positions, const uint32_t* __restrict__ block_tables,
    uint32_t blocks_per_slot, uint32_t n_heads, uint32_t swa_window, float scale,
    const uint32_t* __restrict__ items) {
#if PD_ATTN_TMA_OK && PD_TC5_OK
    constexpr uint32_t HD = 256u, TK = PD_PFT_TK, NS = PD_PFT_NS, PANEL = PD_PFT_PANEL,
                       STG = PD_PFT_STAGE;
    constexpr float L2E = 1.4426950408889634f;
#ifndef PD_PFT_LAZY
#define PD_PFT_LAZY 0.0f
#endif
    constexpr float LAZY = PD_PFT_LAZY;             // reference staleness allowed (log2)
    constexpr float PBIAS = 8.807355f - LAZY;       // log2(448) - LAZY: the max weight stays <= 448
    extern __shared__ __align__(1024) unsigned char pd_pft_raw[];
    unsigned char* sh = pd_pft_raw
        + ((1024u - ((uint32_t)__cvta_generic_to_shared(pd_pft_raw) & 1023u)) & 1023u);
    unsigned char* qimg = sh;                        // hi: 2 panels, then lo: 2 panels
    unsigned char* ring = qimg + 4u * PANEL;         // NS x STG
    unsigned char* pimg = ring + NS * STG;           // 2 x PANEL (tile parity)
    __shared__ __align__(8) uint64_t k_full[NS], v_full[NS], kv_empty[NS], s_full[2], s_empty[2],
        p_full[2], pv_done[2], q_ready;
    __shared__ uint32_t s_hi, s_lo, tmem_slot[1];

    const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
    const uint32_t kvh = blockIdx.x;
    const uint32_t* it = items + (size_t)blockIdx.y * 4u;
    asm volatile("griddepcontrol.wait;" ::: "memory");
    const uint32_t qr0 = it[0], span_rows = it[1], row0 = it[2], slot = it[3];
    const uint32_t Rtot = span_rows * G;
    if (row0 >= Rtot) return;   // uniform: nothing allocated yet

    // the CTA's key range: rows' positions and window floors (pf7rp's)
    const uint32_t r = row0 + tid;                   // softmax threads' row
    const bool rv = tid < 128u && r < Rtot;
    const uint32_t tok = rv ? qr0 + r / G : 0u, head = rv ? kvh * G + r % G : 0u;
    const uint32_t pos = rv ? positions[tok] : 0u;
    const uint32_t flo = rv ? pd_pf_floor(positions[tok], swa_window) : 0xFFFFFFFFu;
    if (tid == 0) {
        s_hi = 0u; s_lo = 0xFFFFFFFFu;
        auto init = [](uint64_t* b, uint32_t n) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(b)), "r"(n));
        };
        for (uint32_t i = 0; i < NS; ++i) { init(&k_full[i], 1); init(&v_full[i], 1); init(&kv_empty[i], 1); }
        for (uint32_t i = 0; i < 2u; ++i) {
            init(&s_full[i], 1); init(&s_empty[i], 4); init(&p_full[i], 4); init(&pv_done[i], 1);
        }
        init(&q_ready, 4);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    if (warp == 5)
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;"
                     ::"r"((uint32_t)__cvta_generic_to_shared(tmem_slot)));
    asm volatile("tcgen05.fence::before_thread_sync;");
    __syncthreads();
    if (rv) { atomicMax(&s_hi, pos + 1u); atomicMin(&s_lo, flo); }
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");
    const uint32_t tmem = tmem_slot[0];   // S: cols 0..255 (2 x 128); O: 256..511 (2 panels)
    const uint32_t hi = s_hi, lo_t = s_lo / TK * TK;
    const uint32_t nt = (hi - lo_t + TK - 1u) / TK;
    asm volatile("griddepcontrol.launch_dependents;");

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

    if (warp == 4) {
        // ---- producer ----
        const uint32_t* bt = block_tables + (size_t)slot * blocks_per_slot;
        for (uint32_t t = 0; t < nt; ++t) {
            const uint32_t s = t % NS, ph = (t / NS) & 1u;
            bar_wait(&kv_empty[s], ph ^ 1u);
            const uint32_t k0 = lo_t + t * TK;
            const uint32_t nb = min(8u, (hi - k0 + 15u) >> 4);
            if (lane == 0) {
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                             ::"r"((uint32_t)__cvta_generic_to_shared(&k_full[s])), "r"(nb * 2u * 2048u));
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                             ::"r"((uint32_t)__cvta_generic_to_shared(&v_full[s])), "r"(nb * 2u * 2048u));
            }
            __syncwarp();
            if (lane < nb * 2u) {
                const uint32_t j = lane >> 1, p = lane & 1u;
                const int y = (int)(__ldg(bt + (k0 >> 4) + j) * 16u);
                const int x = (int)(kvh * HD + p * 128u);
                unsigned char* st = ring + s * STG;
                asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                             " [%0], [%1, {%2, %3}], [%4];"
                             ::"r"((uint32_t)__cvta_generic_to_shared(st + p * PANEL + j * 2048u)), "l"(&tmk),
                               "r"(x), "r"(y), "r"((uint32_t)__cvta_generic_to_shared(&k_full[s])) : "memory");
                asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                             " [%0], [%1, {%2, %3}], [%4];"
                             ::"r"((uint32_t)__cvta_generic_to_shared(st + 2u * PANEL + p * PANEL + j * 2048u)),
                               "l"(&tmv), "r"(x), "r"(y), "r"((uint32_t)__cvta_generic_to_shared(&v_full[s])) : "memory");
            }
        }
    } else if (warp == 5) {
        // ---- mma issuer ----
        const uint32_t tm = __shfl_sync(0xffffffffu, tmem, 0);
        const uint32_t id_s = (1u << 4) | ((TK >> 3) << 17) | ((128u >> 4) << 24);
        const uint32_t id_o = (1u << 4) | (1u << 16) | ((128u >> 3) << 17) | ((128u >> 4) << 24);
        const uint32_t q16 = (uint32_t)__cvta_generic_to_shared(qimg) >> 4;
        const uint32_t ring16 = (uint32_t)__cvta_generic_to_shared(ring) >> 4;
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
        auto issue_pv = [&](uint32_t t) {
            const uint32_t s = t % NS, ph = (t / NS) & 1u;
            bar_wait(&p_full[t & 1u], (t >> 1) & 1u);
            bar_wait(&v_full[s], ph);
            asm volatile("tcgen05.fence::after_thread_sync;");
            const uint32_t v16 = ring16 + (s * STG + 2u * PANEL) / 16u;
            #pragma unroll
            for (uint32_t pn = 0; pn < 2u; ++pn)
                #pragma unroll
                for (uint32_t kb = 0; kb < TK / 32u; ++kb)
                    mma(tm + 256u + pn * 128u, pd_tc5_sdesc(p16 + (t & 1u) * (PANEL / 16u) + kb * 2u),
                        pd_tc5_sdesc(v16 + pn * (PANEL / 16u) + kb * 256u), id_o, (t > 0u || kb > 0u) ? 1u : 0u);
            commit(&pv_done[t & 1u]);
            commit(&kv_empty[s]);
        };
        bar_wait(&q_ready, 0u);
        for (uint32_t t = 0; t < nt; ++t) {
            const uint32_t s = t % NS, ph = (t / NS) & 1u;
            bar_wait(&k_full[s], ph);
            bar_wait(&s_empty[t & 1u], ((t >> 1) & 1u) ^ 1u);
            asm volatile("tcgen05.fence::after_thread_sync;");
            const uint32_t k16 = ring16 + (s * STG) / 16u;
            #pragma unroll
            for (uint32_t hl = 0; hl < 2u; ++hl)
                #pragma unroll
                for (uint32_t kb = 0; kb < HD / 32u; ++kb)
                    mma(tm + (t & 1u) * 128u,
                        pd_tc5_sdesc(q16 + hl * (2u * PANEL / 16u) + (kb >> 2) * (PANEL / 16u) + (kb & 3u) * 2u),
                        pd_tc5_sdesc(k16 + (kb >> 2) * (PANEL / 16u) + (kb & 3u) * 2u), id_s,
                        (hl > 0u || kb > 0u) ? 1u : 0u);
            commit(&s_full[t & 1u]);
            if (t > 0u) issue_pv(t - 1u);
        }
        if (nt) issue_pv(nt - 1u);
    } else {
        // ---- softmax (warps 0-3; thread = row) ----
        const uint32_t tl = (warp * 32u) << 16;
        // Q images. One e4m3 copy of the scaled row cost the truth gate 1054/1068 (vs 1064-1068):
        // 3 mantissa bits, and scaled rows sit partly in e4m3's subnormals. So: a per-row power of
        // two puts the row's amax near 224 (exact to undo - the softmax multiplies S back), and
        // the row rides as hi = e4m3(v) plus lo = e4m3(v - hi), two mmas into one accumulator
        // (~7 mantissa bits; K and V stay the pool's e4m3, no repack).
        float qinv = 1.0f;   // S multiplier undoing this row's power of two
        {
            const float* qs = q + ((size_t)tok * n_heads + head) * HD;
            float amax = 0.0f;
            if (rv) {
                #pragma unroll 8
                for (uint32_t i = 0; i < HD; i += 4u) {
                    const float4 v = __ldg((const float4*)(qs + i));
                    amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
                }
            }
            amax *= scale;
            int ex = 0;
            if (amax > 0.0f) { frexpf(224.0f / amax, &ex); ex -= 1; }   // amax * 2^ex in (112, 224]
            const float qmul = ldexpf(scale, ex);
            qinv = ldexpf(1.0f, -ex);
            #pragma unroll 2
            for (uint32_t c16 = 0; c16 < 16u; ++c16) {
                uint32_t wh[4] = {0u, 0u, 0u, 0u}, wl[4] = {0u, 0u, 0u, 0u};
                if (rv) {
                    #pragma unroll
                    for (uint32_t e4 = 0; e4 < 4u; ++e4) {
                        const float4 v4 = __ldg((const float4*)(qs + c16 * 16u + e4 * 4u));
                        const float v[4] = {v4.x * qmul, v4.y * qmul, v4.z * qmul, v4.w * qmul};
                        #pragma unroll
                        for (uint32_t e = 0; e < 4u; ++e) {
                            const __nv_fp8_e4m3 h(v[e]);
                            const __nv_fp8_e4m3 l(v[e] - (float)h);
                            wh[e4] |= (uint32_t)h.__x << (8u * e);
                            wl[e4] |= (uint32_t)l.__x << (8u * e);
                        }
                    }
                }
                const uint32_t off = (c16 >> 3) * PANEL + tid * 128u + (((c16 & 7u) ^ (tid & 7u)) << 4);
                *(uint4*)(qimg + off) = make_uint4(wh[0], wh[1], wh[2], wh[3]);
                *(uint4*)(qimg + 2u * PANEL + off) = make_uint4(wl[0], wl[1], wl[2], wl[3]);
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncwarp();
            if (lane == 0) arrive(&q_ready);
        }
        auto wait_pv = [&](uint32_t t) { bar_wait(&pv_done[t & 1u], (t >> 1) & 1u); };
        float m_ref = -INFINITY, l_run = 0.0f;
        const float sl2e = qinv * L2E;
        for (uint32_t t = 0; t < nt; ++t) {
            const uint32_t k0 = lo_t + t * TK;
            bar_wait(&s_full[t & 1u], (t >> 1) & 1u);
            asm volatile("tcgen05.fence::after_thread_sync;");
            uint32_t sr[128];
            #pragma unroll
            for (uint32_t c = 0; c < 4u; ++c) pd_vtc5_ld32(tmem + tl + (t & 1u) * 128u + c * 32u, sr + c * 32u);
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            asm volatile("tcgen05.fence::before_thread_sync;");
            __syncwarp();
            if (lane == 0) arrive(&s_empty[t & 1u]);
            // visible keys [flo, pos] (pf7rp's); rows past the span see none
            const int vlo = rv ? (int)flo - (int)k0 : 1 << 30, vhi = rv ? (int)pos - (int)k0 : -1;
            float m8[8];
            #pragma unroll
            for (uint32_t i = 0; i < 8u; ++i) m8[i] = -INFINITY;
            if (__all_sync(0xffffffffu, vlo <= 0 && vhi >= (int)TK - 1)) {
                // every key of the tile visible to all 32 rows (all but the diagonal and window
                // edges): the same scale and max, without the per-key mask test, the scale two
                // keys an instruction (FMUL2: each lane the scalar multiply's rounding)
                const float2 sc2 = make_float2(sl2e, sl2e);
                #pragma unroll
                for (uint32_t i = 0; i < 128u; i += 2u) {
                    const float2 s2 = __fmul2_rn(make_float2(__uint_as_float(sr[i]), __uint_as_float(sr[i + 1u])), sc2);
                    sr[i] = __float_as_uint(s2.x);
                    sr[i + 1u] = __float_as_uint(s2.y);
                    m8[i & 7u] = fmaxf(m8[i & 7u], s2.x);
                    m8[(i + 1u) & 7u] = fmaxf(m8[(i + 1u) & 7u], s2.y);
                }
            } else {
                #pragma unroll
                for (uint32_t i = 0; i < 128u; ++i) {
                    const bool ok = (int)i >= vlo && (int)i <= vhi;
                    const float s2 = ok ? __uint_as_float(sr[i]) * sl2e : -INFINITY;
                    sr[i] = __float_as_uint(s2);
                    m8[i & 7u] = fmaxf(m8[i & 7u], s2);
                }
            }
            const float mt = fmaxf(fmaxf(fmaxf(m8[0], m8[1]), fmaxf(m8[2], m8[3])),
                                   fmaxf(fmaxf(m8[4], m8[5]), fmaxf(m8[6], m8[7])));
            const bool need = mt > m_ref + LAZY;
            const float alpha = need ? exp2f(m_ref - mt) : 1.0f;   // 0 when m_ref is -inf
            if (need) { l_run *= alpha; m_ref = mt; }
// (Measured: moving this rescale after the P(t) weights so they overlap PV(t-1) - the value
// order unchanged, bit-identical - ran 292 -> 302 us at 1770 rows: the PV wait is not the path.)
#if defined(PD_PFT_NORESCALE)   // timing experiment only: skips the O round trip (wrong numbers)
            if (false)
#endif
            if (__any_sync(0xffffffffu, need) && t > 0u) {
                // O holds PV(0..t-1) under the old references: the last one must land first
                wait_pv(t - 1u);
                asm volatile("tcgen05.fence::after_thread_sync;");
                #pragma unroll
                for (uint32_t c = 0; c < 8u; ++c) {
                    uint32_t o32[32];
                    pd_vtc5_ld32(tmem + tl + 256u + c * 32u, o32);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (uint32_t j = 0; j < 32u; j += 2u) {
                        const float2 o2 = __fmul2_rn(make_float2(__uint_as_float(o32[j]), __uint_as_float(o32[j + 1u])),
                                                     make_float2(alpha, alpha));
                        o32[j] = __float_as_uint(o2.x);
                        o32[j + 1u] = __float_as_uint(o2.y);
                    }
                    pd_vtc5_st32(tmem + tl + 256u + c * 32u, o32);
                }
                asm volatile("tcgen05.wait::st.sync.aligned;");
            }
            // the tile's V rows past the CTA's last key would meet P = 0, but they are unwritten
            // pool rows or earlier tiles' (0 x NaN poisons O): zero them once V landed
            const uint32_t vk = hi - k0;
            if (vk < TK) {
                bar_wait(&v_full[t % NS], (t / NS) & 1u);
                unsigned char* vb = ring + (t % NS) * STG + 2u * PANEL;
                for (uint32_t i = tid; i < (TK - vk) * 16u; i += 128u) {
                    const uint32_t rr = vk + (i >> 4), c = i & 15u;
                    *(uint4*)(vb + (c >> 3) * PANEL + rr * 128u + (c & 7u) * 16u) = make_uint4(0u, 0u, 0u, 0u);
                }
            }
            // P image t & 1 was last read by PV(t - 2)
            if (t >= 2u) wait_pv(t - 2u);
            // the weights: one MUFU ex2 each (the e4m3 round keeps 3 mantissa bits, so ex2.approx's
            // ~2 ulp is far below it; masked keys are -inf -> 0) and one cvt per pair - the
            // accurate exp2f and per-value converts held the softmax at ~4.6 us a tile
            unsigned char* pi = pimg + (t & 1u) * PANEL + tid * 128u;
            const float off = PBIAS - m_ref;
            // the exponent adds and the four sum chains two lanes an instruction (FADD2, each lane
            // the scalar add's rounding and order): sum2[0] = chains 0 / 1, sum2[1] = 2 / 3
            const float2 off2 = make_float2(off, off);
            float2 sum2[2] = {make_float2(0.0f, 0.0f), make_float2(0.0f, 0.0f)};
            #pragma unroll
            for (uint32_t c = 0; c < 8u; ++c) {
                uint32_t w[4];
                #pragma unroll
                for (uint32_t e4 = 0; e4 < 4u; ++e4) {
                    float p[4];
                    #pragma unroll
                    for (uint32_t h = 0; h < 2u; ++h) {
                        const uint32_t k = c * 16u + e4 * 4u + 2u * h;
                        const float2 a = __fadd2_rn(make_float2(__uint_as_float(sr[k]), __uint_as_float(sr[k + 1u])), off2);
                        p[2u * h] = pd_vtc5_ex2(a.x);
                        p[2u * h + 1u] = pd_vtc5_ex2(a.y);
                        sum2[h] = __fadd2_rn(sum2[h], make_float2(p[2u * h], p[2u * h + 1u]));
                    }
                    uint16_t lo2, hi2;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(lo2) : "f"(p[1]), "f"(p[0]));
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(hi2) : "f"(p[3]), "f"(p[2]));
                    w[e4] = (uint32_t)lo2 | ((uint32_t)hi2 << 16);
                }
                *(uint4*)(pi + ((c ^ (tid & 7u)) << 4)) = make_uint4(w[0], w[1], w[2], w[3]);
            }
            l_run += (sum2[0].x + sum2[0].y) + (sum2[1].x + sum2[1].y);
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            __syncwarp();
            if (lane == 0) arrive(&p_full[t & 1u]);
        }
        // epilogue: the sink merge (pf7rp's), then O / l, 1 KB a row
        if (nt) wait_pv(nt - 1u);
        asm volatile("tcgen05.fence::after_thread_sync;");
        // (qwen's sink plane is -inf: a no-op; gpt-oss-class sinks merge as pf7rp's do)
        float fac = 1.0f;
        if (rv && sinks != nullptr) {
            const float sk = sinks[head] * L2E;
            if (sk > -INFINITY) {
                const float M = fmaxf(m_ref, sk);
                fac = exp2f(m_ref - M);
                l_run = l_run * fac + exp2f(sk - M + PBIAS);
            }
        }
        const float inv = l_run > 0.0f ? fac / l_run : 0.0f;
        float* orow = out + ((size_t)tok * n_heads + head) * HD;
        #pragma unroll
        for (uint32_t c = 0; c < 8u; ++c) {
            uint32_t o32[32];
            pd_vtc5_ld32(tmem + tl + 256u + c * 32u, o32);
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            if (rv) {
                #pragma unroll
                for (uint32_t j = 0; j < 32u; j += 4u)
                    *(float4*)(orow + c * 32u + j) = make_float4(__uint_as_float(o32[j]) * inv, __uint_as_float(o32[j + 1u]) * inv,
                                                                 __uint_as_float(o32[j + 2u]) * inv, __uint_as_float(o32[j + 3u]) * inv);
            }
        }
    }
    asm volatile("tcgen05.fence::before_thread_sync;");
    __syncthreads();
    if (warp == 5)
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" ::"r"(tmem));
#else
    (void)tmk; (void)tmv; (void)q; (void)sinks; (void)out; (void)positions; (void)block_tables;
    (void)blocks_per_slot; (void)n_heads; (void)swa_window; (void)scale; (void)items;
#endif
}

// 872: the varlen packed prefill (pd_attn_prefill_f16_paged_vl's contract) at 128-row tiles on
// tcgen05: items stride 4 (q_row0, span_rows, tile_row0, slot) with tile_row0 stepping 128.
// hd 256, fp8 pools, G 6; cc 10.0 only. Kill: PADDOCK_NO_PF_TC5 (the engine then builds 64-row
// items for slot 322).
PD_EXPORT
int pd_attn_prefill_tc5_vl(const void* q, const void* pool_k, const void* pool_v, const void* sinks,
                           void* out, const void* positions, const void* items, uint32_t n_tiles,
                           const void* block_tables, uint32_t blocks_per_slot, uint32_t n_heads,
                           uint32_t n_kv_heads, uint32_t head_dim, uint32_t kv_dim, uint32_t swa_window,
                           float scale, uint32_t kv_dtype, void* stream) {
#if !defined(PD_TC5_HOST)
    (void)q; (void)pool_k; (void)pool_v; (void)sinks; (void)out; (void)positions; (void)items;
    (void)n_tiles; (void)block_tables; (void)blocks_per_slot; (void)n_heads; (void)n_kv_heads;
    (void)head_dim; (void)kv_dim; (void)swa_window; (void)scale; (void)kv_dtype; (void)stream;
    return cudaErrorNotSupported;
#else
    if (n_tiles == 0) return 0;
    static int ok = -1;
    if (ok < 0) {
        int dev = 0, ccM = 0, ccm = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&ccM, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&ccm, cudaDevAttrComputeCapabilityMinor, dev);
        ok = (ccM == 10 && ccm == 0 && pd_tmap_encode() != nullptr
              && cudaFuncSetAttribute((const void*)pd_attn_prefill_tc5_kernel<6u>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      (int)PD_PFT_SMEM) == cudaSuccess) ? 1 : 0;
    }
    if (!ok || kv_dtype != PD_KV_FP8_E4M3 || head_dim != 256u || n_kv_heads == 0
        || n_heads != 6u * n_kv_heads)
        return cudaErrorNotSupported;
    struct PdTmEntP { const void* p; uint32_t kd; CUtensorMap m; };
    static PdTmEntP tpc[128];
    static uint32_t tpn = 0;
    auto get_tm = [&](const void* base) -> const CUtensorMap* {
        for (uint32_t i = 0; i < tpn; ++i)
            if (tpc[i].p == base && tpc[i].kd == kv_dim) return &tpc[i].m;
        if (tpn >= 128u) tpn = 0;
        if (!pd_attn_tmap_kv_f8s(&tpc[tpn].m, base, kv_dim)) return nullptr;
        tpc[tpn].p = base; tpc[tpn].kd = kv_dim;
        return &tpc[tpn++].m;
    };
    const CUtensorMap* tk = get_tm(pool_k);
    const CUtensorMap* tv = tk ? get_tm(pool_v) : nullptr;
    if (!tk || !tv) return cudaErrorNotSupported;
    static bool wit = false;
    if (!wit) {
        ::fprintf(stderr, "[pf-tc5] ENGAGED (tcgen05 varlen prefill, 128-row tiles)\n");
        wit = true;
    }
    pd_pdl_go(pd_attn_prefill_tc5_kernel<6u>, dim3(n_kv_heads, n_tiles), dim3(192), PD_PFT_SMEM,
              (cudaStream_t)stream, *tk, *tv, (const float*)q, (const float*)sinks, (float*)out,
              (const unsigned int*)positions, (const uint32_t*)block_tables, blocks_per_slot, n_heads,
              swa_window, scale, (const uint32_t*)items);
    return pd_launch_status();
#endif
}
