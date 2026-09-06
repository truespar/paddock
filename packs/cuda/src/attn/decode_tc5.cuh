// tcgen05/TMEM decode attention: the R2d/v6 "two-team ping-pong" kernel
// - 12.17us c32 graph against v9q's 17.86, bit-deterministic at every tick
// depth tested (3-21), fp64-truth error at-or-below the v9q class on both
// probe fills.
// FINAL-output contract (fused_gqa16 precedent): batch-major out rows,
// no partials, no out_ml, no combine - the engine skips the combine when
// this export accepts. Shape gate: fp8-e4m3 paged KV, head_dim 256,
// group 2, swa_window > 0 (bounds the tick table), rc -2 otherwise, rc
// -3 over the smem opt-in. Election: PADDOCK_ATTN_TC5=1 (engine side).
// It was developed in an out-of-tree probe that includes pack.cu and
// benchmarks this kernel - measure there, not here.
__device__ unsigned long long tc5d_prof[16];
__device__ unsigned long long tc5d_stuck[4];

template <uint32_t HD, uint32_t G, bool PROF = false>
__global__ void __launch_bounds__(256) pd_attn_decode_tc5e_kernel(
    const __grid_constant__ PdTmap tmk,
    const __grid_constant__ PdTmap tmv,
    const float* __restrict__ q, float* __restrict__ out_o,
    const unsigned int* __restrict__ positions,
    const unsigned int* __restrict__ slots,
    const uint32_t* __restrict__ block_tables, uint32_t blocks_per_slot,
    uint32_t swa_window, float scale, uint32_t n_kv, uint32_t n_seq,
    uint32_t cpc, uint32_t dbg = 0u, uint32_t kvh_div = 1u) {
#if PD_ATTN_TMA_OK && PD_TC5_OK
    PD_PDL_ARM();   // cascade: full predecessor completion before any
                    // gmem read (positions/tables/q/KV TMA)
    // ---- v6 "two-team ping-pong": 8 warps; softmax OWNERSHIP of tick t
    // belongs to team T = t&1 (warps 4T..4T+3; TMEM quadrant = warp%4).
    // One global tick stream; adjacent ticks run on alternate teams so each
    // team's chain latencies hide under the other team's compute.
    // Wedge fabric (all ld||issue exclusion by CONSTRUCTION):
    //   s_bsp[2] count-3 "phase" mbarriers: tick t's three issuers (S(t+1)
    //     on tid 128T; PV(t) ct-halves on tid 128T+32/64) commit to
    //     s_bsp[1-T]; the phase fires only when all three chains EXECUTED,
    //     so the next team's softmax lds (gated on it) can never overlap
    //     any tick-t issue window (bs-gating theorem).
    //   s_ldc[2] count-128 "ld-clear": each team arrives at its tick-body
    //     end (after its last possible ld: softmax/rescale/epilogue); the
    //     other team's issuers wait it before issuing - closes the reverse
    //     race (stale epilogue lds into a fresh issue window).
    //   s_bpe2[2] count-2: cell-end epilogue gate (owner's PV commits).
    // Slots become team-constant: team T's V slot = T, its S-issue reads
    // K slot 1-T; P slot = T. Cell state (mr/l_run) lives in smem by cell
    // parity so multi-chunk cells can alternate teams (ordering: state
    // stores precede the team-bar; issuer commits release; phase-wait
    // acquires).
    constexpr uint32_t NP = HD / 128u;
    // chunk keys scale inversely with HD: slab bytes (NP*CPB = CH*HD) and
    // the TMA request count per stage (BPC*NP = 24) are conserved across
    // both geometries, so smem and the staging schedule are shape-invariant.
    constexpr uint32_t CH = HD == 256u ? 192u : 96u;
    constexpr uint32_t BPC = CH / 16u;    // KV blocks per chunk (12 / 6)
    constexpr uint32_t CPB = CH * 128u;   // bytes per partition slab
    constexpr uint32_t MAXT = 128u, MAXC = 16u;
    constexpr uint32_t QSZ = 8u * HD;     // per-cell Q slab (8-row M tile)
    // TMEM columns: two S slots (one per team; 2 halves only when CH>128)
    // then two O parity regions of NP*8 cols each.
    constexpr uint32_t SCOLS = CH > 128u ? 16u : 8u;
    constexpr uint32_t OB0 = 2u * SCOLS;
    constexpr uint32_t TMCOLS = (OB0 + 2u * NP * 8u) > 64u ? 128u : 64u;
    // <256,2> gemma SWA, <512,8> gemma GLB, <256,6> qwen3.8 full-attn
    // (24q/4kv/hd256). G only ever indexes the 8-row M tile the S and O
    // MMAs already emit (id_s/id_o encode M=8 for every instantiation), so
    // a group of 6 rides the same tmem, the same P image and the same
    // reductions with two padding rows nothing reads.
    static_assert((HD == 256u && (G == 2u || G == 6u))
                      || (G == 8u && HD == 512u),
                  "shaped for gemma-31b <256,2>/<512,8> and qwen3.8 <256,6>");
    unsigned long long tp[15] = {};
    unsigned long long tk0c = PROF ? clock64() : 0ull, tcur = tk0c;
    auto stamp = [&](int idx) {
        if (PROF) {
            const unsigned long long n = clock64();
            tp[idx] += n - tcur; tcur = n;
        }
    };
    const uint32_t n_cells = n_seq * n_kv;
    const uint32_t cell0 = blockIdx.x * cpc;
    if (cell0 >= n_cells) return;
    const uint32_t ncl = min(cpc, n_cells - cell0);
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const uint32_t team = warp >> 2, wl = warp & 3u, ttid = tid & 127u;
    const uint32_t n_heads = G * n_kv;

    extern __shared__ __align__(128) unsigned char raw[];
    unsigned char* sh = raw
        + ((1024u - ((uint32_t)__cvta_generic_to_shared(raw) & 1023u)) & 1023u);
    unsigned char* s_k = sh;
    unsigned char* s_v = s_k + 2u * NP * CPB;
    unsigned char* s_q = s_v + 2u * NP * CPB;
    unsigned char* s_p = s_q + (size_t)cpc * QSZ;
    __shared__ __align__(8) uint64_t s_bk[2], s_bv[2], s_bsp[2], s_ldc[2],
                                     s_bpe2[2], s_rsc[2];
    __shared__ float s_red[2][4][2u * G];
    __shared__ float s_mr[2][G], s_lr[2][G];       // cell state by ci&1
    __shared__ uint32_t tmem_slot[1], s_ntick;
    __shared__ uint16_t s_tick[MAXT];
    __shared__ uint32_t s_ckvh[MAXC], s_cb[MAXC], s_cglo[MAXC], s_cghi[MAXC],
                        s_cB0[MAXC], s_cnblk[MAXC], s_cnch[MAXC];
    __shared__ uint32_t s_btw[MAXT * BPC];

    if (tid == 0) {
        #pragma unroll
        for (uint32_t i = 0; i < 2u; ++i) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&s_bk[i])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&s_bv[i])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 3;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&s_bsp[i])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 128;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&s_ldc[i])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 2;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&s_bpe2[i])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 128;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&s_rsc[i])));
        }
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    // per-cell params: one warp-1 lane per cell
    if (warp == 1u && lane < ncl) {
        const uint32_t cell = cell0 + lane;
        const uint32_t kvh = cell % n_kv, b = cell / n_kv;
        const uint32_t pos = __ldg(positions + b);
        const uint32_t glo =
            (swa_window > 0 && pos + 1 > swa_window) ? pos + 1 - swa_window : 0u;
        const uint32_t ghi = pos + 1u;
        const uint32_t B0 = glo >> 4;
        const uint32_t nblk = ((ghi + 15u) >> 4) - B0;
        s_ckvh[lane] = kvh; s_cb[lane] = b; s_cglo[lane] = glo;
        s_cghi[lane] = ghi; s_cB0[lane] = B0; s_cnblk[lane] = nblk;
        s_cnch[lane] = (nblk + (BPC - 1u)) / BPC;
    }
    if (tid < 32u) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                     ::"r"((uint32_t)__cvta_generic_to_shared(tmem_slot)),
                       "r"(TMCOLS));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
    if (tid < 128u)
        for (uint32_t i = tid; i < 2u * 2048u / 16u; i += 128u)
            ((uint4*)s_p)[i] = make_uint4(0u, 0u, 0u, 0u);
    __syncthreads();
    if (tid == 0) {
        uint32_t nt = 0;
        for (uint32_t ci = 0; ci < ncl; ++ci)
            for (uint32_t ch = 0; ch < s_cnch[ci] && nt < MAXT; ++ch)
                s_tick[nt++] = (uint16_t)((ci << 8) | ch);
        s_ntick = nt;
    }
    __syncthreads();
    if (tid < 128u)
        for (uint32_t i = tid; i < s_ntick * BPC; i += 128u) {
            const uint32_t t = i / BPC, j = i % BPC;
            const uint32_t ci = s_tick[t] >> 8, ch = s_tick[t] & 0xffu;
            const uint32_t nbc = min(BPC, s_cnblk[ci] - ch * BPC);
            if (j < nbc) {
                const uint32_t slot = slots ? slots[s_cb[ci]] : s_cb[ci];
                s_btw[t * BPC + j] = __ldg(block_tables
                    + (size_t)slot * blocks_per_slot + s_cB0[ci] + ch * BPC + j);
            }
        }
    __syncthreads();
    const uint32_t ntick = s_ntick;

    auto bar_wait = [&](uint64_t* bar, uint32_t parity, uint32_t bid) {
        const uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
        for (uint32_t it = 0;; ++it) {
            uint32_t ok;
            asm volatile("{\n\t.reg .pred p;\n\t"
                         "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\t"
                         "selp.b32 %0, 1, 0, p;\n\t}"
                         : "=r"(ok) : "r"(a), "r"(parity));
            if (ok) return;
            if ((it & 0xffffu) == 0u
                && *(volatile unsigned long long*)&tc5d_stuck[0]) return;
            if (it > 50000000u) {
                const unsigned long long v = ((unsigned long long)bid << 48)
                    | ((unsigned long long)blockIdx.x << 32)
                    | ((unsigned long long)threadIdx.x << 16) | parity;
                for (int s = 0; s < 4; ++s)
                    if (atomicCAS(&tc5d_stuck[s], 0ull, v) == 0ull) break;
                return;
            }
        }
    };
    // stage lambdas: warp-agnostic (caller = one warp); tick-indexed s_btw
    auto stage_k = [&](uint32_t t) {
        const uint32_t ci = s_tick[t] >> 8, ch = s_tick[t] & 0xffu;
        const uint32_t sl = t & 1u;
        const uint32_t nbc = min(BPC, s_cnblk[ci] - ch * BPC);
        const uint32_t mk = (uint32_t)__cvta_generic_to_shared(&s_bk[sl]);
        if (lane == 0)
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                         ::"r"(mk), "r"(nbc * NP * 2048u));
        __syncwarp();
        if (lane < nbc * NP) {
            const uint32_t blk = s_btw[t * BPC + lane / NP];
            const uint32_t p = lane % NP;
            const int y = (int)(blk * 16u);
            // kvh_div: VIRTUAL kv heads (G=12 split as two G=6 cells - qwen4exp
            // 24q/2kv/hd256). Q rows, cells and the output all index by the
            // virtual head (kvh*G+g is exactly the right q row); only the KV
            // pool offset needs the PHYSICAL head, virt/kvh_div.
            const int x = (int)((s_ckvh[ci] / kvh_div) * HD + p * 128u);
            const uint32_t dk = (uint32_t)__cvta_generic_to_shared(
                s_k + sl * NP * CPB + p * CPB + (lane / NP) * 2048u);
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                " [%0], [%1, {%2, %3}], [%4];"
                ::"r"(dk), "l"(&tmk), "r"(x), "r"(y), "r"(mk) : "memory");
        }
    };
    auto stage_v = [&](uint32_t t) {
        const uint32_t ci = s_tick[t] >> 8, ch = s_tick[t] & 0xffu;
        const uint32_t sl = t & 1u;
        const uint32_t nbc = min(BPC, s_cnblk[ci] - ch * BPC);
        const uint32_t mv = (uint32_t)__cvta_generic_to_shared(&s_bv[sl]);
        if (lane == 0)
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                         ::"r"(mv), "r"(nbc * NP * 2048u));
        __syncwarp();
        if (lane < nbc * NP) {
            const uint32_t blk = s_btw[t * BPC + lane / NP];
            const uint32_t p = lane % NP;
            const int y = (int)(blk * 16u);
            const int x = (int)((s_ckvh[ci] / kvh_div) * HD + p * 128u);
            const uint32_t dv = (uint32_t)__cvta_generic_to_shared(
                s_v + sl * NP * CPB + p * CPB + (lane / NP) * 2048u);
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                " [%0], [%1, {%2, %3}], [%4];"
                ::"r"(dv), "l"(&tmv), "r"(x), "r"(y), "r"(mv) : "memory");
        }
    };
    const uint32_t id_s = (1u << 4) | ((8u >> 3) << 17) | ((128u >> 4) << 24);
    const uint32_t id_o = (1u << 4) | (1u << 15)
        | ((8u >> 3) << 17) | ((128u >> 4) << 24);
    const uint32_t p16b = (uint32_t)__cvta_generic_to_shared(s_p) >> 4;
    const uint32_t q16b = (uint32_t)__cvta_generic_to_shared(s_q) >> 4;
    const uint32_t k16b = (uint32_t)__cvta_generic_to_shared(s_k) >> 4;
    const uint32_t v16b = (uint32_t)__cvta_generic_to_shared(s_v) >> 4;
    auto mma_acc = [&](uint32_t d, uint64_t ad, uint64_t bd, uint32_t idc) {
        asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, 1, 0;\n\t"
                     "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t}"
                     ::"r"(d), "l"(ad), "l"(bd), "r"(idc));
    };
    auto mma_new = [&](uint32_t d, uint64_t ad, uint64_t bd, uint32_t idc) {
        asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, 0, 0;\n\t"
                     "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t}"
                     ::"r"(d), "l"(ad), "l"(bd), "r"(idc));
    };
    auto s_issue = [&](uint32_t t) {
        const uint32_t ci = s_tick[t] >> 8, ch = s_tick[t] & 0xffu;
        const uint32_t sl = t & 1u;
        const uint32_t gbase = (s_cB0[ci] + ch * BPC) * 16u;
        const uint32_t nbc = min(BPC, s_cnblk[ci] - ch * BPC);
        const uint32_t vkeys = min(s_cghi[ci] - gbase, nbc * 16u);
        const uint32_t two = vkeys > 128u ? 1u : 0u;
        const uint64_t adb = pd_tc5_sdesc(k16b + sl * (NP * CPB / 16u));
        const uint64_t bdb = pd_tc5_sdesc(q16b + ci * (QSZ / 16u));
        for (uint32_t T2 = 0; T2 <= two; ++T2) {
            const uint32_t d = tmem_slot[0] + sl * SCOLS + T2 * 8u;
            mma_new(d, adb + T2 * 1024u, bdb, id_s);
            #pragma unroll
            for (uint32_t kb = 1; kb < HD / 32u; ++kb)
                mma_acc(d, adb + (kb >> 2) * (CPB / 16u) + (kb & 3u) * 2u
                             + T2 * 1024u,
                        bdb + (kb >> 2) * 64u + (kb & 3u) * 2u, id_s);
        }
    };
    auto commit_to = [&](uint64_t* bar) {
        asm volatile(
            "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
            ::"r"((uint32_t)__cvta_generic_to_shared(bar)));
    };
    auto team_bar = [&]() {
        if (team == 0u) asm volatile("bar.sync 1, 128;");
        else asm volatile("bar.sync 2, 128;");
    };

    // ---- prologue staging + Q build --------------------------------------
    stamp(0);
    if (warp == 3u) {
        stage_k(0u); stage_v(0u);
        if (ntick > 1u) { stage_k(1u); stage_v(1u); }
    } else if (warp < 3u) {
        for (uint32_t ci = 0; ci < ncl; ++ci) {
            const float* qb = q
                + ((size_t)s_cb[ci] * n_heads + s_ckvh[ci] * G) * HD;
            unsigned char* dst = s_q + ci * QSZ;
            for (uint32_t i = tid; i < 8u * (HD / 16u); i += 96u) {
                const uint32_t r = i / (HD / 16u), c16 = i % (HD / 16u);
                unsigned char tmp[16];
                if (r < G) {
                    const float* src2 = qb + (size_t)r * HD + c16 * 16u;
                    #pragma unroll
                    for (uint32_t e = 0; e < 16u; ++e)
                        tmp[e] = __nv_fp8_e4m3(src2[e] * scale).__x;
                } else {
                    #pragma unroll
                    for (uint32_t e = 0; e < 16u; ++e) tmp[e] = 0u;
                }
                const uint32_t off = (c16 >> 3) * 1024u + r * 128u
                    + (((c16 & 7u) ^ r) << 4);
                *(uint4*)(dst + off) = *(const uint4*)tmp;
            }
        }
    }
    asm volatile("fence.proxy.async.shared::cta;");
    __syncthreads();
    stamp(12);
    if (dbg == 1u) {
        bar_wait(&s_bk[0], 0u, 10u); bar_wait(&s_bv[0], 0u, 11u);
        if (tid == 0)
            out_o[((size_t)s_cb[0] * n_heads + s_ckvh[0] * G) * HD] = 1.0f;
        return;
    }
    const uint32_t tmem = tmem_slot[0];
    stamp(1);
    // bootstrap phase 0 on s_bsp[0]: S(0) + two bare commits (team A issuers)
    uint32_t pk2 = team;
    if (tid == 0) {
        bar_wait(&s_bk[0], 0u, 10u);
        s_issue(0u);
        commit_to(&s_bsp[0]);
    } else if (tid == 32u || tid == 64u) {
        commit_to(&s_bsp[0]);
    }
    stamp(2);

    // ---- team tick loop --------------------------------------------------
    // parities (per thread; all barriers this thread waits flip 1:1)
    uint32_t psp = 0u, pep = 0u, pldc = 0u, pvp = 0u, prs = 0u;
    float l_run[G] = {}, mr[G];
    #pragma unroll
    for (uint32_t g = 0; g < G; ++g) mr[g] = -INFINITY;
    for (uint32_t t = team; t < ntick; t += 2u) {
        const uint32_t ci = s_tick[t] >> 8, ch = s_tick[t] & 0xffu;
        const uint32_t sl = team;                  // t&1 == team, always
        const uint32_t gbase = (s_cB0[ci] + ch * BPC) * 16u;
        const uint32_t nbc = min(BPC, s_cnblk[ci] - ch * BPC);
        const uint32_t vkeys = min(s_cghi[ci] - gbase, nbc * 16u);
        const uint32_t two = vkeys > 128u ? 1u : 0u;
        const uint32_t nlnk = (vkeys + 31u) >> 5;
        const uint32_t ob = OB0 + ((cell0 + ci) & 1u) * (NP * 8u);
        // ---- prework: V(t) landed + tail zero (overlaps other team) ------
        bar_wait(&s_bv[sl], pvp, 14u); pvp ^= 1u;
        if (vkeys < nlnk * 32u) {
            unsigned char* vb = s_v + sl * NP * CPB;
            const uint32_t zr = nlnk * 32u - vkeys;
            for (uint32_t i = ttid; i < zr * (HD / 16u); i += 128u) {
                const uint32_t r = vkeys + i / (HD / 16u);
                const uint32_t c16 = i % (HD / 16u);
                const uint32_t off = (c16 >> 3) * CPB + (r >> 3) * 1024u
                    + (r & 7u) * 128u + (((c16 & 7u) ^ (r & 7u)) << 4);
                *(uint4*)(vb + off) = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        stamp(4);
        // ---- softmax gate: phase(t-1) fired = all prior issue closed -----
        bar_wait(&s_bsp[team], psp, 16u); psp ^= 1u;
        asm volatile("fence.acq_rel.cta;");        // pair with state release
        stamp(3);
        // ---- cell state ---------------------------------------------------
        if (ch == 0u) {
            #pragma unroll
            for (uint32_t g = 0; g < G; ++g) { l_run[g] = 0.0f; mr[g] = -INFINITY; }
        } else {
            #pragma unroll
            for (uint32_t g = 0; g < G; ++g) {
                mr[g] = s_mr[ci & 1u][g]; l_run[g] = s_lr[ci & 1u][g];
            }
        }
        // ---- quadrant softmax (team-scoped) ------------------------------
        float sc[2][G];
        if constexpr (G == 2u) {
            uint32_t r2[2], r3[2];
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x2.b32 {%0,%1}, [%2];"
                         : "=r"(r2[0]), "=r"(r2[1]) : "r"(tmem + sl * SCOLS));
            if (two)
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x2.b32 {%0,%1}, [%2];"
                             : "=r"(r3[0]), "=r"(r3[1])
                             : "r"(tmem + sl * SCOLS + 8u));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            sc[0][0] = __uint_as_float(r2[0]);
            sc[0][1] = __uint_as_float(r2[1]);
            sc[1][0] = two ? __uint_as_float(r3[0]) : -INFINITY;
            sc[1][1] = two ? __uint_as_float(r3[1]) : -INFINITY;
        } else {
            // G > 2: the S tile is 8 M rows wide in tmem, so one x8 load
            // carries every head of the group. G8/hd512 has CH = 96 <= 128
            // and therefore one key half; G6/hd256 has CH = 192 and takes
            // the second half from +8, exactly where SCOLS = 16 puts it for
            // the G2 arm on the same geometry.
            uint32_t r8[8], r9[8] = {};
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 "
                         "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                         : "=r"(r8[0]), "=r"(r8[1]), "=r"(r8[2]), "=r"(r8[3]),
                           "=r"(r8[4]), "=r"(r8[5]), "=r"(r8[6]), "=r"(r8[7])
                         : "r"(tmem + sl * SCOLS));
            if constexpr (SCOLS > 8u) {
                if (two)
                    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 "
                                 "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                                 : "=r"(r9[0]), "=r"(r9[1]), "=r"(r9[2]),
                                   "=r"(r9[3]), "=r"(r9[4]), "=r"(r9[5]),
                                   "=r"(r9[6]), "=r"(r9[7])
                                 : "r"(tmem + sl * SCOLS + 8u));
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            #pragma unroll
            for (uint32_t g = 0; g < G; ++g) {
                sc[0][g] = __uint_as_float(r8[g]);
                sc[1][g] = (SCOLS > 8u && two) ? __uint_as_float(r9[g])
                                               : -INFINITY;
            }
        }
        const uint32_t kc0 = wl * 32u + lane, kc1 = 128u + wl * 32u + lane;
        const bool ok0 = gbase + kc0 >= s_cglo[ci] && gbase + kc0 < s_cghi[ci]
            && kc0 < vkeys;
        const bool ok1 = two && gbase + kc1 >= s_cglo[ci]
            && gbase + kc1 < s_cghi[ci] && kc1 < vkeys;
        #pragma unroll
        for (uint32_t g = 0; g < G; ++g) {
            if (!ok0) sc[0][g] = -INFINITY;
            if (!ok1) sc[1][g] = -INFINITY;
        }
        float mt[G];
        #pragma unroll
        for (uint32_t g = 0; g < G; ++g) {
            const float mv0 = fmaxf(sc[0][g], sc[1][g]);
            asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;"
                         : "=f"(mt[g]) : "f"(mv0));
        }
        if (lane == 0) {
            #pragma unroll
            for (uint32_t g = 0; g < G; ++g) s_red[team][wl][g] = mt[g];
        }
        team_bar();
        float corr[G], mn[G];
        #pragma unroll
        for (uint32_t g = 0; g < G; ++g) {
            const float mw = fmaxf(fmaxf(s_red[team][0][g], s_red[team][1][g]),
                                   fmaxf(s_red[team][2][g], s_red[team][3][g]));
            mn[g] = fmaxf(mr[g], mw);
            corr[g] = mn[g] > -INFINITY && mr[g] > -INFINITY
                ? __expf(mr[g] - mn[g]) : 1.0f;
            mr[g] = mn[g];
        }
        unsigned char* pimg = s_p + sl * 2048u;
        float lp[G];
        #pragma unroll
        for (uint32_t g = 0; g < G; ++g) {
            const float w0 = ok0 ? __expf(sc[0][g] - mn[g] + 6.1047935f) : 0.0f;
            const float w1 = ok1 ? __expf(sc[1][g] - mn[g] + 6.1047935f) : 0.0f;
            pimg[(kc0 >> 7) * 1024u + g * 128u
                 + (((((kc0 >> 4) & 7u)) ^ g) << 4) + (kc0 & 15u)] =
                __nv_fp8_e4m3(w0).__x;
            if (two)
                pimg[(kc1 >> 7) * 1024u + g * 128u
                     + (((((kc1 >> 4) & 7u)) ^ g) << 4) + (kc1 & 15u)] =
                    __nv_fp8_e4m3(w1).__x;
            lp[g] = w0 + w1;
            #pragma unroll
            for (uint32_t off = 1; off <= 16u; off <<= 1)
                lp[g] += __shfl_xor_sync(0xffffffffu, lp[g], off);
        }
        if (lane == 0) {
            #pragma unroll
            for (uint32_t g = 0; g < G; ++g) s_red[team][wl][G + g] = lp[g];
        }
        team_bar();
        float ls[G];
        #pragma unroll
        for (uint32_t g = 0; g < G; ++g) {
            ls[g] = s_red[team][0][G + g] + s_red[team][1][G + g]
                  + s_red[team][2][G + g] + s_red[team][3][G + g];
            l_run[g] = l_run[g] * corr[g] + ls[g];
        }
        if (wl == 0u && lane == 0) {
            #pragma unroll
            for (uint32_t g = 0; g < G; ++g) {
                s_mr[ci & 1u][g] = mr[g]; s_lr[ci & 1u][g] = l_run[g];
            }
            // release: the phase signal (tcgen05.commit) is relaxed; make
            // the state stores visible to the next chunk's phase-acquirer
            asm volatile("fence.acq_rel.cta;");
        }
        stamp(5);
        // ---- vote-gated rescale (multi-chunk only; my team, my O region) -
        bool anyc = false;
        #pragma unroll
        for (uint32_t g = 0; g < G; ++g) anyc |= corr[g] != 1.0f;
        const bool resc = dbg != 3u && ch > 0u && anyc;
        if (resc) {
            #pragma unroll
            for (uint32_t ct = 0; ct < NP; ++ct) {
                if constexpr (G == 2u) {
                    uint32_t r2[2];
                    asm volatile("tcgen05.ld.sync.aligned.32x32b.x2.b32 {%0,%1}, [%2];"
                                 : "=r"(r2[0]), "=r"(r2[1])
                                 : "r"(tmem + ob + ct * 8u));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    r2[0] = __float_as_uint(__uint_as_float(r2[0]) * corr[0]);
                    r2[1] = __float_as_uint(__uint_as_float(r2[1]) * corr[1]);
                    asm volatile("tcgen05.st.sync.aligned.32x32b.x2.b32 [%2], {%0,%1};"
                                 :: "r"(r2[0]), "r"(r2[1]), "r"(tmem + ob + ct * 8u));
                } else {
                    uint32_t r8[8];
                    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 "
                                 "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                                 : "=r"(r8[0]), "=r"(r8[1]), "=r"(r8[2]),
                                   "=r"(r8[3]), "=r"(r8[4]), "=r"(r8[5]),
                                   "=r"(r8[6]), "=r"(r8[7])
                                 : "r"(tmem + ob + ct * 8u));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (uint32_t g = 0; g < G; ++g)
                        r8[g] = __float_as_uint(__uint_as_float(r8[g]) * corr[g]);
                    asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%8], "
                                 "{%0,%1,%2,%3,%4,%5,%6,%7};"
                                 :: "r"(r8[0]), "r"(r8[1]), "r"(r8[2]),
                                    "r"(r8[3]), "r"(r8[4]), "r"(r8[5]),
                                    "r"(r8[6]), "r"(r8[7]),
                                    "r"(tmem + ob + ct * 8u));
                }
            }
            asm volatile("tcgen05.wait::st.sync.aligned;");
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&s_rsc[team])));
            bar_wait(&s_rsc[team], prs, 20u); prs ^= 1u;
        }
        asm volatile("fence.proxy.async.shared::cta;");    // P + V zeros
        team_bar();                                        // pre-issue
        stamp(6);
        // ---- stage next chunks (my stager warp 4T+3) ---------------------
        if (wl == 3u) {
            if (t + 2u < ntick) stage_k(t + 2u);   // K slot T: freed by S(t)
            // V slot 1-T: PV(t-1) done. t=0 SKIPS: V(1) was staged in the
            // prologue; a second arm on the count-1 s_bv[1] either overflows
            // the arrival count (UB) or starts an extra phase, leaving team
            // B's V-gate one phase behind forever (the ntick>=8 corruption).
            if (t > 0u && t + 1u < ntick) stage_v(t + 1u);
        }
        stamp(13);
        const bool lastch = (ch + 1u == s_cnch[ci]);
        // ---- issue block (my team's three issuers) -----------------------
        // order per issuer: [wait other team's ld-clear] -> chains -> commit
        if (ttid == 0u || ttid == 32u || ttid == 64u) {
            if (t > 0u) { bar_wait(&s_ldc[1u - team], pldc, 19u); pldc ^= 1u; }
            if (ttid == 0u) {
                if (t + 1u < ntick) {
                    stamp(7);
                    bar_wait(&s_bk[1u - team], pk2, 11u); pk2 ^= 1u;
                    stamp(2);
                    s_issue(t + 1u);
                }
                commit_to(&s_bsp[1u - team]);
            } else {
                if (resc) asm volatile("tcgen05.fence::after_thread_sync;");
                // each PV issuer owns NP/2 partitions (one chain each at
                // NP=2 - bit-identical to the shipped G2 form)
                const uint32_t ct0 = ttid == 32u ? 0u : NP / 2u;
                const uint64_t adb = pd_tc5_sdesc(v16b + sl * (NP * CPB / 16u));
                const uint64_t bdb = pd_tc5_sdesc(p16b + sl * 128u);
                #pragma unroll
                for (uint32_t ct = ct0; ct < ct0 + NP / 2u; ++ct) {
                    const uint32_t d = tmem + ob + ct * 8u;
                    if (ch == 0u) mma_new(d, adb + ct * (CPB / 16u), bdb, id_o);
                    else mma_acc(d, adb + ct * (CPB / 16u), bdb, id_o);
                    #pragma unroll
                    for (uint32_t kb = 1; kb < CH / 32u; ++kb)
                        if (kb < nlnk)
                            mma_acc(d, adb + ct * (CPB / 16u) + kb * 256u,
                                    bdb + (kb >> 2) * 64u + (kb & 3u) * 2u, id_o);
                }
                commit_to(&s_bsp[1u - team]);
                if (lastch) commit_to(&s_bpe2[team]);
            }
        }
        stamp(7);
        // ---- epilogue (lastch): my team, gated on my PV commits ----------
        if (lastch) {
            bar_wait(&s_bpe2[team], pep, 17u); pep ^= 1u;
            stamp(8);
            float inv_l[G];
            #pragma unroll
            for (uint32_t g = 0; g < G; ++g)
                inv_l[g] = l_run[g] > 0.0f ? 1.0f / l_run[g] : 0.0f;
            #pragma unroll
            for (uint32_t ct = 0; ct < NP; ++ct) {
                // the x8 load below writes eight registers whatever G is
                constexpr uint32_t NRG = (G <= 2u) ? 2u : 8u;
                uint32_t rg[NRG];
                if constexpr (G == 2u) {
                    asm volatile("tcgen05.ld.sync.aligned.32x32b.x2.b32 {%0,%1}, [%2];"
                                 : "=r"(rg[0]), "=r"(rg[1])
                                 : "r"(tmem + ob + ct * 8u));
                } else {
                    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 "
                                 "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                                 : "=r"(rg[0]), "=r"(rg[1]), "=r"(rg[2]),
                                   "=r"(rg[3]), "=r"(rg[4]), "=r"(rg[5]),
                                   "=r"(rg[6]), "=r"(rg[7])
                                 : "r"(tmem + ob + ct * 8u));
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                const uint32_t dim = ct * 128u + wl * 32u + lane;
                #pragma unroll
                for (uint32_t g = 0; g < G; ++g)
                    out_o[((size_t)s_cb[ci] * n_heads + s_ckvh[ci] * G + g) * HD
                          + dim] = __uint_as_float(rg[g]) * inv_l[g];
            }
            stamp(9);
        }
        // ---- ld-clear: no more lds from me this tick ---------------------
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
                     ::"r"((uint32_t)__cvta_generic_to_shared(&s_ldc[team])));
    }
    __syncthreads();
    if (tid < 32u)
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                     ::"r"(tmem), "r"(TMCOLS));
    if (PROF && tid == 0) {
        #pragma unroll
        for (int i = 0; i < 15; ++i)
            if (i != 10 && i != 11) atomicAdd(&tc5d_prof[i], tp[i]);
        atomicAdd(&tc5d_prof[10], 1ull);
        atomicAdd(&tc5d_prof[11], clock64() - tk0c);
    }
#else
    (void)tmk; (void)tmv; (void)q; (void)out_o; (void)positions; (void)slots;
    (void)block_tables; (void)blocks_per_slot; (void)swa_window; (void)scale;
    (void)n_kv; (void)n_seq; (void)cpc; (void)dbg; (void)kvh_div;
#endif
}

// slot 431: final-output tcgen05 decode attention (see the file header).
// Same param list as pd_attn_decode_batch_paged; `sinks` accepted for
// signature parity and ignored (gemma passes -inf sinks = a no-op fold).
// rc 0 = launched (final rows in `out`, caller skips partials+combine);
// rc -2 = shape/arch not covered; rc -3 = smem over the opt-in cap.
PD_EXPORT
int pd_attn_decode_tc5_paged(const void* q, const void* pool_k, const void* pool_v,
                             const void* sinks, void* out, const void* positions,
                             const void* slots, const void* block_tables,
                             uint32_t blocks_per_slot, uint32_t n_heads,
                             uint32_t n_kv_heads, uint32_t head_dim, uint32_t kv_dim,
                             uint32_t swa_window, uint32_t batch, float scale,
                             uint32_t kv_dtype, void* stream) {
    (void)sinks;
#if !defined(PD_TC5_HOST)
    (void)q; (void)pool_k; (void)pool_v; (void)out; (void)positions; (void)slots;
    (void)block_tables; (void)blocks_per_slot; (void)n_heads; (void)n_kv_heads;
    (void)head_dim; (void)kv_dim; (void)swa_window; (void)batch; (void)scale;
    (void)kv_dtype; (void)stream;
    return -2;
#else
    if (n_heads == 0 || batch == 0) return 0;
    const uint32_t group = n_kv_heads ? n_heads / n_kv_heads : 1u;
    // n_kv_heads may be VIRTUAL: a G=12 model (qwen4exp 24q/2kv/hd256) rides
    // the <256,6> instantiation as two virtual heads per physical one, and the
    // caller passes the virtual count with the PHYSICAL kv_dim. The divisor
    // falls out of the mismatch; 1 for every native geometry.
    const uint32_t kvh_div =
        (kv_dim != 0u && (n_kv_heads * head_dim) % kv_dim == 0u)
            ? (n_kv_heads * head_dim) / kv_dim : 1u;
    if (kvh_div != 1u && kvh_div != 2u) return -2;
    static const int tc5ok = [] {
        int dev = 0, cc = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&cc, cudaDevAttrComputeCapabilityMajor, dev);
        return (cc == 10 && pd_tmap_encode()) ? 1 : 0;
    }();
    // two covered geometries (gemma-31b): SWA <hd256,g2> and GLB <hd512,g8>.
    // GLB layers have no window - the ENGINE passes a banded effective
    // window (kv_split_band * 128 >= pos_max, riding the decode-graph key)
    // so swa_window > 0 stays the tick-table bound for both shapes.
    const bool swa5 = head_dim == 256u && group == 2u;
    const bool glb5 = head_dim == 512u && group == 8u;
    // qwen3.8-27b full-attn: 24q/4kv/hd256, no window. Same hd256 geometry
    // as gemma SWA (CH=192, BPC=12, SCOLS=16), only the group differs, and
    // G only indexes the 8-row M tile both already emit. The caller passes a
    // BANDED effective window the way gemma's GLB layers do, so swa_window
    // stays the tick-table bound for all three shapes.
    const bool dns5 = head_dim == 256u && group == 6u;
    if (!tc5ok || kv_dtype != PD_KV_FP8_E4M3 || !(swa5 || glb5 || dns5)
        || n_heads != n_kv_heads * group || swa_window == 0u) {
        static uint32_t declined = 0;
        if (!declined) {
            fprintf(stderr, "[tc5-attn] declined: ok=%d kv=%u hd=%u g=%u win=%u\n",
                    tc5ok, kv_dtype, head_dim, group, swa_window);
            declined = 1;
        }
        return -2;
    }
    // one-wave cell packing: cpc cells/CTA, tick table capped at 128 ticks
    // and 16 cells (kernel MAXT/MAXC) - swa_window bounds nch so both caps
    // are host-checkable without reading positions
    static const uint32_t nsm = [] {
        int dev = 0, n = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
        return (uint32_t)(n > 0 ? n : 1);
    }();
    const uint32_t n_cells = batch * n_kv_heads;
    const uint32_t cpc = (n_cells + nsm - 1u) / nsm;
    // blocks-per-chunk mirrors the kernel's BPC (12 @ hd256, 6 @ hd512);
    // K/V slab bytes are shape-invariant (CH*HD conserved), only the
    // per-cell Q slab (8 rows * hd) differs.
    const uint32_t bpc = head_dim == 256u ? 12u : 6u;
    const uint32_t qsz = 8u * head_dim;
    const uint32_t nch_max = ((swa_window >> 4) + 1u + bpc - 1u) / bpc;
    if (cpc > 16u || cpc * nch_max > 128u) return -2;
    const uint32_t smem5 = 2u * 2u * 49152u + cpc * qsz + 2u * 2048u + 1024u;
    static int smem_cap = -1;
    if (smem_cap < 0)
        cudaDeviceGetAttribute(&smem_cap, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    if ((int)smem5 > smem_cap) return -3;
    // per-pool tensor maps: pools are per LAYER (v9q's t9 cache precedent)
    struct PdTmEnt5 { const void* p; uint32_t kd; CUtensorMap m; };
    static PdTmEnt5 t5c[64];
    static uint32_t t5n = 0;
    auto get_tm5 = [&](const void* base) -> const CUtensorMap* {
        for (uint32_t i = 0; i < t5n; ++i)
            if (t5c[i].p == base && t5c[i].kd == kv_dim) return &t5c[i].m;
        if (t5n >= 64u) t5n = 0;
        if (!pd_attn_tmap_kv_f8s(&t5c[t5n].m, base, kv_dim)) return nullptr;
        t5c[t5n].p = base; t5c[t5n].kd = kv_dim;
        return &t5c[t5n++].m;
    };
    const CUtensorMap* tk = get_tm5(pool_k);
    const CUtensorMap* tv = tk ? get_tm5(pool_v) : nullptr;
    if (!tk || !tv) return -2;
    const uint32_t nctas = (n_cells + cpc - 1u) / cpc;
    // route witness (serve-log engagement proof) - one-shot per SHAPE so a
    // GLB A/B can prove hd512 engagement even when SWA elected first.
    // stderr, unbuffered ([widenth] precedent: stdout printf never flushes
    // under a redirected serve log).
    static uint32_t witnessed[3] = {0u, 0u, 0u};
    const uint32_t wix = glb5 ? 1u : (dns5 ? 2u : 0u);
    if (!witnessed[wix]) {
        fprintf(stderr, "[tc5-attn] elected: hd=%u g=%u cells=%u cpc=%u grid=%u smem=%u window=%u\n",
                head_dim, group, n_cells, cpc, nctas, smem5, swa_window);
        witnessed[wix] = 1;
    }
    if (dns5) {
        static uint32_t smset5d = 0;
        if (smem5 > smset5d) {
            cudaFuncSetAttribute((const void*)pd_attn_decode_tc5e_kernel<256u, 6u>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem5);
            smset5d = smem5;
        }
        pd_pdl_go(pd_attn_decode_tc5e_kernel<256u, 6u>, nctas, 256u, smem5,
                  (cudaStream_t)stream,
                  *tk, *tv, (const float*)q, (float*)out,
                  (const unsigned int*)positions, (const unsigned int*)slots,
                  (const uint32_t*)block_tables, blocks_per_slot, swa_window, scale,
                  n_kv_heads, batch, cpc, 0u, kvh_div);
    } else if (swa5) {
        static uint32_t smset5 = 0;
        if (smem5 > smset5) {
            cudaFuncSetAttribute((const void*)pd_attn_decode_tc5e_kernel<256u, 2u>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem5);
            smset5 = smem5;
        }
        pd_pdl_go(pd_attn_decode_tc5e_kernel<256u, 2u>, nctas, 256u, smem5,
                  (cudaStream_t)stream,
                  *tk, *tv, (const float*)q, (float*)out,
                  (const unsigned int*)positions, (const unsigned int*)slots,
                  (const uint32_t*)block_tables, blocks_per_slot, swa_window, scale,
                  n_kv_heads, batch, cpc, 0u, kvh_div);
    } else {
        static uint32_t smset5g = 0;
        if (smem5 > smset5g) {
            cudaFuncSetAttribute((const void*)pd_attn_decode_tc5e_kernel<512u, 8u>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem5);
            smset5g = smem5;
        }
        pd_pdl_go(pd_attn_decode_tc5e_kernel<512u, 8u>, nctas, 256u, smem5,
                  (cudaStream_t)stream,
                  *tk, *tv, (const float*)q, (float*)out,
                  (const unsigned int*)positions, (const unsigned int*)slots,
                  (const uint32_t*)block_tables, blocks_per_slot, swa_window, scale,
                  n_kv_heads, batch, cpc, 0u, kvh_div);
    }
    return pd_launch_status();
#endif
}
