// attn/prefill_fa2.cuh - the FA-2 prefill tile and the pf7 lineage.
// Textually-included segment of the single pack translation unit.
// Not standalone-compilable: include order is defined by ../pack.cu.
//
// Split out of attn/prefill.cuh (8045 lines against the ~2500 line
// ceiling), cut on the file's own section markers.
//
// Holds the FA-2 prefill tile, the f16 v4 tile, and pf7 / pf7rp.
//
// Also carries the CK macro - read the DO-NOT-DELETE note above its
// definition before touching it; out-of-tree harnesses depend on it.
//
// Include after attn/prefill.cuh.
// ── FA-2 prefill tile (prefill-attention rung) ─────────────────────────────
// RUNG VERDICT: FALSIFIED both ways at the 2048-row tick shape (ctx
// 4096, locked-clock harness) - kept env-gated off as the measured record.
//   SWA hd256: best variant (PT32 split-pipelined SB) 1011 us vs the WMMA
//   tile's 943 (-7%) with FEWER executed instructions (239.6M vs 248.0M);
//   the residual is phase-chain latency at 1 CTA/SM vs the WMMA tile's two
//   independent co-resident 128-thread CTAs. GLOBAL hd512 (M=32, k1=4):
//   6743 us vs v3w's 5124 (-32%) - 2x KV walks + score-phase warp
//   starvation (4 tasks / 8 warps at mt=2). Both incumbents are confirmed
//   local optima; the 73-80-TF-vs-440-peak headroom is not reachable via
//   this restructure at these geometries (matches the v3s precedent).
// Mechanisms measured on the way (transferable, esp. to the spec-verify FA
// in attn/decode.cuh which shares the unpadded/runtime-geometry design):
//   +8-half row pads (head_dim rows are 0 mod 32 banks -> 8-way ldmatrix
//   conflicts): 2819->1358 us (2.4x!). constexpr HD/G (runtime div/mod
//   chains cost +53% inst_executed): -7%. Split-pipelined SB (K staged
//   after score, V after o, wait_group 1): -9%, DB-class overlap at SB
//   smem. Warp-parallel softmax: NEGATIVE below ~32 elements/row. PT must
//   be a multiple of 16 (o-mma strip). Co-residency (128thr/M32): neutral.
//
// The design: CTA = (kv head, 32-row chunk) fuses the whole G-group, so
// K/V is staged once per chunk for all G q-heads through the cp.async ring
// (G=2: half the WMMA tile's L2 traffic), scores and O ride m16n8k16 f16
// mma with O in REGISTERS (f32 accumulate - strictly tighter than the WMMA
// tile's f16 O; same v2-vs-v1 numeric class, would gate by the greedy
// oracle probe if ever defaulted).
//
// CONTRACT (vs the WMMA tile's arbitrary positions): rows rb..rb+nrows-1
// are CONSECUTIVE POSITIONS of one slot - pos_j = positions[rb] + j - which
// is exactly the engine's per-slot chunk prefill shape (slots[0] is already
// the launcher-wide slot). The per-row masks derive arithmetically, which
// is what lets M reach 32*G q-vectors (the spec kernel's pos_r[8] register
// arrays cap it at k1<=8).
// PT positions/stage; DB double-buffers the KV ring (hd256/M64/PT16 lands
// ~71 KB with DB - under sm_120's 99 KB cap with overlap intact).
template <uint32_t PT, uint32_t TPW, bool DB = true, uint32_t NT = 256u,
          uint32_t HD = 256u, uint32_t GL = 1u, bool SPL = false>
__global__ void __launch_bounds__(NT, 256u / NT) pd_attn_prefill_fa_kernel(
    const float* __restrict__ q, const __half* __restrict__ pool_k,
    const __half* __restrict__ pool_v, const float* __restrict__ sinks,
    float* __restrict__ out, const unsigned int* __restrict__ positions,
    const unsigned int* __restrict__ slots,
    const uint32_t* __restrict__ block_tables, uint32_t blocks_per_slot,
    uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim_rt, uint32_t kv_dim,
    uint32_t swa_window, uint32_t rows, uint32_t k1, float scale) {
#if PD_FA_OK
    const uint32_t kvh = blockIdx.x, c = blockIdx.y;
    // HD/GL compile-time: v2 with these runtime executed +53% instructions
    // vs the WMMA tile (inst_executed 379M vs 248M) - div/mod chains in
    // the staging loop and rr/G masks; constexpr geometry folds them all
    constexpr uint32_t G = 1u << GL;
    const uint32_t M = k1 * G;
    const uint32_t mt = (M + 15u) / 16u;
    const uint32_t Mp = mt * 16u;
    const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
    const uint32_t rb = c * k1;
    const uint32_t nrows = (rows - rb) < k1 ? (rows - rb) : k1;
    const uint32_t slot = slots ? slots[0] : 0u;

    const uint32_t pos0 = positions[rb];
    const uint32_t lo0 =
        (swa_window > 0 && pos0 + 1u > swa_window) ? (pos0 + 1u - swa_window) : 0u;
    const uint32_t hi = pos0 + nrows - lo0;  // walk [lo0, pos0+nrows-1] rel

    // row strides padded +8 halfs / +1 f32: HD halfs is 512B = 0 mod
    // the 32 banks, which makes every ldmatrix's 8 rows land on the same 4
    // banks (8-way conflict) and the thread-per-row softmax scan fully
    // serial - first build measured L1/TEX 71% vs compute 8.6% from exactly
    // this. +8 halfs keeps rows 16B-aligned for cp.async/ldmatrix.
    const uint32_t KP = HD + 8u;   // q / K / V row stride (halfs)
    const uint32_t PP = PT + 1u;         // score row stride (f32)
    const uint32_t FP = PT + 8u;         // f16 P row stride (ldmatrix-fed)
    extern __shared__ __align__(16) unsigned char pfa_sm[];
    __half* s_q = (__half*)pfa_sm;                              // [Mp][KP]
    __half* s_kv = (__half*)(s_q + (size_t)Mp * KP);            // [DB?2:1][2][PT][KP]
    float* s_p = (float*)(s_kv + (size_t)(DB ? 4u : 2u) * PT * KP);  // [Mp][PP]
    float* s_m = s_p + (size_t)Mp * PP;                          // [Mp] x3
    float* s_l = s_m + Mp;
    float* s_corr = s_l + Mp;
    __half* s_pf = (__half*)(s_corr + Mp);                       // [Mp][FP] f16

    for (uint32_t i = tid; i < Mp * HD; i += NT) {
        const uint32_t rh = i / HD, e = i % HD;
        const uint32_t j = rh / G, g = rh % G;
        s_q[(size_t)rh * KP + e] = (j < nrows)
            ? __float2half(q[((size_t)(rb + j) * n_heads + (size_t)kvh * G + g) * HD + e])
            : __half(0.f);
    }
    for (uint32_t i = tid; i < Mp; i += NT) { s_m[i] = -INFINITY; s_l[i] = 0.f; }

    float o_acc[TPW][8][4];
    #pragma unroll
    for (uint32_t a = 0; a < TPW; ++a)
        #pragma unroll
        for (uint32_t b = 0; b < 8u; ++b)
            #pragma unroll
            for (uint32_t cc2 = 0; cc2 < 4u; ++cc2) o_acc[a][b][cc2] = 0.f;

    const uint32_t* bt = block_tables + (size_t)slot * blocks_per_slot;
    constexpr uint32_t LINES = (HD * 2u) >> 4;  // 16B lines/row, pow2
    // kv2: 2 = both planes (classic), 0 = K only, 1 = V only (SPL split-
    // pipeline stages the planes at different points in the phase chain)
    auto stage = [&](uint32_t bf, uint32_t t0, uint32_t kv2) {
        const uint32_t n_t = hi - t0 < PT ? hi - t0 : PT;
        const uint32_t nrl = (kv2 == 2u ? 2u : 1u) * n_t;
        for (uint32_t i = tid; i < nrl * LINES; i += NT) {
            const uint32_t rl = i / LINES;  // constexpr pow2 -> shift
            const uint32_t l = i % LINES;
            const uint32_t kvsel = kv2 == 2u ? (rl >= n_t ? 1u : 0u) : kv2;
            const uint32_t p = (kv2 == 2u && kvsel) ? rl - n_t : rl;
            const uint32_t gpos = lo0 + t0 + p;
            const uint32_t blk = bt[gpos >> 4];
            const __half* src = (kvsel ? pool_v : pool_k)
                + (size_t)blk * 16u * kv_dim + (size_t)(gpos & 15u) * kv_dim
                + (size_t)kvh * HD;
            __half* dst = s_kv + ((size_t)(bf * 2u + kvsel) * PT + p) * KP;
            pd_attn_cpa16((char*)dst + l * 16u, (const char*)src + l * 16u);
        }
        pd_attn_cpa_commit();
    };

    __syncthreads();
    if (DB) stage(0u, 0u, 2u);
    if (SPL) { stage(0u, 0u, 0u); stage(0u, 0u, 1u); }  // K then V groups
    uint32_t bf = 0;
    for (uint32_t t0 = 0; t0 < hi; t0 += PT, bf ^= (DB ? 1u : 0u)) {
        const uint32_t n_t = hi - t0 < PT ? hi - t0 : PT;
        const bool more = t0 + PT < hi;
        if (SPL) {
            // outstanding groups here: {K(t), V(t)} (or just {K,V} of the
            // final tile) - wait_group 1 completes K(t), V(t) keeps flying
            pd_attn_cpa_wait1();
        } else if (DB) {
            if (more) stage(bf ^ 1u, t0 + PT, 2u);
            if (more) pd_attn_cpa_wait1(); else pd_attn_cpa_wait0();
        } else {
            stage(0u, t0, 2u);
            pd_attn_cpa_wait0();
        }
        __syncthreads();
        const __half* kbuf = s_kv + (size_t)(bf * 2u) * PT * KP;
        const __half* vbuf = s_kv + ((size_t)(bf * 2u) + 1u) * PT * KP;

        {
            const uint32_t tasks = mt * (PT / 8u);
            for (uint32_t task = warp; task < tasks; task += NT / 32u) {
                const uint32_t tm = task / (PT / 8u), cs = task % (PT / 8u);
                const uint32_t r0 = tm * 16u, p0 = cs * 8u;
                float d[4] = {0.f, 0.f, 0.f, 0.f};
                for (uint32_t kk = 0; kk < HD; kk += 16u) {
                    uint32_t af[4];
                    const __half* ap = s_q + (size_t)(r0 + (lane & 15u)) * KP
                                     + kk + ((lane >> 4) ? 8u : 0u);
                    pd_ldm_x4(af, (const unsigned char*)ap);
                    uint32_t bfr[2];
                    const __half* bp = kbuf + (size_t)(p0 + (lane & 7u)) * KP
                                     + kk + (((lane >> 3) & 1u) ? 8u : 0u);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                                 : "=r"(bfr[0]), "=r"(bfr[1])
                                 : "r"((unsigned)__cvta_generic_to_shared(bp)));
                    pd_fa_mma16(d, af[0], af[1], af[2], af[3], bfr[0], bfr[1]);
                }
                #pragma unroll
                for (uint32_t half = 0; half < 2u; ++half) {
                    const uint32_t rr = r0 + (lane >> 2) + half * 8u;
                    #pragma unroll
                    for (uint32_t cc = 0; cc < 2u; ++cc) {
                        const uint32_t pp = p0 + 2u * (lane & 3u) + cc;
                        const uint32_t j = rr / G;
                        const uint32_t gpos = lo0 + t0 + pp;
                        float v = d[half * 2u + cc] * scale;
                        // consecutive-position masks: causal gpos > pos0+j,
                        // window gpos + w <= pos0 + j (w>0), dead rows/tail
                        if (pp >= n_t || j >= nrows || gpos > pos0 + j
                            || (swa_window > 0 && gpos + swa_window <= pos0 + j))
                            v = -INFINITY;
                        s_p[rr * PP + pp] = v;
                    }
                }
            }
        }
        __syncthreads();
        // SPL: score consumers are past kbuf - refill the K plane under the
        // softmax + o phases
        if (SPL && more) stage(0u, t0 + PT, 0u);
        // thread-per-row softmax: 64 independent threads give cross-thread
        // ILP; a warp-parallel shfl-tree version measured worse (1195->1245
        // at PT32) - 16-32 elements/row is below the reduction's break-even
        if (tid < Mp) {
            const uint32_t rr = tid;
            float mx = s_m[rr];
            for (uint32_t pp = 0; pp < n_t; ++pp) mx = fmaxf(mx, s_p[rr * PP + pp]);
            const float corr = (mx == -INFINITY) ? 1.f : __expf(s_m[rr] - mx);
            float ls = 0.f;
            for (uint32_t pp = 0; pp < n_t; ++pp) {
                const float sp = s_p[rr * PP + pp];
                const float w = (sp == -INFINITY) ? 0.f : __expf(sp - mx);
                s_pf[rr * FP + pp] = __float2half(w);
                ls += w;
            }
            for (uint32_t pp = n_t; pp < PT; ++pp) s_pf[rr * FP + pp] = __half(0.f);
            s_corr[rr] = corr;
            s_l[rr] = s_l[rr] * corr + ls;
            s_m[rr] = mx;
        }
        // SPL: V(t) must have landed before the o phase - each thread waits
        // its own groups, the phase barrier below publishes cross-thread
        // (outstanding {V(t), K(t+1)} when more -> wait 1; {V(t)} last tile)
        if (SPL) { if (more) pd_attn_cpa_wait1(); else pd_attn_cpa_wait0(); }
        __syncthreads();
        {
            const uint32_t slices = HD / 64u;
            const uint32_t tasks = mt * slices;
            #pragma unroll
            for (uint32_t ti = 0; ti < TPW; ++ti) {
                const uint32_t task = warp + ti * (NT / 32u);
                if (task >= tasks) break;
                const uint32_t tm = task / slices, sl = task % slices;
                const uint32_t r0 = tm * 16u, n_base = sl * 64u;
                #pragma unroll
                for (uint32_t half = 0; half < 2u; ++half) {
                    const uint32_t rr = r0 + (lane >> 2) + half * 8u;
                    const float corr = s_corr[rr];
                    #pragma unroll
                    for (uint32_t sub = 0; sub < 8u; ++sub) {
                        o_acc[ti][sub][half * 2u] *= corr;
                        o_acc[ti][sub][half * 2u + 1u] *= corr;
                    }
                }
                for (uint32_t kk = 0; kk < n_t; kk += 16u) {
                    uint32_t af[4];
                    const __half* ap = s_pf + (size_t)(r0 + (lane & 15u)) * FP
                                     + kk + ((lane >> 4) ? 8u : 0u);
                    pd_ldm_x4(af, (const unsigned char*)ap);
                    #pragma unroll
                    for (uint32_t sub = 0; sub < 8u; ++sub) {
                        uint32_t bfr[2];
                        const __half* bp = vbuf + (size_t)(kk + (lane & 15u)) * KP
                                         + n_base + sub * 8u;
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                                     : "=r"(bfr[0]), "=r"(bfr[1])
                                     : "r"((unsigned)__cvta_generic_to_shared(bp)));
                        pd_fa_mma16(o_acc[ti][sub], af[0], af[1], af[2], af[3],
                                    bfr[0], bfr[1]);
                    }
                }
            }
        }
        __syncthreads();
        // SPL: o consumers are past vbuf - refill the V plane; it flies
        // through the next tile's score + softmax phases
        if (SPL && more) stage(0u, t0 + PT, 1u);
    }
    // epilogue: sink fold + normalize + direct write (no splits/combine)
    {
        const uint32_t slices = HD / 64u;
        const uint32_t tasks = mt * slices;
        #pragma unroll
        for (uint32_t ti = 0; ti < TPW; ++ti) {
            const uint32_t task = warp + ti * (NT / 32u);
            if (task >= tasks) break;
            const uint32_t tm = task / slices, sl = task % slices;
            const uint32_t r0 = tm * 16u, n_base = sl * 64u;
            #pragma unroll
            for (uint32_t half = 0; half < 2u; ++half) {
                const uint32_t rr = r0 + (lane >> 2) + half * 8u;
                const uint32_t j = rr / G, g = rr % G;
                if (j >= nrows) continue;
                const uint32_t h = kvh * G + g;
                const float mm = s_m[rr], ll = s_l[rr];
                const float sv = sinks[h];
                const float mtot = fmaxf(mm, sv);
                const float cm = __expf(mm - mtot);
                const float cs2 = __expf(sv - mtot);
                const float l = ll * cm + cs2;
                const float nrm = l > 0.f ? cm / l : 0.f;
                #pragma unroll
                for (uint32_t sub = 0; sub < 8u; ++sub) {
                    float* op = out + ((size_t)(rb + j) * n_heads + h) * HD
                              + n_base + sub * 8u + 2u * (lane & 3u);
                    op[0] = o_acc[ti][sub][half * 2u] * nrm;
                    op[1] = o_acc[ti][sub][half * 2u + 1u] * nrm;
                }
            }
        }
    }
#else
    (void)q; (void)pool_k; (void)pool_v; (void)sinks; (void)out; (void)positions;
    (void)slots; (void)block_tables; (void)blocks_per_slot; (void)n_heads;
    (void)n_kv_heads; (void)head_dim_rt; (void)kv_dim; (void)swa_window;
    (void)rows; (void)k1; (void)scale;
#endif
}

// pf_v4: Act-14 playbook applied to PREFILL attention. The incumbent WMMA
// tiles wmma::load_matrix_sync K/V straight from GLOBAL per q-head (no
// staging, no pipeline, G-fold redundant KV reads) and run at 60-68 TF on
// B200 - ~1.3 % of the tensor floor. This kernel: one CTA per (kv head,
// token tile), K/V cp.async double-buffered into smem once for all G heads,
// HMMA m16n8k16 scores AND PV (spec-FA frag patterns: K-side B-frag no
// .trans, V-side .trans), O in registers, predicate masks (never trust
// scores of masked cols - garbage K rows would NaN the max).
// MR mma rows = TQ tokens x G heads; MR=64 @ HD256/G2, 32 @ HD512/G8
// (o_acc register budget). TK = 32 keys/tile @ HD256, 16 @ HD512.
// G=6 (qwen3.6-27B 24q/4kv/hd256) takes MR=48 - the only 16-multiple that
// divides by 6; before this instantiation the shape fell to the per-q-head
// WMMA tile (f16) / the SCALAR paged walk (fp8, the elected kv8 class).
// G=12 (Qwen3.8-Flash-Next 24q/2kv/hd256) takes MR=48 too: 4 tokens x 12
// heads a CTA, the o_acc budget G=6 already runs at.
// KVT arm: e4m3 pools ride the v3c PIPE class - cp.async lands raw byte
// tiles in sh_raw, a post-wait smem->smem widened-cvt expand fills the same
// padded-row half layout f16 staging writes (staging bit-equal to an f16
// pool holding the e4m3 values; e4m3 -> f16 is exact). Double-buffered raw
// like the half tiles; smem +4*TK*HD bytes, occupancy unchanged (1 CTA/SM
// either way at these sizes).
template <uint32_t HD, uint32_t G, uint32_t TK, typename KVT = __half>
__global__ void __launch_bounds__(256) pd_attn_prefill_f16_v4_kernel(
    const float* __restrict__ q, const KVT* __restrict__ pool_k,
    const KVT* __restrict__ pool_v, const float* __restrict__ sinks,
    float* __restrict__ out, const unsigned int* __restrict__ positions,
    const uint32_t* __restrict__ block_tables, uint32_t blocks_per_slot,
    const unsigned int* __restrict__ slots,
    uint32_t n_heads, uint32_t kv_dim, uint32_t swa_window, uint32_t n_rows,
    float scale, const uint32_t* __restrict__ run_offs,
    const unsigned int* __restrict__ win_pos) {
#if PD_FA_OK
    constexpr uint32_t MR = (G == 6u || G == 12u)
        ? 48u : (G == 9u ? 144u : (HD >= 512u ? 32u : 64u));  // mma rows
    static_assert(MR % G == 0u && MR % 16u == 0u, "MR must fit G and mma frags");
    constexpr uint32_t TQ = MR / G;                    // tokens per CTA
    constexpr uint32_t row_e = HD + 8u;                // +8-half pad
    constexpr uint32_t t_s = TK + 8u;                  // score/P row stride
    constexpr uint32_t NRF = MR / 16u;                 // row fragments
    constexpr uint32_t NW = 8u;                        // warps
    constexpr uint32_t SLICE = HD / NW;                // PV dims per warp
    // batched-runs arm (pf5's prologue verbatim): grid.z indexes
    // the run table; every row-indexed base re-aims at this run. The classic
    // one-run launch (run_offs == nullptr) is bit-identical. Required here
    // because this kernel reads slots[0] once (chunk = one sequence): a
    // whole-chunk multi-slot launch without the table attends slot 0's KV
    // for every row (measured: muse temp-0 divergence, OSL p50 128 -> 106).
    // `slots` is not re-aimed here: nvcc 13.3 lowers `if (slots) slots +=
    // roff;` on sm_120a to a 32-bit uniform ULEA with the high word zeroed
    // (SASS: UMOV UR5, URZ; ULEA UR4, UR4(roff), UR14, 0x2; MOV.64 R6, UR4),
    // so the pointer loses its top 32 bits and the slot read lands at
    // low32(slots) - memcheck'd on the qwen35 dflash ring
    // (rung E2): "Access to 0x742e6800... 214946650112 bytes
    // before the nearest allocation at 0x3280000000". The run's first row
    // is carried as an index and applied at the one use site instead,
    // which lowers to the 64-bit IMAD.WIDE every other pointer gets. The
    // pf5 / pf5g-c2 / pf6s / pf6g-c2 runs prologues still carry the
    // original pattern - audit them before trusting PF_RUNS on sm_120.
    // window floors from the TRUE positions (null = positions, pd_pf_wp);
    // selected before the re-aim so the run offset is one unconditional
    // 64-bit add, never the `if (p) p += roff` shape the note above flags
    const unsigned int* wp = pd_pf_wp(positions, win_pos);
    uint32_t run_row0 = 0;
    if (run_offs != nullptr) {
        const uint32_t roff = run_offs[blockIdx.z];
        n_rows = run_offs[blockIdx.z + 1u] - roff;
        q += (size_t)roff * n_heads * HD;
        out += (size_t)roff * n_heads * HD;
        positions += roff;
        wp += roff;
        run_row0 = roff;
    }
    const uint32_t kvh = blockIdx.x;
    const uint32_t row0 = blockIdx.y * TQ;
    const uint32_t d = threadIdx.x, warp = d >> 5, lane = d & 31u;

    constexpr bool F8 = sizeof(KVT) == 1u;             // e4m3 pools (PIPE arm)
    extern __shared__ __align__(16) unsigned char pf4_smraw[];
    __half* sh_q = (__half*)pf4_smraw;                  // [MR][row_e]
    __half* sh_kv = sh_q + (size_t)MR * row_e;          // [2][K,V][TK][row_e]
    // F8 only: raw e4m3 tiles, [2][K,V][TK][HD] bytes (16B-aligned: MR%16==0
    // and TK*row_e*2 are both 16-multiples)
    unsigned char* sh_raw = (unsigned char*)(sh_kv + (size_t)4u * TK * row_e);
    float* sh_s = (float*)(sh_raw + (F8 ? 4u * (size_t)TK * HD : 0u));  // [MR][t_s]
    __half* sh_p = (__half*)(sh_s + (size_t)MR * t_s);  // [MR][t_s]
    float* sh_corr = (float*)(sh_p + (size_t)MR * t_s); // [MR]
    float* sh_onorm = sh_corr + MR;                     // [MR]
    unsigned int* sh_pos = (unsigned int*)(sh_onorm + MR); // [TQ]
    __shared__ unsigned int sh_wpos[TQ];                 // true positions (floors)

    // Q stage (scaled): row r = token j * G + head g
    for (uint32_t i = d; i < MR * row_e; i += 256u) {
        const uint32_t r = i / row_e, c = i % row_e;
        const uint32_t j = r / G, g = r % G, b = row0 + j;
        float v = 0.f;
        if (b < n_rows && c < HD)
            v = q[((size_t)b * n_heads + kvh * G + g) * HD + c] * scale;
        sh_q[(size_t)r * row_e + c] = __float2half(v);
    }
    if (d < TQ) {
        sh_pos[d] = (row0 + d) < n_rows ? positions[row0 + d] : 0u;
        sh_wpos[d] = (row0 + d) < n_rows ? wp[row0 + d] : 0u;
    }
    for (uint32_t i = d; i < MR * t_s; i += 256u) sh_p[i] = __half(0.f);
    __syncthreads();
    uint32_t hi = 0, lo = 0xFFFFFFFFu;
    #pragma unroll
    for (uint32_t j = 0; j < TQ; ++j) {
        if (row0 + j < n_rows) {
            hi = max(hi, sh_pos[j] + 1u);
            lo = min(lo, pd_pf_floor(sh_wpos[j], swa_window));
        }
    }
    if (hi == 0) return;                 // dead tile (all rows past n_rows)
    const uint32_t lo_t = (lo == 0xFFFFFFFFu ? 0u : lo) / TK * TK;

    // chunk = one sequence (run_row0 = the run's first row under the runs
    // arm, 0 otherwise - see the prologue note)
    const uint32_t slot = slots ? slots[run_row0] : 0u;
    const uint32_t* bt = block_tables + (size_t)slot * blocks_per_slot;

    // per-row softmax state: warp w owns rows w*RPW..w*RPW+RPW-1, one row
    // per 4-lane quad (8 quads/warp) - RPW<=8 (every shipped shape before
    // G=9) covers all of a warp's rows in one quad-pass; RPW>8 (G=9, hd128:
    // MR=144/NW=8=18) needs SUBW=ceil(RPW/8) sequential quad-passes per
    // warp, hence m_st/l_st are SUBW-sized (SUBW=1 reproduces the original
    // single-pass code exactly for every RPW<=8 shape).
    constexpr uint32_t RPW = MR / NW;
    constexpr uint32_t SUBW = (RPW + 7u) / 8u;
    float m_st[SUBW], l_st[SUBW];
    #pragma unroll
    for (uint32_t s = 0; s < SUBW; ++s) { m_st[s] = -1e30f; l_st[s] = 0.f; }

    float o_acc[NRF][SLICE / 8u][4];
    #pragma unroll
    for (uint32_t a = 0; a < NRF; ++a)
        #pragma unroll
        for (uint32_t s2 = 0; s2 < SLICE / 8u; ++s2)
            #pragma unroll
            for (uint32_t k = 0; k < 4u; ++k) o_acc[a][s2][k] = 0.f;

    constexpr uint32_t lines = F8 ? HD >> 4 : (HD * 2u) >> 4;
    auto stage = [&](uint32_t bf, uint32_t t0) {
        const uint32_t n_t = hi - t0 < TK ? hi - t0 : TK;
        // zero tail rows: PV reads all TK V rows; 0-weight x Inf = NaN.
        // F8 zero-fills at EXPAND time instead (stage-time zeros into the
        // half layout would be overwritten by the expand anyway).
        if constexpr (!F8) {
            if (n_t < TK) {
                for (uint32_t i = d; i < 2u * (TK - n_t) * lines; i += 256u) {
                    const uint32_t kvsel = i / ((TK - n_t) * lines);
                    const uint32_t jj = i - kvsel * (TK - n_t) * lines;
                    const uint32_t p = n_t + jj / lines, l = jj % lines;
                    *(uint4*)((char*)(sh_kv
                        + ((size_t)(bf * 2u + kvsel) * TK + p) * row_e) + l * 16u)
                        = make_uint4(0u, 0u, 0u, 0u);
                }
            }
        }
        const uint32_t rows = 2u * n_t;
        const uint32_t gsz = 256u / rows ? 256u / rows : 1u;
        const uint32_t r = d / gsz, lt = d % gsz;
        if (r < rows) {
            const uint32_t kvsel = r >= n_t ? 1u : 0u;
            const uint32_t p = kvsel ? r - n_t : r;
            const uint32_t gpos = t0 + p;
            const uint32_t blk = bt[gpos >> 4];
            const char* src = (const char*)((kvsel ? pool_v : pool_k)
                + (size_t)blk * 16u * kv_dim + (size_t)(gpos & 15u) * kv_dim
                + (size_t)kvh * HD);
            // F8: raw e4m3 row into sh_raw[bf]; else the half row direct
            char* dst = F8
                ? (char*)sh_raw + ((size_t)(bf * 2u + kvsel) * TK + p) * HD
                : (char*)(sh_kv + ((size_t)(bf * 2u + kvsel) * TK + p) * row_e);
            #pragma unroll
            for (uint32_t l = lt; l < lines; l += gsz)
                pd_attn_cpa16(dst + l * 16u, src + l * 16u);
        }
        pd_attn_cpa_commit();
    };

    stage(0u, lo_t);
    uint32_t bf = 0;
    for (uint32_t t0 = lo_t; t0 < hi; t0 += TK, bf ^= 1u) {
        const bool more = t0 + TK < hi;
        if (more) stage(bf ^ 1u, t0 + TK);
        if (more) pd_attn_cpa_wait1(); else pd_attn_cpa_wait0();
        __syncthreads();
        const __half* kbuf = sh_kv + (size_t)(bf * 2u) * TK * row_e;
        const __half* vbuf = sh_kv + ((size_t)(bf * 2u) + 1u) * TK * row_e;
        if constexpr (F8) {
            // raw(bf) landed - expand to the padded-row half layout with the
            // widened cvt pairs (v3c PIPE: identical contents to f16 direct
            // staging). Rows >= n_t zero-fill here (the PV NaN guard).
            const uint32_t n_t = hi - t0 < TK ? hi - t0 : TK;
            constexpr uint32_t SPAN8 = TK * (HD / 8u);
            const unsigned char* rbase = sh_raw + (size_t)(bf * 2u) * TK * HD;
            for (uint32_t u = d; u < 2u * SPAN8; u += 256u) {
                const bool isv = u >= SPAN8;
                const uint32_t ur = isv ? u - SPAN8 : u;
                const uint32_t kk = ur / (HD / 8u), d8 = (ur % (HD / 8u)) * 8u;
                __half* dst = sh_kv
                    + ((size_t)(bf * 2u + (isv ? 1u : 0u)) * TK + kk) * row_e + d8;
                if (kk < n_t) {
                    const uint2 raw = *reinterpret_cast<const uint2*>(
                        rbase + ((size_t)(isv ? TK : 0u) + kk) * HD + d8);
                    const unsigned short* p16 = (const unsigned short*)&raw;
                    #pragma unroll
                    for (uint32_t j = 0; j < 4u; ++j)
                        *reinterpret_cast<__half2*>(dst + 2u * j) =
                            __half2(__nv_cvt_fp8x2_to_halfraw2(p16[j], __NV_E4M3));
                } else {
                    *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
                }
            }
            __syncthreads();   // expanded halves visible before the mmas
        }
        // scores: NRF row-frags x TK/8 col-subtiles, split over 8 warps
        {
            constexpr uint32_t NCS = TK / 8u;
            constexpr uint32_t TASKS = NRF * NCS;          // 16 or 4
            #pragma unroll
            for (uint32_t task = warp; task < TASKS; task += NW) {
                const uint32_t rf = task / NCS, cs = task % NCS;
                const uint32_t p0 = cs * 8u;
                float dfr[4] = {0.f, 0.f, 0.f, 0.f};
                for (uint32_t kk = 0; kk < HD; kk += 16u) {
                    uint32_t af[4];
                    const __half* ap = sh_q + (size_t)(rf * 16u + (lane & 15u)) * row_e
                                     + kk + ((lane >> 4) ? 8u : 0u);
                    pd_ldm_x4(af, (const unsigned char*)ap);
                    uint32_t bfr[2];
                    const __half* bp = kbuf + (size_t)(p0 + (lane & 7u)) * row_e
                                     + kk + (((lane >> 3) & 1u) ? 8u : 0u);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                                 : "=r"(bfr[0]), "=r"(bfr[1])
                                 : "r"((unsigned)__cvta_generic_to_shared(bp)));
                    pd_fa_mma16(dfr, af[0], af[1], af[2], af[3], bfr[0], bfr[1]);
                }
                #pragma unroll
                for (uint32_t half = 0; half < 2u; ++half) {
                    const uint32_t rr = rf * 16u + (lane >> 2) + half * 8u;
                    #pragma unroll
                    for (uint32_t cc = 0; cc < 2u; ++cc) {
                        const uint32_t pp = p0 + 2u * (lane & 3u) + cc;
                        sh_s[(size_t)rr * t_s + pp] = dfr[half * 2u + cc];
                    }
                }
            }
        }
        __syncthreads();
        // softmax: warp w owns rows w*RPW..; lane quad per row, 8 cols/lane;
        // SUBW passes when RPW>8 (see m_st/l_st's declaration above)
        #pragma unroll
        for (uint32_t sub = 0; sub < SUBW; ++sub) {
            const uint32_t rloc = sub * 8u + (lane >> 2);
            const bool rvalid = rloc < RPW;
            const uint32_t r = warp * RPW + (rvalid ? rloc : 0u);
            const uint32_t j = r / G;
            const uint32_t pos = sh_pos[j];
            const uint32_t wpos = sh_wpos[j];
            const uint32_t c0 = (lane & 3u) * (TK / 4u);
            float w8[TK / 4u];
            float mx = -1e30f;
            #pragma unroll
            for (uint32_t c = 0; c < TK / 4u; ++c) {
                const uint32_t p = t0 + c0 + c;
                const bool valid = p < hi && p <= pos
                    && (!swa_window || p + swa_window > wpos);
                w8[c] = valid ? sh_s[(size_t)r * t_s + c0 + c] : -1e30f;
                mx = fmaxf(mx, w8[c]);
            }
            #pragma unroll
            for (uint32_t off = 1; off <= 2; off <<= 1)
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
            const float m_new = fmaxf(m_st[sub], mx);
            const float corr = (m_st[sub] > -1e29f) ? __expf(m_st[sub] - m_new) : 0.f;
            float ps = 0.f;
            #pragma unroll
            for (uint32_t c = 0; c < TK / 4u; ++c) {
                const float w = w8[c] > -1e29f ? __expf(w8[c] - m_new) : 0.f;
                ps += w;
                sh_p[(size_t)r * t_s + c0 + c] = __float2half(w);
            }
            #pragma unroll
            for (uint32_t off = 1; off <= 2; off <<= 1)
                ps += __shfl_xor_sync(0xffffffffu, ps, off);
            l_st[sub] = l_st[sub] * corr + ps;
            m_st[sub] = m_new;
            if ((lane & 3u) == 0 && rvalid) sh_corr[r] = corr;
        }
        __syncthreads();
        // PV: warp w owns dims [w*SLICE, +SLICE); P rows via ldm_x4
        {
            const uint32_t n_base = warp * SLICE;
            #pragma unroll
            for (uint32_t a = 0; a < NRF; ++a) {
                #pragma unroll
                for (uint32_t half = 0; half < 2u; ++half) {
                    const uint32_t rr = a * 16u + (lane >> 2) + half * 8u;
                    const float corr = sh_corr[rr];
                    #pragma unroll
                    for (uint32_t s2 = 0; s2 < SLICE / 8u; ++s2) {
                        o_acc[a][s2][half * 2u] *= corr;
                        o_acc[a][s2][half * 2u + 1u] *= corr;
                    }
                }
                #pragma unroll
                for (uint32_t kt = 0; kt < TK; kt += 16u) {
                    uint32_t af[4];
                    const __half* ap = sh_p + (size_t)(a * 16u + (lane & 15u)) * t_s
                                     + kt + ((lane >> 4) ? 8u : 0u);
                    pd_ldm_x4(af, (const unsigned char*)ap);
                    #pragma unroll
                    for (uint32_t s2 = 0; s2 < SLICE / 8u; ++s2) {
                        uint32_t bfr[2];
                        const __half* bp = vbuf + (size_t)(kt + (lane & 15u)) * row_e
                                         + n_base + s2 * 8u;
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                                     : "=r"(bfr[0]), "=r"(bfr[1])
                                     : "r"((unsigned)__cvta_generic_to_shared(bp)));
                        pd_fa_mma16(o_acc[a][s2], af[0], af[1], af[2], af[3],
                                    bfr[0], bfr[1]);
                    }
                }
            }
        }
        __syncthreads();
    }
    // sink fold + 1/l publish (quad-lane 0 of each softmax row); SUBW
    // passes mirroring the softmax loop above
    #pragma unroll
    for (uint32_t sub = 0; sub < SUBW; ++sub) {
        const uint32_t rloc = sub * 8u + (lane >> 2);
        const bool rvalid = rloc < RPW;
        const uint32_t r = warp * RPW + (rvalid ? rloc : 0u);
        if ((lane & 3u) == 0 && rvalid) {
            const uint32_t g = r % G;
            const float s = sinks[kvh * G + g];
            const float mt = fmaxf(m_st[sub], s);
            const float dm = m_st[sub] - mt, ds = s - mt;
            const float cm = dm >= -20.f ? __expf(dm) : 0.f;
            const float cs = ds >= -20.f ? __expf(ds) : 0.f;
            const float l = l_st[sub] * cm + cs;
            sh_onorm[r] = l > 0.f ? cm / l : 0.f;
        }
    }
    __syncthreads();
    // O write straight from frags
    {
        const uint32_t n_base = warp * SLICE;
        #pragma unroll
        for (uint32_t a = 0; a < NRF; ++a)
            #pragma unroll
            for (uint32_t s2 = 0; s2 < SLICE / 8u; ++s2)
                #pragma unroll
                for (uint32_t half = 0; half < 2u; ++half) {
                    const uint32_t rr = a * 16u + (lane >> 2) + half * 8u;
                    const uint32_t j = rr / G, g = rr % G, b = row0 + j;
                    if (b >= n_rows) continue;
                    const float sc = sh_onorm[rr];
                    float* dst = out + ((size_t)b * n_heads + kvh * G + g) * HD
                               + n_base + s2 * 8u + 2u * (lane & 3u);
                    dst[0] = o_acc[a][s2][half * 2u] * sc;
                    dst[1] = o_acc[a][s2][half * 2u + 1u] * sc;
                }
    }
#else
    (void)q; (void)pool_k; (void)pool_v; (void)sinks; (void)out;
    (void)positions; (void)block_tables; (void)blocks_per_slot; (void)n_heads;
    (void)kv_dim; (void)swa_window; (void)n_rows; (void)scale; (void)slots;
    (void)run_offs;
#endif
}

// pf7 (attention front): fa2-class register-resident
// prefill tile for the hd256 fp8-KV shapes - the class flashinfer runs far
// faster than our v4 did per prefill row. Three structural moves
// vs v4 (algorithm study: flashinfer's fa2 prefill - original
// implementation, layouts re-derived and oracle-pinned):
//  1. Warp-owned rows, S/P register-resident: each of 4 warps owns 16 flat
//     head-rows; scores, online softmax, and the P operand never touch
//     smem (v4 round-trips sh_s AND sh_p every 16-key tile). 2 barriers
//     per 64-key tile vs v4's 4 per 16 keys.
//  2. Raw-fp8-resident KV: no expanded-f16 KV smem, no smem->smem expand
//     pass. B-fragments come from half-width b16 ldmatrix on the packed
//     bytes + a register byte-swizzle (shfl_xor+byte_perm; K 2-step, V
//     .trans 3-step + frag swap) + fp8x2->half2 cvt. This is the third
//     fragment path the "NO byte-granular ldmatrix on sm_120" fact left
//     open: ldmatrix the bytes as b16, fix ownership in registers.
//  3. Flat (token,head) GQA row packing: rows = token*G + g, tiled by 64
//     with no MR%G constraint (tokens may split across CTAs) - the same
//     packing that lets the proto ladder run TK=64 (elected: -37% vs the
//     v4 fp8 arm at fresh-2048, -57% at the starved 384-row continuation
//     shape, -40% at 2048@6144; TK=80 flat, TK=32/48 worse; 243 regs,
//     LOCAL:0, 1 CTA/SM at 86.3KB smem).
// K double-buffered, V single-buffered (the smem budget: Q 64x264 halves
// + 3 x 64x272B raw = 86,272B); V[i+1] stages after the post-PV barrier
// and its latency hides under S[i+1]. Numeric class: online-softmax
// regroup vs v4 (wider tile), the v3c precedent - serve-gated, oracle
// maxrel at v4's own level in the proto (0.0040 vs v4's 0.0043 fresh-2048).
// fp8 pools only; f16 pools keep the v4 arm. Kill: PADDOCK_NO_PF7 -> v4.
__device__ __forceinline__ uint32_t pd_pf7_swz(uint32_t x) {
    uint32_t t = __shfl_xor_sync(0xffffffffu, x, 1);
    x = __byte_perm(x, t, (threadIdx.x & 1u) ? 0x3276u : 0x5410u);
    t = __shfl_xor_sync(0xffffffffu, x, 2);
    x = __byte_perm(x, t, (threadIdx.x & 2u) ? 0x3276u : 0x5410u);
    return x;
}
__device__ __forceinline__ uint32_t pd_pf7_swzt(uint32_t x) {
    uint32_t t = __shfl_xor_sync(0xffffffffu, x, 4);
    x = __byte_perm(x, t, (threadIdx.x & 4u) ? 0x3175u : 0x6420u);
    t = __shfl_xor_sync(0xffffffffu, x, 8);
    x = __byte_perm(x, t, (threadIdx.x & 8u) ? 0x3276u : 0x5410u);
    t = __shfl_xor_sync(0xffffffffu, x, 16);
    x = __byte_perm(x, t, (threadIdx.x & 16u) ? 0x3276u : 0x5410u);
    return x;
}
__device__ __forceinline__ uint32_t pd_pf7_cvt2(unsigned short u) {
    const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(u, __NV_E4M3);
    return (uint32_t)h.x | ((uint32_t)h.y << 16);
}
// .trans twin of pd_ldm_x4 (pf7rp's PV B-frags; hoisted out of the mma loop
// by the software pipeline, so it needs a callable form)
__device__ __forceinline__ void pd_pf7_ldmt(uint32_t r[4], const __half* p) {
    const unsigned sp = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
                 "{%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(sp));
}

// VL arm (AF3): varlen tile packing - One launch covers every
// eligible prefill span of the tick. vl_items is stride-4 per Y-tile:
// (q_row0, span_rows, tile_flat_row0, slot). Tiles never cross spans, so
// each packed CTA computes bit-identically to its per-span twin - only the
// grid packing changes (kills the per-span small-launch band).
// NW (warps per CTA) - the q-tile WIDTH knob. Each warp owns 16
// (row,head) units regardless, so NW scales MR without touching a single line
// of the per-row math: warp w still walks rows [16w, 16w+16) over the same TK
// tiles in the same order, so every row's online-softmax fold sequence is
// identical and NW=8 is bit-exact against NW=4.
//
// Why it exists: profiling the granite 2048-row prefill put pf7 at grid 1024
// where FlashInfer needs 512 for the same work - our q-tile was half theirs,
// so every K/V tile got staged and fp8->f16 converted twice as many times
// across the grid. That is ~1.65x the L1 traffic for identical math, visible
// as much higher L1/TEX and compute-pipe utilisation at identical occupancy.
// K/V staging
// work is (q-tiles x span); NW=8 halves the q-tiles and therefore halves it.
// Warps/SM is UNCHANGED: NW=4 runs 2 CTA/SM x 4 warps, NW=8 runs 1 x 8.
// SUB (row sub-tiles per warp) is the knob NW turned out not to
// be. NW=8 halved the grid as intended and was bit-exact -- and ran 30%
// slower, because MR=128 at TK=64 costs 63 KB and drops the kernel to 1
// CTA/SM, and this pipeline needs the second CTA to cover its __syncthreads.
// FlashInfer gets the wide tile AND keeps 2 CTA/SM by spending REGISTERS
// instead of warps: 128 threads / 4 warps / 255 regs /
// 49152 B, i.e. a 128-row q tile with 32 rows per warp. pf7 sat at 158 regs
// with ~98/thread of headroom unused. SUB=2 buys exactly that: each warp
// carries two 16-row sub-tiles of softmax state and o_acc, the K/V tile is
// staged once and consumed by both, and TKT=32 keeps smem at 49152 so 2
// CTA/SM survives. SUB changes the softmax TILE granularity when it comes
// with a TK change, so <4,2,32> is a numerics-class change vs <4,1,64>, not
// pure code motion -- parity gates and serve acceptance arbitrate.
template <uint32_t HD, uint32_t G, bool VL = false, uint32_t NW = 4u,
          uint32_t SUB = 1u, uint32_t TKT = 64u>
__global__ void __launch_bounds__(NW * 32u) pd_attn_prefill_pf7_kernel(
    const float* __restrict__ q, const unsigned char* __restrict__ pool_k,
    const unsigned char* __restrict__ pool_v, const float* __restrict__ sinks,
    float* __restrict__ out, const unsigned int* __restrict__ positions,
    const uint32_t* __restrict__ block_tables, uint32_t blocks_per_slot,
    const unsigned int* __restrict__ slots, uint32_t n_heads, uint32_t kv_dim,
    uint32_t swa_window, uint32_t n_rows, float scale,
    const uint32_t* __restrict__ vl_items, const uint32_t* __restrict__ run_offs,
    const unsigned int* __restrict__ win_pos) {
#if PD_FA_OK
    // HD templated: the raw-fp8-resident convert-in-register tile
    // generalizes from its hd256 origin to hd128 (granite imax) unchanged --
    // every loop is HD/16, the byte-swizzle + fp8x2->half2 cvt are HD-agnostic.
    // hd128 halves QROW/KROW/o_acc => ~45 KB smem + ~180 regs => 2 CTA/SM,
    // vs the v4 fp8 arm's 1-CTA/SM expand-pass tile. Kill: PADDOCK_NO_PF7.
    constexpr uint32_t NTH = NW * 32u;
    constexpr uint32_t MR = NW * 16u * SUB, TK = TKT, NKC = TK / 16u;
    constexpr uint32_t QROW = HD + 8u;     // sh_q halves per row (+8 pad)
    constexpr uint32_t KROW = HD + 16u;    // raw row bytes: 16B-aligned pad
                                           // spreads 8 ldmatrix row phases
    const uint32_t kvh = blockIdx.x;
    uint32_t row0, qr0 = 0, span_rows = n_rows;
    if constexpr (VL) {
        const uint32_t* it = vl_items + (size_t)blockIdx.y * 4u;
        qr0 = it[0]; span_rows = it[1]; row0 = it[2];
    } else if (run_offs != nullptr) {
        // multi-run (pf_runs) arm: blockIdx.z selects the run,
        // the same contiguous [off, off+len) partition v4's g4n consumes, so
        // each packed CTA computes its per-run twin bit-for-bit. Tiles past a
        // short run skip via the Rtot guards. row0 tiles within the run
        // (query rows = tokens*G, flat-packed).
        qr0 = run_offs[blockIdx.z];
        span_rows = run_offs[blockIdx.z + 1u] - qr0;
        row0 = blockIdx.y * MR;
    } else {
        row0 = blockIdx.y * MR;
    }
    const uint32_t Rtot = span_rows * G;
    // window floors from the true positions (null = positions; pd_pf_wp),
    // indexed exactly as `positions` is below - never re-aimed
    const unsigned int* wp = pd_pf_wp(positions, win_pos);
    const uint32_t d = threadIdx.x, warp = d >> 5, lane = d & 31u;

    extern __shared__ __align__(16) unsigned char pf7_sm[];
    __half* sh_q = (__half*)pf7_sm;
    unsigned char* kraw = pf7_sm + (size_t)MR * QROW * 2u;
    unsigned char* vraw = kraw + 2u * (size_t)TK * KROW;
    unsigned int* sh_rpos = (unsigned int*)(vraw + (size_t)TK * KROW);
    __shared__ unsigned int sh_rwin[MR];   // per-row true positions (floors)

    // Q stage (scaled f32 -> f16) + per-row positions
    for (uint32_t i = d; i < MR * QROW; i += NTH) {
        const uint32_t r = i / QROW, c = i % QROW;
        const uint32_t R = row0 + r;
        float v = 0.f;
        if (R < Rtot && c < HD)
            v = q[((size_t)(qr0 + R / G) * n_heads + kvh * G + R % G) * HD + c]
                * scale;
        sh_q[i] = __float2half(v);
    }
    if (d < MR) {
        sh_rpos[d] =
            (row0 + d) < Rtot ? positions[qr0 + (row0 + d) / G] : 0xFFFFFFFFu;
        sh_rwin[d] =
            (row0 + d) < Rtot ? wp[qr0 + (row0 + d) / G] : 0xFFFFFFFFu;
    }
    __syncthreads();

    uint32_t hi = 0, lo = 0xFFFFFFFFu;
    for (uint32_t r = 0; r < MR; ++r) {
        const uint32_t p = sh_rpos[r];
        if (p != 0xFFFFFFFFu) {
            hi = max(hi, p + 1u);
            lo = min(lo, pd_pf_floor(sh_rwin[r], swa_window));
        }
    }
    if (hi == 0u) return;
    const uint32_t lo_t = (lo == 0xFFFFFFFFu ? 0u : lo) / TK * TK;
    // slot: VL carries it per tile; the run arm reads the run's own slot at
    // its row start (each run is a distinct sequence); single-run is slots[0].
    const uint32_t slot = VL ? vl_items[(size_t)blockIdx.y * 4u + 3u]
                             : (run_offs ? (slots ? slots[qr0] : 0u)
                                         : (slots ? slots[0] : 0u));
    const uint32_t* bt = block_tables + (size_t)slot * blocks_per_slot;

    // Row identity is now per sub-tile: warp w owns rows
    // [16*SUB*w, 16*SUB*(w+1)), split into SUB m16 fragments.
    uint32_t r_lo[SUB], r_hi[SUB], pos_lo[SUB], pos_hi[SUB], win_lo[SUB], win_hi[SUB];
    #pragma unroll
    for (uint32_t sb = 0; sb < SUB; ++sb) {
        r_lo[sb] = (warp * SUB + sb) * 16u + (lane >> 2);
        r_hi[sb] = r_lo[sb] + 8u;
        pos_lo[sb] = sh_rpos[r_lo[sb]];
        pos_hi[sb] = sh_rpos[r_hi[sb]];
        win_lo[sb] = sh_rwin[r_lo[sb]];
        win_hi[sb] = sh_rwin[r_hi[sb]];
    }
    const uint32_t c2 = 2u * (lane & 3u);

    // raw K/V tile stage: cp.async 16B lines; rows past hi zero-fill
    // (stale e4m3 can be NaN; PV's 0-weight x NaN would poison O)
    auto stage_one = [&](unsigned char* dstbuf, const unsigned char* pool,
                         uint32_t t0) {
        const uint32_t n_t = hi - t0 < TK ? hi - t0 : TK;
        for (uint32_t u = d; u < TK * (HD / 16u); u += NTH) {
            const uint32_t p = u / (HD / 16u), l = u % (HD / 16u);
            unsigned char* dst = dstbuf + (size_t)p * KROW + l * 16u;
            if (p < n_t) {
                const uint32_t gpos = t0 + p;
                pd_attn_cpa16(dst, pool + (size_t)bt[gpos >> 4] * 16u * kv_dim
                    + (size_t)(gpos & 15u) * kv_dim + (size_t)kvh * HD + l * 16u);
            } else {
                *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        pd_attn_cpa_commit();
    };

    float m_lo[SUB], m_hi[SUB], l_lo[SUB], l_hi[SUB];
    float o_acc[SUB][HD / 16u][8];
    #pragma unroll
    for (uint32_t sb = 0; sb < SUB; ++sb) {
        m_lo[sb] = -1e30f; m_hi[sb] = -1e30f; l_lo[sb] = 0.f; l_hi[sb] = 0.f;
        #pragma unroll
        for (uint32_t dc = 0; dc < HD / 16u; ++dc)
            #pragma unroll
            for (uint32_t i = 0; i < 8u; ++i) o_acc[sb][dc][i] = 0.f;
    }

    stage_one(kraw, pool_k, lo_t);
    stage_one(vraw, pool_v, lo_t);
    uint32_t bf = 0;
    for (uint32_t t0 = lo_t; t0 < hi; t0 += TK, bf ^= 1u) {
        const bool more = t0 + TK < hi;
        if (more) stage_one(kraw + (bf ^ 1u) * (size_t)TK * KROW, pool_k, t0 + TK);
        // pending: {V[i] (committed at the last iter's tail), K[i+1]} -
        // wait1 clears V[i] (and the long-retired K[i])
        if (more) pd_attn_cpa_wait1(); else pd_attn_cpa_wait0();
        __syncthreads();

        // The K/V tile is now staged once per CTA and consumed by every row
        // sub-tile below - that reuse is the lever (profiled: pf7's
        // half-width q tile doubled the grid, so each K/V tile was staged
        // + fp8->f16 converted twice as often for identical math).
        const unsigned char* kbuf_s = kraw + bf * (size_t)TK * KROW;
        #pragma unroll
        for (uint32_t sb = 0; sb < SUB; ++sb) {
        // scores: warp-owned m16 x TK, S stays in registers
        float s[NKC][8];
        #pragma unroll
        for (uint32_t kc = 0; kc < NKC; ++kc)
            #pragma unroll
            for (uint32_t i = 0; i < 8u; ++i) s[kc][i] = 0.f;
        const unsigned char* kbuf = kbuf_s;
        #pragma unroll
        for (uint32_t dc = 0; dc < HD / 16u; ++dc) {
            uint32_t af[4];
            pd_ldm_x4(af, (const unsigned char*)(sh_q
                + (size_t)((warp * SUB + sb) * 16u + (lane & 15u)) * QROW
                + dc * 16u + ((lane >> 4) ? 8u : 0u)));
            #pragma unroll
            for (uint32_t kc = 0; kc < NKC; ++kc) {
                const unsigned char* kp = kbuf
                    + (size_t)(kc * 16u + (lane & 15u)) * KROW + dc * 16u;
                uint32_t qk[2];
                const unsigned sp = (unsigned)__cvta_generic_to_shared(kp);
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                    : "=r"(qk[0]), "=r"(qk[1]) : "r"(sp));
                qk[0] = pd_pf7_swz(qk[0]);
                qk[1] = pd_pf7_swz(qk[1]);
                pd_fa_mma16(&s[kc][0], af[0], af[1], af[2], af[3],
                            pd_pf7_cvt2((unsigned short)qk[0]),
                            pd_pf7_cvt2((unsigned short)(qk[0] >> 16)));
                pd_fa_mma16(&s[kc][4], af[0], af[1], af[2], af[3],
                            pd_pf7_cvt2((unsigned short)qk[1]),
                            pd_pf7_cvt2((unsigned short)(qk[1] >> 16)));
            }
        }

        // mask + online softmax, in registers; the lane quad covers the row
        float mx_lo = -1e30f, mx_hi = -1e30f;
        #pragma unroll
        for (uint32_t kc = 0; kc < NKC; ++kc) {
            const uint32_t k0 = t0 + kc * 16u;
            #pragma unroll
            for (uint32_t i = 0; i < 8u; ++i) {
                const uint32_t p = k0 + c2 + (i & 1u) + ((i >> 2) ? 8u : 0u);
                const bool hirow = (i & 2u) != 0u;
                const uint32_t pos = hirow ? pos_hi[sb] : pos_lo[sb];
                const uint32_t wpos = hirow ? win_hi[sb] : win_lo[sb];
                const bool valid = p < hi && p <= pos
                    && (!swa_window || p + swa_window > wpos);
                const float v = valid ? s[kc][i] : -1e30f;
                s[kc][i] = v;
                if (hirow) mx_hi = fmaxf(mx_hi, v); else mx_lo = fmaxf(mx_lo, v);
            }
        }
        #pragma unroll
        for (uint32_t off = 1; off <= 2; off <<= 1) {
            mx_lo = fmaxf(mx_lo, __shfl_xor_sync(0xffffffffu, mx_lo, off));
            mx_hi = fmaxf(mx_hi, __shfl_xor_sync(0xffffffffu, mx_hi, off));
        }
        const float mn_lo = fmaxf(m_lo[sb], mx_lo), mn_hi = fmaxf(m_hi[sb], mx_hi);
        const float corr_lo = (m_lo[sb] > -1e29f) ? __expf(m_lo[sb] - mn_lo) : 0.f;
        const float corr_hi = (m_hi[sb] > -1e29f) ? __expf(m_hi[sb] - mn_hi) : 0.f;
        float ps_lo = 0.f, ps_hi = 0.f;
        uint32_t pf[NKC][4];
        #pragma unroll
        for (uint32_t kc = 0; kc < NKC; ++kc) {
            #pragma unroll
            for (uint32_t i = 0; i < 8u; ++i) {
                const bool hirow = (i & 2u) != 0u;
                const float w = s[kc][i] > -1e29f
                    ? __expf(s[kc][i] - (hirow ? mn_hi : mn_lo)) : 0.f;
                s[kc][i] = w;
                if (hirow) ps_hi += w; else ps_lo += w;
            }
            pf[kc][0] = (uint32_t)__half_as_ushort(__float2half(s[kc][0]))
                | ((uint32_t)__half_as_ushort(__float2half(s[kc][1])) << 16);
            pf[kc][1] = (uint32_t)__half_as_ushort(__float2half(s[kc][2]))
                | ((uint32_t)__half_as_ushort(__float2half(s[kc][3])) << 16);
            pf[kc][2] = (uint32_t)__half_as_ushort(__float2half(s[kc][4]))
                | ((uint32_t)__half_as_ushort(__float2half(s[kc][5])) << 16);
            pf[kc][3] = (uint32_t)__half_as_ushort(__float2half(s[kc][6]))
                | ((uint32_t)__half_as_ushort(__float2half(s[kc][7])) << 16);
        }
        #pragma unroll
        for (uint32_t off = 1; off <= 2; off <<= 1) {
            ps_lo += __shfl_xor_sync(0xffffffffu, ps_lo, off);
            ps_hi += __shfl_xor_sync(0xffffffffu, ps_hi, off);
        }
        l_lo[sb] = l_lo[sb] * corr_lo + ps_lo; m_lo[sb] = mn_lo;
        l_hi[sb] = l_hi[sb] * corr_hi + ps_hi; m_hi[sb] = mn_hi;

        // O rescale + PV (V raw fp8, trans frag build in registers)
        #pragma unroll
        for (uint32_t dc = 0; dc < HD / 16u; ++dc) {
            o_acc[sb][dc][0] *= corr_lo; o_acc[sb][dc][1] *= corr_lo;
            o_acc[sb][dc][2] *= corr_hi; o_acc[sb][dc][3] *= corr_hi;
            o_acc[sb][dc][4] *= corr_lo; o_acc[sb][dc][5] *= corr_lo;
            o_acc[sb][dc][6] *= corr_hi; o_acc[sb][dc][7] *= corr_hi;
        }
        #pragma unroll
        for (uint32_t dc = 0; dc < HD / 16u; ++dc) {
            #pragma unroll
            for (uint32_t kc = 0; kc < NKC; ++kc) {
                const unsigned char* vp = vraw
                    + (size_t)(kc * 16u + (lane & 15u)) * KROW + dc * 16u;
                uint32_t qv[2];
                const unsigned sp = (unsigned)__cvta_generic_to_shared(vp);
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                    : "=r"(qv[0]), "=r"(qv[1]) : "r"(sp));
                qv[0] = pd_pf7_swzt(qv[0]);
                qv[1] = pd_pf7_swzt(qv[1]);
                pd_fa_mma16(&o_acc[sb][dc][0], pf[kc][0], pf[kc][1], pf[kc][2],
                            pf[kc][3], pd_pf7_cvt2((unsigned short)qv[0]),
                            pd_pf7_cvt2((unsigned short)qv[1]));
                pd_fa_mma16(&o_acc[sb][dc][4], pf[kc][0], pf[kc][1], pf[kc][2],
                            pf[kc][3], pd_pf7_cvt2((unsigned short)(qv[0] >> 16)),
                            pd_pf7_cvt2((unsigned short)(qv[1] >> 16)));
            }
        }
        }  // sub-tile loop
        __syncthreads();       // buffer-reuse fence before the next stage
        if (more) stage_one(vraw, pool_v, t0 + TK);
    }

    // epilogue: sink merge + normalize + direct f32 stores, per lane 2 rows
    // per sub-tile (SUB=1 reduces to the original 2-row form exactly)
    #pragma unroll
    for (uint32_t sb = 0; sb < SUB; ++sb)
    #pragma unroll
    for (uint32_t half = 0; half < 2u; ++half) {
        const uint32_t R = row0 + (half ? r_hi[sb] : r_lo[sb]);
        if (R >= Rtot) continue;
        const uint32_t h = kvh * G + R % G;
        const float mm = half ? m_hi[sb] : m_lo[sb];
        const float ll = half ? l_hi[sb] : l_lo[sb];
        const float sv = sinks[h];
        const float mtot = fmaxf(mm, sv);
        const float cm = __expf(mm - mtot), cs2 = __expf(sv - mtot);
        const float l = ll * cm + cs2;
        const float nrm = l > 0.f ? cm / l : 0.f;
        float* op = out + ((size_t)(qr0 + R / G) * n_heads + h) * HD;
        #pragma unroll
        for (uint32_t dc = 0; dc < HD / 16u; ++dc) {
            const uint32_t i0 = half ? 2u : 0u;
            *reinterpret_cast<float2*>(op + dc * 16u + c2) =
                make_float2(o_acc[sb][dc][i0] * nrm, o_acc[sb][dc][i0 + 1u] * nrm);
            *reinterpret_cast<float2*>(op + dc * 16u + 8u + c2) =
                make_float2(o_acc[sb][dc][i0 + 4u] * nrm, o_acc[sb][dc][i0 + 5u] * nrm);
        }
    }
#else
    (void)q; (void)pool_k; (void)pool_v; (void)sinks; (void)out;
    (void)positions; (void)block_tables; (void)blocks_per_slot; (void)slots;
    (void)n_heads; (void)kv_dim; (void)swa_window; (void)n_rows; (void)scale;
    (void)vl_items;
#endif
}

// pf7rp (door 2): pf7 with a REPACKED f16 KV pane -
// the flashinfer structure (their hd256/e4m3 traits:
// CTA_TILE_Q=64, 4 q-warps x 1 kv-warp, REG:250-255, USE_KV_REPACK, one
// bf16 repack pane reused K-then-V per tile, raw fp8 K/V single-buffered;
// re-derived and re-implemented, not copied). pf7's per-fragment byte
// swizzle (2-3 shfl + 2-3 prmt + 2 cvt per B-frag, inside the mma loop)
// becomes one bulk fp8->f16 repack per tile, and the mma loop runs clean
// b16 ldmatrix: QK-B = x4 NON-trans on the row-major pane (the m16n8k16
// col-major B-frag is the non-trans tile layout when smem rows are the n
// dim); PV-B = x4.trans. Operand halves are the same cvt on the same
// bytes in the same mma order as pf7 -> output is BIT-IDENTICAL to pf7
// (word-compare gated). smem 100,608 B: Q
// 64x264h + pane 64x264h + raw K/V 64x256B each -> still 1 CTA/SM; the
// win is per-warp instruction density, not occupancy. Barriers 4/tile
// (vs pf7's 2): K-repack and V-repack each need a visibility fence, but
// K[i+1]/V[i+1] cp.async overlap QK/PV through the freed raw panes.
// HD/TK templated: generalized from the hd256 original to serve
// granite's hd128 prefill. The body was already written in HD/QROW/RROW/MR/TK
// throughout -- only the constexpr header, the repack group width and the
// NKC=4 max tree were shape-locked. hd128 cannot take TK=64: at MR=64,
// QROW=HD+8, RROW=HD the tile costs 51,456 B and misses 2 CTA/SM by 256
// BYTES, leaving 4 warps/SM against pf7's 8 -- and the pf7 NW=8 experiment
// already measured what 1 CTA/SM does to this pipeline (+30%). TK=48 lands
// at 43,008 B and holds 2 CTA/SM. TK moves the online-softmax tile boundary,
// so hd128 is a NUMERICS-CLASS change vs pf7 (hd256's port was bit-identical
// because it kept TK=64).
// compile-time tag for pf7rp's interior/boundary softmax instances
template <bool B> struct PdPfIn { static constexpr bool value = B; };
//
// SUB (row sub-tiles per warp, the wide rung): SUB=2 gives each warp two
// 16-row m-tiles - a 128-row CTA tile, FlashInfer's own hd128 config for this
// class (CTA_TILE_Q 128 = 4 warps x 2 m-tiles, a 32-key KV tile, one bf16
// repack pane, 2 CTA/SM). Every K/V tile is then staged, repacked and
// ldmatrix'd once for 128 rows instead of 64: each B-frag feeds two mma rows
// of work. The 128-row Q tile only fits two CTAs per SM unpadded, so SUB=2
// stores Q XOR-swizzled (16-byte chunk c of row r at c ^ (r & 7): the 8 rows
// one ldmatrix 8x8 reads sit in 8 distinct bank groups) where SUB=1 keeps
// the +8-half pad. SUB=1 is this kernel as it was, op for op; SUB=2 at TK=32
// moves the online-softmax tile boundary, a numerics-class change.
template <uint32_t G, bool VL = false, uint32_t HD = 256u, uint32_t TK = 64u,
          uint32_t SUB = 1u>
__global__ void __launch_bounds__(128) pd_attn_prefill_pf7rp_kernel(
    const float* __restrict__ q, const unsigned char* __restrict__ pool_k,
    const unsigned char* __restrict__ pool_v, const float* __restrict__ sinks,
    float* __restrict__ out, const unsigned int* __restrict__ positions,
    const uint32_t* __restrict__ block_tables, uint32_t blocks_per_slot,
    const unsigned int* __restrict__ slots, uint32_t n_heads, uint32_t kv_dim,
    uint32_t swa_window, uint32_t n_rows, float scale,
    const uint32_t* __restrict__ vl_items, const uint32_t* __restrict__ run_offs,
    const unsigned int* __restrict__ win_pos) {
#if PD_FA_OK
    constexpr uint32_t MR = 64u * SUB, NKC = TK / 16u;
    constexpr uint32_t QROW = HD + 8u;   // pane (and SUB=1 sh_q) halves per row (+8 pad)
    constexpr bool QSW = SUB > 1u;       // swizzled, unpadded sh_q (see header)
    constexpr uint32_t QROWQ = QSW ? HD : QROW;
    static_assert(SUB == 1u || SUB == 2u, "one or two m-tiles a warp");
    static_assert(!(VL && SUB > 1u), "the varlen items tile 64 rows");
    // SUB=2 runs the softmax in the log2 domain (FlashInfer's form): log2(e)
    // rides the Q scale, so every score's exp is one ex2 - __expf is ex2 of
    // x * log2(e), a multiply an element. Running max/sum stay consistent
    // (sum of 2^(s2 - m2) == sum of e^(s - m)); the sink merge converts.
    constexpr bool L2D = SUB > 1u;
    constexpr float LOG2E = 1.4426950408889634f;
    auto ex = [](float x) -> float {
        if constexpr (L2D) {
            float y;
            asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
            return y;
        } else {
            return __expf(x);
        }
    };
    const float qscale = L2D ? scale * LOG2E : scale;
    constexpr uint32_t RROW = HD;        // raw row bytes: repack-only reads,
                                         // no ldmatrix on raw -> no pad
    const uint32_t kvh = blockIdx.x;
    uint32_t row0, qr0 = 0, span_rows = n_rows;
    if constexpr (VL) {
        const uint32_t* it = vl_items + (size_t)blockIdx.y * 4u;
        qr0 = it[0]; span_rows = it[1]; row0 = it[2];
    } else if (run_offs != nullptr) {
        // multi-run (pf_runs) arm: the same contiguous [off, off+len)
        // partition pf7's run arm consumes, so a packed CTA computes its
        // per-run twin exactly. Tiles past a short run skip via Rtot.
        qr0 = run_offs[blockIdx.z];
        span_rows = run_offs[blockIdx.z + 1u] - qr0;
        row0 = blockIdx.y * MR;
    } else {
        row0 = blockIdx.y * MR;
    }
    const uint32_t Rtot = span_rows * G;
    // window floors from the true positions (null = positions; pd_pf_wp),
    // indexed exactly as `positions` is below - never re-aimed
    const unsigned int* wp = pd_pf_wp(positions, win_pos);
    const uint32_t d = threadIdx.x, warp = d >> 5, lane = d & 31u;

    extern __shared__ __align__(16) unsigned char pf7_sm[];
    __half* sh_q = (__half*)pf7_sm;
    __half* pane = sh_q + (size_t)MR * QROWQ;
    unsigned char* kraw = (unsigned char*)(pane + (size_t)TK * QROW);
    unsigned char* vraw = kraw + (size_t)TK * RROW;
    // per-row positions and true positions (floors) live in the head of the
    // repack pane: every read of them is in the prologue, before the main
    // loop's first barrier, and the pane's first write is K0's repack after
    // it. Their own 512 B put the hd256 tile 256 B past GB10's 101,376-byte
    // opt-in (static included), so it could not launch there at all.
    static_assert(2u * MR * 4u <= TK * QROW * 2u, "positions must fit the pane");
    unsigned int* sh_rpos = (unsigned int*)pane;
    unsigned int* sh_rwin = sh_rpos + MR;

    // positions first: hi/lo gate the K0/V0 cp.async, which must be in
    // flight before the (much longer) Q stage - pf7's Q-then-stage order
    // left the first tiles' latency fully exposed, and the prologue was
    // 32% of rp's wall at fresh-2048 (profiled)
    if (d < MR) {
        sh_rpos[d] =
            (row0 + d) < Rtot ? positions[qr0 + (row0 + d) / G] : 0xFFFFFFFFu;
        sh_rwin[d] =
            (row0 + d) < Rtot ? wp[qr0 + (row0 + d) / G] : 0xFFFFFFFFu;
    }
    __syncthreads();

    uint32_t hi = 0, lo = 0xFFFFFFFFu;
    // pmin / fmax: the lowest live row position and the highest window
    // floor - a KV tile inside [fmax, pmin] is visible to every row of the
    // CTA, so it needs no mask (the SUB=2 interior path below)
    uint32_t pmin = 0xFFFFFFFFu, fmax = 0u;
    for (uint32_t r = 0; r < MR; ++r) {
        const uint32_t p = sh_rpos[r];
        if (p != 0xFFFFFFFFu) {
            hi = max(hi, p + 1u);
            const uint32_t f = pd_pf_floor(sh_rwin[r], swa_window);
            lo = min(lo, f);
            pmin = min(pmin, p);
            fmax = max(fmax, f);
        }
    }
    if (hi == 0u) return;
    const uint32_t lo_t = (lo == 0xFFFFFFFFu ? 0u : lo) / TK * TK;
    // the run arm reads the run's own slot at its row start (each run is a
    // distinct sequence); single-run is slots[0].
    const uint32_t slot = VL ? vl_items[(size_t)blockIdx.y * 4u + 3u]
                             : (run_offs ? (slots ? slots[qr0] : 0u)
                                         : (slots ? slots[0] : 0u));
    const uint32_t* bt = block_tables + (size_t)slot * blocks_per_slot;

    // warp w owns rows [16*SUB*w, 16*SUB*(w+1)), as SUB m16 sub-tiles
    uint32_t r_lo[SUB], r_hi[SUB], pos_lo[SUB], pos_hi[SUB], win_lo[SUB], win_hi[SUB];
    #pragma unroll
    for (uint32_t sb = 0; sb < SUB; ++sb) {
        r_lo[sb] = (warp * SUB + sb) * 16u + (lane >> 2);
        r_hi[sb] = r_lo[sb] + 8u;
        pos_lo[sb] = sh_rpos[r_lo[sb]];
        pos_hi[sb] = sh_rpos[r_hi[sb]];
        win_lo[sb] = sh_rwin[r_lo[sb]];
        win_hi[sb] = sh_rwin[r_hi[sb]];
    }
    const uint32_t c2 = 2u * (lane & 3u);

    // raw K/V tile stage: cp.async 16B lines; rows past hi zero-fill
    // (stale e4m3 can be NaN; PV's 0-weight x NaN would poison O).
    // Unrolled + block-table hoist (mma-pipeline door): the
    // rolled form re-read bt[] from gmem once per 16B chunk - 8x per
    // thread for only 4 unique entries per tile (t0 is TK-aligned) - and
    // that LDG->IMAD address chain was the kernel's single hottest stall
    // site. Same cp.asyncs at the same addresses.
    auto stage_one = [&](unsigned char* dstbuf, const unsigned char* pool,
                         uint32_t t0) {
        const uint32_t n_t = hi - t0 < TK ? hi - t0 : TK;
        uint32_t bt4[TK / 16u];
        #pragma unroll
        for (uint32_t j = 0; j < TK / 16u; ++j)
            bt4[j] = t0 + j * 16u < hi ? bt[(t0 >> 4) + j] : 0u;
        #pragma unroll
        for (uint32_t i = 0; i < TK * (HD / 16u) / 128u; ++i) {
            const uint32_t u = d + i * 128u;
            const uint32_t p = u / (HD / 16u), l = u % (HD / 16u);
            unsigned char* dst = dstbuf + (size_t)p * RROW + l * 16u;
            if (p < n_t) {
                const uint32_t gpos = t0 + p;
                pd_attn_cpa16(dst, pool + (size_t)bt4[p >> 4] * 16u * kv_dim
                    + (size_t)(gpos & 15u) * kv_dim + (size_t)kvh * HD + l * 16u);
            } else {
                *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        pd_attn_cpa_commit();
    };

    // bulk fp8 -> f16 repack: one 16B raw chunk (16 e4m3) -> 32B pane
    // (16 halves), same per-element cvt as pf7's in-loop cvt2. 8 chunks
    // per thread at TK=64; consecutive threads walk one row's 16 chunks.
    // Unrolled (mma-pipeline door): the u+=128 form's dynamic
    // bound kept the loop rolled, serializing 8 LDS->8xF2FP->STS latency
    // chains per thread; the stall profile put the E4M3 cvt chains among
    // the kernel's top stall sites. Constant trip count -> unrolled -> the
    // 8 independent chains overlap.
    auto repack = [&](const unsigned char* raw) {
        // group-staged like the Q loads: a flat unroll let ptxas funnel
        // every chunk's 8 cvts through one temp cluster (serial again) -
        // stage 4 LDS.128 results first, then convert the group
        // CPT = 16B chunks per thread; GRP = how many are staged before the
        // cvt burst. hd256/TK=64 gives CPT=8 -> GRP=4, two groups: the
        // original shape exactly. hd128/TK=48 gives CPT=3, which the old
        // hardcoded /4 turned into zero groups (a silently empty repack).
        constexpr uint32_t CPT = TK * (HD / 16u) / 128u;
        constexpr uint32_t GRP = (CPT % 4u == 0u) ? 4u : CPT;
        static_assert(TK * (HD / 16u) % 128u == 0u,
                      "repack needs whole 16B chunks per thread");
        #pragma unroll
        for (uint32_t gr = 0; gr < CPT / GRP; ++gr) {
            uint4 rw[GRP];
            #pragma unroll
            for (uint32_t j = 0; j < GRP; ++j) {
                const uint32_t u = d + (gr * GRP + j) * 128u;
                rw[j] = *reinterpret_cast<const uint4*>(
                    raw + (size_t)(u / (HD / 16u)) * RROW
                    + (u % (HD / 16u)) * 16u);
            }
            #pragma unroll
            for (uint32_t j = 0; j < GRP; ++j) {
                const uint32_t u = d + (gr * GRP + j) * 128u;
                const uint32_t p = u / (HD / 16u), l = u % (HD / 16u);
                uint4 e0, e1;
                e0.x = pd_pf7_cvt2((unsigned short)rw[j].x);
                e0.y = pd_pf7_cvt2((unsigned short)(rw[j].x >> 16));
                e0.z = pd_pf7_cvt2((unsigned short)rw[j].y);
                e0.w = pd_pf7_cvt2((unsigned short)(rw[j].y >> 16));
                e1.x = pd_pf7_cvt2((unsigned short)rw[j].z);
                e1.y = pd_pf7_cvt2((unsigned short)(rw[j].z >> 16));
                e1.z = pd_pf7_cvt2((unsigned short)rw[j].w);
                e1.w = pd_pf7_cvt2((unsigned short)(rw[j].w >> 16));
                __half* pd_ = pane + (size_t)p * QROW + l * 16u;
                *reinterpret_cast<uint4*>(pd_) = e0;
                *reinterpret_cast<uint4*>(pd_ + 8u) = e1;
            }
        }
    };

    float m_lo[SUB], m_hi[SUB], l_lo[SUB], l_hi[SUB];
    float o_acc[SUB][HD / 16u][8];
    #pragma unroll
    for (uint32_t sb = 0; sb < SUB; ++sb) {
        m_lo[sb] = -1e30f; m_hi[sb] = -1e30f; l_lo[sb] = 0.f; l_hi[sb] = 0.f;
        #pragma unroll
        for (uint32_t dc = 0; dc < HD / 16u; ++dc)
            #pragma unroll
            for (uint32_t i = 0; i < 8u; ++i) o_acc[sb][dc][i] = 0.f;
    }
    // sh_q half index of (row, col): the +8 pad at SUB=1, the chunk swizzle
    // at SUB=2 (col's 8-half chunk XOR row & 7; within-chunk order kept)
    auto qix = [&](uint32_t r, uint32_t c) -> size_t {
        if constexpr (QSW)
            return (size_t)r * QROWQ + ((((c >> 3) ^ (r & 7u)) << 3) | (c & 7u));
        else
            return (size_t)r * QROWQ + c;
    };

    // K0/V0 fly while Q stages
    stage_one(kraw, pool_k, lo_t);
    stage_one(vraw, pool_v, lo_t);

    // Q stage (scaled f32 -> f16), float4-wide: the scalar version's
    // LDG->FMUL chains had no memory-level parallelism and were 32% of
    // rp's wall (profiled). 16B loads, half2 packs, 8B stores.
    // Grouped loads (mma-pipeline door): a flat full unroll let ptxas
    // serialize all 32 iterations through one temp cluster (every FMUL
    // consumer stalling on its own LDG - 25% of kernel stall samples).
    // An explicit 8-wide load array forces 8 LDG.128 in flight per group.
    #pragma unroll
    for (uint32_t gq = 0; gq < MR * (HD / 4u) / 128u / 8u; ++gq) {
        float4 x[8];
        #pragma unroll
        for (uint32_t j = 0; j < 8u; ++j) {
            const uint32_t u = d + (gq * 8u + j) * 128u;
            uint32_t R = row0 + u / (HD / 4u);
            // clamped UNCONDITIONAL load: predicated loads made ptxas emit
            // preserve-MOVs per lane; out-of-range rows read a valid row
            // and the convert loop's guard still stores exact zeros
            R = R < Rtot ? R : Rtot - 1u;
            x[j] = *reinterpret_cast<const float4*>(
                q + ((size_t)(qr0 + R / G) * n_heads + kvh * G + R % G)
                    * HD + (u % (HD / 4u)) * 4u);
        }
        #pragma unroll
        for (uint32_t j = 0; j < 8u; ++j) {
            const uint32_t u = d + (gq * 8u + j) * 128u;
            const uint32_t r = u / (HD / 4u), c4 = (u % (HD / 4u)) * 4u;
            __half2 h0 = __float2half2_rn(0.f), h1 = h0;
            if (row0 + r < Rtot) {
                h0 = __floats2half2_rn(x[j].x * qscale, x[j].y * qscale);
                h1 = __floats2half2_rn(x[j].z * qscale, x[j].w * qscale);
            }
            *reinterpret_cast<__half2*>(sh_q + qix(r, c4)) = h0;
            *reinterpret_cast<__half2*>(sh_q + qix(r, c4 + 2u)) = h1;
        }
    }
    if constexpr (!QSW) {
        if (d < MR)  // zero the +8 pad halves so ldmatrix never sees junk
            *reinterpret_cast<uint4*>(sh_q + (size_t)d * QROW + HD) =
                make_uint4(0u, 0u, 0u, 0u);
    }

    for (uint32_t t0 = lo_t; t0 < hi; t0 += TK) {
        const bool more = t0 + TK < hi;
        // pending {K[i], V[i]} -> wait1 retires K[i] (commit order), V may fly
        pd_attn_cpa_wait1();
        __syncthreads();           // prev PV done reading pane + kraw ready
                                   // (first iter: Q stage visible too)
        repack(kraw);
        __syncthreads();           // pane K visible; kraw free
        if (more) stage_one(kraw, pool_k, t0 + TK);

        // scores: warp-owned m16 x TK; QK-B = x4 non-trans on the pane.
        // Software-pipelined (mma-pipeline door): 1
        // CTA/SM x 4 warps = one warp per scheduler, so the old ldmatrix ->
        // immediately-dependent mma order stalled the full smem latency on
        // every step (both helpers are asm volatile - the compiler cannot
        // hoist the loads itself). Prefetch the next B-frag (and the next
        // dc's A-frag) one step ahead of the consuming mma pair. Loads are
        // pure reads on panes nobody writes during the phase and the mma
        // issue order/operands are untouched -> bit-identical.
        float s[SUB][NKC][8];
        #pragma unroll
        for (uint32_t sb = 0; sb < SUB; ++sb)
            #pragma unroll
            for (uint32_t kc = 0; kc < NKC; ++kc)
                #pragma unroll
                for (uint32_t i = 0; i < 8u; ++i) s[sb][kc][i] = 0.f;
        // lanes 0-7: keys 0-7 cols 0-7 | 8-15: keys 0-7 cols 8-15 |
        // 16-23: keys 8-15 cols 0-7 | 24-31: keys 8-15 cols 8-15
        // A-frag of sub-tile sb at head-dim chunk dc (16 halves): at SUB=1
        // exactly the old qk_a0 + dc * 32 B
        auto qk_a = [&](uint32_t sb, uint32_t dc) -> const unsigned char* {
            return (const unsigned char*)(sh_q
                + qix((warp * SUB + sb) * 16u + (lane & 15u),
                      dc * 16u + ((lane >> 4) ? 8u : 0u)));
        };
        const unsigned char* qk_b0 = (const unsigned char*)(pane
            + (size_t)(((lane >> 4) ? 8u : 0u) + (lane & 7u)) * QROW
            + (((lane >> 3) & 1u) ? 8u : 0u));
        uint32_t af[SUB][2][4], bf[2][4];
        #pragma unroll
        for (uint32_t sb = 0; sb < SUB; ++sb) pd_ldm_x4(af[sb][0], qk_a(sb, 0u));
        pd_ldm_x4(bf[0], qk_b0);
        #pragma unroll
        for (uint32_t u = 0; u < HD / 16u * NKC; ++u) {
            const uint32_t dc = u / NKC, kc = u % NKC;
            if (u + 1u < HD / 16u * NKC)
                pd_ldm_x4(bf[(u + 1u) & 1u], qk_b0
                    + (size_t)(((u + 1u) % NKC) * 16u) * QROW * 2u
                    + ((u + 1u) / NKC) * 32u);
            if (kc == 0u && dc + 1u < HD / 16u) {
                #pragma unroll
                for (uint32_t sb = 0; sb < SUB; ++sb)
                    pd_ldm_x4(af[sb][(dc + 1u) & 1u], qk_a(sb, dc + 1u));
            }
            // one B-frag pair, SUB sub-tiles of mma (the wide rung's reuse)
            #pragma unroll
            for (uint32_t sb = 0; sb < SUB; ++sb) {
                pd_fa_mma16(&s[sb][kc][0], af[sb][dc & 1u][0], af[sb][dc & 1u][1],
                            af[sb][dc & 1u][2], af[sb][dc & 1u][3],
                            bf[u & 1u][0], bf[u & 1u][1]);
                pd_fa_mma16(&s[sb][kc][4], af[sb][dc & 1u][0], af[sb][dc & 1u][1],
                            af[sb][dc & 1u][2], af[sb][dc & 1u][3],
                            bf[u & 1u][2], bf[u & 1u][3]);
            }
        }

        // mask + online softmax, in registers (same VALUES as pf7). Max
        // reduce as per-kc partials + pairwise combine (mma-pipeline door):
        // the single-accumulator form was a serial 16-deep FMNMX chain; max
        // over the same set is order-independent (exact, no rounding), so
        // the tree is bit-identical with dependency depth 4+2.
        // per sub-tile (SUB=1: the single-tile code, unchanged)
        float corr_lo[SUB], corr_hi[SUB];
        uint32_t pf[SUB][NKC][4];
        // INTERIOR (SUB=2 only): a KV tile every row of the CTA sees in full
        // - all keys at or below the lowest row position and at or above the
        // highest window floor - skips the per-element mask and the dead-key
        // guard. On such a tile the masked path keeps every score and every
        // guard passes, so the two instances compute the same bits; the mask
        // was ~7 ALU ops an element, a third of the kernel's issue at depth.
        const bool interior = SUB > 1u && t0 + TK <= pmin + 1u && t0 >= fmax;
        auto softmax_tile = [&](auto in_tag) {
            constexpr bool IN = decltype(in_tag)::value;
            #pragma unroll
            for (uint32_t sb = 0; sb < SUB; ++sb) {
                float mxl[NKC], mxh[NKC];
                #pragma unroll
                for (uint32_t kc = 0; kc < NKC; ++kc) {
                    mxl[kc] = -1e30f; mxh[kc] = -1e30f;
                    const uint32_t k0 = t0 + kc * 16u;
                    (void)k0;
                    #pragma unroll
                    for (uint32_t i = 0; i < 8u; ++i) {
                        const bool hirow = (i & 2u) != 0u;
                        float v = s[sb][kc][i];
                        if constexpr (!IN) {
                            const uint32_t p = k0 + c2 + (i & 1u) + ((i >> 2) ? 8u : 0u);
                            const uint32_t pos = hirow ? pos_hi[sb] : pos_lo[sb];
                            const uint32_t wpos = hirow ? win_hi[sb] : win_lo[sb];
                            const bool valid = p < hi && p <= pos
                                && (!swa_window || p + swa_window > wpos);
                            v = valid ? v : -1e30f;
                            s[sb][kc][i] = v;
                        }
                        if (hirow) mxh[kc] = fmaxf(mxh[kc], v);
                        else mxl[kc] = fmaxf(mxl[kc], v);
                    }
                }
                // max over the same set is exact and order-independent, so the shape
                // of this combine is free; keep the depth-2 4-way tree where NKC==4
                // (hd256) and fall to a generic fold otherwise (hd128/TK=48 has
                // NKC=3, where the old literal indices read mxl[3] out of bounds).
                float mx_lo, mx_hi;
                if constexpr (NKC == 4u) {
                    mx_lo = fmaxf(fmaxf(mxl[0], mxl[1]), fmaxf(mxl[2], mxl[3]));
                    mx_hi = fmaxf(fmaxf(mxh[0], mxh[1]), fmaxf(mxh[2], mxh[3]));
                } else {
                    mx_lo = mxl[0]; mx_hi = mxh[0];
                    #pragma unroll
                    for (uint32_t kc = 1u; kc < NKC; ++kc) {
                        mx_lo = fmaxf(mx_lo, mxl[kc]);
                        mx_hi = fmaxf(mx_hi, mxh[kc]);
                    }
                }
                #pragma unroll
                for (uint32_t off = 1; off <= 2; off <<= 1) {
                    mx_lo = fmaxf(mx_lo, __shfl_xor_sync(0xffffffffu, mx_lo, off));
                    mx_hi = fmaxf(mx_hi, __shfl_xor_sync(0xffffffffu, mx_hi, off));
                }
                const float mn_lo = fmaxf(m_lo[sb], mx_lo), mn_hi = fmaxf(m_hi[sb], mx_hi);
                corr_lo[sb] = (m_lo[sb] > -1e29f) ? ex(m_lo[sb] - mn_lo) : 0.f;
                corr_hi[sb] = (m_hi[sb] > -1e29f) ? ex(m_hi[sb] - mn_hi) : 0.f;
                float ps_lo = 0.f, ps_hi = 0.f;
                #pragma unroll
                for (uint32_t kc = 0; kc < NKC; ++kc) {
                    #pragma unroll
                    for (uint32_t i = 0; i < 8u; ++i) {
                        const bool hirow = (i & 2u) != 0u;
                        const float w = (IN || s[sb][kc][i] > -1e29f)
                            ? ex(s[sb][kc][i] - (hirow ? mn_hi : mn_lo)) : 0.f;
                        s[sb][kc][i] = w;
                        if (hirow) ps_hi += w; else ps_lo += w;
                    }
                    // paired F2FP.PACK (mma-pipeline door): identical
                    // RN rounding to the old per-half cvt+shift+or, one op per word
                    // instead of three (word-gated bit-identical)
                    const __half2 pk0 = __floats2half2_rn(s[sb][kc][0], s[sb][kc][1]);
                    const __half2 pk1 = __floats2half2_rn(s[sb][kc][2], s[sb][kc][3]);
                    const __half2 pk2 = __floats2half2_rn(s[sb][kc][4], s[sb][kc][5]);
                    const __half2 pk3 = __floats2half2_rn(s[sb][kc][6], s[sb][kc][7]);
                    pf[sb][kc][0] = *reinterpret_cast<const uint32_t*>(&pk0);
                    pf[sb][kc][1] = *reinterpret_cast<const uint32_t*>(&pk1);
                    pf[sb][kc][2] = *reinterpret_cast<const uint32_t*>(&pk2);
                    pf[sb][kc][3] = *reinterpret_cast<const uint32_t*>(&pk3);
                }
                #pragma unroll
                for (uint32_t off = 1; off <= 2; off <<= 1) {
                    ps_lo += __shfl_xor_sync(0xffffffffu, ps_lo, off);
                    ps_hi += __shfl_xor_sync(0xffffffffu, ps_hi, off);
                }
                l_lo[sb] = l_lo[sb] * corr_lo[sb] + ps_lo; m_lo[sb] = mn_lo;
                l_hi[sb] = l_hi[sb] * corr_hi[sb] + ps_hi; m_hi[sb] = mn_hi;
            }
        };
        if (interior) softmax_tile(PdPfIn<true>{});
        else softmax_tile(PdPfIn<false>{});

        __syncthreads();           // pane K consumed by all warps
        pd_attn_cpa_wait1();       // V[i] arrived (K[i+1] may fly);
                                   // last tile: pending {V[i]} only -> wait0
        if (!more) pd_attn_cpa_wait0();
        repack(vraw);
        __syncthreads();           // pane V visible; vraw free
        if (more) stage_one(vraw, pool_v, t0 + TK);

        // O rescale + PV; PV-B = x4.trans on the pane. Loop nest SWAPPED to
        // kc-outer/dc-inner + distance-1 B prefetch (same mma-pipeline
        // door): the old dc-outer nest re-accumulated the same o_acc[dc]
        // registers on back-to-back mma pairs - a dependency-distance-1
        // HMMA chain, 64x per tile, with no second warp per scheduler to
        // fill the wait. kc-outer puts 2*HD/16 accumulator-independent mmas
        // between o_acc[dc] revisits, and each o_acc[dc] still receives its
        // kc terms in the same 0..NKC-1 order with identical operands ->
        // bit-identical.
        #pragma unroll
        for (uint32_t sb = 0; sb < SUB; ++sb)
            #pragma unroll
            for (uint32_t dc = 0; dc < HD / 16u; ++dc) {
                o_acc[sb][dc][0] *= corr_lo[sb]; o_acc[sb][dc][1] *= corr_lo[sb];
                o_acc[sb][dc][2] *= corr_hi[sb]; o_acc[sb][dc][3] *= corr_hi[sb];
                o_acc[sb][dc][4] *= corr_lo[sb]; o_acc[sb][dc][5] *= corr_lo[sb];
                o_acc[sb][dc][6] *= corr_hi[sb]; o_acc[sb][dc][7] *= corr_hi[sb];
            }
        // lanes 0-7: keys 0-7 cols 0-7 | 8-15: keys 8-15 cols 0-7 |
        // 16-23: keys 0-7 cols 8-15 | 24-31: keys 8-15 cols 8-15
        const __half* pv_b0 = pane
            + (size_t)((((lane >> 3) & 1u) ? 8u : 0u) + (lane & 7u)) * QROW
            + ((lane >> 4) ? 8u : 0u);
        uint32_t bt2[2][4];
        pd_pf7_ldmt(bt2[0], pv_b0);
        #pragma unroll
        for (uint32_t u = 0; u < NKC * (HD / 16u); ++u) {
            const uint32_t kc = u / (HD / 16u), dc = u % (HD / 16u);
            if (u + 1u < NKC * (HD / 16u))
                pd_pf7_ldmt(bt2[(u + 1u) & 1u], pv_b0
                    + (size_t)(((u + 1u) / (HD / 16u)) * 16u) * QROW
                    + ((u + 1u) % (HD / 16u)) * 16u);
            #pragma unroll
            for (uint32_t sb = 0; sb < SUB; ++sb) {
                pd_fa_mma16(&o_acc[sb][dc][0], pf[sb][kc][0], pf[sb][kc][1],
                            pf[sb][kc][2], pf[sb][kc][3],
                            bt2[u & 1u][0], bt2[u & 1u][1]);
                pd_fa_mma16(&o_acc[sb][dc][4], pf[sb][kc][0], pf[sb][kc][1],
                            pf[sb][kc][2], pf[sb][kc][3],
                            bt2[u & 1u][2], bt2[u & 1u][3]);
            }
        }
    }

    // epilogue: sink merge + normalize + direct f32 stores (identical)
    #pragma unroll
    for (uint32_t sb = 0; sb < SUB; ++sb)
    #pragma unroll
    for (uint32_t half = 0; half < 2u; ++half) {
        const uint32_t R = row0 + (half ? r_hi[sb] : r_lo[sb]);
        if (R >= Rtot) continue;
        const uint32_t h = kvh * G + R % G;
        const float mm = half ? m_hi[sb] : m_lo[sb], ll = half ? l_hi[sb] : l_lo[sb];
        const float sv = L2D ? sinks[h] * LOG2E : sinks[h];
        const float mtot = fmaxf(mm, sv);
        const float cm = ex(mm - mtot), cs2 = ex(sv - mtot);
        const float l = ll * cm + cs2;
        const float nrm = l > 0.f ? cm / l : 0.f;
        float* op = out + ((size_t)(qr0 + R / G) * n_heads + h) * HD;
        #pragma unroll
        for (uint32_t dc = 0; dc < HD / 16u; ++dc) {
            const uint32_t i0 = half ? 2u : 0u;
            *reinterpret_cast<float2*>(op + dc * 16u + c2) =
                make_float2(o_acc[sb][dc][i0] * nrm, o_acc[sb][dc][i0 + 1u] * nrm);
            *reinterpret_cast<float2*>(op + dc * 16u + 8u + c2) =
                make_float2(o_acc[sb][dc][i0 + 4u] * nrm, o_acc[sb][dc][i0 + 5u] * nrm);
        }
    }
#else
    (void)q; (void)pool_k; (void)pool_v; (void)sinks; (void)out;
    (void)positions; (void)block_tables; (void)blocks_per_slot; (void)slots;
    (void)n_heads; (void)kv_dim; (void)swa_window; (void)n_rows; (void)scale;
    (void)vl_items; (void)run_offs;
#endif
}

// hd128 fp8 launches of pf7rp for pd_attn_prefill_f16_paged (attn/prefill_pf5.cuh):
// the wide rung, then the 64-row arm. True when one launched (its status in
// *rc); false sends the caller on to the pf7 / v4 arms.
static inline bool pd_pf7rp128_go(int* rc, bool f8v4n, uint32_t g_, const void* q,
                                  const void* pool_k, const void* pool_v, const void* sinks,
                                  void* out, const void* positions, const void* block_tables,
                                  uint32_t blocks_per_slot, const void* slots,
                                  uint32_t n_heads, uint32_t n_kv_heads, uint32_t kv_dim,
                                  uint32_t swa_window, uint32_t batch, float scale,
                                  const unsigned int* wp, void* stream) {
    // hd128 REPACK-PANE arm (pf7rp) - sits above the pf7
    // convert-in-register arm below because it wins the long spans that
    // arm loses. Measured against FlashInfer's own prefill kernel, pf7
    // held a FLAT deficit
    // from 1536 rows on, and reading their kernel named the cause:
    // USE_KV_REPACK is true for this config, so their mainloop does
    // repack_fp8_tile_to_bf16() once per tile and then runs native b16
    // ldmatrix. pf7 converts per FRAGMENT LOAD -- pd_pf7_swz is 2 shfl +
    // 2 prmt and pd_pf7_swzt is 3+3, inside the mma loop, so all four
    // warps re-swizzle and re-convert the whole K tile (~640 warp
    // shuffles per KV tile per CTA). Two q-tile-WIDTH arms of pf7 were
    // falsified first (NW=8 +30%, SUB=2 +15% at FlashInfer's exact launch
    // config): there the per-fragment convert scales with the math, so the
    // tile could not be the lever. pf7rp already was the repack structure at
    // hd256 (door 2, bit-identical there, -5..-21%); this is the hd128
    // generalization - and with the convert out of the mma loop, the 128-row
    // tile IS the next lever (the wide rung, first below). The 64-row arm
    // runs TK=48 not 64: at TK=64 its tile is 51,456 B and misses 2 CTA/SM by
    // 256 BYTES -> 4 warps/SM, and NW=8 already measured what 1 CTA/SM
    // costs this pipeline. Kill: PADDOCK_NO_PF7RP -> the caller's pf7 arm.
    // G in {4,6,8,9,12,16} (12: Kolibri 1's 48q/4kv): the first wiring instantiated only
    // 4/6/8 -- granite's ratio and its neighbours -- and the fleet sweep
    // found the two ratios outside that set were the worst prefill
    // cells by far, because they fell through to v4 and never saw
    // the repack pane at all (laguna-s 72q/8kv G=9 and nemotron-3.5
    // 32q/2kv G=16), while a G=8 model on the same kernel was fine.
    // G only ever indexes the
    // row->head map (R / G, R % G) and need not divide MR -- the
    // R < Rtot guard covers the tail -- so this set is a pure
    // instantiation list, and a ratio missing from it is silently a
    // different, slower kernel. The outer gate already admitted all five.
    static const bool no_rp128 = pd_env("PADDOCK_NO_PF7RP") != nullptr;
    // the wide rung (pf7rp SUB=2): FlashInfer's own hd128 config - a
    // 128-row CTA tile (two m-tiles a warp), a 32-key KV tile, Q
    // XOR-swizzled to keep 2 CTA/SM - plus an unmasked instance for KV
    // tiles every row sees in full and the softmax in the log2 domain.
    // GB10, same KV bytes as FlashInfer 0.7 (bench/rival_flashinfer_attn.py):
    // 8192-row prompts 64-65 -> 83-89 TF/s on every hd128 shape (0.81x ->
    // 1.06-1.10x theirs; Kolibri G12 chunks at 32K-262K depth 1.06-1.07x),
    // 2048 rows 1.01-1.04x, 512 rows still behind (launch-bound there).
    // Numerics class: TK 48 -> 32 moves the softmax tile boundary and
    // log2(e) rides the f16 Q scale; agreement with FlashInfer's bf16
    // output is unchanged (relRMS 1.9e-3..2.4e-3, both arms).
    // Kill: PADDOCK_NO_PF7RP_SUB2 -> the 64-row arm below.
    static const bool rp128w = pd_env("PADDOCK_NO_PF7RP_SUB2") == nullptr;
    if (f8v4n && !no_rp128 && rp128w && (g_ == 4u || g_ == 6u || g_ == 8u
                                         || g_ == 9u || g_ == 12u || g_ == 16u)) {
        constexpr uint32_t WMR = 128u, WTK = 32u, WHD = 128u;
        // Q MR*HD h (swizzled, unpadded) | pane TK*(HD+8) h | raw K,V
        // TK*HD B each: 49,664 -> 2 CTA/SM with the 1 KB reserve each
        constexpr uint32_t WSM =
            WMR * WHD * 2u + WTK * (WHD + 8u) * 2u + 2u * WTK * WHD;
#define PD_PF7RPW_FITS(GV)                                                     \
pd_smem_fits((const void*)pd_attn_prefill_pf7rp_kernel<GV, false, WHD,     \
                                                       WTK, 2u>, WSM)
        static const bool w_fits = PD_PF7RPW_FITS(4u) && PD_PF7RPW_FITS(6u)
            && PD_PF7RPW_FITS(8u) && PD_PF7RPW_FITS(9u) && PD_PF7RPW_FITS(12u)
            && PD_PF7RPW_FITS(16u);
#undef PD_PF7RPW_FITS
        if (w_fits) {
            const bool multi = pd_pf_runs_offs != nullptr;
            const dim3 gr = multi
                ? dim3(n_kv_heads, (pd_pf_runs_maxn * g_ + WMR - 1u) / WMR,
                       pd_pf_runs_n)
                : dim3(n_kv_heads, (batch * g_ + WMR - 1u) / WMR);
            const uint32_t* ro = multi ? (const uint32_t*)pd_pf_runs_offs : nullptr;
#define PD_PF7RPW_LAUNCH(GV)                                                   \
pd_attn_prefill_pf7rp_kernel<GV, false, WHD, WTK, 2u>                      \
    <<<gr, 128, WSM, (cudaStream_t)stream>>>(                              \
    (const float*)q, (const unsigned char*)pool_k,                         \
    (const unsigned char*)pool_v,                                          \
    (const float*)sinks, (float*)out, (const unsigned int*)positions,      \
    (const uint32_t*)block_tables, blocks_per_slot,                        \
    (const unsigned int*)slots, n_heads, kv_dim, swa_window, batch, scale, \
    nullptr, ro, wp)
            if (g_ == 4u) PD_PF7RPW_LAUNCH(4u);
            else if (g_ == 6u) PD_PF7RPW_LAUNCH(6u);
            else if (g_ == 8u) PD_PF7RPW_LAUNCH(8u);
            else if (g_ == 9u) PD_PF7RPW_LAUNCH(9u);
            else if (g_ == 12u) PD_PF7RPW_LAUNCH(12u);
            else PD_PF7RPW_LAUNCH(16u);
#undef PD_PF7RPW_LAUNCH
            *rc = pd_launch_status();
            return true;
        }
    }
    if (f8v4n && !no_rp128 && (g_ == 4u || g_ == 6u || g_ == 8u
                               || g_ == 9u || g_ == 12u || g_ == 16u)) {
        constexpr uint32_t RPMR = 64u, RPTK = 48u, RPHD = 128u;
        // Q MR*(HD+8)h | pane TK*(HD+8)h | raw K,V TK*HD B each (the
        // per-row positions ride the pane). Every term carries its own
        // shape factor.
        constexpr uint32_t RPSM =
            RPMR * (RPHD + 8u) * 2u + RPTK * (RPHD + 8u) * 2u
            + 2u * RPTK * RPHD;                         // 42,752 -> 2 CTA/SM
#define PD_PF7RP_128_FITS(GV)                                                  \
pd_smem_fits((const void*)pd_attn_prefill_pf7rp_kernel<GV, false, RPHD,    \
                                                       RPTK>, RPSM)
        static const bool rp128_fits = PD_PF7RP_128_FITS(4u)
            && PD_PF7RP_128_FITS(6u) && PD_PF7RP_128_FITS(8u)
            && PD_PF7RP_128_FITS(9u) && PD_PF7RP_128_FITS(12u)
            && PD_PF7RP_128_FITS(16u);
#undef PD_PF7RP_128_FITS
        if (rp128_fits) {
            const bool multi = pd_pf_runs_offs != nullptr;
            const dim3 gr = multi
                ? dim3(n_kv_heads, (pd_pf_runs_maxn * g_ + RPMR - 1u) / RPMR,
                       pd_pf_runs_n)
                : dim3(n_kv_heads, (batch * g_ + RPMR - 1u) / RPMR);
            const uint32_t* ro = multi ? (const uint32_t*)pd_pf_runs_offs : nullptr;
#define PD_PF7RP_128_LAUNCH(GV)                                                \
pd_attn_prefill_pf7rp_kernel<GV, false, RPHD, RPTK>                        \
    <<<gr, 128, RPSM, (cudaStream_t)stream>>>(                             \
    (const float*)q, (const unsigned char*)pool_k,                         \
    (const unsigned char*)pool_v,                                          \
    (const float*)sinks, (float*)out, (const unsigned int*)positions,      \
    (const uint32_t*)block_tables, blocks_per_slot,                        \
    (const unsigned int*)slots, n_heads, kv_dim, swa_window, batch, scale, \
    nullptr, ro, wp)
            if (g_ == 4u) PD_PF7RP_128_LAUNCH(4u);
            else if (g_ == 6u) PD_PF7RP_128_LAUNCH(6u);
            else if (g_ == 8u) PD_PF7RP_128_LAUNCH(8u);
            else if (g_ == 9u) PD_PF7RP_128_LAUNCH(9u);
            else if (g_ == 12u) PD_PF7RP_128_LAUNCH(12u);
            else PD_PF7RP_128_LAUNCH(16u);
#undef PD_PF7RP_128_LAUNCH
            *rc = pd_launch_status();
            return true;
        }
    }
    return false;
}

// pf7 varlen launcher (AF3, ABI 322): one launch per layer covering
// every eligible prefill span of the tick via the stride-4 tile items -
// see the kernel's VL comment. fp8 pools at the pf7 shapes only (hd256,
// G in {4,6,8}); the engine pre-checks eligibility and per-span launches
// remain the fallback, so anything else here is a hard error, not a route.
// Kill: PADDOCK_NO_PF7 disables the election engine-side via has_ + this
// guard (shared latch with the per-span pf7 election).
int pd_attn_prefill_f16_paged_vl(const void* q, const void* pool_k,
                                 const void* pool_v, const void* sinks,
                                 void* out, const void* positions,
                                 const void* vl_items, uint32_t n_tiles,
                                 const void* block_tables,
                                 uint32_t blocks_per_slot, uint32_t n_heads,
                                 uint32_t n_kv_heads, uint32_t head_dim,
                                 uint32_t kv_dim, uint32_t swa_window,
                                 float scale, uint32_t kv_dtype, void* stream) {
    if (n_tiles == 0) return 0;
    static const bool no_pf7_vl = pd_env("PADDOCK_NO_PF7") != nullptr
        || pd_env("PADDOCK_NO_PF_V4") != nullptr;
    if (no_pf7_vl || kv_dtype != PD_KV_FP8_E4M3 || head_dim != 256u
        || n_kv_heads == 0 || n_heads % n_kv_heads != 0)
        return cudaErrorInvalidValue;
    const uint32_t g_ = n_heads / n_kv_heads;
    if (g_ != 4u && g_ != 6u && g_ != 8u) return cudaErrorInvalidValue;
    // pf7rp arm first (door 2) - bit-identical, leaner mainloop;
    // kill PADDOCK_NO_PF7RP -> pf7. Same VL item contract for both.
    static const bool no_rp_vl = pd_env("PADDOCK_NO_PF7RP") != nullptr;
    // Q and pane 64 x 264 halves, raw K/V 64 x 256 B (positions ride the pane)
    constexpr uint32_t RPSM = 2u * 64u * 264u * 2u + 2u * 64u * 256u;
    static const bool rp_vl_fits =
        pd_smem_fits((const void*)pd_attn_prefill_pf7rp_kernel<4u, true>, RPSM)
        && pd_smem_fits((const void*)pd_attn_prefill_pf7rp_kernel<6u, true>, RPSM)
        && pd_smem_fits((const void*)pd_attn_prefill_pf7rp_kernel<8u, true>, RPSM);
    if (!no_rp_vl && rp_vl_fits) {
        static bool arpv = false;
        if (!arpv) {
            ::fprintf(stderr, "[pf7rp-vl] ENGAGED (repacked packed varlen)\n");
            arpv = true;
        }
        dim3 gv(n_kv_heads, n_tiles);
#define PD_RPVL_LAUNCH(GV)                                                     \
    pd_attn_prefill_pf7rp_kernel<GV, true>                                     \
        <<<gv, 128, RPSM, (cudaStream_t)stream>>>(                             \
            (const float*)q, (const unsigned char*)pool_k,                     \
            (const unsigned char*)pool_v, (const float*)sinks, (float*)out,    \
            (const unsigned int*)positions, (const uint32_t*)block_tables,     \
            blocks_per_slot, nullptr, n_heads, kv_dim, swa_window, 0u, scale,  \
            (const uint32_t*)vl_items, nullptr, nullptr)
        if (g_ == 4u) PD_RPVL_LAUNCH(4u);
        else if (g_ == 6u) PD_RPVL_LAUNCH(6u);
        else PD_RPVL_LAUNCH(8u);
#undef PD_RPVL_LAUNCH
        return pd_launch_status();
    }
    constexpr uint32_t P7SM = 64u * 264u * 2u + 3u * 64u * 272u + 256u;
    static const bool p7_vl_fits =
        pd_smem_fits((const void*)pd_attn_prefill_pf7_kernel<256u, 4u, true>, P7SM)
        && pd_smem_fits((const void*)pd_attn_prefill_pf7_kernel<256u, 6u, true>, P7SM)
        && pd_smem_fits((const void*)pd_attn_prefill_pf7_kernel<256u, 8u, true>, P7SM);
    if (!p7_vl_fits) return cudaErrorInvalidValue;
    static bool a7v = false;
    if (!a7v) {
        ::fprintf(stderr, "[pf7-vl] ENGAGED (packed varlen prefill)\n");
        a7v = true;
    }
    dim3 gv(n_kv_heads, n_tiles);
#define PD_PF7VL_LAUNCH(GV)                                                    \
    pd_attn_prefill_pf7_kernel<256u, GV, true>                                       \
        <<<gv, 128, P7SM, (cudaStream_t)stream>>>(                             \
            (const float*)q, (const unsigned char*)pool_k,                     \
            (const unsigned char*)pool_v, (const float*)sinks, (float*)out,    \
            (const unsigned int*)positions, (const uint32_t*)block_tables,     \
            blocks_per_slot, nullptr, n_heads, kv_dim, swa_window, 0u, scale,  \
            (const uint32_t*)vl_items, nullptr, nullptr)
    if (g_ == 4u) PD_PF7VL_LAUNCH(4u);
    else if (g_ == 6u) PD_PF7VL_LAUNCH(6u);
    else PD_PF7VL_LAUNCH(8u);
#undef PD_PF7VL_LAUNCH
    return pd_launch_status();
}

// Do not DELETE - looks dead, is not. Nothing under src/ uses CK, so a grep
// says "unused macro", but out-of-tree harnesses that `#include "pack.cu"`
// and do not define their own get this definition. Harnesses that define
// theirs after the include shadow this one.
#define CK(x) do { cudaError_t _e=(x); if(_e!=cudaSuccess){printf("FAIL %d %s\n",__LINE__,cudaGetErrorString(_e));exit(1);} } while(0)

