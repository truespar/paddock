// attn/prefill_pf5.cuh - the tcgen05 prefill families (pf5, pf5g, pf6s and
// the cluster twins) plus the multi-slot batch f16 WMMA prefill.
// Textually-included segment of the single pack translation unit.
// Not standalone-compilable: include order is defined by ../pack.cu.
//
// Split out of attn/prefill.cuh (see attn/prefill_fa2.cuh).
//
// Everything in the pf5/pf6 family is gated on PD_TC5_OK (sm_100a only), so
// on any other arch this whole segment compiles to empty kernel bodies -
// which is exactly why a change here has to be verified with an sm_100 build,
// not just sm_120. PD_TC5_OK and pd_tc5_sdesc come from ../tma_desc.cuh.
//
// Include after attn/prefill_fa2.cuh.
// Paged f16 WMMA prefill launcher (P4b-2). Same shape as the dense f16 prefill,
// K/V from the block pool via block tables. No max_ctx (the pool + table replace
// it); assumes max_ctx % 64 == 0 (the engine's is) so a tile's over-read past hi
// stays within the slot's block-table row - those keys are masked in softmax.
// pf5 (class rework 2): tcgen05-S prefill attention. Q·K^T runs on
// the tensor-memory pipe kind::f16 (a probe pins idesc/sdesc; K tiles
// stride TK*128B), softmax is THREAD-LOCAL from tcgen05.ld rows (no
// shuffles, no score smem), P·V stays register-HMMA with the online
// rescale, K/V cp.async double-buffered. SWA hd256/G2: 587 -> 395 us at
// the 2048-row shape (174 TF), exact class vs the incumbent.
// PD_TC5_OK and pd_tc5_sdesc come from ../tma_desc.cuh, which pack.cu includes
// right after abi.cuh - i.e. Before this file. Both used to live in
// gemm/dense_fp4_w8.cuh, included after this one, so this segment carried its
// own `PD_PF5_OK` guard and a hand-copied `pd_pf5_sdesc` twin.
//
// Keep the ordering in mind if these ever move again: an undefined PD_TC5_OK
// does not fail the build, it evaluates as `#if 0`. That is exactly how the
// first pack build silently compiled an empty pf5 body and emitted garbage -
// while an out-of-tree TU that defined the macro after its include was fine.
#if PD_TC5_OK

// F8 arm of the pf5 family: in-place e4m3 -> f16 expansion of the
// staged byte strips into the K (SW128) and V (padded-row) half layouts.
// The strips sit in the UPPER half of each buffer's byte range (K region is
// TK*HD halves, V region TK*row_e halves; the strip needs TK*HD bytes) so
// the port costs zero extra smem. Register-staged per region - the half
// writes cover the strip bytes they read - K wave then V wave so one wave's
// chunks are live across a barrier at a time. Rows >= nkeys ZERO-fill here,
// not at stage time: the f16 path's swizzled zero stores would overlap the
// strip byte range and race the strip cp.asyncs. (Zero-fill also kills the
// stale-e4m3 NaN hazard: 0x7f/0xff decode to NaN and would poison the PV
// accumulate through 0-weight products.) 3 extra barriers per tile; the
// mmas and softmax see the exact layouts f16 staging produces.
//  (fp8-native QK^T): swizzle-only port of the K strip - e4m3 bytes
// move from the linear strip into the SW128 e4m3 operand layout the
// kind::f8f6f4 mma reads directly. No f16 conversion, half the write bytes
// of the expand path; the strip (upper TK*HD bytes) and the target (lower
// TK*HD) are disjoint, so one wave + one barrier. Stale rows >= nkeys stay
// garbage: the softmax mask turns any NaN S column into -INF before use.
template <uint32_t TK, uint32_t HD>
__device__ __forceinline__ void pd_pf5_f8_swz_k(__half* kb, uint32_t tid) {
    const unsigned char* strip = (const unsigned char*)kb + (size_t)TK * HD;
    unsigned char* dst = (unsigned char*)kb;
    // 16B chunks: row kr (TK), chunk c16 of HD/16 per row
    #pragma unroll 2
    for (uint32_t i = tid; i < TK * (HD / 16u); i += 256u) {
        const uint32_t kr = i / (HD / 16u), c16 = i % (HD / 16u);
        const uint4 v = *(const uint4*)(strip + (size_t)kr * HD + ((size_t)c16 << 4));
        const uint32_t t = c16 >> 3, cc = c16 & 7u;
        const uint32_t off16 = t * (TK * 8u) + (kr >> 3) * 64u + (kr & 7u) * 8u
                             + (cc ^ (kr & 7u));
        *(uint4*)(dst + ((size_t)off16 << 4)) = v;
    }
    __syncthreads();
}

// 16 e4m3 bytes (one strip chunk) -> 4 packed half2 words
__device__ __forceinline__ void pd_pf5_f8_cvt8(uint32_t o[4], uint32_t w0,
                                               uint32_t w1) {
    const uint32_t w[2] = {w0, w1};
    #pragma unroll
    for (uint32_t qd = 0; qd < 2u; ++qd) {
        const __half2 lo2 = __half2(__nv_cvt_fp8x2_to_halfraw2(
            (__nv_fp8x2_storage_t)(w[qd] & 0xffffu), __NV_E4M3));
        const __half2 hi2 = __half2(__nv_cvt_fp8x2_to_halfraw2(
            (__nv_fp8x2_storage_t)(w[qd] >> 16), __NV_E4M3));
        o[qd * 2u] = *(const uint32_t*)&lo2;
        o[qd * 2u + 1u] = *(const uint32_t*)&hi2;
    }
}

// K side: strip chunk c covers elements l*16..l*16+15 of row kr - two
// 8-half SW128 units (c16 = 2l, 2l+1), one uint4 store each. TK is the
// K buffer's own row count (the c2 kernel stages TK/2 rows per CTA).
// Contains one internal barrier (reads before overlapping writes).
template <uint32_t TK, uint32_t HD>
__device__ __forceinline__ void pd_pf5_f8_expand_k(__half* kb, uint32_t nkeys,
                                                   uint32_t tid) {
    constexpr uint32_t LN = HD / 16u;
    constexpr uint32_t CHW = (TK * LN + 255u) / 256u;
    uint4 rg[CHW];
    const unsigned char* kstrip = (const unsigned char*)kb + (size_t)TK * HD;
    #pragma unroll
    for (uint32_t ci = 0; ci < CHW; ++ci) {
        const uint32_t c = tid + ci * 256u;
        if (c >= TK * LN) break;
        const uint32_t kr = c / LN;
        rg[ci] = (kr < nkeys) ? *(const uint4*)(kstrip + (size_t)c * 16u)
                              : make_uint4(0u, 0u, 0u, 0u);
    }
    __syncthreads();  // all K strip reads before the overlapping writes
    #pragma unroll
    for (uint32_t ci = 0; ci < CHW; ++ci) {
        const uint32_t c = tid + ci * 256u;
        if (c >= TK * LN) break;
        const uint32_t kr = c / LN, l = c - kr * LN;
        const uint32_t* w = (const uint32_t*)&rg[ci];
        #pragma unroll
        for (uint32_t h = 0; h < 2u; ++h) {
            const uint32_t c16 = l * 2u + h, t = c16 >> 3, cc = c16 & 7u;
            const uint32_t off16 = (kr >> 3) * 64u + (kr & 7u) * 8u + (cc ^ (kr & 7u));
            uint32_t o[4];
            pd_pf5_f8_cvt8(o, w[h * 2u], w[h * 2u + 1u]);
            *(uint4*)((unsigned char*)kb + (size_t)t * TK * 128u
                      + ((size_t)off16 << 4)) = *(const uint4*)o;
        }
    }
}

// V side: padded [TK][HD+8] rows. Same read/barrier/write shape.
template <uint32_t TK, uint32_t HD>
__device__ __forceinline__ void pd_pf5_f8_expand_v(__half* vb, uint32_t nkeys,
                                                   uint32_t tid) {
    constexpr uint32_t row_e = HD + 8u;
    constexpr uint32_t LN = HD / 16u;
    constexpr uint32_t CHW = (TK * LN + 255u) / 256u;
    uint4 rg[CHW];
    const unsigned char* vstrip = (const unsigned char*)vb + (size_t)TK * row_e;
    #pragma unroll
    for (uint32_t ci = 0; ci < CHW; ++ci) {
        const uint32_t c = tid + ci * 256u;
        if (c >= TK * LN) break;
        const uint32_t kr = c / LN;
        rg[ci] = (kr < nkeys) ? *(const uint4*)(vstrip + (size_t)c * 16u)
                              : make_uint4(0u, 0u, 0u, 0u);
    }
    __syncthreads();  // all V strip reads before the overlapping writes
    #pragma unroll
    for (uint32_t ci = 0; ci < CHW; ++ci) {
        const uint32_t c = tid + ci * 256u;
        if (c >= TK * LN) break;
        const uint32_t kr = c / LN, l = c - kr * LN;
        const uint32_t* w = (const uint32_t*)&rg[ci];
        unsigned char* dst = (unsigned char*)(vb + (size_t)kr * row_e + l * 16u);
        #pragma unroll
        for (uint32_t h = 0; h < 2u; ++h) {
            uint32_t o[4];
            pd_pf5_f8_cvt8(o, w[h * 2u], w[h * 2u + 1u]);
            *(uint4*)(dst + h * 16u) = *(const uint4*)o;
        }
    }
}

// combined form for the ::1 kernels (pf5 / pf5g): K then V (disjoint
// regions, no barrier between), trailing barrier before the mmas
template <uint32_t TK, uint32_t HD>
__device__ __forceinline__ void pd_pf5_f8_expand(__half* kb, __half* vb,
                                                 uint32_t nkeys, uint32_t tid) {
    pd_pf5_f8_expand_k<TK, HD>(kb, nkeys, tid);
    pd_pf5_f8_expand_v<TK, HD>(vb, nkeys, tid);
    __syncthreads();  // expanded halves visible before the mmas
}
#endif
// KB: bulk KV staging - full 16-token pool pages ride
// cp.async.bulk.tensor.2d (K via the SW128 f8s map straight into the mma's
// canonical layout, V linear into the strip), ragged head/tail rows keep the
// per-thread cp.async with the K swizzle applied inline. Retires the last
// commit-group bulk-movement idiom on sm_100a (dead there, CURRENT
// on sm_80-89 twins, which keep cp.async). Same bytes, same layouts, same
// mma - the gate is BIT-IDENTICAL output vs the F8QK baseline.
// arm/disarm the batched-runs pf5 launch for one coalesced pass.
// run_offs = device u32 prefix array [n_runs+1]; null disarms.
PD_EXPORT
int pd_pf_runs_register(const void* run_offs, unsigned int n_runs,
                        unsigned int max_n) {
    pd_pf_runs_offs = run_offs;
    pd_pf_runs_n = n_runs;
    pd_pf_runs_maxn = max_n;
    return 0;
}

template <uint32_t HD, uint32_t G, uint32_t TK, bool F8 = false, bool F8QK = false,
          bool KB = false>
__global__ void __launch_bounds__(256, 1) pd_attn_prefill_pf5_kernel(
    const __grid_constant__ PdTmap tmk, const __grid_constant__ PdTmap tmv,
    const float* __restrict__ q, const __half* __restrict__ pool_k,
    const __half* __restrict__ pool_v, const float* __restrict__ sinks,
    float* __restrict__ out, const unsigned int* __restrict__ positions,
    const unsigned int* __restrict__ slots,
    const uint32_t* __restrict__ block_tables, uint32_t blocks_per_slot,
    uint32_t n_heads, uint32_t kv_dim, uint32_t swa_window, uint32_t rows,
    float scale, const uint32_t* __restrict__ run_offs = nullptr) {
#if PD_TC5_OK
    // batched-runs arm: grid.z indexes a run table; every
    // pointer is base-valued and the prologue re-aims it at this run.
    // run_offs == nullptr is the classic one-run launch, bit-identical.
    if (run_offs != nullptr) {
        const uint32_t roff = run_offs[blockIdx.z];
        rows = run_offs[blockIdx.z + 1u] - roff;
        q += (size_t)roff * n_heads * HD;
        out += (size_t)roff * n_heads * HD;
        positions += roff;
        if (slots) slots += roff;
    }
    constexpr uint32_t MR = 128u;                  // mma rows = TQ*G
    constexpr uint32_t TQ = MR / G;
    constexpr uint32_t row_e = HD + 8u;            // padded f16 V rows
    constexpr uint32_t p_s = TK + 8u;              // P strip stride
    extern __shared__ __align__(1024) unsigned char sh_raw[];
    // SW128's xor phase rides the ABSOLUTE smem address (the
    // lesson) - pad the tile base to 1KB (launcher adds 1KB headroom)
    unsigned char* sh_ = sh_raw
        + ((1024u - ((uint32_t)__cvta_generic_to_shared(sh_raw) & 1023u)) & 1023u);
    __half* qs = (__half*)sh_;                     // SW128 [MR x HD]
    __half* ks = qs + (size_t)MR * HD;             // 2x SW128 [TK x HD]
    __half* vs = ks + 2u * (size_t)TK * HD;        // 2x padded [TK][row_e]
    __half* ps = vs + 2u * (size_t)TK * row_e;     // [MR][p_s] P
    float* s_corr = (float*)(ps + (size_t)MR * p_s);
    float* s_l = s_corr + MR;
    uint64_t* bdone = (uint64_t*)(s_l + MR);
    uint64_t* bkv = bdone + 1;                     // KB: 2 staging mbarriers
    __shared__ uint32_t tmem_slot[1];

    const uint32_t kvh = blockIdx.x, tq0 = blockIdx.y * TQ;
    const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
    if (tq0 >= rows) return;
    const uint32_t ntok = rows - tq0 < TQ ? rows - tq0 : TQ;

    if (tid == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                     ::"r"((uint32_t)__cvta_generic_to_shared(bdone)));
        if (KB) {
            const uint32_t m0 = (uint32_t)__cvta_generic_to_shared(bkv);
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(m0));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(m0 + 8u));
        }
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();
    if (tid < 32)
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                     ::"r"((uint32_t)__cvta_generic_to_shared(tmem_slot)), "r"(TK));
    __syncthreads();
    const uint32_t tmem = tmem_slot[0];

    // Q image: mma row r = token*G + g -> q row (tq0+token)*n_heads +
    // kvh*G + g, baked SW128 (16B chunk c of row r at tile c>>3, swizzled)
    for (uint32_t i = tid; i < MR * (HD / 8u); i += 256u) {
        const uint32_t r = i / (HD / 8u), c16 = i % (HD / 8u);
        const uint32_t tok = r / G, g = r % G;
        __half tmp[8];
        if (tok < ntok) {
            const float* src = q + ((size_t)(tq0 + tok) * n_heads
                             + (size_t)kvh * G + g) * HD + c16 * 8u;
            #pragma unroll
            for (uint32_t e = 0; e < 8u; ++e) tmp[e] = __float2half(src[e]);
        } else {
            #pragma unroll
            for (uint32_t e = 0; e < 8u; ++e) tmp[e] = __half(0.f);
        }
        const uint32_t t = c16 >> 3, c = c16 & 7u;
        const uint32_t off16 = (r >> 3) * 64u + (r & 7u) * 8u + (c ^ (r & 7u));
        *(uint4*)((unsigned char*)qs + ((size_t)t << 14) + ((size_t)off16 << 4)) =
            *(const uint4*)tmp;
    }
    if (F8QK) {
        // e4m3 Q image for the native QK^T - direct-cast, the same
        // numeric treatment KV8 gives K (no scales). 16-elem 16B chunks;
        // built in a separate region (upper half of the f16 Q image space,
        // which the native path never reads as f16).
        __syncthreads();
        unsigned char* q8 = (unsigned char*)qs + (size_t)MR * HD;   // MR*HD bytes
        for (uint32_t i = tid; i < MR * (HD / 16u); i += 256u) {
            const uint32_t r = i / (HD / 16u), c16 = i % (HD / 16u);
            const uint32_t tok = r / G, g = r % G;
            unsigned char tmp8[16];
            if (tok < ntok) {
                const float* src = q + ((size_t)(tq0 + tok) * n_heads
                                 + (size_t)kvh * G + g) * HD + c16 * 16u;
                #pragma unroll
                for (uint32_t e = 0; e < 8u; ++e) {
                    const __nv_fp8x2_storage_t p2 = __nv_cvt_float2_to_fp8x2(
                        make_float2(src[e * 2u], src[e * 2u + 1u]),
                        __NV_SATFINITE, __NV_E4M3);
                    tmp8[e * 2u] = (unsigned char)(p2 & 0xffu);
                    tmp8[e * 2u + 1u] = (unsigned char)(p2 >> 8);
                }
            } else {
                #pragma unroll
                for (uint32_t e = 0; e < 16u; ++e) tmp8[e] = 0u;
            }
            const uint32_t t = c16 >> 3, cc = c16 & 7u;
            const uint32_t off16 = t * (MR * 8u) + (r >> 3) * 64u + (r & 7u) * 8u
                                 + (cc ^ (r & 7u));
            *(uint4*)(q8 + ((size_t)off16 << 4)) = *(const uint4*)tmp8;
        }
    }
    if (tid < MR) { s_l[tid] = 0.0f; }
    // per-lane softmax state (warps 0-3 own rows)
    float m_run = -INFINITY, l_run = 0.0f;
    const uint32_t my_row = warp * 32u + lane;         // rows for warps 0-3
    const uint32_t my_tok = my_row / G;
    const uint32_t my_pos = my_tok < ntok ? positions[tq0 + my_tok] : 0u;
    const uint32_t my_lo = (swa_window > 0 && my_pos + 1 > swa_window)
        ? my_pos + 1 - swa_window : 0u;

    // O accumulators: warp -> 16-row m-tile, all HD dims (HD/8 n8-subs)
    constexpr uint32_t NSUB = HD / 8u;
    float o_acc[NSUB][4];
    #pragma unroll
    for (uint32_t s2 = 0; s2 < NSUB; ++s2)
        #pragma unroll
        for (uint32_t j = 0; j < 4u; ++j) o_acc[s2][j] = 0.0f;

    // walk: union span of the CTA's windows
    const uint32_t pos_last = positions[tq0 + ntok - 1u];
    const uint32_t lo0 = (swa_window > 0 && positions[tq0] + 1 > swa_window)
        ? positions[tq0] + 1 - swa_window : 0u;
    const uint32_t span = pos_last + 1u - lo0;
    const uint32_t ntiles = (span + TK - 1u) / TK;
    // spans never cross slot boundaries (caller contract) - one table
    const uint32_t slot = slots ? slots[0] : 0u;   // span = one sequence (v4 convention)
    const uint32_t* bt = block_tables + (size_t)slot * blocks_per_slot;
    // KB engages per-CTA: the SW128 landing keeps its phase only when the
    // smem row index tracks the pool row mod 16, i.e. a 16-aligned span
    // start. Full-attn CTAs (lo0 = 0) always qualify; SWA windows (ragged
    // lo0) take the classic path - same bytes either way.
    const bool use_bulk = KB && ((lo0 & 15u) == 0u);

    // cp.async staging into buffer bf: K into its SW128 slot, V padded.
    // F8: raw e4m3 bytes land in the buffers' upper byte strips (no zero
    // stores here - they'd overlap the strip range and race the cp.asyncs;
    // pd_pf5_f8_expand zero-fills rows >= nkeys during expansion).
    auto stage_kv = [&](uint32_t kt, uint32_t bf) {
        const uint32_t k0s = lo0 + kt * TK;
        const uint32_t nkeys_s = span - kt * TK < TK ? span - kt * TK : TK;
        __half* kb = ks + (size_t)bf * TK * HD;
        __half* vb = vs + (size_t)bf * TK * row_e;
        if constexpr (F8) {
            const unsigned char* pk8 = (const unsigned char*)pool_k;
            const unsigned char* pv8 = (const unsigned char*)pool_v;
            unsigned char* kstrip = (unsigned char*)kb + (size_t)TK * HD;
            unsigned char* vstrip = (unsigned char*)vb + (size_t)TK * row_e;
            if (use_bulk) {
                // full pages via TMA - K boxes land SW128 straight in
                // the mma's tile layout (no swz step on this path), V boxes
                // land linear in the strip for the unchanged expand. tid 0
                // issues; the tx expectation closes the phase when all bulk
                // bytes arrive (nf == 0 closes it immediately).
                const uint32_t nf = nkeys_s >> 4;      // full 16-token pages
                if (tid == 0) {
                    const uint32_t m = (uint32_t)__cvta_generic_to_shared(bkv + bf);
                    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                                 ::"r"(m), "r"(nf * 8192u));  // 2x2KB K + 4KB V per page
                    for (uint32_t j = 0; j < nf; ++j) {
                        const int y = (int)(bt[(k0s >> 4) + j] * 16u);
                        #pragma unroll
                        for (uint32_t t = 0; t < 2u; ++t) {
                            const int x = (int)(kvh * HD + t * 128u);
                            const uint32_t dst = (uint32_t)__cvta_generic_to_shared(
                                (unsigned char*)kb + t * (TK * 128u) + j * (16u * 128u));
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                                " [%0], [%1, {%2, %3}], [%4];"
                                ::"r"(dst), "l"(&tmk), "r"(x), "r"(y), "r"(m) : "memory");
                        }
                        const uint32_t dv = (uint32_t)__cvta_generic_to_shared(
                            vstrip + (size_t)j * (16u * HD));
                        const int xv = (int)(kvh * HD);
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                            " [%0], [%1, {%2, %3}], [%4];"
                            ::"r"(dv), "l"(&tmv), "r"(xv), "r"(y), "r"(m) : "memory");
                    }
                }
                // ragged tail (< one page): classic cp.async, K swizzled
                // inline - the same SW128 formula TMA applies to full pages
                for (uint32_t i = tid; i < TK * (HD / 16u); i += 256u) {
                    const uint32_t kr = i / (HD / 16u), l = i - kr * (HD / 16u);
                    if (kr >= nf * 16u && kr < nkeys_s) {
                        const uint32_t gpos = k0s + kr;
                        const uint32_t blk = bt[gpos >> 4];
                        const size_t b8 = (size_t)blk * 16u * kv_dim
                            + (size_t)(gpos & 15u) * kv_dim + (size_t)kvh * HD + l * 16u;
                        const uint32_t t = l >> 3, cc = l & 7u;
                        pd_attn_cpa16((char*)((unsigned char*)kb + t * (TK * 128u)
                                          + kr * 128u + ((cc ^ (kr & 7u)) << 4)),
                                      (const char*)(pk8 + b8));
                        pd_attn_cpa16((char*)(vstrip + (size_t)kr * HD + l * 16u),
                                      (const char*)(pv8 + b8));
                    }
                }
                pd_attn_cpa_commit();
                return;
            }
            for (uint32_t i = tid; i < TK * (HD / 16u); i += 256u) {
                const uint32_t kr = i / (HD / 16u), l = i - kr * (HD / 16u);
                if (kr < nkeys_s) {
                    const uint32_t gpos = k0s + kr;
                    const uint32_t blk = bt[gpos >> 4];
                    const size_t b8 = (size_t)blk * 16u * kv_dim
                        + (size_t)(gpos & 15u) * kv_dim + (size_t)kvh * HD + l * 16u;
                    pd_attn_cpa16((char*)(kstrip + (size_t)kr * HD + l * 16u),
                                  (const char*)(pk8 + b8));
                    pd_attn_cpa16((char*)(vstrip + (size_t)kr * HD + l * 16u),
                                  (const char*)(pv8 + b8));
                }
            }
            pd_attn_cpa_commit();
            return;
        } else {
            for (uint32_t i = tid; i < TK * (HD / 8u); i += 256u) {
                const uint32_t kr = i / (HD / 8u), c16 = i % (HD / 8u);
                const uint32_t t = c16 >> 3, c = c16 & 7u;
                const uint32_t off16 = (kr >> 3) * 64u + (kr & 7u) * 8u + (c ^ (kr & 7u));
                unsigned char* kdst = (unsigned char*)kb
                    + ((size_t)t * TK * 128u) + ((size_t)off16 << 4);
                unsigned char* vdst = (unsigned char*)(vb + (size_t)kr * row_e + c16 * 8u);
                if (kr < nkeys_s) {
                    const uint32_t gpos = k0s + kr;
                    const uint32_t blk = bt[gpos >> 4];
                    const size_t base = (size_t)blk * 16u * kv_dim
                        + (size_t)(gpos & 15u) * kv_dim + (size_t)kvh * HD + c16 * 8u;
                    pd_attn_cpa16(kdst, (const char*)(pool_k + base));
                    pd_attn_cpa16(vdst, (const char*)(pool_v + base));
                } else {
                    *(uint4*)kdst = make_uint4(0u, 0u, 0u, 0u);
                    *(uint4*)vdst = make_uint4(0u, 0u, 0u, 0u);
                }
            }
            pd_attn_cpa_commit();
        }
    };

    uint32_t done_ph = 0;
    if (ntiles) stage_kv(0u, 0u);
    for (uint32_t kt = 0; kt < ntiles; ++kt) {
        const uint32_t bf = kt & 1u;
        const uint32_t k0 = lo0 + kt * TK;
        const uint32_t nkeys = span - kt * TK < TK ? span - kt * TK : TK;
        const bool more = kt + 1u < ntiles;
        if (more) stage_kv(kt + 1u, bf ^ 1u);
        if (more) pd_attn_cpa_wait1(); else pd_attn_cpa_wait0();
        if (use_bulk) {
            // buffer bf's i-th use is tile 2i+bf -> phase parity (kt>>1)&1
            const uint32_t a = (uint32_t)__cvta_generic_to_shared(bkv + bf);
            asm volatile("{\n\t.reg .pred p;\nWKV%=:\n\t"
                         "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                         "@!p bra WKV%=;\n\t}" ::"r"(a), "r"((kt >> 1) & 1u));
        }
        __syncthreads();
        __half* kcur = ks + (size_t)bf * TK * HD;
        __half* vcur = vs + (size_t)bf * TK * row_e;
        if (F8 && F8QK) {
            // K stays e4m3 - swizzle-only port for the f8f6f4 mma,
            // V expands to f16 as before (PV is still the f16 pipe).
            //  bulk path: TMA already landed K in this exact layout.
            if (!use_bulk) pd_pf5_f8_swz_k<TK, HD>(kcur, tid);
            pd_pf5_f8_expand_v<TK, HD>(vcur, nkeys, tid);
            __syncthreads();
        } else if (F8) {
            pd_pf5_f8_expand<TK, HD>(kcur, vcur, nkeys, tid);
        }
        // S = Q·K^T on tcgen05 (one issuer)
        if (tid == 0 && F8QK) {
            // native e4m3 x e4m3 QK^T: same tile-stride expressions (128B
            // tiles), half the mma count (K=32/mma vs f16's 16)
            const uint32_t a16 = (uint32_t)__cvta_generic_to_shared(
                (unsigned char*)qs + (size_t)MR * HD) >> 4;
            const uint32_t b16 = (uint32_t)__cvta_generic_to_shared(kcur) >> 4;
            #pragma unroll
            for (uint32_t kb = 0; kb < HD / 32u; ++kb) {
                const uint32_t t = kb >> 2, c = (kb & 3u) * 2u;
                const uint64_t ad = pd_tc5_sdesc(a16 + t * (MR * 8u) + c);
                const uint64_t bd = pd_tc5_sdesc(b16 + t * (TK * 8u) + c);
                const uint32_t id = (1u << 4) | ((TK >> 3) << 17) | ((MR >> 4) << 24);
                asm volatile(
                    "{\n\t.reg .pred p;\n\t"
                    "setp.ne.b32 p, %4, 0;\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t}"
                    ::"r"(tmem), "l"(ad), "l"(bd), "r"(id), "r"(kb > 0 ? 1u : 0u));
            }
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                ::"r"((uint32_t)__cvta_generic_to_shared(bdone)));
        } else if (tid == 0) {
            const uint32_t a16 = (uint32_t)__cvta_generic_to_shared(qs) >> 4;
            const uint32_t b16 = (uint32_t)__cvta_generic_to_shared(kcur) >> 4;
            #pragma unroll
            for (uint32_t kb = 0; kb < HD / 16u; ++kb) {
                const uint32_t t = kb >> 2, c = (kb & 3u) * 2u;
                const uint64_t ad = pd_tc5_sdesc(a16 + t * 1024u + c);
                const uint64_t bd = pd_tc5_sdesc(b16 + t * (TK * 8u) + c);
                const uint32_t id = (1u << 4) | ((TK >> 3) << 17) | ((MR >> 4) << 24);
                asm volatile(
                    "{\n\t.reg .pred p;\n\t"
                    "setp.ne.b32 p, %4, 0;\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}"
                    ::"r"(tmem), "l"(ad), "l"(bd), "r"(id), "r"(kb > 0 ? 1u : 0u));
            }
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                ::"r"((uint32_t)__cvta_generic_to_shared(bdone)));
        }
        {
            const uint32_t a2 = (uint32_t)__cvta_generic_to_shared(bdone);
            asm volatile("{\n\t.reg .pred p;\nW%=:\n\t"
                         "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                         "@!p bra W%=;\n\t}" ::"r"(a2), "r"(done_ph));
        }
        done_ph ^= 1u;
        __syncthreads();
        // thread-local softmax: warps 0-3, lane owns row my_row
        if (warp < 4u) {
            float sv[TK / 32u][32];
            #pragma unroll
            for (uint32_t cc = 0; cc < TK / 32u; ++cc) {
                uint32_t rr[32];
                const uint32_t taddr = tmem + ((warp * 32u) << 16) + cc * 32u;
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
                    "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
                    "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
                    : "=r"(rr[0]),"=r"(rr[1]),"=r"(rr[2]),"=r"(rr[3]),"=r"(rr[4]),"=r"(rr[5]),"=r"(rr[6]),"=r"(rr[7]),
                      "=r"(rr[8]),"=r"(rr[9]),"=r"(rr[10]),"=r"(rr[11]),"=r"(rr[12]),"=r"(rr[13]),"=r"(rr[14]),"=r"(rr[15]),
                      "=r"(rr[16]),"=r"(rr[17]),"=r"(rr[18]),"=r"(rr[19]),"=r"(rr[20]),"=r"(rr[21]),"=r"(rr[22]),"=r"(rr[23]),
                      "=r"(rr[24]),"=r"(rr[25]),"=r"(rr[26]),"=r"(rr[27]),"=r"(rr[28]),"=r"(rr[29]),"=r"(rr[30]),"=r"(rr[31])
                    : "r"(taddr));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (uint32_t j = 0; j < 32u; ++j) sv[cc][j] = __uint_as_float(rr[j]);
            }
            // mask + scale in regs; per-key validity vs this row's window
            float m_tile = -INFINITY;
            #pragma unroll
            for (uint32_t cc = 0; cc < TK / 32u; ++cc)
                #pragma unroll
                for (uint32_t j = 0; j < 32u; ++j) {
                    const uint32_t kp = k0 + cc * 32u + j;
                    const bool ok = my_tok < ntok && kp >= my_lo && kp <= my_pos
                        && (cc * 32u + j) < nkeys;
                    sv[cc][j] = ok ? sv[cc][j] * scale : -INFINITY;
                    m_tile = fmaxf(m_tile, sv[cc][j]);
                }
            const float m_new = fmaxf(m_run, m_tile);
            const float corr = m_new > -INFINITY ? __expf(m_run - m_new) : 1.0f;
            float lsum = 0.0f;
            #pragma unroll
            for (uint32_t cc = 0; cc < TK / 32u; ++cc)
                #pragma unroll
                for (uint32_t j = 0; j < 32u; j += 2u) {
                    const float w0 = sv[cc][j] > -INFINITY ? __expf(sv[cc][j] - m_new) : 0.0f;
                    const float w1 = sv[cc][j + 1u] > -INFINITY ? __expf(sv[cc][j + 1u] - m_new) : 0.0f;
                    lsum += w0 + w1;
                    *(__half2*)(ps + (size_t)my_row * p_s + cc * 32u + j) =
                        __floats2half2_rn(w0, w1);
                }
            l_run = l_run * corr + lsum;
            m_run = m_new;
            s_corr[my_row] = corr;
        }
        __syncthreads();
        // P·V: 8 warps, warp = 16-row m-tile, HMMA over TK keys
        {
            const uint32_t r0 = warp * 16u;
            #pragma unroll
            for (uint32_t half = 0; half < 2u; ++half) {
                const uint32_t rr = r0 + (lane >> 2) + half * 8u;
                const float corr = s_corr[rr];
                #pragma unroll
                for (uint32_t s2 = 0; s2 < NSUB; ++s2) {
                    o_acc[s2][half * 2u] *= corr;
                    o_acc[s2][half * 2u + 1u] *= corr;
                }
            }
            for (uint32_t kk = 0; kk < TK; kk += 16u) {
                uint32_t af[4];
                const __half* ap = ps + (size_t)(r0 + (lane & 15u)) * p_s
                                 + kk + ((lane >> 4) ? 8u : 0u);
                pd_ldm_x4(af, (const unsigned char*)ap);
                #pragma unroll
                for (uint32_t s2 = 0; s2 < NSUB; ++s2) {
                    uint32_t bfr[2];
                    const __half* bp = vcur + (size_t)(kk + (lane & 15u)) * row_e
                                     + s2 * 8u;
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                                 : "=r"(bfr[0]), "=r"(bfr[1])
                                 : "r"((unsigned)__cvta_generic_to_shared(bp)));
                    pd_fa_mma16(o_acc[s2], af[0], af[1], af[2], af[3], bfr[0], bfr[1]);
                }
            }
        }
        __syncthreads();
    }
    // publish per-row l (softmax warps own it)
    if (warp < 4u && my_tok < ntok) s_l[my_row] = l_run;
    __syncthreads();
    // epilogue: o / l -> out[(tq0+tok)*n_heads + kvh*G + g][dim]
    {
        const uint32_t r0 = warp * 16u;
        #pragma unroll
        for (uint32_t half = 0; half < 2u; ++half) {
            const uint32_t rr = r0 + (lane >> 2) + half * 8u;
            const uint32_t tok = rr / G, g = rr % G;
            if (tok >= ntok) continue;
            const float inv_l = s_l[rr] > 0.0f ? 1.0f / s_l[rr] : 0.0f;
            float* dst = out + ((size_t)(tq0 + tok) * n_heads
                        + (size_t)kvh * G + g) * HD;
            #pragma unroll
            for (uint32_t s2 = 0; s2 < NSUB; ++s2) {
                dst[s2 * 8u + 2u * (lane & 3u)] = o_acc[s2][half * 2u] * inv_l;
                dst[s2 * 8u + 2u * (lane & 3u) + 1u] = o_acc[s2][half * 2u + 1u] * inv_l;
            }
        }
    }
    __syncthreads();
    if (tid < 32)
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                     ::"r"(tmem), "r"(TK));
#else
    (void)q; (void)pool_k; (void)pool_v; (void)sinks; (void)out; (void)positions;
    (void)slots; (void)block_tables; (void)blocks_per_slot; (void)n_heads;
    (void)kv_dim; (void)swa_window; (void)rows; (void)scale;
#endif
}

// pf5g (class rework 2b): the GLB (hd512, G8, full-causal) pf5
// variant. Register budget forces MR=64 (TQ=8); O splits 8 warps as 4
// m-tiles x 2 dim-halves (128 f32/thread); softmax thread-local on warps
// 0-1 (lane = row). S = QK^T runs tcgen05 kind::f16 at M=128 with Q
// PACKED to 64 real rows: the mma's atom walk for rows 64-127 reads the
// next tile column (garbage) into tmem lanes 64-127, which are never
// ld'd - M=64 is not an option (it silently no-ops, probed).
// TK=32 double-buffered == TK=64 single-buffered (2606 vs 2607 us); we
// ship TK=32/2x for the smaller softmax-warp register load. Measured:
// 3372.5 -> 2606.1 us at the 2048-row tick (158 vs 122 TF), oracle-tied.
template <uint32_t HD, uint32_t G, uint32_t TK, bool F8 = false>
__global__ void __launch_bounds__(256, 1) pd_attn_prefill_pf5g_kernel(
    const float* __restrict__ q, const __half* __restrict__ pool_k,
    const __half* __restrict__ pool_v, const float* __restrict__ sinks,
    float* __restrict__ out, const unsigned int* __restrict__ positions,
    const unsigned int* __restrict__ slots,
    const uint32_t* __restrict__ block_tables, uint32_t blocks_per_slot,
    uint32_t n_heads, uint32_t kv_dim, uint32_t swa_window, uint32_t rows,
    float scale) {
#if PD_TC5_OK
    constexpr uint32_t MR = 64u;                   // real O/P rows
    constexpr uint32_t MS = 128u;                  // S-GEMM M (rows 64+ pad:
                                                   // the M=64 tmem D layout
                                                   // is nonstandard; M=128's
                                                   // lane=row is proven)
    constexpr uint32_t TQ = MR / G;
    constexpr uint32_t row_e = HD + 8u;
    constexpr uint32_t p_s = TK + 8u;
    extern __shared__ __align__(1024) unsigned char shg5_raw[];
    unsigned char* shg = shg5_raw
        + ((1024u - ((uint32_t)__cvta_generic_to_shared(shg5_raw) & 1023u)) & 1023u);
    // Q is PACKED: only MR real rows staged (8KB per 128B tile column). The
    // mma still runs M=MS=128 - its atom walk for rows 64-127 reads the next
    // tile column's data (and, on the last column, the K buffer) as garbage;
    // those land in tmem lanes 64-127 which we never ld. Saves 64KB -> K/V
    // double-buffering fits at TK=32.
    constexpr uint32_t NBUF = TK >= 64u ? 1u : 2u; // TK=64 won't fit 2x
    __half* qs = (__half*)shg;                     // SW128 packed [MR x HD]
    __half* ks = qs + (size_t)MR * HD;             // NBUF x SW128 [TK x HD]
    __half* vs = ks + NBUF * (size_t)TK * HD;      // NBUF x padded [TK][row_e]
    __half* ps = vs + NBUF * (size_t)TK * row_e;   // [MR][p_s]
    float* s_corr = (float*)(ps + (size_t)MR * p_s);
    float* s_l = s_corr + MR;
    uint64_t* bdone = (uint64_t*)(s_l + MR);
    __shared__ uint32_t tmem_slot[1];

    const uint32_t kvh = blockIdx.x, tq0 = blockIdx.y * TQ;
    const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
    if (tq0 >= rows) return;
    const uint32_t ntok = rows - tq0 < TQ ? rows - tq0 : TQ;

    if (tid == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                     ::"r"((uint32_t)__cvta_generic_to_shared(bdone)));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();
    constexpr uint32_t TMEM_COLS = TK < 32u ? 32u : TK;
    if (tid < 32)
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                     ::"r"((uint32_t)__cvta_generic_to_shared(tmem_slot)), "r"(TMEM_COLS));
    __syncthreads();
    const uint32_t tmem = tmem_slot[0];

    for (uint32_t i = tid; i < MR * (HD / 8u); i += 256u) {
        const uint32_t r = i / (HD / 8u), c16 = i % (HD / 8u);
        const uint32_t tok = r / G, g = r % G;
        __half tmp[8];
        if (tok < ntok) {
            const float* srcp = q + ((size_t)(tq0 + tok) * n_heads
                              + (size_t)kvh * G + g) * HD + c16 * 8u;
            #pragma unroll
            for (uint32_t e = 0; e < 8u; ++e) tmp[e] = __float2half(srcp[e]);
        } else {
            #pragma unroll
            for (uint32_t e = 0; e < 8u; ++e) tmp[e] = __half(0.f);
        }
        const uint32_t t = c16 >> 3, c = c16 & 7u;
        const uint32_t off16 = (r >> 3) * 64u + (r & 7u) * 8u + (c ^ (r & 7u));
        *(uint4*)((unsigned char*)qs + ((size_t)t * MR * 128u) + ((size_t)off16 << 4)) =
            *(const uint4*)tmp;
    }
    if (tid < MR) s_l[tid] = 0.0f;
    float m_run = -INFINITY, l_run = 0.0f;
    const uint32_t my_row = warp * 32u + lane;         // warps 0-1 own rows
    const uint32_t my_tok = my_row / G;
    const uint32_t my_pos = (warp < 2u && my_tok < ntok)
        ? positions[tq0 + my_tok] : 0u;
    const uint32_t my_lo = (swa_window > 0 && my_pos + 1 > swa_window)
        ? my_pos + 1 - swa_window : 0u;

    // O: warp w -> m-tile (w>>1), dim half (w&1): 16 rows x 256 dims
    constexpr uint32_t DHALF = HD / 2u;
    constexpr uint32_t NSUB = DHALF / 8u;
    const uint32_t mt0 = (warp >> 1) * 16u;
    const uint32_t d0 = (warp & 1u) * DHALF;
    float o_acc[NSUB][4];
    #pragma unroll
    for (uint32_t s2 = 0; s2 < NSUB; ++s2)
        #pragma unroll
        for (uint32_t j = 0; j < 4u; ++j) o_acc[s2][j] = 0.0f;

    const uint32_t pos_last = positions[tq0 + ntok - 1u];
    const uint32_t lo0 = (swa_window > 0 && positions[tq0] + 1 > swa_window)
        ? positions[tq0] + 1 - swa_window : 0u;
    const uint32_t span = pos_last + 1u - lo0;
    const uint32_t ntiles = (span + TK - 1u) / TK;
    const uint32_t slot = slots ? slots[0] : 0u;   // span = one sequence (v4 convention)
    const uint32_t* bt = block_tables + (size_t)slot * blocks_per_slot;

    auto stage_kv = [&](uint32_t kt, uint32_t bf) {
        const uint32_t k0s = lo0 + kt * TK;
        const uint32_t nkeys_s = span - kt * TK < TK ? span - kt * TK : TK;
        __half* kb = ks + (size_t)bf * TK * HD;
        __half* vb = vs + (size_t)bf * TK * row_e;
        if constexpr (F8) {
            // e4m3 bytes to the upper strips; zero-fill happens in
            // pd_pf5_f8_expand (stage-time zeros would race the strips)
            const unsigned char* pk8 = (const unsigned char*)pool_k;
            const unsigned char* pv8 = (const unsigned char*)pool_v;
            unsigned char* kstrip = (unsigned char*)kb + (size_t)TK * HD;
            unsigned char* vstrip = (unsigned char*)vb + (size_t)TK * row_e;
            for (uint32_t i = tid; i < TK * (HD / 16u); i += 256u) {
                const uint32_t kr = i / (HD / 16u), l = i - kr * (HD / 16u);
                if (kr < nkeys_s) {
                    const uint32_t gpos = k0s + kr;
                    const uint32_t blk = bt[gpos >> 4];
                    const size_t b8 = (size_t)blk * 16u * kv_dim
                        + (size_t)(gpos & 15u) * kv_dim + (size_t)kvh * HD + l * 16u;
                    pd_attn_cpa16((char*)(kstrip + (size_t)kr * HD + l * 16u),
                                  (const char*)(pk8 + b8));
                    pd_attn_cpa16((char*)(vstrip + (size_t)kr * HD + l * 16u),
                                  (const char*)(pv8 + b8));
                }
            }
            pd_attn_cpa_commit();
            return;
        } else {
            for (uint32_t i = tid; i < TK * (HD / 8u); i += 256u) {
                const uint32_t kr = i / (HD / 8u), c16 = i % (HD / 8u);
                const uint32_t t = c16 >> 3, c = c16 & 7u;
                const uint32_t off16 = (kr >> 3) * 64u + (kr & 7u) * 8u + (c ^ (kr & 7u));
                unsigned char* kdst = (unsigned char*)kb
                    + ((size_t)t * TK * 128u) + ((size_t)off16 << 4);
                unsigned char* vdst = (unsigned char*)(vb + (size_t)kr * row_e + c16 * 8u);
                if (kr < nkeys_s) {
                    const uint32_t gpos = k0s + kr;
                    const uint32_t blk = bt[gpos >> 4];
                    const size_t base = (size_t)blk * 16u * kv_dim
                        + (size_t)(gpos & 15u) * kv_dim + (size_t)kvh * HD + c16 * 8u;
                    pd_attn_cpa16(kdst, (const char*)(pool_k + base));
                    pd_attn_cpa16(vdst, (const char*)(pool_v + base));
                } else {
                    *(uint4*)kdst = make_uint4(0u, 0u, 0u, 0u);
                    *(uint4*)vdst = make_uint4(0u, 0u, 0u, 0u);
                }
            }
            pd_attn_cpa_commit();
        }
    };

    uint32_t done_ph = 0;
    if (NBUF == 2u && ntiles) stage_kv(0u, 0u);
    for (uint32_t kt = 0; kt < ntiles; ++kt) {
        const uint32_t bf = NBUF == 2u ? (kt & 1u) : 0u;
        const uint32_t k0 = lo0 + kt * TK;
        const uint32_t nkeys = span - kt * TK < TK ? span - kt * TK : TK;
        if (NBUF == 2u) {
            const bool more = kt + 1u < ntiles;
            if (more) stage_kv(kt + 1u, bf ^ 1u);
            if (more) pd_attn_cpa_wait1(); else pd_attn_cpa_wait0();
        } else {
            stage_kv(kt, 0u);
            pd_attn_cpa_wait0();
        }
        __syncthreads();
        __half* kcur = ks + (size_t)bf * TK * HD;
        __half* vcur = vs + (size_t)bf * TK * row_e;
        if (F8) pd_pf5_f8_expand<TK, HD>(kcur, vcur, nkeys, tid);
        if (tid == 0) {
            const uint32_t a16 = (uint32_t)__cvta_generic_to_shared(qs) >> 4;
            const uint32_t b16 = (uint32_t)__cvta_generic_to_shared(kcur) >> 4;
            #pragma unroll
            for (uint32_t kb2 = 0; kb2 < HD / 16u; ++kb2) {
                const uint32_t t = kb2 >> 2, c = (kb2 & 3u) * 2u;
                const uint64_t ad = pd_tc5_sdesc(a16 + t * (MR * 8u) + c);
                const uint64_t bd = pd_tc5_sdesc(b16 + t * (TK * 8u) + c);
                const uint32_t id = (1u << 4) | ((TK >> 3) << 17) | ((MS >> 4) << 24);
                asm volatile(
                    "{\n\t.reg .pred p;\n\t"
                    "setp.ne.b32 p, %4, 0;\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}"
                    ::"r"(tmem), "l"(ad), "l"(bd), "r"(id), "r"(kb2 > 0 ? 1u : 0u));
            }
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                ::"r"((uint32_t)__cvta_generic_to_shared(bdone)));
        }
        {
            const uint32_t a2 = (uint32_t)__cvta_generic_to_shared(bdone);
            asm volatile("{\n\t.reg .pred p;\nW%=:\n\t"
                         "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                         "@!p bra W%=;\n\t}" ::"r"(a2), "r"(done_ph));
        }
        done_ph ^= 1u;
        __syncthreads();
        if (warp < 2u) {
            constexpr uint32_t NCC = (TK + 31u) / 32u;
            float sv[NCC][32];
            #pragma unroll
            for (uint32_t cc = 0; cc < NCC; ++cc) {
                uint32_t rr[32];
                const uint32_t taddr = tmem + ((warp * 32u) << 16) + cc * 32u;
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
                    "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
                    "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
                    : "=r"(rr[0]),"=r"(rr[1]),"=r"(rr[2]),"=r"(rr[3]),"=r"(rr[4]),"=r"(rr[5]),"=r"(rr[6]),"=r"(rr[7]),
                      "=r"(rr[8]),"=r"(rr[9]),"=r"(rr[10]),"=r"(rr[11]),"=r"(rr[12]),"=r"(rr[13]),"=r"(rr[14]),"=r"(rr[15]),
                      "=r"(rr[16]),"=r"(rr[17]),"=r"(rr[18]),"=r"(rr[19]),"=r"(rr[20]),"=r"(rr[21]),"=r"(rr[22]),"=r"(rr[23]),
                      "=r"(rr[24]),"=r"(rr[25]),"=r"(rr[26]),"=r"(rr[27]),"=r"(rr[28]),"=r"(rr[29]),"=r"(rr[30]),"=r"(rr[31])
                    : "r"(taddr));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (uint32_t j = 0; j < 32u; ++j) sv[cc][j] = __uint_as_float(rr[j]);
            }
            float m_tile = -INFINITY;
            #pragma unroll
            for (uint32_t cc = 0; cc < NCC; ++cc)
                #pragma unroll
                for (uint32_t j = 0; j < 32u; ++j) {
                    const uint32_t kp = k0 + cc * 32u + j;
                    const bool ok = my_tok < ntok && kp >= my_lo && kp <= my_pos
                        && (cc * 32u + j) < nkeys;
                    sv[cc][j] = ok ? sv[cc][j] * scale : -INFINITY;
                    m_tile = fmaxf(m_tile, sv[cc][j]);
                }
            const float m_new = fmaxf(m_run, m_tile);
            const float corr = m_new > -INFINITY ? __expf(m_run - m_new) : 1.0f;
            float lsum = 0.0f;
            #pragma unroll
            for (uint32_t cc = 0; cc < NCC; ++cc)
                #pragma unroll
                for (uint32_t j = 0; j < 32u; j += 2u) {
                    const float w0 = sv[cc][j] > -INFINITY ? __expf(sv[cc][j] - m_new) : 0.0f;
                    const float w1 = sv[cc][j + 1u] > -INFINITY ? __expf(sv[cc][j + 1u] - m_new) : 0.0f;
                    lsum += w0 + w1;
                    *(__half2*)(ps + (size_t)my_row * p_s + cc * 32u + j) =
                        __floats2half2_rn(w0, w1);
                }
            l_run = l_run * corr + lsum;
            m_run = m_new;
            s_corr[my_row] = corr;
        }
        __syncthreads();
        {
            #pragma unroll
            for (uint32_t half = 0; half < 2u; ++half) {
                const uint32_t rr = mt0 + (lane >> 2) + half * 8u;
                const float corr = s_corr[rr];
                #pragma unroll
                for (uint32_t s2 = 0; s2 < NSUB; ++s2) {
                    o_acc[s2][half * 2u] *= corr;
                    o_acc[s2][half * 2u + 1u] *= corr;
                }
            }
            for (uint32_t kk = 0; kk < TK; kk += 16u) {
                uint32_t af[4];
                const __half* ap = ps + (size_t)(mt0 + (lane & 15u)) * p_s
                                 + kk + ((lane >> 4) ? 8u : 0u);
                pd_ldm_x4(af, (const unsigned char*)ap);
                #pragma unroll
                for (uint32_t s2 = 0; s2 < NSUB; ++s2) {
                    uint32_t bfr[2];
                    const __half* bp = vcur + (size_t)(kk + (lane & 15u)) * row_e
                                     + d0 + s2 * 8u;
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                                 : "=r"(bfr[0]), "=r"(bfr[1])
                                 : "r"((unsigned)__cvta_generic_to_shared(bp)));
                    pd_fa_mma16(o_acc[s2], af[0], af[1], af[2], af[3], bfr[0], bfr[1]);
                }
            }
        }
        __syncthreads();
    }
    if (warp < 2u && my_tok < ntok) s_l[my_row] = l_run;
    __syncthreads();
    {
        #pragma unroll
        for (uint32_t half = 0; half < 2u; ++half) {
            const uint32_t rr = mt0 + (lane >> 2) + half * 8u;
            const uint32_t tok = rr / G, g = rr % G;
            if (tok >= ntok) continue;
            const float inv_l = s_l[rr] > 0.0f ? 1.0f / s_l[rr] : 0.0f;
            float* dst = out + ((size_t)(tq0 + tok) * n_heads
                        + (size_t)kvh * G + g) * HD + d0;
            #pragma unroll
            for (uint32_t s2 = 0; s2 < NSUB; ++s2) {
                dst[s2 * 8u + 2u * (lane & 3u)] = o_acc[s2][half * 2u] * inv_l;
                dst[s2 * 8u + 2u * (lane & 3u) + 1u] = o_acc[s2][half * 2u + 1u] * inv_l;
            }
        }
    }
    __syncthreads();
    if (tid < 32)
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                     ::"r"(tmem), "r"(TMEM_COLS));
#else
    (void)q; (void)pool_k; (void)pool_v; (void)sinks; (void)out; (void)positions;
    (void)slots; (void)block_tables; (void)blocks_per_slot; (void)n_heads;
    (void)kv_dim; (void)swa_window; (void)rows; (void)scale;
#endif
}


// pf5g-c2: cluster-pair pf5g. cta_group::2 collective M=128 = 2 CTAs x 64
// real Q rows (16 queries per pair) - no garbage mma half. Probe-proven
// layout (clm128_probe): each CTA stages its 64 A rows as compact 64-row
// SW128 tiles (the same walk as the ::1 packed-Q form) plus its N-half of B
// (TK/2 keys); D per CTA = its 64 rows x all TK collective cols FOLDED as
// 128 lanes x TK/2 tmem cols (lanes 64-127 hold key cols TK/2..TK-1). Net:
// K bytes/query and S-GEMM issue/query both halve. V stays full per CTA
// (P.V is CUDA-core and reads local smem). Softmax goes quadrant-parallel
// on warps 0-3 (tmem ld is quadrant-locked) with a smem cross-half
// max/sum exchange; warps 0-1 own the running row state. The tile span is
// PAIR-uniform - the collective loop must iterate identically in both
// ranks - so early rows just mask the extra tail keys (<= 15 positions).
// __cluster_dims__ is a DECLARATION attribute: unlike the PD_TC5_OK body
// guard it is evaluated on every device pass, and nvcc hard-errors on it
// below sm_90 - one unguarded cluster kernel breaks the whole multi-arch
// fatbin for Ampere/Ada. Keep the definition out of <900 device passes;
// the host pass (no __CUDA_ARCH__) still sees it so the cc-gated launcher
// compiles.
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 900)
template <uint32_t HD, uint32_t G, uint32_t TK, bool F8 = false>
__global__ void __launch_bounds__(256, 1) __cluster_dims__(2, 1, 1)
pd_attn_prefill_pf5g_c2_kernel(
    const float* __restrict__ q, const __half* __restrict__ pool_k,
    const __half* __restrict__ pool_v, const float* __restrict__ sinks,
    float* __restrict__ out, const unsigned int* __restrict__ positions,
    const unsigned int* __restrict__ slots,
    const uint32_t* __restrict__ block_tables, uint32_t blocks_per_slot,
    uint32_t n_heads, uint32_t kv_dim, uint32_t swa_window, uint32_t rows,
    float scale, const uint32_t* __restrict__ run_offs = nullptr) {
#if PD_TC5_OK
    // batched-runs arm: same convention as pf5 - grid.z indexes
    // the armed run table, pointers re-aim, nullptr = classic launch.
    if (run_offs != nullptr) {
        const uint32_t roff = run_offs[blockIdx.z];
        rows = run_offs[blockIdx.z + 1u] - roff;
        q += (size_t)roff * n_heads * HD;
        out += (size_t)roff * n_heads * HD;
        positions += roff;
        if (slots) slots += roff;
    }
    constexpr uint32_t MR = 64u;                   // real rows per CTA
    constexpr uint32_t TKH = TK / 2u;              // this CTA's K N-half
    constexpr uint32_t TQ = MR / G;
    constexpr uint32_t row_e = HD + 8u;
    constexpr uint32_t p_s = TK + 8u;
    constexpr uint32_t NBUF = TK >= 64u ? 1u : 2u;
    extern __shared__ __align__(1024) unsigned char shgc2_raw[];
    unsigned char* shg = shgc2_raw
        + ((1024u - ((uint32_t)__cvta_generic_to_shared(shgc2_raw) & 1023u)) & 1023u);
    __half* qs = (__half*)shg;                     // SW128 packed [MR x HD]
    __half* ks = qs + (size_t)MR * HD;             // NBUF x SW128 [TKH x HD]
    __half* vs = ks + NBUF * (size_t)TKH * HD;     // NBUF x padded [TK][row_e]
    __half* ps = vs + NBUF * (size_t)TK * row_e;   // [MR][p_s]
    float* s_corr = (float*)(ps + (size_t)MR * p_s);
    float* s_x1 = s_corr + MR;                     // half-1 partial max, then lsum
    float* s_mnew = s_x1 + MR;
    float* s_l = s_mnew + MR;
    uint64_t* bdone = (uint64_t*)(s_l + MR);       // [1] mma done (per CTA)
    uint64_t* bpeer = bdone + 1u;                  // [NBUF] rank1 K-half ready
    __shared__ uint32_t tmem_slot[1];

    const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
    uint32_t crank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(crank));
    const uint32_t kvh = blockIdx.y;
    const uint32_t tq_pair = (blockIdx.x >> 1) * 2u * TQ;   // pair's first query
    const uint32_t tq0 = tq_pair + crank * TQ;              // this CTA's first
    // no early return: both ranks must run the collective loop even when this
    // rank has no queries (rank1 on a short tail) - a return would deadlock
    // the peer's barriers and starve the collective mma of its B half.
    const uint32_t ntok = tq0 < rows ? (rows - tq0 < TQ ? rows - tq0 : TQ) : 0u;

    if (tid == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                     ::"r"((uint32_t)__cvta_generic_to_shared(bdone)));
        #pragma unroll
        for (uint32_t s = 0; s < NBUF; ++s)
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bpeer[s])));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();
    asm volatile("barrier.cluster.arrive;");
    asm volatile("barrier.cluster.wait;");
    // tcgen05.alloc wants a power-of-2 column count >= 32
    constexpr uint32_t TMEM_COLS = TKH <= 32u ? 32u : (TKH <= 64u ? 64u : 128u);
    if (tid < 32)
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
                     ::"r"((uint32_t)__cvta_generic_to_shared(tmem_slot)), "r"(TMEM_COLS));
    __syncthreads();
    const uint32_t tmem = tmem_slot[0];
    auto peer_addr = [&](void* p) -> uint32_t {
        const uint32_t pa0 = (uint32_t)__cvta_generic_to_shared(p);
        uint32_t pa;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                     : "=r"(pa) : "r"(pa0), "r"(crank ^ 1u));
        return pa;
    };
    auto bar_wait = [&](uint64_t* bar, uint32_t parity) {
        const uint32_t a2 = (uint32_t)__cvta_generic_to_shared(bar);
        asm volatile("{\n\t.reg .pred p;\nW%=:\n\t"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                     "@!p bra W%=;\n\t}" ::"r"(a2), "r"(parity));
    };

    // pack this CTA's 64 Q rows (compact 64-row tiles - probe awalk1)
    for (uint32_t i = tid; i < MR * (HD / 8u); i += 256u) {
        const uint32_t r = i / (HD / 8u), c16 = i % (HD / 8u);
        const uint32_t tok = r / G, g = r % G;
        __half tmp[8];
        if (tok < ntok) {
            const float* srcp = q + ((size_t)(tq0 + tok) * n_heads
                              + (size_t)kvh * G + g) * HD + c16 * 8u;
            #pragma unroll
            for (uint32_t e = 0; e < 8u; ++e) tmp[e] = __float2half(srcp[e]);
        } else {
            #pragma unroll
            for (uint32_t e = 0; e < 8u; ++e) tmp[e] = __half(0.f);
        }
        const uint32_t t = c16 >> 3, c = c16 & 7u;
        const uint32_t off16 = (r >> 3) * 64u + (r & 7u) * 8u + (c ^ (r & 7u));
        *(uint4*)((unsigned char*)qs + ((size_t)t * MR * 128u) + ((size_t)off16 << 4)) =
            *(const uint4*)tmp;
    }
    if (tid < MR) s_l[tid] = 0.0f;
    float m_run = -INFINITY, l_run = 0.0f;          // state lives in warps 0-1
    const uint32_t qh = warp >> 1;                  // col half (warps 0-3)
    const uint32_t my_row = (warp & 1u) * 32u + lane;
    const uint32_t my_tok = my_row / G;
    const uint32_t my_pos = (warp < 4u && my_tok < ntok)
        ? positions[tq0 + my_tok] : 0u;
    const uint32_t my_lo = (swa_window > 0 && my_pos + 1 > swa_window)
        ? my_pos + 1 - swa_window : 0u;

    // O: warp w -> m-tile (w>>1), dim half (w&1): 16 rows x 256 dims
    constexpr uint32_t DHALF = HD / 2u;
    constexpr uint32_t NSUB = DHALF / 8u;
    const uint32_t mt0 = (warp >> 1) * 16u;
    const uint32_t d0 = (warp & 1u) * DHALF;
    float o_acc[NSUB][4];
    #pragma unroll
    for (uint32_t s2 = 0; s2 < NSUB; ++s2)
        #pragma unroll
        for (uint32_t j = 0; j < 4u; ++j) o_acc[s2][j] = 0.0f;

    // PAIR-uniform span (both ranks must agree on ntiles)
    const uint32_t last_q = tq_pair + 2u * TQ <= rows ? tq_pair + 2u * TQ - 1u
                                                      : rows - 1u;
    const uint32_t pos_last = positions[last_q];
    const uint32_t lo0 = (swa_window > 0 && positions[tq_pair] + 1 > swa_window)
        ? positions[tq_pair] + 1 - swa_window : 0u;
    const uint32_t span = pos_last + 1u - lo0;
    const uint32_t ntiles = (span + TK - 1u) / TK;
    const uint32_t slot = slots ? slots[0] : 0u;
    const uint32_t* bt = block_tables + (size_t)slot * blocks_per_slot;

    auto stage_kv = [&](uint32_t kt, uint32_t bf) {
        const uint32_t k0s = lo0 + kt * TK;
        const uint32_t nkeys_s = span - kt * TK < TK ? span - kt * TK : TK;
        __half* kb = ks + (size_t)bf * TKH * HD;
        __half* vb = vs + (size_t)bf * TK * row_e;
        if constexpr (F8) {
            // e4m3 to the upper strips (K: this CTA's TKH-row half, local
            // rows; V: all TK); zero-fill lives in the expansion
            const unsigned char* pk8 = (const unsigned char*)pool_k;
            const unsigned char* pv8 = (const unsigned char*)pool_v;
            unsigned char* kstrip = (unsigned char*)kb + (size_t)TKH * HD;
            unsigned char* vstrip = (unsigned char*)vb + (size_t)TK * row_e;
            for (uint32_t i = tid; i < TKH * (HD / 16u); i += 256u) {
                const uint32_t krl = i / (HD / 16u), l = i - krl * (HD / 16u);
                const uint32_t kr = crank * TKH + krl;
                if (kr < nkeys_s) {
                    const uint32_t gpos = k0s + kr;
                    const uint32_t blk = bt[gpos >> 4];
                    const size_t b8 = (size_t)blk * 16u * kv_dim
                        + (size_t)(gpos & 15u) * kv_dim + (size_t)kvh * HD + l * 16u;
                    pd_attn_cpa16((char*)(kstrip + (size_t)krl * HD + l * 16u),
                                  (const char*)(pk8 + b8));
                }
            }
            for (uint32_t i = tid; i < TK * (HD / 16u); i += 256u) {
                const uint32_t kr = i / (HD / 16u), l = i - kr * (HD / 16u);
                if (kr < nkeys_s) {
                    const uint32_t gpos = k0s + kr;
                    const uint32_t blk = bt[gpos >> 4];
                    const size_t b8 = (size_t)blk * 16u * kv_dim
                        + (size_t)(gpos & 15u) * kv_dim + (size_t)kvh * HD + l * 16u;
                    pd_attn_cpa16((char*)(vstrip + (size_t)kr * HD + l * 16u),
                                  (const char*)(pv8 + b8));
                }
            }
            pd_attn_cpa_commit();
            return;
        } else {
            // K: this CTA's N-half = collective key cols [crank*TKH, +TKH)
            for (uint32_t i = tid; i < TKH * (HD / 8u); i += 256u) {
                const uint32_t krl = i / (HD / 8u), c16 = i % (HD / 8u);
                const uint32_t kr = crank * TKH + krl;
                const uint32_t t = c16 >> 3, c = c16 & 7u;
                const uint32_t off16 = (krl >> 3) * 64u + (krl & 7u) * 8u + (c ^ (krl & 7u));
                unsigned char* kdst = (unsigned char*)kb
                    + ((size_t)t * TKH * 128u) + ((size_t)off16 << 4);
                if (kr < nkeys_s) {
                    const uint32_t gpos = k0s + kr;
                    const uint32_t blk = bt[gpos >> 4];
                    const size_t base = (size_t)blk * 16u * kv_dim
                        + (size_t)(gpos & 15u) * kv_dim + (size_t)kvh * HD + c16 * 8u;
                    pd_attn_cpa16(kdst, (const char*)(pool_k + base));
                } else {
                    *(uint4*)kdst = make_uint4(0u, 0u, 0u, 0u);
                }
            }
            // V: all TK keys (P.V is CTA-local)
            for (uint32_t i = tid; i < TK * (HD / 8u); i += 256u) {
                const uint32_t kr = i / (HD / 8u), c16 = i % (HD / 8u);
                unsigned char* vdst = (unsigned char*)(vb + (size_t)kr * row_e + c16 * 8u);
                if (kr < nkeys_s) {
                    const uint32_t gpos = k0s + kr;
                    const uint32_t blk = bt[gpos >> 4];
                    const size_t base = (size_t)blk * 16u * kv_dim
                        + (size_t)(gpos & 15u) * kv_dim + (size_t)kvh * HD + c16 * 8u;
                    pd_attn_cpa16(vdst, (const char*)(pool_v + base));
                } else {
                    *(uint4*)vdst = make_uint4(0u, 0u, 0u, 0u);
                }
            }
            pd_attn_cpa_commit();
        }
    };

    uint32_t done_ph = 0, peer_ph = 0;
    if (NBUF == 2u && ntiles) stage_kv(0u, 0u);
    for (uint32_t kt = 0; kt < ntiles; ++kt) {
        const uint32_t bf = NBUF == 2u ? (kt & 1u) : 0u;
        const uint32_t k0 = lo0 + kt * TK;
        const uint32_t nkeys = span - kt * TK < TK ? span - kt * TK : TK;
        if (NBUF == 2u) {
            const bool more = kt + 1u < ntiles;
            if (more) stage_kv(kt + 1u, bf ^ 1u);
            if (more) pd_attn_cpa_wait1(); else pd_attn_cpa_wait0();
        } else {
            stage_kv(kt, 0u);
            pd_attn_cpa_wait0();
        }
        __syncthreads();
        if (F8) {
            // expand before the bpeer release: the collective mma reads the
            // peer's K half over DSMEM, so rank1's halves must be real f16
            // before the leader is unblocked. K uses this CTA's LOCAL row
            // count within the pair-uniform nkeys.
            const uint32_t nk_loc = nkeys > crank * TKH
                ? (nkeys - crank * TKH < TKH ? nkeys - crank * TKH : TKH) : 0u;
            pd_pf5_f8_expand_k<TKH, HD>(ks + (size_t)bf * TKH * HD, nk_loc, tid);
            pd_pf5_f8_expand_v<TK, HD>(vs + (size_t)bf * TK * row_e, nkeys, tid);
            __syncthreads();  // expanded halves CTA-visible before the release
        }
        // rank1's K half staged & CTA-visible: release it to the leader.
        // Progression past tile kt needs the bdone(kt) wait below, and the
        // leader only issues kt after consuming bpeer[bf]@kt - so a slot is
        // never re-armed before its previous phase was consumed.
        if (crank == 1u && tid == 0)
            asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];"
                         ::"r"(peer_addr(&bpeer[bf])) : "memory");
        __half* vcur = vs + (size_t)bf * TK * row_e;
        if (crank == 0u && tid == 0) {
            bar_wait(&bpeer[bf], (peer_ph >> bf) & 1u);
            peer_ph ^= 1u << bf;
            const uint32_t a16 = (uint32_t)__cvta_generic_to_shared(qs) >> 4;
            const uint32_t b16 = (uint32_t)__cvta_generic_to_shared(
                ks + (size_t)bf * TKH * HD) >> 4;
            #pragma unroll
            for (uint32_t kb2 = 0; kb2 < HD / 16u; ++kb2) {
                const uint32_t t = kb2 >> 2, c = (kb2 & 3u) * 2u;
                const uint64_t ad = pd_tc5_sdesc(a16 + t * (MR * 8u) + c);
                const uint64_t bd = pd_tc5_sdesc(b16 + t * (TKH * 8u) + c);
                const uint32_t id = (1u << 4) | ((TK >> 3) << 17) | ((128u >> 4) << 24);
                asm volatile(
                    "{\n\t.reg .pred p;\n\t"
                    "setp.ne.b32 p, %4, 0;\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n\t}"
                    ::"r"(tmem), "l"(ad), "l"(bd), "r"(id), "r"(kb2 > 0 ? 1u : 0u));
            }
            asm volatile(
                "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                ::"r"((uint32_t)__cvta_generic_to_shared(bdone)));
            asm volatile(
                "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                ::"r"(peer_addr(bdone)));
        }
        bar_wait(bdone, done_ph);
        done_ph ^= 1u;
        __syncthreads();
        // quadrant softmax: warp w reads lanes w*32.. = rows (w&1)*32..
        // x key cols qh*TKH.. ; cross-half exchange through smem
        constexpr uint32_t NCC = (TKH + 31u) / 32u;
        float sv[NCC][32];
        float pm = -INFINITY;
        if (warp < 4u) {
            #pragma unroll
            for (uint32_t cc = 0; cc < NCC; ++cc) {
                uint32_t rr[32];
                const uint32_t taddr = tmem + ((warp * 32u) << 16) + cc * 32u;
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
                    "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
                    "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
                    : "=r"(rr[0]),"=r"(rr[1]),"=r"(rr[2]),"=r"(rr[3]),"=r"(rr[4]),"=r"(rr[5]),"=r"(rr[6]),"=r"(rr[7]),
                      "=r"(rr[8]),"=r"(rr[9]),"=r"(rr[10]),"=r"(rr[11]),"=r"(rr[12]),"=r"(rr[13]),"=r"(rr[14]),"=r"(rr[15]),
                      "=r"(rr[16]),"=r"(rr[17]),"=r"(rr[18]),"=r"(rr[19]),"=r"(rr[20]),"=r"(rr[21]),"=r"(rr[22]),"=r"(rr[23]),
                      "=r"(rr[24]),"=r"(rr[25]),"=r"(rr[26]),"=r"(rr[27]),"=r"(rr[28]),"=r"(rr[29]),"=r"(rr[30]),"=r"(rr[31])
                    : "r"(taddr));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (uint32_t j = 0; j < 32u; ++j) sv[cc][j] = __uint_as_float(rr[j]);
            }
            #pragma unroll
            for (uint32_t cc = 0; cc < NCC; ++cc)
                #pragma unroll
                for (uint32_t j = 0; j < 32u; ++j) {
                    const uint32_t kc = qh * TKH + cc * 32u + j;   // collective col
                    const uint32_t kp = k0 + kc;
                    const bool ok = my_tok < ntok && kp >= my_lo && kp <= my_pos
                        && kc < nkeys && (cc * 32u + j) < TKH;
                    sv[cc][j] = ok ? sv[cc][j] * scale : -INFINITY;
                    pm = fmaxf(pm, sv[cc][j]);
                }
            if (qh == 1u) s_x1[my_row] = pm;
        }
        __syncthreads();
        float m_new = -INFINITY, corr = 1.0f;
        if (warp < 2u) {
            m_new = fmaxf(fmaxf(m_run, pm), s_x1[my_row]);
            corr = m_new > -INFINITY ? __expf(m_run - m_new) : 1.0f;
            s_mnew[my_row] = m_new;
        }
        __syncthreads();
        float lsum = 0.0f;
        if (warp < 4u) {
            if (qh == 1u) m_new = s_mnew[my_row];
            #pragma unroll
            for (uint32_t cc = 0; cc < NCC; ++cc)
                #pragma unroll
                for (uint32_t j = 0; j < 32u; j += 2u) {
                    if (cc * 32u + j >= TKH) break;    // TKH<32 pad cols
                    const float w0 = sv[cc][j] > -INFINITY ? __expf(sv[cc][j] - m_new) : 0.0f;
                    const float w1 = sv[cc][j + 1u] > -INFINITY ? __expf(sv[cc][j + 1u] - m_new) : 0.0f;
                    lsum += w0 + w1;
                    *(__half2*)(ps + (size_t)my_row * p_s + qh * TKH + cc * 32u + j) =
                        __floats2half2_rn(w0, w1);
                }
            if (qh == 1u) s_x1[my_row] = lsum;
        }
        __syncthreads();
        if (warp < 2u) {
            l_run = l_run * corr + lsum + s_x1[my_row];
            m_run = m_new;
            s_corr[my_row] = corr;
        }
        __syncthreads();
        {
            #pragma unroll
            for (uint32_t half = 0; half < 2u; ++half) {
                const uint32_t rr = mt0 + (lane >> 2) + half * 8u;
                const float corr2 = s_corr[rr];
                #pragma unroll
                for (uint32_t s2 = 0; s2 < NSUB; ++s2) {
                    o_acc[s2][half * 2u] *= corr2;
                    o_acc[s2][half * 2u + 1u] *= corr2;
                }
            }
            for (uint32_t kk = 0; kk < TK; kk += 16u) {
                uint32_t af[4];
                const __half* ap = ps + (size_t)(mt0 + (lane & 15u)) * p_s
                                 + kk + ((lane >> 4) ? 8u : 0u);
                pd_ldm_x4(af, (const unsigned char*)ap);
                #pragma unroll
                for (uint32_t s2 = 0; s2 < NSUB; ++s2) {
                    uint32_t bfr[2];
                    const __half* bp = vcur + (size_t)(kk + (lane & 15u)) * row_e
                                     + d0 + s2 * 8u;
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                                 : "=r"(bfr[0]), "=r"(bfr[1])
                                 : "r"((unsigned)__cvta_generic_to_shared(bp)));
                    pd_fa_mma16(o_acc[s2], af[0], af[1], af[2], af[3], bfr[0], bfr[1]);
                }
            }
        }
        __syncthreads();
    }
    if (warp < 2u && my_tok < ntok) s_l[my_row] = l_run;
    __syncthreads();
    {
        #pragma unroll
        for (uint32_t half = 0; half < 2u; ++half) {
            const uint32_t rr = mt0 + (lane >> 2) + half * 8u;
            const uint32_t tok = rr / G, g = rr % G;
            if (tok >= ntok) continue;
            const float inv_l = s_l[rr] > 0.0f ? 1.0f / s_l[rr] : 0.0f;
            float* dst = out + ((size_t)(tq0 + tok) * n_heads
                        + (size_t)kvh * G + g) * HD + d0;
            #pragma unroll
            for (uint32_t s2 = 0; s2 < NSUB; ++s2) {
                dst[s2 * 8u + 2u * (lane & 3u)] = o_acc[s2][half * 2u] * inv_l;
                dst[s2 * 8u + 2u * (lane & 3u) + 1u] = o_acc[s2][half * 2u + 1u] * inv_l;
            }
        }
    }
    __syncthreads();
    asm volatile("barrier.cluster.arrive;");
    asm volatile("barrier.cluster.wait;");
    if (tid < 32)
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
                     ::"r"(tmem), "r"(TMEM_COLS));
#else
    (void)q; (void)pool_k; (void)pool_v; (void)sinks; (void)out; (void)positions;
    (void)slots; (void)block_tables; (void)blocks_per_slot; (void)n_heads;
    (void)kv_dim; (void)swa_window; (void)rows; (void)scale;
#endif
}
#endif  // __cluster_dims__ guard (host pass || sm_90+)

// pf6s: the SWA (hd256) twin of pf6g - pf5's tile with P.V moved off
// register-HMMA onto tcgen05 kind::f8f6f4 (KV8 serving arm only; the f16-KV
// route keeps pf5). ::1 makes everything STRICTLY simpler than the c2 port:
// D folds straight (lane = row, no rank halves), B is full-per-CTA so each
// N=128 issue reads one whole [TK x 128B] e4m3 column tile (the probed
// geometry), softmax state stays thread-local. V is cp.async'd directly
// into the swizzled e4m3 B image (pf5's expand_v dies); P-tilde quantizes
// e4m3 at store, l stays the exact f32 sum. O lives in tmem cols TK..TK+256
// with the vote-gated corr rescale. Double-buffered staging is kept - the
// PV(kt-1) completion wait sits at the loop top, where it overlaps the
// whole staging+S+softmax front of tile kt.
template <uint32_t HD, uint32_t G, uint32_t TK>
__global__ void __launch_bounds__(256, 1) pd_attn_prefill_pf6s_kernel(
    const float* __restrict__ q, const __half* __restrict__ pool_k,
    const __half* __restrict__ pool_v, const float* __restrict__ sinks,
    float* __restrict__ out, const unsigned int* __restrict__ positions,
    const unsigned int* __restrict__ slots,
    const uint32_t* __restrict__ block_tables, uint32_t blocks_per_slot,
    uint32_t n_heads, uint32_t kv_dim, uint32_t swa_window, uint32_t rows,
    float scale, const uint32_t* __restrict__ run_offs = nullptr) {
#if PD_TC5_OK
    // batched-runs arm: identical convention to pf5/pf6g -
    // grid.z indexes the armed run table; nullptr = classic launch.
    if (run_offs != nullptr) {
        const uint32_t roff = run_offs[blockIdx.z];
        rows = run_offs[blockIdx.z + 1u] - roff;
        q += (size_t)roff * n_heads * HD;
        out += (size_t)roff * n_heads * HD;
        positions += roff;
        if (slots) slots += roff;
    }
    constexpr uint32_t MR = 128u;
    constexpr uint32_t TQ = MR / G;
    constexpr uint32_t NCT8 = HD / 128u;           // N=128 PV issues per K32
    constexpr uint32_t OB = TK;                    // O cols after the S region
    static_assert(TK % 32u == 0u && OB + HD <= 512u, "pf6s tmem: S + O over budget");
    extern __shared__ __align__(1024) unsigned char sh6s_raw[];
    unsigned char* sh_ = sh6s_raw
        + ((1024u - ((uint32_t)__cvta_generic_to_shared(sh6s_raw) & 1023u)) & 1023u);
    __half* qs = (__half*)sh_;                     // SW128 [MR x HD]
    __half* ks = qs + (size_t)MR * HD;             // 2x SW128 [TK x HD] (+e4m3 strip)
    unsigned char* vs = (unsigned char*)(ks + 2u * (size_t)TK * HD);  // 2x e4m3 SW128 [TK x HD]
    unsigned char* ps = vs + 2u * (size_t)TK * HD; // e4m3 [MR x 128B]
    float* s_corr = (float*)(ps + (size_t)MR * 128u);
    float* s_l = s_corr + MR;
    uint32_t* s_flag = (uint32_t*)(s_l + MR);
    uint64_t* bdone = (uint64_t*)(s_flag + 2u);
    uint64_t* bpvd = bdone + 1u;
    __shared__ uint32_t tmem_slot[1];

    const uint32_t kvh = blockIdx.x, tq0 = blockIdx.y * TQ;
    const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
    if (tq0 >= rows) return;
    const uint32_t ntok = rows - tq0 < TQ ? rows - tq0 : TQ;

    if (tid == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                     ::"r"((uint32_t)__cvta_generic_to_shared(bdone)));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                     ::"r"((uint32_t)__cvta_generic_to_shared(bpvd)));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();
    if (tid < 32)
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;"
                     ::"r"((uint32_t)__cvta_generic_to_shared(tmem_slot)));
    __syncthreads();
    const uint32_t tmem = tmem_slot[0];
    auto bar_wait = [&](uint64_t* bar, uint32_t parity) {
        const uint32_t a2 = (uint32_t)__cvta_generic_to_shared(bar);
        asm volatile("{\n\t.reg .pred p;\nW%=:\n\t"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                     "@!p bra W%=;\n\t}" ::"r"(a2), "r"(parity));
    };

    // Q image (pf5's walk: MR=128 tiles are 16KB)
    for (uint32_t i = tid; i < MR * (HD / 8u); i += 256u) {
        const uint32_t r = i / (HD / 8u), c16 = i % (HD / 8u);
        const uint32_t tok = r / G, g = r % G;
        __half tmp[8];
        if (tok < ntok) {
            const float* src = q + ((size_t)(tq0 + tok) * n_heads
                             + (size_t)kvh * G + g) * HD + c16 * 8u;
            #pragma unroll
            for (uint32_t e = 0; e < 8u; ++e) tmp[e] = __float2half(src[e]);
        } else {
            #pragma unroll
            for (uint32_t e = 0; e < 8u; ++e) tmp[e] = __half(0.f);
        }
        const uint32_t t = c16 >> 3, c = c16 & 7u;
        const uint32_t off16 = (r >> 3) * 64u + (r & 7u) * 8u + (c ^ (r & 7u));
        *(uint4*)((unsigned char*)qs + ((size_t)t << 14) + ((size_t)off16 << 4)) =
            *(const uint4*)tmp;
    }
    if (tid < MR) s_l[tid] = 0.0f;
    float m_run = -INFINITY, l_run = 0.0f;
    const uint32_t my_row = warp * 32u + lane;         // rows for warps 0-3
    const uint32_t my_tok = my_row / G;
    const uint32_t my_pos = my_tok < ntok ? positions[tq0 + my_tok] : 0u;
    const uint32_t my_lo = (swa_window > 0 && my_pos + 1 > swa_window)
        ? my_pos + 1 - swa_window : 0u;

    const uint32_t pos_last = positions[tq0 + ntok - 1u];
    const uint32_t lo0 = (swa_window > 0 && positions[tq0] + 1 > swa_window)
        ? positions[tq0] + 1 - swa_window : 0u;
    const uint32_t span = pos_last + 1u - lo0;
    const uint32_t ntiles = (span + TK - 1u) / TK;
    const uint32_t slot = slots ? slots[0] : 0u;
    const uint32_t* bt = block_tables + (size_t)slot * blocks_per_slot;

    // stage slab bf: K e4m3 strip (upper half, pf5's expand_k eats it),
    // V straight into the swizzled e4m3 image (tails zero-fill: stale
    // e4m3 can decode NaN and 0 x NaN poisons the accumulate)
    auto stage_kv = [&](uint32_t kt, uint32_t bf) {
        const uint32_t k0s = lo0 + kt * TK;
        const uint32_t nkeys_s = span - kt * TK < TK ? span - kt * TK : TK;
        const unsigned char* pk8 = (const unsigned char*)pool_k;
        const unsigned char* pv8 = (const unsigned char*)pool_v;
        unsigned char* kstrip = (unsigned char*)(ks + (size_t)bf * TK * HD)
            + (size_t)TK * HD;
        unsigned char* vb = vs + (size_t)bf * TK * HD;
        for (uint32_t i = tid; i < TK * (HD / 16u); i += 256u) {
            const uint32_t kr = i / (HD / 16u), l = i - kr * (HD / 16u);
            if (kr < nkeys_s) {
                const uint32_t gpos = k0s + kr;
                const uint32_t blk = bt[gpos >> 4];
                const size_t b8 = (size_t)blk * 16u * kv_dim
                    + (size_t)(gpos & 15u) * kv_dim + (size_t)kvh * HD + l * 16u;
                pd_attn_cpa16((char*)(kstrip + (size_t)kr * HD + l * 16u),
                              (const char*)(pk8 + b8));
            }
        }
        for (uint32_t i = tid; i < TK * (HD / 16u); i += 256u) {
            const uint32_t kr = i / (HD / 16u), l = i - kr * (HD / 16u);
            const uint32_t t = l >> 3, u = l & 7u;
            const uint32_t off16 = (kr >> 3) * 64u + (kr & 7u) * 8u + (u ^ (kr & 7u));
            unsigned char* vdst = vb + (size_t)t * TK * 128u + ((size_t)off16 << 4);
            if (kr < nkeys_s) {
                const uint32_t gpos = k0s + kr;
                const uint32_t blk = bt[gpos >> 4];
                const size_t b8 = (size_t)blk * 16u * kv_dim
                    + (size_t)(gpos & 15u) * kv_dim + (size_t)kvh * HD + l * 16u;
                pd_attn_cpa16((char*)vdst, (const char*)(pv8 + b8));
            } else {
                *(uint4*)vdst = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        pd_attn_cpa_commit();
    };

    uint32_t done_ph = 0, pvd_ph = 0;
    if (ntiles) stage_kv(0u, 0u);
    for (uint32_t kt = 0; kt < ntiles; ++kt) {
        const uint32_t bf = kt & 1u;
        const uint32_t k0 = lo0 + kt * TK;
        const uint32_t nkeys = span - kt * TK < TK ? span - kt * TK : TK;
        // PV(kt-1) must retire before slab bf is restaged (kt+1 shares its
        // parity) and before this tile's P stores / O rescale - one wait
        // here covers all three and overlaps the previous tile's tail
        if (kt > 0) bar_wait(bpvd, pvd_ph ^ 1u);
        const bool more = kt + 1u < ntiles;
        if (more) stage_kv(kt + 1u, bf ^ 1u);
        if (more) pd_attn_cpa_wait1(); else pd_attn_cpa_wait0();
        __syncthreads();
        __half* kcur = ks + (size_t)bf * TK * HD;
        unsigned char* vcur = vs + (size_t)bf * TK * HD;
        pd_pf5_f8_expand_k<TK, HD>(kcur, nkeys, tid);
        __syncthreads();
        if (tid == 0) *s_flag = 0u;
        if (tid == 0) {
            const uint32_t a16 = (uint32_t)__cvta_generic_to_shared(qs) >> 4;
            const uint32_t b16 = (uint32_t)__cvta_generic_to_shared(kcur) >> 4;
            #pragma unroll
            for (uint32_t kb = 0; kb < HD / 16u; ++kb) {
                const uint32_t t = kb >> 2, c = (kb & 3u) * 2u;
                const uint64_t ad = pd_tc5_sdesc(a16 + t * 1024u + c);
                const uint64_t bd = pd_tc5_sdesc(b16 + t * (TK * 8u) + c);
                const uint32_t id = (1u << 4) | ((TK >> 3) << 17) | ((MR >> 4) << 24);
                asm volatile(
                    "{\n\t.reg .pred p;\n\t"
                    "setp.ne.b32 p, %4, 0;\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}"
                    ::"r"(tmem), "l"(ad), "l"(bd), "r"(id), "r"(kb > 0 ? 1u : 0u));
            }
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                ::"r"((uint32_t)__cvta_generic_to_shared(bdone)));
        }
        bar_wait(bdone, done_ph);
        done_ph ^= 1u;
        __syncthreads();
        if (warp < 4u) {
            float sv[TK / 32u][32];
            #pragma unroll
            for (uint32_t cc = 0; cc < TK / 32u; ++cc) {
                uint32_t rr[32];
                const uint32_t taddr = tmem + ((warp * 32u) << 16) + cc * 32u;
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
                    "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
                    "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
                    : "=r"(rr[0]),"=r"(rr[1]),"=r"(rr[2]),"=r"(rr[3]),"=r"(rr[4]),"=r"(rr[5]),"=r"(rr[6]),"=r"(rr[7]),
                      "=r"(rr[8]),"=r"(rr[9]),"=r"(rr[10]),"=r"(rr[11]),"=r"(rr[12]),"=r"(rr[13]),"=r"(rr[14]),"=r"(rr[15]),
                      "=r"(rr[16]),"=r"(rr[17]),"=r"(rr[18]),"=r"(rr[19]),"=r"(rr[20]),"=r"(rr[21]),"=r"(rr[22]),"=r"(rr[23]),
                      "=r"(rr[24]),"=r"(rr[25]),"=r"(rr[26]),"=r"(rr[27]),"=r"(rr[28]),"=r"(rr[29]),"=r"(rr[30]),"=r"(rr[31])
                    : "r"(taddr));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (uint32_t j = 0; j < 32u; ++j) sv[cc][j] = __uint_as_float(rr[j]);
            }
            float m_tile = -INFINITY;
            #pragma unroll
            for (uint32_t cc = 0; cc < TK / 32u; ++cc)
                #pragma unroll
                for (uint32_t j = 0; j < 32u; ++j) {
                    const uint32_t kp = k0 + cc * 32u + j;
                    const bool ok = my_tok < ntok && kp >= my_lo && kp <= my_pos
                        && (cc * 32u + j) < nkeys;
                    sv[cc][j] = ok ? sv[cc][j] * scale : -INFINITY;
                    m_tile = fmaxf(m_tile, sv[cc][j]);
                }
            const float m_new = fmaxf(m_run, m_tile);
            const float corr = m_new > -INFINITY ? __expf(m_run - m_new) : 1.0f;
            float lsum = 0.0f;
            #pragma unroll
            for (uint32_t cc = 0; cc < TK / 32u; ++cc)
                #pragma unroll
                for (uint32_t j = 0; j < 32u; j += 2u) {
                    const float w0 = sv[cc][j] > -INFINITY ? __expf(sv[cc][j] - m_new) : 0.0f;
                    const float w1 = sv[cc][j + 1u] > -INFINITY ? __expf(sv[cc][j + 1u] - m_new) : 0.0f;
                    lsum += w0 + w1;
                    // e4m3 pair into the [MR x 128B] A image
                    const uint32_t col = cc * 32u + j;
                    const uint32_t c16 = col >> 4;
                    const uint32_t off16 = (my_row >> 3) * 64u + (my_row & 7u) * 8u
                        + (c16 ^ (my_row & 7u));
                    unsigned char* pdst = ps + ((size_t)off16 << 4) + (col & 15u);
                    pdst[0] = __nv_fp8_e4m3(w0).__x;
                    pdst[1] = __nv_fp8_e4m3(w1).__x;
                }
            l_run = l_run * corr + lsum;
            m_run = m_new;
            s_corr[my_row] = corr;
            if (kt > 0 && corr != 1.0f && my_tok < ntok) *s_flag = 1u;
        }
        __syncthreads();
        // O rescale when any row's max moved (vote; rare after warmup)
        if (*s_flag) {
            if (warp < 4u) {
                const float rc = s_corr[warp * 32u + lane];
                // the whole O region: HD cols = HD/32 x32 chunks
                #pragma unroll
                for (uint32_t ct = 0; ct < HD / 32u; ++ct) {
                    uint32_t rr[32];
                    const uint32_t taddr = tmem + ((warp * 32u) << 16) + OB + ct * 32u;
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
                        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
                        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
                        : "=r"(rr[0]),"=r"(rr[1]),"=r"(rr[2]),"=r"(rr[3]),"=r"(rr[4]),"=r"(rr[5]),"=r"(rr[6]),"=r"(rr[7]),
                          "=r"(rr[8]),"=r"(rr[9]),"=r"(rr[10]),"=r"(rr[11]),"=r"(rr[12]),"=r"(rr[13]),"=r"(rr[14]),"=r"(rr[15]),
                          "=r"(rr[16]),"=r"(rr[17]),"=r"(rr[18]),"=r"(rr[19]),"=r"(rr[20]),"=r"(rr[21]),"=r"(rr[22]),"=r"(rr[23]),
                          "=r"(rr[24]),"=r"(rr[25]),"=r"(rr[26]),"=r"(rr[27]),"=r"(rr[28]),"=r"(rr[29]),"=r"(rr[30]),"=r"(rr[31])
                        : "r"(taddr));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (uint32_t j = 0; j < 32u; ++j)
                        rr[j] = __float_as_uint(__uint_as_float(rr[j]) * rc);
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x32.b32 [%32], "
                        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
                        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31};"
                        :: "r"(rr[0]),"r"(rr[1]),"r"(rr[2]),"r"(rr[3]),"r"(rr[4]),"r"(rr[5]),"r"(rr[6]),"r"(rr[7]),
                           "r"(rr[8]),"r"(rr[9]),"r"(rr[10]),"r"(rr[11]),"r"(rr[12]),"r"(rr[13]),"r"(rr[14]),"r"(rr[15]),
                           "r"(rr[16]),"r"(rr[17]),"r"(rr[18]),"r"(rr[19]),"r"(rr[20]),"r"(rr[21]),"r"(rr[22]),"r"(rr[23]),
                           "r"(rr[24]),"r"(rr[25]),"r"(rr[26]),"r"(rr[27]),"r"(rr[28]),"r"(rr[29]),"r"(rr[30]),"r"(rr[31]),
                           "r"(taddr));
                }
                asm volatile("tcgen05.wait::st.sync.aligned;");
                asm volatile("tcgen05.fence::before_thread_sync;");
            }
        }
        __syncthreads();
        if (tid == 0) {
            asm volatile("tcgen05.fence::after_thread_sync;");
            const uint32_t p16 = (uint32_t)__cvta_generic_to_shared(ps) >> 4;
            const uint32_t v16 = (uint32_t)__cvta_generic_to_shared(vcur) >> 4;
            #pragma unroll
            for (uint32_t kb = 0; kb < TK / 32u; ++kb) {
                const uint64_t ad = pd_tc5_sdesc(p16 + kb * 2u);
                #pragma unroll
                for (uint32_t ct = 0; ct < NCT8; ++ct) {
                    const uint64_t bd = pd_tc5_sdesc(v16 + ct * (TK * 8u) + kb * 256u);
                    const uint32_t id = (1u << 4) | (1u << 16)
                        | ((128u >> 3) << 17) | ((MR >> 4) << 24);
                    asm volatile(
                        "{\n\t.reg .pred p;\n\t"
                        "setp.ne.b32 p, %4, 0;\n\t"
                        "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t}"
                        ::"r"(tmem + OB + ct * 128u), "l"(ad), "l"(bd), "r"(id),
                          "r"(kt > 0 || kb > 0 ? 1u : 0u));
                }
            }
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                ::"r"((uint32_t)__cvta_generic_to_shared(bpvd)));
        }
        pvd_ph ^= 1u;
    }
    if (warp < 4u && my_tok < ntok) s_l[my_row] = l_run;
    if (ntiles) bar_wait(bpvd, pvd_ph ^ 1u);
    __syncthreads();
    // epilogue: straight ::1 fold - lane = row, dims = ct*128 + ch*32 + j
    if (warp < 4u) {
        const uint32_t row = warp * 32u + lane;
        const uint32_t tok = row / G, g = row % G;
        const float inv_l = (tok < ntok && s_l[row] > 0.0f) ? 1.0f / s_l[row] : 0.0f;
        float* dst = tok < ntok
            ? out + ((size_t)(tq0 + tok) * n_heads + (size_t)kvh * G + g) * HD
            : nullptr;
        #pragma unroll
        for (uint32_t ct = 0; ct < NCT8; ++ct) {
            #pragma unroll
            for (uint32_t ch = 0; ch < 4u; ++ch) {
                uint32_t rr[32];
                const uint32_t taddr = tmem + ((warp * 32u) << 16) + OB
                    + ct * 128u + ch * 32u;
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
                    "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
                    "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
                    : "=r"(rr[0]),"=r"(rr[1]),"=r"(rr[2]),"=r"(rr[3]),"=r"(rr[4]),"=r"(rr[5]),"=r"(rr[6]),"=r"(rr[7]),
                      "=r"(rr[8]),"=r"(rr[9]),"=r"(rr[10]),"=r"(rr[11]),"=r"(rr[12]),"=r"(rr[13]),"=r"(rr[14]),"=r"(rr[15]),
                      "=r"(rr[16]),"=r"(rr[17]),"=r"(rr[18]),"=r"(rr[19]),"=r"(rr[20]),"=r"(rr[21]),"=r"(rr[22]),"=r"(rr[23]),
                      "=r"(rr[24]),"=r"(rr[25]),"=r"(rr[26]),"=r"(rr[27]),"=r"(rr[28]),"=r"(rr[29]),"=r"(rr[30]),"=r"(rr[31])
                    : "r"(taddr));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                if (dst) {
                    const uint32_t d0c = ct * 128u + ch * 32u;
                    #pragma unroll
                    for (uint32_t j = 0; j < 32u; ++j)
                        dst[d0c + j] = __uint_as_float(rr[j]) * inv_l;
                }
            }
        }
    }
    __syncthreads();
    if (tid < 32)
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;"
                     ::"r"(tmem));
#else
    (void)q; (void)pool_k; (void)pool_v; (void)sinks; (void)out; (void)positions;
    (void)slots; (void)block_tables; (void)blocks_per_slot; (void)n_heads;
    (void)kv_dim; (void)swa_window; (void)rows; (void)scale;
#endif
}

// pf6g: the hd512 GLOBAL arm rebuilt with both GEMMs on tcgen05.
// pf5g_c2's P.V ran M=64 x N=512 x K=TK
// per CTA-tile on legacy mma.sync HMMA - sm_100's dead-class pipe - and at
// hd512 that was the single most expensive attention block (601 us/chunk-
// layer, 5 global layers costing more than 25 SWA). Structure vs c2:
//  - P is staged as an SW128 [MR x 128-half] image (the qs pack walk) and
//    becomes the collective A operand of a second tcgen05 mma chain.
//  - V's SW128 [TK x HD] image is consumed directly as B^T via instruction-
//    descriptor bit 16 ("transpose B") - probed exact for f16 AND e4m3,
//    so the F8 V expand just retargets the K-expand
//    helper (same strip trick, SW128 out) and the row_e pad dies.
//  - O accumulates in tmem: per PV issue N=64 (one 64-half column tile -
//    the probe-proven span), D fold = 128 lanes x 32 cols per CTA, HD/64=8
//    issues per K16 chunk into cols OB+ct*32. S keeps cols 0..TKH.
//    tmem alloc 512 (pow2 >= 64+256); smem ~202KB pins 1 CTA/SM anyway so
//    the co-residency lesson does not bite here.
//  - Online-softmax rescale: when any row's max moves (block vote through
//    s_flag - rare after the first tiles), warps 0-3 tcgen05.ld the O
//    quadrants, multiply by corr[row], tcgen05.st back (probe stage 3:
//    st -> wait::st -> barrier -> mma-accumulate is sound).
//  - Barriers: bdone (S commit) and bpeer (K half) as c2, plus bpv (rank1
//    P+rescale ready -> leader) and bpvd (PV commit -> both, gates vs/ps
//    reuse next tile and the epilogue).
// Numeric class: f32 tmem accumulation in the same tile order as c2's
// register HMMA - reduction-tree reorder only (coherence-gate class).
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 900)
template <uint32_t HD, uint32_t G, uint32_t TK, bool F8 = false>
__global__ void __launch_bounds__(256, 1) __cluster_dims__(2, 1, 1)
pd_attn_prefill_pf6g_kernel(
    const float* __restrict__ q, const __half* __restrict__ pool_k,
    const __half* __restrict__ pool_v, const float* __restrict__ sinks,
    float* __restrict__ out, const unsigned int* __restrict__ positions,
    const unsigned int* __restrict__ slots,
    const uint32_t* __restrict__ block_tables, uint32_t blocks_per_slot,
    uint32_t n_heads, uint32_t kv_dim, uint32_t swa_window, uint32_t rows,
    float scale, const uint32_t* __restrict__ run_offs = nullptr) {
#if PD_TC5_OK
    // batched-runs arm: same convention as pf5 - grid.z indexes
    // the armed run table, pointers re-aim, nullptr = classic launch.
    if (run_offs != nullptr) {
        const uint32_t roff = run_offs[blockIdx.z];
        rows = run_offs[blockIdx.z + 1u] - roff;
        q += (size_t)roff * n_heads * HD;
        out += (size_t)roff * n_heads * HD;
        positions += roff;
        if (slots) slots += roff;
    }
    constexpr uint32_t MR = 64u;                   // real rows per CTA
    constexpr uint32_t TKH = TK / 2u;              // this CTA's K N-half
    constexpr uint32_t TQ = MR / G;
    // PV runs HD/128 collective N=128 issues per K16 chunk; in cta_group::2
    // each rank supplies a COMPACT [TK x 64-half] column tile at the shared
    // descriptor address (the same rank-local-compact rule the S GEMM's K
    // halves follow - and each rank's per-issue operand is then exactly the
    // [K x 64] SW128 geometry the PV probe validated). So this CTA
    // stages only its 64-col slice of each 128-col band: HDH = HD/2 halves.
    constexpr uint32_t NCT = HD / 128u;            // f16 PV: collective N=128 issues
    // F8 PV (phase 2): e4m3 operands via kind::f8f6f4 - V is cp.async'd
    // STRAIGHT into the swizzled B image (no strip, no expand: that pass
    // only existed to widen bytes to halves), P-tilde quantizes to e4m3 at
    // store (l stays the exact f32 sum - the usual fp8-attention
    // normalizer convention). Collective N=256 per issue so each rank's compact slice
    // is a [TK x 128-BYTE] SW128 tile; K per mma = 32 forces TK % 32 == 0.
    constexpr uint32_t NCT8 = HD / 256u;           // f8 PV: collective N=256 issues
    constexpr uint32_t HDH = HD / 2u;              // this CTA's V col slice total
    // O tmem col base: S owns cols 0..TKH, so clear it (TK=160's TKH=80
    // overlapping a fixed 64 was a real parity kill); O spans OB..OB+256
    constexpr uint32_t OB = (TKH + 31u) & ~31u;
    static_assert(OB + 256u <= 512u, "S + O must fit the tmem alloc");
    static_assert(TK >= 64u && TK % 16u == 0u, "pf6g tile contract");
    static_assert(!F8 || TK % 32u == 0u, "f8 PV needs K32 chunks");
    constexpr uint32_t VS_BYTES = F8 ? TK * HDH : TK * HDH * 2u;
    // P image column tiles: e4m3 rows are TK bytes (2 tiles past 128),
    // f16 rows TK halves (one 128-half tile up to TK=128)
    constexpr uint32_t PS_BYTES = F8 ? MR * (TK > 128u ? 256u : 128u)
                                     : MR * (TK > 128u ? 384u : 256u);
    extern __shared__ __align__(1024) unsigned char shg6_raw[];
    unsigned char* shg = shg6_raw
        + ((1024u - ((uint32_t)__cvta_generic_to_shared(shg6_raw) & 1023u)) & 1023u);
    __half* qs = (__half*)shg;                     // SW128 packed [MR x HD]
    __half* ks = qs + (size_t)MR * HD;             // SW128 [TKH x HD]
    unsigned char* vs = (unsigned char*)(ks + (size_t)TKH * HD);  // rank V slice
    unsigned char* ps = vs + VS_BYTES;             // P image (f16 or e4m3)
    float* s_corr = (float*)(ps + PS_BYTES);
    float* s_x1 = s_corr + MR;                     // half-1 partial max, then lsum
    float* s_mnew = s_x1 + MR;
    float* s_l = s_mnew + MR;
    uint32_t* s_flag = (uint32_t*)(s_l + MR);      // [1] any-corr vote
    uint64_t* bdone = (uint64_t*)(s_flag + 2u);    // [1] S mma done
    uint64_t* bpeer = bdone + 1u;                  // [1] rank1 K-half ready
    uint64_t* bpv = bpeer + 1u;                    // [1] rank1 P ready (rank0's live)
    uint64_t* bpvd = bpv + 1u;                     // [1] PV committed
    __shared__ uint32_t tmem_slot[1];

    const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
    uint32_t crank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(crank));
    const uint32_t kvh = blockIdx.y;
    const uint32_t tq_pair = (blockIdx.x >> 1) * 2u * TQ;
    const uint32_t tq0 = tq_pair + crank * TQ;
    // No early return: pair-uniform collective loop (see c2)
    const uint32_t ntok = tq0 < rows ? (rows - tq0 < TQ ? rows - tq0 : TQ) : 0u;

    if (tid == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                     ::"r"((uint32_t)__cvta_generic_to_shared(bdone)));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                     ::"r"((uint32_t)__cvta_generic_to_shared(bpeer)));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                     ::"r"((uint32_t)__cvta_generic_to_shared(bpv)));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                     ::"r"((uint32_t)__cvta_generic_to_shared(bpvd)));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();
    asm volatile("barrier.cluster.arrive;");
    asm volatile("barrier.cluster.wait;");
    if (tid < 32)
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], 512;"
                     ::"r"((uint32_t)__cvta_generic_to_shared(tmem_slot)));
    __syncthreads();
    const uint32_t tmem = tmem_slot[0];
    auto peer_addr = [&](void* p) -> uint32_t {
        const uint32_t pa0 = (uint32_t)__cvta_generic_to_shared(p);
        uint32_t pa;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                     : "=r"(pa) : "r"(pa0), "r"(crank ^ 1u));
        return pa;
    };
    auto bar_wait = [&](uint64_t* bar, uint32_t parity) {
        const uint32_t a2 = (uint32_t)__cvta_generic_to_shared(bar);
        asm volatile("{\n\t.reg .pred p;\nW%=:\n\t"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                     "@!p bra W%=;\n\t}" ::"r"(a2), "r"(parity));
    };

    // pack this CTA's 64 Q rows (compact SW128, the c2 walk)
    for (uint32_t i = tid; i < MR * (HD / 8u); i += 256u) {
        const uint32_t r = i / (HD / 8u), c16 = i % (HD / 8u);
        const uint32_t tok = r / G, g = r % G;
        __half tmp[8];
        if (tok < ntok) {
            const float* srcp = q + ((size_t)(tq0 + tok) * n_heads
                              + (size_t)kvh * G + g) * HD + c16 * 8u;
            #pragma unroll
            for (uint32_t e = 0; e < 8u; ++e) tmp[e] = __float2half(srcp[e]);
        } else {
            #pragma unroll
            for (uint32_t e = 0; e < 8u; ++e) tmp[e] = __half(0.f);
        }
        const uint32_t t = c16 >> 3, c = c16 & 7u;
        const uint32_t off16 = (r >> 3) * 64u + (r & 7u) * 8u + (c ^ (r & 7u));
        *(uint4*)((unsigned char*)qs + ((size_t)t * MR * 128u) + ((size_t)off16 << 4)) =
            *(const uint4*)tmp;
    }
    if (tid < MR) s_l[tid] = 0.0f;
    float m_run = -INFINITY, l_run = 0.0f;          // state lives in warps 0-1
    const uint32_t qh = warp >> 1;                  // col half (warps 0-3)
    const uint32_t my_row = (warp & 1u) * 32u + lane;
    const uint32_t my_tok = my_row / G;
    const uint32_t my_pos = (warp < 4u && my_tok < ntok)
        ? positions[tq0 + my_tok] : 0u;
    const uint32_t my_lo = (swa_window > 0 && my_pos + 1 > swa_window)
        ? my_pos + 1 - swa_window : 0u;

    // PAIR-uniform span (both ranks must agree on ntiles)
    const uint32_t last_q = tq_pair + 2u * TQ <= rows ? tq_pair + 2u * TQ - 1u
                                                      : rows - 1u;
    const uint32_t pos_last = positions[last_q];
    const uint32_t lo0 = (swa_window > 0 && positions[tq_pair] + 1 > swa_window)
        ? positions[tq_pair] + 1 - swa_window : 0u;
    const uint32_t span = pos_last + 1u - lo0;
    const uint32_t ntiles = (span + TK - 1u) / TK;
    const uint32_t slot = slots ? slots[0] : 0u;
    const uint32_t* bt = block_tables + (size_t)slot * blocks_per_slot;

    auto stage_kv = [&](uint32_t kt) {
        const uint32_t k0s = lo0 + kt * TK;
        const uint32_t nkeys_s = span - kt * TK < TK ? span - kt * TK : TK;
        if constexpr (F8) {
            // e4m3 strips to the upper byte halves; zero-fill in the expand
            const unsigned char* pk8 = (const unsigned char*)pool_k;
            const unsigned char* pv8 = (const unsigned char*)pool_v;
            unsigned char* kstrip = (unsigned char*)ks + (size_t)TKH * HD;
            for (uint32_t i = tid; i < TKH * (HD / 16u); i += 256u) {
                const uint32_t krl = i / (HD / 16u), l = i - krl * (HD / 16u);
                const uint32_t kr = crank * TKH + krl;
                if (kr < nkeys_s) {
                    const uint32_t gpos = k0s + kr;
                    const uint32_t blk = bt[gpos >> 4];
                    const size_t b8 = (size_t)blk * 16u * kv_dim
                        + (size_t)(gpos & 15u) * kv_dim + (size_t)kvh * HD + l * 16u;
                    pd_attn_cpa16((char*)(kstrip + (size_t)krl * HD + l * 16u),
                                  (const char*)(pk8 + b8));
                }
            }
            // V rank slice STRAIGHT into the swizzled e4m3 B image (2
            // column tiles of [TK x 128B]): local 16B unit u of band t <->
            // global unit t*16 + crank*8 + u. Tail rows ZERO-fill - stale
            // e4m3 bytes can decode NaN and 0 x NaN poisons the accumulate.
            for (uint32_t i = tid; i < TK * (HDH / 16u); i += 256u) {
                const uint32_t kr = i / (HDH / 16u), l = i - kr * (HDH / 16u);
                const uint32_t t = l >> 3, u = l & 7u;
                const uint32_t off16 = (kr >> 3) * 64u + (kr & 7u) * 8u + (u ^ (kr & 7u));
                unsigned char* vdst = vs + (size_t)t * TK * 128u + ((size_t)off16 << 4);
                if (kr < nkeys_s) {
                    const uint32_t lg = t * 16u + crank * 8u + u;
                    const uint32_t gpos = k0s + kr;
                    const uint32_t blk = bt[gpos >> 4];
                    const size_t b8 = (size_t)blk * 16u * kv_dim
                        + (size_t)(gpos & 15u) * kv_dim + (size_t)kvh * HD + lg * 16u;
                    pd_attn_cpa16((char*)vdst, (const char*)(pv8 + b8));
                } else {
                    *(uint4*)vdst = make_uint4(0u, 0u, 0u, 0u);
                }
            }
            pd_attn_cpa_commit();
            return;
        } else {
            // f16 KV: K this CTA's N-half, SW128 (c2 walk)
            for (uint32_t i = tid; i < TKH * (HD / 8u); i += 256u) {
                const uint32_t krl = i / (HD / 8u), c16 = i % (HD / 8u);
                const uint32_t kr = crank * TKH + krl;
                const uint32_t t = c16 >> 3, c = c16 & 7u;
                const uint32_t off16 = (krl >> 3) * 64u + (krl & 7u) * 8u + (c ^ (krl & 7u));
                unsigned char* kdst = (unsigned char*)ks
                    + ((size_t)t * TKH * 128u) + ((size_t)off16 << 4);
                if (kr < nkeys_s) {
                    const uint32_t gpos = k0s + kr;
                    const uint32_t blk = bt[gpos >> 4];
                    const size_t base = (size_t)blk * 16u * kv_dim
                        + (size_t)(gpos & 15u) * kv_dim + (size_t)kvh * HD + c16 * 8u;
                    pd_attn_cpa16(kdst, (const char*)(pool_k + base));
                } else {
                    *(uint4*)kdst = make_uint4(0u, 0u, 0u, 0u);
                }
            }
            // V rank slice, SW128 [TK x HDH]: local unit u of band t <-> global
            // dim chunk t*8 + crank*4 + u (see the F8 strip above)
            for (uint32_t i = tid; i < TK * (HDH / 8u); i += 256u) {
                const uint32_t kr = i / (HDH / 8u), c16 = i % (HDH / 8u);
                const uint32_t t = c16 >> 3, c = c16 & 7u;
                const uint32_t off16 = (kr >> 3) * 64u + (kr & 7u) * 8u + (c ^ (kr & 7u));
                unsigned char* vdst = (unsigned char*)vs
                    + ((size_t)t * TK * 128u) + ((size_t)off16 << 4);
                if (kr < nkeys_s) {
                    const uint32_t c16g = (c16 >> 3) * 16u + crank * 8u + (c16 & 7u);
                    const uint32_t gpos = k0s + kr;
                    const uint32_t blk = bt[gpos >> 4];
                    const size_t base = (size_t)blk * 16u * kv_dim
                        + (size_t)(gpos & 15u) * kv_dim + (size_t)kvh * HD + c16g * 8u;
                    pd_attn_cpa16(vdst, (const char*)(pool_v + base));
                } else {
                    *(uint4*)vdst = make_uint4(0u, 0u, 0u, 0u);
                }
            }
            pd_attn_cpa_commit();
        }
    };

    uint32_t done_ph = 0, peer_ph = 0, pv_ph = 0, pvd_ph = 0;
    for (uint32_t kt = 0; kt < ntiles; ++kt) {
        const uint32_t k0 = lo0 + kt * TK;
        const uint32_t nkeys = span - kt * TK < TK ? span - kt * TK : TK;
        if (kt > 0) {                     // vs/ps live until PV retires
            bar_wait(bpvd, pvd_ph ^ 1u);
        }
        stage_kv(kt);
        pd_attn_cpa_wait0();
        __syncthreads();
        if (F8) {
            // K half local rows only - V needs no expand: the e4m3 image is
            // staged in final swizzled form and the f8f6f4 PV eats it direct
            const uint32_t nk_loc = nkeys > crank * TKH
                ? (nkeys - crank * TKH < TKH ? nkeys - crank * TKH : TKH) : 0u;
            pd_pf5_f8_expand_k<TKH, HD>(ks, nk_loc, tid);
            __syncthreads();
        }
        if (tid == 0) *s_flag = 0u;
        if (crank == 1u && tid == 0)
            asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];"
                         ::"r"(peer_addr(bpeer)) : "memory");
        if (crank == 0u && tid == 0) {
            bar_wait(bpeer, peer_ph);
            const uint32_t a16 = (uint32_t)__cvta_generic_to_shared(qs) >> 4;
            const uint32_t b16 = (uint32_t)__cvta_generic_to_shared(ks) >> 4;
            #pragma unroll
            for (uint32_t kb2 = 0; kb2 < HD / 16u; ++kb2) {
                const uint32_t t = kb2 >> 2, c = (kb2 & 3u) * 2u;
                const uint64_t ad = pd_tc5_sdesc(a16 + t * (MR * 8u) + c);
                const uint64_t bd = pd_tc5_sdesc(b16 + t * (TKH * 8u) + c);
                const uint32_t id = (1u << 4) | ((TK >> 3) << 17) | ((128u >> 4) << 24);
                asm volatile(
                    "{\n\t.reg .pred p;\n\t"
                    "setp.ne.b32 p, %4, 0;\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n\t}"
                    ::"r"(tmem), "l"(ad), "l"(bd), "r"(id), "r"(kb2 > 0 ? 1u : 0u));
            }
            asm volatile(
                "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                ::"r"((uint32_t)__cvta_generic_to_shared(bdone)));
            asm volatile(
                "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                ::"r"(peer_addr(bdone)));
        }
        peer_ph ^= 1u;
        bar_wait(bdone, done_ph);
        done_ph ^= 1u;
        __syncthreads();
        // quadrant softmax (c2 form): warp w reads lanes w*32.. = rows
        // (w&1)*32.. x key cols qh*TKH..
        constexpr uint32_t NCC = (TKH + 31u) / 32u;
        float sv[NCC][32];
        float pm = -INFINITY;
        if (warp < 4u) {
            #pragma unroll
            for (uint32_t cc = 0; cc < NCC; ++cc) {
                uint32_t rr[32];
                const uint32_t taddr = tmem + ((warp * 32u) << 16) + cc * 32u;
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
                    "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
                    "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
                    : "=r"(rr[0]),"=r"(rr[1]),"=r"(rr[2]),"=r"(rr[3]),"=r"(rr[4]),"=r"(rr[5]),"=r"(rr[6]),"=r"(rr[7]),
                      "=r"(rr[8]),"=r"(rr[9]),"=r"(rr[10]),"=r"(rr[11]),"=r"(rr[12]),"=r"(rr[13]),"=r"(rr[14]),"=r"(rr[15]),
                      "=r"(rr[16]),"=r"(rr[17]),"=r"(rr[18]),"=r"(rr[19]),"=r"(rr[20]),"=r"(rr[21]),"=r"(rr[22]),"=r"(rr[23]),
                      "=r"(rr[24]),"=r"(rr[25]),"=r"(rr[26]),"=r"(rr[27]),"=r"(rr[28]),"=r"(rr[29]),"=r"(rr[30]),"=r"(rr[31])
                    : "r"(taddr));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (uint32_t j = 0; j < 32u; ++j) sv[cc][j] = __uint_as_float(rr[j]);
            }
            #pragma unroll
            for (uint32_t cc = 0; cc < NCC; ++cc)
                #pragma unroll
                for (uint32_t j = 0; j < 32u; ++j) {
                    const uint32_t kc = qh * TKH + cc * 32u + j;
                    const uint32_t kp = k0 + kc;
                    const bool ok = my_tok < ntok && kp >= my_lo && kp <= my_pos
                        && kc < nkeys && (cc * 32u + j) < TKH;
                    sv[cc][j] = ok ? sv[cc][j] * scale : -INFINITY;
                    pm = fmaxf(pm, sv[cc][j]);
                }
            if (qh == 1u) s_x1[my_row] = pm;
        }
        __syncthreads();
        float m_new = -INFINITY, corr = 1.0f;
        if (warp < 2u) {
            m_new = fmaxf(fmaxf(m_run, pm), s_x1[my_row]);
            corr = m_new > -INFINITY ? __expf(m_run - m_new) : 1.0f;
            s_mnew[my_row] = m_new;
            s_corr[my_row] = corr;
            if (kt > 0 && corr != 1.0f && my_tok < ntok) *s_flag = 1u;
        }
        __syncthreads();
        float lsum = 0.0f;
        if (warp < 4u) {
            if (qh == 1u) m_new = s_mnew[my_row];
            #pragma unroll
            for (uint32_t cc = 0; cc < NCC; ++cc)
                #pragma unroll
                for (uint32_t j = 0; j < 32u; j += 2u) {
                    if (cc * 32u + j >= TKH) break;
                    const float w0 = sv[cc][j] > -INFINITY ? __expf(sv[cc][j] - m_new) : 0.0f;
                    const float w1 = sv[cc][j + 1u] > -INFINITY ? __expf(sv[cc][j + 1u] - m_new) : 0.0f;
                    lsum += w0 + w1;
                    const uint32_t col = qh * TKH + cc * 32u + j;
                    if (F8) {
                        // e4m3 pair into the [MR x 128-BYTE-tiled] A image
                        const uint32_t tt = col >> 7, c16 = (col >> 4) & 7u;
                        const uint32_t off16 = (my_row >> 3) * 64u + (my_row & 7u) * 8u
                            + (c16 ^ (my_row & 7u));
                        unsigned char* pd = ps + ((size_t)tt * MR * 128u)
                            + ((size_t)off16 << 4) + (col & 15u);
                        pd[0] = __nv_fp8_e4m3(w0).__x;
                        pd[1] = __nv_fp8_e4m3(w1).__x;
                    } else {
                        // f16 pair into the SW128 [MR x 128-half] image
                        const uint32_t tt = col >> 6, c16 = (col >> 3) & 7u;
                        const uint32_t off16 = (my_row >> 3) * 64u + (my_row & 7u) * 8u
                            + (c16 ^ (my_row & 7u));
                        *(__half2*)(ps + ((size_t)tt * MR * 128u)
                            + ((size_t)off16 << 4) + (col & 7u) * 2u) =
                            __floats2half2_rn(w0, w1);
                    }
                }
            if (qh == 1u) s_x1[my_row] = lsum;
        }
        __syncthreads();
        if (warp < 2u) {
            l_run = l_run * corr + lsum + s_x1[my_row];
            m_run = s_mnew[my_row];
        }
        // O-rescale in tmem when any row's max moved (rare after warmup):
        // warps 0-3 own the quadrants; corr multiplies the row's whole slab
        if (*s_flag) {
            if (warp < 4u) {
                const float rc = s_corr[(warp * 32u + lane) & 63u];
                // the whole O region: 256 cols = 8 x32 chunks (both arms)
                #pragma unroll
                for (uint32_t ct = 0; ct < 8u; ++ct) {
                    uint32_t rr[32];
                    const uint32_t taddr = tmem + ((warp * 32u) << 16) + OB + ct * 32u;
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
                        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
                        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
                        : "=r"(rr[0]),"=r"(rr[1]),"=r"(rr[2]),"=r"(rr[3]),"=r"(rr[4]),"=r"(rr[5]),"=r"(rr[6]),"=r"(rr[7]),
                          "=r"(rr[8]),"=r"(rr[9]),"=r"(rr[10]),"=r"(rr[11]),"=r"(rr[12]),"=r"(rr[13]),"=r"(rr[14]),"=r"(rr[15]),
                          "=r"(rr[16]),"=r"(rr[17]),"=r"(rr[18]),"=r"(rr[19]),"=r"(rr[20]),"=r"(rr[21]),"=r"(rr[22]),"=r"(rr[23]),
                          "=r"(rr[24]),"=r"(rr[25]),"=r"(rr[26]),"=r"(rr[27]),"=r"(rr[28]),"=r"(rr[29]),"=r"(rr[30]),"=r"(rr[31])
                        : "r"(taddr));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (uint32_t j = 0; j < 32u; ++j)
                        rr[j] = __float_as_uint(__uint_as_float(rr[j]) * rc);
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x32.b32 [%32], "
                        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
                        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31};"
                        :: "r"(rr[0]),"r"(rr[1]),"r"(rr[2]),"r"(rr[3]),"r"(rr[4]),"r"(rr[5]),"r"(rr[6]),"r"(rr[7]),
                           "r"(rr[8]),"r"(rr[9]),"r"(rr[10]),"r"(rr[11]),"r"(rr[12]),"r"(rr[13]),"r"(rr[14]),"r"(rr[15]),
                           "r"(rr[16]),"r"(rr[17]),"r"(rr[18]),"r"(rr[19]),"r"(rr[20]),"r"(rr[21]),"r"(rr[22]),"r"(rr[23]),
                           "r"(rr[24]),"r"(rr[25]),"r"(rr[26]),"r"(rr[27]),"r"(rr[28]),"r"(rr[29]),"r"(rr[30]),"r"(rr[31]),
                           "r"(taddr));
                }
                asm volatile("tcgen05.wait::st.sync.aligned;");
                // producer-side fence: the rescale STs must be visible to
                // the PV mma issued by the (possibly peer-CTA) leader
                asm volatile("tcgen05.fence::before_thread_sync;");
            }
        }
        __syncthreads();
        // P (both CTAs) + rescale done: release rank1 to the leader, issue PV
        if (crank == 1u && tid == 0)
            asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];"
                         ::"r"(peer_addr(bpv)) : "memory");
        if (crank == 0u && tid == 0) {
            bar_wait(bpv, pv_ph);
            asm volatile("tcgen05.fence::after_thread_sync;");  // pairs the rescale fence
            const uint32_t p16 = (uint32_t)__cvta_generic_to_shared(ps) >> 4;
            const uint32_t v16 = (uint32_t)__cvta_generic_to_shared(vs) >> 4;
            if (F8) {
                // e4m3 PV: K32 chunks from the [MR x 128B] A image; B walk
                // per chunk = 32 image rows = 256 units; collective N=256 -
                // each rank's compact slice is its [TK x 128B] column tile
                #pragma unroll
                for (uint32_t kb = 0; kb < TK / 32u; ++kb) {
                    const uint32_t tp8 = kb >> 2, cp8 = (kb & 3u) * 2u;
                    const uint64_t ad = pd_tc5_sdesc(p16 + tp8 * (MR * 8u) + cp8);
                    #pragma unroll
                    for (uint32_t ct = 0; ct < NCT8; ++ct) {
                        const uint64_t bd = pd_tc5_sdesc(v16 + ct * (TK * 8u) + kb * 256u);
                        const uint32_t id = (1u << 4) | (1u << 16)
                            | ((256u >> 3) << 17) | ((128u >> 4) << 24);
                        asm volatile(
                            "{\n\t.reg .pred p;\n\t"
                            "setp.ne.b32 p, %4, 0;\n\t"
                            "tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t}"
                            ::"r"(tmem + OB + ct * 128u), "l"(ad), "l"(bd), "r"(id),
                              "r"(kt > 0 || kb > 0 ? 1u : 0u));
                    }
                }
            } else {
                #pragma unroll
                for (uint32_t kb = 0; kb < TK / 16u; ++kb) {
                    const uint32_t tp = kb >> 2, cp = (kb & 3u) * 2u;
                    const uint64_t ad = pd_tc5_sdesc(p16 + tp * (MR * 8u) + cp);
                    #pragma unroll
                    for (uint32_t ct = 0; ct < NCT; ++ct) {
                        // B read transposed (idesc bit 16), collective N=128:
                        // each rank supplies its compact [TK x 64] column tile
                        // ct at this shared address; per K16 chunk the walk
                        // advances 16 image rows = 128 units
                        const uint64_t bd = pd_tc5_sdesc(v16 + ct * (TK * 8u) + kb * 128u);
                        const uint32_t id = (1u << 4) | (1u << 16)
                            | ((128u >> 3) << 17) | ((128u >> 4) << 24);
                        asm volatile(
                            "{\n\t.reg .pred p;\n\t"
                            "setp.ne.b32 p, %4, 0;\n\t"
                            "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n\t}"
                            ::"r"(tmem + OB + ct * 64u), "l"(ad), "l"(bd), "r"(id),
                              "r"(kt > 0 || kb > 0 ? 1u : 0u));
                    }
                }
            }
            asm volatile(
                "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                ::"r"((uint32_t)__cvta_generic_to_shared(bpvd)));
            asm volatile(
                "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                ::"r"(peer_addr(bpvd)));
        }
        pv_ph ^= 1u;
        pvd_ph ^= 1u;
    }
    if (warp < 2u && my_tok < ntok) s_l[my_row] = l_run;
    if (ntiles) bar_wait(bpvd, pvd_ph ^ 1u);
    __syncthreads();
    // epilogue from tmem: warp w holds tmem lanes w*32.. - row = lane&63;
    // issue ct's fold covers dims ct*(2*CPI) + (lane>=64 ? CPI : 0) + [0,CPI)
    if (warp < 4u) {
        constexpr uint32_t NCTE = F8 ? NCT8 : NCT;   // issues
        constexpr uint32_t CPI = F8 ? 128u : 64u;    // fold cols per issue
        const uint32_t tl = warp * 32u + lane;
        const uint32_t row = tl & 63u;
        const uint32_t tok = row / G, g = row % G;
        const float inv_l = (tok < ntok && s_l[row] > 0.0f) ? 1.0f / s_l[row] : 0.0f;
        float* dst = tok < ntok
            ? out + ((size_t)(tq0 + tok) * n_heads + (size_t)kvh * G + g) * HD
            : nullptr;
        #pragma unroll
        for (uint32_t ct = 0; ct < NCTE; ++ct) {
            #pragma unroll
            for (uint32_t ch = 0; ch < CPI / 32u; ++ch) {
                uint32_t rr[32];
                const uint32_t taddr = tmem + ((warp * 32u) << 16) + OB
                    + ct * CPI + ch * 32u;
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
                    "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
                    "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
                    : "=r"(rr[0]),"=r"(rr[1]),"=r"(rr[2]),"=r"(rr[3]),"=r"(rr[4]),"=r"(rr[5]),"=r"(rr[6]),"=r"(rr[7]),
                      "=r"(rr[8]),"=r"(rr[9]),"=r"(rr[10]),"=r"(rr[11]),"=r"(rr[12]),"=r"(rr[13]),"=r"(rr[14]),"=r"(rr[15]),
                      "=r"(rr[16]),"=r"(rr[17]),"=r"(rr[18]),"=r"(rr[19]),"=r"(rr[20]),"=r"(rr[21]),"=r"(rr[22]),"=r"(rr[23]),
                      "=r"(rr[24]),"=r"(rr[25]),"=r"(rr[26]),"=r"(rr[27]),"=r"(rr[28]),"=r"(rr[29]),"=r"(rr[30]),"=r"(rr[31])
                    : "r"(taddr));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                if (dst) {
                    const uint32_t d0c = ct * 2u * CPI + (tl >= 64u ? CPI : 0u) + ch * 32u;
                    #pragma unroll
                    for (uint32_t j = 0; j < 32u; ++j)
                        dst[d0c + j] = __uint_as_float(rr[j]) * inv_l;
                }
            }
        }
    }
    __syncthreads();
    asm volatile("barrier.cluster.arrive;");
    asm volatile("barrier.cluster.wait;");
    if (tid < 32)
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, 512;"
                     ::"r"(tmem));
#else
    (void)q; (void)pool_k; (void)pool_v; (void)sinks; (void)out; (void)positions;
    (void)slots; (void)block_tables; (void)blocks_per_slot; (void)n_heads;
    (void)kv_dim; (void)swa_window; (void)rows; (void)scale;
#endif
}
#endif  // __cluster_dims__ guard (host pass || sm_90+)


PD_EXPORT
int pd_attn_prefill_f16_paged(const void* q, const void* pool_k, const void* pool_v,
                              const void* sinks, void* out, const void* positions,
                              const void* slots, const void* block_tables,
                              uint32_t blocks_per_slot, uint32_t n_heads, uint32_t n_kv_heads,
                              uint32_t head_dim, uint32_t kv_dim, uint32_t swa_window,
                              uint32_t batch, float scale, uint32_t kv_dtype, void* stream) {
    if (n_heads == 0 || batch == 0) return 0;
    if (head_dim != 256u && head_dim != 64u && head_dim != 512u && head_dim != 128u)
        return cudaErrorInvalidValue;
    // fp8 caches: the v3w (hd512 8:1) and v3s (hd256 2:1) tiles convert at
    // staging, the v4 qwen35 arm (hd256, G in {4,6,8}) expands raw e4m3
    // tiles in-kernel, the v4 granite/laguna/muse/paddleocr arm (hd128, G in
    // {4,6,8,9,16}) does the same, and the v4 gpt-oss arm (hd64, G=8)
    // too; every other f16-fragment tile requires half in memory
    if (kv_dtype == PD_KV_FP8_E4M3
        && !(head_dim == 512u && n_heads == 8u * n_kv_heads)
        && !(head_dim == 256u && n_heads == 2u * n_kv_heads && (n_kv_heads & 3u) == 0u)
        && !(head_dim == 256u
             && (n_heads == 4u * n_kv_heads || n_heads == 6u * n_kv_heads
                 || n_heads == 8u * n_kv_heads))
        && !(head_dim == 128u
             && (n_heads == 4u * n_kv_heads || n_heads == 6u * n_kv_heads
                 || n_heads == 8u * n_kv_heads || n_heads == 9u * n_kv_heads
                 || n_heads == 16u * n_kv_heads))
        && !(head_dim == 64u && n_heads == 8u * n_kv_heads))
        return cudaErrorInvalidValue;
    static bool carveout_done_p = false;
    if (!carveout_done_p) {
        pd_prefer_max_shared(pd_attn_prefill_f16_paged_kernel<256u>);
        pd_prefer_max_shared(pd_attn_prefill_f16_paged_kernel<64u>);
        pd_prefer_max_shared(pd_attn_prefill_f16_paged_kernel<512u>);
        pd_prefer_max_shared(pd_attn_prefill_f16_paged_kernel<128u>);
        carveout_done_p = true;
    }
    // 512 runs the NC=16 tile (see the kernel's smem note); dispatch before
    // the v2/v3 experiment arms - those are 256/64-only
    const uint32_t nc = head_dim >= 512u ? 16u : PD_AF16_NCOLS;
    dim3 grid(n_heads, (batch + nc - 1u) / nc);
    if (head_dim == 512u) {
        // pf5g-c2 cluster pair: cta_group::2 collective M=128 = 2
        // CTAs x 64 real Q rows - no garbage mma half, K staged split N-wise
        // (TK/2 keys per CTA), TK=64 in the freed smem. 2606.1 -> 2275.6 us
        // at the GLB 2048-row tick (-12.7%, 181 TF), oracle-tied; layout
        // probe-proven (clm128_probe: D folds 64 rows x TK cols into 128
        // lanes x TK/2, fold at N/2). kill: PADDOCK_NO_PF5GC2 -> pf5g below.
        static const bool no_pf5gc2 = pd_env("PADDOCK_NO_PF5GC2") != nullptr;
        // F8 arm: the cluster pair takes fp8 too - K expands the
        // CTA's TKH-row half before the bpeer release (the collective mma
        // reads the peer half over DSMEM). Kill: PADDOCK_NO_PF_F8 (family)
        // -> pf5g-F8 below; PADDOCK_NO_PF5GC2 -> pf5g either mode.
        static const bool no_pf_f8c2 = pd_env("PADDOCK_NO_PF_F8") != nullptr;
        const bool f8c2 = kv_dtype == PD_KV_FP8_E4M3;
        // pf6g: both GEMMs on tcgen05 - PV
        // via the transpose-B descriptor, O in tmem,
        // fp8 P.V on the KV8 arm. DEFAULT on after the serve gates (lab
        // -42% at done=6144, coherence clean).
        // Kill: PADDOCK_NO_PF6G -> pf5g_c2 below.
        static const bool pf6g_on = pd_env("PADDOCK_NO_PF6G") == nullptr;
        if (pf6g_on && swa_window == 0u && n_heads == 8u * n_kv_heads) {
            int dev6 = 0, ccm6 = 0;
            cudaGetDevice(&dev6);
            cudaDeviceGetAttribute(&ccm6, cudaDevAttrComputeCapabilityMajor, dev6);
            if (ccm6 == 10) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 900)
                constexpr uint32_t MR6 = 64u;
                // TK sweep (f8 PV needs TK%32): 64 2013 / 96 1813 / 128 1617
                // / 160 1530 us at done=6144 - deeper tiles amortize the
                // serialized per-tile path; 192 blows the smem launch limit.
                // Default 160; PADDOCK_PF6G_TK overrides for re-sweeps.
                static const uint32_t tk6 = [] {
                    const char* e = pd_env("PADDOCK_PF6G_TK");
                    const uint32_t v = e ? (uint32_t)atoi(e) : 160u;
                    return (v == 64u || v == 96u || v == 128u) ? v : 160u;
                }();
                dim3 g6(2u * ((batch + 15u) / 16u), n_kv_heads);
                const uint32_t* r6 = nullptr;
                if (pd_pf_runs_offs != nullptr) {
                    g6 = dim3(2u * ((pd_pf_runs_maxn + 15u) / 16u),
                              n_kv_heads, pd_pf_runs_n);
                    r6 = (const uint32_t*)pd_pf_runs_offs;
                }
                const bool f86 = kv_dtype == PD_KV_FP8_E4M3;
                // the f16 arm's wider slabs blow the smem ceiling past 128
                const uint32_t tk6e = (!f86 && tk6 > 128u) ? 128u : tk6;
                auto launch6 = [&](auto tkc) -> int {
                    constexpr uint32_t TK6 = decltype(tkc)::value;
                    // qs [64 x 512] + ks [TKH x 512] + V rank slice + P image
                    // + softmax state + barriers + align; F8 arm: e4m3 V + P
                    const uint32_t base6 = MR6 * 512u * 2u + (TK6 / 2u) * 512u * 2u
                        + 4u * MR6 * 4u + 8u + 4u * 8u + 1024u;
                    const uint32_t smem6f8 = base6 + TK6 * 256u
                        + MR6 * (TK6 > 128u ? 256u : 128u);
                    const uint32_t smem6f16 = base6 + TK6 * 256u * 2u
                        + MR6 * (TK6 > 128u ? 384u : 256u);
                    static bool a6 = false;
                    if (!a6) {
                        cudaFuncSetAttribute(
                            (const void*)pd_attn_prefill_pf6g_kernel<512u, 8u, TK6>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem6f16);
                        cudaFuncSetAttribute(
                            (const void*)pd_attn_prefill_pf6g_kernel<512u, 8u, TK6, true>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem6f8);
                        a6 = true;
                    }
                    if (f86)
                        pd_attn_prefill_pf6g_kernel<512u, 8u, TK6, true><<<g6, 256, smem6f8, (cudaStream_t)stream>>>(
                            (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                            (const float*)sinks, (float*)out, (const unsigned int*)positions,
                            (const unsigned int*)slots, (const uint32_t*)block_tables,
                            blocks_per_slot, n_heads, kv_dim, swa_window, batch, scale, r6);
                    else
                        pd_attn_prefill_pf6g_kernel<512u, 8u, TK6><<<g6, 256, smem6f16, (cudaStream_t)stream>>>(
                            (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                            (const float*)sinks, (float*)out, (const unsigned int*)positions,
                            (const unsigned int*)slots, (const uint32_t*)block_tables,
                            blocks_per_slot, n_heads, kv_dim, swa_window, batch, scale, r6);
                    return pd_launch_status();
                };
                if (tk6e == 96u)
                    return launch6(std::integral_constant<uint32_t, 96u>{});
                if (tk6e == 128u)
                    return launch6(std::integral_constant<uint32_t, 128u>{});
                if (tk6e == 64u)
                    return launch6(std::integral_constant<uint32_t, 64u>{});
                return launch6(std::integral_constant<uint32_t, 160u>{});
#endif
            }
        }
        if (!no_pf5gc2 && (!f8c2 || !no_pf_f8c2) && swa_window == 0u
            && n_heads == 8u * n_kv_heads) {
            int devc2 = 0, ccmc2 = 0;
            cudaGetDevice(&devc2);
            cudaDeviceGetAttribute(&ccmc2, cudaDevAttrComputeCapabilityMajor, devc2);
            if (ccmc2 == 10) {
// the kernel definition is compiled out below sm_90 (see its __cluster_dims__
// guard); these references must vanish from those device passes too - the
// host pass keeps them and the runtime cc gate above keeps them unreachable
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 900)
                constexpr uint32_t TKC2 = 80u, MRC2 = 64u;   // TK sweep: 64 2275.6 / 80 2041.6 / 96 2130.5 us
                const uint32_t smemc2 = MRC2 * 512u * 2u + (TKC2 / 2u) * 512u * 2u
                    + TKC2 * (512u + 8u) * 2u + MRC2 * (TKC2 + 8u) * 2u
                    + 4u * MRC2 * 4u + 16u + 1024u;
                static bool ac2 = false;
                if (!ac2) {
                    cudaFuncSetAttribute(
                        (const void*)pd_attn_prefill_pf5g_c2_kernel<512u, 8u, TKC2>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemc2);
                    cudaFuncSetAttribute(
                        (const void*)pd_attn_prefill_pf5g_c2_kernel<512u, 8u, TKC2, true>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemc2);
                    ac2 = true;
                }
                // pair covers 16 queries; clusters tile grid.x (2 CTAs each)
                dim3 gc2(2u * ((batch + 15u) / 16u), n_kv_heads);
                if (f8c2)
                    pd_attn_prefill_pf5g_c2_kernel<512u, 8u, TKC2, true><<<gc2, 256, smemc2, (cudaStream_t)stream>>>(
                        (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                        (const float*)sinks, (float*)out, (const unsigned int*)positions,
                        (const unsigned int*)slots, (const uint32_t*)block_tables,
                        blocks_per_slot, n_heads, kv_dim, swa_window, batch, scale);
                else
                    pd_attn_prefill_pf5g_c2_kernel<512u, 8u, TKC2><<<gc2, 256, smemc2, (cudaStream_t)stream>>>(
                        (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                        (const float*)sinks, (float*)out, (const unsigned int*)positions,
                        (const unsigned int*)slots, (const uint32_t*)block_tables,
                        blocks_per_slot, n_heads, kv_dim, swa_window, batch, scale);
                return pd_launch_status();
#endif
            }
        }
        // pf5g tcgen05-S tile: 3372.5 -> 2606.1 us at the GLB
        // 2048-row tick (158 vs 122 TF), same f16 class. sm_100a only;
        // kill: PADDOCK_NO_PF5G falls to the v4/GQA arms below.
        static const bool no_pf5g = pd_env("PADDOCK_NO_PF5G") != nullptr;
        // F8 arm: fp8 KV takes the same tcgen05 tile via in-place
        // e4m3 expansion (pd_pf5_f8_expand) - before this, fp8 GLB prefill
        // fell to the v3-512 walk class. Same smem (upper-strip staging).
        // pf5g-c2 stays f16-only for now; fp8 lands here (TK=32 tile).
        // PADDOCK_NO_PF_F8 kills the whole prefill f8 family.
        static const bool no_pf_f8 = pd_env("PADDOCK_NO_PF_F8") != nullptr;
        const bool f8g = kv_dtype == PD_KV_FP8_E4M3;
        if (!no_pf5g && (!f8g || !no_pf_f8) && swa_window == 0u
            && n_heads == 8u * n_kv_heads) {
            int dev5g = 0, ccm5g = 0;
            cudaGetDevice(&dev5g);
            cudaDeviceGetAttribute(&ccm5g, cudaDevAttrComputeCapabilityMajor, dev5g);
            if (ccm5g == 10) {
                constexpr uint32_t TK5G = 32u, MR5G = 64u, TQ5G = MR5G / 8u;
                const uint32_t smem5g = MR5G * 512u * 2u + 2u * TK5G * 512u * 2u
                    + 2u * TK5G * (512u + 8u) * 2u + MR5G * (TK5G + 8u) * 2u
                    + 2u * MR5G * 4u + 16u + 1024u;
                static bool a5g = false;
                if (!a5g) {
                    cudaFuncSetAttribute(
                        (const void*)pd_attn_prefill_pf5g_kernel<512u, 8u, TK5G>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem5g);
                    cudaFuncSetAttribute(
                        (const void*)pd_attn_prefill_pf5g_kernel<512u, 8u, TK5G, true>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem5g);
                    a5g = true;
                }
                dim3 g5g(n_kv_heads, (batch + TQ5G - 1u) / TQ5G);
                if (f8g)
                    pd_attn_prefill_pf5g_kernel<512u, 8u, TK5G, true><<<g5g, 256, smem5g, (cudaStream_t)stream>>>(
                        (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                        (const float*)sinks, (float*)out, (const unsigned int*)positions,
                        (const unsigned int*)slots, (const uint32_t*)block_tables,
                        blocks_per_slot, n_heads, kv_dim, swa_window, batch, scale);
                else
                    pd_attn_prefill_pf5g_kernel<512u, 8u, TK5G><<<g5g, 256, smem5g, (cudaStream_t)stream>>>(
                        (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                        (const float*)sinks, (float*)out, (const unsigned int*)positions,
                        (const unsigned int*)slots, (const uint32_t*)block_tables,
                        blocks_per_slot, n_heads, kv_dim, swa_window, batch, scale);
                return pd_launch_status();
            }
        }
        // GQA-fused v3-512 tile (default): 4 heads x 2 D-half warps per
        // block, K/V staged once per 4 heads - the per-q-head WMMA tile
        // below re-walked K/V 8x and ran ~50 TF vs the hd256 tile's 162
        // PD_NO_PF512_GQA pins the WMMA arm for A/B.
        // FA-512 rung (PADDOCK_G4_PF_FA512=1): the prefill-FA tile at the
        // global geometry - M=32 (k1=4 rows x G8), PT16, split-pipelined SB.
        // 2x the KV walks of v3w (k1 4 vs NR 16 in half-groups) but v3w is
        // latency-bound (compute 36%, L2 21%), so traffic has headroom.
        static const bool pf_fa512 = pd_env("PADDOCK_G4_PF_FA512") != nullptr;
        if (pf_fa512 && kv_dtype != PD_KV_FP8_E4M3 && n_heads == 8u * n_kv_heads) {
            constexpr uint32_t FK1 = 4u, FMp = 32u, FPT = 16u;
            const uint32_t smem = FMp * (512u + 8u) * 2u
                + 2u * FPT * (512u + 8u) * 2u
                + FMp * (FPT + 1u) * 4u + 3u * FMp * 4u + FMp * (FPT + 8u) * 2u;
            static bool a512 = false;
            if (!a512) {
                cudaFuncSetAttribute(
                    (const void*)pd_attn_prefill_fa_kernel<FPT, 2u, false, 256u, 512u, 3u, true>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
                a512 = true;
            }
            dim3 gf(n_kv_heads, (batch + FK1 - 1u) / FK1);
            pd_attn_prefill_fa_kernel<FPT, 2u, false, 256u, 512u, 3u, true><<<gf, 256, smem, (cudaStream_t)stream>>>(
                (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                (const float*)sinks, (float*)out, (const unsigned int*)positions,
                (const unsigned int*)slots, (const uint32_t*)block_tables,
                blocks_per_slot, n_heads, n_kv_heads, head_dim, kv_dim,
                swa_window, batch, FK1, scale);
            return pd_launch_status();
        }
        // v4 staged-HMMA tile (B200): K/V cp.async-staged once per
        // (kv head, token tile) for all 8 heads, HMMA scores+PV, O in regs.
        // 2.9 ms -> 1.6 ms at the pf8 global shape (122 vs 68 TF); same
        // f16 numeric class (f64 oracle: 0.0717 vs v3w's 0.0718).
        // f16 caches only; kill: PADDOCK_NO_PF_V4.
        static const bool no_v4 = pd_env("PADDOCK_NO_PF_V4") != nullptr;
        if (!no_v4 && kv_dtype != PD_KV_FP8_E4M3 && n_heads == 8u * n_kv_heads) {
            constexpr uint32_t V4TK = 16u, V4MR = 32u, V4TQ = 4u;
            const uint32_t rowe = 512u + 8u, ts4 = V4TK + 8u;
            const uint32_t smem = V4MR * rowe * 2u + 4u * V4TK * rowe * 2u
                + V4MR * ts4 * 4u + V4MR * ts4 * 2u + 2u * V4MR * 4u + V4TQ * 4u;
            // device smem-cap guard (sm_120 port fix, the FA-port lesson):
            // this tile is 104,720B - fine on B200's 227KB, 3.3KB over
            // sm_120's 101,376 opt-in cap. The unguarded attr call failed
            // silently and every prefill launch after it returned error 1
            // (every prefill request 500'd). Oversized geometries
            // fall through to v3w.
            static int v4w_cap = -1;
            if (v4w_cap < 0) {
                int dev = 0;
                cudaGetDevice(&dev);
                if (cudaDeviceGetAttribute(&v4w_cap,
                        cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess)
                    v4w_cap = 48 * 1024;
            }
            if (smem <= (uint32_t)v4w_cap) {
            static bool a4w = false;
            if (!a4w) {
                cudaFuncSetAttribute(
                    (const void*)pd_attn_prefill_f16_v4_kernel<512u, 8u, V4TK>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
                a4w = true;
            }
            dim3 g4(n_kv_heads, (batch + V4TQ - 1u) / V4TQ);
            pd_attn_prefill_f16_v4_kernel<512u, 8u, V4TK><<<g4, 256, smem, (cudaStream_t)stream>>>(
                (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                (const float*)sinks, (float*)out, (const unsigned int*)positions,
                (const uint32_t*)block_tables, blocks_per_slot,
                (const unsigned int*)slots, n_heads, kv_dim, swa_window, batch,
                scale);
            return pd_launch_status();
            }
        }
        static const bool no_gqa512 = pd_env("PD_NO_PF512_GQA") != nullptr;
        if ((!no_gqa512 || kv_dtype == PD_KV_FP8_E4M3) && n_heads == 8u * n_kv_heads) {
            static bool a3w = false;
            if (!a3w) {
                cudaFuncSetAttribute((const void*)pd_attn_prefill_f16_v3w_kernel<512u, __half>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     PD_AF3W_SMEM);
                cudaFuncSetAttribute((const void*)pd_attn_prefill_f16_v3w_kernel<512u, __nv_fp8_e4m3>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     PD_AF3W_SMEM);
                a3w = true;
            }
            dim3 gw(n_kv_heads, (batch + PD_AF3W_NR - 1u) / PD_AF3W_NR, 2u);
            if (kv_dtype == PD_KV_FP8_E4M3)
                pd_attn_prefill_f16_v3w_kernel<512u, __nv_fp8_e4m3><<<gw, 256, PD_AF3W_SMEM, (cudaStream_t)stream>>>(
                    (const float*)q, (const __nv_fp8_e4m3*)pool_k, (const __nv_fp8_e4m3*)pool_v,
                    (const float*)sinks, (float*)out, (const unsigned int*)positions,
                    (const unsigned int*)slots, (const uint32_t*)block_tables, blocks_per_slot,
                    n_heads, n_kv_heads, 0u, kv_dim, swa_window, batch, scale);
            else
                pd_attn_prefill_f16_v3w_kernel<512u, __half><<<gw, 256, PD_AF3W_SMEM, (cudaStream_t)stream>>>(
                    (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                    (const float*)sinks, (float*)out, (const unsigned int*)positions,
                    (const unsigned int*)slots, (const uint32_t*)block_tables, blocks_per_slot,
                    n_heads, n_kv_heads, 0u, kv_dim, swa_window, batch, scale);
            return pd_launch_status();
        }
        pd_attn_prefill_f16_paged_kernel<512u><<<grid, 128, 0, (cudaStream_t)stream>>>(
            (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
            (const float*)sinks, (float*)out, (const unsigned int*)positions,
            (const unsigned int*)slots, (const uint32_t*)block_tables, blocks_per_slot,
            n_heads, n_kv_heads, kv_dim, swa_window, batch, scale);
        return pd_launch_status();
    }
    // FA-2 prefill tile (PADDOCK_G4_PF_FA=1, A/B rung): hd256 group-fused
    // 32-row chunks, KV staged once per chunk for the whole q-group. Needs
    // the consecutive-position contract (engine per-slot chunk prefill) and
    // the 99KB opt-in smem window; f16 caches only for now.
    // variant select by env value: "1"/"16" = PT16 double-buffered ring,
    // "32"/"48" = wider single-buffered stages (fewer barrier rounds)
    // pf5 tcgen05-S tile: 587 -> 395 us at the pf8 tick (174 vs
    // 117 TF), exact class vs v4. sm_100a only (tcgen05); kill:
    // PADDOCK_NO_PF5 falls to the v4 arm below.
    static const bool no_pf5 = pd_env("PADDOCK_NO_PF5") != nullptr;
    // F8 arm: fp8 SWA prefill rides the same tcgen05 tile via
    // in-place e4m3 expansion; was falling to the v3s walk class.
    // PADDOCK_NO_PF_F8 kills (same env as the hd512 arm).
    static const bool no_pf_f8s = pd_env("PADDOCK_NO_PF_F8") != nullptr;
    const bool f8s = kv_dtype == PD_KV_FP8_E4M3;
    if (!no_pf5 && head_dim == 256u && (!f8s || !no_pf_f8s)
        && n_heads == 2u * n_kv_heads) {
        int dev5 = 0, ccm5 = 0, ccn5 = 0;
        cudaGetDevice(&dev5);
        cudaDeviceGetAttribute(&ccm5, cudaDevAttrComputeCapabilityMajor, dev5);
        cudaDeviceGetAttribute(&ccn5, cudaDevAttrComputeCapabilityMinor, dev5);
        if (ccm5 == 10) {
            constexpr uint32_t TK5 = 64u, MR5 = 128u, TQ5 = MR5 / 2u;
            const uint32_t smem5 = MR5 * 256u * 2u + 2u * TK5 * 256u * 2u
                + 2u * TK5 * (256u + 8u) * 2u + MR5 * (TK5 + 8u) * 2u
                + 2u * MR5 * 4u + 32u + 1024u;
            static bool a5 = false;
            if (!a5) {
                cudaFuncSetAttribute(
                    (const void*)pd_attn_prefill_pf5_kernel<256u, 2u, TK5>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem5);
                cudaFuncSetAttribute(
                    (const void*)pd_attn_prefill_pf5_kernel<256u, 2u, TK5, true>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem5);
                cudaFuncSetAttribute(
                    (const void*)pd_attn_prefill_pf5_kernel<256u, 2u, TK5, true, true>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem5);
                cudaFuncSetAttribute(
                    (const void*)pd_attn_prefill_pf5_kernel<256u, 2u, TK5, true, true, true>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem5);
                a5 = true;
            }
            dim3 g5(n_kv_heads, (batch + TQ5 - 1u) / TQ5);
            const uint32_t* r5 = nullptr;
            if (pd_pf_runs_offs != nullptr) {
                g5 = dim3(n_kv_heads,
                          (pd_pf_runs_maxn + TQ5 - 1u) / TQ5,
                          pd_pf_runs_n);
                r5 = (const uint32_t*)pd_pf_runs_offs;
            }
            // pf6s: the pf6g treatment on
            // the SWA tile - P.V on tcgen05 kind::f8f6f4, O in tmem, e4m3
            // V staged swizzle-direct (expand_v dies). KV8 arm only.
            // Bring-up opt-in PADDOCK_PF6S=1; PADDOCK_NO_PF6S kills.
            static const bool pf6s_on = pd_env("PADDOCK_PF6S") != nullptr
                && pd_env("PADDOCK_NO_PF6S") == nullptr;
            if (pf6s_on && f8s) {
                const uint32_t smem6s = MR5 * 256u * 2u + 2u * TK5 * 256u * 2u
                    + 2u * TK5 * 256u + MR5 * 128u + 2u * MR5 * 4u + 8u
                    + 2u * 8u + 1024u;
                static bool a6s = false;
                if (!a6s) {
                    cudaFuncSetAttribute(
                        (const void*)pd_attn_prefill_pf6s_kernel<256u, 2u, TK5>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem6s);
                    a6s = true;
                }
                pd_attn_prefill_pf6s_kernel<256u, 2u, TK5><<<g5, 256, smem6s, (cudaStream_t)stream>>>(
                    (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                    (const float*)sinks, (float*)out, (const unsigned int*)positions,
                    (const unsigned int*)slots, (const uint32_t*)block_tables,
                    blocks_per_slot, n_heads, kv_dim, swa_window, batch, scale, r5);
                return pd_launch_status();
            }
            //  arm (PADDOCK_PF_F8QK=1): fp8-NATIVE QK^T - K consumed
            // as e4m3 by kind::f8f6f4, Q direct-cast e4m3 in-kernel (the
            // same numeric treatment KV8 gives K). Labeled precision class:
            // coherence + greedy parity gate. V/PV stays the f16 pipe.
            static const bool f8qk = pd_env("PADDOCK_PF_F8QK") != nullptr;
            //  arm (PADDOCK_PF_KVBULK=1, rides F8QK): bulk KV staging
            // - full pages via cp.async.bulk.tensor.2d over the pool tensor
            // maps, ragged edges keep cp.async. Same bytes/layout/mma:
            // bit-identical class. Maps cached per (pool, kv_dim) like the
            // decode caches; encode failure falls back to the classic path.
            static const bool kvbulk = pd_env("PADDOCK_PF_KVBULK") != nullptr;
            static CUtensorMap pf5_tmk, pf5_tmv;
            static const CUtensorMap pf5_tm0 = {};
            static const void* pf5_tmkey = nullptr;
            static uint32_t pf5_tmkd = 0;
            bool bulk_ok = false;
            if (f8s && f8qk && kvbulk) {
                if (pf5_tmkey == (const void*)pool_k && pf5_tmkd == kv_dim) {
                    bulk_ok = true;
                } else if (pd_attn_tmap_kv_f8s(&pf5_tmk, pool_k, kv_dim)
                        && pd_attn_tmap_v256(&pf5_tmv, pool_v, kv_dim)) {
                    pf5_tmkey = (const void*)pool_k;
                    pf5_tmkd = kv_dim;
                    bulk_ok = true;
                }
                // route-fired witness (55 trap: a silent encode
                // failure would fall back bit-identically - make it loud)
                static bool bulk_said = false;
                if (!bulk_said) {
                    // stderr: unbuffered, so the line survives a kill -9
                    ::fprintf(stderr, "[pf5-kvbulk] %s\n",
                              bulk_ok ? "ENGAGED (tmaps encoded)"
                                      : "FALLBACK (tmap encode failed)");
                    bulk_said = true;
                }
            }
            if (f8s && f8qk && bulk_ok)
                pd_attn_prefill_pf5_kernel<256u, 2u, TK5, true, true, true><<<g5, 256, smem5, (cudaStream_t)stream>>>(
                    pf5_tmk, pf5_tmv,
                    (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                    (const float*)sinks, (float*)out, (const unsigned int*)positions,
                    (const unsigned int*)slots, (const uint32_t*)block_tables,
                    blocks_per_slot, n_heads, kv_dim, swa_window, batch, scale, r5);
            else if (f8s && f8qk)
                pd_attn_prefill_pf5_kernel<256u, 2u, TK5, true, true><<<g5, 256, smem5, (cudaStream_t)stream>>>(
                    pf5_tm0, pf5_tm0,
                    (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                    (const float*)sinks, (float*)out, (const unsigned int*)positions,
                    (const unsigned int*)slots, (const uint32_t*)block_tables,
                    blocks_per_slot, n_heads, kv_dim, swa_window, batch, scale, r5);
            else if (f8s)
                pd_attn_prefill_pf5_kernel<256u, 2u, TK5, true><<<g5, 256, smem5, (cudaStream_t)stream>>>(
                    pf5_tm0, pf5_tm0,
                    (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                    (const float*)sinks, (float*)out, (const unsigned int*)positions,
                    (const unsigned int*)slots, (const uint32_t*)block_tables,
                    blocks_per_slot, n_heads, kv_dim, swa_window, batch, scale, r5);
            else
                pd_attn_prefill_pf5_kernel<256u, 2u, TK5><<<g5, 256, smem5, (cudaStream_t)stream>>>(
                    pf5_tm0, pf5_tm0,
                    (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                    (const float*)sinks, (float*)out, (const unsigned int*)positions,
                    (const unsigned int*)slots, (const uint32_t*)block_tables,
                    blocks_per_slot, n_heads, kv_dim, swa_window, batch, scale, r5);
            return pd_launch_status();
        }
    }
    // v4 staged-HMMA tile for the SWA shape (B200): 1137 -> 589 us
    // at the pf8 tick (117 vs 60 TF), same f16 class (oracle 0.085 vs the
    // WMMA tile's 0.060). f16 caches; kill: PADDOCK_NO_PF_V4.
    static const bool no_v4s = pd_env("PADDOCK_NO_PF_V4") != nullptr;
    if (!no_v4s && head_dim == 256u && kv_dtype != PD_KV_FP8_E4M3
        && n_heads == 2u * n_kv_heads) {
        constexpr uint32_t V4TK = 16u, V4MR = 64u, V4TQ = 32u;
        const uint32_t rowe = 256u + 8u, ts4 = V4TK + 8u;
        const uint32_t smem = V4MR * rowe * 2u + 4u * V4TK * rowe * 2u
            + V4MR * ts4 * 4u + V4MR * ts4 * 2u + 2u * V4MR * 4u + V4TQ * 4u;
        // same device-cap guard as the hd512 arm (77.4KB fits sm_120 today;
        // the guard protects any retune)
        static int v4s_cap = -1;
        if (v4s_cap < 0) {
            int dev = 0;
            cudaGetDevice(&dev);
            if (cudaDeviceGetAttribute(&v4s_cap,
                    cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess)
                v4s_cap = 48 * 1024;
        }
        if (smem <= (uint32_t)v4s_cap) {
        static bool a4s = false;
        if (!a4s) {
            cudaFuncSetAttribute(
                (const void*)pd_attn_prefill_f16_v4_kernel<256u, 2u, V4TK>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
            a4s = true;
        }
        dim3 g4(n_kv_heads, (batch + V4TQ - 1u) / V4TQ);
        pd_attn_prefill_f16_v4_kernel<256u, 2u, V4TK><<<g4, 256, smem, (cudaStream_t)stream>>>(
            (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
            (const float*)sinks, (float*)out, (const unsigned int*)positions,
            (const uint32_t*)block_tables, blocks_per_slot,
            (const unsigned int*)slots, n_heads, kv_dim, swa_window, batch,
            scale);
        return pd_launch_status();
        }
    }
    // v4 staged-HMMA tile for the qwen35 full-attn shapes: G=4 (Qwen3.5-9B
    // 16q/4kv), G=8 (Qwen3.6-35B-A3B 16q/2kv) and G=6 (Qwen3.6-27B 24q/4kv,
    // the MR=48 instantiation - 64 % 6 != 0 was the WMMA fallthrough). Both
    // cache dtypes: the fp8 arm rides the v3c PIPE class inside the same
    // kernel (raw e4m3 cp.async tiles + widened-cvt expand into the identical
    // half layout) - before it the elected kv8 class ran the SCALAR paged
    // tile (per-q-head grid, G-fold redundant KV walks, no tensor cores).
    // The incumbent WMMA tile stays the f16 fallthrough; fp8 terminates at
    // the scalar launcher (nothing below can read e4m3). Kill:
    // PADDOCK_NO_PF_V4 (shared with the gemma4 arms). Per-instantiation
    // exact smem attrs - a shared latch is the deltanet/core.cuh x2_optin trap.
    if (head_dim == 256u
        && (n_heads == 4u * n_kv_heads || n_heads == 6u * n_kv_heads
            || n_heads == 8u * n_kv_heads)) {
        const bool f8v4 = kv_dtype == PD_KV_FP8_E4M3;
        const uint32_t g_ = n_heads / n_kv_heads;
        // pf7: fa2-class register-resident tile, the fp8 election above the
        // v4 PIPE arm (attention front - see the kernel's comment
        // for the class rationale and proto ladder). Kill: PADDOCK_NO_PF7
        // -> the v4 fp8 arm below (PADDOCK_NO_PF_V4 still kills both).
        static const bool no_pf7 = pd_env("PADDOCK_NO_PF7") != nullptr;
        // pf7rp: repacked-f16-pane arm above pf7 (door 2)
        // - BIT-IDENTICAL output to pf7 (same cvt on the same
        // bytes in the same mma order; word-compare gated in the proto),
        // -5..-21% across the ladder legs. Kill: PADDOCK_NO_PF7RP -> pf7.
        static const bool no_rp = pd_env("PADDOCK_NO_PF7RP") != nullptr;
        if (f8v4 && !no_pf7 && !no_rp && !no_v4s) {
            constexpr uint32_t RPSM =
                2u * 64u * 264u * 2u + 2u * 64u * 256u + 256u;
            static int rpcap = -1;
            if (rpcap < 0) {
                int dev = 0;
                cudaGetDevice(&dev);
                if (cudaDeviceGetAttribute(&rpcap,
                        cudaDevAttrMaxSharedMemoryPerBlockOptin, dev)
                    != cudaSuccess)
                    rpcap = 48 * 1024;
            }
            if (RPSM <= (uint32_t)rpcap) {
                static bool arp = false;
                if (!arp) {
                    cudaFuncSetAttribute(
                        (const void*)pd_attn_prefill_pf7rp_kernel<4u>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)RPSM);
                    cudaFuncSetAttribute(
                        (const void*)pd_attn_prefill_pf7rp_kernel<6u>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)RPSM);
                    cudaFuncSetAttribute(
                        (const void*)pd_attn_prefill_pf7rp_kernel<8u>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)RPSM);
                    ::fprintf(stderr, "[pf7rp] ENGAGED (repacked-pane hd256 fp8)\n");
                    arp = true;
                }
                dim3 g7(n_kv_heads, (batch * g_ + 63u) / 64u);
#define PD_PF7RP_LAUNCH(GV)                                                    \
    pd_attn_prefill_pf7rp_kernel<GV>                                           \
        <<<g7, 128, RPSM, (cudaStream_t)stream>>>(                             \
            (const float*)q, (const unsigned char*)pool_k,                     \
            (const unsigned char*)pool_v, (const float*)sinks, (float*)out,    \
            (const unsigned int*)positions, (const uint32_t*)block_tables,     \
            blocks_per_slot, (const unsigned int*)slots, n_heads, kv_dim,      \
            swa_window, batch, scale)
                if (g_ == 4u) PD_PF7RP_LAUNCH(4u);
                else if (g_ == 6u) PD_PF7RP_LAUNCH(6u);
                else PD_PF7RP_LAUNCH(8u);
#undef PD_PF7RP_LAUNCH
                return pd_launch_status();
            }
        }
        if (f8v4 && !no_pf7 && !no_v4s) {
            constexpr uint32_t P7SM = 64u * 264u * 2u + 3u * 64u * 272u + 256u;
            static int p7cap = -1;
            if (p7cap < 0) {
                int dev = 0;
                cudaGetDevice(&dev);
                if (cudaDeviceGetAttribute(&p7cap,
                        cudaDevAttrMaxSharedMemoryPerBlockOptin, dev)
                    != cudaSuccess)
                    p7cap = 48 * 1024;
            }
            if (P7SM <= (uint32_t)p7cap) {
                static bool a7 = false;
                if (!a7) {
                    cudaFuncSetAttribute(
                        (const void*)pd_attn_prefill_pf7_kernel<256u, 4u>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)P7SM);
                    cudaFuncSetAttribute(
                        (const void*)pd_attn_prefill_pf7_kernel<256u, 6u>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)P7SM);
                    cudaFuncSetAttribute(
                        (const void*)pd_attn_prefill_pf7_kernel<256u, 8u>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)P7SM);
                    // route-fired witness (55 trap: silent fallback
                    // would be bit-plausible - make the election loud)
                    ::fprintf(stderr, "[pf7] ENGAGED (hd256 fp8 fa2-class)\n");
                    a7 = true;
                }
                dim3 g7(n_kv_heads, (batch * g_ + 63u) / 64u);
#define PD_PF7_LAUNCH(GV)                                                      \
    pd_attn_prefill_pf7_kernel<256u, GV>                                             \
        <<<g7, 128, P7SM, (cudaStream_t)stream>>>(                             \
            (const float*)q, (const unsigned char*)pool_k,                     \
            (const unsigned char*)pool_v, (const float*)sinks, (float*)out,    \
            (const unsigned int*)positions, (const uint32_t*)block_tables,     \
            blocks_per_slot, (const unsigned int*)slots, n_heads, kv_dim,      \
            swa_window, batch, scale)
                if (g_ == 4u) PD_PF7_LAUNCH(4u);
                else if (g_ == 6u) PD_PF7_LAUNCH(6u);
                else PD_PF7_LAUNCH(8u);
#undef PD_PF7_LAUNCH
                return pd_launch_status();
            }
        }
        if (!no_v4s) {
        constexpr uint32_t V4TK = 16u;
        const uint32_t v4mr = g_ == 6u ? 48u : 64u;
        const uint32_t v4tq = v4mr / g_;  // 16 rows/CTA at G=4, 8 at G=6/G=8
        const uint32_t rowe = 256u + 8u, ts4 = V4TK + 8u;
        // f8 arms add the raw e4m3 stage region (2 bufs x K,V x TK x 256B)
        auto v4q_smem = [&](uint32_t mr, uint32_t tq, bool f8) {
            return mr * rowe * 2u + 4u * V4TK * rowe * 2u
                + (f8 ? 4u * V4TK * 256u : 0u)
                + mr * ts4 * 4u + mr * ts4 * 2u + 2u * mr * 4u + tq * 4u;
        };
        const uint32_t smem = v4q_smem(v4mr, v4tq, f8v4);
        static int v4q_cap = -1;
        if (v4q_cap < 0) {
            int dev = 0;
            cudaGetDevice(&dev);
            if (cudaDeviceGetAttribute(&v4q_cap,
                    cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess)
                v4q_cap = 48 * 1024;
        }
        if (smem <= (uint32_t)v4q_cap) {
            static bool a4q = false;
            if (!a4q) {
                cudaFuncSetAttribute(
                    (const void*)pd_attn_prefill_f16_v4_kernel<256u, 4u, V4TK>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    (int)v4q_smem(64u, 16u, false));
                cudaFuncSetAttribute(
                    (const void*)pd_attn_prefill_f16_v4_kernel<256u, 6u, V4TK>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    (int)v4q_smem(48u, 8u, false));
                cudaFuncSetAttribute(
                    (const void*)pd_attn_prefill_f16_v4_kernel<256u, 8u, V4TK>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    (int)v4q_smem(64u, 8u, false));
                cudaFuncSetAttribute(
                    (const void*)pd_attn_prefill_f16_v4_kernel<256u, 4u, V4TK, __nv_fp8_e4m3>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    (int)v4q_smem(64u, 16u, true));
                cudaFuncSetAttribute(
                    (const void*)pd_attn_prefill_f16_v4_kernel<256u, 6u, V4TK, __nv_fp8_e4m3>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    (int)v4q_smem(48u, 8u, true));
                cudaFuncSetAttribute(
                    (const void*)pd_attn_prefill_f16_v4_kernel<256u, 8u, V4TK, __nv_fp8_e4m3>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    (int)v4q_smem(64u, 8u, true));
                a4q = true;
            }
            dim3 g4q(n_kv_heads, (batch + v4tq - 1u) / v4tq);
#define PD_PFV4Q_LAUNCH(GV, KVT_)                                              \
    pd_attn_prefill_f16_v4_kernel<256u, GV, V4TK, KVT_>                        \
        <<<g4q, 256, smem, (cudaStream_t)stream>>>(                            \
            (const float*)q, (const KVT_*)pool_k, (const KVT_*)pool_v,         \
            (const float*)sinks, (float*)out, (const unsigned int*)positions,  \
            (const uint32_t*)block_tables, blocks_per_slot,                    \
            (const unsigned int*)slots, n_heads, kv_dim, swa_window, batch,    \
            scale)
            if (f8v4) {
                if (g_ == 4u) PD_PFV4Q_LAUNCH(4u, __nv_fp8_e4m3);
                else if (g_ == 6u) PD_PFV4Q_LAUNCH(6u, __nv_fp8_e4m3);
                else PD_PFV4Q_LAUNCH(8u, __nv_fp8_e4m3);
            } else {
                if (g_ == 4u) PD_PFV4Q_LAUNCH(4u, __half);
                else if (g_ == 6u) PD_PFV4Q_LAUNCH(6u, __half);
                else PD_PFV4Q_LAUNCH(8u, __half);
            }
#undef PD_PFV4Q_LAUNCH
            return pd_launch_status();
        }
        }
        // e4m3 must never fall into the f16-fragment ladder below (they read
        // half pools); the scalar paged tile is the fp8 terminal here
        if (f8v4)
            return pd_attn_prefill_paged_launch<__nv_fp8_e4m3, 256u>(q, pool_k,
                pool_v, sinks, out, positions, slots, block_tables,
                blocks_per_slot, n_heads, n_kv_heads, kv_dim, swa_window,
                batch, scale, (cudaStream_t)stream);
    }
    // v4 staged-HMMA tile for the granite/laguna/muse/paddleocr hd128 shapes:
    // G=4 (granite 32q/8kv), G=6 (laguna full-attn layers, 48q/8kv), G=8
    // (paddleocr-vl's ERNIE decoder, 16q/2kv - ; MR=64/TQ=8, the
    // exact geometry the hd64 gpt-oss G=8 arm already runs), G=9 (laguna
    // SWA layers, 72q/8kv) and G=16 (muse-glimmer, 32q/2kv - the default MR=64
    // already fits it, 64 % 16 == 0, so it costs one instantiation and buys
    // muse its fp8-KV lane; TQ falls to 4 rows/CTA, which is the thinnest in
    // the family - a perf question, not a correctness
    // one - no existing tile covered these ratios at any head_dim;
    // MR must be a multiple of 16 AND G, so G=9 forces MR=144, the largest
    // o_acc register footprint in the whole v4 family - verify occupancy
    // before trusting this arm at scale, same discipline as every other
    // tile here). Mirrors the hd256 qwen35/qwen3.6 block immediately above
    // exactly (same fp8-in-kernel PIPE convert, same smem-cap-checked
    // fallback to the scalar paged tile) - see that block's comment for the
    // shared rationale. Kill: PADDOCK_NO_PF_V4 (same env, shared family).
    if (head_dim == 128u
        && (n_heads == 4u * n_kv_heads || n_heads == 6u * n_kv_heads
            || n_heads == 8u * n_kv_heads || n_heads == 9u * n_kv_heads
            || n_heads == 16u * n_kv_heads)) {
        const bool f8v4n = kv_dtype == PD_KV_FP8_E4M3;
        const uint32_t g_ = n_heads / n_kv_heads;
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
        // shuffles per KV tile per CTA). Two q-tile-WIDTH arms were
        // falsified first (NW=8 +30%, SUB=2 +15% at FlashInfer's exact launch
        // config) -- the tile was never the lever; the per-fragment convert
        // is. pf7rp already was this structure at hd256 (door 2,
        // bit-identical there, -5..-21%); this is the hd128 generalization.
        // TK=48 not 64: at TK=64 the tile is 51,456 B and misses 2 CTA/SM by
        // 256 BYTES -> 4 warps/SM, and NW=8 already measured what 1 CTA/SM
        // costs this pipeline. Kill: PADDOCK_NO_PF7RP -> the pf7 arm below.
        // G in {4,6,8,9,16}: the first wiring instantiated only
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
        if (f8v4n && !no_rp128 && (g_ == 4u || g_ == 6u || g_ == 8u
                                   || g_ == 9u || g_ == 16u)) {
            constexpr uint32_t RPMR = 64u, RPTK = 48u, RPHD = 128u;
            // Q MR*(HD+8)h | pane TK*(HD+8)h | raw K,V TK*HD B each |
            // rpos MR uints. Every term carries its own shape factor.
            constexpr uint32_t RPSM =
                RPMR * (RPHD + 8u) * 2u + RPTK * (RPHD + 8u) * 2u
                + 2u * RPTK * RPHD + RPMR * 4u;             // 43,008 -> 2 CTA/SM
            static int rpc = -1;
            if (rpc < 0) {
                int dev = 0; cudaGetDevice(&dev);
                if (cudaDeviceGetAttribute(&rpc,
                        cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess)
                    rpc = 48 * 1024;
            }
            if (RPSM <= (uint32_t)rpc) {
                static bool arp128 = false;
                if (!arp128) {
#define PD_PF7RP_128_SMEM(GV)                                                  \
    cudaFuncSetAttribute((const void*)pd_attn_prefill_pf7rp_kernel<GV, false,  \
                                                                   RPHD, RPTK>, \
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)RPSM)
                    PD_PF7RP_128_SMEM(4u);
                    PD_PF7RP_128_SMEM(6u);
                    PD_PF7RP_128_SMEM(8u);
                    PD_PF7RP_128_SMEM(9u);
                    PD_PF7RP_128_SMEM(16u);
#undef PD_PF7RP_128_SMEM
                    arp128 = true;
                }
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
        nullptr, ro)
                if (g_ == 4u) PD_PF7RP_128_LAUNCH(4u);
                else if (g_ == 6u) PD_PF7RP_128_LAUNCH(6u);
                else if (g_ == 8u) PD_PF7RP_128_LAUNCH(8u);
                else if (g_ == 9u) PD_PF7RP_128_LAUNCH(9u);
                else PD_PF7RP_128_LAUNCH(16u);
#undef PD_PF7RP_128_LAUNCH
                return pd_launch_status();
            }
        }
        // hd128 convert-in-register arm: the pf7 raw-fp8-resident
        // FA2 tile generalized from hd256 to hd128 (granite imax). Raw e4m3 KV
        // stays in smem (no f16 expand pass, no barrier), TK=64, ~44 KB smem
        // => 2 CTA/SM vs the v4 fp8 arm's 1. Handles both single-run and the
        // pf_runs multi-prompt burst (run_offs arm, grid.z = n_runs, per-run
        // slot). G in {4,6,8} (granite is 4). Kill: PADDOCK_NO_PF7 -> v4.
        static const bool no_pf7_128 = pd_env("PADDOCK_NO_PF7") != nullptr;
        // Same G set as the pf7rp arm above, so PADDOCK_NO_PF7RP lands on
        // pf7 rather than falling all the way to v4 for ratios 9/16.
        if (f8v4n && !no_pf7_128 && (g_ == 4u || g_ == 6u || g_ == 8u
                                     || g_ == 9u || g_ == 16u)) {
            // Q-TILE WIDTH is not the LEVER (two arms, both falsified).
            // Profiling had pf7 at grid 1024 where FlashInfer needs 512 for
            // identical work, with much higher L1/TEX, which read as K/V
            // staging redundancy from our half-width q tile. Widening it via warps (NW=8, MR=128, TK=64) cost 2
            // CTA/SM and ran 30% slower; widening it via registers
            // (SUB=2, TK=32) reproduced FlashInfer's launch config to the byte
            // - grid 512, 128 threads, 49152 B, 2 CTA/SM, 1.36 waves, 236 regs,
            // no spill - and was still 15% slower, with L1/TEX barely moved
            // (60.3 -> 57.5%). Halving the q-tiles should have roughly halved
            // staging traffic; it didn't, so the L1 excess is not the
            // global->smem staging. It is the per-fragment smem->register
            // ldmatrix + pd_pf7_cvt2 fp8->f16 convert, which scales with the
            // MATH and is invariant to tile width. FlashInfer's 49152 B
            // decomposes as Q 128x128x2 + K/V 32x128x2: bf16 KV in smem,
            // converted once at staging. That - not the tile - is the open
            // rung. The kernel keeps its NW/SUB/TKT template axes (SUB=1 is
            // bit-exact with the pre-refactor kernel, digests verified) so the
            // next rung has the scaffolding; only the narrow arm instantiates.
            // Keep the MR factor visible in every term: sh_q MR*(HD+8)
            // halves | K double-buffered + V, TK*(HD+16) bytes each |
            // sh_rpos MR uints. The first wide-tile build hand-carried the
            // MR=64 tail (256 B) into an MR=128 constant and took an illegal
            // access on the 64 rows past the end of sh_rpos.
            constexpr uint32_t P7MR = 64u, P7TK = 64u;
            constexpr uint32_t P7SM128 =
                P7MR * 136u * 2u + 3u * P7TK * 144u + P7MR * 4u;  // 45312, 2 CTA/SM
            static int p7c128 = -1;
            if (p7c128 < 0) {
                int dev = 0; cudaGetDevice(&dev);
                if (cudaDeviceGetAttribute(&p7c128,
                        cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess)
                    p7c128 = 48 * 1024;
            }
            if (P7SM128 <= (uint32_t)p7c128) {
                static bool a7_128 = false;
                if (!a7_128) {
                    cudaFuncSetAttribute((const void*)pd_attn_prefill_pf7_kernel<128u, 4u>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)P7SM128);
                    cudaFuncSetAttribute((const void*)pd_attn_prefill_pf7_kernel<128u, 6u>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)P7SM128);
                    cudaFuncSetAttribute((const void*)pd_attn_prefill_pf7_kernel<128u, 8u>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)P7SM128);
                    cudaFuncSetAttribute((const void*)pd_attn_prefill_pf7_kernel<128u, 9u>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)P7SM128);
                    cudaFuncSetAttribute((const void*)pd_attn_prefill_pf7_kernel<128u, 16u>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)P7SM128);
                    a7_128 = true;
                }
                const bool multi = pd_pf_runs_offs != nullptr;
                const dim3 g7 = multi
                    ? dim3(n_kv_heads, (pd_pf_runs_maxn * g_ + 63u) / 64u, pd_pf_runs_n)
                    : dim3(n_kv_heads, (batch * g_ + 63u) / 64u);
                const uint32_t* ro = multi ? (const uint32_t*)pd_pf_runs_offs : nullptr;
#define PD_PF7_128_LAUNCH(GV)                                                  \
    pd_attn_prefill_pf7_kernel<128u, GV><<<g7, 128, P7SM128, (cudaStream_t)stream>>>( \
        (const float*)q, (const unsigned char*)pool_k, (const unsigned char*)pool_v, \
        (const float*)sinks, (float*)out, (const unsigned int*)positions,      \
        (const uint32_t*)block_tables, blocks_per_slot, (const unsigned int*)slots, \
        n_heads, kv_dim, swa_window, batch, scale, nullptr, ro)
                if (g_ == 4u) PD_PF7_128_LAUNCH(4u);
                else if (g_ == 6u) PD_PF7_128_LAUNCH(6u);
                else if (g_ == 8u) PD_PF7_128_LAUNCH(8u);
                else if (g_ == 9u) PD_PF7_128_LAUNCH(9u);
                else PD_PF7_128_LAUNCH(16u);
#undef PD_PF7_128_LAUNCH
                return pd_launch_status();
            }
        }
        if (!no_v4s) {
        constexpr uint32_t V4TK = 16u;
        // Occupancy fix: v4 at MR=64 uses 51.6 KB smem => 1 CTA/SM
        // (8 warps, latency-bound => ~19 TF, far off the achievable rate).
        // Shrinking the
        // GQA-4 query tile to MR=32 drops smem to 38.3 KB => 2 CTA/SM (16
        // warps), the imax prefill-attention lever. TK=32 could not reach 2
        // CTA/SM at any MR. Other G ratios keep their tuned MR.
        const uint32_t v4mr = g_ == 6u ? 48u : (g_ == 9u ? 144u : 64u);
        const uint32_t v4tq = v4mr / g_;  // 16 rows/CTA at G=4/G=9, 8 at G=6/G=8
        const uint32_t rowe = 128u + 8u, ts4 = V4TK + 8u;
        // f8 arms add the raw e4m3 stage region (2 bufs x K,V x TK x 128B)
        auto v4n_smem = [&](uint32_t mr, uint32_t tq, bool f8) {
            return mr * rowe * 2u + 4u * V4TK * rowe * 2u
                + (f8 ? 4u * V4TK * 128u : 0u)
                + mr * ts4 * 4u + mr * ts4 * 2u + 2u * mr * 4u + tq * 4u;
        };
        const uint32_t smem = v4n_smem(v4mr, v4tq, f8v4n);
        static int v4n_cap = -1;
        if (v4n_cap < 0) {
            int dev = 0;
            cudaGetDevice(&dev);
            if (cudaDeviceGetAttribute(&v4n_cap,
                    cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess)
                v4n_cap = 48 * 1024;
        }
        if (smem <= (uint32_t)v4n_cap) {
            static bool a4n = false;
            if (!a4n) {
                cudaFuncSetAttribute(
                    (const void*)pd_attn_prefill_f16_v4_kernel<128u, 4u, V4TK>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    (int)v4n_smem(64u, 16u, false));
                cudaFuncSetAttribute(
                    (const void*)pd_attn_prefill_f16_v4_kernel<128u, 6u, V4TK>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    (int)v4n_smem(48u, 8u, false));
                cudaFuncSetAttribute(
                    (const void*)pd_attn_prefill_f16_v4_kernel<128u, 9u, V4TK>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    (int)v4n_smem(144u, 16u, false));
                cudaFuncSetAttribute(
                    (const void*)pd_attn_prefill_f16_v4_kernel<128u, 4u, V4TK, __nv_fp8_e4m3>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    (int)v4n_smem(64u, 16u, true));
                cudaFuncSetAttribute(
                    (const void*)pd_attn_prefill_f16_v4_kernel<128u, 6u, V4TK, __nv_fp8_e4m3>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    (int)v4n_smem(48u, 8u, true));
                cudaFuncSetAttribute(
                    (const void*)pd_attn_prefill_f16_v4_kernel<128u, 9u, V4TK, __nv_fp8_e4m3>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    (int)v4n_smem(144u, 16u, true));
                cudaFuncSetAttribute(
                    (const void*)pd_attn_prefill_f16_v4_kernel<128u, 16u, V4TK>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    (int)v4n_smem(64u, 4u, false));
                cudaFuncSetAttribute(
                    (const void*)pd_attn_prefill_f16_v4_kernel<128u, 16u, V4TK, __nv_fp8_e4m3>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    (int)v4n_smem(64u, 4u, true));
                cudaFuncSetAttribute(
                    (const void*)pd_attn_prefill_f16_v4_kernel<128u, 8u, V4TK>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    (int)v4n_smem(64u, 8u, false));
                cudaFuncSetAttribute(
                    (const void*)pd_attn_prefill_f16_v4_kernel<128u, 8u, V4TK, __nv_fp8_e4m3>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    (int)v4n_smem(64u, 8u, true));
                a4n = true;
            }
            dim3 g4n(n_kv_heads, (batch + v4tq - 1u) / v4tq);
            // batched-runs arm: the hd128 family is the muse
            // (G=16) election, whose engine-side PF_RUNS whole-chunk launch
            // otherwise lands here with 32 slots and no run table.
            const uint32_t* r4n = nullptr;
            if (pd_pf_runs_offs != nullptr) {
                g4n = dim3(n_kv_heads,
                           (pd_pf_runs_maxn + v4tq - 1u) / v4tq, pd_pf_runs_n);
                r4n = (const uint32_t*)pd_pf_runs_offs;
            }
#define PD_PFV4N_LAUNCH(GV, KVT_)                                              \
    pd_attn_prefill_f16_v4_kernel<128u, GV, V4TK, KVT_>                        \
        <<<g4n, 256, smem, (cudaStream_t)stream>>>(                            \
            (const float*)q, (const KVT_*)pool_k, (const KVT_*)pool_v,         \
            (const float*)sinks, (float*)out, (const unsigned int*)positions,  \
            (const uint32_t*)block_tables, blocks_per_slot,                    \
            (const unsigned int*)slots, n_heads, kv_dim, swa_window, batch,    \
            scale, r4n)
            if (f8v4n) {
                if (g_ == 4u) PD_PFV4N_LAUNCH(4u, __nv_fp8_e4m3);
                else if (g_ == 6u) PD_PFV4N_LAUNCH(6u, __nv_fp8_e4m3);
                else if (g_ == 8u) PD_PFV4N_LAUNCH(8u, __nv_fp8_e4m3);
                else if (g_ == 9u) PD_PFV4N_LAUNCH(9u, __nv_fp8_e4m3);
                else PD_PFV4N_LAUNCH(16u, __nv_fp8_e4m3);
            } else {
                if (g_ == 4u) PD_PFV4N_LAUNCH(4u, __half);
                else if (g_ == 6u) PD_PFV4N_LAUNCH(6u, __half);
                else if (g_ == 8u) PD_PFV4N_LAUNCH(8u, __half);
                else if (g_ == 9u) PD_PFV4N_LAUNCH(9u, __half);
                else PD_PFV4N_LAUNCH(16u, __half);
            }
#undef PD_PFV4N_LAUNCH
            return pd_launch_status();
        }
        }
        // e4m3 must never fall into the f16-fragment ladder below (they read
        // half pools); the scalar paged tile is the fp8 terminal here
        if (f8v4n)
            return pd_attn_prefill_paged_launch<__nv_fp8_e4m3, 128u>(q, pool_k,
                pool_v, sinks, out, positions, slots, block_tables,
                blocks_per_slot, n_heads, n_kv_heads, kv_dim, swa_window,
                batch, scale, (cudaStream_t)stream);
    }
    // v4 staged-HMMA tile, FP8 ARM only, for gpt-oss's shape (G=8, hd64).
    // This family's own hd64 kernel (pd_attn_prefill_f16_paged_kernel<64>,
    // dispatched further below) never grew an e4m3 arm: requesting
    // fp8-e4m3 KV for gpt-oss silently downgraded every prefill to the
    // scalar split/combine tile (correct output, no tensor cores - the
    // hd128/hd256 siblings above got this pass, gpt-oss never did).
    // Deliberately narrower than those siblings: this gate takes only the
    // fp8 case, leaving gpt-oss's f16 traffic on its own already-tuned hd64
    // kernel untouched below - no A/B yet shows the v4 tile beats it at
    // this shape, and that's a separate question from closing the fp8 gap.
    // gpt_oss.rs's `pf16_attn` gate only ever calls this function with
    // fp8-e4m3 when also paged (the non-paged `pd_attn_prefill_f16`
    // sibling has no v4/fp8 support and hard-rejects fp8 at the top of this
    // file), so this arm only needs to exist here. Kill: PADDOCK_NO_PF_V4
    // (shared with the other v4 arms).
    if (head_dim == 64u && n_heads == 8u * n_kv_heads
        && kv_dtype == PD_KV_FP8_E4M3) {
        if (!no_v4s) {
        constexpr uint32_t V4TK = 16u;
        constexpr uint32_t v4gmr = 64u;              // G != 6,9 -> MR=64
        constexpr uint32_t v4gtq = v4gmr / 8u;        // 8 rows/CTA
        constexpr uint32_t rowe = 64u + 8u, ts4 = V4TK + 8u;
        // fixed geometry (HD/G/TK are all compile-time for this one gpt-oss
        // shape) -> smem is a compile-time constant, not a per-call value;
        // ~31.5 KB here, comfortably under even the 48 KB static default,
        // let alone the opt-in cap - the runtime check below is kept for
        // consistency with the hd128/hd256 arms' pattern, not because this
        // shape is expected to ever fail it.
        constexpr uint32_t smem = v4gmr * rowe * 2u + 4u * V4TK * rowe * 2u
            + 4u * V4TK * 64u + v4gmr * ts4 * 4u + v4gmr * ts4 * 2u
            + 2u * v4gmr * 4u + v4gtq * 4u;
        static int v4g_cap = -1;
        if (v4g_cap < 0) {
            int dev = 0;
            cudaGetDevice(&dev);
            if (cudaDeviceGetAttribute(&v4g_cap,
                    cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess)
                v4g_cap = 48 * 1024;
        }
        if (smem <= (uint32_t)v4g_cap) {
            static bool a4g = false;
            if (!a4g) {
                cudaFuncSetAttribute(
                    (const void*)pd_attn_prefill_f16_v4_kernel<64u, 8u, V4TK, __nv_fp8_e4m3>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
                a4g = true;
            }
            dim3 g4g(n_kv_heads, (batch + v4gtq - 1u) / v4gtq);
            pd_attn_prefill_f16_v4_kernel<64u, 8u, V4TK, __nv_fp8_e4m3>
                <<<g4g, 256, smem, (cudaStream_t)stream>>>(
                    (const float*)q, (const __nv_fp8_e4m3*)pool_k,
                    (const __nv_fp8_e4m3*)pool_v, (const float*)sinks,
                    (float*)out, (const unsigned int*)positions,
                    (const uint32_t*)block_tables, blocks_per_slot,
                    (const unsigned int*)slots, n_heads, kv_dim, swa_window,
                    batch, scale);
            return pd_launch_status();
        }
        }
        // no_v4s, or (never expected to happen) this device's opt-in cap is
        // under the fixed ~31.5 KB this shape needs: an honest failure, not
        // a silent fall-through onto the f16-only kernel below (it would
        // read these e4m3 bytes as __half - real corruption, not a
        // graceful degrade). No hd64 scalar-paged-tile instantiation exists
        // to fall back to (pd_attn_prefill_paged_launch's own top-level
        // export rejects head_dim 64 outright) - gpt_oss.rs's own
        // n_splits>1 / attn_gqa_fused scalar path is the honest fallback,
        // reached by not setting pf16_attn in the first place; this
        // function is never called for fp8+hd64 unless pf16_attn already
        // decided it should be.
        return cudaErrorInvalidValue;
    }
    static const int pf_fa = [] {
        const char* e = pd_env("PADDOCK_G4_PF_FA");
        return e ? atoi(e) : 0;
    }();
    if (pf_fa && head_dim == 256u && kv_dtype != PD_KV_FP8_E4M3
        && (n_heads % n_kv_heads) == 0u && n_heads == 2u * n_kv_heads) {
        constexpr uint32_t PFA_TPW = 2u;
        // "2" = co-residency variant: 128 threads, M=32 (k1=16), PT16/SB -
        // ~37.9KB/CTA so two CTAs interleave per SM (the WMMA tile's hidden
        // latency-hiding advantage; the 256-thread variants run 1 CTA/SM
        // with ~30% barrier-stall from the phase-serial chain)
        const bool cores = pf_fa == 2;
        const bool spl = pf_fa == 3;  // split-pipelined SB (PT32)
        const uint32_t nt = cores ? 128u : 256u;
        const uint32_t PFA_K1 = cores ? 16u : 32u;
        const uint32_t pt = spl ? 32u
            : (pf_fa == 24 || pf_fa == 32 || pf_fa == 40 || pf_fa == 48)
            ? (uint32_t)pf_fa : 16u;
        const bool db = !cores && pt <= 24u;
        const uint32_t Mp = PFA_K1 * 2u;  // G=2, already 16-aligned
        // padded strides - must mirror the kernel's KP/PP/FP exactly
        const uint32_t smem = Mp * (head_dim + 8u) * 2u
            + (db ? 4u : 2u) * pt * (head_dim + 8u) * 2u  // KV ring
            + Mp * (pt + 1u) * 4u + 3u * Mp * 4u + Mp * (pt + 8u) * 2u;
        static int pfa_ok = -1;
        if (pfa_ok < 0) {
            int dev = 0, cap = 0;
            cudaGetDevice(&dev);
            cudaDeviceGetAttribute(&cap, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
            pfa_ok = (int)smem <= cap ? 1 : 0;
            if (pfa_ok) {
                const void* kf = spl
                    ? (const void*)pd_attn_prefill_fa_kernel<32u, PFA_TPW, false, 256u, 256u, 1u, true>
                    : cores
                    ? (const void*)pd_attn_prefill_fa_kernel<16u, PFA_TPW, false, 128u>
                    : pt == 24u
                    ? (const void*)pd_attn_prefill_fa_kernel<24u, PFA_TPW, true>
                    : pt == 32u
                    ? (const void*)pd_attn_prefill_fa_kernel<32u, PFA_TPW, false>
                    : pt == 40u
                    ? (const void*)pd_attn_prefill_fa_kernel<40u, PFA_TPW, false>
                    : pt == 48u
                    ? (const void*)pd_attn_prefill_fa_kernel<48u, PFA_TPW, false>
                    : (const void*)pd_attn_prefill_fa_kernel<16u, PFA_TPW, true>;
                cudaFuncSetAttribute(kf,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
            }
        }
        if (pfa_ok) {
            dim3 gf(n_kv_heads, (batch + PFA_K1 - 1u) / PFA_K1);
            if (spl)
                pd_attn_prefill_fa_kernel<32u, PFA_TPW, false, 256u, 256u, 1u, true><<<gf, nt, smem, (cudaStream_t)stream>>>(
                    (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                    (const float*)sinks, (float*)out, (const unsigned int*)positions,
                    (const unsigned int*)slots, (const uint32_t*)block_tables,
                    blocks_per_slot, n_heads, n_kv_heads, head_dim, kv_dim,
                    swa_window, batch, PFA_K1, scale);
            else if (cores)
                pd_attn_prefill_fa_kernel<16u, PFA_TPW, false, 128u><<<gf, nt, smem, (cudaStream_t)stream>>>(
                    (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                    (const float*)sinks, (float*)out, (const unsigned int*)positions,
                    (const unsigned int*)slots, (const uint32_t*)block_tables,
                    blocks_per_slot, n_heads, n_kv_heads, head_dim, kv_dim,
                    swa_window, batch, PFA_K1, scale);
            else if (pt == 24u)
                pd_attn_prefill_fa_kernel<24u, PFA_TPW, true><<<gf, nt, smem, (cudaStream_t)stream>>>(
                    (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                    (const float*)sinks, (float*)out, (const unsigned int*)positions,
                    (const unsigned int*)slots, (const uint32_t*)block_tables,
                    blocks_per_slot, n_heads, n_kv_heads, head_dim, kv_dim,
                    swa_window, batch, PFA_K1, scale);
            else if (pt == 40u)
                pd_attn_prefill_fa_kernel<40u, PFA_TPW, false><<<gf, nt, smem, (cudaStream_t)stream>>>(
                    (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                    (const float*)sinks, (float*)out, (const unsigned int*)positions,
                    (const unsigned int*)slots, (const uint32_t*)block_tables,
                    blocks_per_slot, n_heads, n_kv_heads, head_dim, kv_dim,
                    swa_window, batch, PFA_K1, scale);
            else if (pt == 32u)
                pd_attn_prefill_fa_kernel<32u, PFA_TPW, false><<<gf, 256, smem, (cudaStream_t)stream>>>(
                    (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                    (const float*)sinks, (float*)out, (const unsigned int*)positions,
                    (const unsigned int*)slots, (const uint32_t*)block_tables,
                    blocks_per_slot, n_heads, n_kv_heads, head_dim, kv_dim,
                    swa_window, batch, PFA_K1, scale);
            else if (pt == 48u)
                pd_attn_prefill_fa_kernel<48u, PFA_TPW, false><<<gf, 256, smem, (cudaStream_t)stream>>>(
                    (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                    (const float*)sinks, (float*)out, (const unsigned int*)positions,
                    (const unsigned int*)slots, (const uint32_t*)block_tables,
                    blocks_per_slot, n_heads, n_kv_heads, head_dim, kv_dim,
                    swa_window, batch, PFA_K1, scale);
            else
                pd_attn_prefill_fa_kernel<16u, PFA_TPW, true><<<gf, 256, smem, (cudaStream_t)stream>>>(
                    (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                    (const float*)sinks, (float*)out, (const unsigned int*)positions,
                    (const unsigned int*)slots, (const uint32_t*)block_tables,
                    blocks_per_slot, n_heads, n_kv_heads, head_dim, kv_dim,
                    swa_window, batch, PFA_K1, scale);
            return pd_launch_status();
        }
    }
    // v3s: gemma4's SWA geometry (hd256, group 2, n_kv%4==0). MANDATORY for
    // fp8 caches (nothing else can read them); f16 opt-in via env for A/B -
    // the WMMA tile stays the measured f16 default.
    static const bool v3s_f16 = pd_env("PADDOCK_G4_SWA_V3S") != nullptr;
    if (head_dim == 256u && n_heads == 2u * n_kv_heads && (n_kv_heads & 3u) == 0u
        && (kv_dtype == PD_KV_FP8_E4M3 || v3s_f16)) {
        // v3c: the probed tile optimum of this class (TK=64/NR=64, K+V
        // co-staged; measured -35% at the churn shape). Default
        // for the geometry; PADDOCK_NO_PF_V3C reverts to v3s.
        static const bool v3c = pd_env("PADDOCK_NO_PF_V3C") == nullptr;
        if (v3c) {
            static bool a3c = false;
            if (!a3c) {
                cudaFuncSetAttribute((const void*)pd_attn_prefill_f16_v3c_kernel<__half>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, PD_AF3C_SMEM);
                // fp8 carries the Phase-75 raw cp.async stage region
                cudaFuncSetAttribute((const void*)pd_attn_prefill_f16_v3c_kernel<__nv_fp8_e4m3>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, PD_AF3C_SMEM_P8);
                a3c = true;
            }
            dim3 gc(n_kv_heads, (batch + PD_AF3C_NR - 1u) / PD_AF3C_NR);
            if (kv_dtype == PD_KV_FP8_E4M3)
                pd_attn_prefill_f16_v3c_kernel<__nv_fp8_e4m3><<<gc, 256, PD_AF3C_SMEM_P8, (cudaStream_t)stream>>>(
                    (const float*)q, (const __nv_fp8_e4m3*)pool_k, (const __nv_fp8_e4m3*)pool_v,
                    (const float*)sinks, (float*)out, (const unsigned int*)positions,
                    (const unsigned int*)slots, (const uint32_t*)block_tables, blocks_per_slot,
                    n_heads, n_kv_heads, 0u, kv_dim, swa_window, batch, scale);
            else
                pd_attn_prefill_f16_v3c_kernel<__half><<<gc, 256, PD_AF3C_SMEM, (cudaStream_t)stream>>>(
                    (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                    (const float*)sinks, (float*)out, (const unsigned int*)positions,
                    (const unsigned int*)slots, (const uint32_t*)block_tables, blocks_per_slot,
                    n_heads, n_kv_heads, 0u, kv_dim, swa_window, batch, scale);
            return pd_launch_status();
        }
        static bool a3s = false;
        if (!a3s) {
            cudaFuncSetAttribute((const void*)pd_attn_prefill_f16_v3s_kernel<__half>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, PD_AF3S_SMEM);
            cudaFuncSetAttribute((const void*)pd_attn_prefill_f16_v3s_kernel<__nv_fp8_e4m3>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, PD_AF3S_SMEM);
            a3s = true;
        }
        dim3 gs(n_kv_heads / 4u, (batch + PD_AF3S_NR - 1u) / PD_AF3S_NR);
        if (kv_dtype == PD_KV_FP8_E4M3)
            pd_attn_prefill_f16_v3s_kernel<__nv_fp8_e4m3><<<gs, 256, PD_AF3S_SMEM, (cudaStream_t)stream>>>(
                (const float*)q, (const __nv_fp8_e4m3*)pool_k, (const __nv_fp8_e4m3*)pool_v,
                (const float*)sinks, (float*)out, (const unsigned int*)positions,
                (const unsigned int*)slots, (const uint32_t*)block_tables, blocks_per_slot,
                n_heads, n_kv_heads, 0u, kv_dim, swa_window, batch, scale);
        else
            pd_attn_prefill_f16_v3s_kernel<__half><<<gs, 256, PD_AF3S_SMEM, (cudaStream_t)stream>>>(
                (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                (const float*)sinks, (float*)out, (const unsigned int*)positions,
                (const unsigned int*)slots, (const uint32_t*)block_tables, blocks_per_slot,
                n_heads, n_kv_heads, 0u, kv_dim, swa_window, batch, scale);
        return pd_launch_status();
    }
    static const bool v3p = pd_env("PADDOCK_ATTN_PF_V3") != nullptr;
    if (v3p && head_dim == 256u && n_heads == 8u * n_kv_heads && batch > 0) {
        static bool a3p = false;
        if (!a3p) {
            cudaFuncSetAttribute((const void*)pd_attn_prefill_f16_v3_kernel<256u>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, PD_AF3_SMEM);
            a3p = true;
        }
        dim3 g3(n_kv_heads, (batch + PD_AF3_NR - 1u) / PD_AF3_NR);
        pd_attn_prefill_f16_v3_kernel<256u><<<g3, 256, PD_AF3_SMEM, (cudaStream_t)stream>>>(
            (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
            (const float*)sinks, (float*)out, (const unsigned int*)positions,
            (const unsigned int*)slots, (const uint32_t*)block_tables, blocks_per_slot,
            n_heads, n_kv_heads, 0u, kv_dim, swa_window, batch, scale);
        return pd_launch_status();
    }
    static const bool v2p = pd_env("PADDOCK_ATTN_PF_V2") != nullptr;
    if (v2p && (head_dim == 256u || head_dim == 64u)) {  // v2 arms are 256/64-only
        if (head_dim == 256u)
            pd_attn_prefill_f16_v2_kernel<256u><<<grid, 128, 0, (cudaStream_t)stream>>>(
                (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                (const float*)sinks, (float*)out, (const unsigned int*)positions,
                (const unsigned int*)slots, (const uint32_t*)block_tables, blocks_per_slot,
                n_heads, n_kv_heads, 0u, kv_dim, swa_window, batch, scale);
        else
            pd_attn_prefill_f16_v2_kernel<64u><<<grid, 128, 0, (cudaStream_t)stream>>>(
                (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                (const float*)sinks, (float*)out, (const unsigned int*)positions,
                (const unsigned int*)slots, (const uint32_t*)block_tables, blocks_per_slot,
                n_heads, n_kv_heads, 0u, kv_dim, swa_window, batch, scale);
        return pd_launch_status();
    }
    if (head_dim == 256u) {
        pd_attn_prefill_f16_paged_kernel<256u><<<grid, 128, 0, (cudaStream_t)stream>>>(
            (const float*)q, (const __half*)pool_k, (const __half*)pool_v, (const float*)sinks,
            (float*)out, (const unsigned int*)positions, (const unsigned int*)slots,
            (const uint32_t*)block_tables, blocks_per_slot, n_heads, n_kv_heads, kv_dim,
            swa_window, batch, scale);
    } else if (head_dim == 128u) {
        // hd 128 (laguna XS / the qwen3 head shape): same NC=32 tile as 256/64,
        // ~26 KB static smem - instantiation was simply never needed before
        pd_attn_prefill_f16_paged_kernel<128u><<<grid, 128, 0, (cudaStream_t)stream>>>(
            (const float*)q, (const __half*)pool_k, (const __half*)pool_v, (const float*)sinks,
            (float*)out, (const unsigned int*)positions, (const unsigned int*)slots,
            (const uint32_t*)block_tables, blocks_per_slot, n_heads, n_kv_heads, kv_dim,
            swa_window, batch, scale);
    } else {
        pd_attn_prefill_f16_paged_kernel<64u><<<grid, 128, 0, (cudaStream_t)stream>>>(
            (const float*)q, (const __half*)pool_k, (const __half*)pool_v, (const float*)sinks,
            (float*)out, (const unsigned int*)positions, (const unsigned int*)slots,
            (const uint32_t*)block_tables, blocks_per_slot, n_heads, n_kv_heads, kv_dim,
            swa_window, batch, scale);
    }
    return pd_launch_status();
}

// a16 twin (attention streams): q and out are f16 planes. Own
// dispatch, not a flag on the big election above - only the v3c/v3s/v3w
// arms have TQ/TO forms, and any geometry that would fall to another
// kernel must be a hard error (the plane dtype is decided before launch;
// there is no per-call fallback). The q side is bit-equal at scale=1.0
// (those kernels round q to f16 into fragments anyway); the out side
// rounds once at the store - serve acceptance arbitrates.
PD_EXPORT
int pd_attn_prefill_f16_paged2(const void* q, const void* pool_k, const void* pool_v,
                               const void* sinks, void* out, const void* positions,
                               const void* slots, const void* block_tables,
                               uint32_t blocks_per_slot, uint32_t n_heads, uint32_t n_kv_heads,
                               uint32_t head_dim, uint32_t kv_dim, uint32_t swa_window,
                               uint32_t batch, float scale, uint32_t kv_dtype, uint32_t a16,
                               void* stream) {
    if (!a16)
        return pd_attn_prefill_f16_paged(q, pool_k, pool_v, sinks, out, positions,
                                         slots, block_tables, blocks_per_slot,
                                         n_heads, n_kv_heads, head_dim, kv_dim,
                                         swa_window, batch, scale, kv_dtype, stream);
    if (n_heads == 0 || batch == 0) return 0;
    if (head_dim == 512u && n_heads == 8u * n_kv_heads) {
        static bool a3w16 = false;
        if (!a3w16) {
            cudaFuncSetAttribute(
                (const void*)pd_attn_prefill_f16_v3w_kernel<512u, __half, __half, __half>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, PD_AF3W_SMEM);
            cudaFuncSetAttribute(
                (const void*)pd_attn_prefill_f16_v3w_kernel<512u, __nv_fp8_e4m3, __half, __half>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, PD_AF3W_SMEM);
            a3w16 = true;
        }
        dim3 gw(n_kv_heads, (batch + PD_AF3W_NR - 1u) / PD_AF3W_NR, 2u);
        if (kv_dtype == PD_KV_FP8_E4M3)
            pd_attn_prefill_f16_v3w_kernel<512u, __nv_fp8_e4m3, __half, __half>
                <<<gw, 256, PD_AF3W_SMEM, (cudaStream_t)stream>>>(
                (const __half*)q, (const __nv_fp8_e4m3*)pool_k, (const __nv_fp8_e4m3*)pool_v,
                (const float*)sinks, (__half*)out, (const unsigned int*)positions,
                (const unsigned int*)slots, (const uint32_t*)block_tables, blocks_per_slot,
                n_heads, n_kv_heads, 0u, kv_dim, swa_window, batch, scale);
        else
            pd_attn_prefill_f16_v3w_kernel<512u, __half, __half, __half>
                <<<gw, 256, PD_AF3W_SMEM, (cudaStream_t)stream>>>(
                (const __half*)q, (const __half*)pool_k, (const __half*)pool_v,
                (const float*)sinks, (__half*)out, (const unsigned int*)positions,
                (const unsigned int*)slots, (const uint32_t*)block_tables, blocks_per_slot,
                n_heads, n_kv_heads, 0u, kv_dim, swa_window, batch, scale);
        return pd_launch_status();
    }
    if (head_dim == 256u && n_heads == 2u * n_kv_heads && (n_kv_heads & 3u) == 0u
        && kv_dtype == PD_KV_FP8_E4M3) {
        static const bool v3c16 = pd_env("PADDOCK_NO_PF_V3C") == nullptr;
        if (v3c16) {
            static bool a3c16 = false;
            if (!a3c16) {
                cudaFuncSetAttribute(
                    (const void*)pd_attn_prefill_f16_v3c_kernel<__nv_fp8_e4m3, __half, __half>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, PD_AF3C_SMEM_P8);
                a3c16 = true;
            }
            dim3 gc(n_kv_heads, (batch + PD_AF3C_NR - 1u) / PD_AF3C_NR);
            pd_attn_prefill_f16_v3c_kernel<__nv_fp8_e4m3, __half, __half>
                <<<gc, 256, PD_AF3C_SMEM_P8, (cudaStream_t)stream>>>(
                (const __half*)q, (const __nv_fp8_e4m3*)pool_k, (const __nv_fp8_e4m3*)pool_v,
                (const float*)sinks, (__half*)out, (const unsigned int*)positions,
                (const unsigned int*)slots, (const uint32_t*)block_tables, blocks_per_slot,
                n_heads, n_kv_heads, 0u, kv_dim, swa_window, batch, scale);
            return pd_launch_status();
        }
        static bool a3s16 = false;
        if (!a3s16) {
            cudaFuncSetAttribute(
                (const void*)pd_attn_prefill_f16_v3s_kernel<__nv_fp8_e4m3, __half, __half>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, PD_AF3S_SMEM);
            a3s16 = true;
        }
        dim3 gs(n_kv_heads / 4u, (batch + PD_AF3S_NR - 1u) / PD_AF3S_NR);
        pd_attn_prefill_f16_v3s_kernel<__nv_fp8_e4m3, __half, __half>
            <<<gs, 256, PD_AF3S_SMEM, (cudaStream_t)stream>>>(
            (const __half*)q, (const __nv_fp8_e4m3*)pool_k, (const __nv_fp8_e4m3*)pool_v,
            (const float*)sinks, (__half*)out, (const unsigned int*)positions,
            (const unsigned int*)slots, (const uint32_t*)block_tables, blocks_per_slot,
            n_heads, n_kv_heads, 0u, kv_dim, swa_window, batch, scale);
        return pd_launch_status();
    }
    return cudaErrorInvalidValue;
}

// --------------------------------- attn prefill batch f16 (multi-slot WMMA)
// pd_attn_prefill_f16's tensor-core structure applied to the encoder's
// ragged many-text batches through the same per-text tiling contract as
// pd_attn_prefill_batch: the host emits 32-query tiles that never cross a
// text; rows spilled past a text are masked by slots[b] == slot. This is the
// hot encoder attention (the scalar f32 batch kernel ran at ~11% of f32 peak
// and was 37% of the whole reranker forward at 58-row suffixes x 127-key
// texts). Only head_dim 128 is instantiated - every Qwen3 encoder size.
// Numeric class: f16 Q/K/V inputs, f32 score accumulate + softmax, f16 O
// accumulate - pd_attn_prefill_f16's class, not bit-identical to the scalar
// batch kernel; the load-time calibration gate re-validates encoder quality
// (its Q8_0 baseline runs the same kernel, so both sides of the gate move
// together).
// Requirements: head_dim == 128, fp16 KV, max_ctx % 64 == 0 (K/V fragment
// loads touch keys up to t0+63; hi <= max_ctx and t0 stepping by 64 keep
// them inside the slot's cache rows - same bound trick as the single-slot
// kernel).
#define PD_ABF16_NC 32  // queries per tile (host tiling stride)
#define PD_ABF16_TK 64  // keys per tile (4 warps x 16 fragment rows)
template<uint32_t D>
__global__ void __launch_bounds__(128) pd_attn_prefill_batch_f16_kernel(
    const float* __restrict__ q, const __half* __restrict__ kc,
    const __half* __restrict__ vc, const float* __restrict__ sinks,
    float* __restrict__ out, const unsigned int* __restrict__ positions,
    const unsigned int* __restrict__ slots, const unsigned int* __restrict__ tile_row0,
    const unsigned int* __restrict__ tile_slot,
    uint32_t n_heads, uint32_t n_kv_heads, uint32_t max_ctx, uint32_t kv_dim,
    uint32_t swa_window, uint32_t n_rows, float scale) {
#if PD_MMA_OK
    using namespace nvcuda;
    constexpr uint32_t NC = PD_ABF16_NC, TK = PD_ABF16_TK;
    constexpr uint32_t RPW = NC / 4u;  // query rows per warp (matches f16 twins)
    constexpr uint32_t DW = D / 4u;    // dims per warp in the V GEMM
    constexpr uint32_t DP = D + 8u;    // half rows, conflict-avoid pad
    constexpr uint32_t KQP = TK + 8u;  // f32 score row stride
    typedef wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_k;
    typedef wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> frag_v;
    typedef wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_b;
    typedef wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_s;
    typedef wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_o;

    const uint32_t h = blockIdx.x;
    const uint32_t row0 = tile_row0[blockIdx.y];
    const uint32_t slot = tile_slot[blockIdx.y];
    const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
    const uint32_t kvh = h / (n_heads / n_kv_heads);

    __shared__ half sh_q[NC * DP];
    __shared__ float sh_s[NC * KQP];   // scores f32; P overwrites as f16
    __shared__ half sh_o[NC * DP];     // running O accumulator
    __shared__ float sh_corr[NC];
    __shared__ float sh_onorm[NC];
    __shared__ uint32_t sh_hi[NC];
    half* sh_p = (half*)sh_s;          // P at half stride 2*KQP, in place

    // liveness first: the CUDA-graph path pads the tile grid with dead
    // sentinels, and those blocks must exit before the Q/O staging cost
    if (tid < NC) {
        const uint32_t b = row0 + tid;
        sh_hi[tid] = (b < n_rows && slots[b] == slot) ? positions[b] + 1u : 0u;
    }
    __syncthreads();
    uint32_t hi = 0;
    #pragma unroll
    for (uint32_t i = 0; i < NC; ++i) hi = max(hi, sh_hi[i]);
    if (hi == 0u) return;  // whole tile masked (sentinel or spill-only)

    // stage Q (f32 -> f16, pre-scaled); a row is live only if in range AND
    // in this slot - spilled rows past the text zero out here
    #pragma unroll
    for (uint32_t it = 0; it < NC * D / 128u; ++it) {
        const uint32_t i = it * 128u + tid, j = i / D, dd = i % D;
        const uint32_t b = row0 + j;
        const bool live = b < n_rows && slots[b] == slot;
        sh_q[j * DP + dd] = __float2half(
            live ? q[((size_t)b * n_heads + h) * D + dd] * scale : 0.f);
        sh_o[j * DP + dd] = __float2half(0.f);
    }
    __syncthreads();

    // pin Q in fragments: Q_b[dim frag][query frag]
    frag_b Q_b[D / 16u][NC / 16u];
    #pragma unroll
    for (uint32_t d0 = 0; d0 < D / 16u; ++d0)
        #pragma unroll
        for (uint32_t j0 = 0; j0 < NC / 16u; ++j0)
            wmma::load_matrix_sync(Q_b[d0][j0], sh_q + j0 * 16u * DP + d0 * 16u, DP);
    __syncthreads();

    // softmax state: this warp owns queries [8*warp, 8*warp+8)
    float m_st[NC / 4u], l_st[NC / 4u];
    #pragma unroll
    for (uint32_t jj = 0; jj < NC / 4u; ++jj) { m_st[jj] = -1e30f; l_st[jj] = 0.f; }

    const __half* kcb = kc + (size_t)slot * max_ctx * kv_dim + (size_t)kvh * D;
    const __half* vcb = vc + (size_t)slot * max_ctx * kv_dim + (size_t)kvh * D;

    // SWA layers: START at the block's window edge instead of masking ~all
    // of history (a 4k-prompt chunk computed ~4000 masked KV columns per
    // row on the sliding-window layers - 812 us/layer avg measured).
    // Live rows in a tile are one slot's consecutive positions,
    // so tiles below the min live row's window edge are fully masked and
    // contribute exact zeros to the online softmax (skip = bit-exact).
    uint32_t lo_t = 0;
    if (swa_window > 0) {
        uint32_t lo1 = 0xFFFFFFFFu;
        #pragma unroll
        for (uint32_t i = 0; i < NC; ++i)
            if (sh_hi[i]) lo1 = min(lo1, sh_hi[i]);
        if (lo1 != 0xFFFFFFFFu && lo1 > swa_window)
            lo_t = ((lo1 - swa_window) / TK) * TK;
    }
    for (uint32_t t0 = lo_t; t0 < hi; t0 += TK) {
        // S = Q K^T for this warp's 16-key strip x all 32 queries
        {
            frag_s S_c[NC / 16u];
            #pragma unroll
            for (uint32_t j0 = 0; j0 < NC / 16u; ++j0) wmma::fill_fragment(S_c[j0], 0.f);
            #pragma unroll
            for (uint32_t d0 = 0; d0 < D / 16u; ++d0) {
                frag_k K_a;
                wmma::load_matrix_sync(
                    K_a, kcb + (size_t)(t0 + 16u * warp) * kv_dim + d0 * 16u, kv_dim);
                #pragma unroll
                for (uint32_t j0 = 0; j0 < NC / 16u; ++j0)
                    wmma::mma_sync(S_c[j0], K_a, Q_b[d0][j0], S_c[j0]);
            }
            #pragma unroll
            for (uint32_t j0 = 0; j0 < NC / 16u; ++j0)
                wmma::store_matrix_sync(sh_s + j0 * 16u * KQP + 16u * warp, S_c[j0],
                                        KQP, wmma::mem_col_major);
        }
        __syncthreads();

        // online softmax: warp owns 8 query rows, 2 keys per lane
        #pragma unroll
        for (uint32_t jj = 0; jj < 8u; ++jj) {
            const uint32_t j = warp * 8u + jj;
            const uint32_t b = row0 + j;
            const bool live = b < n_rows && slots[b] == slot;
            const uint32_t pos = live ? positions[b] : 0u;
            const uint32_t fp =
                (swa_window > 0 && pos + 1u > swa_window) ? pos + 1u - swa_window : 0u;
            float s0 = -1e30f, s1 = -1e30f;
            const uint32_t k0 = t0 + lane, k1 = t0 + 32u + lane;
            if (live && k0 >= fp && k0 <= pos && k0 < hi) s0 = sh_s[j * KQP + lane];
            if (live && k1 >= fp && k1 <= pos && k1 < hi) s1 = sh_s[j * KQP + 32u + lane];
            float mn = fmaxf(m_st[jj], fmaxf(s0, s1));
            #pragma unroll
            for (uint32_t o = 16; o > 0; o >>= 1)
                mn = fmaxf(mn, __shfl_xor_sync(0xffffffffu, mn, o));
            const float dc = m_st[jj] - mn;
            const float corr = dc >= -20.f ? __expf(dc) : 0.f;
            const float d0 = s0 - mn, d1 = s1 - mn;
            const float w0 = d0 >= -20.f ? __expf(d0) : 0.f;
            const float w1 = d1 >= -20.f ? __expf(d1) : 0.f;
            float ws = w0 + w1;
            #pragma unroll
            for (uint32_t o = 16; o > 0; o >>= 1)
                ws += __shfl_xor_sync(0xffffffffu, ws, o);
            l_st[jj] = l_st[jj] * corr + ws;
            m_st[jj] = mn;
            // P in place over the f32 scores (half stride 2*KQP)
            sh_p[j * 2u * KQP + lane] = __float2half(w0);
            sh_p[j * 2u * KQP + 32u + lane] = __float2half(w1);
            if (lane == 0) sh_corr[j] = corr;
        }
        __syncthreads();

        // rescale running O by corr[q] (f16, half2)
        #pragma unroll
        for (uint32_t it = 0; it < NC * D / 2u / 128u; ++it) {
            const uint32_t i = it * 128u + tid, j = i / (D / 2u), d2 = i % (D / 2u);
            half2* o2 = (half2*)(sh_o + j * DP);
            const float c = sh_corr[j];
            o2[d2] = __hmul2(o2[d2], __float2half2_rn(c));
        }
        __syncthreads();

        // O += V P: this warp owns dims [DW*warp, DW*warp+DW)
        {
            frag_o O_c[DW / 16u][NC / 16u];
            #pragma unroll
            for (uint32_t df = 0; df < DW / 16u; ++df)
                #pragma unroll
                for (uint32_t j0 = 0; j0 < NC / 16u; ++j0)
                    wmma::load_matrix_sync(
                        O_c[df][j0], sh_o + j0 * 16u * DP + DW * warp + df * 16u,
                        DP, wmma::mem_col_major);
            #pragma unroll
            for (uint32_t kf = 0; kf < TK / 16u; ++kf) {
                frag_v V_a[DW / 16u];
                #pragma unroll
                for (uint32_t df = 0; df < DW / 16u; ++df)
                    wmma::load_matrix_sync(
                        V_a[df], vcb + (size_t)(t0 + kf * 16u) * kv_dim + DW * warp + df * 16u,
                        kv_dim);
                #pragma unroll
                for (uint32_t j0 = 0; j0 < NC / 16u; ++j0) {
                    frag_b P_b;
                    wmma::load_matrix_sync(P_b, sh_p + j0 * 16u * 2u * KQP + kf * 16u,
                                           2u * KQP);
                    #pragma unroll
                    for (uint32_t df = 0; df < DW / 16u; ++df)
                        wmma::mma_sync(O_c[df][j0], V_a[df], P_b, O_c[df][j0]);
                }
            }
            #pragma unroll
            for (uint32_t df = 0; df < DW / 16u; ++df)
                #pragma unroll
                for (uint32_t j0 = 0; j0 < NC / 16u; ++j0)
                    wmma::store_matrix_sync(
                        sh_o + j0 * 16u * DP + DW * warp + df * 16u, O_c[df][j0],
                        DP, wmma::mem_col_major);
        }
        __syncthreads();
    }

    // epilogue: fold the sink into l with the same max-rebase as the other
    // attention kernels, publish per-query 1/l for the write-out
    #pragma unroll
    for (uint32_t jj = 0; jj < RPW; ++jj) {
        const uint32_t j = warp * RPW + jj;
        if (lane == 0) {
            const float s = sinks[h];
            const float mt = fmaxf(m_st[jj], s);
            const float dm = m_st[jj] - mt, ds = s - mt;
            const float cm = dm >= -20.f ? __expf(dm) : 0.f;
            const float cs = ds >= -20.f ? __expf(ds) : 0.f;
            const float l = l_st[jj] * cm + cs;
            sh_onorm[j] = l > 0.f ? cm / l : 0.f;
        }
    }
    __syncthreads();
    #pragma unroll
    for (uint32_t it = 0; it < NC * D / 128u; ++it) {
        const uint32_t i = it * 128u + tid, j = i / D, dd = i % D;
        const uint32_t b = row0 + j;
        if (b < n_rows && slots[b] == slot)
            out[((size_t)b * n_heads + h) * D + dd] =
                __half2float(sh_o[j * DP + dd]) * sh_onorm[j];
    }
#else
    (void)q; (void)kc; (void)vc; (void)sinks; (void)out; (void)positions;
    (void)slots; (void)tile_row0; (void)tile_slot; (void)n_heads; (void)n_kv_heads;
    (void)max_ctx; (void)kv_dim; (void)swa_window; (void)n_rows; (void)scale;
#endif
}

PD_EXPORT
int pd_attn_prefill_batch_f16(const void* q, const void* kc, const void* vc,
                              const void* sinks, void* out, const void* positions,
                              const void* slots, const void* tile_row0,
                              const void* tile_slot, uint32_t n_qtiles,
                              uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
                              uint32_t max_ctx, uint32_t kv_dim, uint32_t swa_window,
                              uint32_t n_rows, float scale, uint32_t kv_dtype,
                              void* stream) {
    if (n_heads == 0 || n_qtiles == 0 || n_rows == 0) return 0;
    if (head_dim != 128u || kv_dtype == PD_KV_FP8_E4M3 || (max_ctx & 63u))
        return cudaErrorInvalidValue;
    dim3 grid(n_heads, n_qtiles);
    pd_attn_prefill_batch_f16_kernel<128u><<<grid, 128, 0, (cudaStream_t)stream>>>(
        (const float*)q, (const __half*)kc, (const __half*)vc, (const float*)sinks,
        (float*)out, (const unsigned int*)positions, (const unsigned int*)slots,
        (const unsigned int*)tile_row0, (const unsigned int*)tile_slot,
        n_heads, n_kv_heads, max_ctx, kv_dim, swa_window, n_rows, scale);
    return pd_launch_status();
}

// (A fused rmsnorm+quantize_q8 kernel for the B=1 decode graph measured
// slower end to end: 20b decode 217.7 -> 208.2, 120b 147.4 -> 140.1. The
// separate quantize launch runs 90 32-thread blocks in parallel across SMs;
// the fused single-block phase serialized them behind the norm barrier, and
// graph-replay launch overhead is smaller than the ~4 us/pair it needed to
// win. Two launches stay.)

