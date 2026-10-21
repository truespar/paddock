// deltanet/walk_tc5.cuh - the chunked gated-delta-rule WALK on tcgen05, state resident in TMEM.
// Textually-included segment of the single pack translation unit (after deltanet/walk_rs.cuh,
// whose pd_dnrs_walk_go routes here; needs tma_desc's pd_tc5_sdesc / pd_tmap_encode).
//
// Why. pd_dnc_walk_rs_kernel (mma.sync, state in registers) splits each head's state into four
// 32-column groups (192 CTAs) and runs ~6 us a chunk: per chunk every group re-stages dw / q / k /
// coef through a cp.async ring and walks four register-fragment passes behind CTA barriers. On
// the B200 recover load the walk is 215 us a prefill layer of ~3000 rows. FlashInfer's Blackwell
// chunked GDN shows the better structure: one CTA a (sequence, head), the 128x128 state in
// TMEM. This kernel is that structure applied to the walk's own inputs - stage1 is unchanged
// and so is every operand the walk reads.
//
// Math (the transposed form the state is already stored in: S^T[c][a], c = value column, a = key
// index; s_head[c * D + a]). Per chunk of C = 64 rows, cl valid:
//   X1 = S^T . dw^T          [128 x 64]   (dl^T: dw @ S0)
//   X2 = S^T . q^T           [128 x 64]   (o1^T: q @ S0)
//   D  = du^T - X1           (columns j >= cl forced to 0)
//   O  = X2 . diag(gam) + D . coef^T          -> out rows c0 + i, i < cl
//   S^T = gall * S^T + (D . diag(w)) . k
// with the walk_rs operand classes: S, D and w*D rounded to bf16 as mma operands, every
// accumulation f32 (TMEM), the state itself never rounded inside a span. Same math and operand
// class as walk_rs; the k16 grouping is the tensor core's, so outputs agree to the class.
//
// Shape (320 threads, one CTA a (span, head)):
//   warp 4  producer: per chunk the 512 B gate row (its own barrier, so the exps land early),
//           then dw, q, k (two 64-col SW128 panels each), coef (one SW128 panel) and du (plain
//           [64][128]) by TMA into a 2-deep ring of ~73 KB stages; it exps the gate vectors into
//           the stage as soon as the row lands.
//   warp 5  mma issuer (warp-converged, elect.sync): X1, X2 with S^T bf16 as the TMEM A operand,
//           then S^T += (wD) k (k as an MN-major B, a 64-col panel a group) and O += D coef^T.
//   warps 0-3 the chain (thread = TMEM lane = value column): state load / store, the S^T -> bf16
//           copy, the gall decay, D and w*D.
//   warps 6-9 the output (their lane quadrants warp % 4 cover all 128): the gam scale of X2 and
//           the drain, from a double-buffered O, a chunk ahead of the buffer's reuse: O -> a
//           double-buffered f32 smem tile -> one 32 KB TMA store a chunk (plain stores only for a
//           span's partial last chunk).
// TMEM (512 cols): S^T f32 [0,128) | X1 [128,192) | X2 / O, double-buffered [192,320) |
//                  S^T bf16 [320,384) | D bf16 [384,416) | wD bf16 [416,448).
// The serial chain a chunk is S -> bf16 copy -> X1 -> D -> S update; X2, its gam scale, the
// O accumulate and the output drain (a chunk behind, from the other O buffer) ride beside it.

#define PD_DNT5_C 64u
#define PD_DNT5_D 128u
#define PD_DNT5_PANEL 8192u                                   // 64 rows x 128 B
#define PD_DNT5_STAGE (7u * PD_DNT5_PANEL + 2u * PD_DNT5_PANEL + 2048u)  // dw2 q2 k2 cf1 | du (16 KB) | gates (512 B raw + 129 f32)
#define PD_DNT5_NS 2u
#define PD_DNT5_OSTG (PD_DNT5_C * PD_DNT5_D * 4u)                 // one chunk's O, f32 [64][128]
#define PD_DNT5_SMEM (PD_DNT5_NS * PD_DNT5_STAGE + 2u * PD_DNT5_OSTG + 1024u)

// PD_DNT5_PROF (bench only): per-chunk clock64 stamps of the four roles of CTA (head 0, span 0)
#ifdef PD_DNT5_PROF
__device__ unsigned long long pd_dnt5_prof[4][64][12];
#define PD_DNT5_T(role, ch, slot)                                                               \
    do {                                                                                        \
        if (blockIdx.y == 0u && blockIdx.z == 0u && (threadIdx.x & 31u) == 0u && (ch) < 64u)    \
            pd_dnt5_prof[role][ch][slot] = clock64();                                           \
    } while (0)
#else
#define PD_DNT5_T(role, ch, slot) do { } while (0)
#endif

#if PD_TC5_OK
__device__ __forceinline__ void pd_dnt5_ld32(uint32_t taddr, uint32_t (&r)[32]) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]),
          "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]),
          "=r"(r[14]), "=r"(r[15]), "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]),
          "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]), "=r"(r[25]),
          "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
        : "r"(taddr));
}
__device__ __forceinline__ void pd_dnt5_st32(uint32_t taddr, const uint32_t (&r)[32]) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], "
        "{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,"
        "%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32};"
        ::"r"(taddr), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]),
          "r"(r[6]), "r"(r[7]), "r"(r[8]), "r"(r[9]), "r"(r[10]), "r"(r[11]), "r"(r[12]),
          "r"(r[13]), "r"(r[14]), "r"(r[15]), "r"(r[16]), "r"(r[17]), "r"(r[18]), "r"(r[19]),
          "r"(r[20]), "r"(r[21]), "r"(r[22]), "r"(r[23]), "r"(r[24]), "r"(r[25]), "r"(r[26]),
          "r"(r[27]), "r"(r[28]), "r"(r[29]), "r"(r[30]), "r"(r[31]));
}
__device__ __forceinline__ void pd_dnt5_st16(uint32_t taddr, const uint32_t (&r)[16]) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], "
        "{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};"
        ::"r"(taddr), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]),
          "r"(r[6]), "r"(r[7]), "r"(r[8]), "r"(r[9]), "r"(r[10]), "r"(r[11]), "r"(r[12]),
          "r"(r[13]), "r"(r[14]), "r"(r[15]));
}
// two state elements -> one 32-bit word, pd_dns_st's rounding (RN) for the 2-byte state classes
__device__ __forceinline__ uint32_t pd_dnt5_pk2(const __half*, float lo, float hi) {
    const __half2 p = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&p);
}
__device__ __forceinline__ uint32_t pd_dnt5_pk2(const __nv_bfloat16*, float lo, float hi) {
    const __nv_bfloat162 p = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&p);
}
__device__ __forceinline__ uint32_t pd_dnt5_bf2(float lo, float hi) {
    const __nv_bfloat162 p = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&p);
}
#endif

template <typename ST>
__global__ void __launch_bounds__(320, 1) pd_dnc_walk_tc5_kernel(
    const __grid_constant__ PdTmap tm_dw, const __grid_constant__ PdTmap tm_q,
    const __grid_constant__ PdTmap tm_k, const __grid_constant__ PdTmap tm_du,
    const __grid_constant__ PdTmap tm_cf, const __grid_constant__ PdTmap tm_out, ST* __restrict__ state,
    const float* __restrict__ gsh,
    float* __restrict__ out, uint32_t n_tokens, uint32_t n_heads,
    const uint32_t* __restrict__ vl_items, uint32_t n_k_heads) {
#if PD_TC5_OK
    constexpr uint32_t C = PD_DNT5_C, D = PD_DNT5_D, PANEL = PD_DNT5_PANEL, STG = PD_DNT5_STAGE,
                       NS = PD_DNT5_NS, GOFF = 9u * PD_DNT5_PANEL;   // gate area in a stage
    extern __shared__ __align__(1024) unsigned char pd_dnt5_raw[];
    unsigned char* ring = pd_dnt5_raw
        + ((1024u - ((uint32_t)__cvta_generic_to_shared(pd_dnt5_raw) & 1023u)) & 1023u);
    __shared__ __align__(8) uint64_t full[NS], gfull[NS], empty[NS], gate_rdy[NS], sb_rdy, x1_done, x2_done, d_rdy,
        s_done, o_rdy, o_done[2], o_free[2];
    __shared__ uint32_t tmem_slot[1];

    const uint32_t h = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
    uint32_t cb0 = 0u, orow = 0u;
    if (vl_items != nullptr) {
        const uint32_t* it = vl_items + (size_t)blockIdx.z * 4u;
        cb0 = it[0];
        n_tokens = it[1];
        state += (size_t)it[2];
        orow = it[3];
    }
    const uint32_t orow0 = orow;
    out += (size_t)orow * n_heads * D;
    gsh += (size_t)cb0 * n_heads * 2u * C;
    const uint32_t nc = (n_tokens + C - 1u) / C;

    if (tid == 0) {
        auto init = [](uint64_t* bar, uint32_t n) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(n));
        };
        for (uint32_t s = 0; s < NS; ++s) { init(&full[s], 1u); init(&gfull[s], 1u); init(&empty[s], 1u); init(&gate_rdy[s], 1u); }
        init(&sb_rdy, 4u); init(&x1_done, 1u); init(&x2_done, 1u); init(&d_rdy, 4u);
        init(&s_done, 1u); init(&o_rdy, 4u);
        for (uint32_t b = 0; b < 2u; ++b) { init(&o_done[b], 1u); init(&o_free[b], 4u); }
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    if (warp == 5)
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;"
                     ::"r"((uint32_t)__cvta_generic_to_shared(tmem_slot)));
    asm volatile("tcgen05.fence::before_thread_sync;");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");
    const uint32_t tmem = tmem_slot[0];
    // TMEM columns: S^T f32 | X1 | O buffer 0 | O buffer 1 | S^T bf16 | D bf16 | wD bf16
    constexpr uint32_t T_S = 0u, T_X1 = 128u, T_O = 192u, T_SB = 320u, T_D = 384u, T_WD = 416u;

    auto bar_wait = [&](uint64_t* bar, uint32_t parity) {
        const uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
        asm volatile("{\n\t.reg .pred p;\nW%=:\n\t"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                     "@!p bra W%=;\n\t}" ::"r"(a), "r"(parity) : "memory");
    };
    auto arrive1 = [](uint64_t* bar) {
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
                     ::"r"((uint32_t)__cvta_generic_to_shared(bar)) : "memory");
    };
    // compute-warp handoff: TMEM stores / smem patches visible, then one arrive a warp
    auto warp_arrive = [&](uint64_t* bar) {
        asm volatile("tcgen05.wait::st.sync.aligned;");
        asm volatile("fence.proxy.async.shared::cta;");
        asm volatile("tcgen05.fence::before_thread_sync;");
        __syncwarp();
        if (lane == 0) arrive1(bar);
    };
    // a stage's gate vectors: [0,64) w, [64,128) gam, [128] gall (exp'd by the producer warp)
    auto gates = [&](uint32_t s) { return (const float*)(ring + s * STG + GOFF + 512u); };

    if (warp == 4) {
        // ---- producer: a 72 KB stage a chunk by TMA; then, a chunk behind, its gate vectors ----
        auto gate_fill = [&](uint32_t ch) {
            const uint32_t s = ch % NS, ph = (ch / NS) & 1u;
            const uint32_t cl = min(C, n_tokens - ch * C);
            bar_wait(&gfull[s], ph);
            const float* raw = (const float*)(ring + s * STG + GOFF);
            float* gv = (float*)(ring + s * STG + GOFF + 512u);
            // walk_rs's expressions on stage1's pre-diffed f32 gates
            for (uint32_t j = lane; j < C; j += 32u) {
                gv[j] = j < cl ? expf(raw[j]) : 0.0f;
                gv[C + j] = j < cl ? expf(raw[C + j]) : 0.0f;
            }
            if (lane == 0) gv[2u * C] = expf(raw[C + cl - 1u]);
            __syncwarp();
            if (lane == 0) arrive1(&gate_rdy[s]);
            PD_DNT5_T(3, ch, 1);
        };
        for (uint32_t ch = 0; ch < nc; ++ch) {
            const uint32_t s = ch % NS, ph = (ch / NS) & 1u;
            bar_wait(&empty[s], ph ^ 1u);
            PD_DNT5_T(3, ch, 0);
            if (lane == 0) {
                // the 512 B gate row first, on its own barrier: its exps must not wait for the 72 KB
                const uint32_t gbar = (uint32_t)__cvta_generic_to_shared(&gfull[s]);
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(gbar), "r"(512u));
                asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], 512, [%2];"
                             ::"r"((uint32_t)__cvta_generic_to_shared(ring + s * STG + GOFF)),
                               "l"(gsh + ((size_t)ch * n_heads + h) * 2u * C), "r"(gbar) : "memory");
                const uint32_t bar = (uint32_t)__cvta_generic_to_shared(&full[s]);
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                             ::"r"(bar), "r"(9u * PANEL));
                const int y = (int)(((cb0 + ch) * n_heads + h) * C);
                // q / k: stage1's chunk-major copies, or (n_k_heads, the QKD route) the conv's compact
                // [launch row][HK][D] planes at the chunk's own rows - rows past a partial chunk read
                // the next span (or zero fill), masked exactly like the copies' unwritten pad
                const int yq = n_k_heads ? (int)(orow0 + ch * C) : y;
                const int xq = n_k_heads ? (int)((h % n_k_heads) * D) : 0;
                unsigned char* st = ring + s * STG;
                auto tma = [&](const PdTmap& m, unsigned char* dst, int x, int yy) {
                    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                                 " [%0], [%1, {%2, %3}], [%4];"
                                 ::"r"((uint32_t)__cvta_generic_to_shared(dst)), "l"(&m), "r"(x), "r"(yy),
                                   "r"(bar) : "memory");
                };
                tma(tm_dw, st + 0u * PANEL, 0, y);
                tma(tm_dw, st + 1u * PANEL, 64, y);
                tma(tm_q, st + 2u * PANEL, xq, yq);
                tma(tm_q, st + 3u * PANEL, xq + 64, yq);
                tma(tm_k, st + 4u * PANEL, xq, yq);
                tma(tm_k, st + 5u * PANEL, xq + 64, yq);
                tma(tm_cf, st + 6u * PANEL, 0, y);
                tma(tm_du, st + 7u * PANEL, 0, y);
            }
            __syncwarp();
            gate_fill(ch);
        }
    } else if (warp == 5) {
        // ---- mma issuer. The serial chain is S -> X1 -> D -> S update; X2, the O accumulate
        // and the drain ride beside it ----
        const uint32_t tm = __shfl_sync(0xffffffffu, tmem, 0);
        // bf16 x bf16 -> f32, M = 128, N = 64; B K-major (dw, q, coef) / MN-major (k)
        const uint32_t id_k = (1u << 4) | (1u << 7) | (1u << 10) | ((64u >> 3) << 17) | ((128u >> 4) << 24);
        const uint32_t id_m = id_k | (1u << 16);
        const uint32_t ring16 = (uint32_t)__cvta_generic_to_shared(ring) >> 4;
        auto mma = [](uint32_t d, uint32_t a, uint64_t b, uint32_t id, uint32_t en) {
            asm volatile("{\n\t.reg .pred p, e;\n\telect.sync _|e, 0xffffffff;\n\t"
                         "setp.ne.b32 p, %4, 0;\n\t"
                         "@e tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n\t}"
                         ::"r"(d), "r"(a), "l"(b), "r"(id), "r"(en));
        };
        auto commit = [](uint64_t* b) {
            asm volatile("{\n\t.reg .pred e;\n\telect.sync _|e, 0xffffffff;\n\t"
                         "@e tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n\t}"
                         ::"r"((uint32_t)__cvta_generic_to_shared(b)) : "memory");
        };
        for (uint32_t ch = 0; ch < nc; ++ch) {
            const uint32_t s = ch % NS, ph = (ch / NS) & 1u, ob = ch & 1u;
            const uint32_t st16 = ring16 + (s * STG) / 16u;
            bar_wait(&full[s], ph);
            PD_DNT5_T(1, ch, 0);
            bar_wait(&sb_rdy, ch & 1u);
            PD_DNT5_T(1, ch, 1);
            asm volatile("tcgen05.fence::after_thread_sync;");
            // X1 = S^T dw^T: K = 128 over two 64-col panels, 4 k16 steps a panel
            #pragma unroll
            for (uint32_t kb = 0; kb < 8u; ++kb)
                mma(tm + T_X1, tm + T_SB + kb * 8u,
                    pd_tc5_sdesc(st16 + (kb >> 2) * (PANEL / 16u) + (kb & 3u) * 2u), id_k, kb);
            commit(&x1_done);
            PD_DNT5_T(1, ch, 2);
            // X2 = S^T q^T into this chunk's O buffer, once the drain two chunks back freed it
            bar_wait(&o_free[ob], ((ch >> 1) & 1u) ^ 1u);
            asm volatile("tcgen05.fence::after_thread_sync;");
            #pragma unroll
            for (uint32_t kb = 0; kb < 8u; ++kb)
                mma(tm + T_O + ob * 64u, tm + T_SB + kb * 8u,
                    pd_tc5_sdesc(st16 + (2u + (kb >> 2)) * (PANEL / 16u) + (kb & 3u) * 2u), id_k, kb);
            commit(&x2_done);
            PD_DNT5_T(1, ch, 3);
            // S^T += (wD) k, a 64-col key panel a group (MN-major k) - the next chunk's input
            bar_wait(&d_rdy, ch & 1u);
            PD_DNT5_T(1, ch, 4);
            asm volatile("tcgen05.fence::after_thread_sync;");
            #pragma unroll
            for (uint32_t pn = 0; pn < 2u; ++pn)
                #pragma unroll
                for (uint32_t kb = 0; kb < 4u; ++kb)
                    mma(tm + T_S + pn * 64u, tm + T_WD + kb * 8u,
                        pd_tc5_sdesc(st16 + (4u + pn) * (PANEL / 16u) + kb * 128u), id_m, 1u);
            commit(&s_done);
            PD_DNT5_T(1, ch, 5);
            // O += D coef^T (K = 64 = j) once X2 carries its gam scale
            bar_wait(&o_rdy, ch & 1u);
            PD_DNT5_T(1, ch, 6);
            asm volatile("tcgen05.fence::after_thread_sync;");
            #pragma unroll
            for (uint32_t kb = 0; kb < 4u; ++kb)
                mma(tm + T_O + ob * 64u, tm + T_D + kb * 8u,
                    pd_tc5_sdesc(st16 + 6u * (PANEL / 16u) + kb * 2u), id_k, 1u);
            commit(&o_done[ob]);
            commit(&empty[s]);
        }
    } else if (warp < 4u) {
        // ---- chain group, warps 0-3: thread = TMEM lane = value column c ----
        const uint32_t c = tid;
        const uint32_t tl = tmem + ((warp * 32u) << 16);
        ST* s_row = state + (size_t)h * D * D + (size_t)c * D;
        if constexpr (sizeof(ST) == 2u) {
            // the 256 B state row as 16-byte vectors, all in flight before the first use (128 scalar
            // 2-byte loads at a 256 B lane stride held the first chunk back ~17K clk)
            uint4 v[16];
            #pragma unroll
            for (uint32_t k = 0; k < 16u; ++k) v[k] = reinterpret_cast<const uint4*>(s_row)[k];
            #pragma unroll
            for (uint32_t q4 = 0; q4 < 4u; ++q4) {
                uint32_t r[32];
                #pragma unroll
                for (uint32_t j = 0; j < 32u; ++j)
                    r[j] = __float_as_uint(pd_dns_ld(reinterpret_cast<const ST*>(&v[q4 * 4u + j / 8u]) + j % 8u));
                pd_dnt5_st32(tl + T_S + q4 * 32u, r);
            }
        } else {
            #pragma unroll
            for (uint32_t q4 = 0; q4 < 4u; ++q4) {
                uint32_t r[32];
                #pragma unroll
                for (uint32_t j = 0; j < 32u; ++j) r[j] = __float_as_uint(pd_dns_ld(s_row + q4 * 32u + j));
                pd_dnt5_st32(tl + T_S + q4 * 32u, r);
            }
        }
        asm volatile("tcgen05.wait::st.sync.aligned;");
        for (uint32_t ch = 0; ch < nc; ++ch) {
            const uint32_t s = ch % NS, ph = (ch / NS) & 1u;
            const uint32_t cl = min(C, n_tokens - ch * C);
            // S^T (updated by the previous chunk) -> its bf16 operand copy: the chain's head
            PD_DNT5_T(0, ch, 0);
            if (ch > 0u) {
                bar_wait(&s_done, (ch - 1u) & 1u);
                asm volatile("tcgen05.fence::after_thread_sync;");
            }
            PD_DNT5_T(0, ch, 1);
            // all four 32-col loads in flight before one wait; the bf16 copy and the decay both
            // come from these registers
            uint32_t sr[128];
            #pragma unroll
            for (uint32_t q4 = 0; q4 < 4u; ++q4) {
                uint32_t* r = sr + q4 * 32u;
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
                    "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
                    "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
                    : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]),
                      "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]),
                      "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]), "=r"(r[17]), "=r"(r[18]),
                      "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]),
                      "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]),
                      "=r"(r[31])
                    : "r"(tl + T_S + q4 * 32u));
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            #pragma unroll
            for (uint32_t q4 = 0; q4 < 4u; ++q4) {
                uint32_t b[16];
                #pragma unroll
                for (uint32_t j = 0; j < 16u; ++j)
                    b[j] = pd_dnt5_bf2(__uint_as_float(sr[q4 * 32u + 2u * j]),
                                       __uint_as_float(sr[q4 * 32u + 2u * j + 1u]));
                pd_dnt5_st16(tl + T_SB + q4 * 16u, b);
            }
            warp_arrive(&sb_rdy);
            PD_DNT5_T(0, ch, 2);
            // S^T *= gall in place, under X1 (the S update needs it, via d_rdy below)
            bar_wait(&gate_rdy[s], ph);
            PD_DNT5_T(0, ch, 6);
            const float* gv = gates(s);
            const float gall = gv[2u * C];
            #pragma unroll
            for (uint32_t q4 = 0; q4 < 4u; ++q4) {
                uint32_t r[32];
                #pragma unroll
                for (uint32_t j = 0; j < 32u; ++j) r[j] = __float_as_uint(__uint_as_float(sr[q4 * 32u + j]) * gall);
                pd_dnt5_st32(tl + T_S + q4 * 32u, r);
            }
            // D = du^T - X1 (j >= cl -> 0) and wD; du is TMA-written: observe its full barrier
            PD_DNT5_T(0, ch, 3);
            bar_wait(&full[s], ph);
            bar_wait(&x1_done, ch & 1u);
            PD_DNT5_T(0, ch, 4);
            asm volatile("tcgen05.fence::after_thread_sync;");
            const __nv_bfloat16* du = (const __nv_bfloat16*)(ring + s * STG + 7u * PANEL);
            uint32_t xx[2][32];
            pd_dnt5_ld32(tl + T_X1, xx[0]);
            pd_dnt5_ld32(tl + T_X1 + 32u, xx[1]);
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            #pragma unroll
            for (uint32_t hf = 0; hf < 2u; ++hf) {
                uint32_t db[16], wb[16];
                const uint32_t* x = xx[hf];
                #pragma unroll
                for (uint32_t j2 = 0; j2 < 16u; ++j2) {
                    const uint32_t j = hf * 32u + 2u * j2;
                    const float d0 = j < cl ? __bfloat162float(du[j * D + c]) - __uint_as_float(x[2u * j2]) : 0.0f;
                    const float d1 = j + 1u < cl
                        ? __bfloat162float(du[(j + 1u) * D + c]) - __uint_as_float(x[2u * j2 + 1u]) : 0.0f;
                    db[j2] = pd_dnt5_bf2(d0, d1);
                    wb[j2] = pd_dnt5_bf2(d0 * gv[j], d1 * gv[j + 1u]);
                }
                pd_dnt5_st16(tl + T_D + hf * 16u, db);
                pd_dnt5_st16(tl + T_WD + hf * 16u, wb);
            }
            if (cl < C) {
                // partial chunk: coef columns j >= cl and k rows j >= cl meet zeros in the mma
                // (D / wD columns are 0 there) - clear them so stale words cannot be NaN
                unsigned char* cf = ring + s * STG + 6u * PANEL;
                for (uint32_t u = tid; u < C * C; u += 128u) {
                    const uint32_t r = u >> 6, j = u & 63u;
                    if (j >= cl)
                        *(__nv_bfloat16*)(cf + r * 128u + (((j >> 3) ^ (r & 7u)) << 4) + (j & 7u) * 2u) =
                            __float2bfloat16(0.0f);
                }
                for (uint32_t pn = 0; pn < 2u; ++pn) {
                    uint4* kp = (uint4*)(ring + s * STG + (4u + pn) * PANEL);
                    for (uint32_t u = cl * 8u + tid; u < C * 8u; u += 128u) kp[u] = make_uint4(0u, 0u, 0u, 0u);
                }
                asm volatile("bar.sync 1, 128;" ::: "memory");
            }
            warp_arrive(&d_rdy);   // D, wD, the decayed S (and any patches) visible
            PD_DNT5_T(0, ch, 5);
        }
        if (nc > 0u) {
            bar_wait(&s_done, (nc - 1u) & 1u);
            asm volatile("tcgen05.fence::after_thread_sync;");
        }
        #pragma unroll
        for (uint32_t q4 = 0; q4 < 4u; ++q4) {
            uint32_t r[32];
            pd_dnt5_ld32(tl + T_S + q4 * 32u, r);
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            if constexpr (sizeof(ST) == 2u) {
                #pragma unroll
                for (uint32_t k = 0; k < 4u; ++k) {
                    uint4 w;
                    w.x = pd_dnt5_pk2(s_row, __uint_as_float(r[8u * k + 0u]), __uint_as_float(r[8u * k + 1u]));
                    w.y = pd_dnt5_pk2(s_row, __uint_as_float(r[8u * k + 2u]), __uint_as_float(r[8u * k + 3u]));
                    w.z = pd_dnt5_pk2(s_row, __uint_as_float(r[8u * k + 4u]), __uint_as_float(r[8u * k + 5u]));
                    w.w = pd_dnt5_pk2(s_row, __uint_as_float(r[8u * k + 6u]), __uint_as_float(r[8u * k + 7u]));
                    reinterpret_cast<uint4*>(s_row)[q4 * 4u + k] = w;
                }
            } else {
                #pragma unroll
                for (uint32_t j = 0; j < 32u; ++j) pd_dns_st(s_row + q4 * 32u + j, __uint_as_float(r[j]));
            }
        }
    } else {
        // ---- output group, warps 6-9: their lane quadrants (warp % 4) cover the 128 lanes ----
        const uint32_t qd = warp & 3u, c = qd * 32u + lane;
        float* ostg = (float*)(ring + NS * STG);
        const uint32_t tl = tmem + ((qd * 32u) << 16);
        for (uint32_t ch = 0; ch < nc; ++ch) {
            const uint32_t s = ch % NS, ph = (ch / NS) & 1u, ob = ch & 1u;
            const uint32_t c0 = ch * C, cl = min(C, n_tokens - c0);
            // X2 *= gam (o1's pre-scale, before the coef chain)
            bar_wait(&gate_rdy[s], ph);
            const float* gv = gates(s);
            PD_DNT5_T(2, ch, 0);
            bar_wait(&x2_done, ch & 1u);
            PD_DNT5_T(2, ch, 1);
            asm volatile("tcgen05.fence::after_thread_sync;");
            #pragma unroll
            for (uint32_t hf = 0; hf < 2u; ++hf) {
                uint32_t o[32];
                pd_dnt5_ld32(tl + T_O + ob * 64u + hf * 32u, o);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (uint32_t j = 0; j < 32u; ++j) {
                    const uint32_t i = hf * 32u + j;
                    o[j] = __float_as_uint(__uint_as_float(o[j]) * gv[C + min(i, cl - 1u)]);
                }
                pd_dnt5_st32(tl + T_O + ob * 64u + hf * 32u, o);
            }
            warp_arrive(&o_rdy);
            PD_DNT5_T(2, ch, 2);
            // the chunk's O -> out rows; then hand the buffer back
            bar_wait(&o_done[ob], (ch >> 1) & 1u);
            PD_DNT5_T(2, ch, 3);
            asm volatile("tcgen05.fence::after_thread_sync;");
            uint32_t r[2][32];
            pd_dnt5_ld32(tl + T_O + ob * 64u, r[0]);
            pd_dnt5_ld32(tl + T_O + ob * 64u + 32u, r[1]);
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            asm volatile("tcgen05.fence::before_thread_sync;");
            __syncwarp();
            if (lane == 0) arrive1(&o_free[ob]);   // in registers: the buffer is the mma's again
            if (cl == C) {
                // a full chunk: stage [64 rows][128 c] f32 (row stores conflict-free) and one 32 KB
                // TMA store - per-thread f32 stores ran ~4600 clk a chunk, the walk's bottleneck
                float* sg = ostg + ob * (C * D);
                if (tid == 192u) asm volatile("cp.async.bulk.wait_group.read 1;" ::: "memory");
                asm volatile("bar.sync 2, 128;" ::: "memory");
                #pragma unroll
                for (uint32_t j = 0; j < 64u; ++j) sg[j * D + c] = __uint_as_float(r[j >> 5][j & 31u]);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("bar.sync 2, 128;" ::: "memory");
                if (tid == 192u) {
                    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
                                 ::"l"(&tm_out), "r"((int)(h * D)), "r"((int)(orow0 + c0)),
                                   "r"((uint32_t)__cvta_generic_to_shared(sg)) : "memory");
                    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
                }
            } else {
                // a span's partial last chunk: a box would spill into the next span's rows
                #pragma unroll
                for (uint32_t j = 0; j < 64u; ++j)
                    if (j < cl)
                        out[((size_t)(c0 + j) * n_heads + h) * D + c] = __uint_as_float(r[j >> 5][j & 31u]);
            }
            PD_DNT5_T(2, ch, 4);
        }
        if (tid == 192u) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
    }
    asm volatile("tcgen05.fence::before_thread_sync;");
    __syncthreads();
    if (warp == 5)
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" ::"r"(tmem));
#else
    (void)tm_dw; (void)tm_q; (void)tm_k; (void)tm_du; (void)tm_cf; (void)tm_out; (void)state; (void)gsh;
    (void)out; (void)n_tokens; (void)n_heads; (void)vl_items; (void)n_k_heads;
#endif
}

#if defined(PD_TC5_HOST) && defined(PD_DNT5_PROF)
extern "C" void pd_dnt5_prof_dump(unsigned long long* host) {
    cudaMemcpyFromSymbol(host, pd_dnt5_prof, sizeof(pd_dnt5_prof));
}
#endif

#if defined(PD_TC5_HOST)
// [rows][inner] 2D map (bf16, or f32 for the output), `box_x` x 64-row box; SW128 for the mma
// panels, none for du / out.
static bool pd_dnt5_tmap(CUtensorMap* out, const void* base, uint64_t inner, uint64_t rows,
                         uint32_t box_x, bool sw128, bool f32 = false) {
    struct E { const void* b; uint64_t in, rows; uint32_t bx; bool sw, f; CUtensorMap m; };
    static E cache[32];
    static uint32_t n = 0, next = 0;
    for (uint32_t i = 0; i < n; ++i)
        if (cache[i].b == base && cache[i].in == inner && cache[i].rows == rows && cache[i].bx == box_x &&
            cache[i].sw == sw128 && cache[i].f == f32) {
            *out = cache[i].m;
            return true;
        }
    pd_tmap_encode_fn enc = pd_tmap_encode();
    if (!enc || ((uintptr_t)base & 15u)) return false;
    const cuuint64_t gdim[2] = {inner, rows};
    const cuuint64_t gstride[1] = {inner * (f32 ? 4u : 2u)};
    const cuuint32_t box[2] = {box_x, 64u};
    const cuuint32_t estride[2] = {1u, 1u};
    CUtensorMap m;
    if (enc(&m, f32 ? CU_TENSOR_MAP_DATA_TYPE_FLOAT32 : CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2u, (void*)base, gdim, gstride, box, estride,
            CU_TENSOR_MAP_INTERLEAVE_NONE, sw128 ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
        return false;
    E& e = cache[n < 32u ? n++ : (next++ % 32u)];
    e = E{base, inner, rows, box_x, sw128, f32, m};
    *out = m;
    return true;
}

// The walk on tcgen05 (cc 10.0 + TMA encode; kill PADDOCK_NO_DN_WALK_TC5). `n_chunk_rows` = the
// rows the stage1 planes hold (launch chunks x heads x 64). `n_k_heads` != 0: q / k come from the
// compact bf16 [n_tokens][n_k_heads][D] planes qc / kc (the QKD route) instead of qb / kb.
// False = not taken (caller runs walk_rs).
template <typename ST>
static bool pd_dnt5_walk_go(dim3 gw, cudaStream_t s, const __nv_bfloat16* qb, const __nv_bfloat16* kb,
                            ST* state, const __nv_bfloat16* dw, const __nv_bfloat16* du, const float* gsh,
                            const __nv_bfloat16* cb, float* out, uint32_t n_tokens, uint32_t n_heads,
                            const uint32_t* items, uint64_t n_chunk_rows,
                            const __nv_bfloat16* qc = nullptr, const __nv_bfloat16* kc = nullptr,
                            uint32_t n_k_heads = 0u) {
    static int ok = -1;
    if (ok < 0) {
        int dev = 0, ccM = 0, ccm = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&ccM, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&ccm, cudaDevAttrComputeCapabilityMinor, dev);
        ok = (ccM == 10 && ccm == 0 && pd_tmap_encode() != nullptr &&
              pd_env("PADDOCK_NO_DN_WALK_TC5") == nullptr) ? 1 : 0;
        if (ok) {
            cudaFuncSetAttribute((const void*)pd_dnc_walk_tc5_kernel<ST>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)PD_DNT5_SMEM);
            ::fprintf(stderr, "[dn-walk-tc5] ENGAGED (tcgen05 walk, TMEM state)\n");
        }
    }
    if (!ok || n_chunk_rows == 0u) return false;
    CUtensorMap mdw, mq, mk, mdu, mcf, mo;
    if (!pd_dnt5_tmap(&mdw, dw, PD_DNT5_D, n_chunk_rows, 64u, true) ||
        !(n_k_heads ? pd_dnt5_tmap(&mq, qc, (uint64_t)n_k_heads * PD_DNT5_D, n_tokens, 64u, true)
                    : pd_dnt5_tmap(&mq, qb, PD_DNT5_D, n_chunk_rows, 64u, true)) ||
        !(n_k_heads ? pd_dnt5_tmap(&mk, kc, (uint64_t)n_k_heads * PD_DNT5_D, n_tokens, 64u, true)
                    : pd_dnt5_tmap(&mk, kb, PD_DNT5_D, n_chunk_rows, 64u, true)) ||
        !pd_dnt5_tmap(&mdu, du, PD_DNT5_D, n_chunk_rows, 128u, false) ||
        !pd_dnt5_tmap(&mcf, cb, PD_DNT5_C, n_chunk_rows, 64u, true) ||
        !pd_dnt5_tmap(&mo, out, (uint64_t)n_heads * PD_DNT5_D, n_tokens, PD_DNT5_D, false, true))
        return false;
    const PdTmap pdw(mdw), pq(mq), pk(mk), pdu(mdu), pcf(mcf), po(mo);
    pd_dnc_walk_tc5_kernel<ST><<<dim3(1u, gw.y, gw.z), 320, PD_DNT5_SMEM, s>>>(
        pdw, pq, pk, pdu, pcf, po, state, gsh, out, n_tokens, n_heads, items, n_k_heads);
    return true;
}
#endif
