// deltanet/walk_rs.cuh - REGISTER-STATE bf16-operand stage-2 walk.
// Textually-included segment of the single pack translation
// unit; not standalone-compilable (order: after deltanet/split.cuh for
// pd_dnc_cpa16 + the PD_DNC_* defines, before deltanet/stage2_sample.cuh
// whose launcher dispatches this kernel).
//
// Why (attribution correction): an earlier close bounded the classic
// v2 walk as the optimum of its FORMAT - 8 schedule falsifications plus the
// DWB16 operand-BYTES zero-delta - and attributed the reference kernels'
// large edge to "bf16 staging + 128-thread state-in-registers pipeline
// depth". Profiling the register-state rebuild
// pinned what that actually means: both walks are ISSUE-LATENCY bound, and
// the classic format taxes the instruction stream 6 cvt.rna.tf32 per mma
// plus 4 scalar ld.shared per A-fragment. The DWB16 rung halved staged
// BYTES but kept f32 compute - zero delta, correctly, because bytes were
// never the bottleneck; INSTRUCTION ECONOMY is. This kernel moves the mma
// operands to bf16 (m16n8k16 = half the mma count, ldmatrix.x4 = one
// instruction per 16x16 A-tile, no per-use cvts) while the STATE stays
// f32 in accumulator fragments - the bf16-STATE axis is falsified twice
// (+1.47% PPL) and stays untouched; every accumulation is f32.
// Proto ladder (T=2048, H=48): classic 15.0 us/chunk -> 5.8 (-61%); the
// tf32 register-state twin (bit-exact vs classic, 0/12.6M words) measured
// +26% and proves the structure while isolating the win to operand class.
//
// STRUCTURE: 128 threads (4 warps), grid (H, D/G) - the same 192-CTA
// one-wave geometry as classic and the reference. Each warp owns the full
// K=128 rows x an 8-V-col slice of the state, held TRANSPOSED (S^T[a][c])
// as m16n8 f32 acc fragments of the hop mma. That partition makes every
// per-chunk product warp-local (no sh_s, no sh_dT, no cross-warp k):
//   pass1  dl = dw @ S0, o1 = q @ S0    (k-dim = a, inside the warp)
//   delta  dl = du - dl                 (register subtract)
//   pass2  o  = gam*o1 + coef @ deltaT  (k-dim = j, delta in registers)
//   hop    S^T = gall*S^T + kT @ (w o deltaT)   [w folded into delta so
//                                        the k pane loads raw via ldmatrix]
// smem = the classic 8-item/4-slot cp.async ring only (bf16 panes; du pane
// bf16 too - matches the reference's bf16 u; 40 KB, under the 48 KB default).
//
// OPERAND SOURCING: dw/du/aqk arrive bf16 in the f32-sized call-internal
// buffers (stage1 AT/OT=bf16 arms - the DWB16 buffer-reuse precedent);
// q/k bf16 copies come from stage1's tail epilogue (L2-hot re-read).
//
// NUMERIC CLASS: bf16-rounded products, k16 grouping - Not bit-exact vs
// classic (proto band: out rmsrel 2.4e-3, bf16-class). This is exactly the
// the reference class for the same products (their k/w/u/A are bf16; their h
// STORAGE is bf16 where ours stays f32). Gates: PPL-distance vs reference,
// greedy fork-shape vs llama.cpp, suite, serve A/B. Kill: PADDOCK_DNC_RS=0.

static __device__ __forceinline__ void pd_dnrs_mma16(float (&acc)[4],
                                                     const uint32_t (&a)[4],
                                                     const uint32_t (&b)[2]) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// A-frag (m16n8k16, rows r0.., cols c0.. of a row-major bf16 pane).
static __device__ __forceinline__ void pd_dnrs_lda(const __nv_bfloat16* base,
                                                   uint32_t stride_h, uint32_t r0,
                                                   uint32_t c0, uint32_t (&a)[4]) {
    const uint32_t lane = threadIdx.x & 31u, t = lane >> 3;
    const uint32_t row = r0 + (lane & 7u) + ((t & 1u) << 3);
    const uint32_t col = c0 + ((t >> 1) << 3);
    const unsigned sm = (unsigned)__cvta_generic_to_shared(base + row * stride_h + col);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(sm));
}

// Transposed A-frag: A[m=a][k=j] = pane[j][a]^T (the hop's kT operand).
static __device__ __forceinline__ void pd_dnrs_ldat(const __nv_bfloat16* base,
                                                    uint32_t stride_h, uint32_t j0,
                                                    uint32_t a0, uint32_t (&a)[4]) {
    const uint32_t lane = threadIdx.x & 31u, t = lane >> 3;
    const uint32_t row = j0 + (lane & 7u) + ((t >> 1) << 3);
    const uint32_t col = a0 + ((t & 1u) << 3);
    const unsigned sm = (unsigned)__cvta_generic_to_shared(base + row * stride_h + col);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(sm));
}

// f32 acc tile (16 k-rows x 8 n-cols) -> m16n8k16 bf16 B fragment, with an
// optional per-k-row scale (the hop's folded w). Lane wants rows
// {t4*2, t4*2+1} (b0) and {+8, +9} (b1) at col g8; the holder of (r,c) is
// lane r*4+(c>>1) at element (c&1) + 2*(r>=8): rows t4*2 / t4*2+1 sit at
// lanes ha = t4*8+(g8>>1) / hb = ha+4 in elements {0,1}; the +8 rows are
// the same lanes' elements {2,3}.
template <bool SCALED>
static __device__ __forceinline__ void pd_dnrs_cb16(const float (&fr)[4],
                                                    const float* wrow,
                                                    uint32_t& b0, uint32_t& b1) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t g8 = lane >> 2, t4 = lane & 3u;
    const uint32_t ha = t4 * 8u + (g8 >> 1), hb = ha + 4u;
    const float r0e = __shfl_sync(0xffffffffu, fr[0], ha);
    const float r0o = __shfl_sync(0xffffffffu, fr[1], ha);
    const float r1e = __shfl_sync(0xffffffffu, fr[0], hb);
    const float r1o = __shfl_sync(0xffffffffu, fr[1], hb);
    const float r8e = __shfl_sync(0xffffffffu, fr[2], ha);
    const float r8o = __shfl_sync(0xffffffffu, fr[3], ha);
    const float r9e = __shfl_sync(0xffffffffu, fr[2], hb);
    const float r9o = __shfl_sync(0xffffffffu, fr[3], hb);
    const bool od = (g8 & 1u) != 0u;
    float v0 = od ? r0o : r0e, v1 = od ? r1o : r1e;
    float v2 = od ? r8o : r8e, v3 = od ? r9o : r9e;
    if (SCALED) {
        v0 *= wrow[t4 * 2u];
        v1 *= wrow[t4 * 2u + 1u];
        v2 *= wrow[t4 * 2u + 8u];
        v3 *= wrow[t4 * 2u + 9u];
    }
    const __nv_bfloat162 p0 = __floats2bfloat162_rn(v0, v1);
    const __nv_bfloat162 p1 = __floats2bfloat162_rn(v2, v3);
    b0 = *reinterpret_cast<const uint32_t*>(&p0);
    b1 = *reinterpret_cast<const uint32_t*>(&p1);
}

// pane strides in halves: 32-col rows (dw/q/du slices) pad to 40, 64-col
// rows (coef / k halves) pad to 72 - both keep cp.async rows 16B-aligned
// and spread the 8-row ldmatrix accesses across banks.
#define PD_DNRS_SW 40u
#define PD_DNRS_SK 72u
// slot: A-slab item = dw(64x40h) + q(64x40h) = 10240 B; du (64x40h bf16),
// coef and k items (64x72h) all fit under it. 4 slots = 40 KB dynamic.
// DEPTH 8 FALSIFIED: 80 KB kills the 2-CTA co-residency that
// absorbs the 192-CTAs-on-188-SMs tail - the 4 doubled SMs serialize and
// the wall goes ~2x CTA-time (187 -> 272 us at T=2048). Any deeper ring
// must stay under ~49.5 KB/block or shrink the grid below the SM count.
#ifndef PD_DNC_PROBE
#define PD_DNC_PROBE 0
#endif
#ifndef PD_DNRS_QK_TOKENMAJOR
#define PD_DNRS_QK_TOKENMAJOR 0
#endif
// walk mma alias: PD_DNC_PROBE=4 strips the mma (keeps the ldmatrix / shfl
// operand chain alive through a bit fold) for the GB10 phase split
#if PD_DNC_PROBE == 4
#define PD_DNRS_WALK_MMA(acc, a, b) do { (acc)[0] += __uint_as_float((a)[0] & 1u) + __uint_as_float((b)[0] & 1u); } while (0)
#else
#define PD_DNRS_WALK_MMA(acc, a, b) pd_dnrs_mma16((acc), (a), (b))
#endif
#define PD_DNRS_SLOT_B 10240u
#define PD_DNRS_SMEM (4u * PD_DNRS_SLOT_B)   // the deep-ring shape's dynamic smem (the attribute opt-in)

// ST: the walk touches state only at entry load and final
// writeback (registers in between), so the narrow-state class rides two
// helper swaps. f16 state passed the DN PPL gate at +0.09%; bf16 stays
// falsified and is excluded at every launch gate.
// QKD (2026-09-08, GB10): the walk reads q/k straight from the conv's
// COMPACT bf16 plane ([tick row][HK][D]) by k-head - on the qkc route the
// stage1 panes hold exactly those bytes, so the 25 MB expanded copies
// stage1 emitted and the walk re-read were the same values in a wider
// coat. qc/kc/n_k_heads carry the plane; qb/kb are unused under QKD.
// S = ring depth in items, MINB = the launch-bounds block floor: the
// co-residency shape. Shipped big-die shape <4, 1> (41 KB, ~168 registers,
// 1-2 CTAs/SM); small dies (< 128 SMs) take <2, 4> - a 20 KB ring under a
// 128-register cap puts 4 CTAs on each SM, so the 192-CTA grid is ONE wave
// on 48 SMs instead of two (GB10 2026-09-08: the pair 812 -> 752 us at
// 1024 rows, byte-identical; the shallower ring is hidden by the three
// co-resident CTAs where the deep ring hid it within one).
template <typename ST = float, bool QKD = false, uint32_t S = 4u, int MINB = 1>
__global__ void __launch_bounds__(128, MINB)
pd_dnc_walk_rs_kernel(const __nv_bfloat16* __restrict__ qb,
                      const __nv_bfloat16* __restrict__ kb, ST* __restrict__ state,
                      const __nv_bfloat16* __restrict__ dwb,
                      const __nv_bfloat16* __restrict__ dub,
                      const float* __restrict__ gsh,
                      const __nv_bfloat16* __restrict__ cb, float* __restrict__ out,
                      uint32_t n_tokens, uint32_t n_heads,
                      const uint32_t* __restrict__ vl_items = nullptr,
                      const __nv_bfloat16* __restrict__ qc = nullptr,
                      const __nv_bfloat16* __restrict__ kc = nullptr,
                      uint32_t n_k_heads = 0u) {
    constexpr uint32_t D = PD_DNC_D, C = PD_DNC_C, G = PD_DNC_G;
    // grid (D/G, H): the column group is blockIdx.x so a head's four CTAs
    // are adjacent in launch order. On GB10 (2 CTAs/SM, 192 CTAs = two
    // waves, 24 MB L2) the head-fastest order had each head's groups in
    // different waves, and the walk re-read the shared dw/q/k/coef items
    // from DRAM once per group: ~190 MB per layer at 273 GB/s is the 735
    // us it measured (the chain alone: 166; the unique bytes: ~57 MB).
    const uint32_t h = blockIdx.y, col0 = blockIdx.x * G;
    // varlen span items (GDN formulation band):
    // blockIdx.z picks the span; each item is (first launch chunk, span
    // rows, state f32 offset, out row0). Every operand is re-based here so
    // the whole walk body runs unchanged in span-local coordinates -
    // per-span results are bit-identical to the per-span launches, only
    // the grid packing differs. Legacy launches (gridDim.z == 1, nullptr)
    // take none of this.
    if (vl_items != nullptr) {
        const uint32_t* it = vl_items + (size_t)blockIdx.z * 4u;
        const uint32_t cb0 = it[0];
        n_tokens = it[1];
        state += (size_t)it[2];
        out += (size_t)it[3] * n_heads * D;
        if (QKD) {   // the span's rows in the tick's compact plane
            qc += (size_t)it[3] * n_k_heads * D;
            kc += (size_t)it[3] * n_k_heads * D;
        }
        qb += (size_t)cb0 * C * n_heads * D;
        kb += (size_t)cb0 * C * n_heads * D;
        dwb += (size_t)cb0 * n_heads * C * D;
        dub += (size_t)cb0 * n_heads * C * D;
        cb += (size_t)cb0 * n_heads * C * C;
        gsh += (size_t)cb0 * n_heads * 2u * C;
    }
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const uint32_t g8 = lane >> 2, t4 = lane & 3u;
    const uint32_t nc = (n_tokens + C - 1u) / C;
    const uint32_t cw = warp * 8u;  // warp's V-col slice within the CTA's G
    const uint32_t hk = QKD ? (h % n_k_heads) : 0u;   // the k-head this v-head shares

    extern __shared__ char shm_rs[];
    __shared__ float sh_w[C], sh_gam[C];
    __shared__ float sh_gall;

    // classic 8-item schedule (0..3 A-slab [dw;q] K-cols ty*32..+32, 4 du
    // G-slice, 5 coef, 6/7 raw-k a-halves), bf16 panes, 128-thread strides.
    // Empty commits past nc keep the wait counts aligned.
    auto issue_item = [&](uint32_t it) {
        const uint32_t ch = it / 8u;
#if PD_DNC_PROBE == 3
        if (false) {   // probe: no ring loads, the chain runs on stale panes
#else
        if (ch < nc) {
#endif
            const uint32_t ty = it % 8u;
            const uint32_t c0i = ch * C;
            const uint32_t cli = min(C, n_tokens - c0i);
            const size_t tbi = (size_t)ch * n_heads + h;
            char* pane = shm_rs + (size_t)(it % S) * PD_DNRS_SLOT_B;
            if (ty < 4u) {
                const uint32_t k0s = ty * 32u;
                __nv_bfloat16* wp = (__nv_bfloat16*)pane;
                for (uint32_t u = tid; u < C * 4u; u += 128u) {
                    const uint32_t r = u / 4u, ce = (u % 4u) * 8u;
                    pd_dnc_cpa16(wp + r * PD_DNRS_SW + ce,
                                 dwb + (tbi * C + r) * D + k0s + ce,
                                 r < cli ? 16u : 0u);
                }
                __nv_bfloat16* qp = (__nv_bfloat16*)(pane + C * PD_DNRS_SW * 2u);
#if PD_DNC_PROBE != 5
                for (uint32_t u = tid; u < C * 4u; u += 128u) {
                    const uint32_t i = u / 4u, ce = (u % 4u) * 8u;
#if PD_DNRS_QK_TOKENMAJOR
                    pd_dnc_cpa16(qp + i * PD_DNRS_SW + ce,
                                 qb + ((size_t)(c0i + i) * n_heads + h) * D + k0s + ce,
                                 i < cli ? 16u : 0u);
#else
                    pd_dnc_cpa16(qp + i * PD_DNRS_SW + ce,
                                 QKD ? qc + ((size_t)(c0i + i) * n_k_heads + hk) * D + k0s + ce
                                     : qb + (tbi * C + i) * D + k0s + ce,
                                 i < cli ? 16u : 0u);
#endif
                }
#else
                (void)qp;
#endif
            } else if (ty == 4u) {
                __nv_bfloat16* up = (__nv_bfloat16*)pane;
                for (uint32_t u = tid; u < C * (G / 8u); u += 128u) {
                    const uint32_t i = u / (G / 8u), ce = (u % (G / 8u)) * 8u;
                    pd_dnc_cpa16(up + i * PD_DNRS_SW + ce,
                                 dub + (tbi * C + i) * D + col0 + ce,
                                 i < cli ? 16u : 0u);
                }
            } else if (ty == 5u) {
                __nv_bfloat16* cp = (__nv_bfloat16*)pane;
                for (uint32_t u = tid; u < C * 8u; u += 128u) {
                    const uint32_t r = u / 8u, c8 = (u % 8u) * 8u;
                    uint32_t by = 0u;
                    if (r < cli && c8 < cli) by = min(16u, (cli - c8) * 2u);
                    pd_dnc_cpa16(cp + r * PD_DNRS_SK + c8, cb + (tbi * C + r) * C + c8, by);
                }
            } else {
                const uint32_t a0 = (ty - 6u) * 64u;
                __nv_bfloat16* kp = (__nv_bfloat16*)pane;
#if PD_DNC_PROBE != 5
                for (uint32_t u = tid; u < C * 8u; u += 128u) {
                    const uint32_t j = u / 8u, c8 = (u % 8u) * 8u;
#if PD_DNRS_QK_TOKENMAJOR
                    pd_dnc_cpa16(kp + j * PD_DNRS_SK + c8,
                                 kb + ((size_t)(c0i + j) * n_heads + h) * D + a0 + c8,
                                 j < cli ? 16u : 0u);
#else
                    pd_dnc_cpa16(kp + j * PD_DNRS_SK + c8,
                                 QKD ? kc + ((size_t)(c0i + j) * n_k_heads + hk) * D + a0 + c8
                                     : kb + (tbi * C + j) * D + a0 + c8,
                                 j < cli ? 16u : 0u);
#endif
                }
#else
                (void)a0; (void)kp;
#endif
            }
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    };
    auto ring_wait = [&] {
        if constexpr (S == 4u) asm volatile("cp.async.wait_group 3;" ::: "memory");
        else if constexpr (S == 3u) asm volatile("cp.async.wait_group 2;" ::: "memory");
        else asm volatile("cp.async.wait_group 1;" ::: "memory");
        __syncthreads();
    };

    // state S^T[a][c]: 8 m16n8 f32 acc tiles (m = a over 128, n = the
    // warp's 8 cols). Element (ma,e): a = ma*16+g8+8*(e>=2), c = 2*t4+(e&1).
    ST* s_head = state + (size_t)h * D * D;
    float st[8][4];
#pragma unroll
    for (uint32_t ma = 0; ma < 8u; ++ma)
#pragma unroll
        for (uint32_t e = 0; e < 4u; ++e) {
            const uint32_t a = ma * 16u + g8 + (e >= 2u ? 8u : 0u);
            const uint32_t c = col0 + cw + 2u * t4 + (e & 1u);
            st[ma][e] = pd_dns_ld(s_head + (size_t)c * D + a);
        }

    for (uint32_t i = 0; i < S; ++i) issue_item(i);
    __syncthreads();

    for (uint32_t ch = 0; ch < nc; ++ch) {
        const uint32_t c0 = ch * C;
        const uint32_t cl = min(C, n_tokens - c0);
        const size_t tb = (size_t)ch * n_heads + h;
        const uint32_t it0 = 8u * ch;

        // gate vectors come PRE-DIFFED as f32 from stage1's f64 cumsum:
        // the old per-read f64 loads + F2F.F64 were 6%+ of
        // the walk's stall samples. Same f64 subtract, same rounding,
        // same expf -> bit-identical gates.
        if (tid < C) {
            sh_w[tid] = tid < cl ? expf(gsh[tb * 2u * C + tid]) : 0.0f;
            sh_gam[tid] = tid < cl ? expf(gsh[tb * 2u * C + C + tid]) : 0.0f;
        }
        if (tid == 0) sh_gall = expf(gsh[tb * 2u * C + C + cl - 1u]);

        // ---- pass 1: dl = dw @ S0, o1 = q @ S0 - m = chunk rows (4 m16
        // tiles), n = the warp's 8 cols, k = a ascending. S0 comes from the
        // acc frags via the shfl conversion, once per k16 slab for all 8 mmas.
        float dl[4][4], o[4][4];
#pragma unroll
        for (uint32_t mi = 0; mi < 4u; ++mi)
#pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) { dl[mi][e] = 0.f; o[mi][e] = 0.f; }
#pragma unroll
        for (uint32_t slab = 0; slab < 4u; ++slab) {
            ring_wait();
            const char* pane = shm_rs + (size_t)((it0 + slab) % S) * PD_DNRS_SLOT_B;
            const __nv_bfloat16* wp = (const __nv_bfloat16*)pane;
            const __nv_bfloat16* qp = (const __nv_bfloat16*)(pane + C * PD_DNRS_SW * 2u);
#pragma unroll
            for (uint32_t kk = 0; kk < 32u; kk += 16u) {
                uint32_t br[2];
                pd_dnrs_cb16<false>(st[slab * 2u + (kk >> 4)], nullptr, br[0], br[1]);
#pragma unroll
                for (uint32_t mi = 0; mi < 4u; ++mi) {
                    uint32_t aw[4], aq[4];
                    pd_dnrs_lda(wp, PD_DNRS_SW, mi * 16u, kk, aw);
                    PD_DNRS_WALK_MMA(dl[mi], aw, br);
                    pd_dnrs_lda(qp, PD_DNRS_SW, mi * 16u, kk, aq);
                    PD_DNRS_WALK_MMA(o[mi], aq, br);
                }
            }
            __syncthreads();
            issue_item(it0 + slab + S);
        }

        // ---- delta = du - dl (explicit zero past cl) + the gam pre-scale
        // of o1 (classic order: scale before the coef chain).
        {
            ring_wait();  // du item it0+4
            const __nv_bfloat16* dup =
                (const __nv_bfloat16*)(shm_rs + (size_t)((it0 + 4u) % S) * PD_DNRS_SLOT_B);
#pragma unroll
            for (uint32_t mi = 0; mi < 4u; ++mi)
#pragma unroll
                for (uint32_t e = 0; e < 4u; ++e) {
                    const uint32_t i = mi * 16u + g8 + (e >= 2u ? 8u : 0u);
                    const uint32_t cc = cw + 2u * t4 + (e & 1u);
                    float d = 0.f;
                    if (i < cl)
                        d = __bfloat162float(dup[i * PD_DNRS_SW + cc]) - dl[mi][e];
                    dl[mi][e] = d;
                    o[mi][e] *= sh_gam[min(i, cl - 1u)];
                }
            __syncthreads();
            issue_item(it0 + 4u + S);
        }

        // ---- pass 2: o += coef @ deltaT, then the guarded writeback.
        {
            ring_wait();  // coef item it0+5
            const __nv_bfloat16* cf =
                (const __nv_bfloat16*)(shm_rs + (size_t)((it0 + 5u) % S) * PD_DNRS_SLOT_B);
#pragma unroll
            for (uint32_t j0 = 0; j0 < C; j0 += 16u) {
                uint32_t br[2];
                pd_dnrs_cb16<false>(dl[j0 >> 4], nullptr, br[0], br[1]);
#pragma unroll
                for (uint32_t mi = 0; mi < 4u; ++mi) {
                    uint32_t ac[4];
                    pd_dnrs_lda(cf, PD_DNRS_SK, mi * 16u, j0, ac);
                    PD_DNRS_WALK_MMA(o[mi], ac, br);
                }
            }
#pragma unroll
            for (uint32_t mi = 0; mi < 4u; ++mi)
#pragma unroll
                for (uint32_t e = 0; e < 4u; ++e) {
                    const uint32_t i = mi * 16u + g8 + (e >= 2u ? 8u : 0u);
                    const uint32_t c = col0 + cw + 2u * t4 + (e & 1u);
                    if (i < cl)
                        out[((size_t)(c0 + i) * n_heads + h) * D + c] = o[mi][e];
                }
            __syncthreads();
            issue_item(it0 + 5u + S);
        }

        // ---- hop: S^T = gall*S^T + kT @ (w o deltaT), one 64-a-col half
        // per k item, j ascending; w folds into the delta conversion so the
        // k pane rides raw ldmatrix.trans.
#pragma unroll
        for (uint32_t ma = 0; ma < 8u; ++ma)
#pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) st[ma][e] *= sh_gall;
#pragma unroll
        for (uint32_t half = 0; half < 2u; ++half) {
            ring_wait();  // k half item it0+6+half
            const __nv_bfloat16* kp =
                (const __nv_bfloat16*)(shm_rs +
                                       (size_t)((it0 + 6u + half) % S) * PD_DNRS_SLOT_B);
#pragma unroll
            for (uint32_t j0 = 0; j0 < C; j0 += 16u) {
                uint32_t br[2];
                pd_dnrs_cb16<true>(dl[j0 >> 4], sh_w + j0, br[0], br[1]);
#pragma unroll
                for (uint32_t mt = 0; mt < 4u; ++mt) {
                    uint32_t ak[4];
                    pd_dnrs_ldat(kp, PD_DNRS_SK, j0, mt * 16u, ak);
                    PD_DNRS_WALK_MMA(st[half * 4u + mt], ak, br);
                }
            }
            __syncthreads();
            issue_item(it0 + 6u + half + S);
        }
    }

    // final state writeback (in-place walk: entry == state)
#pragma unroll
    for (uint32_t ma = 0; ma < 8u; ++ma)
#pragma unroll
        for (uint32_t e = 0; e < 4u; ++e) {
            const uint32_t a = ma * 16u + g8 + (e >= 2u ? 8u : 0u);
            const uint32_t c = col0 + cw + 2u * t4 + (e & 1u);
            pd_dns_st(s_head + (size_t)c * D + a, st[ma][e]);
        }
}

// ==================== stage1 RS (bf16-operand rebuild) =====================

// One launch helper for the three launch sites: the small-die shape is a
// per-die election, PADDOCK_NO_DNRS_ONEWAVE=1 pins the big-die shape.
static inline bool pd_dnrs_small_die() {
    static int nsm = 0;
    if (nsm == 0) { int d = 0; cudaGetDevice(&d); cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, d); if (nsm <= 0) nsm = 148; }
    static const bool off = pd_env("PADDOCK_NO_DNRS_ONEWAVE") != nullptr;
    return nsm < 128 && !off;
}
template <typename ST, bool QKD>
static inline void pd_dnrs_walk_go(dim3 gw, cudaStream_t s, const __nv_bfloat16* qb,
                                   const __nv_bfloat16* kb, ST* state,
                                   const __nv_bfloat16* dw, const __nv_bfloat16* du,
                                   const float* gsh, const __nv_bfloat16* cb, float* out,
                                   uint32_t n_tokens, uint32_t n_heads,
                                   const uint32_t* items, const __nv_bfloat16* qc,
                                   const __nv_bfloat16* kc, uint32_t n_k_heads) {
    if (pd_dnrs_small_die())
        pd_dnc_walk_rs_kernel<ST, QKD, 2u, 4><<<gw, 128, 2u * PD_DNRS_SLOT_B, s>>>(
            qb, kb, state, dw, du, gsh, cb, out, n_tokens, n_heads, items, qc, kc, n_k_heads);
    else
        pd_dnc_walk_rs_kernel<ST, QKD, 4u, 1><<<gw, 128, 4u * PD_DNRS_SLOT_B, s>>>(
            qb, kb, state, dw, du, gsh, cb, out, n_tokens, n_heads, items, qc, kc, n_k_heads);
}

// pd_dnc_stage1_rs_kernel (dnc chunk rung): the walk's own
// operand-class transformation applied to its un-rebuilt sibling. Profiling
// stage1_v2 at PREC=1 (B200): issue-starved (0.47 inst/cycle,
// 53% no-eligible), L1/TEX 60% on the scalar fragment loads, DRAM 7%,
// compute 42% - the classic format's 6 cvt.rna.tf32 + 6 scalar lds per k8
// mma is the wall, not tensor math (a fused kkt_solve+wy does the same work
// at bf16 in a single pass). This kernel:
//   - q/k staged once as bf16 [C][SKB] panes (no 4-round slab ring; the k
//     pane stays RESIDENT through the dw pass - no restage there either)
//   - dots + dw/du chains as m16n8k16 bf16 mma, A via ldmatrix.x4
//     (pd_dnrs_lda), B via ldmatrix.x2 (new pd_dnrs_ldb / _ldbt below), f32
//     accumulators throughout - the reference's exact operand class for these
//     products (their k/w/u/A are bf16); acc fragment layout is identical
//     to the k8 form, so the M/aqk emission and store code carry over
//     verbatim from stage1_v2
//   - T = (I+M)^-1 build UNCHANGED (scalar f32, stage1_v2's hierarchical
//     code verbatim; M now comes from bf16-rounded dots - the class change)
//   - dw/du A-operand = T packed-upper folded with the per-token scale into
//     a bf16 pane before rounding: Tw[i][j] = bf16(T[i][j] * scale[j]) -
//     the scale multiplies in f32 exactly as the classic B-load fold did
//   - f64 g-cumsum, gsh gate vectors, beta/bg: stage1_v2 verbatim (cg and
//     gsh outputs BIT-IDENTICAL; qb16/kb16 = pane copies, bit-identical
//     rounds of the same f32 words)
// Outputs are the RS-route set only (OT/AT = bf16, qb16/kb16/gsh required).
// Not bit-exact vs stage1_v2 (bf16-rounded products); gates: proto band,
// PPL-distance, greedy fork-shape, suite, serve A/B - the walk election
// precedent. Kill: route env PADDOCK_DNC_S1RS.

// B-frag (m16n8k16, k16 x n8) from a row-major [n][k] bf16 pane: piece p
// covers k in [k0+8p, k0+8p+8); non-trans ldmatrix rows = the 8 n-rows.
static __device__ __forceinline__ void pd_dnrs_ldb(const __nv_bfloat16* base,
                                                   uint32_t stride_h, uint32_t n0,
                                                   uint32_t k0, uint32_t (&b)[2]) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row = n0 + (lane & 7u);
    const uint32_t col = k0 + (((lane >> 3) & 1u) << 3);
    const unsigned sm = (unsigned)__cvta_generic_to_shared(base + row * stride_h + col);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                 : "=r"(b[0]), "=r"(b[1]) : "r"(sm));
}

// Transposed B-frag: B[k][n] = pane[k][n] with the pane row-major in k
// (rows = the 16 k-rows, cols = n) - trans ldmatrix flips each 8x8 piece.
static __device__ __forceinline__ void pd_dnrs_ldbt(const __nv_bfloat16* base,
                                                    uint32_t stride_h, uint32_t k0,
                                                    uint32_t n0, uint32_t (&b)[2]) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row = k0 + (lane & 15u);
    const unsigned sm = (unsigned)__cvta_generic_to_shared(base + row * stride_h + n0);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                 : "=r"(b[0]), "=r"(b[1]) : "r"(sm));
}

// pane strides in halves: 272 B rows (mod 128 = 16) put consecutive rows 4
// banks apart -> 8-row ldmatrix blocks span 32 banks conflict-free (the
// walk's 144 B pattern, same residue class).
#define PD_DNS1RS_SKB 136u
#define PD_DNS1RS_STB 72u
#define PD_DNS1RS_SMEM                                                         \
    ((2u * PD_DNC_C * PD_DNS1RS_SKB + PD_DNC_C * PD_DNS1RS_STB) * 2u +         \
     PD_DNC_C * PD_DNS1_SMT * 4u + PD_DNC_C * 8u)

// occupancy: 63.0 KB smem -> 3 CTAs/SM (189 of 227 KB); the thin bf16
// instruction stream exposed latency at 2 CTAs (77%
// no-eligible, 15.4 cyc/inst at 3.5 warps/scheduler) - the third CTA is
// the latency-hiding lever, and it also interleaves other blocks' mma over
// this block's serial T-build phases.
// PH (proto-only phase ladder): 0 = full kernel (production); 1 = stop
// after staging+gates, 2 = after dots+emission, 3 = after T-build,
// 4 = after the dw pass. Early returns are dead code at PH=0.
// QKC: q/k arrive COMPACT bf16 [rows, n_k_heads, D] from the conv's qkc
// twin (one bf16 round of the same f32 the false-arm rounds here itself -
// panes and every downstream word BIT-IDENTICAL); staging halves to four
// 16B copies per thread with the 3 sharing v-heads riding L2.
template <typename VT = float, int PH = 0, bool QKC = false>
__global__ void __launch_bounds__(256, 3)
pd_dnc_stage1_rs_kernel(const float* __restrict__ q, const float* __restrict__ k,
                        const VT* __restrict__ v, const float* __restrict__ g,
                        const float* __restrict__ beta,
                        __nv_bfloat16* __restrict__ dw,
                        __nv_bfloat16* __restrict__ du,
                        __nv_bfloat16* __restrict__ aqk, double* __restrict__ cg,
                        uint32_t n_tokens, uint32_t n_heads,
                        __nv_bfloat16* __restrict__ qb16,
                        __nv_bfloat16* __restrict__ kb16,
                        float* __restrict__ gsh,
                        const uint32_t* __restrict__ vl_items = nullptr,
                        uint32_t n_k_heads = 0) {
    constexpr uint32_t D = PD_DNC_D, C = PD_DNC_C;
    constexpr uint32_t SKB = PD_DNS1RS_SKB, STB = PD_DNS1RS_STB;
    constexpr uint32_t SMT = PD_DNS1_SMT;
    static_assert(D == 128u && C == 64u, "stage1_rs assumes D=128, C=64");
    const uint32_t ch = blockIdx.x, h = blockIdx.y;
    uint32_t c0, cl;
    if (vl_items != nullptr) {
        c0 = vl_items[ch * 2u];
        cl = vl_items[ch * 2u + 1u];
    } else {
        c0 = ch * C;
        cl = min(C, n_tokens - c0);
    }
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const uint32_t g8 = lane >> 2, t4 = lane & 3u;
    const size_t tb = (size_t)ch * n_heads + h;

    extern __shared__ float sh1[];
    __nv_bfloat16* sh_kb = (__nv_bfloat16*)sh1;        // [C][SKB] k pane
    __nv_bfloat16* sh_qb = sh_kb + C * SKB;            // [C][SKB] q pane; T-build
                                                       // scratch + v pane later
    __nv_bfloat16* sh_tb = sh_qb + C * SKB;            // [C][STB] scaled-T pane
    float* sh_mt = (float*)(sh_tb + C * STB);          // [C][SMT] M lower/T upper
    double* sh_cg = (double*)(sh_mt + C * SMT);        // [C]
    // merge scratch lives in the sh_tb region (dead until the dw pass) so
    // the q pane can take the v rows DURING T-build (the v global loads
    // issue before the build and store after - latency hidden behind it)
    float* sh_w32 = (float*)sh_tb;
    __shared__ float sh_b[PD_DNC_C], sh_bg[PD_DNC_C];

    // f32 rows -> bf16 pane, zero-filled past cl. All 8 global loads issue
    // before any convert/store: the naive fused loop serialized 8 dependent
    // ld->cvt->sts chains per thread and the staging phase alone measured
    // 53.8 us of the 140 (phase ladder) - batching pays one
    // global latency instead of eight.
    auto stage_pane = [&](__nv_bfloat16* pane, const float* src) {
        float4 f[8];
#pragma unroll
        for (uint32_t e = 0; e < 8u; ++e) {
            const uint32_t u = tid + e * 256u;
            const uint32_t r = u >> 5, c4 = (u & 31u) * 4u;
            f[e] = make_float4(0.f, 0.f, 0.f, 0.f);
            if (r < cl)
                f[e] = *(const float4*)(src +
                                        ((size_t)(c0 + r) * n_heads + h) * D + c4);
        }
#pragma unroll
        for (uint32_t e = 0; e < 8u; ++e) {
            const uint32_t u = tid + e * 256u;
            const uint32_t r = u >> 5, c4 = (u & 31u) * 4u;
            const __nv_bfloat162 p0 = __floats2bfloat162_rn(f[e].x, f[e].y);
            const __nv_bfloat162 p1 = __floats2bfloat162_rn(f[e].z, f[e].w);
            *(__nv_bfloat162*)(pane + r * SKB + c4) = p0;
            *(__nv_bfloat162*)(pane + r * SKB + c4 + 2u) = p1;
        }
    };

    // compact bf16 rows -> pane, zero-filled past cl (QKC arm)
    auto stage_pane_c = [&](__nv_bfloat16* pane, const __nv_bfloat16* src) {
        const uint32_t hk = h % n_k_heads;
        uint4 f[4];
#pragma unroll
        for (uint32_t e = 0; e < 4u; ++e) {
            const uint32_t u = tid + e * 256u;
            const uint32_t r = u >> 4, c8 = (u & 15u) * 8u;
            f[e] = make_uint4(0u, 0u, 0u, 0u);
            if (r < cl)
                f[e] = *(const uint4*)(src +
                                       ((size_t)(c0 + r) * n_k_heads + hk) * D + c8);
        }
#pragma unroll
        for (uint32_t e = 0; e < 4u; ++e) {
            const uint32_t u = tid + e * 256u;
            *(uint4*)(pane + (u >> 4) * SKB + (u & 15u) * 8u) = f[e];
        }
    };

    // g load + f64 cumsum (thread 0) - stage1_v2 verbatim; overlapped with
    // the pane staging below (the panes barrier after).
    if (tid < C) sh_b[tid] = tid < cl ? g[(size_t)(c0 + tid) * n_heads + h] : 0.f;
    __syncthreads();
    if constexpr (QKC) {
        stage_pane_c(sh_kb, (const __nv_bfloat16*)k);
        stage_pane_c(sh_qb, (const __nv_bfloat16*)q);
    } else {
        stage_pane(sh_kb, k);
        stage_pane(sh_qb, q);
    }
    if (tid == 0) {
        double run = 0.0;
        for (uint32_t i = 0; i < C; ++i) {
            if (i < cl) run += (double)sh_b[i];
            sh_cg[i] = i < cl ? run : 0.0;
            if (i < cl) cg[tb * C + i] = run;
        }
    }
    __syncthreads();
    if (tid < C) {
        const float b = tid < cl ? beta[(size_t)(c0 + tid) * n_heads + h] : 0.f;
        sh_b[tid] = b;
        sh_bg[tid] = tid < cl ? b * __expf((float)sh_cg[tid]) : 0.f;
    }
    if (tid < C) {
        const double cgl = sh_cg[cl - 1u];
        gsh[tb * 2u * C + tid] = tid < cl ? (float)(cgl - sh_cg[tid]) : 0.f;
        gsh[tb * 2u * C + C + tid] = tid < cl ? (float)sh_cg[tid] : 0.f;
    }
    // ORDER the gates phase (the sh_b beta overwrite + sh_bg) before the
    // dots' M-emission reads. stage1_v2 never needed this barrier explicitly
    // - its 4-round cp.async staging loop carried barriers that incidentally
    // ordered the pair; the single-shot panes here removed them and the
    // emission's sh_b[i] reads raced the overwrite (racecheck:
    // warp-level RAW, writer lane w -> reader lane 4w = the g8 index map;
    // canary ~100K dw words/launch, tiny-exponent class).
    __syncthreads();
    if constexpr (PH == 1) return;

    // dots: warps 0-3 akk (A = k pane), 4-7 aqk (A = q pane); B = kT via
    // non-trans x2 off the k pane. One A-frag per k16 step feeds all 8
    // n-tiles. acc fragment layout == the k8 form -> emission verbatim.
    {
        const uint32_t mt = warp & 3u;
        const __nv_bfloat16* apane = warp < 4u ? sh_kb : sh_qb;
        float acc[8][4];
#pragma unroll
        for (uint32_t nt = 0; nt < 8u; ++nt)
#pragma unroll
            for (uint32_t e = 0; e < 4; ++e) acc[nt][e] = 0.f;
#pragma unroll
        for (uint32_t kk = 0; kk < D; kk += 16u) {
            uint32_t af[4];
            pd_dnrs_lda(apane, SKB, mt * 16u, kk, af);
#pragma unroll
            for (uint32_t nt = 0; nt < 8u; ++nt) {
                uint32_t bf[2];
                pd_dnrs_ldb(sh_kb, SKB, nt * 8u, kk, bf);
                pd_dnrs_mma16(acc[nt], af, bf);
            }
        }
        // emission: stage1_v2 verbatim (aqk with decay ratio to global as
        // bf16 pairs; M = b_i ratio akk strict-lower into sh_mt).
#pragma unroll
        for (uint32_t nt = 0; nt < 8u; ++nt) {
#pragma unroll
            for (uint32_t ep = 0; ep < 2u; ++ep) {
                const uint32_t i = mt * 16u + g8 + (ep ? 8u : 0u);
                const uint32_t j = nt * 8u + 2u * t4;
                const float r0 =
                    j <= i ? __expf((float)(sh_cg[i] - sh_cg[j])) : 0.f;
                const float r1 =
                    j + 1u <= i ? __expf((float)(sh_cg[i] - sh_cg[j + 1u])) : 0.f;
                const float v0 = r0 * acc[nt][2u * ep];
                const float v1 = r1 * acc[nt][2u * ep + 1u];
                if (warp >= 4u)
                    *reinterpret_cast<pd_dns1_pair<__nv_bfloat16>*>(
                        aqk + tb * C * C + i * C + j) =
                        pd_dns1_pair<__nv_bfloat16>{(__nv_bfloat16)v0,
                                                    (__nv_bfloat16)v1};
                else {
                    if (j < i) sh_mt[i * SMT + j] = sh_b[i] * v0;
                    if (j + 1u < i) sh_mt[i * SMT + j + 1u] = sh_b[i] * v1;
                }
            }
        }
    }
    // qb16/kb16: pure pane copies (the same f32->bf16 rounds the panes
    // already hold). Runs pre-barrier: reads race nothing (panes stable,
    // sh_mt writes disjoint), and the q pane dies right after.
    // qb16 == nullptr: the walk reads the compact plane itself (QKD, the
    // qkc varlen route on small dies) and the 25 MB emission is skipped.
    for (uint32_t u = tid; u < C * (D / 8u) && qb16 != nullptr; u += 256u) {
        const uint32_t r = u >> 4, c8 = (u & 15u) * 8u;
        if (r < cl) {
            // CHUNK-HEAD-MAJOR (2026-09-08, GB10): the copies used to sit
            // token-major ([launch row][H][D]), so the walk's per-chunk q
            // slabs and k halves were 64-byte gathers at a 12 KB stride -
            // fine on a 128 MB L2 that held the whole working set, but on a
            // 24 MB L2 the walk read them from DRAM at ~100 GB/s and its
            // 735 us was 570 us of ring loads (the chain alone runs in 166).
            // Same bytes, [chunk*H + h][C][D] like dw/du: every walk item is
            // one contiguous 16 KB block. PD_DNRS_QK_TOKENMAJOR=1 keeps the
            // old layout for A/B (bench only).
#if PD_DNRS_QK_TOKENMAJOR
            const size_t dst = ((size_t)(ch * C + r) * n_heads + h) * D + c8;
#else
            const size_t dst = ((size_t)(ch * n_heads + h) * C + r) * D + c8;
#endif
            *(uint4*)(kb16 + dst) = *(const uint4*)(sh_kb + r * SKB + c8);
            *(uint4*)(qb16 + dst) = *(const uint4*)(sh_qb + r * SKB + c8);
        }
    }
    // v rows -> registers now, stored into the (dead) q pane after the
    // T-build: the global latency hides behind the build's compute.
    float4 vf[8];
    uint4 vb[4];
    if (sizeof(VT) == 2u) {
#pragma unroll
        for (uint32_t e = 0; e < 4u; ++e) {
            const uint32_t u = tid + e * 256u;
            const uint32_t r = u >> 4, c8 = (u & 15u) * 8u;
            vb[e] = make_uint4(0u, 0u, 0u, 0u);
            if (r < cl)
                vb[e] = *(const uint4*)((const __nv_bfloat16*)v +
                                        ((size_t)(c0 + r) * n_heads + h) * D + c8);
        }
    } else {
#pragma unroll
        for (uint32_t e = 0; e < 8u; ++e) {
            const uint32_t u = tid + e * 256u;
            const uint32_t r = u >> 5, c4 = (u & 31u) * 4u;
            vf[e] = make_float4(0.f, 0.f, 0.f, 0.f);
            if (r < cl)
                vf[e] = *(const float4*)((const float*)v +
                                         ((size_t)(c0 + r) * n_heads + h) * D + c4);
        }
    }
    __syncthreads();
    if constexpr (PH == 2) return;

    // T = (I+M)^-1 - stage1_v2 verbatim (base 16x16 register-resident
    // solves + two merge levels; merge scratch overlays the dead q pane).
    if (tid < C) {
        const uint32_t cl0 = tid & 15u, c = (tid >> 4) * 16u + cl0;
        float a[15];
#pragma unroll
        for (uint32_t e = 0; e < 15u; ++e)
            a[e] = cl0 + 1u + e < 16u ? -sh_mt[(c + 1u + e) * SMT + c] : 0.f;
#pragma unroll
        for (uint32_t s = 1; s < 15u; ++s) {
            const float tv = a[s - 1u];
#pragma unroll
            for (uint32_t e = s; e < 15u; ++e)
                if (cl0 + 1u + e < 16u)
                    a[e] = fmaf(-sh_mt[(c + 1u + e) * SMT + c + s], tv, a[e]);
        }
#pragma unroll
        for (uint32_t e = 0; e < 15u; ++e)
            if (cl0 + 1u + e < 16u) sh_mt[c * SMT + c + 1u + e] = a[e];
    }
    __syncthreads();
    {
        const uint32_t pair = tid >> 7, r = (tid >> 4) & 7u, c = tid & 15u;
        const uint32_t bg = pair * 32u, rg = bg + 16u + r;
        for (uint32_t u = tid; u < 512u; u += 256u) {
            const uint32_t b = u >> 8, j = (u >> 4) & 15u, cc = u & 15u;
            const float tv = sh_mt[(b * 32u + cc) * SMT + b * 32u + j];
            sh_w32[b * 272u + j * 17u + cc] = j > cc ? tv : 0.f;
        }
        __syncthreads();
        float w[2];
#pragma unroll
        for (uint32_t rr = 0; rr < 2u; ++rr)
            w[rr] = sh_mt[(rg + rr * 8u) * SMT + bg + c];
#pragma unroll
        for (uint32_t j = 0; j < 16u; ++j) {
            const float tv = sh_w32[pair * 272u + j * 17u + c];
#pragma unroll
            for (uint32_t rr = 0; rr < 2u; ++rr) {
                const float m = sh_mt[(rg + rr * 8u) * SMT + bg + j];
                w[rr] = j > c ? fmaf(m, tv, w[rr]) : w[rr];
            }
        }
#pragma unroll
        for (uint32_t rr = 0; rr < 2u; ++rr)
            sh_w32[(rg + rr * 8u) * 36u + c] = w[rr];
        __syncthreads();
        float t2[2] = {w[0], w[1]};
        for (uint32_t rj = bg + 16u; rj < rg; ++rj) {
            const float wv = sh_w32[rj * 36u + c];
            t2[0] = fmaf(sh_mt[rj * SMT + rg], wv, t2[0]);
            t2[1] = fmaf(sh_mt[rj * SMT + rg + 8u], wv, t2[1]);
        }
#pragma unroll
        for (uint32_t kk = 0; kk < 8u; ++kk) {
            const uint32_t rj = rg + kk;
            t2[1] = fmaf(sh_mt[rj * SMT + rg + 8u], sh_w32[rj * 36u + c], t2[1]);
        }
        sh_mt[(bg + c) * SMT + rg] = -t2[0];
        sh_mt[(bg + c) * SMT + rg + 8u] = -t2[1];
    }
    __syncthreads();
    {
        const uint32_t r = tid >> 5, c = tid & 31u;
        for (uint32_t u = tid; u < 1024u; u += 256u) {
            const uint32_t j = u >> 5, cc = u & 31u;
            const float tv = sh_mt[cc * SMT + j];
            sh_w32[j * 33u + cc] = j > cc ? tv : 0.f;
        }
        __syncthreads();
        float w[4];
#pragma unroll
        for (uint32_t rr = 0; rr < 4u; ++rr)
            w[rr] = sh_mt[(32u + r + rr * 8u) * SMT + c];
#pragma unroll
        for (uint32_t j = 0; j < 32u; ++j) {
            const float tv = sh_w32[j * 33u + c];
#pragma unroll
            for (uint32_t rr = 0; rr < 4u; ++rr) {
                const float m = sh_mt[(32u + r + rr * 8u) * SMT + j];
                w[rr] = j > c ? fmaf(m, tv, w[rr]) : w[rr];
            }
        }
#pragma unroll
        for (uint32_t rr = 0; rr < 4u; ++rr)
            sh_w32[(32u + r + rr * 8u) * 36u + c] = w[rr];
        __syncthreads();
        float tt4[4] = {w[0], w[1], w[2], w[3]};
        for (uint32_t rj = 32u; rj < 32u + r; ++rj) {
            const float wv = sh_w32[rj * 36u + c];
#pragma unroll
            for (uint32_t rr = 0; rr < 4u; ++rr)
                tt4[rr] = fmaf(sh_mt[rj * SMT + 32u + r + rr * 8u], wv, tt4[rr]);
        }
#pragma unroll
        for (uint32_t s = 0; s < 3u; ++s)
#pragma unroll
            for (uint32_t kk = 0; kk < 8u; ++kk) {
                const uint32_t rj = 32u + r + s * 8u + kk;
                const float wv = sh_w32[rj * 36u + c];
#pragma unroll
                for (uint32_t rr = s + 1u; rr < 4u; ++rr)
                    tt4[rr] = fmaf(sh_mt[rj * SMT + 32u + r + rr * 8u], wv, tt4[rr]);
            }
#pragma unroll
        for (uint32_t rr = 0; rr < 4u; ++rr)
            sh_mt[c * SMT + 32u + r + rr * 8u] = -tt4[rr];
    }
    __syncthreads();
    if constexpr (PH == 3) return;
    if constexpr (PH == 6) {
        // debug: raw T words (sh_mt rows 0-56) and sh_bg
        for (uint32_t u = tid; u < 4096u; u += 256u)
            ((uint32_t*)dw)[tb * 4096u + u] = *(const uint32_t*)(sh_mt + u);
        if (tid < C) ((uint32_t*)du)[tb * C + tid] = __float_as_uint(sh_bg[tid]);
        return;
    }

    // v pane store (loads issued pre-T-build)
    if (sizeof(VT) == 2u) {
#pragma unroll
        for (uint32_t e = 0; e < 4u; ++e) {
            const uint32_t u = tid + e * 256u;
            *(uint4*)(sh_qb + (u >> 4) * SKB + (u & 15u) * 8u) = vb[e];
        }
    } else {
#pragma unroll
        for (uint32_t e = 0; e < 8u; ++e) {
            const uint32_t u = tid + e * 256u;
            const uint32_t r = u >> 5, c4 = (u & 31u) * 4u;
            *(__nv_bfloat162*)(sh_qb + r * SKB + c4) =
                __floats2bfloat162_rn(vf[e].x, vf[e].y);
            *(__nv_bfloat162*)(sh_qb + r * SKB + c4 + 2u) =
                __floats2bfloat162_rn(vf[e].z, vf[e].w);
        }
    }

    // dw = (T diag(bg)) K and du = (T diag(b)) V: the scale folds into the
    // bf16 T pane in F32 (exactly where the classic B-load fold applied it);
    // A = Tw via ldmatrix, B = the resident k pane (trans) / the v pane.
    for (uint32_t pass = 0; pass < 2u; ++pass) {
        const float* scale = pass == 0u ? sh_bg : sh_b;
        __nv_bfloat16* dst = pass == 0u ? dw : du;
        for (uint32_t u = tid; u < C * (C / 2u); u += 256u) {
            const uint32_t i = u >> 5, j2 = (u & 31u) * 2u;
            const float s0 = scale[j2], s1 = scale[j2 + 1u];
            const float tv0 =
                j2 < i ? sh_mt[j2 * SMT + i] : (j2 == i ? 1.f : 0.f);
            const float tv1 =
                j2 + 1u < i ? sh_mt[(j2 + 1u) * SMT + i] : (j2 + 1u == i ? 1.f : 0.f);
            *(__nv_bfloat162*)(sh_tb + i * STB + j2) =
                __floats2bfloat162_rn(tv0 * s0, tv1 * s1);
        }
        __syncthreads();
        if constexpr (PH == 5) {
            // debug: dump the emitted Tw pane words (plain LDS) and stop
            if (pass == 0u) {
                for (uint32_t u = tid; u < C * C; u += 256u) {
                    const uint32_t i = u >> 6, j = u & 63u;
                    ((uint16_t*)dw)[tb * C * C + u] =
                        *(const uint16_t*)(sh_tb + i * STB + j);
                }
            }
            return;
        }
        const uint32_t mt = warp & 3u, nh = warp >> 2;
        float acc[8][4];
#pragma unroll
        for (uint32_t nt = 0; nt < 8u; ++nt)
#pragma unroll
            for (uint32_t e = 0; e < 4; ++e) acc[nt][e] = 0.f;
#pragma unroll
        for (uint32_t kk = 0; kk < C; kk += 16u) {
            // A-frag by four scalar 4B loads (bf16 pairs are the m16n8k16
            // fragment registers): the ldmatrix.x4 read of this pane raced
            // the bf162 emit stores above run-to-run (canary:
            // ~100K dw words/launch differed, du clean, all inputs proven
            // stable - E1 extra barrier and E3 pass-order swap both null;
            // scalar loads read the same pane race-free). Same op count
            // class; the dots' panes (cp.async-fed) keep ldmatrix.
            const uint32_t ar = mt * 16u + g8, ac = kk + 2u * t4;
            uint32_t af[4];
            af[0] = *(const uint32_t*)(sh_tb + ar * STB + ac);
            af[1] = *(const uint32_t*)(sh_tb + (ar + 8u) * STB + ac);
            af[2] = *(const uint32_t*)(sh_tb + ar * STB + ac + 8u);
            af[3] = *(const uint32_t*)(sh_tb + (ar + 8u) * STB + ac + 8u);
            const __nv_bfloat16* bp = pass == 0u ? sh_kb : sh_qb;
#pragma unroll
            for (uint32_t nt = 0; nt < 8u; ++nt) {
                uint32_t bf[2];
                pd_dnrs_ldbt(bp, SKB, kk, nh * 64u + nt * 8u, bf);
                pd_dnrs_mma16(acc[nt], af, bf);
            }
        }
#pragma unroll
        for (uint32_t nt = 0; nt < 8u; ++nt) {
#pragma unroll
            for (uint32_t ep = 0; ep < 2u; ++ep) {
                const uint32_t i = mt * 16u + g8 + (ep ? 8u : 0u);
                const uint32_t a = nh * 64u + nt * 8u + 2u * t4;
                if (i < cl)
                    *reinterpret_cast<pd_dns1_pair<__nv_bfloat16>*>(
                        dst + (tb * C + i) * D + a) =
                        pd_dns1_pair<__nv_bfloat16>{(__nv_bfloat16)acc[nt][2u * ep],
                                                    (__nv_bfloat16)acc[nt][2u * ep + 1u]};
            }
        }
        __syncthreads();
    }
}
