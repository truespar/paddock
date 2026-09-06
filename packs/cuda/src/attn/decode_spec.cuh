// attn/decode_spec.cuh - speculative-decode verify attention.
// Textually-included segment of the single pack translation unit.
// Not standalone-compilable: include order is defined by ../pack.cu.
//
// Split out of attn/decode.cuh (8216 lines against the ~2500 line ceiling),
// cut on the file's own section markers.
//
// Holds the spec-verify GQA walk (one block per (kv-head x q-sub-group,
// slot-CHUNK, split), so KV traffic scales with rows/k1 rather than rows),
// the FA-lite spec-verify tile, and the two spec-FA krs fp8-resident-K arms.
//
// Include after attn/decode.cuh - uses its ldm/mma/cpa helpers and partial
// layout. Nothing here is used by decode.cuh; the dependency runs one way.
// ── Spec-verify GQA walk (wide-batch speculative decoding) ──────────────────
// One block per (kv-head × q-sub-group, slot-CHUNK, split) walks the KV pool
// once for a chunk's k1 CONSECUTIVE verify rows (same slot, positions
// start..start+k1-1). The per-row GQA kernel made 160-row verify ticks
// attention-bound - 22% of the c32 GPU, more than the whole GEMM walk -
// because KV traffic scales with ROWS; here it scales with rows/k1 (SWA
// layers, full q-group fusion) or rows·gsub/(k1·G) (global layers).
//
// Per-row causality/window: scores mask to -inf outside [first_pos_j, pos_j]
// before the max fold, so a row never sees its successors' in-flight K/V and
// the SWA window slides per row. Masked terms add exp(-inf)=0 weight - the
// unmasked terms' fold order per row matches the per-row kernel's walk over
// its own range tile-by-tile (same TILE grid relative to lo0, which for the
// SWA arm may shift tile boundaries vs the per-row kernel's first_pos_j -
// same numeric CLASS as a split-count change, arbitrated like any lane
// change; greedy picks are argmax-stable under it in the gate).
//
// Q for k1×gsub vectors sits in smem (PD_GQA_SMEM with G:=k1*gsub); the
// (row,head) softmax states generalize the per-head warp fold to a strided
// loop (nwarps may be < k1*gsub). Accumulators: K1MAX-unrolled register
// array, guard j<nrows keeps indexing static.
// ldmatrix.x4: one warp instruction loads four 8x8 b16 tiles (512 B) from
// shared - the fragment-load workhorse (technique from CUTLASS/FlashAttention;
// per-lane LDS.32 fragment loads left the kernel LDS-pipe-bound at ~52% of
// the MMA issue peak). `p` is this lane's row pointer: lanes 0-7 address
// matrix 0's rows, 8-15 matrix 1, etc; the result distribution per matrix is
// the standard 8-bit fragment layout (row = lane>>2, bytes 4*(lane&3)..+3).
// ldmatrix is sm_75+, so this lives outside the PD_BS_OK block: the f8w8
// family (PD_F8W8_OK, sm_89+) consumes it on non-120a passes too.
__device__ __forceinline__ void pd_ldm_x4(uint32_t r[4], const unsigned char* p) {
    const unsigned sp = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(sp));
}

// ── FA-lite spec-verify attention (B200 bring-up) ───────────────────────
// f16-mma scores AND o-update with fragment reuse; full G-head fusion per
// kv head (no gs-subgroup KV duplication); distributed softmax; 3 barriers
// per 32-position tile. Replaces the smem-bandwidth-bound walk of
// pd_attn_spec_gqa_paged where eligible (f16 KV, hd<=512, k1<=8):
// a controlled harness measured 2.2x on the hd512 global
// class, 1.3x swa, maxrel vs the old kernel 0.21 - the f16-score class
// (llama.cpp flash-attn's class; KV is f16 already). Numeric-class change
// arbitrated by the serving gates. Lesson baked in: o-frag indexing must
// be fully static (a runtime task index moved the accumulators to local
// memory and halved the win).
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
#define PD_FA_OK 1
#else
#define PD_FA_OK 0
#endif

// krs decode occupancy target (B200 c32). Profiling the elected q36
// arm: 176 regs/thread x 256 threads = 45056 of the SM's 65536, i.e. One CTA
// resident -- sm__warps_active 11.71% of peak (8 of 64 warp slots) while the
// grid offers 13.84 waves, dram sits at 0.70% and sm__throughput at 27.6%.
// The kernel is neither bandwidth- nor issue-bound; it simply cannot place
// the parallelism it already has. __launch_bounds__'s second argument is what
// licenses 176: telling ptxas one block per SM suffices lets it spend freely.
// Raising it caps regs at 65536/(256*OCC) and buys a second/third resident
// CTA, paying in spills. Swept per die; 1 is the original behaviour.
// MEASURED on the qwen3.6-27b c32 leg: OCC=1 -> 176 regs, warps_active
// 11.71%, kernel 75.4 us; OCC=2 -> 128 regs, 22.21%, 52.6 us; OCC=3 is a
// wash at c32 and slightly worse at c1, so 2 is
// the elected rung. Shared memory was measured innocent -- see the carveout
// note in pd_spec_fa_krs_go. Numerics unaffected: OCC=1 and OCC=2 score
// wikitext-2 bit-identically at 1.72805 / 5.62968 / 0.5994, and the c32
// column-correspondence probe is 32/32 on-topic at both.
#ifndef PD_FA_KRS_OCC
#define PD_FA_KRS_OCC 2
#endif

// GV-arm adaptive-split target: tiles of PT keys each live CTA should own.
// The original 4 was tuned at ctx>=300 against the graph-baked 32 splits
// (1 tile/CTA, prelude-dominated). But at kv<=256 it collapses the grid to
// s_eff=2 - 256 live CTAs on a 148-SM die (<2/SM, 22% warps) against a ~2us
// DRAM floor, and a surplus-shrink capture showed the kernel
// wall is set by exactly these live CTAs. Lower target = more live CTAs at
// short kv; inert at long ctx (s_eff already clamps at n_splits).
#ifndef PD_FA_KRS_TPC
#define PD_FA_KRS_TPC 4
#endif

__device__ __forceinline__ void pd_fa_mma16(float d[4], uint32_t a0, uint32_t a1,
                                            uint32_t a2, uint32_t a3, uint32_t b0,
                                            uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// PT: positions/tile (16 at hd512 for smem, 32 at hd256). OI: o float2 items
// per thread = M*hd/2/256.
// DB: double-buffered KV stage ring (the B200 shape). DB=false single-
// buffers the ring - no stage/compute overlap (one cp.async stall per
// tile) but 32KB less smem, which is what lets hd512 global-layer tiles
// fit sm_120's 99KB cap (M<=48 at PT=16). The loop-end barrier doubles as
// the write-hazard fence before the next stage.
// HD/GL (constexpr-geometry fold): 0 = runtime head_dim / GQA
// group (the generic path); a nonzero HD pins head_dim and GL pins
// log2(group) at compile time so the staging loop's div/mod chains and the
// rr/G mask divisions fold to shifts - the prefill twin measured the
// runtime forms at +53% inst_executed / -7% wall on this same structure.
// Bit-identical either way (addressing arithmetic only).
// F8 (KV8): pool_k/pool_v hold raw e4m3 bytes. Staging
// cp.asyncs the byte strip into the UPPER half of each region's byte range
// (a region is PT*KP halves = 2*PT*KP bytes; head_dim <= KP so it fits -
// zero extra smem, which is what keeps the hd512 M=64 geometry under the
// B200 cap) and expands in place to the identical half layout before the
// mmas. Halved KV DRAM traffic; numerics class unchanged (e4m3 -> f16 is
// exact, score/o mmas stay f16 - no Q quantization, unlike v9q).
// E4: fin-only epilogue twin - quantizes the finalized
// rows to e4m3 + per-row scale IN-KERNEL, bit-identical to the standalone
// pd_quantize_e4m3_row1pc recipe on the same f32 values (exact order-free
// max, frexp e-9 scale, same convert). Only legal when one CTA owns whole
// output rows (n_kv_heads==1: row j == m-tile j). out_o carries the i8
// plane [rows x n_heads*head_dim], out_ml the f32 row scales [rows].
template <uint32_t PT, uint32_t TPW, bool DB = true, bool PAD = true,
          uint32_t HD = 0u, uint32_t GL = 0u, bool F8 = false, bool E4 = false>
__global__ void __launch_bounds__(256, 1) pd_attn_spec_fa_kernel(
    const float* __restrict__ q, const __half* __restrict__ pool_k,
    const __half* __restrict__ pool_v, float* __restrict__ out_o, float* __restrict__ out_ml,
    const unsigned int* __restrict__ positions, const unsigned int* __restrict__ slots,
    const uint32_t* __restrict__ block_tables, uint32_t blocks_per_slot,
    uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim_rt,
    uint32_t kv_dim, uint32_t swa_window, uint32_t n_splits,
    uint32_t rows, uint32_t k1, float scale) {
#if PD_FA_OK
    PD_PDL_ARM();
    // FIN (top bit of n_splits, door-3): n_splits==1 in-kernel
    // finalize - v = o/l is the -inf-sink combine's exact math at one split
    // (sc = exp(0) = 1, sink term exp(-inf - m) = 0), written batch-major
    // to the final buffer; the ml store is skipped. Runtime flag, not a
    // template dim: epilogue-only branch, and it spares 2x the
    // instantiation matrix.
    const bool fin = (n_splits & 0x80000000u) != 0u;
    // FE4S - fin stores e4m3 at STATIC scale 1.0 into
    // out_o reinterpreted as the i8 quantized plane (pf_e4q); the wo-in
    // quantize_e4m3_row launch disappears and the GEMM's xrs is a ones
    // vector. Runtime-uniform branch, no new instantiations.
    const bool fe4s = (n_splits & 0x40000000u) != 0u;
    n_splits &= 0x3fffffffu;
    const uint32_t kvh = blockIdx.x, c = blockIdx.y, s = blockIdx.z;
    const uint32_t head_dim = HD ? HD : head_dim_rt;
    const uint32_t G = GL ? (1u << GL) : n_heads / n_kv_heads;
    const uint32_t M = k1 * G;
    const uint32_t mt = (M + 15u) / 16u;
    // fragment rows span mt*16 - every row-indexed plane pads to Mp so
    // non-multiple M (k1 3/5/7) can't read past its allocation; padded rows
    // are masked rows (j >= nrows) and fold to zero everywhere
    const uint32_t Mp = mt * 16u;
    const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
    const uint32_t rb = c * k1;
    const uint32_t nrows = (rows - rb) < k1 ? (rows - rb) : k1;
    const uint32_t slot = slots ? slots[rb] : rb;

    uint32_t pos_r[8], first_r[8];
    #pragma unroll
    for (uint32_t j = 0; j < 8u; ++j) {
        if (j < nrows) {
            const uint32_t pj = positions[rb + j];
            pos_r[j] = pj;
            first_r[j] = (swa_window > 0 && pj + 1 > swa_window) ? (pj + 1 - swa_window) : 0u;
        } else { pos_r[j] = 0u; first_r[j] = 1u; }
    }
    const uint32_t lo0 = first_r[0];
    const uint32_t pos_max = pos_r[nrows - 1];
    const uint32_t n_pos = pos_max + 1u - lo0;
    const uint32_t chunk = (n_pos + n_splits - 1u) / n_splits;
    const uint32_t lo = s * chunk;
    uint32_t hi = lo + chunk;
    if (hi > n_pos) hi = n_pos;

    // PAD: +8-half / +1-f32 row strides - head_dim rows are 0 mod the 32
    // banks, so unpadded every ldmatrix's 8 rows share 4 banks (8-way
    // conflict) and the thread-per-row softmax scan serializes; the prefill
    // twin (attn/prefill.cuh) measured the pads at 2.4x on this exact
    // structure. Bit-identical (layout only); PAD=false keeps the original
    // for A/B (PADDOCK_SPEC_FA_NOPAD).
    const uint32_t KP = head_dim + (PAD ? 8u : 0u);
    const uint32_t PP = PT + (PAD ? 1u : 0u);
    const uint32_t FP = PT + (PAD ? 8u : 0u);
    extern __shared__ __align__(16) unsigned char fa_sm[];
    __half* s_q = (__half*)fa_sm;                              // [M][KP]
    __half* s_kv = (__half*)(s_q + (size_t)Mp * KP);            // [2][2][PT][KP]
    float* s_p = (float*)(s_kv + (size_t)(DB ? 4u : 2u) * PT * KP);  // [M][PP]
    float* s_m = s_p + (size_t)Mp * PP;                         // [M] x3
    float* s_l = s_m + Mp;
    float* s_corr = s_l + Mp;
    __half* s_pf = (__half*)(s_corr + Mp);                      // [M][FP] f16

    for (uint32_t i = tid; i < Mp * head_dim; i += 256u) {
        const uint32_t rh = i / head_dim, e = i % head_dim;
        const uint32_t j = rh / G, g = rh % G;
        s_q[(size_t)rh * KP + e] = (j < nrows)
            ? __float2half(q[((size_t)(rb + j) * n_heads + (size_t)kvh * G + g) * head_dim + e])
            : __half(0.f);
    }
    for (uint32_t i = tid; i < Mp; i += 256u) { s_m[i] = -INFINITY; s_l[i] = 0.f; }

    // o frag accumulators: STATIC indexing throughout (a runtime task
    // index put these in local memory - 576B stack, the v2 slowness)
    float o_acc[TPW][8][4];
    #pragma unroll
    for (uint32_t a = 0; a < TPW; ++a)
        #pragma unroll
        for (uint32_t b = 0; b < 8u; ++b)
            #pragma unroll
            for (uint32_t cc2 = 0; cc2 < 4u; ++cc2) o_acc[a][b][cc2] = 0.f;

    const uint32_t* bt = block_tables + (size_t)slot * blocks_per_slot;
    // 16-byte cp.async lines per position row: head_dim*2 bytes at f16,
    // head_dim raw e4m3 bytes under F8 (staged to the region's upper byte
    // strip; expanded in place after the wait). Both sides 16B-aligned:
    // head_dim is a multiple of 64 and kv_dim = n_kv*head_dim.
    const uint32_t lines = F8 ? (head_dim >> 4) : ((head_dim * 2u) >> 4);
    auto stage = [&](uint32_t bf, uint32_t t0) {
        const uint32_t n_t = hi - t0 < PT ? hi - t0 : PT;
        for (uint32_t i = tid; i < 2u * n_t * lines; i += 256u) {
            const uint32_t kvsel = i / (n_t * lines);
            const uint32_t jj = i - kvsel * n_t * lines;
            const uint32_t p = jj / lines, l = jj - p * lines;
            const uint32_t gpos = lo0 + t0 + p;
            const uint32_t blk = bt[gpos >> 4];
            if (F8) {
                const unsigned char* src8 = (const unsigned char*)(kvsel ? pool_v : pool_k)
                    + (size_t)blk * 16u * kv_dim + (size_t)(gpos & 15u) * kv_dim
                    + (size_t)kvh * head_dim;
                char* strip = (char*)(s_kv + (size_t)(bf * 2u + kvsel) * PT * KP)
                            + (size_t)PT * KP + (size_t)p * head_dim;
                pd_attn_cpa16(strip + l * 16u, (const char*)src8 + l * 16u);
            } else {
                const __half* src = (kvsel ? pool_v : pool_k)
                    + (size_t)blk * 16u * kv_dim + (size_t)(gpos & 15u) * kv_dim
                    + (size_t)kvh * head_dim;
                __half* dst = s_kv + ((size_t)(bf * 2u + kvsel) * PT + p) * KP;
                pd_attn_cpa16((char*)dst + l * 16u, (const char*)src + l * 16u);
            }
        }
        pd_attn_cpa_commit();
    };

    __syncthreads();
    if (DB && lo < hi) stage(0u, lo);
    uint32_t bf = 0;
    for (uint32_t t0 = lo; t0 < hi; t0 += PT, bf ^= (DB ? 1u : 0u)) {
        const uint32_t n_t = hi - t0 < PT ? hi - t0 : PT;
        const bool more = t0 + PT < hi;
        if (DB) {
            if (more) stage(bf ^ 1u, t0 + PT);
            if (more) pd_attn_cpa_wait1(); else pd_attn_cpa_wait0();
        } else {
            stage(0u, t0);
            pd_attn_cpa_wait0();
        }
        __syncthreads();
        const __half* kbuf = s_kv + (size_t)(bf * 2u) * PT * KP;
        const __half* vbuf = s_kv + ((size_t)(bf * 2u) + 1u) * PT * KP;

        if (F8) {
            // in-place e4m3 -> f16 expansion of the staged strips. Register-
            // staged per region (the half writes cover the byte strip they
            // read), K wave then V wave so only one region's chunks are live
            // across a barrier (16 regs worst case, not 32). Rows >= n_t are
            // ZERO-filled: stale e4m3 bytes can decode to NaN (0x7f/0xff)
            // and the o-mma's 0-weight x stale-V product would poison the
            // accumulator - f16 staging never had this hazard (stale halves
            // are finite). Chunks are 16B and never cross a position row
            // (head_dim % 16 == 0), so each maps to 32 contiguous half bytes.
            constexpr uint32_t CHW = PT / 8u;  // chunks/thread/region at hd512
            const uint32_t rchunks = PT * lines;  // full PT so tail rows zero
            uint4 rg[CHW];
            #pragma unroll
            for (uint32_t kvsel = 0; kvsel < 2u; ++kvsel) {
                const char* strip = (const char*)(s_kv
                    + (size_t)(bf * 2u + kvsel) * PT * KP) + (size_t)PT * KP;
                #pragma unroll
                for (uint32_t ci = 0; ci < CHW; ++ci) {
                    const uint32_t c = tid + ci * 256u;
                    if (c >= rchunks) break;
                    const uint32_t p = c / lines;
                    rg[ci] = (p < n_t) ? *(const uint4*)(strip + (size_t)c * 16u)
                                       : make_uint4(0u, 0u, 0u, 0u);
                }
                __syncthreads();  // all strip reads before the overlapping writes
                #pragma unroll
                for (uint32_t ci = 0; ci < CHW; ++ci) {
                    const uint32_t c = tid + ci * 256u;
                    if (c >= rchunks) break;
                    const uint32_t p = c / lines, l = c - p * lines;
                    __half* dst = s_kv
                        + ((size_t)(bf * 2u + kvsel) * PT + p) * KP + l * 16u;
                    const uint32_t* w = (const uint32_t*)&rg[ci];
                    #pragma unroll
                    for (uint32_t qd = 0; qd < 4u; ++qd) {
                        ((__half2*)dst)[qd * 2u] = __half2(__nv_cvt_fp8x2_to_halfraw2(
                            (__nv_fp8x2_storage_t)(w[qd] & 0xffffu), __NV_E4M3));
                        ((__half2*)dst)[qd * 2u + 1u] = __half2(__nv_cvt_fp8x2_to_halfraw2(
                            (__nv_fp8x2_storage_t)(w[qd] >> 16), __NV_E4M3));
                    }
                }
            }
            __syncthreads();  // expanded halves visible before the mmas
        }

        // scores on tensor cores: warp w < mt*(PT/8)/? -> assign (m_tile,
        // col-subtile) pairs across warps: tasks = mt * PT/8
        {
            const uint32_t tasks = mt * (PT / 8u);
            for (uint32_t task = warp; task < tasks; task += 8u) {
                const uint32_t tm = task / (PT / 8u), cs = task % (PT / 8u);
                const uint32_t r0 = tm * 16u, p0 = cs * 8u;
                float d[4] = {0.f, 0.f, 0.f, 0.f};
                for (uint32_t kk = 0; kk < head_dim; kk += 16u) {
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
                        if (pp >= n_t || j >= nrows || gpos > pos_r[j] || gpos < first_r[j])
                            v = -INFINITY;
                        s_p[rr * PP + pp] = v;
                    }
                }
            }
        }
        __syncthreads();
        // softmax state, WARP-PARALLEL (ks3): the
        // old thread-per-row pass (Mp of 256 threads, two serial PT-scans
        // with expf) was the tile's real serial chain - a variant that merely
        // DOUBLED its per-element work lost 30-40%, and this form wins 306.6
        // -> 257.5 us fin on the GB202 SWA shape. Warp w owns rows w, w+8,
        // ...; lanes stride positions; shfl-tree max/sum. The max fold is
        // order-exact; l's summation order changes (lane tree vs serial) -
        // same numeric class as a split-count change, gate-arbitrated. The
        // old s_p weight store is DROPPED (dead - only s_pf feeds the o-mma).
        for (uint32_t rr = warp; rr < Mp; rr += 8u) {
            const float m_old = s_m[rr];
            float mx = m_old;
            for (uint32_t pp = lane; pp < n_t; pp += 32u)
                mx = fmaxf(mx, s_p[rr * PP + pp]);
            #pragma unroll
            for (uint32_t o = 16; o; o >>= 1)
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float corr = (mx == -INFINITY) ? 1.f : __expf(m_old - mx);
            float ls = 0.f;
            for (uint32_t pp = lane; pp < n_t; pp += 32u) {
                const float sp = s_p[rr * PP + pp];
                const float w = (sp == -INFINITY) ? 0.f : __expf(sp - mx);
                s_pf[rr * FP + pp] = __float2half(w);
                ls += w;
            }
            // zero the f16 strip beyond n_t: tail tiles otherwise feed the
            // o-mma stale weights against stale V rows
            for (uint32_t pp = n_t + lane; pp < PT; pp += 32u)
                s_pf[rr * FP + pp] = __half(0.f);
            #pragma unroll
            for (uint32_t o = 16; o; o >>= 1)
                ls += __shfl_xor_sync(0xffffffffu, ls, o);
            if (lane == 0) {
                s_corr[rr] = corr;
                s_l[rr] = s_l[rr] * corr + ls;
                s_m[rr] = mx;
            }
        }
        __syncthreads();
        // o update on tensor cores: warp tasks (m_tile, 64-dim slice);
        // A = p f16 strip (16 rows x 16 pos), B = V via ldmatrix.x2.trans
        // (V rows are pos-major -> trans gives the [k=pos][n=dim] frag).
        // o frags: per task 8 subtiles x 4 f32 = OS regs.
        {
            const uint32_t slices = head_dim / 64u;
            const uint32_t tasks = mt * slices;
            #pragma unroll
            for (uint32_t ti = 0; ti < TPW; ++ti) {
                const uint32_t task = warp + ti * 8u;
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
    }
    // E4 epilogue (fin implied): CTA-local row max over the finalized f32
    // values, then the exact row1pc scale recipe + e4m3 store. Two walks of
    // o_acc with a smem reduction between; the f32 row never lands in DRAM.
    if constexpr (E4) {
        __shared__ int s_rmx[8];     // per-row |v| max as ordered int (k1<=8)
        __shared__ float s_rinv[8];  // per-row inverse scale
        if (tid < 8u) s_rmx[tid] = 0;
        __syncthreads();
        const uint32_t slices = head_dim / 64u;
        const uint32_t tasks = mt * slices;
        #pragma unroll
        for (uint32_t ti = 0; ti < TPW; ++ti) {
            const uint32_t task = warp + ti * 8u;
            if (task >= tasks) break;
            const uint32_t tm = task / slices;
            const uint32_t r0 = tm * 16u;
            #pragma unroll
            for (uint32_t sub = 0; sub < 8u; ++sub) {
                #pragma unroll
                for (uint32_t half = 0; half < 2u; ++half) {
                    const uint32_t rr = r0 + (lane >> 2) + half * 8u;
                    const uint32_t j = rr / G;
                    if (j >= nrows) continue;
                    // same f32 division the f32 store would have written --
                    // the max is taken over bit-identical values
                    const float l = s_l[rr];
                    const float v0 = o_acc[ti][sub][half * 2u] / l;
                    const float v1 = o_acc[ti][sub][half * 2u + 1u] / l;
                    atomicMax(&s_rmx[j],
                              __float_as_int(fmaxf(fabsf(v0), fabsf(v1))));
                }
            }
        }
        __syncthreads();
        if (tid < 8u && tid < nrows) {
            // pd_quantize_e4m3_row1pc recipe, verbatim (frexp e-9)
            const float m = __int_as_float(s_rmx[tid]);
            int e = 0;
            if (m > 0.0f) {
                int ex;
                float fr = frexpf(m, &ex);
                e = ex - 9 + (fr > 0.875f ? 1 : 0);
            }
            s_rinv[tid] = ldexpf(1.0f, -e);
            out_ml[rb + tid] = ldexpf(1.0f, e);   // row scale plane
        }
        __syncthreads();
        #pragma unroll
        for (uint32_t ti = 0; ti < TPW; ++ti) {
            const uint32_t task = warp + ti * 8u;
            if (task >= tasks) break;
            const uint32_t tm = task / slices, sl = task % slices;
            const uint32_t r0 = tm * 16u, n_base = sl * 64u;
            #pragma unroll
            for (uint32_t sub = 0; sub < 8u; ++sub) {
                #pragma unroll
                for (uint32_t half = 0; half < 2u; ++half) {
                    const uint32_t rr = r0 + (lane >> 2) + half * 8u;
                    const uint32_t j = rr / G, g = rr % G;
                    if (j >= nrows) continue;
                    const float l = s_l[rr];
                    const float inv = s_rinv[j];
                    const float v0 = o_acc[ti][sub][half * 2u] / l;
                    const float v1 = o_acc[ti][sub][half * 2u + 1u] / l;
                    unsigned char* dst = (unsigned char*)out_o
                        + ((size_t)(rb + j) * n_heads + kvh * G + g) * head_dim
                        + n_base + sub * 8u + 2u * (lane & 3u);
                    dst[0] = __nv_fp8_e4m3(v0 * inv).__x;
                    dst[1] = __nv_fp8_e4m3(v1 * inv).__x;
                }
            }
        }
    }
    // epilogue: partial o and (m, l) in the same layout the production
    // combine expects: out_o[((rh_global) * n_splits + s) * hd + d],
    // out_ml[..] with rh_global = (rb + j) * n_heads + kvh*G + g
    if constexpr (!E4) {
        const uint32_t slices = head_dim / 64u;
        const uint32_t tasks = mt * slices;
        #pragma unroll
        for (uint32_t ti = 0; ti < TPW; ++ti) {
            const uint32_t task = warp + ti * 8u;
            if (task >= tasks) break;
            const uint32_t tm = task / slices, sl = task % slices;
            const uint32_t r0 = tm * 16u, n_base = sl * 64u;
            #pragma unroll
            for (uint32_t sub = 0; sub < 8u; ++sub) {
                #pragma unroll
                for (uint32_t half = 0; half < 2u; ++half) {
                    const uint32_t rr = r0 + (lane >> 2) + half * 8u;
                    const uint32_t j = rr / G, g = rr % G;
                    if (j >= nrows) continue;
                    if (fin) {
                        // finalized batch-major row (pf_attn layout): same
                        // operands and one divide == the 1-split combine
                        const float l = s_l[rr];
                        const size_t off = ((size_t)(rb + j) * n_heads + kvh * G + g) * head_dim
                                         + n_base + sub * 8u + 2u * (lane & 3u);
                        if (fe4s) {
                            unsigned char* dq = (unsigned char*)out_o + off;
                            dq[0] = __nv_fp8_e4m3(o_acc[ti][sub][half * 2u] / l).__x;
                            dq[1] = __nv_fp8_e4m3(o_acc[ti][sub][half * 2u + 1u] / l).__x;
                        } else {
                            float* dst = out_o + off;
                            dst[0] = o_acc[ti][sub][half * 2u] / l;
                            dst[1] = o_acc[ti][sub][half * 2u + 1u] / l;
                        }
                    } else {
                        const size_t hg = (size_t)(kvh * G + g) * rows + (rb + j);
                        float* dst = out_o + (hg * n_splits + s) * head_dim
                                   + n_base + sub * 8u + 2u * (lane & 3u);
                        dst[0] = o_acc[ti][sub][half * 2u];
                        dst[1] = o_acc[ti][sub][half * 2u + 1u];
                    }
                }
            }
        }
    }
    if (!fin && tid < Mp && tid < M) {
        const uint32_t rr = tid, j = rr / G, g = rr % G;
        if (j < nrows) {
            const size_t hg = (size_t)(kvh * G + g) * rows + (rb + j);
            out_ml[(hg * n_splits + s) * 2 + 0] = s_m[rr];
            out_ml[(hg * n_splits + s) * 2 + 1] = s_l[rr];
        }
#ifdef PD_KRS_DUMP
        // final per-row m/l (dump tail rows: [496..511] m, [512..527] l)
        if (kvh == 0u && c == 0u && s == 0u) {
            pd_krs_dump[16 * 33 - 32 + rr] = s_m[rr];
            pd_krs_dump[16 * 33 - 16 + rr] = s_l[rr];
        }
#endif
    }
#else
    (void)q; (void)pool_k; (void)pool_v; (void)out_o; (void)out_ml;
    (void)positions; (void)slots; (void)block_tables; (void)blocks_per_slot;
    (void)n_heads; (void)n_kv_heads; (void)head_dim_rt; (void)kv_dim;
    (void)swa_window; (void)n_splits; (void)rows; (void)k1; (void)scale;
#endif
}

// ── spec-FA krs: fp8-RESIDENT K ─────────────────────────────────────────
// The F8 route stages raw bytes then does an IN-PLACE f16 expansion round
// trip for both sides every tile (read strip -> sync -> rewrite as halves).
// Here K never expands: it stays raw e4m3 in a compact [PT][HD+16]-byte
// region and the score B-frags are built by 2x u16 loads + fp8x2 cvt per
// mma - the ldmatrix.x2 fragment map is lane l <- K[p0+(l>>2)][kk+2(l&3)]
// / +8 (v3c's kb recipe), so at equal PT this is BIT-equal to the F8 path
// (e4m3->f16 exact, same operand values and fold order; KR32 verified
// BITEQ). V keeps the expansion - its .trans ldmatrix needs b16
// (ldmatrix.b8 is not on sm_120). What it buys
// (measured at the 128-row/k1=4 serve point): K's expansion wave
// gone + ~8.4KB smem freed -> PT ladder headroom. SWA hd256: KR40 sp2
// 214.7 vs PT32-SB 221.2us (occ 2). GLB hd512: KR32 sp4 287.9 vs
// PT16-SB 410.4us (PT32 halves the occ-1 arm's un-overlapped
// tile stalls and now FITS the 99KB cap). PTv pads the V region to a
// 16-row multiple so the o-mma's k=16 .trans never reads outside it
// (PT=40); V rows [n_t, PTv) and the s_pf strip zero to PTv - stale
// f16/NaN would poison the 0-weight products. PT!=32 shifts tile
// boundaries = split-count numeric class, serve-gate arbitrated.
// SB + F8 + PAD specialized (the sm_120 elections that route here).
// VR (fp8 PxV rung A): V raw-resident too - the o-mma builds its
// B-frags from raw e4m3 at the mma seam (kb recipe in the PV orientation:
// halves run along POSITIONS, so 2 byte loads + pack per register instead
// of the .trans ldmatrix). Kills the f16 V region + expansion pass:
// ~16-25KB smem back on the class where smem headroom bought KR32's win.
// e4m3->f16 per element is exact either way = BIT-equal at equal PT.
// QK8: e4m3-Q scores on the GLB arm - Q staged as
// raw e4m3 (v8q/v9q production class), QK^T on mma.m16n8k32.e4m3 with
// B-frags ldmatrix'd straight off the raw K rows: half the chain steps and
// zero cvt instructions on it, s_q bytes halve. Softmax/PV untouched. The
// measured GLB sp4 288.6 vs 308.4us at the
// wide serve point; SWA stays f16 (occupancy cliff, R1/R2 falsified).
// Numeric class: e4m3-Q rounding (same as the v9q dense arm) - Not
// bitwise; serve acceptance arbitrates.
// DBK: double-buffered KV stage - tile t+1's
// cp.asyncs issue before the wait on tile t, so the walk's per-tile DRAM
// round trip overlaps the score/softmax/o math instead of serializing
// after it (the "un-overlapped tile stalls" the PT16->32 widening only
// halved; every prior rung bought them with occupancy, never overlap).
// Same PT, same tile order, same math = BIT-equal to the SB arm. VR-only:
// the !VR route aliases its staging strip into s_v's top half and would
// need the strip doubled too - VR raw residency has no strip. Cost is one
// extra K+V tile of smem: SWA PT32 occ 2 (46.8KB), PT16 keeps occ 3
// (27.9KB); GLB PT32+QK8+VR fits occ-1 at 91.6KB (the arm with zero
// cross-CTA overlap, where the exposed walk latency is the recorded bind).
// TQ (attention streams): f16 q plane - bit-equal (!QK8 stages
// via __float2half already; (float)h expand is exact). QK8 folds the f16
// rounding into its e4m3 quantize (same acceptance class as QK8 itself).
// P8: e4m3 P x raw e4m3 V on mma.m16n8k32 - after
// softmax the P strip is e4m3 BYTES ([Mp][PTv+16]) and the o-mma runs at
// the fp8 rate: A-frags via the plain b16 ldmatrix on the byte strip (u16
// = 2 adjacent positions, the QK8 score recipe on the A side), B-frags =
// the VR byte gather widened to 4-position packs with the cvts deleted.
// The class vLLM ships by default under fp8 KV (triton unified attention:
// `acc += tl.dot(P.to(V.dtype), V)`). Measured: GLB
// P8+QK8+DBK fin 265.2 vs elected xV 287.5, and ahead at sp4; SWA is
// NEUTRAL at occ 3 (209us every class - walk-BW-bound) so SWA stays f16-P.
// Requires VR (raw V bytes feed the mma) and PTv % 32 == 0 (the k32 mma
// reads positions through the 16-pad; PT32/64 legal, PT16/40/48 not).
// KVS: K/V SPLIT-COMMIT walk - K and V stage as
// separate cp.async groups; the score mma waits only on K while V (and,
// under DBK, the whole prefetched tile) stays in flight through
// score+softmax to a wait right before the o-update barrier. Same tile
// order, same math = BIT-equal (every leg verified BITEQ). GLB
// P8QD fin -3.5% / sp4 -2.5%; SWA neutral at occ 3 (cross-CTA overlap
// already hides what the split defers) so SWA stays !KVS. Sub-tile
// K-half commits (RS2) falsified: the mid-score barrier costs more
// than the granularity buys (+7.9% GLB, +18 regs).
// GV (q36): direct group-size override for non-power-of-two GQA -
// GV != 0 replaces 1<<GL (the 24q/4kv class is G=6; same wall hit
// at prefill, same fix: the missing instantiation, not the math - M=k1*G
// already pads through mt to Mp).
template <uint32_t PT, uint32_t TPW, uint32_t HD, uint32_t GL, bool VR = false,
          bool QK8 = false, bool DBK = false, typename TQ = float,
          bool P8 = false, bool KVS = false, uint32_t GV = 0>
__global__ void __launch_bounds__(256, PD_FA_KRS_OCC) pd_attn_spec_fa_krs_kernel(
    const TQ* __restrict__ q, const __half* __restrict__ pool_k,
    const __half* __restrict__ pool_v, float* __restrict__ out_o, float* __restrict__ out_ml,
    const unsigned int* __restrict__ positions, const unsigned int* __restrict__ slots,
    const uint32_t* __restrict__ block_tables, uint32_t blocks_per_slot,
    uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim_rt,
    uint32_t kv_dim, uint32_t swa_window, uint32_t n_splits,
    uint32_t rows, uint32_t k1, float scale, uint32_t fill_sms) {
#if PD_FA_OK
    PD_PDL_ARM();
    const bool fin = (n_splits & 0x80000000u) != 0u;
    // FE4S - fin stores e4m3 at STATIC scale 1.0 into
    // out_o reinterpreted as the i8 quantized plane (pf_e4q); the wo-in
    // quantize_e4m3_row launch disappears and the GEMM's xrs is a ones
    // vector. Runtime-uniform branch, no new instantiations.
    const bool fe4s = (n_splits & 0x40000000u) != 0u;
    n_splits &= 0x3fffffffu;
    const uint32_t kvh = blockIdx.x, c = blockIdx.y, s = blockIdx.z;
    constexpr uint32_t head_dim = HD;
    constexpr uint32_t G = GV != 0u ? GV : (1u << GL);
    const uint32_t M = k1 * G;
    const uint32_t mt = (M + 15u) / 16u;
    const uint32_t Mp = mt * 16u;
    const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
    const uint32_t rb = c * k1;
    const uint32_t nrows = (rows - rb) < k1 ? (rows - rb) : k1;
    const uint32_t slot = slots ? slots[rb] : rb;

    uint32_t pos_r[8], first_r[8];
    #pragma unroll
    for (uint32_t j = 0; j < 8u; ++j) {
        if (j < nrows) {
            const uint32_t pj = positions[rb + j];
            pos_r[j] = pj;
            first_r[j] = (swa_window > 0 && pj + 1 > swa_window) ? (pj + 1 - swa_window) : 0u;
        } else { pos_r[j] = 0u; first_r[j] = 1u; }
    }
    const uint32_t lo0 = first_r[0];
    const uint32_t pos_max = pos_r[nrows - 1];
    const uint32_t n_pos = pos_max + 1u - lo0;
    // GV (dense-decode) arm: adaptive effective splits - the scalar walk's
    // port (target >= 4 tiles per live CTA; at short kv the graph-baked 32
    // splits hand each CTA ~1 tile and prelude dominates: measured
    // +70% at ctx300 before this). Surplus splits see lo >= hi and write the
    // (-inf, 0) empty partial the combine folds. Spec arms (GV == 0) keep
    // the fixed split - their geometry is gated elsewhere.
    uint32_t s_eff = n_splits;
    if constexpr (GV != 0u) {
        constexpr uint32_t tpc = PD_FA_KRS_TPC;
        s_eff = (n_pos + tpc * PT - 1u) / (tpc * PT);
        // Die-fill floor. The tiles-per-CTA target above
        // derives s_eff from CONTEXT alone, so at small batch it leaves the
        // die idle: the live grid is gridDim.x*gridDim.y*s_eff, which at the
        // c8 serve point (4 kv x 8 rows x 2) is 64 CTAs on 148 SMs - and this
        // kernel has no cross-CTA overlap to hide its serial tile walk, so
        // that is the wall. When the live grid does not cover one wave, split
        // finer, down to one tile per CTA and bounded at ~2 waves.
        //
        // Two guards, each measured, each load-bearing:
        //  (a) base*4 <= fill_sms - engage only when one split covers <= a
        //      quarter of the die. b=8 (base 32, 22% of 148) wins -11..-24%
        //      at every ctx 130..288; b=16 (base 64, 43%) loses +1.5..+2.7%
        //      across ctx 130..192, the regime a 128-token-prompt leg actually
        //      decodes through - at 128/148 CTAs the die is already covered and
        //      the extra partial+combine traffic is pure cost. A c16 serve ABBA
        //      measured exactly that (+0.14 ms ITL on its clean boot). This
        //      guard keeps c16/c32 bit-identical to the unsplit form.
        //  (b) base*s_eff < fill_sms - never split a run that already fills the
        //      die. Forcing one tile per CTA regardless (TPC=1) measured +85%
        //      at b=32/ctx1150 and +20% at ctx2048; it also keeps b=8 off that
        //      cliff at long context (ctx2048 -> base*s_eff = 512 > 148, skip).
        // fill_sms = 0 restores the context-only clamp exactly (PADDOCK_NO_FA_FILL).
        const uint32_t base = gridDim.x * gridDim.y;   // live CTAs per split
        if (fill_sms && base && base * 4u <= fill_sms
            && base * s_eff < fill_sms) {
            const uint32_t s_cap = (n_pos + PT - 1u) / PT;   // 1 tile/CTA
            const uint32_t s_room = (2u * fill_sms) / base;
            if (s_room > s_eff) s_eff = s_room;
            if (s_eff > s_cap) s_eff = s_cap;
        }
        if (s_eff > n_splits) s_eff = n_splits;
        if (s_eff < 1u) s_eff = 1u;
    }
    const uint32_t chunk = (n_pos + s_eff - 1u) / s_eff;
    const uint32_t lo = s * chunk;
    uint32_t hi = lo + chunk;
    if (hi > n_pos) hi = n_pos;

    // Fully-surplus split: lo/hi are CTA-uniform, so the whole CTA exits here
    // - before the Q-stage, smem init, and o-store epilogue it would otherwise
    // pay just to emit zeros. Only the (-inf, 0) ml pair is written; the
    // combine skips the o read on m == -inf, so the zero o-store is dead. At
    // kv <= 256 the GV arm's s_eff is ~2 and 14/16 CTAs take this exit (the
    // graph-baked grid still schedules them; this makes them prologue-only).
    // fin (n_splits == 1) can never be surplus: hi = n_pos >= 1.
    if (!fin && lo >= hi) {
        if (tid < M) {
            const uint32_t j = tid / G, g = tid % G;
            if (j < nrows) {
                const size_t hg = (size_t)(kvh * G + g) * rows + (rb + j);
                out_ml[(hg * n_splits + s) * 2 + 0] = -INFINITY;
                out_ml[(hg * n_splits + s) * 2 + 1] = 0.0f;
            }
        }
        return;
    }

    static_assert(!DBK || VR, "DBK requires the VR (raw-V) layout - the !VR "
                              "staging strip aliases s_v and is not doubled");
    constexpr uint32_t KP = head_dim + 8u;        // f16 row stride (halves)
    constexpr uint32_t KPB = head_dim + 16u;      // K byte row stride (16B-aligned)
    constexpr uint32_t PTv = (PT + 15u) & ~15u;   // V rows padded to the mma k=16
    constexpr uint32_t PP = PT + 1u;
    constexpr uint32_t FP = PT + 8u;
    constexpr uint32_t FPB = PTv + 16u;  // P8: e4m3 P-strip byte pitch (16B rows)
    static_assert(!P8 || VR, "P8 o-mma eats raw V bytes - VR layout required");
    static_assert(!P8 || (PTv & 31u) == 0u,
                  "P8 k32 o-mma reads 32-position packs through the zero pad");
    constexpr uint32_t NBUF = DBK ? 2u : 1u;      // K/V tile buffers
    extern __shared__ __align__(16) unsigned char fa_krs_sm[];
    __half* s_q = (__half*)fa_krs_sm;                           // [Mp][KP] f16 (!QK8)
    unsigned char* s_q8 = fa_krs_sm;                            // [Mp][KPB] e4m3 (QK8)
    unsigned char* s_kb = (unsigned char*)fa_krs_sm
        + (QK8 ? (size_t)Mp * KPB : (size_t)Mp * KP * 2u);      // [NBUF][PT][KPB] raw e4m3
    // VR: V raw e4m3 in [PTv][KPB] bytes (16B rows for the cpa lines; the
    // KPB pitch spreads the o-gather's column banks). !VR: f16 [PTv][KP]
    // with the staging byte strip aliased into its top half.
    __half* s_v = (__half*)(s_kb + (size_t)NBUF * PT * KPB);    // [PTv][KP] (!VR)
    unsigned char* s_vb = (unsigned char*)s_v;                  // [NBUF][PTv][KPB] (VR)
    float* s_p = (float*)((unsigned char*)s_v
                          + (VR ? (size_t)NBUF * PTv * KPB
                                : (size_t)PTv * KP * 2u));      // [Mp][PP]
    float* s_m = s_p + (size_t)Mp * PP;                         // [Mp] x3
    float* s_l = s_m + Mp;
    float* s_corr = s_l + Mp;
    __half* s_pf = (__half*)(s_corr + Mp);                      // [Mp][FP] f16
    unsigned char* s_pf8 = (unsigned char*)(s_corr + Mp);       // [Mp][FPB] e4m3 (P8)

    if constexpr (QK8) {
        // e4m3 Q over the full KPB pitch (zero pad cols - the k32
        // A-ldmatrix reads through them)
        for (uint32_t i = tid; i < Mp * KPB; i += 256u) {
            const uint32_t rh = i / KPB, e = i % KPB;
            const uint32_t j = rh / G, g = rh % G;
            const float v = (j < nrows && e < head_dim)
                ? (float)q[((size_t)(rb + j) * n_heads + (size_t)kvh * G + g) * head_dim + e]
                : 0.0f;
            s_q8[(size_t)rh * KPB + e] = __nv_fp8_e4m3(v).__x;
        }
    } else {
        for (uint32_t i = tid; i < Mp * head_dim; i += 256u) {
            const uint32_t rh = i / head_dim, e = i % head_dim;
            const uint32_t j = rh / G, g = rh % G;
            s_q[(size_t)rh * KP + e] = (j < nrows)
                ? __float2half((float)q[((size_t)(rb + j) * n_heads + (size_t)kvh * G + g) * head_dim + e])
                : __half(0.f);
        }
    }
    for (uint32_t i = tid; i < Mp; i += 256u) { s_m[i] = -INFINITY; s_l[i] = 0.f; }

    float o_acc[TPW][8][4];
    #pragma unroll
    for (uint32_t a = 0; a < TPW; ++a)
        #pragma unroll
        for (uint32_t b = 0; b < 8u; ++b)
            #pragma unroll
            for (uint32_t cc2 = 0; cc2 < 4u; ++cc2) o_acc[a][b][cc2] = 0.f;

    const uint32_t* bt = block_tables + (size_t)slot * blocks_per_slot;
    constexpr uint32_t lines = head_dim >> 4;     // 16-byte e4m3 lines per row
    auto stage = [&](uint32_t bsel, uint32_t t0) {
        const uint32_t n_t = hi - t0 < PT ? hi - t0 : PT;
        for (uint32_t i = tid; i < 2u * n_t * lines; i += 256u) {
            const uint32_t kvsel = i / (n_t * lines);
            const uint32_t jj = i - kvsel * n_t * lines;
            const uint32_t p = jj / lines, l = jj - p * lines;
            const uint32_t gpos = lo0 + t0 + p;
            const uint32_t blk = bt[gpos >> 4];
            const unsigned char* src8 = (const unsigned char*)(kvsel ? pool_v : pool_k)
                + (size_t)blk * 16u * kv_dim + (size_t)(gpos & 15u) * kv_dim
                + (size_t)kvh * head_dim;
            char* dst = kvsel
                ? (VR ? (char*)s_vb + ((size_t)bsel * PTv + p) * KPB      // V stays raw (VR)
                      : (char*)s_v + (size_t)PTv * KP + (size_t)p * head_dim) // V byte strip
                : (char*)s_kb + ((size_t)bsel * PT + p) * KPB;            // K stays raw
            pd_attn_cpa16(dst + l * 16u, (const char*)src8 + l * 16u);
        }
        pd_attn_cpa_commit();
    };
    // KVS: one SIDE per commit group - K (kvsel 0) or V (kvsel 1) into
    // buffer bsel. The commit is unconditional so the group ledger stays
    // uniform on short tiles (an empty group completes immediately).
    auto stage_kv = [&](uint32_t bsel, uint32_t t0, uint32_t kvsel) {
        const uint32_t n_t = hi - t0 < PT ? hi - t0 : PT;
        for (uint32_t i = tid; i < n_t * lines; i += 256u) {
            const uint32_t p = i / lines, l = i - p * lines;
            const uint32_t gpos = lo0 + t0 + p;
            const uint32_t blk = bt[gpos >> 4];
            const unsigned char* src8 = (const unsigned char*)(kvsel ? pool_v : pool_k)
                + (size_t)blk * 16u * kv_dim + (size_t)(gpos & 15u) * kv_dim
                + (size_t)kvh * head_dim;
            char* dst = kvsel
                ? (VR ? (char*)s_vb + ((size_t)bsel * PTv + p) * KPB
                      : (char*)s_v + (size_t)PTv * KP + (size_t)p * head_dim)
                : (char*)s_kb + ((size_t)bsel * PT + p) * KPB;
            pd_attn_cpa16(dst + l * 16u, (const char*)src8 + l * 16u);
        }
        pd_attn_cpa_commit();
    };
    auto issue = [&](uint32_t bsel, uint32_t t0) {
        if constexpr (KVS) { stage_kv(bsel, t0, 0u); stage_kv(bsel, t0, 1u); }
        else stage(bsel, t0);
    };

    // DBK prologue: tile 0's cp.asyncs ride ahead of the Q-stage barrier -
    // nothing reads the K/V regions before the in-loop wait.
    if constexpr (DBK) { if (lo < hi) issue(0u, lo); }
    __syncthreads();
    uint32_t bf = 0u;
    for (uint32_t t0 = lo; t0 < hi; t0 += PT, bf ^= (NBUF - 1u)) {
        const uint32_t n_t = hi - t0 < PT ? hi - t0 : PT;
        bool pf = false;  // prefetch group(s) in flight this iteration
        if constexpr (DBK) {
            // issue tile t+1 into the other buffer before waiting on tile
            // t - the prefetch stays in flight through the math (KVS:
            // both of its groups do).
            if (t0 + PT < hi) { issue(bf ^ 1u, t0 + PT); pf = true; }
        } else {
            issue(0u, t0);
        }
        (void)pf;
        // K-wait. FIFO ledger: !KVS = 1 group/tile (wait1 under a
        // prefetch, wait0 else - the original DBK levels); KVS = 2
        // groups (K then V): the scores block only on K, V rides on.
        if constexpr (KVS) {
            if (pf) pd_attn_cpa_wait3(); else pd_attn_cpa_wait1();
        } else {
            if (pf) pd_attn_cpa_wait1(); else pd_attn_cpa_wait0();
        }
        unsigned char* kbuf = s_kb + (size_t)bf * PT * KPB;
        unsigned char* vbuf = s_vb + (size_t)bf * PTv * KPB;
        __syncthreads();

        // VR: no expansion at all - the o-mma consumes raw bytes. Rows
        // [n_t, PTv) still must zero on every short tile: stale e4m3 0x7f
        // decodes to NaN and 0-weight x NaN poisons o (e4m3 0x00 is exact
        // +0.0, so zero BYTES give zero halves). No extra sync: the o-mma
        // reads sit behind the score/softmax __syncthreads chain.
        if constexpr (VR) {
            if (n_t < PTv) {
                constexpr uint32_t rl = KPB >> 4;
                for (uint32_t i = tid; i < (PTv - n_t) * rl; i += 256u)
                    ((uint4*)(vbuf + (size_t)(n_t + i / rl) * KPB))[i % rl] =
                        make_uint4(0u, 0u, 0u, 0u);
            }
        } else {
        // in-place e4m3 -> f16 expansion, V only (K is consumed raw). Rows
        // [n_t, PTv) zero: stale e4m3 can decode to NaN and 0-weight x NaN
        // poisons the o accumulate.
            constexpr uint32_t rchunks = PTv * lines;
            constexpr uint32_t CHW = (rchunks + 255u) / 256u;
            uint4 rg[CHW];
            const char* strip = (const char*)s_v + (size_t)PTv * KP;
            #pragma unroll
            for (uint32_t ci = 0; ci < CHW; ++ci) {
                const uint32_t cc = tid + ci * 256u;
                if (cc >= rchunks) break;
                const uint32_t p = cc / lines;
                rg[ci] = (p < n_t) ? *(const uint4*)(strip + (size_t)cc * 16u)
                                   : make_uint4(0u, 0u, 0u, 0u);
            }
            __syncthreads();  // all strip reads before the overlapping writes
            #pragma unroll
            for (uint32_t ci = 0; ci < CHW; ++ci) {
                const uint32_t cc = tid + ci * 256u;
                if (cc >= rchunks) break;
                const uint32_t p = cc / lines, l = cc - p * lines;
                __half* dst = s_v + (size_t)p * KP + l * 16u;
                const uint32_t* w = (const uint32_t*)&rg[ci];
                #pragma unroll
                for (uint32_t qd = 0; qd < 4u; ++qd) {
                    ((__half2*)dst)[qd * 2u] = __half2(__nv_cvt_fp8x2_to_halfraw2(
                        (__nv_fp8x2_storage_t)(w[qd] & 0xffffu), __NV_E4M3));
                    ((__half2*)dst)[qd * 2u + 1u] = __half2(__nv_cvt_fp8x2_to_halfraw2(
                        (__nv_fp8x2_storage_t)(w[qd] >> 16), __NV_E4M3));
                }
            }
            __syncthreads();  // expanded halves visible before the mmas
        }

        // QK8: K rows past n_t must be finite for the k32 B-ldmatrix - pad
        // COLUMNS hit real A rows, and NaN x 0 poisons the f32 accum (the
        // f16 route's cvt path never touches them). Sync: these zeroes are
        // read by the score mma right below.
        if constexpr (QK8) {
            if (n_t < PT) {
                constexpr uint32_t rl = KPB >> 4;
                for (uint32_t i = tid; i < (PT - n_t) * rl; i += 256u)
                    ((uint4*)(kbuf + (size_t)(n_t + i / rl) * KPB))[i % rl] =
                        make_uint4(0u, 0u, 0u, 0u);
                __syncthreads();
            }
        }
        // scores: A = Q via ldmatrix.x4, B = raw K bytes + cvt (the krs
        // swap); QK8 swaps the chain for e4m3 A x e4m3 B m16n8k32 (half
        // the steps, zero cvt - GLB measured -6.4%)
        {
            const uint32_t tasks = mt * (PT / 8u);
            for (uint32_t task = warp; task < tasks; task += 8u) {
                const uint32_t tm = task / (PT / 8u), cs = task % (PT / 8u);
                const uint32_t r0 = tm * 16u, p0 = cs * 8u;
                float d[4] = {0.f, 0.f, 0.f, 0.f};
                if constexpr (QK8) {
// fp8 mma is sm_89+; older fatbin passes just need this body to compile.
//
// The `#else` below BELONGS to the `if constexpr`, not to this `#if`. There is
// no arch fallback: on sm_86 this block is empty, d[] stays zero, and the
// kernel stores zeros while its launch reports success (- it did
// exactly that on the qwen35 fp8 verify arm, undetected until the spec parity
// gate). The comment that used to sit here claimed "only ever launches on the
// cc12 F8 geometry"; nothing enforced it. What enforces it now is
// pd_fp8_mma_ok() in gemm/f32_qkv.cuh, which every election of a QK8/P8
// instantiation must call. Keep that true when adding one.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
                    for (uint32_t kk = 0; kk < head_dim; kk += 32u) {
                        uint32_t af[4];
                        const unsigned char* ap = s_q8
                            + (size_t)(r0 + (lane & 15u)) * KPB + kk
                            + ((lane >> 4) ? 16u : 0u);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];"
                                     : "=r"(af[0]), "=r"(af[1]), "=r"(af[2]), "=r"(af[3])
                                     : "r"((unsigned)__cvta_generic_to_shared(ap)));
                        uint32_t bfr[2];
                        const unsigned char* bp = kbuf
                            + (size_t)(p0 + (lane & 7u)) * KPB + kk
                            + (((lane >> 3) & 1u) ? 16u : 0u);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0,%1}, [%2];"
                                     : "=r"(bfr[0]), "=r"(bfr[1])
                                     : "r"((unsigned)__cvta_generic_to_shared(bp)));
                        asm volatile(
                            "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                            : "r"(af[0]), "r"(af[1]), "r"(af[2]), "r"(af[3]),
                              "r"(bfr[0]), "r"(bfr[1]));
                    }
#endif
                } else {
                const unsigned char* krow = kbuf
                    + (size_t)(p0 + (lane >> 2)) * KPB + 2u * (lane & 3u);
                for (uint32_t kk = 0; kk < head_dim; kk += 16u) {
                    uint32_t af[4];
                    const __half* ap = s_q + (size_t)(r0 + (lane & 15u)) * KP
                                     + kk + ((lane >> 4) ? 8u : 0u);
                    pd_ldm_x4(af, (const unsigned char*)ap);
                    uint32_t bfr[2];
                    const __half2 b0 = __half2(__nv_cvt_fp8x2_to_halfraw2(
                        *(const unsigned short*)(krow + kk), __NV_E4M3));
                    const __half2 b1 = __half2(__nv_cvt_fp8x2_to_halfraw2(
                        *(const unsigned short*)(krow + kk + 8u), __NV_E4M3));
                    bfr[0] = *(const uint32_t*)&b0;
                    bfr[1] = *(const uint32_t*)&b1;
                    pd_fa_mma16(d, af[0], af[1], af[2], af[3], bfr[0], bfr[1]);
                }
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
                        if (pp >= n_t || j >= nrows || gpos > pos_r[j] || gpos < first_r[j])
                            v = -INFINITY;
                        s_p[rr * PP + pp] = v;
                    }
                }
            }
        }
        __syncthreads();
#ifdef PD_KRS_DUMP
        // rung-4 debug (debug-only define): CTA (0,0,0)'s first tile score
        // strip + (rows 16*33-64..) the FINAL per-row m/l after the last
        // tile - read via cudaMemcpyFromSymbol.
#ifndef PD_KRS_DUMP_TAIL
        if (kvh == 0u && c == 0u && s == 0u && t0 == lo) {
            for (uint32_t i = tid; i < Mp * PP; i += 256u)
                pd_krs_dump[i] = s_p[i];
        }
#endif
        __syncthreads();
#endif
#ifdef PD_KRS_DUMP_TAIL
        // g4 sub-16-tail hunt: the FAILING tile's e4m3 P strip (CTA
        // kvh=1,c=14,s=0 tail) - pad bytes past n_t must be zero.
        (void)0;
#endif
        // warp-parallel softmax (ks3 form); f16 strip zeroed to PTv (not
        // PT) - the o-mma's k=16 steps read that far when PT % 16 != 0
        for (uint32_t rr = warp; rr < Mp; rr += 8u) {
            const float m_old = s_m[rr];
            float mx = m_old;
            for (uint32_t pp = lane; pp < n_t; pp += 32u)
                mx = fmaxf(mx, s_p[rr * PP + pp]);
            #pragma unroll
            for (uint32_t o = 16; o; o >>= 1)
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float corr = (mx == -INFINITY) ? 1.f : __expf(m_old - mx);
            float ls = 0.f;
            for (uint32_t pp = lane; pp < n_t; pp += 32u) {
                const float sp = s_p[rr * PP + pp];
                const float w = (sp == -INFINITY) ? 0.f : __expf(sp - mx);
                // P8: e4m3 P (w in [0,1]; l stays the exact-sum denominator
                // - the class vLLM ships: L from float p, dot on P.to(fp8))
                if constexpr (P8) s_pf8[rr * FPB + pp] = __nv_fp8_e4m3(w).__x;
                else s_pf[rr * FP + pp] = __float2half(w);
                ls += w;
            }
            for (uint32_t pp = n_t + lane; pp < PTv; pp += 32u) {
                if constexpr (P8) s_pf8[rr * FPB + pp] = 0u;  // e4m3 0x00 = +0.0
                else s_pf[rr * FP + pp] = __half(0.f);
            }
            #pragma unroll
            for (uint32_t o = 16; o; o >>= 1)
                ls += __shfl_xor_sync(0xffffffffu, ls, o);
            if (lane == 0) {
                s_corr[rr] = corr;
                s_l[rr] = s_l[rr] * corr + ls;
                s_m[rr] = mx;
            }
        }
#ifdef PD_KRS_DUMP
        // g4 sub-16-tail hunt: the failing tile's P strip (e4m3 bytes as
        // floats) + this tile's s_corr, CTA (1,14,0), TAIL tile only.
        __syncthreads();
        if (kvh == 1u && c == 14u && s == 0u && t0 + PT >= hi) {
            if constexpr (P8) {
                for (uint32_t i = tid; i < Mp * FPB && i < 500u; i += 256u)
                    pd_krs_dump[i] = (float)__half(__nv_cvt_fp8_to_halfraw(
                        s_pf8[i], __NV_E4M3));
            }
            if (tid < Mp) pd_krs_dump[500u + tid] = s_corr[tid];
        }
        __syncthreads();
#endif
        // KVS: V's group drains only now - it rode behind score+softmax.
        // The barrier below publishes the softmax strip AND V block-wide.
        if constexpr (KVS) { if (pf) pd_attn_cpa_wait2(); else pd_attn_cpa_wait0(); }
        __syncthreads();
        // o update: unchanged (A = p f16 strip, B = expanded V via .trans)
        {
            const uint32_t slices = head_dim / 64u;
            const uint32_t tasks = mt * slices;
            #pragma unroll
            for (uint32_t ti = 0; ti < TPW; ++ti) {
                const uint32_t task = warp + ti * 8u;
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
                if constexpr (P8) {
                // P8 o-mma: e4m3 P x raw e4m3 V on m16n8k32 f32-accum.
                // A-frags via the plain b16 ldmatrix on the BYTE strip
                // (u16 = 2 adjacent positions - the QK8 score recipe on
                // the A side); B-frags: lane l <- V[kk+4(l&3)+i][ch]
                // i=0..3 (b0) / +16 (b1), ch = n_base+8*sub+(l>>2) - 8
                // byte loads + 6 perms per k32 pair, zero cvts. Short
                // tiles read the zeroed pad (P strip and V rows zero to
                // PTv; e4m3 0x00 is exact +0.0).
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
                for (uint32_t kk = 0; kk < n_t; kk += 32u) {
                    uint32_t af[4];
                    pd_ldm_x4(af, s_pf8 + (size_t)(r0 + (lane & 15u)) * FPB
                                  + kk + ((lane >> 4) ? 16u : 0u));
                    const unsigned char* vr0 = vbuf
                        + (size_t)(kk + 4u * (lane & 3u)) * KPB
                        + n_base + (lane >> 2);
                    #pragma unroll
                    for (uint32_t sub = 0; sub < 8u; ++sub) {
                        const unsigned char* vc = vr0 + sub * 8u;
#ifdef PD_KRS_SWAPBB
                        // within-u16 byte-order experiment (rung-4 probe):
                        // u16 = (p+1, p) instead of (p, p+1)
                        const uint32_t w0 = __byte_perm(
                            __byte_perm(vc[0], vc[KPB], 0x0004u),
                            __byte_perm(vc[2u * KPB], vc[3u * KPB], 0x0004u), 0x5410u);
                        const uint32_t w1 = __byte_perm(
                            __byte_perm(vc[16u * KPB], vc[17u * KPB], 0x0004u),
                            __byte_perm(vc[18u * KPB], vc[19u * KPB], 0x0004u), 0x5410u);
#elif defined(PD_KRS_SWAPB)
                        // k-half order experiment (rung-4 probe)
                        const uint32_t w1 = __byte_perm(
                            __byte_perm(vc[0], vc[KPB], 0x0040u),
                            __byte_perm(vc[2u * KPB], vc[3u * KPB], 0x0040u), 0x5410u);
                        const uint32_t w0 = __byte_perm(
                            __byte_perm(vc[16u * KPB], vc[17u * KPB], 0x0040u),
                            __byte_perm(vc[18u * KPB], vc[19u * KPB], 0x0040u), 0x5410u);
#else
                        const uint32_t w0 = __byte_perm(
                            __byte_perm(vc[0], vc[KPB], 0x0040u),
                            __byte_perm(vc[2u * KPB], vc[3u * KPB], 0x0040u), 0x5410u);
                        const uint32_t w1 = __byte_perm(
                            __byte_perm(vc[16u * KPB], vc[17u * KPB], 0x0040u),
                            __byte_perm(vc[18u * KPB], vc[19u * KPB], 0x0040u), 0x5410u);
#endif
                        asm volatile(
                            "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                            : "+f"(o_acc[ti][sub][0]), "+f"(o_acc[ti][sub][1]),
                              "+f"(o_acc[ti][sub][2]), "+f"(o_acc[ti][sub][3])
#ifdef PD_KRS_SWAPA
                            : "r"(af[2]), "r"(af[3]), "r"(af[0]), "r"(af[1]),
#else
                            : "r"(af[0]), "r"(af[1]), "r"(af[2]), "r"(af[3]),
#endif
                              "r"(w0), "r"(w1));
                    }
                }
#endif
                } else {
                for (uint32_t kk = 0; kk < n_t; kk += 16u) {
                    uint32_t af[4];
                    const __half* ap = s_pf + (size_t)(r0 + (lane & 15u)) * FP
                                     + kk + ((lane >> 4) ? 8u : 0u);
                    pd_ldm_x4(af, (const unsigned char*)ap);
                    // VR: raw-V B-frags at the mma seam (kb recipe in the PV
                    // orientation): lane l <- V[kk+2(l&3)(+1)][ch] with
                    // ch = n_base+8*sub+(l>>2) - the halves run along
                    // POSITIONS (pitch KPB), so each register is 2 byte
                    // loads + a pack + one fp8x2 cvt (exact = bit-equal to
                    // the expanded route). Row pointers hoisted per kk.
                    const unsigned char* vr0 = vbuf
                        + (size_t)(kk + 2u * (lane & 3u)) * KPB
                        + n_base + (lane >> 2);
                    #pragma unroll
                    for (uint32_t sub = 0; sub < 8u; ++sub) {
                        uint32_t bfr[2];
                        if constexpr (VR) {
                            const unsigned char* vc = vr0 + sub * 8u;
                            const uint32_t w0 = __byte_perm(vc[0], vc[KPB], 0x0040u);
                            const uint32_t w1 = __byte_perm(vc[8u * KPB], vc[9u * KPB], 0x0040u);
                            const __half2 b0 = __half2(__nv_cvt_fp8x2_to_halfraw2(
                                (__nv_fp8x2_storage_t)w0, __NV_E4M3));
                            const __half2 b1 = __half2(__nv_cvt_fp8x2_to_halfraw2(
                                (__nv_fp8x2_storage_t)w1, __NV_E4M3));
                            bfr[0] = *(const uint32_t*)&b0;
                            bfr[1] = *(const uint32_t*)&b1;
                        } else {
                            const __half* bp = s_v + (size_t)(kk + (lane & 15u)) * KP
                                             + n_base + sub * 8u;
                            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                                         : "=r"(bfr[0]), "=r"(bfr[1])
                                         : "r"((unsigned)__cvta_generic_to_shared(bp)));
                        }
                        pd_fa_mma16(o_acc[ti][sub], af[0], af[1], af[2], af[3],
                                    bfr[0], bfr[1]);
                    }
                }
                }
            }
        }
        __syncthreads();
    }
    {
        const uint32_t slices = head_dim / 64u;
        const uint32_t tasks = mt * slices;
        #pragma unroll
        for (uint32_t ti = 0; ti < TPW; ++ti) {
            const uint32_t task = warp + ti * 8u;
            if (task >= tasks) break;
            const uint32_t tm = task / slices, sl = task % slices;
            const uint32_t r0 = tm * 16u, n_base = sl * 64u;
            #pragma unroll
            for (uint32_t sub = 0; sub < 8u; ++sub) {
                #pragma unroll
                for (uint32_t half = 0; half < 2u; ++half) {
                    const uint32_t rr = r0 + (lane >> 2) + half * 8u;
                    const uint32_t j = rr / G, g = rr % G;
                    if (j >= nrows) continue;
                    if (fin) {
                        const float l = s_l[rr];
                        const size_t off = ((size_t)(rb + j) * n_heads + kvh * G + g) * head_dim
                                         + n_base + sub * 8u + 2u * (lane & 3u);
                        if (fe4s) {
                            unsigned char* dq = (unsigned char*)out_o + off;
                            dq[0] = __nv_fp8_e4m3(o_acc[ti][sub][half * 2u] / l).__x;
                            dq[1] = __nv_fp8_e4m3(o_acc[ti][sub][half * 2u + 1u] / l).__x;
                        } else {
                            float* dst = out_o + off;
                            dst[0] = o_acc[ti][sub][half * 2u] / l;
                            dst[1] = o_acc[ti][sub][half * 2u + 1u] / l;
                        }
                    } else {
                        const size_t hg = (size_t)(kvh * G + g) * rows + (rb + j);
                        float* dst = out_o + (hg * n_splits + s) * head_dim
                                   + n_base + sub * 8u + 2u * (lane & 3u);
                        dst[0] = o_acc[ti][sub][half * 2u];
                        dst[1] = o_acc[ti][sub][half * 2u + 1u];
                    }
                }
            }
        }
    }
    if (!fin && tid < Mp && tid < M) {
        const uint32_t rr = tid, j = rr / G, g = rr % G;
        if (j < nrows) {
            const size_t hg = (size_t)(kvh * G + g) * rows + (rb + j);
            out_ml[(hg * n_splits + s) * 2 + 0] = s_m[rr];
            out_ml[(hg * n_splits + s) * 2 + 1] = s_l[rr];
        }
#ifdef PD_KRS_DUMP
        // final per-row m/l (dump tail rows: [496..511] m, [512..527] l)
        if (kvh == 0u && c == 0u && s == 0u) {
            pd_krs_dump[16 * 33 - 32 + rr] = s_m[rr];
            pd_krs_dump[16 * 33 - 16 + rr] = s_l[rr];
        }
#endif
    }
#else
    (void)q; (void)pool_k; (void)pool_v; (void)out_o; (void)out_ml;
    (void)positions; (void)slots; (void)block_tables; (void)blocks_per_slot;
    (void)n_heads; (void)n_kv_heads; (void)head_dim_rt; (void)kv_dim;
    (void)swa_window; (void)n_splits; (void)rows; (void)k1; (void)scale;
#endif
}

// ── spec-FA krs: fp8-RESIDENT K ─────────────────────────────────────────
// The F8 route stages raw bytes then does an IN-PLACE f16 expansion round
// trip for both sides every tile (read strip -> sync -> rewrite as halves).
// Here K never expands: it stays raw e4m3 in a compact [PT][HD+16]-byte
// region and the score B-frags are built by 2x u16 loads + fp8x2 cvt per
// mma - the ldmatrix.x2 fragment map is lane l <- K[p0+(l>>2)][kk+2(l&3)]
// / +8 (v3c's kb recipe), so at equal PT this is BIT-equal to the F8 path
// (e4m3->f16 exact, same operand values and fold order; KR32 verified
// BITEQ). V keeps the expansion - its .trans ldmatrix needs b16
// (ldmatrix.b8 is not on sm_120). What it buys
// (measured at the 128-row/k1=4 serve point): K's expansion wave
// gone + ~8.4KB smem freed -> PT ladder headroom. SWA hd256: KR40 sp2
// 214.7 vs PT32-SB 221.2us (occ 2). GLB hd512: KR32 sp4 287.9 vs
// PT16-SB 410.4us (PT32 halves the occ-1 arm's un-overlapped
// tile stalls and now FITS the 99KB cap). PTv pads the V region to a
// 16-row multiple so the o-mma's k=16 .trans never reads outside it
// (PT=40); V rows [n_t, PTv) and the s_pf strip zero to PTv - stale
// f16/NaN would poison the 0-weight products. PT!=32 shifts tile
// boundaries = split-count numeric class, serve-gate arbitrated.
// SB + F8 + PAD specialized (the sm_120 elections that route here).
template <uint32_t PT, uint32_t TPW, uint32_t HD, uint32_t GL, bool VR = false>
__global__ void __launch_bounds__(256, 1) pd_attn_spec_fa_lco_kernel(
    const float* __restrict__ q, const __half* __restrict__ pool_k,
    const __half* __restrict__ pool_v, float* __restrict__ out_o, float* __restrict__ out_ml,
    const unsigned int* __restrict__ positions, const unsigned int* __restrict__ slots,
    const uint32_t* __restrict__ block_tables, uint32_t blocks_per_slot,
    uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim_rt,
    uint32_t kv_dim, uint32_t swa_window, uint32_t n_splits,
    uint32_t rows, uint32_t k1, float scale,
    const float* __restrict__ sinks, float* __restrict__ out_f,
    unsigned int* __restrict__ tickets) {
#if PD_FA_OK
    PD_PDL_ARM();
    const bool fin = (n_splits & 0x80000000u) != 0u;
    // FE4S - fin stores e4m3 at STATIC scale 1.0 into
    // out_o reinterpreted as the i8 quantized plane (pf_e4q); the wo-in
    // quantize_e4m3_row launch disappears and the GEMM's xrs is a ones
    // vector. Runtime-uniform branch, no new instantiations.
    const bool fe4s = (n_splits & 0x40000000u) != 0u;
    n_splits &= 0x3fffffffu;
    const uint32_t kvh = blockIdx.x, c = blockIdx.y, s = blockIdx.z;
    constexpr uint32_t head_dim = HD;
    constexpr uint32_t G = 1u << GL;
    const uint32_t M = k1 * G;
    const uint32_t mt = (M + 15u) / 16u;
    const uint32_t Mp = mt * 16u;
    const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
    const uint32_t rb = c * k1;
    const uint32_t nrows = (rows - rb) < k1 ? (rows - rb) : k1;
    const uint32_t slot = slots ? slots[rb] : rb;

    uint32_t pos_r[8], first_r[8];
    #pragma unroll
    for (uint32_t j = 0; j < 8u; ++j) {
        if (j < nrows) {
            const uint32_t pj = positions[rb + j];
            pos_r[j] = pj;
            first_r[j] = (swa_window > 0 && pj + 1 > swa_window) ? (pj + 1 - swa_window) : 0u;
        } else { pos_r[j] = 0u; first_r[j] = 1u; }
    }
    const uint32_t lo0 = first_r[0];
    const uint32_t pos_max = pos_r[nrows - 1];
    const uint32_t n_pos = pos_max + 1u - lo0;
    const uint32_t chunk = (n_pos + n_splits - 1u) / n_splits;
    const uint32_t lo = s * chunk;
    uint32_t hi = lo + chunk;
    if (hi > n_pos) hi = n_pos;

    constexpr uint32_t KP = head_dim + 8u;        // f16 row stride (halves)
    constexpr uint32_t KPB = head_dim + 16u;      // K byte row stride (16B-aligned)
    constexpr uint32_t PTv = (PT + 15u) & ~15u;   // V rows padded to the mma k=16
    constexpr uint32_t PP = PT + 1u;
    constexpr uint32_t FP = PT + 8u;
    extern __shared__ __align__(16) unsigned char fa_lco_sm[];
    __half* s_q = (__half*)fa_lco_sm;                           // [Mp][KP]
    unsigned char* s_kb = (unsigned char*)(s_q + (size_t)Mp * KP);  // [PT][KPB] raw e4m3
    // VR/!VR layout split: see the krs twin
    __half* s_v = (__half*)(s_kb + (size_t)PT * KPB);           // [PTv][KP] (!VR)
    unsigned char* s_vb = (unsigned char*)s_v;                  // [PTv][KPB] (VR)
    float* s_p = (float*)((unsigned char*)s_v
                          + (VR ? (size_t)PTv * KPB : (size_t)PTv * KP * 2u)); // [Mp][PP]
    float* s_m = s_p + (size_t)Mp * PP;                         // [Mp] x3
    float* s_l = s_m + Mp;
    float* s_corr = s_l + Mp;
    __half* s_pf = (__half*)(s_corr + Mp);                      // [Mp][FP] f16

    for (uint32_t i = tid; i < Mp * head_dim; i += 256u) {
        const uint32_t rh = i / head_dim, e = i % head_dim;
        const uint32_t j = rh / G, g = rh % G;
        s_q[(size_t)rh * KP + e] = (j < nrows)
            ? __float2half(q[((size_t)(rb + j) * n_heads + (size_t)kvh * G + g) * head_dim + e])
            : __half(0.f);
    }
    for (uint32_t i = tid; i < Mp; i += 256u) { s_m[i] = -INFINITY; s_l[i] = 0.f; }

    float o_acc[TPW][8][4];
    #pragma unroll
    for (uint32_t a = 0; a < TPW; ++a)
        #pragma unroll
        for (uint32_t b = 0; b < 8u; ++b)
            #pragma unroll
            for (uint32_t cc2 = 0; cc2 < 4u; ++cc2) o_acc[a][b][cc2] = 0.f;

    const uint32_t* bt = block_tables + (size_t)slot * blocks_per_slot;
    constexpr uint32_t lines = head_dim >> 4;     // 16-byte e4m3 lines per row
    auto stage = [&](uint32_t t0) {
        const uint32_t n_t = hi - t0 < PT ? hi - t0 : PT;
        for (uint32_t i = tid; i < 2u * n_t * lines; i += 256u) {
            const uint32_t kvsel = i / (n_t * lines);
            const uint32_t jj = i - kvsel * n_t * lines;
            const uint32_t p = jj / lines, l = jj - p * lines;
            const uint32_t gpos = lo0 + t0 + p;
            const uint32_t blk = bt[gpos >> 4];
            const unsigned char* src8 = (const unsigned char*)(kvsel ? pool_v : pool_k)
                + (size_t)blk * 16u * kv_dim + (size_t)(gpos & 15u) * kv_dim
                + (size_t)kvh * head_dim;
            char* dst = kvsel
                ? (VR ? (char*)s_vb + (size_t)p * KPB                     // V stays raw (VR)
                      : (char*)s_v + (size_t)PTv * KP + (size_t)p * head_dim) // V byte strip
                : (char*)s_kb + (size_t)p * KPB;                          // K stays raw
            pd_attn_cpa16(dst + l * 16u, (const char*)src8 + l * 16u);
        }
        pd_attn_cpa_commit();
    };

    __syncthreads();
    for (uint32_t t0 = lo; t0 < hi; t0 += PT) {
        const uint32_t n_t = hi - t0 < PT ? hi - t0 : PT;
        stage(t0);
        pd_attn_cpa_wait0();
        __syncthreads();

        // VR: no expansion at all - the o-mma consumes raw bytes. Rows
        // [n_t, PTv) still must zero on every short tile: stale e4m3 0x7f
        // decodes to NaN and 0-weight x NaN poisons o (e4m3 0x00 is exact
        // +0.0, so zero BYTES give zero halves). No extra sync: the o-mma
        // reads sit behind the score/softmax __syncthreads chain.
        if constexpr (VR) {
            if (n_t < PTv) {
                constexpr uint32_t rl = KPB >> 4;
                for (uint32_t i = tid; i < (PTv - n_t) * rl; i += 256u)
                    ((uint4*)(s_vb + (size_t)(n_t + i / rl) * KPB))[i % rl] =
                        make_uint4(0u, 0u, 0u, 0u);
            }
        } else {
        // in-place e4m3 -> f16 expansion, V only (K is consumed raw). Rows
        // [n_t, PTv) zero: stale e4m3 can decode to NaN and 0-weight x NaN
        // poisons the o accumulate.
            constexpr uint32_t rchunks = PTv * lines;
            constexpr uint32_t CHW = (rchunks + 255u) / 256u;
            uint4 rg[CHW];
            const char* strip = (const char*)s_v + (size_t)PTv * KP;
            #pragma unroll
            for (uint32_t ci = 0; ci < CHW; ++ci) {
                const uint32_t cc = tid + ci * 256u;
                if (cc >= rchunks) break;
                const uint32_t p = cc / lines;
                rg[ci] = (p < n_t) ? *(const uint4*)(strip + (size_t)cc * 16u)
                                   : make_uint4(0u, 0u, 0u, 0u);
            }
            __syncthreads();  // all strip reads before the overlapping writes
            #pragma unroll
            for (uint32_t ci = 0; ci < CHW; ++ci) {
                const uint32_t cc = tid + ci * 256u;
                if (cc >= rchunks) break;
                const uint32_t p = cc / lines, l = cc - p * lines;
                __half* dst = s_v + (size_t)p * KP + l * 16u;
                const uint32_t* w = (const uint32_t*)&rg[ci];
                #pragma unroll
                for (uint32_t qd = 0; qd < 4u; ++qd) {
                    ((__half2*)dst)[qd * 2u] = __half2(__nv_cvt_fp8x2_to_halfraw2(
                        (__nv_fp8x2_storage_t)(w[qd] & 0xffffu), __NV_E4M3));
                    ((__half2*)dst)[qd * 2u + 1u] = __half2(__nv_cvt_fp8x2_to_halfraw2(
                        (__nv_fp8x2_storage_t)(w[qd] >> 16), __NV_E4M3));
                }
            }
            __syncthreads();  // expanded halves visible before the mmas
        }

        // scores: A = Q via ldmatrix.x4, B = raw K bytes + cvt (the krs swap)
        {
            const uint32_t tasks = mt * (PT / 8u);
            for (uint32_t task = warp; task < tasks; task += 8u) {
                const uint32_t tm = task / (PT / 8u), cs = task % (PT / 8u);
                const uint32_t r0 = tm * 16u, p0 = cs * 8u;
                float d[4] = {0.f, 0.f, 0.f, 0.f};
                const unsigned char* krow = s_kb
                    + (size_t)(p0 + (lane >> 2)) * KPB + 2u * (lane & 3u);
                for (uint32_t kk = 0; kk < head_dim; kk += 16u) {
                    uint32_t af[4];
                    const __half* ap = s_q + (size_t)(r0 + (lane & 15u)) * KP
                                     + kk + ((lane >> 4) ? 8u : 0u);
                    pd_ldm_x4(af, (const unsigned char*)ap);
                    uint32_t bfr[2];
                    const __half2 b0 = __half2(__nv_cvt_fp8x2_to_halfraw2(
                        *(const unsigned short*)(krow + kk), __NV_E4M3));
                    const __half2 b1 = __half2(__nv_cvt_fp8x2_to_halfraw2(
                        *(const unsigned short*)(krow + kk + 8u), __NV_E4M3));
                    bfr[0] = *(const uint32_t*)&b0;
                    bfr[1] = *(const uint32_t*)&b1;
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
                        if (pp >= n_t || j >= nrows || gpos > pos_r[j] || gpos < first_r[j])
                            v = -INFINITY;
                        s_p[rr * PP + pp] = v;
                    }
                }
            }
        }
        __syncthreads();
        // warp-parallel softmax (ks3 form); f16 strip zeroed to PTv (not
        // PT) - the o-mma's k=16 steps read that far when PT % 16 != 0
        for (uint32_t rr = warp; rr < Mp; rr += 8u) {
            const float m_old = s_m[rr];
            float mx = m_old;
            for (uint32_t pp = lane; pp < n_t; pp += 32u)
                mx = fmaxf(mx, s_p[rr * PP + pp]);
            #pragma unroll
            for (uint32_t o = 16; o; o >>= 1)
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float corr = (mx == -INFINITY) ? 1.f : __expf(m_old - mx);
            float ls = 0.f;
            for (uint32_t pp = lane; pp < n_t; pp += 32u) {
                const float sp = s_p[rr * PP + pp];
                const float w = (sp == -INFINITY) ? 0.f : __expf(sp - mx);
                s_pf[rr * FP + pp] = __float2half(w);
                ls += w;
            }
            for (uint32_t pp = n_t + lane; pp < PTv; pp += 32u)
                s_pf[rr * FP + pp] = __half(0.f);
            #pragma unroll
            for (uint32_t o = 16; o; o >>= 1)
                ls += __shfl_xor_sync(0xffffffffu, ls, o);
            if (lane == 0) {
                s_corr[rr] = corr;
                s_l[rr] = s_l[rr] * corr + ls;
                s_m[rr] = mx;
            }
        }
        __syncthreads();
        // o update: unchanged (A = p f16 strip, B = expanded V via .trans)
        {
            const uint32_t slices = head_dim / 64u;
            const uint32_t tasks = mt * slices;
            #pragma unroll
            for (uint32_t ti = 0; ti < TPW; ++ti) {
                const uint32_t task = warp + ti * 8u;
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
                    // VR: raw-V B-frags at the mma seam (kb recipe in the PV
                    // orientation): lane l <- V[kk+2(l&3)(+1)][ch] with
                    // ch = n_base+8*sub+(l>>2) - the halves run along
                    // POSITIONS (pitch KPB), so each register is 2 byte
                    // loads + a pack + one fp8x2 cvt (exact = bit-equal to
                    // the expanded route). Row pointers hoisted per kk.
                    const unsigned char* vr0 = s_vb
                        + (size_t)(kk + 2u * (lane & 3u)) * KPB
                        + n_base + (lane >> 2);
                    #pragma unroll
                    for (uint32_t sub = 0; sub < 8u; ++sub) {
                        uint32_t bfr[2];
                        if constexpr (VR) {
                            const unsigned char* vc = vr0 + sub * 8u;
                            const uint32_t w0 = __byte_perm(vc[0], vc[KPB], 0x0040u);
                            const uint32_t w1 = __byte_perm(vc[8u * KPB], vc[9u * KPB], 0x0040u);
                            const __half2 b0 = __half2(__nv_cvt_fp8x2_to_halfraw2(
                                (__nv_fp8x2_storage_t)w0, __NV_E4M3));
                            const __half2 b1 = __half2(__nv_cvt_fp8x2_to_halfraw2(
                                (__nv_fp8x2_storage_t)w1, __NV_E4M3));
                            bfr[0] = *(const uint32_t*)&b0;
                            bfr[1] = *(const uint32_t*)&b1;
                        } else {
                            const __half* bp = s_v + (size_t)(kk + (lane & 15u)) * KP
                                             + n_base + sub * 8u;
                            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                                         : "=r"(bfr[0]), "=r"(bfr[1])
                                         : "r"((unsigned)__cvta_generic_to_shared(bp)));
                        }
                        pd_fa_mma16(o_acc[ti][sub], af[0], af[1], af[2], af[3],
                                    bfr[0], bfr[1]);
                    }
                }
            }
        }
        __syncthreads();
    }
    {
        const uint32_t slices = head_dim / 64u;
        const uint32_t tasks = mt * slices;
        #pragma unroll
        for (uint32_t ti = 0; ti < TPW; ++ti) {
            const uint32_t task = warp + ti * 8u;
            if (task >= tasks) break;
            const uint32_t tm = task / slices, sl = task % slices;
            const uint32_t r0 = tm * 16u, n_base = sl * 64u;
            #pragma unroll
            for (uint32_t sub = 0; sub < 8u; ++sub) {
                #pragma unroll
                for (uint32_t half = 0; half < 2u; ++half) {
                    const uint32_t rr = r0 + (lane >> 2) + half * 8u;
                    const uint32_t j = rr / G, g = rr % G;
                    if (j >= nrows) continue;
                    if (fin) {
                        const float l = s_l[rr];
                        const size_t off = ((size_t)(rb + j) * n_heads + kvh * G + g) * head_dim
                                         + n_base + sub * 8u + 2u * (lane & 3u);
                        if (fe4s) {
                            unsigned char* dq = (unsigned char*)out_o + off;
                            dq[0] = __nv_fp8_e4m3(o_acc[ti][sub][half * 2u] / l).__x;
                            dq[1] = __nv_fp8_e4m3(o_acc[ti][sub][half * 2u + 1u] / l).__x;
                        } else {
                            float* dst = out_o + off;
                            dst[0] = o_acc[ti][sub][half * 2u] / l;
                            dst[1] = o_acc[ti][sub][half * 2u + 1u] / l;
                        }
                    } else {
                        const size_t hg = (size_t)(kvh * G + g) * rows + (rb + j);
                        float* dst = out_o + (hg * n_splits + s) * head_dim
                                   + n_base + sub * 8u + 2u * (lane & 3u);
                        dst[0] = o_acc[ti][sub][half * 2u];
                        dst[1] = o_acc[ti][sub][half * 2u + 1u];
                    }
                }
            }
        }
    }
    if (!fin && tid < Mp && tid < M) {
        const uint32_t rr = tid, j = rr / G, g = rr % G;
        if (j < nrows) {
            const size_t hg = (size_t)(kvh * G + g) * rows + (rb + j);
            out_ml[(hg * n_splits + s) * 2 + 0] = s_m[rr];
            out_ml[(hg * n_splits + s) * 2 + 1] = s_l[rr];
        }
#ifdef PD_KRS_DUMP
        // final per-row m/l (dump tail rows: [496..511] m, [512..527] l)
        if (kvh == 0u && c == 0u && s == 0u) {
            pd_krs_dump[16 * 33 - 32 + rr] = s_m[rr];
            pd_krs_dump[16 * 33 - 16 + rr] = s_l[rr];
        }
#endif
    }

    // ── LCO merge (last-CTA-out): after this CTA's partial + ml
    // writes land, bump the (kvh, c) ticket; the last split CTA re-reads all
    // splits' partials and folds them exactly like
    // pd_attn_decode_batch_combine_kernel (same s order, same sink fold) -
    // bit-identical final rows, and the separate combine launch (its grid +
    // PDL-wait span) disappears from the tick. atomicInc wraps at
    // n_splits-1: no zeroing pass between launches, graph-replay-safe.
    if (!fin && n_splits > 1u) {
        __threadfence();
        __syncthreads();
        __shared__ unsigned int lco_last;
        if (tid == 0u)
            lco_last = (atomicInc(&tickets[(size_t)kvh * gridDim.y + c],
                                  n_splits - 1u) == n_splits - 1u) ? 1u : 0u;
        __syncthreads();
        if (lco_last != 0u) {
            __threadfence();
            for (uint32_t i2 = tid; i2 < M * head_dim; i2 += 256u) {
                const uint32_t rr2 = i2 / head_dim, d2 = i2 % head_dim;
                const uint32_t j2 = rr2 / G, g2 = rr2 % G;
                if (j2 >= nrows) continue;
                const uint32_t h2 = kvh * G + g2;
                const size_t pb = ((size_t)h2 * rows + (rb + j2)) * n_splits;
                float gm = sinks[h2];
                for (uint32_t ss = 0; ss < n_splits; ++ss)
                    gm = fmaxf(gm, out_ml[(pb + ss) * 2 + 0]);
                float acc2 = 0.0f, l2 = 0.0f;
                for (uint32_t ss = 0; ss < n_splits; ++ss) {
                    const float m2 = out_ml[(pb + ss) * 2 + 0];
                    const float ls2 = out_ml[(pb + ss) * 2 + 1];
                    const float sc2 = __expf(m2 - gm);
                    acc2 += sc2 * out_o[(pb + ss) * head_dim + d2];
                    l2 += sc2 * ls2;
                }
                l2 += __expf(sinks[h2] - gm);
                out_f[((size_t)(rb + j2) * n_heads + h2) * head_dim + d2] = acc2 / l2;
            }
        }
    }
#else
    (void)sinks; (void)out_f; (void)tickets;
    (void)q; (void)pool_k; (void)pool_v; (void)out_o; (void)out_ml;
    (void)positions; (void)slots; (void)block_tables; (void)blocks_per_slot;
    (void)n_heads; (void)n_kv_heads; (void)head_dim_rt; (void)kv_dim;
    (void)swa_window; (void)n_splits; (void)rows; (void)k1; (void)scale;
#endif
}

// Dense-decode v3 (B200): the GQA walk's staging skeleton
// (+8-half padded rows, hoisted per-row staging addresses) with the two
// scalar passes replaced by
// warp-task HMMA (m16n8k16, f32 accumulate) - the kernel is TOTAL-ISSUE
// bound and the scalar convert+FMA stream is ~2 inst/element; mma is ~50x
// fewer instructions for the same math. Frag patterns lifted from
// pd_attn_spec_fa_kernel (K-side B-frag no .trans, V-side .trans);
// constexpr HD/G per the prefill-FA measurement (runtime div/mod +53% inst).
// q and the tile weights ride fp16 (f32 accumulate) - same numerics class
// as the shipped spec-FA attention; rel-tol validated vs the walk.
template <uint32_t HD, uint32_t G, uint32_t TILE>
__global__ void pd_attn_decode_v3_kernel(
    const float* __restrict__ q, const __half* __restrict__ pool_k,
    const __half* __restrict__ pool_v, float* __restrict__ out_o,
    float* __restrict__ out_ml, const unsigned int* __restrict__ positions,
    const unsigned int* __restrict__ slots,
    const uint32_t* __restrict__ block_tables, uint32_t blocks_per_slot,
    uint32_t kv_dim, uint32_t swa_window, uint32_t n_splits, float scale) {
#if PD_FA_OK
    const uint32_t kvh = blockIdx.x, b = blockIdx.y, sp = blockIdx.z;
    const uint32_t slot = slots ? slots[b] : b;
    const uint32_t d = threadIdx.x, nth = blockDim.x;
    const uint32_t warp = d >> 5, lane = d & 31u;
    const uint32_t n_heads = G * gridDim.x;

    const uint32_t pos = positions[b];
    const uint32_t first_pos =
        (swa_window > 0 && pos + 1 > swa_window) ? (pos + 1 - swa_window) : 0;
    const uint32_t n_pos = pos + 1 - first_pos;
    const uint32_t chunk = (n_pos + n_splits - 1) / n_splits;
    const uint32_t lo = sp * chunk;
    uint32_t hi = lo + chunk;
    if (hi > n_pos) hi = n_pos;

    extern __shared__ __align__(16) unsigned char v3_smraw[];
    constexpr uint32_t t_s = TILE + 1u;
    constexpr uint32_t row_e = HD + 8u;        // +8-half pad (0ebac09: 2.4x)
    constexpr uint32_t w_s = TILE + 8u;        // f16 weight-strip stride
    float* s_sc = (float*)v3_smraw;                     // [G][t_s]
    float* s_w = s_sc + (size_t)G * t_s;                // [G][t_s] (l-sum)
    __half* s_qh = (__half*)(s_w + (size_t)G * t_s);    // [16][row_e] f16 q
    __half* s_wh = s_qh + (size_t)16u * row_e;          // [16][w_s] f16 wts
    __half* s_kv = s_wh + (size_t)16u * w_s;            // [2][K,V][TILE][row_e]
    __shared__ float s_m[G], s_l[G], s_mnew[G], s_corr[G];

    const float* qb = q + (size_t)b * n_heads * HD;
    const uint32_t* bt = block_tables + (size_t)slot * blocks_per_slot;

    // f16 q tile: rows < G real, rows G..15 zero (mma pad rows)
    for (uint32_t i = d; i < 16u * row_e; i += nth) {
        const uint32_t r = i / row_e, c = i % row_e;
        float v = (r < G && c < HD)
            ? qb[((size_t)kvh * G + r) * HD + c] : 0.0f;
        s_qh[(size_t)r * row_e + c] = __float2half(v);
    }
    for (uint32_t i = d; i < 16u * w_s; i += nth) s_wh[i] = __half(0.f);
    if (d < G) { s_m[d] = -INFINITY; s_l[d] = 0.0f; }

    // o accumulators: warp owns a 32-dim slice (4 n8 subtiles x 4 f32)
    constexpr uint32_t NW_V = (HD / 32u < 8u) ? HD / 32u : 8u;  // V warps
    constexpr uint32_t SLICE = HD / NW_V;                        // dims/warp
    float o_acc[SLICE / 8u][4];
    #pragma unroll
    for (uint32_t i = 0; i < SLICE / 8u; ++i)
        #pragma unroll
        for (uint32_t j = 0; j < 4u; ++j) o_acc[i][j] = 0.0f;

    constexpr uint32_t lines = (HD * 2u) >> 4;
    auto stage = [&](uint32_t bf, uint32_t t0) {
        const uint32_t n_t = hi - t0 < TILE ? hi - t0 : TILE;
        // partial tiles: the mma passes read all 16 rows of the K/V tiles
        // (zero-weight folds only null a stale row if its bytes are finite -
        // 0 x Inf/NaN poisons o). Zero the tail rows before staging.
        if (n_t < TILE) {
            for (uint32_t i = d; i < 2u * (TILE - n_t) * lines; i += nth) {
                const uint32_t kvsel = i / ((TILE - n_t) * lines);
                const uint32_t j = i - kvsel * (TILE - n_t) * lines;
                const uint32_t p = n_t + j / lines, l = j % lines;
                *(uint4*)((char*)(s_kv
                    + ((size_t)(bf * 2u + kvsel) * TILE + p) * row_e) + l * 16u)
                    = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        const uint32_t rows = 2u * n_t;
        const uint32_t gsz = nth / rows ? nth / rows : 1u;
        const uint32_t r = d / gsz, lt = d % gsz;
        if (r < rows) {
            const uint32_t kvsel = r >= n_t ? 1u : 0u;
            const uint32_t p = kvsel ? r - n_t : r;
            const uint32_t gpos = first_pos + t0 + p;
            const uint32_t blk = bt[gpos >> 4];
            const char* src = (const char*)((kvsel ? pool_v : pool_k)
                + (size_t)blk * 16u * kv_dim + (size_t)(gpos & 15u) * kv_dim
                + (size_t)kvh * HD);
            char* dst = (char*)(s_kv + ((size_t)(bf * 2u + kvsel) * TILE + p) * row_e);
            #pragma unroll
            for (uint32_t l = lt; l < lines; l += gsz)
                pd_attn_cpa16(dst + l * 16u, src + l * 16u);
        }
        pd_attn_cpa_commit();
    };

    __syncthreads();
    if (lo < hi) stage(0u, lo);
    uint32_t bf = 0;
    for (uint32_t t0 = lo; t0 < hi; t0 += TILE, bf ^= 1u) {
        const uint32_t n_t = hi - t0 < TILE ? hi - t0 : TILE;
        const bool more = t0 + TILE < hi;
        if (more) stage(bf ^ 1u, t0 + TILE);
        if (more) pd_attn_cpa_wait1(); else pd_attn_cpa_wait0();
        __syncthreads();
        const __half* kbuf = s_kv + (size_t)(bf * 2u) * TILE * row_e;
        const __half* vbuf = s_kv + ((size_t)(bf * 2u) + 1u) * TILE * row_e;
        // scores: warps 0..TILE/8-1, one n8 col-subtile each
        if (warp < TILE / 8u) {
            const uint32_t p0 = warp * 8u;
            float dfr[4] = {0.f, 0.f, 0.f, 0.f};
            for (uint32_t kk = 0; kk < HD; kk += 16u) {
                uint32_t af[4];
                const __half* ap = s_qh + (size_t)(lane & 15u) * row_e
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
                const uint32_t rr = (lane >> 2) + half * 8u;
                #pragma unroll
                for (uint32_t cc = 0; cc < 2u; ++cc) {
                    const uint32_t pp = p0 + 2u * (lane & 3u) + cc;
                    if (rr < G && pp < n_t)
                        s_sc[rr * t_s + pp] = dfr[half * 2u + cc] * scale;
                }
            }
        }
        __syncthreads();
        {
            if (warp < G) {
                float v = (lane < n_t) ? s_sc[warp * t_s + lane] : -INFINITY;
                #pragma unroll
                for (uint32_t off = 16; off > 0; off >>= 1)
                    v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
                if (lane == 0) {
                    const float m_new = fmaxf(s_m[warp], v);
                    s_mnew[warp] = m_new;
                    s_corr[warp] = __expf(s_m[warp] - m_new);
                }
            }
        }
        __syncthreads();
        for (uint32_t idx = d; idx < G * TILE; idx += nth) {
            const uint32_t h = idx / TILE, p = idx % TILE;
            const float w = p < n_t
                ? __expf(s_sc[h * t_s + p] - s_mnew[h]) : 0.0f;
            if (p < n_t) s_w[h * t_s + p] = w;
            s_wh[h * w_s + p] = __float2half(w);   // zero past n_t (stale-tail)
        }
        __syncthreads();
        // o update: warp w < NW_V owns dims [w*SLICE, +SLICE)
        if (warp < NW_V) {
            const uint32_t n_base_w = warp * SLICE;
            #pragma unroll
            for (uint32_t half = 0; half < 2u; ++half) {
                const uint32_t rr = (lane >> 2) + half * 8u;
                const float corr = rr < G ? s_corr[rr] : 1.0f;
                #pragma unroll
                for (uint32_t sub = 0; sub < SLICE / 8u; ++sub) {
                    o_acc[sub][half * 2u] *= corr;
                    o_acc[sub][half * 2u + 1u] *= corr;
                }
            }
            #pragma unroll
            for (uint32_t kk = 0; kk < TILE; kk += 16u) {
                uint32_t af[4];
                const __half* ap = s_wh + (size_t)(lane & 15u) * w_s
                                 + kk + ((lane >> 4) ? 8u : 0u);
                pd_ldm_x4(af, (const unsigned char*)ap);
                #pragma unroll
                for (uint32_t sub = 0; sub < SLICE / 8u; ++sub) {
                    uint32_t bfr[2];
                    const __half* bp = vbuf + (size_t)(kk + (lane & 15u)) * row_e
                                     + n_base_w + sub * 8u;
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                                 : "=r"(bfr[0]), "=r"(bfr[1])
                                 : "r"((unsigned)__cvta_generic_to_shared(bp)));
                    pd_fa_mma16(o_acc[sub], af[0], af[1], af[2], af[3], bfr[0], bfr[1]);
                }
            }
        }
        if (d < G) {
            float ws = 0.0f;
            for (uint32_t p = 0; p < n_t; ++p) ws += s_w[d * t_s + p];
            s_l[d] = s_l[d] * s_corr[d] + ws;
            s_m[d] = s_mnew[d];
        }
        __syncthreads();
    }
    __syncthreads();
    // partial store straight from o frags in the production combine layout
    if (warp < NW_V) {
        const uint32_t n_base_w = warp * SLICE;
        #pragma unroll
        for (uint32_t sub = 0; sub < SLICE / 8u; ++sub) {
            #pragma unroll
            for (uint32_t half = 0; half < 2u; ++half) {
                const uint32_t rr = (lane >> 2) + half * 8u;
                if (rr >= G) continue;
                const size_t pidx =
                    ((size_t)(kvh * G + rr) * gridDim.y + b) * n_splits + sp;
                float* dst = out_o + pidx * HD + n_base_w + sub * 8u
                           + 2u * (lane & 3u);
                dst[0] = o_acc[sub][half * 2u];
                dst[1] = o_acc[sub][half * 2u + 1u];
            }
        }
    }
    if (d < G) {
        const size_t pidx = ((size_t)(kvh * G + d) * gridDim.y + b) * n_splits + sp;
        out_ml[pidx * 2 + 0] = s_m[d];
        out_ml[pidx * 2 + 1] = s_l[d];
    }
#else
    (void)q; (void)pool_k; (void)pool_v; (void)out_o; (void)out_ml;
    (void)positions; (void)slots; (void)block_tables; (void)blocks_per_slot;
    (void)kv_dim; (void)swa_window; (void)n_splits; (void)scale;
#endif
}

// Dense-decode v5: v3 with the max/weights phases folded into the
// score warps - scaled scores never leave registers. Each score warp
// quad-shuffles its 8-col partial row max, one barrier publishes the
// partials, every score warp redundantly folds m_new/corr (idempotent:
// warp 0's late s_m write already includes all partials, so a racing read
// of the new value folds to the same max), computes w = exp in regs, and
// stores the s_wh halves directly plus partial l sums. 3 barriers per tile
// instead of 4 and the s_sc/s_w roundtrips are gone: c32 SWA shape
// 137 -> 116 us, GLB 187 -> 152 us, bit-exact vs the
// f64 oracle. Same fp16-q/weights f32-accumulate numerics class as v3.
template <uint32_t HD, uint32_t G, uint32_t TILE>
__global__ void pd_attn_decode_v5_kernel(
    const float* __restrict__ q, const __half* __restrict__ pool_k,
    const __half* __restrict__ pool_v, float* __restrict__ out_o,
    float* __restrict__ out_ml, const unsigned int* __restrict__ positions,
    const unsigned int* __restrict__ slots,
    const uint32_t* __restrict__ block_tables, uint32_t blocks_per_slot,
    uint32_t kv_dim, uint32_t swa_window, uint32_t n_splits, float scale) {
#if PD_FA_OK
    const uint32_t kvh = blockIdx.x, b = blockIdx.y, sp = blockIdx.z;
    const uint32_t slot = slots ? slots[b] : b;
    const uint32_t d = threadIdx.x, nth = blockDim.x;
    const uint32_t warp = d >> 5, lane = d & 31u;
    const uint32_t n_heads = G * gridDim.x;

    const uint32_t pos = positions[b];
    const uint32_t first_pos =
        (swa_window > 0 && pos + 1 > swa_window) ? (pos + 1 - swa_window) : 0;
    const uint32_t n_pos = pos + 1 - first_pos;
    const uint32_t chunk = (n_pos + n_splits - 1) / n_splits;
    const uint32_t lo = sp * chunk;
    uint32_t hi = lo + chunk;
    if (hi > n_pos) hi = n_pos;

    extern __shared__ __align__(16) unsigned char v3_smraw[];
    constexpr uint32_t row_e = HD + 8u;        // +8-half pad (0ebac09: 2.4x)
    constexpr uint32_t w_s = TILE + 8u;        // f16 weight-strip stride
    constexpr uint32_t NSW = TILE / 8u;        // score warps
    __half* s_qh = (__half*)v3_smraw;                   // [16][row_e] f16 q
    __half* s_wh = s_qh + (size_t)16u * row_e;          // [16][w_s] f16 wts
    __half* s_kv = s_wh + (size_t)16u * w_s;            // [2][K,V][TILE][row_e]
    __shared__ float s_m[G], s_l[G], s_corr[G];
    __shared__ float s_pmax[NSW][G], s_psum[NSW][G];

    const float* qb = q + (size_t)b * n_heads * HD;
    const uint32_t* bt = block_tables + (size_t)slot * blocks_per_slot;

    // f16 q tile: rows < G real, rows G..15 zero (mma pad rows)
    for (uint32_t i = d; i < 16u * row_e; i += nth) {
        const uint32_t r = i / row_e, c = i % row_e;
        float v = (r < G && c < HD)
            ? qb[((size_t)kvh * G + r) * HD + c] : 0.0f;
        s_qh[(size_t)r * row_e + c] = __float2half(v);
    }
    for (uint32_t i = d; i < 16u * w_s; i += nth) s_wh[i] = __half(0.f);
    if (d < G) { s_m[d] = -INFINITY; s_l[d] = 0.0f; }

    constexpr uint32_t NW_V = (HD / 32u < 8u) ? HD / 32u : 8u;  // V warps
    constexpr uint32_t SLICE = HD / NW_V;                        // dims/warp
    float o_acc[SLICE / 8u][4];
    #pragma unroll
    for (uint32_t i = 0; i < SLICE / 8u; ++i)
        #pragma unroll
        for (uint32_t j = 0; j < 4u; ++j) o_acc[i][j] = 0.0f;

    constexpr uint32_t lines = (HD * 2u) >> 4;
    auto stage = [&](uint32_t bf, uint32_t t0) {
        const uint32_t n_t = hi - t0 < TILE ? hi - t0 : TILE;
        // partial tiles: the mma passes read all 16 rows of the K/V tiles
        // (zero-weight folds only null a stale row if its bytes are finite -
        // 0 x Inf/NaN poisons o). Zero the tail rows before staging.
        if (n_t < TILE) {
            for (uint32_t i = d; i < 2u * (TILE - n_t) * lines; i += nth) {
                const uint32_t kvsel = i / ((TILE - n_t) * lines);
                const uint32_t j = i - kvsel * (TILE - n_t) * lines;
                const uint32_t p = n_t + j / lines, l = j % lines;
                *(uint4*)((char*)(s_kv
                    + ((size_t)(bf * 2u + kvsel) * TILE + p) * row_e) + l * 16u)
                    = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        const uint32_t rows = 2u * n_t;
        const uint32_t gsz = nth / rows ? nth / rows : 1u;
        const uint32_t r = d / gsz, lt = d % gsz;
        if (r < rows) {
            const uint32_t kvsel = r >= n_t ? 1u : 0u;
            const uint32_t p = kvsel ? r - n_t : r;
            const uint32_t gpos = first_pos + t0 + p;
            const uint32_t blk = bt[gpos >> 4];
            const char* src = (const char*)((kvsel ? pool_v : pool_k)
                + (size_t)blk * 16u * kv_dim + (size_t)(gpos & 15u) * kv_dim
                + (size_t)kvh * HD);
            char* dst = (char*)(s_kv + ((size_t)(bf * 2u + kvsel) * TILE + p) * row_e);
            #pragma unroll
            for (uint32_t l = lt; l < lines; l += gsz)
                pd_attn_cpa16(dst + l * 16u, src + l * 16u);
        }
        pd_attn_cpa_commit();
    };

    __syncthreads();
    if (lo < hi) stage(0u, lo);
    uint32_t bf = 0;
    for (uint32_t t0 = lo; t0 < hi; t0 += TILE, bf ^= 1u) {
        const uint32_t n_t = hi - t0 < TILE ? hi - t0 : TILE;
        const bool more = t0 + TILE < hi;
        if (more) stage(bf ^ 1u, t0 + TILE);
        if (more) pd_attn_cpa_wait1(); else pd_attn_cpa_wait0();
        __syncthreads();
        const __half* kbuf = s_kv + (size_t)(bf * 2u) * TILE * row_e;
        const __half* vbuf = s_kv + ((size_t)(bf * 2u) + 1u) * TILE * row_e;
        // scores: warps 0..NSW-1, one n8 col-subtile each; scaled scores
        // stay in dfr. Frag rows rr = (lane>>2) + half*8, cols
        // pp = p0 + 2*(lane&3) + cc - per row, the 8 cols live in one
        // 4-lane quad group, so quad shfl reduces the partial max.
        float dfr[4] = {0.f, 0.f, 0.f, 0.f};
        if (warp < NSW) {
            const uint32_t p0 = warp * 8u;
            for (uint32_t kk = 0; kk < HD; kk += 16u) {
                uint32_t af[4];
                const __half* ap = s_qh + (size_t)(lane & 15u) * row_e
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
            // mask cols past n_t and scale in regs; rows >= G irrelevant
            #pragma unroll
            for (uint32_t half = 0; half < 2u; ++half)
                #pragma unroll
                for (uint32_t cc = 0; cc < 2u; ++cc) {
                    const uint32_t pp = p0 + 2u * (lane & 3u) + cc;
                    dfr[half * 2u + cc] = pp < n_t
                        ? dfr[half * 2u + cc] * scale : -INFINITY;
                }
            // quad-shfl partial row max (rows < G live in half 0 when G<=8,
            // half 1 rows are 8..15 - only relevant if G > 8, never here)
            float pm = fmaxf(dfr[0], dfr[1]);
            #pragma unroll
            for (uint32_t off = 1; off <= 2; off <<= 1)
                pm = fmaxf(pm, __shfl_xor_sync(0xffffffffu, pm, off));
            const uint32_t rr = lane >> 2;
            if ((lane & 3u) == 0 && rr < G) s_pmax[warp][rr] = pm;
        }
        __syncthreads();
        // fold + weights, still in the score warps (redundant per warp)
        if (warp < NSW) {
            const uint32_t p0 = warp * 8u;
            const uint32_t rr = lane >> 2;
            float mnew = 0.f, corr = 1.f, w0 = 0.f, w1 = 0.f;
            if (rr < G) {
                float m = s_m[rr];
                #pragma unroll
                for (uint32_t sw = 0; sw < NSW; ++sw)
                    m = fmaxf(m, s_pmax[sw][rr]);
                mnew = m;
                corr = __expf(s_m[rr] - m);
                w0 = dfr[0] > -INFINITY ? __expf(dfr[0] - m) : 0.f;
                w1 = dfr[1] > -INFINITY ? __expf(dfr[1] - m) : 0.f;
                const uint32_t pp = p0 + 2u * (lane & 3u);
                // two adjacent halves, one u32 store (pp even, 4B aligned)
                *(__half2*)(s_wh + (size_t)rr * w_s + pp) =
                    __floats2half2_rn(w0, w1);
            }
            // partial l over the quad's 8 cols
            float ps = w0 + w1;
            #pragma unroll
            for (uint32_t off = 1; off <= 2; off <<= 1)
                ps += __shfl_xor_sync(0xffffffffu, ps, off);
            if ((lane & 3u) == 0 && rr < G) {
                s_psum[warp][rr] = ps;
                if (warp == 0) s_corr[rr] = corr;
                if (warp == 0) s_m[rr] = mnew;   // safe: fold is idempotent
            }
        }
        __syncthreads();
        // o update: warp w < NW_V owns dims [w*SLICE, +SLICE)
        if (warp < NW_V) {
            const uint32_t n_base_w = warp * SLICE;
            #pragma unroll
            for (uint32_t half = 0; half < 2u; ++half) {
                const uint32_t rr = (lane >> 2) + half * 8u;
                const float corr = rr < G ? s_corr[rr] : 1.0f;
                #pragma unroll
                for (uint32_t sub = 0; sub < SLICE / 8u; ++sub) {
                    o_acc[sub][half * 2u] *= corr;
                    o_acc[sub][half * 2u + 1u] *= corr;
                }
            }
            #pragma unroll
            for (uint32_t kk = 0; kk < TILE; kk += 16u) {
                uint32_t af[4];
                const __half* ap = s_wh + (size_t)(lane & 15u) * w_s
                                 + kk + ((lane >> 4) ? 8u : 0u);
                pd_ldm_x4(af, (const unsigned char*)ap);
                #pragma unroll
                for (uint32_t sub = 0; sub < SLICE / 8u; ++sub) {
                    uint32_t bfr[2];
                    const __half* bp = vbuf + (size_t)(kk + (lane & 15u)) * row_e
                                     + n_base_w + sub * 8u;
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                                 : "=r"(bfr[0]), "=r"(bfr[1])
                                 : "r"((unsigned)__cvta_generic_to_shared(bp)));
                    pd_fa_mma16(o_acc[sub], af[0], af[1], af[2], af[3], bfr[0], bfr[1]);
                }
            }
        }
        if (d < G) {
            float ws = 0.0f;
            #pragma unroll
            for (uint32_t sw = 0; sw < NSW; ++sw) ws += s_psum[sw][d];
            s_l[d] = s_l[d] * s_corr[d] + ws;
        }
        __syncthreads();
    }
    __syncthreads();
    // partial store straight from o frags in the production combine layout
    if (warp < NW_V) {
        const uint32_t n_base_w = warp * SLICE;
        #pragma unroll
        for (uint32_t sub = 0; sub < SLICE / 8u; ++sub) {
            #pragma unroll
            for (uint32_t half = 0; half < 2u; ++half) {
                const uint32_t rr = (lane >> 2) + half * 8u;
                if (rr >= G) continue;
                const size_t pidx =
                    ((size_t)(kvh * G + rr) * gridDim.y + b) * n_splits + sp;
                float* dst = out_o + pidx * HD + n_base_w + sub * 8u
                           + 2u * (lane & 3u);
                dst[0] = o_acc[sub][half * 2u];
                dst[1] = o_acc[sub][half * 2u + 1u];
            }
        }
    }
    if (d < G) {
        const size_t pidx = ((size_t)(kvh * G + d) * gridDim.y + b) * n_splits + sp;
        out_ml[pidx * 2 + 0] = s_m[d];
        out_ml[pidx * 2 + 1] = s_l[d];
    }
#else
    (void)q; (void)pool_k; (void)pool_v; (void)out_o; (void)out_ml;
    (void)positions; (void)slots; (void)block_tables; (void)blocks_per_slot;
    (void)kv_dim; (void)swa_window; (void)n_splits; (void)scale;
#endif
}

// Dense-decode v7: v5 with the ~1024-instruction cp.async staging
// stream replaced by TMA. Tiles are BLOCK-ALIGNED - every tile is one
// 16-position paged block (contiguous pool rows), so staging is 2*SEGS
// cp.async.bulk.tensor.2d boxes of [16 rows x 128B] issued by warp 0.
// K/V smem tiles are SW128-canonical (the GEMM stage layout: one swizzle
// tile per 128B column segment keeps ldmatrix conflict-free - a seg-major
// image would degenerate the xor across rows); ldmatrix addresses xor per
// the swizzle. Window edges become per-tile [plo, phi) masks (-INF scores
// -> exact zero weights; edge rows hold real, finite pool bytes so zero
// weights null them exactly); V rows past the written pool tail are zeroed
// post-arrival (uninitialized bytes: 0 x NaN poisons the o mma). Ring
// depth 2 is deliberate: R3/R4 traded co-residency for depth and LOST
// (132/137 vs 113 us - the recurring occupancy-beats-depth lesson).
// Measured: SWA 116 -> 112.8 us, GLB 152 -> 131.6 us, bit-exact.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#define PD_ATTN_TMA_OK 1
#else
#define PD_ATTN_TMA_OK 0
#endif
template <uint32_t HD, uint32_t G>
__global__ void pd_attn_decode_v7_kernel(
    const __grid_constant__ PdTmap tmk,
    const __grid_constant__ PdTmap tmv,
    const float* __restrict__ q, float* __restrict__ out_o,
    float* __restrict__ out_ml, const unsigned int* __restrict__ positions,
    const unsigned int* __restrict__ slots,
    const uint32_t* __restrict__ block_tables, uint32_t blocks_per_slot,
    uint32_t kv_dim, uint32_t swa_window, uint32_t n_splits, float scale) {
#if PD_ATTN_TMA_OK
    PD_PDL_ARM();
    constexpr uint32_t TILE = 16u;
    const uint32_t kvh = blockIdx.x, b = blockIdx.y, sp = blockIdx.z;
    const uint32_t slot = slots ? slots[b] : b;
    const uint32_t d = threadIdx.x, nth = blockDim.x;
    const uint32_t warp = d >> 5, lane = d & 31u;
    const uint32_t n_heads = G * gridDim.x;

    const uint32_t pos = positions[b];
    const uint32_t first_pos =
        (swa_window > 0 && pos + 1 > swa_window) ? (pos + 1 - swa_window) : 0;
    const uint32_t n_pos = pos + 1 - first_pos;
    const uint32_t chunk = (n_pos + n_splits - 1) / n_splits;
    const uint32_t lo = sp * chunk;
    uint32_t hi = lo + chunk;
    if (hi > n_pos) hi = n_pos;

    // block-aligned walk: absolute positions, tile k = paged block B0+k
    const uint32_t g_lo = first_pos + lo, g_hi = first_pos + hi;
    const uint32_t B0 = g_lo >> 4;
    const uint32_t ntiles = lo < hi ? ((g_hi + 15u) >> 4) - B0 : 0u;
    const uint32_t nw = pos + 1u;                 // written-pool bound

    extern __shared__ __align__(128) unsigned char v7_smraw[];
    constexpr uint32_t SEGS = (HD * 2u) / 128u;   // 128B col segments/row
    constexpr uint32_t KVB = SEGS * 2048u;        // one K or V tile set
    constexpr uint32_t row_e = HD + 8u;
    constexpr uint32_t w_s = TILE + 8u;
    constexpr uint32_t NSW = TILE / 8u;
    // SW128's xor phase rides the ABSOLUTE smem address ((addr>>7)&7), not
    // the box-local row - pad the tile base to 1KB so the canonical formula
    // holds (the launcher adds 1KB to the dynamic-smem request)
    unsigned char* s_kv = v7_smraw
        + ((1024u - ((uint32_t)__cvta_generic_to_shared(v7_smraw) & 1023u)) & 1023u);
    __half* s_qh = (__half*)(s_kv + 4u * KVB);
    __half* s_wh = s_qh + (size_t)16u * row_e;
    __shared__ float s_m[G], s_l[G], s_corr[G];
    __shared__ float s_pmax[NSW][G], s_psum[NSW][G];
    __shared__ __align__(8) uint64_t s_bar[2];

    if (d == 0) {
        const uint32_t m0 = (uint32_t)__cvta_generic_to_shared(&s_bar[0]);
        // count=1: lane 0's single expect_tx arrive + the tx bytes close it
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(m0));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(m0 + 8u));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    const float* qb = q + (size_t)b * n_heads * HD;
    const uint32_t* bt = block_tables + (size_t)slot * blocks_per_slot;

    for (uint32_t i = d; i < 16u * row_e; i += nth) {
        const uint32_t r = i / row_e, c = i % row_e;
        float v = (r < G && c < HD)
            ? qb[((size_t)kvh * G + r) * HD + c] : 0.0f;
        s_qh[(size_t)r * row_e + c] = __float2half(v);
    }
    for (uint32_t i = d; i < 16u * w_s; i += nth) s_wh[i] = __half(0.f);
    if (d < G) { s_m[d] = -INFINITY; s_l[d] = 0.0f; }

    constexpr uint32_t NW_V = (HD / 32u < 8u) ? HD / 32u : 8u;
    constexpr uint32_t SLICE = HD / NW_V;
    float o_acc[SLICE / 8u][4];
    #pragma unroll
    for (uint32_t i = 0; i < SLICE / 8u; ++i)
        #pragma unroll
        for (uint32_t j = 0; j < 4u; ++j) o_acc[i][j] = 0.0f;

    // SW128 within a [16 x 128B] tile: byte off of (row r, 16B chunk c)
    auto sw = [](uint32_t r, uint32_t c) -> uint32_t {
        return ((r >> 3) << 10) + ((r & 7u) << 7) + ((c ^ (r & 7u)) << 4);
    };
    auto stage_t = [&](uint32_t bf, uint32_t k) {
        if (warp == 0) {
            const uint32_t m = (uint32_t)__cvta_generic_to_shared(&s_bar[bf]);
            if (lane == 0)
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                             ::"r"(m), "r"(2u * KVB));
            __syncwarp();
            if (lane < 2u * SEGS) {
                const uint32_t blk = bt[B0 + k];
                const uint32_t kv = lane / SEGS, s = lane % SEGS;
                const int y = (int)(blk * 16u);
                const int x = (int)(kvh * HD * 2u + s * 128u);
                const uint32_t dst = (uint32_t)__cvta_generic_to_shared(
                    s_kv + (size_t)(bf * 2u + kv) * KVB + s * 2048u);
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3}], [%4];"
                    ::"r"(dst), "l"(kv ? &tmv : &tmk), "r"(x), "r"(y), "r"(m)
                    : "memory");
            }
        }
    };
    auto bar_wait = [&](uint32_t bf, uint32_t parity) {
        const uint32_t a = (uint32_t)__cvta_generic_to_shared(&s_bar[bf]);
        asm volatile("{\n\t.reg .pred p;\nW%=:\n\t"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                     "@!p bra W%=;\n\t}" ::"r"(a), "r"(parity));
    };

    __syncthreads();
    if (ntiles) stage_t(0u, 0u);
    uint32_t bf = 0;
    uint32_t par[2] = {0u, 0u};
    for (uint32_t k = 0; k < ntiles; ++k, bf ^= 1u) {
        const uint32_t gbase = (B0 + k) * 16u;
        const uint32_t plo = g_lo > gbase ? g_lo - gbase : 0u;
        const uint32_t phi = g_hi - gbase < 16u ? g_hi - gbase : 16u;
        const bool more = k + 1u < ntiles;
        if (more) stage_t(bf ^ 1u, k + 1u);
        bar_wait(bf, par[bf]);
        par[bf] ^= 1u;
        unsigned char* kb = s_kv + (size_t)(bf * 2u) * KVB;
        unsigned char* vb = s_kv + (size_t)(bf * 2u + 1u) * KVB;
        // V rows past the written pool tail hold uninitialized bytes -
        // zero them (0 x NaN = NaN in the o mma). Only the last block
        // globally can hit this.
        const uint32_t pval = nw > gbase ? (nw - gbase < 16u ? nw - gbase : 16u) : 0u;
        if (pval < 16u) {
            for (uint32_t i = d; i < (16u - pval) * 8u * SEGS; i += nth) {
                const uint32_t p = pval + i / (8u * SEGS);
                const uint32_t j = i % (8u * SEGS);
                const uint32_t T = j >> 3, c = j & 7u;
                *(uint4*)(vb + T * 2048u + sw(p, c)) = make_uint4(0u, 0u, 0u, 0u);
            }
            __syncthreads();
        }
        float dfr[4] = {0.f, 0.f, 0.f, 0.f};
        if (warp < NSW) {
            const uint32_t p0 = warp * 8u;
            const uint32_t r = p0 + (lane & 7u);
            for (uint32_t kk = 0; kk < HD; kk += 16u) {
                uint32_t af[4];
                const __half* ap = s_qh + (size_t)(lane & 15u) * row_e
                                 + kk + ((lane >> 4) ? 8u : 0u);
                pd_ldm_x4(af, (const unsigned char*)ap);
                uint32_t bfr[2];
                const uint32_t C = (kk >> 3) + ((lane >> 3) & 1u);
                const unsigned char* bp = kb + (C >> 3) * 2048u + sw(r, C & 7u);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                             : "=r"(bfr[0]), "=r"(bfr[1])
                             : "r"((unsigned)__cvta_generic_to_shared(bp)));
                pd_fa_mma16(dfr, af[0], af[1], af[2], af[3], bfr[0], bfr[1]);
            }
            #pragma unroll
            for (uint32_t half = 0; half < 2u; ++half)
                #pragma unroll
                for (uint32_t cc = 0; cc < 2u; ++cc) {
                    const uint32_t pp = p0 + 2u * (lane & 3u) + cc;
                    dfr[half * 2u + cc] = pp >= plo && pp < phi
                        ? dfr[half * 2u + cc] * scale : -INFINITY;
                }
            float pm = fmaxf(dfr[0], dfr[1]);
            #pragma unroll
            for (uint32_t off = 1; off <= 2; off <<= 1)
                pm = fmaxf(pm, __shfl_xor_sync(0xffffffffu, pm, off));
            const uint32_t rr = lane >> 2;
            if ((lane & 3u) == 0 && rr < G) s_pmax[warp][rr] = pm;
        }
        __syncthreads();
        if (warp < NSW) {
            const uint32_t p0 = warp * 8u;
            const uint32_t rr = lane >> 2;
            float mnew = 0.f, corr = 1.f, w0 = 0.f, w1 = 0.f;
            if (rr < G) {
                float m = s_m[rr];
                #pragma unroll
                for (uint32_t sw2 = 0; sw2 < NSW; ++sw2)
                    m = fmaxf(m, s_pmax[sw2][rr]);
                mnew = m;
                corr = __expf(s_m[rr] - m);
                w0 = dfr[0] > -INFINITY ? __expf(dfr[0] - m) : 0.f;
                w1 = dfr[1] > -INFINITY ? __expf(dfr[1] - m) : 0.f;
                const uint32_t pp = p0 + 2u * (lane & 3u);
                *(__half2*)(s_wh + (size_t)rr * w_s + pp) =
                    __floats2half2_rn(w0, w1);
            }
            float ps = w0 + w1;
            #pragma unroll
            for (uint32_t off = 1; off <= 2; off <<= 1)
                ps += __shfl_xor_sync(0xffffffffu, ps, off);
            if ((lane & 3u) == 0 && rr < G) {
                s_psum[warp][rr] = ps;
                if (warp == 0) s_corr[rr] = corr;
                if (warp == 0) s_m[rr] = mnew;   // safe: fold is idempotent
            }
        }
        __syncthreads();
        if (warp < NW_V) {
            const uint32_t n_base_w = warp * SLICE;
            #pragma unroll
            for (uint32_t half = 0; half < 2u; ++half) {
                const uint32_t rr = (lane >> 2) + half * 8u;
                const float corr = rr < G ? s_corr[rr] : 1.0f;
                #pragma unroll
                for (uint32_t sub = 0; sub < SLICE / 8u; ++sub) {
                    o_acc[sub][half * 2u] *= corr;
                    o_acc[sub][half * 2u + 1u] *= corr;
                }
            }
            const uint32_t r = lane & 15u;       // V position row (TILE=16)
            uint32_t af[4];
            const __half* ap = s_wh + (size_t)(lane & 15u) * w_s
                             + ((lane >> 4) ? 8u : 0u);
            pd_ldm_x4(af, (const unsigned char*)ap);
            #pragma unroll
            for (uint32_t sub = 0; sub < SLICE / 8u; ++sub) {
                uint32_t bfr[2];
                const uint32_t C = (n_base_w + sub * 8u) >> 3;
                const unsigned char* bp = vb + (C >> 3) * 2048u + sw(r, C & 7u);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                             : "=r"(bfr[0]), "=r"(bfr[1])
                             : "r"((unsigned)__cvta_generic_to_shared(bp)));
                pd_fa_mma16(o_acc[sub], af[0], af[1], af[2], af[3], bfr[0], bfr[1]);
            }
        }
        if (d < G) {
            float ws = 0.0f;
            #pragma unroll
            for (uint32_t sw2 = 0; sw2 < NSW; ++sw2) ws += s_psum[sw2][d];
            s_l[d] = s_l[d] * s_corr[d] + ws;
        }
        __syncthreads();
    }
    __syncthreads();
    if (warp < NW_V) {
        const uint32_t n_base_w = warp * SLICE;
        #pragma unroll
        for (uint32_t sub = 0; sub < SLICE / 8u; ++sub) {
            #pragma unroll
            for (uint32_t half = 0; half < 2u; ++half) {
                const uint32_t rr = (lane >> 2) + half * 8u;
                if (rr >= G) continue;
                const size_t pidx =
                    ((size_t)(kvh * G + rr) * gridDim.y + b) * n_splits + sp;
                float* dst = out_o + pidx * HD + n_base_w + sub * 8u
                           + 2u * (lane & 3u);
                dst[0] = o_acc[sub][half * 2u];
                dst[1] = o_acc[sub][half * 2u + 1u];
            }
        }
    }
    if (d < G) {
        const size_t pidx = ((size_t)(kvh * G + d) * gridDim.y + b) * n_splits + sp;
        out_ml[pidx * 2 + 0] = s_m[d];
        out_ml[pidx * 2 + 1] = s_l[d];
    }
#else
    (void)tmk; (void)tmv; (void)q; (void)out_o; (void)out_ml;
    (void)positions; (void)slots; (void)block_tables; (void)blocks_per_slot;
    (void)kv_dim; (void)swa_window; (void)n_splits; (void)scale;
#endif
}


template <uint32_t HD, uint32_t G>
__global__ void pd_attn_decode_v7ks_kernel(
    const __grid_constant__ PdTmap tmk,
    const __grid_constant__ PdTmap tmv,
    const float* __restrict__ q, float* __restrict__ out_o,
    float* __restrict__ out_ml, const unsigned int* __restrict__ positions,
    const unsigned int* __restrict__ slots,
    const uint32_t* __restrict__ block_tables, uint32_t blocks_per_slot,
    uint32_t kv_dim, uint32_t swa_window, uint32_t n_splits, float scale) {
#if PD_ATTN_TMA_OK
    PD_PDL_ARM();
    constexpr uint32_t TILE = 16u;
    const uint32_t kvh = blockIdx.x, b = blockIdx.y, sp = blockIdx.z;
    const uint32_t slot = slots ? slots[b] : b;
    const uint32_t d = threadIdx.x, nth = blockDim.x;
    const uint32_t warp = d >> 5, lane = d & 31u;
    const uint32_t n_heads = G * gridDim.x;

    const uint32_t pos = positions[b];
    const uint32_t first_pos =
        (swa_window > 0 && pos + 1 > swa_window) ? (pos + 1 - swa_window) : 0;
    const uint32_t n_pos = pos + 1 - first_pos;
    const uint32_t chunk = (n_pos + n_splits - 1) / n_splits;
    const uint32_t lo = sp * chunk;
    uint32_t hi = lo + chunk;
    if (hi > n_pos) hi = n_pos;

    // block-aligned walk: absolute positions, tile k = paged block B0+k
    const uint32_t g_lo = first_pos + lo, g_hi = first_pos + hi;
    const uint32_t B0 = g_lo >> 4;
    const uint32_t ntiles = lo < hi ? ((g_hi + 15u) >> 4) - B0 : 0u;
    const uint32_t nw = pos + 1u;                 // written-pool bound

    extern __shared__ __align__(128) unsigned char v7ks_p_smraw[];
    constexpr uint32_t SEGS = (HD * 2u) / 128u;   // 128B col segments/row
    constexpr uint32_t KVB = SEGS * 2048u;        // one K or V tile set
    constexpr uint32_t row_e = HD + 8u;
    constexpr uint32_t w_s = TILE + 8u;
    constexpr uint32_t NSW = TILE / 8u;
    // SW128's xor phase rides the ABSOLUTE smem address ((addr>>7)&7), not
    // the box-local row - pad the tile base to 1KB so the canonical formula
    // holds (the launcher adds 1KB to the dynamic-smem request)
    unsigned char* s_kv = v7ks_p_smraw
        + ((1024u - ((uint32_t)__cvta_generic_to_shared(v7ks_p_smraw) & 1023u)) & 1023u);
    __half* s_qh = (__half*)(s_kv + 4u * KVB);
    __half* s_wh = s_qh + (size_t)16u * row_e;
    __shared__ float s_m[G], s_l[G], s_corr[G];
    __shared__ float s_pmax[NSW][G], s_psum[NSW][G];
    __shared__ float s_sc[4][NSW][128];   // K-slice partial score fragments
    __shared__ __align__(8) uint64_t s_bar[2];

    if (d == 0) {
        const uint32_t m0 = (uint32_t)__cvta_generic_to_shared(&s_bar[0]);
        // count=1: lane 0's single expect_tx arrive + the tx bytes close it
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(m0));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(m0 + 8u));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    const float* qb = q + (size_t)b * n_heads * HD;
    const uint32_t* bt = block_tables + (size_t)slot * blocks_per_slot;

    for (uint32_t i = d; i < 16u * row_e; i += nth) {
        const uint32_t r = i / row_e, c = i % row_e;
        float v = (r < G && c < HD)
            ? qb[((size_t)kvh * G + r) * HD + c] : 0.0f;
        s_qh[(size_t)r * row_e + c] = __float2half(v);
    }
    for (uint32_t i = d; i < 16u * w_s; i += nth) s_wh[i] = __half(0.f);
    if (d < G) { s_m[d] = -INFINITY; s_l[d] = 0.0f; }

    constexpr uint32_t NW_V = (HD / 32u < 8u) ? HD / 32u : 8u;
    constexpr uint32_t SLICE = HD / NW_V;
    float o_acc[SLICE / 8u][4];
    #pragma unroll
    for (uint32_t i = 0; i < SLICE / 8u; ++i)
        #pragma unroll
        for (uint32_t j = 0; j < 4u; ++j) o_acc[i][j] = 0.0f;

    // SW128 within a [16 x 128B] tile: byte off of (row r, 16B chunk c)
    auto sw = [](uint32_t r, uint32_t c) -> uint32_t {
        return ((r >> 3) << 10) + ((r & 7u) << 7) + ((c ^ (r & 7u)) << 4);
    };
    auto stage_t = [&](uint32_t bf, uint32_t k) {
        if (warp == 0) {
            const uint32_t m = (uint32_t)__cvta_generic_to_shared(&s_bar[bf]);
            if (lane == 0)
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                             ::"r"(m), "r"(2u * KVB));
            __syncwarp();
            if (lane < 2u * SEGS) {
                const uint32_t blk = bt[B0 + k];
                const uint32_t kv = lane / SEGS, s = lane % SEGS;
                const int y = (int)(blk * 16u);
                const int x = (int)(kvh * HD * 2u + s * 128u);
                const uint32_t dst = (uint32_t)__cvta_generic_to_shared(
                    s_kv + (size_t)(bf * 2u + kv) * KVB + s * 2048u);
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3}], [%4];"
                    ::"r"(dst), "l"(kv ? &tmv : &tmk), "r"(x), "r"(y), "r"(m)
                    : "memory");
            }
        }
    };
    auto bar_wait = [&](uint32_t bf, uint32_t parity) {
        const uint32_t a = (uint32_t)__cvta_generic_to_shared(&s_bar[bf]);
        asm volatile("{\n\t.reg .pred p;\nW%=:\n\t"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                     "@!p bra W%=;\n\t}" ::"r"(a), "r"(parity));
    };

    __syncthreads();
    if (ntiles) stage_t(0u, 0u);
    uint32_t bf = 0;
    uint32_t par[2] = {0u, 0u};
    for (uint32_t k = 0; k < ntiles; ++k, bf ^= 1u) {
        const uint32_t gbase = (B0 + k) * 16u;
        const uint32_t plo = g_lo > gbase ? g_lo - gbase : 0u;
        const uint32_t phi = g_hi - gbase < 16u ? g_hi - gbase : 16u;
        const bool more = k + 1u < ntiles;
        if (more) stage_t(bf ^ 1u, k + 1u);
        bar_wait(bf, par[bf]);
        par[bf] ^= 1u;
        unsigned char* kb = s_kv + (size_t)(bf * 2u) * KVB;
        unsigned char* vb = s_kv + (size_t)(bf * 2u + 1u) * KVB;
        // V rows past the written pool tail hold uninitialized bytes -
        // zero them (0 x NaN = NaN in the o mma). Only the last block
        // globally can hit this.
        const uint32_t pval = nw > gbase ? (nw - gbase < 16u ? nw - gbase : 16u) : 0u;
        if (pval < 16u) {
            for (uint32_t i = d; i < (16u - pval) * 8u * SEGS; i += nth) {
                const uint32_t p = pval + i / (8u * SEGS);
                const uint32_t j = i % (8u * SEGS);
                const uint32_t T = j >> 3, c = j & 7u;
                *(uint4*)(vb + T * 2048u + sw(p, c)) = make_uint4(0u, 0u, 0u, 0u);
            }
            __syncthreads();
        }
        float dfr[4] = {0.f, 0.f, 0.f, 0.f};
        {
            // K-split score: warp w = keys (w & 1)*8, K-slice w >> 1
            // - at hd512 the 2-warp score ran 32 HMMA steps while 6 warps
            // idled; all 8 now compute 8-step slices, warps 0-1 sum them.
            // Cross-slice fp32 sum: numeric-class change (gate arbitrates).
            const uint32_t swp = warp & 1u;
            const uint32_t ksl = warp >> 1;
            const uint32_t p0 = swp * 8u;
            const uint32_t r = p0 + (lane & 7u);
            const uint32_t kk0 = ksl * (HD / 4u);
            for (uint32_t kk = kk0; kk < kk0 + HD / 4u; kk += 16u) {
                uint32_t af[4];
                const __half* ap = s_qh + (size_t)(lane & 15u) * row_e
                                 + kk + ((lane >> 4) ? 8u : 0u);
                pd_ldm_x4(af, (const unsigned char*)ap);
                uint32_t bfr[2];
                const uint32_t C = (kk >> 3) + ((lane >> 3) & 1u);
                const unsigned char* bp = kb + (C >> 3) * 2048u + sw(r, C & 7u);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                             : "=r"(bfr[0]), "=r"(bfr[1])
                             : "r"((unsigned)__cvta_generic_to_shared(bp)));
                pd_fa_mma16(dfr, af[0], af[1], af[2], af[3], bfr[0], bfr[1]);
            }
            if (ksl > 0u) {
                #pragma unroll
                for (uint32_t j = 0; j < 4u; ++j)
                    s_sc[ksl][swp][lane * 4u + j] = dfr[j];
            }
        }
        __syncthreads();
        if (warp < NSW) {
            #pragma unroll
            for (uint32_t sl = 1u; sl < 4u; ++sl)
                #pragma unroll
                for (uint32_t j = 0; j < 4u; ++j)
                    dfr[j] += s_sc[sl][warp][lane * 4u + j];
            const uint32_t p0 = warp * 8u;
            #pragma unroll
            for (uint32_t half = 0; half < 2u; ++half)
                #pragma unroll
                for (uint32_t cc = 0; cc < 2u; ++cc) {
                    const uint32_t pp = p0 + 2u * (lane & 3u) + cc;
                    dfr[half * 2u + cc] = pp >= plo && pp < phi
                        ? dfr[half * 2u + cc] * scale : -INFINITY;
                }
            float pm = fmaxf(dfr[0], dfr[1]);
            #pragma unroll
            for (uint32_t off = 1; off <= 2; off <<= 1)
                pm = fmaxf(pm, __shfl_xor_sync(0xffffffffu, pm, off));
            const uint32_t rr = lane >> 2;
            if ((lane & 3u) == 0 && rr < G) s_pmax[warp][rr] = pm;
        }
        __syncthreads();
        if (warp < NSW) {
            const uint32_t p0 = warp * 8u;
            const uint32_t rr = lane >> 2;
            float mnew = 0.f, corr = 1.f, w0 = 0.f, w1 = 0.f;
            if (rr < G) {
                float m = s_m[rr];
                #pragma unroll
                for (uint32_t sw2 = 0; sw2 < NSW; ++sw2)
                    m = fmaxf(m, s_pmax[sw2][rr]);
                mnew = m;
                corr = __expf(s_m[rr] - m);
                w0 = dfr[0] > -INFINITY ? __expf(dfr[0] - m) : 0.f;
                w1 = dfr[1] > -INFINITY ? __expf(dfr[1] - m) : 0.f;
                const uint32_t pp = p0 + 2u * (lane & 3u);
                *(__half2*)(s_wh + (size_t)rr * w_s + pp) =
                    __floats2half2_rn(w0, w1);
            }
            float ps = w0 + w1;
            #pragma unroll
            for (uint32_t off = 1; off <= 2; off <<= 1)
                ps += __shfl_xor_sync(0xffffffffu, ps, off);
            if ((lane & 3u) == 0 && rr < G) {
                s_psum[warp][rr] = ps;
                if (warp == 0) s_corr[rr] = corr;
                if (warp == 0) s_m[rr] = mnew;   // safe: fold is idempotent
            }
        }
        __syncthreads();
        if (warp < NW_V) {
            const uint32_t n_base_w = warp * SLICE;
            #pragma unroll
            for (uint32_t half = 0; half < 2u; ++half) {
                const uint32_t rr = (lane >> 2) + half * 8u;
                const float corr = rr < G ? s_corr[rr] : 1.0f;
                #pragma unroll
                for (uint32_t sub = 0; sub < SLICE / 8u; ++sub) {
                    o_acc[sub][half * 2u] *= corr;
                    o_acc[sub][half * 2u + 1u] *= corr;
                }
            }
            const uint32_t r = lane & 15u;       // V position row (TILE=16)
            uint32_t af[4];
            const __half* ap = s_wh + (size_t)(lane & 15u) * w_s
                             + ((lane >> 4) ? 8u : 0u);
            pd_ldm_x4(af, (const unsigned char*)ap);
            #pragma unroll
            for (uint32_t sub = 0; sub < SLICE / 8u; ++sub) {
                uint32_t bfr[2];
                const uint32_t C = (n_base_w + sub * 8u) >> 3;
                const unsigned char* bp = vb + (C >> 3) * 2048u + sw(r, C & 7u);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                             : "=r"(bfr[0]), "=r"(bfr[1])
                             : "r"((unsigned)__cvta_generic_to_shared(bp)));
                pd_fa_mma16(o_acc[sub], af[0], af[1], af[2], af[3], bfr[0], bfr[1]);
            }
        }
        if (d < G) {
            float ws = 0.0f;
            #pragma unroll
            for (uint32_t sw2 = 0; sw2 < NSW; ++sw2) ws += s_psum[sw2][d];
            s_l[d] = s_l[d] * s_corr[d] + ws;
        }
        __syncthreads();
    }
    __syncthreads();
    if (warp < NW_V) {
        const uint32_t n_base_w = warp * SLICE;
        #pragma unroll
        for (uint32_t sub = 0; sub < SLICE / 8u; ++sub) {
            #pragma unroll
            for (uint32_t half = 0; half < 2u; ++half) {
                const uint32_t rr = (lane >> 2) + half * 8u;
                if (rr >= G) continue;
                const size_t pidx =
                    ((size_t)(kvh * G + rr) * gridDim.y + b) * n_splits + sp;
                float* dst = out_o + pidx * HD + n_base_w + sub * 8u
                           + 2u * (lane & 3u);
                dst[0] = o_acc[sub][half * 2u];
                dst[1] = o_acc[sub][half * 2u + 1u];
            }
        }
    }
    if (d < G) {
        const size_t pidx = ((size_t)(kvh * G + d) * gridDim.y + b) * n_splits + sp;
        out_ml[pidx * 2 + 0] = s_m[d];
        out_ml[pidx * 2 + 1] = s_l[d];
    }
#else
    (void)tmk; (void)tmv; (void)q; (void)out_o; (void)out_ml;
    (void)positions; (void)slots; (void)block_tables; (void)blocks_per_slot;
    (void)kv_dim; (void)swa_window; (void)n_splits; (void)scale;
#endif
}

// F8: KV pools hold e4m3 bytes (KV8). TMA stages half the bytes swizzle-free
// into the buffer TAIL; the consumer warps register-stage the raw strip,
// barrier, and expand fp8->f16 into the exact 128B-swizzled layout the
// unchanged score/V pipeline expects. Numerics = the fp8-storage class; the
// pipeline, barriers, and mma paths are bit-identical to the f16 form.
template <uint32_t HD, uint32_t G, bool F8 = false>
__global__ void pd_attn_decode_v8_kernel(
    const __grid_constant__ PdTmap tmk,
    const __grid_constant__ PdTmap tmv,
    const float* __restrict__ q, float* __restrict__ out_o,
    float* __restrict__ out_ml, const unsigned int* __restrict__ positions,
    const unsigned int* __restrict__ slots,
    const uint32_t* __restrict__ block_tables, uint32_t blocks_per_slot,
    uint32_t kv_dim, uint32_t swa_window, uint32_t n_splits, float scale) {
#if PD_ATTN_TMA_OK
    PD_PDL_ARM();
    constexpr uint32_t TILE = 16u;
    const uint32_t kvh = blockIdx.x, b = blockIdx.y, sp = blockIdx.z;
    const uint32_t d = threadIdx.x, nth = blockDim.x;
    const uint32_t warp = d >> 5, lane = d & 31u;
    const uint32_t n_heads = G * gridDim.x;

    const uint32_t pos = positions[b];
    const uint32_t first_pos =
        (swa_window > 0 && pos + 1 > swa_window) ? (pos + 1 - swa_window) : 0;
    const uint32_t n_pos = pos + 1 - first_pos;
    const uint32_t chunk = (n_pos + n_splits - 1) / n_splits;
    const uint32_t lo = sp * chunk;
    uint32_t hi = lo + chunk;
    if (hi > n_pos) hi = n_pos;
    const uint32_t g_lo = first_pos + lo, g_hi = first_pos + hi;
    const uint32_t B0 = g_lo >> 4;
    const uint32_t ntiles = lo < hi ? ((g_hi + 15u) >> 4) - B0 : 0u;
    const uint32_t nw = pos + 1u;

    extern __shared__ __align__(128) unsigned char pd_v8_smraw[];
    constexpr uint32_t SEGS = (HD * 2u) / 128u;
    constexpr uint32_t KVB = SEGS * 2048u;
    constexpr uint32_t row_e = HD + 8u;
    constexpr uint32_t w_s = TILE + 8u;
    constexpr uint32_t NSW = TILE / 8u;            // 2 score warps
    unsigned char* s_kv = pd_v8_smraw
        + ((1024u - ((uint32_t)__cvta_generic_to_shared(pd_v8_smraw) & 1023u)) & 1023u);
    unsigned char* s_k = s_kv;                     // 2 x KVB
    unsigned char* s_v = s_kv + 2u * KVB;          // 3 x KVB
    // F8: raw e4m3 lands at each buffer's TAIL (no extra smem - occupancy
    // is the binding constraint: +20KB of strips cost a block/SM and
    // measurably more time). The expander warps register-stage all raw bytes
    // before writing (barrier between), so the in-place doubling is safe.
    __half* s_qh = (__half*)(s_kv + 5u * KVB);
    __half* s_wh = s_qh + (size_t)16u * row_e;     // 2 slots x [16][w_s]
    __shared__ float s_m[G], s_l[G], s_corr[2][G];
    __shared__ float s_pmax[NSW][G], s_psum[NSW][G];
    __shared__ __align__(8) uint64_t s_bk[2], s_bv[3];
    // F8: expansion-done barriers the consumers wait instead of the TMA pair
    __shared__ __align__(8) uint64_t s_bek[2], s_bev[3];

    if (d == 0) {
        const uint32_t mk = (uint32_t)__cvta_generic_to_shared(&s_bk[0]);
        const uint32_t mv = (uint32_t)__cvta_generic_to_shared(&s_bv[0]);
        #pragma unroll
        for (uint32_t i = 0; i < 2u; ++i)
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(mk + i * 8u));
        #pragma unroll
        for (uint32_t i = 0; i < 3u; ++i)
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(mv + i * 8u));
        if (F8) {
            const uint32_t ek = (uint32_t)__cvta_generic_to_shared(&s_bek[0]);
            const uint32_t ev = (uint32_t)__cvta_generic_to_shared(&s_bev[0]);
            #pragma unroll
            for (uint32_t i = 0; i < 2u; ++i)
                asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(ek + i * 8u));
            #pragma unroll
            for (uint32_t i = 0; i < 3u; ++i)
                asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(ev + i * 8u));
        }
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    const float* qb = q + (size_t)b * n_heads * HD;
    const uint32_t slot = slots ? slots[b] : b;
    const uint32_t* bt = block_tables + (size_t)slot * blocks_per_slot;

    for (uint32_t i = d; i < 16u * row_e; i += nth) {
        const uint32_t r = i / row_e, c = i % row_e;
        float v = (r < G && c < HD)
            ? qb[((size_t)kvh * G + r) * HD + c] : 0.0f;
        s_qh[(size_t)r * row_e + c] = __float2half(v);
    }
    for (uint32_t i = d; i < 2u * 16u * w_s; i += nth) s_wh[i] = __half(0.f);
    if (d < G) { s_m[d] = -INFINITY; s_l[d] = 0.0f; }

    auto sw = [](uint32_t r, uint32_t c) -> uint32_t {
        return ((r >> 3) << 10) + ((r & 7u) << 7) + ((c ^ (r & 7u)) << 4);
    };
    // F8: half the TMA bytes, landed at the buffer TAIL (the expansion
    // in-place-doubles them forward; register staging makes that safe)
    constexpr uint32_t KVTX = F8 ? KVB / 2u : KVB;
    constexpr uint32_t RSEG = F8 ? SEGS / 2u : SEGS;
    constexpr uint32_t ROFF = F8 ? KVB / 2u : 0u;
    auto stage_k = [&](uint32_t bf, uint32_t k) {   // warp 0 only
        const uint32_t m = (uint32_t)__cvta_generic_to_shared(&s_bk[bf]);
        if (lane == 0)
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                         ::"r"(m), "r"(KVTX));
        __syncwarp();
        if (lane < RSEG) {
            const uint32_t blk = bt[B0 + k];
            const int y = (int)(blk * 16u);
            const int x = (int)(kvh * HD * (F8 ? 1u : 2u) + lane * 128u);
            const uint32_t dst = (uint32_t)__cvta_generic_to_shared(
                s_k + (size_t)bf * KVB + ROFF + lane * 2048u);
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                " [%0], [%1, {%2, %3}], [%4];"
                ::"r"(dst), "l"(&tmk), "r"(x), "r"(y), "r"(m) : "memory");
        }
    };
    auto stage_v = [&](uint32_t bf, uint32_t k) {   // warp 2 only
        const uint32_t m = (uint32_t)__cvta_generic_to_shared(&s_bv[bf]);
        if (lane == 0)
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                         ::"r"(m), "r"(KVTX));
        __syncwarp();
        if (lane < RSEG) {
            const uint32_t blk = bt[B0 + k];
            const int y = (int)(blk * 16u);
            const int x = (int)(kvh * HD * (F8 ? 1u : 2u) + lane * 128u);
            const uint32_t dst = (uint32_t)__cvta_generic_to_shared(
                s_v + (size_t)bf * KVB + ROFF + lane * 2048u);
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                " [%0], [%1, {%2, %3}], [%4];"
                ::"r"(dst), "l"(&tmv), "r"(x), "r"(y), "r"(m) : "memory");
        }
    };
    // fp8 -> f16 expansion (expander-warp, IN-PLACE): raw strip = the tail
    // half of the f16 buffer. All 64 pair-lanes register-stage their raw
    // chunks, bar, then write the doubled f16 - the barrier lives on the
    // expander warps, off the score/V critical paths.
    auto expand_f8 = [&](unsigned char* buf, uint32_t lid, uint32_t barid) {
        if constexpr (F8) {
            const unsigned char* raw = buf + KVB / 2u;
            constexpr uint32_t NCH = SEGS * 16u * 8u;
            uint2 st[(NCH + 63u) / 64u];
            uint32_t n = 0;
            for (uint32_t id = lid; id < NCH; id += 64u)
                st[n++] = *(const uint2*)(raw + ((id >> 7) >> 1) * 2048u
                                          + (((id >> 3) & 15u)) * 128u
                                          + ((id >> 7) & 1u) * 64u + (id & 7u) * 8u);
            asm volatile("bar.sync %0, 64;" ::"r"(barid));
            n = 0;
            for (uint32_t id = lid; id < NCH; id += 64u) {
                const uint32_t s = id >> 7, r = (id >> 3) & 15u, c = id & 7u;
                const uint2 v8 = st[n++];
                uint4 h4;
                #pragma unroll
                for (uint32_t hh = 0; hh < 2u; ++hh) {
                    const uint32_t w = hh == 0 ? v8.x : v8.y;
                    ((__half2*)&h4)[hh * 2u] = __half2(__nv_cvt_fp8x2_to_halfraw2(
                        (__nv_fp8x2_storage_t)(w & 0xffffu), __NV_E4M3));
                    ((__half2*)&h4)[hh * 2u + 1u] = __half2(__nv_cvt_fp8x2_to_halfraw2(
                        (__nv_fp8x2_storage_t)(w >> 16), __NV_E4M3));
                }
                *(uint4*)(buf + s * 2048u + sw(r, c)) = h4;
            }
            asm volatile("bar.sync %0, 64;" ::"r"(barid));
        } else {
            (void)buf; (void)lid; (void)barid;
        }
    };
    auto bar_wait = [&](uint64_t* bar, uint32_t parity) {
        const uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
        asm volatile("{\n\t.reg .pred p;\nW%=:\n\t"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                     "@!p bra W%=;\n\t}" ::"r"(a), "r"(parity));
    };

    __syncthreads();
    if (ntiles) {
        if (warp == 0) stage_k(0u, 0u);
        if (warp == 2) stage_v(0u, 0u);
    }
    if (ntiles > 1u) {
        if (warp == 0) stage_k(1u, 1u);
        if (warp == 2) stage_v(1u, 1u);
    }

    if (warp < 2u) {
        // ---------- score side ----------
        uint32_t pk[2] = {0u, 0u};
        float dfr[4];
        const uint32_t p0 = warp * 8u;
        const uint32_t rk = p0 + (lane & 7u);
        const uint32_t rr = lane >> 2;
        auto score_t = [&](uint32_t t) {
            const uint32_t bf = t & 1u;
            bar_wait(F8 ? &s_bek[bf] : &s_bk[bf], pk[bf]); pk[bf] ^= 1u;
            const uint32_t gbase = (B0 + t) * 16u;
            const uint32_t plo = g_lo > gbase ? g_lo - gbase : 0u;
            const uint32_t phi = g_hi - gbase < 16u ? g_hi - gbase : 16u;
            #pragma unroll
            for (uint32_t i = 0; i < 4u; ++i) dfr[i] = 0.f;
            unsigned char* kb = s_k + (size_t)bf * KVB;
            for (uint32_t kk = 0; kk < HD; kk += 16u) {
                uint32_t af[4];
                const __half* ap = s_qh + (size_t)(lane & 15u) * row_e
                                 + kk + ((lane >> 4) ? 8u : 0u);
                pd_ldm_x4(af, (const unsigned char*)ap);
                uint32_t bfr[2];
                const uint32_t C = (kk >> 3) + ((lane >> 3) & 1u);
                const unsigned char* bp = kb + (C >> 3) * 2048u + sw(rk, C & 7u);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                             : "=r"(bfr[0]), "=r"(bfr[1])
                             : "r"((unsigned)__cvta_generic_to_shared(bp)));
                pd_fa_mma16(dfr, af[0], af[1], af[2], af[3], bfr[0], bfr[1]);
            }
            #pragma unroll
            for (uint32_t half = 0; half < 2u; ++half)
                #pragma unroll
                for (uint32_t cc = 0; cc < 2u; ++cc) {
                    const uint32_t pp = p0 + 2u * (lane & 3u) + cc;
                    dfr[half * 2u + cc] = pp >= plo && pp < phi
                        ? dfr[half * 2u + cc] * scale : -INFINITY;
                }
            float pm = fmaxf(dfr[0], dfr[1]);
            #pragma unroll
            for (uint32_t off = 1; off <= 2; off <<= 1)
                pm = fmaxf(pm, __shfl_xor_sync(0xffffffffu, pm, off));
            if ((lane & 3u) == 0 && rr < G) s_pmax[warp][rr] = pm;
        };
        auto fold_t = [&](uint32_t t) {
            const uint32_t slot = t & 1u;
            float mnew = 0.f, corr = 1.f, w0 = 0.f, w1 = 0.f;
            if (rr < G) {
                float m = s_m[rr];
                #pragma unroll
                for (uint32_t sw2 = 0; sw2 < NSW; ++sw2)
                    m = fmaxf(m, s_pmax[sw2][rr]);
                mnew = m;
                corr = __expf(s_m[rr] - m);
                w0 = dfr[0] > -INFINITY ? __expf(dfr[0] - m) : 0.f;
                w1 = dfr[1] > -INFINITY ? __expf(dfr[1] - m) : 0.f;
                const uint32_t pp = p0 + 2u * (lane & 3u);
                *(__half2*)(s_wh + (size_t)slot * 16u * w_s
                            + (size_t)rr * w_s + pp) = __floats2half2_rn(w0, w1);
            }
            float ps = w0 + w1;
            #pragma unroll
            for (uint32_t off = 1; off <= 2; off <<= 1)
                ps += __shfl_xor_sync(0xffffffffu, ps, off);
            if ((lane & 3u) == 0 && rr < G) {
                s_psum[warp][rr] = ps;
                if (warp == 0) { s_corr[slot][rr] = corr; s_m[rr] = mnew; }
            }
            asm volatile("bar.sync 1, 64;");       // both warps' psum landed
            if (d < G) {
                float ws = 0.0f;
                #pragma unroll
                for (uint32_t sw2 = 0; sw2 < NSW; ++sw2) ws += s_psum[sw2][d];
                s_l[d] = s_l[d] * s_corr[slot][d] + ws;
            }
        };
        if (ntiles) {
            score_t(0u);
            asm volatile("bar.sync 1, 64;");
            fold_t(0u);
            asm volatile("bar.arrive 2, 256;");
        }
        for (uint32_t j = 0; j < ntiles; ++j) {
            const uint32_t t = j + 1u;
            if (t >= ntiles) break;                // tail: V side finishes alone
            if (warp == 0 && t + 1u < ntiles) stage_k((t + 1u) & 1u, t + 1u);
            score_t(t);
            asm volatile("bar.sync 1, 64;");       // pmax exchange
            asm volatile("bar.sync 3, 256;");      // w slot (t&1) freed by V(t-2)
            fold_t(t);
            asm volatile("bar.arrive 2, 256;");    // w(t) ready
        }
    } else if (F8 && warp >= 8u) {
        // ---------- expander warps (F8 only; blockDim 320) ----------
        // warp 8 = K, warp 9 = V: wait the TMA barrier, stream-convert the
        // raw strip into the f16 buffer, arrive the expansion barrier the
        // consumers wait. Tile t+1 expands while tile t is consumed - the
        // conversion leaves the score/V critical paths entirely.
        if (warp < 10u) {           // warps 8-9: K expander pair (64 lanes)
            uint32_t pk8[2] = {0u, 0u};
            const uint32_t lid = d - 256u;
            for (uint32_t t = 0; t < ntiles; ++t) {
                const uint32_t bf = t & 1u;
                bar_wait(&s_bk[bf], pk8[bf]); pk8[bf] ^= 1u;
                expand_f8(s_k + (size_t)bf * KVB, lid, 5u);
                if (lid == 0)
                    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
                                 ::"r"((uint32_t)__cvta_generic_to_shared(&s_bek[bf])));
            }
        } else {                     // warps 10-11: V expander pair (64 lanes)
            uint32_t pv8[3] = {0u, 0u, 0u};
            const uint32_t lid = d - 320u;
            for (uint32_t t = 0; t < ntiles; ++t) {
                const uint32_t bf = t % 3u;
                bar_wait(&s_bv[bf], pv8[bf]); pv8[bf] ^= 1u;
                expand_f8(s_v + (size_t)bf * KVB, lid, 6u);
                if (lid == 0)
                    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
                                 ::"r"((uint32_t)__cvta_generic_to_shared(&s_bev[bf])));
            }
        }
    } else {
        // ---------- V side ----------
        uint32_t pv[3] = {0u, 0u, 0u};
        constexpr uint32_t SLICE0 = (HD / 48u) * 8u;
        constexpr uint32_t LASTS = HD - 5u * SLICE0;
        constexpr uint32_t MAXSUB = LASTS / 8u;
        const uint32_t vw = warp - 2u;
        const uint32_t n_base_w = vw * SLICE0;
        const uint32_t nsub = vw == 5u ? MAXSUB : SLICE0 / 8u;
        float o_acc[MAXSUB][4];
        #pragma unroll
        for (uint32_t i = 0; i < MAXSUB; ++i)
            #pragma unroll
            for (uint32_t j2 = 0; j2 < 4u; ++j2) o_acc[i][j2] = 0.0f;
        for (uint32_t j = 0; j < ntiles; ++j) {
            asm volatile("bar.sync 2, 256;");      // w(j)/corr(j) ready
            if (warp == 2 && j + 2u < ntiles) stage_v((j + 2u) % 3u, j + 2u);
            asm volatile("bar.arrive 3, 256;");    // V(j-1) fully consumed
            const uint32_t slot = j & 1u;
            const uint32_t vbf = j % 3u;
            bar_wait(F8 ? &s_bev[vbf] : &s_bv[vbf], pv[vbf]); pv[vbf] ^= 1u;
            unsigned char* vb = s_v + (size_t)vbf * KVB;
            const uint32_t gbase = (B0 + j) * 16u;
            const uint32_t pval = nw > gbase
                ? (nw - gbase < 16u ? nw - gbase : 16u) : 0u;
            if (pval < 16u) {
                for (uint32_t i = d - 64u; i < (16u - pval) * 8u * SEGS; i += 192u) {
                    const uint32_t p = pval + i / (8u * SEGS);
                    const uint32_t jj = i % (8u * SEGS);
                    const uint32_t T = jj >> 3, c = jj & 7u;
                    *(uint4*)(vb + T * 2048u + sw(p, c)) = make_uint4(0u, 0u, 0u, 0u);
                }
                asm volatile("bar.sync 4, 192;");
            }
            #pragma unroll
            for (uint32_t half = 0; half < 2u; ++half) {
                const uint32_t rr = (lane >> 2) + half * 8u;
                const float corr = rr < G ? s_corr[slot][rr] : 1.0f;
                for (uint32_t sub = 0; sub < nsub; ++sub) {
                    o_acc[sub][half * 2u] *= corr;
                    o_acc[sub][half * 2u + 1u] *= corr;
                }
            }
            const uint32_t r = lane & 15u;
            uint32_t af[4];
            const __half* ap = s_wh + (size_t)slot * 16u * w_s
                             + (size_t)(lane & 15u) * w_s + ((lane >> 4) ? 8u : 0u);
            pd_ldm_x4(af, (const unsigned char*)ap);
            for (uint32_t sub = 0; sub < nsub; ++sub) {
                uint32_t bfr[2];
                const uint32_t C = (n_base_w + sub * 8u) >> 3;
                const unsigned char* bp = vb + (C >> 3) * 2048u + sw(r, C & 7u);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                             : "=r"(bfr[0]), "=r"(bfr[1])
                             : "r"((unsigned)__cvta_generic_to_shared(bp)));
                pd_fa_mma16(o_acc[sub], af[0], af[1], af[2], af[3], bfr[0], bfr[1]);
            }
        }
        // no rejoin needed: the last b2 sync ordered fold(ntiles-1)'s s_m/s_l
        if (n_splits == 1u) {
            // FIN: single-split ticks finalize IN-KERNEL - at one
            // split with -inf sinks the combine kernel's math is exactly
            // o / l (its exp weights are expf(0)=1), so the separate
            // 27us/layer combine pass (1.63 ms/tick at c32) disappears. Output in the COMBINED batch-major
            // layout the wo input expects. Bit-identical division.
            for (uint32_t sub = 0; sub < nsub; ++sub) {
                #pragma unroll
                for (uint32_t half = 0; half < 2u; ++half) {
                    const uint32_t rr = (lane >> 2) + half * 8u;
                    if (rr >= G) continue;
                    float* dst = out_o
                        + ((size_t)b * n_heads + kvh * G + rr) * HD
                        + n_base_w + sub * 8u + 2u * (lane & 3u);
                    dst[0] = o_acc[sub][half * 2u] / s_l[rr];
                    dst[1] = o_acc[sub][half * 2u + 1u] / s_l[rr];
                }
            }
            return;
        }
        for (uint32_t sub = 0; sub < nsub; ++sub) {
            #pragma unroll
            for (uint32_t half = 0; half < 2u; ++half) {
                const uint32_t rr = (lane >> 2) + half * 8u;
                if (rr >= G) continue;
                const size_t pidx =
                    ((size_t)(kvh * G + rr) * gridDim.y + b) * n_splits + sp;
                float* dst = out_o + pidx * HD + n_base_w + sub * 8u
                           + 2u * (lane & 3u);
                dst[0] = o_acc[sub][half * 2u];
                dst[1] = o_acc[sub][half * 2u + 1u];
            }
        }
        if (out_ml && d - 64u < G) {
            const uint32_t g = d - 64u;
            const size_t pidx = ((size_t)(kvh * G + g) * gridDim.y + b) * n_splits + sp;
            out_ml[pidx * 2u] = s_m[g];
            out_ml[pidx * 2u + 1u] = s_l[g];
        }
        return;
    }
#else
    (void)tmk; (void)tmv; (void)q; (void)out_o; (void)out_ml; (void)positions;
    (void)slots; (void)block_tables; (void)blocks_per_slot; (void)kv_dim; (void)swa_window;
    (void)n_splits; (void)scale;
#endif
}


template <uint32_t HD, uint32_t G>
__global__ void pd_attn_decode_v8ks_kernel(
    const __grid_constant__ PdTmap tmk,
    const __grid_constant__ PdTmap tmv,
    const float* __restrict__ q, float* __restrict__ out_o,
    float* __restrict__ out_ml, const unsigned int* __restrict__ positions,
    const unsigned int* __restrict__ slots,
    const uint32_t* __restrict__ block_tables, uint32_t blocks_per_slot,
    uint32_t kv_dim, uint32_t swa_window, uint32_t n_splits, float scale) {
#if PD_ATTN_TMA_OK
    PD_PDL_ARM();
    constexpr uint32_t TILE = 16u;
    const uint32_t kvh = blockIdx.x, b = blockIdx.y, sp = blockIdx.z;
    const uint32_t d = threadIdx.x, nth = blockDim.x;
    const uint32_t warp = d >> 5, lane = d & 31u;
    const uint32_t n_heads = G * gridDim.x;

    const uint32_t pos = positions[b];
    const uint32_t first_pos =
        (swa_window > 0 && pos + 1 > swa_window) ? (pos + 1 - swa_window) : 0;
    const uint32_t n_pos = pos + 1 - first_pos;
    const uint32_t chunk = (n_pos + n_splits - 1) / n_splits;
    const uint32_t lo = sp * chunk;
    uint32_t hi = lo + chunk;
    if (hi > n_pos) hi = n_pos;
    const uint32_t g_lo = first_pos + lo, g_hi = first_pos + hi;
    const uint32_t B0 = g_lo >> 4;
    const uint32_t ntiles = lo < hi ? ((g_hi + 15u) >> 4) - B0 : 0u;
    const uint32_t nw = pos + 1u;

    extern __shared__ __align__(128) unsigned char v8ks_p_smraw[];
    constexpr uint32_t SEGS = (HD * 2u) / 128u;
    constexpr uint32_t KVB = SEGS * 2048u;
    constexpr uint32_t row_e = HD + 8u;
    constexpr uint32_t w_s = TILE + 8u;
    constexpr uint32_t NSW = TILE / 8u;            // 2 score warps
    unsigned char* s_kv = v8ks_p_smraw
        + ((1024u - ((uint32_t)__cvta_generic_to_shared(v8ks_p_smraw) & 1023u)) & 1023u);
    unsigned char* s_k = s_kv;                     // 2 x KVB
    unsigned char* s_v = s_kv + 2u * KVB;          // 3 x KVB
    __half* s_qh = (__half*)(s_kv + 5u * KVB);
    __half* s_wh = s_qh + (size_t)16u * row_e;     // 2 slots x [16][w_s]
    __shared__ float s_m[G], s_l[G], s_corr[2][G];
    __shared__ float s_pmax[NSW][G], s_psum[NSW][G];
    __shared__ float s_sc1[2][128];   // K-slice-1 partial fragments
    __shared__ __align__(8) uint64_t s_bk[2], s_bv[3];

    if (d == 0) {
        const uint32_t mk = (uint32_t)__cvta_generic_to_shared(&s_bk[0]);
        const uint32_t mv = (uint32_t)__cvta_generic_to_shared(&s_bv[0]);
        #pragma unroll
        for (uint32_t i = 0; i < 2u; ++i)
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(mk + i * 8u));
        #pragma unroll
        for (uint32_t i = 0; i < 3u; ++i)
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(mv + i * 8u));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    const float* qb = q + (size_t)b * n_heads * HD;
    const uint32_t slot = slots ? slots[b] : b;
    const uint32_t* bt = block_tables + (size_t)slot * blocks_per_slot;

    for (uint32_t i = d; i < 16u * row_e; i += nth) {
        const uint32_t r = i / row_e, c = i % row_e;
        float v = (r < G && c < HD)
            ? qb[((size_t)kvh * G + r) * HD + c] : 0.0f;
        s_qh[(size_t)r * row_e + c] = __float2half(v);
    }
    for (uint32_t i = d; i < 2u * 16u * w_s; i += nth) s_wh[i] = __half(0.f);
    if (d < G) { s_m[d] = -INFINITY; s_l[d] = 0.0f; }

    auto sw = [](uint32_t r, uint32_t c) -> uint32_t {
        return ((r >> 3) << 10) + ((r & 7u) << 7) + ((c ^ (r & 7u)) << 4);
    };
    auto stage_k = [&](uint32_t bf, uint32_t k) {   // warp 0 only
        const uint32_t m = (uint32_t)__cvta_generic_to_shared(&s_bk[bf]);
        if (lane == 0)
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                         ::"r"(m), "r"(KVB));
        __syncwarp();
        if (lane < SEGS) {
            const uint32_t blk = bt[B0 + k];
            const int y = (int)(blk * 16u);
            const int x = (int)(kvh * HD * 2u + lane * 128u);
            const uint32_t dst = (uint32_t)__cvta_generic_to_shared(
                s_k + (size_t)bf * KVB + lane * 2048u);
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                " [%0], [%1, {%2, %3}], [%4];"
                ::"r"(dst), "l"(&tmk), "r"(x), "r"(y), "r"(m) : "memory");
        }
    };
    auto stage_v = [&](uint32_t bf, uint32_t k) {   // warp 4 only
        const uint32_t m = (uint32_t)__cvta_generic_to_shared(&s_bv[bf]);
        if (lane == 0)
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                         ::"r"(m), "r"(KVB));
        __syncwarp();
        if (lane < SEGS) {
            const uint32_t blk = bt[B0 + k];
            const int y = (int)(blk * 16u);
            const int x = (int)(kvh * HD * 2u + lane * 128u);
            const uint32_t dst = (uint32_t)__cvta_generic_to_shared(
                s_v + (size_t)bf * KVB + lane * 2048u);
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                " [%0], [%1, {%2, %3}], [%4];"
                ::"r"(dst), "l"(&tmv), "r"(x), "r"(y), "r"(m) : "memory");
        }
    };
    auto bar_wait = [&](uint64_t* bar, uint32_t parity) {
        const uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
        asm volatile("{\n\t.reg .pred p;\nW%=:\n\t"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                     "@!p bra W%=;\n\t}" ::"r"(a), "r"(parity));
    };

    __syncthreads();
    if (ntiles) {
        if (warp == 0) stage_k(0u, 0u);
        if (warp == 4) stage_v(0u, 0u);
    }
    if (ntiles > 1u) {
        if (warp == 0) stage_k(1u, 1u);
        if (warp == 4) stage_v(1u, 1u);
    }

    if (warp < 4u) {
        // ---------- score side (4 warps: keys (w&1)*8 x K-slice (w>>1)) ----------
        uint32_t pk[2] = {0u, 0u};
        float dfr[4];
        const uint32_t swp = warp & 1u;
        const uint32_t ksl = warp >> 1;
        const uint32_t p0 = swp * 8u;
        const uint32_t rk = p0 + (lane & 7u);
        const uint32_t rr = lane >> 2;
        auto score_t = [&](uint32_t t) {
            const uint32_t bf = t & 1u;
            bar_wait(&s_bk[bf], pk[bf]); pk[bf] ^= 1u;
            #pragma unroll
            for (uint32_t i = 0; i < 4u; ++i) dfr[i] = 0.f;
            unsigned char* kb = s_k + (size_t)bf * KVB;
            const uint32_t kk0 = ksl * (HD / 2u);
            for (uint32_t kk = kk0; kk < kk0 + HD / 2u; kk += 16u) {
                uint32_t af[4];
                const __half* ap = s_qh + (size_t)(lane & 15u) * row_e
                                 + kk + ((lane >> 4) ? 8u : 0u);
                pd_ldm_x4(af, (const unsigned char*)ap);
                uint32_t bfr[2];
                const uint32_t C = (kk >> 3) + ((lane >> 3) & 1u);
                const unsigned char* bp = kb + (C >> 3) * 2048u + sw(rk, C & 7u);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                             : "=r"(bfr[0]), "=r"(bfr[1])
                             : "r"((unsigned)__cvta_generic_to_shared(bp)));
                pd_fa_mma16(dfr, af[0], af[1], af[2], af[3], bfr[0], bfr[1]);
            }
            if (ksl == 1u) {
                #pragma unroll
                for (uint32_t i = 0; i < 4u; ++i)
                    s_sc1[swp][lane * 4u + i] = dfr[i];
            }
        };
        auto fold_t = [&](uint32_t t) {
            const uint32_t slot = t & 1u;
            if (warp < 2u) {
                #pragma unroll
                for (uint32_t i = 0; i < 4u; ++i) dfr[i] += s_sc1[warp][lane * 4u + i];
                const uint32_t gbase = (B0 + t) * 16u;
                const uint32_t plo = g_lo > gbase ? g_lo - gbase : 0u;
                const uint32_t phi = g_hi - gbase < 16u ? g_hi - gbase : 16u;
                #pragma unroll
                for (uint32_t half = 0; half < 2u; ++half)
                    #pragma unroll
                    for (uint32_t cc = 0; cc < 2u; ++cc) {
                        const uint32_t pp = p0 + 2u * (lane & 3u) + cc;
                        dfr[half * 2u + cc] = pp >= plo && pp < phi
                            ? dfr[half * 2u + cc] * scale : -INFINITY;
                    }
                float pm = fmaxf(dfr[0], dfr[1]);
                #pragma unroll
                for (uint32_t off = 1; off <= 2; off <<= 1)
                    pm = fmaxf(pm, __shfl_xor_sync(0xffffffffu, pm, off));
                if ((lane & 3u) == 0 && rr < G) s_pmax[warp][rr] = pm;
            }
            asm volatile("bar.sync 1, 128;");   // pmax landed
            float mnew = 0.f, corr = 1.f, w0 = 0.f, w1 = 0.f;
            if (warp < 2u && rr < G) {
                float m = s_m[rr];
                #pragma unroll
                for (uint32_t sw2 = 0; sw2 < NSW; ++sw2)
                    m = fmaxf(m, s_pmax[sw2][rr]);
                mnew = m;
                corr = __expf(s_m[rr] - m);
                w0 = dfr[0] > -INFINITY ? __expf(dfr[0] - m) : 0.f;
                w1 = dfr[1] > -INFINITY ? __expf(dfr[1] - m) : 0.f;
                const uint32_t pp = p0 + 2u * (lane & 3u);
                *(__half2*)(s_wh + (size_t)slot * 16u * w_s
                            + (size_t)rr * w_s + pp) = __floats2half2_rn(w0, w1);
            }
            float ps = w0 + w1;
            #pragma unroll
            for (uint32_t off = 1; off <= 2; off <<= 1)
                ps += __shfl_xor_sync(0xffffffffu, ps, off);
            if (warp < 2u && (lane & 3u) == 0 && rr < G) {
                s_psum[warp][rr] = ps;
                if (warp == 0) { s_corr[slot][rr] = corr; s_m[rr] = mnew; }
            }
            asm volatile("bar.sync 1, 128;");      // both warps' psum landed
            if (d < G) {
                float ws = 0.0f;
                #pragma unroll
                for (uint32_t sw2 = 0; sw2 < NSW; ++sw2) ws += s_psum[sw2][d];
                s_l[d] = s_l[d] * s_corr[slot][d] + ws;
            }
        };
        if (ntiles) {
            score_t(0u);
            asm volatile("bar.sync 1, 128;");
            fold_t(0u);
            asm volatile("bar.arrive 2, 256;");
        }
        for (uint32_t j = 0; j < ntiles; ++j) {
            const uint32_t t = j + 1u;
            if (t >= ntiles) break;                // tail: V side finishes alone
            if (warp == 0 && t + 1u < ntiles) stage_k((t + 1u) & 1u, t + 1u);
            score_t(t);
            asm volatile("bar.sync 1, 128;");      // partial exchange
            asm volatile("bar.sync 3, 256;");      // w slot (t&1) freed by V(t-2)
            fold_t(t);
            asm volatile("bar.arrive 2, 256;");    // w(t) ready
        }
    } else {
        // ---------- V side ----------
        uint32_t pv[3] = {0u, 0u, 0u};
        constexpr uint32_t SLICE0 = HD / 4u;     // 4 V warps, uniform slices
        constexpr uint32_t MAXSUB = SLICE0 / 8u;
        const uint32_t vw = warp - 4u;
        const uint32_t n_base_w = vw * SLICE0;
        const uint32_t nsub = MAXSUB;
        float o_acc[MAXSUB][4];
        #pragma unroll
        for (uint32_t i = 0; i < MAXSUB; ++i)
            #pragma unroll
            for (uint32_t j2 = 0; j2 < 4u; ++j2) o_acc[i][j2] = 0.0f;
        for (uint32_t j = 0; j < ntiles; ++j) {
            asm volatile("bar.sync 2, 256;");      // w(j)/corr(j) ready
            if (warp == 4 && j + 2u < ntiles) stage_v((j + 2u) % 3u, j + 2u);
            asm volatile("bar.arrive 3, 256;");    // V(j-1) fully consumed
            const uint32_t slot = j & 1u;
            const uint32_t vbf = j % 3u;
            bar_wait(&s_bv[vbf], pv[vbf]); pv[vbf] ^= 1u;
            unsigned char* vb = s_v + (size_t)vbf * KVB;
            const uint32_t gbase = (B0 + j) * 16u;
            const uint32_t pval = nw > gbase
                ? (nw - gbase < 16u ? nw - gbase : 16u) : 0u;
            if (pval < 16u) {
                for (uint32_t i = d - 128u; i < (16u - pval) * 8u * SEGS; i += 128u) {
                    const uint32_t p = pval + i / (8u * SEGS);
                    const uint32_t jj = i % (8u * SEGS);
                    const uint32_t T = jj >> 3, c = jj & 7u;
                    *(uint4*)(vb + T * 2048u + sw(p, c)) = make_uint4(0u, 0u, 0u, 0u);
                }
                asm volatile("bar.sync 4, 128;");
            }
            #pragma unroll
            for (uint32_t half = 0; half < 2u; ++half) {
                const uint32_t rr = (lane >> 2) + half * 8u;
                const float corr = rr < G ? s_corr[slot][rr] : 1.0f;
                for (uint32_t sub = 0; sub < nsub; ++sub) {
                    o_acc[sub][half * 2u] *= corr;
                    o_acc[sub][half * 2u + 1u] *= corr;
                }
            }
            const uint32_t r = lane & 15u;
            uint32_t af[4];
            const __half* ap = s_wh + (size_t)slot * 16u * w_s
                             + (size_t)(lane & 15u) * w_s + ((lane >> 4) ? 8u : 0u);
            pd_ldm_x4(af, (const unsigned char*)ap);
            for (uint32_t sub = 0; sub < nsub; ++sub) {
                uint32_t bfr[2];
                const uint32_t C = (n_base_w + sub * 8u) >> 3;
                const unsigned char* bp = vb + (C >> 3) * 2048u + sw(r, C & 7u);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                             : "=r"(bfr[0]), "=r"(bfr[1])
                             : "r"((unsigned)__cvta_generic_to_shared(bp)));
                pd_fa_mma16(o_acc[sub], af[0], af[1], af[2], af[3], bfr[0], bfr[1]);
            }
        }
        // no rejoin needed: the last b2 sync ordered fold(ntiles-1)'s s_m/s_l
        if (n_splits == 1u) {
            // FIN: see the v8 note - in-kernel o/l finalize in the
            // combined batch-major layout; the combine pass disappears.
            for (uint32_t sub = 0; sub < nsub; ++sub) {
                #pragma unroll
                for (uint32_t half = 0; half < 2u; ++half) {
                    const uint32_t rr = (lane >> 2) + half * 8u;
                    if (rr >= G) continue;
                    float* dst = out_o
                        + ((size_t)b * n_heads + kvh * G + rr) * HD
                        + n_base_w + sub * 8u + 2u * (lane & 3u);
                    dst[0] = o_acc[sub][half * 2u] / s_l[rr];
                    dst[1] = o_acc[sub][half * 2u + 1u] / s_l[rr];
                }
            }
            return;
        }
        for (uint32_t sub = 0; sub < nsub; ++sub) {
            #pragma unroll
            for (uint32_t half = 0; half < 2u; ++half) {
                const uint32_t rr = (lane >> 2) + half * 8u;
                if (rr >= G) continue;
                const size_t pidx =
                    ((size_t)(kvh * G + rr) * gridDim.y + b) * n_splits + sp;
                float* dst = out_o + pidx * HD + n_base_w + sub * 8u
                           + 2u * (lane & 3u);
                dst[0] = o_acc[sub][half * 2u];
                dst[1] = o_acc[sub][half * 2u + 1u];
            }
        }
        if (out_ml && d - 128u < G) {
            const uint32_t g = d - 128u;
            const size_t pidx = ((size_t)(kvh * G + g) * gridDim.y + b) * n_splits + sp;
            out_ml[pidx * 2u] = s_m[g];
            out_ml[pidx * 2u + 1u] = s_l[g];
        }
        return;
    }
#else
    (void)tmk; (void)tmv; (void)q; (void)out_o; (void)out_ml; (void)positions;
    (void)slots; (void)block_tables; (void)blocks_per_slot; (void)kv_dim; (void)swa_window;
    (void)n_splits; (void)scale;
#endif
}




