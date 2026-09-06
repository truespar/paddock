// gemm/dense_tc5.cuh - the tcgen05 (tensor-memory MMA) dense GEMM families.
// Textually-included segment of the single pack translation unit.
// Not standalone-compilable: include order is defined by ../pack.cu.
//
// Split out of gemm/dense_fp4_w8.cuh, which had reached 10962 lines against
// the repo's ~2500-line file ceiling. Cut at the file's own banner
// boundaries; every kernel here is sm_100a-gated behind PD_TC5_OK.
//
// Contains, in the original order: tc5 rowwise-e4m3, tc5bs block-scaled and
// its deferred-recycle / wide / pipelined-issuer / A-from-tmem variants, the
// decode-shaped block-scale K-split, v4 tc5p, the two-stage row-quant
// widening, tc5q, tc5t, tc5m, tc5r and tc5s.
//
// ORDER is LOAD-BEARING: this segment must be included after dense_fp4_w8.cuh
// (it uses that file's mma/tmap helpers) and before dense_f8_decode.cuh, which
// consumes six symbols defined here - pd_quantize_e4m3_row2_kernel,
// pd_rowmax_part_kernel, pd_rowq_chunks, pd_rowq_scr, pd_tc5p_fctr and
// pd_tc5q_ctr. PD_TC5_OK and the tc5 descriptor constructors come from
// ../tma_desc.cuh.
// ---- tcgen05 rowwise-e4m3 GEMM (sm_100a - the real Blackwell-DC pipe) -----
// The legacy warp-mma skeleton caps ~300 TF on B200 while a single issuing
// thread drives tcgen05 at the SM's full FP8 rate (~31 TF/SM, prototype
// verified bit-exact). This kernel: TMA bulk-tensor staging into
// SW128-canonical 128x128B tiles (the same byte arrangement the tma_kt
// family stages - pd_tmap_2d's SWIZZLE_128B), single-thread tcgen05.mma
// issue over 32B K-chunks accumulating in tensor memory, per-stage
// tcgen05.commit mbarriers to recycle smem under the TMA ring, and a
// tcgen05.ld epilogue applying the per-row scales (the rowwise class: no
// scale ever near the K loop). PD_TC5_OK and the sdesc/idesc constructors
// this family uses now live in ../tma_desc.cuh (included early).

//  probe side-channel: per-CTA timeline stamps. Bench-only TS=true
// instantiations of the tc5p kernels write 8 u64 per CTA here; production
// TS=false compiles every trace away. Defined outside the arch gate so the
// host-side cudaMemcpyToSymbol in the bench compiles on every pass.
__device__ unsigned long long* pd_tc5p_ts = nullptr;
__device__ __forceinline__ unsigned long long pd_ts_now() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
__device__ __forceinline__ uint32_t pd_ts_smid() {
    uint32_t s; asm("mov.u32 %0, %%smid;" : "=r"(s)); return s;
}

template <uint32_t S>
__global__ void __launch_bounds__(128) pd_f8row_gemm_tc5_kt(
    const __grid_constant__ PdTmap wmap, const __grid_constant__ PdTmap ymap,
    const float* __restrict__ wrs, const float* __restrict__ xrs,
    float* __restrict__ y, uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
    // grid.y = K-split z: decode-shaped grids (168 CTAs at gate/r<=128) fill
    // barely half the die at 2 CTAs/SM - z-planes + the format-blind ks
    // combine buy the missing waves (the mma_ks convention). z==0-only code
    // paths keep the single-plane prefill launch byte-identical.
#if PD_TC5_OK
    extern __shared__ __align__(1024) unsigned char pd_tc5_sh[];
    unsigned char* wt = pd_tc5_sh;                    // S x 16 KB
    unsigned char* yt = pd_tc5_sh + S * 16384u;      // S x 16 KB
    uint64_t* bfull = (uint64_t*)(pd_tc5_sh + 2u * S * 16384u);  // [S]
    uint64_t* bdone = bfull + S;                                  // [S]
    __shared__ uint32_t tmem_slot[1];

    const uint32_t tid = threadIdx.x;
    const uint32_t nk_all = (in_dim + 127u) / 128u;  // 128B K-slabs
    const uint32_t nz = gridDim.y;
    const uint32_t per = (nk_all + nz - 1u) / nz;
    const uint32_t k_lo = blockIdx.y * per;
    const uint32_t nk = (k_lo + per < nk_all ? per : (k_lo < nk_all ? nk_all - k_lo : 0u));
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t nct = batch_pad >> 7;
    const uint32_t row_base = (blockIdx.x / nct) * 128u;
    const uint32_t col_base = (blockIdx.x % nct) * 128u;
    if (nz > 1u) y += (size_t)blockIdx.y * out_dim * batch;

    if (tid == 0) {
        #pragma unroll
        for (uint32_t s = 0; s < S; ++s) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bfull[s])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bdone[s])));
        }
    }
    __syncthreads();
    if (tid < 32) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 128;"
                     ::"r"((uint32_t)__cvta_generic_to_shared(tmem_slot)));
    }
    __syncthreads();
    const uint32_t tmem = tmem_slot[0];

    if (tid == 0) {
        // TMA issue helper: expect both tiles' bytes on full[s], then the
        // two bulk loads (W rows at row_base, Y rows at col_base; K-slab kt)
        auto tma_stage = [&](uint32_t kt, uint32_t s) {
            const uint32_t m = (uint32_t)__cvta_generic_to_shared(&bfull[s]);
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 32768;" ::"r"(m));
            const uint32_t wd = (uint32_t)__cvta_generic_to_shared(wt + s * 16384u);
            const uint32_t yd = (uint32_t)__cvta_generic_to_shared(yt + s * 16384u);
            const int ck = (int)((k_lo + kt) * 128u);
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                " [%0], [%1, {%2, %3}], [%4];" ::"r"(wd), "l"(&wmap), "r"(ck),
                "r"((int)row_base), "r"(m) : "memory");
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                " [%0], [%1, {%2, %3}], [%4];" ::"r"(yd), "l"(&ymap), "r"(ck),
                "r"((int)col_base), "r"(m) : "memory");
        };
        auto bar_wait = [&](uint64_t* bar, uint32_t parity) {
            const uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
            asm volatile("{\n\t.reg .pred p;\n"
                         "W%=:\n\t"
                         "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                         "@!p bra W%=;\n\t}" ::"r"(a), "r"(parity));
        };
        #pragma unroll
        for (uint32_t s = 0; s < S; ++s)
            if (s < nk) tma_stage(s, s);
        const uint32_t idesc = pd_tc5_idesc();
        uint32_t fph = 0, dph = 0;  // parity bitsets, slot s at bit s
        for (uint32_t kt = 0; kt < nk; ++kt) {
            const uint32_t s = kt % S;
            bar_wait(&bfull[s], (fph >> s) & 1u); fph ^= 1u << s;
            const uint32_t w16 = (uint32_t)__cvta_generic_to_shared(wt + s * 16384u) >> 4;
            const uint32_t y16 = (uint32_t)__cvta_generic_to_shared(yt + s * 16384u) >> 4;
            #pragma unroll
            for (uint32_t kc = 0; kc < 4u; ++kc) {  // 4 x 32B chunks per slab
                const uint64_t ad = pd_tc5_sdesc(w16 + kc * 2u);
                const uint64_t bd = pd_tc5_sdesc(y16 + kc * 2u);
                const uint32_t en = (kt > 0 || kc > 0) ? 1u : 0u;
                asm volatile(
                    "{\n\t.reg .pred p;\n\t"
                    "setp.ne.b32 p, %4, 0;\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t}"
                    ::"r"(tmem), "l"(ad), "l"(bd), "r"(idesc), "r"(en));
            }
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                ::"r"((uint32_t)__cvta_generic_to_shared(&bdone[s])));
            const uint32_t pf = kt + S;
            if (pf < nk) {
                // recycle slot s: its just-committed MMAs must retire before
                // the TMA overwrite (commits complete in order, so this also
                // covers every earlier use of the slot)
                bar_wait(&bdone[s], (dph >> s) & 1u); dph ^= 1u << s;
                tma_stage(pf, s);
            }
        }
        // drain: wait the last commit (in-order completion covers the rest)
        if (nk > 0u) {
            const uint32_t ls = (nk - 1u) % S;
            bar_wait(&bdone[ls], (dph >> ls) & 1u);
        }
    }
    __syncthreads();

    // epilogue: warp w owns tmem rows 32w..32w+31; scales touch only here
    {
        const uint32_t warp = tid >> 5, lane = tid & 31u;
        const uint32_t row = row_base + warp * 32u + lane;
        const float w0 = row < out_dim ? wrs[row] : 0.0f;
        #pragma unroll
        for (uint32_t cc = 0; cc < 4u; ++cc) {
            uint32_t r[32];
            const uint32_t taddr = tmem + ((warp * 32u) << 16) + cc * 32u;
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
                "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
                "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
                : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
                  "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),"=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15]),
                  "=r"(r[16]),"=r"(r[17]),"=r"(r[18]),"=r"(r[19]),"=r"(r[20]),"=r"(r[21]),"=r"(r[22]),"=r"(r[23]),
                  "=r"(r[24]),"=r"(r[25]),"=r"(r[26]),"=r"(r[27]),"=r"(r[28]),"=r"(r[29]),"=r"(r[30]),"=r"(r[31])
                : "r"(taddr));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            if (row < out_dim) {
                #pragma unroll
                for (uint32_t j = 0; j < 32u; ++j) {
                    const uint32_t col = col_base + cc * 32u + j;
                    if (col < batch)
                        y[(size_t)col * out_dim + row] = nk == 0u
                            ? 0.0f
                            : __uint_as_float(r[j]) * w0 * xrs[col];
                }
            }
        }
    }
    __syncthreads();
    if (tid < 32) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 128;" ::"r"(tmem));
    }
#else
    (void)wmap; (void)ymap; (void)wrs; (void)xrs; (void)y;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}


// ---- tcgen05 BLOCK-SCALED e4m3 GEMM (sm_100a): the hardware ue8m0 fold ----
// The missing piece that made per-32 f8w8 lose on this die: the sw fold
// taxed every mma; tcgen05.mma kind::mxf8f6f4.block_scale applies both
// per-32 scales in hardware from tensor memory. SF tmem layout (PTX ISA /
// CUTLASS reference; probed exact): a K-slab's 4
// k-blocks of row scales pack as row m, block j -> lane m%32, column m/32,
// byte j, DUPLICATED to all four 32-lane partitions (each of the 4 warps
// tcgen05.st's its own partition); the 2-bit sf ids in the INSTRUCTION
// DESCRIPTOR (a: bits 29-31, b: 4-6) select the k-block byte - taddr high
// bits are ignored (probed). Structure otherwise = pd_f8row_gemm_tc5_kt:
// TMA SW128 tiles, S-deep ring, single-thread issue, tcgen05.ld epilogue
// (no scales there - the fold already happened).
#if PD_TC5_OK
// (pd_tc5_bs_idesc_bn / _bs_idesc / tc5s_idesc live in ../tma_desc.cuh)
// P74 o16 epilogue: bf16-store twin for the tc5 lanes (the o16 chain's
// GEMM half on sm_100, where the TMA lane is process-killed). y keeps the
// f32* ABI type; O16 reinterprets it as bf16 rows of the same geometry.
template <bool O16>
__device__ __forceinline__ void pd_tc5_store(float* y, size_t idx, uint32_t v) {
    if constexpr (O16)
        ((__nv_bfloat16*)y)[idx] = __float2bfloat16_rn(__uint_as_float(v));
    else
        y[idx] = __uint_as_float(v);
}
#endif

template <uint32_t S>
__global__ void __launch_bounds__(128) pd_f8bs_gemm_tc5_kt(
    const __grid_constant__ PdTmap wmap, const __grid_constant__ PdTmap ymap,
    const unsigned char* __restrict__ wsc, const unsigned char* __restrict__ xsc,
    float* __restrict__ y, uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
#if PD_TC5_OK
    // v2 (async-SF): zero per-slab CTA barriers. SF words ride cp.async into
    // the slot's smem in the CANONICAL 512B tile (lane l's 16 bytes at l*16;
    // word c = the 4 k-block bytes of row c*32+l) joining the slot's TMA
    // mbarrier via arrive.noinc (init count 129 = tid0's expect_tx + 128
    // per-thread arrivals); tid0 then tcgen05.cp's them to tensor memory
    // (desc LBO=1 SBO=8, probed exact) - cp and mma
    // form the implicit in-order tcgen05 pipeline, so no wait::st, and the
    // all-thread bdone phase-wait per slab covers every smem/tmem reuse
    // hazard (phases are observed, never consumed - one wait point per
    // iteration keeps every thread's parity bookkeeping identical).
    extern __shared__ __align__(1024) unsigned char pd_tc5b_sh[];
    unsigned char* wt = pd_tc5b_sh;
    unsigned char* yt = pd_tc5b_sh + S * 16384u;
    unsigned char* sfs = pd_tc5b_sh + 2u * S * 16384u;   // S x (512 SFA | 512 SFB)
    uint64_t* bfull = (uint64_t*)(sfs + S * 1024u);
    uint64_t* bdone = bfull + S;
    __shared__ uint32_t tmem_slot[1];

    const uint32_t tid = threadIdx.x;
    const uint32_t nk = (in_dim + 127u) / 128u;
    const uint32_t n_kb = in_dim >> 5;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t nct = batch_pad >> 7;
    const uint32_t row_base = (blockIdx.x / nct) * 128u;
    const uint32_t col_base = (blockIdx.x % nct) * 128u;

    if (tid == 0) {
        #pragma unroll
        for (uint32_t s = 0; s < S; ++s) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 129;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bfull[s])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bdone[s])));
        }
    }
    __syncthreads();
    if (tid < 32)
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 256;"
                     ::"r"((uint32_t)__cvta_generic_to_shared(tmem_slot)));
    __syncthreads();
    const uint32_t tmem = tmem_slot[0];        // D: cols 0..127
    const uint32_t sf_base = tmem + 128u;      // SF ring: S x (4 SFA + 4 SFB)

    auto bar_wait = [&](uint64_t* bar, uint32_t parity) {
        const uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
        asm volatile("{\n\t.reg .pred p;\nW%=:\n\t"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                     "@!p bra W%=;\n\t}" ::"r"(a), "r"(parity));
    };
    auto tma_stage = [&](uint32_t kt, uint32_t s) {
        const uint32_t m = (uint32_t)__cvta_generic_to_shared(&bfull[s]);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 32768;" ::"r"(m));
        const uint32_t wd = (uint32_t)__cvta_generic_to_shared(wt + s * 16384u);
        const uint32_t yd = (uint32_t)__cvta_generic_to_shared(yt + s * 16384u);
        const int ck = (int)(kt * 128u);
        asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                     " [%0], [%1, {%2, %3}], [%4];" ::"r"(wd), "l"(&wmap), "r"(ck),
                     "r"((int)row_base), "r"(m) : "memory");
        asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                     " [%0], [%1, {%2, %3}], [%4];" ::"r"(yd), "l"(&ymap), "r"(ck),
                     "r"((int)col_base), "r"(m) : "memory");
    };
    // per-thread SF staging for slab kt into slot s: this thread owns row
    // `tid` of both SF tiles (128 threads, one row each), 4B cp.asyncs into
    // the canonical positions, then one noinc arrival on the slot barrier
    auto sf_stage = [&](uint32_t kt, uint32_t s) {
        const uint32_t kb0 = kt * 4u;
        unsigned char* base = sfs + s * 1024u;
        const uint32_t off = (tid % 32u) * 16u + (tid / 32u) * 4u;
        const uint32_t rw = row_base + tid, rc = col_base + tid;
        pd_mma_cpa4p(base + off, wsc + (size_t)rw * n_kb + kb0,
                     rw < out_dim && kb0 + 4u <= n_kb);
        pd_mma_cpa4p(base + 512u + off, xsc + (size_t)rc * n_kb + kb0,
                     rc < batch && kb0 + 4u <= n_kb);
        const uint32_t m = (uint32_t)__cvta_generic_to_shared(&bfull[s]);
        asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(m) : "memory");
    };

    // prologue: stage slabs 0..S-1 (SF by every thread, TMA by tid0)
    #pragma unroll
    for (uint32_t s = 0; s < S; ++s) {
        if (s < nk) {
            sf_stage(s, s);
            if (tid == 0) tma_stage(s, s);
        }
    }
    uint32_t fph = 0, dph = 0;
    for (uint32_t kt = 0; kt < nk; ++kt) {
        const uint32_t s = kt % S;
        if (tid == 0) {
            bar_wait(&bfull[s], (fph >> s) & 1u);
            fph ^= 1u << s;
            const uint32_t v = (uint32_t)__cvta_generic_to_shared(sfs + s * 1024u) >> 4;
            const uint64_t da = ((uint64_t)(v & 0x3FFFu)) | ((uint64_t)1u << 16)
                              | ((uint64_t)8u << 32);
            const uint64_t db = ((uint64_t)((v + 32u) & 0x3FFFu)) | ((uint64_t)1u << 16)
                              | ((uint64_t)8u << 32);
            const uint32_t sfa_t = sf_base + s * 8u;
            const uint32_t sfb_t = sfa_t + 4u;
            asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                         ::"r"(sfa_t), "l"(da) : "memory");
            asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                         ::"r"(sfb_t), "l"(db) : "memory");
            const uint32_t w16 = (uint32_t)__cvta_generic_to_shared(wt + s * 16384u) >> 4;
            const uint32_t y16 = (uint32_t)__cvta_generic_to_shared(yt + s * 16384u) >> 4;
            #pragma unroll
            for (uint32_t kb = 0; kb < 4u; ++kb) {
                const uint64_t ad = pd_tc5_sdesc(w16 + kb * 2u);
                const uint64_t bd = pd_tc5_sdesc(y16 + kb * 2u);
                const uint32_t en = (kt > 0 || kb > 0) ? 1u : 0u;
                asm volatile(
                    "{\n\t.reg .pred p;\n\t"
                    "setp.ne.b32 p, %6, 0;\n\t"
                    "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X"
                    " [%0], %1, %2, %3, [%4], [%5], p;\n\t}"
                    ::"r"(tmem), "l"(ad), "l"(bd), "r"(pd_tc5_bs_idesc(kb)),
                      "r"(sfa_t), "r"(sfb_t), "r"(en));
            }
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                ::"r"((uint32_t)__cvta_generic_to_shared(&bdone[s])));
        }
        // slab kt's mmas complete -> slot s smem AND its SF tmem buffer are
        // reusable; every thread observes the same phase once per iteration
        bar_wait(&bdone[s], (dph >> s) & 1u);
        dph ^= 1u << s;
        const uint32_t pf = kt + S;
        if (pf < nk) {
            sf_stage(pf, s);
            if (tid == 0) tma_stage(pf, s);
        }
    }
    __syncthreads();
    #pragma unroll
    for (uint32_t cc = 0; cc < 4u; ++cc) {
        uint32_t r[32];
        const uint32_t warp = tid >> 5, lane = tid & 31u;
        const uint32_t taddr = tmem + ((warp * 32u) << 16) + cc * 32u;
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
            "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
            : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
              "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),"=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15]),
              "=r"(r[16]),"=r"(r[17]),"=r"(r[18]),"=r"(r[19]),"=r"(r[20]),"=r"(r[21]),"=r"(r[22]),"=r"(r[23]),
              "=r"(r[24]),"=r"(r[25]),"=r"(r[26]),"=r"(r[27]),"=r"(r[28]),"=r"(r[29]),"=r"(r[30]),"=r"(r[31])
            : "r"(taddr));
        asm volatile("tcgen05.wait::ld.sync.aligned;");
        const uint32_t row = row_base + warp * 32u + lane;
        if (row < out_dim) {
            #pragma unroll
            for (uint32_t j = 0; j < 32u; ++j) {
                const uint32_t col = col_base + cc * 32u + j;
                if (col < batch)
                    y[(size_t)col * out_dim + row] = __uint_as_float(r[j]);
            }
        }
    }
    __syncthreads();
    if (tid < 32)
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 256;" ::"r"(tmem));
#else
    (void)wmap; (void)ymap; (void)wsc; (void)xsc; (void)y;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}


// ---- prefill tc5bs with a DEFERRED recycle wait (mma pipelining) ----------
// The v18 c32/pf8 profiles put pd_f8bs_gemm_tc5_kt at 48-57 % of GPU time
// running at its measured 900-950 TF ceiling - ~21 % of the B200 e4m3 peak.
// Structural diagnosis: the per-slab bdone wait sits IMMEDIATELY after the
// slab's own mma issue, so the CTA stalls until the current slab's mmas
// complete before staging the next - only one slab is ever in the tensor
// pipe, which drains during every [commit-observe + sf_stage + tma issue]
// window. This variant defers the recycle wait by `dwait` slabs (0 = the
// shipped behavior): at iteration kt it waits bdone(kt-dwait) and restages
// that slot, so up to dwait+1 slabs' mmas overlap. All the original hazard
// arguments survive: the in-order tcgen05 pipe still covers cp-vs-mma on
// tid0's program order, smem/SF-slot recycling still happens strictly after
// the owning slab's bdone, and TMA slack shrinks from S to S-dwait slabs
// (keep dwait < S). Tail: the last dwait slabs' bdone waits drain after
// the loop, before the tcgen05.ld epilogue.
template <uint32_t S>
__global__ void __launch_bounds__(128) pd_f8bs_gemm_tc5pp_kt(
    const __grid_constant__ PdTmap wmap, const __grid_constant__ PdTmap ymap,
    const unsigned char* __restrict__ wsc, const unsigned char* __restrict__ xsc,
    float* __restrict__ y, uint32_t in_dim, uint32_t out_dim, uint32_t batch,
    uint32_t dwait) {
#if PD_TC5_OK
    extern __shared__ __align__(1024) unsigned char pd_tc5b_sh[];
    unsigned char* wt = pd_tc5b_sh;
    unsigned char* yt = pd_tc5b_sh + S * 16384u;
    unsigned char* sfs = pd_tc5b_sh + 2u * S * 16384u;   // S x (512 SFA | 512 SFB)
    uint64_t* bfull = (uint64_t*)(sfs + S * 1024u);
    uint64_t* bdone = bfull + S;
    __shared__ uint32_t tmem_slot[1];

    const uint32_t tid = threadIdx.x;
    const uint32_t nk = (in_dim + 127u) / 128u;
    const uint32_t n_kb = in_dim >> 5;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t nct = batch_pad >> 7;
    const uint32_t row_base = (blockIdx.x / nct) * 128u;
    const uint32_t col_base = (blockIdx.x % nct) * 128u;

    if (tid == 0) {
        #pragma unroll
        for (uint32_t s = 0; s < S; ++s) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 129;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bfull[s])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bdone[s])));
        }
    }
    __syncthreads();
    if (tid < 32)
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 256;"
                     ::"r"((uint32_t)__cvta_generic_to_shared(tmem_slot)));
    __syncthreads();
    const uint32_t tmem = tmem_slot[0];        // D: cols 0..127
    const uint32_t sf_base = tmem + 128u;      // SF ring: S x (4 SFA + 4 SFB)

    auto bar_wait = [&](uint64_t* bar, uint32_t parity) {
        const uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
        asm volatile("{\n\t.reg .pred p;\nW%=:\n\t"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                     "@!p bra W%=;\n\t}" ::"r"(a), "r"(parity));
    };
    auto tma_stage = [&](uint32_t kt, uint32_t s) {
        const uint32_t m = (uint32_t)__cvta_generic_to_shared(&bfull[s]);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 32768;" ::"r"(m));
        const uint32_t wd = (uint32_t)__cvta_generic_to_shared(wt + s * 16384u);
        const uint32_t yd = (uint32_t)__cvta_generic_to_shared(yt + s * 16384u);
        const int ck = (int)(kt * 128u);
        asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                     " [%0], [%1, {%2, %3}], [%4];" ::"r"(wd), "l"(&wmap), "r"(ck),
                     "r"((int)row_base), "r"(m) : "memory");
        asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                     " [%0], [%1, {%2, %3}], [%4];" ::"r"(yd), "l"(&ymap), "r"(ck),
                     "r"((int)col_base), "r"(m) : "memory");
    };
    auto sf_stage = [&](uint32_t kt, uint32_t s) {
        const uint32_t kb0 = kt * 4u;
        unsigned char* base = sfs + s * 1024u;
        const uint32_t off = (tid % 32u) * 16u + (tid / 32u) * 4u;
        const uint32_t rw = row_base + tid, rc = col_base + tid;
        pd_mma_cpa4p(base + off, wsc + (size_t)rw * n_kb + kb0,
                     rw < out_dim && kb0 + 4u <= n_kb);
        pd_mma_cpa4p(base + 512u + off, xsc + (size_t)rc * n_kb + kb0,
                     rc < batch && kb0 + 4u <= n_kb);
        const uint32_t m = (uint32_t)__cvta_generic_to_shared(&bfull[s]);
        asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(m) : "memory");
    };

    #pragma unroll
    for (uint32_t s = 0; s < S; ++s) {
        if (s < nk) {
            sf_stage(s, s);
            if (tid == 0) tma_stage(s, s);
        }
    }
    uint32_t fph = 0, dph = 0;
    for (uint32_t kt = 0; kt < nk; ++kt) {
        const uint32_t s = kt % S;
        if (tid == 0) {
            bar_wait(&bfull[s], (fph >> s) & 1u);
            fph ^= 1u << s;
            const uint32_t v = (uint32_t)__cvta_generic_to_shared(sfs + s * 1024u) >> 4;
            const uint64_t da = ((uint64_t)(v & 0x3FFFu)) | ((uint64_t)1u << 16)
                              | ((uint64_t)8u << 32);
            const uint64_t db = ((uint64_t)((v + 32u) & 0x3FFFu)) | ((uint64_t)1u << 16)
                              | ((uint64_t)8u << 32);
            const uint32_t sfa_t = sf_base + s * 8u;
            const uint32_t sfb_t = sfa_t + 4u;
            asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                         ::"r"(sfa_t), "l"(da) : "memory");
            asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                         ::"r"(sfb_t), "l"(db) : "memory");
            const uint32_t w16 = (uint32_t)__cvta_generic_to_shared(wt + s * 16384u) >> 4;
            const uint32_t y16 = (uint32_t)__cvta_generic_to_shared(yt + s * 16384u) >> 4;
            #pragma unroll
            for (uint32_t kb = 0; kb < 4u; ++kb) {
                const uint64_t ad = pd_tc5_sdesc(w16 + kb * 2u);
                const uint64_t bd = pd_tc5_sdesc(y16 + kb * 2u);
                const uint32_t en = (kt > 0 || kb > 0) ? 1u : 0u;
                asm volatile(
                    "{\n\t.reg .pred p;\n\t"
                    "setp.ne.b32 p, %6, 0;\n\t"
                    "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X"
                    " [%0], %1, %2, %3, [%4], [%5], p;\n\t}"
                    ::"r"(tmem), "l"(ad), "l"(bd), "r"(pd_tc5_bs_idesc(kb)),
                      "r"(sfa_t), "r"(sfb_t), "r"(en));
            }
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                ::"r"((uint32_t)__cvta_generic_to_shared(&bdone[s])));
        }
        // deferred recycle: settle slab kt-dwait, keeping up to dwait+1
        // slabs' mmas in flight
        if (kt >= dwait) {
            const uint32_t kd = kt - dwait, sd = kd % S;
            bar_wait(&bdone[sd], (dph >> sd) & 1u);
            dph ^= 1u << sd;
            const uint32_t pf = kd + S;
            if (pf < nk) {
                sf_stage(pf, sd);
                if (tid == 0) tma_stage(pf, sd);
            }
        }
    }
    // drain the deferred tail (covers the final slabs' mma completion, so
    // the tcgen05.ld below reads a settled accumulator)
    for (uint32_t kd = nk > dwait ? nk - dwait : 0; kd < nk; ++kd) {
        const uint32_t sd = kd % S;
        bar_wait(&bdone[sd], (dph >> sd) & 1u);
        dph ^= 1u << sd;
    }
    __syncthreads();
    #pragma unroll
    for (uint32_t cc = 0; cc < 4u; ++cc) {
        uint32_t r[32];
        const uint32_t warp = tid >> 5, lane = tid & 31u;
        const uint32_t taddr = tmem + ((warp * 32u) << 16) + cc * 32u;
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
            "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
            : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
              "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),"=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15]),
              "=r"(r[16]),"=r"(r[17]),"=r"(r[18]),"=r"(r[19]),"=r"(r[20]),"=r"(r[21]),"=r"(r[22]),"=r"(r[23]),
              "=r"(r[24]),"=r"(r[25]),"=r"(r[26]),"=r"(r[27]),"=r"(r[28]),"=r"(r[29]),"=r"(r[30]),"=r"(r[31])
            : "r"(taddr));
        asm volatile("tcgen05.wait::ld.sync.aligned;");
        const uint32_t row = row_base + warp * 32u + lane;
        if (row < out_dim) {
            #pragma unroll
            for (uint32_t j = 0; j < 32u; ++j) {
                const uint32_t col = col_base + cc * 32u + j;
                if (col < batch)
                    y[(size_t)col * out_dim + row] = __uint_as_float(r[j]);
            }
        }
    }
    __syncthreads();
    if (tid < 32)
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 256;" ::"r"(tmem));
#else
    (void)wmap; (void)ymap; (void)wsc; (void)xsc; (void)y;
    (void)in_dim; (void)out_dim; (void)batch; (void)dwait;
#endif
}

// ---- WIDE prefill tc5bs: one W stage feeds two Y col-tiles (N=256) --------
//  diagnosis: the 128x128 tc5_kt is L2-BANDWIDTH-bound, not mma-bound
// - the N=128 mma-rate probe (scratch mma_rate128.cu) clocks 130 ns/slab
// for block-scale AND plain f8 alike (4.76 PF aggregate floor, grid-blind),
// while the kernel moves W*(r/128) + Y*(out/128) = 1.83 GB of L2->SM
// traffic per gate GEMM at r=1024 and lands exactly at the ~7.2 TB/s the
// fabric delivers (900-950 TF, flat in r - the L2-bound signature; the
// deferred-wait experiment tc5pp was neutral, killing the pipeline-drain
// theory). The only lever is arithmetic intensity: this variant stages one
// 16KB W slab and two 16KB Y tiles per slab (effective M128xN256), doubles
// D to 256 tmem cols (alloc 512), runs 8 mmas/slab (still 2.4x under the
// mma floor), and halves the W component of the traffic:
// 452 + 924 = 1.38 GB at the gate shape -> ~1.2 PF if the L2 model holds.
// COL: A-collector qualifiers on the mma sequence - each kb's A slab is
// read from smem once (::fill at t=0) and reused from the TensorCore
// collector buffer for the remaining N tiles (::use / ::lastuse). Reuse is
// opportunistic per the ISA, so exactness is unconditional.
// WN: wide-N mmas - cover the NT*128 cols in N=256 mmas (+ an N=128 tail
// when NT is odd) instead of NT separate N=128 mmas, so A is ingested once
// per 256 cols by construction. Legal because the NT stacked 16KB Y tiles
// are the canonical SW128 image of a (NT*128)x128 matrix (16 KB per 128
// rows, SBO walks straight through), the SFB ring tiles are contiguous in
// tmem, and D cols are contiguous. N caps at 256 for kind::mxf8f6f4 ::1.
template <uint32_t S, uint32_t NT, bool COL = false, bool WN = false>
__global__ void __launch_bounds__(128) pd_f8bs_gemm_tc5w_kt(
    const __grid_constant__ PdTmap wmap, const __grid_constant__ PdTmap ymap,
    const unsigned char* __restrict__ wsc, const unsigned char* __restrict__ xsc,
    float* __restrict__ y, uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
#if PD_TC5_OK
    extern __shared__ __align__(1024) unsigned char pd_tc5w_sh[];
    unsigned char* wt = pd_tc5w_sh;                       // S x 16KB W
    unsigned char* yt = pd_tc5w_sh + S * 16384u;          // S x NT x 16KB Y
    unsigned char* sfs = pd_tc5w_sh + (1u + NT) * S * 16384u;  // S x (512 SFA | NT x 512 SFB)
    uint64_t* bfull = (uint64_t*)(sfs + S * (512u + NT * 512u));
    uint64_t* bdone = bfull + S;
    __shared__ uint32_t tmem_slot[1];

    const uint32_t tid = threadIdx.x;
    const uint32_t nk = (in_dim + 127u) / 128u;
    const uint32_t n_kb = in_dim >> 5;
    const uint32_t batch_pad = (batch + NT * 128u - 1u) / (NT * 128u) * (NT * 128u);
    const uint32_t nct = batch_pad / (NT * 128u);
    const uint32_t row_base = (blockIdx.x / nct) * 128u;
    const uint32_t col_base = (blockIdx.x % nct) * NT * 128u;

    if (tid == 0) {
        #pragma unroll
        for (uint32_t s = 0; s < S; ++s) {
            // 129 = tid0's expect_tx arrive + 128 per-thread cp arrivals
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 129;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bfull[s])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bdone[s])));
        }
    }
    __syncthreads();
    if (tid < 32)
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;"
                     ::"r"((uint32_t)__cvta_generic_to_shared(tmem_slot)));
    __syncthreads();
    const uint32_t tmem = tmem_slot[0];        // D_t: cols t*128..t*128+127
    const uint32_t sf_base = tmem + NT * 128u; // SF ring: S x (4 SFA + NT x 4 SFB)

    auto bar_wait = [&](uint64_t* bar, uint32_t parity) {
        const uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
        asm volatile("{\n\t.reg .pred p;\nW%=:\n\t"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                     "@!p bra W%=;\n\t}" ::"r"(a), "r"(parity));
    };
    auto tma_stage = [&](uint32_t kt, uint32_t s) {
        const uint32_t m = (uint32_t)__cvta_generic_to_shared(&bfull[s]);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                     ::"r"(m), "r"((1u + NT) * 16384u));
        const uint32_t wd = (uint32_t)__cvta_generic_to_shared(wt + s * 16384u);
        const uint32_t yd = (uint32_t)__cvta_generic_to_shared(yt + s * NT * 16384u);
        const int ck = (int)(kt * 128u);
        asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                     " [%0], [%1, {%2, %3}], [%4];" ::"r"(wd), "l"(&wmap), "r"(ck),
                     "r"((int)row_base), "r"(m) : "memory");
        #pragma unroll
        for (uint32_t t = 0; t < NT; ++t)
            asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                         " [%0], [%1, {%2, %3}], [%4];" ::"r"(yd + t * 16384u), "l"(&ymap),
                         "r"(ck), "r"((int)(col_base + t * 128u)), "r"(m) : "memory");
    };
    // SF staging: thread owns row `tid` of the W-scale tile and both
    // activation-scale tiles (canonical 512B layout each)
    auto sf_stage = [&](uint32_t kt, uint32_t s) {
        const uint32_t kb0 = kt * 4u;
        unsigned char* base = sfs + s * (512u + NT * 512u);
        const uint32_t off = (tid % 32u) * 16u + (tid / 32u) * 4u;
        const uint32_t rw = row_base + tid;
        pd_mma_cpa4p(base + off, wsc + (size_t)rw * n_kb + kb0,
                     rw < out_dim && kb0 + 4u <= n_kb);
        #pragma unroll
        for (uint32_t t = 0; t < NT; ++t) {
            const uint32_t rc = col_base + t * 128u + tid;
            pd_mma_cpa4p(base + 512u + t * 512u + off, xsc + (size_t)rc * n_kb + kb0,
                         rc < batch && kb0 + 4u <= n_kb);
        }
        const uint32_t m = (uint32_t)__cvta_generic_to_shared(&bfull[s]);
        asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(m) : "memory");
    };

    #pragma unroll
    for (uint32_t s = 0; s < S; ++s) {
        if (s < nk) {
            sf_stage(s, s);
            if (tid == 0) tma_stage(s, s);
        }
    }
    uint32_t fph = 0, dph = 0;
    for (uint32_t kt = 0; kt < nk; ++kt) {
        const uint32_t s = kt % S;
        if (tid == 0) {
            bar_wait(&bfull[s], (fph >> s) & 1u);
            fph ^= 1u << s;
            const uint32_t v = (uint32_t)__cvta_generic_to_shared(
                sfs + s * (512u + NT * 512u)) >> 4;
            const uint64_t da = ((uint64_t)(v & 0x3FFFu)) | ((uint64_t)1u << 16)
                              | ((uint64_t)8u << 32);
            const uint32_t sfa_t = sf_base + s * (4u + NT * 4u);
            asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                         ::"r"(sfa_t), "l"(da) : "memory");
            #pragma unroll
            for (uint32_t t = 0; t < NT; ++t) {
                const uint64_t db = ((uint64_t)((v + 32u * (t + 1u)) & 0x3FFFu))
                                  | ((uint64_t)1u << 16) | ((uint64_t)8u << 32);
                asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                             ::"r"(sfa_t + 4u * (t + 1u)), "l"(db) : "memory");
            }
            const uint32_t w16 = (uint32_t)__cvta_generic_to_shared(wt + s * 16384u) >> 4;
            const uint32_t y16 = (uint32_t)__cvta_generic_to_shared(yt + s * NT * 16384u) >> 4;
            #pragma unroll
            for (uint32_t kb = 0; kb < 4u; ++kb) {
                const uint64_t ad = pd_tc5_sdesc(w16 + kb * 2u);
                const uint32_t en = (kt > 0 || kb > 0) ? 1u : 0u;
                if (WN) {
                    #pragma unroll
                    for (uint32_t t = 0; t < NT; t += 2u) {
                        const uint32_t nn = (NT - t >= 2u) ? 256u : 128u;
                        const uint64_t bd = pd_tc5_sdesc(y16 + t * 1024u + kb * 2u);
                        const uint32_t id = ((kb & 3u) << 4) | ((nn >> 3) << 17)
                            | (1u << 23) | ((128u >> 4) << 24) | ((kb & 3u) << 29);
                        asm volatile(
                            "{\n\t.reg .pred p;\n\t"
                            "setp.ne.b32 p, %6, 0;\n\t"
                            "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X"
                            " [%0], %1, %2, %3, [%4], [%5], p;\n\t}"
                            ::"r"(tmem + t * 128u), "l"(ad), "l"(bd), "r"(id),
                              "r"(sfa_t), "r"(sfa_t + 4u * (t + 1u)), "r"(en));
                    }
                    continue;
                }
                #pragma unroll
                for (uint32_t t = 0; t < NT; ++t) {
                    const uint64_t bd = pd_tc5_sdesc(y16 + t * 1024u + kb * 2u);
                    if (COL && NT > 1u && t == 0u) {
                        asm volatile(
                            "{\n\t.reg .pred p;\n\t"
                            "setp.ne.b32 p, %6, 0;\n\t"
                            "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X"
                            ".collector::a::fill"
                            " [%0], %1, %2, %3, [%4], [%5], p;\n\t}"
                            ::"r"(tmem + t * 128u), "l"(ad), "l"(bd), "r"(pd_tc5_bs_idesc(kb)),
                              "r"(sfa_t), "r"(sfa_t + 4u * (t + 1u)), "r"(en));
                    } else if (COL && NT > 1u && t + 1u < NT) {
                        asm volatile(
                            "{\n\t.reg .pred p;\n\t"
                            "setp.ne.b32 p, %6, 0;\n\t"
                            "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X"
                            ".collector::a::use"
                            " [%0], %1, %2, %3, [%4], [%5], p;\n\t}"
                            ::"r"(tmem + t * 128u), "l"(ad), "l"(bd), "r"(pd_tc5_bs_idesc(kb)),
                              "r"(sfa_t), "r"(sfa_t + 4u * (t + 1u)), "r"(en));
                    } else if (COL && NT > 1u) {
                        asm volatile(
                            "{\n\t.reg .pred p;\n\t"
                            "setp.ne.b32 p, %6, 0;\n\t"
                            "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X"
                            ".collector::a::lastuse"
                            " [%0], %1, %2, %3, [%4], [%5], p;\n\t}"
                            ::"r"(tmem + t * 128u), "l"(ad), "l"(bd), "r"(pd_tc5_bs_idesc(kb)),
                              "r"(sfa_t), "r"(sfa_t + 4u * (t + 1u)), "r"(en));
                    } else {
                        asm volatile(
                            "{\n\t.reg .pred p;\n\t"
                            "setp.ne.b32 p, %6, 0;\n\t"
                            "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X"
                            " [%0], %1, %2, %3, [%4], [%5], p;\n\t}"
                            ::"r"(tmem + t * 128u), "l"(ad), "l"(bd), "r"(pd_tc5_bs_idesc(kb)),
                              "r"(sfa_t), "r"(sfa_t + 4u * (t + 1u)), "r"(en));
                    }
                }
            }
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                ::"r"((uint32_t)__cvta_generic_to_shared(&bdone[s])));
        }
        bar_wait(&bdone[s], (dph >> s) & 1u);
        dph ^= 1u << s;
        const uint32_t pf = kt + S;
        if (pf < nk) {
            sf_stage(pf, s);
            if (tid == 0) tma_stage(pf, s);
        }
    }
    __syncthreads();
    #pragma unroll
    for (uint32_t dt = 0; dt < NT; ++dt) {
        #pragma unroll
        for (uint32_t cc = 0; cc < 4u; ++cc) {
            uint32_t r[32];
            const uint32_t warp = tid >> 5, lane = tid & 31u;
            const uint32_t taddr = tmem + dt * 128u + ((warp * 32u) << 16) + cc * 32u;
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
                "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
                "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
                : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
                  "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),"=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15]),
                  "=r"(r[16]),"=r"(r[17]),"=r"(r[18]),"=r"(r[19]),"=r"(r[20]),"=r"(r[21]),"=r"(r[22]),"=r"(r[23]),
                  "=r"(r[24]),"=r"(r[25]),"=r"(r[26]),"=r"(r[27]),"=r"(r[28]),"=r"(r[29]),"=r"(r[30]),"=r"(r[31])
                : "r"(taddr));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            const uint32_t row = row_base + warp * 32u + lane;
            if (row < out_dim) {
                #pragma unroll
                for (uint32_t j = 0; j < 32u; ++j) {
                    const uint32_t col = col_base + dt * 128u + cc * 32u + j;
                    if (col < batch)
                        y[(size_t)col * out_dim + row] = __uint_as_float(r[j]);
                }
            }
        }
    }
    __syncthreads();
    if (tid < 32)
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" ::"r"(tmem));
#else
    (void)wmap; (void)ymap; (void)wsc; (void)xsc; (void)y;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}

// ---- pipelined-issuer wide tile: tc5w minus the per-stage pipe drain ------
//  diagnosis (closes the prefill GEMM model): the production-stream
// mma rate is 574 ns per NT=3 K-stage (21.9 TF/SM, grid-blind - probed with
// the exact cp+mma+commit group, scratch mma_rate_w.cu), but tc5w times at
// 1167 ns/stage (10.8 TF/SM at 1 CTA, L2-resident). The gap is the loop
// structure: the same thread that issues the mmas also waits the CURRENT
// stage's bdone (the smem-recycle guard) before issuing the next stage, so
// the tensor pipe fully drains at every stage boundary - a fixed ~590 ns of
// unoverlapped ramp+commit+sync per stage. Every earlier fit ("smem operand
// B/F", "TMA ingest") was really counting stage boundaries per FLOP: time =
// 574*(FLOPs/NT3-stage) + 590 ns reproduces kt/NT2/NT3 within 2%.
// Fix: split the roles. A 5th warp's leader (tid 128) issues cps+mmas,
// waiting only bfull[s] and the S-AGO bdone (SF-tmem-slot recycle, already
// satisfied at steady state - the rate probe's exact pattern); warps 0-3
// stage SF, issue TMA behind the CURRENT bdone as before, and run the
// epilogue. The pipe never drains; both sides watch the same mbarrier
// phases with independent parity masks (parity waits are non-consuming).
template <uint32_t S, uint32_t NT, bool O16 = false>
__global__ void __launch_bounds__(160) pd_f8bs_gemm_tc5v_kt(
    const __grid_constant__ PdTmap wmap, const __grid_constant__ PdTmap ymap,
    const unsigned char* __restrict__ wsc, const unsigned char* __restrict__ xsc,
    float* __restrict__ y, uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
#if PD_TC5_OK
    extern __shared__ __align__(1024) unsigned char pd_tc5v_sh[];
    unsigned char* wt = pd_tc5v_sh;                       // S x 16KB W
    unsigned char* yt = pd_tc5v_sh + S * 16384u;          // S x NT x 16KB Y
    unsigned char* sfs = pd_tc5v_sh + (1u + NT) * S * 16384u;  // S x (512 SFA | NT x 512 SFB)
    uint64_t* bfull = (uint64_t*)(sfs + S * (512u + NT * 512u));
    uint64_t* bdone = bfull + S;
    __shared__ uint32_t tmem_slot[1];

    const uint32_t tid = threadIdx.x;
    const uint32_t nk = (in_dim + 127u) / 128u;
    const uint32_t n_kb = in_dim >> 5;
    const uint32_t batch_pad = (batch + NT * 128u - 1u) / (NT * 128u) * (NT * 128u);
    const uint32_t nct = batch_pad / (NT * 128u);
    const uint32_t row_base = (blockIdx.x / nct) * 128u;
    const uint32_t col_base = (blockIdx.x % nct) * NT * 128u;

    if (tid == 0) {
        #pragma unroll
        for (uint32_t s = 0; s < S; ++s) {
            // 129 = tid0's expect_tx arrive + 128 sf-staging cp arrivals
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 129;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bfull[s])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bdone[s])));
        }
    }
    __syncthreads();
    if (tid < 32)
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;"
                     ::"r"((uint32_t)__cvta_generic_to_shared(tmem_slot)));
    __syncthreads();
    const uint32_t tmem = tmem_slot[0];        // D_t: cols t*128..t*128+127
    const uint32_t sf_base = tmem + NT * 128u; // SF ring: S x (4 SFA + NT x 4 SFB)

    auto bar_wait = [&](uint64_t* bar, uint32_t parity) {
        const uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
        asm volatile("{\n\t.reg .pred p;\nW%=:\n\t"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                     "@!p bra W%=;\n\t}" ::"r"(a), "r"(parity));
    };
    auto tma_stage = [&](uint32_t kt, uint32_t s) {
        const uint32_t m = (uint32_t)__cvta_generic_to_shared(&bfull[s]);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                     ::"r"(m), "r"((1u + NT) * 16384u));
        const uint32_t wd = (uint32_t)__cvta_generic_to_shared(wt + s * 16384u);
        const uint32_t yd = (uint32_t)__cvta_generic_to_shared(yt + s * NT * 16384u);
        const int ck = (int)(kt * 128u);
        asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                     " [%0], [%1, {%2, %3}], [%4];" ::"r"(wd), "l"(&wmap), "r"(ck),
                     "r"((int)row_base), "r"(m) : "memory");
        #pragma unroll
        for (uint32_t t = 0; t < NT; ++t)
            asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                         " [%0], [%1, {%2, %3}], [%4];" ::"r"(yd + t * 16384u), "l"(&ymap),
                         "r"(ck), "r"((int)(col_base + t * 128u)), "r"(m) : "memory");
    };
    auto sf_stage = [&](uint32_t kt, uint32_t s) {
        const uint32_t kb0 = kt * 4u;
        unsigned char* base = sfs + s * (512u + NT * 512u);
        const uint32_t off = (tid % 32u) * 16u + (tid / 32u) * 4u;
        const uint32_t rw = row_base + tid;
        pd_mma_cpa4p(base + off, wsc + (size_t)rw * n_kb + kb0,
                     rw < out_dim && kb0 + 4u <= n_kb);
        #pragma unroll
        for (uint32_t t = 0; t < NT; ++t) {
            const uint32_t rc = col_base + t * 128u + tid;
            pd_mma_cpa4p(base + 512u + t * 512u + off, xsc + (size_t)rc * n_kb + kb0,
                         rc < batch && kb0 + 4u <= n_kb);
        }
        const uint32_t m = (uint32_t)__cvta_generic_to_shared(&bfull[s]);
        asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(m) : "memory");
    };

    if (tid < 128) {
        #pragma unroll
        for (uint32_t s = 0; s < S; ++s) {
            if (s < nk) {
                sf_stage(s, s);
                if (tid == 0) tma_stage(s, s);
            }
        }
    }
    if (tid == 128) {
        // issuer: never waits the current stage - only bfull and the S-ago
        // bdone (guards the SF-tmem ring slot against the mma group still
        // reading it; a no-op at steady state)
        uint32_t fph = 0, iph = 0;
        for (uint32_t kt = 0; kt < nk; ++kt) {
            const uint32_t s = kt % S;
            if (kt >= S) {
                bar_wait(&bdone[s], (iph >> s) & 1u);
                iph ^= 1u << s;
            }
            bar_wait(&bfull[s], (fph >> s) & 1u);
            fph ^= 1u << s;
            const uint32_t v = (uint32_t)__cvta_generic_to_shared(
                sfs + s * (512u + NT * 512u)) >> 4;
            const uint64_t da = ((uint64_t)(v & 0x3FFFu)) | ((uint64_t)1u << 16)
                              | ((uint64_t)8u << 32);
            const uint32_t sfa_t = sf_base + s * (4u + NT * 4u);
            asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                         ::"r"(sfa_t), "l"(da) : "memory");
            #pragma unroll
            for (uint32_t t = 0; t < NT; ++t) {
                const uint64_t db = ((uint64_t)((v + 32u * (t + 1u)) & 0x3FFFu))
                                  | ((uint64_t)1u << 16) | ((uint64_t)8u << 32);
                asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                             ::"r"(sfa_t + 4u * (t + 1u)), "l"(db) : "memory");
            }
            const uint32_t w16 = (uint32_t)__cvta_generic_to_shared(wt + s * 16384u) >> 4;
            const uint32_t y16 = (uint32_t)__cvta_generic_to_shared(yt + s * NT * 16384u) >> 4;
            #pragma unroll
            for (uint32_t kb = 0; kb < 4u; ++kb) {
                const uint64_t ad = pd_tc5_sdesc(w16 + kb * 2u);
                const uint32_t en = (kt > 0 || kb > 0) ? 1u : 0u;
                #pragma unroll
                for (uint32_t t = 0; t < NT; ++t) {
                    const uint64_t bd = pd_tc5_sdesc(y16 + t * 1024u + kb * 2u);
                    asm volatile(
                        "{\n\t.reg .pred p;\n\t"
                        "setp.ne.b32 p, %6, 0;\n\t"
                        "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X"
                        " [%0], %1, %2, %3, [%4], [%5], p;\n\t}"
                        ::"r"(tmem + t * 128u), "l"(ad), "l"(bd), "r"(pd_tc5_bs_idesc(kb)),
                          "r"(sfa_t), "r"(sfa_t + 4u * (t + 1u)), "r"(en));
                }
            }
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                ::"r"((uint32_t)__cvta_generic_to_shared(&bdone[s])));
        }
    } else if (tid < 128) {
        // producers: recycle smem slot s behind the CURRENT stage's bdone,
        // exactly the tc5w cadence - off the mma critical path now
        uint32_t dph = 0;
        for (uint32_t kt = 0; kt < nk; ++kt) {
            const uint32_t s = kt % S;
            bar_wait(&bdone[s], (dph >> s) & 1u);
            dph ^= 1u << s;
            const uint32_t pf = kt + S;
            if (pf < nk) {
                sf_stage(pf, s);
                if (tid == 0) tma_stage(pf, s);
            }
        }
    }
    __syncthreads();
    if (tid < 128) {
        #pragma unroll
        for (uint32_t dt = 0; dt < NT; ++dt) {
            #pragma unroll
            for (uint32_t cc = 0; cc < 4u; ++cc) {
                uint32_t r[32];
                const uint32_t warp = tid >> 5, lane = tid & 31u;
                const uint32_t taddr = tmem + dt * 128u + ((warp * 32u) << 16) + cc * 32u;
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
                    "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
                    "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
                    : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
                      "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),"=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15]),
                      "=r"(r[16]),"=r"(r[17]),"=r"(r[18]),"=r"(r[19]),"=r"(r[20]),"=r"(r[21]),"=r"(r[22]),"=r"(r[23]),
                      "=r"(r[24]),"=r"(r[25]),"=r"(r[26]),"=r"(r[27]),"=r"(r[28]),"=r"(r[29]),"=r"(r[30]),"=r"(r[31])
                    : "r"(taddr));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                const uint32_t row = row_base + warp * 32u + lane;
                if (row < out_dim) {
                    #pragma unroll
                    for (uint32_t j = 0; j < 32u; ++j) {
                        const uint32_t col = col_base + dt * 128u + cc * 32u + j;
                        if (col < batch)
                            pd_tc5_store<O16>(y, (size_t)col * out_dim + row, r[j]);
                    }
                }
            }
        }
    }
    __syncthreads();
    if (tid < 32)
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" ::"r"(tmem));
#else
    (void)wmap; (void)ymap; (void)wsc; (void)xsc; (void)y;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}

// ---- A-from-tmem wide tile: tc5w with the W operand read from tmem --------
//  closed on the per-SM smem-operand feed (~49 GB/s/SM): the tensor
// pipe re-reads the 16KB W stage NT times per K-stage from smem (once per Y
// col tile), and that stream - not TMA, not L2, not the mma rate - is the
// wall (multicast and 2-SM MMA were both neutral because they only cut the
// TMA/L2 side). This variant moves W out of the smem operand stream: one
// tcgen05.cp per K=32 slab (128x256b, validated exact)
// lands the stage in a tmem ring, and the mmas take [a-tmem] - B becomes
// the only smem operand, dropping tensor-pipe smem reads per stage from
// (1+NT)x16KB to NTx16KB (the cp's own read is a different engine). tmem
// budget: NTx128 D + Sx32 A ring + Sx(4+NTx4) SF ring <= 512 for both
// production shapes (S2/NT3: 480, S3/NT2: 388). A-slot recycle is free: the
// stage-kt commit tracks its cps AND mmas, and slot s's previous reader
// (stage kt-S) committed before stage kt's TMA was issued.
template <uint32_t S, uint32_t NT>
__global__ void __launch_bounds__(128) pd_f8bs_gemm_tc5z_kt(
    const __grid_constant__ PdTmap wmap, const __grid_constant__ PdTmap ymap,
    const unsigned char* __restrict__ wsc, const unsigned char* __restrict__ xsc,
    float* __restrict__ y, uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
#if PD_TC5_OK
    extern __shared__ __align__(1024) unsigned char pd_tc5z_sh[];
    unsigned char* wt = pd_tc5z_sh;                       // S x 16KB W
    unsigned char* yt = pd_tc5z_sh + S * 16384u;          // S x NT x 16KB Y
    unsigned char* sfs = pd_tc5z_sh + (1u + NT) * S * 16384u;  // S x (512 SFA | NT x 512 SFB)
    uint64_t* bfull = (uint64_t*)(sfs + S * (512u + NT * 512u));
    uint64_t* bdone = bfull + S;
    __shared__ uint32_t tmem_slot[1];

    const uint32_t tid = threadIdx.x;
    const uint32_t nk = (in_dim + 127u) / 128u;
    const uint32_t n_kb = in_dim >> 5;
    const uint32_t batch_pad = (batch + NT * 128u - 1u) / (NT * 128u) * (NT * 128u);
    const uint32_t nct = batch_pad / (NT * 128u);
    const uint32_t row_base = (blockIdx.x / nct) * 128u;
    const uint32_t col_base = (blockIdx.x % nct) * NT * 128u;

    if (tid == 0) {
        #pragma unroll
        for (uint32_t s = 0; s < S; ++s) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 129;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bfull[s])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bdone[s])));
        }
    }
    __syncthreads();
    if (tid < 32)
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;"
                     ::"r"((uint32_t)__cvta_generic_to_shared(tmem_slot)));
    __syncthreads();
    const uint32_t tmem = tmem_slot[0];        // D_t: cols t*128..t*128+127
    const uint32_t a_base = tmem + NT * 128u;  // A ring: S x 32 cols (K=128 e4m3)
    const uint32_t sf_base = a_base + S * 32u; // SF ring: S x (4 SFA + NT x 4 SFB)

    auto bar_wait = [&](uint64_t* bar, uint32_t parity) {
        const uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
        asm volatile("{\n\t.reg .pred p;\nW%=:\n\t"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                     "@!p bra W%=;\n\t}" ::"r"(a), "r"(parity));
    };
    auto tma_stage = [&](uint32_t kt, uint32_t s) {
        const uint32_t m = (uint32_t)__cvta_generic_to_shared(&bfull[s]);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                     ::"r"(m), "r"((1u + NT) * 16384u));
        const uint32_t wd = (uint32_t)__cvta_generic_to_shared(wt + s * 16384u);
        const uint32_t yd = (uint32_t)__cvta_generic_to_shared(yt + s * NT * 16384u);
        const int ck = (int)(kt * 128u);
        asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                     " [%0], [%1, {%2, %3}], [%4];" ::"r"(wd), "l"(&wmap), "r"(ck),
                     "r"((int)row_base), "r"(m) : "memory");
        #pragma unroll
        for (uint32_t t = 0; t < NT; ++t)
            asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                         " [%0], [%1, {%2, %3}], [%4];" ::"r"(yd + t * 16384u), "l"(&ymap),
                         "r"(ck), "r"((int)(col_base + t * 128u)), "r"(m) : "memory");
    };
    auto sf_stage = [&](uint32_t kt, uint32_t s) {
        const uint32_t kb0 = kt * 4u;
        unsigned char* base = sfs + s * (512u + NT * 512u);
        const uint32_t off = (tid % 32u) * 16u + (tid / 32u) * 4u;
        const uint32_t rw = row_base + tid;
        pd_mma_cpa4p(base + off, wsc + (size_t)rw * n_kb + kb0,
                     rw < out_dim && kb0 + 4u <= n_kb);
        #pragma unroll
        for (uint32_t t = 0; t < NT; ++t) {
            const uint32_t rc = col_base + t * 128u + tid;
            pd_mma_cpa4p(base + 512u + t * 512u + off, xsc + (size_t)rc * n_kb + kb0,
                         rc < batch && kb0 + 4u <= n_kb);
        }
        const uint32_t m = (uint32_t)__cvta_generic_to_shared(&bfull[s]);
        asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(m) : "memory");
    };

    #pragma unroll
    for (uint32_t s = 0; s < S; ++s) {
        if (s < nk) {
            sf_stage(s, s);
            if (tid == 0) tma_stage(s, s);
        }
    }
    uint32_t fph = 0, dph = 0;
    for (uint32_t kt = 0; kt < nk; ++kt) {
        const uint32_t s = kt % S;
        if (tid == 0) {
            bar_wait(&bfull[s], (fph >> s) & 1u);
            fph ^= 1u << s;
            const uint32_t v = (uint32_t)__cvta_generic_to_shared(
                sfs + s * (512u + NT * 512u)) >> 4;
            const uint64_t da = ((uint64_t)(v & 0x3FFFu)) | ((uint64_t)1u << 16)
                              | ((uint64_t)8u << 32);
            const uint32_t sfa_t = sf_base + s * (4u + NT * 4u);
            asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                         ::"r"(sfa_t), "l"(da) : "memory");
            #pragma unroll
            for (uint32_t t = 0; t < NT; ++t) {
                const uint64_t db = ((uint64_t)((v + 32u * (t + 1u)) & 0x3FFFu))
                                  | ((uint64_t)1u << 16) | ((uint64_t)8u << 32);
                asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                             ::"r"(sfa_t + 4u * (t + 1u)), "l"(db) : "memory");
            }
            const uint32_t w16 = (uint32_t)__cvta_generic_to_shared(wt + s * 16384u) >> 4;
            const uint32_t y16 = (uint32_t)__cvta_generic_to_shared(yt + s * NT * 16384u) >> 4;
            // W stage -> tmem A slot (one 128x256b cp per K=32 slab); the
            // cp->mma implicit pipeline orders these against the mmas below
            const uint32_t at = a_base + s * 32u;
            #pragma unroll
            for (uint32_t kb = 0; kb < 4u; ++kb)
                asm volatile("tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                             ::"r"(at + kb * 8u), "l"(pd_tc5_sdesc(w16 + kb * 2u)) : "memory");
            #pragma unroll
            for (uint32_t kb = 0; kb < 4u; ++kb) {
                const uint32_t en = (kt > 0 || kb > 0) ? 1u : 0u;
                #pragma unroll
                for (uint32_t t = 0; t < NT; ++t) {
                    const uint64_t bd = pd_tc5_sdesc(y16 + t * 1024u + kb * 2u);
                    asm volatile(
                        "{\n\t.reg .pred p;\n\t"
                        "setp.ne.b32 p, %6, 0;\n\t"
                        "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X"
                        " [%0], [%1], %2, %3, [%4], [%5], p;\n\t}"
                        ::"r"(tmem + t * 128u), "r"(at + kb * 8u), "l"(bd),
                          "r"(pd_tc5_bs_idesc(kb)),
                          "r"(sfa_t), "r"(sfa_t + 4u * (t + 1u)), "r"(en));
                }
            }
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                ::"r"((uint32_t)__cvta_generic_to_shared(&bdone[s])));
        }
        bar_wait(&bdone[s], (dph >> s) & 1u);
        dph ^= 1u << s;
        const uint32_t pf = kt + S;
        if (pf < nk) {
            sf_stage(pf, s);
            if (tid == 0) tma_stage(pf, s);
        }
    }
    __syncthreads();
    #pragma unroll
    for (uint32_t dt = 0; dt < NT; ++dt) {
        #pragma unroll
        for (uint32_t cc = 0; cc < 4u; ++cc) {
            uint32_t r[32];
            const uint32_t warp = tid >> 5, lane = tid & 31u;
            const uint32_t taddr = tmem + dt * 128u + ((warp * 32u) << 16) + cc * 32u;
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
                "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
                "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
                : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
                  "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),"=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15]),
                  "=r"(r[16]),"=r"(r[17]),"=r"(r[18]),"=r"(r[19]),"=r"(r[20]),"=r"(r[21]),"=r"(r[22]),"=r"(r[23]),
                  "=r"(r[24]),"=r"(r[25]),"=r"(r[26]),"=r"(r[27]),"=r"(r[28]),"=r"(r[29]),"=r"(r[30]),"=r"(r[31])
                : "r"(taddr));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            const uint32_t row = row_base + warp * 32u + lane;
            if (row < out_dim) {
                #pragma unroll
                for (uint32_t j = 0; j < 32u; ++j) {
                    const uint32_t col = col_base + dt * 128u + cc * 32u + j;
                    if (col < batch)
                        y[(size_t)col * out_dim + row] = __uint_as_float(r[j]);
                }
            }
        }
    }
    __syncthreads();
    if (tid < 32)
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" ::"r"(tmem));
#else
    (void)wmap; (void)ymap; (void)wsc; (void)xsc; (void)y;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}

// ---- decode-shaped block-scale tcgen05 GEMM (M=128, N=64, K-split) --------
// The 128x128 tc5bs tile underfills decode shapes (gate at r<=64 = 168
// CTAs, 0.57 waves at 2 CTAs/SM) and measured 1.5-2.5 TB/s there while the
// same machinery streams 947 TF at prefill. M=64 would double the row tiles
// but the block-scale kind rejects it: probed on sm_100a, m_dim=4 raises
// illegal instruction, m_dim=8 survives - block_scale needs the full
// 128-lane SF datapath. So the decode shape keeps M=128 (16KB W slabs) and
// shrinks N to 64 (8KB Y via pd_tmap_2d_h64, halves the tail waste at
// r<=64); the CTA multiplier is the grid.y K-split with the format-blind
// ks combine. The bar for this shape is ~5-6 TB/s of weight stream.
template <uint32_t S>
__global__ void __launch_bounds__(128) pd_f8bs_gemm_tc5d_kt(
    const __grid_constant__ PdTmap wmap, const __grid_constant__ PdTmap ymap,
    const unsigned char* __restrict__ wsc, const unsigned char* __restrict__ xsc,
    float* __restrict__ y, uint32_t in_dim, uint32_t out_dim, uint32_t batch,
    uint32_t nz, uint32_t pdist) {
#if PD_TC5_OK
    extern __shared__ __align__(1024) unsigned char pd_tc5d_sh[];
    unsigned char* wt = pd_tc5d_sh;                       // S x 16 KB
    unsigned char* yt = pd_tc5d_sh + S * 16384u;          // S x 8 KB
    unsigned char* sfs = pd_tc5d_sh + S * 24576u;         // S x 1024
    uint64_t* bfull = (uint64_t*)(sfs + S * 1024u);
    uint64_t* bdone = bfull + S;
    __shared__ uint32_t tmem_slot[1];

    const uint32_t tid = threadIdx.x;
    const uint32_t nk_all = (in_dim + 127u) / 128u;
    const uint32_t per = (nk_all + nz - 1u) / nz;
    const uint32_t k0 = blockIdx.y * per;
    const uint32_t nk = k0 + per < nk_all ? per : (k0 < nk_all ? nk_all - k0 : 0u);
    const uint32_t n_kb = in_dim >> 5;
    const uint32_t batch_pad = (batch + 63u) & ~63u;
    const uint32_t nct = batch_pad >> 6;
    const uint32_t row_base = (blockIdx.x / nct) * 128u;
    const uint32_t col_base = (blockIdx.x % nct) * 64u;
    if (nz > 1u) y += (size_t)blockIdx.y * out_dim * batch;

    if (tid == 0) {
        #pragma unroll
        for (uint32_t s = 0; s < S; ++s) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 129;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bfull[s])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bdone[s])));
        }
    }
    __syncthreads();
    if (tid < 32)
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 128;"
                     ::"r"((uint32_t)__cvta_generic_to_shared(tmem_slot)));
    __syncthreads();
    const uint32_t tmem = tmem_slot[0];        // D: cols 0..63 (lanes 0..63)
    const uint32_t sf_base = tmem + 64u;       // SF ring: S x (4 SFA + 4 SFB)

    auto bar_wait = [&](uint64_t* bar, uint32_t parity) {
        const uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
        asm volatile("{\n\t.reg .pred p;\nW%=:\n\t"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                     "@!p bra W%=;\n\t}" ::"r"(a), "r"(parity));
    };
    auto tma_stage = [&](uint32_t kt, uint32_t s) {
        const uint32_t m = (uint32_t)__cvta_generic_to_shared(&bfull[s]);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 24576;" ::"r"(m));
        const uint32_t wd = (uint32_t)__cvta_generic_to_shared(wt + s * 16384u);
        const uint32_t yd = (uint32_t)__cvta_generic_to_shared(yt + s * 8192u);
        const int ck = (int)((k0 + kt) * 128u);
        asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                     " [%0], [%1, {%2, %3}], [%4];" ::"r"(wd), "l"(&wmap), "r"(ck),
                     "r"((int)row_base), "r"(m) : "memory");
        asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                     " [%0], [%1, {%2, %3}], [%4];" ::"r"(yd), "l"(&ymap), "r"(ck),
                     "r"((int)col_base), "r"(m) : "memory");
    };
    auto sf_stage = [&](uint32_t kt, uint32_t s) {
        const uint32_t kb0 = (k0 + kt) * 4u;
        unsigned char* base = sfs + s * 1024u;
        const uint32_t off = (tid % 32u) * 16u + (tid / 32u) * 4u;
        // SFA: all 128 rows real; SFB: rows 0..63 real, 64..127 pad
        const uint32_t rw = row_base + tid, rc = col_base + tid;
        pd_mma_cpa4p(base + off, wsc + (size_t)rw * n_kb + kb0,
                     rw < out_dim && kb0 + 4u <= n_kb);
        pd_mma_cpa4p(base + 512u + off, xsc + (size_t)rc * n_kb + kb0,
                     tid < 64u && rc < batch && kb0 + 4u <= n_kb);
        const uint32_t m = (uint32_t)__cvta_generic_to_shared(&bfull[s]);
        asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(m) : "memory");
    };

    // v3 pipeline: prefetch distance pdist (< S) decouples slab refill from
    // the freshly issued mma. v2 waited bdone[s] for the same iteration's mma
    // before prefetching (pf = kt + S lands on occupant kt), which drained the
    // tcgen05 pipe to one slab in flight and pinned the loop at the per-slab
    // mma+commit latency - measured flat in S (2.2-2.5 TB/s at every depth).
    // With pf = kt + pdist the reuse guard waits on occupant kt + pdist - S,
    // i.e. an mma issued S - pdist slabs ago: the pipe keeps S - pdist slabs
    // of mmas in flight and the loop runs at TMA/issue throughput instead.
    #pragma unroll 1
    for (uint32_t s = 0; s < S; ++s) {
        if (s < nk && s < pdist) {
            sf_stage(s, s);
            if (tid == 0) tma_stage(s, s);
        }
    }
    for (uint32_t kt = 0; kt < nk; ++kt) {
        const uint32_t s = kt % S;
        if (tid == 0) {
            bar_wait(&bfull[s], (kt / S) & 1u);
            const uint32_t v = (uint32_t)__cvta_generic_to_shared(sfs + s * 1024u) >> 4;
            const uint64_t da = ((uint64_t)(v & 0x3FFFu)) | ((uint64_t)1u << 16)
                              | ((uint64_t)8u << 32);
            const uint64_t db = ((uint64_t)((v + 32u) & 0x3FFFu)) | ((uint64_t)1u << 16)
                              | ((uint64_t)8u << 32);
            const uint32_t sfa_t = sf_base + s * 8u;
            const uint32_t sfb_t = sfa_t + 4u;
            asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                         ::"r"(sfa_t), "l"(da) : "memory");
            asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                         ::"r"(sfb_t), "l"(db) : "memory");
            const uint32_t w16 = (uint32_t)__cvta_generic_to_shared(wt + s * 16384u) >> 4;
            const uint32_t y16 = (uint32_t)__cvta_generic_to_shared(yt + s * 8192u) >> 4;
            #pragma unroll
            for (uint32_t kb = 0; kb < 4u; ++kb) {
                // SBO = 8-row core group stride: SW128 canonical (1024B = 64)
                // for both the 128-row W tile and the 64-row Y tile
                const uint64_t ad = pd_tc5_sdesc(w16 + kb * 2u);
                const uint64_t bd = pd_tc5_sdesc(y16 + kb * 2u);
                const uint32_t en = (kt > 0 || kb > 0) ? 1u : 0u;
                const uint32_t idesc = ((kb & 3u) << 4)
                    | ((64u >> 3) << 17) | (1u << 23) | ((128u >> 4) << 24)
                    | ((kb & 3u) << 29);
                asm volatile(
                    "{\n\t.reg .pred p;\n\t"
                    "setp.ne.b32 p, %6, 0;\n\t"
                    "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X"
                    " [%0], %1, %2, %3, [%4], [%5], p;\n\t}"
                    ::"r"(tmem), "l"(ad), "l"(bd), "r"(idesc),
                      "r"(sfa_t), "r"(sfb_t), "r"(en));
            }
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                ::"r"((uint32_t)__cvta_generic_to_shared(&bdone[s])));
        }
        const uint32_t pf = kt + pdist;
        if (pf < nk) {
            const uint32_t ps = pf % S;
            if (pf >= S) {
                // slab ps was last used by occupant pf - S; its bdone fire
                // number is (pf - S) / S, so that's the parity to observe
                bar_wait(&bdone[ps], ((pf - S) / S) & 1u);
            }
            sf_stage(pf, ps);
            if (tid == 0) tma_stage(pf, ps);
        }
    }
    // drain: the epilogue reads D from tmem, so the last occupant's commit
    // must have landed for every thread
    if (nk > 0) bar_wait(&bdone[(nk - 1u) % S], ((nk - 1u) / S) & 1u);
    __syncthreads();
    // epilogue: all 4 warps read D lanes 0..127, 2 col-chunks of 32; nk==0
    // z-tails store zeros (combine sums every plane)
    {
        const uint32_t warp = tid >> 5, lane = tid & 31u;
        {
            #pragma unroll
            for (uint32_t cc = 0; cc < 2u; ++cc) {
                uint32_t r[32];
                const uint32_t taddr = tmem + ((warp * 32u) << 16) + cc * 32u;
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
                    "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
                    "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
                    : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
                      "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),"=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15]),
                      "=r"(r[16]),"=r"(r[17]),"=r"(r[18]),"=r"(r[19]),"=r"(r[20]),"=r"(r[21]),"=r"(r[22]),"=r"(r[23]),
                      "=r"(r[24]),"=r"(r[25]),"=r"(r[26]),"=r"(r[27]),"=r"(r[28]),"=r"(r[29]),"=r"(r[30]),"=r"(r[31])
                    : "r"(taddr));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                const uint32_t row = row_base + warp * 32u + lane;
                if (row < out_dim) {
                    #pragma unroll
                    for (uint32_t j = 0; j < 32u; ++j) {
                        const uint32_t col = col_base + cc * 32u + j;
                        if (col < batch)
                            y[(size_t)col * out_dim + row] =
                                nk == 0u ? 0.0f : __uint_as_float(r[j]);
                    }
                }
            }
        }
    }
    __syncthreads();
    if (tid < 32)
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 128;" ::"r"(tmem));
#else
    (void)wmap; (void)ymap; (void)wsc; (void)xsc; (void)y;
    (void)in_dim; (void)out_dim; (void)batch; (void)nz; (void)pdist;
#endif
}


// ---- v4 decode GEMM: tile-image rowwise tc5p -------------------------------
// 's three findings, turned into a kernel class:
//   1. the "3.2 TB/s wall" was the L2-HIT path (microbench W fits the 126 MB
//      L2); HBM-cold streams 6.4-7.2 TB/s at these very grids;
//   2. TMA-2D's strided 128B row segments HALVE cold streaming, so W ships as
//      a PRE-SWIZZLED TILE-IMAGE plane - the SW128 smem image baked into gmem
//      at plane-build time, staged verbatim with 1D cp.async.bulk, consumed by
//      the same smem descriptors;
//   3. rowwise e4m3 (plain kind::f8f6f4) keeps the serial tcgen05 pipe free of
//      the two per-slab SF tcgen05.cp copies that cost tc5bs ~25-35 %.
// Scale fold happens once in the epilogue: y = D * wrs[row] * xrs[col], which
// commutes with the grid.y K-split (format-blind ks combine sums planes).

// one thread per 16B chunk: bake the TMA SW128 image (8-row core group =
// 1024 B, chunk column XORed with row&7) into contiguous 16 KB tiles laid
// (row_tile, k_slab)-major so a CTA's whole K walk is one linear stream
__global__ void pd_f8_tiles_repack_kernel(const unsigned char* __restrict__ src,
                                          unsigned char* __restrict__ dst,
                                          uint32_t in_dim, uint32_t out_dim) {
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t cpr = in_dim >> 4;               // 16B chunks per row
    if (idx >= (uint64_t)cpr * out_dim) return;
    const uint32_t row = (uint32_t)(idx / cpr), c16 = (uint32_t)(idx % cpr);
    const uint32_t kt = c16 >> 3, c = c16 & 7u;
    const uint32_t tr = row >> 7, r = row & 127u;
    const uint32_t nkt = in_dim >> 7;
    const uint32_t off16 = (r >> 3) * 64u + (r & 7u) * 8u + (c ^ (r & 7u));
    const uint4 v = *(const uint4*)(src + (size_t)row * in_dim + ((size_t)c16 << 4));
    *(uint4*)(dst + (((size_t)tr * nkt + kt) << 14) + ((size_t)off16 << 4)) = v;
}

// rowwise e4m3 decode GEMM over the tile-image plane. M=128 x N=64 per CTA,
// grid = (out/128, nz) with the usual grid.y K-split; batch <= 64 (single
// column tile - the decode band). Ring of S (16 KB W + 8 KB Y) slabs; W via
// 1D bulk from the tile stream, Y via strided TMA-2D (tiny, L2-served).
// pdist < S keeps mma slabs in flight; parities are computed per occupant.
// EF: L2::evict_first on the W stream. : production decode re-reads
// the whole weight set every step; the plane is PARTIALLY L2-resident from
// the previous step and the L2-HIT path serves at ~3.2 TB/s while cold
// misses stream 6.4-7.2 (finding #1). Evict-first keeps W lines out
// of L2 so every step streams at the cold rate. Y stays default (tiny,
// genuinely reused).
template <uint32_t S, bool EF = false, bool PDL = false, bool TS = false>
__global__ void __launch_bounds__(128) pd_f8row_gemm_tc5p_kt(
    const unsigned char* __restrict__ wtiles, const __grid_constant__ PdTmap ymap,
    const float* __restrict__ wrs, const float* __restrict__ xrs,
    float* __restrict__ y, uint32_t in_dim, uint32_t out_dim, uint32_t batch,
    uint32_t nz, uint32_t pdist, float* __restrict__ yfin, uint32_t* __restrict__ tctr,
    uint32_t l2pf) {
#if PD_TC5_OK
    extern __shared__ __align__(1024) unsigned char pd_tc5p_sh[];
    unsigned char* wt = pd_tc5p_sh;                       // S x 16 KB
    unsigned char* yt = pd_tc5p_sh + S * 16384u;          // S x 8 KB
    uint64_t* bfull = (uint64_t*)(yt + S * 8192u);
    uint64_t* bdone = bfull + S;
    __shared__ uint32_t tmem_slot[1];

    const uint32_t tid = threadIdx.x;
    unsigned long long ts_ent = 0, ts_fill = 0, ts_mma = 0, ts_epi = 0;
    bool ts_folder = false;
    if (TS && tid == 0) ts_ent = pd_ts_now();
    const uint32_t nk_all = (in_dim + 127u) / 128u;
    const uint32_t per = (nk_all + nz - 1u) / nz;
    const uint32_t k0 = blockIdx.y * per;
    const uint32_t nk = k0 + per < nk_all ? per : (k0 < nk_all ? nk_all - k0 : 0u);
    const uint32_t row_base = blockIdx.x * 128u;
    if (nz > 1u) y += (size_t)blockIdx.y * out_dim * batch;

    if (tid == 0) {
        #pragma unroll
        for (uint32_t s = 0; s < S; ++s) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bfull[s])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bdone[s])));
        }
    }
    __syncthreads();
    if (tid < 32)
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 64;"
                     ::"r"((uint32_t)__cvta_generic_to_shared(tmem_slot)));
    __syncthreads();
    const uint32_t tmem = tmem_slot[0];        // D: 64 f32 cols, 128 lanes

    auto bar_wait = [&](uint64_t* bar, uint32_t parity) {
        const uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
        asm volatile("{\n\t.reg .pred p;\nW%=:\n\t"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                     "@!p bra W%=;\n\t}" ::"r"(a), "r"(parity));
    };
    auto tma_stage = [&](uint32_t kt, uint32_t s) {
        const uint32_t m = (uint32_t)__cvta_generic_to_shared(&bfull[s]);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 24576;" ::"r"(m));
        const uint32_t wd = (uint32_t)__cvta_generic_to_shared(wt + s * 16384u);
        const unsigned char* wsrc = wtiles
            + (((size_t)blockIdx.x * nk_all + k0 + kt) << 14);
        if (EF) {
            uint64_t pol;
            asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
            asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                         " [%0], [%1], 16384, [%2], %3;" ::"r"(wd), "l"(wsrc), "r"(m), "l"(pol) : "memory");
        } else {
            asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                         " [%0], [%1], 16384, [%2];" ::"r"(wd), "l"(wsrc), "r"(m) : "memory");
        }
        const uint32_t yd = (uint32_t)__cvta_generic_to_shared(yt + s * 8192u);
        const int ck = (int)((k0 + kt) * 128u);
        asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                     " [%0], [%1, {%2, %3}], [%4];" ::"r"(yd), "l"(&ymap), "r"(ck),
                     "r"(0), "r"(m) : "memory");
    };

    // split issuer (the tc5v lesson applied here): thread 0's old
    // serial loop (wait-bfull -> mma -> wait-bdone -> tma) capped each SM
    // at ~31 GB/s of W stream - the chains serialize in one thread. Thread
    // 32 now owns staging (S-ago bdone guard + tma), thread 0 owns the mma
    // chain; they overlap (tc5p_stall: wo -14.5%, down -25.7%, 5.07 TB/s).
    // Numerics identical: same slabs, same mma order, same barriers.
    if (tid == 32) {
        if (PDL) {
            // PDL prologue: W halves are dependency-free (static weight
            // plane) - issue them before griddepcontrol.wait so they stream
            // during the predecessor kernel's execution. expect_tx without
            // arrive (16 KB W); the arrive rides the Y half post-wait so the
            // barrier still sees exactly one arrival + 24576 bytes.
            for (uint32_t s = 0; s < S && s < nk && s < pdist; ++s) {
                const uint32_t m = (uint32_t)__cvta_generic_to_shared(&bfull[s]);
                asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], 16384;" ::"r"(m));
                const uint32_t wd = (uint32_t)__cvta_generic_to_shared(wt + s * 16384u);
                const unsigned char* wsrc = wtiles
                    + (((size_t)blockIdx.x * nk_all + k0 + s) << 14);
                if (EF) {
                    uint64_t pol;
                    asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
                    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                 " [%0], [%1], 16384, [%2], %3;" ::"r"(wd), "l"(wsrc), "r"(m), "l"(pol) : "memory");
                } else {
                    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                                 " [%0], [%1], 16384, [%2];" ::"r"(wd), "l"(wsrc), "r"(m) : "memory");
                }
            }
            if (l2pf) {
                // whole-stream L2 prefetch in the early-launch window.
                // The PDL cascade launches this grid during the predecessor
                // band (attention, for wo), so pulling the CTA's full W range
                // into L2 now means the steady-state stages hit L2 instead of
                // pacing DRAM. The launcher enables it only when the GEMM's
                // aggregate W plane fits L2 (wo/qkv class; down/gu would
                // thrash). Fire-and-forget: no barrier, no completion.
                const uint32_t s0v = S < pdist ? S : pdist;
                const uint32_t s0 = s0v < nk ? s0v : nk;
                const unsigned char* p0 = wtiles
                    + (((size_t)blockIdx.x * nk_all + k0 + s0) << 14);
                const uint32_t total = (nk - s0) << 14;
                for (uint32_t off = 0; off < total; off += 65536u) {
                    const uint32_t sz = total - off < 65536u ? total - off : 65536u;
                    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;"
                                 ::"l"(p0 + off), "r"(sz) : "memory");
                }
            }
            asm volatile("griddepcontrol.wait;" ::: "memory");
            for (uint32_t s = 0; s < S && s < nk && s < pdist; ++s) {
                const uint32_t m = (uint32_t)__cvta_generic_to_shared(&bfull[s]);
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 8192;" ::"r"(m));
                const uint32_t yd = (uint32_t)__cvta_generic_to_shared(yt + s * 8192u);
                const int ck = (int)((k0 + s) * 128u);
                asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                             " [%0], [%1, {%2, %3}], [%4];" ::"r"(yd), "l"(&ymap), "r"(ck),
                             "r"(0), "r"(m) : "memory");
            }
        } else {
            for (uint32_t s = 0; s < S && s < nk && s < pdist; ++s) tma_stage(s, s);
        }
        for (uint32_t pf = pdist; pf < nk; ++pf) {
            const uint32_t ps = pf % S;
            if (pf >= S) bar_wait(&bdone[ps], ((pf - S) / S) & 1u);
            tma_stage(pf, ps);
        }
        if (PDL) asm volatile("griddepcontrol.launch_dependents;");
    } else if (tid == 0) {
        if (PDL) asm volatile("griddepcontrol.wait;" ::: "memory");
        for (uint32_t kt = 0; kt < nk; ++kt) {
            const uint32_t s = kt % S;
            bar_wait(&bfull[s], (kt / S) & 1u);
            if (TS && kt == 0) ts_fill = pd_ts_now();
            const uint32_t w16 = (uint32_t)__cvta_generic_to_shared(wt + s * 16384u) >> 4;
            const uint32_t y16 = (uint32_t)__cvta_generic_to_shared(yt + s * 8192u) >> 4;
            #pragma unroll
            for (uint32_t kb = 0; kb < 4u; ++kb) {
                const uint64_t ad = pd_tc5_sdesc(w16 + kb * 2u);
                const uint64_t bd = pd_tc5_sdesc(y16 + kb * 2u);
                const uint32_t en = (kt > 0 || kb > 0) ? 1u : 0u;
                const uint32_t idesc = (1u << 4) | ((64u >> 3) << 17) | ((128u >> 4) << 24);
                asm volatile(
                    "{\n\t.reg .pred p;\n\t"
                    "setp.ne.b32 p, %4, 0;\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t}"
                    ::"r"(tmem), "l"(ad), "l"(bd), "r"(idesc), "r"(en));
            }
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                ::"r"((uint32_t)__cvta_generic_to_shared(&bdone[s])));
        }
    }
    if (PDL && tid != 0 && tid != 32)
        asm volatile("griddepcontrol.wait;" ::: "memory");
    if (tid == 0 && nk > 0) bar_wait(&bdone[(nk - 1u) % S], ((nk - 1u) / S) & 1u);
    if (TS && tid == 0) ts_mma = pd_ts_now();
    __syncthreads();
    // epilogue: fold the rowwise scales exactly once per plane (the K-split
    // combine just sums planes, and sum-then-fold == fold-then-sum)
    {
        const uint32_t warp = tid >> 5, lane = tid & 31u;
        const uint32_t row = row_base + warp * 32u + lane;
        const float wsc = row < out_dim ? wrs[row] : 0.0f;
        #pragma unroll
        for (uint32_t cc = 0; cc < 2u; ++cc) {
            uint32_t r[32];
            const uint32_t taddr = tmem + ((warp * 32u) << 16) + cc * 32u;
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
                "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
                "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
                : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
                  "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),"=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15]),
                  "=r"(r[16]),"=r"(r[17]),"=r"(r[18]),"=r"(r[19]),"=r"(r[20]),"=r"(r[21]),"=r"(r[22]),"=r"(r[23]),
                  "=r"(r[24]),"=r"(r[25]),"=r"(r[26]),"=r"(r[27]),"=r"(r[28]),"=r"(r[29]),"=r"(r[30]),"=r"(r[31])
                : "r"(taddr));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            if (row < out_dim) {
                #pragma unroll
                for (uint32_t j = 0; j < 32u; ++j) {
                    const uint32_t col = cc * 32u + j;
                    if (col < batch)
                        y[(size_t)col * out_dim + row] = nk == 0u ? 0.0f
                            : __uint_as_float(r[j]) * wsc * xrs[col];
                }
            }
        }
    }
    if (TS && tid == 0) ts_epi = pd_ts_now();
    __syncthreads();
    if (tid < 32)
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 64;" ::"r"(tmem));
    // last-CTA K-split fold: the CTA whose partial lands last for this tile
    // sums the nz planes in FIXED ascending z (identical order to
    // pd_q8_0_gemm_mma_ks_combine -> bit-equal) and writes final y - the
    // separate combine launch disappears and the fold overlaps other tiles'
    // mma tails. Counter protocol: arrivals count up from 0; the folder
    // resets its slot so the buffer never needs a per-launch memset.
    if (nz > 1u && tctr) {
        // every thread wrote partials: fence all of them, then the barrier
        // carries the fences into tid 0's atomic (fence->sync->signal edge)
        __threadfence();
        __syncthreads();
        __shared__ bool pd_fold_last;
        if (tid == 0)
            pd_fold_last = atomicAdd(&tctr[blockIdx.x], 1u) == nz - 1u;
        __syncthreads();
        if (pd_fold_last) {
            if (TS) ts_folder = true;
            const size_t np = (size_t)out_dim * batch;
            const float* part = y - (size_t)blockIdx.y * np;   // plane 0 base
            const uint32_t row = row_base + tid;               // 128 rows/tile
            if (row < out_dim) {
                // 8 independent column chains per z-step keep ~8 loads in
                // flight per thread - the single-chain version serialized on
                // L2 latency and tripled the kernel tail at b=64
                uint32_t col = 0;
                for (; col + 8u <= batch; col += 8u) {
                    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                    for (uint32_t z = 0; z < nz; ++z) {
                        const float* p = part + z * np + (size_t)col * out_dim + row;
                        #pragma unroll
                        for (uint32_t c = 0; c < 8u; ++c)
                            acc[c] += p[(size_t)c * out_dim];
                    }
                    #pragma unroll
                    for (uint32_t c = 0; c < 8u; ++c)
                        yfin[(size_t)(col + c) * out_dim + row] = acc[c];
                }
                for (; col < batch; ++col) {
                    float acc = 0.0f;
                    for (uint32_t z = 0; z < nz; ++z)
                        acc += part[z * np + (size_t)col * out_dim + row];
                    yfin[(size_t)col * out_dim + row] = acc;
                }
            }
            if (tid == 0) tctr[blockIdx.x] = 0u;
        }
    }
    if (TS && tid == 0 && pd_tc5p_ts) {
        unsigned long long* o = pd_tc5p_ts
            + (size_t)(blockIdx.y * gridDim.x + blockIdx.x) * 24u;
        o[0] = pd_ts_smid(); o[1] = ts_ent; o[2] = ts_fill; o[3] = ts_mma;
        o[4] = ts_epi; o[5] = pd_ts_now(); o[6] = ts_folder ? 1u : 0u;
        o[7] = ((unsigned long long)blockIdx.y << 32) | blockIdx.x;
        #pragma unroll
        for (uint32_t q = 0; q < 12u; ++q) o[8u + q] = 0u;
    }
#else
    (void)wtiles; (void)ymap; (void)wrs; (void)xrs; (void)y;
    (void)in_dim; (void)out_dim; (void)batch; (void)nz; (void)pdist;
    (void)yfin; (void)tctr;
#endif
}

// M2 rowwise decode GEMM: two row-tiles per CTA. The probe
// chain proved the tc5p band's remaining loss is LAUNCH GRANULARITY (10-24
// slab streams vs the ~3-6us fill/drain/wave ramp; DRAM, Y, smem contention
// and mma co-execution all measured innocent) - so halve the CTAs and
// double each stream: slab = 2x16KB W (both tiles' k-tile, two bulk copies)
// + one shared 8KB Y (halves Y traffic), 8 mmas/slab into two tmem windows
// (alloc 128: cols 0-63 / 64-127), two-window epilogue. Same nz as the ::1
// form -> identical K-split partials, combine order and per-tile mma order:
// BIT-IDENTICAL per row (harness: wo 3.12 -> 3.78 TB/s +21%, down +5%).
// No-fold path only (batch > 8); the ksfold b<=8 regime keeps the ::1 form.
template <uint32_t S, bool EF = false, bool PDL = false, bool TS = false>
__global__ void __launch_bounds__(128) pd_f8row_gemm_tc5p_m2_kt(
    const unsigned char* __restrict__ wtiles, const __grid_constant__ PdTmap ymap,
    const float* __restrict__ wrs, const float* __restrict__ xrs,
    float* __restrict__ y, uint32_t in_dim, uint32_t out_dim, uint32_t batch,
    uint32_t nz, uint32_t pdist, uint32_t l2pf) {
#if PD_TC5_OK
    extern __shared__ __align__(1024) unsigned char pd_tc5m2_sh[];
    unsigned char* wt = pd_tc5m2_sh;                      // S x 2 x 16 KB
    unsigned char* yt = pd_tc5m2_sh + S * 32768u;         // S x 8 KB
    uint64_t* bfull = (uint64_t*)(yt + S * 8192u);
    uint64_t* bdone = bfull + S;
    __shared__ uint32_t tmem_slot[1];

    const uint32_t tid = threadIdx.x;
    unsigned long long ts_ent = 0, ts_fill = 0, ts_mma = 0, ts_epi = 0;
    unsigned long long ts_c[12] = {};   // per epi chunk: pre-ld/post-wait/post-store
    if (TS && tid == 0) ts_ent = pd_ts_now();
    const uint32_t nk_all = (in_dim + 127u) / 128u;
    const uint32_t per = (nk_all + nz - 1u) / nz;
    const uint32_t k0 = blockIdx.y * per;
    const uint32_t nk = k0 + per < nk_all ? per : (k0 < nk_all ? nk_all - k0 : 0u);
    if (nz > 1u) y += (size_t)blockIdx.y * out_dim * batch;

    if (tid == 0) {
        #pragma unroll
        for (uint32_t s = 0; s < S; ++s) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bfull[s])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bdone[s])));
        }
    }
    __syncthreads();
    if (tid < 32)
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 128;"
                     ::"r"((uint32_t)__cvta_generic_to_shared(tmem_slot)));
    __syncthreads();
    const uint32_t tmem = tmem_slot[0];      // two D windows: +0 / +64
    // the epilogue's wrs/xrs first-touch misses serialize ~0.5us
    // per half at stream end (chunk probe) - issue them at ENTRY so the
    // lines are L1/L2-hot ~6us later. Same values, bit-identical.
    const uint32_t pre_lane = tid & 31u, pre_warp = tid >> 5;
    const uint32_t pre_r0 = blockIdx.x * 256u + pre_warp * 32u + pre_lane;
    const float pre_w0 = pre_r0 < out_dim ? wrs[pre_r0] : 0.0f;
    const float pre_w1 = pre_r0 + 128u < out_dim ? wrs[pre_r0 + 128u] : 0.0f;
    if (tid < 2u) (void)__ldg(&xrs[tid * 32u]);   // touch both xrs lines

    auto bar_wait = [&](uint64_t* bar, uint32_t parity) {
        const uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
        asm volatile("{\n\t.reg .pred p;\nW%=:\n\t"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                     "@!p bra W%=;\n\t}" ::"r"(a), "r"(parity));
    };
    auto stage_w = [&](uint32_t kt, uint32_t s, uint32_t m) {
        const uint32_t wd = (uint32_t)__cvta_generic_to_shared(wt + s * 32768u);
        #pragma unroll
        for (uint32_t half = 0; half < 2u; ++half) {
            const unsigned char* wsrc = wtiles
                + ((((size_t)blockIdx.x * 2u + half) * nk_all + k0 + kt) << 14);
            if (EF) {
                uint64_t pol;
                asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
                asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                             " [%0], [%1], 16384, [%2], %3;"
                             ::"r"(wd + half * 16384u), "l"(wsrc), "r"(m), "l"(pol) : "memory");
            } else {
                asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                             " [%0], [%1], 16384, [%2];"
                             ::"r"(wd + half * 16384u), "l"(wsrc), "r"(m) : "memory");
            }
        }
    };
    auto stage_y = [&](uint32_t kt, uint32_t s, uint32_t m) {
        const uint32_t yd = (uint32_t)__cvta_generic_to_shared(yt + s * 8192u);
        const int ck = (int)((k0 + kt) * 128u);
        asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                     " [%0], [%1, {%2, %3}], [%4];" ::"r"(yd), "l"(&ymap), "r"(ck),
                     "r"(0), "r"(m) : "memory");
    };
    auto tma_stage = [&](uint32_t kt, uint32_t s) {
        const uint32_t m = (uint32_t)__cvta_generic_to_shared(&bfull[s]);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 40960;" ::"r"(m));
        stage_w(kt, s, m);
        stage_y(kt, s, m);
    };

    if (tid == 32) {
        if (PDL) {
            // dep-free W prologue before griddepcontrol.wait (the ::1 form's
            // split-barrier trick: expect without arrive for the 32KB W pair,
            // the arrive rides the Y half post-wait)
            for (uint32_t s = 0; s < S && s < nk && s < pdist; ++s) {
                const uint32_t m = (uint32_t)__cvta_generic_to_shared(&bfull[s]);
                asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], 32768;" ::"r"(m));
                stage_w(s, s, m);
            }
            if (l2pf) {
                // whole-stream L2 prefetch in the early-launch window.
                // The PDL cascade launches this grid during the predecessor
                // band (attention, for wo), so pulling the CTA's full W range
                // into L2 now means the steady-state stages hit L2 instead of
                // pacing DRAM. The launcher enables it only when the GEMM's
                // aggregate W plane fits L2 (wo/qkv class; down/gu would
                // thrash). Fire-and-forget: no barrier, no completion.
                const uint32_t s0v = S < pdist ? S : pdist;
                const uint32_t s0 = s0v < nk ? s0v : nk;
                #pragma unroll
                for (uint32_t half = 0; half < 2u; ++half) {
                    const unsigned char* p0 = wtiles
                        + ((((size_t)blockIdx.x * 2u + half) * nk_all + k0 + s0) << 14);
                    const uint32_t total = (nk - s0) << 14;
                    for (uint32_t off = 0; off < total; off += 65536u) {
                        const uint32_t sz = total - off < 65536u ? total - off : 65536u;
                        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;"
                                     ::"l"(p0 + off), "r"(sz) : "memory");
                    }
                }
            }
            asm volatile("griddepcontrol.wait;" ::: "memory");
            for (uint32_t s = 0; s < S && s < nk && s < pdist; ++s) {
                const uint32_t m = (uint32_t)__cvta_generic_to_shared(&bfull[s]);
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 8192;" ::"r"(m));
                stage_y(s, s, m);
            }
        } else {
            for (uint32_t s = 0; s < S && s < nk && s < pdist; ++s) tma_stage(s, s);
        }
        for (uint32_t pf = pdist; pf < nk; ++pf) {
            const uint32_t ps = pf % S;
            if (pf >= S) bar_wait(&bdone[ps], ((pf - S) / S) & 1u);
            tma_stage(pf, ps);
        }
        if (PDL) asm volatile("griddepcontrol.launch_dependents;");
    } else if (tid == 0) {
        if (PDL) asm volatile("griddepcontrol.wait;" ::: "memory");
        for (uint32_t kt = 0; kt < nk; ++kt) {
            const uint32_t s = kt % S;
            bar_wait(&bfull[s], (kt / S) & 1u);
            if (TS && kt == 0) ts_fill = pd_ts_now();
            const uint32_t w16 = (uint32_t)__cvta_generic_to_shared(wt + s * 32768u) >> 4;
            const uint32_t y16 = (uint32_t)__cvta_generic_to_shared(yt + s * 8192u) >> 4;
            #pragma unroll
            for (uint32_t half = 0; half < 2u; ++half)
                #pragma unroll
                for (uint32_t kb = 0; kb < 4u; ++kb) {
                    const uint64_t ad = pd_tc5_sdesc(w16 + half * 1024u + kb * 2u);
                    const uint64_t bd = pd_tc5_sdesc(y16 + kb * 2u);
                    const uint32_t en = (kt > 0 || kb > 0) ? 1u : 0u;
                    const uint32_t idesc = (1u << 4) | ((64u >> 3) << 17) | ((128u >> 4) << 24);
                    asm volatile(
                        "{\n\t.reg .pred p;\n\t"
                        "setp.ne.b32 p, %4, 0;\n\t"
                        "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t}"
                        ::"r"(tmem + half * 64u), "l"(ad), "l"(bd), "r"(idesc), "r"(en));
                }
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                ::"r"((uint32_t)__cvta_generic_to_shared(&bdone[s])));
        }
    }
    if (PDL && tid != 0 && tid != 32)
        asm volatile("griddepcontrol.wait;" ::: "memory");
    if (tid == 0 && nk > 0) bar_wait(&bdone[(nk - 1u) % S], ((nk - 1u) / S) & 1u);
    if (TS && tid == 0) ts_mma = pd_ts_now();
    __syncthreads();
    // two-window epilogue, ld-pipelined: the chunk probe showed
    // wait::ld retires instantly and the ~0.5us tcgen05.ld DATA latency
    // surfaces at first register use - chunk-serialized, 4x exposed. Issue
    // all four lds first (r[4][32] fits the 512-reg budget), one wait, then
    // store: 1x exposed latency, bit-identical values and store order.
    {
        const uint32_t warp = tid >> 5, lane = tid & 31u;
        uint32_t r[4][32];
        if (TS && tid == 0) ts_c[0] = pd_ts_now();
        #pragma unroll
        for (uint32_t ch = 0; ch < 4u; ++ch) {
            const uint32_t half = ch >> 1, cc = ch & 1u;
            const uint32_t taddr = tmem + half * 64u + ((warp * 32u) << 16) + cc * 32u;
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
                "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
                "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
                : "=r"(r[ch][0]),"=r"(r[ch][1]),"=r"(r[ch][2]),"=r"(r[ch][3]),"=r"(r[ch][4]),"=r"(r[ch][5]),"=r"(r[ch][6]),"=r"(r[ch][7]),
                  "=r"(r[ch][8]),"=r"(r[ch][9]),"=r"(r[ch][10]),"=r"(r[ch][11]),"=r"(r[ch][12]),"=r"(r[ch][13]),"=r"(r[ch][14]),"=r"(r[ch][15]),
                  "=r"(r[ch][16]),"=r"(r[ch][17]),"=r"(r[ch][18]),"=r"(r[ch][19]),"=r"(r[ch][20]),"=r"(r[ch][21]),"=r"(r[ch][22]),"=r"(r[ch][23]),
                  "=r"(r[ch][24]),"=r"(r[ch][25]),"=r"(r[ch][26]),"=r"(r[ch][27]),"=r"(r[ch][28]),"=r"(r[ch][29]),"=r"(r[ch][30]),"=r"(r[ch][31])
                : "r"(taddr));
        }
        asm volatile("tcgen05.wait::ld.sync.aligned;");
        if (TS && tid == 0) ts_c[1] = pd_ts_now();
        #pragma unroll
        for (uint32_t half = 0; half < 2u; ++half) {
            const uint32_t row = (blockIdx.x * 2u + half) * 128u + warp * 32u + lane;
            const float wsc = half == 0u ? pre_w0 : pre_w1;
            #pragma unroll
            for (uint32_t cc = 0; cc < 2u; ++cc) {
                const uint32_t ch = half * 2u + cc;
                if (row < out_dim) {
                    #pragma unroll
                    for (uint32_t j = 0; j < 32u; ++j) {
                        const uint32_t col = cc * 32u + j;
                        if (col < batch)
                            y[(size_t)col * out_dim + row] = nk == 0u ? 0.0f
                                : __uint_as_float(r[ch][j]) * wsc * xrs[col];
                    }
                }
                if (TS && tid == 0) ts_c[3u * ch + 2u] = pd_ts_now();
            }
        }
    }
    if (TS && tid == 0) ts_epi = pd_ts_now();
    __syncthreads();
    if (tid < 32)
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 128;" ::"r"(tmem));
    if (TS && tid == 0 && pd_tc5p_ts) {
        unsigned long long* o = pd_tc5p_ts
            + (size_t)(blockIdx.y * gridDim.x + blockIdx.x) * 24u;
        o[0] = pd_ts_smid(); o[1] = ts_ent; o[2] = ts_fill; o[3] = ts_mma;
        o[4] = ts_epi; o[5] = pd_ts_now(); o[6] = 0u;
        o[7] = ((unsigned long long)blockIdx.y << 32) | blockIdx.x;
        #pragma unroll
        for (uint32_t q = 0; q < 12u; ++q) o[8u + q] = ts_c[q];
    }
#else
    (void)wtiles; (void)ymap; (void)wrs; (void)xrs; (void)y;
    (void)in_dim; (void)out_dim; (void)batch; (void)nz; (void)pdist;
#endif
}

// wo-c2col: cluster-2 COLUMN-split rowwise e4m3 GEMM for the
// no-fold band (batch > 8) small-tile shapes (wo/down, 42 tiles). The
// probe showed the M2 split-K route's mma stream already runs at the DRAM
// floor - its 2.14 TB/s(w) wall is pure split-K OVERHEAD: 9.6 MB of partial
// planes, a 7.45 us/CTA epilogue, and the combine pass. So: don't split K.
// One row-tile per CLUSTER; the two CTAs split the BATCH cols (rank r takes
// cols r*32..). W and Y are staged once per cluster: rank 0 issues 1D-bulk
// (W) and tensor-2d (Y) MULTICAST copies feeding both CTAs' rings on a
// single DRAM fetch - L2-pairing is not an alternative on this die (the
// L2-HIT path serves ~3.2 TB/s vs 6.4-7.2 cold, /23). Full-K streams
// (wo: 64 slabs), no partials, no combine; the epilogue folds the scales
// and writes final y. Ring pacing: each CTA re-arms its own bfull
// (arrive.expect_tx) before the producer may issue the slot; rank 1 signals
// readiness through a cluster-mapped bpeer arrive (the pf5g-c2 pattern);
// the producer waits its own bdone AND bpeer. Numeric class: the full-K
// accumulation reorders vs the z-partial combine (coherence-gate class).
// __cluster_dims__ guard: same rule as pf5g-c2 - compiled out below sm_90.
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 900)
template <uint32_t S, bool EF = false, bool TS = false>
__global__ void __launch_bounds__(128, 1) __cluster_dims__(2, 1, 1)
pd_f8row_gemm_c2col_kt(
    const unsigned char* __restrict__ wtiles, const __grid_constant__ PdTmap ymap,
    const float* __restrict__ wrs, const float* __restrict__ xrs,
    float* __restrict__ y, uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
#if PD_TC5_OK
    extern __shared__ __align__(1024) unsigned char pd_c2col_sh[];
    unsigned char* wt = pd_c2col_sh;                      // S x 16 KB
    unsigned char* yt = pd_c2col_sh + S * 16384u;         // S x 8 KB
    uint64_t* bfull = (uint64_t*)(yt + S * 8192u);        // [S]
    uint64_t* bdone = bfull + S;                          // [S]
    uint64_t* bpeer = bdone + S;                          // [S] (rank 0's live)
    __shared__ uint32_t tmem_slot[1];

    const uint32_t tid = threadIdx.x;
    unsigned long long ts_ent = 0, ts_fill = 0, ts_mma = 0, ts_epi = 0;
    if (TS && tid == 0) ts_ent = pd_ts_now();
    uint32_t crank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(crank));
    const uint32_t nk = (in_dim + 127u) / 128u;
    const uint32_t tile = blockIdx.x >> 1;                // cluster id

    if (tid == 0) {
        #pragma unroll
        for (uint32_t s = 0; s < S; ++s) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bfull[s])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bdone[s])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bpeer[s])));
        }
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();
    asm volatile("barrier.cluster.arrive;");
    asm volatile("barrier.cluster.wait;");
    if (tid < 32)
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;"
                     ::"r"((uint32_t)__cvta_generic_to_shared(tmem_slot)));
    __syncthreads();
    const uint32_t tmem = tmem_slot[0];                   // D: 128 lanes x 32 cols

    auto bar_wait = [&](uint64_t* bar, uint32_t parity) {
        const uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
        asm volatile("{\n\t.reg .pred p;\nW%=:\n\t"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                     "@!p bra W%=;\n\t}" ::"r"(a), "r"(parity));
    };
    auto peer_addr = [&](void* p) -> uint32_t {
        const uint32_t pa0 = (uint32_t)__cvta_generic_to_shared(p);
        uint32_t pa;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                     : "=r"(pa) : "r"(pa0), "r"(crank ^ 1u));
        return pa;
    };

    if (crank == 0u && tid == 32) {
        // producer: one multicast W + Y per slot feeds both CTAs
        for (uint32_t kt = 0; kt < nk; ++kt) {
            const uint32_t s = kt % S;
            const uint32_t wrap = kt / S;
            if (kt >= S) bar_wait(&bdone[s], (wrap - 1u) & 1u);
            bar_wait(&bpeer[s], wrap & 1u);   // rank 1: slot free + bfull re-armed
            const uint32_t m = (uint32_t)__cvta_generic_to_shared(&bfull[s]);
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 24576;" ::"r"(m));
            const uint32_t wd = (uint32_t)__cvta_generic_to_shared(wt + s * 16384u);
            const unsigned char* wsrc = wtiles + (((size_t)tile * nk + kt) << 14);
            if (EF) {
                uint64_t pol;
                asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
                asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
                             ".multicast::cluster.L2::cache_hint"
                             " [%0], [%1], 16384, [%2], %3, %4;"
                             ::"r"(wd), "l"(wsrc), "r"(m), "h"((unsigned short)3u), "l"(pol) : "memory");
            } else {
                asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
                             ".multicast::cluster"
                             " [%0], [%1], 16384, [%2], %3;"
                             ::"r"(wd), "l"(wsrc), "r"(m), "h"((unsigned short)3u) : "memory");
            }
            const uint32_t yd = (uint32_t)__cvta_generic_to_shared(yt + s * 8192u);
            const int ck = (int)(kt * 128u);
            asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
                         ".multicast::cluster"
                         " [%0], [%1, {%2, %3}], [%4], %5;"
                         ::"r"(yd), "l"(&ymap), "r"(ck), "r"(0), "r"(m),
                         "h"((unsigned short)3u) : "memory");
        }
    } else if (crank == 1u && tid == 32) {
        // slot-release loop: re-arm own bfull for the incoming multicast,
        // then hand the slot to the producer over the cluster
        for (uint32_t kt = 0; kt < nk; ++kt) {
            const uint32_t s = kt % S;
            const uint32_t wrap = kt / S;
            if (kt >= S) bar_wait(&bdone[s], (wrap - 1u) & 1u);
            const uint32_t m = (uint32_t)__cvta_generic_to_shared(&bfull[s]);
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 24576;" ::"r"(m));
            asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];"
                         ::"r"(peer_addr(&bpeer[s])) : "memory");
        }
    } else if (tid == 0) {
        for (uint32_t kt = 0; kt < nk; ++kt) {
            const uint32_t s = kt % S;
            bar_wait(&bfull[s], (kt / S) & 1u);
            if (TS && kt == 0) ts_fill = pd_ts_now();
            const uint32_t w16 = (uint32_t)__cvta_generic_to_shared(wt + s * 16384u) >> 4;
            const uint32_t y16 = (uint32_t)__cvta_generic_to_shared(yt + s * 8192u) >> 4;
            #pragma unroll
            for (uint32_t kb = 0; kb < 4u; ++kb) {
                const uint64_t ad = pd_tc5_sdesc(w16 + kb * 2u);
                // B = this rank's 32-col half of the Y slab (row 32 = +4 KB)
                const uint64_t bd = pd_tc5_sdesc(y16 + crank * 256u + kb * 2u);
                const uint32_t en = (kt > 0 || kb > 0) ? 1u : 0u;
                const uint32_t idesc = (1u << 4) | ((32u >> 3) << 17) | ((128u >> 4) << 24);
                asm volatile(
                    "{\n\t.reg .pred p;\n\t"
                    "setp.ne.b32 p, %4, 0;\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t}"
                    ::"r"(tmem), "l"(ad), "l"(bd), "r"(idesc), "r"(en));
            }
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                ::"r"((uint32_t)__cvta_generic_to_shared(&bdone[s])));
        }
    }
    if (tid == 0 && nk > 0) bar_wait(&bdone[(nk - 1u) % S], ((nk - 1u) / S) & 1u);
    if (TS && tid == 0) ts_mma = pd_ts_now();
    __syncthreads();
    // epilogue: fold the scales, write FINAL y (no partials, no combine)
    {
        const uint32_t warp = tid >> 5, lane = tid & 31u;
        const uint32_t row = tile * 128u + warp * 32u + lane;
        const float wsc = row < out_dim ? wrs[row] : 0.0f;
        uint32_t r[32];
        const uint32_t taddr = tmem + ((warp * 32u) << 16);
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
            "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
            : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
              "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),"=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15]),
              "=r"(r[16]),"=r"(r[17]),"=r"(r[18]),"=r"(r[19]),"=r"(r[20]),"=r"(r[21]),"=r"(r[22]),"=r"(r[23]),
              "=r"(r[24]),"=r"(r[25]),"=r"(r[26]),"=r"(r[27]),"=r"(r[28]),"=r"(r[29]),"=r"(r[30]),"=r"(r[31])
            : "r"(taddr));
        asm volatile("tcgen05.wait::ld.sync.aligned;");
        if (row < out_dim) {
            #pragma unroll
            for (uint32_t j = 0; j < 32u; ++j) {
                const uint32_t col = crank * 32u + j;
                if (col < batch)
                    y[(size_t)col * out_dim + row] =
                        __uint_as_float(r[j]) * wsc * xrs[col];
            }
        }
    }
    if (TS && tid == 0) ts_epi = pd_ts_now();
    __syncthreads();
    if (tid < 32)
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;" ::"r"(tmem));
    // cluster teardown: neither CTA may exit while its peer might still
    // arrive on our barriers / receive our multicasts
    asm volatile("barrier.cluster.arrive;");
    asm volatile("barrier.cluster.wait;");
    if (TS && tid == 0 && pd_tc5p_ts) {
        unsigned long long* o = pd_tc5p_ts + (size_t)blockIdx.x * 24u;
        o[0] = pd_ts_smid(); o[1] = ts_ent; o[2] = ts_fill; o[3] = ts_mma;
        o[4] = ts_epi; o[5] = pd_ts_now(); o[6] = 0u;
        o[7] = ((unsigned long long)crank << 32) | tile;
        #pragma unroll
        for (uint32_t q = 0; q < 12u; ++q) o[8u + q] = 0u;
    }
#else
    (void)wtiles; (void)ymap; (void)wrs; (void)xrs; (void)y;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}
#endif  // __cluster_dims__ guard (host pass || sm_90+)

// tc5q work counter: allocated at plane-build time because the decode step
// is CUDA-graph-captured and cudaMalloc is illegal inside capture (error
// 900). The graph replays memsetAsync(0) + kernel, so reuse is capture-safe.
static uint32_t* pd_tc5q_ctr(bool make = false) {
    static uint32_t* ctr = nullptr;
    if (make && !ctr) {
        if (cudaMalloc(&ctr, 4) != cudaSuccess) ctr = nullptr;
        else cudaMemset(ctr, 0, 4);   // kernel resets it thereafter
    }
    return ctr;
}

// ---- two-stage row-quant widening -----------------------------------------
// The per-row quant kernels run one block per row (r<=64 -> 32 blocks on a
// 148-SM die, ~10us for 1-11 MB - pure starvation, and ~1.2 ms/tick for the
// pair). Two-stage: stage A computes per-CHUNK abs-maxes at
// rows x C parallelism; stage B blocks each read the row's C partials,
// derive the exponent LOCALLY (no atomics, no spin - deadlock-free under
// any co-residency), and transform their own chunk. Max is
// partition-invariant, so the exponent and every quantized byte are
// BIT-IDENTICAL to the single-block walk.
// TI: f16 input plane (pd_ld4f exact expand).
//
// Even widened this stays two kernels with a global barrier between them,
// and that is STRUCTURAL rather than a tuning miss. Quantizing against a
// PER-128-GROUP scale instead (DeepGEMM's "1d1d" - 1D scales on both
// operands, which is what SGLang's per_token_group_quant_flat does) lets
// every block reduce its own group locally, so the whole thing is one pass
// with no cross-block dependency. Our f8t plane carries a PER-ROW
// activation scale, so the row max must be complete before any element can
// be encoded - hence the barrier, and hence the widening could only spread
// the work rather than remove the pass.
// Closing it means either fusing the pair for the decode band (rows are 5120-
// 17408 elements at b<=8; one CTA can carry max+encode across a single
// __syncthreads) or moving the f8t activation class to group scales. With the
// projection GEMMs this pair feeds already at the roof, the quantize pass is
// worth more than any further GEMM work.
template <typename TI = float>
__global__ void pd_rowmax_part_kernel(const TI* __restrict__ x,
                                      float* __restrict__ parts,
                                      uint32_t n_dim, uint32_t nzp) {
    PD_PDL_ARM();  // no-op below sm_90 (multi-arch fatbin: raw asm breaks ptxas)
    const uint32_t row = blockIdx.x, c = blockIdx.y, C = gridDim.y;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    const TI* xr = x + (size_t)row * n_dim;
    const size_t np = (size_t)gridDim.x * n_dim;
    const uint32_t n4 = n_dim >> 2;
    const uint32_t n4c = (n4 + C - 1u) / C;
    const uint32_t i0 = c * n4c, i1 = min(n4, i0 + n4c);
    __shared__ float wmax[32];
    float a = 0.0f;
    for (uint32_t i = i0 + tid; i < i1; i += nth) {
        float4 v = pd_ld4f(xr + (size_t)i * 4u);
        for (uint32_t z = 1; z < nzp; ++z) {
            const float4 pz = pd_ld4f(xr + (size_t)z * np + (size_t)i * 4u);
            v.x += pz.x; v.y += pz.y; v.z += pz.z; v.w += pz.w;
        }
        a = fmaxf(a, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)),
                           fmaxf(fabsf(v.z), fabsf(v.w))));
    }
    for (uint32_t sh = 16; sh > 0; sh >>= 1)
        a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, sh));
    if ((tid & 31u) == 0) wmax[tid >> 5] = a;
    __syncthreads();
    if (tid == 0) {
        float m = 0.0f;
        for (uint32_t w = 0; w < ((nth + 31u) >> 5); ++w) m = fmaxf(m, wmax[w]);
        parts[(size_t)row * C + c] = m;
    }
}

__device__ __forceinline__ int pd_rowq_exp_from_parts(const float* parts,
                                                      uint32_t row, uint32_t C) {
    float m = 0.0f;
    for (uint32_t c = 0; c < C; ++c) m = fmaxf(m, parts[(size_t)row * C + c]);
    int e = 0;
    if (m > 0.0f) {
        int ex;
        float fr = frexpf(m, &ex);
        e = ex - 9 + (fr > 0.875f ? 1 : 0);
    }
    return e;
}

// TI: f16 input plane for the attention streams (pd_ld4f exact expand).
template <typename TI = float>
__global__ void pd_quantize_e4m3_row2_kernel(const TI* __restrict__ x,
                                             unsigned char* __restrict__ q,
                                             float* __restrict__ rscale,
                                             uint32_t n_dim,
                                             const float* __restrict__ parts,
                                             uint32_t nzp) {
    PD_PDL_ARM();  // no-op below sm_90 (multi-arch fatbin: raw asm breaks ptxas)
    const uint32_t row = blockIdx.x, c = blockIdx.y, C = gridDim.y;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    const int e = pd_rowq_exp_from_parts(parts, row, C);
    if (c == 0 && tid == 0) rscale[row] = ldexpf(1.0f, e);
    const float inv = ldexpf(1.0f, -e);
    const TI* xr = x + (size_t)row * n_dim;
    const size_t np = (size_t)gridDim.x * n_dim;
    unsigned char* qr = q + (size_t)row * n_dim;
    const uint32_t n4 = n_dim >> 2;
    const uint32_t n4c = (n4 + C - 1u) / C;
    const uint32_t i0 = c * n4c, i1 = min(n4, i0 + n4c);
    for (uint32_t i = i0 + tid; i < i1; i += nth) {
        float4 v = pd_ld4f(xr + (size_t)i * 4u);
        for (uint32_t z = 1; z < nzp; ++z) {
            const float4 pz = pd_ld4f(xr + (size_t)z * np + (size_t)i * 4u);
            v.x += pz.x; v.y += pz.y; v.z += pz.z; v.w += pz.w;
        }
        uchar4 o;
        o.x = __nv_fp8_e4m3(v.x * inv).__x;
        o.y = __nv_fp8_e4m3(v.y * inv).__x;
        o.z = __nv_fp8_e4m3(v.z * inv).__x;
        o.w = __nv_fp8_e4m3(v.w * inv).__x;
        *(uchar4*)(qr + (size_t)i * 4u) = o;
    }
}

// scratch for the row-partial maxes: alloc'd outside graph capture (the
// decode step is captured; cudaMalloc is illegal inside). 64 rows x 8
// chunks covers every widened shape - wider rows keep the single-block path.
static float* pd_rowq_scr(bool make = false) {
    static float* scr = nullptr;
    if (make && !scr) {
        if (cudaMalloc(&scr, 64 * 8 * sizeof(float)) != cudaSuccess) scr = nullptr;
        else cudaMemset(scr, 0, 64 * 8 * sizeof(float));
    }
    return scr;
}

// chunk count: ~2 blocks/SM target, capped by the scratch layout (C<=8) and
// only when the single-block grid would starve the die
static uint32_t pd_rowq_chunks(uint32_t rows) {
    static int no2 = -1;
    if (no2 < 0) no2 = pd_env("PADDOCK_NO_ROWQ2") ? 1 : 0;
    if (no2 || rows > 48u || !pd_rowq_scr()) return 1u;
    static int nsm = 0;
    if (nsm == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
        if (nsm <= 0) nsm = 128;
    }
    uint32_t c = ((uint32_t)nsm * 2u + rows - 1u) / rows;
    if (c > 8u) c = 8u;
    if (c < 1u) c = 1u;
    return c;
}

// per-tile K-split fold counters (tc5p last-CTA combine fold): zeroed once
// at alloc. Each launch's folder CTA resets its slot to 0 after folding, so
// no per-launch memset - same-stream kernels serialize completely, so the
// next launch never observes a mid-flight counter. 256 slots covers every
// tc5p shape (tiles = out_dim/128 <= 168; tiles >= 256 goes tc5q).
static uint32_t* pd_tc5p_fctr(bool make = false) {
    static uint32_t* ctr = nullptr;
    if (make && !ctr) {
        if (cudaMalloc(&ctr, 1024) != cudaSuccess) ctr = nullptr;
        else cudaMemset(ctr, 0, 1024);
    }
    return ctr;
}

PD_EXPORT
int pd_f8_repack_tiles(const void* rowmajor, void* tiles,
                       uint32_t in_dim, uint32_t out_dim, void* stream) {
    if ((in_dim & 127u) || (out_dim & 127u)) return -1;
    pd_tc5q_ctr(true);
    pd_tc5p_fctr(true);
    pd_rowq_scr(true);
    pd_smp_scr(true);   // P71b multi-block sampler scratch (capture-safe)
    pd_topp_scr(true);  // mode-6 multi-block truncation scratch (capture-safe)
    const uint64_t nchunk = ((uint64_t)in_dim >> 4) * out_dim;
    const uint32_t nb = (uint32_t)((nchunk + 255u) / 256u);
    pd_f8_tiles_repack_kernel<<<nb, 256, 0, (cudaStream_t)stream>>>(
        (const unsigned char*)rowmajor, (unsigned char*)tiles, in_dim, out_dim);
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -2;
}



// ---- tc5q: persistent tc5p (item-loop design) -----------------------
// tc5p relaunches per GEMM and its 42-slab streams pay tail waves + a full
// pipeline drain per launch; the mode-5 probe (long cold streams, live mmas)
// measured 3.8 TB/s vs tc5p's 2.8. tc5q makes the stream long inside one
// launch: a persistent grid claims (row_tile, kz) items from a gmem counter,
// the W/Y slab ring runs CONTINUOUSLY across items (global slab index, the
// v3 computed parities), and the D accumulator ping-pongs between tmem cols
// 0..63 / 64..127 so an item's epilogue overlaps the next item's mma stream
// instead of draining the pipe. Items partition K into nz planes (the ks
// combine sums them, same contract as tc5p).
// N2: fuse 2 adjacent row-tiles per mma by SWAPPING operands -
// A = Y (M=64 batch rows), B = the W pair (N=256 fused rows). Same K-major
// SW128 descriptors (SBO walks the contiguous 32 KB pair uniformly); halves
// the mma issue count AND the Y stage count per output row. M=64 packs D as
// 128 lanes x N/2 cols (upper lanes hold the second col half - the same
// quadrant layout the pf5g-c2 per-CTA halves use), so the ping-pong is
// 2 x 128 tmem cols and all 4 warps carry the epilogue with per-thread
// CONTIGUOUS 128B stores (the N=64 orientation scatters 4B x out_dim).
// 40 KB slots (32 KB W pair + 8 KB Y) cap the ring at S=5 under the 227 KB
// smem limit. Same per-element K accumulation order -> bit-identical gate.
template <uint32_t S, bool EF = false, bool N2 = false, bool TS = false>
__global__ void __launch_bounds__(128) pd_f8row_gemm_tc5q_kt(
    const unsigned char* __restrict__ wtiles, const __grid_constant__ PdTmap ymap,
    const float* __restrict__ wrs, const float* __restrict__ xrs,
    float* __restrict__ y, uint32_t in_dim, uint32_t out_dim, uint32_t batch,
    uint32_t nz, uint32_t pdist, uint32_t* __restrict__ ctr) {
#if PD_TC5_OK
    constexpr uint32_t WSLAB = N2 ? 32768u : 16384u;  // W bytes per ring slot
    // M=64 (::1) D is STRAIGHT lanes 0-63 x N cols (measured: the
    // packed dual-half layout is a cta_group::2 artifact - upper lanes stay
    // unwritten). Ping-pong = 2 x 256 = all 512 tmem cols; tc5q runs 1
    // CTA/SM so nothing else wants tmem.
    constexpr uint32_t DCOLS = N2 ? 256u : 64u;       // tmem cols per D buffer
    extern __shared__ __align__(1024) unsigned char pd_tc5q_sh[];
    unsigned char* wt = pd_tc5q_sh;                       // S x WSLAB
    unsigned char* yt = pd_tc5q_sh + S * WSLAB;           // S x 8 KB
    uint64_t* bfull = (uint64_t*)(yt + S * 8192u);
    uint64_t* bdone = bfull + S;
    // bepi: one barrier per D ping-pong buffer, fired once per finished item.
    // The slab-ring bdone can't serve the deferred epilogue: the pending
    // item's last slab gets REUSED while the epilogue is deferred, so its
    // parity aliases (up to nk/S phases pass). bepi fires exactly once per
    // buffer use and is waited exactly once one use later - lag < 2, never
    // aliases.
    uint64_t* bepi = bdone + S;
    // item queue: producer (tid0) publishes claimed items; consumers pace on
    // __syncthreads at item boundaries. Ring of 4 is deep enough: items are
    // >= ceil(nk_all/nz) >= pdist slabs, so the producer leads by <= 1 item.
    __shared__ uint32_t iq_item[4];      // item id or ~0u = out of work
    __shared__ uint32_t tmem_slot[1];

    const uint32_t tid = threadIdx.x;
    // TS: per-item timeline - (start, issue-end, epi-end) x 5 items
    unsigned long long ts_ent = 0, ts_it[15] = {};
    uint32_t ts_n = 0;
    if (TS && tid == 0) ts_ent = pd_ts_now();
    const uint32_t nk_all = (in_dim + 127u) / 128u;
    const uint32_t per = (nk_all + nz - 1u) / nz;
    const uint32_t ntiles = out_dim >> (N2 ? 8 : 7);  // N2 items span 256 rows
    const uint32_t total_items = ntiles * nz;

    if (tid == 0) {
        #pragma unroll
        for (uint32_t s2 = 0; s2 < S; ++s2) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bfull[s2])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bdone[s2])));
        }
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                     ::"r"((uint32_t)__cvta_generic_to_shared(&bepi[0])));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                     ::"r"((uint32_t)__cvta_generic_to_shared(&bepi[1])));
    }
    __syncthreads();
    if (tid < 32) {
        if (N2)
            asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(tmem_slot)));
        else
            asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 128;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(tmem_slot)));
    }
    __syncthreads();
    const uint32_t tmem = tmem_slot[0];        // D ping-pong: +0 / +DCOLS

    auto bar_wait = [&](uint64_t* bar, uint32_t parity) {
        const uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
        asm volatile("{\n\t.reg .pred p;\nW%=:\n\t"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                     "@!p bra W%=;\n\t}" ::"r"(a), "r"(parity));
    };
    // item helpers: id -> (tile, kz, k0, nk)
    auto item_nk = [&](uint32_t id, uint32_t& tile, uint32_t& k0, uint32_t& nk) {
        tile = id % ntiles;
        const uint32_t kz = id / ntiles;
        k0 = kz * per;
        nk = k0 + per < nk_all ? per : (k0 < nk_all ? nk_all - k0 : 0u);
    };
    auto tma_stage = [&](uint32_t tile, uint32_t k0, uint32_t kt, uint32_t s2) {
        const uint32_t m = (uint32_t)__cvta_generic_to_shared(&bfull[s2]);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                     ::"r"(m), "r"(WSLAB + 8192u));
        const uint32_t wd = (uint32_t)__cvta_generic_to_shared(wt + s2 * WSLAB);
        // N2: the fused pair (2*tile, 2*tile+1) lands contiguous in smem -
        // gmem slabs are tile-major so the twin sits nk_all slabs later
        const unsigned char* wsrc = wtiles
            + (((size_t)(N2 ? tile * 2u : tile) * nk_all + k0 + kt) << 14);
        if (EF) {
            uint64_t pol;
            asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
            asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                         " [%0], [%1], 16384, [%2], %3;" ::"r"(wd), "l"(wsrc), "r"(m), "l"(pol) : "memory");
            if (N2)
                asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                             " [%0], [%1], 16384, [%2], %3;" ::"r"(wd + 16384u),
                             "l"(wsrc + ((size_t)nk_all << 14)), "r"(m), "l"(pol) : "memory");
        } else {
            asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                         " [%0], [%1], 16384, [%2];" ::"r"(wd), "l"(wsrc), "r"(m) : "memory");
            if (N2)
                asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                             " [%0], [%1], 16384, [%2];" ::"r"(wd + 16384u),
                             "l"(wsrc + ((size_t)nk_all << 14)), "r"(m) : "memory");
        }
        const uint32_t yd = (uint32_t)__cvta_generic_to_shared(yt + s2 * 8192u);
        const int ck = (int)((k0 + kt) * 128u);
        asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                     " [%0], [%1, {%2, %3}], [%4];" ::"r"(yd), "l"(&ymap), "r"(ck),
                     "r"(0), "r"(m) : "memory");
    };

    // split issuer (the tc5v lesson a fourth time): producer state
    // lives on tid 32 - it claims items and stages the whole current item
    // + pdist ahead while tid 0 runs the mma chain concurrently (the old
    // interleave serialized both in tid 0: gu-shape harness 69.8 -> 51.6 us,
    // bit-identical). Warp 1 reconverges after the staging block before the
    // collective epilogue ld; iq_item publication stays ordered by the
    // item-boundary __syncthreads.
    uint32_t p_item = ~0u, p_tile = 0, p_k0 = 0, p_nk = 0, p_kt = 0, p_slot = 0;
    uint32_t qp = 0;                 // global slab index (producer)
    if (tid == 32) {
        p_item = atomicAdd(ctr, 1u);
        iq_item[0] = p_item < total_items ? p_item : ~0u;
        if (p_item < total_items) item_nk(p_item, p_tile, p_k0, p_nk);
        // A CTA that is DRY on its first claim breaks out of the consumer loop
        // below (`if (item == ~0u) break`) without ever entering produce_one,
        // so the counter-reset check there never runs for it. Its claim still
        // counted, so if that claim happened to be the last one the counter is
        // never zeroed and every subsequent tc5q launch claims out of range and
        // silently computes nothing.
        //
        // At gridDim.x == nsm this is masked: all CTAs start together and take
        // the prologue claims 0..gridDim-1, so the max value
        // (total_items+gridDim-1) always lands on a working CTA's later dry
        // claim. Raise the grid past nsm and the later waves claim last, so the
        // max lands here every time -- which is why PADDOCK_TC5Q_CTA=2 produced
        // corrupt output (32/32 -> 0/32 on-topic) and looked 43% faster: after
        // the first launch, every tc5q GEMM was a no-op.
        else if (p_item == total_items + gridDim.x - 1u) *ctr = 0u;
    }
    __syncthreads();

    auto produce_one = [&]() {   // tid32 only: stage one slab (or claim next item)
        while (p_item != ~0u && p_item < total_items && p_kt >= p_nk) {
            p_item = atomicAdd(ctr, 1u);
            p_slot = (p_slot + 1u) & 3u;
            iq_item[p_slot] = p_item < total_items ? p_item : ~0u;
            p_kt = 0;
            if (p_item < total_items) item_nk(p_item, p_tile, p_k0, p_nk);
        }
        if (p_item == ~0u || p_item >= total_items) {
            // last CTA out resets the item counter for the next launch: every
            // CTA dry-claims exactly once, so the final claim value is
            // total_items + gridDim.x - 1. Kernel completion flushes the
            // store; the next tc5q is stream-ordered after us. Kills the
            // per-launch 4-byte memsetAsync NODE (3-8 us of SM idle inside
            // the decode graph, 60x per tick).
            if (p_item == total_items + gridDim.x - 1u) *ctr = 0u;
            asm volatile("griddepcontrol.launch_dependents;");
            return false;
        }
        const uint32_t s2 = qp % S;
        if (qp >= S) bar_wait(&bdone[s2], ((qp - S) / S) & 1u);
        tma_stage(p_tile, p_k0, p_kt, s2);
        ++p_kt; ++qp;
        return true;
    };

    // N2 epilogue: M=64 ::1 D rides "Layout F" (probed): batch row r
    // lives at lane (r>>4)*32 + (r&15), col = n - each warp owns 16 rows in
    // its first 16 lanes (upper 16 unused). A raw store would scatter (lane =
    // batch, stride out_dim - 32 lines per instruction, measured -35% at
    // b=64), so: bounce each 32-col chunk through a padded smem tile, then
    // all 128 threads flush lane<->row coalesced 128B lines.
    float* bnc = (float*)(bepi + 2);                 // [64][33] f32, +pad
    auto epi_n2 = [&](uint32_t ptile, uint32_t pk0, uint32_t ppp) {
        const uint32_t warp = tid >> 5, lane = tid & 31u;
        const uint32_t bcol = warp * 16u + lane;     // valid for lane < 16
        const bool live = lane < 16u && bcol < batch;
        const float xsc = live ? xrs[bcol] : 0.0f;
        float* yz = y + (size_t)(pk0 / per) * out_dim * batch;
        #pragma unroll
        for (uint32_t cc = 0; cc < 8u; ++cc) {
            {
                uint32_t r[32];
                const uint32_t taddr = tmem + ppp * DCOLS + ((warp * 32u) << 16) + cc * 32u;
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
                    "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
                    "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
                    : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
                      "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),"=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15]),
                      "=r"(r[16]),"=r"(r[17]),"=r"(r[18]),"=r"(r[19]),"=r"(r[20]),"=r"(r[21]),"=r"(r[22]),"=r"(r[23]),
                      "=r"(r[24]),"=r"(r[25]),"=r"(r[26]),"=r"(r[27]),"=r"(r[28]),"=r"(r[29]),"=r"(r[30]),"=r"(r[31])
                    : "r"(taddr));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                if (live) {
                    #pragma unroll
                    for (uint32_t j = 0; j < 32u; ++j)
                        bnc[bcol * 33u + j] = __uint_as_float(r[j]) * xsc;
                }
            }
            __syncthreads();
            const uint32_t rbase = ptile * 256u + cc * 32u;
            const float wv = wrs[rbase + lane];      // row < out_dim: whole tiles
            #pragma unroll
            for (uint32_t q = 0; q < 16u; ++q) {
                const uint32_t bc = warp * 16u + q;
                if (bc < batch)
                    yz[(size_t)bc * out_dim + rbase + lane] = bnc[bc * 33u + lane] * wv;
            }
            __syncthreads();                         // bnc reused next chunk
        }
    };

    uint32_t qc = 0;                 // global slab index (consumer)
    uint32_t c_slot = 0, pp = 0, epi_phase = 0;
    uint32_t pend_tile = ~0u, pend_pp = 0;
    uint32_t pend_k0 = 0;
    for (;;) {
        __syncthreads();             // iq_item[c_slot] published
        const uint32_t item = iq_item[c_slot];
        if (item == ~0u) break;
        if (TS && tid == 0 && ts_n < 5u) ts_it[ts_n * 3u] = pd_ts_now();
        uint32_t tile, k0, nk;
        item_nk(item, tile, k0, nk);
        if (nk == 0) {
            // empty kz tail chunk: its plane must read as zeros (combine
            // sums every plane); D never accumulated for it
            float* yz = y + (size_t)(k0 / per) * out_dim * batch;
            if (N2) {
                for (uint32_t rr = tid; rr < 256u; rr += 128u) {
                    const uint32_t row = tile * 256u + rr;
                    for (uint32_t col = 0; col < batch; ++col)
                        yz[(size_t)col * out_dim + row] = 0.0f;
                }
            } else {
                const uint32_t warp = tid >> 5, lane = tid & 31u;
                const uint32_t row = tile * 128u + warp * 32u + lane;
                if (row < out_dim)
                    for (uint32_t col = 0; col < batch; ++col)
                        yz[(size_t)col * out_dim + row] = 0.0f;
            }
            c_slot = (c_slot + 1u) & 3u;
            continue;
        }
        if (tid == 32) {
            // stage the whole current item + pdist into the next (bounded,
            // so warp 1 reconverges before the epilogue's collective ld)
            while (qp < qc + nk + pdist && produce_one()) {}
        } else if (tid == 0) {
            for (uint32_t kt = 0; kt < nk; ++kt) {
                const uint32_t s2 = (qc + kt) % S;
                bar_wait(&bfull[s2], ((qc + kt) / S) & 1u);
                const uint32_t w16 = (uint32_t)__cvta_generic_to_shared(wt + s2 * WSLAB) >> 4;
                const uint32_t y16 = (uint32_t)__cvta_generic_to_shared(yt + s2 * 8192u) >> 4;
                const uint32_t dT = tmem + pp * DCOLS;
                #pragma unroll
                for (uint32_t kb = 0; kb < 4u; ++kb) {
                    // N2 swaps the operands: A=Y (M=64), B=W pair (N=256)
                    const uint64_t wdsc = pd_tc5_sdesc(w16 + kb * 2u);
                    const uint64_t ydsc = pd_tc5_sdesc(y16 + kb * 2u);
                    const uint64_t ad = N2 ? ydsc : wdsc;
                    const uint64_t bd = N2 ? wdsc : ydsc;
                    const uint32_t en = (kt > 0 || kb > 0) ? 1u : 0u;
                    const uint32_t idesc = N2
                        ? (1u << 4) | ((256u >> 3) << 17) | ((64u >> 4) << 24)
                        : (1u << 4) | ((64u >> 3) << 17) | ((128u >> 4) << 24);
                    asm volatile(
                        "{\n\t.reg .pred p;\n\t"
                        "setp.ne.b32 p, %4, 0;\n\t"
                        "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t}"
                        ::"r"(dT), "l"(ad), "l"(bd), "r"(idesc), "r"(en));
                }
                asm volatile(
                    "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                    ::"r"((uint32_t)__cvta_generic_to_shared(&bdone[s2])));
                if (kt == nk - 1u)   // item complete on the pipe -> fire bepi
                    asm volatile(
                        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                        ::"r"((uint32_t)__cvta_generic_to_shared(&bepi[pp])));
            }
            if (TS && ts_n < 5u) ts_it[ts_n * 3u + 1u] = pd_ts_now();
        }
        qc += nk;
        // deferred epilogue: pending item's mmas finished >= 1 item ago
        if (N2 && pend_tile != ~0u) {
            bar_wait(&bepi[pend_pp], (epi_phase >> pend_pp) & 1u);
            epi_n2(pend_tile, pend_k0, pend_pp);
        } else if (pend_tile != ~0u) {
            bar_wait(&bepi[pend_pp], (epi_phase >> pend_pp) & 1u);
            // ld-pipelined - this epilogue sits on the serial item
            // chain, so the exposed tmem-ld latency (~0.5us/chunk, the
            // wait::ld finding) is real wall time here, unlike M2's overlapped
            // steady state. Both chunks issue before the single wait.
            const uint32_t warp = tid >> 5, lane = tid & 31u;
            const uint32_t row = pend_tile * 128u + warp * 32u + lane;
            const float wsc = row < out_dim ? wrs[row] : 0.0f;
            float* yz = y + (size_t)(pend_k0 / per) * out_dim * batch;
            uint32_t r2[2][32];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
                "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
                "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
                : "=r"(r2[0][0]),"=r"(r2[0][1]),"=r"(r2[0][2]),"=r"(r2[0][3]),"=r"(r2[0][4]),"=r"(r2[0][5]),"=r"(r2[0][6]),"=r"(r2[0][7]),
                  "=r"(r2[0][8]),"=r"(r2[0][9]),"=r"(r2[0][10]),"=r"(r2[0][11]),"=r"(r2[0][12]),"=r"(r2[0][13]),"=r"(r2[0][14]),"=r"(r2[0][15]),
                  "=r"(r2[0][16]),"=r"(r2[0][17]),"=r"(r2[0][18]),"=r"(r2[0][19]),"=r"(r2[0][20]),"=r"(r2[0][21]),"=r"(r2[0][22]),"=r"(r2[0][23]),
                  "=r"(r2[0][24]),"=r"(r2[0][25]),"=r"(r2[0][26]),"=r"(r2[0][27]),"=r"(r2[0][28]),"=r"(r2[0][29]),"=r"(r2[0][30]),"=r"(r2[0][31])
                : "r"(tmem + pend_pp * 64u + ((warp * 32u) << 16) + 0u * 32u));
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
                "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
                "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
                : "=r"(r2[1][0]),"=r"(r2[1][1]),"=r"(r2[1][2]),"=r"(r2[1][3]),"=r"(r2[1][4]),"=r"(r2[1][5]),"=r"(r2[1][6]),"=r"(r2[1][7]),
                  "=r"(r2[1][8]),"=r"(r2[1][9]),"=r"(r2[1][10]),"=r"(r2[1][11]),"=r"(r2[1][12]),"=r"(r2[1][13]),"=r"(r2[1][14]),"=r"(r2[1][15]),
                  "=r"(r2[1][16]),"=r"(r2[1][17]),"=r"(r2[1][18]),"=r"(r2[1][19]),"=r"(r2[1][20]),"=r"(r2[1][21]),"=r"(r2[1][22]),"=r"(r2[1][23]),
                  "=r"(r2[1][24]),"=r"(r2[1][25]),"=r"(r2[1][26]),"=r"(r2[1][27]),"=r"(r2[1][28]),"=r"(r2[1][29]),"=r"(r2[1][30]),"=r"(r2[1][31])
                : "r"(tmem + pend_pp * 64u + ((warp * 32u) << 16) + 1u * 32u));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            if (row < out_dim) {
                #pragma unroll
                for (uint32_t cc = 0; cc < 2u; ++cc) {
                    #pragma unroll
                    for (uint32_t j = 0; j < 32u; ++j) {
                        const uint32_t col = cc * 32u + j;
                        if (col < batch)
                            yz[(size_t)col * out_dim + row] =
                                __uint_as_float(r2[cc][j]) * wsc * xrs[col];
                    }
                }
            }
        }
        if (pend_tile != ~0u) epi_phase ^= 1u << pend_pp;
        if (TS && tid == 0 && ts_n < 5u) { ts_it[ts_n * 3u + 2u] = pd_ts_now(); ++ts_n; }
        pend_tile = tile; pend_k0 = k0; pend_pp = pp;
        pp ^= 1u;
        c_slot = (c_slot + 1u) & 3u;
    }
    // final drain
    if (N2 && pend_tile != ~0u) {
        bar_wait(&bepi[pend_pp], (epi_phase >> pend_pp) & 1u);
        epi_n2(pend_tile, pend_k0, pend_pp);
    } else if (pend_tile != ~0u) {
        bar_wait(&bepi[pend_pp], (epi_phase >> pend_pp) & 1u);
        // ld-pipelined - this epilogue sits on the serial item
        // chain, so the exposed tmem-ld latency (~0.5us/chunk, the
        // wait::ld finding) is real wall time here, unlike M2's overlapped
        // steady state. Both chunks issue before the single wait.
        const uint32_t warp = tid >> 5, lane = tid & 31u;
        const uint32_t row = pend_tile * 128u + warp * 32u + lane;
        const float wsc = row < out_dim ? wrs[row] : 0.0f;
        float* yz = y + (size_t)(pend_k0 / per) * out_dim * batch;
        uint32_t r2[2][32];
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
            "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
            : "=r"(r2[0][0]),"=r"(r2[0][1]),"=r"(r2[0][2]),"=r"(r2[0][3]),"=r"(r2[0][4]),"=r"(r2[0][5]),"=r"(r2[0][6]),"=r"(r2[0][7]),
              "=r"(r2[0][8]),"=r"(r2[0][9]),"=r"(r2[0][10]),"=r"(r2[0][11]),"=r"(r2[0][12]),"=r"(r2[0][13]),"=r"(r2[0][14]),"=r"(r2[0][15]),
              "=r"(r2[0][16]),"=r"(r2[0][17]),"=r"(r2[0][18]),"=r"(r2[0][19]),"=r"(r2[0][20]),"=r"(r2[0][21]),"=r"(r2[0][22]),"=r"(r2[0][23]),
              "=r"(r2[0][24]),"=r"(r2[0][25]),"=r"(r2[0][26]),"=r"(r2[0][27]),"=r"(r2[0][28]),"=r"(r2[0][29]),"=r"(r2[0][30]),"=r"(r2[0][31])
            : "r"(tmem + pend_pp * 64u + ((warp * 32u) << 16) + 0u * 32u));
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
            "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
            : "=r"(r2[1][0]),"=r"(r2[1][1]),"=r"(r2[1][2]),"=r"(r2[1][3]),"=r"(r2[1][4]),"=r"(r2[1][5]),"=r"(r2[1][6]),"=r"(r2[1][7]),
              "=r"(r2[1][8]),"=r"(r2[1][9]),"=r"(r2[1][10]),"=r"(r2[1][11]),"=r"(r2[1][12]),"=r"(r2[1][13]),"=r"(r2[1][14]),"=r"(r2[1][15]),
              "=r"(r2[1][16]),"=r"(r2[1][17]),"=r"(r2[1][18]),"=r"(r2[1][19]),"=r"(r2[1][20]),"=r"(r2[1][21]),"=r"(r2[1][22]),"=r"(r2[1][23]),
              "=r"(r2[1][24]),"=r"(r2[1][25]),"=r"(r2[1][26]),"=r"(r2[1][27]),"=r"(r2[1][28]),"=r"(r2[1][29]),"=r"(r2[1][30]),"=r"(r2[1][31])
            : "r"(tmem + pend_pp * 64u + ((warp * 32u) << 16) + 1u * 32u));
        asm volatile("tcgen05.wait::ld.sync.aligned;");
        if (row < out_dim) {
            #pragma unroll
            for (uint32_t cc = 0; cc < 2u; ++cc) {
                #pragma unroll
                for (uint32_t j = 0; j < 32u; ++j) {
                    const uint32_t col = cc * 32u + j;
                    if (col < batch)
                        yz[(size_t)col * out_dim + row] =
                            __uint_as_float(r2[cc][j]) * wsc * xrs[col];
                }
            }
        }
    }
    __syncthreads();
    if (tid < 32) {
        if (N2)
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" ::"r"(tmem));
        else
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 128;" ::"r"(tmem));
    }
    if (TS && tid == 0 && pd_tc5p_ts) {
        unsigned long long* o = pd_tc5p_ts + (size_t)blockIdx.x * 24u;
        o[0] = pd_ts_smid(); o[1] = ts_ent; o[2] = 0; o[3] = 0;
        o[4] = 0; o[5] = pd_ts_now(); o[6] = ts_n; o[7] = blockIdx.x;
        #pragma unroll
        for (uint32_t q = 0; q < 15u; ++q) o[8u + q] = ts_it[q];
    }
#else
    (void)wtiles; (void)ymap; (void)wrs; (void)xrs; (void)y;
    (void)in_dim; (void)out_dim; (void)batch; (void)nz; (void)pdist; (void)ctr;
#endif
}

// ---- tc5t: persistent item-loop GEMM for the 65..128-row band  ---
// The spec-verify tick (~96 rows at c32) is the only band still on per-launch
// kernels: tc5r's cluster runs 42-84 CTAs on the o/down/qkv planes at ~30
// GB/s/CTA while the die holds 148, and P44's isolated probe showed the cost
// is the launch/starvation geometry, not the mma, padding, or ring depth.
// tc5t is tc5q-N2's item loop at M=128: same claim counter, same continuous
// W/Y ring across items, same D ping-pong deferred epilogue - with
//   - A = Y tile 16KB SW128 (M=128 batch rows; the descriptor math is the
//     W-tile math, and the 128-row-box ymap is tc5r's),
//   - B = the contiguous W pair 32KB (N=256 fused weight rows, the N2 desc),
//   - idesc M=128/N=256 (each half proven separately in ::1),
//   - D straight lanes (lane = batch row) -> per-thread CONTIGUOUS 128B
//     stores, no N2 bounce tile.
// Items partition K into nz planes (ks combine sums, tc5p/q contract); the
// launcher fills the die with ~ceil(nsm/ntiles) z-planes per the P46 ledger.
// Numerics: per-element K order identical to tc5p/q/r (k-ascending, 4xK32
// per tile, fixed-order combine). Ring: 48KB slots cap S=4 under 227KB.
template <uint32_t S, bool EF = false>
__global__ void __launch_bounds__(128) pd_f8row_gemm_tc5t_kt(
    const unsigned char* __restrict__ wtiles, const __grid_constant__ PdTmap ymap,
    const float* __restrict__ wrs, const float* __restrict__ xrs,
    float* __restrict__ y, uint32_t in_dim, uint32_t out_dim, uint32_t batch,
    uint32_t nz, uint32_t pdist, uint32_t* __restrict__ ctr) {
#if PD_TC5_OK
    constexpr uint32_t WSLAB = 32768u;            // W pair bytes per ring slot
    constexpr uint32_t YSLAB = 16384u;            // 128-row Y tile per slot
    constexpr uint32_t DCOLS = 256u;              // tmem cols per D buffer
    extern __shared__ __align__(1024) unsigned char pd_tc5t_sh[];
    unsigned char* wt = pd_tc5t_sh;                       // S x WSLAB
    unsigned char* yt = pd_tc5t_sh + S * WSLAB;           // S x YSLAB
    uint64_t* bfull = (uint64_t*)(yt + S * YSLAB);
    uint64_t* bdone = bfull + S;
    // bepi: fired once per D buffer per item, waited once one item later -
    // the slab-ring bdone parities alias under the deferred epilogue (see
    // tc5q). Lag < 2, never aliases.
    uint64_t* bepi = bdone + S;
    __shared__ uint32_t iq_item[4];      // item id or ~0u = out of work
    __shared__ uint32_t tmem_slot[1];

    const uint32_t tid = threadIdx.x;
    const uint32_t nk_all = (in_dim + 127u) / 128u;
    const uint32_t per = (nk_all + nz - 1u) / nz;
    const uint32_t ntiles = out_dim >> 8;         // items span 256 rows
    const uint32_t total_items = ntiles * nz;

    if (tid == 0) {
        #pragma unroll
        for (uint32_t s2 = 0; s2 < S; ++s2) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bfull[s2])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bdone[s2])));
        }
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                     ::"r"((uint32_t)__cvta_generic_to_shared(&bepi[0])));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                     ::"r"((uint32_t)__cvta_generic_to_shared(&bepi[1])));
    }
    __syncthreads();
    if (tid < 32)
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;"
                     ::"r"((uint32_t)__cvta_generic_to_shared(tmem_slot)));
    __syncthreads();
    const uint32_t tmem = tmem_slot[0];        // D ping-pong: +0 / +DCOLS

    auto bar_wait = [&](uint64_t* bar, uint32_t parity) {
        const uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
        asm volatile("{\n\t.reg .pred p;\nW%=:\n\t"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                     "@!p bra W%=;\n\t}" ::"r"(a), "r"(parity));
    };
    auto item_nk = [&](uint32_t id, uint32_t& tile, uint32_t& k0, uint32_t& nk) {
        tile = id % ntiles;
        const uint32_t kz = id / ntiles;
        k0 = kz * per;
        nk = k0 + per < nk_all ? per : (k0 < nk_all ? nk_all - k0 : 0u);
    };
    auto tma_stage = [&](uint32_t tile, uint32_t k0, uint32_t kt, uint32_t s2) {
        const uint32_t m = (uint32_t)__cvta_generic_to_shared(&bfull[s2]);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                     ::"r"(m), "r"(WSLAB + YSLAB));
        // the fused pair (2*tile, 2*tile+1) lands contiguous in smem - gmem
        // slabs are tile-major so the twin sits nk_all slabs later
        const uint32_t wd = (uint32_t)__cvta_generic_to_shared(wt + s2 * WSLAB);
        const unsigned char* wsrc = wtiles
            + (((size_t)tile * 2u * nk_all + k0 + kt) << 14);
        if (EF) {
            uint64_t pol;
            asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
            asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                         " [%0], [%1], 16384, [%2], %3;" ::"r"(wd), "l"(wsrc), "r"(m), "l"(pol) : "memory");
            asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                         " [%0], [%1], 16384, [%2], %3;" ::"r"(wd + 16384u),
                         "l"(wsrc + ((size_t)nk_all << 14)), "r"(m), "l"(pol) : "memory");
        } else {
            asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                         " [%0], [%1], 16384, [%2];" ::"r"(wd), "l"(wsrc), "r"(m) : "memory");
            asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                         " [%0], [%1], 16384, [%2];" ::"r"(wd + 16384u),
                         "l"(wsrc + ((size_t)nk_all << 14)), "r"(m) : "memory");
        }
        // Y: 128-row box (batch <= 128 = one col tile, row origin always 0);
        // re-reads across items stay L2-resident (Y is MBs, W evicts first)
        const uint32_t yd = (uint32_t)__cvta_generic_to_shared(yt + s2 * YSLAB);
        const int ck = (int)((k0 + kt) * 128u);
        asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                     " [%0], [%1, {%2, %3}], [%4];" ::"r"(yd), "l"(&ymap), "r"(ck),
                     "r"(0), "r"(m) : "memory");
    };

    // split issuer (the tc5q shape): producer state on tid 32, mma
    // chain on tid 0, reconverging at item boundaries
    uint32_t p_item = ~0u, p_tile = 0, p_k0 = 0, p_nk = 0, p_kt = 0, p_slot = 0;
    uint32_t qp = 0;                 // global slab index (producer)
    if (tid == 32) {
        p_item = atomicAdd(ctr, 1u);
        iq_item[0] = p_item < total_items ? p_item : ~0u;
        if (p_item < total_items) item_nk(p_item, p_tile, p_k0, p_nk);
        // dry-on-first-claim reset contract: see tc5q - masked at grid==nsm,
        // which is the only grid this kernel is launched with
        else if (p_item == total_items + gridDim.x - 1u) *ctr = 0u;
    }
    __syncthreads();

    auto produce_one = [&]() {   // tid32 only: stage one slab (or claim next item)
        while (p_item != ~0u && p_item < total_items && p_kt >= p_nk) {
            p_item = atomicAdd(ctr, 1u);
            p_slot = (p_slot + 1u) & 3u;
            iq_item[p_slot] = p_item < total_items ? p_item : ~0u;
            p_kt = 0;
            if (p_item < total_items) item_nk(p_item, p_tile, p_k0, p_nk);
        }
        if (p_item == ~0u || p_item >= total_items) {
            // last CTA out resets the counter for the next launch (tc5q
            // contract: every CTA dry-claims exactly once)
            if (p_item == total_items + gridDim.x - 1u) *ctr = 0u;
            asm volatile("griddepcontrol.launch_dependents;");
            return false;
        }
        const uint32_t s2 = qp % S;
        if (qp >= S) bar_wait(&bdone[s2], ((qp - S) / S) & 1u);
        tma_stage(p_tile, p_k0, p_kt, s2);
        ++p_kt; ++qp;
        return true;
    };

    // straight-lane deferred epilogue: lane = batch row, cols = the item's
    // 256 weight rows; per-thread contiguous 16B stores. Scale fold
    // y = D * xrs[brow] * wrs[col] (exact under K-split: linear in the sum).
    auto epi_t = [&](uint32_t ptile, uint32_t pk0, uint32_t ppp) {
        const uint32_t warp = tid >> 5, lane = tid & 31u;
        const uint32_t brow = warp * 32u + lane;
        const bool live = brow < batch;
        const float xsc = live ? xrs[brow] : 0.0f;
        float* yz = y + (size_t)(pk0 / per) * out_dim * batch;
        #pragma unroll
        for (uint32_t cp = 0; cp < 4u; ++cp) {           // 2 chunks per pass
            uint32_t r2[2][32];
            #pragma unroll
            for (uint32_t h = 0; h < 2u; ++h) {
                const uint32_t taddr = tmem + ppp * DCOLS + ((warp * 32u) << 16)
                    + (cp * 2u + h) * 32u;
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
                    "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
                    "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
                    : "=r"(r2[h][0]),"=r"(r2[h][1]),"=r"(r2[h][2]),"=r"(r2[h][3]),"=r"(r2[h][4]),"=r"(r2[h][5]),"=r"(r2[h][6]),"=r"(r2[h][7]),
                      "=r"(r2[h][8]),"=r"(r2[h][9]),"=r"(r2[h][10]),"=r"(r2[h][11]),"=r"(r2[h][12]),"=r"(r2[h][13]),"=r"(r2[h][14]),"=r"(r2[h][15]),
                      "=r"(r2[h][16]),"=r"(r2[h][17]),"=r"(r2[h][18]),"=r"(r2[h][19]),"=r"(r2[h][20]),"=r"(r2[h][21]),"=r"(r2[h][22]),"=r"(r2[h][23]),
                      "=r"(r2[h][24]),"=r"(r2[h][25]),"=r"(r2[h][26]),"=r"(r2[h][27]),"=r"(r2[h][28]),"=r"(r2[h][29]),"=r"(r2[h][30]),"=r"(r2[h][31])
                    : "r"(taddr));
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            #pragma unroll
            for (uint32_t h = 0; h < 2u; ++h) {
                const uint32_t rbase = ptile * 256u + (cp * 2u + h) * 32u;
                const float wv = wrs[rbase + lane];      // whole tiles: in range
                float* dst = yz + (size_t)brow * out_dim + rbase;
                // shuffles outside the live guard: a mixed live/dead warp
                // (batch not a multiple of 32 - e.g. the 65..127 verify
                // widths) deadlocks a full-mask shfl placed under the guard
                // (P46 localization, tc5t_dbg w2=103). Dead lanes compute
                // and discard; only the store is guarded.
                #pragma unroll
                for (uint32_t q = 0; q < 8u; ++q) {
                    float4 v;
                    v.x = __uint_as_float(r2[h][q * 4u + 0u]) * xsc
                        * __shfl_sync(0xffffffffu, wv, (int)(q * 4u + 0u));
                    v.y = __uint_as_float(r2[h][q * 4u + 1u]) * xsc
                        * __shfl_sync(0xffffffffu, wv, (int)(q * 4u + 1u));
                    v.z = __uint_as_float(r2[h][q * 4u + 2u]) * xsc
                        * __shfl_sync(0xffffffffu, wv, (int)(q * 4u + 2u));
                    v.w = __uint_as_float(r2[h][q * 4u + 3u]) * xsc
                        * __shfl_sync(0xffffffffu, wv, (int)(q * 4u + 3u));
                    if (live) *(float4*)(dst + q * 4u) = v;
                }
            }
        }
    };

    uint32_t qc = 0;                 // global slab index (consumer)
    uint32_t c_slot = 0, pp = 0, epi_phase = 0;
    uint32_t pend_tile = ~0u, pend_pp = 0;
    uint32_t pend_k0 = 0;
    for (;;) {
        __syncthreads();             // iq_item[c_slot] published
        const uint32_t item = iq_item[c_slot];
        if (item == ~0u) break;
        uint32_t tile, k0, nk;
        item_nk(item, tile, k0, nk);
        if (nk == 0) {
            // empty kz tail chunk: its plane must read as zeros (combine
            // sums every plane); D never accumulated for it
            float* yz = y + (size_t)(k0 / per) * out_dim * batch;
            for (uint32_t rr = tid; rr < 256u; rr += 128u) {
                const uint32_t row = tile * 256u + rr;
                for (uint32_t col = 0; col < batch; ++col)
                    yz[(size_t)col * out_dim + row] = 0.0f;
            }
            c_slot = (c_slot + 1u) & 3u;
            continue;
        }
        if (tid == 32) {
            while (qp < qc + nk + pdist && produce_one()) {}
        } else if (tid == 0) {
            for (uint32_t kt = 0; kt < nk; ++kt) {
                const uint32_t s2 = (qc + kt) % S;
                bar_wait(&bfull[s2], ((qc + kt) / S) & 1u);
                const uint32_t w16 = (uint32_t)__cvta_generic_to_shared(wt + s2 * WSLAB) >> 4;
                const uint32_t y16 = (uint32_t)__cvta_generic_to_shared(yt + s2 * YSLAB) >> 4;
                const uint32_t dT = tmem + pp * DCOLS;
                #pragma unroll
                for (uint32_t kb = 0; kb < 4u; ++kb) {
                    // A = Y (M=128 batch rows), B = W pair (N=256)
                    const uint64_t ad = pd_tc5_sdesc(y16 + kb * 2u);
                    const uint64_t bd = pd_tc5_sdesc(w16 + kb * 2u);
                    const uint32_t en = (kt > 0 || kb > 0) ? 1u : 0u;
                    const uint32_t idesc =
                        (1u << 4) | ((256u >> 3) << 17) | ((128u >> 4) << 24);
                    asm volatile(
                        "{\n\t.reg .pred p;\n\t"
                        "setp.ne.b32 p, %4, 0;\n\t"
                        "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t}"
                        ::"r"(dT), "l"(ad), "l"(bd), "r"(idesc), "r"(en));
                }
                asm volatile(
                    "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                    ::"r"((uint32_t)__cvta_generic_to_shared(&bdone[s2])));
                if (kt == nk - 1u)   // item complete on the pipe -> fire bepi
                    asm volatile(
                        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                        ::"r"((uint32_t)__cvta_generic_to_shared(&bepi[pp])));
            }
        }
        qc += nk;
        // deferred epilogue: pending item's mmas finished >= 1 item ago
        if (pend_tile != ~0u) {
            bar_wait(&bepi[pend_pp], (epi_phase >> pend_pp) & 1u);
            epi_t(pend_tile, pend_k0, pend_pp);
            epi_phase ^= 1u << pend_pp;
        }
        pend_tile = tile; pend_k0 = k0; pend_pp = pp;
        pp ^= 1u;
        c_slot = (c_slot + 1u) & 3u;
    }
    // final drain
    if (pend_tile != ~0u) {
        bar_wait(&bepi[pend_pp], (epi_phase >> pend_pp) & 1u);
        epi_t(pend_tile, pend_k0, pend_pp);
    }
    __syncthreads();
    if (tid < 32)
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" ::"r"(tmem));
#else
    (void)wtiles; (void)ymap; (void)wrs; (void)xrs; (void)y;
    (void)in_dim; (void)out_dim; (void)batch; (void)nz; (void)pdist; (void)ctr;
#endif
}

// ---- tc5m: M64-tile decode GEMM  --------------------------------
// The CUTLASS M64-OUTPUT-TILE decode geometry is what this copies: gu
// (1,672,1) at 37.9us = 6.1 TB/s, i.e. the weight-stream floor, where tc5q
// streams the same planes from 148 pinned persistent CTAs at only
// 4.2-4.4 TB/s. Every within-geometry knob measured flat, and the
// TC5Q_CTA=2 retest that looked like a falsification was tmem-starved
// (tc5q claims all 512 cols/CTA), not a verdict on the stream count.
// tc5m is the finer-tile rewrite: one
// 64-row W tile per CTA, grid = out/64 - gu launches 672 CTAs, each its own
// TMA stream. The 64-row half-tile is CONTIGUOUS in the SW128 image (rows
// 0..63 = the first 8KB of a 128-row tile), so the shipped weight planes
// serve it unchanged; Y rides the h64 map (8KB, batch <= 64, callers'
// 256-align guarantee). Operands swap like tc5q-N2: A = Y (M=64 batch
// rows), B = W half-tile (N=64) -> D = 64 tmem cols, so 3 CTAs/SM at S=4
// (2 at S=6) fit both smem and tmem. Layout-F epilogue via the smem bounce
// (the N2 recipe at 2 chunks). No K-split, no combine: output is final.
// Numeric class: same M=64 datapath as tc5q-N2 (few-ULP vs the N=64
// route), same K walk order within the tile.
template <uint32_t S>
__global__ void __launch_bounds__(128) pd_f8row_gemm_tc5m_kt(
    const unsigned char* __restrict__ wtiles, const __grid_constant__ PdTmap ymap,
    const float* __restrict__ wrs, const float* __restrict__ xrs,
    float* __restrict__ y, uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
#if PD_TC5_OK
    extern __shared__ __align__(1024) unsigned char pd_tc5m_sh[];
    unsigned char* wt = pd_tc5m_sh;                    // S x 8KB W half-tiles
    unsigned char* yt = pd_tc5m_sh + S * 8192u;        // S x 8KB Y slabs
    uint64_t* bfull = (uint64_t*)(yt + S * 8192u);
    uint64_t* bdone = bfull + S;
    float* bnc = (float*)(bdone + S);                  // [64][33] bounce
    __shared__ uint32_t tmem_slot[1];

    const uint32_t tid = threadIdx.x;
    const uint32_t nk = (in_dim + 127u) / 128u;
    const uint32_t t64 = blockIdx.x;
    const uint32_t t128 = t64 >> 1, half = t64 & 1u;

    if (tid == 0) {
        #pragma unroll
        for (uint32_t s = 0; s < S; ++s) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bfull[s])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bdone[s])));
        }
    }
    __syncthreads();
    if (tid < 32)
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 64;"
                     ::"r"((uint32_t)__cvta_generic_to_shared(tmem_slot)));
    __syncthreads();
    const uint32_t tmem = tmem_slot[0];

    auto bar_wait = [&](uint64_t* bar, uint32_t parity) {
        const uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
        asm volatile("{\n\t.reg .pred p;\nW%=:\n\t"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                     "@!p bra W%=;\n\t}" ::"r"(a), "r"(parity));
    };

    if (tid == 32) {
        // producer: 8KB W half-tile + 8KB Y slab per k-tile
        for (uint32_t kt = 0; kt < nk; ++kt) {
            const uint32_t s = kt % S;
            if (kt >= S) bar_wait(&bdone[s], ((kt - S) / S) & 1u);
            const uint32_t m = (uint32_t)__cvta_generic_to_shared(&bfull[s]);
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                         ::"r"(m), "r"(16384u));
            const uint32_t wd = (uint32_t)__cvta_generic_to_shared(wt + s * 8192u);
            const unsigned char* wsrc = wtiles
                + (((size_t)t128 * nk + kt) << 14) + ((size_t)half << 13);
            asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                         " [%0], [%1], 8192, [%2];" ::"r"(wd), "l"(wsrc), "r"(m)
                         : "memory");
            const uint32_t yd = (uint32_t)__cvta_generic_to_shared(yt + s * 8192u);
            asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                         " [%0], [%1, {%2, %3}], [%4];" ::"r"(yd),
                         "l"(&ymap), "r"((int)(kt * 128u)), "r"(0), "r"(m)
                         : "memory");
        }
    } else if (tid == 0) {
        for (uint32_t kt = 0; kt < nk; ++kt) {
            const uint32_t s = kt % S;
            bar_wait(&bfull[s], (kt / S) & 1u);
            const uint32_t w16 = (uint32_t)__cvta_generic_to_shared(wt + s * 8192u) >> 4;
            const uint32_t y16 = (uint32_t)__cvta_generic_to_shared(yt + s * 8192u) >> 4;
            #pragma unroll
            for (uint32_t kb = 0; kb < 4u; ++kb) {
                const uint64_t ad = pd_tc5_sdesc(y16 + kb * 2u);   // A = Y, M=64
                const uint64_t bd = pd_tc5_sdesc(w16 + kb * 2u);   // B = W, N=64
                const uint32_t en = (kt > 0 || kb > 0) ? 1u : 0u;
                const uint32_t idesc =
                    (1u << 4) | ((64u >> 3) << 17) | ((64u >> 4) << 24);
                asm volatile(
                    "{\n\t.reg .pred p;\n\t"
                    "setp.ne.b32 p, %4, 0;\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t}"
                    ::"r"(tmem), "l"(ad), "l"(bd), "r"(idesc), "r"(en));
            }
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                ::"r"((uint32_t)__cvta_generic_to_shared(&bdone[s])));
        }
    }
    __syncthreads();
    if (tid == 0 && nk > 0) bar_wait(&bdone[(nk - 1u) % S], ((nk - 1u) / S) & 1u);
    __syncthreads();
    // Layout-F epilogue (the tc5q-N2 recipe at 2 chunks): batch row r lives
    // at lane (r>>4)*32 + (r&15); bounce each 32-col chunk through padded
    // smem so the flush stores coalesced 128B lines.
    {
        const uint32_t warp = tid >> 5, lane = tid & 31u;
        const uint32_t bcol = warp * 16u + lane;
        const bool live = lane < 16u && bcol < batch;
        const float xsc = live ? xrs[bcol] : 0.0f;
        #pragma unroll
        for (uint32_t cc = 0; cc < 2u; ++cc) {
            uint32_t r[32];
            const uint32_t taddr = tmem + ((warp * 32u) << 16) + cc * 32u;
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
                "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
                "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
                : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
                  "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),"=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15]),
                  "=r"(r[16]),"=r"(r[17]),"=r"(r[18]),"=r"(r[19]),"=r"(r[20]),"=r"(r[21]),"=r"(r[22]),"=r"(r[23]),
                  "=r"(r[24]),"=r"(r[25]),"=r"(r[26]),"=r"(r[27]),"=r"(r[28]),"=r"(r[29]),"=r"(r[30]),"=r"(r[31])
                : "r"(taddr));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            if (live) {
                #pragma unroll
                for (uint32_t j = 0; j < 32u; ++j)
                    bnc[bcol * 33u + j] = __uint_as_float(r[j]) * xsc;
            }
            __syncthreads();
            const uint32_t row = t64 * 64u + cc * 32u + lane;
            const float wv = wrs[row];                 // whole tiles: out%64==0
            #pragma unroll
            for (uint32_t q = 0; q < 16u; ++q) {
                const uint32_t bc = warp * 16u + q;
                if (bc < batch)
                    y[(size_t)bc * out_dim + row] = bnc[bc * 33u + lane] * wv;
            }
            __syncthreads();
        }
    }
    if (tid < 32)
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 64;" ::"r"(tmem));
#else
    (void)wtiles; (void)ymap; (void)wrs; (void)xrs; (void)y;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}


// fused-plane epilogue for the f8t decode band: input [rows][2*n_ff] f32
// ([gate | up] per token, exactly what the fused gu GEMM writes), gelu_tanh
// fold, per-ROW e4m3 quant into a COMPACT [rows][n_ff] buffer (contiguous
// rows - the down GEMM's 64-row TMA boxes need them). Row max is
// order-exact, so results are bit-stable across launch geometries.
// geglu2 two-stage twins (act()-based max; same partition-invariant
// exponent argument as the plain row pair above)
// ACT picks the arch's gate nonlinearity (pd_glu_act, abi.cuh); the GELU
// instantiation is byte-for-byte what shipped before the template.
template <int ACT>
__global__ void pd_rowmax_glu2_part_kernel(const float* __restrict__ gu,
                                           float* __restrict__ parts,
                                           uint32_t n_ff, uint32_t nzp) {
    PD_PDL_ARM();  // no-op below sm_90 (multi-arch fatbin: raw asm breaks ptxas)
    const uint32_t row = blockIdx.x, c = blockIdx.y, C = gridDim.y;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    const float* g = gu + (size_t)row * 2u * n_ff;
    const float* u = g + n_ff;
    const size_t np2 = (size_t)gridDim.x * 2u * n_ff;
    auto ld4z = [&](const float* base, uint32_t i) -> float4 {
        float4 v = *(const float4*)(base + (size_t)i * 4u);
        for (uint32_t z = 1; z < nzp; ++z) {
            const float4 pz = *(const float4*)(base + (size_t)z * np2 + (size_t)i * 4u);
            v.x += pz.x; v.y += pz.y; v.z += pz.z; v.w += pz.w;
        }
        return v;
    };
    auto act = [](float x, float y) { return pd_glu_act<ACT>(x) * y; };
    const uint32_t n4 = n_ff >> 2;
    const uint32_t n4c = (n4 + C - 1u) / C;
    const uint32_t i0 = c * n4c, i1 = min(n4, i0 + n4c);
    __shared__ float wmax[32];
    float a = 0.0f;
    for (uint32_t i = i0 + tid; i < i1; i += nth) {
        const float4 gv = ld4z(g, i);
        const float4 uv = ld4z(u, i);
        a = fmaxf(a, fabsf(act(gv.x, uv.x)));
        a = fmaxf(a, fabsf(act(gv.y, uv.y)));
        a = fmaxf(a, fabsf(act(gv.z, uv.z)));
        a = fmaxf(a, fabsf(act(gv.w, uv.w)));
    }
    for (uint32_t sh = 16; sh > 0; sh >>= 1)
        a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, sh));
    if ((tid & 31u) == 0) wmax[tid >> 5] = a;
    __syncthreads();
    if (tid == 0) {
        float m = 0.0f;
        for (uint32_t w = 0; w < ((nth + 31u) >> 5); ++w) m = fmaxf(m, wmax[w]);
        parts[(size_t)row * C + c] = m;
    }
}

template <int ACT>
__global__ void pd_quantize_e4m3_glu2_row2_kernel(const float* __restrict__ gu,
                                                  unsigned char* __restrict__ q,
                                                  float* __restrict__ rscale,
                                                  uint32_t n_ff,
                                                  const float* __restrict__ parts,
                                                  uint32_t nzp) {
    PD_PDL_ARM();  // no-op below sm_90 (multi-arch fatbin: raw asm breaks ptxas)
    const uint32_t row = blockIdx.x, c = blockIdx.y, C = gridDim.y;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    const float* g = gu + (size_t)row * 2u * n_ff;
    const float* u = g + n_ff;
    const size_t np2 = (size_t)gridDim.x * 2u * n_ff;
    auto ld4z = [&](const float* base, uint32_t i) -> float4 {
        float4 v = *(const float4*)(base + (size_t)i * 4u);
        for (uint32_t z = 1; z < nzp; ++z) {
            const float4 pz = *(const float4*)(base + (size_t)z * np2 + (size_t)i * 4u);
            v.x += pz.x; v.y += pz.y; v.z += pz.z; v.w += pz.w;
        }
        return v;
    };
    auto act = [](float x, float y) { return pd_glu_act<ACT>(x) * y; };
    const int e = pd_rowq_exp_from_parts(parts, row, C);
    if (c == 0 && tid == 0) rscale[row] = ldexpf(1.0f, e);
    const float inv = ldexpf(1.0f, -e);
    unsigned char* qr = q + (size_t)row * n_ff;
    const uint32_t n4 = n_ff >> 2;
    const uint32_t n4c = (n4 + C - 1u) / C;
    const uint32_t i0 = c * n4c, i1 = min(n4, i0 + n4c);
    for (uint32_t i = i0 + tid; i < i1; i += nth) {
        const float4 gv = ld4z(g, i);
        const float4 uv = ld4z(u, i);
        uchar4 o;
        o.x = __nv_fp8_e4m3(act(gv.x, uv.x) * inv).__x;
        o.y = __nv_fp8_e4m3(act(gv.y, uv.y) * inv).__x;
        o.z = __nv_fp8_e4m3(act(gv.z, uv.z) * inv).__x;
        o.w = __nv_fp8_e4m3(act(gv.w, uv.w) * inv).__x;
        *(uchar4*)(qr + (size_t)i * 4u) = o;
    }
}

// P55 glue fusion: rowmax_part + quantize_row2 as one kernel.
//
// The split exists only because block (row,c) of the quantize needs the max
// over all C chunks of its row, which is a cross-block dependency. But C is
// capped at 8 by pd_rowq_chunks, and 8 is a legal CLUSTER size - so a cluster
// barrier can carry that dependency and the kernel boundary goes away. The
// occupancy that motivated the split (rows alone would leave a 148-SM die
// mostly idle) is untouched: the grid is still (rows, C).
//
// NUMERICALLY BIT-IDENTICAL, and provably so: the only value crossing blocks
// is a MAX, which is exact and order-independent. That is the opposite of the
// wide-nth thread-count change, which regrouped a SUM and duly failed the PPL
// gate (see gemma4_ppl.rs). Nothing here re-associates any addition - the
// nzp z-fold inside ld4z keeps its ascending order in both phases.
//
// Peer reads use mapa + ld.shared::cluster. A mapa'd address must not be
// turned into a generic pointer and dereferenced (that faults); the explicit
// ld.shared::cluster is the only correct read. __cluster_dims__ takes LITERAL
// dims, never a template parameter (a template-dependent form miscompiles),
// which is why this arm is C==8 only and the launcher falls back otherwise.
// The declaration attribute is evaluated on every device pass, so the whole
// kernel is guarded below sm_90 or the multi-arch fatbin breaks on Ampere/Ada.
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 900)
template <int ACT>
__global__ void __launch_bounds__(256) __cluster_dims__(1, 8, 1)
pd_quantize_e4m3_glu2_row2c_kernel(const float* __restrict__ gu,
                                   unsigned char* __restrict__ q,
                                   float* __restrict__ rscale,
                                   uint32_t n_ff, uint32_t nzp) {
    PD_PDL_ARM();
    constexpr uint32_t C = 8u;
    const uint32_t row = blockIdx.x, c = blockIdx.y;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    const float* g = gu + (size_t)row * 2u * n_ff;
    const float* u = g + n_ff;
    const size_t np2 = (size_t)gridDim.x * 2u * n_ff;
    auto ld4z = [&](const float* base, uint32_t i) -> float4 {
        float4 v = *(const float4*)(base + (size_t)i * 4u);
        for (uint32_t z = 1; z < nzp; ++z) {
            const float4 pz = *(const float4*)(base + (size_t)z * np2 + (size_t)i * 4u);
            v.x += pz.x; v.y += pz.y; v.z += pz.z; v.w += pz.w;
        }
        return v;
    };
    auto act = [](float x, float y) { return pd_glu_act<ACT>(x) * y; };
    const uint32_t n4 = n_ff >> 2;
    const uint32_t n4c = (n4 + C - 1u) / C;
    const uint32_t i0 = c * n4c, i1 = min(n4, i0 + n4c);

    // phase 1 == pd_rowmax_glu2_part_kernel, verbatim
    __shared__ float wmax[32];
    __shared__ float s_part;
    float a = 0.0f;
    for (uint32_t i = i0 + tid; i < i1; i += nth) {
        const float4 gv = ld4z(g, i);
        const float4 uv = ld4z(u, i);
        a = fmaxf(a, fabsf(act(gv.x, uv.x)));
        a = fmaxf(a, fabsf(act(gv.y, uv.y)));
        a = fmaxf(a, fabsf(act(gv.z, uv.z)));
        a = fmaxf(a, fabsf(act(gv.w, uv.w)));
    }
    for (uint32_t sh = 16; sh > 0; sh >>= 1)
        a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, sh));
    if ((tid & 31u) == 0) wmax[tid >> 5] = a;
    __syncthreads();
    if (tid == 0) {
        float m0 = 0.0f;
        for (uint32_t w = 0; w < ((nth + 31u) >> 5); ++w) m0 = fmaxf(m0, wmax[w]);
        s_part = m0;
    }
    // carries the cross-block dependency the second kernel used to carry
    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;" ::: "memory");

    // phase 2 == pd_quantize_e4m3_glu2_row2_kernel, with the row max taken
    // from peer shared memory instead of the `parts` scratch buffer. Same set
    // of C values, same fmaxf fold order (ascending rank == ascending c).
    float m = 0.0f;
    {
        const uint32_t sa = (uint32_t)__cvta_generic_to_shared(&s_part);
        #pragma unroll
        for (uint32_t p = 0; p < C; ++p) {
            uint32_t pa; float v;
            asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                         : "=r"(pa) : "r"(sa), "r"(p));
            asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(pa));
            m = fmaxf(m, v);
        }
    }
    int e = 0;                       // identical to pd_rowq_exp_from_parts
    if (m > 0.0f) {
        int ex;
        float fr = frexpf(m, &ex);
        e = ex - 9 + (fr > 0.875f ? 1 : 0);
    }
    if (c == 0 && tid == 0) rscale[row] = ldexpf(1.0f, e);
    const float inv = ldexpf(1.0f, -e);
    unsigned char* qr = q + (size_t)row * n_ff;
    for (uint32_t i = i0 + tid; i < i1; i += nth) {
        const float4 gv = ld4z(g, i);
        const float4 uv = ld4z(u, i);
        uchar4 o;
        o.x = __nv_fp8_e4m3(act(gv.x, uv.x) * inv).__x;
        o.y = __nv_fp8_e4m3(act(gv.y, uv.y) * inv).__x;
        o.z = __nv_fp8_e4m3(act(gv.z, uv.z) * inv).__x;
        o.w = __nv_fp8_e4m3(act(gv.w, uv.w) * inv).__x;
        *(uchar4*)(qr + (size_t)i * 4u) = o;
    }
}
#endif  // __cluster_dims__ guard (host pass || sm_90+)

// cluster launch gate: sm_90+ only, queried once.
// DEFAULT ON. Unlike the wide-nth thread-count arm - which was refused as a
// default because it regrouped a SUM and measurably moved PPL on 4/4 corpora -
// this one is BIT-IDENTICAL, not merely a near-tie: the only value crossing
// blocks is a max, which is exact and order-independent. Verified end to end,
// mean_nll 9.96530 and ppl 21275.32169 to the digit with the arm on and off.
// PADDOCK_GLU2_FUSE=0 forces the two-kernel path back.
static bool pd_glu2_fuse_on() {
    static int v = -1;
    if (v < 0) {
        const char* e = pd_env("PADDOCK_GLU2_FUSE");
        if (e && e[0] == '0') { v = 0; }
        else {
            int dev = 0, maj = 0;
            cudaGetDevice(&dev);
            cudaDeviceGetAttribute(&maj, cudaDevAttrComputeCapabilityMajor, dev);
            v = (maj >= 9) ? 1 : 0;
            if (e) fprintf(stderr, "[glu2fuse] %s (cc major %d)\n", v ? "ON" : "off (needs sm_90+)", maj);
        }
    }
    return v == 1;
}

template <int ACT>
__global__ void pd_quantize_e4m3_glu2_row_kernel(const float* __restrict__ gu,
                                                 unsigned char* __restrict__ q,
                                                 float* __restrict__ rscale,
                                                 uint32_t n_ff, uint32_t nzp) {
    // PDL: let the next (dependent-launched) GEMM start its dep-free W
    // prefetch while this kernel runs; its griddepcontrol.wait still gates
    // every dependent read on our full completion (probe-proven semantics).
    //  cascade: this kernel now also launches as a dependent, so gate
    // the body on full predecessor completion (no-op under plain launches).
    // PD_PDL_ARM (not raw asm): no-op below sm_90 - this kernel builds for
    // every arch and raw griddepcontrol breaks ptxas there.
    PD_PDL_ARM();

    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    const float* g = gu + (size_t)row * 2u * n_ff;
    const float* u = g + n_ff;
    // nzp > 1: gu is the fused GEMM's nz partial planes - every
    // load sums them ascending-z (the combine kernel's order -> bit-equal).
    // The [2*n_ff] row (172 KB) cannot stage in smem, so both passes re-sum;
    // that still beats combine (44 MB total reads vs its 55 MB round trip)
    // and deletes the launch. nzp == 1 is the original single load.
    const size_t np2 = (size_t)gridDim.x * 2u * n_ff;
    auto ld4z = [&](const float* base, uint32_t i) -> float4 {
        float4 v = *(const float4*)(base + (size_t)i * 4u);
        for (uint32_t z = 1; z < nzp; ++z) {
            const float4 pz = *(const float4*)(base + (size_t)z * np2 + (size_t)i * 4u);
            v.x += pz.x; v.y += pz.y; v.z += pz.z; v.w += pz.w;
        }
        return v;
    };
    __shared__ float wmax[32];
    __shared__ int s_e;
    auto act = [](float x, float y) { return pd_glu_act<ACT>(x) * y; };
    const uint32_t n4 = n_ff >> 2;
    float a = 0.0f;
    for (uint32_t i = tid; i < n4; i += nth) {
        const float4 gv = ld4z(g, i);
        const float4 uv = ld4z(u, i);
        a = fmaxf(a, fabsf(act(gv.x, uv.x)));
        a = fmaxf(a, fabsf(act(gv.y, uv.y)));
        a = fmaxf(a, fabsf(act(gv.z, uv.z)));
        a = fmaxf(a, fabsf(act(gv.w, uv.w)));
    }
    for (uint32_t sh = 16; sh > 0; sh >>= 1)
        a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, sh));
    if ((tid & 31u) == 0) wmax[tid >> 5] = a;
    __syncthreads();
    if (tid == 0) {
        float m = 0.0f;
        for (uint32_t w = 0; w < ((nth + 31u) >> 5); ++w) m = fmaxf(m, wmax[w]);
        int e = 0;
        if (m > 0.0f) {
            int ex;
            float fr = frexpf(m, &ex);
            e = ex - 9 + (fr > 0.875f ? 1 : 0);
        }
        s_e = e;
        rscale[row] = ldexpf(1.0f, e);
    }
    __syncthreads();
    const float inv = ldexpf(1.0f, -s_e);
    unsigned char* qr = q + (size_t)row * n_ff;
    for (uint32_t i = tid; i < n4; i += nth) {
        const float4 gv = ld4z(g, i);
        const float4 uv = ld4z(u, i);
        uchar4 o;
        o.x = __nv_fp8_e4m3(act(gv.x, uv.x) * inv).__x;
        o.y = __nv_fp8_e4m3(act(gv.y, uv.y) * inv).__x;
        o.z = __nv_fp8_e4m3(act(gv.z, uv.z) * inv).__x;
        o.w = __nv_fp8_e4m3(act(gv.w, uv.w) * inv).__x;
        *(uchar4*)(qr + (size_t)i * 4u) = o;
    }
}

template <int ACT>
static inline int pd_quantize_e4m3_glu2_row_launch(const void* gu, void* q,
                                                   void* rscale, uint32_t n_ff,
                                                   uint32_t rows, uint32_t nzp,
                                                   void* stream) {
    if (rows == 0) return 0;
    if ((n_ff & 31u) || nzp == 0) return cudaErrorInvalidValue;
    {
        const uint32_t C = pd_rowq_chunks(rows);
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 900)
        // C == 8 is the only cluster-legal width (the dims are literal), and
        // pd_rowq_chunks caps at 8, so this covers every decode shape with
        // rows <= 42. Anything else keeps the two-kernel path.
        if (C == 8u && pd_glu2_fuse_on()) {
            pd_pdl_go(pd_quantize_e4m3_glu2_row2c_kernel<ACT>, dim3(rows, 8u), 256, 0u,
                      (cudaStream_t)stream, (const float*)gu, (unsigned char*)q,
                      (float*)rscale, n_ff, nzp);
            return pd_launch_status();
        }
#endif
        if (C > 1u) {
            float* scr = pd_rowq_scr();
            pd_pdl_go(pd_rowmax_glu2_part_kernel<ACT>, dim3(rows, C), 256, 0u,
                      (cudaStream_t)stream, (const float*)gu, scr, n_ff, nzp);
            pd_pdl_go(pd_quantize_e4m3_glu2_row2_kernel<ACT>, dim3(rows, C), 256, 0u,
                      (cudaStream_t)stream, (const float*)gu, (unsigned char*)q,
                      (float*)rscale, n_ff, (const float*)scr, nzp);
            return pd_launch_status();
        }
    }
    pd_pdl_go(pd_quantize_e4m3_glu2_row_kernel<ACT>, rows, 1024, 0u,
              (cudaStream_t)stream,
              (const float*)gu, (unsigned char*)q, (float*)rscale, n_ff, nzp);
    return pd_launch_status();
}

//  single-pass SMEM twin: the two-pass form below evaluates the
// SiLU (software expf, bit-parity doctrine - abi.cuh pd_glu_act) twice per
// element and re-reads gu; profiling the muse c32 wave (5984x19968) put it at
// SM 80% / DRAM 18% - compute-bound on the doubled expf, not bandwidth.
// This twin computes act once and parks the row in dynamic shared memory
// as f32 across the max reduction - same bits as a register stage, so
// BIT-IDENTICAL to the two-pass form (same per-element act value, and
// rowmax is exact under any evaluation order), but without the register
// cost: the V-register variant below needs ~51 regs/thread at the muse
// n_ff and caps co-residency at one 1024-thread CTA per SM, serializing
// the grid into 40 latency-dominated CTA waves (413us vs the
// two-pass 479 - the halved expf bought back what the lost overlap cost).
// 78KB of smem per CTA keeps two CTAs resident. Kill: PADDOCK_NO_GLU2_1P.
template <int ACT>
__global__ void __launch_bounds__(1024) pd_quantize_e4m3_glu2_row_b16_1ps_kernel(
    const __nv_bfloat16* __restrict__ gu, unsigned char* __restrict__ q,
    float* __restrict__ rscale, uint32_t n_ff) {
    PD_PDL_ARM();
    extern __shared__ float sact[];
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    const __nv_bfloat16* g = gu + (size_t)row * 2u * n_ff;
    const __nv_bfloat16* u = g + n_ff;
    auto ld4 = [&](const __nv_bfloat16* base, uint32_t i) -> float4 {
        const uint2 raw = *(const uint2*)(base + (size_t)i * 4u);
        const __nv_bfloat162 lo = *(const __nv_bfloat162*)&raw.x;
        const __nv_bfloat162 hi = *(const __nv_bfloat162*)&raw.y;
        const float2 a = __bfloat1622float2(lo), b = __bfloat1622float2(hi);
        return make_float4(a.x, a.y, b.x, b.y);
    };
    __shared__ float wmax[32];
    __shared__ int s_e;
    auto act = [](float x, float y) { return pd_glu_act<ACT>(x) * y; };
    const uint32_t n4 = n_ff >> 2;
    float a = 0.0f;
    for (uint32_t i = tid; i < n4; i += nth) {
        const float4 gv = ld4(g, i);
        const float4 uv = ld4(u, i);
        const float4 r = make_float4(act(gv.x, uv.x), act(gv.y, uv.y),
                                     act(gv.z, uv.z), act(gv.w, uv.w));
        *(float4*)(sact + (size_t)i * 4u) = r;
        a = fmaxf(a, fmaxf(fmaxf(fabsf(r.x), fabsf(r.y)),
                           fmaxf(fabsf(r.z), fabsf(r.w))));
    }
    for (uint32_t sh = 16; sh > 0; sh >>= 1)
        a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, sh));
    if ((tid & 31u) == 0) wmax[tid >> 5] = a;
    __syncthreads();
    if (tid == 0) {
        float m = 0.0f;
        for (uint32_t w = 0; w < ((nth + 31u) >> 5); ++w) m = fmaxf(m, wmax[w]);
        int e = 0;
        if (m > 0.0f) {
            int ex;
            float fr = frexpf(m, &ex);
            e = ex - 9 + (fr > 0.875f ? 1 : 0);
        }
        s_e = e;
        rscale[row] = ldexpf(1.0f, e);
    }
    __syncthreads();
    const float inv = ldexpf(1.0f, -s_e);
    unsigned char* qr = q + (size_t)row * n_ff;
    for (uint32_t i = tid; i < n4; i += nth) {
        const float4 r = *(const float4*)(sact + (size_t)i * 4u);
        uchar4 o;
        o.x = __nv_fp8_e4m3(r.x * inv).__x;
        o.y = __nv_fp8_e4m3(r.y * inv).__x;
        o.z = __nv_fp8_e4m3(r.z * inv).__x;
        o.w = __nv_fp8_e4m3(r.w * inv).__x;
        *(uchar4*)(qr + (size_t)i * 4u) = o;
    }
}

// register-staged variant of the same single-pass idea - kept for the
// n_ff band above the smem twin's window (see the launcher election)
template <int ACT, uint32_t V>
__global__ void __launch_bounds__(1024) pd_quantize_e4m3_glu2_row_b16_1p_kernel(
    const __nv_bfloat16* __restrict__ gu, unsigned char* __restrict__ q,
    float* __restrict__ rscale, uint32_t n_ff) {
    PD_PDL_ARM();
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    const __nv_bfloat16* g = gu + (size_t)row * 2u * n_ff;
    const __nv_bfloat16* u = g + n_ff;
    auto ld4 = [&](const __nv_bfloat16* base, uint32_t i) -> float4 {
        const uint2 raw = *(const uint2*)(base + (size_t)i * 4u);
        const __nv_bfloat162 lo = *(const __nv_bfloat162*)&raw.x;
        const __nv_bfloat162 hi = *(const __nv_bfloat162*)&raw.y;
        const float2 a = __bfloat1622float2(lo), b = __bfloat1622float2(hi);
        return make_float4(a.x, a.y, b.x, b.y);
    };
    __shared__ float wmax[32];
    __shared__ int s_e;
    auto act = [](float x, float y) { return pd_glu_act<ACT>(x) * y; };
    const uint32_t n4 = n_ff >> 2;
    float4 r[V];
    float a = 0.0f;
    #pragma unroll
    for (uint32_t s = 0; s < V; ++s) {
        const uint32_t i = tid + s * nth;
        if (i < n4) {
            const float4 gv = ld4(g, i);
            const float4 uv = ld4(u, i);
            r[s] = make_float4(act(gv.x, uv.x), act(gv.y, uv.y),
                               act(gv.z, uv.z), act(gv.w, uv.w));
            a = fmaxf(a, fmaxf(fmaxf(fabsf(r[s].x), fabsf(r[s].y)),
                               fmaxf(fabsf(r[s].z), fabsf(r[s].w))));
        }
    }
    for (uint32_t sh = 16; sh > 0; sh >>= 1)
        a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, sh));
    if ((tid & 31u) == 0) wmax[tid >> 5] = a;
    __syncthreads();
    if (tid == 0) {
        float m = 0.0f;
        for (uint32_t w = 0; w < ((nth + 31u) >> 5); ++w) m = fmaxf(m, wmax[w]);
        int e = 0;
        if (m > 0.0f) {
            int ex;
            float fr = frexpf(m, &ex);
            e = ex - 9 + (fr > 0.875f ? 1 : 0);
        }
        s_e = e;
        rscale[row] = ldexpf(1.0f, e);
    }
    __syncthreads();
    const float inv = ldexpf(1.0f, -s_e);
    unsigned char* qr = q + (size_t)row * n_ff;
    #pragma unroll
    for (uint32_t s = 0; s < V; ++s) {
        const uint32_t i = tid + s * nth;
        if (i < n4) {
            uchar4 o;
            o.x = __nv_fp8_e4m3(r[s].x * inv).__x;
            o.y = __nv_fp8_e4m3(r[s].y * inv).__x;
            o.z = __nv_fp8_e4m3(r[s].z * inv).__x;
            o.w = __nv_fp8_e4m3(r[s].w * inv).__x;
            *(uchar4*)(qr + (size_t)i * 4u) = o;
        }
    }
}

//  b16 twin: gu holds bf16 (the cutlass b16-D epilogue's output,
// always nz==1). Same rowmax/exponent/quantize structure as the f32
// whole-row kernel; loads 4 bf16 per step (uint2).
template <int ACT>
__global__ void pd_quantize_e4m3_glu2_row_b16_kernel(
    const __nv_bfloat16* __restrict__ gu, unsigned char* __restrict__ q,
    float* __restrict__ rscale, uint32_t n_ff) {
    PD_PDL_ARM();
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    const __nv_bfloat16* g = gu + (size_t)row * 2u * n_ff;
    const __nv_bfloat16* u = g + n_ff;
    auto ld4 = [&](const __nv_bfloat16* base, uint32_t i) -> float4 {
        const uint2 raw = *(const uint2*)(base + (size_t)i * 4u);
        const __nv_bfloat162 lo = *(const __nv_bfloat162*)&raw.x;
        const __nv_bfloat162 hi = *(const __nv_bfloat162*)&raw.y;
        const float2 a = __bfloat1622float2(lo), b = __bfloat1622float2(hi);
        return make_float4(a.x, a.y, b.x, b.y);
    };
    __shared__ float wmax[32];
    __shared__ int s_e;
    auto act = [](float x, float y) { return pd_glu_act<ACT>(x) * y; };
    const uint32_t n4 = n_ff >> 2;
    float a = 0.0f;
    for (uint32_t i = tid; i < n4; i += nth) {
        const float4 gv = ld4(g, i);
        const float4 uv = ld4(u, i);
        a = fmaxf(a, fabsf(act(gv.x, uv.x)));
        a = fmaxf(a, fabsf(act(gv.y, uv.y)));
        a = fmaxf(a, fabsf(act(gv.z, uv.z)));
        a = fmaxf(a, fabsf(act(gv.w, uv.w)));
    }
    for (uint32_t sh = 16; sh > 0; sh >>= 1)
        a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, sh));
    if ((tid & 31u) == 0) wmax[tid >> 5] = a;
    __syncthreads();
    if (tid == 0) {
        float m = 0.0f;
        for (uint32_t w = 0; w < ((nth + 31u) >> 5); ++w) m = fmaxf(m, wmax[w]);
        int e = 0;
        if (m > 0.0f) {
            int ex;
            float fr = frexpf(m, &ex);
            e = ex - 9 + (fr > 0.875f ? 1 : 0);
        }
        s_e = e;
        rscale[row] = ldexpf(1.0f, e);
    }
    __syncthreads();
    const float inv = ldexpf(1.0f, -s_e);
    unsigned char* qr = q + (size_t)row * n_ff;
    for (uint32_t i = tid; i < n4; i += nth) {
        const float4 gv = ld4(g, i);
        const float4 uv = ld4(u, i);
        uchar4 o;
        o.x = __nv_fp8_e4m3(act(gv.x, uv.x) * inv).__x;
        o.y = __nv_fp8_e4m3(act(gv.y, uv.y) * inv).__x;
        o.z = __nv_fp8_e4m3(act(gv.z, uv.z) * inv).__x;
        o.w = __nv_fp8_e4m3(act(gv.w, uv.w) * inv).__x;
        *(uchar4*)(qr + (size_t)i * 4u) = o;
    }
}

PD_EXPORT
int pd_quantize_e4m3_glu2_row_b16(const void* gu, void* q, void* rscale,
                                  uint32_t n_ff, uint32_t rows, uint32_t act,
                                  void* stream) {
    if (rows == 0) return 0;
    if (n_ff & 31u) return cudaErrorInvalidValue;
    static int no1p = -1;
    if (no1p < 0) no1p = pd_env("PADDOCK_NO_GLU2_1P") ? 1 : 0;
    // smem twin first: act row parked in dynamic shared memory, 2 CTAs/SM
    // (113KB ceiling = two CTAs inside the 228KB SM budget)
    const uint32_t smem = n_ff * 4u;
    if (!no1p && smem <= 113u * 1024u) {
        if (act == 1u) {
            static cudaError_t attr = cudaFuncSetAttribute(
                (const void*)pd_quantize_e4m3_glu2_row_b16_1ps_kernel<PD_ACT_SILU>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, 113 * 1024);
            (void)attr;
            pd_pdl_go(pd_quantize_e4m3_glu2_row_b16_1ps_kernel<PD_ACT_SILU>,
                      rows, 1024, smem, (cudaStream_t)stream,
                      (const __nv_bfloat16*)gu, (unsigned char*)q,
                      (float*)rscale, n_ff);
        } else {
            static cudaError_t attr = cudaFuncSetAttribute(
                (const void*)pd_quantize_e4m3_glu2_row_b16_1ps_kernel<PD_ACT_GELU>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, 113 * 1024);
            (void)attr;
            pd_pdl_go(pd_quantize_e4m3_glu2_row_b16_1ps_kernel<PD_ACT_GELU>,
                      rows, 1024, smem, (cudaStream_t)stream,
                      (const __nv_bfloat16*)gu, (unsigned char*)q,
                      (float*)rscale, n_ff);
        }
        return pd_launch_status();
    }
    const uint32_t need = ((n_ff >> 2) + 1023u) / 1024u;
    if (!no1p && need <= 8u) {
        // exact V=5 case: the muse n_ff=19968 wave (need=5) would round to 8
        // and carry 12 dead act registers at the 1024-thread 64-reg limit
        const uint32_t V = need <= 1u ? 1u
                         : need <= 2u ? 2u
                         : need <= 4u ? 4u
                         : need <= 5u ? 5u : 8u;
        #define PD_GLU2_1P(AA, VV)                                              \
            pd_pdl_go(pd_quantize_e4m3_glu2_row_b16_1p_kernel<AA, VV>, rows,    \
                      1024, 0u, (cudaStream_t)stream, (const __nv_bfloat16*)gu, \
                      (unsigned char*)q, (float*)rscale, n_ff)
        if (act == 1u) {
            if (V == 1u)      PD_GLU2_1P(PD_ACT_SILU, 1u);
            else if (V == 2u) PD_GLU2_1P(PD_ACT_SILU, 2u);
            else if (V == 4u) PD_GLU2_1P(PD_ACT_SILU, 4u);
            else if (V == 5u) PD_GLU2_1P(PD_ACT_SILU, 5u);
            else              PD_GLU2_1P(PD_ACT_SILU, 8u);
        } else {
            if (V == 1u)      PD_GLU2_1P(PD_ACT_GELU, 1u);
            else if (V == 2u) PD_GLU2_1P(PD_ACT_GELU, 2u);
            else if (V == 4u) PD_GLU2_1P(PD_ACT_GELU, 4u);
            else if (V == 5u) PD_GLU2_1P(PD_ACT_GELU, 5u);
            else              PD_GLU2_1P(PD_ACT_GELU, 8u);
        }
        #undef PD_GLU2_1P
        return pd_launch_status();
    }
    if (act == 1u)
        pd_pdl_go(pd_quantize_e4m3_glu2_row_b16_kernel<PD_ACT_SILU>, rows, 1024,
                  0u, (cudaStream_t)stream, (const __nv_bfloat16*)gu,
                  (unsigned char*)q, (float*)rscale, n_ff);
    else
        pd_pdl_go(pd_quantize_e4m3_glu2_row_b16_kernel<PD_ACT_GELU>, rows, 1024,
                  0u, (cudaStream_t)stream, (const __nv_bfloat16*)gu,
                  (unsigned char*)q, (float*)rscale, n_ff);
    return pd_launch_status();
}

PD_EXPORT
int pd_quantize_e4m3_geglu2_row(const void* gu, void* q, void* rscale,
                                uint32_t n_ff, uint32_t rows, void* stream) {
    return pd_quantize_e4m3_glu2_row_launch<PD_ACT_GELU>(gu, q, rscale, n_ff,
                                                         rows, 1u, stream);
}

// SiLU twin (muse-glimmer's FFN). Same kernels, same row-max exponent rule -
// only pd_glu_act's branch differs.
PD_EXPORT
int pd_quantize_e4m3_swiglu2_row(const void* gu, void* q, void* rscale,
                                 uint32_t n_ff, uint32_t rows, void* stream) {
    return pd_quantize_e4m3_glu2_row_launch<PD_ACT_SILU>(gu, q, rscale, n_ff,
                                                         rows, 1u, stream);
}

// nz-aware twin - `gu` is the fused GEMM's nz partial planes.
PD_EXPORT
int pd_quantize_e4m3_geglu2_nz(const void* gu, void* q, void* rscale,
                               uint32_t n_ff, uint32_t rows, uint32_t nzp,
                               void* stream) {
    return pd_quantize_e4m3_glu2_row_launch<PD_ACT_GELU>(gu, q, rscale, n_ff,
                                                         rows, nzp, stream);
}

PD_EXPORT
int pd_quantize_e4m3_swiglu2_nz(const void* gu, void* q, void* rscale,
                                uint32_t n_ff, uint32_t rows, uint32_t nzp,
                                void* stream) {
    return pd_quantize_e4m3_glu2_row_launch<PD_ACT_SILU>(gu, q, rscale, n_ff,
                                                         rows, nzp, stream);
}

// ---- tc5r: 2-SM rowwise prefill GEMM  -------------------------------
// The falsification chain (Acts 22-26) narrowed the prefill GEMM to this
// point: cta_group::2 mma (M=256/cluster, one issuer, two tensor pipes),
// rowwise e4m3 (the SF chain measured as pure tax), B split N-wise per the
//  placement model (each CTA stages half the Y), W via 1D bulk from
// the same tile-image planes the decode band serves, and a DEEP W ring
// (S=6, affordable only in this geometry). Harness vs the shipped tc5v:
// gu +8.1%, down +32.3%, qkv +17.7%, wo +22.4%, exact vs f64 (1e-07).
// __cluster_dims__ is a DECLARATION attribute evaluated on every device pass
// (the PD_TC5_OK body guard is not enough): nvcc hard-errors below sm_90, so
// an unguarded cluster kernel breaks the multi-arch fatbin for Ampere/Ada.
// Host pass (no __CUDA_ARCH__) keeps the declaration for the cc-gated launcher.
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 900)
template <uint32_t S, uint32_t NT, uint32_t NW = 0, uint32_t KT = 1u,
          uint32_t BB = 16384u, uint32_t O16 = 0u>
__global__ void __launch_bounds__(128) __cluster_dims__(2, 1, 1)
pd_f8t_gemm_tc5r_kt(const unsigned char* __restrict__ wtiles,
        const __grid_constant__ PdTmap ymap,
        const float* __restrict__ wrs, const float* __restrict__ xrs,
        float* __restrict__ y, uint32_t in_dim, uint32_t out_dim,
        uint32_t batch) {
#if PD_TC5_OK
    // NW (narrow arm): N = 128 instead of 256. At verify widths
    // (batch <= 128) the high 128 cluster cols are pure padding and the
    // tensor pipes pay them in full - N=128 halves the per-k-tile mma time.
    // Hardware N-split: each rank supplies NC/2 cols from its local tile,
    // so the B box (still 128 cols) fetches from col0 + t*NC + crank*(NC/2)
    // and the mma reads only its first NC/2 cols; the box stays inside the
    // callers' 256-aligned activation guarantee. Same k order - real cols
    // are bit-identical to the N=256 arm.
    constexpr uint32_t NC = NW ? NT * 128u : 2u * NT * 128u;  // cluster N
    extern __shared__ __align__(1024) unsigned char sh_[];
    // BB (b64 arm): B tile bytes. 16384 = the classic 128-col box;
    // 8192 = a pd_tmap_2d_h64 64-col box holding exactly this rank's real
    // cols under NW (the 128-col box wastes half its stream on cols the
    // narrow mma never reads). Same SW128 layout, same sdesc math (64 cols
    // = 8 core groups); the smem saved buys S=9.
    unsigned char* wt = sh_;                              // S x KT x 16KB A half
    unsigned char* yt = sh_ + S * KT * 16384u;            // S x KT x NT x BB B
    uint64_t* bfull = (uint64_t*)(yt + S * KT * NT * BB);
    uint64_t* bdone = bfull + S;
    uint64_t* bpeer = bdone + S;                          // leader: peer ready
    __shared__ uint32_t tmem_slot[1];

    const uint32_t tid = threadIdx.x;
    uint32_t crank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(crank));
    const uint32_t nk = (in_dim + 127u) / 128u;
    // K-split: grid.y carries nz. Each z streams the disjoint
    // k-tile range [kt0, kt0+cnt) and writes its own partial plane at
    // y + z*out_dim*batch (the launcher points y at `part` and runs the
    // fixed-order combine). The launcher's nz re-derivation guarantees
    // cnt >= 1 for every z; grid.y == 1 collapses to the direct path
    // bit-for-bit (kt0 = 0, cnt = nk, plane offset 0).
    const uint32_t nkz = (nk + gridDim.y - 1u) / gridDim.y;
    const uint32_t kt0 = blockIdx.y * nkz;
    const uint32_t cnt = nk - kt0 < nkz ? nk - kt0 : nkz;
    const uint32_t batch_pad = (batch + NC - 1u) / NC * NC;
    const uint32_t nct = batch_pad / NC;
    const uint32_t pair = blockIdx.x >> 1;
    // A: this CTA's 128 rows (its own tile-image stream); B: this CTA's
    // NT tiles = its interleaved half of the pair's NC cols. Placement per
    // mma j's N=256 takes low 128 from rank 0's tile j, high 128
    // from rank 1's tile j - so rank r's tile j = cluster cols j*256+r*128.
    const uint32_t row_tile = (pair / nct) * 2u + crank;
    const uint32_t row_base = row_tile * 128u;
    const uint32_t col0 = (pair % nct) * NC;              // pair col origin

    if (tid == 0) {
        #pragma unroll
        for (uint32_t s = 0; s < S; ++s) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bfull[s])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bdone[s])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bpeer[s])));
        }
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();
    asm volatile("barrier.cluster.arrive;");
    asm volatile("barrier.cluster.wait;");
    if (tid < 32)
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], 512;"
                     ::"r"((uint32_t)__cvta_generic_to_shared(tmem_slot)));
    __syncthreads();
    const uint32_t tmem = tmem_slot[0];        // D: 128 lanes x NC cols

    auto bar_wait = [&](uint64_t* bar, uint32_t parity) {
        const uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
        asm volatile("{\n\t.reg .pred p;\nW%=:\n\t"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                     "@!p bra W%=;\n\t}" ::"r"(a), "r"(parity));
    };
    auto peer_addr = [&](void* p) -> uint32_t {
        const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
        uint32_t pa;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                     : "=r"(pa) : "r"(a), "r"(crank ^ 1u));
        return pa;
    };
    // KT: each ring stage covers KT k-tiles. The n128 ship
    // shrank the mma until the per-tile sync round-trips (bfull turnaround,
    // bpeer cluster hop, dual commit) dominate every plane (per-tile
    // 0.77-1.84us vs ~0.27us of narrow mma; gu barely moved
    // when the mma halved). KT=2 halves the round-trip count at the same
    // in-flight k depth (S=3 x KT=2 = 6 tiles, same 192KB). ns stages cover
    // cnt k-tiles; the last stage may carry fewer (tcnt).
    const uint32_t ns = (cnt + KT - 1u) / KT;
    auto stage = [&](uint32_t st, uint32_t s) {           // per-CTA staging
        const uint32_t kt = kt0 + st * KT;
        const uint32_t tcnt = cnt - st * KT < KT ? cnt - st * KT : KT;
        const uint32_t m = (uint32_t)__cvta_generic_to_shared(&bfull[s]);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                     ::"r"(m), "r"(tcnt * (16384u + NT * BB)));
        // W: own row tile's k-slab run from the (row_tile, k)-major image
        // (consecutive k-tiles are contiguous: one bulk copy covers KT)
        const uint32_t wd = (uint32_t)__cvta_generic_to_shared(wt + s * KT * 16384u);
        const unsigned char* wsrc = wtiles
            + (((size_t)row_tile * nk + kt) << 14);
        asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                     " [%0], [%1], %2, [%3];" ::"r"(wd), "l"(wsrc),
                     "r"(tcnt * 16384u), "r"(m)
                     : "memory");
        // B: tcnt x NT tiles (cluster cols j*NC + crank*NC/2 per k-tile;
        // the box height rides the tensor map, so BB needs no coord change)
        const uint32_t yd = (uint32_t)__cvta_generic_to_shared(yt + s * KT * NT * BB);
        for (uint32_t i = 0; i < tcnt; ++i) {
            const int ck = (int)((kt + i) * 128u);
            #pragma unroll
            for (uint32_t t = 0; t < NT; ++t)
                asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                             " [%0], [%1, {%2, %3}], [%4];"
                             ::"r"(yd + (i * NT + t) * BB),
                             "l"(&ymap), "r"(ck),
                             "r"((int)(col0 + t * NC + crank * (NC / 2u))), "r"(m)
                             : "memory");
        }
    };

    if (tid == 32) {
        // producer: prologue + S-ago-guarded staging (the tc5v cadence);
        // absolute k-tile = kt0 + relative index (W and Y are k-addressed)
        for (uint32_t s = 0; s < S && s < ns; ++s) stage(s, s);
        for (uint32_t pf = S; pf < ns; ++pf) {
            const uint32_t ps = pf % S;
            bar_wait(&bdone[ps], ((pf - S) / S) & 1u);
            stage(pf, ps);
        }
    } else if (tid == 0 && crank == 1) {
        // odd CTA: forward slab-ready to the leader's bpeer (per STAGE)
        uint32_t fph = 0;
        for (uint32_t kt = 0; kt < ns; ++kt) {
            const uint32_t s = kt % S;
            bar_wait(&bfull[s], (fph >> s) & 1u);
            fph ^= 1u << s;
            asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];"
                         ::"r"(peer_addr(&bpeer[s])) : "memory");
        }
    } else if (tid == 0 && crank == 0) {
        // leader: waits own slab + peer-ready + the S-ago recycle, then
        // 4 kb x NT mmas of M256 x N256 (B tiles pair across the CTAs)
        uint32_t fph = 0, pph = 0, iph = 0;
        for (uint32_t kt = 0; kt < ns; ++kt) {            // kt = stage index
            const uint32_t s = kt % S;
            const uint32_t tcnt = cnt - kt * KT < KT ? cnt - kt * KT : KT;
            if (kt >= S) {
                bar_wait(&bdone[s], (iph >> s) & 1u);
                iph ^= 1u << s;
            }
            bar_wait(&bfull[s], (fph >> s) & 1u);
            fph ^= 1u << s;
            bar_wait(&bpeer[s], (pph >> s) & 1u);
            pph ^= 1u << s;
            const uint32_t w16 = (uint32_t)__cvta_generic_to_shared(wt + s * KT * 16384u) >> 4;
            const uint32_t y16 = (uint32_t)__cvta_generic_to_shared(yt + s * KT * NT * BB) >> 4;
            for (uint32_t i = 0; i < tcnt; ++i) {
            #pragma unroll
            for (uint32_t kb = 0; kb < 4u; ++kb) {
                const uint64_t ad = pd_tc5_sdesc(w16 + i * 1024u + kb * 2u);
                const uint32_t en = (kt > 0 || i > 0 || kb > 0) ? 1u : 0u;
                #pragma unroll
                for (uint32_t t = 0; t < NT; ++t) {
                    const uint64_t bd = pd_tc5_sdesc(y16 + (i * NT + t) * (BB >> 4) + kb * 2u);
                    // plain f8f6f4 ::2 idesc (probe-validated): d f32 @4,
                    // N>>3 @17, M>>4 @24 (N follows NC for the NW arm)
                    const uint32_t id = (1u << 4) | ((NC >> 3) << 17)
                        | ((256u >> 4) << 24);
                    asm volatile(
                        "{\n\t.reg .pred p;\n\t"
                        "setp.ne.b32 p, %4, 0;\n\t"
                        "tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t}"
                        ::"r"(tmem + t * 256u), "l"(ad), "l"(bd), "r"(id), "r"(en));
                }
            }
            }
            asm volatile(
                "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                ::"r"((uint32_t)__cvta_generic_to_shared(&bdone[s])));
            asm volatile(
                "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                ::"r"(peer_addr(&bdone[s])));
        }
    }
    if (tid == 0 && ns > 0) bar_wait(&bdone[(ns - 1u) % S], ((ns - 1u) / S) & 1u);
    __syncthreads();
    // rowwise epilogue: this CTA's 128 rows x NC cluster cols, scale fold
    // (per-plane fold is exact under K-split: the fold is linear in the sum)
    {
        // O16 (f8t16): bf16 y for the prefill-chunk o/down planes -
        // their consumer is rmsnorm_add_scale, whose p16 twin ships (Phase
        // 76). Halves the write + the norm's proj read. K-split never fires
        // at chunk widths (nz gate is batch<=128), so O16 output is final.
        float* const yz = y + (size_t)blockIdx.y * (size_t)out_dim * batch;
        __nv_bfloat16* const yz16 =
            (__nv_bfloat16*)y + (size_t)blockIdx.y * (size_t)out_dim * batch;
        const uint32_t warp = tid >> 5, lane = tid & 31u;
        const uint32_t row = row_base + warp * 32u + lane;
        const float ws = row < out_dim ? wrs[row] : 0.0f;
        #pragma unroll
        for (uint32_t cc = 0; cc < NC / 32u; ++cc) {
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
            if (row < out_dim) {
                #pragma unroll
                for (uint32_t j = 0; j < 32u; ++j) {
                    const uint32_t col = col0 + cc * 32u + j;
                    if (col < batch) {
                        const float v = __uint_as_float(rr[j]) * ws * xrs[col];
                        if (O16)
                            yz16[(size_t)col * out_dim + row] = __float2bfloat16(v);
                        else
                            yz[(size_t)col * out_dim + row] = v;
                    }
                }
            }
        }
    }
    __syncthreads();
    asm volatile("barrier.cluster.arrive;");
    asm volatile("barrier.cluster.wait;");
    if (tid < 32)
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, 512;" ::"r"(tmem));
#else
    (void)wtiles; (void)ymap; (void)wrs; (void)xrs; (void)y;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}
#endif  // __cluster_dims__ guard (host pass || sm_90+)

// ---- tc5s: PERSISTENT ::2 block-scale prefill GEMM  -------------
// The c32 M-large band (r 1013..1738) ran ~1.55 PF where cuBLAS and
// DeepGEMM reach 2.4+ PF at the same shapes on this die - the loss
// anatomy was wave quantization (down@r1290: 160 CTAs = 1.08 -> 2 waves,
// -46%), per-tile fill/drain/epilogue, and col padding; not the mma stream
// (nt_eff work-skipping measured flat: tail CTAs are off the critical
// path). Structure per the DeepGEMM recipe, w8-plane ABI so the fused
// norm-quant feeds it with no extra quant pass (the f8t chunk route's
// kernel wins were eaten exactly by that tax):
//   - one persistent cluster pair per 2 SMs; contiguous tile chunks, cols
//     inner (the pair's W k-slabs stay L2-hot across its col walk)
//   - cta_group::2 mma M=256 N=256 K=32, tc5r's proven operand geometry
//   - continuous S-deep ring across tiles (never drains); SF tcgen05.cp
//     issued by the mma warp right before each mma (tensor-pipe order
//     replaces tc5v's SF ring + recycle waits)
// Probed ::2 BLOCK_SCALE contract (bit-exact vs tc5v incl. random scales;
// none of this is documented for ::2):
//   - A: rank-local compact 128-row halves; D lanes land XOR-64 within
//     each rank (the epilogue relabels: lane l -> row l^64)
//   - B: global N=256, each rank staging its own canonical 128-col tile
//     (an N=128 ::2 mma splits 64/64 per rank compact - unusable here)
//   - SFA: per-rank LOCAL-M slots 0-127 (4 tmem cols)
//   - SFB: per-rank GLOBAL-N slots 0-255 (8 tmem cols; each rank needs
//     all 256 cols' scales - under half staging rank1 applied col-128's
//     scale to col 0)
// tmem: D 256 + SFA 4 + SFB 8 = 268 of 512 (no ping-pong: 2x256+12 > 512;
// the epilogue is exposed per tile - the known follow-up).
// Harness vs tc5v at r=1664 (random scales): qkv +23%, inqk +19%, down
// +40%, wo +47%, gu +8%. Kill: PADDOCK_NO_TC5S.
// SF-prefetch twin (PF=true) - built on the hazard theory, REFUTED
// as a speed lever (bit-exact 0/50M but 317 vs 309us at gate/m2871). The
// K-sweep facts stand: ~597ns/step, ~0 per-tile intercept (epilogue fully
// hidden - the tc5s header's "exposed epilogue" concern is amortized in
// practice). The pipe executes cp/mma in ISSUE ORDER regardless of operand
// hazards - re-banking SF tmem changes nothing.
// Batched-SF arm, closed: that road is
// also refuted - 3x 128x256b cps per K-pair (contract decoded: core-matrix
// walk LBO=8/SBO=16, 16B piece planes; bit-exact on every shape) runs
// -23-25% because the tensor pipe prices cps by SOURCE bytes (~8.4 B/ns
// with an ~85ns op floor) and warpx4's 4x broadcast is free hw
// amplification the batched form must materialize (4x the source bytes).
// Classic sits at both floors already (3 cps, 1.5KB/step). ~597ns =
// ~255ns SF + ~340ns mma is the FORMAT floor for per-32 scales (DeepGEMM's
// ~2.8PF = per-128 scale granularity, not reachable here). Mainloop CLOSED.
// Live lever from the same probe: S=4 ring beats S=6 (down +12-16%,
// gu/wo 0..+3%) - 143KB vs 209KB smem carveout returns L1 to the
// producer's cp.async.ca scale streams. Opt-in PADDOCK_TC5S_S4=1.
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 900)
template <uint32_t S, bool PF = false, bool O16 = false>
__global__ void __launch_bounds__(320) __cluster_dims__(2, 1, 1)
pd_f8bs_gemm_tc5s_kt(
    const __grid_constant__ PdTmap wmap, const __grid_constant__ PdTmap ymap,
    const unsigned char* __restrict__ wsc, const unsigned char* __restrict__ xsc,
    float* __restrict__ y, uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
#if PD_TC5_OK
    extern __shared__ __align__(1024) unsigned char sh_s5[];
    unsigned char* wt = sh_s5;                       // S x 16KB own-rank W rows
    unsigned char* yt = sh_s5 + S * 16384u;          // S x 16KB own-rank Y half
    unsigned char* sfa = sh_s5 + 2u * S * 16384u;    // S x 1KB (low 512B live)
    unsigned char* sfb = sfa + S * 1024u;            // S x 1KB all-256-col scales
    uint64_t* bfull  = (uint64_t*)(sfb + S * 1024u);
    uint64_t* bempty = bfull + S;
    uint64_t* bpeer  = bempty + S;
    uint64_t* tfull  = bpeer + S;
    uint64_t* tempty = tfull + 2;
    __shared__ uint32_t tmem_slot[1];

    const uint32_t tid = threadIdx.x;
    uint32_t crank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(crank));
    const uint32_t nk = in_dim >> 7;
    const uint32_t n_kb = in_dim >> 5;
    const uint32_t row_tiles = (out_dim + 127u) >> 7;
    const uint32_t row_pairs = (row_tiles + 1u) >> 1;
    const uint32_t n_cols = (batch + 255u) >> 8;
    const uint32_t T = row_pairs * n_cols;
    const uint32_t n_clusters = gridDim.x >> 1;
    const uint32_t cid = blockIdx.x >> 1;
    const uint32_t per = (T + n_clusters - 1u) / n_clusters;
    const uint32_t t0 = cid * per;
    const uint32_t t1 = t0 + per < T ? t0 + per : T;

    if (tid == 0) {
        #pragma unroll
        for (uint32_t s = 0; s < S; ++s) {
            // 33 = lane0 expect_tx arrive + 32 SF cp arrivals
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 33;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bfull[s])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bempty[s])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bpeer[s])));
        }
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                     ::"r"((uint32_t)__cvta_generic_to_shared(&tfull[0])));
        // 16 = 8 epilogue warps x 2 ranks, all arriving at the leader
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 16;"
                     ::"r"((uint32_t)__cvta_generic_to_shared(&tempty[0])));
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
    const uint32_t tm_sfa = tmem + 384u, tm_sfb = tmem + 388u;

    auto bar_wait = [&](uint64_t* bar, uint32_t parity) {
        const uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
        asm volatile("{\n\t.reg .pred p;\nW%=:\n\t"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                     "@!p bra W%=;\n\t}" ::"r"(a), "r"(parity));
    };
    auto peer_addr = [&](void* p) -> uint32_t {
        const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
        uint32_t pa;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                     : "=r"(pa) : "r"(a), "r"(crank ^ 1u));
        return pa;
    };

    if (tid < 32) {
        // producer: continuous ring across every (tile, kt)
        uint32_t n = 0, eph = 0;
        for (uint32_t t = t0; t < t1; ++t) {
            const uint32_t pair = t / n_cols;
            const uint32_t col = t % n_cols;
            const uint32_t row_base = pair * 256u + crank * 128u;
            const uint32_t cb = col * 256u + crank * 128u;   // own B half
            for (uint32_t kt = 0; kt < nk; ++kt, ++n) {
                const uint32_t s = n % S;
                if (n >= S) {
                    bar_wait(&bempty[s], (eph >> s) & 1u);
                    eph ^= 1u << s;
                }
                {
                    const uint32_t kb0 = kt * 4u;
                    unsigned char* fa = sfa + s * 1024u;
                    unsigned char* fb = sfb + s * 1024u;
                    const uint32_t c_base = col * 256u;
                    #pragma unroll
                    for (uint32_t cq = 0; cq < 4u; ++cq) {
                        const uint32_t rw = row_base + cq * 32u + tid;
                        pd_mma_cpa4p(fa + tid * 16u + cq * 4u,
                                     wsc + (size_t)rw * n_kb + kb0, rw < out_dim);
                    }
                    #pragma unroll
                    for (uint32_t c = 0; c < 8u; ++c) {
                        const uint32_t half = c >> 2, cq = c & 3u;
                        const uint32_t rc = c_base + c * 32u + tid;
                        pd_mma_cpa4p(fb + half * 512u + tid * 16u + cq * 4u,
                                     xsc + (size_t)rc * n_kb + kb0, rc < batch);
                    }
                }
                const uint32_t m = (uint32_t)__cvta_generic_to_shared(&bfull[s]);
                if (tid == 0) {
                    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 32768;"
                                 ::"r"(m));
                    const uint32_t wd = (uint32_t)__cvta_generic_to_shared(wt + s * 16384u);
                    const int ck = (int)(kt * 128u);
                    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                                 " [%0], [%1, {%2, %3}], [%4];" ::"r"(wd), "l"(&wmap),
                                 "r"(ck), "r"((int)row_base), "r"(m) : "memory");
                    const uint32_t yd = (uint32_t)__cvta_generic_to_shared(yt + s * 16384u);
                    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                                 " [%0], [%1, {%2, %3}], [%4];" ::"r"(yd), "l"(&ymap),
                                 "r"(ck), "r"((int)cb), "r"(m) : "memory");
                }
                asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];"
                             ::"r"(m) : "memory");
            }
        }
        // PDL: a persistent full-die grid has no tail waves for dependents
        // to overlap into (the -1.9% board mechanism vs tc5v). Signal as
        // soon as this CTA's staging is done - the last tile's mma+epilogue
        // still runs while dependents' prologues spin up; their data access
        // is gated by their own griddepcontrol.wait (semantics).
        if (tid == 0)
            asm volatile("griddepcontrol.launch_dependents;");
    } else if (tid == 32 && crank == 1u) {
        // rank1 watcher: forward slab-ready to the leader
        uint32_t fph = 0;
        const uint32_t total = (t1 > t0 ? t1 - t0 : 0u) * nk;
        for (uint32_t n = 0; n < total; ++n) {
            const uint32_t s = n % S;
            bar_wait(&bfull[s], (fph >> s) & 1u);
            fph ^= 1u << s;
            asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];"
                         ::"r"(peer_addr(&bpeer[s])) : "memory");
        }
    } else if (tid == 32 && crank == 0u) {
        // issuer (leader only)
        uint32_t n = 0, fph = 0, pph = 0;
        uint32_t tcount = 0;
        // shared helpers: wait slab n's staging and copy its SF planes into
        // the tmem bank `sfd` (PF alternates banks; classic uses bank 0)
        auto sf_stage = [&](uint32_t nn, uint32_t sfa_t, uint32_t sfb_t) {
            const uint32_t s = nn % S;
            bar_wait(&bfull[s], (fph >> s) & 1u);
            fph ^= 1u << s;
            bar_wait(&bpeer[s], (pph >> s) & 1u);
            pph ^= 1u << s;
            const uint32_t va = (uint32_t)__cvta_generic_to_shared(sfa + s * 1024u) >> 4;
            const uint32_t vb = (uint32_t)__cvta_generic_to_shared(sfb + s * 1024u) >> 4;
            const uint64_t da = ((uint64_t)(va & 0x3FFFu))
                              | ((uint64_t)1u << 16) | ((uint64_t)8u << 32);
            asm volatile("tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                         ::"r"(sfa_t), "l"(da) : "memory");
            #pragma unroll
            for (uint32_t hf = 0; hf < 2u; ++hf) {
                const uint64_t db = ((uint64_t)((vb + hf * 32u) & 0x3FFFu))
                                  | ((uint64_t)1u << 16) | ((uint64_t)8u << 32);
                asm volatile("tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                             ::"r"(sfb_t + hf * 4u), "l"(db) : "memory");
            }
        };
        auto mma_step = [&](uint32_t s, uint32_t kt, uint32_t sfa_t, uint32_t sfb_t) {
            const uint32_t w16 = (uint32_t)__cvta_generic_to_shared(wt + s * 16384u) >> 4;
            const uint32_t y16 = (uint32_t)__cvta_generic_to_shared(yt + s * 16384u) >> 4;
            #pragma unroll
            for (uint32_t kb = 0; kb < 4u; ++kb) {
                const uint64_t ad = pd_tc5_sdesc(w16 + kb * 2u);
                const uint64_t bd = pd_tc5_sdesc(y16 + kb * 2u);
                const uint32_t en = (kt > 0 || kb > 0) ? 1u : 0u;
                asm volatile(
                    "{\n\t.reg .pred p;\n\t"
                    "setp.ne.b32 p, %6, 0;\n\t"
                    "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X"
                    " [%0], %1, %2, %3, [%4], [%5], p;\n\t}"
                    ::"r"(tmem), "l"(ad), "l"(bd), "r"(pd_tc5s_idesc(kb)),
                      "r"(sfa_t), "r"(sfb_t), "r"(en));
            }
        };
        auto slab_release = [&](uint32_t s) {
            asm volatile(
                "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                ::"r"((uint32_t)__cvta_generic_to_shared(&bempty[s])));
            asm volatile(
                "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                ::"r"(peer_addr(&bempty[s])));
        };
        if constexpr (PF) {
            // PF software pipeline: SF cps for step n+1 issue right after
            // step n's mmas, into the other tmem bank - no operand hazard,
            // the cps execute under the mmas. SF banks: 384..395 / 396..407.
            const uint32_t total = (t1 > t0 ? t1 - t0 : 0u) * nk;
            auto bank_a = [&](uint32_t nn) { return tmem + 384u + (nn & 1u) * 12u; };
            if (total > 0)
                sf_stage(0, bank_a(0), bank_a(0) + 4u);
            for (uint32_t t = t0; t < t1; ++t, ++tcount) {
                if (tcount >= 1u)
                    bar_wait(&tempty[0], (tcount - 1u) & 1u);
                asm volatile("tcgen05.fence::after_thread_sync;");
                for (uint32_t kt = 0; kt < nk; ++kt, ++n) {
                    const uint32_t s = n % S;
                    mma_step(s, kt, bank_a(n), bank_a(n) + 4u);
                    slab_release(s);
                    if (n + 1u < total)
                        sf_stage(n + 1u, bank_a(n + 1u), bank_a(n + 1u) + 4u);
                }
                asm volatile(
                    "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                    ::"r"((uint32_t)__cvta_generic_to_shared(&tfull[0])));
                asm volatile(
                    "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                    ::"r"(peer_addr(&tfull[0])));
            }
        } else {
            for (uint32_t t = t0; t < t1; ++t, ++tcount) {
                if (tcount >= 1u)
                    bar_wait(&tempty[0], (tcount - 1u) & 1u);
                asm volatile("tcgen05.fence::after_thread_sync;");
                for (uint32_t kt = 0; kt < nk; ++kt, ++n) {
                    const uint32_t s = n % S;
                    sf_stage(n, tm_sfa, tm_sfb);
                    mma_step(s, kt, tm_sfa, tm_sfb);
                    slab_release(s);
                }
                asm volatile(
                    "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                    ::"r"((uint32_t)__cvta_generic_to_shared(&tfull[0])));
                asm volatile(
                    "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                    ::"r"(peer_addr(&tfull[0])));
            }
        }
    } else if (tid >= 64) {
        // epilogue: 8 warps, the 8 col chunks split 2-ways per 32-row band
        // (halves the serial ~0.5us tmem-ld chain per warp: +1-3% by shape)
        const uint32_t ewarp = (tid - 64u) >> 5, lane = tid & 31u;
        const uint32_t warp = ewarp & 3u, chalf = ewarp >> 2;
        uint32_t tcount = 0;
        for (uint32_t t = t0; t < t1; ++t, ++tcount) {
            bar_wait(&tfull[0], tcount & 1u);
            asm volatile("tcgen05.fence::after_thread_sync;");
            const uint32_t pair = t / n_cols;
            const uint32_t col0 = (t % n_cols) * 256u;
            // ::2 block_scale D lanes land 64-half-swapped within the rank
            const uint32_t row = pair * 256u + crank * 128u
                               + (((warp * 32u + lane) ^ 64u) & 127u);
            #pragma unroll
            for (uint32_t ci = 0; ci < 4u; ++ci) {
                const uint32_t cc = chalf * 4u + ci;
                uint32_t r[32];
                const uint32_t taddr = tmem + ((warp * 32u) << 16) + cc * 32u;
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
                    "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
                    "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
                    : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
                      "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),"=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15]),
                      "=r"(r[16]),"=r"(r[17]),"=r"(r[18]),"=r"(r[19]),"=r"(r[20]),"=r"(r[21]),"=r"(r[22]),"=r"(r[23]),
                      "=r"(r[24]),"=r"(r[25]),"=r"(r[26]),"=r"(r[27]),"=r"(r[28]),"=r"(r[29]),"=r"(r[30]),"=r"(r[31])
                    : "r"(taddr));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                if (row < out_dim) {
                    #pragma unroll
                    for (uint32_t j = 0; j < 32u; ++j) {
                        const uint32_t c = col0 + cc * 32u + j;
                        if (c < batch)
                            pd_tc5_store<O16>(y, (size_t)c * out_dim + row, r[j]);
                    }
                }
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            if (lane == 0) {
                if (crank == 0u)
                    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
                                 ::"r"((uint32_t)__cvta_generic_to_shared(&tempty[0])) : "memory");
                else
                    asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];"
                                 ::"r"(peer_addr(&tempty[0])) : "memory");
            }
        }
    }
    __syncthreads();
    asm volatile("barrier.cluster.arrive;");
    asm volatile("barrier.cluster.wait;");
    if (tid < 32)
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, 512;" ::"r"(tmem));
#else
    (void)wmap; (void)ymap; (void)wsc; (void)xsc; (void)y;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}
#endif  // tc5s __cluster_dims__ guard

