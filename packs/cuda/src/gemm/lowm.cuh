// ---- low-M dense GEMM (slot 543): the pr4266-class decode kernel ---------
//
// Port of the flashinfer PR#4266 low-M design (docs/qwen38-flash-next/
// pr4266-port-spec.md) scoped to what its regime actually needs at decode
// rows: 256 threads/CTA, TMA-fed W slabs, and a CLUSTER SPLIT-K whose peers
// publish f32 partials into rank 0's smem via st.async with mbarrier
// tx-credits - rank 0 reduces in FIXED PEER ORDER (bit-deterministic) and
// stores once. tcgen05/TMEM are deliberately not ported: at batch <= 8 the
// arithmetic is GEMV-class and the c1 trace puts the whole gap in latency,
// not MMA throughput (big planes already stream 4.8 TB/s on the GEMV band;
// the 7-15us small planes are the target, rival class 4.7us).
//
// Operands: W = the Dual plane's f16 twin, row-major [out][in]; X = f16
// [batch][in] (the f16 route's existing cast); Y = f32 [batch][out].
// Election: batch <= 8, in%128 == 0. CLUSTER LAW: first launch of any
// cluster kernel must happen on a quiet context (bench/cluster_fork_probe);
// pd_lowm_warmup() exists for the engine to call at load.
#pragma once

#if defined(PD_TC5_HOST)

#define PD_LOWM_BM 64u
#define PD_LOWM_KC 128u

// one CTA: 64 out rows x (K/split) slice, ring of 2 W slabs (64x128 f16).
// 256 thr = 4 thr per row x 64 rows; each thread carries `batch` f32 accs.
template <uint32_t NB>
__global__ void __launch_bounds__(256) pd_lowm_kernel(
        const __grid_constant__ PdTmap wmap,
        const __half* __restrict__ x, float* __restrict__ y,
        uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
#if PD_TC5_OK
    PD_PDL_ARM();
    namespace cg = cooperative_groups;
    const uint32_t rank = blockIdx.y;          // cluster rank = K slice
    const uint32_t split = gridDim.y;
    const uint32_t row0 = blockIdx.x * PD_LOWM_BM;
    const uint32_t tid = threadIdx.x;

    // K slice bounds in 128-element chunks
    const uint32_t nk = in_dim / PD_LOWM_KC;
    const uint32_t per = (nk + split - 1u) / split;
    // clamp both ends: ranks past the last chunk carry an empty slice (their
    // zero partials still ride the reduce) - unclamped c0 underflowed xcnt
    // and the staging loop walked ~4B elements (first probe run, b1 640-out).
    const uint32_t c0 = (rank * per < nk) ? rank * per : nk;
    const uint32_t c1 = (c0 + per < nk) ? (c0 + per) : nk;

    extern __shared__ __align__(1024) unsigned char pd_lowm_shm[];
    __half* sW = (__half*)pd_lowm_shm;                              // 2 x 64x128
    __half* sX = (__half*)(pd_lowm_shm + 2u * PD_LOWM_BM * PD_LOWM_KC * 2u); // per-slice X [kc][NB]
    float* mail = (float*)(sX + (size_t)per * PD_LOWM_KC * NB);      // (split-1) x 64 x NB
    uint64_t* bfull = (uint64_t*)(mail + (size_t)(split - 1u) * PD_LOWM_BM * NB);
    uint64_t* bred = bfull + 2;

    if (tid == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                     ::"r"((uint32_t)__cvta_generic_to_shared(&bfull[0])));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                     ::"r"((uint32_t)__cvta_generic_to_shared(&bfull[1])));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                     ::"r"((uint32_t)__cvta_generic_to_shared(bred)));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    // X slice: each thread strides the [c0*128, c1*128) x NB halfs
    const uint32_t xoff = c0 * PD_LOWM_KC;
    const uint32_t xcnt = (c1 - c0) * PD_LOWM_KC;
    for (uint32_t i = tid; i < xcnt * NB; i += 256u) {
        const uint32_t k = i / NB, b = i % NB;
        sX[(size_t)k * NB + b] =
            (b < batch) ? x[(size_t)b * in_dim + xoff + k] : __ushort_as_half(0);
    }
    cg::cluster_group cl = cg::this_cluster();
    cl.sync();          // barriers published + X resident before peers race

    const uint32_t r = tid >> 2;             // 0..63: my out row
    const uint32_t q = tid & 3u;             // 4-way K interleave per row
    float acc[NB];
    #pragma unroll
    for (uint32_t b = 0; b < NB; ++b) acc[b] = 0.0f;

    auto bar_wait = [&](uint64_t* bar, uint32_t parity) {
        const uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
        asm volatile("{\n\t.reg .pred p;\nW%=:\n\t"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                     "@!p bra W%=;\n\t}" ::"r"(a), "r"(parity));
    };

    // double-buffered prefetch: chunk c+1's TMA is in FLIGHT while chunk c
    // computes - the serial issue->wait->compute form measured 0.75 TB/s
    // (7us/chunk of pure latency); this is tc5g's producer discipline in
    // two-slot form.
    auto issue = [&](uint32_t c, uint32_t s) {
        const uint32_t m = (uint32_t)__cvta_generic_to_shared(&bfull[s]);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                     ::"r"(m), "r"(PD_LOWM_BM * PD_LOWM_KC * 2u));
        asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes"
                     " [%0], [%1, {0, %2, %3}], [%4];"
                     ::"r"((uint32_t)__cvta_generic_to_shared(sW + (size_t)s * PD_LOWM_BM * PD_LOWM_KC)),
                       "l"(&wmap), "r"((int)row0), "r"((int)c), "r"(m) : "memory");
    };
    if (tid == 0 && c0 < c1) {
        issue(c0, 0u);
        if (c0 + 1u < c1) issue(c0 + 1u, 1u);
    }
    uint32_t ph = 0;
    for (uint32_t c = c0; c < c1; ++c) {
        const uint32_t s = (c - c0) & 1u;
        bar_wait(&bfull[s], (ph >> s) & 1u);
        ph ^= (1u << s);
        // CONTIGUOUS k-run per thread (q owns [q*32, q*32+32)), both
        // operands as 16B vector loads: the strided-scalar form measured
        // shsb-stall 6.66 (8-way W bank conflicts + 8 scalar X loads per k).
        const __half* wr = sW + (size_t)s * PD_LOWM_BM * PD_LOWM_KC
                         + (size_t)r * PD_LOWM_KC + q * 32u;
        const uint32_t kb = (c - c0) * PD_LOWM_KC + q * 32u;
        #pragma unroll
        for (uint32_t k8 = 0; k8 < 32u; k8 += 8u) {
            float4 wv4a = *(const float4*)(wr + k8);          // 8 halfs of W
            const __half2* wh = (const __half2*)&wv4a;
            #pragma unroll
            for (uint32_t j = 0; j < 4u; ++j) {
                const float2 wf = __half22float2(wh[j]);
                const __half2* x0 = (const __half2*)(sX + (size_t)(kb + k8 + 2u * j) * NB);
                const __half2* x1 = (const __half2*)(sX + (size_t)(kb + k8 + 2u * j + 1u) * NB);
                #pragma unroll
                for (uint32_t b2 = 0; b2 < NB / 2u; ++b2) {
                    const float2 xa = __half22float2(x0[b2]);
                    const float2 xb = __half22float2(x1[b2]);
                    acc[2u * b2] = fmaf(wf.x, xa.x, acc[2u * b2]);
                    acc[2u * b2 + 1u] = fmaf(wf.x, xa.y, acc[2u * b2 + 1u]);
                    acc[2u * b2] = fmaf(wf.y, xb.x, acc[2u * b2]);
                    acc[2u * b2 + 1u] = fmaf(wf.y, xb.y, acc[2u * b2 + 1u]);
                }
            }
        }
        // slot s is free once every thread passed the compute above; issue
        // chunk c+2 into it (the +1 slot is already in flight)
        __syncthreads();
        if (tid == 0 && c + 2u < c1) issue(c + 2u, s);
    }
    // fold the 4-way interleave: threads q=1..3 pass to q=0 via shfl
    #pragma unroll
    for (uint32_t b = 0; b < NB; ++b) {
        acc[b] += __shfl_down_sync(0xffffffffu, acc[b], 1, 4);
        acc[b] += __shfl_down_sync(0xffffffffu, acc[b], 2, 4);
    }
    // q==0 threads hold row r's slice-partial for all NB cols
    if (rank != 0u) {
        if (q == 0u) {
            // LOCAL shared addresses of the mailbox slot and barrier, mapped
            // into rank 0's window with mapa (the pr4266 protocol; a generic
            // peer pointer from map_shared_rank cannot go through cvta).
            const uint32_t lm = (uint32_t)__cvta_generic_to_shared(
                mail + (size_t)(rank - 1u) * PD_LOWM_BM * NB + (size_t)r * NB);
            const uint32_t lb = (uint32_t)__cvta_generic_to_shared(bred);
            uint32_t pm, pb;
            asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(pm) : "r"(lm));
            asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(pb) : "r"(lb));
            #pragma unroll
            for (uint32_t b = 0; b < NB; b += 4u)
                asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32"
                             " [%0], {%1, %2, %3, %4}, [%5];"
                             ::"r"(pm + b * 4u),
                               "r"(__float_as_uint(acc[b])), "r"(__float_as_uint(acc[b + 1])),
                               "r"(__float_as_uint(acc[b + 2])), "r"(__float_as_uint(acc[b + 3])),
                               "r"(pb) : "memory");
        }
        cl.sync();
    } else {
        if (split > 1u) {
            if (tid == 0) {
                const uint32_t mb = (uint32_t)__cvta_generic_to_shared(bred);
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                             ::"r"(mb), "r"((split - 1u) * PD_LOWM_BM * NB * 4u));
            }
            __syncthreads();
            if (tid == 0) bar_wait(bred, 0u);
            __syncthreads();
        }
        if (q == 0u && row0 + r < out_dim) {
            #pragma unroll
            for (uint32_t b = 0; b < NB; ++b) {
                float v = acc[b];
                for (uint32_t p = 0; p + 1u < split; ++p)     // FIXED order
                    v += mail[(size_t)p * PD_LOWM_BM * NB + (size_t)r * NB + b];
                if (b < batch) y[(size_t)b * out_dim + row0 + r] = v;
            }
        }
        cl.sync();
    }
#endif  // PD_TC5_OK (gen-1 dead body: cg cluster needs sm_90+)
}

// launcher: grid (ceil(out/64), split), cluster (1, split, 1), PDL-attributed.
// split elected to fill the die without exceeding the K chunks.
static int pd_lowm_launch(const void* w16, const void* x16, void* y,
                          uint32_t in_dim, uint32_t out_dim, uint32_t batch,
                          cudaStream_t st) {
    if (batch == 0u || batch > 8u) return (int)cudaErrorInvalidValue;
    if (in_dim % PD_LOWM_KC || out_dim == 0u) return (int)cudaErrorInvalidValue;
    static const uint32_t nsm = [] {
        int d = 0, n = 0; cudaGetDevice(&d);
        cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, d);
        return (uint32_t)(n > 0 ? n : 148);
    }();
    const uint32_t nk = in_dim / PD_LOWM_KC;
    const uint32_t gx = (out_dim + PD_LOWM_BM - 1u) / PD_LOWM_BM;
    uint32_t split = gx >= 2u * nsm ? 1u : (2u * nsm + gx - 1u) / gx;
    // cap at 8 (cluster-16 launch latency dominated the small planes) and at
    // nk/2 so no rank is empty (per>=2 chunks keeps the prefetch fed).
    const uint32_t smax = nk >= 4u ? (nk / 2u < 8u ? nk / 2u : 8u) : 1u;
    for (uint32_t c : {8u, 4u, 2u, 1u})
        if (c <= split && c <= smax) { split = c; break; }
    if (split > smax) split = smax > 0u ? smax : 1u;
    if (split == 0u) split = 1u;

    CUtensorMap wm;
    {
        pd_tmap_encode_fn enc = pd_tmap_encode();
        const cuuint64_t gdim[3] = {PD_LOWM_KC * 2u, out_dim, nk};
        const cuuint64_t gstride[2] = {(cuuint64_t)in_dim * 2u, PD_LOWM_KC * 2u};
        const cuuint32_t box[3] = {PD_LOWM_KC * 2u, PD_LOWM_BM, 1u};
        const cuuint32_t estride[3] = {1u, 1u, 1u};
        if (!enc || ((uintptr_t)w16 & 15u) ||
            enc(&wm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3u, (void*)w16, gdim, gstride,
                box, estride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
            return (int)cudaErrorInvalidValue;
    }
    const uint32_t per = (nk + split - 1u) / split;
    const uint32_t smem = 2u * PD_LOWM_BM * PD_LOWM_KC * 2u
                        + per * PD_LOWM_KC * 8u * 2u
                        + (split - 1u) * PD_LOWM_BM * 8u * 4u
                        + 3u * 8u + 1024u;
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute((const void*)pd_lowm_kernel<8u>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024);
        cudaFuncSetAttribute((const void*)pd_lowm_kernel<8u>,
                             cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
        attr = true;
    }
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(gx, split, 1u);
    cfg.blockDim = dim3(256u, 1u, 1u);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = st;
    cudaLaunchAttribute at[2];
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim = {1u, split, 1u};
    at[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[1].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at;
    cfg.numAttrs = pd_pdl_off() ? 1u : 2u;
    cudaError_t e = cudaLaunchKernelEx(&cfg, pd_lowm_kernel<8u>,
                                       wm, (const __half*)x16, (float*)y,
                                       in_dim, out_dim, batch);
    return (int)e;
}

#endif  // PD_TC5_HOST


#if defined(PD_TC5_HOST)
// ---- pd_lowm5: tc5g forked onto a cluster split-K (fork-safe) -----------
// identical pipeline to pd_f16_gemm_tc5g_kernel (producer tid0 TMA ring,
// warp1 tcgen05 MMA into TMEM, 4 epilogue warps with tcgen05.ld) with three
// changes and nothing else:
//   * each CTA owns one 128-row tile; its K slice comes from the CLUSTER
//     rank (blockIdx.y), not the flags-KS U-walk;
//   * the machine-owning pd_f16ks_flags spin and red.global combine are
//     DELETED - peers st.async their epilogue registers into rank 0's smem
//     mailbox with mbarrier tx credits (the pr4266 protocol, fork-safe:
//     cluster CTAs gang-schedule, the wait is cluster-internal);
//   * rank 0 reduces in FIXED peer order and stores once (bit-stable).
// nto is fixed 16 (the b<=8 band), ncols 32.
template <uint32_t S>
__global__ void __launch_bounds__(192) pd_lowm5_kernel(
    const __grid_constant__ PdTmap wmap, const float* __restrict__ x,
    float* __restrict__ y,
    uint32_t in_dim, uint32_t out_dim, uint32_t batch, uint32_t nto) {
#if PD_TC5_OK
    extern __shared__ __align__(1024) unsigned char shL[];
    unsigned char* wt = shL;                          // S x 16KB W
    unsigned char* xt = shL + S * 16384u;             // S x nto*128 X
    float* mail = (float*)(xt + S * (size_t)(nto * 128u));   // (split-1) x 128 x nto
    const uint32_t split = gridDim.y;
    uint64_t* bfull  = (uint64_t*)(mail + (size_t)(split - 1u) * 128u * nto);
    uint64_t* bempty = bfull + S;
    uint64_t* bmma   = bempty + S;   // mma-done (1 arrival)
    uint64_t* bred   = bmma + 1;     // reduce credits
    __shared__ uint32_t tmem_slot[1];

    const uint32_t tid = threadIdx.x;
    const uint32_t rank = blockIdx.y;
    const uint32_t nk = (in_dim * 2u + 127u) / 128u;
    const uint32_t xsl = nto * 128u;
    const uint32_t t = blockIdx.x;                    // 128-row tile
    const uint32_t kbase = nk / split, krem = nk % split;
    const uint32_t ks0 = rank * kbase + (rank < krem ? rank : krem);
    const uint32_t kcnt0 = kbase + (rank < krem ? 1u : 0u);
    const uint32_t kcnt = ks0 + kcnt0 > nk ? (nk > ks0 ? nk - ks0 : 0u) : kcnt0;

    if (tid == 0) {
        #pragma unroll
        for (uint32_t sI = 0; sI < S; ++sI) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 2;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bfull[sI])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bempty[sI])));
        }
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                     ::"r"((uint32_t)__cvta_generic_to_shared(bmma)));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                     ::"r"((uint32_t)__cvta_generic_to_shared(bred)));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();
    // SPLIT cluster sync (ncu: barrier stall 19.55 dominated the kernel):
    // arrive here - the wait happens only where peer visibility is needed,
    // right before the epilogue's cross-CTA stores, so the whole mainloop
    // overlaps the other ranks' init instead of serializing on it.
    asm volatile("barrier.cluster.arrive.relaxed;");

    auto bar_wait = [&](uint64_t* bar, uint32_t parity) {
        const uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
        asm volatile("{\n\t.reg .pred p;\nW%=:\n\t"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                     "@!p bra W%=;\n\t}" ::"r"(a), "r"(parity));
    };

    if (tid == 0) {
        // producer: W TMA only (round 5b - X staging moved to the four
        // epilogue warps, which idle through the whole mainloop anyway;
        // round 5's inline staging put gather+cvt+fences on this warp's
        // critical path and cost +1-8us/launch).
        uint32_t eph = 0;
        for (uint32_t j = 0; j < kcnt; ++j) {
            const uint32_t sI = j % S;
            if (j >= S) { bar_wait(&bempty[sI], (eph >> sI) & 1u); eph ^= 1u << sI; }
            const uint32_t m = (uint32_t)__cvta_generic_to_shared(&bfull[sI]);
            const int ck = (int)(ks0 + j);
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                         ::"r"(m), "r"(16384u));
            asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes"
                         " [%0], [%1, {0, %2, %3}], [%4];"
                         ::"r"((uint32_t)__cvta_generic_to_shared(wt + sI * 16384u)),
                           "l"(&wmap), "r"((int)(t * 128u)), "r"(ck), "r"(m) : "memory");
        }
        } else if (tid >= 32 && tid < 64) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;"
                     ::"r"((uint32_t)__cvta_generic_to_shared(tmem_slot)));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
        if (tid == 32) {
            const uint32_t tmem = tmem_slot[0];
            const uint32_t id = (1u << 4) | ((nto >> 3) << 17) | ((128u >> 4) << 24);
            uint32_t fph = 0;
            for (uint32_t j = 0; j < kcnt; ++j) {
                const uint32_t sI = j % S;
                bar_wait(&bfull[sI], (fph >> sI) & 1u); fph ^= 1u << sI;
                const uint32_t w16a = (uint32_t)__cvta_generic_to_shared(wt + sI * 16384u) >> 4;
                const uint32_t x16a = (uint32_t)__cvta_generic_to_shared(xt + (size_t)sI * xsl) >> 4;
                #pragma unroll
                for (uint32_t kb = 0; kb < 4u; ++kb) {
                    const uint32_t en = (j > 0 || kb > 0) ? 1u : 0u;
                    asm volatile(
                        "{\n\t.reg .pred p;\n\t"
                        "setp.ne.b32 p, %4, 0;\n\t"
                        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}"
                        ::"r"(tmem), "l"(pd_tc5_sdesc(w16a + kb * 2u)),
                          "l"(pd_tc5_sdesc(x16a + kb * 2u)), "r"(id), "r"(en));
                }
                asm volatile(
                    "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                    ::"r"((uint32_t)__cvta_generic_to_shared(&bempty[sI])));
            }
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                ::"r"((uint32_t)__cvta_generic_to_shared(bmma)));
        }
    } else if (tid >= 64) {
        // 4 epilogue warps. Phase 1 (round 5b): stage X from F32 for every
        // chunk - 128 threads, one 16B flit each, SWIZZLE_128B placement
        // (flit g of row b lands at flit g ^ (b & 7), matching the W tmap's
        // swizzle that the MMA sdesc declares). These warps idle during the
        // mainloop otherwise; the in-walk A/B priced the old call path
        // (cast launch + host tmap encodes) at +0.44ms/tick.
        {
            PD_PDL_ARM();              // x is the prior kernel's output
            const uint32_t pI = tid - 64u;      // 0..127: (b, flit)
            const uint32_t b = pI >> 3, g = pI & 7u;
            uint32_t eph = 0;
            for (uint32_t j = 0; j < kcnt; ++j) {
                const uint32_t sI = j % S;
                if (j >= S) { bar_wait(&bempty[sI], (eph >> sI) & 1u); eph ^= 1u << sI; }
                __half* xs = (__half*)(xt + (size_t)sI * xsl);
                const uint32_t k0 = (uint32_t)(ks0 + j) * 64u;
                __half hv[8];
                if (b < batch) {
                    const float* xr = x + (size_t)b * in_dim + k0 + g * 8u;
                    #pragma unroll
                    for (uint32_t hI = 0; hI < 8u; ++hI)
                        hv[hI] = __float2half_rn(xr[hI]);
                } else {
                    #pragma unroll
                    for (uint32_t hI = 0; hI < 8u; ++hI)
                        hv[hI] = __ushort_as_half(0);
                }
                *(uint4*)(xs + (size_t)b * 64u + (size_t)(g ^ (b & 7u)) * 8u) =
                    *(const uint4*)hv;
                // generic-proxy stores -> async proxy, ordered across the
                // 128 stagers by the numbered barrier before the arrive.
                asm volatile("fence.proxy.async.shared::cta;");
                asm volatile("bar.sync 4, 128;");
                if (tid == 64)
                    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
                                 ::"r"((uint32_t)__cvta_generic_to_shared(&bfull[sI])));
            }
        }
        const uint32_t lane = tid & 31u;
        const uint32_t pw = (tid >> 5) & 3u;
        const uint32_t rl = pw * 32u + lane;
        if (tid == 64) bar_wait(bmma, 0u);
        asm volatile("bar.sync 2, 128;");
        asm volatile("tcgen05.fence::after_thread_sync;");
        const uint32_t tmem = tmem_slot[0];
        uint32_t r[16];
        const uint32_t taddr = tmem + ((pw * 32u) << 16);
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
            : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),
              "=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
              "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),
              "=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15])
            : "r"(taddr));
        asm volatile("tcgen05.wait::ld.sync.aligned;");
        const uint32_t row = t * 128u + rl;
        asm volatile("barrier.cluster.wait;");   // peers' bred init visible
        if (rank != 0u) {
            // publish 16 f32 to rank 0's mailbox slot for this rank/row
            const uint32_t lm = (uint32_t)__cvta_generic_to_shared(
                mail + (size_t)(rank - 1u) * 128u * nto + (size_t)rl * nto);
            const uint32_t lb = (uint32_t)__cvta_generic_to_shared(bred);
            uint32_t pm, pb;
            asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(pm) : "r"(lm));
            asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(pb) : "r"(lb));
            #pragma unroll
            for (uint32_t qv = 0; qv < 16u; qv += 4u)
                asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32"
                             " [%0], {%1, %2, %3, %4}, [%5];"
                             ::"r"(pm + qv * 4u), "r"(r[qv]), "r"(r[qv + 1u]),
                               "r"(r[qv + 2u]), "r"(r[qv + 3u]), "r"(pb) : "memory");
        } else {
            if (split > 1u) {
                if (tid == 64) {
                    const uint32_t mb = (uint32_t)__cvta_generic_to_shared(bred);
                    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                                 ::"r"(mb), "r"((split - 1u) * 128u * nto * 4u));
                    bar_wait(bred, 0u);
                }
                asm volatile("bar.sync 3, 128;");
            }
            if (row < out_dim) {
                #pragma unroll
                for (uint32_t qv = 0; qv < 16u; ++qv) {
                    float v = __uint_as_float(r[qv]);
                    for (uint32_t pI = 0; pI + 1u < split; ++pI)   // FIXED order
                        v += mail[(size_t)pI * 128u * nto + (size_t)rl * nto + qv];
                    if (qv < batch) y[(size_t)qv * out_dim + row] = v;
                }
            }
        }
    }
    if (tid < 64) asm volatile("barrier.cluster.wait;");
    asm volatile("barrier.cluster.arrive;");
    asm volatile("barrier.cluster.wait;");
    __syncthreads();
    if (tid >= 32 && tid < 64)
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;"
                     ::"r"(tmem_slot[0]));
#else
    (void)wmap; (void)x; (void)y; (void)in_dim; (void)out_dim; (void)batch; (void)nto;
#endif
}

// local copy of the f16 lane's 3D tmap encoder (include-order safe)
static bool pd_lowm_tmap_3d(CUtensorMap* map, const void* base, uint64_t kbytes,
                            uint64_t rows, uint32_t rows_box,
                            uint32_t kchunks = 2u) {
    pd_tmap_encode_fn enc = pd_tmap_encode();
    if (!enc || ((uintptr_t)base & 15u) || (kbytes & 127u)) return false;
    const cuuint64_t gdim[3] = {128u, rows, kbytes >> 7};
    const cuuint64_t gstride[2] = {kbytes, 128u};
    const cuuint32_t box[3] = {128u, rows_box, kchunks};
    const cuuint32_t estride[3] = {1u, 1u, 1u};
    return enc(map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3u, (void*)base, gdim, gstride,
               box, estride, CU_TENSOR_MAP_INTERLEAVE_NONE,
               CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

static int pd_lowm5_launch(const void* w16, const void* x16, void* y,
                           uint32_t in_dim, uint32_t out_dim, uint32_t batch,
                           cudaStream_t st) {
    if (batch == 0u || batch > 8u) return (int)cudaErrorInvalidValue;
    if (in_dim % 64u) return (int)cudaErrorInvalidValue;
    static const uint32_t nsm = [] {
        int d = 0, n = 0; cudaGetDevice(&d);
        cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, d);
        return (uint32_t)(n > 0 ? n : 148);
    }();
    const uint32_t nto = 16u;
    const uint32_t nk = (in_dim * 2u + 127u) / 128u;
    const uint32_t gx = (out_dim + 127u) >> 7;
    uint32_t split = gx >= 2u * nsm ? 1u : (2u * nsm + gx - 1u) / gx;
    uint32_t smax = nk >= 4u ? (nk / 2u < 8u ? nk / 2u : 8u) : 1u;
    // DIAGNOSTIC (round 6): cap the cluster split - a split-N gang must
    // co-schedule N CTAs, and beside the forked side stream that wait may
    // be the +7us/call the in-walk A/B keeps finding (probe is idle-GPU).
    { static const uint32_t s_cap = [] {
          const char* v = getenv("PADDOCK_LOWM_SMAX");
          return v ? (uint32_t)atoi(v) : 0u; }();
      if (s_cap && smax > s_cap) smax = s_cap; }
    { uint32_t pick = 1u;
      for (uint32_t c : {8u, 4u, 2u, 1u}) if (c <= split && c <= smax) { pick = c; break; }
      split = pick; }
    // W tmap CACHED per plane (round 5: 2 host encodes x 84 calls/tick of
    // the old form were real wall). Launch path is single-threaded (forked
    // walk launches from one host thread); plain statics suffice.
    struct WmEnt { const void* w; uint32_t in, out; CUtensorMap m; };
    static WmEnt* wcache = new WmEnt[512];
    static uint32_t nwc = 0;
    CUtensorMap wm; bool hit = false;
    for (uint32_t i = 0; i < nwc; ++i)
        if (wcache[i].w == w16 && wcache[i].in == in_dim && wcache[i].out == out_dim) {
            wm = wcache[i].m; hit = true; break;
        }
    if (!hit) {
        if (!pd_lowm_tmap_3d(&wm, (const __half*)w16, (uint64_t)in_dim * 2u, out_dim, 128u, 1u))
            return (int)cudaErrorInvalidValue;
        if (nwc < 512u) wcache[nwc++] = WmEnt{w16, in_dim, out_dim, wm};
    }
    // per-shape ring depth: 6 pays on the narrow-out/deep-K class (small
    // grid keeps occupancy) and hurts wide grids (S=6 regressed qkv 20->27
    // in round 2); the boundary is grid size, not K.
    const uint32_t S = (gx * split <= 128u) ? 6u : 4u;
    const uint32_t smem = S * (16384u + nto * 128u)
                        + (split - 1u) * 128u * nto * 4u + 8u * (2u * S + 2u) + 1024u;
    static bool attr = false;
    if (!attr) {
        for (auto k : {(const void*)pd_lowm5_kernel<4u>, (const void*)pd_lowm5_kernel<6u>}) {
            cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024);
            cudaFuncSetAttribute(k, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
        }
        attr = true;
    }
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(gx, split, 1u);
    cfg.blockDim = dim3(192u, 1u, 1u);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = st;
    cudaLaunchAttribute at[2];
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim = {1u, split, 1u};
    at[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[1].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at;
    cfg.numAttrs = pd_pdl_off() ? 1u : 2u;
    cudaError_t e = (S == 6u)
        ? cudaLaunchKernelEx(&cfg, pd_lowm5_kernel<6u>,
                             wm, (const float*)x16, (float*)y, in_dim, out_dim, batch, nto)
        : cudaLaunchKernelEx(&cfg, pd_lowm5_kernel<4u>,
                             wm, (const float*)x16, (float*)y, in_dim, out_dim, batch, nto);
    return (int)e;
}
#endif  // PD_TC5_HOST

// slot 543: low-M cluster GEMM (f16 W twin x F32 X -> f32 Y), batch <= 8.
// Round 5: x is the walk's f32 activation; the kernel casts while staging.
PD_EXPORT
int pd_lowm_gemm(const void* w16, const void* x16, void* y, uint32_t in_dim,
                 uint32_t out_dim, uint32_t batch, void* stream) {
#if defined(PD_TC5_HOST)
    return pd_lowm5_launch(w16, x16, y, in_dim, out_dim, batch, (cudaStream_t)stream);
#else
    (void)w16; (void)x16; (void)y; (void)in_dim; (void)out_dim; (void)batch; (void)stream;
    return (int)cudaErrorNotSupported;
#endif
}

// slot 544: cluster warmup - the first cluster launch must happen on a quiet
// context at engine load (bench/cluster_fork_probe law). Launches the kernel
// on a caller-owned 64x128 dummy so lazy module/cluster setup completes
// before any fork or capture exists.
PD_EXPORT
int pd_lowm_warmup(const void* w16_dummy, const void* x16_dummy, void* y_dummy,
                   void* stream) {
#if defined(PD_TC5_HOST)
    return pd_lowm5_launch(w16_dummy, x16_dummy, y_dummy, 128u, 128u,
                          1u, (cudaStream_t)stream);
#else
    (void)w16_dummy; (void)x16_dummy; (void)y_dummy; (void)stream;
    return (int)cudaErrorNotSupported;
#endif
}
