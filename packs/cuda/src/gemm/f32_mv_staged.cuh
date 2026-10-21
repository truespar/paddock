// f32_mv_staged.cuh - pd_matvec_f32_batch's tile arm with K staged through shared memory.
// Textually-included segment of the single pack translation unit (needs abi.cuh's pd_env only);
// f32_qkv.cuh's launcher tries pd_mv_staged_go first.
//
// Why. The tile kernel (pd_matvec_f32_tile_kernel) walks K one L2/DRAM round trip per step: a
// lane loads w[o][i] and x[t][i], FMAs, then loads i + 32. At the qwen3.8 decode decay plane (in
// 5120, out 96 = alpha|beta over 48 heads, ~55 rows) that is 160 serial round trips a warp - 41.5
// us a layer in the B200 serve, 2.2 ms of every decode tick (no-PDL nsys, 2026-10-19), for 2 MB
// of weights. bench/matvec_f32_tile_probe.cu, weights cold: shipped 86 us; this 16 us against a
// 7 us byte floor.
//
// Same bits. A block takes 8 outputs x 8 tokens; K streams through a 3-deep cp.async ring of
// 512-wide slabs (8 w rows + 8 x rows). Warp (wo, wt) owns 4 outputs x 2 tokens and each lane
// keeps those 8 chains - per (o, t) the tile kernel's chain exactly: lane l sums i = l, l + 32,
// ... in ascending order with the same contracted FMA, then the same shfl_down tree. The probe
// checks every variant bitwise against the shipped kernel.
//
// Staging (2026-10-21): warp 0 lands a slab as 16 bulk row copies (lane r: row r, 2 KB) on the
// slot's mbarrier, not 2048 per-thread 16 B cp.asyncs with their index math (~120 instructions a
// thread a slab); one thread issuing the 16 serially was TMA-issue-bound (a ~1 us slab floor at KC
// 128). The weight rows of the first NS - 1 slabs depend on nothing upstream, so they issue BEFORE
// griddepcontrol.wait and stream during the predecessor's tail. Cold plane, x hot, B=128: 24.5 ->
// 16.4 us event-timed (the compute loop and the shuffle tree are untouched - same bits).
//
// Falsified around it (B200, depth 128, 2026-10-20; ~20 us a layer here, cold plane): the K-split
// exact-f32 tiles (pd_f32nt_ks_go, S 8..24: 34.6 .. 18.4 us, and a different sum order) - see
// bench/ba_decode_ks_probe.cu; and forking this matvec + delta_gate_ab onto the side stream
// beside the conv front (the graph branches ran concurrently, the decode-only tick stayed 19.8
// ms) - the two share SMs rather than filling idle ones.

#define PD_MVS_KC 512u
#define PD_MVS_NS 3u
#define PD_MVS_SMEM (PD_MVS_NS * 16u * PD_MVS_KC * 4u)

__global__ void __launch_bounds__(256) pd_mv_staged_kernel(const float* __restrict__ w,
                                                           const float* __restrict__ x,
                                                           float* __restrict__ out,
                                                           uint32_t in_dim, uint32_t out_dim,
                                                           uint32_t batch) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    constexpr uint32_t KC = PD_MVS_KC, NS = PD_MVS_NS, SLAB = 16u * KC;
    extern __shared__ __align__(128) float pd_mvs_sm[];
    __shared__ __align__(8) uint64_t full[NS];
    const uint32_t ob = blockIdx.x * 8u, t0 = blockIdx.y * 8u;
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t wo = (warp & 1u) * 4u, wt = (warp >> 1) * 2u;
    const uint32_t nk = in_dim / KC;
    // rows 0..7 of a slab: w rows ob..ob+7 (out % 8 == 0: all real); rows 8..8+nx-1: x rows
    // t0..; rows past the batch are never copied (their FMAs are predicated off below)
    const uint32_t nx = batch - t0 < 8u ? batch - t0 : 8u;
    if (threadIdx.x == 0) {
        for (uint32_t i = 0; i < NS; ++i)
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&full[i])));
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    // warp 0: lane 0 arms the slot for the whole slab, lane r copies row r (w rows when wx & 1,
    // x rows when wx & 2)
    auto stage = [&](uint32_t kc, uint32_t slot, uint32_t wx) {
        float* dst = pd_mvs_sm + slot * SLAB;
        const uint32_t m = (uint32_t)__cvta_generic_to_shared(&full[slot]);
        if ((wx & 1u) && lane == 0)
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                         ::"r"(m), "r"((8u + nx) * KC * 4u) : "memory");
        __syncwarp();
        const uint32_t r = lane;
        if (r < 8u ? (wx & 1u) != 0u : (wx & 2u) != 0u && r < 8u + nx) {
            const float* src = r < 8u ? w + (size_t)(ob + r) * in_dim + kc * KC
                                      : x + (size_t)(t0 + r - 8u) * in_dim + kc * KC;
            asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                         " [%0], [%1], %2, [%3];"
                         ::"r"((uint32_t)__cvta_generic_to_shared(dst + r * KC)), "l"(src),
                           "r"(KC * 4u), "r"(m) : "memory");
        }
    };
    if (warp == 0)
        for (uint32_t i = 0; i < NS - 1u && i < nk; ++i) stage(i, i, 1u);
    PD_PDL_ARM_WAIT();   // programmatic dependent (pd_pdl_go): full predecessor completion
    if (warp == 0)
        for (uint32_t i = 0; i < NS - 1u && i < nk; ++i) stage(i, i, 2u);
    float acc[4][2] = {};
    const bool v0 = t0 + wt < batch, v1 = t0 + wt + 1u < batch;
    for (uint32_t kc = 0; kc < nk; ++kc) {
        // the slot refilled here was last read at kc - 1, released by that iteration's barrier
        if (warp == 0 && kc + NS - 1u < nk) stage(kc + NS - 1u, (kc + NS - 1u) % NS, 3u);
        asm volatile("{\n\t.reg .pred p;\n\tPD_MVS_WAIT:\n\t"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                     "@!p bra PD_MVS_WAIT;\n\t}"
                     ::"r"((uint32_t)__cvta_generic_to_shared(&full[kc % NS])), "r"((kc / NS) & 1u)
                     : "memory");
        const float* sl = pd_mvs_sm + (kc % NS) * SLAB;
        #pragma unroll 4
        for (uint32_t i = lane; i < KC; i += 32u) {
            float wv[4];
            #pragma unroll
            for (uint32_t a = 0; a < 4u; ++a) wv[a] = sl[(wo + a) * KC + i];
            const float x0 = sl[(8u + wt) * KC + i], x1 = sl[(9u + wt) * KC + i];
            #pragma unroll
            for (uint32_t a = 0; a < 4u; ++a) {
                if (v0) acc[a][0] += wv[a] * x0;
                if (v1) acc[a][1] += wv[a] * x1;
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (uint32_t a = 0; a < 4u; ++a)
        #pragma unroll
        for (uint32_t b = 0; b < 2u; ++b) {
            float v = acc[a][b];
            for (uint32_t sh = 16; sh > 0; sh >>= 1)
                v += __shfl_down_sync(0xffffffffu, v, sh);
            const uint32_t o = ob + wo + a, t = t0 + wt + b;
            if (lane == 0 && o < out_dim && t < batch) out[(size_t)t * out_dim + o] = v;
        }
#else
    (void)w; (void)x; (void)out; (void)in_dim; (void)out_dim; (void)batch;
#endif
}

// The tile arm's shapes (batch >= 16, out % 8 == 0) with K in whole slabs, on cc 10 - the die
// it was measured on (bit-identical everywhere, so other dies are a perf call, not a numerics
// one). Kill: PADDOCK_NO_MV_STAGED. Returns true when it launched.
static bool pd_mv_staged_go(const void* w, const void* x, void* out, uint32_t in_dim,
                            uint32_t out_dim, uint32_t batch, cudaStream_t st) {
    static int ok = -1;
    if (ok < 0) {
        int dev = 0, cc = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&cc, cudaDevAttrComputeCapabilityMajor, dev);
        ok = cc == 10 && pd_env("PADDOCK_NO_MV_STAGED") == nullptr &&
             cudaFuncSetAttribute((const void*)pd_mv_staged_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  (int)PD_MVS_SMEM) == cudaSuccess;
    }
    if (!ok || batch < 16u || (out_dim & 7u) || in_dim % PD_MVS_KC) return false;
    pd_pdl_go(pd_mv_staged_kernel, dim3(out_dim / 8u, (batch + 7u) / 8u), 256, PD_MVS_SMEM, st,
        (const float*)w, (const float*)x, (float*)out, in_dim, out_dim, batch);
    return true;
}
