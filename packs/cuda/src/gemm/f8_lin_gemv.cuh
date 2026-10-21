// gemm/f8_lin_gemv.cuh - the b=1 GEMV over the tile-linear f8 boxes. Textually-included segment
// of the single pack translation unit, right after gemm/f8_lin.cuh (its box layout and PD_LINV_*
// geometry); moved out of it unchanged.

// ---- b=1 GEMV over the lin boxes ------------------------------------------
//
// Why this exists: the lin planes were built for the WIDTH GEMM, so the small
// bands kept reading the Q8_0 originals and the model carried both formats
// (~26 GB of duplicate residency on the 27B). pd_f8_gemv cannot read boxes,
// and a row-major f8w twin would cost the width GEMM ~11% (this file's header
// records the 1323-vs-1490 GB/s access-pattern fact). So the GEMV comes to
// the boxes instead.
//
// CTA (bt, z) owns a BM-row slice of row-tile rt = bt/(128/BM) and walks its
// boxes over K range [z*kpc, ...) - a contiguous byte range at box
// granularity, which is the per-CTA-sequential pattern the layout was made
// for. Three geometry facts, each measured worth ~5-8%:
//   - x is staged in WINDOWS of 48 boxes (24 KB smem), not per box: per-box
//     staging stutters the weight stream on __syncthreads.
//   - the K-split combine is a fused last-CTA ticket, not a second launch:
//     a combine launch costs a ~3 us floor, which was the whole gap at nz>1.
//     atomicInc wraps at nz-1 so the counter self-resets, and the combine
//     sums z in fixed order - same bytes every run, which serving requires.
//   - BM=64 (the layout's own contiguous half-boxes) beats BM=128 on skinny
//     planes; quarter-boxes add nothing.
// Rate on the FFN/head shapes: 1400-1519 GB/s, at or past the tuned Q8_0
// GEMV. The skinny PROJECTIONS stay on Q8 (only 40 row-tiles: 128-row box
// granularity starves the die where the Q8 kernel runs one CTA per output
// row).
#define PD_LINV_XWIN 48u
template <int THREADS, int BM>
__global__ void pd_f8lin_gemv_kernel(
        const unsigned char* __restrict__ boxes, const float* __restrict__ x,
        float* __restrict__ part, float* __restrict__ y,
        unsigned int* __restrict__ ticket, uint32_t in_dim, uint32_t out_dim,
        uint32_t kpc, uint32_t xwin) {
#if PD_LIN_DEV_OK
    const uint32_t nk = in_dim >> 7;
    const uint32_t bt = blockIdx.x, z = blockIdx.y, nz = gridDim.y;
    const uint32_t sub = 128u / BM;
    const uint32_t rt = bt / sub, slice = bt % sub;
    const uint32_t k0 = z * kpc, k1 = min(nk, k0 + kpc);
    if (k0 >= k1) return;
    extern __shared__ float xs[];   // min(PD_LINV_XWIN, kpc) boxes
    const uint32_t tid = threadIdx.x;
    // THREADS == 2*BM: two lanes per row, each taking 4 of the 8 16 B chunks.
    // THREADS == BM: one lane per row, all 8.
    const uint32_t r = (THREADS == 2 * BM) ? (tid >> 1) : tid;
    const uint32_t c_lo = (THREADS == 2 * BM) ? ((tid & 1u) * 4u) : 0u;
    const uint32_t c_n = (THREADS == 2 * BM) ? 4u : 8u;
    const uint32_t rb = slice * BM + r;
    float acc = 0.0f;
    for (uint32_t w0 = k0; w0 < k1; w0 += xwin) {
        const uint32_t w1 = min(k1, w0 + xwin);
        __syncthreads();
        for (uint32_t i = tid; i < (w1 - w0) * 32u; i += THREADS)
            *reinterpret_cast<float4*>(xs + i * 4u) =
                *reinterpret_cast<const float4*>(x + w0 * 128u + i * 4u);
        __syncthreads();
        const unsigned char* box = boxes + (size_t)(rt * nk + w0) * PD_LIN_BOX;
        for (uint32_t kt = w0; kt < w1; ++kt, box += PD_LIN_BOX) {
            const uint32_t sw = *reinterpret_cast<const uint32_t*>(
                box + PD_LIN_DATA + rb * 4u);
            const float* xw = xs + (kt - w0) * 128u;
            #pragma unroll
            for (uint32_t ci = 0; ci < c_n; ++ci) {
                const uint32_t c = c_lo + ci;
                const int4 wv = *reinterpret_cast<const int4*>(
                    box + rb * 128u + ((c ^ (rb & 7u)) * 16u));
                const __nv_fp8_e4m3* wb =
                    reinterpret_cast<const __nv_fp8_e4m3*>(&wv);
                const float* xv = xw + c * 16u;
                float s = 0.0f;
                #pragma unroll
                for (uint32_t j = 0; j < 16u; ++j) s += (float)wb[j] * xv[j];
                acc += __int_as_float(((int)((sw >> ((c >> 1) * 8u)) & 0xffu)) << 23) * s;
            }
        }
    }
    const uint32_t row = rt * 128u + rb;
    // nz==1 has no combine, so the rows are the answer and go straight to y -
    // the kernel elects the split, so the kernel owns the destination. (A
    // caller that passed part=y hid this: the engine passes real scratch, and
    // every nz==1 plane then returned stale memory. Caught in serving, not
    // by a unit test.)
    float* __restrict__ dst = (nz == 1) ? y : part;
    if (THREADS == 2 * BM) {
        acc += __shfl_xor_sync(0xffffffffu, acc, 1);
        if ((tid & 1u) == 0 && row < out_dim)
            dst[(size_t)z * out_dim + row] = acc;
    } else if (row < out_dim) {
        dst[(size_t)z * out_dim + row] = acc;
    }
    if (nz == 1) return;
    __threadfence();
    __shared__ unsigned int last;
    __syncthreads();
    if (tid == 0) last = atomicInc(&ticket[bt], nz - 1u);
    __syncthreads();
    if (last != nz - 1u) return;
    // acquire, then the other CTAs' partials from L2 (abi.cuh, PD_LAST_BLOCK_FOLD)
    __threadfence();
    for (uint32_t rr = tid; rr < BM; rr += THREADS) {
        const uint32_t o = rt * 128u + slice * BM + rr;
        if (o >= out_dim) continue;
        float v = 0.0f;
        for (uint32_t zz = 0; zz < nz; ++zz) v += __ldcg(&part[(size_t)zz * out_dim + o]);
        y[o] = v;
    }
#else
    (void)boxes; (void)x; (void)part; (void)y; (void)ticket;
    (void)in_dim; (void)out_dim; (void)kpc; (void)xwin;
#endif
}

// `part` needs nz * out_dim floats; `ticket` needs 2*ceil(out_dim/128) u32,
// ZEROED once at allocation and then owned by launches of this shape (the
// wrap value is nz, so a buffer must not be shared across differing nz).
// ticket == NULL pins nz = 1 and `part` may alias `y`.
PD_EXPORT
int pd_f8lin_gemv(const void* wlin, const void* x, void* part, void* y,
                  void* ticket, uint32_t in_dim, uint32_t out_dim,
                  void* stream) {
    if (out_dim == 0) return 0;
    if ((in_dim & 127u) || (out_dim & 15u)) return cudaErrorInvalidValue;
    static int nsm = 0;
    if (!nsm) {
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) return cudaErrorInvalidValue;
        if (cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev)
            != cudaSuccess)
            return cudaErrorInvalidValue;
    }
    const uint32_t nk = in_dim >> 7, nrt = (out_dim + 127u) / 128u;
    // BM=64 always (measured >= BM=128 on every shape that isn't starved, and
    // strictly better when starved).
    const uint32_t ctas = nrt * 2u;
    // K-split election. Three rules were measured end-to-end in serving on
    // qwen3.8-27B c1 (state-matched A/Bs) and the ranking is monotonic in
    // less splitting:
    //     CTA/SM floor (nz 1..5)   fastest   (-0.84% vs the Q8 chain)
    //     wave fill    (nz 5..7)             (-1.1%)
    //     max split    (nz = nk)             (-1.4%)
    // So the split is not free: partials land in the shared ks scratch (71 MB
    // on this model, also used by the attention/DN ks GEMMs), and every extra
    // z evicts more of the other lane's working set than the added parallelism
    // buys. Keep the split to what the die actually needs to be busy.
    //
    // NOTE this does not close the gap to the Q8 chain, and geometry is not
    // why: the Q8 GEMV overlaps its neighbours (norm 55.8
    // ms, swiglu 19.7, itself 16.2 per profiled window) and this kernel
    // overlaps nothing at any of the three geometries. That is a structural
    // property of one fused call replacing two independent ones, not a
    // tuning parameter.
    //
    // The three rules above were ranked on 188 SMs, where 544 gate|up CTAs
    // are 2.9 per SM in one wave. On 48 SMs (GB10, 2026-09-07) the same
    // launch is 11 CTAs per SM against a 9-per-SM smem ceiling (24 KB of x
    // window each) - a 1.26-wave tail - and the down plane's nz=2 is 160
    // CTAs, 3.3 per SM: both ran at 42-43% of that die's roof while the
    // lin<32> arm on the same token ran at 84%. Two dev switches so the next
    // die can be MEASURED instead of argued: PADDOCK_LINV_NZ pins the split
    // (1..16), PADDOCK_LINV_XWIN the x window in boxes (1..48; smaller =
    // more resident CTAs). Unset = the rule as shipped.
    static int env_nz = -1, env_xwin = -1;
    if (env_nz < 0) {
        const char* e = pd_env("PADDOCK_LINV_NZ");
        env_nz = e ? atoi(e) : 0;
        const char* w = pd_env("PADDOCK_LINV_XWIN");
        env_xwin = w ? atoi(w) : 0;
    }
    uint32_t nz = 1;
    if (ticket) {
        const uint32_t want = (2u * (uint32_t)nsm + ctas - 1u) / ctas;
        nz = want < 1u ? 1u : (want > 16u ? 16u : want);
        if (env_nz > 0) nz = env_nz > 16 ? 16u : (uint32_t)env_nz;
        if (nz > nk) nz = nk;
    }
    const uint32_t kpc = (nk + nz - 1u) / nz;
    // every z plane must own a non-empty K range or the ticket never fills
    const uint32_t nz_eff = (nk + kpc - 1u) / kpc;
    dim3 grid(ctas, nz_eff);
    uint32_t xwin = PD_LINV_XWIN;
    if (env_xwin > 0 && env_xwin <= (int)PD_LINV_XWIN) xwin = (uint32_t)env_xwin;
    const uint32_t win = kpc < xwin ? kpc : xwin;
    const uint32_t shm = win * 128u * (uint32_t)sizeof(float);
    pd_f8lin_gemv_kernel<128, 64><<<grid, 128, shm, (cudaStream_t)stream>>>(
        (const unsigned char*)wlin, (const float*)x, (float*)part, (float*)y,
        (unsigned int*)ticket, in_dim, out_dim, kpc, xwin);
    return pd_launch_status();
}
