// gemm/mmq.cuh (formerly 09_mmq.cuh) - mmq quantize+GEMM family (base, hi, pipe, pipe64, split-K)
// Textually-included segment of the single pack translation unit.
// Not standalone-compilable: include order is defined by ../pack.cu.
// ---------------------------------------------------------------- mmq (P6e)
// The mmq-class Q8_0 GEMM: our MMA kernel rebuilt at llama's mul_mat_q
// design point (design studied from ggml mmq.cuh; implementation ours). The
// pieces interlock -- adopting any one alone regressed (P6d):
//   - K is staged 256 int8 deep per weight-tile load (2x the earlier kernel), the activation
//     streamed through one reused 128-int8 shared buffer in two halves, so
//     weight staging amortizes over 2x the MMAs at the same sync density.
//   - ntx=2 warp shape: warp pairs share a 32-row strip; within a pair each
//     warp takes every other 8-column group. Per k-block the B fragment is
//     reused across 2 MMAs (B shared-loads halved) while the accumulator
//     count is UNCHANGED (the P6d attempt doubled it -> register cliff).
//   - The A fragments + weight scales for a warp's 32x128-int8 slice are
//     preloaded into registers (48 regs); __launch_bounds__(256, 1) targets
//     one block/SM (llama runs 254 regs at 16.7% occupancy and 80% DRAM --
//     latency hiding comes from ILP, not occupancy).
//   - Activations arrive in the flat mmq layout ([chunk][col][4xf32 scales +
//     128 int8], pd_quantize_q8_mmq below), so the y-tile stage is a single
//     contiguous coalesced copy and scale reads stay in-tile.
// All shared traffic is 4-byte (coalescing does the work) -- that is what
// makes dynamic shared safe here where int4 staging regressed (P6d): there
// is no vectorized store for the compiler to lose across extern-shared.
// Per-block accumulation order matches the earlier kernel (k-major, f32 scale-accumulate),
// the same int8 numeric class as llama's own prefill.
#define PD_MMQ_XK 76   // x-tile row stride, int32: 64 data + 8 scales + 4 pad
#define PD_MMQ_YK 36   // y-tile col stride, int32: 4 scales + 32 data (%8==4)
#define PD_MMQ_SMEM ((128u * PD_MMQ_YK + 128u * PD_MMQ_XK) * 4u)
// High-occupancy variant of the mmq tile for the very-large-M encoder regime.
// Same 128x128 OUTPUT tile (weight L2 reuse preserved -> DRAM unchanged), but K
// is staged 128-deep instead of 256, so the weight tile is half the shared
// (32 data int32 + 4 scales + 4 pad = 40/row): tile_x 20 KB + tile_y 18 KB =
// 38 KB, so two blocks fit the 100 KB/SM budget. At the encoder's M the mmq is
// barrier-bound (profiled: __syncthreads is the top warp stall at 1 block/SM,
// where a sync idles the whole SM); a second resident block fills the gaps.
#define PD_MMQ_HI_XK 40  // x-tile row stride, int32: 32 data (128 int8) + 4 scale + 4 pad
#define PD_MMQ_HI_SMEM ((128u * PD_MMQ_YK + 128u * PD_MMQ_HI_XK) * 4u)
// Software-pipelined (2-stage cp.async, double-buffered both tiles) variant --
// the mul_mat_q approach (idea from ggml, implementation ours): at the same
// 57 KB / 1 block/SM / 254 regs as the sync mmq, cp.async keeps the next
// K-chunk in flight so __syncthreads essentially never waits (barrier stall
// drops ~6x). K staged 128-deep so two weight + two activation buffers fit
// 1 block/SM.
// Weight scales ride as f16 in shared and convert at the A-fragment read.
#define PD_MMQ_PIPE_WK 36  // weight row: 32 data int32 + 4 f16 scale (2 int32) + 2 pad
#define PD_MMQ_PIPE_SMEM ((2u * 128u * PD_MMQ_PIPE_WK + 2u * 128u * PD_MMQ_YK) * 4u)

// Activation quantize into the mmq layout. One warp per (128-value chunk,
// column): 4 values per lane, amax over 8-lane groups. Scale math identical
// to pd_quantize_q8 (max reduction is order-free; same rn + clamp), so the
// int8/scale VALUES are bit-identical to the strided layout -- only the
// placement differs. Columns are zero-padded to a multiple of 128 so the
// GEMM's flat tile copy never needs a column guard.
__global__ void pd_quantize_q8_mmq_kernel(const float* __restrict__ x,
                                          uint8_t* __restrict__ yq,
                                          uint32_t in_dim, uint32_t batch) {
    const uint32_t chunk = blockIdx.x, col = blockIdx.y, lane = threadIdx.x;
    uint8_t* blk = yq + ((size_t)chunk * gridDim.y + col) * 144u;
    const uint32_t k0 = chunk * 128u + lane * 4u;
    float v[4] = {};
    if (col < batch) {
        #pragma unroll
        for (uint32_t j = 0; j < 4u; ++j)
            if (k0 + j < in_dim) v[j] = x[(size_t)col * in_dim + k0 + j];
    }
    float a = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
    #pragma unroll
    for (uint32_t s = 4; s > 0; s >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, s));
    const float scl = a * (1.0f / 127.0f);
    const float inv = scl > 0.0f ? 1.0f / scl : 0.0f;
    char4 q;
    int qi;
    qi = __float2int_rn(v[0] * inv); q.x = (char)(qi < -127 ? -127 : (qi > 127 ? 127 : qi));
    qi = __float2int_rn(v[1] * inv); q.y = (char)(qi < -127 ? -127 : (qi > 127 ? 127 : qi));
    qi = __float2int_rn(v[2] * inv); q.z = (char)(qi < -127 ? -127 : (qi > 127 ? 127 : qi));
    qi = __float2int_rn(v[3] * inv); q.w = (char)(qi < -127 ? -127 : (qi > 127 ? 127 : qi));
    ((char4*)(blk + 16u))[lane] = q;
    if ((lane & 7u) == 0u) ((float*)blk)[lane >> 3] = scl;
}

PD_EXPORT
int pd_quantize_q8_mmq(const void* x, void* yq, uint32_t in_dim, uint32_t batch,
                       void* stream) {
    if (in_dim == 0 || batch == 0) return 0;
    const uint32_t n_chunks = (in_dim + 127u) / 128u;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    dim3 grid(n_chunks, batch_pad);
    pd_quantize_q8_mmq_kernel<<<grid, 32, 0, (cudaStream_t)stream>>>(
        (const float*)x, (uint8_t*)yq, in_dim, batch);
    return pd_launch_status();
}

// SwiGLU fused into the mmq quantize (P6j): the ffn_down input used to be
// materialized by pd_swiglu (read gate+up, write gate) and then re-read by
// the quantize - a full ff-sized round-trip per layer. Here v = silu(gate) *
// up is computed inline (bit-identical formula to pd_swiglu) and quantized
// directly; gate/up are read once and the f32 activation never lands.
__global__ void pd_quantize_q8_mmq_swiglu_kernel(const float* __restrict__ gate,
                                                 const float* __restrict__ up,
                                                 uint8_t* __restrict__ yq,
                                                 uint32_t in_dim, uint32_t batch) {
    const uint32_t chunk = blockIdx.x, col = blockIdx.y, lane = threadIdx.x;
    uint8_t* blk = yq + ((size_t)chunk * gridDim.y + col) * 144u;
    const uint32_t k0 = chunk * 128u + lane * 4u;
    float v[4] = {};
    if (col < batch) {
        #pragma unroll
        for (uint32_t j = 0; j < 4u; ++j)
            if (k0 + j < in_dim) {
                const float g = gate[(size_t)col * in_dim + k0 + j];
                const float u = up[(size_t)col * in_dim + k0 + j];
                v[j] = (g / (1.0f + expf(-g))) * u;  // == pd_swiglu
            }
    }
    float a = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
    #pragma unroll
    for (uint32_t s = 4; s > 0; s >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, s));
    const float scl = a * (1.0f / 127.0f);
    const float inv = scl > 0.0f ? 1.0f / scl : 0.0f;
    char4 q;
    int qi;
    qi = __float2int_rn(v[0] * inv); q.x = (char)(qi < -127 ? -127 : (qi > 127 ? 127 : qi));
    qi = __float2int_rn(v[1] * inv); q.y = (char)(qi < -127 ? -127 : (qi > 127 ? 127 : qi));
    qi = __float2int_rn(v[2] * inv); q.z = (char)(qi < -127 ? -127 : (qi > 127 ? 127 : qi));
    qi = __float2int_rn(v[3] * inv); q.w = (char)(qi < -127 ? -127 : (qi > 127 ? 127 : qi));
    ((char4*)(blk + 16u))[lane] = q;
    if ((lane & 7u) == 0u) ((float*)blk)[lane >> 3] = scl;
}

// Residual-add + rmsnorm + mmq quantize in one pass (P6k). The prefill
// pattern `x += proj; xn = rmsnorm(x); yq = quantize(xn)` round-trips x and
// xn through DRAM between three kernels; here the row lives in shared for
// the whole pipeline. proj == NULL skips the add (the attn_norm site);
// xn == NULL skips materializing the normalized row (legal wherever the
// quantize is its only consumer - the alpha/beta sites pass xn). Bit-exact
// with the separate kernels: the square-sum uses pd_rmsnorm_batch's exact
// reduction order (strided float4 + shfl_down + tid-0 combine + 1/sqrtf),
// the normalize applies v*inv*w in the same order, and the quantize phase
// is pd_quantize_q8_mmq's math on the same values. Rows past `batch` (the
// mmq column pad) quantize a zero row, matching pd_quantize_q8_mmq.
template <bool PB16 = false>
__global__ void pd_add_rmsnorm_quant_mmq_kernel(
        float* __restrict__ x, const float* __restrict__ proj,
        const float* __restrict__ w, float* __restrict__ xn,
        uint8_t* __restrict__ yq, uint32_t n, uint32_t batch, float eps) {
    extern __shared__ float pd_arq_row[];
    const uint32_t b = blockIdx.x;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    const uint32_t warp = tid >> 5, lane = tid & 31u;
    const uint32_t n4 = n >> 2;
    __shared__ float wsum[32];
    __shared__ float s_inv;
    float4* row4 = reinterpret_cast<float4*>(pd_arq_row);

    if (b < batch) {
        float* xb = x + (size_t)b * n;
        float4* x4 = reinterpret_cast<float4*>(xb);
        const float4* p4 = proj ? reinterpret_cast<const float4*>(proj + (size_t)b * n) : nullptr;
        float acc = 0.0f;
        for (uint32_t i = tid; i < n4; i += nth) {
            float4 v = x4[i];
            if (p4) {
                float4 p;
                if (PB16) {
                    // bf16 residual (the o16 down-GEMM epilogue): 4 x bf16
                    const __nv_bfloat162* pb =
                        (const __nv_bfloat162*)((const __nv_bfloat16*)proj + (size_t)b * n + i * 4u);
                    p.x = __bfloat162float(pb[0].x); p.y = __bfloat162float(pb[0].y);
                    p.z = __bfloat162float(pb[1].x); p.w = __bfloat162float(pb[1].y);
                } else {
                    p = p4[i];
                }
                v.x += p.x; v.y += p.y; v.z += p.z; v.w += p.w;
                x4[i] = v;  // the residual stream keeps its update
            }
            row4[i] = v;
            acc += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
        for (uint32_t s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, s);
        if (lane == 0) wsum[warp] = acc;
        __syncthreads();
        if (tid == 0) {
            float sum = 0.0f;
            const uint32_t nwarps = (nth + 31u) >> 5;
            for (uint32_t wi = 0; wi < nwarps; ++wi) sum += wsum[wi];
            s_inv = 1.0f / sqrtf(sum / (float)n + eps);
        }
        __syncthreads();
        const float inv = s_inv;
        const float4* w4 = reinterpret_cast<const float4*>(w);
        float4* xn4 = xn ? reinterpret_cast<float4*>(xn + (size_t)b * n) : nullptr;
        for (uint32_t i = tid; i < n4; i += nth) {
            float4 v = row4[i];
            const float4 wv = w4[i];
            v.x = v.x * inv * wv.x;
            v.y = v.y * inv * wv.y;
            v.z = v.z * inv * wv.z;
            v.w = v.w * inv * wv.w;
            row4[i] = v;
            if (xn4) xn4[i] = v;
        }
    } else {
        // mmq column pad: quantize a zero row (scales 0, qs 0)
        for (uint32_t i = tid; i < n4; i += nth)
            row4[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    __syncthreads();

    // quantize phase - pd_quantize_q8_mmq's exact math on the shared row;
    // this warp handles chunks [warp, n_chunks) striding by warp count
    const uint32_t n_chunks = (n + 127u) / 128u;
    for (uint32_t chunk = warp; chunk < n_chunks; chunk += nth >> 5) {
        uint8_t* blk = yq + ((size_t)chunk * gridDim.x + b) * 144u;
        const uint32_t k0 = chunk * 128u + lane * 4u;
        float v[4] = {};
        #pragma unroll
        for (uint32_t j = 0; j < 4u; ++j)
            if (k0 + j < n) v[j] = pd_arq_row[k0 + j];
        float a = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
        #pragma unroll
        for (uint32_t s = 4; s > 0; s >>= 1)
            a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, s));
        const float scl = a * (1.0f / 127.0f);
        const float invs = scl > 0.0f ? 1.0f / scl : 0.0f;
        char4 q;
        int qi;
        qi = __float2int_rn(v[0] * invs); q.x = (char)(qi < -127 ? -127 : (qi > 127 ? 127 : qi));
        qi = __float2int_rn(v[1] * invs); q.y = (char)(qi < -127 ? -127 : (qi > 127 ? 127 : qi));
        qi = __float2int_rn(v[2] * invs); q.z = (char)(qi < -127 ? -127 : (qi > 127 ? 127 : qi));
        qi = __float2int_rn(v[3] * invs); q.w = (char)(qi < -127 ? -127 : (qi > 127 ? 127 : qi));
        ((char4*)(blk + 16u))[lane] = q;
        if ((lane & 7u) == 0u) ((float*)blk)[lane >> 3] = scl;
    }
}

PD_EXPORT
int pd_add_rmsnorm_quant_mmq(void* x, const void* proj, const void* w, void* xn,
                             void* yq, uint32_t n, uint32_t batch, float eps,
                             void* stream) {
    if (n == 0 || batch == 0) return 0;
    if ((n & 3u) != 0) return cudaErrorInvalidValue;
    const uint32_t smem = n * 4u;
    if (smem > 96u * 1024u) return cudaErrorInvalidValue;
    static cudaError_t attr = cudaFuncSetAttribute(
        (const void*)pd_add_rmsnorm_quant_mmq_kernel<false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
    if (attr != cudaSuccess) return attr;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    pd_add_rmsnorm_quant_mmq_kernel<false><<<batch_pad, 256, smem, (cudaStream_t)stream>>>(
        (float*)x, (const float*)proj, (const float*)w, (float*)xn, (uint8_t*)yq,
        n, batch, eps);
    return pd_launch_status();
}

// bf16-residual twin (the o16 down-GEMM epilogue's consumer): proj is bf16.
PD_EXPORT
int pd_add_rmsnorm_quant_mmq_b16(void* x, const void* proj, const void* w, void* xn,
                                 void* yq, uint32_t n, uint32_t batch, float eps,
                                 void* stream) {
    if (n == 0 || batch == 0) return 0;
    if ((n & 3u) != 0) return cudaErrorInvalidValue;
    const uint32_t smem = n * 4u;
    if (smem > 96u * 1024u) return cudaErrorInvalidValue;
    static cudaError_t attr = cudaFuncSetAttribute(
        (const void*)pd_add_rmsnorm_quant_mmq_kernel<true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
    if (attr != cudaSuccess) return attr;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    pd_add_rmsnorm_quant_mmq_kernel<true><<<batch_pad, 256, smem, (cudaStream_t)stream>>>(
        (float*)x, (const float*)proj, (const float*)w, (float*)xn, (uint8_t*)yq,
        n, batch, eps);
    return pd_launch_status();
}

PD_EXPORT
int pd_quantize_q8_mmq_swiglu(const void* gate, const void* up, void* yq,
                              uint32_t in_dim, uint32_t batch, void* stream) {
    if (in_dim == 0 || batch == 0) return 0;
    const uint32_t n_chunks = (in_dim + 127u) / 128u;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    dim3 grid(n_chunks, batch_pad);
    pd_quantize_q8_mmq_swiglu_kernel<<<grid, 32, 0, (cudaStream_t)stream>>>(
        (const float*)gate, (const float*)up, (uint8_t*)yq, in_dim, batch);
    return pd_launch_status();
}

// GEGLU twin of the swiglu-fused quantize (gemma4's parallel-GEGLU FFN):
// v = gelu_tanh(gate) * up computed inline (bit-identical formula to
// pd_geglu) and quantized directly into the mmq layout - gate/up are read
// once and the f32 activation never lands in memory, saving the pd_geglu
// round trip (read gate+up, write gate) plus the quantize's re-read per
// FFN-down. Same scale math as pd_quantize_q8 (order-free max reduction).
__global__ void pd_quantize_q8_mmq_geglu_kernel(const float* __restrict__ gate,
                                                const float* __restrict__ up,
                                                uint8_t* __restrict__ yq,
                                                uint32_t in_dim, uint32_t batch) {
    const uint32_t chunk = blockIdx.x, col = blockIdx.y, lane = threadIdx.x;
    uint8_t* blk = yq + ((size_t)chunk * gridDim.y + col) * 144u;
    const uint32_t k0 = chunk * 128u + lane * 4u;
    float v[4] = {};
    if (col < batch) {
        #pragma unroll
        for (uint32_t j = 0; j < 4u; ++j)
            if (k0 + j < in_dim) {
                const float g = gate[(size_t)col * in_dim + k0 + j];
                const float u = up[(size_t)col * in_dim + k0 + j];
                const float gelu = 0.5f * g
                    * (1.0f + tanhf(0.79788456080286535587989211986876f * g
                                    * (1.0f + 0.044715f * g * g)));  // == pd_geglu
                v[j] = gelu * u;
            }
    }
    float a = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
    #pragma unroll
    for (uint32_t s = 4; s > 0; s >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, s));
    const float scl = a * (1.0f / 127.0f);
    const float inv = scl > 0.0f ? 1.0f / scl : 0.0f;
    char4 q;
    int qi;
    qi = __float2int_rn(v[0] * inv); q.x = (char)(qi < -127 ? -127 : (qi > 127 ? 127 : qi));
    qi = __float2int_rn(v[1] * inv); q.y = (char)(qi < -127 ? -127 : (qi > 127 ? 127 : qi));
    qi = __float2int_rn(v[2] * inv); q.z = (char)(qi < -127 ? -127 : (qi > 127 ? 127 : qi));
    qi = __float2int_rn(v[3] * inv); q.w = (char)(qi < -127 ? -127 : (qi > 127 ? 127 : qi));
    ((char4*)(blk + 16u))[lane] = q;
    if ((lane & 7u) == 0u) ((float*)blk)[lane >> 3] = scl;
}

PD_EXPORT
int pd_quantize_q8_mmq_geglu(const void* gate, const void* up, void* yq,
                             uint32_t in_dim, uint32_t batch, void* stream) {
    if (in_dim == 0 || batch == 0) return 0;
    const uint32_t n_chunks = (in_dim + 127u) / 128u;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    dim3 grid(n_chunks, batch_pad);
    pd_quantize_q8_mmq_geglu_kernel<<<grid, 32, 0, (cudaStream_t)stream>>>(
        (const float*)gate, (const float*)up, (uint8_t*)yq, in_dim, batch);
    return pd_launch_status();
}

__global__ void __launch_bounds__(256, 1) pd_q8_0_gemm_mmq_kernel(
        const int8_t* __restrict__ data, const __half* __restrict__ scale,
        const uint8_t* __restrict__ yq, const float* __restrict__ bias,
        float* __restrict__ fixup, unsigned int* __restrict__ flags,
        float* __restrict__ y, uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
#if PD_MMA_OK
    extern __shared__ int pd_mmq_sh[];
    int* tile_y = pd_mmq_sh;                    // 128 cols x 36 int32
    int* tile_x = pd_mmq_sh + 128 * PD_MMQ_YK;  // 128 rows x 76 int32

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, t = lane & 3u;
    const uint32_t i0 = (warp >> 1) * 32u;   // warp pair's 32-row strip
    const uint32_t joff = (warp & 1u) * 8u;  // which 8-col group of each 16
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t n_k32 = in_dim >> 2;      // K in int32 units
    const uint32_t n_blocks = in_dim >> 5;   // K in Q8_0 blocks
    const uint32_t n_chunks = (in_dim + 127u) / 128u;

    // Stream-k work partitioning (the llama mul_mat_q scheme, arXiv
    // 2301.03598): the flattened (tile, k-chunk) space is split evenly over
    // gridDim.x blocks, so low-tile-count launches (grid == #SMs) have no
    // wave-quantization tail. A block whose range reaches a tile's K end
    // writes dst directly; a trailing partial goes to its fixup slot and the
    // tail owner adds it in (pd_q8_0_gemm_mmq_fixup_kernel). Launched with
    // gridDim.x == ntiles this degenerates to plain tiling (kb0 == 0,
    // kb1 == nk for every block) and fixup is never touched.
    // Tile order is COLUMN-fastest: the ceil(batch/128) column tiles of one
    // weight row-strip run concurrently, so the strip is read once from DRAM
    // and the re-reads hit L2 (row-fastest order evicts it between visits).
    const uint32_t nct = batch_pad >> 7;
    const uint32_t nk = (in_dim + 255u) >> 8;              // K in 256-int8 iters
    const uint32_t total = ((out_dim + 127u) >> 7) * nct * nk;
    uint32_t kbc      = (uint32_t)((uint64_t)blockIdx.x * total / gridDim.x);
    uint32_t kbc_stop = (uint32_t)(((uint64_t)blockIdx.x + 1u) * total / gridDim.x);
    while (kbc < kbc_stop) {
    const uint32_t tile = kbc / nk;
    const uint32_t kb0 = kbc - tile * nk;
    const uint32_t kb1 = min(nk, kb0 + (kbc_stop - kbc));
    const uint32_t row_base = (tile / nct) * 128u;
    const uint32_t col_base = (tile % nct) * 128u;

    float acc[16][4] = {};
    for (uint32_t kt = kb0; kt < kb1; ++kt) {
        // stage the weight tile: 128 rows x 64 int32 (256 int8 of K) + 8
        // scales. Compile-time trip counts, fully unrolled: with one block
        // per SM there is no other block to hide global latency behind, so
        // the loads must all be in flight before the first dependent store
        // (a rolled loop serializes load->store->branch and idles the SM).
        #pragma unroll
        for (uint32_t it = 0; it < 32u; ++it) {
            const uint32_t i = it * 256u + tid;
            const uint32_t row = i >> 6, k = i & 63u, gk = kt * 64u + k;
            tile_x[row * PD_MMQ_XK + k] = (gk < n_k32 && (row_base + row) < out_dim)
                ? ((const int*)(data + (size_t)(row_base + row) * in_dim))[gk] : 0;
        }
        #pragma unroll
        for (uint32_t it = 0; it < 4u; ++it) {
            const uint32_t i = it * 256u + tid;
            const uint32_t row = i >> 3, b = i & 7u, gb = kt * 8u + b;
            ((float*)tile_x)[row * PD_MMQ_XK + 64u + b] =
                (gb < n_blocks && (row_base + row) < out_dim)
                ? __half2float(scale[(size_t)(row_base + row) * n_blocks + gb]) : 0.f;
        }

        #pragma unroll
        for (uint32_t h = 0; h < 2u; ++h) {
            // stage one 128-int8 activation chunk: a flat contiguous copy
            const uint32_t chunk = kt * 2u + h;
            const int* by = (const int*)(yq + ((size_t)chunk * batch_pad + col_base) * 144u);
            #pragma unroll
            for (uint32_t it = 0; it < 18u; ++it) {  // 128*36 == 18*256 exactly
                const uint32_t l = it * 256u + tid;
                tile_y[l] = (chunk < n_chunks) ? by[l] : 0;
            }
            __syncthreads();  // h==0 also covers the tile_x stores above

            const uint32_t k00 = h * 32u;  // x-side int32 offset of this chunk
            // preload this warp's A fragments + weight scales for the chunk
            int A[2][4][4];
            float dA[2][2][4];
            #pragma unroll
            for (uint32_t n = 0; n < 2u; ++n) {
                const uint32_t r0 = (i0 + n * 16u + g) * PD_MMQ_XK;
                const uint32_t r8 = (i0 + n * 16u + 8u + g) * PD_MMQ_XK;
                #pragma unroll
                for (uint32_t kk = 0; kk < 4u; ++kk) {
                    const uint32_t ko = k00 + kk * 8u;
                    A[n][kk][0] = tile_x[r0 + ko + t];
                    A[n][kk][1] = tile_x[r8 + ko + t];
                    A[n][kk][2] = tile_x[r0 + ko + 4u + t];
                    A[n][kk][3] = tile_x[r8 + ko + 4u + t];
                    dA[n][0][kk] = ((const float*)tile_x)[r0 + 64u + (k00 >> 3) + kk];
                    dA[n][1][kk] = ((const float*)tile_x)[r8 + 64u + (k00 >> 3) + kk];
                }
            }
            #pragma unroll
            for (uint32_t j0 = 0; j0 < 128u; j0 += 16u) {
                const uint32_t jc = j0 + joff;
                #pragma unroll
                for (uint32_t kk = 0; kk < 4u; ++kk) {
                    const uint32_t ko = kk * 8u;
                    const int b0 = tile_y[(jc + g) * PD_MMQ_YK + 4u + ko + t];
                    const int b1 = tile_y[(jc + g) * PD_MMQ_YK + 4u + ko + 4u + t];
                    const float dB0 = ((const float*)tile_y)[(jc + 2u * t) * PD_MMQ_YK + kk];
                    const float dB1 = ((const float*)tile_y)[(jc + 2u * t + 1u) * PD_MMQ_YK + kk];
                    #pragma unroll
                    for (uint32_t n = 0; n < 2u; ++n) {
                        int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                        asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                            : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                            : "r"(A[n][kk][0]), "r"(A[n][kk][1]), "r"(A[n][kk][2]),
                              "r"(A[n][kk][3]), "r"(b0), "r"(b1));
                        acc[(j0 >> 3) + n][0] += dA[n][0][kk] * dB0 * (float)d0;
                        acc[(j0 >> 3) + n][1] += dA[n][0][kk] * dB1 * (float)d1;
                        acc[(j0 >> 3) + n][2] += dA[n][1][kk] * dB0 * (float)d2;
                        acc[(j0 >> 3) + n][3] += dA[n][1][kk] * dB1 * (float)d3;
                    }
                }
            }
            __syncthreads();  // tile_y is reloaded next half / next kt
        }
    }

    if (kb1 == nk) {
        // owned the tile's K tail: store (row, tok) -> y[tok*out_dim + row].
        // bias folds here only for unsplit tiles (kb0 == 0 means this block
        // owned the whole K range); split tiles get their bias in the fixup
        // pass after the head partials, preserving the exact add order of the
        // old GEMM -> fixup -> pd_bias_add sequence (bit-exact).
        const bool bfold = bias != nullptr && kb0 == 0u;
        #pragma unroll
        for (uint32_t j0 = 0; j0 < 128u; j0 += 16u) {
            const uint32_t c0 = col_base + j0 + joff + 2u * t;
            #pragma unroll
            for (uint32_t n = 0; n < 2u; ++n) {
                const uint32_t r0 = row_base + i0 + n * 16u + g;
                const uint32_t r8 = r0 + 8u;
                const float b0f = (bfold && r0 < out_dim) ? bias[r0] : 0.0f;
                const float b8f = (bfold && r8 < out_dim) ? bias[r8] : 0.0f;
                if (r0 < out_dim) {
                    if (c0 < batch) y[(size_t)c0 * out_dim + r0] = bfold ? acc[(j0 >> 3) + n][0] + b0f : acc[(j0 >> 3) + n][0];
                    if (c0 + 1u < batch) y[(size_t)(c0 + 1u) * out_dim + r0] = bfold ? acc[(j0 >> 3) + n][1] + b0f : acc[(j0 >> 3) + n][1];
                }
                if (r8 < out_dim) {
                    if (c0 < batch) y[(size_t)c0 * out_dim + r8] = bfold ? acc[(j0 >> 3) + n][2] + b8f : acc[(j0 >> 3) + n][2];
                    if (c0 + 1u < batch) y[(size_t)(c0 + 1u) * out_dim + r8] = bfold ? acc[(j0 >> 3) + n][3] + b8f : acc[(j0 >> 3) + n][3];
                }
            }
        }
    } else {
        // trailing partial: park the whole 128x128 in this block's fixup slot
        float* fx = fixup + (size_t)blockIdx.x * 16384u;
        #pragma unroll
        for (uint32_t j0 = 0; j0 < 128u; j0 += 16u) {
            const uint32_t c0 = j0 + joff + 2u * t;
            #pragma unroll
            for (uint32_t n = 0; n < 2u; ++n) {
                const uint32_t r0 = i0 + n * 16u + g;
                fx[c0 * 128u + r0] = acc[(j0 >> 3) + n][0];
                fx[(c0 + 1u) * 128u + r0] = acc[(j0 >> 3) + n][1];
                fx[c0 * 128u + r0 + 8u] = acc[(j0 >> 3) + n][2];
                fx[(c0 + 1u) * 128u + r0 + 8u] = acc[(j0 >> 3) + n][3];
            }
        }
        if (flags) {
            // publish the park for the in-kernel fold: the bar.sync orders
            // every thread's fx stores before tid0's cumulative gl fence, so
            // an adder that acquires the flag sees the whole 128x128 slot. A
            // trailing partial only happens on the block's last tile, so
            // this fires at most once.
            __syncthreads();
            if (tid == 0) {
                __threadfence();
                atomicExch(&flags[blockIdx.x], 1u);
            }
        }
    }
    kbc += kb1 - kb0;
    }  // while (kbc < kbc_stop)

    // Deferred in-kernel fold (replaces the separate fixup launch when the
    // host passes `flags`): same adder selection and the exact backward walk
    // order of pd_q8_0_gemm_mmq_fixup_kernel - bit-exact, one launch fewer,
    // and the parked slots are still L2-hot. Deferring to after the block's
    // whole range keeps the spin short (predecessors finish around the same
    // time); grid == nsm at 1 CTA/SM makes every spin target co-resident, so
    // waiting cannot deadlock.
    if (flags != nullptr) {
        const uint32_t kbc0      = (uint32_t)((uint64_t)blockIdx.x * total / gridDim.x);
        const uint32_t kbc0_stop = (uint32_t)(((uint64_t)blockIdx.x + 1u) * total / gridDim.x);
        if (kbc0 == kbc0_stop) return;   // no work assigned
        if (kbc0 % nk == 0u) return;     // started my first tile myself: no head owed
        const uint32_t tile0 = kbc0 / nk;
        // never reached my first tile's K end -> a later block owns the add
        if (tile0 == kbc0_stop / nk && kbc0_stop % nk != 0u) return;

        float sum[64] = {};
        uint32_t bidx = blockIdx.x;
        uint32_t bstop = bidx;
        while (bidx > 0u) {
            --bidx;
            const uint32_t pk   = (uint32_t)((uint64_t)bidx * total / gridDim.x);
            const uint32_t pkst = (uint32_t)(((uint64_t)bidx + 1u) * total / gridDim.x);
            if (pk == pkst) continue;    // empty block, keep walking
            if (tid == 0) {
                while (atomicAdd(&flags[bidx], 0u) == 0u) __nanosleep(64);
            }
            __syncthreads();             // flag observed -> parked slot visible
            #pragma unroll
            for (uint32_t l = 0; l < 64u; ++l)
                sum[l] += fixup[(size_t)bidx * 16384u + l * 256u + tid];
            bstop = bidx;
            if (pk % nk == 0u || pk / nk < tile0) break;
        }

        const uint32_t rb = (tile0 / nct) * 128u;
        const uint32_t cb = (tile0 % nct) * 128u;
        #pragma unroll
        for (uint32_t l = 0; l < 64u; ++l) {
            const uint32_t idx = l * 256u + tid, r = idx & 127u, c = idx >> 7;
            if (rb + r < out_dim && cb + c < batch) {
                float v = y[(size_t)(cb + c) * out_dim + rb + r];
                v += sum[l];
                if (bias) v += bias[rb + r];
                y[(size_t)(cb + c) * out_dim + rb + r] = v;
            }
        }
        // self-clean the consumed flags for the next launch (each parked
        // slot has exactly one adder). The sums above are consumed before
        // the barrier; resets can't be observed early within this launch.
        __syncthreads();
        if (tid == 0)
            for (uint32_t b2 = bstop; b2 < blockIdx.x; ++b2) flags[b2] = 0u;
    }
#else
    (void)data; (void)scale; (void)yq; (void)bias; (void)fixup; (void)flags; (void)y;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}

// Stream-k fixup: for each tile whose K range was split across blocks, the
// block that owned the tile's TAIL (it wrote dst directly, minus the head
// contributions) walks backward over the preceding blocks' fixup slots and
// adds them into dst. Exactly one adder per split tile, fixed walk order -
// deterministic. Blocks that owned whole tiles exit immediately.
__global__ void pd_q8_0_gemm_mmq_fixup_kernel(
        const float* __restrict__ fixup, const float* __restrict__ bias,
        float* __restrict__ y, uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t nct = batch_pad >> 7;
    const uint32_t nk = (in_dim + 255u) >> 8;
    const uint32_t total = ((out_dim + 127u) >> 7) * nct * nk;
    const uint32_t tid = threadIdx.x;
    const uint32_t kbc0      = (uint32_t)((uint64_t)blockIdx.x * total / gridDim.x);
    const uint32_t kbc0_stop = (uint32_t)(((uint64_t)blockIdx.x + 1u) * total / gridDim.x);
    if (kbc0 == kbc0_stop) return;   // no work assigned
    if (kbc0 % nk == 0u) return;     // started my first tile myself: no head owed
    const uint32_t tile = kbc0 / nk;
    // never reached my first tile's K end -> a later block owns the add
    if (tile == kbc0_stop / nk && kbc0_stop % nk != 0u) return;

    float sum[64] = {};
    uint32_t bidx = blockIdx.x;
    while (bidx > 0u) {
        --bidx;
        const uint32_t kbc  = (uint32_t)((uint64_t)bidx * total / gridDim.x);
        const uint32_t stop = (uint32_t)(((uint64_t)bidx + 1u) * total / gridDim.x);
        if (kbc == stop) continue;   // empty block, keep walking
        #pragma unroll
        for (uint32_t l = 0; l < 64u; ++l)
            sum[l] += fixup[(size_t)bidx * 16384u + l * 256u + tid];
        // stop at the block that owned the tile's start (or spilled in from
        // an earlier tile - its trailing partial was my tile's head)
        if (kbc % nk == 0u || kbc / nk < tile) break;
    }

    const uint32_t row_base = (tile / nct) * 128u;
    const uint32_t col_base = (tile % nct) * 128u;
    #pragma unroll
    for (uint32_t l = 0; l < 64u; ++l) {
        const uint32_t idx = l * 256u + tid, r = idx & 127u, c = idx >> 7;
        if (row_base + r < out_dim && col_base + c < batch) {
            // split-tile bias lands here, as a separate add after the head
            // sum - the exact sequence of the old fixup + pd_bias_add pair
            float v = y[(size_t)(col_base + c) * out_dim + row_base + r];
            v += sum[l];
            if (bias) v += bias[row_base + r];
            y[(size_t)(col_base + c) * out_dim + row_base + r] = v;
        }
    }
}

static int pd_q8_0_gemm_mmq_impl(const void* data, const void* scale, const void* yq,
                                 const void* bias, void* fixup, void* y, uint32_t in_dim,
                                 uint32_t out_dim, uint32_t batch, void* stream) {
    if (out_dim == 0 || batch == 0) return 0;
    // Same reasoning as pd_q8_0_gemm_mma: in_dim must be Q8_0-block-aligned
    // (format invariant, 32 elements/block); out_dim has no such requirement
    // - the weight-tile staging zero-pads rows past out_dim (line ~394,
    // `(row_base + row) < out_dim`) and the writeback bounds-checks every
    // row individually, so a ragged last row-tile is already handled. The
    // historical `out_dim & 15u` reject was over-conservative (verified by
    // tests/gemm_ragged_out_dim.rs - laguna S-2.1's g_proj is out_dim=72,
    // not 16-aligned).
    if (in_dim & 31u) return cudaErrorInvalidValue;
    // 57344 B of dynamic shared > the default 48 KB window: opt in once.
    static cudaError_t attr = cudaFuncSetAttribute(
        (const void*)pd_q8_0_gemm_mmq_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)PD_MMQ_SMEM);
    if (attr != cudaSuccess) return attr;
    static int nsm = 0;
    if (nsm == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
        if (nsm <= 0) nsm = 1;
    }
    const uint32_t ntiles = ((out_dim + 127u) / 128u) * ((batch + 127u) / 128u);
    const uint32_t waves = (ntiles + (uint32_t)nsm - 1u) / (uint32_t)nsm;
    // >= 90% tile efficiency (llama's threshold): plain tiling, skip the
    // fixup pass entirely (and stay bit-exact with the mma route). Stream-k
    // needs a fixup buffer of >= nsm * 128 * 128 floats; callers that pass
    // NULL always get tiling. 256 SMs is the fixup-buffer sizing contract.
    const bool tiled = fixup == nullptr || (uint32_t)nsm > 256u ||
                       100u * ntiles >= 90u * waves * (uint32_t)nsm;
    const uint32_t gridx = tiled ? ntiles : (uint32_t)nsm;
    // In-kernel deferred fold (default): the parked-partial adds ride the
    // GEMM kernel's tail behind per-block flags at fixup + 256*16384 floats
    // (the buffer sizing contract grew by 256: 256 tiles + the flag words,
    // zeroed at alloc, self-cleaning). PADDOCK_NO_SK_FOLD=1 pins the old
    // separate fixup launch for A/B.
    static int fold = -1;
    if (fold < 0) {
        const char* e = pd_env("PADDOCK_NO_SK_FOLD");
        fold = (e && e[0] == '1') ? 0 : 1;
    }
    unsigned int* flags =
        (!tiled && fold) ? (unsigned int*)((float*)fixup + (size_t)256u * 16384u) : nullptr;
    pd_q8_0_gemm_mmq_kernel<<<gridx, 256, PD_MMQ_SMEM, (cudaStream_t)stream>>>(
        (const int8_t*)data, (const __half*)scale, (const uint8_t*)yq,
        (const float*)bias, (float*)fixup, flags, (float*)y, in_dim, out_dim, batch);
    if (!tiled && flags == nullptr) {
        pd_q8_0_gemm_mmq_fixup_kernel<<<gridx, 256, 0, (cudaStream_t)stream>>>(
            (const float*)fixup, (const float*)bias, (float*)y, in_dim, out_dim, batch);
    }
    return pd_launch_status();
}

PD_EXPORT
int pd_q8_0_gemm_mmq(const void* data, const void* scale, const void* yq,
                     void* fixup, void* y, uint32_t in_dim, uint32_t out_dim,
                     uint32_t batch, void* stream) {
    return pd_q8_0_gemm_mmq_impl(data, scale, yq, nullptr, fixup, y, in_dim, out_dim,
                                 batch, stream);
}

// Bias-carrying variant for the serving b>64 dense rung: unsplit tiles fold
// bias in the GEMM store, split tiles in the fixup - bit-exact vs the old
// GEMM -> fixup -> pd_bias_add sequence either way.
PD_EXPORT
int pd_q8_0_gemm_mmq_b(const void* data, const void* scale, const void* yq,
                       const void* bias, void* fixup, void* y, uint32_t in_dim,
                       uint32_t out_dim, uint32_t batch, void* stream) {
    return pd_q8_0_gemm_mmq_impl(data, scale, yq, bias, fixup, y, in_dim, out_dim,
                                 batch, stream);
}

// -------------------------------------------------- mmq_hi (E1c large-M variant)
// High-occupancy sibling of pd_q8_0_gemm_mmq for the encoder's very-large-M
// prefill (batch = thousands of rows). Keeps the full 128x128 OUTPUT tile (so
// weight L2 reuse -- and DRAM -- are unchanged; the half-COLUMN tile went DRAM-
// bound at 94%), but stages K 128-deep instead of 256, halving the weight tile's
// shared footprint (40 int32/row): tile_x 20 KB + tile_y 18 KB = 38 KB, so
// __launch_bounds__(256, 2) lands two resident blocks (33% occupancy). At the
// encoder's M the mmq is BARRIER-bound (profiled: __syncthreads is the top warp
// stall, because at 1 block/SM a sync idles the whole SM); a second block fills the
// barrier gaps. TILED only (batch > 1024 => ntiles >> #SMs). Numerically the
// mmq body staged in 128-K chunks (same data/MMAs/per-k32 fold) -> identical
// Q8_0 int8 numeric class (greedy/cos gates hold).
__global__ void __launch_bounds__(256, 2) pd_q8_0_gemm_mmq_hi_kernel(
        const int8_t* __restrict__ data, const __half* __restrict__ scale,
        const uint8_t* __restrict__ yq, float* __restrict__ y,
        uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
#if PD_MMA_OK
    extern __shared__ int pd_mmq_hi_sh[];
    int* tile_y = pd_mmq_hi_sh;                       // 128 cols x 36 int32
    int* tile_x = pd_mmq_hi_sh + 128 * PD_MMQ_YK;     // 128 rows x 40 int32

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, t = lane & 3u;
    const uint32_t i0 = (warp >> 1) * 32u;   // warp pair's 32-row strip
    const uint32_t joff = (warp & 1u) * 8u;  // which 8-col group of each 16
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t n_k32 = in_dim >> 2;
    const uint32_t n_blocks = in_dim >> 5;
    const uint32_t n_chunks = (in_dim + 127u) / 128u;  // 128-K chunks
    const uint32_t nct = batch_pad >> 7;               // 128-col tiles

    const uint32_t tile = blockIdx.x;
    const uint32_t row_base = (tile / nct) * 128u;
    const uint32_t col_base = (tile % nct) * 128u;

    float acc[16][4] = {};
    for (uint32_t kc = 0; kc < n_chunks; ++kc) {
        // stage weight: 128 rows x 32 int32 (128 int8 of K) + 4 scales
        #pragma unroll
        for (uint32_t it = 0; it < 16u; ++it) {  // 16*256 == 128 rows * 32 int32
            const uint32_t i = it * 256u + tid;
            const uint32_t row = i >> 5, k = i & 31u, gk = kc * 32u + k;
            tile_x[row * PD_MMQ_HI_XK + k] = (gk < n_k32 && (row_base + row) < out_dim)
                ? ((const int*)(data + (size_t)(row_base + row) * in_dim))[gk] : 0;
        }
        #pragma unroll
        for (uint32_t it = 0; it < 2u; ++it) {  // 2*256 == 128 rows * 4 scales
            const uint32_t i = it * 256u + tid;
            const uint32_t row = i >> 2, b = i & 3u, gb = kc * 4u + b;
            ((float*)tile_x)[row * PD_MMQ_HI_XK + 32u + b] =
                (gb < n_blocks && (row_base + row) < out_dim)
                ? __half2float(scale[(size_t)(row_base + row) * n_blocks + gb]) : 0.f;
        }
        // stage activation: one 128-K chunk (128 cols x 36 int32)
        const int* by = (const int*)(yq + ((size_t)kc * batch_pad + col_base) * 144u);
        #pragma unroll
        for (uint32_t it = 0; it < 18u; ++it) {  // 128*36 == 18*256 exactly
            const uint32_t l = it * 256u + tid;
            tile_y[l] = by[l];
        }
        __syncthreads();

        int A[2][4][4];
        float dA[2][2][4];
        #pragma unroll
        for (uint32_t n = 0; n < 2u; ++n) {
            const uint32_t r0 = (i0 + n * 16u + g) * PD_MMQ_HI_XK;
            const uint32_t r8 = (i0 + n * 16u + 8u + g) * PD_MMQ_HI_XK;
            #pragma unroll
            for (uint32_t kk = 0; kk < 4u; ++kk) {
                const uint32_t ko = kk * 8u;
                A[n][kk][0] = tile_x[r0 + ko + t];
                A[n][kk][1] = tile_x[r8 + ko + t];
                A[n][kk][2] = tile_x[r0 + ko + 4u + t];
                A[n][kk][3] = tile_x[r8 + ko + 4u + t];
                dA[n][0][kk] = ((const float*)tile_x)[r0 + 32u + kk];
                dA[n][1][kk] = ((const float*)tile_x)[r8 + 32u + kk];
            }
        }
        #pragma unroll
        for (uint32_t j0 = 0; j0 < 128u; j0 += 16u) {
            const uint32_t jc = j0 + joff;
            #pragma unroll
            for (uint32_t kk = 0; kk < 4u; ++kk) {
                const uint32_t ko = kk * 8u;
                const int b0 = tile_y[(jc + g) * PD_MMQ_YK + 4u + ko + t];
                const int b1 = tile_y[(jc + g) * PD_MMQ_YK + 4u + ko + 4u + t];
                const float dB0 = ((const float*)tile_y)[(jc + 2u * t) * PD_MMQ_YK + kk];
                const float dB1 = ((const float*)tile_y)[(jc + 2u * t + 1u) * PD_MMQ_YK + kk];
                #pragma unroll
                for (uint32_t n = 0; n < 2u; ++n) {
                    int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                        : "r"(A[n][kk][0]), "r"(A[n][kk][1]), "r"(A[n][kk][2]),
                          "r"(A[n][kk][3]), "r"(b0), "r"(b1));
                    acc[(j0 >> 3) + n][0] += dA[n][0][kk] * dB0 * (float)d0;
                    acc[(j0 >> 3) + n][1] += dA[n][0][kk] * dB1 * (float)d1;
                    acc[(j0 >> 3) + n][2] += dA[n][1][kk] * dB0 * (float)d2;
                    acc[(j0 >> 3) + n][3] += dA[n][1][kk] * dB1 * (float)d3;
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (uint32_t j0 = 0; j0 < 128u; j0 += 16u) {
        const uint32_t c0 = col_base + j0 + joff + 2u * t;
        #pragma unroll
        for (uint32_t n = 0; n < 2u; ++n) {
            const uint32_t r0 = row_base + i0 + n * 16u + g;
            const uint32_t r8 = r0 + 8u;
            if (r0 < out_dim) {
                if (c0 < batch) y[(size_t)c0 * out_dim + r0] = acc[(j0 >> 3) + n][0];
                if (c0 + 1u < batch) y[(size_t)(c0 + 1u) * out_dim + r0] = acc[(j0 >> 3) + n][1];
            }
            if (r8 < out_dim) {
                if (c0 < batch) y[(size_t)c0 * out_dim + r8] = acc[(j0 >> 3) + n][2];
                if (c0 + 1u < batch) y[(size_t)(c0 + 1u) * out_dim + r8] = acc[(j0 >> 3) + n][3];
            }
        }
    }
#else
    (void)data; (void)scale; (void)yq; (void)y;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}

PD_EXPORT
int pd_q8_0_gemm_mmq_hi(const void* data, const void* scale, const void* yq,
                        void* y, uint32_t in_dim, uint32_t out_dim,
                        uint32_t batch, void* stream) {
    if (out_dim == 0 || batch == 0) return 0;
    // out_dim needs no alignment - same ragged-tolerant staging/writeback as
    // pd_q8_0_gemm_mmq (relaxation empirically verified).
    if (in_dim & 31u) return cudaErrorInvalidValue;
    static cudaError_t attr = cudaFuncSetAttribute(
        (const void*)pd_q8_0_gemm_mmq_hi_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)PD_MMQ_HI_SMEM);
    if (attr != cudaSuccess) return attr;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t nct = batch_pad >> 7;
    const uint32_t ntiles = ((out_dim + 127u) / 128u) * nct;
    pd_q8_0_gemm_mmq_hi_kernel<<<ntiles, 256, PD_MMQ_HI_SMEM, (cudaStream_t)stream>>>(
        (const int8_t*)data, (const __half*)scale, (const uint8_t*)yq,
        (float*)y, in_dim, out_dim, batch);
    return pd_launch_status();
}

// ------------------------------------------------ mmq_pipe (E1c large-M variant)
// The llama mul_mat_q approach, found by profiling llama's kernel: it runs at the
// same 57 KB / 1 block/SM / 16.6% occupancy as our sync mmq, but its barrier
// stall is 0.20 vs our 1.16 because a 2-STAGE cp.async PIPELINE keeps the next
// K-chunk's tiles in flight, so __syncthreads never waits on a load. This kernel
// double-buffers both the weight and activation tiles (K staged 128-deep so two
// of each fit 1 block/SM at 72 KB) and prefetches chunk kc+1 while computing kc.
// Weight scales ride as f16 in shared (cp.async'd 8B) and convert at the A read.
// TILED only (batch > 1024). in_dim only needs %32: the final partial 128-chunk
// is K-padded (weight tail masked to 0, activation zero-quantized), so K=2880
// rides bit-exactly. Same Q8_0 int8 numeric class as mmq (identical per-k32 fold).

// 16-byte and 8-byte global->shared async copies (pd_cp_async16 is defined later).
__device__ __forceinline__ void pd_cpa16p(void* smem, const void* gmem, bool ok) {
#if PD_MMA_OK
    const unsigned sm = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(sm), "l"(gmem),
                 "r"(ok ? 16u : 0u));
#endif
}
__device__ __forceinline__ void pd_cpa8p(void* smem, const void* gmem, bool ok) {
#if PD_MMA_OK
    const unsigned sm = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;" ::"r"(sm), "l"(gmem),
                 "r"(ok ? 8u : 0u));
#endif
}
// 4-byte (2 f16) predicated copy - used to split the scale fetch so the final
// partial K-chunk (K-padded in_dim, e.g. 2880 -> 23 chunks) never over-reads
// past the scale row for the last output row.
__device__ __forceinline__ void pd_cpa4p(void* smem, const void* gmem, bool ok) {
#if PD_MMA_OK
    const unsigned sm = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(sm), "l"(gmem),
                 "r"(ok ? 4u : 0u));
#endif
}

// Issue one 128-K weight chunk (data via 16B cp.async, 4 f16 scales via 8B) into wbuf.
__device__ __forceinline__ void pd_mmqp_issue_w(
    int* __restrict__ wbuf, const int8_t* __restrict__ data, const __half* __restrict__ scale,
    uint32_t row_base, uint32_t out_dim, uint32_t n_k32, uint32_t n_blocks,
    uint32_t in_dim, uint32_t kc, uint32_t tid) {
#if PD_MMA_OK
    #pragma unroll
    for (uint32_t it = 0; it < 4u; ++it) {  // 4*256 == 128 rows * 8 segments
        const uint32_t i = it * 256u + tid;
        const uint32_t row = i >> 3, seg = i & 7u, gk4 = kc * 32u + seg * 4u;
        const bool ok = gk4 < n_k32 && (row_base + row) < out_dim;
        pd_cpa16p(wbuf + row * PD_MMQ_PIPE_WK + seg * 4u,
                  (const char*)data + (size_t)(row_base + row) * in_dim
                      + (size_t)(kc * 128u + seg * 16u), ok);
    }
    if (tid < 128u) {  // 4 f16 scales/row (8 bytes), split 2+2 for the K-pad tail
        const uint32_t gb = kc * 4u;
        const bool row_ok = (row_base + tid) < out_dim;
        char* dst = (char*)(wbuf + tid * PD_MMQ_PIPE_WK + 32u);
        const char* src = (const char*)(scale + (size_t)(row_base + tid) * n_blocks + gb);
        // blocks gb,gb+1 then gb+2,gb+3; when in_dim isn't a multiple of 128 the
        // last chunk's high blocks (the zero-data pad) are masked off, so the
        // read stops at the true row end. Pad-block scales are never used (their
        // weight data is 0 -> 0 mma contribution), only kept memory-safe.
        pd_cpa4p(dst, src, row_ok && (gb + 1u < n_blocks));
        pd_cpa4p(dst + 4u, src + 4u, row_ok && (gb + 3u < n_blocks));
    }
#endif
}

__global__ void __launch_bounds__(256, 1) pd_q8_0_gemm_mmq_pipe_kernel(
        const int8_t* __restrict__ data, const __half* __restrict__ scale,
        const uint8_t* __restrict__ yq, const float* __restrict__ bias,
        float* __restrict__ y,
        uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
#if PD_MMA_OK
    extern __shared__ int pd_mmqp_sh[];
    int* wbuf0 = pd_mmqp_sh;
    int* wbuf1 = wbuf0 + 128 * PD_MMQ_PIPE_WK;
    int* ybuf0 = wbuf1 + 128 * PD_MMQ_PIPE_WK;
    int* ybuf1 = ybuf0 + 128 * PD_MMQ_YK;

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, t = lane & 3u;
    const uint32_t i0 = (warp >> 1) * 32u;
    const uint32_t joff = (warp & 1u) * 8u;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t n_k32 = in_dim >> 2;
    const uint32_t n_blocks = in_dim >> 5;
    const uint32_t n_chunks = (in_dim + 127u) >> 7;  // ceil: K-pad the last partial chunk
    const uint32_t nct = batch_pad >> 7;

    const uint32_t tile = blockIdx.x;
    const uint32_t row_base = (tile / nct) * 128u;
    const uint32_t col_base = (tile % nct) * 128u;

    float acc[16][4] = {};
    // prologue: chunk 0 flies into buffer 0
    pd_mmqp_issue_w(wbuf0, data, scale, row_base, out_dim, n_k32, n_blocks, in_dim, 0u, tid);
    {
        const int* by0 = (const int*)(yq + ((size_t)0u * batch_pad + col_base) * 144u);
        #pragma unroll
        for (uint32_t it = 0; it < 5u; ++it)
            if (it * 256u + tid < 1152u)
                pd_cpa16p(ybuf0 + (it * 256u + tid) * 4u,
                          (const char*)by0 + (size_t)(it * 256u + tid) * 16u, true);
    }
    asm volatile("cp.async.commit_group;");

    for (uint32_t kc = 0; kc < n_chunks; ++kc) {
        int* tw = (kc & 1u) ? wbuf1 : wbuf0;
        int* ty = (kc & 1u) ? ybuf1 : ybuf0;
        if (kc + 1u < n_chunks) {  // prefetch chunk kc+1 into the other buffers
            int* nw = (kc & 1u) ? wbuf0 : wbuf1;
            int* ny = (kc & 1u) ? ybuf0 : ybuf1;
            pd_mmqp_issue_w(nw, data, scale, row_base, out_dim, n_k32, n_blocks, in_dim, kc + 1u, tid);
            const int* by1 = (const int*)(yq + ((size_t)(kc + 1u) * batch_pad + col_base) * 144u);
            #pragma unroll
            for (uint32_t it = 0; it < 5u; ++it)
                if (it * 256u + tid < 1152u)
                    pd_cpa16p(ny + (it * 256u + tid) * 4u,
                              (const char*)by1 + (size_t)(it * 256u + tid) * 16u, true);
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 1;");  // this chunk done; kc+1 stays in flight
        } else {
            asm volatile("cp.async.wait_group 0;");  // drain: wait for the last chunk
        }
        __syncthreads();

        int A[2][4][4];
        float dA[2][2][4];
        #pragma unroll
        for (uint32_t n = 0; n < 2u; ++n) {
            const uint32_t r0 = (i0 + n * 16u + g) * PD_MMQ_PIPE_WK;
            const uint32_t r8 = (i0 + n * 16u + 8u + g) * PD_MMQ_PIPE_WK;
            #pragma unroll
            for (uint32_t kk = 0; kk < 4u; ++kk) {
                const uint32_t ko = kk * 8u;
                A[n][kk][0] = tw[r0 + ko + t];
                A[n][kk][1] = tw[r8 + ko + t];
                A[n][kk][2] = tw[r0 + ko + 4u + t];
                A[n][kk][3] = tw[r8 + ko + 4u + t];
                dA[n][0][kk] = __half2float(((const __half*)(tw + r0 + 32u))[kk]);
                dA[n][1][kk] = __half2float(((const __half*)(tw + r8 + 32u))[kk]);
            }
        }
        #pragma unroll
        for (uint32_t j0 = 0; j0 < 128u; j0 += 16u) {
            const uint32_t jc = j0 + joff;
            #pragma unroll
            for (uint32_t kk = 0; kk < 4u; ++kk) {
                const uint32_t ko = kk * 8u;
                const int b0 = ty[(jc + g) * PD_MMQ_YK + 4u + ko + t];
                const int b1 = ty[(jc + g) * PD_MMQ_YK + 4u + ko + 4u + t];
                const float dB0 = ((const float*)ty)[(jc + 2u * t) * PD_MMQ_YK + kk];
                const float dB1 = ((const float*)ty)[(jc + 2u * t + 1u) * PD_MMQ_YK + kk];
                #pragma unroll
                for (uint32_t n = 0; n < 2u; ++n) {
                    int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                        : "r"(A[n][kk][0]), "r"(A[n][kk][1]), "r"(A[n][kk][2]),
                          "r"(A[n][kk][3]), "r"(b0), "r"(b1));
                    acc[(j0 >> 3) + n][0] += dA[n][0][kk] * dB0 * (float)d0;
                    acc[(j0 >> 3) + n][1] += dA[n][0][kk] * dB1 * (float)d1;
                    acc[(j0 >> 3) + n][2] += dA[n][1][kk] * dB0 * (float)d2;
                    acc[(j0 >> 3) + n][3] += dA[n][1][kk] * dB1 * (float)d3;
                }
            }
        }
        __syncthreads();  // this buffer free before chunk kc+2 prefetches into it
    }

    #pragma unroll
    for (uint32_t j0 = 0; j0 < 128u; j0 += 16u) {
        const uint32_t c0 = col_base + j0 + joff + 2u * t;
        #pragma unroll
        for (uint32_t n = 0; n < 2u; ++n) {
            const uint32_t r0 = row_base + i0 + n * 16u + g;
            const uint32_t r8 = r0 + 8u;
            // fold the per-output-row bias here (kills the separate bias_add
            // pass that cost ~4-6us/GEMM and erased the pipe's win on wo)
            const float b0 = (bias && r0 < out_dim) ? bias[r0] : 0.0f;
            const float b8 = (bias && r8 < out_dim) ? bias[r8] : 0.0f;
            if (r0 < out_dim) {
                if (c0 < batch) y[(size_t)c0 * out_dim + r0] = acc[(j0 >> 3) + n][0] + b0;
                if (c0 + 1u < batch) y[(size_t)(c0 + 1u) * out_dim + r0] = acc[(j0 >> 3) + n][1] + b0;
            }
            if (r8 < out_dim) {
                if (c0 < batch) y[(size_t)c0 * out_dim + r8] = acc[(j0 >> 3) + n][2] + b8;
                if (c0 + 1u < batch) y[(size_t)(c0 + 1u) * out_dim + r8] = acc[(j0 >> 3) + n][3] + b8;
            }
        }
    }
#else
    (void)data; (void)scale; (void)yq; (void)bias; (void)y;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}

PD_EXPORT
int pd_q8_0_gemm_mmq_pipe(const void* data, const void* scale, const void* yq,
                          const void* bias, void* y, uint32_t in_dim,
                          uint32_t out_dim, uint32_t batch, void* stream) {
    if (out_dim == 0 || batch == 0) return 0;
    // in_dim only needs %32 now: the kernel K-pads the final partial 128-chunk
    // (weight pad masked to 0, activation zero-quantized by pd_quantize_q8_mmq's
    // ceil-chunk grid), so K=2880 rides the pipe bit-exactly. out_dim needs no
    // alignment: RepackedQ8.dims[0] is in_dim, not out_dim - the
    // caller's `dims[0] % 128` gate does not exclude a narrow out_dim like
    // laguna S-2.1's g_proj; this guard was the actual thing stopping it, and
    // it's over-conservative like its siblings - writeback bounds-checks
    // every row, staging zero-pads the rest).
    if (in_dim & 31u) return cudaErrorInvalidValue;
    static cudaError_t attr = cudaFuncSetAttribute(
        (const void*)pd_q8_0_gemm_mmq_pipe_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)PD_MMQ_PIPE_SMEM);
    if (attr != cudaSuccess) return attr;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t nct = batch_pad >> 7;
    const uint32_t ntiles = ((out_dim + 127u) / 128u) * nct;
    pd_q8_0_gemm_mmq_pipe_kernel<<<ntiles, 256, PD_MMQ_PIPE_SMEM, (cudaStream_t)stream>>>(
        (const int8_t*)data, (const __half*)scale, (const uint8_t*)yq,
        (const float*)bias, (float*)y, in_dim, out_dim, batch);
    return pd_launch_status();
}

// ---------------------------------------- mmq_pipe gate|up + GLU landing (slot 865)
// The FFN's gate and up GEMMs in ONE launch whose landing is the down GEMM's
// input: v = glu(gate) * up (tanh GELU or SiLU), quantized straight into the
// mmq layout.
// The 3-launch chain it replaces (gate pipe, up pipe, then
// pd_quantize_q8_mmq_geglu) wrote both f32 planes and read them back - 252
// MB a 2048-row chunk of a 3840 -> 15360 FFN - and the fused form runs that
// chunk 1.35x faster with the same bytes out.
//
// A CTA's 128 tile rows are 64 gate rows (tile rows 0-63) over the SAME 64 up
// rows (64-127), so mmq_pipe's warp map hands warp w (gate) and warp w + 4
// (up) matching (row, column) accumulators lane for lane; the up warps pass
// theirs through shared memory once the K loop is done. The main loop is
// mmq_pipe's verbatim (same staging, fragments and per-k32 fold), so gate and
// up are mmq_pipe's bits; v takes pd_geglu's / pd_swiglu's formula, and the quantize
// pd_quantize_q8_mmq's scale math (amax over the 32-row block - order-free -
// then the same rn + clamp). A warp's 32 rows are exactly one quantize block:
// for each of its columns the block's 32 values sit 4 a lane in the 8 lanes
// sharing t. Pad columns (>= batch) quantize to 0 as the chain wrote them.
// n_ff % 128 == 0 (every chunk of the output fully covered).
__device__ __forceinline__ void pd_mmqgu_issue_w(
    int* __restrict__ wbuf, const int8_t* __restrict__ gd, const __half* __restrict__ gs,
    const int8_t* __restrict__ ud, const __half* __restrict__ us, uint32_t f0, uint32_t n_ff,
    uint32_t n_k32, uint32_t n_blocks, uint32_t in_dim, uint32_t kc, uint32_t tid) {
#if PD_MMA_OK
    #pragma unroll
    for (uint32_t it = 0; it < 4u; ++it) {  // 4*256 == 128 rows * 8 segments
        const uint32_t i = it * 256u + tid;
        const uint32_t row = i >> 3, seg = i & 7u, gk4 = kc * 32u + seg * 4u;
        const uint32_t fr = f0 + (row & 63u);
        const int8_t* d = row < 64u ? gd : ud;
        const bool ok = gk4 < n_k32 && fr < n_ff;
        pd_cpa16p(wbuf + row * PD_MMQ_PIPE_WK + seg * 4u,
                  (const char*)d + (size_t)fr * in_dim + (size_t)(kc * 128u + seg * 16u), ok);
    }
    if (tid < 128u) {  // the rows' 4 f16 scales, split 2+2 as pd_mmqp_issue_w does
        const uint32_t gb = kc * 4u, fr = f0 + (tid & 63u);
        const bool row_ok = fr < n_ff;
        const __half* sc = tid < 64u ? gs : us;
        char* dst = (char*)(wbuf + tid * PD_MMQ_PIPE_WK + 32u);
        const char* src = (const char*)(sc + (size_t)fr * n_blocks + gb);
        pd_cpa4p(dst, src, row_ok && (gb + 1u < n_blocks));
        pd_cpa4p(dst + 4u, src + 4u, row_ok && (gb + 3u < n_blocks));
    }
#endif
}

template <int ACT>
__global__ void __launch_bounds__(256, 1) pd_q8_0_gemm_mmq_gu_kernel(
        const int8_t* __restrict__ gd, const __half* __restrict__ gs,
        const int8_t* __restrict__ ud, const __half* __restrict__ us,
        const uint8_t* __restrict__ yq, uint8_t* __restrict__ yo,
        uint32_t in_dim, uint32_t n_ff, uint32_t batch) {
#if PD_MMA_OK
    extern __shared__ int pd_mmqgu_sh[];
    int* wbuf0 = pd_mmqgu_sh;
    int* wbuf1 = wbuf0 + 128 * PD_MMQ_PIPE_WK;
    int* ybuf0 = wbuf1 + 128 * PD_MMQ_PIPE_WK;
    int* ybuf1 = ybuf0 + 128 * PD_MMQ_YK;

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, t = lane & 3u;
    const uint32_t i0 = (warp >> 1) * 32u;
    const uint32_t joff = (warp & 1u) * 8u;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t n_k32 = in_dim >> 2;
    const uint32_t n_blocks = in_dim >> 5;
    const uint32_t n_chunks = (in_dim + 127u) >> 7;
    const uint32_t nct = batch_pad >> 7;
    const uint32_t tile = blockIdx.x;
    const uint32_t f0 = (tile / nct) * 64u;
    const uint32_t col_base = (tile % nct) * 128u;

    float acc[16][4] = {};
    pd_mmqgu_issue_w(wbuf0, gd, gs, ud, us, f0, n_ff, n_k32, n_blocks, in_dim, 0u, tid);
    {
        const int* by0 = (const int*)(yq + ((size_t)0u * batch_pad + col_base) * 144u);
        #pragma unroll
        for (uint32_t it = 0; it < 5u; ++it)
            if (it * 256u + tid < 1152u)
                pd_cpa16p(ybuf0 + (it * 256u + tid) * 4u,
                          (const char*)by0 + (size_t)(it * 256u + tid) * 16u, true);
    }
    asm volatile("cp.async.commit_group;");

    for (uint32_t kc = 0; kc < n_chunks; ++kc) {
        int* tw = (kc & 1u) ? wbuf1 : wbuf0;
        int* ty = (kc & 1u) ? ybuf1 : ybuf0;
        if (kc + 1u < n_chunks) {
            int* nw = (kc & 1u) ? wbuf0 : wbuf1;
            int* ny = (kc & 1u) ? ybuf0 : ybuf1;
            pd_mmqgu_issue_w(nw, gd, gs, ud, us, f0, n_ff, n_k32, n_blocks, in_dim, kc + 1u, tid);
            const int* by1 = (const int*)(yq + ((size_t)(kc + 1u) * batch_pad + col_base) * 144u);
            #pragma unroll
            for (uint32_t it = 0; it < 5u; ++it)
                if (it * 256u + tid < 1152u)
                    pd_cpa16p(ny + (it * 256u + tid) * 4u,
                              (const char*)by1 + (size_t)(it * 256u + tid) * 16u, true);
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();

        int A[2][4][4];
        float dA[2][2][4];
        #pragma unroll
        for (uint32_t n = 0; n < 2u; ++n) {
            const uint32_t r0 = (i0 + n * 16u + g) * PD_MMQ_PIPE_WK;
            const uint32_t r8 = (i0 + n * 16u + 8u + g) * PD_MMQ_PIPE_WK;
            #pragma unroll
            for (uint32_t kk = 0; kk < 4u; ++kk) {
                const uint32_t ko = kk * 8u;
                A[n][kk][0] = tw[r0 + ko + t];
                A[n][kk][1] = tw[r8 + ko + t];
                A[n][kk][2] = tw[r0 + ko + 4u + t];
                A[n][kk][3] = tw[r8 + ko + 4u + t];
                dA[n][0][kk] = __half2float(((const __half*)(tw + r0 + 32u))[kk]);
                dA[n][1][kk] = __half2float(((const __half*)(tw + r8 + 32u))[kk]);
            }
        }
        #pragma unroll
        for (uint32_t j0 = 0; j0 < 128u; j0 += 16u) {
            const uint32_t jc = j0 + joff;
            #pragma unroll
            for (uint32_t kk = 0; kk < 4u; ++kk) {
                const uint32_t ko = kk * 8u;
                const int b0 = ty[(jc + g) * PD_MMQ_YK + 4u + ko + t];
                const int b1 = ty[(jc + g) * PD_MMQ_YK + 4u + ko + 4u + t];
                const float dB0 = ((const float*)ty)[(jc + 2u * t) * PD_MMQ_YK + kk];
                const float dB1 = ((const float*)ty)[(jc + 2u * t + 1u) * PD_MMQ_YK + kk];
                #pragma unroll
                for (uint32_t n = 0; n < 2u; ++n) {
                    int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                        : "r"(A[n][kk][0]), "r"(A[n][kk][1]), "r"(A[n][kk][2]),
                          "r"(A[n][kk][3]), "r"(b0), "r"(b1));
                    acc[(j0 >> 3) + n][0] += dA[n][0][kk] * dB0 * (float)d0;
                    acc[(j0 >> 3) + n][1] += dA[n][0][kk] * dB1 * (float)d1;
                    acc[(j0 >> 3) + n][2] += dA[n][1][kk] * dB0 * (float)d2;
                    acc[(j0 >> 3) + n][3] += dA[n][1][kk] * dB1 * (float)d3;
                }
            }
        }
        __syncthreads();  // this buffer free before chunk kc+2 prefetches into it
    }

    // the up warps' accumulators to the gate warps: [4 warps][16][4][32 lanes]
    // f32 = 32 KB over the (drained) staging buffers
    float* xch = reinterpret_cast<float*>(pd_mmqgu_sh);
    if (warp >= 4u) {
        #pragma unroll
        for (uint32_t q = 0; q < 16u; ++q)
            #pragma unroll
            for (uint32_t e = 0; e < 4u; ++e)
                xch[(((warp - 4u) * 16u + q) * 4u + e) * 32u + lane] = acc[q][e];
    }
    __syncthreads();
    if (warp >= 4u) return;
    #pragma unroll
    for (uint32_t q = 0; q < 16u; ++q)
        #pragma unroll
        for (uint32_t e = 0; e < 4u; ++e) {
            const float gv = acc[q][e];
            const float uv = xch[((warp * 16u + q) * 4u + e) * 32u + lane];
            if (ACT == PD_ACT_SILU) {
                acc[q][e] = (gv / (1.0f + expf(-gv))) * uv;  // == pd_swiglu
            } else {
                const float gelu = 0.5f * gv
                    * (1.0f + tanhf(0.79788456080286535587989211986876f * gv
                                    * (1.0f + 0.044715f * gv * gv)));  // == pd_geglu
                acc[q][e] = gelu * uv;
            }
        }
    // this warp's 32 rows are one quantize block: (chunk, block in chunk)
    const uint32_t fblk = f0 + i0;
    const uint32_t kco = fblk >> 7, bi = (fblk & 127u) >> 5;
    #pragma unroll
    for (uint32_t j0 = 0; j0 < 128u; j0 += 16u) {
        #pragma unroll
        for (uint32_t c = 0; c < 2u; ++c) {  // column 2t + c of the warp's 8-col group
            // rows g, g + 8 (n = 0) and 16 + g, 24 + g (n = 1)
            const float v0 = acc[(j0 >> 3)][c], v1 = acc[(j0 >> 3)][2u + c];
            const float v2 = acc[(j0 >> 3) + 1u][c], v3 = acc[(j0 >> 3) + 1u][2u + c];
            float a = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
            #pragma unroll
            for (uint32_t s = 4; s < 32u; s <<= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, s));
            const float scl = a * (1.0f / 127.0f);
            const float inv = scl > 0.0f ? 1.0f / scl : 0.0f;
            auto q8 = [&](float v) -> char {
                const int qi = __float2int_rn(v * inv);
                return (char)(qi < -127 ? -127 : (qi > 127 ? 127 : qi));
            };
            const uint32_t col = col_base + j0 + joff + 2u * t + c;
            uint8_t* blk = yo + ((size_t)kco * batch_pad + col) * 144u;
            char* dq = reinterpret_cast<char*>(blk + 16u + bi * 32u);
            dq[g] = q8(v0);
            dq[8u + g] = q8(v1);
            dq[16u + g] = q8(v2);
            dq[24u + g] = q8(v3);
            if (g == 0u) reinterpret_cast<float*>(blk)[bi] = scl;
        }
    }
#else
    (void)gd; (void)gs; (void)ud; (void)us; (void)yq; (void)yo;
    (void)in_dim; (void)n_ff; (void)batch;
#endif
}

// gate|up GEMM + GLU + mmq quantize: yq is the FFN input in the mmq layout
// (pd_quantize_q8_mmq over in_dim), yo the down GEMM's input in the same
// layout over n_ff (n_ff / 128 chunks x the 128-padded batch, 144 B each).
// act: PD_ACT_GELU (tanh form, == pd_quantize_q8_mmq_geglu's) or PD_ACT_SILU
// (== pd_quantize_q8_mmq_swiglu's); anything else is refused.
PD_EXPORT
int pd_q8_0_gemm_mmq_gu(const void* gate_data, const void* gate_scale, const void* up_data,
                        const void* up_scale, const void* yq, void* yo, uint32_t in_dim,
                        uint32_t n_ff, uint32_t batch, uint32_t act, void* stream) {
    if (n_ff == 0 || batch == 0) return 0;
    if (act != PD_ACT_GELU && act != PD_ACT_SILU) return cudaErrorNotSupported;
    if ((in_dim & 31u) || (n_ff & 127u)) return cudaErrorInvalidValue;
    auto kern = act == PD_ACT_SILU ? pd_q8_0_gemm_mmq_gu_kernel<PD_ACT_SILU>
                                   : pd_q8_0_gemm_mmq_gu_kernel<PD_ACT_GELU>;
    static cudaError_t attr = [] {
        cudaError_t e = cudaFuncSetAttribute((const void*)pd_q8_0_gemm_mmq_gu_kernel<PD_ACT_GELU>,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                                             (int)PD_MMQ_PIPE_SMEM);
        if (e != cudaSuccess) return e;
        return cudaFuncSetAttribute((const void*)pd_q8_0_gemm_mmq_gu_kernel<PD_ACT_SILU>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)PD_MMQ_PIPE_SMEM);
    }();
    if (attr != cudaSuccess) return attr;
    const uint32_t nct = ((batch + 127u) & ~127u) >> 7;
    const uint32_t ntiles = (n_ff / 64u) * nct;
    kern<<<ntiles, 256, PD_MMQ_PIPE_SMEM, (cudaStream_t)stream>>>(
        (const int8_t*)gate_data, (const __half*)gate_scale, (const int8_t*)up_data,
        (const __half*)up_scale, (const uint8_t*)yq, (uint8_t*)yo, in_dim, n_ff, batch);
    return pd_launch_status();
}

// ---------------------------------------- mmq_pipe + hyper-connection mix (slot 605)
// The hyper-connection up ([lowrank -> 4 * hidden] Q8_0) and its gated mix
//   out[d] = (1/4) sum_s sigmoid(gate[s*hidden + d]) * xn[s*hidden + d]
// in one launch. The weight staging puts plane row (d, s) at tile slot
// 32*floor(d/8) + d%8 + 8s, which makes a thread's four accumulator rows
// (i0 + g, +8, +16, +24) one d's four streams for both of its columns, so the
// epilogue folds the mix and writes the hidden-wide output: the [rows][4 *
// hidden] gate plane is never written or read back (at prefill the separate
// mix read 84 MB a call, at the bandwidth wall). A lane loads the float4 line
// of xn holding its d - four lanes of a warp share it - where scalar loads
// left the epilogue latency-bound. Every gate is the pipe tile's bits (its
// bias-free `acc + 0.0f`) and the fold is pd_q4x_hc_mix's order, so the output
// is byte-identical to pipe + mix (bench/hcup_mix_gb10_bench.cu, 1024 rows:
// 613 -> 415 us a call). hidden % 8 == 0.
__device__ __forceinline__ uint32_t pd_mmqp_hcmix_src(uint32_t p, uint32_t hidden) {
    return ((p % 32u) >> 3) * hidden + 8u * (p / 32u) + ((p % 32u) & 7u);
}

__device__ __forceinline__ void pd_mmqp_hcmix_issue_w(
    int* __restrict__ wbuf, const int8_t* __restrict__ data, const __half* __restrict__ scale,
    uint32_t row_base, uint32_t out_dim, uint32_t hidden, uint32_t n_k32, uint32_t n_blocks,
    uint32_t in_dim, uint32_t kc, uint32_t tid) {
#if PD_MMA_OK
    #pragma unroll
    for (uint32_t it = 0; it < 4u; ++it) {
        const uint32_t i = it * 256u + tid;
        const uint32_t row = i >> 3, seg = i & 7u, gk4 = kc * 32u + seg * 4u;
        const uint32_t p = row_base + row;
        const bool ok = gk4 < n_k32 && p < out_dim;
        const uint32_t src = ok ? pd_mmqp_hcmix_src(p, hidden) : 0u;
        pd_cpa16p(wbuf + row * PD_MMQ_PIPE_WK + seg * 4u,
                  (const char*)data + (size_t)src * in_dim + (size_t)(kc * 128u + seg * 16u), ok);
    }
    if (tid < 128u) {
        const uint32_t gb = kc * 4u;
        const uint32_t p = row_base + tid;
        const bool row_ok = p < out_dim;
        const uint32_t src = row_ok ? pd_mmqp_hcmix_src(p, hidden) : 0u;
        char* dst = (char*)(wbuf + tid * PD_MMQ_PIPE_WK + 32u);
        const char* s = (const char*)(scale + (size_t)src * n_blocks + gb);
        pd_cpa4p(dst, s, row_ok && (gb + 1u < n_blocks));
        pd_cpa4p(dst + 4u, s + 4u, row_ok && (gb + 3u < n_blocks));
    }
#endif
}

__global__ void __launch_bounds__(256, 1) pd_q8_0_gemm_mmq_pipe_hcmix_kernel(
        const int8_t* __restrict__ data, const __half* __restrict__ scale,
        const uint8_t* __restrict__ yq, const float* __restrict__ xn,
        float* __restrict__ out, uint32_t in_dim, uint32_t hidden, uint32_t batch) {
#if PD_MMA_OK
    constexpr uint32_t HC = 4u;
    extern __shared__ int pd_mmqp_sh[];
    int* wbuf0 = pd_mmqp_sh;
    int* wbuf1 = wbuf0 + 128 * PD_MMQ_PIPE_WK;
    int* ybuf0 = wbuf1 + 128 * PD_MMQ_PIPE_WK;
    int* ybuf1 = ybuf0 + 128 * PD_MMQ_YK;

    const uint32_t out_dim = HC * hidden;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, t = lane & 3u;
    const uint32_t i0 = (warp >> 1) * 32u;
    const uint32_t joff = (warp & 1u) * 8u;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t n_k32 = in_dim >> 2;
    const uint32_t n_blocks = in_dim >> 5;
    const uint32_t n_chunks = (in_dim + 127u) >> 7;
    const uint32_t nct = batch_pad >> 7;

    const uint32_t tile = blockIdx.x;
    const uint32_t row_base = (tile / nct) * 128u;
    const uint32_t col_base = (tile % nct) * 128u;

    float acc[16][4] = {};
    pd_mmqp_hcmix_issue_w(wbuf0, data, scale, row_base, out_dim, hidden, n_k32, n_blocks, in_dim, 0u, tid);
    {
        const int* by0 = (const int*)(yq + ((size_t)0u * batch_pad + col_base) * 144u);
        #pragma unroll
        for (uint32_t it = 0; it < 5u; ++it)
            if (it * 256u + tid < 1152u)
                pd_cpa16p(ybuf0 + (it * 256u + tid) * 4u,
                          (const char*)by0 + (size_t)(it * 256u + tid) * 16u, true);
    }
    asm volatile("cp.async.commit_group;");

    for (uint32_t kc = 0; kc < n_chunks; ++kc) {
        int* tw = (kc & 1u) ? wbuf1 : wbuf0;
        int* ty = (kc & 1u) ? ybuf1 : ybuf0;
        if (kc + 1u < n_chunks) {
            int* nw = (kc & 1u) ? wbuf0 : wbuf1;
            int* ny = (kc & 1u) ? ybuf0 : ybuf1;
            pd_mmqp_hcmix_issue_w(nw, data, scale, row_base, out_dim, hidden, n_k32, n_blocks, in_dim,
                                  kc + 1u, tid);
            const int* by1 = (const int*)(yq + ((size_t)(kc + 1u) * batch_pad + col_base) * 144u);
            #pragma unroll
            for (uint32_t it = 0; it < 5u; ++it)
                if (it * 256u + tid < 1152u)
                    pd_cpa16p(ny + (it * 256u + tid) * 4u,
                              (const char*)by1 + (size_t)(it * 256u + tid) * 16u, true);
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();

        int A[2][4][4];
        float dA[2][2][4];
        #pragma unroll
        for (uint32_t n = 0; n < 2u; ++n) {
            const uint32_t r0 = (i0 + n * 16u + g) * PD_MMQ_PIPE_WK;
            const uint32_t r8 = (i0 + n * 16u + 8u + g) * PD_MMQ_PIPE_WK;
            #pragma unroll
            for (uint32_t kk = 0; kk < 4u; ++kk) {
                const uint32_t ko = kk * 8u;
                A[n][kk][0] = tw[r0 + ko + t];
                A[n][kk][1] = tw[r8 + ko + t];
                A[n][kk][2] = tw[r0 + ko + 4u + t];
                A[n][kk][3] = tw[r8 + ko + 4u + t];
                dA[n][0][kk] = __half2float(((const __half*)(tw + r0 + 32u))[kk]);
                dA[n][1][kk] = __half2float(((const __half*)(tw + r8 + 32u))[kk]);
            }
        }
        #pragma unroll
        for (uint32_t j0 = 0; j0 < 128u; j0 += 16u) {
            const uint32_t jc = j0 + joff;
            #pragma unroll
            for (uint32_t kk = 0; kk < 4u; ++kk) {
                const uint32_t ko = kk * 8u;
                const int b0 = ty[(jc + g) * PD_MMQ_YK + 4u + ko + t];
                const int b1 = ty[(jc + g) * PD_MMQ_YK + 4u + ko + 4u + t];
                const float dB0 = ((const float*)ty)[(jc + 2u * t) * PD_MMQ_YK + kk];
                const float dB1 = ((const float*)ty)[(jc + 2u * t + 1u) * PD_MMQ_YK + kk];
                #pragma unroll
                for (uint32_t n = 0; n < 2u; ++n) {
                    int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                        : "r"(A[n][kk][0]), "r"(A[n][kk][1]), "r"(A[n][kk][2]),
                          "r"(A[n][kk][3]), "r"(b0), "r"(b1));
                    acc[(j0 >> 3) + n][0] += dA[n][0][kk] * dB0 * (float)d0;
                    acc[(j0 >> 3) + n][1] += dA[n][0][kk] * dB1 * (float)d1;
                    acc[(j0 >> 3) + n][2] += dA[n][1][kk] * dB0 * (float)d2;
                    acc[(j0 >> 3) + n][3] += dA[n][1][kk] * dB1 * (float)d3;
                }
            }
        }
        __syncthreads();
    }

    // epilogue: this thread's d, its four streams, its 16 columns
    const uint32_t p0 = row_base + i0;
    if (p0 >= out_dim) return;
    const uint32_t d = 8u * (p0 / 32u) + g;
    const uint32_t dl = d & ~3u, dk = d & 3u;
    #pragma unroll
    for (uint32_t j0 = 0; j0 < 128u; j0 += 16u) {
        const uint32_t c0 = col_base + j0 + joff + 2u * t;
        #pragma unroll
        for (uint32_t cc = 0; cc < 2u; ++cc) {
            const uint32_t c = c0 + cc;
            if (c >= batch) continue;
            const float gs[4] = {acc[(j0 >> 3)][cc] + 0.0f, acc[(j0 >> 3)][2u + cc] + 0.0f,
                                 acc[(j0 >> 3) + 1u][cc] + 0.0f, acc[(j0 >> 3) + 1u][2u + cc] + 0.0f};
            const size_t xr = (size_t)c * out_dim;
            float xv[4];
            #pragma unroll
            for (uint32_t s = 0; s < HC; ++s) {
                const float4 f = *reinterpret_cast<const float4*>(xn + xr + (size_t)s * hidden + dl);
                xv[s] = dk == 0u ? f.x : dk == 1u ? f.y : dk == 2u ? f.z : f.w;
            }
            float m = 0.0f;
            #pragma unroll
            for (uint32_t s = 0; s < HC; ++s) m += pd_q4x_sig(gs[s]) * xv[s];
            out[(size_t)c * hidden + d] = m / (float)HC;
        }
    }
#else
    (void)data; (void)scale; (void)yq; (void)xn; (void)out;
    (void)in_dim; (void)hidden; (void)batch;
#endif
}

PD_EXPORT
int pd_q8_0_gemm_mmq_pipe_hcmix(const void* data, const void* scale, const void* yq,
                                const void* xn, void* out, uint32_t in_dim, uint32_t hidden,
                                uint32_t batch, void* stream) {
    if (hidden == 0 || batch == 0) return 0;
    if ((in_dim & 31u) || (hidden & 7u) || xn == nullptr) return cudaErrorInvalidValue;
    static cudaError_t attr = cudaFuncSetAttribute(
        (const void*)pd_q8_0_gemm_mmq_pipe_hcmix_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)PD_MMQ_PIPE_SMEM);
    if (attr != cudaSuccess) return attr;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t nct = batch_pad >> 7;
    const uint32_t ntiles = ((4u * hidden + 127u) / 128u) * nct;
    pd_q8_0_gemm_mmq_pipe_hcmix_kernel<<<ntiles, 256, PD_MMQ_PIPE_SMEM, (cudaStream_t)stream>>>(
        (const int8_t*)data, (const __half*)scale, (const uint8_t*)yq, (const float*)xn,
        (float*)out, in_dim, hidden, batch);
    return pd_launch_status();
}

// ---------------------------------------- slot 605 over a REBUILT normalized state (slot 608)
// pd_q8_0_gemm_mmq_pipe_hcmix with each normalized value rebuilt from the
// residual `h` with the combine's own expression (v * inv + (v * inv) * w, off
// `aux` = [norm_w (4 * hidden) | 1/rms (rows * 4)] that slot 606 publishes)
// instead of read from a stored [rows][4 * hidden] state. The epilogue issues
// all of its loads (the 16 columns' four h lines and their 1/rms rows) before
// any arithmetic: loading and rebuilding per column measured 583 us a call
// against 357 staged (bench/combine_rn_gb10_bench.cu, 1024 rows; slot 605 over
// the stored state 414). Eight kernel arguments deliberately - the same tile at
// ten measured 590 us with its rebuild reduced to a plain load, 383 at eight.
// Byte-identical to slot 605 over the stored state. dims = in_dim | hidden << 16.
template <uint32_t HC>
__global__ void __launch_bounds__(256, 1) pd_q8_0_gemm_mmq_pipe_hcmix_rn_kernel(
        const int8_t* __restrict__ data, const __half* __restrict__ scale,
        const uint8_t* __restrict__ yq, const float* __restrict__ h,
        const float* __restrict__ aux, float* __restrict__ out, uint32_t dims, uint32_t batch) {
#if PD_MMA_OK
    extern __shared__ int pd_mmqp_sh[];
    int* wbuf0 = pd_mmqp_sh;
    int* wbuf1 = wbuf0 + 128 * PD_MMQ_PIPE_WK;
    int* ybuf0 = wbuf1 + 128 * PD_MMQ_PIPE_WK;
    int* ybuf1 = ybuf0 + 128 * PD_MMQ_YK;

    const uint32_t in_dim = dims & 0xFFFFu, hidden = dims >> 16u;
    const uint32_t out_dim = HC * hidden;
    const float* norm_w = aux;
    const float* nscale = aux + out_dim;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, t = lane & 3u;
    const uint32_t i0 = (warp >> 1) * 32u;
    const uint32_t joff = (warp & 1u) * 8u;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t n_k32 = in_dim >> 2;
    const uint32_t n_blocks = in_dim >> 5;
    const uint32_t n_chunks = (in_dim + 127u) >> 7;
    const uint32_t nct = batch_pad >> 7;

    const uint32_t tile = blockIdx.x;
    const uint32_t row_base = (tile / nct) * 128u;
    const uint32_t col_base = (tile % nct) * 128u;

    float acc[16][4] = {};
    pd_mmqp_hcmix_issue_w(wbuf0, data, scale, row_base, out_dim, hidden, n_k32, n_blocks, in_dim, 0u, tid);
    {
        const int* by0 = (const int*)(yq + ((size_t)0u * batch_pad + col_base) * 144u);
        #pragma unroll
        for (uint32_t it = 0; it < 5u; ++it)
            if (it * 256u + tid < 1152u)
                pd_cpa16p(ybuf0 + (it * 256u + tid) * 4u,
                          (const char*)by0 + (size_t)(it * 256u + tid) * 16u, true);
    }
    asm volatile("cp.async.commit_group;");

    for (uint32_t kc = 0; kc < n_chunks; ++kc) {
        int* tw = (kc & 1u) ? wbuf1 : wbuf0;
        int* ty = (kc & 1u) ? ybuf1 : ybuf0;
        if (kc + 1u < n_chunks) {
            int* nw = (kc & 1u) ? wbuf0 : wbuf1;
            int* ny = (kc & 1u) ? ybuf0 : ybuf1;
            pd_mmqp_hcmix_issue_w(nw, data, scale, row_base, out_dim, hidden, n_k32, n_blocks, in_dim,
                                  kc + 1u, tid);
            const int* by1 = (const int*)(yq + ((size_t)(kc + 1u) * batch_pad + col_base) * 144u);
            #pragma unroll
            for (uint32_t it = 0; it < 5u; ++it)
                if (it * 256u + tid < 1152u)
                    pd_cpa16p(ny + (it * 256u + tid) * 4u,
                              (const char*)by1 + (size_t)(it * 256u + tid) * 16u, true);
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();

        int A[2][4][4];
        float dA[2][2][4];
        #pragma unroll
        for (uint32_t n = 0; n < 2u; ++n) {
            const uint32_t r0 = (i0 + n * 16u + g) * PD_MMQ_PIPE_WK;
            const uint32_t r8 = (i0 + n * 16u + 8u + g) * PD_MMQ_PIPE_WK;
            #pragma unroll
            for (uint32_t kk = 0; kk < 4u; ++kk) {
                const uint32_t ko = kk * 8u;
                A[n][kk][0] = tw[r0 + ko + t];
                A[n][kk][1] = tw[r8 + ko + t];
                A[n][kk][2] = tw[r0 + ko + 4u + t];
                A[n][kk][3] = tw[r8 + ko + 4u + t];
                dA[n][0][kk] = __half2float(((const __half*)(tw + r0 + 32u))[kk]);
                dA[n][1][kk] = __half2float(((const __half*)(tw + r8 + 32u))[kk]);
            }
        }
        #pragma unroll
        for (uint32_t j0 = 0; j0 < 128u; j0 += 16u) {
            const uint32_t jc = j0 + joff;
            #pragma unroll
            for (uint32_t kk = 0; kk < 4u; ++kk) {
                const uint32_t ko = kk * 8u;
                const int b0 = ty[(jc + g) * PD_MMQ_YK + 4u + ko + t];
                const int b1 = ty[(jc + g) * PD_MMQ_YK + 4u + ko + 4u + t];
                const float dB0 = ((const float*)ty)[(jc + 2u * t) * PD_MMQ_YK + kk];
                const float dB1 = ((const float*)ty)[(jc + 2u * t + 1u) * PD_MMQ_YK + kk];
                #pragma unroll
                for (uint32_t n = 0; n < 2u; ++n) {
                    int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                        : "r"(A[n][kk][0]), "r"(A[n][kk][1]), "r"(A[n][kk][2]),
                          "r"(A[n][kk][3]), "r"(b0), "r"(b1));
                    acc[(j0 >> 3) + n][0] += dA[n][0][kk] * dB0 * (float)d0;
                    acc[(j0 >> 3) + n][1] += dA[n][0][kk] * dB1 * (float)d1;
                    acc[(j0 >> 3) + n][2] += dA[n][1][kk] * dB0 * (float)d2;
                    acc[(j0 >> 3) + n][3] += dA[n][1][kk] * dB1 * (float)d3;
                }
            }
        }
        __syncthreads();
    }

    // epilogue: this thread's d, its HC streams, its 16 columns - loads first
    const uint32_t p0 = row_base + i0;
    if (p0 >= out_dim) return;
    const uint32_t d = 8u * (p0 / 32u) + g;
    const uint32_t dl = d & ~3u, dk = d & 3u;
    float wk[HC];
    #pragma unroll
    for (uint32_t s = 0; s < HC; ++s) {
        const float4 wl = *reinterpret_cast<const float4*>(norm_w + (size_t)s * hidden + dl);
        wk[s] = dk == 0u ? wl.x : dk == 1u ? wl.y : dk == 2u ? wl.z : wl.w;
    }
    float hs[16][HC];
    float ns[16][HC];
    #pragma unroll
    for (uint32_t j = 0; j < 16u; ++j) {
        const uint32_t c = col_base + (j >> 1) * 16u + joff + 2u * t + (j & 1u);
        const uint32_t cs = c < batch ? c : 0u;  // a padded column loads row 0, never stores
        const float4 nv = *reinterpret_cast<const float4*>(nscale + (size_t)cs * HC);
        ns[j][0] = nv.x; ns[j][1] = nv.y; ns[j][2] = nv.z; ns[j][3] = nv.w;
        #pragma unroll
        for (uint32_t s = 0; s < HC; ++s) {
            const float4 v = *reinterpret_cast<const float4*>(h + (size_t)cs * out_dim + (size_t)s * hidden + dl);
            hs[j][s] = dk == 0u ? v.x : dk == 1u ? v.y : dk == 2u ? v.z : v.w;
        }
    }
    #pragma unroll
    for (uint32_t j = 0; j < 16u; ++j) {
        const uint32_t j0 = (j >> 1) * 16u, cc = j & 1u;
        const uint32_t c = col_base + j0 + joff + 2u * t + cc;
        if (c >= batch) continue;
        const float gs[4] = {acc[(j0 >> 3)][cc] + 0.0f, acc[(j0 >> 3)][2u + cc] + 0.0f,
                             acc[(j0 >> 3) + 1u][cc] + 0.0f, acc[(j0 >> 3) + 1u][2u + cc] + 0.0f};
        float m = 0.0f;
        #pragma unroll
        for (uint32_t s = 0; s < HC; ++s) {
            const float hv = hs[j][s], inv = ns[j][s], wvv = wk[s];
            const float xv = hv * inv + (hv * inv) * wvv;
            m += pd_q4x_sig(gs[s]) * xv;
        }
        out[(size_t)c * hidden + d] = m / (float)HC;
    }
#else
    (void)data; (void)scale; (void)yq; (void)h; (void)aux; (void)out; (void)dims; (void)batch;
#endif
}

PD_EXPORT
int pd_q8_0_gemm_mmq_pipe_hcmix_rn(const void* data, const void* scale, const void* yq,
                                   const void* h, const void* aux, void* out, uint32_t in_dim,
                                   uint32_t hidden, uint32_t batch, void* stream) {
    if (hidden == 0 || batch == 0) return 0;
    if ((in_dim & 31u) || in_dim > 0xFFFFu || (hidden & 7u) || hidden > 0xFFFFu || h == nullptr ||
        aux == nullptr)
        return cudaErrorInvalidValue;
    static cudaError_t attr = cudaFuncSetAttribute(
        (const void*)pd_q8_0_gemm_mmq_pipe_hcmix_rn_kernel<4u>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)PD_MMQ_PIPE_SMEM);
    if (attr != cudaSuccess) return attr;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t nct = batch_pad >> 7;
    const uint32_t ntiles = ((4u * hidden + 127u) / 128u) * nct;
    pd_q8_0_gemm_mmq_pipe_hcmix_rn_kernel<4u><<<ntiles, 256, PD_MMQ_PIPE_SMEM, (cudaStream_t)stream>>>(
        (const int8_t*)data, (const __half*)scale, (const uint8_t*)yq, (const float*)h,
        (const float*)aux, (float*)out, in_dim | (hidden << 16), batch);
    return pd_launch_status();
}

// ---------------------------------------- mmq_pipe64 (small-grid variant)
// pd_q8_0_gemm_mmq_pipe with the K stage halved to 64: each buffer drops to
// 128 rows x 20 int32, the four tiles fit 40 KB, and __launch_bounds__(256,2)
// lands two resident blocks per SM. Same 2-stage cp.async pipeline, same
// 128x128 output tile, same per-k32 fold (bit-identical Q8_0 class). The
// point is WAVE QUANTIZATION at serving batch sizes: a 1024-out projection
// at ~4k rows is 232 blocks - 1.23 waves at 1 block/SM rounds up to 2 full
// waves (~40% idle); at 2 blocks/SM the same grid fits inside one wave and
// the co-resident block fills the extra barrier gaps the shorter stage
// introduces. Selected by grid size in the engine dispatch; large grids
// keep pipe-128 (deeper prefetch wins when the tail is amortized).
// (pd_cpa4p is defined once, up by pd_cpa8p, before its first use.)

#define PD_MMQ_P64_WK 20  // 16 data int32 + 1 int32 (2 f16 scales) + 3 pad
#define PD_MMQ_P64_YK 20  // 2 f32 scales + 2 pad + 16 data int32
#define PD_MMQ_P64_SMEM ((4u * 128u * 20u) * 4u)

// Issue one 64-K weight half-chunk (16B data segs + 4B scale pair) into wbuf.
__device__ __forceinline__ void pd_mmqp64_issue_w(
    int* __restrict__ wbuf, const int8_t* __restrict__ data, const __half* __restrict__ scale,
    uint32_t row_base, uint32_t out_dim, uint32_t n_k32, uint32_t n_blocks,
    uint32_t in_dim, uint32_t kc, uint32_t tid) {
#if PD_MMA_OK
    #pragma unroll
    for (uint32_t it = 0; it < 2u; ++it) {  // 2*256 == 128 rows * 4 segments
        const uint32_t i = it * 256u + tid;
        const uint32_t row = i >> 2, seg = i & 3u, gk4 = kc * 16u + seg * 4u;
        const bool ok = gk4 < n_k32 && (row_base + row) < out_dim;
        pd_cpa16p(wbuf + row * PD_MMQ_P64_WK + seg * 4u,
                  (const char*)data + (size_t)(row_base + row) * in_dim
                      + (size_t)(kc * 64u + seg * 16u), ok);
    }
    if (tid < 128u) {  // 2 f16 scales/row (4 bytes, 4B-aligned: n_blocks is even)
        const uint32_t gb = kc * 2u;
        const bool ok = gb < n_blocks && (row_base + tid) < out_dim;
        pd_cpa4p((char*)(wbuf + tid * PD_MMQ_P64_WK + 16u),
                 (const char*)(scale + (size_t)(row_base + tid) * n_blocks + gb), ok);
    }
#endif
}

// Issue one 64-K activation half-chunk: the quantizer's 144-byte col unit is
// 128-K-granular, so the half stages per column (4B-scale pair + 64B data).
__device__ __forceinline__ void pd_mmqp64_issue_y(
    int* __restrict__ ybuf, const uint8_t* __restrict__ yq, uint32_t batch_pad,
    uint32_t col_base, uint32_t kc, uint32_t tid) {
#if PD_MMA_OK
    const uint32_t kc128 = kc >> 1, h = kc & 1u;
    const uint8_t* by = yq + ((size_t)kc128 * batch_pad + col_base) * 144u;
    #pragma unroll
    for (uint32_t it = 0; it < 2u; ++it) {  // 2*256 == 128 cols * 4 data segs
        const uint32_t i = it * 256u + tid;
        const uint32_t col = i >> 2, seg = i & 3u;
        pd_cpa16p(ybuf + col * PD_MMQ_P64_YK + 4u + seg * 4u,
                  by + (size_t)col * 144u + 16u + h * 64u + seg * 16u, true);
    }
    if (tid < 128u)  // this half's 2 f32 scales
        pd_cpa8p(ybuf + tid * PD_MMQ_P64_YK,
                 by + (size_t)tid * 144u + h * 8u, true);
#endif
}

__global__ void __launch_bounds__(256, 2) pd_q8_0_gemm_mmq_pipe64_kernel(
        const int8_t* __restrict__ data, const __half* __restrict__ scale,
        const uint8_t* __restrict__ yq, float* __restrict__ y,
        uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
#if PD_MMA_OK
    extern __shared__ int pd_mmqp64_sh[];
    int* wbuf0 = pd_mmqp64_sh;
    int* wbuf1 = wbuf0 + 128 * PD_MMQ_P64_WK;
    int* ybuf0 = wbuf1 + 128 * PD_MMQ_P64_WK;
    int* ybuf1 = ybuf0 + 128 * PD_MMQ_P64_YK;

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, t = lane & 3u;
    const uint32_t i0 = (warp >> 1) * 32u;
    const uint32_t joff = (warp & 1u) * 8u;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t n_k32 = in_dim >> 2;
    const uint32_t n_blocks = in_dim >> 5;
    const uint32_t n_chunks = in_dim >> 6;   // in_dim % 64 == 0 (launcher-checked)
    const uint32_t nct = batch_pad >> 7;

    const uint32_t tile = blockIdx.x;
    const uint32_t row_base = (tile / nct) * 128u;
    const uint32_t col_base = (tile % nct) * 128u;

    float acc[16][4] = {};
    pd_mmqp64_issue_w(wbuf0, data, scale, row_base, out_dim, n_k32, n_blocks, in_dim, 0u, tid);
    pd_mmqp64_issue_y(ybuf0, yq, batch_pad, col_base, 0u, tid);
    asm volatile("cp.async.commit_group;");
    for (uint32_t kc = 0; kc < n_chunks; ++kc) {
        int* tw = (kc & 1u) ? wbuf1 : wbuf0;
        int* ty = (kc & 1u) ? ybuf1 : ybuf0;
        if (kc + 1u < n_chunks) {
            pd_mmqp64_issue_w((kc & 1u) ? wbuf0 : wbuf1, data, scale, row_base, out_dim,
                              n_k32, n_blocks, in_dim, kc + 1u, tid);
            pd_mmqp64_issue_y((kc & 1u) ? ybuf0 : ybuf1, yq, batch_pad, col_base,
                              kc + 1u, tid);
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();

        int A[2][2][4];
        float dA[2][2][2];
        #pragma unroll
        for (uint32_t n = 0; n < 2u; ++n) {
            const uint32_t r0 = (i0 + n * 16u + g) * PD_MMQ_P64_WK;
            const uint32_t r8 = (i0 + n * 16u + 8u + g) * PD_MMQ_P64_WK;
            #pragma unroll
            for (uint32_t kk = 0; kk < 2u; ++kk) {
                const uint32_t ko = kk * 8u;
                A[n][kk][0] = tw[r0 + ko + t];
                A[n][kk][1] = tw[r8 + ko + t];
                A[n][kk][2] = tw[r0 + ko + 4u + t];
                A[n][kk][3] = tw[r8 + ko + 4u + t];
                dA[n][0][kk] = __half2float(((const __half*)(tw + r0 + 16u))[kk]);
                dA[n][1][kk] = __half2float(((const __half*)(tw + r8 + 16u))[kk]);
            }
        }
        #pragma unroll
        for (uint32_t j0 = 0; j0 < 128u; j0 += 16u) {
            const uint32_t jc = j0 + joff;
            #pragma unroll
            for (uint32_t kk = 0; kk < 2u; ++kk) {
                const uint32_t ko = kk * 8u;
                const int b0 = ty[(jc + g) * PD_MMQ_P64_YK + 4u + ko + t];
                const int b1 = ty[(jc + g) * PD_MMQ_P64_YK + 4u + ko + 4u + t];
                const float dB0 = ((const float*)ty)[(jc + 2u * t) * PD_MMQ_P64_YK + kk];
                const float dB1 = ((const float*)ty)[(jc + 2u * t + 1u) * PD_MMQ_P64_YK + kk];
                #pragma unroll
                for (uint32_t n = 0; n < 2u; ++n) {
                    int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                        : "r"(A[n][kk][0]), "r"(A[n][kk][1]), "r"(A[n][kk][2]),
                          "r"(A[n][kk][3]), "r"(b0), "r"(b1));
                    acc[(j0 >> 3) + n][0] += dA[n][0][kk] * dB0 * (float)d0;
                    acc[(j0 >> 3) + n][1] += dA[n][0][kk] * dB1 * (float)d1;
                    acc[(j0 >> 3) + n][2] += dA[n][1][kk] * dB0 * (float)d2;
                    acc[(j0 >> 3) + n][3] += dA[n][1][kk] * dB1 * (float)d3;
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (uint32_t j0 = 0; j0 < 128u; j0 += 16u) {
        const uint32_t c0 = col_base + j0 + joff + 2u * t;
        #pragma unroll
        for (uint32_t n = 0; n < 2u; ++n) {
            const uint32_t r0 = row_base + i0 + n * 16u + g;
            const uint32_t r8 = r0 + 8u;
            if (r0 < out_dim) {
                if (c0 < batch) y[(size_t)c0 * out_dim + r0] = acc[(j0 >> 3) + n][0];
                if (c0 + 1u < batch) y[(size_t)(c0 + 1u) * out_dim + r0] = acc[(j0 >> 3) + n][1];
            }
            if (r8 < out_dim) {
                if (c0 < batch) y[(size_t)c0 * out_dim + r8] = acc[(j0 >> 3) + n][2];
                if (c0 + 1u < batch) y[(size_t)(c0 + 1u) * out_dim + r8] = acc[(j0 >> 3) + n][3];
            }
        }
    }
#else
    (void)data; (void)scale; (void)yq; (void)y;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}

PD_EXPORT
int pd_q8_0_gemm_mmq_pipe64(const void* data, const void* scale, const void* yq,
                            void* y, uint32_t in_dim, uint32_t out_dim,
                            uint32_t batch, void* stream) {
    if (out_dim == 0 || batch == 0) return 0;
    // out_dim needs no alignment (ragged rows zero-pad + bounds-check, same
    // as the rest of this family); in_dim keeps %128 - yq is
    // 128-K packed, a real format constraint.
    if (in_dim & 127u) return cudaErrorInvalidValue;
    static cudaError_t attr = cudaFuncSetAttribute(
        (const void*)pd_q8_0_gemm_mmq_pipe64_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)PD_MMQ_P64_SMEM);
    if (attr != cudaSuccess) return attr;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t nct = batch_pad >> 7;
    const uint32_t ntiles = ((out_dim + 127u) / 128u) * nct;
    pd_q8_0_gemm_mmq_pipe64_kernel<<<ntiles, 256, PD_MMQ_P64_SMEM, (cudaStream_t)stream>>>(
        (const int8_t*)data, (const __half*)scale, (const uint8_t*)yq,
        (float*)y, in_dim, out_dim, batch);
    return pd_launch_status();
}

// ------------------------------- mmq_pipe tail split-K (Stream-K lite)
// Wave quantization, third attempt - the two occupancy variants (mmq_hi,
// pipe64) lost because shrinking the K stage costs more per block than the
// recovered wave. This keeps the full pipe kernel untouched for the whole
// waves and splits only the tail tiles over the K dimension: a 232-tile
// grid on 188 SMs runs 188 tiles as one exact wave (unchanged kernel),
// then the 44 tail tiles as 44xS partial blocks (same 128-deep pipeline,
// each covering a contiguous K span) plus a tiny deterministic reduce.
// Wall: 2.0 -> ~1.4 tile-times. NUMERIC NOTE: tail tiles sum K in S groups
// (fixed order, deterministic) - same Q8_0 per-k32 fold, but the outer f32
// addition regroups, so results are the mmq CLASS, not bit-identical to
// the single-block kernel; the load-time calibration gate re-measures both
// sides. Engaged by the launcher only when the tail is worth it.
__global__ void __launch_bounds__(256, 1) pd_q8_0_gemm_mmq_pipe_span_kernel(
        const int8_t* __restrict__ data, const __half* __restrict__ scale,
        const uint8_t* __restrict__ yq, float* __restrict__ partials,
        uint32_t in_dim, uint32_t out_dim, uint32_t batch,
        uint32_t tile_off, uint32_t splits) {
#if PD_MMA_OK
    extern __shared__ int pd_mmqp_sh2[];
    int* wbuf0 = pd_mmqp_sh2;
    int* wbuf1 = wbuf0 + 128 * PD_MMQ_PIPE_WK;
    int* ybuf0 = wbuf1 + 128 * PD_MMQ_PIPE_WK;
    int* ybuf1 = ybuf0 + 128 * PD_MMQ_YK;

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, t = lane & 3u;
    const uint32_t i0 = (warp >> 1) * 32u;
    const uint32_t joff = (warp & 1u) * 8u;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t n_k32 = in_dim >> 2;
    const uint32_t n_blocks = in_dim >> 5;
    const uint32_t n_chunks = in_dim >> 7;
    const uint32_t nct = batch_pad >> 7;

    // block -> (tail tile, K span). Span boundaries in 128-K chunks, fixed
    // by (n_chunks, splits) alone - deterministic partial grouping.
    const uint32_t lt = blockIdx.x / splits;
    const uint32_t sp = blockIdx.x % splits;
    const uint32_t tile = tile_off + lt;
    const uint32_t row_base = (tile / nct) * 128u;
    const uint32_t col_base = (tile % nct) * 128u;
    const uint32_t kc_beg = sp * n_chunks / splits;
    const uint32_t kc_end = (sp + 1u) * n_chunks / splits;

    float acc[16][4] = {};
    pd_mmqp_issue_w(wbuf0, data, scale, row_base, out_dim, n_k32, n_blocks, in_dim, kc_beg, tid);
    {
        const int* by0 = (const int*)(yq + ((size_t)kc_beg * batch_pad + col_base) * 144u);
        #pragma unroll
        for (uint32_t it = 0; it < 5u; ++it)
            if (it * 256u + tid < 1152u)
                pd_cpa16p(ybuf0 + (it * 256u + tid) * 4u,
                          (const char*)by0 + (size_t)(it * 256u + tid) * 16u, true);
    }
    asm volatile("cp.async.commit_group;");
    for (uint32_t kc = kc_beg; kc < kc_end; ++kc) {
        int* tw = ((kc - kc_beg) & 1u) ? wbuf1 : wbuf0;
        int* ty = ((kc - kc_beg) & 1u) ? ybuf1 : ybuf0;
        if (kc + 1u < kc_end) {
            int* nw = ((kc - kc_beg) & 1u) ? wbuf0 : wbuf1;
            int* ny = ((kc - kc_beg) & 1u) ? ybuf0 : ybuf1;
            pd_mmqp_issue_w(nw, data, scale, row_base, out_dim, n_k32, n_blocks, in_dim,
                            kc + 1u, tid);
            const int* by1 =
                (const int*)(yq + ((size_t)(kc + 1u) * batch_pad + col_base) * 144u);
            #pragma unroll
            for (uint32_t it = 0; it < 5u; ++it)
                if (it * 256u + tid < 1152u)
                    pd_cpa16p(ny + (it * 256u + tid) * 4u,
                              (const char*)by1 + (size_t)(it * 256u + tid) * 16u, true);
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();

        int A[2][4][4];
        float dA[2][2][4];
        #pragma unroll
        for (uint32_t n = 0; n < 2u; ++n) {
            const uint32_t r0 = (i0 + n * 16u + g) * PD_MMQ_PIPE_WK;
            const uint32_t r8 = (i0 + n * 16u + 8u + g) * PD_MMQ_PIPE_WK;
            #pragma unroll
            for (uint32_t kk = 0; kk < 4u; ++kk) {
                const uint32_t ko = kk * 8u;
                A[n][kk][0] = tw[r0 + ko + t];
                A[n][kk][1] = tw[r8 + ko + t];
                A[n][kk][2] = tw[r0 + ko + 4u + t];
                A[n][kk][3] = tw[r8 + ko + 4u + t];
                dA[n][0][kk] = __half2float(((const __half*)(tw + r0 + 32u))[kk]);
                dA[n][1][kk] = __half2float(((const __half*)(tw + r8 + 32u))[kk]);
            }
        }
        #pragma unroll
        for (uint32_t j0 = 0; j0 < 128u; j0 += 16u) {
            const uint32_t jc = j0 + joff;
            #pragma unroll
            for (uint32_t kk = 0; kk < 4u; ++kk) {
                const uint32_t ko = kk * 8u;
                const int b0 = ty[(jc + g) * PD_MMQ_YK + 4u + ko + t];
                const int b1 = ty[(jc + g) * PD_MMQ_YK + 4u + ko + 4u + t];
                const float dB0 = ((const float*)ty)[(jc + 2u * t) * PD_MMQ_YK + kk];
                const float dB1 = ((const float*)ty)[(jc + 2u * t + 1u) * PD_MMQ_YK + kk];
                #pragma unroll
                for (uint32_t n = 0; n < 2u; ++n) {
                    int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                        : "r"(A[n][kk][0]), "r"(A[n][kk][1]), "r"(A[n][kk][2]),
                          "r"(A[n][kk][3]), "r"(b0), "r"(b1));
                    acc[(j0 >> 3) + n][0] += dA[n][0][kk] * dB0 * (float)d0;
                    acc[(j0 >> 3) + n][1] += dA[n][0][kk] * dB1 * (float)d1;
                    acc[(j0 >> 3) + n][2] += dA[n][1][kk] * dB0 * (float)d2;
                    acc[(j0 >> 3) + n][3] += dA[n][1][kk] * dB1 * (float)d3;
                }
            }
        }
        __syncthreads();
    }

    // partial layout: [split][local tile][128x128] with the same per-thread
    // element mapping as the write-out (the reduce re-derives it)
    float* pt = partials + ((size_t)sp * gridDim.x / splits + lt) * (128u * 128u);
    #pragma unroll
    for (uint32_t j0 = 0; j0 < 128u; j0 += 16u) {
        const uint32_t c0 = j0 + joff + 2u * t;
        #pragma unroll
        for (uint32_t n = 0; n < 2u; ++n) {
            const uint32_t r0 = i0 + n * 16u + g;
            const uint32_t r8 = r0 + 8u;
            pt[(size_t)c0 * 128u + r0] = acc[(j0 >> 3) + n][0];
            pt[(size_t)(c0 + 1u) * 128u + r0] = acc[(j0 >> 3) + n][1];
            pt[(size_t)c0 * 128u + r8] = acc[(j0 >> 3) + n][2];
            pt[(size_t)(c0 + 1u) * 128u + r8] = acc[(j0 >> 3) + n][3];
        }
    }
#else
    (void)data; (void)scale; (void)yq; (void)partials;
    (void)in_dim; (void)out_dim; (void)batch; (void)tile_off; (void)splits;
#endif
}

// Sum the S partials of each tail tile into y (fixed order: s ascending).
__global__ void pd_mmq_sk_reduce_kernel(const float* __restrict__ partials,
                                        float* __restrict__ y, uint32_t in_dim,
                                        uint32_t out_dim, uint32_t batch,
                                        uint32_t tile_off, uint32_t n_tail,
                                        uint32_t splits) {
    const uint32_t lt = blockIdx.y;
    const uint32_t e = blockIdx.x * blockDim.x + threadIdx.x;  // elem in tile
    if (e >= 128u * 128u) return;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t nct = batch_pad >> 7;
    const uint32_t tile = tile_off + lt;
    const uint32_t row = (tile / nct) * 128u + (e & 127u);
    const uint32_t col = (tile % nct) * 128u + (e >> 7);
    if (row >= out_dim || col >= batch) return;
    float acc = 0.f;
    for (uint32_t sp = 0; sp < splits; ++sp)
        acc += partials[((size_t)sp * n_tail + lt) * (128u * 128u) + e];
    y[(size_t)col * out_dim + row] = acc;
}

// Orchestrating launcher: full waves on the untouched pipe kernel, the tail
// as split-K spans + reduce. `partials` must hold tail x splits x 128x128
// f32 (the engine passes a persistent plane; tail <= sm_count/2, splits<=4).
PD_EXPORT
int pd_q8_0_gemm_mmq_pipe_sk(const void* data, const void* scale, const void* yq,
                             void* y, void* partials, uint32_t in_dim,
                             uint32_t out_dim, uint32_t batch, uint32_t sm_count,
                             void* stream) {
    if (out_dim == 0 || batch == 0) return 0;
    // out_dim needs no alignment (same ragged-tolerant family);
    // in_dim keeps %128 - yq is 128-K packed, a real format constraint.
    if (in_dim & 127u) return cudaErrorInvalidValue;
    auto st = (cudaStream_t)stream;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t nct = batch_pad >> 7;
    const uint32_t ntiles = ((out_dim + 127u) / 128u) * nct;
    const uint32_t tail = ntiles % sm_count;
    // engage only when the tail wastes a big slice of the last wave and the
    // K depth is worth splitting; otherwise the plain kernel is optimal
    const uint32_t n_chunks = in_dim >> 7;
    if (tail == 0 || tail > sm_count / 2 || n_chunks < 8 || partials == NULL) {
        return pd_q8_0_gemm_mmq_pipe(data, scale, yq, nullptr, y, in_dim, out_dim, batch, stream);
    }
    const uint32_t full = ntiles - tail;
    uint32_t splits = sm_count / tail;
    if (splits > 4u) splits = 4u;
    if (splits < 2u) splits = 2u;
    static cudaError_t attr = cudaFuncSetAttribute(
        (const void*)pd_q8_0_gemm_mmq_pipe_span_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)PD_MMQ_PIPE_SMEM);
    if (attr != cudaSuccess) return attr;
    // the full-wave part launches the PLAIN kernel directly - its smem
    // opt-in must be set here too (the plain launcher may not have run yet)
    static cudaError_t attr_full = cudaFuncSetAttribute(
        (const void*)pd_q8_0_gemm_mmq_pipe_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)PD_MMQ_PIPE_SMEM);
    if (attr_full != cudaSuccess) return attr_full;
    if (full > 0) {
        // tiles [0, full) on the unchanged kernel - one bit-identical launch
        pd_q8_0_gemm_mmq_pipe_kernel<<<full, 256, PD_MMQ_PIPE_SMEM, st>>>(
            (const int8_t*)data, (const __half*)scale, (const uint8_t*)yq,
            (const float*)nullptr, (float*)y, in_dim, out_dim, batch);
    }
    pd_q8_0_gemm_mmq_pipe_span_kernel<<<tail * splits, 256, PD_MMQ_PIPE_SMEM, st>>>(
        (const int8_t*)data, (const __half*)scale, (const uint8_t*)yq,
        (float*)partials, in_dim, out_dim, batch, full, splits);
    dim3 rg((128u * 128u + 255u) / 256u, tail);
    pd_mmq_sk_reduce_kernel<<<rg, 256, 0, st>>>(
        (const float*)partials, (float*)y, in_dim, out_dim, batch, full, tail, splits);
    return pd_launch_status();
}


// ---------------------------------------- pipe-i (interleaved-word variant)
// Next rung on the pipe: its inner loop issues
// ~168 smem loads per chunk per thread - L1TEX 46.8% SOL / 0.6 eligible
// warps with nothing else saturated (DRAM 10%, L2 hit 89%) - the kernel is
// smem-ISSUE-bound. The mma fragment layout needs word pairs (kk*8+t,
// kk*8+4+t); in the classic layout they sit 16B apart -> two 4B loads each
// for A and B. Interleaving the DATA WORD ORDER (pairs adjacent) turns both
// into single 8B loads: ~120 loads/chunk/thread (-29%). Fold order is
// UNCHANGED (same kk sequence) -> bit-equal to the classic pipe.
//   word permutation: old word w (kk = w>>3, r = w&7) -> kk*8 + (r<4 ?
//   2r : 2(r-4)+1); scales stay in place.
// The Y layout change lives in pd_quantize_q8_mmq_i (+ _geglu_i twin) - the
// serving lane picks quantizer+GEMM as a pair per tick (r>1024 lane only).
__device__ __forceinline__ uint32_t pd_mmqi_perm(uint32_t w) {
    const uint32_t kk = w >> 3, r = w & 7u;
    return kk * 8u + (r < 4u ? 2u * r : 2u * (r - 4u) + 1u);
}

__global__ void pd_quantize_q8_mmq_i_kernel(const float* __restrict__ x,
                                            uint8_t* __restrict__ yq,
                                            uint32_t in_dim, uint32_t batch) {
    const uint32_t chunk = blockIdx.x, col = blockIdx.y, lane = threadIdx.x;
    uint8_t* blk = yq + ((size_t)chunk * gridDim.y + col) * 144u;
    const uint32_t k0 = chunk * 128u + lane * 4u;
    float v[4] = {};
    if (col < batch) {
        #pragma unroll
        for (uint32_t j = 0; j < 4u; ++j)
            if (k0 + j < in_dim) v[j] = x[(size_t)col * in_dim + k0 + j];
    }
    float a = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
    #pragma unroll
    for (uint32_t s = 4; s > 0; s >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, s));
    const float scl = a * (1.0f / 127.0f);
    const float inv = scl > 0.0f ? 1.0f / scl : 0.0f;
    char4 q;
    int qi;
    qi = __float2int_rn(v[0] * inv); q.x = (char)(qi < -127 ? -127 : (qi > 127 ? 127 : qi));
    qi = __float2int_rn(v[1] * inv); q.y = (char)(qi < -127 ? -127 : (qi > 127 ? 127 : qi));
    qi = __float2int_rn(v[2] * inv); q.z = (char)(qi < -127 ? -127 : (qi > 127 ? 127 : qi));
    qi = __float2int_rn(v[3] * inv); q.w = (char)(qi < -127 ? -127 : (qi > 127 ? 127 : qi));
    ((char4*)(blk + 16u))[pd_mmqi_perm(lane)] = q;
    if ((lane & 7u) == 0u) ((float*)blk)[lane >> 3] = scl;
}

PD_EXPORT
int pd_quantize_q8_mmq_i(const void* x, void* yq, uint32_t in_dim, uint32_t batch,
                         void* stream) {
    if (in_dim == 0 || batch == 0) return 0;
    const uint32_t n_chunks = (in_dim + 127u) / 128u;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    dim3 grid(n_chunks, batch_pad);
    pd_quantize_q8_mmq_i_kernel<<<grid, 32, 0, (cudaStream_t)stream>>>(
        (const float*)x, (uint8_t*)yq, in_dim, batch);
    return pd_launch_status();
}

// W staging with the same word interleave: 4B copies to permuted slots
// (16 iters vs the classic 4x16B - staging is once per chunk and pipelined;
// the compute loop's halved A loads repay it hundreds of times over).
__device__ __forceinline__ void pd_mmqp_issue_w_i(
    int* __restrict__ wbuf, const int8_t* __restrict__ data, const __half* __restrict__ scale,
    uint32_t row_base, uint32_t out_dim, uint32_t n_k32, uint32_t n_blocks,
    uint32_t in_dim, uint32_t kc, uint32_t tid) {
#if PD_MMA_OK
    #pragma unroll
    for (uint32_t it = 0; it < 16u; ++it) {  // 16*256 == 128 rows * 32 words
        const uint32_t i = it * 256u + tid;
        const uint32_t row = i >> 5, w = i & 31u;
        const uint32_t gk4 = kc * 32u + w;
        const bool ok = gk4 < n_k32 && (row_base + row) < out_dim;
        pd_cpa4p(wbuf + row * PD_MMQ_PIPE_WK + pd_mmqi_perm(w),
                 (const char*)data + (size_t)(row_base + row) * in_dim
                     + (size_t)(kc * 128u + w * 4u), ok);
    }
    if (tid < 128u) {
        const uint32_t gb = kc * 4u;
        const bool row_ok = (row_base + tid) < out_dim;
        char* dst = (char*)(wbuf + tid * PD_MMQ_PIPE_WK + 32u);
        const char* src = (const char*)(scale + (size_t)(row_base + tid) * n_blocks + gb);
        pd_cpa4p(dst, src, row_ok && (gb + 1u < n_blocks));
        pd_cpa4p(dst + 4u, src + 4u, row_ok && (gb + 3u < n_blocks));
    }
#endif
}

__global__ void __launch_bounds__(256, 1) pd_q8_0_gemm_mmq_pipe_i_kernel(
        const int8_t* __restrict__ data, const __half* __restrict__ scale,
        const uint8_t* __restrict__ yq, const float* __restrict__ bias,
        float* __restrict__ y,
        uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
#if PD_MMA_OK
    extern __shared__ int pd_mmqpi_sh[];
    int* wbuf0 = pd_mmqpi_sh;
    int* wbuf1 = wbuf0 + 128 * PD_MMQ_PIPE_WK;
    int* ybuf0 = wbuf1 + 128 * PD_MMQ_PIPE_WK;
    int* ybuf1 = ybuf0 + 128 * PD_MMQ_YK;

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, t = lane & 3u;
    const uint32_t i0 = (warp >> 1) * 32u;
    const uint32_t joff = (warp & 1u) * 8u;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t n_k32 = in_dim >> 2;
    const uint32_t n_blocks = in_dim >> 5;
    const uint32_t n_chunks = (in_dim + 127u) >> 7;
    const uint32_t nct = batch_pad >> 7;

    const uint32_t tile = blockIdx.x;
    const uint32_t row_base = (tile / nct) * 128u;
    const uint32_t col_base = (tile % nct) * 128u;

    float acc[16][4] = {};
    pd_mmqp_issue_w(wbuf0, data, scale, row_base, out_dim, n_k32, n_blocks, in_dim, 0u, tid);
    {
        const int* by0 = (const int*)(yq + ((size_t)0u * batch_pad + col_base) * 144u);
        #pragma unroll
        for (uint32_t it = 0; it < 5u; ++it)
            if (it * 256u + tid < 1152u)
                pd_cpa16p(ybuf0 + (it * 256u + tid) * 4u,
                          (const char*)by0 + (size_t)(it * 256u + tid) * 16u, true);
    }
    asm volatile("cp.async.commit_group;");

    for (uint32_t kc = 0; kc < n_chunks; ++kc) {
        int* tw = (kc & 1u) ? wbuf1 : wbuf0;
        int* ty = (kc & 1u) ? ybuf1 : ybuf0;
        if (kc + 1u < n_chunks) {
            int* nw = (kc & 1u) ? wbuf0 : wbuf1;
            int* ny = (kc & 1u) ? ybuf0 : ybuf1;
            pd_mmqp_issue_w(nw, data, scale, row_base, out_dim, n_k32, n_blocks, in_dim, kc + 1u, tid);
            const int* by1 = (const int*)(yq + ((size_t)(kc + 1u) * batch_pad + col_base) * 144u);
            #pragma unroll
            for (uint32_t it = 0; it < 5u; ++it)
                if (it * 256u + tid < 1152u)
                    pd_cpa16p(ny + (it * 256u + tid) * 4u,
                              (const char*)by1 + (size_t)(it * 256u + tid) * 16u, true);
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();

        int A[2][4][4];
        float dA[2][2][4];
        #pragma unroll
        for (uint32_t n = 0; n < 2u; ++n) {
            const uint32_t r0 = (i0 + n * 16u + g) * PD_MMQ_PIPE_WK;
            const uint32_t r8 = (i0 + n * 16u + 8u + g) * PD_MMQ_PIPE_WK;
            #pragma unroll
            for (uint32_t kk = 0; kk < 4u; ++kk) {
                const uint32_t ko = kk * 8u;
                // W classic (16B-staged raw order); the Y side carries the
                // interleave - the 4B-staged W permute cost 4x transactions
                // on the weight stream (+13% at the batch-128 bandwidth floor)
                A[n][kk][0] = tw[r0 + ko + t];
                A[n][kk][1] = tw[r8 + ko + t];
                A[n][kk][2] = tw[r0 + ko + 4u + t];
                A[n][kk][3] = tw[r8 + ko + 4u + t];
                dA[n][0][kk] = __half2float(((const __half*)(tw + r0 + 32u))[kk]);
                dA[n][1][kk] = __half2float(((const __half*)(tw + r8 + 32u))[kk]);
            }
        }
        #pragma unroll
        for (uint32_t j0 = 0; j0 < 128u; j0 += 16u) {
            const uint32_t jc = j0 + joff;
            #pragma unroll
            for (uint32_t kk = 0; kk < 4u; ++kk) {
                const uint32_t ko = kk * 8u;
                const int2 b01 = *reinterpret_cast<const int2*>(
                    &ty[(jc + g) * PD_MMQ_YK + 4u + ko + 2u * t]);
                const float dB0 = ((const float*)ty)[(jc + 2u * t) * PD_MMQ_YK + kk];
                const float dB1 = ((const float*)ty)[(jc + 2u * t + 1u) * PD_MMQ_YK + kk];
                #pragma unroll
                for (uint32_t n = 0; n < 2u; ++n) {
                    int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                        : "r"(A[n][kk][0]), "r"(A[n][kk][1]), "r"(A[n][kk][2]),
                          "r"(A[n][kk][3]), "r"(b01.x), "r"(b01.y));
                    acc[(j0 >> 3) + n][0] += dA[n][0][kk] * dB0 * (float)d0;
                    acc[(j0 >> 3) + n][1] += dA[n][0][kk] * dB1 * (float)d1;
                    acc[(j0 >> 3) + n][2] += dA[n][1][kk] * dB0 * (float)d2;
                    acc[(j0 >> 3) + n][3] += dA[n][1][kk] * dB1 * (float)d3;
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (uint32_t j0 = 0; j0 < 128u; j0 += 16u) {
        const uint32_t c0 = col_base + j0 + joff + 2u * t;
        #pragma unroll
        for (uint32_t n = 0; n < 2u; ++n) {
            const uint32_t r0 = row_base + i0 + n * 16u + g;
            const uint32_t r8 = r0 + 8u;
            const float b0 = (bias && r0 < out_dim) ? bias[r0] : 0.0f;
            const float b8 = (bias && r8 < out_dim) ? bias[r8] : 0.0f;
            if (r0 < out_dim) {
                if (c0 < batch) y[(size_t)c0 * out_dim + r0] = acc[(j0 >> 3) + n][0] + b0;
                if (c0 + 1u < batch) y[(size_t)(c0 + 1u) * out_dim + r0] = acc[(j0 >> 3) + n][1] + b0;
            }
            if (r8 < out_dim) {
                if (c0 < batch) y[(size_t)c0 * out_dim + r8] = acc[(j0 >> 3) + n][2] + b8;
                if (c0 + 1u < batch) y[(size_t)(c0 + 1u) * out_dim + r8] = acc[(j0 >> 3) + n][3] + b8;
            }
        }
    }
#else
    (void)data; (void)scale; (void)yq; (void)bias; (void)y;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}

PD_EXPORT
int pd_q8_0_gemm_mmq_pipe_i(const void* data, const void* scale, const void* yq,
                            const void* bias, void* y, uint32_t in_dim,
                            uint32_t out_dim, uint32_t batch, void* stream) {
    if (out_dim == 0 || batch == 0) return 0;
    // out_dim needs no alignment (same ragged-tolerant family).
    if (in_dim & 31u) return cudaErrorInvalidValue;
    static cudaError_t attr = cudaFuncSetAttribute(
        (const void*)pd_q8_0_gemm_mmq_pipe_i_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)PD_MMQ_PIPE_SMEM);
    if (attr != cudaSuccess) return attr;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t nct = batch_pad >> 7;
    const uint32_t ntiles = ((out_dim + 127u) / 128u) * nct;
    pd_q8_0_gemm_mmq_pipe_i_kernel<<<ntiles, 256, PD_MMQ_PIPE_SMEM, (cudaStream_t)stream>>>(
        (const int8_t*)data, (const __half*)scale, (const uint8_t*)yq,
        (const float*)bias, (float*)y, in_dim, out_dim, batch);
    return pd_launch_status();
}
