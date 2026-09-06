#include <cuda_bf16.h>
// moe/q8.cuh (formerly 16_q8_moe.cuh) - Q8_0 routed/sorted/int8-MMA MoE (qwen3.6-A3B class)
// Textually-included segment of the single pack translation unit.
// Not standalone-compilable: include order is defined by ../pack.cu.
// ---- Q8_0 routed-expert MoE (qwen3.6-A3B class: many small experts) --------
// Experts arrive as Q8_0 (repacked: int8 rows + f16 block scales), not mxfp4,
// so the gpt-oss MoE family does not apply. Same dp4a numeric class as the
// dense q8 serving GEMMs (int8 x int8, f32 block-scale accumulate).

// Fused gate+up+act over routed experts, token-batched: grid
// (ff, n_active, batch), one block per (out row, slot, token). Weight row for
// (expert e, out o) sits at (e*ff + o) in the repacked stream; both dots
// share one pass over the token's int8 activations. ff is small on this
// class (512) - per block work is 2 x in_dim bytes of weights, and the
// grid (512 x 8 x B blocks) fills the die from B=1 (4096 blocks).
// GELU=false is the qwen SwiGLU original; GELU=true is the gemma4-A4B twin
// (gelu_tanh(gate)*up, exactly pd_geglu's constants so the MoE branch sits
// in the same numeric class as the dense gemma4 FFN).
template <bool GELU>
__global__ void __launch_bounds__(256) pd_q8_0_moe_gate_up_dp4a_kernel(
    const int8_t* __restrict__ gate_data, const __half* __restrict__ gate_scale,
    const int8_t* __restrict__ up_data, const __half* __restrict__ up_scale,
    const unsigned int* __restrict__ idx, const int8_t* __restrict__ xq,
    const float* __restrict__ xs, float* __restrict__ out, uint32_t in_dim,
    uint32_t ff, uint32_t n_active) {
    const uint32_t o = blockIdx.x, slot = blockIdx.y, b = blockIdx.z;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    const uint32_t n_blocks = in_dim >> 5;
    const uint32_t e = idx[(size_t)b * n_active + slot];
    const int8_t* grow = gate_data + ((size_t)e * ff + o) * in_dim;
    const __half* gsc = gate_scale + ((size_t)e * ff + o) * n_blocks;
    const int8_t* urow = up_data + ((size_t)e * ff + o) * in_dim;
    const __half* usc = up_scale + ((size_t)e * ff + o) * n_blocks;
    const int8_t* xrow = xq + (size_t)b * in_dim;
    const float* xsc = xs + (size_t)b * n_blocks;

    float accg = 0.0f, accu = 0.0f;
    for (uint32_t base = tid * 16u; base < in_dim; base += nth * 16u) {
        const float x_s = xsc[base >> 5];
        const int4 xv = *reinterpret_cast<const int4*>(xrow + base);
        int4 wv = __ldcs(reinterpret_cast<const int4*>(grow + base));
        int s = __dp4a(wv.x, xv.x, 0);
        s = __dp4a(wv.y, xv.y, s);
        s = __dp4a(wv.z, xv.z, s);
        s = __dp4a(wv.w, xv.w, s);
        accg += __half2float(__ldcs(gsc + (base >> 5))) * x_s * (float)s;
        wv = __ldcs(reinterpret_cast<const int4*>(urow + base));
        s = __dp4a(wv.x, xv.x, 0);
        s = __dp4a(wv.y, xv.y, s);
        s = __dp4a(wv.z, xv.z, s);
        s = __dp4a(wv.w, xv.w, s);
        accu += __half2float(__ldcs(usc + (base >> 5))) * x_s * (float)s;
    }
    __shared__ float wsum[2][8];
    const uint32_t lane = tid & 31u, warp = tid >> 5, nwarps = (nth + 31u) >> 5;
    for (uint32_t s2 = 16; s2 > 0; s2 >>= 1) {
        accg += __shfl_down_sync(0xffffffffu, accg, s2);
        accu += __shfl_down_sync(0xffffffffu, accu, s2);
    }
    if (lane == 0) { wsum[0][warp] = accg; wsum[1][warp] = accu; }
    __syncthreads();
    if (tid == 0) {
        float g = 0.0f, u = 0.0f;
        for (uint32_t w = 0; w < nwarps; ++w) { g += wsum[0][w]; u += wsum[1][w]; }
        // silu(g) * u, or gelu_tanh(g) * u on the gemma4 twin
        const float act = GELU
            ? 0.5f * g * (1.0f + tanhf(0.79788456080286535587989211986876f * g
                                       * (1.0f + 0.044715f * g * g)))
            : (g / (1.0f + __expf(-g)));
        out[((size_t)b * n_active + slot) * ff + o] = act * u;
    }
}

PD_EXPORT
int pd_q8_0_moe_gate_up_dp4a(const void* gate_data, const void* gate_scale,
                             const void* up_data, const void* up_scale, const void* idx,
                             const void* xq, const void* xs, void* out, uint32_t in_dim,
                             uint32_t ff, uint32_t n_active, uint32_t batch, void* stream) {
    if (ff == 0 || n_active == 0 || batch == 0) return 0;
    if ((in_dim & 31u) != 0) return cudaErrorInvalidValue;
    dim3 grid(ff, n_active, batch);
    pd_q8_0_moe_gate_up_dp4a_kernel<false><<<grid, 256, 0, (cudaStream_t)stream>>>(
        (const int8_t*)gate_data, (const __half*)gate_scale, (const int8_t*)up_data,
        (const __half*)up_scale, (const unsigned int*)idx, (const int8_t*)xq,
        (const float*)xs, (float*)out, in_dim, ff, n_active);
    return pd_launch_status();
}

// GEGLU twin of the launcher above - the gemma4-A4B hybrid FFN's routed
// branch (LLM_FFN_GELU experts). Same layout/signature; only the epilogue
// activation differs.
PD_EXPORT
int pd_q8_0_moe_gate_up_dp4a_geglu(const void* gate_data, const void* gate_scale,
                                   const void* up_data, const void* up_scale, const void* idx,
                                   const void* xq, const void* xs, void* out, uint32_t in_dim,
                                   uint32_t ff, uint32_t n_active, uint32_t batch, void* stream) {
    if (ff == 0 || n_active == 0 || batch == 0) return 0;
    if ((in_dim & 31u) != 0) return cudaErrorInvalidValue;
    dim3 grid(ff, n_active, batch);
    pd_q8_0_moe_gate_up_dp4a_kernel<true><<<grid, 256, 0, (cudaStream_t)stream>>>(
        (const int8_t*)gate_data, (const __half*)gate_scale, (const int8_t*)up_data,
        (const __half*)up_scale, (const unsigned int*)idx, (const int8_t*)xq,
        (const float*)xs, (float*)out, in_dim, ff, n_active);
    return pd_launch_status();
}

// Fold per-expert scalars into routed top-k weights: w[i] *= scale[idx[i]].
// gemma4-A4B's `ffn_down_exps.scale` multiplies each expert's down output;
// since the combine is sum_e w_e * (s_e * down_e), folding s into w before
// the down kernel keeps the reference math with zero changes to the GEMMs
// (f32-associative - greedy-match class, same as the plan's other folds).
__global__ void pd_moe_scale_w_kernel(float* __restrict__ w,
                                      const unsigned int* __restrict__ idx,
                                      const float* __restrict__ scale, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) w[i] *= scale[idx[i]];
}

PD_EXPORT
int pd_moe_scale_w(void* w, const void* idx, const void* scale, uint32_t n, void* stream) {
    if (n == 0) return 0;
    pd_moe_scale_w_kernel<<<(n + 255u) / 256u, 256, 0, (cudaStream_t)stream>>>(
        (float*)w, (const unsigned int*)idx, (const float*)scale, n);
    return pd_launch_status();
}

// Routed-expert down + weighted combine: out[b][o] = sum_slot topk_w *
// dot(down[e][o], fused_q[b][slot]). grid (embd, batch); warp w owns slot w
// (n_active <= 16, launcher sizes the block to 32*n_active), lanes stride
// the ff dimension in 16-byte segments - at ff = 512 one pass covers it.
// Plain write (the caller adds the shared expert and the residual). 16
// matches pd_moe_topk_warp's top-k ceiling - was hard-capped at 8 (XS-2.1's
// top-8); Laguna S-2.1's top-10 MoE hit the cap, same
// bug as kquant.cuh's k-quant sibling.
__global__ void __launch_bounds__(512) pd_q8_0_moe_down_dp4a_kernel(
    const int8_t* __restrict__ down_data, const __half* __restrict__ down_scale,
    const unsigned int* __restrict__ idx, const float* __restrict__ topk_w,
    const int8_t* __restrict__ fq, const float* __restrict__ fs,
    float* __restrict__ out, uint32_t ff, uint32_t embd, uint32_t n_active) {
    const uint32_t o = blockIdx.x, b = blockIdx.y;
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const uint32_t n_blocks = ff >> 5;
    __shared__ float sh[16];
    if (warp < n_active) {
        const size_t srow = (size_t)b * n_active + warp;
        const uint32_t e = idx[srow];
        const int8_t* row = down_data + ((size_t)e * embd + o) * ff;
        const __half* rsc = down_scale + ((size_t)e * embd + o) * n_blocks;
        const int8_t* xrow = fq + srow * ff;
        const float* xsc = fs + srow * n_blocks;
        float acc = 0.0f;
        for (uint32_t base = lane * 16u; base < ff; base += 32u * 16u) {
            const int4 wv = __ldcs(reinterpret_cast<const int4*>(row + base));
            const int4 xv = *reinterpret_cast<const int4*>(xrow + base);
            int s = __dp4a(wv.x, xv.x, 0);
            s = __dp4a(wv.y, xv.y, s);
            s = __dp4a(wv.z, xv.z, s);
            s = __dp4a(wv.w, xv.w, s);
            acc += __half2float(__ldcs(rsc + (base >> 5))) * xsc[base >> 5] * (float)s;
        }
        for (uint32_t s2 = 16; s2 > 0; s2 >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, s2);
        if (lane == 0) sh[warp] = topk_w[srow] * acc;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float v = 0.0f;
        for (uint32_t w = 0; w < n_active; ++w) v += sh[w];
        out[(size_t)b * embd + o] = v;
    }
}

PD_EXPORT
int pd_q8_0_moe_down_dp4a(const void* down_data, const void* down_scale, const void* idx,
                          const void* topk_w, const void* fq, const void* fs, void* out,
                          uint32_t ff, uint32_t embd, uint32_t n_active, uint32_t batch,
                          void* stream) {
    if (embd == 0 || n_active == 0 || batch == 0) return 0;
    if ((ff & 31u) != 0 || n_active > 16u) return cudaErrorInvalidValue;
    dim3 grid(embd, batch);
    pd_q8_0_moe_down_dp4a_kernel<<<grid, 32u * n_active, 0, (cudaStream_t)stream>>>(
        (const int8_t*)down_data, (const __half*)down_scale, (const unsigned int*)idx,
        (const float*)topk_w, (const int8_t*)fq, (const float*)fs, (float*)out, ff, embd,
        n_active);
    return pd_launch_status();
}

// Token-batched expert up + squared-relu (nemotron_h_moe class: experts carry
// no gate matrix; activation is relu(up(x))^2 - llama.cpp's LLM_FFN_RELU_SQR,
// vLLM's relu2). Single-plane twin of pd_q8_0_moe_gate_up_dp4a_kernel: same
// grid (ff, n_active, batch), same dp4a class, half the weight streams.
// Serves the shared expert too: n_active=1 with idx pointing at a constant 0
// over the 1-expert shared plane (the nvf4 lane's own convention).
__global__ void __launch_bounds__(256) pd_q8_0_moe_up_relu2_dp4a_kernel(
    const int8_t* __restrict__ up_data, const __half* __restrict__ up_scale,
    const unsigned int* __restrict__ idx, const int8_t* __restrict__ xq,
    const float* __restrict__ xs, float* __restrict__ out, uint32_t in_dim,
    uint32_t ff, uint32_t n_active) {
    const uint32_t o = blockIdx.x, slot = blockIdx.y, b = blockIdx.z;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    const uint32_t n_blocks = in_dim >> 5;
    const uint32_t e = idx[(size_t)b * n_active + slot];
    const int8_t* urow = up_data + ((size_t)e * ff + o) * in_dim;
    const __half* usc = up_scale + ((size_t)e * ff + o) * n_blocks;
    const int8_t* xrow = xq + (size_t)b * in_dim;
    const float* xsc = xs + (size_t)b * n_blocks;

    float accu = 0.0f;
    for (uint32_t base = tid * 16u; base < in_dim; base += nth * 16u) {
        const int4 xv = *reinterpret_cast<const int4*>(xrow + base);
        const int4 wv = __ldcs(reinterpret_cast<const int4*>(urow + base));
        int s = __dp4a(wv.x, xv.x, 0);
        s = __dp4a(wv.y, xv.y, s);
        s = __dp4a(wv.z, xv.z, s);
        s = __dp4a(wv.w, xv.w, s);
        accu += __half2float(__ldcs(usc + (base >> 5))) * xsc[base >> 5] * (float)s;
    }
    __shared__ float wsum[8];
    const uint32_t lane = tid & 31u, warp = tid >> 5, nwarps = (nth + 31u) >> 5;
    for (uint32_t s2 = 16; s2 > 0; s2 >>= 1) accu += __shfl_down_sync(0xffffffffu, accu, s2);
    if (lane == 0) wsum[warp] = accu;
    __syncthreads();
    if (tid == 0) {
        float u = 0.0f;
        for (uint32_t w = 0; w < nwarps; ++w) u += wsum[w];
        const float v = fmaxf(u, 0.0f);
        out[((size_t)b * n_active + slot) * ff + o] = v * v;
    }
}

PD_EXPORT
int pd_q8_0_moe_up_relu2_dp4a(const void* up_data, const void* up_scale, const void* idx,
                              const void* xq, const void* xs, void* out, uint32_t in_dim,
                              uint32_t ff, uint32_t n_active, uint32_t batch, void* stream) {
    if (ff == 0 || n_active == 0 || batch == 0) return 0;
    if ((in_dim & 31u) != 0) return cudaErrorInvalidValue;
    dim3 grid(ff, n_active, batch);
    pd_q8_0_moe_up_relu2_dp4a_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(
        (const int8_t*)up_data, (const __half*)up_scale, (const unsigned int*)idx,
        (const int8_t*)xq, (const float*)xs, (float*)out, in_dim, ff, n_active);
    return pd_launch_status();
}

// ---- relu2 decode-band expert up (the dec2 class, single plane) ------------
// Single-plane sibling of pd_q8_0_moe_gu_dec2_kernel (moe/f8.cuh) for the
// nemotron_h_moe shape: one weight matrix, relu(x)^2, no gate. It exists
// because both neighbours above are the wrong shape at decode widths
// (measured, A6000 c4):
//   - the sorted tile below packs (row, expert) pairs into BM=32 blocks, and
//     at r=4/top-6 over 128 experts nearly every block holds one real row and
//     computes 32. The routed pair ran at ~22% of this card's stream roof.
//   - the dp4a original above spends a whole 256-thread CTA per OUTPUT
//     element: 44k CTAs at r=4, and at in_dim 2688 only 168 of the 256
//     threads have an int4 window to load before a full block reduction.
// The middle is dec2's: warp per output row, ROWS rows per CTA, each lane
// streaming int4 windows with __ldcs. No pad rows, no smem staging, no block
// reduction - lane 0 writes its own row. The activation row is re-read by
// every warp in the CTA, which is an L1 hit after the first.
//
// Not a dedup lane, and that is what bounds the band: two rows routed to the
// same expert stream its plane twice where the sorted tile shares one block.
// At r=4/top-6 over 128 experts ~22 of the 24 picks are distinct (9% re-read);
// by r=32 the picks collide ~2x and sorted wins on bytes. The engine gates on
// the measured crossover - see gpu_model/nemotron/batch.rs.
//
// REORDER class vs pd_q8_0_moe_up_relu2_dp4a (per-lane partials over 32 lanes
// instead of per-thread over 256), same as qwen's dec2 pair - hence a
// separate export rather than a widened launcher.
#define PD_QMOE_DEC2_ROWS 8u   // elected rows/CTA

template <uint32_t ROWS>
__global__ void __launch_bounds__(ROWS * 32u) pd_q8_0_moe_up_relu2_dec2_kernel(
    const int8_t* __restrict__ up_data, const __half* __restrict__ up_scale,
    const unsigned int* __restrict__ idx, const int8_t* __restrict__ xq,
    const float* __restrict__ xs, float* __restrict__ out, uint32_t in_dim,
    uint32_t ff, uint32_t n_active) {
    const uint32_t o = blockIdx.x * ROWS + (threadIdx.x >> 5);
    const uint32_t slot = blockIdx.y, b = blockIdx.z;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t n_blocks = in_dim >> 5;
    const uint32_t e = idx[(size_t)b * n_active + slot];
    // clamp keeps the row address in range on the ff tail; the store is guarded
    const uint32_t oc = o < ff ? o : ff - 1u;
    const int8_t* urow = up_data + ((size_t)e * ff + oc) * in_dim;
    const __half* usc = up_scale + ((size_t)e * ff + oc) * n_blocks;
    const int8_t* xrow = xq + (size_t)b * in_dim;
    const float* xsc = xs + (size_t)b * n_blocks;

    float acc = 0.0f;
    // in_dim is a multiple of 32 (q8 block granularity), so every 16-byte
    // window lies wholly inside one scale block and base <= in_dim - 16
    for (uint32_t base = lane * 16u; base < in_dim; base += 32u * 16u) {
        const int4 xv = *reinterpret_cast<const int4*>(xrow + base);
        const int4 wv = __ldcs(reinterpret_cast<const int4*>(urow + base));
        int s = __dp4a(wv.x, xv.x, 0);
        s = __dp4a(wv.y, xv.y, s);
        s = __dp4a(wv.z, xv.z, s);
        s = __dp4a(wv.w, xv.w, s);
        acc += __half2float(__ldcs(usc + (base >> 5))) * xsc[base >> 5] * (float)s;
    }
    for (uint32_t s2 = 16; s2 > 0; s2 >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, s2);
    if (lane == 0 && o < ff) {
        const float v = fmaxf(acc, 0.0f);
        out[((size_t)b * n_active + slot) * ff + o] = v * v;
    }
}

// rows_pb: 0 takes the elected default. The parameter is a LAB instrument
// (examples/nemo_moe_kbench.rs sweeps it); the engine always passes 0, so the
// shipped route carries no tuning surface.
PD_EXPORT
int pd_q8_0_moe_up_relu2_dec2(const void* up_data, const void* up_scale, const void* idx,
                              const void* xq, const void* xs, void* out, uint32_t in_dim,
                              uint32_t ff, uint32_t n_active, uint32_t batch,
                              uint32_t rows_pb, void* stream) {
    if (ff == 0 || n_active == 0 || batch == 0) return 0;
    if ((in_dim & 31u) != 0) return cudaErrorInvalidValue;  // q8 block granularity
    if (rows_pb == 0) rows_pb = PD_QMOE_DEC2_ROWS;
#define PD_DEC2_UP_LAUNCH(R)                                                        \
    {                                                                               \
        dim3 grid((ff + (R) - 1u) / (R), n_active, batch);                          \
        pd_q8_0_moe_up_relu2_dec2_kernel<R><<<grid, (R) * 32u, 0,                    \
                                              (cudaStream_t)stream>>>(              \
            (const int8_t*)up_data, (const __half*)up_scale,                        \
            (const unsigned int*)idx, (const int8_t*)xq, (const float*)xs,          \
            (float*)out, in_dim, ff, n_active);                                     \
    }
    switch (rows_pb) {
        case 2u: PD_DEC2_UP_LAUNCH(2u) break;
        case 4u: PD_DEC2_UP_LAUNCH(4u) break;
        case 8u: PD_DEC2_UP_LAUNCH(8u) break;
        case 16u: PD_DEC2_UP_LAUNCH(16u) break;
        default: return cudaErrorInvalidValue;
    }
#undef PD_DEC2_UP_LAUNCH
    return pd_launch_status();
}

// ---- sorted Q8_0 MoE (the prefill/serving class) ---------------------------
// Reads the moe_align layout: 32-row blocks of same-expert (token, slot)
// pairs, so each expert's weights stream from DRAM once per pass regardless
// of token count - the token-batched kernels above re-read routed rows per
// token, which is why bring-up prefill sat at 0.18x llama. Same dp4a int8
// numeric class as the token-batched pair.
#define PD_QMOE_BM 32u
#define PD_QMOE_BN 16u
#define PD_QMOE_BK 256u
// shared strides in int32 words, padded to dodge bank conflicts on the
// row-indexed x reads (stride 64 puts every row's word k in bank k%32)
#define PD_QMOE_XW 65u
#define PD_QMOE_WW 65u

// Fused sorted gate+up+SwiGLU: grid (ceil(ff/BN), max_blocks), 256 threads =
// 32 rows x 8 outs. K staged in 256-element chunks: x rows via sorted_row
// (PD_MOE_PAD -> zeros), both weight tiles for the block's expert. Output is
// written SORTED-contiguous (fused[(blk*BM + row)*ff + o]) - the down kernel
// reads that layout directly; PAD rows write zeros so the downstream
// quantize never sees uninitialized bytes.
__global__ void __launch_bounds__(256) pd_q8_0_moe_gate_up_sorted_kernel(
    const int8_t* __restrict__ gate_data, const __half* __restrict__ gate_scale,
    const int8_t* __restrict__ up_data, const __half* __restrict__ up_scale,
    const unsigned int* __restrict__ sorted_row, const unsigned int* __restrict__ block_expert,
    const int8_t* __restrict__ xq, const float* __restrict__ xs,
    float* __restrict__ fused, uint32_t in_dim, uint32_t ff) {
    const uint32_t blk = blockIdx.y;
    const uint32_t e = block_expert[blk];
    if (e == PD_MOE_PAD) return;
    const uint32_t o0 = blockIdx.x * PD_QMOE_BN;
    const uint32_t tid = threadIdx.x;
    const uint32_t row = tid >> 3, n = tid & 7u; // this thread's (row, out) pair
    const uint32_t n_blocks = in_dim >> 5;

    __shared__ int sx[PD_QMOE_BM * PD_QMOE_XW];
    __shared__ float sxs[PD_QMOE_BM][PD_QMOE_BK / 32u];
    __shared__ int swg[PD_QMOE_BN * PD_QMOE_WW];
    __shared__ int swu[PD_QMOE_BN * PD_QMOE_WW];
    __shared__ float swsg[PD_QMOE_BN][PD_QMOE_BK / 32u];
    __shared__ float swsu[PD_QMOE_BN][PD_QMOE_BK / 32u];

    const unsigned int srow = sorted_row[blk * PD_QMOE_BM + row];
    float accg[2] = {0.0f, 0.0f}, accu[2] = {0.0f, 0.0f};

    for (uint32_t k0 = 0; k0 < in_dim; k0 += PD_QMOE_BK) {
        // stage x: 32 rows x 64 int32 words (8 per thread), zeros for PAD rows
        {
            const uint32_t r = tid >> 3, w0 = (tid & 7u) * 8u;
            const unsigned int xr = sorted_row[blk * PD_QMOE_BM + r];
            const bool live_r = xr != PD_MOE_PAD;
            // clamp PAD rows to row 0 so the base pointer stays in-bounds;
            // their staged values are zeroed below
            const int* src =
                reinterpret_cast<const int*>(xq + (size_t)(live_r ? xr : 0u) * in_dim + k0);
#pragma unroll
            for (uint32_t i = 0; i < 8u; ++i)
                sx[r * PD_QMOE_XW + w0 + i] = live_r ? src[w0 + i] : 0;
            if ((tid & 7u) == 0) {
#pragma unroll
                for (uint32_t b = 0; b < PD_QMOE_BK / 32u; ++b)
                    sxs[r][b] = live_r
                        ? xs[(size_t)xr * n_blocks + (k0 >> 5) + b]
                        : 0.0f;
            }
        }
        // stage both weight tiles: 8 outs x 64 words each (2 outs per warp)
        for (uint32_t i = tid; i < PD_QMOE_BN * 64u; i += 256u) {
            const uint32_t on = i >> 6, w = i & 63u;
            const uint32_t o = o0 + on;
            const size_t wrow = ((size_t)e * ff + (o < ff ? o : ff - 1u)) * in_dim + k0;
            swg[on * PD_QMOE_WW + w] = reinterpret_cast<const int*>(gate_data + wrow)[w];
            swu[on * PD_QMOE_WW + w] = reinterpret_cast<const int*>(up_data + wrow)[w];
        }
        if (tid < PD_QMOE_BN * (PD_QMOE_BK / 32u)) {
            const uint32_t on = tid / (PD_QMOE_BK / 32u), b = tid % (PD_QMOE_BK / 32u);
            const uint32_t o = o0 + on;
            const size_t srow_w = ((size_t)e * ff + (o < ff ? o : ff - 1u)) * n_blocks + (k0 >> 5) + b;
            swsg[on][b] = __half2float(gate_scale[srow_w]);
            swsu[on][b] = __half2float(up_scale[srow_w]);
        }
        __syncthreads();
        // 8 q8-blocks per chunk, 8 dp4a words per block, 2 outs per thread
#pragma unroll
        for (uint32_t b = 0; b < PD_QMOE_BK / 32u; ++b) {
            int ig0 = 0, iu0 = 0, ig1 = 0, iu1 = 0;
#pragma unroll
            for (uint32_t i = 0; i < 8u; ++i) {
                const int xv = sx[row * PD_QMOE_XW + b * 8u + i];
                ig0 = __dp4a(swg[n * PD_QMOE_WW + b * 8u + i], xv, ig0);
                iu0 = __dp4a(swu[n * PD_QMOE_WW + b * 8u + i], xv, iu0);
                ig1 = __dp4a(swg[(n + 8u) * PD_QMOE_WW + b * 8u + i], xv, ig1);
                iu1 = __dp4a(swu[(n + 8u) * PD_QMOE_WW + b * 8u + i], xv, iu1);
            }
            const float xsb = sxs[row][b];
            accg[0] += swsg[n][b] * xsb * (float)ig0;
            accu[0] += swsu[n][b] * xsb * (float)iu0;
            accg[1] += swsg[n + 8u][b] * xsb * (float)ig1;
            accu[1] += swsu[n + 8u][b] * xsb * (float)iu1;
        }
        __syncthreads();
    }
#pragma unroll
    for (uint32_t h = 0; h < 2u; ++h) {
        const uint32_t o = o0 + n + h * 8u;
        if (o < ff) {
            const float g = accg[h], u = accu[h];
            fused[((size_t)blk * PD_QMOE_BM + row) * ff + o] =
                (srow != PD_MOE_PAD) ? (g / (1.0f + __expf(-g))) * u : 0.0f;
        }
    }
}

PD_EXPORT
int pd_q8_0_moe_gate_up_sorted(const void* gate_data, const void* gate_scale,
                               const void* up_data, const void* up_scale,
                               const void* sorted_row, const void* block_expert,
                               const void* xq, const void* xs, void* fused,
                               uint32_t in_dim, uint32_t ff, uint32_t max_blocks,
                               void* stream) {
    if (ff == 0 || max_blocks == 0) return 0;
    if ((in_dim & 255u) != 0) return cudaErrorInvalidValue; // BK-chunked staging
    dim3 grid((ff + PD_QMOE_BN - 1u) / PD_QMOE_BN, max_blocks);
    pd_q8_0_moe_gate_up_sorted_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(
        (const int8_t*)gate_data, (const __half*)gate_scale, (const int8_t*)up_data,
        (const __half*)up_scale, (const unsigned int*)sorted_row,
        (const unsigned int*)block_expert, (const int8_t*)xq, (const float*)xs,
        (float*)fused, in_dim, ff);
    return pd_launch_status();
}

// Sorted single-plane up + squared-relu (nemotron_h_moe: no gate matrix).
// Same tile shape/staging as the gate_up kernel above with one weight
// stream, plus a K-tail guard: nemotron's dims are 32-aligned but not
// 256-aligned (hidden 2688, moe_ff 1856, shared_ff 3712), so the last BK
// chunk stages partially - out-of-range words and scales stage as zero
// (dp4a over zeros adds nothing, and the zero scale keeps 0*x finite),
// which leaves fully-aligned shapes bit-identical to the unguarded walk.
__global__ void __launch_bounds__(256) pd_q8_0_moe_up_relu2_sorted_kernel(
    const int8_t* __restrict__ up_data, const __half* __restrict__ up_scale,
    const unsigned int* __restrict__ sorted_row, const unsigned int* __restrict__ block_expert,
    const int8_t* __restrict__ xq, const float* __restrict__ xs,
    float* __restrict__ fused, uint32_t in_dim, uint32_t ff) {
    const uint32_t blk = blockIdx.y;
    const uint32_t e = block_expert[blk];
    if (e == PD_MOE_PAD) return;
    const uint32_t o0 = blockIdx.x * PD_QMOE_BN;
    const uint32_t tid = threadIdx.x;
    const uint32_t row = tid >> 3, n = tid & 7u;
    const uint32_t n_blocks = in_dim >> 5;
    const uint32_t n_words = in_dim >> 2; // int32 words per row

    __shared__ int sx[PD_QMOE_BM * PD_QMOE_XW];
    __shared__ float sxs[PD_QMOE_BM][PD_QMOE_BK / 32u];
    __shared__ int swu[PD_QMOE_BN * PD_QMOE_WW];
    __shared__ float swsu[PD_QMOE_BN][PD_QMOE_BK / 32u];

    const unsigned int srow = sorted_row[blk * PD_QMOE_BM + row];
    float accu[2] = {0.0f, 0.0f};

    for (uint32_t k0 = 0; k0 < in_dim; k0 += PD_QMOE_BK) {
        const uint32_t w_base = k0 >> 2, b_base = k0 >> 5;
        {
            const uint32_t r = tid >> 3, w0 = (tid & 7u) * 8u;
            const unsigned int xr = sorted_row[blk * PD_QMOE_BM + r];
            const bool live_r = xr != PD_MOE_PAD;
            const int* src = reinterpret_cast<const int*>(xq + (size_t)(live_r ? xr : 0u) * in_dim);
#pragma unroll
            for (uint32_t i = 0; i < 8u; ++i) {
                const uint32_t w = w0 + i;
                sx[r * PD_QMOE_XW + w] = (live_r && w_base + w < n_words) ? src[w_base + w] : 0;
            }
            if ((tid & 7u) == 0) {
#pragma unroll
                for (uint32_t b = 0; b < PD_QMOE_BK / 32u; ++b)
                    sxs[r][b] = (live_r && b_base + b < n_blocks)
                        ? xs[(size_t)xr * n_blocks + b_base + b]
                        : 0.0f;
            }
        }
        for (uint32_t i = tid; i < PD_QMOE_BN * 64u; i += 256u) {
            const uint32_t on = i >> 6, w = i & 63u;
            const uint32_t o = o0 + on;
            const int* src = reinterpret_cast<const int*>(
                up_data + ((size_t)e * ff + (o < ff ? o : ff - 1u)) * in_dim);
            swu[on * PD_QMOE_WW + w] = (w_base + w < n_words) ? src[w_base + w] : 0;
        }
        if (tid < PD_QMOE_BN * (PD_QMOE_BK / 32u)) {
            const uint32_t on = tid / (PD_QMOE_BK / 32u), b = tid % (PD_QMOE_BK / 32u);
            const uint32_t o = o0 + on;
            swsu[on][b] = (b_base + b < n_blocks)
                ? __half2float(up_scale[((size_t)e * ff + (o < ff ? o : ff - 1u)) * n_blocks + b_base + b])
                : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (uint32_t b = 0; b < PD_QMOE_BK / 32u; ++b) {
            int iu0 = 0, iu1 = 0;
#pragma unroll
            for (uint32_t i = 0; i < 8u; ++i) {
                const int xv = sx[row * PD_QMOE_XW + b * 8u + i];
                iu0 = __dp4a(swu[n * PD_QMOE_WW + b * 8u + i], xv, iu0);
                iu1 = __dp4a(swu[(n + 8u) * PD_QMOE_WW + b * 8u + i], xv, iu1);
            }
            const float xsb = sxs[row][b];
            accu[0] += swsu[n][b] * xsb * (float)iu0;
            accu[1] += swsu[n + 8u][b] * xsb * (float)iu1;
        }
        __syncthreads();
    }
#pragma unroll
    for (uint32_t h = 0; h < 2u; ++h) {
        const uint32_t o = o0 + n + h * 8u;
        if (o < ff) {
            const float v = fmaxf(accu[h], 0.0f);
            fused[((size_t)blk * PD_QMOE_BM + row) * ff + o] =
                (srow != PD_MOE_PAD) ? v * v : 0.0f;
        }
    }
}

PD_EXPORT
int pd_q8_0_moe_up_relu2_sorted(const void* up_data, const void* up_scale,
                                const void* sorted_row, const void* block_expert,
                                const void* xq, const void* xs, void* fused,
                                uint32_t in_dim, uint32_t ff, uint32_t max_blocks,
                                void* stream) {
    if (ff == 0 || max_blocks == 0) return 0;
    if ((in_dim & 31u) != 0) return cudaErrorInvalidValue; // q8 block granularity
    dim3 grid((ff + PD_QMOE_BN - 1u) / PD_QMOE_BN, max_blocks);
    pd_q8_0_moe_up_relu2_sorted_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(
        (const int8_t*)up_data, (const __half*)up_scale, (const unsigned int*)sorted_row,
        (const unsigned int*)block_expert, (const int8_t*)xq, (const float*)xs,
        (float*)fused, in_dim, ff);
    return pd_launch_status();
}

// Sorted down + per-(token, slot) weighted partials: same tile shape, one
// matrix, K = ff. Reads the gate_up kernel's sorted-contiguous quantized
// output; writes part[(token*n_active + slot)*embd + o] = topk_w * dot for
// pd_moe_slot_combine to fold (deterministic slot order there).
// K-tail-guarded like the relu2 kernel above: nemotron's
// moe_ff 1856 / shared_ff 3712 are 32- but not 256-aligned; zero-staged
// tails leave the previously-supported 256-aligned shapes bit-identical.
__global__ void __launch_bounds__(256) pd_q8_0_moe_down_sorted_kernel(
    const int8_t* __restrict__ down_data, const __half* __restrict__ down_scale,
    const unsigned int* __restrict__ sorted_row, const unsigned int* __restrict__ sorted_slot,
    const unsigned int* __restrict__ block_expert, const float* __restrict__ topk_w,
    const int8_t* __restrict__ fq, const float* __restrict__ fs,
    float* __restrict__ part, uint32_t ff, uint32_t embd, uint32_t n_active) {
    const uint32_t blk = blockIdx.y;
    const uint32_t e = block_expert[blk];
    if (e == PD_MOE_PAD) return;
    const uint32_t o0 = blockIdx.x * PD_QMOE_BN;
    const uint32_t tid = threadIdx.x;
    const uint32_t row = tid >> 3, n = tid & 7u;
    const uint32_t n_blocks = ff >> 5;

    __shared__ int sx[PD_QMOE_BM * PD_QMOE_XW];
    __shared__ float sxs[PD_QMOE_BM][PD_QMOE_BK / 32u];
    __shared__ int sw[PD_QMOE_BN * PD_QMOE_WW];
    __shared__ float sws[PD_QMOE_BN][PD_QMOE_BK / 32u];

    const unsigned int srow = sorted_row[blk * PD_QMOE_BM + row];
    float acc[2] = {0.0f, 0.0f};
    const uint32_t n_words = ff >> 2; // int32 words per fused row

    for (uint32_t k0 = 0; k0 < ff; k0 += PD_QMOE_BK) {
        const uint32_t w_base = k0 >> 2, b_base = k0 >> 5;
        {
            // activations here are the SORTED fused rows: index blk*BM + r
            const uint32_t r = tid >> 3, w0 = (tid & 7u) * 8u;
            const size_t frow = (size_t)blk * PD_QMOE_BM + r;
            const int* src = reinterpret_cast<const int*>(fq + frow * ff);
#pragma unroll
            for (uint32_t i = 0; i < 8u; ++i) {
                const uint32_t w = w0 + i;
                sx[r * PD_QMOE_XW + w] = (w_base + w < n_words) ? src[w_base + w] : 0;
            }
            if ((tid & 7u) == 0) {
#pragma unroll
                for (uint32_t b = 0; b < PD_QMOE_BK / 32u; ++b)
                    sxs[r][b] = (b_base + b < n_blocks) ? fs[frow * n_blocks + b_base + b] : 0.0f;
            }
        }
        for (uint32_t i = tid; i < PD_QMOE_BN * 64u; i += 256u) {
            const uint32_t on = i >> 6, w = i & 63u;
            const uint32_t o = o0 + on;
            const int* src = reinterpret_cast<const int*>(
                down_data + ((size_t)e * embd + (o < embd ? o : embd - 1u)) * ff);
            sw[on * PD_QMOE_WW + w] = (w_base + w < n_words) ? src[w_base + w] : 0;
        }
        if (tid < PD_QMOE_BN * (PD_QMOE_BK / 32u)) {
            const uint32_t on = tid / (PD_QMOE_BK / 32u), b = tid % (PD_QMOE_BK / 32u);
            const uint32_t o = o0 + on;
            sws[on][b] = (b_base + b < n_blocks)
                ? __half2float(
                      down_scale[((size_t)e * embd + (o < embd ? o : embd - 1u)) * n_blocks + b_base + b])
                : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (uint32_t b = 0; b < PD_QMOE_BK / 32u; ++b) {
            int id0 = 0, id1 = 0;
#pragma unroll
            for (uint32_t i = 0; i < 8u; ++i) {
                const int xv = sx[row * PD_QMOE_XW + b * 8u + i];
                id0 = __dp4a(sw[n * PD_QMOE_WW + b * 8u + i], xv, id0);
                id1 = __dp4a(sw[(n + 8u) * PD_QMOE_WW + b * 8u + i], xv, id1);
            }
            const float xsb = sxs[row][b];
            acc[0] += sws[n][b] * xsb * (float)id0;
            acc[1] += sws[n + 8u][b] * xsb * (float)id1;
        }
        __syncthreads();
    }
    if (srow != PD_MOE_PAD) {
        const uint32_t slot = sorted_slot[blk * PD_QMOE_BM + row];
        const size_t pair = (size_t)srow * n_active + slot;
#pragma unroll
        for (uint32_t h = 0; h < 2u; ++h) {
            const uint32_t o = o0 + n + h * 8u;
            if (o < embd) part[pair * embd + o] = topk_w[pair] * acc[h];
        }
    }
}

PD_EXPORT
int pd_q8_0_moe_down_sorted(const void* down_data, const void* down_scale,
                            const void* sorted_row, const void* sorted_slot,
                            const void* block_expert, const void* topk_w, const void* fq,
                            const void* fs, void* part, uint32_t ff, uint32_t embd,
                            uint32_t n_active, uint32_t max_blocks, void* stream) {
    if (embd == 0 || max_blocks == 0) return 0;
    if ((ff & 31u) != 0) return cudaErrorInvalidValue; // q8 block granularity (K tail staged as zeros)
    dim3 grid((embd + PD_QMOE_BN - 1u) / PD_QMOE_BN, max_blocks);
    pd_q8_0_moe_down_sorted_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(
        (const int8_t*)down_data, (const __half*)down_scale, (const unsigned int*)sorted_row,
        (const unsigned int*)sorted_slot, (const unsigned int*)block_expert,
        (const float*)topk_w, (const int8_t*)fq, (const float*)fs, (float*)part, ff, embd,
        n_active);
    return pd_launch_status();
}

// ---- int8-MMA sorted Q8_0 MoE (tensor-core prefill/serving class) ----------
// The mxfp4 moe mmq structure applied to Q8_0 experts: same sorted layout,
// same m16n8k32 s8 MMA with per-k32-block scale folding, same cp.async
// double-buffered weight walk. Q8 weight blocks are 32 B (vs mxfp4's packed
// 16 B), so the 128-row strip's double buffer would not fit 2 blocks/SM --
// this kernel takes 64-row output strips (2 x 18 KB w-buffers + the 9.5 KB
// y-tile = 46.5 KB, the same budget as the mxfp4 kernel) with the warp map
// i0 = (warp>>2)*32 rows, joff = (warp&3)*8 token-columns (no j-loop).
// Weight int8 data arrives UNPACKED (8 words per k32 block; word w = elems
// 4w..4w+3), so A-fragment k-halves are direct word loads: the proven mxfp4
// lane mapping (k-half0 = elems {4t..4t+3}, k-half1 = {16+4t..4t+3+16},
// the same halves the B fragment uses) becomes words t and 4+t.
#define PD_QMMA_WK 76u   // 64 data int32 + 4 int32 of f16 scales + 8 pad (76%32=12 kills the g-lane bank conflicts, same trick as PD_MMQ_XK)
#define PD_QMMA_ROWS 64u
#define PD_QMMA_W_INT32 (PD_QMMA_ROWS * PD_QMMA_WK)
#define PD_QMMA_SMEM ((2u * PD_QMMA_W_INT32 + PD_MOEQ_Y_INT32) * 4u)

// Issue one chunk's q8 weight data (64 rows x 8 blocks x 32 B, two 16-byte
// cp.asyncs per (row, block)).
__device__ __forceinline__ void pd_qmma_issue_w(
    int* __restrict__ tile, const int8_t* __restrict__ data, size_t wrow0,
    uint32_t row_base, uint32_t out_dim, uint32_t in_dim, uint32_t kt, uint32_t tid) {
#if PD_MMA_OK
    const uint32_t n_blocks = in_dim >> 5;
    #pragma unroll
    for (uint32_t it = 0; it < 4u; ++it) {
        const uint32_t i = it * 256u + tid;
        const uint32_t row = i >> 4, half = i & 15u;
        const uint32_t b = half >> 1, h16 = half & 1u, gb = kt * 8u + b;
        const bool ok = gb < n_blocks && (row_base + row) < out_dim;
        pd_cp_async16(tile + row * PD_QMMA_WK + b * 8u + h16 * 4u,
                      data + ((wrow0 + row) * (size_t)in_dim) + (ok ? gb : 0u) * 32u + h16 * 16u,
                      ok);
    }
#endif
}

// Stage one chunk's f16 weight scales (64 rows x 8 per chunk, 2-byte loads).
__device__ __forceinline__ void pd_qmma_stage_ws(
    int* __restrict__ tile, const __half* __restrict__ scale, size_t wrow0,
    uint32_t row_base, uint32_t out_dim, uint32_t n_blocks, uint32_t kt, uint32_t tid) {
#if PD_MMA_OK
    #pragma unroll
    for (uint32_t it = 0; it < 2u; ++it) {
        const uint32_t i = it * 256u + tid;
        const uint32_t row = i >> 3, b = i & 7u, gb = kt * 8u + b;
        ((__half*)(tile + row * PD_QMMA_WK + 64u))[b] =
            (gb < n_blocks && (row_base + row) < out_dim)
                ? scale[(wrow0 + row) * n_blocks + gb]
                : __float2half(0.0f);
    }
#endif
}

// BM = tokens per sorted block; DB = double-buffered weights. (32,true) is the
// serving/decode default. (64,false) is the prefill variant: a wider token block
// halves the weight-DRAM re-reads, but the doubled activation tile would push
// smem past the ~50 KB 2-CTA/SM cliff -- so it drops the weight double-buffer
// (single buffer, no K-prefetch overlap) to STAY at 2 CTA/SM. Measured ~1.15x on
// gate_up at pf-2048 vs (32,true). See.
// GELU=false is the qwen SwiGLU original; GELU=true is the gemma4-A4B twin
// (gelu_tanh(gate)*up in the same in-register quantize epilogue).
template <uint32_t BM, bool DB, bool GELU = false>
__global__ void __launch_bounds__(256, 2) pd_q8_0_moe_gate_up_mma_kernel(
    const int8_t* __restrict__ gate_data, const __half* __restrict__ gate_scale,
    const int8_t* __restrict__ up_data, const __half* __restrict__ up_scale,
    const unsigned int* __restrict__ sorted_row, const unsigned int* __restrict__ block_expert,
    const int8_t* __restrict__ xq, const float* __restrict__ xs,
    int8_t* __restrict__ fq, float* __restrict__ fs, uint32_t in_dim, uint32_t ff) {
#if PD_MMA_OK
    const uint32_t blk = blockIdx.x;                 // token block (fast axis: L2 strip reuse)
    const uint32_t e = block_expert[blk];
    if (e == PD_MOE_PAD) return;
    const uint32_t row_base = blockIdx.y * PD_QMMA_ROWS;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, t = lane & 3u;
    const uint32_t i0 = (warp >> 2) * 32u;           // 32-row group (0 or 32)
    const uint32_t joff = (warp & 3u) * 8u;          // 8-token column quarter
    const uint32_t n_blocks = in_dim >> 5;
    const uint32_t nk = (in_dim + 255u) >> 8;

    constexpr uint32_t NH = BM / 32u;                // 32-token halves per block
    extern __shared__ int pd_qmma_sh[];
    int* tile_y = pd_qmma_sh;
    int* wbuf0 = pd_qmma_sh + BM * PD_MMQ_XK;
    int* wbuf1 = DB ? (wbuf0 + PD_QMMA_W_INT32) : wbuf0;   // SB aliases wbuf0
    __shared__ unsigned int tok[BM];
    for (uint32_t i = tid; i < BM; i += 256u) tok[i] = sorted_row[(size_t)blk * BM + i];
    __syncthreads();

    float acc_g[NH][2][4] = {}, acc_u[NH][2][4] = {};
    const size_t wrow0 = (size_t)e * ff + row_base;
    #pragma unroll
    for (uint32_t mat = 0; mat < 2u; ++mat) {
        const int8_t* wd = mat ? up_data : gate_data;
        const __half* ws = mat ? up_scale : gate_scale;
        pd_qmma_issue_w(wbuf0, wd, wrow0, row_base, ff, in_dim, 0, tid);
        asm volatile("cp.async.commit_group;");
        for (uint32_t kt = 0; kt < nk; ++kt) {
            int* tw = (DB && (kt & 1u)) ? wbuf1 : wbuf0;
            asm volatile("cp.async.wait_group 0;");
            pd_qmma_stage_ws(tw, ws, wrow0, row_base, ff, n_blocks, kt, tid);
            pd_moeq_stage_y<BM>(tile_y, (const int*)xq, xs, tok, in_dim, kt, tid);
            __syncthreads();
            // DB: prefetch kt+1 into the other buffer now (overlaps this kt's mma).
            // SB: the single buffer is still being read below -- defer to after.
            if (DB && kt + 1u < nk) {
                pd_qmma_issue_w((kt & 1u) ? wbuf0 : wbuf1, wd, wrow0, row_base, ff,
                                in_dim, kt + 1u, tid);
                asm volatile("cp.async.commit_group;");
            }

            #pragma unroll
            for (uint32_t th = 0; th < NH; ++th) {
                const uint32_t jb = th * 32u + joff;       // this half's token base
                float (*acc)[4] = mat ? acc_u[th] : acc_g[th];
                #pragma unroll
                for (uint32_t h = 0; h < 2u; ++h) {
                    const uint32_t k00 = h * 32u;
                    #pragma unroll
                    for (uint32_t kk = 0; kk < 4u; ++kk) {
                        const uint32_t bb = (k00 >> 3) + kk;
                        const uint32_t ko = k00 + kk * 8u;
                        const int b0 = tile_y[(jb + g) * PD_MMQ_XK + ko + t];
                        const int b1 = tile_y[(jb + g) * PD_MMQ_XK + ko + 4u + t];
                        const float dB0 =
                            ((const float*)tile_y)[(jb + 2u * t) * PD_MMQ_XK + 64u + bb];
                        const float dB1 =
                            ((const float*)tile_y)[(jb + 2u * t + 1u) * PD_MMQ_XK + 64u + bb];
                        #pragma unroll
                        for (uint32_t n = 0; n < 2u; ++n) {
                            const uint32_t r0 = (i0 + n * 16u + g) * PD_QMMA_WK;
                            const uint32_t r8 = (i0 + n * 16u + 8u + g) * PD_QMMA_WK;
                            // PTX m16n8k32.s8 A layout: k-half0 = elems {4t..4t+3}
                            // (word t of the 8-word block), k-half1 = {16+4t..}
                            // (word 4+t) - the same halves the B fragment uses.
                            const int A0 = tw[r0 + bb * 8u + t];
                            const int A2 = tw[r0 + bb * 8u + 4u + t];
                            const int A1 = tw[r8 + bb * 8u + t];
                            const int A3 = tw[r8 + bb * 8u + 4u + t];
                            const float dA0 = __half2float(((const __half*)(tw + r0 + 64u))[bb]);
                            const float dA1 = __half2float(((const __half*)(tw + r8 + 64u))[bb]);
                            int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                            asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                                : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                                : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(b0), "r"(b1));
                            acc[n][0] += dA0 * dB0 * (float)d0;
                            acc[n][1] += dA0 * dB1 * (float)d1;
                            acc[n][2] += dA1 * dB0 * (float)d2;
                            acc[n][3] += dA1 * dB1 * (float)d3;
                        }
                    }
                }
            }
            __syncthreads();  // tile_y + the buffers are rewritten next kt
            // SB: buffer now free to reload; issue kt+1 (no compute overlap).
            if (!DB && kt + 1u < nk) {
                pd_qmma_issue_w(wbuf0, wd, wrow0, row_base, ff, in_dim, kt + 1u, tid);
                asm volatile("cp.async.commit_group;");
            }
        }
    }

    // plain silu(g)*u (qwen: no bias, no clamp), quantized per-32 output block
    // in REGISTERS -- fq/fs are the down GEMM's direct input, the f32
    // activation never lands in memory. Same in-register amax/shfl scheme as
    // the mxfp4 epilogue; PAD rows write exact zeros (data and scale).
    const uint32_t n_sb = ff >> 5;
    #pragma unroll
    for (uint32_t th = 0; th < NH; ++th) {
        const uint32_t jb = th * 32u + joff;
        #pragma unroll
        for (uint32_t qc = 0; qc < 2u; ++qc) {
            const uint32_t c = jb + 2u * t + qc;
            const bool pad = tok[c] == PD_MOE_PAD;
            const uint32_t rb = row_base + i0;
            float sw[4];
            #pragma unroll
            for (uint32_t n = 0; n < 2u; ++n) {
                #pragma unroll
                for (uint32_t hq = 0; hq < 2u; ++hq) {
                    const uint32_t q = qc + 2u * hq;
                    const uint32_t r = rb + n * 16u + hq * 8u + g;
                    float out = 0.f;
                    if (!pad && r < ff) {
                        const float gv = acc_g[th][n][q];
                        const float uv = acc_u[th][n][q];
                        out = GELU
                            ? 0.5f * gv
                                  * (1.0f
                                     + tanhf(0.79788456080286535587989211986876f * gv
                                             * (1.0f + 0.044715f * gv * gv)))
                                  * uv
                            : (gv / (1.0f + __expf(-gv))) * uv;
                    }
                    sw[n * 2u + hq] = out;
                }
            }
            float a = fmaxf(fmaxf(fabsf(sw[0]), fabsf(sw[1])), fmaxf(fabsf(sw[2]), fabsf(sw[3])));
            #pragma unroll
            for (uint32_t o = 4; o <= 16u; o <<= 1)
                a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
            const float scl = a * (1.0f / 127.0f);
            const float invs = scl > 0.f ? 1.0f / scl : 0.f;
            const size_t row = (size_t)blk * BM + c;
            if (rb < ff) {
                #pragma unroll
                for (uint32_t v = 0; v < 4u; ++v) {
                    const uint32_t r = rb + (v >> 1) * 16u + (v & 1u) * 8u + g;
                    int qi = __float2int_rn(sw[v] * invs);
                    qi = qi < -127 ? -127 : (qi > 127 ? 127 : qi);
                    fq[row * ff + r] = (int8_t)qi;
                }
                if (g == 0) fs[row * n_sb + (rb >> 5)] = scl;
            }
        }
    }
#else
    (void)gate_data; (void)gate_scale; (void)up_data; (void)up_scale; (void)sorted_row;
    (void)block_expert; (void)xq; (void)xs; (void)fq; (void)fs; (void)in_dim; (void)ff;
#endif
}

// Down half: same tile shape over K = ff; activation rows are the sorted
// fused rows (indexed directly by blk*32 + c, no gather). DETERMINISTIC
// epilogue: one writer per (token, slot, r), plain stores into the partials
// buffer, pd_moe_slot_combine folds in fixed slot order (the mxfp4 down_mmq
// atomic-scatter lesson).
template <uint32_t BM, bool DB>
__global__ void __launch_bounds__(256, 2) pd_q8_0_moe_down_mma_kernel(
    const int8_t* __restrict__ down_data, const __half* __restrict__ down_scale,
    const unsigned int* __restrict__ sorted_row, const unsigned int* __restrict__ sorted_slot,
    const unsigned int* __restrict__ block_expert, const float* __restrict__ topk_w,
    const int8_t* __restrict__ fq, const float* __restrict__ fs,
    float* __restrict__ part, uint32_t ff, uint32_t embd, uint32_t n_active) {
#if PD_MMA_OK
    const uint32_t blk = blockIdx.x;
    const uint32_t e = block_expert[blk];
    if (e == PD_MOE_PAD) return;
    const uint32_t row_base = blockIdx.y * PD_QMMA_ROWS;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, t = lane & 3u;
    const uint32_t i0 = (warp >> 2) * 32u;
    const uint32_t joff = (warp & 3u) * 8u;
    const uint32_t n_blocks = ff >> 5;
    const uint32_t nk = (ff + 255u) >> 8;

    constexpr uint32_t NH = BM / 32u;
    extern __shared__ int pd_qmma_sh[];
    int* tile_y = pd_qmma_sh;
    int* wbuf0 = pd_qmma_sh + BM * PD_MMQ_XK;
    int* wbuf1 = DB ? (wbuf0 + PD_QMMA_W_INT32) : wbuf0;
    __shared__ unsigned int tok[BM], slt[BM], idn[BM];
    for (uint32_t i = tid; i < BM; i += 256u) {
        tok[i] = sorted_row[(size_t)blk * BM + i];
        slt[i] = sorted_slot[(size_t)blk * BM + i];
        idn[i] = blk * BM + i;  // fq rows are sorted-contiguous
    }
    __syncthreads();

    float acc[NH][2][4] = {};
    const size_t wrow0 = (size_t)e * embd + row_base;
    pd_qmma_issue_w(wbuf0, down_data, wrow0, row_base, embd, ff, 0, tid);
    asm volatile("cp.async.commit_group;");
    for (uint32_t kt = 0; kt < nk; ++kt) {
        int* tw = (DB && (kt & 1u)) ? wbuf1 : wbuf0;
        asm volatile("cp.async.wait_group 0;");
        pd_qmma_stage_ws(tw, down_scale, wrow0, row_base, embd, n_blocks, kt, tid);
        pd_moeq_stage_y<BM>(tile_y, (const int*)fq, fs, idn, ff, kt, tid);
        __syncthreads();
        if (DB && kt + 1u < nk) {
            pd_qmma_issue_w((kt & 1u) ? wbuf0 : wbuf1, down_data, wrow0, row_base, embd,
                            ff, kt + 1u, tid);
            asm volatile("cp.async.commit_group;");
        }
        #pragma unroll
        for (uint32_t th = 0; th < NH; ++th) {
            const uint32_t jb = th * 32u + joff;
            #pragma unroll
            for (uint32_t h = 0; h < 2u; ++h) {
                const uint32_t k00 = h * 32u;
                #pragma unroll
                for (uint32_t kk = 0; kk < 4u; ++kk) {
                    const uint32_t bb = (k00 >> 3) + kk;
                    const uint32_t ko = k00 + kk * 8u;
                    const int b0 = tile_y[(jb + g) * PD_MMQ_XK + ko + t];
                    const int b1 = tile_y[(jb + g) * PD_MMQ_XK + ko + 4u + t];
                    const float dB0 =
                        ((const float*)tile_y)[(jb + 2u * t) * PD_MMQ_XK + 64u + bb];
                    const float dB1 =
                        ((const float*)tile_y)[(jb + 2u * t + 1u) * PD_MMQ_XK + 64u + bb];
                    #pragma unroll
                    for (uint32_t n = 0; n < 2u; ++n) {
                        const uint32_t r0 = (i0 + n * 16u + g) * PD_QMMA_WK;
                        const uint32_t r8 = (i0 + n * 16u + 8u + g) * PD_QMMA_WK;
                        // PTX A layout: k-halves at words t and 4+t (see gate_up)
                        const int A0 = tw[r0 + bb * 8u + t];
                        const int A2 = tw[r0 + bb * 8u + 4u + t];
                        const int A1 = tw[r8 + bb * 8u + t];
                        const int A3 = tw[r8 + bb * 8u + 4u + t];
                        const float dA0 = __half2float(((const __half*)(tw + r0 + 64u))[bb]);
                        const float dA1 = __half2float(((const __half*)(tw + r8 + 64u))[bb]);
                        int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                        asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                            : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                            : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(b0), "r"(b1));
                        acc[th][n][0] += dA0 * dB0 * (float)d0;
                        acc[th][n][1] += dA0 * dB1 * (float)d1;
                        acc[th][n][2] += dA1 * dB0 * (float)d2;
                        acc[th][n][3] += dA1 * dB1 * (float)d3;
                    }
                }
            }
        }
        __syncthreads();
        if (!DB && kt + 1u < nk) {
            pd_qmma_issue_w(wbuf0, down_data, wrow0, row_base, embd, ff, kt + 1u, tid);
            asm volatile("cp.async.commit_group;");
        }
    }

    #pragma unroll
    for (uint32_t th = 0; th < NH; ++th) {
        const uint32_t c0 = th * 32u + joff + 2u * t;
        #pragma unroll
        for (uint32_t n = 0; n < 2u; ++n) {
            const uint32_t r0 = row_base + i0 + n * 16u + g;
            const uint32_t r8 = r0 + 8u;
            #pragma unroll
            for (uint32_t q = 0; q < 4u; ++q) {
                const uint32_t r = (q & 2u) ? r8 : r0;
                const uint32_t c = c0 + (q & 1u);
                const unsigned int token = tok[c];
                if (r >= embd || token == PD_MOE_PAD) continue;
                const float w = topk_w[(size_t)token * n_active + slt[c]];
                part[((size_t)token * n_active + slt[c]) * embd + r] = w * acc[th][n][q];
            }
        }
    }
#else
    (void)down_data; (void)down_scale; (void)sorted_row; (void)sorted_slot;
    (void)block_expert; (void)topk_w; (void)fq; (void)fs; (void)part;
    (void)ff; (void)embd; (void)n_active;
#endif
}

// smem footprint for a (BM, DB) instantiation: activation tile + 1 or 2 weight
// buffers. (32,true)=48640 B, (64,false)=38912 B - both stay 2 CTA/SM.
template <uint32_t BM, bool DB>
static constexpr uint32_t pd_qmma_smem() {
    return (BM * PD_MMQ_XK + (DB ? 2u : 1u) * PD_QMMA_W_INT32) * 4u;
}

template <uint32_t BM, bool DB, bool GELU = false>
static int pd_launch_qmma_gu(const int8_t* gd, const __half* gs, const int8_t* ud,
                             const __half* us, const unsigned int* sr,
                             const unsigned int* be, const int8_t* xq, const float* xs,
                             int8_t* fq, float* fs, uint32_t in_dim, uint32_t ff,
                             uint32_t max_blocks, cudaStream_t stream) {
    constexpr uint32_t smem = pd_qmma_smem<BM, DB>();
    static bool attr = false;   // per-instantiation (template statics)
    if (!attr) {
        cudaFuncSetAttribute((const void*)pd_q8_0_moe_gate_up_mma_kernel<BM, DB, GELU>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        attr = true;
    }
    dim3 grid(max_blocks, (ff + PD_QMMA_ROWS - 1u) / PD_QMMA_ROWS);
    pd_q8_0_moe_gate_up_mma_kernel<BM, DB, GELU><<<grid, 256, smem, stream>>>(
        gd, gs, ud, us, sr, be, xq, xs, fq, fs, in_dim, ff);
    return pd_launch_status();
}

// bm selects the block tile: 64 -> wider prefill block (single-buffered weights),
// else the 32-token serving/decode default (double-buffered). The sorted layout
// (moe_align bm) and fq/fs sizing at the call site must match bm.
PD_EXPORT
int pd_q8_0_moe_gate_up_mma(const void* gate_data, const void* gate_scale,
                            const void* up_data, const void* up_scale,
                            const void* sorted_row, const void* block_expert,
                            const void* xq, const void* xs, void* fq, void* fs,
                            uint32_t in_dim, uint32_t ff, uint32_t max_blocks,
                            uint32_t bm, void* stream) {
    if (ff == 0 || max_blocks == 0) return 0;
    if ((in_dim & 255u) != 0 || (ff & 31u) != 0) return cudaErrorInvalidValue;
    const int8_t* gd = (const int8_t*)gate_data; const __half* gs = (const __half*)gate_scale;
    const int8_t* ud = (const int8_t*)up_data; const __half* us = (const __half*)up_scale;
    const unsigned int* sr = (const unsigned int*)sorted_row;
    const unsigned int* be = (const unsigned int*)block_expert;
    const int8_t* xqp = (const int8_t*)xq; const float* xsp = (const float*)xs;
    int8_t* fqp = (int8_t*)fq; float* fsp = (float*)fs;
    cudaStream_t st = (cudaStream_t)stream;
    if (bm >= 64u)
        return pd_launch_qmma_gu<64u, false>(gd, gs, ud, us, sr, be, xqp, xsp, fqp, fsp,
                                             in_dim, ff, max_blocks, st);
    return pd_launch_qmma_gu<32u, true>(gd, gs, ud, us, sr, be, xqp, xsp, fqp, fsp,
                                        in_dim, ff, max_blocks, st);
}

// GEGLU twin of the mma launcher above (gemma4-A4B routed experts): same
// sorted layout/fq handshake, gelu_tanh epilogue instantiations.
PD_EXPORT
int pd_q8_0_moe_gate_up_mma_geglu(const void* gate_data, const void* gate_scale,
                                  const void* up_data, const void* up_scale,
                                  const void* sorted_row, const void* block_expert,
                                  const void* xq, const void* xs, void* fq, void* fs,
                                  uint32_t in_dim, uint32_t ff, uint32_t max_blocks,
                                  uint32_t bm, void* stream) {
    if (ff == 0 || max_blocks == 0) return 0;
    if ((in_dim & 255u) != 0 || (ff & 31u) != 0) return cudaErrorInvalidValue;
    const int8_t* gd = (const int8_t*)gate_data; const __half* gs = (const __half*)gate_scale;
    const int8_t* ud = (const int8_t*)up_data; const __half* us = (const __half*)up_scale;
    const unsigned int* sr = (const unsigned int*)sorted_row;
    const unsigned int* be = (const unsigned int*)block_expert;
    const int8_t* xqp = (const int8_t*)xq; const float* xsp = (const float*)xs;
    int8_t* fqp = (int8_t*)fq; float* fsp = (float*)fs;
    cudaStream_t st = (cudaStream_t)stream;
    if (bm >= 64u)
        return pd_launch_qmma_gu<64u, false, true>(gd, gs, ud, us, sr, be, xqp, xsp, fqp,
                                                   fsp, in_dim, ff, max_blocks, st);
    return pd_launch_qmma_gu<32u, true, true>(gd, gs, ud, us, sr, be, xqp, xsp, fqp, fsp,
                                              in_dim, ff, max_blocks, st);
}

template <uint32_t BM, bool DB>
static int pd_launch_qmma_dn(const int8_t* dd, const __half* ds, const unsigned int* sr,
                             const unsigned int* sl, const unsigned int* be,
                             const float* tw, const int8_t* fq, const float* fs,
                             float* part, uint32_t ff, uint32_t embd, uint32_t n_active,
                             uint32_t max_blocks, cudaStream_t stream) {
    constexpr uint32_t smem = pd_qmma_smem<BM, DB>();
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute((const void*)pd_q8_0_moe_down_mma_kernel<BM, DB>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        attr = true;
    }
    dim3 grid(max_blocks, (embd + PD_QMMA_ROWS - 1u) / PD_QMMA_ROWS);
    pd_q8_0_moe_down_mma_kernel<BM, DB><<<grid, 256, smem, stream>>>(
        dd, ds, sr, sl, be, tw, fq, fs, part, ff, embd, n_active);
    return pd_launch_status();
}

PD_EXPORT
int pd_q8_0_moe_down_mma(const void* down_data, const void* down_scale,
                         const void* sorted_row, const void* sorted_slot,
                         const void* block_expert, const void* topk_w, const void* fq,
                         const void* fs, void* part, uint32_t ff, uint32_t embd,
                         uint32_t n_active, uint32_t max_blocks, uint32_t bm, void* stream) {
    if (embd == 0 || max_blocks == 0) return 0;
    // K = ff only needs Q8-block granularity: the K walk is fully guarded
    // (issue_w `gb < n_blocks` -> zfill cp.async, stage_y zero-fills data AND
    // scales past n_blocks; zero x anything accumulates exactly 0). The old
    // 256-multiple check was conservative - the gemma4-A4B's ff_exp=704 is
    // the first ragged-K consumer (validated vs the token-batched pair).
    if ((ff & 31u) != 0 || (embd & 31u) != 0) return cudaErrorInvalidValue;
    const int8_t* dd = (const int8_t*)down_data; const __half* ds = (const __half*)down_scale;
    const unsigned int* sr = (const unsigned int*)sorted_row;
    const unsigned int* sl = (const unsigned int*)sorted_slot;
    const unsigned int* be = (const unsigned int*)block_expert;
    const float* tw = (const float*)topk_w; const int8_t* fqp = (const int8_t*)fq;
    const float* fsp = (const float*)fs; float* pp = (float*)part;
    cudaStream_t st = (cudaStream_t)stream;
    if (bm >= 64u)
        return pd_launch_qmma_dn<64u, false>(dd, ds, sr, sl, be, tw, fqp, fsp, pp, ff, embd,
                                             n_active, max_blocks, st);
    return pd_launch_qmma_dn<32u, true>(dd, ds, sr, sl, be, tw, fqp, fsp, pp, ff, embd,
                                        n_active, max_blocks, st);
}

// ---- v2 ring twins  --------------------------
// Same math, same per-accumulator fold order, same fq/fs/part handshake as
// the (32,true) pair above -- bitwise on every live output. What changes is
// the data engine:
//   - S-stage cp.async ring over (W data + W scales + Y): the pair's
//     wait_group-0 ping-pong holds ~1 chunk in flight per CTA and measured
//     1.1-1.6 TB/s effective on the g26a4b c32 band (a4b_moe_kbench u64
//     r=32: gu 167.9us / dn 123.1us vs the ~34/17us uniq-64 weight floor).
//   - the gate->up mat boundary pipelines too: one flat 2*nk tick walk, no
//     ring drain between the two K passes (acc_g / acc_u are separate
//     accumulators, so each keeps the shipped fold order exactly).
//   - W scales and the Y tile ride the ring as cp.async -- the shipped pair
//     paid a synchronous gmem round trip for both, every 256-K tick.
//   - live-quarter skip: moe_align packs live pairs from col 0; at c32 real
//     routing is ~4 pairs per 32-block, so 3 of 4 warp-owned 8-col quarters
//     are all-PAD and skip mma+fold+epilogue entirely. MMA columns are
//     independent accumulators: a dead quarter's unwritten fq/fs feed only
//     dead-column accumulators in the down half, which stores nothing for
//     them (token == PAD).
// BM=32 only (the serving/decode shape; bm >= 64 callers keep the pair
// above). The 4B async scale copies need even n_blocks on the K side:
// in_dim % 256 covers gate_up (2816), down needs ff % 64 (704 ok) -- the
// launchers refuse otherwise and the engine keeps the shipped pair.
#ifndef PD_QMMA2_S
#define PD_QMMA2_S 2   // ring depth; swept 08-25: S=2/OCC=3 beat S=3/OCC=2 on both
                       // kernels (u64r32 gu 88.3->85.9, dn 55.5->45.1us) - occupancy
                       // buys more than prefetch depth here; sweep via PD_DEFS.
#endif
#ifndef PD_QMMA2_OCC
#define PD_QMMA2_OCC 3 // CTAs/SM the compiler budgets regs for (25%
#endif                 // theoretical occupancy at S=3/OCC=2 was the v2 cap)
#ifndef PD_QMMA2_RB
#define PD_QMMA2_RB 64u // W rows per (CTA, tick). 32 halves the stage for the
#endif                  // high-occupancy arm (PD_DEFS sweep; epilogue adapts)
#ifndef PD_QMMA2_LDM
#define PD_QMMA2_LDM 1 // ldmatrix fragment loads - bit-identical byte mapping
#endif                 // (int8_mma.cuh contract, kbench-verified BITWISE);
                       // swept 08-25: gu 86.0 -> 81.6us, dn wash. 0 = scalar.
#ifndef PD_QMMA2_YSYNC
#define PD_QMMA2_YSYNC 0 // 1 = Y staged synchronously per tick (L2-resident
#endif                   // activations) so ring stages carry W only: stage
                         // shrinks 29.2 -> 19.4KB and S=3 x OCC=3 fit together
#ifndef PD_QMMA2_OCC_GU
#define PD_QMMA2_OCC_GU 2 // ILV's ~107KB gu stage caps at 2 CTAs/SM
#endif
#ifndef PD_QMMA2_OCC_DN
#define PD_QMMA2_OCC_DN 3 // down keeps the 58.4KB stage and its 3rd CTA
#endif
#ifndef PD_QMMA2_ILV
#define PD_QMMA2_ILV 1 // One K-walk for gate+up (stage carries both W chunks,
#endif                 // Y staged once): 11 ticks instead of 22, half the
                       // barrier chain. Swept 08-26: gu 81.6 -> 72.9us (u64),
                       // 72.4 -> 58.7 (u48), verify gu -16%; dn parity;
                       // BITWISE. 0 = the flat two-pass walk (A/B).
#ifndef PD_QMMA2_WMAP
#define PD_QMMA2_WMAP 1 // live-shape warp map: token-quarter = warp>>1 so the
#endif                  // first live quarter lands on warps {0,1} = SMSPs {0,1}.
                        // DEFAULT on 08-26: bitwise (kbench all cells), kbench gu
                        // -3.7% u64:32 / -7.0% uni:128 / -12.3% uni:256; serve
                        // c64 +2.0% / c128 +1.4% (3 boots/arm alternating).
                        // 0 reverts. sm-agnostic relabel.
                        // (shipped map (warp&3) puts it on warps {0,4} = SMSP 0
                        // twice - the whole partial-liveness mma+fold chain on one
                        // scheduler). Pure tile-to-warp relabel: per-output k-fold
                        // order untouched => bitwise.
#ifndef PD_QMMA2_DN_S
#define PD_QMMA2_DN_S PD_QMMA2_S // down-only ring depth (K=704 = 3 ticks; sweep)
#endif
#ifndef PD_QMMA2_DN_NT
#define PD_QMMA2_DN_NT 1 // >1 = down NT twin: one CTA walks NT consecutive RB-row
#endif                   // slices as a flat W-tick stream with the block's whole
                         // Y staged once (K=ff<=768 fits smem); grid.y and Y
                         // traffic shrink NT-fold. Bitwise: per-output ascending
                         // fold and epilogue unchanged.
#if PD_QMMA2_YSYNC
#define PD_QMMA2_STAGE_INT32 (PD_QMMA2_RB * PD_MMQ_XK)
#define PD_QMMA2_SMEM_INT32 (PD_QMMA2_S * PD_QMMA2_STAGE_INT32 + 32u * PD_MMQ_XK)
#elif PD_QMMA2_ILV
// gate_up stage: Wg + Wu + Y. (The down kernel has one mat; its launcher
// sizes smem with the non-ILV stage.)
#define PD_QMMA2_STAGE_INT32 ((2u * PD_QMMA2_RB + 32u) * PD_MMQ_XK)
#define PD_QMMA2_DN_STAGE_INT32 ((PD_QMMA2_RB + 32u) * PD_MMQ_XK)
#define PD_QMMA2_SMEM_INT32 (PD_QMMA2_S * PD_QMMA2_STAGE_INT32)
#define PD_QMMA2_DN_SMEM_INT32 (PD_QMMA2_DN_S * PD_QMMA2_DN_STAGE_INT32)
#else
#define PD_QMMA2_STAGE_INT32 ((PD_QMMA2_RB + 32u) * PD_MMQ_XK)
#define PD_QMMA2_SMEM_INT32 (PD_QMMA2_S * PD_QMMA2_STAGE_INT32)
#endif
#ifndef PD_QMMA2_DN_STAGE_INT32
#define PD_QMMA2_DN_STAGE_INT32 PD_QMMA2_STAGE_INT32
#define PD_QMMA2_DN_SMEM_INT32 PD_QMMA2_SMEM_INT32
#endif

// Stage one 256-K tick into ring slot buffers (W tile: PD_QMMA_WK stride,
// data words + f16 scales at +64; Y tile: PD_MMQ_XK stride, data + f32
// scales at +64 -- the exact layouts the mma body reads). All guards
// zero-fill (cp.async src-size 0), reproducing the shipped stages' exact
// zeros. One commit_group per call, at the call site.
// FS64 (P1 dn64): Y scales are PER-64 groups (xs at in_dim/64 stride); each
// per-32 slot pair gets the group value DUPLICATED (L2-hot dup reads), so
// the tile layout - and any non-restructured fold - stays verbatim.
template <bool FS64 = false>
__device__ __forceinline__ void pd_qmma2_stage(
    int* __restrict__ wtile, int* __restrict__ ytile,
    const int8_t* __restrict__ wd, const __half* __restrict__ ws,
    const int* __restrict__ xq32, const float* __restrict__ xs,
    const unsigned int* __restrict__ rows, size_t wrow0, uint32_t row_base,
    uint32_t out_dim, uint32_t in_dim, uint32_t kt, uint32_t tid,
    bool with_y = true) {
#if PD_MMA_OK
    const uint32_t n_blocks = in_dim >> 5, n_k32 = in_dim >> 2;
    // W data: RB rows x 8 k32-blocks x 2 16B halves
    #pragma unroll
    for (uint32_t it = 0; it < PD_QMMA2_RB / 16u; ++it) {
        const uint32_t i = it * 256u + tid;
        const uint32_t row = i >> 4, half = i & 15u;
        const uint32_t b = half >> 1, h16 = half & 1u, gb = kt * 8u + b;
        const bool ok = gb < n_blocks && (row_base + row) < out_dim;
        pd_cp_async16(wtile + row * PD_QMMA_WK + b * 8u + h16 * 4u,
                      wd + ((wrow0 + row) * (size_t)in_dim) + (ok ? gb : 0u) * 32u
                          + h16 * 16u,
                      ok);
    }
    // W scales: 8 f16 per row = four 4B copies. gb is even and n_blocks is
    // even, so a pair is in or out together and the source stays 4B-aligned.
    if (tid < PD_QMMA2_RB * 4u) {
        const uint32_t row = tid >> 2, c = tid & 3u, gb = kt * 8u + c * 2u;
        const bool ok = gb < n_blocks && (row_base + row) < out_dim;
        pd_mma_cpa4p((__half*)(wtile + row * PD_QMMA_WK + 64u) + c * 2u,
                     ws + (wrow0 + row) * n_blocks + (ok ? gb : 0u), ok);
    }
#if !PD_QMMA2_YSYNC
    if (with_y) {
    // Y data: 32 cols x 16 16B chunks, gathered through rows[] (PAD -> zeros)
    #pragma unroll
    for (uint32_t it = 0; it < 2u; ++it) {
        const uint32_t i = it * 256u + tid;
        const uint32_t c = i >> 4, ch = i & 15u;
        const unsigned int r = rows[c];
        const uint32_t gk = kt * 64u + ch * 4u;
        const bool ok = r != PD_MOE_PAD && gk < n_k32;
        pd_cp_async16(ytile + c * PD_MMQ_XK + ch * 4u,
                      xq32 + (ok ? ((size_t)r * n_k32 + gk) : 0u), ok);
    }
    // Y scales: 8 f32 per col, one 4B copy each (FS64: the per-64 group
    // value lands in both slots of its per-32 pair; n_blocks is even)
    {
        const uint32_t c = tid >> 3, b = tid & 7u, gb = kt * 8u + b;
        const unsigned int r = rows[c];
        const bool ok = r != PD_MOE_PAD && gb < n_blocks;
        const size_t six = FS64 ? ((size_t)r * (n_blocks >> 1) + (gb >> 1))
                                : ((size_t)r * n_blocks + gb);
        pd_mma_cpa4p((float*)(ytile + c * PD_MMQ_XK + 64u) + b,
                     xs + (ok ? six : 0u), ok);
    }
    }
#else
    (void)ytile; (void)xq32; (void)xs; (void)rows; (void)with_y;
#endif  // !PD_QMMA2_YSYNC
#endif
}

// P1-2 stage twin: identical W staging; Y SCALES are per-128 groups
// (2 f32/row/tick instead of 8, xs stride n/128). Data layout unchanged.
__device__ __forceinline__ void pd_qmma2g_stage(
    int* __restrict__ wtile, int* __restrict__ ytile,
    const int8_t* __restrict__ wd, const __half* __restrict__ ws,
    const int* __restrict__ xq32, const float* __restrict__ xs,
    const unsigned int* __restrict__ rows, size_t wrow0, uint32_t row_base,
    uint32_t out_dim, uint32_t in_dim, uint32_t kt, uint32_t tid,
    bool with_y = true) {
#if PD_MMA_OK
    const uint32_t n_blocks = in_dim >> 5, n_k32 = in_dim >> 2;
    #pragma unroll
    for (uint32_t it = 0; it < PD_QMMA2_RB / 16u; ++it) {
        const uint32_t i = it * 256u + tid;
        const uint32_t row = i >> 4, half = i & 15u;
        const uint32_t b = half >> 1, h16 = half & 1u, gb = kt * 8u + b;
        const bool ok = gb < n_blocks && (row_base + row) < out_dim;
        pd_cp_async16(wtile + row * PD_QMMA_WK + b * 8u + h16 * 4u,
                      wd + ((wrow0 + row) * (size_t)in_dim) + (ok ? gb : 0u) * 32u
                          + h16 * 16u,
                      ok);
    }
    if (tid < PD_QMMA2_RB * 4u) {
        const uint32_t row = tid >> 2, c = tid & 3u, gb = kt * 8u + c * 2u;
        const bool ok = gb < n_blocks && (row_base + row) < out_dim;
        pd_mma_cpa4p((__half*)(wtile + row * PD_QMMA_WK + 64u) + c * 2u,
                     ws + (wrow0 + row) * n_blocks + (ok ? gb : 0u), ok);
    }
    if (with_y) {
    #pragma unroll
    for (uint32_t it = 0; it < 2u; ++it) {
        const uint32_t i = it * 256u + tid;
        const uint32_t c = i >> 4, ch = i & 15u;
        const unsigned int r = rows[c];
        const uint32_t gk = kt * 64u + ch * 4u;
        const bool ok = r != PD_MOE_PAD && gk < n_k32;
        pd_cp_async16(ytile + c * PD_MMQ_XK + ch * 4u,
                      xq32 + (ok ? ((size_t)r * n_k32 + gk) : 0u), ok);
    }
    // Y scales: 2 f32 per col per 256-K tick (per-128 groups, xs stride n/128)
    if (tid < 64u) {
        const uint32_t c = tid >> 1, g2 = tid & 1u;
        const uint32_t n_g = in_dim >> 7;
        const uint32_t gg = kt * 2u + g2;
        const unsigned int r = rows[c];
        const bool ok = r != PD_MOE_PAD && gg < n_g;
        pd_mma_cpa4p((float*)(ytile + c * PD_MMQ_XK + 64u) + g2,
                     xs + (ok ? ((size_t)r * n_g + gg) : 0u), ok);
    }
    }
#else
    (void)wtile;(void)ytile;(void)wd;(void)ws;(void)xq32;(void)xs;(void)rows;
    (void)wrow0;(void)row_base;(void)out_dim;(void)in_dim;(void)kt;(void)tid;(void)with_y;
#endif
}

// P1-2 consumer twin (per-128 activation scales): the mma2 ILV gate_up with
// the fold reassociated per 4-block group: t[q] += dA_bb * d (4 FMAs), then
// acc[q] = fma(dB_g, t[q], acc[q]) once per group - fold fp ops ~/1.8, dB
// smem loads /4. Precision-class (lane); LDM path only.
template <uint32_t S, bool GELU, bool Y64 = false>
__global__ void __launch_bounds__(256, PD_QMMA2_OCC_GU) pd_q8_0_moe_gate_up_mma2g_kernel(
    const int8_t* __restrict__ gate_data, const __half* __restrict__ gate_scale,
    const int8_t* __restrict__ up_data, const __half* __restrict__ up_scale,
    const unsigned int* __restrict__ sorted_row, const unsigned int* __restrict__ block_expert,
    const int8_t* __restrict__ xq, const float* __restrict__ xs,
    int8_t* __restrict__ fq, float* __restrict__ fs, uint32_t in_dim, uint32_t ff) {
#if PD_MMA_OK && PD_QMMA2_LDM && PD_QMMA2_ILV
    constexpr uint32_t BM = 32u;
    const uint32_t blk = blockIdx.x;
    const uint32_t e = block_expert[blk];
    if (e == PD_MOE_PAD) return;
    const uint32_t row_base = blockIdx.y * PD_QMMA2_RB;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, t = lane & 3u;
    constexpr uint32_t NW = PD_QMMA2_RB / 32u;
#if PD_QMMA2_WMAP
    const uint32_t i0 = (warp & 1u) * 16u * NW;
    const uint32_t joff = (warp >> 1) * 8u;
#else
    const uint32_t i0 = (warp >> 2) * 16u * NW;
    const uint32_t joff = (warp & 3u) * 8u;
#endif
    const uint32_t nk = (in_dim + 255u) >> 8;

    extern __shared__ int pd_qmma2g_sh[];
    __shared__ unsigned int tok[BM];
    __shared__ uint32_t nlive_sh;
    for (uint32_t i = tid; i < BM; i += 256u) tok[i] = sorted_row[(size_t)blk * BM + i];
    __syncthreads();
    if (warp == 0) {
        const uint32_t m = __ballot_sync(0xffffffffu, tok[lane] == PD_MOE_PAD);
        if (lane == 0) nlive_sh = m ? (uint32_t)(__ffs((int)m) - 1) : BM;
    }
    const size_t wrow0 = (size_t)e * ff + row_base;
    const uint32_t T = nk;
    auto stage_buf = [&](uint32_t s) { return pd_qmma2g_sh + s * PD_QMMA2_STAGE_INT32; };
    auto tick_stage = [&](uint32_t tk, uint32_t s) {
        int* yt = stage_buf(s) + 2u * PD_QMMA2_RB * PD_QMMA_WK;
        pd_qmma2g_stage(stage_buf(s), yt, gate_data, gate_scale, (const int*)xq,
                        xs, tok, wrow0, row_base, ff, in_dim, tk, tid);
        pd_qmma2g_stage(stage_buf(s) + PD_QMMA2_RB * PD_QMMA_WK, yt, up_data,
                        up_scale, (const int*)xq, xs, tok, wrow0, row_base, ff,
                        in_dim, tk, tid, /*with_y=*/false);
    };
    #pragma unroll
    for (uint32_t s = 0; s < S; ++s) {
        if (s < T) tick_stage(s, s);
        asm volatile("cp.async.commit_group;");
    }
    __syncthreads();
    const bool q_live = joff < nlive_sh;

    float acc_g[NW][4] = {}, acc_u[NW][4] = {};
    for (uint32_t tk = 0; tk < T; ++tk) {
        const uint32_t s = tk % S;
        pd_mma_cpa_waitN<(int)S - 1>();
        __syncthreads();
        if (q_live) {
          #pragma unroll
          for (uint32_t mat = 0; mat < 2u; ++mat) {
            int* tw = stage_buf(s) + mat * PD_QMMA2_RB * PD_QMMA_WK;
            int* tile_y = stage_buf(s) + 2u * PD_QMMA2_RB * PD_QMMA_WK;
            float(*acc)[4] = mat ? acc_u : acc_g;
            const uint32_t l7 = lane & 7u;
            const uint32_t arow_off = ((lane & 8u) ? 8u : 0u) + l7;
            const uint32_t akof = (lane & 16u) ? 16u : 0u;
            const uint32_t bkof = (lane & 8u) ? 16u : 0u;
            #pragma unroll
            for (uint32_t g2 = 0; g2 < 2u; ++g2) {
                const float dB0 =
                    ((const float*)tile_y)[(joff + 2u * t) * PD_MMQ_XK + 64u + g2];
                const float dB1 =
                    ((const float*)tile_y)[(joff + 2u * t + 1u) * PD_MMQ_XK + 64u + g2];
                float tp[NW][4];
                #pragma unroll
                for (uint32_t n = 0; n < NW; ++n) {
                    tp[n][0] = 0.f; tp[n][1] = 0.f; tp[n][2] = 0.f; tp[n][3] = 0.f;
                }
                #pragma unroll
                for (uint32_t bi = 0; bi < 4u; ++bi) {
                    const uint32_t bb = g2 * 4u + bi;
                    int b0, b1;
                    pd_mma_ldm_x2((const char*)(tile_y + (joff + l7) * PD_MMQ_XK)
                                      + bb * 32u + bkof,
                                  b0, b1);
                    #pragma unroll
                    for (uint32_t n = 0; n < NW; ++n) {
                        int A0, A1, A2, A3;
                        pd_mma_ldm_x4((const char*)(tw + (i0 + n * 16u + arow_off) * PD_QMMA_WK)
                                          + bb * 32u + akof,
                                      A0, A1, A2, A3);
                        const uint32_t r0 = (i0 + n * 16u + g) * PD_QMMA_WK;
                        const uint32_t r8 = (i0 + n * 16u + 8u + g) * PD_QMMA_WK;
                        const float dA0 = __half2float(((const __half*)(tw + r0 + 64u))[bb]);
                        const float dA1 = __half2float(((const __half*)(tw + r8 + 64u))[bb]);
                        int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                        asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                            : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                            : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(b0), "r"(b1));
                        tp[n][0] = __fmaf_rn(dA0, (float)d0, tp[n][0]);
                        tp[n][1] = __fmaf_rn(dA0, (float)d1, tp[n][1]);
                        tp[n][2] = __fmaf_rn(dA1, (float)d2, tp[n][2]);
                        tp[n][3] = __fmaf_rn(dA1, (float)d3, tp[n][3]);
                    }
                }
                #pragma unroll
                for (uint32_t n = 0; n < NW; ++n) {
                    acc[n][0] = __fmaf_rn(dB0, tp[n][0], acc[n][0]);
                    acc[n][1] = __fmaf_rn(dB1, tp[n][1], acc[n][1]);
                    acc[n][2] = __fmaf_rn(dB0, tp[n][2], acc[n][2]);
                    acc[n][3] = __fmaf_rn(dB1, tp[n][3], acc[n][3]);
                }
            }
          }
        }
        __syncthreads();
        if (tk + S < T) tick_stage(tk + S, s);
        asm volatile("cp.async.commit_group;");
    }

    const uint32_t n_sb = ff >> 5;
    if constexpr (!Y64) {
    if (q_live) {
        #pragma unroll
        for (uint32_t qc = 0; qc < 2u; ++qc) {
            const uint32_t c = joff + 2u * t + qc;
            const bool pad = tok[c] == PD_MOE_PAD;
            const uint32_t rb = row_base + i0;
            float sw[4];
            #pragma unroll
            for (uint32_t n = 0; n < NW; ++n) {
                #pragma unroll
                for (uint32_t hq = 0; hq < 2u; ++hq) {
                    const uint32_t q = qc + 2u * hq;
                    const uint32_t r = rb + n * 16u + hq * 8u + g;
                    float out = 0.f;
                    if (!pad && r < ff) {
                        const float gv = acc_g[n][q];
                        const float uv = acc_u[n][q];
                        out = GELU
                            ? 0.5f * gv
                                  * (1.0f
                                     + tanhf(0.79788456080286535587989211986876f * gv
                                             * (1.0f + 0.044715f * gv * gv)))
                                  * uv
                            : (gv / (1.0f + __expf(-gv))) * uv;
                    }
                    sw[n * 2u + hq] = out;
                }
            }
            float a = fmaxf(fmaxf(fabsf(sw[0]), fabsf(sw[1])), fmaxf(fabsf(sw[2]), fabsf(sw[3])));
            #pragma unroll
            for (uint32_t o = 4; o <= 16u; o <<= 1)
                a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
            const float scl = a * (1.0f / 127.0f);
            const float invs = scl > 0.f ? 1.0f / scl : 0.f;
            const size_t row = (size_t)blk * BM + c;
            if (rb < ff) {
                #pragma unroll
                for (uint32_t v = 0; v < 4u; ++v) {
                    const uint32_t r = rb + (v >> 1) * 16u + (v & 1u) * 8u + g;
                    int qi = __float2int_rn(sw[v] * invs);
                    qi = qi < -127 ? -127 : (qi > 127 ? 127 : qi);
                    fq[row * ff + r] = (int8_t)qi;
                }
                if (g == 0) fs[row * n_sb + (rb >> 5)] = scl;
            }
        }
    }
    } else {
    // Y64 (P1 dn64): PER-64 scale groups. Column c's 64-span [row_base,
    // row_base+64) is split across the two warps sharing joff (i0 = 0/32);
    // pair abs-max via a 32x2 smem plane (the ring is dead post-loop), then
    // each warp quantizes its own 32-span with the shared scale. All warps
    // reach both barriers (q_live guards only the data work); ff % 64 == 0
    // (launcher-gated) so every window is full.
    float* smax = (float*)pd_qmma2g_sh;
    float swq[2][4];
    float aq[2];
    #pragma unroll
    for (uint32_t qc = 0; qc < 2u; ++qc) {
        const uint32_t c = joff + 2u * t + qc;
        const bool pad = !q_live || tok[c] == PD_MOE_PAD;
        const uint32_t rb = row_base + i0;
        #pragma unroll
        for (uint32_t n = 0; n < NW; ++n) {
            #pragma unroll
            for (uint32_t hq = 0; hq < 2u; ++hq) {
                const uint32_t q = qc + 2u * hq;
                const uint32_t r = rb + n * 16u + hq * 8u + g;
                float out = 0.f;
                if (!pad && r < ff) {
                    const float gv = acc_g[n][q];
                    const float uv = acc_u[n][q];
                    out = GELU
                        ? 0.5f * gv
                              * (1.0f
                                 + tanhf(0.79788456080286535587989211986876f * gv
                                         * (1.0f + 0.044715f * gv * gv)))
                              * uv
                        : (gv / (1.0f + __expf(-gv))) * uv;
                }
                swq[qc][n * 2u + hq] = out;
            }
        }
        float a = fmaxf(fmaxf(fabsf(swq[qc][0]), fabsf(swq[qc][1])),
                        fmaxf(fabsf(swq[qc][2]), fabsf(swq[qc][3])));
        #pragma unroll
        for (uint32_t o = 4; o <= 16u; o <<= 1)
            a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
        aq[qc] = a;
        if (q_live && g == 0) smax[(i0 >> 5) * 32u + c] = a;
    }
    __syncthreads();
    if (q_live) {
        #pragma unroll
        for (uint32_t qc = 0; qc < 2u; ++qc) {
            const uint32_t c = joff + 2u * t + qc;
            const uint32_t rb = row_base + i0;
            const float am = fmaxf(aq[qc], smax[((i0 >> 5) ^ 1u) * 32u + c]);
            const float scl = am * (1.0f / 127.0f);
            const float invs = scl > 0.f ? 1.0f / scl : 0.f;
            const size_t row = (size_t)blk * BM + c;
            if (rb < ff) {
                #pragma unroll
                for (uint32_t v = 0; v < 4u; ++v) {
                    const uint32_t r = rb + (v >> 1) * 16u + (v & 1u) * 8u + g;
                    int qi = __float2int_rn(swq[qc][v] * invs);
                    qi = qi < -127 ? -127 : (qi > 127 ? 127 : qi);
                    fq[row * ff + r] = (int8_t)qi;
                }
                if (g == 0 && i0 == 0) fs[row * (n_sb >> 1) + (row_base >> 6)] = scl;
            }
        }
    }
    }
#else
    (void)gate_data; (void)gate_scale; (void)up_data; (void)up_scale; (void)sorted_row;
    (void)block_expert; (void)xq; (void)xs; (void)fq; (void)fs; (void)in_dim; (void)ff;
#endif
}

PD_EXPORT
int pd_q8_0_moe_gate_up_mma2g_geglu(const void* gate_data, const void* gate_scale,
                                    const void* up_data, const void* up_scale,
                                    const void* sorted_row, const void* block_expert,
                                    const void* xq, const void* xs, void* fq, void* fs,
                                    uint32_t in_dim, uint32_t ff, uint32_t max_blocks,
                                    uint32_t bm, void* stream) {
    if (ff == 0 || max_blocks == 0) return 0;
    if (bm != 32u) return cudaErrorNotSupported;
    if ((in_dim & 255u) != 0 || (ff & 31u) != 0) return cudaErrorInvalidValue;
    constexpr uint32_t S = (uint32_t)PD_QMMA2_S;
    constexpr uint32_t smem = PD_QMMA2_SMEM_INT32 * 4u;
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute((const void*)pd_q8_0_moe_gate_up_mma2g_kernel<S, true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        attr = true;
    }
    dim3 grid(max_blocks, (ff + PD_QMMA2_RB - 1u) / PD_QMMA2_RB);
    pd_q8_0_moe_gate_up_mma2g_kernel<S, true><<<grid, 256, smem, (cudaStream_t)stream>>>(
        (const int8_t*)gate_data, (const __half*)gate_scale, (const int8_t*)up_data,
        (const __half*)up_scale, (const unsigned int*)sorted_row,
        (const unsigned int*)block_expert, (const int8_t*)xq, (const float*)xs,
        (int8_t*)fq, (float*)fs, in_dim, ff);
    return pd_launch_status();
}

// GELU=true is the gemma4-A4B GEGLU twin (the exported one); the silu
// instantiation exists for a later qwen-A3B election.
template <uint32_t S, bool GELU>
__global__ void __launch_bounds__(256, PD_QMMA2_OCC_GU) pd_q8_0_moe_gate_up_mma2_kernel(
    const int8_t* __restrict__ gate_data, const __half* __restrict__ gate_scale,
    const int8_t* __restrict__ up_data, const __half* __restrict__ up_scale,
    const unsigned int* __restrict__ sorted_row, const unsigned int* __restrict__ block_expert,
    const int8_t* __restrict__ xq, const float* __restrict__ xs,
    int8_t* __restrict__ fq, float* __restrict__ fs, uint32_t in_dim, uint32_t ff) {
#if PD_MMA_OK
    constexpr uint32_t BM = 32u;
    const uint32_t blk = blockIdx.x;
    const uint32_t e = block_expert[blk];
    if (e == PD_MOE_PAD) return;
    const uint32_t row_base = blockIdx.y * PD_QMMA2_RB;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, t = lane & 3u;
    constexpr uint32_t NW = PD_QMMA2_RB / 32u;   // 16-row mma tiles per warp
#if PD_QMMA2_WMAP
    const uint32_t i0 = (warp & 1u) * 16u * NW;
    const uint32_t joff = (warp >> 1) * 8u;
#else
    const uint32_t i0 = (warp >> 2) * 16u * NW;
    const uint32_t joff = (warp & 3u) * 8u;
#endif
    const uint32_t nk = (in_dim + 255u) >> 8;

    extern __shared__ int pd_qmma2_sh[];
    __shared__ unsigned int tok[BM];
    __shared__ uint32_t nlive_sh;
    for (uint32_t i = tid; i < BM; i += 256u) tok[i] = sorted_row[(size_t)blk * BM + i];
    __syncthreads();
    if (warp == 0) {
        const uint32_t m = __ballot_sync(0xffffffffu, tok[lane] == PD_MOE_PAD);
        if (lane == 0) nlive_sh = m ? (uint32_t)(__ffs((int)m) - 1) : BM;
    }

    const size_t wrow0 = (size_t)e * ff + row_base;
#if PD_QMMA2_ILV
    const uint32_t T = nk;       // one fused gate+up K-walk
    auto stage_buf = [&](uint32_t s) { return pd_qmma2_sh + s * PD_QMMA2_STAGE_INT32; };
    auto tick_stage = [&](uint32_t tk, uint32_t s) {
        int* yt = stage_buf(s) + 2u * PD_QMMA2_RB * PD_QMMA_WK;
        pd_qmma2_stage(stage_buf(s), yt, gate_data, gate_scale, (const int*)xq,
                       xs, tok, wrow0, row_base, ff, in_dim, tk, tid);
        // second mat: W only (with_y=false - Y was staged once above)
        pd_qmma2_stage(stage_buf(s) + PD_QMMA2_RB * PD_QMMA_WK, yt, up_data,
                       up_scale, (const int*)xq, xs, tok, wrow0, row_base, ff,
                       in_dim, tk, tid, /*with_y=*/false);
    };
#else
    const uint32_t T = 2u * nk;  // flat gate-then-up tick walk
    auto stage_buf = [&](uint32_t s) { return pd_qmma2_sh + s * PD_QMMA2_STAGE_INT32; };
    auto tick_stage = [&](uint32_t tk, uint32_t s) {
        const bool up = tk >= nk;
        pd_qmma2_stage(stage_buf(s), stage_buf(s) + PD_QMMA2_RB * PD_QMMA_WK,
                       up ? up_data : gate_data, up ? up_scale : gate_scale,
                       (const int*)xq, xs, tok, wrow0, row_base, ff, in_dim,
                       up ? tk - nk : tk, tid);
    };
#endif
    // exactly S groups always outstanding: real stages while they last,
    // empty commit_groups past the end -- keeps the wait immediate constant
    // for any T (down at S=3 has T=3).
    #pragma unroll
    for (uint32_t s = 0; s < S; ++s) {
        if (s < T) tick_stage(s, s);
        asm volatile("cp.async.commit_group;");
    }
    __syncthreads();  // nlive_sh (and tok) visible to every warp
    const bool q_live = joff < nlive_sh;

    float acc_g[NW][4] = {}, acc_u[NW][4] = {};
    for (uint32_t tk = 0; tk < T; ++tk) {
        const uint32_t s = tk % S;
        pd_mma_cpa_waitN<(int)S - 1>();
#if PD_QMMA2_YSYNC
        pd_moeq_stage_y<32u>(pd_qmma2_sh + PD_QMMA2_S * PD_QMMA2_STAGE_INT32,
                             (const int*)xq, xs, tok, in_dim,
                             (tk >= nk) ? tk - nk : tk, tid);
#endif
        __syncthreads();
        if (q_live) {
#if PD_QMMA2_ILV
          #pragma unroll
          for (uint32_t mat = 0; mat < 2u; ++mat) {
            int* tw = stage_buf(s) + mat * PD_QMMA2_RB * PD_QMMA_WK;
            int* tile_y = stage_buf(s) + 2u * PD_QMMA2_RB * PD_QMMA_WK;
            float(*acc)[4] = mat ? acc_u : acc_g;
#else
            int* tw = stage_buf(s);
            int* tile_y = PD_QMMA2_YSYNC ? pd_qmma2_sh + PD_QMMA2_S * PD_QMMA2_STAGE_INT32
                                         : tw + PD_QMMA2_RB * PD_QMMA_WK;
            float(*acc)[4] = (tk >= nk) ? acc_u : acc_g;
#endif
#if PD_QMMA2_LDM
            {
            const uint32_t l7 = lane & 7u;
            const uint32_t arow_off = ((lane & 8u) ? 8u : 0u) + l7;
            const uint32_t akof = (lane & 16u) ? 16u : 0u;  // bytes
            const uint32_t bkof = (lane & 8u) ? 16u : 0u;   // bytes
            #pragma unroll
            for (uint32_t bb = 0; bb < 8u; ++bb) {
                const uint32_t ko = bb * 8u;
                int b0, b1;
                pd_mma_ldm_x2((const char*)(tile_y + (joff + l7) * PD_MMQ_XK)
                                  + ko * 4u + bkof,
                              b0, b1);
                const float dB0 =
                    ((const float*)tile_y)[(joff + 2u * t) * PD_MMQ_XK + 64u + bb];
                const float dB1 =
                    ((const float*)tile_y)[(joff + 2u * t + 1u) * PD_MMQ_XK + 64u + bb];
                #pragma unroll
                for (uint32_t n = 0; n < NW; ++n) {
                    int A0, A1, A2, A3;
                    pd_mma_ldm_x4((const char*)(tw + (i0 + n * 16u + arow_off) * PD_QMMA_WK)
                                      + bb * 32u + akof,
                                  A0, A1, A2, A3);
                    const uint32_t r0 = (i0 + n * 16u + g) * PD_QMMA_WK;
                    const uint32_t r8 = (i0 + n * 16u + 8u + g) * PD_QMMA_WK;
                    const float dA0 = __half2float(((const __half*)(tw + r0 + 64u))[bb]);
                    const float dA1 = __half2float(((const __half*)(tw + r8 + 64u))[bb]);
                    int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                        : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(b0), "r"(b1));
                    acc[n][0] += dA0 * dB0 * (float)d0;
                    acc[n][1] += dA0 * dB1 * (float)d1;
                    acc[n][2] += dA1 * dB0 * (float)d2;
                    acc[n][3] += dA1 * dB1 * (float)d3;
                }
            }
            }
#else
            #pragma unroll
            for (uint32_t h = 0; h < 2u; ++h) {
                const uint32_t k00 = h * 32u;
                #pragma unroll
                for (uint32_t kk = 0; kk < 4u; ++kk) {
                    const uint32_t bb = (k00 >> 3) + kk;
                    const uint32_t ko = k00 + kk * 8u;
                    const int b0 = tile_y[(joff + g) * PD_MMQ_XK + ko + t];
                    const int b1 = tile_y[(joff + g) * PD_MMQ_XK + ko + 4u + t];
                    const float dB0 =
                        ((const float*)tile_y)[(joff + 2u * t) * PD_MMQ_XK + 64u + bb];
                    const float dB1 =
                        ((const float*)tile_y)[(joff + 2u * t + 1u) * PD_MMQ_XK + 64u + bb];
                    #pragma unroll
                    for (uint32_t n = 0; n < NW; ++n) {
                        const uint32_t r0 = (i0 + n * 16u + g) * PD_QMMA_WK;
                        const uint32_t r8 = (i0 + n * 16u + 8u + g) * PD_QMMA_WK;
                        const int A0 = tw[r0 + bb * 8u + t];
                        const int A2 = tw[r0 + bb * 8u + 4u + t];
                        const int A1 = tw[r8 + bb * 8u + t];
                        const int A3 = tw[r8 + bb * 8u + 4u + t];
                        const float dA0 = __half2float(((const __half*)(tw + r0 + 64u))[bb]);
                        const float dA1 = __half2float(((const __half*)(tw + r8 + 64u))[bb]);
                        int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                        asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                            : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                            : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(b0), "r"(b1));
                        acc[n][0] += dA0 * dB0 * (float)d0;
                        acc[n][1] += dA0 * dB1 * (float)d1;
                        acc[n][2] += dA1 * dB0 * (float)d2;
                        acc[n][3] += dA1 * dB1 * (float)d3;
                    }
                }
            }
#endif  // PD_QMMA2_LDM
#if PD_QMMA2_ILV
          }
#endif
        }
        __syncthreads();  // ring slot s is rewritten by the next stage
        if (tk + S < T) tick_stage(tk + S, s);
        asm volatile("cp.async.commit_group;");
    }

    // Epilogue, under the quarter-skip: dead quarters leave their fq/fs
    // unwritten (see the header note). NW==2 (RB=64): the shipped epilogue
    // verbatim - one warp holds the whole 32-row quantize block. NW==1
    // (RB=32): the block spans the two warp row-tiles, so per-thread outs
    // bounce through a [32 x 32] f32 smem plane (the ring is dead by now)
    // and warps 0-3 run the same amax/quantize shape over it - the paired
    // warp's mma chain is the identical add sequence on identical
    // fragments, so values, tree and stores stay bit-identical.
    const uint32_t n_sb = ff >> 5;
    if constexpr (NW == 2u) {
    if (q_live) {
        #pragma unroll
        for (uint32_t qc = 0; qc < 2u; ++qc) {
            const uint32_t c = joff + 2u * t + qc;
            const bool pad = tok[c] == PD_MOE_PAD;
            const uint32_t rb = row_base + i0;
            float sw[4];
            #pragma unroll
            for (uint32_t n = 0; n < NW; ++n) {
                #pragma unroll
                for (uint32_t hq = 0; hq < 2u; ++hq) {
                    const uint32_t q = qc + 2u * hq;
                    const uint32_t r = rb + n * 16u + hq * 8u + g;
                    float out = 0.f;
                    if (!pad && r < ff) {
                        const float gv = acc_g[n][q];
                        const float uv = acc_u[n][q];
                        out = GELU
                            ? 0.5f * gv
                                  * (1.0f
                                     + tanhf(0.79788456080286535587989211986876f * gv
                                             * (1.0f + 0.044715f * gv * gv)))
                                  * uv
                            : (gv / (1.0f + __expf(-gv))) * uv;
                    }
                    sw[n * 2u + hq] = out;
                }
            }
            float a = fmaxf(fmaxf(fabsf(sw[0]), fabsf(sw[1])), fmaxf(fabsf(sw[2]), fabsf(sw[3])));
            #pragma unroll
            for (uint32_t o = 4; o <= 16u; o <<= 1)
                a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
            const float scl = a * (1.0f / 127.0f);
            const float invs = scl > 0.f ? 1.0f / scl : 0.f;
            const size_t row = (size_t)blk * BM + c;
            if (rb < ff) {
                #pragma unroll
                for (uint32_t v = 0; v < 4u; ++v) {
                    const uint32_t r = rb + (v >> 1) * 16u + (v & 1u) * 8u + g;
                    int qi = __float2int_rn(sw[v] * invs);
                    qi = qi < -127 ? -127 : (qi > 127 ? 127 : qi);
                    fq[row * ff + r] = (int8_t)qi;
                }
                if (g == 0) fs[row * n_sb + (rb >> 5)] = scl;
            }
        }
    }
    } else {
    float* plane = (float*)pd_qmma2_sh;  // 32 rows x 32 cols
    if (q_live) {
        #pragma unroll
        for (uint32_t qc = 0; qc < 2u; ++qc) {
            const uint32_t c = joff + 2u * t + qc;
            const bool pad = tok[c] == PD_MOE_PAD;
            #pragma unroll
            for (uint32_t hq = 0; hq < 2u; ++hq) {
                const uint32_t q = qc + 2u * hq;
                const uint32_t r16 = i0 + hq * 8u + g;
                const uint32_t r = row_base + r16;
                float out = 0.f;
                if (!pad && r < ff) {
                    const float gv = acc_g[0][q];
                    const float uv = acc_u[0][q];
                    out = GELU
                        ? 0.5f * gv
                              * (1.0f
                                 + tanhf(0.79788456080286535587989211986876f * gv
                                         * (1.0f + 0.044715f * gv * gv)))
                              * uv
                        : (gv / (1.0f + __expf(-gv))) * uv;
                }
                plane[r16 * 32u + c] = out;
            }
        }
    }
    __syncthreads();
    if (q_live && (warp >> 2) == 0u) {
        #pragma unroll
        for (uint32_t qc = 0; qc < 2u; ++qc) {
            const uint32_t c = joff + 2u * t + qc;
            float sw[4];
            #pragma unroll
            for (uint32_t v = 0; v < 4u; ++v)
                sw[v] = plane[((v >> 1) * 16u + (v & 1u) * 8u + g) * 32u + c];
            float a = fmaxf(fmaxf(fabsf(sw[0]), fabsf(sw[1])), fmaxf(fabsf(sw[2]), fabsf(sw[3])));
            #pragma unroll
            for (uint32_t o = 4; o <= 16u; o <<= 1)
                a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
            const float scl = a * (1.0f / 127.0f);
            const float invs = scl > 0.f ? 1.0f / scl : 0.f;
            const size_t row = (size_t)blk * BM + c;
            if (row_base < ff) {
                #pragma unroll
                for (uint32_t v = 0; v < 4u; ++v) {
                    const uint32_t r = row_base + (v >> 1) * 16u + (v & 1u) * 8u + g;
                    int qi = __float2int_rn(sw[v] * invs);
                    qi = qi < -127 ? -127 : (qi > 127 ? 127 : qi);
                    fq[row * ff + r] = (int8_t)qi;
                }
                if (g == 0) fs[row * n_sb + (row_base >> 5)] = scl;
            }
        }
    }
    }
#else
    (void)gate_data; (void)gate_scale; (void)up_data; (void)up_scale; (void)sorted_row;
    (void)block_expert; (void)xq; (void)xs; (void)fq; (void)fs; (void)in_dim; (void)ff;
#endif
}

// Down twin: K = ff, activations are the sorted fq rows (direct index, no
// gather), deterministic scattered epilogue -- the shipped down_mma body on
// the v2 ring, with the quarter skip.
// FS64 (P1 dn64): fs is per-64 groups; staging duplicates each group value
// into its slot pair and the LDM fold regroups per pair (tp += dA*d, then
// one dB FMA) - the mma2g group-fold shape on the down side.
template <uint32_t S, bool PBF16 = false, bool FS64 = false>
__global__ void __launch_bounds__(256, PD_QMMA2_OCC_DN) pd_q8_0_moe_down_mma2_kernel(
    const int8_t* __restrict__ down_data, const __half* __restrict__ down_scale,
    const unsigned int* __restrict__ sorted_row, const unsigned int* __restrict__ sorted_slot,
    const unsigned int* __restrict__ block_expert, const float* __restrict__ topk_w,
    const int8_t* __restrict__ fq, const float* __restrict__ fs,
    float* __restrict__ part, uint32_t ff, uint32_t embd, uint32_t n_active) {
#if PD_MMA_OK
    constexpr uint32_t BM = 32u;
    const uint32_t blk = blockIdx.x;
    const uint32_t e = block_expert[blk];
    if (e == PD_MOE_PAD) return;
    const uint32_t row_base = blockIdx.y * PD_QMMA2_RB;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, t = lane & 3u;
    constexpr uint32_t NW = PD_QMMA2_RB / 32u;   // 16-row mma tiles per warp
#if PD_QMMA2_WMAP
    const uint32_t i0 = (warp & 1u) * 16u * NW;
    const uint32_t joff = (warp >> 1) * 8u;
#else
    const uint32_t i0 = (warp >> 2) * 16u * NW;
    const uint32_t joff = (warp & 3u) * 8u;
#endif
    const uint32_t nk = (ff + 255u) >> 8;

    extern __shared__ int pd_qmma2_sh[];
    __shared__ unsigned int tok[BM], slt[BM], idn[BM];
    __shared__ uint32_t nlive_sh;
    for (uint32_t i = tid; i < BM; i += 256u) {
        tok[i] = sorted_row[(size_t)blk * BM + i];
        slt[i] = sorted_slot[(size_t)blk * BM + i];
        idn[i] = blk * BM + i;  // fq rows are sorted-contiguous
    }
    __syncthreads();
    if (warp == 0) {
        const uint32_t m = __ballot_sync(0xffffffffu, tok[lane] == PD_MOE_PAD);
        if (lane == 0) nlive_sh = m ? (uint32_t)(__ffs((int)m) - 1) : BM;
    }

    const size_t wrow0 = (size_t)e * embd + row_base;
    const uint32_t T = nk;
    auto stage_buf = [&](uint32_t s) { return pd_qmma2_sh + s * PD_QMMA2_DN_STAGE_INT32; };
    auto tick_stage = [&](uint32_t tk, uint32_t s) {
        pd_qmma2_stage<FS64>(stage_buf(s), stage_buf(s) + PD_QMMA2_RB * PD_QMMA_WK, down_data,
                       down_scale, (const int*)fq, fs, idn, wrow0, row_base, embd,
                       ff, tk, tid);
    };
    #pragma unroll
    for (uint32_t s = 0; s < S; ++s) {
        if (s < T) tick_stage(s, s);
        asm volatile("cp.async.commit_group;");
    }
    __syncthreads();
    const bool q_live = joff < nlive_sh;

    float acc[NW][4] = {};
    for (uint32_t tk = 0; tk < T; ++tk) {
        const uint32_t s = tk % S;
        pd_mma_cpa_waitN<(int)S - 1>();
#if PD_QMMA2_YSYNC
        pd_moeq_stage_y<32u>(pd_qmma2_sh + PD_QMMA2_S * PD_QMMA2_STAGE_INT32,
                             (const int*)fq, fs, idn, ff, tk, tid);
#endif
        __syncthreads();
        if (q_live) {
            int* tw = stage_buf(s);
            int* tile_y = PD_QMMA2_YSYNC ? pd_qmma2_sh + PD_QMMA2_S * PD_QMMA2_STAGE_INT32
                                         : tw + PD_QMMA2_RB * PD_QMMA_WK;
#if PD_QMMA2_LDM
            {
            const uint32_t l7 = lane & 7u;
            const uint32_t arow_off = ((lane & 8u) ? 8u : 0u) + l7;
            const uint32_t akof = (lane & 16u) ? 16u : 0u;  // bytes
            const uint32_t bkof = (lane & 8u) ? 16u : 0u;   // bytes
            if constexpr (!FS64) {
            #pragma unroll
            for (uint32_t bb = 0; bb < 8u; ++bb) {
                const uint32_t ko = bb * 8u;
                int b0, b1;
                pd_mma_ldm_x2((const char*)(tile_y + (joff + l7) * PD_MMQ_XK)
                                  + ko * 4u + bkof,
                              b0, b1);
                const float dB0 =
                    ((const float*)tile_y)[(joff + 2u * t) * PD_MMQ_XK + 64u + bb];
                const float dB1 =
                    ((const float*)tile_y)[(joff + 2u * t + 1u) * PD_MMQ_XK + 64u + bb];
                #pragma unroll
                for (uint32_t n = 0; n < NW; ++n) {
                    int A0, A1, A2, A3;
                    pd_mma_ldm_x4((const char*)(tw + (i0 + n * 16u + arow_off) * PD_QMMA_WK)
                                      + bb * 32u + akof,
                                  A0, A1, A2, A3);
                    const uint32_t r0 = (i0 + n * 16u + g) * PD_QMMA_WK;
                    const uint32_t r8 = (i0 + n * 16u + 8u + g) * PD_QMMA_WK;
                    const float dA0 = __half2float(((const __half*)(tw + r0 + 64u))[bb]);
                    const float dA1 = __half2float(((const __half*)(tw + r8 + 64u))[bb]);
                    int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                        : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(b0), "r"(b1));
                    acc[n][0] += dA0 * dB0 * (float)d0;
                    acc[n][1] += dA0 * dB1 * (float)d1;
                    acc[n][2] += dA1 * dB0 * (float)d2;
                    acc[n][3] += dA1 * dB1 * (float)d3;
                }
            }
            } else {
            // FS64: dB is per-64 (duplicated slots) - regroup per bb pair:
            // tp += dA*d over the pair, then one dB FMA (mma2g fold shape).
            #pragma unroll
            for (uint32_t g2 = 0; g2 < 4u; ++g2) {
                const float dB0 =
                    ((const float*)tile_y)[(joff + 2u * t) * PD_MMQ_XK + 64u + g2 * 2u];
                const float dB1 =
                    ((const float*)tile_y)[(joff + 2u * t + 1u) * PD_MMQ_XK + 64u + g2 * 2u];
                float tp[NW][4];
                #pragma unroll
                for (uint32_t n = 0; n < NW; ++n) {
                    tp[n][0] = 0.f; tp[n][1] = 0.f; tp[n][2] = 0.f; tp[n][3] = 0.f;
                }
                #pragma unroll
                for (uint32_t bi = 0; bi < 2u; ++bi) {
                    const uint32_t bb = g2 * 2u + bi;
                    const uint32_t ko = bb * 8u;
                    int b0, b1;
                    pd_mma_ldm_x2((const char*)(tile_y + (joff + l7) * PD_MMQ_XK)
                                      + ko * 4u + bkof,
                                  b0, b1);
                    #pragma unroll
                    for (uint32_t n = 0; n < NW; ++n) {
                        int A0, A1, A2, A3;
                        pd_mma_ldm_x4((const char*)(tw + (i0 + n * 16u + arow_off) * PD_QMMA_WK)
                                          + bb * 32u + akof,
                                      A0, A1, A2, A3);
                        const uint32_t r0 = (i0 + n * 16u + g) * PD_QMMA_WK;
                        const uint32_t r8 = (i0 + n * 16u + 8u + g) * PD_QMMA_WK;
                        const float dA0 = __half2float(((const __half*)(tw + r0 + 64u))[bb]);
                        const float dA1 = __half2float(((const __half*)(tw + r8 + 64u))[bb]);
                        int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                        asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                            : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                            : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(b0), "r"(b1));
                        tp[n][0] = __fmaf_rn(dA0, (float)d0, tp[n][0]);
                        tp[n][1] = __fmaf_rn(dA0, (float)d1, tp[n][1]);
                        tp[n][2] = __fmaf_rn(dA1, (float)d2, tp[n][2]);
                        tp[n][3] = __fmaf_rn(dA1, (float)d3, tp[n][3]);
                    }
                }
                #pragma unroll
                for (uint32_t n = 0; n < NW; ++n) {
                    acc[n][0] = __fmaf_rn(dB0, tp[n][0], acc[n][0]);
                    acc[n][1] = __fmaf_rn(dB1, tp[n][1], acc[n][1]);
                    acc[n][2] = __fmaf_rn(dB0, tp[n][2], acc[n][2]);
                    acc[n][3] = __fmaf_rn(dB1, tp[n][3], acc[n][3]);
                }
            }
            }
            }
#else
            #pragma unroll
            for (uint32_t h = 0; h < 2u; ++h) {
                const uint32_t k00 = h * 32u;
                #pragma unroll
                for (uint32_t kk = 0; kk < 4u; ++kk) {
                    const uint32_t bb = (k00 >> 3) + kk;
                    const uint32_t ko = k00 + kk * 8u;
                    const int b0 = tile_y[(joff + g) * PD_MMQ_XK + ko + t];
                    const int b1 = tile_y[(joff + g) * PD_MMQ_XK + ko + 4u + t];
                    const float dB0 =
                        ((const float*)tile_y)[(joff + 2u * t) * PD_MMQ_XK + 64u + bb];
                    const float dB1 =
                        ((const float*)tile_y)[(joff + 2u * t + 1u) * PD_MMQ_XK + 64u + bb];
                    #pragma unroll
                    for (uint32_t n = 0; n < NW; ++n) {
                        const uint32_t r0 = (i0 + n * 16u + g) * PD_QMMA_WK;
                        const uint32_t r8 = (i0 + n * 16u + 8u + g) * PD_QMMA_WK;
                        const int A0 = tw[r0 + bb * 8u + t];
                        const int A2 = tw[r0 + bb * 8u + 4u + t];
                        const int A1 = tw[r8 + bb * 8u + t];
                        const int A3 = tw[r8 + bb * 8u + 4u + t];
                        const float dA0 = __half2float(((const __half*)(tw + r0 + 64u))[bb]);
                        const float dA1 = __half2float(((const __half*)(tw + r8 + 64u))[bb]);
                        int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                        asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                            : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                            : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(b0), "r"(b1));
                        acc[n][0] += dA0 * dB0 * (float)d0;
                        acc[n][1] += dA0 * dB1 * (float)d1;
                        acc[n][2] += dA1 * dB0 * (float)d2;
                        acc[n][3] += dA1 * dB1 * (float)d3;
                    }
                }
            }
#endif  // PD_QMMA2_LDM
        }
        __syncthreads();
        if (tk + S < T) tick_stage(tk + S, s);
        asm volatile("cp.async.commit_group;");
    }

    if (q_live) {
        const uint32_t c0 = joff + 2u * t;
        #pragma unroll
        for (uint32_t n = 0; n < NW; ++n) {
            const uint32_t r0 = row_base + i0 + n * 16u + g;
            const uint32_t r8 = r0 + 8u;
            #pragma unroll
            for (uint32_t q = 0; q < 4u; ++q) {
                const uint32_t r = (q & 2u) ? r8 : r0;
                const uint32_t c = c0 + (q & 1u);
                const unsigned int token = tok[c];
                if (r >= embd || token == PD_MOE_PAD) continue;
                const float w = topk_w[(size_t)token * n_active + slt[c]];
                const size_t pidx = ((size_t)token * n_active + slt[c]) * embd + r;
                if (PBF16)   // hibatch P1-1: bf16 partials (tail sums f32)
                    ((__nv_bfloat16*)part)[pidx] = __float2bfloat16(w * acc[n][q]);
                else
                    part[pidx] = w * acc[n][q];
            }
        }
    }
#else
    (void)down_data; (void)down_scale; (void)sorted_row; (void)sorted_slot;
    (void)block_expert; (void)topk_w; (void)fq; (void)fs; (void)part;
    (void)ff; (void)embd; (void)n_active;
#endif
}


#if PD_QMMA2_DN_NT > 1
// Down NT twin (high-batch lane): grid (blk, embd/(RB*NT)). The Y tile is
// identical for every RB-row slice of a token-block, so it is staged once
// (its own commit group) and the S-slot ring carries W only. The flat tick
// stream f = sl*nk + tk walks NT slices back-to-back: the ring never drains
// between slices, the fill amortizes NT-fold. Per (row, token) output the
// mma/fold/epilogue sequence is the NT=1 kernel's verbatim => bitwise.
#define PD_QMMA2_DNNT_WSTAGE (PD_QMMA2_RB * PD_QMMA_WK)
template <uint32_t S, uint32_t NT>
__global__ void __launch_bounds__(256, PD_QMMA2_OCC_DN) pd_q8_0_moe_down_mma2nt_kernel(
    const int8_t* __restrict__ down_data, const __half* __restrict__ down_scale,
    const unsigned int* __restrict__ sorted_row, const unsigned int* __restrict__ sorted_slot,
    const unsigned int* __restrict__ block_expert, const float* __restrict__ topk_w,
    const int8_t* __restrict__ fq, const float* __restrict__ fs,
    float* __restrict__ part, uint32_t ff, uint32_t embd, uint32_t n_active) {
#if PD_MMA_OK
    constexpr uint32_t BM = 32u;
    const uint32_t blk = blockIdx.x;
    const uint32_t e = block_expert[blk];
    if (e == PD_MOE_PAD) return;
    const uint32_t slice0 = blockIdx.y * NT;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, t = lane & 3u;
    constexpr uint32_t NW = PD_QMMA2_RB / 32u;
#if PD_QMMA2_WMAP
    const uint32_t i0 = (warp & 1u) * 16u * NW;
    const uint32_t joff = (warp >> 1) * 8u;
#else
    const uint32_t i0 = (warp >> 2) * 16u * NW;
    const uint32_t joff = (warp & 3u) * 8u;
#endif
    const uint32_t nk = (ff + 255u) >> 8;
    const uint32_t nslice_all = (embd + PD_QMMA2_RB - 1u) / PD_QMMA2_RB;
    const uint32_t nsl = (slice0 >= nslice_all) ? 0u
                          : ((nslice_all - slice0 < NT) ? (nslice_all - slice0) : NT);
    const uint32_t TT = nsl * nk;
    if (TT == 0) return;

    extern __shared__ int pd_qmma2nt_sh[];
    int* ybase = pd_qmma2nt_sh + S * PD_QMMA2_DNNT_WSTAGE;
    __shared__ unsigned int tok[BM], slt[BM];
    __shared__ uint32_t nlive_sh;
    for (uint32_t i = tid; i < BM; i += 256u) {
        tok[i] = sorted_row[(size_t)blk * BM + i];
        slt[i] = sorted_slot[(size_t)blk * BM + i];
    }
    __syncthreads();
    if (warp == 0) {
        const uint32_t m = __ballot_sync(0xffffffffu, tok[lane] == PD_MOE_PAD);
        if (lane == 0) nlive_sh = m ? (uint32_t)(__ffs((int)m) - 1) : BM;
    }

    // Stage the block's entire Y (all nk ticks) once; fq rows are the
    // sorted-contiguous blk*BM+c, exactly the NT=1 kernel's idn addressing.
    {
        const uint32_t n_blocks = ff >> 5, n_k32 = ff >> 2;
        for (uint32_t i = tid; i < nk * 512u; i += 256u) {
            const uint32_t tk = i >> 9, j = i & 511u;
            const uint32_t c = j >> 4, ch = j & 15u;
            const uint32_t gk = tk * 64u + ch * 4u;
            const bool ok = gk < n_k32;
            pd_cp_async16(ybase + tk * PD_MOEQ_Y_INT32 + c * PD_MMQ_XK + ch * 4u,
                          (const int*)fq + (ok ? ((size_t)(blk * BM + c) * n_k32 + gk) : 0u),
                          ok);
        }
        for (uint32_t i = tid; i < nk * 256u; i += 256u) {
            const uint32_t tk = i >> 8, j = i & 255u;
            const uint32_t c = j >> 3, b = j & 7u;
            const uint32_t gb = tk * 8u + b;
            const bool ok = gb < n_blocks;
            pd_mma_cpa4p((float*)(ybase + tk * PD_MOEQ_Y_INT32 + c * PD_MMQ_XK + 64u) + b,
                         fs + (ok ? ((size_t)(blk * BM + c) * n_blocks + gb) : 0u), ok);
        }
    }
    asm volatile("cp.async.commit_group;");

    auto slot = [&](uint32_t s2) { return pd_qmma2nt_sh + s2 * PD_QMMA2_DNNT_WSTAGE; };
    auto stage_w = [&](uint32_t f, uint32_t s2) {
        const uint32_t sl = f / nk, tk = f % nk;
        const uint32_t row_base = (slice0 + sl) * PD_QMMA2_RB;
        const size_t wrow0 = (size_t)e * embd + row_base;
        pd_qmma2_stage(slot(s2), (int*)nullptr, down_data, down_scale,
                       (const int*)nullptr, (const float*)nullptr,
                       (const unsigned int*)nullptr, wrow0, row_base, embd, ff, tk,
                       tid, /*with_y=*/false);
    };
    #pragma unroll
    for (uint32_t s2 = 0; s2 < S; ++s2) {
        if (s2 < TT) stage_w(s2, s2);
        asm volatile("cp.async.commit_group;");
    }
    __syncthreads();
    const bool q_live = joff < nlive_sh;

    float acc[NW][4] = {};
    uint32_t sl_cur = 0;
    for (uint32_t f = 0; f < TT; ++f) {
        const uint32_t s = f % S;
        const uint32_t tk = f % nk;
        pd_mma_cpa_waitN<(int)S - 1>();
        __syncthreads();
        if (q_live) {
            int* tw = slot(s);
            int* tile_y = ybase + tk * PD_MOEQ_Y_INT32;
#if PD_QMMA2_LDM
            {
            const uint32_t l7 = lane & 7u;
            const uint32_t arow_off = ((lane & 8u) ? 8u : 0u) + l7;
            const uint32_t akof = (lane & 16u) ? 16u : 0u;  // bytes
            const uint32_t bkof = (lane & 8u) ? 16u : 0u;   // bytes
            #pragma unroll
            for (uint32_t bb = 0; bb < 8u; ++bb) {
                const uint32_t ko = bb * 8u;
                int b0, b1;
                pd_mma_ldm_x2((const char*)(tile_y + (joff + l7) * PD_MMQ_XK)
                                  + ko * 4u + bkof,
                              b0, b1);
                const float dB0 =
                    ((const float*)tile_y)[(joff + 2u * t) * PD_MMQ_XK + 64u + bb];
                const float dB1 =
                    ((const float*)tile_y)[(joff + 2u * t + 1u) * PD_MMQ_XK + 64u + bb];
                #pragma unroll
                for (uint32_t n = 0; n < NW; ++n) {
                    int A0, A1, A2, A3;
                    pd_mma_ldm_x4((const char*)(tw + (i0 + n * 16u + arow_off) * PD_QMMA_WK)
                                      + bb * 32u + akof,
                                  A0, A1, A2, A3);
                    const uint32_t r0 = (i0 + n * 16u + g) * PD_QMMA_WK;
                    const uint32_t r8 = (i0 + n * 16u + 8u + g) * PD_QMMA_WK;
                    const float dA0 = __half2float(((const __half*)(tw + r0 + 64u))[bb]);
                    const float dA1 = __half2float(((const __half*)(tw + r8 + 64u))[bb]);
                    int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                        : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(b0), "r"(b1));
                    acc[n][0] += dA0 * dB0 * (float)d0;
                    acc[n][1] += dA0 * dB1 * (float)d1;
                    acc[n][2] += dA1 * dB0 * (float)d2;
                    acc[n][3] += dA1 * dB1 * (float)d3;
                }
            }
            }
#else
            #pragma unroll
            for (uint32_t h = 0; h < 2u; ++h) {
                const uint32_t k00 = h * 32u;
                #pragma unroll
                for (uint32_t kk = 0; kk < 4u; ++kk) {
                    const uint32_t bb = (k00 >> 3) + kk;
                    const uint32_t ko = k00 + kk * 8u;
                    const int b0 = tile_y[(joff + g) * PD_MMQ_XK + ko + t];
                    const int b1 = tile_y[(joff + g) * PD_MMQ_XK + ko + 4u + t];
                    const float dB0 =
                        ((const float*)tile_y)[(joff + 2u * t) * PD_MMQ_XK + 64u + bb];
                    const float dB1 =
                        ((const float*)tile_y)[(joff + 2u * t + 1u) * PD_MMQ_XK + 64u + bb];
                    #pragma unroll
                    for (uint32_t n = 0; n < NW; ++n) {
                        const uint32_t r0 = (i0 + n * 16u + g) * PD_QMMA_WK;
                        const uint32_t r8 = (i0 + n * 16u + 8u + g) * PD_QMMA_WK;
                        const int A0 = tw[r0 + bb * 8u + t];
                        const int A2 = tw[r0 + bb * 8u + 4u + t];
                        const int A1 = tw[r8 + bb * 8u + t];
                        const int A3 = tw[r8 + bb * 8u + 4u + t];
                        const float dA0 = __half2float(((const __half*)(tw + r0 + 64u))[bb]);
                        const float dA1 = __half2float(((const __half*)(tw + r8 + 64u))[bb]);
                        int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                        asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                            : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                            : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(b0), "r"(b1));
                        acc[n][0] += dA0 * dB0 * (float)d0;
                        acc[n][1] += dA0 * dB1 * (float)d1;
                        acc[n][2] += dA1 * dB0 * (float)d2;
                        acc[n][3] += dA1 * dB1 * (float)d3;
                    }
                }
            }
#endif  // PD_QMMA2_LDM
        }
        __syncthreads();
        if (f + S < TT) stage_w(f + S, s);
        asm volatile("cp.async.commit_group;");
        if (tk == nk - 1u) {   // slice done: flush (registers->global only)
            if (q_live) {
                const uint32_t row_base = (slice0 + sl_cur) * PD_QMMA2_RB;
                const uint32_t c0 = joff + 2u * t;
                #pragma unroll
                for (uint32_t n = 0; n < NW; ++n) {
                    const uint32_t r0 = row_base + i0 + n * 16u + g;
                    const uint32_t r8 = r0 + 8u;
                    #pragma unroll
                    for (uint32_t q = 0; q < 4u; ++q) {
                        const uint32_t r = (q & 2u) ? r8 : r0;
                        const uint32_t c = c0 + (q & 1u);
                        const unsigned int token = tok[c];
                        if (r >= embd || token == PD_MOE_PAD) continue;
                        const float w = topk_w[(size_t)token * n_active + slt[c]];
                        part[((size_t)token * n_active + slt[c]) * embd + r] = w * acc[n][q];
                    }
                }
            }
            #pragma unroll
            for (uint32_t n = 0; n < NW; ++n) {
                acc[n][0] = 0.f; acc[n][1] = 0.f; acc[n][2] = 0.f; acc[n][3] = 0.f;
            }
            ++sl_cur;
        }
    }
#else
    (void)down_data; (void)down_scale; (void)sorted_row; (void)sorted_slot;
    (void)block_expert; (void)topk_w; (void)fq; (void)fs; (void)part;
    (void)ff; (void)embd; (void)n_active;
#endif
}
#endif  // PD_QMMA2_DN_NT > 1

template <bool GELU>
static int pd_launch_qmma2_gu(const int8_t* gd, const __half* gs, const int8_t* ud,
                              const __half* us, const unsigned int* sr,
                              const unsigned int* be, const int8_t* xq, const float* xs,
                              int8_t* fq, float* fs, uint32_t in_dim, uint32_t ff,
                              uint32_t max_blocks, cudaStream_t stream) {
    constexpr uint32_t S = (uint32_t)PD_QMMA2_S;
    constexpr uint32_t smem = PD_QMMA2_SMEM_INT32 * 4u;
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute((const void*)pd_q8_0_moe_gate_up_mma2_kernel<S, GELU>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        attr = true;
    }
    dim3 grid(max_blocks, (ff + PD_QMMA2_RB - 1u) / PD_QMMA2_RB);
    pd_q8_0_moe_gate_up_mma2_kernel<S, GELU><<<grid, 256, smem, stream>>>(
        gd, gs, ud, us, sr, be, xq, xs, fq, fs, in_dim, ff);
    return pd_launch_status();
}

// v2 GEGLU gate_up (slot 492). Same signature as the shipped launcher; bm
// must be 32 (NotSupported otherwise -- the engine elects host-side).
PD_EXPORT
int pd_q8_0_moe_gate_up_mma2_geglu(const void* gate_data, const void* gate_scale,
                                   const void* up_data, const void* up_scale,
                                   const void* sorted_row, const void* block_expert,
                                   const void* xq, const void* xs, void* fq, void* fs,
                                   uint32_t in_dim, uint32_t ff, uint32_t max_blocks,
                                   uint32_t bm, void* stream) {
    if (ff == 0 || max_blocks == 0) return 0;
    if (bm != 32u) return cudaErrorNotSupported;
    if ((in_dim & 255u) != 0 || (ff & 31u) != 0) return cudaErrorInvalidValue;
    return pd_launch_qmma2_gu<true>(
        (const int8_t*)gate_data, (const __half*)gate_scale, (const int8_t*)up_data,
        (const __half*)up_scale, (const unsigned int*)sorted_row,
        (const unsigned int*)block_expert, (const int8_t*)xq, (const float*)xs,
        (int8_t*)fq, (float*)fs, in_dim, ff, max_blocks, (cudaStream_t)stream);
}

// v2 down (slot 493). ff % 64 (even n_blocks for the 4B async scale copies);
// gemma4-A4B's 704 qualifies.
PD_EXPORT
int pd_q8_0_moe_down_mma2(const void* down_data, const void* down_scale,
                          const void* sorted_row, const void* sorted_slot,
                          const void* block_expert, const void* topk_w, const void* fq,
                          const void* fs, void* part, uint32_t ff, uint32_t embd,
                          uint32_t n_active, uint32_t max_blocks, uint32_t bm,
                          void* stream) {
    if (embd == 0 || max_blocks == 0) return 0;
    if (bm != 32u) return cudaErrorNotSupported;
    if ((ff & 63u) != 0 || (embd & 31u) != 0) return cudaErrorInvalidValue;
    constexpr uint32_t S = (uint32_t)PD_QMMA2_DN_S;
#if PD_QMMA2_DN_NT > 1
    constexpr uint32_t NT = (uint32_t)PD_QMMA2_DN_NT;
    const uint32_t nk = (ff + 255u) >> 8;
    const uint32_t smem = (S * PD_QMMA2_DNNT_WSTAGE + nk * PD_MOEQ_Y_INT32) * 4u;
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute((const void*)pd_q8_0_moe_down_mma2nt_kernel<S, NT>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        attr = true;
    }
    dim3 grid(max_blocks, (embd + PD_QMMA2_RB * NT - 1u) / (PD_QMMA2_RB * NT));
    pd_q8_0_moe_down_mma2nt_kernel<S, NT><<<grid, 256, smem, (cudaStream_t)stream>>>(
#else
    constexpr uint32_t smem = PD_QMMA2_DN_SMEM_INT32 * 4u;
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute((const void*)pd_q8_0_moe_down_mma2_kernel<S>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        attr = true;
    }
    dim3 grid(max_blocks, (embd + PD_QMMA2_RB - 1u) / PD_QMMA2_RB);
    pd_q8_0_moe_down_mma2_kernel<S><<<grid, 256, smem, (cudaStream_t)stream>>>(
#endif
        (const int8_t*)down_data, (const __half*)down_scale,
        (const unsigned int*)sorted_row, (const unsigned int*)sorted_slot,
        (const unsigned int*)block_expert, (const float*)topk_w, (const int8_t*)fq,
        (const float*)fs, (float*)part, ff, embd, n_active);
    return pd_launch_status();
}

// hibatch P1-1: bf16-partials twin of the v2 down (part stored bf16 at the
// scatter; tail reads bf16, sums f32 in the same fixed order). Same
// signature as pd_q8_0_moe_down_mma2; lane-gated.
PD_EXPORT
int pd_q8_0_moe_down_mma2_pbf16(const void* down_data, const void* down_scale,
                                const void* sorted_row, const void* sorted_slot,
                                const void* block_expert, const void* topk_w,
                                const void* fq, const void* fs, void* part,
                                uint32_t ff, uint32_t embd, uint32_t n_active,
                                uint32_t max_blocks, uint32_t bm, void* stream) {
    if (embd == 0 || max_blocks == 0) return 0;
    if (bm != 32u) return cudaErrorNotSupported;
    if ((ff & 63u) != 0 || (embd & 31u) != 0) return cudaErrorInvalidValue;
    constexpr uint32_t S = (uint32_t)PD_QMMA2_DN_S;
    constexpr uint32_t smem = PD_QMMA2_DN_SMEM_INT32 * 4u;
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute((const void*)pd_q8_0_moe_down_mma2_kernel<S, true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        attr = true;
    }
    dim3 grid(max_blocks, (embd + PD_QMMA2_RB - 1u) / PD_QMMA2_RB);
    pd_q8_0_moe_down_mma2_kernel<S, true><<<grid, 256, smem, (cudaStream_t)stream>>>(
        (const int8_t*)down_data, (const __half*)down_scale,
        (const unsigned int*)sorted_row, (const unsigned int*)sorted_slot,
        (const unsigned int*)block_expert, (const float*)topk_w, (const int8_t*)fq,
        (const float*)fs, (float*)part, ff, embd, n_active);
    return pd_launch_status();
}

// P1 dn64 producer: mma2g twin quantizing the GEGLU output with PER-64
// scale groups (fs at ff/64 stride). Same signature as the mma2g launcher;
// ff % 64 required (full 64-row windows; gemma4-A4B's 704 qualifies).
PD_EXPORT
int pd_q8_0_moe_gate_up_mma2g_y64_geglu(const void* gate_data, const void* gate_scale,
                                        const void* up_data, const void* up_scale,
                                        const void* sorted_row, const void* block_expert,
                                        const void* xq, const void* xs, void* fq, void* fs,
                                        uint32_t in_dim, uint32_t ff, uint32_t max_blocks,
                                        uint32_t bm, void* stream) {
    if (ff == 0 || max_blocks == 0) return 0;
    if (bm != 32u) return cudaErrorNotSupported;
    if ((in_dim & 255u) != 0 || (ff & 63u) != 0) return cudaErrorInvalidValue;
    constexpr uint32_t S = (uint32_t)PD_QMMA2_S;
    constexpr uint32_t smem = PD_QMMA2_SMEM_INT32 * 4u;
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute((const void*)pd_q8_0_moe_gate_up_mma2g_kernel<S, true, true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        attr = true;
    }
    dim3 grid(max_blocks, (ff + PD_QMMA2_RB - 1u) / PD_QMMA2_RB);
    pd_q8_0_moe_gate_up_mma2g_kernel<S, true, true><<<grid, 256, smem, (cudaStream_t)stream>>>(
        (const int8_t*)gate_data, (const __half*)gate_scale, (const int8_t*)up_data,
        (const __half*)up_scale, (const unsigned int*)sorted_row,
        (const unsigned int*)block_expert, (const int8_t*)xq, (const float*)xs,
        (int8_t*)fq, (float*)fs, in_dim, ff);
    return pd_launch_status();
}

// P1 dn64 consumer: v2 down consuming PER-64 Y scales (fs at ff/64 stride,
// pair-grouped fold); trailing pbf16 flag selects the bf16 partials store
// (P1-1 composition). Plain-ring LDM form only - table entry is nullptr on
// NT/YSYNC/no-LDM builds so the lane gates itself off.
PD_EXPORT
int pd_q8_0_moe_down_mma2_fs64(const void* down_data, const void* down_scale,
                               const void* sorted_row, const void* sorted_slot,
                               const void* block_expert, const void* topk_w,
                               const void* fq, const void* fs, void* part,
                               uint32_t ff, uint32_t embd, uint32_t n_active,
                               uint32_t max_blocks, uint32_t bm, uint32_t pbf16,
                               void* stream) {
#if PD_QMMA2_DN_NT > 1 || PD_QMMA2_YSYNC || !PD_QMMA2_LDM
    (void)down_data; (void)down_scale; (void)sorted_row; (void)sorted_slot;
    (void)block_expert; (void)topk_w; (void)fq; (void)fs; (void)part; (void)ff;
    (void)embd; (void)n_active; (void)max_blocks; (void)bm; (void)pbf16; (void)stream;
    return cudaErrorNotSupported;
#else
    if (embd == 0 || max_blocks == 0) return 0;
    if (bm != 32u) return cudaErrorNotSupported;
    if ((ff & 63u) != 0 || (embd & 31u) != 0) return cudaErrorInvalidValue;
    constexpr uint32_t S = (uint32_t)PD_QMMA2_DN_S;
    constexpr uint32_t smem = PD_QMMA2_DN_SMEM_INT32 * 4u;
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute((const void*)pd_q8_0_moe_down_mma2_kernel<S, false, true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        cudaFuncSetAttribute((const void*)pd_q8_0_moe_down_mma2_kernel<S, true, true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        attr = true;
    }
    dim3 grid(max_blocks, (embd + PD_QMMA2_RB - 1u) / PD_QMMA2_RB);
    if (pbf16)
        pd_q8_0_moe_down_mma2_kernel<S, true, true><<<grid, 256, smem, (cudaStream_t)stream>>>(
            (const int8_t*)down_data, (const __half*)down_scale,
            (const unsigned int*)sorted_row, (const unsigned int*)sorted_slot,
            (const unsigned int*)block_expert, (const float*)topk_w, (const int8_t*)fq,
            (const float*)fs, (float*)part, ff, embd, n_active);
    else
        pd_q8_0_moe_down_mma2_kernel<S, false, true><<<grid, 256, smem, (cudaStream_t)stream>>>(
            (const int8_t*)down_data, (const __half*)down_scale,
            (const unsigned int*)sorted_row, (const unsigned int*)sorted_slot,
            (const unsigned int*)block_expert, (const float*)topk_w, (const int8_t*)fq,
            (const float*)fs, (float*)part, ff, embd, n_active);
    return pd_launch_status();
#endif
}

// ---- v5 gate_up: the small-CTA geometry port (slot 488) --------------------
// A tuned fused_moe at decode-M on this die and shape (E=128 N=704) runs
// BM=16 token tiles x BN=64..128 out tiles x BK=128 with
// 4 warps and 2-3 stages - small CTAs at 5-ish/SM instead of our 2-3 fat
// ones. This port keeps the exact v2 math (same m16n8k32 per q8 block, same
// ascending block fold per output, same GEGLU/quantize epilogue) on that
// geometry, and needs no layout change: each bm32 CSR block is viewed as
// two 16-token m-tiles (the live-prefix contract makes the second tile
// all-PAD at decode routing, and it exits), and each CTA walks one 64-row
// slice of gate plus the same slice of up, so the GEGLU pairing stays
// in-warp. fq/fs land in the same bm32-contiguous rows the v2 down reads.
//   CTA: 128 thr / 4 warps; warp w owns out rows [n64*64 + w*16, +16) of
//   both mats x all 16 tokens. K ticks of 128 (4 q8 blocks), S-stage ring.
//   smem/stage: Wg 64x36 + Wu 64x36 + Y 16x36 int32 = 20.8KB; S=2 -> 5
//   CTAs/SM, S=3 -> 3.
#ifndef PD_QMMA3_S
#define PD_QMMA3_S 2
#endif
#ifndef PD_QMMA3_OCC
#define PD_QMMA3_OCC 5
#endif
#define PD_QMMA3_WK 36u   // 32 data int32 (4 q8 blocks) + 2 scale + 2 pad
#define PD_QMMA3_STAGE_INT32 ((64u + 64u + 16u) * PD_QMMA3_WK)

// Stage one 128-K tick: one mat's 64-row W slice into wt, Y for 16 tokens
// into yt (with_y). Guards zero-fill exactly like the v2 stage.
__device__ __forceinline__ void pd_qmma3_stage(
    int* __restrict__ wt, int* __restrict__ yt, const int8_t* __restrict__ wd,
    const __half* __restrict__ ws, const int* __restrict__ xq32,
    const float* __restrict__ xs, const unsigned int* __restrict__ tok,
    size_t wrow0, uint32_t row_base, uint32_t out_dim, uint32_t in_dim,
    uint32_t kt, uint32_t tid, bool with_y) {
#if PD_MMA_OK
    const uint32_t n_blocks = in_dim >> 5, n_k32 = in_dim >> 2;
    // W data: 64 rows x 4 blocks x 2 16B halves = 512 copies over 128 thr
    #pragma unroll
    for (uint32_t it = 0; it < 4u; ++it) {
        const uint32_t i = it * 128u + tid;
        const uint32_t row = i >> 3, half = i & 7u;
        const uint32_t b = half >> 1, h16 = half & 1u, gb = kt * 4u + b;
        const bool ok = gb < n_blocks && (row_base + row) < out_dim;
        pd_cp_async16(wt + row * PD_QMMA3_WK + b * 8u + h16 * 4u,
                      wd + ((wrow0 + row) * (size_t)in_dim) + (ok ? gb : 0u) * 32u
                          + h16 * 16u,
                      ok);
    }
    // W scales: 4 f16 per row = two 4B copies (gb even, n_blocks even)
    {
        const uint32_t row = tid >> 1, c = tid & 1u, gb = kt * 4u + c * 2u;
        const bool ok = gb < n_blocks && (row_base + row) < out_dim;
        pd_mma_cpa4p((__half*)(wt + row * PD_QMMA3_WK + 32u) + c * 2u,
                     ws + (wrow0 + row) * n_blocks + (ok ? gb : 0u), ok);
    }
    if (with_y) {
        // Y data: 16 cols x 8 16B chunks
        {
            const uint32_t c = tid >> 3, ch = tid & 7u;
            const unsigned int r = tok[c];
            const uint32_t gk = kt * 32u + ch * 4u;
            const bool ok = r != PD_MOE_PAD && gk < n_k32;
            pd_cp_async16(yt + c * PD_QMMA3_WK + ch * 4u,
                          xq32 + (ok ? ((size_t)r * n_k32 + gk) : 0u), ok);
        }
        // Y scales: 4 f32 per col
        {
            const uint32_t c = tid >> 3, b = tid & 7u;
            if (b < 4u) {
                const uint32_t gb = kt * 4u + b;
                const unsigned int r = tok[c];
                const bool ok = r != PD_MOE_PAD && gb < n_blocks;
                pd_mma_cpa4p((float*)(yt + c * PD_QMMA3_WK + 32u) + b,
                             xs + (ok ? ((size_t)r * n_blocks + gb) : 0u), ok);
            }
        }
    }
#endif
}

template <uint32_t S, bool GELU>
__global__ void __launch_bounds__(128, PD_QMMA3_OCC) pd_q8_0_moe_gate_up_mma3_kernel(
    const int8_t* __restrict__ gate_data, const __half* __restrict__ gate_scale,
    const int8_t* __restrict__ up_data, const __half* __restrict__ up_scale,
    const unsigned int* __restrict__ sorted_row, const unsigned int* __restrict__ block_expert,
    const int8_t* __restrict__ xq, const float* __restrict__ xs,
    int8_t* __restrict__ fq, float* __restrict__ fs, uint32_t in_dim, uint32_t ff) {
#if PD_MMA_OK
    const uint32_t mt = blockIdx.x;               // 16-token tile (2 per bm32 block)
    const uint32_t blk = mt >> 1, half16 = mt & 1u;
    const uint32_t e = block_expert[blk];
    if (e == PD_MOE_PAD) return;
    const uint32_t row_base = blockIdx.y * 64u;   // out-row slice (both mats)
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, t = lane & 3u;
    const uint32_t nk = (in_dim + 127u) >> 7;

    extern __shared__ int pd_qmma3_sh[];
    __shared__ unsigned int tok[16];
    __shared__ uint32_t nlive_sh;
    if (tid < 16u) tok[tid] = sorted_row[(size_t)blk * 32u + half16 * 16u + tid];
    __syncthreads();
    if (warp == 0) {
        const bool pad = (lane < 16u) ? tok[lane] == PD_MOE_PAD : true;
        const uint32_t m = __ballot_sync(0xffffffffu, pad);
        if (lane == 0) nlive_sh = (uint32_t)(__ffs((int)m) - 1);
    }
    __syncthreads();
    if (nlive_sh == 0) return;                    // all-PAD tile (the 2nd half, mostly)

    const size_t wrow0 = (size_t)e * ff + row_base;
    auto sb = [&](uint32_t s2) { return pd_qmma3_sh + s2 * PD_QMMA3_STAGE_INT32; };
    auto tick_stage = [&](uint32_t tk, uint32_t s2) {
        int* base = sb(s2);
        int* yt = base + 128u * PD_QMMA3_WK;
        pd_qmma3_stage(base, yt, gate_data, gate_scale, (const int*)xq, xs, tok,
                       wrow0, row_base, ff, in_dim, tk, tid, true);
        pd_qmma3_stage(base + 64u * PD_QMMA3_WK, yt, up_data, up_scale,
                       (const int*)xq, xs, tok, wrow0, row_base, ff, in_dim, tk,
                       tid, false);
    };
    #pragma unroll
    for (uint32_t s2 = 0; s2 < S; ++s2) {
        if (s2 < nk) tick_stage(s2, s2);
        asm volatile("cp.async.commit_group;");
    }

    // warp w: out rows [w*16, +16) of both mats x 16 tokens.
    // acc[mat][n(=2 token n8-quarters)][4]
    float acc_g[2][4] = {}, acc_u[2][4] = {};
    const uint32_t wr0 = warp * 16u;
    for (uint32_t tk = 0; tk < nk; ++tk) {
        const uint32_t s2 = tk % S;
        pd_mma_cpa_waitN<(int)S - 1>();
        __syncthreads();
        {
            int* base = sb(s2);
            int* tile_y = base + 128u * PD_QMMA3_WK;
            #pragma unroll
            for (uint32_t mat = 0; mat < 2u; ++mat) {
                int* tw = base + mat * 64u * PD_QMMA3_WK;
                float(*acc)[4] = mat ? acc_u : acc_g;
                #pragma unroll
                for (uint32_t bb = 0; bb < 4u; ++bb) {
                    const uint32_t ko = bb * 8u;
                    #pragma unroll
                    for (uint32_t n = 0; n < 2u; ++n) {
                        const uint32_t jb = n * 8u;
                        const int b0 = tile_y[(jb + g) * PD_QMMA3_WK + ko + t];
                        const int b1 = tile_y[(jb + g) * PD_QMMA3_WK + ko + 4u + t];
                        const float dB0 =
                            ((const float*)tile_y)[(jb + 2u * t) * PD_QMMA3_WK + 32u + bb];
                        const float dB1 =
                            ((const float*)tile_y)[(jb + 2u * t + 1u) * PD_QMMA3_WK + 32u + bb];
                        const uint32_t r0 = (wr0 + g) * PD_QMMA3_WK;
                        const uint32_t r8 = (wr0 + 8u + g) * PD_QMMA3_WK;
                        const int A0 = tw[r0 + bb * 8u + t];
                        const int A2 = tw[r0 + bb * 8u + 4u + t];
                        const int A1 = tw[r8 + bb * 8u + t];
                        const int A3 = tw[r8 + bb * 8u + 4u + t];
                        const float dA0 = __half2float(((const __half*)(tw + r0 + 32u))[bb]);
                        const float dA1 = __half2float(((const __half*)(tw + r8 + 32u))[bb]);
                        int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                        asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                            : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                            : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(b0), "r"(b1));
                        acc[n][0] += dA0 * dB0 * (float)d0;
                        acc[n][1] += dA0 * dB1 * (float)d1;
                        acc[n][2] += dA1 * dB0 * (float)d2;
                        acc[n][3] += dA1 * dB1 * (float)d3;
                    }
                }
            }
        }
        __syncthreads();
        if (tk + S < nk) tick_stage(tk + S, s2);
        asm volatile("cp.async.commit_group;");
    }

    // Epilogue: fused = GELU(g)*u per (row, token), then the per-32-out-row
    // q8 quantize. A warp holds 16 rows; the 32-row block spans warp pairs,
    // so outs bounce through a [64 rows x 16 cols] f32 plane (ring smem is
    // dead) and warps 0/2 run the v2 amax/quantize shape over rows
    // [w*32, +32). Values, tree and stores match the v2 epilogue exactly
    // (same GELU expression, same shfl amax over the same 4 values, same
    // store addressing into the bm32-contiguous fq rows).
    float* plane = (float*)pd_qmma3_sh;   // 64*16 f32 = 4KB
    {
        #pragma unroll
        for (uint32_t n = 0; n < 2u; ++n) {
            #pragma unroll
            for (uint32_t q = 0; q < 4u; ++q) {
                const uint32_t r16 = (q & 2u) ? (8u + g) : g;   // row within warp tile
                const uint32_t c = n * 8u + 2u * t + (q & 1u);  // token 0..15
                const uint32_t r = row_base + wr0 + r16;
                const bool pad = tok[c] == PD_MOE_PAD;
                float out = 0.f;
                if (!pad && r < ff) {
                    const float gv = acc_g[n][q];
                    const float uv = acc_u[n][q];
                    out = GELU
                        ? 0.5f * gv
                              * (1.0f
                                 + tanhf(0.79788456080286535587989211986876f * gv
                                         * (1.0f + 0.044715f * gv * gv)))
                              * uv
                        : (gv / (1.0f + __expf(-gv))) * uv;
                }
                plane[(wr0 + r16) * 16u + c] = out;
            }
        }
    }
    __syncthreads();
    if ((warp & 1u) == 0) {
        const uint32_t rb = row_base + (warp >> 1) * 32u;   // 32-row quantize block
        const uint32_t pr0 = (warp >> 1) * 32u;
        const uint32_t n_sb = ff >> 5;
        #pragma unroll
        for (uint32_t qc = 0; qc < 4u; ++qc) {              // 16 tokens / (4 lanes t) ... c = 4*qc? no:
            const uint32_t c = qc * 4u + t;                 // token 0..15 per lane t
            float sw[4];
            #pragma unroll
            for (uint32_t v = 0; v < 4u; ++v)
                sw[v] = plane[(pr0 + (v >> 1) * 16u + (v & 1u) * 8u + g) * 16u + c];
            float a = fmaxf(fmaxf(fabsf(sw[0]), fabsf(sw[1])), fmaxf(fabsf(sw[2]), fabsf(sw[3])));
            #pragma unroll
            for (uint32_t o = 4; o <= 16u; o <<= 1)
                a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
            const float scl = a * (1.0f / 127.0f);
            const float invs = scl > 0.f ? 1.0f / scl : 0.f;
            const size_t row = (size_t)blk * 32u + half16 * 16u + c;
            if (rb < ff && tok[c] != PD_MOE_PAD) {
                #pragma unroll
                for (uint32_t v = 0; v < 4u; ++v) {
                    const uint32_t r = rb + (v >> 1) * 16u + (v & 1u) * 8u + g;
                    int qi = __float2int_rn(sw[v] * invs);
                    qi = qi < -127 ? -127 : (qi > 127 ? 127 : qi);
                    fq[row * ff + r] = (int8_t)qi;
                }
                if (g == 0) fs[row * n_sb + (rb >> 5)] = scl;
            } else if (rb < ff && tok[c] == PD_MOE_PAD) {
                // keep the v2 contract: pad cols inside a live tile get
                // exact zeros (the plane already holds 0 for them)
                #pragma unroll
                for (uint32_t v = 0; v < 4u; ++v) {
                    const uint32_t r = rb + (v >> 1) * 16u + (v & 1u) * 8u + g;
                    fq[row * ff + r] = 0;
                }
                if (g == 0) fs[row * n_sb + (rb >> 5)] = 0.f;
            }
        }
    }
#else
    (void)gate_data; (void)gate_scale; (void)up_data; (void)up_scale; (void)sorted_row;
    (void)block_expert; (void)xq; (void)xs; (void)fq; (void)fs; (void)in_dim; (void)ff;
#endif
}

// v5 launcher (slot 488): grid (2*max_blocks m-tiles, ff/64 both-mat
// slices); bm must be 32 (the m-tiles VIEW the bm32 CSR).
PD_EXPORT
int pd_q8_0_moe_gate_up_mma3_geglu(const void* gate_data, const void* gate_scale,
                                   const void* up_data, const void* up_scale,
                                   const void* sorted_row, const void* block_expert,
                                   const void* xq, const void* xs, void* fq, void* fs,
                                   uint32_t in_dim, uint32_t ff, uint32_t max_blocks,
                                   uint32_t bm, void* stream) {
    if (ff == 0 || max_blocks == 0) return 0;
    if (bm != 32u) return cudaErrorNotSupported;
    if ((in_dim & 255u) != 0 || (ff & 63u) != 0) return cudaErrorInvalidValue;
    constexpr uint32_t S = (uint32_t)PD_QMMA3_S;
    constexpr uint32_t smem = S * PD_QMMA3_STAGE_INT32 * 4u;
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute((const void*)pd_q8_0_moe_gate_up_mma3_kernel<S, true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        attr = true;
    }
    dim3 grid(2u * max_blocks, ff / 64u);
    pd_q8_0_moe_gate_up_mma3_kernel<S, true><<<grid, 128, smem, (cudaStream_t)stream>>>(
        (const int8_t*)gate_data, (const __half*)gate_scale, (const int8_t*)up_data,
        (const __half*)up_scale, (const unsigned int*)sorted_row,
        (const unsigned int*)block_expert, (const int8_t*)xq, (const float*)xs,
        (int8_t*)fq, (float*)fs, in_dim, ff);
    return pd_launch_status();
}

// Shared-expert scalar gate fold: dst[b][i] += sigmoid(dot(x[b], w)) * src[b][i]
// (qwen3.6-A3B: the shared expert output rides a per-token sigmoid gate,
// w = ffn_gate_inp_shexp [n_in]). Every block recomputes the tiny dot for its
// token (n_in floats from L2) - cheaper than a separate kernel + readback.
__global__ void pd_shexp_gate_add_kernel(float* __restrict__ dst,
                                         const float* __restrict__ src,
                                         const float* __restrict__ x,
                                         const float* __restrict__ w, uint32_t n_out,
                                         uint32_t n_in) {
    const uint32_t b = blockIdx.y;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    __shared__ float red[8];
    __shared__ float s_gate;
    float acc = 0.0f;
    const float* xb = x + (size_t)b * n_in;
    for (uint32_t i = tid; i < n_in; i += nth) acc += xb[i] * w[i];
    const uint32_t lane = tid & 31u, warp = tid >> 5, nwarps = (nth + 31u) >> 5;
    for (uint32_t s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, s);
    if (lane == 0) red[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        float d = 0.0f;
        for (uint32_t w2 = 0; w2 < nwarps; ++w2) d += red[w2];
        s_gate = 1.0f / (1.0f + __expf(-d));
    }
    __syncthreads();
    const float g = s_gate;
    for (uint32_t i = blockIdx.x * nth + tid; i < n_out; i += gridDim.x * nth)
        dst[(size_t)b * n_out + i] += g * src[(size_t)b * n_out + i];
}

PD_EXPORT
int pd_shexp_gate_add(void* dst, const void* src, const void* x, const void* w,
                      uint32_t n_out, uint32_t n_in, uint32_t batch, void* stream) {
    if (n_out == 0 || batch == 0) return 0;
    dim3 grid((n_out + 255u) / 256u, batch);
    pd_shexp_gate_add_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(
        (float*)dst, (const float*)src, (const float*)x, (const float*)w, n_out, n_in);
    return pd_launch_status();
}


// Fused residual-add + RMSNorm for the decode/serving paths: x += proj
// (written back), out = rmsnorm(x, w). Bit-identical to the add-then-norm
// two-kernel sequence (same add, same square-sum order over the summed
// values) - it just removes one graph-node drain per layer (the b=1 tail
// measured ~2-4 us per small launch; the mid-layer add+norm pair is 2 of
// the ~9 tail launches on every DeltaNet layer).
// `pscale` folds a residual MULTIPLIER into the fused add (granite's
// `residual_multiplier`). It exists because granite was the one
// family that could not use this kernel: its residual is `x += res_s * proj`,
// so it paid pd_scale_add + pd_rmsnorm_batch as two launches -- ~80 extra
// launches per decode tick on the 8b, measured at 82 us/token of pure
// scale_add. fmaf(1, p, v) is exactly p + v, so `pd_add_rmsnorm_batch` passing
// 1.0f keeps every existing caller bit-identical, and the granite chain is
// bit-identical too because pd_scale_add_kernel's `x[i] += w * y[i]`
// contracts to the same fma.
__global__ void pd_add_rmsnorm_batch_kernel(float* __restrict__ x,
                                            const float* __restrict__ proj,
                                            const float* __restrict__ w,
                                            float* __restrict__ out, uint32_t n,
                                            float eps, float pscale) {
    // cascade (laguna chain): proj is the predecessor
    // wo-GEMV's output. No-op under plain launches / pre-sm90.
    PD_PDL_ARM();
    const uint32_t b = blockIdx.x;
    float* xb = x + (size_t)b * n;
    const float* pb = proj + (size_t)b * n;
    float* ob = out + (size_t)b * n;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    __shared__ float wsum[32];
    __shared__ float s_inv;
    float acc = 0.0f;
    const bool vec = (n & 3u) == 0;
    if (vec) {
        const uint32_t n4 = n >> 2;
        float4* x4 = reinterpret_cast<float4*>(xb);
        const float4* p4 = reinterpret_cast<const float4*>(pb);
        for (uint32_t i = tid; i < n4; i += nth) {
            float4 v = x4[i];
            const float4 pv = p4[i];
            v.x = fmaf(pscale, pv.x, v.x); v.y = fmaf(pscale, pv.y, v.y);
            v.z = fmaf(pscale, pv.z, v.z); v.w = fmaf(pscale, pv.w, v.w);
            x4[i] = v;
            acc += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
    } else {
        for (uint32_t i = tid; i < n; i += nth) {
            const float v = fmaf(pscale, pb[i], xb[i]);
            xb[i] = v;
            acc += v * v;
        }
    }
    for (uint32_t sh = 16; sh > 0; sh >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, sh);
    const uint32_t warp = tid >> 5, lane = tid & 31u;
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
    for (uint32_t i = tid; i < n; i += nth) ob[i] = xb[i] * inv * w[i];
}

// pd_add_rmsnorm_scaled_batch's twin that consumes RAW split-K partials in
// place of a pre-reduced `proj` (the nvf4 reduce-fold): the
// predecessor nvf4 GEMM wrote `nz` partial slices (stride batch*n) instead of
// calling pd_nvf4_sk_reduce, so this folds them with the same fixed-order sum
// and `scale2` the reduce would (pd_nvf4_sk_reduce's own math) before the
// residual fmaf. That makes `proj_i = (sum_k part[k*batch*n + b*n + i]) *
// scale2 (+bias)` bit-identical to the reduce's y[i], and the float4 residual
// fmaf + square-sum below is pd_add_rmsnorm_batch_kernel's path verbatim ->
// BIT-IDENTICAL to reduce-then-add_rmsnorm_scaled. Saves the reduce launch and
// its y round trip per GEMM. `bias` is null for granite's projections.
__global__ void pd_add_rmsnorm_scaled_from_parts_kernel(
    float* __restrict__ x, const float* __restrict__ part,
    const float* __restrict__ w, float* __restrict__ out,
    const float* __restrict__ bias, uint32_t n, float eps, float pscale,
    float scale2, uint32_t batch, uint32_t nz) {
    PD_PDL_ARM();
    const uint32_t b = blockIdx.x;
    float* xb = x + (size_t)b * n;
    const float* pb0 = part + (size_t)b * n;   // partial slice 0 for this row
    float* ob = out + (size_t)b * n;
    const size_t partN = (size_t)batch * n;    // stride between partial slices
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    __shared__ float wsum[32];
    __shared__ float s_inv;
    float acc = 0.0f;
    const bool vec = (n & 3u) == 0;
    if (vec) {
        const uint32_t n4 = n >> 2;
        float4* x4 = reinterpret_cast<float4*>(xb);
        const float4* p4 = reinterpret_cast<const float4*>(pb0);
        const float4* b4 = bias ? reinterpret_cast<const float4*>(bias) : nullptr;
        const size_t partN4 = partN >> 2;
        for (uint32_t i = tid; i < n4; i += nth) {
            float4 s = p4[i];
            for (uint32_t k = 1; k < nz; ++k) {
                const float4 pk = p4[(size_t)k * partN4 + i];
                s.x += pk.x; s.y += pk.y; s.z += pk.z; s.w += pk.w;
            }
            float4 pv;
            pv.x = s.x * scale2; pv.y = s.y * scale2;
            pv.z = s.z * scale2; pv.w = s.w * scale2;
            if (b4) { const float4 bb = b4[i]; pv.x += bb.x; pv.y += bb.y; pv.z += bb.z; pv.w += bb.w; }
            float4 v = x4[i];
            v.x = fmaf(pscale, pv.x, v.x); v.y = fmaf(pscale, pv.y, v.y);
            v.z = fmaf(pscale, pv.z, v.z); v.w = fmaf(pscale, pv.w, v.w);
            x4[i] = v;
            acc += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
    } else {
        for (uint32_t i = tid; i < n; i += nth) {
            float s = pb0[i];
            for (uint32_t k = 1; k < nz; ++k) s += pb0[(size_t)k * partN + i];
            float pv = s * scale2;
            if (bias) pv += bias[i];
            const float v = fmaf(pscale, pv, xb[i]);
            xb[i] = v;
            acc += v * v;
        }
    }
    for (uint32_t sh = 16; sh > 0; sh >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, sh);
    const uint32_t warp = tid >> 5, lane = tid & 31u;
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
    for (uint32_t i = tid; i < n; i += nth) ob[i] = xb[i] * inv * w[i];
}



// ===================== v3t: TMA-staged v2 ring =============================
// The v2 pair verbatim - same ring, same mma, same fold order, same
// epilogues (=> bitwise) - with the W stream staged by cp.async.bulk.tensor
// (SW128 boxes) instead of 256-thread cp.async, and TEMPLATED on RB:
// RB=64 mirrors v2's tile; RB=32 halves the stage (GU 27.6KB/stage) so 4
// CTAs co-reside (the profiler verdict on both v2 and v3t@64: latency-bound at
// 16 warps/SM, No-Eligible ~60%). Runtime select: PADDOCK_Q2T_RB=32 (default
// 64). SW128 TMA dst atoms must be 1024B-aligned (stage strides are padded
// to 1KB - the unaligned first cut produced wholesale-wrong output), and
// definition-level guards must be host-visible (PD_MMA_OK is arch-gated;
// stripping the decls from the host pass breaks the cudafe stubs).
#if PD_QMMA2_ILV && PD_QMMA2_LDM && !PD_QMMA2_YSYNC

#define PD_Q2T_Y_INT32 (32u * PD_MMQ_XK)
__host__ __device__ constexpr uint32_t pd_q2t_align1k(uint32_t b) {
    return (b + 1023u) & ~1023u;
}
__host__ __device__ constexpr uint32_t pd_q2t_gu_stride(uint32_t rb) {
    return pd_q2t_align1k(2u * rb * 256u + PD_Q2T_Y_INT32 * 4u + 2u * rb * 16u);
}
__host__ __device__ constexpr uint32_t pd_q2t_dn_stride(uint32_t rb) {
    return pd_q2t_align1k(rb * 256u + PD_Q2T_Y_INT32 * 4u + rb * 16u);
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && PD_MMA_OK
#define PD_Q2T_DEV 1
#else
#define PD_Q2T_DEV 0
#endif

#if PD_Q2T_DEV
// one 128B-swizzled W tick (2 subtiles of RB rows) into dst, rows at wrow0
template <uint32_t RB>
__device__ __forceinline__ void pd_q2t_wtick(const void* map, uint32_t dst,
                                             uint32_t ck, uint32_t wrow0,
                                             uint32_t mbar) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];" ::"r"(dst),
        "l"(map), "r"((int)ck), "r"((int)wrow0), "r"(mbar)
        : "memory");
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];" ::"r"(dst + RB * 128u),
        "l"(map), "r"((int)(ck + 128u)), "r"((int)wrow0), "r"(mbar)
        : "memory");
}
__device__ __forceinline__ void pd_q2t_wait(uint32_t mbar, uint32_t parity) {
    asm volatile(
        "{.reg .pred p; W%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;"
        " @!p bra W%=;}" ::"r"(mbar), "r"(parity));
}
// v2's Y stage (data + f32 scales), verbatim guards/layout
__device__ __forceinline__ void pd_q2t_ystage(int* yt, const int* xq32, const float* xs,
                                              const unsigned int* rows, uint32_t in_dim,
                                              uint32_t kt, uint32_t tid) {
    const uint32_t n_blocks = in_dim / 32u, n_k32 = in_dim / 4u;
    #pragma unroll
    for (uint32_t it = 0; it < 2u; ++it) {
        const uint32_t i = it * 256u + tid;
        const uint32_t c = i / 16u, ch = i & 15u;
        const unsigned int r = rows[c];
        const uint32_t gk = kt * 64u + ch * 4u;
        const bool ok = r != PD_MOE_PAD && gk < n_k32;
        pd_cp_async16(yt + c * PD_MMQ_XK + ch * 4u,
                      xq32 + (ok ? ((size_t)r * n_k32 + gk) : 0u), ok);
    }
    {
        const uint32_t c = tid / 8u, b = tid & 7u, gb = kt * 8u + b;
        const unsigned int r = rows[c];
        const bool ok = r != PD_MOE_PAD && gb < n_blocks;
        pd_mma_cpa4p((float*)(yt + c * PD_MMQ_XK + 64u) + b,
                     xs + (ok ? ((size_t)r * n_blocks + gb) : 0u), ok);
    }
}
// v2's W-scale stage into the dedicated plane (8 halfs per row, RB rows)
template <uint32_t RB>
__device__ __forceinline__ void pd_q2t_wsc(__half* dst, const __half* ws, size_t wrow0,
                                           uint32_t row_base, uint32_t out_dim,
                                           uint32_t n_blocks, uint32_t kt, uint32_t tid) {
    if (tid < RB * 4u) {
        const uint32_t row = tid / 4u, c = tid & 3u, gb = kt * 8u + c * 2u;
        const bool ok = gb < n_blocks && (row_base + row) < out_dim;
        pd_mma_cpa4p(dst + row * 8u + c * 2u, ws + (wrow0 + row) * n_blocks + (ok ? gb : 0u),
                     ok);
    }
}
#endif  // PD_Q2T_DEV

template <uint32_t S, bool GELU, uint32_t RB>
__global__ void __launch_bounds__(256, RB == 32u ? 4 : 2) pd_q8_0_moe_gate_up_mma2t_kernel(
    const __grid_constant__ PdTmap gmap, const __grid_constant__ PdTmap umap,
    const __half* __restrict__ gate_scale, const __half* __restrict__ up_scale,
    const unsigned int* __restrict__ sorted_row, const unsigned int* __restrict__ block_expert,
    const int8_t* __restrict__ xq, const float* __restrict__ xs,
    int8_t* __restrict__ fq, float* __restrict__ fs, uint32_t in_dim, uint32_t ff) {
#if PD_Q2T_DEV
    constexpr uint32_t BM = 32u;
    constexpr uint32_t NW = RB / 32u;
    constexpr uint32_t WPL = RB * 256u;
    constexpr uint32_t STRIDE = pd_q2t_gu_stride(RB);
    constexpr uint32_t YOFF = 2u * WPL;
    constexpr uint32_t SCOFF = YOFF + PD_Q2T_Y_INT32 * 4u;
    const uint32_t blk = blockIdx.x;
    const uint32_t e = block_expert[blk];
    if (e == PD_MOE_PAD) return;
    const uint32_t row_base = blockIdx.y * RB;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid / 32u;
    const uint32_t g = lane / 4u, t = lane & 3u;
    const uint32_t i0 = (warp & 1u) * 16u * NW;
    const uint32_t joff = (warp / 2u) * 8u;
    const uint32_t nk = (in_dim + 255u) / 256u;
    const uint32_t n_blocks = in_dim / 32u;

    extern __shared__ __align__(128) char pd_q2t_sh[];
    __shared__ __align__(8) uint64_t pd_q2t_mb[2];
    __shared__ unsigned int tok[BM];
    __shared__ uint32_t nlive_sh;
    for (uint32_t i = tid; i < BM; i += 256u) tok[i] = sorted_row[(size_t)blk * BM + i];
    const uint32_t mb0 = (uint32_t)__cvta_generic_to_shared(pd_q2t_mb);
    if (tid == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(mb0));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(mb0 + 8u));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();
    if (warp == 0) {
        const uint32_t m = __ballot_sync(0xffffffffu, tok[lane] == PD_MOE_PAD);
        if (lane == 0) nlive_sh = m ? (uint32_t)(__ffs((int)m) - 1) : BM;
    }

    const size_t wrow0 = (size_t)e * ff + row_base;
    const uint32_t T = nk;
    auto stage = [&](uint32_t s) { return pd_q2t_sh + s * STRIDE; };
    auto tick_stage = [&](uint32_t tk, uint32_t s) {
        char* sb = stage(s);
        if (tid == 0) {
            const uint32_t m = mb0 + s * 8u;
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(m),
                         "r"(4u * RB * 128u));
            const uint32_t ck = tk * 256u;
            pd_q2t_wtick<RB>(&gmap, (uint32_t)__cvta_generic_to_shared(sb), ck,
                         (uint32_t)wrow0, m);
            pd_q2t_wtick<RB>(&umap, (uint32_t)__cvta_generic_to_shared(sb + WPL), ck,
                         (uint32_t)wrow0, m);
        }
        pd_q2t_ystage((int*)(sb + YOFF), (const int*)xq, xs, tok, in_dim, tk, tid);
        pd_q2t_wsc<RB>((__half*)(sb + SCOFF), gate_scale, wrow0, row_base, ff,
                   n_blocks, tk, tid);
        pd_q2t_wsc<RB>((__half*)(sb + SCOFF + RB * 16u), up_scale, wrow0, row_base, ff,
                   n_blocks, tk, tid);
    };
    #pragma unroll
    for (uint32_t s = 0; s < S; ++s) {
        if (s < T) tick_stage(s, s);
        asm volatile("cp.async.commit_group;");
    }
    __syncthreads();
    const bool q_live = joff < nlive_sh;
    uint32_t phv[2] = {0u, 0u};

    float acc_g[NW][4] = {}, acc_u[NW][4] = {};
    for (uint32_t tk = 0; tk < T; ++tk) {
        const uint32_t s = tk % S;
        pd_mma_cpa_waitN<(int)S - 1>();
        pd_q2t_wait(mb0 + s * 8u, phv[s]);
        phv[s] ^= 1u;
        __syncthreads();
        if (q_live) {
            char* sb = stage(s);
            int* tile_y = (int*)(sb + YOFF);
            const uint32_t l7 = lane & 7u;
            const uint32_t arow_off = ((lane & 8u) ? 8u : 0u) + l7;
            const uint32_t akof = (lane & 16u) ? 16u : 0u;
            const uint32_t bkof = (lane & 8u) ? 16u : 0u;
            #pragma unroll
            for (uint32_t mat = 0; mat < 2u; ++mat) {
                const char* wd = sb + mat * WPL;
                const __half* wsc = (const __half*)(sb + SCOFF + mat * (RB * 16u));
                float(*acc)[4] = mat ? acc_u : acc_g;
                #pragma unroll
                for (uint32_t bb = 0; bb < 8u; ++bb) {
                    const uint32_t ko = bb * 8u;
                    int b0, b1;
                    pd_mma_ldm_x2((const char*)(tile_y + (joff + l7) * PD_MMQ_XK)
                                      + ko * 4u + bkof,
                                  b0, b1);
                    const float dB0 =
                        ((const float*)tile_y)[(joff + 2u * t) * PD_MMQ_XK + 64u + bb];
                    const float dB1 =
                        ((const float*)tile_y)[(joff + 2u * t + 1u) * PD_MMQ_XK + 64u + bb];
                    #pragma unroll
                    for (uint32_t n = 0; n < NW; ++n) {
                        const uint32_t r = i0 + n * 16u + arow_off;
                        const uint32_t kb = bb * 32u + akof;
                        const uint32_t sw = (kb & 127u) ^ ((r & 7u) * 16u);
                        int A0, A1, A2, A3;
                        pd_mma_ldm_x4(wd + (kb & 128u) * RB + r * 128u + sw, A0, A1, A2,
                                      A3);
                        const uint32_t r0h = i0 + n * 16u + g;
                        const float dA0 = __half2float(wsc[r0h * 8u + bb]);
                        const float dA1 = __half2float(wsc[(r0h + 8u) * 8u + bb]);
                        int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                        asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                            : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                            : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(b0), "r"(b1));
                        acc[n][0] += dA0 * dB0 * (float)d0;
                        acc[n][1] += dA0 * dB1 * (float)d1;
                        acc[n][2] += dA1 * dB0 * (float)d2;
                        acc[n][3] += dA1 * dB1 * (float)d3;
                    }
                }
            }
        }
        __syncthreads();
        if (tk + S < T) tick_stage(tk + S, s);
        asm volatile("cp.async.commit_group;");
    }

    // epilogue: the shipped quantize, NW==2 verbatim; NW==1 = v2's
    // smem-bounce branch (ring dead by now, plane reuses the stage area)
    const uint32_t n_sb = ff / 32u;
    if constexpr (NW == 2u) {
    if (q_live) {
        #pragma unroll
        for (uint32_t qc = 0; qc < 2u; ++qc) {
            const uint32_t c = joff + 2u * t + qc;
            const bool pad = tok[c] == PD_MOE_PAD;
            const uint32_t rb = row_base + i0;
            float sw[4];
            #pragma unroll
            for (uint32_t n = 0; n < NW; ++n) {
                #pragma unroll
                for (uint32_t hq = 0; hq < 2u; ++hq) {
                    const uint32_t q = qc + 2u * hq;
                    const uint32_t r = rb + n * 16u + hq * 8u + g;
                    float out = 0.f;
                    if (!pad && r < ff) {
                        const float gv = acc_g[n][q];
                        const float uv = acc_u[n][q];
                        out = GELU
                            ? 0.5f * gv
                                  * (1.0f
                                     + tanhf(0.79788456080286535587989211986876f * gv
                                             * (1.0f + 0.044715f * gv * gv)))
                                  * uv
                            : (gv / (1.0f + __expf(-gv))) * uv;
                    }
                    sw[n * 2u + hq] = out;
                }
            }
            float a = fmaxf(fmaxf(fabsf(sw[0]), fabsf(sw[1])), fmaxf(fabsf(sw[2]), fabsf(sw[3])));
            #pragma unroll
            for (uint32_t o = 4; o <= 16u; o *= 2u)
                a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
            const float scl = a * (1.0f / 127.0f);
            const float invs = scl > 0.f ? 1.0f / scl : 0.f;
            const size_t row = (size_t)blk * BM + c;
            if (rb < ff) {
                #pragma unroll
                for (uint32_t v = 0; v < 4u; ++v) {
                    const uint32_t r = rb + (v / 2u) * 16u + (v & 1u) * 8u + g;
                    int qi = __float2int_rn(sw[v] * invs);
                    qi = qi < -127 ? -127 : (qi > 127 ? 127 : qi);
                    fq[row * ff + r] = (int8_t)qi;
                }
                if (g == 0) fs[row * n_sb + (rb / 32u)] = scl;
            }
        }
    }
    } else {
    float* plane = (float*)pd_q2t_sh;  // 32 rows x 32 cols
    if (q_live) {
        #pragma unroll
        for (uint32_t qc = 0; qc < 2u; ++qc) {
            const uint32_t c = joff + 2u * t + qc;
            const bool pad = tok[c] == PD_MOE_PAD;
            #pragma unroll
            for (uint32_t hq = 0; hq < 2u; ++hq) {
                const uint32_t q = qc + 2u * hq;
                const uint32_t r16 = i0 + hq * 8u + g;
                const uint32_t r = row_base + r16;
                float out = 0.f;
                if (!pad && r < ff) {
                    const float gv = acc_g[0][q];
                    const float uv = acc_u[0][q];
                    out = GELU
                        ? 0.5f * gv
                              * (1.0f
                                 + tanhf(0.79788456080286535587989211986876f * gv
                                         * (1.0f + 0.044715f * gv * gv)))
                              * uv
                        : (gv / (1.0f + __expf(-gv))) * uv;
                }
                plane[r16 * 32u + c] = out;
            }
        }
    }
    __syncthreads();
    if (q_live && (warp / 4u) == 0u) {
        #pragma unroll
        for (uint32_t qc = 0; qc < 2u; ++qc) {
            const uint32_t c = joff + 2u * t + qc;
            float sw[4];
            #pragma unroll
            for (uint32_t v = 0; v < 4u; ++v)
                sw[v] = plane[((v / 2u) * 16u + (v & 1u) * 8u + g) * 32u + c];
            float a = fmaxf(fmaxf(fabsf(sw[0]), fabsf(sw[1])), fmaxf(fabsf(sw[2]), fabsf(sw[3])));
            #pragma unroll
            for (uint32_t o = 4; o <= 16u; o *= 2u)
                a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
            const float scl = a * (1.0f / 127.0f);
            const float invs = scl > 0.f ? 1.0f / scl : 0.f;
            const size_t row = (size_t)blk * BM + c;
            if (row_base < ff) {
                #pragma unroll
                for (uint32_t v = 0; v < 4u; ++v) {
                    const uint32_t r = row_base + (v / 2u) * 16u + (v & 1u) * 8u + g;
                    int qi = __float2int_rn(sw[v] * invs);
                    qi = qi < -127 ? -127 : (qi > 127 ? 127 : qi);
                    fq[row * ff + r] = (int8_t)qi;
                }
                if (g == 0) fs[row * n_sb + (row_base / 32u)] = scl;
            }
        }
    }
    }
#else
    (void)gmap; (void)umap; (void)gate_scale; (void)up_scale; (void)sorted_row;
    (void)block_expert; (void)xq; (void)xs; (void)fq; (void)fs; (void)in_dim; (void)ff;
#endif
}

template <uint32_t S, uint32_t RB>
__global__ void __launch_bounds__(256, RB == 32u ? 4 : PD_QMMA2_OCC_DN) pd_q8_0_moe_down_mma2t_kernel(
    const __grid_constant__ PdTmap dmap, const __half* __restrict__ down_scale,
    const unsigned int* __restrict__ sorted_row, const unsigned int* __restrict__ sorted_slot,
    const unsigned int* __restrict__ block_expert, const float* __restrict__ topk_w,
    const int8_t* __restrict__ fq, const float* __restrict__ fs,
    float* __restrict__ part, uint32_t ff, uint32_t embd, uint32_t n_active) {
#if PD_Q2T_DEV
    constexpr uint32_t BM = 32u;
    constexpr uint32_t NW = RB / 32u;
    constexpr uint32_t WPL = RB * 256u;
    constexpr uint32_t STRIDE = pd_q2t_dn_stride(RB);
    constexpr uint32_t YOFF = WPL;
    constexpr uint32_t SCOFF = YOFF + PD_Q2T_Y_INT32 * 4u;
    const uint32_t blk = blockIdx.x;
    const uint32_t e = block_expert[blk];
    if (e == PD_MOE_PAD) return;
    const uint32_t row_base = blockIdx.y * RB;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid / 32u;
    const uint32_t g = lane / 4u, t = lane & 3u;
    const uint32_t i0 = (warp & 1u) * 16u * NW;
    const uint32_t joff = (warp / 2u) * 8u;
    const uint32_t nk = (ff + 255u) / 256u;
    const uint32_t n_blocks = ff / 32u;

    extern __shared__ __align__(128) char pd_q2t_sh[];
    __shared__ __align__(8) uint64_t pd_q2t_mb[2];
    __shared__ unsigned int tok[BM], slt[BM], idn[BM];
    __shared__ uint32_t nlive_sh;
    for (uint32_t i = tid; i < BM; i += 256u) {
        tok[i] = sorted_row[(size_t)blk * BM + i];
        slt[i] = sorted_slot[(size_t)blk * BM + i];
        idn[i] = blk * BM + i;
    }
    const uint32_t mb0 = (uint32_t)__cvta_generic_to_shared(pd_q2t_mb);
    if (tid == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(mb0));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(mb0 + 8u));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();
    if (warp == 0) {
        const uint32_t m = __ballot_sync(0xffffffffu, tok[lane] == PD_MOE_PAD);
        if (lane == 0) nlive_sh = m ? (uint32_t)(__ffs((int)m) - 1) : BM;
    }

    const size_t wrow0 = (size_t)e * embd + row_base;
    const uint32_t T = nk;
    auto stage = [&](uint32_t s) { return pd_q2t_sh + s * STRIDE; };
    auto tick_stage = [&](uint32_t tk, uint32_t s) {
        char* sb = stage(s);
        if (tid == 0) {
            const uint32_t m = mb0 + s * 8u;
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(m),
                         "r"(2u * RB * 128u));
            pd_q2t_wtick<RB>(&dmap, (uint32_t)__cvta_generic_to_shared(sb), tk * 256u,
                         (uint32_t)wrow0, m);
        }
        pd_q2t_ystage((int*)(sb + YOFF), (const int*)fq, fs, idn, ff, tk, tid);
        pd_q2t_wsc<RB>((__half*)(sb + SCOFF), down_scale, wrow0, row_base, embd,
                   n_blocks, tk, tid);
    };
    #pragma unroll
    for (uint32_t s = 0; s < S; ++s) {
        if (s < T) tick_stage(s, s);
        asm volatile("cp.async.commit_group;");
    }
    __syncthreads();
    const bool q_live = joff < nlive_sh;
    uint32_t phv[2] = {0u, 0u};

    float acc[NW][4] = {};
    for (uint32_t tk = 0; tk < T; ++tk) {
        const uint32_t s = tk % S;
        pd_mma_cpa_waitN<(int)S - 1>();
        pd_q2t_wait(mb0 + s * 8u, phv[s]);
        phv[s] ^= 1u;
        __syncthreads();
        if (q_live) {
            char* sb = stage(s);
            int* tile_y = (int*)(sb + YOFF);
            const char* wd = sb;
            const __half* wsc = (const __half*)(sb + SCOFF);
            const uint32_t l7 = lane & 7u;
            const uint32_t arow_off = ((lane & 8u) ? 8u : 0u) + l7;
            const uint32_t akof = (lane & 16u) ? 16u : 0u;
            const uint32_t bkof = (lane & 8u) ? 16u : 0u;
            #pragma unroll
            for (uint32_t bb = 0; bb < 8u; ++bb) {
                const uint32_t ko = bb * 8u;
                int b0, b1;
                pd_mma_ldm_x2((const char*)(tile_y + (joff + l7) * PD_MMQ_XK)
                                  + ko * 4u + bkof,
                              b0, b1);
                const float dB0 =
                    ((const float*)tile_y)[(joff + 2u * t) * PD_MMQ_XK + 64u + bb];
                const float dB1 =
                    ((const float*)tile_y)[(joff + 2u * t + 1u) * PD_MMQ_XK + 64u + bb];
                #pragma unroll
                for (uint32_t n = 0; n < NW; ++n) {
                    const uint32_t r = i0 + n * 16u + arow_off;
                    const uint32_t kb = bb * 32u + akof;
                    const uint32_t sw = (kb & 127u) ^ ((r & 7u) * 16u);
                    int A0, A1, A2, A3;
                    pd_mma_ldm_x4(wd + (kb & 128u) * RB + r * 128u + sw, A0, A1, A2, A3);
                    const uint32_t r0h = i0 + n * 16u + g;
                    const float dA0 = __half2float(wsc[r0h * 8u + bb]);
                    const float dA1 = __half2float(wsc[(r0h + 8u) * 8u + bb]);
                    int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                        : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(b0), "r"(b1));
                    acc[n][0] += dA0 * dB0 * (float)d0;
                    acc[n][1] += dA0 * dB1 * (float)d1;
                    acc[n][2] += dA1 * dB0 * (float)d2;
                    acc[n][3] += dA1 * dB1 * (float)d3;
                }
            }
        }
        __syncthreads();
        if (tk + S < T) tick_stage(tk + S, s);
        asm volatile("cp.async.commit_group;");
    }

    if (q_live) {
        const uint32_t c0 = joff + 2u * t;
        #pragma unroll
        for (uint32_t n = 0; n < NW; ++n) {
            const uint32_t r0 = row_base + i0 + n * 16u + g;
            const uint32_t r8 = r0 + 8u;
            #pragma unroll
            for (uint32_t q = 0; q < 4u; ++q) {
                const uint32_t r = (q & 2u) ? r8 : r0;
                const uint32_t c = c0 + (q & 1u);
                const unsigned int token = tok[c];
                if (r >= embd || token == PD_MOE_PAD) continue;
                const float w = topk_w[(size_t)token * n_active + slt[c]];
                const size_t pidx = ((size_t)token * n_active + slt[c]) * embd + r;
                part[pidx] = w * acc[n][q];
            }
        }
    }
#else
    (void)dmap; (void)down_scale; (void)sorted_row; (void)sorted_slot; (void)block_expert;
    (void)topk_w; (void)fq; (void)fs; (void)part; (void)ff; (void)embd; (void)n_active;
#endif
}


// ---- v4 "ws": warp-specialized producer form of the RB=64 twins --------
// 320 threads: warps 0-7 = the v2 consumer map VERBATIM (same fold =>
// bitwise); warps 8-9 (64 thr) stage Y + W-scales via cp.async and drive
// the W TMA. full[s]/empty[s] mbarrier pairs replace the per-tick block
// syncs (the measured wall: OCC-2 starved, OCC-4 wasteful - the barrier-
// coupled tick is what serializes). Producer per-tick: issue, commit,
// wait_group 0, arrive full[s] (64 arrivals + TMA tx); consumers arrive
// empty[s] (256) after their last slot read. Opt-in PADDOCK_Q2T_WS=1.
#if PD_Q2T_DEV
__device__ __forceinline__ void pd_q2t_ystage64(int* yt, const int* xq32,
                                                const float* xs,
                                                const unsigned int* rows,
                                                uint32_t in_dim, uint32_t kt,
                                                uint32_t ptid) {
    const uint32_t n_blocks = in_dim / 32u, n_k32 = in_dim / 4u;
    #pragma unroll
    for (uint32_t it = 0; it < 8u; ++it) {
        const uint32_t i = it * 64u + ptid;
        const uint32_t c = i / 16u, ch = i & 15u;
        const unsigned int r = rows[c];
        const uint32_t gk = kt * 64u + ch * 4u;
        const bool ok = r != PD_MOE_PAD && gk < n_k32;
        pd_cp_async16(yt + c * PD_MMQ_XK + ch * 4u,
                      xq32 + (ok ? ((size_t)r * n_k32 + gk) : 0u), ok);
    }
    #pragma unroll
    for (uint32_t it = 0; it < 4u; ++it) {
        const uint32_t i = it * 64u + ptid;
        const uint32_t c = i / 8u, b = i & 7u, gb = kt * 8u + b;
        const unsigned int r = rows[c];
        const bool ok = r != PD_MOE_PAD && gb < n_blocks;
        pd_mma_cpa4p((float*)(yt + c * PD_MMQ_XK + 64u) + b,
                     xs + (ok ? ((size_t)r * n_blocks + gb) : 0u), ok);
    }
}
__device__ __forceinline__ void pd_q2t_wsc64(__half* dst, const __half* ws,
                                             size_t wrow0, uint32_t row_base,
                                             uint32_t out_dim, uint32_t n_blocks,
                                             uint32_t kt, uint32_t ptid) {
    #pragma unroll
    for (uint32_t it = 0; it < 4u; ++it) {
        const uint32_t i = it * 64u + ptid;
        const uint32_t row = i / 4u, c = i & 3u, gb = kt * 8u + c * 2u;
        const bool ok = gb < n_blocks && (row_base + row) < out_dim;
        pd_mma_cpa4p(dst + row * 8u + c * 2u,
                     ws + (wrow0 + row) * n_blocks + (ok ? gb : 0u), ok);
    }
}
#endif  // PD_Q2T_DEV

template <uint32_t S, bool GELU>
__global__ void __launch_bounds__(320, 2) pd_q8_0_moe_gate_up_mma2w_kernel(
    const __grid_constant__ PdTmap gmap, const __grid_constant__ PdTmap umap,
    const __half* __restrict__ gate_scale, const __half* __restrict__ up_scale,
    const unsigned int* __restrict__ sorted_row, const unsigned int* __restrict__ block_expert,
    const int8_t* __restrict__ xq, const float* __restrict__ xs,
    int8_t* __restrict__ fq, float* __restrict__ fs, uint32_t in_dim, uint32_t ff) {
#if PD_Q2T_DEV
    constexpr uint32_t BM = 32u;
    constexpr uint32_t RB = 64u;
    constexpr uint32_t NW = 2u;
    constexpr uint32_t WPL = RB * 256u;
    constexpr uint32_t STRIDE = pd_q2t_gu_stride(RB);
    constexpr uint32_t YOFF = 2u * WPL;
    constexpr uint32_t SCOFF = YOFF + PD_Q2T_Y_INT32 * 4u;
    const uint32_t blk = blockIdx.x;
    const uint32_t e = block_expert[blk];
    if (e == PD_MOE_PAD) return;
    const uint32_t row_base = blockIdx.y * RB;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid / 32u;
    const uint32_t g = lane / 4u, t = lane & 3u;
    const uint32_t i0 = (warp & 1u) * 16u * NW;
    const uint32_t joff = (warp / 2u) * 8u;
    const uint32_t nk = (in_dim + 255u) / 256u;
    const uint32_t n_blocks = in_dim / 32u;

    extern __shared__ __align__(128) char pd_q2t_sh[];
    __shared__ __align__(8) uint64_t mbf[2], mbe[2];
    __shared__ unsigned int tok[BM];
    __shared__ uint32_t nlive_sh;
    for (uint32_t i = tid; i < BM; i += 320u) tok[i] = sorted_row[(size_t)blk * BM + i];
    const uint32_t mf0 = (uint32_t)__cvta_generic_to_shared(mbf);
    const uint32_t me0 = (uint32_t)__cvta_generic_to_shared(mbe);
    if (tid == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 64;" ::"r"(mf0));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 64;" ::"r"(mf0 + 8u));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 256;" ::"r"(me0));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 256;" ::"r"(me0 + 8u));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();
    if (warp == 0) {
        const uint32_t m = __ballot_sync(0xffffffffu, tok[lane] == PD_MOE_PAD);
        if (lane == 0) nlive_sh = m ? (uint32_t)(__ffs((int)m) - 1) : BM;
    }
    __syncthreads();

    const size_t wrow0 = (size_t)e * ff + row_base;
    const uint32_t T = nk;
    auto stage = [&](uint32_t s) { return pd_q2t_sh + s * STRIDE; };

    if (warp >= 8u) {
        // ---------------- producer warps 8-9 ----------------
        const uint32_t ptid = tid - 256u;
        uint32_t phe[2] = {0u, 0u};
        for (uint32_t tk = 0; tk < T; ++tk) {
            const uint32_t s = tk % S;
            if (tk >= S) {
                pd_q2t_wait(me0 + s * 8u, phe[s]);
                phe[s] ^= 1u;
            }
            char* sb = stage(s);
            if (ptid == 0) {
                asm volatile(
                    "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(
                        mf0 + s * 8u),
                    "r"(4u * RB * 128u));
                const uint32_t ck = tk * 256u;
                pd_q2t_wtick<RB>(&gmap, (uint32_t)__cvta_generic_to_shared(sb), ck,
                             (uint32_t)wrow0, mf0 + s * 8u);
                pd_q2t_wtick<RB>(&umap,
                             (uint32_t)__cvta_generic_to_shared(sb + WPL), ck,
                             (uint32_t)wrow0, mf0 + s * 8u);
            }
            pd_q2t_ystage64((int*)(sb + YOFF), (const int*)xq, xs, tok, in_dim, tk,
                            ptid);
            pd_q2t_wsc64((__half*)(sb + SCOFF), gate_scale, wrow0, row_base, ff,
                         n_blocks, tk, ptid);
            pd_q2t_wsc64((__half*)(sb + SCOFF + RB * 16u), up_scale, wrow0, row_base,
                         ff, n_blocks, tk, ptid);
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 0;");
            if (ptid != 0)
                asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(mf0 +
                                                                              s * 8u));
        }
        return;
    }

    // ---------------- consumer warps 0-7 (v2 map verbatim) ----------------
    const bool q_live = joff < nlive_sh;
    uint32_t phf[2] = {0u, 0u};
    float acc_g[NW][4] = {}, acc_u[NW][4] = {};
    for (uint32_t tk = 0; tk < T; ++tk) {
        const uint32_t s = tk % S;
        pd_q2t_wait(mf0 + s * 8u, phf[s]);
        phf[s] ^= 1u;
        if (q_live) {
            char* sb = stage(s);
            int* tile_y = (int*)(sb + YOFF);
            const uint32_t l7 = lane & 7u;
            const uint32_t arow_off = ((lane & 8u) ? 8u : 0u) + l7;
            const uint32_t akof = (lane & 16u) ? 16u : 0u;
            const uint32_t bkof = (lane & 8u) ? 16u : 0u;
            #pragma unroll
            for (uint32_t mat = 0; mat < 2u; ++mat) {
                const char* wd = sb + mat * WPL;
                const __half* wsc = (const __half*)(sb + SCOFF + mat * (RB * 16u));
                float(*acc)[4] = mat ? acc_u : acc_g;
                #pragma unroll
                for (uint32_t bb = 0; bb < 8u; ++bb) {
                    const uint32_t ko = bb * 8u;
                    int b0, b1;
                    pd_mma_ldm_x2((const char*)(tile_y + (joff + l7) * PD_MMQ_XK)
                                      + ko * 4u + bkof,
                                  b0, b1);
                    const float dB0 =
                        ((const float*)tile_y)[(joff + 2u * t) * PD_MMQ_XK + 64u + bb];
                    const float dB1 =
                        ((const float*)tile_y)[(joff + 2u * t + 1u) * PD_MMQ_XK + 64u + bb];
                    #pragma unroll
                    for (uint32_t n = 0; n < NW; ++n) {
                        const uint32_t r = i0 + n * 16u + arow_off;
                        const uint32_t kb = bb * 32u + akof;
                        const uint32_t sw = (kb & 127u) ^ ((r & 7u) * 16u);
                        int A0, A1, A2, A3;
                        pd_mma_ldm_x4(wd + (kb & 128u) * RB + r * 128u + sw, A0, A1,
                                      A2, A3);
                        const uint32_t r0h = i0 + n * 16u + g;
                        const float dA0 = __half2float(wsc[r0h * 8u + bb]);
                        const float dA1 = __half2float(wsc[(r0h + 8u) * 8u + bb]);
                        int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                        asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                            : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                            : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(b0), "r"(b1));
                        acc[n][0] += dA0 * dB0 * (float)d0;
                        acc[n][1] += dA0 * dB1 * (float)d1;
                        acc[n][2] += dA1 * dB0 * (float)d2;
                        acc[n][3] += dA1 * dB1 * (float)d3;
                    }
                }
            }
        }
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(me0 + s * 8u));
    }

    // epilogue: the shipped NW==2 quantize verbatim (acc-only)
    const uint32_t n_sb = ff / 32u;
    if (q_live) {
        #pragma unroll
        for (uint32_t qc = 0; qc < 2u; ++qc) {
            const uint32_t c = joff + 2u * t + qc;
            const bool pad = tok[c] == PD_MOE_PAD;
            const uint32_t rb = row_base + i0;
            float sw[4];
            #pragma unroll
            for (uint32_t n = 0; n < NW; ++n) {
                #pragma unroll
                for (uint32_t hq = 0; hq < 2u; ++hq) {
                    const uint32_t q = qc + 2u * hq;
                    const uint32_t r = rb + n * 16u + hq * 8u + g;
                    float out = 0.f;
                    if (!pad && r < ff) {
                        const float gv = acc_g[n][q];
                        const float uv = acc_u[n][q];
                        out = GELU
                            ? 0.5f * gv
                                  * (1.0f
                                     + tanhf(0.79788456080286535587989211986876f * gv
                                             * (1.0f + 0.044715f * gv * gv)))
                                  * uv
                            : (gv / (1.0f + __expf(-gv))) * uv;
                    }
                    sw[n * 2u + hq] = out;
                }
            }
            float a = fmaxf(fmaxf(fabsf(sw[0]), fabsf(sw[1])), fmaxf(fabsf(sw[2]), fabsf(sw[3])));
            #pragma unroll
            for (uint32_t o = 4; o <= 16u; o *= 2u)
                a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
            const float scl = a * (1.0f / 127.0f);
            const float invs = scl > 0.f ? 1.0f / scl : 0.f;
            const size_t row = (size_t)blk * BM + c;
            if (rb < ff) {
                #pragma unroll
                for (uint32_t v = 0; v < 4u; ++v) {
                    const uint32_t r = rb + (v / 2u) * 16u + (v & 1u) * 8u + g;
                    int qi = __float2int_rn(sw[v] * invs);
                    qi = qi < -127 ? -127 : (qi > 127 ? 127 : qi);
                    fq[row * ff + r] = (int8_t)qi;
                }
                if (g == 0) fs[row * n_sb + (rb / 32u)] = scl;
            }
        }
    }
#else
    (void)gmap; (void)umap; (void)gate_scale; (void)up_scale; (void)sorted_row;
    (void)block_expert; (void)xq; (void)xs; (void)fq; (void)fs; (void)in_dim; (void)ff;
#endif
}

template <uint32_t S>
__global__ void __launch_bounds__(320, 2) pd_q8_0_moe_down_mma2w_kernel(
    const __grid_constant__ PdTmap dmap, const __half* __restrict__ down_scale,
    const unsigned int* __restrict__ sorted_row, const unsigned int* __restrict__ sorted_slot,
    const unsigned int* __restrict__ block_expert, const float* __restrict__ topk_w,
    const int8_t* __restrict__ fq, const float* __restrict__ fs,
    float* __restrict__ part, uint32_t ff, uint32_t embd, uint32_t n_active) {
#if PD_Q2T_DEV
    constexpr uint32_t BM = 32u;
    constexpr uint32_t RB = 64u;
    constexpr uint32_t NW = 2u;
    constexpr uint32_t WPL = RB * 256u;
    constexpr uint32_t STRIDE = pd_q2t_dn_stride(RB);
    constexpr uint32_t YOFF = WPL;
    constexpr uint32_t SCOFF = YOFF + PD_Q2T_Y_INT32 * 4u;
    const uint32_t blk = blockIdx.x;
    const uint32_t e = block_expert[blk];
    if (e == PD_MOE_PAD) return;
    const uint32_t row_base = blockIdx.y * RB;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid / 32u;
    const uint32_t g = lane / 4u, t = lane & 3u;
    const uint32_t i0 = (warp & 1u) * 16u * NW;
    const uint32_t joff = (warp / 2u) * 8u;
    const uint32_t nk = (ff + 255u) / 256u;
    const uint32_t n_blocks = ff / 32u;

    extern __shared__ __align__(128) char pd_q2t_sh[];
    __shared__ __align__(8) uint64_t mbf[2], mbe[2];
    __shared__ unsigned int tok[BM], slt[BM], idn[BM];
    __shared__ uint32_t nlive_sh;
    for (uint32_t i = tid; i < BM; i += 320u) {
        tok[i] = sorted_row[(size_t)blk * BM + i];
        slt[i] = sorted_slot[(size_t)blk * BM + i];
        idn[i] = blk * BM + i;
    }
    const uint32_t mf0 = (uint32_t)__cvta_generic_to_shared(mbf);
    const uint32_t me0 = (uint32_t)__cvta_generic_to_shared(mbe);
    if (tid == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 64;" ::"r"(mf0));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 64;" ::"r"(mf0 + 8u));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 256;" ::"r"(me0));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 256;" ::"r"(me0 + 8u));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();
    if (warp == 0) {
        const uint32_t m = __ballot_sync(0xffffffffu, tok[lane] == PD_MOE_PAD);
        if (lane == 0) nlive_sh = m ? (uint32_t)(__ffs((int)m) - 1) : BM;
    }
    __syncthreads();

    const size_t wrow0 = (size_t)e * embd + row_base;
    const uint32_t T = nk;
    auto stage = [&](uint32_t s) { return pd_q2t_sh + s * STRIDE; };

    if (warp >= 8u) {
        const uint32_t ptid = tid - 256u;
        uint32_t phe[2] = {0u, 0u};
        for (uint32_t tk = 0; tk < T; ++tk) {
            const uint32_t s = tk % S;
            if (tk >= S) {
                pd_q2t_wait(me0 + s * 8u, phe[s]);
                phe[s] ^= 1u;
            }
            char* sb = stage(s);
            if (ptid == 0) {
                asm volatile(
                    "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(
                        mf0 + s * 8u),
                    "r"(2u * RB * 128u));
                pd_q2t_wtick<RB>(&dmap, (uint32_t)__cvta_generic_to_shared(sb),
                             tk * 256u, (uint32_t)wrow0, mf0 + s * 8u);
            }
            pd_q2t_ystage64((int*)(sb + YOFF), (const int*)fq, fs, idn, ff, tk, ptid);
            pd_q2t_wsc64((__half*)(sb + SCOFF), down_scale, wrow0, row_base, embd,
                         n_blocks, tk, ptid);
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 0;");
            if (ptid != 0)
                asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(mf0 +
                                                                              s * 8u));
        }
        return;
    }

    const bool q_live = joff < nlive_sh;
    uint32_t phf[2] = {0u, 0u};
    float acc[NW][4] = {};
    for (uint32_t tk = 0; tk < T; ++tk) {
        const uint32_t s = tk % S;
        pd_q2t_wait(mf0 + s * 8u, phf[s]);
        phf[s] ^= 1u;
        if (q_live) {
            char* sb = stage(s);
            int* tile_y = (int*)(sb + YOFF);
            const char* wd = sb;
            const __half* wsc = (const __half*)(sb + SCOFF);
            const uint32_t l7 = lane & 7u;
            const uint32_t arow_off = ((lane & 8u) ? 8u : 0u) + l7;
            const uint32_t akof = (lane & 16u) ? 16u : 0u;
            const uint32_t bkof = (lane & 8u) ? 16u : 0u;
            #pragma unroll
            for (uint32_t bb = 0; bb < 8u; ++bb) {
                const uint32_t ko = bb * 8u;
                int b0, b1;
                pd_mma_ldm_x2((const char*)(tile_y + (joff + l7) * PD_MMQ_XK)
                                  + ko * 4u + bkof,
                              b0, b1);
                const float dB0 =
                    ((const float*)tile_y)[(joff + 2u * t) * PD_MMQ_XK + 64u + bb];
                const float dB1 =
                    ((const float*)tile_y)[(joff + 2u * t + 1u) * PD_MMQ_XK + 64u + bb];
                #pragma unroll
                for (uint32_t n = 0; n < NW; ++n) {
                    const uint32_t r = i0 + n * 16u + arow_off;
                    const uint32_t kb = bb * 32u + akof;
                    const uint32_t sw = (kb & 127u) ^ ((r & 7u) * 16u);
                    int A0, A1, A2, A3;
                    pd_mma_ldm_x4(wd + (kb & 128u) * RB + r * 128u + sw, A0, A1, A2,
                                  A3);
                    const uint32_t r0h = i0 + n * 16u + g;
                    const float dA0 = __half2float(wsc[r0h * 8u + bb]);
                    const float dA1 = __half2float(wsc[(r0h + 8u) * 8u + bb]);
                    int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                        : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(b0), "r"(b1));
                    acc[n][0] += dA0 * dB0 * (float)d0;
                    acc[n][1] += dA0 * dB1 * (float)d1;
                    acc[n][2] += dA1 * dB0 * (float)d2;
                    acc[n][3] += dA1 * dB1 * (float)d3;
                }
            }
        }
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(me0 + s * 8u));
    }

    if (q_live) {
        const uint32_t c0 = joff + 2u * t;
        #pragma unroll
        for (uint32_t n = 0; n < NW; ++n) {
            const uint32_t r0 = row_base + i0 + n * 16u + g;
            const uint32_t r8 = r0 + 8u;
            #pragma unroll
            for (uint32_t q = 0; q < 4u; ++q) {
                const uint32_t r = (q & 2u) ? r8 : r0;
                const uint32_t c = c0 + (q & 1u);
                const unsigned int token = tok[c];
                if (r >= embd || token == PD_MOE_PAD) continue;
                const float w = topk_w[(size_t)token * n_active + slt[c]];
                const size_t pidx = ((size_t)token * n_active + slt[c]) * embd + r;
                part[pidx] = w * acc[n][q];
            }
        }
    }
#else
    (void)dmap; (void)down_scale; (void)sorted_row; (void)sorted_slot; (void)block_expert;
    (void)topk_w; (void)fq; (void)fs; (void)part; (void)ff; (void)embd; (void)n_active;
#endif
}

// v3t launchers (slots 502/503). PADDOCK_Q2T_RB=32 selects the 4-CTA/SM
// variant (default 64). NotSupported without the host map builders; the
// exports resolver NULLs the slots below cc 9.
#if defined(PD_BS_HOST) || defined(PD_TC5_HOST)
static inline int pd_q2t_rb(void) {
    static int rb = -1;
    if (rb < 0) {
        const char* e = pd_env("PADDOCK_Q2T_RB");
        rb = (e && e[0] == '3') ? 32 : 64;
    }
    return rb;
}
static inline int pd_q2t_ws(void) {
    static int ws = -1;
    if (ws < 0) {
        const char* e = pd_env("PADDOCK_Q2T_WS");
        ws = (e && e[0] == '1') ? 1 : 0;
    }
    return ws;
}
#endif

PD_EXPORT
int pd_q8_0_moe_gate_up_mma2t_geglu(const void* gate_data, const void* gate_scale,
                                    const void* up_data, const void* up_scale,
                                    const void* sorted_row, const void* block_expert,
                                    const void* xq, const void* xs, void* fq, void* fs,
                                    uint32_t in_dim, uint32_t ff, uint32_t n_expert,
                                    uint32_t max_blocks, uint32_t bm, void* stream) {
#if defined(PD_BS_HOST) || defined(PD_TC5_HOST)
    if (ff == 0 || max_blocks == 0) return 0;
    if (bm != 32u) return cudaErrorNotSupported;
    if ((in_dim & 255u) != 0 || (ff & 63u) != 0) return cudaErrorInvalidValue;
    CUtensorMap gmap, umap;
    const int rb = pd_q2t_ws() ? 64 : pd_q2t_rb();
    const bool ok = (rb == 32)
        ? (pd_tmap_2d_h32(&gmap, gate_data, in_dim, (uint64_t)n_expert * ff) &&
           pd_tmap_2d_h32(&umap, up_data, in_dim, (uint64_t)n_expert * ff))
        : (pd_tmap_2d_h64(&gmap, gate_data, in_dim, (uint64_t)n_expert * ff) &&
           pd_tmap_2d_h64(&umap, up_data, in_dim, (uint64_t)n_expert * ff));
    if (!ok) return cudaErrorNotSupported;
    constexpr uint32_t S = 2u;
    if (pd_q2t_ws()) {
        constexpr uint32_t smem = pd_q2t_gu_stride(64u) * S;
        static bool attr = false;
        if (!attr) {
            cudaFuncSetAttribute((const void*)pd_q8_0_moe_gate_up_mma2w_kernel<S, true>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
            attr = true;
        }
        dim3 grid(max_blocks, (ff + 63u) / 64u);
        pd_q8_0_moe_gate_up_mma2w_kernel<S, true><<<grid, 320, smem,
            (cudaStream_t)stream>>>(
            gmap, umap, (const __half*)gate_scale, (const __half*)up_scale,
            (const unsigned int*)sorted_row, (const unsigned int*)block_expert,
            (const int8_t*)xq, (const float*)xs, (int8_t*)fq, (float*)fs, in_dim, ff);
    } else if (rb == 32) {
        constexpr uint32_t smem = pd_q2t_gu_stride(32u) * S;
        static bool attr = false;
        if (!attr) {
            cudaFuncSetAttribute((const void*)pd_q8_0_moe_gate_up_mma2t_kernel<S, true, 32u>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
            attr = true;
        }
        dim3 grid(max_blocks, (ff + 31u) / 32u);
        pd_q8_0_moe_gate_up_mma2t_kernel<S, true, 32u><<<grid, 256, smem,
            (cudaStream_t)stream>>>(
            gmap, umap, (const __half*)gate_scale, (const __half*)up_scale,
            (const unsigned int*)sorted_row, (const unsigned int*)block_expert,
            (const int8_t*)xq, (const float*)xs, (int8_t*)fq, (float*)fs, in_dim, ff);
    } else {
        constexpr uint32_t smem = pd_q2t_gu_stride(64u) * S;
        static bool attr = false;
        if (!attr) {
            cudaFuncSetAttribute((const void*)pd_q8_0_moe_gate_up_mma2t_kernel<S, true, 64u>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
            attr = true;
        }
        dim3 grid(max_blocks, (ff + 63u) / 64u);
        pd_q8_0_moe_gate_up_mma2t_kernel<S, true, 64u><<<grid, 256, smem,
            (cudaStream_t)stream>>>(
            gmap, umap, (const __half*)gate_scale, (const __half*)up_scale,
            (const unsigned int*)sorted_row, (const unsigned int*)block_expert,
            (const int8_t*)xq, (const float*)xs, (int8_t*)fq, (float*)fs, in_dim, ff);
    }
    return pd_launch_status();
#else
    (void)gate_data; (void)gate_scale; (void)up_data; (void)up_scale; (void)sorted_row;
    (void)block_expert; (void)xq; (void)xs; (void)fq; (void)fs; (void)in_dim; (void)ff;
    (void)n_expert; (void)max_blocks; (void)bm; (void)stream;
    return cudaErrorNotSupported;
#endif
}

PD_EXPORT
int pd_q8_0_moe_down_mma2t(const void* down_data, const void* down_scale,
                           const void* sorted_row, const void* sorted_slot,
                           const void* block_expert, const void* topk_w, const void* fq,
                           const void* fs, void* part, uint32_t ff, uint32_t embd,
                           uint32_t n_expert, uint32_t n_active, uint32_t max_blocks,
                           uint32_t bm, void* stream) {
#if defined(PD_BS_HOST) || defined(PD_TC5_HOST)
    if (embd == 0 || max_blocks == 0) return 0;
    if (bm != 32u) return cudaErrorNotSupported;
    if ((ff & 63u) != 0 || (embd & 63u) != 0) return cudaErrorInvalidValue;
    CUtensorMap dmap;
    const int rb = pd_q2t_ws() ? 64 : pd_q2t_rb();
    const bool ok = (rb == 32)
        ? pd_tmap_2d_h32(&dmap, down_data, ff, (uint64_t)n_expert * embd)
        : pd_tmap_2d_h64(&dmap, down_data, ff, (uint64_t)n_expert * embd);
    if (!ok) return cudaErrorNotSupported;
    constexpr uint32_t S = (uint32_t)PD_QMMA2_DN_S;
    if (pd_q2t_ws()) {
        constexpr uint32_t smem = pd_q2t_dn_stride(64u) * S;
        static bool attr = false;
        if (!attr) {
            cudaFuncSetAttribute((const void*)pd_q8_0_moe_down_mma2w_kernel<S>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
            attr = true;
        }
        dim3 grid(max_blocks, (embd + 63u) / 64u);
        pd_q8_0_moe_down_mma2w_kernel<S><<<grid, 320, smem, (cudaStream_t)stream>>>(
            dmap, (const __half*)down_scale, (const unsigned int*)sorted_row,
            (const unsigned int*)sorted_slot, (const unsigned int*)block_expert,
            (const float*)topk_w, (const int8_t*)fq, (const float*)fs, (float*)part, ff,
            embd, n_active);
    } else if (rb == 32) {
        constexpr uint32_t smem = pd_q2t_dn_stride(32u) * S;
        static bool attr = false;
        if (!attr) {
            cudaFuncSetAttribute((const void*)pd_q8_0_moe_down_mma2t_kernel<S, 32u>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
            attr = true;
        }
        dim3 grid(max_blocks, (embd + 31u) / 32u);
        pd_q8_0_moe_down_mma2t_kernel<S, 32u><<<grid, 256, smem, (cudaStream_t)stream>>>(
            dmap, (const __half*)down_scale, (const unsigned int*)sorted_row,
            (const unsigned int*)sorted_slot, (const unsigned int*)block_expert,
            (const float*)topk_w, (const int8_t*)fq, (const float*)fs, (float*)part, ff,
            embd, n_active);
    } else {
        constexpr uint32_t smem = pd_q2t_dn_stride(64u) * S;
        static bool attr = false;
        if (!attr) {
            cudaFuncSetAttribute((const void*)pd_q8_0_moe_down_mma2t_kernel<S, 64u>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
            attr = true;
        }
        dim3 grid(max_blocks, (embd + 63u) / 64u);
        pd_q8_0_moe_down_mma2t_kernel<S, 64u><<<grid, 256, smem, (cudaStream_t)stream>>>(
            dmap, (const __half*)down_scale, (const unsigned int*)sorted_row,
            (const unsigned int*)sorted_slot, (const unsigned int*)block_expert,
            (const float*)topk_w, (const int8_t*)fq, (const float*)fs, (float*)part, ff,
            embd, n_active);
    }
    return pd_launch_status();
#else
    (void)down_data; (void)down_scale; (void)sorted_row; (void)sorted_slot;
    (void)block_expert; (void)topk_w; (void)fq; (void)fs; (void)part; (void)ff;
    (void)embd; (void)n_expert; (void)n_active; (void)max_blocks; (void)bm; (void)stream;
    return cudaErrorNotSupported;
#endif
}

#else  // shipped-config gate
PD_EXPORT
int pd_q8_0_moe_gate_up_mma2t_geglu(const void*, const void*, const void*, const void*,
                                    const void*, const void*, const void*, const void*,
                                    void*, void*, uint32_t, uint32_t, uint32_t, uint32_t,
                                    uint32_t, void*) {
    return cudaErrorNotSupported;
}
PD_EXPORT
int pd_q8_0_moe_down_mma2t(const void*, const void*, const void*, const void*,
                           const void*, const void*, const void*, const void*, void*,
                           uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                           void*) {
    return cudaErrorNotSupported;
}
#endif  // PD_QMMA2_ILV && PD_QMMA2_LDM && !PD_QMMA2_YSYNC


// ===================== g2: token-major gate_up =============================
// BM=16 tokens in mma-M, BN=64 ff-rows with both mats co-resident, W via
// TMA SW128 + per-lane chunk-XOR ldmatrix (the v3t-proven staging), A =
// gathered token tile at 272B pitch. BITWISE to the v2 pair (probed
// exact on uni:128/512/1024 fq+fs) and 1.13x faster at
// uni:128; slower at wave widths (mb16 splits full experts) => elected at
// decode widths only. The epilogue writes fq/fs at BM32 rows via the
// pair map (map[token*k + slot] = bm32 row) so the v2 down and the whole
// downstream chain are untouched.
#if PD_QMMA2_ILV && PD_QMMA2_LDM && !PD_QMMA2_YSYNC

#define PD_G2_BM 16u
#define PD_G2_BN 64u
#define PD_G2_AP 68u
#define PD_G2_WOFF_U 16384u
#define PD_G2_AOFF   32768u
#define PD_G2_SWG    37120u
#define PD_G2_SWU    38144u
#define PD_G2_SX     39168u
#define PD_G2_STRIDE 39936u

template <uint32_t S, bool GELU>
__global__ void __launch_bounds__(256, 2) pd_q8_0_moe_gate_up_g2_kernel(
    const __grid_constant__ PdTmap gmap, const __grid_constant__ PdTmap umap,
    const __half* __restrict__ gate_scale, const __half* __restrict__ up_scale,
    const unsigned int* __restrict__ sorted_row, const unsigned int* __restrict__ sorted_slot,
    const unsigned int* __restrict__ block_expert, const unsigned int* __restrict__ pmap,
    const int8_t* __restrict__ xq, const float* __restrict__ xs,
    int8_t* __restrict__ fq, float* __restrict__ fs, uint32_t in_dim, uint32_t ff,
    uint32_t n_active) {
#if PD_Q2T_DEV
    const uint32_t blk = blockIdx.x;
    const uint32_t e = block_expert[blk];
    if (e == PD_MOE_PAD) return;
    const uint32_t row_base = blockIdx.y * PD_G2_BN;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid / 32u;
    const uint32_t g = lane / 4u, t = lane & 3u;
    const uint32_t nk = (in_dim + 255u) / 256u;
    const uint32_t n_blocks = in_dim / 32u, n_k32 = in_dim / 4u;
    const uint32_t wrow = warp * 8u;

    extern __shared__ __align__(128) char pd_g2_sh[];
    __shared__ __align__(8) uint64_t pd_g2_mb[2];
    __shared__ unsigned int tok[PD_G2_BM], slt[PD_G2_BM];
    if (tid < PD_G2_BM) {
        tok[tid] = sorted_row[(size_t)blk * PD_G2_BM + tid];
        slt[tid] = sorted_slot[(size_t)blk * PD_G2_BM + tid];
    }
    const uint32_t mb0 = (uint32_t)__cvta_generic_to_shared(pd_g2_mb);
    if (tid == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(mb0));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(mb0 + 8u));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();

    const size_t wrow0 = (size_t)e * ff + row_base;
    const uint32_t T = nk;
    auto stage = [&](uint32_t s) { return pd_g2_sh + s * PD_G2_STRIDE; };
    auto tick_stage = [&](uint32_t tk, uint32_t s) {
        char* sb = stage(s);
        if (tid == 0) {
            const uint32_t m = mb0 + s * 8u;
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(m),
                         "r"(32768u));
            const uint32_t ck = tk * 256u;
            const uint32_t d0 = (uint32_t)__cvta_generic_to_shared(sb);
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                " [%0], [%1, {%2, %3}], [%4];" ::"r"(d0), "l"(&gmap), "r"((int)ck),
                "r"((int)wrow0), "r"(m) : "memory");
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                " [%0], [%1, {%2, %3}], [%4];" ::"r"(d0 + 8192u), "l"(&gmap),
                "r"((int)(ck + 128u)), "r"((int)wrow0), "r"(m) : "memory");
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                " [%0], [%1, {%2, %3}], [%4];" ::"r"(d0 + PD_G2_WOFF_U), "l"(&umap),
                "r"((int)ck), "r"((int)wrow0), "r"(m) : "memory");
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                " [%0], [%1, {%2, %3}], [%4];" ::"r"(d0 + PD_G2_WOFF_U + 8192u),
                "l"(&umap), "r"((int)(ck + 128u)), "r"((int)wrow0), "r"(m) : "memory");
        }
        {
            int* at = (int*)(sb + PD_G2_AOFF);
            const uint32_t c = tid / 16u, ch = tid & 15u;
            const unsigned int r = tok[c];
            const uint32_t gk = tk * 64u + ch * 4u;
            const bool ok = r != PD_MOE_PAD && gk < n_k32;
            pd_cp_async16(at + c * PD_G2_AP + ch * 4u,
                          (const int*)xq + (ok ? ((size_t)r * n_k32 + gk) : 0u), ok);
        }
        if (tid < PD_G2_BM * 8u) {
            float* sx = (float*)(sb + PD_G2_SX);
            const uint32_t c = tid / 8u, b = tid & 7u, gb = tk * 8u + b;
            const unsigned int r = tok[c];
            const bool ok = r != PD_MOE_PAD && gb < n_blocks;
            pd_mma_cpa4p(sx + c * 8u + b, xs + (ok ? ((size_t)r * n_blocks + gb) : 0u),
                         ok);
        }
        {
            const uint32_t row = tid / 4u, cc = tid & 3u, gb = tk * 8u + cc * 2u;
            const bool ok = gb < n_blocks && (row_base + row) < ff;
            pd_mma_cpa4p((__half*)(sb + PD_G2_SWG) + row * 8u + cc * 2u,
                         gate_scale + (wrow0 + row) * n_blocks + (ok ? gb : 0u), ok);
            pd_mma_cpa4p((__half*)(sb + PD_G2_SWU) + row * 8u + cc * 2u,
                         up_scale + (wrow0 + row) * n_blocks + (ok ? gb : 0u), ok);
        }
    };
    #pragma unroll
    for (uint32_t s = 0; s < S; ++s) {
        if (s < T) tick_stage(s, s);
        asm volatile("cp.async.commit_group;");
    }
    __syncthreads();
    uint32_t phv[2] = {0u, 0u};

    float acc_g[4] = {}, acc_u[4] = {};
    for (uint32_t tk = 0; tk < T; ++tk) {
        const uint32_t s = tk % S;
        pd_mma_cpa_waitN<(int)S - 1>();
        asm volatile(
            "{.reg .pred p; W%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;"
            " @!p bra W%=;}" ::"r"(mb0 + s * 8u), "r"(phv[s]));
        phv[s] ^= 1u;
        __syncthreads();
        {
            char* sb = stage(s);
            const int* at = (const int*)(sb + PD_G2_AOFF);
            const float* sx = (const float*)(sb + PD_G2_SX);
            const uint32_t a16 = (lane & 16u) ? 16u : 0u;
            const uint32_t arow = lane & 15u;
            const uint32_t l7 = lane & 7u;
            const uint32_t bk8 = (lane & 8u) ? 16u : 0u;
            #pragma unroll
            for (uint32_t bb = 0; bb < 8u; ++bb) {
                int A0, A1, A2, A3;
                pd_mma_ldm_x4((const char*)(at + arow * PD_G2_AP) + bb * 32u + a16, A0,
                              A1, A2, A3);
                const float dX0 = sx[g * 8u + bb];
                const float dX1 = sx[(g + 8u) * 8u + bb];
                #pragma unroll
                for (uint32_t mat = 0; mat < 2u; ++mat) {
                    const char* wp = sb + mat * PD_G2_WOFF_U;
                    const __half* wsc =
                        (const __half*)(sb + PD_G2_SWG + mat * 1024u) + wrow * 8u;
                    const uint32_t kb = bb * 32u + bk8;
                    const uint32_t r = wrow + l7;
                    int B0, B1;
                    pd_mma_ldm_x2(wp + (kb & 128u) * 64u + r * 128u
                                      + ((kb & 127u) ^ ((r & 7u) * 16u)),
                                  B0, B1);
                    const float dW0 = __half2float(wsc[(2u * t) * 8u + bb]);
                    const float dW1 = __half2float(wsc[(2u * t + 1u) * 8u + bb]);
                    float(*acc) = mat ? acc_u : acc_g;
                    int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                        : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(B0), "r"(B1));
                    acc[0] += dW0 * dX0 * (float)d0;
                    acc[1] += dW1 * dX0 * (float)d1;
                    acc[2] += dW0 * dX1 * (float)d2;
                    acc[3] += dW1 * dX1 * (float)d3;
                }
            }
        }
        __syncthreads();
        if (tk + S < T) tick_stage(tk + S, s);
        asm volatile("cp.async.commit_group;");
    }

    // epilogue: GEGLU in-warp, cross-warp per-32-ff quantize via plane,
    // fq/fs written at the BM32 rows (pair map) - v2 layout downstream.
    float* plane = (float*)pd_g2_sh;
    #pragma unroll
    for (uint32_t q = 0; q < 4u; ++q) {
        const uint32_t tokr = ((q & 2u) ? 8u : 0u) + g;
        const uint32_t ffc = wrow + 2u * t + (q & 1u);
        const uint32_t r = row_base + ffc;
        const bool pad = tok[tokr] == PD_MOE_PAD;
        const float gv = acc_g[q];
        const float uv = acc_u[q];
        float out = 0.f;
        if (!pad && r < ff) {
            out = GELU
                ? 0.5f * gv
                      * (1.0f
                         + tanhf(0.79788456080286535587989211986876f * gv
                                 * (1.0f + 0.044715f * gv * gv)))
                      * uv
                : (gv / (1.0f + __expf(-gv))) * uv;
        }
        plane[tokr * 64u + ffc] = out;
    }
    __syncthreads();
    const uint32_t n_sb = ff / 32u;
    {
        const uint32_t unit = warp * 4u + (lane / 8u);
        const uint32_t tokr = unit / 2u, grp = unit & 1u;
        const uint32_t l8 = lane & 7u;
        if (tok[tokr] != PD_MOE_PAD) {
            const uint32_t ff0 = grp * 32u;
            float a = 0.f;
            #pragma unroll
            for (uint32_t v = 0; v < 4u; ++v)
                a = fmaxf(a, fabsf(plane[tokr * 64u + ff0 + l8 * 4u + v]));
            #pragma unroll
            for (uint32_t o = 1; o < 8u; o *= 2u)
                a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
            const float scl = a * (1.0f / 127.0f);
            const float invs = scl > 0.f ? 1.0f / scl : 0.f;
            const size_t row =
                pmap[(size_t)tok[tokr] * n_active + slt[tokr]];   // bm32 row
            const uint32_t rgrp = row_base + ff0;
            if (rgrp < ff) {
                #pragma unroll
                for (uint32_t v = 0; v < 4u; ++v) {
                    const float ov = plane[tokr * 64u + ff0 + l8 * 4u + v];
                    int qi = __float2int_rn(ov * invs);
                    qi = qi < -127 ? -127 : (qi > 127 ? 127 : qi);
                    fq[row * ff + rgrp + l8 * 4u + v] = (int8_t)qi;
                }
                if (l8 == 0) fs[row * n_sb + (rgrp / 32u)] = scl;
            }
        }
    }
#else
    (void)gmap; (void)umap; (void)gate_scale; (void)up_scale; (void)sorted_row;
    (void)sorted_slot; (void)block_expert; (void)pmap; (void)xq; (void)xs; (void)fq;
    (void)fs; (void)in_dim; (void)ff; (void)n_active;
#endif
}

// g2 launcher (slot 504): bm must be 16; NotSupported without the host map
// builders; resolver NULLs below cc 9.
PD_EXPORT
int pd_q8_0_moe_gate_up_g2_geglu(const void* gate_data, const void* gate_scale,
                                 const void* up_data, const void* up_scale,
                                 const void* sorted_row, const void* sorted_slot,
                                 const void* block_expert, const void* pmap,
                                 const void* xq, const void* xs, void* fq, void* fs,
                                 uint32_t in_dim, uint32_t ff, uint32_t n_expert,
                                 uint32_t n_active, uint32_t max_blocks, uint32_t bm,
                                 void* stream) {
#if defined(PD_BS_HOST) || defined(PD_TC5_HOST)
    if (ff == 0 || max_blocks == 0) return 0;
    if (bm != 16u) return cudaErrorNotSupported;
    if ((in_dim & 255u) != 0 || (ff & 63u) != 0) return cudaErrorInvalidValue;
    CUtensorMap gmap, umap;
    if (!pd_tmap_2d_h64(&gmap, gate_data, in_dim, (uint64_t)n_expert * ff) ||
        !pd_tmap_2d_h64(&umap, up_data, in_dim, (uint64_t)n_expert * ff))
        return cudaErrorNotSupported;
    constexpr uint32_t S = 2u;
    constexpr uint32_t smem = PD_G2_STRIDE * S;
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute((const void*)pd_q8_0_moe_gate_up_g2_kernel<S, true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        attr = true;
    }
    dim3 grid(max_blocks, (ff + PD_G2_BN - 1u) / PD_G2_BN);
    pd_q8_0_moe_gate_up_g2_kernel<S, true><<<grid, 256, smem,
        (cudaStream_t)stream>>>(
        gmap, umap, (const __half*)gate_scale, (const __half*)up_scale,
        (const unsigned int*)sorted_row, (const unsigned int*)sorted_slot,
        (const unsigned int*)block_expert, (const unsigned int*)pmap,
        (const int8_t*)xq, (const float*)xs, (int8_t*)fq, (float*)fs, in_dim, ff,
        n_active);
    return pd_launch_status();
#else
    (void)gate_data; (void)gate_scale; (void)up_data; (void)up_scale; (void)sorted_row;
    (void)sorted_slot; (void)block_expert; (void)pmap; (void)xq; (void)xs; (void)fq;
    (void)fs; (void)in_dim; (void)ff; (void)n_expert; (void)n_active; (void)max_blocks;
    (void)bm; (void)stream;
    return cudaErrorNotSupported;
#endif
}

#else  // shipped-config gate
PD_EXPORT
int pd_q8_0_moe_gate_up_g2_geglu(const void*, const void*, const void*, const void*,
                                 const void*, const void*, const void*, const void*,
                                 const void*, const void*, void*, void*, uint32_t,
                                 uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                                 void*) {
    return cudaErrorNotSupported;
}
#endif  // g2 shipped-config gate
