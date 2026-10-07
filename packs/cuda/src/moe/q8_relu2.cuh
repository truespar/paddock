// moe/q8_relu2.cuh - nemotron_h_moe's sorted squared-relu Q8_0 expert lane.
// The family's experts (and its shared expert, served as a one-expert sorted
// MoE) carry a single up plane and relu(up)^2 - no gate matrix. Two classes
// over the same moe_align layout and fq/fs handshake: the dp4a sorted up
// (+ quantize_q8 by the caller) and its int8-MMA twin, which quantizes in
// its own epilogue and is bitwise to the dp4a pair. The token-batched and
// dec2 relu2 kernels stay beside their swiglu siblings in q8.cuh. Included
// right after q8.cuh: it uses that file's PD_QMOE_* tile constants and the
// mma gate_up launcher template.

// Sorted single-plane up + squared-relu (nemotron_h_moe: no gate matrix).
// Same tile shape/staging as q8.cuh's sorted gate_up kernel with one weight
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

// Squared-relu single-plane twin (nemotron_h_moe): the routed experts, and
// the shared expert as its one-expert sorted MoE. It replaces the dp4a
// pd_q8_0_moe_up_relu2_sorted + quantize_q8 pair BITWISE: the same exact
// int32 k32 dots, the same (w_scale * x_scale) * dot fold in ascending K,
// the same fmaxf/square and per-32 amax rounding, PAD rows exact zeros.
// in_dim needs only Q8-block granularity (nemotron's hidden 2688 is not a
// 256 multiple): every K-walk stage zero-fills past n_blocks, as q8.cuh's
// mma down launcher already relies on.
PD_EXPORT
int pd_q8_0_moe_up_relu2_mma(const void* up_data, const void* up_scale,
                             const void* sorted_row, const void* block_expert,
                             const void* xq, const void* xs, void* fq, void* fs,
                             uint32_t in_dim, uint32_t ff, uint32_t max_blocks,
                             uint32_t bm, void* stream) {
    if (ff == 0 || max_blocks == 0) return 0;
    if ((in_dim & 31u) != 0 || (ff & 31u) != 0) return cudaErrorInvalidValue;
    const int8_t* ud = (const int8_t*)up_data; const __half* us = (const __half*)up_scale;
    const unsigned int* sr = (const unsigned int*)sorted_row;
    const unsigned int* be = (const unsigned int*)block_expert;
    const int8_t* xqp = (const int8_t*)xq; const float* xsp = (const float*)xs;
    int8_t* fqp = (int8_t*)fq; float* fsp = (float*)fs;
    cudaStream_t st = (cudaStream_t)stream;
    if (bm >= 64u)
        return pd_launch_qmma_gu<64u, false, false, true>(nullptr, nullptr, ud, us, sr, be, xqp,
                                                          xsp, fqp, fsp, in_dim, ff,
                                                          max_blocks, st);
    return pd_launch_qmma_gu<32u, true, false, true>(nullptr, nullptr, ud, us, sr, be, xqp, xsp,
                                                     fqp, fsp, in_dim, ff, max_blocks, st);
}

