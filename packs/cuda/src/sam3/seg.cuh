// sam3/seg.cuh - SAM 3's segmentation head glue: the pixel decoder's nearest-x2-plus-skip seam and GroupNorm + ReLU
// Textually-included segment of the single pack translation unit.
// Not standalone-compilable: include order is defined by ../pack.cu.
// ------------------------------------------------------------ sam3 seg head
// Meta's PixelDecoder (3 stages, nearest): starting from the prompt-attended
// encoder memory at 72x72, twice: up x2 (nearest) + the next FPN level ->
// conv3x3 -> GroupNorm(8) -> ReLU, ending at 288x288 x 256 - the plane every
// query's mask is a dot product against. The convs are the dense lane's im2row
// + GEMM; these are the two passes around them.
//
// Needs dense_pred.cuh (pd_dp_gn_reduce_kernel, pd_dp_gn_fold_kernel, PD_DP_GN_Q).

// 767: out[p][Y][X] = f16(skip[p][Y][X] + prev[p][Y/2][X/2]) - F.interpolate
// nearest by exactly 2 picks pixel floor(dst / 2), so it is an index, no
// weights. prev f16 [pics][h][w][C], skip f32 [pics][2h][2w][C] (an FPN level).
__global__ void pd_sam3_up2_add_h_kernel(const __half* __restrict__ prev,
                                         const float* __restrict__ skip, __half* __restrict__ out,
                                         uint32_t h, uint32_t w, uint32_t C) {
    const uint32_t o = blockIdx.x;
    const uint32_t W2 = 2u * w, Pout = 4u * h * w;
    const uint32_t pic = o / Pout, pix = o - pic * Pout;
    const uint32_t Y = pix / W2, X = pix - Y * W2;
    const __half* pr = prev + (((size_t)pic * h + (Y >> 1)) * w + (X >> 1)) * C;
    const float* sr = skip + (size_t)o * C;
    __half* orow = out + (size_t)o * C;
    for (uint32_t c = threadIdx.x; c < C; c += blockDim.x)
        orow[c] = __float2half(sr[c] + __half2float(pr[c]));
}

PD_EXPORT
int pd_sam3_up2_add_h(const void* prev, const void* skip, void* out, uint32_t pics, uint32_t h,
                      uint32_t w, uint32_t C, void* stream) {
    if (pics == 0 || h == 0 || w == 0 || C == 0) return 0;
    const uint32_t nth = C < 256u ? ((C + 31u) & ~31u) : 256u;
    pd_sam3_up2_add_h_kernel<<<pics * 4u * h * w, nth, 0, (cudaStream_t)stream>>>(
        (const __half*)prev, (const float*)skip, (__half*)out, h, w, C);
    return pd_launch_status();
}

__global__ void pd_sam3_gn_relu_f16_kernel(const float* __restrict__ x,
                                           const float* __restrict__ xb,
                                           const float* __restrict__ mean,
                                           const float* __restrict__ inv,
                                           const float* __restrict__ w,
                                           const float* __restrict__ b, __half* __restrict__ out,
                                           uint32_t P, uint32_t C, uint32_t cg, uint32_t G,
                                           uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint64_t pix = i / C;
    const uint32_t c = (uint32_t)(i - pix * C);
    const size_t s = (size_t)(pix / P) * G + c / cg;
    const float y = ((x[i] + xb[c]) - mean[s]) * inv[s] * w[c] + b[c];
    out[i] = __float2half(fmaxf(y, 0.0f));
}

// 766: 613 with ReLU in place of GELU - same arguments, same scratch, the same
// two-pass statistics in the same order (the reduce and fold kernels are 613's).
PD_EXPORT
int pd_sam3_gn_relu_f16(const void* x, const void* xb, const void* w, const void* b, void* out,
                        void* part, void* stat, uint32_t chips, uint32_t P, uint32_t C, uint32_t G,
                        float eps, void* stream) {
    if (chips == 0 || P == 0 || C == 0) return 0;
    if (G == 0 || C % G != 0 || C > 1024u) return cudaErrorInvalidValue;
    const cudaStream_t st = (cudaStream_t)stream;
    const uint32_t cg = C / G;
    const uint32_t nchunk = (P + PD_DP_GN_Q - 1u) / PD_DP_GN_Q;
    const uint32_t nth = (C + 31u) & ~31u;
    const uint32_t rblocks = chips * nchunk;
    const double inv_n = 1.0 / ((double)P * (double)cg);
    float* mean = (float*)stat;
    float* inv = (float*)stat + (size_t)chips * G;
    pd_dp_gn_reduce_kernel<false><<<rblocks, nth, 0, st>>>(
        (const float*)x, (const float*)xb, nullptr, (float*)part, P, C, cg, G, nchunk);
    pd_dp_gn_fold_kernel<<<chips * G, 256, 0, st>>>((const float*)part, mean, nchunk, C, cg, G,
                                                    inv_n, eps, 0u);
    pd_dp_gn_reduce_kernel<true><<<rblocks, nth, 0, st>>>(
        (const float*)x, (const float*)xb, mean, (float*)part, P, C, cg, G, nchunk);
    pd_dp_gn_fold_kernel<<<chips * G, 256, 0, st>>>((const float*)part, inv, nchunk, C, cg, G,
                                                    inv_n, eps, 1u);
    const uint64_t n = (uint64_t)chips * P * C;
    pd_sam3_gn_relu_f16_kernel<<<(uint32_t)((n + 255ull) / 256ull), 256, 0, st>>>(
        (const float*)x, (const float*)xb, mean, inv, (const float*)w, (const float*)b,
        (__half*)out, P, C, cg, G, n);
    return pd_launch_status();
}
