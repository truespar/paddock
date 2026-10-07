// sam3/memory.cuh - SAM 3's memory encoder: the mask downsampler's small stride-2 convs, the fuser's depthwise 7x7 + LayerNorm2d, and the LayerNorm2d + GELU seam behind the wide convs
// Textually-included segment of the single pack translation unit.
// Not standalone-compilable: include order is defined by ../pack.cu.
// ------------------------------------------------------------ sam3 memory encoder
// Every frame the tracker writes one memory per object (Meta's
// SimpleMaskEncoder): the object's mask, made a plane in [-10, 10], runs down
// four Conv2d(k3, s2, p1) -> LayerNorm2d -> GELU stages (1 -> 4 -> 16 -> 64 ->
// 256 channels, 1152^2 -> 72^2) and a 1x1, is added to the frame's own 72^2
// feature (a 1x1 of it), goes through two ConvNeXt blocks and lands at 64
// channels. The two wide stages are the f16 ring's implicit conv (slot 785);
// the 1x1s and the blocks' MLPs are the dense lane's GEMMs. What is here is
// the rest: everything is channel-last f32 unless it feeds a GEMM, then f16.
//
// Needs f32_qkv.cuh's pd_launch_status.

// The input plane of the first stage at (Y, X) of an H x W grid, from an
// object's [sh][sw] logits: sigmoid (mode 1) or binarized (> 0, mode 2 -
// Meta's point-prompted frames), then * 20 - 10 (sigmoid_scale_for_mem_enc
// and its bias), then bilinear (align_corners false) from sh x sw to H x W
// when the two differ, the order Meta applies them in. Meta resizes with
// antialias on; going UP that is the plain bilinear, whose two taps and
// edge clamp are the antialiased filter's support exactly.
__device__ __forceinline__ float pd_sam3_mem_tf(float v, uint32_t mode) {
    const float p = mode == 2u ? (v > 0.0f ? 1.0f : 0.0f) : 1.0f / (1.0f + expf(-v));
    return p * 20.0f - 10.0f;
}
__device__ __forceinline__ float pd_sam3_mem_in(const float* __restrict__ m, uint32_t sh,
                                                uint32_t sw, uint32_t H, uint32_t W, uint32_t Y,
                                                uint32_t X, uint32_t mode) {
    if (sh == H && sw == W) return pd_sam3_mem_tf(m[(size_t)Y * sw + X], mode);
    const float fy = fmaxf((float)sh / (float)H * ((float)Y + 0.5f) - 0.5f, 0.0f);
    const float fx = fmaxf((float)sw / (float)W * ((float)X + 0.5f) - 0.5f, 0.0f);
    const uint32_t y0 = (uint32_t)fy, x0 = (uint32_t)fx;
    const uint32_t y1 = y0 + (y0 + 1u < sh ? 1u : 0u), x1 = x0 + (x0 + 1u < sw ? 1u : 0u);
    const float ly = fy - (float)y0, lx = fx - (float)x0;
    const float a = pd_sam3_mem_tf(m[(size_t)y0 * sw + x0], mode);
    const float b = pd_sam3_mem_tf(m[(size_t)y0 * sw + x1], mode);
    const float c = pd_sam3_mem_tf(m[(size_t)y1 * sw + x0], mode);
    const float d = pd_sam3_mem_tf(m[(size_t)y1 * sw + x1], mode);
    return (1.0f - ly) * ((1.0f - lx) * a + lx * b) + ly * ((1.0f - lx) * c + lx * d);
}

// 784: one small stage of the mask downsampler, Conv2d(k3, s2, p1) ->
// LayerNorm2d(eps) -> GELU(erf), for B objects. mode 0: in is channel-last
// f32 [B][H][W][cin]; modes 1 / 2: in is the objects' logits [B][sh][sw]
// (cin 1) made the stage's input on the fly (pd_sam3_mem_in) - the 1152^2
// plane is never written. Zero padding is of that input plane, as the conv
// sees it. w [COUT][cin][3][3] as Conv2d stores it; out [B][H/2][W/2][COUT]
// f32, or f16 when out16 (the implicit conv's source). One thread an output
// pixel: these are 4 and 16 channels.
template <uint32_t COUT>
__global__ void pd_sam3_mem_down3_kernel(const float* __restrict__ in, const float* __restrict__ w,
                                         const float* __restrict__ b, const float* __restrict__ lw,
                                         const float* __restrict__ lb, void* __restrict__ out,
                                         uint32_t nb, uint32_t H, uint32_t W, uint32_t cin,
                                         uint32_t mode, uint32_t sh, uint32_t sw, float eps,
                                         uint32_t out16) {
    const uint32_t oh = H / 2u, ow = W / 2u, P = oh * ow;
    const uint32_t o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= nb * P) return;
    const uint32_t ob = o / P, pix = o - ob * P;
    const uint32_t oy = pix / ow, ox = pix - oy * ow;
    float acc[COUT];
    #pragma unroll
    for (uint32_t co = 0; co < COUT; ++co) acc[co] = b[co];
    for (uint32_t ky = 0; ky < 3u; ++ky) {
        const int32_t yy = (int32_t)(2u * oy + ky) - 1;
        if (yy < 0 || yy >= (int32_t)H) continue;
        for (uint32_t kx = 0; kx < 3u; ++kx) {
            const int32_t xx = (int32_t)(2u * ox + kx) - 1;
            if (xx < 0 || xx >= (int32_t)W) continue;
            const uint32_t t = ky * 3u + kx;
            if (mode == 0u) {
                const float* src = in + (((size_t)ob * H + (uint32_t)yy) * W + (uint32_t)xx) * cin;
                for (uint32_t ci = 0; ci < cin; ++ci) {
                    const float v = src[ci];
                    #pragma unroll
                    for (uint32_t co = 0; co < COUT; ++co) acc[co] += w[(co * cin + ci) * 9u + t] * v;
                }
            } else {
                const float v = pd_sam3_mem_in(in + (size_t)ob * sh * sw, sh, sw, H, W, (uint32_t)yy,
                                               (uint32_t)xx, mode);
                #pragma unroll
                for (uint32_t co = 0; co < COUT; ++co) acc[co] += w[co * 9u + t] * v;
            }
        }
    }
    // LayerNorm2d over the channels as Meta writes it: mean, then the mean
    // square of the centred values, then divide by the root
    float mean = 0.0f;
    #pragma unroll
    for (uint32_t co = 0; co < COUT; ++co) mean += acc[co];
    mean /= (float)COUT;
    float var = 0.0f;
    #pragma unroll
    for (uint32_t co = 0; co < COUT; ++co) var += (acc[co] - mean) * (acc[co] - mean);
    const float den = sqrtf(var / (float)COUT + eps);
    #pragma unroll
    for (uint32_t co = 0; co < COUT; ++co) {
        float v = (acc[co] - mean) / den * lw[co] + lb[co];
        v = 0.5f * v * (1.0f + erff(v * 0.70710678118654752440084436210484f));
        if (out16) ((__half*)out)[(size_t)o * COUT + co] = __float2half_rn(v);
        else ((float*)out)[(size_t)o * COUT + co] = v;
    }
}

PD_EXPORT
int pd_sam3_mem_down3(const void* in, const void* w, const void* b, const void* lw, const void* lb,
                      void* out, uint32_t nb, uint32_t H, uint32_t W, uint32_t cin, uint32_t cout,
                      uint32_t mode, uint32_t sh, uint32_t sw, float eps, uint32_t out16,
                      void* stream) {
    if (nb == 0u || H < 2u || W < 2u) return 0;
    if ((H & 1u) != 0u || (W & 1u) != 0u || mode > 2u || cin == 0u) return cudaErrorInvalidValue;
    if (mode != 0u && (cin != 1u || sh == 0u || sw == 0u)) return cudaErrorInvalidValue;
    const uint64_t n = (uint64_t)nb * (H / 2u) * (W / 2u);
    if (n > 0xffffffffull) return cudaErrorInvalidValue;
    const uint32_t grid = (uint32_t)((n + 127u) / 128u);
    cudaStream_t st = (cudaStream_t)stream;
    const float *pi = (const float*)in, *pw = (const float*)w, *pb = (const float*)b,
                *plw = (const float*)lw, *plb = (const float*)lb;
    switch (cout) {
    case 4u:
        pd_sam3_mem_down3_kernel<4u><<<grid, 128, 0, st>>>(pi, pw, pb, plw, plb, out, nb, H, W, cin,
                                                           mode, sh, sw, eps, out16);
        break;
    case 16u:
        pd_sam3_mem_down3_kernel<16u><<<grid, 128, 0, st>>>(pi, pw, pb, plw, plb, out, nb, H, W, cin,
                                                            mode, sh, sw, eps, out16);
        break;
    default:
        return cudaErrorInvalidValue;
    }
    return pd_launch_status();
}

// LayerNorm over one row held one value a thread (blockDim.x == n, a
// multiple of 32, <= 1024): Meta's LayerNorm2d - the mean, the mean square
// of the centred values, a divide by the root. `red` is 32 floats of smem.
__device__ __forceinline__ float pd_sam3_row_ln(float v, float* red, float lwv, float lbv,
                                                float eps) {
    const uint32_t lane = threadIdx.x & 31u, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
    float s = v;
    for (uint32_t o = 16u; o > 0u; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (lane == 0u) red[wid] = s;
    __syncthreads();
    float tot = 0.0f;
    for (uint32_t i = 0; i < nw; ++i) tot += red[i];
    const float mean = tot / (float)blockDim.x;
    __syncthreads();
    const float c = v - mean;
    float q = c * c;
    for (uint32_t o = 16u; o > 0u; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
    if (lane == 0u) red[wid] = q;
    __syncthreads();
    float qt = 0.0f;
    for (uint32_t i = 0; i < nw; ++i) qt += red[i];
    return c / sqrtf(qt / (float)blockDim.x + eps) * lwv + lbv;
}

// 786: the ConvNeXt block's front, depthwise Conv2d(k7, p3, groups C) + bias
// -> LayerNorm2d(eps), for B objects: in f32 [B][H][W][C], w [49][C] (tap
// major, re-laid at load from [C][1][7][7]), out f16 [B][H][W][C] - the
// pointwise MLP's input. One block a pixel, a thread a channel, so every tap
// is one coalesced row read.
__global__ void pd_sam3_dwconv7_ln_h_kernel(const float* __restrict__ in, const float* __restrict__ w,
                                            const float* __restrict__ b, const float* __restrict__ lw,
                                            const float* __restrict__ lb, __half* __restrict__ out,
                                            uint32_t H, uint32_t W, float eps) {
    __shared__ float red[32];
    const uint32_t C = blockDim.x, c = threadIdx.x;
    const uint32_t P = H * W, ob = blockIdx.x / P, pix = blockIdx.x - ob * P;
    const uint32_t y = pix / W, x = pix - y * W;
    const float* src = in + (size_t)ob * P * C;
    float a = b[c];
    for (uint32_t ky = 0; ky < 7u; ++ky) {
        const int32_t yy = (int32_t)(y + ky) - 3;
        if (yy < 0 || yy >= (int32_t)H) continue;
        for (uint32_t kx = 0; kx < 7u; ++kx) {
            const int32_t xx = (int32_t)(x + kx) - 3;
            if (xx < 0 || xx >= (int32_t)W) continue;
            a += w[(ky * 7u + kx) * C + c] * src[((size_t)yy * W + (uint32_t)xx) * C + c];
        }
    }
    out[(size_t)blockIdx.x * C + c] = __float2half_rn(pd_sam3_row_ln(a, red, lw[c], lb[c], eps));
}

PD_EXPORT
int pd_sam3_dwconv7_ln_h(const void* in, const void* w, const void* b, const void* lw, const void* lb,
                         void* out, uint32_t nb, uint32_t H, uint32_t W, uint32_t C, float eps,
                         void* stream) {
    if (nb == 0u || H == 0u || W == 0u) return 0;
    if (C == 0u || C > 1024u || (C & 31u) != 0u) return cudaErrorInvalidValue;
    const uint64_t n = (uint64_t)nb * H * W;
    if (n > 0x7fffffffull) return cudaErrorInvalidValue;
    pd_sam3_dwconv7_ln_h_kernel<<<(uint32_t)n, C, 0, (cudaStream_t)stream>>>(
        (const float*)in, (const float*)w, (const float*)b, (const float*)lw, (const float*)lb,
        (__half*)out, H, W, eps);
    return pd_launch_status();
}

// 787: LayerNorm2d(eps) -> GELU(erf) over channel-last rows, f32 [rows][n]
// in, f16 out: the seam behind the downsampler's wide stages, whose conv
// lands f32 with its bias (slot 785) and whose next consumer is the ring
// again. n a multiple of 32, <= 1024; one block a row.
__global__ void pd_sam3_ln_gelu_h_kernel(const float* __restrict__ in, const float* __restrict__ lw,
                                         const float* __restrict__ lb, __half* __restrict__ out,
                                         float eps) {
    __shared__ float red[32];
    const uint32_t n = blockDim.x, c = threadIdx.x;
    const size_t i = (size_t)blockIdx.x * n + c;
    float v = pd_sam3_row_ln(in[i], red, lw[c], lb[c], eps);
    v = 0.5f * v * (1.0f + erff(v * 0.70710678118654752440084436210484f));
    out[i] = __float2half_rn(v);
}

PD_EXPORT
int pd_sam3_ln_gelu_h(const void* in, const void* lw, const void* lb, void* out, uint32_t rows,
                      uint32_t n, float eps, void* stream) {
    if (rows == 0u) return 0;
    if (n == 0u || n > 1024u || (n & 31u) != 0u || rows > 0x7fffffffu) return cudaErrorInvalidValue;
    pd_sam3_ln_gelu_h_kernel<<<rows, n, 0, (cudaStream_t)stream>>>(
        (const float*)in, (const float*)lw, (const float*)lb, (__half*)out, eps);
    return pd_launch_status();
}
