// nemotron_vl.cuh - Nemotron 3.5 Super VL's picture lane (slots 904-907): the
// processor's resize and patchify on the device, the tower's position field,
// and the projector's squared ReLU. The RADIO blocks themselves ride Parse's
// tower kernels (LN, the f16 rings, the q/v-bias split, row-ranged
// bidirectional attention, the residual + LN seams) and the projector the
// bf16 GEMM; what is here is the glue those do not cover.
// Textually-included segment of the single pack translation unit.
// Not standalone-compilable: include order is defined by ../pack.cu.
//
// The reference is the checkpoint's own image processor and RADIO model:
//
// 904 pd_nvl_resample - one separable pass of torch's FLOAT antialiased
//     bicubic (`interpolate(..., "bicubic", align_corners=False,
//     antialias=True)` on the u8 picture widened to f32): per output its taps
//     and f32 weights (the plan comes with the call), width pass first, an f32
//     intermediate, the products summed tap by tap in f32 from the first. The
//     processor deliberately does NOT use PIL here (its kernel differs and the
//     difference survives the whole stack) - neither do we.
// 905 pd_nvl_patchify_h - (v / 255 - mean) / std in f32, the 16 x 16 patch
//     flattened (c, ky, kx), one f16 row a patch for the patch GEMM.
// 906 pd_nvl_pos_add - RADIO's cropped position embedding: the 128 x 128
//     learned grid resized bilinearly (align_corners=False, no antialias) to
//     M x M, M the tile's longer side (it can pass 128: a 4000 x 3000 picture
//     tiles 132 x 100), and its top-left (hp, wp) corner added
//     to the patch rows. torch's operation order: the source index
//     scale * (dst + 0.5) - 0.5 clamped at 0, the height lerp outside the
//     width lerp.
// 907 pd_nvl_relu2 - relu(x)^2 in place, the projector's activation.
//
// Every output element is a fixed function of its own inputs - no reduction
// crosses rows - so a picture's rows are the same bits alone or in a pass.

// ------------------------------------------------------------------ 904
// horiz: src [lines][in][C] -> dst [lines][out][C], taps along a row;
// else:  src [in][lines][C] -> dst [out][lines][C], taps down a column.
__global__ void pd_nvl_resample_kernel(const void* __restrict__ src, float* __restrict__ dst,
                                       uint32_t src_u8, uint32_t lines, uint32_t in,
                                       uint32_t out, uint32_t horiz,
                                       const uint32_t* __restrict__ xmin,
                                       const uint32_t* __restrict__ xsize,
                                       const float* __restrict__ w, uint32_t taps, uint32_t ch) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t)lines * out * ch;
    if (i >= total) return;
    const uint32_t c = (uint32_t)(i % ch);
    const uint32_t o = (uint32_t)((i / ch) % out);
    const uint32_t l = (uint32_t)(i / ((uint64_t)ch * out));
    const uint32_t x0 = xmin[o], n = xsize[o];
    const float* wo = w + (size_t)o * taps;
    const uint8_t* s8 = (const uint8_t*)src;
    const float* sf = (const float*)src;
    float acc = 0.0f;
    for (uint32_t j = 0; j < n; ++j) {
        const uint64_t at = horiz ? ((uint64_t)l * in + (x0 + j)) * ch + c
                                  : ((uint64_t)(x0 + j) * lines + l) * ch + c;
        const float v = src_u8 ? (float)s8[at] : sf[at];
        const float p = __fmul_rn(v, wo[j]);
        acc = j == 0 ? p : __fadd_rn(acc, p);
    }
    const uint64_t dat = horiz ? ((uint64_t)l * out + o) * ch + c : ((uint64_t)o * lines + l) * ch + c;
    dst[dat] = acc;
}

PD_EXPORT
int pd_nvl_resample(const void* src, void* dst, uint32_t src_u8, uint32_t lines, uint32_t in,
                    uint32_t out, uint32_t horiz, const void* xmin, const void* xsize,
                    const void* w, uint32_t taps, uint32_t ch, void* stream) {
    if (lines == 0 || out == 0 || ch == 0) return 0;
    if (in == 0 || taps == 0) return cudaErrorInvalidValue;
    const uint64_t total = (uint64_t)lines * out * ch;
    const uint32_t blocks = (uint32_t)((total + 255u) / 256u);
    pd_nvl_resample_kernel<<<blocks, 256, 0, (cudaStream_t)stream>>>(
        src, (float*)dst, src_u8, lines, in, out, horiz, (const uint32_t*)xmin,
        (const uint32_t*)xsize, (const float*)w, taps, ch);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 905
// img [h][w][3] f32 (0..255 scale) -> out [(h/p)(w/p)][3 p p] f16, patch rows
// row-major over the patch grid, each flattened (c, ky, kx).
__global__ void pd_nvl_patchify_h_kernel(const float* __restrict__ img, __half* __restrict__ out,
                                         uint32_t w, uint32_t p, float m0, float m1, float m2,
                                         float s0, float s1, float s2) {
    const uint32_t patch = blockIdx.x;
    const uint32_t pw = w / p;
    const uint32_t py = patch / pw, px = patch % pw;
    const uint32_t k = 3u * p * p;
    for (uint32_t t = threadIdx.x; t < k; t += blockDim.x) {
        const uint32_t c = t / (p * p);
        const uint32_t ky = (t % (p * p)) / p, kx = t % p;
        const float v = img[((size_t)(py * p + ky) * w + (px * p + kx)) * 3u + c];
        const float mean = c == 0 ? m0 : (c == 1 ? m1 : m2);
        const float std = c == 0 ? s0 : (c == 1 ? s1 : s2);
        out[(size_t)patch * k + t] = __float2half_rn(__fdiv_rn(__fsub_rn(__fdiv_rn(v, 255.0f), mean), std));
    }
}

PD_EXPORT
int pd_nvl_patchify_h(const void* img, void* out, uint32_t h, uint32_t w, uint32_t p, float m0,
                      float m1, float m2, float s0, float s1, float s2, void* stream) {
    if (h == 0 || w == 0) return 0;
    if (p == 0 || (h % p) != 0u || (w % p) != 0u) return cudaErrorInvalidValue;
    const uint32_t n = (h / p) * (w / p);
    pd_nvl_patchify_h_kernel<<<n, 256, 0, (cudaStream_t)stream>>>(
        (const float*)img, (__half*)out, w, p, m0, m1, m2, s0, s1, s2);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 906
// x [hp * wp][d] += grid resized g x g -> m x m, cropped to its top-left
// (hp, wp). One block a patch row.
__global__ void pd_nvl_pos_add_kernel(float* __restrict__ x, const float* __restrict__ grid,
                                      uint32_t g, uint32_t m, uint32_t wp, uint32_t d) {
    const uint32_t r = blockIdx.x;
    const uint32_t y = r / wp, xx = r % wp;
    const float scale = __fdiv_rn((float)g, (float)m);
    float sy = __fsub_rn(__fmul_rn(scale, __fadd_rn((float)y, 0.5f)), 0.5f);
    float sx = __fsub_rn(__fmul_rn(scale, __fadd_rn((float)xx, 0.5f)), 0.5f);
    sy = sy < 0.0f ? 0.0f : sy;
    sx = sx < 0.0f ? 0.0f : sx;
    const uint32_t y0 = (uint32_t)sy, x0 = (uint32_t)sx;
    const uint32_t y1 = y0 + (y0 < g - 1u ? 1u : 0u);
    const uint32_t x1 = x0 + (x0 < g - 1u ? 1u : 0u);
    const float ly1 = __fsub_rn(sy, (float)y0), ly0 = __fsub_rn(1.0f, ly1);
    const float lx1 = __fsub_rn(sx, (float)x0), lx0 = __fsub_rn(1.0f, lx1);
    const float* g00 = grid + ((size_t)y0 * g + x0) * d;
    const float* g01 = grid + ((size_t)y0 * g + x1) * d;
    const float* g10 = grid + ((size_t)y1 * g + x0) * d;
    const float* g11 = grid + ((size_t)y1 * g + x1) * d;
    float* xr = x + (size_t)r * d;
    for (uint32_t c = threadIdx.x; c < d; c += blockDim.x) {
        const float top = __fadd_rn(__fmul_rn(lx0, g00[c]), __fmul_rn(lx1, g01[c]));
        const float bot = __fadd_rn(__fmul_rn(lx0, g10[c]), __fmul_rn(lx1, g11[c]));
        const float v = __fadd_rn(__fmul_rn(ly0, top), __fmul_rn(ly1, bot));
        xr[c] = __fadd_rn(xr[c], v);
    }
}

PD_EXPORT
int pd_nvl_pos_add(void* x, const void* grid, uint32_t g, uint32_t m, uint32_t hp, uint32_t wp,
                   uint32_t d, void* stream) {
    if (hp == 0 || wp == 0) return 0;
    if (g == 0 || m == 0 || hp > m || wp > m || d == 0) return cudaErrorInvalidValue;
    pd_nvl_pos_add_kernel<<<hp * wp, 256, 0, (cudaStream_t)stream>>>(
        (float*)x, (const float*)grid, g, m, wp, d);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 907
__global__ void pd_nvl_relu2_kernel(float* __restrict__ x, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float r = x[i] > 0.0f ? x[i] : 0.0f;
    x[i] = __fmul_rn(r, r);
}

PD_EXPORT
int pd_nvl_relu2(void* x, uint64_t n, void* stream) {
    if (n == 0) return 0;
    const uint32_t blocks = (uint32_t)((n + 255u) / 256u);
    pd_nvl_relu2_kernel<<<blocks, 256, 0, (cudaStream_t)stream>>>((float*)x, n);
    return pd_launch_status();
}
