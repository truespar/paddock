// inkling2_vision.cuh - Inkling 2 Mini's picture tower, the three passes that
// are its own: the processor's patches, the embedding (bilinear position
// field + class row + ln_pre) and the q / k / v split with the axial rope.
// Everything else in a block rides the half tower interface (LN into f16,
// the f16 GEMMs, slot 620's attention, slot 622's residual + LayerScale +
// LN seam).
//
// A picture's rows, as this file lays them: its 4T native patches in
// TOKEN-major order (language-model token t = ty * cols + tx covers the
// 2 x 2 native patches (2ty + my, 2tx + mx), m = 2 my + mx), then its class
// row. That is the reference's patch order (`get_vision_position_ids` lays
// patches out block-major over the merge blocks) with the class token moved
// from the front to the back: attention without a mask is permutation
// equivariant, and with the class row last the adapter's 2 x 2 merge is the
// patch rows read as [T][4 x width] - no gather.

// slot 1011: the processor. No resize: the picture is padded to the
// 40-pixel cell grid with -1 on the 0..255 scale, rescaled by 1/255 and
// normalized with mean = std = 0.5 in the reference's own f32 order, then cut
// into native 20 x 20 patches - one row of (py, px, channel) values each,
// the patch embedding's input order. rgb is the decoded picture (w x h,
// interleaved 8-bit RGB, row-major); out [4 * trows * cols][1200] halves
// (the bf16 values the reference feeds its patch GEMM are all f16-exact, and
// these carry 3 more bits).
__global__ void __launch_bounds__(256) pd_inkling2_vis_patches_kernel(
        const uint8_t* rgb, uint32_t w, uint32_t h, uint32_t cols, __half* __restrict__ out) {
    constexpr uint32_t P = 20u, F = P * P * 3u;
    PD_PDL_ARM();
    const uint32_t r = blockIdx.x, t = r >> 2, m = r & 3u;
    const uint32_t y0 = (2u * (t / cols) + (m >> 1)) * P, x0 = (2u * (t % cols) + (m & 1u)) * P;
    const uint8_t* src = pd_pdl_after(rgb);
    __half* o = out + (size_t)r * F;
    for (uint32_t f = threadIdx.x; f < F; f += blockDim.x) {
        const uint32_t py = f / (P * 3u), px = (f % (P * 3u)) / 3u, c = f % 3u;
        const uint32_t y = y0 + py, x = x0 + px;
        const float u = (y < h && x < w) ? (float)src[((size_t)y * w + x) * 3u + c] : -1.0f;
        // the processor's rescale (the f32 of its 1/255) then normalize
        const float v = (u * 0.003921568859368563f - 0.5f) / 0.5f;
        o[f] = __float2half_rn(v);
    }
}

PD_EXPORT
int pd_inkling2_vis_patches(const void* rgb, uint32_t w, uint32_t h, uint32_t trows,
                            uint32_t cols, void* out, void* stream) {
    if (trows == 0 || cols == 0) return 0;
    if (w == 0 || h == 0 || (uint64_t)trows * 40u < h || (uint64_t)cols * 40u < w)
        return cudaErrorInvalidValue;
    pd_pdl_go(pd_inkling2_vis_patches_kernel, 4u * trows * cols, 256, 0u, (cudaStream_t)stream,
              (const uint8_t*)rgb, w, h, cols, (__half*)out);
    return pd_launch_status();
}

// One axis of the position field's resample: F.interpolate(bilinear,
// align_corners=False) from a `side`-long table onto `size` positions, as
// the reference spells it (`_interpolation_axis_taps_weights`): taps floor
// and floor + 1 clamped to the table, weights the hat kernel of the
// UNclamped distance (a clamped pair lands both taps on the edge row).
__device__ __forceinline__ void pd_ik2v_axis(uint32_t i, uint32_t size, uint32_t side,
                                             uint32_t (&tap)[2], float (&wt)[2]) {
    const float src = ((float)i + 0.5f) * (float)side / (float)size - 0.5f;
    const float fl = floorf(src);
    #pragma unroll
    for (int o = 0; o < 2; ++o) {
        const float raw = fl + (float)o;
        tap[o] = (uint32_t)fminf(fmaxf(raw, 0.0f), (float)(side - 1u));
        wt[o] = fmaxf(1.0f - fabsf(src - raw), 0.0f);
    }
}

// slot 1012: the embedding, in place on the residual x [4T + 1][1024] f32.
// Patch row r (token-major, see the top) holds its patch GEMM row; it gets
// the position field resampled onto the picture's native grid (2 trows x 2
// cols) at its (y, x). The class row (r = 4T) becomes the class embedding +
// its position embedding (`cls`, summed at load). Then ln_pre (two-pass f32
// statistics). A row's CTA reads all of it before writing any of it back.
// grid [side * side][1024] f32.
__global__ void __launch_bounds__(256) pd_inkling2_vis_embed_kernel(
        float* x, const float* __restrict__ grid, const float* __restrict__ cls,
        const float* __restrict__ lw, const float* __restrict__ lb, uint32_t trows,
        uint32_t cols, uint32_t side, float eps) {
    constexpr uint32_t D = 1024u, V = D / 256u;
    PD_PDL_ARM();
    __shared__ float red[8 * 1];
    const uint32_t r = blockIdx.x, tid = threadIdx.x, n_patch = 4u * trows * cols;
    // every row waits: the class row writes x too
    const float* pw = pd_pdl_after((const float*)x);
    float a[V];
    if (r == n_patch) {
        #pragma unroll
        for (uint32_t k = 0; k < V; ++k) a[k] = cls[tid + k * 256u];
    } else {
        const uint32_t t = r >> 2, m = r & 3u;
        const uint32_t y = 2u * (t / cols) + (m >> 1), xx = 2u * (t % cols) + (m & 1u);
        uint32_t ty[2], tx[2];
        float wy[2], wx[2];
        pd_ik2v_axis(y, 2u * trows, side, ty, wy);
        pd_ik2v_axis(xx, 2u * cols, side, tx, wx);
        const float* p = pw + (size_t)r * D;
        #pragma unroll
        for (uint32_t k = 0; k < V; ++k) {
            const uint32_t d = tid + k * 256u;
            // the four taps in the reference's (h0 w0, h0 w1, h1 w0, h1 w1)
            // order, each weight the product of its axes'
            float pos = 0.0f;
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                #pragma unroll
                for (int j = 0; j < 2; ++j)
                    pos += grid[((size_t)ty[i] * side + tx[j]) * D + d] * (wy[i] * wx[j]);
            a[k] = p[d] + pos;
        }
    }
    float s[1] = {0.0f};
    #pragma unroll
    for (uint32_t k = 0; k < V; ++k) s[0] += a[k];
    pd_ik2_block_sum<1>(s, 1, red);
    const float mean = s[0] / (float)D;
    float q[1] = {0.0f};
    #pragma unroll
    for (uint32_t k = 0; k < V; ++k) {
        a[k] -= mean;
        q[0] = fmaf(a[k], a[k], q[0]);
    }
    pd_ik2_block_sum<1>(q, 1, red);
    const float inv = rsqrtf(q[0] / (float)D + eps);
    float* o = x + (size_t)r * D;
    #pragma unroll
    for (uint32_t k = 0; k < V; ++k) {
        const uint32_t d = tid + k * 256u;
        o[d] = fmaf(a[k] * inv, lw[d], lb[d]);
    }
}

PD_EXPORT
int pd_inkling2_vis_embed(void* x, const void* grid, const void* cls, const void* lw,
                          const void* lb, uint32_t trows, uint32_t cols, uint32_t side, float eps,
                          void* stream) {
    if (trows == 0 || cols == 0) return 0;
    if (side < 2u) return cudaErrorInvalidValue;
    pd_pdl_go(pd_inkling2_vis_embed_kernel, 4u * trows * cols + 1u, 256, 0u, (cudaStream_t)stream,
              (float*)x, (const float*)grid, (const float*)cls, (const float*)lw,
              (const float*)lb, trows, cols, side, eps);
    return pd_launch_status();
}

// slot 1013: a picture's fused q | k | v landing (f16, [4T + 1][3 * 1024], no
// bias) into slot 620's three half planes [row][16][64], every bias added
// (with a rope the k bias does NOT cancel: each key rotates by its own
// position, so a shared vector no longer shifts every score alike), q and k
// rotated, q scaled. The rope is the reference's axial one on interleaved
// pairs: pair i of a head turns by x * f_i for i < hd / 4 and y * f_(i - hd/4)
// above, f_j = theta^(-4j / hd) (f64, rounded to f32), at the patch's native
// (y + 1, x + 1) - the class row sits at (0, 0). The angle is the reference's
// f32 product, its cos / sin full precision.
__global__ void __launch_bounds__(256) pd_inkling2_vis_qkv_rope_kernel(
        const __half* qkv, const float* __restrict__ bq, const float* __restrict__ bk,
        const float* __restrict__ bv, __half* __restrict__ q, __half* __restrict__ k,
        __half* __restrict__ v, uint32_t trows, uint32_t cols, uint32_t hd, float theta,
        float qscale) {
    constexpr uint32_t D = 1024u;
    PD_PDL_ARM();
    // the hd / 4 frequencies once a CTA (f64 pow is a slow pipe here)
    __shared__ float fr[64];
    if (threadIdx.x < hd / 4u)
        fr[threadIdx.x] = (float)pow((double)theta, -(double)(4u * threadIdx.x) / (double)hd);
    __syncthreads();
    const uint32_t r = blockIdx.x, n_patch = 4u * trows * cols;
    float py = 0.0f, px = 0.0f;
    if (r < n_patch) {
        const uint32_t t = r >> 2, m = r & 3u;
        py = (float)(2u * (t / cols) + (m >> 1) + 1u);
        px = (float)(2u * (t % cols) + (m & 1u) + 1u);
    }
    const __half* src = pd_pdl_after(qkv) + (size_t)r * 3u * D;
    const size_t dst = (size_t)r * D;
    const uint32_t half = hd / 2u, quarter = hd / 4u;
    for (uint32_t pp = threadIdx.x; pp < D / 2u; pp += blockDim.x) {
        const uint32_t i = pp % half, e0 = 2u * pp, e1 = e0 + 1u;
        const float f = fr[i < quarter ? i : i - quarter];
        float s, c;
        sincosf((i < quarter ? px : py) * f, &s, &c);
        const float q0 = __half2float(src[e0]) + bq[e0], q1 = __half2float(src[e1]) + bq[e1];
        const float k0 = __half2float(src[D + e0]) + bk[e0];
        const float k1 = __half2float(src[D + e1]) + bk[e1];
        q[dst + e0] = __float2half_rn((q0 * c - q1 * s) * qscale);
        q[dst + e1] = __float2half_rn((q1 * c + q0 * s) * qscale);
        k[dst + e0] = __float2half_rn(k0 * c - k1 * s);
        k[dst + e1] = __float2half_rn(k1 * c + k0 * s);
        v[dst + e0] = __float2half_rn(__half2float(src[2u * D + e0]) + bv[e0]);
        v[dst + e1] = __float2half_rn(__half2float(src[2u * D + e1]) + bv[e1]);
    }
}

PD_EXPORT
int pd_inkling2_vis_qkv_rope(const void* qkv, const void* bq, const void* bk, const void* bv,
                             void* q, void* k, void* v, uint32_t trows, uint32_t cols,
                             uint32_t hd, float theta, float qscale, void* stream) {
    if (trows == 0 || cols == 0) return 0;
    if (hd == 0 || hd % 4u != 0 || hd > 256u || 1024u % hd != 0) return cudaErrorInvalidValue;
    pd_pdl_go(pd_inkling2_vis_qkv_rope_kernel, 4u * trows * cols + 1u, 256, 0u,
              (cudaStream_t)stream, (const __half*)qkv, (const float*)bq, (const float*)bk,
              (const float*)bv, (__half*)q, (__half*)k, (__half*)v, trows, cols, hd, theta,
              qscale);
    return pd_launch_status();
}
