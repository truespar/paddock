// gemma4u.cuh - Gemma 4 12B's encoder-free media embedders (slots 862-864).
//
// The 12B ("unified") has no vision or audio tower: a picture's 48 px
// patches and a clip's 40 ms frames are projected straight into the
// residual. The projections are three plain GEMMs on slot 722 (the F32-class
// stored-weight GEMM - every weight in the file is bf16-exact, f32-widened
// or not); these kernels are the norms around them:
//
//   862 patches  the resized u8 picture -> one 6912-wide row a patch, in the
//                file's CHW order (c * 2304 + y * 48 + x - the converter
//                permuted patch_dense's columns and patch_ln1 to match
//                ggml's im2col), rescaled by the processor's 1/255 factor,
//                then patch_ln1 (LayerNorm with bias, torch's 1e-5)
//   863 post     the patch GEMM's rows (bias already in): patch_ln2, the
//                factorized position (x table + y table, summed first, as
//                the reference sums its two axes), pos_norm, then the
//                multimodal embedder's weightless RMS norm (1e-6) - the
//                final projection's input
//   864 audio    raw 16 kHz samples -> 640-sample frames (the last one
//                zero-padded), each through the weightless RMS norm
//
// F32 class: every row statistic is a double-float sum (pd_df, width-stable),
// divisions and square roots correctly rounded. Rows are independent, so a
// picture's rows never depend on what shares the launch.
//
// Needs abi.cuh's pd_df helpers and f32_qkv's pd_launch_status.

#define PD_G4U_SIDE 48u                                      // model patch side (16 px x 3)
#define PD_G4U_PATCH (PD_G4U_SIDE * PD_G4U_SIDE * 3u)       // 6912
#define PD_G4U_FRAME 640u                                    // audio samples a token
// torch's `image * rescale_factor` with the processor's 0.00392156862745098
#define PD_G4U_RESCALE 0.00392156862745098f

// Block-wide sum of every thread's double-float, merged in warp order by
// thread 0; the total as a double. `ws` holds one entry a warp.
__device__ __forceinline__ double pd_g4u_block_sum(pd_df acc, pd_df* ws, double* s_tot) {
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    for (uint32_t s = 16; s > 0; s >>= 1) {
        pd_df o;
        o.hi = __shfl_down_sync(0xffffffffu, acc.hi, s);
        o.lo = __shfl_down_sync(0xffffffffu, acc.lo, s);
        acc = pd_df_merge(acc, o);
    }
    __syncthreads();   // ws / s_tot may still be read from the previous sum
    if ((tid & 31u) == 0) ws[tid >> 5] = acc;
    __syncthreads();
    if (tid == 0) {
        pd_df sum;
        sum.hi = 0.0f;
        sum.lo = 0.0f;
        for (uint32_t w = 0; w < ((nth + 31u) >> 5); ++w) sum = pd_df_merge(sum, ws[w]);
        *s_tot = (double)sum.hi + (double)sum.lo;
    }
    __syncthreads();
    return *s_tot;
}

// LayerNorm statistics of the R values a thread holds (n in the row):
// (mean, 1 / sqrt(var + eps)), var the biased one torch uses.
template <uint32_t R>
__device__ __forceinline__ float2 pd_g4u_ln_stats(const float (&v)[R], uint32_t n, float eps,
                                                  pd_df* ws, double* s_tot) {
    pd_df a;
    a.hi = 0.0f;
    a.lo = 0.0f;
    #pragma unroll
    for (uint32_t k = 0; k < R; ++k) pd_df_add(a, v[k]);
    const float mean = (float)(pd_g4u_block_sum(a, ws, s_tot) / (double)n);
    pd_df b;
    b.hi = 0.0f;
    b.lo = 0.0f;
    #pragma unroll
    for (uint32_t k = 0; k < R; ++k) {
        const float d = __fsub_rn(v[k], mean);
        pd_df_add(b, __fmul_rn(d, d));
    }
    const float var = (float)(pd_g4u_block_sum(b, ws, s_tot) / (double)n);
    return make_float2(mean, __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(var, eps))));
}

// ------------------------------------------------------------------ 862 patches
// One patch a block, 256 threads x 27 values: patch p = (p % gw, p / gw) of a
// [h][w][3] u8 picture (raster order - the processor's merged-patch order).
__global__ void __launch_bounds__(256)
pd_g4u_patches_kernel(const uint8_t* __restrict__ rgb, uint32_t w, uint32_t gw,
                      const float* __restrict__ lw, const float* __restrict__ lb,
                      float* __restrict__ out, float eps) {
    constexpr uint32_t R = PD_G4U_PATCH / 256u;
    __shared__ pd_df ws[8];
    __shared__ double s_tot;
    const uint32_t p = blockIdx.x, px = p % gw, py = p / gw, tid = threadIdx.x;
    constexpr uint32_t PP = PD_G4U_SIDE * PD_G4U_SIDE;
    float v[R];
    #pragma unroll
    for (uint32_t k = 0; k < R; ++k) {
        const uint32_t i = tid + k * 256u;
        const uint32_t c = i / PP, r = i % PP, y = r / PD_G4U_SIDE, x = r % PD_G4U_SIDE;
        const size_t at = ((size_t)(py * PD_G4U_SIDE + y) * w + px * PD_G4U_SIDE + x) * 3u + c;
        v[k] = __fmul_rn((float)rgb[at], PD_G4U_RESCALE);
    }
    const float2 st = pd_g4u_ln_stats<R>(v, PD_G4U_PATCH, eps, ws, &s_tot);
    float* dst = out + (size_t)p * PD_G4U_PATCH;
    #pragma unroll
    for (uint32_t k = 0; k < R; ++k) {
        const uint32_t i = tid + k * 256u;
        dst[i] = __fadd_rn(__fmul_rn(__fmul_rn(__fsub_rn(v[k], st.x), st.y), lw[i]), lb[i]);
    }
}

PD_EXPORT
int pd_g4u_patches(const void* rgb, uint32_t w, uint32_t gw, uint32_t n, const void* lnw,
                   const void* lnb, void* out, float eps, void* stream) {
    if (n == 0) return 0;
    if (gw == 0 || w < gw * PD_G4U_SIDE) return (int)cudaErrorInvalidValue;
    pd_g4u_patches_kernel<<<n, 256, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)rgb, w, gw, (const float*)lnw, (const float*)lnb, (float*)out, eps);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 863 post
// One row a block over `dim` (3840: 256 threads x 15) in place: patch_ln2,
// + (pos_x[px] + pos_y[py]) from the [2][pos_size][dim] table, pos_norm, then
// x * (mean(x^2) + eps_rms)^-1/2.
template <uint32_t R>
__global__ void __launch_bounds__(256)
pd_g4u_post_kernel(float* __restrict__ h, const float* __restrict__ w2,
                   const float* __restrict__ b2, const float* __restrict__ pos,
                   uint32_t pos_size, uint32_t gw, const float* __restrict__ w3,
                   const float* __restrict__ b3, uint32_t dim, float eps_ln, float eps_rms) {
    __shared__ pd_df ws[8];
    __shared__ double s_tot;
    const uint32_t row = blockIdx.x, px = row % gw, py = row / gw, tid = threadIdx.x;
    float* x = h + (size_t)row * dim;
    const float* tx = pos + (size_t)px * dim;
    const float* ty = pos + ((size_t)pos_size + py) * dim;
    float v[R];
    #pragma unroll
    for (uint32_t k = 0; k < R; ++k) v[k] = x[tid + k * 256u];
    float2 st = pd_g4u_ln_stats<R>(v, dim, eps_ln, ws, &s_tot);
    #pragma unroll
    for (uint32_t k = 0; k < R; ++k) {
        const uint32_t i = tid + k * 256u;
        const float n2 = __fadd_rn(__fmul_rn(__fmul_rn(__fsub_rn(v[k], st.x), st.y), w2[i]), b2[i]);
        v[k] = __fadd_rn(n2, __fadd_rn(tx[i], ty[i]));
    }
    st = pd_g4u_ln_stats<R>(v, dim, eps_ln, ws, &s_tot);
    pd_df sq;
    sq.hi = 0.0f;
    sq.lo = 0.0f;
    #pragma unroll
    for (uint32_t k = 0; k < R; ++k) {
        const uint32_t i = tid + k * 256u;
        v[k] = __fadd_rn(__fmul_rn(__fmul_rn(__fsub_rn(v[k], st.x), st.y), w3[i]), b3[i]);
        pd_df_add(sq, __fmul_rn(v[k], v[k]));
    }
    const float ms = (float)(pd_g4u_block_sum(sq, ws, &s_tot) / (double)dim);
    const float inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(ms, eps_rms)));
    #pragma unroll
    for (uint32_t k = 0; k < R; ++k) x[tid + k * 256u] = __fmul_rn(v[k], inv);
}

PD_EXPORT
int pd_g4u_post(void* h, const void* ln2w, const void* ln2b, const void* pos, uint32_t pos_size,
                uint32_t gw, const void* ln3w, const void* ln3b, uint32_t n, uint32_t dim,
                float eps_ln, float eps_rms, void* stream) {
    if (n == 0) return 0;
    // 3840 is the only width the checkpoint has; the per-thread count is a
    // template constant so the row lives in registers
    if (dim != 3840u || gw == 0 || n % gw != 0 || gw > pos_size || n / gw > pos_size)
        return (int)cudaErrorInvalidValue;
    pd_g4u_post_kernel<15u><<<n, 256, 0, (cudaStream_t)stream>>>(
        (float*)h, (const float*)ln2w, (const float*)ln2b, (const float*)pos, pos_size, gw,
        (const float*)ln3w, (const float*)ln3b, dim, eps_ln, eps_rms);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 864 audio
// One frame a block, 128 threads x 5 samples: samples [t*640, t*640 + 640)
// (zeros past the clip's end - the extractor's pad), weightless RMS norm.
__global__ void __launch_bounds__(128)
pd_g4u_audio_kernel(const float* __restrict__ s, uint32_t ns, float* __restrict__ out,
                    float eps) {
    constexpr uint32_t R = PD_G4U_FRAME / 128u;
    __shared__ pd_df ws[4];
    __shared__ double s_tot;
    const uint32_t t = blockIdx.x, tid = threadIdx.x;
    float v[R];
    pd_df sq;
    sq.hi = 0.0f;
    sq.lo = 0.0f;
    #pragma unroll
    for (uint32_t k = 0; k < R; ++k) {
        const size_t at = (size_t)t * PD_G4U_FRAME + tid + k * 128u;
        v[k] = at < ns ? s[at] : 0.0f;
        pd_df_add(sq, __fmul_rn(v[k], v[k]));
    }
    const float ms = (float)(pd_g4u_block_sum(sq, ws, &s_tot) / (double)PD_G4U_FRAME);
    const float inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(ms, eps)));
    float* dst = out + (size_t)t * PD_G4U_FRAME;
    #pragma unroll
    for (uint32_t k = 0; k < R; ++k) dst[tid + k * 128u] = __fmul_rn(v[k], inv);
}

PD_EXPORT
int pd_g4u_audio(const void* samples, uint32_t n_samples, uint32_t n_tok, void* out, float eps,
                 void* stream) {
    if (n_tok == 0) return 0;
    if ((uint64_t)n_tok * PD_G4U_FRAME < (uint64_t)n_samples) return (int)cudaErrorInvalidValue;
    pd_g4u_audio_kernel<<<n_tok, 128, 0, (cudaStream_t)stream>>>(
        (const float*)samples, n_samples, (float*)out, eps);
    return pd_launch_status();
}
