// embedding_gemma2_audio.cuh - EmbeddingGemma 2's audio tower (slots 813-819):
// Gemma 4's audio Conformer ("gemma4a", the USM recipe) from 16 kHz samples
// to rows of the 512-wide text input.
//
//   frontend (813)  log-mel: 320-sample periodic-Hann frames every 160
//                   samples (the clip left-padded by 160), zero-padded to a
//                   512-point FFT, MAGNITUDE spectrum, 128 HTK triangles over
//                   0-8 kHz (unit peak), ln(mel + 1e-3)
//   subsample (814) twice: conv 3x3 stride 2 pad 1 (1 -> 128 -> 32 channels,
//                   time x frequency), LayerNorm over channels (weight, no
//                   bias), ReLU; rows are [time][32 freq x 32 ch]; then the
//                   f32 input projection 1024 -> 1024 (the shared f32 GEMM)
//   12 blocks       FFN1 (half step) - attention - light conv - FFN2 (half
//                   step) - RMS norm; every linear but the relative-key one
//                   CLIPPED: clamp(W . clamp(x, in_lo, in_hi), out_lo, out_hi)
//   output          W . x + b (1024 -> 1536), weightless RMS norm (819), then
//                   the 1536 -> 512 projection into the text model
//
// The kernels here are the glue between the f16 tensor-core GEMMs (f16
// activations, f32 accumulate - the clamps bound every GEMM input to
// |x| < 33, far inside f16's range): 815 the residual/norm seam that also
// writes the next GEMM's clamped f16 input, 816 a clamped activation pass,
// 817 the attention, 818 the conv module's middle.
//
// Attention is chunked local attention in Hugging Face's blocked form
// (chunk 12, context_left 13): every query sees itself and the 11 rows
// before it, nothing after. Unblocked, that is a causal window of 12 keys
// per row, which is what 817 computes - the blocks are layout, not math.
// Scores are Transformer-XL's: q.k plus q.r[d], r[d] the relative-key
// projection of the sinusoid at distance d (computed once at load); q is
// scaled by head_dim^-0.5 / ln 2 and softplus(per_dim_scale) (the GGUF
// stores the softplus), k by ln(1 + e) / ln 2; then tanh softcap 50 and a
// softmax.
//
// References (read, not copied): transformers' modeling_gemma4.py /
// feature_extraction_gemma4.py (the graph and the frontend, the tie-breaker)
// and llama.cpp's gemma4a graph (tools/mtmd/models/gemma4a.cpp, the same-
// weights reference). One clip is encoded alone, so no batch-invariance
// contract applies; every reduction still runs in one fixed order.
//
// Needs f32_qkv's pd_launch_status.

#define PD_EG2A_W 1024u     // conformer width
#define PD_EG2A_HD 128u     // head dim (8 heads)
#define PD_EG2A_KEYS 12u    // keys a query sees: itself and 11 before it
#define PD_EG2A_MEL 128u
#define PD_EG2A_BINS 257u
#define PD_EG2A_HOP 160u
#define PD_EG2A_WIN 320u

// Fixed-order sum over a 256-thread block: butterfly within each warp, then
// the eight warp totals in index order.
__device__ __forceinline__ float pd_eg2a_sum256(float v, float* red) {
#pragma unroll
    for (uint32_t o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    __syncthreads();   // red may still hold the previous reduction's values
    if ((threadIdx.x & 31u) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float s = red[0];
#pragma unroll
    for (uint32_t i = 1; i < 8u; ++i) s += red[i];
    return s;
}

__device__ __forceinline__ float pd_eg2a_clamp(float v, float lo, float hi) {
    return fminf(fmaxf(v, lo), hi);
}

__device__ __forceinline__ float pd_eg2a_warp_sum(float v) {
#pragma unroll
    for (uint32_t o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// ------------------------------------------------------------------ 813 log-mel
// One frame a CTA. Frame f reads samples f * 160 - 160 .. f * 160 + 159
// (outside the clip is zero), times the periodic Hann window, in a 512-point
// radix-2 FFT in shared memory (F64-derived twiddles); magnitudes of the 257
// bins, each mel row's triangle over its nonzero span, then the log: the
// processor's ln(mel + 1e-3), or with `llama_floor` llama.cpp's
// ln(max(mel, 1e-3)) - a test switch, the reference's own frontend.
__global__ void __launch_bounds__(256)
pd_eg2a_mel_kernel(const float* __restrict__ pcm, uint32_t n, const float* __restrict__ window,
                   const float* __restrict__ fb, const uint2* __restrict__ spans,
                   const float2* __restrict__ tw, float* __restrict__ out, uint32_t llama_floor) {
    __shared__ float2 z[512];
    __shared__ float mag[PD_EG2A_BINS];
    const uint32_t f = blockIdx.x, tid = threadIdx.x;
    for (uint32_t i = tid; i < 512u; i += 256u) {
        float x = 0.f;
        if (i < PD_EG2A_WIN) {
            const int64_t p = (int64_t)f * PD_EG2A_HOP + i - PD_EG2A_HOP;
            if (p >= 0 && p < (int64_t)n) x = __fmul_rn(pcm[p], window[i]);
        }
        z[__brev(i) >> 23] = make_float2(x, 0.f);
    }
    __syncthreads();
#pragma unroll
    for (uint32_t size = 2u; size <= 512u; size <<= 1) {
        const uint32_t half = size >> 1, k = tid % half, a = tid / half * size + k, b = a + half;
        const float2 t = tw[k * (512u / size)], e = z[a], w = z[b];
        const float re = fmaf(t.x, w.x, -__fmul_rn(t.y, w.y));
        const float im = fmaf(t.y, w.x, __fmul_rn(t.x, w.y));
        z[a] = make_float2(__fadd_rn(e.x, re), __fadd_rn(e.y, im));
        z[b] = make_float2(__fsub_rn(e.x, re), __fsub_rn(e.y, im));
        __syncthreads();
    }
    for (uint32_t k = tid; k < PD_EG2A_BINS; k += 256u) {
        const float2 v = z[k];
        mag[k] = sqrtf(fmaf(v.x, v.x, __fmul_rn(v.y, v.y)));
    }
    __syncthreads();
    if (tid < PD_EG2A_MEL) {
        const uint2 span = spans[tid];
        const float* row = fb + (size_t)tid * PD_EG2A_BINS;
        float sum = 0.f;
        for (uint32_t k = span.x; k < span.y; ++k) sum = fmaf(row[k], mag[k], sum);
        out[(size_t)f * PD_EG2A_MEL + tid] =
            llama_floor ? logf(fmaxf(sum, 1e-3f)) : logf(__fadd_rn(sum, 1e-3f));
    }
}

PD_EXPORT
int pd_eg2a_mel(const void* pcm, uint32_t n, const void* window, const void* fb, const void* spans,
                const void* twiddle, void* out, uint32_t frames, uint32_t llama_floor,
                void* stream) {
    if (frames == 0) return 0;
    // the last frame's window ends at sample frames * 160 - 1
    if ((uint64_t)frames * PD_EG2A_HOP > (uint64_t)n + PD_EG2A_HOP)
        return (int)cudaErrorInvalidValue;
    pd_eg2a_mel_kernel<<<frames, 256, 0, (cudaStream_t)stream>>>(
        (const float*)pcm, n, (const float*)window, (const float*)fb, (const uint2*)spans,
        (const float2*)twiddle, (float*)out, llama_floor);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 814 subsampling
// One output (time, frequency) cell a CTA, a thread an output channel:
// conv 3x3 stride 2 pad 1 over [t][f][c_in] input (the 3x3 x c_in patch
// staged once), weights [c_out][c_in][3][3] as torch stores them; then
// torch's LayerNorm over the c_out channels (mean, biased variance, weight,
// no bias) and ReLU, into [t][f][c_out].
template <uint32_t COUT>
__global__ void __launch_bounds__(COUT)
pd_eg2a_sscp_kernel(const float* __restrict__ in, const float* __restrict__ w,
                    const float* __restrict__ nw, float* __restrict__ out, uint32_t t_in,
                    uint32_t f_in, uint32_t c_in, uint32_t f_out, float eps) {
    extern __shared__ float patch[];   // [9][c_in]
    __shared__ float red[COUT / 32u];
    const uint32_t fo = blockIdx.x, to = blockIdx.y, co = threadIdx.x;
    for (uint32_t i = co; i < 9u * c_in; i += COUT) {
        const uint32_t kk = i / c_in, ci = i % c_in;
        const int ti = 2 * (int)to - 1 + (int)(kk / 3u), fi = 2 * (int)fo - 1 + (int)(kk % 3u);
        patch[i] = ti >= 0 && ti < (int)t_in && fi >= 0 && fi < (int)f_in
                       ? in[((size_t)ti * f_in + fi) * c_in + ci]
                       : 0.f;
    }
    __syncthreads();
    const float* wr = w + (size_t)co * c_in * 9u;
    float acc = 0.f;
    for (uint32_t ci = 0; ci < c_in; ++ci) {
#pragma unroll
        for (uint32_t kk = 0; kk < 9u; ++kk) acc = fmaf(wr[ci * 9u + kk], patch[kk * c_in + ci], acc);
    }
    auto sum = [&](float v) {
        v = pd_eg2a_warp_sum(v);
        __syncthreads();
        if ((co & 31u) == 0) red[co >> 5] = v;
        __syncthreads();
        float s = red[0];
#pragma unroll
        for (uint32_t i = 1; i < COUT / 32u; ++i) s += red[i];
        return s;
    };
    const float mean = sum(acc) / (float)COUT;
    const float d = acc - mean;
    const float var = sum(d * d) / (float)COUT;
    const float y = d * rsqrtf(var + eps) * nw[co];
    out[((size_t)to * f_out + fo) * COUT + co] = fmaxf(y, 0.f);
}

PD_EXPORT
int pd_eg2a_sscp(const void* in, const void* w, const void* nw, void* out, uint32_t t_in,
                 uint32_t f_in, uint32_t c_in, uint32_t c_out, float eps, void* stream) {
    if (t_in == 0) return 0;
    const uint32_t t_out = (t_in - 1u) / 2u + 1u, f_out = (f_in - 1u) / 2u + 1u;
    const size_t smem = (size_t)9u * c_in * sizeof(float);
    if (f_in == 0 || c_in == 0 || smem > 48u * 1024u) return (int)cudaErrorInvalidValue;
    const dim3 grid(f_out, t_out);
    if (c_out == 128u) {
        pd_eg2a_sscp_kernel<128u><<<grid, 128, smem, (cudaStream_t)stream>>>(
            (const float*)in, (const float*)w, (const float*)nw, (float*)out, t_in, f_in, c_in,
            f_out, eps);
    } else if (c_out == 32u) {
        pd_eg2a_sscp_kernel<32u><<<grid, 32, smem, (cudaStream_t)stream>>>(
            (const float*)in, (const float*)w, (const float*)nw, (float*)out, t_in, f_in, c_in,
            f_out, eps);
    } else {
        return (int)cudaErrorInvalidValue;
    }
    return pd_launch_status();
}

// ------------------------------------------------------------------ 815 the seam
// One 1024-wide row a CTA, thread t holding columns 4t..4t+3. In order, each
// step optional:
//   y given:     v = clamp(y, y_lo, y_hi); with post_w v = rms(v) * post_w;
//                x += y_scale * v           (a sublayer's residual)
//   out_w given: x = rms(x) * out_w         (the block's closing norm)
//   x16 given:   x16 = f16(clamp(norm ? rms(x) * next_w : x, n_lo, n_hi))
//                (the next GEMM's input; next_w null = weightless)
// rms(v) = v * (mean(v^2) + eps)^-0.5, the reference's form.
__device__ __forceinline__ float4 pd_eg2a_rms4(float4 v, const float* __restrict__ w, float eps,
                                               float* red) {
    const float ss = pd_eg2a_sum256(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w, red);
    const float r = 1.0f / sqrtf(ss / (float)PD_EG2A_W + eps);
    v = make_float4(v.x * r, v.y * r, v.z * r, v.w * r);
    if (w) {
        const float4 g = reinterpret_cast<const float4*>(w)[threadIdx.x];
        v = make_float4(v.x * g.x, v.y * g.y, v.z * g.z, v.w * g.w);
    }
    return v;
}

__global__ void __launch_bounds__(256)
pd_eg2a_rows_kernel(float* __restrict__ x, const float* __restrict__ y, float y_lo, float y_hi,
                    const float* __restrict__ post_w, float y_scale,
                    const float* __restrict__ out_w, uint32_t next_norm,
                    const float* __restrict__ next_w, float n_lo, float n_hi,
                    __half* __restrict__ x16, float eps) {
    __shared__ float red[8];
    const size_t at = (size_t)blockIdx.x * PD_EG2A_W / 4u + threadIdx.x;
    float4 xv = reinterpret_cast<float4*>(x)[at];
    if (y) {
        float4 v = reinterpret_cast<const float4*>(y)[at];
        v = make_float4(pd_eg2a_clamp(v.x, y_lo, y_hi), pd_eg2a_clamp(v.y, y_lo, y_hi),
                        pd_eg2a_clamp(v.z, y_lo, y_hi), pd_eg2a_clamp(v.w, y_lo, y_hi));
        if (post_w) v = pd_eg2a_rms4(v, post_w, eps, red);
        xv = make_float4(xv.x + v.x * y_scale, xv.y + v.y * y_scale, xv.z + v.z * y_scale,
                         xv.w + v.w * y_scale);
    }
    if (out_w) xv = pd_eg2a_rms4(xv, out_w, eps, red);
    if (y || out_w) reinterpret_cast<float4*>(x)[at] = xv;
    if (x16) {
        float4 u = next_norm ? pd_eg2a_rms4(xv, next_w, eps, red) : xv;
        __half2* o = reinterpret_cast<__half2*>(x16) + at * 2u;
        o[0] = __floats2half2_rn(pd_eg2a_clamp(u.x, n_lo, n_hi), pd_eg2a_clamp(u.y, n_lo, n_hi));
        o[1] = __floats2half2_rn(pd_eg2a_clamp(u.z, n_lo, n_hi), pd_eg2a_clamp(u.w, n_lo, n_hi));
    }
}

PD_EXPORT
int pd_eg2a_rows(void* x, const void* y, float y_lo, float y_hi, const void* post_w,
                 float y_scale, const void* out_w, uint32_t next_norm, const void* next_w,
                 float n_lo, float n_hi, void* x16, uint32_t rows, float eps, void* stream) {
    if (rows == 0) return 0;
    pd_eg2a_rows_kernel<<<rows, 256, 0, (cudaStream_t)stream>>>(
        (float*)x, (const float*)y, y_lo, y_hi, (const float*)post_w, y_scale,
        (const float*)out_w, next_norm, (const float*)next_w, n_lo, n_hi, (__half*)x16, eps);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 816 activation
// A clipped linear's output into the next one's input: clamp, optionally
// SiLU (x / (1 + e^-x), torch's form), clamp to the next input range, f16.
__global__ void pd_eg2a_act_kernel(const float* __restrict__ y, __half* __restrict__ out,
                                   uint64_t total, float lo, float hi, uint32_t silu, float lo2,
                                   float hi2) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float v = pd_eg2a_clamp(y[i], lo, hi);
    if (silu) v = v / (1.0f + expf(-v));
    out[i] = __float2half_rn(pd_eg2a_clamp(v, lo2, hi2));
}

PD_EXPORT
int pd_eg2a_act(const void* y, void* out, uint64_t total, float lo, float hi, uint32_t silu,
                float lo2, float hi2, void* stream) {
    if (total == 0) return 0;
    pd_eg2a_act_kernel<<<(uint32_t)((total + 255u) / 256u), 256, 0, (cudaStream_t)stream>>>(
        (const float*)y, (__half*)out, total, lo, hi, silu, lo2, hi2);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 817 attention
// One warp a (row, head), a CTA a row's eight heads; lane l holds the head's
// dims 4l..4l+3. qkv is the fused projection's raw landing [rows][3 x 1024];
// each of q, k, v is clamped to its linear's output range first. rel is the
// layer's relative keys [13][1024], row p at distance 12 - p. Out: the
// output projection's clamped f16 input.
struct PdEg2aAttnArgs {
    float q_lo, q_hi, k_lo, k_hi, v_lo, v_hi, o_lo, o_hi, q_scale, k_scale, cap;
};

__global__ void __launch_bounds__(256)
pd_eg2a_attn_kernel(const float* __restrict__ qkv, const float* __restrict__ rel,
                    const float* __restrict__ pds, __half* __restrict__ out, PdEg2aAttnArgs a) {
    const uint32_t row = blockIdx.x, h = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t col = h * PD_EG2A_HD + lane * 4u;
    const size_t ld = 3u * PD_EG2A_W;
    const float4 qr = *reinterpret_cast<const float4*>(qkv + row * ld + col);
    const float4 sc = *reinterpret_cast<const float4*>(pds + lane * 4u);
    // the reference's order: clamp, times q_scale, times softplus(per_dim)
    const float4 q = make_float4(pd_eg2a_clamp(qr.x, a.q_lo, a.q_hi) * a.q_scale * sc.x,
                                 pd_eg2a_clamp(qr.y, a.q_lo, a.q_hi) * a.q_scale * sc.y,
                                 pd_eg2a_clamp(qr.z, a.q_lo, a.q_hi) * a.q_scale * sc.z,
                                 pd_eg2a_clamp(qr.w, a.q_lo, a.q_hi) * a.q_scale * sc.w);
    float s[PD_EG2A_KEYS];
    float m = -INFINITY;
#pragma unroll
    for (uint32_t d = 0; d < PD_EG2A_KEYS; ++d) {
        s[d] = -INFINITY;
        if (d > row) continue;   // before the clip: masked
        const float4 kr = *reinterpret_cast<const float4*>(qkv + (row - d) * ld + PD_EG2A_W + col);
        const float4 k = make_float4(pd_eg2a_clamp(kr.x, a.k_lo, a.k_hi) * a.k_scale,
                                     pd_eg2a_clamp(kr.y, a.k_lo, a.k_hi) * a.k_scale,
                                     pd_eg2a_clamp(kr.z, a.k_lo, a.k_hi) * a.k_scale,
                                     pd_eg2a_clamp(kr.w, a.k_lo, a.k_hi) * a.k_scale);
        const float4 r = *reinterpret_cast<const float4*>(rel + (12u - d) * PD_EG2A_W + col);
        // content and position terms reduce apart, then add (matrix_ac + matrix_bd)
        const float ac = pd_eg2a_warp_sum(q.x * k.x + q.y * k.y + q.z * k.z + q.w * k.w);
        const float bd = pd_eg2a_warp_sum(q.x * r.x + q.y * r.y + q.z * r.z + q.w * r.w);
        s[d] = tanhf((ac + bd) / a.cap) * a.cap;
        m = fmaxf(m, s[d]);
    }
    float l = 0.f;
#pragma unroll
    for (uint32_t d = 0; d < PD_EG2A_KEYS; ++d) {
        s[d] = d > row ? 0.f : expf(s[d] - m);
        l += s[d];
    }
    // probabilities first, then the weighted values oldest key first (the
    // reference's softmax-then-matmul over its key axis)
    float4 o = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int d = (int)PD_EG2A_KEYS - 1; d >= 0; --d) {
        if ((uint32_t)d > row) continue;
        const float p = s[d] / l;
        const float4 vr =
            *reinterpret_cast<const float4*>(qkv + (row - d) * ld + 2u * PD_EG2A_W + col);
        o.x = fmaf(p, pd_eg2a_clamp(vr.x, a.v_lo, a.v_hi), o.x);
        o.y = fmaf(p, pd_eg2a_clamp(vr.y, a.v_lo, a.v_hi), o.y);
        o.z = fmaf(p, pd_eg2a_clamp(vr.z, a.v_lo, a.v_hi), o.z);
        o.w = fmaf(p, pd_eg2a_clamp(vr.w, a.v_lo, a.v_hi), o.w);
    }
    __half2* dst = reinterpret_cast<__half2*>(out + (size_t)row * PD_EG2A_W + col);
    dst[0] = __floats2half2_rn(pd_eg2a_clamp(o.x, a.o_lo, a.o_hi), pd_eg2a_clamp(o.y, a.o_lo, a.o_hi));
    dst[1] = __floats2half2_rn(pd_eg2a_clamp(o.z, a.o_lo, a.o_hi), pd_eg2a_clamp(o.w, a.o_lo, a.o_hi));
}

PD_EXPORT
int pd_eg2a_attn(const void* qkv, const void* rel, const void* pds, void* out, uint32_t rows,
                 const float* lims, float q_scale, float k_scale, float cap, void* stream) {
    if (rows == 0) return 0;
    const PdEg2aAttnArgs a{lims[0], lims[1], lims[2], lims[3], lims[4], lims[5],
                           lims[6], lims[7], q_scale, k_scale, cap};
    pd_eg2a_attn_kernel<<<rows, 256, 0, (cudaStream_t)stream>>>(
        (const float*)qkv, (const float*)rel, (const float*)pds, (__half*)out, a);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 818 light conv
// The conv module's middle, one row a CTA (thread t, channels 4t..4t+3):
// GLU of the clamped pointwise landing g [rows][2048] (first half times
// sigmoid of the second), the causal depthwise conv (kernel 5, taps on this
// row and the four before it, zero before the clip; dw [1024][5]), RMS norm
// with weight, SiLU, clamped into the second pointwise linear's f16 input.
__global__ void __launch_bounds__(256)
pd_eg2a_conv_kernel(const float* __restrict__ g, const float* __restrict__ dw,
                    const float* __restrict__ nw, __half* __restrict__ out, float g_lo,
                    float g_hi, float n_lo, float n_hi, float eps) {
    __shared__ float red[8];
    const uint32_t t = blockIdx.x, c0 = threadIdx.x * 4u;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (uint32_t k = 0; k < 5u; ++k) {
        if (t + k < 4u) continue;
        const float* gr = g + (size_t)(t + k - 4u) * 2u * PD_EG2A_W;
        const float4 a4 = *reinterpret_cast<const float4*>(gr + c0);
        const float4 b4 = *reinterpret_cast<const float4*>(gr + PD_EG2A_W + c0);
        const float av[4] = {a4.x, a4.y, a4.z, a4.w}, bv[4] = {b4.x, b4.y, b4.z, b4.w};
#pragma unroll
        for (uint32_t i = 0; i < 4u; ++i) {
            const float b = pd_eg2a_clamp(bv[i], g_lo, g_hi);
            const float glu = pd_eg2a_clamp(av[i], g_lo, g_hi) * (1.0f / (1.0f + expf(-b)));
            acc[i] = fmaf(dw[(c0 + i) * 5u + k], glu, acc[i]);
        }
    }
    float4 v = pd_eg2a_rms4(make_float4(acc[0], acc[1], acc[2], acc[3]), nw, eps, red);
    const float vv[4] = {v.x, v.y, v.z, v.w};
    float o[4];
#pragma unroll
    for (uint32_t i = 0; i < 4u; ++i) o[i] = pd_eg2a_clamp(vv[i] / (1.0f + expf(-vv[i])), n_lo, n_hi);
    __half2* dst = reinterpret_cast<__half2*>(out + (size_t)t * PD_EG2A_W + c0);
    dst[0] = __floats2half2_rn(o[0], o[1]);
    dst[1] = __floats2half2_rn(o[2], o[3]);
}

PD_EXPORT
int pd_eg2a_conv(const void* g, const void* dw, const void* nw, void* out, uint32_t rows,
                 float g_lo, float g_hi, float n_lo, float n_hi, float eps, void* stream) {
    if (rows == 0) return 0;
    pd_eg2a_conv_kernel<<<rows, 256, 0, (cudaStream_t)stream>>>(
        (const float*)g, (const float*)dw, (const float*)nw, (__half*)out, g_lo, g_hi, n_lo,
        n_hi, eps);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 819 output norm
// The tower's last seam, one 1536-wide row a CTA: y + bias, then the
// embedder's weightless RMS norm, f16 for the projection into the text model.
__global__ void __launch_bounds__(256)
pd_eg2a_out_kernel(const float* __restrict__ y, const float* __restrict__ bias,
                   __half* __restrict__ out, uint32_t n, float eps) {
    __shared__ float red[8];
    const float* yr = y + (size_t)blockIdx.x * n;
    float v[6], ss = 0.f;
#pragma unroll
    for (uint32_t i = 0; i < 6u; ++i) {
        const uint32_t c = threadIdx.x + 256u * i;
        v[i] = c < n ? yr[c] + bias[c] : 0.f;
        ss += v[i] * v[i];
    }
    const float r = 1.0f / sqrtf(pd_eg2a_sum256(ss, red) / (float)n + eps);
#pragma unroll
    for (uint32_t i = 0; i < 6u; ++i) {
        const uint32_t c = threadIdx.x + 256u * i;
        if (c < n) out[(size_t)blockIdx.x * n + c] = __float2half_rn(v[i] * r);
    }
}

PD_EXPORT
int pd_eg2a_out(const void* y, const void* bias, void* out, uint32_t rows, uint32_t n, float eps,
                void* stream) {
    if (rows == 0) return 0;
    if (n == 0 || n > 1536u) return (int)cudaErrorInvalidValue;
    pd_eg2a_out_kernel<<<rows, 256, 0, (cudaStream_t)stream>>>(
        (const float*)y, (const float*)bias, (__half*)out, n, eps);
    return pd_launch_status();
}
