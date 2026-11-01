// phonon.cuh - Gradium Phonon 31M (FlowLM + the Mimi decoder) on CUDA, slots
// 908-915. Textually-included segment of the single pack translation unit.
// Not standalone-compilable: include order is defined by ../pack.cu.
//
// The checkpoint stores FlowLM in BF16 and Mimi in F32. Everything here is
// F32 class: a BF16 weight widens exactly, products, sums and activations
// are F32, and the multi-row passes (the prompt, the voice prompt, Mimi's
// transformer and convolutions) ride the pack's F32-class tensor-core GEMMs
// (722 for FlowLM's stored BF16 planes, 698's 3xTF32 for Mimi's F32 ones)
// with the staging kernels below. The gate is the model's F64 equations.
//
// The frame step is one row through FlowLM - ~55 MB of BF16 weights for 80
// ms of audio, a weight stream, not FLOPs. So its linears are GEMVs with
// the input LayerNorm (and the flow head's adaLN modulation) inside the
// product that consumes it, the residual / GELU / SiLU / gate - and, for the
// qkv projection, the rope and the K/V cache writes - inside the one that
// produces it; attention merges its key splits in its own last block. Every
// position a kernel reads comes from device memory (`pos_dev`), so captured
// graphs of whole frame steps replay with no host argument, and every kernel
// launches as a programmatic dependent with its weights loaded ahead of the
// wait. Hence the pointer rule here: ptxas treats a `const __restrict__`
// load as read-only for the kernel's lifetime and may schedule it above
// griddepcontrol.wait, so weights, norms and tables are `const __restrict__`
// - loading them early is the point - and anything a predecessor kernel
// writes is a plain pointer (the 2048-wide GEMV read its input before linear1
// had written it until it was).
//
// 908 pd_ph_linear     y = epi(W prologue(x) + b), a warp two outputs
// 909 pd_ph_rope_kv    q/k RoPE (interleaved pairs), k/v into a ring cache
// 910 pd_ph_attn       causal (optionally windowed) attention, split keys,
//                      the splits merged by the last block of a head
// 911 pd_ph_layernorm  LayerNorm rows (the wide passes' GEMM inputs)
// 912 pd_ph_point      elementwise ops (add, ELU, SiLU, denormalize, ...)
// 913 pd_ph_im2col     a causal conv's input rows, taps side by side
// 914 pd_ph_history    a conv's history update after a pass
// 915 pd_ph_upsample   Mimi's depthwise x16 upsample (width 32)
// 916 pd_ph_conv       a causal conv straight from its input rows (no im2col)
// 917 pd_ph_conv_gemm  a causal conv as the 3xTF32 GEMM over overlapping rows

#define PD_PH_LW 4u      // warps a linear block
#define PD_PH_LO 2u      // outputs a warp
#define PD_PH_SPLIT 64u  // keys an attention split holds
#define PD_PH_HD 64u     // head width (8 heads x 64 in both transformers)

__device__ __forceinline__ float pd_ph_gelu(float v) {
    return 0.5f * v * (1.0f + erff(v * 0.70710678118654752f));
}
__device__ __forceinline__ float pd_ph_silu(float v) { return v / (1.0f + expf(-v)); }
__device__ __forceinline__ float pd_ph_elu(float v) { return v > 0.0f ? v : expm1f(v); }

// RoPE frequencies 10000^(-2i / 64) of a 64-wide head's 32 pairs, each the
// correctly rounded f32: powf's few-ulp error is the same at every position,
// so the angle's error would grow with p (and an in-kernel double pow costs
// the frame step microseconds at GB10's FP64 rate).
__device__ __constant__ float pd_ph_rope_freq[32] = {
    0x1p+0f, 0x1.7ff222p-1f, 0x1.1feb34p-1f, 0x1.afd136p-2f,
    0x1.43d136p-2f, 0x1.e5a848p-3f, 0x1.6c310ep-3f, 0x1.111aeep-3f,
    0x1.99999ap-4f, 0x1.33281cp-4f, 0x1.ccab86p-5f, 0x1.59742ap-5f,
    0x1.030dc4p-5f, 0x1.8486ap-6f, 0x1.235a72p-6f, 0x1.b4f7e2p-7f,
    0x1.47ae14p-7f, 0x1.eb736p-8f, 0x1.708938p-8f, 0x1.145ceep-8f,
    0x1.9e7c6ep-9f, 0x1.36d21ap-9f, 0x1.d22a5p-10f, 0x1.5d931cp-10f,
    0x1.0624dep-10f, 0x1.892918p-11f, 0x1.26d42cp-11f, 0x1.ba2e4cp-12f,
    0x1.4b96bep-12f, 0x1.f15028p-13f, 0x1.74eea6p-13f, 0x1.17a8e4p-13f};
__device__ __forceinline__ float pd_ph_freq(uint32_t pair) { return pd_ph_rope_freq[pair]; }

// (v0, v1) rotated by position p's angle for pair `pair`.
__device__ __forceinline__ void pd_ph_rotate(float& v0, float& v1, uint32_t p, uint32_t pair) {
    float sn, cs;
    sincosf((float)p * pd_ph_freq(pair), &sn, &cs);
    const float a = v0 * cs - v1 * sn, b = v0 * sn + v1 * cs;
    v0 = a;
    v1 = b;
}

// Block-wide sum over 128 threads (4 warps), fixed order.
__device__ __forceinline__ float pd_ph_block_sum(float v, float* red) {
    for (uint32_t o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    __syncthreads();
    if ((threadIdx.x & 31u) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    return (red[0] + red[1]) + (red[2] + red[3]);
}

// A row's LayerNorm statistics over 128 threads: two passes (mean, then the
// centered squares), the exact 1 / sqrt.
__device__ __forceinline__ void pd_ph_row_stats(const float* xr, uint32_t in, float eps, float* red,
                                                float& mean, float& rstd) {
    float s = 0.0f;
    for (uint32_t k = threadIdx.x; k < in; k += 128u) s += xr[k];
    mean = pd_ph_block_sum(s, red) / (float)in;
    float v = 0.0f;
    for (uint32_t k = threadIdx.x; k < in; k += 128u) {
        const float d = xr[k] - mean;
        v += d * d;
    }
    rstd = 1.0f / sqrtf(pd_ph_block_sum(v, red) / (float)in + eps);
}

// The same statistics over a thread's register-held elements (zeros past
// the row) - the row read once, the two passes from registers, each
// thread's terms in the same order, so the result is pd_ph_row_stats' to
// the bit. (A one-pass form accumulating in double was probed: GB10's FP64
// rate made the reduction ~3 us slower than these two f32 ones.)
template <uint32_t E>
__device__ __forceinline__ void pd_ph_stats(const float (&xe)[E], uint32_t in, float eps, float* red,
                                            float& mean, float& rstd) {
    float s = 0.0f;
#pragma unroll
    for (uint32_t e = 0; e < E; ++e) s += xe[e];
    mean = pd_ph_block_sum(s, red) / (float)in;
    float v = 0.0f;
#pragma unroll
    for (uint32_t e = 0; e < E; ++e) {
        if (threadIdx.x + e * 128u >= in) break;
        const float d = xe[e] - mean;
        v += d * d;
    }
    rstd = 1.0f / sqrtf(pd_ph_block_sum(v, red) / (float)in + eps);
}

// ------------------------------------------------------------------ 908
// x [rows][in] f32 -> y [rows][out] f32 against W [out][in] (bf16 or f32).
// pro: 0 x as is, 1 LayerNorm(x; ln_w, ln_b), 2 LayerNorm affine then
//      (1 + scale) * . + shift, 3 LayerNorm without affine then the same
//      modulation (shift = mod, scale = mod + in).
// epi: 0 v, 1 res + v, 2 gelu_erf(v), 3 silu(v), 4 res + gate * v (gate per
//      output: the adaLN gate), 5 the qkv projection's rope and cache
//      writes: q rotated into y ([rows][512]), k rotated and v into the
//      caches kc / vc [cap][512] at slot p % cap, p = *pos_dev + pos0 + r.
//      Outputs from `esplit` on take epilogue 0 (a head that rides along:
//      the EOS logit beside cond_embed's SiLU outputs).
// hint: the weights' L2 policy (0 default, 1 evict_first, 2 evict_last).
// A block is one row: it stages the row (after the prologue) in shared
// memory, and its 4 warps take output pairs, `kw` warps to a pair splitting
// the row's 8-wide chunks between them (16 B BF16 loads), each warp a fixed
// xor tree, the pair's warps then summed in warp order. kw depends on the
// shape only (the host picks it so a narrow projection still fills the
// die), so an output's sequence is the same at any row count.
// A warp's first two chunks of weights are loaded into registers BEFORE the
// wait on the producer of x: no kernel writes a weight, and with every pass
// a programmatic dependent the chain's weight streams then overlap instead
// of each pass paying its own DRAM latency after its predecessor.
template <bool BF16, uint32_t E, bool AFF>
__global__ void __launch_bounds__(PD_PH_LW * 32u, E <= 4u ? 8 : 4) pd_ph_linear_kernel(
    const float* x, const float* __restrict__ ln_w, const float* __restrict__ ln_b,
    const float* mod, const void* __restrict__ w, const float* __restrict__ bias,
    const float* res, const float* gate, float* y, uint32_t in, uint32_t out, float eps,
    uint32_t pro, uint32_t epi, float* kc, float* vc,
    const uint32_t* __restrict__ pos_dev, uint32_t pos0, uint32_t cap, uint32_t kw,
    uint32_t esplit, uint32_t hint) {
    using Chunk = typename std::conditional<BF16, uint4, float4[2]>::type;
    extern __shared__ float4 pd_ph_sx4[];
    float* sx = reinterpret_cast<float*>(pd_ph_sx4);
    __shared__ float red[4];
    __shared__ float s_part[PD_PH_LW][PD_PH_LO];
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u, r = blockIdx.y;
    const uint32_t pairs = PD_PH_LW / kw, og = warp / kw, wk = warp % kw;
    const uint32_t o0 = blockIdx.x * pairs * PD_PH_LO + og * PD_PH_LO;
    const uint32_t chunks = in / 8u, c0 = wk * 32u + lane, cstep = 32u * kw;
    // the weights' L2 policy: a plane the next frame reads again stays
    // (evict_last), a streamed one makes room for it (evict_first)
    uint64_t pol = 0;
    if (hint == 1u) asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
    if (hint == 2u) asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
    // one 8-wide chunk of output o's row, k = 8 ch
    auto load = [&](Chunk& d, uint32_t o, uint32_t ch) {
        if constexpr (BF16) {
            const uint16_t* a = (const uint16_t*)w + (size_t)o * in + ch * 8u;
            if (hint != 0u)
                asm("ld.global.nc.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(d.x), "=r"(d.y), "=r"(d.z), "=r"(d.w)
                    : "l"(a), "l"(pol));
            else
                d = __ldg(reinterpret_cast<const uint4*>(a));
        } else {
            const float4* p = reinterpret_cast<const float4*>((const float*)w + (size_t)o * in + ch * 8u);
            d[0] = p[0];
            d[1] = p[1];
        }
    };
    auto fma8 = [&](float& acc, const Chunk& d, const float (&xv)[8]) {
        float wv[8];
        if constexpr (BF16) {
            const uint32_t qq[4] = {d.x, d.y, d.z, d.w};
#pragma unroll
            for (uint32_t e = 0; e < 4; ++e) {
                wv[2 * e] = __uint_as_float(qq[e] << 16);
                wv[2 * e + 1] = __uint_as_float(qq[e] & 0xffff0000u);
            }
        } else {
            wv[0] = d[0].x; wv[1] = d[0].y; wv[2] = d[0].z; wv[3] = d[0].w;
            wv[4] = d[1].x; wv[5] = d[1].y; wv[6] = d[1].z; wv[7] = d[1].w;
        }
#pragma unroll
        for (uint32_t e = 0; e < 8; ++e) acc = fmaf(wv[e], xv[e], acc);
    };
    Chunk pre[2][PD_PH_LO];
#pragma unroll
    for (uint32_t c = 0; c < 2; ++c)
#pragma unroll
        for (uint32_t j = 0; j < PD_PH_LO; ++j)
            if (c0 + c * cstep < chunks && o0 + j < out) load(pre[c][j], o0 + j, c0 + c * cstep);
    // the row's elements a thread normalizes: k = tid + 128 e (E of them,
    // in <= 128 E), and the norm's own (immutable) planes, also ahead of
    // the wait
    float xe[E], lw[AFF ? E : 1], lb[AFF ? E : 1];
    if constexpr (AFF) {
#pragma unroll
        for (uint32_t e = 0; e < E; ++e) {
            const uint32_t k = threadIdx.x + e * PD_PH_LW * 32u;
            lw[e] = k < in ? ln_w[k] : 1.0f;
            lb[e] = k < in ? ln_b[k] : 0.0f;
        }
    }
    PD_PDL_ARM();
    const float* xr = x + (size_t)r * in;
#pragma unroll
    for (uint32_t e = 0; e < E; ++e) {
        const uint32_t k = threadIdx.x + e * PD_PH_LW * 32u;
        xe[e] = k < in ? xr[k] : 0.0f;
    }
    if (pro != 0u) {
        float mean, rstd;
        pd_ph_stats(xe, in, eps, red, mean, rstd);
#pragma unroll
        for (uint32_t e = 0; e < E; ++e) {
            const uint32_t k = threadIdx.x + e * PD_PH_LW * 32u;
            if (k >= in) break;
            float v = (xe[e] - mean) * rstd;
            if constexpr (AFF) v = v * lw[e] + lb[e];
            if (pro >= 2u) v = v * (1.0f + mod[in + k]) + mod[k];
            sx[k] = v;
        }
    } else {
#pragma unroll
        for (uint32_t e = 0; e < E; ++e) {
            const uint32_t k = threadIdx.x + e * PD_PH_LW * 32u;
            if (k >= in) break;
            sx[k] = xe[e];
        }
    }
    __syncthreads();
    float acc[PD_PH_LO] = {0.0f, 0.0f};
    auto xchunk = [&](uint32_t ch, float (&xv)[8]) {
        const float4 xa = pd_ph_sx4[2u * ch], xb = pd_ph_sx4[2u * ch + 1u];
        xv[0] = xa.x; xv[1] = xa.y; xv[2] = xa.z; xv[3] = xa.w;
        xv[4] = xb.x; xv[5] = xb.y; xv[6] = xb.z; xv[7] = xb.w;
    };
#pragma unroll
    for (uint32_t c = 0; c < 2; ++c) {
        const uint32_t ch = c0 + c * cstep;
        if (ch >= chunks) break;
        float xv[8];
        xchunk(ch, xv);
#pragma unroll
        for (uint32_t j = 0; j < PD_PH_LO; ++j)
            if (o0 + j < out) fma8(acc[j], pre[c][j], xv);
    }
    // rows wider than two chunk strides finish after the wait
    for (uint32_t ch = c0 + 2u * cstep; ch < chunks; ch += cstep) {
        float xv[8];
        xchunk(ch, xv);
#pragma unroll
        for (uint32_t j = 0; j < PD_PH_LO; ++j) {
            if (o0 + j >= out) continue;
            Chunk d;
            load(d, o0 + j, ch);
            fma8(acc[j], d, xv);
        }
    }
#pragma unroll
    for (uint32_t j = 0; j < PD_PH_LO; ++j)
        for (uint32_t s = 16; s > 0; s >>= 1) acc[j] += __shfl_xor_sync(0xffffffffu, acc[j], s);
    if (kw > 1u) {
        if (lane == 0) {
            s_part[warp][0] = acc[0];
            s_part[warp][1] = acc[1];
        }
        __syncthreads();
        if (wk != 0u) return;
        for (uint32_t i = 1; i < kw; ++i) {
            acc[0] += s_part[warp + i][0];
            acc[1] += s_part[warp + i][1];
        }
    }
    if (lane != 0 || o0 >= out) return;
    float v[PD_PH_LO];
#pragma unroll
    for (uint32_t j = 0; j < PD_PH_LO; ++j) v[j] = acc[j] + (bias && o0 + j < out ? bias[o0 + j] : 0.0f);
    if (epi == 5u) {
        // o0 is even: (o0, o0 + 1) is one rope pair of one head
        const uint32_t p = (pos_dev ? *pos_dev : 0u) + pos0 + r, slot = p % cap;
        const uint32_t part = o0 / 512u, c = o0 % 512u;
        if (part < 2u) pd_ph_rotate(v[0], v[1], p, (c % PD_PH_HD) / 2u);
        float* dst = part == 0u ? y + (size_t)r * 512u : (part == 1u ? kc : vc) + (size_t)slot * 512u;
        dst[c] = v[0];
        dst[c + 1u] = v[1];
        return;
    }
#pragma unroll
    for (uint32_t j = 0; j < PD_PH_LO; ++j) {
        const uint32_t o = o0 + j;
        if (o >= out) break;
        const size_t at = (size_t)r * out + o;
        float z = v[j];
        switch (o < esplit ? epi : 0u) {
        case 1: z = res[at] + z; break;
        case 2: z = pd_ph_gelu(z); break;
        case 3: z = pd_ph_silu(z); break;
        case 4: z = res[at] + gate[o] * z; break;
        default: break;
        }
        y[at] = z;
    }
}

// The K split of a shape: as many of a block's warps on one output pair as
// the row has 256-wide chunk strides for, until the grid holds four blocks
// an SM of GB10's 48 (192) - wide projections stay one warp a pair.
static inline uint32_t pd_ph_linear_kw(uint32_t in, uint32_t out) {
    uint32_t kw = 1u;
    while (kw < PD_PH_LW && in / 256u >= 2u * kw &&
           (out + PD_PH_LO * PD_PH_LW / kw - 1u) / (PD_PH_LO * PD_PH_LW / kw) < 192u)
        kw *= 2u;
    return kw;
}

PD_EXPORT
int pd_ph_linear(const void* x, const void* ln_w, const void* ln_b, const void* mod,
                 const void* w, uint32_t w_bf16, const void* bias, const void* res,
                 const void* gate, void* y, uint32_t rows, uint32_t in, uint32_t out, float eps,
                 uint32_t pro, uint32_t epi, void* kc, void* vc, const void* pos_dev,
                 uint32_t pos0, uint32_t cap, uint32_t esplit, uint32_t hint, void* stream) {
    if (rows == 0 || out == 0) return 0;
    // 8-wide chunks of 16 B-aligned rows
    if (in == 0 || in % 8u || in > 2048u || pro > 3u || epi > 5u || ((uintptr_t)w & 15u) ||
        ((uintptr_t)x & 15u) || rows > 65535u)
        return cudaErrorInvalidValue;
    if ((pro == 1u || pro == 2u) && (!ln_w || !ln_b)) return cudaErrorInvalidValue;
    if (pro >= 2u && !mod) return cudaErrorInvalidValue;
    if ((epi == 1u || epi == 4u) && !res) return cudaErrorInvalidValue;
    if (epi == 4u && !gate) return cudaErrorInvalidValue;
    if (epi == 5u && (out != 1536u || !kc || !vc || cap == 0u)) return cudaErrorInvalidValue;
    if (hint > 2u) return cudaErrorInvalidValue;
    const uint32_t kw = pd_ph_linear_kw(in, out), per = PD_PH_LO * PD_PH_LW / kw;
    const uint32_t smem = in * 4u;
    const dim3 grid((out + per - 1u) / per, rows);
    const cudaStream_t s = (cudaStream_t)stream;
    const bool aff = pro == 1u || pro == 2u;
    // the instantiation for this row width and prologue: a row of at most
    // 128 E elements, the affine planes only when the norm has them
    auto go = [&](auto kern) {
        pd_pdl_go(kern, grid, dim3(PD_PH_LW * 32u), smem, s, (const float*)x, (const float*)ln_w,
                  (const float*)ln_b, (const float*)mod, w, (const float*)bias, (const float*)res,
                  (const float*)gate, (float*)y, in, out, eps, pro, epi, (float*)kc, (float*)vc,
                  (const uint32_t*)pos_dev, pos0, cap, kw, esplit, hint);
    };
#define PD_PH_LIN(B, E)                                                                       \
    (aff ? go(pd_ph_linear_kernel<B, E, true>) : go(pd_ph_linear_kernel<B, E, false>))
    if (w_bf16) {
        if (in <= 128u) PD_PH_LIN(true, 1u);
        else if (in <= 512u) PD_PH_LIN(true, 4u);
        else PD_PH_LIN(true, 16u);
    } else {
        if (in <= 128u) PD_PH_LIN(false, 1u);
        else if (in <= 512u) PD_PH_LIN(false, 4u);
        else PD_PH_LIN(false, 16u);
    }
#undef PD_PH_LIN
    return pd_launch_status();
}

// ------------------------------------------------------------------ 909
// qkv [rows][3 * 512] -> q [rows][512] (rotated), k/v rows into the ring
// caches [cap][512] at slot (p % cap), p = *pos_dev + pos0 + r. RoPE pairs
// adjacent channels (2i, 2i + 1) of each 64-wide head, angle p * 10000^(-2i
// / 64) - the same rotation as 908's epilogue.
__global__ void pd_ph_rope_kv_kernel(const float* qkv, float* q, float* kc, float* vc,
                                     const uint32_t* __restrict__ pos_dev, uint32_t pos0,
                                     uint32_t rows, uint32_t cap) {
    PD_PDL_ARM();
    const uint32_t r = blockIdx.x;
    if (r >= rows) return;
    const uint32_t p = (pos_dev ? *pos_dev : 0u) + pos0 + r;
    const uint32_t slot = p % cap;
    const float* src = qkv + (size_t)r * 1536u;
    for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) {
        // i = head * 32 + pair
        const uint32_t pair = i & 31u, c = 2u * i;
        float q0 = src[c], q1 = src[c + 1u], k0 = src[512u + c], k1 = src[512u + c + 1u];
        pd_ph_rotate(q0, q1, p, pair);
        pd_ph_rotate(k0, k1, p, pair);
        q[(size_t)r * 512u + c] = q0;
        q[(size_t)r * 512u + c + 1u] = q1;
        kc[(size_t)slot * 512u + c] = k0;
        kc[(size_t)slot * 512u + c + 1u] = k1;
        vc[(size_t)slot * 512u + c] = src[1024u + c];
        vc[(size_t)slot * 512u + c + 1u] = src[1024u + c + 1u];
    }
}

PD_EXPORT
int pd_ph_rope_kv(const void* qkv, void* q, void* kc, void* vc, const void* pos_dev,
                  uint32_t pos0, uint32_t rows, uint32_t cap, void* stream) {
    if (rows == 0) return 0;
    if (cap == 0) return cudaErrorInvalidValue;
    pd_pdl_go(pd_ph_rope_kv_kernel, dim3(rows), dim3(128), 0, (cudaStream_t)stream,
              (const float*)qkv, (float*)q, (float*)kc, (float*)vc, (const uint32_t*)pos_dev, pos0,
              rows, cap);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 910
// One block per (head, row, split): the split's keys [lo, hi) of the row's
// visible range [max(0, p + 1 - window), p] (window 0 = everything), 64
// keys a split (a frame's handful of rows still fills the die). Every K and
// V load of the block is issued up front (8 lanes a key row for the scores,
// a lane on two channels of each value row), so the split costs one memory
// latency, not two. Scores by a 3-step xor tree, softmax over the split in
// shared memory, P V in key order and a fixed 4-warp sum. The split's (max,
// sum, values) go to `part` [rows][8][nsplit][66]; the block drawing a
// head's last ticket merges the non-empty splits - online softmax, split
// order, the headers staged in parallel and the value reads independent -
// into out [rows][512] and re-arms the ticket.
__global__ void __launch_bounds__(128) pd_ph_attn_kernel(
    const float* q, const float* kc, const float* vc, float* part, float* out, uint32_t* ctr,
    const uint32_t* __restrict__ pos_dev,
    uint32_t pos0, uint32_t rows, uint32_t cap, uint32_t window, uint32_t nsplit) {
    constexpr uint32_t KW = PD_PH_SPLIT / 4u;  // keys a warp
    // the cache streams through L2 (evict_first), so the planes the frame
    // keeps resident there (908's evict_last) survive it
    uint64_t pol;
    asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
    auto ld4 = [&](const float* a) {
        float4 v;
        asm volatile("ld.global.L2::cache_hint.v4.f32 {%0, %1, %2, %3}, [%4], %5;"
                     : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
                     : "l"(a), "l"(pol));
        return v;
    };
    auto ld2 = [&](const float* a) {
        float2 v;
        asm volatile("ld.global.L2::cache_hint.v2.f32 {%0, %1}, [%2], %3;"
                     : "=f"(v.x), "=f"(v.y)
                     : "l"(a), "l"(pol));
        return v;
    };
    constexpr uint32_t PS = PD_PH_HD + 2u;     // a partial's stride
    PD_PDL_ARM();
    const uint32_t h = blockIdx.x, r = blockIdx.y, sp = blockIdx.z;
    const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
    const uint32_t p = (pos_dev ? *pos_dev : 0u) + pos0 + r;
    const uint32_t first = (window && p + 1u > window) ? p + 1u - window : 0u;
    const uint32_t nact = (p + 1u - first + PD_PH_SPLIT - 1u) / PD_PH_SPLIT;  // non-empty splits
    const uint32_t lo = first + sp * PD_PH_SPLIT;
    const uint32_t hi = min(lo + PD_PH_SPLIT, p + 1u);
    const uint32_t n = lo < hi ? hi - lo : 0u;
    __shared__ float s_p[PD_PH_SPLIT];
    __shared__ float red[4];
    __shared__ float s_acc[4][PD_PH_HD];
    __shared__ float s_m[64], s_f[64];
    __shared__ bool s_last;
    if (n != 0u) {
        const uint32_t g = lane >> 3, gl = lane & 7u;
        float4 ka[KW / 4u], kb[KW / 4u];
        float2 vv[KW];
#pragma unroll
        for (uint32_t it = 0; it < KW / 4u; ++it) {
            const uint32_t j = warp * KW + it * 4u + g;
            if (j < n) {
                const float* kp = kc + (size_t)((lo + j) % cap) * 512u + h * PD_PH_HD + gl * 8u;
                ka[it] = ld4(kp);
                kb[it] = ld4(kp + 4);
            } else {
                ka[it] = kb[it] = make_float4(0.f, 0.f, 0.f, 0.f);
            }
        }
#pragma unroll
        for (uint32_t jj = 0; jj < KW; ++jj) {
            const uint32_t j = warp * KW + jj;
            vv[jj] = j < n ? ld2(vc + (size_t)((lo + j) % cap) * 512u + h * PD_PH_HD + 2u * lane)
                           : make_float2(0.f, 0.f);
        }
        const float4* qp = reinterpret_cast<const float4*>(q + (size_t)r * 512u + h * PD_PH_HD + gl * 8u);
        const float4 qa = qp[0], qb = qp[1];
#pragma unroll
        for (uint32_t it = 0; it < KW / 4u; ++it) {
            const uint32_t j = warp * KW + it * 4u + g;
            float d = 0.0f;
            d = fmaf(qa.x, ka[it].x, d); d = fmaf(qa.y, ka[it].y, d);
            d = fmaf(qa.z, ka[it].z, d); d = fmaf(qa.w, ka[it].w, d);
            d = fmaf(qb.x, kb[it].x, d); d = fmaf(qb.y, kb[it].y, d);
            d = fmaf(qb.z, kb[it].z, d); d = fmaf(qb.w, kb[it].w, d);
            d += __shfl_xor_sync(0xffffffffu, d, 1);
            d += __shfl_xor_sync(0xffffffffu, d, 2);
            d += __shfl_xor_sync(0xffffffffu, d, 4);
            if (gl == 0) s_p[j] = j < n ? d * 0.125f : -INFINITY;  // 1 / sqrt(64)
        }
        __syncthreads();
        // block max, then the exponentials and their sum
        float m = tid < PD_PH_SPLIT ? s_p[tid] : -INFINITY;
        for (uint32_t o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        if (lane == 0) red[warp] = m;
        __syncthreads();
        m = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
        const float e = tid < n ? expf(s_p[tid] - m) : 0.0f;
        const float l = pd_ph_block_sum(e, red);
        if (tid < PD_PH_SPLIT) s_p[tid] = e;
        __syncthreads();
        // P V: warp w owns keys w * KW .. (past n the weight is 0)
        float2 acc = make_float2(0.0f, 0.0f);
#pragma unroll
        for (uint32_t jj = 0; jj < KW; ++jj) {
            const float pj = s_p[warp * KW + jj];
            acc.x = fmaf(pj, vv[jj].x, acc.x);
            acc.y = fmaf(pj, vv[jj].y, acc.y);
        }
        s_acc[warp][2u * lane] = acc.x;
        s_acc[warp][2u * lane + 1u] = acc.y;
        __syncthreads();
        float* pp = part + (((size_t)r * 8u + h) * nsplit + sp) * PS;
        if (tid < PD_PH_HD)
            pp[2u + tid] = (s_acc[0][tid] + s_acc[1][tid]) + (s_acc[2][tid] + s_acc[3][tid]);
        if (tid == 0) {
            pp[0] = m;
            pp[1] = l;
        }
    }
    // last block out merges the head's splits (PD_LAST_BLOCK_FOLD in
    // abi.cuh: release fence, ticket, acquire fence, L1-bypassing reads)
    __threadfence();
    __syncthreads();
    uint32_t* ticket = ctr + (size_t)r * 8u + h;
    if (tid == 0) s_last = atomicAdd(ticket, 1u) == nsplit - 1u;
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    const float* hp = part + ((size_t)r * 8u + h) * nsplit * PS;
    if (tid < nact) s_m[tid] = __ldcg(hp + tid * PS);
    __syncthreads();
    float mm = -INFINITY;
    for (uint32_t s = 0; s < nact; ++s) mm = fmaxf(mm, s_m[s]);
    if (tid < nact) s_f[tid] = expf(s_m[tid] - mm);
    __syncthreads();
    if (tid < PD_PH_HD) {
        float ll = 0.0f, a = 0.0f;
#pragma unroll 4
        for (uint32_t s = 0; s < nact; ++s) {
            const float* ep = hp + s * PS;
            ll = fmaf(__ldcg(ep + 1), s_f[s], ll);
            a = fmaf(__ldcg(ep + 2u + tid), s_f[s], a);
        }
        out[(size_t)r * 512u + h * PD_PH_HD + tid] = a / ll;
    }
    if (tid == 0) *ticket = 0u;
}

PD_EXPORT
int pd_ph_attn(const void* q, const void* kc, const void* vc, void* part, void* out, void* ctr,
               const void* pos_dev, uint32_t pos0, uint32_t rows, uint32_t cap, uint32_t window,
               uint32_t nsplit, void* stream) {
    if (rows == 0) return 0;
    if (cap == 0 || nsplit == 0 || nsplit > 64u || rows > 65535u || !ctr) return cudaErrorInvalidValue;
    pd_pdl_go(pd_ph_attn_kernel, dim3(8, rows, nsplit), dim3(128), 0, (cudaStream_t)stream,
              (const float*)q, (const float*)kc, (const float*)vc, (float*)part, (float*)out,
              (uint32_t*)ctr, (const uint32_t*)pos_dev, pos0, rows, cap, window, nsplit);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 911
// y [rows][d] = LayerNorm(x; w, b) - 908's prologue arithmetic, standalone
// for the wide passes whose product is a tensor-core GEMM.
__global__ void __launch_bounds__(128) pd_ph_layernorm_kernel(const float* x,
                                                              const float* __restrict__ w,
                                                              const float* __restrict__ b, float* y,
                                                              uint32_t d, float eps) {
    __shared__ float red[4];
    PD_PDL_ARM();
    const float* xr = x + (size_t)blockIdx.x * d;
    float mean, rstd;
    pd_ph_row_stats(xr, d, eps, red, mean, rstd);
    for (uint32_t k = threadIdx.x; k < d; k += 128u)
        y[(size_t)blockIdx.x * d + k] = (xr[k] - mean) * rstd * w[k] + b[k];
}

PD_EXPORT
int pd_ph_layernorm(const void* x, const void* w, const void* b, void* y, uint32_t rows,
                    uint32_t d, float eps, void* stream) {
    if (rows == 0) return 0;
    if (d == 0 || !w || !b) return cudaErrorInvalidValue;
    pd_pdl_go(pd_ph_layernorm_kernel, dim3(rows), dim3(128), 0, (cudaStream_t)stream,
              (const float*)x, (const float*)w, (const float*)b, (float*)y, d, eps);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 912
// mode 0: y = a + alpha * b[i % period]      (period 0: y = a)
// mode 1: y = elu(a)
// mode 2: y = silu(a + b[i % period])
// mode 3: y = a * b[i % period] + b[period + i % period]  (b = [std ; mean])
// mode 4: y = [cos(alpha a) ; sin(alpha a)] over a's first `period` values
//         (n = 2 period) - the time embedding's input
// mode 5: y = y + b[i % period] * a          (Mimi's layer-scaled residual)
__global__ void pd_ph_point_kernel(const float* a, const float* b, float* y,
                                   uint64_t n, uint32_t period, uint32_t mode, float alpha) {
    PD_PDL_ARM();
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t j = period ? (uint32_t)(i % period) : 0u;
    const float v = mode == 4u ? 0.0f : a[i];
    float o;
    switch (mode) {
    case 1: o = pd_ph_elu(v); break;
    case 2: o = pd_ph_silu(v + b[j]); break;
    case 3: o = v * b[j] + b[period + j]; break;
    case 4: {
        const float t = alpha * a[j];
        o = i < period ? cosf(t) : sinf(t);
        break;
    }
    case 5: o = y[i] + b[j] * v; break;
    default: o = period ? v + alpha * b[j] : v; break;
    }
    y[i] = o;
}

PD_EXPORT
int pd_ph_point(const void* a, const void* b, void* y, uint64_t n, uint32_t period,
                uint32_t mode, float alpha, void* stream) {
    if (n == 0) return 0;
    if (mode > 5u || ((mode == 2u || mode == 3u || mode == 5u) && (period == 0u || !b)) ||
        (mode == 0u && period && !b) || (mode == 4u && (period == 0u || n != 2ull * period)) ||
        (n + 255u) / 256u > 0x7fffffffull)
        return cudaErrorInvalidValue;
    const uint32_t blocks = (uint32_t)((n + 255u) / 256u);
    pd_pdl_go(pd_ph_point_kernel, dim3(blocks), dim3(256), 0, (cudaStream_t)stream, (const float*)a,
              (const float*)b, (float*)y, n, period, mode, alpha);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 913
// A causal conv's GEMM input: y [rows][k * c], y[t][j * c + i] =
// act(in(t - (k - 1) + j)[i]), in(tau) = tau < 0 ? hist[k - 1 + tau] : x[tau]
// (hist: the k - 1 raw rows before x), act = ELU when `elu`. A transposed
// conv of stride s is the k = 2 case against its phase-packed weights.
__global__ void pd_ph_im2col_kernel(const float* x, const float* hist, float* y, uint32_t rows,
                                    uint32_t c4, uint32_t k,
                                    uint32_t elu) {
    PD_PDL_ARM();
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (uint64_t)rows * k * c4) return;
    const uint32_t i4 = (uint32_t)(idx % c4);
    const uint64_t tj = idx / c4;
    const uint32_t j = (uint32_t)(tj % k), t = (uint32_t)(tj / k);
    const int64_t tau = (int64_t)t - (int64_t)(k - 1u) + j;
    const float4* src = reinterpret_cast<const float4*>(
        tau < 0 ? hist + (size_t)(k - 1u + tau) * c4 * 4u : x + (size_t)tau * c4 * 4u);
    float4 v = src[i4];
    if (elu) {
        v.x = pd_ph_elu(v.x);
        v.y = pd_ph_elu(v.y);
        v.z = pd_ph_elu(v.z);
        v.w = pd_ph_elu(v.w);
    }
    reinterpret_cast<float4*>(y)[idx] = v;
}

PD_EXPORT
int pd_ph_im2col(const void* x, const void* hist, void* y, uint32_t rows, uint32_t c, uint32_t k,
                 uint32_t elu, void* stream) {
    if (rows == 0) return 0;
    if (c == 0 || c % 4u || k == 0 || (k > 1u && !hist) || ((uintptr_t)x & 15u) ||
        ((uintptr_t)y & 15u) || ((uintptr_t)hist & 15u))
        return cudaErrorInvalidValue;
    const uint64_t n = (uint64_t)rows * k * (c / 4u);
    pd_pdl_go(pd_ph_im2col_kernel, dim3((uint32_t)((n + 255u) / 256u)), dim3(256), 0,
              (cudaStream_t)stream, (const float*)x, (const float*)hist, (float*)y, rows, c / 4u, k,
              elu);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 914
// hist [keep][c] <- the last `keep` raw rows of [hist ; x] (x [rows][c]).
// Each thread stages its channel's rows, so the shift is in place.
__global__ void pd_ph_history_kernel(const float* x, float* hist,
                                     uint32_t rows, uint32_t c, uint32_t keep) {
    PD_PDL_ARM();
    const uint32_t ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= c) return;
    float tmp[8];
    for (uint32_t j = 0; j < keep; ++j) {
        // row j of the new history is row (rows + j) of [hist ; x]
        const uint32_t src = rows + j;
        tmp[j] = src < keep ? hist[(size_t)src * c + ch] : x[(size_t)(src - keep) * c + ch];
    }
    for (uint32_t j = 0; j < keep; ++j) hist[(size_t)j * c + ch] = tmp[j];
}

PD_EXPORT
int pd_ph_history(const void* x, void* hist, uint32_t rows, uint32_t c, uint32_t keep,
                  void* stream) {
    if (keep == 0 || c == 0) return 0;
    if (keep > 8u) return cudaErrorInvalidValue;
    pd_pdl_go(pd_ph_history_kernel, dim3((c + 255u) / 256u), dim3(256), 0, (cudaStream_t)stream,
              (const float*)x, (float*)hist, rows, c, keep);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 915
// conv_transpose1d(x, W [512][1][32], stride 16, groups 512)[..., :-16]
// streamed: row 16 f + j, channel c = W[c][j] x[f][c] + W[c][j + 16] x[f-1][c],
// x[-1] = hist.
__global__ void pd_ph_upsample_kernel(const float* x, const float* hist,
                                      const float* __restrict__ w, float* y,
                                      uint32_t frames) {
    PD_PDL_ARM();
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (uint64_t)frames * 16u * 512u) return;
    const uint32_t c = (uint32_t)(idx % 512u), row = (uint32_t)(idx / 512u);
    const uint32_t f = row / 16u, j = row % 16u;
    const float cur = x[(size_t)f * 512u + c];
    const float prev = f ? x[(size_t)(f - 1u) * 512u + c] : hist[c];
    y[idx] = fmaf(w[c * 32u + j], cur, w[c * 32u + j + 16u] * prev);
}

PD_EXPORT
int pd_ph_upsample(const void* x, const void* hist, const void* w, void* y, uint32_t frames,
                   void* stream) {
    if (frames == 0) return 0;
    if (!hist) return cudaErrorInvalidValue;
    const uint64_t n = (uint64_t)frames * 16u * 512u;
    pd_pdl_go(pd_ph_upsample_kernel, dim3((uint32_t)((n + 255u) / 256u)), dim3(256), 0,
              (cudaStream_t)stream, (const float*)x, (const float*)hist, (const float*)w, (float*)y,
              frames);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 916
// A causal conv computed straight from its input rows - 913's im2col and
// the GEMM over it in one pass, without the k-times expanded plane between
// them: y [rows][n] = (resid ? y : 0) + (sum_kk W[o][kk] act(in(t - (k - 1)
// + j)[i]) + b[o]), kk = j cin + i ascending, in(tau) = tau < 0 ? hist[k - 1
// + tau] : x[tau], act = ELU when `elu` - the same weights `[n][k cin]` and
// history as the GEMM path. For the narrow convolutions (few channels,
// many rows: Mimi's last stage), where the expanded plane is most of the
// traffic. A block stages its TO outputs' weights (k-major, ahead of the
// wait - no kernel writes them), then its TR rows plus the k - 1 before them,
// ELU applied, channel-major at an odd pitch (float4 loads, conflict-free
// transposed stores); 256 threads as RG row groups x 256 / RG output groups,
// a thread RT = TR / RG rows by OT outputs in registers. An output's sum
// walks kk in order whatever the tile, so a row's result is the same at any
// row count.
template <uint32_t TR, uint32_t TO, uint32_t RG>
__global__ void __launch_bounds__(256) pd_ph_conv_kernel(
    const float* x, const float* hist, const float* __restrict__ w, const float* __restrict__ bias,
    float* y, uint32_t rows, uint32_t cin, uint32_t n, uint32_t k, uint32_t elu, uint32_t resid) {
    constexpr uint32_t OG = 256u / RG, RT = TR / RG, OT = TO / OG;
    static_assert(RG % 32u == 0 && TR % RG == 0 && TO % OG == 0, "tile");
    extern __shared__ float4 pd_ph_cxs4[];
    float* ws = reinterpret_cast<float*>(pd_ph_cxs4);  // [K][TO]
    const uint32_t K = k * cin, t0 = blockIdx.x * TR, n0 = blockIdx.y * TO;
    float* xs = ws + K * TO;  // [cin][pitch]
    const uint32_t pitch = (TR + k - 1u) | 1u;
    // the block's weights, k-major (immutable: staged before the wait)
    for (uint32_t e = threadIdx.x; e < TO * K; e += 256u) {
        const uint32_t oo = e / K, kk = e - oo * K;
        ws[kk * TO + oo] = n0 + oo < n ? w[(size_t)(n0 + oo) * K + kk] : 0.0f;
    }
    PD_PDL_ARM();
    // rows t0 - (k - 1) .. t0 + TR - 1, four channels a load
    const uint32_t c4 = cin / 4u, span = TR + k - 1u;
#pragma unroll 4
    for (uint32_t e = threadIdx.x; e < span * c4; e += 256u) {
        const uint32_t rr = e / c4, i = (e - rr * c4) * 4u;
        const int64_t g = (int64_t)t0 - (int64_t)(k - 1u) + rr;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (g < 0)
            v = *reinterpret_cast<const float4*>(hist + (size_t)(k - 1u + g) * cin + i);
        else if (g < (int64_t)rows)
            v = *reinterpret_cast<const float4*>(x + (size_t)g * cin + i);
        if (elu) {
            v.x = pd_ph_elu(v.x);
            v.y = pd_ph_elu(v.y);
            v.z = pd_ph_elu(v.z);
            v.w = pd_ph_elu(v.w);
        }
        xs[(i + 0u) * pitch + rr] = v.x;
        xs[(i + 1u) * pitch + rr] = v.y;
        xs[(i + 2u) * pitch + rr] = v.z;
        xs[(i + 3u) * pitch + rr] = v.w;
    }
    __syncthreads();
    const uint32_t rg = threadIdx.x % RG, og = threadIdx.x / RG;
    float acc[RT][OT];
#pragma unroll
    for (uint32_t m = 0; m < RT; ++m)
#pragma unroll
        for (uint32_t q = 0; q < OT; ++q) acc[m][q] = 0.0f;
    for (uint32_t j = 0; j < k; ++j) {
#pragma unroll 4
        for (uint32_t i = 0; i < cin; ++i) {
            const float* xp = xs + i * pitch + j + rg;
            const float* wp = ws + (j * cin + i) * TO + og * OT;
            float xv[RT], wv[OT];
#pragma unroll
            for (uint32_t m = 0; m < RT; ++m) xv[m] = xp[m * RG];
#pragma unroll
            for (uint32_t q = 0; q < OT; ++q) wv[q] = wp[q];
#pragma unroll
            for (uint32_t m = 0; m < RT; ++m)
#pragma unroll
                for (uint32_t q = 0; q < OT; ++q) acc[m][q] = fmaf(wv[q], xv[m], acc[m][q]);
        }
    }
#pragma unroll
    for (uint32_t m = 0; m < RT; ++m) {
        const uint32_t t = t0 + rg + m * RG;
        if (t >= rows) continue;
#pragma unroll
        for (uint32_t q = 0; q < OT; ++q) {
            const uint32_t o = n0 + og * OT + q;
            if (o >= n) continue;
            const size_t at = (size_t)t * n + o;
            const float v = acc[m][q] + bias[o];
            y[at] = resid ? y[at] + v : v;
        }
    }
}

// The tile a shape takes: one output -> 256 rows a block, a row a thread;
// up to 32 outputs -> 128 rows x 32 (4 x 4 a thread); wider -> 128 rows x
// 64 (4 x 8). Shapes whose staged weights and rows pass 96 KB are refused -
// those convolutions belong on the GEMM. Elected by shape only.
PD_EXPORT
int pd_ph_conv(const void* x, const void* hist, const void* w, const void* bias, void* y,
               uint32_t rows, uint32_t cin, uint32_t n, uint32_t k, uint32_t elu, uint32_t resid,
               void* stream) {
    if (rows == 0 || n == 0) return 0;
    if (cin == 0 || cin % 4u || k == 0 || k > 8u || (k > 1u && !hist) || !w || !bias ||
        ((uintptr_t)x & 15u) || ((uintptr_t)hist & 15u))
        return cudaErrorInvalidValue;
    const cudaStream_t s = (cudaStream_t)stream;
    auto go = [&](auto kern, uint32_t tr, uint32_t to) {
        const uint32_t smem = (k * cin * to + ((tr + k - 1u) | 1u) * cin) * 4u;
        if (smem > 96u * 1024u) return (int)cudaErrorInvalidValue;
        static thread_local const void* opted[8] = {};
        bool done = false;
        for (auto& o : opted) done |= o == (const void*)kern;
        if (!done) {
            cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
            for (auto& o : opted)
                if (!o) {
                    o = (const void*)kern;
                    break;
                }
        }
        const dim3 grid((rows + tr - 1u) / tr, (n + to - 1u) / to);
        pd_pdl_go(kern, grid, dim3(256), smem, s, (const float*)x, (const float*)hist,
                  (const float*)w, (const float*)bias, (float*)y, rows, cin, n, k, elu, resid);
        return pd_launch_status();
    };
    if (n == 1u) return go(pd_ph_conv_kernel<256, 1, 256>, 256u, 1u);
    if (n <= 32u) return go(pd_ph_conv_kernel<128, 32, 32>, 128u, 32u);
    return go(pd_ph_conv_kernel<128, 64, 32>, 128u, 64u);
}

// ------------------------------------------------------------------ 917
// A causal conv as the house F32-class GEMM (698, 3xTF32) reading its im2col
// rows in place: x is the conv's input with its k - 1 history rows stacked
// before it ([k - 1 + rows][cin], ELU already applied where the conv takes
// it), so row t's k taps are the k cin floats from x + t cin - the GEMM's A
// at pitch cin instead of K = k cin. y [rows][n] = (resid ? y : 0) + (x' W^T
// + b), W the GEMM rows [n][k cin]; the same sequence as 698 over a
// materialized im2col plane, without writing or reading that plane.
PD_EXPORT
int pd_ph_conv_gemm(const void* x, const void* w, const void* bias, void* y, uint32_t rows,
                    uint32_t cin, uint32_t n, uint32_t k, uint32_t resid, void* stream) {
    if (rows == 0 || n == 0) return 0;
    const uint32_t K = k * cin;
    if (cin == 0 || cin % 4u || k == 0 || K % PD_KUMO_BK || ((uintptr_t)x & 15u) ||
        ((uintptr_t)w & 15u) || (rows + 63u) / 64u > 65535u)
        return (int)cudaErrorInvalidValue;
    PdKumoFuse f{};
    f.lda = cin;
    pd_kumo_gemm_elect<false>((const float*)x, (const float*)w, (const float*)bias, (float*)y,
                              nullptr, K, n, rows, resid ? PD_KUMO_RESID : PD_KUMO_STORE, 1, 0, 0,
                              0, f, (cudaStream_t)stream);
    return pd_launch_status();
}

