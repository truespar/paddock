// moe/inkling2_experts.cuh - Inkling 2 Mini's routed experts at decode
// widths, in the checkpoint's declared classes: NVFP4 W4A4 (layers 0-19) and
// MXFP8 W8A8 (20-23).
//
// Activations: per token, NVFP4 with an outer scale amax / 1536 and, per
// 16-block, whichever e4m3 scale (6.0- or 4.0-based) reconstructs the block
// with the smaller absolute error ("fouroversix_mae"); MXFP8 with a round-up
// power-of-two scale per 32. Weights: the checkpoint's bytes as shipped -
// NVFP4 e2m1 two a byte (even element low), e4m3 per 16, a per-expert factor
// amax / denominator; MXFP8 e4m3 with a ue8m0 exponent per 32. gate and up
// rows interleaved (g0, u0, g1, u1, ...) as the checkpoint stores w13.
//
// The GEMVs decode both operands: e2m1 x e2m1 and e4m3 x e4m3 products are
// exact in f32, so a block's sum is the tensor-core MMA's to the bit and only
// the order of the f32 accumulation across blocks differs. The prefill class
// (tensor cores, expert-grouped) is a separate arm.

// e2m1 of f32 x: round to nearest even, saturating (cvt.rn.satfinite) - the
// thresholds between the grid points 0, .5, 1, 1.5, 2, 3, 4, 6 with ties to
// the even code. Returns the 4-bit code (sign in bit 3).
__device__ __forceinline__ uint32_t pd_ik2_e2m1(float x) {
    const float a = fabsf(x);
    uint32_t c = a <= 0.25f ? 0u
               : a < 0.75f  ? 1u
               : a <= 1.25f ? 2u
               : a < 1.75f  ? 3u
               : a <= 2.5f  ? 4u
               : a < 3.5f   ? 5u
               : a <= 5.0f  ? 6u : 7u;
    return c | ((__float_as_uint(x) >> 31) << 3);
}

// value of an e2m1 code: e = c >> 1 & 3, m = c & 1; e 0 is 0 / 0.5.
__device__ __forceinline__ float pd_ik2_e2m1_val(uint32_t c) {
    const uint32_t e = (c >> 1) & 3u, m = c & 1u, s = (c >> 3) & 1u;
    const uint32_t mag = e ? (((e + 126u) << 23) | (m << 22)) : (m ? 0x3F000000u : 0u);
    return __uint_as_float(mag | (s << 31));
}

__device__ __forceinline__ float pd_ik2_e4m3_val(uint8_t b) {
    __nv_fp8_e4m3 v;
    v.__x = b;
    return float(v);
}

__device__ __forceinline__ float pd_ik2_ue8m0_val(uint8_t e) {
    // 2^(e - 127); e 0 is 2^-127 (a subnormal), 255 NaN
    return e == 0u ? 5.877471754111438e-39f : e == 255u ? __uint_as_float(0x7FC00000u)
                                                       : __uint_as_float((uint32_t)e << 23);
}

// The block sum of 16 e2m1 x e2m1 products, codes packed 8 a uint32 pair
// (element 2k in the low nibble of byte k).
#if PD_BS_OK
// four e2m1x2 bytes of `w` as four f16x2 (the low nibble in the low half)
__device__ __forceinline__ void pd_ik2_e2m1x8_h2(uint32_t w, uint32_t (&h)[4]) {
    asm("{\n\t.reg .b8 b0, b1, b2, b3;\n\t"
        "mov.b32 {b0, b1, b2, b3}, %4;\n\t"
        "cvt.rn.f16x2.e2m1x2 %0, b0;\n\t"
        "cvt.rn.f16x2.e2m1x2 %1, b1;\n\t"
        "cvt.rn.f16x2.e2m1x2 %2, b2;\n\t"
        "cvt.rn.f16x2.e2m1x2 %3, b3;\n\t}"
        : "=r"(h[0]), "=r"(h[1]), "=r"(h[2]), "=r"(h[3])
        : "r"(w));
}
#endif

// The sum of 16 e2m1 x e2m1 products. Every product is a multiple of 1/4
// with |p| <= 36, so every partial sum of up to 16 is exact in f32 and of up
// to 8 exact in f16 (|s| <= 288 on a 1/4 grid) - the sum does not depend on
// its order. Where the die converts e2m1 in hardware the 16 run as two f16
// lanes of eight (HFMA2) and one f32 add: the decode GEMVs were ALU-bound on
// the bit-assembled values, at the same result to the bit.
__device__ __forceinline__ float pd_ik2_dot16(uint2 a, uint2 b) {
#if PD_BS_OK
    uint32_t ha[8], hb[8];
    pd_ik2_e2m1x8_h2(a.x, *(uint32_t(*)[4])&ha[0]);
    pd_ik2_e2m1x8_h2(a.y, *(uint32_t(*)[4])&ha[4]);
    pd_ik2_e2m1x8_h2(b.x, *(uint32_t(*)[4])&hb[0]);
    pd_ik2_e2m1x8_h2(b.y, *(uint32_t(*)[4])&hb[4]);
    __half2 acc = __float2half2_rn(0.0f);
    #pragma unroll
    for (uint32_t j = 0; j < 8; ++j)
        acc = __hfma2(*(const __half2*)&ha[j], *(const __half2*)&hb[j], acc);
    const float2 f = __half22float2(acc);
    return f.x + f.y;
#else
    float s = 0.0f;
    #pragma unroll
    for (uint32_t k = 0; k < 8; ++k) {
        const uint32_t wa = k < 4 ? a.x : a.y, wb = k < 4 ? b.x : b.y;
        const uint32_t sh = (k & 3u) * 8u;
        const uint32_t ba = (wa >> sh) & 0xFFu, bb = (wb >> sh) & 0xFFu;
        s = fmaf(pd_ik2_e2m1_val(ba & 0xFu), pd_ik2_e2m1_val(bb & 0xFu), s);
        s = fmaf(pd_ik2_e2m1_val(ba >> 4), pd_ik2_e2m1_val(bb >> 4), s);
    }
    return s;
#endif
}

// The sum of 32 e4m3 x e4m3 products, 32 bytes each as two uint4.
__device__ __forceinline__ float pd_ik2_dot32_e4m3(const uint4* a, const uint4* b) {
    float s = 0.0f;
    #pragma unroll
    for (uint32_t q = 0; q < 2; ++q) {
        const uint4 va = a[q], vb = b[q];
        const uint32_t wa[4] = {va.x, va.y, va.z, va.w}, wb[4] = {vb.x, vb.y, vb.z, vb.w};
        #pragma unroll
        for (uint32_t k = 0; k < 4; ++k) {
            #pragma unroll
            for (uint32_t j = 0; j < 4; ++j) {
                s = fmaf(pd_ik2_e4m3_val((wa[k] >> (8 * j)) & 0xFFu),
                         pd_ik2_e4m3_val((wb[k] >> (8 * j)) & 0xFFu), s);
            }
        }
    }
    return s;
}

// slot 983: per-row activation quantization. x [rows][K] (row stride xs);
// optionally times gamma[row] (the router weight the down input carries)
// and rounded to bf16 first (the serving class's bf16 activations).
// bf16_in 2 (packs with slot 1006): x holds bf16 rows already (slot 1006's
// bf16(act x gamma)), read as they are.
// NVFP4 (fmt 0): codes [rows][K/2], e4m3 scales [rows][K/16], outer [rows]
// = amax / 1536, four-over-six per block. MXFP8 (fmt 1): e4m3 [rows][K],
// ue8m0 exponents [rows][K/32]; outer unused. One CTA a row, one 16- (32-)
// block a thread. K <= 256 * 16.
__global__ void __launch_bounds__(256) pd_inkling2_quant_kernel(
        const float* x, uint32_t xs, const float* gamma, uint32_t bf16_in, uint32_t fmt,
        uint8_t* __restrict__ q, uint8_t* __restrict__ sc, float* __restrict__ outer, uint32_t K) {
    PD_PDL_ARM();
    __shared__ float red[8];
    const uint32_t r = blockIdx.x, tid = threadIdx.x;
    const uint32_t B = fmt ? 32u : 16u;
    const uint32_t nb = K / B;
    const float* xr = pd_pdl_after(x) + (size_t)r * xs;
    const __nv_bfloat16* xb = (const __nv_bfloat16*)pd_pdl_after(x) + (size_t)r * xs;
    const float g = gamma ? pd_pdl_after(gamma)[r] : 1.0f;
    float v[32];
    float bmax = 0.0f;
    if (tid < nb) {
        #pragma unroll
        for (uint32_t i = 0; i < 32; ++i) {
            if (i >= B) break;
            float a;
            if (bf16_in == 2u) {
                a = __bfloat162float(xb[tid * B + i]);
            } else {
                a = xr[tid * B + i];
                if (gamma) a *= g;
                if (bf16_in) a = __bfloat162float(__float2bfloat16_rn(a));
            }
            v[i] = a;
            bmax = fmaxf(bmax, fabsf(a));
        }
    }
    if (fmt) {
        if (tid >= nb) return;
        // round-up power-of-two scale: 2^ceil(log2(amax / 448)) via the bits
        const float raw = bmax * (1.0f / 448.0f);
        uint32_t ex = ((__float_as_uint(raw) + 0x007FFFFFu) & 0x7F800000u) >> 23;
        ex = raw <= 5.877471754111438e-39f ? 0u : min(ex, 254u);
        const float scale = fmaxf(__uint_as_float(ex << 23), 5.877471754111438e-39f);
        const float inv = ex == 0u ? 1.7014118346046923e38f : __frcp_rn(scale);
        uint8_t* qr = q + (size_t)r * K + tid * 32u;
        #pragma unroll
        for (uint32_t i = 0; i < 32; ++i) {
            const float y = fminf(fmaxf(v[i] * inv, -448.0f), 448.0f);
            qr[i] = __nv_fp8_e4m3(y).__x;
        }
        sc[(size_t)r * (K / 32u) + tid] = (uint8_t)ex;
        return;
    }
    // the row amax
    float m = bmax;
    for (uint32_t sh = 16; sh > 0; sh >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, sh));
    if ((tid & 31u) == 0) red[tid >> 5] = m;
    __syncthreads();
    float amax = 0.0f;
    for (uint32_t w = 0; w < 8u; ++w) amax = fmaxf(amax, red[w]);
    if (tid >= nb) return;
    const float FMAX = 3.4028234663852886e38f;
    const float enc = fminf(__fdiv_rn(1536.0f, amax), FMAX);
    const float dec = __fdiv_rn(1.0f, enc);
    const float out_s = amax * (1.0f / 1536.0f);
    if (tid == 0) outer[r] = out_s;
    float best_s = 0.0f, best_e = 0.0f;
    uint32_t best_c[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    #pragma unroll
    for (uint32_t cand = 0; cand < 2; ++cand) {
        // cand 0: the 6.0-based scale, 1: the 4.0-based one
        float s = fminf(bmax * (enc * (cand == 0 ? (1.0f / 6.0f) : 0.25f)), 448.0f);
        s = float(__nv_fp8_e4m3(s));
        const float inv = fminf(__fdiv_rn(1.0f, s * dec), FMAX);
        uint32_t codes[8];
        float err[16];
        #pragma unroll
        for (uint32_t k = 0; k < 8; ++k) {
            const uint32_t lo = pd_ik2_e2m1(v[2 * k] * inv), hi = pd_ik2_e2m1(v[2 * k + 1] * inv);
            codes[k] = lo | (hi << 4);
            // rounded op by op (no FMA contraction): the reference runs this
            // with fusion off, and the 4-vs-6 choice compares these sums
            err[2 * k] = fabsf(__fsub_rn(__fmul_rn(__fmul_rn(pd_ik2_e2m1_val(lo), s), out_s), v[2 * k]));
            err[2 * k + 1] = fabsf(__fsub_rn(__fmul_rn(__fmul_rn(pd_ik2_e2m1_val(hi), s), out_s), v[2 * k + 1]));
        }
        // the reference's pairwise tree: pairs, quads, octets, the block
        #pragma unroll
        for (uint32_t w = 16; w > 1; w >>= 1)
            #pragma unroll
            for (uint32_t k = 0; k < w / 2; ++k) err[k] = __fadd_rn(err[2 * k], err[2 * k + 1]);
        if (cand == 0 || err[0] < best_e) {
            best_e = err[0];
            best_s = s;
            #pragma unroll
            for (uint32_t k = 0; k < 8; ++k) best_c[k] = codes[k];
        }
    }
    uint8_t* qr = q + (size_t)r * (K / 2u) + tid * 8u;
    #pragma unroll
    for (uint32_t k = 0; k < 8; ++k) qr[k] = (uint8_t)best_c[k];
    sc[(size_t)r * (K / 16u) + tid] = __nv_fp8_e4m3(best_s).__x;
}

PD_EXPORT
int pd_inkling2_quant(const void* x, uint32_t xs, const void* gamma, uint32_t bf16_in, uint32_t fmt,
                      void* q, void* sc, void* outer, uint32_t rows, uint32_t K, void* stream) {
    if (rows == 0) return 0;
    const uint32_t B = fmt ? 32u : 16u;
    if (K % B || K / B > 256u || (fmt == 0 && outer == nullptr) || bf16_in > 2u ||
        (bf16_in == 2u && gamma))
        return cudaErrorInvalidValue;
    pd_pdl_go(pd_inkling2_quant_kernel, rows, 256, 0u, (cudaStream_t)stream, (const float*)x, xs,
              (const float*)gamma, bf16_in, fmt, (uint8_t*)q, (uint8_t*)sc, (float*)outer, K);
    return pd_launch_status();
}

// slot 984: gate|up for every (row, slot) pair at decode widths, with the
// softcapped SwiGLU in the epilogue. Pair p = r * topk + j takes expert
// idx[p] and the row's quantized activation; one warp an output column i
// (weight rows 2i = gate, 2i + 1 = up of that expert); g and u land as the
// bf16 the serving GEMM writes, then act[p][i] = (cap_g(g) * sigmoid(g)) *
// cap_u(u) in f32. fmt 0: NVFP4 (xq/xs/xo from the fmt-0 quantizer, wfac
// the per-expert factor); fmt 1: MXFP8.
__global__ void __launch_bounds__(256) pd_inkling2_moe_gu_kernel(
        const uint8_t* __restrict__ w, const uint8_t* __restrict__ wsc,
        const float* __restrict__ wfac, const uint8_t* xq, const uint8_t* xsc, const float* xo,
        const uint32_t* idx, uint32_t topk, float* __restrict__ act, uint32_t K, uint32_t ff,
        uint32_t fmt, float gcap, float ucap) {
    PD_PDL_ARM();
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t i = blockIdx.x * 8u + (threadIdx.x >> 5);
    const uint32_t p = blockIdx.y, r = p / topk;
    if (i >= ff) return;
    const uint32_t e = pd_pdl_after(idx)[p];
    const uint8_t* xqr = pd_pdl_after(xq);
    const uint8_t* xsr = pd_pdl_after(xsc);
    float g = 0.0f, u = 0.0f;
    if (fmt == 0) {
        const uint32_t nb = K / 16u, rb = K / 2u;
        const uint8_t* wg = w + ((size_t)e * 2u * ff + 2u * i) * rb;
        const uint8_t* sg = wsc + ((size_t)e * 2u * ff + 2u * i) * nb;
        const uint8_t* xr = xqr + (size_t)r * rb;
        const uint8_t* sr = xsr + (size_t)r * nb;
        for (uint32_t b = lane; b < nb; b += 32u) {
            const uint2 xa = *(const uint2*)(xr + b * 8u);
            const float xsb = pd_ik2_e4m3_val(sr[b]);
            g = fmaf(pd_ik2_dot16(xa, *(const uint2*)(wg + b * 8u)), xsb * pd_ik2_e4m3_val(sg[b]), g);
            u = fmaf(pd_ik2_dot16(xa, *(const uint2*)(wg + rb + b * 8u)), xsb * pd_ik2_e4m3_val(sg[nb + b]), u);
        }
    } else {
        const uint32_t nb = K / 32u;
        const uint8_t* wg = w + ((size_t)e * 2u * ff + 2u * i) * K;
        const uint8_t* sg = wsc + ((size_t)e * 2u * ff + 2u * i) * nb;
        const uint8_t* xr = xqr + (size_t)r * K;
        const uint8_t* sr = xsr + (size_t)r * nb;
        for (uint32_t b = lane; b < nb; b += 32u) {
            const uint4* xa = (const uint4*)(xr + b * 32u);
            const float xsb = pd_ik2_ue8m0_val(sr[b]);
            g = fmaf(pd_ik2_dot32_e4m3(xa, (const uint4*)(wg + b * 32u)), xsb * pd_ik2_ue8m0_val(sg[b]), g);
            u = fmaf(pd_ik2_dot32_e4m3(xa, (const uint4*)(wg + K + b * 32u)), xsb * pd_ik2_ue8m0_val(sg[nb + b]), u);
        }
    }
    for (uint32_t sh = 16; sh > 0; sh >>= 1) {
        g += __shfl_xor_sync(0xffffffffu, g, sh);
        u += __shfl_xor_sync(0xffffffffu, u, sh);
    }
    if (lane != 0) return;
    if (fmt == 0) {
        const float f = pd_pdl_after(xo)[r] * wfac[e];
        g *= f;
        u *= f;
    }
    g = __bfloat162float(__float2bfloat16_rn(g));
    u = __bfloat162float(__float2bfloat16_rn(u));
    const float sig = 1.0f / (1.0f + expf(-g));
    act[(size_t)p * ff + i] = ((gcap * tanhf(g / gcap)) * sig) * (ucap * tanhf(u / ucap));
}

PD_EXPORT
int pd_inkling2_moe_gu(const void* w, const void* wsc, const void* wfac, const void* xq,
                       const void* xsc, const void* xo, const void* idx, uint32_t topk, void* act,
                       uint32_t rows, uint32_t K, uint32_t ff, uint32_t fmt, float gcap, float ucap,
                       void* stream) {
    if (rows == 0) return 0;
    if (K % (fmt ? 32u : 16u) || (fmt == 0 && (wfac == nullptr || xo == nullptr)))
        return cudaErrorInvalidValue;
    pd_pdl_go(pd_inkling2_moe_gu_kernel, dim3((ff + 7u) / 8u, rows * topk), 256, 0u,
              (cudaStream_t)stream, (const uint8_t*)w, (const uint8_t*)wsc, (const float*)wfac,
              (const uint8_t*)xq, (const uint8_t*)xsc, (const float*)xo, (const uint32_t*)idx, topk,
              (float*)act, K, ff, fmt, gcap, ucap);
    return pd_launch_status();
}

// slot 985: the down projection for every row, summing its topk pairs in
// slot order (deterministic; the router weight is already inside the
// quantized activation). aq/asc/ao: the pairs' quantized activations
// ([rows * topk][ff] at the fmt's layout); w [E][hidden][ff]. One warp an
// output column, all topk pairs of the row.
__global__ void __launch_bounds__(256) pd_inkling2_moe_dn_kernel(
        const uint8_t* __restrict__ w, const uint8_t* __restrict__ wsc,
        const float* __restrict__ wfac, const uint8_t* aq, const uint8_t* asc, const float* ao,
        const uint32_t* idx, uint32_t topk, float* __restrict__ out, uint32_t ff, uint32_t hidden,
        uint32_t fmt) {
    PD_PDL_ARM();
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t o = blockIdx.x * 8u + (threadIdx.x >> 5);
    const uint32_t r = blockIdx.y;
    if (o >= hidden) return;
    const uint32_t* ix = pd_pdl_after(idx);
    const uint8_t* aqr = pd_pdl_after(aq);
    const uint8_t* asr = pd_pdl_after(asc);
    float sum = 0.0f;
    for (uint32_t j = 0; j < topk; ++j) {
        const uint32_t p = r * topk + j, e = ix[p];
        float a = 0.0f;
        if (fmt == 0) {
            const uint32_t nb = ff / 16u, rb = ff / 2u;
            const uint8_t* wr = w + ((size_t)e * hidden + o) * rb;
            const uint8_t* sr = wsc + ((size_t)e * hidden + o) * nb;
            for (uint32_t b = lane; b < nb; b += 32u)
                a = fmaf(pd_ik2_dot16(*(const uint2*)(aqr + (size_t)p * rb + b * 8u), *(const uint2*)(wr + b * 8u)),
                         pd_ik2_e4m3_val(asr[(size_t)p * nb + b]) * pd_ik2_e4m3_val(sr[b]), a);
        } else {
            const uint32_t nb = ff / 32u;
            const uint8_t* wr = w + ((size_t)e * hidden + o) * ff;
            const uint8_t* sr = wsc + ((size_t)e * hidden + o) * nb;
            for (uint32_t b = lane; b < nb; b += 32u)
                a = fmaf(pd_ik2_dot32_e4m3((const uint4*)(aqr + (size_t)p * ff + b * 32u), (const uint4*)(wr + b * 32u)),
                         pd_ik2_ue8m0_val(asr[(size_t)p * nb + b]) * pd_ik2_ue8m0_val(sr[b]), a);
        }
        for (uint32_t sh = 16; sh > 0; sh >>= 1) a += __shfl_xor_sync(0xffffffffu, a, sh);
        if (fmt == 0) a *= pd_pdl_after(ao)[p] * wfac[e];
        sum += a;
    }
    if (lane == 0) out[(size_t)r * hidden + o] = sum;
}

PD_EXPORT
int pd_inkling2_moe_dn(const void* w, const void* wsc, const void* wfac, const void* aq,
                       const void* asc, const void* ao, const void* idx, uint32_t topk, void* out,
                       uint32_t rows, uint32_t ff, uint32_t hidden, uint32_t fmt, void* stream) {
    if (rows == 0) return 0;
    if (ff % (fmt ? 32u : 16u) || (fmt == 0 && (wfac == nullptr || ao == nullptr)))
        return cudaErrorInvalidValue;
    pd_pdl_go(pd_inkling2_moe_dn_kernel, dim3((hidden + 7u) / 8u, rows), 256, 0u,
              (cudaStream_t)stream, (const uint8_t*)w, (const uint8_t*)wsc, (const float*)wfac,
              (const uint8_t*)aq, (const uint8_t*)asc, (const float*)ao, (const uint32_t*)idx, topk,
              (float*)out, ff, hidden, fmt);
    return pd_launch_status();
}

// ---- prompt widths: expert-grouped over moe_align's 32-row blocks ----------
// The decode GEMVs above re-stream an expert's planes for every pair; at
// prompt widths a block of up to 32 pairs of one expert (moe_align, BM 32)
// shares each weight tile instead. CUDA cores over dequantized tiles: every
// operand is its exact decoded value (code x block scale), the per-row outer
// scale and the per-expert factor applied once at the end, so a dot differs
// from the GEMV's only in its f32 summation order. 64-element K chunks.

// Dequantize 8 consecutive K elements from k of a row at the fmt's layout
// (NVFP4: K/2 code bytes + K/16 e4m3 a row; MXFP8: K bytes + K/32 ue8m0).
// k % 8 == 0, so the 8 never straddle a scale block.
__device__ __forceinline__ void pd_ik2_deq8(const uint8_t* codes, const uint8_t* sc, uint32_t k,
                                            uint32_t fmt, float* dst) {
    if (fmt == 0) {
        const uint32_t w = *(const uint32_t*)(codes + k / 2u);
        const float s = pd_ik2_e4m3_val(sc[k / 16u]);
        #pragma unroll
        for (uint32_t j = 0; j < 4; ++j) {
            const uint32_t byte = (w >> (8u * j)) & 0xFFu;
            dst[2u * j] = pd_ik2_e2m1_val(byte & 0xFu) * s;
            dst[2u * j + 1u] = pd_ik2_e2m1_val(byte >> 4) * s;
        }
    } else {
        const uint2 b = *(const uint2*)(codes + k);
        const float s = pd_ik2_ue8m0_val(sc[k / 32u]);
        #pragma unroll
        for (uint32_t j = 0; j < 8; ++j)
            dst[j] = pd_ik2_e4m3_val(((j < 4u ? b.x : b.y) >> (8u * (j & 3u))) & 0xFFu) * s;
    }
}

__device__ __forceinline__ uint32_t pd_ik2_rowbytes(uint32_t K, uint32_t fmt) {
    return fmt == 0 ? K / 2u : K;
}
__device__ __forceinline__ uint32_t pd_ik2_rowscales(uint32_t K, uint32_t fmt) {
    return fmt == 0 ? K / 16u : K / 32u;
}

// slot 987: gate|up over moe_align's blocks. grid (ff / 32, max_blocks); a
// CTA: the block's (up to) 32 pairs x 32 output columns (64 interleaved
// weight rows); thread (row t / 8, columns 4 (t % 8) ..+4). act[p][i] for
// pair p = token * topk + slot, epilogue as slot 984's.
__global__ void __launch_bounds__(256) pd_inkling2_moe_gu_grp_kernel(
        const uint8_t* __restrict__ w, const uint8_t* __restrict__ wsc,
        const float* __restrict__ wfac, const uint8_t* xq, const uint8_t* xsc, const float* xo,
        const uint32_t* sorted_row, const uint32_t* sorted_slot, const uint32_t* block_expert,
        uint32_t topk, float* __restrict__ act, uint32_t K, uint32_t ff, uint32_t fmt,
        float gcap, float ucap) {
    constexpr uint32_t BM = 32u, KC = 64u;
    PD_PDL_ARM();
    __shared__ float xs[BM][KC + 1];
    __shared__ float ws[2u * BM][KC + 1];
    __shared__ uint32_t tok[BM], slt[BM];
    const uint32_t blk = blockIdx.y, i0 = blockIdx.x * BM, tid = threadIdx.x;
    const uint32_t e = pd_pdl_after(block_expert)[blk];
    if (e == PD_MOE_PAD) return;
    if (tid < BM) {
        tok[tid] = pd_pdl_after(sorted_row)[blk * BM + tid];
        slt[tid] = pd_pdl_after(sorted_slot)[blk * BM + tid];
    }
    __syncthreads();
    const uint32_t rb = pd_ik2_rowbytes(K, fmt), rs = pd_ik2_rowscales(K, fmt);
    const uint8_t* xqr = pd_pdl_after(xq);
    const uint8_t* xsr = pd_pdl_after(xsc);
    const uint32_t tr = tid / 8u, tc = tid % 8u;
    float g[4] = {0, 0, 0, 0}, u[4] = {0, 0, 0, 0};
    for (uint32_t k0 = 0; k0 < K; k0 += KC) {
        {   // x: 32 rows x 64 = 256 x 8
            const uint32_t row = tid / 8u, part = tid % 8u;
            float d[8];
            if (tok[row] != PD_MOE_PAD) {
                pd_ik2_deq8(xqr + (size_t)tok[row] * rb, xsr + (size_t)tok[row] * rs, k0 + part * 8u, fmt, d);
            } else {
                #pragma unroll
                for (uint32_t j = 0; j < 8; ++j) d[j] = 0.0f;
            }
            #pragma unroll
            for (uint32_t j = 0; j < 8; ++j) xs[row][part * 8u + j] = d[j];
        }
        #pragma unroll
        for (uint32_t h = 0; h < 2; ++h) {  // w: 64 rows x 64 = 2 x 256 x 8
            const uint32_t row = h * 32u + tid / 8u, part = tid % 8u;
            const size_t wr = (size_t)e * 2u * ff + 2u * i0 + row;
            float d[8];
            if (2u * i0 + row < 2u * ff) {
                pd_ik2_deq8(w + wr * rb, wsc + wr * rs, k0 + part * 8u, fmt, d);
            } else {
                #pragma unroll
                for (uint32_t j = 0; j < 8; ++j) d[j] = 0.0f;
            }
            #pragma unroll
            for (uint32_t j = 0; j < 8; ++j) ws[row][part * 8u + j] = d[j];
        }
        __syncthreads();
        #pragma unroll 8
        for (uint32_t k = 0; k < KC; ++k) {
            const float xv = xs[tr][k];
            #pragma unroll
            for (uint32_t c = 0; c < 4; ++c) {
                g[c] = fmaf(xv, ws[2u * (tc * 4u + c)][k], g[c]);
                u[c] = fmaf(xv, ws[2u * (tc * 4u + c) + 1u][k], u[c]);
            }
        }
        __syncthreads();
    }
    if (tok[tr] == PD_MOE_PAD) return;
    const uint32_t p = tok[tr] * topk + slt[tr];
    const float f = fmt == 0 ? pd_pdl_after(xo)[tok[tr]] * wfac[e] : 1.0f;
    #pragma unroll
    for (uint32_t c = 0; c < 4; ++c) {
        const uint32_t i = i0 + tc * 4u + c;
        if (i >= ff) continue;
        const float gv = __bfloat162float(__float2bfloat16_rn(g[c] * f));
        const float uv = __bfloat162float(__float2bfloat16_rn(u[c] * f));
        const float sig = 1.0f / (1.0f + expf(-gv));
        act[(size_t)p * ff + i] = ((gcap * tanhf(gv / gcap)) * sig) * (ucap * tanhf(uv / ucap));
    }
}

PD_EXPORT
int pd_inkling2_moe_gu_grp(const void* w, const void* wsc, const void* wfac, const void* xq,
                           const void* xsc, const void* xo, const void* sorted_row,
                           const void* sorted_slot, const void* block_expert, uint32_t max_blocks,
                           uint32_t topk, void* act, uint32_t K, uint32_t ff, uint32_t fmt,
                           float gcap, float ucap, void* stream) {
    if (max_blocks == 0) return 0;
    if (K % 64u || (fmt == 0 && (wfac == nullptr || xo == nullptr))) return cudaErrorInvalidValue;
    pd_pdl_go(pd_inkling2_moe_gu_grp_kernel, dim3((ff + 31u) / 32u, max_blocks), 256, 0u,
              (cudaStream_t)stream, (const uint8_t*)w, (const uint8_t*)wsc, (const float*)wfac,
              (const uint8_t*)xq, (const uint8_t*)xsc, (const float*)xo, (const uint32_t*)sorted_row,
              (const uint32_t*)sorted_slot, (const uint32_t*)block_expert, topk, (float*)act, K, ff,
              fmt, gcap, ucap);
    return pd_launch_status();
}

// slot 988: down over moe_align's blocks into per-pair partials part[p][o]
// (p = token * topk + slot; the fixed-order slot fold, moe_slot_combine,
// sums a token's). grid (hidden / 64, max_blocks); thread (pair t / 8,
// columns 8 (t % 8) ..+8).
__global__ void __launch_bounds__(256) pd_inkling2_moe_dn_grp_kernel(
        const uint8_t* __restrict__ w, const uint8_t* __restrict__ wsc,
        const float* __restrict__ wfac, const uint8_t* aq, const uint8_t* asc, const float* ao,
        const uint32_t* sorted_row, const uint32_t* sorted_slot, const uint32_t* block_expert,
        uint32_t topk, float* __restrict__ part, uint32_t ff, uint32_t hidden, uint32_t fmt) {
    constexpr uint32_t BM = 32u, BN = 64u, KC = 64u;
    PD_PDL_ARM();
    __shared__ float as_[BM][KC + 1];
    __shared__ float ws[BN][KC + 1];
    __shared__ uint32_t prs[BM];
    const uint32_t blk = blockIdx.y, o0 = blockIdx.x * BN, tid = threadIdx.x;
    const uint32_t e = pd_pdl_after(block_expert)[blk];
    if (e == PD_MOE_PAD) return;
    if (tid < BM) {
        const uint32_t t = pd_pdl_after(sorted_row)[blk * BM + tid];
        prs[tid] = t == PD_MOE_PAD ? PD_MOE_PAD : t * topk + pd_pdl_after(sorted_slot)[blk * BM + tid];
    }
    __syncthreads();
    const uint32_t rb = pd_ik2_rowbytes(ff, fmt), rs = pd_ik2_rowscales(ff, fmt);
    const uint8_t* aqr = pd_pdl_after(aq);
    const uint8_t* asr = pd_pdl_after(asc);
    const uint32_t tr = tid / 8u, tc = tid % 8u;
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (uint32_t k0 = 0; k0 < ff; k0 += KC) {
        {
            const uint32_t row = tid / 8u, piece = tid % 8u;
            float d[8];
            if (prs[row] != PD_MOE_PAD) {
                pd_ik2_deq8(aqr + (size_t)prs[row] * rb, asr + (size_t)prs[row] * rs, k0 + piece * 8u, fmt, d);
            } else {
                #pragma unroll
                for (uint32_t j = 0; j < 8; ++j) d[j] = 0.0f;
            }
            #pragma unroll
            for (uint32_t j = 0; j < 8; ++j) as_[row][piece * 8u + j] = d[j];
        }
        #pragma unroll
        for (uint32_t h = 0; h < 2; ++h) {
            const uint32_t row = h * 32u + tid / 8u, piece = tid % 8u;
            const size_t wr = (size_t)e * hidden + o0 + row;
            float d[8];
            if (o0 + row < hidden) {
                pd_ik2_deq8(w + wr * rb, wsc + wr * rs, k0 + piece * 8u, fmt, d);
            } else {
                #pragma unroll
                for (uint32_t j = 0; j < 8; ++j) d[j] = 0.0f;
            }
            #pragma unroll
            for (uint32_t j = 0; j < 8; ++j) ws[row][piece * 8u + j] = d[j];
        }
        __syncthreads();
        #pragma unroll 8
        for (uint32_t k = 0; k < KC; ++k) {
            const float av = as_[tr][k];
            #pragma unroll
            for (uint32_t c = 0; c < 8; ++c) acc[c] = fmaf(av, ws[tc * 8u + c][k], acc[c]);
        }
        __syncthreads();
    }
    if (prs[tr] == PD_MOE_PAD) return;
    const uint32_t p = prs[tr];
    const float f = fmt == 0 ? pd_pdl_after(ao)[p] * wfac[e] : 1.0f;
    #pragma unroll
    for (uint32_t c = 0; c < 8; ++c) {
        const uint32_t o = o0 + tc * 8u + c;
        if (o < hidden) part[(size_t)p * hidden + o] = acc[c] * f;
    }
}

PD_EXPORT
int pd_inkling2_moe_dn_grp(const void* w, const void* wsc, const void* wfac, const void* aq,
                           const void* asc, const void* ao, const void* sorted_row,
                           const void* sorted_slot, const void* block_expert, uint32_t max_blocks,
                           uint32_t topk, void* part, uint32_t ff, uint32_t hidden, uint32_t fmt,
                           void* stream) {
    if (max_blocks == 0) return 0;
    if (ff % 64u || (fmt == 0 && (wfac == nullptr || ao == nullptr))) return cudaErrorInvalidValue;
    pd_pdl_go(pd_inkling2_moe_dn_grp_kernel, dim3((hidden + 63u) / 64u, max_blocks), 256, 0u,
              (cudaStream_t)stream, (const uint8_t*)w, (const uint8_t*)wsc, (const float*)wfac,
              (const uint8_t*)aq, (const uint8_t*)asc, (const float*)ao, (const uint32_t*)sorted_row,
              (const uint32_t*)sorted_slot, (const uint32_t*)block_expert, topk, (float*)part, ff,
              hidden, fmt);
    return pd_launch_status();
}
