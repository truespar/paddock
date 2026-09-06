// quant/nvf4.cuh (formerly 15_nvf4.cuh) - mxf4/nvf4 fp4xfp4 GEMM rungs, fused swiglu->nvf4, SmoothQuant fold, Hadamard rotation
// Textually-included segment of the single pack translation unit.
// Not standalone-compilable: include order is defined by ../pack.cu.
// ---- mxf4 (fp4 x fp4, m16n8k64) dense GEMM - the full Blackwell fp4 rate --
// The mxf8f6f4 kind issues m16n8k32 at the same rate as int8 (~273 TFLOPs
// measured on GB202); kind::mxf4 does m16n8k64 at that rate = 2x FLOP per
// issue (546 measured) AND halves the activation bytes. Cost: activations
// drop to e2m1 too - a lossier class, held to the retrieval-quality gate,
// with the e4m3 route (above) as the quality fallback.
//
// Operand layout (pinned empirically on the live MMA, exact): nibbles pack
// ADJACENT (low nibble of byte j = element 2j, high = 2j+1 - Not the GGUF
// split order), fragments slice the 32-byte k64 row as 16-byte halves at
// byte 4*tq, and scale_vec::2X consumes two ue8m0 bytes per operand (one
// per k32 block, byte 0 = k0..31).

// Q8_0 -> packed-adjacent e2m1 planes (the mxf4 A format). Identical scale
// pick + RN-even encode as pd_q8_0_to_mxfp4; only the byte order differs.
__global__ void pd_q8_0_to_fp4p_kernel(const int8_t* __restrict__ q8,
                                       const __half* __restrict__ s8,
                                       unsigned char* __restrict__ data,
                                       unsigned char* __restrict__ scale,
                                       uint64_t n_blocks) {
    uint64_t blk = blockIdx.x;
    uint32_t d = threadIdx.x;
    if (blk >= n_blocks) return;
    float v = (float)q8[blk * 32u + d] * __half2float(s8[blk]);
    float a = fabsf(v);
    for (uint32_t off = 16; off > 0; off >>= 1)
        a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, off));
    int e = 0;
    if (a > 0.0f) {
        int ex;
        float m = frexpf(a, &ex);
        e = ex - 3 + (m > 0.75f ? 1 : 0);
    }
    unsigned nib = pd_e2m1_rn(v * ldexpf(1.0f, -e));
    unsigned lo = __shfl_sync(0xffffffffu, nib, 2u * d);
    unsigned hi = __shfl_sync(0xffffffffu, nib, 2u * d + 1u);
    if (d < 16u) data[blk * 16u + d] = (unsigned char)(lo | (hi << 4));
    if (d == 0) scale[blk] = (unsigned char)(e + 127);
}

PD_EXPORT
int pd_q8_0_to_fp4p(const void* q8_data, const void* q8_scale, void* mx_data,
                    void* mx_scale, uint64_t n_blocks, void* stream) {
#ifndef PD_BS_HOST
    (void)q8_data; (void)q8_scale; (void)mx_data; (void)mx_scale; (void)n_blocks;
    (void)stream;
    return cudaErrorNotSupported;
#else
    if (n_blocks == 0) return 0;
    pd_q8_0_to_fp4p_kernel<<<(uint32_t)n_blocks, 32, 0, (cudaStream_t)stream>>>(
        (const int8_t*)q8_data, (const __half*)q8_scale, (unsigned char*)mx_data,
        (unsigned char*)mx_scale, n_blocks);
    return pd_launch_status();
#endif
}

// f32 -> packed-adjacent e2m1 + ue8m0 per-32 (the mxf4 B side). Each thread
// owns 8 consecutive elements (two 16B loads, one u32 nibble store); a
// 4-lane group covers a 32-block. Same power-of-2 scale construction as the
// e4m3 quantizer with the e2m1 bound (amax/2^e <= 6).
__device__ __forceinline__ void pd_e2m1_quant8(const float4 v0, const float4 v1,
                                               uint32_t lane4,
                                               unsigned char* __restrict__ q,
                                               unsigned char* __restrict__ scale,
                                               uint32_t i) {
    float a = fmaxf(fmaxf(fabsf(v0.x), fabsf(v0.y)), fmaxf(fabsf(v0.z), fabsf(v0.w)));
    a = fmaxf(a, fmaxf(fmaxf(fabsf(v1.x), fabsf(v1.y)), fmaxf(fabsf(v1.z), fabsf(v1.w))));
    const uint32_t gm = 0xfu << ((threadIdx.x & 31u) & ~3u);  // 4-lane group
    a = fmaxf(a, __shfl_xor_sync(gm, a, 2));
    a = fmaxf(a, __shfl_xor_sync(gm, a, 1));
    int e = 0;
    if (a > 0.0f) {
        int ex;
        float m = frexpf(a, &ex);
        e = ex - 3 + (m > 0.75f ? 1 : 0);
    }
    const float inv = ldexpf(1.0f, -e);
    const uint32_t p = pd_e2m1_rn(v0.x * inv) | (pd_e2m1_rn(v0.y * inv) << 4)
                     | (pd_e2m1_rn(v0.z * inv) << 8) | (pd_e2m1_rn(v0.w * inv) << 12)
                     | (pd_e2m1_rn(v1.x * inv) << 16) | (pd_e2m1_rn(v1.y * inv) << 20)
                     | (pd_e2m1_rn(v1.z * inv) << 24) | (pd_e2m1_rn(v1.w * inv) << 28);
    *(uint32_t*)(q + (i >> 1)) = p;
    if (lane4 == 0) scale[i >> 5] = (unsigned char)(e + 127);
}

__global__ void pd_quantize_e2m1_kernel(const float* __restrict__ x,
                                        unsigned char* __restrict__ q,
                                        unsigned char* __restrict__ scale, uint32_t n) {
    const uint32_t i = (blockIdx.x * 256u + threadIdx.x) * 8u;
    if (i >= n) return;  // n % 32 == 0: 4-lane groups exit whole
    pd_e2m1_quant8(*(const float4*)(x + i), *(const float4*)(x + i + 4u),
                   threadIdx.x & 3u, q, scale, i);
}

PD_EXPORT
int pd_quantize_e2m1(const void* x, void* q, void* scale, uint32_t n, void* stream) {
    if (n == 0) return 0;
    pd_quantize_e2m1_kernel<<<(n / 8u + 255u) / 256u, 256u, 0, (cudaStream_t)stream>>>(
        (const float*)x, (unsigned char*)q, (unsigned char*)scale, n);
    return pd_launch_status();
}

// SwiGLU fused into the e2m1 quantize (silu math identical to pd_swiglu).
__global__ void pd_quantize_e2m1_swiglu_kernel(const float* __restrict__ gate,
                                               const float* __restrict__ up,
                                               unsigned char* __restrict__ q,
                                               unsigned char* __restrict__ scale,
                                               uint32_t n) {
    const uint32_t i = (blockIdx.x * 256u + threadIdx.x) * 8u;
    if (i >= n) return;
    float4 v[2];
    #pragma unroll
    for (uint32_t h = 0; h < 2u; ++h) {
        const float4 g = *(const float4*)(gate + i + h * 4u);
        const float4 u = *(const float4*)(up + i + h * 4u);
        v[h].x = (g.x / (1.0f + expf(-g.x))) * u.x;
        v[h].y = (g.y / (1.0f + expf(-g.y))) * u.y;
        v[h].z = (g.z / (1.0f + expf(-g.z))) * u.z;
        v[h].w = (g.w / (1.0f + expf(-g.w))) * u.w;
    }
    pd_e2m1_quant8(v[0], v[1], threadIdx.x & 3u, q, scale, i);
}

PD_EXPORT
int pd_quantize_e2m1_swiglu(const void* gate, const void* up, void* q, void* scale,
                            uint32_t n, void* stream) {
    if (n == 0) return 0;
    pd_quantize_e2m1_swiglu_kernel<<<(n / 8u + 255u) / 256u, 256u, 0,
                                     (cudaStream_t)stream>>>(
        (const float*)gate, (const float*)up, (unsigned char*)q,
        (unsigned char*)scale, n);
    return pd_launch_status();
}

#if PD_BS_OK
// The k64 mxf4 MMA. scale_vec::2X: sfa/sfb carry two ue8m0 bytes (k32
// halves), byte-id 0.
__device__ __forceinline__ void pd_f4_mma(float d[4], uint32_t a0, uint32_t a1,
                                          uint32_t a2, uint32_t a3, uint32_t b0,
                                          uint32_t b1, uint32_t sfa, uint32_t sfb) {
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X"
        ".f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
        "{%10}, {0, 0}, {%11}, {0, 0};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "r"(sfb));
}
#endif

// mxf4 dense GEMM: same mmq_pipe-shaped skeleton as pd_mxfp4_gemm_bs but
// both operands packed e2m1 (adjacent order) and K staged 128-deep - a k64
// row is 32 B for both sides, so even the double-deep chunk double-buffers
// in 40 KB and keeps 2 blocks/SM. Fragments via ldmatrix.x4: one op is a
// complete k64 A-fragment set for a 16-row group (parts = row-half x
// k32-halves), one op is b0/b1 of both k64 blocks for a col octet.
#define PD_F4_WROW 80u   // 64B packed fp4 (KC=128) + 4 ue8m0 + pad
#define PD_F4_YROW 80u   // 4 ue8m0 + 64B packed fp4 at +16
#define PD_F4_SMEM (2u * 128u * (PD_F4_WROW + PD_F4_YROW))

__global__ void __launch_bounds__(256, 2) pd_mxfp4_gemm_f4_kernel(
    const unsigned char* __restrict__ data, const unsigned char* __restrict__ scale,
    const unsigned char* __restrict__ xq, const unsigned char* __restrict__ xs,
    float* __restrict__ y, uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
#if PD_BS_OK
    extern __shared__ unsigned char pd_bs_sh[];
    unsigned char* wb0 = pd_bs_sh;
    unsigned char* wb1 = wb0 + 128u * PD_F4_WROW;
    unsigned char* yb0 = wb1 + 128u * PD_F4_WROW;
    unsigned char* yb1 = yb0 + 128u * PD_F4_YROW;

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, tq = lane & 3u;
    const uint32_t i0 = (warp >> 1) * 32u;
    const uint32_t joff = (warp & 1u) * 8u;
    const uint32_t n_kb = in_dim >> 5;
    const uint32_t nk = (in_dim + 127u) / 128u;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t nct = batch_pad >> 7;
    const uint32_t tile = blockIdx.x;
    const uint32_t row_base = (tile / nct) * 128u;
    const uint32_t col_base = (tile % nct) * 128u;

    float acc[16][4] = {};

    // W and Y: 128 rows x 64B packed fp4 (4 16B segs, each one k32 block)
    #define PD_F4_ISSUE_W(dst, kt)                                                    \
        for (uint32_t u = tid; u < 512u; u += 256u) {                                 \
            const uint32_t row = u >> 2, seg = u & 3u;                                \
            const bool ok = (row_base + row) < out_dim && (kt) * 4u + seg < n_kb;     \
            pd_cp_async16((int*)((dst) + row * PD_F4_WROW + seg * 16u),               \
                          data + (size_t)(row_base + row) * (in_dim >> 1) +           \
                              (kt) * 64u + seg * 16u,                                 \
                          ok);                                                        \
        }
    #define PD_F4_ISSUE_Y(dst, kt)                                                    \
        for (uint32_t u = tid; u < 512u; u += 256u) {                                 \
            const uint32_t col = u >> 2, seg = u & 3u;                                \
            const bool ok =                                                           \
                (col_base + col) < batch && (kt) * 4u + seg < n_kb;                   \
            pd_cp_async16((int*)((dst) + col * PD_F4_YROW + 16u + seg * 16u),         \
                          xq + ((size_t)(ok ? col_base + col : 0u) * in_dim >> 1) +   \
                              (kt) * 64u + seg * 16u,                                 \
                          ok);                                                        \
        }

    PD_F4_ISSUE_W(wb0, 0u)
    PD_F4_ISSUE_Y(yb0, 0u)
    asm volatile("cp.async.commit_group;");
    for (uint32_t kt = 0; kt < nk; ++kt) {
        unsigned char* tw = (kt & 1u) ? wb1 : wb0;
        unsigned char* ty = (kt & 1u) ? yb1 : yb0;
        if (kt + 1u < nk) {
            PD_F4_ISSUE_W((kt & 1u) ? wb0 : wb1, kt + 1u)
            PD_F4_ISSUE_Y((kt & 1u) ? yb0 : yb1, kt + 1u)
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        {   // ue8m0 planes: 4 bytes per row per chunk, 2 bytes per thread
            const uint32_t row = tid >> 1, k2 = (tid & 1u) * 2u;
            #pragma unroll
            for (uint32_t kk = 0; kk < 2u; ++kk) {
                const uint32_t kb = k2 + kk;
                const bool wok = (row_base + row) < out_dim && kt * 4u + kb < n_kb;
                tw[row * PD_F4_WROW + 64u + kb] =
                    wok ? scale[(size_t)(row_base + row) * n_kb + kt * 4u + kb] : 0u;
                const bool yok = (col_base + row) < batch && kt * 4u + kb < n_kb;
                ty[row * PD_F4_YROW + kb] =
                    yok ? xs[(size_t)(col_base + row) * n_kb + kt * 4u + kb] : 0u;
            }
        }
        __syncthreads();

        // A: one ldmatrix.x4 per (16-row group, k64 block) - parts are
        // {rows+0 k-lo16B, rows+8 k-lo16B, rows+0 k-hi16B, rows+8 k-hi16B},
        // which is exactly {a0, a1, a2, a3}. Scale u16 = the block's 2 bytes.
        uint32_t am[2][2][4], sa[2][2];
        #pragma unroll
        for (uint32_t n = 0; n < 2u; ++n) {
            const uint32_t r0 = i0 + n * 16u + g;
            const uint32_t rs = (tq & 1u) ? r0 + 8u : r0;
            #pragma unroll
            for (uint32_t k64 = 0; k64 < 2u; ++k64) {
                pd_ldm_x4(am[n][k64],
                          tw + (i0 + n * 16u + ((lane >> 3) & 1u) * 8u + (lane & 7u)) *
                                  PD_F4_WROW +
                              k64 * 32u + (lane >> 4) * 16u);
                sa[n][k64] =
                    *(const unsigned short*)(tw + rs * PD_F4_WROW + 64u + k64 * 2u);
            }
        }
        #pragma unroll
        for (uint32_t j0 = 0; j0 < 128u; j0 += 16u) {
            // B: one ldmatrix.x4 = b0/b1 of both k64 blocks for 8 cols
            uint32_t bm[4];
            pd_ldm_x4(bm, ty + (j0 + joff + (lane & 7u)) * PD_F4_YROW + 16u +
                              (lane >> 3) * 16u);
            const unsigned char* ysr = ty + (j0 + joff + g) * PD_F4_YROW;
            #pragma unroll
            for (uint32_t k64 = 0; k64 < 2u; ++k64) {
                const uint32_t sb = *(const unsigned short*)(ysr + k64 * 2u);
                #pragma unroll
                for (uint32_t n = 0; n < 2u; ++n)
                    pd_f4_mma(acc[(j0 >> 3) + n], am[n][k64][0], am[n][k64][1],
                              am[n][k64][2], am[n][k64][3], bm[k64 * 2u],
                              bm[k64 * 2u + 1u], sa[n][k64], sb);
            }
        }
        __syncthreads();
    }
    #undef PD_F4_ISSUE_W
    #undef PD_F4_ISSUE_Y

    #pragma unroll
    for (uint32_t j0 = 0; j0 < 128u; j0 += 16u) {
        const uint32_t c0 = col_base + j0 + joff + 2u * tq;
        #pragma unroll
        for (uint32_t n = 0; n < 2u; ++n) {
            const uint32_t r0 = row_base + i0 + n * 16u + g;
            const uint32_t r8 = r0 + 8u;
            if (r0 < out_dim) {
                if (c0 < batch) y[(size_t)c0 * out_dim + r0] = acc[(j0 >> 3) + n][0];
                if (c0 + 1u < batch) y[(size_t)(c0 + 1u) * out_dim + r0] = acc[(j0 >> 3) + n][1];
            }
            if (r8 < out_dim) {
                if (c0 < batch) y[(size_t)c0 * out_dim + r8] = acc[(j0 >> 3) + n][2];
                if (c0 + 1u < batch) y[(size_t)(c0 + 1u) * out_dim + r8] = acc[(j0 >> 3) + n][3];
            }
        }
    }
#else
    (void)data; (void)scale; (void)xq; (void)xs; (void)y;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}

PD_EXPORT
int pd_mxfp4_gemm_f4(const void* data, const void* scale, const void* xq,
                     const void* xs, void* y, uint32_t in_dim, uint32_t out_dim,
                     uint32_t batch, void* stream) {
#ifndef PD_BS_HOST
    (void)data; (void)scale; (void)xq; (void)xs; (void)y; (void)in_dim;
    (void)out_dim; (void)batch; (void)stream;
    return cudaErrorNotSupported;
#else
    if (out_dim == 0 || batch == 0) return 0;
    if ((in_dim & 31u) != 0) return cudaErrorInvalidValue;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t ntiles = ((out_dim + 127u) / 128u) * (batch_pad >> 7);
    pd_mxfp4_gemm_f4_kernel<<<ntiles, 256, PD_F4_SMEM, (cudaStream_t)stream>>>(
        (const unsigned char*)data, (const unsigned char*)scale,
        (const unsigned char*)xq, (const unsigned char*)xs, (float*)y, in_dim,
        out_dim, batch);
    return pd_launch_status();
#endif
}

// ---- nvf4 (fp4 x fp4, E4M3 scales per 16) - the outlier-tolerant fp4 rung -
// Same m16n8k64 issue rate as mxf4 (kind::mxf4nvf4, scale_vec::4X) but the
// scales are E4M3 numbers per SIXTEEN elements instead of powers of two per
// 32 - the NVFP4 recipe, built for exactly the activation-outlier problem
// that keeps the mxf4 rung off the post-norm hidden (recall@1 gate). Finer,
// non-power-of-2 scales also make the WEIGHT planes more precise than the
// GGUF mxfp4 class. Layout pinned empirically: fragments identical to mxf4
// (adjacent nibbles, 16 B k-half slices), 4 scale bytes per k64 operand,
// ue4m3 = the positive-e4m3 byte encoding.

// amax -> (e4m3 scale byte, 1/scale). RN to e4m3 of amax/6 (values then
// RN-clamp onto the e2m1 grid); zero blocks get scale 0 with inv 0 so the
// nibbles stay 0 instead of NaN.
__device__ __forceinline__ unsigned pd_nvf4_scale(float amax, float* inv) {
    if (amax <= 0.0f) {
        *inv = 0.0f;
        return 0u;
    }
    __nv_fp8_e4m3 s;
    s.__x = __nv_fp8_e4m3(amax * (1.0f / 6.0f)).__x;
    if (s.__x == 0) s.__x = 1;  // smallest subnormal: amax > 0 needs s > 0
    *inv = 1.0f / (float)s;
    return s.__x;
}

// Q8_0 -> packed-adjacent e2m1 + e4m3 per-16 planes (the nvf4 format; the
// scale plane is numel/16 bytes, twice the mxfp4 plane). One warp per Q8_0
// block = two nvf4 blocks.
__global__ void pd_q8_0_to_nvf4_kernel(const int8_t* __restrict__ q8,
                                       const __half* __restrict__ s8,
                                       unsigned char* __restrict__ data,
                                       unsigned char* __restrict__ scale,
                                       uint64_t n_blocks) {
    uint64_t blk = blockIdx.x;
    uint32_t d = threadIdx.x;
    if (blk >= n_blocks) return;
    float v = (float)q8[blk * 32u + d] * __half2float(s8[blk]);
    float a = fabsf(v);
    const uint32_t gm = 0xFFFFu << (d & 16u);  // 16-lane half-warp group
    for (uint32_t off = 8; off > 0; off >>= 1)
        a = fmaxf(a, __shfl_xor_sync(gm, a, off));
    float inv;
    unsigned sb = pd_nvf4_scale(a, &inv);
    unsigned nib = pd_e2m1_rn(v * inv);
    unsigned lo = __shfl_sync(0xffffffffu, nib, 2u * d);
    unsigned hi = __shfl_sync(0xffffffffu, nib, 2u * d + 1u);
    if (d < 16u) data[blk * 16u + d] = (unsigned char)(lo | (hi << 4));
    if ((d & 15u) == 0) scale[blk * 2u + (d >> 4)] = (unsigned char)sb;
}

PD_EXPORT
int pd_q8_0_to_nvf4(const void* q8_data, const void* q8_scale, void* mx_data,
                    void* mx_scale, uint64_t n_blocks, void* stream) {
#ifndef PD_BS_HOST
    (void)q8_data; (void)q8_scale; (void)mx_data; (void)mx_scale; (void)n_blocks;
    (void)stream;
    return cudaErrorNotSupported;
#else
    if (n_blocks == 0) return 0;
    pd_q8_0_to_nvf4_kernel<<<(uint32_t)n_blocks, 32, 0, (cudaStream_t)stream>>>(
        (const int8_t*)q8_data, (const __half*)q8_scale, (unsigned char*)mx_data,
        (unsigned char*)mx_scale, n_blocks);
    return pd_launch_status();
#endif
}

// f32 -> packed-adjacent e2m1 + e4m3 per-16 (the nvf4 B side). Each thread
// owns 8 consecutive elements = half an nvf4 block; the pair-lane shfl
// closes the 16-element amax.
__device__ __forceinline__ void pd_nvf4_quant8(const float4 v0, const float4 v1,
                                               uint32_t lane, unsigned char* __restrict__ q,
                                               unsigned char* __restrict__ scale,
                                               uint32_t i) {
    float a = fmaxf(fmaxf(fabsf(v0.x), fabsf(v0.y)), fmaxf(fabsf(v0.z), fabsf(v0.w)));
    a = fmaxf(a, fmaxf(fmaxf(fabsf(v1.x), fabsf(v1.y)), fmaxf(fabsf(v1.z), fabsf(v1.w))));
    const uint32_t gm = 0x3u << (lane & 30u);  // 2-lane group = one 16-block
    a = fmaxf(a, __shfl_xor_sync(gm, a, 1));
    float inv;
    unsigned sb = pd_nvf4_scale(a, &inv);
    const uint32_t p = pd_e2m1_rn(v0.x * inv) | (pd_e2m1_rn(v0.y * inv) << 4)
                     | (pd_e2m1_rn(v0.z * inv) << 8) | (pd_e2m1_rn(v0.w * inv) << 12)
                     | (pd_e2m1_rn(v1.x * inv) << 16) | (pd_e2m1_rn(v1.y * inv) << 20)
                     | (pd_e2m1_rn(v1.z * inv) << 24) | (pd_e2m1_rn(v1.w * inv) << 28);
    *(uint32_t*)(q + (i >> 1)) = p;
    if ((lane & 1u) == 0) scale[i >> 4] = (unsigned char)sb;
}

__global__ void pd_quantize_nvf4_kernel(const float* __restrict__ x,
                                        unsigned char* __restrict__ q,
                                        unsigned char* __restrict__ scale, uint32_t n) {
    const uint32_t i = (blockIdx.x * 256u + threadIdx.x) * 8u;
    if (i >= n) return;  // n % 32 == 0: pair groups exit whole
    pd_nvf4_quant8(*(const float4*)(x + i), *(const float4*)(x + i + 4u),
                   threadIdx.x & 31u, q, scale, i);
}

PD_EXPORT
int pd_quantize_nvf4(const void* x, void* q, void* scale, uint32_t n, void* stream) {
    if (n == 0) return 0;
    pd_quantize_nvf4_kernel<<<(n / 8u + 255u) / 256u, 256u, 0, (cudaStream_t)stream>>>(
        (const float*)x, (unsigned char*)q, (unsigned char*)scale, n);
    return pd_launch_status();
}

// Residual-add + rmsnorm + nvf4 quantize, one CTA per row (glue rung).
// nemotron's MoE layers run this as three launches - pd_add_inplace,
// pd_rmsnorm_batch, pd_quantize_nvf4 - 23 times per decode tick, and at c32
// each one is a ~10 us latency-bound trickle over ~344 KB (34 GB/s) whose
// nominal sum is the wall. Every layer of the checkpoint's pattern is
// (mamba|attention) -> moe, so all 23 MoE prologues carry the add.
//
// BIT-EXACT to that chain by construction, and the construction is the whole
// point:
//   - section 1 is pd_rmsnorm_batch_kernel's vectorized branch verbatim over
//     the SUMMED values (f64 acc, float4 index stride nth, same shfl tree,
//     same `1.0f / sqrtf((float)(sum / (double)n) + eps)` -- not rsqrtf), so
//     the launcher must pass the same nth pd_rmsnorm_batch would have picked;
//     regrouping the reduction is the sanctioned near-tie class, not a free
//     choice.
//   - `out` keeps the exact expression `v * inv * w`, so the f32 normed row
//     the router still reads is unchanged element for element. Only which
//     THREAD computes a given element moves, which cannot move a value.
//   - the quantize is pd_nvf4_quant8 unmodified, so the 2-lane 16-block amax
//     group is intact: n % 32 == 0 makes n8 = n/8 even, threads 0..n8-1 are
//     contiguous, and nth is a multiple of 32, so a pair is never split
//     across the loop bound (a split pair would deadlock the shfl, not just
//     round differently).
// Sections 2+3 share one 8-element walk: fusing them drops a second read of
// the row and lets the quantize consume `r` straight out of registers rather
// than re-reading what section 2 just stored.
__global__ void pd_add_rmsnorm_quant_nvf4_batch_kernel(
    float* __restrict__ x, const float* __restrict__ proj,
    const float* __restrict__ w, float* __restrict__ out,
    unsigned char* __restrict__ q, unsigned char* __restrict__ scale,
    uint32_t n, float eps) {
    PD_PDL_ARM();
    const uint32_t b = blockIdx.x;
    float* xb = x + (size_t)b * n;
    const float* pb = proj ? proj + (size_t)b * n : nullptr;
    float* ob = out + (size_t)b * n;
    unsigned char* qb = q + (size_t)b * (n >> 1);
    unsigned char* sb = scale + (size_t)b * (n >> 4);
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    const uint32_t warp = tid >> 5, lane = tid & 31u;
    __shared__ double wsum[32];
    __shared__ float s_inv;

    // section 1: x += proj (skipped when proj==null: the attn-input norm has
    // no residual), write back, sum squares of the SUMMED values
    const uint32_t n4 = n >> 2;
    double acc = 0.0;
    {
        float4* x4 = reinterpret_cast<float4*>(xb);
        if (proj) {
            const float4* p4 = reinterpret_cast<const float4*>(pb);
            for (uint32_t i = tid; i < n4; i += nth) {
                float4 v = x4[i];
                const float4 pv = p4[i];
                v.x += pv.x; v.y += pv.y; v.z += pv.z; v.w += pv.w;
                x4[i] = v;
                acc += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
            }
        } else {
            for (uint32_t i = tid; i < n4; i += nth) {
                const float4 v = x4[i];
                acc += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
            }
        }
    }
    for (uint32_t s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, s);
    if (lane == 0) wsum[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        double sum = 0.0;
        const uint32_t nwarps = (nth + 31u) >> 5;
        for (uint32_t wi = 0; wi < nwarps; ++wi) sum += wsum[wi];
        s_inv = 1.0f / sqrtf((float)(sum / (double)n) + eps);
    }
    __syncthreads();   // also publishes section 1's x writes to the whole CTA
    const float inv = s_inv;

    // sections 2+3: normed row out, and its nvf4 planes, in one 8-element walk
    const uint32_t n8 = n >> 3;
    for (uint32_t j = tid; j < n8; j += nth) {
        const uint32_t i = j << 3;
        const float4 v0 = *(const float4*)(xb + i), v1 = *(const float4*)(xb + i + 4u);
        const float4 w0 = *(const float4*)(w + i), w1 = *(const float4*)(w + i + 4u);
        float4 r0, r1;
        r0.x = v0.x * inv * w0.x; r0.y = v0.y * inv * w0.y;
        r0.z = v0.z * inv * w0.z; r0.w = v0.w * inv * w0.w;
        r1.x = v1.x * inv * w1.x; r1.y = v1.y * inv * w1.y;
        r1.z = v1.z * inv * w1.z; r1.w = v1.w * inv * w1.w;
        *(float4*)(ob + i) = r0;
        *(float4*)(ob + i + 4u) = r1;
        pd_nvf4_quant8(r0, r1, lane, qb, sb, i);
    }
}

PD_EXPORT
int pd_add_rmsnorm_quant_nvf4_batch(void* x, const void* proj, const void* w,
                                    void* out, void* q, void* scale, uint32_t n,
                                    float eps, uint32_t batch, void* stream) {
    if (n == 0 || batch == 0) return 0;
    // n % 32 keeps the 16-block scale groups whole AND keeps n4/n8 exact; the
    // nth pick must match pd_rmsnorm_batch or the reduction regroups.
    if ((n & 31u) != 0) return cudaErrorInvalidValue;
    const uint32_t nth = batch >= 64u ? pd_norm_wide_nth_ws(batch) : pd_norm_decode_nth();
    pd_pdl_go(pd_add_rmsnorm_quant_nvf4_batch_kernel, batch, nth, 0u,
              (cudaStream_t)stream, (float*)x, (const float*)proj,
              (const float*)w, (float*)out, (unsigned char*)q,
              (unsigned char*)scale, n, eps);
    return pd_launch_status();
}

__global__ void pd_quantize_nvf4_swiglu_kernel(const float* __restrict__ gate,
                                               const float* __restrict__ up,
                                               unsigned char* __restrict__ q,
                                               unsigned char* __restrict__ scale,
                                               uint32_t n) {
    const uint32_t i = (blockIdx.x * 256u + threadIdx.x) * 8u;
    if (i >= n) return;
    float4 v[2];
    #pragma unroll
    for (uint32_t h = 0; h < 2u; ++h) {
        const float4 g = *(const float4*)(gate + i + h * 4u);
        const float4 u = *(const float4*)(up + i + h * 4u);
        v[h].x = (g.x / (1.0f + expf(-g.x))) * u.x;
        v[h].y = (g.y / (1.0f + expf(-g.y))) * u.y;
        v[h].z = (g.z / (1.0f + expf(-g.z))) * u.z;
        v[h].w = (g.w / (1.0f + expf(-g.w))) * u.w;
    }
    pd_nvf4_quant8(v[0], v[1], threadIdx.x & 31u, q, scale, i);
}

// Merged twin of pd_quantize_nvf4_swiglu: swiglu over a fused
// [rows, 2*ff] gate|up landing straight into the nvf4 down-input staging, one
// launch. The granite NVFP4 FFN keeps gate|up as one plane, so its down input
// was pd_swiglu_fused (f32 ffn_gate) + pd_quantize_nvf4 -- this removes the f32
// round trip of the widest activation in the mixed tick (rows x ff, ff up to
// 32768 on the 30B). Values are pd_swiglu_fused_kernel's expression verbatim
// and the quant is pd_nvf4_quant8 unmodified, so the down GEMM sees the same
// bytes the pair produced -- bit-identical. ff % 32 == 0 keeps each thread's
// 8-element span inside one row and the 16-block amax groups whole.
__global__ void pd_swiglu_fused_nvf4_kernel(const float* __restrict__ fused,
                                            unsigned char* __restrict__ q,
                                            unsigned char* __restrict__ scale,
                                            uint32_t ff, uint32_t n) {
    const uint32_t i = (blockIdx.x * 256u + threadIdx.x) * 8u;
    if (i >= n) return;
    const uint32_t tok = i / ff, j = i % ff;
    const float* row = fused + (size_t)tok * 2u * ff;
    float4 v[2];
    #pragma unroll
    for (uint32_t h = 0; h < 2u; ++h) {
        const float4 g = *(const float4*)(row + j + h * 4u);
        const float4 u = *(const float4*)(row + ff + j + h * 4u);
        v[h].x = (g.x / (1.0f + expf(-g.x))) * u.x;
        v[h].y = (g.y / (1.0f + expf(-g.y))) * u.y;
        v[h].z = (g.z / (1.0f + expf(-g.z))) * u.z;
        v[h].w = (g.w / (1.0f + expf(-g.w))) * u.w;
    }
    pd_nvf4_quant8(v[0], v[1], threadIdx.x & 31u, q, scale, i);
}
PD_EXPORT
int pd_swiglu_fused_nvf4(const void* fused, void* q, void* scale, uint32_t ff,
                         uint32_t n_rows, void* stream) {
    if (n_rows == 0 || ff == 0) return 0;
    if (ff & 31u) return cudaErrorInvalidValue;
    const uint32_t n = n_rows * ff;
    pd_swiglu_fused_nvf4_kernel<<<(n / 8u + 255u) / 256u, 256u, 0,
                                  (cudaStream_t)stream>>>(
        (const float*)fused, (unsigned char*)q, (unsigned char*)scale, ff, n);
    return pd_launch_status();
}

// ---------------------------------------------------------------------------
// F1/F2
template <int ACC, bool F32DIV>
__global__ void pd_add_rmsnorm_quant_nvf4_from_parts_kernel(
    float* __restrict__ x, const float* __restrict__ part,
    const float* __restrict__ w, float* __restrict__ out,
    const float* __restrict__ bias, unsigned char* __restrict__ q,
    unsigned char* __restrict__ scale, uint32_t n, float eps, float pscale,
    float scale2, uint32_t batch, uint32_t nz) {
    using A = typename pd_acc_of<ACC>::type;
    PD_PDL_ARM();
    const uint32_t b = blockIdx.x;
    float* xb = x + (size_t)b * n;
    const float* pb0 = part + (size_t)b * n;
    float* ob = out ? out + (size_t)b * n : nullptr;
    unsigned char* qb = q + (size_t)b * (n >> 1);
    unsigned char* sb = scale + (size_t)b * (n >> 4);
    const size_t partN4 = ((size_t)batch * n) >> 2;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    const uint32_t warp = tid >> 5, lane = tid & 31u;
    __shared__ A wsum[32];
    __shared__ float s_inv;
    A acc;
    if constexpr (ACC == PD_ACC_DF) { acc.hi = 0.0f; acc.lo = 0.0f; } else { acc = (A)0; }
    // section 1 (n % 32 == 0 -> the float4 path of every kernel replaced):
    // fold nz raw partials -> proj, residual fmaf, write back, sum squares
    {
        const uint32_t n4 = n >> 2;
        float4* x4 = reinterpret_cast<float4*>(xb);
        const float4* p4 = reinterpret_cast<const float4*>(pb0);
        const float4* b4 = bias ? reinterpret_cast<const float4*>(bias) : nullptr;
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
            if constexpr (ACC == PD_ACC_DF) {
                pd_df_add(acc, v.x * v.x);
                pd_df_add(acc, v.y * v.y);
                pd_df_add(acc, v.z * v.z);
                pd_df_add(acc, v.w * v.w);
            } else {
                acc += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
            }
        }
    }
    for (uint32_t s = 16; s > 0; s >>= 1) {
        if constexpr (ACC == PD_ACC_DF) {
            pd_df o;
            o.hi = __shfl_down_sync(0xffffffffu, acc.hi, s);
            o.lo = __shfl_down_sync(0xffffffffu, acc.lo, s);
            acc = pd_df_merge(acc, o);
        } else {
            acc += __shfl_down_sync(0xffffffffu, acc, s);
        }
    }
    if (lane == 0) wsum[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        const uint32_t nwarps = (nth + 31u) >> 5;
        if constexpr (ACC == PD_ACC_DF) {
            pd_df sum; sum.hi = 0.0f; sum.lo = 0.0f;
            for (uint32_t wi = 0; wi < nwarps; ++wi) sum = pd_df_merge(sum, wsum[wi]);
            const double total = (double)sum.hi + (double)sum.lo;
            s_inv = 1.0f / sqrtf((float)(total / (double)n) + eps);
        } else if constexpr (F32DIV) {
            A sum = (A)0;
            for (uint32_t wi = 0; wi < nwarps; ++wi) sum += wsum[wi];
            s_inv = 1.0f / sqrtf(sum / (float)n + eps);
        } else {
            A sum = (A)0;
            for (uint32_t wi = 0; wi < nwarps; ++wi) sum += wsum[wi];
            const double total = (double)sum;
            s_inv = 1.0f / sqrtf((float)(total / (double)n) + eps);
        }
    }
    __syncthreads();   // also publishes section 1's x writes to the whole CTA
    const float inv = s_inv;
    // sections 2+3: normed row (optional f32 copy) + its nvf4 planes, one
    // 8-element walk per thread; lane pairs (2k, 2k+1) hold one 16-block
    const uint32_t n8 = n >> 3;
    for (uint32_t j = tid; j < n8; j += nth) {
        const uint32_t i = j << 3;
        const float4 v0 = *(const float4*)(xb + i), v1 = *(const float4*)(xb + i + 4u);
        const float4 w0 = *(const float4*)(w + i), w1 = *(const float4*)(w + i + 4u);
        float4 r0, r1;
        r0.x = v0.x * inv * w0.x; r0.y = v0.y * inv * w0.y;
        r0.z = v0.z * inv * w0.z; r0.w = v0.w * inv * w0.w;
        r1.x = v1.x * inv * w1.x; r1.y = v1.y * inv * w1.y;
        r1.z = v1.z * inv * w1.z; r1.w = v1.w * inv * w1.w;
        if (ob) { *(float4*)(ob + i) = r0; *(float4*)(ob + i + 4u) = r1; }
        pd_nvf4_quant8(r0, r1, lane, qb, sb, i);
    }
}

// acc_sel: 0 = the add_rmsnorm family (f32 accumulate, f32 divide) ==
// pd_add_rmsnorm_scaled_from_parts + quantize; 1 = the rmsnorm_batch family
// (pd_norm_acc_mode(), double divide) == sk_reduce + scale_add + rmsnorm_batch
// + quantize. `out` may be null (no f32 normed copy needed by a W4A4 consumer).
PD_EXPORT
int pd_add_rmsnorm_quant_nvf4_from_parts(void* x, const void* part, const void* w,
                                         void* out, const void* bias, void* q,
                                         void* scale, uint32_t n, float eps,
                                         uint32_t batch, float pscale, float scale2,
                                         uint32_t nz, uint32_t acc_sel, void* stream) {
    if (n == 0 || batch == 0) return 0;
    if ((n & 31u) != 0 || nz == 0) return cudaErrorInvalidValue;
    // decode-only contract: the replaced launches all pick pd_norm_decode_nth()
    // below 64 rows; wider rows belong to the mixed tick (its own path)
    if (batch >= 64u) return cudaErrorInvalidValue;
    const uint32_t nth = pd_norm_decode_nth();
    if (acc_sel == 0u) {
        pd_pdl_go((pd_add_rmsnorm_quant_nvf4_from_parts_kernel<PD_ACC_F32, true>), batch, nth, 0u,
                  (cudaStream_t)stream, (float*)x, (const float*)part, (const float*)w,
                  (float*)out, (const float*)bias, (unsigned char*)q, (unsigned char*)scale,
                  n, eps, pscale, scale2, batch, nz);
        return pd_launch_status();
    }
    const int accm = pd_norm_acc_mode();
    if (accm == PD_ACC_DF) {
        pd_pdl_go((pd_add_rmsnorm_quant_nvf4_from_parts_kernel<PD_ACC_DF, false>), batch, nth, 0u,
                  (cudaStream_t)stream, (float*)x, (const float*)part, (const float*)w,
                  (float*)out, (const float*)bias, (unsigned char*)q, (unsigned char*)scale,
                  n, eps, pscale, scale2, batch, nz);
    } else if (accm == PD_ACC_F64) {
        pd_pdl_go((pd_add_rmsnorm_quant_nvf4_from_parts_kernel<PD_ACC_F64, false>), batch, nth, 0u,
                  (cudaStream_t)stream, (float*)x, (const float*)part, (const float*)w,
                  (float*)out, (const float*)bias, (unsigned char*)q, (unsigned char*)scale,
                  n, eps, pscale, scale2, batch, nz);
    } else {
        pd_pdl_go((pd_add_rmsnorm_quant_nvf4_from_parts_kernel<PD_ACC_F32, false>), batch, nth, 0u,
                  (cudaStream_t)stream, (float*)x, (const float*)part, (const float*)w,
                  (float*)out, (const float*)bias, (unsigned char*)q, (unsigned char*)scale,
                  n, eps, pscale, scale2, batch, nz);
    }
    return pd_launch_status();
}

// ---------------------------------------------------------------------------
// F3: fold the gate|up split-K partials, swiglu, nvf4-quantize the down input.
// part holds nz slices of [rows, 2*ff] (stride rows*2*ff); the merged plane's
// column j is gate, ff + j is up (pd_swiglu_fused_kernel's layout).
// IL: the interleaved gate|up plane (Nvf4Plane::gu_pairs) --
// the partial slices hold (g,u,g,u,...) per row; the fold, silu and quant
// are the same expressions on the same values.
template <bool IL>
__global__ void pd_swiglu_quant_nvf4_from_parts_kernel(
    const float* __restrict__ part, const float* __restrict__ bias,
    unsigned char* __restrict__ q, unsigned char* __restrict__ scale,
    float scale2, uint32_t ff, uint32_t rows, uint32_t nz, uint32_t n) {
    PD_PDL_ARM();
    const uint32_t i = (blockIdx.x * 256u + threadIdx.x) * 8u;
    if (i >= n) return;
    const uint32_t tok = i / ff, j = i % ff;
    const size_t partN = (size_t)rows * 2u * ff;
    const float* row = part + (size_t)tok * 2u * ff;
    float4 v[2];
    #pragma unroll
    for (uint32_t h = 0; h < 2u; ++h) {
        float4 g, u;
        if constexpr (IL) {
            const float4 p0 = *(const float4*)(row + 2u * (j + h * 4u));
            const float4 p1 = *(const float4*)(row + 2u * (j + h * 4u) + 4u);
            g.x = p0.x; u.x = p0.y; g.y = p0.z; u.y = p0.w;
            g.z = p1.x; u.z = p1.y; g.w = p1.z; u.w = p1.w;
            for (uint32_t k = 1; k < nz; ++k) {
                const float4 q0 = *(const float4*)(row + (size_t)k * partN + 2u * (j + h * 4u));
                const float4 q1 = *(const float4*)(row + (size_t)k * partN + 2u * (j + h * 4u) + 4u);
                g.x += q0.x; u.x += q0.y; g.y += q0.z; u.y += q0.w;
                g.z += q1.x; u.z += q1.y; g.w += q1.z; u.w += q1.w;
            }
        } else {
            g = *(const float4*)(row + j + h * 4u);
            u = *(const float4*)(row + ff + j + h * 4u);
            for (uint32_t k = 1; k < nz; ++k) {
                const float4 gk = *(const float4*)(row + (size_t)k * partN + j + h * 4u);
                const float4 uk = *(const float4*)(row + (size_t)k * partN + ff + j + h * 4u);
                g.x += gk.x; g.y += gk.y; g.z += gk.z; g.w += gk.w;
                u.x += uk.x; u.y += uk.y; u.z += uk.z; u.w += uk.w;
            }
        }
        g.x *= scale2; g.y *= scale2; g.z *= scale2; g.w *= scale2;
        u.x *= scale2; u.y *= scale2; u.z *= scale2; u.w *= scale2;
        if (bias) {
            if constexpr (IL) {
                const float4 b0 = *(const float4*)(bias + 2u * (j + h * 4u));
                const float4 b1 = *(const float4*)(bias + 2u * (j + h * 4u) + 4u);
                g.x += b0.x; u.x += b0.y; g.y += b0.z; u.y += b0.w;
                g.z += b1.x; u.z += b1.y; g.w += b1.z; u.w += b1.w;
            } else {
                const float4 bg = *(const float4*)(bias + j + h * 4u);
                const float4 bu = *(const float4*)(bias + ff + j + h * 4u);
                g.x += bg.x; g.y += bg.y; g.z += bg.z; g.w += bg.w;
                u.x += bu.x; u.y += bu.y; u.z += bu.z; u.w += bu.w;
            }
        }
        v[h].x = (g.x / (1.0f + expf(-g.x))) * u.x;
        v[h].y = (g.y / (1.0f + expf(-g.y))) * u.y;
        v[h].z = (g.z / (1.0f + expf(-g.z))) * u.z;
        v[h].w = (g.w / (1.0f + expf(-g.w))) * u.w;
    }
    pd_nvf4_quant8(v[0], v[1], threadIdx.x & 31u, q, scale, i);
}
PD_EXPORT
int pd_swiglu_quant_nvf4_from_parts(const void* part, const void* bias, void* q,
                                    void* scale, uint32_t ff, uint32_t n_rows,
                                    float scale2, uint32_t nz, void* stream) {
    if (n_rows == 0 || ff == 0) return 0;
    if ((ff & 31u) != 0 || nz == 0) return cudaErrorInvalidValue;
    const uint32_t n = n_rows * ff;
    pd_swiglu_quant_nvf4_from_parts_kernel<false><<<(n / 8u + 255u) / 256u, 256u, 0,
                                                    (cudaStream_t)stream>>>(
        (const float*)part, (const float*)bias, (unsigned char*)q,
        (unsigned char*)scale, scale2, ff, n_rows, nz, n);
    return pd_launch_status();
}
PD_EXPORT
int pd_swiglu_quant_nvf4_from_parts_il(const void* part, const void* bias, void* q,
                                       void* scale, uint32_t ff, uint32_t n_rows,
                                       float scale2, uint32_t nz, void* stream) {
    if (n_rows == 0 || ff == 0) return 0;
    if ((ff & 31u) != 0 || nz == 0) return cudaErrorInvalidValue;
    const uint32_t n = n_rows * ff;
    pd_swiglu_quant_nvf4_from_parts_kernel<true><<<(n / 8u + 255u) / 256u, 256u, 0,
                                                   (cudaStream_t)stream>>>(
        (const float*)part, (const float*)bias, (unsigned char*)q,
        (unsigned char*)scale, scale2, ff, n_rows, nz, n);
    return pd_launch_status();
}

// NOTE on the sk_reduce vs from_parts float4 sum: the reduce is scalar
// (`s += part[k*n+i]` per element) and the from_parts float4 form does the
// same per-component scalar adds in the same k order -- the float4 is a load
// width, not an arithmetic change (the shipped from_parts kernel proved this
// with diffs=0 vs reduce-then-scaled).

// Interleaved twin of pd_swiglu_fused_nvf4: y row = (g0,u0,
// g1,u1,...) -- Nvf4Plane::gu_pairs. Each thread's 8 outputs are 16
// consecutive floats; same expression, same pd_nvf4_quant8.
__global__ void pd_swiglu_fused_nvf4_il_kernel(const float* __restrict__ fused,
                                               unsigned char* __restrict__ q,
                                               unsigned char* __restrict__ scale,
                                               uint32_t ff, uint32_t n) {
    const uint32_t i = (blockIdx.x * 256u + threadIdx.x) * 8u;
    if (i >= n) return;
    const uint32_t tok = i / ff, j = i % ff;
    const float* row = fused + (size_t)tok * 2u * ff + 2u * j;
    float4 v[2];
    #pragma unroll
    for (uint32_t h = 0; h < 2u; ++h) {
        const float4 p0 = *(const float4*)(row + h * 8u);        // g,u,g,u
        const float4 p1 = *(const float4*)(row + h * 8u + 4u);   // g,u,g,u
        v[h].x = (p0.x / (1.0f + expf(-p0.x))) * p0.y;
        v[h].y = (p0.z / (1.0f + expf(-p0.z))) * p0.w;
        v[h].z = (p1.x / (1.0f + expf(-p1.x))) * p1.y;
        v[h].w = (p1.z / (1.0f + expf(-p1.z))) * p1.w;
    }
    pd_nvf4_quant8(v[0], v[1], threadIdx.x & 31u, q, scale, i);
}
PD_EXPORT
int pd_swiglu_fused_nvf4_il(const void* fused, void* q, void* scale, uint32_t ff,
                            uint32_t n_rows, void* stream) {
    if (n_rows == 0 || ff == 0) return 0;
    if (ff & 31u) return cudaErrorInvalidValue;
    const uint32_t n = n_rows * ff;
    pd_swiglu_fused_nvf4_il_kernel<<<(n / 8u + 255u) / 256u, 256u, 0,
                                     (cudaStream_t)stream>>>(
        (const float*)fused, (unsigned char*)q, (unsigned char*)scale, ff, n);
    return pd_launch_status();
}

PD_EXPORT
int pd_quantize_nvf4_swiglu(const void* gate, const void* up, void* q, void* scale,
                            uint32_t n, void* stream) {
    if (n == 0) return 0;
    pd_quantize_nvf4_swiglu_kernel<<<(n / 8u + 255u) / 256u, 256u, 0,
                                     (cudaStream_t)stream>>>(
        (const float*)gate, (const float*)up, (unsigned char*)q,
        (unsigned char*)scale, n);
    return pd_launch_status();
}

#if PD_BS_OK
// The nvf4 k64 MMA: 4 e4m3 scale bytes per operand (one per 16 elements).
__device__ __forceinline__ void pd_nv4_mma(float d[4], uint32_t a0, uint32_t a1,
                                           uint32_t a2, uint32_t a3, uint32_t b0,
                                           uint32_t b1, uint32_t sfa, uint32_t sfb) {
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X"
        ".f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
        "{%10}, {0, 0}, {%11}, {0, 0};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "r"(sfb));
}
#endif

// nvf4 dense GEMM: the mxf4 kernel with 8 scale bytes per row per 128-chunk
// (u32 operand per k64) instead of 4. Same 80 B row strides, same ldmatrix
// fragment plan, same 2 blocks/SM.
//
// EPI adds the checkpoint-plane epilogue: acc*scale2 (the
// Nvf4Plane per-tensor global), +bias per out-row when present - the same
// order as the scalar nvf4 family. EPI=false leaves the store loop exactly
// as it was (scale2/bias unread), so the qwen35 projection lane's numerics
// are untouched.
template <bool EPI>
__global__ void __launch_bounds__(256, 2) pd_mxfp4_gemm_nv4_kernel(
    const unsigned char* __restrict__ data, const unsigned char* __restrict__ scale,
    const unsigned char* __restrict__ xq, const unsigned char* __restrict__ xs,
    float* __restrict__ y, float scale2, const float* __restrict__ bias,
    uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
#if PD_BS_OK
    extern __shared__ unsigned char pd_bs_sh[];
    unsigned char* wb0 = pd_bs_sh;
    unsigned char* wb1 = wb0 + 128u * PD_F4_WROW;
    unsigned char* yb0 = wb1 + 128u * PD_F4_WROW;
    unsigned char* yb1 = yb0 + 128u * PD_F4_YROW;

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, tq = lane & 3u;
    const uint32_t i0 = (warp >> 1) * 32u;
    const uint32_t joff = (warp & 1u) * 8u;
    const uint32_t n_kb = in_dim >> 5;
    const uint32_t n_k16 = in_dim >> 4;
    const uint32_t nk = (in_dim + 127u) / 128u;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t nct = batch_pad >> 7;
    const uint32_t tile = blockIdx.x;
    const uint32_t row_base = (tile / nct) * 128u;
    const uint32_t col_base = (tile % nct) * 128u;

    float acc[16][4] = {};

    #define PD_NV4_ISSUE_W(dst, kt)                                                   \
        for (uint32_t u = tid; u < 512u; u += 256u) {                                 \
            const uint32_t row = u >> 2, seg = u & 3u;                                \
            const bool ok = (row_base + row) < out_dim && (kt) * 4u + seg < n_kb;     \
            pd_cp_async16((int*)((dst) + row * PD_F4_WROW + seg * 16u),               \
                          data + (size_t)(row_base + row) * (in_dim >> 1) +           \
                              (kt) * 64u + seg * 16u,                                 \
                          ok);                                                        \
        }
    #define PD_NV4_ISSUE_Y(dst, kt)                                                   \
        for (uint32_t u = tid; u < 512u; u += 256u) {                                 \
            const uint32_t col = u >> 2, seg = u & 3u;                                \
            const bool ok =                                                           \
                (col_base + col) < batch && (kt) * 4u + seg < n_kb;                   \
            pd_cp_async16((int*)((dst) + col * PD_F4_YROW + 16u + seg * 16u),         \
                          xq + ((size_t)(ok ? col_base + col : 0u) * in_dim >> 1) +   \
                              (kt) * 64u + seg * 16u,                                 \
                          ok);                                                        \
        }

    PD_NV4_ISSUE_W(wb0, 0u)
    PD_NV4_ISSUE_Y(yb0, 0u)
    asm volatile("cp.async.commit_group;");
    for (uint32_t kt = 0; kt < nk; ++kt) {
        unsigned char* tw = (kt & 1u) ? wb1 : wb0;
        unsigned char* ty = (kt & 1u) ? yb1 : yb0;
        if (kt + 1u < nk) {
            PD_NV4_ISSUE_W((kt & 1u) ? wb0 : wb1, kt + 1u)
            PD_NV4_ISSUE_Y((kt & 1u) ? yb0 : yb1, kt + 1u)
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        {   // e4m3 planes: 8 bytes per row per chunk, 4 bytes per thread
            const uint32_t row = tid >> 1, b0 = (tid & 1u) * 4u;
            #pragma unroll
            for (uint32_t bb = 0; bb < 4u; ++bb) {
                const uint32_t kb16 = b0 + bb;
                const bool wok = (row_base + row) < out_dim && kt * 8u + kb16 < n_k16;
                tw[row * PD_F4_WROW + 64u + kb16] =
                    wok ? scale[(size_t)(row_base + row) * n_k16 + kt * 8u + kb16] : 0u;
                const bool yok = (col_base + row) < batch && kt * 8u + kb16 < n_k16;
                ty[row * PD_F4_YROW + kb16] =
                    yok ? xs[(size_t)(col_base + row) * n_k16 + kt * 8u + kb16] : 0u;
            }
        }
        __syncthreads();

        uint32_t am[2][2][4], sa[2][2];
        #pragma unroll
        for (uint32_t n = 0; n < 2u; ++n) {
            const uint32_t r0 = i0 + n * 16u + g;
            const uint32_t rs = (tq & 1u) ? r0 + 8u : r0;
            #pragma unroll
            for (uint32_t k64 = 0; k64 < 2u; ++k64) {
                pd_ldm_x4(am[n][k64],
                          tw + (i0 + n * 16u + ((lane >> 3) & 1u) * 8u + (lane & 7u)) *
                                  PD_F4_WROW +
                              k64 * 32u + (lane >> 4) * 16u);
                sa[n][k64] =
                    *(const uint32_t*)(tw + rs * PD_F4_WROW + 64u + k64 * 4u);
            }
        }
        #pragma unroll
        for (uint32_t j0 = 0; j0 < 128u; j0 += 16u) {
            uint32_t bm[4];
            pd_ldm_x4(bm, ty + (j0 + joff + (lane & 7u)) * PD_F4_YROW + 16u +
                              (lane >> 3) * 16u);
            const unsigned char* ysr = ty + (j0 + joff + g) * PD_F4_YROW;
            #pragma unroll
            for (uint32_t k64 = 0; k64 < 2u; ++k64) {
                const uint32_t sb = *(const uint32_t*)(ysr + k64 * 4u);
                #pragma unroll
                for (uint32_t n = 0; n < 2u; ++n)
                    pd_nv4_mma(acc[(j0 >> 3) + n], am[n][k64][0], am[n][k64][1],
                               am[n][k64][2], am[n][k64][3], bm[k64 * 2u],
                               bm[k64 * 2u + 1u], sa[n][k64], sb);
            }
        }
        __syncthreads();
    }
    #undef PD_NV4_ISSUE_W
    #undef PD_NV4_ISSUE_Y

    #pragma unroll
    for (uint32_t j0 = 0; j0 < 128u; j0 += 16u) {
        const uint32_t c0 = col_base + j0 + joff + 2u * tq;
        #pragma unroll
        for (uint32_t n = 0; n < 2u; ++n) {
            const uint32_t r0 = row_base + i0 + n * 16u + g;
            const uint32_t r8 = r0 + 8u;
            float v00 = acc[(j0 >> 3) + n][0], v01 = acc[(j0 >> 3) + n][1];
            float v10 = acc[(j0 >> 3) + n][2], v11 = acc[(j0 >> 3) + n][3];
            if constexpr (EPI) {
                v00 *= scale2; v01 *= scale2; v10 *= scale2; v11 *= scale2;
                // bias stays conditional (an unconditional +0.0 flips -0.0 -
                // the rung-7 lattice-gate lesson), reads row-guarded
                if (bias) {
                    if (r0 < out_dim) { const float b = bias[r0]; v00 += b; v01 += b; }
                    if (r8 < out_dim) { const float b = bias[r8]; v10 += b; v11 += b; }
                }
            }
            if (r0 < out_dim) {
                if (c0 < batch) y[(size_t)c0 * out_dim + r0] = v00;
                if (c0 + 1u < batch) y[(size_t)(c0 + 1u) * out_dim + r0] = v01;
            }
            if (r8 < out_dim) {
                if (c0 < batch) y[(size_t)c0 * out_dim + r8] = v10;
                if (c0 + 1u < batch) y[(size_t)(c0 + 1u) * out_dim + r8] = v11;
            }
        }
    }
#else
    (void)data; (void)scale; (void)xq; (void)xs; (void)y; (void)scale2;
    (void)bias; (void)in_dim; (void)out_dim; (void)batch;
#endif
}

PD_EXPORT
int pd_mxfp4_gemm_nv4(const void* data, const void* scale, const void* xq,
                      const void* xs, void* y, uint32_t in_dim, uint32_t out_dim,
                      uint32_t batch, void* stream) {
#ifndef PD_BS_HOST
    (void)data; (void)scale; (void)xq; (void)xs; (void)y; (void)in_dim;
    (void)out_dim; (void)batch; (void)stream;
    return cudaErrorNotSupported;
#else
    if (out_dim == 0 || batch == 0) return 0;
    if ((in_dim & 31u) != 0) return cudaErrorInvalidValue;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t ntiles = ((out_dim + 127u) / 128u) * (batch_pad >> 7);
    pd_mxfp4_gemm_nv4_kernel<false><<<ntiles, 256, PD_F4_SMEM, (cudaStream_t)stream>>>(
        (const unsigned char*)data, (const unsigned char*)scale,
        (const unsigned char*)xq, (const unsigned char*)xs, (float*)y, 1.0f,
        nullptr, in_dim, out_dim, batch);
    return pd_launch_status();
#endif
}

// Checkpoint-plane W4A4 GEMM: the same fp4 x fp4 block-scale mma
// kernel, EPI instantiation - acc*scale2 (+bias) per the Nvf4Plane contract.
// The checkpoint's own recipe (compressed-tensors group_1: input_activations
// 4-bit, group 16, e4m3 scales, dynamic-local) declares W4A4 serving, and
// that is how it is served - this arm matches that class instead of
// paying the bf16-mma rate for off-recipe W4A16 precision. xq/xs come from
// pd_quantize_nvf4[_swiglu]; weight planes are consumed in the checkpoint
// layout verbatim (adjacent-packed nibbles, flat e4m3-per-16 scale plane -
// the same indexing pd_nvf4_gemv uses).
PD_EXPORT
int pd_nvf4_gemm_f4(const void* data, const void* scale, const void* bias,
                    const void* xq, const void* xs, void* y, float scale2,
                    uint32_t in_dim, uint32_t out_dim, uint32_t batch,
                    void* stream) {
#ifndef PD_BS_HOST
    (void)data; (void)scale; (void)bias; (void)xq; (void)xs; (void)y;
    (void)scale2; (void)in_dim; (void)out_dim; (void)batch; (void)stream;
    return cudaErrorNotSupported;
#else
    if (out_dim == 0 || batch == 0) return 0;
    if ((in_dim & 31u) != 0) return cudaErrorInvalidValue;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t ntiles = ((out_dim + 127u) / 128u) * (batch_pad >> 7);
    pd_mxfp4_gemm_nv4_kernel<true><<<ntiles, 256, PD_F4_SMEM, (cudaStream_t)stream>>>(
        (const unsigned char*)data, (const unsigned char*)scale,
        (const unsigned char*)xq, (const unsigned char*)xs, (float*)y, scale2,
        (const float*)bias, in_dim, out_dim, batch);
    return pd_launch_status();
#endif
}

// v2 of the checkpoint W4A4 GEMM. Same 128x128 tile,
// fragment plan, mma and epilogue as pd_mxfp4_gemm_nv4_kernel<true> - the
// inner loop is restructured against the measured stall budget (at the
// qwen3.8 FFN shapes: mio_throttle 5.3, long_scoreboard 4.5, barrier 2.1
// cycles/inst; schedulers idle 82%):
//   1. Both e4m3 scale planes ride cp.async (one 8 B copy per row per
//      128-K chunk, same commit group as the data) - the synchronous
//      per-byte global scale loads were the long-scoreboard stall.
//   2. One __syncthreads per K-step, any ST: the wait->sync->issue->compute
//      order makes the top sync double as the WAR fence for the issue
//      target (read by compute(kt-1), which every warp passed before this
//      sync). ST=2 keeps 2 CTAs/SM; ST=3 trades residency for a deeper
//      ring (61.4 KB dyn smem, 1 CTA/SM - the rung-7 "barrier count beats
//      CTA residency" direction). The probe elects.
// Requires in_dim % 128 == 0 (both qwen3.8 FFN dims are; the launcher
// rejects others so the tail scale fetch can stay one aligned 8 B copy).
// SPLIT adds a K-split axis (the decode-starved grids:
// down at batch<=128 is 40 CTAs on 188 SMs): grid.y slices the K-chunk walk,
// each slice writes RAW partial sums (no scale2/bias) to `y` at slice
// stride batch*out_dim, and pd_nvf4_sk_reduce folds slices + epilogue in a
// fixed order - deterministic, same class as the unsplit kernel.
template <uint32_t ST, bool SPLIT>
__global__ void __launch_bounds__(256, ST >= 3 ? 1 : 2) pd_mxfp4_gemm_nv4b_kernel(
    const unsigned char* __restrict__ data, const unsigned char* __restrict__ scale,
    const unsigned char* __restrict__ xq, const unsigned char* __restrict__ xs,
    float* __restrict__ y, float scale2, const float* __restrict__ bias,
    uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
#if PD_BS_OK
    extern __shared__ unsigned char pd_bs_sh[];
    unsigned char* wb = pd_bs_sh;                              // ST x 128 x WROW
    unsigned char* ybs = pd_bs_sh + ST * 128u * PD_F4_WROW;    // ST x 128 x YROW

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, tq = lane & 3u;
    const uint32_t i0 = (warp >> 1) * 32u;
    const uint32_t joff = (warp & 1u) * 8u;
    const uint32_t n_kb = in_dim >> 5;
    const uint32_t n_k16 = in_dim >> 4;
    const uint32_t nk = in_dim >> 7;  // %128 == 0 by the launcher gate
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t nct = batch_pad >> 7;
    const uint32_t tile = blockIdx.x;
    const uint32_t row_base = (tile / nct) * 128u;
    const uint32_t col_base = (tile % nct) * 128u;

    float acc[16][4] = {};

    // one stage: 512 data copies (16 B) + 256 scale copies (8 B), all async
    #define PD_NV4B_ISSUE(kt)                                                         \
        {                                                                             \
            unsigned char* dw = wb + ((kt) % ST) * 128u * PD_F4_WROW;                 \
            unsigned char* dy = ybs + ((kt) % ST) * 128u * PD_F4_YROW;                \
            for (uint32_t u = tid; u < 512u; u += 256u) {                             \
                const uint32_t row = u >> 2, seg = u & 3u;                            \
                const bool wok = (row_base + row) < out_dim && (kt) * 4u + seg < n_kb;\
                pd_cp_async16((int*)(dw + row * PD_F4_WROW + seg * 16u),              \
                              data + (size_t)(row_base + row) * (in_dim >> 1) +       \
                                  (kt) * 64u + seg * 16u,                             \
                              wok);                                                   \
                const uint32_t col = row;                                             \
                const bool yok =                                                      \
                    (col_base + col) < batch && (kt) * 4u + seg < n_kb;               \
                pd_cp_async16((int*)(dy + col * PD_F4_YROW + 16u + seg * 16u),        \
                              xq + ((size_t)(yok ? col_base + col : 0u) * in_dim >>   \
                                    1) +                                              \
                                  (kt) * 64u + seg * 16u,                             \
                              yok);                                                   \
            }                                                                         \
            {                                                                         \
                const uint32_t r = tid & 127u;                                        \
                if (tid < 128u) {                                                     \
                    const bool ok = (row_base + r) < out_dim;                         \
                    pd_cpa8p(dw + r * PD_F4_WROW + 64u,                               \
                             scale + (size_t)(row_base + r) * n_k16 + (kt) * 8u, ok); \
                } else {                                                              \
                    const bool ok = (col_base + r) < batch;                           \
                    pd_cpa8p(dy + r * PD_F4_YROW,                                     \
                             xs + (size_t)(col_base + r) * n_k16 + (kt) * 8u, ok);    \
                }                                                                     \
            }                                                                         \
        }

    // K walk bounds: the whole chunk range unsplit; slice blockIdx.y's share
    // when SPLIT (ceil-divided so every chunk lands in exactly one slice)
    uint32_t k0 = 0, k1 = nk;
    if (SPLIT) {
        const uint32_t ck = (nk + gridDim.y - 1u) / gridDim.y;
        k0 = blockIdx.y * ck;
        k1 = k0 + ck < nk ? k0 + ck : nk;
        if (k0 > k1) k1 = k0;  // empty slice: acc stays 0, store writes zeros
    }
    #pragma unroll
    for (uint32_t s = 0; s < ST - 1u; ++s) {
        if (k0 + s < k1) PD_NV4B_ISSUE(k0 + s)
        asm volatile("cp.async.commit_group;");
    }
    for (uint32_t kt = k0; kt < k1; ++kt) {
        if (kt + ST - 1u < k1)
            asm volatile("cp.async.wait_group %0;" ::"n"(ST - 2u));
        else
            asm volatile("cp.async.wait_group 0;");
        __syncthreads();
        if (kt + ST - 1u < k1) {
            PD_NV4B_ISSUE(kt + ST - 1u)
            asm volatile("cp.async.commit_group;");
        }
        unsigned char* tw = wb + (kt % ST) * 128u * PD_F4_WROW;
        unsigned char* ty = ybs + (kt % ST) * 128u * PD_F4_YROW;

        uint32_t am[2][2][4], sa[2][2];
        #pragma unroll
        for (uint32_t n = 0; n < 2u; ++n) {
            const uint32_t r0 = i0 + n * 16u + g;
            const uint32_t rs = (tq & 1u) ? r0 + 8u : r0;
            #pragma unroll
            for (uint32_t k64 = 0; k64 < 2u; ++k64) {
                pd_ldm_x4(am[n][k64],
                          tw + (i0 + n * 16u + ((lane >> 3) & 1u) * 8u + (lane & 7u)) *
                                  PD_F4_WROW +
                              k64 * 32u + (lane >> 4) * 16u);
                sa[n][k64] =
                    *(const uint32_t*)(tw + rs * PD_F4_WROW + 64u + k64 * 4u);
            }
        }
        #pragma unroll
        for (uint32_t j0 = 0; j0 < 128u; j0 += 16u) {
            uint32_t bm[4];
            pd_ldm_x4(bm, ty + (j0 + joff + (lane & 7u)) * PD_F4_YROW + 16u +
                              (lane >> 3) * 16u);
            const unsigned char* ysr = ty + (j0 + joff + g) * PD_F4_YROW;
            #pragma unroll
            for (uint32_t k64 = 0; k64 < 2u; ++k64) {
                const uint32_t sb = *(const uint32_t*)(ysr + k64 * 4u);
                #pragma unroll
                for (uint32_t n = 0; n < 2u; ++n)
                    pd_nv4_mma(acc[(j0 >> 3) + n], am[n][k64][0], am[n][k64][1],
                               am[n][k64][2], am[n][k64][3], bm[k64 * 2u],
                               bm[k64 * 2u + 1u], sa[n][k64], sb);
            }
        }
    }
    #undef PD_NV4B_ISSUE

    float* yo = SPLIT ? y + (size_t)blockIdx.y * batch * out_dim : y;
    #pragma unroll
    for (uint32_t j0 = 0; j0 < 128u; j0 += 16u) {
        const uint32_t c0 = col_base + j0 + joff + 2u * tq;
        #pragma unroll
        for (uint32_t n = 0; n < 2u; ++n) {
            const uint32_t r0 = row_base + i0 + n * 16u + g;
            const uint32_t r8 = r0 + 8u;
            float v00 = acc[(j0 >> 3) + n][0], v01 = acc[(j0 >> 3) + n][1];
            float v10 = acc[(j0 >> 3) + n][2], v11 = acc[(j0 >> 3) + n][3];
            if (!SPLIT) {  // split writes RAW partials; the reduce owns the epilogue
                v00 *= scale2; v01 *= scale2; v10 *= scale2; v11 *= scale2;
                if (bias) {
                    if (r0 < out_dim) { const float b = bias[r0]; v00 += b; v01 += b; }
                    if (r8 < out_dim) { const float b = bias[r8]; v10 += b; v11 += b; }
                }
            }
            if (r0 < out_dim) {
                if (c0 < batch) yo[(size_t)c0 * out_dim + r0] = v00;
                if (c0 + 1u < batch) yo[(size_t)(c0 + 1u) * out_dim + r0] = v01;
            }
            if (r8 < out_dim) {
                if (c0 < batch) yo[(size_t)c0 * out_dim + r8] = v10;
                if (c0 + 1u < batch) yo[(size_t)(c0 + 1u) * out_dim + r8] = v11;
            }
        }
    }
#else
    (void)data; (void)scale; (void)xq; (void)xs; (void)y; (void)scale2;
    (void)bias; (void)in_dim; (void)out_dim; (void)batch;
#endif
}

// v2 launcher; `st` picks the ring depth (2 or 3) while the probe elects.
PD_EXPORT
int pd_nvf4_gemm_f4b(const void* data, const void* scale, const void* bias,
                     const void* xq, const void* xs, void* y, float scale2,
                     uint32_t in_dim, uint32_t out_dim, uint32_t batch,
                     uint32_t st, void* stream) {
#ifndef PD_BS_HOST
    (void)data; (void)scale; (void)bias; (void)xq; (void)xs; (void)y;
    (void)scale2; (void)in_dim; (void)out_dim; (void)batch; (void)st;
    (void)stream;
    return cudaErrorNotSupported;
#else
    if (out_dim == 0 || batch == 0) return 0;
    if ((in_dim & 127u) != 0) return cudaErrorInvalidValue;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t ntiles = ((out_dim + 127u) / 128u) * (batch_pad >> 7);
    if (st >= 3) {
        static int smem_set = 0;
        const uint32_t smem = 3u * 128u * (PD_F4_WROW + PD_F4_YROW);
        if (!smem_set) {
            cudaFuncSetAttribute(pd_mxfp4_gemm_nv4b_kernel<3, false>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)smem);
            smem_set = 1;
        }
        pd_mxfp4_gemm_nv4b_kernel<3, false>
            <<<ntiles, 256, smem, (cudaStream_t)stream>>>(
                (const unsigned char*)data, (const unsigned char*)scale,
                (const unsigned char*)xq, (const unsigned char*)xs, (float*)y,
                scale2, (const float*)bias, in_dim, out_dim, batch);
        return pd_launch_status();
    }
    pd_mxfp4_gemm_nv4b_kernel<2, false>
        <<<ntiles, 256, PD_F4_SMEM, (cudaStream_t)stream>>>(
            (const unsigned char*)data, (const unsigned char*)scale,
            (const unsigned char*)xq, (const unsigned char*)xs, (float*)y,
            scale2, (const float*)bias, in_dim, out_dim, batch);
    return pd_launch_status();
#endif
}

// v3: the KC=256 arm. At c16 decode v2 put
// the f4 GEMMs at 1.6-2x their weight floor (gate 54 us vs 33; the stall
// mix was mio_throttle + wait with schedulers idle 82%), and the rung-7
// law says barrier count beats residency on this die - so double the
// K-chunk per barrier: 20 iters instead of 40 on the gate walk, 2x the
// bytes in flight per stage, and the per-row scale fetch becomes one
// aligned 16 B cp.async. Row layout: 128 B nibbles + 16 B e4m3 scales +
// pad = 160 B both sides; ST=2 ring = 80 KB dyn smem, 1 CTA/SM. The
// compute walks the chunk as two KC=128 halves (kh ascending), so the
// global K order - and therefore the accumulation - is identical to
// v2/v1: bit-exact. Requires in_dim % 256 == 0 (both qwen3.8 FFN dims
// are; the launcher rejects others).
// The row stride is a template param. 160 B is the packed minimum (128 B
// nibbles + 16 B scales + 16 pad) but its 40-word pitch lands each ldmatrix
// phase on banks {0,8,16,24, 0,8,16,24} -- a built-in 2-way conflict on
// every fragment load, weight and activation side alike (v2's 80 B stride
// was conflict-free by accident; v3 traded that away for the halved
// barriers). 176 B = 44 words puts rows 0-7 on banks 0,12,24,4,16,28,8,20:
// conflict-free, still 16 B-aligned for cp.async, ring stays under the
// sm_120 cap (2x128x2x176 = 88 KB < 99 KB). Elected by a same-run A/B
// (v3 vs the v3s160 arm).
#define PD_F4C_WR 176u
#define PD_F4C_SMEM_W(WR) (2u * 128u * (2u * (WR)))
// Decode twin: a NARROW batch tile (BN < 128) + a tight row stride so both the
// weight+activation smem AND the accumulator registers shrink enough for
// 2 CTA/SM. At decode batch (<=32) the wide tile wasted 96 phantom batch cols
// (acc[16][4]=64 regs) and its 88 KB smem pinned the kernel at 1 CTA/SM /
// 16.6% occupancy (DRAM 40%, compute 20% -- pure occupancy
// starvation). BN=32, WRN=144 => 45 KB smem + acc[4][4] => 2 blocks/SM.
#define PD_F4CN_WR 144u
#define PD_F4CN_BN 32u
#define PD_F4CN_SMEM (2u * 128u * PD_F4CN_WR + 2u * PD_F4CN_BN * PD_F4CN_WR)

// BN = batch-tile columns (128 = the FFN wide tile, unchanged; <128 = decode).
// ST = cp.async ring depth. ST=2 is the original kernel,
// instruction for instruction (prologue issues one chunk, wait_group 0).
// Deeper rings exist for the small-out DECODE shapes (qkv 48 tiles, o 32):
// with ~16 mma per warp per 256-K chunk the compute is far below one DRAM
// latency, so a 2-stage ring that waits for everything before each chunk
// runs latency-SERIALIZED -- one DRAM round trip per chunk, ~50% of the byte
// floor at every split depth. ST-1 chunks in flight
// turn that into one latency plus the byte stream. Same tile, same fragment
// and mma order per chunk: the accumulation sequence per output is
// unchanged, so ST changes nothing numerically.
// RT = row tile: 128 (the original warp layout: 4 row groups x
// 2 col halves of 16 batch cols) or 64 (2 row groups x 4 col groups of 8 --
// twice the CTAs per plane for the small-out decode shapes, whose single CTA
// per SM caps at ~26 GB/s of cp.async issue). Each output's K accumulation
// sequence is the same in both layouts.
template <bool SPLIT, uint32_t WR, uint32_t BN = 128u, uint32_t ST = 2u, uint32_t RT = 128u>
__global__ void __launch_bounds__(256, (BN <= 32u && (ST <= 2u || (RT <= 64u && ST <= 3u))) ? 2 : 1) pd_mxfp4_gemm_nv4c_kernel(
    const unsigned char* __restrict__ data, const unsigned char* __restrict__ scale,
    const unsigned char* __restrict__ xq, const unsigned char* __restrict__ xs,
    float* __restrict__ y, float scale2, const float* __restrict__ bias,
    uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
#if PD_BS_OK
    extern __shared__ unsigned char pd_bs_sh[];
    unsigned char* wb = pd_bs_sh;                                // ST x RT x WR
    unsigned char* ybs = pd_bs_sh + ST * RT * WR;       // ST x BN x WR

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, tq = lane & 3u;
    // RT=128: warp -> (row group of 32) x (16-col half); RT=64: (row group of
    // 32) x (8-col quarter), i.e. one 8-col block per warp
    const uint32_t i0 = (RT == 128u ? (warp >> 1) : (warp >> 2)) * 32u;
    const uint32_t joff = (RT == 128u ? (warp & 1u) : (warp & 3u)) * 8u;
    constexpr uint32_t JW = RT == 128u ? BN : 16u;   // j0 loop extent (step 16)
    const uint32_t n_kb = in_dim >> 5;
    const uint32_t n_k16 = in_dim >> 4;
    const uint32_t nk = in_dim >> 8;  // 256-wide chunks (launcher gates %256)
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t nct = batch_pad >> 7;
    const uint32_t tile = blockIdx.x;
    const uint32_t row_base = (tile / nct) * RT;
    const uint32_t col_base = (tile % nct) * 128u;

    float acc[BN / 8u][4] = {};

    // one stage: 8 x 16 B data copies per row per side (u < 1024 at 256
    // threads = 4 iters) + one 16 B scale copy per row per side
    #define PD_NV4C_ISSUE(kt)                                                         \
        {                                                                             \
            unsigned char* dw = wb + ((kt) % ST) * RT * WR;                  \
            unsigned char* dy = ybs + ((kt) % ST) * BN * WR;                  \
            for (uint32_t u = tid; u < RT * 8u; u += 256u) {                          \
                const uint32_t row = u >> 3, seg = u & 7u;                            \
                const bool wok = (row_base + row) < out_dim && (kt) * 8u + seg < n_kb;\
                pd_cp_async16((int*)(dw + row * WR + seg * 16u),             \
                              data + (size_t)(row_base + row) * (in_dim >> 1) +       \
                                  (kt) * 128u + seg * 16u,                            \
                              wok);                                                   \
                const bool yok =                                                      \
                    row < BN && (col_base + row) < batch && (kt) * 8u + seg < n_kb;   \
                if (row < BN)                                                         \
                    pd_cp_async16((int*)(dy + row * WR + 16u + seg * 16u),   \
                                  xq + ((size_t)(yok ? col_base + row : 0u) * in_dim >>\
                                        1) +                                          \
                                      (kt) * 128u + seg * 16u,                        \
                                  yok);                                               \
            }                                                                         \
            {                                                                         \
                const uint32_t r = tid & 127u;                                        \
                if (tid < RT) {                                                       \
                    const bool ok = (row_base + r) < out_dim;                         \
                    pd_cp_async16((int*)(dw + r * WR + 128u),                \
                                  scale + (size_t)(row_base + r) * n_k16 +            \
                                      (kt) * 16u,                                     \
                                  ok);                                                \
                } else if (r < BN) {                                                  \
                    const bool ok = (col_base + r) < batch;                           \
                    pd_cp_async16((int*)(dy + r * WR),                       \
                                  xs + (size_t)(col_base + r) * n_k16 + (kt) * 16u,   \
                                  ok);                                                \
                }                                                                     \
            }                                                                         \
        }

    uint32_t k0 = 0, k1 = nk;
    if (SPLIT) {
        const uint32_t ck = (nk + gridDim.y - 1u) / gridDim.y;
        k0 = blockIdx.y * ck;
        k1 = k0 + ck < nk ? k0 + ck : nk;
        if (k0 > k1) k1 = k0;
    }
    // prologue: ST-1 chunks in flight, one commit group each
    #pragma unroll
    for (uint32_t s = 0; s < ST - 1u; ++s) {
        if (k0 + s < k1) PD_NV4C_ISSUE(k0 + s)
        asm volatile("cp.async.commit_group;");
    }
    for (uint32_t kt = k0; kt < k1; ++kt) {
        // chunk kt's group is the oldest of the ST-1 outstanding: leave ST-2
        // pending (groups retire in order; the tail commits empty groups so
        // the count stays uniform)
        asm volatile("cp.async.wait_group %0;" ::"n"(ST - 2u));
        __syncthreads();
        if (kt + (ST - 1u) < k1) PD_NV4C_ISSUE(kt + (ST - 1u))
        asm volatile("cp.async.commit_group;");
        unsigned char* tw = wb + (kt % ST) * RT * WR;
        unsigned char* ty = ybs + (kt % ST) * BN * WR;

        // two KC=128 halves, kh ascending - the same fragment/mma sequence
        // as v2 per half, so the accumulation order matches bit-for-bit.
        // (A register-staged variant - hoist all bm/sbv then mma back-to-back
        // - measured EQUAL, 457.8 vs 457.0 us at gate b2048: ptxas already
        // schedules across these unrolled loops. The wait stalls the
        // profiler shows are real mma latency at 8 warps/SM, not an
        // artifact of scheduling.)
        #pragma unroll
        for (uint32_t kh = 0; kh < 2u; ++kh) {
            const uint32_t db = kh * 64u;   // data byte base within the row
            const uint32_t sb8 = kh * 8u;   // scale byte base within the 16
            uint32_t am[2][2][4], sa[2][2];
            #pragma unroll
            for (uint32_t n = 0; n < 2u; ++n) {
                const uint32_t r0 = i0 + n * 16u + g;
                const uint32_t rs = (tq & 1u) ? r0 + 8u : r0;
                #pragma unroll
                for (uint32_t k64 = 0; k64 < 2u; ++k64) {
                    pd_ldm_x4(am[n][k64],
                              tw + (i0 + n * 16u + ((lane >> 3) & 1u) * 8u +
                                    (lane & 7u)) *
                                      WR +
                                  db + k64 * 32u + (lane >> 4) * 16u);
                    sa[n][k64] = *(const uint32_t*)(tw + rs * WR + 128u +
                                                    sb8 + k64 * 4u);
                }
            }
            #pragma unroll
            for (uint32_t j0 = 0; j0 < JW; j0 += 16u) {
                uint32_t bm[4];
                pd_ldm_x4(bm, ty + (j0 + joff + (lane & 7u)) * WR + 16u +
                                  db + (lane >> 3) * 16u);
                const unsigned char* ysr = ty + (j0 + joff + g) * WR + sb8;
                #pragma unroll
                for (uint32_t k64 = 0; k64 < 2u; ++k64) {
                    const uint32_t sbv = *(const uint32_t*)(ysr + k64 * 4u);
                    #pragma unroll
                    for (uint32_t n = 0; n < 2u; ++n)
                        pd_nv4_mma(acc[(j0 >> 3) + n], am[n][k64][0], am[n][k64][1],
                                   am[n][k64][2], am[n][k64][3], bm[k64 * 2u],
                                   bm[k64 * 2u + 1u], sa[n][k64], sbv);
                }
            }
        }
    }
    #undef PD_NV4C_ISSUE

    float* yo = SPLIT ? y + (size_t)blockIdx.y * batch * out_dim : y;
    #pragma unroll
    for (uint32_t j0 = 0; j0 < JW; j0 += 16u) {
        const uint32_t c0 = col_base + j0 + joff + 2u * tq;
        #pragma unroll
        for (uint32_t n = 0; n < 2u; ++n) {
            const uint32_t r0 = row_base + i0 + n * 16u + g;
            const uint32_t r8 = r0 + 8u;
            float v00 = acc[(j0 >> 3) + n][0], v01 = acc[(j0 >> 3) + n][1];
            float v10 = acc[(j0 >> 3) + n][2], v11 = acc[(j0 >> 3) + n][3];
            if (!SPLIT) {
                v00 *= scale2; v01 *= scale2; v10 *= scale2; v11 *= scale2;
                if (bias) {
                    if (r0 < out_dim) { const float b = bias[r0]; v00 += b; v01 += b; }
                    if (r8 < out_dim) { const float b = bias[r8]; v10 += b; v11 += b; }
                }
            }
            if (r0 < out_dim) {
                if (c0 < batch) yo[(size_t)c0 * out_dim + r0] = v00;
                if (c0 + 1u < batch) yo[(size_t)(c0 + 1u) * out_dim + r0] = v01;
            }
            if (r8 < out_dim) {
                if (c0 < batch) yo[(size_t)c0 * out_dim + r8] = v10;
                if (c0 + 1u < batch) yo[(size_t)(c0 + 1u) * out_dim + r8] = v11;
            }
        }
    }
#else
    (void)data; (void)scale; (void)xq; (void)xs; (void)y; (void)scale2;
    (void)bias; (void)in_dim; (void)out_dim; (void)batch;
#endif
}

__global__ void pd_nvf4_sk_reduce_kernel(const float* __restrict__ part,
                                         const float* __restrict__ bias,
                                         float* __restrict__ y, float scale2,
                                         uint32_t n, uint32_t out_dim,
                                         uint32_t sk);

// Decode launcher: the BN=32 / WR=144 narrow-tile twin at
// launch_bounds(256,2) => 2 CTA/SM. batch must be <= 32 (its 32-col tile does
// not cover more). Always split (sk>=2) so the block count fills 2/SM; the
// caller passes the sk. Bit-exact vs pd_nvf4_gemm_f4c<false> at the same shape
// (same fragment/mma order, just a narrower batch extent; the split partials
// are raw, folded by pd_nvf4_sk_reduce identically).
PD_EXPORT
int pd_nvf4_gemm_f4cn(const void* data, const void* scale, const void* bias,
                      const void* xq, const void* xs, void* part, void* y,
                      float scale2, uint32_t in_dim, uint32_t out_dim,
                      uint32_t batch, uint32_t sk, void* stream) {
#ifndef PD_BS_HOST
    (void)data;(void)scale;(void)bias;(void)xq;(void)xs;(void)part;(void)y;
    (void)scale2;(void)in_dim;(void)out_dim;(void)batch;(void)sk;(void)stream;
    return cudaErrorNotSupported;
#else
    if (out_dim == 0 || batch == 0) return 0;
    if ((in_dim & 255u) != 0 || batch > PD_F4CN_BN) return cudaErrorInvalidValue;
    const uint32_t ntiles = (out_dim + 127u) / 128u;   // One 32-col batch tile
    static int smem_set = 0;
    if (!smem_set) {
        cudaFuncSetAttribute(pd_mxfp4_gemm_nv4c_kernel<false, PD_F4CN_WR, PD_F4CN_BN>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)PD_F4CN_SMEM);
        cudaFuncSetAttribute(pd_mxfp4_gemm_nv4c_kernel<true, PD_F4CN_WR, PD_F4CN_BN>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)PD_F4CN_SMEM);
        smem_set = 1;
    }
    if (sk >= 2u) {
        dim3 grid(ntiles, sk);
        pd_mxfp4_gemm_nv4c_kernel<true, PD_F4CN_WR, PD_F4CN_BN>
            <<<grid, 256, PD_F4CN_SMEM, (cudaStream_t)stream>>>(
                (const unsigned char*)data, (const unsigned char*)scale,
                (const unsigned char*)xq, (const unsigned char*)xs,
                (float*)part, scale2, (const float*)bias, in_dim, out_dim, batch);
        int rc = pd_launch_status(); if (rc) return rc;
        const uint32_t n = batch * out_dim;
        pd_nvf4_sk_reduce_kernel<<<(n + 255u) / 256u, 256u, 0, (cudaStream_t)stream>>>(
            (const float*)part, (const float*)bias, (float*)y, scale2, n, out_dim, sk);
        return pd_launch_status();
    }
    pd_mxfp4_gemm_nv4c_kernel<false, PD_F4CN_WR, PD_F4CN_BN>
        <<<ntiles, 256, PD_F4CN_SMEM, (cudaStream_t)stream>>>(
            (const unsigned char*)data, (const unsigned char*)scale,
            (const unsigned char*)xq, (const unsigned char*)xs, (float*)y,
            scale2, (const float*)bias, in_dim, out_dim, batch);
    return pd_launch_status();
#endif
}

// f4cd: the f4cn tile with a DEEP cp.async ring (ST = 3 or 4
// chunks in flight, 69 / 92 KB dynamic smem, 1 CTA/SM) for the small-out
// decode shapes whose grids never fill the die (qkv 48 tiles, o 32, qwen3.8's
// 40-56): there the 2-stage ring is latency-serialized and a split only adds
// partials traffic. Same kernel body as f4cn -- the ring depth is a template
// parameter -- so sk==1 here is bit-identical to f4cn sk==1 and a split here
// writes the same partial slices f4cn's split does. `st` selects 3 or 4.
#define PD_F4CD_SMEM(ST, RT) ((ST) * ((RT) * PD_F4CN_WR + PD_F4CN_BN * PD_F4CN_WR))
template <bool SPLIT, uint32_t ST, uint32_t RT>
static inline void pd_f4cd_attr_once() {
    static int done = 0;
    if (!done) {
        cudaFuncSetAttribute(pd_mxfp4_gemm_nv4c_kernel<SPLIT, PD_F4CN_WR, PD_F4CN_BN, ST, RT>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)PD_F4CD_SMEM(ST, RT));
        done = 1;
    }
}
template <bool SPLIT, uint32_t ST, uint32_t RT>
static inline int pd_f4cd_launch(const void* data, const void* scale, const void* bias,
                                 const void* xq, const void* xs, float* out, float scale2,
                                 uint32_t in_dim, uint32_t out_dim, uint32_t batch,
                                 uint32_t sk, cudaStream_t stream) {
    pd_f4cd_attr_once<SPLIT, ST, RT>();
    const uint32_t ntiles = (out_dim + RT - 1u) / RT;
    dim3 grid(ntiles, SPLIT ? sk : 1u);
    pd_mxfp4_gemm_nv4c_kernel<SPLIT, PD_F4CN_WR, PD_F4CN_BN, ST, RT>
        <<<grid, 256, PD_F4CD_SMEM(ST, RT), stream>>>(
            (const unsigned char*)data, (const unsigned char*)scale,
            (const unsigned char*)xq, (const unsigned char*)xs, out, scale2,
            (const float*)bias, in_dim, out_dim, batch);
    return pd_launch_status();
}
// (st, rt) -> one of the four instantiations; rt 128 = the f4cn layout
template <bool SPLIT>
static inline int pd_f4cd_dispatch(uint32_t st, uint32_t rt, const void* data, const void* scale,
                                   const void* bias, const void* xq, const void* xs, float* out,
                                   float scale2, uint32_t in_dim, uint32_t out_dim, uint32_t batch,
                                   uint32_t sk, cudaStream_t stream) {
    if (rt == 64u)
        return st == 4u ? pd_f4cd_launch<SPLIT, 4u, 64u>(data, scale, bias, xq, xs, out, scale2, in_dim, out_dim, batch, sk, stream)
                        : pd_f4cd_launch<SPLIT, 3u, 64u>(data, scale, bias, xq, xs, out, scale2, in_dim, out_dim, batch, sk, stream);
    return st == 4u ? pd_f4cd_launch<SPLIT, 4u, 128u>(data, scale, bias, xq, xs, out, scale2, in_dim, out_dim, batch, sk, stream)
                    : pd_f4cd_launch<SPLIT, 3u, 128u>(data, scale, bias, xq, xs, out, scale2, in_dim, out_dim, batch, sk, stream);
}
PD_EXPORT
int pd_nvf4_gemm_f4cd(const void* data, const void* scale, const void* bias,
                      const void* xq, const void* xs, void* part, void* y,
                      float scale2, uint32_t in_dim, uint32_t out_dim,
                      uint32_t batch, uint32_t sk, uint32_t st, uint32_t rt, void* stream) {
#ifndef PD_BS_HOST
    (void)data;(void)scale;(void)bias;(void)xq;(void)xs;(void)part;(void)y;
    (void)scale2;(void)in_dim;(void)out_dim;(void)batch;(void)sk;(void)st;(void)rt;(void)stream;
    return cudaErrorNotSupported;
#else
    if (out_dim == 0 || batch == 0) return 0;
    if ((in_dim & 255u) != 0 || batch > PD_F4CN_BN || (st != 3u && st != 4u) || (rt != 64u && rt != 128u)) return cudaErrorInvalidValue;
    cudaStream_t cs = (cudaStream_t)stream;
    if (sk >= 2u) {
        if (!part) return cudaErrorInvalidValue;
        int rc = pd_f4cd_dispatch<true>(st, rt, data, scale, bias, xq, xs, (float*)part, scale2, in_dim, out_dim, batch, sk, cs);
        if (rc) return rc;
        const uint32_t n = batch * out_dim;
        pd_nvf4_sk_reduce_kernel<<<(n + 255u) / 256u, 256u, 0, cs>>>(
            (const float*)part, (const float*)bias, (float*)y, scale2, n, out_dim, sk);
        return pd_launch_status();
    }
    return pd_f4cd_dispatch<false>(st, rt, data, scale, bias, xq, xs, (float*)y, scale2, in_dim, out_dim, batch, 1u, cs);
#endif
}
// raw twin: `sk` (>= 1) raw partial slices into `part`, no reduce, no scale2
// -- the from_parts consumers fold nz = sk slices (nz == 1 is a plain fold).
PD_EXPORT
int pd_nvf4_gemm_f4cd_raw(const void* data, const void* scale, const void* bias,
                          const void* xq, const void* xs, void* part,
                          float scale2, uint32_t in_dim, uint32_t out_dim,
                          uint32_t batch, uint32_t sk, uint32_t st, uint32_t rt, void* stream) {
#ifndef PD_BS_HOST
    (void)data;(void)scale;(void)bias;(void)xq;(void)xs;(void)part;
    (void)scale2;(void)in_dim;(void)out_dim;(void)batch;(void)sk;(void)st;(void)rt;(void)stream;
    return cudaErrorNotSupported;
#else
    if (out_dim == 0 || batch == 0) return 0;
    if ((in_dim & 255u) != 0 || batch > PD_F4CN_BN || sk < 1u || (st != 3u && st != 4u) || (rt != 64u && rt != 128u)) return cudaErrorInvalidValue;
    cudaStream_t cs = (cudaStream_t)stream;
    return pd_f4cd_dispatch<true>(st, rt, data, scale, bias, xq, xs, (float*)part, scale2, in_dim, out_dim, batch, sk, cs);
#endif
}

// f4cn RAW twin (the nvf4 reduce-fold): the same split GEMM as
// pd_nvf4_gemm_f4cn but STOPS after writing the `sk` raw partial slices into
// `part` -- no pd_nvf4_sk_reduce launch, no `y` round trip. The consumer folds
// the slices (fixed-order sum * scale2) inline, exactly as the fused q|k|v seat
// does via pd_nvf4_gemm_f4c_raw. Keeps f4cn's 2-CTA/SM decode speed (which
// pd_nvf4_gemm_f4c_raw -- the f4c/1-CTA arm -- does not). sk must be >= 2 (the
// unsplit path has nothing to fold; the caller keeps pd_nvf4_gemm_f4cn there).
PD_EXPORT
int pd_nvf4_gemm_f4cn_raw(const void* data, const void* scale, const void* bias,
                          const void* xq, const void* xs, void* part,
                          float scale2, uint32_t in_dim, uint32_t out_dim,
                          uint32_t batch, uint32_t sk, void* stream) {
#ifndef PD_BS_HOST
    (void)data;(void)scale;(void)bias;(void)xq;(void)xs;(void)part;
    (void)scale2;(void)in_dim;(void)out_dim;(void)batch;(void)sk;(void)stream;
    return cudaErrorNotSupported;
#else
    if (out_dim == 0 || batch == 0) return 0;
    if ((in_dim & 255u) != 0 || batch > PD_F4CN_BN || sk < 2u) return cudaErrorInvalidValue;
    const uint32_t ntiles = (out_dim + 127u) / 128u;
    static int smem_set = 0;
    if (!smem_set) {
        cudaFuncSetAttribute(pd_mxfp4_gemm_nv4c_kernel<true, PD_F4CN_WR, PD_F4CN_BN>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)PD_F4CN_SMEM);
        smem_set = 1;
    }
    dim3 grid(ntiles, sk);
    pd_mxfp4_gemm_nv4c_kernel<true, PD_F4CN_WR, PD_F4CN_BN>
        <<<grid, 256, PD_F4CN_SMEM, (cudaStream_t)stream>>>(
            (const unsigned char*)data, (const unsigned char*)scale,
            (const unsigned char*)xq, (const unsigned char*)xs,
            (float*)part, scale2, (const float*)bias, in_dim, out_dim, batch);
    return pd_launch_status();
#endif
}

// v3 launcher (probe-elected before any ABI slot): plain or split via sk.
PD_EXPORT
int pd_nvf4_gemm_f4c(const void* data, const void* scale, const void* bias,
                     const void* xq, const void* xs, void* part, void* y,
                     float scale2, uint32_t in_dim, uint32_t out_dim,
                     uint32_t batch, uint32_t sk, void* stream) {
#ifndef PD_BS_HOST
    (void)data; (void)scale; (void)bias; (void)xq; (void)xs; (void)part;
    (void)y; (void)scale2; (void)in_dim; (void)out_dim; (void)batch; (void)sk;
    (void)stream;
    return cudaErrorNotSupported;
#else
    if (out_dim == 0 || batch == 0) return 0;
    if ((in_dim & 255u) != 0) return cudaErrorInvalidValue;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t ntiles = ((out_dim + 127u) / 128u) * (batch_pad >> 7);
    static int smem_set = 0;
    if (!smem_set) {
        cudaFuncSetAttribute(pd_mxfp4_gemm_nv4c_kernel<false, PD_F4C_WR>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)PD_F4C_SMEM_W(PD_F4C_WR));
        cudaFuncSetAttribute(pd_mxfp4_gemm_nv4c_kernel<true, PD_F4C_WR>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)PD_F4C_SMEM_W(PD_F4C_WR));
        smem_set = 1;
    }
    if (sk >= 2u) {
        dim3 grid(ntiles, sk);
        pd_mxfp4_gemm_nv4c_kernel<true, PD_F4C_WR>
            <<<grid, 256, PD_F4C_SMEM_W(PD_F4C_WR), (cudaStream_t)stream>>>(
                (const unsigned char*)data, (const unsigned char*)scale,
                (const unsigned char*)xq, (const unsigned char*)xs,
                (float*)part, scale2, (const float*)bias, in_dim, out_dim,
                batch);
        int rc = pd_launch_status();
        if (rc != 0) return rc;
        const uint32_t n = batch * out_dim;
        pd_nvf4_sk_reduce_kernel<<<(n + 255u) / 256u, 256u, 0,
                                   (cudaStream_t)stream>>>(
            (const float*)part, (const float*)bias, (float*)y, scale2, n,
            out_dim, sk);
        return pd_launch_status();
    }
    pd_mxfp4_gemm_nv4c_kernel<false, PD_F4C_WR>
        <<<ntiles, 256, PD_F4C_SMEM_W(PD_F4C_WR), (cudaStream_t)stream>>>(
            (const unsigned char*)data, (const unsigned char*)scale,
            (const unsigned char*)xq, (const unsigned char*)xs, (float*)y,
            scale2, (const float*)bias, in_dim, out_dim, batch);
    return pd_launch_status();
#endif
}

// Raw-partials twin of the f4c split launcher (granite NVFP4
// fused qkv): the same split GEMM, but the fold + epilogue are left to the
// CONSUMER (pd_qkv_rope_norm_from_parts_paged sums the sk planes in the same
// fixed order and applies scale2 after the fold -- bit-identical to
// pd_nvf4_sk_reduce, minus one launch and one y round trip per GEMM).
// Layout in `part`: [sk][batch][out_dim] f32, slice stride batch*out_dim.
// No bias (the callers that have one keep the reducing launcher).
PD_EXPORT
int pd_nvf4_gemm_f4c_raw(const void* data, const void* scale, const void* xq,
                         const void* xs, void* part, uint32_t in_dim,
                         uint32_t out_dim, uint32_t batch, uint32_t sk,
                         void* stream) {
#ifndef PD_BS_HOST
    (void)data; (void)scale; (void)xq; (void)xs; (void)part; (void)in_dim;
    (void)out_dim; (void)batch; (void)sk; (void)stream;
    return cudaErrorNotSupported;
#else
    if (out_dim == 0 || batch == 0) return 0;
    if ((in_dim & 255u) != 0 || sk < 2u) return cudaErrorInvalidValue;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t ntiles = ((out_dim + 127u) / 128u) * (batch_pad >> 7);
    static int smem_set = 0;
    if (!smem_set) {
        cudaFuncSetAttribute(pd_mxfp4_gemm_nv4c_kernel<true, PD_F4C_WR>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)PD_F4C_SMEM_W(PD_F4C_WR));
        smem_set = 1;
    }
    dim3 grid(ntiles, sk);
    // scale2 = 1 and no bias: the split kernel writes RAW slice sums either
    // way (the epilogue lives in the fold), so these are documentary only
    pd_mxfp4_gemm_nv4c_kernel<true, PD_F4C_WR>
        <<<grid, 256, PD_F4C_SMEM_W(PD_F4C_WR), (cudaStream_t)stream>>>(
            (const unsigned char*)data, (const unsigned char*)scale,
            (const unsigned char*)xq, (const unsigned char*)xs,
            (float*)part, 1.0f, (const float*)nullptr, in_dim, out_dim,
            batch);
    return pd_launch_status();
#endif
}

// Fold the SK raw partial planes (fixed slice order - deterministic) and
// apply the checkpoint epilogue once: y = scale2 * sum (+bias[out-row]).
__global__ void pd_nvf4_sk_reduce_kernel(const float* __restrict__ part,
                                         const float* __restrict__ bias,
                                         float* __restrict__ y, float scale2,
                                         uint32_t n, uint32_t out_dim,
                                         uint32_t sk) {
    const uint32_t i = blockIdx.x * 256u + threadIdx.x;
    if (i >= n) return;
    float s = part[i];
    for (uint32_t k = 1; k < sk; ++k) s += part[(size_t)k * n + i];
    float v = s * scale2;
    if (bias) v += bias[i % out_dim];
    y[i] = v;
}

// Split-K launcher: the decode-band grids starve the
// machine (down at batch<=128 is 40 CTAs on 188 SMs), so grid.y slices the
// K walk into `sk` ranges writing raw partials into `part` (>= sk * batch *
// out_dim f32), then one elementwise reduce folds them with the epilogue.
// ST=3 ring (the probe-elected inner loop).
PD_EXPORT
int pd_nvf4_gemm_f4s(const void* data, const void* scale, const void* bias,
                     const void* xq, const void* xs, void* part, void* y,
                     float scale2, uint32_t in_dim, uint32_t out_dim,
                     uint32_t batch, uint32_t sk, void* stream) {
#ifndef PD_BS_HOST
    (void)data; (void)scale; (void)bias; (void)xq; (void)xs; (void)part;
    (void)y; (void)scale2; (void)in_dim; (void)out_dim; (void)batch; (void)sk;
    (void)stream;
    return cudaErrorNotSupported;
#else
    if (out_dim == 0 || batch == 0) return 0;
    if ((in_dim & 127u) != 0 || sk < 2u) return cudaErrorInvalidValue;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t ntiles = ((out_dim + 127u) / 128u) * (batch_pad >> 7);
    static int smem_set = 0;
    const uint32_t smem = 3u * 128u * (PD_F4_WROW + PD_F4_YROW);
    if (!smem_set) {
        cudaFuncSetAttribute(pd_mxfp4_gemm_nv4b_kernel<3, true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem);
        smem_set = 1;
    }
    dim3 grid(ntiles, sk);
    pd_mxfp4_gemm_nv4b_kernel<3, true><<<grid, 256, smem, (cudaStream_t)stream>>>(
        (const unsigned char*)data, (const unsigned char*)scale,
        (const unsigned char*)xq, (const unsigned char*)xs, (float*)part,
        scale2, (const float*)bias, in_dim, out_dim, batch);
    int rc = pd_launch_status();
    if (rc != 0) return rc;
    const uint32_t n = batch * out_dim;
    pd_nvf4_sk_reduce_kernel<<<(n + 255u) / 256u, 256u, 0, (cudaStream_t)stream>>>(
        (const float*)part, (const float*)bias, (float*)y, scale2, n, out_dim,
        sk);
    return pd_launch_status();
#endif
}

// ---- fused dense gate+up+swiglu->nvf4 (the encoder FFN front half) ---------
// One kernel walks both FFN matrices over a shared activation tile and emits
// silu(g)*u already quantized to nvf4 planes - the down GEMM's direct input.
// vs the unfused chain (2x pd_mxfp4_gemm_bs + pd_quantize_nvf4_swiglu) this
// stages the e4m3 activations once for two matrices, runs 2x the MMA work
// per barrier, and never materializes the f32 gate/up planes (2 writes + 2
// reads of batch x ff f32 per layer gone). Numerically BIT-IDENTICAL to the
// chain: same per-acc MMA sequence, same swiglu, same quantize. Tile 128
// ff-rows x 64 tokens (acc doubles per thread, so the token side halves);
// warp = 32 rows x 32 cols. Epilogue quantize groups run ALONG ff (the down
// GEMM's K): a 16-block's values live on the 8 lanes of one tq column, so
// the amax and the adjacent-nibble packing ride warp shuffles (the sorted-
// MoE gate_up_bs epilogue pattern, per-16 instead of per-32).
#define PD_GU_WROW 48u
#define PD_GU_YROW 80u
#define PD_GU_SMEM (2u * (2u * 128u * PD_GU_WROW + 64u * PD_GU_YROW))

__global__ void __launch_bounds__(256, 2) pd_mxfp4_gemm_bs_gu_kernel(
    const unsigned char* __restrict__ gate_data, const unsigned char* __restrict__ gate_scale,
    const unsigned char* __restrict__ up_data, const unsigned char* __restrict__ up_scale,
    const unsigned char* __restrict__ xq, const unsigned char* __restrict__ xs,
    unsigned char* __restrict__ fq, unsigned char* __restrict__ fs,
    uint32_t in_dim, uint32_t ff, uint32_t batch) {
#if PD_BS_OK
    extern __shared__ unsigned char pd_bs_sh[];
    unsigned char* wg0 = pd_bs_sh;
    unsigned char* wg1 = wg0 + 128u * PD_GU_WROW;
    unsigned char* wu0 = wg1 + 128u * PD_GU_WROW;
    unsigned char* wu1 = wu0 + 128u * PD_GU_WROW;
    unsigned char* yb0 = wu1 + 128u * PD_GU_WROW;
    unsigned char* yb1 = yb0 + 64u * PD_GU_YROW;

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, tq = lane & 3u;
    const uint32_t i0 = (warp >> 1) * 32u;
    const uint32_t c0w = (warp & 1u) * 32u;
    const uint32_t n_kb = in_dim >> 5;
    const uint32_t nk = (in_dim + 63u) / 64u;
    const uint32_t batch_pad = (batch + 63u) & ~63u;
    const uint32_t nct = batch_pad >> 6;
    const uint32_t tile = blockIdx.x;
    const uint32_t row_base = (tile / nct) * 128u;
    const uint32_t col_base = (tile % nct) * 64u;

    float accg[8][4] = {}, accu[8][4] = {};  // [sub*4 + oct][quad]

    #define PD_GU_ISSUE_W(dst, data, kt)                                              \
        {                                                                             \
            const uint32_t row = tid >> 1, seg = tid & 1u;                            \
            const bool ok = (row_base + row) < ff && (kt) * 2u + seg < n_kb;          \
            pd_cp_async16((int*)((dst) + row * PD_GU_WROW + seg * 16u),               \
                          (data) + (size_t)(row_base + row) * (in_dim >> 1) +         \
                              (kt) * 32u + seg * 16u,                                 \
                          ok);                                                        \
        }
    #define PD_GU_ISSUE_Y(dst, kt)                                                   \
        {                                                                             \
            const uint32_t col = tid >> 2, seg = tid & 3u;                            \
            const bool ok =                                                           \
                (col_base + col) < batch && ((kt) * 4u + seg) * 16u < in_dim;         \
            pd_cp_async16((int*)((dst) + col * PD_GU_YROW + 16u + seg * 16u),         \
                          xq + (size_t)(ok ? col_base + col : 0u) * in_dim +          \
                              (kt) * 64u + seg * 16u,                                 \
                          ok);                                                        \
        }

    PD_GU_ISSUE_W(wg0, gate_data, 0u)
    PD_GU_ISSUE_W(wu0, up_data, 0u)
    PD_GU_ISSUE_Y(yb0, 0u)
    asm volatile("cp.async.commit_group;");
    for (uint32_t kt = 0; kt < nk; ++kt) {
        unsigned char* twg = (kt & 1u) ? wg1 : wg0;
        unsigned char* twu = (kt & 1u) ? wu1 : wu0;
        unsigned char* ty = (kt & 1u) ? yb1 : yb0;
        if (kt + 1u < nk) {
            PD_GU_ISSUE_W((kt & 1u) ? wg0 : wg1, gate_data, kt + 1u)
            PD_GU_ISSUE_W((kt & 1u) ? wu0 : wu1, up_data, kt + 1u)
            PD_GU_ISSUE_Y((kt & 1u) ? yb0 : yb1, kt + 1u)
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        {   // ue8m0 planes: per matrix 128 rows x 2 kb; y 64 x 2
            const uint32_t row = tid >> 1, kb = tid & 1u;
            const bool wok = (row_base + row) < ff && kt * 2u + kb < n_kb;
            twg[row * PD_GU_WROW + 32u + kb] =
                wok ? gate_scale[(size_t)(row_base + row) * n_kb + kt * 2u + kb] : 0u;
            twu[row * PD_GU_WROW + 32u + kb] =
                wok ? up_scale[(size_t)(row_base + row) * n_kb + kt * 2u + kb] : 0u;
            if (tid < 128u) {
                const bool yok = (col_base + row) < batch && kt * 2u + kb < n_kb;
                ty[row * PD_GU_YROW + kb] =
                    yok ? xs[(size_t)(col_base + row) * n_kb + kt * 2u + kb] : 0u;
            }
        }
        __syncthreads();

        uint32_t amg[2][2][4], amu[2][2][4], sag[2], sau[2];
        #pragma unroll
        for (uint32_t s = 0; s < 2u; ++s) {
            const uint32_t r0 = i0 + s * 16u + g;
            const uint32_t rs = (tq & 1u) ? r0 + 8u : r0;
            const uint32_t ldoff =
                (i0 + s * 16u + ((lane >> 3) & 1u) * 8u + (lane & 7u)) * PD_GU_WROW +
                (lane >> 4) * 16u;
            uint32_t raw[4];
            pd_ldm_x4(raw, twg + ldoff);
            #pragma unroll
            for (uint32_t kb = 0; kb < 2u; ++kb) {
                amg[s][kb][0] = (raw[kb * 2u] & 0x0F0F0F0Fu) << 2;
                amg[s][kb][1] = (raw[kb * 2u + 1u] & 0x0F0F0F0Fu) << 2;
                amg[s][kb][2] = (raw[kb * 2u] & 0xF0F0F0F0u) >> 2;
                amg[s][kb][3] = (raw[kb * 2u + 1u] & 0xF0F0F0F0u) >> 2;
            }
            pd_ldm_x4(raw, twu + ldoff);
            #pragma unroll
            for (uint32_t kb = 0; kb < 2u; ++kb) {
                amu[s][kb][0] = (raw[kb * 2u] & 0x0F0F0F0Fu) << 2;
                amu[s][kb][1] = (raw[kb * 2u + 1u] & 0x0F0F0F0Fu) << 2;
                amu[s][kb][2] = (raw[kb * 2u] & 0xF0F0F0F0u) >> 2;
                amu[s][kb][3] = (raw[kb * 2u + 1u] & 0xF0F0F0F0u) >> 2;
            }
            sag[s] = *(const unsigned short*)(twg + rs * PD_GU_WROW + 32u);
            sau[s] = *(const unsigned short*)(twu + rs * PD_GU_WROW + 32u);
        }
        #pragma unroll
        for (uint32_t j = 0; j < 4u; ++j) {
            uint32_t bm[4];
            pd_ldm_x4(bm, ty + (c0w + j * 8u + (lane & 7u)) * PD_GU_YROW + 16u +
                              (lane >> 3) * 16u);
            const uint32_t sb =
                *(const unsigned short*)(ty + (c0w + j * 8u + g) * PD_GU_YROW);
            #pragma unroll
            for (uint32_t s = 0; s < 2u; ++s) {
                pd_bs_mma_kb<0>(accg[s * 4u + j], amg[s][0][0], amg[s][0][1],
                                amg[s][0][2], amg[s][0][3], bm[0], bm[1], sag[s], sb);
                pd_bs_mma_kb<1>(accg[s * 4u + j], amg[s][1][0], amg[s][1][1],
                                amg[s][1][2], amg[s][1][3], bm[2], bm[3], sag[s], sb);
                pd_bs_mma_kb<0>(accu[s * 4u + j], amu[s][0][0], amu[s][0][1],
                                amu[s][0][2], amu[s][0][3], bm[0], bm[1], sau[s], sb);
                pd_bs_mma_kb<1>(accu[s * 4u + j], amu[s][1][0], amu[s][1][1],
                                amu[s][1][2], amu[s][1][3], bm[2], bm[3], sau[s], sb);
            }
        }
        __syncthreads();
    }
    #undef PD_GU_ISSUE_W
    #undef PD_GU_ISSUE_Y

    // epilogue: silu(g)*u, then nvf4 quantize per 16 ALONG ff. For a fixed
    // token column, the 16-block's values live on this tq's 8 lanes (rows g
    // and g+8) - amax rides 3 strided shfls, and each lane assembles one
    // adjacent-nibble byte from its neighbours' e2m1 codes.
    const uint32_t tmask = 0x11111111u << tq;  // the 8 lanes of this tq column
    #pragma unroll
    for (uint32_t s = 0; s < 2u; ++s) {
        const uint32_t rb = row_base + i0 + s * 16u;  // 16-block base row
        #pragma unroll
        for (uint32_t j = 0; j < 4u; ++j) {
            #pragma unroll
            for (uint32_t qc = 0; qc < 2u; ++qc) {
                const uint32_t c = col_base + c0w + j * 8u + 2u * tq + qc;
                const float gv0 = accg[s * 4u + j][qc];
                const float gv1 = accg[s * 4u + j][qc + 2u];
                const float uv0 = accu[s * 4u + j][qc];
                const float uv1 = accu[s * 4u + j][qc + 2u];
                const float v0 = (gv0 / (1.0f + expf(-gv0))) * uv0;  // row rb + g
                const float v1 = (gv1 / (1.0f + expf(-gv1))) * uv1;  // row rb + 8 + g
                float a = fmaxf(fabsf(v0), fabsf(v1));
                a = fmaxf(a, __shfl_xor_sync(tmask, a, 4));
                a = fmaxf(a, __shfl_xor_sync(tmask, a, 8));
                a = fmaxf(a, __shfl_xor_sync(tmask, a, 16));
                float inv;
                const unsigned sbyte = pd_nvf4_scale(a, &inv);
                const uint32_t n0 = pd_e2m1_rn(v0 * inv);
                const uint32_t n1 = pd_e2m1_rn(v1 * inv);
                // lane g assembles byte g of the block: elems (2g, 2g+1) -
                // rows rb+2g,+2g+1 for g<4 (the n0 plane), rb+8+.. for g>=4.
                // Shuffle both planes and select locally: a shfl source lane
                // contributes its evaluation of the operand, so a per-lane
                // n0/n1 selector inside the shfl would pick the wrong plane.
                const uint32_t m = (g & 3u) * 2u;
                const uint32_t lo0 = __shfl_sync(0xffffffffu, n0, m * 4u + tq);
                const uint32_t hi0 = __shfl_sync(0xffffffffu, n0, (m + 1u) * 4u + tq);
                const uint32_t lo1 = __shfl_sync(0xffffffffu, n1, m * 4u + tq);
                const uint32_t hi1 = __shfl_sync(0xffffffffu, n1, (m + 1u) * 4u + tq);
                const uint32_t lo = (g < 4u) ? lo0 : lo1;
                const uint32_t hi = (g < 4u) ? hi0 : hi1;
                if (rb < ff && c < batch) {
                    fq[(size_t)c * (ff >> 1) + (rb >> 1) + g] =
                        (unsigned char)(lo | (hi << 4));
                    if (g == 0)
                        fs[(size_t)c * (ff >> 4) + (rb >> 4)] = (unsigned char)sbyte;
                }
            }
        }
    }
#else
    (void)gate_data; (void)gate_scale; (void)up_data; (void)up_scale; (void)xq;
    (void)xs; (void)fq; (void)fs; (void)in_dim; (void)ff; (void)batch;
#endif
}

PD_EXPORT
int pd_mxfp4_gemm_bs_gu(const void* gate_data, const void* gate_scale,
                        const void* up_data, const void* up_scale, const void* xq,
                        const void* xs, void* fq, void* fs, uint32_t in_dim,
                        uint32_t ff, uint32_t batch, void* stream) {
#ifndef PD_BS_HOST
    (void)gate_data; (void)gate_scale; (void)up_data; (void)up_scale; (void)xq;
    (void)xs; (void)fq; (void)fs; (void)in_dim; (void)ff; (void)batch; (void)stream;
    return cudaErrorNotSupported;
#else
    if (ff == 0 || batch == 0) return 0;
    if ((in_dim & 31u) != 0 || (ff & 15u) != 0) return cudaErrorInvalidValue;
    const uint32_t batch_pad = (batch + 63u) & ~63u;
    const uint32_t ntiles = ((ff + 127u) / 128u) * (batch_pad >> 6);
    pd_mxfp4_gemm_bs_gu_kernel<<<ntiles, 256, PD_GU_SMEM, (cudaStream_t)stream>>>(
        (const unsigned char*)gate_data, (const unsigned char*)gate_scale,
        (const unsigned char*)up_data, (const unsigned char*)up_scale,
        (const unsigned char*)xq, (const unsigned char*)xs, (unsigned char*)fq,
        (unsigned char*)fs, in_dim, ff, batch);
    return pd_launch_status();
#endif
}

// ---- v4: TMA + mbarrier-phased W4A4 GEMM (pd_nvf4_gemm_f4t) ----------------
// The prefill-band rung: v3's 2-stage cp.async ring + full-CTA barrier
// per K-step leaves the tensor pipe waiting (wait 3.63 +
// math_pipe_throttle 2.55 per issue, SM 46%, ~800 TF where a
// TMA-warp-specialized CUTLASS tile reaches ~1257 TF at gate b2048).
// This ports the kt3 shape
// (f8_lin.cuh): 384 threads = 8 consumer warps (v3's exact fragment/mma
// plan, XOR-8 de-swizzle) + 4 producer warps (single-lane TMA issue via 4
// __grid_constant__ maps over the PLAIN weight/activation layouts - no
// repack, no extra VRAM), 2-buffer mbarrier ring at KC=256 per stage,
// named-barrier WAR protection. Bit-exact vs v1/v2/v3: same kt/kh/k64
// accumulation order per acc.

#define PD_F4T_STAGE (128u * 128u + 128u * 16u)   // data box + scale box
#define PD_F4T_SMEM (2u * 2u * PD_F4T_STAGE + 32u)

// SWQ: the swiglu + nvf4-quant EPILOGUE over an INTERLEAVED
// gate|up plane (row 2j = gate_j, row 2j+1 = up_j; Nvf4Plane::gu_pairs).
// Instead of the f32 [batch, 2ff] landing (302 MB per layer on granite-30b
// at 1152 rows, 1.07 GB at 4096 -- re-read by pd_swiglu_fused_nvf4 from
// DRAM once it spills L2), each consumer lane exchanges its gate/up
// accumulators with its partner lane (g ^ 1 = lane ^ 4), computes
// silu(g)*u on the same f32 values the f32 epilogue would have stored
// (acc * scale2), and quantizes the warp's 16 pairs per batch column with
// pd_nvf4_scale / pd_e2m1_rn -- the standalone quantizer's own helpers,
// amax order-free -- writing the down GEMM's q/scale planes directly.
// Bit-identical to {f4t -> pd_swiglu_fused_nvf4} by construction
// (bench/nv4_swq_cmp.cu). SWQ=false is the original kernel.
template <bool SWQ>
__global__ void __launch_bounds__(384, 1) pd_nvf4_gemm_f4t_kernel(
    const __grid_constant__ PdTmap wdm, const __grid_constant__ PdTmap wsm,
    const __grid_constant__ PdTmap ydm, const __grid_constant__ PdTmap ysm,
    float* __restrict__ y, unsigned char* __restrict__ qo, unsigned char* __restrict__ qs,
    float scale2, const float* __restrict__ bias,
    uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
#if PD_BS_OK
    extern __shared__ __align__(1024) unsigned char pd_f4t_sh[];
    // per buffer b: wdat 16 KB | wsc 2 KB | ydat 16 KB | ysc 2 KB
    unsigned char* wdat = pd_f4t_sh;                       // 2 x 16384
    unsigned char* wsc = pd_f4t_sh + 32768u;               // 2 x 2048
    unsigned char* ydat = pd_f4t_sh + 36864u;              // 2 x 16384
    unsigned char* ysc = pd_f4t_sh + 69632u;               // 2 x 2048
    unsigned long long* mb = (unsigned long long*)(pd_f4t_sh + 73728u);

    const uint32_t tid = threadIdx.x;
    const uint32_t nk = in_dim >> 8;  // KC=256 chunks (launcher gates %256)
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t nct = batch_pad >> 7;
    const uint32_t tile = blockIdx.x;
    const uint32_t row_base = (tile / nct) * 128u;
    const uint32_t col_base = (tile % nct) * 128u;

    if (tid == 0u) {
        const uint32_t m0 = (uint32_t)__cvta_generic_to_shared(mb);
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 128;" ::"r"(m0));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 128;" ::"r"(m0 + 8u));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    asm volatile("bar.sync 0, 384;");

    if (tid >= 256u) {
        // ---------------- producer warps 8-11 ----------------
        const uint32_t ptid = tid - 256u;
        for (uint32_t kt = 0; kt < nk; ++kt) {
            const uint32_t b = kt & 1u;
            if (kt >= 2u) asm volatile("bar.sync %0, 384;" ::"r"(1u + b));
            const uint32_t m = (uint32_t)__cvta_generic_to_shared(mb) + b * 8u;
            if (ptid == 0u) {
                asm volatile(
                    "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(m),
                    "r"(2u * PD_F4T_STAGE));
                const uint32_t wd = (uint32_t)__cvta_generic_to_shared(wdat + b * 16384u);
                const uint32_t ws = (uint32_t)__cvta_generic_to_shared(wsc + b * 2048u);
                const uint32_t yd = (uint32_t)__cvta_generic_to_shared(ydat + b * 16384u);
                const uint32_t ys = (uint32_t)__cvta_generic_to_shared(ysc + b * 2048u);
                const int ck = (int)(kt * 128u), sk16 = (int)(kt * 16u);
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3}], [%4];" ::"r"(wd), "l"(&wdm), "r"(ck),
                    "r"((int)row_base), "r"(m) : "memory");
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3}], [%4];" ::"r"(ws), "l"(&wsm), "r"(sk16),
                    "r"((int)row_base), "r"(m) : "memory");
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3}], [%4];" ::"r"(yd), "l"(&ydm), "r"(ck),
                    "r"((int)col_base), "r"(m) : "memory");
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3}], [%4];" ::"r"(ys), "l"(&ysm), "r"(sk16),
                    "r"((int)col_base), "r"(m) : "memory");
            } else {
                asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(m));
            }
        }
        return;
    }

    // ---------------- consumer warps 0-7 (v3 fragment plan) ----------------
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, tq = lane & 3u;
    const uint32_t i0 = (warp >> 1) * 32u;
    const uint32_t joff = (warp & 1u) * 8u;

    float acc[16][4] = {};
    uint32_t ph0 = 0u, ph1 = 0u;

    for (uint32_t kt = 0; kt < nk; ++kt) {
        const uint32_t b = kt & 1u;
        const uint32_t m = (uint32_t)__cvta_generic_to_shared(mb) + b * 8u;
        const uint32_t ph = (b == 0u) ? ph0 : ph1;
        asm volatile(
            "{\n\t.reg .pred P;\n"
            "PD_F4T_WAIT_%=:\n\t"
            "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
            "@!P bra PD_F4T_WAIT_%=;\n\t}" ::"r"(m), "r"(ph) : "memory");
        if (b == 0u) ph0 ^= 1u; else ph1 ^= 1u;

        const unsigned char* tw = wdat + b * 16384u;
        const unsigned char* tws = wsc + b * 2048u;
        const unsigned char* ty = ydat + b * 16384u;
        const unsigned char* tys = ysc + b * 2048u;

        #pragma unroll
        for (uint32_t kh = 0; kh < 2u; ++kh) {
            const uint32_t sb8 = kh * 8u;
            uint32_t am[2][2][4], sa[2][2];
            #pragma unroll
            for (uint32_t n = 0; n < 2u; ++n) {
                const uint32_t r0 = i0 + n * 16u + g;
                const uint32_t rs = (tq & 1u) ? r0 + 8u : r0;
                const uint32_t rr = i0 + n * 16u + ((lane >> 3) & 1u) * 8u + (lane & 7u);
                #pragma unroll
                for (uint32_t k64 = 0; k64 < 2u; ++k64) {
                    const uint32_t c = kh * 4u + k64 * 2u + (lane >> 4);
                    pd_ldm_x4(am[n][k64], tw + rr * 128u + ((c ^ (rr & 7u)) * 16u));
                    sa[n][k64] = *(const uint32_t*)(tws + rs * 16u + sb8 + k64 * 4u);
                }
            }
            #pragma unroll
            for (uint32_t j0 = 0; j0 < 128u; j0 += 16u) {
                const uint32_t col = j0 + joff + (lane & 7u);
                const uint32_t cy = kh * 4u + (lane >> 3);
                uint32_t bm[4];
                pd_ldm_x4(bm, ty + col * 128u + ((cy ^ (col & 7u)) * 16u));
                const unsigned char* ysr = tys + (j0 + joff + g) * 16u + sb8;
                #pragma unroll
                for (uint32_t k64 = 0; k64 < 2u; ++k64) {
                    const uint32_t sbv = *(const uint32_t*)(ysr + k64 * 4u);
                    #pragma unroll
                    for (uint32_t n = 0; n < 2u; ++n)
                        pd_nv4_mma(acc[(j0 >> 3) + n], am[n][k64][0], am[n][k64][1],
                                   am[n][k64][2], am[n][k64][3], bm[k64 * 2u],
                                   bm[k64 * 2u + 1u], sa[n][k64], sbv);
                }
            }
        }
        asm volatile("bar.arrive %0, 384;" ::"r"(1u + b));
    }

    if constexpr (!SWQ) {
    // v3's exact epilogue: scale2, conditional bias, f32 y
    #pragma unroll
    for (uint32_t j0 = 0; j0 < 128u; j0 += 16u) {
        const uint32_t c0 = col_base + j0 + joff + 2u * tq;
        #pragma unroll
        for (uint32_t n = 0; n < 2u; ++n) {
            const uint32_t r0 = row_base + i0 + n * 16u + g;
            const uint32_t r8 = r0 + 8u;
            float v00 = acc[(j0 >> 3) + n][0], v01 = acc[(j0 >> 3) + n][1];
            float v10 = acc[(j0 >> 3) + n][2], v11 = acc[(j0 >> 3) + n][3];
            v00 *= scale2; v01 *= scale2; v10 *= scale2; v11 *= scale2;
            if (bias) {
                if (r0 < out_dim) { const float bv = bias[r0]; v00 += bv; v01 += bv; }
                if (r8 < out_dim) { const float bv = bias[r8]; v10 += bv; v11 += bv; }
            }
            if (r0 < out_dim) {
                if (c0 < batch) y[(size_t)c0 * out_dim + r0] = v00;
                if (c0 + 1u < batch) y[(size_t)(c0 + 1u) * out_dim + r0] = v01;
            }
            if (r8 < out_dim) {
                if (c0 < batch) y[(size_t)c0 * out_dim + r8] = v10;
                if (c0 + 1u < batch) y[(size_t)(c0 + 1u) * out_dim + r8] = v11;
            }
        }
    }
    } else {
    // SWQ epilogue. Lane rows r0 = row_base + i0 + n*16 + g and r8 = r0 + 8
    // share parity with g: even-g lanes hold GATE rows (2p), odd-g lanes the
    // matching up rows (2p+1) of the same pairs -- partner = lane ^ 4. The
    // warp's 32 rows are one 16-pair block per batch column; pair k of the
    // block = n*8 + r*4 + g/2 (r: 0 = r0, 1 = r8), held on even-g lanes.
    const uint32_t ff = out_dim >> 1;
    const bool is_g = (g & 1u) == 0u;
    const uint32_t pb = (row_base + i0) >> 1;   // block's first pair (multiple of 16)
    #pragma unroll
    for (uint32_t j0 = 0; j0 < 128u; j0 += 16u) {
        const uint32_t c0 = col_base + j0 + joff + 2u * tq;
        float v[2][2][2];   // [n][r][cc] = silu(gate) * up (meaningful on even-g lanes)
        #pragma unroll
        for (uint32_t n = 0; n < 2u; ++n) {
            #pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) {
                const float a = acc[(j0 >> 3) + n][e] * scale2;
                const float b = __shfl_xor_sync(0xffffffffu, a, 4);
                const float gv = is_g ? a : b;
                const float uv = is_g ? b : a;
                v[n][e >> 1][e & 1u] = (gv / (1.0f + expf(-gv))) * uv;
            }
        }
        #pragma unroll
        for (uint32_t cc = 0; cc < 2u; ++cc) {
            float amax = fmaxf(fmaxf(fabsf(v[0][0][cc]), fabsf(v[0][1][cc])),
                               fmaxf(fabsf(v[1][0][cc]), fabsf(v[1][1][cc])));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 8));    // g ^ 2
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 16));   // g ^ 4
            float inv;
            const unsigned sbyte = pd_nvf4_scale(amax, &inv);
            uint32_t code[2][2], mate[2][2];
            #pragma unroll
            for (uint32_t n = 0; n < 2u; ++n)
                #pragma unroll
                for (uint32_t r = 0; r < 2u; ++r) {
                    code[n][r] = pd_e2m1_rn(v[n][r][cc] * inv);
                    mate[n][r] = __shfl_xor_sync(0xffffffffu, code[n][r], 8);   // pair k+1 (g+2)
                }
            const uint32_t c = c0 + cc;
            // the block's 8 bytes for column c: byte n*4 + r*2 + (g>>2) = own | mate<<4
            // on the assembler lanes g = 0 (even bytes) and g = 4 (odd bytes).
            // Pack each lane's four bytes, fetch g = 4's from lane ^ 16, and let
            // g = 0 store the whole block as one u64 (pb % 16 == 0 and ff % 16
            // == 0 keep it 8-byte aligned) -- one sector per column instead of
            // eight scattered single-byte stores.
            const uint32_t mine = (code[0][0] | (mate[0][0] << 4))
                                | ((code[0][1] | (mate[0][1] << 4)) << 8)
                                | ((code[1][0] | (mate[1][0] << 4)) << 16)
                                | ((code[1][1] | (mate[1][1] << 4)) << 24);
            const uint32_t odd = __shfl_xor_sync(0xffffffffu, mine, 16);   // g ^ 4
            if (is_g && (g & 6u) == 0u && c < batch && pb < ff) {           // g == 0
                const uint64_t lo = (uint64_t)((mine & 0xFFu) | ((odd & 0xFFu) << 8)
                                             | ((mine & 0xFF00u) << 8) | ((odd & 0xFF00u) << 16));
                const uint64_t hi = (uint64_t)(((mine >> 16) & 0xFFu) | (((odd >> 16) & 0xFFu) << 8)
                                             | (((mine >> 16) & 0xFF00u) << 8) | (((odd >> 16) & 0xFF00u) << 16));
                *(uint64_t*)(qo + (size_t)c * (ff >> 1) + (pb >> 1)) = lo | (hi << 32);
                qs[(size_t)c * (ff >> 4) + (pb >> 4)] = (unsigned char)sbyte;
            }
        }
    }
    }
#else
    (void)wdm; (void)wsm; (void)ydm; (void)ysm; (void)y; (void)qo; (void)qs; (void)scale2;
    (void)bias; (void)in_dim; (void)out_dim; (void)batch;
#endif
}

// 16 B-inner scale-plane map: rows x (in/16) e4m3 bytes, 128x16 boxes, no
// swizzle (the box inner is one swizzle atom).
//
// Guarded like its sibling pd_tmap_2d in gemm/dense_fp4_w8.cuh, and for the
// same reason: pd_tmap_encode() only EXISTS under PD_BS_HOST. Without this the
// pack does not compile for an arch list that omits both 100 and 120 - a real
// hole found, since every list we habitually build carries one of
// them. Its only caller is already inside the `#else` of `#ifndef PD_BS_HOST`,
// so nothing loses a definition it can reach.
#ifdef PD_BS_HOST
static bool pd_tmap_2d_s16(CUtensorMap* map, const void* base, uint64_t inner,
                           uint64_t rows) {
    pd_tmap_encode_fn enc = pd_tmap_encode();
    if (!enc || ((uintptr_t)base & 15u) || (inner & 15u)) return false;
    const cuuint64_t gdim[2] = {inner, rows};
    const cuuint64_t gstride[1] = {inner};
    const cuuint32_t box[2] = {16u, 128u};
    const cuuint32_t estride[2] = {1u, 1u};
    return enc(map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2u, (void*)base, gdim, gstride,
               box, estride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
               CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}
#endif

// TMA-map caches. A cached map encodes (base ptr, gdim, box) and TMA
// zero-fills any coordinate past gdim -- so the key must pin every dim that
// shapes the map, or a stale entry silently reads the wrong region. This
// was the bug behind the granite-4.2 NVFP4 batch garbage: the
// f4t KERNEL is bit-exact vs f4c at every shape (nv4_f4t_1shot.cu, one shape
// per process) -- the corruption was entirely a mis-keyed map.
//   - activation: one scratch (`sc.xq`) is re-staged at different widths in
//     one tick (granite q/k/v/o/gate/up at n_embd, down at n_ff) AND at
//     different batch, so the key is (ptr, rows=batch, k=in_dim). Missing
//     in_dim handed `down` the narrow map -> zero-filled K tail -> all-zero
//     activation -> fluent garbage above the f4c->f4t election (batch>=128).
//   - weight: pointers are stable in serving, but a freed-then-realloced
//     plane (bench, or a model reload) can alias an old address; key on
//     (ptr, k=in_dim, o=out_dim) so a same-address different-shape plane
//     rebuilds. (This is what made the nv4_f4t oracle/sweep report false
//     "broken" cells when it reused freed buffers across shapes.)
struct PdF4tWEnt { const void* p; uint32_t k, o; CUtensorMap dm, sm; };
struct PdF4tYEnt { const void* p; uint32_t rows, k; CUtensorMap dm, sm; };

#ifdef PD_BS_HOST
// tensor-map caches shared by the f4t launchers (weight key (ptr, in, out),
// activation key (ptr, rows, in) -- see the corruption note above)
static inline bool pd_f4t_maps(const void* data, const void* scale, const void* xq, const void* xs,
                               uint32_t in_dim, uint32_t out_dim, uint32_t batch,
                               CUtensorMap** wdm, CUtensorMap** wsm, CUtensorMap** ydm, CUtensorMap** ysm) {
    static PdF4tWEnt wc[256]; static uint32_t wn = 0;
    static PdF4tYEnt yc[64]; static uint32_t yn = 0;
    *wdm = nullptr; *wsm = nullptr; *ydm = nullptr; *ysm = nullptr;
    for (uint32_t i = 0; i < wn; ++i)
        if (wc[i].p == data && wc[i].k == in_dim && wc[i].o == out_dim) { *wdm = &wc[i].dm; *wsm = &wc[i].sm; break; }
    if (!*wdm) {
        PdF4tWEnt& e = wc[wn % 256u];
        if (!pd_tmap_2d(&e.dm, data, in_dim >> 1, out_dim) ||
            !pd_tmap_2d_s16(&e.sm, scale, in_dim >> 4, out_dim))
            return false;
        e.p = data; e.k = in_dim; e.o = out_dim; *wdm = &e.dm; *wsm = &e.sm; wn = wn < 256u ? wn + 1u : wn;
    }
    for (uint32_t i = 0; i < yn; ++i)
        if (yc[i].p == xq && yc[i].rows == batch && yc[i].k == in_dim) { *ydm = &yc[i].dm; *ysm = &yc[i].sm; break; }
    if (!*ydm) {
        PdF4tYEnt& e = yc[yn % 64u];
        if (!pd_tmap_2d(&e.dm, xq, in_dim >> 1, batch) ||
            !pd_tmap_2d_s16(&e.sm, xs, in_dim >> 4, batch))
            return false;
        e.p = xq; e.rows = batch; e.k = in_dim; *ydm = &e.dm; *ysm = &e.sm;
        yn = yn < 64u ? yn + 1u : yn;
    }
    return true;
}
#endif

PD_EXPORT
int pd_nvf4_gemm_f4t(const void* data, const void* scale, const void* bias,
                     const void* xq, const void* xs, void* y, float scale2,
                     uint32_t in_dim, uint32_t out_dim, uint32_t batch,
                     void* stream) {
#ifndef PD_BS_HOST
    (void)data; (void)scale; (void)bias; (void)xq; (void)xs; (void)y;
    (void)scale2; (void)in_dim; (void)out_dim; (void)batch; (void)stream;
    return cudaErrorNotSupported;
#else
    if (out_dim == 0 || batch == 0) return 0;
    if ((in_dim & 255u) != 0) return cudaErrorInvalidValue;
    CUtensorMap *wdm, *wsm, *ydm, *ysm;
    if (!pd_f4t_maps(data, scale, xq, xs, in_dim, out_dim, batch, &wdm, &wsm, &ydm, &ysm))
        return cudaErrorNotSupported;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t ntiles = ((out_dim + 127u) / 128u) * (batch_pad >> 7);
    static int smem_set = 0;
    if (!smem_set) {
        cudaFuncSetAttribute(pd_nvf4_gemm_f4t_kernel<false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)PD_F4T_SMEM);
        smem_set = 1;
    }
    pd_nvf4_gemm_f4t_kernel<false><<<ntiles, 384, PD_F4T_SMEM, (cudaStream_t)stream>>>(
        *wdm, *wsm, *ydm, *ysm, (float*)y, nullptr, nullptr, scale2, (const float*)bias, in_dim,
        out_dim, batch);
    return pd_launch_status();
#endif
}

// f4t with the swiglu + nvf4-quant epilogue over an INTERLEAVED gate|up plane
// (out_dim = 2*ff, rows 2j/2j+1 = gate_j/up_j): writes the down GEMM's
// activation planes q [batch, ff/2] and qs [batch, ff/16] directly -- no f32
// landing. Same geometry gates as f4t; out_dim % 256 == 0 keeps every warp's
// 32 rows one whole 16-pair block. Bit-identical to f4t + pd_swiglu_fused_nvf4
// on the plain plane (bench/nv4_swq_cmp.cu).
PD_EXPORT
int pd_nvf4_gemm_f4t_swq(const void* data, const void* scale, const void* xq,
                         const void* xs, void* q, void* qs, float scale2,
                         uint32_t in_dim, uint32_t out_dim, uint32_t batch,
                         void* stream) {
#ifndef PD_BS_HOST
    (void)data; (void)scale; (void)xq; (void)xs; (void)q; (void)qs;
    (void)scale2; (void)in_dim; (void)out_dim; (void)batch; (void)stream;
    return cudaErrorNotSupported;
#else
    if (out_dim == 0 || batch == 0) return 0;
    if ((in_dim & 255u) != 0 || (out_dim & 255u) != 0) return cudaErrorInvalidValue;
    CUtensorMap *wdm, *wsm, *ydm, *ysm;
    if (!pd_f4t_maps(data, scale, xq, xs, in_dim, out_dim, batch, &wdm, &wsm, &ydm, &ysm))
        return cudaErrorNotSupported;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t ntiles = ((out_dim + 127u) / 128u) * (batch_pad >> 7);
    static int smem_set = 0;
    if (!smem_set) {
        cudaFuncSetAttribute(pd_nvf4_gemm_f4t_kernel<true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)PD_F4T_SMEM);
        smem_set = 1;
    }
    pd_nvf4_gemm_f4t_kernel<true><<<ntiles, 384, PD_F4T_SMEM, (cudaStream_t)stream>>>(
        *wdm, *wsm, *ydm, *ysm, nullptr, (unsigned char*)q, (unsigned char*)qs, scale2,
        nullptr, in_dim, out_dim, batch);
    return pd_launch_status();
#endif
}

// ---- SmoothQuant-folded nvf4 (per-channel scale migration) -----------------
// The post-norm hidden's outlier CHANNELS are what break fp4 activations
// (measured: raw and H128-rotated nvf4 both fail the recall gate there).
// SmoothQuant (Xiao et al. 2022 - technique studied, implementation
// original) migrates the difficulty: activations divide by a per-channel
// s[c] inside the quantizer (outlier channels shrink toward the rest) and
// the consuming weights multiply by s[c] at requant - W'x' = Wx exactly in
// reals, with the per-16 e4m3 weight scales absorbing the migrated range.
// s comes from calibration statistics: s[c] = sqrt(act_amax[c] /
// w_colmax[c]) (the alpha = 0.5 balance). Applied only on the fp4 planes;
// the Q8_0 path never sees the transform.

// Per-column running abs-max over a row-major [rows, n] f32 plane,
// accumulated across CALLS into `out` (caller zeroes once): 8 threads per
// column walk row-strided but column-adjacent (coalesced), then one
// positive-float atomicMax each (float bits compare monotonically >= 0).
__global__ void pd_col_absmax_kernel(const float* __restrict__ x,
                                     float* __restrict__ out, uint32_t rows,
                                     uint32_t n) {
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t c = t % n;
    const uint32_t r0 = t / n;
    const uint32_t rstride = (gridDim.x * blockDim.x) / n;
    float m = 0.0f;
    for (uint32_t r = r0; r < rows; r += rstride)
        m = fmaxf(m, fabsf(x[(size_t)r * n + c]));
    atomicMax((int*)&out[c], __float_as_int(m));
}

PD_EXPORT
int pd_col_absmax(const void* x, void* out, uint32_t rows, uint32_t n, void* stream) {
    if (rows == 0 || n == 0) return 0;
    if ((n & 31u) != 0) return cudaErrorInvalidValue;
    const uint32_t total = n * 8u;  // n % 32 == 0 -> total % 256 == 0
    pd_col_absmax_kernel<<<total / 256u, 256u, 0, (cudaStream_t)stream>>>(
        (const float*)x, (float*)out, rows, n);
    return pd_launch_status();
}

// Per-column abs-max of a repacked Q8_0 weight ([out_dim, in_dim] int8 rows
// + f16 per-32 scales) - the weight half of the SmoothQuant balance.
__global__ void pd_q8_0_col_absmax_kernel(const int8_t* __restrict__ data,
                                          const __half* __restrict__ scale,
                                          float* __restrict__ out, uint32_t in_dim,
                                          uint32_t out_dim) {
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t c = t % in_dim;
    const uint32_t r0 = t / in_dim;
    const uint32_t rstride = (gridDim.x * blockDim.x) / in_dim;
    const uint32_t n_blocks = in_dim >> 5;
    float m = 0.0f;
    for (uint32_t r = r0; r < out_dim; r += rstride) {
        const float d = fabsf(__half2float(scale[(size_t)r * n_blocks + (c >> 5)]));
        m = fmaxf(m, (float)abs((int)data[(size_t)r * in_dim + c]) * d);
    }
    atomicMax((int*)&out[c], __float_as_int(m));
}

PD_EXPORT
int pd_q8_0_col_absmax(const void* data, const void* scale, void* out,
                       uint32_t in_dim, uint32_t out_dim, void* stream) {
    if (in_dim == 0 || out_dim == 0) return 0;
    if ((in_dim & 31u) != 0) return cudaErrorInvalidValue;
    const uint32_t total = in_dim * 8u;
    pd_q8_0_col_absmax_kernel<<<total / 256u, 256u, 0, (cudaStream_t)stream>>>(
        (const int8_t*)data, (const __half*)scale, (float*)out, in_dim, out_dim);
    return pd_launch_status();
}

// f32 -> nvf4 with the SmoothQuant activation fold: v[c] * sinv[c] before
// the per-16 quantize. Same layout/numerics as pd_quantize_nvf4 otherwise.
__global__ void pd_quantize_nvf4_smooth_kernel(const float* __restrict__ x,
                                               const float* __restrict__ sinv,
                                               unsigned char* __restrict__ q,
                                               unsigned char* __restrict__ scale,
                                               uint32_t n, uint32_t in_dim) {
    const uint32_t i = (blockIdx.x * 256u + threadIdx.x) * 8u;
    if (i >= n) return;
    const uint32_t c0 = i % in_dim;  // 8 elems stay inside one row
    float4 v0 = *(const float4*)(x + i);
    float4 v1 = *(const float4*)(x + i + 4u);
    v0.x *= sinv[c0];
    v0.y *= sinv[c0 + 1u];
    v0.z *= sinv[c0 + 2u];
    v0.w *= sinv[c0 + 3u];
    v1.x *= sinv[c0 + 4u];
    v1.y *= sinv[c0 + 5u];
    v1.z *= sinv[c0 + 6u];
    v1.w *= sinv[c0 + 7u];
    pd_nvf4_quant8(v0, v1, threadIdx.x & 31u, q, scale, i);
}

PD_EXPORT
int pd_quantize_nvf4_smooth(const void* x, const void* sinv, void* q, void* scale,
                            uint32_t n, uint32_t in_dim, void* stream) {
    if (n == 0) return 0;
    if ((n & 31u) != 0 || (in_dim & 7u) != 0) return cudaErrorInvalidValue;
    pd_quantize_nvf4_smooth_kernel<<<(n / 8u + 255u) / 256u, 256u, 0,
                                     (cudaStream_t)stream>>>(
        (const float*)x, (const float*)sinv, (unsigned char*)q,
        (unsigned char*)scale, n, in_dim);
    return pd_launch_status();
}

// Q8_0 -> nvf4 weight requant with the SmoothQuant weight fold: w[r][c] *
// svec[c] before the per-16 quantize (the migrated activation range lands
// in the e4m3 weight scales). Same layout as pd_q8_0_to_nvf4 otherwise.
__global__ void pd_q8_0_to_nvf4_smooth_kernel(const int8_t* __restrict__ q8,
                                              const __half* __restrict__ s8,
                                              const float* __restrict__ svec,
                                              unsigned char* __restrict__ data,
                                              unsigned char* __restrict__ scale,
                                              uint64_t n_blocks, uint32_t in_dim) {
    uint64_t blk = blockIdx.x;
    uint32_t d = threadIdx.x;
    if (blk >= n_blocks) return;
    const uint64_t i = blk * 32u + d;
    float v = (float)q8[i] * __half2float(s8[blk]) * svec[(uint32_t)(i % in_dim)];
    float a = fabsf(v);
    const uint32_t gm = 0xFFFFu << (d & 16u);
    for (uint32_t off = 8; off > 0; off >>= 1)
        a = fmaxf(a, __shfl_xor_sync(gm, a, off));
    float inv;
    unsigned sb = pd_nvf4_scale(a, &inv);
    unsigned nib = pd_e2m1_rn(v * inv);
    unsigned lo = __shfl_sync(0xffffffffu, nib, 2u * d);
    unsigned hi = __shfl_sync(0xffffffffu, nib, 2u * d + 1u);
    if (d < 16u) data[blk * 16u + d] = (unsigned char)(lo | (hi << 4));
    if ((d & 15u) == 0) scale[blk * 2u + (d >> 4)] = (unsigned char)sb;
}

PD_EXPORT
int pd_q8_0_to_nvf4_smooth(const void* q8_data, const void* q8_scale,
                           const void* svec, void* mx_data, void* mx_scale,
                           uint64_t n_blocks, uint32_t in_dim, void* stream) {
#ifndef PD_BS_HOST
    (void)q8_data; (void)q8_scale; (void)svec; (void)mx_data; (void)mx_scale;
    (void)n_blocks; (void)in_dim; (void)stream;
    return cudaErrorNotSupported;
#else
    if (n_blocks == 0) return 0;
    if ((in_dim & 31u) != 0) return cudaErrorInvalidValue;
    pd_q8_0_to_nvf4_smooth_kernel<<<(uint32_t)n_blocks, 32, 0, (cudaStream_t)stream>>>(
        (const int8_t*)q8_data, (const __half*)q8_scale, (const float*)svec,
        (unsigned char*)mx_data, (unsigned char*)mx_scale, n_blocks, in_dim);
    return pd_launch_status();
#endif
}

// SwiGLU + SmoothQuant fold + nvf4 quantize in one pass (the smoothed down
// site's input): v = silu(g)*u * sinv[c].
__global__ void pd_quantize_nvf4_swiglu_smooth_kernel(
    const float* __restrict__ gate, const float* __restrict__ up,
    const float* __restrict__ sinv, unsigned char* __restrict__ q,
    unsigned char* __restrict__ scale, uint32_t n, uint32_t in_dim) {
    const uint32_t i = (blockIdx.x * 256u + threadIdx.x) * 8u;
    if (i >= n) return;
    const uint32_t c0 = i % in_dim;
    float4 v[2];
    #pragma unroll
    for (uint32_t h = 0; h < 2u; ++h) {
        const float4 g = *(const float4*)(gate + i + h * 4u);
        const float4 u = *(const float4*)(up + i + h * 4u);
        v[h].x = (g.x / (1.0f + expf(-g.x))) * u.x * sinv[c0 + h * 4u];
        v[h].y = (g.y / (1.0f + expf(-g.y))) * u.y * sinv[c0 + h * 4u + 1u];
        v[h].z = (g.z / (1.0f + expf(-g.z))) * u.z * sinv[c0 + h * 4u + 2u];
        v[h].w = (g.w / (1.0f + expf(-g.w))) * u.w * sinv[c0 + h * 4u + 3u];
    }
    pd_nvf4_quant8(v[0], v[1], threadIdx.x & 31u, q, scale, i);
}

PD_EXPORT
int pd_quantize_nvf4_swiglu_smooth(const void* gate, const void* up, const void* sinv,
                                   void* q, void* scale, uint32_t n, uint32_t in_dim,
                                   void* stream) {
    if (n == 0) return 0;
    if ((n & 31u) != 0 || (in_dim & 7u) != 0) return cudaErrorInvalidValue;
    pd_quantize_nvf4_swiglu_smooth_kernel<<<(n / 8u + 255u) / 256u, 256u, 0,
                                            (cudaStream_t)stream>>>(
        (const float*)gate, (const float*)up, (const float*)sinv, (unsigned char*)q,
        (unsigned char*)scale, n, in_dim);
    return pd_launch_status();
}

// SmoothQuant folds for the mxf8f6f4 (fp4-weight x e4m3-activation) class.
// Here the migration usually runs the other way (low alpha, s ~ 1/w_colmax):
// the fp4 weights' per-32 ue8m0 blocks are the coarse side - normalizing
// weight columns lets every column use the block scale, and the e4m3
// activations (finer grid, own per-32 scales) absorb the migrated range.

// f32 -> e4m3 + ue8m0 with the activation fold (v[c] * sinv[c]).
__global__ void pd_quantize_e4m3_smooth_kernel(const float* __restrict__ x,
                                               const float* __restrict__ sinv,
                                               unsigned char* __restrict__ q,
                                               unsigned char* __restrict__ scale,
                                               uint32_t n, uint32_t in_dim) {
    const uint32_t i = (blockIdx.x * 256u + threadIdx.x) * 4u;
    if (i >= n) return;
    const uint32_t c0 = i % in_dim;
    float4 v = *(const float4*)(x + i);
    v.x *= sinv[c0];
    v.y *= sinv[c0 + 1u];
    v.z *= sinv[c0 + 2u];
    v.w *= sinv[c0 + 3u];
    pd_e4m3_quant4(v, threadIdx.x & 7u, q, scale, i);
}

PD_EXPORT
int pd_quantize_e4m3_smooth(const void* x, const void* sinv, void* q, void* scale,
                            uint32_t n, uint32_t in_dim, void* stream) {
    if (n == 0) return 0;
    if ((n & 31u) != 0 || (in_dim & 3u) != 0) return cudaErrorInvalidValue;
    pd_quantize_e4m3_smooth_kernel<<<(n / 4u + 255u) / 256u, 256u, 0,
                                     (cudaStream_t)stream>>>(
        (const float*)x, (const float*)sinv, (unsigned char*)q,
        (unsigned char*)scale, n, in_dim);
    return pd_launch_status();
}

// SwiGLU + fold + e4m3 quantize in one pass (the smoothed-F8 down input).
__global__ void pd_quantize_e4m3_swiglu_smooth_kernel(
    const float* __restrict__ gate, const float* __restrict__ up,
    const float* __restrict__ sinv, unsigned char* __restrict__ q,
    unsigned char* __restrict__ scale, uint32_t n, uint32_t in_dim) {
    const uint32_t i = (blockIdx.x * 256u + threadIdx.x) * 4u;
    if (i >= n) return;
    const uint32_t c0 = i % in_dim;
    const float4 g = *(const float4*)(gate + i);
    const float4 u = *(const float4*)(up + i);
    float4 v;
    v.x = (g.x / (1.0f + expf(-g.x))) * u.x * sinv[c0];
    v.y = (g.y / (1.0f + expf(-g.y))) * u.y * sinv[c0 + 1u];
    v.z = (g.z / (1.0f + expf(-g.z))) * u.z * sinv[c0 + 2u];
    v.w = (g.w / (1.0f + expf(-g.w))) * u.w * sinv[c0 + 3u];
    pd_e4m3_quant4(v, threadIdx.x & 7u, q, scale, i);
}

PD_EXPORT
int pd_quantize_e4m3_swiglu_smooth(const void* gate, const void* up, const void* sinv,
                                   void* q, void* scale, uint32_t n, uint32_t in_dim,
                                   void* stream) {
    if (n == 0) return 0;
    if ((n & 31u) != 0 || (in_dim & 3u) != 0) return cudaErrorInvalidValue;
    pd_quantize_e4m3_swiglu_smooth_kernel<<<(n / 4u + 255u) / 256u, 256u, 0,
                                            (cudaStream_t)stream>>>(
        (const float*)gate, (const float*)up, (const float*)sinv, (unsigned char*)q,
        (unsigned char*)scale, n, in_dim);
    return pd_launch_status();
}

// Q8_0 -> split-order mxfp4 (the mxf8f6f4 A format) with the weight fold:
// w[r][c] * svec[c] before the per-32 ue8m0 quantize.
__global__ void pd_q8_0_to_mxfp4_smooth_kernel(const int8_t* __restrict__ q8,
                                               const __half* __restrict__ s8,
                                               const float* __restrict__ svec,
                                               unsigned char* __restrict__ data,
                                               unsigned char* __restrict__ scale,
                                               uint64_t n_blocks, uint32_t in_dim) {
    uint64_t blk = blockIdx.x;
    uint32_t d = threadIdx.x;
    if (blk >= n_blocks) return;
    const uint64_t i = blk * 32u + d;
    float v = (float)q8[i] * __half2float(s8[blk]) * svec[(uint32_t)(i % in_dim)];
    float a = fabsf(v);
    for (uint32_t off = 16; off > 0; off >>= 1)
        a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, off));
    int e = 0;
    if (a > 0.0f) {
        int ex;
        float m = frexpf(a, &ex);
        e = ex - 3 + (m > 0.75f ? 1 : 0);
    }
    unsigned nib = pd_e2m1_rn(v * ldexpf(1.0f, -e));
    unsigned hi = __shfl_sync(0xffffffffu, nib, d + 16u);
    if (d < 16u) data[blk * 16u + d] = (unsigned char)(nib | (hi << 4));
    if (d == 0) scale[blk] = (unsigned char)(e + 127);
}

PD_EXPORT
int pd_q8_0_to_mxfp4_smooth(const void* q8_data, const void* q8_scale,
                            const void* svec, void* mx_data, void* mx_scale,
                            uint64_t n_blocks, uint32_t in_dim, void* stream) {
#ifndef PD_BS_HOST
    (void)q8_data; (void)q8_scale; (void)svec; (void)mx_data; (void)mx_scale;
    (void)n_blocks; (void)in_dim; (void)stream;
    return cudaErrorNotSupported;
#else
    if (n_blocks == 0) return 0;
    if ((in_dim & 31u) != 0) return cudaErrorInvalidValue;
    pd_q8_0_to_mxfp4_smooth_kernel<<<(uint32_t)n_blocks, 32, 0, (cudaStream_t)stream>>>(
        (const int8_t*)q8_data, (const __half*)q8_scale, (const float*)svec,
        (unsigned char*)mx_data, (unsigned char*)mx_scale, n_blocks, in_dim);
    return pd_launch_status();
#endif
}

// ---- rotated nvf4: fused Hadamard-128 + nvf4 quantize (QuaRot class) -------
// The post-norm hidden carries outlier channels that fail 4-bit activation
// quantization under both fp4 scale flavors (recall@1 gate). The QuaRot fix
// (Ashkboos et al. 2024 - technique studied, implementation original):
// multiply the hidden by a block-diagonal orthonormal Hadamard H (128-wide
// blocks here) before quantizing, and fold H into the consuming weights'
// input dim at load. W'x' = (WH^T)(Hx) = Wx exactly in reals; the rotation
// smears each outlier across 128 channels so the per-16 nvf4 blocks see
// near-Gaussian values. Runtime cost is ~zero: the transform fuses into the
// activation quantizer (7 butterfly stages, 5 of them warp shuffles).

// In-register FHT over one 128-group: lane l holds elements 4l..4l+3 of the
// group as a float4. Stages: distance 1,2 inside the float4, then lane-xor
// 1,2,4,8,16 (element distance 4..64). Normalized by 1/sqrt(128) so H is
// orthonormal (applied on both the weight and activation sides).
__device__ __forceinline__ float4 pd_fht128(float4 v) {
    float t;
    t = v.x; v.x = t + v.y; v.y = t - v.y;
    t = v.z; v.z = t + v.w; v.w = t - v.w;
    t = v.x; v.x = t + v.z; v.z = t - v.z;
    t = v.y; v.y = t + v.w; v.w = t - v.w;
    const uint32_t lane = threadIdx.x & 31u;
    #pragma unroll
    for (uint32_t d = 1; d <= 16u; d <<= 1) {
        const bool hi = (lane & d) != 0u;
        float4 o;
        o.x = __shfl_xor_sync(0xffffffffu, v.x, d);
        o.y = __shfl_xor_sync(0xffffffffu, v.y, d);
        o.z = __shfl_xor_sync(0xffffffffu, v.z, d);
        o.w = __shfl_xor_sync(0xffffffffu, v.w, d);
        v.x = hi ? o.x - v.x : v.x + o.x;
        v.y = hi ? o.y - v.y : v.y + o.y;
        v.z = hi ? o.z - v.z : v.z + o.z;
        v.w = hi ? o.w - v.w : v.w + o.w;
    }
    const float r = 0.08838834764831845f;  // 1/sqrt(128)
    v.x *= r; v.y *= r; v.z *= r; v.w *= r;
    return v;
}

// Shared tail: per-16 nvf4 quantize of a rotated float4 (lane quad = one
// 16-block), u16 nibble store.
__device__ __forceinline__ void pd_nvf4_rot_store(float4 v,
                                                  unsigned char* __restrict__ q,
                                                  unsigned char* __restrict__ scale,
                                                  uint32_t i) {
    const uint32_t lane = threadIdx.x & 31u;
    float a = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
    const uint32_t gm = 0xfu << (lane & 28u);  // 4-lane group = one 16-block
    a = fmaxf(a, __shfl_xor_sync(gm, a, 2));
    a = fmaxf(a, __shfl_xor_sync(gm, a, 1));
    float inv;
    unsigned sb = pd_nvf4_scale(a, &inv);
    const uint32_t p = pd_e2m1_rn(v.x * inv) | (pd_e2m1_rn(v.y * inv) << 4)
                     | (pd_e2m1_rn(v.z * inv) << 8) | (pd_e2m1_rn(v.w * inv) << 12);
    *(unsigned short*)(q + (i >> 1)) = (unsigned short)p;
    if ((lane & 3u) == 0) scale[i >> 4] = (unsigned char)sb;
}

// f32 -> H128-rotated nvf4 (the rotated hidden). One warp per 128-group;
// n % 128 == 0 (all hidden/attention widths are).
__global__ void pd_quantize_nvf4_rot_kernel(const float* __restrict__ x,
                                            unsigned char* __restrict__ q,
                                            unsigned char* __restrict__ scale,
                                            uint32_t n) {
    const uint32_t i =
        ((blockIdx.x * 256u + threadIdx.x) >> 5) * 128u + (threadIdx.x & 31u) * 4u;
    if (i >= n) return;  // whole warps exit together
    pd_nvf4_rot_store(pd_fht128(*(const float4*)(x + i)), q, scale, i);
}

PD_EXPORT
int pd_quantize_nvf4_rot(const void* x, void* q, void* scale, uint32_t n, void* stream) {
    if (n == 0) return 0;
    if ((n & 127u) != 0) return cudaErrorInvalidValue;
    pd_quantize_nvf4_rot_kernel<<<(n / 128u * 32u + 255u) / 256u, 256u, 0,
                                  (cudaStream_t)stream>>>(
        (const float*)x, (unsigned char*)q, (unsigned char*)scale, n);
    return pd_launch_status();
}

// Q8_0 -> H128-rotated nvf4 weight planes: dequant, rotate each row's
// 128-chunks (rows are contiguous and in_dim % 128 == 0, so flat 128-groups
// never straddle rows), per-16 e4m3 quantize. The load-time twin of the
// activation kernel - the same H on both sides makes the GEMM an identity
// rotation in reals.
__global__ void pd_q8_0_to_nvf4_rot_kernel(const int8_t* __restrict__ q8,
                                           const __half* __restrict__ s8,
                                           unsigned char* __restrict__ data,
                                           unsigned char* __restrict__ scale,
                                           uint64_t n_blocks) {
    const uint64_t i =
        ((uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5)) * 128u + (threadIdx.x & 31u) * 4u;
    if (i >= n_blocks * 32u) return;
    const float d = __half2float(s8[i >> 5]);
    const char4 qv = *(const char4*)(q8 + i);
    float4 v = make_float4((float)qv.x * d, (float)qv.y * d, (float)qv.z * d,
                           (float)qv.w * d);
    pd_nvf4_rot_store(pd_fht128(v), data, scale, (uint32_t)i);
}

PD_EXPORT
int pd_q8_0_to_nvf4_rot(const void* q8_data, const void* q8_scale, void* mx_data,
                        void* mx_scale, uint64_t n_blocks, void* stream) {
#ifndef PD_BS_HOST
    (void)q8_data; (void)q8_scale; (void)mx_data; (void)mx_scale; (void)n_blocks;
    (void)stream;
    return cudaErrorNotSupported;
#else
    if (n_blocks == 0) return 0;
    if ((n_blocks & 3u) != 0) return cudaErrorInvalidValue;  // 128-groups
    const uint32_t groups = (uint32_t)(n_blocks / 4u);
    pd_q8_0_to_nvf4_rot_kernel<<<(groups + 7u) / 8u, 256u, 0, (cudaStream_t)stream>>>(
        (const int8_t*)q8_data, (const __half*)q8_scale, (unsigned char*)mx_data,
        (unsigned char*)mx_scale, n_blocks);
    return pd_launch_status();
#endif
}


// ---- modelopt NVFP4 checkpoint consumers  -----------------------
// Layout as shipped by NVIDIA ModelOpt W4A16_NVFP4 exports (first user:
// Nemotron 3.5 Lightning; verified against the shipped shards):
//   data   u8  [out, in/2]   e2m1 nibbles packed ADJACENT (low nibble = even
//                            element - the order the nv4 MMA lane pins above)
//   scale  u8  [out, in/16]  e4m3 per-16 block scales, row-major flat
//   scale2 f32 scalar        per-tensor global scale
// Dequant: w = (e2m1 * e4m3) * scale2. scale2 is per-TENSOR, so it never
// rides the k loop: consumers fold it once in the epilogue (exact - it
// factors out of every dot product); only the oracle kernel applies it per
// element, in the exact multiply order the engine's host reference uses.

// Oracle/debug primitive only - serving through a materialized f32 plane
// forfeits the 4-bit byte advantage that motivates the format; serving
// consumers read the packed plane directly (pd_nvf4_gemv below, MoE lanes
// to follow on the f8row skeleton). One CTA per output row.
__global__ void pd_nvf4_dequant_kernel(const uint8_t* __restrict__ data,
                                       const uint8_t* __restrict__ scale,
                                       float scale2, float* __restrict__ y,
                                       uint32_t in_dim, uint32_t out_dim) {
#if PD_NV4_OK
    const uint32_t o = blockIdx.x;
    if (o >= out_dim) return;
    const uint32_t kh = in_dim >> 1;
    const uint8_t* row = data + (size_t)o * kh;
    const uint8_t* srow = scale + (size_t)o * (in_dim >> 4);
    float* yrow = y + (size_t)o * in_dim;
    for (uint32_t j = threadIdx.x; j < kh; j += blockDim.x) {
        const uint8_t b = row[j];
        const float s = (float)reinterpret_cast<const __nv_fp8_e4m3&>(srow[j >> 3]);
        yrow[2u * j] = (pd_e2m1_val(b & 15u) * s) * scale2;
        yrow[2u * j + 1u] = (pd_e2m1_val(b >> 4) * s) * scale2;
    }
#else
    (void)data; (void)scale; (void)scale2; (void)y; (void)in_dim; (void)out_dim;
#endif
}

PD_EXPORT
int pd_nvf4_dequant(const void* data, const void* scale, float scale2, void* y,
                    uint32_t in_dim, uint32_t out_dim, void* stream) {
#ifndef PD_BS_HOST
    (void)data; (void)scale; (void)scale2; (void)y; (void)in_dim; (void)out_dim;
    (void)stream;
    return cudaErrorNotSupported;
#else
    if (out_dim == 0) return 0;
    if ((in_dim & 31u) != 0) return cudaErrorInvalidValue;
    pd_nvf4_dequant_kernel<<<out_dim, 256u, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)data, (const uint8_t*)scale, scale2, (float*)y, in_dim,
        out_dim);
    return pd_launch_status();
#endif
}

// W4A16-class GEMV over a checkpoint NVFP4 plane: f32-exact activations
// (the checkpoint recipe keeps activations at 16-bit+; f32 is the engine's
// GEMV activation class), e2m1 decoded via pd_fp4_gemv's measured-best prmt
// path, per-tensor scale2 folded once after the reduction. Structure is
// warp-per-row, 8 rows per CTA (the bf16_gemv_mr rationale): at this
// model's K (1856-2688) a one-row CTA runs its whole K walk in one loop
// iteration - load-latency-bound, measured 621 GB/s on lm_head - while
// eight independent warps per CTA keep the pipe full. No smem: each lane
// owns a full 16-element-aligned 32-chunk, so its two e4m3 scale bytes are
// adjacent loads (64 B/warp/iter, L1-coalesced).
#define PD_NV4G_STEP(e)                                                    \
    {                                                                      \
        const uint32_t wb =                                                \
            (uint32_t)*reinterpret_cast<const uint16_t*>(row + ((e) >> 1)); \
        const float s =                                                    \
            (float)reinterpret_cast<const __nv_fp8_e4m3&>(srow[(e) >> 4]); \
        const float4 xv = *reinterpret_cast<const float4*>(x + (e));       \
        const uint32_t v = (wb & 0xFu) | ((wb & 0xF0u) << 4)               \
                         | ((wb & 0xF00u) << 8) | ((wb & 0xF000u) << 12);  \
        const uint32_t mag = v & 0x07070707u;                              \
        const uint32_t t = (mag | (mag >> 4)) & 0x00FF00FFu;               \
        const uint32_t e4 = __byte_perm(T0, T1, (t | (t >> 8)) & 0xFFFFu)  \
                          | ((v & 0x08080808u) << 4);                      \
        const __nv_fp8_e4m3* eb =                                          \
            reinterpret_cast<const __nv_fp8_e4m3*>(&e4);                   \
        acc += s * ((float)eb[0] * xv.x + (float)eb[1] * xv.y              \
                  + (float)eb[2] * xv.z + (float)eb[3] * xv.w);            \
    }
__global__ void pd_nvf4_gemv_kernel(
    const uint8_t* __restrict__ data, const uint8_t* __restrict__ scale,
    const float* __restrict__ bias, const float* __restrict__ x,
    float* __restrict__ y, float scale2, uint32_t in_dim, uint32_t out_dim) {
#if PD_NV4_OK
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t o = blockIdx.x * (blockDim.x >> 5) + warp;
    if (o >= out_dim) return;
    const uint8_t* row = data + (size_t)o * (in_dim >> 1);
    const uint8_t* srow = scale + (size_t)o * (in_dim >> 4);
    float acc = 0.0f;
    constexpr uint32_t T0 = 0x3C383000u, T1 = 0x4C484440u;
    // Warp-COHERENT 128-element steps (the lane-owns-chunk layout hit
    // the L1/TEX wall at 97.9% with DRAM at 46.6% - each x float4 warp load
    // touched 32 distinct sectors because lanes strode 128 B apart). Here
    // lane l owns elements k0+4l..k0+4l+3, so per step the warp issues one
    // contiguous 512 B x load, one contiguous 64 B weight load (u16/lane),
    // and one scale byte per 4-lane group - 3 coherent load instructions
    // per 128 elements instead of 10 fragmented ones per 32.
    //
    // The ragged tail (in_dim % 128 may be 32/64/96) is handled outside the
    // main loop: an in-loop `break` costs 24% of the kernel - ptxas can't
    // pipeline loads past a data-dependent exit, and the branch never even
    // fires at lm_head's K=2688 (21 exact steps). Hoisting it took the
    // serve-shape sweep (DRAM-cold) from
    // 1100 -> 1442 GB/s (94% of the 1531 practical roof), bit-exact; wider
    // per-lane loads and 2-rows-per-warp bought nothing on top.
    const uint32_t full = in_dim & ~127u;
    #pragma unroll 4
    for (uint32_t k0 = 0; k0 < full; k0 += 128u) PD_NV4G_STEP(k0 + lane * 4u)
    if (full < in_dim) {
        const uint32_t e = full + lane * 4u;
        if (e < in_dim) PD_NV4G_STEP(e)
    }
    for (uint32_t s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, s);
    if (lane == 0) {
        float v = acc * scale2;
        if (bias) v += bias[o];
        y[o] = v;
    }
#else
    (void)data; (void)scale; (void)bias; (void)x; (void)y; (void)scale2;
    (void)in_dim; (void)out_dim;
#endif
}

// Merged q|k|v NVFP4 GEMV. Same inner loop as
// pd_nvf4_gemv_kernel; the only change is that one grid covers three output
// planes that all read the same x.
//
// Why: granite's k/v are out_dim 1024, which at rows_per_cta 8 is 128 CTAs on
// a 188-SM die -- under one wave, so they cost the same ~8.5 us as q/o at 4x
// the bytes (granite-4.2-8b c1: q/k/v/o all ~8.5 us, 45% of the read
// roof while the FFN shapes hit 71%). Merging gives 6144 rows = 768 CTAs and
// reads x once. This is the same lever deltanet/core.cuh already measured for
// the Q8 lane on this die: 1024-row 724 GB/s, 4096-row 1254, merged 6144-row
// 1303 -- separate 26.5 us vs merged 20.5.
//
// The segment resolve is a compile-time if-chain, not segs.s[si]: a runtime
// index forces the by-value struct to local memory and every pointer read
// becomes an LDL round trip (-19%/-42% measured on the Q8 twin).
struct PdNv4GemvSeg {
    const uint8_t* data;
    const uint8_t* scale;
    const float* bias;
    float* y;
    float scale2;
    uint32_t out_dim;
};
struct PdNv4GemvSegs3 { PdNv4GemvSeg s[3]; };

__global__ void pd_nvf4_gemv_multi_kernel(
    PdNv4GemvSegs3 segs, const float* __restrict__ x, uint32_t in_dim) {
#if PD_NV4_OK
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    uint32_t o = blockIdx.x * (blockDim.x >> 5) + warp;
    const uint8_t* __restrict__ data;
    const uint8_t* __restrict__ scale;
    const float* bias;
    float* __restrict__ y;
    float scale2;
    const uint32_t d0 = segs.s[0].out_dim, d1 = segs.s[1].out_dim;
    if (o < d0) {
        data = segs.s[0].data; scale = segs.s[0].scale;
        bias = segs.s[0].bias; y = segs.s[0].y; scale2 = segs.s[0].scale2;
    } else if (o < d0 + d1) {
        o -= d0;
        data = segs.s[1].data; scale = segs.s[1].scale;
        bias = segs.s[1].bias; y = segs.s[1].y; scale2 = segs.s[1].scale2;
    } else {
        o -= d0 + d1;
        if (o >= segs.s[2].out_dim) return;
        data = segs.s[2].data; scale = segs.s[2].scale;
        bias = segs.s[2].bias; y = segs.s[2].y; scale2 = segs.s[2].scale2;
    }
    const uint8_t* row = data + (size_t)o * (in_dim >> 1);
    const uint8_t* srow = scale + (size_t)o * (in_dim >> 4);
    float acc = 0.0f;
    constexpr uint32_t T0 = 0x3C383000u, T1 = 0x4C484440u;
    #define PD_NV4GM_STEP(e)                                                   \
        {                                                                      \
            const uint32_t wb =                                                \
                (uint32_t)*reinterpret_cast<const uint16_t*>(row + ((e) >> 1)); \
            const float sc =                                                   \
                (float)reinterpret_cast<const __nv_fp8_e4m3&>(srow[(e) >> 4]); \
            const float4 xv = *reinterpret_cast<const float4*>(x + (e));       \
            const uint32_t v = (wb & 0xFu) | ((wb & 0xF0u) << 4)               \
                             | ((wb & 0xF00u) << 8) | ((wb & 0xF000u) << 12);  \
            const uint32_t mag = v & 0x07070707u;                              \
            const uint32_t t = (mag | (mag >> 4)) & 0x00FF00FFu;               \
            const uint32_t e4 = __byte_perm(T0, T1, (t | (t >> 8)) & 0xFFFFu)  \
                              | ((v & 0x08080808u) << 4);                      \
            const __nv_fp8_e4m3* eb =                                          \
                reinterpret_cast<const __nv_fp8_e4m3*>(&e4);                   \
            acc += sc * ((float)eb[0] * xv.x + (float)eb[1] * xv.y             \
                       + (float)eb[2] * xv.z + (float)eb[3] * xv.w);           \
        }
    const uint32_t full = in_dim & ~127u;
    #pragma unroll 4
    for (uint32_t k0 = 0; k0 < full; k0 += 128u) PD_NV4GM_STEP(k0 + lane * 4u)
    if (full < in_dim) {
        const uint32_t e = full + lane * 4u;
        if (e < in_dim) PD_NV4GM_STEP(e)
    }
    #undef PD_NV4GM_STEP
    for (uint32_t sh = 16; sh > 0; sh >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, sh);
    if (lane == 0) {
        float vv = acc * scale2;
        if (bias) vv += bias[o];
        y[o] = vv;
    }
#else
    (void)segs; (void)x; (void)in_dim;
#endif
}

PD_EXPORT
int pd_nvf4_gemv_multi(const void* segs_in, const void* x, uint32_t in_dim,
                       uint32_t n_segs, void* stream) {
#ifndef PD_BS_HOST
    (void)segs_in; (void)x; (void)in_dim; (void)n_segs; (void)stream;
    return cudaErrorNotSupported;
#else
    if (n_segs == 0 || n_segs > 3) return cudaErrorInvalidValue;
    if ((in_dim & 31u) != 0) return cudaErrorInvalidValue;
    PdNv4GemvSegs3 segs{};
    const PdNv4GemvSeg* in = (const PdNv4GemvSeg*)segs_in;
    uint32_t total = 0;
    for (uint32_t i = 0; i < n_segs; ++i) { segs.s[i] = in[i]; total += in[i].out_dim; }
    if (total == 0) return 0;
    // Same out_dim-elected CTA width as the single-plane entry, keyed on the
    // COMBINED row count so the two entries cannot disagree -- and the merged
    // count is what matters: q|k|v is 6144 rows together, which the sweep puts
    // squarely in the narrow-CTA regime (14.16 us at 64 threads) even though
    // its largest SEGMENT is only 4096. Keying this on a segment would have
    // left it wide. (gate|up would also cross, but merging it loses 6% for an
    // unrelated reason -- see the note at its seat in granite/batch.rs.)
    const uint32_t rows_per_cta = total >= 4096u ? 2u : 8u;
    const uint32_t grid = (total + rows_per_cta - 1u) / rows_per_cta;
    pd_nvf4_gemv_multi_kernel<<<grid, rows_per_cta * 32u, 0, (cudaStream_t)stream>>>(
        segs, (const float*)x, in_dim);
    return pd_launch_status();
#endif
}


#undef PD_NV4G_STEP

// Warp-private cp.async staging, for planes whose out_dim leaves the
// register-return load path starved.
//
// Profiling the plain kernel at [4096, 4096]: DRAM 43.8%, L1/TEX 22.6%, L2 21.8%
// -- no cache near its wall -- but 65.4% of warp cycles are a scoreboard stall
// on a global LOAD. That is the SM out of outstanding-miss capacity on the
// global->register path, and it is why adding warps, unroll depth, per-lane
// load width, or CTA count all measured neutral: they keep that path.
//
// A CTA-wide cp.async tile was tried first and LOST (765 vs 808 GB/s here),
// because its two __syncthreads() per stage couple all warps to the slowest.
// This version keeps the async prefetch and deletes the coupling: every warp
// stages its own row, so the only ordering is cp.async.wait_group plus
// __syncwarp(). Measured, DRAM-cold, against the best plain arm:
//   [4096, 4096]   11.72 -> 10.29 us   805 -> 917 GB/s   +12.2%
//   [4096, 12800]  30.87 -> 28.40 us   955 -> 1039 GB/s   +8.0%
//   [1024, 4096]    9.88 ->  6.32 us   239 ->  373 GB/s  +37%
// It loses above ~8192 rows (25.62 -> 26.84 at [12800, 4096]), where the plain
// kernel already has the CTAs to hide the latency and the staging is pure
// overhead -- hence the out_dim gate at the call site.
//
// REORDER class: a lane owns 16 contiguous elements (exactly one scale group)
// rather than a strided four, so the summation order differs from the plain
// kernel. maxrel 4e-4..1e-3 against it, i.e. far inside e2m1's own step.
#define PD_NV4CPW_KC 512u   // elements per stage: 256 B of nibbles + 32 B of scale
#define PD_NV4CPW_ROW 288u  // that row, padded to a cp.async 16 B multiple
template <uint32_t STAGES, uint32_t WARPS>
__global__ void pd_nvf4_gemv_cpw_kernel(
    const uint8_t* __restrict__ data, const uint8_t* __restrict__ scale,
    const float* __restrict__ bias, const float* __restrict__ x,
    float* __restrict__ y, float scale2, uint32_t in_dim, uint32_t out_dim) {
#if PD_NV4_OK
    __shared__ __align__(16) uint8_t buf[WARPS][STAGES][PD_NV4CPW_ROW];
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t o = blockIdx.x * WARPS + warp;
    const bool live = o < out_dim;
    const uint32_t nstage = in_dim / PD_NV4CPW_KC;
    // an out-of-range warp still walks the pipeline (its cp.async are issued
    // with src_ok=false, which zero-fills) so its group counts stay in step
    const uint8_t* row = data + (size_t)(live ? o : 0u) * (size_t)(in_dim >> 1);
    const uint8_t* srow = scale + (size_t)(live ? o : 0u) * (size_t)(in_dim >> 4);

    #define PD_NV4CPW_ISSUE(bufi, st)                                              \
        {                                                                          \
            const uint32_t k = (st) * PD_NV4CPW_KC;                                \
            if (lane < 16u)                                                        \
                pd_cp_async16(&buf[warp][bufi][lane << 4],                         \
                              row + (k >> 1) + (lane << 4), live);                 \
            else if (lane < 18u)                                                   \
                pd_cp_async16(&buf[warp][bufi][256u + ((lane - 16u) << 4)],        \
                              srow + (k >> 4) + ((lane - 16u) << 4), live);        \
        }

    #pragma unroll
    for (uint32_t st = 0; st + 1u < STAGES; ++st) {
        PD_NV4CPW_ISSUE(st, st)
        asm volatile("cp.async.commit_group;");
    }

    float acc = 0.0f;
    constexpr uint32_t T0 = 0x3C383000u, T1 = 0x4C484440u;
    for (uint32_t st = 0; st < nstage; ++st) {
        asm volatile("cp.async.wait_group %0;" ::"n"(STAGES - 2));
        __syncwarp();
        const uint8_t* wp = &buf[warp][st % STAGES][lane << 3];
        const float sc = (float)reinterpret_cast<const __nv_fp8_e4m3&>(
            buf[warp][st % STAGES][256u + lane]);
        const float* xp = x + st * PD_NV4CPW_KC + (lane << 4);
        float sub = 0.0f;
        #pragma unroll
        for (uint32_t q = 0; q < 4u; ++q) {
            const uint32_t wb = (uint32_t)*reinterpret_cast<const uint16_t*>(wp + q * 2u);
            const float4 xv = *reinterpret_cast<const float4*>(xp + q * 4u);
            const uint32_t v = (wb & 0xFu) | ((wb & 0xF0u) << 4)
                             | ((wb & 0xF00u) << 8) | ((wb & 0xF000u) << 12);
            const uint32_t mag = v & 0x07070707u;
            const uint32_t tt = (mag | (mag >> 4)) & 0x00FF00FFu;
            const uint32_t e4 = __byte_perm(T0, T1, (tt | (tt >> 8)) & 0xFFFFu)
                              | ((v & 0x08080808u) << 4);
            const __nv_fp8_e4m3* eb = reinterpret_cast<const __nv_fp8_e4m3*>(&e4);
            sub += (float)eb[0] * xv.x + (float)eb[1] * xv.y
                 + (float)eb[2] * xv.z + (float)eb[3] * xv.w;
        }
        acc += sc * sub;
        __syncwarp();  // the slot is fully read before it is refilled
        const uint32_t nxt = st + STAGES - 1u;
        if (nxt < nstage) PD_NV4CPW_ISSUE(nxt % STAGES, nxt)
        asm volatile("cp.async.commit_group;");
    }
    #undef PD_NV4CPW_ISSUE

    for (uint32_t r = 16; r > 0; r >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, r);
    if (lane == 0 && live) {
        float v = acc * scale2;
        if (bias) v += bias[o];
        y[o] = v;
    }
#else
    (void)data; (void)scale; (void)bias; (void)x; (void)y; (void)scale2;
    (void)in_dim; (void)out_dim;
#endif
}

PD_EXPORT
int pd_nvf4_gemv(const void* data, const void* scale, const void* bias,
                 const void* x, void* y, float scale2, uint32_t in_dim,
                 uint32_t out_dim, void* stream) {
#ifndef PD_BS_HOST
    (void)data; (void)scale; (void)bias; (void)x; (void)y; (void)scale2;
    (void)in_dim; (void)out_dim; (void)stream;
    return cudaErrorNotSupported;
#else
    if (out_dim == 0) return 0;
    if ((in_dim & 31u) != 0) return cudaErrorInvalidValue;
    // The staged kernel above owns the starved shapes. in_dim must divide the
    // 512-element stage: it walks in_dim/512 stages with no ragged tail, so an
    // unaligned K (nemotron's lm_head is 2688) would silently drop the
    // remainder -- caught in the bench, where those arms produced nothing at
    // all rather than a wrong number.
    if ((in_dim % PD_NV4CPW_KC) == 0u && out_dim <= 4096u) {
        const uint32_t grid = (out_dim + 3u) / 4u;
        pd_nvf4_gemv_cpw_kernel<3u, 4u><<<grid, 128u, 0, (cudaStream_t)stream>>>(
            (const uint8_t*)data, (const uint8_t*)scale, (const float*)bias,
            (const float*)x, (float*)y, scale2, in_dim, out_dim);
        return pd_launch_status();
    }
    // CTA width is elected by out_dim, measured DRAM-cold over a
    // granite-4.2 sweep. A wide
    // plane wants NARROW CTAs and a narrow plane wants wide ones:
    //   out 6144   64 thr 14.16 us  128 thr 14.21  256 thr  -
    //   out 8192   64 thr 16.23 us  128 thr 16.25  256 thr 18.29
    //   out 12800  64 thr 25.73 us  128 thr 25.97  256 thr 28.07
    //   [131072, 2688]  64 thr 1445 GB/s  128 thr 1444  256 thr 1440
    // 64 threads (2 rows) wins or ties everywhere above 4096 rows, and below
    // that the staged kernel has already taken the plane, so the only callers
    // left down there are the K-unaligned ones -- where all widths tie and the
    // wide CTA is kept for its eight independent warp streams.
    // 4 rows/CTA doubles the CTA count, which only pays once there are enough
    // rows to hand every SM several of them; below that the wider CTA's eight
    // independent warp streams matter more. Bit-exact either way - the kernel
    // derives rows-per-CTA from blockDim, so this is launch config only.
    const uint32_t rows_per_cta = out_dim >= 4096u ? 2u : 8u;
    const uint32_t grid = (out_dim + rows_per_cta - 1u) / rows_per_cta;
    pd_nvf4_gemv_kernel<<<grid, rows_per_cta * 32u, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)data, (const uint8_t*)scale, (const float*)bias,
        (const float*)x, (float*)y, scale2, in_dim, out_dim);
    return pd_launch_status();
#endif
}

