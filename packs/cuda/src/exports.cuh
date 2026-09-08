// exports.cuh (formerly 17_glue_exports.cuh) - glue fusions (norm+quant batch) + kernel table + pack entry points
// Textually-included segment of the single pack translation unit.
// Not standalone-compilable: include order is defined by ../pack.cu.
// ---- glue fusions: the norm output used
// to take a full round trip (write f32, separate kernel reads it back to
// quantize) plus a launch, per layer per projection class. Both fused
// kernels keep the f32 plane (router / fallback paths read it) and emit the
// quantized plane in phase 2 from registers. Quantize math is verbatim
// (q8: pd_quantize_q8's warp amax - commutative, so values are identical;
// e4m3: pd_e4m3_quant4). Requires n % 32 == 0 and 1024-thread blocks so a
// warp iteration covers exactly one 32-block.

// rmsnorm + Q8_0 quantize (attn-norm -> wqkv GEMM, out-norm -> lm_head)
__global__ void pd_rmsnorm_quant_q8_batch_kernel(
    const float* __restrict__ x, const float* __restrict__ w,
    float* __restrict__ out, signed char* __restrict__ q,
    float* __restrict__ qs, uint32_t n, float eps) {
    const uint32_t b = blockIdx.x;
    const float* xb = x + (size_t)b * n;
    float* ob = out + (size_t)b * n;
    signed char* qb = q + (size_t)b * n;
    float* sb = qs + (size_t)b * (n >> 5);
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    __shared__ float wsum[32];
    __shared__ float s_inv;
    float acc = 0.0f;
    for (uint32_t i = tid; i < n; i += nth) {
        const float v = xb[i];
        acc += v * v;
    }
    for (uint32_t sh = 16; sh > 0; sh >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, sh);
    const uint32_t warp = tid >> 5, lane = tid & 31u;
    if (lane == 0) wsum[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        float sum = 0.0f;
        const uint32_t nwarps = (nth + 31u) >> 5;
        for (uint32_t wi = 0; wi < nwarps; ++wi) sum += wsum[wi];
        s_inv = 1.0f / sqrtf(sum / (float)n + eps);
    }
    __syncthreads();
    const float inv = s_inv;
    // n % 32 == 0 and nth % 32 == 0: each warp iteration is one q8 block
    for (uint32_t i = tid; i < n; i += nth) {
        const float v = xb[i] * inv * w[i];
        ob[i] = v;
        float a = fabsf(v);
        for (uint32_t sh = 16; sh > 0; sh >>= 1)
            a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, sh));
        const float scl = a * (1.0f / 127.0f);
        if (lane == 0) sb[i >> 5] = scl;
        const float qinv = scl > 0.0f ? 1.0f / scl : 0.0f;
        int qi = __float2int_rn(v * qinv);
        qi = qi < -127 ? -127 : (qi > 127 ? 127 : qi);
        qb[i] = (signed char)qi;
    }
}

// Cross-layer glue fold: layer N's MoE slot-combine rides layer N+1's
// norm+quantize pass. Phase 1 folds the per-(token, slot) down partials
// into the residual in FIXED ascending slot order (exactly
// pd_moe_slot_combine's per-element math) while accumulating the square
// sum; phase 2 is pd_rmsnorm_quant_q8_batch_kernel verbatim. Kills the
// standalone combine launch AND its separate residual read/write pass.
__global__ void pd_moe_combine_rmsnorm_quant_q8_kernel(
    float* __restrict__ x, const float* __restrict__ part, const float* __restrict__ w,
    float* __restrict__ out, signed char* __restrict__ q, float* __restrict__ qs,
    uint32_t n, uint32_t n_active, float eps) {
    const uint32_t b = blockIdx.x;
    float* xb = x + (size_t)b * n;
    float* ob = out + (size_t)b * n;
    signed char* qb = q + (size_t)b * n;
    float* sb = qs + (size_t)b * (n >> 5);
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    __shared__ float wsum[32];
    __shared__ float s_inv;
    float acc = 0.0f;
    for (uint32_t i = tid; i < n; i += nth) {
        float v = xb[i];
        for (uint32_t k = 0; k < n_active; ++k)
            v += part[((size_t)b * n_active + k) * n + i];
        xb[i] = v;
        acc += v * v;
    }
    for (uint32_t sh = 16; sh > 0; sh >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, sh);
    const uint32_t warp = tid >> 5, lane = tid & 31u;
    if (lane == 0) wsum[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        float sum = 0.0f;
        const uint32_t nwarps = (nth + 31u) >> 5;
        for (uint32_t wi = 0; wi < nwarps; ++wi) sum += wsum[wi];
        s_inv = 1.0f / sqrtf(sum / (float)n + eps);
    }
    __syncthreads();
    const float inv = s_inv;
    for (uint32_t i = tid; i < n; i += nth) {
        const float v = xb[i] * inv * w[i];
        ob[i] = v;
        float a = fabsf(v);
        for (uint32_t sh = 16; sh > 0; sh >>= 1)
            a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, sh));
        const float scl = a * (1.0f / 127.0f);
        if (lane == 0) sb[i >> 5] = scl;
        const float qinv = scl > 0.0f ? 1.0f / scl : 0.0f;
        int qi = __float2int_rn(v * qinv);
        qi = qi < -127 ? -127 : (qi > 127 ? 127 : qi);
        qb[i] = (signed char)qi;
    }
}

PD_EXPORT
int pd_moe_combine_rmsnorm_quant_q8(void* x, const void* part, const void* w,
                                    void* out, void* q, void* qs, uint32_t n,
                                    uint32_t n_active, float eps, uint32_t batch,
                                    void* stream) {
    if (n == 0 || batch == 0) return 0;
    if (n & 31u) return cudaErrorInvalidValue;
    pd_moe_combine_rmsnorm_quant_q8_kernel<<<batch, (batch >= 64u ? pd_norm_wide_nth(batch) : 1024u), 0, (cudaStream_t)stream>>>(
        (float*)x, (const float*)part, (const float*)w, (float*)out,
        (signed char*)q, (float*)qs, n, n_active, eps);
    return pd_launch_status();
}

PD_EXPORT
int pd_rmsnorm_quant_q8_batch(const void* x, const void* w, void* out, void* q,
                              void* qs, uint32_t n, float eps, uint32_t batch,
                              void* stream) {
    if (n == 0 || batch == 0) return 0;
    if (n & 31u) return cudaErrorInvalidValue;
    pd_rmsnorm_quant_q8_batch_kernel<<<batch, (batch >= 64u ? pd_norm_wide_nth(batch) : 1024u), 0, (cudaStream_t)stream>>>(
        (const float*)x, (const float*)w, (float*)out, (signed char*)q, (float*)qs, n, eps);
    return pd_launch_status();
}

// residual-add + rmsnorm + e4m3/ue8m0 quantize (post-norm -> block-scale MoE)
__global__ void pd_add_rmsnorm_quant_e4m3_batch_kernel(
    float* __restrict__ x, const float* __restrict__ proj,
    const float* __restrict__ w, float* __restrict__ out,
    unsigned char* __restrict__ q, unsigned char* __restrict__ s8, uint32_t n,
    float eps) {
    const uint32_t b = blockIdx.x;
    float* xb = x + (size_t)b * n;
    const float* pb = proj + (size_t)b * n;
    float* ob = out + (size_t)b * n;
    unsigned char* qb = q + (size_t)b * n;
    unsigned char* sb = s8 + (size_t)b * (n >> 5);
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    __shared__ float wsum[32];
    __shared__ float s_inv;
    float acc = 0.0f;
    const uint32_t n4 = n >> 2;
    float4* x4 = reinterpret_cast<float4*>(xb);
    const float4* p4 = reinterpret_cast<const float4*>(pb);
    for (uint32_t i = tid; i < n4; i += nth) {
        float4 v = x4[i];
        const float4 pv = p4[i];
        v.x += pv.x; v.y += pv.y; v.z += pv.z; v.w += pv.w;
        x4[i] = v;
        acc += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    for (uint32_t sh = 16; sh > 0; sh >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, sh);
    const uint32_t warp = tid >> 5, lane = tid & 31u;
    if (lane == 0) wsum[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        float sum = 0.0f;
        const uint32_t nwarps = (nth + 31u) >> 5;
        for (uint32_t wi = 0; wi < nwarps; ++wi) sum += wsum[wi];
        s_inv = 1.0f / sqrtf(sum / (float)n + eps);
    }
    __syncthreads();
    const float inv = s_inv;
    const float4* w4 = reinterpret_cast<const float4*>(w);
    float4* o4 = reinterpret_cast<float4*>(ob);
    // n % 32 == 0: 8-lane groups own whole 32-blocks (pd_e4m3_quant4 contract)
    for (uint32_t i = tid; i < n4; i += nth) {
        const float4 xv = x4[i];
        const float4 wv = w4[i];
        float4 ov;
        ov.x = xv.x * inv * wv.x;
        ov.y = xv.y * inv * wv.y;
        ov.z = xv.z * inv * wv.z;
        ov.w = xv.w * inv * wv.w;
        o4[i] = ov;
        pd_e4m3_quant4(ov, tid & 7u, qb, sb, i * 4u);
    }
}

PD_EXPORT
int pd_add_rmsnorm_quant_e4m3_batch(void* x, const void* proj, const void* w,
                                    void* out, void* q, void* s8, uint32_t n,
                                    float eps, uint32_t batch, void* stream) {
    if (n == 0 || batch == 0) return 0;
    if (n & 31u) return cudaErrorInvalidValue;
    pd_add_rmsnorm_quant_e4m3_batch_kernel<<<batch, (batch >= 64u ? pd_norm_wide_nth(batch) : 1024u), 0, (cudaStream_t)stream>>>(
        (float*)x, (const float*)proj, (const float*)w, (float*)out,
        (unsigned char*)q, (unsigned char*)s8, n, eps);
    return pd_launch_status();
}

// residual-add + rmsnorm + Q8_0 quantize (deepseek-ocr decode
// glue): the dp4a-class sibling of the e4m3 kernel above - the OCR MoE
// pre-norm ran add_rmsnorm_batch then a separate quantize_q8 per layer per
// tick, and the router still needs the f32 plane. Phase 1 is
// pd_add_rmsnorm_batch_kernel's float4 add + square-sum verbatim (the
// residual write lands in x); phase 2 is pd_rmsnorm_quant_q8_batch_kernel's
// epilogue verbatim (out + q + qs). Values identical to the two-kernel
// sequence run at the same block width.
__global__ void pd_add_rmsnorm_quant_q8_batch_kernel(
    float* __restrict__ x, const float* __restrict__ proj,
    const float* __restrict__ w, float* __restrict__ out,
    signed char* __restrict__ q, float* __restrict__ qs, uint32_t n,
    float eps) {
    PD_PDL_ARM();  // proj is the predecessor GEMM's output (cascade)
    const uint32_t b = blockIdx.x;
    float* xb = x + (size_t)b * n;
    const float* pb = proj + (size_t)b * n;
    float* ob = out + (size_t)b * n;
    signed char* qb = q + (size_t)b * n;
    float* sb = qs + (size_t)b * (n >> 5);
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    __shared__ float wsum[32];
    __shared__ float s_inv;
    float acc = 0.0f;
    const uint32_t n4 = n >> 2;
    float4* x4 = reinterpret_cast<float4*>(xb);
    const float4* p4 = reinterpret_cast<const float4*>(pb);
    for (uint32_t i = tid; i < n4; i += nth) {
        float4 v = x4[i];
        const float4 pv = p4[i];
        v.x += pv.x; v.y += pv.y; v.z += pv.z; v.w += pv.w;
        x4[i] = v;
        acc += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    for (uint32_t sh = 16; sh > 0; sh >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, sh);
    const uint32_t warp = tid >> 5, lane = tid & 31u;
    if (lane == 0) wsum[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        float sum = 0.0f;
        const uint32_t nwarps = (nth + 31u) >> 5;
        for (uint32_t wi = 0; wi < nwarps; ++wi) sum += wsum[wi];
        s_inv = 1.0f / sqrtf(sum / (float)n + eps);
    }
    __syncthreads();
    const float inv = s_inv;
    // n % 32 == 0 and nth % 32 == 0: each warp iteration is one q8 block
    for (uint32_t i = tid; i < n; i += nth) {
        const float v = xb[i] * inv * w[i];
        ob[i] = v;
        float a = fabsf(v);
        for (uint32_t sh = 16; sh > 0; sh >>= 1)
            a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, sh));
        const float scl = a * (1.0f / 127.0f);
        if (lane == 0) sb[i >> 5] = scl;
        const float qinv = scl > 0.0f ? 1.0f / scl : 0.0f;
        int qi = __float2int_rn(v * qinv);
        qi = qi < -127 ? -127 : (qi > 127 ? 127 : qi);
        qb[i] = (signed char)qi;
    }
}

PD_EXPORT
int pd_add_rmsnorm_quant_q8_batch(void* x, const void* proj, const void* w,
                                  void* out, void* q, void* qs, uint32_t n,
                                  float eps, uint32_t batch, void* stream) {
    if (n == 0 || batch == 0) return 0;
    if (n & 31u) return cudaErrorInvalidValue;
    // width election mirrors pd_add_rmsnorm_batch so inv stays in the same
    // reduction class as the unfused chain on every die
    pd_pdl_go(pd_add_rmsnorm_quant_q8_batch_kernel, batch,
              (batch >= 64u ? pd_norm_wide_nth(batch) : pd_norm_decode_nth()), 0u, (cudaStream_t)stream,
              (float*)x, (const float*)proj, (const float*)w, (float*)out,
              (signed char*)q, (float*)qs, n, eps);
    return pd_launch_status();
}

// SwiGLU + Q8_0 quantize in one pass: the decode FFN/shexp band
// ran silu(gate)*up in place then read it back to quantize - two tiny
// launches per layer per tick and a full activation round trip. One warp
// per 32-block, exactly pd_quantize_q8_kernel's shape; the activation is
// computed in registers with pd_swiglu_kernel's expression verbatim, so
// q/scale values are bit-identical to swiglu -> quantize_q8. gate is left
// unmodified (nothing reads the activated plane after the down GEMM).
__global__ void pd_swiglu_quant_q8_kernel(const float* __restrict__ gate,
                                          const float* __restrict__ up,
                                          signed char* __restrict__ q,
                                          float* __restrict__ scale,
                                          uint32_t n_blocks) {
    PD_PDL_ARM();  // gate/up are the predecessor GEMMs' outputs
    uint32_t b = blockIdx.x;
    if (b >= n_blocks) return;
    uint32_t d = threadIdx.x;                 // 0..31, one warp per 32-block
    float g = gate[b * 32u + d];
    float v = (g / (1.0f + expf(-g))) * up[b * 32u + d];
    float a = fabsf(v);
    for (uint32_t s = 16; s > 0; s >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, s));
    float scl = a * (1.0f / 127.0f);
    if (d == 0) scale[b] = scl;
    float inv = scl > 0.0f ? 1.0f / scl : 0.0f;
    int qi = __float2int_rn(v * inv);
    qi = qi < -127 ? -127 : (qi > 127 ? 127 : qi);
    q[b * 32u + d] = (signed char)qi;
}

PD_EXPORT
int pd_swiglu_quant_q8(const void* gate, const void* up, void* q, void* scale,
                       uint32_t n, void* stream) {
    if (n == 0) return 0;
    if (n & 31u) return cudaErrorInvalidValue;
    uint32_t n_blocks = n >> 5;
    pd_pdl_go(pd_swiglu_quant_q8_kernel, n_blocks, 32u, 0u, (cudaStream_t)stream,
              (const float*)gate, (const float*)up, (signed char*)q,
              (float*)scale, n_blocks);
    return pd_launch_status();
}


// prefill-width SwiGLU + e4m3-row quant:
// silu(g)*u is pd_swiglu_kernel's expression verbatim; every chunk CTA walks
// the whole row for the exact max (row1p's scale derivation: order-free
// abs-max, power-of-two scale) and stores only its slice of q; gate is left
// unmodified -- the down GEMM reads the staged q, nothing reads the activated
// f32 plane. Bit-identical to swiglu -> quantize_e4m3_row by construction.
__global__ void __launch_bounds__(256) pd_swiglu_quant_e4m3_row_kernel(
        const float* __restrict__ gate, const float* __restrict__ up,
        unsigned char* __restrict__ q, float* __restrict__ rscale, uint32_t n) {
    PD_PDL_ARM();
    const uint32_t b = blockIdx.x, ch = blockIdx.y, C = gridDim.y;
    const float* gb = gate + (size_t)b * n;
    const float* ub = up + (size_t)b * n;
    unsigned char* qb = q + (size_t)b * n;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    __shared__ float wmax[32];
    __shared__ int s_e;
    float am = 0.0f;
    const uint32_t n4 = n >> 2;
    const float4* g4 = reinterpret_cast<const float4*>(gb);
    const float4* u4 = reinterpret_cast<const float4*>(ub);
    for (uint32_t i = tid; i < n4; i += nth) {
        const float4 g = g4[i], u = u4[i];
        const float vx = (g.x / (1.0f + expf(-g.x))) * u.x;
        const float vy = (g.y / (1.0f + expf(-g.y))) * u.y;
        const float vz = (g.z / (1.0f + expf(-g.z))) * u.z;
        const float vw = (g.w / (1.0f + expf(-g.w))) * u.w;
        am = fmaxf(am, fmaxf(fmaxf(fabsf(vx), fabsf(vy)), fmaxf(fabsf(vz), fabsf(vw))));
    }
    for (uint32_t sh = 16; sh > 0; sh >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, sh));
    if ((tid & 31u) == 0) wmax[tid >> 5] = am;
    __syncthreads();
    if (tid == 0) {
        float m = 0.0f;
        for (uint32_t w = 0; w < ((nth + 31u) >> 5); ++w) m = fmaxf(m, wmax[w]);
        int e = 0;
        if (m > 0.0f) { int ex; float fr = frexpf(m, &ex); e = ex - 9 + (fr > 0.875f ? 1 : 0); }
        s_e = e;
        if (ch == 0) rscale[b] = ldexpf(1.0f, e);
    }
    __syncthreads();
    const float inv = ldexpf(1.0f, -s_e);
    const uint32_t i0 = (n4 * ch) / C, i1 = (n4 * (ch + 1u)) / C;
    for (uint32_t i = i0 + tid; i < i1; i += nth) {
        const float4 g = g4[i], u = u4[i];
        uchar4 o;
        o.x = __nv_fp8_e4m3(((g.x / (1.0f + expf(-g.x))) * u.x) * inv).__x;
        o.y = __nv_fp8_e4m3(((g.y / (1.0f + expf(-g.y))) * u.y) * inv).__x;
        o.z = __nv_fp8_e4m3(((g.z / (1.0f + expf(-g.z))) * u.z) * inv).__x;
        o.w = __nv_fp8_e4m3(((g.w / (1.0f + expf(-g.w))) * u.w) * inv).__x;
        *(uchar4*)(qb + (size_t)i * 4u) = o;
    }
}

// Single-pass twin of pd_swiglu_quant_e4m3_row_kernel for prefill-width rows
// (measured on granite-4.2-30b prefill widths): the two-pass kernel
// walks the whole row for the amax and then walks it again to quantize; at
// 12800-wide rows the second walk hits L2 (2.3 TB/s effective), at 32768-wide
// rows (262 KB of f32 per row, ~376 rows in flight) it goes back to DRAM --
// 589 MB per launch in 408 us = 1.44 TB/s, the roof for twice the traffic the
// job needs. Here silu(g)*u is computed once and held in registers across the
// block amax reduction, so each row is read once and written once.
// Bit-identical to the two-pass kernel by construction: same per-element
// expression, order-free abs-max, same power-of-two scale derivation, same
// e4m3 conversion. EPT floats per thread; a row wider than NTH*EPT falls back
// to the two-pass kernel in the launcher.
template <uint32_t NTH, uint32_t EPT>
__global__ void __launch_bounds__(NTH) pd_swiglu_quant_e4m3_row1p_kernel(
        const float* __restrict__ gate, const float* __restrict__ up,
        unsigned char* __restrict__ q, float* __restrict__ rscale, uint32_t n) {
    PD_PDL_ARM();
    static_assert(EPT % 4u == 0u, "EPT is in float4 units");
    const uint32_t b = blockIdx.x;
    const float* gb = gate + (size_t)b * n;
    const float* ub = up + (size_t)b * n;
    unsigned char* qb = q + (size_t)b * n;
    const uint32_t tid = threadIdx.x;
    __shared__ float wmax[NTH / 32u];
    __shared__ int s_e;
    const uint32_t n4 = n >> 2;
    const float4* g4 = reinterpret_cast<const float4*>(gb);
    const float4* u4 = reinterpret_cast<const float4*>(ub);
    float v[EPT];
    float am = 0.0f;
#pragma unroll
    for (uint32_t k = 0; k < EPT / 4u; ++k) {
        const uint32_t i = tid + k * NTH;          // thread-strided: coalesced
        float4 o = make_float4(0.f, 0.f, 0.f, 0.f);
        if (i < n4) {
            const float4 g = g4[i], u = u4[i];
            o.x = (g.x / (1.0f + expf(-g.x))) * u.x;
            o.y = (g.y / (1.0f + expf(-g.y))) * u.y;
            o.z = (g.z / (1.0f + expf(-g.z))) * u.z;
            o.w = (g.w / (1.0f + expf(-g.w))) * u.w;
            am = fmaxf(am, fmaxf(fmaxf(fabsf(o.x), fabsf(o.y)), fmaxf(fabsf(o.z), fabsf(o.w))));
        }
        v[4u * k] = o.x; v[4u * k + 1u] = o.y; v[4u * k + 2u] = o.z; v[4u * k + 3u] = o.w;
    }
    for (uint32_t sh = 16; sh > 0; sh >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, sh));
    if ((tid & 31u) == 0) wmax[tid >> 5] = am;
    __syncthreads();
    if (tid == 0) {
        float m = 0.0f;
        for (uint32_t w = 0; w < NTH / 32u; ++w) m = fmaxf(m, wmax[w]);
        int e = 0;
        if (m > 0.0f) { int ex; float fr = frexpf(m, &ex); e = ex - 9 + (fr > 0.875f ? 1 : 0); }
        s_e = e;
        rscale[b] = ldexpf(1.0f, e);
    }
    __syncthreads();
    const float inv = ldexpf(1.0f, -s_e);
#pragma unroll
    for (uint32_t k = 0; k < EPT / 4u; ++k) {
        const uint32_t i = tid + k * NTH;
        if (i < n4) {
            uchar4 o;
            o.x = __nv_fp8_e4m3(v[4u * k] * inv).__x;
            o.y = __nv_fp8_e4m3(v[4u * k + 1u] * inv).__x;
            o.z = __nv_fp8_e4m3(v[4u * k + 2u] * inv).__x;
            o.w = __nv_fp8_e4m3(v[4u * k + 3u] * inv).__x;
            *(uchar4*)(qb + (size_t)i * 4u) = o;
        }
    }
}

PD_EXPORT
int pd_swiglu_quant_e4m3_row(const void* gate, const void* up, void* q, void* rscale,
                             uint32_t n, uint32_t batch, void* stream) {
    if (n == 0 || batch == 0) return 0;
    if (n & 31u) return cudaErrorInvalidValue;
    static int nsm = 0;
    if (nsm == 0) { int d = 0; cudaGetDevice(&d); cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, d); if (nsm <= 0) nsm = 148; }
    uint32_t C = 1u;
    if (batch * 2u < (uint32_t)nsm) { C = ((uint32_t)nsm + batch * 2u - 1u) / (batch * 2u); if (C > 8u) C = 8u; }
    // Prefill widths (C == 1: the machine is already full of rows) take the
    // single-pass twin -- one read per row instead of two (measured at
    // 17.7% of GPU time and 0.76 TB/s effective before the fold).
    // Rows wider than 512*64 floats keep the two-pass kernel. Kill:
    // PADDOCK_NO_SWIGLU_1P.
    static const bool no_1p = pd_env("PADDOCK_NO_SWIGLU_1P") != nullptr;
    if (C == 1u && !no_1p) {
        const uint32_t n4 = n >> 2;
        if (n4 <= 512u * 4u) {
            pd_pdl_go(pd_swiglu_quant_e4m3_row1p_kernel<512u, 16u>, dim3(batch), 512u, 0u, (cudaStream_t)stream,
                      (const float*)gate, (const float*)up, (unsigned char*)q, (float*)rscale, n);
            return pd_launch_status();
        } else if (n4 <= 512u * 8u) {
            pd_pdl_go(pd_swiglu_quant_e4m3_row1p_kernel<512u, 32u>, dim3(batch), 512u, 0u, (cudaStream_t)stream,
                      (const float*)gate, (const float*)up, (unsigned char*)q, (float*)rscale, n);
            return pd_launch_status();
        } else if (n4 <= 512u * 16u) {
            pd_pdl_go(pd_swiglu_quant_e4m3_row1p_kernel<512u, 64u>, dim3(batch), 512u, 0u, (cudaStream_t)stream,
                      (const float*)gate, (const float*)up, (unsigned char*)q, (float*)rscale, n);
            return pd_launch_status();
        }
    }
    pd_pdl_go(pd_swiglu_quant_e4m3_row_kernel, dim3(batch, C), 256u, 0u, (cudaStream_t)stream,
              (const float*)gate, (const float*)up, (unsigned char*)q, (float*)rscale, n);
    return pd_launch_status();
}

PD_EXPORT
int pd_add_rmsnorm_batch(void* x, const void* proj, const void* w, void* out,
                         uint32_t n, float eps, uint32_t batch, void* stream) {
    if (n == 0 || batch == 0) return 0;
    // Width by batch (the pd_rmsnorm_batch occupancy note): 1024-wide for
    // decode latency (b=32: 8.9us at 256-wide), 256-wide at
    // >=64 rows where 1024-thread blocks can't co-reside on a 1536-thread SM
    // and prefill widths ran ~15x over bandwidth floor. Reduction order
    // changes with the stride - sanctioned, because the parity bar is the
    // llama.cpp knife-edge gates, not a fixed sum order.
    // decode width overridable per die -- see pd_norm_decode_nth (the 1536-
    // thread-SM premise above is sm_120's, not B200's)
    pd_pdl_go(pd_add_rmsnorm_batch_kernel, batch, (batch >= 64u ? pd_norm_wide_nth(batch) : pd_norm_decode_nth()), 0u, (cudaStream_t)stream,
        (float*)x, (const float*)proj, (const float*)w, (float*)out, n, eps, 1.0f);
    return pd_launch_status();
}

// Granite's residual carries a multiplier, so it could not use the fused entry
// above and paid an extra pd_scale_add launch per norm. Same kernel, same
// widths, `pscale` folded into the add.
PD_EXPORT
int pd_add_rmsnorm_scaled_batch(void* x, const void* proj, const void* w, void* out,
                         uint32_t n, float eps, uint32_t batch, void* stream, float pscale) {
    if (n == 0 || batch == 0) return 0;
    // Width by batch (the pd_rmsnorm_batch occupancy note): 1024-wide for
    // decode latency (b=32: 8.9us at 256-wide), 256-wide at
    // >=64 rows where 1024-thread blocks can't co-reside on a 1536-thread SM
    // and prefill widths ran ~15x over bandwidth floor. Reduction order
    // changes with the stride - sanctioned, because the parity bar is the
    // llama.cpp knife-edge gates, not a fixed sum order.
    // decode width overridable per die -- see pd_norm_decode_nth (the 1536-
    // thread-SM premise above is sm_120's, not B200's)
    pd_pdl_go(pd_add_rmsnorm_batch_kernel, batch, (batch >= 64u ? pd_norm_wide_nth(batch) : pd_norm_decode_nth()), 0u, (cudaStream_t)stream,
        (float*)x, (const float*)proj, (const float*)w, (float*)out, n, eps, pscale);
    return pd_launch_status();
}

// The from_parts twin (nvf4 reduce-fold): `proj` arrives as `nz`
// raw split-K partial slices in `part` (stride batch*n) rather than a reduced
// plane, so the residual is folded from them with `scale2` inline -- the
// pd_nvf4_sk_reduce launch and its y round trip are gone. Bit-identical to
// reduce-then-add_rmsnorm_scaled (same fold order + the same float4 residual
// path). `bias` null for granite. Same block-width election as the batch twin.
PD_EXPORT
int pd_add_rmsnorm_scaled_from_parts(void* x, const void* part, const void* w,
                                     void* out, const void* bias, uint32_t n,
                                     float eps, uint32_t batch, float pscale,
                                     float scale2, uint32_t nz, void* stream) {
    if (n == 0 || batch == 0) return 0;
    pd_pdl_go(pd_add_rmsnorm_scaled_from_parts_kernel, batch, (batch >= 64u ? pd_norm_wide_nth(batch) : pd_norm_decode_nth()), 0u, (cudaStream_t)stream,
        (float*)x, (const float*)part, (const float*)w, (float*)out, (const float*)bias, n, eps, pscale, scale2, batch, nz);
    return pd_launch_status();
}

extern "C" int pd_nvf4_gemv_multi(const void*, const void*, uint32_t, uint32_t, void*);
extern "C" int pd_moe_head_router_hb(const void*, const void*, const void*, const void*, const void*, void*, void*, void*, void*, void*, uint32_t, uint32_t, uint32_t, float, uint32_t, void*);
extern "C" int pd_moe_head_xg(const void*, const void*, const void*, void*, void*, void*, void*, uint32_t, float, uint32_t, void*);
extern "C" int pd_q8_0_moe_gate_up_mma2g_geglu(const void*, const void*, const void*, const void*, const void*, const void*, const void*, const void*, void*, void*, uint32_t, uint32_t, uint32_t, uint32_t, void*);
extern "C" int pd_q8_0_moe_down_mma2_pbf16(const void*, const void*, const void*, const void*, const void*, const void*, const void*, const void*, void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, void*);
extern "C" int pd_moe_tail_combine_bf16(void*, const void*, const void*, const void*, const void*, const void*, uint32_t, uint32_t, float, float, uint32_t, void*);
extern "C" int pd_moe_router_stage(const void*, const void*, void*, const void*, void*, void*, uint32_t, uint32_t, uint32_t, uint32_t, void*);
extern "C" int pd_q8_0_moe_gate_up_mma2g_y64_geglu(const void*, const void*, const void*, const void*, const void*, const void*, const void*, const void*, void*, void*, uint32_t, uint32_t, uint32_t, uint32_t, void*);
extern "C" int pd_q8_0_moe_down_mma2_fs64(const void*, const void*, const void*, const void*, const void*, const void*, const void*, const void*, void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, void*);
extern "C" int pd_q8_0_moe_gate_up_mma2t_geglu(const void*, const void*, const void*, const void*, const void*, const void*, const void*, const void*, void*, void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, void*);
extern "C" int pd_q8_0_moe_down_mma2t(const void*, const void*, const void*, const void*, const void*, const void*, const void*, const void*, void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, void*);
extern "C" int pd_q8_0_moe_gate_up_g2_geglu(const void*, const void*, const void*, const void*, const void*, const void*, const void*, const void*, const void*, const void*, void*, void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, void*);
extern "C" int pd_moe_align_dual(const void*, void*, void*, void*, void*, void*, void*, void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, void*);
extern "C" int pd_quantize_e4m3_glu2_row_b16(const void*, void*, void*,
                                             unsigned int, unsigned int,
                                             unsigned int, void*);
extern "C" int pd_q4x_group_norm_1p(const void*, const void*, void*, void*, uint32_t, uint32_t, uint32_t, float, void*);
extern "C" int pd_q4x_hc_mix(const void*, const void*, void*, void*, uint32_t, uint32_t, uint32_t, void*);
extern "C" int pd_q4x_hc_combine(void*, const void*, const void*, uint32_t, uint32_t, uint32_t, void*);
extern "C" int pd_q4x_scale_silu(void*, uint32_t, float, void*);
extern "C" int pd_q4x_ple_gate(const void*, const void*, const void*, void*, uint32_t, uint32_t, uint32_t, void*);
extern "C" int pd_q4x_conv_dil(const void*, const void*, void*, uint32_t, uint32_t, uint32_t, uint32_t, void*);
extern "C" int pd_q4x_conv_dil_step(const void*, const void*, const void*, void*, uint32_t, uint32_t, uint32_t, void*);
extern "C" int pd_q4x_gdn_gated_norm(const void*, const void*, const void*, void*, void*, uint32_t, uint32_t, float, void*);
extern "C" int pd_q4x_gdn_split_widen(const void*, void*, void*, void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, void*);
extern "C" int pd_q4x_gdn_split_widen_tiled(const void*, void*, void*, void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, void*);
extern "C" int pd_moe_wave_plan(const void*, uint32_t, uint32_t, uint32_t, uint32_t, void*, void*, void*, void*);
extern "C" int pd_moe_cache_resolve_dev(const void*, const void*, uint32_t, void*, void*, void*, void*, void*, void*, void*, void*);
extern "C" int pd_moe_wave_mask(const void*, uint32_t, uint32_t, const void*, const void*, uint32_t, uint32_t, void*, void*, void*, void*, void*, void*);
extern "C" int pd_kquant_moe_gate_up_list(const void*, const void*, const void*, const void*, const void*, const void*, const void*, const void*, void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, const void*, const void*, void*);
extern "C" int pd_kquant_moe_gate_up_grp(const void*, const void*, const void*, const void*, const void*, const void*, const void*, const void*, const void*, const void*, void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, void*);
extern "C" int pd_kquant_moe_down_cols(const void*, const void*, const void*, const void*, const void*, const void*, const void*, void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, void*);
extern "C" int pd_q8_0_gemv_sk(const void*, const void*, const void*, const void*, void*, void*, void*, uint32_t, uint32_t, uint32_t, uint32_t, void*);
extern "C" int pd_kquant_moe_down_grp(const void*, const void*, const void*, const void*, const void*, const void*, const void*, const void*, const void*, void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, void*);
extern "C" int pd_moe_part_fold_at(const void*, void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, void*);
extern "C" int pd_kquant_moe_gate_up_tile(const void*, const void*, const void*, const void*, const void*, const void*, const void*, const void*, const void*, const void*, void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, void*);
extern "C" int pd_kquant_moe_down_tile(const void*, const void*, const void*, const void*, const void*, const void*, const void*, const void*, const void*, void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, void*);
extern "C" int pd_gated_delta_recurrent_pn(const void*, const void*, const void*, const void*, const void*, void*, void*, uint32_t, uint32_t, uint32_t, void*, void*);
extern "C" int pd_kquant_moe_down_list(const void*, const void*, const void*, const void*, const void*, const void*, const void*, void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, const void*, const void*, void*);
extern "C" int pd_q4x_add_gated_row(void*, const void*, const void*, uint32_t, uint32_t, void*);
extern "C" int pd_q4x_add_gated_row_s(void*, const void*, const void*, uint32_t, uint32_t, uint32_t, void*);
extern "C" int pd_moe_topk_batch_s(const void*, const void*, uint32_t, uint32_t, uint32_t, void*, void*, uint32_t, void*);
extern "C" int pd_gated_delta_recurrent_runs_pn(const void*, const void*, const void*, const void*, const void*, void*, void*, const void*, const void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, void*, void*);
extern "C" int pd_lowm_gemm(const void*, const void*, void*, uint32_t, uint32_t, uint32_t, void*);
extern "C" int pd_lowm_warmup(const void*, const void*, void*, void*);
extern "C" int pd_attn_decode_fmha_sp(const void*, const void*, const void*, const void*, void*, void*, const void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float, uint32_t, void*);
extern "C" int pd_bf16_gemv2_swiglu(const void*, const void*, const void*, void*, uint32_t, uint32_t, uint32_t, void*);
extern "C" int pd_convert_f32_bf16(const void*, void*, uint64_t, void*);
extern "C" int pd_bf16_gemv_up_hcmix(const void*, const void*, const void*, void*, void*, uint32_t, uint32_t, uint32_t, void*);
extern "C" int pd_conv_step_slots_split(void*, const void*, const void*, void*, void*, void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, void*);
extern "C" int pd_gated_delta_recurrent_slots_gn(const void*, const void*, const void*, const void*, const void*, const void*, void*, void*, const void*, const void*, void*, float, uint32_t, uint32_t, uint32_t, void*);
extern "C" int pd_bf16_gemv_mrow_f32(const void*, const void*, const void*, void*, void*, uint32_t, uint32_t, uint32_t, uint32_t, float, void*, uint32_t, void*);
extern "C" int pd_convert_bf16_f32(const void*, void*, uint64_t, void*);
extern "C" int pd_convert_bf16_f32_rows(const void*, void*, uint32_t, uint32_t, uint32_t, uint32_t, void*);
extern "C" int pd_swiglu_mir(void*, const void*, void*, uint32_t, void*);
extern "C" int pd_bf16_pad_rows(const void*, void*, uint32_t, uint32_t, uint32_t, void*);
extern "C" int pd_bf16_hc_perm_pad(const void*, void*, uint32_t, uint32_t, uint32_t, uint32_t, void*);
extern "C" int pd_q4x_moe_gu_swiglu(const void*, const void*, const void*, const void*, const void*, const void*, const void*, const void*, void*, uint32_t, uint32_t, uint32_t, uint32_t, void*);
extern "C" int pd_q4x_combine_norm(void*, const void*, const void*, const void*, void*, uint32_t, uint32_t, uint32_t, float, void*, void*);
extern "C" int pd_bf16_gemv_nk_f32(const void*, const void*, const void*, void*, uint32_t, uint32_t, void*);
extern "C" int pd_matvec_f32_sk(const void*, const void*, void*, void*, void*, uint32_t, uint32_t, uint32_t, void*);
extern "C" int pd_bf16_gemv_silu_f32(const void*, const void*, const void*, void*, void*, uint32_t, uint32_t, uint32_t, float, void*);
extern "C" int pd_bf16_gemv_nk_mr_f32(const void*, const void*, const void*, void*, uint32_t, uint32_t, uint32_t, void*);
extern "C" int pd_q4x_conv_dil_step_slots(const void*, const void*, const void*, void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, void*);

static const KernelTableV1 PD_KERNELS = {
    (uint32_t)sizeof(KernelTableV1),
    0,
    pd_mxfp4_dequant_f32,
    pd_q8_0_dequant_f32,
    pd_rmsnorm_f32,
    pd_rope_yarn_f32,
    pd_softmax_sink_f32,
    pd_swiglu_oai_f32,
    pd_add_inplace_f32,
    pd_scale_add_f32,
    pd_attn_decode_f32,
    pd_moe_topk,
    pd_mxfp4_gemv_indexed,
    pd_scale_add_dev,
    pd_q8_0_gemv,
    pd_attn_decode_partial,
    pd_attn_decode_combine,
    pd_mxfp4_moe_gate_up,
    pd_mxfp4_moe_down,
    pd_q8_0_gemm,
    pd_quantize_q8,
    pd_q8_0_gemv_dp4a,
    pd_mxfp4_gemv_indexed_dp4a,
    pd_mxfp4_moe_gate_up_dp4a,
    pd_mxfp4_moe_down_dp4a,
    pd_attn_decode_batch,
    pd_rmsnorm_batch,
    pd_rope_yarn_batch,
    pd_kv_append_batch,
    pd_mxfp4_moe_gate_up_batch,
    pd_mxfp4_moe_down_batch,
    pd_moe_topk_batch,
    pd_moe_slot_map,
    pd_mxfp4_moe_gate_up_grouped,
    pd_mxfp4_moe_down_grouped,
    pd_mxfp4_moe_gate_up_gemm,
    pd_moe_align,
    pd_mxfp4_moe_gate_up_gemm_sorted,
    pd_mxfp4_moe_down_gemm_sorted,
    pd_mxfp4_repack,
    pd_convert_f32_f16,
    pd_attn_decode_batch_partial,
    pd_attn_decode_batch_combine,
    pd_gated_delta_recurrent,
    pd_causal_conv1d_silu,
    pd_delta_gate,
    pd_gated_rmsnorm,
    pd_deltanet_split_gqa,
    pd_mrope,
    pd_mul_sigmoid,
    pd_swiglu,
    pd_split_qg,
    pd_conv_step,
    pd_q8_0_repack,
    pd_q8_0_gemv_repacked,
    pd_embed_gather,
    pd_argmax_advance,
    pd_q8_0_ffn_gate_up_swiglu,
    pd_deltanet_alpha_beta_gate,
    pd_q8_0_gemm_repacked,
    pd_embed_gather_batch,
    pd_q8_0_repacked_to_f16,
    pd_gated_delta_recurrent_snap,
    pd_q8_0_gemm_repacked_mt,
    pd_q8_0_gemm_mt_dp4a,
    pd_layernorm,
    pd_gelu,
    pd_bias_add,
    pd_mrope_vision,
    pd_vision_attn,
    pd_gated_delta_recurrent_slots,
    pd_conv_step_slots,
    pd_q8_0_gemv_dp4a_nc,
    pd_q8_0_gemm_mt_dp4a_wide,
    pd_deltanet_split_gqa_norm,
    pd_gated_delta_recurrent_v2,
    pd_argmax_rows,
    pd_conv_ext_build_slots,
    pd_conv_chunk_ext,
    pd_state_restore_slots,
    pd_conv_commit_slots,
    pd_bump_rows_u32,
    pd_q8_0_gemm_mma,
    pd_quantize_q8_mmq,
    pd_q8_0_gemm_mmq,
    pd_attn_prefill,
    pd_attn_prefill_f16,
    pd_quantize_q8_mmq_swiglu,
    pd_add_rmsnorm_quant_mmq,
    pd_gated_delta_chunked,
    pd_mxfp4_moe_gate_up_mmq,
    pd_mxfp4_moe_down_mmq,
    pd_batched_copy,
    pd_moe_slot_combine,
    pd_mxfp4_moe_gate_up_dp4a_b,
    pd_mxfp4_moe_down_dp4a_b,
    pd_matvec_f32_batch,
    pd_q8_0_gemv_dp4a_nc_b,
    pd_quantize_e4m3,
    // block-scale entries are NULL unless the pack was built with the
    // sm_120a gencode (+ -DPD_BS_HOST=1): the engine's capability probe is
    // `entry.is_some()`, and a non-null pointer to a launcher that returns
    // cudaErrorNotSupported passes that probe and crashes every gpt-oss
    // serving batch at b > 4 (Launch(801), found on the Q1 box whose
    // bootstrap built plain sm_120). Null here = the honest "not supported",
    // and the engine keeps the s8 mmq path.
#ifdef PD_BS_HOST
    pd_mxfp4_moe_gate_up_bs,
    pd_mxfp4_moe_down_bs,
#else
    NULL,
    NULL,
#endif
    pd_q8_0_gemm_mma_ks,
    pd_q8_0_moe_gate_up_dp4a,
    pd_q8_0_moe_down_dp4a,
    pd_shexp_gate_add,
    pd_q8_0_moe_gate_up_sorted,
    pd_q8_0_moe_down_sorted,
    pd_q8_0_moe_gate_up_mma,
    pd_q8_0_moe_down_mma,
    pd_add_rmsnorm_batch,
    pd_q8_0_gemm_mmq_hi,
    pd_q8_0_gemm_mmq_pipe,
    pd_attn_prefill_batch,
    // block-scale dense pair: NULL unless built for sm_120a (same honest-
    // capability rule as the moe_*_bs entries above)
#ifdef PD_BS_HOST
    pd_q8_0_to_mxfp4,
    pd_mxfp4_gemm_bs,
#else
    NULL,
    NULL,
#endif
    pd_quantize_e4m3_swiglu,
#ifdef PD_BS_HOST
    pd_q8_0_to_fp4p,
#else
    NULL,
#endif
    pd_quantize_e2m1,
    pd_quantize_e2m1_swiglu,
#ifdef PD_BS_HOST
    pd_mxfp4_gemm_f4,
    pd_q8_0_to_nvf4,
#else
    NULL,
    NULL,
#endif
    pd_quantize_nvf4,
    pd_quantize_nvf4_swiglu,
#ifdef PD_BS_HOST
    pd_mxfp4_gemm_nv4,
    pd_q8_0_to_nvf4_rot,
#else
    NULL,
    NULL,
#endif
    pd_quantize_nvf4_rot,
#ifdef PD_BS_HOST
    pd_mxfp4_gemm_bs_gu,
#else
    NULL,
#endif
    pd_col_absmax,
    pd_q8_0_col_absmax,
    pd_quantize_nvf4_smooth,
#ifdef PD_BS_HOST
    pd_q8_0_to_nvf4_smooth,
#else
    NULL,
#endif
    pd_quantize_nvf4_swiglu_smooth,
    pd_quantize_e4m3_smooth,
    pd_quantize_e4m3_swiglu_smooth,
#ifdef PD_BS_HOST
    pd_q8_0_to_mxfp4_smooth,
#else
    NULL,
#endif
    pd_attn_prefill_batch_f16,
    pd_q_norm_rope,
    pd_k_norm_rope_append,
#ifdef PD_BS_HOST
    pd_q8_0_to_f8w,
    pd_f8_gemm_w8,
#else
    NULL,
    NULL,
#endif
    pd_qkv_norm_rope_append,
    pd_q8_0_gemm_mmq_pipe64,
    pd_q8_0_gemm_mmq_pipe_sk,
    pd_sample_rows,
    pd_q8_0_gemm_mt_dp4a_b,
    pd_q8_0_gemm_mma_ks_b,
    pd_q8_0_gemm_mmq_b,
    pd_mxfp4_moe_down_bs_res,
    pd_moe_align_bm,
    pd_mxfp4_moe_gate_up_bs64,
    pd_mxfp4_moe_down_bs64,
    pd_qkv_rope_append_batch,
    pd_pipe_advance,
    pd_rmsnorm_quant_q8_batch,
    pd_add_rmsnorm_quant_e4m3_batch,
    pd_q8_0_gemm_mma_ks_qkv_rope,
    pd_moe_combine_rmsnorm_quant_q8,
    pd_mxfp4_gu_interleave,
    pd_attn_decode_batch_paged,
    pd_kv_append_batch_paged,
    pd_attn_decode_batch_partial_paged,
    pd_attn_prefill_paged,
    pd_attn_prefill_f16_paged,
    pd_qkv_rope_append_batch_paged,
    pd_q8_0_gemm_mma_ks_qkv_rope_paged,
    pd_q8_0_gemm_repacked_x2,
    pd_delta_gate_ab,
    pd_moe_slot_combine_bf16,
    pd_geglu,
    pd_rope_factors_batch,
    pd_rope2d_neox,
    pd_softcap,
    pd_embed_gather_q8,
    pd_add_scale,
    pd_geglu_pair,
    pd_gemma_qkv_nra,
    pd_rmsnorm_add_scale,
    pd_quantize_q8_mmq_geglu,
    pd_attn_spec_batch_paged,
    pd_gemma_qkv_nra2,
    pd_f8_gemv,
    pd_f8_gemv_batch,
    pd_f8_gemm_mma_ks,
    pd_quantize_e4m3_geglu,
    // per-ROW e4m3 prefill class: launchers self-gate on PD_BS_HOST; the
    // runtime resolution above NULLs them below cc 8.9 with the f8w8 family
#ifdef PD_BS_HOST
    pd_q8_0_to_f8row,
#else
    NULL,
#endif
    pd_quantize_e4m3_row,
#ifdef PD_BS_HOST
    pd_f8row_gemm,
#else
    NULL,
#endif
    pd_f8_repack_tiles,
    pd_f8t_gemm,
    pd_quantize_e4m3_geglu2,
    pd_gemma_qkv_nra2s,
    pd_quantize_e4m3_geglu2_row,
    pd_rmsnorm_e4m3_batch,
    pd_fp4_gemv,
    pd_fp4_gemm_mma_ks,
    pd_rmsnorm_e4m3,
    pd_rmsnorm_e4m3_row,
    pd_addnorm_e4m3_row,
    pd_attn_combine_e4m3_row,
    pd_qkv_norm_rope_batch,
    pd_u32_addk,
    pd_f8t_gemm2,
    pd_addnorm_e4m3_nz,
    pd_quantize_e4m3_geglu2_nz,
    pd_kquant_dequant,
    pd_kquant_repack,
    pd_kquant_gemv,
    pd_kquant_gather,
    pd_kquant_dequant_rp,
    pd_gemm_f32,
    pd_mmq_sums,
    pd_kquant_gemm_w4a8,
    pd_q8_sums_strided,
    pd_kquant_gemm_dp4a,
    pd_kquant_gemm_mma_ks,
    pd_kquant_moe_gate_up,
    pd_kquant_moe_down,
    pd_kquant_moe_gate_up_mma,
    pd_kquant_moe_down_mma,
    pd_kquant_gemv_w4a8,
    pd_quantize_q8_sums,
    pd_q8_0_moe_gate_up_dp4a_geglu,
    pd_moe_scale_w,
    pd_q8_0_moe_gate_up_mma_geglu,
    pd_q8_0_to_f8w_pad,
    pd_moe_gather_e4m3,
    pd_quantize_e4m3_geglu2_pad,
    pd_f8bs_moe_gemm_gu,
    pd_f8bs_moe_gemm_dn,
    pd_q8_0_moe_gu_dec2_geglu,
    pd_q8_0_moe_dn_dec2,
    pd_moe_head,
    pd_moe_topk_scaled,
    pd_moe_tail,
    pd_kquant_gemv_w4a8_nc,
    pd_matvec_ab_gate,
#ifdef PD_BS_HOST
    // dec3 launchers are real only in PD_BS_HOST builds (the pack then
    // carries sm_90+ SASS); elsewhere the entries are honest NULLs
    pd_q8_0_moe_gu_dec3_geglu,
    pd_q8_0_moe_dn_dec3,
#else
    NULL,
    NULL,
#endif
    pd_moe_combine_dec3,
    //  decode-band f8 trio: the tc5 launchers stub to NotSupported off
    // PD_BS_HOST/PD_TC5_HOST and are per-device NULLed off cc 10 below
    pd_f8bs_moe_gemm_gu_d32,
    pd_f8bs_moe_gemm_dn_d32,
    pd_quantize_e4m3_geglu2_pad_b,
    //  uniq-routing diagnostic - plain CUDA, every arch
    pd_moe_uniq_hist,
    // fusion program: merged gate_up plane epilogue - plain CUDA, every arch
    pd_swiglu_fused,
    // fusion program: merged-plane split epilogue - plain CUDA, every arch
    pd_row_slice,
    // native-fp8 decode lane (e4m3 mma, sm_89+; arch-gated below)
    pd_f8d_gemm_mma_ks,
    // bf16-epilogue prefill pair (tma-route; self-gates via NotSupported)
    pd_f8_gemm_w8_o16,
    pd_quantize_e4m3_swiglu_b16,
    pd_add_rmsnorm_quant_mmq_b16,
    pd_add_inplace_b16,
    // fp8-native ingestion (bf16 -> f8w; PD_BS_HOST route)
    pd_bf16_to_f8w,
    pd_bf16_to_f8r,
    pd_f8r_gemm_mma_ks,
    pd_swiglu_fused_e4m3,
    pd_add_rmsnorm_e4m3_xn,
    // tile-linear f8 weight lane (gemm/f8_lin.cuh)
    pd_f8w_repack_lin,
    pd_f8_gemm_lin,
    pd_f8_gemm_lin_kt,
    pd_add_rmsnorm_e4m3_xn_b16,
    pd_gated_rmsnorm_e4m3,
    // official-FP8 byte passthrough (block-scale lin lane)
    pd_f8w_repack_lin_bs,
    pd_f8_gemm_lin_bs,
    pd_quantize_e4m3_swiglu_b16_gu,
    pd_causal_conv1d_silu_qkv,
    pd_causal_conv1d_silu_qkv_b16,
    pd_gated_delta_chunked_vb16,
    pd_addnorm_e4m3_b32,
    pd_attn_spec_batch_fin,
    // gu epilogue-fusion trio (interleaved lin plane; gemm/f8_lin.cuh)
    pd_f8w_repack_lin_gui,
    pd_quantize_e4m3_geglu2i,
    pd_f8_gemm_lin_gu,
    // spec-verify LCO (in-kernel last-CTA-out combine on krs)
    pd_attn_spec_lco_paged,
    // per-channel gu GEMM (kt4a scale-free mainloop)
    pd_f8_gemm_lin_gu_pc,
    // pc lin GEMM for the qkv/wo classes (kt4 scale-free twin)
    pd_f8_gemm_w8_pc,
    // down twin (kt4d: weights-pc, activations per-32)
    pd_f8_gemm_w8_pcd,
    // async spec round token assembly
    pd_spec_toks,
    // device-side spec accept (rung B1)
    pd_spec_accept,
    // accept + next-round prep, and the h gather (rung B2)
    pd_spec_prep,
    pd_spec_hgather,
    // fused K/V norm+rope+append (kv-epilogue fold)
    pd_kv_nra_rows,
    // canonical spec rejection sampling: sampled draft chain +
    // full-q verify resolve
    pd_draft_rs,
    pd_spec_rs_resolve,
    // batched drafter stitches (224 DtoD memcpys/round -> 4 launches)
    pd_spec_xh_stitch,
    pd_hrow_gather,
    // rowwise (strip-free) pc plane lane (gemm/f8_lin.cuh)
    pd_f8w_repack_lin_bs_gui,
    pd_f8_gemm_lin_r,
    pd_f8_gemm_lin_kt_r,
    pd_f8_gemm_lin_gu_r,
    pd_f8_gemm_lin_gu_pc_r,
    pd_f8_gemm_w8_pc_r,
    pd_f8_gemm_w8_pcd_r,
    // fused qkv single-launch on the rowwise plane
    pd_f8_gemm_w8_pc_qkv_r,
    // chunk-band 16-bit streams: o16 qkv GEMM + the four bf16-in
    // consumer twins (291..295)
    pd_f8_gemm_w8_pc_qkv_r2,
    pd_qkv_norm_rope_batch2,
    pd_kv_nra_rows2,
    pd_addnorm_e4m3_row2,
    pd_rmsnorm_add_scale2,
    // attention streams: f16 pf_qn/pf_attn mixed-tick route
    // (296..303)
    pd_qkv_norm_rope_batch3,
    pd_attn_prefill_f16_paged2,
    pd_attn_spec_batch_paged2,
    pd_attn_decode_batch_paged2,
    pd_attn_decode_batch_partial_paged2,
    pd_attn_decode_batch_combine2,
    pd_quantize_e4m3_i16,
    pd_quantize_e4m3_row_i16,
    // Laguna family (slots 304..306): per-head softplus attention gate + sigmoid
    // MoE router - plain CUDA, every arch
    pd_mul_softplus_head,
    pd_moe_topk_sigmoid_batch,
    // Laguna decode-tick epilogue fold (norm+rope+append, 6 launches -> 1)
    pd_lag_qk_nra_rows,
    // Granite family (slot 307): standalone x *= s. Granite's
    // embedding/logit multipliers; general enough for minicpm/grok, which
    // carry the same scalar-multiplier shape.
    pd_scale_f32,
    // Granite (slot 308): NORM-convention rope. Granite is
    // llama.cpp's ROPE_TYPE_NORM while every other family here is NEOX; the
    // kernel is templated on the convention so the NEOX path is unchanged.
    pd_rope_yarn_batch_norm,
    // Qwen3.5-family fused-plane prefill consumer (309: split_qg + 2x norm +
    // 2x mrope + 2x append -> 1 launch off the one-GEMM qkv plane)
    pd_q36_qkg_nra_rows,
    // q36 DN  (310): fused in_qkv|gate prefill GEMM, two-buffer kt3
    // epilogue (gemm/f8_lin.cuh)
    pd_f8_gemm_lin_kt_split,
    // granite-vision Q-Former quartet (311-314, vision.cuh): batched
    // cross/self attention, gather-with-fan-in (windowing + both
    // downsamplers), exact-erf GELU, broadcast row add
    pd_vision_attn_x,
    pd_gather_rows_avg,
    pd_gelu_erf,
    pd_add_rows_bcast,
    // 315: pipelined k-quant W4A8 GEMM (cp.async, >64-batch rung)
    pd_kquant_gemm_w4a8_pipe,
    // 316: its genuinely-double-buffered sibling (2-deep raw ring, half-width
    // tile_x, __launch_bounds__(256,1))
    pd_kquant_gemm_w4a8_pipe2,
    // 317: multi-segment q8_0_gemv_repacked (decode QKV / gate|up merge)
    pd_q8_0_gemv_repacked_multi,
    // 318: fused NORM-rope(q,k) + paged K/V append (granite decode band)
    pd_rope_norm_qk_append_paged,
    // 319: multi-segment W4A8 k-quant decode GEMV (granite-30b QKV / gate|up)
    pd_kquant_gemv_w4a8_multi,
    // 320: multi-segment nc GEMV (r=2..4 batched-decode q|k|v|g / gate|up)
    pd_q8_0_gemv_dp4a_nc_multi,
    // 321: packed multi-span gated delta recurrence (decode len-1 items +
    // short prefill span walks, one launch via (row0,len,slot) triples)
    pd_gated_delta_recurrent_v2_packed,
    // 322: pf7 varlen packed prefill attention (one launch/layer over all
    // eligible spans via stride-4 tile items; fp8 hd256 G 4/6/8)
    pd_attn_prefill_f16_paged_vl,
    // 323: varlen chunked-GDN (one stage1+walk pair over all eligible spans
    // of the tick; RS route only, chunk pairs + span quads)
    pd_gated_delta_chunked_rs_vl,
    // 324: fused-GLU W4A8 decode GEMV (gate+up+SwiGLU one launch, bit-exact
    // vs the multi<4,128>+swiglu split path)
    pd_kquant_gemv_w4a8_glu,
    // 325: qwen twin of addnorm_e4m3_row -- PLAIN residual add (no post-norm,
    // no stream scale) + pre-norm + row-e4m3, one launch. Bit-identical to
    // pd_add_rmsnorm_batch + pd_quantize_e4m3_row.
    pd_add_rmsnorm_e4m3_row,
    // 326: multi-slice split of a fused GEMM landing (240 launches -> 64)
    pd_row_slice4,
    // 327: swiglu + row-e4m3 on the FFN down input (the widest row)
    pd_swiglu_e4m3_row,
    // 328-334: the whisper decode lane  - flash-decoding
    // cross/self attention over f16 K/V slot planes, plus the fused
    // epilogues that collapse the 32-launch-per-layer decode step.
    pd_whisper_dec_attn,
    pd_whisper_embed_pos,
    pd_whisper_qkv_split,
    pd_whisper_kv_store,
    pd_whisper_ln_f16,
    pd_whisper_res_ln_f16,
    pd_whisper_bias_gelu_f16,
    // 335-341: the granite-speech conformer tower  - the macaron
    // FFN / GLU / centered-depthwise-conv / Shaw-RPE-attention pieces that
    // have no counterpart elsewhere in the pack, plus the CTC head and the
    // two scaled residual seams.
    pd_gs_bias_silu_f16,
    pd_gs_bias_glu,
    pd_gs_dwconv_bn_silu_f16,
    pd_gs_conf_attn,
    pd_gs_bias_softmax_f16,
    pd_gs_res_ln_f16,
    pd_gs_post_ln_f16,
    // 342: gated_rmsnorm_e4m3_row (DN out_proj decode row fuse)
    pd_gated_rmsnorm_e4m3_row,
    // 343: the confidence readout - greedy pick, the RUNNER-UP, and
    // {log p(top1), p(probe), log p(top2), Renyi-2 entropy} out of a single
    // log-sum-exp pass (per-token confidence + no_speech_prob,
    // widened for the earlier margin).
    pd_argmax_top2_rows,
    // 344: whisper's own timestamp-token grammar, applied to the logits before
    // the pick - without it a fine-tune greedily opts out of timestamps
    // (measured: KB-Whisper picks `<|notimestamps|>` at p=0.794).
    pd_whisper_ts_rules,
    // 345: DN split with the delta gate folded into the ab parts
    pd_row_slice2_gate,
    // 346-347: the flat-scale (per-output-ROW) e4m3 expert lane (
    // change A) - a per-32 e4m3 activation quantizer and the gate_up GEMM
    // whose k loop carries no weight scale at all. The weight plane reuses
    // the dense lane's q8_0_to_f8row converter (slot 216).
    pd_quantize_e4m3_b32f,
    pd_f8row_moe_gate_up_mma_geglu,
    // 348-349: the down half of the same lane, plus the gate_up variant whose
    // epilogue hands it e4m3 per-32 instead of int8 per-32.
    pd_f8row_moe_gate_up_mma_geglu_f8,
    pd_f8row_moe_down_mma,
    // 350: conv-window VL store - span geometry from device contents (
    // chunk-tick graph capture; also a 576->48 launch fold in the batched pass)
    pd_conv_win_store_vl,
    // 351-354: bf16 dense weight planes - the per-tensor quant-dispatch lane
    // for mixed UD files
    pd_bf16_gemv_f32,
    pd_bf16_gemm_f32,
    pd_bf16_dequant_f32,
    pd_embed_gather_bf16,
    // 355-363: SiLU twins of the gated-FFN carrier set. Same
    // kernels on pd_glu_act<PD_ACT_SILU>; the GELU entries above are the
    // untouched incumbents.
    pd_swiglu_pair,
    pd_quantize_e4m3_swiglu2,
    pd_quantize_e4m3_swiglu2i,
    pd_quantize_e4m3_swiglu2_row,
    pd_quantize_e4m3_swiglu2_nz,
    pd_f8_gemm_lin_gu_silu,
    pd_f8_gemm_lin_gu_r_silu,
    pd_f8_gemm_lin_gu_pc_silu,
    pd_f8_gemm_lin_gu_pc_r_silu,
    // 364-365: ROPE_TYPE_NORM rope twins (- muse-glimmer ropes
    // NORM where gemma4 ropes NEOX)
    pd_rope_factors_batch_norm,
    pd_qkv_norm_rope_batch4,
    pd_qkv_norm_rope_batch5,
    pd_kv_nra_rows3,
    pd_gemma_qkv_nra3,
    // 369: cross-attention probabilities for the alignment heads - whisper's
    // word-level timing read-out. A pure append.
    pd_whisper_xattn_probs,
    // 370-371: muse-glimmer's vision tower  - the NORM-paired 2D
    // rope and the channel-outer pixel-shuffle merge
    pd_rope2d,
    pd_pixel_shuffle_rows,
    // 374:  dim-major twin V pool sync (v9q VD arm)
    pd_vdim_sync,
    // 375: vdim pool registration (engine -> launcher side channel)
    pd_vdim_register,
    // 376: batched-runs prefill attention arm
    pd_pf_runs_register,
    pd_quantize_e4m3_glu2_row_b16,
    // 379: bf16 -> e4m3 + f32 row scale  - the converter a bf16
    // lm_head needs to build an F8RowPlane and ride the f8t tile route
    pd_bf16_to_f8row,
    // 380: SAM ViTDet attention with the decomposed rel-pos bias
    pd_sam_attn,
    // 381: DeepSeek-greedy router epilogue - full-softmax topk weights
    pd_moe_topk_softmax_all,
    // 382: fused single-pass GQA-16 decode attention (opt-in arm)
    pd_attn_decode_fused_gqa16,
    // 383: in-house f16xf16->f32 tensor-core dense GEMM (PADDOCK_INHOUSE_F16)
    pd_f16_gemm,
    // 384: ring twin of 318 - rope by true position, append at the R-SWA
    // write slot, NEOX/NORM by arg (deepseek-ocr decode fold)
    pd_rope_qk_append_paged_ring,
    // 385-386: dp4a-class decode glue folds  - add+rmsnorm+q8
    // quantize, and swiglu+q8 quantize
    pd_add_rmsnorm_quant_q8_batch,
    pd_swiglu_quant_q8,
    // 387: OCR tower patch stem - u8 RGB views to normalized f16 im2row
    // rows in one gather
    pd_ocr_patches_u8,
    // 388: encoder fused-qkv split, biases folded (encoder fusion)
    pd_whisper_enc_qkv_split,
    // 389: layer-batched cross-K/V store off one fused landing
    pd_whisper_kv_store_batch,
    // 390: decode-band multi-row bf16 GEMV
    pd_bf16_gemv_mr_f32,
    // 391: bf16 tensor-core prefill GEMM, bf16-cast activations
    pd_bf16_gemm_mma,
    // 392-396: PaddleOCR-VL tower elementwise fusions
    pd_layernorm_f16,
    pd_gelu_bias_f16,
    pd_gelu_erf_bias_f16,
    pd_add_bias_res,
    pd_mrope_vision_bias,
    // 397-398: modelopt NVFP4 checkpoint consumers
    pd_nvf4_dequant,
    pd_nvf4_gemv,
    // 399-402: nemotron_h_moe mamba-2 lane  - arch-generic SIMT
    pd_f8r_gemv,
    pd_mamba_conv_step,
    pd_mamba2_scan_seq,
    pd_mamba_rmsnorm_gated_g,
    // 403-404: NVFP4 MoE expert consumers  - cc-gated with 397-398
    pd_nvf4_moe_up_relu2,
    pd_nvf4_moe_down_acc,
    // 405: cross-K/V store off an audio-major batched landing - slot 389
    // with rows_per_slot (batched admission)
    pd_whisper_kv_store_slots,
    // 406: bulk mamba-2 conv span (prefill) - arch-generic SIMT
    pd_mamba_conv_seq,
    // 407-408: sorted-tile NVFP4 MoE expert GEMMs  -
    // cc-gated with 397-398
    pd_nvf4_moe_up_relu2_bs,
    pd_nvf4_moe_down_bs,
    // 409-410: decode multi-task NVFP4 MoE expert GEMVs (decode
    // rung) - cc-gated with 397-398
    pd_nvf4_moe_up_relu2_mt,
    pd_nvf4_moe_down_part,
    // 411: capture-time f16 mmaf election gate (overlap routing;
    // renumbered from 409 on rebase - both sides appended)
    pd_f16_mmaf_set,
    // 412-414: batched decode steps over slot arenas (
    // stage A) - the continuous-batching tick's per-slot state advance
    pd_mamba_conv_step_batch,
    pd_mamba2_scan_step_batch,
    pd_nvf4_gemv_batch,
    // 415-416: Q8_0 up-only relu^2 expert kernels (nemotron GGUF
    // lane) - same dp4a class as the gate_up pair, arch-generic
    pd_q8_0_moe_up_relu2_dp4a,
    pd_q8_0_moe_up_relu2_sorted,
    // 417-418: spec verify core  - per-row-snapshot scan +
    // strided-rows copy, arch-generic SIMT
    pd_mamba2_scan_seq_snap,
    pd_copy_rows_strided,
    // 419: multi-row W4A16 nvf4 GEMM  - cc-gated with 414
    pd_nvf4_gemm_mr,
    // 420: topk router + appended shared pseudo-expert picks (
    // shared fold-in) - arch-generic SIMT
    pd_moe_topk_sigmoid_batch_sh,
    // 421: packed-bf16 q/k/v read twin of gemma_qkv_nra3 (spec
    // verify b16-D election) - arch-generic SIMT
    pd_gemma_qkv_nra3_b16,
    // 422: tensor-core NVFP4 GEMM  - cc-gated with 419
    pd_nvf4_gemm_tc,
    // 423: fin-e4 attention (in-kernel wo-in row quantize)
    pd_attn_spec_batch_fin_e4,
    // 424: fused q|k|v decode-band bf16 GEMM, segmented store (
    // thin-k/v rung) - same PD_BF16MMA_OK in-body guard as bf16_gemm_mma
    pd_bf16_qkv_gemm_mma,
    // 425: fin-e4s attention (static-scale e4m3 fin store)
    pd_attn_spec_batch_fin_e4s,
    // 426: checkpoint-plane W4A4 GEMM  - cc-gated with 414
    pd_nvf4_gemm_f4,
    // 427: v2 (async scales, one barrier, st ring) - cc-gated with 414
    pd_nvf4_gemm_f4b,
    // 428: split-K twin + reduce - cc-gated with 414
    pd_nvf4_gemm_f4s,
    // 429: KC=256 arm (plain or split via sk) - cc-gated with 414
    pd_nvf4_gemm_f4c,
    // 430: TMA + mbarrier ring (prefill band) - cc-gated with 414
    pd_nvf4_gemm_f4t,
    // 431: tcgen05 decode attention, final-output contract
    pd_attn_decode_tc5_paged,
    // 434: device top-K prefilter (host-head sampling; deltanet/
    // stage2_sample.cuh)
    pd_topk_rows,
    // 435: full-device truncation sampling (mode 5 - head build + the
    // host nucleus pipeline on device; zero-host rows)
    pd_sample_rows_t,
    // 436: truncation stage (c) general truncation sampling (deltanet/
    // stage2_sample.cuh)
    pd_sample_rows_p,
    // 437: fused decode recurrence (split+l2norm folded into the v2
    // body; gemm/int8_mma.cuh)
    pd_gated_delta_recurrent_v2f,
    // 438: strided conv step (deltanet/stage2_sample.cuh)
    pd_conv_step_slots_s,
    // 439: strided gated rmsnorm (deltanet/core.cuh)
    pd_gated_rmsnorm_s,
    // 440: gate-inline v2f (gemm/int8_mma.cuh)
    pd_gated_delta_recurrent_v2f_g,
    // 441: VL conv+silu+qkv (deltanet/core.cuh)
    pd_causal_conv1d_silu_qkv_vl,
    // 442:  glue rung, add+rmsnorm+nvf4 quant (quant/nvf4.cuh)
    pd_add_rmsnorm_quant_nvf4_batch,
    // 443-445:  scan rung, f16 SSM-state class (mamba/core.cuh)
    pd_mamba2_scan_seq_f16,
    pd_mamba2_scan_seq_snap_f16,
    pd_mamba2_scan_step_batch_f16,
    // 446-447: f16 state <-> f32 checkpoint blob (mamba/core.cuh)
    pd_ssm_state_widen,
    pd_ssm_state_narrow,
    // 448-449: QKC compact-bf16 q/k pair (deltanet/core.cuh + stage2_sample
    // .cuh) - conv emits Hg-compact bf16 q/k, the vl chunked-GDN entry reads
    // them; one engine latch drives both. Bit-identical to the expanded pair.
    pd_causal_conv1d_silu_qkv_vl_qkc,
    pd_gated_delta_chunked_rs_vl_qkc,
    // 450-451: the nemotron decode-band relu^2 pair.
    pd_q8_0_moe_up_relu2_dec2,
    pd_quantize_q8_relu2,
    // 452-454: tile-major NVFP4 plane twins (lm_head repack
    // rung) - cc-gated with 414/419/422
    pd_nvf4_gemv_batch_tm,
    pd_nvf4_gemm_mr_tm,
    pd_nvf4_gemm_tc_tm,
    // 455-457: fragment-layout NVFP4 plane twins (fragment
    // rung) - cc-gated with the _tm trio
    pd_nvf4_gemv_batch_tf,
    pd_nvf4_gemm_mr_tf,
    pd_nvf4_gemm_tc_tf,
    // 458: Q16xKv128 tensor-core decode attention for the muse hd128/G16
    // geometry (attn/fmha16.cuh). FINAL-output like 382 - the caller skips
    // the combine. 5.35x the shipped vec8 splits=2 + combine pair at
    // B=32/ctx256, and it wins at every rung, so no row gate.
    pd_attn_decode_fmha16,
    // 459: DFlash2 grouped dynamic convolution (dflash.cuh) - the depthwise
    // token-axis conv that wraps each drafter sublayer, masked to the runtime
    // block so a tap never reaches into another slot's rows.
    pd_dflash_conv,
    // 460-461: DFlash2 candidate selector - id unpack for the codebook gather,
    // then the greedy edge walk that turns per-row argmax into a chosen PATH.
    pd_dflash_cand_ids,
    pd_dflash_select,
    // 462: spec-verify hold twin of v2 (no snapshots, no state writeback) -
    // arch-generic SIMT
    pd_gated_delta_verify_hold,
    // 463: commit-time accepted-prefix recompute (replaces the snapshot
    // rollback on the qwen35 spec path) - arch-generic SIMT
    pd_gated_delta_commit_walk,
    // 464: dflash async-round pick copy into the chain layout - arch-generic
    pd_dflash_chain_picks,
    // 469: dflash conditioning fold (norm+rope+paged store over written
    // rows, one launch per drafter layer) - arch-generic
    pd_dflash_cond_append,
    // 470/471: DFlash2 sampled selector walk + K-candidate RS resolve
    // (rung G) - arch-generic
    pd_dflash_select_rs,
    pd_dflash_rs_resolve,
    // 472-477: NVFP4 MoE consumers over the TILED expert-plane layout
    // (moe/nvf4_st.cuh) - cc12-gated below as a set
    pd_nvf4_moe_up_relu2_st,
    pd_nvf4_moe_down_st,
    pd_nvf4_moe_up_relu2_stw,
    pd_nvf4_moe_down_stw,
    pd_nvf4_moe_up_relu2_mtt,
    pd_nvf4_moe_down_part_tt,
    pd_kquant_q40,
    // 479/480: KV tier extent gather/scatter (tier/xfer.cuh) - arch-generic
    pd_kv_gather_blocks,
    pd_kv_scatter_blocks,
    // slot 481: b=1 GEMV over the lin boxes (non-KV-overhead R2.2)
    pd_f8lin_gemv,
    // slot 482: granite's f32/Q8 residual fusion
    pd_add_rmsnorm_q8_xn,
    // slots 483/484: v2 ring twins of the sorted q8 MMA pair (g26a4b MoE
    // decode band act)
    pd_q8_0_moe_gate_up_mma2_geglu,
    pd_q8_0_moe_down_mma2,
    // slot 485: write-out slot combine (kills the per-tick moe_xn memset)
    pd_moe_slot_combine_init,
    // slot 486: K-split decode router matvec (scratch-fed, deterministic)
    pd_matvec_f32_ks,
    // slot 487: head+router+topk single-launch fusion (bit-identical chain)
    pd_moe_head_router,
    // slot 488: v5 gate_up - the small-CTA geometry port (BM16 view x 64-row
    // both-mat slices, 128 thr)
    pd_q8_0_moe_gate_up_mma3_geglu,
    // slots 489/490: prefill dn hybrid - bm128->bm32 pair map + q8 GEGLU
    // remap quantize (f8s-gu output feeds the v2 down)
    pd_moe_pair_map,
    pd_quantize_q8_geglu_remap,
    // slot 491: tail+combine fold (kills the standalone combine launch and
    // the moe_xn round trip; bitwise the combine_init -> moe_tail chain)
    pd_moe_tail_combine,
    // slot 492: merged q|k|v NVFP4 GEMV - one grid over three planes that
    // share x. granite's k/v (out 1024) are 128 CTAs on 188 SMs, so they cost
    // a full-size launch for a quarter of the bytes; the Q8 twin already
    // measured 26.5 -> 20.5 us for the same merge on this die.
    pd_nvf4_gemv_multi,
    pd_add_rmsnorm_scaled_batch,
    // hibatch lane M1: hb head+router+topk - 8-token blocks, bf16 smem rows,
    // rw plane read once per 8 tokens (headr's per-token re-read fixed).
    pd_moe_head_router_hb,
    // P1-2 (hibatch path 1): per-128 activation-scale pair - head producer
    // twin + mma2 ILV consumer with the reassociated group fold.
    pd_moe_head_xg,
    pd_q8_0_moe_gate_up_mma2g_geglu,
    // P1-1: bf16 partials pair (down bf16 store + tail bf16 read, f32 sums)
    pd_q8_0_moe_down_mma2_pbf16,
    pd_moe_tail_combine_bf16,
    // B3-1: cooperative router stage (matvec+topk, grid.sync)
    pd_moe_router_stage,
    // P1 dn64: per-64 Y-scale pair (mma2g y64 producer + fs64 down consumer,
    // down takes a trailing pbf16 flag). nullptr on pack configs the twins
    // don't support, so the engine lane gates itself off via has_.
#if PD_QMMA2_LDM && PD_QMMA2_ILV
    pd_q8_0_moe_gate_up_mma2g_y64_geglu,
#else
    nullptr,
#endif
#if PD_QMMA2_LDM && (PD_QMMA2_DN_NT == 1) && !PD_QMMA2_YSYNC
    pd_q8_0_moe_down_mma2_fs64,
#else
    nullptr,
#endif
    // v3t TMA twins (502/503): real only on the shipped v2 config; the
    // resolver additionally NULLs them below cc 9.
#if PD_QMMA2_ILV && PD_QMMA2_LDM && !PD_QMMA2_YSYNC && (defined(PD_BS_HOST) || defined(PD_TC5_HOST))
    pd_q8_0_moe_gate_up_mma2t_geglu,
    pd_q8_0_moe_down_mma2t,
#else
    nullptr,
    nullptr,
#endif
    // g2 (504): token-major gate_up
#if PD_QMMA2_ILV && PD_QMMA2_LDM && !PD_QMMA2_YSYNC && (defined(PD_BS_HOST) || defined(PD_TC5_HOST))
    pd_q8_0_moe_gate_up_g2_geglu,
#else
    nullptr,
#endif
    // 505: dual-output align (plain CUDA, every arch)
    pd_moe_align_dual,
    // 506-514: qwen4_exp (Qwen3.8-Flash-Next) new math - plain CUDA, every arch
    pd_q4x_group_norm_1p,
    pd_q4x_hc_mix,
    pd_q4x_hc_combine,
    pd_q4x_scale_silu,
    pd_q4x_ple_gate,
    pd_q4x_conv_dil,
    pd_q4x_conv_dil_step,
    pd_q4x_gdn_gated_norm,
    pd_q4x_gdn_split_widen,
    // 515: shared-expert scalar-gate fold (plain CUDA, every arch)
    pd_q4x_add_gated_row,
    // 516: NVFP4 MoE gate+up swiglu - cc-gated like the other nvf4 experts
#ifdef PD_BS_HOST
    pd_q4x_moe_gu_swiglu,
#else
    nullptr,
#endif
    // 517: fused combine+norm (plain CUDA, every arch)
    pd_q4x_combine_norm,
    // granite fused wqkv (f8row): self-gates on PD_BS_HOST + runtime cc>=8.9
    pd_f8row_gemm_mma_qkv_norm_paged,
    // pf-side rope-only twin (self-gates on PD_BS_HOST)
    pd_f8row_qkv_rope_norm_from_y_paged,
    // two-segment decode GEMM
    pd_f8row_gemm2,
    // prefill swiglu + e4m3-row quant
    pd_swiglu_quant_e4m3_row,
    // norm -> e4m3-row quant fusion
    pd_rmsnorm_quant_e4m3_row,
    pd_add_rmsnorm_scaled_quant_e4m3_row,
    // 523/524: granite NVFP4 fused qkv: raw split GEMM + partials
    // consumer (self-gate on PD_BS_HOST)
    pd_nvf4_gemm_f4c_raw,
    pd_qkv_rope_norm_from_parts_paged,
    // 525: merged-swiglu -> nvf4 down staging
    pd_swiglu_fused_nvf4,
    // 526: decode narrow-tile W4A4 GEMM
    pd_nvf4_gemm_f4cn,
    // 527: f4cn raw-partials twin (no reduce), 528: from-parts scaled norm
    pd_nvf4_gemm_f4cn_raw,
    pd_add_rmsnorm_scaled_from_parts,
    // 529/530: nvf4 decode fold-2: residual+norm+quant from
    // parts, gate|up parts -> swiglu -> nvf4 down input
    pd_add_rmsnorm_quant_nvf4_from_parts,
    pd_swiglu_quant_nvf4_from_parts,
    // 531/532: deep-ring decode GEMM for small-out short-K shapes
    pd_nvf4_gemm_f4cd,
    pd_nvf4_gemm_f4cd_raw,
    // 533-536: swiglu+quant epilogue on f4t + the interleaved-plane twins
    pd_nvf4_gemm_f4t_swq,
    pd_swiglu_fused_il,
    pd_swiglu_fused_nvf4_il,
    pd_swiglu_quant_nvf4_from_parts_il,
    pd_bf16_gemv_nk_f32,
    pd_matvec_f32_sk,
    pd_bf16_gemv_silu_f32,
    pd_bf16_gemv_nk_mr_f32,
    pd_q4x_conv_dil_step_slots,
    pd_bf16_seg2_gemm_mma,
    pd_bf16_hcmix_permute,
    pd_bf16_hcmix_gemm,
    pd_q4x_ple_gather,
    pd_q4x_conv_dil_step_ring,
    pd_gated_delta_recurrent_runs,
    pd_f16_ksplit_set,
    pd_attn_decode_batch_ps,
    // 537: FMHA-style decode attention - per-warp key streams, no per-tile
    // barrier, register-resident (m, l, acc). Its own numeric class.
    pd_attn_decode_fmha,
    pd_moe_topk_batch_s,
    pd_q4x_add_gated_row_s,
    pd_gated_delta_recurrent_runs_pn,
    pd_exp_lt_gemm,
    pd_lowm_gemm,
    pd_lowm_warmup,
    pd_attn_decode_fmha_sp,
    pd_bf16_gemv2_swiglu,
    pd_convert_f32_bf16,
    pd_bf16_gemv_up_hcmix,
    pd_conv_step_slots_split,
    pd_gated_delta_recurrent_slots_gn,
    pd_bf16_gemv_mrow_f32,
    pd_convert_bf16_f32,
    pd_convert_bf16_f32_rows,
    pd_swiglu_mir,
    pd_bf16_pad_rows,
    pd_bf16_hc_perm_pad,
    pd_moe_cache_resolve,
    pd_moe_cache_fill,
    pd_kquant_iq,
    pd_kquant_iq_dense,
    pd_q4x_gdn_split_widen_tiled,
    pd_moe_wave_plan,
    pd_moe_cache_resolve_dev,
    pd_moe_wave_mask,
    pd_kquant_moe_gate_up_list,
    pd_kquant_moe_down_list,
    pd_kquant_iq_tile,
    pd_kquant_moe_gate_up_grp,
    pd_kquant_moe_down_cols,
    pd_q8_0_gemv_sk,
    // 589: nvf4_gemm_f4t4 - the f4t ring at four 128-K stages (block-scale
    // SASS like 430; NULLed with it). Small-die election, see fp4.rs.
    pd_nvf4_gemm_f4t4,
    pd_kquant_moe_down_grp,
    pd_moe_part_fold_at,
    pd_kquant_moe_gate_up_tile,
    pd_kquant_moe_down_tile,
    // 594/595: prefill add+rmsnorm with the e4m3 / nvf4 quant epilogue
    // (the mmq prenorm's exact norm; generic CUDA + the fp8 cvt, no NULL rule
    // beyond the arch floor every fp8 quant already has).
    pd_add_rmsnorm_quant_e4m3_pf,
    pd_add_rmsnorm_quant_nvf4_pf,
    pd_gated_delta_recurrent_pn,
};

PD_EXPORT const PackInfo* paddock_pack_info(void) {
    return &PD_INFO;
}

PD_EXPORT const KernelTableV1* paddock_pack_kernels_v1(void) {
#ifdef PD_BS_HOST
    // Honest per-DEVICE capability: a PD_BS_HOST fatbin can land
    // on a device whose SASS pass compiled the block-scale bodies empty (the
    // multi-arch build always could; the sm_100 lane now builds
    // PD_BS_HOST for cc 10.x deliberately). A non-null entry backed by an
    // empty kernel is a silent no-op - worse than the Launch(801) the
    // null-probe rule exists to prevent - so resolve the table against the
    // running device once:
    //   - kind::mxf8f6f4 block-scale families -> consumer Blackwell (cc 12)
    //   - f8w8 family (plain e4m3 mma + sw ue8m0 fold, PD_F8W8_OK) -> cc >= 8.9
    // Single-device assumption: resolution uses the device current at first
    // table fetch (the engine binds its device before loading the pack).
    static KernelTableV1 t = PD_KERNELS;
    static int resolved = 0;
    if (!resolved) {
        int dev = 0, cma = 0, cmi = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&cma, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&cmi, cudaDevAttrComputeCapabilityMinor, dev);
        // Exact cc match, not `cma != 12`. The device gate for this family is
        // `__CUDA_ARCH__ >= 1200 && __CUDA_ARCH_FEAT_SM120_ALL`, satisfied only
        // by the sm_120a target we actually build. A 12.1 die (DGX Spark GB10)
        // reports cma==12, so a major-only test would advertise these entries
        // on a die whose SASS was compiled with PD_BS_OK=0 -- empty kernel
        // bodies, silently. Minor revisions must fall back, not no-op.
        // pd_dev_bs_sass (abi.cuh) IS that rule; the in-launcher elections
        // read the same function, after the kt3 one drifted to major-only.
        if (!pd_dev_bs_sass()) {
            t.mxfp4_moe_gate_up_bs = NULL;
            t.mxfp4_moe_down_bs = NULL;
            t.mxfp4_moe_down_bs_res = NULL;
            t.mxfp4_moe_gate_up_bs64 = NULL;
            t.mxfp4_moe_down_bs64 = NULL;
            t.q8_0_to_mxfp4 = NULL;
            t.mxfp4_gemm_bs = NULL;
            t.q8_0_to_fp4p = NULL;
            t.mxfp4_gemm_f4 = NULL;
            t.q8_0_to_nvf4 = NULL;
            t.mxfp4_gemm_nv4 = NULL;
            t.nvf4_gemm_f4 = NULL;
            t.nvf4_gemm_f4b = NULL;
            t.nvf4_gemm_f4s = NULL;
            t.nvf4_gemm_f4c = NULL;
            t.nvf4_gemm_f4t = NULL;
            t.nvf4_gemm_f4t4 = NULL;
            t.q8_0_to_nvf4_rot = NULL;
            t.mxfp4_gemm_bs_gu = NULL;
            t.q8_0_to_nvf4_smooth = NULL;
            t.q8_0_to_mxfp4_smooth = NULL;
            // NOTE: nvf4_moe_up_relu2_bs / nvf4_moe_down_bs are not nulled
            // here any more. Their cc12 bodies are block-scale SASS, but the
            // launcher now elects a weight-only bf16 tile arm on every other
            // die (pd_nv4t_arm), so the ENTRY is live wherever either arm
            // exists - gated with the rest of the family below.
            // rowwise pc lane: the kt3/kt3g/kt4* RW bodies are cc12
            // block-scale SASS; the strip-free boxes have no other consumer
            t.f8w_repack_lin_bs_gui = NULL;
            t.f8_gemm_lin_r = NULL;
            t.f8_gemm_lin_kt_r = NULL;
            t.f8_gemm_lin_gu_r = NULL;
            t.f8_gemm_lin_gu_pc_r = NULL;
            t.f8_gemm_lin_gu_r_silu = NULL;
            t.f8_gemm_lin_gu_pc_r_silu = NULL;
            t.f8_gemm_w8_pc_r = NULL;
            t.f8_gemm_w8_pcd_r = NULL;
            t.f8_gemm_w8_pc_qkv_r = NULL;
            t.f8_gemm_w8_pc_qkv_r2 = NULL;
            // the tiled-layout MoE family (472-477): the st/stw bodies are
            // cc12 block-scale SASS (empty elsewhere), and the mtt twins are
            // useless without them - a tiled plane must never exist unless
            // all SIX consumers are live, so the whole set nulls together.
            t.nvf4_moe_up_relu2_st = NULL;
            t.nvf4_moe_down_st = NULL;
            t.nvf4_moe_up_relu2_stw = NULL;
            t.nvf4_moe_down_stw = NULL;
            t.nvf4_moe_up_relu2_mtt = NULL;
            t.nvf4_moe_down_part_tt = NULL;
        }
        // NVFP4 checkpoint-plane consumers: the host half of PD_NV4_OK
        // (moe/block_scale_quant.cuh), and it must stay identical to it -
        // these two gates are the pair the arch audit says must be
        // widened together or not at all.
        //
        // These kernels decode e2m1 in SIMT and accumulate on FFMA (the tc
        // tile adds plain bf16 mma, itself 8.0+); none issues a block-scale
        // instruction, so cc 8.9 is the real floor and the cc-12 test above
        // was wrong for them. It cost the whole nemotron NVFP4 lane on B200:
        // the checkpoint loaded and then failed every generation with "kernel
        // nvf4_moe_up_relu2 missing from the loaded pack".
        //
        // Deliberately not moved: q8_0_to_nvf4{,_rot,_smooth} and the
        // nvf4_gemm_f4* family. The converts are real SASS on sm_100a too, but
        // they feed the Q8_0->NVFP4 repack lane, not this one; the f4* GEMMs
        // are the mxfp4/TMA alternates. Neither is on the nemotron path and
        // neither is validated here - widening an unmeasured gate is how the
        // silent-empty-kernel bug got written in the first place.
        if (!(cma > 8 || (cma == 8 && cmi >= 9))) {
            t.nvf4_dequant = NULL;
            t.nvf4_gemv = NULL;
            t.nvf4_gemv_batch = NULL;
            t.nvf4_moe_up_relu2 = NULL;
            t.nvf4_moe_down_acc = NULL;
            t.nvf4_moe_up_relu2_mt = NULL;
            t.nvf4_moe_down_part = NULL;
            t.nvf4_gemm_mr = NULL;
            t.nvf4_gemm_tc = NULL;
            t.nvf4_gemv_batch_tm = NULL;
            t.nvf4_gemm_mr_tm = NULL;
            t.nvf4_gemm_tc_tm = NULL;
            t.nvf4_gemv_batch_tf = NULL;
            t.nvf4_gemm_mr_tf = NULL;
            t.nvf4_gemm_tc_tf = NULL;
            // the sorted-tile MoE pair: cc12 runs the block-scale bodies,
            // every other die runs the weight-only bf16 tile arm, and both
            // need the 8.9 floor (nibble decode via e4m3, bf16 mma is 8.0)
            t.nvf4_moe_up_relu2_bs = NULL;
            t.nvf4_moe_down_bs = NULL;
        }
        // Exact cc match, same reason: the device gate is
        // `__CUDA_ARCH__ == 1000 && __CUDA_ARCH_FEAT_SM100_ALL`, so a 10.3 die
        // (Blackwell Ultra) compiles these bodies away while still reporting
        // cma==10. Without the minor test it would launch empty tcgen05
        // kernels instead of taking the portable path.
        if (!(cma == 10 && cmi == 0)) {
            // tc5p SASS is sm_100a-only; the tile plane has no other consumer
            t.f8_repack_tiles = NULL;
            t.f8t_gemm = NULL;
            //  decode-band f8 GEMMs ride the same sm_100a-only class
            t.f8bs_moe_gemm_gu_d32 = NULL;
            t.f8bs_moe_gemm_dn_d32 = NULL;
            // slots 543/544: pd_lowm_gemm dispatches ONLY the tcgen05/TMEM
            // form (pd_lowm5_kernel, body under PD_TC5_OK == sm_100a). On any
            // other die the entry was a lie in both pack flavours: a
            // multi-arch pack launched an EMPTY kernel (warm-up "passed",
            // and PADDOCK_Q38FN_LOWM=1 would have returned zeros), an
            // sm_120-only pack answered 801 and the engine printed a
            // warm-up-refused warning on every Flash-Next load (5060 Ti,
            // 2026-09-06). Null = the engine sees the arm as absent and
            // never warms it up. The gen-1 SIMT cluster kernel in lowm.cuh
            // is not dispatched anywhere, so there is no non-tcgen05 route
            // to keep.
            t.lowm_gemm = NULL;
            t.lowm_warmup = NULL;
        }
        if (cma < 9 && !(cma == 8 && cmi >= 9)) {
            t.add_rmsnorm_e4m3_xn_b16 = NULL;
            t.gated_rmsnorm_e4m3 = NULL;
            t.gated_rmsnorm_e4m3_row = NULL;
            t.q8_0_to_f8w = NULL;
            t.f8_gemm_w8 = NULL;
            t.f8_gemv = NULL;
            t.f8_gemv_batch = NULL;
            t.f8_gemm_mma_ks = NULL;
            t.q8_0_to_f8row = NULL;
            t.f8row_gemm = NULL;
            // flat-scale e4m3 expert lane: mma.sync e4m3 is sm_89+
            t.quantize_e4m3_b32f = NULL;
            t.f8row_moe_gate_up_mma_geglu = NULL;
            t.f8row_moe_gate_up_mma_geglu_f8 = NULL;
            t.f8row_moe_down_mma = NULL;
            t.f8d_gemm_mma_ks = NULL;
            t.f8_gemm_w8_o16 = NULL;
            t.f8r_gemm_mma_ks = NULL;
        }
        if (cma < 9) {
            // tile-linear f8 lane: bulk/mbarrier bodies are sm_90+ SASS
            t.f8w_repack_lin = NULL;
            t.f8_gemm_lin = NULL;
            t.f8_gemm_lin_kt = NULL;
            t.f8_gemm_lin_kt_split = NULL;
            t.f8w_repack_lin_bs = NULL;
            t.f8_gemm_lin_bs = NULL;
            // gu-fusion trio rides the same lane (kt3-clone SASS; geglu2i is
            // plain CUDA but pointless without the interleaved repack)
            t.f8w_repack_lin_gui = NULL;
            t.quantize_e4m3_geglu2i = NULL;
            t.f8_gemm_lin_gu = NULL;
            t.quantize_e4m3_swiglu2i = NULL;
            t.f8_gemm_lin_gu_silu = NULL;
        }
        if (cma < 9) {
            // dec3 streams via cp.async.bulk - no SASS below sm_90
            t.q8_0_moe_gu_dec3_geglu = NULL;
            t.q8_0_moe_dn_dec3 = NULL;
            // v3t twins stage W via TMA - sm_90+ only
            t.q8_0_moe_gate_up_mma2t_geglu = NULL;
            t.q8_0_moe_down_mma2t = NULL;
            t.q8_0_moe_gate_up_g2_geglu = NULL;
        }
        if (cma != 10) {
        }
        // Honest per-DRIVER capability: the launchers below have no route
        // that does not start from a cuTensorMapEncodeTiled tensor map, and
        // every one of them answers cudaErrorNotSupported (or InvalidValue)
        // when pd_tmap_encode() is null - which it is whenever the driver
        // cannot serve the entry point (a pack built with a toolkit newer
        // than the driver was the 2026-09 case: 13.3 nvcc on a 13.2 driver,
        // RTX PRO 6000 Max-Q, self-built). The engine elects these lanes by
        // entry presence (has_f8_lin, has_f8_o16, the f4t elector...), so a
        // non-null pointer here is exactly the Launch(801)-on-every-prefill
        // failure the null-probe rule exists to prevent. Null = the engine
        // keeps its non-TMA class; pd_tmap_encode() prints the [tma] witness
        // once so the serve log names the driver as the cause. Launchers
        // that fall back to a non-TMA route on their own (f8_gemm_w8, the pc
        // family's -2 decline, the attention bulk paths, f8row, mxfp4_bs)
        // keep their entries.
        if (pd_tmap_encode() == nullptr) {
            t.f8_gemm_lin_kt = NULL;
            t.f8_gemm_lin_kt_r = NULL;
            t.f8_gemm_w8_o16 = NULL;
            t.f8t_gemm = NULL;
            t.f8t_gemm2 = NULL;
            t.f8bs_moe_gemm_gu = NULL;
            t.f8bs_moe_gemm_dn = NULL;
            t.f8bs_moe_gemm_gu_d32 = NULL;
            t.f8bs_moe_gemm_dn_d32 = NULL;
            t.nvf4_gemm_f4t = NULL;
            t.nvf4_gemm_f4t4 = NULL;
            t.nvf4_gemm_f4t_swq = NULL;
            t.q8_0_moe_gate_up_g2_geglu = NULL;
            t.q8_0_moe_gate_up_mma2t_geglu = NULL;
            t.q8_0_moe_down_mma2t = NULL;
            t.lowm_gemm = NULL;
            t.lowm_warmup = NULL;
        }
        resolved = 1;
    }
    return &t;
#else
    return &PD_KERNELS;
#endif
}
