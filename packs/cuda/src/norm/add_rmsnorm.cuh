// norm/add_rmsnorm.cuh - the fused residual-add + RMSNorm row kernels behind pd_add_rmsnorm_batch
// and pd_add_rmsnorm_scaled_batch (exports.cuh). Textually-included segment of the single pack
// translation unit, right after moe/q8.cuh (moved out of it unchanged).

// Fused residual-add + RMSNorm for the decode/serving paths: x += proj
// (written back), out = rmsnorm(x, w). Bit-identical to the add-then-norm
// two-kernel sequence (same add, same square-sum order over the summed
// values) - it just removes one graph-node drain per layer (the b=1 tail
// measured ~2-4 us per small launch; the mid-layer add+norm pair is 2 of
// the ~9 tail launches on every DeltaNet layer).
// `pscale` folds a residual MULTIPLIER into the fused add (granite's
// `residual_multiplier`). It exists because granite was the one
// family that could not use this kernel: its residual is `x += res_s * proj`,
// so it paid pd_scale_add + pd_rmsnorm_batch as two launches -- ~80 extra
// launches per decode tick on the 8b, measured at 82 us/token of pure
// scale_add. fmaf(1, p, v) is exactly p + v, so `pd_add_rmsnorm_batch` passing
// 1.0f keeps every existing caller bit-identical, and the granite chain is
// bit-identical too because pd_scale_add_kernel's `x[i] += w * y[i]`
// contracts to the same fma.
__global__ void pd_add_rmsnorm_batch_kernel(float* __restrict__ x,
                                            const float* __restrict__ proj,
                                            const float* __restrict__ w,
                                            float* __restrict__ out, uint32_t n,
                                            float eps, float pscale) {
    // cascade (laguna chain): proj is the predecessor
    // wo-GEMV's output. No-op under plain launches / pre-sm90.
    PD_PDL_ARM();
    const uint32_t b = blockIdx.x;
    float* xb = x + (size_t)b * n;
    const float* pb = proj + (size_t)b * n;
    float* ob = out + (size_t)b * n;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    __shared__ float wsum[32];
    __shared__ float s_inv;
    float acc = 0.0f;
    const bool vec = (n & 3u) == 0;
    if (vec) {
        const uint32_t n4 = n >> 2;
        float4* x4 = reinterpret_cast<float4*>(xb);
        const float4* p4 = reinterpret_cast<const float4*>(pb);
        for (uint32_t i = tid; i < n4; i += nth) {
            float4 v = x4[i];
            const float4 pv = p4[i];
            v.x = fmaf(pscale, pv.x, v.x); v.y = fmaf(pscale, pv.y, v.y);
            v.z = fmaf(pscale, pv.z, v.z); v.w = fmaf(pscale, pv.w, v.w);
            x4[i] = v;
            acc += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
    } else {
        for (uint32_t i = tid; i < n; i += nth) {
            const float v = fmaf(pscale, pb[i], xb[i]);
            xb[i] = v;
            acc += v * v;
        }
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
    for (uint32_t i = tid; i < n; i += nth) ob[i] = xb[i] * inv * w[i];
}

// The row-in-registers form of pd_add_rmsnorm_batch_kernel for launches of <= 256 threads (the
// wide-branch width at decode batches 64+): all of a thread's (<= 8) float4 loads issue before the
// first use, and the output is float4 from the summed values in registers. The looped kernel waited
// a DRAM round trip per float4 and re-read x and w through 20 scalar iterations - 20.6 us a 124-row
// launch under ncu (3.7% of DRAM peak). Same per-thread add and square-sum order, same output
// expression: bit-identical. Its register arrays do not fit the 1024-thread decode launches
// (pd_norm_decode_nth below 64 rows - CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES), which keep the looped
// kernel.
__global__ void __launch_bounds__(256) pd_add_rmsnorm_batch_r_kernel(float* __restrict__ x,
                                                                     const float* __restrict__ proj,
                                                                     const float* __restrict__ w,
                                                                     float* __restrict__ out,
                                                                     uint32_t n, float eps, float pscale) {
    // the norm weights are the model's, not the predecessor's: in flight before the PDL wait,
    // landed by the time the reduction is done (they were a second round trip after it)
    constexpr uint32_t PD_ARN_R = 8u;
    float4 wv[PD_ARN_R];
    {
        const float4* w4 = reinterpret_cast<const float4*>(w);
#pragma unroll
        for (uint32_t r = 0; r < PD_ARN_R; ++r) {
            const uint32_t i = threadIdx.x + r * blockDim.x;
            if (i < (n >> 2)) wv[r] = w4[i];
        }
    }
    PD_PDL_ARM();
    const uint32_t b = blockIdx.x;
    float* xb = x + (size_t)b * n;
    const float* pb = proj + (size_t)b * n;
    float* ob = out + (size_t)b * n;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    __shared__ float wsum[32];
    __shared__ float s_inv;
    float acc = 0.0f;
    {
        const uint32_t n4 = n >> 2;
        float4* x4 = reinterpret_cast<float4*>(xb);
        const float4* p4 = reinterpret_cast<const float4*>(pb);
        float4 v[PD_ARN_R], pv[PD_ARN_R];
#pragma unroll
        for (uint32_t r = 0; r < PD_ARN_R; ++r) {
            const uint32_t i = tid + r * nth;
            if (i < n4) { v[r] = x4[i]; pv[r] = p4[i]; }
        }
#pragma unroll
        for (uint32_t r = 0; r < PD_ARN_R; ++r) {
            const uint32_t i = tid + r * nth;
            if (i < n4) {
                v[r].x = fmaf(pscale, pv[r].x, v[r].x); v[r].y = fmaf(pscale, pv[r].y, v[r].y);
                v[r].z = fmaf(pscale, pv[r].z, v[r].z); v[r].w = fmaf(pscale, pv[r].w, v[r].w);
                x4[i] = v[r];
                acc += v[r].x * v[r].x + v[r].y * v[r].y + v[r].z * v[r].z + v[r].w * v[r].w;
            }
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
        float4* o4 = reinterpret_cast<float4*>(ob);
#pragma unroll
        for (uint32_t r = 0; r < PD_ARN_R; ++r) {
            const uint32_t i = tid + r * nth;
            if (i < n4)
                o4[i] = make_float4(v[r].x * inv * wv[r].x, v[r].y * inv * wv[r].y,
                                    v[r].z * inv * wv[r].z, v[r].w * inv * wv[r].w);
        }
    }
}

// pd_add_rmsnorm_batch_r_kernel + pd_quantize_nvf4 (E4M3 = false, slot 876) or
// pd_quantize_e4m3 (E4M3 = true, slot 877) in one launch: the residual add,
// sumsq, inverse and `v * inv * w` exactly as the r kernel above (same loads, same order, same
// width - the launcher elects it only where the r kernel would run), then the normed row packed
// to nvf4 for the GEMM behind it. A 16-element block is four consecutive float4 slots - four
// adjacent lanes - so the block amax is two xor shuffles over the same 16 values pd_nvf4_quant8
// sees (fmaxf is order-free), then the same pd_nvf4_scale and pd_e2m1_rn per value, low nibble
// first: q and the scales are pd_quantize_nvf4's bytes. The e4m3 twin hands each float4 to
// pd_e4m3_quant4 exactly as pd_quantize_e4m3_kernel does (an 8-lane group = one 32-element
// block, the same lane8 = tid & 7 since nth % 8 == 0). `out` still lands the f32 normed row.
// SW (slots 879 nvf4, 884 e4m3): the scales land in the tcgen05 GEMM's swizzled atom layout
// (pd_nvf4_swz_off) instead of row-major - the same bytes, so the GEMM skips pd_sf_swz128.
template <bool E4M3, bool SW = false>
__global__ void __launch_bounds__(256) pd_add_rmsnorm_q_r_kernel(float* __restrict__ x,
                                                                    const float* __restrict__ proj,
                                                                    const float* __restrict__ w,
                                                                    float* __restrict__ out,
                                                                    unsigned char* __restrict__ q,
                                                                    unsigned char* __restrict__ qs,
                                                                    uint32_t n, float eps, float pscale) {
    constexpr uint32_t PD_ARN_R = 8u;
    float4 wv[PD_ARN_R];   // the norm weights ahead of the PDL wait (the r kernel above)
    {
        const float4* w4 = reinterpret_cast<const float4*>(w);
#pragma unroll
        for (uint32_t r = 0; r < PD_ARN_R; ++r) {
            const uint32_t i = threadIdx.x + r * blockDim.x;
            if (i < (n >> 2)) wv[r] = w4[i];
        }
    }
    PD_PDL_ARM();
    const uint32_t b = blockIdx.x;
    float* xb = x + (size_t)b * n;
    const float* pb = proj + (size_t)b * n;
    float* ob = out + (size_t)b * n;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    __shared__ float wsum[32];
    __shared__ float s_inv;
    float acc = 0.0f;
    const uint32_t n4 = n >> 2;
    float4* x4 = reinterpret_cast<float4*>(xb);
    const float4* p4 = reinterpret_cast<const float4*>(pb);
    float4 v[PD_ARN_R], pv[PD_ARN_R];
#pragma unroll
    for (uint32_t r = 0; r < PD_ARN_R; ++r) {
        const uint32_t i = tid + r * nth;
        if (i < n4) { v[r] = x4[i]; pv[r] = p4[i]; }
    }
#pragma unroll
    for (uint32_t r = 0; r < PD_ARN_R; ++r) {
        const uint32_t i = tid + r * nth;
        if (i < n4) {
            v[r].x = fmaf(pscale, pv[r].x, v[r].x); v[r].y = fmaf(pscale, pv[r].y, v[r].y);
            v[r].z = fmaf(pscale, pv[r].z, v[r].z); v[r].w = fmaf(pscale, pv[r].w, v[r].w);
            x4[i] = v[r];
            acc += v[r].x * v[r].x + v[r].y * v[r].y + v[r].z * v[r].z + v[r].w * v[r].w;
        }
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
    float4* o4 = reinterpret_cast<float4*>(ob);
    unsigned char* qb = q + (size_t)b * (E4M3 ? n : (n >> 1));
    unsigned char* sb = qs + (size_t)b * (n >> (E4M3 ? 5 : 4));
#pragma unroll
    for (uint32_t r = 0; r < PD_ARN_R; ++r) {
        const uint32_t i = tid + r * nth;
        // n % 16 == 0 and nth % 4 == 0: a 4-lane block group is wholly in or out of range
        float4 o = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (i < n4) {
            o = make_float4(v[r].x * inv * wv[r].x, v[r].y * inv * wv[r].y, v[r].z * inv * wv[r].z,
                            v[r].w * inv * wv[r].w);
            o4[i] = o;
        }
        if constexpr (E4M3) {
            // n % 32 == 0: an 8-lane group is wholly in or out of range (group-masked shfls)
            if (i < n4) {
                if constexpr (SW)
                    pd_e4m3_quant4_at(o, tid & 7u, qb, i * 4u, qs + pd_nvf4_swz_off(b, i >> 3, n >> 7));
                else
                    pd_e4m3_quant4(o, tid & 7u, qb, sb, i * 4u);
            }
            continue;
        }
        float a = fmaxf(fmaxf(fabsf(o.x), fabsf(o.y)), fmaxf(fabsf(o.z), fabsf(o.w)));
        a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 1));
        a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 2));
        if (i < n4) {
            float qi;
            const unsigned sc = pd_nvf4_scale(a, &qi);
            const uint32_t p = pd_e2m1_rn(o.x * qi) | (pd_e2m1_rn(o.y * qi) << 4)
                             | (pd_e2m1_rn(o.z * qi) << 8) | (pd_e2m1_rn(o.w * qi) << 12);
            *(uint16_t*)(qb + 2u * i) = (uint16_t)p;
            if ((i & 3u) == 0u) {
                if constexpr (SW) qs[pd_nvf4_swz_off(b, i >> 2, n >> 6)] = (unsigned char)sc;
                else sb[i >> 2] = (unsigned char)sc;
            }
        }
    }
}
