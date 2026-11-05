// norm/add_norm16.cuh - the residual tails of a prefill layer in one pass, with the bf16 rows
// the next tensor-core GEMM reads (slot 924). Textually-included segment of the single pack
// translation unit, after norm/add_rmsnorm.cuh (needs elementwise.cuh's pd_df / pd_acc_of and
// abi.cuh's width rules).
//
// Two chains, each bit-identical to the launches it replaces at every row count it accepts:
//
//   mode 0 - pd_add_rmsnorm_batch + convert_f32_bf16: x += proj, xn = rmsnorm(x) * w (f32
//            accumulate at add_rmsnorm's width), plus bf16(xn).
//   mode 1 - moe_slot_combine_bf16 + add_inplace + pd_rmsnorm_batch + convert_f32_bf16: the k
//            bf16 per-slot partials summed in f32 in ascending slot order and added onto proj,
//            that added onto x, then the batched norm (its accumulator election, its width) of
//            the NEXT layer, plus bf16(xn). k = 0 is a plain add.
//
// At 8192 rows of 2048 mode 1 moves ~425 MB where its four launches moved ~750 (the partials
// are read once, x is never re-read from DRAM, xn's f32 plane is optional). Each thread reads
// back only the x elements it wrote itself, the partition the reductions walk in.

__global__ void pd_add_norm16_f32_kernel(float* __restrict__ x, const float* __restrict__ proj,
                                         const float* __restrict__ w, float* __restrict__ xn,
                                         __nv_bfloat16* __restrict__ x16, uint32_t n4, float eps) {
    const uint32_t b = blockIdx.x, tid = threadIdx.x, nth = blockDim.x;
    float4* xb = reinterpret_cast<float4*>(x + (size_t)b * n4 * 4u);
    const float4* pb = reinterpret_cast<const float4*>(proj + (size_t)b * n4 * 4u);
    __shared__ float wsum[32];
    __shared__ float s_inv;
    // pd_add_rmsnorm_batch_kernel's vec branch: fmaf(1, p, v) == p + v
    float acc = 0.0f;
    for (uint32_t i = tid; i < n4; i += nth) {
        float4 v = xb[i];
        const float4 pv = pb[i];
        v.x = fmaf(1.0f, pv.x, v.x); v.y = fmaf(1.0f, pv.y, v.y);
        v.z = fmaf(1.0f, pv.z, v.z); v.w = fmaf(1.0f, pv.w, v.w);
        xb[i] = v;
        acc += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    for (uint32_t sh = 16; sh > 0; sh >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, sh);
    if ((tid & 31u) == 0) wsum[tid >> 5] = acc;
    __syncthreads();
    if (tid == 0) {
        float sum = 0.0f;
        for (uint32_t wi = 0; wi < (nth + 31u) >> 5; ++wi) sum += wsum[wi];
        s_inv = 1.0f / sqrtf(sum / (float)(n4 * 4u) + eps);
    }
    __syncthreads();
    const float inv = s_inv;
    const float4* w4 = reinterpret_cast<const float4*>(w);
    float4* o4 = xn ? reinterpret_cast<float4*>(xn + (size_t)b * n4 * 4u) : nullptr;
    __nv_bfloat16* h = x16 ? x16 + (size_t)b * n4 * 4u : nullptr;
    for (uint32_t i = tid; i < n4; i += nth) {
        const float4 v = xb[i], wv = w4[i];
        const float4 r = make_float4(v.x * inv * wv.x, v.y * inv * wv.y, v.z * inv * wv.z,
                                     v.w * inv * wv.w);
        if (o4) o4[i] = r;
        if (h) {
            const __nv_bfloat16 t[4] = {__float2bfloat16(r.x), __float2bfloat16(r.y),
                                        __float2bfloat16(r.z), __float2bfloat16(r.w)};
            *reinterpret_cast<uint2*>(h + 4u * i) = *reinterpret_cast<const uint2*>(t);
        }
    }
}

template <int ACC>
__global__ void pd_add_norm16_fold_kernel(float* __restrict__ x, const float* __restrict__ proj,
                                          const uint2* __restrict__ part, uint32_t k,
                                          const float* __restrict__ w, float* __restrict__ xn,
                                          __nv_bfloat16* __restrict__ x16, uint32_t n4,
                                          float eps) {
    using A = typename pd_acc_of<ACC>::type;
    const uint32_t b = blockIdx.x, tid = threadIdx.x, nth = blockDim.x;
    float4* xb = reinterpret_cast<float4*>(x + (size_t)b * n4 * 4u);
    const float4* pb = reinterpret_cast<const float4*>(proj + (size_t)b * n4 * 4u);
    // ---- the fold (pd_moe_slot_combine_bf16_kernel, then pd_add_inplace_kernel), and the
    // batched norm's square-sum over the same lanes (pd_rmsnorm_batch_kernel_t's vec branch)
    __shared__ A wsum[32];
    __shared__ float s_inv;
    A acc;
    if constexpr (ACC == PD_ACC_DF) { acc.hi = 0.0f; acc.lo = 0.0f; } else { acc = (A)0; }
    for (uint32_t i = tid; i < n4; i += nth) {
        float4 d = pb[i];
        if (k != 0u) {
            float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
            for (uint32_t j = 0; j < k; ++j) {
                const uint2 pk = part[((size_t)b * k + j) * n4 + i];
                const __nv_bfloat162 lo = *reinterpret_cast<const __nv_bfloat162*>(&pk.x);
                const __nv_bfloat162 hi = *reinterpret_cast<const __nv_bfloat162*>(&pk.y);
                s.x += __bfloat162float(lo.x); s.y += __bfloat162float(lo.y);
                s.z += __bfloat162float(hi.x); s.w += __bfloat162float(hi.y);
            }
            d.x += s.x; d.y += s.y; d.z += s.z; d.w += s.w;
        }
        float4 v = xb[i];
        v.x += d.x; v.y += d.y; v.z += d.z; v.w += d.w;
        xb[i] = v;
        if constexpr (ACC == PD_ACC_DF) {
            pd_df_add(acc, v.x * v.x);
            pd_df_add(acc, v.y * v.y);
            pd_df_add(acc, v.z * v.z);
            pd_df_add(acc, v.w * v.w);
        } else {
            acc += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
    }
    for (uint32_t sh = 16; sh > 0; sh >>= 1) {
        if constexpr (ACC == PD_ACC_DF) {
            pd_df o;
            o.hi = __shfl_down_sync(0xffffffffu, acc.hi, sh);
            o.lo = __shfl_down_sync(0xffffffffu, acc.lo, sh);
            acc = pd_df_merge(acc, o);
        } else {
            acc += __shfl_down_sync(0xffffffffu, acc, sh);
        }
    }
    if ((tid & 31u) == 0) wsum[tid >> 5] = acc;
    __syncthreads();
    if (tid == 0) {
        const uint32_t nwarps = (nth + 31u) >> 5;
        double total;
        if constexpr (ACC == PD_ACC_DF) {
            pd_df sum; sum.hi = 0.0f; sum.lo = 0.0f;
            for (uint32_t wi = 0; wi < nwarps; ++wi) sum = pd_df_merge(sum, wsum[wi]);
            total = (double)sum.hi + (double)sum.lo;
        } else {
            A sum = (A)0;
            for (uint32_t wi = 0; wi < nwarps; ++wi) sum += wsum[wi];
            total = (double)sum;
        }
        s_inv = 1.0f / sqrtf((float)(total / (double)(n4 * 4u)) + eps);
    }
    __syncthreads();
    const float inv = s_inv;
    const float4* w4 = reinterpret_cast<const float4*>(w);
    float4* o4 = xn ? reinterpret_cast<float4*>(xn + (size_t)b * n4 * 4u) : nullptr;
    __nv_bfloat16* h = x16 ? x16 + (size_t)b * n4 * 4u : nullptr;
    for (uint32_t i = tid; i < n4; i += nth) {
        const float4 v = xb[i], wv = w4[i];
        const float4 r = make_float4(v.x * inv * wv.x, v.y * inv * wv.y, v.z * inv * wv.z,
                                     v.w * inv * wv.w);
        if (o4) o4[i] = r;
        if (h) {
            const __nv_bfloat16 t[4] = {__float2bfloat16(r.x), __float2bfloat16(r.y),
                                        __float2bfloat16(r.z), __float2bfloat16(r.w)};
            *reinterpret_cast<uint2*>(h + 4u * i) = *reinterpret_cast<const uint2*>(t);
        }
    }
}

// 924: mode 0 / 1 as above. part = bf16 [rows][k][n] (mode 1; k = 0 skips it); xn / x16 each
// optional. -2 when n % 4 != 0 or a plane is off 16 B (8 B for x16) - the caller keeps the
// launches it would have made.
PD_EXPORT
int pd_add_norm16(void* x, const void* proj, const void* part, uint32_t k, const void* w,
                  void* xn, void* x16, uint32_t n, float eps, uint32_t rows, uint32_t mode,
                  void* stream) {
    if (n == 0 || rows == 0) return 0;
    if ((n & 3u) != 0u || mode > 1u || (mode == 0u && k != 0u) || (k != 0u && part == nullptr)
        || (((uintptr_t)x | (uintptr_t)proj | (uintptr_t)part | (uintptr_t)w | (uintptr_t)xn)
            & 15u) != 0u
        || (((uintptr_t)x16) & 7u) != 0u)
        return -2;
    cudaStream_t st = (cudaStream_t)stream;
    if (mode == 0u) {
        // pd_add_rmsnorm_batch's width
        const uint32_t nth = rows >= 64u ? pd_norm_wide_nth(rows) : pd_norm_decode_nth();
        pd_add_norm16_f32_kernel<<<rows, nth, 0, st>>>((float*)x, (const float*)proj,
                                                       (const float*)w, (float*)xn,
                                                       (__nv_bfloat16*)x16, n >> 2, eps);
        return pd_launch_status();
    }
    // pd_rmsnorm_batch's width and accumulator
    const uint32_t nth = rows >= 64u ? pd_norm_wide_nth_ws(rows) : pd_norm_decode_nth();
    const int accm = pd_norm_acc_mode();
#define PD_AN16_GO(ACC)                                                                         \
    pd_add_norm16_fold_kernel<ACC><<<rows, nth, 0, st>>>(                                      \
        (float*)x, (const float*)proj, (const uint2*)part, k, (const float*)w, (float*)xn,      \
        (__nv_bfloat16*)x16, n >> 2, eps)
    if (accm == PD_ACC_DF) PD_AN16_GO(PD_ACC_DF);
    else if (accm == PD_ACC_F64) PD_AN16_GO(PD_ACC_F64);
    else PD_AN16_GO(PD_ACC_F32);
#undef PD_AN16_GO
    return pd_launch_status();
}
