// sandwich_norm.cuh - the sandwich blocks' fused post-norm + next norm
// (slot 777). Its two halves are pd_rmsnorm_add_scale_v4f_kernel
// (gemm/f32_qkv.cuh) and pd_rmsnorm_batch_kernel_t's vectorized branch
// (elementwise.cuh) verbatim; needs abi.cuh's width/accumulator rules and
// elementwise.cuh's pd_df / pd_acc_of, so it sits after both in pack.cu.

// Slot 777: a sandwich block's post-norm fused with the NEXT norm, one row a
// block - x += rmsnorm(proj) * w_post * s (section A, the v4f kernel above
// verbatim), then xn = rmsnorm(x) * w_pre (section B, pd_rmsnorm_batch's
// vectorized branch and epilogue verbatim over the row this thread just
// wrote, at the same block width), with an optional bf16 copy of xn (the
// prefill pair's input, convert_f32_bf16's round-to-nearest) and an optional
// f32 xn. Bit-identical to rmsnorm_add_scale + rmsnorm_batch (+
// convert_f32_bf16) at the widths it accepts: it re-reads x from L1/L2 instead
// of DRAM, and a consumer that only wants bf16 rows skips the f32 plane.
// Accepts only the vec4 add-scale regime (rows >= 256, n % 4, 16 B aligned);
// -2 elsewhere, and the caller keeps the two launches.
//
// Slot 866 adds an mmq-layout copy of xn (section C, yq non-null):
// pd_quantize_q8_mmq's arithmetic on the values section B forms - the amax of
// each 32-value block (8 consecutive float4 lanes - order-free), the same
// scale, reciprocal, round-to-nearest and clamp - so the GEMM's activation
// bytes come out of the same pass that wrote x, and xn's f32 plane is
// optional. The grid then spans the 128-padded batch: blocks past `rows`
// zero their pad rows (the quantize wrote zeros there) and touch nothing else.
template <int ACC>
__global__ void pd_rmsnorm_add_scale_norm_kernel(float* __restrict__ x,
                                                 const float* __restrict__ proj,
                                                 const float* __restrict__ wpost,
                                                 const float* __restrict__ wpre,
                                                 float* __restrict__ xn,
                                                 __nv_bfloat16* __restrict__ x16,
                                                 uint32_t n4, float eps, float s,
                                                 uint8_t* __restrict__ yq = nullptr,
                                                 uint32_t rows = 0u,
                                                 uint32_t batch_pad = 0u) {
    using A = typename pd_acc_of<ACC>::type;
    const uint32_t b = blockIdx.x;
    if (yq && b >= rows) {   // a pad row of the mmq layout: zeros, nothing else
        const uint32_t nch = n4 >> 5;
        for (uint32_t j = threadIdx.x; j < nch * 36u; j += blockDim.x)
            reinterpret_cast<uint32_t*>(yq + ((size_t)(j / 36u) * batch_pad + b) * 144u)[j % 36u] = 0u;
        return;
    }
    const float4* pb = (const float4*)(proj + (size_t)b * n4 * 4u);
    float4* xb = (float4*)(x + (size_t)b * n4 * 4u);
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    // ---- section A: pd_rmsnorm_add_scale_v4f_kernel
    {
        const float4* w4 = (const float4*)wpost;
        __shared__ double wsum[32];
        __shared__ float s_inv;
        double acc = 0.0;
        for (uint32_t i = tid; i < n4; i += nth) {
            const float4 v = pb[i];
            acc += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
        for (uint32_t o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
        if ((tid & 31u) == 0) wsum[tid >> 5] = acc;
        __syncthreads();
        if (tid == 0) {
            double sum = 0.0;
            for (uint32_t wi = 0; wi < (nth + 31u) >> 5; ++wi) sum += wsum[wi];
            s_inv = rsqrtf((float)(sum / (double)(n4 * 4u)) + eps);
        }
        __syncthreads();
        const float inv = s_inv;
        for (uint32_t i = tid; i < n4; i += nth) {
            float4 xv = xb[i];
            const float4 pv = pb[i];
            const float4 wv = w4[i];
            xv.x = (xv.x + pv.x * inv * wv.x) * s;
            xv.y = (xv.y + pv.y * inv * wv.y) * s;
            xv.z = (xv.z + pv.z * inv * wv.z) * s;
            xv.w = (xv.w + pv.w * inv * wv.w) * s;
            xb[i] = xv;
        }
    }
    // ---- section B: pd_rmsnorm_batch_kernel_t's vectorized branch. Each
    // thread reads back exactly the elements it wrote above (same partition).
    __shared__ A wsum2[32];
    __shared__ float s_inv2;
    A acc;
    if constexpr (ACC == PD_ACC_DF) { acc.hi = 0.0f; acc.lo = 0.0f; } else { acc = (A)0; }
    for (uint32_t i = tid; i < n4; i += nth) {
        float4 v = xb[i];
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
    const uint32_t warp = tid >> 5, lane = tid & 31u;
    if (lane == 0) wsum2[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        const uint32_t nwarps = (nth + 31u) >> 5;
        double total;
        if constexpr (ACC == PD_ACC_DF) {
            pd_df sum; sum.hi = 0.0f; sum.lo = 0.0f;
            for (uint32_t wi = 0; wi < nwarps; ++wi) sum = pd_df_merge(sum, wsum2[wi]);
            total = (double)sum.hi + (double)sum.lo;
        } else {
            A sum = (A)0;
            for (uint32_t wi = 0; wi < nwarps; ++wi) sum += wsum2[wi];
            total = (double)sum;
        }
        s_inv2 = 1.0f / sqrtf((float)(total / (double)(n4 * 4u)) + eps);
    }
    __syncthreads();
    const float inv = s_inv2;
    const float4* w4 = reinterpret_cast<const float4*>(wpre);
    float4* o4 = xn ? reinterpret_cast<float4*>(xn + (size_t)b * n4 * 4u) : nullptr;
    __nv_bfloat16* h = x16 ? x16 + (size_t)b * n4 * 4u : nullptr;
    for (uint32_t i = tid; i < n4; i += nth) {
        float4 v = xb[i], wv = w4[i], r;
        r.x = v.x * inv * wv.x;
        r.y = v.y * inv * wv.y;
        r.z = v.z * inv * wv.z;
        r.w = v.w * inv * wv.w;
        if (o4) o4[i] = r;
        if (h) {
            __nv_bfloat16 t[4] = {__float2bfloat16(r.x), __float2bfloat16(r.y),
                                  __float2bfloat16(r.z), __float2bfloat16(r.w)};
            *reinterpret_cast<uint2*>(h + 4u * i) = *reinterpret_cast<const uint2*>(t);
        }
        if (yq) {
            // ---- section C: the 32-value block is lanes (lane & ~7) .. +7 (n4 % 32
            // == 0, so every warp runs every trip whole)
            float a = fmaxf(fmaxf(fabsf(r.x), fabsf(r.y)), fmaxf(fabsf(r.z), fabsf(r.w)));
            #pragma unroll
            for (uint32_t sh = 4; sh > 0; sh >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, sh));
            const float scl = a * (1.0f / 127.0f);
            const float qinv = scl > 0.0f ? 1.0f / scl : 0.0f;
            const uint32_t e0 = 4u * i, within = e0 & 127u;
            uint8_t* blk = yq + ((size_t)(e0 >> 7) * batch_pad + b) * 144u;
            char4 q;
            int qi;
            qi = __float2int_rn(r.x * qinv); q.x = (char)(qi < -127 ? -127 : (qi > 127 ? 127 : qi));
            qi = __float2int_rn(r.y * qinv); q.y = (char)(qi < -127 ? -127 : (qi > 127 ? 127 : qi));
            qi = __float2int_rn(r.z * qinv); q.z = (char)(qi < -127 ? -127 : (qi > 127 ? 127 : qi));
            qi = __float2int_rn(r.w * qinv); q.w = (char)(qi < -127 ? -127 : (qi > 127 ? 127 : qi));
            *reinterpret_cast<char4*>(blk + 16u + within) = q;
            if ((lane & 7u) == 0u) reinterpret_cast<float*>(blk)[within >> 5] = scl;
        }
    }
}

PD_EXPORT
int pd_rmsnorm_add_scale_norm(void* x, const void* proj, const void* wpost,
                              const void* wpre, void* xn, void* x16, uint32_t n,
                              float eps, float s, uint32_t rows, void* stream) {
    if (n == 0 || rows == 0) return 0;
    if (rows < 256u || (n & 3u) != 0 ||
        (((uintptr_t)x | (uintptr_t)proj | (uintptr_t)wpost | (uintptr_t)wpre |
          (uintptr_t)xn) & 15u) != 0 || (((uintptr_t)x16) & 7u) != 0)
        return -2;
    // both halves' own width: the add-scale and the batched norm read the
    // same rule at these row counts
    const uint32_t nth = pd_norm_wide_nth_ws(rows);
    const int accm = pd_norm_acc_mode();
    cudaStream_t st = (cudaStream_t)stream;
#define PD_RASN_GO(ACC)                                                              \
    pd_rmsnorm_add_scale_norm_kernel<ACC><<<rows, nth, 0, st>>>(                     \
        (float*)x, (const float*)proj, (const float*)wpost, (const float*)wpre,      \
        (float*)xn, (__nv_bfloat16*)x16, n >> 2, eps, s)
    if (accm == PD_ACC_DF) PD_RASN_GO(PD_ACC_DF);
    else if (accm == PD_ACC_F64) PD_RASN_GO(PD_ACC_F64);
    else PD_RASN_GO(PD_ACC_F32);
#undef PD_RASN_GO
    return pd_launch_status();
}

// Slot 866: slot 777 with the mmq-quantized xn (section C) - x += rmsnorm(proj)
// * wpost * s, then yq = quantize_q8_mmq(rmsnorm(x) * wpre); xn's f32 plane only
// when `xn` is set. Bit-identical to rmsnorm_add_scale + rmsnorm_batch +
// quantize_q8_mmq at the rows it accepts (>= 256, the add-scale's vec4 regime)
// and n % 128 == 0; -2 elsewhere, and the caller keeps the chain.
PD_EXPORT
int pd_rmsnorm_add_scale_norm_q8(void* x, const void* proj, const void* wpost, const void* wpre,
                                 void* xn, void* yq, uint32_t n, float eps, float s,
                                 uint32_t rows, void* stream) {
    if (n == 0 || rows == 0) return 0;
    if (rows < 256u || (n & 127u) != 0 || yq == nullptr ||
        (((uintptr_t)x | (uintptr_t)proj | (uintptr_t)wpost | (uintptr_t)wpre |
          (uintptr_t)xn | (uintptr_t)yq) & 15u) != 0)
        return -2;
    const uint32_t nth = pd_norm_wide_nth_ws(rows);
    const uint32_t batch_pad = (rows + 127u) & ~127u;
    const int accm = pd_norm_acc_mode();
    cudaStream_t st = (cudaStream_t)stream;
#define PD_RASNQ_GO(ACC)                                                             \
    pd_rmsnorm_add_scale_norm_kernel<ACC><<<batch_pad, nth, 0, st>>>(                \
        (float*)x, (const float*)proj, (const float*)wpost, (const float*)wpre,      \
        (float*)xn, nullptr, n >> 2, eps, s, (uint8_t*)yq, rows, batch_pad)
    if (accm == PD_ACC_DF) PD_RASNQ_GO(PD_ACC_DF);
    else if (accm == PD_ACC_F64) PD_RASNQ_GO(PD_ACC_F64);
    else PD_RASNQ_GO(PD_ACC_F32);
#undef PD_RASNQ_GO
    return pd_launch_status();
}
