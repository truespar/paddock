// quant/nvf4_gs.cuh - the static-scale twin of pd_quantize_nvf4 (slot 971): nvfp4 activations
// quantized against a checkpoint's per-tensor input scale. Textually-included segment of the single
// pack translation unit, after quant/nvf4.cuh (pd_nvf4_scale_gs and pd_e2m1_rn from
// quant/e2m1.cuh); split out of quant/nvf4.cuh to keep it under its ratchet.

// 971: pd_quantize_nvf4 against a static per-tensor scale (pd_nvf4_scale_gs):
// the activation side of a checkpoint calibrated with input scales (Ripple's
// MLP), whose GEMM then carries scale2 * input_scale. Same layout and walk.
__global__ void pd_quantize_nvf4_gs_kernel(const float* __restrict__ x,
                                           unsigned char* __restrict__ q,
                                           unsigned char* __restrict__ scale, uint32_t n,
                                           float gs) {
    PD_PDL_ARM_WAIT();
    const uint32_t i = (blockIdx.x * 256u + threadIdx.x) * 8u;
    if (i >= n) return;  // n % 32 == 0: pair groups exit whole
    const float4 v0 = *(const float4*)(x + i), v1 = *(const float4*)(x + i + 4u);
    float a = fmaxf(fmaxf(fabsf(v0.x), fabsf(v0.y)), fmaxf(fabsf(v0.z), fabsf(v0.w)));
    a = fmaxf(a, fmaxf(fmaxf(fabsf(v1.x), fabsf(v1.y)), fmaxf(fabsf(v1.z), fabsf(v1.w))));
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t gm = 0x3u << (lane & 30u);  // 2-lane group = one 16-block
    a = fmaxf(a, __shfl_xor_sync(gm, a, 1));
    float inv;
    const unsigned sb = pd_nvf4_scale_gs(a, gs, &inv);
    const uint32_t p = pd_e2m1_rn(v0.x * inv) | (pd_e2m1_rn(v0.y * inv) << 4)
                     | (pd_e2m1_rn(v0.z * inv) << 8) | (pd_e2m1_rn(v0.w * inv) << 12)
                     | (pd_e2m1_rn(v1.x * inv) << 16) | (pd_e2m1_rn(v1.y * inv) << 20)
                     | (pd_e2m1_rn(v1.z * inv) << 24) | (pd_e2m1_rn(v1.w * inv) << 28);
    *(uint32_t*)(q + (i >> 1)) = p;
    if ((lane & 1u) == 0) scale[i >> 4] = (unsigned char)sb;
}

PD_EXPORT
int pd_quantize_nvf4_gs(const void* x, void* q, void* scale, uint32_t n, float gs,
                        void* stream) {
    if (n == 0) return 0;
    if ((n & 31u) != 0u || !(gs > 0.0f)) return cudaErrorInvalidValue;
    pd_pdl_go(pd_quantize_nvf4_gs_kernel, (n / 8u + 255u) / 256u, 256u, 0, (cudaStream_t)stream,
        (const float*)x, (unsigned char*)q, (unsigned char*)scale, n, gs);
    return pd_launch_status();
}
