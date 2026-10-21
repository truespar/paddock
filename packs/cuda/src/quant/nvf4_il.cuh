// quant/nvf4_il.cuh - the interleaved-plane swiglu + nvf4 quantize (the decode W4A4 FFN's
// middle step over the fused gate|up landing) and its scale-swizzled twin (slot 880). Textually-
// included segment of the single pack translation unit, after quant/nvf4.cuh (pd_nvf4_quant8_at,
// pd_nvf4_swz_off from quant/e2m1.cuh); split out of quant/nvf4.cuh to keep it under its ratchet.

// Interleaved twin of pd_swiglu_fused_nvf4: y row = (g0,u0,
// g1,u1,...) -- Nvf4Plane::gu_pairs. Each thread's 8 outputs are 16
// consecutive floats; same expression, same pd_nvf4_quant8.
// SW (slot 880): the scales in the tcgen05 GEMM's swizzled atom layout (pd_nvf4_swz_off).
template <bool SW = false>
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
    pd_nvf4_quant8_at(v[0], v[1], threadIdx.x & 31u, q, scale, i,
                      SW ? pd_nvf4_swz_off(tok, j >> 4, ff >> 6) : i >> 4);
}
PD_EXPORT
int pd_swiglu_fused_nvf4_il(const void* fused, void* q, void* scale, uint32_t ff,
                            uint32_t n_rows, void* stream) {
    if (n_rows == 0 || ff == 0) return 0;
    if (ff & 31u) return cudaErrorInvalidValue;
    const uint32_t n = n_rows * ff;
    pd_swiglu_fused_nvf4_il_kernel<false><<<(n / 8u + 255u) / 256u, 256u, 0,
                                     (cudaStream_t)stream>>>(
        (const float*)fused, (unsigned char*)q, (unsigned char*)scale, ff, n);
    return pd_launch_status();
}
// Slot 880: pd_swiglu_fused_nvf4_il with the scales swizzled (scale holds ceil(n_rows / 128) * 128
// rows' worth) for the scale-swizzled W4A4 entry (881); ff % 64 == 0.
PD_EXPORT
int pd_swiglu_fused_nvf4_il_sw(const void* fused, void* q, void* scale, uint32_t ff,
                               uint32_t n_rows, void* stream) {
    if (n_rows == 0 || ff == 0) return 0;
    if (ff & 63u) return cudaErrorInvalidValue;
    const uint32_t n = n_rows * ff;
    pd_swiglu_fused_nvf4_il_kernel<true><<<(n / 8u + 255u) / 256u, 256u, 0,
                                           (cudaStream_t)stream>>>(
        (const float*)fused, (unsigned char*)q, (unsigned char*)scale, ff, n);
    return pd_launch_status();
}
