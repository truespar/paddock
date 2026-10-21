// quant/e2m1.cuh - the e2m1 / nvf4 element math every nvf4 producer shares: the RN-even e2m1
// nibble and the per-16 e4m3 block scale. Textually-included segment of the single pack
// translation unit, ahead of every user (gemm/dense_fp4_w8.cuh, gemm/dense_nvf4_tc5.cuh's SWQ
// epilogue, quant/nvf4.cuh); one definition, so the fused epilogue and the standalone quantizers
// cannot drift apart.
#pragma once
#include <cuda_fp8.h>

// f32 -> e2m1 nibble, round-to-nearest-even on the e2m1 grid
// {0, .5, 1, 1.5, 2, 3, 4, 6} (ties resolve toward even mantissa, matching
// IEEE RN semantics). Caller guarantees |v| <= 6 via the block scale.
__device__ __forceinline__ unsigned pd_e2m1_rn(float v) {
    unsigned s = v < 0.0f ? 8u : 0u;
    float a = fabsf(v);
    unsigned m;
    if (a <= 0.25f) m = 0u;            // .25 tie -> 0   (mantissa 0 even)
    else if (a < 0.75f) m = 1u;        // .75 tie -> 1.0 (mantissa 0)
    else if (a <= 1.25f) m = 2u;       // 1.25 tie -> 1.0
    else if (a < 1.75f) m = 3u;        // 1.75 tie -> 2.0
    else if (a <= 2.5f) m = 4u;        // 2.5 tie -> 2.0
    else if (a < 3.5f) m = 5u;         // 3.5 tie -> 4.0
    else if (a <= 5.0f) m = 6u;        // 5.0 tie -> 4.0
    else m = 7u;
    return s | m;
}

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

// pd_nvf4_swz_off (the swizzled scale atoms' byte offset) lives in tma_desc.cuh.
