// sam3/neck.cuh - SAM 3's feature-pyramid necks: the 2x2 / stride-2 transposed conv's depth-to-space with its bias and GELU
// Textually-included segment of the single pack translation unit.
// Not standalone-compilable: include order is defined by ../pack.cu.
// ------------------------------------------------------------ sam3 necks
// Two necks with separate weights read the same 72x72x1024 trunk output (the
// detector's and the tracker's, Meta's Sam3DualViTDetNeck): x4 is
// convT -> GELU -> convT -> 1x1 -> 3x3, x2 convT -> 1x1 -> 3x3, x1 1x1 -> 3x3,
// all at 256 channels out. The 1x1s are plain GEMMs and the 3x3s the dense
// lane's im2row + GEMM (dense_pred.cuh). What is SAM 3's own is the convT
// seam below.
//
// Interim, known short of the target: the 3x3s run on a materialized im2row,
// which at the x4 level is a 288^2 x 2304 f16 plane (382 MB). The implicit-GEMM
// form - the 3x3 taps gathered into the GEMM's own shared-memory staging, no
// plane - is the SOTA conv and the target once the neck is a measurable share
// of a frame.

// 754: out = f16(act(convT2x2s2(x) + bias)). A 2x2 / stride-2 transposed conv
// never overlaps: input pixel (i, j) writes output (2i+ky, 2j+kx) and nothing
// else does, so it is one GEMM C_in -> 4*C_out per input pixel (the loader
// lays the weight tap-major: row (ky*2+kx)*C_out + co) and this depth-to-space
// - the shape 616 has, minus its skip. `g` is the GEMM's f32 landing
// [pics][h*w][4*C]; the output is the next GEMM's f16 input [pics][2h][2w][C],
// raster. gelu != 0 applies the exact (erf) GELU, nn.GELU's default, before
// the one round - the x4 level's first convT is followed by one.
__global__ void pd_sam3_convt2_bias_h_kernel(const float* __restrict__ g,
                                             const float* __restrict__ bias,
                                             __half* __restrict__ out, uint32_t h, uint32_t w,
                                             uint32_t C, uint32_t gelu) {
    const uint32_t o = blockIdx.x;
    const uint32_t W2 = 2u * w, Pout = 4u * h * w;
    const uint32_t pic = o / Pout, pix = o - pic * Pout;
    const uint32_t Y = pix / W2, X = pix - Y * W2;
    const uint32_t tap = (Y & 1u) * 2u + (X & 1u);
    const float* gr = g + (((size_t)pic * h + (Y >> 1)) * w + (X >> 1)) * 4u * C
                      + (size_t)tap * C;
    __half* orow = out + (size_t)o * C;
    for (uint32_t c = threadIdx.x; c < C; c += blockDim.x) {
        float v = gr[c] + bias[c];
        if (gelu) v = 0.5f * v * (1.0f + erff(v * 0.70710678118654752440084436210484f));
        orow[c] = __float2half(v);
    }
}

PD_EXPORT
int pd_sam3_convt2_bias_h(const void* g, const void* bias, void* out, uint32_t pics, uint32_t h,
                          uint32_t w, uint32_t C, uint32_t gelu, void* stream) {
    if (pics == 0 || h == 0 || w == 0 || C == 0) return 0;
    const uint32_t nth = C < 256u ? ((C + 31u) & ~31u) : 256u;
    pd_sam3_convt2_bias_h_kernel<<<pics * 4u * h * w, nth, 0, (cudaStream_t)stream>>>(
        (const float*)g, (const float*)bias, (__half*)out, h, w, C, gelu);
    return pd_launch_status();
}
