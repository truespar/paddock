// gemm/f16_tower.cuh - a vision tower's FFN up landed in f16 with its bias and
// activation in the GEMM epilogue (slot 940), the exact-GELU and quick-GELU
// twins of slot 834's tanh form. Included right after f16_dense.cuh: it rides
// that file's tile election (pd_f16_tower_fusable, pd_f16_gemm_h_route).
//
// DeepEncoder's two towers ran their FFN ups as pd_f16_gemm -> bias add ->
// activation -> f32->f16 convert: SAM ViT-B's 3072-wide up plane is ~50 MB a
// layer at a 1024 page, written once and read three times before the down
// GEMM ever sees it. Landed here it never exists - the same accumulator, the
// same f32 ops in the same order, one round to f16. Where the fused landing
// would not run pd_f16_gemm's own tile (see pd_f16_tower_fusable) the entry
// runs the plain GEMM into the caller's f32 scratch and one elementwise pass,
// so every shape lands the unfused chain's bits.

// The fallback's elementwise half: y16 = f16(act(x + bias)), act as 940's.
__global__ void pd_bias_act_f16_kernel(const float* __restrict__ x, const float* __restrict__ bias,
                                       __half* __restrict__ y, uint32_t rows, uint32_t n,
                                       uint32_t act) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)rows * n) return;
    const float v = x[i] + bias[i % n];
    y[i] = __float2half(act == 0u ? pd_f16_gelu_erf(v) : pd_f16_quick_gelu(v));
}

// 940: y16 [batch][out_dim] halves = f16(act(W x + bias)); act 0 = exact GELU
// (SAM's nn.GELU), 1 = quick GELU x * sigmoid(1.702 x) (CLIP-L's). bias
// [out_dim] f32 required; scratch an f32 [batch][out_dim] plane, touched only
// where the landing is not fusable. in_dim, out_dim, batch as pd_f16_gemm.
PD_EXPORT int pd_f16_gemm_h_act(const void* w, const void* x, void* y16, const void* bias,
                                void* scratch, unsigned int in_dim, unsigned int out_dim,
                                unsigned int batch, unsigned int act, void* stream) {
    if (out_dim == 0u || batch == 0u) return 0;
    if (bias == nullptr || act > 1u) return (int)cudaErrorInvalidValue;
    const float* b = (const float*)bias;
    if (pd_f16_tower_fusable(in_dim, out_dim, batch)) {
        return act == 0u
                   ? pd_f16_gemm_h_route<PD_F16_EPI_GELU>(w, x, y16, b, in_dim, out_dim, batch,
                                                          stream)
                   : pd_f16_gemm_h_route<PD_F16_EPI_QUICK_GELU>(w, x, y16, b, in_dim, out_dim,
                                                                batch, stream);
    }
    if (scratch == nullptr) return (int)cudaErrorInvalidValue;
    const int r = pd_f16_gemm(w, x, scratch, 0.0f, in_dim, out_dim, batch, stream);
    if (r != 0) return r;
    const uint64_t n = (uint64_t)batch * out_dim;
    pd_bias_act_f16_kernel<<<(uint32_t)((n + 255u) / 256u), 256, 0, (cudaStream_t)stream>>>(
        (const float*)scratch, b, (__half*)y16, batch, out_dim, act);
    return (int)cudaGetLastError();
}
