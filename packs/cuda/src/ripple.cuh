// ripple.cuh - Ricursive's Ripple-65B elementwise glue: the FFN's clamped
// SwiGLU and the value-residual blend that feeds a layer's v projection.
// Plain scalar kernels; their consumers are the NVFP4 / bf16 GEMMs around
// them (abi.cuh helpers only).

// slot 968: the clamped SwiGLU, in place on gate:
// gate[i] = silu(min(gate[i], L)) * clamp(up[i], -L, L) - the reference's
// clamp(max=L) on the gate projection and clamp(-L, L) on the up projection,
// before silu(gate) * up.
__global__ void pd_ripple_swiglu_clamp_kernel(float* __restrict__ gate,
                                              const float* __restrict__ up, uint32_t n,
                                              float limit) {
    PD_PDL_ARM();
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = fminf(gate[i], limit);
    float u = fminf(fmaxf(up[i], -limit), limit);
    gate[i] = (g / (1.0f + expf(-g))) * u;
}

PD_EXPORT
int pd_ripple_swiglu_clamp(void* gate, const void* up, uint32_t n, float limit, void* stream) {
    if (n == 0) return 0;
    pd_pdl_go(pd_ripple_swiglu_clamp_kernel, (n + 255u) / 256u, 256, 0u, (cudaStream_t)stream,
              (float*)gate, (const float*)up, n, limit);
    return pd_launch_status();
}

// slot 969: the value-residual blend, out = e + t * (x - e) over n elements
// (torch.lerp(e, x, t)): e the scaled token embedding (the layer-0 input), x
// the layer's normed input, t = sigmoid(lambda) of the layer.
__global__ void pd_ripple_valres_mix_kernel(float* __restrict__ out, const float* __restrict__ e,
                                            const float* __restrict__ x, uint32_t n, float t) {
    PD_PDL_ARM();
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float a = e[i];
    out[i] = a + t * (x[i] - a);
}

PD_EXPORT
int pd_ripple_valres_mix(void* out, const void* e, const void* x, uint32_t n, float t,
                         void* stream) {
    if (n == 0) return 0;
    pd_pdl_go(pd_ripple_valres_mix_kernel, (n + 255u) / 256u, 256, 0u, (cudaStream_t)stream,
              (float*)out, (const float*)e, (const float*)x, n, t);
    return pd_launch_status();
}
