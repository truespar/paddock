// deltanet/conv_step_qkvn.cuh - the decode DeltaNet front in one launch: the k = 4 causal conv
// step over each slot's window (shifted in place) + silu + the q / k / v split + the q, k l2
// norms + the GQA widening to the value heads. Textually-included segment of the single pack
// translation unit, after deltanet/stage2_sample.cuh (slot 875).
//
// Why. Above the v2f band (b > 24) the decode tick ran conv_step_slots(_s) (conv + silu -> a
// [b][conv_dim] f32 plane) and then pd_deltanet_split_gqa_norm (one 128-thread block a (row,
// value head), re-reading that plane and recomputing each key head's norms three times): two
// launches, two contended kernel boundaries and a 40 KB-a-row round trip on every one of the 48
// DeltaNet layers. This is pd_causal_conv1d_silu_qkv_vl_kernel's shape (blocks [0, HK): q and k
// of key head hk, written to every value head hv = hk (mod HK); blocks [HK, HK + HV): v of head
// hv) with the conv taps coming from the window - each channel's window shifted by exactly one
// block, so no block reads a tap another has overwritten.
//
// Bit-identical to the pair: the conv is pd_conv_step_slots_k4v4_kernel's FMA chain per channel
// (w0 v0, then fmaf with w1 v1, w2 v2, w3 x), the silu its expression, and the norms
// pd_deltanet_split_gqa_norm_kernel's (squares warp-reduced by xor 16..1, the four warp partials
// summed in order, q * rsqrt(qs + 1e-6) * rsqrt(s), k * rsqrt(ks + 1e-6)).

__global__ void pd_conv_step_qkvn_kernel(float* __restrict__ wins, const float* __restrict__ x_new,
                                         const float* __restrict__ w, float* __restrict__ q_out,
                                         float* __restrict__ k_out, float* __restrict__ v_out,
                                         const unsigned int* __restrict__ slots, uint32_t x_stride,
                                         uint32_t n_k_heads, uint32_t n_v_heads, uint32_t s) {
    const uint32_t row = blockIdx.y, j = threadIdx.x;
    const uint32_t key_dim = s * n_k_heads, conv_dim = 2u * key_dim + s * n_v_heads;
    // Only x_new is the predecessor's (the in-proj GEMM): the slot table (uploaded before the
    // tick's first kernel), the window taps (this layer's last tick) and the conv weights are
    // read ahead of the PDL wait - the slot -> window chain was a second round trip after it.
    const bool vb = blockIdx.x >= n_k_heads;
    const uint32_t c0 = vb ? 2u * key_dim + (blockIdx.x - n_k_heads) * s + j : blockIdx.x * s + j;
    const uint32_t c1 = key_dim + blockIdx.x * s + j;   // the k channel (q / k blocks only)
    float* win = wins + (size_t)slots[row] * 3u * conv_dim;
    struct Tap { float4 wc; float v0, v1, v2; };
    auto taps = [&](uint32_t c) {
        return Tap{*reinterpret_cast<const float4*>(w + (size_t)c * 4u), win[c], win[conv_dim + c],
                   win[2u * conv_dim + c]};
    };
    const Tap t0 = taps(c0);
    Tap t1{};
    if (!vb) t1 = taps(c1);
    // wait, and let the decay matvec behind start its own predecessor-free weight stream now
    PD_PDL_ARM();
    const float* xr = x_new + (size_t)row * x_stride;
    auto step = [&](uint32_t c, const Tap& t) {
        const float v3 = xr[c];
        float o = __fmaf_rn(t.wc.w, v3, __fmaf_rn(t.wc.z, t.v2, __fmaf_rn(t.wc.y, t.v1, t.wc.x * t.v0)));
        o = o / (1.0f + expf(-o));
        win[c] = t.v1;
        win[conv_dim + c] = t.v2;
        win[2u * conv_dim + c] = v3;
        return o;
    };
    if (vb) {
        const uint32_t hv = blockIdx.x - n_k_heads;
        v_out[((size_t)row * n_v_heads + hv) * s + j] = step(c0, t0);
        return;
    }
    const uint32_t hk = blockIdx.x;
    const float qj = step(c0, t0), kj = step(c1, t1);
    float q2 = qj * qj, k2 = kj * kj;
#pragma unroll
    for (uint32_t off = 16; off > 0; off >>= 1) {
        q2 += __shfl_xor_sync(0xffffffffu, q2, off);
        k2 += __shfl_xor_sync(0xffffffffu, k2, off);
    }
    __shared__ float sh[8];                       // up to 4 warps x {q2, k2}
    const uint32_t lane = j & 31u, warp = j >> 5, nwarps = (s + 31u) >> 5;
    if (lane == 0) { sh[warp] = q2; sh[4 + warp] = k2; }
    __syncthreads();
    float qs = 0.0f, ks = 0.0f;
    for (uint32_t wi = 0; wi < nwarps; ++wi) { qs += sh[wi]; ks += sh[4 + wi]; }
    const float qn = qj * rsqrtf(qs + 1e-6f) * rsqrtf((float)s);
    const float kn = kj * rsqrtf(ks + 1e-6f);
    for (uint32_t hv = hk; hv < n_v_heads; hv += n_k_heads) {
        const size_t oidx = ((size_t)row * n_v_heads + hv) * s + j;
        q_out[oidx] = qn;
        k_out[oidx] = kn;
    }
}

// Slot 875: (wins, x_new, w, q, k, v, slots, batch, x_stride, HK, HV, s, k, stream). k must be 4
// and s a multiple of 32 up to 128 (the norm's warp partials); x_new rows are x_stride floats
// apart (the fused in-proj plane, conv half first). cudaErrorNotSupported otherwise.
PD_EXPORT
int pd_conv_step_slots_qkvn(void* wins, const void* x_new, const void* w, void* q_out, void* k_out,
                            void* v_out, const void* slots, uint32_t batch, uint32_t x_stride,
                            uint32_t n_k_heads, uint32_t n_v_heads, uint32_t s, uint32_t k,
                            void* stream) {
    if (batch == 0 || n_v_heads == 0 || s == 0) return 0;
    if (k != 4u || s > 128u || (s & 31u) || n_k_heads == 0 || (n_v_heads % n_k_heads) != 0u)
        return cudaErrorNotSupported;
    if (x_stride < (2u * n_k_heads + n_v_heads) * s) return cudaErrorInvalidValue;
    pd_pdl_go(pd_conv_step_qkvn_kernel, dim3(n_k_heads + n_v_heads, batch), dim3(s), 0u,
              (cudaStream_t)stream, (float*)wins, (const float*)x_new, (const float*)w, (float*)q_out,
              (float*)k_out, (float*)v_out, (const unsigned int*)slots, x_stride, n_k_heads, n_v_heads,
              s);
    return pd_launch_status();
}
