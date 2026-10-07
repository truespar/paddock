// Mamba-2 (SSD) family kernels - first user: nemotron_h_moe (
// NVIDIA Nemotron 3.5 Lightning 30B-A3B). The GDN lane (deltanet/) is the
// structural sibling but its scan is the rank-1 delta rule over a SQUARE
// [H][D][D] state; Mamba-2's state is [H][head_dim][d_state] (64x64x128
// here) with a scalar-per-head decay and a diagonal-plus-outer-product
// update - cheaper elementwise compose, different geometry, so it gets its
// own kernels rather than arms on the DN ones.
//
// Reference semantics pinned from vLLM v0.27.1 MambaMixer2 + the llama.cpp
// b10410 graph:
//   in_proj rows = [z (d_inner) | x B C (conv_dim) | dt (n_heads)]
//   conv1d over conv_dim channels with bias, SiLU on the whole span
//   dt = softplus(dt_raw + dt_bias)  (threshold 20, the Triton convention;
//        dt_limit is (0, inf) i.e. positivity only)
//   decay a_h = exp(dt * A_h), A stored as -exp(A_log) (transformed at load)
//   S[h,i,:] = a_h * S[h,i,:] + (dt * x[h,i]) * B[g(h),:]
//   y[h,i]   = S[h,i,:] . C[g(h),:] + D_h * x[h,i]
//   group broadcast is repeat_interleave: g(h) = h / (H/G) - Not the GDN
//   split kernels' tiling (h % G), which is silently wrong for G > 1.
//   gated norm: x*silu(z) first (f32), then RMS variance per GROUP of
//   d_inner/G channels, then per-channel weight [d_inner].

// Single-token (or short-span) conv step with a persistent window and a
// bias term - pd_conv_step + bias. One thread per channel; window layout
// [k-1, conv_dim] oldest-first, pre-conv rows (same contract as GDN).
__global__ void pd_mamba_conv_step_kernel(
        float* __restrict__ win, const float* __restrict__ x_new,
        const float* __restrict__ w, const float* __restrict__ b,
        float* __restrict__ out, uint32_t conv_dim, uint32_t k) {
    uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    uint32_t km1 = k - 1u;
    float vals[PD_CONV_K_MAX];
    for (uint32_t j = 0; j < km1; ++j) vals[j] = win[(size_t)j * conv_dim + c];
    vals[km1] = x_new[c];
    float acc = b[c];
    for (uint32_t j = 0; j < k; ++j) acc += w[(size_t)c * k + j] * vals[j];
    out[c] = acc / (1.0f + expf(-acc));
    for (uint32_t j = 1; j < km1; ++j) win[(size_t)(j - 1) * conv_dim + c] = vals[j];
    if (km1 >= 1u) win[(size_t)(km1 - 1) * conv_dim + c] = vals[km1];
}

PD_EXPORT
int pd_mamba_conv_step(void* win, const void* x_new, const void* w, const void* b,
                       void* out, uint32_t conv_dim, uint32_t k, void* stream) {
    if (conv_dim == 0) return 0;
    if (k > PD_CONV_K_MAX) return cudaErrorInvalidValue;
    uint32_t threads = 256;
    uint32_t blocks = (conv_dim + threads - 1) / threads;
    pd_mamba_conv_step_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        (float*)win, (const float*)x_new, (const float*)w, (const float*)b,
        (float*)out, conv_dim, k);
    return pd_launch_status();
}

// Bulk causal conv over a token span with the same persistent window - the
// step kernel's math T times in one launch (bulk prefill). The
// k-1 deep window rides registers/local across the whole span and is
// written back once at exit; identical FMA order per token to
// pd_mamba_conv_step_kernel, so bulk-vs-serial stays bit-exact. Reads the
// x|B|C span straight from the fused in_proj output rows via
// x_off/x_stride (no staging copy of [T, conv_dim]).
__global__ void pd_mamba_conv_seq_kernel(
        float* __restrict__ win, const float* __restrict__ xbc,
        uint32_t x_off, uint32_t x_stride, const float* __restrict__ w,
        const float* __restrict__ b, float* __restrict__ out,
        uint32_t conv_dim, uint32_t k, uint32_t n_tokens) {
    uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    uint32_t km1 = k - 1u;
    float vals[PD_CONV_K_MAX];
    for (uint32_t j = 0; j < km1; ++j) vals[j] = win[(size_t)j * conv_dim + c];
    float wc[PD_CONV_K_MAX];
    for (uint32_t j = 0; j < k; ++j) wc[j] = w[(size_t)c * k + j];
    const float bc = b[c];
    for (uint32_t t = 0; t < n_tokens; ++t) {
        vals[km1] = xbc[(size_t)t * x_stride + x_off + c];
        float acc = bc;
        for (uint32_t j = 0; j < k; ++j) acc += wc[j] * vals[j];
        out[(size_t)t * conv_dim + c] = acc / (1.0f + expf(-acc));
        for (uint32_t j = 0; j < km1; ++j) vals[j] = vals[j + 1];
    }
    for (uint32_t j = 0; j < km1; ++j) win[(size_t)j * conv_dim + c] = vals[j];
}

// The same conv over a span cut into TCH-token chunks along T (grid.y): a
// chunk's k-1 deep halo is the previous chunk's last k-1 inputs, which are
// plain rows of xbc (chunk 0 takes them from `win`), and only the last chunk
// writes the window back. Every output token runs the serial kernel's FMA
// sequence verbatim (same wc[j] * vals[j] order, same bias seed), so
// chunked-vs-serial is bit-exact - the halo is data the serial walk would
// have shifted through `vals` anyway. Why: the serial kernel is 24 CTAs of
// 256 threads on a 6144-channel plane (grid = conv_dim / 256), each thread
// walking the whole span - on GB10's 48 SMs a 1022-row span cost 75-190 us
// per launch and 172 launches per 1k prefill request (12.9 ms, vLLM's
// triton conv 2.2), half the die idle and every thread latency-bound on
// its own dependent load chain. TCH = 64 puts 16 chunks x 24 = 384 CTAs on
// the span.
template <uint32_t TCH>
__global__ void pd_mamba_conv_seq_chunked_kernel(
        float* __restrict__ win, const float* __restrict__ xbc,
        uint32_t x_off, uint32_t x_stride, const float* __restrict__ w,
        const float* __restrict__ b, float* __restrict__ out,
        uint32_t conv_dim, uint32_t k, uint32_t n_tokens) {
    uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    const uint32_t km1 = k - 1u;
    const uint32_t t0 = blockIdx.y * TCH;
    if (t0 >= n_tokens) return;
    const uint32_t t1 = (t0 + TCH < n_tokens) ? t0 + TCH : n_tokens;
    float vals[PD_CONV_K_MAX];
    // halo: input rows t0-km1 .. t0-1; rows before the span come from the
    // carried window (win[j] is the j-th oldest of the km1 inputs before
    // row 0, exactly what the serial kernel loads)
    for (uint32_t j = 0; j < km1; ++j) {
        const int32_t tr = (int32_t)t0 - (int32_t)km1 + (int32_t)j;
        vals[j] = tr >= 0 ? xbc[(size_t)tr * x_stride + x_off + c]
                          : win[(size_t)(tr + (int32_t)km1) * conv_dim + c];
    }
    float wc[PD_CONV_K_MAX];
    for (uint32_t j = 0; j < k; ++j) wc[j] = w[(size_t)c * k + j];
    const float bc = b[c];
    for (uint32_t t = t0; t < t1; ++t) {
        vals[km1] = xbc[(size_t)t * x_stride + x_off + c];
        float acc = bc;
        for (uint32_t j = 0; j < k; ++j) acc += wc[j] * vals[j];
        out[(size_t)t * conv_dim + c] = acc / (1.0f + expf(-acc));
        for (uint32_t j = 0; j < km1; ++j) vals[j] = vals[j + 1];
    }
    if (t1 == n_tokens)
        for (uint32_t j = 0; j < km1; ++j) win[(size_t)j * conv_dim + c] = vals[j];
}

// Chunk width and the election floor for the chunked conv: spans of at
// least 2 x PD_CONV_TCH tokens chunk (a one-chunk launch is the serial
// kernel with an extra halo read); PADDOCK_NO_CONV_CHUNK pins the serial
// walk for the A/B.
#define PD_CONV_TCH 64u

PD_EXPORT
int pd_mamba_conv_seq(void* win, const void* xbc, uint32_t x_off,
                      uint32_t x_stride, const void* w, const void* b,
                      void* out, uint32_t conv_dim, uint32_t k,
                      uint32_t n_tokens, void* stream) {
    if (conv_dim == 0 || n_tokens == 0) return 0;
    if (k > PD_CONV_K_MAX || k == 0) return cudaErrorInvalidValue;
    uint32_t threads = 256;
    uint32_t blocks = (conv_dim + threads - 1) / threads;
    static const bool no_chunk = pd_env("PADDOCK_NO_CONV_CHUNK") != nullptr;
    if (!no_chunk && n_tokens >= 2u * PD_CONV_TCH) {
        dim3 grid(blocks, (n_tokens + PD_CONV_TCH - 1u) / PD_CONV_TCH);
        pd_mamba_conv_seq_chunked_kernel<PD_CONV_TCH>
            <<<grid, threads, 0, (cudaStream_t)stream>>>(
                (float*)win, (const float*)xbc, x_off, x_stride, (const float*)w,
                (const float*)b, (float*)out, conv_dim, k, n_tokens);
        return pd_launch_status();
    }
    pd_mamba_conv_seq_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        (float*)win, (const float*)xbc, x_off, x_stride, (const float*)w,
        (const float*)b, (float*)out, conv_dim, k, n_tokens);
    return pd_launch_status();
}

// Sequential SSD scan over a token span, one launch: state rides registers
// for the whole span (the walk_rs lesson - state touched in global only at
// entry/exit), tokens advance in-kernel. Thread (h, i) owns state row
// S[h][i][0..S_-1]; y needs no cross-thread reduce because the C dot runs
// along the register row. B/C for the block's (single) group stage through
// shared per token. Correctness shape - the chunked/occupancy rung comes
// after the parity gate, profiler-guided.
//
// xbc rows are the conv OUTPUT [T, conv_dim] = [x (H*hd) | B (G*S) | C (G*S)];
// dt_raw rides its caller stride (it lives inside the in_proj output rows).
#define PD_M2_SOFTPLUS_LIM 20.0f

// Chunked SSD runner (mamba/ssd.cuh, defined later in this TU): the seq
// launchers below elect it for long segments on the pinned geometry.
static int pd_mamba2_ssd_run(void* state, int state_f16, const void* xbc,
                             const void* dt_raw, uint32_t dt_stride,
                             const void* A, const void* D,
                             const void* dt_bias, void* y, uint32_t n_tokens,
                             uint32_t n_heads, uint32_t n_groups,
                             void* stream);

// SSD election, shared by the f32 and f16 seq launchers: the chunked form
// pays only when the segment amortizes its 5-launch chain, and its kernels
// pin head_dim 64 / d_state 128 (the nemotron geometry). PADDOCK_NO_SSD_SEQ
// = dev A/B kill switch (pd_env; hardened builds seal the environment) -
// the engagement witness is the pd_ssd_* kernel names in a capture.
static inline bool pd_m2_ssd_elect(uint32_t n_tokens, uint32_t n_heads,
                                   uint32_t head_dim, uint32_t d_state,
                                   uint32_t n_groups) {
    static const bool off = pd_env_on("PADDOCK_NO_SSD_SEQ");
    return !off && n_tokens >= 256u /* PD_SSD_MIN_T, measured crossover */ &&
           head_dim == 64u && d_state == 128u && n_groups > 0 &&
           n_heads % n_groups == 0;
}
// HD_ = compile-time head_dim (0 = use the runtime arg). The [h, S, i]
// layout makes the j walk a runtime-strided pointer; with a runtime stride
// the compiler cannot prove srow stores distinct from later srow loads and
// SERIALIZES the unrolled loop (measured: nemotron c1 270 -> 229 out tok/s
// before the fold). A constant stride restores provably-distinct offsets.
template <uint32_t S_, uint32_t HD_ = 0u>
__global__ void __launch_bounds__(128, 1) pd_mamba2_scan_seq_kernel(
        float* __restrict__ state, const float* __restrict__ xbc,
        const float* __restrict__ dt_raw, uint32_t dt_stride,
        const float* __restrict__ A, const float* __restrict__ D,
        const float* __restrict__ dt_bias, float* __restrict__ y,
        uint32_t n_tokens, uint32_t n_heads, uint32_t head_dim_rt,
        uint32_t n_groups) {
    const uint32_t head_dim = HD_ ? HD_ : head_dim_rt;
    const uint32_t hpb = blockDim.x / head_dim;       // heads per block
    const uint32_t h = blockIdx.x * hpb + threadIdx.x / head_dim;
    const uint32_t i = threadIdx.x % head_dim;
    const uint32_t d_inner = n_heads * head_dim;
    const uint32_t conv_dim = d_inner + 2u * n_groups * S_;
    const uint32_t g = h / (n_heads / n_groups);      // repeat_interleave
    __shared__ float sB[S_], sC[S_];

    // Load this row's state slice. Plain unrolled scalars: every j is a
    // compile-time index after the unroll, which is what keeps st[] in
    // registers - a float4 pun would take the array's address and demote
    // it to local memory.
    //
    // State layout is [h, S, i] - i (head_dim) minor, not S. Thread (h,i)
    // still owns the whole S row (same serial FMA order, bit-exact), but a
    // warp's lanes are CONSECUTIVE floats at every j: the [h, i, S] layout
    // put lanes 512 B apart and the batched decode kernel measured at half
    // the DRAM roof (nemotron c32 - vLLM's triton SSU walks
    // dstate across lanes for the same reason).
    float st[S_];
    float* srow = state + (size_t)h * head_dim * S_ + i;
    #pragma unroll
    for (uint32_t j = 0; j < S_; ++j) st[j] = srow[(size_t)j * head_dim];

    const float a_h = A[h], d_h = D[h], db_h = dt_bias[h];
    for (uint32_t t = 0; t < n_tokens; ++t) {
        const float* row = xbc + (size_t)t * conv_dim;
        if (threadIdx.x < S_) {
            sB[threadIdx.x] = row[d_inner + (size_t)g * S_ + threadIdx.x];
            sC[threadIdx.x] = row[d_inner + (size_t)(n_groups + g) * S_ + threadIdx.x];
        }
        __syncthreads();
        float dt = dt_raw[(size_t)t * dt_stride + h] + db_h;
        dt = (dt <= PD_M2_SOFTPLUS_LIM) ? log1pf(expf(dt)) : dt;
        const float decay = expf(dt * a_h);
        const float x_ti = row[(size_t)h * head_dim + i];
        const float contrib = dt * x_ti;
        float acc = 0.0f;
        #pragma unroll
        for (uint32_t j = 0; j < S_; ++j) {
            const float s = decay * st[j] + contrib * sB[j];
            st[j] = s;
            acc += s * sC[j];
        }
        y[(size_t)t * d_inner + (size_t)h * head_dim + i] = acc + d_h * x_ti;
        __syncthreads();  // before the next token overwrites sB/sC
    }

    #pragma unroll
    for (uint32_t j = 0; j < S_; ++j) srow[(size_t)j * head_dim] = st[j];
}

PD_EXPORT
int pd_mamba2_scan_seq(void* state, const void* xbc, const void* dt_raw,
                       uint32_t dt_stride, const void* A, const void* D,
                       const void* dt_bias, void* y, uint32_t n_tokens,
                       uint32_t n_heads, uint32_t head_dim, uint32_t d_state,
                       uint32_t n_groups, void* stream) {
    if (n_tokens == 0) return 0;
    if (d_state != 128u) return cudaErrorInvalidValue;  // template pin; widen on demand
    if (head_dim == 0 || 128u % head_dim != 0) return cudaErrorInvalidValue;
    const uint32_t hpb = 128u / head_dim;
    // every head in a block must share its B/C group (one smem stage)
    if (n_heads % hpb != 0 || n_groups == 0 || n_heads % n_groups != 0 ||
        (n_heads / n_groups) % hpb != 0)
        return cudaErrorInvalidValue;
    if (pd_m2_ssd_elect(n_tokens, n_heads, head_dim, d_state, n_groups)) {
        const int r = pd_mamba2_ssd_run(state, 0, xbc, dt_raw, dt_stride, A,
                                        D, dt_bias, y, n_tokens, n_heads,
                                        n_groups, stream);
        if (r >= 0) return r;
        // stream-ordered scratch alloc failed - serial walk below
    }
    if (head_dim == 64u)
        pd_mamba2_scan_seq_kernel<128u, 64u>
            <<<n_heads / hpb, 128u, 0, (cudaStream_t)stream>>>(
                (float*)state, (const float*)xbc, (const float*)dt_raw,
                dt_stride, (const float*)A, (const float*)D,
                (const float*)dt_bias, (float*)y, n_tokens, n_heads, head_dim,
                n_groups);
    else
        pd_mamba2_scan_seq_kernel<128u>
            <<<n_heads / hpb, 128u, 0, (cudaStream_t)stream>>>(
                (float*)state, (const float*)xbc, (const float*)dt_raw,
                dt_stride, (const float*)A, (const float*)D,
                (const float*)dt_bias, (float*)y, n_tokens, n_heads, head_dim,
                n_groups);
    return pd_launch_status();
}

// Verify-round scan (spec core): the identical register-resident
// walk as pd_mamba2_scan_seq, plus a per-row state snapshot - after row t
// the register state also stores to snap[t] (the qwen35 recur_snap design:
// the recurrence is not idempotent, so a partial spec accept rolls the live
// state back to the accepted row's snapshot; a full accept needs no
// rollback because the live state already ended there). The extra work is
// stores only - the walk itself is bit-identical to the plain kernel.
template <uint32_t S_, uint32_t HD_ = 0u>
__global__ void __launch_bounds__(128, 1) pd_mamba2_scan_seq_snap_kernel(
        float* __restrict__ state, const float* __restrict__ xbc,
        const float* __restrict__ dt_raw, uint32_t dt_stride,
        const float* __restrict__ A, const float* __restrict__ D,
        const float* __restrict__ dt_bias, float* __restrict__ y,
        float* __restrict__ snap, uint32_t n_tokens, uint32_t n_heads,
        uint32_t head_dim_rt, uint32_t n_groups) {
    const uint32_t head_dim = HD_ ? HD_ : head_dim_rt;
    const uint32_t hpb = blockDim.x / head_dim;
    const uint32_t h = blockIdx.x * hpb + threadIdx.x / head_dim;
    const uint32_t i = threadIdx.x % head_dim;
    const uint32_t d_inner = n_heads * head_dim;
    const uint32_t conv_dim = d_inner + 2u * n_groups * S_;
    const uint32_t g = h / (n_heads / n_groups);
    __shared__ float sB[S_], sC[S_];

    float st[S_];
    float* srow = state + (size_t)h * head_dim * S_ + i;  // [h, S, i] layout
    #pragma unroll
    for (uint32_t j = 0; j < S_; ++j) st[j] = srow[(size_t)j * head_dim];

    const float a_h = A[h], d_h = D[h], db_h = dt_bias[h];
    for (uint32_t t = 0; t < n_tokens; ++t) {
        const float* row = xbc + (size_t)t * conv_dim;
        if (threadIdx.x < S_) {
            sB[threadIdx.x] = row[d_inner + (size_t)g * S_ + threadIdx.x];
            sC[threadIdx.x] = row[d_inner + (size_t)(n_groups + g) * S_ + threadIdx.x];
        }
        __syncthreads();
        float dt = dt_raw[(size_t)t * dt_stride + h] + db_h;
        dt = (dt <= PD_M2_SOFTPLUS_LIM) ? log1pf(expf(dt)) : dt;
        const float decay = expf(dt * a_h);
        const float x_ti = row[(size_t)h * head_dim + i];
        const float contrib = dt * x_ti;
        float acc = 0.0f;
        #pragma unroll
        for (uint32_t j = 0; j < S_; ++j) {
            const float s = decay * st[j] + contrib * sB[j];
            st[j] = s;
            acc += s * sC[j];
        }
        y[(size_t)t * d_inner + (size_t)h * head_dim + i] = acc + d_h * x_ti;
        // snap blob rows carry the same [h, S, i] layout as the live arena,
        // so a partial-accept rollback stays a flat copy
        float* snrow = snap + (size_t)t * d_inner * S_
                       + (size_t)h * head_dim * S_ + i;
        #pragma unroll
        for (uint32_t j = 0; j < S_; ++j) snrow[(size_t)j * head_dim] = st[j];
        __syncthreads();
    }

    #pragma unroll
    for (uint32_t j = 0; j < S_; ++j) srow[(size_t)j * head_dim] = st[j];
}

PD_EXPORT
int pd_mamba2_scan_seq_snap(void* state, const void* xbc, const void* dt_raw,
                            uint32_t dt_stride, const void* A, const void* D,
                            const void* dt_bias, void* y, void* snap,
                            uint32_t n_tokens, uint32_t n_heads,
                            uint32_t head_dim, uint32_t d_state,
                            uint32_t n_groups, void* stream) {
    if (n_tokens == 0) return 0;
    if (d_state != 128u) return cudaErrorInvalidValue;
    if (head_dim == 0 || 128u % head_dim != 0) return cudaErrorInvalidValue;
    const uint32_t hpb = 128u / head_dim;
    if (n_heads % hpb != 0 || n_groups == 0 || n_heads % n_groups != 0 ||
        (n_heads / n_groups) % hpb != 0)
        return cudaErrorInvalidValue;
    if (head_dim == 64u)
        pd_mamba2_scan_seq_snap_kernel<128u, 64u>
            <<<n_heads / hpb, 128u, 0, (cudaStream_t)stream>>>(
                (float*)state, (const float*)xbc, (const float*)dt_raw,
                dt_stride, (const float*)A, (const float*)D,
                (const float*)dt_bias, (float*)y, (float*)snap, n_tokens,
                n_heads, head_dim, n_groups);
    else
        pd_mamba2_scan_seq_snap_kernel<128u>
            <<<n_heads / hpb, 128u, 0, (cudaStream_t)stream>>>(
                (float*)state, (const float*)xbc, (const float*)dt_raw,
                dt_stride, (const float*)A, (const float*)D,
                (const float*)dt_bias, (float*)y, (float*)snap, n_tokens,
                n_heads, head_dim, n_groups);
    return pd_launch_status();
}

// Strided-rows copy: dst[r][0..len) = src[src_off + r*src_stride ..+len).
// The verify round snapshots each mamba layer's conv-input rows (the xBC
// span living inside the fused in_proj output at stride in_proj_rows) so a
// partial accept can rebuild the conv window without replaying the layer.
__global__ void pd_copy_rows_strided_kernel(
        const float* __restrict__ src, uint32_t src_off, uint32_t src_stride,
        float* __restrict__ dst, uint32_t len, uint32_t rows) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)rows * len) return;
    const uint32_t r = (uint32_t)(i / len), c = (uint32_t)(i % len);
    dst[(size_t)r * len + c] = src[(size_t)r * src_stride + src_off + c];
}

PD_EXPORT
int pd_copy_rows_strided(const void* src, uint32_t src_off, uint32_t src_stride,
                         void* dst, uint32_t len, uint32_t rows, void* stream) {
    if (len == 0 || rows == 0) return 0;
    const size_t n = (size_t)rows * len;
    const uint32_t blocks = (uint32_t)((n + 255) / 256);
    pd_copy_rows_strided_kernel<<<blocks, 256, 0, (cudaStream_t)stream>>>(
        (const float*)src, src_off, src_stride, (float*)dst, len, rows);
    return pd_launch_status();
}

// Grouped gated RMSNorm (vLLM Mixer2RMSNormGated forward_native, the
// n_groups > 1 branch): gate first in f32 (x * silu(z)), variance over each
// group of d/n_groups channels, then the per-CHANNEL weight [d]. One block
// per (token, group); blockDim = group size (a power of two <= 1024).
__global__ void pd_mamba_rmsnorm_gated_g_kernel(
        const float* __restrict__ x, const float* __restrict__ z,
        uint32_t z_stride, const float* __restrict__ weight,
        float* __restrict__ out, uint32_t n_groups, uint32_t d, float eps) {
    const uint32_t t = blockIdx.x / n_groups;
    const uint32_t grp = blockIdx.x % n_groups;
    const uint32_t gsize = d / n_groups;
    const uint32_t j = threadIdx.x;
    if (j >= gsize) return;
    extern __shared__ float sm[];
    const uint32_t c = grp * gsize + j;
    const float xv = x[(size_t)t * d + c];
    const float zv = z[(size_t)t * z_stride + c];
    const float gated = xv * (zv / (1.0f + expf(-zv)));
    sm[j] = gated * gated;
    __syncthreads();
    for (uint32_t s = gsize >> 1; s > 0; s >>= 1) {
        if (j < s) sm[j] += sm[j + s];
        __syncthreads();
    }
    const float inv = rsqrtf(sm[0] / (float)gsize + eps);
    out[(size_t)t * d + c] = gated * inv * weight[c];
}

PD_EXPORT
int pd_mamba_rmsnorm_gated_g(const void* x, const void* z, uint32_t z_stride,
                             const void* weight, void* out, uint32_t n_tokens,
                             uint32_t d, uint32_t n_groups, float eps,
                             void* stream) {
    if (n_tokens == 0) return 0;
    if (n_groups == 0 || d % n_groups != 0) return cudaErrorInvalidValue;
    const uint32_t gsize = d / n_groups;
    if (gsize > 1024u || (gsize & (gsize - 1u)) != 0) return cudaErrorInvalidValue;
    pd_mamba_rmsnorm_gated_g_kernel
        <<<n_tokens * n_groups, gsize, gsize * sizeof(float), (cudaStream_t)stream>>>(
            (const float*)x, (const float*)z, z_stride, (const float*)weight,
            (float*)out, n_groups, d, eps);
    return pd_launch_status();
}

// GEMV over a checkpoint FP8 plane held as e4m3 bytes + one f32 scale per
// output row (the modelopt per-TENSOR weight_scale broadcast into the row
// array - byte-exact, no requantization; mamba in/out_proj
// consumer). nvf4_gemv's warp-coherent geometry at 1 B/element: lane l owns
// elements k0+4l..k0+4l+3, so each 128-element step is one contiguous
// 128 B weight load + one contiguous 512 B x load per warp.
__global__ void pd_f8r_gemv_kernel(
        const uint8_t* __restrict__ data, const float* __restrict__ rscale,
        const float* __restrict__ x, float* __restrict__ y, uint32_t in_dim,
        uint32_t out_dim) {
    // Split PDL arm (split-arm shape): WAIT at the top without triggering, and
    // RELEASE once this CTA's own loads are issued. The plain-launch form is
    // a cascade BREAK - a kernel that replaces one must not trigger at top, or
    // every downstream prologue is released at trigger speed. Both PTX ops are
    // no-ops under a non-PDL launch, so arming the body costs nothing to the
    // lanes that keep the plain launcher.
    PD_PDL_ARM_WAIT();
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t o = blockIdx.x * (blockDim.x >> 5) + warp;
    if (o >= out_dim) return;
    const uint8_t* row = data + (size_t)o * in_dim;
    float acc = 0.0f;
    // ragged tail hoisted out of the walk (nvf4.cuh precedent +
    //  probed fact: an in-loop per-lane break blocks ptxas load
    // pipelining, -24% on the nvf4 twin; out_proj's K=8256 has a live
    // 64-wide tail here, in_proj's K=2688 never fired the branch)
    #define PD_F8R_STEP(e)                                                     \
        {                                                                      \
            const uint32_t wb = *reinterpret_cast<const uint32_t*>(row + (e)); \
            const float4 xv = *reinterpret_cast<const float4*>(x + (e));       \
            const __nv_fp8_e4m3* eb =                                          \
                reinterpret_cast<const __nv_fp8_e4m3*>(&wb);                   \
            acc += (float)eb[0] * xv.x + (float)eb[1] * xv.y                   \
                 + (float)eb[2] * xv.z + (float)eb[3] * xv.w;                  \
        }
    const uint32_t full = in_dim & ~127u;
    #pragma unroll 4
    for (uint32_t k0 = 0; k0 < full; k0 += 128u) PD_F8R_STEP(k0 + lane * 4u)
    if (full < in_dim) {
        const uint32_t e = full + lane * 4u;
        if (e < in_dim) PD_F8R_STEP(e)
    }
    #undef PD_F8R_STEP
    // late release: this CTA's weight/x loads are all issued, so successors
    // may start their prologues now (the split-arm half law).
    PD_PDL_RELEASE();
    for (uint32_t s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, s);
    if (lane == 0) y[o] = acc * rscale[o];
}

PD_EXPORT
int pd_f8r_gemv(const void* data, const void* rscale, const void* x, void* y,
                uint32_t in_dim, uint32_t out_dim, void* stream) {
    if (out_dim == 0) return 0;
    if ((in_dim & 3u) != 0) return cudaErrorInvalidValue;
    const uint32_t rows_per_cta = 8u;
    const uint32_t grid = (out_dim + rows_per_cta - 1u) / rows_per_cta;
    // Opt-in cascade entry (PADDOCK_F8R_PDL): a dense lane that issues this
    // GEMV hundreds of times per token loses the dependent-launch overlap its
    // bf16 twin has, and the measured penalty (qwen4_exp) was
    // ~5.8 us per launch - far more than the 12% kernel-time the narrower
    // plane buys. Default stays the plain launch so the mamba/granite lanes
    // that elected this kernel keep their stamped numbers.
    static const bool pdl = pd_env("PADDOCK_F8R_PDL") != nullptr;
    if (pdl) {
        pd_pdl_go(pd_f8r_gemv_kernel, grid, rows_per_cta * 32u, 0u,
                  (cudaStream_t)stream,
            (const uint8_t*)data, (const float*)rscale, (const float*)x,
            (float*)y, in_dim, out_dim);
    } else {
        pd_f8r_gemv_kernel<<<grid, rows_per_cta * 32u, 0, (cudaStream_t)stream>>>(
            (const uint8_t*)data, (const float*)rscale, (const float*)x, (float*)y,
            in_dim, out_dim);
    }
    return pd_launch_status();
}

// ---- batched decode steps over slot arenas (stage A) -----
// The serial step/scan kernels above are single-sequence (one window, one
// state). The continuous-batching tick drives B rows, each advancing its
// own slot's persistent state via a d_slots indirection - the same contract
// kv_append_batch/attn_prefill already carry. Per-row math is the serial
// kernel verbatim (same FMA order), so batched-vs-serial stays bit-exact.

// Windows arena [n_slots, k-1, conv_dim]; x rows [batch, x_stride] with the
// conv span at x_off (the in_proj output's [z | x B C | dt] layout); out
// [batch, conv_dim]. Row r advances slot slots[r].
__global__ void pd_mamba_conv_step_batch_kernel(
        float* __restrict__ win, const float* __restrict__ x, uint32_t x_off,
        uint32_t x_stride, const uint32_t* __restrict__ slots,
        const float* __restrict__ w, const float* __restrict__ b,
        float* __restrict__ out, uint32_t conv_dim, uint32_t k) {
    const uint32_t r = blockIdx.y;
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    const uint32_t km1 = k - 1u;
    float* wr = win + (size_t)slots[r] * km1 * conv_dim;
    float vals[PD_CONV_K_MAX];
    for (uint32_t j = 0; j < km1; ++j) vals[j] = wr[(size_t)j * conv_dim + c];
    vals[km1] = x[(size_t)r * x_stride + x_off + c];
    float acc = b[c];
    for (uint32_t j = 0; j < k; ++j) acc += w[(size_t)c * k + j] * vals[j];
    out[(size_t)r * conv_dim + c] = acc / (1.0f + expf(-acc));
    for (uint32_t j = 1; j < km1; ++j) wr[(size_t)(j - 1) * conv_dim + c] = vals[j];
    if (km1 >= 1u) wr[(size_t)(km1 - 1) * conv_dim + c] = vals[km1];
}

PD_EXPORT
int pd_mamba_conv_step_batch(void* win, const void* x, uint32_t x_off,
                             uint32_t x_stride, const void* slots,
                             const void* w, const void* b, void* out,
                             uint32_t conv_dim, uint32_t k, uint32_t batch,
                             void* stream) {
    if (conv_dim == 0 || batch == 0) return 0;
    if (k > PD_CONV_K_MAX || k == 0) return cudaErrorInvalidValue;
    dim3 grid((conv_dim + 255u) / 256u, batch);
    pd_mamba_conv_step_batch_kernel<<<grid, 256u, 0, (cudaStream_t)stream>>>(
        (float*)win, (const float*)x, x_off, x_stride, (const uint32_t*)slots,
        (const float*)w, (const float*)b, (float*)out, conv_dim, k);
    return pd_launch_status();
}

// Single-token SSD scan step, batched: states arena [n_slots, n_heads,
// head_dim, S]; xbc = conv OUTPUT rows [batch, conv_dim]; dt_raw rows
// [batch, dt_stride] (caller applies dt_off on the base pointer, the serial
// wrapper's convention); y [batch, d_inner]. One token means no in-kernel
// span loop - state reads/writes go straight through global (the register
// residency that pays on seq spans has nothing to amortize here).
template <uint32_t S_, uint32_t HD_ = 0u>
__global__ void __launch_bounds__(128, 1) pd_mamba2_scan_step_batch_kernel(
        float* __restrict__ state, const float* __restrict__ xbc,
        const float* __restrict__ dt_raw, uint32_t dt_stride,
        const uint32_t* __restrict__ slots, const float* __restrict__ A,
        const float* __restrict__ D, const float* __restrict__ dt_bias,
        float* __restrict__ y, uint32_t n_heads, uint32_t head_dim_rt,
        uint32_t n_groups) {
    const uint32_t head_dim = HD_ ? HD_ : head_dim_rt;
    const uint32_t r = blockIdx.y;
    const uint32_t hpb = blockDim.x / head_dim;
    const uint32_t h = blockIdx.x * hpb + threadIdx.x / head_dim;
    const uint32_t i = threadIdx.x % head_dim;
    const uint32_t d_inner = n_heads * head_dim;
    const uint32_t conv_dim = d_inner + 2u * n_groups * S_;
    const uint32_t g = h / (n_heads / n_groups);  // repeat_interleave
    __shared__ float sB[S_], sC[S_];
    const float* row = xbc + (size_t)r * conv_dim;
    if (threadIdx.x < S_) {
        sB[threadIdx.x] = row[d_inner + (size_t)g * S_ + threadIdx.x];
        sC[threadIdx.x] = row[d_inner + (size_t)(n_groups + g) * S_ + threadIdx.x];
    }
    __syncthreads();
    // [h, S, i] layout: a warp's lanes (consecutive i) are consecutive
    // floats at every j - the [h, i, S] layout put lanes 512 B apart and
    // this kernel measured 780 GB/s (half the DRAM roof) at c32
    float* srow = state + (size_t)slots[r] * n_heads * head_dim * S_
                  + (size_t)h * head_dim * S_ + i;
    float dt = dt_raw[(size_t)r * dt_stride + h] + dt_bias[h];
    dt = (dt <= PD_M2_SOFTPLUS_LIM) ? log1pf(expf(dt)) : dt;
    const float decay = expf(dt * A[h]);
    const float x_ti = row[(size_t)h * head_dim + i];
    const float contrib = dt * x_ti;
    float acc = 0.0f;
    #pragma unroll
    for (uint32_t j = 0; j < S_; ++j) {
        const float s = decay * srow[(size_t)j * head_dim] + contrib * sB[j];
        srow[(size_t)j * head_dim] = s;
        acc += s * sC[j];
    }
    y[(size_t)r * d_inner + (size_t)h * head_dim + i] = acc + D[h] * x_ti;
}

// NOTE (FALSIFIED): a fill-the-die j-split of this
// kernel family (JS=4 j-slices of the same (h, i-pair) per CTA, 4x grid.x)
// was built, gated, and measured DEAD-EVEN at every batch. The waves-0.45
// reading was occupancy-as-description: at width the kernel is
// memory-bound with every SM already fed, and at small batch the per-layer
// state is L2-resident (23 x b MB footprint, warm below b~6) and the ~4.3 us
// launch is fixed-cost-bound. Do not rebuild the split without new evidence.

PD_EXPORT
int pd_mamba2_scan_step_batch(void* state, const void* xbc, const void* dt_raw,
                              uint32_t dt_stride, const void* slots,
                              const void* A, const void* D,
                              const void* dt_bias, void* y, uint32_t batch,
                              uint32_t n_heads, uint32_t head_dim,
                              uint32_t d_state, uint32_t n_groups,
                              void* stream) {
    if (batch == 0) return 0;
    if (d_state != 128u) return cudaErrorInvalidValue;  // template pin, as scan_seq
    if (head_dim == 0 || 128u % head_dim != 0) return cudaErrorInvalidValue;
    const uint32_t hpb = 128u / head_dim;
    if (n_heads % hpb != 0 || n_groups == 0 || n_heads % n_groups != 0 ||
        (n_heads / n_groups) % hpb != 0)
        return cudaErrorInvalidValue;
    dim3 grid(n_heads / hpb, batch);
    if (head_dim == 64u)
        pd_mamba2_scan_step_batch_kernel<128u, 64u>
            <<<grid, 128u, 0, (cudaStream_t)stream>>>(
                (float*)state, (const float*)xbc, (const float*)dt_raw,
                dt_stride, (const uint32_t*)slots, (const float*)A,
                (const float*)D, (const float*)dt_bias, (float*)y, n_heads,
                head_dim, n_groups);
    else
        pd_mamba2_scan_step_batch_kernel<128u>
            <<<grid, 128u, 0, (cudaStream_t)stream>>>(
                (float*)state, (const float*)xbc, (const float*)dt_raw,
                dt_stride, (const uint32_t*)slots, (const float*)A,
                (const float*)D, (const float*)dt_bias, (float*)y, n_heads,
                head_dim, n_groups);
    return pd_launch_status();
}

// ===== f16 SSM-state class =================================================
// The recurrent state is STORED as f16 and computed in f32 - the numeric
// class of vLLM's `--mamba-ssm-cache-dtype float16`, not a lighter one.
//
// The state is the scan's whole byte term: [slot][h][S][i] = 64 heads x 64
// head_dim x 128 S = 524,288 elems, streamed in and out on every decode
// step. Measured at the served geometry, both arms
// sweeping a matched 256 MB so neither sits in L2:
//   f32  91.12 us  1473 GB/s          f16x2  46.05 us  1457 GB/s  (+49.5%)
// Applied to the measured 1.357 ms/step band that is ~0.67 ms/step, and it
// halves the mamba arena (~0.77 GiB back at 32 slots).
//
// Two facts the bench established and these kernels depend on:
//  1. Naive `__half` runs at 1337 GB/s - Below the f32 kernel's 1473 -
//     because a warp's 32 lanes cover consecutive `i`, so halving the
//     element halves the transaction from 128 B to 64 B, and this pack's
//     stream_pattern table prices 64 B at that stride ~40% under 128 B.
//     The decode kernel therefore pairs `i` per thread (__half2) to keep the
//     warp at 128 B. The byte count alone does not earn this class its win.
//  2. The seq walks keep the state register-resident for the whole span and
//     touch the arena only at the two ends, so their per-token arithmetic is
//     BIT-IDENTICAL to the f32 kernels. Only the hand-off between launches
//     rounds. The decode step rounds once per token, which is the recurrent
//     compounding this class has to be gated on.
// Rounding is __float2half_rn throughout and `y` always uses the PRE-round
// f32 state: the rounding is a storage decision, never an arithmetic one.

// Seq walk, f16 arena. Interior math identical to pd_mamba2_scan_seq_kernel.
template <uint32_t S_, uint32_t HD_ = 0u>
__global__ void __launch_bounds__(128, 1) pd_mamba2_scan_seq_f16_kernel(
        __half* __restrict__ state, const float* __restrict__ xbc,
        const float* __restrict__ dt_raw, uint32_t dt_stride,
        const float* __restrict__ A, const float* __restrict__ D,
        const float* __restrict__ dt_bias, float* __restrict__ y,
        uint32_t n_tokens, uint32_t n_heads, uint32_t head_dim_rt,
        uint32_t n_groups) {
    const uint32_t head_dim = HD_ ? HD_ : head_dim_rt;
    const uint32_t hpb = blockDim.x / head_dim;
    const uint32_t h = blockIdx.x * hpb + threadIdx.x / head_dim;
    const uint32_t i = threadIdx.x % head_dim;
    const uint32_t d_inner = n_heads * head_dim;
    const uint32_t conv_dim = d_inner + 2u * n_groups * S_;
    const uint32_t g = h / (n_heads / n_groups);
    __shared__ float sB[S_], sC[S_];

    float st[S_];
    __half* srow = state + (size_t)h * head_dim * S_ + i;
    #pragma unroll
    for (uint32_t j = 0; j < S_; ++j) st[j] = __half2float(srow[(size_t)j * head_dim]);

    const float a_h = A[h], d_h = D[h], db_h = dt_bias[h];
    for (uint32_t t = 0; t < n_tokens; ++t) {
        const float* row = xbc + (size_t)t * conv_dim;
        if (threadIdx.x < S_) {
            sB[threadIdx.x] = row[d_inner + (size_t)g * S_ + threadIdx.x];
            sC[threadIdx.x] = row[d_inner + (size_t)(n_groups + g) * S_ + threadIdx.x];
        }
        __syncthreads();
        float dt = dt_raw[(size_t)t * dt_stride + h] + db_h;
        dt = (dt <= PD_M2_SOFTPLUS_LIM) ? log1pf(expf(dt)) : dt;
        const float decay = expf(dt * a_h);
        const float x_ti = row[(size_t)h * head_dim + i];
        const float contrib = dt * x_ti;
        float acc = 0.0f;
        #pragma unroll
        for (uint32_t j = 0; j < S_; ++j) {
            const float s = decay * st[j] + contrib * sB[j];
            st[j] = s;
            acc += s * sC[j];
        }
        y[(size_t)t * d_inner + (size_t)h * head_dim + i] = acc + d_h * x_ti;
        __syncthreads();
    }

    #pragma unroll
    for (uint32_t j = 0; j < S_; ++j)
        srow[(size_t)j * head_dim] = __float2half_rn(st[j]);
}

PD_EXPORT
int pd_mamba2_scan_seq_f16(void* state, const void* xbc, const void* dt_raw,
                           uint32_t dt_stride, const void* A, const void* D,
                           const void* dt_bias, void* y, uint32_t n_tokens,
                           uint32_t n_heads, uint32_t head_dim,
                           uint32_t d_state, uint32_t n_groups, void* stream) {
    if (n_tokens == 0) return 0;
    if (d_state != 128u) return cudaErrorInvalidValue;
    if (head_dim == 0 || 128u % head_dim != 0) return cudaErrorInvalidValue;
    const uint32_t hpb = 128u / head_dim;
    if (n_heads % hpb != 0 || n_groups == 0 || n_heads % n_groups != 0 ||
        (n_heads / n_groups) % hpb != 0)
        return cudaErrorInvalidValue;
    if (pd_m2_ssd_elect(n_tokens, n_heads, head_dim, d_state, n_groups)) {
        const int r = pd_mamba2_ssd_run(state, 1, xbc, dt_raw, dt_stride, A,
                                        D, dt_bias, y, n_tokens, n_heads,
                                        n_groups, stream);
        if (r >= 0) return r;
        // stream-ordered scratch alloc failed - serial walk below
    }
    if (head_dim == 64u)
        pd_mamba2_scan_seq_f16_kernel<128u, 64u>
            <<<n_heads / hpb, 128u, 0, (cudaStream_t)stream>>>(
                (__half*)state, (const float*)xbc, (const float*)dt_raw,
                dt_stride, (const float*)A, (const float*)D,
                (const float*)dt_bias, (float*)y, n_tokens, n_heads, head_dim,
                n_groups);
    else
        pd_mamba2_scan_seq_f16_kernel<128u>
            <<<n_heads / hpb, 128u, 0, (cudaStream_t)stream>>>(
                (__half*)state, (const float*)xbc, (const float*)dt_raw,
                dt_stride, (const float*)A, (const float*)D,
                (const float*)dt_bias, (float*)y, n_tokens, n_heads, head_dim,
                n_groups);
    return pd_launch_status();
}

// Seq walk + per-row snapshots, f16 arena AND f16 snap blob - a partial
// spec accept rolls back by flat-copying a snap row over the live state, so
// the two must share a representation or the rollback re-rounds.
// Row t+1 starts from row t's STORED (f16-rounded) state, exactly as a
// one-row decode step starts from the arena: this walk verifies decode
// steps, so it reproduces them bit for bit (the step's arithmetic verbatim;
// y reads the pre-round state as the step's does). It carried the f32 state
// across rows instead - more precise, but every verify row after the first
// then scored its token from a state no decode step ever holds, and the
// drift reached the logits (GB10 2026-09-26: 0.02-0.2 at rows 1-7).
template <uint32_t S_, uint32_t HD_ = 0u>
__global__ void __launch_bounds__(128, 1) pd_mamba2_scan_seq_snap_f16_kernel(
        __half* __restrict__ state, const float* __restrict__ xbc,
        const float* __restrict__ dt_raw, uint32_t dt_stride,
        const float* __restrict__ A, const float* __restrict__ D,
        const float* __restrict__ dt_bias, float* __restrict__ y,
        __half* __restrict__ snap, uint32_t n_tokens, uint32_t n_heads,
        uint32_t head_dim_rt, uint32_t n_groups) {
    const uint32_t head_dim = HD_ ? HD_ : head_dim_rt;
    const uint32_t hpb = blockDim.x / head_dim;
    const uint32_t h = blockIdx.x * hpb + threadIdx.x / head_dim;
    const uint32_t i = threadIdx.x % head_dim;
    const uint32_t d_inner = n_heads * head_dim;
    const uint32_t conv_dim = d_inner + 2u * n_groups * S_;
    const uint32_t g = h / (n_heads / n_groups);
    __shared__ float sB[S_], sC[S_];

    float st[S_];
    __half* srow = state + (size_t)h * head_dim * S_ + i;
    #pragma unroll
    for (uint32_t j = 0; j < S_; ++j) st[j] = __half2float(srow[(size_t)j * head_dim]);

    const float a_h = A[h], d_h = D[h], db_h = dt_bias[h];
    for (uint32_t t = 0; t < n_tokens; ++t) {
        const float* row = xbc + (size_t)t * conv_dim;
        if (threadIdx.x < S_) {
            sB[threadIdx.x] = row[d_inner + (size_t)g * S_ + threadIdx.x];
            sC[threadIdx.x] = row[d_inner + (size_t)(n_groups + g) * S_ + threadIdx.x];
        }
        __syncthreads();
        float dt = dt_raw[(size_t)t * dt_stride + h] + db_h;
        dt = (dt <= PD_M2_SOFTPLUS_LIM) ? log1pf(expf(dt)) : dt;
        const float decay = expf(dt * a_h);
        const float x_ti = row[(size_t)h * head_dim + i];
        const float contrib = dt * x_ti;
        float acc = 0.0f;
        __half* snrow = snap + (size_t)t * d_inner * S_
                        + (size_t)h * head_dim * S_ + i;
        #pragma unroll
        for (uint32_t j = 0; j < S_; ++j) {
            const float s = decay * st[j] + contrib * sB[j];
            const __half sh = __float2half_rn(s);
            snrow[(size_t)j * head_dim] = sh;
            st[j] = __half2float(sh);
            acc += s * sC[j];
        }
        y[(size_t)t * d_inner + (size_t)h * head_dim + i] = acc + d_h * x_ti;
        __syncthreads();
    }

    #pragma unroll
    for (uint32_t j = 0; j < S_; ++j)
        srow[(size_t)j * head_dim] = __float2half_rn(st[j]);
}

PD_EXPORT
int pd_mamba2_scan_seq_snap_f16(void* state, const void* xbc,
                                const void* dt_raw, uint32_t dt_stride,
                                const void* A, const void* D,
                                const void* dt_bias, void* y, void* snap,
                                uint32_t n_tokens, uint32_t n_heads,
                                uint32_t head_dim, uint32_t d_state,
                                uint32_t n_groups, void* stream) {
    if (n_tokens == 0) return 0;
    if (d_state != 128u) return cudaErrorInvalidValue;
    if (head_dim == 0 || 128u % head_dim != 0) return cudaErrorInvalidValue;
    const uint32_t hpb = 128u / head_dim;
    if (n_heads % hpb != 0 || n_groups == 0 || n_heads % n_groups != 0 ||
        (n_heads / n_groups) % hpb != 0)
        return cudaErrorInvalidValue;
    if (head_dim == 64u)
        pd_mamba2_scan_seq_snap_f16_kernel<128u, 64u>
            <<<n_heads / hpb, 128u, 0, (cudaStream_t)stream>>>(
                (__half*)state, (const float*)xbc, (const float*)dt_raw,
                dt_stride, (const float*)A, (const float*)D,
                (const float*)dt_bias, (float*)y, (__half*)snap, n_tokens,
                n_heads, head_dim, n_groups);
    else
        pd_mamba2_scan_seq_snap_f16_kernel<128u>
            <<<n_heads / hpb, 128u, 0, (cudaStream_t)stream>>>(
                (__half*)state, (const float*)xbc, (const float*)dt_raw,
                dt_stride, (const float*)A, (const float*)D,
                (const float*)dt_bias, (float*)y, (__half*)snap, n_tokens,
                n_heads, head_dim, n_groups);
    return pd_launch_status();
}

// The verify walk without per-row snapshots, and its replay (slots 746-747).
// The snap walk above writes a full [H, hd, S] f16 state for every verify
// row - 1 MiB a row on Nemotron 3.5 - so a 6-row DSpark round dirtied ~7 MiB
// of L2 per mamba layer, and those lines were written back on the DRAM time
// of whatever streamed next (GB10 2026-10-06: the q|k|v projection after the
// last mamba layer of a run read 113 -> 146 us). The keep walk leaves the
// live state untouched (it IS the pre-round state until the commit) and
// stores instead what the walk consumed from each row - x, B and the raw
// dt, ~20 KiB a row; the commit replays the accepted rows from those over
// the live state in place. Both walks run pd_m2_rows_f16 below, the snap
// walk's arithmetic verbatim (row t+1 from row t's f16-rounded state, y from
// the pre-round f32 state), so the replayed state is the snap row bit for
// bit and y is the snap walk's y.
//
// keep row layout [x (H*hd) | B (G*S) | dt_raw (H)], stride H*hd + G*S + H:
// x and B sit at their conv-row offsets, so the replay reads a keep row
// through the very indexing a conv row takes.
template <uint32_t S_, uint32_t HD_, bool Y>
__device__ __forceinline__ void pd_m2_rows_f16(
        const __half* __restrict__ st_in, __half* __restrict__ st_out,
        const float* __restrict__ xbc, uint32_t xbc_stride,
        const float* __restrict__ dt_raw, uint32_t dt_stride,
        const float* __restrict__ A, const float* __restrict__ D,
        const float* __restrict__ dt_bias, float* __restrict__ y,
        float* __restrict__ keep, uint32_t n_tokens, uint32_t n_heads,
        uint32_t head_dim_rt, uint32_t n_groups, uint32_t cta) {
    const uint32_t head_dim = HD_ ? HD_ : head_dim_rt;
    const uint32_t hpb = blockDim.x / head_dim;
    const uint32_t h = cta * hpb + threadIdx.x / head_dim;
    const uint32_t i = threadIdx.x % head_dim;
    const uint32_t d_inner = n_heads * head_dim;
    const uint32_t g = h / (n_heads / n_groups);
    const uint32_t ks = d_inner + n_groups * S_ + n_heads;
    // one CTA of each group stores the group's B row
    const bool keep_b = keep != nullptr && (cta * hpb) % (n_heads / n_groups) == 0u;
    __shared__ float sB[S_], sC[S_];

    float st[S_];
    const __half* srow = st_in + (size_t)h * head_dim * S_ + i;
    #pragma unroll
    for (uint32_t j = 0; j < S_; ++j) st[j] = __half2float(srow[(size_t)j * head_dim]);

    const float a_h = A[h], db_h = dt_bias[h];
    const float d_h = Y ? D[h] : 0.0f;
    for (uint32_t t = 0; t < n_tokens; ++t) {
        const float* row = xbc + (size_t)t * xbc_stride;
        if (threadIdx.x < S_) {
            sB[threadIdx.x] = row[d_inner + (size_t)g * S_ + threadIdx.x];
            if (Y) sC[threadIdx.x] = row[d_inner + (size_t)(n_groups + g) * S_ + threadIdx.x];
            if (keep_b)
                keep[(size_t)t * ks + d_inner + (size_t)g * S_ + threadIdx.x] = sB[threadIdx.x];
        }
        __syncthreads();
        const float dtr = dt_raw[(size_t)t * dt_stride + h];
        float dt = dtr + db_h;
        dt = (dt <= PD_M2_SOFTPLUS_LIM) ? log1pf(expf(dt)) : dt;
        const float decay = expf(dt * a_h);
        const float x_ti = row[(size_t)h * head_dim + i];
        const float contrib = dt * x_ti;
        if (keep != nullptr) {
            keep[(size_t)t * ks + (size_t)h * head_dim + i] = x_ti;
            if (i == 0u) keep[(size_t)t * ks + d_inner + (size_t)n_groups * S_ + h] = dtr;
        }
        float acc = 0.0f;
        #pragma unroll
        for (uint32_t j = 0; j < S_; ++j) {
            const float s = decay * st[j] + contrib * sB[j];
            const __half sh = __float2half_rn(s);
            st[j] = __half2float(sh);
            if (Y) acc += s * sC[j];
        }
        if (Y) y[(size_t)t * d_inner + (size_t)h * head_dim + i] = acc + d_h * x_ti;
        __syncthreads();
    }

    if (st_out != nullptr) {
        __half* orow = st_out + (size_t)h * head_dim * S_ + i;
        #pragma unroll
        for (uint32_t j = 0; j < S_; ++j)
            orow[(size_t)j * head_dim] = __float2half_rn(st[j]);
    }
}

template <uint32_t S_, uint32_t HD_ = 0u>
__global__ void __launch_bounds__(128, 1) pd_mamba2_scan_seq_keep_f16_kernel(
        const __half* __restrict__ state, const float* __restrict__ xbc,
        const float* __restrict__ dt_raw, uint32_t dt_stride,
        const float* __restrict__ A, const float* __restrict__ D,
        const float* __restrict__ dt_bias, float* __restrict__ y,
        float* __restrict__ keep, uint32_t n_tokens, uint32_t n_heads,
        uint32_t head_dim_rt, uint32_t n_groups) {
    const uint32_t d_inner = n_heads * (HD_ ? HD_ : head_dim_rt);
    pd_m2_rows_f16<S_, HD_, true>(state, nullptr, xbc, d_inner + 2u * n_groups * S_, dt_raw,
                                  dt_stride, A, D, dt_bias, y, keep, n_tokens, n_heads,
                                  head_dim_rt, n_groups, blockIdx.x);
}

// One replay a (descriptor, head block): {src state, dst state, keep rows,
// A, dt_bias, rows} as six u64 words. src == dst replays in place (each
// thread reads its own state elements before it writes them).
template <uint32_t S_, uint32_t HD_ = 0u>
__global__ void __launch_bounds__(128, 1) pd_mamba2_rescan_f16_kernel(
        const unsigned long long* __restrict__ descs, uint32_t n_heads,
        uint32_t head_dim_rt, uint32_t n_groups) {
    const unsigned long long* d = descs + (size_t)blockIdx.y * 6u;
    const uint32_t n = (uint32_t)d[5];
    if (n == 0u) return;
    const uint32_t d_inner = n_heads * (HD_ ? HD_ : head_dim_rt);
    const uint32_t ks = d_inner + n_groups * S_ + n_heads;
    const float* keep = (const float*)(uintptr_t)d[2];
    pd_m2_rows_f16<S_, HD_, false>(
        (const __half*)(uintptr_t)d[0], (__half*)(uintptr_t)d[1], keep, ks,
        keep + d_inner + (size_t)n_groups * S_, ks, (const float*)(uintptr_t)d[3], nullptr,
        (const float*)(uintptr_t)d[4], nullptr, nullptr, n, n_heads, head_dim_rt, n_groups,
        blockIdx.x);
}

static int pd_m2_rows_geometry(uint32_t n_heads, uint32_t head_dim, uint32_t d_state,
                               uint32_t n_groups) {
    if (d_state != 128u) return cudaErrorInvalidValue;
    if (head_dim == 0 || 128u % head_dim != 0) return cudaErrorInvalidValue;
    const uint32_t hpb = 128u / head_dim;
    if (n_heads % hpb != 0 || n_groups == 0 || n_heads % n_groups != 0 ||
        (n_heads / n_groups) % hpb != 0)
        return cudaErrorInvalidValue;
    return 0;
}

// ABI 746. The snap walk's (slot 444) verify without snapshots: `state` is
// read, never written; y as the snap walk's; `keep` [n_tokens, H*hd + G*S +
// H] f32 receives each row's x | B | raw dt for the replay (slot 747).
PD_EXPORT
int pd_mamba2_scan_seq_keep_f16(const void* state, const void* xbc, const void* dt_raw,
                                uint32_t dt_stride, const void* A, const void* D,
                                const void* dt_bias, void* y, void* keep, uint32_t n_tokens,
                                uint32_t n_heads, uint32_t head_dim, uint32_t d_state,
                                uint32_t n_groups, void* stream) {
    if (n_tokens == 0) return 0;
    if (const int e = pd_m2_rows_geometry(n_heads, head_dim, d_state, n_groups)) return e;
    if (keep == nullptr) return cudaErrorInvalidValue;
    const uint32_t hpb = 128u / head_dim;
    if (head_dim == 64u)
        pd_mamba2_scan_seq_keep_f16_kernel<128u, 64u>
            <<<n_heads / hpb, 128u, 0, (cudaStream_t)stream>>>(
                (const __half*)state, (const float*)xbc, (const float*)dt_raw, dt_stride,
                (const float*)A, (const float*)D, (const float*)dt_bias, (float*)y,
                (float*)keep, n_tokens, n_heads, head_dim, n_groups);
    else
        pd_mamba2_scan_seq_keep_f16_kernel<128u>
            <<<n_heads / hpb, 128u, 0, (cudaStream_t)stream>>>(
                (const __half*)state, (const float*)xbc, (const float*)dt_raw, dt_stride,
                (const float*)A, (const float*)D, (const float*)dt_bias, (float*)y,
                (float*)keep, n_tokens, n_heads, head_dim, n_groups);
    return pd_launch_status();
}

// ABI 747. Replays `n` descriptors (device, six u64 words each: src f16
// state, dst f16 state, keep rows from slot 746, A, dt_bias, rows) in one
// launch: dst <- the state after `rows` keep rows walked from src, the bits
// slot 444's snap row `rows - 1` holds. rows == 0 leaves dst alone.
PD_EXPORT
int pd_mamba2_rescan_f16(const void* descs, uint32_t n, uint32_t n_heads, uint32_t head_dim,
                         uint32_t d_state, uint32_t n_groups, void* stream) {
    if (n == 0) return 0;
    if (const int e = pd_m2_rows_geometry(n_heads, head_dim, d_state, n_groups)) return e;
    if (n > 65535u) return cudaErrorInvalidValue;
    const dim3 grid(n_heads / (128u / head_dim), n);
    if (head_dim == 64u)
        pd_mamba2_rescan_f16_kernel<128u, 64u><<<grid, 128u, 0, (cudaStream_t)stream>>>(
            (const unsigned long long*)descs, n_heads, head_dim, n_groups);
    else
        pd_mamba2_rescan_f16_kernel<128u><<<grid, 128u, 0, (cudaStream_t)stream>>>(
            (const unsigned long long*)descs, n_heads, head_dim, n_groups);
    return pd_launch_status();
}

// Decode step, f16 arena, __half2-paired along i. One thread owns two
// consecutive i, so a warp's 32 lanes cover 64 halves = 128 B contiguous at
// every j - the same transaction width the f32 kernel gets. head_dim must be
// even; HD_/2 threads cover a head.
template <uint32_t S_, uint32_t HD_>
__global__ void __launch_bounds__(128, 1) pd_mamba2_scan_step_batch_f16_kernel(
        __half2* __restrict__ state, const float* __restrict__ xbc,
        const float* __restrict__ dt_raw, uint32_t dt_stride,
        const uint32_t* __restrict__ slots, const float* __restrict__ A,
        const float* __restrict__ D, const float* __restrict__ dt_bias,
        float* __restrict__ y, uint32_t n_heads, uint32_t n_groups) {
    constexpr uint32_t HH = HD_ / 2u;
    const uint32_t r = blockIdx.y;
    const uint32_t hpb = blockDim.x / HH;
    const uint32_t h = blockIdx.x * hpb + threadIdx.x / HH;
    const uint32_t i2 = threadIdx.x % HH;
    const uint32_t d_inner = n_heads * HD_;
    const uint32_t conv_dim = d_inner + 2u * n_groups * S_;
    const uint32_t g = h / (n_heads / n_groups);
    __shared__ float sB[S_], sC[S_];
    const float* row = xbc + (size_t)r * conv_dim;
    if (threadIdx.x < S_) {
        sB[threadIdx.x] = row[d_inner + (size_t)g * S_ + threadIdx.x];
        sC[threadIdx.x] = row[d_inner + (size_t)(n_groups + g) * S_ + threadIdx.x];
    }
    __syncthreads();
    __half2* srow = state + ((size_t)slots[r] * n_heads * HD_ * S_) / 2u
                  + ((size_t)h * HD_ * S_) / 2u + i2;
    float dt = dt_raw[(size_t)r * dt_stride + h] + dt_bias[h];
    dt = (dt <= PD_M2_SOFTPLUS_LIM) ? log1pf(expf(dt)) : dt;
    const float decay = expf(dt * A[h]);
    const float2 x_ti = make_float2(row[(size_t)h * HD_ + 2u * i2],
                                    row[(size_t)h * HD_ + 2u * i2 + 1u]);
    const float2 contrib = make_float2(dt * x_ti.x, dt * x_ti.y);
    float2 acc = make_float2(0.0f, 0.0f);
    #pragma unroll
    for (uint32_t j = 0; j < S_; ++j) {
        const float2 sv = __half22float2(srow[(size_t)j * HH]);
        const float a = decay * sv.x + contrib.x * sB[j];
        const float b = decay * sv.y + contrib.y * sB[j];
        srow[(size_t)j * HH] = __floats2half2_rn(a, b);
        acc.x += a * sC[j];
        acc.y += b * sC[j];
    }
    float* yr = y + (size_t)r * d_inner + (size_t)h * HD_ + 2u * i2;
    yr[0] = acc.x + D[h] * x_ti.x;
    yr[1] = acc.y + D[h] * x_ti.y;
}

PD_EXPORT
int pd_mamba2_scan_step_batch_f16(void* state, const void* xbc,
                                  const void* dt_raw, uint32_t dt_stride,
                                  const void* slots, const void* A,
                                  const void* D, const void* dt_bias, void* y,
                                  uint32_t batch, uint32_t n_heads,
                                  uint32_t head_dim, uint32_t d_state,
                                  uint32_t n_groups, void* stream) {
    if (batch == 0) return 0;
    if (d_state != 128u) return cudaErrorInvalidValue;
    // the half2 pairing needs an even head_dim and HD_/2 dividing 128
    if (head_dim != 64u) return cudaErrorInvalidValue;  // template pin, as above
    const uint32_t hh = head_dim / 2u;
    const uint32_t hpb = 128u / hh;
    if (n_heads % hpb != 0 || n_groups == 0 || n_heads % n_groups != 0 ||
        (n_heads / n_groups) % hpb != 0)
        return cudaErrorInvalidValue;
    dim3 grid(n_heads / hpb, batch);
    pd_mamba2_scan_step_batch_f16_kernel<128u, 64u>
        <<<grid, 128u, 0, (cudaStream_t)stream>>>(
            (__half2*)state, (const float*)xbc, (const float*)dt_raw, dt_stride,
            (const uint32_t*)slots, (const float*)A, (const float*)D,
            (const float*)dt_bias, (float*)y, n_heads, n_groups);
    return pd_launch_status();
}

// ---- f16 SSM-state <-> f32 checkpoint blob -------------------------------
// The prefix-cache checkpoint pool, its staging blobs and the radix
// accounting all key off an f32 [state | win] layout per mamba layer. That
// is the correctness-critical restore path, so the f16 state class does not
// re-lay it out: it converts at the arena boundary instead, which is
// LOSSLESS in both directions - widening f16->f32 is exact, and the narrowing
// back only ever sees values that originated as f16, so they round-trip
// bit-for-bit. Conversions run at checkpoint breaks (rare), never per step.
__global__ void pd_ssm_state_widen_kernel(const __half* __restrict__ src,
                                          float* __restrict__ dst, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half2float(src[i]);
}

__global__ void pd_ssm_state_narrow_kernel(const float* __restrict__ src,
                                           __half* __restrict__ dst, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half_rn(src[i]);
}

PD_EXPORT
int pd_ssm_state_widen(const void* src, void* dst, uint32_t n, void* stream) {
    if (n == 0) return 0;
    pd_ssm_state_widen_kernel<<<(n + 255u) / 256u, 256u, 0, (cudaStream_t)stream>>>(
        (const __half*)src, (float*)dst, n);
    return pd_launch_status();
}

PD_EXPORT
int pd_ssm_state_narrow(const void* src, void* dst, uint32_t n, void* stream) {
    if (n == 0) return 0;
    pd_ssm_state_narrow_kernel<<<(n + 255u) / 256u, 256u, 0, (cudaStream_t)stream>>>(
        (const float*)src, (__half*)dst, n);
    return pd_launch_status();
}
