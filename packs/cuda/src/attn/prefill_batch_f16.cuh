// attn/prefill_batch_f16.cuh - the multi-slot WMMA prefill (pd_attn_prefill_batch_f16),
// split from attn/prefill_pf5.cuh to keep it under the file-length ceiling.
// Textually-included segment of the single pack translation unit, right after
// attn/prefill_pf5.cuh; include order is defined by ../../pack.cu.

// --------------------------------- attn prefill batch f16 (multi-slot WMMA)
// pd_attn_prefill_f16's tensor-core structure applied to the encoder's
// ragged many-text batches through the same per-text tiling contract as
// pd_attn_prefill_batch: the host emits 32-query tiles that never cross a
// text; rows spilled past a text are masked by slots[b] == slot. This is the
// hot encoder attention (the scalar f32 batch kernel ran at ~11% of f32 peak
// and was 37% of the whole reranker forward at 58-row suffixes x 127-key
// texts). Only head_dim 128 is instantiated - every Qwen3 encoder size.
// Numeric class: f16 Q/K/V inputs, f32 score accumulate + softmax, f16 O
// accumulate - pd_attn_prefill_f16's class, not bit-identical to the scalar
// batch kernel; the load-time calibration gate re-validates encoder quality
// (its Q8_0 baseline runs the same kernel, so both sides of the gate move
// together).
// Requirements: head_dim == 128, fp16 KV, max_ctx % 64 == 0 (K/V fragment
// loads touch keys up to t0+63; hi <= max_ctx and t0 stepping by 64 keep
// them inside the slot's cache rows - same bound trick as the single-slot
// kernel).
#define PD_ABF16_NC 32  // queries per tile (host tiling stride)
#define PD_ABF16_TK 64  // keys per tile (4 warps x 16 fragment rows)
template<uint32_t D>
__global__ void __launch_bounds__(128) pd_attn_prefill_batch_f16_kernel(
    const float* __restrict__ q, const __half* __restrict__ kc,
    const __half* __restrict__ vc, const float* __restrict__ sinks,
    float* __restrict__ out, const unsigned int* __restrict__ positions,
    const unsigned int* __restrict__ slots, const unsigned int* __restrict__ tile_row0,
    const unsigned int* __restrict__ tile_slot,
    uint32_t n_heads, uint32_t n_kv_heads, uint32_t max_ctx, uint32_t kv_dim,
    uint32_t swa_window, uint32_t n_rows, float scale,
    const unsigned int* __restrict__ win_pos) {
#if PD_MMA_OK
    using namespace nvcuda;
    constexpr uint32_t NC = PD_ABF16_NC, TK = PD_ABF16_TK;
    constexpr uint32_t RPW = NC / 4u;  // query rows per warp (matches f16 twins)
    constexpr uint32_t DW = D / 4u;    // dims per warp in the V GEMM
    constexpr uint32_t DP = D + 8u;    // half rows, conflict-avoid pad
    constexpr uint32_t KQP = TK + 8u;  // f32 score row stride
    typedef wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_k;
    typedef wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> frag_v;
    typedef wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_b;
    typedef wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_s;
    typedef wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_o;

    const uint32_t h = blockIdx.x;
    const uint32_t row0 = tile_row0[blockIdx.y];
    const uint32_t slot = tile_slot[blockIdx.y];
    const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
    const uint32_t kvh = h / (n_heads / n_kv_heads);
    const unsigned int* wp = pd_pf_wp(positions, win_pos);

    __shared__ half sh_q[NC * DP];
    __shared__ float sh_s[NC * KQP];   // scores f32; P overwrites as f16
    __shared__ half sh_o[NC * DP];     // running O accumulator
    __shared__ float sh_corr[NC];
    __shared__ float sh_onorm[NC];
    __shared__ uint32_t sh_hi[NC];
    __shared__ uint32_t sh_lo[NC];     // window floors (+1), from wp
    half* sh_p = (half*)sh_s;          // P at half stride 2*KQP, in place

    // liveness first: the CUDA-graph path pads the tile grid with dead
    // sentinels, and those blocks must exit before the Q/O staging cost
    if (tid < NC) {
        const uint32_t b = row0 + tid;
        const bool live = b < n_rows && slots[b] == slot;
        sh_hi[tid] = live ? positions[b] + 1u : 0u;
        sh_lo[tid] = live ? wp[b] + 1u : 0u;
    }
    __syncthreads();
    uint32_t hi = 0;
    #pragma unroll
    for (uint32_t i = 0; i < NC; ++i) hi = max(hi, sh_hi[i]);
    if (hi == 0u) return;  // whole tile masked (sentinel or spill-only)

    // stage Q (f32 -> f16, pre-scaled); a row is live only if in range AND
    // in this slot - spilled rows past the text zero out here
    #pragma unroll
    for (uint32_t it = 0; it < NC * D / 128u; ++it) {
        const uint32_t i = it * 128u + tid, j = i / D, dd = i % D;
        const uint32_t b = row0 + j;
        const bool live = b < n_rows && slots[b] == slot;
        sh_q[j * DP + dd] = __float2half(
            live ? q[((size_t)b * n_heads + h) * D + dd] * scale : 0.f);
        sh_o[j * DP + dd] = __float2half(0.f);
    }
    __syncthreads();

    // pin Q in fragments: Q_b[dim frag][query frag]
    frag_b Q_b[D / 16u][NC / 16u];
    #pragma unroll
    for (uint32_t d0 = 0; d0 < D / 16u; ++d0)
        #pragma unroll
        for (uint32_t j0 = 0; j0 < NC / 16u; ++j0)
            wmma::load_matrix_sync(Q_b[d0][j0], sh_q + j0 * 16u * DP + d0 * 16u, DP);
    __syncthreads();

    // softmax state: this warp owns queries [8*warp, 8*warp+8)
    float m_st[NC / 4u], l_st[NC / 4u];
    #pragma unroll
    for (uint32_t jj = 0; jj < NC / 4u; ++jj) { m_st[jj] = -1e30f; l_st[jj] = 0.f; }

    const __half* kcb = kc + (size_t)slot * max_ctx * kv_dim + (size_t)kvh * D;
    const __half* vcb = vc + (size_t)slot * max_ctx * kv_dim + (size_t)kvh * D;

    // SWA layers: START at the block's window edge instead of masking ~all
    // of history (a 4k-prompt chunk computed ~4000 masked KV columns per
    // row on the sliding-window layers - 812 us/layer avg measured).
    // Live rows in a tile are one slot's consecutive positions,
    // so tiles below the min live row's window edge are fully masked and
    // contribute exact zeros to the online softmax (skip = bit-exact).
    uint32_t lo_t = 0;
    if (swa_window > 0) {
        uint32_t lo1 = 0xFFFFFFFFu;
        #pragma unroll
        for (uint32_t i = 0; i < NC; ++i)
            if (sh_lo[i]) lo1 = min(lo1, sh_lo[i]);
        if (lo1 != 0xFFFFFFFFu && lo1 > swa_window)
            lo_t = ((lo1 - swa_window) / TK) * TK;
    }
    for (uint32_t t0 = lo_t; t0 < hi; t0 += TK) {
        // S = Q K^T for this warp's 16-key strip x all 32 queries
        {
            frag_s S_c[NC / 16u];
            #pragma unroll
            for (uint32_t j0 = 0; j0 < NC / 16u; ++j0) wmma::fill_fragment(S_c[j0], 0.f);
            #pragma unroll
            for (uint32_t d0 = 0; d0 < D / 16u; ++d0) {
                frag_k K_a;
                wmma::load_matrix_sync(
                    K_a, kcb + (size_t)(t0 + 16u * warp) * kv_dim + d0 * 16u, kv_dim);
                #pragma unroll
                for (uint32_t j0 = 0; j0 < NC / 16u; ++j0)
                    wmma::mma_sync(S_c[j0], K_a, Q_b[d0][j0], S_c[j0]);
            }
            #pragma unroll
            for (uint32_t j0 = 0; j0 < NC / 16u; ++j0)
                wmma::store_matrix_sync(sh_s + j0 * 16u * KQP + 16u * warp, S_c[j0],
                                        KQP, wmma::mem_col_major);
        }
        __syncthreads();

        // online softmax: warp owns 8 query rows, 2 keys per lane
        #pragma unroll
        for (uint32_t jj = 0; jj < 8u; ++jj) {
            const uint32_t j = warp * 8u + jj;
            const uint32_t b = row0 + j;
            const bool live = b < n_rows && slots[b] == slot;
            const uint32_t pos = live ? positions[b] : 0u;
            const uint32_t fp = pd_pf_floor(live ? wp[b] : 0u, swa_window);
            float s0 = -1e30f, s1 = -1e30f;
            const uint32_t k0 = t0 + lane, k1 = t0 + 32u + lane;
            if (live && k0 >= fp && k0 <= pos && k0 < hi) s0 = sh_s[j * KQP + lane];
            if (live && k1 >= fp && k1 <= pos && k1 < hi) s1 = sh_s[j * KQP + 32u + lane];
            float mn = fmaxf(m_st[jj], fmaxf(s0, s1));
            #pragma unroll
            for (uint32_t o = 16; o > 0; o >>= 1)
                mn = fmaxf(mn, __shfl_xor_sync(0xffffffffu, mn, o));
            const float dc = m_st[jj] - mn;
            const float corr = dc >= -20.f ? __expf(dc) : 0.f;
            const float d0 = s0 - mn, d1 = s1 - mn;
            const float w0 = d0 >= -20.f ? __expf(d0) : 0.f;
            const float w1 = d1 >= -20.f ? __expf(d1) : 0.f;
            float ws = w0 + w1;
            #pragma unroll
            for (uint32_t o = 16; o > 0; o >>= 1)
                ws += __shfl_xor_sync(0xffffffffu, ws, o);
            l_st[jj] = l_st[jj] * corr + ws;
            m_st[jj] = mn;
            // P in place over the f32 scores (half stride 2*KQP)
            sh_p[j * 2u * KQP + lane] = __float2half(w0);
            sh_p[j * 2u * KQP + 32u + lane] = __float2half(w1);
            if (lane == 0) sh_corr[j] = corr;
        }
        __syncthreads();

        // rescale running O by corr[q] (f16, half2)
        #pragma unroll
        for (uint32_t it = 0; it < NC * D / 2u / 128u; ++it) {
            const uint32_t i = it * 128u + tid, j = i / (D / 2u), d2 = i % (D / 2u);
            half2* o2 = (half2*)(sh_o + j * DP);
            const float c = sh_corr[j];
            o2[d2] = __hmul2(o2[d2], __float2half2_rn(c));
        }
        __syncthreads();

        // O += V P: this warp owns dims [DW*warp, DW*warp+DW)
        {
            frag_o O_c[DW / 16u][NC / 16u];
            #pragma unroll
            for (uint32_t df = 0; df < DW / 16u; ++df)
                #pragma unroll
                for (uint32_t j0 = 0; j0 < NC / 16u; ++j0)
                    wmma::load_matrix_sync(
                        O_c[df][j0], sh_o + j0 * 16u * DP + DW * warp + df * 16u,
                        DP, wmma::mem_col_major);
            #pragma unroll
            for (uint32_t kf = 0; kf < TK / 16u; ++kf) {
                frag_v V_a[DW / 16u];
                #pragma unroll
                for (uint32_t df = 0; df < DW / 16u; ++df)
                    wmma::load_matrix_sync(
                        V_a[df], vcb + (size_t)(t0 + kf * 16u) * kv_dim + DW * warp + df * 16u,
                        kv_dim);
                #pragma unroll
                for (uint32_t j0 = 0; j0 < NC / 16u; ++j0) {
                    frag_b P_b;
                    wmma::load_matrix_sync(P_b, sh_p + j0 * 16u * 2u * KQP + kf * 16u,
                                           2u * KQP);
                    #pragma unroll
                    for (uint32_t df = 0; df < DW / 16u; ++df)
                        wmma::mma_sync(O_c[df][j0], V_a[df], P_b, O_c[df][j0]);
                }
            }
            #pragma unroll
            for (uint32_t df = 0; df < DW / 16u; ++df)
                #pragma unroll
                for (uint32_t j0 = 0; j0 < NC / 16u; ++j0)
                    wmma::store_matrix_sync(
                        sh_o + j0 * 16u * DP + DW * warp + df * 16u, O_c[df][j0],
                        DP, wmma::mem_col_major);
        }
        __syncthreads();
    }

    // epilogue: fold the sink into l with the same max-rebase as the other
    // attention kernels, publish per-query 1/l for the write-out
    #pragma unroll
    for (uint32_t jj = 0; jj < RPW; ++jj) {
        const uint32_t j = warp * RPW + jj;
        if (lane == 0) {
            const float s = sinks[h];
            const float mt = fmaxf(m_st[jj], s);
            const float dm = m_st[jj] - mt, ds = s - mt;
            const float cm = dm >= -20.f ? __expf(dm) : 0.f;
            const float cs = ds >= -20.f ? __expf(ds) : 0.f;
            const float l = l_st[jj] * cm + cs;
            sh_onorm[j] = l > 0.f ? cm / l : 0.f;
        }
    }
    __syncthreads();
    #pragma unroll
    for (uint32_t it = 0; it < NC * D / 128u; ++it) {
        const uint32_t i = it * 128u + tid, j = i / D, dd = i % D;
        const uint32_t b = row0 + j;
        if (b < n_rows && slots[b] == slot)
            out[((size_t)b * n_heads + h) * D + dd] =
                __half2float(sh_o[j * DP + dd]) * sh_onorm[j];
    }
#else
    (void)q; (void)kc; (void)vc; (void)sinks; (void)out; (void)positions;
    (void)slots; (void)tile_row0; (void)tile_slot; (void)n_heads; (void)n_kv_heads;
    (void)max_ctx; (void)kv_dim; (void)swa_window; (void)n_rows; (void)scale;
#endif
}

PD_EXPORT
int pd_attn_prefill_batch_f16(const void* q, const void* kc, const void* vc,
                              const void* sinks, void* out, const void* positions,
                              const void* slots, const void* tile_row0,
                              const void* tile_slot, uint32_t n_qtiles,
                              uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
                              uint32_t max_ctx, uint32_t kv_dim, uint32_t swa_window,
                              uint32_t n_rows, float scale, uint32_t kv_dtype,
                              void* stream) {
    if (n_heads == 0 || n_qtiles == 0 || n_rows == 0) return 0;
    if (head_dim != 128u || kv_dtype == PD_KV_FP8_E4M3 || (max_ctx & 63u))
        return cudaErrorInvalidValue;
    dim3 grid(n_heads, n_qtiles);
    pd_attn_prefill_batch_f16_kernel<128u><<<grid, 128, 0, (cudaStream_t)stream>>>(
        (const float*)q, (const __half*)kc, (const __half*)vc, (const float*)sinks,
        (float*)out, (const unsigned int*)positions, (const unsigned int*)slots,
        (const unsigned int*)tile_row0, (const unsigned int*)tile_slot,
        n_heads, n_kv_heads, max_ctx, kv_dim, swa_window, n_rows, scale,
        nullptr);  // causal rows only (the encoder's ragged batch): floors from positions
    return pd_launch_status();
}

// (A fused rmsnorm+quantize_q8 kernel for the B=1 decode graph measured
// slower end to end: 20b decode 217.7 -> 208.2, 120b 147.4 -> 140.1. The
// separate quantize launch runs 90 32-thread blocks in parallel across SMs;
// the fused single-block phase serialized them behind the norm barrier, and
// graph-replay launch overhead is smaller than the ~4 us/pair it needed to
// win. Two launches stay.)

