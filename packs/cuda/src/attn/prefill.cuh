// attn/prefill.cuh (formerly 11_attn_prefill.cuh) - prefill attention family (tiled, paged, batch, f16 wmma, FA-class v2, v3 refutation)
// Textually-included segment of the single pack translation unit.
// Not standalone-compilable: include order is defined by ../pack.cu.
// ------------------------------------------------------------ attn prefill
// Tiled prefill attention (P6f). The decode kernel walks keys SEQUENTIALLY,
// one per ~4 barriers, per (head, row) block - 3.9 ms/layer at pp512. This
// kernel is the prefill shape: one block per (q-head, 16-query tile), 8
// warps x 2 queries/warp, K/V streamed through shared 32 keys at a time,
// online softmax per query in registers. Same value math as the decode
// kernel (f32 dot + scale, __expf online-softmax with m init = sinks[h] and
// l init = 1, out = acc/l) but per-32-key-tile update order, so results are
// the same numeric class, not bit-identical.
// Layout notes (bank-conflict audit):
//   - K tile is D-MAJOR sh_k[dim][key+pad1]: the dot loop has lane j reading
//     key j at dim d - consecutive lanes hit consecutive banks; q is a
//     broadcast read from sh_q.
//   - V tile is KEY-MAJOR sh_v[key][dim] with rows padded to 132 floats
//     (528 B, 16-aligned): lane l accumulates dims [4l,4l+4) via one float4.
//   - All staging loops are compile-time-trip #pragma unroll (the P6e
//     lesson: rolled tid-strided loops serialize global latency).
// Requirements: head_dim == 128, threads = 256, slots non-null and UNIFORM
// across rows (true for every prefill path: all rows of one pass share one
// KV slot) - the engine dispatch falls back to the decode kernel otherwise.
#define PD_APF_TQ 16
template<typename KV, uint32_t HD>
__global__ void __launch_bounds__(256) pd_attn_prefill_kernel(
    const float* __restrict__ q, const KV* __restrict__ kc,
    const KV* __restrict__ vc, const float* __restrict__ sinks,
    float* __restrict__ out, const unsigned int* __restrict__ positions,
    const unsigned int* __restrict__ slots,
    uint32_t n_heads, uint32_t n_kv_heads, uint32_t max_ctx, uint32_t kv_dim,
    uint32_t swa_window, uint32_t n_rows, float scale) {
    // TK sized so the f32 K/V/Q tiles fit the opt-in shared window; DPL =
    // dims per lane (the slice of the output vector each lane accumulates)
    constexpr uint32_t TK = HD == 128u ? 32u : 16u;
    constexpr uint32_t DPL = HD / 32u;
    constexpr uint32_t QPAD = HD + 4u;  // 16B-aligned rows, conflict-free f4
    const uint32_t h = blockIdx.x;
    const uint32_t row0 = blockIdx.y * PD_APF_TQ;
    const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
    const uint32_t kvh = h / (n_heads / n_kv_heads);
    const uint32_t slot = slots ? slots[0] : 0u;

    extern __shared__ float pd_apf_sh[];
    float* sh_q = pd_apf_sh;                            // [TQ][QPAD]
    float* sh_k = sh_q + PD_APF_TQ * QPAD;              // [HD][TK+1] d-major
    float* sh_v = sh_k + HD * (TK + 1u);                // [TK][QPAD]
    uint32_t* sh_hi = (uint32_t*)(sh_v + TK * QPAD);    // [TQ]

    // stage the 16 queries (dead rows -> 0) and per-query key bounds
    #pragma unroll
    for (uint32_t it = 0; it < PD_APF_TQ * HD / 256u; ++it) {
        const uint32_t i = it * 256u + tid, qi = i / HD, dd = i % HD;
        const uint32_t b = row0 + qi;
        sh_q[qi * QPAD + dd] = b < n_rows ? q[((size_t)b * n_heads + h) * HD + dd] : 0.f;
    }
    if (tid < PD_APF_TQ)
        sh_hi[tid] = (row0 + tid) < n_rows ? positions[row0 + tid] + 1u : 0u;
    __syncthreads();
    uint32_t hi = 0;
    #pragma unroll
    for (uint32_t i = 0; i < PD_APF_TQ; ++i) hi = max(hi, sh_hi[i]);

    // this warp's two queries
    const uint32_t qi0 = 2u * warp, qi1 = qi0 + 1u;
    const uint32_t b0 = row0 + qi0, b1 = row0 + qi1;
    const bool live0 = b0 < n_rows, live1 = b1 < n_rows;
    const uint32_t pos0 = live0 ? positions[b0] : 0u;
    const uint32_t pos1 = live1 ? positions[b1] : 0u;
    const uint32_t fp0 = (swa_window > 0 && pos0 + 1u > swa_window) ? pos0 + 1u - swa_window : 0u;
    const uint32_t fp1 = (swa_window > 0 && pos1 + 1u > swa_window) ? pos1 + 1u - swa_window : 0u;
    float m0 = sinks[h], l0 = 1.f, m1 = m0, l1 = 1.f;
    float a0[DPL] = {}, a1[DPL] = {};

    const KV* kcb = kc + (size_t)slot * max_ctx * kv_dim + (size_t)kvh * HD;
    const KV* vcb = vc + (size_t)slot * max_ctx * kv_dim + (size_t)kvh * HD;

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
        for (uint32_t i = 0; i < PD_APF_TQ; ++i)
            if (sh_hi[i]) lo1 = min(lo1, sh_hi[i]);
        if (lo1 != 0xFFFFFFFFu && lo1 > swa_window)
            lo_t = ((lo1 - swa_window) / TK) * TK;
    }
    for (uint32_t t0 = lo_t; t0 < hi; t0 += TK) {
        __syncthreads();  // previous tile's readers are done
        #pragma unroll
        for (uint32_t it = 0; it < TK * HD / 256u; ++it) {
            const uint32_t i = it * 256u + tid, kk = i / HD, dd = i % HD;
            const uint32_t kp = t0 + kk;
            const float kvl = kp < hi ? pd_kv_load(kcb[(size_t)kp * kv_dim + dd]) : 0.f;
            const float vvl = kp < hi ? pd_kv_load(vcb[(size_t)kp * kv_dim + dd]) : 0.f;
            sh_k[dd * (TK + 1u) + kk] = kvl;
            sh_v[kk * QPAD + dd] = vvl;
        }
        __syncthreads();

        // lane j owns key t0+j (TK<32: lanes >= TK are dead weight-0 lanes):
        // two full dots against the shared queries
        const uint32_t kj = lane % TK;  // TK==32 -> lane
        const uint32_t kp = t0 + kj;
        float s0 = 0.f, s1 = 0.f;
        #pragma unroll 32
        for (uint32_t d = 0; d < HD; ++d) {
            const float kv = sh_k[d * (TK + 1u) + kj];
            s0 = fmaf(sh_q[qi0 * QPAD + d], kv, s0);
            s1 = fmaf(sh_q[qi1 * QPAD + d], kv, s1);
        }
        const bool dup = TK < 32u && lane >= TK;  // duplicate dot, mask it
        const bool v0 = live0 && !dup && kp >= fp0 && kp <= pos0 && kp < hi;
        const bool v1 = live1 && !dup && kp >= fp1 && kp <= pos1 && kp < hi;
        s0 = v0 ? s0 * scale : -1e30f;
        s1 = v1 ? s1 * scale : -1e30f;

        // per-query online softmax across the lane-scores
        float t0m = s0, t1m = s1;
        #pragma unroll
        for (uint32_t o = 16; o > 0; o >>= 1) {
            t0m = fmaxf(t0m, __shfl_xor_sync(0xffffffffu, t0m, o));
            t1m = fmaxf(t1m, __shfl_xor_sync(0xffffffffu, t1m, o));
        }
        const float n0 = fmaxf(m0, t0m), n1 = fmaxf(m1, t1m);
        const float c0 = __expf(m0 - n0), c1 = __expf(m1 - n1);
        // masked lanes get weight exactly 0 - the -1e30 sentinel alone is not
        // enough: in a fully-masked tile the running max is -1e30, so
        // exp(s - n) == exp(0) == 1 and every masked lane would vote
        const float w0 = v0 ? __expf(s0 - n0) : 0.f;
        const float w1 = v1 ? __expf(s1 - n1) : 0.f;
        float ws0 = w0, ws1 = w1;
        #pragma unroll
        for (uint32_t o = 16; o > 0; o >>= 1) {
            ws0 += __shfl_xor_sync(0xffffffffu, ws0, o);
            ws1 += __shfl_xor_sync(0xffffffffu, ws1, o);
        }
        l0 = l0 * c0 + ws0;
        l1 = l1 * c1 + ws1;
        m0 = n0;
        m1 = n1;
        #pragma unroll
        for (uint32_t c = 0; c < DPL; ++c) { a0[c] *= c0; a1[c] *= c1; }
        #pragma unroll
        for (uint32_t j = 0; j < TK; ++j) {
            const float wj0 = __shfl_sync(0xffffffffu, w0, j);
            const float wj1 = __shfl_sync(0xffffffffu, w1, j);
            #pragma unroll
            for (uint32_t c = 0; c < DPL; c += 4u) {
                const float4 vv =
                    *reinterpret_cast<const float4*>(&sh_v[j * QPAD + DPL * lane + c]);
                a0[c] = fmaf(wj0, vv.x, a0[c]); a0[c + 1] = fmaf(wj0, vv.y, a0[c + 1]);
                a0[c + 2] = fmaf(wj0, vv.z, a0[c + 2]); a0[c + 3] = fmaf(wj0, vv.w, a0[c + 3]);
                a1[c] = fmaf(wj1, vv.x, a1[c]); a1[c + 1] = fmaf(wj1, vv.y, a1[c + 1]);
                a1[c + 2] = fmaf(wj1, vv.z, a1[c + 2]); a1[c + 3] = fmaf(wj1, vv.w, a1[c + 3]);
            }
        }
    }

    if (live0) {
        #pragma unroll
        for (uint32_t c = 0; c < DPL; c += 4u) {
            float4 o0 = make_float4(a0[c] / l0, a0[c + 1] / l0, a0[c + 2] / l0, a0[c + 3] / l0);
            *reinterpret_cast<float4*>(
                &out[((size_t)b0 * n_heads + h) * HD + DPL * lane + c]) = o0;
        }
    }
    if (live1) {
        #pragma unroll
        for (uint32_t c = 0; c < DPL; c += 4u) {
            float4 o1 = make_float4(a1[c] / l1, a1[c + 1] / l1, a1[c + 2] / l1, a1[c + 3] / l1);
            *reinterpret_cast<float4*>(
                &out[((size_t)b1 * n_heads + h) * HD + DPL * lane + c]) = o1;
        }
    }
}

template<typename KV, uint32_t HD>
static int pd_attn_prefill_launch(const void* q, const void* kc, const void* vc,
                                  const void* sinks, void* out, const void* positions,
                                  const void* slots, uint32_t n_heads, uint32_t n_kv_heads,
                                  uint32_t max_ctx, uint32_t kv_dim, uint32_t swa_window,
                                  uint32_t batch, float scale, cudaStream_t stream) {
    constexpr uint32_t TK = HD == 128u ? 32u : 16u;
    constexpr uint32_t QPAD = HD + 4u;
    constexpr uint32_t SMEM =
        (PD_APF_TQ * QPAD + HD * (TK + 1u) + TK * QPAD + PD_APF_TQ) * 4u;
    static cudaError_t attr = cudaFuncSetAttribute(
        (const void*)pd_attn_prefill_kernel<KV, HD>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM);
    if (attr != cudaSuccess) return attr;
    dim3 grid(n_heads, (batch + PD_APF_TQ - 1u) / PD_APF_TQ);
    pd_attn_prefill_kernel<KV, HD><<<grid, 256, SMEM, stream>>>(
        (const float*)q, (const KV*)kc, (const KV*)vc, (const float*)sinks,
        (float*)out, (const unsigned int*)positions, (const unsigned int*)slots,
        n_heads, n_kv_heads, max_ctx, kv_dim, swa_window, batch, scale);
    return pd_launch_status();
}

PD_EXPORT
int pd_attn_prefill(const void* q, const void* kc, const void* vc, const void* sinks,
                    void* out, const void* positions, const void* slots, uint32_t n_heads,
                    uint32_t n_kv_heads, uint32_t head_dim, uint32_t max_ctx, uint32_t kv_dim,
                    uint32_t swa_window, uint32_t batch, float scale, uint32_t kv_dtype,
                    void* stream) {
    if (n_heads == 0 || batch == 0) return 0;
    // 512 = gemma4's global-layer geometry; its smem (TQ 16, TK 16) is
    // 100,928 B - inside sm_120's 101,376 B opt-in cap with 448 B to spare
    if (head_dim != 128u && head_dim != 256u && head_dim != 512u)
        return cudaErrorInvalidValue;
    auto st = (cudaStream_t)stream;
    if (kv_dtype == PD_KV_FP8_E4M3) {
        return head_dim == 128u
            ? pd_attn_prefill_launch<__nv_fp8_e4m3, 128u>(q, kc, vc, sinks, out, positions,
                  slots, n_heads, n_kv_heads, max_ctx, kv_dim, swa_window, batch, scale, st)
            : head_dim == 256u
            ? pd_attn_prefill_launch<__nv_fp8_e4m3, 256u>(q, kc, vc, sinks, out, positions,
                  slots, n_heads, n_kv_heads, max_ctx, kv_dim, swa_window, batch, scale, st)
            : pd_attn_prefill_launch<__nv_fp8_e4m3, 512u>(q, kc, vc, sinks, out, positions,
                  slots, n_heads, n_kv_heads, max_ctx, kv_dim, swa_window, batch, scale, st);
    }
    return head_dim == 128u
        ? pd_attn_prefill_launch<__half, 128u>(q, kc, vc, sinks, out, positions, slots,
              n_heads, n_kv_heads, max_ctx, kv_dim, swa_window, batch, scale, st)
        : head_dim == 256u
        ? pd_attn_prefill_launch<__half, 256u>(q, kc, vc, sinks, out, positions, slots,
              n_heads, n_kv_heads, max_ctx, kv_dim, swa_window, batch, scale, st)
        : pd_attn_prefill_launch<__half, 512u>(q, kc, vc, sinks, out, positions, slots,
              n_heads, n_kv_heads, max_ctx, kv_dim, swa_window, batch, scale, st);
}

// ---------------------------------------- PAGED tiled prefill (P4b)
// Paged twin of pd_attn_prefill_kernel: same tiled flash-prefill (single slot,
// all rows share it), but K/V come from the block pool via the slot's block
// table. The only change is the per-token KV base (kc/vc + slot*max_ctx*kv_dim
// -> block table lookup) - everything else is byte-identical, so it is bit-exact
// vs the dense tiled prefill. Gives paged prefill the tiled perf class instead
// of the decode-class fallback (P4). The WMMA (f16) prefill is P4b-2.
template<typename KV, uint32_t HD>
__global__ void __launch_bounds__(256) pd_attn_prefill_paged_kernel(
    const float* __restrict__ q, const KV* __restrict__ pool_k,
    const KV* __restrict__ pool_v, const float* __restrict__ sinks,
    float* __restrict__ out, const unsigned int* __restrict__ positions,
    const unsigned int* __restrict__ slots,
    const uint32_t* __restrict__ block_tables, uint32_t blocks_per_slot,
    uint32_t n_heads, uint32_t n_kv_heads, uint32_t kv_dim,
    uint32_t swa_window, uint32_t n_rows, float scale) {
    constexpr uint32_t TK = HD == 128u ? 32u : 16u;
    constexpr uint32_t DPL = HD / 32u;
    constexpr uint32_t QPAD = HD + 4u;  // 16B-aligned rows, conflict-free f4
    const uint32_t h = blockIdx.x;
    const uint32_t row0 = blockIdx.y * PD_APF_TQ;
    const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
    const uint32_t kvh = h / (n_heads / n_kv_heads);
    const uint32_t slot = slots ? slots[0] : 0u;

    extern __shared__ float pd_apf_sh[];
    float* sh_q = pd_apf_sh;                            // [TQ][QPAD]
    float* sh_k = sh_q + PD_APF_TQ * QPAD;              // [HD][TK+1] d-major
    float* sh_v = sh_k + HD * (TK + 1u);                // [TK][QPAD]
    uint32_t* sh_hi = (uint32_t*)(sh_v + TK * QPAD);    // [TQ]

    #pragma unroll
    for (uint32_t it = 0; it < PD_APF_TQ * HD / 256u; ++it) {
        const uint32_t i = it * 256u + tid, qi = i / HD, dd = i % HD;
        const uint32_t b = row0 + qi;
        sh_q[qi * QPAD + dd] = b < n_rows ? q[((size_t)b * n_heads + h) * HD + dd] : 0.f;
    }
    if (tid < PD_APF_TQ)
        sh_hi[tid] = (row0 + tid) < n_rows ? positions[row0 + tid] + 1u : 0u;
    __syncthreads();
    uint32_t hi = 0;
    #pragma unroll
    for (uint32_t i = 0; i < PD_APF_TQ; ++i) hi = max(hi, sh_hi[i]);

    const uint32_t qi0 = 2u * warp, qi1 = qi0 + 1u;
    const uint32_t b0 = row0 + qi0, b1 = row0 + qi1;
    const bool live0 = b0 < n_rows, live1 = b1 < n_rows;
    const uint32_t pos0 = live0 ? positions[b0] : 0u;
    const uint32_t pos1 = live1 ? positions[b1] : 0u;
    const uint32_t fp0 = (swa_window > 0 && pos0 + 1u > swa_window) ? pos0 + 1u - swa_window : 0u;
    const uint32_t fp1 = (swa_window > 0 && pos1 + 1u > swa_window) ? pos1 + 1u - swa_window : 0u;
    float m0 = sinks[h], l0 = 1.f, m1 = m0, l1 = 1.f;
    float a0[DPL] = {}, a1[DPL] = {};

    // paged: the slot's block table replaces the dense slot base. Per-token
    // address is resolved in the stage loop below.
    const uint32_t* bt = block_tables + (size_t)slot * blocks_per_slot;

    uint32_t lo_t = 0;
    if (swa_window > 0) {
        uint32_t lo1 = 0xFFFFFFFFu;
        #pragma unroll
        for (uint32_t i = 0; i < PD_APF_TQ; ++i)
            if (sh_hi[i]) lo1 = min(lo1, sh_hi[i]);
        if (lo1 != 0xFFFFFFFFu && lo1 > swa_window)
            lo_t = ((lo1 - swa_window) / TK) * TK;
    }
    for (uint32_t t0 = lo_t; t0 < hi; t0 += TK) {
        __syncthreads();  // previous tile's readers are done
        #pragma unroll
        for (uint32_t it = 0; it < TK * HD / 256u; ++it) {
            const uint32_t i = it * 256u + tid, kk = i / HD, dd = i % HD;
            const uint32_t kp = t0 + kk;
            float kvl = 0.f, vvl = 0.f;
            if (kp < hi) {
                const uint32_t blk = bt[kp >> 4];
                const size_t base = (size_t)blk * 16u * kv_dim + (size_t)(kp & 15u) * kv_dim
                                    + (size_t)kvh * HD + dd;
                kvl = pd_kv_load(pool_k[base]);
                vvl = pd_kv_load(pool_v[base]);
            }
            sh_k[dd * (TK + 1u) + kk] = kvl;
            sh_v[kk * QPAD + dd] = vvl;
        }
        __syncthreads();

        const uint32_t kj = lane % TK;  // TK==32 -> lane
        const uint32_t kp = t0 + kj;
        float s0 = 0.f, s1 = 0.f;
        #pragma unroll 32
        for (uint32_t d = 0; d < HD; ++d) {
            const float kv = sh_k[d * (TK + 1u) + kj];
            s0 = fmaf(sh_q[qi0 * QPAD + d], kv, s0);
            s1 = fmaf(sh_q[qi1 * QPAD + d], kv, s1);
        }
        const bool dup = TK < 32u && lane >= TK;  // duplicate dot, mask it
        const bool v0 = live0 && !dup && kp >= fp0 && kp <= pos0 && kp < hi;
        const bool v1 = live1 && !dup && kp >= fp1 && kp <= pos1 && kp < hi;
        s0 = v0 ? s0 * scale : -1e30f;
        s1 = v1 ? s1 * scale : -1e30f;

        float t0m = s0, t1m = s1;
        #pragma unroll
        for (uint32_t o = 16; o > 0; o >>= 1) {
            t0m = fmaxf(t0m, __shfl_xor_sync(0xffffffffu, t0m, o));
            t1m = fmaxf(t1m, __shfl_xor_sync(0xffffffffu, t1m, o));
        }
        const float n0 = fmaxf(m0, t0m), n1 = fmaxf(m1, t1m);
        const float c0 = __expf(m0 - n0), c1 = __expf(m1 - n1);
        const float w0 = v0 ? __expf(s0 - n0) : 0.f;
        const float w1 = v1 ? __expf(s1 - n1) : 0.f;
        float ws0 = w0, ws1 = w1;
        #pragma unroll
        for (uint32_t o = 16; o > 0; o >>= 1) {
            ws0 += __shfl_xor_sync(0xffffffffu, ws0, o);
            ws1 += __shfl_xor_sync(0xffffffffu, ws1, o);
        }
        l0 = l0 * c0 + ws0;
        l1 = l1 * c1 + ws1;
        m0 = n0;
        m1 = n1;
        #pragma unroll
        for (uint32_t c = 0; c < DPL; ++c) { a0[c] *= c0; a1[c] *= c1; }
        #pragma unroll
        for (uint32_t j = 0; j < TK; ++j) {
            const float wj0 = __shfl_sync(0xffffffffu, w0, j);
            const float wj1 = __shfl_sync(0xffffffffu, w1, j);
            #pragma unroll
            for (uint32_t c = 0; c < DPL; c += 4u) {
                const float4 vv =
                    *reinterpret_cast<const float4*>(&sh_v[j * QPAD + DPL * lane + c]);
                a0[c] = fmaf(wj0, vv.x, a0[c]); a0[c + 1] = fmaf(wj0, vv.y, a0[c + 1]);
                a0[c + 2] = fmaf(wj0, vv.z, a0[c + 2]); a0[c + 3] = fmaf(wj0, vv.w, a0[c + 3]);
                a1[c] = fmaf(wj1, vv.x, a1[c]); a1[c + 1] = fmaf(wj1, vv.y, a1[c + 1]);
                a1[c + 2] = fmaf(wj1, vv.z, a1[c + 2]); a1[c + 3] = fmaf(wj1, vv.w, a1[c + 3]);
            }
        }
    }

    if (live0) {
        #pragma unroll
        for (uint32_t c = 0; c < DPL; c += 4u) {
            float4 o0 = make_float4(a0[c] / l0, a0[c + 1] / l0, a0[c + 2] / l0, a0[c + 3] / l0);
            *reinterpret_cast<float4*>(
                &out[((size_t)b0 * n_heads + h) * HD + DPL * lane + c]) = o0;
        }
    }
    if (live1) {
        #pragma unroll
        for (uint32_t c = 0; c < DPL; c += 4u) {
            float4 o1 = make_float4(a1[c] / l1, a1[c + 1] / l1, a1[c + 2] / l1, a1[c + 3] / l1);
            *reinterpret_cast<float4*>(
                &out[((size_t)b1 * n_heads + h) * HD + DPL * lane + c]) = o1;
        }
    }
}

template<typename KV, uint32_t HD>
static int pd_attn_prefill_paged_launch(const void* q, const void* pool_k, const void* pool_v,
                                        const void* sinks, void* out, const void* positions,
                                        const void* slots, const void* block_tables,
                                        uint32_t blocks_per_slot, uint32_t n_heads,
                                        uint32_t n_kv_heads, uint32_t kv_dim, uint32_t swa_window,
                                        uint32_t batch, float scale, cudaStream_t stream) {
    constexpr uint32_t TK = HD == 128u ? 32u : 16u;
    constexpr uint32_t QPAD = HD + 4u;
    constexpr uint32_t SMEM =
        (PD_APF_TQ * QPAD + HD * (TK + 1u) + TK * QPAD + PD_APF_TQ) * 4u;
    static cudaError_t attr = cudaFuncSetAttribute(
        (const void*)pd_attn_prefill_paged_kernel<KV, HD>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM);
    if (attr != cudaSuccess) return attr;
    dim3 grid(n_heads, (batch + PD_APF_TQ - 1u) / PD_APF_TQ);
    pd_attn_prefill_paged_kernel<KV, HD><<<grid, 256, SMEM, stream>>>(
        (const float*)q, (const KV*)pool_k, (const KV*)pool_v, (const float*)sinks,
        (float*)out, (const unsigned int*)positions, (const unsigned int*)slots,
        (const uint32_t*)block_tables, blocks_per_slot, n_heads, n_kv_heads, kv_dim,
        swa_window, batch, scale);
    return pd_launch_status();
}

PD_EXPORT
int pd_attn_prefill_paged(const void* q, const void* pool_k, const void* pool_v, const void* sinks,
                          void* out, const void* positions, const void* slots,
                          const void* block_tables, uint32_t blocks_per_slot, uint32_t n_heads,
                          uint32_t n_kv_heads, uint32_t head_dim, uint32_t kv_dim,
                          uint32_t swa_window, uint32_t batch, float scale, uint32_t kv_dtype,
                          void* stream) {
    if (n_heads == 0 || batch == 0) return 0;
    // 512 = gemma4's global-layer geometry (its budget-pool prefill rides
    // this); same barely-fits smem math as the dense entry above
    if (head_dim != 128u && head_dim != 256u && head_dim != 512u)
        return cudaErrorInvalidValue;
    auto st = (cudaStream_t)stream;
    if (kv_dtype == PD_KV_FP8_E4M3) {
        return head_dim == 128u
            ? pd_attn_prefill_paged_launch<__nv_fp8_e4m3, 128u>(q, pool_k, pool_v, sinks, out,
                  positions, slots, block_tables, blocks_per_slot, n_heads, n_kv_heads, kv_dim,
                  swa_window, batch, scale, st)
            : head_dim == 256u
            ? pd_attn_prefill_paged_launch<__nv_fp8_e4m3, 256u>(q, pool_k, pool_v, sinks, out,
                  positions, slots, block_tables, blocks_per_slot, n_heads, n_kv_heads, kv_dim,
                  swa_window, batch, scale, st)
            : pd_attn_prefill_paged_launch<__nv_fp8_e4m3, 512u>(q, pool_k, pool_v, sinks, out,
                  positions, slots, block_tables, blocks_per_slot, n_heads, n_kv_heads, kv_dim,
                  swa_window, batch, scale, st);
    }
    return head_dim == 128u
        ? pd_attn_prefill_paged_launch<__half, 128u>(q, pool_k, pool_v, sinks, out, positions,
              slots, block_tables, blocks_per_slot, n_heads, n_kv_heads, kv_dim, swa_window,
              batch, scale, st)
        : head_dim == 256u
        ? pd_attn_prefill_paged_launch<__half, 256u>(q, pool_k, pool_v, sinks, out, positions,
              slots, block_tables, blocks_per_slot, n_heads, n_kv_heads, kv_dim, swa_window,
              batch, scale, st)
        : pd_attn_prefill_paged_launch<__half, 512u>(q, pool_k, pool_v, sinks, out, positions,
              slots, block_tables, blocks_per_slot, n_heads, n_kv_heads, kv_dim, swa_window,
              batch, scale, st);
}

// ------------------------------------------- attn prefill BATCH (multi-slot)
// The encoder runs many short sequences at once; each query row belongs to a
// slot (text). pd_attn_prefill_kernel is the fast tiled flash path but assumes
// one slot for the whole block, so the encoder was stuck on the decode-class
// attn_decode_batch (~17% of the encode). This is that same tiled kernel made
// MULTI-SLOT by per-text tiling: the host builds tile_row0[]/tile_slot[] so
// each 16-query tile lives entirely inside one text (never crossing a slot).
// A tile's rows beyond the text (spilled into the next text) are masked by the
// slots[b]==slot test, so they contribute nothing and are covered by the next
// text's own tile. Same value math / numeric class as attn_prefill (and thus
// attn_decode_batch): f32 dot + scale, per-32-key online softmax.
template<typename KV, uint32_t HD>
__global__ void __launch_bounds__(256) pd_attn_prefill_batch_kernel(
    const float* __restrict__ q, const KV* __restrict__ kc,
    const KV* __restrict__ vc, const float* __restrict__ sinks,
    float* __restrict__ out, const unsigned int* __restrict__ positions,
    const unsigned int* __restrict__ slots, const unsigned int* __restrict__ tile_row0,
    const unsigned int* __restrict__ tile_slot,
    uint32_t n_heads, uint32_t n_kv_heads, uint32_t max_ctx, uint32_t kv_dim,
    uint32_t swa_window, uint32_t n_rows, float scale) {
    constexpr uint32_t TK = HD == 128u ? 32u : 16u;
    constexpr uint32_t DPL = HD / 32u;
    constexpr uint32_t QPAD = HD + 4u;
    const uint32_t h = blockIdx.x;
    const uint32_t row0 = tile_row0[blockIdx.y];
    const uint32_t slot = tile_slot[blockIdx.y];
    const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
    const uint32_t kvh = h / (n_heads / n_kv_heads);

    extern __shared__ float pd_apf_sh[];
    float* sh_q = pd_apf_sh;                            // [TQ][QPAD]
    float* sh_k = sh_q + PD_APF_TQ * QPAD;              // [HD][TK+1] d-major
    float* sh_v = sh_k + HD * (TK + 1u);                // [TK][QPAD]
    uint32_t* sh_hi = (uint32_t*)(sh_v + TK * QPAD);    // [TQ]

    // stage the 16 queries; a row is live only if in range AND in this slot
    #pragma unroll
    for (uint32_t it = 0; it < PD_APF_TQ * HD / 256u; ++it) {
        const uint32_t i = it * 256u + tid, qi = i / HD, dd = i % HD;
        const uint32_t b = row0 + qi;
        const bool live = b < n_rows && slots[b] == slot;
        sh_q[qi * QPAD + dd] = live ? q[((size_t)b * n_heads + h) * HD + dd] : 0.f;
    }
    if (tid < PD_APF_TQ) {
        const uint32_t b = row0 + tid;
        sh_hi[tid] = (b < n_rows && slots[b] == slot) ? positions[b] + 1u : 0u;
    }
    __syncthreads();
    uint32_t hi = 0;
    #pragma unroll
    for (uint32_t i = 0; i < PD_APF_TQ; ++i) hi = max(hi, sh_hi[i]);
    if (hi == 0u) return;  // whole tile masked (should not happen with per-text tiling)

    const uint32_t qi0 = 2u * warp, qi1 = qi0 + 1u;
    const uint32_t b0 = row0 + qi0, b1 = row0 + qi1;
    const bool live0 = b0 < n_rows && slots[b0] == slot;
    const bool live1 = b1 < n_rows && slots[b1] == slot;
    const uint32_t pos0 = live0 ? positions[b0] : 0u;
    const uint32_t pos1 = live1 ? positions[b1] : 0u;
    const uint32_t fp0 = (swa_window > 0 && pos0 + 1u > swa_window) ? pos0 + 1u - swa_window : 0u;
    const uint32_t fp1 = (swa_window > 0 && pos1 + 1u > swa_window) ? pos1 + 1u - swa_window : 0u;
    float m0 = sinks[h], l0 = 1.f, m1 = m0, l1 = 1.f;
    float a0[DPL] = {}, a1[DPL] = {};

    const KV* kcb = kc + (size_t)slot * max_ctx * kv_dim + (size_t)kvh * HD;
    const KV* vcb = vc + (size_t)slot * max_ctx * kv_dim + (size_t)kvh * HD;

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
        for (uint32_t i = 0; i < PD_APF_TQ; ++i)
            if (sh_hi[i]) lo1 = min(lo1, sh_hi[i]);
        if (lo1 != 0xFFFFFFFFu && lo1 > swa_window)
            lo_t = ((lo1 - swa_window) / TK) * TK;
    }
    for (uint32_t t0 = lo_t; t0 < hi; t0 += TK) {
        __syncthreads();
        #pragma unroll
        for (uint32_t it = 0; it < TK * HD / 256u; ++it) {
            const uint32_t i = it * 256u + tid, kk = i / HD, dd = i % HD;
            const uint32_t kp = t0 + kk;
            const float kvl = kp < hi ? pd_kv_load(kcb[(size_t)kp * kv_dim + dd]) : 0.f;
            const float vvl = kp < hi ? pd_kv_load(vcb[(size_t)kp * kv_dim + dd]) : 0.f;
            sh_k[dd * (TK + 1u) + kk] = kvl;
            sh_v[kk * QPAD + dd] = vvl;
        }
        __syncthreads();

        const uint32_t kj = lane % TK;
        const uint32_t kp = t0 + kj;
        float s0 = 0.f, s1 = 0.f;
        #pragma unroll 32
        for (uint32_t d = 0; d < HD; ++d) {
            const float kv = sh_k[d * (TK + 1u) + kj];
            s0 = fmaf(sh_q[qi0 * QPAD + d], kv, s0);
            s1 = fmaf(sh_q[qi1 * QPAD + d], kv, s1);
        }
        const bool dup = TK < 32u && lane >= TK;
        const bool v0 = live0 && !dup && kp >= fp0 && kp <= pos0 && kp < hi;
        const bool v1 = live1 && !dup && kp >= fp1 && kp <= pos1 && kp < hi;
        s0 = v0 ? s0 * scale : -1e30f;
        s1 = v1 ? s1 * scale : -1e30f;

        float t0m = s0, t1m = s1;
        #pragma unroll
        for (uint32_t o = 16; o > 0; o >>= 1) {
            t0m = fmaxf(t0m, __shfl_xor_sync(0xffffffffu, t0m, o));
            t1m = fmaxf(t1m, __shfl_xor_sync(0xffffffffu, t1m, o));
        }
        const float n0 = fmaxf(m0, t0m), n1 = fmaxf(m1, t1m);
        const float c0 = __expf(m0 - n0), c1 = __expf(m1 - n1);
        const float w0 = v0 ? __expf(s0 - n0) : 0.f;
        const float w1 = v1 ? __expf(s1 - n1) : 0.f;
        float ws0 = w0, ws1 = w1;
        #pragma unroll
        for (uint32_t o = 16; o > 0; o >>= 1) {
            ws0 += __shfl_xor_sync(0xffffffffu, ws0, o);
            ws1 += __shfl_xor_sync(0xffffffffu, ws1, o);
        }
        l0 = l0 * c0 + ws0;
        l1 = l1 * c1 + ws1;
        m0 = n0;
        m1 = n1;
        #pragma unroll
        for (uint32_t c = 0; c < DPL; ++c) { a0[c] *= c0; a1[c] *= c1; }
        #pragma unroll
        for (uint32_t j = 0; j < TK; ++j) {
            const float wj0 = __shfl_sync(0xffffffffu, w0, j);
            const float wj1 = __shfl_sync(0xffffffffu, w1, j);
            #pragma unroll
            for (uint32_t c = 0; c < DPL; c += 4u) {
                const float4 vv =
                    *reinterpret_cast<const float4*>(&sh_v[j * QPAD + DPL * lane + c]);
                a0[c] = fmaf(wj0, vv.x, a0[c]); a0[c + 1] = fmaf(wj0, vv.y, a0[c + 1]);
                a0[c + 2] = fmaf(wj0, vv.z, a0[c + 2]); a0[c + 3] = fmaf(wj0, vv.w, a0[c + 3]);
                a1[c] = fmaf(wj1, vv.x, a1[c]); a1[c + 1] = fmaf(wj1, vv.y, a1[c + 1]);
                a1[c + 2] = fmaf(wj1, vv.z, a1[c + 2]); a1[c + 3] = fmaf(wj1, vv.w, a1[c + 3]);
            }
        }
    }

    if (live0) {
        #pragma unroll
        for (uint32_t c = 0; c < DPL; c += 4u) {
            float4 o0 = make_float4(a0[c] / l0, a0[c + 1] / l0, a0[c + 2] / l0, a0[c + 3] / l0);
            *reinterpret_cast<float4*>(
                &out[((size_t)b0 * n_heads + h) * HD + DPL * lane + c]) = o0;
        }
    }
    if (live1) {
        #pragma unroll
        for (uint32_t c = 0; c < DPL; c += 4u) {
            float4 o1 = make_float4(a1[c] / l1, a1[c + 1] / l1, a1[c + 2] / l1, a1[c + 3] / l1);
            *reinterpret_cast<float4*>(
                &out[((size_t)b1 * n_heads + h) * HD + DPL * lane + c]) = o1;
        }
    }
}

template<typename KV, uint32_t HD>
static int pd_attn_prefill_batch_launch(
        const void* q, const void* kc, const void* vc, const void* sinks, void* out,
        const void* positions, const void* slots, const void* tile_row0,
        const void* tile_slot, uint32_t n_qtiles, uint32_t n_heads,
        uint32_t n_kv_heads, uint32_t max_ctx, uint32_t kv_dim, uint32_t swa_window,
        uint32_t n_rows, float scale, cudaStream_t st) {
    constexpr uint32_t TK = HD == 128u ? 32u : 16u;
    constexpr uint32_t QPAD = HD + 4u;
    constexpr uint32_t SMEM =
        (PD_APF_TQ * QPAD + HD * (TK + 1u) + TK * QPAD + PD_APF_TQ) * 4u;
    static cudaError_t attr = cudaFuncSetAttribute(
        (const void*)pd_attn_prefill_batch_kernel<KV, HD>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM);
    if (attr != cudaSuccess) return attr;
    dim3 grid(n_heads, n_qtiles);
    pd_attn_prefill_batch_kernel<KV, HD><<<grid, 256, SMEM, st>>>(
        (const float*)q, (const KV*)kc, (const KV*)vc, (const float*)sinks,
        (float*)out, (const unsigned int*)positions, (const unsigned int*)slots,
        (const unsigned int*)tile_row0, (const unsigned int*)tile_slot,
        n_heads, n_kv_heads, max_ctx, kv_dim, swa_window, n_rows, scale);
    return pd_launch_status();
}

PD_EXPORT
int pd_attn_prefill_batch(const void* q, const void* kc, const void* vc, const void* sinks,
                          void* out, const void* positions, const void* slots,
                          const void* tile_row0, const void* tile_slot, uint32_t n_qtiles,
                          uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
                          uint32_t max_ctx, uint32_t kv_dim, uint32_t swa_window,
                          uint32_t n_rows, float scale, uint32_t kv_dtype, void* stream) {
    if (n_heads == 0 || n_qtiles == 0 || n_rows == 0) return 0;
    // 128 is the encoder's geometry (the original caller); 256 is qwen4_exp's,
    // whose batched-runs PREFILL wave needs a per-TILE slot - the single-slot
    // twin reads slots[0] for every row, which silently attends every run in
    // the wave to the first run's cache. 512 rides along for symmetry with
    // pd_attn_prefill's own dispatch.
    if (head_dim != 128u && head_dim != 256u && head_dim != 512u)
        return cudaErrorInvalidValue;
    auto st = (cudaStream_t)stream;
#define PD_APB(KVT)                                                            \
    return head_dim == 128u                                                    \
        ? pd_attn_prefill_batch_launch<KVT, 128u>(                             \
              q, kc, vc, sinks, out, positions, slots, tile_row0, tile_slot,   \
              n_qtiles, n_heads, n_kv_heads, max_ctx, kv_dim, swa_window,      \
              n_rows, scale, st)                                               \
        : head_dim == 256u                                                     \
        ? pd_attn_prefill_batch_launch<KVT, 256u>(                             \
              q, kc, vc, sinks, out, positions, slots, tile_row0, tile_slot,   \
              n_qtiles, n_heads, n_kv_heads, max_ctx, kv_dim, swa_window,      \
              n_rows, scale, st)                                               \
        : pd_attn_prefill_batch_launch<KVT, 512u>(                             \
              q, kc, vc, sinks, out, positions, slots, tile_row0, tile_slot,   \
              n_qtiles, n_heads, n_kv_heads, max_ctx, kv_dim, swa_window,      \
              n_rows, scale, st)
    if (kv_dtype == PD_KV_FP8_E4M3) { PD_APB(__nv_fp8_e4m3); }
    PD_APB(__half);
#undef PD_APB
}

// ------------------------------------------------------- attn prefill f16
// Tensor-core prefill attention (P6i) - the fattn-wmma class (design studied
// from ggml fattn-wmma-f16.cu; implementation ours). S = Q K^T and O += V P
// both run on f16 WMMA fragments; the online softmax runs in f32 between
// them. Structural notes:
//   - K and V load directly from the fp16 KV cache into fragments (global
//     pointers to load_matrix_sync) - no staging, no conversion.
//   - Q is pre-scaled by `scale`, converted to f16 once, and pinned in
//     matrix_b fragments for the whole kernel (llama's trick; scores come
//     out of the MMA already scaled).
//   - P overwrites the f32 score buffer in PLACE as f16 (half stride =
//     2*row): the same shared bytes serve both GEMMs.
//   - The running O accumulator lives in shared as f16. Per tile it is
//     scaled by the softmax correction, loaded BACK into the accumulator
//     fragments, and the V GEMM accumulates on top - no parts buffer.
//   - Sinks fold in at the epilogue: l += exp(sink - m) with the same
//     max-rebase as the decode kernel, so semantics match it exactly.
//   - exp() flushes to zero below -20 (the fattn FTZ rule) - also what
//     makes masked keys exact zeros; stale cache rows beyond a query's
//     bound are finite garbage times a 0.0 weight, never NaN.
// Numeric class: f16 Q/K/V inputs, f32 score accumulate + softmax, f16 O
// accumulate - llama's own prefill attention class on this hardware.
// Requirements: head_dim == 64 or 256, fp16 KV, max_ctx % 64 == 0, slots
// uniform. (Templated on D: 256 = qwen, 64 = gpt-oss; the V GEMM splits D
// over the 4 warps, so D must be a multiple of 64.)
#define PD_AF16_NCOLS 32  // queries per block
#define PD_AF16_TK 64     // keys per tile (= 4 warps x 16 fragment rows)
template<uint32_t D>
__global__ void __launch_bounds__(128) pd_attn_prefill_f16_kernel(
    const float* __restrict__ q, const __half* __restrict__ kc,
    const __half* __restrict__ vc, const float* __restrict__ sinks,
    float* __restrict__ out, const unsigned int* __restrict__ positions,
    const unsigned int* __restrict__ slots,
    uint32_t n_heads, uint32_t n_kv_heads, uint32_t max_ctx, uint32_t kv_dim,
    uint32_t swa_window, uint32_t n_rows, float scale) {
#if PD_MMA_OK
    using namespace nvcuda;
    // D=512 (gemma4 global): halve the query tile - NC 32's static smem
    // (sh_q+sh_o = 2*32*520 half) would blow the 48 KB static window; NC 16
    // lands at ~38 KB with identical math (RPW = rows each warp owns in the
    // softmax/epilogue scales with NC, so D=64/256 codegen is unchanged).
    constexpr uint32_t NC = D >= 512u ? 16u : PD_AF16_NCOLS;
    constexpr uint32_t TK = PD_AF16_TK;
    constexpr uint32_t RPW = NC / 4u;  // query rows per warp
    constexpr uint32_t DW = D / 4u;    // dims per warp in the V GEMM
    constexpr uint32_t DP = D + 8u;    // half rows, conflict-avoid pad
    constexpr uint32_t KQP = TK + 8u;  // f32 score row stride
    typedef wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_k;
    typedef wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> frag_v;
    typedef wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_b;
    typedef wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_s;
    typedef wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_o;

    const uint32_t h = blockIdx.x;
    const uint32_t row0 = blockIdx.y * NC;
    const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
    const uint32_t kvh = h / (n_heads / n_kv_heads);
    const uint32_t slot = slots ? slots[0] : 0u;

    __shared__ half sh_q[NC * DP];
    __shared__ float sh_s[NC * KQP];   // scores f32; P overwrites as f16
    __shared__ half sh_o[NC * DP];     // running O accumulator
    __shared__ float sh_corr[NC];
    __shared__ float sh_onorm[NC];
    __shared__ uint32_t sh_hi[NC];
    half* sh_p = (half*)sh_s;          // P at half stride 2*KQP, in place

    // stage Q (f32 -> f16, pre-scaled; dead rows 0), zero O, key bounds
    #pragma unroll
    for (uint32_t it = 0; it < NC * D / 128u; ++it) {
        const uint32_t i = it * 128u + tid, j = i / D, dd = i % D;
        const uint32_t b = row0 + j;
        sh_q[j * DP + dd] = __float2half(
            b < n_rows ? q[((size_t)b * n_heads + h) * D + dd] * scale : 0.f);
        sh_o[j * DP + dd] = __float2half(0.f);
    }
    if (tid < NC)
        sh_hi[tid] = (row0 + tid) < n_rows ? positions[row0 + tid] + 1u : 0u;
    __syncthreads();
    uint32_t hi = 0;
    #pragma unroll
    for (uint32_t i = 0; i < NC; ++i) hi = max(hi, sh_hi[i]);

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
            if (sh_hi[i]) lo1 = min(lo1, sh_hi[i]);
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
        for (uint32_t jj = 0; jj < RPW; ++jj) {
            const uint32_t j = warp * RPW + jj;
            const uint32_t b = row0 + j;
            const bool live = b < n_rows;
            const uint32_t pos = live ? positions[b] : 0u;
            const uint32_t fp =
                (swa_window > 0 && pos + 1u > swa_window) ? pos + 1u - swa_window : 0u;
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

        // rescale running O by corr[q] (f16, half2 like llama's VKQ)
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

    // epilogue: fold the sink into l with the same max-rebase as the decode kernel, then
    // publish per-query 1/l (sink-corrected) for the write-out
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
        if (b < n_rows)
            out[((size_t)b * n_heads + h) * D + dd] =
                __half2float(sh_o[j * DP + dd]) * sh_onorm[j];
    }
#else
    (void)q; (void)kc; (void)vc; (void)sinks; (void)out; (void)positions;
    (void)slots; (void)n_heads; (void)n_kv_heads; (void)max_ctx; (void)kv_dim;
    (void)swa_window; (void)n_rows; (void)scale;
#endif
}

// Paged twin of pd_attn_prefill_f16_kernel (P4b-2): the same f16 WMMA prefill,
// but K/V come from the block pool. Page = 16 aligns exactly with the 16-key
// WMMA tile (PD_AF16_TK=64 = 4 warps x 16, and every t0 is TK-aligned), so each
// load_matrix_sync tile is one contiguous block - the FlashInfer BSR property.
// Only the base setup + the two K/V load bases change (dense slot base ->
// block-table lookup); the fragments, mma_sync, and softmax are byte-identical,
// so it is bit-exact vs the dense WMMA prefill.
template<uint32_t D>
__global__ void __launch_bounds__(128) pd_attn_prefill_f16_paged_kernel(
    const float* __restrict__ q, const __half* __restrict__ pool_k,
    const __half* __restrict__ pool_v, const float* __restrict__ sinks,
    float* __restrict__ out, const unsigned int* __restrict__ positions,
    const unsigned int* __restrict__ slots,
    const uint32_t* __restrict__ block_tables, uint32_t blocks_per_slot,
    uint32_t n_heads, uint32_t n_kv_heads, uint32_t kv_dim,
    uint32_t swa_window, uint32_t n_rows, float scale) {
#if PD_MMA_OK
    using namespace nvcuda;
    // D=512 (gemma4 global): halve the query tile - NC 32's static smem
    // (sh_q+sh_o = 2*32*520 half) would blow the 48 KB static window; NC 16
    // lands at ~38 KB with identical math (RPW = rows each warp owns in the
    // softmax/epilogue scales with NC, so D=64/256 codegen is unchanged).
    constexpr uint32_t NC = D >= 512u ? 16u : PD_AF16_NCOLS;
    constexpr uint32_t TK = PD_AF16_TK;
    constexpr uint32_t RPW = NC / 4u;  // query rows per warp
    constexpr uint32_t DW = D / 4u;    // dims per warp in the V GEMM
    constexpr uint32_t DP = D + 8u;    // half rows, conflict-avoid pad
    constexpr uint32_t KQP = TK + 8u;  // f32 score row stride
    typedef wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_k;
    typedef wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> frag_v;
    typedef wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_b;
    typedef wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_s;
    typedef wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_o;

    const uint32_t h = blockIdx.x;
    const uint32_t row0 = blockIdx.y * NC;
    const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
    const uint32_t kvh = h / (n_heads / n_kv_heads);
    const uint32_t slot = slots ? slots[0] : 0u;

    __shared__ half sh_q[NC * DP];
    __shared__ float sh_s[NC * KQP];   // scores f32; P overwrites as f16
    __shared__ half sh_o[NC * DP];     // running O accumulator
    __shared__ float sh_corr[NC];
    __shared__ float sh_onorm[NC];
    __shared__ uint32_t sh_hi[NC];
    half* sh_p = (half*)sh_s;          // P at half stride 2*KQP, in place

    #pragma unroll
    for (uint32_t it = 0; it < NC * D / 128u; ++it) {
        const uint32_t i = it * 128u + tid, j = i / D, dd = i % D;
        const uint32_t b = row0 + j;
        sh_q[j * DP + dd] = __float2half(
            b < n_rows ? q[((size_t)b * n_heads + h) * D + dd] * scale : 0.f);
        sh_o[j * DP + dd] = __float2half(0.f);
    }
    if (tid < NC)
        sh_hi[tid] = (row0 + tid) < n_rows ? positions[row0 + tid] + 1u : 0u;
    __syncthreads();
    uint32_t hi = 0;
    #pragma unroll
    for (uint32_t i = 0; i < NC; ++i) hi = max(hi, sh_hi[i]);

    frag_b Q_b[D / 16u][NC / 16u];
    #pragma unroll
    for (uint32_t d0 = 0; d0 < D / 16u; ++d0)
        #pragma unroll
        for (uint32_t j0 = 0; j0 < NC / 16u; ++j0)
            wmma::load_matrix_sync(Q_b[d0][j0], sh_q + j0 * 16u * DP + d0 * 16u, DP);
    __syncthreads();

    float m_st[NC / 4u], l_st[NC / 4u];
    #pragma unroll
    for (uint32_t jj = 0; jj < NC / 4u; ++jj) { m_st[jj] = -1e30f; l_st[jj] = 0.f; }

    // paged: the slot's block table replaces the dense slot base. Each 16-key
    // tile below is 16-aligned, so it is one contiguous block (bt[key>>4]).
    const uint32_t* bt = block_tables + (size_t)slot * blocks_per_slot;

    uint32_t lo_t = 0;
    if (swa_window > 0) {
        uint32_t lo1 = 0xFFFFFFFFu;
        #pragma unroll
        for (uint32_t i = 0; i < NC; ++i)
            if (sh_hi[i]) lo1 = min(lo1, sh_hi[i]);
        if (lo1 != 0xFFFFFFFFu && lo1 > swa_window)
            lo_t = ((lo1 - swa_window) / TK) * TK;
    }
    for (uint32_t t0 = lo_t; t0 < hi; t0 += TK) {
        {
            frag_s S_c[NC / 16u];
            #pragma unroll
            for (uint32_t j0 = 0; j0 < NC / 16u; ++j0) wmma::fill_fragment(S_c[j0], 0.f);
            #pragma unroll
            for (uint32_t d0 = 0; d0 < D / 16u; ++d0) {
                frag_k K_a;
                // paged K tile: block bt[(t0+16*warp)>>4], within-block row 0.
                wmma::load_matrix_sync(
                    K_a,
                    pool_k + (size_t)bt[(t0 + 16u * warp) >> 4] * 16u * kv_dim
                        + (size_t)kvh * D + d0 * 16u,
                    kv_dim);
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

        #pragma unroll
        for (uint32_t jj = 0; jj < RPW; ++jj) {
            const uint32_t j = warp * RPW + jj;
            const uint32_t b = row0 + j;
            const bool live = b < n_rows;
            const uint32_t pos = live ? positions[b] : 0u;
            const uint32_t fp =
                (swa_window > 0 && pos + 1u > swa_window) ? pos + 1u - swa_window : 0u;
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
            sh_p[j * 2u * KQP + lane] = __float2half(w0);
            sh_p[j * 2u * KQP + 32u + lane] = __float2half(w1);
            if (lane == 0) sh_corr[j] = corr;
        }
        __syncthreads();

        #pragma unroll
        for (uint32_t it = 0; it < NC * D / 2u / 128u; ++it) {
            const uint32_t i = it * 128u + tid, j = i / (D / 2u), d2 = i % (D / 2u);
            half2* o2 = (half2*)(sh_o + j * DP);
            const float c = sh_corr[j];
            o2[d2] = __hmul2(o2[d2], __float2half2_rn(c));
        }
        __syncthreads();

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
                    // paged V tile: block bt[(t0+kf*16)>>4], within-block row 0.
                    wmma::load_matrix_sync(
                        V_a[df],
                        pool_v + (size_t)bt[(t0 + kf * 16u) >> 4] * 16u * kv_dim
                            + (size_t)kvh * D + DW * warp + df * 16u,
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
        if (b < n_rows)
            out[((size_t)b * n_heads + h) * D + dd] =
                __half2float(sh_o[j * DP + dd]) * sh_onorm[j];
    }
#else
    (void)q; (void)pool_k; (void)pool_v; (void)sinks; (void)out; (void)positions;
    (void)slots; (void)block_tables; (void)blocks_per_slot; (void)n_heads;
    (void)n_kv_heads; (void)kv_dim; (void)swa_window; (void)n_rows; (void)scale;
#endif
}

// ---- FA-class prefill attention v2. Same math semantics as the
// WMMA kernel above (online softmax, FTZ at -20, sink fold, SWA start-skip,
// causal bounds from positions[]) but restructured around raw mma.sync
// m16n8k16 with the FlashAttention-2 register posture:
//   - The running O accumulator is PERSISTENT per-warp f32 mma fragments
//     (warp owns dims [64w, 64w+64) x all 32 queries). The WMMA kernel kept O
//     in shared f16 and round-tripped it every 64-key tile (rescale pass +
//     fragment reload + 2 extra barriers, ~64 KB smem traffic/tile) because
//     wmma accumulator layouts are opaque; mma.sync's layout is architectural
//     (c-frag row = lane/4 (+8), col = 2*(lane%4) (+1)), so the softmax
//     correction is an in-register multiply keyed by the fragment's QUERY
//     column, and O accumulates in f32 (a strictly tighter class than the
//     WMMA kernel's f16 O).
//   - 3 barriers/tile (S ready, P ready, tile end) vs 5.
//   - ~27 KB smem (Q pane + scores; O only touches shared once, at the
//     epilogue, reusing the dead Q pane for coalesced writes) => 3 CTAs/SM.
//   - One kernel serves dense and paged: bt == nullptr is the dense slot
//     base; page=16 aligns with every 16-key strip (same BSR property as the
//     paged WMMA twin).
// Env PADDOCK_ATTN_PF_V2=1 routes the f16 entry points here (A/B lever).
#if PD_MMA_OK
__device__ __forceinline__ void pd_af2_mma(float d[4], const uint32_t a[4],
                                           const uint32_t b[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
#endif

template<uint32_t D>
__global__ void __launch_bounds__(128)
pd_attn_prefill_f16_v2_kernel(
    const float* __restrict__ q, const __half* __restrict__ kc,
    const __half* __restrict__ vc, const float* __restrict__ sinks,
    float* __restrict__ out, const unsigned int* __restrict__ positions,
    const unsigned int* __restrict__ slots,
    const uint32_t* __restrict__ block_tables, uint32_t blocks_per_slot,
    uint32_t n_heads, uint32_t n_kv_heads, uint32_t max_ctx, uint32_t kv_dim,
    uint32_t swa_window, uint32_t n_rows, float scale) {
#if PD_MMA_OK
    // D=512 (gemma4 global): halve the query tile - NC 32's static smem
    // (sh_q+sh_o = 2*32*520 half) would blow the 48 KB static window; NC 16
    // lands at ~38 KB with identical math (RPW = rows each warp owns in the
    // softmax/epilogue scales with NC, so D=64/256 codegen is unchanged).
    constexpr uint32_t NC = D >= 512u ? 16u : PD_AF16_NCOLS;
    constexpr uint32_t TK = PD_AF16_TK;
    constexpr uint32_t RPW = NC / 4u;  // query rows per warp
    constexpr uint32_t DW = D / 4u;   // dims this warp owns in PV
    constexpr uint32_t DP = D + 8u;   // half stride, conflict pad
    constexpr uint32_t KQP = TK + 8u; // f32 score row stride

    const uint32_t h = blockIdx.x;
    const uint32_t row0 = blockIdx.y * NC;
    const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
    const uint32_t g8 = lane >> 2, t4 = lane & 3u;
    const uint32_t kvh = h / (n_heads / n_kv_heads);
    const uint32_t slot = slots ? slots[0] : 0u;

    __shared__ half sh_q[NC * DP];
    __shared__ float sh_s[NC * KQP];  // scores f32; P overwrites as f16
    __shared__ float sh_corr[NC];
    __shared__ float sh_onorm[NC];
    __shared__ uint32_t sh_hi[NC];
    half* sh_p = (half*)sh_s;

    // stage Q (f32 -> f16 pre-scaled; dead rows zero) + key bounds
    #pragma unroll
    for (uint32_t it = 0; it < NC * D / 128u; ++it) {
        const uint32_t i = it * 128u + tid, j = i / D, dd = i % D;
        const uint32_t b = row0 + j;
        sh_q[j * DP + dd] = __float2half(
            b < n_rows ? q[((size_t)b * n_heads + h) * D + dd] * scale : 0.f);
    }
    if (tid < NC)
        sh_hi[tid] = (row0 + tid) < n_rows ? positions[row0 + tid] + 1u : 0u;
    __syncthreads();
    uint32_t hi = 0;
    #pragma unroll
    for (uint32_t i = 0; i < NC; ++i) hi = max(hi, sh_hi[i]);

    const uint32_t* bt =
        block_tables ? block_tables + (size_t)slot * blocks_per_slot : nullptr;
    const __half* kcb = kc + (bt ? (size_t)kvh * D
                                 : (size_t)slot * max_ctx * kv_dim + (size_t)kvh * D);
    const __half* vcb = vc + (bt ? (size_t)kvh * D
                                 : (size_t)slot * max_ctx * kv_dim + (size_t)kvh * D);

    // softmax state (this warp owns queries [8*warp, 8*warp+8))
    float m_st[8], l_st[8];
    #pragma unroll
    for (uint32_t jj = 0; jj < 8u; ++jj) { m_st[jj] = -1e30f; l_st[jj] = 0.f; }

    // persistent O: warp dims [DW*warp .. +DW) x 32 queries, f32 c-frags
    float o_acc[DW / 16u][NC / 8u][4];
    #pragma unroll
    for (uint32_t mt = 0; mt < DW / 16u; ++mt)
        #pragma unroll
        for (uint32_t nt = 0; nt < NC / 8u; ++nt)
            #pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) o_acc[mt][nt][e] = 0.f;

    // SWA start-skip (see the WMMA kernel note - bit-exact tile skip)
    uint32_t lo_t = 0;
    if (swa_window > 0) {
        uint32_t lo1 = 0xFFFFFFFFu;
        #pragma unroll
        for (uint32_t i = 0; i < NC; ++i)
            if (sh_hi[i]) lo1 = min(lo1, sh_hi[i]);
        if (lo1 != 0xFFFFFFFFu && lo1 > swa_window)
            lo_t = ((lo1 - swa_window) / TK) * TK;
    }

    for (uint32_t t0 = lo_t; t0 < hi; t0 += TK) {
        // ---- S = Q K^T: this warp's 16-key strip x all 32 queries
        {
            const uint32_t ks = t0 + 16u * warp;  // strip base (16-aligned)
            const __half* kt = kcb + (bt ? (size_t)bt[ks >> 4] * 16u * kv_dim
                                         : (size_t)ks * kv_dim);
            float s_acc[NC / 8u][4];
            #pragma unroll
            for (uint32_t nt = 0; nt < NC / 8u; ++nt)
                #pragma unroll
                for (uint32_t e = 0; e < 4u; ++e) s_acc[nt][e] = 0.f;
            #pragma unroll
            for (uint32_t d0 = 0; d0 < D / 16u; ++d0) {
                // A = K strip rows (m=key, k=dim), row-major from cache
                uint32_t ka[4];
                {
                    const __half* r0p = kt + (size_t)g8 * kv_dim + d0 * 16u + 2u * t4;
                    const __half* r8p = r0p + 8u * kv_dim;
                    ka[0] = *reinterpret_cast<const uint32_t*>(r0p);
                    ka[1] = *reinterpret_cast<const uint32_t*>(r8p);
                    ka[2] = *reinterpret_cast<const uint32_t*>(r0p + 8u);
                    ka[3] = *reinterpret_cast<const uint32_t*>(r8p + 8u);
                }
                #pragma unroll
                for (uint32_t nt = 0; nt < NC / 8u; ++nt) {
                    // B = Q slice (k=dim, n=query) col-major from sh_q
                    uint32_t qb[2];
                    const half* qp = sh_q + (size_t)(nt * 8u + g8) * DP + d0 * 16u + 2u * t4;
                    qb[0] = *reinterpret_cast<const uint32_t*>(qp);
                    qb[1] = *reinterpret_cast<const uint32_t*>(qp + 8u);
                    pd_af2_mma(s_acc[nt], ka, qb);
                }
            }
            // scores to shared: element (r=key g8/+8, c=query 2*t4/+1)
            #pragma unroll
            for (uint32_t nt = 0; nt < NC / 8u; ++nt) {
                const uint32_t qq = nt * 8u + 2u * t4;
                sh_s[(qq + 0u) * KQP + 16u * warp + g8] = s_acc[nt][0];
                sh_s[(qq + 1u) * KQP + 16u * warp + g8] = s_acc[nt][1];
                sh_s[(qq + 0u) * KQP + 16u * warp + g8 + 8u] = s_acc[nt][2];
                sh_s[(qq + 1u) * KQP + 16u * warp + g8 + 8u] = s_acc[nt][3];
            }
        }
        __syncthreads();

        // ---- online softmax (identical math to the WMMA kernel)
        #pragma unroll
        for (uint32_t jj = 0; jj < RPW; ++jj) {
            const uint32_t j = warp * RPW + jj;
            const uint32_t b = row0 + j;
            const bool live = b < n_rows;
            const uint32_t pos = live ? positions[b] : 0u;
            const uint32_t fp =
                (swa_window > 0 && pos + 1u > swa_window) ? pos + 1u - swa_window : 0u;
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
            sh_p[j * 2u * KQP + lane] = __float2half(w0);
            sh_p[j * 2u * KQP + 32u + lane] = __float2half(w1);
            if (lane == 0) sh_corr[j] = corr;
        }
        __syncthreads();

        // ---- O = O*corr + V^T P, all in registers
        {
            // in-register rescale: element column = query
            #pragma unroll
            for (uint32_t nt = 0; nt < NC / 8u; ++nt) {
                const float c0 = sh_corr[nt * 8u + 2u * t4];
                const float c1 = sh_corr[nt * 8u + 2u * t4 + 1u];
                #pragma unroll
                for (uint32_t mt = 0; mt < DW / 16u; ++mt) {
                    o_acc[mt][nt][0] *= c0;
                    o_acc[mt][nt][1] *= c1;
                    o_acc[mt][nt][2] *= c0;
                    o_acc[mt][nt][3] *= c1;
                }
            }
            #pragma unroll
            for (uint32_t kf = 0; kf < TK / 16u; ++kf) {
                const uint32_t ks = t0 + kf * 16u;
                const __half* vt = vcb + (bt ? (size_t)bt[ks >> 4] * 16u * kv_dim
                                             : (size_t)ks * kv_dim);
                #pragma unroll
                for (uint32_t mt = 0; mt < DW / 16u; ++mt) {
                    // A = V^T tile (m=dim, k=key): transposed 2-byte gathers
                    uint32_t va[4];
                    {
                        // A-frag (m=dim, k=key) reg order {r,c}{r+8,c}{r,c+8}
                        // {r+8,c+8}: pairs are ADJACENT KEYS at one dim, so
                        // each half2 is two 2-byte gathers kv_dim apart; +8
                        // dim steps stay within the contiguous V row.
                        const uint32_t dr = DW * warp + mt * 16u + g8;
                        const __half* c0p = vt + (size_t)(2u * t4) * kv_dim + dr;
                        const __half* c1p = c0p + kv_dim;
                        const __half* c8p = c0p + 8u * kv_dim;
                        const __half* c9p = c8p + kv_dim;
                        va[0] = ((uint32_t)__half_as_ushort(c1p[0]) << 16) |
                                (uint32_t)__half_as_ushort(c0p[0]);
                        va[1] = ((uint32_t)__half_as_ushort(c1p[8]) << 16) |
                                (uint32_t)__half_as_ushort(c0p[8]);
                        va[2] = ((uint32_t)__half_as_ushort(c9p[0]) << 16) |
                                (uint32_t)__half_as_ushort(c8p[0]);
                        va[3] = ((uint32_t)__half_as_ushort(c9p[8]) << 16) |
                                (uint32_t)__half_as_ushort(c8p[8]);
                    }
                    #pragma unroll
                    for (uint32_t nt = 0; nt < NC / 8u; ++nt) {
                        uint32_t pb[2];
                        const half* pp =
                            sh_p + (size_t)(nt * 8u + g8) * 2u * KQP + kf * 16u + 2u * t4;
                        pb[0] = *reinterpret_cast<const uint32_t*>(pp);
                        pb[1] = *reinterpret_cast<const uint32_t*>(pp + 8u);
                        pd_af2_mma(o_acc[mt][nt], va, pb);
                    }
                }
            }
        }
        __syncthreads();
    }

    // sink fold (same max-rebase as the WMMA kernel)
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
    // epilogue: stage O to the dead Q pane (f16 after the f32 accumulate has
    // been normalized - write-out precision matches the incumbent's f16 O
    // store) then write coalesced
    half* sh_ostage = sh_q;
    #pragma unroll
    for (uint32_t nt = 0; nt < NC / 8u; ++nt) {
        const uint32_t q0 = nt * 8u + 2u * t4, q1 = q0 + 1u;
        const float n0 = sh_onorm[q0], n1 = sh_onorm[q1];
        #pragma unroll
        for (uint32_t mt = 0; mt < DW / 16u; ++mt) {
            const uint32_t dr = DW * warp + mt * 16u + g8;
            sh_ostage[q0 * DP + dr] = __float2half(o_acc[mt][nt][0] * n0);
            sh_ostage[q1 * DP + dr] = __float2half(o_acc[mt][nt][1] * n1);
            sh_ostage[q0 * DP + dr + 8u] = __float2half(o_acc[mt][nt][2] * n0);
            sh_ostage[q1 * DP + dr + 8u] = __float2half(o_acc[mt][nt][3] * n1);
        }
    }
    __syncthreads();
    #pragma unroll
    for (uint32_t it = 0; it < NC * D / 128u; ++it) {
        const uint32_t i = it * 128u + tid, j = i / D, dd = i % D;
        const uint32_t b = row0 + j;
        if (b < n_rows)
            out[((size_t)b * n_heads + h) * D + dd] = __half2float(sh_ostage[j * DP + dd]);
    }
#else
    (void)q; (void)kc; (void)vc; (void)sinks; (void)out; (void)positions;
    (void)slots; (void)block_tables; (void)blocks_per_slot; (void)n_heads;
    (void)n_kv_heads; (void)max_ctx; (void)kv_dim; (void)swa_window;
    (void)n_rows; (void)scale;
#endif
}

// ---- v3: GQA head-fusion. v2's profiled wall is L2 at 65-66%:
// with grid (n_heads, rowtiles), the 8 q-head blocks sharing a kv-head each
// fetch the same K/V tiles from L2 (8x amplification). v3 fuses them: one
// block = (kv-head, 16 rows) serving all 8 q-heads, K/V staged in shared
// once per 64-key tile. Structure (FA-2 posture, 512 threads = 16 warps as
// (head, D-half) pairs):
//   - S = Q K^T with QUERIES as the mma m-dim: warp owns 16 (row, head)
//     queries, softmax runs entirely in registers (row max/sum via the
//     lane-quad shuffles the c-frag layout affords) - no score pane at all.
//   - O = per-warp persistent f32 fragments over the warp's 128-dim half
//     (hd256 O for a full head is 128 regs/lane - the D-half split keeps
//     Q(64) + O(64) + S(32) under the cap).
//   - The head's D-half-1 warp consumes P + corr via a small smem pane (the
//     only cross-warp traffic); it idles during S (~1/3 of the MMA) - the
//     accepted v3 bubble.
//   - K staged as loaded ([key][dim]); V staged TRANSPOSED ([dim][key]) so
//     the PV B-fragments are clean pair loads. ~90 KB smem, 1 CTA/SM
//     (16 warps resident vs v2's 12).
// Same math semantics and numeric class as v2 (f32 S/O, f16 P, FTZ -20,
// sink fold, SWA start-skip). D=256 only; other shapes fall through.
// Env PADDOCK_ATTN_PF_V3=1 (takes precedence over V2 for hd256).
#define PD_AF3_NR 16u   // query rows per block
#define PD_AF3_TK 32u   // keys per staged tile (32 -> 2 CTAs/SM at hd256)
template<uint32_t D>
__global__ void __launch_bounds__(256)
pd_attn_prefill_f16_v3_kernel(
    const float* __restrict__ q, const __half* __restrict__ kc,
    const __half* __restrict__ vc, const float* __restrict__ sinks,
    float* __restrict__ out, const unsigned int* __restrict__ positions,
    const unsigned int* __restrict__ slots,
    const uint32_t* __restrict__ block_tables, uint32_t blocks_per_slot,
    uint32_t n_heads, uint32_t n_kv_heads, uint32_t max_ctx, uint32_t kv_dim,
    uint32_t swa_window, uint32_t n_rows, float scale) {
#if PD_MMA_OK
    // 256 threads = 8 warps = the kv-head's 8 q-heads, one whole head per
    // warp: S, softmax, P and O all stay in that warp's registers (no P/corr
    // panes, 2 barriers per tile). Q(64) + S(32) + O(128) rides the 255-reg
    // budget of a 256-thread launch - the 512-thread D-split variant capped
    // regs at 128 and spilled itself 2x slower than v2.
    constexpr uint32_t NR = PD_AF3_NR, TK = PD_AF3_TK;
    constexpr uint32_t GH = 8u;
    constexpr uint32_t KP = TK + 8u;
    constexpr uint32_t DPD = D + 8u;

    const uint32_t kvh = blockIdx.x;
    const uint32_t row0 = blockIdx.y * NR;
    const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
    const uint32_t g8 = lane >> 2, t4 = lane & 3u;
    const uint32_t h = kvh * GH + warp;
    const uint32_t slot = slots ? slots[0] : 0u;

    extern __shared__ unsigned char af3sh[];
    __half* sh_k = reinterpret_cast<__half*>(af3sh);   // [TK][DPD]
    __half* sh_vt = sh_k + TK * DPD;                   // [D][KP] transposed
    __shared__ uint32_t sh_hi[NR];

    if (tid < NR)
        sh_hi[tid] = (row0 + tid) < n_rows ? positions[row0 + tid] + 1u : 0u;
    __syncthreads();
    uint32_t hi = 0;
    #pragma unroll
    for (uint32_t i = 0; i < NR; ++i) hi = max(hi, sh_hi[i]);

    const uint32_t jr[2] = {g8, g8 + 8u};
    uint32_t posr[2]; bool liver[2];
    #pragma unroll
    for (uint32_t e = 0; e < 2u; ++e) {
        const uint32_t b = row0 + jr[e];
        liver[e] = b < n_rows;
        posr[e] = liver[e] ? positions[b] : 0u;
    }

    // Q pinned in registers, pre-scaled
    uint32_t qa[D / 16u][4];
    #pragma unroll
    for (uint32_t d0 = 0; d0 < D / 16u; ++d0) {
        #pragma unroll
        for (uint32_t e = 0; e < 2u; ++e) {
            const uint32_t b = row0 + jr[e];
            const float* qp = liver[e]
                ? q + ((size_t)b * n_heads + h) * D + d0 * 16u + 2u * t4
                : nullptr;
            const float q0 = qp ? qp[0] * scale : 0.f;
            const float q1 = qp ? qp[1] * scale : 0.f;
            const float q8v = qp ? qp[8] * scale : 0.f;
            const float q9 = qp ? qp[9] * scale : 0.f;
            qa[d0][e] = ((uint32_t)__half_as_ushort(__float2half(q1)) << 16) |
                        (uint32_t)__half_as_ushort(__float2half(q0));
            qa[d0][e + 2u] = ((uint32_t)__half_as_ushort(__float2half(q9)) << 16) |
                             (uint32_t)__half_as_ushort(__float2half(q8v));
        }
    }

    float m_st[2] = {-1e30f, -1e30f}, l_st[2] = {0.f, 0.f};
    float o_acc[D / 8u][4];
    #pragma unroll
    for (uint32_t nt = 0; nt < D / 8u; ++nt)
        #pragma unroll
        for (uint32_t e = 0; e < 4u; ++e) o_acc[nt][e] = 0.f;

    const uint32_t* bt =
        block_tables ? block_tables + (size_t)slot * blocks_per_slot : nullptr;
    const __half* kcb = kc + (bt ? (size_t)kvh * D
                                 : (size_t)slot * max_ctx * kv_dim + (size_t)kvh * D);
    const __half* vcb = vc + (bt ? (size_t)kvh * D
                                 : (size_t)slot * max_ctx * kv_dim + (size_t)kvh * D);

    uint32_t lo_t = 0;
    if (swa_window > 0) {
        uint32_t lo1 = 0xFFFFFFFFu;
        #pragma unroll
        for (uint32_t i = 0; i < NR; ++i)
            if (sh_hi[i]) lo1 = min(lo1, sh_hi[i]);
        if (lo1 != 0xFFFFFFFFu && lo1 > swa_window)
            lo_t = ((lo1 - swa_window) / TK) * TK;
    }

    for (uint32_t t0 = lo_t; t0 < hi; t0 += TK) {
        // stage K [key][dim] + V transposed [dim][key], all 256 threads
        for (uint32_t u = tid; u < TK * (D / 8u); u += 256u) {
            const uint32_t kk = u / (D / 8u), d8 = (u % (D / 8u)) * 8u;
            const uint32_t ks = t0 + kk;
            const __half* src = (bt ? kcb + ((size_t)bt[ks >> 4] * 16u + (ks & 15u)) * kv_dim
                                    : kcb + (size_t)ks * kv_dim) + d8;
            *reinterpret_cast<uint4*>(sh_k + kk * DPD + d8) =
                *reinterpret_cast<const uint4*>(src);
            const __half* vsrc = (bt ? vcb + ((size_t)bt[ks >> 4] * 16u + (ks & 15u)) * kv_dim
                                     : vcb + (size_t)ks * kv_dim) + d8;
            __half vv[8];
            *reinterpret_cast<uint4*>(vv) = *reinterpret_cast<const uint4*>(vsrc);
            #pragma unroll
            for (uint32_t j = 0; j < 8u; ++j) sh_vt[(d8 + j) * KP + kk] = vv[j];
        }
        __syncthreads();

        // S = Q K^T (c-frags rows = queries)
        float s_acc[TK / 8u][4];
        #pragma unroll
        for (uint32_t nt = 0; nt < TK / 8u; ++nt)
            #pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) s_acc[nt][e] = 0.f;
        #pragma unroll
        for (uint32_t d0 = 0; d0 < D / 16u; ++d0) {
            #pragma unroll
            for (uint32_t nt = 0; nt < TK / 8u; ++nt) {
                uint32_t kb[2];
                const __half* kp = sh_k + (size_t)(nt * 8u + g8) * DPD + d0 * 16u + 2u * t4;
                kb[0] = *reinterpret_cast<const uint32_t*>(kp);
                kb[1] = *reinterpret_cast<const uint32_t*>(kp + 8u);
                pd_af2_mma(s_acc[nt], qa[d0], kb);
            }
        }
        // register online softmax
        float mn[2] = {m_st[0], m_st[1]};
        #pragma unroll
        for (uint32_t nt = 0; nt < TK / 8u; ++nt) {
            const uint32_t kbase = t0 + nt * 8u + 2u * t4;
            #pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) {
                const uint32_t r = e >> 1, kk = kbase + (e & 1u);
                const uint32_t fp = (swa_window > 0 && posr[r] + 1u > swa_window)
                                        ? posr[r] + 1u - swa_window : 0u;
                const bool ok = liver[r] && kk >= fp && kk <= posr[r] && kk < hi;
                if (!ok) s_acc[nt][e] = -1e30f;
                mn[r] = fmaxf(mn[r], s_acc[nt][e]);
            }
        }
        #pragma unroll
        for (uint32_t o = 1; o <= 2u; o <<= 1) {
            mn[0] = fmaxf(mn[0], __shfl_xor_sync(0xffffffffu, mn[0], o));
            mn[1] = fmaxf(mn[1], __shfl_xor_sync(0xffffffffu, mn[1], o));
        }
        float ws[2] = {0.f, 0.f};
        #pragma unroll
        for (uint32_t nt = 0; nt < TK / 8u; ++nt) {
            #pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) {
                const uint32_t r = e >> 1;
                const float d = s_acc[nt][e] - mn[r];
                const float w = d >= -20.f ? __expf(d) : 0.f;
                s_acc[nt][e] = w;
                ws[r] += w;
            }
        }
        #pragma unroll
        for (uint32_t o = 1; o <= 2u; o <<= 1) {
            ws[0] += __shfl_xor_sync(0xffffffffu, ws[0], o);
            ws[1] += __shfl_xor_sync(0xffffffffu, ws[1], o);
        }
        float corr_r[2];
        #pragma unroll
        for (uint32_t r = 0; r < 2u; ++r) {
            const float dc = m_st[r] - mn[r];
            corr_r[r] = dc >= -20.f ? __expf(dc) : 0.f;
            l_st[r] = l_st[r] * corr_r[r] + ws[r];
            m_st[r] = mn[r];
        }
        // O = O*corr + P V, P a-frags packed in-register from the S c-frags
        #pragma unroll
        for (uint32_t nt = 0; nt < D / 8u; ++nt) {
            o_acc[nt][0] *= corr_r[0];
            o_acc[nt][1] *= corr_r[0];
            o_acc[nt][2] *= corr_r[1];
            o_acc[nt][3] *= corr_r[1];
        }
        #pragma unroll
        for (uint32_t kf = 0; kf < TK / 16u; ++kf) {
            uint32_t pa[4];
            {
                const uint32_t c0 = 2u * kf, c1 = 2u * kf + 1u;
                pa[0] = ((uint32_t)__half_as_ushort(__float2half(s_acc[c0][1])) << 16) |
                        (uint32_t)__half_as_ushort(__float2half(s_acc[c0][0]));
                pa[1] = ((uint32_t)__half_as_ushort(__float2half(s_acc[c0][3])) << 16) |
                        (uint32_t)__half_as_ushort(__float2half(s_acc[c0][2]));
                pa[2] = ((uint32_t)__half_as_ushort(__float2half(s_acc[c1][1])) << 16) |
                        (uint32_t)__half_as_ushort(__float2half(s_acc[c1][0]));
                pa[3] = ((uint32_t)__half_as_ushort(__float2half(s_acc[c1][3])) << 16) |
                        (uint32_t)__half_as_ushort(__float2half(s_acc[c1][2]));
            }
            #pragma unroll
            for (uint32_t nt = 0; nt < D / 8u; ++nt) {
                uint32_t vb[2];
                const __half* vp = sh_vt + (size_t)(nt * 8u + g8) * KP + kf * 16u + 2u * t4;
                vb[0] = *reinterpret_cast<const uint32_t*>(vp);
                vb[1] = *reinterpret_cast<const uint32_t*>(vp + 8u);
                pd_af2_mma(o_acc[nt], pa, vb);
            }
        }
        __syncthreads();
    }

    // sink fold + direct write-out (element cols = dims, pairs contiguous)
    float nrm[2];
    #pragma unroll
    for (uint32_t r = 0; r < 2u; ++r) {
        const float sv = sinks[h];
        const float mt = fmaxf(m_st[r], sv);
        const float dm = m_st[r] - mt, ds = sv - mt;
        const float cm = dm >= -20.f ? __expf(dm) : 0.f;
        const float cs = ds >= -20.f ? __expf(ds) : 0.f;
        const float l = l_st[r] * cm + cs;
        nrm[r] = l > 0.f ? cm / l : 0.f;
    }
    #pragma unroll
    for (uint32_t nt = 0; nt < D / 8u; ++nt) {
        const uint32_t dcol = nt * 8u + 2u * t4;
        #pragma unroll
        for (uint32_t r = 0; r < 2u; ++r) {
            const uint32_t b = row0 + jr[r];
            if (b < n_rows) {
                float* op = out + ((size_t)b * n_heads + h) * D + dcol;
                op[0] = o_acc[nt][2u * r + 0u] * nrm[r];
                op[1] = o_acc[nt][2u * r + 1u] * nrm[r];
            }
        }
    }
#else
    (void)q; (void)kc; (void)vc; (void)sinks; (void)out; (void)positions;
    (void)slots; (void)block_tables; (void)blocks_per_slot; (void)n_heads;
    (void)n_kv_heads; (void)max_ctx; (void)kv_dim; (void)swa_window;
    (void)n_rows; (void)scale;
#endif
}

// ---- v3-512: the GQA-fused prefill tile at gemma4's GLOBAL geometry
// (head_dim 512, GQA 8:1). Same posture as v3 (K/V staged once per tile for
// the q-group, register S/O, FA-2 online softmax) but hd512 doubles both the
// pinned Q (128 u32) and the O accumulator (256 f32) past the 255-reg cap -
// so each block takes half the q-group and each head splits across two
// D-HALF warps: 256 threads = 8 warps = 4 heads x 2 halves,
// grid (n_kv, rows/NR, 2 head-pairs). Per warp: Q-half 64 regs + O-half 128.
// The halves' partial S exchange through one smem pane per head (the only
// cross-warp traffic; +1 barrier vs v3), softmax then runs REDUNDANTLY in
// both halves from the summed S - no second exchange. K/V staging costs are
// per 4 heads instead of per q-head (the WMMA<512> tile's 8x re-walk was
// ~50 TF vs 162 for its hd256 sibling).
// Numeric class: f32 S/O, f16 P, FTZ -20, sink fold, SWA start-skip -
// identical to v3/v2; vs the WMMA<512> tile it is the v2-vs-v1 class change
// (f32 O is strictly tighter), gated by the greedy oracle probe.
#define PD_AF3W_NR 16u
#define PD_AF3W_TK 32u
// TQ/TO: f16 q/out planes - same contract as the v3c arms
// (q bit-equal at scale=1, out rounded once at the store).
template<uint32_t D, typename KV = __half, typename TQ = float,
         typename TO = float>  // D = full head_dim (512)
__global__ void __launch_bounds__(256)
pd_attn_prefill_f16_v3w_kernel(
    const TQ* __restrict__ q, const KV* __restrict__ kc,
    const KV* __restrict__ vc, const float* __restrict__ sinks,
    TO* __restrict__ out, const unsigned int* __restrict__ positions,
    const unsigned int* __restrict__ slots,
    const uint32_t* __restrict__ block_tables, uint32_t blocks_per_slot,
    uint32_t n_heads, uint32_t n_kv_heads, uint32_t max_ctx, uint32_t kv_dim,
    uint32_t swa_window, uint32_t n_rows, float scale) {
#if PD_MMA_OK
    constexpr uint32_t NR = PD_AF3W_NR, TK = PD_AF3W_TK;
    constexpr uint32_t HB = 4u;        // heads per block (half the 8-group)
    constexpr uint32_t DH = D / 2u;    // dims per warp
    constexpr uint32_t DPD = D + 8u;

    const uint32_t kvh = blockIdx.x;
    const uint32_t row0 = blockIdx.y * NR;
    const uint32_t pair = blockIdx.z;  // which 4-head half of the q-group
    const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
    const uint32_t g8 = lane >> 2, t4 = lane & 3u;
    const uint32_t hw = warp >> 1;     // head-in-block 0..4
    const uint32_t hf = warp & 1u;     // D-half 0..2
    const uint32_t h = kvh * 8u + pair * HB + hw;
    const uint32_t dbase = hf * DH;    // this warp's dim range [dbase, dbase+DH)
    const uint32_t slot = slots ? slots[0] : 0u;

    extern __shared__ unsigned char af3wsh[];
    __half* sh_k = reinterpret_cast<__half*>(af3wsh);   // [TK][DPD] full D
    // V rides through sh_k after the scores phase (vsmem - the
    // v3s mechanism): K is dead once the partial-S exchange barrier passes,
    // so V restages into the same region with K's coalesced 8B pattern and
    // the PV phase reads smem. Kills the fp8 arm's per-BYTE V loads AND the
    // 4x per-CTA V re-walk (each head read V from L2 independently). Round
    // 2's staged-V loss was a separate buffer (smem doubling -> 1 CTA/SM);
    // sh_k reuse keeps ~49KB and the 2-CTA co-residency. Same converts,
    // same mma order -> bit-identical (checksum-gated).
    // partial-S exchange panes: [HB][2][TK/8][4][32] f32, lane-owned words
    float* sh_sp = reinterpret_cast<float*>(sh_k + (size_t)TK * DPD);
    __shared__ uint32_t sh_hi[NR];

    if (tid < NR)
        sh_hi[tid] = (row0 + tid) < n_rows ? positions[row0 + tid] + 1u : 0u;
    __syncthreads();
    uint32_t hi = 0;
    #pragma unroll
    for (uint32_t i = 0; i < NR; ++i) hi = max(hi, sh_hi[i]);

    const uint32_t jr[2] = {g8, g8 + 8u};
    uint32_t posr[2]; bool liver[2];
    #pragma unroll
    for (uint32_t e = 0; e < 2u; ++e) {
        const uint32_t b = row0 + jr[e];
        liver[e] = b < n_rows;
        posr[e] = liver[e] ? positions[b] : 0u;
    }

    // Q-HALF pinned in registers, pre-scaled
    uint32_t qa[DH / 16u][4];
    #pragma unroll
    for (uint32_t d0 = 0; d0 < DH / 16u; ++d0) {
        #pragma unroll
        for (uint32_t e = 0; e < 2u; ++e) {
            const uint32_t b = row0 + jr[e];
            const TQ* qp = liver[e]
                ? q + ((size_t)b * n_heads + h) * D + dbase + d0 * 16u + 2u * t4
                : nullptr;
            float q0, q1, q8v, q9;
            if constexpr (sizeof(TQ) == 2u) {
                // f16 plane: the pair (qp[0], qp[1]) is 4B-aligned (even
                // element offset) - one .b32 load per pair keeps the f32
                // path's transaction width; expansion is exact.
                float2 f01 = make_float2(0.f, 0.f), f89 = make_float2(0.f, 0.f);
                if (qp) {
                    f01 = __half22float2(*reinterpret_cast<const __half2*>(qp));
                    f89 = __half22float2(*reinterpret_cast<const __half2*>(qp + 8));
                }
                q0 = f01.x * scale; q1 = f01.y * scale;
                q8v = f89.x * scale; q9 = f89.y * scale;
            } else {
                q0 = qp ? (float)qp[0] * scale : 0.f;
                q1 = qp ? (float)qp[1] * scale : 0.f;
                q8v = qp ? (float)qp[8] * scale : 0.f;
                q9 = qp ? (float)qp[9] * scale : 0.f;
            }
            qa[d0][e] = ((uint32_t)__half_as_ushort(__float2half(q1)) << 16) |
                        (uint32_t)__half_as_ushort(__float2half(q0));
            qa[d0][e + 2u] = ((uint32_t)__half_as_ushort(__float2half(q9)) << 16) |
                             (uint32_t)__half_as_ushort(__float2half(q8v));
        }
    }

    float m_st[2] = {-1e30f, -1e30f}, l_st[2] = {0.f, 0.f};
    float o_acc[DH / 8u][4];
    #pragma unroll
    for (uint32_t nt = 0; nt < DH / 8u; ++nt)
        #pragma unroll
        for (uint32_t e = 0; e < 4u; ++e) o_acc[nt][e] = 0.f;

    const uint32_t* bt =
        block_tables ? block_tables + (size_t)slot * blocks_per_slot : nullptr;
    const KV* kcb = kc + (bt ? (size_t)kvh * D
                             : (size_t)slot * max_ctx * kv_dim + (size_t)kvh * D);
    const KV* vcb = vc + (bt ? (size_t)kvh * D
                             : (size_t)slot * max_ctx * kv_dim + (size_t)kvh * D);

    uint32_t lo_t = 0;
    if (swa_window > 0) {
        uint32_t lo1 = 0xFFFFFFFFu;
        #pragma unroll
        for (uint32_t i = 0; i < NR; ++i)
            if (sh_hi[i]) lo1 = min(lo1, sh_hi[i]);
        if (lo1 != 0xFFFFFFFFu && lo1 > swa_window)
            lo_t = ((lo1 - swa_window) / TK) * TK;
    }

    for (uint32_t t0 = lo_t; t0 < hi; t0 += TK) {
        // stage K [key][dim] only, all 256 threads, full D. fp8 caches
        // convert to half here (smem stays half; the mma path is unchanged).
        for (uint32_t u = tid; u < TK * (D / 8u); u += 256u) {
            const uint32_t kk = u / (D / 8u), d8 = (u % (D / 8u)) * 8u;
            const uint32_t ks = t0 + kk;
            const KV* src = (bt ? kcb + ((size_t)bt[ks >> 4] * 16u + (ks & 15u)) * kv_dim
                                : kcb + (size_t)ks * kv_dim) + d8;
            if (sizeof(KV) == 2u) {
                *reinterpret_cast<uint4*>(sh_k + kk * DPD + d8) =
                    *reinterpret_cast<const uint4*>(src);
            } else {
                // fp8: one 8-byte load, convert from registers (the scalar
                // per-byte loads were 8 transactions/group - pf8 -2% net
                // despite the decode-side bandwidth win)
                const uint2 raw = *reinterpret_cast<const uint2*>(src);
                const __nv_fp8_e4m3* b8 = reinterpret_cast<const __nv_fp8_e4m3*>(&raw);
                #pragma unroll
                for (uint32_t j = 0; j < 8u; ++j)
                    sh_k[kk * DPD + d8 + j] = __float2half((float)b8[j]);
            }
        }
        __syncthreads();

        // partial S over this warp's D-half
        float s_acc[TK / 8u][4];
        #pragma unroll
        for (uint32_t nt = 0; nt < TK / 8u; ++nt)
            #pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) s_acc[nt][e] = 0.f;
        #pragma unroll
        for (uint32_t d0 = 0; d0 < DH / 16u; ++d0) {
            #pragma unroll
            for (uint32_t nt = 0; nt < TK / 8u; ++nt) {
                uint32_t kb[2];
                const __half* kp =
                    sh_k + (size_t)(nt * 8u + g8) * DPD + dbase + d0 * 16u + 2u * t4;
                kb[0] = *reinterpret_cast<const uint32_t*>(kp);
                kb[1] = *reinterpret_cast<const uint32_t*>(kp + 8u);
                pd_af2_mma(s_acc[nt], qa[d0], kb);
            }
        }
        // exchange: both halves publish their partials to their own pane,
        // one barrier, then each recomputes the sum as pane0 + pane1 in
        // fixed index order - identical operand order in both halves, so
        // the summed S (and everything downstream) is bit-equal across the
        // pair. Lane-owned words: no transpose, no bank conflicts.
        {
            float* mine = sh_sp + ((size_t)(hw * 2u + hf) * (TK / 8u) * 4u) * 32u;
            #pragma unroll
            for (uint32_t nt = 0; nt < TK / 8u; ++nt)
                #pragma unroll
                for (uint32_t e = 0; e < 4u; ++e)
                    mine[(nt * 4u + e) * 32u + lane] = s_acc[nt][e];
            __syncthreads();
            const float* p0 = sh_sp + ((size_t)(hw * 2u + 0u) * (TK / 8u) * 4u) * 32u;
            const float* p1 = sh_sp + ((size_t)(hw * 2u + 1u) * (TK / 8u) * 4u) * 32u;
            #pragma unroll
            for (uint32_t nt = 0; nt < TK / 8u; ++nt)
                #pragma unroll
                for (uint32_t e = 0; e < 4u; ++e)
                    s_acc[nt][e] = p0[(nt * 4u + e) * 32u + lane]
                                 + p1[(nt * 4u + e) * 32u + lane];
        }
        // restage V into the dead sh_k (all S-mma reads completed at the
        // exchange barrier; sh_sp is a separate region). Same read set as
        // the old direct PV loads - masked keys' V was always read, their
        // P weights are 0. Softmax below is register-only; one barrier
        // before the PV loop publishes the staged tile.
        for (uint32_t u = tid; u < TK * (D / 8u); u += 256u) {
            const uint32_t kk = u / (D / 8u), d8 = (u % (D / 8u)) * 8u;
            const uint32_t ks = t0 + kk;
            const KV* src = (bt ? vcb + ((size_t)bt[ks >> 4] * 16u + (ks & 15u)) * kv_dim
                                : vcb + (size_t)ks * kv_dim) + d8;
            if (sizeof(KV) == 2u) {
                *reinterpret_cast<uint4*>(sh_k + kk * DPD + d8) =
                    *reinterpret_cast<const uint4*>(src);
            } else {
                const uint2 raw = *reinterpret_cast<const uint2*>(src);
                const __nv_fp8_e4m3* b8 = reinterpret_cast<const __nv_fp8_e4m3*>(&raw);
                #pragma unroll
                for (uint32_t j = 0; j < 8u; ++j)
                    sh_k[kk * DPD + d8 + j] = __float2half((float)b8[j]);
            }
        }
        // online softmax, redundant per half (identical inputs -> identical
        // m/l/P in both halves; the fold order matches v3 exactly)
        float mn[2] = {m_st[0], m_st[1]};
        #pragma unroll
        for (uint32_t nt = 0; nt < TK / 8u; ++nt) {
            const uint32_t kbase = t0 + nt * 8u + 2u * t4;
            #pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) {
                const uint32_t r = e >> 1, kk = kbase + (e & 1u);
                const uint32_t fp = (swa_window > 0 && posr[r] + 1u > swa_window)
                                        ? posr[r] + 1u - swa_window : 0u;
                const bool ok = liver[r] && kk >= fp && kk <= posr[r] && kk < hi;
                if (!ok) s_acc[nt][e] = -1e30f;
                mn[r] = fmaxf(mn[r], s_acc[nt][e]);
            }
        }
        #pragma unroll
        for (uint32_t o = 1; o <= 2u; o <<= 1) {
            mn[0] = fmaxf(mn[0], __shfl_xor_sync(0xffffffffu, mn[0], o));
            mn[1] = fmaxf(mn[1], __shfl_xor_sync(0xffffffffu, mn[1], o));
        }
        float ws[2] = {0.f, 0.f};
        #pragma unroll
        for (uint32_t nt = 0; nt < TK / 8u; ++nt) {
            #pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) {
                const uint32_t r = e >> 1;
                const float d = s_acc[nt][e] - mn[r];
                const float w = d >= -20.f ? __expf(d) : 0.f;
                s_acc[nt][e] = w;
                ws[r] += w;
            }
        }
        #pragma unroll
        for (uint32_t o = 1; o <= 2u; o <<= 1) {
            ws[0] += __shfl_xor_sync(0xffffffffu, ws[0], o);
            ws[1] += __shfl_xor_sync(0xffffffffu, ws[1], o);
        }
        float corr_r[2];
        #pragma unroll
        for (uint32_t r = 0; r < 2u; ++r) {
            const float dc = m_st[r] - mn[r];
            corr_r[r] = dc >= -20.f ? __expf(dc) : 0.f;
            l_st[r] = l_st[r] * corr_r[r] + ws[r];
            m_st[r] = mn[r];
        }
        #pragma unroll
        for (uint32_t nt = 0; nt < DH / 8u; ++nt) {
            o_acc[nt][0] *= corr_r[0];
            o_acc[nt][1] *= corr_r[0];
            o_acc[nt][2] *= corr_r[1];
            o_acc[nt][3] *= corr_r[1];
        }
        __syncthreads();   // staged V visible to every warp
        #pragma unroll
        for (uint32_t kf = 0; kf < TK / 16u; ++kf) {
            uint32_t pa[4];
            {
                const uint32_t c0 = 2u * kf, c1 = 2u * kf + 1u;
                pa[0] = ((uint32_t)__half_as_ushort(__float2half(s_acc[c0][1])) << 16) |
                        (uint32_t)__half_as_ushort(__float2half(s_acc[c0][0]));
                pa[1] = ((uint32_t)__half_as_ushort(__float2half(s_acc[c0][3])) << 16) |
                        (uint32_t)__half_as_ushort(__float2half(s_acc[c0][2]));
                pa[2] = ((uint32_t)__half_as_ushort(__float2half(s_acc[c1][1])) << 16) |
                        (uint32_t)__half_as_ushort(__float2half(s_acc[c1][0]));
                pa[3] = ((uint32_t)__half_as_ushort(__float2half(s_acc[c1][3])) << 16) |
                        (uint32_t)__half_as_ushort(__float2half(s_acc[c1][2]));
            }
            // V rows for this kf half-tile from the staged sh_k: the same
            // half values the direct L2 reads produced (fp8 converted once
            // at staging), read at [key][dim] - g8 lanes span 8 consecutive
            // dims = 16B per octet, DPD padding keeps banks spread.
            const uint32_t kl = kf * 16u + 2u * t4;
            #pragma unroll
            for (uint32_t nt = 0; nt < DH / 8u; ++nt) {
                const uint32_t dd = dbase + nt * 8u + g8;
                uint32_t vb[2];
                const __half v0 = sh_k[(size_t)kl * DPD + dd];
                const __half v1 = sh_k[(size_t)(kl + 1u) * DPD + dd];
                const __half v8 = sh_k[(size_t)(kl + 8u) * DPD + dd];
                const __half v9 = sh_k[(size_t)(kl + 9u) * DPD + dd];
                vb[0] = ((uint32_t)__half_as_ushort(v1) << 16) |
                        (uint32_t)__half_as_ushort(v0);
                vb[1] = ((uint32_t)__half_as_ushort(v9) << 16) |
                        (uint32_t)__half_as_ushort(v8);
                pd_af2_mma(o_acc[nt], pa, vb);
            }
        }
        __syncthreads();
    }

    float nrm[2];
    #pragma unroll
    for (uint32_t r = 0; r < 2u; ++r) {
        const float sv = sinks[h];
        const float mt = fmaxf(m_st[r], sv);
        const float dm = m_st[r] - mt, ds = sv - mt;
        const float cm = dm >= -20.f ? __expf(dm) : 0.f;
        const float cs = ds >= -20.f ? __expf(ds) : 0.f;
        const float l = l_st[r] * cm + cs;
        nrm[r] = l > 0.f ? cm / l : 0.f;
    }
    #pragma unroll
    for (uint32_t nt = 0; nt < DH / 8u; ++nt) {
        const uint32_t dcol = dbase + nt * 8u + 2u * t4;
        #pragma unroll
        for (uint32_t r = 0; r < 2u; ++r) {
            const uint32_t b = row0 + jr[r];
            if (b < n_rows) {
                TO* op = out + ((size_t)b * n_heads + h) * D + dcol;
                if constexpr (sizeof(TO) == 2u) {
                    // one .b32 store per adjacent pair (4B-aligned; the
                    // __floats2half2_rn rounding == per-element __float2half)
                    *reinterpret_cast<__half2*>(op) = __floats2half2_rn(
                        o_acc[nt][2u * r + 0u] * nrm[r],
                        o_acc[nt][2u * r + 1u] * nrm[r]);
                } else {
                    op[0] = (TO)(o_acc[nt][2u * r + 0u] * nrm[r]);
                    op[1] = (TO)(o_acc[nt][2u * r + 1u] * nrm[r]);
                }
            }
        }
    }
#else
    (void)q; (void)kc; (void)vc; (void)sinks; (void)out; (void)positions;
    (void)slots; (void)block_tables; (void)blocks_per_slot; (void)n_heads;
    (void)n_kv_heads; (void)max_ctx; (void)kv_dim; (void)swa_window;
    (void)n_rows; (void)scale;
#endif
}

// ---- v3s: group-2 SWA prefill tile (gemma4 geometry: 32q/16kv, hd256) ----
// Purpose: an fp8-CAPABLE SWA prefill attention tile - the WMMA<256> tile
// loads f16 fragments straight from the pool and can never read an fp8
// cache, which blocks SWA-KV fp8 (the ring is 23.5GB of f16 at 32 slots -
// the VRAM that funds the f8 prefill planes).
// Design = v3 minus the D-split (hd256 fits: Q 64 + O 128 regs/warp), plus
// v3w round-2 lessons: K-only smem (TK=16 -> ~34KB -> 2 CTAs/SM), V read
// DIRECT from L2 with per-type converts. 256 threads = 8 warps = 8 q-heads
// covering 4 KV heads; K staged once per 4 heads (the WMMA tile re-walked
// it per q-head). Same math class as v3/v3w: f32 S/O, f16 P, FTZ -20,
// sink fold, SWA start-skip.
#define PD_AF3S_NR 16u
#define PD_AF3S_TK 16u
// TQ/TO: f16 q/out planes - same contract as the v3c arms.
template<typename KV, typename TQ = float, typename TO = float>
__global__ void __launch_bounds__(256)
pd_attn_prefill_f16_v3s_kernel(
    const TQ* __restrict__ q, const KV* __restrict__ kc,
    const KV* __restrict__ vc, const float* __restrict__ sinks,
    TO* __restrict__ out, const unsigned int* __restrict__ positions,
    const unsigned int* __restrict__ slots,
    const uint32_t* __restrict__ block_tables, uint32_t blocks_per_slot,
    uint32_t n_heads, uint32_t n_kv_heads, uint32_t max_ctx, uint32_t kv_dim,
    uint32_t swa_window, uint32_t n_rows, float scale) {
#if PD_MMA_OK
    constexpr uint32_t D = 256u;
    constexpr uint32_t NR = PD_AF3S_NR, TK = PD_AF3S_TK;
    constexpr uint32_t KB = 4u;      // kv heads per block
    constexpr uint32_t DPD = D + 8u;

    const uint32_t kvb = blockIdx.x;         // block of 4 kv heads
    const uint32_t row0 = blockIdx.y * NR;
    const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
    const uint32_t g8 = lane >> 2, t4 = lane & 3u;
    const uint32_t kvl = warp >> 1;          // kv head within block 0..4
    const uint32_t kvh = kvb * KB + kvl;
    const uint32_t h = kvh * 2u + (warp & 1u);
    const uint32_t slot = slots ? slots[0] : 0u;

    extern __shared__ unsigned char af3ssh[];
    __half* sh_k = reinterpret_cast<__half*>(af3ssh);  // [KB][TK][DPD]
    __shared__ uint32_t sh_hi[NR];

    if (tid < NR)
        sh_hi[tid] = (row0 + tid) < n_rows ? positions[row0 + tid] + 1u : 0u;
    __syncthreads();
    uint32_t hi = 0;
    #pragma unroll
    for (uint32_t i = 0; i < NR; ++i) hi = max(hi, sh_hi[i]);

    const uint32_t jr[2] = {g8, g8 + 8u};
    uint32_t posr[2]; bool liver[2];
    #pragma unroll
    for (uint32_t e = 0; e < 2u; ++e) {
        const uint32_t b = row0 + jr[e];
        liver[e] = b < n_rows;
        posr[e] = liver[e] ? positions[b] : 0u;
    }

    uint32_t qa[D / 16u][4];
    #pragma unroll
    for (uint32_t d0 = 0; d0 < D / 16u; ++d0) {
        #pragma unroll
        for (uint32_t e = 0; e < 2u; ++e) {
            const uint32_t b = row0 + jr[e];
            const TQ* qp = liver[e]
                ? q + ((size_t)b * n_heads + h) * D + d0 * 16u + 2u * t4
                : nullptr;
            float q0, q1, q8v, q9;
            if constexpr (sizeof(TQ) == 2u) {
                // f16 plane: the pair (qp[0], qp[1]) is 4B-aligned (even
                // element offset) - one .b32 load per pair keeps the f32
                // path's transaction width; expansion is exact.
                float2 f01 = make_float2(0.f, 0.f), f89 = make_float2(0.f, 0.f);
                if (qp) {
                    f01 = __half22float2(*reinterpret_cast<const __half2*>(qp));
                    f89 = __half22float2(*reinterpret_cast<const __half2*>(qp + 8));
                }
                q0 = f01.x * scale; q1 = f01.y * scale;
                q8v = f89.x * scale; q9 = f89.y * scale;
            } else {
                q0 = qp ? (float)qp[0] * scale : 0.f;
                q1 = qp ? (float)qp[1] * scale : 0.f;
                q8v = qp ? (float)qp[8] * scale : 0.f;
                q9 = qp ? (float)qp[9] * scale : 0.f;
            }
            qa[d0][e] = ((uint32_t)__half_as_ushort(__float2half(q1)) << 16) |
                        (uint32_t)__half_as_ushort(__float2half(q0));
            qa[d0][e + 2u] = ((uint32_t)__half_as_ushort(__float2half(q9)) << 16) |
                             (uint32_t)__half_as_ushort(__float2half(q8v));
        }
    }

    float m_st[2] = {-1e30f, -1e30f}, l_st[2] = {0.f, 0.f};
    float o_acc[D / 8u][4];
    #pragma unroll
    for (uint32_t nt = 0; nt < D / 8u; ++nt)
        #pragma unroll
        for (uint32_t e = 0; e < 4u; ++e) o_acc[nt][e] = 0.f;

    const uint32_t* bt =
        block_tables ? block_tables + (size_t)slot * blocks_per_slot : nullptr;

    uint32_t lo_t = 0;
    if (swa_window > 0) {
        uint32_t lo1 = 0xFFFFFFFFu;
        #pragma unroll
        for (uint32_t i = 0; i < NR; ++i)
            if (sh_hi[i]) lo1 = min(lo1, sh_hi[i]);
        if (lo1 != 0xFFFFFFFFu && lo1 > swa_window)
            lo_t = ((lo1 - swa_window) / TK) * TK;
    }

    for (uint32_t t0 = lo_t; t0 < hi; t0 += TK) {
        // stage K for all 4 kv heads of the block, all 256 threads. fp8
        // converts from an 8-byte register load (the v3w round-2 lesson:
        // never per-byte global loads on the staging path).
        for (uint32_t u = tid; u < KB * TK * (D / 8u); u += 256u) {
            const uint32_t kl = u / (TK * (D / 8u));
            const uint32_t rem = u % (TK * (D / 8u));
            const uint32_t kk = rem / (D / 8u), d8 = (rem % (D / 8u)) * 8u;
            const uint32_t ks = t0 + kk;
            const uint32_t kh = kvb * KB + kl;
            const KV* base = kc + (bt ? (size_t)kh * D
                                      : (size_t)slot * max_ctx * kv_dim + (size_t)kh * D);
            const KV* src = (bt ? base + ((size_t)bt[ks >> 4] * 16u + (ks & 15u)) * kv_dim
                                : base + (size_t)ks * kv_dim) + d8;
            __half* dst = sh_k + ((size_t)kl * TK + kk) * DPD + d8;
            if (sizeof(KV) == 2u) {
                *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
            } else {
                const uint2 raw = *reinterpret_cast<const uint2*>(src);
                // widened cvt: fp8x2 -> half2 hardware pairs - 4
                // cvt + 4 vector stores instead of 8 scalar chains; e4m3 ->
                // f16 is exact both ways (the krs BIT-EQUAL precedent)
                const unsigned short* p16 = reinterpret_cast<const unsigned short*>(&raw);
                #pragma unroll
                for (uint32_t j = 0; j < 4u; ++j)
                    *reinterpret_cast<__half2*>(dst + 2u * j) =
                        __half2(__nv_cvt_fp8x2_to_halfraw2(p16[j], __NV_E4M3));
            }
        }
        __syncthreads();

        float s_acc[TK / 8u][4];
        #pragma unroll
        for (uint32_t nt = 0; nt < TK / 8u; ++nt)
            #pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) s_acc[nt][e] = 0.f;
        const __half* kw = sh_k + (size_t)kvl * TK * DPD;
        #pragma unroll
        for (uint32_t d0 = 0; d0 < D / 16u; ++d0) {
            #pragma unroll
            for (uint32_t nt = 0; nt < TK / 8u; ++nt) {
                uint32_t kb[2];
                const __half* kp = kw + (size_t)(nt * 8u + g8) * DPD + d0 * 16u + 2u * t4;
                kb[0] = *reinterpret_cast<const uint32_t*>(kp);
                kb[1] = *reinterpret_cast<const uint32_t*>(kp + 8u);
                pd_af2_mma(s_acc[nt], qa[d0], kb);
            }
        }
        float mn[2] = {m_st[0], m_st[1]};
        #pragma unroll
        for (uint32_t nt = 0; nt < TK / 8u; ++nt) {
            const uint32_t kbase = t0 + nt * 8u + 2u * t4;
            #pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) {
                const uint32_t r = e >> 1, kk = kbase + (e & 1u);
                const uint32_t fp = (swa_window > 0 && posr[r] + 1u > swa_window)
                                        ? posr[r] + 1u - swa_window : 0u;
                const bool ok = liver[r] && kk >= fp && kk <= posr[r] && kk < hi;
                if (!ok) s_acc[nt][e] = -1e30f;
                mn[r] = fmaxf(mn[r], s_acc[nt][e]);
            }
        }
        #pragma unroll
        for (uint32_t o = 1; o <= 2u; o <<= 1) {
            mn[0] = fmaxf(mn[0], __shfl_xor_sync(0xffffffffu, mn[0], o));
            mn[1] = fmaxf(mn[1], __shfl_xor_sync(0xffffffffu, mn[1], o));
        }
        float ws[2] = {0.f, 0.f};
        #pragma unroll
        for (uint32_t nt = 0; nt < TK / 8u; ++nt) {
            #pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) {
                const uint32_t r = e >> 1;
                const float d = s_acc[nt][e] - mn[r];
                const float w = d >= -20.f ? __expf(d) : 0.f;
                s_acc[nt][e] = w;
                ws[r] += w;
            }
        }
        #pragma unroll
        for (uint32_t o = 1; o <= 2u; o <<= 1) {
            ws[0] += __shfl_xor_sync(0xffffffffu, ws[0], o);
            ws[1] += __shfl_xor_sync(0xffffffffu, ws[1], o);
        }
        float corr_r[2];
        #pragma unroll
        for (uint32_t r = 0; r < 2u; ++r) {
            const float dc = m_st[r] - mn[r];
            corr_r[r] = dc >= -20.f ? __expf(dc) : 0.f;
            l_st[r] = l_st[r] * corr_r[r] + ws[r];
            m_st[r] = mn[r];
        }
        #pragma unroll
        for (uint32_t nt = 0; nt < D / 8u; ++nt) {
            o_acc[nt][0] *= corr_r[0];
            o_acc[nt][1] *= corr_r[0];
            o_acc[nt][2] *= corr_r[1];
            o_acc[nt][3] *= corr_r[1];
        }
        // V through smem (vsmem): the direct
        // path read fp8 V one BYTE per element at stride 8 (1/8 sector use).
        // sh_k is dead once scores are folded - restage it with V using the
        // same coalesced 8B-load pattern as K, and the o-phase reads smem.
        // BIT-IDENTICAL (same converts, same accumulate order; probe: 0 of
        // 1.6M elements differ) at -8.5% on the churn shape (1052 -> 963us).
        // The probe also closed two doors: transposed-V smem loses (store
        // scatter exceeds the wide-read gain, 1205us at best) and the P.V
        // phase's remaining cost is its mma/packing issue stream - the
        // FA-2-falsification "local optimum" verdict stands beyond this.
        __syncthreads();  // all score-phase sh_k reads complete
        for (uint32_t u = tid; u < KB * TK * (D / 8u); u += 256u) {
            const uint32_t kl = u / (TK * (D / 8u));
            const uint32_t rem = u % (TK * (D / 8u));
            const uint32_t kk = rem / (D / 8u), d8 = (rem % (D / 8u)) * 8u;
            const uint32_t ks = t0 + kk;
            const uint32_t kh = kvb * KB + kl;
            const KV* base = vc + (bt ? (size_t)kh * D
                                      : (size_t)slot * max_ctx * kv_dim + (size_t)kh * D);
            const KV* src = (bt ? base + ((size_t)bt[ks >> 4] * 16u + (ks & 15u)) * kv_dim
                                : base + (size_t)ks * kv_dim) + d8;
            __half* dst = sh_k + ((size_t)kl * TK + kk) * DPD + d8;
            if (sizeof(KV) == 2u) {
                *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
            } else {
                const uint2 raw = *reinterpret_cast<const uint2*>(src);
                // widened cvt: fp8x2 -> half2 hardware pairs - 4
                // cvt + 4 vector stores instead of 8 scalar chains; e4m3 ->
                // f16 is exact both ways (the krs BIT-EQUAL precedent)
                const unsigned short* p16 = reinterpret_cast<const unsigned short*>(&raw);
                #pragma unroll
                for (uint32_t j = 0; j < 4u; ++j)
                    *reinterpret_cast<__half2*>(dst + 2u * j) =
                        __half2(__nv_cvt_fp8x2_to_halfraw2(p16[j], __NV_E4M3));
            }
        }
        __syncthreads();
        #pragma unroll
        for (uint32_t kf = 0; kf < TK / 16u; ++kf) {
            uint32_t pa[4];
            {
                const uint32_t c0 = 2u * kf, c1 = 2u * kf + 1u;
                pa[0] = ((uint32_t)__half_as_ushort(__float2half(s_acc[c0][1])) << 16) |
                        (uint32_t)__half_as_ushort(__float2half(s_acc[c0][0]));
                pa[1] = ((uint32_t)__half_as_ushort(__float2half(s_acc[c0][3])) << 16) |
                        (uint32_t)__half_as_ushort(__float2half(s_acc[c0][2]));
                pa[2] = ((uint32_t)__half_as_ushort(__float2half(s_acc[c1][1])) << 16) |
                        (uint32_t)__half_as_ushort(__float2half(s_acc[c1][0]));
                pa[3] = ((uint32_t)__half_as_ushort(__float2half(s_acc[c1][3])) << 16) |
                        (uint32_t)__half_as_ushort(__float2half(s_acc[c1][2]));
            }
            const uint32_t vrow = kf * 16u + 2u * t4;
            const __half* shv = sh_k + (size_t)kvl * TK * DPD;
            #pragma unroll
            for (uint32_t nt = 0; nt < D / 8u; ++nt) {
                const uint32_t dd = nt * 8u + g8;
                uint32_t vb[2];
                const __half v0 = shv[(size_t)vrow * DPD + dd];
                const __half v1 = shv[(size_t)(vrow + 1u) * DPD + dd];
                const __half v8 = shv[(size_t)(vrow + 8u) * DPD + dd];
                const __half v9 = shv[(size_t)(vrow + 9u) * DPD + dd];
                vb[0] = ((uint32_t)__half_as_ushort(v1) << 16) |
                        (uint32_t)__half_as_ushort(v0);
                vb[1] = ((uint32_t)__half_as_ushort(v9) << 16) |
                        (uint32_t)__half_as_ushort(v8);
                pd_af2_mma(o_acc[nt], pa, vb);
            }
        }
        __syncthreads();
    }

    float nrm[2];
    #pragma unroll
    for (uint32_t r = 0; r < 2u; ++r) {
        const float sv = sinks[h];
        const float mt = fmaxf(m_st[r], sv);
        const float dm = m_st[r] - mt, ds = sv - mt;
        const float cm = dm >= -20.f ? __expf(dm) : 0.f;
        const float cs = ds >= -20.f ? __expf(ds) : 0.f;
        const float l = l_st[r] * cm + cs;
        nrm[r] = l > 0.f ? cm / l : 0.f;
    }
    #pragma unroll
    for (uint32_t nt = 0; nt < D / 8u; ++nt) {
        const uint32_t dcol = nt * 8u + 2u * t4;
        #pragma unroll
        for (uint32_t r = 0; r < 2u; ++r) {
            const uint32_t b = row0 + jr[r];
            if (b < n_rows) {
                TO* op = out + ((size_t)b * n_heads + h) * D + dcol;
                if constexpr (sizeof(TO) == 2u) {
                    // one .b32 store per adjacent pair (4B-aligned; the
                    // __floats2half2_rn rounding == per-element __float2half)
                    *reinterpret_cast<__half2*>(op) = __floats2half2_rn(
                        o_acc[nt][2u * r + 0u] * nrm[r],
                        o_acc[nt][2u * r + 1u] * nrm[r]);
                } else {
                    op[0] = (TO)(o_acc[nt][2u * r + 0u] * nrm[r]);
                    op[1] = (TO)(o_acc[nt][2u * r + 1u] * nrm[r]);
                }
            }
        }
    }
#else
    (void)q; (void)kc; (void)vc; (void)sinks; (void)out; (void)positions;
    (void)slots; (void)block_tables; (void)blocks_per_slot; (void)n_heads;
    (void)n_kv_heads; (void)max_ctx; (void)kv_dim; (void)swa_window;
    (void)n_rows; (void)scale;
#endif
}

#define PD_AF3S_SMEM (4u * PD_AF3S_TK * (256u + 8u) * 2u)

// ---- v3c: v3s at the probed tile optimum.
// A comparative attribution put v3s at 11-13% of the f16 mma
// ceiling on the churn shape - TK=16 pays stage+2sync+softmax+o-rescale
// every 16 keys and re-stages each kv head's window once per 16 ROWS. The
// tile ladder (constant math class, same warp mma geometry) found TK=64 /
// KB=1 / NR=64 with K AND V co-staged (V region after K, 2 syncs per tile
// instead of 4, no mid-tile restage) at -35% / 104 TF: overheads amortize
// 4x and each window is staged once per 64 rows. Registers already cap the
// class at 1 CTA/SM (254 regs x 256 thr - the v3s "2 CTAs/SM" note was
// stale), so the 67.6 KB smem is free real estate. Not bit-equal to v3s
// (online-max regroups at the wider tile, maxrel ~4e-3 - the accepted
// regroup class); v16 instantiation of the same proto is bit-equal, which
// pins the clone. Kill: PADDOCK_NO_PF_V3C -> v3s.
//
// Profiled: the class is LATENCY
// bound at 2 warps/scheduler (issue 1/4.6 cyc; stalls: wait 2.06,
// long_scoreboard 1.87, math 1.16, short_sb 0.78) - nothing near a wall
// (compute 30%, DRAM 12%). Two bit-equal rungs land on that profile:
// (1) score B-frags via NON-trans ldmatrix.x4 - K stored [key][dim] is
//     the B[k=dim][n=key] fragment orientation (lane l -> row l/4, cols
//     2(l%4)), so ldmatrix replaces the 256 scalar ld.shared.b32 + addr
//     chains per tile, same values same mma order (-2%).
// (2) fp8 KV staging via cp.async prefetch: the next tile's raw bytes
//     stream into a 32KB smem region under this tile's score/PV compute
//     (cp.async needs no registers - the "deeper K pipelining" door never
//     needed Q-in-smem); the stage pass becomes a smem->smem expand.
//     long_scoreboard 1.87 -> 0.34, inst 93.0M -> 83.7M, -17% total
//     (290.4 -> 240.3us churn). f16 KV keeps the direct-store path (64KB
//     raw tile would blow the 101,376B opt-in cap). Remaining top stall
//     is wait 2.19 = the mma dependency chains - restructure classes
//     already falsified (v3w TK ladder, FA tile); local optimum again.
#define PD_AF3C_TK 64u
#define PD_AF3C_NR 64u
#define PD_AF3C_SMEM (2u * PD_AF3C_TK * (256u + 8u) * 2u)
// fp8 PIPE arm: + raw K+V tile staging bytes (2 pools x TK x 256B)
#define PD_AF3C_SMEM_P8 (PD_AF3C_SMEM + 2u * PD_AF3C_TK * 256u)
// TQ/TO (attention streams): f16 q/out planes. The q path is
// BIT-equal at serve's scale=1.0 - the kernel rounds q to f16 into its
// mma fragments anyway ((float)h expand is exact, *1.0 is identity, and
// __float2half re-rounds to the same value). The out side rounds the f32
// o_acc*nrm once at the store (the class change the serve acceptance gate
// arbitrates).
template<typename KV, typename TQ = float, typename TO = float>
__global__ void __launch_bounds__(256)
pd_attn_prefill_f16_v3c_kernel(
    const TQ* __restrict__ q, const KV* __restrict__ kc,
    const KV* __restrict__ vc, const float* __restrict__ sinks,
    TO* __restrict__ out, const unsigned int* __restrict__ positions,
    const unsigned int* __restrict__ slots,
    const uint32_t* __restrict__ block_tables, uint32_t blocks_per_slot,
    uint32_t n_heads, uint32_t n_kv_heads, uint32_t max_ctx, uint32_t kv_dim,
    uint32_t swa_window, uint32_t n_rows, float scale) {
#if PD_MMA_OK
    constexpr uint32_t D = 256u;
    constexpr uint32_t NR = PD_AF3C_NR, TK = PD_AF3C_TK;
    constexpr uint32_t DPD = D + 8u;

    const uint32_t kvh = blockIdx.x;         // one kv head per CTA
    const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
    const uint32_t g8 = lane >> 2, t4 = lane & 3u;
    const uint32_t qh = (warp >> 2) & 1u;    // q head within group
    const uint32_t rg = warp & 3u;           // row group
    const uint32_t h = kvh * 2u + qh;
    const uint32_t row0 = blockIdx.y * NR + rg * 16u;
    const uint32_t slot = slots ? slots[0] : 0u;

    // fp8 KV runs the cp.async pipe; f16 tiles are 2x the bytes
    // and would not fit a raw stage region under the smem cap
    constexpr bool PIPE = sizeof(KV) == 1u;

    extern __shared__ unsigned char af3csh[];
    __half* sh_k = reinterpret_cast<__half*>(af3csh);          // [TK][DPD]
    __half* sh_v = sh_k + (size_t)TK * DPD;                    // [TK][DPD]
    // PIPE only: next tile's raw fp8 bytes land here via cp.async
    unsigned char* sh_raw = reinterpret_cast<unsigned char*>(sh_v + (size_t)TK * DPD);
    __shared__ uint32_t sh_hi[NR];

    if (tid < NR) {
        const uint32_t b = blockIdx.y * NR + tid;
        sh_hi[tid] = b < n_rows ? positions[b] + 1u : 0u;
    }
    __syncthreads();
    uint32_t hi = 0;
    #pragma unroll
    for (uint32_t i = 0; i < NR; ++i) hi = max(hi, sh_hi[i]);

    const uint32_t jr[2] = {g8, g8 + 8u};
    uint32_t posr[2]; bool liver[2];
    #pragma unroll
    for (uint32_t e = 0; e < 2u; ++e) {
        const uint32_t b = row0 + jr[e];
        liver[e] = b < n_rows;
        posr[e] = liver[e] ? positions[b] : 0u;
    }

    uint32_t qa[D / 16u][4];
    #pragma unroll
    for (uint32_t d0 = 0; d0 < D / 16u; ++d0) {
        #pragma unroll
        for (uint32_t e = 0; e < 2u; ++e) {
            const uint32_t b = row0 + jr[e];
            const TQ* qp = liver[e]
                ? q + ((size_t)b * n_heads + h) * D + d0 * 16u + 2u * t4
                : nullptr;
            float q0, q1, q8v, q9;
            if constexpr (sizeof(TQ) == 2u) {
                // f16 plane: the pair (qp[0], qp[1]) is 4B-aligned (even
                // element offset) - one .b32 load per pair keeps the f32
                // path's transaction width; expansion is exact.
                float2 f01 = make_float2(0.f, 0.f), f89 = make_float2(0.f, 0.f);
                if (qp) {
                    f01 = __half22float2(*reinterpret_cast<const __half2*>(qp));
                    f89 = __half22float2(*reinterpret_cast<const __half2*>(qp + 8));
                }
                q0 = f01.x * scale; q1 = f01.y * scale;
                q8v = f89.x * scale; q9 = f89.y * scale;
            } else {
                q0 = qp ? (float)qp[0] * scale : 0.f;
                q1 = qp ? (float)qp[1] * scale : 0.f;
                q8v = qp ? (float)qp[8] * scale : 0.f;
                q9 = qp ? (float)qp[9] * scale : 0.f;
            }
            qa[d0][e] = ((uint32_t)__half_as_ushort(__float2half(q1)) << 16) |
                        (uint32_t)__half_as_ushort(__float2half(q0));
            qa[d0][e + 2u] = ((uint32_t)__half_as_ushort(__float2half(q9)) << 16) |
                             (uint32_t)__half_as_ushort(__float2half(q8v));
        }
    }

    float m_st[2] = {-1e30f, -1e30f}, l_st[2] = {0.f, 0.f};
    float o_acc[D / 8u][4];
    #pragma unroll
    for (uint32_t nt = 0; nt < D / 8u; ++nt)
        #pragma unroll
        for (uint32_t e = 0; e < 4u; ++e) o_acc[nt][e] = 0.f;

    const uint32_t* bt =
        block_tables ? block_tables + (size_t)slot * blocks_per_slot : nullptr;

    uint32_t lo_t = 0;
    if (swa_window > 0) {
        uint32_t lo1 = 0xFFFFFFFFu;
        #pragma unroll
        for (uint32_t i = 0; i < NR; ++i)
            if (sh_hi[i]) lo1 = min(lo1, sh_hi[i]);
        if (lo1 != 0xFFFFFFFFu && lo1 > swa_window)
            lo_t = ((lo1 - swa_window) / TK) * TK;
    }

    // PIPE: one 16B cp.async per (pool, key, 16B-chunk) issues a whole
    // tile's raw fp8 without touching the register file - one group in
    // flight, consumed at the next loop-top wait
    auto issue_tile = [&](uint32_t t0p) {
        constexpr uint32_t CH = TK * (D / 16u);
        for (uint32_t u = tid; u < 2u * CH; u += 256u) {
            const bool isv = u >= CH;
            const uint32_t ur = isv ? u - CH : u;
            const uint32_t kk = ur / (D / 16u), c16 = (ur % (D / 16u)) * 16u;
            const uint32_t ks = t0p + kk;
            const KV* pool = isv ? vc : kc;
            const KV* base = pool + (bt ? (size_t)kvh * D
                                        : (size_t)slot * max_ctx * kv_dim + (size_t)kvh * D);
            const KV* src = (bt ? base + ((size_t)bt[ks >> 4] * 16u + (ks & 15u)) * kv_dim
                                : base + (size_t)ks * kv_dim) + c16;
            const uint32_t da = (uint32_t)__cvta_generic_to_shared(
                sh_raw + ((size_t)(isv ? TK : 0u) + kk) * D + c16);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                         :: "r"(da), "l"(src));
        }
        asm volatile("cp.async.commit_group;");
    };
    if (PIPE && lo_t < hi) issue_tile(lo_t);

    for (uint32_t t0 = lo_t; t0 < hi; t0 += TK) {
        constexpr uint32_t SPAN = TK * (D / 8u);
        if (PIPE) {
            asm volatile("cp.async.wait_group 0;");
            __syncthreads();
            // raw(t) landed - expand smem->smem with the same widened cvt
            // pairs, so the sh_k/sh_v contents are identical to
            // the direct-store path: the whole pipe is bit-equal
            for (uint32_t u = tid; u < 2u * SPAN; u += 256u) {
                const bool isv = u >= SPAN;
                const uint32_t ur = isv ? u - SPAN : u;
                const uint32_t kk = ur / (D / 8u), d8 = (ur % (D / 8u)) * 8u;
                const uint2 raw = *reinterpret_cast<const uint2*>(
                    sh_raw + ((size_t)(isv ? TK : 0u) + kk) * D + d8);
                __half* dst = (isv ? sh_v : sh_k) + (size_t)kk * DPD + d8;
                const unsigned short* p16 = reinterpret_cast<const unsigned short*>(&raw);
                #pragma unroll
                for (uint32_t j = 0; j < 4u; ++j)
                    *reinterpret_cast<__half2*>(dst + 2u * j) =
                        __half2(__nv_cvt_fp8x2_to_halfraw2(p16[j], __NV_E4M3));
            }
            __syncthreads();
            // raw consumed - prefetch t+1 under this tile's score/PV compute
            if (t0 + TK < hi) issue_tile(t0 + TK);
        } else {
        // co-stage K and V (V region after K) - coalesced 8B loads, fp8
        // converts from the register load (the v3w round-2 staging lesson)
        for (uint32_t u = tid; u < 2u * SPAN; u += 256u) {
            const bool isv = u >= SPAN;
            const uint32_t ur = isv ? u - SPAN : u;
            const uint32_t kk = ur / (D / 8u), d8 = (ur % (D / 8u)) * 8u;
            const uint32_t ks = t0 + kk;
            const KV* pool = isv ? vc : kc;
            const KV* base = pool + (bt ? (size_t)kvh * D
                                        : (size_t)slot * max_ctx * kv_dim + (size_t)kvh * D);
            const KV* src = (bt ? base + ((size_t)bt[ks >> 4] * 16u + (ks & 15u)) * kv_dim
                                : base + (size_t)ks * kv_dim) + d8;
            __half* dst = (isv ? sh_v : sh_k) + (size_t)kk * DPD + d8;
            if (sizeof(KV) == 2u) {
                *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
            } else {
                const uint2 raw = *reinterpret_cast<const uint2*>(src);
                // widened cvt: fp8x2 -> half2 hardware pairs - 4
                // cvt + 4 vector stores instead of 8 scalar chains; e4m3 ->
                // f16 is exact both ways (the krs BIT-EQUAL precedent)
                const unsigned short* p16 = reinterpret_cast<const unsigned short*>(&raw);
                #pragma unroll
                for (uint32_t j = 0; j < 4u; ++j)
                    *reinterpret_cast<__half2*>(dst + 2u * j) =
                        __half2(__nv_cvt_fp8x2_to_halfraw2(p16[j], __NV_E4M3));
            }
        }
        __syncthreads();
        }

        float s_acc[TK / 8u][4];
        #pragma unroll
        for (uint32_t nt = 0; nt < TK / 8u; ++nt)
            #pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) s_acc[nt][e] = 0.f;
        {
            // score B-frags via NON-trans ldmatrix.x4: K stored
            // [key][dim] is the B[k=dim][n=key] fragment orientation (lane
            // l -> row l/4, cols 2(l%4)) - the scalar walk's kb pair is
            // exactly ldmatrix's output, so one x4 feeds 2 consecutive nt.
            // Same values, same mma order -> BIT-equal (the PV precedent).
            const uint32_t lg = lane >> 3u;
            #pragma unroll
            for (uint32_t d0 = 0; d0 < D / 16u; ++d0) {
                #pragma unroll
                for (uint32_t np = 0; np < TK / 16u; ++np) {
                    const __half* kp = sh_k
                        + (size_t)(np * 16u + (lg >> 1) * 8u + (lane & 7u)) * DPD
                        + d0 * 16u + (lg & 1u) * 8u;
                    uint32_t kb4[4];
                    const uint32_t ka = (uint32_t)__cvta_generic_to_shared(kp);
                    asm volatile(
                        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                        : "=r"(kb4[0]), "=r"(kb4[1]), "=r"(kb4[2]), "=r"(kb4[3]) : "r"(ka));
                    uint32_t b0[2] = {kb4[0], kb4[1]}, b1[2] = {kb4[2], kb4[3]};
                    pd_af2_mma(s_acc[np * 2u], qa[d0], b0);
                    pd_af2_mma(s_acc[np * 2u + 1u], qa[d0], b1);
                }
            }
        }
        float mn[2] = {m_st[0], m_st[1]};
        #pragma unroll
        for (uint32_t nt = 0; nt < TK / 8u; ++nt) {
            const uint32_t kbase = t0 + nt * 8u + 2u * t4;
            #pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) {
                const uint32_t r = e >> 1, kk = kbase + (e & 1u);
                const uint32_t fp = (swa_window > 0 && posr[r] + 1u > swa_window)
                                        ? posr[r] + 1u - swa_window : 0u;
                const bool ok = liver[r] && kk >= fp && kk <= posr[r] && kk < hi;
                if (!ok) s_acc[nt][e] = -1e30f;
                mn[r] = fmaxf(mn[r], s_acc[nt][e]);
            }
        }
        #pragma unroll
        for (uint32_t o = 1; o <= 2u; o <<= 1) {
            mn[0] = fmaxf(mn[0], __shfl_xor_sync(0xffffffffu, mn[0], o));
            mn[1] = fmaxf(mn[1], __shfl_xor_sync(0xffffffffu, mn[1], o));
        }
        float ws[2] = {0.f, 0.f};
        #pragma unroll
        for (uint32_t nt = 0; nt < TK / 8u; ++nt) {
            #pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) {
                const uint32_t r = e >> 1;
                const float d = s_acc[nt][e] - mn[r];
                const float w = d >= -20.f ? __expf(d) : 0.f;
                s_acc[nt][e] = w;
                ws[r] += w;
            }
        }
        #pragma unroll
        for (uint32_t o = 1; o <= 2u; o <<= 1) {
            ws[0] += __shfl_xor_sync(0xffffffffu, ws[0], o);
            ws[1] += __shfl_xor_sync(0xffffffffu, ws[1], o);
        }
        float corr_r[2];
        #pragma unroll
        for (uint32_t r = 0; r < 2u; ++r) {
            const float dc = m_st[r] - mn[r];
            corr_r[r] = dc >= -20.f ? __expf(dc) : 0.f;
            l_st[r] = l_st[r] * corr_r[r] + ws[r];
            m_st[r] = mn[r];
        }
        #pragma unroll
        for (uint32_t nt = 0; nt < D / 8u; ++nt) {
            o_acc[nt][0] *= corr_r[0];
            o_acc[nt][1] *= corr_r[0];
            o_acc[nt][2] *= corr_r[1];
            o_acc[nt][3] *= corr_r[1];
        }
        #pragma unroll
        for (uint32_t kf = 0; kf < TK / 16u; ++kf) {
            uint32_t pa[4];
            {
                const uint32_t c0 = 2u * kf, c1 = 2u * kf + 1u;
                pa[0] = ((uint32_t)__half_as_ushort(__float2half(s_acc[c0][1])) << 16) |
                        (uint32_t)__half_as_ushort(__float2half(s_acc[c0][0]));
                pa[1] = ((uint32_t)__half_as_ushort(__float2half(s_acc[c0][3])) << 16) |
                        (uint32_t)__half_as_ushort(__float2half(s_acc[c0][2]));
                pa[2] = ((uint32_t)__half_as_ushort(__float2half(s_acc[c1][1])) << 16) |
                        (uint32_t)__half_as_ushort(__float2half(s_acc[c1][0]));
                pa[3] = ((uint32_t)__half_as_ushort(__float2half(s_acc[c1][3])) << 16) |
                        (uint32_t)__half_as_ushort(__float2half(s_acc[c1][2]));
            }
            // PV B-fragments via ldmatrix.x4.trans - one instruction per 2
            // dim-tiles replaces 8 scalar half loads + 4 packs (the vprobe's
            // "mma/packing issue stream"; lv64 measured 578 -> 466 us, and the
            // 528 B row stride lands the 8-row phases in distinct banks).
            // BIT-equal to the scalar form - same values, same mma order.
            const uint32_t lg = lane >> 3;
            const uint32_t vr = kf * 16u + (lg & 1u) * 8u + (lane & 7u);
            const __half* vp = sh_v + (size_t)vr * DPD + (lg >> 1) * 8u;
            #pragma unroll
            for (uint32_t nt = 0; nt < D / 8u; nt += 2u) {
                uint32_t vb4[4];
                const uint32_t va = (uint32_t)__cvta_generic_to_shared(vp + nt * 8u);
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                    : "=r"(vb4[0]), "=r"(vb4[1]), "=r"(vb4[2]), "=r"(vb4[3]) : "r"(va));
                uint32_t b0[2] = {vb4[0], vb4[1]}, b1[2] = {vb4[2], vb4[3]};
                pd_af2_mma(o_acc[nt], pa, b0);
                pd_af2_mma(o_acc[nt + 1u], pa, b1);
            }
        }
        __syncthreads();  // K/V reads done before the next co-stage
    }

    float nrm[2];
    #pragma unroll
    for (uint32_t r = 0; r < 2u; ++r) {
        const float sv = sinks[h];
        const float mt = fmaxf(m_st[r], sv);
        const float dm = m_st[r] - mt, ds = sv - mt;
        const float cm = dm >= -20.f ? __expf(dm) : 0.f;
        const float cs = ds >= -20.f ? __expf(ds) : 0.f;
        const float l = l_st[r] * cm + cs;
        nrm[r] = l > 0.f ? cm / l : 0.f;
    }
    #pragma unroll
    for (uint32_t nt = 0; nt < D / 8u; ++nt) {
        const uint32_t dcol = nt * 8u + 2u * t4;
        #pragma unroll
        for (uint32_t r = 0; r < 2u; ++r) {
            const uint32_t b = row0 + jr[r];
            if (b < n_rows) {
                TO* op = out + ((size_t)b * n_heads + h) * D + dcol;
                if constexpr (sizeof(TO) == 2u) {
                    // one .b32 store per adjacent pair (4B-aligned; the
                    // __floats2half2_rn rounding == per-element __float2half)
                    *reinterpret_cast<__half2*>(op) = __floats2half2_rn(
                        o_acc[nt][2u * r + 0u] * nrm[r],
                        o_acc[nt][2u * r + 1u] * nrm[r]);
                } else {
                    op[0] = (TO)(o_acc[nt][2u * r + 0u] * nrm[r]);
                    op[1] = (TO)(o_acc[nt][2u * r + 1u] * nrm[r]);
                }
            }
        }
    }
#else
    (void)q; (void)kc; (void)vc; (void)sinks; (void)out; (void)positions;
    (void)slots; (void)block_tables; (void)blocks_per_slot; (void)n_heads;
    (void)n_kv_heads; (void)max_ctx; (void)kv_dim; (void)swa_window;
    (void)n_rows; (void)scale;
#endif
}

#define PD_AF3W_SMEM                                                          \
    ((PD_AF3W_TK * (512u + 8u)) * 2u +                                        \
     4u * 2u * (PD_AF3W_TK / 8u) * 4u * 32u * 4u)

#define PD_AF3_SMEM                                                            \
    ((PD_AF3_TK * (256u + 8u) + 256u * (PD_AF3_TK + 8u)) * 2u)

PD_EXPORT
int pd_attn_prefill_f16(const void* q, const void* kc, const void* vc, const void* sinks,
                        void* out, const void* positions, const void* slots, uint32_t n_heads,
                        uint32_t n_kv_heads, uint32_t head_dim, uint32_t max_ctx, uint32_t kv_dim,
                        uint32_t swa_window, uint32_t batch, float scale, uint32_t kv_dtype,
                        void* stream) {
    if (n_heads == 0 || batch == 0) return 0;
    if ((head_dim != 256u && head_dim != 64u && head_dim != 512u) ||
        kv_dtype == PD_KV_FP8_E4M3 || (max_ctx & 63u))
        return cudaErrorInvalidValue;
    static bool carveout_done = false;
    if (!carveout_done) {
        pd_prefer_max_shared(pd_attn_prefill_f16_kernel<256u>);
        pd_prefer_max_shared(pd_attn_prefill_f16_kernel<64u>);
        pd_prefer_max_shared(pd_attn_prefill_f16_kernel<512u>);
        carveout_done = true;
    }
    // 512 runs the NC=16 tile (see the kernel's smem note)
    const uint32_t nc = head_dim >= 512u ? 16u : PD_AF16_NCOLS;
    dim3 grid(n_heads, (batch + nc - 1u) / nc);
    if (head_dim == 512u) {
        pd_attn_prefill_f16_kernel<512u><<<grid, 128, 0, (cudaStream_t)stream>>>(
            (const float*)q, (const __half*)kc, (const __half*)vc, (const float*)sinks,
            (float*)out, (const unsigned int*)positions, (const unsigned int*)slots,
            n_heads, n_kv_heads, max_ctx, kv_dim, swa_window, batch, scale);
        return pd_launch_status();
    }
    static const bool v3 = pd_env("PADDOCK_ATTN_PF_V3") != nullptr;
    if (v3 && head_dim == 256u && n_heads == 8u * n_kv_heads && batch > 0) {
        static bool a3 = false;
        if (!a3) {
            cudaFuncSetAttribute((const void*)pd_attn_prefill_f16_v3_kernel<256u>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, PD_AF3_SMEM);
            a3 = true;
        }
        dim3 g3(n_kv_heads, (batch + PD_AF3_NR - 1u) / PD_AF3_NR);
        pd_attn_prefill_f16_v3_kernel<256u><<<g3, 256, PD_AF3_SMEM, (cudaStream_t)stream>>>(
            (const float*)q, (const __half*)kc, (const __half*)vc, (const float*)sinks,
            (float*)out, (const unsigned int*)positions, (const unsigned int*)slots,
            nullptr, 0, n_heads, n_kv_heads, max_ctx, kv_dim, swa_window, batch, scale);
        return pd_launch_status();
    }
    static const bool v2 = pd_env("PADDOCK_ATTN_PF_V2") != nullptr;
    if (v2) {
        if (head_dim == 256u)
            pd_attn_prefill_f16_v2_kernel<256u><<<grid, 128, 0, (cudaStream_t)stream>>>(
                (const float*)q, (const __half*)kc, (const __half*)vc, (const float*)sinks,
                (float*)out, (const unsigned int*)positions, (const unsigned int*)slots,
                nullptr, 0, n_heads, n_kv_heads, max_ctx, kv_dim, swa_window, batch, scale);
        else
            pd_attn_prefill_f16_v2_kernel<64u><<<grid, 128, 0, (cudaStream_t)stream>>>(
                (const float*)q, (const __half*)kc, (const __half*)vc, (const float*)sinks,
                (float*)out, (const unsigned int*)positions, (const unsigned int*)slots,
                nullptr, 0, n_heads, n_kv_heads, max_ctx, kv_dim, swa_window, batch, scale);
        return pd_launch_status();
    }
    if (head_dim == 256u) {
        pd_attn_prefill_f16_kernel<256u><<<grid, 128, 0, (cudaStream_t)stream>>>(
            (const float*)q, (const __half*)kc, (const __half*)vc, (const float*)sinks,
            (float*)out, (const unsigned int*)positions, (const unsigned int*)slots,
            n_heads, n_kv_heads, max_ctx, kv_dim, swa_window, batch, scale);
    } else {
        pd_attn_prefill_f16_kernel<64u><<<grid, 128, 0, (cudaStream_t)stream>>>(
            (const float*)q, (const __half*)kc, (const __half*)vc, (const float*)sinks,
            (float*)out, (const unsigned int*)positions, (const unsigned int*)slots,
            n_heads, n_kv_heads, max_ctx, kv_dim, swa_window, batch, scale);
    }
    return pd_launch_status();
}

