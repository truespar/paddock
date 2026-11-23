// attn/prefill_tree.cuh - a draft TREE's verify on the paged pools: its
// attention (the pf7rp wide rung with the tree mask, slot 975) and the KV
// compaction that moves an accepted path's rows down to their true positions
// (slot 976). Follows prefill_fa2.cuh (the kernel) in pack.cu's order.
//
// A tree round appends node j of a slot's tree at cache slot `base + j`
// (base = the slot's committed end), so the round's rows never collide; a
// node's logical position is `base + depth`, which RoPE and the sliding
// window read. After the walk accepts path nodes n_1 < n_2 < ... (node
// indices grow with depth - the tree is laid out parents first), node n_k's
// K/V move from `base + n_k` to `base + k`: every move's source sits at or
// past its destination and past every earlier move's destination, so moving
// in depth order never reads a row an earlier move wrote.

// slot 975: one same-slot run of tree rows (q / out / positions / win_pos /
// tmask / slots already offset to the run), hd128, fp8 or f16 pools, the
// ratios the wide rung instantiates. positions = cache slots, win_pos =
// logical positions, tmask = (ancestor mask, base) a row. -2 = declined.
PD_EXPORT
int pd_attn_prefill_tree(const void* q, const void* pool_k, const void* pool_v,
                         const void* sinks, void* out, const void* positions,
                         const void* win_pos, const void* tmask, const void* block_tables,
                         uint32_t blocks_per_slot, const void* slots, uint32_t n_heads,
                         uint32_t n_kv_heads, uint32_t head_dim, uint32_t kv_dim,
                         uint32_t swa_window, uint32_t rows, float scale, uint32_t kv_dtype,
                         void* stream) {
    // (no PD_FA_OK here: that is the DEVICE pass's arch test - the kernel's
    // body carries it; a host launcher behind it declines everywhere)
    if (rows == 0u) return 0;
    if (head_dim != 128u || n_kv_heads == 0u || n_heads % n_kv_heads) return -2;
    const uint32_t g_ = n_heads / n_kv_heads;
    if (!(g_ == 4u || g_ == 6u || g_ == 8u || g_ == 9u || g_ == 10u || g_ == 12u || g_ == 16u))
        return -2;
    constexpr uint32_t WMR = 128u, WTK = 32u, WHD = 128u;
    const dim3 gr(n_kv_heads, (rows * g_ + WMR - 1u) / WMR);
    const auto st = (cudaStream_t)stream;
    if (kv_dtype == PD_KV_FP8_E4M3) {
        constexpr uint32_t WSM = WMR * WHD * 2u + WTK * (WHD + 8u) * 2u + 2u * WTK * WHD;
#define PD_TREE_FP8(GV)                                                              \
        {                                                                            \
            auto kern = pd_attn_prefill_pf7rp_kernel<GV, false, WHD, WTK, 2u, false,  \
                                                      false, false, true>;           \
            static const bool fits = pd_smem_fits((const void*)kern, WSM);          \
            if (!fits) return -2;                                                    \
            kern<<<gr, 128, WSM, st>>>(                                              \
                (const float*)q, (const unsigned char*)pool_k,                       \
                (const unsigned char*)pool_v, (const float*)sinks, (float*)out,      \
                (const unsigned int*)positions, (const uint32_t*)block_tables,       \
                blocks_per_slot, (const unsigned int*)slots, n_heads, kv_dim,        \
                swa_window, rows, scale, nullptr, nullptr,                           \
                (const unsigned int*)win_pos, (const uint32_t*)tmask);               \
        }
        if (g_ == 4u) PD_TREE_FP8(4u)
        else if (g_ == 6u) PD_TREE_FP8(6u)
        else if (g_ == 8u) PD_TREE_FP8(8u)
        else if (g_ == 9u) PD_TREE_FP8(9u)
        else if (g_ == 10u) PD_TREE_FP8(10u)
        else if (g_ == 12u) PD_TREE_FP8(12u)
        else PD_TREE_FP8(16u)
#undef PD_TREE_FP8
        return pd_launch_status();
    }
    if (kv_dtype != PD_KV_FP16) return -2;
    constexpr uint32_t WSM16 = WMR * WHD * 2u + 2u * WTK * (WHD + 8u) * 2u;
#define PD_TREE_F16(GV)                                                              \
    {                                                                                \
        auto kern = pd_attn_prefill_pf7rp_kernel<GV, false, WHD, WTK, 2u, true,       \
                                                  false, false, true>;               \
        static const bool fits = pd_smem_fits((const void*)kern, WSM16);            \
        if (!fits) return -2;                                                        \
        kern<<<gr, 128, WSM16, st>>>(                                                \
            (const float*)q, (const unsigned char*)pool_k,                           \
            (const unsigned char*)pool_v, (const float*)sinks, (float*)out,          \
            (const unsigned int*)positions, (const uint32_t*)block_tables,           \
            blocks_per_slot, (const unsigned int*)slots, n_heads, kv_dim,            \
            swa_window, rows, scale, nullptr, nullptr,                               \
            (const unsigned int*)win_pos, (const uint32_t*)tmask);                   \
    }
    if (g_ == 4u) PD_TREE_F16(4u)
    else if (g_ == 6u) PD_TREE_F16(6u)
    else if (g_ == 8u) PD_TREE_F16(8u)
    else if (g_ == 9u) PD_TREE_F16(9u)
    else if (g_ == 10u) PD_TREE_F16(10u)
    else if (g_ == 12u) PD_TREE_F16(12u)
    else PD_TREE_F16(16u)
#undef PD_TREE_F16
    return pd_launch_status();
}

// slot 976: the accepted paths' K/V rows down to their true positions, every
// layer in one launch. `plan` is n_groups records of `stride` u32: [slot, n,
// src_0, dst_0, ..., src_{n-1}, dst_{n-1}] (cache positions, in depth
// order). Per layer: K and V pool bases (`kp` / `vp`, device pointers as
// u64) and its table (`kind` 0 = bt0 / bps0, 1 = bt1 / bps1). grid (n_groups,
// layers, 2): one CTA moves one group's rows of one pool, in order, a
// barrier between moves; `row_bytes` = kv_dim * the element size.
__global__ void pd_kv_tree_move_kernel(const unsigned long long* __restrict__ kp,
                                       const unsigned long long* __restrict__ vp,
                                       const uint32_t* __restrict__ kind,
                                       const uint32_t* __restrict__ bt0, uint32_t bps0,
                                       const uint32_t* __restrict__ bt1, uint32_t bps1,
                                       const uint32_t* __restrict__ plan, uint32_t stride,
                                       uint32_t row_bytes) {
    const uint32_t* g = plan + (size_t)blockIdx.x * stride;
    const uint32_t l = blockIdx.y;
    unsigned char* pool =
        (unsigned char*)(blockIdx.z == 0u ? kp[l] : vp[l]);
    const bool swa = kind[l] != 0u;
    const uint32_t* bt = swa ? bt1 : bt0;
    const uint32_t bps = swa ? bps1 : bps0;
    const uint32_t slot = g[0], n = g[1];
    const uint32_t chunks = row_bytes / 16u;
    for (uint32_t m = 0; m < n; ++m) {
        const uint32_t src = g[2u + 2u * m], dst = g[3u + 2u * m];
        if (src != dst) {
            const uint32_t sb = bt[(size_t)slot * bps + (src >> 4)];
            const uint32_t db = bt[(size_t)slot * bps + (dst >> 4)];
            const uint4* s4 = (const uint4*)(pool + ((size_t)sb * 16u + (src & 15u)) * row_bytes);
            uint4* d4 = (uint4*)(pool + ((size_t)db * 16u + (dst & 15u)) * row_bytes);
            for (uint32_t c = threadIdx.x; c < chunks; c += blockDim.x) d4[c] = s4[c];
        }
        __syncthreads();
    }
}

PD_EXPORT
int pd_kv_tree_move(const void* kp, const void* vp, const void* kind, uint32_t layers,
                    const void* bt0, uint32_t bps0, const void* bt1, uint32_t bps1,
                    const void* plan, uint32_t n_groups, uint32_t stride, uint32_t row_bytes,
                    void* stream) {
    if (n_groups == 0u || layers == 0u) return 0;
    if (row_bytes % 16u || stride < 2u) return cudaErrorInvalidValue;
    pd_kv_tree_move_kernel<<<dim3(n_groups, layers, 2u), 128u, 0, (cudaStream_t)stream>>>(
        (const unsigned long long*)kp, (const unsigned long long*)vp, (const uint32_t*)kind,
        (const uint32_t*)bt0, bps0, (const uint32_t*)bt1, bps1, (const uint32_t*)plan, stride,
        row_bytes);
    return pd_launch_status();
}
