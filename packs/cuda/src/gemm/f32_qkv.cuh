#include <cooperative_groups.h>
// gemm/f32_qkv.cuh (formerly 04_f32_gemm_qkv.cuh) - f32 tiled GEMM (router) + matvec, mxfp4 gu-interleave, fused-QKV norm+rope+KV-scatter, launchers
// Textually-included segment of the single pack translation unit.
// Not standalone-compilable: include order is defined by ../pack.cu.
// ------------------------------------------------------------------ launchers

static int pd_launch_status() { return (int)cudaGetLastError(); }

// pd_pdl_off / pd_pdl_dev_ok / pd_pdl_go live in abi.cuh so
// launchers in segments included before this one (elementwise, attn/decode,
// moe/*) can arm into the cascade. Same semantics, same PADDOCK_NO_PDL kill.

// Host-side tensor-map encode for the v7 TMA decode-attention staging comes
// from ../tma_desc.cuh (pd_tmap_encode) - this segment used to carry its own
// byte-identical resolver because that one was PD_BS_HOST-gated and included
// later. Both conditions are gone. Semantics unchanged: resolved
// once via the runtime, no libcuda link, nullptr on old drivers routes the
// launcher to the v5 fallback.

// KV pool as a dense 2D byte matrix: rows = global position index (block*16
// + in-block offset - the paged pool is row-dense), inner = kv_dim*2 bytes.
// [16 x 128B] boxes, SW128 - one box per 128B column segment of one block,
// landing as the canonical GEMM-stage swizzle tile. The row extent is a
// loose upper bound: block ids from the table always land inside the real
// allocation, so the OOB path never triggers.
static bool pd_attn_tmap_kv(CUtensorMap* map, const void* base, uint32_t kv_dim) {
    pd_tmap_encode_fn enc = pd_tmap_encode();
    if (!enc || ((uintptr_t)base & 127u)) return false;
    const cuuint64_t gdim[2] = {(cuuint64_t)kv_dim * 2u, 1ull << 30};
    const cuuint64_t gstride[1] = {(cuuint64_t)kv_dim * 2u};
    const cuuint32_t box[2] = {128u, 16u};
    const cuuint32_t estride[2] = {1u, 1u};
    return enc(map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2u, (void*)base, gdim,
               gstride, box, estride, CU_TENSOR_MAP_INTERLEAVE_NONE,
               CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

// fp8 pools (KV8): 1 B/elem row width and SWIZZLE_NONE - the v8f8 kernel
// stages raw linear bytes and applies the f16 128B-swizzle itself during the
// fp8->f16 expansion (at half the byte width TMA's own swizzle would no
// longer land where the consumers' sw() addressing expects).
// vdim pool registration: the engine registers the twin pool at
// alloc; the v9q launcher builds the VD map from it. pool_v stays the legacy
// pool for every non-VD reader (HD512 global layers, v8 fallbacks, prefill).
PD_EXPORT
int pd_vdim_register(void* base) {
    pd_vdim_base = base;
    return 0;
}

// plain (16, HD<=256)-box map over vdim[block][kv_dim][16]: inner dim = 16
// key bytes, rows = every (block, dim) pair. No swizzle - the panel lands
// [dim][16] in smem, which is exactly the VD PV word-load layout.
static bool pd_attn_tmap_vdim(CUtensorMap* map, const void* base,
                              uint64_t rows_total, uint32_t hd) {
    pd_tmap_encode_fn enc = pd_tmap_encode();
    if (!enc || ((uintptr_t)base & 15u)) return false;
    const cuuint64_t gdim[2] = {16u, rows_total};
    const cuuint64_t gstride[1] = {16u};
    const cuuint32_t box[2] = {16u, hd};
    const cuuint32_t estride[2] = {1u, 1u};
    return enc(map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2u, (void*)base, gdim,
               gstride, box, estride, CU_TENSOR_MAP_INTERLEAVE_NONE,
               CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

static bool pd_attn_tmap_kv_f8s(CUtensorMap* map, const void* base, uint32_t kv_dim) {
    // v8q: fp8 pools with the 128B swizzle on - the score side ldmatrix
    // addresses raw fp8 through sw8() (the same XOR pattern the f16 sw()
    // encodes), conflict-free like the f16 path
    pd_tmap_encode_fn enc = pd_tmap_encode();
    if (!enc || ((uintptr_t)base & 127u)) return false;
    const cuuint64_t gdim[2] = {(cuuint64_t)kv_dim, 1ull << 30};
    const cuuint64_t gstride[1] = {(cuuint64_t)kv_dim};
    const cuuint32_t box[2] = {128u, 16u};
    const cuuint32_t estride[2] = {1u, 1u};
    return enc(map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2u, (void*)base, gdim,
               gstride, box, estride, CU_TENSOR_MAP_INTERLEAVE_NONE,
               CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

static bool pd_attn_tmap_kv_f8(CUtensorMap* map, const void* base, uint32_t kv_dim) {
    pd_tmap_encode_fn enc = pd_tmap_encode();
    if (!enc || ((uintptr_t)base & 127u)) return false;
    const cuuint64_t gdim[2] = {(cuuint64_t)kv_dim, 1ull << 30};
    const cuuint64_t gstride[1] = {(cuuint64_t)kv_dim};
    const cuuint32_t box[2] = {128u, 16u};
    const cuuint32_t estride[2] = {1u, 1u};
    return enc(map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2u, (void*)base, gdim,
               gstride, box, estride, CU_TENSOR_MAP_INTERLEAVE_NONE,
               CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

//  (pf5 bulk KV staging): whole e4m3 V head-rows land LINEAR - one
// [256B x 16] box per pool page, SWIZZLE_NONE, so 16 rows arrive contiguous
// exactly as the per-thread path left the V strip (HD=256 e4m3 = 256B/row).
// The K side reuses pd_attn_tmap_kv_f8s (SW128 - TMA's landing pattern is
// bit-identical to pd_pf5_f8_swz_k's manual swizzle, probed /69 class).
static bool pd_attn_tmap_v256(CUtensorMap* map, const void* base, uint32_t kv_dim) {
    pd_tmap_encode_fn enc = pd_tmap_encode();
    if (!enc || ((uintptr_t)base & 127u)) return false;
    const cuuint64_t gdim[2] = {(cuuint64_t)kv_dim, 1ull << 30};
    const cuuint64_t gstride[1] = {(cuuint64_t)kv_dim};
    const cuuint32_t box[2] = {256u, 16u};
    const cuuint32_t estride[2] = {1u, 1u};
    return enc(map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2u, (void*)base, gdim,
               gstride, box, estride, CU_TENSOR_MAP_INTERLEAVE_NONE,
               CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

// Request the max shared-memory carveout so shared-heavy blocks (the 4x4 sorted
// GEMM, ~37 KB) pack more blocks/SM than the default ~48 KB carveout allows. The
// "max" adapts per arch (sm_80 164 KB, sm_86/89 100 KB, sm_90 227 KB, ...), so this
// is portable - no hardcoded byte count. Idempotent; call once per kernel.
template <typename F>
static void pd_prefer_max_shared(F kernel) {
    cudaFuncSetAttribute((const void*)kernel,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
}

PD_EXPORT
int pd_mxfp4_dequant_f32(const void* in, void* out, uint64_t n_blocks, void* stream) {
    uint64_t total = n_blocks * 32ull;
    if (total == 0) return 0;
    uint32_t threads = 256;
    uint32_t blocks = (uint32_t)((total + threads - 1) / threads);
    pd_mxfp4_dequant_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)in, (float*)out, total);
    return pd_launch_status();
}

PD_EXPORT
int pd_q8_0_dequant_f32(const void* in, void* out, uint64_t n_blocks, void* stream) {
    uint64_t total = n_blocks * 32ull;
    if (total == 0) return 0;
    uint32_t threads = 256;
    uint32_t blocks = (uint32_t)((total + threads - 1) / threads);
    pd_q8_0_dequant_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)in, (float*)out, total);
    return pd_launch_status();
}

PD_EXPORT
int pd_rmsnorm_f32(const void* x, const void* w, void* out, uint32_t n, float eps, void* stream) {
    pd_rmsnorm_kernel<<<1, 256, 0, (cudaStream_t)stream>>>(
        (const float*)x, (const float*)w, (float*)out, n, eps);
    return pd_launch_status();
}

PD_EXPORT
int pd_rope_yarn_f32(void* x, uint32_t n_heads, uint32_t head_dim, uint32_t pos,
                     float theta_scale, float freq_scale, float corr_low, float corr_high,
                     float ext_factor, float mscale, void* stream) {
    uint32_t threads = 64;
    uint32_t blocks = (n_heads + threads - 1) / threads;
    pd_rope_yarn_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        (float*)x, n_heads, head_dim, pos, theta_scale, freq_scale,
        corr_low, corr_high, ext_factor, mscale);
    return pd_launch_status();
}

PD_EXPORT
int pd_softmax_sink_f32(void* scores, uint32_t n, float sink, void* stream) {
    pd_softmax_sink_kernel<<<1, 256, 0, (cudaStream_t)stream>>>((float*)scores, n, sink);
    return pd_launch_status();
}

PD_EXPORT
int pd_swiglu_oai_f32(void* gate, const void* up, uint32_t n, float alpha, float limit, void* stream) {
    uint32_t threads = 256;
    uint32_t blocks = (n + threads - 1) / threads;
    pd_swiglu_oai_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        (float*)gate, (const float*)up, n, alpha, limit);
    return pd_launch_status();
}

PD_EXPORT
int pd_add_inplace_f32(void* x, const void* y, uint32_t n, void* stream) {
    uint32_t threads = 256;
    uint32_t blocks = (n + threads - 1) / threads;
    pd_pdl_go(pd_add_inplace_kernel, blocks, threads, 0u, (cudaStream_t)stream,
        (float*)x, (const float*)y, n);
    return pd_launch_status();
}

PD_EXPORT
int pd_scale_add_f32(void* x, const void* y, float w, uint32_t n, void* stream) {
    uint32_t threads = 256;
    uint32_t blocks = (n + threads - 1) / threads;
    pd_pdl_go(pd_scale_add_kernel, blocks, threads, 0u, (cudaStream_t)stream,
        (float*)x, (const float*)y, w, n);
    return pd_launch_status();
}

PD_EXPORT
int pd_scale_f32(void* x, float s, uint32_t n, void* stream) {
    uint32_t threads = 256;
    uint32_t blocks = (n + threads - 1) / threads;
    pd_scale_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>((float*)x, s, n);
    return pd_launch_status();
}

PD_EXPORT
int pd_moe_topk(const void* logits, uint32_t n_expert, uint32_t k,
                void* out_idx, void* out_w, void* stream) {
    if (k > 16u || n_expert > PD_MOE_TOPK_MAX_EXPERTS) return cudaErrorInvalidValue;
    if (n_expert > 256u) {
        pd_moe_topk_kernel_t<16u><<<1, 32, 0, (cudaStream_t)stream>>>(
            (const float*)logits, n_expert, k, (uint32_t*)out_idx, (float*)out_w);
    } else {
        pd_moe_topk_kernel_t<8u><<<1, 32, 0, (cudaStream_t)stream>>>(
            (const float*)logits, n_expert, k, (uint32_t*)out_idx, (float*)out_w);
    }
    return pd_launch_status();
}

PD_EXPORT
int pd_mxfp4_gemv_indexed(const void* W, const void* bias, const void* idx, uint32_t slot,
                          const void* x, void* y, uint32_t in_dim, uint32_t out_dim, void* stream) {
    uint32_t threads = 256;   // LUT + wsum are static shared, so no dynamic shmem
    pd_mxfp4_gemv_indexed_kernel<<<out_dim, threads, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)W, (const float*)bias, (const uint32_t*)idx, slot,
        (const float*)x, (float*)y, in_dim, out_dim);
    return pd_launch_status();
}

PD_EXPORT
int pd_mxfp4_moe_gate_up(const void* gate_data, const void* gate_scale, const void* gate_bias,
                         const void* up_data, const void* up_scale, const void* up_bias,
                         const void* idx, const void* x, void* out,
                         uint32_t in_dim, uint32_t ff, uint32_t n_active,
                         float alpha, float limit, void* stream) {
    if (ff == 0 || n_active == 0) return 0;
    dim3 grid(ff, n_active);
    size_t shmem = (80u + 2u * (in_dim >> 5)) * sizeof(float);   // lut+wg+wu + 2 scale rows
    pd_mxfp4_moe_gate_up_kernel<<<grid, 256, shmem, (cudaStream_t)stream>>>(
        (const uint8_t*)gate_data, (const uint8_t*)gate_scale, (const float*)gate_bias,
        (const uint8_t*)up_data, (const uint8_t*)up_scale, (const float*)up_bias,
        (const uint32_t*)idx, (const float*)x, (float*)out, in_dim, ff, alpha, limit);
    return pd_launch_status();
}

PD_EXPORT
int pd_mxfp4_moe_down(const void* down_data, const void* down_scale, const void* down_bias,
                      const void* idx, const void* topk_w, const void* fused, void* residual,
                      uint32_t ff, uint32_t embd, uint32_t n_active, void* stream) {
    if (embd == 0 || n_active == 0) return 0;
    size_t shmem = (48u + (size_t)n_active * (ff >> 5)) * sizeof(float);   // lut+wsum + scales
    pd_mxfp4_moe_down_kernel<<<embd, 256, shmem, (cudaStream_t)stream>>>(
        (const uint8_t*)down_data, (const uint8_t*)down_scale, (const float*)down_bias,
        (const uint32_t*)idx, (const float*)topk_w, (const float*)fused, (float*)residual,
        ff, embd, n_active);
    return pd_launch_status();
}

PD_EXPORT
int pd_q8_0_gemv(const void* W, const void* bias, const void* x, void* y,
                 uint32_t in_dim, uint32_t out_dim, void* stream) {
    uint32_t threads = 256;
    size_t shmem = (size_t)(in_dim >> 5) * sizeof(float);   // one f32 scale per block
    pd_q8_0_gemv_kernel<<<out_dim, threads, shmem, (cudaStream_t)stream>>>(
        (const uint8_t*)W, (const float*)bias, (const float*)x, (float*)y, in_dim, out_dim);
    return pd_launch_status();
}

PD_EXPORT
int pd_q8_0_gemm(const void* W, const void* bias, const void* x, void* y,
                 uint32_t in_dim, uint32_t out_dim, uint32_t batch, void* stream) {
    if (out_dim == 0 || batch == 0) return 0;
    size_t shmem = (size_t)(in_dim >> 5) * sizeof(float);
    pd_q8_0_gemm_kernel<<<out_dim, 256, shmem, (cudaStream_t)stream>>>(
        (const uint8_t*)W, (const float*)bias, (const float*)x, (float*)y, in_dim, out_dim, batch);
    return pd_launch_status();
}

PD_EXPORT
int pd_quantize_q8(const void* x, void* q, void* scale, uint32_t n, void* stream) {
    uint32_t n_blocks = n >> 5;
    if (n_blocks == 0) return 0;
    // 8 warps/CTA: grid was n_blocks 1-warp CTAs - CTA-dispatch-
    // bound at verify-plane sizes (see the kernel comment). Bit-exact.
    pd_pdl_go(pd_quantize_q8_kernel, (n_blocks + 7u) / 8u, 256, 0u, (cudaStream_t)stream,
        (const float*)x, (signed char*)q, (float*)scale, n_blocks);
    return pd_launch_status();
}

PD_EXPORT
int pd_quantize_q8_relu2(const void* x, void* q, void* scale, uint32_t n, void* stream) {
    uint32_t n_blocks = n >> 5;
    if (n_blocks == 0) return 0;
    // 8 warps/CTA, same geometry change as pd_quantize_q8 (bit-exact)
    pd_pdl_go(pd_quantize_q8_relu2_kernel, (n_blocks + 7u) / 8u, 256, 0u, (cudaStream_t)stream,
        (const float*)x, (signed char*)q, (float*)scale, n_blocks);
    return pd_launch_status();
}

PD_EXPORT
int pd_quantize_q8_sums(const void* x, void* q, void* scale, void* sums,
                        uint32_t n, void* stream) {
    uint32_t n_blocks = n >> 5;
    if (n_blocks == 0) return 0;
    pd_pdl_go(pd_quantize_q8_sums_kernel, n_blocks, 32, 0u, (cudaStream_t)stream,
        (const float*)x, (signed char*)q, (float*)scale, (float*)sums, n_blocks);
    return pd_launch_status();
}

PD_EXPORT
int pd_q8_0_gemv_dp4a(const void* W, const void* bias, const void* xq, const void* xs,
                      void* y, uint32_t in_dim, uint32_t out_dim, void* stream) {
    if (out_dim == 0) return 0;
    pd_q8_0_gemv_dp4a_kernel<<<(out_dim + 7) / 8, 256, 0, (cudaStream_t)stream>>>(
        (const unsigned char*)W, (const float*)bias, (const signed char*)xq,
        (const float*)xs, (float*)y, in_dim, out_dim);
    return pd_launch_status();
}

PD_EXPORT
int pd_mxfp4_gemv_indexed_dp4a(const void* W, const void* bias, const void* idx, uint32_t slot,
                               const void* xq, const void* xs, void* y,
                               uint32_t in_dim, uint32_t out_dim, void* stream) {
    if (out_dim == 0) return 0;
    pd_mxfp4_gemv_indexed_dp4a_kernel<<<out_dim, 256, 0, (cudaStream_t)stream>>>(
        (const unsigned char*)W, (const float*)bias, (const unsigned int*)idx, slot,
        (const signed char*)xq, (const float*)xs, (float*)y, in_dim, out_dim);
    return pd_launch_status();
}

PD_EXPORT
int pd_mxfp4_moe_gate_up_dp4a(const void* gate_data, const void* gate_scale, const void* gate_bias,
                              const void* up_data, const void* up_scale, const void* up_bias,
                              const void* idx, const void* xq, const void* xs,
                              void* out, uint32_t in_dim, uint32_t ff, uint32_t n_active,
                              float alpha, float limit, void* stream) {
    if (ff == 0 || n_active == 0) return 0;
    dim3 grid((ff + 7) / 8, n_active); // 8 warps/block, warp per output row
    pd_mxfp4_moe_gate_up_dp4a_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(
        (const unsigned char*)gate_data, (const unsigned char*)gate_scale, (const float*)gate_bias,
        (const unsigned char*)up_data, (const unsigned char*)up_scale, (const float*)up_bias,
        (const unsigned int*)idx, (const signed char*)xq,
        (const float*)xs, (float*)out, in_dim, ff, alpha, limit);
    return pd_launch_status();
}

PD_EXPORT
int pd_mxfp4_moe_down_dp4a(const void* down_data, const void* down_scale, const void* down_bias,
                           const void* idx, const void* topk_w, const void* fused_q,
                           const void* fused_s, void* residual, uint32_t ff, uint32_t embd,
                           uint32_t n_active, void* stream) {
    if (embd == 0 || n_active == 0) return 0;
    pd_mxfp4_moe_down_dp4a_kernel<<<(embd + 7) / 8, 256, 0, (cudaStream_t)stream>>>(
        (const unsigned char*)down_data, (const unsigned char*)down_scale, (const float*)down_bias,
        (const unsigned int*)idx, (const float*)topk_w, (const signed char*)fused_q,
        (const float*)fused_s, (float*)residual, ff, embd, n_active);
    return pd_launch_status();
}

// Batched (grid.z = token) launch of the fused dp4a MoE pair, for tiny serving
// batches: each token re-reads its own experts' weights (b * n_active GEMV
// strips), which beats the sorted mmq tiles below ~b=5 - there the mmq grid is
// a handful of blocks whose deep-staged K-walks run latency-bound (~150 us per
// launch at B=2 vs a ~40 us weight-traffic floor here). Same numeric class and
// per-row math as the single-token launchers above.
PD_EXPORT
int pd_mxfp4_moe_gate_up_dp4a_b(const void* gate_data, const void* gate_scale, const void* gate_bias,
                                const void* up_data, const void* up_scale, const void* up_bias,
                                const void* idx, const void* xq, const void* xs,
                                void* out, uint32_t in_dim, uint32_t ff, uint32_t n_active,
                                uint32_t batch, float alpha, float limit, void* stream) {
    if (ff == 0 || n_active == 0 || batch == 0) return 0;
    dim3 grid((ff + 7) / 8, n_active, batch);
    pd_mxfp4_moe_gate_up_dp4a_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(
        (const unsigned char*)gate_data, (const unsigned char*)gate_scale, (const float*)gate_bias,
        (const unsigned char*)up_data, (const unsigned char*)up_scale, (const float*)up_bias,
        (const unsigned int*)idx, (const signed char*)xq,
        (const float*)xs, (float*)out, in_dim, ff, alpha, limit);
    return pd_launch_status();
}

PD_EXPORT
int pd_mxfp4_moe_down_dp4a_b(const void* down_data, const void* down_scale, const void* down_bias,
                             const void* idx, const void* topk_w, const void* fused_q,
                             const void* fused_s, void* residual, uint32_t ff, uint32_t embd,
                             uint32_t n_active, uint32_t batch, void* stream) {
    if (embd == 0 || n_active == 0 || batch == 0) return 0;
    dim3 grid((embd + 7) / 8, batch);
    pd_mxfp4_moe_down_dp4a_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(
        (const unsigned char*)down_data, (const unsigned char*)down_scale, (const float*)down_bias,
        (const unsigned int*)idx, (const float*)topk_w, (const signed char*)fused_q,
        (const float*)fused_s, (float*)residual, ff, embd, n_active);
    return pd_launch_status();
}

// Output-tiled router matvec (b >= 16): block = 8 warps, warp = one output
// Tiled f32 GEMM for the LARGE-batch router: C[batch, out] = X[batch, in] ·
// W[out, in]^T. The tile<4> matvec below re-reads every x row per 8-output
// group AND every w row per 4-token group - at a 2048-row prefill chunk
// that's ~0.5 GB of L2 per call (pd_matvec_f32_tile avg
// ~210 us = 7-8% of all GPU time at pf8/c32). Classic smem tiling (BM=64
// tokens x BN=64 outs x BK=32) loads each operand tile once per block:
// traffic drops to (out/BN)·X + (batch/BM)·W ≈ 130 MB, and the FMAs run on
// register micro-tiles. Accumulation is plain ascending-k per output -
// different order from the matvec's lane-strided+shuffle tree, so this is
// not bit-exact with the old path (router feeds top-k; a last-ulp change
// can flip a near-tie) - hence env-gated PADDOCK_ROUTER_GEMM until the
// serving parity gate passes. 16x16 threads, 4x4 micro-tile each.
#define PD_RGEMM_BK 32u
// Templated on the output tile: MW x NW = the thread micro-tile grid (16x16
// threads fixed), micro-tile (MT x NT) => BM = 16*MT, BN = 16*NT. Small tiles
// buy GRID (the router shape only has out_dim/BN col-blocks, and at 128
// blocks the 188-SM die idles); big tiles buy L2 reuse. Launcher picks by
// grid fill.
template <uint32_t MT, uint32_t NT>
__global__ void __launch_bounds__(256)
pd_gemm_f32_nt_kernel(const float* __restrict__ w, const float* __restrict__ x,
                      float* __restrict__ out, uint32_t in_dim,
                      uint32_t out_dim, uint32_t batch) {
    constexpr uint32_t BM = 16u * MT, BN = 16u * NT, BK = PD_RGEMM_BK;
    __shared__ float xs[BK][BM + 4];  // x tile, k-major (transposed at load)
    __shared__ float ws[BK][BN + 4];  // w tile, k-major
    const uint32_t m0 = blockIdx.x * BM, n0 = blockIdx.y * BN;
    const uint32_t tid = threadIdx.x;
    // micro-tile owner: 16x16 grid of threads, thread (tm, tn) owns rows
    // m0+tm*MT..+MT and cols n0+tn*NT..+NT
    const uint32_t tm = tid >> 4, tn = tid & 15u;
    float acc[MT][NT] = {};
    for (uint32_t k0 = 0; k0 < in_dim; k0 += BK) {
        // tile loads as float4 (BK/4 vecs per row), rows past the batch edge
        // stage zeros (their outputs are never stored)
        #pragma unroll
        for (uint32_t l = 0; l < BM * (BK / 4u) / 256u; ++l) {
            const uint32_t idx = tid + l * 256u;
            const uint32_t r = idx / (BK / 4u), k4 = (idx % (BK / 4u)) * 4u;
            float4 xv = make_float4(0.f, 0.f, 0.f, 0.f);
            if (m0 + r < batch)
                xv = *reinterpret_cast<const float4*>(
                    x + (size_t)(m0 + r) * in_dim + k0 + k4);
            xs[k4 + 0][r] = xv.x; xs[k4 + 1][r] = xv.y;
            xs[k4 + 2][r] = xv.z; xs[k4 + 3][r] = xv.w;
        }
        #pragma unroll
        for (uint32_t l = 0; l < BN * (BK / 4u) / 256u; ++l) {
            const uint32_t idx = tid + l * 256u;
            const uint32_t r = idx / (BK / 4u), k4 = (idx % (BK / 4u)) * 4u;
            const float4 wv = *reinterpret_cast<const float4*>(
                w + (size_t)(n0 + r) * in_dim + k0 + k4);
            ws[k4 + 0][r] = wv.x; ws[k4 + 1][r] = wv.y;
            ws[k4 + 2][r] = wv.z; ws[k4 + 3][r] = wv.w;
        }
        __syncthreads();
        #pragma unroll
        for (uint32_t kk = 0; kk < BK; ++kk) {
            float xr[MT], wr[NT];
            #pragma unroll
            for (uint32_t i = 0; i < MT; ++i) xr[i] = xs[kk][tm * MT + i];
            #pragma unroll
            for (uint32_t j = 0; j < NT; ++j) wr[j] = ws[kk][tn * NT + j];
            #pragma unroll
            for (uint32_t i = 0; i < MT; ++i)
                #pragma unroll
                for (uint32_t j = 0; j < NT; ++j)
                    acc[i][j] = fmaf(xr[i], wr[j], acc[i][j]);
        }
        __syncthreads();
    }
    #pragma unroll
    for (uint32_t i = 0; i < MT; ++i) {
        const uint32_t m = m0 + tm * MT + i;
        if (m < batch) {
            #pragma unroll
            for (uint32_t j = 0; j < NT; ++j)
                out[(size_t)m * out_dim + n0 + tn * NT + j] = acc[i][j];
        }
    }
}

// K-SPLIT twin of the tiled GEMM above: the decay/ba plane
// (out <= 128) leaves the router GEMM a 24-96-block grid on a 188-SM die at
// ~9 TF. Splitting K across grid-z refills the wave; each split keeps the
// exact per-element f32 FMA order of its window (same body), partials
// combine in ascending split order - deterministic, f32-class regroup only
// (a 3xTF32 mma variant was built first and read +0.58% PPL at pf1024;
// the chaos-band probe - a known-benign perturbation moves pf1024 by
// +0.51% - showed that delta is in-band resampling, not bias, but the
// exact-f32 window costs nothing extra and keeps the strictly tighter
// class on the decay path, so it is the one that shipped).
template <uint32_t MT, uint32_t NT>
__global__ void __launch_bounds__(256)
pd_gemm_f32_nt_ks_kernel(const float* __restrict__ w, const float* __restrict__ x,
                         float* __restrict__ part, uint32_t in_dim,
                         uint32_t out_dim, uint32_t batch, uint32_t kwin) {
    constexpr uint32_t BM = 16u * MT, BN = 16u * NT, BK = PD_RGEMM_BK;
    __shared__ float xs[BK][BM + 4];
    __shared__ float ws[BK][BN + 4];
    const uint32_t m0 = blockIdx.x * BM, n0 = blockIdx.y * BN;
    const uint32_t k_lo = blockIdx.z * kwin;
    const uint32_t k_hi = min(k_lo + kwin, in_dim);
    const uint32_t tid = threadIdx.x;
    const uint32_t tm = tid >> 4, tn = tid & 15u;
    float acc[MT][NT] = {};
    for (uint32_t k0 = k_lo; k0 < k_hi; k0 += BK) {
        #pragma unroll
        for (uint32_t l = 0; l < BM * (BK / 4u) / 256u; ++l) {
            const uint32_t idx = tid + l * 256u;
            const uint32_t r = idx / (BK / 4u), k4 = (idx % (BK / 4u)) * 4u;
            float4 xv = make_float4(0.f, 0.f, 0.f, 0.f);
            if (m0 + r < batch)
                xv = *reinterpret_cast<const float4*>(
                    x + (size_t)(m0 + r) * in_dim + k0 + k4);
            xs[k4 + 0][r] = xv.x; xs[k4 + 1][r] = xv.y;
            xs[k4 + 2][r] = xv.z; xs[k4 + 3][r] = xv.w;
        }
        #pragma unroll
        for (uint32_t l = 0; l < BN * (BK / 4u) / 256u; ++l) {
            const uint32_t idx = tid + l * 256u;
            const uint32_t r = idx / (BK / 4u), k4 = (idx % (BK / 4u)) * 4u;
            const float4 wv = *reinterpret_cast<const float4*>(
                w + (size_t)(n0 + r) * in_dim + k0 + k4);
            ws[k4 + 0][r] = wv.x; ws[k4 + 1][r] = wv.y;
            ws[k4 + 2][r] = wv.z; ws[k4 + 3][r] = wv.w;
        }
        __syncthreads();
        #pragma unroll
        for (uint32_t kk = 0; kk < BK; ++kk) {
            float xr[MT], wr[NT];
            #pragma unroll
            for (uint32_t i = 0; i < MT; ++i) xr[i] = xs[kk][tm * MT + i];
            #pragma unroll
            for (uint32_t j = 0; j < NT; ++j) wr[j] = ws[kk][tn * NT + j];
            #pragma unroll
            for (uint32_t i = 0; i < MT; ++i)
                #pragma unroll
                for (uint32_t j = 0; j < NT; ++j)
                    acc[i][j] = fmaf(xr[i], wr[j], acc[i][j]);
        }
        __syncthreads();
    }
    #pragma unroll
    for (uint32_t i = 0; i < MT; ++i) {
        const uint32_t m = m0 + tm * MT + i;
        if (m < batch) {
            #pragma unroll
            for (uint32_t j = 0; j < NT; ++j)
                part[((size_t)blockIdx.z * batch + m) * out_dim + n0 + tn * NT + j] =
                    acc[i][j];
        }
    }
}

__global__ void pd_f32nt_comb_kernel(const float* __restrict__ part,
                                     float* __restrict__ y, uint32_t n,
                                     uint32_t s_count) {
    const uint32_t i = (blockIdx.x * 256u + threadIdx.x) * 4u;
    if (i >= n) return;
    float4 acc = *reinterpret_cast<const float4*>(part + i);
    for (uint32_t s = 1; s < s_count; ++s) {
        const float4 p = *reinterpret_cast<const float4*>(part + (size_t)s * n + i);
        acc.x += p.x; acc.y += p.y; acc.z += p.z; acc.w += p.w;
    }
    *reinterpret_cast<float4*>(y + i) = acc;
}

static int pd_f32nt_ks_go(const float* w, const float* x, float* y,
                          uint32_t in_dim, uint32_t out_dim, uint32_t batch,
                          uint32_t S, cudaStream_t s) {
    // 64x32 tiles (4x2 micro) double the per-thread arithmetic intensity
    // over 32x32 - the ks grid stays wave-filled through the K splits
    const uint32_t mtiles = (batch + 63u) / 64u, ntiles = out_dim / 32u;
    const uint32_t kw_steps = (in_dim / PD_RGEMM_BK + S - 1u) / S;
    const uint32_t kwin = kw_steps * PD_RGEMM_BK;
    // grow-once partial planes (2026-09-08): the per-call cudaMallocAsync /
    // FreeAsync pair was this arm's only cost over the tf32 one (the c16
    // TTFT p90 straggler class on the big die) and kept it opt-in.
    // Launchers run under the engine's per-stream serialization, so the
    // statics are safe; the buffer never shrinks.
    static float* part = nullptr;
    static size_t part_cap = 0;
    const size_t need = (size_t)S * batch * out_dim * 4u;
    if (need > part_cap) {
        if (part) cudaFree(part);
        part = nullptr; part_cap = 0;
        if (cudaMalloc(&part, need) != cudaSuccess) { part = nullptr; return cudaErrorMemoryAllocation; }
        part_cap = need;
    }
    dim3 g(mtiles, ntiles, S);
    pd_gemm_f32_nt_ks_kernel<4u, 2u><<<g, 256, 0, s>>>(w, x, part, in_dim,
                                                       out_dim, batch, kwin);
    const uint32_t n = batch * out_dim;
    pd_f32nt_comb_kernel<<<(n / 4u + 255u) / 256u, 256u, 0, s>>>(part, y, n, S);
    return pd_launch_status();
}

// tf32 tensor-core twin of the nt/nt_ks pair (ba rung): at the
// c16 wave the ba plane (out=96, batch~4k, K=5120) ran the SIMT kernel at
// its own LDS-bound ceiling (204.8 us/layer, ~20 TF - the 4x2 micro-tile
// pays 6 shared loads per 8 FMAs) where a bf16 nvjet runs the same plane in
// 35.1 us. 3xTF32 (big+small correction, ascending-k in k8
// groups) is STRICTLY finer than that bf16 class, and the earlier
// 3xTF32 arm's +0.58% PPL was already shown to be chaos-band RESAMPLING
// (in-band vs the known-benign perturbation), not bias - f32 shipped then
// only because the grid, not the rate, was the problem. No K-split here:
// the route gates on (batch/64)x(out/32) >= 148 CTAs, so the wave fills
// without partials (the K-split existed for the compute-starved SIMT
// form). PREC=1 (single tf32) kept as a probe arm.
static __device__ __forceinline__ uint32_t pd_bnt_tf32(float v) {
    uint32_t r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(v));
    return r;
}
template <uint32_t PREC>
__global__ void __launch_bounds__(128)
pd_gemm_tf32_nt_kernel(const float* __restrict__ w, const float* __restrict__ x,
                       float* __restrict__ out, uint32_t in_dim,
                       uint32_t out_dim, uint32_t batch) {
    // ROW-major tiles + cp.async 2-slot ring: the SIMT family's k-major
    // transpose scatter costs 12 scalar STS/thread/step (~384 STS per
    // warp-step - the family's real wall; the probe read both the mma
    // rewrite and a register double-buffer as ~x1.2). Row-major stages as
    // three 16B cp.async per thread and the tf32 fragments read it
    // conflict-free (stride 36 floats: bank = 4*row + col mod 32).
    constexpr uint32_t BM = 32u, BN = 32u, BK = PD_RGEMM_BK, SK = BK + 4u;
    __shared__ float xs[2][BM][SK];
    __shared__ float ws[2][BN][SK];
    const uint32_t m0 = blockIdx.x * BM, n0 = blockIdx.y * BN;
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const uint32_t gr = lane >> 2, t4 = lane & 3u;
    const uint32_t mt = warp & 1u, nh = warp >> 1;  // m16 tile / n16 half
    const uint32_t xr = tid / (BK / 4u), xk = (tid % (BK / 4u)) * 4u;
    // Two-level accumulation. The DPU behind mma.f32.tf32
    // accumulates the C chain with TRUNCATION (RZ) and a few guard bits
    // (Fasi/Higham/Mikaitis/Pranesh 2021 - study ref), so a running K-sum
    // passed through C compounds a ~2^-27-per-pass BIAS: measured K*2^-27
    // exactly on the kquant parity rung (K=4096 -> 3.83e-5, K=12288 ->
    // 1.15e-4 - LINEAR in K, the truncation-bias signature), drowning the
    // 3xTF32 split's ~1e-6 operand class. acc[] therefore chains only the
    // 12 mma of one BK tile; fac[] carries the cross-tile sum in CUDA-core
    // FADD (round-nearest), turning the linear bias into a per-tile random
    // walk. Same two-level technique as CUTLASS/FA3 fp8 accumulation
    // promotion (inspiration only - original implementation). The extra 8
    // FADDs per tile ride free: the kernel is staging-bound (rung
    // 9 measured 3x the mma work at 1.00x wall).
    float acc[2][4] = {};
    float fac[2][4] = {};
    auto cp16 = [](float* dst, const float* src, uint32_t bytes) {
        const unsigned sm = (unsigned)__cvta_generic_to_shared(dst);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(sm),
                     "l"(src), "r"(bytes));
    };
    auto stage = [&](uint32_t slot, uint32_t k0) {
        cp16(&xs[slot][xr][xk], x + (size_t)(m0 + xr) * in_dim + k0 + xk,
             m0 + xr < batch ? 16u : 0u);
        cp16(&xs[slot][xr + 16u][xk],
             x + (size_t)(m0 + xr + 16u) * in_dim + k0 + xk,
             m0 + xr + 16u < batch ? 16u : 0u);
        cp16(&ws[slot][xr][xk], w + (size_t)(n0 + xr) * in_dim + k0 + xk, 16u);
        cp16(&ws[slot][xr + 16u][xk],
             w + (size_t)(n0 + xr + 16u) * in_dim + k0 + xk, 16u);
        asm volatile("cp.async.commit_group;" ::: "memory");
    };
    stage(0u, 0u);
    for (uint32_t k0 = 0; k0 < in_dim; k0 += BK) {
        const uint32_t slot = (k0 / BK) & 1u;
        const bool more = k0 + BK < in_dim;
        if (more) stage(slot ^ 1u, k0 + BK);
        if (more) asm volatile("cp.async.wait_group 1;" ::: "memory");
        else asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        #pragma unroll
        for (uint32_t k8 = 0; k8 < BK; k8 += 8u) {
            // A m16k8, converted once for both n8 tiles
            uint32_t ab[4], as[4];
            {
                const float a0 = xs[slot][mt * 16u + gr][k8 + t4];
                const float a1 = xs[slot][mt * 16u + gr + 8u][k8 + t4];
                const float a2 = xs[slot][mt * 16u + gr][k8 + t4 + 4u];
                const float a3 = xs[slot][mt * 16u + gr + 8u][k8 + t4 + 4u];
                ab[0] = pd_bnt_tf32(a0); ab[1] = pd_bnt_tf32(a1);
                ab[2] = pd_bnt_tf32(a2); ab[3] = pd_bnt_tf32(a3);
                if (PREC == 3u) {
                    as[0] = pd_bnt_tf32(a0 - __uint_as_float(ab[0]));
                    as[1] = pd_bnt_tf32(a1 - __uint_as_float(ab[1]));
                    as[2] = pd_bnt_tf32(a2 - __uint_as_float(ab[2]));
                    as[3] = pd_bnt_tf32(a3 - __uint_as_float(ab[3]));
                }
            }
            #pragma unroll
            for (uint32_t nt = 0; nt < 2u; ++nt) {
                const float b0 = ws[slot][nh * 16u + nt * 8u + gr][k8 + t4];
                const float b1 = ws[slot][nh * 16u + nt * 8u + gr][k8 + t4 + 4u];
                uint32_t bb[2] = {pd_bnt_tf32(b0), pd_bnt_tf32(b1)};
                asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+f"(acc[nt][0]), "+f"(acc[nt][1]), "+f"(acc[nt][2]),
                      "+f"(acc[nt][3])
                    : "r"(ab[0]), "r"(ab[1]), "r"(ab[2]), "r"(ab[3]),
                      "r"(bb[0]), "r"(bb[1]));
                if (PREC == 3u) {
                    uint32_t bs[2] = {pd_bnt_tf32(b0 - __uint_as_float(bb[0])),
                                      pd_bnt_tf32(b1 - __uint_as_float(bb[1]))};
                    asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+f"(acc[nt][0]), "+f"(acc[nt][1]), "+f"(acc[nt][2]),
                          "+f"(acc[nt][3])
                        : "r"(ab[0]), "r"(ab[1]), "r"(ab[2]), "r"(ab[3]),
                          "r"(bs[0]), "r"(bs[1]));
                    asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+f"(acc[nt][0]), "+f"(acc[nt][1]), "+f"(acc[nt][2]),
                          "+f"(acc[nt][3])
                        : "r"(as[0]), "r"(as[1]), "r"(as[2]), "r"(as[3]),
                          "r"(bb[0]), "r"(bb[1]));
                }
            }
        }
        // drain the tile's mma chain into the RN accumulator (see fac[])
        #pragma unroll
        for (uint32_t nt = 0; nt < 2u; ++nt)
            #pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) {
                fac[nt][e] += acc[nt][e];
                acc[nt][e] = 0.f;
            }
        __syncthreads();
    }
    #pragma unroll
    for (uint32_t nt = 0; nt < 2u; ++nt)
        #pragma unroll
        for (uint32_t e = 0; e < 4u; ++e) {
            const uint32_t m = m0 + mt * 16u + gr + (e >= 2u ? 8u : 0u);
            const uint32_t n = n0 + nh * 16u + nt * 8u + 2u * t4 + (e & 1u);
            if (m < batch) out[(size_t)m * out_dim + n] = fac[nt][e];
        }
}

// row x TT tokens, lanes stride K by 32. The block-per-(o, tile) shape
// re-read every x row once per OUTPUT (47 MB of L2 per 128x32 launch - the
// 18.5 us/layer line in the c32 ledger); tiling outputs 8-wide cuts x
// traffic 8x. Lane-strided sums regroup vs the 256-thread stride (order
// change is sanctioned; the token gates arbitrate).
template <uint32_t TT>
__global__ void pd_matvec_f32_tile_kernel(const float* __restrict__ w,
                                          const float* __restrict__ x,
                                          float* __restrict__ out,
                                          uint32_t in_dim, uint32_t out_dim,
                                          uint32_t batch) {
    const uint32_t o = blockIdx.x * 8u + (threadIdx.x >> 5);
    const uint32_t t0 = blockIdx.y * TT;
    const uint32_t lane = threadIdx.x & 31u;
    if (o >= out_dim) return;
    const float* wr = w + (size_t)o * in_dim;
    float acc[TT] = {};
    for (uint32_t i = lane; i < in_dim; i += 32u) {
        const float wv = wr[i];
        #pragma unroll
        for (uint32_t b = 0; b < TT; ++b)
            if (t0 + b < batch) acc[b] += wv * x[(size_t)(t0 + b) * in_dim + i];
    }
    #pragma unroll
    for (uint32_t b = 0; b < TT; ++b) {
        float v = acc[b];
        for (uint32_t sh = 16; sh > 0; sh >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, sh);
        if (lane == 0 && t0 + b < batch) out[(size_t)(t0 + b) * out_dim + o] = v;
    }
}

// K-split router matvec (slot 486; g26a4b act):
// the decode router shape (out 128, batch 16-32) ran the tile matvec at 0.34
// waves - an 88-iteration K walk of almost pure exposed latency, 15.8us for
// a 1.4MB plane (~22x the byte floor). Split K S ways across grid.z: warp =
// one output row x BT tokens over its window (the tile kernel's lane-stride
// walk verbatim inside the window), partials to caller scratch, then a fold
// kernel sums s ASCENDING per (b, o) - deterministic, launch-order-free.
// New summation order vs the tile matvec (router feeds top-k; the token
// gates arbitrate, same sanction as the tile-matvec regroup). Scratch is
// CALLER-OWNED (moe_part is dead at router time) - the nt_ks
// cudaMallocAsync straggler class stays out of the decode graph.
#define PD_RKS_S 8u
#define PD_RKS_BT 4u
__global__ void pd_matvec_f32_ks_kernel(const float* __restrict__ w,
                                        const float* __restrict__ x,
                                        float* __restrict__ part,
                                        uint32_t in_dim, uint32_t out_dim,
                                        uint32_t batch, uint32_t kwin) {
    const uint32_t o = blockIdx.x * 8u + (threadIdx.x >> 5);
    const uint32_t t0 = blockIdx.y * PD_RKS_BT;
    const uint32_t sidx = blockIdx.z;
    const uint32_t lane = threadIdx.x & 31u;
    if (o >= out_dim) return;
    const uint32_t k0 = sidx * kwin;
    const uint32_t k1 = min(k0 + kwin, in_dim);
    const float* wr = w + (size_t)o * in_dim;
    float acc[PD_RKS_BT] = {};
    for (uint32_t i = k0 + lane; i < k1; i += 32u) {
        const float wv = wr[i];
        #pragma unroll
        for (uint32_t b = 0; b < PD_RKS_BT; ++b)
            if (t0 + b < batch) acc[b] += wv * x[(size_t)(t0 + b) * in_dim + i];
    }
    #pragma unroll
    for (uint32_t b = 0; b < PD_RKS_BT; ++b) {
        float v = acc[b];
        for (uint32_t sh = 16; sh > 0; sh >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, sh);
        if (lane == 0 && t0 + b < batch)
            part[((size_t)sidx * batch + t0 + b) * out_dim + o] = v;
    }
}

// B3-1 (hibatch phase B): cooperative router stage - matvec + topk in one
// die-filling kernel with grid.sync between phases. Per-logit math is the
// tile matvec's VERBATIM walk (lane-stride ascending i + shfl_down tree),
// so logits are bit-identical to the chain; phase 2 is pd_moe_topk_warp +
// dscale fold verbatim. Work items distribute round-robin over all warps -
// no phase ever runs below die width (the die-starvation law's answer).
__global__ void __launch_bounds__(128) pd_moe_router_stage_kernel(
    const float* __restrict__ w, const float* __restrict__ x,
    float* __restrict__ logits, const float* __restrict__ dscale,
    unsigned int* __restrict__ out_idx, float* __restrict__ out_w,
    uint32_t in_dim, uint32_t out_dim, uint32_t batch, uint32_t k) {
#if __CUDA_ARCH__ >= 800
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();
    const uint32_t warp_g = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const uint32_t nwarp_g = (gridDim.x * blockDim.x) >> 5;
    const uint32_t lane = threadIdx.x & 31u;
    // phase 1: one (output x 4-token group) item per warp - the tile
    // kernel's exact per-warp shape (one 88-iter walk, TT=4), so the serial
    // depth matches the chain and per-logit math is verbatim.
    const uint32_t tgroups = (batch + 3u) >> 2;
    const uint32_t items = out_dim * tgroups;
    for (uint32_t it = warp_g; it < items; it += nwarp_g) {
        const uint32_t o = it % out_dim, tg = it / out_dim;
        const uint32_t t0 = tg * 4u;
        const float* wr = w + (size_t)o * in_dim;
        float acc[4] = {};
        for (uint32_t i = lane; i < in_dim; i += 32u) {
            const float wv = wr[i];
            #pragma unroll
            for (uint32_t b = 0; b < 4u; ++b)
                if (t0 + b < batch) acc[b] += wv * x[(size_t)(t0 + b) * in_dim + i];
        }
        #pragma unroll
        for (uint32_t b = 0; b < 4u; ++b) {
            float v = acc[b];
            for (uint32_t sh = 16; sh > 0; sh >>= 1)
                v += __shfl_down_sync(0xffffffffu, v, sh);
            if (lane == 0 && t0 + b < batch)
                logits[(size_t)(t0 + b) * out_dim + o] = v;
        }
    }
    grid.sync();
    // phase 2: token-per-warp topk + per-expert scale fold (verbatim)
    for (uint32_t t = warp_g; t < batch; t += nwarp_g) {
        unsigned int* oi = out_idx + (size_t)t * k;
        float* ow = out_w + (size_t)t * k;
        pd_moe_topk_warp(logits + (size_t)t * out_dim, (const float*)0, out_dim, k, oi, ow);
        if (lane == 0)
            for (uint32_t s2 = 0; s2 < k; ++s2) ow[s2] *= dscale[oi[s2]];
    }
#else
    (void)w;(void)x;(void)logits;(void)dscale;(void)out_idx;(void)out_w;
    (void)in_dim;(void)out_dim;(void)batch;(void)k;
#endif
}

PD_EXPORT
int pd_moe_router_stage(const void* w, const void* x, void* logits,
                        const void* dscale, void* out_idx, void* out_w,
                        uint32_t in_dim, uint32_t out_dim, uint32_t batch,
                        uint32_t k, void* stream) {
    if (batch == 0 || out_dim == 0) return 0;
    if ((out_dim & 7u) || out_dim > 256u || k > 16u) return cudaErrorInvalidValue;
    static int nsm = 0, cores = 0;
    if (!nsm) {
        int dev = 0; cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &cores, (const void*)pd_moe_router_stage_kernel, 128, 0);
        if (cores < 1) cores = 1;
    }
    // co-residency-sized grid: enough warps for one round of phase-1 items
    // (out_dim * ceil(batch/4)); 128-thr CTAs multiply co-residency.
    const uint32_t items = out_dim * ((batch + 3u) >> 2);
    const uint32_t want = (items * 32u + 127u) / 128u;   // CTAs for 1 round
    uint32_t gx = (uint32_t)nsm;
    while (gx < want && gx + (uint32_t)nsm <= (uint32_t)(nsm * cores)) gx += (uint32_t)nsm;
    dim3 g(gx), b(128);
    void* args[] = {(void*)&w, (void*)&x, (void*)&logits, (void*)&dscale,
                    (void*)&out_idx, (void*)&out_w, (void*)&in_dim,
                    (void*)&out_dim, (void*)&batch, (void*)&k};
    cudaError_t e = cudaLaunchCooperativeKernel(
        (const void*)pd_moe_router_stage_kernel, g, b, args, 0, (cudaStream_t)stream);
    return e ? (int)e : pd_launch_status();
}

__global__ void pd_matvec_f32_ks_fold_kernel(const float* __restrict__ part,
                                             float* __restrict__ out,
                                             uint32_t n, uint32_t nsplit,
                                             uint32_t stride) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = 0.f;
    for (uint32_t s = 0; s < nsplit; ++s) acc += part[(size_t)s * stride + i];
    out[i] = acc;
}

// scratch must hold PD_RKS_S * batch * out_dim floats (128KB at the c32
// router shape - moe_part covers it thousands of times over).
PD_EXPORT
int pd_matvec_f32_ks(const void* w, const void* x, void* scratch, void* out,
                     uint32_t in_dim, uint32_t out_dim, uint32_t batch,
                     void* stream) {
    if (out_dim == 0 || batch == 0) return 0;
    if ((out_dim & 7u) != 0 || (in_dim & 31u) != 0) return cudaErrorInvalidValue;
    const uint32_t kwin = ((in_dim / 32u + PD_RKS_S - 1u) / PD_RKS_S) * 32u;
    dim3 grid(out_dim / 8u, (batch + PD_RKS_BT - 1u) / PD_RKS_BT, PD_RKS_S);
    pd_matvec_f32_ks_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(
        (const float*)w, (const float*)x, (float*)scratch, in_dim, out_dim,
        batch, kwin);
    const uint32_t n = batch * out_dim;
    pd_matvec_f32_ks_fold_kernel<<<(n + 255u) / 256u, 256, 0,
                                   (cudaStream_t)stream>>>(
        (const float*)scratch, (float*)out, n, PD_RKS_S, n);
    return pd_launch_status();
}

// ---------------------------------------------------------------------------
// Batch-1 split-K matvec (slot 519). Every K-split arm above gates on
// batch >= 16 (or >= 1024), so a DECODE row falls through to one block per
// output row. For a skinny-out plane that is a starved launch: the qwen4_exp
// GDN alpha||beta plane is [in=2560, out=96] -> 96 blocks on a 148-SM die,
// measured 65 GB/s, and the MoE router [in=2560, out=513] -> 513 blocks.
//
// grid = (out_dim, SPLIT). Each block owns one K window; the last block to
// finish a row combines that row's partials in INDEX ORDER, so the result is
// bit-stable run to run (unlike an atomicAdd combine). The counter is reset by
// that same block, so a captured graph replays with the buffers back in their
// initial state.
//
// `partials` and `counters` are CALLER-OWNED and address-stable by contract:
// a lazy cudaMallocAsync in a kernel entry runs during decode graph capture,
// which is how a previous rung turned 700 tok/s into 17.9.
__global__ __launch_bounds__(256) void pd_matvec_f32_sk_kernel(
        const float* __restrict__ w, const float* __restrict__ x,
        float* __restrict__ out, float* __restrict__ partials,
        unsigned int* __restrict__ counters, uint32_t in_dim, uint32_t out_dim,
        uint32_t split) {
    const uint32_t o = blockIdx.x, sp = blockIdx.y;
    if (o >= out_dim) return;
    PD_PDL_ARM();
    const uint32_t chunk = (in_dim + split - 1u) / split;
    const uint32_t k0 = sp * chunk;
    const uint32_t k1 = (k0 + chunk < in_dim) ? (k0 + chunk) : in_dim;
    const float* __restrict__ row = w + (size_t)o * in_dim;
    float acc = 0.0f;
    for (uint32_t i = k0 + threadIdx.x; i < k1; i += 256u)
        acc += row[i] * x[i];
    __shared__ float ws[8];
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    for (uint32_t t = 16; t > 0; t >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, t);
    if (lane == 0) ws[warp] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float v = 0.0f;
        for (uint32_t t = 0; t < 8u; ++t) v += ws[t];
        partials[(size_t)o * split + sp] = v;
        __threadfence();
        unsigned int prev = atomicAdd(&counters[o], 1u);
        if (prev == split - 1u) {
            float sum = 0.0f;
            for (uint32_t t = 0; t < split; ++t)
                sum += partials[(size_t)o * split + t];
            out[o] = sum;
            counters[o] = 0u;      // graph-replay safe: back to initial state
        }
    }
}

PD_EXPORT
int pd_matvec_f32_sk(const void* w, const void* x, void* out, void* partials,
                     void* counters, uint32_t in_dim, uint32_t out_dim,
                     uint32_t split, void* stream) {
    if (out_dim == 0 || in_dim == 0) return 0;
    if (split < 2u || split > 64u) return cudaErrorInvalidValue;
    dim3 grid(out_dim, split);
    pd_pdl_go(pd_matvec_f32_sk_kernel, grid, 256u, 0u, (cudaStream_t)stream,
              (const float*)w, (const float*)x, (float*)out, (float*)partials,
              (unsigned int*)counters, in_dim, out_dim, split);
    return pd_launch_status();
}

PD_EXPORT
int pd_matvec_f32_batch(const void* w, const void* x, void* out, uint32_t in_dim,
                        uint32_t out_dim, uint32_t batch, void* stream) {
    if (out_dim == 0 || batch == 0) return 0;
    // Large-batch (prefill-chunk) router: tiled GEMM instead of the matvec
    // tile - see pd_gemm_f32_nt_kernel. Alignment: BN|out_dim, 4|in_dim's
    // float4 loads, BK|in_dim. Env-gated (accumulation-order change).
    static const bool rgemm = pd_env("PADDOCK_ROUTER_GEMM") != nullptr;
    // skinny-out K-split rung (pd_gemm_f32_nt_ks_kernel above): exact-f32
    // FMA per window, deterministic combine; refills the wave the decay/ba
    // plane class (out <= 128) leaves idle. Opt-in until gated.
    static const bool nt_ks = pd_env("PADDOCK_F32NT_KS") != nullptr;
    // tf32 rung (ba - kernel above): PADDOCK_BA_TF32=1 arms the
    // 3xTF32 arm, ="p1" the single-tf32 arm. Replaces nt_ks across the whole
    // batch>=1024 band: faster at every grid (the 1.3-wave tail zone still
    // halves nt_ks's wall) and ALLOCATION-FREE - nt_ks's per-call
    // cudaMallocAsync was the c16 ctl legs' ~865 ms TTFT p90 straggler class
    // (arm legs read p90 153-155 on the same boots).
    static const int ba_tf32 = [] {
        const char* e = pd_env("PADDOCK_BA_TF32");
        return e ? (e[0] == 'p' ? 1 : (atoi(e) != 0 ? 3 : 0)) : 0;
    }();
    // PADDOCK_ROUTER_GEMM_MIN (bench/A-B): the row floor the three tiled
    // arms engage at (default 1024, the big-die measurement)
    static const uint32_t rg_min = [] {
        const char* e = pd_env("PADDOCK_ROUTER_GEMM_MIN");
        return e ? (uint32_t)atoi(e) : 1024u;
    }();
    if (ba_tf32 && batch >= rg_min && out_dim <= 128u && (out_dim % 32u) == 0u &&
        (in_dim % PD_RGEMM_BK) == 0u) {
        dim3 grid((batch + 31u) / 32u, out_dim / 32u);
        if (ba_tf32 == 1)
            pd_gemm_tf32_nt_kernel<1u><<<grid, 128, 0, (cudaStream_t)stream>>>(
                (const float*)w, (const float*)x, (float*)out, in_dim, out_dim,
                batch);
        else
            pd_gemm_tf32_nt_kernel<3u><<<grid, 128, 0, (cudaStream_t)stream>>>(
                (const float*)w, (const float*)x, (float*)out, in_dim, out_dim,
                batch);
        return pd_launch_status();
    }
    // Small dies (< 128 SMs, GB10 2026-09-08): the K-split arm is the
    // election for skinny-out planes from 256 rows, no env needed. The
    // qwen3.8-27b alpha/beta plane ([96 x 5120] f32, 48 layers) on the tile
    // matvec re-reads x 12x and w 255x per launch (~760 MB through a 24 MB
    // L2 for a 23 MB problem): 516 us at 1017 rows, 284 at 512, 153 at
    // 256. Measured on the same die (bench/skinny_f32_gb10_bench.cu, best
    // of 8): K-split 199 / 135 / 95 us, 3xTF32 200 / 137 / 108, exact-f32
    // tiles 227 / 180 / 168. Exact f32 FMA in a fixed-order combine (the
    // regroup class the tile matvec itself ships under - the token gates
    // arbitrate); the big-die floors are untouched.
    static int nsm_sk = 0;
    if (nsm_sk == 0) { int d = 0; cudaGetDevice(&d); cudaDeviceGetAttribute(&nsm_sk, cudaDevAttrMultiProcessorCount, d); if (nsm_sk <= 0) nsm_sk = 148; }
    const bool ks_small = nsm_sk < 128 && batch >= 256u && pd_env("PADDOCK_NO_F32NT_KS") == nullptr;
    if ((nt_ks || ks_small) && (ks_small || batch >= rg_min) && out_dim <= 128u && (out_dim % 32u) == 0u &&
        (in_dim % PD_RGEMM_BK) == 0u) {
        const uint32_t base = ((batch + 63u) / 64u) * (out_dim / 32u);
        const uint32_t S = base >= 376u ? 1u : (376u + base - 1u) / base > 16u ? 16u : (376u + base - 1u) / base;
        if (S >= 2u)
            return pd_f32nt_ks_go((const float*)w, (const float*)x, (float*)out,
                                  in_dim, out_dim, batch, S, (cudaStream_t)stream);
    }
    if (rgemm && batch >= rg_min && (out_dim % 32u) == 0u &&
        (in_dim % PD_RGEMM_BK) == 0u) {
        // Tile by GRID FILL. Per output element the K walk (k0 chunks, kk
        // ascending, one owning thread) is tile-size-invariant, so every
        // rung is BIT-IDENTICAL - BM/BN only choose the owner. 64x64 was
        // the original pick (86 us @ router/2048 vs matvec 269 = 3.1x) but
        // its ceiling was grid fill: router = 128 blocks on a 188-SM die,
        // the alpha/beta plane (out 64) a catastrophic 32. Halving BM (and
        // BN when out allows) restores the wave: fewer rows/block = more
        // blocks, same L2-resident skinny weight.
        const uint32_t b44 = ((batch + 63u) / 64u) * (out_dim / 64u);
        if (b44 >= 160u) {
            dim3 grid((batch + 63u) / 64u, out_dim / 64u);
            pd_gemm_f32_nt_kernel<4u, 4u><<<grid, 256, 0, (cudaStream_t)stream>>>(
                (const float*)w, (const float*)x, (float*)out, in_dim, out_dim, batch);
        } else if ((out_dim % 64u) == 0u &&
                   ((batch + 31u) / 32u) * (out_dim / 64u) >= 160u) {
            dim3 grid((batch + 31u) / 32u, out_dim / 64u);
            pd_gemm_f32_nt_kernel<2u, 4u><<<grid, 256, 0, (cudaStream_t)stream>>>(
                (const float*)w, (const float*)x, (float*)out, in_dim, out_dim, batch);
        } else {
            dim3 grid((batch + 31u) / 32u, out_dim / 32u);
            pd_gemm_f32_nt_kernel<2u, 2u><<<grid, 256, 0, (cudaStream_t)stream>>>(
                (const float*)w, (const float*)x, (float*)out, in_dim, out_dim, batch);
        }
        return pd_launch_status();
    }
    if (batch >= 16u && (out_dim & 7u) == 0u) {
        // Fill the die. The tile grid is (out_dim/8) x ceil(batch/TT), and for
        // the MoE router (out = n_expert = 128) at the decode band that is
        // 16x8 = 128 blocks on a 188-SM part -- 0.34 waves, so the 88-iteration
        // K walk is almost pure exposed latency (~22x this shape's byte floor).
        // Shrinking TT multiplies the block count, and per-token sums are
        // TT-invariant (same i stride, same shfl tree) so every TT is
        // BIT-EXACT: it only trades extra (L2-resident) weight re-reads for
        // wave occupancy. Which way that trade goes is empirical, so TT stays
        // 4 until the sweep says otherwise -- PADDOCK_ROUTER_TT pins it for the
        // A/B and default behaviour is unchanged.
        // SWEPT on the A4B c32 shape (out=128, in=2816, batch=32),
        // 2 interleaved reps each, output throughput:
        //   TT=4  128 blocks  control
        //   TT=2  256 blocks  +1.17%
        //   TT=1  512 blocks  -2.30%
        // So fill-the-die is not monotonic: the second halving's extra weight
        // re-reads (32x the router plane per layer instead of 8x) overtake what
        // the wave fill buys. A "shrink until blocks >= 376" rule would have
        // picked TT=1 and cost 2.3% -- hence one step only, when TT=4 underfills.
        static const char* tt_e = pd_env("PADDOCK_ROUTER_TT");
        static const int tt_env = tt_e ? atoi(tt_e) : 0;
        const uint32_t cols = out_dim / 8u;
        const uint32_t tt = (tt_env == 1 || tt_env == 2 || tt_env == 4)
                                ? (uint32_t)tt_env
                                : (cols * ((batch + 3u) / 4u) < 376u ? 2u : 4u);
        dim3 grid(cols, (batch + tt - 1u) / tt);
        if (tt == 1u)
            pd_matvec_f32_tile_kernel<1u><<<grid, 256, 0, (cudaStream_t)stream>>>(
                (const float*)w, (const float*)x, (float*)out, in_dim, out_dim, batch);
        else if (tt == 2u)
            pd_matvec_f32_tile_kernel<2u><<<grid, 256, 0, (cudaStream_t)stream>>>(
                (const float*)w, (const float*)x, (float*)out, in_dim, out_dim, batch);
        else
            pd_matvec_f32_tile_kernel<4u><<<grid, 256, 0, (cudaStream_t)stream>>>(
                (const float*)w, (const float*)x, (float*)out, in_dim, out_dim, batch);
        return pd_launch_status();
    }
    // BT adapts to fill the die: at b=8 a BT=8 grid is (128,1) = 0.7 waves of
    // pure latency (11.4 us in the c8 profile); BT=2 quadruples the blocks
    // for the same L2-resident weight. Per-token sums are BT-invariant.
    //
    // "for the same L2-RESIDENT weight" is the load-bearing clause, and it is
    // false for the qwen4_exp router. Both elections below were swept on
    // laguna's [in=2816, out=128] plane -- 1.44 MB, which stays in L2 across
    // the whole tick, so gridY re-reads cost L2 bandwidth and buying blocks
    // with them is free. qwen4_exp routes over 513 rows of 2560: a 5.25 MB
    // plane, 48 DISTINCT ones per tick, and ~290 MB of other weights streams
    // between two consecutive routers -- so every gridY re-read is a DRAM
    // re-read. Measured in the c8 serve profile (nsys, median tick): grid
    // (512,4), 18.80 us/launch, 903 us/tick, 21 MB of traffic for a 5.25 MB
    // plane -- 0.28 TB/s, against sglang's one 5.02 us bf16 router GEMM.
    //
    // BT is therefore PINNABLE while the two regimes are measured apart. The
    // pin is bit-exact by the same argument the elections already carry (same
    // per-thread i stride, same shfl tree, same serial cross-warp sum), so it
    // trades DRAM re-reads against block count and nothing else.
    static const char* bt_e = pd_env("PADDOCK_MATVEC_BT");
    static const int bt_pin = bt_e ? atoi(bt_e) : 0;
    const uint32_t bt = (bt_pin == 2 || bt_pin == 4 || bt_pin == 8 || bt_pin == 16)
                            ? (uint32_t)bt_pin
                            : (batch < 16u ? 2u : 4u);
    dim3 grid(out_dim, (batch + bt - 1u) / bt);
    switch (bt) {
    case 16u:
        pd_pdl_go(pd_matvec_f32_batch_kernel<16u>, grid, 256, 0u, (cudaStream_t)stream,
            (const float*)w, (const float*)x, (float*)out, in_dim, out_dim, batch);
        break;
    case 8u:
        pd_pdl_go(pd_matvec_f32_batch_kernel<8u>, grid, 256, 0u, (cudaStream_t)stream,
            (const float*)w, (const float*)x, (float*)out, in_dim, out_dim, batch);
        break;
    case 4u:
        // BT=4 (was 8): at c32 the 8-token tile ran 18.5 us/layer - the 8
        // in-flight x rows starve the per-thread load pipe; 4 doubles the
        // block count for the same L2-resident weight. Per-token sums are
        // BT-invariant (same i stride) - bit-exact across BT.
        pd_pdl_go(pd_matvec_f32_batch_kernel<4u>, grid, 256, 0u, (cudaStream_t)stream,
            (const float*)w, (const float*)x, (float*)out, in_dim, out_dim, batch);
        break;
    default:
        pd_pdl_go(pd_matvec_f32_batch_kernel<2u>, grid, 256, 0u, (cudaStream_t)stream,
            (const float*)w, (const float*)x, (float*)out, in_dim, out_dim, batch);
        break;
    }
    return pd_launch_status();
}

// Unconditionally-tiled f32 GEMM over a [out, in]-major weight (y = x·Wᵀ per
// row): the k-quant batch/prefill interim's compute stage - its weights are
// WIDE (ffn outs), where the matvec tile's per-TT-token weight re-read is
// pathological. Same pd_gemm_f32_nt_kernel rungs as the router (K walk is
// tile-size-invariant -> rungs bit-identical); falls back to the matvec tile
// only on misaligned shapes. Separate export so the ROUTER keeps its env-gated
// numerics (its callers are b9895-parity-pinned).
PD_EXPORT
int pd_gemm_f32(const void* w, const void* x, void* out, uint32_t in_dim,
                uint32_t out_dim, uint32_t batch, void* stream) {
    if (out_dim == 0 || batch == 0) return 0;
    // tf32 tensor-core arm (nemotron prefill rung). This
    // export is nemotron's prefill attention q/k/v/o, and the SIMT tile runs
    // it at ~23% of f32 peak: a 537x4096x2688 q_proj reads 411 us, o_proj
    // 365, k/v 81 each -> 5.6 ms per prefill tick, 24 launches, ~2% of the
    // c32 wall. Same kernel as the `ba` rung above; 3xTF32 (PREC=3) is
    // STRICTLY FINER than the bf16 class the batched decode twins already
    // run these exact planes in, so this raises prefill toward the SIMT
    // f32 it replaces rather than dropping below the served class.
    //
    // The gate is SHAPE-ONLY deliberately. layer_walk's contract is granite's
    // law -- every prefill row takes the same rungs at any r, so a
    // warm-resume tail reproduces the cold chunk's bytes -- so this must
    // never key on batch. The kernel is r-safe: M staging zero-fills via
    // cp.async (`m0+xr < batch ? 16u : 0u`) and the store guards `m < batch`;
    // the unguarded N axis is what the out_dim % 32 gate is for.
    //
    // ELECTED DEFAULT: PREC=3. Unset = 3, "0" = the SIMT
    // f32 tile (the A/B pin, PADDOCK_NVQKV=0 precedent), "p" = the single-tf32
    // probe arm. PREC=3 is elected over PREC=1 because on these shapes the
    // kernel is STAGING-bound, not compute-bound -- 3 mma passes vs 1 measured
    // 1.00x/0.99x on q_proj/o_proj (342.1 vs 340.7 us, 348.5 vs 350.6), so the
    // finer numerics are FREE. Both arms measured closer to the parity-pinned
    // serial reference than the f32 they replace (bulk-vs-serial mean |d|
    // 0.18814 -> 0.16544; resume 0.215193 -> 0.165267), because the SIMT
    // tile's accumulation ORDER was a bigger error source than tf32's mantissa.
    //
    // The other caller of this export is qwen35 ops.rs:192 (kq_gemm, the wdq
    // plane) -- checked at election time and not affected in any shipped
    // config: it sits behind PADDOCK_KQ_F32_PREFILL, which is off by default.
    static const int gf_tf32 = [] {
        const char* e = pd_env("PADDOCK_GEMMF32_TF32");
        return e ? (e[0] == 'p' ? 1 : (atoi(e) != 0 ? 3 : 0)) : 3;
    }();
    if (gf_tf32 && (out_dim % 32u) == 0u && (in_dim % PD_RGEMM_BK) == 0u) {
        dim3 grid((batch + 31u) / 32u, out_dim / 32u);
        if (gf_tf32 == 1)
            pd_gemm_tf32_nt_kernel<1u><<<grid, 128, 0, (cudaStream_t)stream>>>(
                (const float*)w, (const float*)x, (float*)out, in_dim, out_dim,
                batch);
        else
            pd_gemm_tf32_nt_kernel<3u><<<grid, 128, 0, (cudaStream_t)stream>>>(
                (const float*)w, (const float*)x, (float*)out, in_dim, out_dim,
                batch);
        return pd_launch_status();
    }
    if ((out_dim % 32u) == 0u && (in_dim % PD_RGEMM_BK) == 0u) {
        const uint32_t b44 = ((batch + 63u) / 64u) * (out_dim / 64u);
        if ((out_dim % 64u) == 0u && b44 >= 160u) {
            dim3 grid((batch + 63u) / 64u, out_dim / 64u);
            pd_gemm_f32_nt_kernel<4u, 4u><<<grid, 256, 0, (cudaStream_t)stream>>>(
                (const float*)w, (const float*)x, (float*)out, in_dim, out_dim, batch);
        } else if ((out_dim % 64u) == 0u &&
                   ((batch + 31u) / 32u) * (out_dim / 64u) >= 160u) {
            dim3 grid((batch + 31u) / 32u, out_dim / 64u);
            pd_gemm_f32_nt_kernel<2u, 4u><<<grid, 256, 0, (cudaStream_t)stream>>>(
                (const float*)w, (const float*)x, (float*)out, in_dim, out_dim, batch);
        } else {
            dim3 grid((batch + 31u) / 32u, out_dim / 32u);
            pd_gemm_f32_nt_kernel<2u, 2u><<<grid, 256, 0, (cudaStream_t)stream>>>(
                (const float*)w, (const float*)x, (float*)out, in_dim, out_dim, batch);
        }
        return pd_launch_status();
    }
    if (batch >= 16u && (out_dim & 7u) == 0u) {
        dim3 grid(out_dim / 8u, (batch + 3u) / 4u);
        pd_matvec_f32_tile_kernel<4u><<<grid, 256, 0, (cudaStream_t)stream>>>(
            (const float*)w, (const float*)x, (float*)out, in_dim, out_dim, batch);
        return pd_launch_status();
    }
    dim3 grid(out_dim, (batch + 3u) / 4u);
    pd_pdl_go(pd_matvec_f32_batch_kernel<4u>, grid, 256, 0u, (cudaStream_t)stream,
        (const float*)w, (const float*)x, (float*)out, in_dim, out_dim, batch);
    return pd_launch_status();
}

static int pd_moe_topk_batch_rs(const void* logits, const void* bias, uint32_t n_expert,
                                uint32_t rs, uint32_t k, void* out_idx, void* out_w,
                                uint32_t batch, void* stream) {
    if (batch == 0) return 0;
    if (k > 16u || n_expert > PD_MOE_TOPK_MAX_EXPERTS) return cudaErrorInvalidValue;
    // > 256 experts needs the wide walk (qwen4_exp routes over 512); the
    // narrow one would silently ignore everything past expert 255.
    // Stage the logits through shared memory with a full block when the row is
    // wide enough that a lone warp's strided global read dominates. Selection
    // is the same routine on the same values in the same order => bit-identical
    // picks and weights; this is a pure schedule change, like the BT tile note
    // above. PADDOCK_MOE_TOPK_WARP=1 returns to the warp-only launch.
    static const bool warp_only = pd_env("PADDOCK_MOE_TOPK_WARP") != nullptr;
    // BLOCK-PARALLEL selection, OPT-IN (PADDOCK_MOE_TOPK_BLK=1). It shortens
    // the dependent chain (k*(VJ+10) steps instead of k*(VJ+7) in a lone warp
    // with 15 idle ones) and its picks are identical -- and it MEASURED A LOSS
    // of 1.2% on the qwen4_exp c1 serve cell (161.37 with the warp selection
    // vs 159.36/159.54 with this one, alternating boots 2026-08-31). The
    // 20 __syncthreads a launch cost more than the chain they save. Kept
    // because it is the right shape for wider routers/batches, not defaulted.
    static const bool blk_on = [] {
        const char* e = pd_env("PADDOCK_MOE_TOPK_BLK");
        return e && e[0] == '1';
    }();
    if (!warp_only && blk_on && n_expert == 512u && k <= 16u) {
        pd_moe_topk_batch_blk_kernel_t<16u, 1u><<<batch, 512, 0, (cudaStream_t)stream>>>(
            (const float*)logits, (const float*)bias, n_expert, rs, k,
            (uint32_t*)out_idx, (float*)out_w);
        return pd_launch_status();
    }
    if (!warp_only && n_expert >= 128u) {
        const uint32_t nth = n_expert >= 512u ? 512u : 256u;
        const size_t shm = (size_t)n_expert * sizeof(float);
        if (n_expert > 256u)
            pd_moe_topk_batch_shm_kernel_t<16u><<<batch, nth, shm, (cudaStream_t)stream>>>(
                (const float*)logits, (const float*)bias, n_expert, rs, k, (uint32_t*)out_idx, (float*)out_w);
        else
            pd_moe_topk_batch_shm_kernel_t<8u><<<batch, nth, shm, (cudaStream_t)stream>>>(
                (const float*)logits, (const float*)bias, n_expert, rs, k, (uint32_t*)out_idx, (float*)out_w);
        return pd_launch_status();
    }
    if (n_expert > 256u) {
        pd_moe_topk_batch_kernel_t<16u><<<batch, 32, 0, (cudaStream_t)stream>>>(
            (const float*)logits, (const float*)bias, n_expert, rs, k, (uint32_t*)out_idx, (float*)out_w);
    } else {
        pd_moe_topk_batch_kernel_t<8u><<<batch, 32, 0, (cudaStream_t)stream>>>(
            (const float*)logits, (const float*)bias, n_expert, rs, k, (uint32_t*)out_idx, (float*)out_w);
    }
    return pd_launch_status();
}

PD_EXPORT
int pd_moe_topk_batch(const void* logits, const void* bias, uint32_t n_expert, uint32_t k,
                      void* out_idx, void* out_w, uint32_t batch, void* stream) {
    return pd_moe_topk_batch_rs(logits, bias, n_expert, n_expert, k, out_idx, out_w, batch, stream);
}

// slot 539: strided-logits twin - [n, rs] plane, selection bit-identical.
PD_EXPORT
int pd_moe_topk_batch_s(const void* logits, const void* bias, uint32_t n_expert, uint32_t rs,
                        uint32_t k, void* out_idx, void* out_w, uint32_t batch, void* stream) {
    if (rs < n_expert) return cudaErrorInvalidValue;
    return pd_moe_topk_batch_rs(logits, bias, n_expert, rs, k, out_idx, out_w, batch, stream);
}

PD_EXPORT
int pd_moe_slot_map(const void* idx, void* slot_of, uint32_t n_active, uint32_t n_expert,
                    uint32_t batch, void* stream) {
    if (batch == 0) return 0;
    pd_moe_slot_map_kernel<<<batch, 1, 0, (cudaStream_t)stream>>>(
        (const unsigned int*)idx, (unsigned char*)slot_of, n_active, n_expert, batch);
    return pd_launch_status();
}

PD_EXPORT
int pd_mxfp4_moe_gate_up_grouped(const void* gate_data, const void* gate_scale,
                                 const void* gate_bias, const void* up_data, const void* up_scale,
                                 const void* up_bias, const void* slot_of, const void* x, void* out,
                                 uint32_t in_dim, uint32_t ff, uint32_t n_expert, uint32_t n_active,
                                 uint32_t batch, float alpha, float limit, void* stream) {
    if (ff == 0 || n_expert == 0 || batch == 0) return 0;
    uint32_t threads = 256, nwarps = (threads + 31u) >> 5, n_blocks = in_dim >> 5;
    size_t shmem = ((size_t)16 + 2 * in_dim + 2 * n_blocks + 2 * nwarps) * sizeof(float);
    dim3 grid(ff, n_expert);
    pd_mxfp4_moe_gate_up_grouped_kernel<<<grid, threads, shmem, (cudaStream_t)stream>>>(
        (const unsigned char*)gate_data, (const unsigned char*)gate_scale, (const float*)gate_bias,
        (const unsigned char*)up_data, (const unsigned char*)up_scale, (const float*)up_bias,
        (const unsigned char*)slot_of, (const float*)x, (float*)out,
        in_dim, ff, n_expert, n_active, batch, alpha, limit);
    return pd_launch_status();
}

PD_EXPORT
int pd_mxfp4_moe_gate_up_gemm(const void* gate_W, const void* gate_bias, const void* up_W,
                              const void* up_bias, const void* slot_of, const void* x, void* out,
                              uint32_t in_dim, uint32_t ff, uint32_t n_expert, uint32_t n_active,
                              uint32_t batch, float alpha, float limit, void* stream) {
    if (ff == 0 || n_expert == 0 || batch == 0) return 0;
    uint32_t threads = 256;
    dim3 grid((ff + PD_MOE_BN - 1) / PD_MOE_BN, n_expert);
    // lut[16] + As[BM*BK] + Bg/Bu[BN*BK each] floats + gb/gs[batch] ints
    size_t shmem = ((size_t)16 + PD_MOE_BM * PD_MOE_BK + 2 * PD_MOE_BN * PD_MOE_BK) * sizeof(float)
                 + (size_t)2 * batch * sizeof(int);
    pd_mxfp4_moe_gate_up_gemm_kernel<<<grid, threads, shmem, (cudaStream_t)stream>>>(
        (const unsigned char*)gate_W, (const float*)gate_bias, (const unsigned char*)up_W,
        (const float*)up_bias, (const unsigned char*)slot_of, (const float*)x, (float*)out,
        in_dim, ff, n_expert, n_active, batch, alpha, limit);
    return pd_launch_status();
}

PD_EXPORT
int pd_qkv_rope_append_batch(const void* qkv, void* q_out, void* k_cache, void* v_cache,
                             const void* positions, const void* slots, uint32_t n_heads,
                             uint32_t n_kv_heads, uint32_t head_dim, uint32_t max_ctx,
                             float theta_scale, float freq_scale, float corr_low,
                             float corr_high, float ext_factor, float mscale,
                             uint32_t batch, uint32_t kv_dtype, void* stream) {
    if (batch == 0) return 0;
    const uint32_t warps = batch * (n_heads + 2u * n_kv_heads);
    const uint32_t blocks = (warps + 7u) / 8u;
    auto st = (cudaStream_t)stream;
    if (kv_dtype == PD_KV_FP8_E4M3) {
        pd_qkv_rope_append_batch_kernel<__nv_fp8_e4m3><<<blocks, 256, 0, st>>>(
            (const float*)qkv, (float*)q_out, (__nv_fp8_e4m3*)k_cache,
            (__nv_fp8_e4m3*)v_cache, (const unsigned int*)positions,
            (const unsigned int*)slots, n_heads, n_kv_heads, head_dim, max_ctx,
            theta_scale, freq_scale, corr_low, corr_high, ext_factor, mscale, batch);
    } else {
        pd_qkv_rope_append_batch_kernel<__half><<<blocks, 256, 0, st>>>(
            (const float*)qkv, (float*)q_out, (__half*)k_cache, (__half*)v_cache,
            (const unsigned int*)positions, (const unsigned int*)slots, n_heads,
            n_kv_heads, head_dim, max_ctx, theta_scale, freq_scale, corr_low,
            corr_high, ext_factor, mscale, batch);
    }
    return pd_launch_status();
}

// Paged twin launcher: block-table append into the [n_blocks,16,kvdim] pool.
// Same grid/threads; max_ctx dropped, block_tables + blocks_per_slot appended.
PD_EXPORT
int pd_qkv_rope_append_batch_paged(const void* qkv, void* q_out, void* k_cache, void* v_cache,
                                   const void* positions, const void* slots, uint32_t n_heads,
                                   uint32_t n_kv_heads, uint32_t head_dim,
                                   float theta_scale, float freq_scale, float corr_low,
                                   float corr_high, float ext_factor, float mscale,
                                   uint32_t batch, const void* block_tables,
                                   uint32_t blocks_per_slot, uint32_t kv_dtype, void* stream) {
    if (batch == 0) return 0;
    const uint32_t warps = batch * (n_heads + 2u * n_kv_heads);
    const uint32_t blocks = (warps + 7u) / 8u;
    auto st = (cudaStream_t)stream;
    if (kv_dtype == PD_KV_FP8_E4M3) {
        pd_qkv_rope_append_batch_paged_kernel<__nv_fp8_e4m3><<<blocks, 256, 0, st>>>(
            (const float*)qkv, (float*)q_out, (__nv_fp8_e4m3*)k_cache,
            (__nv_fp8_e4m3*)v_cache, (const unsigned int*)positions,
            (const unsigned int*)slots, n_heads, n_kv_heads, head_dim,
            theta_scale, freq_scale, corr_low, corr_high, ext_factor, mscale, batch,
            (const uint32_t*)block_tables, blocks_per_slot);
    } else {
        pd_qkv_rope_append_batch_paged_kernel<__half><<<blocks, 256, 0, st>>>(
            (const float*)qkv, (float*)q_out, (__half*)k_cache, (__half*)v_cache,
            (const unsigned int*)positions, (const unsigned int*)slots, n_heads,
            n_kv_heads, head_dim, theta_scale, freq_scale, corr_low,
            corr_high, ext_factor, mscale, batch,
            (const uint32_t*)block_tables, blocks_per_slot);
    }
    return pd_launch_status();
}

PD_EXPORT
int pd_moe_align(const void* idx, void* sorted_row, void* sorted_slot, void* block_expert,
                 uint32_t rows, uint32_t n_active, uint32_t n_expert, uint32_t max_blocks,
                 void* stream) {
    if (rows == 0 || n_expert == 0) return 0;
    size_t shmem = (size_t)3 * n_expert * sizeof(unsigned int);
    pd_pdl_go(pd_moe_align_kernel, 1, 1024, shmem, (cudaStream_t)stream, 
        (const unsigned int*)idx, (unsigned int*)sorted_row, (unsigned int*)sorted_slot,
        (unsigned int*)block_expert, rows, n_active, n_expert, PD_MOE_BM, max_blocks);
    return pd_launch_status();
}

// moe_align with a caller-chosen block tile (the bs64 prefill path sorts into
// 64-row blocks; everything else stays on PD_MOE_BM=32). Same kernel.
PD_EXPORT
int pd_moe_align_bm(const void* idx, void* sorted_row, void* sorted_slot,
                    void* block_expert, uint32_t rows, uint32_t n_active,
                    uint32_t n_expert, uint32_t bm, uint32_t max_blocks, void* stream) {
    if (rows == 0 || n_expert == 0) return 0;
    if (bm == 0 || (bm & (bm - 1u))) return cudaErrorInvalidValue;
    size_t shmem = (size_t)3 * n_expert * sizeof(unsigned int);
    pd_moe_align_kernel<<<1, 1024, shmem, (cudaStream_t)stream>>>(
        (const unsigned int*)idx, (unsigned int*)sorted_row, (unsigned int*)sorted_slot,
        (unsigned int*)block_expert, rows, n_active, n_expert, bm, max_blocks);
    return pd_launch_status();
}

// Dual-output align (g2 lane, slot 505): one pass emits the bm32 CSR, the
// bm16 CSR AND the pair map (pmap[tok*n_active+slot] = bm32 row) - replaces
// the align + align16 + pair_map triple when the g2 GU is elected. Same
// histogram/scan/scatter machinery as pd_moe_align_kernel; the two scans
// run sequentially over shared windows.
__global__ void pd_moe_align_dual_kernel(
    const unsigned int* __restrict__ idx, unsigned int* __restrict__ sr32,
    unsigned int* __restrict__ ss32, unsigned int* __restrict__ be32,
    unsigned int* __restrict__ sr16, unsigned int* __restrict__ ss16,
    unsigned int* __restrict__ be16, unsigned int* __restrict__ pmap,
    uint32_t rows, uint32_t n_active, uint32_t n_expert, uint32_t mb32,
    uint32_t mb16) {
    extern __shared__ unsigned int pd_dal_sh[];
    unsigned int* count = pd_dal_sh;                    // [n_expert]
    unsigned int* boff32 = count + n_expert;      // [n_expert]
    unsigned int* boff16 = boff32 + n_expert;     // [n_expert]
    unsigned int* fill32 = boff16 + n_expert;     // [n_expert]
    unsigned int* fill16 = fill32 + n_expert;     // [n_expert]
    __shared__ unsigned int wsum[32];
    __shared__ unsigned int carry_sh;
    uint32_t tid = threadIdx.x, nth = blockDim.x;
    uint32_t lane = tid & 31u, warp = tid >> 5, nwarp = (nth + 31u) >> 5;
    uint32_t npairs = rows * n_active;

    for (uint32_t e = tid; e < n_expert; e += nth) {
        count[e] = 0; fill32[e] = 0; fill16[e] = 0;
    }
    if (tid == 0) carry_sh = 0;
    __syncthreads();
    for (uint32_t p = tid; p < npairs; p += nth) atomicAdd(&count[idx[p]], 1u);
    __syncthreads();

    #pragma unroll
    for (uint32_t pass = 0; pass < 2u; ++pass) {
        const uint32_t bm = pass ? 16u : 32u;
        unsigned int* boff = pass ? boff16 : boff32;
        if (tid == 0) carry_sh = 0;
        __syncthreads();
        for (uint32_t base = 0; base < n_expert; base += nth) {
            uint32_t e = base + tid;
            uint32_t nb = (e < n_expert) ? (count[e] + bm - 1u) / bm : 0u;
            uint32_t x = nb;
            #pragma unroll
            for (uint32_t d = 1; d < 32u; d *= 2u) {
                uint32_t v = __shfl_up_sync(0xffffffffu, x, d);
                if (lane >= d) x += v;
            }
            if (lane == 31u) wsum[warp] = x;
            __syncthreads();
            if (warp == 0) {
                uint32_t w = (lane < nwarp) ? wsum[lane] : 0u;
                #pragma unroll
                for (uint32_t d = 1; d < 32u; d *= 2u) {
                    uint32_t v = __shfl_up_sync(0xffffffffu, w, d);
                    if (lane >= d) w += v;
                }
                wsum[lane] = w;
            }
            __syncthreads();
            uint32_t excl = x - nb + (warp ? wsum[warp - 1u] : 0u) + carry_sh;
            if (e < n_expert) boff[e] = excl;
            __syncthreads();
            if (tid == nth - 1u) carry_sh = excl + nb;
            __syncthreads();
        }
        const uint32_t bacc = carry_sh;
        unsigned int* be = pass ? be16 : be32;
        unsigned int* sr = pass ? sr16 : sr32;
        const uint32_t mbmax = pass ? mb16 : mb32;
        for (uint32_t e = tid; e < n_expert; e += nth) {
            uint32_t nb = (count[e] + bm - 1u) / bm, b0 = boff[e];
            for (uint32_t b = 0; b < nb; ++b) be[b0 + b] = e;
        }
        for (uint32_t b = bacc + tid; b < mbmax; b += nth) be[b] = PD_MOE_PAD;
        for (uint32_t i = tid; i < bacc * bm; i += nth) sr[i] = PD_MOE_PAD;
        __syncthreads();
    }

    for (uint32_t p = tid; p < npairs; p += nth) {
        unsigned int e = idx[p];
        unsigned int p32 = boff32[e] * 32u + atomicAdd(&fill32[e], 1u);
        unsigned int p16 = boff16[e] * 16u + atomicAdd(&fill16[e], 1u);
        sr32[p32] = p / n_active;
        ss32[p32] = p % n_active;
        sr16[p16] = p / n_active;
        ss16[p16] = p % n_active;
        pmap[p] = p32;
    }
}

PD_EXPORT
int pd_moe_align_dual(const void* idx, void* sr32, void* ss32, void* be32,
                      void* sr16, void* ss16, void* be16, void* pmap,
                      uint32_t rows, uint32_t n_active, uint32_t n_expert,
                      uint32_t mb32, uint32_t mb16, void* stream) {
    if (rows == 0 || n_expert == 0) return 0;
    size_t shmem = (size_t)5 * n_expert * sizeof(unsigned int);
    pd_moe_align_dual_kernel<<<1, 1024, shmem, (cudaStream_t)stream>>>(
        (const unsigned int*)idx, (unsigned int*)sr32, (unsigned int*)ss32,
        (unsigned int*)be32, (unsigned int*)sr16, (unsigned int*)ss16,
        (unsigned int*)be16, (unsigned int*)pmap, rows, n_active, n_expert,
        mb32, mb16);
    return pd_launch_status();
}

PD_EXPORT
int pd_mxfp4_moe_gate_up_gemm_sorted(const void* gate_data, const void* gate_scale,
                                     const void* gate_bias, const void* up_data,
                                     const void* up_scale, const void* up_bias,
                                     const void* sorted_row, const void* block_expert, const void* x,
                                     void* fused_sorted, uint32_t in_dim, uint32_t ff,
                                     uint32_t max_blocks, float alpha, float limit, uint32_t use_tc,
                                     void* stream) {
    if (ff == 0 || max_blocks == 0) return 0;
    uint32_t threads = 256;
    dim3 grid((ff + PD_MOE_SBN - 1) / PD_MOE_SBN, max_blocks);
    if (use_tc) {
        // f16 A + gate/up B (padded TC layout); the Bg/Bu region doubles as the
        // per-warp f32 epilogue scratch (8 warps × 2×16×16 fits inside it).
        size_t shmem = (size_t)16 * sizeof(float)
                     + ((size_t)PD_MOE_BM * PD_MOE_TC_LDA + 2 * PD_MOE_BK * PD_MOE_TC_LDB)
                           * sizeof(__half);
        static bool carveout_tc = false;
        if (!carveout_tc) {
            pd_prefer_max_shared(pd_mxfp4_moe_gate_up_gemm_sorted_tc_kernel);
            carveout_tc = true;
        }
        pd_mxfp4_moe_gate_up_gemm_sorted_tc_kernel<<<grid, threads, shmem, (cudaStream_t)stream>>>(
            (const unsigned char*)gate_data, (const unsigned char*)gate_scale, (const float*)gate_bias,
            (const unsigned char*)up_data, (const unsigned char*)up_scale, (const float*)up_bias,
            (const unsigned int*)sorted_row, (const unsigned int*)block_expert,
            (const float*)x, (float*)fused_sorted, in_dim, ff, alpha, limit);
        return pd_launch_status();
    }
    size_t shmem = ((size_t)16 + PD_MOE_BK * PD_MOE_LDA) * sizeof(float)
                 + (size_t)2 * PD_MOE_BK * PD_MOE_LDB * sizeof(__half);
    static bool carveout = false;
    if (!carveout) { pd_prefer_max_shared(pd_mxfp4_moe_gate_up_gemm_sorted_kernel); carveout = true; }
    pd_mxfp4_moe_gate_up_gemm_sorted_kernel<<<grid, threads, shmem, (cudaStream_t)stream>>>(
        (const unsigned char*)gate_data, (const unsigned char*)gate_scale, (const float*)gate_bias,
        (const unsigned char*)up_data, (const unsigned char*)up_scale, (const float*)up_bias,
        (const unsigned int*)sorted_row, (const unsigned int*)block_expert,
        (const float*)x, (float*)fused_sorted, in_dim, ff, alpha, limit);
    return pd_launch_status();
}

PD_EXPORT
int pd_mxfp4_moe_down_gemm_sorted(const void* down_data, const void* down_scale,
                                  const void* down_bias, const void* sorted_row,
                                  const void* sorted_slot, const void* block_expert,
                                  const void* topk_w, const void* fused_sorted, void* residual,
                                  uint32_t ff, uint32_t embd, uint32_t n_active, uint32_t max_blocks,
                                  uint32_t use_tc, void* stream) {
    if (embd == 0 || max_blocks == 0) return 0;
    uint32_t threads = 256;
    dim3 grid((embd + PD_MOE_SBN - 1) / PD_MOE_SBN, max_blocks);
    if (use_tc) {
        size_t shmem = (size_t)16 * sizeof(float)
                     + ((size_t)PD_MOE_BM * PD_MOE_TC_LDA + PD_MOE_BK * PD_MOE_TC_LDB) * sizeof(__half);
        static bool carveout_tc = false;
        if (!carveout_tc) {
            pd_prefer_max_shared(pd_mxfp4_moe_down_gemm_sorted_tc_kernel);
            carveout_tc = true;
        }
        pd_mxfp4_moe_down_gemm_sorted_tc_kernel<<<grid, threads, shmem, (cudaStream_t)stream>>>(
            (const unsigned char*)down_data, (const unsigned char*)down_scale, (const float*)down_bias,
            (const unsigned int*)sorted_row, (const unsigned int*)sorted_slot,
            (const unsigned int*)block_expert, (const float*)topk_w,
            (const float*)fused_sorted, (float*)residual, ff, embd, n_active);
        return pd_launch_status();
    }
    size_t shmem = ((size_t)16 + PD_MOE_BK * PD_MOE_LDA) * sizeof(float)
                 + (size_t)PD_MOE_BK * PD_MOE_LDB * sizeof(__half);
    static bool carveout = false;
    if (!carveout) { pd_prefer_max_shared(pd_mxfp4_moe_down_gemm_sorted_kernel); carveout = true; }
    pd_mxfp4_moe_down_gemm_sorted_kernel<<<grid, threads, shmem, (cudaStream_t)stream>>>(
        (const unsigned char*)down_data, (const unsigned char*)down_scale, (const float*)down_bias,
        (const unsigned int*)sorted_row, (const unsigned int*)sorted_slot,
        (const unsigned int*)block_expert, (const float*)topk_w,
        (const float*)fused_sorted, (float*)residual, ff, embd, n_active);
    return pd_launch_status();
}

PD_EXPORT
int pd_mxfp4_repack(const void* src, void* dst_data, void* dst_scale, uint64_t n_blocks,
                    void* stream) {
    if (n_blocks == 0) return 0;
    uint32_t threads = 256;
    uint64_t blocks = (n_blocks + threads - 1) / threads;
    pd_mxfp4_repack_kernel<<<(uint32_t)blocks, threads, 0, (cudaStream_t)stream>>>(
        (const unsigned char*)src, (unsigned char*)dst_data, (unsigned char*)dst_scale, n_blocks);
    return pd_launch_status();
}

// Load-time g||u interleave: fuse the repacked gate and up planes into one
// plane of 128 B pairs - pair p of row R = [gate blocks 4p..4p+3 (64 B) |
// up blocks 4p..4p+3 (64 B)], row pitch ceil(n_kb/4)*128 (tail pad stays
// zero from alloc). The bs producers and the dp4a MoE read this layout as
// one contiguous 128 B stream per (row, chunk) - the KC=128 granularity fix.
__global__ void pd_mxfp4_gu_interleave_kernel(
    const unsigned char* __restrict__ gate, const unsigned char* __restrict__ up,
    unsigned char* __restrict__ dst, uint32_t n_kb, uint64_t rows) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = rows * n_kb;
    if (i >= total) return;
    const uint64_t r = i / n_kb;
    const uint32_t b = (uint32_t)(i % n_kb);
    const uint64_t src = i * 16u;
    const uint64_t d = r * (uint64_t)(((n_kb + 3u) >> 2) * 128u) +
                       (uint64_t)(b >> 2) * 128u + (b & 3u) * 16u;
    *(uint4*)(dst + d) = *(const uint4*)(gate + src);
    *(uint4*)(dst + d + 64u) = *(const uint4*)(up + src);
}

PD_EXPORT
int pd_mxfp4_gu_interleave(const void* gate, const void* up, void* dst,
                           uint32_t n_kb, uint64_t rows, void* stream) {
    if (rows == 0 || n_kb == 0) return 0;
    const uint64_t total = rows * n_kb;
    const uint32_t blocks = (uint32_t)((total + 255u) / 256u);
    pd_mxfp4_gu_interleave_kernel<<<blocks, 256, 0, (cudaStream_t)stream>>>(
        (const unsigned char*)gate, (const unsigned char*)up, (unsigned char*)dst,
        n_kb, rows);
    return pd_launch_status();
}

PD_EXPORT
int pd_mxfp4_moe_down_grouped(const void* down_data, const void* down_scale, const void* down_bias,
                              const void* slot_of, const void* topk_w, const void* fused,
                              void* residual, uint32_t ff, uint32_t embd, uint32_t n_expert,
                              uint32_t n_active, uint32_t batch, void* stream) {
    if (embd == 0 || n_expert == 0 || batch == 0) return 0;
    uint32_t threads = 256, nwarps = (threads + 31u) >> 5, n_blocks = ff >> 5;
    size_t shmem = ((size_t)16 + ff + n_blocks + nwarps) * sizeof(float);
    dim3 grid(embd, n_expert);
    pd_mxfp4_moe_down_grouped_kernel<<<grid, threads, shmem, (cudaStream_t)stream>>>(
        (const unsigned char*)down_data, (const unsigned char*)down_scale, (const float*)down_bias,
        (const unsigned char*)slot_of, (const float*)topk_w, (const float*)fused, (float*)residual,
        ff, embd, n_expert, n_active, batch);
    return pd_launch_status();
}

PD_EXPORT
int pd_mxfp4_moe_gate_up_batch(const void* gate_W, const void* gate_bias, const void* up_W,
                               const void* up_bias, const void* idx, const void* x, void* out,
                               uint32_t in_dim, uint32_t ff, uint32_t n_active, uint32_t batch,
                               float alpha, float limit, void* stream) {
    if (ff == 0 || n_active == 0 || batch == 0) return 0;
    dim3 grid(ff, n_active, batch);
    pd_mxfp4_moe_gate_up_batch_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(
        (const unsigned char*)gate_W, (const float*)gate_bias, (const unsigned char*)up_W,
        (const float*)up_bias, (const unsigned int*)idx, (const float*)x, (float*)out,
        in_dim, ff, n_active, alpha, limit);
    return pd_launch_status();
}

PD_EXPORT
int pd_mxfp4_moe_down_batch(const void* down_W, const void* down_bias, const void* idx,
                            const void* topk_w, const void* fused, void* residual,
                            uint32_t ff, uint32_t embd, uint32_t n_active, uint32_t batch,
                            void* stream) {
    if (embd == 0 || n_active == 0 || batch == 0) return 0;
    dim3 grid(embd, batch);
    pd_mxfp4_moe_down_batch_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(
        (const unsigned char*)down_W, (const float*)down_bias, (const unsigned int*)idx,
        (const float*)topk_w, (const float*)fused, (float*)residual, ff, embd, n_active);
    return pd_launch_status();
}

PD_EXPORT
int pd_scale_add_dev(void* x, const void* y, const void* w, uint32_t slot, uint32_t n, void* stream) {
    uint32_t threads = 256;
    uint32_t blocks = (n + threads - 1) / threads;
    pd_scale_add_dev_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        (float*)x, (const float*)y, (const float*)w, slot, n);
    return pd_launch_status();
}

PD_EXPORT
int pd_attn_decode_f32(const void* q, const void* kc, const void* vc, const void* sinks,
                       void* out, uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
                       uint32_t first_pos, uint32_t n_pos, uint32_t kv_dim, float scale, void* stream) {
    if (n_pos == 0 || n_heads == 0) return 0;
    pd_attn_decode_kernel<<<n_heads, head_dim, 0, (cudaStream_t)stream>>>(
        (const float*)q, (const __half*)kc, (const __half*)vc, (const float*)sinks,
        (float*)out, n_heads, n_kv_heads, head_dim, first_pos, n_pos, kv_dim, scale);
    return pd_launch_status();
}

PD_EXPORT
int pd_rmsnorm_batch(const void* x, const void* w, void* out, uint32_t n, float eps,
                     uint32_t batch, void* stream) {
    if (n == 0 || batch == 0) return 0;
    // Width by batch: 1024-wide blocks can't co-reside on a 1536-thread SM
    // (1 block/SM, no latency overlap) - at prefill row counts that ran the
    // norm ~15x over its bandwidth floor (59.8us avg, 578 max vs
    // ~35 floor, 7.1% of the window). 256-wide at batch>=64 gets 6 blocks/SM.
    // Decode lanes (<64 rows) keep the original 1024-wide walk BIT-EXACT;
    // the >=64 prefill lanes change reduction grouping = the sanctioned
    // near-tie class (rung-10 precedent, realign gates arbitrate).
    // B200 bring-up: the 1536-thread-SM premise above is an sm_120 property
    // (this die has 2048), and an honest NO-PDL capture puts the decode
    // norm at 4.24 us against ~3.5 for a fused equivalent. PADDOCK_NORM_NTH
    // overrides the decode width so the election can be re-measured per die.
    const uint32_t nth = batch >= 64u ? pd_norm_wide_nth_ws(batch) : pd_norm_decode_nth();
    const int accm = pd_norm_acc_mode();
    if (accm == PD_ACC_DF) {
        pd_pdl_go(pd_rmsnorm_batch_kernel_t<PD_ACC_DF>, batch, nth, 0u, (cudaStream_t)stream,
            (const float*)x, (const float*)w, (float*)out, n, eps);
    } else if (accm == PD_ACC_F64) {
        pd_pdl_go(pd_rmsnorm_batch_kernel_t<PD_ACC_F64>, batch, nth, 0u, (cudaStream_t)stream,
            (const float*)x, (const float*)w, (float*)out, n, eps);
    } else {
        pd_pdl_go(pd_rmsnorm_batch_kernel_t<PD_ACC_F32>, batch, nth, 0u, (cudaStream_t)stream,
            (const float*)x, (const float*)w, (float*)out, n, eps);
    }
    return pd_launch_status();
}

PD_EXPORT
int pd_rope_yarn_batch(void* x, const void* positions, uint32_t n_heads, uint32_t head_dim,
                       float theta_scale, float freq_scale, float corr_low, float corr_high,
                       float ext_factor, float mscale, uint32_t batch, void* stream) {
    if (n_heads == 0 || batch == 0) return 0;
    uint32_t total = batch * n_heads, threads = 256; // 8 warps = 8 heads per block
    uint32_t blocks = (total * 32 + threads - 1) / threads;
    pd_rope_yarn_batch_kernel<true><<<blocks, threads, 0, (cudaStream_t)stream>>>(
        (float*)x, (const unsigned int*)positions, n_heads, head_dim, theta_scale, freq_scale,
        corr_low, corr_high, ext_factor, mscale, batch);
    return pd_launch_status();
}

// NORM-convention twin of pd_rope_yarn_batch (llama.cpp ROPE_TYPE_NORM):
// interleaved (2k, 2k+1) pairs instead of half-split. Same signature, same
// theta chain - granite and the llama-arch lineage rope this way. A separate
// entry point rather than a mode argument so the NEOX instantiation keeps its
// exact SASS, matching how llama.cpp ships rope_norm and rope_neox separately.
PD_EXPORT
int pd_rope_yarn_batch_norm(void* x, const void* positions, uint32_t n_heads, uint32_t head_dim,
                            float theta_scale, float freq_scale, float corr_low, float corr_high,
                            float ext_factor, float mscale, uint32_t batch, void* stream) {
    if (n_heads == 0 || batch == 0) return 0;
    uint32_t total = batch * n_heads, threads = 256;
    uint32_t blocks = (total * 32 + threads - 1) / threads;
    pd_rope_yarn_batch_kernel<false><<<blocks, threads, 0, (cudaStream_t)stream>>>(
        (float*)x, (const unsigned int*)positions, n_heads, head_dim, theta_scale, freq_scale,
        corr_low, corr_high, ext_factor, mscale, batch);
    return pd_launch_status();
}

__global__ void pd_softcap_kernel(float* __restrict__ x, uint32_t n, float cap) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = cap * tanhf(x[i] / cap);
}

// Final-logit softcapping (gemma4: 30*tanh(l/30)) applied in PLACE on the
// device logits plane so device sampling sees the capped distribution
// (monotonic - greedy argmax is unchanged; categorical needs the values).
PD_EXPORT
int pd_softcap(void* x, uint32_t n, float cap, void* stream) {
    if (n == 0) return 0;
    uint32_t threads = 256;
    uint32_t blocks = (n + threads - 1) / threads;
    pd_softcap_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>((float*)x, n, cap);
    return pd_launch_status();
}

__global__ void pd_add_scale_kernel(float* __restrict__ x, const float* __restrict__ y,
                                    float s, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = (x[i] + y[i]) * s;
}

// Residual add + whole-stream scale in one pass: x = (x + y) * s - gemma4's
// layer tail (FFN residual + layer_output_scale) collapses 4 launches
// (add, swap-side memset, scale_add) into 1.
PD_EXPORT
int pd_add_scale(void* x, const void* y, float s, uint32_t n, void* stream) {
    if (n == 0) return 0;
    uint32_t threads = 256;
    uint32_t blocks = (n + threads - 1) / threads;
    pd_add_scale_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        (float*)x, (const float*)y, s, n);
    return pd_launch_status();
}

// pd_geglu over the CONCATENATED gate|up row layout ([rows, 2*ff] from the
// fused gate|up GEMV): gate half updates in place, exactly pd_geglu math.
// ACT picks the arch's gate nonlinearity (see pd_glu_act in abi.cuh) - the
// GELU instantiation is byte-for-byte the kernel that shipped before.
template <int ACT>
__global__ void pd_glu_pair_kernel(float* __restrict__ x, uint32_t ff, uint32_t rows) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ff * rows) return;
    uint32_t r = i / ff, c = i % ff;
    float* row = x + (size_t)r * 2u * ff;
    row[c] = pd_glu_act<ACT>(row[c]) * row[ff + c];
}

template <int ACT>
static inline int pd_glu_pair_launch(void* x, uint32_t ff, uint32_t rows, void* stream) {
    if (ff == 0 || rows == 0) return 0;
    uint32_t threads = 256;
    uint32_t blocks = (ff * rows + threads - 1) / threads;
    pd_glu_pair_kernel<ACT><<<blocks, threads, 0, (cudaStream_t)stream>>>(
        (float*)x, ff, rows);
    return pd_launch_status();
}

PD_EXPORT
int pd_geglu_pair(void* x, uint32_t ff, uint32_t rows, void* stream) {
    return pd_glu_pair_launch<PD_ACT_GELU>(x, ff, rows, stream);
}

PD_EXPORT
int pd_swiglu_pair(void* x, uint32_t ff, uint32_t rows, void* stream) {
    return pd_glu_pair_launch<PD_ACT_SILU>(x, ff, rows, stream);
}

// Post-norm + residual + stream-scale in one pass, per row:
// x[row] = (x[row] + rmsnorm(proj[row])·w) · s. Gemma4 uses it on both layer
// halves (attention side with s=1, FFN side with s=layer_output_scale) -
// replaces rmsnorm_batch + add(_scale), two launches per half.
template <typename TP = float>
__global__ void pd_rmsnorm_add_scale_kernel(float* __restrict__ x,
                                            const TP* __restrict__ proj,
                                            const float* __restrict__ w, uint32_t n,
                                            float eps, float s) {
    const uint32_t b = blockIdx.x;
    const TP* pb = proj + (size_t)b * n;
    float* xb = x + (size_t)b * n;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    // width-stable: f64 sumsq (f32 products) - see pd_norm_wide_nth_ws.
    __shared__ double wsum[32];
    __shared__ float s_inv;
    double acc = 0.0;
    for (uint32_t i = tid; i < n; i += nth) {
        float v = (float)pb[i];
        acc += v * v;
    }
    for (uint32_t o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if ((tid & 31u) == 0) wsum[tid >> 5] = acc;
    __syncthreads();
    if (tid == 0) {
        double sum = 0.0;
        for (uint32_t wi = 0; wi < (nth + 31u) >> 5; ++wi) sum += wsum[wi];
        s_inv = rsqrtf((float)(sum / (double)n) + eps);
    }
    __syncthreads();
    const float inv = s_inv;
    for (uint32_t i = tid; i < n; i += nth) {
        xb[i] = (xb[i] + (float)pb[i] * inv * w[i]) * s;
    }
}

// Chunked twin: grid (rows, C). The verify band (rows 65..148)
// is this kernel's worst case - the rows>=64 election drops to 256 threads
// while the grid still leaves a third of the die idle, and each row pays
// two dependent phases at 1 CTA. Every block recomputes the full row sumsq
// with the identical thread count and walk order (bit-identical inv), then
// writes only its 1/C slice - the row1pc recipe. proj re-reads are L2-hot.
template <typename TP = float>
__global__ void pd_rmsnorm_add_scale_c_kernel(float* __restrict__ x,
                                              const TP* __restrict__ proj,
                                              const float* __restrict__ w, uint32_t n,
                                              float eps, float s) {
    const uint32_t b = blockIdx.x, ch = blockIdx.y, C = gridDim.y;
    const TP* pb = proj + (size_t)b * n;
    float* xb = x + (size_t)b * n;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    // width-stable: f64 sumsq (f32 products) - see pd_norm_wide_nth_ws.
    __shared__ double wsum[32];
    __shared__ float s_inv;
    double acc = 0.0;
    for (uint32_t i = tid; i < n; i += nth) {
        float v = (float)pb[i];
        acc += v * v;
    }
    for (uint32_t o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if ((tid & 31u) == 0) wsum[tid >> 5] = acc;
    __syncthreads();
    if (tid == 0) {
        double sum = 0.0;
        for (uint32_t wi = 0; wi < (nth + 31u) >> 5; ++wi) sum += wsum[wi];
        s_inv = rsqrtf((float)(sum / (double)n) + eps);
    }
    __syncthreads();
    const float inv = s_inv;
    const uint32_t per = (n + C - 1u) / C;
    const uint32_t i0 = ch * per, i1 = min(n, i0 + per);
    for (uint32_t i = i0 + tid; i < i1; i += nth) {
        xb[i] = (xb[i] + (float)pb[i] * inv * w[i]) * s;
    }
}

// float4 twin (f32-proj only): the muse c32 wide prefill pass runs
// the scalar kernel over [5984 x 6656] rows at ~4.3 TB/s effective; 16B
// transactions close that band. The per-thread sumsq accumulation partition
// changes (4 consecutive elements per step instead of nth-strided singles),
// so inv is a NUMERICS-CLASS change - elected only at rows >= 256 below so
// every decode/verify launch keeps the classic kernel bit-for-bit.
__global__ void pd_rmsnorm_add_scale_v4f_kernel(float* __restrict__ x,
                                                const float* __restrict__ proj,
                                                const float* __restrict__ w,
                                                uint32_t n4, float eps, float s) {
    const uint32_t b = blockIdx.x;
    const float4* pb = (const float4*)(proj + (size_t)b * n4 * 4u);
    float4* xb = (float4*)(x + (size_t)b * n4 * 4u);
    const float4* w4 = (const float4*)w;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    // width-stable: f64 sumsq (f32 products) - see pd_norm_wide_nth_ws.
    __shared__ double wsum[32];
    __shared__ float s_inv;
    double acc = 0.0;
    for (uint32_t i = tid; i < n4; i += nth) {
        const float4 v = pb[i];
        acc += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    for (uint32_t o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if ((tid & 31u) == 0) wsum[tid >> 5] = acc;
    __syncthreads();
    if (tid == 0) {
        double sum = 0.0;
        for (uint32_t wi = 0; wi < (nth + 31u) >> 5; ++wi) sum += wsum[wi];
        s_inv = rsqrtf((float)(sum / (double)(n4 * 4u)) + eps);
    }
    __syncthreads();
    const float inv = s_inv;
    for (uint32_t i = tid; i < n4; i += nth) {
        float4 xv = xb[i];
        const float4 pv = pb[i];
        const float4 wv = w4[i];
        xv.x = (xv.x + pv.x * inv * wv.x) * s;
        xv.y = (xv.y + pv.y * inv * wv.y) * s;
        xv.z = (xv.z + pv.z * inv * wv.z) * s;
        xv.w = (xv.w + pv.w * inv * wv.w) * s;
        xb[i] = xv;
    }
}

// Chunk election: off unless PADDOCK_RMSAS_C is set (probe gate). C sized so
// rows*C covers ~2 waves of the die, capped at 8.
static inline uint32_t pd_rmsas_chunks(uint32_t rows) {
    static int en = -1;
    if (en < 0) {
        const char* e = pd_env("PADDOCK_RMSAS_C");
        en = e ? atoi(e) : 0;
    }
    if (en <= 0 || rows < 64u) return 1u;
    static int nsm = 0;
    if (nsm == 0) {
        int d = 0;
        cudaGetDevice(&d);
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, d);
        if (nsm <= 0) nsm = 148;
    }
    if (rows >= (uint32_t)nsm * 2u) return 1u;
    uint32_t c = en > 1 ? (uint32_t)en : ((uint32_t)nsm * 2u + rows - 1u) / rows;
    return c > 8u ? 8u : (c < 1u ? 1u : c);
}

PD_EXPORT
int pd_rmsnorm_add_scale(void* x, const void* proj, const void* w, uint32_t n, float eps,
                         float s, uint32_t rows, void* stream) {
    if (n == 0 || rows == 0) return 0;
    // same width-by-rows occupancy fix as pd_rmsnorm_batch (see its note)
    const uint32_t nth = rows >= 64u ? pd_norm_wide_nth_ws(rows) : 1024u;
    // wide-band vec4 election: chunk passes only - decode and
    // verify launches keep the classic kernel (numerics class, see the twin)
    if (rows >= 256u && (n & 3u) == 0
        && (((uintptr_t)x | (uintptr_t)proj | (uintptr_t)w) & 15u) == 0) {
        pd_rmsnorm_add_scale_v4f_kernel<<<rows, nth, 0, (cudaStream_t)stream>>>(
            (float*)x, (const float*)proj, (const float*)w, n >> 2, eps, s);
        return pd_launch_status();
    }
    const uint32_t C = pd_rmsas_chunks(rows);
    if (C > 1u)
        pd_rmsnorm_add_scale_c_kernel<float><<<dim3(rows, C), nth, 0, (cudaStream_t)stream>>>(
            (float*)x, (const float*)proj, (const float*)w, n, eps, s);
    else
        pd_rmsnorm_add_scale_kernel<float><<<rows, nth, 0, (cudaStream_t)stream>>>(
            (float*)x, (const float*)proj, (const float*)w, n, eps, s);
    return pd_launch_status();
}

// p16 twin: `proj` bytes are bf16 (the o16 GEMM epilogue's
// stream). Appended per the ABI growth rule.
PD_EXPORT
int pd_rmsnorm_add_scale2(void* x, const void* proj, const void* w, uint32_t n,
                          float eps, float s, uint32_t rows, uint32_t p16,
                          void* stream) {
    if (n == 0 || rows == 0) return 0;
    const uint32_t nth = rows >= 64u ? pd_norm_wide_nth_ws(rows) : 1024u;
    // same wide-band vec4 election as pd_rmsnorm_add_scale
    if (!p16 && rows >= 256u && (n & 3u) == 0
        && (((uintptr_t)x | (uintptr_t)proj | (uintptr_t)w) & 15u) == 0) {
        pd_rmsnorm_add_scale_v4f_kernel<<<rows, nth, 0, (cudaStream_t)stream>>>(
            (float*)x, (const float*)proj, (const float*)w, n >> 2, eps, s);
        return pd_launch_status();
    }
    const uint32_t C = pd_rmsas_chunks(rows);
    if (p16) {
        if (C > 1u)
            pd_rmsnorm_add_scale_c_kernel<__nv_bfloat16><<<dim3(rows, C), nth, 0, (cudaStream_t)stream>>>(
                (float*)x, (const __nv_bfloat16*)proj, (const float*)w, n, eps, s);
        else
            pd_rmsnorm_add_scale_kernel<__nv_bfloat16><<<rows, nth, 0, (cudaStream_t)stream>>>(
                (float*)x, (const __nv_bfloat16*)proj, (const float*)w, n, eps, s);
    } else {
        if (C > 1u)
            pd_rmsnorm_add_scale_c_kernel<float><<<dim3(rows, C), nth, 0, (cudaStream_t)stream>>>(
                (float*)x, (const float*)proj, (const float*)w, n, eps, s);
        else
            pd_rmsnorm_add_scale_kernel<float><<<rows, nth, 0, (cudaStream_t)stream>>>(
                (float*)x, (const float*)proj, (const float*)w, n, eps, s);
    }
    return pd_launch_status();
}

// Gemma4 fused QKV epilogue over the CONCATENATED [q|k|v] GEMV output row:
// per-head RMS norms (q,k learned; V weightless unless `vnorm` says
// otherwise), rope with optional per-pair factors on q and k, then K and V
// append into the (dense or ring-paged) f16 caches. Replaces 7 launches per
// layer per step. One block per (row, head-slot): head-slots 0..n_head = q,
// then n_kv k, then n_kv v.
// Rope math mirrors pd_rope_factors_batch exactly (theta chain per lane).
//
// The last three arguments are the ARCHITECTURE constants this family used to
// assume: `freq_scale` (1 on every roped layer served before
// muse-glimmer; 0 on muse-glimmer's NoPE full-attention layers, where it makes
// the rotation a bit-exact identity), `neox` (half-split (k, k+half) pairs vs
// ROPE_TYPE_NORM's interleaved (2k, 2k+1)), and `vnorm` (gemma4 RMS-norms V
// weightlessly, muse-glimmer leaves V alone). The yarn ramp is deliberately
// absent here as it always was - every consumer of this kernel runs
// ext_factor 0 / mscale 1, so the angle reduces to freq_scale * theta.
template<bool PAGED, typename KVW = __half, typename XT = float>
__global__ void pd_gemma_qkv_nra_kernel(
    XT* __restrict__ qp, XT* __restrict__ kp, XT* __restrict__ vp,
    const float* __restrict__ wq_norm,
    const float* __restrict__ wk_norm, float* __restrict__ q_out,
    uint8_t* __restrict__ kc, uint8_t* __restrict__ vc,
    const unsigned int* __restrict__ positions, const unsigned int* __restrict__ slots,
    const float* __restrict__ factors, const uint32_t* __restrict__ block_tables,
    uint32_t bps, uint32_t n_head, uint32_t n_kv, uint32_t head_dim, uint32_t max_ctx,
    float eps, float theta_scale, uint32_t qkv_stride = 0u,
    float freq_scale = 1.0f, bool neox = true, bool vnorm = true,
    uint8_t* __restrict__ vdim = nullptr) {
    PD_PDL_ARM();
    const uint32_t b = blockIdx.y, hs = blockIdx.x;
    const uint32_t d = threadIdx.x, nth = blockDim.x;
    const uint32_t half = head_dim / 2;
    const uint32_t q_dim = n_head * head_dim, kv_dim = n_kv * head_dim;
    // qkv_stride: nonzero when q/k/v are one CONCATENATED [r][stride] GEMM
    // output (qkv-fused plane) - all three pointers then share the row
    // stride; 0 keeps the classic dense per-plane strides
    const uint32_t qs = qkv_stride ? qkv_stride : q_dim;
    const uint32_t ks = qkv_stride ? qkv_stride : kv_dim;
    const uint32_t kind = hs < n_head ? 0u : (hs < n_head + n_kv ? 1u : 2u);
    const uint32_t h = kind == 0 ? hs : (kind == 1 ? hs - n_head : hs - n_head - n_kv);
    XT* src = kind == 0 ? qp + (size_t)b * qs + (size_t)h * head_dim
            : kind == 1 ? kp + (size_t)b * ks + (size_t)h * head_dim
                        : vp + (size_t)b * ks + (size_t)h * head_dim;

    // per-head RMS norm (block reduction)
    extern __shared__ float sh[];  // [head_dim] staged values + reduce tail
    __shared__ float s_inv;
    float acc = 0.0f;
    for (uint32_t i = d; i < head_dim; i += nth) {
        float v = (float)src[i];
        sh[i] = v;
        acc += v * v;
    }
    for (uint32_t o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    __shared__ float warp_sum[32];
    if ((d & 31u) == 0) warp_sum[d >> 5] = acc;
    __syncthreads();
    if (d == 0) {
        float s = 0.0f;
        for (uint32_t w = 0; w < (nth + 31u) >> 5; ++w) s += warp_sum[w];
        s_inv = rsqrtf(s / (float)head_dim + eps);
    }
    __syncthreads();
    const float inv = s_inv;
    // normed value: q/k learned weight, v weightless - or, where the arch
    // does not norm V at all, V passes through exactly as staged
    if (kind != 2u || vnorm) {
        for (uint32_t i = d; i < head_dim; i += nth) {
            float w = kind == 0 ? wq_norm[i] : (kind == 1 ? wk_norm[i] : 1.0f);
            sh[i] = sh[i] * inv * w;
        }
    }
    __syncthreads();

    // rope on q/k, theta chain per pd_rope_factors. `kind` is block-uniform
    // (it comes off blockIdx.x), so the __syncthreads below is whole-block.
    if (kind != 2u) {
        const float pos = (float)positions[b];
        for (uint32_t k = d; k < half; k += nth) {
            float theta = pos;
            for (uint32_t i = 0; i < k; ++i) theta *= theta_scale;
            float t = factors ? theta / factors[k] : theta;
            const float ang = freq_scale * t;
            float sn = sinf(ang), cs = cosf(ang);
            // NEOX pairs (k, k+half); NORM pairs (2k, 2k+1). Either way each
            // thread owns both members of its own pair, so the in-place
            // rotation over shared memory stays race-free.
            const uint32_t i0 = neox ? k : 2u * k;
            const uint32_t i1 = neox ? k + half : 2u * k + 1u;
            float a = sh[i0], bb = sh[i1];
            sh[i0] = a * cs - bb * sn;
            sh[i1] = a * sn + bb * cs;
        }
        __syncthreads();
    }

    if (kind == 0u) {
        float* dst = q_out + (size_t)b * q_dim + (size_t)h * head_dim;
        for (uint32_t i = d; i < head_dim; i += nth) dst[i] = sh[i];
        return;
    }
    // append into the f16/fp8 cache (dense slot plane or ring block pool)
    const uint32_t slot = slots ? slots[b] : b;
    const uint32_t pos = positions[b];
    KVW* cache = (KVW*)(kind == 1u ? kc : vc);
    size_t base;
    uint32_t blk = 0u;
    if (PAGED) {
        blk = block_tables[(size_t)slot * bps + pos / 16u];
        base = ((size_t)blk * 16u + (pos & 15u)) * kv_dim + (size_t)h * head_dim;
    } else {
        base = ((size_t)slot * max_ctx + pos) * kv_dim + (size_t)h * head_dim;
    }
    //  writer-fused double-store: the dim-major twin gets the same
    // converted byte at append time (fp8 paged only; the value is already in
    // registers, so this is one extra store, no extra pass)
    if (PAGED && sizeof(KVW) == 1 && kind == 2u && vdim) {
        uint8_t* vd = vdim + ((size_t)blk * kv_dim + (size_t)h * head_dim) * 16u
                    + (pos & 15u);
        for (uint32_t i = d; i < head_dim; i += nth) {
            const KVW cv = (KVW)sh[i];
            cache[base + i] = cv;
            vd[(size_t)i * 16u] = *(const uint8_t*)&cv;
        }
    } else {
        for (uint32_t i = d; i < head_dim; i += nth) cache[base + i] = (KVW)sh[i];
    }
}

// One launcher behind all four exports. The three older entries pin the
// architecture constants this family assumed before muse-glimmer
// (freq_scale 1, NEOX pairing, V normed) so their SASS and their behaviour
// are unchanged; nra3 exposes them.
static int pd_gemma_qkv_nra_impl(void* qp, void* kp, void* vp, const void* wq_norm,
                                 const void* wk_norm, void* q_out, void* kc, void* vc,
                                 const void* positions, const void* slots,
                                 const void* factors, const void* block_tables,
                                 uint32_t bps, uint32_t n_head, uint32_t n_kv,
                                 uint32_t head_dim, uint32_t max_ctx, uint32_t batch,
                                 float eps, float theta_scale, uint32_t kv_dtype,
                                 uint32_t qkv_stride, float freq_scale, bool neox,
                                 bool vnorm, bool x_b16, void* stream) {
    if (batch == 0 || n_head == 0) return 0;
    dim3 grid(n_head + 2u * n_kv, batch);
    uint32_t threads = head_dim >= 256u ? 256u : head_dim;
    uint32_t smem = (head_dim + 32u) * sizeof(float);
    // fp8-cache arm (KV8): the kernel is KVW-generic and qkv_stride
    // is plain pointer math - the old cudaErrorInvalidValue here was
    // unplumbed launcher code, and it is what forced qkvfuse off under KV8
    #define PD_GQNRA_GO(PAGED_, KVW_, XT_)                                     \
        pd_pdl_go(pd_gemma_qkv_nra_kernel<PAGED_, KVW_, XT_>, grid, threads,   \
                  smem, (cudaStream_t)stream,                                  \
                  (XT_*)qp, (XT_*)kp, (XT_*)vp, (const float*)wq_norm,         \
                  (const float*)wk_norm, (float*)q_out, (uint8_t*)kc,          \
                  (uint8_t*)vc, (const unsigned int*)positions,                \
                  (const unsigned int*)slots, (const float*)factors,           \
                  (const uint32_t*)block_tables, bps, n_head, n_kv, head_dim,  \
                  max_ctx, eps, theta_scale, qkv_stride, freq_scale, neox,     \
                  vnorm, (uint8_t*)pd_vdim_base)
    if (x_b16) {
        if (kv_dtype == PD_KV_FP8_E4M3) {
            if (block_tables) PD_GQNRA_GO(true, __nv_fp8_e4m3, __nv_bfloat16);
            else PD_GQNRA_GO(false, __nv_fp8_e4m3, __nv_bfloat16);
        } else {
            if (block_tables) PD_GQNRA_GO(true, __half, __nv_bfloat16);
            else PD_GQNRA_GO(false, __half, __nv_bfloat16);
        }
    } else if (kv_dtype == PD_KV_FP8_E4M3) {
        if (block_tables) PD_GQNRA_GO(true, __nv_fp8_e4m3, float);
        else PD_GQNRA_GO(false, __nv_fp8_e4m3, float);
    } else {
        if (block_tables) PD_GQNRA_GO(true, __half, float);
        else PD_GQNRA_GO(false, __half, float);
    }
    #undef PD_GQNRA_GO
    return pd_launch_status();
}

PD_EXPORT
int pd_gemma_qkv_nra(void* qp, void* kp, void* vp, const void* wq_norm, const void* wk_norm,
                     void* q_out, void* kc, void* vc, const void* positions, const void* slots,
                     const void* factors, const void* block_tables, uint32_t bps,
                     uint32_t n_head, uint32_t n_kv, uint32_t head_dim, uint32_t max_ctx,
                     uint32_t batch, float eps, float theta_scale, void* stream) {
    return pd_gemma_qkv_nra_impl(qp, kp, vp, wq_norm, wk_norm, q_out, kc, vc, positions,
                                 slots, factors, block_tables, bps, n_head, n_kv, head_dim,
                                 max_ctx, batch, eps, theta_scale, 0u, 0u, 1.0f, true, true,
                                 false, stream);
}

// kv_dtype-aware twin of pd_gemma_qkv_nra (ABI append-only: the original
// keeps its f16-only signature). fp8 appends cast per element (e4m3 rn-sat),
// same epilogue math.
PD_EXPORT
int pd_gemma_qkv_nra2(void* qp, void* kp, void* vp, const void* wq_norm, const void* wk_norm,
                      void* q_out, void* kc, void* vc, const void* positions, const void* slots,
                      const void* factors, const void* block_tables, uint32_t bps,
                      uint32_t n_head, uint32_t n_kv, uint32_t head_dim, uint32_t max_ctx,
                      uint32_t batch, float eps, float theta_scale, uint32_t kv_dtype,
                      void* stream) {
    return pd_gemma_qkv_nra_impl(qp, kp, vp, wq_norm, wk_norm, q_out, kc, vc, positions,
                                 slots, factors, block_tables, bps, n_head, n_kv, head_dim,
                                 max_ctx, batch, eps, theta_scale, kv_dtype, 0u, 1.0f, true,
                                 true, false, stream);
}

// nra2 twin for the qkv-CONCAT GEMM layout (qkv-fusion -07-21):
// identical math, q/k/v pointers share one row stride.
PD_EXPORT
int pd_gemma_qkv_nra2s(void* qp, void* kp, void* vp, const void* wq_norm, const void* wk_norm,
                       void* q_out, void* kc, void* vc, const void* positions, const void* slots,
                       const void* factors, const void* block_tables, uint32_t bps,
                       uint32_t n_head, uint32_t n_kv, uint32_t head_dim, uint32_t max_ctx,
                       uint32_t batch, float eps, float theta_scale, uint32_t kv_dtype,
                       uint32_t qkv_stride, void* stream) {
    return pd_gemma_qkv_nra_impl(qp, kp, vp, wq_norm, wk_norm, q_out, kc, vc, positions,
                                 slots, factors, block_tables, bps, n_head, n_kv, head_dim,
                                 max_ctx, batch, eps, theta_scale, kv_dtype, qkv_stride,
                                 1.0f, true, true, false, stream);
}

// arch-constant superset. Everything nra2s takes, plus the three
// things the epilogue used to hardcode: `freq_scale` (muse-glimmer's
// full-attention layers are NoPE - freq_scale 0 makes this a bit-exact
// identity rotation, and its ABSENCE here is what kept re-roping them on every
// decode step while prefill correctly left them alone), `neox` (pair layout),
// and `vnorm` (whether V is RMS-normed at all).
PD_EXPORT
int pd_gemma_qkv_nra3(void* qp, void* kp, void* vp, const void* wq_norm, const void* wk_norm,
                      void* q_out, void* kc, void* vc, const void* positions, const void* slots,
                      const void* factors, const void* block_tables, uint32_t bps,
                      uint32_t n_head, uint32_t n_kv, uint32_t head_dim, uint32_t max_ctx,
                      uint32_t batch, float eps, float theta_scale, uint32_t kv_dtype,
                      uint32_t qkv_stride, float freq_scale, uint32_t neox, uint32_t vnorm,
                      void* stream) {
    return pd_gemma_qkv_nra_impl(qp, kp, vp, wq_norm, wk_norm, q_out, kc, vc, positions,
                                 slots, factors, block_tables, bps, n_head, n_kv, head_dim,
                                 max_ctx, batch, eps, theta_scale, kv_dtype, qkv_stride,
                                 freq_scale, neox != 0u, vnorm != 0u, false, stream);
}

// packed-bf16 q/k/v read twin of nra3 (spec verify b16-D
// election): the GEMM planes hold bf16 at f32 element indexing and half the
// bytes (p16 convention). q_out stays f32 and the KV appends are unchanged -
// only the staging loads differ. Plane byte offsets are the caller's to halve.
PD_EXPORT
int pd_gemma_qkv_nra3_b16(void* qp, void* kp, void* vp, const void* wq_norm,
                          const void* wk_norm, void* q_out, void* kc, void* vc,
                          const void* positions, const void* slots,
                          const void* factors, const void* block_tables, uint32_t bps,
                          uint32_t n_head, uint32_t n_kv, uint32_t head_dim,
                          uint32_t max_ctx, uint32_t batch, float eps, float theta_scale,
                          uint32_t kv_dtype, uint32_t qkv_stride, float freq_scale,
                          uint32_t neox, uint32_t vnorm, void* stream) {
    return pd_gemma_qkv_nra_impl(qp, kp, vp, wq_norm, wk_norm, q_out, kc, vc, positions,
                                 slots, factors, block_tables, bps, n_head, n_kv, head_dim,
                                 max_ctx, batch, eps, theta_scale, kv_dtype, qkv_stride,
                                 freq_scale, neox != 0u, vnorm != 0u, true, stream);
}

PD_EXPORT
int pd_geglu(void* gate, const void* up, uint32_t n, void* stream) {
    if (n == 0) return 0;
    uint32_t threads = 256;
    uint32_t blocks = (n + threads - 1) / threads;
    pd_geglu_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        (float*)gate, (const float*)up, n);
    return pd_launch_status();
}

PD_EXPORT
int pd_rope_factors_batch(void* x, const void* positions, const void* factors,
                          uint32_t n_heads, uint32_t head_dim, float theta_scale,
                          float freq_scale, float corr_low, float corr_high,
                          float ext_factor, float mscale, uint32_t batch, void* stream) {
    if (n_heads == 0 || batch == 0) return 0;
    uint32_t total = batch * n_heads, threads = 256; // 8 warps = 8 heads per block
    uint32_t blocks = (total * 32 + threads - 1) / threads;
    pd_rope_factors_batch_kernel<true><<<blocks, threads, 0, (cudaStream_t)stream>>>(
        (float*)x, (const unsigned int*)positions, (const float*)factors, n_heads, head_dim,
        theta_scale, freq_scale, corr_low, corr_high, ext_factor, mscale, batch);
    return pd_launch_status();
}

// NORM-convention twin (llama.cpp ROPE_TYPE_NORM): interleaved (2k, 2k+1)
// pairs instead of half-split, same signature and the same theta chain -
// exactly the pd_rope_yarn_batch / _norm split, extended to the factors
// carrier because muse-glimmer ropes NORM while gemma4, which shares its
// graph in this engine, ropes NEOX.
PD_EXPORT
int pd_rope_factors_batch_norm(void* x, const void* positions, const void* factors,
                               uint32_t n_heads, uint32_t head_dim, float theta_scale,
                               float freq_scale, float corr_low, float corr_high,
                               float ext_factor, float mscale, uint32_t batch, void* stream) {
    if (n_heads == 0 || batch == 0) return 0;
    uint32_t total = batch * n_heads, threads = 256;
    uint32_t blocks = (total * 32 + threads - 1) / threads;
    pd_rope_factors_batch_kernel<false><<<blocks, threads, 0, (cudaStream_t)stream>>>(
        (float*)x, (const unsigned int*)positions, (const float*)factors, n_heads, head_dim,
        theta_scale, freq_scale, corr_low, corr_high, ext_factor, mscale, batch);
    return pd_launch_status();
}

PD_EXPORT
int pd_kv_append_batch(const void* kv, void* cache, const void* positions, const void* slots,
                       uint32_t kv_dim, uint32_t max_ctx, uint32_t batch, uint32_t kv_dtype,
                       void* stream) {
    if (kv_dim == 0 || batch == 0) return 0;
    uint32_t threads = 256;
    dim3 grid((kv_dim + threads - 1) / threads, batch);
    if (kv_dtype == PD_KV_FP8_E4M3)
        pd_kv_append_batch_kernel<__nv_fp8_e4m3><<<grid, threads, 0, (cudaStream_t)stream>>>(
            (const float*)kv, (__nv_fp8_e4m3*)cache, (const unsigned int*)positions,
            (const unsigned int*)slots, kv_dim, max_ctx, batch);
    else
        pd_kv_append_batch_kernel<__half><<<grid, threads, 0, (cudaStream_t)stream>>>(
            (const float*)kv, (__half*)cache, (const unsigned int*)positions,
            (const unsigned int*)slots, kv_dim, max_ctx, batch);
    return pd_launch_status();
}

PD_EXPORT
int pd_q_norm_rope(const void* x, const void* w, void* out, const void* positions,
                   uint32_t n_heads, uint32_t head_dim, float eps, float theta_scale,
                   float freq_scale, float corr_low, float corr_high, float ext_factor,
                   float mscale, uint32_t batch, void* stream) {
    if (batch == 0 || n_heads == 0) return 0;
    if (head_dim != 128u) return cudaErrorInvalidValue;
    const uint32_t total = batch * n_heads;
    PdRopeArgs rp = {theta_scale, freq_scale, corr_low, corr_high, ext_factor, mscale};
    pd_q_norm_rope_kernel<<<(total + 7u) / 8u, 256, 0, (cudaStream_t)stream>>>(
        (const float*)x, (const float*)w, (float*)out,
        (const unsigned int*)positions, n_heads, eps, rp, total);
    return pd_launch_status();
}

PD_EXPORT
int pd_k_norm_rope_append(const void* x, const void* w, void* cache, const void* positions,
                          const void* slots, uint32_t n_kv_heads, uint32_t head_dim,
                          uint32_t max_ctx, float eps, float theta_scale, float freq_scale,
                          float corr_low, float corr_high, float ext_factor, float mscale,
                          uint32_t batch, uint32_t kv_dtype, void* stream) {
    if (batch == 0 || n_kv_heads == 0) return 0;
    if (head_dim != 128u) return cudaErrorInvalidValue;
    const uint32_t total = batch * n_kv_heads;
    PdRopeArgs rp = {theta_scale, freq_scale, corr_low, corr_high, ext_factor, mscale};
    auto st = (cudaStream_t)stream;
    if (kv_dtype == PD_KV_FP8_E4M3) {
        pd_k_norm_rope_append_kernel<__nv_fp8_e4m3><<<(total + 7u) / 8u, 256, 0, st>>>(
            (const float*)x, (const float*)w, (__nv_fp8_e4m3*)cache,
            (const unsigned int*)positions, (const unsigned int*)slots,
            n_kv_heads, max_ctx, eps, rp, total);
    } else {
        pd_k_norm_rope_append_kernel<__half><<<(total + 7u) / 8u, 256, 0, st>>>(
            (const float*)x, (const float*)w, (__half*)cache,
            (const unsigned int*)positions, (const unsigned int*)slots,
            n_kv_heads, max_ctx, eps, rp, total);
    }
    return pd_launch_status();
}

// ---------------- fused-QKV consumer: norm+rope (q,k) + KV scatter (k,v)
// The wave-efficiency companion of the FUSED qkv GEMM: with q/k/v projected
// by one GEMM into a combined [batch, q_dim + 2*kv_dim] plane (three narrow
// GEMMs were 1.2-wave launches idling ~40% of the GPU at serving batches),
// this single kernel consumes the strided rows directly - q heads norm+rope
// into the packed attention input, k heads norm+rope+scatter into the K
// cache, v heads convert+scatter into the V cache. Replaces three separate
// launches (q_norm_rope, k_norm_rope_append, kv_append v) and the packed
// intermediate q/k/v planes. Bit-exact per head with the separate kernels
// (identical pd_norm_rope_head / pd_kv_store math).
template<typename KV>
__global__ void pd_qkv_norm_rope_append_kernel(
    const float* __restrict__ x, const float* __restrict__ wq,
    const float* __restrict__ wk, float* __restrict__ qn,
    KV* __restrict__ kcache, KV* __restrict__ vcache,
    const unsigned int* __restrict__ positions, const unsigned int* __restrict__ slots,
    uint32_t n_heads, uint32_t n_kv_heads, uint32_t max_ctx, float eps,
    PdRopeArgs rp, uint32_t total_tasks) {
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t idx = blockIdx.x * (blockDim.x >> 5) + warp;
    if (idx >= total_tasks) return;
    const uint32_t th = n_heads + 2u * n_kv_heads;  // tasks per row
    const uint32_t b = idx / th, t = idx % th;
    const uint32_t x_stride = th * 128u;
    const uint32_t kv_dim = n_kv_heads * 128u;
    const float* head = x + (size_t)b * x_stride + (size_t)t * 128u;
    if (t < n_heads) {
        // q: norm + rope -> packed attention input
        const float4 r = pd_norm_rope_head(head, wq, eps, positions[b], rp, lane);
        reinterpret_cast<float4*>(qn + ((size_t)b * n_heads + t) * 128u)[lane] = r;
        return;
    }
    const uint32_t slot = slots ? slots[b] : b;
    KV* base = (t < n_heads + n_kv_heads ? kcache : vcache) +
               (size_t)slot * max_ctx * kv_dim + (size_t)positions[b] * kv_dim +
               (size_t)((t - n_heads) % n_kv_heads) * 128u;
    if (t < n_heads + n_kv_heads) {
        // k: norm + rope -> cache scatter
        const float4 r = pd_norm_rope_head(head, wk, eps, positions[b], rp, lane);
        pd_kv_store(&base[4u * lane + 0u], r.x);
        pd_kv_store(&base[4u * lane + 1u], r.y);
        pd_kv_store(&base[4u * lane + 2u], r.z);
        pd_kv_store(&base[4u * lane + 3u], r.w);
    } else {
        // v: plain convert + scatter
        const float4 v = reinterpret_cast<const float4*>(head)[lane];
        pd_kv_store(&base[4u * lane + 0u], v.x);
        pd_kv_store(&base[4u * lane + 1u], v.y);
        pd_kv_store(&base[4u * lane + 2u], v.z);
        pd_kv_store(&base[4u * lane + 3u], v.w);
    }
}

PD_EXPORT
int pd_qkv_norm_rope_append(const void* x, const void* wq, const void* wk, void* qn,
                            void* kcache, void* vcache, const void* positions,
                            const void* slots, uint32_t n_heads, uint32_t n_kv_heads,
                            uint32_t head_dim, uint32_t max_ctx, float eps,
                            float theta_scale, float freq_scale, float corr_low,
                            float corr_high, float ext_factor, float mscale,
                            uint32_t batch, uint32_t kv_dtype, void* stream) {
    if (batch == 0 || n_heads == 0) return 0;
    if (head_dim != 128u) return cudaErrorInvalidValue;
    const uint32_t total = batch * (n_heads + 2u * n_kv_heads);
    PdRopeArgs rp = {theta_scale, freq_scale, corr_low, corr_high, ext_factor, mscale};
    auto st = (cudaStream_t)stream;
    if (kv_dtype == PD_KV_FP8_E4M3) {
        pd_qkv_norm_rope_append_kernel<__nv_fp8_e4m3><<<(total + 7u) / 8u, 256, 0, st>>>(
            (const float*)x, (const float*)wq, (const float*)wk, (float*)qn,
            (__nv_fp8_e4m3*)kcache, (__nv_fp8_e4m3*)vcache,
            (const unsigned int*)positions, (const unsigned int*)slots,
            n_heads, n_kv_heads, max_ctx, eps, rp, total);
    } else {
        pd_qkv_norm_rope_append_kernel<__half><<<(total + 7u) / 8u, 256, 0, st>>>(
            (const float*)x, (const float*)wq, (const float*)wk, (float*)qn,
            (__half*)kcache, (__half*)vcache,
            (const unsigned int*)positions, (const unsigned int*)slots,
            n_heads, n_kv_heads, max_ctx, eps, rp, total);
    }
    return pd_launch_status();
}

PD_EXPORT
int pd_convert_f32_f16(const void* src, void* dst, uint64_t n, void* stream) {
    if (n == 0) return 0;
    uint32_t threads = 256;
    uint64_t blocks = (n + threads - 1) / threads;
    pd_convert_f32_f16_kernel<<<(uint32_t)blocks, threads, 0, (cudaStream_t)stream>>>(
        (const float*)src, (__half*)dst, n);
    return pd_launch_status();
}

PD_EXPORT
int pd_attn_decode_batch(const void* q, const void* kc, const void* vc, const void* sinks,
                         void* out, const void* positions, const void* slots, uint32_t n_heads,
                         uint32_t n_kv_heads, uint32_t head_dim, uint32_t max_ctx, uint32_t kv_dim,
                         uint32_t swa_window, uint32_t batch, float scale, uint32_t kv_dtype,
                         void* stream) {
    if (n_heads == 0 || batch == 0) return 0;
    dim3 grid(n_heads, batch);
    // see pd_attn_decode_batch_partial for the block-size/carveout/smem rationale
    uint32_t attn_nth = head_dim > 256 ? head_dim : 256;
    uint32_t attn_smem = (uint32_t)PD_ATTN_TILE_SMEM(head_dim);
    static uint32_t smem_set_b = 0;
    if (smem_set_b == 0) {
        pd_prefer_max_shared(pd_attn_decode_batch_kernel<__nv_fp8_e4m3>);
        pd_prefer_max_shared(pd_attn_decode_batch_kernel<__half>);
        smem_set_b = 1;
    }
    if (attn_smem > 48u * 1024u && attn_smem > smem_set_b) {
        cudaFuncSetAttribute((const void*)pd_attn_decode_batch_kernel<__nv_fp8_e4m3>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, attn_smem);
        cudaFuncSetAttribute((const void*)pd_attn_decode_batch_kernel<__half>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, attn_smem);
        smem_set_b = attn_smem;
    }
    if (kv_dtype == PD_KV_FP8_E4M3)
        pd_attn_decode_batch_kernel<__nv_fp8_e4m3><<<grid, attn_nth, attn_smem, (cudaStream_t)stream>>>(
            (const float*)q, (const __nv_fp8_e4m3*)kc, (const __nv_fp8_e4m3*)vc, (const float*)sinks,
            (float*)out, (const unsigned int*)positions, (const unsigned int*)slots, n_heads,
            n_kv_heads, head_dim, max_ctx, kv_dim, swa_window, scale);
    else
        pd_attn_decode_batch_kernel<__half><<<grid, attn_nth, attn_smem, (cudaStream_t)stream>>>(
            (const float*)q, (const __half*)kc, (const __half*)vc, (const float*)sinks, (float*)out,
            (const unsigned int*)positions, (const unsigned int*)slots, n_heads, n_kv_heads, head_dim,
            max_ctx, kv_dim, swa_window, scale);
    return pd_launch_status();
}

// 536: pd_attn_decode_batch with the PARALLEL-SCORE walk. Identical signature,
// identical grid/carveout/smem; the only difference is that the per-key dot
// product is spread across all threads instead of n_t of them.
//
// Why it is a separate entry rather than a flip: the partial sums are
// interleaved rather than sequential, so it is a different summation order and
// therefore not bit-identical to what every other family's gates were stamped
// against. Callers opt in.
PD_EXPORT
int pd_attn_decode_batch_ps(const void* q, const void* kc, const void* vc, const void* sinks,
                            void* out, const void* positions, const void* slots, uint32_t n_heads,
                            uint32_t n_kv_heads, uint32_t head_dim, uint32_t max_ctx, uint32_t kv_dim,
                            uint32_t swa_window, uint32_t batch, float scale, uint32_t kv_dtype,
                            void* stream) {
    if (n_heads == 0 || batch == 0) return 0;
    dim3 grid(n_heads, batch);
    uint32_t attn_nth = head_dim > 256 ? head_dim : 256;
    uint32_t attn_smem = (uint32_t)PD_ATTN_TILE_SMEM(head_dim);
    static uint32_t smem_set_ps = 0;
    if (smem_set_ps == 0) {
        pd_prefer_max_shared(pd_attn_decode_batch_kernel<__nv_fp8_e4m3, true>);
        pd_prefer_max_shared(pd_attn_decode_batch_kernel<__half, true>);
        smem_set_ps = 1;
    }
    if (attn_smem > 48u * 1024u && attn_smem > smem_set_ps) {
        cudaFuncSetAttribute((const void*)pd_attn_decode_batch_kernel<__nv_fp8_e4m3, true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, attn_smem);
        cudaFuncSetAttribute((const void*)pd_attn_decode_batch_kernel<__half, true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, attn_smem);
        smem_set_ps = attn_smem;
    }
    if (kv_dtype == PD_KV_FP8_E4M3)
        pd_attn_decode_batch_kernel<__nv_fp8_e4m3, true><<<grid, attn_nth, attn_smem, (cudaStream_t)stream>>>(
            (const float*)q, (const __nv_fp8_e4m3*)kc, (const __nv_fp8_e4m3*)vc, (const float*)sinks,
            (float*)out, (const unsigned int*)positions, (const unsigned int*)slots, n_heads,
            n_kv_heads, head_dim, max_ctx, kv_dim, swa_window, scale);
    else
        pd_attn_decode_batch_kernel<__half, true><<<grid, attn_nth, attn_smem, (cudaStream_t)stream>>>(
            (const float*)q, (const __half*)kc, (const __half*)vc, (const float*)sinks, (float*)out,
            (const unsigned int*)positions, (const unsigned int*)slots, n_heads, n_kv_heads, head_dim,
            max_ctx, kv_dim, swa_window, scale);
    return pd_launch_status();
}

// FMHA-style decode attention (slot 537). Same ABI as pd_attn_decode_batch.
// The register layout needs head_dim % 32 == 0 and (head_dim/32) % 4 == 0, so
// only 128 and 256 are served. The CALLER elects on head_dim; the guard below
// is a safety net and returns non-zero (surfacing as a launch error) rather
// than silently writing garbage.
PD_EXPORT
int pd_attn_decode_fmha(const void* q, const void* kc, const void* vc, const void* sinks,
                        void* out, const void* positions, const void* slots, uint32_t n_heads,
                        uint32_t n_kv_heads, uint32_t head_dim, uint32_t max_ctx, uint32_t kv_dim,
                        uint32_t swa_window, uint32_t batch, float scale, uint32_t kv_dtype,
                        void* stream) {
    if (n_heads == 0 || batch == 0) return 0;
    if (head_dim != 128u && head_dim != 256u) return 1;
    dim3 grid(n_heads, batch);
    // Warp count = KV-split parallelism: the walk stripes t across warps, so
    // more warps is split-KV inside one launch. At c1 (24 blocks) the 8-warp
    // form is 39 us/layer vs the rival fmha's 9.1 - 4x the warps quadruples
    // the stream count for a one-line change. Merge grouping changes with nw
    // (f32 online-softmax merge order), so the widening is env-gated and
    // battery-judged, never a silent default.
    // DEFAULT 512 (16 warps), measured 2026-09-01 on qwen4exp's decode geometry
    // (24 q heads / 4 kv / head_dim 256) with scratch fmha_probe, us a launch:
    //
    //           b8 sl256   b8 sl4095   b1 sl256
    //   nw=8      20.73      245.51      18.45
    //   nw=16     18.58      185.71      14.37
    //   nw=32     24.62      176.33      14.34
    //
    // 16 warps dominates 8 at every point measured, so this is a strict
    // improvement on the old 256 default. 32 wins only once the walk is long
    // enough to amortise the epilogue -- warp 0 merges nw partials SERIALLY --
    // and is 24% worse at the c8 serve shape, which is what the 1024 the board
    // had been electing was costing: c8 688.1 at 512 against 683.7 at 1024,
    // four legs against three, non-overlapping.
    static const uint32_t nth = [] {
        const char* v = pd_env("PADDOCK_Q38FN_FMHA_NTH");
        const uint32_t n = v ? (uint32_t)atoi(v) : 512u;
        return (n == 256u || n == 512u || n == 1024u) ? n : 512u;
    }();
    const uint32_t nw = nth / 32u;
    const uint32_t smem = (nw * head_dim + 2u * nw) * (uint32_t)sizeof(float);
    if (head_dim == 256u) {
        if (kv_dtype == PD_KV_FP8_E4M3)
            pd_attn_decode_fmha_kernel<__nv_fp8_e4m3, 8><<<grid, nth, smem, (cudaStream_t)stream>>>(
                (const float*)q, (const __nv_fp8_e4m3*)kc, (const __nv_fp8_e4m3*)vc,
                (const float*)sinks, (float*)out, (const unsigned int*)positions,
                (const unsigned int*)slots, n_heads, n_kv_heads, head_dim, max_ctx, kv_dim,
                swa_window, scale);
        else
            pd_attn_decode_fmha_kernel<__half, 8><<<grid, nth, smem, (cudaStream_t)stream>>>(
                (const float*)q, (const __half*)kc, (const __half*)vc, (const float*)sinks,
                (float*)out, (const unsigned int*)positions, (const unsigned int*)slots, n_heads,
                n_kv_heads, head_dim, max_ctx, kv_dim, swa_window, scale);
    } else {
        if (kv_dtype == PD_KV_FP8_E4M3)
            pd_attn_decode_fmha_kernel<__nv_fp8_e4m3, 4><<<grid, nth, smem, (cudaStream_t)stream>>>(
                (const float*)q, (const __nv_fp8_e4m3*)kc, (const __nv_fp8_e4m3*)vc,
                (const float*)sinks, (float*)out, (const unsigned int*)positions,
                (const unsigned int*)slots, n_heads, n_kv_heads, head_dim, max_ctx, kv_dim,
                swa_window, scale);
        else
            pd_attn_decode_fmha_kernel<__half, 4><<<grid, nth, smem, (cudaStream_t)stream>>>(
                (const float*)q, (const __half*)kc, (const __half*)vc, (const float*)sinks,
                (float*)out, (const unsigned int*)positions, (const unsigned int*)slots, n_heads,
                n_kv_heads, head_dim, max_ctx, kv_dim, swa_window, scale);
    }
    return pd_launch_status();
}

// SPLIT-KV decode attention (slot 545): the fmha walk striped over
// grid.z KV slices + a fixed-order merge pass. `part` is CALLER-OWNED,
// address-stable scratch of batch*n_heads*S*(head_dim+2) floats (graph
// capture forbids lazy allocation - pack-scratch law). S is the caller's
// split election; the un-split slot-537 form is 24 CTAs at c1 and 39
// us/layer vs the rival's 9.1. Own numeric class (per-slice partial
// merges), env-gated and battery-judged by the engine.
PD_EXPORT
int pd_attn_decode_fmha_sp(const void* q, const void* kc, const void* vc, const void* sinks,
                           void* out, void* part, const void* positions, const void* slots,
                           uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
                           uint32_t max_ctx, uint32_t kv_dim, uint32_t swa_window,
                           uint32_t batch, uint32_t split, float scale, uint32_t kv_dtype,
                           void* stream) {
    if (n_heads == 0 || batch == 0) return 0;
    if (head_dim != 128u && head_dim != 256u) return 1;
    if (split < 2u || split > 64u) return 1;
    // same 16-warp default as the unsplit form above
    static const uint32_t nth = [] {
        const char* v = pd_env("PADDOCK_Q38FN_FMHA_NTH");
        const uint32_t n = v ? (uint32_t)atoi(v) : 512u;
        return (n == 256u || n == 512u || n == 1024u) ? n : 512u;
    }();
    const uint32_t nw = nth / 32u;
    const uint32_t smem = (nw * head_dim + 2u * nw) * (uint32_t)sizeof(float);
    dim3 grid(n_heads, batch, split);
    if (head_dim == 256u) {
        if (kv_dtype == PD_KV_FP8_E4M3)
            pd_attn_decode_fmha_sp_kernel<__nv_fp8_e4m3, 8><<<grid, nth, smem, (cudaStream_t)stream>>>(
                (const float*)q, (const __nv_fp8_e4m3*)kc, (const __nv_fp8_e4m3*)vc,
                (float*)part, (const unsigned int*)positions, (const unsigned int*)slots,
                n_heads, n_kv_heads, head_dim, max_ctx, kv_dim, swa_window, scale);
        else
            pd_attn_decode_fmha_sp_kernel<__half, 8><<<grid, nth, smem, (cudaStream_t)stream>>>(
                (const float*)q, (const __half*)kc, (const __half*)vc, (float*)part,
                (const unsigned int*)positions, (const unsigned int*)slots,
                n_heads, n_kv_heads, head_dim, max_ctx, kv_dim, swa_window, scale);
    } else {
        if (kv_dtype == PD_KV_FP8_E4M3)
            pd_attn_decode_fmha_sp_kernel<__nv_fp8_e4m3, 4><<<grid, nth, smem, (cudaStream_t)stream>>>(
                (const float*)q, (const __nv_fp8_e4m3*)kc, (const __nv_fp8_e4m3*)vc,
                (float*)part, (const unsigned int*)positions, (const unsigned int*)slots,
                n_heads, n_kv_heads, head_dim, max_ctx, kv_dim, swa_window, scale);
        else
            pd_attn_decode_fmha_sp_kernel<__half, 4><<<grid, nth, smem, (cudaStream_t)stream>>>(
                (const float*)q, (const __half*)kc, (const __half*)vc, (float*)part,
                (const unsigned int*)positions, (const unsigned int*)slots,
                n_heads, n_kv_heads, head_dim, max_ctx, kv_dim, swa_window, scale);
    }
    dim3 mgrid(n_heads, batch);
    pd_attn_fmha_merge_kernel<<<mgrid, head_dim, 0, (cudaStream_t)stream>>>(
        (const float*)part, (const float*)sinks, (float*)out, n_heads, head_dim, split);
    return pd_launch_status();
}

// Paged decode launcher: same shape/carveout as pd_attn_decode_batch, but K/V
// come from the block pool (pool_k/pool_v) addressed through block_tables. No
// max_ctx argument - the pool + block table replace the per-slot reservation.
PD_EXPORT
int pd_attn_decode_batch_paged(const void* q, const void* pool_k, const void* pool_v,
                               const void* sinks, void* out, const void* positions,
                               const void* slots, const void* block_tables,
                               uint32_t blocks_per_slot, uint32_t n_heads, uint32_t n_kv_heads,
                               uint32_t head_dim, uint32_t kv_dim, uint32_t swa_window,
                               uint32_t batch, float scale, uint32_t kv_dtype, void* stream) {
    if (n_heads == 0 || batch == 0) return 0;
    dim3 grid(n_heads, batch);
    uint32_t attn_nth = head_dim > 256 ? head_dim : 256;
    uint32_t attn_smem = (uint32_t)PD_ATTN_TILE_SMEM(head_dim);
    static uint32_t smem_set_bp = 0;
    if (smem_set_bp == 0) {
        pd_prefer_max_shared(pd_attn_decode_batch_paged_kernel<__nv_fp8_e4m3>);
        pd_prefer_max_shared(pd_attn_decode_batch_paged_kernel<__half>);
        smem_set_bp = 1;
    }
    if (attn_smem > 48u * 1024u && attn_smem > smem_set_bp) {
        cudaFuncSetAttribute((const void*)pd_attn_decode_batch_paged_kernel<__nv_fp8_e4m3>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, attn_smem);
        cudaFuncSetAttribute((const void*)pd_attn_decode_batch_paged_kernel<__half>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, attn_smem);
        smem_set_bp = attn_smem;
    }
    if (kv_dtype == PD_KV_FP8_E4M3)
        pd_attn_decode_batch_paged_kernel<__nv_fp8_e4m3><<<grid, attn_nth, attn_smem, (cudaStream_t)stream>>>(
            (const float*)q, (const __nv_fp8_e4m3*)pool_k, (const __nv_fp8_e4m3*)pool_v,
            (const float*)sinks, (float*)out, (const unsigned int*)positions,
            (const unsigned int*)slots, (const uint32_t*)block_tables, blocks_per_slot,
            n_heads, n_kv_heads, head_dim, kv_dim, swa_window, scale);
    else
        pd_attn_decode_batch_paged_kernel<__half><<<grid, attn_nth, attn_smem, (cudaStream_t)stream>>>(
            (const float*)q, (const __half*)pool_k, (const __half*)pool_v, (const float*)sinks,
            (float*)out, (const unsigned int*)positions, (const unsigned int*)slots,
            (const uint32_t*)block_tables, blocks_per_slot, n_heads, n_kv_heads, head_dim,
            kv_dim, swa_window, scale);
    return pd_launch_status();
}

// a16 twin for the attention streams: q and out are f16 planes (the
// splits==1 direct-write walk). Appended as its own export per the ABI
// growth rule.
PD_EXPORT
int pd_attn_decode_batch_paged2(const void* q, const void* pool_k, const void* pool_v,
                                const void* sinks, void* out, const void* positions,
                                const void* slots, const void* block_tables,
                                uint32_t blocks_per_slot, uint32_t n_heads, uint32_t n_kv_heads,
                                uint32_t head_dim, uint32_t kv_dim, uint32_t swa_window,
                                uint32_t batch, float scale, uint32_t kv_dtype, uint32_t a16,
                                void* stream) {
    if (!a16)
        return pd_attn_decode_batch_paged(q, pool_k, pool_v, sinks, out, positions,
                                          slots, block_tables, blocks_per_slot, n_heads,
                                          n_kv_heads, head_dim, kv_dim, swa_window,
                                          batch, scale, kv_dtype, stream);
    if (n_heads == 0 || batch == 0) return 0;
    dim3 grid(n_heads, batch);
    uint32_t attn_nth = head_dim > 256 ? head_dim : 256;
    uint32_t attn_smem = (uint32_t)PD_ATTN_TILE_SMEM(head_dim);
    static uint32_t smem_set_bp16 = 0;
    if (smem_set_bp16 == 0) {
        pd_prefer_max_shared(pd_attn_decode_batch_paged_kernel<__nv_fp8_e4m3, __half, __half>);
        pd_prefer_max_shared(pd_attn_decode_batch_paged_kernel<__half, __half, __half>);
        smem_set_bp16 = 1;
    }
    if (attn_smem > 48u * 1024u && attn_smem > smem_set_bp16) {
        cudaFuncSetAttribute((const void*)pd_attn_decode_batch_paged_kernel<__nv_fp8_e4m3, __half, __half>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, attn_smem);
        cudaFuncSetAttribute((const void*)pd_attn_decode_batch_paged_kernel<__half, __half, __half>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, attn_smem);
        smem_set_bp16 = attn_smem;
    }
    if (kv_dtype == PD_KV_FP8_E4M3)
        pd_attn_decode_batch_paged_kernel<__nv_fp8_e4m3, __half, __half><<<grid, attn_nth, attn_smem, (cudaStream_t)stream>>>(
            (const __half*)q, (const __nv_fp8_e4m3*)pool_k, (const __nv_fp8_e4m3*)pool_v,
            (const float*)sinks, (__half*)out, (const unsigned int*)positions,
            (const unsigned int*)slots, (const uint32_t*)block_tables, blocks_per_slot,
            n_heads, n_kv_heads, head_dim, kv_dim, swa_window, scale);
    else
        pd_attn_decode_batch_paged_kernel<__half, __half, __half><<<grid, attn_nth, attn_smem, (cudaStream_t)stream>>>(
            (const __half*)q, (const __half*)pool_k, (const __half*)pool_v, (const float*)sinks,
            (__half*)out, (const unsigned int*)positions, (const unsigned int*)slots,
            (const uint32_t*)block_tables, blocks_per_slot, n_heads, n_kv_heads, head_dim,
            kv_dim, swa_window, scale);
    return pd_launch_status();
}

// Paged KV-append launcher: scatter into the block pool via block_tables.
PD_EXPORT
int pd_kv_append_batch_paged(const void* kv, void* pool, const void* positions, const void* slots,
                             const void* block_tables, uint32_t blocks_per_slot, uint32_t kv_dim,
                             uint32_t batch, uint32_t kv_dtype, void* stream) {
    if (kv_dim == 0 || batch == 0) return 0;
    uint32_t threads = 256;
    dim3 grid((kv_dim + threads - 1) / threads, batch);
    if (kv_dtype == PD_KV_FP8_E4M3)
        pd_kv_append_batch_paged_kernel<__nv_fp8_e4m3><<<grid, threads, 0, (cudaStream_t)stream>>>(
            (const float*)kv, (__nv_fp8_e4m3*)pool, (const unsigned int*)positions,
            (const unsigned int*)slots, (const uint32_t*)block_tables, blocks_per_slot, kv_dim, batch);
    else
        pd_kv_append_batch_paged_kernel<__half><<<grid, threads, 0, (cudaStream_t)stream>>>(
            (const float*)kv, (__half*)pool, (const unsigned int*)positions,
            (const unsigned int*)slots, (const uint32_t*)block_tables, blocks_per_slot, kv_dim, batch);
    return pd_launch_status();
}

// DFlash conditioning-fold launcher (rung C,  - see the kernel
// note in elementwise.cuh). One block per (written row, kv head, k/v
// plane); nth mirrors pd_rmsnorm_batch's election for norm_batch rows so
// the k-norm reduction grouping (and therefore every pool byte) matches
// the norm -> rope -> append chain exactly. head_dim caps at 256 (shared
// staging) - the dflash drafters are hd 128; a bigger head declines and
// the caller falls back to the chain.
PD_EXPORT
int pd_dflash_cond_append(const void* fk, const void* fv, const void* kw,
                          void* pool_k, void* pool_v, const void* rows_w,
                          const void* positions, const void* slots,
                          const void* block_tables, uint32_t blocks_per_slot,
                          uint32_t n_kv, uint32_t head_dim, float eps,
                          float theta_scale, float freq_scale, float corr_low,
                          float corr_high, float ext_factor, float mscale,
                          uint32_t nw, uint32_t norm_batch, void* stream) {
    if (nw == 0 || n_kv == 0) return 0;
    if (head_dim > 256u) return 1;
    const uint32_t nth =
        norm_batch >= 64u ? pd_norm_wide_nth_ws(norm_batch) : pd_norm_decode_nth();
    dim3 grid(nw, n_kv, 2);
    pd_dflash_cond_append_kernel<<<grid, nth, 0, (cudaStream_t)stream>>>(
        (const float*)fk, (const float*)fv, (const float*)kw,
        (__half*)pool_k, (__half*)pool_v, (const uint32_t*)rows_w,
        (const unsigned int*)positions, (const unsigned int*)slots,
        (const uint32_t*)block_tables, blocks_per_slot, n_kv, head_dim, eps,
        theta_scale, freq_scale, corr_low, corr_high, ext_factor, mscale);
    return pd_launch_status();
}

// Fused NORM-rope(q,k) + paged K/V append launcher (granite decode-chain
// fold - see the kernel note in elementwise.cuh). Warp per (row, slot),
// slots = n_heads q + n_kv k + n_kv v.
PD_EXPORT
int pd_rope_norm_qk_append_paged(void* q, void* k, const void* v, void* pool_k, void* pool_v,
                                 const void* positions, const void* slots,
                                 const void* block_tables, uint32_t blocks_per_slot,
                                 uint32_t n_heads, uint32_t n_kv, uint32_t head_dim,
                                 float theta_scale, float freq_scale, float corr_low,
                                 float corr_high, float ext_factor, float mscale,
                                 uint32_t batch, uint32_t kv_dtype, void* stream) {
    if (n_heads == 0 || n_kv == 0 || batch == 0) return 0;
    uint32_t total = batch * (n_heads + 2u * n_kv), threads = 256; // 8 warps/block
    uint32_t blocks = (total * 32u + threads - 1) / threads;
    if (kv_dtype == PD_KV_FP8_E4M3)
        pd_pdl_go(pd_rope_norm_qk_append_paged_kernel<__nv_fp8_e4m3>, blocks, threads, 0u, (cudaStream_t)stream,
            (float*)q, (float*)k, (const float*)v, (__nv_fp8_e4m3*)pool_k,
            (__nv_fp8_e4m3*)pool_v, (const unsigned int*)positions,
            (const unsigned int*)nullptr, (const unsigned int*)slots,
            (const uint32_t*)block_tables, blocks_per_slot, n_heads, n_kv, head_dim,
            theta_scale, freq_scale, corr_low, corr_high, ext_factor, mscale, batch);
    else
        pd_pdl_go(pd_rope_norm_qk_append_paged_kernel<__half>, blocks, threads, 0u, (cudaStream_t)stream,
            (float*)q, (float*)k, (const float*)v, (__half*)pool_k, (__half*)pool_v,
            (const unsigned int*)positions, (const unsigned int*)nullptr,
            (const unsigned int*)slots,
            (const uint32_t*)block_tables, blocks_per_slot, n_heads, n_kv, head_dim,
            theta_scale, freq_scale, corr_low, corr_high, ext_factor, mscale, batch);
    return pd_launch_status();
}

// Ring twin (deepseek-ocr): same fold, two position streams -
// `positions` turns the rope (true position, absolute forever), `wpos` lands
// the appends (the R-SWA ring's write slot; equal to positions everywhere
// the ring hasn't engaged). `neox` picks the rope pair layout at compile
// time via the kernel template. Appended per the ABI growth rule; the
// granite export above keeps its exact signature and byte behavior.
PD_EXPORT
int pd_rope_qk_append_paged_ring(void* q, void* k, const void* v, void* pool_k, void* pool_v,
                                 const void* positions, const void* wpos, const void* slots,
                                 const void* block_tables, uint32_t blocks_per_slot,
                                 uint32_t n_heads, uint32_t n_kv, uint32_t head_dim,
                                 float theta_scale, float freq_scale, float corr_low,
                                 float corr_high, float ext_factor, float mscale,
                                 uint32_t batch, uint32_t neox, uint32_t kv_dtype,
                                 void* stream) {
    if (n_heads == 0 || n_kv == 0 || batch == 0) return 0;
    uint32_t total = batch * (n_heads + 2u * n_kv), threads = 256; // 8 warps/block
    uint32_t blocks = (total * 32u + threads - 1) / threads;
    #define PD_RQAR_GO(KV_, NX_)                                                    \
        pd_pdl_go(pd_rope_norm_qk_append_paged_kernel<KV_, NX_>, blocks, threads,   \
            0u, (cudaStream_t)stream,                                               \
            (float*)q, (float*)k, (const float*)v, (KV_*)pool_k, (KV_*)pool_v,      \
            (const unsigned int*)positions, (const unsigned int*)wpos,              \
            (const unsigned int*)slots,                                             \
            (const uint32_t*)block_tables, blocks_per_slot, n_heads, n_kv,          \
            head_dim, theta_scale, freq_scale, corr_low, corr_high, ext_factor,     \
            mscale, batch)
    if (kv_dtype == PD_KV_FP8_E4M3) {
        if (neox) PD_RQAR_GO(__nv_fp8_e4m3, true); else PD_RQAR_GO(__nv_fp8_e4m3, false);
    } else {
        if (neox) PD_RQAR_GO(__half, true); else PD_RQAR_GO(__half, false);
    }
    #undef PD_RQAR_GO
    return pd_launch_status();
}

// fused single-pass GQA-16 decode attention - see the kernel note
// in attn/decode.cuh. FINAL-output contract (sinks folded in-kernel): the
// caller skips the combine entirely. `pos_max` is the host-side max position
// over the rows and sizes the smem stage; callers pass the kv_split_band
// ceiling so captured graphs stay valid across the band. Shape-gated to the
// fp8/hd128/G16 geometry (muse-glimmer) - rc -2 otherwise, rc -3 when the
// windowed context exceeds the smem opt-in (the engine's band gate should
// never let either happen).
PD_EXPORT
int pd_attn_decode_fused_gqa16(const void* q, const void* pool_k, const void* pool_v,
                               const void* sinks, void* out, const void* positions,
                               const void* slots, const void* block_tables,
                               uint32_t blocks_per_slot, uint32_t n_heads,
                               uint32_t n_kv_heads, uint32_t head_dim, uint32_t kv_dim,
                               uint32_t swa_window, uint32_t batch, uint32_t pos_max,
                               float scale, uint32_t kv_dtype, void* stream) {
    if (n_heads == 0 || batch == 0) return 0;
    const uint32_t group = n_kv_heads ? n_heads / n_kv_heads : 1;
    if (kv_dtype != PD_KV_FP8_E4M3 || head_dim != 128u || group != 16u
        || n_heads != n_kv_heads * group)
        return -2;
    uint32_t n_eff = pos_max + 1u;
    if (swa_window > 0 && n_eff > swa_window) n_eff = swa_window;
    const uint32_t smem = n_eff * head_dim * 2u + 16u * head_dim * 4u;
    static int smem_cap = -1;
    if (smem_cap < 0)
        cudaDeviceGetAttribute(&smem_cap, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    if ((int)smem > smem_cap) return -3;
    static uint32_t fsmem_set = 0;
    if (smem > 48u * 1024u && smem > fsmem_set) {
        cudaFuncSetAttribute((const void*)pd_attn_decode_fused_gqa16_kernel<128u, 16u>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        fsmem_set = smem;
    }
    dim3 grid(n_kv_heads, batch);
    pd_pdl_go(pd_attn_decode_fused_gqa16_kernel<128u, 16u>,
        grid, 32u * 16u, smem, (cudaStream_t)stream,
            (const float*)q, (const __nv_fp8_e4m3*)pool_k,
            (const __nv_fp8_e4m3*)pool_v, (float*)out,
            (const unsigned int*)positions, (const unsigned int*)slots,
            (const uint32_t*)block_tables, blocks_per_slot, n_heads,
            n_kv_heads, kv_dim, swa_window, scale, (const float*)sinks);
    return pd_launch_status();
}

PD_EXPORT
int pd_attn_decode_partial(const void* q, const void* kc, const void* vc,
                           void* out_o, void* out_ml, uint32_t n_heads, uint32_t n_kv_heads,
                           uint32_t head_dim, uint32_t first_pos, uint32_t n_pos,
                           uint32_t n_splits, uint32_t kv_dim, float scale, void* stream) {
    if (n_pos == 0 || n_heads == 0 || n_splits == 0) return 0;
    dim3 grid(n_heads, n_splits);
    pd_attn_decode_partial_kernel<<<grid, head_dim, 0, (cudaStream_t)stream>>>(
        (const float*)q, (const __half*)kc, (const __half*)vc, (float*)out_o, (float*)out_ml,
        n_heads, n_kv_heads, head_dim, first_pos, n_pos, n_splits, kv_dim, scale);
    return pd_launch_status();
}

PD_EXPORT
int pd_attn_decode_combine(const void* in_o, const void* in_ml, const void* sinks,
                           void* out, uint32_t n_heads, uint32_t head_dim,
                           uint32_t n_splits, void* stream) {
    if (n_heads == 0 || n_splits == 0) return 0;
    pd_attn_decode_combine_kernel<<<n_heads, head_dim, 0, (cudaStream_t)stream>>>(
        (const float*)in_o, (const float*)in_ml, (const float*)sinks, (float*)out,
        n_heads, head_dim, n_splits);
    return pd_launch_status();
}

// PV-mma mode for the GQA-fused partial kernels (dense + paged share the
// latch so the bitwise dense↔paged invariant holds under either setting).
// Default on at the hd256/f16 serving shape (measured c8 293.7 -> 294.2,
// dc32 987.8 -> 994.1, greedy-exact); PADDOCK_NO_ATTN_MMA_PV pins the
// scalar V fold.
static inline int pd_attn_pv_mma() {
    static int m = -1;
    if (m < 0) m = pd_env("PADDOCK_NO_ATTN_MMA_PV") ? 0 : 1;
    return m;
}

// lagd election: the hd128 v5-class
// partial - see src/attn/lagd.cuh for the design + harness numbers. Default
// on at hd128/f16 fused decode shapes (dense + paged share the latch so the
// bitwise dense<->paged invariant holds under either setting).
// PADDOCK_NO_LAGD falls back to the GQA walk (the previous state);
// PADDOCK_LAGD_F16 selects the straight-f16 planes (llama fattn's class)
// instead of the class-preserving splits.
static inline int pd_attn_lagd() {
    static int m = -1;
    if (m < 0) m = pd_env("PADDOCK_NO_LAGD") ? 0 : 1;
    return m;
}
static inline int pd_attn_lagd_f16() {
    static int m = -1;
    if (m < 0) m = pd_env("PADDOCK_LAGD_F16") ? 1 : 0;
    return m;
}
#define PD_LAGD_SMEM_SPLIT \
    (2u * 16u * 136u * 2u + 3u * 16u * 40u * 2u + 2u * 2u * 32u * 136u * 2u)
#define PD_LAGD_SMEM_F16 \
    (16u * 136u * 2u + 16u * 40u * 2u + 2u * 2u * 32u * 136u * 2u)

PD_EXPORT
int pd_attn_decode_batch_partial(const void* q, const void* kc, const void* vc, void* out_o,
                                 void* out_ml, const void* positions, const void* slots,
                                 uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
                                 uint32_t max_ctx, uint32_t kv_dim, uint32_t swa_window,
                                 uint32_t n_splits, uint32_t batch, float scale, uint32_t kv_dtype,
                                 void* stream) {
    if (n_heads == 0 || batch == 0 || n_splits == 0) return 0;
    dim3 grid(n_heads, batch, n_splits);
    // 256-thread blocks: the tile walk's staging loop is the memory phase and a
    // head_dim-sized block cannot keep enough loads in flight; opt into the max
    // shared carveout so the ~17-34 KB tiles do not halve resident blocks. At
    // head_dim 256 the tile is ~67 KB - past the 48 KB default window - so the
    // dynamic-shared ceiling must be raised explicitly (delta-net pattern).
    uint32_t attn_nth = head_dim > 256 ? head_dim : 256;
    uint32_t attn_smem = (uint32_t)PD_ATTN_TILE_SMEM(head_dim);
    static uint32_t smem_set_p = 0;
    if (smem_set_p == 0) {
        pd_prefer_max_shared(pd_attn_decode_batch_partial_kernel<__nv_fp8_e4m3>);
        pd_prefer_max_shared(pd_attn_decode_batch_partial_kernel<__half>);
        // the GQA-fused instantiations too - at the default carveout their
        // ~22 KB dynamic smem capped residency at 1-2 blocks/SM and the
        // fused walk ran at ~7% of DRAM (570 us/launch at B=32 depth 1100);
        // the carveout alone is worth ~4x occupancy
        pd_prefer_max_shared(pd_attn_decode_batch_partial_gqa_kernel<__nv_fp8_e4m3, 16u>);
        pd_prefer_max_shared(pd_attn_decode_batch_partial_gqa_kernel<__half, 16u>);
        pd_prefer_max_shared(pd_attn_decode_batch_partial_gqa_kernel<__nv_fp8_e4m3, 32u>);
        pd_prefer_max_shared(pd_attn_decode_batch_partial_gqa_kernel<__half, 32u>);
        smem_set_p = 1;
    }
    if (attn_smem > 48u * 1024u && attn_smem > smem_set_p) {
        cudaFuncSetAttribute((const void*)pd_attn_decode_batch_partial_kernel<__nv_fp8_e4m3>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, attn_smem);
        cudaFuncSetAttribute((const void*)pd_attn_decode_batch_partial_kernel<__half>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, attn_smem);
        smem_set_p = attn_smem;
    }
    // GQA fusion: one block per KV head serves the whole q-group with a
    // single K/V stage - bit-identical partials, group_size-x less KV
    // traffic. Two tile classes: hd > 128 (qwen full-attn, tile 16) and
    // hd <= 128 (gpt-oss hd 64, tile 32 - originally left per-q-head from
    // A6000 B=1 economics, but GB202 SERVING batch measured the per-q-head
    // grid at ~11% of bandwidth: 8 q-heads re-reading every tile cost
    // +10.7 ms/step at B=32 depth 1100; engine-side attn_splits budgets the
    // fused grid to match).
    const uint32_t group = n_kv_heads ? n_heads / n_kv_heads : 1;
    // PD_NO_GQA_FUSE=1 pins the per-q-head grid (A/B). n_kv_heads >= 4 keeps
    // the fused grid >= 128 blocks at 32 splits - the 2-KV-head 35B measured
    // the fused walk LOSING at depth (64 blocks hit the serial-tile latency
    // wall harder than 8x KV re-reads cost: see the A/B in the commit).
    static int no_gqa = -1;
    if (no_gqa < 0) no_gqa = pd_env("PD_NO_GQA_FUSE") ? 1 : 0;
    if (!no_gqa && head_dim <= 128u && group >= 2u && group <= 8u && n_kv_heads >= 2u
        && n_heads == n_kv_heads * group) {
        const uint32_t tile = 32u;  // PD_ATTN_TILE class (hd <= 128)
        const uint32_t gqa_smem = (uint32_t)PD_GQA_SMEM(group, head_dim, tile);
        // (G=8, hd=64, T=32): ~20.7 KB - inside the default smem window, but
        // keep the raise-if-needed pattern for bigger shapes
        static uint32_t gqa32_set = 0;
        if (gqa_smem > 48u * 1024u && gqa_smem > gqa32_set) {
            cudaFuncSetAttribute(
                (const void*)pd_attn_decode_batch_partial_gqa_kernel<__nv_fp8_e4m3, 32u>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, gqa_smem);
            cudaFuncSetAttribute(
                (const void*)pd_attn_decode_batch_partial_gqa_kernel<__half, 32u>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, gqa_smem);
            gqa32_set = gqa_smem;
        }
        dim3 ggrid(n_kv_heads, batch, n_splits);
        // lagd - dense twin of the paged arm (shared latch keeps the bitwise
        // dense<->paged invariant under either setting; see the paged arm).
        if (pd_attn_lagd() && head_dim == 128u && kv_dtype != PD_KV_FP8_E4M3) {
            static uint32_t lagd_set_d = 0;
            if (lagd_set_d == 0) {
                pd_prefer_max_shared(pd_attn_decode_lagd_kernel<32u, 2u, 3u, false>);
                pd_prefer_max_shared(pd_attn_decode_lagd_kernel<32u, 1u, 1u, false>);
                lagd_set_d = 1;
            }
            if (pd_attn_lagd_f16())
                pd_attn_decode_lagd_kernel<32u, 1u, 1u, false>
                    <<<ggrid, 256u, PD_LAGD_SMEM_F16, (cudaStream_t)stream>>>(
                        (const float*)q, (const __half*)kc, (const __half*)vc,
                        (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                        (const unsigned int*)slots, nullptr, 0u, max_ctx,
                        n_heads, n_kv_heads, kv_dim, swa_window, n_splits, scale);
            else
                pd_attn_decode_lagd_kernel<32u, 2u, 3u, false>
                    <<<ggrid, 256u, PD_LAGD_SMEM_SPLIT, (cudaStream_t)stream>>>(
                        (const float*)q, (const __half*)kc, (const __half*)vc,
                        (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                        (const unsigned int*)slots, nullptr, 0u, max_ctx,
                        n_heads, n_kv_heads, kv_dim, swa_window, n_splits, scale);
            return pd_launch_status();
        }
        // PV-mma hd128 - dense twin of the paged arm (shared latch keeps the
        // bitwise dense<->paged invariant under either setting).
        if (pd_attn_pv_mma() && head_dim == 128u && kv_dtype != PD_KV_FP8_E4M3) {
            static uint32_t pv32_set = 0;
            if (pv32_set == 0) {
                pd_prefer_max_shared(pd_attn_decode_batch_partial_gqa_kernel<__half, 32u, true>);
                pv32_set = 1;
            }
            if (gqa_smem > 48u * 1024u && gqa_smem > pv32_set) {
                cudaFuncSetAttribute(
                    (const void*)pd_attn_decode_batch_partial_gqa_kernel<__half, 32u, true>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, gqa_smem);
                pv32_set = gqa_smem;
            }
            pd_attn_decode_batch_partial_gqa_kernel<__half, 32u, true>
                <<<ggrid, attn_nth, gqa_smem, (cudaStream_t)stream>>>(
                    (const float*)q, (const __half*)kc, (const __half*)vc, (float*)out_o,
                    (float*)out_ml, (const unsigned int*)positions,
                    (const unsigned int*)slots, n_heads, n_kv_heads, head_dim, max_ctx,
                    kv_dim, swa_window, n_splits, scale);
            return pd_launch_status();
        }
        if (kv_dtype == PD_KV_FP8_E4M3)
            pd_attn_decode_batch_partial_gqa_kernel<__nv_fp8_e4m3, 32u>
                <<<ggrid, attn_nth, gqa_smem, (cudaStream_t)stream>>>(
                    (const float*)q, (const __nv_fp8_e4m3*)kc, (const __nv_fp8_e4m3*)vc,
                    (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                    (const unsigned int*)slots, n_heads, n_kv_heads, head_dim, max_ctx,
                    kv_dim, swa_window, n_splits, scale);
        else
            pd_attn_decode_batch_partial_gqa_kernel<__half, 32u>
                <<<ggrid, attn_nth, gqa_smem, (cudaStream_t)stream>>>(
                    (const float*)q, (const __half*)kc, (const __half*)vc, (float*)out_o,
                    (float*)out_ml, (const unsigned int*)positions,
                    (const unsigned int*)slots, n_heads, n_kv_heads, head_dim, max_ctx,
                    kv_dim, swa_window, n_splits, scale);
        return pd_launch_status();
    }
    if (!no_gqa && head_dim > 128u && head_dim <= 256u && group >= 2u && group <= 8u
        && n_kv_heads >= 2u && n_heads == n_kv_heads * group) {
        // PD_ATTN_TILE256=32: A/B the 32-token tile at hd 256 (halves the
        // serial tile iterations + barriers per split; doubles staged bytes
        // in flight; ~67 KB smem -> occupancy trade measured, not assumed)
        static int t256 = -1;
        if (t256 < 0) { const char* e = pd_env("PD_ATTN_TILE256"); t256 = (e && atoi(e) == 32) ? 32 : 16; }
        if (t256 == 32) {
            const uint32_t gqa_smem32 = (uint32_t)PD_GQA_SMEM(group, head_dim, 32u);
            static uint32_t g32set = 0;
            if (gqa_smem32 > g32set) {
                cudaFuncSetAttribute(
                    (const void*)pd_attn_decode_batch_partial_gqa_kernel<__nv_fp8_e4m3, 32u>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, gqa_smem32);
                cudaFuncSetAttribute(
                    (const void*)pd_attn_decode_batch_partial_gqa_kernel<__half, 32u>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, gqa_smem32);
                g32set = gqa_smem32;
            }
            dim3 ggrid(n_kv_heads, batch, n_splits);
            if (kv_dtype == PD_KV_FP8_E4M3)
                pd_attn_decode_batch_partial_gqa_kernel<__nv_fp8_e4m3, 32u>
                    <<<ggrid, attn_nth, gqa_smem32, (cudaStream_t)stream>>>(
                        (const float*)q, (const __nv_fp8_e4m3*)kc, (const __nv_fp8_e4m3*)vc,
                        (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                        (const unsigned int*)slots, n_heads, n_kv_heads, head_dim, max_ctx,
                        kv_dim, swa_window, n_splits, scale);
            else
                pd_attn_decode_batch_partial_gqa_kernel<__half, 32u>
                    <<<ggrid, attn_nth, gqa_smem32, (cudaStream_t)stream>>>(
                        (const float*)q, (const __half*)kc, (const __half*)vc, (float*)out_o,
                        (float*)out_ml, (const unsigned int*)positions,
                        (const unsigned int*)slots, n_heads, n_kv_heads, head_dim, max_ctx,
                        kv_dim, swa_window, n_splits, scale);
            return pd_launch_status();
        }
        const uint32_t tile = 16u;  // PD_ATTN_TILE_FOR(hd > 128)
        const uint32_t gqa_smem = (uint32_t)PD_GQA_SMEM(group, head_dim, tile);
        static uint32_t gqa_set = 0;
        if (gqa_smem > 48u * 1024u && gqa_smem > gqa_set) {
            cudaFuncSetAttribute(
                (const void*)pd_attn_decode_batch_partial_gqa_kernel<__nv_fp8_e4m3, 16u>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, gqa_smem);
            cudaFuncSetAttribute(
                (const void*)pd_attn_decode_batch_partial_gqa_kernel<__half, 16u>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, gqa_smem);
            gqa_set = gqa_smem;
        }
        dim3 ggrid(n_kv_heads, batch, n_splits);
        // PV-mma (default-on, kill PADDOCK_NO_ATTN_MMA_PV): hd256/f16 P*V on
        // tf32 tensor cores - the dense twin of the paged launcher's arm, so
        // paged-vs-dense stays bitwise (identical instruction sequence).
        if (pd_attn_pv_mma() && kv_dtype != PD_KV_FP8_E4M3 && head_dim == 256u) {
            static uint32_t pvd_set = 0;
            if (pvd_set == 0) {
                pd_prefer_max_shared(pd_attn_decode_batch_partial_gqa_kernel<__half, 16u, true>);
                pvd_set = 1;
            }
            if (gqa_smem > 48u * 1024u && gqa_smem > pvd_set) {
                cudaFuncSetAttribute(
                    (const void*)pd_attn_decode_batch_partial_gqa_kernel<__half, 16u, true>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, gqa_smem);
                pvd_set = gqa_smem;
            }
            pd_attn_decode_batch_partial_gqa_kernel<__half, 16u, true>
                <<<ggrid, attn_nth, gqa_smem, (cudaStream_t)stream>>>(
                    (const float*)q, (const __half*)kc, (const __half*)vc, (float*)out_o,
                    (float*)out_ml, (const unsigned int*)positions,
                    (const unsigned int*)slots, n_heads, n_kv_heads, head_dim, max_ctx,
                    kv_dim, swa_window, n_splits, scale);
            return pd_launch_status();
        }
        if (kv_dtype == PD_KV_FP8_E4M3)
            pd_attn_decode_batch_partial_gqa_kernel<__nv_fp8_e4m3, 16u>
                <<<ggrid, attn_nth, gqa_smem, (cudaStream_t)stream>>>(
                    (const float*)q, (const __nv_fp8_e4m3*)kc, (const __nv_fp8_e4m3*)vc,
                    (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                    (const unsigned int*)slots, n_heads, n_kv_heads, head_dim, max_ctx,
                    kv_dim, swa_window, n_splits, scale);
        else
            pd_attn_decode_batch_partial_gqa_kernel<__half, 16u>
                <<<ggrid, attn_nth, gqa_smem, (cudaStream_t)stream>>>(
                    (const float*)q, (const __half*)kc, (const __half*)vc, (float*)out_o,
                    (float*)out_ml, (const unsigned int*)positions,
                    (const unsigned int*)slots, n_heads, n_kv_heads, head_dim, max_ctx,
                    kv_dim, swa_window, n_splits, scale);
        return pd_launch_status();
    }
    if (kv_dtype == PD_KV_FP8_E4M3)
        pd_attn_decode_batch_partial_kernel<__nv_fp8_e4m3><<<grid, attn_nth, attn_smem, (cudaStream_t)stream>>>(
            (const float*)q, (const __nv_fp8_e4m3*)kc, (const __nv_fp8_e4m3*)vc, (float*)out_o,
            (float*)out_ml, (const unsigned int*)positions, (const unsigned int*)slots, n_heads,
            n_kv_heads, head_dim, max_ctx, kv_dim, swa_window, n_splits, scale);
    else
        pd_attn_decode_batch_partial_kernel<__half><<<grid, attn_nth, attn_smem, (cudaStream_t)stream>>>(
            (const float*)q, (const __half*)kc, (const __half*)vc, (float*)out_o, (float*)out_ml,
            (const unsigned int*)positions, (const unsigned int*)slots, n_heads, n_kv_heads, head_dim,
            max_ctx, kv_dim, swa_window, n_splits, scale);
    return pd_launch_status();
}

// Paged FlashDecoding partial launcher (P3b): plain per-q-head partial reading
// the block pool. Pairs with the unchanged pd_attn_decode_batch_combine (which
// is position-agnostic - it merges (O, m, l) partials + the sink). GQA-fused
// paging is P3b-2; this plain path is correct on every geometry.
// FA-route mode shared by the spec and dense-partial launchers: default-on
// at cc 10 (PADDOCK_SPEC_FA forces elsewhere, PADDOCK_NO_SPEC_FA kills)
// constexpr-geometry dispatch for pd_attn_spec_fa_kernel (fold rung):
// pins (head_dim, log2 group) at compile time for the known
// serving geometries - gemma4 SWA (256,G2), gemma4 global / qwen3.6
// (256/512, G8) - so the kernel's div/mod chains fold to shifts (the
// prefill twin measured the runtime forms at +53% inst_executed). Unknown
// shapes keep the runtime-generic instantiation; PADDOCK_SPEC_FA_GENERIC=1
// pins it everywhere for A/B. Attr high-waters are per-arm statics (one
// set per helper instantiation x arm).
template <uint32_t PT, uint32_t TPW, bool DB, bool F8 = false>
static int pd_spec_fa_go(dim3 grid, uint32_t smem, void* stream,
    const float* q, const __half* pk, const __half* pv, float* oo, float* ml,
    const unsigned int* pos, const unsigned int* slots, const uint32_t* bt,
    uint32_t bps, uint32_t nh, uint32_t nkv, uint32_t hd, uint32_t kvd,
    uint32_t swa, uint32_t ns, uint32_t rows, uint32_t k1, float scale) {
    static int generic = -1;
    if (generic < 0) generic = pd_env("PADDOCK_SPEC_FA_GENERIC") ? 1 : 0;
    const uint32_t g = nkv ? nh / nkv : 1u;
#define PD_FA_ARM(HDv, GLv, HW)                                                   {                                                                                 static uint32_t HW = 0;                                                       if (smem > 48u * 1024u && smem > HW) {                                            cudaFuncSetAttribute(                                                             (const void*)pd_attn_spec_fa_kernel<PT, TPW, DB, true, HDv, GLv, F8>,             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);                       HW = smem;                                                                }                                                                             pd_pdl_go(pd_attn_spec_fa_kernel<PT, TPW, DB, true, HDv, GLv, F8>,                grid, 256, smem, (cudaStream_t)stream,                                        q, pk, pv, oo, ml, pos, slots, bt, bps, nh, nkv, hd, kvd,                     swa, ns, rows, k1, scale);                                            return pd_launch_status();                                                }
    if (!generic) {
        if (hd == 256u && g == 2u) PD_FA_ARM(256u, 1u, hw_a)
        if (hd == 256u && g == 8u) PD_FA_ARM(256u, 3u, hw_b)
        if (hd == 512u && g == 8u) PD_FA_ARM(512u, 3u, hw_c)
    }
    PD_FA_ARM(0u, 0u, hw_g)
#undef PD_FA_ARM
}

// krs election: fp8-resident-K spec-FA, the default
// for the two gemma4 F8 serving geometries on sm_120 (SWA hd256/G2 PT=40,
// GLB hd512/G8 PT=32 - measured -2.9% / -29.9%). Kill:
// PADDOCK_NO_SPEC_KR -> the f16-expansion route; PADDOCK_SPEC_KR_PT=32
// pins the bit-equal SWA rung for A/B.
static int pd_spec_fa_krs() {
    static int v = -1;
    if (v < 0) v = pd_env("PADDOCK_NO_SPEC_KR") ? 0 : 1;
    return v;
}
// fp8 PxV rung A: V-raw-resident spec-FA - the o-mma consumes
// raw e4m3 via seam cvts (bit-equal at equal PT); the f16 V region and its
// expansion pass die (~16-33KB smem back on the class where smem headroom
// bought KR32's -30%). PADDOCK_SPEC_VR_PT pins the tile for the ladder
// (SWA 32/40/64, GLB 32/48; 0 = the arm's krs default).
// value is a per-arm mask: bit0 = SWA hd256, bit1 = GLB hd512.
// "1"/"on" = both; "swa" / "glb" pick one arm (the ladder says
// VR is a fin-leg/occupancy lever on SWA and a loss on the occ-1 GLB arm).
// DEFAULT = SWA: VR32 fin 207.8 vs KR
// 241.8 (-14%, bit-equal at equal PT), and the composite VR+FIN serve ABAB
// was never-negative (wide +0.7%, churn +0.4%). PADDOCK_SPEC_VR=0 kills.
static int pd_spec_fa_vr() {
    static int v = -1;
    if (v < 0) {
        const char* e = pd_env("PADDOCK_SPEC_VR");
        v = !e ? 1
          : e[0] == '0' || e[0] == 'n' ? 0
          : e[0] == 's' ? 1 : e[0] == 'g' ? 2 : 3;
    }
    return v;
}
static int pd_spec_vr_pt() {
    static int v = -1;
    if (v < 0) {
        const char* e = pd_env("PADDOCK_SPEC_VR_PT");
        v = e ? atoi(e) : 0;
    }
    return v;
}
// DBK: double-buffered krs KV stage - tile t+1 prefetches while
// tile t computes (bit-equal at equal PT; VR layout required). Mask like
// VR: bit0 = SWA hd256 (VR arm), bit1 = GLB hd512 (VR+QK8 arm, the occ-1
// class with zero cross-CTA overlap). Default off until the ladder + serve
// legs arbitrate. PADDOCK_SPEC_DBK=swa|glb|1; PADDOCK_SPEC_DBK_PT pins the
// SWA tile (32 = occ-2 default, 16 = the occ-3 shape).
static int pd_spec_fa_dbk() {
    static int v = -1;
    if (v < 0) {
        const char* e = pd_env("PADDOCK_SPEC_DBK");
        v = !e ? 0
          : e[0] == '0' || e[0] == 'n' ? 0
          : e[0] == 's' ? 1 : e[0] == 'g' ? 2 : 3;
    }
    return v;
}
static int pd_spec_dbk_pt() {
    static int v = -1;
    if (v < 0) {
        const char* e = pd_env("PADDOCK_SPEC_DBK_PT");
        v = e ? atoi(e) : 32;
    }
    return v;
}
template <uint32_t PT, uint32_t TPW, uint32_t HDv, uint32_t GLv, bool VRv = false,
          bool QK8v = false, bool DBv = false, typename TQ = float,
          bool P8v = false, bool KVSv = false, uint32_t GVv = 0>
static int pd_spec_fa_krs_go(dim3 grid, void* stream,
    const TQ* q, const __half* pk, const __half* pv, float* oo, float* ml,
    const unsigned int* pos, const unsigned int* slots, const uint32_t* bt,
    uint32_t bps, uint32_t nh, uint32_t nkv, uint32_t hd, uint32_t kvd,
    uint32_t swa, uint32_t ns, uint32_t rows, uint32_t k1, float scale,
    uint32_t Mp, uint32_t fill_sms = 0u) {
    // engagement witness for the krs hoist: once per instantiation - the
    // krs route has no other serve-log fingerprint, and the A/B law says
    // prove the arm fired before reading a leg as a verdict.
    static bool krs_w = false;
    if (!krs_w) {
        krs_w = true;
        fprintf(stderr, "[spec-krs] PT=%u TPW=%u HD=%u VR=%d QK8=%d DBK=%d P8=%d KVS=%d\n",
                PT, TPW, HDv, (int)VRv, (int)QK8v, (int)DBv, (int)P8v, (int)KVSv);
    }
    constexpr uint32_t PTv = (PT + 15u) & ~15u;
    constexpr uint32_t NBUF = DBv ? 2u : 1u;
    const uint32_t hp = hd + 8u;
    // QK8: Q staged e4m3 at the byte pitch (hd+16) instead of f16 (2*(hd+8));
    // P8: the P strip is e4m3 bytes at the [PTv+16] pitch instead of f16
    const uint32_t smem = (QK8v ? Mp * (hd + 16u) : Mp * hp * 2u)
        + NBUF * PT * (hd + 16u)
        + (VRv ? NBUF * PTv * (hd + 16u) : PTv * hp * 2u)
        + Mp * (PT + 1u) * 4u + 3u * Mp * 4u
        + (P8v ? Mp * (PTv + 16u) : Mp * (PT + 8u) * 2u);
    // Shared memory is not what caps krs decode occupancy on B200.
    // The elected q36 arm asks 42240 B/CTA, under the 48 KB opt-in threshold,
    // so this branch never fires for it -- and the profiler duly reported
    // launch__occupancy_limit_shared_mem = 1, which reads like the cap. It is
    // not: requesting cudaSharedmemCarveoutMaxShared moves that metric 1 -> 5
    // and leaves sm__warps_active at 11.71% and c32 throughput unchanged
    // (2142 against a 2159 baseline). Registers were the real cap; see
    // PD_FA_KRS_OCC in attn/decode.cuh. Second time this metric has misled a
    // B200 occupancy hunt -- it is a static model, not a measurement.
    static uint32_t hw = 0;
    if (smem > 48u * 1024u && smem > hw) {
        cudaFuncSetAttribute((const void*)pd_attn_spec_fa_krs_kernel<PT, TPW, HDv, GLv, VRv, QK8v, DBv, TQ, P8v, KVSv, GVv>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        hw = smem;
    }
    pd_pdl_go(pd_attn_spec_fa_krs_kernel<PT, TPW, HDv, GLv, VRv, QK8v, DBv, TQ, P8v, KVSv, GVv>, grid, 256, smem,
              (cudaStream_t)stream, q, pk, pv, oo, ml, pos, slots, bt, bps,
              nh, nkv, hd, kvd, swa, ns, rows, k1, scale, fill_sms);
    return pd_launch_status();
}
// QK8 election: e4m3-Q scores on the krs GLB arm - default on
// for the hd512 geometry (GLB sp4 -6.4%, and the attention band +27.6%
// at depth). e4m3-Q rounding class
// (v9q precedent): serve acceptance is the gate.
// PADDOCK_NO_SPEC_QK8=1 kills.
static int pd_spec_qk8() {
    static int v = -1;
    if (v < 0) v = pd_env("PADDOCK_NO_SPEC_QK8") ? 0 : 1;
    return v;
}
// Does this device have the e4m3 tensor-core mma the QK8/P8 krs arms are built
// on?
//
// This is a CORRECTNESS gate, not a tuning one. The QK8 score block and the P8
// PV block in pd_attn_spec_fa_krs_kernel are both
// `#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)` with no #else - the
// `} else {` next to them belongs to the `if constexpr`, not to the #if. So on
// an older arch those instantiations compile to a kernel that accumulates
// nothing: d[] stays {0,0,0,0}, o_acc stays zero, and it stores zeros. The
// launch SUCCEEDS, pd_launch_status() returns 0, and nothing upstream can tell.
// Measured on sm_86: 0 of 196608 output floats written, while the fallthrough
// walk on the identical buffers is 5.96e-8 from the reference.
//
// Two elections reach those instantiations and neither used to check the arch:
// the dense decode fa8 arm and the spec verify fa6 arm, both fp8 + hd256 + G=6.
// Both now call this, and both simply fall through to a route that needs no
// e4m3 mma. Losing the arm below sm_89 costs nothing real - the hardware cannot
// run its instructions either way.
static bool pd_fp8_mma_ok() {
    static int ok = -1;
    if (ok < 0) {
        int dev = 0, ma = 0, mi = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&ma, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&mi, cudaDevAttrComputeCapabilityMinor, dev);
        ok = (ma > 8 || (ma == 8 && mi >= 9)) ? 1 : 0;
        if (!ok) {
            fprintf(stderr,
                    "[fp8-mma] sm_%d%d has no e4m3 mma - the QK8/P8 krs arms "
                    "stay unelected (they would store zeros)\n", ma, mi);
        }
    }
    return ok == 1;
}
// P8 election: e4m3-P PV on the krs GLB arm, stacked on QK8
// over the VR+DBK structure - default on for the hd512 geometry only
// (GLB fin 265.2 vs elected xV 287.5 = -7.8%, sp4 -3.6%;
// SWA is class-neutral at occ 3 and stays f16-P). The DBK overlap this
// occ-1 arm was built for finally elects here because the xV arm
// structurally can't DBK (its staging strip aliases s_v). e4m3-P rounding
// class (the industry default under fp8 KV): serve acceptance is the
// gate. PADDOCK_NO_SPEC_P8=1 kills (falls back to
// the xV election).
static int pd_spec_p8() {
    static int v = -1;
    if (v < 0) v = pd_env("PADDOCK_NO_SPEC_P8") ? 0 : 1;
    return v;
}
// KVS election: K/V split-commit walk on the krs GLB P8+DBK
// arm - scores wait only on K, V's wait defers behind score+softmax
// (GLB fin -3.5%, sp4 -2.5%, BITEQ everywhere; SWA neutral
// at occ 3 so the SWA arms stay unsplit). BIT-equal - no acceptance
// class change. PADDOCK_NO_SPEC_KVS=1 kills (single-group walk restored).
static int pd_spec_kvs() {
    static int v = -1;
    if (v < 0) v = pd_env("PADDOCK_NO_SPEC_KVS") ? 0 : 1;
    return v;
}

// spec-FA padding A/B (-07-21): padded strides are the default
// (bit-identical, 2.4x on the prefill twin); PADDOCK_SPEC_FA_NOPAD=1 pins
// the original layout
static bool pd_spec_fa_nopad() {
    static int v = -1;
    if (v < 0) v = pd_env("PADDOCK_SPEC_FA_NOPAD") ? 1 : 0;
    return v == 1;
}
static int pd_spec_fa_mode() {
    static const int m = [] {
        if (pd_env("PADDOCK_NO_SPEC_FA")) return 0;
        if (pd_env("PADDOCK_SPEC_FA")) return 1;
        int dev = 0, cc = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&cc, cudaDevAttrComputeCapabilityMajor, dev);
        return cc == 10 ? 1 : 0;
    }();
    return m;
}

// LCO election: in-kernel last-CTA-out combine on the krs spec-FA
// arms - the separate combine launch (and its PDL-wait span) disappears for
// the two gemma4 F8 geometries. Opt-in while gating: PADDOCK_SPEC_LCO=1;
// the engine mirrors the env and skips its combine call when the LCO export
// takes the launch. Bit-identical to partial+combine (BITEQ on both arms).
static int pd_spec_lco() {
    static int v = -1;
    // PADDOCK_SPEC_LCO_POS: the engine elects per-tick by
    // position band and only calls this entry when elected - the env
    // here is just the enable, the engine is the router.
    if (v < 0)
        v = (pd_env("PADDOCK_SPEC_LCO") || pd_env("PADDOCK_SPEC_LCO_POS")) ? 1 : 0;
    return v;
}
template <uint32_t PT, uint32_t TPW, uint32_t HDv, uint32_t GLv>
static int pd_spec_fa_lco_go(dim3 grid, void* stream,
    const float* q, const __half* pk, const __half* pv, float* oo, float* ml,
    const unsigned int* pos, const unsigned int* slots, const uint32_t* bt,
    uint32_t bps, uint32_t nh, uint32_t nkv, uint32_t hd, uint32_t kvd,
    uint32_t swa, uint32_t ns, uint32_t rows, uint32_t k1, float scale,
    uint32_t Mp, const float* sinks, float* out_f, unsigned int* tickets) {
    constexpr uint32_t PTv = (PT + 15u) & ~15u;
    const uint32_t hp = hd + 8u;
    const uint32_t smem = Mp * hp * 2u + PT * (hd + 16u) + PTv * hp * 2u
        + Mp * (PT + 1u) * 4u + 3u * Mp * 4u + Mp * (PT + 8u) * 2u;
    static uint32_t hw = 0;
    if (smem > 48u * 1024u && smem > hw) {
        cudaFuncSetAttribute((const void*)pd_attn_spec_fa_lco_kernel<PT, TPW, HDv, GLv>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        hw = smem;
    }
    pd_pdl_go(pd_attn_spec_fa_lco_kernel<PT, TPW, HDv, GLv>, grid, 256, smem,
              (cudaStream_t)stream, q, pk, pv, oo, ml, pos, slots, bt, bps,
              nh, nkv, hd, kvd, swa, ns, rows, k1, scale, sinks, out_f, tickets);
    return pd_launch_status();
}

PD_EXPORT
int pd_attn_spec_lco_paged(const void* q, const void* pool_k, const void* pool_v,
                           void* out_o, void* out_ml, const void* sinks,
                           void* out_f, void* tickets, const void* positions,
                           const void* slots, const void* block_tables,
                           uint32_t blocks_per_slot, uint32_t n_heads,
                           uint32_t n_kv_heads, uint32_t head_dim, uint32_t kv_dim,
                           uint32_t swa_window, uint32_t n_splits, uint32_t rows,
                           uint32_t k1, float scale, uint32_t kv_dtype, void* stream) {
    // LCO entry: krs geometries only, splits >= 2 (one split takes
    // the FIN route which already skips combine). Return -2 = not covered -
    // the caller runs the partial+combine chain. Predicates mirror
    // pd_attn_spec_batch_paged's krs elections for the two gemma4 F8 serving
    // geometries; anything else stays on the proven path.
    if (rows == 0 || n_splits < 2u || n_splits > 16u) return -2;
    if (!pd_spec_lco() || !pd_spec_fa_mode() || pd_spec_fa_nopad() || !pd_spec_fa_krs())
        return -2;
    if (kv_dtype != PD_KV_FP8_E4M3) return -2;
    const uint32_t group = n_kv_heads ? n_heads / n_kv_heads : 0;
    if (k1 == 0 || k1 > 8u || rows % k1 != 0u || group == 0
        || n_heads != n_kv_heads * group || rows / k1 < 4u)
        return -2;
    static int fa_max_smem = -1;
    if (fa_max_smem < 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        if (cudaDeviceGetAttribute(&fa_max_smem,
                cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess)
            fa_max_smem = 48 * 1024;
    }
    const uint32_t M = k1 * group;
    const uint32_t Mp = ((M + 15u) / 16u) * 16u;
    dim3 fgrid(n_kv_heads, rows / k1, n_splits);
    if (head_dim == 256u && group == 2u && M <= 64u) {
        static int krs_pt = -1;
        if (krs_pt < 0) {
            const char* e = pd_env("PADDOCK_SPEC_KR_PT");
            krs_pt = e ? atoi(e) : 32;
        }
        if (krs_pt == 40)
            return pd_spec_fa_lco_go<40u, 1u, 256u, 1u>(fgrid, stream,
                (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                (const unsigned int*)slots, (const uint32_t*)block_tables,
                blocks_per_slot, n_heads, n_kv_heads, head_dim, kv_dim,
                swa_window, n_splits, rows, k1, scale, Mp,
                (const float*)sinks, (float*)out_f, (unsigned int*)tickets);
        return pd_spec_fa_lco_go<32u, 1u, 256u, 1u>(fgrid, stream,
            (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
            (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
            (const unsigned int*)slots, (const uint32_t*)block_tables,
            blocks_per_slot, n_heads, n_kv_heads, head_dim, kv_dim,
            swa_window, n_splits, rows, k1, scale, Mp,
            (const float*)sinks, (float*)out_f, (unsigned int*)tickets);
    }
    if (head_dim == 512u && group == 8u && M <= 32u) {
        const uint32_t hp = head_dim + 8u;
        const uint32_t smem32 = Mp * hp * 2u + 32u * (head_dim + 16u) + 32u * hp * 2u
            + Mp * 33u * 4u + 3u * Mp * 4u + Mp * 40u * 2u;
        if (smem32 <= (uint32_t)fa_max_smem)
            return pd_spec_fa_lco_go<32u, 4u, 512u, 3u>(fgrid, stream,
                (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                (const unsigned int*)slots, (const uint32_t*)block_tables,
                blocks_per_slot, n_heads, n_kv_heads, head_dim, kv_dim,
                swa_window, n_splits, rows, k1, scale, Mp,
                (const float*)sinks, (float*)out_f, (unsigned int*)tickets);
    }
    return -2;
}

PD_EXPORT
int pd_attn_decode_batch_partial_paged(const void* q, const void* pool_k, const void* pool_v,
                                       void* out_o, void* out_ml, const void* positions,
                                       const void* slots, const void* block_tables,
                                       uint32_t blocks_per_slot, uint32_t n_heads,
                                       uint32_t n_kv_heads, uint32_t head_dim, uint32_t kv_dim,
                                       uint32_t swa_window, uint32_t n_splits, uint32_t batch,
                                       float scale, uint32_t kv_dtype, void* stream) {
    if (n_heads == 0 || batch == 0 || n_splits == 0) return 0;
    dim3 grid(n_heads, batch, n_splits);
    uint32_t attn_nth = head_dim > 256 ? head_dim : 256;
    uint32_t attn_smem = (uint32_t)PD_ATTN_TILE_SMEM(head_dim);
    static uint32_t smem_set_pp = 0;
    if (smem_set_pp == 0) {
        pd_prefer_max_shared(pd_attn_decode_batch_partial_paged_kernel<__nv_fp8_e4m3>);
        pd_prefer_max_shared(pd_attn_decode_batch_partial_paged_kernel<__half>);
        pd_prefer_max_shared(pd_attn_decode_batch_partial_gqa_paged_kernel<__nv_fp8_e4m3, 16u>);
        pd_prefer_max_shared(pd_attn_decode_batch_partial_gqa_paged_kernel<__half, 16u>);
        pd_prefer_max_shared(pd_attn_decode_batch_partial_gqa_paged_kernel<__nv_fp8_e4m3, 32u>);
        pd_prefer_max_shared(pd_attn_decode_batch_partial_gqa_paged_kernel<__half, 32u>);
        smem_set_pp = 1;
    }
    if (attn_smem > 48u * 1024u && attn_smem > smem_set_pp) {
        cudaFuncSetAttribute((const void*)pd_attn_decode_batch_partial_paged_kernel<__nv_fp8_e4m3>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, attn_smem);
        cudaFuncSetAttribute((const void*)pd_attn_decode_batch_partial_paged_kernel<__half>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, attn_smem);
        smem_set_pp = attn_smem;
    }
    // GQA-fused paging (P3b-2): one block per KV head serves the whole q-group
    // (group_size-x less KV traffic). Bit-identical partials to the plain paged
    // kernel; PD_NO_GQA_FUSE pins the plain per-q-head grid.
    // n_kv_heads>=2 (was >=4): qwen3.6 is GQA 8:1 (n_kv_heads=2, head_dim=256).
    // The >=4 heuristic assumed the (n_kv_heads,batch,n_splits) grid needed >=4
    // KV heads to fill the die, but the partial kernel only runs when splitting
    // engages (n_splits=32 on a >=128-SM die), so even n_kv_heads=2 gives
    // 2*batch*32 blocks - ample. MEASURED (35B shape): the
    // slow per-q-head kernel this excluded was 34.5% of decode;
    // the GQA kernel is BIT-EXACT and 2.7-5.2x faster at B=8..32 (capture pivot,
    const uint32_t group = n_kv_heads ? n_heads / n_kv_heads : 1;
    static int no_gqa_p = -1;
    if (no_gqa_p < 0) no_gqa_p = pd_env("PD_NO_GQA_FUSE") ? 1 : 0;
    // Dense-decode FA route (PADDOCK_DENSE_FA=1, A/B): the spec FA kernel at
    // k1=1 - mma scores + mma o vs the GQA walk's serial scalar tiles. Same
    // partial layout ((head*rows + row)*splits + s), same combine.
    static int dense_fa = -1;
    if (dense_fa < 0) dense_fa = (pd_env("PADDOCK_DENSE_FA") && pd_spec_fa_mode()) ? 1 : 0;
    if (dense_fa && kv_dtype != PD_KV_FP8_E4M3 && head_dim >= 64u && head_dim <= 512u
        && (head_dim & 63u) == 0u && group >= 2u && group <= 8u && batch >= 4u
        && n_heads == n_kv_heads * group) {
        const uint32_t M = group;
        const uint32_t Mp = 16u;
        constexpr uint32_t PT = 32u;
        const bool fpad = !pd_spec_fa_nopad();
        const uint32_t hp = head_dim + (fpad ? 8u : 0u);
        const uint32_t smem = Mp * hp * 2u + 4u * PT * hp * 2u
            + Mp * (PT + (fpad ? 1u : 0u)) * 4u + 3u * Mp * 4u
            + Mp * (PT + (fpad ? 8u : 0u)) * 2u;
        static uint32_t dfa_set = 0;
        if (smem > 48u * 1024u && smem > dfa_set) {
            cudaFuncSetAttribute((const void*)pd_attn_spec_fa_kernel<PT, 1u>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            cudaFuncSetAttribute((const void*)pd_attn_spec_fa_kernel<PT, 1u, true, false>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            dfa_set = smem;
        }
        (void)M;
        dim3 fgrid(n_kv_heads, batch, n_splits);
        if (!fpad) {
            pd_attn_spec_fa_kernel<PT, 1u, true, false><<<fgrid, 256, smem, (cudaStream_t)stream>>>(
                (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                (const unsigned int*)slots, (const uint32_t*)block_tables,
                blocks_per_slot, n_heads, n_kv_heads, head_dim, kv_dim,
                swa_window, n_splits, batch, 1u, scale);
            return pd_launch_status();
        }
        return pd_spec_fa_go<PT, 1u, true>(fgrid, smem, stream,
            (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
            (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
            (const unsigned int*)slots, (const uint32_t*)block_tables,
            blocks_per_slot, n_heads, n_kv_heads, head_dim, kv_dim,
            swa_window, n_splits, batch, 1u, scale);
    }
    // fp8 dense-decode FA (q36, ELECTED DEFAULT): the krs
    // P8+QK8+VR class at k1=1 for the 24q/4kv/hd256 geometry - G=6 via the
    // GV template override. Replaces the scalar GQA walk (~150us/layer at
    // ctx~1150, ~0.5 TB/s): isolated -28..-50% at ctx>=1150 with adaptive
    // splits; serve +2.9/+4.0/+3.4% on 128x2048/imax/2048x128 c32, neutral
    // short-ctx. Numerics: the P8+QK8 class - P weights e4m3-rounded per
    // tile (the industry fp8 paths ship the same P.to(fp8)); a P8-faithful
    // oracle gates 1.4e-4 at the once-suspect configs, and the
    // gemma hd512 GLB default has long shipped this class. Serve
    // coherence gate passed. Kill: PADDOCK_NO_DENSE_FA8.
    static int dense_fa8 = -1;
    if (dense_fa8 < 0) dense_fa8 = pd_env("PADDOCK_NO_DENSE_FA8") ? 0 : 1;
    if (dense_fa8 && pd_fp8_mma_ok() && kv_dtype == PD_KV_FP8_E4M3
        && head_dim == 256u
        && group == 6u && n_heads == n_kv_heads * group) {
        dim3 fgrid(n_kv_heads, batch, n_splits);
        // Die-fill floor for the GV arm's adaptive split - the kernel
        // needs the SM count to know whether its live grid covers a wave. Host
        // side, latched once: a device constant keeps it graph-replay safe
        // (the split framing still recomputes from device positions).
        // PADDOCK_NO_FA_FILL=1 passes 0 and restores the context-only clamp.
        static uint32_t fa_fill = 0xffffffffu;
        if (fa_fill == 0xffffffffu) {
            if (pd_env("PADDOCK_NO_FA_FILL")) {
                fa_fill = 0u;
            } else {
                int dev = 0, nsm = 0;
                cudaGetDevice(&dev);
                cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
                fa_fill = nsm > 0 ? (uint32_t)nsm : 0u;
                fprintf(stderr, "[fa-fill] die-fill split floor armed (%u SMs)\n", fa_fill);
            }
        }
        return pd_spec_fa_krs_go<32u, 2u, 256u, 0u, true, true, true, float, true, true, 6u>(
            fgrid, stream,
            (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
            (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
            (const unsigned int*)slots, (const uint32_t*)block_tables,
            blocks_per_slot, n_heads, n_kv_heads, head_dim, kv_dim,
            swa_window, n_splits, batch, 1u, scale, 16u, fa_fill);
    }
    // head-packed lagd (later WIDENED): the fp8
    // KV hd128 arm. Started as nemotron's full-attention shape (G=16, no
    // window), which rode vec8's scalar per-q-head walk - profiled at 4.7%
    // DRAM, latency-bound, where a head-packed mma kernel (FlashInfer's
    // shape) streams KV once per group. lagd is that structure and G=16
    // fills its m16 tile exactly.
    //
    // The original gate said `group >= 9 && swa_window == 0`, and that cost
    // real ground: decode performance sorts by ARM, not by model. Every mma
    // arm lands within ~5% of the best kernel available; every shape falling
    // through to a scalar walk is 2-5x behind, and G=8 plus every windowed
    // shape had no mma arm at all.
    //
    // Both clauses were conservatism, not capability, and the kernel says so:
    //   - WINDOW: lagd computes `first_pos` from swa_window with the same
    //     expression as every other arm and stages from it. A decode row's
    //     window is a contiguous RANGE, not a mask, so there is no masking to
    //     add. The f16 KV lagd arm below has served laguna's windowed layers
    //     for a long time - the windowed path is production-proven, it was only
    //     the fp8 twin that was fenced off.
    //   - G=8: the base (WIDE=false) build is the G<=8 build - q rows pad to
    //     16 mma rows and every store guards `rr < G`. It only ever lacked
    //     the fp8 instantiation, because nemotron did not need one.
    // So this is an instantiation + gate widening, the same class of change
    // that lifted the equivalent prefill shapes out of the scalar band.
    //
    // WIDE is picked by group: >8 uses the frag half-1 rows as real heads,
    // ==8 fills half 0 exactly and leaves half 1 zero-padded (running WIDE
    // at G=8 would be correct but would double the mma work on dead rows).
    // pd_pdl_go, not a plain launch: the arms this replaces (vec8, the
    // GQA-fused walk) ride the decode cascade and the laguna chain law
    // stands - arm + launcher are one change. The engine mirrors this
    // election in its split budget (n_kv-based, not nh-based).
    // Kill: PADDOCK_NO_ATTN_HP16 (falls back to vec8 / the GQA walk).
    static int no_hp16 = -1;
    if (no_hp16 < 0) no_hp16 = pd_env("PADDOCK_NO_ATTN_HP16") ? 1 : 0;
    if (!no_hp16 && kv_dtype == PD_KV_FP8_E4M3 && head_dim == 128u
        && group >= 8u && group <= 16u && batch >= 2u
        && n_heads == n_kv_heads * group) {
        static uint32_t hp16_set_p = 0;
        if (hp16_set_p == 0) {
            pd_prefer_max_shared(
                pd_attn_decode_lagd_kernel<32u, 2u, 3u, true, __nv_fp8_e4m3, true>);
            pd_prefer_max_shared(
                pd_attn_decode_lagd_kernel<32u, 2u, 3u, true, __nv_fp8_e4m3, false>);
            hp16_set_p = 1;
        }
        dim3 hgrid(n_kv_heads, batch, n_splits);
        if (group > 8u)
            pd_pdl_go(pd_attn_decode_lagd_kernel<32u, 2u, 3u, true, __nv_fp8_e4m3, true>,
                      hgrid, 256u, PD_LAGD_SMEM_SPLIT, (cudaStream_t)stream,
                      (const float*)q, (const __nv_fp8_e4m3*)pool_k,
                      (const __nv_fp8_e4m3*)pool_v, (float*)out_o, (float*)out_ml,
                      (const unsigned int*)positions, (const unsigned int*)slots,
                      (const uint32_t*)block_tables, blocks_per_slot, 0u, n_heads,
                      n_kv_heads, kv_dim, swa_window, n_splits, scale);
        else
            pd_pdl_go(pd_attn_decode_lagd_kernel<32u, 2u, 3u, true, __nv_fp8_e4m3, false>,
                      hgrid, 256u, PD_LAGD_SMEM_SPLIT, (cudaStream_t)stream,
                      (const float*)q, (const __nv_fp8_e4m3*)pool_k,
                      (const __nv_fp8_e4m3*)pool_v, (float*)out_o, (float*)out_ml,
                      (const unsigned int*)positions, (const unsigned int*)slots,
                      (const uint32_t*)block_tables, blocks_per_slot, 0u, n_heads,
                      n_kv_heads, kv_dim, swa_window, n_splits, scale);
        return pd_launch_status();
    }
    // vec8: the register-resident per-(q-head, split)
    // fp8 walk beats the GQA-fused walk at every swept (ctx, B) cell on the
    // granite hd128/G4 geometry (-43% at serve
    // contexts incl. combine) - the fused walk's DRAM-byte savings don't
    // bind when the hot KV is L2-resident, and its 8-16 live CTAs starve the
    // die. Gate: exactly the granite-4.1 shape (fp8 KV, hd128, G4, no
    // window) - laguna/qwen shapes keep their arms until their own
    // measurements say otherwise. Splits arrive q-head-budgeted from the engine (cap
    // 32). Kill: PADDOCK_NO_ATTN_VEC8.
    static int no_vec8 = -1;
    if (no_vec8 < 0) no_vec8 = pd_env("PADDOCK_NO_ATTN_VEC8") ? 1 : 0;
    // Extended to group>8: the fused walk's group<=8 gate meant
    // laguna's G=9 windowed layers (36 of 48) rode the scalar per-head
    // fallback - vec8 measured -55..-70%
    // vs that fallback across every (ctx, B) cell, window path included
    // (rel 2e-7). Laguna's G=6 full layers stay on the fused walk: vec8 wins
    // short ctx there but loses >=1024 (6-way K re-read leaves L2 residency
    // at the 48-layer footprint) - recorded as a future door, not elected.
    // group>8 is additionally batch-gated (>=2): live c1 legs measured the
    // per-head fallback better at B=1 (127.0 vs 125.4 - the isolated win
    // hides behind the launch train, the granite lesson) while B=4 takes
    // +5.1%. `batch` is a launch parameter, so the split is graph-safe -
    // a geometry election, not a knob.
    //
    // Granite's G4 arm is now also batch-gated to B=1. The old
    // "unconditional (measured B=1 win)" claim came from an isolated
    // microbench - the SERVING pipeline disagrees. Live
    // granite-4.2-8b-nvfp4 decode (8-way + 32-way, ctx 128 and 1.3k,
    // graph-traced): vec8 is a TIE at B=1 (its win hides behind the
    // sampler/GEMV launch train, the same granite lesson as the group>8 arm)
    // and a growing LOSS at batch - the non-vec8 GQA-fused walk below beats it
    // +5..13% at r=8 and +40.8% at r=32/1.3k (the imax regime). The vec8
    // kernel's fixed ~41us/layer floor at r=8 (it runs the KV read at ~240
    // GB/s, ~15% of roof - latency/overhead bound, not bandwidth bound)
    // dominates once several rows share the die. So vec8 stays only where it
    // is free-or-a-tie (B=1); every batched decode row takes the fused walk.
    // Kill (dev A/B only): PADDOCK_NO_ATTN_VEC8.
    if (!no_vec8 && kv_dtype == PD_KV_FP8_E4M3 && head_dim == 128u
        && ((group == 4u && swa_window == 0u && batch < 2u) || (group > 8u && batch >= 2u))
        && n_heads == n_kv_heads * group) {
        dim3 vgrid(n_heads, batch, n_splits);
        // pd_pdl_go, not a plain launch: the kernels this branch replaces
        // (fused walk / per-head fallback) ride the PDL cascade, and the
        // laguna chain pass's law stands - arm + launcher are one change.
        // The plain-launch first cut broke the cascade at laguna's 36 G9
        // layers and read as a real -1.5% c1 drift (127.0 -> 125.1).
        pd_pdl_go(pd_attn_decode_vec8_paged_kernel<128u, 32u>,
            vgrid, 128, 0u, (cudaStream_t)stream,
                (const float*)q, (const __nv_fp8_e4m3*)pool_k,
                (const __nv_fp8_e4m3*)pool_v, (float*)out_o, (float*)out_ml,
                (const unsigned int*)positions, (const unsigned int*)slots,
                (const uint32_t*)block_tables, blocks_per_slot, n_heads,
                n_kv_heads, kv_dim, swa_window, n_splits, scale);
        return pd_launch_status();
    }
    // granite hd128/G4 v9q arm. Sits before the GQA-fused walk
    // arm below deliberately: that arm's head_dim <= 128 gate returns first for
    // this shape, so the v8f8/v9q block further down (hd256/G2 | hd512/G8) is
    // unreachable here -- the first wiring measured c8 +1.2% / c32 0.0% and the
    // in-graph capture showed no v9q launches (the CT=32 lesson: prove the
    // kernel NAME in a capture). Only at n_splits >= 2: the kernel's
    // n_splits == 1 path writes FINAL rows and skips out_ml, which this
    // export's partial+combine consumer cannot take. The engine clamps the
    // split count to 2..4 for this shape. WS=4/MB=1 and the HD-generic smem
    // (STB = HD*32) from the v9q probe arm: B=8 9.3us vs
    // the walk's 17.5, B=32/ctx128 12.3 vs 24.6, rel 2.6e-2 (the v9q numerics
    // class; text gates + serve A/B arbitrate). Kill: PADDOCK_NO_V9Q.
    // n_splits == 1 is admitted since the ns1 rung: the engine then passes the
    // FINAL attention buffer as out_o (the kernel's ns==1 path writes
    // [b][head][hd] rows and skips out_ml) and launches no combine.
    if (kv_dtype == PD_KV_FP8_E4M3 && head_dim == 128u && group == 4u
        && n_heads == n_kv_heads * group && n_splits >= 1u && batch >= 2u
        && !pd_env("PADDOCK_NO_V9Q") && pd_tmap_encode()) {
        struct PdTmEnt9g { const void* p; uint32_t kd; CUtensorMap m; };
        static PdTmEnt9g t9g[64];
        static uint32_t t9gn = 0;
        auto get_tm9g = [&](const void* base) -> const CUtensorMap* {
            for (uint32_t i = 0; i < t9gn; ++i)
                if (t9g[i].p == base && t9g[i].kd == kv_dim) return &t9g[i].m;
            if (t9gn >= 64u) t9gn = 0;
            if (!pd_attn_tmap_kv_f8s(&t9g[t9gn].m, base, kv_dim)) return nullptr;
            t9g[t9gn].p = base; t9g[t9gn].kd = kv_dim;
            return &t9g[t9gn++].m;
        };
        const CUtensorMap* qk = get_tm9g(pool_k);
        const CUtensorMap* qv = qk ? get_tm9g(pool_v) : nullptr;
        if (qk && qv) {
            constexpr uint32_t STB128 = 128u * 32u;
            const uint32_t v9q128_smem = 2u * STB128 + 3u * STB128
                + 16u * (128u + 16u) + 2u * 16u * 48u + 1024u;
            static uint32_t hw128 = 0;
            if (hw128 == 0) {
                cudaFuncSetAttribute(
                    (const void*)pd_attn_decode_v9q_kernel<128u, 4u, 1u, 0u, 4u>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)v9q128_smem);
                hw128 = 1;
            }
            dim3 qgrid(n_kv_heads, batch, n_splits);
            pd_pdl_go(pd_attn_decode_v9q_kernel<128u, 4u, 1u, 0u, 4u>,
                      qgrid, 256, v9q128_smem, (cudaStream_t)stream,
                      *qk, *qv, (const float*)q, (float*)out_o, (float*)out_ml,
                      (const unsigned int*)positions, (const unsigned int*)slots,
                      (const uint32_t*)block_tables, blocks_per_slot,
                      kv_dim, swa_window, n_splits, scale);
            return pd_launch_status();
        }
    }
    if (!no_gqa_p && head_dim <= 128u && group >= 2u && group <= 8u && n_kv_heads >= 2u
        && n_heads == n_kv_heads * group) {
        dim3 ggrid(n_kv_heads, batch, n_splits);
        const uint32_t gqa_smem = (uint32_t)PD_GQA_SMEM(group, head_dim, 32u);
        static uint32_t gqa32_set_p = 0;
        if (gqa_smem > 48u * 1024u && gqa_smem > gqa32_set_p) {
            cudaFuncSetAttribute((const void*)pd_attn_decode_batch_partial_gqa_paged_kernel<__nv_fp8_e4m3, 32u>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, gqa_smem);
            cudaFuncSetAttribute((const void*)pd_attn_decode_batch_partial_gqa_paged_kernel<__half, 32u>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, gqa_smem);
            gqa32_set_p = gqa_smem;
        }
        // hd128 f16 gets the tf32 score mma inside the TILE-32 walk (two
        // 16-token slabs per tile - see the kernel's sc_mma note; the walk
        // itself is TOTAL-ISSUE bound: on laguna it ran
        // 19.7 µs 512-key SWA / 49.4 µs full layers at ~1.9k ctx, where a
        // fused decode-attention kernel does ~7.7 µs).
        // PADDOCK_NO_ATTN_MMA128 pins the scalar
        // walk via the SCM=false twin instantiation (numeric-class escape).
        static int no_mma128 = -1;
        if (no_mma128 < 0) no_mma128 = pd_env("PADDOCK_NO_ATTN_MMA128") ? 1 : 0;
        if (no_mma128 && head_dim == 128u && kv_dtype != PD_KV_FP8_E4M3) {
            static uint32_t scm_set = 0;
            if (gqa_smem > 48u * 1024u && gqa_smem > scm_set) {
                cudaFuncSetAttribute(
                    (const void*)pd_attn_decode_batch_partial_gqa_paged_kernel<__half, 32u, false, false>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, gqa_smem);
                scm_set = gqa_smem;
            }
            pd_attn_decode_batch_partial_gqa_paged_kernel<__half, 32u, false, false>
                <<<ggrid, attn_nth, gqa_smem, (cudaStream_t)stream>>>(
                    (const float*)q, (const __half*)pool_k, (const __half*)pool_v, (float*)out_o,
                    (float*)out_ml, (const unsigned int*)positions, (const unsigned int*)slots,
                    (const uint32_t*)block_tables, blocks_per_slot, n_heads, n_kv_heads, head_dim,
                    kv_dim, swa_window, n_splits, scale);
            return pd_launch_status();
        }
        // lagd: the v5-class hd128 partial - scores AND
        // PV on f16 ldmatrix/mma with split planes preserving the walk's
        // numeric classes (src/attn/lagd.cuh). B=1 pair 21.4 -> 16.5 us full /
        // 20.2 -> 15.4-18.2 SWA; B=8 -17..22% - the walk was ~2x its own
        // stage-only floor (barrier+wait stalls at 1 CTA/SM), lagd runs at the
        // floor. The walk stays the PADDOCK_NO_LAGD / PADDOCK_NO_ATTN_MMA128
        // fallback.
        if (pd_attn_lagd() && head_dim == 128u && kv_dtype != PD_KV_FP8_E4M3) {
            static uint32_t lagd_set_p = 0;
            if (lagd_set_p == 0) {
                pd_prefer_max_shared(pd_attn_decode_lagd_kernel<32u, 2u, 3u, true>);
                pd_prefer_max_shared(pd_attn_decode_lagd_kernel<32u, 1u, 1u, true>);
                lagd_set_p = 1;
            }
            if (pd_attn_lagd_f16())
                pd_attn_decode_lagd_kernel<32u, 1u, 1u, true>
                    <<<ggrid, 256u, PD_LAGD_SMEM_F16, (cudaStream_t)stream>>>(
                        (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                        (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                        (const unsigned int*)slots, (const uint32_t*)block_tables,
                        blocks_per_slot, 0u, n_heads, n_kv_heads, kv_dim, swa_window,
                        n_splits, scale);
            else
                pd_attn_decode_lagd_kernel<32u, 2u, 3u, true>
                    <<<ggrid, 256u, PD_LAGD_SMEM_SPLIT, (cudaStream_t)stream>>>(
                        (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                        (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                        (const unsigned int*)slots, (const uint32_t*)block_tables,
                        blocks_per_slot, 0u, n_heads, n_kv_heads, kv_dim, swa_window,
                        n_splits, scale);
            return pd_launch_status();
        }
        // PV-mma hd128 (same pd_attn_pv_mma() latch as the hd256
        // arm): the TILE-32 walk's scalar V fold moves onto tf32 tensor cores
        // - one 16-dim M-tile per warp, four 8-token chunks per tile (the
        // kernel's pv_on arm). Numeric class: hd256's accepted 3-split tf32 P
        // (~33-bit carry), not bit-identical to the scalar fold. hd == 128
        // exactly - hd64 (gpt-oss) has no sc_mma and stays scalar.
        if (pd_attn_pv_mma() && head_dim == 128u && kv_dtype != PD_KV_FP8_E4M3) {
            static uint32_t pv32p_set = 0;
            if (pv32p_set == 0) {
                pd_prefer_max_shared(pd_attn_decode_batch_partial_gqa_paged_kernel<__half, 32u, true>);
                pv32p_set = 1;
            }
            if (gqa_smem > 48u * 1024u && gqa_smem > pv32p_set) {
                cudaFuncSetAttribute(
                    (const void*)pd_attn_decode_batch_partial_gqa_paged_kernel<__half, 32u, true>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, gqa_smem);
                pv32p_set = gqa_smem;
            }
            pd_attn_decode_batch_partial_gqa_paged_kernel<__half, 32u, true>
                <<<ggrid, attn_nth, gqa_smem, (cudaStream_t)stream>>>(
                    (const float*)q, (const __half*)pool_k, (const __half*)pool_v, (float*)out_o,
                    (float*)out_ml, (const unsigned int*)positions, (const unsigned int*)slots,
                    (const uint32_t*)block_tables, blocks_per_slot, n_heads, n_kv_heads, head_dim,
                    kv_dim, swa_window, n_splits, scale);
            return pd_launch_status();
        }
        if (kv_dtype == PD_KV_FP8_E4M3)
            pd_attn_decode_batch_partial_gqa_paged_kernel<__nv_fp8_e4m3, 32u>
                <<<ggrid, attn_nth, gqa_smem, (cudaStream_t)stream>>>(
                    (const float*)q, (const __nv_fp8_e4m3*)pool_k, (const __nv_fp8_e4m3*)pool_v,
                    (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                    (const unsigned int*)slots, (const uint32_t*)block_tables, blocks_per_slot,
                    n_heads, n_kv_heads, head_dim, kv_dim, swa_window, n_splits, scale);
        else
            pd_attn_decode_batch_partial_gqa_paged_kernel<__half, 32u>
                <<<ggrid, attn_nth, gqa_smem, (cudaStream_t)stream>>>(
                    (const float*)q, (const __half*)pool_k, (const __half*)pool_v, (float*)out_o,
                    (float*)out_ml, (const unsigned int*)positions, (const unsigned int*)slots,
                    (const uint32_t*)block_tables, blocks_per_slot, n_heads, n_kv_heads, head_dim,
                    kv_dim, swa_window, n_splits, scale);
        return pd_launch_status();
    }
    // v3 mma-pass dense decode (B200): the GQA walk is
    // TOTAL-ISSUE bound (staging skeleton alone streams 6.0 TB/s; the scalar
    // convert+FMA passes drag it to 2.5). HMMA passes on the same skeleton:
    // SWA hd256 218->137 us, global hd512 509->187 us on the c32 bisect
    // harness. f16 KV only; fp16 q/weights with
    // f32 accumulate - the spec-FA numerics class, serving gates arbitrate.
    // Kill: PADDOCK_NO_ATTN_V3.
    static int no_v3 = -1;
    if (no_v3 < 0) no_v3 = pd_env("PADDOCK_NO_ATTN_V3") ? 1 : 0;
    // v8f8 (KV8): the SWA tile pipeline on e4m3 pools - fp8 TMA
    // staging + in-kernel expansion (see the kernel note). Same smem/grid as
    // v8; its own map cache (byte-width 1, swizzle none). hd512 fp8 stays on
    // the walk fallback until v8ks gets the same arm. Kill: PADDOCK_NO_V8F8.
    if (kv_dtype == PD_KV_FP8_E4M3
        && ((head_dim == 256u && group == 2u) || (head_dim == 512u && group == 8u))
        && n_heads == n_kv_heads * group && !pd_env("PADDOCK_NO_V8F8")
        && pd_tmap_encode()) {
        struct PdTmEnt8 { const void* p; uint32_t kd; CUtensorMap m; };
        static PdTmEnt8 t8cache[64];
        static uint32_t t8n = 0;
        auto get_tm8 = [&](const void* base) -> const CUtensorMap* {
            for (uint32_t i = 0; i < t8n; ++i)
                if (t8cache[i].p == base && t8cache[i].kd == kv_dim)
                    return &t8cache[i].m;
            if (t8n >= 64u) t8n = 0;
            if (!pd_attn_tmap_kv_f8(&t8cache[t8n].m, base, kv_dim)) return nullptr;
            t8cache[t8n].p = base; t8cache[t8n].kd = kv_dim;
            return &t8cache[t8n++].m;
        };
        // v8q: fp8-NATIVE score side (qgmma fragments straight off
        // the swizzled fp8 tiles), V via a single expander warp. Falls to
        // v8f8 (full-expansion) under PADDOCK_NO_V8Q.
        // v9q (QGMMA redesign): 32-key supertiles, fp8 end-to-end,
        // Q-scale folded into the cast (real-Q e4m3 saturation fix). The
        // KV8-mode DEFAULT: beats v8f8 in production (c32 1149.7 vs 1091.7,
        // dc32 1879.0 vs 1765.5, c8 712.6 vs 699.9), coherent, burst-clean.
        // Kill: PADDOCK_NO_V9Q -> v8f8.
        static int v9q = -1;
        if (v9q < 0) v9q = pd_env("PADDOCK_NO_V9Q") ? 0 : 1;
        if (v9q) {
            struct PdTmEnt9 { const void* p; uint32_t kd; CUtensorMap m; };
            static PdTmEnt9 t9[64];
            static uint32_t t9n = 0;
            auto get_tm9 = [&](const void* base) -> const CUtensorMap* {
                for (uint32_t i = 0; i < t9n; ++i)
                    if (t9[i].p == base && t9[i].kd == kv_dim) return &t9[i].m;
                if (t9n >= 64u) t9n = 0;
                if (!pd_attn_tmap_kv_f8s(&t9[t9n].m, base, kv_dim)) return nullptr;
                t9[t9n].p = base; t9[t9n].kd = kv_dim;
                return &t9[t9n++].m;
            };
            const CUtensorMap* qk = get_tm9(pool_k);
            const CUtensorMap* qv = qk ? get_tm9(pool_v) : nullptr;
            if (qk && qv) {
                dim3 qgrid(n_kv_heads, batch, n_splits);
                // MB occupancy A/B: the unbounded build reg-caps at
                // 2 blocks/SM SWA / 1 GLB (106 / 212 regs) - half the smem
                // design's co-residency. Env-dispatched min-blocks variants.
#define PD_V9Q_GO(HDv, Gv, MBv, HW, SM)                                       \
    {                                                                         \
        static uint32_t HW = 0;                                               \
        if (HW == 0) {                                                        \
            cudaFuncSetAttribute(                                             \
                (const void*)pd_attn_decode_v9q_kernel<HDv, Gv, MBv>,         \
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(SM));      \
            HW = 1;                                                           \
        }                                                                     \
        pd_pdl_go(pd_attn_decode_v9q_kernel<HDv, Gv, MBv>, qgrid, 256, (SM), \
                  (cudaStream_t)stream,                                       \
                  *qk, *qv, (const float*)q, (float*)out_o, (float*)out_ml,   \
                  (const unsigned int*)positions, (const unsigned int*)slots, \
                  (const uint32_t*)block_tables, blocks_per_slot,             \
                  kv_dim, swa_window, n_splits, scale);                       \
        return pd_launch_status();                                            \
    }
                if (head_dim == 256u) {
                    const uint32_t v9q_smem = 2u * 8192u + 3u * 8192u
                        + 16u * (256u + 16u) + 2u * 16u * 48u + 1024u;
                    // MB=4 default (sweep): 89.5-95.3us at splits 2-4
                    // vs f16 v8's 93.2-100.2 - the 64-reg spill is fully
                    // covered by 4-block co-residency. (MB sweep: 1: 135-157,
                    // 2: 99-107, 3: 95-108, 4: 90-95.) GLB keeps MB=1: its
                    // 212-reg body spills catastrophically under a cap.
                    static int mb9 = -1;
                    if (mb9 < 0) {
                        const char* e = pd_env("PADDOCK_V9Q_MB");
                        mb9 = e ? atoi(e) : 4;
                        if (mb9 < 1 || mb9 > 4) mb9 = 4;
                    }
                    // VD arm: the engine registers a dim-major
                    // twin pool (pd_vdim_register); the plain (16,256)-box
                    // map turns every PV B fragment into one u32 load.
                    // SWA/hd256 only (a (16,512) box exceeds TMA's row cap);
                    // GLB layers keep VD=0 + the legacy pool. Probe gate.
                    static int vdim9 = -1;
                    if (vdim9 < 0) vdim9 = pd_env("PADDOCK_VDIM") ? 1 : 0;
                    if (vdim9 && pd_vdim_base) {
                        // pools are per LAYER: cache one map per registered
                        // base (the engine re-registers before each layer's
                        // call) - a single static would read another layer's
                        // V after a switch
                        struct PdVdEnt { const void* p; CUtensorMap m; };
                        static PdVdEnt vdc[64];
                        static uint32_t vdn = 0;
                        const CUtensorMap* qvd = nullptr;
                        for (uint32_t i = 0; i < vdn; ++i)
                            if (vdc[i].p == pd_vdim_base) { qvd = &vdc[i].m; break; }
                        if (!qvd && vdn < 64u
                            && pd_attn_tmap_vdim(&vdc[vdn].m, pd_vdim_base,
                                                 1ull << 30, 256u)) {
                            vdc[vdn].p = pd_vdim_base;
                            qvd = &vdc[vdn++].m;
                        }
                        if (qvd) {
#define PD_V9Q_GO_VD(MBv, HW)                                                     {                                                                                 static uint32_t HW = 0;                                                       if (HW == 0) {                                                                    cudaFuncSetAttribute(                                                             (const void*)pd_attn_decode_v9q_kernel<256u, 2u, MBv, 1u>,                    cudaFuncAttributeMaxDynamicSharedMemorySize,                                  (int)(v9q_smem));                                                         HW = 1;                                                                   }                                                                             pd_pdl_go(pd_attn_decode_v9q_kernel<256u, 2u, MBv, 1u>, qgrid, 256,                     (v9q_smem), (cudaStream_t)stream,                                             *qk, *qvd, (const float*)q, (float*)out_o, (float*)out_ml,                     (const unsigned int*)positions, (const unsigned int*)slots,                   (const uint32_t*)block_tables, blocks_per_slot,                               kv_dim, swa_window, n_splits, scale);                               return pd_launch_status();                                                }
                            if (mb9 == 4) PD_V9Q_GO_VD(4u, hwv4)
                            if (mb9 == 3) PD_V9Q_GO_VD(3u, hwv3)
                            if (mb9 == 2) PD_V9Q_GO_VD(2u, hwv2)
                            PD_V9Q_GO_VD(1u, hwv1)
#undef PD_V9Q_GO_VD
                        }
                    }
                    // v9q2 ST64 arm: 64-key TMA/handshake windows,
                    // per-32 fold sequence VERBATIM -> BIT-IDENTICAL to v9q
                    // (probe-gated: memcmp 0/196608 diffs,
                    // -4.9% at the c32-nospec shape). smem 71KB -> 3 CTAs/SM,
                    // which still holds the whole (16,b,1) decode grid
                    // resident. DEFAULT off: the kernel wins its own
                    // wall - probe -4.9%, serve capture -11.6% on the hot
                    // 16x24x1 grid with FLAT neighbors, bit-identical (probe
                    // memcmp 0/196608) - but the HARNESS CELL loses with it
                    // on: nospec 2672.02 -> 2575.82 (-3.6%), all-ITL
                    // (10.5-class -> 10.87, TTFT flat), clocks guard-verified
                    // 1965 both runs. A kernel-level win that loses the cell
                    // is a loss; the suspected mechanism is launch-edge slot
                    // pressure (3 CTAs/SM leaves less PDL-overlap headroom
                    // than v9q's 4) and is not yet confirmed. Re-flip only on
                    // a harness A/B >= the v9q cell.
                    static int st64 = -1;
                    if (st64 < 0) st64 = pd_env("PADDOCK_V9Q_ST64") ? 1 : 0;
                    if (st64) {
                        const uint32_t v9q2_smem = 4u * 2u * 8192u
                            + 16u * (256u + 16u) + 2u * 16u * 48u + 1024u;
                        static uint32_t hw9q2 = 0;
                        if (hw9q2 == 0) {
                            cudaFuncSetAttribute(
                                (const void*)pd_attn_decode_v9q2_kernel<256u, 2u, 3u>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                (int)v9q2_smem);
                            hw9q2 = 1;
                        }
                        pd_pdl_go(pd_attn_decode_v9q2_kernel<256u, 2u, 3u>,
                                  qgrid, 256, v9q2_smem, (cudaStream_t)stream,
                                  *qk, *qv, (const float*)q, (float*)out_o,
                                  (float*)out_ml, (const unsigned int*)positions,
                                  (const unsigned int*)slots,
                                  (const uint32_t*)block_tables, blocks_per_slot,
                                  kv_dim, swa_window, n_splits, scale);
                        return pd_launch_status();
                    }
                    // WS4 arm: 4 score + 4 PV warps.
                    // The 2S/6V split left the score stage ~2x the V stage
                    // per supertile and barrier stall dominant (3.13);
                    // balancing the classes + freeing the dead padding-row
                    // accumulators (o_dead) = 20.7 -> 18.7us (-9.7%)
                    // at UNCHANGED occ 4/SM (3 reps).
                    // NUMERICS-CLASS: the l/psum fold tree differs (maxrel
                    // 4.8e-07 vs WS=2, 41% of outputs still bit-equal).
                    // DEFAULT on: a harness A/B is >= base
                    // on both lanes (nospec 2705.85 vs 2702.16, spec 3885.21
                    // vs 3800.16, 3 reps each, clock-guard green) - meets the
                    // v9q2-epilogue re-flip condition; distinct 32/32
                    // on-topic at c32; the 4.8e-07 bound is ~4 orders under
                    // the lane's kv-fp8 quantization noise. PADDOCK_V9Q_WS4=0
                    // reverts to the WS=2 layout.
                    static int ws4 = -1;
                    if (ws4 < 0) {
                        const char* e = pd_env("PADDOCK_V9Q_WS4");
                        ws4 = e ? atoi(e) : 1;
                    }
                    if (ws4) {
                        static uint32_t hww4 = 0;
                        if (hww4 == 0) {
                            cudaFuncSetAttribute(
                                (const void*)pd_attn_decode_v9q_kernel<256u, 2u, 4u, 0u, 4u>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                (int)v9q_smem);
                            hww4 = 1;
                        }
                        pd_pdl_go(pd_attn_decode_v9q_kernel<256u, 2u, 4u, 0u, 4u>,
                                  qgrid, 256, v9q_smem, (cudaStream_t)stream,
                                  *qk, *qv, (const float*)q, (float*)out_o,
                                  (float*)out_ml, (const unsigned int*)positions,
                                  (const unsigned int*)slots,
                                  (const uint32_t*)block_tables, blocks_per_slot,
                                  kv_dim, swa_window, n_splits, scale);
                        return pd_launch_status();
                    }
                    if (mb9 == 4) PD_V9Q_GO(256u, 2u, 4u, hw9s4, v9q_smem)
                    if (mb9 == 3) PD_V9Q_GO(256u, 2u, 3u, hw9s3, v9q_smem)
                    if (mb9 == 2) PD_V9Q_GO(256u, 2u, 2u, hw9s2, v9q_smem)
                    PD_V9Q_GO(256u, 2u, 1u, hw9s1, v9q_smem)
                }
                // GLB arm: the same template at <512,8> - 16 KB
                // supertiles, ~91 KB smem -> 2 blocks/SM, which covers the
                // small GLB grid (4 kv-heads x batch x splits) in one wave
                const uint32_t v9g_smem = 2u * 16384u + 3u * 16384u
                    + 16u * (512u + 16u) + 2u * 16u * 48u + 1024u;
                static int mb9g = -1;
                if (mb9g < 0) {
                    const char* e = pd_env("PADDOCK_V9Q_MB_GLB");
                    mb9g = e ? atoi(e) : 1;
                    if (mb9g < 1 || mb9g > 2) mb9g = 1;
                }
                if (mb9g == 2) PD_V9Q_GO(512u, 8u, 2u, hw9g2, v9g_smem)
                PD_V9Q_GO(512u, 8u, 1u, hw9g1, v9g_smem)
#undef PD_V9Q_GO
            }
        }
        // v8q measured BEHIND v8f8 in production KV8 (c32 1082.5 vs 1091.7,
        // dc32 1737.1 vs 1765.5): the fp8-score win does not cover its
        // occupancy cost at this architecture - score was never the hd256
        // wall. OPT-IN (PADDOCK_V8Q=1) pending a QGMMA redesign.
        static int no_v8q = -1;
        if (no_v8q < 0) no_v8q = pd_env("PADDOCK_V8Q") ? 0 : 1;
        if (!no_v8q) {
            struct PdTmEnt8s { const void* p; uint32_t kd; CUtensorMap m; };
            static PdTmEnt8s t8s[64];
            static uint32_t t8sn = 0;
            auto get_tm8s = [&](const void* base) -> const CUtensorMap* {
                for (uint32_t i = 0; i < t8sn; ++i)
                    if (t8s[i].p == base && t8s[i].kd == kv_dim) return &t8s[i].m;
                if (t8sn >= 64u) t8sn = 0;
                if (!pd_attn_tmap_kv_f8s(&t8s[t8sn].m, base, kv_dim)) return nullptr;
                t8s[t8sn].p = base; t8s[t8sn].kd = kv_dim;
                return &t8s[t8sn++].m;
            };
            const CUtensorMap* qk = get_tm8s(pool_k);
            const CUtensorMap* qv = qk ? get_tm8s(pool_v) : nullptr;
            if (qk && qv) {
                const uint32_t v8q_smem = 2u * 4096u + 3u * 4096u + 3u * 8192u
                    + 16u * (256u + 16u) + 2u * 16u * 24u * 2u + 1024u;
                static uint32_t v8q_set = 0;
                if (v8q_set == 0) {
                    cudaFuncSetAttribute(
                        (const void*)pd_attn_decode_v8q_kernel<256u, 2u>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)v8q_smem);
                    v8q_set = 1;
                }
                dim3 qgrid(n_kv_heads, batch, n_splits);
                pd_pdl_go(pd_attn_decode_v8q_kernel<256u, 2u>, qgrid, 320, v8q_smem,
                          (cudaStream_t)stream,
                          *qk, *qv, (const float*)q, (float*)out_o, (float*)out_ml,
                          (const unsigned int*)positions, (const unsigned int*)slots,
                          (const uint32_t*)block_tables, blocks_per_slot,
                          kv_dim, swa_window, n_splits, scale);
                return pd_launch_status();
            }
        }
        const CUtensorMap* tk = get_tm8(pool_k);
        const CUtensorMap* tv = tk ? get_tm8(pool_v) : nullptr;
        if (tk && tv) {
            const uint32_t row_e8 = 256u + 8u, w_s8 = 16u + 8u;
            // same smem as the f16 form: raw lands in the buffer tails
            const uint32_t v8f8_smem = 5u * 4u * 2048u
                + 16u * row_e8 * 2u + 2u * 16u * w_s8 * 2u + 1024u;
            static uint32_t v8f8_set = 0;
            if (v8f8_set == 0) {
                cudaFuncSetAttribute(
                    (const void*)pd_attn_decode_v8_kernel<256u, 2u, true>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)v8f8_smem);
                v8f8_set = 1;
            }
            dim3 v8grid(n_kv_heads, batch, n_splits);
            // 384 threads: warps 0-7 = the f16 pipeline roles, 8-11 = the
            // fp8->f16 expander warps (see the kernel note)
            pd_pdl_go(pd_attn_decode_v8_kernel<256u, 2u, true>, v8grid, 384, v8f8_smem,
                      (cudaStream_t)stream,
                      *tk, *tv, (const float*)q, (float*)out_o, (float*)out_ml,
                      (const unsigned int*)positions, (const unsigned int*)slots,
                      (const uint32_t*)block_tables, blocks_per_slot,
                      kv_dim, swa_window, n_splits, scale);
            return pd_launch_status();
        }
    }
    if (!no_v3 && kv_dtype != PD_KV_FP8_E4M3
        && ((head_dim == 256u && group == 2u) || (head_dim == 512u && group == 8u))
        && n_heads == n_kv_heads * group) {
        const uint32_t t_s = 16u + 1u, row_e = head_dim + 8u, w_s = 16u + 8u;
        const uint32_t v3_smem = 2u * group * t_s * 4u + 16u * row_e * 2u
            + 16u * w_s * 2u + 4u * 16u * row_e * 2u;
        // v5: softmax folded into the score warps - smaller smem
        // (no s_sc/s_w float strips), 3 barriers/tile. Kill: PADDOCK_NO_ATTN_V5
        // pins v3.
        const uint32_t v5_smem = 16u * row_e * 2u + 16u * w_s * 2u
            + 4u * 16u * row_e * 2u;
        static int no_v5 = -1;
        if (no_v5 < 0) no_v5 = pd_env("PADDOCK_NO_ATTN_V5") ? 1 : 0;
        // v7: TMA block-aligned staging (SWA 112.8 vs v5 116,
        // GLB 131.6 vs 152 on the dec_attn bench, bit-exact). Needs
        // cc >= 9 (bulk tensor) + the driver encode fn; falls back to v5.
        // Kill: PADDOCK_NO_ATTN_V7.
        static int no_v7 = -1;
        if (no_v7 < 0) {
            int dev = 0, ccm = 0;
            cudaGetDevice(&dev);
            cudaDeviceGetAttribute(&ccm, cudaDevAttrComputeCapabilityMajor, dev);
            no_v7 = (pd_env("PADDOCK_NO_ATTN_V7") || ccm < 9
                     || !pd_tmap_encode()) ? 1 : 0;
        }
        if (!no_v7 && !no_v5) {
            // per-(pool, kv_dim) tmap cache: one pool pair per layer class,
            // stable pointers for the model's life (same wraparound pattern
            // as the fp4 launcher's shape cache)
            struct PdTmEnt { const void* p; uint32_t kd; CUtensorMap m; };
            static PdTmEnt tcache[64]; static uint32_t tn = 0;
            auto get_tm = [&](const void* base) -> const CUtensorMap* {
                for (uint32_t i = 0; i < tn; ++i)
                    if (tcache[i].p == base && tcache[i].kd == kv_dim)
                        return &tcache[i].m;
                if (tn >= 64u) tn = 0;
                if (!pd_attn_tmap_kv(&tcache[tn].m, base, kv_dim)) return nullptr;
                tcache[tn].p = base; tcache[tn].kd = kv_dim;
                return &tcache[tn++].m;
            };
            const CUtensorMap* tk = get_tm(pool_k);
            const CUtensorMap* tv = tk ? get_tm(pool_v) : nullptr;
            if (tk && tv) {
                const uint32_t segs = head_dim * 2u / 128u;
                const uint32_t v7_smem = 4u * segs * 2048u
                    + 16u * row_e * 2u + 16u * w_s * 2u + 1024u;
                static uint32_t v7_set = 0;
                if (v7_set == 0) {
                    cudaFuncSetAttribute((const void*)pd_attn_decode_v7_kernel<256u, 2u>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         4u * 4u * 2048u + 16u * 264u * 2u + 16u * 24u * 2u + 1024u);
                    cudaFuncSetAttribute((const void*)pd_attn_decode_v7_kernel<512u, 8u>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         4u * 8u * 2048u + 16u * 520u * 2u + 16u * 24u * 2u + 1024u);
                    v7_set = 1;
                }
                dim3 v7grid(n_kv_heads, batch, n_splits);
                // v8: warp-specialized pipeline for the SWA shape -
                // score warps run tile t+1 while V warps run tile t (named
                // barriers, K ring 2 / V ring 3, w ping-pong). dec_attn
                // bench: 107.2 -> 90.4 us at the c32 shape (batch 32,
                // splits 1), 37.1 -> 33.1 at batch 8/splits 4 - bit-exact
                // vs v7 at matched splits. GLB loses (6-warp V split over
                // 512 dims) - v7 stays for hd512. Kill: PADDOCK_NO_ATTN_V8.
                static int no_v8 = -1;
                if (no_v8 < 0) no_v8 = pd_env("PADDOCK_NO_ATTN_V8") ? 1 : 0;
                // v8ks at the SWA shape: FALSIFIED (bench,
                // swa ctx2048 b32: v8 87.8us/6.11 TB/s vs v8ks 88.2us -
                // flat at every split). hd256 SWA is already at the KV
                // memory roof; the 2/6 imbalance only binds at hd512.
                // Route kept as a rebuild-free re-check knob, never default.
                static int swa_v8ks = -1;
                if (swa_v8ks < 0) swa_v8ks = pd_env("PADDOCK_SWA_V8KS") ? 1 : 0;
                if (swa_v8ks && head_dim == 256u) {
                    const uint32_t ks_smem = 5u * ((256u * 2u / 128u) * 2048u)
                        + 16u * (256u + 8u) * 2u + 2u * 16u * 24u * 2u + 1024u;
                    static bool ks256_attr = false;
                    if (!ks256_attr) {
                        cudaFuncSetAttribute(
                            (const void*)pd_attn_decode_v8ks_kernel<256u, 2u>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)ks_smem);
                        ks256_attr = true;
                    }
                    pd_pdl_go(pd_attn_decode_v8ks_kernel<256u, 2u>, v7grid, 256, ks_smem,
                              (cudaStream_t)stream,
                              *tk, *tv, (const float*)q, (float*)out_o, (float*)out_ml,
                              (const unsigned int*)positions, (const unsigned int*)slots,
                              (const uint32_t*)block_tables, blocks_per_slot,
                              kv_dim, swa_window, n_splits, scale);
                    return pd_launch_status();
                }
                if (!no_v8 && head_dim == 256u) {
                    const uint32_t v8_smem = 5u * segs * 2048u
                        + 16u * row_e * 2u + 2u * 16u * w_s * 2u + 1024u;
                    static uint32_t v8_set = 0;
                    if (v8_set == 0) {
                        cudaFuncSetAttribute(
                            (const void*)pd_attn_decode_v8_kernel<256u, 2u>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                            5u * 4u * 2048u + 16u * 264u * 2u
                                + 2u * 16u * 24u * 2u + 1024u);
                        v8_set = 1;
                    }
                    pd_pdl_go(pd_attn_decode_v8_kernel<256u, 2u>, v7grid, 256, v8_smem,
                   (cudaStream_t)stream,
                            *tk, *tv, (const float*)q, (float*)out_o, (float*)out_ml,
                            (const unsigned int*)positions, (const unsigned int*)slots,
                            (const uint32_t*)block_tables, blocks_per_slot,
                            kv_dim, swa_window, n_splits, scale);
                    return pd_launch_status();
                }
                if (head_dim == 256u)
                    pd_pdl_go(pd_attn_decode_v7_kernel<256u, 2u>, v7grid, 256, v7_smem,
                   (cudaStream_t)stream,
                            *tk, *tv, (const float*)q, (float*)out_o, (float*)out_ml,
                            (const unsigned int*)positions, (const unsigned int*)slots,
                            (const uint32_t*)block_tables, blocks_per_slot,
                            kv_dim, swa_window, n_splits, scale);
                else {
                    // v7ks: K-split score across all 8 warps - the
                    // hd512 score phase idled 6 warps for 32 HMMA steps.
                    // Bench: 142.9 -> 125.5 us (-12%, 4.28 TB/s), class
                    // change (cross-slice sum). Kill: PADDOCK_NO_V7KS.
                    // v8ks: the cross-tile pipeline at the BALANCED
                    // 4-score/4-V split (K-split score x warp specialization) -
                    // bench glb 142.7 -> 95.5 us (-33%, 5.62 TB/s = the
                    // both-halves bound). Kill: PADDOCK_NO_V8KS -> v7ks.
                    static int no_v8ks = -1;
                    if (no_v8ks < 0) no_v8ks = pd_env("PADDOCK_NO_V8KS") ? 1 : 0;
                    static int no_v7ks = -1;
                    if (no_v7ks < 0) no_v7ks = pd_env("PADDOCK_NO_V7KS") ? 1 : 0;
                    static bool ks_attr = false;
                    if (!ks_attr) {
                        cudaFuncSetAttribute(
                            (const void*)pd_attn_decode_v7ks_kernel<512u, 8u>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)v7_smem);
                        ks_attr = true;
                    }
                    static bool ks8_attr = false;
                    if (!ks8_attr) {
                        cudaFuncSetAttribute(
                            (const void*)pd_attn_decode_v8ks_kernel<512u, 8u>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                            (int)(5u * ((512u * 2u / 128u) * 2048u)
                                  + 16u * (512u + 8u) * 2u + 2u * 16u * 24u * 2u + 1024u));
                        ks8_attr = true;
                    }
                    if (!no_v8ks) {
                        const uint32_t v8s = 5u * ((512u * 2u / 128u) * 2048u)
                            + 16u * (512u + 8u) * 2u + 2u * 16u * 24u * 2u + 1024u;
                        pd_pdl_go(pd_attn_decode_v8ks_kernel<512u, 8u>, v7grid, 256, v8s,
                   (cudaStream_t)stream,
                                *tk, *tv, (const float*)q, (float*)out_o, (float*)out_ml,
                                (const unsigned int*)positions, (const unsigned int*)slots,
                                (const uint32_t*)block_tables, blocks_per_slot,
                                kv_dim, swa_window, n_splits, scale);
                    } else if (!no_v7ks)
                        pd_pdl_go(pd_attn_decode_v7ks_kernel<512u, 8u>, v7grid, 256, v7_smem,
                   (cudaStream_t)stream,
                                *tk, *tv, (const float*)q, (float*)out_o, (float*)out_ml,
                                (const unsigned int*)positions, (const unsigned int*)slots,
                                (const uint32_t*)block_tables, blocks_per_slot,
                                kv_dim, swa_window, n_splits, scale);
                    else
                        pd_pdl_go(pd_attn_decode_v7_kernel<512u, 8u>, v7grid, 256, v7_smem,
                   (cudaStream_t)stream,
                                *tk, *tv, (const float*)q, (float*)out_o, (float*)out_ml,
                                (const unsigned int*)positions, (const unsigned int*)slots,
                                (const uint32_t*)block_tables, blocks_per_slot,
                                kv_dim, swa_window, n_splits, scale);
                }
                return pd_launch_status();
            }
        }
        static uint32_t v3_set = 0;
        if (v3_set == 0) {
            cudaFuncSetAttribute((const void*)pd_attn_decode_v3_kernel<256u, 2u, 16u>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 2u * 2u * 17u * 4u + 16u * 264u * 2u + 16u * 24u * 2u
                                     + 4u * 16u * 264u * 2u);
            cudaFuncSetAttribute((const void*)pd_attn_decode_v3_kernel<512u, 8u, 16u>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 2u * 8u * 17u * 4u + 16u * 520u * 2u + 16u * 24u * 2u
                                     + 4u * 16u * 520u * 2u);
            cudaFuncSetAttribute((const void*)pd_attn_decode_v5_kernel<256u, 2u, 16u>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 16u * 264u * 2u + 16u * 24u * 2u + 4u * 16u * 264u * 2u);
            cudaFuncSetAttribute((const void*)pd_attn_decode_v5_kernel<512u, 8u, 16u>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 16u * 520u * 2u + 16u * 24u * 2u + 4u * 16u * 520u * 2u);
            v3_set = 1;
        }
        dim3 vgrid(n_kv_heads, batch, n_splits);
        if (!no_v5) {
            if (head_dim == 256u)
                pd_attn_decode_v5_kernel<256u, 2u, 16u>
                    <<<vgrid, 256, v5_smem, (cudaStream_t)stream>>>(
                        (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                        (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                        (const unsigned int*)slots, (const uint32_t*)block_tables,
                        blocks_per_slot, kv_dim, swa_window, n_splits, scale);
            else
                pd_attn_decode_v5_kernel<512u, 8u, 16u>
                    <<<vgrid, 256, v5_smem, (cudaStream_t)stream>>>(
                        (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                        (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                        (const unsigned int*)slots, (const uint32_t*)block_tables,
                        blocks_per_slot, kv_dim, swa_window, n_splits, scale);
            return pd_launch_status();
        }
        if (head_dim == 256u)
            pd_attn_decode_v3_kernel<256u, 2u, 16u>
                <<<vgrid, 256, v3_smem, (cudaStream_t)stream>>>(
                    (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                    (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                    (const unsigned int*)slots, (const uint32_t*)block_tables,
                    blocks_per_slot, kv_dim, swa_window, n_splits, scale);
        else
            pd_attn_decode_v3_kernel<512u, 8u, 16u>
                <<<vgrid, 256, v3_smem, (cudaStream_t)stream>>>(
                    (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                    (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                    (const unsigned int*)slots, (const uint32_t*)block_tables,
                    blocks_per_slot, kv_dim, swa_window, n_splits, scale);
        return pd_launch_status();
    }
    if (!no_gqa_p && head_dim > 128u && head_dim <= 512u && group >= 2u && group <= 8u
        && n_kv_heads >= 2u && n_heads == n_kv_heads * group) {
        // head_dim 512 (gemma4 global layers, GQA 8:1): the TILE-16 GQA tile's
        // smem is 84 KB at hd512 - inside the opt-in window, kernel body is
        // hd-generic. Without this arm the per-q-head kernel re-reads each
        // KV slice GROUP(8)x: 25.7% of the pf8 GPU at 4.45 ms/layer;
        // windowless full-causal walks amplify it at long ctx.
        // PD_ATTN_TILE256=32 (same A/B as the dense arm): 32-token tiles at
        // hd<=256 - half the tile iterations/barriers, ~74 KB smem at g6/hd256
        // (vs ~40 KB) so co-residency drops 2->1; measured, not assumed.
        static int t256p = -1;
        if (t256p < 0) { const char* e = pd_env("PD_ATTN_TILE256"); t256p = (e && atoi(e) == 32) ? 32 : 16; }
        if (t256p == 32 && head_dim <= 256u) {
            const uint32_t gs32 = (uint32_t)PD_GQA_SMEM(group, head_dim, 32u);
            static uint32_t g32p_set = 0;
            if (gs32 > g32p_set) {
                cudaFuncSetAttribute(
                    (const void*)pd_attn_decode_batch_partial_gqa_paged_kernel<__nv_fp8_e4m3, 32u>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, gs32);
                cudaFuncSetAttribute(
                    (const void*)pd_attn_decode_batch_partial_gqa_paged_kernel<__half, 32u>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, gs32);
                g32p_set = gs32;
            }
            dim3 ggrid(n_kv_heads, batch, n_splits);
            if (kv_dtype == PD_KV_FP8_E4M3)
                pd_pdl_go(pd_attn_decode_batch_partial_gqa_paged_kernel<__nv_fp8_e4m3, 32u>,
                    ggrid, attn_nth, gs32, (cudaStream_t)stream,
                        (const float*)q, (const __nv_fp8_e4m3*)pool_k, (const __nv_fp8_e4m3*)pool_v,
                        (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                        (const unsigned int*)slots, (const uint32_t*)block_tables, blocks_per_slot,
                        n_heads, n_kv_heads, head_dim, kv_dim, swa_window, n_splits, scale);
            else
                pd_pdl_go(pd_attn_decode_batch_partial_gqa_paged_kernel<__half, 32u>,
                    ggrid, attn_nth, gs32, (cudaStream_t)stream,
                        (const float*)q, (const __half*)pool_k, (const __half*)pool_v, (float*)out_o,
                        (float*)out_ml, (const unsigned int*)positions, (const unsigned int*)slots,
                        (const uint32_t*)block_tables, blocks_per_slot, n_heads, n_kv_heads, head_dim,
                        kv_dim, swa_window, n_splits, scale);
            return pd_launch_status();
        }
        const uint32_t gqa_smem = (uint32_t)PD_GQA_SMEM(group, head_dim, 16u);
        static uint32_t gqa_set_p = 0;
        if (gqa_smem > 48u * 1024u && gqa_smem > gqa_set_p) {
            cudaFuncSetAttribute((const void*)pd_attn_decode_batch_partial_gqa_paged_kernel<__nv_fp8_e4m3, 16u>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, gqa_smem);
            cudaFuncSetAttribute((const void*)pd_attn_decode_batch_partial_gqa_paged_kernel<__half, 16u>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, gqa_smem);
            gqa_set_p = gqa_smem;
        }
        dim3 ggrid(n_kv_heads, batch, n_splits);
        // PV-mma (default-on, kill PADDOCK_NO_ATTN_MMA_PV): the hd256/f16
        // serving shape's P*V fold on tf32 tensor cores (see the kernel's
        // pv_on arm; attn_nth is exactly 256 at hd256, which the kernel gate
        // re-checks).
        if (pd_attn_pv_mma() && kv_dtype != PD_KV_FP8_E4M3 && head_dim == 256u) {
            static uint32_t pv_set = 0;
            if (pv_set == 0) {
                pd_prefer_max_shared(pd_attn_decode_batch_partial_gqa_paged_kernel<__half, 16u, true>);
                pv_set = 1;
            }
            if (gqa_smem > 48u * 1024u && gqa_smem > pv_set) {
                cudaFuncSetAttribute(
                    (const void*)pd_attn_decode_batch_partial_gqa_paged_kernel<__half, 16u, true>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, gqa_smem);
                pv_set = gqa_smem;
            }
            pd_attn_decode_batch_partial_gqa_paged_kernel<__half, 16u, true>
                <<<ggrid, attn_nth, gqa_smem, (cudaStream_t)stream>>>(
                    (const float*)q, (const __half*)pool_k, (const __half*)pool_v, (float*)out_o,
                    (float*)out_ml, (const unsigned int*)positions, (const unsigned int*)slots,
                    (const uint32_t*)block_tables, blocks_per_slot, n_heads, n_kv_heads, head_dim,
                    kv_dim, swa_window, n_splits, scale);
            return pd_launch_status();
        }
        if (kv_dtype == PD_KV_FP8_E4M3)
            pd_pdl_go(pd_attn_decode_batch_partial_gqa_paged_kernel<__nv_fp8_e4m3, 16u>,
                ggrid, attn_nth, gqa_smem, (cudaStream_t)stream,
                    (const float*)q, (const __nv_fp8_e4m3*)pool_k, (const __nv_fp8_e4m3*)pool_v,
                    (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                    (const unsigned int*)slots, (const uint32_t*)block_tables, blocks_per_slot,
                    n_heads, n_kv_heads, head_dim, kv_dim, swa_window, n_splits, scale);
        else
            pd_pdl_go(pd_attn_decode_batch_partial_gqa_paged_kernel<__half, 16u>,
                ggrid, attn_nth, gqa_smem, (cudaStream_t)stream,
                    (const float*)q, (const __half*)pool_k, (const __half*)pool_v, (float*)out_o,
                    (float*)out_ml, (const unsigned int*)positions, (const unsigned int*)slots,
                    (const uint32_t*)block_tables, blocks_per_slot, n_heads, n_kv_heads, head_dim,
                    kv_dim, swa_window, n_splits, scale);
        return pd_launch_status();
    }
    if (kv_dtype == PD_KV_FP8_E4M3)
        pd_pdl_go(pd_attn_decode_batch_partial_paged_kernel<__nv_fp8_e4m3>, grid, attn_nth, attn_smem, (cudaStream_t)stream,
            (const float*)q, (const __nv_fp8_e4m3*)pool_k, (const __nv_fp8_e4m3*)pool_v, (float*)out_o,
            (float*)out_ml, (const unsigned int*)positions, (const unsigned int*)slots,
            (const uint32_t*)block_tables, blocks_per_slot, n_heads, n_kv_heads, head_dim,
            kv_dim, swa_window, n_splits, scale);
    else
        pd_pdl_go(pd_attn_decode_batch_partial_paged_kernel<__half>, grid, attn_nth, attn_smem, (cudaStream_t)stream,
            (const float*)q, (const __half*)pool_k, (const __half*)pool_v, (float*)out_o, (float*)out_ml,
            (const unsigned int*)positions, (const unsigned int*)slots,
            (const uint32_t*)block_tables, blocks_per_slot, n_heads, n_kv_heads, head_dim,
            kv_dim, swa_window, n_splits, scale);
    return pd_launch_status();
}

// a16 twin for the attention streams: q is an f16 plane; partials stay
// f32. Plain per-(head,row,split) walk only (the serve fallback for chunk
// counts below the krs width gate) - the gqa-fused (hd<=128) and dense-FA
// arms have no f16-q form and this geometry never routes there at gemma4.
PD_EXPORT
int pd_attn_decode_batch_partial_paged2(const void* q, const void* pool_k, const void* pool_v,
                                        void* out_o, void* out_ml, const void* positions,
                                        const void* slots, const void* block_tables,
                                        uint32_t blocks_per_slot, uint32_t n_heads,
                                        uint32_t n_kv_heads, uint32_t head_dim, uint32_t kv_dim,
                                        uint32_t swa_window, uint32_t n_splits, uint32_t batch,
                                        float scale, uint32_t kv_dtype, uint32_t a16,
                                        void* stream) {
    if (!a16)
        return pd_attn_decode_batch_partial_paged(q, pool_k, pool_v, out_o, out_ml,
                                                  positions, slots, block_tables,
                                                  blocks_per_slot, n_heads, n_kv_heads,
                                                  head_dim, kv_dim, swa_window, n_splits,
                                                  batch, scale, kv_dtype, stream);
    if (n_heads == 0 || batch == 0 || n_splits == 0) return 0;
    if (head_dim <= 128u) return -3;  // gqa-fuse geometry has no a16 arm
    dim3 grid(n_heads, batch, n_splits);
    uint32_t attn_nth = head_dim > 256 ? head_dim : 256;
    uint32_t attn_smem = (uint32_t)PD_ATTN_TILE_SMEM(head_dim);
    static uint32_t smem_set_pp16 = 0;
    if (smem_set_pp16 == 0) {
        pd_prefer_max_shared(pd_attn_decode_batch_partial_paged_kernel<__nv_fp8_e4m3, __half>);
        pd_prefer_max_shared(pd_attn_decode_batch_partial_paged_kernel<__half, __half>);
        smem_set_pp16 = 1;
    }
    if (attn_smem > 48u * 1024u && attn_smem > smem_set_pp16) {
        cudaFuncSetAttribute((const void*)pd_attn_decode_batch_partial_paged_kernel<__nv_fp8_e4m3, __half>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, attn_smem);
        cudaFuncSetAttribute((const void*)pd_attn_decode_batch_partial_paged_kernel<__half, __half>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, attn_smem);
        smem_set_pp16 = attn_smem;
    }
    if (kv_dtype == PD_KV_FP8_E4M3)
        pd_pdl_go(pd_attn_decode_batch_partial_paged_kernel<__nv_fp8_e4m3, __half>, grid, attn_nth, attn_smem, (cudaStream_t)stream,
            (const __half*)q, (const __nv_fp8_e4m3*)pool_k, (const __nv_fp8_e4m3*)pool_v, (float*)out_o,
            (float*)out_ml, (const unsigned int*)positions, (const unsigned int*)slots,
            (const uint32_t*)block_tables, blocks_per_slot, n_heads, n_kv_heads, head_dim,
            kv_dim, swa_window, n_splits, scale);
    else
        pd_pdl_go(pd_attn_decode_batch_partial_paged_kernel<__half, __half>, grid, attn_nth, attn_smem, (cudaStream_t)stream,
            (const __half*)q, (const __half*)pool_k, (const __half*)pool_v, (float*)out_o, (float*)out_ml,
            (const unsigned int*)positions, (const unsigned int*)slots,
            (const uint32_t*)block_tables, blocks_per_slot, n_heads, n_kv_heads, head_dim,
            kv_dim, swa_window, n_splits, scale);
    return pd_launch_status();
}

PD_EXPORT
int pd_attn_decode_batch_combine(const void* in_o, const void* in_ml, const void* sinks, void* out,
                                 uint32_t n_heads, uint32_t head_dim, uint32_t n_splits,
                                 uint32_t batch, void* stream) {
    if (n_heads == 0 || batch == 0 || n_splits == 0) return 0;
    dim3 grid(n_heads, batch);
    pd_pdl_go(pd_attn_decode_batch_combine_kernel<float>, grid, head_dim, 0u,
              (cudaStream_t)stream,
              (const float*)in_o, (const float*)in_ml, (const float*)sinks, (float*)out,
              n_heads, head_dim, n_splits);
    return pd_launch_status();
}

// o16 twin for the attention streams: the final plane is f16; the
// (o, m, l) partials stay f32. Appended as its own export per the ABI
// growth rule.
PD_EXPORT
int pd_attn_decode_batch_combine2(const void* in_o, const void* in_ml, const void* sinks,
                                  void* out, uint32_t n_heads, uint32_t head_dim,
                                  uint32_t n_splits, uint32_t batch, uint32_t o16,
                                  void* stream) {
    if (!o16)
        return pd_attn_decode_batch_combine(in_o, in_ml, sinks, out, n_heads,
                                            head_dim, n_splits, batch, stream);
    if (n_heads == 0 || batch == 0 || n_splits == 0) return 0;
    dim3 grid(n_heads, batch);
    pd_pdl_go(pd_attn_decode_batch_combine_kernel<__half>, grid, head_dim, 0u,
              (cudaStream_t)stream,
              (const float*)in_o, (const float*)in_ml, (const float*)sinks, (__half*)out,
              n_heads, head_dim, n_splits);
    return pd_launch_status();
}

// Fused combine + per-ROW e4m3 quant (elementwise): the wo
// GEMM consumes only the quantized attention output, so the f32 row never
// lands - one CTA per row combines all (head, split) partials into smem,
// takes the exact row max, and emits e4m3 + the f32 row scale. Per-element
// combine math and the quant clone their parents exactly -> bit-identical
// to attn_decode_batch_combine + quantize_e4m3_row.
__global__ void pd_attn_combine_e4m3_row_kernel(
    const float* __restrict__ in_o, const float* __restrict__ in_ml,
    const float* __restrict__ sinks, unsigned char* __restrict__ q,
    float* __restrict__ rscale, uint32_t n_heads, uint32_t head_dim,
    uint32_t n_splits, uint32_t batch) {
    extern __shared__ float cm_row[];                  // [n_heads*head_dim]
    const uint32_t b = blockIdx.x;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    const uint32_t n = n_heads * head_dim;
    __shared__ float wmax[32];
    __shared__ int s_e;
    float lm = 0.0f;
    for (uint32_t i = tid; i < n; i += nth) {
        const uint32_t h = i / head_dim, d = i % head_dim;
        const size_t pbase = (size_t)(h * batch + b) * n_splits;
        float gm = sinks[h];
        for (uint32_t s = 0; s < n_splits; ++s)
            gm = fmaxf(gm, in_ml[(pbase + s) * 2 + 0]);
        float acc = 0.0f, l = 0.0f;
        for (uint32_t s = 0; s < n_splits; ++s) {
            float m = in_ml[(pbase + s) * 2 + 0];
            float ls = in_ml[(pbase + s) * 2 + 1];
            float sc = __expf(m - gm);
            acc += sc * in_o[(pbase + s) * head_dim + d];
            l += sc * ls;
        }
        l += __expf(sinks[h] - gm);
        const float v = acc / l;
        cm_row[i] = v;
        lm = fmaxf(lm, fabsf(v));
    }
    for (uint32_t sh = 16; sh > 0; sh >>= 1)
        lm = fmaxf(lm, __shfl_xor_sync(0xffffffffu, lm, sh));
    if ((tid & 31u) == 0) wmax[tid >> 5] = lm;
    __syncthreads();
    if (tid == 0) {
        float m = 0.0f;
        for (uint32_t w = 0; w < ((nth + 31u) >> 5); ++w) m = fmaxf(m, wmax[w]);
        int e = 0;
        if (m > 0.0f) {
            int ex;
            float fr = frexpf(m, &ex);
            e = ex - 9 + (fr > 0.875f ? 1 : 0);
        }
        s_e = e;
        rscale[b] = ldexpf(1.0f, e);
    }
    __syncthreads();
    const float qinv = ldexpf(1.0f, -s_e);
    unsigned char* qr = q + (size_t)b * n;
    const uint32_t n4 = n >> 2;
    for (uint32_t i = tid; i < n4; i += nth) {
        const float4 v = *(const float4*)(cm_row + (size_t)i * 4u);
        uchar4 o;
        o.x = __nv_fp8_e4m3(v.x * qinv).__x;
        o.y = __nv_fp8_e4m3(v.y * qinv).__x;
        o.z = __nv_fp8_e4m3(v.z * qinv).__x;
        o.w = __nv_fp8_e4m3(v.w * qinv).__x;
        *(uchar4*)(qr + (size_t)i * 4u) = o;
    }
}

PD_EXPORT
int pd_attn_combine_e4m3_row(const void* in_o, const void* in_ml, const void* sinks,
                             void* q, void* rscale, uint32_t n_heads,
                             uint32_t head_dim, uint32_t n_splits, uint32_t batch,
                             void* stream) {
    if (n_heads == 0 || batch == 0 || n_splits == 0) return 0;
    const uint32_t n = n_heads * head_dim;
    if (n & 3u) return cudaErrorInvalidValue;
    const uint32_t smem = n * 4u;
    static uint32_t cset = 0;
    if (smem > 48u * 1024u && smem > cset) {
        cudaFuncSetAttribute((const void*)pd_attn_combine_e4m3_row_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        cset = smem;
    }
    pd_attn_combine_e4m3_row_kernel<<<batch, 1024, smem, (cudaStream_t)stream>>>(
        (const float*)in_o, (const float*)in_ml, (const float*)sinks,
        (unsigned char*)q, (float*)rscale, n_heads, head_dim, n_splits, batch);
    return pd_launch_status();
}


// Wide-batch spec-verify attention launcher: rows are PADDED slot-major
// chunks (k1 consecutive positions per slot - the serving verify layout).
// gsub=2 sub-groups keep the register/smem budget: SWA (group 2) keeps full
// GQA fusion at TILE 16; hd512 global layers ride TILE 8. Requires
// rows%k1==0, k1<=PD_SPEC_K1_MAX, uniform slot per chunk.
static int pd_attn_spec_batch_paged_impl(const void* q, const void* pool_k, const void* pool_v,
                             void* out_o, void* out_ml, const void* positions,
                             const void* slots, const void* block_tables,
                             uint32_t blocks_per_slot, uint32_t n_heads,
                             uint32_t n_kv_heads, uint32_t head_dim, uint32_t kv_dim,
                             uint32_t swa_window, uint32_t n_splits, uint32_t rows,
                             uint32_t k1, float scale, uint32_t kv_dtype, uint32_t a16,
                             void* stream) {
    // FIN sentinel (top bit, set by pd_attn_spec_batch_fin): FA route only,
    // n_splits==1, in-kernel finalize. Grid/gating use the masked count; the
    // flagged value goes to the kernels; non-FA fallbacks return -2.
    // bit30 = FE4S (fin static-scale e4m3 store into the
    // wo-in quantized plane) rides with the FIN bit - extracted together so
    // every `n_splits | fin_bit` launch site carries it unchanged and the
    // non-FA fallthrough below still rejects with -2. Only ever set by
    // pd_attn_spec_batch_fin_e4s (slot 425), always alongside FIN.
    const uint32_t fin_bit = n_splits & 0xC0000000u;
    n_splits &= 0x3fffffffu;
    if (rows == 0 || n_splits == 0) return 0;
    const uint32_t group = n_kv_heads ? n_heads / n_kv_heads : 0;
    if (k1 == 0 || k1 > 8u || rows % k1 != 0u || group == 0 || head_dim > 512u
        || n_heads != n_kv_heads * group) {
        return -2; // caller falls back to the per-row kernel
    }
    // FA-lite route (f16 KV, hd 64..512, k1<=8): the mma-score/mma-o kernel,
    // default-on for cc 10 (B200 - the old walk measured 56% of the c32 GPU
    // there), PADDOCK_SPEC_FA=1 forces elsewhere, PADDOCK_NO_SPEC_FA kills.
    const int fa_mode = pd_spec_fa_mode();
    // F8 arm: KV8 pools take the same FA route with in-kernel e4m3
    // expansion (halved KV DRAM) - before this, fp8 verify ticks rode the
    // old walk, the exact 56%-of-GPU shape FA was built to kill. Same smem
    // (in-place strip) and geometry gates; padded layout only (the NOPAD
    // A/B pin keeps fp8 on the walk). PADDOCK_NO_SPEC_FA_F8 kills.
    static int fa_f8 = -1;
    if (fa_f8 < 0) fa_f8 = pd_env("PADDOCK_NO_SPEC_FA_F8") ? 0 : 1;
    const bool f8 = (kv_dtype == PD_KV_FP8_E4M3);
    // qwen35 verify arm (rung E1): the 24q/4kv/hd256
    // fp8 geometry at k1 > 1. This is the dense-decode GV=6 krs class
    // (P8+QK8+VR+DBK+KVS - the q36 rung-4 default at k1=1, same
    // instantiation) widened to the verify chunk: one KV walk per (kv-head,
    // slot block, split) serves the block's k1 rows, where the engine's
    // decode dispatch at >=47 rows fell to the scalar per-(q-head, row)
    // walk - 6144 CTAs each re-reading its slot's whole KV, which captured
    // at 42% of GPU time where attention should be a small fraction of it).
    // Placed above the fa_mode tree deliberately: that tree is env-armed on
    // sm_120 (gemma4 sets PADDOCK_SPEC_FA at load) and its generic routes
    // have no G=6 tile; this arm elects itself from geometry alone, like
    // the dense fa8 arm in pd_attn_decode_batch_partial_paged.
    // fill_sms = 0: the spec arms keep the FIXED-split law (mod.rs
    // attn_splits) - a die-fill floor lets the GRID size pick a row's
    // reduction order, and the per-slot exact gates compare a single-slot
    // run against the batched one. s_eff still clamps to the chunk's own
    // context (CTA-local, position-invariant across batch shapes).
    // Numerics: the shipped fp8 dense-decode class (e4m3 Q/P rounding);
    // the f16-KV exact gates never reach this arm. Kill: PADDOCK_NO_SPEC_FA6.
    static int fa6 = -1;
    if (fa6 < 0) fa6 = pd_env("PADDOCK_NO_SPEC_FA6") ? 0 : 1;
    if (fa6 && pd_fp8_mma_ok() && f8 && head_dim == 256u && group == 6u
        && k1 >= 2u) {
        if (a16) return -3;  // no f16-q twin of the GV arm; the engine only arms f32 q
        const uint32_t Mp = ((k1 * group + 15u) / 16u) * 16u;  // <= 48 at k1 <= 8
        dim3 fgrid(n_kv_heads, rows / k1, n_splits);
        return pd_spec_fa_krs_go<32u, 2u, 256u, 0u, true, true, true, float, true, true, 6u>(
            fgrid, stream,
            (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
            (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
            (const unsigned int*)slots, (const uint32_t*)block_tables,
            blocks_per_slot, n_heads, n_kv_heads, head_dim, kv_dim,
            swa_window, n_splits | fin_bit, rows, k1, scale, Mp, 0u);
    }
    // chunks >= 4: at 1-2 chunks the FA grid (n_kv x chunks x splits) is
    // 8-32 blocks and starves the die (v7: c1 -9); the old walk's gs-split
    // grid is 4x wider there. Multi-chunk spec (c8-class) is FA territory.
    if (fa_mode && (!f8 || (fa_f8 && !pd_spec_fa_nopad())) && head_dim >= 64u
        && head_dim <= 512u && (head_dim & 63u) == 0u && k1 <= 8u
        && rows / k1 >= 4u) {
        const uint32_t M = k1 * group;
        // NOTE: no smaller-PT fallback is possible in the launcher - the
        // o-update mma is m16n8k16 (16 positions per K step), so PT < 16 is
        // structurally invalid (PT=8 ILLEGAL_ADDRESSed in serving), and
        // PT=16 at M=32/hd512 misses sm_120's 101,376B cap by 384 bytes.
        // Global-layer FA on sm_120 needs a kernel-side smem restructure.
        // Arch smem guard (GB202 port fix): the global-layer
        // geometry (M=64, hd512) wants ~209KB dynamic smem - fine on B200's
        // 227KB/SM where this route was built, over sm_120's 99KB opt-in
        // cap. The attr call failed silently and every launch errored
        // (c8 throughput fell 17x, saturation 0). Only take the FA route when
        // the tile fits this device; oversized geometries (sm_120 global
        // layers) fall through to the tuned GQA walk.
        static int fa_max_smem = -1;
        if (fa_max_smem < 0) {
            int dev = 0;
            cudaGetDevice(&dev);
            if (cudaDeviceGetAttribute(&fa_max_smem,
                    cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess)
                fa_max_smem = 48 * 1024;
        }
        constexpr uint32_t PT = 32u;
        const uint32_t Mp_g = ((M + 15u) / 16u) * 16u;
        const bool fpad = !pd_spec_fa_nopad();
        const uint32_t hp = head_dim + (fpad ? 8u : 0u);
        const uint32_t smem_g = Mp_g * hp * 2u + 4u * PT * hp * 2u
            + Mp_g * (PT + (fpad ? 1u : 0u)) * 4u + 3u * Mp_g * 4u
            + Mp_g * (PT + (fpad ? 8u : 0u)) * 2u;
        // Occupancy election (GB202): on dies whose
        // per-SM smem can't co-locate two DB tiles (sm_120: 2x78KB > 100KB;
        // B200's 227KB can), the single-buffered ring at ~45KB reaches
        // 2 CTAs/SM and the co-resident CTA hides the stage stall better than
        // the second buffer did - 306.6 vs 362.4 us fin, 252.8 at sp2, at the
        // 128-row/k1=4 serve point. Same kernel, DB=false; bit-identical
        // numerics (buffering only). PADDOCK_SPEC_FA_DB=1 pins the double
        // buffer for A/B.
        static int sm_smem = -1;
        if (sm_smem < 0) {
            int dev = 0;
            cudaGetDevice(&dev);
            if (cudaDeviceGetAttribute(&sm_smem,
                    cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev) != cudaSuccess)
                sm_smem = 100 * 1024;
        }
        static int pin_db = -1;
        if (pin_db < 0) pin_db = pd_env("PADDOCK_SPEC_FA_DB") ? 1 : 0;
        // krs hoist: the SWA election used to live inside
        // the SB-occupancy branch below, whose entry test (2*smem_g >
        // sm_smem - "two DB tiles can't co-reside") is a GB202 condition:
        // false on B200's 227KB SMs, so that die kept serving the DB
        // f8-expansion route while the krs VR arm - BIT-equal at PT32,
        // maxrel 0.0000 at both depths - measured -7.6% at the window-
        // saturated serve point and -10.3% shallow (on
        // B200: fin 226.7 -> 209.4us at ctx 2758, 66.7 -> 59.8 at 272; sp2
        // 245.0 -> 226.1). Hoisted above the occupancy fork so every die
        // elects krs first; sm_120 reaches the identical launches it always
        // did. GLB stays unhoisted FOR CAUSE: at its real serve depth (the
        // global budget pool holds ~272 cells) PROD fin 54.6us beats every
        // krs/VR/sp arm (VR48 sp2 +22%); the -20% GLB win exists only at
        // depths the pool never holds. PADDOCK_NO_SPEC_KR restores the
        // expansion route; PADDOCK_SPEC_FA_DB=1 still pins DB everywhere.
        if (!pin_db && fpad && M <= 64u) {
            dim3 fgrid(n_kv_heads, rows / k1, n_splits);
            // krs: the SWA hd256/G2 F8 geometry takes the
            // fp8-resident-K kernel. PT=32 default - BIT-equal to the F8
            // expansion route and the best c8 leg (serve gate: PT40 within
            // ±0.2% on the other cells, c8 +1.1% for PT32); PT=40 (isolated
            // -0.9%, tile-boundary class) stays the A/B rung.
            if (f8 && head_dim == 256u && n_heads / n_kv_heads == 2u
                && pd_spec_fa_krs()) {
                static int krs_pt = -1;
                if (krs_pt < 0) {
                    const char* e = pd_env("PADDOCK_SPEC_KR_PT");
                    krs_pt = e ? atoi(e) : 32;
                }
                // VR ladder (opt-in): raw-V rungs at PT 32 (bit-equal to
                // KR32), 40, 64 (the freed f16 region's headroom)
                if (pd_spec_fa_vr() & 1) {
                    // DBK rungs (bit-equal to the same-PT VR arm): PT32 at
                    // occ-2 with the walk overlapped, PT16 keeps occ-3
                    if (pd_spec_fa_dbk() & 1) {
                        if (a16) return -3;  // a16 arms only the serve-default elections
                        if (pd_spec_dbk_pt() == 16)
                            return pd_spec_fa_krs_go<16u, 1u, 256u, 1u, true, false, true>(
                                fgrid, stream,
                                (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                                (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                                (const unsigned int*)slots, (const uint32_t*)block_tables,
                                blocks_per_slot, n_heads, n_kv_heads, head_dim, kv_dim,
                                swa_window, n_splits | fin_bit, rows, k1, scale, Mp_g);
                        return pd_spec_fa_krs_go<32u, 1u, 256u, 1u, true, false, true>(
                            fgrid, stream,
                            (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                            (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                            (const unsigned int*)slots, (const uint32_t*)block_tables,
                            blocks_per_slot, n_heads, n_kv_heads, head_dim, kv_dim,
                            swa_window, n_splits | fin_bit, rows, k1, scale, Mp_g);
                    }
                    const int vpt = pd_spec_vr_pt();
                    if (a16 && (vpt == 40 || vpt == 64)) return -3;
                    // vr64 DEFAULT (acceptance gate
                    // passed): the f32 arms - fin verify + split rounds,
                    // i.e. all the verify work - take PT64 on cc10. Measured
                    // -17.5% deep / -10.5% shallow vs VR32 (tile-boundary
                    // reassociation class, maxrel 0.0064/0.0236); serve
                    // 4-round ABBA: tput arm-higher 4/4 (+1.14% mean),
                    // TTFT p50 -9.6% with non-overlapping distributions,
                    // ITL wash, comm/slot arm >= ctl 4/4 (acceptance
                    // parity), distinct 32/32, coherence clean, witness
                    // PT=64 4/4. a16 (the mixed-round twin) keeps VR32 -
                    // there is no PT64 a16 instantiation and rc -3 is a
                    // hard engine error, not a fallback.
                    // PADDOCK_SPEC_VR_PT=32 restores the old default; the
                    // explicit 40/64 pins keep the old contract (pair with
                    // PADDOCK_G4_NO_ATTN16). sm_120 default unchanged.
                    static int cc10 = -1;
                    if (cc10 < 0) {
                        int dev = 0, cc = 0;
                        cudaGetDevice(&dev);
                        cudaDeviceGetAttribute(&cc, cudaDevAttrComputeCapabilityMajor, dev);
                        cc10 = cc == 10 ? 1 : 0;
                    }
                    if (vpt == 64 || (vpt == 0 && cc10 && !a16))
                        return pd_spec_fa_krs_go<64u, 1u, 256u, 1u, true>(fgrid, stream,
                            (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                            (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                            (const unsigned int*)slots, (const uint32_t*)block_tables,
                            blocks_per_slot, n_heads, n_kv_heads, head_dim, kv_dim,
                            swa_window, n_splits | fin_bit, rows, k1, scale, Mp_g);
                    if (vpt == 40)
                        return pd_spec_fa_krs_go<40u, 1u, 256u, 1u, true>(fgrid, stream,
                            (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                            (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                            (const unsigned int*)slots, (const uint32_t*)block_tables,
                            blocks_per_slot, n_heads, n_kv_heads, head_dim, kv_dim,
                            swa_window, n_splits | fin_bit, rows, k1, scale, Mp_g);
                    if (a16)
                        return pd_spec_fa_krs_go<32u, 1u, 256u, 1u, true, false, false, __half>(
                            fgrid, stream,
                            (const __half*)q, (const __half*)pool_k, (const __half*)pool_v,
                            (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                            (const unsigned int*)slots, (const uint32_t*)block_tables,
                            blocks_per_slot, n_heads, n_kv_heads, head_dim, kv_dim,
                            swa_window, n_splits | fin_bit, rows, k1, scale, Mp_g);
                    return pd_spec_fa_krs_go<32u, 1u, 256u, 1u, true>(fgrid, stream,
                        (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                        (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                        (const unsigned int*)slots, (const uint32_t*)block_tables,
                        blocks_per_slot, n_heads, n_kv_heads, head_dim, kv_dim,
                        swa_window, n_splits | fin_bit, rows, k1, scale, Mp_g);
                }
                if (a16) return -3;  // VR-off / KR_PT pins need PADDOCK_G4_NO_ATTN16
                if (krs_pt == 40)
                    return pd_spec_fa_krs_go<40u, 1u, 256u, 1u>(fgrid, stream,
                        (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                        (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                        (const unsigned int*)slots, (const uint32_t*)block_tables,
                        blocks_per_slot, n_heads, n_kv_heads, head_dim, kv_dim,
                        swa_window, n_splits | fin_bit, rows, k1, scale, Mp_g);
                return pd_spec_fa_krs_go<32u, 1u, 256u, 1u>(fgrid, stream,
                    (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                    (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                    (const unsigned int*)slots, (const uint32_t*)block_tables,
                    blocks_per_slot, n_heads, n_kv_heads, head_dim, kv_dim,
                    swa_window, n_splits | fin_bit, rows, k1, scale, Mp_g);
            }
        }
        const uint32_t smem_sb32 = Mp_g * hp * 2u + 2u * PT * hp * 2u
            + Mp_g * (PT + 1u) * 4u + 3u * Mp_g * 4u + Mp_g * (PT + 8u) * 2u;
        if (!pin_db && fpad && M <= 64u && smem_sb32 <= (uint32_t)fa_max_smem
            && 2u * smem_g > (uint32_t)sm_smem && 2u * smem_sb32 <= (uint32_t)sm_smem) {
            const uint32_t mt = (M + 15u) / 16u;
            const uint32_t tasks = mt * (head_dim / 64u);
            dim3 fgrid(n_kv_heads, rows / k1, n_splits);
            if (a16) return -3;
            if (tasks > 8u)
                return f8 ? pd_spec_fa_go<PT, 4u, false, true>(fgrid, smem_sb32, stream,
                    (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                    (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                    (const unsigned int*)slots, (const uint32_t*)block_tables,
                    blocks_per_slot, n_heads, n_kv_heads, head_dim, kv_dim,
                    swa_window, n_splits | fin_bit, rows, k1, scale)
                : pd_spec_fa_go<PT, 4u, false>(fgrid, smem_sb32, stream,
                    (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                    (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                    (const unsigned int*)slots, (const uint32_t*)block_tables,
                    blocks_per_slot, n_heads, n_kv_heads, head_dim, kv_dim,
                    swa_window, n_splits | fin_bit, rows, k1, scale);
            return f8 ? pd_spec_fa_go<PT, 1u, false, true>(fgrid, smem_sb32, stream,
                (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                (const unsigned int*)slots, (const uint32_t*)block_tables,
                blocks_per_slot, n_heads, n_kv_heads, head_dim, kv_dim,
                swa_window, n_splits | fin_bit, rows, k1, scale)
            : pd_spec_fa_go<PT, 1u, false>(fgrid, smem_sb32, stream,
                (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                (const unsigned int*)slots, (const uint32_t*)block_tables,
                blocks_per_slot, n_heads, n_kv_heads, head_dim, kv_dim,
                swa_window, n_splits | fin_bit, rows, k1, scale);
        }
        if (M <= 64u && smem_g <= (uint32_t)fa_max_smem) {
            if (a16) return -3;
            const uint32_t smem = smem_g;
            static uint32_t fa_set = 0;
            if (smem > 48u * 1024u && smem > fa_set) {
                cudaFuncSetAttribute((const void*)pd_attn_spec_fa_kernel<PT, 4u>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
                cudaFuncSetAttribute((const void*)pd_attn_spec_fa_kernel<PT, 1u>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
                cudaFuncSetAttribute((const void*)pd_attn_spec_fa_kernel<PT, 4u, true, false>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
                cudaFuncSetAttribute((const void*)pd_attn_spec_fa_kernel<PT, 1u, true, false>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
                fa_set = smem;
            }
            const uint32_t mt = (M + 15u) / 16u;
            const uint32_t tasks = mt * (head_dim / 64u);
            dim3 fgrid(n_kv_heads, rows / k1, n_splits);
            if (!fpad) {
                if (tasks > 8u)
                    pd_attn_spec_fa_kernel<PT, 4u, true, false><<<fgrid, 256, smem, (cudaStream_t)stream>>>(
                        (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                        (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                        (const unsigned int*)slots, (const uint32_t*)block_tables,
                        blocks_per_slot, n_heads, n_kv_heads, head_dim, kv_dim,
                        swa_window, n_splits | fin_bit, rows, k1, scale);
                else
                    pd_attn_spec_fa_kernel<PT, 1u, true, false><<<fgrid, 256, smem, (cudaStream_t)stream>>>(
                        (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                        (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                        (const unsigned int*)slots, (const uint32_t*)block_tables,
                        blocks_per_slot, n_heads, n_kv_heads, head_dim, kv_dim,
                        swa_window, n_splits | fin_bit, rows, k1, scale);
                return pd_launch_status();
            }
            if (tasks > 8u)
                return f8 ? pd_spec_fa_go<PT, 4u, true, true>(fgrid, smem, stream,
                    (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                    (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                    (const unsigned int*)slots, (const uint32_t*)block_tables,
                    blocks_per_slot, n_heads, n_kv_heads, head_dim, kv_dim,
                    swa_window, n_splits | fin_bit, rows, k1, scale)
                : pd_spec_fa_go<PT, 4u, true>(fgrid, smem, stream,
                    (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                    (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                    (const unsigned int*)slots, (const uint32_t*)block_tables,
                    blocks_per_slot, n_heads, n_kv_heads, head_dim, kv_dim,
                    swa_window, n_splits | fin_bit, rows, k1, scale);
            return f8 ? pd_spec_fa_go<PT, 1u, true, true>(fgrid, smem, stream,
                (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                (const unsigned int*)slots, (const uint32_t*)block_tables,
                blocks_per_slot, n_heads, n_kv_heads, head_dim, kv_dim,
                swa_window, n_splits | fin_bit, rows, k1, scale)
            : pd_spec_fa_go<PT, 1u, true>(fgrid, smem, stream,
                (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                (const unsigned int*)slots, (const uint32_t*)block_tables,
                blocks_per_slot, n_heads, n_kv_heads, head_dim, kv_dim,
                swa_window, n_splits | fin_bit, rows, k1, scale);
        }
        // smem-constrained fallback (sm_120 global layers): PT=16 with a
        // SINGLE-buffered KV ring - the 32KB double-buffer was exactly the
        // overflow. Fits hd512 to M<=48 (~87KB; M=64/k1 7-8 still overflows
        // by 3.8KB and keeps the walk). One cp.async stall per tile instead
        // of overlap; still far ahead of the 1.76ms/launch per-row walk.
        constexpr uint32_t PTS = 16u;
        const uint32_t hps = head_dim + (fpad ? 8u : 0u);
        const uint32_t smem_s = Mp_g * hps * 2u + 2u * PTS * hps * 2u
            + Mp_g * (PTS + (fpad ? 1u : 0u)) * 4u + 3u * Mp_g * 4u
            + Mp_g * (PTS + (fpad ? 8u : 0u)) * 2u;
        // Deep-k GLB arm: adaptive k_now doubles to 7-8 after
        // full accepts, and those rounds' M=56-64 skipped the whole FA
        // block (the f16 route wants ~209KB there - hence the M<=48 cap
        // below) and fell to the generic GQA walk at ~4.4ms/layer (60
        // calls = 1.1% of the imax window). The QK8+VR krs shape needs
        // only 80KB at Mp=64 (e4m3 Q + raw K + raw V), so it fits where
        // f16 can't. Numerics: the landed GLB QK8 class (VR bit-equal).
        // TPW=4 covers Mp=64 exactly (mt 4 x slices 8 = 32 tasks/8 warps).
        if (f8 && fpad && head_dim == 512u && group == 8u && M > 32u
            && M <= 64u && pd_spec_fa_krs() && pd_spec_qk8()) {
            const uint32_t smem_dk = Mp_g * (head_dim + 16u)
                + 32u * (head_dim + 16u) + 32u * (head_dim + 16u)
                + Mp_g * 33u * 4u + 3u * Mp_g * 4u + Mp_g * 40u * 2u;
            if (smem_dk <= (uint32_t)fa_max_smem) {
                dim3 fgrid(n_kv_heads, rows / k1, n_splits);
                // P8 on the deep-k arm: same VR+QK8 structure, e4m3-P PV
                // (proto -4.2% on the no-DBK VR arm; DBK doesn't fit at
                // Mp=64 - 113KB > the cap). Strip shrinks 5.1->3.1KB.
                if (pd_spec_p8()) {
                    if (a16)
                        return pd_spec_fa_krs_go<32u, 4u, 512u, 3u, true, true, false, __half, true>(
                            fgrid, stream,
                            (const __half*)q, (const __half*)pool_k, (const __half*)pool_v,
                            (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                            (const unsigned int*)slots, (const uint32_t*)block_tables,
                            blocks_per_slot, n_heads, n_kv_heads, head_dim, kv_dim,
                            swa_window, n_splits | fin_bit, rows, k1, scale, Mp_g);
                    return pd_spec_fa_krs_go<32u, 4u, 512u, 3u, true, true, false, float, true>(
                        fgrid, stream,
                        (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                        (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                        (const unsigned int*)slots, (const uint32_t*)block_tables,
                        blocks_per_slot, n_heads, n_kv_heads, head_dim, kv_dim,
                        swa_window, n_splits | fin_bit, rows, k1, scale, Mp_g);
                }
                if (a16)
                    return pd_spec_fa_krs_go<32u, 4u, 512u, 3u, true, true, false, __half>(
                        fgrid, stream,
                        (const __half*)q, (const __half*)pool_k, (const __half*)pool_v,
                        (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                        (const unsigned int*)slots, (const uint32_t*)block_tables,
                        blocks_per_slot, n_heads, n_kv_heads, head_dim, kv_dim,
                        swa_window, n_splits | fin_bit, rows, k1, scale, Mp_g);
                return pd_spec_fa_krs_go<32u, 4u, 512u, 3u, true, true>(fgrid, stream,
                    (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                    (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                    (const unsigned int*)slots, (const uint32_t*)block_tables,
                    blocks_per_slot, n_heads, n_kv_heads, head_dim, kv_dim,
                    swa_window, n_splits | fin_bit, rows, k1, scale, Mp_g);
            }
        }
        if (M <= 48u && smem_s <= (uint32_t)fa_max_smem) {
            static uint32_t fas_set = 0;
            if (smem_s > 48u * 1024u && smem_s > fas_set) {
                cudaFuncSetAttribute((const void*)pd_attn_spec_fa_kernel<PTS, 4u, false>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, smem_s);
                cudaFuncSetAttribute((const void*)pd_attn_spec_fa_kernel<PTS, 1u, false>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, smem_s);
                cudaFuncSetAttribute((const void*)pd_attn_spec_fa_kernel<PTS, 4u, false, false>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, smem_s);
                cudaFuncSetAttribute((const void*)pd_attn_spec_fa_kernel<PTS, 1u, false, false>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, smem_s);
                fas_set = smem_s;
            }
            const uint32_t mt = (M + 15u) / 16u;
            const uint32_t tasks = mt * (head_dim / 64u);
            dim3 fgrid(n_kv_heads, rows / k1, n_splits);
            if (!fpad) {
                if (a16) return -3;
                if (tasks > 8u)
                    pd_attn_spec_fa_kernel<PTS, 4u, false, false><<<fgrid, 256, smem_s, (cudaStream_t)stream>>>(
                        (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                        (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                        (const unsigned int*)slots, (const uint32_t*)block_tables,
                        blocks_per_slot, n_heads, n_kv_heads, head_dim, kv_dim,
                        swa_window, n_splits | fin_bit, rows, k1, scale);
                else
                    pd_attn_spec_fa_kernel<PTS, 1u, false, false><<<fgrid, 256, smem_s, (cudaStream_t)stream>>>(
                        (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                        (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                        (const unsigned int*)slots, (const uint32_t*)block_tables,
                        blocks_per_slot, n_heads, n_kv_heads, head_dim, kv_dim,
                        swa_window, n_splits | fin_bit, rows, k1, scale);
                return pd_launch_status();
            }
            // krs: the GLB hd512/G8 F8 geometry at M<=32 fits a
            // PT=32 fp8-resident-K tile under the 99KB cap - -29.9% on the
            // occ-1 arm. Larger M keeps the PT16 route.
            if (f8 && head_dim == 512u && n_heads / n_kv_heads == 8u
                && M <= 32u && pd_spec_fa_krs()) {
                // VR ladder (opt-in): raw V halves the V smem - PT 32
                // (bit-equal) or 48 (the tile the f16 route can't fit)
                if (pd_spec_fa_vr() & 2) {
                    if (a16) return -3;
                    const int vpt = pd_spec_vr_pt();
                    const uint32_t ptw = (vpt == 48) ? 48u : 32u;
                    const uint32_t smem_vr = Mp_g * hps * 2u + ptw * (head_dim + 16u)
                        + ((ptw + 15u) & ~15u) * (head_dim + 16u)
                        + Mp_g * (ptw + 1u) * 4u + 3u * Mp_g * 4u
                        + Mp_g * (ptw + 8u) * 2u;
                    if (smem_vr <= (uint32_t)fa_max_smem) {
                        if (ptw == 48u)
                            return pd_spec_fa_krs_go<48u, 4u, 512u, 3u, true>(fgrid, stream,
                                (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                                (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                                (const unsigned int*)slots, (const uint32_t*)block_tables,
                                blocks_per_slot, n_heads, n_kv_heads, head_dim, kv_dim,
                                swa_window, n_splits | fin_bit, rows, k1, scale, Mp_g);
                        return pd_spec_fa_krs_go<32u, 4u, 512u, 3u, true>(fgrid, stream,
                            (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                            (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                            (const unsigned int*)slots, (const uint32_t*)block_tables,
                            blocks_per_slot, n_heads, n_kv_heads, head_dim, kv_dim,
                            swa_window, n_splits | fin_bit, rows, k1, scale, Mp_g);
                    }
                }
                const uint32_t smem32 = Mp_g * hps * 2u + 32u * (head_dim + 16u)
                    + 32u * hps * 2u + Mp_g * 33u * 4u + 3u * Mp_g * 4u
                    + Mp_g * 40u * 2u;
                if (smem32 <= (uint32_t)fa_max_smem) {
                    // P8+QK8+VR+DBK serve default: e4m3-P PV
                    // over the double-buffered raw-KV walk - proto fin
                    // 265.2 vs elected xV 287.5 (-7.8%), sp4 -3.6%. ~88.5KB
                    // at Mp=32, occ-1 like every GLB arm.
                    if (pd_spec_qk8() && pd_spec_p8() && !(pd_spec_fa_dbk() & 2)
                        && !(pd_spec_fa_vr() & 2)) {
                        const uint32_t smem_p8 = Mp_g * (head_dim + 16u)
                            + 4u * 32u * (head_dim + 16u)
                            + Mp_g * 33u * 4u + 3u * Mp_g * 4u + Mp_g * 48u;
                        if (smem_p8 <= (uint32_t)fa_max_smem) {
                            // KVS: split-commit walk on this arm
                            // only - the one it was measured on (fin -3.5%,
                            // sp4 -2.5%; bit-equal, no class change)
                            if (pd_spec_kvs()) {
                                if (a16)
                                    return pd_spec_fa_krs_go<32u, 4u, 512u, 3u, true, true, true, __half, true, true>(
                                        fgrid, stream,
                                        (const __half*)q, (const __half*)pool_k, (const __half*)pool_v,
                                        (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                                        (const unsigned int*)slots, (const uint32_t*)block_tables,
                                        blocks_per_slot, n_heads, n_kv_heads, head_dim, kv_dim,
                                        swa_window, n_splits | fin_bit, rows, k1, scale, Mp_g);
                                return pd_spec_fa_krs_go<32u, 4u, 512u, 3u, true, true, true, float, true, true>(
                                    fgrid, stream,
                                    (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                                    (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                                    (const unsigned int*)slots, (const uint32_t*)block_tables,
                                    blocks_per_slot, n_heads, n_kv_heads, head_dim, kv_dim,
                                    swa_window, n_splits | fin_bit, rows, k1, scale, Mp_g);
                            }
                            if (a16)
                                return pd_spec_fa_krs_go<32u, 4u, 512u, 3u, true, true, true, __half, true>(
                                    fgrid, stream,
                                    (const __half*)q, (const __half*)pool_k, (const __half*)pool_v,
                                    (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                                    (const unsigned int*)slots, (const uint32_t*)block_tables,
                                    blocks_per_slot, n_heads, n_kv_heads, head_dim, kv_dim,
                                    swa_window, n_splits | fin_bit, rows, k1, scale, Mp_g);
                            return pd_spec_fa_krs_go<32u, 4u, 512u, 3u, true, true, true, float, true>(
                                fgrid, stream,
                                (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                                (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                                (const unsigned int*)slots, (const uint32_t*)block_tables,
                                blocks_per_slot, n_heads, n_kv_heads, head_dim, kv_dim,
                                swa_window, n_splits | fin_bit, rows, k1, scale, Mp_g);
                        }
                    }
                    // DBK GLB rung: VR+QK8+DB at PT32 (91.6KB, still occ-1
                    // - this arm has no co-residency, so the doubled tile
                    // buys overlap it can get no other way). VR is bit-
                    // equal at equal PT; numerics = the landed QK8 class.
                    if (a16 && (pd_spec_fa_dbk() & 2)) return -3;
                    if (pd_spec_qk8() && (pd_spec_fa_dbk() & 2))
                        return pd_spec_fa_krs_go<32u, 4u, 512u, 3u, true, true, true>(
                            fgrid, stream,
                            (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                            (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                            (const unsigned int*)slots, (const uint32_t*)block_tables,
                            blocks_per_slot, n_heads, n_kv_heads, head_dim, kv_dim,
                            swa_window, n_splits | fin_bit, rows, k1, scale, Mp_g);
                    if (a16 && pd_spec_qk8())
                        return pd_spec_fa_krs_go<32u, 4u, 512u, 3u, false, true, false, __half>(
                            fgrid, stream,
                            (const __half*)q, (const __half*)pool_k, (const __half*)pool_v,
                            (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                            (const unsigned int*)slots, (const uint32_t*)block_tables,
                            blocks_per_slot, n_heads, n_kv_heads, head_dim, kv_dim,
                            swa_window, n_splits | fin_bit, rows, k1, scale, Mp_g);
                    if (pd_spec_qk8())
                        return pd_spec_fa_krs_go<32u, 4u, 512u, 3u, false, true>(fgrid, stream,
                            (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                            (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                            (const unsigned int*)slots, (const uint32_t*)block_tables,
                            blocks_per_slot, n_heads, n_kv_heads, head_dim, kv_dim,
                            swa_window, n_splits | fin_bit, rows, k1, scale, Mp_g);
                    if (a16) return -3;  // GLB without QK8 has no a16 arm
                    return pd_spec_fa_krs_go<32u, 4u, 512u, 3u>(fgrid, stream,
                        (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                        (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                        (const unsigned int*)slots, (const uint32_t*)block_tables,
                        blocks_per_slot, n_heads, n_kv_heads, head_dim, kv_dim,
                        swa_window, n_splits | fin_bit, rows, k1, scale, Mp_g);
                }
            }
            if (a16) return -3;
            if (tasks > 8u)
                return f8 ? pd_spec_fa_go<PTS, 4u, false, true>(fgrid, smem_s, stream,
                    (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                    (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                    (const unsigned int*)slots, (const uint32_t*)block_tables,
                    blocks_per_slot, n_heads, n_kv_heads, head_dim, kv_dim,
                    swa_window, n_splits | fin_bit, rows, k1, scale)
                : pd_spec_fa_go<PTS, 4u, false>(fgrid, smem_s, stream,
                    (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                    (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                    (const unsigned int*)slots, (const uint32_t*)block_tables,
                    blocks_per_slot, n_heads, n_kv_heads, head_dim, kv_dim,
                    swa_window, n_splits | fin_bit, rows, k1, scale);
            return f8 ? pd_spec_fa_go<PTS, 1u, false, true>(fgrid, smem_s, stream,
                (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                (const unsigned int*)slots, (const uint32_t*)block_tables,
                blocks_per_slot, n_heads, n_kv_heads, head_dim, kv_dim,
                swa_window, n_splits | fin_bit, rows, k1, scale)
            : pd_spec_fa_go<PTS, 1u, false>(fgrid, smem_s, stream,
                (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                (const unsigned int*)slots, (const uint32_t*)block_tables,
                blocks_per_slot, n_heads, n_kv_heads, head_dim, kv_dim,
                swa_window, n_splits | fin_bit, rows, k1, scale);
        }
    }
    if (fin_bit) return -2; // FIN is FA-only: caller keeps partial+combine
    if (a16) return -3;      // a16 arms only the krs serve elections
    const uint32_t gsub = group >= 2u ? 2u : 1u;
    if (group % gsub != 0u) return -2;
    const uint32_t nth = head_dim > 256u ? 512u : 256u;
    if (gsub * head_dim / 2u > nth) return -2; // acc[j] holds one pair/thread
    const uint32_t tile = head_dim > 256u ? 8u : 16u;
    const uint32_t rh = k1 * gsub;
    const uint32_t smem = (uint32_t)PD_GQA_SMEM(rh, head_dim, tile);
    static uint32_t spec_set = 0;
    if (spec_set == 0) {
        pd_prefer_max_shared(pd_attn_spec_gqa_paged_kernel<__half, 8u>);
        pd_prefer_max_shared(pd_attn_spec_gqa_paged_kernel<__half, 16u>);
        pd_prefer_max_shared(pd_attn_spec_gqa_paged_kernel<__nv_fp8_e4m3, 8u>);
        pd_prefer_max_shared(pd_attn_spec_gqa_paged_kernel<__nv_fp8_e4m3, 16u>);
        spec_set = 1;
    }
    if (smem > 48u * 1024u && smem > spec_set) {
        cudaFuncSetAttribute((const void*)pd_attn_spec_gqa_paged_kernel<__half, 8u>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        cudaFuncSetAttribute((const void*)pd_attn_spec_gqa_paged_kernel<__half, 16u>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        cudaFuncSetAttribute((const void*)pd_attn_spec_gqa_paged_kernel<__nv_fp8_e4m3, 8u>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        cudaFuncSetAttribute((const void*)pd_attn_spec_gqa_paged_kernel<__nv_fp8_e4m3, 16u>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        spec_set = smem;
    }
    dim3 grid(n_kv_heads * (group / gsub), rows / k1, n_splits);
    if (kv_dtype == PD_KV_FP8_E4M3) {
        if (tile == 8u)
            pd_attn_spec_gqa_paged_kernel<__nv_fp8_e4m3, 8u><<<grid, nth, smem, (cudaStream_t)stream>>>(
                (const float*)q, (const __nv_fp8_e4m3*)pool_k, (const __nv_fp8_e4m3*)pool_v,
                (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                (const unsigned int*)slots, (const uint32_t*)block_tables, blocks_per_slot,
                n_heads, n_kv_heads, head_dim, kv_dim, swa_window, n_splits, rows, k1, gsub, scale);
        else
            pd_attn_spec_gqa_paged_kernel<__nv_fp8_e4m3, 16u><<<grid, nth, smem, (cudaStream_t)stream>>>(
                (const float*)q, (const __nv_fp8_e4m3*)pool_k, (const __nv_fp8_e4m3*)pool_v,
                (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                (const unsigned int*)slots, (const uint32_t*)block_tables, blocks_per_slot,
                n_heads, n_kv_heads, head_dim, kv_dim, swa_window, n_splits, rows, k1, gsub, scale);
    } else {
        if (tile == 8u)
            pd_attn_spec_gqa_paged_kernel<__half, 8u><<<grid, nth, smem, (cudaStream_t)stream>>>(
                (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                (const unsigned int*)slots, (const uint32_t*)block_tables, blocks_per_slot,
                n_heads, n_kv_heads, head_dim, kv_dim, swa_window, n_splits, rows, k1, gsub, scale);
        else
            pd_attn_spec_gqa_paged_kernel<__half, 16u><<<grid, nth, smem, (cudaStream_t)stream>>>(
                (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
                (float*)out_o, (float*)out_ml, (const unsigned int*)positions,
                (const unsigned int*)slots, (const uint32_t*)block_tables, blocks_per_slot,
                n_heads, n_kv_heads, head_dim, kv_dim, swa_window, n_splits, rows, k1, gsub, scale);
    }
    return pd_launch_status();
}

PD_EXPORT
int pd_attn_spec_batch_paged(const void* q, const void* pool_k, const void* pool_v,
                             void* out_o, void* out_ml, const void* positions,
                             const void* slots, const void* block_tables,
                             uint32_t blocks_per_slot, uint32_t n_heads,
                             uint32_t n_kv_heads, uint32_t head_dim, uint32_t kv_dim,
                             uint32_t swa_window, uint32_t n_splits, uint32_t rows,
                             uint32_t k1, float scale, uint32_t kv_dtype, void* stream) {
    return pd_attn_spec_batch_paged_impl(q, pool_k, pool_v, out_o, out_ml, positions,
                                         slots, block_tables, blocks_per_slot, n_heads,
                                         n_kv_heads, head_dim, kv_dim, swa_window,
                                         n_splits, rows, k1, scale, kv_dtype, 0u, stream);
}

// a16 twin for the attention streams: q is an f16 plane. Only the krs
// serve-default elections have TQ arms; every A/B pin env that diverts the
// election returns -3 (pair those pins with PADDOCK_G4_NO_ATTN16). Partials
// and ml stay f32. Appended as its own export per the ABI growth rule.
PD_EXPORT
int pd_attn_spec_batch_paged2(const void* q, const void* pool_k, const void* pool_v,
                              void* out_o, void* out_ml, const void* positions,
                              const void* slots, const void* block_tables,
                              uint32_t blocks_per_slot, uint32_t n_heads,
                              uint32_t n_kv_heads, uint32_t head_dim, uint32_t kv_dim,
                              uint32_t swa_window, uint32_t n_splits, uint32_t rows,
                              uint32_t k1, float scale, uint32_t kv_dtype, uint32_t a16,
                              void* stream) {
    return pd_attn_spec_batch_paged_impl(q, pool_k, pool_v, out_o, out_ml, positions,
                                         slots, block_tables, blocks_per_slot, n_heads,
                                         n_kv_heads, head_dim, kv_dim, swa_window,
                                         n_splits, rows, k1, scale, kv_dtype, a16, stream);
}

// Door 3: spec-verify FIN entry - the FA route at n_splits==1
// with in-kernel finalize (bit-identical to walk + the -inf-sink combine;
// batch-major final rows land in `out`, ml is dead scratch). Returns -2
// whenever the FA geometry can't engage (sm_120 hd512 global layers, tiny
// chunk counts) so the caller keeps the partial+combine chain per layer.
PD_EXPORT
int pd_attn_spec_batch_fin(const void* q, const void* pool_k, const void* pool_v,
                           void* out, void* out_ml, const void* positions,
                           const void* slots, const void* block_tables,
                           uint32_t blocks_per_slot, uint32_t n_heads,
                           uint32_t n_kv_heads, uint32_t head_dim, uint32_t kv_dim,
                           uint32_t swa_window, uint32_t rows, uint32_t k1,
                           float scale, uint32_t kv_dtype, void* stream) {
    return pd_attn_spec_batch_paged(q, pool_k, pool_v, out, out_ml, positions,
                                    slots, block_tables, blocks_per_slot, n_heads,
                                    n_kv_heads, head_dim, kv_dim, swa_window,
                                    0x80000001u, rows, k1, scale, kv_dtype, stream);
}

// slot 425: FIN twin that stores the finalized rows as
// e4m3 at STATIC scale 1.0 directly into the wo-in quantized plane
// (out = pf_e4q, i8) - the standalone quantize_e4m3_row launch disappears
// and the caller feeds the GEMM a ones xrs vector. fp8 relative precision
// is scale-invariant (3-bit mantissa at every binade); the per-row recipe
// only moves the clip (>448, satfinite) / denorm (<2^-9) cliffs - muse
// o-gate precedent, comm/PPL-gated by the engine A/B. Same accept
// envelope as pd_attn_spec_batch_fin by CONSTRUCTION (same impl, one
// extra sentinel bit): -2 wherever the f32 fin would refuse, and the
// caller keeps the fin + quantize chain. A probe FALSIFIED the
// mixed-input cutlass route this bit replaces.
PD_EXPORT
int pd_attn_spec_batch_fin_e4s(const void* q, const void* pool_k, const void* pool_v,
                               void* out, void* out_ml, const void* positions,
                               const void* slots, const void* block_tables,
                               uint32_t blocks_per_slot, uint32_t n_heads,
                               uint32_t n_kv_heads, uint32_t head_dim, uint32_t kv_dim,
                               uint32_t swa_window, uint32_t rows, uint32_t k1,
                               float scale, uint32_t kv_dtype, void* stream) {
    return pd_attn_spec_batch_paged(q, pool_k, pool_v, out, out_ml, positions,
                                    slots, block_tables, blocks_per_slot, n_heads,
                                    n_kv_heads, head_dim, kv_dim, swa_window,
                                    0xC0000001u, rows, k1, scale, kv_dtype, stream);
}

// slot 423: FIN twin whose epilogue quantizes the
// finalized rows in-kernel - e4m3 plane [rows x n_heads*head_dim] into
// out_q plus f32 per-row scales into out_rs, bit-identical to the
// standalone pd_quantize_e4m3_row recipe on the same values. Legal only
// when one CTA owns whole output rows: the SWA verify geometry
// (n_kv_heads==1, G16, hd256, f8 KV) on the DB FA route. Anything else
// returns -2 and the caller keeps the f32 fin + quantize chain. The
// geometry/election gates below MIRROR pd_attn_spec_batch_paged_impl so
// this arm never fires where the f32 fin would have taken another route.
PD_EXPORT
int pd_attn_spec_batch_fin_e4(const void* q, const void* pool_k, const void* pool_v,
                              void* out_q, void* out_rs, const void* positions,
                              const void* slots, const void* block_tables,
                              uint32_t blocks_per_slot, uint32_t n_heads,
                              uint32_t n_kv_heads, uint32_t head_dim, uint32_t kv_dim,
                              uint32_t swa_window, uint32_t rows, uint32_t k1,
                              float scale, uint32_t kv_dtype, void* stream) {
    if (rows == 0) return 0;
    const uint32_t group = n_kv_heads ? n_heads / n_kv_heads : 0;
    if (n_kv_heads != 1u || group != 16u || head_dim != 256u
        || n_heads != n_kv_heads * group || k1 == 0u || k1 > 8u
        || rows % k1 != 0u || rows / k1 < 4u
        || kv_dtype != PD_KV_FP8_E4M3) return -2;
    if (!pd_spec_fa_mode() || pd_spec_fa_nopad()) return -2;
    static int fa_f8 = -1;
    if (fa_f8 < 0) fa_f8 = pd_env("PADDOCK_NO_SPEC_FA_F8") ? 0 : 1;
    if (!fa_f8) return -2;
    const uint32_t M = k1 * group;
    if (M > 64u) return -2;
    constexpr uint32_t PT = 32u;
    const uint32_t mt = (M + 15u) / 16u;
    const uint32_t Mp = mt * 16u;
    const uint32_t hp = head_dim + 8u;                     // fpad layout
    const uint32_t smem_g = Mp * hp * 2u + 4u * PT * hp * 2u
        + Mp * (PT + 1u) * 4u + 3u * Mp * 4u + Mp * (PT + 8u) * 2u;
    const uint32_t smem_sb32 = Mp * hp * 2u + 2u * PT * hp * 2u
        + Mp * (PT + 1u) * 4u + 3u * Mp * 4u + Mp * (PT + 8u) * 2u;
    static int fa_max_smem = -1;
    if (fa_max_smem < 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        if (cudaDeviceGetAttribute(&fa_max_smem,
                cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess)
            fa_max_smem = 48 * 1024;
    }
    if (smem_g > (uint32_t)fa_max_smem) return -2;
    static int sm_smem = -1;
    if (sm_smem < 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        if (cudaDeviceGetAttribute(&sm_smem,
                cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev) != cudaSuccess)
            sm_smem = 100 * 1024;
    }
    static int pin_db = -1;
    if (pin_db < 0) pin_db = pd_env("PADDOCK_SPEC_FA_DB") ? 1 : 0;
    // impl would elect the single-buffer ring here - only the DB body is
    // instantiated with E4, so refuse and let the f32 route run
    if (!pin_db && smem_sb32 <= (uint32_t)fa_max_smem
        && 2u * smem_g > (uint32_t)sm_smem
        && 2u * smem_sb32 <= (uint32_t)sm_smem) return -2;
    const uint32_t tasks = mt * (head_dim / 64u);
    if (tasks <= 8u) return -2;                            // TPW=4 body only
    dim3 fgrid(n_kv_heads, rows / k1, 1u);
    static uint32_t e4_set = 0;
    if (smem_g > 48u * 1024u && smem_g > e4_set) {
        cudaFuncSetAttribute(
            (const void*)pd_attn_spec_fa_kernel<PT, 4u, true, true, 256u, 4u, true, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_g);
        e4_set = smem_g;
    }
    pd_pdl_go(pd_attn_spec_fa_kernel<PT, 4u, true, true, 256u, 4u, true, true>,
              fgrid, 256, smem_g, (cudaStream_t)stream,
              (const float*)q, (const __half*)pool_k, (const __half*)pool_v,
              (float*)out_q, (float*)out_rs, (const unsigned int*)positions,
              (const unsigned int*)slots, (const uint32_t*)block_tables,
              blocks_per_slot, n_heads, n_kv_heads, head_dim, kv_dim,
              swa_window, 1u | 0x80000000u, rows, k1, scale);
    return pd_launch_status();
}

// slot 381 - see pd_moe_topk_softmax_all_kernel (attn/decode.cuh).
// (logits [batch, n_expert], n_expert <= 256, k <= 16) -> idx/w [batch, k].
PD_EXPORT
int pd_moe_topk_softmax_all(const void* logits, uint32_t n_expert, uint32_t k,
                            void* out_idx, void* out_w, uint32_t batch, void* stream) {
    if (batch == 0) return 0;
    if (n_expert > 256u || k > 16u) return -3;
    pd_moe_topk_softmax_all_kernel<<<batch, 32, 0, (cudaStream_t)stream>>>(
        (const float*)logits, n_expert, k, (uint32_t*)out_idx, (float*)out_w);
    return pd_launch_status();
}
