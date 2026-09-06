// BF16 dense weight planes - the "read the tensor in the class the FILE ships
// it in" lane.
//
// Why this exists: UD quant files are MIXED. muse-glimmer's UD-Q8_K_XL keeps
// token_embd / output / attn_k / attn_v at BF16 while attn_q / attn_gate /
// attn_output / ffn_* are Q8_0 - the quantizer deliberately spends bytes on
// the planes it judges sensitive. The engine's rule for that is
// per-TENSOR dispatch, not a per-model switch, and the correctness spine is
// same-weights parity against llama.cpp on the identical GGUF. Down-quantizing
// a bf16 plane to Q8_0 at load would break both: different weights, no exact
// greedy target, and a quality choice silently overridden.
//
// So: keep the bytes, widen in-register. Weights stay bf16 in DRAM (file
// bytes, no 2x f32 inflation - the 202048-row LM head would otherwise cost
// 5.4 GB and 2x the decode read), activations stay f32, accumulation is f32.
// That is the same arithmetic class the Q8_0 lane runs and strictly more
// precise than it, so nothing downstream has to know which class a plane came
// from beyond picking the launcher.
//
// Layout is the same convention as the repacked Q8_0 planes: a GGUF [in, out]
// tensor is out rows of in contiguous elements, row-major, so `w + o*in_dim`
// is output row o. That is what makes these drop-in twins of
// pd_q8_0_gemv_repacked / pd_q8_0_gemm_repacked at the call site.
//
// The prefill arm is the mma.sync one (pd_bf16_gemm_mma below):
// paddleocr-vl serves bf16 as its PRIMARY class (official-GGUF planes verbatim,
// no quant ladder), which is exactly the condition this file's original header
// reserved the tensor-core arm for. It stages the f32 activations to bf16 in
// shared memory - the same class the parity reference computes: llama.cpp's
// batched BF16 path converts src1 to bf16 and runs cublasGemmEx bf16xbf16 with
// CUBLAS_COMPUTE_32F (ggml-cuda.cu batched_mul_mat_traits<GGML_TYPE_BF16>), and
// the greedy gate holds byte-exact against it. Weights are exact in
// bf16, so only the activation cast (~2^-8 relative) separates it from the
// f32-FMA tile; the correctness battery gates the switch. The f32-FMA tile
// GEMM stays as the fallback for older packs and ragged in_dim.

// One block per output row, 16 elements per thread: one 32-byte weight load
// (two int4) plus four float4 activation loads, so a warp issues fully
// coalesced 512-byte transactions. 128 threads, not 256, for the same reason
// pd_q8_0_gemv_repacked uses 128 - this die's maxThreadsPerMultiProcessor is
// 1536, so 128 gets 12 resident blocks/SM against 256's 6, and this kernel is
// bandwidth-bound end to end.
__global__ void pd_bf16_gemv_f32_kernel(
    const __nv_bfloat16* __restrict__ w, const float* __restrict__ bias,
    const float* __restrict__ x, float* __restrict__ y,
    uint32_t in_dim, uint32_t out_dim) {
    uint32_t o = blockIdx.x;
    if (o >= out_dim) return;
    uint32_t tid = threadIdx.x, nth = blockDim.x;
    // dep-free prologue: nothing above touches chain data, so the wait gates
    // only the x reads below. No-op under plain launches.
    PD_PDL_ARM();
    __shared__ float wsum[32];
    const __nv_bfloat16* row = w + (size_t)o * in_dim;
    float acc = 0.0f;
    if ((in_dim & 15u) == 0u) {
        for (uint32_t base = tid * 16u; base < in_dim; base += nth * 16u) {
            int4 w0 = *reinterpret_cast<const int4*>(row + base);
            int4 w1 = *reinterpret_cast<const int4*>(row + base + 8);
            const __nv_bfloat16* wa = reinterpret_cast<const __nv_bfloat16*>(&w0);
            const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&w1);
            float4 x0 = *reinterpret_cast<const float4*>(x + base);
            float4 x1 = *reinterpret_cast<const float4*>(x + base + 4);
            float4 x2 = *reinterpret_cast<const float4*>(x + base + 8);
            float4 x3 = *reinterpret_cast<const float4*>(x + base + 12);
            acc += __bfloat162float(wa[0]) * x0.x + __bfloat162float(wa[1]) * x0.y
                 + __bfloat162float(wa[2]) * x0.z + __bfloat162float(wa[3]) * x0.w
                 + __bfloat162float(wa[4]) * x1.x + __bfloat162float(wa[5]) * x1.y
                 + __bfloat162float(wa[6]) * x1.z + __bfloat162float(wa[7]) * x1.w
                 + __bfloat162float(wb[0]) * x2.x + __bfloat162float(wb[1]) * x2.y
                 + __bfloat162float(wb[2]) * x2.z + __bfloat162float(wb[3]) * x2.w
                 + __bfloat162float(wb[4]) * x3.x + __bfloat162float(wb[5]) * x3.y
                 + __bfloat162float(wb[6]) * x3.z + __bfloat162float(wb[7]) * x3.w;
        }
    } else {
        // ragged in_dim: scalar walk. No plane in any elected file lands here
        // (every in_dim is a multiple of 128), but a silent wrong answer on
        // one that did would be far worse than the slow path.
        for (uint32_t i = tid; i < in_dim; i += nth)
            acc += __bfloat162float(row[i]) * x[i];
    }
    for (uint32_t s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, s);
    uint32_t warp = tid >> 5, lane = tid & 31u;
    if (lane == 0) wsum[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        float v = 0.0f;
        uint32_t nwarps = (nth + 31u) >> 5;
        for (uint32_t w2 = 0; w2 < nwarps; ++w2) v += wsum[w2];
        if (bias) v += bias[o];
        y[o] = v;
    }
}

PD_EXPORT
int pd_bf16_gemv_f32(const void* w, const void* bias, const void* x, void* y,
                     uint32_t in_dim, uint32_t out_dim, void* stream) {
    if (out_dim == 0 || in_dim == 0) return 0;
    pd_pdl_go(pd_bf16_gemv_f32_kernel, out_dim, 128u, 0u, (cudaStream_t)stream,
              (const __nv_bfloat16*)w, (const float*)bias, (const float*)x,
              (float*)y, in_dim, out_dim);
    return pd_launch_status();
}

// ---------------------------------------------------------------------------
// Dual-plane swiglu GEMV (slot 546): one launch computes both dots of the
// shared-expert gate|up pair against the same x and stores silu(g)*u - the
// n=1 sh chain was 4 launches/pass (gate GEMV, up GEMV, a 15us swiglu over
// 640 floats, down GEMV); this folds the first three into one launch and
// AMORTISES the x reads across both planes. Same block-per-row shape as
// pd_bf16_gemv_f32 (sh planes are [2560 -> 640], below the nk arm's grid
// floor). Values: identical dot order per plane + the swiglu kernel's exact
// silu expression => same numeric class as the 3-launch path per plane,
// battery-judged.
__global__ void pd_bf16_gemv2_swiglu_kernel(
    const __nv_bfloat16* __restrict__ wg, const __nv_bfloat16* __restrict__ wu,
    const float* __restrict__ x, float* __restrict__ y,
    uint32_t in_dim, uint32_t out_dim) {
    uint32_t o = blockIdx.x;
    if (o >= out_dim) return;
    uint32_t tid = threadIdx.x, nth = blockDim.x;
    PD_PDL_ARM();
    __shared__ float wsum[64];
    const __nv_bfloat16* rg = wg + (size_t)o * in_dim;
    const __nv_bfloat16* ru = wu + (size_t)o * in_dim;
    float accg = 0.0f, accu = 0.0f;
    for (uint32_t base = tid * 16u; base < in_dim; base += nth * 16u) {
        int4 g0 = *reinterpret_cast<const int4*>(rg + base);
        int4 g1 = *reinterpret_cast<const int4*>(rg + base + 8);
        int4 u0 = *reinterpret_cast<const int4*>(ru + base);
        int4 u1 = *reinterpret_cast<const int4*>(ru + base + 8);
        const __nv_bfloat16* ga = reinterpret_cast<const __nv_bfloat16*>(&g0);
        const __nv_bfloat16* gb = reinterpret_cast<const __nv_bfloat16*>(&g1);
        const __nv_bfloat16* ua = reinterpret_cast<const __nv_bfloat16*>(&u0);
        const __nv_bfloat16* ub = reinterpret_cast<const __nv_bfloat16*>(&u1);
        float4 x0 = *reinterpret_cast<const float4*>(x + base);
        float4 x1 = *reinterpret_cast<const float4*>(x + base + 4);
        float4 x2 = *reinterpret_cast<const float4*>(x + base + 8);
        float4 x3 = *reinterpret_cast<const float4*>(x + base + 12);
        accg += __bfloat162float(ga[0]) * x0.x + __bfloat162float(ga[1]) * x0.y
              + __bfloat162float(ga[2]) * x0.z + __bfloat162float(ga[3]) * x0.w
              + __bfloat162float(ga[4]) * x1.x + __bfloat162float(ga[5]) * x1.y
              + __bfloat162float(ga[6]) * x1.z + __bfloat162float(ga[7]) * x1.w
              + __bfloat162float(gb[0]) * x2.x + __bfloat162float(gb[1]) * x2.y
              + __bfloat162float(gb[2]) * x2.z + __bfloat162float(gb[3]) * x2.w
              + __bfloat162float(gb[4]) * x3.x + __bfloat162float(gb[5]) * x3.y
              + __bfloat162float(gb[6]) * x3.z + __bfloat162float(gb[7]) * x3.w;
        accu += __bfloat162float(ua[0]) * x0.x + __bfloat162float(ua[1]) * x0.y
              + __bfloat162float(ua[2]) * x0.z + __bfloat162float(ua[3]) * x0.w
              + __bfloat162float(ua[4]) * x1.x + __bfloat162float(ua[5]) * x1.y
              + __bfloat162float(ua[6]) * x1.z + __bfloat162float(ua[7]) * x1.w
              + __bfloat162float(ub[0]) * x2.x + __bfloat162float(ub[1]) * x2.y
              + __bfloat162float(ub[2]) * x2.z + __bfloat162float(ub[3]) * x2.w
              + __bfloat162float(ub[4]) * x3.x + __bfloat162float(ub[5]) * x3.y
              + __bfloat162float(ub[6]) * x3.z + __bfloat162float(ub[7]) * x3.w;
    }
    for (uint32_t sh = 16; sh > 0; sh >>= 1) {
        accg += __shfl_down_sync(0xffffffffu, accg, sh);
        accu += __shfl_down_sync(0xffffffffu, accu, sh);
    }
    uint32_t warp = tid >> 5, lane = tid & 31u;
    if (lane == 0) { wsum[warp] = accg; wsum[32u + warp] = accu; }
    __syncthreads();
    if (tid == 0) {
        float g = 0.0f, u = 0.0f;
        uint32_t nwarps = (nth + 31u) >> 5;
        for (uint32_t w2 = 0; w2 < nwarps; ++w2) { g += wsum[w2]; u += wsum[32u + w2]; }
        y[o] = (g / (1.0f + expf(-g))) * u;
    }
}

PD_EXPORT
int pd_bf16_gemv2_swiglu(const void* wg, const void* wu, const void* x, void* y,
                         uint32_t in_dim, uint32_t out_dim, uint32_t batch,
                         void* stream) {
    if (out_dim == 0 || in_dim == 0 || batch == 0) return 0;
    if (batch != 1u || (in_dim & 15u)) return 1;   // decline: caller keeps 3-launch
    pd_pdl_go(pd_bf16_gemv2_swiglu_kernel, out_dim, 128u, 0u, (cudaStream_t)stream,
              (const __nv_bfloat16*)wg, (const __nv_bfloat16*)wu, (const float*)x,
              (float*)y, in_dim, out_dim);
    return pd_launch_status();
}

// ---------------------------------------------------------------------------
// Fused silu epilogue (slot 520). The qwen4_exp hyper-connection down plane is
// immediately followed by `m = silu(m * (1/hc))` over the low-rank rows only -
// the plane's tail rows are the folded inject logits and must pass through
// untouched. That elementwise pass is 95 launches/token of a kernel doing
// 320 elements of work, sitting on the critical path. Folding it into this
// epilogue is BIT-IDENTICAL: the dot is unchanged and the same two f32 ops run
// on the same value, just before the store instead of after the load.
__global__ void pd_bf16_gemv_silu_f32_kernel(
    const __nv_bfloat16* __restrict__ w, const float* __restrict__ bias,
    const float* __restrict__ x, float* __restrict__ y,
    __nv_bfloat16* __restrict__ y16,
    uint32_t in_dim, uint32_t out_dim, uint32_t silu_rows, float inv) {
    uint32_t o = blockIdx.x;
    if (o >= out_dim) return;
    uint32_t tid = threadIdx.x, nth = blockDim.x;
    PD_PDL_ARM();
    __shared__ float wsum[32];
    const __nv_bfloat16* row = w + (size_t)o * in_dim;
    float acc = 0.0f;
    if ((in_dim & 15u) == 0u) {
        for (uint32_t base = tid * 16u; base < in_dim; base += nth * 16u) {
            int4 w0 = *reinterpret_cast<const int4*>(row + base);
            int4 w1 = *reinterpret_cast<const int4*>(row + base + 8);
            const __nv_bfloat16* wa = reinterpret_cast<const __nv_bfloat16*>(&w0);
            const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&w1);
            float4 x0 = *reinterpret_cast<const float4*>(x + base);
            float4 x1 = *reinterpret_cast<const float4*>(x + base + 4);
            float4 x2 = *reinterpret_cast<const float4*>(x + base + 8);
            float4 x3 = *reinterpret_cast<const float4*>(x + base + 12);
            acc += __bfloat162float(wa[0]) * x0.x + __bfloat162float(wa[1]) * x0.y
                 + __bfloat162float(wa[2]) * x0.z + __bfloat162float(wa[3]) * x0.w
                 + __bfloat162float(wa[4]) * x1.x + __bfloat162float(wa[5]) * x1.y
                 + __bfloat162float(wa[6]) * x1.z + __bfloat162float(wa[7]) * x1.w
                 + __bfloat162float(wb[0]) * x2.x + __bfloat162float(wb[1]) * x2.y
                 + __bfloat162float(wb[2]) * x2.z + __bfloat162float(wb[3]) * x2.w
                 + __bfloat162float(wb[4]) * x3.x + __bfloat162float(wb[5]) * x3.y
                 + __bfloat162float(wb[6]) * x3.z + __bfloat162float(wb[7]) * x3.w;
        }
    } else {
        for (uint32_t i = tid; i < in_dim; i += nth)
            acc += __bfloat162float(row[i]) * x[i];
    }
    for (uint32_t s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, s);
    uint32_t warp = tid >> 5, lane = tid & 31u;
    if (lane == 0) wsum[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        float v = 0.0f;
        uint32_t nwarps = (nth + 31u) >> 5;
        for (uint32_t w2 = 0; w2 < nwarps; ++w2) v += wsum[w2];
        if (bias) v += bias[o];
        if (o < silu_rows) {
            const float t = v * inv;
            v = t * (1.0f / (1.0f + __expf(-t)));
        }
        y[o] = v;
        // bf16 MIRROR for the slot-547 TGV feed (writers mirror at the
        // store; per-call casts were eating the TGV win)
        if (y16) y16[o] = __float2bfloat16(v);
    }
}

PD_EXPORT
int pd_bf16_gemv_silu_f32(const void* w, const void* bias, const void* x, void* y,
                          void* y16, uint32_t in_dim, uint32_t out_dim,
                          uint32_t silu_rows, float inv, void* stream) {
    if (out_dim == 0 || in_dim == 0) return 0;
    // 128 threads means 5 serial int4 steps per thread on the HC down plane
    // (in=10240); the probe (bench/hcgemv_probe.cu) puts 256 at 4.39 us a
    // graph node vs 5.09 at 128, and 512/1024 back off again.
    static int nth_env = -2;
    if (nth_env == -2) {
        const char* e = getenv("PADDOCK_Q38FN_HCDN_THREADS");
        nth_env = (e && *e) ? atoi(e) : 0;
    }
    uint32_t nth_s = 256u;
    if (nth_env >= 32 && nth_env <= 1024) nth_s = (uint32_t)nth_env;
    pd_pdl_go(pd_bf16_gemv_silu_f32_kernel, out_dim, nth_s, 0u, (cudaStream_t)stream,
              (const __nv_bfloat16*)w, (const float*)bias, (const float*)x,
              (float*)y, (__nv_bfloat16*)y16, in_dim, out_dim, silu_rows, inv);
    return pd_launch_status();
}

// ---------------------------------------------------------------------------
// NARROW-K arm (slot 518). The stock gemv above gives every output row its own
// 128-thread block and walks it as `base = tid*16`, which assumes
// in_dim >= 128*16 = 2048. Below that the block is mostly idle AND the launch
// degenerates into one tiny block per row: the qwen4_exp hyper-connection up
// plane [in=320, out=10240] leaves 20 of 128 threads loading and issues 10240
// blocks that read 640 B each - measured 11.5 us for 6.55 MB = 570 GB/s, 7% of
// the B200 roof, on a die where this same kernel reaches 3199 GB/s on the
// lm_head. This arm gives each output row one WARP and packs WPB rows per
// block, so the grid collapses to out_dim/WPB and each warp reads 512 B
// contiguous per step.
//
// Kept as its own export rather than a shape branch inside pd_bf16_gemv_f32:
// the reduction order differs (32-lane shuffle vs the 128-thread two-level
// tree), so folding it into the stock entry would move every other lane's
// numerics for a plane shape they never hit.
template <uint32_t WPB>
__global__ __launch_bounds__(WPB * 32u) void pd_bf16_gemv_nk_f32_kernel(
    const __nv_bfloat16* __restrict__ w, const float* __restrict__ bias,
    const float* __restrict__ x, float* __restrict__ y,
    uint32_t in_dim, uint32_t out_dim) {
    PD_PDL_ARM();
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t o = blockIdx.x * WPB + warp;
    if (o >= out_dim) return;
    const __nv_bfloat16* __restrict__ row = w + (size_t)o * in_dim;
    float acc = 0.0f;
    if ((in_dim & 7u) == 0u) {
        // one int4 (8 bf16) per lane per step => 32 lanes cover 256 contiguous
        // elements = 512 B of W per warp per step, fully coalesced.
        for (uint32_t base = lane * 8u; base < in_dim; base += 256u) {
            int4 wv = *reinterpret_cast<const int4*>(row + base);
            const __nv_bfloat16* wa = reinterpret_cast<const __nv_bfloat16*>(&wv);
            float4 x0 = *reinterpret_cast<const float4*>(x + base);
            float4 x1 = *reinterpret_cast<const float4*>(x + base + 4);
            acc += __bfloat162float(wa[0]) * x0.x + __bfloat162float(wa[1]) * x0.y
                 + __bfloat162float(wa[2]) * x0.z + __bfloat162float(wa[3]) * x0.w
                 + __bfloat162float(wa[4]) * x1.x + __bfloat162float(wa[5]) * x1.y
                 + __bfloat162float(wa[6]) * x1.z + __bfloat162float(wa[7]) * x1.w;
        }
    } else {
        for (uint32_t i = lane; i < in_dim; i += 32u)
            acc += __bfloat162float(row[i]) * x[i];
    }
    for (uint32_t s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, s);
    if (lane == 0) {
        float v = acc;
        if (bias) v += bias[o];
        y[o] = v;
    }
}

// Multi-row twin of the narrow-K arm (slot 522). The arm above is batch-1 only,
// so a batched decode falls back to `pd_bf16_gemv_mr_f32` - measured 39 us a
// launch against this decomposition's 5-13. Here one warp still owns one output
// row, but it carries BT accumulators and multiplies each weight it loads
// against BT activation rows: the WEIGHT read (which is what a decode plane is
// bound by) is amortised BT ways, and `y` comes out row-major [batch, out_dim]
// like every other batched entry.
//
// grid = (ceil(out_dim/WPB), ceil(batch/BT)).
template <uint32_t WPB, uint32_t BT>
__global__ __launch_bounds__(WPB * 32u) void pd_bf16_gemv_nk_mr_f32_kernel(
    const __nv_bfloat16* __restrict__ w, const float* __restrict__ bias,
    const float* __restrict__ x, float* __restrict__ y,
    uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
    PD_PDL_ARM();
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t o = blockIdx.x * WPB + warp;
    if (o >= out_dim) return;
    const uint32_t t0 = blockIdx.y * BT;
    const __nv_bfloat16* __restrict__ row = w + (size_t)o * in_dim;
    float acc[BT];
    #pragma unroll
    for (uint32_t b = 0; b < BT; ++b) acc[b] = 0.0f;
    if ((in_dim & 7u) == 0u) {
        for (uint32_t base = lane * 8u; base < in_dim; base += 256u) {
            int4 wv = *reinterpret_cast<const int4*>(row + base);
            const __nv_bfloat16* wa = reinterpret_cast<const __nv_bfloat16*>(&wv);
            float wf[8];
            #pragma unroll
            for (uint32_t i = 0; i < 8u; ++i) wf[i] = __bfloat162float(wa[i]);
            #pragma unroll
            for (uint32_t b = 0; b < BT; ++b) {
                if (t0 + b >= batch) continue;
                const float* xr = x + (size_t)(t0 + b) * in_dim + base;
                float4 x0 = *reinterpret_cast<const float4*>(xr);
                float4 x1 = *reinterpret_cast<const float4*>(xr + 4);
                acc[b] += wf[0] * x0.x + wf[1] * x0.y + wf[2] * x0.z + wf[3] * x0.w
                        + wf[4] * x1.x + wf[5] * x1.y + wf[6] * x1.z + wf[7] * x1.w;
            }
        }
    } else {
        for (uint32_t i = lane; i < in_dim; i += 32u) {
            const float wf = __bfloat162float(row[i]);
            #pragma unroll
            for (uint32_t b = 0; b < BT; ++b)
                if (t0 + b < batch) acc[b] += wf * x[(size_t)(t0 + b) * in_dim + i];
        }
    }
    #pragma unroll
    for (uint32_t b = 0; b < BT; ++b) {
        float v = acc[b];
        for (uint32_t s = 16; s > 0; s >>= 1) v += __shfl_down_sync(0xffffffffu, v, s);
        if (lane == 0 && t0 + b < batch) {
            if (bias) v += bias[o];
            y[(size_t)(t0 + b) * out_dim + o] = v;
        }
    }
}

PD_EXPORT
int pd_bf16_gemv_nk_mr_f32(const void* w, const void* bias, const void* x,
                           void* y, uint32_t in_dim, uint32_t out_dim,
                           uint32_t batch, void* stream) {
    if (out_dim == 0 || in_dim == 0 || batch == 0) return 0;
    constexpr uint32_t BT = 8u;
    const uint32_t want = 4u * 148u;
#define PD_NKMR_GO(WPB)                                                        \
    do {                                                                       \
        dim3 grid((out_dim + (WPB) - 1u) / (WPB), (batch + BT - 1u) / BT);     \
        pd_pdl_go(pd_bf16_gemv_nk_mr_f32_kernel<WPB, BT>, grid, (WPB) * 32u,   \
                  0u, (cudaStream_t)stream, (const __nv_bfloat16*)w,           \
                  (const float*)bias, (const float*)x, (float*)y, in_dim,      \
                  out_dim, batch);                                             \
        return pd_launch_status();                                             \
    } while (0)
    if (out_dim >= want * 8u) PD_NKMR_GO(8u);
    if (out_dim >= want * 4u) PD_NKMR_GO(4u);
    if (out_dim >= want * 2u) PD_NKMR_GO(2u);
    PD_NKMR_GO(1u);
#undef PD_NKMR_GO
}

// ---------------------------------------------------------------------------
// up + hyper-connection MIX, fused (slot 562). The qwen4_exp HC up plane
// [in=lr, out=hc*hidden] is always followed by q4x_hc_mix, which reads the
// whole gate plane back and collapses it:
//     out[d] = (1/hc) * sum_s sigmoid(gate[s*hidden+d]) * xn[s*hidden+d]
// The `hc` gate values a given d needs are rows {d, hidden+d, ...} of the up
// plane -- so if one warp owns those hc rows, the mix happens in-register and
// the gate plane never round-trips. The per-row lane walk and shuffle
// reduction are the narrow-K arm's verbatim, so every gate value is bit
// identical; the mix then runs the hc_mix kernel's own op order.
//
// Decode-graph accounting (2026-08-31): hc_mix is 4.2 us on the CRITICAL
// branch of every layer, 96 launches a tick.
template <uint32_t WPB, uint32_t HC>
__global__ __launch_bounds__(WPB * 32u) void pd_bf16_gemv_up_hcmix_kernel(
    const __nv_bfloat16* __restrict__ w, const float* __restrict__ x,
    const float* __restrict__ xn, float* __restrict__ out,
    __nv_bfloat16* __restrict__ out16, uint32_t in_dim, uint32_t hidden) {
    PD_PDL_ARM();
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t d = blockIdx.x * WPB + warp;
    if (d >= hidden) return;
    float acc[HC];
    #pragma unroll
    for (uint32_t s = 0; s < HC; ++s) acc[s] = 0.0f;
    if ((in_dim & 7u) == 0u) {
        for (uint32_t base = lane * 8u; base < in_dim; base += 256u) {
            float4 x0 = *reinterpret_cast<const float4*>(x + base);
            float4 x1 = *reinterpret_cast<const float4*>(x + base + 4);
            int4 wv[HC];
            #pragma unroll
            for (uint32_t s = 0; s < HC; ++s)
                wv[s] = *reinterpret_cast<const int4*>(
                    w + ((size_t)s * hidden + d) * in_dim + base);
            #pragma unroll
            for (uint32_t s = 0; s < HC; ++s) {
                const __nv_bfloat16* wa = reinterpret_cast<const __nv_bfloat16*>(&wv[s]);
                acc[s] += __bfloat162float(wa[0]) * x0.x + __bfloat162float(wa[1]) * x0.y
                        + __bfloat162float(wa[2]) * x0.z + __bfloat162float(wa[3]) * x0.w
                        + __bfloat162float(wa[4]) * x1.x + __bfloat162float(wa[5]) * x1.y
                        + __bfloat162float(wa[6]) * x1.z + __bfloat162float(wa[7]) * x1.w;
            }
        }
    } else {
        for (uint32_t i = lane; i < in_dim; i += 32u) {
            const float xv = x[i];
            #pragma unroll
            for (uint32_t s = 0; s < HC; ++s)
                acc[s] += __bfloat162float(w[((size_t)s * hidden + d) * in_dim + i]) * xv;
        }
    }
    #pragma unroll
    for (uint32_t s = 0; s < HC; ++s)
        for (uint32_t r = 16; r > 0; r >>= 1)
            acc[s] += __shfl_down_sync(0xffffffffu, acc[s], r);
    if (lane == 0) {
        // hc_mix's own order: ascending s, sigmoid(gate) * xn, then /hc
        float m = 0.0f;
        #pragma unroll
        for (uint32_t s = 0; s < HC; ++s)
            m += pd_q4x_sig(acc[s]) * xn[(size_t)s * hidden + d];
        const float v = m / (float)HC;
        out[d] = v;
        if (out16) out16[d] = __float2bfloat16(v);
    }
}

// ---------------------------------------------------------------------------
// MULTI-ROW block-per-row gemv (slot 565). The batched decode band routes
// narrow-output planes into the tiled MMA kernels, which tile the OUTPUT: the
// qwen4_exp hc down plane [in=10240, out=320] lands on a 32x32 tile = ELEVEN
// CTAs on a 148-SM die, measured 16.06 us a launch for 6.55 MB (0.41 TB/s,
// 1.56 ms of a c8 tick). Here every output row gets its own block, as in the
// batch-1 arm, and each thread carries BT accumulators so the weight row is
// read once for the whole batch - the shape the batch-1 kernel already wins
// with, extended along the token axis.
//
// Per (row, token) the dot order is the batch-1 kernel's verbatim (same
// tid*16 stride, same shfl tree, same serial cross-warp sum), so a token's
// logits do not move when the batch width changes.
template <uint32_t BT>
__global__ void pd_bf16_gemv_mrow_f32_kernel(
    const __nv_bfloat16* __restrict__ w, const float* __restrict__ bias,
    const float* __restrict__ x, float* __restrict__ y,
    __nv_bfloat16* __restrict__ y16, uint32_t in_dim, uint32_t out_dim,
    uint32_t batch, uint32_t silu_rows, float inv,
    // optional SECOND segment: rows [split, out_dim) land in y2 as a
    // [batch, out_dim-split] plane (the hc down plane's inject tail).
    float* __restrict__ y2 = nullptr, uint32_t split = 0u) {
    PD_PDL_ARM();
    const uint32_t o = blockIdx.x;
    if (o >= out_dim) return;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    __shared__ float wsum[32][BT];
    const __nv_bfloat16* row = w + (size_t)o * in_dim;
    float acc[BT];
    #pragma unroll
    for (uint32_t b = 0; b < BT; ++b) acc[b] = 0.0f;
    if ((in_dim & 15u) == 0u) {
        for (uint32_t base = tid * 16u; base < in_dim; base += nth * 16u) {
            int4 w0 = *reinterpret_cast<const int4*>(row + base);
            int4 w1 = *reinterpret_cast<const int4*>(row + base + 8);
            const __nv_bfloat16* wa = reinterpret_cast<const __nv_bfloat16*>(&w0);
            const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&w1);
            float wf[16];
            #pragma unroll
            for (uint32_t i = 0; i < 8u; ++i) {
                wf[i] = __bfloat162float(wa[i]);
                wf[i + 8] = __bfloat162float(wb[i]);
            }
            #pragma unroll
            for (uint32_t b = 0; b < BT; ++b) {
                if (b >= batch) break;
                const float* xr = x + (size_t)b * in_dim + base;
                float4 x0 = *reinterpret_cast<const float4*>(xr);
                float4 x1 = *reinterpret_cast<const float4*>(xr + 4);
                float4 x2 = *reinterpret_cast<const float4*>(xr + 8);
                float4 x3 = *reinterpret_cast<const float4*>(xr + 12);
                acc[b] += wf[0] * x0.x + wf[1] * x0.y + wf[2] * x0.z + wf[3] * x0.w
                        + wf[4] * x1.x + wf[5] * x1.y + wf[6] * x1.z + wf[7] * x1.w
                        + wf[8] * x2.x + wf[9] * x2.y + wf[10] * x2.z + wf[11] * x2.w
                        + wf[12] * x3.x + wf[13] * x3.y + wf[14] * x3.z + wf[15] * x3.w;
            }
        }
    } else {
        for (uint32_t i = tid; i < in_dim; i += nth) {
            const float wv = __bfloat162float(row[i]);
            #pragma unroll
            for (uint32_t b = 0; b < BT; ++b)
                if (b < batch) acc[b] += wv * x[(size_t)b * in_dim + i];
        }
    }
    const uint32_t warp = tid >> 5, lane = tid & 31u;
    #pragma unroll
    for (uint32_t b = 0; b < BT; ++b) {
        float v = acc[b];
        for (uint32_t s = 16; s > 0; s >>= 1) v += __shfl_down_sync(0xffffffffu, v, s);
        if (lane == 0) wsum[warp][b] = v;
    }
    __syncthreads();
    if (tid < BT && tid < batch) {
        const uint32_t b = tid;
        float v = 0.0f;
        const uint32_t nwarps = (nth + 31u) >> 5;
        for (uint32_t i = 0; i < nwarps; ++i) v += wsum[i][b];
        if (bias) v += bias[o];
        if (o < silu_rows) {
            const float t = v * inv;
            v = t * (1.0f / (1.0f + __expf(-t)));
        }
        if (y2 != nullptr && o >= split) {
            y2[(size_t)b * (out_dim - split) + (o - split)] = v;
        } else {
            const uint32_t ow = (y2 != nullptr) ? split : out_dim;
            y[(size_t)b * ow + o] = v;
            if (y16) y16[(size_t)b * ow + o] = __float2bfloat16(v);
        }
    }
}

PD_EXPORT
int pd_bf16_gemv_mrow_f32(const void* w, const void* bias, const void* x,
                          void* y, void* y16, uint32_t in_dim, uint32_t out_dim,
                          uint32_t batch, uint32_t silu_rows, float inv,
                          void* y2, uint32_t split, void* stream) {
    if (out_dim == 0 || in_dim == 0 || batch == 0 || batch > 8) return -1;
    static int nth_env = -2;
    if (nth_env == -2) {
        const char* e = getenv("PADDOCK_Q38FN_MROW_THREADS");
        nth_env = (e && *e) ? atoi(e) : 0;
    }
    uint32_t nth = 256u;
    if (nth_env >= 32 && nth_env <= 1024) nth = (uint32_t)nth_env;
#define PD_MROW_GO(BT)                                                         \
    pd_pdl_go(pd_bf16_gemv_mrow_f32_kernel<BT>, out_dim, nth, 0u,              \
              (cudaStream_t)stream, (const __nv_bfloat16*)w,                   \
              (const float*)bias, (const float*)x, (float*)y,                  \
              (__nv_bfloat16*)y16, in_dim, out_dim, batch, silu_rows, inv,      \
              (float*)y2, split)
    if (batch > 4) { PD_MROW_GO(8u); }
    else if (batch > 2) { PD_MROW_GO(4u); }
    else { PD_MROW_GO(2u); }
#undef PD_MROW_GO
    return pd_launch_status();
}

PD_EXPORT
int pd_bf16_gemv_up_hcmix(const void* w, const void* x, const void* xn,
                          void* out, void* out16, uint32_t in_dim,
                          uint32_t hidden, uint32_t hc, void* stream) {
    if (in_dim == 0 || hidden == 0 || hc != 4u) return -1;
    constexpr uint32_t WPB = 8u;
    pd_pdl_go(pd_bf16_gemv_up_hcmix_kernel<WPB, 4u>,
              (hidden + WPB - 1u) / WPB, WPB * 32u, 0u, (cudaStream_t)stream,
              (const __nv_bfloat16*)w, (const float*)x, (const float*)xn,
              (float*)out, (__nv_bfloat16*)out16, in_dim, hidden);
    return pd_launch_status();
}

PD_EXPORT
int pd_bf16_gemv_nk_f32(const void* w, const void* bias, const void* x, void* y,
                        uint32_t in_dim, uint32_t out_dim, void* stream) {
    if (out_dim == 0 || in_dim == 0) return 0;
    // The grid is out_dim/WPB, so a fixed WPB starves narrow planes: at WPB=8
    // an out=2560 plane gets 320 blocks, ~2 per SM on this die, and cannot hide
    // memory latency. Pick WPB so the launch keeps roughly four blocks per SM
    // where the plane allows it, and fall back to one row per BLOCK for the
    // narrowest outputs.
    const uint32_t want = 4u * 148u;   // ~4 blocks/SM
#define PD_NK_GO(WPB)                                                              do {                                                                               uint32_t grid = (out_dim + (WPB) - 1u) / (WPB);                                pd_pdl_go(pd_bf16_gemv_nk_f32_kernel<WPB>, grid, (WPB) * 32u, 0u,                        (cudaStream_t)stream, (const __nv_bfloat16*)w,                                 (const float*)bias, (const float*)x, (float*)y, in_dim,                        out_dim);                                                            return pd_launch_status();                                                 } while (0)
    if (out_dim >= want * 8u) PD_NK_GO(8u);
    if (out_dim >= want * 4u) PD_NK_GO(4u);
    if (out_dim >= want * 2u) PD_NK_GO(2u);
    PD_NK_GO(1u);
#undef PD_NK_GO
}

// Decode-band multi-row twin (2 <= batch <= 8). The tile GEMM below collapses
// at decode widths: its grid is (out/64, batch/64), so batch<=8 leaves gridY=1
// and the whole launch is out_dim/64 blocks - 4 blocks for a 256-wide K/V
// plane, measured at 794us for a 0.5 MB weight read on paddleocr at c4
// (kernel-shape-not-weight-class). Same disease the f16 lane fixed
// with pd_f16_gemv_kernel; this is that shape for bf16 W x f32 X: one warp per
// output row, 8 rows per 256-thread block, the W row streamed once as 16B
// packs while the batch's activation rows ride L1 (all 8 warps of a block
// read the same X), f32 products into acc[NB], butterfly reduce, lane b
// stores output row b. No smem X stage: X here is f32 straight from the batch
// scratch, and at these in_dims (<= a few K) the block-shared L1 window
// carries it - the f16 kernel's swizzled stage exists because its X is f16
// activations it must also transpose.
template <uint32_t NB>
__global__ void __launch_bounds__(256) pd_bf16_gemv_mr_f32_kernel(
    const __nv_bfloat16* __restrict__ w, const float* __restrict__ bias,
    const float* __restrict__ x, float* __restrict__ y,
    uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
    const uint32_t o = blockIdx.x * 8u + (threadIdx.x >> 5);
    const uint32_t lane = threadIdx.x & 31u;
    // arm before the ragged-edge return: the grid over-covers out_dim, and
    // every thread must pass the griddepcontrol.wait
    PD_PDL_ARM();
    if (o >= out_dim) return;
    const __nv_bfloat16* row = w + (size_t)o * in_dim;
    float acc[NB];
#pragma unroll
    for (uint32_t b = 0; b < NB; ++b) acc[b] = 0.0f;
    // 16 weights (two int4) per lane per iter, the single-row kernel's pack
    // shape; the launcher gates in_dim % 16 == 0 (every elected plane is a
    // multiple of 128)
    for (uint32_t base = lane * 16u; base < in_dim; base += 32u * 16u) {
        int4 w0 = *reinterpret_cast<const int4*>(row + base);
        int4 w1 = *reinterpret_cast<const int4*>(row + base + 8);
        const __nv_bfloat16* wa = reinterpret_cast<const __nv_bfloat16*>(&w0);
        const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&w1);
        float wf[16];
#pragma unroll
        for (uint32_t i = 0; i < 8; ++i) {
            wf[i] = __bfloat162float(wa[i]);
            wf[8u + i] = __bfloat162float(wb[i]);
        }
#pragma unroll
        for (uint32_t b = 0; b < NB; ++b) {
            if (b >= batch) break;  // NB is rounded up; row b>=batch is not ours to read
            const float* xr = x + (size_t)b * in_dim + base;
            float4 x0 = *reinterpret_cast<const float4*>(xr);
            float4 x1 = *reinterpret_cast<const float4*>(xr + 4);
            float4 x2 = *reinterpret_cast<const float4*>(xr + 8);
            float4 x3 = *reinterpret_cast<const float4*>(xr + 12);
            acc[b] += wf[0] * x0.x + wf[1] * x0.y + wf[2] * x0.z + wf[3] * x0.w
                    + wf[4] * x1.x + wf[5] * x1.y + wf[6] * x1.z + wf[7] * x1.w
                    + wf[8] * x2.x + wf[9] * x2.y + wf[10] * x2.z + wf[11] * x2.w
                    + wf[12] * x3.x + wf[13] * x3.y + wf[14] * x3.z + wf[15] * x3.w;
        }
    }
#pragma unroll
    for (uint32_t b = 0; b < NB; ++b)
        for (uint32_t s = 16; s; s >>= 1) acc[b] += __shfl_xor_sync(~0u, acc[b], s);
    if (lane < batch) {
        float v = acc[lane];  // NB-bounded select, pd_f16_gemv_kernel's epilogue
        if (bias) v += bias[o];
        y[(size_t)lane * out_dim + o] = v;
    }
}

PD_EXPORT
int pd_bf16_gemv_mr_f32(const void* w, const void* bias, const void* x, void* y,
                        uint32_t in_dim, uint32_t out_dim, uint32_t batch,
                        void* stream) {
    if (out_dim == 0 || in_dim == 0 || batch == 0) return 0;
    // outside the decode band or a ragged in_dim: not this kernel's shape -
    // the caller keeps the tile GEMM
    if (batch < 2u || batch > 8u || (in_dim & 15u)) return -2;
    uint32_t grid = (out_dim + 7u) / 8u;
    cudaStream_t st = (cudaStream_t)stream;
    const __nv_bfloat16* wp = (const __nv_bfloat16*)w;
    const float* bp = (const float*)bias;
    const float* xp = (const float*)x;
    float* yp = (float*)y;
    if (batch <= 2u)
        pd_pdl_go(pd_bf16_gemv_mr_f32_kernel<2u>, grid, 256u, 0u, st, wp, bp, xp,
                  yp, in_dim, out_dim, batch);
    else if (batch <= 4u)
        pd_pdl_go(pd_bf16_gemv_mr_f32_kernel<4u>, grid, 256u, 0u, st, wp, bp, xp,
                  yp, in_dim, out_dim, batch);
    else
        pd_pdl_go(pd_bf16_gemv_mr_f32_kernel<8u>, grid, 256u, 0u, st, wp, bp, xp,
                  yp, in_dim, out_dim, batch);
    return pd_launch_status();
}

// Register-tiled f32 GEMM over a bf16 weight plane: y[r][o] = sum_k w[o][k] *
// x[r][k]. 64x64 output tile, 16-deep K chunk, 256 threads each holding a 4x4
// register tile. Both shared tiles are stored k-major so the inner product
// walks them without bank conflicts.
#define PD_BF16_BM 64u
#define PD_BF16_BN 64u
#define PD_BF16_BK 16u

__global__ void pd_bf16_gemm_f32_kernel(
    const __nv_bfloat16* __restrict__ w, const float* __restrict__ bias,
    const float* __restrict__ x, float* __restrict__ y,
    uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
    __shared__ float sw[PD_BF16_BK][PD_BF16_BM];
    __shared__ float sx[PD_BF16_BK][PD_BF16_BN];
    uint32_t o0 = blockIdx.x * PD_BF16_BM;
    uint32_t n0 = blockIdx.y * PD_BF16_BN;
    uint32_t tid = threadIdx.x;
    uint32_t ty = tid >> 4, tx = tid & 15u;   // 16x16 thread grid
    float acc[4][4] = {{0.f}};
    PD_PDL_ARM();
    for (uint32_t k0 = 0; k0 < in_dim; k0 += PD_BF16_BK) {
        // stage W[o0..o0+64)[k0..k0+16) transposed into sw[k][o]
        for (uint32_t idx = tid; idx < PD_BF16_BM * PD_BF16_BK; idx += blockDim.x) {
            uint32_t ol = idx >> 4, kk = idx & 15u;
            uint32_t o = o0 + ol, k = k0 + kk;
            sw[kk][ol] = (o < out_dim && k < in_dim)
                ? __bfloat162float(w[(size_t)o * in_dim + k]) : 0.0f;
        }
        // stage X[n0..n0+64)[k0..k0+16) transposed into sx[k][n]
        for (uint32_t idx = tid; idx < PD_BF16_BN * PD_BF16_BK; idx += blockDim.x) {
            uint32_t nl = idx >> 4, kk = idx & 15u;
            uint32_t n = n0 + nl, k = k0 + kk;
            sx[kk][nl] = (n < batch && k < in_dim)
                ? x[(size_t)n * in_dim + k] : 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (uint32_t kk = 0; kk < PD_BF16_BK; ++kk) {
            float a[4], b[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) a[i] = sw[kk][ty * 4 + i];
            #pragma unroll
            for (int j = 0; j < 4; ++j) b[j] = sx[kk][tx * 4 + j];
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint32_t o = o0 + ty * 4 + i;
        if (o >= out_dim) continue;
        float bv = bias ? bias[o] : 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            uint32_t n = n0 + tx * 4 + j;
            if (n < batch) y[(size_t)n * out_dim + o] = acc[i][j] + bv;
        }
    }
}

PD_EXPORT
int pd_bf16_gemm_f32(const void* w, const void* bias, const void* x, void* y,
                     uint32_t in_dim, uint32_t out_dim, uint32_t batch,
                     void* stream) {
    if (out_dim == 0 || in_dim == 0 || batch == 0) return 0;
    dim3 grid((out_dim + PD_BF16_BM - 1) / PD_BF16_BM,
              (batch + PD_BF16_BN - 1) / PD_BF16_BN);
    pd_pdl_go(pd_bf16_gemm_f32_kernel, grid, dim3(256u), 0u, (cudaStream_t)stream,
              (const __nv_bfloat16*)w, (const float*)bias, (const float*)x,
              (float*)y, in_dim, out_dim, batch);
    return pd_launch_status();
}

// ------------------------------------------------------------- mma prefill arm
// Tensor-core GEMM for the prefill band (batch > 8). The f32-FMA
// tile above runs the paddleocr prefill chunk at ~18 TFLOP/s - several x under
// the f32 roof and ~10x under what a bf16 tensor-core GEMM reaches, and its
// (out/64, rows/64) grid drops to 68 blocks on the 256-wide K/V planes. This
// arm is the elected pd_f16_gemm_mma_kernel ring (same tile census) with two
// deltas: the A plane loads bf16 weights (16-bit lanes - the cp.async staging
// and ldmatrix moves are byte-identical), and the B plane stages f32
// activations through a __float2bfloat16 cast (see the file header for why
// that class is sanctioned). Helpers are self-named pd_bf16m_* because this
// header is included before int8_mma/f16_dense in the pack blob, so their
// equivalents don't exist yet at this point in the translation unit.
//
// Arch note: the guarded-out body (< sm_80) compiles to a silent no-op - the
// pf5 lesson. The validated-arch floor is sm_86 (allowlist), so every arch
// this pack actually serves has the real body; do not ship this slot to a
// pre-Ampere pack without a runtime arch check.
#if defined(__CUDA_ARCH__)
#define PD_BF16MMA_OK (__CUDA_ARCH__ >= 800)
#else
#define PD_BF16MMA_OK 0
#endif

#if PD_BF16MMA_OK
__device__ __forceinline__ void pd_bf16m_ldm_x4(const void* p, uint32_t& r0,
                                                uint32_t& r1, uint32_t& r2,
                                                uint32_t& r3) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(a));
}
__device__ __forceinline__ void pd_bf16m_ldm_x2(const void* p, uint32_t& r0,
                                                uint32_t& r1) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                 : "=r"(r0), "=r"(r1)
                 : "r"(a));
}
// predicated 16B cp.async: ok==false issues a size-0 copy that zero-fills the
// shared destination without reading gmem (OOB source never dereferenced).
__device__ __forceinline__ void pd_bf16m_cpa16(void* smem, const void* gmem, bool ok) {
    const unsigned sm = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(sm),
                 "l"(gmem), "r"(ok ? 16u : 0u));
}
__device__ __forceinline__ void pd_bf16m_cpa_commit() {
    asm volatile("cp.async.commit_group;");
}
template <int N>
__device__ __forceinline__ void pd_bf16m_cpa_waitN() {
    asm volatile("cp.async.wait_group %0;" ::"n"(N));
}
// D = A*B + D, m16n8k16 bf16xbf16->f32 - same fragment maps as the f16 twin
// (A a0=(m=g,k=2t) a1=(m+8,2t) a2=(m,8+2t) a3=(m+8,8+2t); B b0=(n=g,k=2t)
// b1=(n,8+2t); D d0=(m=g,n=2t) d1=(m,2t+1) d2=(m+8,2t) d3=(m+8,2t+1)).
__device__ __forceinline__ void pd_bf16m_mma(float d[4], const uint32_t a[4],
                                             const uint32_t b[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
#endif

// Ring/compute/store structure is pd_f16_gemm_mma_kernel's, verbatim where the
// data allows: BM out-rows x BN batch-cols per CTA, K staged KT deep with ST
// cp.async buffers on the WEIGHT plane; the activation plane is staged with
// plain vector loads + cast (cp.async cannot convert), which the end-of-loop
// __syncthreads() orders exactly like the async stages. RG x CG register
// micro-tile per warp so each fragment feeds RG*CG mmas. Zero-padding past
// M/N/K keeps ragged tiles correct; the launcher gates K%16==0 so B's float4
// loads stay 16B-aligned (no ragged-K arm here - that class keeps the tile
// GEMM).
//
// QKV=true is the fused q|k|v decode arm (thin-k/v rung): W is the
// load-time-concatenated [q;k;v] plane (M = OQ + 2*OKV rows), and the store
// routes each out-row to its segment plane (Y=q, Yk, Yv - each [N, seg]
// row-major). Everything before the epilogue is untouched, so per out-row
// the result is BIT-identical to the plain arm on that segment. The point:
// a 256-row k/v plane on its own launch is latency-starved (~40 us floor at
// 1-3% of the DRAM roof, any kernel shape - bf16_thin_probe.cu); riding the
// fused grid streams it at the big plane's rate, and 3 launches become 1.
// KS=true is the decode-band K-split arm (bf16 K-split rung): the
// same tile with grid.z K-slabs, each block accumulating only its slab's
// KT-runs and writing an UNnormalized f32 partial plane into Y-as-partials
// ([z][col*M + row], flat over the fused M when QKV data is served - segment
// routing and bias move to the combine). Same numeric class as the f8row/
// int8 ks paths: f32 partial regroup only, per-slab sub-sums identical to
// the unsplit kernel's same k-range.
#define PD_BF16KS_ELEMS (8u * 8192u * 64u)
__device__ float pd_bf16ks_part[PD_BF16KS_ELEMS];

// Arrival counters for the FUSED K-split combine, one per (x, y) output tile.
// Zero-initialised by the loader, and the last arriving block resets its own
// slot to 0 before it leaves, so a captured decode graph replays with the plane
// back in its initial state (the same self-reset contract the grid-barrier
// kernels elsewhere in the pack use).
#define PD_BF16KS_TILES 65536u
__device__ unsigned int pd_bf16ks_ctr[PD_BF16KS_TILES];

// HCMIX: the hyper-connection MIX tail folded into the up-GEMM epilogue.
//
// Read out of the rival's own path (rival-src/hyperconnection.py +
// ops_hc_mix.py): at <=24 rows their whole mix is two kernels - silu folded
// into the down GEMM, and sigmoid * x * mean-over-hc folded into the up GEMM -
// enabled by a LOAD-TIME row permute (`permute_pad_up_weight`) that makes the
// hc mean contiguous. Ours was FIVE launches: down GEMM, ks_combine, scale_silu,
// up GEMM, hc_mix.
//
// We can do better than contiguous. At <64,32,8,ST,KT,RG=2,CG=1> the m16n8k16
// fragment already gives each thread rows {wr+g, wr+g+8, wr+g+16, wr+g+24} -
// stride 8 inside a 32-row group. So permuting the up plane as
//     permuted_row(d, s) = (d/8)*32 + s*8 + (d%8)     (verified bijective)
// puts all four hc branches of one output element in one THREAD'S REGISTERS.
// The mean is then free: no shared memory, no second kernel, no reduction to
// serialize - which is the distinction that killed the fused ks_combine
// (O(whole output), lost its parallelism) and that this one satisfies
// (O(tile), the data is already where it needs to be).
//
// Parameter overloading, because this arm needs two extra pointers and the
// plain arm's are dead here: Yk = the normed input `xn` [rows][hc*hidden],
// Yv = the mixed output [rows][hidden], OQ = hidden, OKV = hc. `Y` is unused -
// the gate plane never has to be materialised at all, which is the second
// saving after the launch.
template <uint32_t BM, uint32_t BN, uint32_t NWARP, uint32_t ST, uint32_t KT,
          uint32_t RG, uint32_t CG, bool QKV = false, bool KS = false,
          bool KSF = false, bool HCMIX = false>
__global__ void __launch_bounds__(NWARP * 32) pd_bf16_gemm_mma_kernel(
        const __nv_bfloat16* __restrict__ W, const float* __restrict__ X,
        const float* __restrict__ bias, float* __restrict__ Y,
        uint32_t K, uint32_t M, uint32_t N, float* __restrict__ Yk,
        float* __restrict__ Yv, uint32_t OQ, uint32_t OKV) {
#if PD_BF16MMA_OK
    // decode-graph PDL: joins the tick's launch cascade when launched with
    // the attribute (pd_pdl_go); a plain launch makes this a no-op
    PD_PDL_ARM();
    constexpr uint32_t NTH = NWARP * 32u;
    constexpr uint32_t WM = RG * 16u;      // warp tile rows
    constexpr uint32_t WN = CG * 8u;       // warp tile cols
    constexpr uint32_t WR = BM / WM;       // warp-rows in the CTA tile
    constexpr uint32_t WC = BN / WN;       // warp-cols in the CTA tile
    constexpr uint32_t NSUBK = KT / 16u;   // k16 sub-tiles per stage
    constexpr uint32_t KPAD = KT + 8u;     // padded shared K-stride (bf16s)
    constexpr uint32_t H8PR = KT / 8u;     // 8-element groups per staged row
    static_assert(WR * WC == NWARP, "warp grid");
    static_assert(WM * WR == BM && WN * WC == BN, "tile cover");
    static_assert(KT % 16u == 0u, "KT k16-multiple");
    static_assert(ST >= 2u && ST <= 4u, "stage count (ring only)");

    extern __shared__ __align__(16) __nv_bfloat16 pd_bf16m_dyn[];
    auto sh_a = reinterpret_cast<__nv_bfloat16(*)[BM * KPAD]>(pd_bf16m_dyn);
    auto sh_b = reinterpret_cast<__nv_bfloat16(*)[BN * KPAD]>(pd_bf16m_dyn + ST * BM * KPAD);

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, t = lane & 3u;
    const uint32_t wr = (warp % WR) * WM;   // warp row base within tile
    const uint32_t wc = (warp / WR) * WN;   // warp col base within tile
    const uint32_t row_base = blockIdx.x * BM;
    const uint32_t col_base = blockIdx.y * BN;
    const __nv_bfloat16 zero = __float2bfloat16(0.0f);

    // KS slab bounds: gridDim.z slabs measured in 512-ELEMENT blocks, not
    // KT-runs - 512 is a multiple of every config's KT (32/64/128), so the
    // slab boundaries are a pure function of K and identical across
    // configs. That is what keeps the fused-qkv export's per-segment
    // bit-identity: the qkv tiers run KT=64 where the plain tiers run
    // KT=128, and a KT-derived slab regroups the two sides differently
    // (caught by gpu_nemotron_kernels' to_bits gate at bt=9, 1-ulp drift).
    // k_eff replaces K in every stage guard, so a slab's ragged tail
    // zero-fills exactly the way the plain kernel's K tail does. The
    // launcher guarantees every slab is non-empty.
    uint32_t k_begin = 0u, k_eff = K;
    if constexpr (KS || KSF) {
        const uint32_t nb = (K + 511u) / 512u;
        const uint32_t per = (nb + gridDim.z - 1u) / gridDim.z;
        k_begin = blockIdx.z * per * 512u;
        const uint32_t ke = k_begin + per * 512u;
        k_eff = ke < K ? ke : K;
    }

    // stage kt's A (weights, async) and B (acts, cast) planes into buffer `buf`
    auto stage = [&](uint32_t k0, uint32_t buf) {
        #pragma unroll
        for (uint32_t i = tid; i < BM * H8PR; i += NTH) {
            const uint32_t row = i / H8PR, h8 = (i % H8PR) * 8u, gk = k0 + h8;
            const bool ok = (row_base + row) < M && gk + 8u <= k_eff;
            __nv_bfloat16* dst = &sh_a[buf][row * KPAD + h8];
            const __nv_bfloat16* src = W + (size_t)(row_base + row) * K + gk;
            pd_bf16m_cpa16(dst, src, ok);   // K%16==0: no ragged 8-group exists
        }
        #pragma unroll
        for (uint32_t i = tid; i < BN * H8PR; i += NTH) {
            const uint32_t col = i / H8PR, h8 = (i % H8PR) * 8u, gk = k0 + h8;
            const bool colok = (col_base + col) < N;
            __nv_bfloat16* dst = &sh_b[buf][col * KPAD + h8];
            if (colok && gk + 8u <= k_eff) {
                const float* src = X + (size_t)(col_base + col) * K + gk;
                const float4 v0 = *reinterpret_cast<const float4*>(src);
                const float4 v1 = *reinterpret_cast<const float4*>(src + 4);
                __nv_bfloat16 tmp[8] = {
                    __float2bfloat16(v0.x), __float2bfloat16(v0.y),
                    __float2bfloat16(v0.z), __float2bfloat16(v0.w),
                    __float2bfloat16(v1.x), __float2bfloat16(v1.y),
                    __float2bfloat16(v1.z), __float2bfloat16(v1.w)};
                *reinterpret_cast<int4*>(dst) = *reinterpret_cast<const int4*>(tmp);
            } else {
                #pragma unroll
                for (uint32_t e = 0; e < 8u; ++e)
                    dst[e] = (colok && gk + e < k_eff)
                        ? __float2bfloat16(X[(size_t)(col_base + col) * K + gk + e])
                        : zero;
            }
        }
    };

    const uint32_t l7 = lane & 7u;
    const uint32_t a_roff = ((lane & 8u) ? 8u : 0u) + l7;
    const uint32_t a_kof = (lane & 16u) ? 8u : 0u;
    const uint32_t b_kof = (lane & 8u) ? 8u : 0u;

    float acc[RG][CG][4] = {};
    auto compute = [&](uint32_t buf) {
        #pragma unroll
        for (uint32_t sk = 0; sk < NSUBK; ++sk) {
            const uint32_t ko = sk * 16u;
            uint32_t a[RG][4];
            #pragma unroll
            for (uint32_t rg = 0; rg < RG; ++rg)
                pd_bf16m_ldm_x4(&sh_a[buf][(wr + rg * 16u + a_roff) * KPAD + ko + a_kof],
                                a[rg][0], a[rg][1], a[rg][2], a[rg][3]);
            uint32_t b[CG][2];
            #pragma unroll
            for (uint32_t cg = 0; cg < CG; ++cg)
                pd_bf16m_ldm_x2(&sh_b[buf][(wc + cg * 8u + l7) * KPAD + ko + b_kof],
                                b[cg][0], b[cg][1]);
            #pragma unroll
            for (uint32_t rg = 0; rg < RG; ++rg)
                #pragma unroll
                for (uint32_t cg = 0; cg < CG; ++cg)
                    pd_bf16m_mma(acc[rg][cg], a[rg], b[cg]);
        }
    };

    // ST-deep ring: buffer p computes while up to ST-1 stages stream in. The
    // B-plane's plain smem stores ride the same barriers: they land in a
    // buffer no compute reads until after a later __syncthreads().
    #pragma unroll
    for (uint32_t s = 0; s < ST - 1u; ++s) {
        const uint32_t k0 = k_begin + s * KT;
        if (k0 < k_eff) stage(k0, s);
        pd_bf16m_cpa_commit();
    }
    uint32_t p = 0;
    for (uint32_t k0 = k_begin; k0 < k_eff; k0 += KT) {
        const uint32_t pre = k0 + (ST - 1u) * KT;
        if (pre < k_eff) stage(pre, (p + ST - 1u) % ST);
        pd_bf16m_cpa_commit();
        pd_bf16m_cpa_waitN<(int)ST - 1>();
        __syncthreads();
        compute(p);
        __syncthreads();
        p = (p + 1u) % ST;
    }

    // store: element (m=out row, n=batch col) -> Y[n*M + m]; bias is per
    // out-row. Each element is written by exactly one thread. The QKV arm
    // routes rows to their segment plane instead (see the header comment).
    auto put = [&](uint32_t r, uint32_t c, float v) {
        if constexpr (KSF) {
            // partials to the global plane; Y is the FINAL output, written
            // below by whichever block arrives last for this tile
            pd_bf16ks_part[(size_t)blockIdx.z * M * N + (size_t)c * M + r] = v;
        } else if constexpr (KS) {
            // partial plane z, flat [c][r] over the (possibly fused) M -
            // bias and QKV segment routing happen once, in the combine
            Y[(size_t)blockIdx.z * M * N + (size_t)c * M + r] = v;
        } else if constexpr (!QKV) {
            Y[(size_t)c * M + r] = v;
        } else if (r < OQ) {
            Y[(size_t)c * OQ + r] = v;
        } else if (r < OQ + OKV) {
            Yk[(size_t)c * OKV + (r - OQ)] = v;
        } else {
            Yv[(size_t)c * OKV + (r - OQ - OKV)] = v;
        }
    };
    if constexpr (HCMIX) {
        // acc[rg][cg][{0,2}] are rows wr+g+{rg*16, rg*16+8} for column c0, and
        // [{1,3}] the same rows for c1. Under the load-time permute those four
        // rows are hc branches s = 0..3 of output element d.
        static_assert(RG == 2 && CG == 1, "hcmix epilogue assumes RG=2, CG=1");
        const float* xn = Yk;
        float* mixed = Yv;
        const uint32_t hidden = OQ, hcn = OKV;
        const uint32_t d = ((row_base + wr) >> 5) * 8u + g;
        if (d < hidden) {
            const uint32_t c0 = col_base + wc + 2u * t, c1 = c0 + 1u;
            #pragma unroll
            for (uint32_t half = 0; half < 2u; ++half) {   // c0 then c1
                const uint32_t c = half ? c1 : c0;
                if (c >= N) continue;
                float a = 0.0f;
                #pragma unroll
                for (uint32_t s = 0; s < 4u; ++s) {
                    const float gv = acc[s >> 1][0][(s & 1u) * 2u + half];
                    const float xv =
                        xn[(size_t)c * hcn * hidden + (size_t)s * hidden + d];
                    // expf, not __expf: pd_q4x_hc_mix reaches the sigmoid
                    // through pd_q4x_sig, which uses expf. The fast-math twin
                    // costs a ULP (9048/81920 elements, worst 5.96e-08) and a
                    // greedy gate can see that.
                    a += (1.0f / (1.0f + expf(-gv))) * xv;
                }
                mixed[(size_t)c * hidden + d] = a / (float)hcn;
            }
        }
        return;
    } else {
        #pragma unroll
        for (uint32_t rg = 0; rg < RG; ++rg) {
            const uint32_t r0 = row_base + wr + rg * 16u + g;
            const uint32_t r8 = r0 + 8u;
            const float b0 = (bias && r0 < M) ? bias[r0] : 0.0f;
            const float b8 = (bias && r8 < M) ? bias[r8] : 0.0f;
            #pragma unroll
            for (uint32_t cg = 0; cg < CG; ++cg) {
                const uint32_t c0 = col_base + wc + cg * 8u + 2u * t;
                const uint32_t c1 = c0 + 1u;
                if (r0 < M) {
                    if (c0 < N) put(r0, c0, acc[rg][cg][0] + b0);
                    if (c1 < N) put(r0, c1, acc[rg][cg][1] + b0);
                }
                if (r8 < M) {
                    if (c0 < N) put(r8, c0, acc[rg][cg][2] + b8);
                    if (c1 < N) put(r8, c1, acc[rg][cg][3] + b8);
                }
            }
        }
    }

    // ---- fused K-split combine ------------------------------------------
    // The separate pd_bf16_ks_combine_kernel is 399 launches/step at c32 -
    // 8.31 per layer against the rival's 3.3 - and folding it here takes our
    // dense launch count from 18.64/layer to 10.33, below their 12.7. The
    // reduction work does not disappear; the launch does.
    //
    // Deterministic by construction: the last arriving block sums z ascending,
    // exactly the order the standalone combine used, so the result is bit-
    // identical rather than merely equivalent. Atomics order the ARRIVAL, never
    // the summation.
    if constexpr (KSF) {
        __threadfence();          // partials visible before we announce arrival
        __syncthreads();
        __shared__ unsigned int last;
        if (tid == 0) {
            const uint32_t tile = blockIdx.y * gridDim.x + blockIdx.x;
            last = atomicAdd(&pd_bf16ks_ctr[tile], 1u);
        }
        __syncthreads();
        if (last == gridDim.z - 1u) {
            if (tid == 0) {
                const uint32_t tile = blockIdx.y * gridDim.x + blockIdx.x;
                pd_bf16ks_ctr[tile] = 0u;   // self-reset: graph replay safe
            }
            __threadfence();      // acquire the other slabs' partials
            const uint32_t nz = gridDim.z;
            for (uint32_t e = tid; e < BM * BN; e += NTH) {
                const uint32_t lr = e % BM, lc = e / BM;
                const uint32_t r = row_base + lr, c = col_base + lc;
                if (r >= M || c >= N) continue;
                float a = 0.0f;
                for (uint32_t z = 0; z < nz; ++z)
                    a += pd_bf16ks_part[(size_t)z * M * N + (size_t)c * M + r];
                if (bias) a += bias[r];
                if constexpr (!QKV) {
                    Y[(size_t)c * M + r] = a;
                } else if (r < OQ) {
                    Y[(size_t)c * OQ + r] = a;
                } else if (r < OQ + OKV) {
                    Yk[(size_t)c * OKV + (r - OQ)] = a;
                } else {
                    Yv[(size_t)c * OKV + (r - OQ - OKV)] = a;
                }
            }
        }
    }
#else
    (void)W; (void)X; (void)bias; (void)Y; (void)K; (void)M; (void)N;
    (void)Yk; (void)Yv; (void)OQ; (void)OKV;
#endif
}

// ── TMA-staged weight arm (rung 5: the L1 ceiling) ────────────────────────
// ncu on the lm_head [248320 x 2560] at batch 32, grid 970, 256 thr:
//   DRAM 46.41%   L1/TEX 80.55% SATURATED   occupancy 23.60%   smem limit 2/SM
// The kernel above is not DRAM-bound. `cp.async` routes every weight byte
// through L1/TEX, L1 saturates, and DRAM idles at 46%. That is also why the
// BM sweep plateaued at 54% of roof: bigger tiles mean fewer CTAs mean fewer
// L1 requests, i.e. it was optimising inside an L1-bound regime.
//
// TMA (`cp.async.bulk.tensor`) bypasses L1 entirely. Measured on exactly this
// access pattern (bench/lmhead_mem.cu stream, 1.27 GB plane, 128x128B tiles):
//   plain LDG float4      343.4 us   3.70 TB/s   46% of roof
//   cp.async 16 B         276.9 us   4.59 TB/s   57%
//   TMA 2 stages          225.2 us   5.65 TB/s   71%   <-- 1.23x cp.async
//   TMA 3/4/6/8 stages    231/240/275/455 us     69/66/58/35%
// Depth is monotonically worse past 2, which names the currency: deeper rings
// eat shared memory, shared memory caps blocks/SM, and blocks/SM is what puts
// transactions in flight. So this arm goes the opposite way from the cp.async
// tile - SMALLER (BM=128, KT=64, ST=2 => 41 KB, ~5 blocks/SM) instead of
// bigger - and gets both L1 bypass AND occupancy. The rival's nvjet has the
// same two properties plus 2-CTA multicast, and reaches 82%.
//
// GEOMETRY is FIXED by the BOX. pd_tmap_2d encodes a 128-byte x 128-row box,
// so BM=128 and KT=64 bf16 are not tunable here; only BN/RG/CG/ST are.
//
// SWIZZLE. A TMA box cannot carry the KPAD padding the cp.async tile uses to
// keep ldmatrix conflict-free; it is 128B-swizzled instead. The permutation
// was DERIVED, not assumed (bench/lmhead_mem.cu swz stamps each 16-byte chunk
// with its own id, TMAs it in, and reads back where it landed):
//     chunk_smem = chunk_src XOR (row & 7)     on 16-byte chunks
// Every ldmatrix address here applies it. The B (activation) plane keeps its
// plain padded staging - it is 32 rows, not the traffic.
#if defined(PD_BS_HOST) || defined(PD_TC5_HOST)
template <uint32_t BN, uint32_t NWARP, uint32_t ST, uint32_t RG, uint32_t CG>
__global__ void __launch_bounds__(NWARP * 32) pd_bf16_gemm_tma_kernel(
        const __grid_constant__ PdTmap wmap, const float* __restrict__ X,
        const float* __restrict__ bias, float* __restrict__ Y,
        uint32_t K, uint32_t M, uint32_t N) {
#if PD_BF16MMA_OK && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 900)
    constexpr uint32_t BM = 128u, KT = 64u;      // the TMA box, not a knob
    constexpr uint32_t NTH = NWARP * 32u;
    constexpr uint32_t WM = RG * 16u, WN = CG * 8u;
    constexpr uint32_t WR = BM / WM, WC = BN / WN;
    constexpr uint32_t NSUBK = KT / 16u;
    constexpr uint32_t KPAD = KT + 8u;
    constexpr uint32_t H8PR = KT / 8u;
    constexpr uint32_t ABOX = 128u * 128u;       // bytes per staged W tile
    static_assert(WR * WC == NWARP, "warp grid");
    static_assert(ST >= 2u && ST <= 4u, "stage ring");

    extern __shared__ __align__(128) unsigned char pd_bf16t_dyn[];
    unsigned char* sh_a = pd_bf16t_dyn;                       // ST x 16 KB, swizzled
    auto sh_b = reinterpret_cast<__nv_bfloat16(*)[BN * KPAD]>(
            pd_bf16t_dyn + ST * ABOX);
    __shared__ alignas(8) unsigned long long mbar[ST];

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, t = lane & 3u;
    const uint32_t wr = (warp % WR) * WM, wc = (warp / WR) * WN;
    const uint32_t row_base = blockIdx.x * BM, col_base = blockIdx.y * BN;
    const __nv_bfloat16 zero = __float2bfloat16(0.0f);

    if (tid < ST) {
        const uint32_t m = (uint32_t)__cvta_generic_to_shared(&mbar[tid]);
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(m));
    }
    __syncthreads();

    // W: one thread issues the box. X: the plain padded stage, unchanged.
    auto stage_w = [&](uint32_t k0, uint32_t buf) {
        if (tid == 0u) {
            const uint32_t m = (uint32_t)__cvta_generic_to_shared(&mbar[buf]);
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                         ::"r"(m), "r"(ABOX));
            const uint32_t d = (uint32_t)__cvta_generic_to_shared(sh_a + buf * ABOX);
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                " [%0], [%1, {%2, %3}], [%4];" ::"r"(d), "l"(&wmap),
                "r"((int)(k0 * 2u)), "r"((int)row_base), "r"(m) : "memory");
        }
    };
    auto stage_x = [&](uint32_t k0, uint32_t buf) {
        for (uint32_t i = tid; i < BN * H8PR; i += NTH) {
            const uint32_t col = i / H8PR, h8 = (i % H8PR) * 8u, gk = k0 + h8;
            const bool colok = (col_base + col) < N;
            __nv_bfloat16* dst = &sh_b[buf][col * KPAD + h8];
            if (colok && gk + 8u <= K) {
                const float* src = X + (size_t)(col_base + col) * K + gk;
                const float4 v0 = *reinterpret_cast<const float4*>(src);
                const float4 v1 = *reinterpret_cast<const float4*>(src + 4);
                __nv_bfloat16 tmp[8] = {
                    __float2bfloat16(v0.x), __float2bfloat16(v0.y),
                    __float2bfloat16(v0.z), __float2bfloat16(v0.w),
                    __float2bfloat16(v1.x), __float2bfloat16(v1.y),
                    __float2bfloat16(v1.z), __float2bfloat16(v1.w)};
                *reinterpret_cast<int4*>(dst) = *reinterpret_cast<const int4*>(tmp);
            } else {
                for (uint32_t e = 0; e < 8u; ++e)
                    dst[e] = (colok && gk + e < K)
                        ? __float2bfloat16(X[(size_t)(col_base + col) * K + gk + e])
                        : zero;
            }
        }
    };

    const uint32_t l7 = lane & 7u;
    const uint32_t a_roff = ((lane & 8u) ? 8u : 0u) + l7;
    const uint32_t a_kof = (lane & 16u) ? 8u : 0u;
    const uint32_t b_kof = (lane & 8u) ? 8u : 0u;

    float acc[RG][CG][4] = {};
    auto compute = [&](uint32_t buf) {
        unsigned char* base = sh_a + buf * ABOX;
        #pragma unroll
        for (uint32_t sk = 0; sk < NSUBK; ++sk) {
            const uint32_t ko = sk * 16u;
            uint32_t a[RG][4];
            #pragma unroll
            for (uint32_t rg = 0; rg < RG; ++rg) {
                const uint32_t row = wr + rg * 16u + a_roff;
                const uint32_t kk = ko + a_kof;                 // multiple of 8
                // chunk_smem = chunk_src XOR (row & 7), 16-byte chunks
                const uint32_t off = row * 128u + (((kk >> 3) ^ (row & 7u)) << 4);
                pd_bf16m_ldm_x4(reinterpret_cast<const __nv_bfloat16*>(base + off),
                                a[rg][0], a[rg][1], a[rg][2], a[rg][3]);
            }
            uint32_t b[CG][2];
            #pragma unroll
            for (uint32_t cg = 0; cg < CG; ++cg)
                pd_bf16m_ldm_x2(&sh_b[buf][(wc + cg * 8u + l7) * KPAD + ko + b_kof],
                                b[cg][0], b[cg][1]);
            #pragma unroll
            for (uint32_t rg = 0; rg < RG; ++rg)
                #pragma unroll
                for (uint32_t cg = 0; cg < CG; ++cg)
                    pd_bf16m_mma(acc[rg][cg], a[rg], b[cg]);
        }
    };

    #pragma unroll
    for (uint32_t s = 0; s < ST - 1u; ++s) {
        const uint32_t k0 = s * KT;
        if (k0 < K) { stage_w(k0, s); stage_x(k0, s); }
    }
    uint32_t p = 0, step = 0;
    for (uint32_t k0 = 0; k0 < K; k0 += KT, ++step) {
        const uint32_t pre = k0 + (ST - 1u) * KT;
        if (pre < K) { stage_w(pre, (p + ST - 1u) % ST); stage_x(pre, (p + ST - 1u) % ST); }
        const uint32_t m = (uint32_t)__cvta_generic_to_shared(&mbar[p]);
        const uint32_t ph = (step / ST) & 1u;
        asm volatile("{ .reg .pred P; W: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;"
                     " @P bra D; bra W; D: }" ::"r"(m), "r"(ph));
        __syncthreads();
        compute(p);
        __syncthreads();
        p = (p + 1u) % ST;
    }

    #pragma unroll
    for (uint32_t rg = 0; rg < RG; ++rg) {
        const uint32_t r0 = row_base + wr + rg * 16u + g, r8 = r0 + 8u;
        const float b0 = (bias && r0 < M) ? bias[r0] : 0.0f;
        const float b8 = (bias && r8 < M) ? bias[r8] : 0.0f;
        #pragma unroll
        for (uint32_t cg = 0; cg < CG; ++cg) {
            const uint32_t c0 = col_base + wc + cg * 8u + 2u * t, c1 = c0 + 1u;
            if (r0 < M) {
                if (c0 < N) Y[(size_t)c0 * M + r0] = acc[rg][cg][0] + b0;
                if (c1 < N) Y[(size_t)c1 * M + r0] = acc[rg][cg][1] + b0;
            }
            if (r8 < M) {
                if (c0 < N) Y[(size_t)c0 * M + r8] = acc[rg][cg][2] + b8;
                if (c1 < N) Y[(size_t)c1 * M + r8] = acc[rg][cg][3] + b8;
            }
        }
    }
#else
    (void)wmap; (void)X; (void)bias; (void)Y; (void)K; (void)M; (void)N;
#endif
}
#endif  // PD_BS_HOST || PD_TC5_HOST

// ── decode-band K-split (bf16 K-split rung) ──────────────────────
// At c32 the attn projections launch 42 (wo, BM=64) and 72 (fused qkv)
// blocks on a 188-SM die and run at 24-37% of the DRAM roof, while cuBLASLt
// split-K covers the same shapes at ~69%. Careful with the batch-scaling
// "grid-fill ladder" reading that argues K-split off: that ladder scales
// BATCH, and its GB/s-wt metric divides SINGLE-plane bytes by time, so a
// flat curve means time held constant while physical traffic doubled - CTA
// starvation, exactly what a K-split cures at constant bytes.
//
// Scratch: the mma path runs only inside captured decode ticks (nemotron
// step_replay captures the first tick per r), so f8row's lazy
// grow-outside-capture cudaMalloc can never allocate here. A fixed static
// __device__ plane (16 MB, pd_smp_scr precedent: single-engine-stream
// serving contract) is capture-safe by construction; the fit gate below is
// structural - K-split only matters when the 2D grid is small, and a small
// grid bounds nz*M*N.
// (the K-split partials plane and its arrival counters are declared above
// the mma kernel, which reads both directly in its fused-combine epilogue)

// fixed-order partial combine. Plain arm: y[c*M+r] = sum_z part + bias[r].
// QKV arm (yk != nullptr): the fused row is routed to its segment plane,
// exactly pd_bf16_gemm_mma_kernel's put() - bias is qkv-null upstream.
__global__ void pd_bf16_ks_combine_kernel(
        const float* __restrict__ part, const float* __restrict__ bias,
        float* __restrict__ y, float* __restrict__ yk, float* __restrict__ yv,
        uint32_t n, uint32_t nz, uint32_t m, uint32_t oq, uint32_t okv) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = 0.0f;
    for (uint32_t z = 0; z < nz; ++z) acc += part[(size_t)z * n + i];
    const uint32_t r = i % m, c = i / m;
    if (bias) acc += bias[r];
    if (!yk) {
        y[i] = acc;
    } else if (r < oq) {
        y[(size_t)c * oq + r] = acc;
    } else if (r < oq + okv) {
        yk[(size_t)c * okv + (r - oq)] = acc;
    } else {
        yv[(size_t)c * okv + (r - oq - okv)] = acc;
    }
}

// nz policy. The grid-fill test (blocks2d < 2 CTAs/SM) only decides whether
// to split; the split COUNT is a pure function of K: slabs of 512 elements
// (a multiple of every config's KT), capped at 8 (the scratch's plane
// budget). That purity is what keeps pd_bf16_qkv_gemm_mma's documented
// per-segment bit-identity intact - the fused plane and each segment see
// the same K, hence the same slab boundaries, hence identical regroup
// (gpu_nemotron_kernels.rs asserts to_bits equality; the qkv tiers run a
// different KT than the plain tiers, which is why the unit is elements,
// not KT-runs). The fit clamp can, in principle, diverge nz for M*batch
// beyond ~9k*64 - no shipped bit-compared pairing lives there; every
// nemotron shape fits. 0 = stay unsplit. Kill: PADDOCK_NO_BF16_KSPLIT.
static uint32_t pd_bf16ks_nz(uint32_t blocks2d, uint32_t in_dim,
                             uint32_t out_m, uint32_t batch) {
    static const bool off = pd_env("PADDOCK_NO_BF16_KSPLIT") != nullptr;
    if (off) return 0u;
    static int nsm = 0;
    if (nsm == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
        if (nsm <= 0) nsm = 128;
    }
    // The "already fills the die" early-out. The multiplier is 2 by default
    // and A/B-able, because at the c8 decode width it is what keeps the two
    // biggest streaming planes on the UNSPLIT arm by a hair -- and the unsplit
    // arm is measurably the worse streamer at that width. From the c8 serve
    // nsys (median tick, same kernel, same in_dim=2560, same batch):
    //   GDN qkv [2560->10240] BM=32 -> blocks2d 320 >= 296, nz 0: 1.54 TB/s
    //   GDN z   [2560-> 6144] BM=32 -> blocks2d 192 <  296, nz 5: 2.87 TB/s
    // 320 CTAs on 148 SMs is 2.16 per SM, so the arm pays a half-empty tail
    // wave; nz multiplies the CTA count and smooths it. The c32 sweep in the
    // cap comment below already measured this plane with nz=5 (blocks2d 160 at
    // BM=64) at 2.13 TB/s, i.e. the split arm is not new to this shape -- only
    // the c8 tile makes blocks2d cross the threshold.
    static const uint32_t fill = [] {
        const char* v = pd_env("PADDOCK_BF16KS_FILL");
        const uint32_t n = v ? (uint32_t)atoi(v) : 2u;
        return n >= 1u ? n : 2u;
    }();
    if (blocks2d >= (uint32_t)nsm * fill) return 0u;
    // Slab granularity and the split cap. Both were fixed constants (512, 8),
    // and the cap is what starves the hc chain: at in_dim=10240 nb is 20, the
    // cap takes it to 8, and the non-empty-slab rounding lands on 7 - so the
    // hc INJECT (out_dim 4 -> gridX 1) runs SEVEN blocks on a 148-SM card.
    // Env-swept before any default moves; see bench/lmhead_mem.cu.
    static const uint32_t slab = [] {
        const char* v = pd_env("PADDOCK_BF16KS_SLAB");
        const uint32_t n = v ? (uint32_t)atoi(v) : 512u;
        return n >= 128u ? n : 512u;
    }();
    // The cap used to be a flat 8, and that is what starved the hc chain: at
    // in_dim=10240 nb is 20, the flat cap took it to 8, and the non-empty-slab
    // rounding landed on 7 -- so hc INJECT (gridX 1) ran SEVEN blocks on 148
    // SMs. Split only as far as FILLS the DIE, which is the same criterion the
    // early-out above already uses (blocks2d >= 2*nsm needs no split at all),
    // so the two ends of the rule now agree instead of one being a constant.
    //   hc down   [10240->320] blocks2d 5 -> want 60, nb 20 -> nz 20
    //   hc INJECT [10240->  4] blocks2d 1 -> want 296, nb 20 -> nz 20
    // Measured (bench/lmhead_mem.cu hc, batch 32, 200 iters):
    //   down   22.40 -> 12.31 us (1.82x)   INJECT 20.51 -> 12.30 us (1.67x)
    // 12.3 us is a floor: nz beyond nb cannot help, and a FINER slab (256 or
    // 128, nb 40/80) buys nothing and costs down 12.31 -> 14.37, so the slab
    // stays 512 and only the cap moves.
    // Not bit-neutral -- nz changes the f32 partial regroup grouping. Same
    // sanctioned reorder class the K-split arm already ships (per-slab sub-sums
    // are still exact over their own k-range), gated on the forward/greedy
    // batteries, not asserted bit-equal.
    static const uint32_t capenv = [] {
        const char* v = pd_env("PADDOCK_BF16KS_MAX");
        const uint32_t n = v ? (uint32_t)atoi(v) : 0u;
        return n >= 2u ? n : 0u;
    }();
    const uint32_t nb = (in_dim + slab - 1u) / slab;
    // RAISE ONLY. A pure fill rule (cap = want) also LOWERS nz on shapes that
    // are already well filled, and that regresses them -- measured, which is
    // why the sweep checks every decode cell and not just the two it targets:
    //   GDN qkv [2560->10240] blocks2d 160, nz 5 -> 2:  24.63 -> 28.82 us
    //   GDN z   [2560-> 6144] blocks2d  96, nz 5 -> 4:  18.46 -> 20.50 us
    // Keeping 8 as a floor preserves every shape the old constant served and
    // moves only the starved ones.
    const uint32_t want = ((uint32_t)nsm * 2u + blocks2d - 1u) / blocks2d;
    const uint32_t cap = capenv ? capenv : (want < 8u ? 8u : want);
    uint32_t nz = nb > cap ? cap : nb;
    while (nz >= 2u && (size_t)nz * out_m * batch > (size_t)PD_BF16KS_ELEMS)
        --nz;
    if (nz < 2u) return 0u;
    const uint32_t per = (nb + nz - 1u) / nz;
    return (nb + per - 1u) / per;   // every slab non-empty
}

static float* pd_bf16ks_scr() {
    static float* p = [] {
        void* d = nullptr;
        if (cudaGetSymbolAddress(&d, pd_bf16ks_part) != cudaSuccess) d = nullptr;
        return (float*)d;
    }();
    return p;
}

#if defined(PD_BS_HOST) || defined(PD_TC5_HOST)
// Tensormaps are encoded once per weight plane, not per call: cuTensorMapEncodeTiled
// is a driver round-trip and this runs inside captured decode ticks. Keyed by
// (base, inner, rows) like the f8 lane's caches. Small fixed table; a miss past
// the end declines to the cp.async arm rather than evicting.
struct PdBf16Tmap { const void* base; uint64_t inner, rows; CUtensorMap m; };
static CUtensorMap* pd_bf16_tmap(const void* base, uint64_t inner, uint64_t rows) {
    static PdBf16Tmap tab[64];
    static int n = 0;
    for (int i = 0; i < n; ++i)
        if (tab[i].base == base && tab[i].inner == inner && tab[i].rows == rows)
            return &tab[i].m;
    if (n >= 64) return nullptr;
    if (!pd_tmap_2d(&tab[n].m, base, inner, rows)) return nullptr;
    tab[n].base = base; tab[n].inner = inner; tab[n].rows = rows;
    return &tab[n++].m;
}

template <uint32_t BN, uint32_t NW, uint32_t ST, uint32_t RG, uint32_t CG>
static int pd_bf16_tma_cfg(const __nv_bfloat16* w, const float* x,
                           const float* bias, float* y, uint32_t in_dim,
                           uint32_t out_dim, uint32_t batch, cudaStream_t st) {
    constexpr uint32_t BM = 128u, KT = 64u, KPAD = KT + 8u;
    constexpr uint32_t smem = ST * 128u * 128u + ST * BN * KPAD * 2u;
    // the box is 128 B wide, so K must cover it and stay 8-aligned
    if (in_dim < 64u || (in_dim & 7u)) return -1;
    CUtensorMap* map = pd_bf16_tmap(w, (uint64_t)in_dim * 2u, out_dim);
    if (!map) return -1;
    static bool set = false;
    if (!set) {
        if (cudaFuncSetAttribute(pd_bf16_gemm_tma_kernel<BN, NW, ST, RG, CG>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 smem) != cudaSuccess)
            return -1;
        set = true;
    }
    dim3 grid((out_dim + BM - 1u) / BM, (batch + BN - 1u) / BN);
    pd_bf16_gemm_tma_kernel<BN, NW, ST, RG, CG><<<grid, NW * 32u, smem, st>>>(
            *map, x, bias, y, in_dim, out_dim, batch);
    return (int)cudaGetLastError();
}
#endif

template <uint32_t BM, uint32_t BN, uint32_t NW, uint32_t ST, uint32_t KT,
          uint32_t RG, uint32_t CG>
static int pd_bf16_mma_cfg(const __nv_bfloat16* w, const float* x,
                           const float* bias, float* y, uint32_t in_dim,
                           uint32_t out_dim, uint32_t batch, cudaStream_t st) {
    constexpr uint32_t KPAD = KT + 8u;
    constexpr uint32_t smem = ST * (BM * KPAD + BN * KPAD) * 2u;
    static bool set = false;
    if (!set) {
        cudaFuncSetAttribute(pd_bf16_gemm_mma_kernel<BM, BN, NW, ST, KT, RG, CG>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        cudaFuncSetAttribute(
            pd_bf16_gemm_mma_kernel<BM, BN, NW, ST, KT, RG, CG, false, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        // the FUSED-combine instantiation is a different function: missing it
        // here made the launch fail with cudaErrorInvalidValue, and the
        // launcher's `return cudaGetLastError()` cleared the error, so the
        // bench timed a kernel that never ran (0.98 us and an all-zero output).
        cudaFuncSetAttribute(
            pd_bf16_gemm_mma_kernel<BM, BN, NW, ST, KT, RG, CG, false, false, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        set = true;
    }
    dim3 grid((out_dim + BM - 1u) / BM, (batch + BN - 1u) / BN);
    // K-split arm first (see the block comment above pd_bf16ks_part): same
    // tile, grid.z K-slabs into the static partials plane, fixed-order
    // combine applies the bias. Self-gates on grid fill / slab depth / fit.
    const uint32_t nz = pd_bf16ks_nz(grid.x * grid.y, in_dim, out_dim, batch);
    float* part = nz >= 2u ? pd_bf16ks_scr() : nullptr;
    if (part) {
        dim3 gz(grid.x, grid.y, nz);
        // FUSED ARM - REFUTED, default off, kept only so the claim stays
        // A/B-able (PADDOCK_BF16_KSFUSE=1).
        //
        // The argument for it was launch count: the standalone combine is 399
        // launches/step at c32, 8.31 per layer against the rival's 3.3, and
        // folding it takes our dense launches from 18.64/layer to 10.33 -
        // below their 12.7. Bit-identical output, verified (sum|y| matches to
        // the digit on all eight decode shapes).
        //
        // It is slower on every shape that splits: weighted 6.35 -> 8.39
        // ms/step, worst hc down 12.31 -> 24.59 us. The reason retires the
        // premise: the standalone combine reduces across the whole device
        // (grid = n/256 blocks), while the fused epilogue has only
        // gridX*gridY blocks left alive to do it - SIX for hc down, 4% of the
        // GPU, with every other block already exited.
        //
        // LAW: a launch is not free, but the work inside it does not become
        // free by moving to fewer blocks. Fusing only pays when the folded
        // work is O(tile), not when it is O(whole output) like a reduction.
        static const bool fuse = pd_env("PADDOCK_BF16_KSFUSE") != nullptr;
        if (fuse && (size_t)grid.x * grid.y <= PD_BF16KS_TILES) {
            pd_bf16_gemm_mma_kernel<BM, BN, NW, ST, KT, RG, CG, false, false, true>
                <<<gz, NW * 32u, smem, st>>>(w, x, bias, y, in_dim, out_dim,
                                             batch, nullptr, nullptr, 0u, 0u);
            return (int)cudaGetLastError();
        }
        pd_bf16_gemm_mma_kernel<BM, BN, NW, ST, KT, RG, CG, false, true>
            <<<gz, NW * 32u, smem, st>>>(w, x, nullptr, part, in_dim, out_dim,
                                         batch, nullptr, nullptr, 0u, 0u);
        const uint32_t n = out_dim * batch;
        pd_bf16_ks_combine_kernel<<<(n + 255u) / 256u, 256, 0, st>>>(
            part, bias, y, nullptr, nullptr, n, nz, out_dim, 0u, 0u);
        return (int)cudaGetLastError();
    }
    pd_bf16_gemm_mma_kernel<BM, BN, NW, ST, KT, RG, CG><<<grid, NW * 32u, smem, st>>>(
            w, x, bias, y, in_dim, out_dim, batch, nullptr, nullptr, 0u, 0u);
    return (int)cudaGetLastError();
}

// The QKV=true twin: grid covers the fused row count, launch rides the PDL
// cascade (this arm only runs inside decode ticks, where the launch train
// is armed; pd_pdl_go degrades to a plain launch elsewhere).
template <uint32_t BM, uint32_t BN, uint32_t NW, uint32_t ST, uint32_t KT,
          uint32_t RG, uint32_t CG>
static int pd_bf16_qkv_cfg(const __nv_bfloat16* w, const float* x, float* yq,
                           float* yk, float* yv, uint32_t in_dim, uint32_t oq,
                           uint32_t okv, uint32_t batch, cudaStream_t st) {
    constexpr uint32_t KPAD = KT + 8u;
    constexpr uint32_t smem = ST * (BM * KPAD + BN * KPAD) * 2u;
    static bool set = false;
    if (!set) {
        cudaFuncSetAttribute(
            pd_bf16_gemm_mma_kernel<BM, BN, NW, ST, KT, RG, CG, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        cudaFuncSetAttribute(
            pd_bf16_gemm_mma_kernel<BM, BN, NW, ST, KT, RG, CG, true, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        cudaFuncSetAttribute(
            pd_bf16_gemm_mma_kernel<BM, BN, NW, ST, KT, RG, CG, true, false, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        set = true;
    }
    const uint32_t m = oq + 2u * okv;
    dim3 grid((m + BM - 1u) / BM, (batch + BN - 1u) / BN);
    // K-split arm first (same shape as the plain cfg): partials are flat
    // over the fused m; the combine does the q|k|v segment routing that
    // put() would have done. The partial kernel keeps the PDL arm - it is
    // the tick-cascade member; the combine is stream-ordered behind it.
    const uint32_t nz = pd_bf16ks_nz(grid.x * grid.y, in_dim, m, batch);
    float* part = nz >= 2u ? pd_bf16ks_scr() : nullptr;
    if (part) {
        dim3 gz(grid.x, grid.y, nz);
        pd_pdl_go(pd_bf16_gemm_mma_kernel<BM, BN, NW, ST, KT, RG, CG, true, true>,
                  gz, dim3(NW * 32u), smem, st, w, x, (const float*)nullptr,
                  part, in_dim, m, batch, yk, yv, oq, okv);
        const uint32_t n = m * batch;
        pd_bf16_ks_combine_kernel<<<(n + 255u) / 256u, 256, 0, st>>>(
            part, nullptr, yq, yk, yv, n, nz, m, oq, okv);
        return (int)cudaGetLastError();
    }
    pd_pdl_go(pd_bf16_gemm_mma_kernel<BM, BN, NW, ST, KT, RG, CG, true>, grid,
              dim3(NW * 32u), smem, st, w, x, (const float*)nullptr, yq,
              in_dim, m, batch, yk, yv, oq, okv);
    return (int)cudaGetLastError();
}

// TWO-segment twin of pd_bf16_qkv_cfg. Identical kernel and identical row
// routing; the only difference is the fused row count.
//
// This exists because reusing the q|k|v launcher for a 2-segment plane is a
// silent out-of-bounds read, and it cost a gate failure to learn: that
// launcher computes `m = oq + 2*okv` because it always fuses q with two kv
// segments. Handed a plane of `oq + ob` rows it generates `ob` rows PAST the
// end, reads weights that are not there, and routes them to the third
// segment's pointer. The kernel's own `put()` is fine - it is the launcher's
// row count that is wrong - which is exactly the kind of thing that reads as
// correct until a greedy gate diverges.
//
// Here m = oq + ob, so the third branch of put() is genuinely unreachable and
// `yv` is passed as `yk` only to keep the pointer valid.
template <uint32_t BM, uint32_t BN, uint32_t NW, uint32_t ST, uint32_t KT,
          uint32_t RG, uint32_t CG>
static int pd_bf16_seg2_cfg(const __nv_bfloat16* w, const float* x, float* ya,
                            float* yb, uint32_t in_dim, uint32_t oq,
                            uint32_t ob, uint32_t batch, cudaStream_t st) {
    constexpr uint32_t KPAD = KT + 8u;
    constexpr uint32_t smem = ST * (BM * KPAD + BN * KPAD) * 2u;
    static bool set = false;
    if (!set) {
        cudaFuncSetAttribute(
            pd_bf16_gemm_mma_kernel<BM, BN, NW, ST, KT, RG, CG, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        cudaFuncSetAttribute(
            pd_bf16_gemm_mma_kernel<BM, BN, NW, ST, KT, RG, CG, true, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        cudaFuncSetAttribute(
            pd_bf16_gemm_mma_kernel<BM, BN, NW, ST, KT, RG, CG, true, false, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        set = true;
    }
    const uint32_t m = oq + ob;
    dim3 grid((m + BM - 1u) / BM, (batch + BN - 1u) / BN);
    const uint32_t nz = pd_bf16ks_nz(grid.x * grid.y, in_dim, m, batch);
    float* part = nz >= 2u ? pd_bf16ks_scr() : nullptr;
    if (part) {
        dim3 gz(grid.x, grid.y, nz);
        // same refutation as the plain arm above; default off
        static const bool fuse2 = pd_env("PADDOCK_BF16_KSFUSE") != nullptr;
        if (fuse2 && (size_t)grid.x * grid.y <= PD_BF16KS_TILES) {
            pd_pdl_go(pd_bf16_gemm_mma_kernel<BM, BN, NW, ST, KT, RG, CG, true, false, true>,
                      gz, dim3(NW * 32u), smem, st, w, x, (const float*)nullptr,
                      ya, in_dim, m, batch, yb, yb, oq, ob);
            return (int)cudaGetLastError();
        }
        pd_pdl_go(pd_bf16_gemm_mma_kernel<BM, BN, NW, ST, KT, RG, CG, true, true>,
                  gz, dim3(NW * 32u), smem, st, w, x, (const float*)nullptr,
                  part, in_dim, m, batch, yb, yb, oq, ob);
        const uint32_t n = m * batch;
        pd_bf16_ks_combine_kernel<<<(n + 255u) / 256u, 256, 0, st>>>(
            part, nullptr, ya, yb, yb, n, nz, m, oq, ob);
        return (int)cudaGetLastError();
    }
    pd_pdl_go(pd_bf16_gemm_mma_kernel<BM, BN, NW, ST, KT, RG, CG, true>, grid,
              dim3(NW * 32u), smem, st, w, x, (const float*)nullptr, ya,
              in_dim, m, batch, yb, yb, oq, ob);
    return (int)cudaGetLastError();
}

// slot: one launch over a plane folding exactly two projections.
PD_EXPORT
int pd_bf16_seg2_gemm_mma(const void* w, const void* x, void* ya, void* yb,
                          uint32_t in_dim, uint32_t oq, uint32_t ob,
                          uint32_t batch, void* stream) {
    if (in_dim == 0 || oq == 0 || ob == 0 || batch == 0) return 0;
    if (batch < 2u || (in_dim & 15u)) return -2;
    cudaStream_t st = (cudaStream_t)stream;
    const __nv_bfloat16* wp = (const __nv_bfloat16*)w;
    const float* xp = (const float*)x;
    float* ap = (float*)ya;
    float* bp = (float*)yb;
    if (batch <= 8u)
        return pd_bf16_seg2_cfg<32u, 32u, 4u, 2u, 128u, 1u, 2u>(
                wp, xp, ap, bp, in_dim, oq, ob, batch, st);
    if (batch <= 16u)
        return pd_bf16_seg2_cfg<32u, 32u, 4u, 3u, 64u, 1u, 2u>(
                wp, xp, ap, bp, in_dim, oq, ob, batch, st);
    if (batch <= 32u)
        return pd_bf16_seg2_cfg<64u, 32u, 8u, 4u, 64u, 2u, 1u>(
                wp, xp, ap, bp, in_dim, oq, ob, batch, st);
    return pd_bf16_seg2_cfg<64u, 64u, 8u, 3u, 32u, 2u, 2u>(
            wp, xp, ap, bp, in_dim, oq, ob, batch, st);
}

// Load-time row permute for the HCMIX up plane: original row (s*hidden + d)
// moves to (d/8)*32 + s*8 + (d%8). Bijective over hc*hidden rows. Run once per
// plane; the permuted plane is the only one the mix arm ever reads.
// ---------------------------------------------------------------------------
// Weight prep for the low-M HC island (slots 573/574).
//
// 573: pad a [rows_src, cols] bf16 plane up to rows_dst with zero rows. The
//      low-M GEMM carries the output width in its MMA-M tile, so the hc down
//      plane's 324 rows (320 low-rank + 4 inject) have to reach a multiple of
//      64 before the inject block can be read as its own gemm.
__global__ void pd_bf16_pad_rows_kernel(const __nv_bfloat16* __restrict__ src,
                                        __nv_bfloat16* __restrict__ dst,
                                        uint32_t rows_src, uint32_t rows_dst,
                                        uint32_t cols) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)rows_dst * cols) return;
    const uint32_t r = (uint32_t)(i / cols);
    dst[i] = (r < rows_src) ? src[i] : __float2bfloat16(0.0f);
}

PD_EXPORT
int pd_bf16_pad_rows(const void* src, void* dst, uint32_t rows_src,
                     uint32_t rows_dst, uint32_t cols, void* stream) {
    if (rows_dst == 0 || cols == 0) return 0;
    const size_t n = (size_t)rows_dst * cols;
    pd_bf16_pad_rows_kernel<<<(unsigned)((n + 255) / 256), 256, 0,
                              (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)src, (__nv_bfloat16*)dst, rows_src, rows_dst, cols);
    return (int)cudaGetLastError();
}

// 574: the up plane in the GATE epilogue's row order. That epilogue reduces
// over the hc branches of one hidden index, so branch s of hidden d must sit
// at row d*hc + s; columns are padded to `kpad` (the gate gemm's K) with
// zeros. src is [hc*hidden, lowrank], dst is [hc*hidden, kpad].
__global__ void pd_bf16_hc_perm_pad_kernel(const __nv_bfloat16* __restrict__ src,
                                           __nv_bfloat16* __restrict__ dst,
                                           uint32_t hidden, uint32_t hc,
                                           uint32_t lr, uint32_t kpad) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t rows = (size_t)hidden * hc;
    if (i >= rows * kpad) return;
    const uint32_t r = (uint32_t)(i / kpad);
    const uint32_t j = (uint32_t)(i - (size_t)r * kpad);
    // r is the DESTINATION row: r = d*hc + s
    const uint32_t d = r / hc, s = r - d * hc;
    dst[i] = (j < lr) ? src[((size_t)s * hidden + d) * lr + j] : __float2bfloat16(0.0f);
}

PD_EXPORT
int pd_bf16_hc_perm_pad(const void* src, void* dst, uint32_t hidden, uint32_t hc,
                        uint32_t lr, uint32_t kpad, void* stream) {
    if (hidden == 0 || hc == 0 || kpad == 0) return 0;
    const size_t n = (size_t)hidden * hc * kpad;
    pd_bf16_hc_perm_pad_kernel<<<(unsigned)((n + 255) / 256), 256, 0,
                                 (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)src, (__nv_bfloat16*)dst, hidden, hc, lr, kpad);
    return (int)cudaGetLastError();
}

__global__ void pd_bf16_hcmix_permute_kernel(const __nv_bfloat16* __restrict__ src,
                                             __nv_bfloat16* __restrict__ dst,
                                             uint32_t hidden, uint32_t hc,
                                             uint32_t in_dim) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t rows = (size_t)hidden * hc;
    if (i >= rows * in_dim) return;
    const uint32_t r = (uint32_t)(i / in_dim);
    const uint32_t k = (uint32_t)(i - (size_t)r * in_dim);
    const uint32_t s = r / hidden, d = r % hidden;
    const uint32_t pr = (d >> 3) * 32u + s * 8u + (d & 7u);
    dst[(size_t)pr * in_dim + k] = src[i];
}

PD_EXPORT
int pd_bf16_hcmix_permute(const void* src, void* dst, uint32_t hidden,
                          uint32_t hc, uint32_t in_dim, void* stream) {
    if (hidden == 0 || hc == 0 || in_dim == 0) return 0;
    if ((hidden & 7u) != 0u || hc != 4u) return cudaErrorInvalidValue;
    const size_t n = (size_t)hidden * hc * in_dim;
    pd_bf16_hcmix_permute_kernel<<<(unsigned)((n + 255) / 256), 256, 0,
                                   (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)src, (__nv_bfloat16*)dst, hidden, hc, in_dim);
    return pd_launch_status();
}

// slot: up-GEMM with the hyper-connection mix tail in its epilogue. Consumes a
// plane permuted by pd_bf16_hcmix_permute; writes only the mixed output, so the
// [rows][hc*hidden] gate plane is never materialised.
PD_EXPORT
int pd_bf16_hcmix_gemm(const void* w, const void* x, const void* xn, void* out,
                       uint32_t in_dim, uint32_t hidden, uint32_t hc,
                       uint32_t batch, void* stream) {
    if (in_dim == 0 || hidden == 0 || batch == 0) return 0;
    if (batch < 2u || hc != 4u || (in_dim & 15u) || (hidden & 7u)) return -2;
    constexpr uint32_t BM = 64u, BN = 32u, NW = 8u, ST = 4u, KT = 64u, RG = 2u, CG = 1u;
    constexpr uint32_t KPAD = KT + 8u;
    constexpr uint32_t smem = ST * (BM * KPAD + BN * KPAD) * 2u;
    static bool set = false;
    if (!set) {
        if (cudaFuncSetAttribute(
                pd_bf16_gemm_mma_kernel<BM, BN, NW, ST, KT, RG, CG, false, false, false, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess)
            return -2;
        set = true;
    }
    const uint32_t m = hidden * hc;
    dim3 grid((m + BM - 1u) / BM, (batch + BN - 1u) / BN);
    pd_bf16_gemm_mma_kernel<BM, BN, NW, ST, KT, RG, CG, false, false, false, true>
        <<<grid, NW * 32u, smem, (cudaStream_t)stream>>>(
            (const __nv_bfloat16*)w, (const float*)x, nullptr, nullptr, in_dim,
            m, batch, (float*)xn, (float*)out, hidden, hc);
    return (int)cudaGetLastError();
}

// Batched entry. Declines (-2) below batch 2 (the serial row keeps its GEMV)
// and on ragged in_dim (the tile GEMM stays correct there) - the Rust route
// treats nonzero as "keep the fallback".
//
// It used to decline batch <= 8 as well, on the grounds that the multi-row
// GEMV owns that band. That made `PADDOCK_BF16_MR=0` - the knob whose whole
// purpose is to hand 2..=8 to the tile - fall through to the f32-FMA tile
// instead, and the qwen4_exp c4/c8 A/B it was written for measured 67-75
// ms/step and could never have answered its own question. The fused q|k|v
// twin below has always served batch <= 8 with <32,32,4,2,128,1,2>, which is
// the same config the `batch <= 16` tier here picks, so this band is not new
// ground for the kernel. The DEFAULT route is unchanged: the Rust dispatch
// still gives 2..=8 to the multi-row GEMV unless the knob says otherwise. No PDL arm: this band only runs in the eager
// chunked-prefill pass, never inside a captured decode graph. Configs are the
// f16 census picks (narrow tile below 64 cols, fat 128x128 above) - with a
// grid-fill gate on top: the fat tile's grid on the paddleocr-vl
// decoder's square 1024-wide planes is ceil(1024/128)^2 = 64 blocks, a third
// of the PRO 6000's SMs in a single wave, which held the tick's GEMM band at
// ~56 effective TFLOPS. When the fat grid
// cannot fill the device once, the narrow tile quadruples the block count at
// the same per-element K-walk (summation order is config-independent - RG/CG
// move warp tile ownership, not the k sequence), so it wins on occupancy.
PD_EXPORT
int pd_bf16_gemm_mma(const void* w, const void* bias, const void* x, void* y,
                     uint32_t in_dim, uint32_t out_dim, uint32_t batch,
                     void* stream) {
    if (out_dim == 0 || in_dim == 0 || batch == 0) return 0;
    // batch 2..8 is admitted: on big-out planes (lm_head 4096x100352) the
    // b<=16 tier config holds the wall at b=8 (556.8 us cold vs the mr
    // GEMV's 1020.4 on a 4-plane rotation; NB=8 is the mr kernel's
    // issue-bound arm). The ENGINE elects
    // by out width - small-out planes keep the mr class this entry used to
    // refuse into.
    if (batch < 2u || (in_dim & 15u)) return -2;
    cudaStream_t st = (cudaStream_t)stream;
    const __nv_bfloat16* wp = (const __nv_bfloat16*)w;
    const float* xp = (const float*)x;
    const float* bp = (const float*)bias;
    float* yp = (float*)y;
    static int sms = -1;
    if (sms < 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev)
                != cudaSuccess || sms <= 0)
            sms = 1; // query failure: fall through to the batch-only rule
    }
    // HUGE-out_dim arm (rung 5: the lm_head memory path). Same-instrument nsys
    // at c32 put our lm_head at 439.9 us / 2.89 TB/s = 36% of the 8 TB/s roof
    // against the rival's nvjet at 194.2 us / 6.54 TB/s = 82%, on a 3880-block
    // grid (26 per SM) - so the gap there is not occupancy and not tile shape,
    // it is how the weight plane streams. Every element is read exactly once at
    // decode widths, so this is pure streaming.
    //
    // At vocab-scale out_dim the grid still fills with a 256-row tile, and
    // FEWER-BIGGER CTAs stream it far better than the decode-band tile.
    // Measured on the live lm_head [248320, 2560] bf16 (bench/lmhead_mem.cu,
    // 2-clone rotation, 1.27 GB per pass so it is DRAM-cold by construction):
    //   batch 16: 458.0 -> 279.4 us  (35% -> 57% of roof)
    //   batch 32: 441.4 -> 296.1 us  (36% -> 54%)
    //   batch 64: 900.5 -> 525.3 us  (18% -> 30%)
    // BIT-EXACT against the shipped tile over all 7.9M outputs - the "configs
    // move tile ownership, never the k sequence" claim below is what makes an
    // election change safe, so it is CHECKED in the bench, not quoted.
    //
    // What the sweep REFUTED, worth keeping: the obvious hypothesis was
    // contiguous-run length (the shipped tile asks DRAM for 128 B every 5120).
    // It is wrong - at BM=256, KT=64 (128 B/row/stage) BEATS KT=128 (256 B),
    // and KT=256 is worse than both. Rows per CTA is the axis; run length is
    // not. BM 384 and 512, and 16-warp CTAs, are all worse than 256, so this
    // is a peak and not a trend to extrapolate.
    //
    // Still 54% against their 82%: config sweeping cannot close the rest, which
    // needs the tcgen05 + TMA datapath their nvjet kernel uses. Bounded here.
    // Decode-band TMA widening (2026-08-29 probe, bench/dense_ab.cu, b8):
    // the TMA arm beats the cp.async tile on the two WIDE decode planes --
    // GDN qkv [2560->10240] 36.87 -> 30.98 us, attn q [2560->12288]
    // 36.87 -> 31.71 -- and loses below ~8k rows (out=6144: 30.75 vs 22.54;
    // out=2560 k=6144: 69.69 vs 24.59, the BM=128 tile starves). BIT-EXACT
    // vs the tile by the arm's own documented contract (same k-walk, same
    // per-warp mma order), so this is schedule-only. Opt-in floor in rows;
    // 0/unset = the huge-plane gate below is the only entry.
    static const uint32_t tma_out = [] {
        const char* v = pd_env("PADDOCK_BF16_TMA_OUT");
        return v ? (uint32_t)atoi(v) : 0u;
    }();
#if defined(PD_BS_HOST) || defined(PD_TC5_HOST)
    if (tma_out && out_dim >= tma_out && bp == nullptr) {
        const int r = pd_bf16_tma_cfg<32u, 8u, 2u, 1u, 4u>(
                wp, xp, bp, yp, in_dim, out_dim, batch, st);
        if (r >= 0) return r;
    }
#endif
    if (out_dim >= 256u * (uint32_t)sms) {
#if defined(PD_BS_HOST) || defined(PD_TC5_HOST)
        // TMA arm first where it applies. BIT-EXACT vs the cp.async tile over
        // all 7.9M lm_head outputs (same k-walk, same per-warp mma order), and
        // measured 296.1 -> 261.4 us (54% -> 61% of roof) on that shape.
        // It does not generalise: BM=128 is fixed by the 128x128B box and the
        // arm carries no K-split, so every narrower plane starves. Measured
        // against an unsplit reference at batch 32: GDN qkv 1.04x, attn qkv
        // 1.04x, hc up 0.99x, GDN z 0.86x, out-proj 0.82x, hc down 0.82x -
        // i.e. only the vocab-scale plane pays, which is exactly the gate it
        // sits behind. Declines (returns -1) on any shape it cannot serve.
        if (bp == nullptr) {
            const int r = pd_bf16_tma_cfg<32u, 8u, 2u, 1u, 4u>(
                    wp, xp, bp, yp, in_dim, out_dim, batch, st);
            if (r >= 0) return r;
        }
#endif
        return pd_bf16_mma_cfg<256u, 32u, 8u, 2u, 64u, 8u, 1u>(
                wp, xp, bp, yp, in_dim, out_dim, batch, st);
    }
    // Decode-band tier: at batch<=32
    // the BN=32/KT=128/ST=2 2-CTA config streams the probed shapes ~40%
    // faster than the 64x64 tile (q 4096x2688 b32: 33.2 vs 45.5 us; wo
    // 2688x4096: 46.8 vs 61.4) - BN=64 pads half the B tile away at these
    // widths, and the deeper KT holds more bytes in flight per CTA (the
    // latency lever). Same per-element k-walk as every other config
    // (configs move tile ownership, never the k sequence), so the pick is
    // bit-neutral.
    // A later DRAM-honest 12-clone sweep: <64,32,8,ST=4,KT=64,2,1> beats this
    // arm on both served shapes -- wo 2688x4096 b32 40.9 us vs 46.8 -- and it
    // is the same tile with a DEEPER pipeline and a SHALLOWER K tile, so the
    // earlier pick (which swept BM/BN but held ST/KT) could not see it.
    // Configs move tile ownership, never the k sequence, so the pick stays
    // bit-neutral.
    // CORRECTION to that reading: the grid-fill ladder's "538 GB/s flat
    // across b32..b256" looked like saturation, but a GB/s-wt metric divides
    // SINGLE-plane bytes by time while batch doubles the PHYSICAL weight
    // traffic - flat GB/s-wt means time held constant as bytes doubled, i.e.
    // aggregate bandwidth scaled with CTA count. That is starvation, and the
    // K-split it argued against is exactly what pays: wo b32 40.9 -> 22.2 us
    // (538 -> 990 GB/s), kv b16 26.6 -> 10.2, qkv32 28.7 -> 26.5 (already at
    // 55%). The KS arm below the tier ladder is the cure; the row-stride
    // observation stands only as the reason qkv (k=2688) streams faster per
    // CTA than wo (k=4096).
    // BN=32 pads three quarters of the B tile away below b16, which is why
    // the ST=4/KT=64 pick measured +0.46% at c32 and -0.77% at c8 when it
    // was applied to the whole batch<=32 band.
    // Band it at the same 16 boundary the qkv arm already uses: the narrow
    // tile keeps the small widths, the deep-pipeline tile takes 16<b<=32.
    // NARROW-N decode tile. ncu on gdn qkv b8 (bench/dense_ab.cu) found the
    // gap to the vendor kernel is neither bytes (52.5 MB both), nor occupancy
    // (ours 13.7% vs theirs 9.9%), nor staging depth (174 KB measured worse
    // than 55 KB), nor bulk-async issue (TMA: 1.19x here, negative on two
    // other planes). It is the ACTIVATION tile: BN=32 against a batch of 8
    // stages 32 f32 columns per CTA, ~102 MB of L2 traffic against 52 MB of
    // DRAM -- L2 sector hit 41% where the vendor kernel sits at 2%, and a
    // LOWER memory-stall ratio (3.19 vs 10.41) because we are L1/L2-issue
    // bound, not DRAM bound. Narrowing the B tile to the batch (bench, 200
    // iters, us):
    //   gdn qkv [2560->10240] b4 36.9->18.4  b8 36.9->19.0   (2.0x / 1.94x)
    //   attn q  [2560->12288] b8 36.9->19.0                  (1.94x)
    //   gdn z   [2560-> 6144] b8 22.5->12.3                  (1.83x)
    //   gdn out [6144-> 2560] b4 24.6->14.4  b8 24.6->14.4   (1.71x)
    // BIT-NEUTRAL by CONSTRUCTION, and the construction is the point: BM is
    // unchanged at 32 and `blocks2d` carries ceil(batch/BN), which at batch<=8
    // is 1 for BN=8 and for BN=32 alike -- so the K-split election sees the
    // same grid and picks the same nz, and the per-element k walk is
    // untouched. That coupling is not free above 8 (out=6144 b16 would move
    // nz 5 -> 0), which is why the band stops at 8 rather than 16.
    if (batch <= 8u)
        return pd_bf16_mma_cfg<32u, 8u, 2u, 2u, 128u, 1u, 1u>(
                wp, xp, bp, yp, in_dim, out_dim, batch, st);
    if (batch <= 16u)
        return pd_bf16_mma_cfg<32u, 32u, 4u, 2u, 128u, 1u, 2u>(
                wp, xp, bp, yp, in_dim, out_dim, batch, st);
    if (batch <= 32u)
        return pd_bf16_mma_cfg<64u, 32u, 8u, 4u, 64u, 2u, 1u>(
                wp, xp, bp, yp, in_dim, out_dim, batch, st);
    const uint32_t fat_blocks =
            ((out_dim + 127u) / 128u) * ((batch + 127u) / 128u);
    if (batch <= 64u || fat_blocks < (uint32_t)sms)
        return pd_bf16_mma_cfg<64u, 64u, 8u, 3u, 32u, 2u, 2u>(
                wp, xp, bp, yp, in_dim, out_dim, batch, st);
    return pd_bf16_mma_cfg<128u, 128u, 8u, 3u, 32u, 4u, 4u>(
            wp, xp, bp, yp, in_dim, out_dim, batch, st);
}

// Fused q|k|v decode-band entry (thin-k/v rung): one launch over
// the load-time-concatenated [q;k;v] plane (in_dim x (oq + 2*okv)) against
// the shared x, segmented store into yq/yk/yv ([batch, seg] each). Per
// out-row bit-identical to pd_bf16_gemm_mma on the matching segment. The
// probe's per-band picks: b8 20.8 us / b16 26.6 / b32 34.6 at the nemotron
// fused shape (4608x2688) vs 131 us for the three separate launches at b32
// - the thin k/v rows were latency-starved on their own grids. Declines
// (-2) batch<2 (the serial row keeps its per-segment GEMVs) and ragged
// in_dim.
PD_EXPORT
int pd_bf16_qkv_gemm_mma(const void* w, const void* x, void* yq, void* yk,
                         void* yv, uint32_t in_dim, uint32_t oq, uint32_t okv,
                         uint32_t batch, void* stream) {
    if (in_dim == 0 || (oq == 0 && okv == 0) || batch == 0) return 0;
    if (batch < 2u || (in_dim & 15u)) return -2;
    cudaStream_t st = (cudaStream_t)stream;
    const __nv_bfloat16* wp = (const __nv_bfloat16*)w;
    const float* xp = (const float*)x;
    float* qp = (float*)yq;
    float* kp = (float*)yk;
    float* vp = (float*)yv;
    if (batch <= 8u)
        return pd_bf16_qkv_cfg<32u, 32u, 4u, 2u, 128u, 1u, 2u>(
                wp, xp, qp, kp, vp, in_dim, oq, okv, batch, st);
    if (batch <= 16u)
        return pd_bf16_qkv_cfg<32u, 32u, 4u, 3u, 64u, 1u, 2u>(
                wp, xp, qp, kp, vp, in_dim, oq, okv, batch, st);
    // same sweep: ST=4/KT=64 over ST=2/KT=128 on the fused plane -
    // 4608x2688 b32 28.7 us vs 34.6, bit-neutral (config, not k order).
    return pd_bf16_qkv_cfg<64u, 32u, 8u, 4u, 64u, 2u, 1u>(
            wp, xp, qp, kp, vp, in_dim, oq, okv, batch, st);
}

// bf16 -> f32 widen with the DequantF32Fn shape (src, dst, n_blocks, stream),
// 32 elements per "block" so it slots into the engine's dequant_for table next
// to the real quant types. Feeds dequant_slice, i.e. the single-row embedding
// gather off a bf16 token_embd.
__global__ void pd_bf16_dequant_f32_kernel(const __nv_bfloat16* __restrict__ src,
                                           float* __restrict__ dst, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __bfloat162float(src[i]);
}

PD_EXPORT
int pd_bf16_dequant_f32(const void* src, void* dst, uint64_t n_blocks, void* stream) {
    if (n_blocks == 0) return 0;
    uint64_t n = n_blocks * 32ull;
    uint32_t threads = 256;
    uint64_t blocks = (n + threads - 1) / threads;
    pd_bf16_dequant_f32_kernel<<<(uint32_t)blocks, threads, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)src, (float*)dst, n);
    return pd_launch_status();
}

// bf16 twin of pd_embed_gather_q8: gathers device-selected token rows straight
// out of a bf16 embedding table with the fused output scale. Graph-capturable
// (token ids read from device memory), same as the Q8_0 one.
__global__ void pd_embed_gather_bf16_kernel(const __nv_bfloat16* __restrict__ table,
                                            const uint32_t* __restrict__ tokens,
                                            float* __restrict__ out, uint32_t embd,
                                            float scale) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t t = blockIdx.y;
    if (i >= embd) return;
    out[(size_t)t * embd + i] =
        __bfloat162float(table[(size_t)tokens[t] * embd + i]) * scale;
}

PD_EXPORT
int pd_embed_gather_bf16(const void* table, const void* tokens, void* out,
                         uint32_t embd, uint32_t n_tokens, float scale, void* stream) {
    if (embd == 0 || n_tokens == 0) return 0;
    uint32_t threads = 256;
    dim3 grid((embd + threads - 1) / threads, n_tokens);
    pd_embed_gather_bf16_kernel<<<grid, threads, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)table, (const uint32_t*)tokens, (float*)out, embd, scale);
    return pd_launch_status();
}
