// sam3/text.cuh - SAM 3's text tower glue: causal self-attention over a 32-token CLIP prompt
// Textually-included segment of the single pack translation unit.
// Not standalone-compilable: include order is defined by ../pack.cu.
// ------------------------------------------------------------ sam3 text tower
// The text tower is CLIP's (Meta's TextTransformer: 24 pre-LN blocks, width
// 1024, 16 heads, causal mask, 32 positions). Its GEMMs and seams are the
// dense lane's; the attention is the one op none of the existing kernels
// fits. The vision attention has no mask, and the causal prefill kernels are
// built around a paged KV cache this tower does not have.
//
// The work is tiny - a prompt is 32 rows - so the form is about latency:
// one block per (prompt, head) with K and V staged in shared memory as f32,
// one warp per query row. Lane j scores key j (and j + 32) with the dims in
// order, the warp takes the max and the sum by shuffles, then each lane folds
// two output dims over keys 0..row in order, every weight broadcast from the
// lane that made it. The first form - one THREAD per row, q and the
// accumulator in per-thread arrays the compiler parked in local memory - ran
// 82-90 us a layer on the A6000; this one is a few. Softmax is f32 with the
// max subtracted, in a fixed order, so it is run-to-run identical. Padding
// rows are computed too (they attend causally like any row); every consumer
// masks them out, and a valid row never sees them because padding only
// follows the end marker.
//
// Needs f32_qkv.cuh's pd_launch_status.

#define PD_SAM3_TEXT_MAXT 64u
#define PD_SAM3_TEXT_MAXD 64u
// a key row padded by one float: lane j reading row j at dim e hits bank
// (j + e) % 32, so the 32 lanes of a score step never collide
#define PD_SAM3_TEXT_KP (PD_SAM3_TEXT_MAXD + 1u)

// 759: out[p][t][h] = softmax_{j <= t}(q_t . k_j) v_j. q arrives PRE-SCALED by
// 1/sqrt(hd) (the split kernel folds it into q's one round). All planes f16
// [prompts][T][H][hd], the split kernel's layout. 2 x 64 x 65 f32 of K/V and
// a 64-wide q row a warp: 41 KB of static shared memory.
__global__ void __launch_bounds__(1024) pd_sam3_text_attn_h_kernel(
    const __half* __restrict__ q, const __half* __restrict__ k, const __half* __restrict__ v,
    __half* __restrict__ out, uint32_t T, uint32_t H, uint32_t hd) {
    __shared__ float sk[PD_SAM3_TEXT_MAXT * PD_SAM3_TEXT_KP];
    __shared__ float sv[PD_SAM3_TEXT_MAXT * PD_SAM3_TEXT_KP];
    __shared__ float sq[32][PD_SAM3_TEXT_MAXD];
    const uint32_t p = blockIdx.x / H, h = blockIdx.x % H;
    const size_t rs = (size_t)H * hd;  // stride between rows of one prompt
    const size_t base = (size_t)p * T * rs + (size_t)h * hd;
    for (uint32_t i = threadIdx.x; i < T * hd; i += blockDim.x) {
        const uint32_t j = i / hd, e = i - j * hd;
        sk[j * PD_SAM3_TEXT_KP + e] = __half2float(k[base + j * rs + e]);
        sv[j * PD_SAM3_TEXT_KP + e] = __half2float(v[base + j * rs + e]);
    }
    __syncthreads();
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    float* qs = sq[warp];
    for (uint32_t t = warp; t < T; t += nw) {
        const __half* qr = q + base + t * rs;
        for (uint32_t e = lane; e < hd; e += 32u) qs[e] = __half2float(qr[e]);
        __syncwarp();
        // the lane's keys: lane and lane + 32, causal
        float s[2];
#pragma unroll
        for (uint32_t u = 0; u < 2u; ++u) {
            const uint32_t j = lane + 32u * u;
            float a = -INFINITY;
            if (j <= t) {
                a = 0.0f;
                const float* kr = sk + j * PD_SAM3_TEXT_KP;
                for (uint32_t e = 0; e < hd; ++e) a += qs[e] * kr[e];
            }
            s[u] = a;
        }
        float m = fmaxf(s[0], s[1]);
#pragma unroll
        for (uint32_t o = 16u; o > 0u; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        float w[2];
#pragma unroll
        for (uint32_t u = 0; u < 2u; ++u) w[u] = lane + 32u * u <= t ? expf(s[u] - m) : 0.0f;
        float l = w[0] + w[1];
#pragma unroll
        for (uint32_t o = 16u; o > 0u; o >>= 1) l += __shfl_xor_sync(0xffffffffu, l, o);
        // the lane's dims: lane and lane + 32, keys folded in order
        float a0 = 0.0f, a1 = 0.0f;
        for (uint32_t j = 0; j <= t; ++j) {
            const float wj = __shfl_sync(0xffffffffu, j < 32u ? w[0] : w[1], j & 31u);
            const float* vr = sv + j * PD_SAM3_TEXT_KP;
            a0 += wj * vr[lane];
            a1 += wj * vr[lane + 32u];
        }
        const float inv = 1.0f / l;
        __half* orow = out + base + t * rs;
        if (lane < hd) orow[lane] = __float2half(a0 * inv);
        if (lane + 32u < hd) orow[lane + 32u] = __float2half(a1 * inv);
        __syncwarp();  // every lane done with this q row before the next lands
    }
}

PD_EXPORT
int pd_sam3_text_attn_h(const void* q, const void* k, const void* v, void* out, uint32_t prompts,
                        uint32_t T, uint32_t H, uint32_t hd, void* stream) {
    if (prompts == 0 || T == 0 || H == 0) return 0;
    if (T > PD_SAM3_TEXT_MAXT || hd == 0 || hd > PD_SAM3_TEXT_MAXD) return cudaErrorInvalidValue;
    const uint32_t nth = 32u * (T < 32u ? T : 32u);
    pd_sam3_text_attn_h_kernel<<<prompts * H, nth, 0, (cudaStream_t)stream>>>(
        (const __half*)q, (const __half*)k, (const __half*)v, (__half*)out, T, H, hd);
    return pd_launch_status();
}
