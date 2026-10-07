// sam3/tracker.cuh - SAM 3's tracker heads on one picture (PVS): the prompt encoder's point and mask paths, the mask decoder's upscaling seam, its small MLP heads, the stability count and the hole fill
// Textually-included segment of the single pack translation unit.
// Not standalone-compilable: include order is defined by ../pack.cu.
// ------------------------------------------------------------ sam3 tracker (PVS)
// Interactive single-object segmentation - clicks and a box on one picture,
// Meta's SAM3InteractiveImagePredictor - is the SAM 2 lineage head over the
// tracker neck: a prompt encoder (random-Fourier point features, a learned
// embedding a label, a small conv stack for a mask prompt), a two-way
// transformer (the dense lane's GEMMs, pd_vision_attn_h and pd_sam3_seam_h)
// and an upscaling + hypernetwork head. What is here is everything between
// those that no other tower needed. f32 throughout: these are tiny planes,
// read once.
//
// Needs f32_qkv.cuh's pd_launch_status.

// 778: random-Fourier position features, Meta's PositionEmbeddingRandom:
//   pe(x, y) = [sin(2pi((2x-1) G0 + (2y-1) G1)), cos(...)]   (2 x 128 G)
// for points already normalized to [0, 1] (the caller's +0.5 and / 1008).
// Then the label's embedding, as _embed_points: -1 = padding (the pe is
// REPLACED by not_a_point), 0 negative / 1 positive / 2 box top-left / 3 box
// bottom-right ADD point_embed[label]. Label -2 is the plain pe with nothing
// added - the dense image pe the decoder reads, built once at load from the
// 72^2 pixel centres. One block a point, 128 threads (a frequency each).
__global__ void pd_sam3_point_pe_kernel(const float* __restrict__ xy, const int32_t* __restrict__ labels,
                                        const float* __restrict__ g, const float* __restrict__ emb,
                                        const float* __restrict__ nap, float* __restrict__ out,
                                        uint32_t nf) {
    const uint32_t p = blockIdx.x, f = threadIdx.x;
    if (f >= nf) return;
    const float x = 2.0f * xy[2u * p] - 1.0f, y = 2.0f * xy[2u * p + 1u] - 1.0f;
    const float a = 6.283185307179586f * (x * g[f] + y * g[nf + f]);
    float s = sinf(a), c = cosf(a);
    const int32_t lab = labels ? labels[p] : -2;
    float* o = out + (size_t)p * 2u * nf;
    if (lab == -1) {
        s = nap[f];
        c = nap[nf + f];
    } else if (lab >= 0 && lab <= 3) {
        s += emb[(size_t)lab * 2u * nf + f];
        c += emb[(size_t)lab * 2u * nf + nf + f];
    }
    o[f] = s;
    o[nf + f] = c;
}

PD_EXPORT
int pd_sam3_point_pe(const void* xy, const void* labels, const void* g, const void* emb,
                     const void* nap, void* out, uint32_t n, uint32_t nf, void* stream) {
    if (n == 0) return 0;
    if (nf == 0 || nf > 1024u) return cudaErrorInvalidValue;
    pd_sam3_point_pe_kernel<<<n, (nf + 31u) & ~31u, 0, (cudaStream_t)stream>>>(
        (const float*)xy, (const int32_t*)labels, (const float*)g, (const float*)emb,
        (const float*)nap, (float*)out, nf);
    return pd_launch_status();
}

// 779: one stage of the mask prompt's downscaling, Meta's mask_downscaling:
// Conv2d(k2, s2) -> LayerNorm2d(eps) -> GELU(erf). in f32, channel-last
// [H][W] rows of `in_stride` floats from column `in_off` with Cin channels
// (the first stage reads one column of the decoder's [px][masks] logits
// plane, clamped to +-clamp - the predictor clamps the low-res logits it
// hands back at 32); out f32 [H/2][W/2][Cout] or f16 when out16. Weight
// [Cout][Cin][2][2] as Conv2d stores it. One thread an output pixel: these
// are 4 and 16 channels.
#define PD_SAM3_MD_MAXC 32u
__global__ void pd_sam3_mask_down_kernel(const float* __restrict__ in, const float* __restrict__ w,
                                         const float* __restrict__ b, const float* __restrict__ lw,
                                         const float* __restrict__ lb, void* __restrict__ out,
                                         uint32_t H, uint32_t W, uint32_t cin, uint32_t cout,
                                         uint32_t in_stride, uint32_t in_off, float clamp, float eps,
                                         uint32_t out16) {
    const uint32_t oh = H / 2u, ow = W / 2u;
    const uint32_t o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= oh * ow) return;
    const uint32_t oy = o / ow, ox = o - oy * ow;
    float acc[PD_SAM3_MD_MAXC];
    for (uint32_t co = 0; co < cout; ++co) {
        float a = b[co];
        for (uint32_t ci = 0; ci < cin; ++ci)
            for (uint32_t ky = 0; ky < 2u; ++ky)
                for (uint32_t kx = 0; kx < 2u; ++kx) {
                    const uint32_t yy = 2u * oy + ky, xx = 2u * ox + kx;
                    float v = in[((size_t)yy * W + xx) * in_stride + in_off + ci];
                    if (clamp > 0.0f) v = fminf(fmaxf(v, -clamp), clamp);
                    a += w[((co * cin + ci) * 2u + ky) * 2u + kx] * v;
                }
        acc[co] = a;
    }
    float mean = 0.0f;
    for (uint32_t co = 0; co < cout; ++co) mean += acc[co];
    mean /= (float)cout;
    float var = 0.0f;
    for (uint32_t co = 0; co < cout; ++co) var += (acc[co] - mean) * (acc[co] - mean);
    const float inv = rsqrtf(var / (float)cout + eps);
    for (uint32_t co = 0; co < cout; ++co) {
        float v = (acc[co] - mean) * inv * lw[co] + lb[co];
        v = 0.5f * v * (1.0f + erff(v * 0.70710678118654752440084436210484f));
        if (out16) ((__half*)out)[(size_t)o * cout + co] = __float2half(v);
        else ((float*)out)[(size_t)o * cout + co] = v;
    }
}

PD_EXPORT
int pd_sam3_mask_down(const void* in, const void* w, const void* b, const void* lw, const void* lb,
                      void* out, uint32_t H, uint32_t W, uint32_t cin, uint32_t cout,
                      uint32_t in_stride, uint32_t in_off, float clamp, float eps, uint32_t out16,
                      void* stream) {
    if (H < 2u || W < 2u) return 0;
    if (cout == 0 || cout > PD_SAM3_MD_MAXC || cin == 0 || in_off + cin > in_stride)
        return cudaErrorInvalidValue;
    const uint32_t n = (H / 2u) * (W / 2u);
    pd_sam3_mask_down_kernel<<<(n + 127u) / 128u, 128, 0, (cudaStream_t)stream>>>(
        (const float*)in, (const float*)w, (const float*)b, (const float*)lw, (const float*)lb, out, H,
        W, cin, cout, in_stride, in_off, clamp, eps, out16);
    return pd_launch_status();
}

// 780: the mask decoder's upscaling seam, one of its two convTs:
//   out[Y][X][c] = act(LN?(convT2x2s2(x)[Y][X][c] + bias[c] + skip[Y][X][c]))
// `g` is the convT's GEMM landing [h*w][4*C] in 754's tap-major layout,
// `skip` the tracker's high-resolution feature at the output size f32
// [2h*2w][C] (conv_s1 / conv_s0), LN over the C channels when ln_w is given
// (LayerNorm2d, eps), then the erf GELU. out f16 (the next GEMM's input)
// when out16, else f32. One block an output pixel.
__global__ void pd_sam3_up_skip_kernel(const float* __restrict__ g, const float* __restrict__ bias,
                                       const float* __restrict__ skip, const float* __restrict__ lw,
                                       const float* __restrict__ lb, void* __restrict__ out,
                                       uint32_t h, uint32_t w, uint32_t C, float eps, uint32_t out16) {
    __shared__ float red[2][32];
    const uint32_t o = blockIdx.x;
    const uint32_t W2 = 2u * w;
    const uint32_t Y = o / W2, X = o - Y * W2;
    const uint32_t tap = (Y & 1u) * 2u + (X & 1u);
    const float* gr = g + ((size_t)(Y >> 1) * w + (X >> 1)) * 4u * C + (size_t)tap * C;
    const uint32_t c = threadIdx.x;
    float v = c < C ? gr[c] + bias[c] + skip[(size_t)o * C + c] : 0.0f;
    if (lw != nullptr) {
        // two-pass block statistics over the C (<= 1024) channels, fixed order
        const uint32_t lane = c & 31u, wid = c >> 5, nw = (blockDim.x + 31u) >> 5;
        float s = v;
        for (uint32_t k = 16u; k > 0u; k >>= 1) s += __shfl_xor_sync(0xffffffffu, s, k);
        if (lane == 0u) red[0][wid] = s;
        __syncthreads();
        float mean = 0.0f;
        for (uint32_t i = 0; i < nw; ++i) mean += red[0][i];
        mean /= (float)C;
        const float dv = c < C ? v - mean : 0.0f;
        float q = dv * dv;
        for (uint32_t k = 16u; k > 0u; k >>= 1) q += __shfl_xor_sync(0xffffffffu, q, k);
        if (lane == 0u) red[1][wid] = q;
        __syncthreads();
        float var = 0.0f;
        for (uint32_t i = 0; i < nw; ++i) var += red[1][i];
        v = c < C ? dv * rsqrtf(var / (float)C + eps) * lw[c] + lb[c] : 0.0f;
    }
    if (c >= C) return;
    v = 0.5f * v * (1.0f + erff(v * 0.70710678118654752440084436210484f));
    if (out16) ((__half*)out)[(size_t)o * C + c] = __float2half(v);
    else ((float*)out)[(size_t)o * C + c] = v;
}

PD_EXPORT
int pd_sam3_up_skip(const void* g, const void* bias, const void* skip, const void* lw, const void* lb,
                    void* out, uint32_t h, uint32_t w, uint32_t C, float eps, uint32_t out16,
                    void* stream) {
    if (h == 0 || w == 0 || C == 0) return 0;
    if (C > 1024u || skip == nullptr) return cudaErrorInvalidValue;
    const uint32_t nth = (C + 31u) & ~31u;
    pd_sam3_up_skip_kernel<<<4u * h * w, nth, 0, (cudaStream_t)stream>>>(
        (const float*)g, (const float*)bias, (const float*)skip, (const float*)lw, (const float*)lb, out,
        h, w, C, eps, out16);
    return pd_launch_status();
}

// 781: a three-layer MLP over a few rows, Meta's MLP(num_layers=3): Linear ->
// ReLU -> Linear -> ReLU -> Linear (-> sigmoid when sig). Row r reads its own
// weights at r * w_stride floats into each matrix (the four hypernetwork MLPs
// are four rows of one launch). x f32 [rows][in]; weights [out][in] f32 as
// Linear stores them; out f32 [rows][od], and f16 too when out16 is given
// (the hypernetwork rows land straight into the mask GEMM's weight).
// One block a row; every dot product in index order.
#define PD_SAM3_MLP_MAXH 1024u
__global__ void pd_sam3_mlp3_rows_kernel(const float* __restrict__ x, const float* __restrict__ w1,
                                         const float* __restrict__ b1, const float* __restrict__ w2,
                                         const float* __restrict__ b2, const float* __restrict__ w3,
                                         const float* __restrict__ b3, float* __restrict__ out,
                                         __half* __restrict__ out16, uint32_t in, uint32_t hid,
                                         uint32_t od, uint32_t ws1, uint32_t ws2, uint32_t ws3,
                                         uint32_t wsb, uint32_t wsb3, uint32_t sig) {
    __shared__ float h0[PD_SAM3_MLP_MAXH], h1[PD_SAM3_MLP_MAXH];
    const uint32_t r = blockIdx.x;
    const float* xr = x + (size_t)r * in;
    const float* W1 = w1 + (size_t)r * ws1;
    const float* W2 = w2 + (size_t)r * ws2;
    const float* W3 = w3 + (size_t)r * ws3;
    const float* B1 = b1 + (size_t)r * wsb;
    const float* B2 = b2 + (size_t)r * wsb;
    const float* B3 = b3 + (size_t)r * wsb3;
    for (uint32_t j = threadIdx.x; j < hid; j += blockDim.x) {
        float a = B1[j];
        for (uint32_t i = 0; i < in; ++i) a += W1[(size_t)j * in + i] * xr[i];
        h0[j] = fmaxf(a, 0.0f);
    }
    __syncthreads();
    for (uint32_t j = threadIdx.x; j < hid; j += blockDim.x) {
        float a = B2[j];
        for (uint32_t i = 0; i < hid; ++i) a += W2[(size_t)j * hid + i] * h0[i];
        h1[j] = fmaxf(a, 0.0f);
    }
    __syncthreads();
    for (uint32_t j = threadIdx.x; j < od; j += blockDim.x) {
        float a = B3[j];
        for (uint32_t i = 0; i < hid; ++i) a += W3[(size_t)j * hid + i] * h1[i];
        if (sig) a = 1.0f / (1.0f + expf(-a));
        out[(size_t)r * od + j] = a;
        if (out16 != nullptr) out16[(size_t)r * od + j] = __float2half(a);
    }
}

PD_EXPORT
int pd_sam3_mlp3_rows(const void* x, const void* w1, const void* b1, const void* w2, const void* b2,
                      const void* w3, const void* b3, void* out, void* out16, uint32_t rows,
                      uint32_t in, uint32_t hid, uint32_t od, uint32_t ws1, uint32_t ws2,
                      uint32_t ws3, uint32_t wsb, uint32_t wsb3, uint32_t sig, void* stream) {
    if (rows == 0) return 0;
    if (in == 0 || hid == 0 || hid > PD_SAM3_MLP_MAXH || od == 0) return cudaErrorInvalidValue;
    pd_sam3_mlp3_rows_kernel<<<rows, 256, 0, (cudaStream_t)stream>>>(
        (const float*)x, (const float*)w1, (const float*)b1, (const float*)w2, (const float*)b2,
        (const float*)w3, (const float*)b3, (float*)out, (__half*)out16, in, hid, od, ws1, ws2, ws3,
        wsb, wsb3, sig);
    return pd_launch_status();
}

// 782: the single-mask stability count, Meta's _get_stability_scores: over
// column k of the decoder's [px][nm] logits plane, counts[0] = #(v > delta)
// and counts[1] = #(v > -delta), atomically into two u32 the caller zeroed.
// Integer sums, so the order never matters.
__global__ void pd_sam3_mask_stats_kernel(const float* __restrict__ m, uint32_t* __restrict__ counts,
                                          uint32_t px, uint32_t nm, uint32_t k, float delta) {
    uint32_t a = 0, b = 0;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < px; i += gridDim.x * blockDim.x) {
        const float v = m[(size_t)i * nm + k];
        a += v > delta ? 1u : 0u;
        b += v > -delta ? 1u : 0u;
    }
    for (uint32_t s = 16u; s > 0u; s >>= 1) {
        a += __shfl_xor_sync(0xffffffffu, a, s);
        b += __shfl_xor_sync(0xffffffffu, b, s);
    }
    if ((threadIdx.x & 31u) == 0u) {
        atomicAdd(&counts[0], a);
        atomicAdd(&counts[1], b);
    }
}

PD_EXPORT
int pd_sam3_mask_stats(const void* m, void* counts, uint32_t px, uint32_t nm, uint32_t k,
                       float delta, void* stream) {
    if (px == 0) return 0;
    if (k >= nm) return cudaErrorInvalidValue;
    const cudaStream_t st = (cudaStream_t)stream;
    cudaMemsetAsync(counts, 0, 2 * sizeof(uint32_t), st);
    pd_sam3_mask_stats_kernel<<<64, 256, 0, st>>>((const float*)m, (uint32_t*)counts, px, nm, k,
                                                  delta);
    return pd_launch_status();
}

// 783: Meta's hole fill (SAM2Transforms.postprocess_masks, max_hole_area):
// every 8-connected component of BACKGROUND (logit <= 0) in column k of a
// side x side [px][nm] logits plane whose area is <= max_area becomes
// foreground at logit 10 (mask_threshold + 10). Labelled with union-find on
// the device (Playne & Hawick's atomic union: a pointer only ever moves to a
// smaller index, so the forest stays a forest under any interleaving), then
// every pixel compresses to its root, the roots count their pixels, and the
// small ones fill. Component SIZES do not depend on the order the unions
// ran in, so the fill is deterministic. `lab` and `area` are u32 scratch of
// side^2 each.
#define PD_SAM3_CC_NONE 0xffffffffu
// parents are read past L1 (ld.global.cg): another SM's union lands in L2,
// and a stale L1 line would only cost extra hops - but never reading one at
// all keeps the merge pass's loops short
__device__ __forceinline__ uint32_t pd_sam3_cc_find(const uint32_t* lab, uint32_t i) {
    uint32_t p = __ldcg(&lab[i]);
    while (p != i) {
        i = p;
        p = __ldcg(&lab[i]);
    }
    return i;
}
__device__ __forceinline__ void pd_sam3_cc_union(uint32_t* lab, uint32_t a, uint32_t b) {
    for (;;) {
        a = pd_sam3_cc_find(lab, a);
        b = pd_sam3_cc_find(lab, b);
        if (a == b) return;
        if (a < b) {
            const uint32_t t = a;
            a = b;
            b = t;
        }
        // a > b: point a at b, unless a has moved since find returned it
        const uint32_t old = atomicMin(&lab[a], b);
        if (old == a) return;
        a = old;
    }
}
__global__ void pd_sam3_cc_init_kernel(const float* __restrict__ m, uint32_t* __restrict__ lab,
                                       uint32_t* __restrict__ area, uint32_t px, uint32_t nm,
                                       uint32_t k) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= px) return;
    lab[i] = m[(size_t)i * nm + k] <= 0.0f ? i : PD_SAM3_CC_NONE;
    area[i] = 0;
}
__global__ void pd_sam3_cc_merge_kernel(uint32_t* __restrict__ lab, uint32_t side) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= side * side || lab[i] == PD_SAM3_CC_NONE) return;
    const uint32_t y = i / side, x = i - y * side;
    // the four neighbours before this pixel in raster order cover all eight
    // foreground stays NONE for the whole pass, so these reads are stable
    if (x > 0 && __ldcg(&lab[i - 1u]) != PD_SAM3_CC_NONE) pd_sam3_cc_union(lab, i, i - 1u);
    if (y > 0) {
        const uint32_t up = i - side;
        if (__ldcg(&lab[up]) != PD_SAM3_CC_NONE) pd_sam3_cc_union(lab, i, up);
        if (x > 0 && __ldcg(&lab[up - 1u]) != PD_SAM3_CC_NONE) pd_sam3_cc_union(lab, i, up - 1u);
        if (x + 1u < side && __ldcg(&lab[up + 1u]) != PD_SAM3_CC_NONE)
            pd_sam3_cc_union(lab, i, up + 1u);
    }
}
__global__ void pd_sam3_cc_count_kernel(uint32_t* __restrict__ lab, uint32_t* __restrict__ area,
                                        uint32_t px) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= px || lab[i] == PD_SAM3_CC_NONE) return;
    const uint32_t r = pd_sam3_cc_find(lab, i);
    atomicAdd(&area[r], 1u);
}
__global__ void pd_sam3_cc_fill_kernel(float* __restrict__ m, const uint32_t* __restrict__ lab,
                                       const uint32_t* __restrict__ area, uint32_t px, uint32_t nm,
                                       uint32_t k, uint32_t max_area) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= px || lab[i] == PD_SAM3_CC_NONE) return;
    if (area[pd_sam3_cc_find(lab, i)] <= max_area) m[(size_t)i * nm + k] = 10.0f;
}

PD_EXPORT
int pd_sam3_fill_holes(void* m, void* lab, void* area, uint32_t side, uint32_t nm, uint32_t k,
                       uint32_t max_area, void* stream) {
    if (side == 0 || max_area == 0) return 0;
    if (k >= nm) return cudaErrorInvalidValue;
    const cudaStream_t st = (cudaStream_t)stream;
    const uint32_t px = side * side, nb = (px + 255u) / 256u;
    pd_sam3_cc_init_kernel<<<nb, 256, 0, st>>>((const float*)m, (uint32_t*)lab, (uint32_t*)area, px,
                                                nm, k);
    pd_sam3_cc_merge_kernel<<<nb, 256, 0, st>>>((uint32_t*)lab, side);
    pd_sam3_cc_count_kernel<<<nb, 256, 0, st>>>((uint32_t*)lab, (uint32_t*)area, px);
    pd_sam3_cc_fill_kernel<<<nb, 256, 0, st>>>((float*)m, (const uint32_t*)lab,
                                                (const uint32_t*)area, px, nm, k, max_area);
    return pd_launch_status();
}
