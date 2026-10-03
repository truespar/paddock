// clef_vision.cuh - Clef's image lane (Cloudflare/clef-flash's Qwen3.5 vision
// tower): the processor's resize and patchify, then the tower's own passes the
// backbone GEMM (733) and LayerNorm (724) do not cover.
// Textually-included segment of the single pack translation unit.
// Not standalone-compilable: include order is defined by ../pack.cu.
//
// The reference is Transformers' Qwen2-VL image processor as Clef's
// processor_config selects it (the torchvision backend) and the Qwen3.5
// vision model:
//
//   - resize (735 plan, 736 pass): torch's native uint8 antialiased bicubic -
//     the CPU kernel the processor runs, a separable pass a changed axis,
//     width first, a uint8 intermediate, int16 weights at the precision the
//     largest weight allows, integer sums. Byte for byte, so the tower sees
//     the processor's pixels, not an approximation of them;
//   - patchify (737): (x - 127.5) / 127.5 in F32, the 16 x 16 patches in 2 x 2
//     merge-window order, the one frame written twice (temporal patch 2);
//   - position embedding (738): the 48 x 48 learned grid read at the patch
//     grid's linspace points, bilinear in F32, added to the patch rows;
//   - rope (739): the 2D vision rotary from a table, all 72 dims of a head as
//     36 rotate_half pairs, pair k reading the row position (k < 18) or the
//     column position;
//   - attention (740): bidirectional softmax(q k^T * scale) v within each
//     image, head_dim 72, 3xTF32 on the tensor cores (the backbone's 732 form,
//     one warp a 16-row x 72-wide head tile).
//
// Every reduction walks a fixed order that depends only on its own row (or
// its own image), so an image's rows are the same bits alone or packed with
// other images in a pass.

// ------------------------------------------------------------------ 735 resample plan
// One axis of torch's antialiased bicubic (upsample_bicubic2d_aa, uint8):
// per output index, the first input index, the tap count and the int16
// weights - the Keys cubic (a = -0.5) stretched by the downscale ratio,
// normalized to sum 1 in F64, then quantized at the axis's precision: the
// largest p < 22 with round(max weight * 2^(p+1)) < 2^15. Every F64 op is
// written unfused, the reference's C++ operation order. One CTA an axis
// (the precision is the axis's maximum, so the weights are formed twice:
// once for the maximum, once to quantize - the same F64 values both times;
// `taps` is the reference's max_interp_size, ceil(support) * 2 + 1).
__device__ __forceinline__ double pd_clef_cubic(double x) {
    const double a = -0.5;
    x = fabs(x);
    if (x < 1.0) {
        // ((a + 2) * x - (a + 3)) * x * x + 1
        return __dadd_rn(__dmul_rn(__dmul_rn(__dsub_rn(__dmul_rn(a + 2.0, x), a + 3.0), x), x), 1.0);
    }
    if (x < 2.0) {
        // ((a * x - 5 * a) * x + 8 * a) * x - 4 * a
        return __dsub_rn(__dmul_rn(__dadd_rn(__dmul_rn(__dsub_rn(__dmul_rn(a, x), 5.0 * a), x), 8.0 * a), x),
                         4.0 * a);
    }
    return 0.0;
}

// output index i's (xmin, xsize), and tap j's unnormalized weight
__device__ __forceinline__ void pd_clef_resample_range(uint32_t i, uint32_t in, double scale, double support,
                                                       uint32_t taps, int64_t& xmin, int64_t& xsize,
                                                       double& center, double& inv) {
    center = __dmul_rn(scale, __dadd_rn((double)i, 0.5));
    inv = scale >= 1.0 ? __ddiv_rn(1.0, scale) : 1.0;
    xmin = (int64_t)__dadd_rn(__dsub_rn(center, support), 0.5);
    if (xmin < 0) xmin = 0;
    int64_t hi = (int64_t)__dadd_rn(__dadd_rn(center, support), 0.5);
    if (hi > (int64_t)in) hi = (int64_t)in;
    xsize = hi - xmin;
    if (xsize < 0) xsize = 0;
    if (xsize > (int64_t)taps) xsize = (int64_t)taps;
}

__device__ __forceinline__ double pd_clef_resample_w(int64_t j, int64_t xmin, double center, double inv) {
    return pd_clef_cubic(__dmul_rn(__dadd_rn(__dsub_rn((double)(j + xmin), center), 0.5), inv));
}

// the taps' sum, in tap order (the normalizer)
__device__ __forceinline__ double pd_clef_resample_total(int64_t xmin, int64_t xsize, double center, double inv) {
    double total = 0.0;
    for (int64_t j = 0; j < xsize; ++j) total = __dadd_rn(total, pd_clef_resample_w(j, xmin, center, inv));
    return total;
}

__global__ void __launch_bounds__(256) pd_clef_resample_plan_kernel(uint32_t in, uint32_t out, uint32_t taps,
                                                                    uint32_t* __restrict__ xmin_o,
                                                                    uint32_t* __restrict__ xsize_o,
                                                                    int16_t* __restrict__ w_o,
                                                                    uint32_t* __restrict__ prec_o) {
    __shared__ double red[8];
    __shared__ uint32_t prec_s;
    PD_PDL_ARM();
    // area_pixel_compute_scale (align_corners false, no scale given): in / out
    const double scale = __ddiv_rn((double)in, (double)out);
    const double support = scale >= 1.0 ? __dmul_rn(2.0, scale) : 2.0;  // interp_size 4 * 0.5
    double mx = 0.0;
    for (uint32_t i = threadIdx.x; i < out; i += blockDim.x) {
        int64_t x0, xn;
        double center, inv;
        pd_clef_resample_range(i, in, scale, support, taps, x0, xn, center, inv);
        const double total = pd_clef_resample_total(x0, xn, center, inv);
        if (total != 0.0)
            for (int64_t j = 0; j < xn; ++j) mx = fmax(mx, __ddiv_rn(pd_clef_resample_w(j, x0, center, inv), total));
    }
#pragma unroll
    for (uint32_t o = 16; o > 0; o >>= 1) mx = fmax(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    if ((threadIdx.x & 31u) == 0) red[threadIdx.x >> 5] = mx;
    __syncthreads();
    if (threadIdx.x == 0) {
        double m = 0.0;
        for (uint32_t w = 0; w < (blockDim.x >> 5); ++w) m = fmax(m, red[w]);
        uint32_t p = 0;
        for (p = 0; p < 22u; ++p) {
            const int next = (int)__dadd_rn(0.5, __dmul_rn(m, (double)(1u << (p + 1u))));
            if (next >= (1 << 15)) break;
        }
        prec_s = p;
        *prec_o = p;
    }
    __syncthreads();
    const double q = (double)(1u << prec_s);
    for (uint32_t i = threadIdx.x; i < out; i += blockDim.x) {
        int64_t x0, xn;
        double center, inv;
        pd_clef_resample_range(i, in, scale, support, taps, x0, xn, center, inv);
        const double total = pd_clef_resample_total(x0, xn, center, inv);
        xmin_o[i] = (uint32_t)x0;
        xsize_o[i] = (uint32_t)xn;
        int16_t* wo = w_o + (size_t)i * taps;
        for (uint32_t j = 0; j < taps; ++j) {
            const double w = (int64_t)j < xn && total != 0.0
                                 ? __ddiv_rn(pd_clef_resample_w((int64_t)j, x0, center, inv), total)
                                 : 0.0;
            const double v = __dmul_rn(w, q);
            wo[j] = (int16_t)(v < 0.0 ? (int)__dadd_rn(-0.5, v) : (int)__dadd_rn(0.5, v));
        }
    }
}

PD_EXPORT
int pd_clef_resample_plan(uint32_t in, uint32_t out, uint32_t taps, void* xmin, void* xsize, void* w,
                          void* prec, void* stream) {
    if (in == 0 || out == 0 || taps == 0) return (int)cudaErrorInvalidValue;
    pd_pdl_go(pd_clef_resample_plan_kernel, dim3(1), dim3(256), 0, (cudaStream_t)stream, in, out, taps,
              (uint32_t*)xmin, (uint32_t*)xsize, (int16_t*)w, (uint32_t*)prec);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 736 resample pass
// One separable pass of the uint8 resize on interleaved RGB rows: `horiz`
// resamples along a row (src [rows][in][3] -> dst [rows][out][3]), else along
// the columns (src [in][cols][3] -> dst [out][cols][3]). The reference's
// integer sum: 2^(p-1) + sum of pixel * weight in int32, shifted down by p
// (an arithmetic shift - a floor), clamped to 0..255.
//
// No `const __restrict__` on the inputs: every one of them is the previous
// kernel's output, and ptxas schedules a read-only (LDG.CONSTANT) load at an
// address it knows up front ABOVE the programmatic wait (ACQBULK) - the
// plan's precision was read before the plan kernel had written it, by the
// CTAs that started under it (probed: a picture's first rows resampled with
// the previous picture's precision).
__global__ void pd_clef_resample_kernel(const uint8_t* src, uint8_t* dst, uint32_t lines, uint32_t in,
                                        uint32_t out, uint32_t horiz, const uint32_t* xmin,
                                        const uint32_t* xsize, const int16_t* w, uint32_t taps,
                                        const uint32_t* prec) {
    PD_PDL_ARM();
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t p = *prec;
    if (horiz) {
        // dst [lines][out][3]
        if (idx >= (uint64_t)lines * out * 3u) return;
        const uint32_t c = (uint32_t)(idx % 3u), o = (uint32_t)((idx / 3u) % out), y = (uint32_t)(idx / 3u / out);
        const uint8_t* s = src + ((size_t)y * in + xmin[o]) * 3u + c;
        const int16_t* wo = w + (size_t)o * taps;
        int acc = 1 << (p - 1u);
        for (uint32_t j = 0; j < xsize[o]; ++j) acc += (int)s[(size_t)j * 3u] * (int)wo[j];
        dst[idx] = (uint8_t)min(max(acc >> p, 0), 255);
    } else {
        // dst [out][lines][3], lines = the columns
        if (idx >= (uint64_t)out * lines * 3u) return;
        const uint32_t cx = (uint32_t)(idx % ((uint64_t)lines * 3u)), o = (uint32_t)(idx / ((uint64_t)lines * 3u));
        const uint8_t* s = src + (size_t)xmin[o] * lines * 3u + cx;
        const int16_t* wo = w + (size_t)o * taps;
        int acc = 1 << (p - 1u);
        for (uint32_t j = 0; j < xsize[o]; ++j) acc += (int)s[(size_t)j * lines * 3u] * (int)wo[j];
        dst[idx] = (uint8_t)min(max(acc >> p, 0), 255);
    }
}

PD_EXPORT
int pd_clef_resample(const void* src, void* dst, uint32_t lines, uint32_t in, uint32_t out, uint32_t horiz,
                     const void* xmin, const void* xsize, const void* w, uint32_t taps, const void* prec,
                     void* stream) {
    if (lines == 0 || out == 0) return 0;
    if (in == 0 || taps == 0) return (int)cudaErrorInvalidValue;
    const uint64_t n = (uint64_t)lines * out * 3u;
    if ((n + 255u) / 256u > 0x7fffffffull) return (int)cudaErrorInvalidValue;
    pd_pdl_go(pd_clef_resample_kernel, dim3((uint32_t)((n + 255u) / 256u)), dim3(256), 0, (cudaStream_t)stream,
              (const uint8_t*)src, (uint8_t*)dst, lines, in, out, horiz, (const uint32_t*)xmin,
              (const uint32_t*)xsize, (const int16_t*)w, taps, (const uint32_t*)prec);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 737 patchify
// The processor's pixel rows from the resized image (uint8 [H][W][3], H and W
// multiples of 32): row r = ((bh * gw/2 + bw) * 2 + mh) * 2 + mw covers pixel
// rows (2 bh + mh) * 16 .. and columns (2 bw + mw) * 16 ..; column
// c * 512 + t * 256 + py * 16 + px holds channel c of pixel (py, px) for both
// temporal slots t (one frame, written twice). Value (x - 127.5) / 127.5 in
// F32 - the processor's fused rescale-and-normalize. `out` starts at the
// image's first row of the pass. `img` is the resize's output: a plain
// pointer, for 736's reason.
__global__ void pd_clef_patchify_kernel(const uint8_t* img, uint32_t h, uint32_t w, float* out) {
    PD_PDL_ARM();
    const uint32_t gw = w / 16u, patches = (h / 16u) * gw;
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (uint64_t)patches * 768u) return;
    const uint32_t e = (uint32_t)(idx % 768u), r = (uint32_t)(idx / 768u);
    const uint32_t c = e / 256u, py = (e / 16u) % 16u, px = e % 16u;
    const uint32_t mw = r & 1u, mh = (r >> 1) & 1u, win = r >> 2, bw = win % (gw / 2u), bh = win / (gw / 2u);
    const uint32_t y = (2u * bh + mh) * 16u + py, x = (2u * bw + mw) * 16u + px;
    const float v = __fdiv_rn(__fsub_rn((float)img[((size_t)y * w + x) * 3u + c], 127.5f), 127.5f);
    float* o = out + (size_t)r * 1536u + c * 512u + py * 16u + px;
    o[0] = v;
    o[256] = v;
}

PD_EXPORT
int pd_clef_patchify(const void* img, uint32_t h, uint32_t w, void* out, void* stream) {
    if (h == 0 || w == 0) return 0;
    if (h % 32u || w % 32u) return (int)cudaErrorInvalidValue;
    const uint64_t n = (uint64_t)(h / 16u) * (w / 16u) * 768u;
    pd_pdl_go(pd_clef_patchify_kernel, dim3((uint32_t)((n + 255u) / 256u)), dim3(256), 0, (cudaStream_t)stream,
              (const uint8_t*)img, h, w, (float*)out);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 738 position embedding
// x[r] += the learned grid (BF16 [side * side][d], widened exactly) read at
// row r's patch (hp, wp) of its image's gh x gw patch grid: the grid point is
// linspace(0, side - 1, gh)[hp] (and the same across), formed as torch's CUDA
// linspace forms it in F32 - start + step * i on the first half, end - step *
// (n - 1 - i) (one fma) on the second - then floor, ceil (clamped) and the
// four bilinear weights (1 - fh)(1 - fw), (1 - fh) fw, fh (1 - fw), fh fw.
// Each corner's product rounds once, the four add in corner order. `info`
// is [rows][4] = (hp, wp, gh, gw).
__device__ __forceinline__ float pd_clef_linspace(float end, uint32_t n, uint32_t i) {
    if (n == 1u) return 0.f;
    const float step = __fdiv_rn(end, (float)(n - 1u));
    return i < n / 2u ? __fmul_rn(step, (float)i) : __fmaf_rn(-step, (float)(n - 1u - i), end);
}

__global__ void pd_clef_vpos_kernel(float* __restrict__ x, const __nv_bfloat16* __restrict__ table,
                                    const uint32_t* __restrict__ info, uint32_t side, uint32_t d) {
    PD_PDL_ARM();
    const uint32_t r = blockIdx.x;
    const uint32_t hp = info[4u * r], wp = info[4u * r + 1u], gh = info[4u * r + 2u], gw = info[4u * r + 3u];
    const float end = (float)(side - 1u);
    const float hg = pd_clef_linspace(end, gh, hp), wg = pd_clef_linspace(end, gw, wp);
    const uint32_t hf = (uint32_t)(int)hg, wf = (uint32_t)(int)wg;
    const uint32_t hc = min(hf + 1u, side - 1u), wc = min(wf + 1u, side - 1u);
    const float fh = __fsub_rn(hg, (float)hf), fw = __fsub_rn(wg, (float)wf);
    const float gh1 = __fsub_rn(1.f, fh), gw1 = __fsub_rn(1.f, fw);
    const float w00 = __fmul_rn(gh1, gw1), w01 = __fmul_rn(gh1, fw), w10 = __fmul_rn(fh, gw1),
                w11 = __fmul_rn(fh, fw);
    const __nv_bfloat16* e00 = table + (size_t)(hf * side + wf) * d;
    const __nv_bfloat16* e01 = table + (size_t)(hf * side + wc) * d;
    const __nv_bfloat16* e10 = table + (size_t)(hc * side + wf) * d;
    const __nv_bfloat16* e11 = table + (size_t)(hc * side + wc) * d;
    float* px = x + (size_t)r * d;
    for (uint32_t c = threadIdx.x; c < d; c += blockDim.x) {
        float s = __fmul_rn(__bfloat162float(e00[c]), w00);
        s = __fadd_rn(s, __fmul_rn(__bfloat162float(e01[c]), w01));
        s = __fadd_rn(s, __fmul_rn(__bfloat162float(e10[c]), w10));
        s = __fadd_rn(s, __fmul_rn(__bfloat162float(e11[c]), w11));
        px[c] = __fadd_rn(px[c], s);
    }
}

PD_EXPORT
int pd_clef_vpos(void* x, const void* table, const void* info, uint32_t rows, uint32_t side, uint32_t d,
                 void* stream) {
    if (rows == 0) return 0;
    if (side < 2u || d == 0) return (int)cudaErrorInvalidValue;
    pd_pdl_go(pd_clef_vpos_kernel, dim3(rows), dim3(256), 0, (cudaStream_t)stream, (float*)x,
              (const __nv_bfloat16*)table, (const uint32_t*)info, side, d);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 739 vision rope
// q and k of the fused qkv rows ([rows][3][heads][72], q at column 0, k at
// heads * 72) rotated in place: pair k of a head (dims k, k + 36) by the
// angle of row position hp (k < 18, frequency k) or column position wp
// (frequency k - 18), (cos, sin) from the table [max_pos][18], in the
// reference's order q * cos + rotate_half(q) * sin (F32, each product
// rounded). `info` as 738's.
__global__ void pd_clef_vrope_kernel(float* __restrict__ qkv, uint32_t heads, uint32_t rows,
                                     const uint32_t* __restrict__ info, const float2* __restrict__ table,
                                     uint32_t max_pos) {
    PD_PDL_ARM();
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    // per row: 2 (q, k) x heads x 36 pairs
    const uint32_t per = 2u * heads * 36u;
    if (i >= rows * per) return;
    const uint32_t row = i / per, rem = i % per, k = rem % 36u, hh = rem / 36u;  // hh: q heads, then k heads
    const uint32_t p = min(info[4u * row + (k < 18u ? 0u : 1u)], max_pos - 1u);
    const float2 cs = table[(size_t)p * 18u + (k % 18u)];
    float* h = qkv + (size_t)row * 3u * heads * 72u + hh * 72u;
    const float x1 = h[k], x2 = h[k + 36u];
    h[k] = __fadd_rn(__fmul_rn(x1, cs.x), __fmul_rn(-x2, cs.y));
    h[k + 36u] = __fadd_rn(__fmul_rn(x2, cs.x), __fmul_rn(x1, cs.y));
}

PD_EXPORT
int pd_clef_vrope(void* qkv, uint32_t heads, uint32_t rows, const void* info, const void* table, uint32_t max_pos,
                  void* stream) {
    if (rows == 0 || heads == 0) return 0;
    if (max_pos == 0) return (int)cudaErrorInvalidValue;
    const uint64_t n = (uint64_t)rows * 2u * heads * 36u;
    if (n > 0xffffffffull) return (int)cudaErrorInvalidValue;
    pd_pdl_go(pd_clef_vrope_kernel, dim3((uint32_t)((n + 255u) / 256u)), dim3(256), 0, (cudaStream_t)stream,
              (float*)qkv, heads, rows, (const uint32_t*)info, (const float2*)table, max_pos);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 740 vision attention
// Bidirectional softmax(q k^T * scale) v within each image of a pass, head_dim
// 72, q / k / v read in place from the fused qkv rows (row stride 3 * heads *
// 72), F32 class on the tensor cores: S = Q K^T and O += P V on m16n8k8 as
// 3xTF32, S drained round-nearest every 32 channels (72 = 32 + 32 + 8), O
// every BKV keys - slot 732's form at a width one warp holds whole, so a warp
// owns 16 query rows x 72 dims and no partials trade between warps. A CTA is
// BQ query rows of one head of one image (grid z) over BKV-key tiles of that
// image's K and V, staged as read; keys walk from the image's first in fixed
// tiles whatever tile holds a row - deterministic, and an image's bits do not
// depend on the images beside it.
#define PD_CLEF_V_HD 72u
#define PD_CLEF_V_SK (PD_CLEF_V_HD + 4u)  // a 76-float pitch: the fragment reads land on 32 banks

template <uint32_t BQ, uint32_t BKV, uint32_t TERMS = 3u>
__global__ void __launch_bounds__(BQ / 16u * 32u) pd_clef_vattn_kernel(const float* __restrict__ qkv,
                                                                       float* __restrict__ out,
                                                                       const uint32_t* __restrict__ cu,
                                                                       uint32_t heads, float scale) {
    constexpr uint32_t HD = PD_CLEF_V_HD, SK = PD_CLEF_V_SK, NTK = BKV / 8u, KT8 = HD / 8u, NTD = HD / 8u,
                       NTH = BQ / 16u * 32u, CH = HD / 4u;
    static_assert(BKV % 8u == 0 && BQ % 16u == 0, "whole m16 / n8 tiles");
    extern __shared__ __align__(16) float clef_v_sm[];
    float* Qs = clef_v_sm;      // [BQ][SK]
    float* Ks = Qs + BQ * SK;   // [BKV][SK]
    float* Vs = Ks + BKV * SK;  // [BKV][SK]
    PD_PDL_ARM();
    const uint32_t seg0 = cu[blockIdx.z], len = cu[blockIdx.z + 1u] - seg0;
    const uint32_t q0 = blockIdx.x * BQ;
    if (q0 >= len) return;
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5, gr = lane >> 2, t4 = lane & 3u;
    const uint32_t head = blockIdx.y, ld = 3u * heads * HD;
    const uint32_t count = min(BQ, len - q0);
    const uint32_t r0 = warp * 16u + gr;  // this lane's rows r0 and r0 + 8 of the tile
    const float* qb = qkv + (size_t)seg0 * ld + head * HD;
    const float* kb = qb + heads * HD;
    const float* vb = kb + heads * HD;
    auto stage = [&](float* dst, const float* src, uint32_t base) {
        for (uint32_t f = tid; f < BKV * CH; f += NTH) {
            const uint32_t kj = f / CH, d4 = (f % CH) * 4u;
            const bool ok = base + kj < len;
            pd_clef_cpa16(&dst[kj * SK + d4], ok ? src + (size_t)(base + kj) * ld + d4 : src, ok);
        }
    };
    for (uint32_t f = tid; f < BQ * CH; f += NTH) {
        const uint32_t r = f / CH, d4 = (f % CH) * 4u;
        const bool ok = r < count;
        pd_clef_cpa16(&Qs[r * SK + d4], ok ? qb + (size_t)(q0 + r) * ld + d4 : qb, ok);
    }
    stage(Ks, kb, 0);
    asm volatile("cp.async.commit_group;" ::: "memory");
    stage(Vs, vb, 0);
    asm volatile("cp.async.commit_group;" ::: "memory");
    asm volatile("cp.async.wait_group 1;" ::: "memory");
    __syncthreads();
    float o[NTD][4], m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
#pragma unroll
    for (uint32_t i = 0; i < NTD; ++i)
#pragma unroll
        for (uint32_t e = 0; e < 4u; ++e) o[i][e] = 0.f;
    for (uint32_t base = 0; base < len; base += BKV) {
        const bool next = base + BKV < len;
        float s[NTK][4], acc[NTK][4];
#pragma unroll
        for (uint32_t nt = 0; nt < NTK; ++nt)
#pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) s[nt][e] = acc[nt][e] = 0.f;
#pragma unroll
        for (uint32_t k8 = 0; k8 < KT8; ++k8) {
            const float* qr = Qs + r0 * SK + k8 * 8u + t4;
            const float a[4] = {qr[0], qr[8u * SK], qr[4u], qr[8u * SK + 4u]};
            uint32_t qh[4], ql[4];
#pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) {
                qh[e] = pd_kumo_tf32(a[e]);
                ql[e] = pd_kumo_tf32(a[e] - __uint_as_float(qh[e]));
            }
#pragma unroll
            for (uint32_t nt = 0; nt < NTK; ++nt) {
                const float* br = Ks + (nt * 8u + gr) * SK + k8 * 8u + t4;
                const float b0 = br[0], b1 = br[4u];
                const uint32_t bb0 = pd_kumo_tf32(b0), bb1 = pd_kumo_tf32(b1);
                const uint32_t bs0 = pd_kumo_tf32(b0 - __uint_as_float(bb0));
                const uint32_t bs1 = pd_kumo_tf32(b1 - __uint_as_float(bb1));
                pd_kumo_mma(acc[nt], qh, bb0, bb1);
                if (TERMS == 3u) {
                    pd_kumo_mma(acc[nt], qh, bs0, bs1);
                    pd_kumo_mma(acc[nt], ql, bb0, bb1);
                }
            }
            if (k8 % 4u == 3u || k8 + 1u == KT8) {
#pragma unroll
                for (uint32_t nt = 0; nt < NTK; ++nt)
#pragma unroll
                    for (uint32_t e = 0; e < 4u; ++e) {
                        s[nt][e] = __fadd_rn(s[nt][e], acc[nt][e]);
                        acc[nt][e] = 0.f;
                    }
            }
        }
        __syncthreads();  // every warp is done with this K tile
        if (next) stage(Ks, kb, base + BKV);
        asm volatile("cp.async.commit_group;" ::: "memory");
        // the online softmax, rows r0 (h = 0) and r0 + 8 (h = 1): a lane holds
        // keys nt * 8 + 2 t4 + {0, 1}; a row's four lanes reduce by xor 1, 2
        float corr[2];
#pragma unroll
        for (uint32_t h = 0; h < 2u; ++h) {
            float hi = m[h];
#pragma unroll
            for (uint32_t nt = 0; nt < NTK; ++nt)
#pragma unroll
                for (uint32_t c = 0; c < 2u; ++c) {
                    float& z = s[nt][2u * h + c];
                    z = base + nt * 8u + 2u * t4 + c < len ? __fmul_rn(z, scale) : -INFINITY;
                    hi = fmaxf(hi, z);
                }
            hi = fmaxf(hi, __shfl_xor_sync(0xffffffffu, hi, 1));
            hi = fmaxf(hi, __shfl_xor_sync(0xffffffffu, hi, 2));
            corr[h] = m[h] == -INFINITY ? 0.f : expf(__fsub_rn(m[h], hi));
            float sum = 0.f;
#pragma unroll
            for (uint32_t nt = 0; nt < NTK; ++nt)
#pragma unroll
                for (uint32_t c = 0; c < 2u; ++c) {
                    float& z = s[nt][2u * h + c];
                    z = z == -INFINITY ? 0.f : expf(__fsub_rn(z, hi));
                    sum = __fadd_rn(sum, z);
                }
            sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, 1));
            sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, 2));
            l[h] = fmaf(l[h], corr[h], sum);
            m[h] = hi;
        }
        asm volatile("cp.async.wait_group 1;" ::: "memory");  // this tile's V
        __syncthreads();
        // O = O * corr + P V, P from S's accumulator (within an 8-key block
        // slot t4 <- key 2 t4, slot t4 + 4 <- key 2 t4 + 1, V read with the
        // same permutation), one drain a tile
        float pv[NTD][4];
#pragma unroll
        for (uint32_t nd = 0; nd < NTD; ++nd)
#pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) pv[nd][e] = 0.f;
#pragma unroll
        for (uint32_t kt = 0; kt < NTK; ++kt) {
            const float a[4] = {s[kt][0], s[kt][2], s[kt][1], s[kt][3]};
            uint32_t pb[4], ps[4];
#pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) {
                pb[e] = pd_kumo_tf32(a[e]);
                ps[e] = pd_kumo_tf32(a[e] - __uint_as_float(pb[e]));
            }
            const float* v0 = Vs + (kt * 8u + 2u * t4) * SK + gr;
#pragma unroll
            for (uint32_t nd = 0; nd < NTD; ++nd) {
                const float b0 = v0[nd * 8u], b1 = v0[SK + nd * 8u];
                const uint32_t bb0 = pd_kumo_tf32(b0), bb1 = pd_kumo_tf32(b1);
                const uint32_t bs0 = pd_kumo_tf32(b0 - __uint_as_float(bb0));
                const uint32_t bs1 = pd_kumo_tf32(b1 - __uint_as_float(bb1));
                pd_kumo_mma(pv[nd], pb, bb0, bb1);
                if (TERMS == 3u) {
                    pd_kumo_mma(pv[nd], pb, bs0, bs1);
                    pd_kumo_mma(pv[nd], ps, bb0, bb1);
                }
            }
        }
#pragma unroll
        for (uint32_t nd = 0; nd < NTD; ++nd)
#pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) o[nd][e] = __fadd_rn(__fmul_rn(o[nd][e], corr[e >> 1]), pv[nd][e]);
        __syncthreads();  // every warp is done with this V tile
        if (next) stage(Vs, vb, base + BKV);
        asm volatile("cp.async.commit_group;" ::: "memory");
        asm volatile("cp.async.wait_group 1;" ::: "memory");  // the next tile's K
        __syncthreads();
    }
#pragma unroll
    for (uint32_t h = 0; h < 2u; ++h) {
        const uint32_t r = r0 + 8u * h;
        if (r >= count) continue;
        float* po = out + (size_t)(seg0 + q0 + r) * heads * HD + head * HD + 2u * t4;
#pragma unroll
        for (uint32_t nd = 0; nd < NTD; ++nd)
            *reinterpret_cast<float2*>(po + nd * 8u) =
                make_float2(__fdiv_rn(o[nd][2u * h], l[h]), __fdiv_rn(o[nd][2u * h + 1u], l[h]));
    }
}

template <uint32_t BQ, uint32_t BKV, uint32_t TERMS = 3u>
static void pd_clef_vattn_go(const void* qkv, void* out, const void* cu, uint32_t segs, uint32_t max_len,
                             uint32_t heads, float scale, cudaStream_t s) {
    constexpr uint32_t smem = (BQ + 2u * BKV) * PD_CLEF_V_SK * 4u;
    static bool set = false;
    if (!set) {
        cudaFuncSetAttribute(pd_clef_vattn_kernel<BQ, BKV, TERMS>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem);
        set = true;
    }
    pd_pdl_go(pd_clef_vattn_kernel<BQ, BKV, TERMS>, dim3((max_len + BQ - 1u) / BQ, heads, segs), dim3(BQ / 16u * 32u),
              smem, s, (const float*)qkv, (float*)out, (const uint32_t*)cu, heads, scale);
}

PD_EXPORT
int pd_clef_vattn(const void* qkv, void* out, const void* cu, uint32_t segs, uint32_t max_len, uint32_t heads,
                  float scale, void* stream) {
    if (segs == 0 || max_len == 0) return 0;
    if (heads == 0 || segs > 65535u || heads > 65535u || (((uintptr_t)qkv | (uintptr_t)out) & 15u))
        return (int)cudaErrorInvalidValue;
    pd_clef_vattn_go<64u, 32u>(qkv, out, cu, segs, max_len, heads, scale, (cudaStream_t)stream);
    return pd_launch_status();
}
