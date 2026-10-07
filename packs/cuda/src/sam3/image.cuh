// sam3/image.cuh - SAM 3's picture in and masks out: torchvision's antialiased resize, the mask upsample + threshold, COCO RLE
// Textually-included segment of the single pack translation unit.
// Not standalone-compilable: include order is defined by ../pack.cu.
// ------------------------------------------------------------ sam3 image I/O
// A request picture arrives decoded (libjpeg-turbo's bytes, paddock_jpeg) at
// its own size; the model reads 1008 x 1008. Meta's processor resizes with
// torchvision's v2.Resize on the CUDA tensor, which for uint8 is: cast to f32,
// F.interpolate(bilinear, antialias=True, align_corners=False), round half to
// even, back to uint8. 770 is that, to the bit: torch's antialias weights
// (Pillow's triangle filter, support scaled by the downscale factor,
// normalized by division), its horizontal-then-vertical walk over the span,
// and the exact pattern of fused and rounded steps its compiled kernel has
// (see pd_sam3_aa_span).
//
// On the way out, each kept query's 288^2 mask logits go back to the picture
// with torch's plain bilinear (align_corners=False, no antialias), sigmoid,
// > 0.5 - Meta's processor - written column-major so the run-length encoder
// that follows reads it contiguously. COCO RLE: runs of the column-major
// sequence, starting with a (possibly empty) run of zeros.
//
// Needs f32_qkv.cuh's pd_launch_status.

#define PD_SAM3_AA_MAXW 96u  // weights in a span: support <= 47 px, a 47x downscale

// One output index's span and normalized weights, in the arithmetic torch's
// COMPILED kernel performs - not the arithmetic its source reads as. nvcc
// (fmad on, PyTorch's default) fuses three spots of torch's
// _compute_weights_span / _compute_weights / interpolate_aa_single_dim and
// leaves the rest rounded step by step; written out here with fmaf and the
// _rn intrinsics so this kernel's own compile cannot choose differently.
// Found by emulating every combination against torch's float output on three
// pictures (harness probe_resize_variants.py): exactly one gives zero
// mismatches -
//   fused:   center -/+ support (the span bounds), xmin - center (the offset
//            every weight is measured from), and each accumulation step;
//   rounded: scale * (i + 0.5) where it stands alone, the triangle filter's
//            1 - |x|, the weight sum and its division, the first product.
__device__ __forceinline__ void pd_sam3_aa_span(uint32_t i, uint32_t in, float scale, float* w,
                                                int& xmin, int& xsize) {
    const float support = scale >= 1.0f ? scale : 1.0f;
    const float ih = __fadd_rn((float)i, 0.5f);
    const float lo = fmaf(scale, ih, -support), hi = fmaf(scale, ih, support);
    xmin = max((int)__fadd_rn(lo, 0.5f), 0);
    xsize = min((int)__fadd_rn(hi, 0.5f), (int)in) - xmin;
    // 1.0 / scale in double, as torch's `1.0 / scale` is, then to float
    const float invscale = scale >= 1.0f ? (float)(1.0 / (double)scale) : 1.0f;
    const float xmin_m_center = fmaf(-scale, ih, (float)xmin);
    float total = 0.0f;
    for (int j = 0; j < xsize; ++j) {
        float x = __fmul_rn(__fadd_rn(__fadd_rn((float)j, xmin_m_center), 0.5f), invscale);
        x = x < 0.0f ? -x : x;
        const float v = x < 1.0f ? __fsub_rn(1.0f, x) : 0.0f;
        w[j] = v;
        total = __fadd_rn(total, v);
    }
    if (total != 0.0f)
        for (int j = 0; j < xsize; ++j) w[j] = __fdiv_rn(w[j], total);
}

// 770: u8 HWC [H][W][C] -> u8 HWC [OH][OW][C], one thread an output pixel.
__global__ void pd_sam3_resize_aa_u8_kernel(const uint8_t* __restrict__ src,
                                            uint8_t* __restrict__ dst, uint32_t H, uint32_t W,
                                            uint32_t OH, uint32_t OW, uint32_t C) {
    const uint32_t ox = blockIdx.x * blockDim.x + threadIdx.x, oy = blockIdx.y;
    if (ox >= OW) return;
    const float sw = (float)W / (float)OW, sh = (float)H / (float)OH;
    float wx[PD_SAM3_AA_MAXW], wy[PD_SAM3_AA_MAXW], buf[PD_SAM3_AA_MAXW];
    int xmin, xsize, ymin, ysize;
    pd_sam3_aa_span(ox, W, sw, wx, xmin, xsize);
    pd_sam3_aa_span(oy, H, sh, wy, ymin, ysize);
    for (uint32_t c = 0; c < C; ++c) {
        for (int y = 0; y < ysize; ++y) {
            const uint8_t* row = src + ((size_t)(ymin + y) * W + xmin) * C + c;
            float out = __fmul_rn((float)row[0], wx[0]);
            for (int x = 1; x < xsize; ++x) out = fmaf((float)row[(size_t)x * C], wx[x], out);
            buf[y] = out;
        }
        float out = __fmul_rn(buf[0], wy[0]);
        for (int y = 1; y < ysize; ++y) out = fmaf(buf[y], wy[y], out);
        const float r = rintf(out);
        dst[((size_t)oy * OW + ox) * C + c] = (uint8_t)(r < 0.0f ? 0.0f : (r > 255.0f ? 255.0f : r));
    }
}

PD_EXPORT
int pd_sam3_resize_aa_u8(const void* src, void* dst, uint32_t H, uint32_t W, uint32_t OH,
                         uint32_t OW, uint32_t C, void* stream) {
    if (H == 0 || W == 0 || OH == 0 || OW == 0 || C == 0) return 0;
    // the span a thread holds: ceil(support) * 2 + 1 for the larger downscale
    const float s = fmaxf((float)W / (float)OW, (float)H / (float)OH);
    if ((uint32_t)ceilf(s) * 2u + 1u > PD_SAM3_AA_MAXW) return cudaErrorInvalidValue;
    dim3 grid((OW + 127u) / 128u, OH);
    pd_sam3_resize_aa_u8_kernel<<<grid, 128, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)src, (uint8_t*)dst, H, W, OH, OW, C);
    return pd_launch_status();
}

// 771: one kept query's mask, back at the picture. `logits` is the detector's
// pixel-major landing [side*side][nq], query `q`; out u8 0/1 COLUMN-major
// [W][H] (index x * H + y). torch's upsample_bilinear2d term for term:
// src = max(scale * (dst + 0.5) - 0.5, 0), i1 = i0 + (i0 < n - 1),
// h0 (w0 a + w1 b) + h1 (w0 c + w1 d); then sigmoid(v) > 0.5 in f32, which is
// what Meta thresholds (not v > 0: a tiny positive logit rounds to exactly 0.5).
__global__ void pd_sam3_mask_up_kernel(const float* __restrict__ logits, uint8_t* __restrict__ out,
                                       uint32_t side, uint32_t nq, uint32_t q, uint32_t H,
                                       uint32_t W) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= H * W) return;
    const uint32_t x = i / H, y = i - x * H;  // column-major walk
    const float rh = (float)side / (float)H, rw = (float)side / (float)W;
    float sy = rh * ((float)y + 0.5f) - 0.5f, sx = rw * ((float)x + 0.5f) - 0.5f;
    sy = sy < 0.0f ? 0.0f : sy;
    sx = sx < 0.0f ? 0.0f : sx;
    const uint32_t y0 = (uint32_t)sy, x0 = (uint32_t)sx;
    const uint32_t y1 = y0 + (y0 < side - 1u ? 1u : 0u), x1 = x0 + (x0 < side - 1u ? 1u : 0u);
    const float l1y = sy - (float)y0, l0y = 1.0f - l1y, l1x = sx - (float)x0, l0x = 1.0f - l1x;
    const float a = logits[((size_t)y0 * side + x0) * nq + q];
    const float b = logits[((size_t)y0 * side + x1) * nq + q];
    const float c = logits[((size_t)y1 * side + x0) * nq + q];
    const float d = logits[((size_t)y1 * side + x1) * nq + q];
    const float v = l0y * (l0x * a + l1x * b) + l1y * (l0x * c + l1x * d);
    const float p = 1.0f / (1.0f + expf(-v));
    out[i] = p > 0.5f ? 1u : 0u;
}

PD_EXPORT
int pd_sam3_mask_up(const void* logits, void* out, uint32_t side, uint32_t nq, uint32_t q,
                    uint32_t H, uint32_t W, void* stream) {
    if (H == 0 || W == 0) return 0;
    if (side == 0 || q >= nq) return cudaErrorInvalidValue;
    const uint64_t n = (uint64_t)H * W;
    pd_sam3_mask_up_kernel<<<(uint32_t)((n + 255ull) / 256ull), 256, 0, (cudaStream_t)stream>>>(
        (const float*)logits, (uint8_t*)out, side, nq, q, H, W);
    return pd_launch_status();
}

// 772: COCO RLE of one 0/1 mask already in column-major order: counts[0] the
// leading zeros (0 when the first pixel is set), then alternating run
// lengths; `nruns[0]` the count written. A run starts at a pixel that differs
// from the one before (the sequence starts after an implied 0); run lengths
// are the starts' differences. One block of 1024 walks the mask in rounds of
// 64 KB, 64 bytes a thread a round as four 16 B loads: count the starts in
// them, a block scan places each thread's, write them, carry the round's
// total. Fixed order, no atomics. If the mask needs more than `cap` counts,
// nothing is written and nruns[0] = 0xffffffff.
//
// The first form gave each thread one contiguous 1/1024 of the mask, which
// is ~400 bytes a thread - 32 lanes then read 32 different lines every step,
// and it ran ~250 us a 534 x 800 mask on the A6000.
#define PD_SAM3_RLE_T 1024u
#define PD_SAM3_RLE_V 64u  // bytes a thread a round
__global__ void __launch_bounds__(PD_SAM3_RLE_T) pd_sam3_rle_kernel(
    const uint8_t* __restrict__ m, uint32_t* __restrict__ starts, uint32_t* __restrict__ counts,
    uint32_t* __restrict__ nruns, uint64_t n, uint32_t cap) {
    __shared__ uint32_t wsum[PD_SAM3_RLE_T / 32u];
    __shared__ uint32_t s_round;
    const uint32_t t = threadIdx.x, lane = t & 31u, warp = t >> 5;
    constexpr uint64_t R = (uint64_t)PD_SAM3_RLE_T * PD_SAM3_RLE_V;
    uint32_t carried = 0;  // starts before this round, the same in every thread
    for (uint64_t r0 = 0; r0 < n; r0 += R) {
        const uint64_t a = r0 + (uint64_t)t * PD_SAM3_RLE_V;
        // the thread's 64 bytes as 16 words (byte u = word u / 4, lane u % 4)
        uint32_t wd[PD_SAM3_RLE_V / 4u];
        if (a + PD_SAM3_RLE_V <= n) {
            // the mask plane is a pooled allocation and a is a multiple of 64
#pragma unroll
            for (uint32_t u = 0; u < PD_SAM3_RLE_V / 16u; ++u) {
                const uint4 x = *reinterpret_cast<const uint4*>(m + a + 16u * u);
                wd[4u * u] = x.x;
                wd[4u * u + 1u] = x.y;
                wd[4u * u + 2u] = x.z;
                wd[4u * u + 3u] = x.w;
            }
        } else {
#pragma unroll
            for (uint32_t u = 0; u < PD_SAM3_RLE_V / 4u; ++u) {
                uint32_t x = 0;
                for (uint32_t z = 0; z < 4u; ++z)
                    if (a + 4u * u + z < n) x |= (uint32_t)m[a + 4u * u + z] << (8u * z);
                wd[u] = x;
            }
        }
#define PD_SAM3_RLE_BYTE(u) ((uint8_t)(wd[(u) >> 2] >> (((u) & 3u) * 8u)))
        const uint64_t e = a < n ? (a + PD_SAM3_RLE_V < n ? a + PD_SAM3_RLE_V : n) - a : 0u;
        uint8_t prev = a == 0 || a > n ? 0u : m[a - 1u];
        uint32_t c = 0;
#pragma unroll
        for (uint32_t u = 0; u < PD_SAM3_RLE_V; ++u) {
            const uint8_t x = PD_SAM3_RLE_BYTE(u);
            c += (u < e && x != prev) ? 1u : 0u;
            prev = x;
        }
        // block exclusive scan: shuffles within a warp, then the warp totals
        uint32_t inc = c;
#pragma unroll
        for (uint32_t o = 1; o < 32u; o <<= 1) {
            const uint32_t y = __shfl_up_sync(0xffffffffu, inc, o);
            if (lane >= o) inc += y;
        }
        if (lane == 31u) wsum[warp] = inc;
        __syncthreads();
        if (warp == 0) {
            uint32_t w = wsum[lane];
            const uint32_t own = w;
#pragma unroll
            for (uint32_t o = 1; o < 32u; o <<= 1) {
                const uint32_t y = __shfl_up_sync(0xffffffffu, w, o);
                if (lane >= o) w += y;
            }
            wsum[lane] = w - own;  // exclusive warp offsets
            if (lane == 31u) s_round = w;
        }
        __syncthreads();
        uint32_t o = carried + wsum[warp] + inc - c;
        prev = a == 0 || a > n ? 0u : m[a - 1u];
#pragma unroll
        for (uint32_t u = 0; u < PD_SAM3_RLE_V; ++u) {
            const uint8_t x = PD_SAM3_RLE_BYTE(u);
            if (u < e && x != prev) {
                if (o < cap) starts[o] = (uint32_t)(a + u);
                ++o;
            }
            prev = x;
        }
#undef PD_SAM3_RLE_BYTE
        carried += s_round;
        __syncthreads();  // wsum and s_round are rewritten next round
    }
    const uint32_t total = carried;
    // starts + the closing boundary at n: total + 1 counts
    if (total + 1u > cap) {
        if (t == 0) nruns[0] = 0xffffffffu;
        return;
    }
    for (uint32_t i = t; i <= total; i += PD_SAM3_RLE_T) {
        const uint32_t hi = i < total ? starts[i] : (uint32_t)n;
        const uint32_t lo = i == 0 ? 0u : starts[i - 1u];
        counts[i] = hi - lo;
    }
    if (t == 0) nruns[0] = total + 1u;
}

PD_EXPORT
int pd_sam3_rle(const void* mask, void* starts, void* counts, void* nruns, uint64_t n,
                uint32_t cap, void* stream) {
    if (n == 0 || cap == 0) return cudaErrorInvalidValue;
    if (n > 0xffffffffull) return cudaErrorInvalidValue;  // counts are u32
    pd_sam3_rle_kernel<<<1, PD_SAM3_RLE_T, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)mask, (uint32_t*)starts, (uint32_t*)counts, (uint32_t*)nruns, n, cap);
    return pd_launch_status();
}
