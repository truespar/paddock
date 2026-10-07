// sam3/video.cuh - SAM 3's video path around the tracker: frames in the way Meta's loader makes them (Pillow's bilinear), mask bitplanes and their pairwise overlaps, the hole and sprinkle fill, the resamplings and non-overlap of the birth and memory chains, the output masks at the video's size
// Textually-included segment of the single pack translation unit.
// Not standalone-compilable: include order is defined by ../pack.cu.
// ------------------------------------------------------------ sam3 video masks
// Every decision Meta's video loop makes per frame (sam3_video_base.py) reads
// the 288^2 low-res masks of the detections and of the tracked objects: mask
// NMS among the detections, the detection-to-track association, the
// occlusion-based suppression between tracks, hole filling, and the outputs
// at the video's own size. The decisions themselves are a handful of small
// numbers and run on the host; what touches pixels is here.
//
// Planes are f32 logits, one object a plane (`stride` floats apart); a mask
// is logit > 0, as Meta binarizes everywhere in this loop.
//
// Needs f32_qkv.cuh's pd_launch_status and tracker.cuh's union-find
// (pd_sam3_cc_find / pd_sam3_cc_union / PD_SAM3_CC_NONE).

// 792: masks as bitplanes. Plane j of `planes` (px values at j * stride) ->
// bits[j][words], bit i of word w set when pixel 32 w + i is > 0, and
// area[j] the plane's count (the launcher zeroes it first). A warp a word: 32
// lanes read 32 consecutive pixels (one 128-byte line) and ballot them.
__global__ void pd_sam3_mask_bits_kernel(const float* __restrict__ planes,
                                         uint32_t* __restrict__ bits, uint32_t* __restrict__ area,
                                         uint64_t stride, uint32_t px, uint32_t words) {
    const uint32_t j = blockIdx.y, lane = threadIdx.x & 31u;
    const uint32_t w = blockIdx.x * (blockDim.x / 32u) + threadIdx.x / 32u;
    if (w >= words) return;
    const uint32_t p = w * 32u + lane;
    const bool on = p < px && planes[(size_t)j * stride + p] > 0.0f;
    const uint32_t word = __ballot_sync(0xffffffffu, on);
    if (lane == 0) {
        bits[(size_t)j * words + w] = word;
        if (word != 0u) atomicAdd(&area[j], (uint32_t)__popc(word));
    }
}

__global__ void pd_sam3_zero_u32_kernel(uint32_t* __restrict__ p, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = 0u;
}

PD_EXPORT
int pd_sam3_mask_bits(const void* planes, void* bits, void* area, uint64_t stride, uint32_t n,
                      uint32_t px, void* stream) {
    if (n == 0u || px == 0u) return 0;
    if (n > 65535u) return cudaErrorInvalidValue;
    const cudaStream_t st = (cudaStream_t)stream;
    const uint32_t words = (px + 31u) / 32u;
    pd_sam3_zero_u32_kernel<<<(n + 255u) / 256u, 256, 0, st>>>((uint32_t*)area, n);
    const dim3 grid((words + 7u) / 8u, n);
    pd_sam3_mask_bits_kernel<<<grid, 256, 0, st>>>((const float*)planes, (uint32_t*)bits,
                                                    (uint32_t*)area, stride, px, words);
    return pd_launch_status();
}

// 793: the pairwise intersections of two bitplane sets, inter[i][j] =
// |a_i & b_j|. A warp a pair, popcounts over the words and a warp sum: the
// counts are exact integers, so the IoU the host forms from them is Meta's
// float matmul's (exact below 2^24 pixels) to the bit.
__global__ void pd_sam3_mask_pairs_kernel(const uint32_t* __restrict__ a,
                                          const uint32_t* __restrict__ b,
                                          uint32_t* __restrict__ inter, uint32_t words, uint32_t na,
                                          uint32_t nb) {
    const uint32_t pair = blockIdx.x * (blockDim.x / 32u) + threadIdx.x / 32u;
    const uint32_t lane = threadIdx.x & 31u;
    if (pair >= na * nb) return;
    const uint32_t i = pair / nb, j = pair - i * nb;
    const uint32_t* ai = a + (size_t)i * words;
    const uint32_t* bj = b + (size_t)j * words;
    uint32_t c = 0;
    for (uint32_t w = lane; w < words; w += 32u) c += (uint32_t)__popc(ai[w] & bj[w]);
#pragma unroll
    for (uint32_t o = 16; o > 0; o >>= 1) c += __shfl_xor_sync(0xffffffffu, c, o);
    if (lane == 0) inter[pair] = c;
}

PD_EXPORT
int pd_sam3_mask_pairs(const void* a, const void* b, void* inter, uint32_t words, uint32_t na,
                       uint32_t nb, void* stream) {
    if (na == 0u || nb == 0u) return 0;
    if ((uint64_t)na * nb > 0x7fffffffull / 32ull) return cudaErrorInvalidValue;
    const uint32_t pairs = na * nb;
    pd_sam3_mask_pairs_kernel<<<(pairs + 7u) / 8u, 256, 0, (cudaStream_t)stream>>>(
        (const uint32_t*)a, (const uint32_t*)b, (uint32_t*)inter, words, na, nb);
    return pd_launch_status();
}

// 794: Meta's fill_holes_in_mask_scores, a plane a blockIdx.y. First every
// 8-connected BACKGROUND component (<= 0) of at most max_area pixels takes
// hole_val; then, on the filled plane, every foreground component of at most
// min(foreground / 2, max_area) pixels takes sprinkle_val - Meta keeps a tiny
// object whole when it is all there is. The labelling is tracker.cuh's
// union-find (sizes do not depend on the order the unions ran in), over
// u32 scratch lab / area of n x side^2 and tot of n.
__global__ void pd_sam3_clean_init_kernel(const float* __restrict__ planes,
                                          uint32_t* __restrict__ lab, uint32_t* __restrict__ area,
                                          uint32_t* __restrict__ tot, uint64_t stride, uint32_t px,
                                          uint32_t fg) {
    const uint32_t j = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool in = i < px;
    const float v = in ? planes[(size_t)j * stride + i] : 0.0f;
    const bool sel = in && (fg ? v > 0.0f : v <= 0.0f);
    if (in) {
        lab[(size_t)j * px + i] = sel ? i : PD_SAM3_CC_NONE;
        area[(size_t)j * px + i] = 0u;
    }
    if (fg) {
        const uint32_t c = (uint32_t)__popc(__ballot_sync(0xffffffffu, sel));
        if ((threadIdx.x & 31u) == 0u && c != 0u) atomicAdd(&tot[j], c);
    }
}
__global__ void pd_sam3_clean_merge_kernel(uint32_t* __restrict__ labs, uint32_t side) {
    const uint32_t j = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t px = side * side;
    uint32_t* lab = labs + (size_t)j * px;
    if (i >= px || lab[i] == PD_SAM3_CC_NONE) return;
    const uint32_t y = i / side, x = i - y * side;
    // the four neighbours before this pixel in raster order cover all eight;
    // unselected pixels stay NONE for the whole pass
    if (x > 0 && __ldcg(&lab[i - 1u]) != PD_SAM3_CC_NONE) pd_sam3_cc_union(lab, i, i - 1u);
    if (y > 0) {
        const uint32_t up = i - side;
        if (__ldcg(&lab[up]) != PD_SAM3_CC_NONE) pd_sam3_cc_union(lab, i, up);
        if (x > 0 && __ldcg(&lab[up - 1u]) != PD_SAM3_CC_NONE) pd_sam3_cc_union(lab, i, up - 1u);
        if (x + 1u < side && __ldcg(&lab[up + 1u]) != PD_SAM3_CC_NONE)
            pd_sam3_cc_union(lab, i, up + 1u);
    }
}
__global__ void pd_sam3_clean_count_kernel(uint32_t* __restrict__ labs,
                                           uint32_t* __restrict__ areas, uint32_t px) {
    const uint32_t j = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t* lab = labs + (size_t)j * px;
    if (i >= px || lab[i] == PD_SAM3_CC_NONE) return;
    atomicAdd(&areas[(size_t)j * px + pd_sam3_cc_find(lab, i)], 1u);
}
__global__ void pd_sam3_clean_fill_kernel(float* __restrict__ planes, uint32_t* __restrict__ labs,
                                          const uint32_t* __restrict__ areas,
                                          const uint32_t* __restrict__ tot, uint64_t stride,
                                          uint32_t px, uint32_t max_area, float val, uint32_t fg) {
    const uint32_t j = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t* lab = labs + (size_t)j * px;
    if (i >= px || lab[i] == PD_SAM3_CC_NONE) return;
    uint32_t lim = max_area;
    if (fg) {
        const uint32_t half = tot[j] / 2u;
        lim = half < lim ? half : lim;
    }
    if (areas[(size_t)j * px + pd_sam3_cc_find(lab, i)] <= lim) {
        planes[(size_t)j * stride + i] = val;
    }
}

PD_EXPORT
int pd_sam3_mask_clean(void* planes, void* lab, void* area, void* tot, uint64_t stride,
                       uint32_t side, uint32_t n, uint32_t max_area, float hole_val,
                       float sprinkle_val, void* stream) {
    if (n == 0u || side == 0u || max_area == 0u) return 0;
    if (n > 65535u || side > 46340u) return cudaErrorInvalidValue;
    const cudaStream_t st = (cudaStream_t)stream;
    const uint32_t px = side * side;
    const dim3 grid((px + 255u) / 256u, n);
    pd_sam3_zero_u32_kernel<<<(n + 255u) / 256u, 256, 0, st>>>((uint32_t*)tot, n);
    for (uint32_t fg = 0; fg < 2u; ++fg) {
        pd_sam3_clean_init_kernel<<<grid, 256, 0, st>>>((const float*)planes, (uint32_t*)lab,
                                                         (uint32_t*)area, (uint32_t*)tot, stride,
                                                         px, fg);
        pd_sam3_clean_merge_kernel<<<grid, 256, 0, st>>>((uint32_t*)lab, side);
        pd_sam3_clean_count_kernel<<<grid, 256, 0, st>>>((uint32_t*)lab, (uint32_t*)area, px);
        pd_sam3_clean_fill_kernel<<<grid, 256, 0, st>>>(
            (float*)planes, (uint32_t*)lab, (const uint32_t*)area, (const uint32_t*)tot, stride,
            px, max_area, fg ? sprinkle_val : hole_val, fg);
    }
    return pd_launch_status();
}

// 795: one plane of n values set to `val` (mode 0, Meta's
// masks[suppressed] = -10) or clamped to at most `val` (mode 1, its
// torch.clamp(max=-10) of a shrunk mask).
__global__ void pd_sam3_mask_set_kernel(float* __restrict__ p, uint64_t n, float val,
                                        uint32_t clamp) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    p[i] = clamp ? fminf(p[i], val) : val;
}

PD_EXPORT
int pd_sam3_mask_set(void* plane, uint64_t n, float val, uint32_t mode, void* stream) {
    if (n == 0u) return 0;
    if (mode > 1u || n > 0xffffffffull * 256ull) return cudaErrorInvalidValue;
    pd_sam3_mask_set_kernel<<<(uint32_t)((n + 255u) / 256u), 256, 0, (cudaStream_t)stream>>>(
        (float*)plane, n, val, mode);
    return pd_launch_status();
}

// 796: one plane's logits back at the video's size, u8 0/1 COLUMN-major
// [W][H] (what the RLE walks). The interpolation is 771's - torch's bilinear
// with align_corners=False - and only the threshold differs: Meta's video
// outputs keep logit > 0 (mode 1), its picture processor sigmoid > 0.5
// (mode 0, 771's). `logits` pixel-major [side^2][nq], column q.
__global__ void pd_sam3_mask_up2_kernel(const float* __restrict__ logits, uint8_t* __restrict__ out,
                                        uint32_t side, uint32_t nq, uint32_t q, uint32_t H,
                                        uint32_t W, uint32_t mode) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= H * W) return;
    const uint32_t x = i / H, y = i - x * H;
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
    out[i] = mode ? (v > 0.0f ? 1u : 0u) : (1.0f / (1.0f + expf(-v)) > 0.5f ? 1u : 0u);
}

PD_EXPORT
int pd_sam3_mask_up2(const void* logits, void* out, uint32_t side, uint32_t nq, uint32_t q,
                     uint32_t H, uint32_t W, uint32_t mode, void* stream) {
    if (H == 0 || W == 0) return 0;
    if (side == 0 || q >= nq || mode > 1u || (uint64_t)H * W > 0xffffffffull) {
        return cudaErrorInvalidValue;
    }
    const uint64_t n = (uint64_t)H * W;
    pd_sam3_mask_up2_kernel<<<(uint32_t)((n + 255ull) / 256ull), 256, 0, (cudaStream_t)stream>>>(
        (const float*)logits, (uint8_t*)out, side, nq, q, H, W, mode);
    return pd_launch_status();
}

// 797: one owner a pixel, Meta's _apply_object_wise_non_overlapping_
// constraints on the output masks: over n u8 masks of px pixels (any one
// layout, the same for all), each pixel stays with the mask whose object has
// the highest score among those covering it - the first such mask on a tie,
// torch.argmax's - and only when that score is positive.
__global__ void pd_sam3_mask_owner_kernel(uint8_t* __restrict__ masks,
                                          const float* __restrict__ scores, uint32_t n,
                                          uint32_t px) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= px) return;
    uint32_t best = 0;
    float top = 0.0f;
    for (uint32_t j = 0; j < n; ++j) {
        const float v = masks[(size_t)j * px + i] ? scores[j] : 0.0f;
        if (j == 0 || v > top) {
            top = v;
            best = j;
        }
    }
    for (uint32_t j = 0; j < n; ++j) {
        uint8_t* m = &masks[(size_t)j * px + i];
        if (*m) *m = (j == best && top > 0.0f) ? 1u : 0u;
    }
}

PD_EXPORT
int pd_sam3_mask_owner(void* masks, const void* scores, uint32_t n, uint32_t px, void* stream) {
    if (n < 2u || px == 0u) return 0;
    pd_sam3_mask_owner_kernel<<<(px + 255u) / 256u, 256, 0, (cudaStream_t)stream>>>(
        (uint8_t*)masks, (const float*)scores, n, px);
    return pd_launch_status();
}

// 798: each mask's box and area: n u8 COLUMN-major [W][H] masks -> out[j] =
// {x0, y0, x1, y1, area}, the extremes inclusive (torchvision's
// masks_to_boxes, Meta's perflib twin); an empty mask reads {W, H, 0, 0, 0}.
__global__ void pd_sam3_mask_boxes_init_kernel(uint32_t* __restrict__ out, uint32_t n, uint32_t H,
                                               uint32_t W) {
    const uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    out[j * 5u + 0u] = W;
    out[j * 5u + 1u] = H;
    out[j * 5u + 2u] = 0u;
    out[j * 5u + 3u] = 0u;
    out[j * 5u + 4u] = 0u;
}
__global__ void pd_sam3_mask_boxes_kernel(const uint8_t* __restrict__ masks,
                                          uint32_t* __restrict__ out, uint32_t H, uint32_t W) {
    const uint32_t j = blockIdx.y;
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t px = H * W;
    const bool on = i < px && masks[(size_t)j * px + i] != 0u;
    const uint32_t x = i / H, y = i - (i / H) * H;
    // warp-reduce first: a warp's 32 pixels sit in one or two columns
    uint32_t x0 = on ? x : W, y0 = on ? y : H, x1 = on ? x : 0u, y1 = on ? y : 0u;
    const uint32_t cnt = (uint32_t)__popc(__ballot_sync(0xffffffffu, on));
    if (cnt == 0u) return;
#pragma unroll
    for (uint32_t o = 16; o > 0; o >>= 1) {
        x0 = min(x0, __shfl_xor_sync(0xffffffffu, x0, o));
        y0 = min(y0, __shfl_xor_sync(0xffffffffu, y0, o));
        x1 = max(x1, __shfl_xor_sync(0xffffffffu, x1, o));
        y1 = max(y1, __shfl_xor_sync(0xffffffffu, y1, o));
    }
    if ((threadIdx.x & 31u) == 0u) {
        uint32_t* b = out + (size_t)j * 5u;
        atomicMin(&b[0], x0);
        atomicMin(&b[1], y0);
        atomicMax(&b[2], x1);
        atomicMax(&b[3], y1);
        atomicAdd(&b[4], cnt);
    }
}

PD_EXPORT
int pd_sam3_mask_boxes(const void* masks, void* out, uint32_t n, uint32_t H, uint32_t W,
                       void* stream) {
    if (n == 0u) return 0;
    if (H == 0u || W == 0u || n > 65535u || (uint64_t)H * W > 0xffffffffull - 256ull) {
        return cudaErrorInvalidValue;
    }
    const cudaStream_t st = (cudaStream_t)stream;
    const uint32_t px = H * W;
    pd_sam3_mask_boxes_init_kernel<<<(n + 255u) / 256u, 256, 0, st>>>((uint32_t*)out, n, H, W);
    const dim3 grid((px + 255u) / 256u, n);
    pd_sam3_mask_boxes_kernel<<<grid, 256, 0, st>>>((const uint8_t*)masks, (uint32_t*)out, H, W);
    return pd_launch_status();
}

// ------------------------------------------------------------ sam3 video frames
// Meta's JPEG-folder loader resizes each frame with torchvision's TF.resize on
// the PIL image, which is Pillow's ImagingResample with the bilinear filter:
// two separable passes, horizontal then vertical, each with integer weights
// (the filter's taps normalized in double, then fixed point at 22 bits) and
// rounded + clipped back to u8 in between. Integer arithmetic on both sides,
// so the frames match Pillow's to the byte. (A video FILE goes through cv2's
// INTER_CUBIC instead - a different loader, not this one.)

// 799: one axis's taps, Pillow's precompute_coeffs + normalize_coeffs_8bpc.
// For output index xx: center = (xx + 0.5) * scale, scale = in / out; the
// window runs from (int)(center - support + 0.5) (at least 0) for
// (int)(center + support + 0.5) - xmin taps (at most to `in`), support =
// max(scale, 1); tap x weighs the triangle at (x + xmin - center + 0.5) / max(scale, 1),
// normalized to sum to one, then (int)(+-0.5 + w * 2^22). bounds[xx] =
// {xmin, taps}, kk[xx][ksize] the weights (zero past the taps). Double, as
// Pillow computes them - IEEE on both sides.
__global__ void pd_sam3_pil_coeffs_kernel(int32_t* __restrict__ bounds, int32_t* __restrict__ kk,
                                          uint32_t in_size, uint32_t out_size, uint32_t ksize) {
    const uint32_t xx = blockIdx.x * blockDim.x + threadIdx.x;
    if (xx >= out_size) return;
    const double scale = (double)((float)in_size - 0.0f) / (double)out_size;
    const double filterscale = scale < 1.0 ? 1.0 : scale;
    const double support = 1.0 * filterscale;
    const double center = ((double)xx + 0.5) * scale;
    const double ss = 1.0 / filterscale;
    int xmin = (int)(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    int xmax = (int)(center + support + 0.5);
    if (xmax > (int)in_size) xmax = (int)in_size;
    xmax -= xmin;
    double w[64];
    double ww = 0.0;
    for (int x = 0; x < xmax && x < 64; ++x) {
        double t = ((double)(x + xmin) - center + 0.5) * ss;
        if (t < 0.0) t = -t;
        const double v = t < 1.0 ? 1.0 - t : 0.0;
        w[x] = v;
        ww += v;
    }
    int32_t* k = kk + (size_t)xx * ksize;
    for (int x = 0; x < (int)ksize; ++x) {
        if (x < xmax && x < 64) {
            const double v = ww != 0.0 ? w[x] / ww : w[x];
            k[x] = v < 0.0 ? (int32_t)(-0.5 + v * (double)(1 << 22))
                           : (int32_t)(0.5 + v * (double)(1 << 22));
        } else {
            k[x] = 0;
        }
    }
    bounds[2u * xx] = xmin;
    bounds[2u * xx + 1u] = xmax;
}

PD_EXPORT
int pd_sam3_pil_coeffs(void* bounds, void* kk, uint32_t in_size, uint32_t out_size,
                       uint32_t ksize, void* stream) {
    if (out_size == 0u) return 0;
    if (in_size == 0u || ksize == 0u || ksize > 64u) return cudaErrorInvalidValue;
    pd_sam3_pil_coeffs_kernel<<<(out_size + 127u) / 128u, 128, 0, (cudaStream_t)stream>>>(
        (int32_t*)bounds, (int32_t*)kk, in_size, out_size, ksize);
    return pd_launch_status();
}

// 800: one Pillow pass over a u8 HWC picture. axis 0 (horizontal): src
// [rows][in][c] -> dst [rows][out][c]; axis 1 (vertical): src [in][rows][c]
// -> dst [out][rows][c] (rows = the other dimension). Each output value is
// clip8((2^21 + sum_x src * kk) >> 22) over its window - Pillow's
// ImagingResampleHorizontal_8bpc / Vertical_8bpc, one thread a value.
__global__ void pd_sam3_pil_pass_kernel(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
                                        const int32_t* __restrict__ bounds,
                                        const int32_t* __restrict__ kk, uint32_t rows,
                                        uint32_t in_size, uint32_t out_size, uint32_t ch,
                                        uint32_t ksize, uint32_t axis) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)rows * out_size * ch;
    if (i >= n) return;
    const uint32_t c = (uint32_t)(i % ch);
    const uint64_t pix = i / ch;
    uint32_t r, o;
    if (axis == 0u) {
        r = (uint32_t)(pix / out_size);
        o = (uint32_t)(pix - (uint64_t)r * out_size);
    } else {
        o = (uint32_t)(pix / rows);
        r = (uint32_t)(pix - (uint64_t)o * rows);
    }
    const int32_t xmin = bounds[2u * o], taps = bounds[2u * o + 1u];
    const int32_t* k = kk + (size_t)o * ksize;
    int32_t ss = 1 << 21;
    for (int32_t x = 0; x < taps; ++x) {
        const uint32_t at = (uint32_t)(xmin + x);
        const size_t s = axis == 0u ? ((size_t)r * in_size + at) * ch + c
                                    : ((size_t)at * rows + r) * ch + c;
        ss += (int32_t)src[s] * k[x];
    }
    const int32_t v = ss >> 22;
    dst[i] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
}

PD_EXPORT
int pd_sam3_pil_pass(const void* src, void* dst, const void* bounds, const void* kk,
                     uint32_t rows, uint32_t in_size, uint32_t out_size, uint32_t ch,
                     uint32_t ksize, uint32_t axis, void* stream) {
    const uint64_t n = (uint64_t)rows * out_size * ch;
    if (n == 0u) return 0;
    if (axis > 1u || in_size == 0u || ksize == 0u || n > 0xffffffffull * 256ull) {
        return cudaErrorInvalidValue;
    }
    pd_sam3_pil_pass_kernel<<<(uint32_t)((n + 255ull) / 256ull), 256, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)src, (uint8_t*)dst, (const int32_t*)bounds, (const int32_t*)kk, rows,
        in_size, out_size, ch, ksize, axis);
    return pd_launch_status();
}

// ------------------------------------------------------------ sam3 video mask chains
// 802: column q of a pixel-major [px][nq] logits plane into a plane of its
// own - the chosen candidate of the tracker heads' [px][4] landing, a
// detection's mask out of the detector's [px][queries].
__global__ void pd_sam3_mask_pick_kernel(const float* __restrict__ src, float* __restrict__ dst,
                                         uint32_t px, uint32_t nq, uint32_t q) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < px) dst[i] = src[(size_t)i * nq + q];
}

PD_EXPORT
int pd_sam3_mask_pick(const void* src, void* dst, uint32_t px, uint32_t nq, uint32_t q,
                      void* stream) {
    if (px == 0u) return 0;
    if (q >= nq) return cudaErrorInvalidValue;
    pd_sam3_mask_pick_kernel<<<(px + 255u) / 256u, 256, 0, (cudaStream_t)stream>>>(
        (const float*)src, (float*)dst, px, nq, q);
    return pd_launch_status();
}

// 803: f32 planes resized the way torch's F.interpolate(mode="bilinear",
// align_corners=False) does on CUDA, n planes of ih x iw (`istride` floats
// apart) to oh x ow (`ostride` apart).
// - aa 0: upsample_bilinear2d: source y = (ih/oh) (y + 0.5) - 0.5, at least
//   0, its two rows (the second clamped), lerped as h0 (w0 a + w1 b) + h1 (w0
//   c + w1 d).
// - aa 1: _upsample_bilinear2d_aa (antialias=True): the triangle filter
//   widened by the scale where it shrinks, centre scale (o + 0.5), window
//   from (int)(centre - support + 0.5), taps normalized over the window;
//   rows first (each a float sum over x), then the column of row sums.
// Then `mode`: 0 the value as is, 1 v > thr ? hi : lo (Meta binarizes a
// detection's 1152^2 mask with > 0 and a video-size one with > 0.5 into
// +-1024 straight after resampling).
__device__ __forceinline__ float pd_sam3_tri(float x) {
    if (x < 0.0f) x = -x;
    return x < 1.0f ? 1.0f - x : 0.0f;
}
__device__ __forceinline__ void pd_sam3_aa_span(uint32_t o, uint32_t in, float scale,
                                                float support, int& lo, int& n, float& center) {
    center = scale * ((float)o + 0.5f);
    lo = max((int)(center - support + 0.5f), 0);
    n = min((int)(center + support + 0.5f), (int)in) - lo;
}
__device__ __forceinline__ int pd_sam3_aa_weights(float* w, float scale, int lo, float center,
                                                  int n) {
    const float inv = scale >= 1.0f ? 1.0f / scale : 1.0f;
    const float lmc = (float)lo - center;
    float tot = 0.0f;
    if (n > 32) n = 32;
    for (int j = 0; j < n; ++j) {
        const float v = pd_sam3_tri(((float)j + lmc + 0.5f) * inv);
        w[j] = v;
        tot += v;
    }
    if (tot != 0.0f)
        for (int j = 0; j < n; ++j) w[j] /= tot;
    return n;
}
__global__ void pd_sam3_resize_f32_kernel(const float* __restrict__ src, float* __restrict__ dst,
                                          uint32_t ih, uint32_t iw, uint32_t oh, uint32_t ow,
                                          uint64_t istride, uint64_t ostride, uint32_t aa,
                                          uint32_t mode, float thr, float lo_v, float hi_v) {
    const uint32_t j = blockIdx.y;
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= oh * ow) return;
    const uint32_t y = i / ow, x = i - y * ow;
    const float rh = (float)ih / (float)oh, rw = (float)iw / (float)ow;
    const float* s = src + (size_t)j * istride;
    float v;
    if (aa == 0u) {
        float sy = rh * ((float)y + 0.5f) - 0.5f, sx = rw * ((float)x + 0.5f) - 0.5f;
        sy = sy < 0.0f ? 0.0f : sy;
        sx = sx < 0.0f ? 0.0f : sx;
        const uint32_t y0 = (uint32_t)sy, x0 = (uint32_t)sx;
        const uint32_t y1 = y0 + (y0 < ih - 1u ? 1u : 0u), x1 = x0 + (x0 < iw - 1u ? 1u : 0u);
        const float l1y = sy - (float)y0, l0y = 1.0f - l1y, l1x = sx - (float)x0,
                    l0x = 1.0f - l1x;
        v = l0y * (l0x * s[(size_t)y0 * iw + x0] + l1x * s[(size_t)y0 * iw + x1]) +
            l1y * (l0x * s[(size_t)y1 * iw + x0] + l1x * s[(size_t)y1 * iw + x1]);
    } else {
        const float sh = rh >= 1.0f ? rh : 1.0f, sw = rw >= 1.0f ? rw : 1.0f;
        int ylo, yn, xlo, xn;
        float yc, xc;
        pd_sam3_aa_span(y, ih, rh, sh, ylo, yn, yc);
        pd_sam3_aa_span(x, iw, rw, sw, xlo, xn, xc);
        float wy[32], wx[32];
        yn = pd_sam3_aa_weights(wy, rh, ylo, yc, yn);
        xn = pd_sam3_aa_weights(wx, rw, xlo, xc, xn);
        v = 0.0f;
        for (int yy = 0; yy < yn; ++yy) {
            const float* row = s + (size_t)(ylo + yy) * iw + xlo;
            float r = row[0] * wx[0];
            for (int xx = 1; xx < xn; ++xx) r += row[xx] * wx[xx];
            v = yy == 0 ? r * wy[0] : v + r * wy[yy];
        }
    }
    if (mode == 1u) v = v > thr ? hi_v : lo_v;
    dst[(size_t)j * ostride + i] = v;
}

PD_EXPORT
int pd_sam3_resize_f32(const void* src, void* dst, uint32_t n, uint32_t ih, uint32_t iw,
                       uint32_t oh, uint32_t ow, uint64_t istride, uint64_t ostride, uint32_t aa,
                       uint32_t mode, float thr, float lo, float hi, void* stream) {
    if (n == 0u || oh == 0u || ow == 0u) return 0;
    if (ih == 0u || iw == 0u || aa > 1u || mode > 1u || n > 65535u ||
        (uint64_t)oh * ow > 0xffffffffull - 256ull) {
        return cudaErrorInvalidValue;
    }
    // the antialiased windows hold at most 32 taps: a shrink by 15x or less
    if (aa == 1u && ((float)ih / (float)oh > 15.0f || (float)iw / (float)ow > 15.0f)) {
        return cudaErrorInvalidValue;
    }
    const dim3 grid((oh * ow + 255u) / 256u, n);
    pd_sam3_resize_f32_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(
        (const float*)src, (float*)dst, ih, iw, oh, ow, istride, ostride, aa, mode, thr, lo, hi);
    return pd_launch_status();
}

// 804: the tracker's mask_downsample, Conv2d(1, 1, k=4, s=4) + bias, over n
// planes of 4s x 4s (`istride` apart) into s x s (`ostride` apart): a mask
// at the input-mask size (1152) as the heads' 288^2 mask prompt. Taps in
// raster order, one fma each.
__global__ void pd_sam3_mask_down4_kernel(const float* __restrict__ src, float* __restrict__ dst,
                                          const float* __restrict__ w, const float* __restrict__ b,
                                          uint32_t s, uint64_t istride, uint64_t ostride) {
    const uint32_t j = blockIdx.y;
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= s * s) return;
    const uint32_t y = i / s, x = i - y * s, iw = 4u * s;
    const float* p = src + (size_t)j * istride + (size_t)(4u * y) * iw + 4u * x;
    float acc = 0.0f;
#pragma unroll
    for (uint32_t ky = 0; ky < 4u; ++ky)
#pragma unroll
        for (uint32_t kx = 0; kx < 4u; ++kx) acc = fmaf(w[ky * 4u + kx], p[ky * iw + kx], acc);
    dst[(size_t)j * ostride + i] = acc + b[0];
}

PD_EXPORT
int pd_sam3_mask_down4(const void* src, void* dst, const void* w, const void* b, uint32_t n,
                       uint32_t s, uint64_t istride, uint64_t ostride, void* stream) {
    if (n == 0u || s == 0u) return 0;
    if (n > 65535u || s > 16384u) return cudaErrorInvalidValue;
    const dim3 grid((s * s + 255u) / 256u, n);
    pd_sam3_mask_down4_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(
        (const float*)src, (float*)dst, (const float*)w, (const float*)b, s, istride, ostride);
    return pd_launch_status();
}

// 805: Meta's pixel-wise non-overlap across n planes of px values (`stride`
// apart): at each pixel the plane with the highest value keeps it (the first
// on a tie, torch.argmax), every other is clamped to at most -10. counts (n x
// 2 u32, or null) gets each plane's area before (> 0) and after (kept and >
// 0) - what _suppress_object_pw_area_shrinkage weighs; write 1 applies the
// clamp in place (a newborn state's consolidation), 0 only counts.
__global__ void pd_sam3_nonoverlap_kernel(float* __restrict__ planes, uint32_t* __restrict__ counts,
                                          uint64_t stride, uint32_t n, uint32_t px,
                                          uint32_t write) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool in = i < px;
    uint32_t best = 0;
    float top = 0.0f;
    if (in) {
        for (uint32_t j = 0; j < n; ++j) {
            const float v = planes[(size_t)j * stride + i];
            if (j == 0u || v > top) {
                top = v;
                best = j;
            }
        }
    }
    for (uint32_t j = 0; j < n; ++j) {
        float v = in ? planes[(size_t)j * stride + i] : 0.0f;
        const bool before = in && v > 0.0f;
        const bool after = before && j == best;
        if (in && write && j != best && v > -10.0f) planes[(size_t)j * stride + i] = -10.0f;
        if (counts != nullptr) {
            const uint32_t cb = (uint32_t)__popc(__ballot_sync(0xffffffffu, before));
            const uint32_t ca = (uint32_t)__popc(__ballot_sync(0xffffffffu, after));
            if ((threadIdx.x & 31u) == 0u) {
                if (cb) atomicAdd(&counts[2u * j], cb);
                if (ca) atomicAdd(&counts[2u * j + 1u], ca);
            }
        }
    }
}

PD_EXPORT
int pd_sam3_nonoverlap(void* planes, void* counts, uint64_t stride, uint32_t n, uint32_t px,
                       uint32_t write, void* stream) {
    if (n == 0u || px == 0u) return 0;
    if (write > 1u || px > 0xffffffffu - 256u) return cudaErrorInvalidValue;
    const cudaStream_t st = (cudaStream_t)stream;
    if (counts != nullptr)
        pd_sam3_zero_u32_kernel<<<(2u * n + 255u) / 256u, 256, 0, st>>>((uint32_t*)counts, 2u * n);
    // one plane only: Meta returns it untouched (and every pixel is its own)
    if (n == 1u && write) write = 0u;
    pd_sam3_nonoverlap_kernel<<<(px + 255u) / 256u, 256, 0, st>>>(
        (float*)planes, (uint32_t*)counts, stride, n, px, write);
    return pd_launch_status();
}
