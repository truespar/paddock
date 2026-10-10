// PP-DocLayoutV3 - PaddleOCR-VL's layout stage (an RT-DETR detector: an
// HGNetV2-L backbone, a hybrid encoder, a deformable-attention decoder) - the
// glue around its f16 GEMMs. Every activation plane is NHWC halves (rows =
// pixels, columns = channels), so a 1 x 1 conv is a plain GEMM and a k x k
// conv an im2row plus one; math inside a kernel is f32. The input is fixed at
// 800 x 800, so every shape here is one of a handful.

// 837: the resized page (u8 RGB, HWC) -> halves / 255 (the checkpoint's
// rescale; mean 0, std 1)
__global__ void pd_dl_u8_to_h_kernel(const uint8_t* __restrict__ src, __half* __restrict__ dst,
                                     uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half_rn((float)src[i] / 255.0f);
}

PD_EXPORT int pd_dl_u8_to_h(const void* src, void* dst, uint32_t n, void* stream) {
    if (n == 0u) return 0;
    pd_dl_u8_to_h_kernel<<<(n + 255u) / 256u, 256u, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)src, (__half*)dst, n);
    return pd_launch_status();
}

// 838: k x k im2row of an NHWC plane: dst [oh * ow][kpad] halves in tap order
// (ky, kx, c) - the weight is re-laid to match at load - with zeros where a
// tap falls outside the source (the conv's zero padding: pad_t / pad_l
// before, whatever runs past the bottom / right edge after) and in the K pad
// (kpad >= kh * kw * c, a multiple of 8 for the GEMM's 16 B stage).
__global__ void pd_dl_im2row_h_kernel(const __half* __restrict__ src, __half* __restrict__ dst,
                                      uint32_t ih, uint32_t iw, uint32_t c, uint32_t oh,
                                      uint32_t ow, uint32_t kh, uint32_t kw, uint32_t stride,
                                      int32_t pad_t, int32_t pad_l, uint32_t kpad) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)oh * ow * kpad) return;
    const uint32_t k = (uint32_t)(i % kpad);
    const uint32_t r = (uint32_t)(i / kpad);
    __half v = __float2half_rn(0.0f);
    if (k < kh * kw * c) {
        const uint32_t tap = k / c, ch = k % c;
        const int32_t ky = (int32_t)(tap / kw), kx = (int32_t)(tap % kw);
        const int32_t iy = (int32_t)((r / ow) * stride) - pad_t + ky;
        const int32_t ix = (int32_t)((r % ow) * stride) - pad_l + kx;
        if (iy >= 0 && ix >= 0 && iy < (int32_t)ih && ix < (int32_t)iw)
            v = src[((size_t)iy * iw + ix) * c + ch];
    }
    dst[i] = v;
}

PD_EXPORT int pd_dl_im2row_h(const void* src, void* dst, uint32_t ih, uint32_t iw, uint32_t c,
                             uint32_t oh, uint32_t ow, uint32_t kh, uint32_t kw, uint32_t stride,
                             int32_t pad_t, int32_t pad_l, uint32_t kpad, void* stream) {
    const uint64_t n = (uint64_t)oh * ow * kpad;
    if (n == 0u) return 0;
    if (kpad < kh * kw * c) return (int)cudaErrorInvalidValue;
    pd_dl_im2row_h_kernel<<<(uint32_t)((n + 255u) / 256u), 256u, 0, (cudaStream_t)stream>>>(
        (const __half*)src, (__half*)dst, ih, iw, c, oh, ow, kh, kw, stride, pad_t, pad_l, kpad);
    return pd_launch_status();
}

// 839: depthwise k x k conv over an NHWC plane, padding (k - 1) / 2 a side:
// f32 weights [c][k * k] and bias (BatchNorm folded at load), act 0 none,
// 1 ReLU. HGNetV2's stage downsamples (3 x 3, stride 2) and its light
// blocks' 5 x 5.
__global__ void pd_dl_dwconv_h_kernel(const __half* __restrict__ src, __half* __restrict__ dst,
                                      const float* __restrict__ w, const float* __restrict__ b,
                                      uint32_t ih, uint32_t iw, uint32_t c, uint32_t oh,
                                      uint32_t ow, uint32_t k, uint32_t stride, uint32_t act) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)oh * ow * c) return;
    const uint32_t ch = (uint32_t)(i % c);
    const uint32_t r = (uint32_t)(i / c);
    const int32_t pad = (int32_t)(k - 1u) / 2;
    const int32_t y0 = (int32_t)((r / ow) * stride) - pad;
    const int32_t x0 = (int32_t)((r % ow) * stride) - pad;
    const float* wk = w + (size_t)ch * k * k;
    float acc = 0.0f;
    for (uint32_t ky = 0; ky < k; ++ky) {
        const int32_t iy = y0 + (int32_t)ky;
        if (iy < 0 || iy >= (int32_t)ih) continue;
        for (uint32_t kx = 0; kx < k; ++kx) {
            const int32_t ix = x0 + (int32_t)kx;
            if (ix < 0 || ix >= (int32_t)iw) continue;
            acc = fmaf(__half2float(src[((size_t)iy * iw + ix) * c + ch]), wk[ky * k + kx], acc);
        }
    }
    acc += b[ch];
    if (act == 1u) acc = fmaxf(acc, 0.0f);
    dst[i] = __float2half_rn(acc);
}

PD_EXPORT int pd_dl_dwconv_h(const void* src, void* dst, const void* w, const void* b,
                             uint32_t ih, uint32_t iw, uint32_t c, uint32_t oh, uint32_t ow,
                             uint32_t k, uint32_t stride, uint32_t act, void* stream) {
    const uint64_t n = (uint64_t)oh * ow * c;
    if (n == 0u) return 0;
    pd_dl_dwconv_h_kernel<<<(uint32_t)((n + 255u) / 256u), 256u, 0, (cudaStream_t)stream>>>(
        (const __half*)src, (__half*)dst, (const float*)w, (const float*)b, ih, iw, c, oh, ow, k,
        stride, act);
    return pd_launch_status();
}

// 840: 2 x 2 / stride 1 max pool of an NHWC plane padded by one zero row and
// column at the bottom / right - HGNetV2's stem pool over its padded
// embedding (ceil mode) - so the output keeps the input's h x w.
__global__ void pd_dl_maxpool2_h_kernel(const __half* __restrict__ src, __half* __restrict__ dst,
                                        uint32_t h, uint32_t w, uint32_t c) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)h * w * c) return;
    const uint32_t ch = (uint32_t)(i % c);
    const uint32_t r = (uint32_t)(i / c);
    const uint32_t y = r / w, x = r % w;
    float m = -INFINITY;
    for (uint32_t dy = 0; dy < 2u; ++dy)
        for (uint32_t dx = 0; dx < 2u; ++dx) {
            const bool in = y + dy < h && x + dx < w;
            const float v = in ? __half2float(src[((size_t)(y + dy) * w + x + dx) * c + ch]) : 0.0f;
            m = fmaxf(m, v);
        }
    dst[i] = __float2half_rn(m);
}

PD_EXPORT int pd_dl_maxpool2_h(const void* src, void* dst, uint32_t h, uint32_t w, uint32_t c,
                               void* stream) {
    const uint64_t n = (uint64_t)h * w * c;
    if (n == 0u) return 0;
    pd_dl_maxpool2_h_kernel<<<(uint32_t)((n + 255u) / 256u), 256u, 0, (cudaStream_t)stream>>>(
        (const __half*)src, (__half*)dst, h, w, c);
    return pd_launch_status();
}

// 841: copy an NHWC plane of `c` channels into a wider one of `ctot` at
// channel offset `off` - the backbone's block concats and the encoder's.
__global__ void pd_dl_concat_h_kernel(const __half* __restrict__ src, __half* __restrict__ dst,
                                      uint32_t rows, uint32_t c, uint32_t ctot, uint32_t off) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)rows * c) return;
    const uint64_t r = i / c;
    dst[r * ctot + off + i % c] = src[i];
}

PD_EXPORT int pd_dl_concat_h(const void* src, void* dst, uint32_t rows, uint32_t c, uint32_t ctot,
                             uint32_t off, void* stream) {
    const uint64_t n = (uint64_t)rows * c;
    if (n == 0u) return 0;
    if (off + c > ctot) return (int)cudaErrorInvalidValue;
    pd_dl_concat_h_kernel<<<(uint32_t)((n + 255u) / 256u), 256u, 0, (cudaStream_t)stream>>>(
        (const __half*)src, (__half*)dst, rows, c, ctot, off);
    return pd_launch_status();
}

// 842: 2x upsample of an NHWC plane (h x w -> 2h x 2w): mode 0 nearest
// (F.interpolate scale 2, "nearest"), 1 bilinear at align_corners=False
// (torch's half-pixel source coordinate, clamped at 0). `add`, when not null,
// is a plane of the output's shape added on (the mask branch's sums).
__global__ void pd_dl_up2_h_kernel(const __half* __restrict__ src, __half* __restrict__ dst,
                                   const __half* __restrict__ add, uint32_t h, uint32_t w,
                                   uint32_t c, uint32_t mode) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t oh = 2u * h, ow = 2u * w;
    if (i >= (uint64_t)oh * ow * c) return;
    const uint32_t ch = (uint32_t)(i % c);
    const uint32_t r = (uint32_t)(i / c);
    const uint32_t oy = r / ow, ox = r % ow;
    float v;
    if (mode == 0u) {
        v = __half2float(src[((size_t)(oy / 2u) * w + ox / 2u) * c + ch]);
    } else {
        const float sy = fmaxf(((float)oy + 0.5f) * 0.5f - 0.5f, 0.0f);
        const float sx = fmaxf(((float)ox + 0.5f) * 0.5f - 0.5f, 0.0f);
        const uint32_t y0 = (uint32_t)sy, x0 = (uint32_t)sx;
        const uint32_t y1 = min(y0 + 1u, h - 1u), x1 = min(x0 + 1u, w - 1u);
        const float ly = sy - (float)y0, lx = sx - (float)x0;
        const float a = __half2float(src[((size_t)y0 * w + x0) * c + ch]);
        const float b = __half2float(src[((size_t)y0 * w + x1) * c + ch]);
        const float cc = __half2float(src[((size_t)y1 * w + x0) * c + ch]);
        const float d = __half2float(src[((size_t)y1 * w + x1) * c + ch]);
        v = (1.0f - ly) * ((1.0f - lx) * a + lx * b) + ly * ((1.0f - lx) * cc + lx * d);
    }
    if (add != nullptr) v += __half2float(add[i]);
    dst[i] = __float2half_rn(v);
}

PD_EXPORT int pd_dl_up2_h(const void* src, void* dst, const void* add, uint32_t h, uint32_t w,
                          uint32_t c, uint32_t mode, void* stream) {
    const uint64_t n = 4ull * h * w * c;
    if (n == 0u) return 0;
    pd_dl_up2_h_kernel<<<(uint32_t)((n + 255u) / 256u), 256u, 0, (cudaStream_t)stream>>>(
        (const __half*)src, (__half*)dst, (const __half*)add, h, w, c, mode);
    return pd_launch_status();
}

// 843: dst = a + b over halves, act 0 none / 1 ReLU / 2 SiLU on the sum - the
// HG block residual, the CSP layer's two branches.
__global__ void pd_dl_add_h_kernel(const __half* __restrict__ a, const __half* __restrict__ b,
                                   __half* __restrict__ dst, uint64_t n, uint32_t act) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = __half2float(a[i]) + __half2float(b[i]);
    if (act == 1u) v = fmaxf(v, 0.0f);
    if (act == 2u) v = v / (1.0f + expf(-v));
    dst[i] = __float2half_rn(v);
}

PD_EXPORT int pd_dl_add_h(const void* a, const void* b, void* dst, uint64_t n, uint32_t act,
                          void* stream) {
    if (n == 0u) return 0;
    pd_dl_add_h_kernel<<<(uint32_t)((n + 255u) / 256u), 256u, 0, (cudaStream_t)stream>>>(
        (const __half*)a, (const __half*)b, (__half*)dst, n, act);
    return pd_launch_status();
}

// ---- query selection and the deformable decoder ---------------------------

// 844: plane[r][:] *= scale[r] over halves - the encoder memory times the
// anchors' valid mask (0 / 1) before query selection.
__global__ void pd_dl_rowscale_h_kernel(__half* __restrict__ x, const float* __restrict__ scale,
                                        uint32_t rows, uint32_t cols) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)rows * cols) return;
    x[i] = __float2half_rn(__half2float(x[i]) * scale[i / cols]);
}

PD_EXPORT int pd_dl_rowscale_h(void* x, const void* scale, uint32_t rows, uint32_t cols,
                               void* stream) {
    const uint64_t n = (uint64_t)rows * cols;
    if (n == 0u) return 0;
    pd_dl_rowscale_h_kernel<<<(uint32_t)((n + 255u) / 256u), 256u, 0, (cudaStream_t)stream>>>(
        (__half*)x, (const float*)scale, rows, cols);
    return pd_launch_status();
}

// 845: dst[i][:] = src[idx[i]][:] for rows of `words` 32-bit words - the
// top-k gather of query features (f32 rows, or f16 rows of even width).
__global__ void pd_dl_gather_rows_kernel(const uint32_t* __restrict__ src,
                                         uint32_t* __restrict__ dst,
                                         const uint32_t* __restrict__ idx, uint32_t n,
                                         uint32_t words) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)n * words) return;
    const uint32_t r = (uint32_t)(i / words), w = (uint32_t)(i % words);
    dst[i] = src[(size_t)idx[r] * words + w];
}

PD_EXPORT int pd_dl_gather_rows(const void* src, void* dst, const void* idx, uint32_t n,
                                uint32_t words, void* stream) {
    const uint64_t t = (uint64_t)n * words;
    if (t == 0u) return 0;
    pd_dl_gather_rows_kernel<<<(uint32_t)((t + 255u) / 256u), 256u, 0, (cudaStream_t)stream>>>(
        (const uint32_t*)src, (uint32_t*)dst, (const uint32_t*)idx, n, words);
    return pd_launch_status();
}

// 846: per query, the box of its mask's positive pixels - `logits` [q][h * w]
// f32, a pixel in where its logit > 0 - as the decoder's initial reference
// point: HF's mask_to_box_coordinate (x_min, x_max + 1 over x, likewise y,
// over w / h; an empty mask is the zero box) as (cx, cy, w, h), then through
// inverse_sigmoid (eps 1e-5) and the decoder's sigmoid, as the reference
// runs it. One block a query.
__global__ void __launch_bounds__(256) pd_dl_mask_ref_kernel(const float* __restrict__ logits,
                                                             float* __restrict__ ref,
                                                             uint32_t h, uint32_t w) {
    const uint32_t q = blockIdx.x;
    const float* row = logits + (size_t)q * h * w;
    int x0 = 0x7fffffff, y0 = 0x7fffffff, x1 = -1, y1 = -1;
    for (uint32_t p = threadIdx.x; p < h * w; p += blockDim.x) {
        if (row[p] > 0.0f) {
            const int y = (int)(p / w), x = (int)(p % w);
            x0 = min(x0, x); y0 = min(y0, y); x1 = max(x1, x); y1 = max(y1, y);
        }
    }
    __shared__ int red[4][256];
    red[0][threadIdx.x] = x0; red[1][threadIdx.x] = y0;
    red[2][threadIdx.x] = x1; red[3][threadIdx.x] = y1;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2u; s > 0u; s >>= 1) {
        if (threadIdx.x < s) {
            red[0][threadIdx.x] = min(red[0][threadIdx.x], red[0][threadIdx.x + s]);
            red[1][threadIdx.x] = min(red[1][threadIdx.x], red[1][threadIdx.x + s]);
            red[2][threadIdx.x] = max(red[2][threadIdx.x], red[2][threadIdx.x + s]);
            red[3][threadIdx.x] = max(red[3][threadIdx.x], red[3][threadIdx.x + s]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0u) {
        float b[4] = {0.f, 0.f, 0.f, 0.f};
        if (red[2][0] >= 0) {
            const float xn0 = (float)red[0][0] / (float)w, yn0 = (float)red[1][0] / (float)h;
            const float xn1 = (float)(red[2][0] + 1) / (float)w, yn1 = (float)(red[3][0] + 1) / (float)h;
            b[0] = (xn0 + xn1) / 2.0f; b[1] = (yn0 + yn1) / 2.0f;
            b[2] = xn1 - xn0; b[3] = yn1 - yn0;
        }
        for (int k = 0; k < 4; ++k) {
            const float x = fminf(fmaxf(b[k], 0.0f), 1.0f);
            const float u = logf(fmaxf(x, 1e-5f) / fmaxf(1.0f - x, 1e-5f));
            ref[q * 4 + k] = 1.0f / (1.0f + expf(-u));
        }
    }
}

PD_EXPORT int pd_dl_mask_ref(const void* logits, void* ref, uint32_t q, uint32_t h, uint32_t w,
                             void* stream) {
    if (q == 0u) return 0;
    pd_dl_mask_ref_kernel<<<q, 256u, 0, (cudaStream_t)stream>>>((const float*)logits,
                                                                (float*)ref, h, w);
    return pd_launch_status();
}

// 847: multi-scale deformable attention, 8 heads x 32, 3 levels, 4 points,
// four-coordinate reference boxes (Deformable DETR / RT-DETR):
//   loc  = ref.xy + off / 4 * ref.wh * 0.5          (the same box every level)
//   w    = softmax over the head's 12 (level, point) logits
//   out  = sum w * grid_sample(value_l, loc)         bilinear, zero padding,
//          align_corners=False: pixel = loc * size - 0.5
// value [s][256] halves, levels concatenated row-major (sizes `hw` [3][2] =
// (h, w), starts `st` [3]); off [q][8][3][4][2] and logits [q][8][12] f32 (the
// two Linear outputs as they land); ref [q][4] f32; out [q][256] halves.
// One block a query, a thread per (head, channel).
__global__ void __launch_bounds__(256) pd_dl_msda_kernel(
    const __half* __restrict__ value, const float* __restrict__ off,
    const float* __restrict__ logit, const float* __restrict__ ref, __half* __restrict__ out,
    uint32_t h0, uint32_t w0, uint32_t h1, uint32_t w1, uint32_t h2, uint32_t w2) {
    const uint32_t q = blockIdx.x, t = threadIdx.x, head = t / 32u, ch = t % 32u;
    const uint32_t hs[3] = {h0, h1, h2}, ws[3] = {w0, w1, w2};
    const uint32_t st[3] = {0u, h0 * w0, h0 * w0 + h1 * w1};
    const float* lg = logit + ((size_t)q * 8u + head) * 12u;
    float m = -INFINITY;
    for (int i = 0; i < 12; ++i) m = fmaxf(m, lg[i]);
    float e[12], sum = 0.0f;
    for (int i = 0; i < 12; ++i) { e[i] = expf(lg[i] - m); sum += e[i]; }
    const float* rb = ref + (size_t)q * 4u;
    const float* ob = off + ((size_t)q * 8u + head) * 24u;
    float acc = 0.0f;
    for (uint32_t l = 0; l < 3u; ++l) {
        const float H = (float)hs[l], W = (float)ws[l];
        for (uint32_t p = 0; p < 4u; ++p) {
            const float lx = rb[0] + ob[(l * 4u + p) * 2u] / 4.0f * rb[2] * 0.5f;
            const float ly = rb[1] + ob[(l * 4u + p) * 2u + 1u] / 4.0f * rb[3] * 0.5f;
            const float px = lx * W - 0.5f, py = ly * H - 0.5f;
            const float fx = floorf(px), fy = floorf(py);
            const int x0 = (int)fx, y0 = (int)fy;
            const float ax = px - fx, ay = py - fy;
            float v = 0.0f;
            for (int dy = 0; dy < 2; ++dy) {
                const int yy = y0 + dy;
                if (yy < 0 || yy >= (int)hs[l]) continue;
                for (int dx = 0; dx < 2; ++dx) {
                    const int xx = x0 + dx;
                    if (xx < 0 || xx >= (int)ws[l]) continue;
                    const float wt = (dy ? ay : 1.0f - ay) * (dx ? ax : 1.0f - ax);
                    const size_t row = st[l] + (size_t)yy * ws[l] + xx;
                    v += wt * __half2float(value[row * 256u + head * 32u + ch]);
                }
            }
            acc += (e[l * 4u + p] / sum) * v;
        }
    }
    out[(size_t)q * 256u + t] = __float2half_rn(acc);
}

PD_EXPORT int pd_dl_msda(const void* value, const void* off, const void* logit, const void* ref,
                         void* out, uint32_t q, uint32_t h0, uint32_t w0, uint32_t h1,
                         uint32_t w1, uint32_t h2, uint32_t w2, void* stream) {
    if (q == 0u) return 0;
    pd_dl_msda_kernel<<<q, 256u, 0, (cudaStream_t)stream>>>(
        (const __half*)value, (const float*)off, (const float*)logit, (const float*)ref,
        (__half*)out, h0, w0, h1, w1, h2, w2);
    return pd_launch_status();
}

// 848: the decoder's reference box step, f32 [q][4]: with `delta` (the box
// head's output) ref = sigmoid(delta + inverse_sigmoid(ref)) (eps 1e-5), and
// either way ref16 [q][8] = the box as halves, zero-padded to the query-pos
// head's 8-wide GEMM input.
__global__ void pd_dl_ref_step_kernel(float* __restrict__ ref, const float* __restrict__ delta,
                                      __half* __restrict__ ref16, uint32_t q) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= q * 8u) return;
    const uint32_t r = i / 8u, k = i % 8u;
    if (k >= 4u) { ref16[i] = __float2half_rn(0.0f); return; }
    float x = ref[r * 4u + k];
    if (delta != nullptr) {
        const float c = fminf(fmaxf(x, 0.0f), 1.0f);
        const float u = logf(fmaxf(c, 1e-5f) / fmaxf(1.0f - c, 1e-5f));
        x = 1.0f / (1.0f + expf(-(delta[r * 4u + k] + u)));
        ref[r * 4u + k] = x;
    }
    ref16[i] = __float2half_rn(x);
}

PD_EXPORT int pd_dl_ref_step(void* ref, const void* delta, void* ref16, uint32_t q, void* stream) {
    if (q == 0u) return 0;
    pd_dl_ref_step_kernel<<<(q * 8u + 255u) / 256u, 256u, 0, (cudaStream_t)stream>>>(
        (float*)ref, (const float*)delta, (__half*)ref16, q);
    return pd_launch_status();
}

// 849: reading-order votes from the global pointer's scores s [q][q] (s[i][j]
// = q_i . k_j / 8): the exported graph's antisymmetric form - P = sigmoid(s -
// s^T) off the diagonal - summed down each column: votes[j] = sum_i P[i][j],
// the expected number of queries before j. Ranks follow by ascending votes.
__global__ void pd_dl_order_votes_kernel(const float* __restrict__ s, float* __restrict__ votes,
                                         uint32_t q) {
    const uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= q) return;
    float v = 0.0f;
    for (uint32_t i = 0; i < q; ++i) {
        if (i == j) continue;
        const float l = s[(size_t)i * q + j] - s[(size_t)j * q + i];
        v += 1.0f / (1.0f + expf(-l));
    }
    votes[j] = v;
}

PD_EXPORT int pd_dl_order_votes(const void* s, void* votes, uint32_t q, void* stream) {
    if (q == 0u) return 0;
    pd_dl_order_votes_kernel<<<(q + 127u) / 128u, 128u, 0, (cudaStream_t)stream>>>(
        (const float*)s, (float*)votes, q);
    return pd_launch_status();
}
