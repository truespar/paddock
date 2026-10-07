// sam3/vit.cuh - SAM 3's ViT backbone glue: the window-major patch stem, the q|k|v split with rope, and the seam back to a raster
// Textually-included segment of the single pack translation unit.
// Not standalone-compilable: include order is defined by ../pack.cu.
// ------------------------------------------------------------ sam3 backbone
// SAM 3's image encoder is a ViT-L-class tower (Meta's vitdet.py): 1008^2 in,
// 14 px patches, a 72x72 token grid, 32 blocks, 28 of them attending inside
// non-overlapping 24x24 windows and four (7, 15, 23, 31) over the whole grid.
// Everything else - the GEMMs, the residual seams, the attention - is the
// dense-prediction lane's (dense_pred.cuh), on the same precision class: f32
// residual and accumulate, f16 GEMM operands and the f16 activation interface
// between GEMMs.
//
// The one idea this file carries: the token rows live in WINDOW-MAJOR order
// for the whole tower. Row r of an image is window r / 576, cell r % 576
// inside it. Then a window block is pd_vision_attn_h over 9 groups of 576
// rows a picture with no partition pass, a global block is the same kernel
// over one group of 5184, and the tiled absolute position embedding - tiled
// with period 24, which IS the window side - is one [576][d] table broadcast
// per window. LayerNorm, the GEMMs and the residual are per-row and never see
// the order. Only the rope (position per row) and the exit to the neck (a
// raster) do, and both are written for it here.
//
// Needs dense_pred.cuh (pd_launch_status via f32_qkv.cuh, the same includes).

// 751: the patch stem. u8 RGB HWC pictures (already at side x side) -> the
// patch GEMM's f16 input rows, window-major, K padded with zeros to `kp`.
//
// u8 * f32(1/255), then (x - 0.5) / 0.5, each step rounded on its own:
// torchvision's ToDtype(float32, scale=True) MULTIPLIES by 1/255 (a division
// lands 2 ulp away on some values - checked on all 256), and Normalize is sub
// then div - what Meta's Sam3Processor runs. The intrinsics keep nvcc from
// contracting the multiply and the subtract into one FMA, which would round
// once where torch rounds twice. Column order is c*p*p + ky*p + kx,
// the conv weight [out][c][ky][kx] flattened. The patch conv has no bias, so
// the zero pad columns (588 -> 592: the f16 GEMM stages 16-byte units) meet
// zero weight columns the loader appends and add exactly nothing.
//
// 801 is the same stem for a VIDEO frame as Meta's frame loader leaves it:
// TF.to_tensor DIVIDES by 255 in f32, the frame is stored fp16, then
// normalized in fp16 (x -= 0.5, x /= 0.5, each op rounded to half on its
// own). 140 of the 256 levels land a half step away from the picture path's.
__global__ void pd_sam3_patch_rows_kernel(const uint8_t* __restrict__ px8,
                                          __half* __restrict__ out, uint32_t side,
                                          uint32_t p, uint32_t g, uint32_t win, uint32_t ch,
                                          uint32_t kp, uint64_t n, uint32_t norm) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint64_t row = i / kp;
    const uint32_t col = (uint32_t)(i - row * kp);
    const uint32_t pp = p * p;
    if (col >= ch * pp) {
        out[i] = __float2half(0.0f);
        return;
    }
    const uint32_t gg = g * g;
    const uint64_t b = row / gg;
    const uint32_t r = (uint32_t)(row - b * gg);
    // window-major row -> grid cell
    const uint32_t ww = win * win, nwx = g / win;
    const uint32_t wi = r / ww, l = r - wi * ww;
    const uint32_t gy = (wi / nwx) * win + l / win;
    const uint32_t gx = (wi % nwx) * win + l % win;
    const uint32_t c = col / pp, rem = col - c * pp;
    const uint32_t ky = rem / p, kx = rem - ky * p;
    const uint64_t src = ((b * side + gy * p + ky) * side + (gx * p + kx)) * ch + c;
    if (norm == 1u) {
        const __half h = __float2half_rn(__fdiv_rn((float)px8[src], 255.0f));
        const __half d = __float2half_rn(__fsub_rn(__half2float(h), 0.5f));
        out[i] = __float2half_rn(__fdiv_rn(__half2float(d), 0.5f));
        return;
    }
    const float x = __fmul_rn((float)px8[src], 1.0f / 255.0f);
    out[i] = __float2half(__fdiv_rn(__fsub_rn(x, 0.5f), 0.5f));
}

static int pd_sam3_patch_rows_launch(const void* pixels, void* out, uint32_t pics, uint32_t side,
                                     uint32_t patch, uint32_t win, uint32_t ch, uint32_t kp,
                                     uint32_t norm, void* stream) {
    if (pics == 0) return 0;
    if (norm > 1u) return cudaErrorInvalidValue;
    if (patch == 0 || side % patch != 0 || ch == 0 || kp < ch * patch * patch)
        return cudaErrorInvalidValue;
    const uint32_t g = side / patch;
    if (win == 0 || g % win != 0) return cudaErrorInvalidValue;
    const uint64_t n = (uint64_t)pics * g * g * kp;
    const uint64_t blocks = (n + 255ull) / 256ull;
    pd_sam3_patch_rows_kernel<<<(uint32_t)blocks, 256, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)pixels, (__half*)out, side, patch, g, win, ch, kp, n, norm);
    return pd_launch_status();
}

PD_EXPORT
int pd_sam3_patch_rows(const void* pixels, void* out, uint32_t pics, uint32_t side,
                       uint32_t patch, uint32_t win, uint32_t ch, uint32_t kp, void* stream) {
    return pd_sam3_patch_rows_launch(pixels, out, pics, side, patch, win, ch, kp, 0u, stream);
}

// 801: the stem with a normalization: 0 the picture processor's (751's), 1
// Meta's video frames'.
PD_EXPORT
int pd_sam3_patch_rows_norm(const void* pixels, void* out, uint32_t pics, uint32_t side,
                            uint32_t patch, uint32_t win, uint32_t ch, uint32_t kp, uint32_t norm,
                            void* stream) {
    return pd_sam3_patch_rows_launch(pixels, out, pics, side, patch, win, ch, kp, norm, stream);
}

// 752: the q|k|v landing -> the three half planes pd_vision_attn_h eats, every
// projection's bias folded into the load (SAM 3's qkv Linear carries all
// three, which is why 621 - DINOv3's, k biasless - cannot serve it) and the
// rope applied to q and k on every row, q scaled by 1/sqrt(hd) before its one
// round.
//
// Meta's rope is the complex form: pair (2i, 2i+1) of a head rotates by the
// angle of frequency i (pairs 0..15 by the column, 16..31 by the row). The
// loader permutes the q and k projection ROWS inside each head so pair i lands
// at dims (i, i + hd/2) - a relabel of the head's coordinates applied to q
// and k alike, which leaves every q.k dot product unchanged and v untouched -
// and this kernel then rotates (j, j + hd/2), the same addressing as 621.
//
// The angle table is [chip_rows][hd/2]: row t of a chip_rows-row group. The
// window blocks pass the 576-row window table (local coordinates 0..23, the
// same for every window); the global blocks the 5184-row table of the
// window-major grid at interpolated positions c * 24/72. Both are built once
// on the host in Meta's f32 order. A null table means no rope at all - the
// text tower's split (CLIP positions are learned, added before the tower).
__global__ void pd_sam3_qkv_split_rope_h_kernel(const __half* __restrict__ qkv,
                                                const float* __restrict__ bq,
                                                const float* __restrict__ bk,
                                                const float* __restrict__ bv,
                                                const float* __restrict__ cs,
                                                const float* __restrict__ sn,
                                                __half* __restrict__ q, __half* __restrict__ k,
                                                __half* __restrict__ v, uint32_t d, uint32_t hd,
                                                uint32_t chip_rows, float qs) {
    const uint32_t r = blockIdx.x;
    const size_t src = (size_t)r * 3u * d, dst = (size_t)r * d;
    const uint32_t half = hd / 2u, pairs = d / 2u;
    const uint32_t t = r % chip_rows;
    const bool rope = cs != nullptr;
    const float* cr = rope ? cs + (size_t)t * half : nullptr;
    const float* sr = rope ? sn + (size_t)t * half : nullptr;
    // one thread per pair: consecutive threads walk consecutive dims of a
    // head, so a warp's loads and stores coalesce (621's reasoning)
    for (uint32_t pp = threadIdx.x; pp < pairs; pp += blockDim.x) {
        const uint32_t h = pp / half, j = pp - h * half;
        const uint32_t e0 = h * hd + j, e1 = e0 + half;
        // no table: the identity rotation, which multiplies exactly
        const float c = rope ? cr[j] : 1.0f, s = rope ? sr[j] : 0.0f;
        const float q0 = __half2float(qkv[src + e0]) + bq[e0];
        const float q1 = __half2float(qkv[src + e1]) + bq[e1];
        const float k0 = __half2float(qkv[src + d + e0]) + bk[e0];
        const float k1 = __half2float(qkv[src + d + e1]) + bk[e1];
        q[dst + e0] = __float2half((q0 * c - q1 * s) * qs);
        q[dst + e1] = __float2half((q1 * c + q0 * s) * qs);
        k[dst + e0] = __float2half(k0 * c - k1 * s);
        k[dst + e1] = __float2half(k1 * c + k0 * s);
        v[dst + e0] = __float2half(__half2float(qkv[src + 2u * d + e0]) + bv[e0]);
        v[dst + e1] = __float2half(__half2float(qkv[src + 2u * d + e1]) + bv[e1]);
    }
}

PD_EXPORT
int pd_sam3_qkv_split_rope_h(const void* qkv, const void* bq, const void* bk, const void* bv,
                             const void* cs, const void* sn, void* q, void* k, void* v,
                             uint32_t d, uint32_t hd, uint32_t rows, uint32_t chip_rows,
                             float qscale, void* stream) {
    if (rows == 0 || d == 0) return 0;
    if (hd == 0 || (hd & 1u) != 0 || d % hd != 0 || chip_rows == 0)
        return cudaErrorInvalidValue;
    pd_sam3_qkv_split_rope_h_kernel<<<rows, 256, 0, (cudaStream_t)stream>>>(
        (const __half*)qkv, (const float*)bq, (const float*)bk, (const float*)bv,
        (const float*)cs, (const float*)sn, (__half*)q, (__half*)k, (__half*)v, d, hd,
        chip_rows, qscale);
    return pd_launch_status();
}

// 753: the tower's exit. The f32 residual in window-major rows -> the neck's
// f16 input as a raster (row = y * g + x), in one pass: the neck's first ops
// are GEMMs (they eat f16 anyway) and its 3x3 conv needs real neighbours, so
// this is the one place the order is undone. No ln_post in SAM 3: the block
// output IS the trunk output.
__global__ void pd_sam3_rows_to_raster_h_kernel(const float* __restrict__ x,
                                                __half* __restrict__ out, uint32_t g,
                                                uint32_t win, uint32_t d) {
    const uint32_t o = blockIdx.x;
    const uint32_t gg = g * g;
    const uint32_t b = o / gg, pix = o - b * gg;
    const uint32_t y = pix / g, xg = pix - y * g;
    const uint32_t nwx = g / win;
    const uint32_t wi = (y / win) * nwx + xg / win;
    const uint32_t r = wi * win * win + (y % win) * win + xg % win;
    const float* src = x + ((size_t)b * gg + r) * d;
    __half* dst = out + (size_t)o * d;
    for (uint32_t i = threadIdx.x; i < d; i += blockDim.x) dst[i] = __float2half(src[i]);
}

PD_EXPORT
int pd_sam3_rows_to_raster_h(const void* x, void* out, uint32_t pics, uint32_t g, uint32_t win,
                             uint32_t d, void* stream) {
    if (pics == 0 || g == 0 || d == 0) return 0;
    if (win == 0 || g % win != 0) return cudaErrorInvalidValue;
    pd_sam3_rows_to_raster_h_kernel<<<pics * g * g, 256, 0, (cudaStream_t)stream>>>(
        (const float*)x, (__half*)out, g, win, d);
    return pd_launch_status();
}
