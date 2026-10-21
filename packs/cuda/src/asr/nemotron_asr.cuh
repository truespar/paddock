// Nemotron 3.5 ASR (NVIDIA's cache-aware streaming FastConformer encoder,
// an RNN-T decoder and language-ID prompt conditioning) at F32 class:
// weights are the GGUF's own values (Q8_0 projections read as stored -
// exact int8 operands under their f16 block scales; the F16 convolutions
// and prediction embedding widened exactly), activations stay F32
// throughout - no operand is narrowed.
//
// A pass is the windows of several streams stacked by rows (segments, see
// PdNasrSeg); each window is encoder frames p0 .. p0 + R - 1 of its stream,
// chunk aligned (chunks of C = lookahead + 1 frames), host side
// gpu_model/nemotron_asr:
//
//   mel      = log-mel of a stream's PCM (the diarization frontend, 718), the
//              14 frames before the window's first included - the x8
//              subsampler is causal and reaches that far back, so a window
//              recomputes them instead of carrying three stage caches
//   s1       = ReLU(stem 3x3/2 conv of mel)                       (854)
//   s2, s3   = pointwise(depthwise 3x3/2 conv)  (855; the pointwise GEMMs
//              on the Kumo F32 one, 698, the F16 weights widened; their
//              ReLU rides the next read)
//   x        = flatten(ReLU(s3)) (856) . W_out + b   (853, the GEMM on the
//              stored weights, every segment's rows at once from here on)
//   24 x { x += 0.5 FF1(LN x); qkv = LN(x) . W_qkv; x += attention(q, each
//          stream's K/V rings + its window's rows, the relative table) . W_o
//          (857); K/V into the rings (858); x += pw2(SiLU(LN(dwconv(GLU(
//          pw1(LN x)))))) with the conv rings (859, 858); x += 0.5 FF2(LN x);
//          x = LN(x) }
//   proj     = (ReLU([x | onehot] . W_p1 + b) . W_p2 + b) . W_enc + b - the
//              prompt projector on the Kumo GEMM (each prompt's one-hot
//              column folded into a bias row at load, added per row, 861),
//              the joint's encoder half on 853
//   tokens   = the batched RNN-T greedy walk over proj (860), one
//              cooperative grid for up to 32 streams
//
// Every per-row quantity is a function of that row and its keys alone (the
// GEMM's fixed k walk, the attention's per-row key order, the convolution's
// per-row taps, the decoder's row-independent products), so a stream's
// outputs are the same bits however its audio was cut into windows and
// whatever streams shared its passes - streaming equals offline.
//
// Operation order follows the Transformers reference (huggingface/
// transformers nemotron3_5_asr and nemotron_asr_streaming, F32 PyTorch):
// LayerNorm (x - mean) * rstd * w + b, SiLU x / (1 + exp(-x)), sigmoid
// 1 / (1 + exp(-x)), the GLU a * sigmoid(b), the LSTM's c' = f c + i g and
// h' = o tanh(c').
//
// Plain CUDA; needs diarization.cuh's stored-weight GEMM and
// gemm/f32_qkv.cuh's pd_launch_status.

#define PD_NASR_D 1024u
#define PD_NASR_HEADS 8u
#define PD_NASR_HD 128u
#define PD_NASR_CTX 56u
#define PD_NASR_CONV 9u
#define PD_NASR_SUBC 256u
#define PD_NASR_PRED 640u
#define PD_NASR_PBLK (PD_NASR_PRED / 32u)
#define PD_NASR_VOCAB 13088u
#define PD_NASR_BLANK 13087u
#define PD_NASR_MAXSYM 10u
// the relative-key table: distances -13 .. 69 (a 14-frame chunk's widest
// spread), row d + 13
#define PD_NASR_DMIN 13
#define PD_NASR_REL 83u

// ------------------------------------------------------------------ 853 GEMM on the stored weights
// Slot 722's F32 class (the activation split three ways in bf16 against the
// exact int8 weight, each 32-deep block's integer sum scaled once by the
// file's own f16 scale, PD_DIAR_W_Q8H) with the conformer's epilogues: 0
// store, 2 accumulate, 6 SiLU, 7 accumulate half, 8 GLU (rows interleaved),
// 9 ReLU. The per-element sequence is the same in every tile shape, so the
// election below is shape-only and never moves a bit: a row's bits are its
// own whatever shares its pass - a streaming chunk's rows equal the file's.
//
// The lane's own election, measured with the weights streamed from DRAM as a
// window's 24 layers stream them (bench/nasr_gemm_gb10_bench.cu, GB10; every
// arm bit-identical): a live chunk is 1-14 rows and a batch of chunks tens,
// where 722's ladder (tuned for 300-684-row diarization windows) leaves the
// weight stream short of CTAs. Narrow N tiles on one or two warps and a deep
// cp.async ring for 16 rows and fewer; the long-K planes (FF down, the
// subsampler's 4352) at 16 x 32 on a 16-slot ring; 64 x 64 for the 4096-wide
// FF up from 17 rows. From 97 rows 722's own ladder holds (FF up keeps 64 x
// 64 to 128).
template <uint32_t BM, uint32_t BN, uint32_t WM, uint32_t WN, uint32_t ST>
static void pd_nasr_go(const float* x, const uint8_t* w, const void* scale, const float* bias, float* y, uint32_t K,
                       uint32_t N, uint32_t M, uint32_t mode, cudaStream_t s) {
    pd_diar_gemm_go<BM, BN, WM, WN, ST, PD_DIAR_W_Q8H, 3u>(x, w, scale, bias, y, nullptr, nullptr, K, N, M, mode, s);
}

static void pd_nasr_gemm_elect(const float* x, const uint8_t* w, const void* scale, const float* bias, float* y,
                               uint32_t K, uint32_t N, uint32_t M, uint32_t mode, cudaStream_t s) {
    // the GLU epilogue pairs adjacent columns: an 8-wide tile still holds
    // whole pairs, so every arm serves it
    const bool long_k = K >= 4096u;
    if (M <= 16u) {
        if (long_k) return pd_nasr_go<16u, 32u, 1u, 2u, 16u>(x, w, scale, bias, y, K, N, M, mode, s);
        if (N >= 3072u || N < 1024u) return pd_nasr_go<16u, 8u, 1u, 1u, 8u>(x, w, scale, bias, y, K, N, M, mode, s);
        if (N >= 2048u) return pd_nasr_go<16u, 32u, 1u, 2u, 8u>(x, w, scale, bias, y, K, N, M, mode, s);
        return pd_nasr_go<16u, 16u, 1u, 2u, 8u>(x, w, scale, bias, y, K, N, M, mode, s);
    }
    if (M <= 32u) {
        if (long_k) return pd_nasr_go<16u, 16u, 1u, 2u, 8u>(x, w, scale, bias, y, K, N, M, mode, s);
        if (N >= 4096u) return pd_nasr_go<64u, 64u, 4u, 1u, 3u>(x, w, scale, bias, y, K, N, M, mode, s);
        if (N >= 2048u) return pd_nasr_go<16u, 64u, 1u, 4u, 4u>(x, w, scale, bias, y, K, N, M, mode, s);
        return pd_nasr_go<16u, 16u, 1u, 2u, 8u>(x, w, scale, bias, y, K, N, M, mode, s);
    }
    if (M <= 64u) {
        if (long_k) return pd_nasr_go<16u, 64u, 1u, 4u, 8u>(x, w, scale, bias, y, K, N, M, mode, s);
        if (N >= 4096u) return pd_nasr_go<64u, 64u, 4u, 1u, 3u>(x, w, scale, bias, y, K, N, M, mode, s);
        if (N >= 3072u) return pd_nasr_go<64u, 32u, 4u, 1u, 3u>(x, w, scale, bias, y, K, N, M, mode, s);
        if (N >= 1024u) return pd_nasr_go<16u, 64u, 1u, 4u, 4u>(x, w, scale, bias, y, K, N, M, mode, s);
        return pd_nasr_go<16u, 32u, 1u, 2u, 8u>(x, w, scale, bias, y, K, N, M, mode, s);
    }
    if (M <= 96u) {
        if (long_k) return pd_nasr_go<16u, 64u, 1u, 4u, 8u>(x, w, scale, bias, y, K, N, M, mode, s);
        if (N >= 3072u) return pd_nasr_go<64u, 64u, 4u, 1u, 2u>(x, w, scale, bias, y, K, N, M, mode, s);
        if (N < 1024u) return pd_nasr_go<16u, 64u, 1u, 4u, 8u>(x, w, scale, bias, y, K, N, M, mode, s);
        return pd_nasr_go<16u, 64u, 1u, 4u, 4u>(x, w, scale, bias, y, K, N, M, mode, s);
    }
    // 97-128 rows: 722's ladder is 5-10% ahead of these arms everywhere but
    // the 4096-wide FF up
    if (M <= 128u && N >= 4096u) return pd_nasr_go<64u, 64u, 4u, 1u, 2u>(x, w, scale, bias, y, K, N, M, mode, s);
    pd_diar_gemm_elect<PD_DIAR_W_Q8H, 3u>(x, w, (const float*)scale, bias, y, nullptr, nullptr, K, N, M, mode, s);
}

PD_EXPORT
int pd_nasr_gemm(const void* x, const void* w, const void* scale, const void* bias, void* y, uint32_t K,
                 uint32_t N, uint32_t M, uint32_t mode, void* stream) {
    if (M == 0 || N == 0) return 0;
    const bool known = mode == 0u || mode == PD_KUMO_RESID || mode == PD_NASR_SILU ||
                       mode == PD_NASR_HALF_RESID || mode == PD_NASR_GLU || mode == PD_NASR_RELU;
    // 16 B cp.async rows: K a multiple of the k tile (= the Q8_0 block), N of
    // a scale chunk (8 halves)
    if (!known || K == 0 || K % 32u || N % 8u || !scale || ((uintptr_t)x & 15u) || ((uintptr_t)w & 15u) ||
        ((uintptr_t)scale & 15u) || (M + 15u) / 16u > 65535u)
        return (int)cudaErrorInvalidValue;
    pd_nasr_gemm_elect((const float*)x, (const uint8_t*)w, scale, (const float*)bias, (float*)y, K, N, M, mode,
                       (cudaStream_t)stream);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 854 subsampler stem
// out [count][65][256]: the stem's rows first .. first + count - 1, row t
// ReLU(sum over kt, kf of w[c][kt][kf] mel[2t - 2 + kt][2f - 2 + kf] +
// b[c]) - causal in time (left padding 2), the frequency axis padded (2, 1).
// `mel` holds absolute frames base .. base + rows - 1; a frame before 0, at
// or past `valid` (the recording's end) or outside the plane reads zero.
// The taps sum time-major in order, the bias once at the end.
__global__ void pd_nasr_sub_stem_kernel(const float* __restrict__ mel, int64_t base, uint32_t rows,
                                        int64_t valid, const float* __restrict__ w,
                                        const float* __restrict__ b, float* __restrict__ out,
                                        int64_t first, uint32_t count) {
    PD_PDL_ARM();
    const uint64_t i = (uint64_t)blockIdx.x * 256u + threadIdx.x;
    if (i >= (uint64_t)count * 65u * PD_NASR_SUBC) return;
    const uint32_t c = (uint32_t)(i % PD_NASR_SUBC), f = (uint32_t)(i / PD_NASR_SUBC % 65u);
    const int64_t t = first + (int64_t)(i / (PD_NASR_SUBC * 65u));
    float acc = 0.f;
#pragma unroll
    for (int kt = 0; kt < 3; ++kt) {
        const int64_t r = 2 * t - 2 + kt;
        if (r < 0 || r >= valid || r < base || r >= base + (int64_t)rows) continue;
        const float* row = mel + (size_t)(r - base) * 128u;
#pragma unroll
        for (int kf = 0; kf < 3; ++kf) {
            const int q = 2 * (int)f - 2 + kf;
            if (q < 0 || q >= 128) continue;
            acc = fmaf(w[c * 9u + kt * 3 + kf], row[q], acc);
        }
    }
    out[i] = fmaxf(__fadd_rn(acc, b[c]), 0.f);
}

PD_EXPORT
int pd_nasr_sub_stem(const void* mel, int64_t base, uint32_t rows, int64_t valid, const void* w, const void* b,
                     void* out, int64_t first, uint32_t count, void* stream) {
    if (count == 0) return 0;
    // the window's first stem row reads mel frame 2 first - 2: the plane
    // must hold it (or it must be before the recording)
    if (first < 0 || base < 0 || (base > 0 && base > 2 * first - 2)) return (int)cudaErrorInvalidValue;
    const uint64_t n = (uint64_t)count * 65u * PD_NASR_SUBC;
    pd_pdl_go(pd_nasr_sub_stem_kernel, dim3((unsigned)((n + 255u) / 256u)), dim3(256), 0, (cudaStream_t)stream,
              (const float*)mel, base, rows, valid, (const float*)w, (const float*)b, (float*)out, first, count);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 855 subsampler depthwise stage
// out [count][fout][256], fout = fin / 2 + 1: sum over kt, kf of w[c][kt][kf]
// in[2t - 2 + kt][2f - 2 + kf][c] + b[c], the depthwise 3x3 stride-2 conv of
// a separable stage (its pointwise half is a GEMM). `in` [rows][fin][256]
// holds absolute rows base ..; outside the plane, before 0 or at `valid`
// reads zero. relu_in applies ReLU to every value read - the previous
// stage's pointwise output, whose activation rides here.
__global__ void pd_nasr_sub_dw_kernel(const float* __restrict__ in, int64_t base, uint32_t rows, int64_t valid,
                                      uint32_t fin, const float* __restrict__ w, const float* __restrict__ b,
                                      float* __restrict__ out, int64_t first, uint32_t count, uint32_t relu_in) {
    PD_PDL_ARM();
    const uint32_t fout = fin / 2u + 1u;
    const uint64_t i = (uint64_t)blockIdx.x * 256u + threadIdx.x;
    if (i >= (uint64_t)count * fout * PD_NASR_SUBC) return;
    const uint32_t c = (uint32_t)(i % PD_NASR_SUBC), f = (uint32_t)(i / PD_NASR_SUBC % fout);
    const int64_t t = first + (int64_t)(i / ((uint64_t)PD_NASR_SUBC * fout));
    float acc = 0.f;
#pragma unroll
    for (int kt = 0; kt < 3; ++kt) {
        const int64_t r = 2 * t - 2 + kt;
        if (r < 0 || r >= valid || r < base || r >= base + (int64_t)rows) continue;
        const float* row = in + (size_t)(r - base) * fin * PD_NASR_SUBC;
#pragma unroll
        for (int kf = 0; kf < 3; ++kf) {
            const int q = 2 * (int)f - 2 + kf;
            if (q < 0 || q >= (int)fin) continue;
            float v = row[(size_t)q * PD_NASR_SUBC + c];
            if (relu_in) v = fmaxf(v, 0.f);
            acc = fmaf(w[c * 9u + kt * 3 + kf], v, acc);
        }
    }
    out[i] = __fadd_rn(acc, b[c]);
}

PD_EXPORT
int pd_nasr_sub_dw(const void* in, int64_t base, uint32_t rows, int64_t valid, uint32_t fin, const void* w,
                   const void* b, void* out, int64_t first, uint32_t count, uint32_t relu_in, void* stream) {
    if (count == 0) return 0;
    if (fin < 2u || first < 0 || base < 0 || (base > 0 && base > 2 * first - 2))
        return (int)cudaErrorInvalidValue;
    const uint64_t n = (uint64_t)count * (fin / 2u + 1u) * PD_NASR_SUBC;
    pd_pdl_go(pd_nasr_sub_dw_kernel, dim3((unsigned)((n + 255u) / 256u)), dim3(256), 0, (cudaStream_t)stream,
              (const float*)in, base, rows, valid, fin, (const float*)w, (const float*)b, (float*)out, first, count,
              relu_in);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 856 subsampler flatten
// out [rows][256 * 17] = ReLU(in [rows][17][256]) channel-major - the
// reference's (batch, channel, time, bin) -> (batch, time, channel * bin).
__global__ void pd_nasr_sub_flatten_kernel(const float* __restrict__ in, float* __restrict__ out, uint32_t rows) {
    PD_PDL_ARM();
    const uint64_t i = (uint64_t)blockIdx.x * 256u + threadIdx.x;
    if (i >= (uint64_t)rows * 17u * PD_NASR_SUBC) return;
    const uint32_t j = (uint32_t)(i % (17u * PD_NASR_SUBC)), c = j / 17u, f = j % 17u;
    const uint64_t t = i / (17u * PD_NASR_SUBC);
    out[i] = fmaxf(in[(t * 17u + f) * PD_NASR_SUBC + c], 0.f);
}

PD_EXPORT
int pd_nasr_sub_flatten(const void* in, void* out, uint32_t rows, void* stream) {
    if (rows == 0) return 0;
    const uint64_t n = (uint64_t)rows * 17u * PD_NASR_SUBC;
    pd_pdl_go(pd_nasr_sub_flatten_kernel, dim3((unsigned)((n + 255u) / 256u)), dim3(256), 0, (cudaStream_t)stream,
              (const float*)in, (float*)out, rows);
    return pd_launch_status();
}

// ------------------------------------------------------------------ pass segments
// A pass is the windows of several streams stacked by rows: segment i is
// rows row0 .. row0 + rows - 1 of every plane, absolute encoder frames p0 ..
// of its stream, chunks of `chunk` frames (p0 a whole number of them), its
// language prompt, and its own K/V and convolution rings. Segments are in
// row order; `chunk0` counts the pass's chunks before this one.
struct PdNasrSeg {
    int64_t p0;
    float* kv;    // [24][2][56][1024]: each layer's K then V ring
    float* conv;  // [24][8][1024]: each layer's GLU-row ring
    uint32_t row0;
    uint32_t rows;
    uint32_t chunk;
    uint32_t chunk0;
    uint32_t prompt;
    uint32_t pad;
};

#define PD_NASR_KV_LAYER (2u * PD_NASR_CTX * PD_NASR_D)
#define PD_NASR_CONV_LAYER ((PD_NASR_CONV - 1u) * PD_NASR_D)

// the segment holding pass row r / pass chunk c (segments sorted by both)
static __device__ __forceinline__ uint32_t pd_nasr_seg_of_row(const PdNasrSeg* __restrict__ segs, uint32_t n,
                                                              uint32_t r) {
    uint32_t lo = 0, hi = n - 1u;
    while (lo < hi) {
        const uint32_t mid = (lo + hi + 1u) / 2u;
        if (segs[mid].row0 <= r) lo = mid;
        else hi = mid - 1u;
    }
    return lo;
}

static __device__ __forceinline__ uint32_t pd_nasr_seg_of_chunk(const PdNasrSeg* __restrict__ segs, uint32_t n,
                                                                uint32_t c) {
    uint32_t lo = 0, hi = n - 1u;
    while (lo < hi) {
        const uint32_t mid = (lo + hi + 1u) / 2u;
        if (segs[mid].chunk0 <= c) lo = mid;
        else hi = mid - 1u;
    }
    return lo;
}

// ------------------------------------------------------------------ 857 attention
// Transformer-XL relative-position attention under the chunked_limited
// mask, 8 heads of 128, over every segment of a pass. Every row of a chunk
// has the same keys, so a CTA is one (chunk, head): the chunk's rows are
// absolute frames c0 .. c0 + nq - 1 of its stream (chunk c0 / C), its keys
// frames [max(0, (c0 / C - 56 / C) C), c0 + nq) in order - the cached ones
// (before the segment's p0) from the layer's rings at slot j % 56, the
// window's own from the qkv rows (q | k at +1024 | v at +2048). score =
// ((q + u) . k) s + ((q + v) . P[p - j]) s, P the layer's relative-key table
// (row distance + 13), s = 128^-0.5; softmax over the keys; the weighted
// values.
//
// One shared region at a 129-float pitch (consecutive rows on consecutive
// banks) holds the chunk's K rows, then the distance rows of P it spans,
// then its V rows: a thread a (row, key) pair walks each dot over the 128
// channels in order (q + u against k first, kept, then q + v against P);
// a warp a row closes the softmax - its lane l the keys l, l + 32, l + 64,
// the max and the sum on a fixed xor tree; a thread a (row, channel) sums e
// v over the keys in order and divides by the sum once. Every quantity is a
// function of its row and keys alone - the same bits in any window or pass.
#define PD_NASR_PITCH 129u
#define PD_NASR_KMAX 72u
#define PD_NASR_ATT_SMEM \
    ((PD_NASR_REL * PD_NASR_PITCH + 2u * 14u * PD_NASR_HD + 14u * PD_NASR_KMAX + 16u) * 4u)

__global__ void __launch_bounds__(256)
pd_nasr_attention_kernel(const float* __restrict__ qkv, const PdNasrSeg* __restrict__ segs, uint32_t nseg,
                         uint32_t layer, const float* __restrict__ rel, const float* __restrict__ bu,
                         const float* __restrict__ bv, float* __restrict__ out, float scale) {
    extern __shared__ __align__(16) float nasr_asm[];
    float* kv = nasr_asm;                                  // [83][129]
    float* qs = kv + PD_NASR_REL * PD_NASR_PITCH;          // [2][14][128]: q + u, q + v
    float* ss = qs + 2u * 14u * PD_NASR_HD;                // [14][72]: scores, then e
    float* sums = ss + 14u * PD_NASR_KMAX;                 // [14]
    PD_PDL_ARM();
    const uint32_t h = blockIdx.y, t = threadIdx.x, lane = t & 31u, warp = t >> 5;
    const PdNasrSeg sg = segs[pd_nasr_seg_of_chunk(segs, nseg, blockIdx.x)];
    const uint32_t C = sg.chunk;
    const int64_t p0 = sg.p0, c0 = p0 + (int64_t)(blockIdx.x - sg.chunk0) * C;
    const uint32_t nq = (uint32_t)(p0 + sg.rows - c0 < (int64_t)C ? p0 + sg.rows - c0 : (int64_t)C);
    const int64_t ci = c0 / C, lc = PD_NASR_CTX / C;
    const int64_t j0 = (ci - lc) * (int64_t)C > 0 ? (ci - lc) * (int64_t)C : 0;
    const uint32_t nk = (uint32_t)(c0 + nq - j0);
    const float* kring = sg.kv + (size_t)layer * PD_NASR_KV_LAYER;
    const float* vring = kring + PD_NASR_CTX * PD_NASR_D;
    // pass row of absolute frame j of this stream
    auto row = [&](int64_t j) { return (size_t)sg.row0 + (size_t)(j - p0); };
    for (uint32_t i = t; i < nq * PD_NASR_HD; i += 256u) {
        const uint32_t r = i / PD_NASR_HD, d = i % PD_NASR_HD;
        const float q = qkv[row(c0 + r) * 3u * PD_NASR_D + h * PD_NASR_HD + d];
        qs[r * PD_NASR_HD + d] = __fadd_rn(q, bu[h * PD_NASR_HD + d]);
        qs[(14u + r) * PD_NASR_HD + d] = __fadd_rn(q, bv[h * PD_NASR_HD + d]);
    }
    // the keys' K rows, then V rows (`v` 1)
    auto stage = [&](uint32_t v) {
        for (uint32_t i = t; i < nk * PD_NASR_HD; i += 256u) {
            const uint32_t kk = i / PD_NASR_HD, d = i % PD_NASR_HD;
            const int64_t j = j0 + kk;
            const float* src = j < p0 ? (v ? vring : kring) + (size_t)(j % PD_NASR_CTX) * PD_NASR_D
                                      : qkv + row(j) * 3u * PD_NASR_D + (1u + v) * PD_NASR_D;
            kv[kk * PD_NASR_PITCH + d] = src[h * PD_NASR_HD + d];
        }
    };
    stage(0u);
    __syncthreads();
    for (uint32_t i = t; i < nq * nk; i += 256u) {
        const uint32_t r = i / nk, kk = i % nk;
        const float* a = qs + r * PD_NASR_HD;
        const float* k = kv + kk * PD_NASR_PITCH;
        float ac = 0.f;
#pragma unroll 8
        for (uint32_t d = 0; d < PD_NASR_HD; ++d) ac = fmaf(a[d], k[d], ac);
        ss[r * PD_NASR_KMAX + kk] = ac;
    }
    __syncthreads();
    // P's rows for the distances this chunk spans: -(nq - 1) .. nk - 1, row
    // r = distance dlo + r
    const int64_t dlo = -(int64_t)(nq - 1u);
    const uint32_t np = nq + nk - 1u;
    for (uint32_t i = t; i < np * PD_NASR_HD; i += 256u) {
        const uint32_t r = i / PD_NASR_HD, d = i % PD_NASR_HD;
        kv[r * PD_NASR_PITCH + d] = rel[(size_t)(dlo + r + PD_NASR_DMIN) * PD_NASR_D + h * PD_NASR_HD + d];
    }
    __syncthreads();
    for (uint32_t i = t; i < nq * nk; i += 256u) {
        const uint32_t r = i / nk, kk = i % nk;
        const float* c = qs + (14u + r) * PD_NASR_HD;
        const float* pr = kv + (uint32_t)((c0 + r) - (j0 + kk) - dlo) * PD_NASR_PITCH;
        float bd = 0.f;
#pragma unroll 8
        for (uint32_t d = 0; d < PD_NASR_HD; ++d) bd = fmaf(c[d], pr[d], bd);
        float* o = ss + r * PD_NASR_KMAX + kk;
        *o = __fadd_rn(__fmul_rn(*o, scale), __fmul_rn(bd, scale));
    }
    __syncthreads();
    for (uint32_t r = warp; r < nq; r += 8u) {
        float* rw = ss + r * PD_NASR_KMAX;
        float sc[3];
#pragma unroll
        for (uint32_t s = 0; s < 3u; ++s) {
            const uint32_t kk = lane + 32u * s;
            sc[s] = kk < nk ? rw[kk] : -INFINITY;
        }
        float m = fmaxf(fmaxf(sc[0], sc[1]), sc[2]);
#pragma unroll
        for (uint32_t o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        float sum = 0.f;
#pragma unroll
        for (uint32_t s = 0; s < 3u; ++s) {
            const uint32_t kk = lane + 32u * s;
            const float e = kk < nk ? expf(__fsub_rn(sc[s], m)) : 0.f;
            sum = __fadd_rn(sum, e);
            if (kk < nk) rw[kk] = e;
        }
#pragma unroll
        for (uint32_t o = 16; o > 0; o >>= 1) sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, o));
        if (lane == 0) sums[r] = sum;
    }
    stage(1u);
    __syncthreads();
    for (uint32_t i = t; i < nq * PD_NASR_HD; i += 256u) {
        const uint32_t r = i / PD_NASR_HD, d = i % PD_NASR_HD;
        const float* e = ss + r * PD_NASR_KMAX;
        float acc = 0.f;
        for (uint32_t kk = 0; kk < nk; ++kk) acc = fmaf(e[kk], kv[kk * PD_NASR_PITCH + d], acc);
        out[row(c0 + r) * PD_NASR_D + h * PD_NASR_HD + d] = __fdiv_rn(acc, sums[r]);
    }
}

PD_EXPORT
int pd_nasr_attention(const void* qkv, const void* segs, uint32_t nseg, uint32_t chunks, uint32_t layer,
                      const void* rel, const void* bu, const void* bv, void* out, float scale, void* stream) {
    if (nseg == 0 || chunks == 0) return 0;
    if (layer >= 24u) return (int)cudaErrorInvalidValue;
    static bool set = false;
    if (!set) {
        cudaFuncSetAttribute(pd_nasr_attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)PD_NASR_ATT_SMEM);
        set = true;
    }
    pd_pdl_go(pd_nasr_attention_kernel, dim3(chunks, PD_NASR_HEADS), dim3(256), PD_NASR_ATT_SMEM,
              (cudaStream_t)stream, (const float*)qkv, (const PdNasrSeg*)segs, nseg, layer, (const float*)rel,
              (const float*)bu, (const float*)bv, (float*)out, scale);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 858 ring commit
// Each segment's last min(rows, slots) rows of `src` (row stride ld, from
// column col0 + 1024 z) into its layer's ring z of `which0 + z` (0 the K
// ring, 1 the V ring, 2 the convolution's), slot frame % slots - after the
// pass that read the rings.
__global__ void pd_nasr_ring_put_kernel(const float* __restrict__ src, uint32_t ld, uint32_t col0,
                                        const PdNasrSeg* __restrict__ segs, uint32_t layer, uint32_t which0) {
    PD_PDL_ARM();
    const PdNasrSeg sg = segs[blockIdx.y];
    const uint32_t which = which0 + blockIdx.z;
    const uint32_t slots = which < 2u ? PD_NASR_CTX : PD_NASR_CONV - 1u;
    const uint32_t keep = sg.rows < slots ? sg.rows : slots;
    const uint64_t i = (uint64_t)blockIdx.x * 256u + threadIdx.x;
    if (i >= (uint64_t)keep * PD_NASR_D) return;
    const uint32_t r = sg.rows - keep + (uint32_t)(i / PD_NASR_D), c = (uint32_t)(i % PD_NASR_D);
    float* ring = which < 2u ? sg.kv + (size_t)layer * PD_NASR_KV_LAYER + which * PD_NASR_CTX * PD_NASR_D
                             : sg.conv + (size_t)layer * PD_NASR_CONV_LAYER;
    ring[(size_t)((sg.p0 + r) % slots) * PD_NASR_D + c] =
        src[(size_t)(sg.row0 + r) * ld + col0 + blockIdx.z * PD_NASR_D + c];
}

PD_EXPORT
int pd_nasr_ring_put(const void* src, uint32_t ld, uint32_t col0, const void* segs, uint32_t nseg, uint32_t layer,
                     uint32_t which0, uint32_t count, void* stream) {
    if (nseg == 0 || count == 0) return 0;
    if (layer >= 24u || which0 + count > 3u || ld < col0 + count * PD_NASR_D) return (int)cudaErrorInvalidValue;
    const uint32_t blocks = (PD_NASR_CTX * PD_NASR_D + 255u) / 256u;
    pd_pdl_go(pd_nasr_ring_put_kernel, dim3(blocks, nseg, count), dim3(256), 0, (cudaStream_t)stream,
              (const float*)src, ld, col0, (const PdNasrSeg*)segs, layer, which0);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 859 convolution module core
// One CTA a pass row (absolute frame p of its stream), thread t owning
// channels t + 256 k: x[c] = sum_{k < 9} w[c][k] g[p - 8 + k][c] - the causal
// depthwise conv over the GLU rows, a frame before the segment's p0 from the
// layer's 8-slot ring (slot j % 8), before 0 zero - then LayerNorm over the
// 1024 channels (the mean, then the biased variance of the centered values,
// eps 1e-5, each a thread's partials summed in order and closed on a fixed
// tree) and SiLU.
static __device__ __forceinline__ float pd_nasr_block_sum(float v, float* red) {
#pragma unroll
    for (uint32_t o = 16; o > 0; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, o));
    if ((threadIdx.x & 31u) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float s = 0.f;
#pragma unroll
    for (uint32_t w = 0; w < 8u; ++w) s = __fadd_rn(s, red[w]);
    __syncthreads();
    return s;
}

__global__ void __launch_bounds__(256)
pd_nasr_dwconv_kernel(const float* __restrict__ g, const PdNasrSeg* __restrict__ segs, uint32_t nseg,
                      uint32_t layer, const float* __restrict__ w, const float* __restrict__ nw,
                      const float* __restrict__ nb, float* __restrict__ out) {
    __shared__ float red[8];
    PD_PDL_ARM();
    const uint32_t i = blockIdx.x, t = threadIdx.x;
    const PdNasrSeg sg = segs[pd_nasr_seg_of_row(segs, nseg, i)];
    const int64_t p = sg.p0 + (i - sg.row0);
    const float* ring = sg.conv + (size_t)layer * PD_NASR_CONV_LAYER;
    float x[4];
#pragma unroll
    for (uint32_t k = 0; k < 4u; ++k) {
        const uint32_t c = t + 256u * k;
        float acc = 0.f;
#pragma unroll
        for (uint32_t tap = 0; tap < PD_NASR_CONV; ++tap) {
            const int64_t j = p - (int64_t)(PD_NASR_CONV - 1u) + tap;
            if (j < 0) continue;
            const float v = j < sg.p0 ? ring[(size_t)(j % (PD_NASR_CONV - 1u)) * PD_NASR_D + c]
                                      : g[((size_t)sg.row0 + (size_t)(j - sg.p0)) * PD_NASR_D + c];
            acc = fmaf(w[c * PD_NASR_CONV + tap], v, acc);
        }
        x[k] = acc;
    }
    float s = 0.f;
#pragma unroll
    for (uint32_t k = 0; k < 4u; ++k) s = __fadd_rn(s, x[k]);
    const float mean = __fdiv_rn(pd_nasr_block_sum(s, red), (float)PD_NASR_D);
    float q = 0.f;
#pragma unroll
    for (uint32_t k = 0; k < 4u; ++k) {
        x[k] = __fsub_rn(x[k], mean);
        q = fmaf(x[k], x[k], q);
    }
    const float rstd = __frsqrt_rn(__fadd_rn(__fdiv_rn(pd_nasr_block_sum(q, red), (float)PD_NASR_D), 1e-5f));
#pragma unroll
    for (uint32_t k = 0; k < 4u; ++k) {
        const uint32_t c = t + 256u * k;
        const float y = fmaf(__fmul_rn(x[k], rstd), nw[c], nb[c]);
        out[(size_t)i * PD_NASR_D + c] = pd_glu_act<PD_ACT_SILU>(y);
    }
}

PD_EXPORT
int pd_nasr_dwconv(const void* g, const void* segs, uint32_t nseg, uint32_t rows, uint32_t layer, const void* w,
                   const void* nw, const void* nb, void* out, void* stream) {
    if (rows == 0 || nseg == 0) return 0;
    if (layer >= 24u) return (int)cudaErrorInvalidValue;
    pd_pdl_go(pd_nasr_dwconv_kernel, dim3(rows), dim3(256), 0, (cudaStream_t)stream, (const float*)g,
              (const PdNasrSeg*)segs, nseg, layer, (const float*)w, (const float*)nw, (const float*)nb,
              (float*)out);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 861 prompt bias
// y[r][n] = ReLU(y[r][n] + table[prompt][n]) for pass row r, the prompt its
// segment's: the prompt projector's first layer with each stream's one-hot
// column folded into its bias at load (the product ran without one).
__global__ void pd_nasr_prompt_relu_kernel(float* __restrict__ y, uint32_t n, const PdNasrSeg* __restrict__ segs,
                                           uint32_t nseg, const float* __restrict__ table) {
    PD_PDL_ARM();
    const uint32_t r = blockIdx.x;
    const float* b = table + (size_t)segs[pd_nasr_seg_of_row(segs, nseg, r)].prompt * n;
    float* row = y + (size_t)r * n;
    for (uint32_t c = threadIdx.x; c < n; c += 256u) row[c] = fmaxf(__fadd_rn(row[c], b[c]), 0.f);
}

PD_EXPORT
int pd_nasr_prompt_relu(void* y, uint32_t n, uint32_t rows, const void* segs, uint32_t nseg, const void* table,
                        void* stream) {
    if (rows == 0 || nseg == 0) return 0;
    pd_pdl_go(pd_nasr_prompt_relu_kernel, dim3(rows), dim3(256), 0, (cudaStream_t)stream, (float*)y, n,
              (const PdNasrSeg*)segs, nseg, (const float*)table);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 860 RNN-T greedy walk
// The prediction network, the joint and the greedy rule of up to 32 streams
// over their windows' projected frames, on one cooperative grid (one CTA an
// SM, 8 warps each), every product on the tensor cores at the house F32
// class (the activation split three ways in bf16 against the exact int8
// weight, a 32-deep block's integer sum scaled once by its f16 scale).
//
// A step scores up to 32 (stream, frame) rows at once. The decoder state
// changes only when a token is emitted, so a stream's next frames are scored
// against the same state its current one is: each active stream takes
// 32 / (active streams) consecutive frames (as many as remain). Every CTA
// reduces the grid's per-row argmax partials in CTA order (highest logit,
// lowest index on a tie - torch's argmax) and scans each stream's rows in
// frame order exactly as the sequential walk would: a blank advances the
// frame (and resets the frame's symbol count), the first token is emitted at
// its frame, every row after it is discarded; ten tokens at one frame force
// the advance. A row's logits depend on that row alone, so the tokens are
// those of the one-stream, one-frame walk whatever shares a step.
//
// Streams that emitted step their network on the token (16 at a time): the
// LSTM's gate rows are interleaved at load so an 8-row tile is two units'
// (i, f, g, o) and a thread pair holds one unit's four gates - gates =
// (W_ih x + b_ih) + (W_hh h + b_hh), c' = f c + i g, h' = o tanh(c') - then
// the projection g = W_p h + b_p; a grid barrier after each.
//
// State (`st`): h [2][2][640] and c [2][2][640] ping-pong halves (cur picks
// the live one), g [640], cur, count. A job's `prime` first steps the
// network on the blank from the zeroed state - the decoder start. A job's
// tokens and their absolute frames go to out0 .. (at most cap), their number
// to its state's count and the job's.
struct PdNasrDecW {
    const int8_t* head_q;
    const __half* head_s;
    const float* head_b;
    const int8_t* ih_q[2];
    const __half* ih_s[2];
    const float* ih_b[2];
    const int8_t* hh_q[2];
    const __half* hh_s[2];
    const float* hh_b[2];
    const int8_t* pred_q;
    const __half* pred_s;
    const float* pred_b;
    const __half* embed;
};

struct PdNasrDecState {
    float h[2][2][PD_NASR_PRED];
    float c[2][2][PD_NASR_PRED];
    float g[PD_NASR_PRED];
    uint32_t cur;
    uint32_t count;
};

struct PdNasrJob {
    PdNasrDecState* st;
    int64_t frame0;
    uint32_t row0;
    uint32_t rows;
    uint32_t out0;
    uint32_t cap;
    uint32_t prime;
    uint32_t count;  // out: the tokens this window emitted
};

#define PD_NASR_JOBS 32u
#define PD_NASR_ZP 648u  // the operand rows' pitch: the 40-float pitch's bank pattern
#define PD_NASR_DEC_SMEM (32u * PD_NASR_ZP * 4u)

// One 8-row weight tile against up to two 16-row operand tiles (rows 0-15,
// 16-31 of `zs`), the whole 640-deep walk: per 32-deep block the k16 pairs
// hi, mid, lo then a drain fma(d, acc, fac) - `fac[mt][e]` the accumulator
// fragments (rows gr, gr + 8 of tile mt; columns 2 t4, 2 t4 + 1).
template <uint32_t MT>
static __device__ __forceinline__ void pd_nasr_tile(const float* __restrict__ zs, uint32_t a_row0,
                                                    const int8_t* __restrict__ q, const __half* __restrict__ s,
                                                    uint32_t rows, uint32_t n0, uint32_t lane,
                                                    float (&fac)[MT][4]) {
    const uint32_t gr = lane >> 2, t4 = lane & 3u;
    const int8_t* qr = q + (size_t)(n0 + gr) * PD_NASR_PRED;
#pragma unroll
    for (uint32_t mt = 0; mt < MT; ++mt)
#pragma unroll
        for (uint32_t e = 0; e < 4u; ++e) fac[mt][e] = 0.f;
#pragma unroll 1
    for (uint32_t blk = 0; blk < PD_NASR_PBLK; ++blk) {
        const uint32_t k0 = blk * 32u;
        uint32_t b0[2], b1[2];
#pragma unroll
        for (uint32_t kk = 0; kk < 2u; ++kk) {
            b0[kk] = pd_diar_q2(*reinterpret_cast<const uint16_t*>(qr + k0 + kk * 16u + 2u * t4));
            b1[kk] = pd_diar_q2(*reinterpret_cast<const uint16_t*>(qr + k0 + kk * 16u + 8u + 2u * t4));
        }
        const float d0 = __half2float(s[(size_t)blk * rows + n0 + 2u * t4]);
        const float d1 = __half2float(s[(size_t)blk * rows + n0 + 2u * t4 + 1u]);
#pragma unroll
        for (uint32_t mt = 0; mt < MT; ++mt) {
            float acc[4];
#pragma unroll
            for (uint32_t kk = 0; kk < 2u; ++kk) {
                uint32_t ah[4], am[4], al[4];
                const float* ar = zs + (size_t)(a_row0 + mt * 16u + gr) * PD_NASR_ZP + k0 + kk * 16u + 2u * t4;
#pragma unroll
                for (uint32_t e = 0; e < 4u; ++e) {
                    const float2 v =
                        *reinterpret_cast<const float2*>(ar + (e & 1u) * 8u * PD_NASR_ZP + (e >> 1) * 8u);
                    pd_diar_split(v, ah[e], am[e], al[e]);
                }
                if (kk == 0u)
                    pd_diar_mma0(acc, ah, b0[kk], b1[kk]);
                else
                    pd_diar_mma(acc, ah, b0[kk], b1[kk]);
                pd_diar_mma(acc, am, b0[kk], b1[kk]);
                pd_diar_mma(acc, al, b0[kk], b1[kk]);
            }
#pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) fac[mt][e] = __fmaf_rn(e & 1u ? d1 : d0, acc[e], fac[mt][e]);
        }
    }
}

static __device__ __forceinline__ float pd_nasr_sigmoid(float x) {
    return __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-x)));
}

__global__ void __launch_bounds__(256)
pd_nasr_greedy_kernel(PdNasrDecW W, const float* __restrict__ proj, PdNasrJob* __restrict__ jobs, uint32_t B,
                      uint32_t* __restrict__ tokens, uint32_t* __restrict__ frames, float* __restrict__ part_v,
                      uint32_t* __restrict__ part_i) {
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();
    extern __shared__ __align__(16) float zs[];  // [32][648]
    __shared__ uint32_t jt[PD_NASR_JOBS], jsym[PD_NASR_JOBS], jcur[PD_NASR_JOBS], jcnt[PD_NASR_JOBS];
    __shared__ uint32_t sjob[32], sframe[32], win[32];
    __shared__ uint32_t ejob[PD_NASR_JOBS], etok[PD_NASR_JOBS], ne;
    __shared__ uint32_t nslot;
    __shared__ float wv[8][32];
    __shared__ uint32_t wi[8][32];
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const uint32_t gr = lane >> 2, t4 = lane & 3u;
    const uint32_t gw = blockIdx.x * 8u + warp, nw = gridDim.x * 8u;
    if (tid < B) {
        jt[tid] = 0u;
        jsym[tid] = 0u;
        jcur[tid] = jobs[tid].st->cur;
        jcnt[tid] = 0u;
    }
    __syncthreads();
    // step the network of the emitting streams (ejob / etok, ne of them)
    auto network = [&]() {
        for (uint32_t base = 0; base < ne; base += 16u) {
            const uint32_t E = ne - base < 16u ? ne - base : 16u;
#pragma unroll 1
            for (uint32_t layer = 0; layer < 2u; ++layer) {
                // operand rows: 0-15 the layer's input, 16-31 its hidden state
                for (uint32_t i = tid; i < 32u * PD_NASR_PRED; i += 256u) {
                    const uint32_t r = i / PD_NASR_PRED, k = i % PD_NASR_PRED, e = r & 15u;
                    float v = 0.f;
                    if (e < E) {
                        const uint32_t j = ejob[base + e];
                        const PdNasrDecState* st = jobs[j].st;
                        const uint32_t cur = jcur[j];
                        if (r < 16u)
                            v = layer == 0u ? __half2float(W.embed[(size_t)etok[base + e] * PD_NASR_PRED + k])
                                            : st->h[cur ^ 1u][0][k];
                        else
                            v = st->h[cur][layer][k];
                    }
                    zs[r * PD_NASR_ZP + k] = v;
                }
                __syncthreads();
                for (uint32_t tile = gw; tile < 4u * PD_NASR_PRED / 8u; tile += nw) {
                    float fi[1][4], fh[1][4];
                    pd_nasr_tile<1u>(zs, 0u, W.ih_q[layer], W.ih_s[layer], 4u * PD_NASR_PRED, tile * 8u, lane, fi);
                    pd_nasr_tile<1u>(zs, 16u, W.hh_q[layer], W.hh_s[layer], 4u * PD_NASR_PRED, tile * 8u, lane,
                                     fh);
                    // columns 2 t4, 2 t4 + 1 of the tile: unit tile * 2 + t4 / 2, gates
                    // (i, f) on even t4 and (g, o) on odd
                    float gate[4];
#pragma unroll
                    for (uint32_t e = 0; e < 4u; ++e) {
                        const uint32_t col = tile * 8u + 2u * t4 + (e & 1u);
                        gate[e] = __fadd_rn(__fadd_rn(fi[0][e], W.ih_b[layer][col]),
                                            __fadd_rn(fh[0][e], W.hh_b[layer][col]));
                    }
                    float other[4];
#pragma unroll
                    for (uint32_t e = 0; e < 4u; ++e) other[e] = __shfl_down_sync(0xffffffffu, gate[e], 1);
                    if ((t4 & 1u) == 0u) {
                        const uint32_t unit = tile * 2u + t4 / 2u;
#pragma unroll
                        for (uint32_t half = 0; half < 2u; ++half) {
                            const uint32_t e = gr + half * 8u;
                            if (e >= E) continue;
                            const uint32_t j = ejob[base + e];
                            PdNasrDecState* st = jobs[j].st;
                            const uint32_t cur = jcur[j];
                            const float ig = pd_nasr_sigmoid(gate[half * 2u]), fg = pd_nasr_sigmoid(gate[half * 2u + 1u]);
                            const float cg_ = tanhf(other[half * 2u]), og = pd_nasr_sigmoid(other[half * 2u + 1u]);
                            const float c = __fadd_rn(__fmul_rn(fg, st->c[cur][layer][unit]), __fmul_rn(ig, cg_));
                            st->c[cur ^ 1u][layer][unit] = c;
                            st->h[cur ^ 1u][layer][unit] = __fmul_rn(og, tanhf(c));
                        }
                    }
                }
                grid.sync();
            }
            // the projection: g = W_p h1' + b_p
            for (uint32_t i = tid; i < 16u * PD_NASR_PRED; i += 256u) {
                const uint32_t e = i / PD_NASR_PRED, k = i % PD_NASR_PRED;
                zs[e * PD_NASR_ZP + k] = e < E ? jobs[ejob[base + e]].st->h[jcur[ejob[base + e]] ^ 1u][1][k] : 0.f;
            }
            __syncthreads();
            for (uint32_t tile = gw; tile < PD_NASR_PRED / 8u; tile += nw) {
                float f[1][4];
                pd_nasr_tile<1u>(zs, 0u, W.pred_q, W.pred_s, PD_NASR_PRED, tile * 8u, lane, f);
#pragma unroll
                for (uint32_t e = 0; e < 4u; ++e) {
                    const uint32_t row = gr + (e >= 2u ? 8u : 0u), col = tile * 8u + 2u * t4 + (e & 1u);
                    if (row < E) jobs[ejob[base + row]].st->g[col] = __fadd_rn(f[0][e], W.pred_b[col]);
                }
            }
            grid.sync();
            if (tid < E) jcur[ejob[base + tid]] ^= 1u;
            __syncthreads();
        }
    };
    // the decoder start of every primed stream
    if (tid == 0) {
        ne = 0;
        for (uint32_t j = 0; j < B; ++j)
            if (jobs[j].prime) {
                ejob[ne] = j;
                etok[ne] = PD_NASR_BLANK;
                ++ne;
            }
    }
    __syncthreads();
    if (ne) network();
    for (uint32_t step = 0;; ++step) {
        // the step's rows: each active stream's next frames
        if (tid == 0) {
            uint32_t active = 0;
            for (uint32_t j = 0; j < B; ++j) active += jt[j] < jobs[j].rows ? 1u : 0u;
            uint32_t n = 0;
            if (active) {
                const uint32_t width = 32u / active;
                for (uint32_t j = 0; j < B; ++j) {
                    const uint32_t left = jobs[j].rows - (jt[j] < jobs[j].rows ? jt[j] : jobs[j].rows);
                    const uint32_t w = left < width ? left : width;
                    for (uint32_t f = 0; f < w; ++f) {
                        sjob[n] = j;
                        sframe[n] = jt[j] + f;
                        ++n;
                    }
                }
            }
            nslot = n;
        }
        __syncthreads();
        const uint32_t n = nslot;
        if (n == 0) break;
        for (uint32_t i = tid; i < 32u * PD_NASR_PRED; i += 256u) {
            const uint32_t r = i / PD_NASR_PRED, k = i % PD_NASR_PRED;
            float v = 0.f;
            if (r < n) {
                const PdNasrJob& jb = jobs[sjob[r]];
                v = fmaxf(__fadd_rn(proj[(size_t)(jb.row0 + sframe[r]) * PD_NASR_PRED + k], jb.st->g[k]), 0.f);
            }
            zs[r * PD_NASR_ZP + k] = v;
        }
        __syncthreads();
        // the joint: every warp its 8-row vocabulary tiles, each row's best
        float best[2][2];
        uint32_t bi[2][2];
#pragma unroll
        for (uint32_t mt = 0; mt < 2u; ++mt)
#pragma unroll
            for (uint32_t h = 0; h < 2u; ++h) {
                best[mt][h] = -INFINITY;
                bi[mt][h] = 0xffffffffu;
            }
        auto consider = [&](uint32_t mt, const float (&f)[4], uint32_t n0) {
#pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) {
                const uint32_t col = n0 + 2u * t4 + (e & 1u), h = e >> 1;
                const float l = __fadd_rn(f[e], W.head_b[col]);
                if (l > best[mt][h] || (l == best[mt][h] && col < bi[mt][h])) {
                    best[mt][h] = l;
                    bi[mt][h] = col;
                }
            }
        };
        for (uint32_t tile = gw; tile < PD_NASR_VOCAB / 8u; tile += nw) {
            if (n > 16u) {
                float f[2][4];
                pd_nasr_tile<2u>(zs, 0u, W.head_q, W.head_s, PD_NASR_VOCAB, tile * 8u, lane, f);
                consider(0u, f[0], tile * 8u);
                consider(1u, f[1], tile * 8u);
            } else {
                float f[1][4];
                pd_nasr_tile<1u>(zs, 0u, W.head_q, W.head_s, PD_NASR_VOCAB, tile * 8u, lane, f);
                consider(0u, f[0], tile * 8u);
            }
        }
        // a row's best over the warp: the four t4 lanes, then the warps
#pragma unroll
        for (uint32_t mt = 0; mt < 2u; ++mt)
#pragma unroll
            for (uint32_t h = 0; h < 2u; ++h)
#pragma unroll
                for (uint32_t o = 1; o < 4u; o <<= 1) {
                    const float v = __shfl_xor_sync(0xffffffffu, best[mt][h], o);
                    const uint32_t i = __shfl_xor_sync(0xffffffffu, bi[mt][h], o);
                    if (v > best[mt][h] || (v == best[mt][h] && i < bi[mt][h])) {
                        best[mt][h] = v;
                        bi[mt][h] = i;
                    }
                }
        if (t4 == 0u)
#pragma unroll
            for (uint32_t mt = 0; mt < 2u; ++mt)
#pragma unroll
                for (uint32_t h = 0; h < 2u; ++h) {
                    wv[warp][mt * 16u + h * 8u + gr] = best[mt][h];
                    wi[warp][mt * 16u + h * 8u + gr] = bi[mt][h];
                }
        __syncthreads();
        const uint32_t par = step & 1u;
        if (tid < 32u) {
            float b = wv[0][tid];
            uint32_t idx = wi[0][tid];
            for (uint32_t w = 1; w < 8u; ++w)
                if (wv[w][tid] > b || (wv[w][tid] == b && wi[w][tid] < idx)) {
                    b = wv[w][tid];
                    idx = wi[w][tid];
                }
            part_v[(par * gridDim.x + blockIdx.x) * 32u + tid] = b;
            part_i[(par * gridDim.x + blockIdx.x) * 32u + tid] = idx;
        }
        grid.sync();
        if (tid < n) {
            float b = part_v[(par * gridDim.x) * 32u + tid];
            uint32_t idx = part_i[(par * gridDim.x) * 32u + tid];
            for (uint32_t c = 1; c < gridDim.x; ++c) {
                const float v = part_v[(par * gridDim.x + c) * 32u + tid];
                const uint32_t i = part_i[(par * gridDim.x + c) * 32u + tid];
                if (v > b || (v == b && i < idx)) {
                    b = v;
                    idx = i;
                }
            }
            win[tid] = idx;
        }
        __syncthreads();
        // each stream's rows in frame order, as the sequential walk
        if (tid == 0) {
            ne = 0;
            for (uint32_t r = 0; r < n;) {
                const uint32_t j = sjob[r];
                uint32_t r1 = r;
                while (r1 < n && sjob[r1] == j) ++r1;
                for (uint32_t s = r; s < r1; ++s) {
                    const uint32_t k = win[s];
                    if (k == PD_NASR_BLANK || k >= PD_NASR_VOCAB) {
                        jt[j] = sframe[s] + 1u;
                        jsym[j] = 0u;
                        continue;
                    }
                    const PdNasrJob& jb = jobs[j];
                    if (blockIdx.x == 0 && jcnt[j] < jb.cap) {
                        tokens[jb.out0 + jcnt[j]] = k;
                        frames[jb.out0 + jcnt[j]] = (uint32_t)(jb.frame0 + sframe[s]);
                    }
                    ++jcnt[j];
                    jt[j] = sframe[s];
                    if (++jsym[j] >= PD_NASR_MAXSYM) {
                        jsym[j] = 0u;
                        jt[j] = sframe[s] + 1u;
                    }
                    ejob[ne] = j;
                    etok[ne] = k;
                    ++ne;
                    break;
                }
                r = r1;
            }
        }
        __syncthreads();
        if (ne) network();
    }
    if (blockIdx.x == 0 && tid < B) {
        jobs[tid].st->cur = jcur[tid];
        jobs[tid].st->count = jcnt[tid];
        jobs[tid].count = jcnt[tid];
    }
}

PD_EXPORT
int pd_nasr_greedy(const void* const* w, const void* proj, void* jobs, uint32_t B, void* tokens, void* frames,
                   void* part, void* stream) {
    if (B == 0) return 0;
    if (B > PD_NASR_JOBS || !w || !jobs || !part || !tokens || !frames) return (int)cudaErrorInvalidValue;
    PdNasrDecW W;
    W.head_q = (const int8_t*)w[0];
    W.head_s = (const __half*)w[1];
    W.head_b = (const float*)w[2];
    for (uint32_t l = 0; l < 2u; ++l) {
        W.ih_q[l] = (const int8_t*)w[3 + 6 * l];
        W.ih_s[l] = (const __half*)w[4 + 6 * l];
        W.ih_b[l] = (const float*)w[5 + 6 * l];
        W.hh_q[l] = (const int8_t*)w[6 + 6 * l];
        W.hh_s[l] = (const __half*)w[7 + 6 * l];
        W.hh_b[l] = (const float*)w[8 + 6 * l];
    }
    W.pred_q = (const int8_t*)w[15];
    W.pred_s = (const __half*)w[16];
    W.pred_b = (const float*)w[17];
    W.embed = (const __half*)w[18];
    for (uint32_t i = 0; i < 19u; ++i)
        if (!w[i] || ((uintptr_t)w[i] & 15u)) return (int)cudaErrorInvalidValue;
    static int nsm = 0;
    if (nsm == 0) {
        cudaFuncSetAttribute(pd_nasr_greedy_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)PD_NASR_DEC_SMEM);
        int d = 0;
        cudaGetDevice(&d);
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, d);
        int per = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, (const void*)pd_nasr_greedy_kernel, 256,
                                                      PD_NASR_DEC_SMEM);
        if (nsm <= 0) nsm = 48;
        if (per < 1) nsm = -1;
        if (nsm > 64) nsm = 64;
    }
    if (nsm < 0) return (int)cudaErrorCooperativeLaunchTooLarge;
    // part: 2 parities x 64 CTAs x 32 rows of floats, then the same of u32
    float* pv = (float*)part;
    uint32_t* pi = (uint32_t*)((float*)part + 2 * 64 * 32);
    const float* pj = (const float*)proj;
    PdNasrJob* jb = (PdNasrJob*)jobs;
    uint32_t* tk = (uint32_t*)tokens;
    uint32_t* fr = (uint32_t*)frames;
    void* args[] = {(void*)&W, (void*)&pj, (void*)&jb, (void*)&B, (void*)&tk, (void*)&fr, (void*)&pv, (void*)&pi};
    const cudaError_t e = cudaLaunchCooperativeKernel((const void*)pd_nasr_greedy_kernel, dim3(nsm), dim3(256), args,
                                                      PD_NASR_DEC_SMEM, (cudaStream_t)stream);
    return e ? (int)e : pd_launch_status();
}
