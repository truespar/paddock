// Nemotron 3 Diarization (NVIDIA's streaming Sortformer v3) at F32 class.
// Weights are the checkpoint's own values (BF16 and Q8_0 read as stored -
// exact operands; the head's F16/F32 widened exactly); activations stay F32
// throughout - no operand is narrowed.
//
// One window (host side: gpu_model/diarization), R <= 684 encoder rows of
// which the first V are keys (speaker cache, FIFO and the chunk's real rows):
//
//   features = log-mel of the PCM window (718), frame-major [8R][128], so
//              the stacked rows [R][1024] are the same memory
//   x        = features . W_pre (722, the GEMM on the stored weights)
//   x        = LayerNorm(x) (719)
//   31 x { n = LayerNorm(x); qkv = n . W_qkv with the rope on q and k in
//          its epilogue (722); attention over the V keys (723); x +=
//          att . W_o + b_o (722 residual epilogue); n = LayerNorm(x);
//          h = GELU(n . W_1 + b_1); x += h . W_2 + b_2 }
//   n = LayerNorm(x); p = n . W_proj + b_proj      [R][192]
//   conv rows (720), u = ReLU(rows . W_conv + b)    [R][1536] = [8R][192]
//   ReLU(u . W_dense + b), sigmoid(. . W_out + b)  [8R][8] (the head's
//   GEMMs the Kumo F32 one, 698; 721 the ReLU and sigmoid passes)
//
// Operation order follows the Transformers reference (huggingface/
// transformers nemotron3_diarization, F32 PyTorch): LayerNorm is (x - mean)
// * rstd * w + b, the rope is x * cos + rotate_half(x) * sin with each
// product rounded, sigmoid is 1 / (1 + exp(-x)).
//
// Architecture and streaming references and their licences: the Metal
// lane's diarization.NOTICE.md and the generated third-party notices.
//
// A window is ~260 dependent launches, so every kernel here launches as a
// programmatic dependent (pd_pdl_go) and arms itself (PD_PDL_ARM: trigger,
// then wait for the predecessor's completion before touching its outputs) -
// the next kernel's launch overlaps this one's tail, the numerics are the
// plain launch's by construction.
//
// Plain CUDA; needs gemm/f32_qkv.cuh's pd_launch_status.

#define PD_DIAR_HOP 160u
#define PD_DIAR_MEL 128u
#define PD_DIAR_BINS 257u

// ------------------------------------------------------------------ 718 frontend
// One mel frame per CTA (256 threads). All coordinates are absolute in the
// recording: frame f's centered 512-sample window starts at f * 160 - 256,
// samples outside [0, total) are zero, and the preemphasis x[p] - 0.97 x[p-1]
// reads the sample before p even when that one is outside this window (it is
// in `pcm`: offset <= start * 160 - 257). `pcm` holds samples offset ..
// total - 1. Radix-2 FFT in shared memory with F64-derived twiddles, power,
// the filterbank over each band's nonzero span (an exact zero coefficient
// contributes an exact zero), log(mel + 2^-24). Frames at or past `count`,
// and frames past the recording's last full hop, are zero features (padding,
// not log(silence)).
__global__ void __launch_bounds__(256)
pd_diar_frontend_kernel(const float* __restrict__ pcm, const float* __restrict__ window,
                        const float* __restrict__ fb, const uint2* __restrict__ spans,
                        const float2* __restrict__ tw, float* __restrict__ out, uint32_t offset,
                        uint32_t total, uint32_t start, uint32_t count) {
    __shared__ float2 z[512];
    PD_PDL_ARM();
    const uint32_t j = blockIdx.x, tid = threadIdx.x, id = start + j;
    float* o = out + (size_t)j * PD_DIAR_MEL;
    if (j >= count || id >= total / PD_DIAR_HOP) {
        if (tid < PD_DIAR_MEL) o[tid] = 0.f;
        return;
    }
    for (uint32_t i = tid; i < 512u; i += 256u) {
        const int64_t p = (int64_t)id * PD_DIAR_HOP + i - 256;
        float x = 0.f;
        if (p >= 0 && p < (int64_t)total) {
            const float prev = p > 0 ? pcm[p - 1 - offset] : 0.f;
            x = __fsub_rn(pcm[p - offset], __fmul_rn(0.97f, prev));
        }
        z[__brev(i) >> 23] = make_float2(__fmul_rn(x, window[i]), 0.f);
    }
    __syncthreads();
#pragma unroll
    for (uint32_t size = 2u; size <= 512u; size <<= 1) {
        const uint32_t half = size >> 1, k = tid % half, a = tid / half * size + k, b = a + half;
        const float2 t = tw[k * (512u / size)], e = z[a], w = z[b];
        const float re = fmaf(t.x, w.x, -__fmul_rn(t.y, w.y));
        const float im = fmaf(t.y, w.x, __fmul_rn(t.x, w.y));
        z[a] = make_float2(__fadd_rn(e.x, re), __fadd_rn(e.y, im));
        z[b] = make_float2(__fsub_rn(e.x, re), __fsub_rn(e.y, im));
        __syncthreads();
    }
    if (tid < PD_DIAR_MEL) {
        const uint2 span = spans[tid];
        const float* row = fb + (size_t)tid * PD_DIAR_BINS;
        float sum = 0.f;
        for (uint32_t k = span.x; k < span.y; ++k) {
            const float2 v = z[k];
            sum = fmaf(row[k], fmaf(v.x, v.x, __fmul_rn(v.y, v.y)), sum);
        }
        o[tid] = logf(__fadd_rn(sum, 0x1p-24f));
    }
}

PD_EXPORT
int pd_diar_frontend(const void* pcm, const void* window, const void* fb, const void* spans,
                     const void* twiddle, void* out, uint32_t offset, uint32_t total,
                     uint32_t start, uint32_t count, uint32_t frames, void* stream) {
    if (frames == 0) return 0;
    // the first window's preemphasis reads sample start * 160 - 257
    const uint64_t first = (uint64_t)start * PD_DIAR_HOP;
    if (count > frames || offset > total || offset > (first > 257u ? first - 257u : 0u))
        return (int)cudaErrorInvalidValue;
    pd_pdl_go(pd_diar_frontend_kernel, dim3(frames), dim3(256), 0, (cudaStream_t)stream,
              (const float*)pcm, (const float*)window, (const float*)fb, (const uint2*)spans,
              (const float2*)twiddle, (float*)out, offset, total, start, count);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 719 LayerNorm
// torch.nn.LayerNorm over 512 channels, eps 1e-5: one warp a row, lane
// owning channels lane + 32i; the mean, then the biased variance of the
// centered values, each lane summing upward and a fixed xor tree across
// lanes; y = (x - mean) * rstd * w + b.
__global__ void __launch_bounds__(256)
pd_diar_norm_kernel(const float* __restrict__ x, const float* __restrict__ w,
                    const float* __restrict__ b, float* __restrict__ y, uint32_t rows) {
    constexpr uint32_t V = 16u, D = 512u;
    PD_PDL_ARM();
    const uint32_t row = blockIdx.x * 8u + threadIdx.x / 32u, lane = threadIdx.x % 32u;
    if (row >= rows) return;
    const float* px = x + (size_t)row * D;
    float v[V], s = 0.f;
#pragma unroll
    for (uint32_t i = 0; i < V; ++i) {
        v[i] = px[lane + 32u * i];
        s = __fadd_rn(s, v[i]);
    }
#pragma unroll
    for (uint32_t o = 16; o > 0; o >>= 1) s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, o));
    const float mean = __fdiv_rn(s, (float)D);
    float q = 0.f;
#pragma unroll
    for (uint32_t i = 0; i < V; ++i) {
        v[i] = __fsub_rn(v[i], mean);
        q = fmaf(v[i], v[i], q);
    }
#pragma unroll
    for (uint32_t o = 16; o > 0; o >>= 1) q = __fadd_rn(q, __shfl_xor_sync(0xffffffffu, q, o));
    const float rstd = __frsqrt_rn(__fadd_rn(__fdiv_rn(q, (float)D), 1e-5f));
    float* py = y + (size_t)row * D;
#pragma unroll
    for (uint32_t i = 0; i < V; ++i) {
        const uint32_t c = lane + 32u * i;
        py[c] = fmaf(__fmul_rn(v[i], rstd), w[c], b[c]);
    }
}

PD_EXPORT
int pd_diar_norm(const void* x, const void* w, const void* b, void* y, uint32_t d, uint32_t rows,
                 void* stream) {
    if (rows == 0) return 0;
    if (d != 512u) return (int)cudaErrorInvalidValue;
    pd_pdl_go(pd_diar_norm_kernel, dim3((rows + 7u) / 8u), dim3(256), 0, (cudaStream_t)stream,
              (const float*)x, (const float*)w, (const float*)b, (float*)y, rows);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 720 conv rows
// The sub-pixel convolution (kernel 3, padding 1 over the window's rows) as a
// GEMM operand: rows[r][t * 192 + c] = p[r + t - 1][c], zero outside the
// window - the checkpoint's [out][t][c] weight rows walk the same order.
__global__ void pd_diar_conv_rows_kernel(const float* __restrict__ p, float* __restrict__ rows,
                                         uint32_t R) {
    PD_PDL_ARM();
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= R * 576u) return;
    const int32_t r = (int32_t)(i / 576u) + (int32_t)(i % 576u / 192u) - 1;
    rows[i] = r >= 0 && r < (int32_t)R ? p[(size_t)r * 192u + i % 192u] : 0.f;
}

PD_EXPORT
int pd_diar_conv_rows(const void* p, void* rows, uint32_t R, void* stream) {
    if (R == 0) return 0;
    pd_pdl_go(pd_diar_conv_rows_kernel, dim3((R * 576u + 255u) / 256u), dim3(256), 0,
              (cudaStream_t)stream, (const float*)p, (float*)rows, R);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 721 activations
// In place over n values: op 0 ReLU, op 1 sigmoid 1 / (1 + exp(-x)).
__global__ void pd_diar_act_kernel(float* __restrict__ x, uint64_t n, uint32_t op) {
    PD_PDL_ARM();
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    x[i] = op ? __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-v))) : fmaxf(v, 0.f);
}

PD_EXPORT
int pd_diar_act(void* x, uint64_t n, uint32_t op, void* stream) {
    if (n == 0) return 0;
    if (op > 1u) return (int)cudaErrorInvalidValue;
    pd_pdl_go(pd_diar_act_kernel, dim3((unsigned)((n + 255u) / 256u)), dim3(256), 0,
              (cudaStream_t)stream, (float*)x, n, op);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 722 GEMM on the stored weights
// y[m][n] = epi(sum_k x[m][k] * W[n][k] + bias[n]) with W read as the
// checkpoint stores it: BF16 rows [N][K] (the MLX export), or Q8_0 repacked
// at load to int8 rows [N][K] plus one F32 scale a 32-deep block, laid out
// [K/32][N] so a k tile's scales are one contiguous row. Epilogues: the Kumo
// GEMM's 0 store, 1 GELU, 2 accumulate into y (bias before the residual),
// and 3 the attention's rope - the qkv projection's rows (q | k | v, 8 heads
// of 64 each): split-half rope at position row on q and k, (x1, x2) -> (x1
// cos - x2 sin, x2 cos + x1 sin), each product rounded then added (the
// reference's x * cos + rotate_half(x) * sin), q into its own [M][512]
// plane y2, k in place, v as computed; `rope` holds (cos, sin) [M][32]. A
// 64-wide N tile is one head and the elected tiles give a warp all 64
// columns, so a pair (j, j + 32) is one thread's fragments nt and nt + 4.
//
// F32 class at half the mma of 3xTF32. A weight here is an exact bf16
// operand (BF16 as stored; |q| <= 127 is exact in bf16's 8-bit
// significand), so splitting the activation into three bf16 parts, x = hi +
// mid + lo (each the bf16 rounding of what the earlier parts leave - exact
// differences in F32), and issuing hi.w, mid.w and lo.w IS the six-product
// BF16x6 sum: the three products against a weight's mid and lo parts are
// exactly zero. Three m16n8k16 a k16 span against 3xTF32's six m16n8k8, the
// same F32-level operand error (x is carried to ~24 bits, w exactly). The
// mma chain drains into a round-nearest F32 accumulator once per 32-deep k
// tile, as the house class does (the tensor core's own chain truncates):
// fac + acc for BF16, fma(scale, acc, fac) for Q8_0 - its block IS the k
// tile, so the dequantized product is the integer one scaled once.
// The per-element sequence is the same in every tile shape, so the
// election is shape-only and never moves a bit.
// Q8H is Q8_0 with the file's own f16 scales kept as stored ([K/32][N]
// halves): the resident plane is the GGUF's 8.5 bits a weight, not 9, and
// the scale widens exactly at the drain - the same arithmetic as Q8 (Clef's
// GGUF backbone, slot 741).
enum : uint32_t { PD_DIAR_W_BF16 = 0u, PD_DIAR_W_Q8 = 1u, PD_DIAR_W_Q8H = 2u };
// 4: the gated MLP's gate and up in one pass - the weight rows interleaved
// (row 2j gate j, row 2j + 1 up j), so a lane's accumulator pair (columns
// 2 t4, 2 t4 + 1) is one output's (gate, up): y[m][j] = silu(gate) * up,
// N / 2 wide, the 2N-wide plane never written (Clef's backbone MLP)
enum : uint32_t { PD_DIAR_ROPE = 3u, PD_DIAR_SWIGLU = 4u };
// 5: acc + bias through the tanh GELU (gelu_pytorch_tanh - Clef's vision
// blocks), in torch's CUDA operation order: 0.5 x (1 + tanh(sqrt(2 / pi)
// (x + 0.044715 x^3)))
enum : uint32_t { PD_DIAR_GELU_TANH = 5u };

static __device__ __forceinline__ float pd_diar_gelu_tanh(float x) {
    const float cube = __fmul_rn(__fmul_rn(x, x), x);
    const float inner = __fmul_rn(0.7978845608028654f, __fmaf_rn(0.044715f, cube, x));
    return __fmul_rn(__fmul_rn(0.5f, x), __fadd_rn(1.f, tanhf(inner)));
}

// bf16x2 of (lo, hi): lo in the low half, round to nearest even
static __device__ __forceinline__ uint32_t pd_diar_bf2(float lo, float hi) {
    uint32_t r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}

// The three-part bf16 split of an adjacent pair, each part a bf16x2.
static __device__ __forceinline__ void pd_diar_split(float2 v, uint32_t& h, uint32_t& m, uint32_t& l) {
    h = pd_diar_bf2(v.x, v.y);
    const float r0 = __fsub_rn(v.x, __uint_as_float(h << 16));
    const float r1 = __fsub_rn(v.y, __uint_as_float(h & 0xffff0000u));
    m = pd_diar_bf2(r0, r1);
    const float s0 = __fsub_rn(r0, __uint_as_float(m << 16));
    const float s1 = __fsub_rn(r1, __uint_as_float(m & 0xffff0000u));
    l = pd_diar_bf2(s0, s1);
}

// The two-part split (hi + mid): the activation to 16 significant bits.
static __device__ __forceinline__ void pd_diar_split2(float2 v, uint32_t& h, uint32_t& m) {
    h = pd_diar_bf2(v.x, v.y);
    m = pd_diar_bf2(__fsub_rn(v.x, __uint_as_float(h << 16)), __fsub_rn(v.y, __uint_as_float(h & 0xffff0000u)));
}

static __device__ __forceinline__ void pd_diar_mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0,
                                                   uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// The same with a zero accumulator in: a k tile's first product, so the
// chain starts from +0 exactly as a zeroed accumulator would, without the
// zeroing.
static __device__ __forceinline__ void pd_diar_mma0(float (&c)[4], const uint32_t (&a)[4], uint32_t b0,
                                                    uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(0.f));
}

// Two n tiles' bf16 B fragments in one ldmatrix: matrices (n, k), (n, k + 8),
// (n + 8, k), (n + 8, k + 8) are b0 / b1 of tile n and of tile n + 8 - each
// thread lands row t / 4, columns 2 (t % 4) and + 1, the mma's own layout.
static __device__ __forceinline__ void pd_diar_ldsm4(uint32_t& r0, uint32_t& r1, uint32_t& r2,
                                                     uint32_t& r3, const void* p) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(a));
}

// Two int8 weights (low byte first) as an exact bf16x2.
static __device__ __forceinline__ uint32_t pd_diar_q2(uint16_t q) {
    return pd_diar_bf2((float)(int8_t)(q & 0xffu), (float)(int8_t)(q >> 8));
}

// BM x BN per CTA, WGM x WGN warps of (BM/WGM) x (BN/WGN) m16 x n8
// fragments. A rows staged as F32 at a 40-float pitch (the pair loads land
// conflict-free) and split in registers; W rows as stored at an 80-byte
// (BF16: two n tiles' fragments a ldmatrix, its 8 row addresses on disjoint
// banks) or 48-byte (int8: 16-bit fragment loads) pitch; an ST-slot cp.async
// ring as the Kumo GEMM's, the weights of its first tiles in flight before
// the PDL wait. Rows past M or N stage zeros.
template <uint32_t BM, uint32_t BN, uint32_t WGM, uint32_t WGN, uint32_t ST, uint32_t WT, uint32_t P = 3u>
__global__ void __launch_bounds__(WGM * WGN * 32u)
pd_diar_gemm_kernel(const float* __restrict__ x, const uint8_t* __restrict__ w,
                    const uint8_t* __restrict__ scale, const float* __restrict__ bias,
                    float* __restrict__ y, float* __restrict__ y2, const float2* __restrict__ rope,
                    uint32_t K, uint32_t N, uint32_t M, uint32_t mode, uint32_t group) {
    constexpr bool Q8 = WT != PD_DIAR_W_BF16;
    constexpr uint32_t BK = 32u, SA = BK + 8u, THREADS = WGM * WGN * 32u;
    constexpr uint32_t ROWB = Q8 ? BK : 2u * BK;  // a W row's bytes a k tile
    constexpr uint32_t SB = Q8 ? 48u : 80u;       // its staged pitch
    constexpr uint32_t SE = WT == PD_DIAR_W_Q8H ? 2u : 4u;  // a scale's bytes
    constexpr uint32_t WTM = BM / WGM, WTN = BN / WGN, MT = WTM / 16u, NT = WTN / 8u;
    constexpr uint32_t XC = BM * (BK / 4u) / THREADS;  // 16 B chunks a thread stages
    // W chunks a thread stages, the last round partial when they do not
    // divide (a 32-wide int8 tile is 64 chunks over 128 threads)
    constexpr uint32_t WCH = BN * (ROWB / 16u), WC = (WCH + THREADS - 1u) / THREADS;
    constexpr uint32_t SPC = 16u / SE;                   // scales a 16 B chunk
    constexpr uint32_t SC = Q8 ? BN / SPC : 0u;          // scale chunks a tile
    static_assert(XC * THREADS == BM * (BK / 4u), "even A staging");
    static_assert(SC <= THREADS, "one scale chunk a thread at most");
    static_assert(Q8 || NT % 2u == 0, "BF16 B fragments load two n tiles at a time");
    extern __shared__ __align__(16) uint8_t diar_gsm[];
    float* xs = reinterpret_cast<float*>(diar_gsm);                       // [ST][BM][SA]
    uint8_t* wsm = diar_gsm + ST * BM * SA * 4u;                         // [ST][BN][SB]
    uint8_t* ss = wsm + ST * BN * SB;                                     // [ST][BN] scales
    // Tile order. group 0: the 2D grid, n fastest (a window's planes fit
    // L2 whole). group g: a 1D grid walking g M-tiles fastest, then the next
    // N tile, then the next g M-tiles - a weight tile is fetched from DRAM
    // once a group instead of once an M-tile while the group's activation
    // rows stay L2-resident (Clef's 4096-wide backbone planes are 32-100 MB
    // against a 24 MB L2). Ownership only: every output's k walk is the same.
    uint32_t mi = blockIdx.y, ni = blockIdx.x;
    if (group) {
        const uint32_t nm = (M + BM - 1u) / BM, nn = (N + BN - 1u) / BN, per = group * nn;
        const uint32_t g = blockIdx.x / per, r = blockIdx.x % per, first = g * group;
        const uint32_t gs = min(group, nm - first);
        mi = first + r % gs;
        ni = r / gs;
    }
    const uint32_t m0 = mi * BM, n0 = ni * BN;
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const uint32_t gr = lane >> 2, t4 = lane & 3u, wm = warp / WGN, wn = warp % WGN;
    float acc[MT][NT][4], fac[MT][NT][4];
#pragma unroll
    for (uint32_t i = 0; i < MT; ++i)
#pragma unroll
        for (uint32_t j = 0; j < NT; ++j)
#pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) acc[i][j][e] = fac[i][j][e] = 0.f;
    // a thread stages the same chunks of the same rows every k tile: the
    // addresses resolve once (the Kumo GEMM's lesson), the loop advances them
    const float* xp[XC];
    uint32_t xd[XC], wd[WC];
    const uint8_t* wp[WC];
    uint32_t live = 0, wown = 0;  // live: real rows; wown: chunks this thread stages
#pragma unroll
    for (uint32_t i = 0; i < XC; ++i) {
        const uint32_t c = tid + i * THREADS, r = c / (BK / 4u), m = m0 + r;
        xp[i] = x + (size_t)(m < M ? m : 0u) * K + (c % (BK / 4u)) * 4u;
        xd[i] = r * SA + (c % (BK / 4u)) * 4u;
        live |= (m < M ? 1u : 0u) << i;
    }
#pragma unroll
    for (uint32_t i = 0; i < WC; ++i) {
        const uint32_t c = tid + i * THREADS, r = c / (ROWB / 16u), n = n0 + r;
        const uint32_t kb = Q8 ? K : 2u * K;  // a W row's bytes
        wp[i] = w + (size_t)(n < N ? n : 0u) * kb + (c % (ROWB / 16u)) * 16u;
        wd[i] = r * SB + (c % (ROWB / 16u)) * 16u;
        live |= (n < N ? 1u : 0u) << (XC + i);
        wown |= (c < WCH ? 1u : 0u) << i;
    }
    const bool slive = SC && tid < SC && n0 + tid * SPC < N;
    auto stage_x = [&](uint32_t slot, uint32_t k0) {
#pragma unroll
        for (uint32_t i = 0; i < XC; ++i) {
            const unsigned sm = (unsigned)__cvta_generic_to_shared(xs + slot * BM * SA + xd[i]);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(sm), "l"(xp[i] + k0),
                         "r"((live >> i) & 1u ? 16u : 0u));
        }
    };
    // the weights (and Q8_0 scales): never written by a predecessor
    auto stage_w = [&](uint32_t slot, uint32_t k0) {
#pragma unroll
        for (uint32_t i = 0; i < WC; ++i) {
            if (WCH % THREADS && !((wown >> i) & 1u)) continue;
            const uint32_t kb = Q8 ? k0 : 2u * k0;
            const unsigned sm = (unsigned)__cvta_generic_to_shared(wsm + slot * BN * SB + wd[i]);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(sm), "l"(wp[i] + kb),
                         "r"((live >> (XC + i)) & 1u ? 16u : 0u));
        }
        if (SC && tid < SC) {
            // N a multiple of a chunk's scales (the host checks): a chunk is
            // whole or absent
            const unsigned sm = (unsigned)__cvta_generic_to_shared(ss + (slot * BN + tid * SPC) * SE);
            const uint8_t* src = scale + ((size_t)(k0 / BK) * N + (slive ? n0 + tid * SPC : 0u)) * SE;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(sm), "l"(src),
                         "r"(slive ? 16u : 0u));
        }
    };
    auto stage = [&](uint32_t slot, uint32_t k0) {
        stage_x(slot, k0);
        stage_w(slot, k0);
    };
    // Programmatic dependent: release the next launch, put the first tiles'
    // weights in flight, and only then wait for the predecessor that wrote
    // x. The weight copies join tile 0's commit group, so the ring's
    // "ST - 1 groups pending" accounting is unchanged.
    if (threadIdx.x == 0) PD_PDL_RELEASE();
#pragma unroll
    for (uint32_t s = 0; s + 1u < ST; ++s)
        if (s * BK < K) stage_w(s, s * BK);
    PD_PDL_ARM_WAIT();
#pragma unroll
    for (uint32_t s = 0; s + 1u < ST; ++s) {
        if (s * BK < K) stage_x(s, s * BK);
        asm volatile("cp.async.commit_group;" ::: "memory");
    }
    for (uint32_t t = 0, k0 = 0; k0 < K; ++t, k0 += BK) {
        const uint32_t slot = t % ST;
        if (k0 + (ST - 1u) * BK < K) stage((t + ST - 1u) % ST, k0 + (ST - 1u) * BK);
        asm volatile("cp.async.commit_group;" ::: "memory");
        asm volatile("cp.async.wait_group %0;" ::"n"(ST - 1u) : "memory");
        __syncthreads();
        const float* xsl = xs + slot * BM * SA;
        const uint8_t* wsl = wsm + slot * BN * SB;
#pragma unroll
        for (uint32_t kk = 0; kk < BK; kk += 16u) {
            uint32_t ah[MT][4], am[MT][4], al[MT][4];
#pragma unroll
            for (uint32_t mt = 0; mt < MT; ++mt) {
                const float* ar = xsl + (wm * WTM + mt * 16u + gr) * SA + kk + 2u * t4;
#pragma unroll
                for (uint32_t e = 0; e < 4u; ++e) {
                    // a0 (row, k), a1 (row + 8, k), a2 (row, k + 8), a3 (row + 8, k + 8)
                    const float2 v = *reinterpret_cast<const float2*>(ar + (e & 1u) * 8u * SA + (e >> 1) * 8u);
                    if constexpr (P == 3u)
                        pd_diar_split(v, ah[mt][e], am[mt][e], al[mt][e]);
                    else
                        pd_diar_split2(v, ah[mt][e], am[mt][e]);
                }
            }
            uint32_t b0[NT], b1[NT];
            if constexpr (!Q8) {
                // lane l addresses row (l / 16) * 8 + l % 8 of an n-tile pair
                // at k + ((l / 8) % 2) * 8
#pragma unroll
                for (uint32_t nt = 0; nt < NT; nt += 2u) {
                    const uint8_t* br = wsl + (wn * WTN + nt * 8u + (lane >> 4) * 8u + (lane & 7u)) * SB +
                                        2u * (kk + ((lane >> 3) & 1u) * 8u);
                    pd_diar_ldsm4(b0[nt], b1[nt], b0[nt + 1u], b1[nt + 1u], br);
                }
            } else {
#pragma unroll
                for (uint32_t nt = 0; nt < NT; ++nt) {
                    const uint8_t* br = wsl + (wn * WTN + nt * 8u + gr) * SB;
                    b0[nt] = pd_diar_q2(*reinterpret_cast<const uint16_t*>(br + kk + 2u * t4));
                    b1[nt] = pd_diar_q2(*reinterpret_cast<const uint16_t*>(br + kk + 8u + 2u * t4));
                }
            }
            // every accumulator takes hi, then mid, then lo (the per-element
            // sequence); issued part by part so consecutive mma never wait on
            // each other. A k tile's first product starts from +0.
#pragma unroll
            for (uint32_t mt = 0; mt < MT; ++mt)
#pragma unroll
                for (uint32_t nt = 0; nt < NT; ++nt) {
                    if (kk == 0u)
                        pd_diar_mma0(acc[mt][nt], ah[mt], b0[nt], b1[nt]);
                    else
                        pd_diar_mma(acc[mt][nt], ah[mt], b0[nt], b1[nt]);
                }
#pragma unroll
            for (uint32_t mt = 0; mt < MT; ++mt)
#pragma unroll
                for (uint32_t nt = 0; nt < NT; ++nt) pd_diar_mma(acc[mt][nt], am[mt], b0[nt], b1[nt]);
            if constexpr (P == 3u) {
#pragma unroll
                for (uint32_t mt = 0; mt < MT; ++mt)
#pragma unroll
                    for (uint32_t nt = 0; nt < NT; ++nt) pd_diar_mma(acc[mt][nt], al[mt], b0[nt], b1[nt]);
            }
        }
        const uint8_t* ssl = ss + slot * BN * SE;
#pragma unroll
        for (uint32_t mt = 0; mt < MT; ++mt)
#pragma unroll
            for (uint32_t nt = 0; nt < NT; ++nt)
#pragma unroll
                for (uint32_t e = 0; e < 4u; ++e) {
                    if constexpr (!Q8) {
                        fac[mt][nt][e] = __fadd_rn(fac[mt][nt][e], acc[mt][nt][e]);
                    } else {
                        const uint32_t col = wn * WTN + nt * 8u + 2u * t4 + (e & 1u);
                        const float d = WT == PD_DIAR_W_Q8H
                                            ? __half2float(reinterpret_cast<const __half*>(ssl)[col])
                                            : reinterpret_cast<const float*>(ssl)[col];
                        fac[mt][nt][e] = __fmaf_rn(d, acc[mt][nt][e], fac[mt][nt][e]);
                    }
                }
        __syncthreads();  // tile t's slot is restaged next iteration
    }
    if constexpr (WTN == 64u) {
        if (mode == PD_DIAR_ROPE) {
            // this tile is head n0 / 64: q 0-7, k 8-15, v 16-23
            const uint32_t head = n0 / 64u;
#pragma unroll
            for (uint32_t mt = 0; mt < MT; ++mt)
#pragma unroll
                for (uint32_t nt = 0; nt < 4u; ++nt)
#pragma unroll
                    for (uint32_t e = 0; e < 4u; ++e) {
                        const uint32_t m = m0 + wm * WTM + mt * 16u + gr + (e >= 2u ? 8u : 0u);
                        if (m >= M) continue;
                        const uint32_t j = nt * 8u + 2u * t4 + (e & 1u);
                        float a = fac[mt][nt][e], c = fac[mt][nt + 4u][e];
                        if (bias) {
                            a = __fadd_rn(a, bias[n0 + j]);
                            c = __fadd_rn(c, bias[n0 + j + 32u]);
                        }
                        if (head < 16u) {
                            const float2 cs = rope[(size_t)m * 32u + j];
                            const float x1 = a;
                            a = __fadd_rn(__fmul_rn(x1, cs.x), __fmul_rn(-c, cs.y));
                            c = __fadd_rn(__fmul_rn(c, cs.x), __fmul_rn(x1, cs.y));
                        }
                        float* o = head < 8u ? y2 + (size_t)m * 512u + head * 64u : y + (size_t)m * N + n0;
                        o[j] = a;
                        o[j + 32u] = c;
                    }
            return;
        }
    }
    if (mode == PD_DIAR_SWIGLU) {
#pragma unroll
        for (uint32_t mt = 0; mt < MT; ++mt)
#pragma unroll
            for (uint32_t nt = 0; nt < NT; ++nt)
#pragma unroll
                for (uint32_t e = 0; e < 4u; e += 2u) {
                    const uint32_t m = m0 + wm * WTM + mt * 16u + gr + (e >= 2u ? 8u : 0u);
                    const uint32_t n = n0 + wn * WTN + nt * 8u + 2u * t4;
                    if (m >= M || n >= N) continue;
                    float g = fac[mt][nt][e], u = fac[mt][nt][e + 1u];
                    if (bias) {
                        g = __fadd_rn(g, bias[n]);
                        u = __fadd_rn(u, bias[n + 1u]);
                    }
                    y[(size_t)m * (N / 2u) + n / 2u] = pd_glu_act<PD_ACT_SILU>(g) * u;
                }
        return;
    }
#pragma unroll
    for (uint32_t mt = 0; mt < MT; ++mt)
#pragma unroll
        for (uint32_t nt = 0; nt < NT; ++nt)
#pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) {
                const uint32_t m = m0 + wm * WTM + mt * 16u + gr + (e >= 2u ? 8u : 0u);
                const uint32_t n = n0 + wn * WTN + nt * 8u + 2u * t4 + (e & 1u);
                if (m >= M || n >= N) continue;
                float z = fac[mt][nt][e];
                if (bias) z = __fadd_rn(z, bias[n]);
                if (mode == PD_KUMO_GELU) z = pd_kumo_gelu(z);
                else if (mode == PD_DIAR_GELU_TANH) z = pd_diar_gelu_tanh(z);
                float* o = y + (size_t)m * N + n;
                *o = mode == PD_KUMO_RESID ? __fadd_rn(*o, z) : z;
            }
}

template <uint32_t BM, uint32_t BN, uint32_t WGM, uint32_t WGN, uint32_t ST, uint32_t WT, uint32_t P = 3u>
static void pd_diar_gemm_go(const float* x, const uint8_t* w, const void* scale, const float* bias,
                            float* y, float* y2, const float2* rope, uint32_t K, uint32_t N, uint32_t M,
                            uint32_t mode, cudaStream_t s) {
    constexpr size_t smem = ST * (BM * (32u + 8u) * 4u + BN * (WT == PD_DIAR_W_BF16 ? 80u : 48u) +
                                  (WT == PD_DIAR_W_BF16 ? 0u : BN * (WT == PD_DIAR_W_Q8H ? 2u : 4u)));
    static bool set = false;
    if (!set) {
        cudaFuncSetAttribute(pd_diar_gemm_kernel<BM, BN, WGM, WGN, ST, WT, P>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        set = true;
    }
    const uint32_t nm = (M + BM - 1u) / BM, nn = (N + BN - 1u) / BN;
    // grouped order once the weight plane outgrows half the L2: as many
    // M-tiles a group as keep their F32 activation rows in 8 MB (a third of
    // GB10's 24 MB L2, the rest for the weight tiles and outputs in flight)
    const uint64_t wbytes = (uint64_t)N * K * (WT == PD_DIAR_W_BF16 ? 2u : 1u);
    uint32_t group = 0;
    if (wbytes > (12ull << 20) && nm > 1u) {
        const uint64_t g = (8ull << 20) / ((uint64_t)BM * K * 4u);
        group = (uint32_t)(g < 1u ? 1u : g > nm ? nm : g);
    }
    const dim3 grid = group ? dim3(nm * nn) : dim3(nn, nm);
    pd_pdl_go(pd_diar_gemm_kernel<BM, BN, WGM, WGN, ST, WT, P>, grid, dim3(WGM * WGN * 32u), (uint32_t)smem, s,
              x, w, (const uint8_t*)scale, bias, y, y2, rope, K, N, M, mode, group);
}

template <uint32_t WT, uint32_t P = 3u>
static void pd_diar_gemm_elect(const float* x, const uint8_t* w, const float* scale, const float* bias,
                               float* y, float* y2, const float2* rope, uint32_t K, uint32_t N,
                               uint32_t M, uint32_t mode, cudaStream_t s) {
    static int nsm = 0;
    if (nsm == 0) {
        int d = 0;
        cudaGetDevice(&d);
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, d);
        if (nsm <= 0) nsm = 48;
    }
    // Shape-only election, probed on GB10 (diar_gemm_gb10_bench, every arm
    // bit-identical). Four warps stacked along M (each warp its own rows, so
    // no row is split twice) over a 2-slot ring while 64 x 64 tiles give
    // every SM a CTA; two warps on 32 x 64 while those give 1.5 a SM; then
    // 16 x 64 for a stream's first windows, where the projection is
    // latency-bound and CTAs are what keep the weight stream in flight.
    // The narrow projections (N <= 512: the attention output, the MLP down,
    // the stacked-frame one) take 64 x 32 tiles from 128 rows up - 64 x 64
    // leaves them one CTA an SM at a streaming window's 300-530 rows - with a
    // three-slot ring (two CTAs an SM; down 2048 -> 512 at 320 rows 47 ->
    // 35 us, at 684 84 -> 70), two slots only where that makes the grid one
    // wave of three CTAs an SM (448-532 rows).
    // The rope epilogue needs a warp across a whole head (64 columns): the
    // same ladder with every arm 64 wide, one warp on 16 x 64 at the bottom.
    const uint64_t n64 = (N + 63u) / 64u, n32 = (N + 31u) / 32u, m64 = (M + 63u) / 64u;
#define PD_DIAR_GO(BM, BN, WM, WN, ST) \
    pd_diar_gemm_go<BM, BN, WM, WN, ST, WT, P>(x, w, scale, bias, y, y2, rope, K, N, M, mode, s)
    const uint64_t c32 = m64 * n32, sms = (uint64_t)nsm;
    // Clef's backbone planes (K 4096 / 12288, clef_gemm_gb10_bench, every arm
    // bit-identical): the long-K MLP down takes 128 x 128 tiles once they
    // fill the die (down 12288 -> 4096: 300 rows 1781 -> 1420 us, 1092 rows
    // 5635 -> 4456, 4096 rows 20.3 -> 16.0 ms); a 64 x 64 grid short of two
    // CTAs an SM takes the 3-slot ring (k/v 4096 -> 1024 at 300 rows 158 ->
    // 115 us). Below K 4096 the diarization ladder stays as it was measured.
    if (mode != PD_DIAR_ROPE && K >= 4096u) {
        const uint64_t m128 = (M + 127u) / 128u, n128 = (N + 127u) / 128u;
        if (K >= 8192u && m128 * n128 >= sms) {
            pd_diar_gemm_go<128u, 128u, 4u, 2u, 2u, WT, P>(x, w, scale, bias, y, y2, rope, K, N, M, mode, s);
            return;
        }
        if (m64 * n64 >= sms && m64 * n64 < 2u * sms) {
            pd_diar_gemm_go<64u, 64u, 4u, 1u, 3u, WT, P>(x, w, scale, bias, y, y2, rope, K, N, M, mode, s);
            return;
        }
    }
    const bool narrow = mode != PD_DIAR_ROPE && N <= 512u && 2u * c32 >= sms;
    if (narrow && c32 > 2u * sms && c32 <= 3u * sms)
        PD_DIAR_GO(64u, 32u, 4u, 1u, 2u);  // one wave at three an SM: the ring that fits them
    else if (narrow)
        PD_DIAR_GO(64u, 32u, 4u, 1u, 3u);  // two an SM: the deeper ring
    else if (m64 * n64 >= (uint64_t)nsm)
        PD_DIAR_GO(64u, 64u, 4u, 1u, 2u);
    else if (2u * (uint64_t)((M + 31u) / 32u) * n64 >= 3u * (uint64_t)nsm)
        PD_DIAR_GO(32u, 64u, 2u, 1u, 4u);
    else if (mode == PD_DIAR_ROPE)
        PD_DIAR_GO(16u, 64u, 1u, 1u, 4u);
    else
        PD_DIAR_GO(16u, 64u, 1u, 4u, 4u);
#undef PD_DIAR_GO
}

PD_EXPORT
int pd_diar_gemm(const void* x, const void* w, const void* scale, const void* bias, void* y, void* y2,
                 const void* rope, uint32_t K, uint32_t N, uint32_t M, uint32_t wtype, uint32_t mode,
                 void* stream) {
    if (M == 0 || N == 0) return 0;
    // 16 B cp.async rows: K a multiple of the k tile (= the Q8_0 block), N
    // of 4 (whole scale chunks), operands 16 B aligned; the rope epilogue
    // takes the qkv projection (N 1536) with its q plane and table
    if (K == 0 || K % 32u || N % 4u || wtype > PD_DIAR_W_Q8 || mode > PD_DIAR_GELU_TANH ||
        (mode == PD_DIAR_ROPE && (N != 1536u || !y2 || !rope)) ||
        (wtype == PD_DIAR_W_Q8 && !scale) || ((uintptr_t)x & 15u) || ((uintptr_t)w & 15u) ||
        ((uintptr_t)scale & 15u) || (M + 15u) / 16u > 65535u)
        return (int)cudaErrorInvalidValue;
    const cudaStream_t s = (cudaStream_t)stream;
    if (wtype == PD_DIAR_W_BF16)
        pd_diar_gemm_elect<PD_DIAR_W_BF16>((const float*)x, (const uint8_t*)w, nullptr, (const float*)bias,
                                           (float*)y, (float*)y2, (const float2*)rope, K, N, M, mode, s);
    else
        pd_diar_gemm_elect<PD_DIAR_W_Q8>((const float*)x, (const uint8_t*)w, (const float*)scale,
                                         (const float*)bias, (float*)y, (float*)y2, (const float2*)rope, K, N,
                                         M, mode, s);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 723 attention
// softmax(q k^T / 8) v for one window, 8 heads of 64: q [rows][512], k and v
// read in place out of rows `kvh` heads wide (the qkv rows: k at +512, v at
// +1024, kvh 24), keys 0 .. klen - 1. F32 class on the tensor cores - the
// FlashAttention-2 shape the Kumo lane probed (bench kumo_attn_tc_gb10):
// a CTA is W warps of 16 query rows of one head, each over the whole
// 64-key chunk; S = Q K^T and O += P V on m16n8k8 as 3xTF32 with
// a round-nearest drain every 32 deep (the house GEMM class); the online
// softmax in registers; P fed to the PV mma straight from S's accumulator
// (within an 8-key block slot t4 <- key 2 t4, slot t4 + 4 <- key 2 t4 + 1,
// V read with the same permutation). Q is split once into registers, K and
// V by every warp at fragment load. Row-invariant and deterministic: keys
// walk in fixed 64-key chunks from the first, whatever query tile holds a
// row. Launched as a programmatic dependent like every pass of a window.
template <uint32_t W>
__global__ void __launch_bounds__(W * 32u)
pd_diar_attn_kernel(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
                    float* __restrict__ out, uint32_t rows, uint32_t klen, uint32_t kvh) {
    constexpr uint32_t HD = 64u, H = 8u, BQ = 16u * W, BKV = 64u, SK = HD + 4u, KT8 = HD / 8u,
                       NTK = BKV / 8u, NTD = HD / 8u, THREADS = W * 32u;
    extern __shared__ __align__(16) float diar_asm[];
    float* Ks = diar_asm;       // [BKV][SK]
    float* Vs = Ks + BKV * SK;  // [BKV][SK]
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5, gr = lane >> 2, t4 = lane & 3u;
    const uint32_t head = blockIdx.y, first = blockIdx.x * BQ, count = min(BQ, rows - first);
    PD_PDL_ARM();
    // this warp's rows r0 = warp * 16 + gr and r0 + 8; Q split once into
    // registers, the A fragments of every k8 step of the head
    const uint32_t r0 = warp * 16u + gr;
    const bool any = warp * 16u < count;
    uint32_t qb[KT8][4], qs[KT8][4];
    {
        const float* q0p = q + (size_t)(first + r0) * (H * HD) + head * HD;
        const float* q1p = q0p + (size_t)8u * H * HD;
        const bool l0 = r0 < count, l1 = r0 + 8u < count;
#pragma unroll
        for (uint32_t k8 = 0; k8 < KT8; ++k8) {
            const float a[4] = {l0 ? q0p[k8 * 8u + t4] : 0.f, l1 ? q1p[k8 * 8u + t4] : 0.f,
                                l0 ? q0p[k8 * 8u + t4 + 4u] : 0.f, l1 ? q1p[k8 * 8u + t4 + 4u] : 0.f};
#pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) {
                qb[k8][e] = pd_kumo_tf32(a[e]);
                qs[k8][e] = pd_kumo_tf32(a[e] - __uint_as_float(qb[k8][e]));
            }
        }
    }
    float o[NTD][4], m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
#pragma unroll
    for (uint32_t i = 0; i < NTD; ++i)
#pragma unroll
        for (uint32_t e = 0; e < 4u; ++e) o[i][e] = 0.f;
    const size_t pitch = (size_t)kvh * HD;
    const float* kb = k + head * HD;
    const float* vb = v + head * HD;
    for (uint32_t base = 0; base < klen; base += BKV) {
        __syncthreads();  // the previous chunk's readers are done
        for (uint32_t f = tid; f < BKV * HD / 4u; f += THREADS) {
            const uint32_t kj = f / (HD / 4u), d4 = (f % (HD / 4u)) * 4u;
            float4 zk = make_float4(0.f, 0.f, 0.f, 0.f), zv = zk;
            if (base + kj < klen) {
                const size_t at = (size_t)(base + kj) * pitch + d4;
                zk = *reinterpret_cast<const float4*>(kb + at);
                zv = *reinterpret_cast<const float4*>(vb + at);
            }
            *reinterpret_cast<float4*>(&Ks[kj * SK + d4]) = zk;
            *reinterpret_cast<float4*>(&Vs[kj * SK + d4]) = zv;
        }
        __syncthreads();
        if (!any) continue;
        // S = Q K^T: k8 ascending, three mma each, a drain every 32 channels
        float s[NTK][4], acc[NTK][4];
#pragma unroll
        for (uint32_t nt = 0; nt < NTK; ++nt)
#pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) s[nt][e] = acc[nt][e] = 0.f;
#pragma unroll
        for (uint32_t k8 = 0; k8 < KT8; ++k8) {
#pragma unroll
            for (uint32_t nt = 0; nt < NTK; ++nt) {
                const float* br = Ks + (nt * 8u + gr) * SK + k8 * 8u + t4;
                const float b0 = br[0], b1 = br[4u];
                const uint32_t bb0 = pd_kumo_tf32(b0), bb1 = pd_kumo_tf32(b1);
                const uint32_t bs0 = pd_kumo_tf32(b0 - __uint_as_float(bb0));
                const uint32_t bs1 = pd_kumo_tf32(b1 - __uint_as_float(bb1));
                pd_kumo_mma(acc[nt], qb[k8], bb0, bb1);
                pd_kumo_mma(acc[nt], qb[k8], bs0, bs1);
                pd_kumo_mma(acc[nt], qs[k8], bb0, bb1);
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
        // the online softmax, rows gr (h = 0) and gr + 8 (h = 1): a lane holds
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
                    z = base + nt * 8u + 2u * t4 + c < klen ? __fmul_rn(z, 0.125f) : -INFINITY;
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
#pragma unroll
        for (uint32_t nd = 0; nd < NTD; ++nd)
#pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) o[nd][e] = __fmul_rn(o[nd][e], corr[e >> 1]);
        // O += P V, P from S's accumulator (the slot permutation above)
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
                pd_kumo_mma(pv[nd], pb, bs0, bs1);
                pd_kumo_mma(pv[nd], ps, bb0, bb1);
            }
            if (kt % 4u == 3u) {
#pragma unroll
                for (uint32_t nd = 0; nd < NTD; ++nd)
#pragma unroll
                    for (uint32_t e = 0; e < 4u; ++e) {
                        o[nd][e] = __fadd_rn(o[nd][e], pv[nd][e]);
                        pv[nd][e] = 0.f;
                    }
            }
        }
    }
    if (!any) return;
#pragma unroll
    for (uint32_t h = 0; h < 2u; ++h) {
        const uint32_t r = r0 + 8u * h;
        if (r >= count) continue;
        float* po = out + (size_t)(first + r) * (H * HD) + head * HD;
#pragma unroll
        for (uint32_t nd = 0; nd < NTD; ++nd)
            *reinterpret_cast<float2*>(po + nd * 8u + 2u * t4) =
                make_float2(__fdiv_rn(o[nd][2u * h], l[h]), __fdiv_rn(o[nd][2u * h + 1u], l[h]));
    }
}

template <uint32_t W>
static void pd_diar_attn_go(const void* q, const void* k, const void* v, void* out, uint32_t rows,
                            uint32_t klen, uint32_t kvh, cudaStream_t s) {
    constexpr uint32_t smem = 2u * 64u * (64u + 4u) * 4u;
    pd_pdl_go(pd_diar_attn_kernel<W>, dim3((rows + 16u * W - 1u) / (16u * W), 8u, 1u), dim3(W * 32u), smem, s,
              (const float*)q, (const float*)k, (const float*)v, (float*)out, rows, klen, kvh);
}

PD_EXPORT
int pd_diar_attention(const void* q, const void* k, const void* v, void* out, uint32_t rows,
                      uint32_t klen, uint32_t kvh, void* stream) {
    if (rows == 0) return 0;
    if (klen == 0 || klen > rows || kvh < 8u || (((uintptr_t)q | (uintptr_t)k | (uintptr_t)v) & 15u))
        return (int)cudaErrorInvalidValue;
    // four warps (64 queries) a CTA while that grid gives every SM one;
    // below it two warps, so a window under ~380 rows still fills the die
    // (probed on GB10, row-invariant: the same bits at any warp count)
    static int nsm = 0;
    if (nsm == 0) {
        int d = 0;
        cudaGetDevice(&d);
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, d);
        if (nsm <= 0) nsm = 48;
    }
    if ((rows + 63u) / 64u * 8u >= (uint32_t)nsm)
        pd_diar_attn_go<4u>(q, k, v, out, rows, klen, kvh, (cudaStream_t)stream);
    else
        pd_diar_attn_go<2u>(q, k, v, out, rows, klen, kvh, (cudaStream_t)stream);
    return pd_launch_status();
}
