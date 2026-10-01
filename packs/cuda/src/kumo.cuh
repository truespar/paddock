// Kumo-Tabular (NVIDIA's tabular foundation model): the prepared-table graph
// at F32 - the checkpoints' own precision, the class PyTorch CUDA runs them in
// by default (TF32 matmul off). No operand is narrowed anywhere.
//
// One request's pass (host side: gpu_model/kumo):
//
//   cells  = fourier(x) . W_cell (698, batched per column) + biases, missing
//            and context-label terms (705); readout tokens broadcast (708)
//   E x { column stage: inducing block (queries: the inducing points,
//           keys: context cells) then output block (queries: every cell,
//           keys: the inducing outputs), per column
//         row stage: rows = [readout tokens | cells] (706), a row block with
//           split-half rope and gated log scaling, unpacked again }
//   norm -> row projection -> context labels (707)
//   L x ICL block (queries: every row, keys: the context rows; Test-GQA
//         query rows read the first KV heads only)
//   norm -> head (GELU) -> outputs
//
// A block: norms (699), q and fused k|v projections (698), per-head rope and
// norm (700), query scaling (701), attention (702), output projection, the
// residual add fused with the MLP norm (699), MLP up with GELU and MLP down
// accumulated into the residual (698). That is the reference train; the
// served block fuses its memory-bound passes (709-713: one statistic pass
// for every norm of a plane, the norm in the projections' A staging, the
// head transform in their epilogue with the rope from a table, the gate MLP
// and scaling in one pass) and replays their arithmetic to the bit.
//
// An ensemble's members (same rows, same columns) run as one pass: rows
// stack member-major, the column stage sees a (column, member) column per
// pair, the ICL a batch per member, and 714-717 read each member's own
// column kinds, means and labels (703's note).
//
// Row invariance: every GEMM output walks the same k sequence whatever tile
// computes it (see 698); attention walks keys in fixed 64-key
// chunks from the first key, whatever query tile holds the row. So a row's
// value never depends on how many other rows share its launch - a fitted
// context replayed later reproduces the direct pass's bits.
//
// A reimplementation of the architecture in NVIDIA/structured-data-models
// (Apache-2.0; its query scaling credits TabICLv2, BSD-3-Clause) - the
// attribution and licence texts are packs/metal/kumo.NOTICE.md, shared with
// the Metal lane, and the generated third-party notices.
//
// Plain CUDA; needs gemm/f32_qkv.cuh's pd_launch_status.

#define PD_KUMO_BK 32u
// torch.nn.RMSNorm's default eps: the F32 machine epsilon
#define PD_KUMO_NORM_EPS 1.1920928955078125e-7f

// torch.nn.GELU(): x * 0.5 * (1 + erf(x / sqrt(2))), in that order
static __device__ __forceinline__ float pd_kumo_gelu(float x) {
    return __fmul_rn(__fmul_rn(x, 0.5f), __fadd_rn(1.0f, erff(__fmul_rn(x, 0.70710678118654752440f))));
}

// The arithmetic every kernel below shares, written once so a fused kernel
// and the pass it replaces cannot differ by a bit.
//
// A row's RMS statistic: each lane squares its channels (j = lane + 32i)
// upward in one fma chain, a fixed xor tree sums the lanes, then
// rsqrt(mean + eps).
template <uint32_t V>
static __device__ __forceinline__ float pd_kumo_rms_inv(const float (&v)[V], float d, float eps) {
    float ss = 0.f;
#pragma unroll
    for (uint32_t i = 0; i < V; ++i) ss = fmaf(v[i], v[i], ss);
#pragma unroll
    for (uint32_t o = 16; o > 0; o >>= 1) ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, o));
    return rsqrtf(__fadd_rn(__fdiv_rn(ss, d), eps));
}

// The rope's (cos, sin) for a channel pair at `pos`: angle = pos * inv_freq.
// The heads kernel evaluates it in place; the fused projection reads it from
// a table (713) this same function filled.
static __device__ __forceinline__ float2 pd_kumo_rope_cs(float pos, float inv_freq) {
    const float ang = __fmul_rn(pos, inv_freq);
    return make_float2(cosf(ang), sinf(ang));
}

// One (token, head) vector of HD channels held a warp wide, lane owning
// channels lane + 32i: split-half rope when `rope` is set (x * cos +
// rotate_half(x) * sin, cs[i] the pair of channel lane + 32i), then a
// weightless RMSNorm, eps 1e-6. The rope partner (j + HD/2) % HD is this
// lane's other register at HD 64 and the lane 16 away at HD 32.
template <uint32_t HD>
static __device__ __forceinline__ void pd_kumo_head_cs(float (&v)[HD / 32u], uint32_t lane, bool rope,
                                                       const float2 (&cs)[HD / 32u]) {
    constexpr uint32_t V = HD / 32u;
    if (rope) {
        float z[V];
#pragma unroll
        for (uint32_t i = 0; i < V; ++i) {
            const uint32_t j = lane + 32u * i;
            const float partner = V == 2u ? v[i ^ 1u] : __shfl_xor_sync(0xffffffffu, v[i], 16);
            const float rot = j < HD / 2u ? -partner : partner;
            z[i] = __fadd_rn(__fmul_rn(v[i], cs[i].x), __fmul_rn(rot, cs[i].y));
        }
#pragma unroll
        for (uint32_t i = 0; i < V; ++i) v[i] = z[i];
    }
    const float inv = pd_kumo_rms_inv<V>(v, (float)HD, 1e-6f);
#pragma unroll
    for (uint32_t i = 0; i < V; ++i) v[i] = __fmul_rn(v[i], inv);
}

// The same at `pos`, the rope evaluated here (freq set: rope on).
template <uint32_t HD>
static __device__ __forceinline__ void pd_kumo_head(float (&v)[HD / 32u], uint32_t lane, float pos,
                                                    const float* __restrict__ freq) {
    float2 cs[HD / 32u];
    if (freq) {
#pragma unroll
        for (uint32_t i = 0; i < HD / 32u; ++i) cs[i] = pd_kumo_rope_cs(pos, freq[(lane + 32u * i) % (HD / 2u)]);
    }
    pd_kumo_head_cs<HD>(v, lane, freq != nullptr, cs);
}

// LogScale: q * log(max(klen, 1)) * head_scale, in that order.
static __device__ __forceinline__ float pd_kumo_log_scale(float q, float lk, float hs) {
    return __fmul_rn(__fmul_rn(q, lk), hs);
}

// ------------------------------------------------------------------ 698 GEMM
// y[m][n] = epi(sum_k x[m][k] * w[n][k] + bias[n]) over `batch` independent
// problems (blockIdx.z; x/w/y advance by their batch strides). Epilogues:
// 0 store, 1 GELU, 2 accumulate into y (the MLP-down residual: the bias is
// added before the residual, as a separate Linear then add would round), 3
// split the n axis at N/2 into y and y2 (k|v off one projection).
//
// 3xTF32 on the tensor cores - the F32-accuracy class of the house GEMM
// (gemm/f32_qkv.cuh's wide tile, whose notes carry the measurements): each
// operand splits into a tf32 big part and a tf32 residual, three mma per k8
// (big.big, big.small, small.big), and the mma chain drains into a
// round-nearest F32 accumulator once per 32-deep k tile, because the tensor
// core's own C chain truncates. The per-element sequence - k tiles
// ascending, k8 ascending inside, the three mma in that order - is the same
// in every tile shape below, so the election is shape-only and never moves a
// bit. SIMT F32 is retired as a GEMM class for this reason; cuBLAS's own
// SGEMM measured 12-13.5 TF/s on these shapes on GB10 against this class's
// ~14 TF/s roof. Probed (GB10, 2026-09-30): a fourth mma (small.small)
// moved no output of the 30 reference cases by a measurable amount and cost
// ~8% of the pass - the dropped term is not where this lane's error is; the
// three-product split stays. Against an F64 evaluation of the original model
// this lane's worst output error equals the CPU F32 reference's own.
enum : uint32_t { PD_KUMO_STORE = 0u, PD_KUMO_GELU = 1u, PD_KUMO_RESID = 2u, PD_KUMO_SPLIT = 3u };

static __device__ __forceinline__ uint32_t pd_kumo_tf32(float v) {
    uint32_t r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(v));
    return r;
}

static __device__ __forceinline__ void pd_kumo_mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0,
                                                   uint32_t b1) {
    asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// What a fused projection (710) carries on either side of its product. The
// pass it removes is replayed operation for operation, so the output is the
// unfused sequence's to the bit.
//   ainv, aw     the RMSNorm the A operand stands for: x's rows are staged
//                raw and become (x * ainv[row]) * aw[k] in shared memory, the
//                norm kernel's own product (ainv from 709 or 712)
//   alen, astride  A row m reads x row m / alen * astride + m % alen (the
//                context cells of each column; alen 0 = row m)
//   hd           the per-head rope + weightless norm (pd_kumo_head_cs) on
//                the output's head columns - all of N, or the k half of a
//                split - staged through shared memory so a head's channels
//                meet in one warp; position = row % seq as the heads kernel's
//   rope         the rope's (cos, sin) pairs [seq][hd / 2] from 713 (null: no
//                rope) - a table, so no trig runs in the projection
//   hs, klen     then the log query scaling (pd_kumo_log_scale)
struct PdKumoFuse {
    const float* ainv;
    const float* aw;
    uint32_t alen, astride;
    uint32_t hd, seq;
    const float2* rope;
    const float* hs;
    uint32_t klen;
};

static __host__ __device__ constexpr uint32_t pd_kumo_umin(uint32_t a, uint32_t b) { return a < b ? a : b; }

// The heads epilogue of a staged output tile zt (pitch st): one warp a
// (row, head) vector, the heads kernel's arithmetic (pd_kumo_head_cs, the
// rope pairs from the table, then the log scaling). A warp holds U vectors
// at once so their chains - the shuffle trees, the rsqrt - interleave; row
// groups past M are skipped whole.
template <uint32_t HD, uint32_t BM, uint32_t BN, uint32_t NW, uint32_t U>
static __device__ __forceinline__ void pd_kumo_heads_tile(const float* zt, uint32_t st, float* y,
                                                          uint32_t pitch, uint32_t m0, uint32_t n0,
                                                          uint32_t M, const PdKumoFuse& f, float lk,
                                                          uint32_t warp, uint32_t lane) {
    constexpr uint32_t NH = BN / HD, V = HD / 32u;
    static_assert(BM * NH % (NW * U) == 0, "whole vector groups");
    const uint32_t nv = min(BM, M - m0) * NH;
    for (uint32_t v0 = warp * U; v0 < nv; v0 += NW * U) {
        // loads first (every vector's channels and rope pairs in flight
        // together), then the U chains, then the stores
        float hv[U][V];
        float2 cs[U][V];
#pragma unroll
        for (uint32_t u = 0; u < U; ++u) {
            const uint32_t v = v0 + u, row = v / NH, c0 = (v % NH) * HD;
            const float2* rp = f.rope ? f.rope + (size_t)((m0 + row) % f.seq) * (HD / 2u) : nullptr;
#pragma unroll
            for (uint32_t i = 0; i < V; ++i) {
                hv[u][i] = zt[row * st + c0 + lane + 32u * i];
                cs[u][i] = rp ? rp[(lane + 32u * i) % (HD / 2u)] : make_float2(0.f, 0.f);
            }
        }
#pragma unroll
        for (uint32_t u = 0; u < U; ++u) pd_kumo_head_cs<HD>(hv[u], lane, f.rope != nullptr, cs[u]);
#pragma unroll
        for (uint32_t u = 0; u < U; ++u) {
            const uint32_t v = v0 + u, row = v / NH, c0 = (v % NH) * HD;
            if (v >= nv || n0 + c0 >= pitch) continue;
            const float hsv = f.hs ? f.hs[(n0 + c0) / HD] : 0.f;
            float* out = y + (size_t)(m0 + row) * pitch + n0 + c0;
#pragma unroll
            for (uint32_t i = 0; i < V; ++i)
                out[lane + 32u * i] = f.hs ? pd_kumo_log_scale(hv[u][i], lk, hsv) : hv[u][i];
        }
    }
}

// BM x BN per CTA, WGM x WGN warps, each a (BM/WGM) x (BN/WGN) warp tile of
// m16 x n8 fragments. Rows are staged row-major at a 36-float pitch (scalar
// fragment loads conflict-free) through an ST-slot cp.async ring (tile t +
// ST - 1 is in flight before tile t is waited for, so ST tiles ride one DRAM
// round trip); rows past M or N land as zeros. F compiles the 710 fusions
// in; the plain GEMM is F off.
template <uint32_t BM, uint32_t BN, uint32_t WGM, uint32_t WGN, uint32_t ST, bool F>
__global__ void __launch_bounds__(WGM * WGN * 32u)
pd_kumo_gemm_kernel(const float* __restrict__ x, const float* __restrict__ w,
                    const float* __restrict__ bias, float* __restrict__ y, float* __restrict__ y2,
                    uint32_t K, uint32_t N, uint32_t M, uint32_t mode,
                    uint64_t x_bs, uint64_t w_bs, uint64_t y_bs, PdKumoFuse f) {
    constexpr uint32_t BK = PD_KUMO_BK, SK = BK + 4u, THREADS = WGM * WGN * 32u;
    constexpr uint32_t WTM = BM / WGM, WTN = BN / WGN, MT = WTM / 16u, NT = WTN / 8u;
    extern __shared__ __align__(16) float kumo_gsm[];
    static_assert(ST >= 2u, "a ring");
    float* xs = kumo_gsm;                   // [ST][BM][SK]
    float* ws = kumo_gsm + ST * BM * SK;    // [ST][BN][SK]
    const uint64_t bz = blockIdx.z;
    x += bz * x_bs;
    w += bz * w_bs;
    y += bz * y_bs;
    if (y2) y2 += bz * y_bs;
    const uint32_t m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const uint32_t gr = lane >> 2, t4 = lane & 3u;
    const uint32_t wm = warp / WGN, wn = warp % WGN;
    float acc[MT][NT][4], fac[MT][NT][4];
#pragma unroll
    for (uint32_t i = 0; i < MT; ++i)
#pragma unroll
        for (uint32_t j = 0; j < NT; ++j)
#pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) acc[i][j][e] = fac[i][j][e] = 0.f;
    // A thread stages the same 16 B column of the same rows in every k tile
    // (rows r0, r0 + RPI, ...), so the row addresses - the fused row gather
    // and the norm's row statistics with them - resolve once, here, and the
    // k loop only advances them. Rows past M or N stage zeros.
    constexpr uint32_t CPR = BK / 4u, RPI = THREADS / CPR;  // 16 B chunks a row, rows a sweep
    constexpr uint32_t XCH = BM / RPI, WCH = BN / RPI;
    static_assert(BM % RPI == 0 && BN % RPI == 0 && XCH + WCH <= 32u, "rows split evenly over the threads");
    const uint32_t kc = (tid % CPR) * 4u, r0 = tid / CPR;
    const float* xp[XCH];
    const float* wp[WCH];
    uint32_t live = 0;
    float ainv[F ? XCH : 1];
#pragma unroll
    for (uint32_t i = 0; i < XCH; ++i) {
        const uint32_t m = m0 + r0 + i * RPI;
        const uint32_t xr = m >= M ? 0u : F && f.alen ? m / f.alen * f.astride + m % f.alen : m;
        xp[i] = x + (size_t)xr * K + kc;
        live |= (m < M ? 1u : 0u) << i;
        // the fused norm's row statistic
        if (F) ainv[i] = f.ainv && m < M ? f.ainv[xr] : 0.f;
    }
#pragma unroll
    for (uint32_t i = 0; i < WCH; ++i) {
        const uint32_t n = n0 + r0 + i * RPI;
        wp[i] = w + (size_t)(n < N ? n : 0u) * K + kc;
        live |= (n < N ? 1u : 0u) << (XCH + i);
    }
    auto stage = [&](uint32_t slot, uint32_t k0) {
#pragma unroll
        for (uint32_t i = 0; i < XCH + WCH; ++i) {
            const bool isx = i < XCH;
            const uint32_t r = r0 + (isx ? i : i - XCH) * RPI;
            const float* src = (isx ? xp[i] : wp[i - XCH]) + k0;
            float* dst = (isx ? xs + slot * BM * SK : ws + slot * BN * SK) + r * SK + kc;
            const unsigned sm = (unsigned)__cvta_generic_to_shared(dst);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(sm), "l"(src),
                         "r"((live >> i) & 1u ? 16u : 0u));
        }
    };
    // the fused norm's weights for this thread's 16 B column: a register
    // pair, the next tile's loaded while this one multiplies
    const float4* awp = F && f.ainv ? reinterpret_cast<const float4*>(f.aw + kc) : nullptr;
    float4 awc = awp ? awp[0] : make_float4(0.f, 0.f, 0.f, 0.f);
    // Tiles 0 .. ST-2 up front; then tile t + ST - 1 goes out (into the slot
    // tile t - 1 left) before tile t is waited for. Every step commits a
    // group, empty past K, so "ST - 1 groups pending" always means tile t
    // has landed.
#pragma unroll
    for (uint32_t s = 0; s + 1u < ST; ++s) {
        if (s * BK < K) stage(s, s * BK);
        asm volatile("cp.async.commit_group;" ::: "memory");
    }
    for (uint32_t t = 0, k0 = 0; k0 < K; ++t, k0 += BK) {
        const uint32_t slot = t % ST;
        const bool more = k0 + BK < K;
        if (k0 + (ST - 1u) * BK < K) stage((t + ST - 1u) % ST, k0 + (ST - 1u) * BK);
        asm volatile("cp.async.commit_group;" ::: "memory");
        const float4 awn = F && awp && more ? awp[(k0 + BK) / 4u] : awc;
        asm volatile("cp.async.wait_group %0;" ::"n"(ST - 1u) : "memory");
        if (F && awp) {
            // each thread normalizes the chunks it staged itself (its own
            // copies are complete after the wait; the barrier publishes
            // them) - the norm kernel's product, in its order
            float* xsn = xs + slot * BM * SK;
            const float4 wv = awc;
#pragma unroll
            for (uint32_t i = 0; i < XCH; ++i) {
                if (!((live >> i) & 1u)) continue;
                float* p = xsn + (r0 + i * RPI) * SK + kc;
                p[0] = __fmul_rn(__fmul_rn(p[0], ainv[i]), wv.x);
                p[1] = __fmul_rn(__fmul_rn(p[1], ainv[i]), wv.y);
                p[2] = __fmul_rn(__fmul_rn(p[2], ainv[i]), wv.z);
                p[3] = __fmul_rn(__fmul_rn(p[3], ainv[i]), wv.w);
            }
        }
        __syncthreads();
        const float* xsl = xs + slot * BM * SK;
        const float* wsl = ws + slot * BN * SK;
#pragma unroll
        for (uint32_t k8 = 0; k8 < BK; k8 += 8u) {
            uint32_t ab[MT][4], as[MT][4];
#pragma unroll
            for (uint32_t mt = 0; mt < MT; ++mt) {
                const float* ar = xsl + (wm * WTM + mt * 16u + gr) * SK + k8 + t4;
                const float a[4] = {ar[0], ar[8u * SK], ar[4u], ar[8u * SK + 4u]};
#pragma unroll
                for (uint32_t e = 0; e < 4u; ++e) {
                    ab[mt][e] = pd_kumo_tf32(a[e]);
                    as[mt][e] = pd_kumo_tf32(a[e] - __uint_as_float(ab[mt][e]));
                }
            }
#pragma unroll
            for (uint32_t nt = 0; nt < NT; ++nt) {
                const float* br = wsl + (wn * WTN + nt * 8u + gr) * SK + k8 + t4;
                const float b0 = br[0], b1 = br[4u];
                const uint32_t bb0 = pd_kumo_tf32(b0), bb1 = pd_kumo_tf32(b1);
                const uint32_t bs0 = pd_kumo_tf32(b0 - __uint_as_float(bb0));
                const uint32_t bs1 = pd_kumo_tf32(b1 - __uint_as_float(bb1));
#pragma unroll
                for (uint32_t mt = 0; mt < MT; ++mt) {
                    pd_kumo_mma(acc[mt][nt], ab[mt], bb0, bb1);
                    pd_kumo_mma(acc[mt][nt], ab[mt], bs0, bs1);
                    pd_kumo_mma(acc[mt][nt], as[mt], bb0, bb1);
                }
            }
        }
#pragma unroll
        for (uint32_t mt = 0; mt < MT; ++mt)
#pragma unroll
            for (uint32_t nt = 0; nt < NT; ++nt)
#pragma unroll
                for (uint32_t e = 0; e < 4u; ++e) {
                    fac[mt][nt][e] = __fadd_rn(fac[mt][nt][e], acc[mt][nt][e]);
                    acc[mt][nt][e] = 0.f;
                }
        __syncthreads();  // tile t's slot is restaged next iteration
        if (F) awc = awn;
    }
    const uint32_t half = N / 2u;
    if (F && f.hd && n0 < (mode == PD_KUMO_SPLIT ? half : N)) {
        // A head's channels are spread over warps: stage the tile's outputs
        // (the plain epilogue's z) in the free staging memory, then one warp
        // a (row, head) vector runs the heads kernel's arithmetic on it.
        constexpr uint32_t ZP = BN + 4u;
        float* zt = kumo_gsm;
#pragma unroll
        for (uint32_t mt = 0; mt < MT; ++mt)
#pragma unroll
            for (uint32_t nt = 0; nt < NT; ++nt)
#pragma unroll
                for (uint32_t e = 0; e < 4u; ++e) {
                    const uint32_t ml = wm * WTM + mt * 16u + gr + (e >= 2u ? 8u : 0u);
                    const uint32_t nl = wn * WTN + nt * 8u + 2u * t4 + (e & 1u);
                    float z = fac[mt][nt][e];
                    if (bias && n0 + nl < N) z = __fadd_rn(z, bias[n0 + nl]);
                    zt[ml * ZP + nl] = z;
                }
        __syncthreads();
        const uint32_t pitch = mode == PD_KUMO_SPLIT ? half : N;
        const float lk = f.hs ? logf((float)max(f.klen, 1u)) : 0.f;
        // (the election keeps BN a multiple of the head width)
        constexpr uint32_t NW = THREADS / 32u;
        if constexpr (BN >= 64u) {
            if (f.hd == 64u)
                pd_kumo_heads_tile<64u, BM, BN, NW, pd_kumo_umin(4u, BM * (BN / 64u) / NW)>(
                    zt, ZP, y, pitch, m0, n0, M, f, lk, warp, lane);
        }
        if constexpr (BN >= 32u) {
            if (f.hd == 32u)
                pd_kumo_heads_tile<32u, BM, BN, NW, pd_kumo_umin(8u, BM * (BN / 32u) / NW)>(
                    zt, ZP, y, pitch, m0, n0, M, f, lk, warp, lane);
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
                if (mode == PD_KUMO_RESID) {
                    float* o = y + (size_t)m * N + n;
                    *o = __fadd_rn(*o, z);
                } else if (mode == PD_KUMO_SPLIT) {
                    if (n < half) y[(size_t)m * half + n] = z;
                    else y2[(size_t)m * half + n - half] = z;
                } else {
                    y[(size_t)m * N + n] = z;
                }
            }
}

template <uint32_t BM, uint32_t BN, uint32_t WGM, uint32_t WGN, uint32_t ST, bool F>
static void pd_kumo_gemm_go(const float* x, const float* w, const float* bias, float* y, float* y2,
                            uint32_t K, uint32_t N, uint32_t M, uint32_t mode, uint32_t batch,
                            uint64_t x_bs, uint64_t w_bs, uint64_t y_bs, const PdKumoFuse& f,
                            cudaStream_t s) {
    constexpr size_t smem = ST * (BM + BN) * (PD_KUMO_BK + 4u) * sizeof(float);
    // the heads epilogue's staged tile reuses this memory
    static_assert(BM * (BN + 4u) * sizeof(float) <= smem, "zt must fit the staging ring");
    static bool set = false;
    if (!set) {
        cudaFuncSetAttribute(pd_kumo_gemm_kernel<BM, BN, WGM, WGN, ST, F>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        set = true;
    }
    const dim3 grid((N + BN - 1u) / BN, (M + BM - 1u) / BM, batch);
    pd_kumo_gemm_kernel<BM, BN, WGM, WGN, ST, F><<<grid, WGM * WGN * 32u, smem, s>>>(
        x, w, bias, y, y2, K, N, M, mode, x_bs, w_bs, y_bs, f);
}

// Shape-only election (every tile walks k identically, so the choice never
// moves a bit), probed on GB10 with weights cold in DRAM (2026-09-30). At the
// row counts Kumo's small tables run at - the ICL layers are context + query
// rows - a projection streams its weights once and is bound by how many CTAs
// keep that stream in flight, not by the mma:
//   - 32 x 64 (four warps across N, 2-slot ring) while that grid still gives
//     every SM a CTA: a 64-row tile leaves most of its mma on dead rows over
//     too few CTAs;
//   - below it, 16-row tiles with a 4-slot ring - 16 x 32, or 16 x 64 when a
//     head of 64 must land in one tile - so a 22-row plane still reaches the
//     SMs (past one wave they would re-read the weights per 16 rows and lose
//     to 32 x 64);
//   - past 2048 rows, 64 x 64. The 128-row tiles the first cut elected
//     measured no better at any Kumo shape, so they are gone.
template <bool F>
static void pd_kumo_gemm_elect(const float* x, const float* w, const float* bias, float* y, float* y2,
                               uint32_t K, uint32_t N, uint32_t M, uint32_t mode, uint32_t batch,
                               uint64_t x_bs, uint64_t w_bs, uint64_t y_bs, const PdKumoFuse& f,
                               cudaStream_t s) {
    static int nsm = 0;
    if (nsm == 0) {
        int d = 0;
        cudaGetDevice(&d);
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, d);
        if (nsm <= 0) nsm = 48;
    }
    const uint64_t tiles32 = (uint64_t)((M + 31u) / 32u) * ((N + 63u) / 64u) * batch;
    if (M > 2048u)
        pd_kumo_gemm_go<64u, 64u, 2u, 2u, 2u, F>(x, w, bias, y, y2, K, N, M, mode, batch, x_bs, w_bs, y_bs, f, s);
    else if (tiles32 >= (uint64_t)nsm)
        pd_kumo_gemm_go<32u, 64u, 1u, 4u, 2u, F>(x, w, bias, y, y2, K, N, M, mode, batch, x_bs, w_bs, y_bs, f, s);
    else if (F && f.hd == 64u)
        pd_kumo_gemm_go<16u, 64u, 1u, 4u, 3u, F>(x, w, bias, y, y2, K, N, M, mode, batch, x_bs, w_bs, y_bs, f, s);
    else
        pd_kumo_gemm_go<16u, 32u, 1u, 4u, 4u, F>(x, w, bias, y, y2, K, N, M, mode, batch, x_bs, w_bs, y_bs, f, s);
}

PD_EXPORT
int pd_kumo_gemm(const void* x, const void* w, const void* bias, void* y, void* y2, uint32_t K,
                 uint32_t N, uint32_t M, uint32_t mode, uint32_t batch, uint64_t x_bs,
                 uint64_t w_bs, uint64_t y_bs, void* stream) {
    if (M == 0 || N == 0 || batch == 0) return 0;
    // 16 B cp.async rows: K a multiple of the k tile, operands 16 B aligned
    if (K == 0 || K % PD_KUMO_BK || mode > PD_KUMO_SPLIT || (mode == PD_KUMO_SPLIT && (!y2 || N % 2u)) ||
        ((uintptr_t)x & 15u) || ((uintptr_t)w & 15u) || (x_bs % 4u) || (w_bs % 4u) ||
        batch > 65535u || (M + 63u) / 64u > 65535u)
        return (int)cudaErrorInvalidValue;
    pd_kumo_gemm_elect<false>((const float*)x, (const float*)w, (const float*)bias, (float*)y,
                              (float*)y2, K, N, M, mode, batch, x_bs, w_bs, y_bs, PdKumoFuse{},
                              (cudaStream_t)stream);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 699 norms
// RMSNorm over D channels, eps = F32 epsilon (torch.nn.RMSNorm's default):
// y = x * rsqrt(mean(x^2) + eps) * w. One warp a row; each lane sums its
// channels upward, then a fixed xor tree.
//   b null: x is gathered, source row = row / len * stride + row % len (the
//           context rows of each column, `stride` rows apart)
//   b set : the residual add first, res[row] = a[row % period] + b[row]
//           (period = rows when the addend is not broadcast), y = norm(res)
template <uint32_t V>
__global__ void __launch_bounds__(256)
pd_kumo_norm_kernel(const float* __restrict__ a, const float* __restrict__ b,
                    const float* __restrict__ w, float* __restrict__ res, float* __restrict__ y,
                    uint32_t rows, uint32_t len, uint32_t stride, uint32_t period) {
    constexpr uint32_t D = V * 32u;
    const uint32_t row = blockIdx.x * 8u + threadIdx.x / 32u, lane = threadIdx.x % 32u;
    if (row >= rows) return;
    float v[V];
    if (b) {
        const float* pa = a + (size_t)(row % period) * D;
        const float* pb = b + (size_t)row * D;
#pragma unroll
        for (uint32_t i = 0; i < V; ++i) {
            v[i] = __fadd_rn(pa[lane + 32u * i], pb[lane + 32u * i]);
            res[(size_t)row * D + lane + 32u * i] = v[i];
        }
    } else {
        const float* pa = a + ((size_t)(row / len) * stride + row % len) * D;
#pragma unroll
        for (uint32_t i = 0; i < V; ++i) v[i] = pa[lane + 32u * i];
    }
    const float inv = pd_kumo_rms_inv<V>(v, (float)D, PD_KUMO_NORM_EPS);
#pragma unroll
    for (uint32_t i = 0; i < V; ++i)
        y[(size_t)row * D + lane + 32u * i] = __fmul_rn(__fmul_rn(v[i], inv), w[lane + 32u * i]);
}

PD_EXPORT
int pd_kumo_norm(const void* a, const void* b, const void* w, void* res, void* y, uint32_t D,
                 uint32_t rows, uint32_t len, uint32_t stride, uint32_t period, void* stream) {
    if (rows == 0) return 0;
    if ((b && (!res || period == 0)) || (!b && len == 0)) return (int)cudaErrorInvalidValue;
    const dim3 grid((rows + 7u) / 8u);
    const cudaStream_t s = (cudaStream_t)stream;
#define PD_KUMO_NORM(V)                                                                             \
    pd_kumo_norm_kernel<V><<<grid, 256, 0, s>>>((const float*)a, (const float*)b, (const float*)w, \
                                                (float*)res, (float*)y, rows, len, stride, period)
    switch (D) {
    case 128: PD_KUMO_NORM(4); break;
    case 256: PD_KUMO_NORM(8); break;
    case 512: PD_KUMO_NORM(16); break;
    case 1024: PD_KUMO_NORM(32); break;
    default: return (int)cudaErrorInvalidValue;
    }
#undef PD_KUMO_NORM
    return pd_launch_status();
}

// ------------------------------------------------------------------ 700 heads
// Per (token, head) vector of HD channels (pd_kumo_head): optional
// split-half rope at position token % seq, then a weightless RMSNorm. One warp
// a vector. The fused path carries this in the projection's epilogue (710).
template <uint32_t HD>
__global__ void __launch_bounds__(256)
pd_kumo_heads_kernel(const float* __restrict__ x, const float* __restrict__ freq,
                     float* __restrict__ y, uint32_t H, uint32_t seq, uint32_t vecs, uint32_t rope) {
    constexpr uint32_t V = HD / 32u;
    const uint32_t vec = blockIdx.x * 8u + threadIdx.x / 32u, lane = threadIdx.x % 32u;
    if (vec >= vecs) return;
    const float* px = x + (size_t)vec * HD;
    float v[V];
#pragma unroll
    for (uint32_t i = 0; i < V; ++i) v[i] = px[lane + 32u * i];
    pd_kumo_head<HD>(v, lane, (float)((vec / H) % seq), rope ? freq : nullptr);
#pragma unroll
    for (uint32_t i = 0; i < V; ++i) y[(size_t)vec * HD + lane + 32u * i] = v[i];
}

PD_EXPORT
int pd_kumo_heads(const void* x, const void* freq, void* y, uint32_t H, uint32_t hd, uint32_t seq,
                  uint32_t vecs, uint32_t rope, void* stream) {
    if (vecs == 0) return 0;
    if (H == 0 || seq == 0 || (rope && !freq)) return (int)cudaErrorInvalidValue;
    const dim3 grid((vecs + 7u) / 8u);
    const cudaStream_t s = (cudaStream_t)stream;
    if (hd == 32)
        pd_kumo_heads_kernel<32u><<<grid, 256, 0, s>>>((const float*)x, (const float*)freq, (float*)y, H, seq, vecs, rope);
    else if (hd == 64)
        pd_kumo_heads_kernel<64u><<<grid, 256, 0, s>>>((const float*)x, (const float*)freq, (float*)y, H, seq, vecs, rope);
    else
        return (int)cudaErrorInvalidValue;
    return pd_launch_status();
}

// ------------------------------------------------------------------ 701 query scaling
// LogScale / GatedLogScale: q * log(max(klen, 1)) * head_scale[h], then
// * (1 + tanh(gate)) when `gate` is set (the gate MLP's output, same layout).
__global__ void pd_kumo_scale_kernel(float* __restrict__ q, const float* __restrict__ hs,
                                     const float* __restrict__ gate, uint64_t n, uint32_t hd,
                                     uint32_t H, uint32_t klen) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float lk = logf((float)max(klen, 1u));
    float z = pd_kumo_log_scale(q[i], lk, hs[(i / hd) % H]);
    if (gate) z = __fmul_rn(z, __fadd_rn(1.0f, tanhf(gate[i])));
    q[i] = z;
}

PD_EXPORT
int pd_kumo_scale(void* q, const void* head_scale, const void* gate, uint64_t n, uint32_t hd,
                  uint32_t H, uint32_t klen, void* stream) {
    if (n == 0) return 0;
    if (hd == 0 || H == 0) return (int)cudaErrorInvalidValue;
    pd_kumo_scale_kernel<<<(unsigned)((n + 255u) / 256u), 256, 0, (cudaStream_t)stream>>>(
        (float*)q, (const float*)head_scale, (const float*)gate, n, hd, H, klen);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 702 attention
// softmax(q k^T / sqrt(HD)) v, online over 64-key chunks - no score plane.
// One CTA = 64 queries of one (batch, head); 256 threads.
//   q   [B][q_brows][H][HD] rows at b * q_brows + pos (q_brows 0: every
//       batch shares one query set - the inducing points)
//   k,v [B][klen][KVH][HD]; out [B][qlen][H][HD]
//   queries q0 .. q0 + qcount of each batch; the whole launch reads KV head
//   `head / (H / qkvh)` when qkvh is set (Test-GQA query rows), else `head`
// Thread (tq, tk) = (tid / 16, tid % 16) owns queries tq + 16i and keys
// tk + 16j (i, j < 4): a quarter-warp reads one query row (a broadcast) and
// eight key rows 4 banks apart (the key plane's row pitch is HD + 4). A
// row's statistics reduce over its 16 lanes by a fixed xor tree; PV gives the
// same thread those queries and channels tk * (HD/16).... The probabilities
// reuse the key tile's shared memory once the scores are taken.
template <uint32_t HD>
__global__ void __launch_bounds__(256)
pd_kumo_attn_kernel(const float* __restrict__ q, const float* __restrict__ k,
                    const float* __restrict__ v, float* __restrict__ out, uint32_t H,
                    uint32_t qlen, uint32_t klen, uint32_t kvh, uint32_t q_brows, uint32_t qkvh,
                    uint32_t q0, uint32_t qcount, float scale) {
    constexpr uint32_t BQ = 64u, BKV = 64u, SK = HD + 4u, DV = HD / 16u;
    extern __shared__ __align__(16) float kumo_sm[];
    float* Qs = kumo_sm;                          // [BQ][HD]
    float* Ks = Qs + BQ * HD;                     // [BKV][SK]
    float* Vs = Ks + (BKV * SK > BKV * BQ ? BKV * SK : BKV * BQ);  // [BKV][HD]
    float* Ps = Ks;                               // [BKV][BQ], query slot tq*4 + i
    const uint32_t tid = threadIdx.x, tq = tid / 16u, tk = tid % 16u;
    const uint32_t head = blockIdx.y, b = blockIdx.z;
    const uint32_t first = q0 + blockIdx.x * BQ, count = min(BQ, q0 + qcount - first);
    const uint32_t kv = qkvh ? head / (H / qkvh) : head;
    if (klen == 1u) {
        // a one-key softmax is exactly one: out = v, and q is never read
        // (the host skips the query path for it)
        const float* vr = v + ((size_t)b * kvh + kv) * HD;
        for (uint32_t f = tid; f < count * HD; f += 256u)
            out[(((size_t)b * qlen + first + f / HD) * H + head) * HD + f % HD] = vr[f % HD];
        return;
    }
    for (uint32_t f = tid; f < BQ * HD / 4u; f += 256u) {
        const uint32_t qi = f / (HD / 4u), d4 = (f % (HD / 4u)) * 4u;
        float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
        if (qi < count)
            z = *reinterpret_cast<const float4*>(
                q + (((size_t)b * q_brows + first + qi) * H + head) * HD + d4);
        *reinterpret_cast<float4*>(&Qs[qi * HD + d4]) = z;
    }
    float o[4][DV], m[4], l[4];
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
        m[i] = -INFINITY;
        l[i] = 0.f;
#pragma unroll
        for (uint32_t d = 0; d < DV; ++d) o[i][d] = 0.f;
    }
    const float* kb = k + (size_t)b * klen * kvh * HD;
    const float* vb = v + (size_t)b * klen * kvh * HD;
    for (uint32_t base = 0; base < klen; base += BKV) {
        __syncthreads();  // the previous chunk's readers are done
        for (uint32_t f = tid; f < BKV * HD / 4u; f += 256u) {
            const uint32_t kj = f / (HD / 4u), d4 = (f % (HD / 4u)) * 4u;
            float4 zk = make_float4(0.f, 0.f, 0.f, 0.f), zv = zk;
            if (base + kj < klen) {
                const size_t at = ((size_t)(base + kj) * kvh + kv) * HD + d4;
                zk = *reinterpret_cast<const float4*>(kb + at);
                zv = *reinterpret_cast<const float4*>(vb + at);
            }
            *reinterpret_cast<float4*>(&Ks[kj * SK + d4]) = zk;
            *reinterpret_cast<float4*>(&Vs[kj * HD + d4]) = zv;
        }
        __syncthreads();
        float s[4][4];
#pragma unroll
        for (uint32_t i = 0; i < 4; ++i)
#pragma unroll
            for (uint32_t j = 0; j < 4; ++j) s[i][j] = 0.f;
#pragma unroll 4
        for (uint32_t d4 = 0; d4 < HD; d4 += 4u) {
            float4 qa[4], kc[4];
#pragma unroll
            for (uint32_t i = 0; i < 4; ++i)
                qa[i] = *reinterpret_cast<const float4*>(&Qs[(tq + 16u * i) * HD + d4]);
#pragma unroll
            for (uint32_t j = 0; j < 4; ++j)
                kc[j] = *reinterpret_cast<const float4*>(&Ks[(tk + 16u * j) * SK + d4]);
#pragma unroll
            for (uint32_t i = 0; i < 4; ++i)
#pragma unroll
                for (uint32_t j = 0; j < 4; ++j) {
                    s[i][j] = fmaf(qa[i].x, kc[j].x, s[i][j]);
                    s[i][j] = fmaf(qa[i].y, kc[j].y, s[i][j]);
                    s[i][j] = fmaf(qa[i].z, kc[j].z, s[i][j]);
                    s[i][j] = fmaf(qa[i].w, kc[j].w, s[i][j]);
                }
        }
        float corr[4];
#pragma unroll
        for (uint32_t i = 0; i < 4; ++i) {
            float hi = m[i];
#pragma unroll
            for (uint32_t j = 0; j < 4; ++j) {
                s[i][j] = base + tk + 16u * j < klen ? __fmul_rn(s[i][j], scale) : -INFINITY;
                hi = fmaxf(hi, s[i][j]);
            }
#pragma unroll
            for (uint32_t x = 1; x < 16; x <<= 1) hi = fmaxf(hi, __shfl_xor_sync(0xffffffffu, hi, x));
            corr[i] = m[i] == -INFINITY ? 0.f : expf(__fsub_rn(m[i], hi));
            float sum = 0.f;
#pragma unroll
            for (uint32_t j = 0; j < 4; ++j) {
                s[i][j] = s[i][j] == -INFINITY ? 0.f : expf(__fsub_rn(s[i][j], hi));
                sum = __fadd_rn(sum, s[i][j]);
            }
#pragma unroll
            for (uint32_t x = 1; x < 16; x <<= 1) sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, x));
            l[i] = fmaf(l[i], corr[i], sum);
            m[i] = hi;
        }
        __syncthreads();  // every score is taken: the key tile becomes P
#pragma unroll
        for (uint32_t j = 0; j < 4; ++j)
            *reinterpret_cast<float4*>(&Ps[(tk + 16u * j) * BQ + tq * 4u]) =
                make_float4(s[0][j], s[1][j], s[2][j], s[3][j]);
        __syncthreads();
#pragma unroll
        for (uint32_t i = 0; i < 4; ++i)
#pragma unroll
            for (uint32_t d = 0; d < DV; ++d) o[i][d] = __fmul_rn(o[i][d], corr[i]);
#pragma unroll 4
        for (uint32_t kj = 0; kj < BKV; ++kj) {
            const float4 p = *reinterpret_cast<const float4*>(&Ps[kj * BQ + tq * 4u]);
            const float pp[4] = {p.x, p.y, p.z, p.w};
            float vv[DV];
#pragma unroll
            for (uint32_t d = 0; d < DV; d += 2u) {
                const float2 t = *reinterpret_cast<const float2*>(&Vs[kj * HD + tk * DV + d]);
                vv[d] = t.x;
                vv[d + 1] = t.y;
            }
#pragma unroll
            for (uint32_t i = 0; i < 4; ++i)
#pragma unroll
                for (uint32_t d = 0; d < DV; ++d) o[i][d] = fmaf(pp[i], vv[d], o[i][d]);
        }
    }
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
        const uint32_t qi = tq + 16u * i;
        if (qi >= count) continue;
        float* po = out + (((size_t)b * qlen + first + qi) * H + head) * HD + tk * DV;
#pragma unroll
        for (uint32_t d = 0; d < DV; ++d) po[d] = __fdiv_rn(o[i][d], l[i]);
    }
}

template <uint32_t HD>
static constexpr size_t pd_kumo_attn_smem() {
    return (64u * HD + (64u * (HD + 4u) > 64u * 64u ? 64u * (HD + 4u) : 64u * 64u) + 64u * HD) * sizeof(float);
}

PD_EXPORT
int pd_kumo_attention(const void* q, const void* k, const void* v, void* out, uint32_t H,
                      uint32_t hd, uint32_t batch, uint32_t qlen, uint32_t klen, uint32_t kvh,
                      uint32_t q_brows, uint32_t qkvh, uint32_t q0, uint32_t qcount, void* stream) {
    if (qcount == 0 || batch == 0) return 0;
    if (klen == 0 || H == 0 || kvh == 0 || q0 + qcount > qlen || batch > 65535u || H > 65535u ||
        (qkvh && (H % qkvh || qkvh > kvh)) || (!qkvh && kvh < H) ||
        (((uintptr_t)q | (uintptr_t)k | (uintptr_t)v) & 15u))
        return (int)cudaErrorInvalidValue;
    const dim3 grid((qcount + 63u) / 64u, H, batch);
    const cudaStream_t s = (cudaStream_t)stream;
    if (hd == 32) {
        constexpr size_t smem = pd_kumo_attn_smem<32u>();
        pd_kumo_attn_kernel<32u><<<grid, 256, smem, s>>>((const float*)q, (const float*)k, (const float*)v, (float*)out,
            H, qlen, klen, kvh, q_brows, qkvh, q0, qcount, 1.0f / sqrtf(32.0f));
    } else if (hd == 64) {
        constexpr size_t smem = pd_kumo_attn_smem<64u>();
        static bool set = false;
        if (!set) { cudaFuncSetAttribute(pd_kumo_attn_kernel<64u>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem); set = true; }
        pd_kumo_attn_kernel<64u><<<grid, 256, smem, s>>>((const float*)q, (const float*)k, (const float*)v, (float*)out,
            H, qlen, klen, kvh, q_brows, qkvh, q0, qcount, 0.125f);
    } else {
        return (int)cudaErrorInvalidValue;
    }
    return pd_launch_status();
}

// ------------------------------------------------------------------ 703-705 cell embedding
// A cell's features: group g in 0..2 reads the column (col + 2^g - 1) % C,
// the missing cells imputed with the context mean; each group has 32
// frequencies (numerical or categorical by the SOURCE column), sin then cos.
// fourier [C][R][192]
//
// Members (714-717): an ensemble's same-shape tables in one pass. Their rows
// stack, member-major - row R' = e * rpm + r - so each column's cells hold
// every member's rows ([C][E][rpm]: the column stage sees C * E columns of
// rpm rows, the row stage E * rpm rows of C cells), and the per-column facts
// the members do not share - the categorical flags, the context means, the
// labels - are read at [e][col] ([e][row] for labels). One member (rpm = R)
// is the single-table layout exactly.
__global__ void pd_kumo_fourier_kernel(const float* __restrict__ x, const float* __restrict__ means,
                                       const uint32_t* __restrict__ cat, const float* __restrict__ num_freq,
                                       const float* __restrict__ cat_freq, float* __restrict__ out,
                                       uint32_t R, uint32_t C, uint32_t rpm) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)C * R * 192u) return;
    const uint32_t f = i % 192u, row = (uint32_t)((i / 192u) % R), col = (uint32_t)(i / (192u * (uint64_t)R));
    const uint32_t g = f / 64u, j = f % 64u, src = (col + (1u << g) - 1u) % C;
    const size_t mc = (size_t)(row / rpm) * C + src;  // this member's column
    float z = x[(size_t)row * C + src];
    if (isnan(z)) z = means[mc];
    const float a = __fmul_rn(z, (cat[mc] ? cat_freq : num_freq)[g * 32u + j % 32u]);
    out[i] = j < 32u ? sinf(a) : cosf(a);
}

static int pd_kumo_fourier_go(const void* x, const void* means, const void* cat, const void* num_freq,
                              const void* cat_freq, void* out, uint32_t R, uint32_t C, uint32_t rpm,
                              void* stream) {
    const uint64_t n = (uint64_t)C * R * 192u;
    if (n == 0) return 0;
    if (rpm == 0 || R % rpm) return (int)cudaErrorInvalidValue;
    pd_kumo_fourier_kernel<<<(unsigned)((n + 255u) / 256u), 256, 0, (cudaStream_t)stream>>>(
        (const float*)x, (const float*)means, (const uint32_t*)cat, (const float*)num_freq,
        (const float*)cat_freq, (float*)out, R, C, rpm);
    return pd_launch_status();
}

PD_EXPORT
int pd_kumo_fourier(const void* x, const void* means, const void* cat, const void* num_freq,
                    const void* cat_freq, void* out, uint32_t R, uint32_t C, void* stream) {
    return pd_kumo_fourier_go(x, means, cat, num_freq, cat_freq, out, R, C, R, stream);
}

// Each column's [D][192] projection: group g's 64 features project through
// the numerical or categorical Linear by the group's source column. Members:
// out [C][E][D][192], a (column, member) pair per batched GEMM problem.
__global__ void pd_kumo_cell_weights_kernel(const uint32_t* __restrict__ cat, const float* __restrict__ nw,
                                            const float* __restrict__ cw, float* __restrict__ out,
                                            uint32_t C, uint32_t D, uint32_t E) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)C * E * D * 192u) return;
    const uint32_t f = i % 192u, j = (uint32_t)((i / 192u) % D);
    const uint64_t ce = i / (192u * (uint64_t)D);
    const uint32_t col = (uint32_t)(ce / E), e = (uint32_t)(ce % E);
    const uint32_t src = (col + (1u << (f / 64u)) - 1u) % C;
    out[i] = (cat[(size_t)e * C + src] ? cw : nw)[(size_t)j * 64u + f % 64u];
}

static int pd_kumo_cell_weights_go(const void* cat, const void* nw, const void* cw, void* out, uint32_t C,
                                   uint32_t D, uint32_t E, void* stream) {
    const uint64_t n = (uint64_t)C * E * D * 192u;
    if (n == 0) return 0;
    pd_kumo_cell_weights_kernel<<<(unsigned)((n + 255u) / 256u), 256, 0, (cudaStream_t)stream>>>(
        (const uint32_t*)cat, (const float*)nw, (const float*)cw, (float*)out, C, D, E);
    return pd_launch_status();
}

PD_EXPORT
int pd_kumo_cell_weights(const void* cat, const void* nw, const void* cw, void* out, uint32_t C,
                         uint32_t D, void* stream) {
    return pd_kumo_cell_weights_go(cat, nw, cw, out, C, D, 1u, stream);
}

// cells [C][R][D] += the three groups' biases, the missing-cell projection of
// every group whose source cell is missing, and - context rows only - the
// label embedding (classification: row y of the table; regression: y * w).
// Members: row R' = e * rpm + r is member e's row r, its label y[e * nc + r].
__global__ void pd_kumo_cell_bias_kernel(float* __restrict__ out, const float* __restrict__ x,
                                         const uint32_t* __restrict__ cat, const float* __restrict__ nb,
                                         const float* __restrict__ cb, const float* __restrict__ missing,
                                         const float* __restrict__ target, const float* __restrict__ y,
                                         uint32_t R, uint32_t C, uint32_t D, uint32_t nc, uint32_t cls,
                                         uint32_t rpm) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)C * R * D) return;
    const uint32_t j = i % D, row = (uint32_t)((i / D) % R), col = (uint32_t)(i / ((uint64_t)D * R));
    const uint32_t e = row / rpm, r = row % rpm;
    float b = 0.f, n = 0.f;
#pragma unroll
    for (uint32_t g = 0; g < 3; ++g) {
        const uint32_t src = (col + (1u << g) - 1u) % C;
        b = __fadd_rn(b, (cat[(size_t)e * C + src] ? cb : nb)[j]);
        if (isnan(x[(size_t)row * C + src])) n = __fadd_rn(n, missing[j * 3u + g]);
    }
    float z = __fadd_rn(__fadd_rn(out[i], b), n);
    if (r < nc) {
        const float yr = y[(size_t)e * nc + r];
        z = __fadd_rn(z, cls ? target[(size_t)yr * D + j] : __fmul_rn(target[j], yr));
    }
    out[i] = z;
}

static int pd_kumo_cell_bias_go(void* out, const void* x, const void* cat, const void* nb, const void* cb,
                                const void* missing, const void* target, const void* y, uint32_t R,
                                uint32_t C, uint32_t D, uint32_t nc, uint32_t cls, uint32_t rpm,
                                void* stream) {
    const uint64_t n = (uint64_t)C * R * D;
    if (n == 0) return 0;
    if (rpm == 0 || R % rpm || nc > rpm) return (int)cudaErrorInvalidValue;
    pd_kumo_cell_bias_kernel<<<(unsigned)((n + 255u) / 256u), 256, 0, (cudaStream_t)stream>>>(
        (float*)out, (const float*)x, (const uint32_t*)cat, (const float*)nb, (const float*)cb,
        (const float*)missing, (const float*)target, (const float*)y, R, C, D, nc, cls, rpm);
    return pd_launch_status();
}

PD_EXPORT
int pd_kumo_cell_bias(void* out, const void* x, const void* cat, const void* nb, const void* cb,
                      const void* missing, const void* target, const void* y, uint32_t R, uint32_t C,
                      uint32_t D, uint32_t nc, uint32_t cls, void* stream) {
    return pd_kumo_cell_bias_go(out, x, cat, nb, cb, missing, target, y, R, C, D, nc, cls, R, stream);
}

// ------------------------------------------------------------------ 706 row axis
// pack (dir 0): rows [R][C+4][D] = [readout tokens | cells]; unpack (dir 1)
// the other way. cells are [C][R][D], the readout tokens [R][4][D].
__global__ void pd_kumo_rows_kernel(float* __restrict__ cells, float* __restrict__ cls,
                                    float* __restrict__ rows, uint32_t R, uint32_t C, uint32_t D,
                                    uint32_t dir) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)R * (C + 4u) * D) return;
    const uint32_t j = i % D, c = (uint32_t)((i / D) % (C + 4u)), r = (uint32_t)(i / ((uint64_t)D * (C + 4u)));
    float* other = c < 4u ? cls + ((size_t)r * 4u + c) * D + j : cells + ((size_t)(c - 4u) * R + r) * D + j;
    if (dir) *other = rows[i];
    else rows[i] = *other;
}

PD_EXPORT
int pd_kumo_rows(void* cells, void* cls, void* rows, uint32_t R, uint32_t C, uint32_t D,
                 uint32_t dir, void* stream) {
    const uint64_t n = (uint64_t)R * (C + 4u) * D;
    if (n == 0) return 0;
    pd_kumo_rows_kernel<<<(unsigned)((n + 255u) / 256u), 256, 0, (cudaStream_t)stream>>>(
        (float*)cells, (float*)cls, (float*)rows, R, C, D, dir);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 707 ICL labels
// x [rows][D]: the first nc rows (context) gain their label embedding.
// Members: E blocks of rpm rows, block e's first nc rows labelled from
// y[e * nc ..].
__global__ void pd_kumo_labels_kernel(float* __restrict__ x, const float* __restrict__ y,
                                      const float* __restrict__ target, uint32_t D, uint32_t nc,
                                      uint32_t cls, uint32_t rpm, uint32_t E) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)E * nc * D) return;
    const uint32_t lr = (uint32_t)(i / D), j = i % D, e = lr / nc, r = lr % nc;
    float* xp = x + ((size_t)e * rpm + r) * D + j;
    const float yr = y[lr];
    *xp = __fadd_rn(*xp, cls ? target[(size_t)yr * D + j] : __fmul_rn(target[j], yr));
}

static int pd_kumo_labels_go(void* x, const void* y, const void* target, uint32_t D, uint32_t nc,
                             uint32_t cls, uint32_t rpm, uint32_t E, void* stream) {
    const uint64_t n = (uint64_t)E * nc * D;
    if (n == 0) return 0;
    if (nc > rpm) return (int)cudaErrorInvalidValue;
    pd_kumo_labels_kernel<<<(unsigned)((n + 255u) / 256u), 256, 0, (cudaStream_t)stream>>>(
        (float*)x, (const float*)y, (const float*)target, D, nc, cls, rpm, E);
    return pd_launch_status();
}

PD_EXPORT
int pd_kumo_labels(void* x, const void* y, const void* target, uint32_t D, uint32_t nc, uint32_t cls,
                   void* stream) {
    return pd_kumo_labels_go(x, y, target, D, nc, cls, nc, 1u, stream);
}

// ------------------------------------------------------------------ 708 copy
// dst[r][c] = src[(r % period)][c] over rows x width, each side with its own
// row stride: the readout-token and inducing-point broadcasts, and the first
// KV heads of a projection kept for a fitted context.
__global__ void pd_kumo_copy_kernel(const float* __restrict__ src, float* __restrict__ dst,
                                    uint32_t rows, uint32_t width, uint64_t src_stride,
                                    uint64_t dst_stride, uint32_t period) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)rows * width) return;
    const uint32_t r = (uint32_t)(i / width), c = i % width;
    dst[r * dst_stride + c] = src[(r % period) * src_stride + c];
}

PD_EXPORT
int pd_kumo_copy(const void* src, void* dst, uint32_t rows, uint32_t width, uint64_t src_stride,
                 uint64_t dst_stride, uint32_t period, void* stream) {
    const uint64_t n = (uint64_t)rows * width;
    if (n == 0) return 0;
    if (period == 0) return (int)cudaErrorInvalidValue;
    pd_kumo_copy_kernel<<<(unsigned)((n + 255u) / 256u), 256, 0, (cudaStream_t)stream>>>(
        (const float*)src, (float*)dst, rows, width, src_stride, dst_stride, period);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 709 row statistics
// The RMSNorm statistic alone: inv[row] = rsqrt(mean(x^2) + eps) over D
// channels (pd_kumo_rms_inv, the norm kernel's own), for a projection that
// applies the norm while it stages its operand (710). The statistic does not
// depend on the norm's weight, so one pass serves every norm of a plane - a
// block's query and key/value norms read the same rows.
//   b null: x = a[row]
//   b set : the residual add first, res[row] = a[row % period] + b[row]
template <uint32_t V>
__global__ void __launch_bounds__(256)
pd_kumo_stats_kernel(const float* __restrict__ a, const float* __restrict__ b,
                     float* __restrict__ res, float* __restrict__ inv, uint32_t rows,
                     uint32_t period) {
    constexpr uint32_t D = V * 32u;
    const uint32_t row = blockIdx.x * 8u + threadIdx.x / 32u, lane = threadIdx.x % 32u;
    if (row >= rows) return;
    float v[V];
    if (b) {
        const float* pa = a + (size_t)(row % period) * D;
        const float* pb = b + (size_t)row * D;
#pragma unroll
        for (uint32_t i = 0; i < V; ++i) {
            v[i] = __fadd_rn(pa[lane + 32u * i], pb[lane + 32u * i]);
            res[(size_t)row * D + lane + 32u * i] = v[i];
        }
    } else {
#pragma unroll
        for (uint32_t i = 0; i < V; ++i) v[i] = a[(size_t)row * D + lane + 32u * i];
    }
    const float r = pd_kumo_rms_inv<V>(v, (float)D, PD_KUMO_NORM_EPS);
    if (lane == 0) inv[row] = r;
}

PD_EXPORT
int pd_kumo_stats(const void* a, const void* b, void* res, void* inv, uint32_t D, uint32_t rows,
                  uint32_t period, void* stream) {
    if (rows == 0) return 0;
    if (b && (!res || period == 0)) return (int)cudaErrorInvalidValue;
    const dim3 grid((rows + 7u) / 8u);
    const cudaStream_t s = (cudaStream_t)stream;
#define PD_KUMO_STATS(V)                                                                          \
    pd_kumo_stats_kernel<V><<<grid, 256, 0, s>>>((const float*)a, (const float*)b, (float*)res, \
                                                 (float*)inv, rows, period)
    switch (D) {
    case 128: PD_KUMO_STATS(4); break;
    case 256: PD_KUMO_STATS(8); break;
    case 512: PD_KUMO_STATS(16); break;
    case 1024: PD_KUMO_STATS(32); break;
    default: return (int)cudaErrorInvalidValue;
    }
#undef PD_KUMO_STATS
    return pd_launch_status();
}

// ------------------------------------------------------------------ 710 fused projection
// The 698 GEMM (batch 1) with the PdKumoFuse fusions: the RMSNorm on the A
// operand (ainv null: none; the rows may be gathered) and the per-head
// transform on the output (hd 0: none). A null rope table (713, [seq][hd/2]
// pairs) skips the rope, a null hs the log scaling. The head columns (N, or
// the k half of a split) are a multiple of 128, so no output tile holds both
// head and plain columns.
PD_EXPORT
int pd_kumo_gemm_fused(const void* x, const void* w, const void* bias, void* y, void* y2,
                       uint32_t K, uint32_t N, uint32_t M, uint32_t mode, const void* ainv,
                       const void* aw, uint32_t alen, uint32_t astride, uint32_t hd, uint32_t seq,
                       const void* rope, const void* hs, uint32_t klen, void* stream) {
    if (M == 0 || N == 0) return 0;
    const uint32_t hcols = mode == PD_KUMO_SPLIT ? N / 2u : N;
    if (K == 0 || K % PD_KUMO_BK || mode > PD_KUMO_SPLIT || (mode == PD_KUMO_SPLIT && (!y2 || N % 2u)) ||
        ((uintptr_t)x & 15u) || ((uintptr_t)w & 15u) || (M + 63u) / 64u > 65535u ||
        (ainv && (!aw || ((uintptr_t)aw & 15u))) || (alen && astride < alen) || ((uintptr_t)rope & 7u) ||
        (hd && ((hd != 32u && hd != 64u) || hcols % 128u || seq == 0 ||
                (mode != PD_KUMO_STORE && mode != PD_KUMO_SPLIT))))
        return (int)cudaErrorInvalidValue;
    const PdKumoFuse f{(const float*)ainv, (const float*)aw, alen, astride, hd, seq,
                       (const float2*)rope, (const float*)hs, klen};
    pd_kumo_gemm_elect<true>((const float*)x, (const float*)w, (const float*)bias, (float*)y,
                             (float*)y2, K, N, M, mode, 1u, 0u, 0u, 0u, f, (cudaStream_t)stream);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 711 gated query scaling
// GatedLogScale in one pass, in place over (token, head) vectors of HD:
//   q = (q * log(klen) * hs[head]) * (1 + tanh(G2 . gelu(G0 . q + b0) + b2))
// where q is the normed head vector. Replaces the gate GEMM pair and the
// scale kernel; the two products run the 698 GEMM's exact sequence (k tiles
// of 32 from zero, three mma per k8 - big.big, big.small, small.big - a
// round-nearest drain per tile, the bias added after), so the gate, and with
// it q, is the two-GEMM path's to the bit. A CTA takes 64 vectors, 2 x 2
// warps of 32 x 32 (32 x 16 on the HD 32 second product); rows are staged at
// a pitch of 4 mod 32, conflict-free for the fragment loads.
// One product of the gate MLP: fac[64 x NT*16] = A[64][K] . B[.][K]^T from
// shared memory (pitches PA / PB), warp (wm, wn) of 2 x 2 owning a 32 x NT*8
// tile - the 698 GEMM's per-element sequence exactly (see 711).
template <uint32_t NT>
static __device__ __forceinline__ void pd_kumo_qgate_product(const float* A, uint32_t PA,
                                                             const float* B, uint32_t PB, uint32_t K,
                                                             uint32_t wm, uint32_t wn, uint32_t gr,
                                                             uint32_t t4, float (&fac)[2][NT][4]) {
    constexpr uint32_t WTN = NT * 8u;
#pragma unroll
    for (uint32_t i = 0; i < 2u; ++i)
#pragma unroll
        for (uint32_t j = 0; j < NT; ++j)
#pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) fac[i][j][e] = 0.f;
    for (uint32_t k0 = 0; k0 < K; k0 += PD_KUMO_BK) {
        float acc[2][NT][4];
#pragma unroll
        for (uint32_t i = 0; i < 2u; ++i)
#pragma unroll
            for (uint32_t j = 0; j < NT; ++j)
#pragma unroll
                for (uint32_t e = 0; e < 4u; ++e) acc[i][j][e] = 0.f;
#pragma unroll
        for (uint32_t k8 = 0; k8 < PD_KUMO_BK; k8 += 8u) {
            uint32_t ab[2][4], as[2][4];
#pragma unroll
            for (uint32_t mt = 0; mt < 2u; ++mt) {
                const float* ar = A + (wm * 32u + mt * 16u + gr) * PA + k0 + k8 + t4;
                const float a[4] = {ar[0], ar[8u * PA], ar[4u], ar[8u * PA + 4u]};
#pragma unroll
                for (uint32_t e = 0; e < 4u; ++e) {
                    ab[mt][e] = pd_kumo_tf32(a[e]);
                    as[mt][e] = pd_kumo_tf32(a[e] - __uint_as_float(ab[mt][e]));
                }
            }
#pragma unroll
            for (uint32_t nt = 0; nt < NT; ++nt) {
                const float* br = B + (wn * WTN + nt * 8u + gr) * PB + k0 + k8 + t4;
                const float b0 = br[0], b1 = br[4u];
                const uint32_t bb0 = pd_kumo_tf32(b0), bb1 = pd_kumo_tf32(b1);
                const uint32_t bs0 = pd_kumo_tf32(b0 - __uint_as_float(bb0));
                const uint32_t bs1 = pd_kumo_tf32(b1 - __uint_as_float(bb1));
#pragma unroll
                for (uint32_t mt = 0; mt < 2u; ++mt) {
                    pd_kumo_mma(acc[mt][nt], ab[mt], bb0, bb1);
                    pd_kumo_mma(acc[mt][nt], ab[mt], bs0, bs1);
                    pd_kumo_mma(acc[mt][nt], as[mt], bb0, bb1);
                }
            }
        }
#pragma unroll
        for (uint32_t i = 0; i < 2u; ++i)
#pragma unroll
            for (uint32_t j = 0; j < NT; ++j)
#pragma unroll
                for (uint32_t e = 0; e < 4u; ++e) fac[i][j][e] = __fadd_rn(fac[i][j][e], acc[i][j][e]);
    }
}

template <uint32_t HD>
__global__ void __launch_bounds__(128)
pd_kumo_qgate_kernel(float* __restrict__ q, const float* __restrict__ g0w,
                     const float* __restrict__ g0b, const float* __restrict__ g2w,
                     const float* __restrict__ g2b, const float* __restrict__ hs, uint32_t vecs,
                     uint32_t H, uint32_t klen) {
    constexpr uint32_t BV = 64u, HID = 64u, SA = HD + 4u, SH = HID + 4u;
    constexpr uint32_t R0 = (BV * SA > BV * SH ? BV * SA : BV * SH);
    constexpr uint32_t R1 = (HID * SA > HD * SH ? HID * SA : HD * SH);
    __shared__ __align__(16) float sm0[R0];  // the vectors, then the hidden activations
    __shared__ __align__(16) float sm1[R1];  // G0, then G2
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const uint32_t gr = lane >> 2, t4 = lane & 3u, wm = warp >> 1, wn = warp & 1u;
    const uint32_t v0 = blockIdx.x * BV;
    // stage the vectors (zeros past vecs) and G0 [64][HD]
    for (uint32_t c = tid; c < BV * HD / 4u; c += 128u) {
        const uint32_t r = c / (HD / 4u), k = (c % (HD / 4u)) * 4u;
        const float4 z = v0 + r < vecs ? *reinterpret_cast<const float4*>(q + (size_t)(v0 + r) * HD + k)
                                       : make_float4(0.f, 0.f, 0.f, 0.f);
        *reinterpret_cast<float4*>(&sm0[r * SA + k]) = z;
    }
    for (uint32_t c = tid; c < HID * HD / 4u; c += 128u) {
        const uint32_t r = c / (HD / 4u), k = (c % (HD / 4u)) * 4u;
        *reinterpret_cast<float4*>(&sm1[r * SA + k]) = *reinterpret_cast<const float4*>(g0w + (size_t)r * HD + k);
    }
    __syncthreads();
    // hidden = gelu(q . G0^T + b0): 64 x 64, 2 x 2 warps of 32 x 32
    float h1[2][4][4];
    pd_kumo_qgate_product<4u>(sm0, SA, sm1, SA, HD, wm, wn, gr, t4, h1);
    __syncthreads();  // both tiles read: the hidden plane and G2 take their places
#pragma unroll
    for (uint32_t mt = 0; mt < 2u; ++mt)
#pragma unroll
        for (uint32_t nt = 0; nt < 4u; ++nt)
#pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) {
                const uint32_t r = wm * 32u + mt * 16u + gr + (e >= 2u ? 8u : 0u);
                const uint32_t n = wn * 32u + nt * 8u + 2u * t4 + (e & 1u);
                sm0[r * SH + n] = pd_kumo_gelu(__fadd_rn(h1[mt][nt][e], g0b[n]));
            }
    for (uint32_t c = tid; c < HD * HID / 4u; c += 128u) {
        const uint32_t r = c / (HID / 4u), k = (c % (HID / 4u)) * 4u;
        *reinterpret_cast<float4*>(&sm1[r * SH + k]) = *reinterpret_cast<const float4*>(g2w + (size_t)r * HID + k);
    }
    __syncthreads();
    // gate = hidden . G2^T + b2, then the scaled query, written in place
    constexpr uint32_t NT2 = HD / 16u;
    float g[2][NT2][4];
    pd_kumo_qgate_product<NT2>(sm0, SH, sm1, SH, HID, wm, wn, gr, t4, g);
    const float lk = logf((float)max(klen, 1u));
#pragma unroll
    for (uint32_t mt = 0; mt < 2u; ++mt)
#pragma unroll
        for (uint32_t nt = 0; nt < NT2; ++nt)
#pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) {
                const uint32_t r = wm * 32u + mt * 16u + gr + (e >= 2u ? 8u : 0u);
                const uint32_t n = wn * (NT2 * 8u) + nt * 8u + 2u * t4 + (e & 1u);
                const uint32_t vec = v0 + r;
                if (vec >= vecs) continue;
                float* o = q + (size_t)vec * HD + n;
                const float z = pd_kumo_log_scale(*o, lk, hs[vec % H]);
                *o = __fmul_rn(z, __fadd_rn(1.0f, tanhf(__fadd_rn(g[mt][nt][e], g2b[n]))));
            }
}

PD_EXPORT
int pd_kumo_qgate(void* q, const void* g0w, const void* g0b, const void* g2w, const void* g2b,
                  const void* head_scale, uint32_t vecs, uint32_t hd, uint32_t H, uint32_t klen,
                  void* stream) {
    if (vecs == 0) return 0;
    if (H == 0 || (((uintptr_t)q | (uintptr_t)g0w | (uintptr_t)g2w) & 15u)) return (int)cudaErrorInvalidValue;
    const dim3 grid((vecs + 63u) / 64u);
    const cudaStream_t s = (cudaStream_t)stream;
    if (hd == 32)
        pd_kumo_qgate_kernel<32u><<<grid, 128, 0, s>>>((float*)q, (const float*)g0w, (const float*)g0b,
            (const float*)g2w, (const float*)g2b, (const float*)head_scale, vecs, H, klen);
    else if (hd == 64)
        pd_kumo_qgate_kernel<64u><<<grid, 128, 0, s>>>((float*)q, (const float*)g0w, (const float*)g0b,
            (const float*)g2w, (const float*)g2b, (const float*)head_scale, vecs, H, klen);
    else
        return (int)cudaErrorInvalidValue;
    return pd_launch_status();
}

// ------------------------------------------------------------------ 712 row axis with statistics
// 706's pack (dir 0) / unpack (dir 1), one warp a row, emitting the RMSNorm
// statistic of every row it lands (709's arithmetic) - the next block's
// projections normalize on the fly, so the copy is the statistics pass.
//   pack  : inv[t] for rows-layout row t = r * (C + 4) + c
//   unpack: inv[(c - 4) * R + r] for each cell row (readout tokens are
//           re-packed, and counted, before anything reads them again)
template <uint32_t V>
__global__ void __launch_bounds__(256)
pd_kumo_rows_stats_kernel(float* __restrict__ cells, float* __restrict__ cls,
                          float* __restrict__ rows, float* __restrict__ inv, uint32_t R, uint32_t C,
                          uint32_t dir) {
    constexpr uint32_t D = V * 32u;
    const uint32_t t = blockIdx.x * 8u + threadIdx.x / 32u, lane = threadIdx.x % 32u;
    if (t >= R * (C + 4u)) return;
    const uint32_t r = t / (C + 4u), c = t % (C + 4u);
    float* other = c < 4u ? cls + ((size_t)r * 4u + c) * D : cells + ((size_t)(c - 4u) * R + r) * D;
    float* row = rows + (size_t)t * D;
    float v[V];
#pragma unroll
    for (uint32_t i = 0; i < V; ++i) {
        v[i] = dir ? row[lane + 32u * i] : other[lane + 32u * i];
        if (dir) other[lane + 32u * i] = v[i];
        else row[lane + 32u * i] = v[i];
    }
    if (dir && c < 4u) return;
    const float s = pd_kumo_rms_inv<V>(v, (float)D, PD_KUMO_NORM_EPS);
    if (lane == 0) inv[dir ? (size_t)(c - 4u) * R + r : (size_t)t] = s;
}

PD_EXPORT
int pd_kumo_rows_stats(void* cells, void* cls, void* rows, void* inv, uint32_t R, uint32_t C,
                       uint32_t D, uint32_t dir, void* stream) {
    const uint32_t n = R * (C + 4u);
    if (n == 0) return 0;
    const dim3 grid((n + 7u) / 8u);
    const cudaStream_t s = (cudaStream_t)stream;
    if (D == 128)
        pd_kumo_rows_stats_kernel<4u><<<grid, 256, 0, s>>>((float*)cells, (float*)cls, (float*)rows, (float*)inv, R, C, dir);
    else if (D == 256)
        pd_kumo_rows_stats_kernel<8u><<<grid, 256, 0, s>>>((float*)cells, (float*)cls, (float*)rows, (float*)inv, R, C, dir);
    else
        return (int)cudaErrorInvalidValue;
    return pd_launch_status();
}

// ------------------------------------------------------------------ 713 rope table
// The rope's (cos, sin) pairs for positions 0..seq-1 of one inv_freq vector
// of `half` entries: table[pos][jj] = pd_kumo_rope_cs(pos, freq[jj]), the
// very function the heads kernel evaluates in place, so a projection that
// reads the table (710) rotates to the heads kernel's bits.
__global__ void pd_kumo_rope_table_kernel(const float* __restrict__ freq, float2* __restrict__ table,
                                          uint32_t half, uint32_t seq) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)half * seq) return;
    table[i] = pd_kumo_rope_cs((float)(uint32_t)(i / half), freq[i % half]);
}

PD_EXPORT
int pd_kumo_rope_table(const void* freq, void* table, uint32_t half, uint32_t seq, void* stream) {
    if (half == 0 || seq == 0) return 0;
    if (((uintptr_t)table & 7u)) return (int)cudaErrorInvalidValue;
    const uint64_t n = (uint64_t)half * seq;
    pd_kumo_rope_table_kernel<<<(unsigned)((n + 255u) / 256u), 256, 0, (cudaStream_t)stream>>>(
        (const float*)freq, (float2*)table, half, seq);
    return pd_launch_status();
}

// ------------------------------------------------------------------ 714-717 members
// 703, 704, 705 and 707 over an ensemble's members in one pass (the layout is
// 703's note): R rows of `rpm` rows a member, E = R / rpm members.
PD_EXPORT
int pd_kumo_fourier_m(const void* x, const void* means, const void* cat, const void* num_freq,
                      const void* cat_freq, void* out, uint32_t R, uint32_t C, uint32_t rpm,
                      void* stream) {
    return pd_kumo_fourier_go(x, means, cat, num_freq, cat_freq, out, R, C, rpm, stream);
}

PD_EXPORT
int pd_kumo_cell_weights_m(const void* cat, const void* nw, const void* cw, void* out, uint32_t C,
                           uint32_t D, uint32_t E, void* stream) {
    return pd_kumo_cell_weights_go(cat, nw, cw, out, C, D, E, stream);
}

PD_EXPORT
int pd_kumo_cell_bias_m(void* out, const void* x, const void* cat, const void* nb, const void* cb,
                        const void* missing, const void* target, const void* y, uint32_t R, uint32_t C,
                        uint32_t D, uint32_t nc, uint32_t cls, uint32_t rpm, void* stream) {
    return pd_kumo_cell_bias_go(out, x, cat, nb, cb, missing, target, y, R, C, D, nc, cls, rpm, stream);
}

PD_EXPORT
int pd_kumo_labels_m(void* x, const void* y, const void* target, uint32_t D, uint32_t nc, uint32_t cls,
                     uint32_t rpm, uint32_t E, void* stream) {
    return pd_kumo_labels_go(x, y, target, D, nc, cls, rpm, E, stream);
}
