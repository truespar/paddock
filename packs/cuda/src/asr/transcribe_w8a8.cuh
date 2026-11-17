// Nemotron 3.5 Transcribe's speech encoder in the checkpoint's own class
// (slots 961-963): ModelOpt FP8 W8A8 with static per-tensor scales. Every
// linear of the 32 blocks stores e4m3 weights `[out][in]` with an f32
// `weight_scale` and an f32 `input_scale`, both divisors (`x_fp8 = x /
// scale`); the reference serving (vLLM's NeMo plugin) quantizes each
// linear's input with that static scale - `x * (1 / scale)` in f32, clamped
// to +-448, rounded to nearest - multiplies e4m3 x e4m3 into f32 and lands
// `a_scale * (w_scale * acc) + bias`. These are the same operations, with the
// input quantization moved into whatever produced the input (the LayerNorm,
// the attention's output, the FFN up's GELU) so no plane of the coarse
// class is ever written at f16 and read back.
//
// 961  pd_tr_w8a8_gemm: Y = epi(sx * (sw * W.X)) over e4m3 W `[M][K]` and
//      e4m3 X `[N][K]`, three landings:
//        0  f16 `[N][M]`                         (q|k|v, no bias)
//        1  Y += v + bias[m], f32 `[N][M]`        (out / fc2 onto the residual)
//        2  e4m3((gelu(v + bias[m])) * inv_next)  (fc1, quantized for fc2)
// 962  pd_tr_ln_e4m3: a row LayerNorm (f32 in) landed as e4m3 at 1 / scale
// 963  pd_tr_quant_e4m3_h: an f16 plane to e4m3 at 1 / scale
//
// The GEMM is the f16 tile's twin (gemm/f16_dense.cuh, the blocked tuple
// GB10 elects): 128 x 128 outputs a block, eight warps of 64 x 32, three
// swizzled cp.async stages of 64-byte K rows, one barrier a K step. The
// stage holds the same BYTES as the f16 tile's 32-half stage, and an
// m16n8k32 e4m3 fragment is laid out over K-contiguous rows exactly as an
// m16n8k16 f16 one is (four bytes where the f16 tile has two halves), so the
// staging, the swizzle and every ldmatrix address are that kernel's - each
// mma does twice the multiplies for the same bytes. sm_89 and up (e4m3
// mma); the launcher refuses an older die.

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
#define PD_TRW8_OK 1
#else
#define PD_TRW8_OK 0
#endif

#define PD_TRW8_EPI_H16 0u
#define PD_TRW8_EPI_BIAS_RES 1u
#define PD_TRW8_EPI_GELU_Q 2u

#if PD_TRW8_OK
// D = A*B + D, m16n8k32 e4m3 x e4m3 -> f32. A = 4 regs (16 bytes: a0 = row
// g k 4t..4t+3, a1 = row g+8, a2 = row g k 16+4t.., a3 = row g+8 k 16+4t..),
// B = 2 regs (b0 = col g k 4t.., b1 = col g k 16+4t..), D as m16n8k16's.
__device__ __forceinline__ void pd_trw8_mma(float d[4], const uint32_t a[4], const uint32_t b[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
#endif

// the reference quantizer: x * (1 / scale) clamped to the e4m3 range,
// rounded to nearest (cvt.rn.satfinite)
__device__ __forceinline__ uint8_t pd_trw8_q(float v, float inv) {
    return __nv_fp8_e4m3(v * inv).__x;
}

template <uint32_t EPI>
__global__ void __launch_bounds__(256) pd_tr_w8a8_kernel(
        const uint8_t* __restrict__ W, const uint8_t* __restrict__ X, void* __restrict__ Y,
        const float* __restrict__ bias, float sw, float sx, float inv_next, uint32_t K,
        uint32_t M, uint32_t N, uint32_t gw, uint32_t gx, uint32_t nwg) {
#if PD_TRW8_OK
    constexpr uint32_t BM = 128u, BN = 128u, NTH = 256u, ST = 3u, RG = 4u, CG = 4u;
    // the stage in halves (the f16 tile's index space): 32 halves = 64 bytes
    // = two k32 steps a row
    constexpr uint32_t KT = 32u, KB = 2u * KT, NSUBK = KT / 16u, H8PR = KT / 8u;
    constexpr uint32_t WM = RG * 16u, WN = CG * 8u, WR = BM / WM;
    extern __shared__ __align__(16) __half pd_trw8_dyn[];
    auto sh_a = reinterpret_cast<__half(*)[BM * KT]>(pd_trw8_dyn);
    auto sh_b = reinterpret_cast<__half(*)[BN * KT]>(pd_trw8_dyn + ST * BM * KT);

    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, t = lane & 3u;
    const uint32_t wr = (warp % WR) * WM, wc = (warp / WR) * WN;
    // the blocked raster: grid.z walks (activation group, weight group)
    const uint32_t zx = blockIdx.z / nwg, zw = blockIdx.z - zx * nwg;
    const uint32_t row_base = (blockIdx.x + zw * gw) * BM;
    const uint32_t col_base = (blockIdx.y + zx * gx) * BN;
    if (row_base >= M || col_base >= N) return;  // uniform: a ragged group's overhang

    // stage the K bytes [k0, k0 + 64) of the tile's W rows and X rows; a
    // chunk past M / N / K is zero-filled without a read (K % 16 == 0, so a
    // chunk is never split by the K edge)
    auto stage = [&](uint32_t k0, uint32_t buf) {
        #pragma unroll
        for (uint32_t i = tid; i < BM * H8PR; i += NTH) {
            const uint32_t row = i / H8PR, h8 = (i % H8PR) * 8u, gk = k0 + 2u * h8;
            const bool ok = row_base + row < M && gk < K;
            pd_f16_cpa16(&sh_a[buf][pd_f16_sidx<KT, true>(row, h8)],
                         W + (size_t)(ok ? row_base + row : 0u) * K + (ok ? gk : 0u), ok);
        }
        #pragma unroll
        for (uint32_t i = tid; i < BN * H8PR; i += NTH) {
            const uint32_t col = i / H8PR, h8 = (i % H8PR) * 8u, gk = k0 + 2u * h8;
            const bool ok = col_base + col < N && gk < K;
            pd_f16_cpa16(&sh_b[buf][pd_f16_sidx<KT, true>(col, h8)],
                         X + (size_t)(ok ? col_base + col : 0u) * K + (ok ? gk : 0u), ok);
        }
    };

    const uint32_t l7 = lane & 7u;
    const uint32_t a_roff = ((lane & 8u) ? 8u : 0u) + l7;
    const uint32_t a_kof = (lane & 16u) ? 8u : 0u;
    const uint32_t b_kof = (lane & 8u) ? 8u : 0u;
    float acc[RG][CG][4] = {};
    auto compute = [&](uint32_t buf) {
        #pragma unroll
        for (uint32_t sk = 0; sk < NSUBK; ++sk) {
            const uint32_t ko = sk * 16u;  // 16 halves = one k32 step
            uint32_t a[RG][4], b[CG][2];
            #pragma unroll
            for (uint32_t rg = 0; rg < RG; ++rg)
                pd_f16_ldm_x4(&sh_a[buf][pd_f16_sidx<KT, true>(wr + rg * 16u + a_roff, ko + a_kof)],
                              a[rg][0], a[rg][1], a[rg][2], a[rg][3]);
            #pragma unroll
            for (uint32_t cg = 0; cg < CG; ++cg)
                pd_f16_ldm_x2(&sh_b[buf][pd_f16_sidx<KT, true>(wc + cg * 8u + l7, ko + b_kof)],
                              b[cg][0], b[cg][1]);
            #pragma unroll
            for (uint32_t rg = 0; rg < RG; ++rg)
                #pragma unroll
                for (uint32_t cg = 0; cg < CG; ++cg) pd_trw8_mma(acc[rg][cg], a[rg], b[cg]);
        }
    };

    // the ring with one barrier a K step (the f16 tile's B1)
    #pragma unroll
    for (uint32_t s = 0; s < ST - 1u; ++s) {
        if (s * KB < K) stage(s * KB, s);
        pd_f16_cpa_commit();
    }
    uint32_t p = 0;
    for (uint32_t k0 = 0; k0 < K; k0 += KB) {
        pd_f16_cpa_waitN<(int)ST - 2>();
        __syncthreads();
        const uint32_t pre = k0 + (ST - 1u) * KB;
        if (pre < K) stage(pre, (p + ST - 1u) % ST);
        pd_f16_cpa_commit();
        compute(p);
        p = (p + 1u) % ST;
    }

    // land element (m = out row, n = activation row) at Y[n * M + m]: d0 =
    // (g, 2t), d1 = (g, 2t+1), d2 = (g+8, 2t), d3 = (g+8, 2t+1)
    #pragma unroll
    for (uint32_t rg = 0; rg < RG; ++rg) {
        #pragma unroll
        for (uint32_t h = 0; h < 2u; ++h) {
            const uint32_t r = row_base + wr + rg * 16u + g + 8u * h;
            if (r >= M) continue;
            const float bm = EPI == PD_TRW8_EPI_H16 ? 0.0f : bias[r];
            #pragma unroll
            for (uint32_t cg = 0; cg < CG; ++cg) {
                #pragma unroll
                for (uint32_t e = 0; e < 2u; ++e) {
                    const uint32_t c = col_base + wc + cg * 8u + 2u * t + e;
                    if (c >= N) continue;
                    const float a = __fmul_rn(sw, acc[rg][cg][2u * h + e]);
                    const size_t o = (size_t)c * M + r;
                    if (EPI == PD_TRW8_EPI_H16) {
                        reinterpret_cast<__half*>(Y)[o] = __float2half(__fmul_rn(sx, a));
                    } else if (EPI == PD_TRW8_EPI_BIAS_RES) {
                        // the projection + bias, then the residual add (each
                        // element is this thread's alone)
                        reinterpret_cast<float*>(Y)[o] += fmaf(sx, a, bm);
                    } else {
                        reinterpret_cast<uint8_t*>(Y)[o] =
                                pd_trw8_q(pd_f16_gelu_erf(fmaf(sx, a, bm)), inv_next);
                    }
                }
            }
        }
    }
#else
    (void)W; (void)X; (void)Y; (void)bias; (void)sw; (void)sx; (void)inv_next;
    (void)K; (void)M; (void)N; (void)gw; (void)gx; (void)nwg;
#endif
}

static bool pd_trw8_die_ok() {
    static int ok = -1;
    if (ok < 0) {
        int dev = 0, ccM = 0, ccm = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&ccM, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&ccm, cudaDevAttrComputeCapabilityMinor, dev);
        ok = ccM * 10 + ccm >= 89;
    }
    return ok != 0;
}

PD_EXPORT
int pd_tr_w8a8_gemm(const void* w, const void* x, void* y, const void* bias, float sw, float sx,
                    float inv_next, uint32_t in_dim, uint32_t out_dim, uint32_t batch,
                    uint32_t epi, void* stream) {
    if (batch == 0 || out_dim == 0) return 0;
    if (in_dim == 0 || in_dim % 16u != 0 || epi > PD_TRW8_EPI_GELU_Q ||
        (epi != PD_TRW8_EPI_H16 && bias == nullptr))
        return cudaErrorInvalidValue;
    if (!pd_trw8_die_ok()) return cudaErrorNotSupported;
    constexpr uint32_t BM = 128u, BN = 128u, ST = 3u, KB = 64u;
    constexpr unsigned smem = ST * (BM + BN) * KB;  // 48 KB: two blocks an SM
    const uint32_t nwt = (out_dim + BM - 1u) / BM, nxt = (batch + BN - 1u) / BN;
    // the f16 tile's raster rule (both planes K wide; keep the smaller one
    // resident): every encoder plane fits the L2, so the weight tiles walk in
    // groups of four only when the weight plane is the wider one
    const uint32_t gw = out_dim <= batch ? nwt : (nwt < 4u ? nwt : 4u);
    const uint32_t nwg = (nwt + gw - 1u) / gw;
    const dim3 grid(gw, nxt, nwg);
    cudaStream_t st = (cudaStream_t)stream;
    const uint8_t* wp = (const uint8_t*)w;
    const uint8_t* xp = (const uint8_t*)x;
    const float* bp = (const float*)bias;
    switch (epi) {
    case PD_TRW8_EPI_H16:
        pd_tr_w8a8_kernel<PD_TRW8_EPI_H16><<<grid, 256u, smem, st>>>(
                wp, xp, y, bp, sw, sx, inv_next, in_dim, out_dim, batch, gw, nxt, nwg);
        break;
    case PD_TRW8_EPI_BIAS_RES:
        pd_tr_w8a8_kernel<PD_TRW8_EPI_BIAS_RES><<<grid, 256u, smem, st>>>(
                wp, xp, y, bp, sw, sx, inv_next, in_dim, out_dim, batch, gw, nxt, nwg);
        break;
    default:
        pd_tr_w8a8_kernel<PD_TRW8_EPI_GELU_Q><<<grid, 256u, smem, st>>>(
                wp, xp, y, bp, sw, sx, inv_next, in_dim, out_dim, batch, gw, nxt, nwg);
        break;
    }
    return pd_launch_status();
}

// 962: pd_layernorm_f16_kernel's row LayerNorm (same sums in the same
// order), its value quantized at 1 / scale instead of rounded to f16
__global__ void pd_tr_ln_e4m3_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                     const float* __restrict__ b, uint8_t* __restrict__ out,
                                     uint32_t n, float eps, float inv) {
    const uint32_t row = blockIdx.x;
    const float* xr = x + (size_t)row * n;
    uint8_t* orow = out + (size_t)row * n;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    __shared__ float wsum[32];
    __shared__ float s_mean, s_inv;
    const uint32_t lane = tid & 31u, warp = tid >> 5, nwarps = (nth + 31u) >> 5;
    float acc = 0.0f;
    for (uint32_t i = tid; i < n; i += nth) acc += xr[i];
    for (uint32_t s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, s);
    if (lane == 0) wsum[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        float m = 0.0f;
        for (uint32_t i = 0; i < nwarps; ++i) m += wsum[i];
        s_mean = m / (float)n;
    }
    __syncthreads();
    const float mean = s_mean;
    float vacc = 0.0f;
    for (uint32_t i = tid; i < n; i += nth) {
        const float d = xr[i] - mean;
        vacc += d * d;
    }
    for (uint32_t s = 16; s > 0; s >>= 1) vacc += __shfl_down_sync(0xffffffffu, vacc, s);
    if (lane == 0) wsum[warp] = vacc;
    __syncthreads();
    if (tid == 0) {
        float v = 0.0f;
        for (uint32_t i = 0; i < nwarps; ++i) v += wsum[i];
        s_inv = rsqrtf(v / (float)n + eps);
    }
    __syncthreads();
    const float is = s_inv;
    for (uint32_t i = tid; i < n; i += nth)
        orow[i] = pd_trw8_q((xr[i] - mean) * is * w[i] + b[i], inv);
}

PD_EXPORT
int pd_tr_ln_e4m3(const void* x, const void* w, const void* b, void* out, uint32_t rows,
                  uint32_t n, float eps, float inv, void* stream) {
    if (rows == 0 || n == 0) return 0;
    pd_tr_ln_e4m3_kernel<<<rows, 256, 0, (cudaStream_t)stream>>>(
            (const float*)x, (const float*)w, (const float*)b, (uint8_t*)out, n, eps, inv);
    return pd_launch_status();
}

// 963: an f16 plane to e4m3 at 1 / scale, eight elements a thread
__global__ void pd_tr_quant_e4m3_h_kernel(const __half* __restrict__ in, uint8_t* __restrict__ out,
                                          uint64_t n8, float inv) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n8) return;
    const uint4 raw = reinterpret_cast<const uint4*>(in)[i];
    const __half* h = reinterpret_cast<const __half*>(&raw);
    uint8_t q[8];
    #pragma unroll
    for (uint32_t e = 0; e < 8u; ++e) q[e] = pd_trw8_q(__half2float(h[e]), inv);
    reinterpret_cast<uint2*>(out)[i] = *reinterpret_cast<const uint2*>(q);
}

PD_EXPORT
int pd_tr_quant_e4m3_h(const void* in, void* out, uint64_t n, float inv, void* stream) {
    if (n == 0) return 0;
    if (n % 8u != 0) return cudaErrorInvalidValue;
    const uint64_t n8 = n / 8u;
    pd_tr_quant_e4m3_h_kernel<<<(unsigned)((n8 + 255u) / 256u), 256, 0, (cudaStream_t)stream>>>(
            (const __half*)in, (uint8_t*)out, n8, inv);
    return pd_launch_status();
}
