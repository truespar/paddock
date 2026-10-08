// gemm/mmq_small.cuh - the mmq Q8_0 GEMM at medium row counts (slot 828)
// Textually-included segment of the single pack translation unit (after
// gemm/mmq.cuh, whose PD_MMQ_XK / PD_MMQ_YK layout and quantized activation
// format it shares).
//
// pd_q8_0_gemm_mmq_kernel's 128 x 128 output tile is one block an SM, so a
// pass of a few hundred rows over a narrow weight is a handful of blocks on
// the whole die: EmbeddingGemma 2's 512-wide outputs at a picture's 270 rows
// are 12 blocks on GB10's 48 SMs, 96 such launches a pass. The stream-k arm
// fills the die by splitting K, which regroups each output's sum - so a
// sequence's vector would depend on what it was packed with, and the encoders
// that ride this class promise it does not. This is the other way to fill it:
// the SAME tile at a smaller output footprint, BM rows x BN columns, with
// (BM / 32) x 2 warps of the very same micro-tile (a warp pair on a 32-row
// strip, each warp every other 8-column group of 16), K staged 256 int8 deep,
// the activation chunk streamed in two 128-int8 halves. Every output therefore
// takes the Q8_0 blocks in the same order with the same arithmetic - the exact
// int32 block dot from mma.m16n8k32, then acc = fma(dA * dB, (float)dot, acc)
// (the FMUL + FFMA the base tile's `acc += dA * dB * (float)d` compiles to,
// pinned here so a different surrounding cannot contract it another way) -
// and lands bit-identical to the base tile at any shape (packs/cuda/bench
// mmq_small_bench.cu). Plain tiling only: no K split, no bias, no fixup.

template <uint32_t BM, uint32_t BN>
__global__ void __launch_bounds__(BM * 2u) pd_q8_0_gemm_mmq_s_kernel(
        const int8_t* __restrict__ data, const __half* __restrict__ scale,
        const uint8_t* __restrict__ yq, float* __restrict__ y, uint32_t in_dim,
        uint32_t out_dim, uint32_t batch) {
#if PD_MMA_OK
    static_assert(BM % 32u == 0u && BN % 16u == 0u && BN <= 128u, "tile shape");
    constexpr uint32_t NT = BM * 2u;   // (BM / 32) warp pairs
    static_assert((BM * 64u) % NT == 0u && (BM * 8u) % NT == 0u, "staging trip counts");
    extern __shared__ int pd_mmqs_sh[];
    int* tile_y = pd_mmqs_sh;                    // BN cols x 36 int32
    int* tile_x = pd_mmqs_sh + BN * PD_MMQ_YK;   // BM rows x 76 int32

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, t = lane & 3u;
    const uint32_t i0 = (warp >> 1) * 32u;    // warp pair's 32-row strip
    const uint32_t joff = (warp & 1u) * 8u;   // which 8-col group of each 16
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t n_k32 = in_dim >> 2;
    const uint32_t n_blocks = in_dim >> 5;
    const uint32_t n_chunks = (in_dim + 127u) / 128u;
    const uint32_t nk = (in_dim + 255u) >> 8;
    // column tiles fastest: one weight strip's column tiles run together, so
    // the strip is read once from DRAM and re-read from L2
    const uint32_t col_base = blockIdx.x * BN;
    const uint32_t row_base = blockIdx.y * BM;

    float acc[BN / 8u][4] = {};
    for (uint32_t kt = 0; kt < nk; ++kt) {
        // the weight tile: BM rows x 64 int32 (256 int8 of K) + 8 scales
#pragma unroll
        for (uint32_t it = 0; it < BM * 64u / NT; ++it) {
            const uint32_t i = it * NT + tid;
            const uint32_t row = i >> 6, k = i & 63u, gk = kt * 64u + k;
            tile_x[row * PD_MMQ_XK + k] = (gk < n_k32 && (row_base + row) < out_dim)
                ? ((const int*)(data + (size_t)(row_base + row) * in_dim))[gk] : 0;
        }
#pragma unroll
        for (uint32_t it = 0; it < BM * 8u / NT; ++it) {
            const uint32_t i = it * NT + tid;
            const uint32_t row = i >> 3, b = i & 7u, gb = kt * 8u + b;
            ((float*)tile_x)[row * PD_MMQ_XK + 64u + b] =
                (gb < n_blocks && (row_base + row) < out_dim)
                ? __half2float(scale[(size_t)(row_base + row) * n_blocks + gb]) : 0.f;
        }
#pragma unroll
        for (uint32_t h = 0; h < 2u; ++h) {
            // one 128-int8 activation chunk of the tile's BN columns: a flat
            // contiguous copy (the quantizer pads columns to 128)
            const uint32_t chunk = kt * 2u + h;
            const int* by = (const int*)(yq + ((size_t)chunk * batch_pad + col_base) * 144u);
#pragma unroll
            for (uint32_t it = 0; it < (BN * PD_MMQ_YK + NT - 1u) / NT; ++it) {
                const uint32_t l = it * NT + tid;
                if (l < BN * PD_MMQ_YK) tile_y[l] = (chunk < n_chunks) ? by[l] : 0;
            }
            __syncthreads();   // h == 0 also covers the tile_x stores above

            const uint32_t k00 = h * 32u;
            int A[2][4][4];
            float dA[2][2][4];
#pragma unroll
            for (uint32_t n = 0; n < 2u; ++n) {
                const uint32_t r0 = (i0 + n * 16u + g) * PD_MMQ_XK;
                const uint32_t r8 = (i0 + n * 16u + 8u + g) * PD_MMQ_XK;
#pragma unroll
                for (uint32_t kk = 0; kk < 4u; ++kk) {
                    const uint32_t ko = k00 + kk * 8u;
                    A[n][kk][0] = tile_x[r0 + ko + t];
                    A[n][kk][1] = tile_x[r8 + ko + t];
                    A[n][kk][2] = tile_x[r0 + ko + 4u + t];
                    A[n][kk][3] = tile_x[r8 + ko + 4u + t];
                    dA[n][0][kk] = ((const float*)tile_x)[r0 + 64u + (k00 >> 3) + kk];
                    dA[n][1][kk] = ((const float*)tile_x)[r8 + 64u + (k00 >> 3) + kk];
                }
            }
#pragma unroll
            for (uint32_t j0 = 0; j0 < BN; j0 += 16u) {
                const uint32_t jc = j0 + joff;
#pragma unroll
                for (uint32_t kk = 0; kk < 4u; ++kk) {
                    const uint32_t ko = kk * 8u;
                    const int b0 = tile_y[(jc + g) * PD_MMQ_YK + 4u + ko + t];
                    const int b1 = tile_y[(jc + g) * PD_MMQ_YK + 4u + ko + 4u + t];
                    const float dB0 = ((const float*)tile_y)[(jc + 2u * t) * PD_MMQ_YK + kk];
                    const float dB1 = ((const float*)tile_y)[(jc + 2u * t + 1u) * PD_MMQ_YK + kk];
#pragma unroll
                    for (uint32_t n = 0; n < 2u; ++n) {
                        int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                        asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                            : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                            : "r"(A[n][kk][0]), "r"(A[n][kk][1]), "r"(A[n][kk][2]),
                              "r"(A[n][kk][3]), "r"(b0), "r"(b1));
                        float* a = acc[(j0 >> 3) + n];
                        a[0] = __fmaf_rn(__fmul_rn(dA[n][0][kk], dB0), (float)d0, a[0]);
                        a[1] = __fmaf_rn(__fmul_rn(dA[n][0][kk], dB1), (float)d1, a[1]);
                        a[2] = __fmaf_rn(__fmul_rn(dA[n][1][kk], dB0), (float)d2, a[2]);
                        a[3] = __fmaf_rn(__fmul_rn(dA[n][1][kk], dB1), (float)d3, a[3]);
                    }
                }
            }
            __syncthreads();   // tile_y is reloaded next half / next kt
        }
    }

#pragma unroll
    for (uint32_t j0 = 0; j0 < BN; j0 += 16u) {
        const uint32_t c0 = col_base + j0 + joff + 2u * t;
#pragma unroll
        for (uint32_t n = 0; n < 2u; ++n) {
            const uint32_t r0 = row_base + i0 + n * 16u + g;
            const uint32_t r8 = r0 + 8u;
            const float* a = acc[(j0 >> 3) + n];
            if (r0 < out_dim) {
                if (c0 < batch) y[(size_t)c0 * out_dim + r0] = a[0];
                if (c0 + 1u < batch) y[(size_t)(c0 + 1u) * out_dim + r0] = a[1];
            }
            if (r8 < out_dim) {
                if (c0 < batch) y[(size_t)c0 * out_dim + r8] = a[2];
                if (c0 + 1u < batch) y[(size_t)(c0 + 1u) * out_dim + r8] = a[3];
            }
        }
    }
#else
    (void)data; (void)scale; (void)yq; (void)y; (void)in_dim; (void)out_dim; (void)batch;
#endif
}

template <uint32_t BM, uint32_t BN>
static int pd_q8_0_gemm_mmq_s_launch(const void* data, const void* scale, const void* yq,
                                     void* y, uint32_t in_dim, uint32_t out_dim,
                                     uint32_t batch, cudaStream_t st) {
    constexpr uint32_t smem = (BN * PD_MMQ_YK + BM * PD_MMQ_XK) * 4u;
    static bool set = false;
    if (!set) {
        cudaFuncSetAttribute(pd_q8_0_gemm_mmq_s_kernel<BM, BN>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        set = true;
    }
    dim3 grid((batch + BN - 1u) / BN, (out_dim + BM - 1u) / BM);
    pd_q8_0_gemm_mmq_s_kernel<BM, BN><<<grid, BM * 2u, smem, st>>>(
        (const int8_t*)data, (const __half*)scale, (const uint8_t*)yq, (float*)y, in_dim,
        out_dim, batch);
    return pd_launch_status();
}

// The election (GB10, mmq_small_bench over EmbeddingGemma 2's shapes at 64 to
// 2116 rows, every arm bit-equal): 64 x 128 once its grid holds two blocks an
// SM (one when K >= 1024, where a block's work is long enough to stand
// alone), else 32 x 64 at two blocks an SM, else 32 x 32. Against the 128 x
// 128 tile, the six shapes summed: 64 rows -53%, 144 -41%, 270 -32% (the
// 512-wide outputs -50%), 576 -21%, 982 -8%, 2116 -11%; every shape within
// 16% of its best arm. Around 1K rows the 512-wide outputs are a wash with
// the 128 x 128 tile (down at 982 rows +5%, the one shape behind it).
PD_EXPORT
int pd_q8_0_gemm_mmq_s(const void* data, const void* scale, const void* yq, void* y,
                       uint32_t in_dim, uint32_t out_dim, uint32_t batch, void* stream) {
    if (out_dim == 0 || batch == 0) return 0;
    if (in_dim & 31u) return cudaErrorInvalidValue;
    static int nsm = 0;
    if (nsm == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
        if (nsm <= 0) nsm = 1;
    }
    const uint32_t sm = (uint32_t)nsm;
    auto grid = [&](uint32_t bm, uint32_t bn) {
        return ((out_dim + bm - 1u) / bm) * ((batch + bn - 1u) / bn);
    };
    cudaStream_t st = (cudaStream_t)stream;
    const uint32_t g64 = grid(64u, 128u);
    if (g64 >= 2u * sm || (g64 >= sm && in_dim >= 1024u))
        return pd_q8_0_gemm_mmq_s_launch<64u, 128u>(data, scale, yq, y, in_dim, out_dim, batch, st);
    if (grid(32u, 64u) >= 2u * sm)
        return pd_q8_0_gemm_mmq_s_launch<32u, 64u>(data, scale, yq, y, in_dim, out_dim, batch, st);
    return pd_q8_0_gemm_mmq_s_launch<32u, 32u>(data, scale, yq, y, in_dim, out_dim, batch, st);
}
