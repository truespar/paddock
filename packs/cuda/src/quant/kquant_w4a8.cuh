// quant/kquant_w4a8.cuh (formerly 19_kquant_w4a8.cuh) - stage-2 W4A8 for k-quant weights: int8 tensor-core GEMM
// straight off the 4-6.6 bpw repacked streams (quant/kquant.cuh's layouts). Weights
// never materialize at 8/32-bit in DRAM - nibbles unpack to centered s8 in the
// tile-staging phase; activations ride the mmq int8 layout (pd_quantize_q8_mmq,
// same numeric class as llama's prefill). This replaces the stage-1 interim
// (kquant_dequant_rp + pd_gemm_f32) for batch/prefill; the exact-f32 decode
// GEMV is untouched. QServe-class design point (W4A8-int8 on the existing int8
// ladder - quantization strategy); skeleton cloned from gemm/mmq.cuh's
// pd_q8_0_gemm_mmq_kernel (same 128x128x256K tile, warp shape, fragment maps),
// stream-k/fixup deliberately dropped in v1 (prefill tile counts >> #SMs).
//
// Math per 32-weight sub-block g (activation x ~= dB_g * xq, xq int8):
//   Q4_K/Q5_K: w = d*sc*q - dmin*m, q unsigned 0..15/31. Center q' = q - C
//     (C = 8/16) to fit s8: w = dj*q' + mu, dj = d*sc, mu = C*dj - dmin*m.
//     y += dj * dB * sum(q' * xq)  +  mu * (dB * sum(xq))
//     The second term rides per-block ACTIVATION SUMS S = dB*sum(xq)
//     (pd_mmq_sums below) - the q8_1 trick, weights never see it.
//   Q6_K: w = d*sc16*(q-32), q-32 already s8, but scales are PER-16: the k32
//     mma can't scale halves apart, so Q6 runs two m16n8k16 mmas per 32-block
//     with per-16 f32 scales. No mu.
//   IQ4_XS: w = d*(ls-32)*LUT[q]; LUT values fit s8 -> pure Q8-shaped arm
//     with codebook unpack at staging. No mu.
// Numeric class: int8-quantized activations (rn + clamp, amax/127 per 32) -
// not the exact-f32 class of the stage-1 paths. Per-block int dots are exact
// int32; all scale application is f32.

#define PD_KW_XK 84u  // tile_x row stride, int32: 64 payload + 16 scale f32 + 4 pad
// tile_y reuses the mmq col stride (36); tile_s = 4 f32 sums per col
#define PD_KW_SMEM ((128u * PD_MMQ_YK + 128u * PD_KW_XK + 128u * 4u) * 4u)

// ---- per-32-block activation sums off the mmq layout --------------------------
// S[chunk][col][b] = scl_b * sum(int8 block b) - the min-term operand for
// Q4_K/Q5_K. Reads the ALREADY-quantized yq so no existing quantize kernel
// (incl. the fused add-norm/swiglu variants) needs to change; the extra yq
// read is noise next to the GEMM. Pad columns quantized zero -> S = 0.
__global__ void pd_mmq_sums_kernel(const uint8_t* __restrict__ yq,
                                   float* __restrict__ sums) {
    const size_t blk = (size_t)blockIdx.x * gridDim.y + blockIdx.y;
    const uint32_t lane = threadIdx.x;
    const int w = ((const int*)(yq + blk * 144u + 16u))[lane];
    int s = __dp4a(w, 0x01010101, 0);  // sum of 4 SIGNED bytes
    s += __shfl_xor_sync(0xffffffffu, s, 1);
    s += __shfl_xor_sync(0xffffffffu, s, 2);
    s += __shfl_xor_sync(0xffffffffu, s, 4);
    if ((lane & 7u) == 0u) {
        const float scl = ((const float*)(yq + blk * 144u))[lane >> 3u];
        sums[blk * 4u + (lane >> 3u)] = scl * (float)s;
    }
}

PD_EXPORT
int pd_mmq_sums(const void* yq, void* sums, uint32_t in_dim, uint32_t batch,
                void* stream) {
    if (in_dim == 0 || batch == 0) return 0;
    const uint32_t n_chunks = (in_dim + 127u) / 128u;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    dim3 grid(n_chunks, batch_pad);
    pd_mmq_sums_kernel<<<grid, 32, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)yq, (float*)sums);
    return pd_launch_status();
}

// ---- the W4A8 GEMM ------------------------------------------------------------
// One 128-row x 128-col tile per block, K walked one super-block (256 int8)
// per iteration - the repacked stream's natural granularity. Warp/fragment
// geometry identical to pd_q8_0_gemm_mmq_kernel; tile order column-fastest
// so concurrent blocks share the weight strip through L2.
template <uint32_t DT>
__global__ void __launch_bounds__(256, 1) pd_kquant_w4a8_kernel(
        const uint8_t* __restrict__ data, const uint8_t* __restrict__ scales,
        const uint8_t* __restrict__ yq, const float* __restrict__ xsums,
        float* __restrict__ y, uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
#if PD_MMA_OK
    constexpr bool MU = (DT == PD_KQ_Q4K || DT == PD_KQ_Q5K || DT == PD_KQ_Q40);
    constexpr bool K16 = (DT == PD_KQ_Q6K);
    constexpr uint32_t DATAB = DT == PD_KQ_Q6K ? PD_KQ6_DATA
                             : DT == PD_KQ_Q5K ? PD_KQ5_DATA : PD_KQ4_DATA;
    extern __shared__ int pd_kw_sh[];
    int* tile_y = pd_kw_sh;                              // 128 cols x 36 int32
    int* tile_x = pd_kw_sh + 128 * PD_MMQ_YK;            // 128 rows x 84 int32
    float* tile_s = (float*)(pd_kw_sh + 128 * PD_MMQ_YK + 128 * PD_KW_XK);

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5u;
    const uint32_t g = lane >> 2u, t = lane & 3u;
    const uint32_t i0 = (warp >> 1u) * 32u;   // warp pair's 32-row strip
    const uint32_t joff = (warp & 1u) * 8u;   // which 8-col group of each 16
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t n_super = in_dim >> 8u;
    const uint32_t nct = batch_pad >> 7u;
    const uint32_t tile = blockIdx.x;
    const uint32_t row_base = (tile / nct) * 128u;
    const uint32_t col_base = (tile % nct) * 128u;

    // IQ4 codebook as s8 in shared (divergent __constant__ reads serialize)
    __shared__ int s_lut[16];
    if (DT == PD_KQ_IQ4XS) {
        if (tid < 16u) s_lut[tid] = (int)PD_KQ_IQ4NL[tid];
        __syncthreads();
    }

    float acc[16][4] = {};
    for (uint32_t kt = 0; kt < n_super; ++kt) {
        // ---- weight tile: 128 rows x 1 super, unpacked to centered s8 ----
        // 1024 sixteen-byte tasks over 256 threads; every task emits two
        // 4-int32 runs (lo/hi nibble streams land at fixed strides).
        #pragma unroll
        for (uint32_t it = 0; it < 4u; ++it) {
            const uint32_t i = it * 256u + tid;
            const uint32_t row = i >> 3u, ci = i & 7u;
            const bool live = (row_base + row) < out_dim;
            const uint8_t* sb =
                data + ((size_t)(row_base + row) * n_super + kt) * DATAB;
            int out[8] = {};
            uint32_t obase, hioff;
            if (DT == PD_KQ_Q6K) {
                const uint32_t n = ci >> 2u, a = (ci >> 1u) & 1u, h = ci & 1u;
                obase = n * 32u + a * 8u + h * 4u;  // k/4 of the lo run
                hioff = 16u;                        // hi run: k + 64
                if (live) {
                    const uint4 qv = *(const uint4*)(sb + n * 64u + a * 32u + h * 16u);
                    const uint4 hv = *(const uint4*)(sb + 128u + n * 32u + h * 16u);
                    const uint32_t qw[4] = {qv.x, qv.y, qv.z, qv.w};
                    const uint32_t hw[4] = {hv.x, hv.y, hv.z, hv.w};
                    const uint32_t sh1 = 2u * a, sh2 = 2u * a + 4u;
                    #pragma unroll
                    for (uint32_t wv = 0; wv < 4u; ++wv) {
                        const uint32_t lo = (qw[wv] & 0x0F0F0F0Fu)
                            | (((hw[wv] >> sh1) & 0x03030303u) << 4u);
                        const uint32_t hi = ((qw[wv] >> 4u) & 0x0F0F0F0Fu)
                            | (((hw[wv] >> sh2) & 0x03030303u) << 4u);
                        out[wv] = (int)__vsub4(lo, 0x20202020u);       // q - 32
                        out[4u + wv] = (int)__vsub4(hi, 0x20202020u);
                    }
                }
            } else if (DT == PD_KQ_IQ4XS) {
                obase = ci * 8u;  // sub-block ib = ci: k = ib*32
                hioff = 4u;       // hi run: k + 16
                if (live) {
                    const uint4 qv = *(const uint4*)(sb + ci * 16u);
                    const uint32_t qw[4] = {qv.x, qv.y, qv.z, qv.w};
                    #pragma unroll
                    for (uint32_t wv = 0; wv < 4u; ++wv) {
                        uint32_t lo = 0u, hi = 0u;
                        #pragma unroll
                        for (uint32_t b = 0; b < 4u; ++b) {
                            const uint32_t qb = (qw[wv] >> (8u * b)) & 0xFFu;
                            lo |= ((uint32_t)(uint8_t)s_lut[qb & 0xFu]) << (8u * b);
                            hi |= ((uint32_t)(uint8_t)s_lut[qb >> 4u]) << (8u * b);
                        }
                        out[wv] = (int)lo;
                        out[4u + wv] = (int)hi;
                    }
                }
            } else {  // Q4_K / Q5_K
                const uint32_t gq = ci >> 1u, h = ci & 1u;
                obase = gq * 16u + h * 4u;  // k = g*64 + h*16
                hioff = 8u;                 // hi run: k + 32
                if (live) {
                    const uint4 qv = *(const uint4*)(sb + gq * 32u + h * 16u);
                    uint4 hv = make_uint4(0u, 0u, 0u, 0u);
                    if (DT == PD_KQ_Q5K) hv = *(const uint4*)(sb + 128u + h * 16u);
                    const uint32_t qw[4] = {qv.x, qv.y, qv.z, qv.w};
                    const uint32_t hw[4] = {hv.x, hv.y, hv.z, hv.w};
                    const uint32_t j1 = 2u * gq, j2 = 2u * gq + 1u;
                    const uint32_t C = DT == PD_KQ_Q5K ? 0x10101010u : 0x08080808u;
                    #pragma unroll
                    for (uint32_t wv = 0; wv < 4u; ++wv) {
                        uint32_t lo = qw[wv] & 0x0F0F0F0Fu;
                        uint32_t hi = (qw[wv] >> 4u) & 0x0F0F0F0Fu;
                        if (DT == PD_KQ_Q5K) {
                            lo |= ((hw[wv] >> j1) & 0x01010101u) << 4u;
                            hi |= ((hw[wv] >> j2) & 0x01010101u) << 4u;
                        }
                        out[wv] = (int)__vsub4(lo, C);  // q - 8 / q - 16
                        out[4u + wv] = (int)__vsub4(hi, C);
                    }
                }
            }
            int* dst = tile_x + row * PD_KW_XK + obase;
            #pragma unroll
            for (uint32_t wv = 0; wv < 4u; ++wv) {
                dst[wv] = out[wv];
                dst[hioff + wv] = out[4u + wv];
            }
        }
        // ---- row scales: one thread per row off the 24 B record ----
        if (tid < 128u) {
            const uint32_t row = tid;
            float* sc = (float*)(tile_x + row * PD_KW_XK + 64u);
            const bool live = (row_base + row) < out_dim;
            const uint8_t* rec =
                scales + ((size_t)(row_base + row) * n_super + kt) * PD_KQ_SCB;
            if (DT == PD_KQ_Q6K) {
                float d = 0.0f;
                if (live) {
                    __half hd;
                    memcpy(&hd, rec, 2u);
                    d = __half2float(hd);
                }
                #pragma unroll
                for (uint32_t j = 0; j < 16u; ++j)
                    sc[j] = live ? d * (float)((const int8_t*)rec)[4u + j] : 0.0f;
            } else if (DT == PD_KQ_IQ4XS) {
                float d = 0.0f;
                if (live) {
                    __half hd;
                    memcpy(&hd, rec, 2u);
                    d = __half2float(hd);
                }
                #pragma unroll
                for (uint32_t j = 0; j < 8u; ++j)
                    sc[j] = live ? d * (float)((const int8_t*)rec)[4u + j] : 0.0f;
            } else {
                float d = 0.0f, dmin = 0.0f;
                if (live) {
                    __half hd, hm;
                    memcpy(&hd, rec, 2u);
                    memcpy(&hm, rec + 2u, 2u);
                    d = __half2float(hd);
                    dmin = __half2float(hm);
                }
                const float C = DT == PD_KQ_Q5K ? 16.0f : 8.0f;
                #pragma unroll
                for (uint32_t j = 0; j < 8u; ++j) {
                    const float dj = live ? (DT == PD_KQ_Q40 ? pd_kq40_dj(rec, j)
                                                             : d * (float)rec[4u + j])
                                          : 0.0f;
                    sc[j] = dj;                                            // dj
                    sc[8u + j] = (live && DT != PD_KQ_Q40)                 // mu
                                     ? C * dj - dmin * (float)rec[12u + j]
                                     : 0.0f;
                }
            }
        }

        #pragma unroll
        for (uint32_t h = 0; h < 2u; ++h) {
            // stage one 128-int8 activation chunk (flat contiguous copy) +
            // its block sums. in_dim % 256 == 0 (launcher) -> no chunk guard.
            const uint32_t chunk = kt * 2u + h;
            const int* by = (const int*)(yq + ((size_t)chunk * batch_pad + col_base) * 144u);
            #pragma unroll
            for (uint32_t it = 0; it < 18u; ++it) {  // 128*36 == 18*256 exactly
                const uint32_t l = it * 256u + tid;
                tile_y[l] = by[l];
            }
            if (MU) {
                #pragma unroll
                for (uint32_t it = 0; it < 2u; ++it) {
                    const uint32_t l = it * 256u + tid;  // col*4 + block
                    tile_s[l] = xsums[((size_t)chunk * batch_pad + col_base) * 4u + l];
                }
            }
            __syncthreads();  // h==0 also covers the tile_x stores above

            const uint32_t k00 = h * 32u;  // x-side int32 offset of this chunk
            // preload this warp's A fragments + weight scales for the chunk
            int A[2][4][4];
            float dA[2][2][4];   // per-32 dj (k32 formats)
            float muA[2][2][4];  // per-32 mu (MU only)
            float sA[2][2][8];   // per-16 d*sc (K16 only)
            #pragma unroll
            for (uint32_t n = 0; n < 2u; ++n) {
                const uint32_t r0 = (i0 + n * 16u + g) * PD_KW_XK;
                const uint32_t r8 = (i0 + n * 16u + 8u + g) * PD_KW_XK;
                #pragma unroll
                for (uint32_t kk = 0; kk < 4u; ++kk) {
                    const uint32_t ko = k00 + kk * 8u;
                    A[n][kk][0] = tile_x[r0 + ko + t];
                    A[n][kk][1] = tile_x[r8 + ko + t];
                    A[n][kk][2] = tile_x[r0 + ko + 4u + t];
                    A[n][kk][3] = tile_x[r8 + ko + 4u + t];
                    if (K16) {
                        const uint32_t so = 64u + (k00 >> 2u) + 2u * kk;
                        sA[n][0][2u * kk] = ((const float*)tile_x)[r0 + so];
                        sA[n][0][2u * kk + 1u] = ((const float*)tile_x)[r0 + so + 1u];
                        sA[n][1][2u * kk] = ((const float*)tile_x)[r8 + so];
                        sA[n][1][2u * kk + 1u] = ((const float*)tile_x)[r8 + so + 1u];
                    } else {
                        dA[n][0][kk] = ((const float*)tile_x)[r0 + 64u + (k00 >> 3u) + kk];
                        dA[n][1][kk] = ((const float*)tile_x)[r8 + 64u + (k00 >> 3u) + kk];
                        if (MU) {
                            muA[n][0][kk] = ((const float*)tile_x)[r0 + 72u + (k00 >> 3u) + kk];
                            muA[n][1][kk] = ((const float*)tile_x)[r8 + 72u + (k00 >> 3u) + kk];
                        }
                    }
                }
            }
            #pragma unroll
            for (uint32_t j0 = 0; j0 < 128u; j0 += 16u) {
                const uint32_t jc = j0 + joff;
                #pragma unroll
                for (uint32_t kk = 0; kk < 4u; ++kk) {
                    const uint32_t ko = kk * 8u;
                    const int b0 = tile_y[(jc + g) * PD_MMQ_YK + 4u + ko + t];
                    const int b1 = tile_y[(jc + g) * PD_MMQ_YK + 4u + ko + 4u + t];
                    const float dB0 = ((const float*)tile_y)[(jc + 2u * t) * PD_MMQ_YK + kk];
                    const float dB1 = ((const float*)tile_y)[(jc + 2u * t + 1u) * PD_MMQ_YK + kk];
                    float S0 = 0.0f, S1 = 0.0f;
                    if (MU) {
                        S0 = tile_s[(jc + 2u * t) * 4u + kk];
                        S1 = tile_s[(jc + 2u * t + 1u) * 4u + kk];
                    }
                    #pragma unroll
                    for (uint32_t n = 0; n < 2u; ++n) {
                        if (K16) {
                            // per-16 scales: two k16 mmas, halves scaled apart
                            int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                            int e0 = 0, e1 = 0, e2 = 0, e3 = 0;
                            asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
                                "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                                : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                                : "r"(A[n][kk][0]), "r"(A[n][kk][1]), "r"(b0));
                            asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
                                "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                                : "+r"(e0), "+r"(e1), "+r"(e2), "+r"(e3)
                                : "r"(A[n][kk][2]), "r"(A[n][kk][3]), "r"(b1));
                            const float sa0 = sA[n][0][2u * kk], sa1 = sA[n][0][2u * kk + 1u];
                            const float sb0_ = sA[n][1][2u * kk], sb1_ = sA[n][1][2u * kk + 1u];
                            acc[(j0 >> 3) + n][0] += dB0 * (sa0 * (float)d0 + sa1 * (float)e0);
                            acc[(j0 >> 3) + n][1] += dB1 * (sa0 * (float)d1 + sa1 * (float)e1);
                            acc[(j0 >> 3) + n][2] += dB0 * (sb0_ * (float)d2 + sb1_ * (float)e2);
                            acc[(j0 >> 3) + n][3] += dB1 * (sb0_ * (float)d3 + sb1_ * (float)e3);
                        } else {
                            int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                            asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                                : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                                : "r"(A[n][kk][0]), "r"(A[n][kk][1]), "r"(A[n][kk][2]),
                                  "r"(A[n][kk][3]), "r"(b0), "r"(b1));
                            acc[(j0 >> 3) + n][0] += dA[n][0][kk] * dB0 * (float)d0;
                            acc[(j0 >> 3) + n][1] += dA[n][0][kk] * dB1 * (float)d1;
                            acc[(j0 >> 3) + n][2] += dA[n][1][kk] * dB0 * (float)d2;
                            acc[(j0 >> 3) + n][3] += dA[n][1][kk] * dB1 * (float)d3;
                            if (MU) {
                                acc[(j0 >> 3) + n][0] += muA[n][0][kk] * S0;
                                acc[(j0 >> 3) + n][1] += muA[n][0][kk] * S1;
                                acc[(j0 >> 3) + n][2] += muA[n][1][kk] * S0;
                                acc[(j0 >> 3) + n][3] += muA[n][1][kk] * S1;
                            }
                        }
                    }
                }
            }
            __syncthreads();  // tile_y/tile_s reload next half / next kt
        }
    }

    // store (row, tok) -> y[tok*out_dim + row] (mmq's map, no bias here)
    #pragma unroll
    for (uint32_t j0 = 0; j0 < 128u; j0 += 16u) {
        const uint32_t c0 = col_base + j0 + joff + 2u * t;
        #pragma unroll
        for (uint32_t n = 0; n < 2u; ++n) {
            const uint32_t r0 = row_base + i0 + n * 16u + g;
            const uint32_t r8 = r0 + 8u;
            if (r0 < out_dim) {
                if (c0 < batch) y[(size_t)c0 * out_dim + r0] = acc[(j0 >> 3) + n][0];
                if (c0 + 1u < batch) y[(size_t)(c0 + 1u) * out_dim + r0] = acc[(j0 >> 3) + n][1];
            }
            if (r8 < out_dim) {
                if (c0 < batch) y[(size_t)c0 * out_dim + r8] = acc[(j0 >> 3) + n][2];
                if (c0 + 1u < batch) y[(size_t)(c0 + 1u) * out_dim + r8] = acc[(j0 >> 3) + n][3];
            }
        }
    }
#else
    (void)data; (void)scales; (void)yq; (void)xsums; (void)y;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}

// ---- strided per-16 activation sums (dp4a decode ladder, Q4/Q5 mu term) ------
// sums[col][w16] = sum of 16 SIGNED int8 - raw int as f32; the consumer folds
// the per-32 activation scale in its epilogue. Flat over batch*in/16 groups
// (the strided pd_quantize_q8 layout is row-contiguous).
__global__ void pd_q8_sums_strided_kernel(const int8_t* __restrict__ xq,
                                          float* __restrict__ sums, uint32_t n16) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n16) return;
    const int* w = (const int*)xq + i * 4u;
    int s = __dp4a(w[0], 0x01010101, 0);
    s = __dp4a(w[1], 0x01010101, s);
    s = __dp4a(w[2], 0x01010101, s);
    s = __dp4a(w[3], 0x01010101, s);
    sums[i] = (float)s;
}

PD_EXPORT
int pd_q8_sums_strided(const void* xq, void* sums, uint32_t in_dim, uint32_t batch,
                       void* stream) {
    if (in_dim == 0 || batch == 0) return 0;
    if ((in_dim & 15u) != 0u) return cudaErrorInvalidValue;
    const uint32_t n16 = batch * (in_dim >> 4u);
    pd_q8_sums_strided_kernel<<<(n16 + 255u) / 256u, 256, 0, (cudaStream_t)stream>>>(
        (const int8_t*)xq, (float*)sums, n16);
    return pd_launch_status();
}

// ---- W4A8 dp4a batch GEMM (decode-batch shape: few columns, weight-bound) ----
// The 128x128 mma tile above is the wrong shape at decode batches (b <= 32) -
// out/128 tiles idle most of the die - and two earlier shapes measured far off
// the weight-bandwidth floor here: the GEMV-derived 4-rows/block
// walk was L2-transaction-bound on its per-chunk 16-byte x windows (~110 GB/s
// weight-effective at b=8), and a shared-staged variant of it collapsed to
// 1 block/SM latency-bound (~50). This is the PROVEN small-batch geometry
// instead - pd_q8_0_gemm_mt_dp4a's tile (16 output rows/block, 2 per warp,
// 512-elem staged x chunks, ~11 KB smem) with the k-quant nibble unpack in
// the lane's window load. Each lane owns a 16-weight K-window, which is
// exactly one k-quant sub-block half: one (scale, mu) pair per window, and
// Q6_K's per-16 scales are native (no mma split needed).
// Activations: STRIDED int8 (pd_quantize_q8 layout - the same buffers the Q8
// bmm ladder quantizes) + per-16 sums (pd_q8_sums_strided) for the Q4/Q5 mu
// term. Numeric class: identical to the Q8 batched ladder and the W4A8
// prefill (exact int dots, f32 per-block scale application). Per-row math is
// batch-size-independent (chunk order, lane map, reduce tree fixed), so a
// row's output is bit-identical across z-tiles and batch widths - the spec
// gates' cross-proof property.
#define PD_KDP_ROWS 16u   // batch rows per weight pass (z-tiled)
#define PD_KDP_CHUNK 512u // x elems staged per iteration (2 super-blocks)
__global__ void __launch_bounds__(256) pd_kquant_gemm_dp4a_kernel(
        const uint8_t* __restrict__ data, const uint8_t* __restrict__ scales,
        const int8_t* __restrict__ xq, const float* __restrict__ xs,
        const float* __restrict__ xsums, float* __restrict__ y,
        uint32_t in_dim, uint32_t out_dim, uint32_t batch, uint32_t dtype) {
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5u;
    const uint32_t o0 = blockIdx.x * 16u + warp * 2u;  // two output rows per warp
    const uint32_t nb32 = in_dim >> 5u, nb16 = in_dim >> 4u;
    const uint32_t n_super = in_dim >> 8u;
    const uint32_t datab = dtype == PD_KQ_Q6K ? PD_KQ6_DATA
                         : dtype == PD_KQ_Q5K ? PD_KQ5_DATA : PD_KQ4_DATA;
    {
        // z tiles the batch: each z-tile re-reads the weight once, so weight
        // traffic scales with ceil(B/16)
        const uint32_t b0 = blockIdx.z * PD_KDP_ROWS;
        xq += (size_t)b0 * in_dim;
        xs += (size_t)b0 * nb32;
        if (xsums) xsums += (size_t)b0 * nb16;
        y += (size_t)b0 * out_dim;
        batch = (batch - b0 < PD_KDP_ROWS) ? (batch - b0) : PD_KDP_ROWS;
    }
    __shared__ int4 xqs[PD_KDP_ROWS * (PD_KDP_CHUNK / 16u)];  // int8 chunk rows
    __shared__ float xss[PD_KDP_ROWS * (PD_KDP_CHUNK / 32u)]; // per-32 scales
    __shared__ float sss[PD_KDP_ROWS * (PD_KDP_CHUNK / 16u)]; // per-16 sums (mu)
    __shared__ int s_lut[16];
    if (dtype == PD_KQ_IQ4XS && tid < 16u) s_lut[tid] = (int)PD_KQ_IQ4NL[tid];
    const bool want_mu = pd_kq_has_mu(dtype);

    // compile-time-indexed accumulators (runtime indices would spill to local)
    float acc0[PD_KDP_ROWS], acc1[PD_KDP_ROWS];
    #pragma unroll
    for (uint32_t t = 0; t < PD_KDP_ROWS; ++t) { acc0[t] = 0.0f; acc1[t] = 0.0f; }

    const uint8_t* rowd0 = data + (size_t)o0 * n_super * datab;
    const uint8_t* rowd1 = data + (size_t)(o0 + 1u) * n_super * datab;
    const uint8_t* recs0 = scales + (size_t)o0 * n_super * PD_KQ_SCB;
    const uint8_t* recs1 = scales + (size_t)(o0 + 1u) * n_super * PD_KQ_SCB;

    for (uint32_t c0 = 0; c0 < in_dim; c0 += PD_KDP_CHUNK) {
        // stage the quantized x chunk + per-32 scales (+ per-16 sums when the
        // format's mu term needs them)
        for (uint32_t i = tid; i < batch * (PD_KDP_CHUNK / 16u); i += 256u) {
            const uint32_t t = i / (PD_KDP_CHUNK / 16u), k = i % (PD_KDP_CHUNK / 16u);
            const uint32_t src = c0 + k * 16u;
            xqs[i] = (src < in_dim)
                ? *(const int4*)(xq + (size_t)t * in_dim + src)
                : make_int4(0, 0, 0, 0);
        }
        for (uint32_t i = tid; i < batch * (PD_KDP_CHUNK / 32u); i += 256u) {
            const uint32_t t = i / (PD_KDP_CHUNK / 32u), b = i % (PD_KDP_CHUNK / 32u);
            const uint32_t blk = (c0 >> 5u) + b;
            xss[i] = (blk < nb32) ? xs[(size_t)t * nb32 + blk] : 0.0f;
        }
        if (want_mu) {
            for (uint32_t i = tid; i < batch * (PD_KDP_CHUNK / 16u); i += 256u) {
                const uint32_t t = i / (PD_KDP_CHUNK / 16u), w = i % (PD_KDP_CHUNK / 16u);
                const uint32_t blk = (c0 >> 4u) + w;
                sss[i] = (blk < nb16) ? xsums[(size_t)t * nb16 + blk] : 0.0f;
            }
        }
        __syncthreads();
        const uint32_t base = c0 + lane * 16u;
        if (o0 < out_dim && base < in_dim) {
            // lane's 16-weight window = one sub-block half: unpack both rows'
            // windows to packed s8 words + one (scale, mu) pair per row
            const uint32_t s = base >> 8u, w = (base >> 4u) & 15u;
            int w0[4], w1[4];
            float f0, g0 = 0.0f, f1, g1 = 0.0f;
            if (dtype == PD_KQ_Q6K) {
                const uint32_t n = w >> 3u, rw = (w >> 1u) & 3u, h = w & 1u;
                const uint32_t qoff = n * 64u + (rw & 1u) * 32u + h * 16u;
                const uint32_t hoff = 128u + n * 32u + h * 16u;
                const uint32_t sh = 2u * rw;
                const bool hi = rw >= 2u;
                const uint4 qa = __ldcs((const uint4*)(rowd0 + (size_t)s * datab + qoff));
                const uint4 ha = __ldcs((const uint4*)(rowd0 + (size_t)s * datab + hoff));
                const uint4 qb = __ldcs((const uint4*)(rowd1 + (size_t)s * datab + qoff));
                const uint4 hb = __ldcs((const uint4*)(rowd1 + (size_t)s * datab + hoff));
                const uint32_t qaw[4] = {qa.x, qa.y, qa.z, qa.w};
                const uint32_t haw[4] = {ha.x, ha.y, ha.z, ha.w};
                const uint32_t qbw[4] = {qb.x, qb.y, qb.z, qb.w};
                const uint32_t hbw[4] = {hb.x, hb.y, hb.z, hb.w};
                #pragma unroll
                for (uint32_t v = 0; v < 4u; ++v) {
                    const uint32_t na = hi ? (qaw[v] >> 4u) & 0x0F0F0F0Fu : qaw[v] & 0x0F0F0F0Fu;
                    const uint32_t nb = hi ? (qbw[v] >> 4u) & 0x0F0F0F0Fu : qbw[v] & 0x0F0F0F0Fu;
                    w0[v] = (int)__vsub4(na | (((haw[v] >> sh) & 0x03030303u) << 4u), 0x20202020u);
                    w1[v] = (int)__vsub4(nb | (((hbw[v] >> sh) & 0x03030303u) << 4u), 0x20202020u);
                }
                const uint32_t si = n * 8u + rw * 2u + h;
                const uint8_t* ra = recs0 + (size_t)s * PD_KQ_SCB;
                const uint8_t* rb = recs1 + (size_t)s * PD_KQ_SCB;
                __half hda, hdb;
                memcpy(&hda, ra, 2u);
                memcpy(&hdb, rb, 2u);
                f0 = __half2float(hda) * (float)((const int8_t*)ra)[4u + si];
                f1 = __half2float(hdb) * (float)((const int8_t*)rb)[4u + si];
            } else if (dtype == PD_KQ_IQ4XS) {
                const uint32_t ib = w >> 1u, h = w & 1u;
                const uint4 qa = __ldcs((const uint4*)(rowd0 + (size_t)s * datab + ib * 16u));
                const uint4 qb = __ldcs((const uint4*)(rowd1 + (size_t)s * datab + ib * 16u));
                const uint32_t qaw[4] = {qa.x, qa.y, qa.z, qa.w};
                const uint32_t qbw[4] = {qb.x, qb.y, qb.z, qb.w};
                #pragma unroll
                for (uint32_t v = 0; v < 4u; ++v) {
                    uint32_t la = 0u, lb = 0u;
                    #pragma unroll
                    for (uint32_t bb = 0; bb < 4u; ++bb) {
                        const uint32_t qba = (qaw[v] >> (8u * bb)) & 0xFFu;
                        const uint32_t qbb = (qbw[v] >> (8u * bb)) & 0xFFu;
                        const uint32_t nba = h ? (qba >> 4u) : (qba & 0xFu);
                        const uint32_t nbb = h ? (qbb >> 4u) : (qbb & 0xFu);
                        la |= ((uint32_t)s_lut[nba] & 0xFFu) << (8u * bb);
                        lb |= ((uint32_t)s_lut[nbb] & 0xFFu) << (8u * bb);
                    }
                    w0[v] = (int)la;
                    w1[v] = (int)lb;
                }
                const uint8_t* ra = recs0 + (size_t)s * PD_KQ_SCB;
                const uint8_t* rb = recs1 + (size_t)s * PD_KQ_SCB;
                __half hda, hdb;
                memcpy(&hda, ra, 2u);
                memcpy(&hdb, rb, 2u);
                f0 = __half2float(hda) * (float)((const int8_t*)ra)[4u + ib];
                f1 = __half2float(hdb) * (float)((const int8_t*)rb)[4u + ib];
            } else {  // Q4_K / Q5_K
                const bool q5 = dtype == PD_KQ_Q5K;
                const uint32_t j = w >> 1u, h = w & 1u;  // sub-block, 16-half
                const uint32_t qoff = (j >> 1u) * 32u + h * 16u;
                const bool hi = (j & 1u) != 0u;
                const uint32_t C = q5 ? 0x10101010u : 0x08080808u;
                const uint4 qa = __ldcs((const uint4*)(rowd0 + (size_t)s * datab + qoff));
                const uint4 qb = __ldcs((const uint4*)(rowd1 + (size_t)s * datab + qoff));
                uint4 ha = make_uint4(0u, 0u, 0u, 0u), hbv = ha;
                if (q5) {
                    ha = __ldcs((const uint4*)(rowd0 + (size_t)s * datab + 128u + h * 16u));
                    hbv = __ldcs((const uint4*)(rowd1 + (size_t)s * datab + 128u + h * 16u));
                }
                const uint32_t qaw[4] = {qa.x, qa.y, qa.z, qa.w};
                const uint32_t qbw[4] = {qb.x, qb.y, qb.z, qb.w};
                const uint32_t haw[4] = {ha.x, ha.y, ha.z, ha.w};
                const uint32_t hbw[4] = {hbv.x, hbv.y, hbv.z, hbv.w};
                #pragma unroll
                for (uint32_t v = 0; v < 4u; ++v) {
                    uint32_t na = hi ? (qaw[v] >> 4u) & 0x0F0F0F0Fu : qaw[v] & 0x0F0F0F0Fu;
                    uint32_t nb = hi ? (qbw[v] >> 4u) & 0x0F0F0F0Fu : qbw[v] & 0x0F0F0F0Fu;
                    if (q5) {
                        na |= ((haw[v] >> j) & 0x01010101u) << 4u;
                        nb |= ((hbw[v] >> j) & 0x01010101u) << 4u;
                    }
                    w0[v] = (int)__vsub4(na, C);
                    w1[v] = (int)__vsub4(nb, C);
                }
                const uint8_t* ra = recs0 + (size_t)s * PD_KQ_SCB;
                const uint8_t* rb = recs1 + (size_t)s * PD_KQ_SCB;
                const bool q40 = dtype == PD_KQ_Q40;
                __half hda, hma, hdb, hmb;
                memcpy(&hda, ra, 2u);
                memcpy(&hma, ra + 2u, 2u);
                memcpy(&hdb, rb, 2u);
                memcpy(&hmb, rb + 2u, 2u);
                const float da = __half2float(hda), ma = __half2float(hma);
                const float db_ = __half2float(hdb), mb = __half2float(hmb);
                const float cf = q5 ? 16.0f : 8.0f;
                f0 = q40 ? pd_kq40_dj(ra, j) : da * (float)ra[4u + j];
                g0 = q40 ? 0.0f : cf * f0 - ma * (float)ra[12u + j];
                f1 = q40 ? pd_kq40_dj(rb, j) : db_ * (float)rb[4u + j];
                g1 = q40 ? 0.0f : cf * f1 - mb * (float)rb[12u + j];
            }
            const bool live1 = (o0 + 1u) < out_dim;
            #pragma unroll
            for (uint32_t t = 0; t < PD_KDP_ROWS; ++t) {
                if (t >= batch) break;
                const int4 xv = xqs[t * (PD_KDP_CHUNK / 16u) + lane];
                int s0 = __dp4a(w0[0], xv.x, 0);
                s0 = __dp4a(w0[1], xv.y, s0);
                s0 = __dp4a(w0[2], xv.z, s0);
                s0 = __dp4a(w0[3], xv.w, s0);
                int s1 = __dp4a(w1[0], xv.x, 0);
                s1 = __dp4a(w1[1], xv.y, s1);
                s1 = __dp4a(w1[2], xv.z, s1);
                s1 = __dp4a(w1[3], xv.w, s1);
                const float xsc = xss[t * (PD_KDP_CHUNK / 32u) + (lane >> 1u)];
                acc0[t] += f0 * (xsc * (float)s0);
                if (live1) acc1[t] += f1 * (xsc * (float)s1);
                if (want_mu) {
                    const float sw = sss[t * (PD_KDP_CHUNK / 16u) + lane];
                    acc0[t] += g0 * (xsc * sw);
                    if (live1) acc1[t] += g1 * (xsc * sw);
                }
            }
        }
        __syncthreads();
    }
    if (o0 >= out_dim) return;
    #pragma unroll
    for (uint32_t t = 0; t < PD_KDP_ROWS; ++t) {
        if (t >= batch) break;
        float a0 = acc0[t], a1 = acc1[t];
        for (uint32_t sd = 16; sd > 0; sd >>= 1) {
            a0 += __shfl_down_sync(0xffffffffu, a0, sd);
            a1 += __shfl_down_sync(0xffffffffu, a1, sd);
        }
        if (lane == 0) {
            y[(size_t)t * out_dim + o0] = a0;
            if (o0 + 1u < out_dim) y[(size_t)t * out_dim + o0 + 1u] = a1;
        }
    }
}

PD_EXPORT
int pd_kquant_gemm_dp4a(const void* data, const void* scales, const void* xq,
                        const void* xs, const void* xsums, void* y, uint32_t in_dim,
                        uint32_t out_dim, uint32_t batch, uint32_t dtype,
                        void* stream) {
    if (out_dim == 0 || batch == 0) return 0;
    if (pd_kq_valid_iq(dtype))
        return pd_kq_iq_dense_dp4a(data, scales, xq, xs, xsums, y, in_dim, out_dim, batch, dtype, stream);
    if ((in_dim & 255u) != 0u) return cudaErrorInvalidValue;
    if (!pd_kq_valid(dtype)) return cudaErrorInvalidValue;
    if ((pd_kq_has_mu(dtype)) && xsums == nullptr)
        return cudaErrorInvalidValue;
    dim3 grid((out_dim + 15u) / 16u, 1u, (batch + PD_KDP_ROWS - 1u) / PD_KDP_ROWS);
    pd_kquant_gemm_dp4a_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)data, (const uint8_t*)scales, (const int8_t*)xq,
        (const float*)xs, (const float*)xsums, (float*)y, in_dim, out_dim, batch,
        dtype);
    return pd_launch_status();
}

PD_EXPORT
int pd_kquant_gemm_w4a8_pipe2(const void* data, const void* scales, const void* yq,
                              const void* xsums, void* y, uint32_t in_dim,
                              uint32_t out_dim, uint32_t batch, uint32_t dtype,
                              void* stream);

PD_EXPORT
int pd_kquant_gemm_w4a8(const void* data, const void* scales, const void* yq,
                        const void* xsums, void* y, uint32_t in_dim,
                        uint32_t out_dim, uint32_t batch, uint32_t dtype,
                        void* stream) {
    if (out_dim == 0 || batch == 0) return 0;
    // the i-quant family has one tile rung, pipe2 (its raw ring is sized per
    // type and its tile builds through the shared window unpack)
    if (pd_kq_valid_iq(dtype))
        return pd_kquant_gemm_w4a8_pipe2(data, scales, yq, xsums, y, in_dim, out_dim,
                                         batch, dtype, stream);
    if ((in_dim & 255u) != 0u) return cudaErrorInvalidValue;
    if (!pd_kq_valid(dtype)) return cudaErrorInvalidValue;
    const bool mu = pd_kq_has_mu(dtype);
    if (mu && xsums == nullptr) return cudaErrorInvalidValue;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t ntiles = ((out_dim + 127u) / 128u) * (batch_pad >> 7u);
    // 62 KB dynamic shared > the default 48 KB window: opt in per instantiation
    #define PD_KW_LAUNCH(DTV)                                                     \
        do {                                                                      \
            static cudaError_t attr = cudaFuncSetAttribute(                       \
                (const void*)pd_kquant_w4a8_kernel<DTV>,                          \
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)PD_KW_SMEM);    \
            if (attr != cudaSuccess) return attr;                                 \
            pd_kquant_w4a8_kernel<DTV><<<ntiles, 256, PD_KW_SMEM,                 \
                                         (cudaStream_t)stream>>>(                 \
                (const uint8_t*)data, (const uint8_t*)scales,                     \
                (const uint8_t*)yq, (const float*)xsums, (float*)y,               \
                in_dim, out_dim, batch);                                          \
        } while (0)
    switch (dtype) {
        case PD_KQ_Q40: PD_KW_LAUNCH(PD_KQ_Q40); break;
        case PD_KQ_Q4K: PD_KW_LAUNCH(PD_KQ_Q4K); break;
        case PD_KQ_Q5K: PD_KW_LAUNCH(PD_KQ_Q5K); break;
        case PD_KQ_Q6K: PD_KW_LAUNCH(PD_KQ_Q6K); break;
        default: PD_KW_LAUNCH(PD_KQ_IQ4XS); break;
    }
    #undef PD_KW_LAUNCH
    return pd_launch_status();
}

// ---- K-split W4A8 mma GEMM (the 17..64 decode-batch rung) ---------------------
// The dp4a MT tile above z-tiles the batch at 16 rows per weight pass, so c32
// serving reads the whole weight set twice per step. The Q8 ladder's measured
// fix for this exact rung is the K-split mma (pd_q8_0_gemm_mma_ks: 64-row
// tiles, grid.z K-ranges writing unbiased partial planes, fixed-order combine
// - it took every qwen shape from B=24, and a 32-rows/pass wide dp4a variant
// REGRESSED instead, 600->522 @ B=32).
//
// v2 (pipelined): v1 staged synchronously and parked every warp on DRAM
// latency, the same profile Q8's rung showed pre-pipe (v1 ran 136-325 GB/s
// weight-effective vs Q8's double-buffered ks). Q8's fix (ST-deep cp.async
// ring) doesn't transplant directly - k-quant weights need an unpack, and
// ring-buffering the UNPACKED s8 tile (21.5 KB/stage) would cost the 2nd
// resident block. So the ring holds the RAW compressed strips (4-6.6 bpw is
// cheaper to stage than s8) and the mma loop unpacks nibbles INLINE at
// fragment-load time - Marlin's design point (compressed weights live in
// smem, dequant at consumption). The
// 24 B scale records ride the ring raw and expand per-consuming-thread in
// registers (redundant f32 muls, but identical values - f32 is
// deterministic). Activations + per-32 scales + per-16 sums cp.async
// straight into the ring in final form (already int8/f32).
//
// Numerics: identical expressions in identical K-fold order as v1
// (z-slice-major, super-ascending, kk 0..7) - same class as dp4a/w4a8
// (exact int dots, f32 scale application), per-element batch-width
// invariant, deterministic for a fixed grid.
#define PD_KM_BSTR 68u  // ring activation col stride, int32 (64 payload + 4 pad)

// 8-byte predicated cp.async: scale records are 24 B / 8-B-aligned, so 16 B
// copies would misalign on odd (row, super) indices. src-size 0 zero-fills.
__device__ __forceinline__ void pd_kq_cpa8p(void* smem, const void* gmem, bool ok) {
#if PD_MMA_OK
    const unsigned sm = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;" ::"r"(sm), "l"(gmem),
                 "r"(ok ? 8u : 0u));
#endif
}

// 4-byte predicated cp.async: the i-quant scale records (quant/iquant.cuh
// pd_iq_scb) are 4-24 B and only 4-B aligned. src-size 0 zero-fills.
__device__ __forceinline__ void pd_kq_cpa4p(void* smem, const void* gmem, bool ok) {
#if PD_MMA_OK
    const unsigned sm = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(sm), "l"(gmem),
                 "r"(ok ? 4u : 0u));
#endif
}

// IQ4_NL codebook via prmt, no shared-LUT traffic: the 16 s8 entries live in
// four packed constants; two byte_perms + a bit-3 blend map the 4 per-byte
// nibble indices of idx4 in ~a dozen ALU ops. byte_perm's msb-replicate mode
// garbles the lo lookup exactly where bit 3 selects the hi table, and the
// mask spread ((idx>>3)&0x01010101)*0xFF discards those bytes. Same values
// as PD_KQ_IQ4NL - the s_lut version measured LDS-bound in the mma loop
// (16 shared loads per fragment set held IQ4 at 1.02x while Q4K took 1.63x).
__device__ __forceinline__ int pd_kq_iq4_prmt(uint32_t idx4) {
    const uint32_t c0 = (idx4 | (idx4 >> 4u)) & 0x00FF00FFu;
    const uint32_t s = (c0 | (c0 >> 8u)) & 0xFFFFu;  // 4 indices -> 4 nibbles
    const uint32_t vlo = __byte_perm(0xBFAD9881u, 0xF6EADDCFu, s);
    const uint32_t vhi = __byte_perm(0x26190D01u, 0x71594535u, s & 0x7777u);
    const uint32_t m = ((idx4 >> 3u) & 0x01010101u) * 0xFFu;
    return (int)((vhi & m) | (vlo & ~m));
}

__host__ __device__ __forceinline__ uint32_t pd_kq_datab(uint32_t dt) {
    if (pd_kq_valid_iq(dt)) return pd_iq_datab(dt);
    return dt == PD_KQ_Q6K ? PD_KQ6_DATA
         : dt == PD_KQ_Q5K ? PD_KQ5_DATA : PD_KQ4_DATA;
}

// Bytes one weight ROW of in_dim occupies in the data / scale streams. The
// 256-block families need a whole number of superblocks per row. IQ4_NL is
// 32-weight blocks whose repacked form is 16 payload bytes + one f16 d, so a
// row is simply in_dim/32 blocks laid flat: superblock s of the row is still
// at s*128 / s*16, and a row that is not a whole number of superblocks
// (Qwen3.8-Flash-Next's expert down, 640 = 2.5) carries no padding at all -
// the walk over the row is bounded by in_dim, never by the superblock count.
__host__ __device__ __forceinline__ uint32_t pd_kq_row_datab(uint32_t dt, uint32_t in_dim) {
    if (dt == PD_KQ_IQ4NL_ID) return (in_dim >> 5) * 16u;
    return ((in_dim + 255u) >> 8) * pd_kq_datab(dt);
}
__host__ __device__ __forceinline__ uint32_t pd_kq_row_scb(uint32_t dt, uint32_t in_dim) {
    if (dt == PD_KQ_IQ4NL_ID) return (in_dim >> 5) * 2u;
    return ((in_dim + 255u) >> 8) * pd_kq_scb(dt);
}

// Unpack one 16-weight window (window w of super sb/rec) of a k-quant row to
// 4 packed s8 words + its (scale f, mu g) pair. Ported per-format from the
// dense dp4a kernel's lane windows; g is 0 for the mu-free formats.
// (Consumed by moe/kquant.cuh's token-batched pair.)
__device__ __forceinline__ void pd_kq_win_unpack(
        uint32_t dtype, const uint8_t* __restrict__ sb,
        const uint8_t* __restrict__ rec, uint32_t w, int wq[4], float* f,
        float* g) {
    *g = 0.0f;
    if (pd_kq_valid_iq(dtype)) {
        pd_iq_win_unpack(dtype, sb, rec, w, wq, f, g);
        return;
    }
    if (dtype == PD_KQ_Q6K) {
        const uint32_t n = w >> 3u, rw = (w >> 1u) & 3u, h = w & 1u;
        const uint32_t qoff = n * 64u + (rw & 1u) * 32u + h * 16u;
        const uint32_t hoff = 128u + n * 32u + h * 16u;
        const uint32_t sh = 2u * rw;
        const bool hi = rw >= 2u;
        const uint4 qa = __ldcs((const uint4*)(sb + qoff));
        const uint4 ha = __ldcs((const uint4*)(sb + hoff));
        const uint32_t qw[4] = {qa.x, qa.y, qa.z, qa.w};
        const uint32_t hw[4] = {ha.x, ha.y, ha.z, ha.w};
        #pragma unroll
        for (uint32_t v = 0; v < 4u; ++v) {
            const uint32_t nib = hi ? (qw[v] >> 4u) & 0x0F0F0F0Fu : qw[v] & 0x0F0F0F0Fu;
            wq[v] = (int)__vsub4(nib | (((hw[v] >> sh) & 0x03030303u) << 4u),
                                 0x20202020u);
        }
        const uint32_t si = n * 8u + rw * 2u + h;
        __half hd;
        memcpy(&hd, rec, 2u);
        *f = __half2float(hd) * (float)((const int8_t*)rec)[4u + si];
    } else if (dtype == PD_KQ_IQ4XS) {
        const uint32_t ib = w >> 1u, h = w & 1u;
        const uint4 qa = __ldcs((const uint4*)(sb + ib * 16u));
        const uint32_t qw[4] = {qa.x, qa.y, qa.z, qa.w};
        #pragma unroll
        for (uint32_t v = 0; v < 4u; ++v)
            wq[v] = pd_kq_iq4_prmt((h ? qw[v] >> 4u : qw[v]) & 0x0F0F0F0Fu);
        __half hd;
        memcpy(&hd, rec, 2u);
        *f = __half2float(hd) * (float)((const int8_t*)rec)[4u + ib];
    } else {  // Q4_K / Q5_K / Q4_0 (Q4_0 shares the Q4_K data convention)
        const bool q5 = dtype == PD_KQ_Q5K;
        const bool q40 = dtype == PD_KQ_Q40;
        const uint32_t j = w >> 1u, h = w & 1u;  // sub-block, 16-half
        const uint32_t qoff = (j >> 1u) * 32u + h * 16u;
        const bool hi = (j & 1u) != 0u;
        const uint32_t C = q5 ? 0x10101010u : 0x08080808u;
        const uint4 qa = __ldcs((const uint4*)(sb + qoff));
        uint4 ha = make_uint4(0u, 0u, 0u, 0u);
        if (q5) ha = __ldcs((const uint4*)(sb + 128u + h * 16u));
        const uint32_t qw[4] = {qa.x, qa.y, qa.z, qa.w};
        const uint32_t hw[4] = {ha.x, ha.y, ha.z, ha.w};
        #pragma unroll
        for (uint32_t v = 0; v < 4u; ++v) {
            uint32_t nib = hi ? (qw[v] >> 4u) & 0x0F0F0F0Fu : qw[v] & 0x0F0F0F0Fu;
            if (q5) nib |= ((hw[v] >> j) & 0x01010101u) << 4u;
            wq[v] = (int)__vsub4(nib, C);
        }
        __half hd, hm;
        memcpy(&hd, rec, 2u);
        memcpy(&hm, rec + 2u, 2u);
        const float d = __half2float(hd);
        const float dj = q40 ? pd_kq40_dj(rec, j) : d * (float)rec[4u + j];
        *f = dj;
        // Q4_0's value is the centered dj*(q-8): no mu term at all (a
        // nonzero one would re-add the 8-center a second time)
        *g = q40 ? 0.0f
                 : (q5 ? 16.0f : 8.0f) * dj
                       - __half2float(hm) * (float)rec[12u + j];
    }
}

// The same window unpack with the i-quant tables supplied by the caller
// (a shared-memory copy, see quant/iquant_dense.cuh).
__device__ __forceinline__ void pd_kq_win_unpack_t(
        uint32_t dtype, const uint8_t* __restrict__ sb,
        const uint8_t* __restrict__ rec, uint32_t w, int wq[4], float* f,
        float* g, const PdIqTabs& tabs) {
    *g = 0.0f;
    if (pd_kq_valid_iq(dtype)) {
        pd_iq_win_unpack_t(dtype, sb, rec, w, wq, f, g, tabs);
        return;
    }
    if (dtype == PD_KQ_Q6K) {
        const uint32_t n = w >> 3u, rw = (w >> 1u) & 3u, h = w & 1u;
        const uint32_t qoff = n * 64u + (rw & 1u) * 32u + h * 16u;
        const uint32_t hoff = 128u + n * 32u + h * 16u;
        const uint32_t sh = 2u * rw;
        const bool hi = rw >= 2u;
        const uint4 qa = __ldcs((const uint4*)(sb + qoff));
        const uint4 ha = __ldcs((const uint4*)(sb + hoff));
        const uint32_t qw[4] = {qa.x, qa.y, qa.z, qa.w};
        const uint32_t hw[4] = {ha.x, ha.y, ha.z, ha.w};
        #pragma unroll
        for (uint32_t v = 0; v < 4u; ++v) {
            const uint32_t nib = hi ? (qw[v] >> 4u) & 0x0F0F0F0Fu : qw[v] & 0x0F0F0F0Fu;
            wq[v] = (int)__vsub4(nib | (((hw[v] >> sh) & 0x03030303u) << 4u),
                                 0x20202020u);
        }
        const uint32_t si = n * 8u + rw * 2u + h;
        __half hd;
        memcpy(&hd, rec, 2u);
        *f = __half2float(hd) * (float)((const int8_t*)rec)[4u + si];
    } else if (dtype == PD_KQ_IQ4XS) {
        const uint32_t ib = w >> 1u, h = w & 1u;
        const uint4 qa = __ldcs((const uint4*)(sb + ib * 16u));
        const uint32_t qw[4] = {qa.x, qa.y, qa.z, qa.w};
        #pragma unroll
        for (uint32_t v = 0; v < 4u; ++v)
            wq[v] = pd_kq_iq4_prmt((h ? qw[v] >> 4u : qw[v]) & 0x0F0F0F0Fu);
        __half hd;
        memcpy(&hd, rec, 2u);
        *f = __half2float(hd) * (float)((const int8_t*)rec)[4u + ib];
    } else {  // Q4_K / Q5_K / Q4_0 (Q4_0 shares the Q4_K data convention)
        const bool q5 = dtype == PD_KQ_Q5K;
        const bool q40 = dtype == PD_KQ_Q40;
        const uint32_t j = w >> 1u, h = w & 1u;  // sub-block, 16-half
        const uint32_t qoff = (j >> 1u) * 32u + h * 16u;
        const bool hi = (j & 1u) != 0u;
        const uint32_t C = q5 ? 0x10101010u : 0x08080808u;
        const uint4 qa = __ldcs((const uint4*)(sb + qoff));
        uint4 ha = make_uint4(0u, 0u, 0u, 0u);
        if (q5) ha = __ldcs((const uint4*)(sb + 128u + h * 16u));
        const uint32_t qw[4] = {qa.x, qa.y, qa.z, qa.w};
        const uint32_t hw[4] = {ha.x, ha.y, ha.z, ha.w};
        #pragma unroll
        for (uint32_t v = 0; v < 4u; ++v) {
            uint32_t nib = hi ? (qw[v] >> 4u) & 0x0F0F0F0Fu : qw[v] & 0x0F0F0F0Fu;
            if (q5) nib |= ((hw[v] >> j) & 0x01010101u) << 4u;
            wq[v] = (int)__vsub4(nib, C);
        }
        __half hd, hm;
        memcpy(&hd, rec, 2u);
        memcpy(&hm, rec + 2u, 2u);
        const float d = __half2float(hd);
        const float dj = q40 ? pd_kq40_dj(rec, j) : d * (float)rec[4u + j];
        *f = dj;
        // Q4_0's value is the centered dj*(q-8): no mu term at all (a
        // nonzero one would re-add the 8-center a second time)
        *g = q40 ? 0.0f
                 : (q5 ? 16.0f : 8.0f) * dj
                       - __half2float(hm) * (float)rec[12u + j];
    }
}

// ---- W4A8 decode GEMV (the b=1 SERVING class) ---------------------------------
// The mmvq-class design point (llama.cpp's own decode class; ExLlamaV2 is the
// tuned reference - quant strategy: k-quant files -> W4A8-int8
// runtime): int8-quantized activations staged PACKED in shared, integer dp4a
// dots off the raw nibble streams. Per 32 weights that is ~1 weight LDG.128 +
// ~10 unpack ops + 2 x LDS.128 + 8 dp4a + a handful of f32 folds (~35 issue
// slots) where the exact-f32 GEMV pays an LDS + magic-build + FMA per WEIGHT
// (~128) - the f32 class measured issue-bound at 498-586 GB/s vs the Q8 ref's
// 630-658. Geometry proven by that same GEMV: 4 adjacent rows/block (one
// 4-row DRAM span), 64 threads/row; x + scales + sums staged once (int8 x is
// tiny: 17 KB at in_dim 12288), one barrier, no tiling.
//
// Numeric class: exact int8 dots with f32 scale application - identical
// expressions to pd_kquant_gemm_dp4a's lane windows (the class the whole
// batch ladder serves and the parity suite gates at ~1e-7). Not the exact-f32
// oracle class: pd_kquant_gemv above stays the reference path
// (PADDOCK_KQ_EXACT_GEMV=1 pins serving back to it for attribution runs).
// ROWS = output rows per block, NT = threads per block; TPR = NT/ROWS is the
// thread-per-row width and the only thing the numerics see. 4 rows x 256 is
// the streaming default; 2 doubles the block count for SMALL out_dims
// (wk/wv-class 1024-row mats put only 256 four-row blocks on a die that seats
// 420 - profiling showed the in-graph small shapes running well under the
// solo rate). Different TPR = different chunk->thread partition = commutative
// regrouping only, and the pick is a pure function of (out_dim, in_dim,
// dtype) - deterministic per tensor.
//
// TPR NARROWING is dead here (laguna decode shapes, sm_86).
// A row has nch chunks (in_dim/32 on the 32-weight formats, in_dim/64 on
// Q6_K), and when nch < TPR the surplus threads walk zero chunks - 87% of the
// block on shexp-down (in 512, Q6_K: 8 chunks across 64 threads). Narrowing
// TPR to nch is bit-exact (same one-chunk-per-thread map, same 32-lane tree,
// the dropped warp partials were 0.0f) and it BUYS nothing: min-of-5 over the
// nine laguna decode shapes x eight (ROWS, NT) splits put every variant inside
// +-5% of production with no consistent direction (r2x128 wins qkg-SWA by
// 2.7%, loses qkg-full by 15%; run-to-run on the identical config is 1.3%).
// The idle threads cost nothing because the smem staging and the weight stream
// are the bound. The small shapes' real problem is elsewhere: 1.25 MB in
// 7.8 us is 160 GB/s against the card's ~630 (lm_head, same kernel, same run)
// - they are LAUNCH-floored, and the fix is fusing them into fewer launches,
// not re-mapping threads inside one. NT stays a template parameter only so the
// bench can keep asking; production picks ROWS by out_dim as it always has.
// The same narrowing is a win on moe/kquant.cuh's gate_up (-22%) - the
// difference is that that kernel stages nothing, so its idle half was pure
// tail. Do not generalize either result to the other shape without a bench.
// (TPR must stay >= 32 in any case - the fold assumes warp-aligned rows.)
// Per-row chunk walk shared by the single-plane kernel below and the
// multi-segment sibling (pd_kquant_gemv_w4a8_multi_kernel): one row
// accumulates over its ns_row super-blocks, TPR threads striding chunks.
// Extracted verbatim from the single kernel so the merge
// sibling runs the exact same unpack math - an edit here changes BOTH.
template <uint32_t TPR>
__device__ __forceinline__ float pd_kq_w4a8_row_acc(
        const uint8_t* __restrict__ rowd, const uint8_t* __restrict__ rows,
        const int* sxq, const float* sxs, const float* ssm,
        uint32_t tt, uint32_t ns_row, uint32_t dtype, uint32_t datab) {
    float acc = 0.0f;

    if (dtype == PD_KQ_Q6K) {
        // merged (s, n, h) task: both ql 32-halves against one qh chunk (the
        // quant/kquant.cuh multi-tile walk's mapping) - 64 weights per 3 loads.
        const uint32_t nch = ns_row << 2u;
        for (uint32_t c = tt; c < nch; c += TPR) {
            const uint32_t s = c >> 2u, ci = c & 3u;
            const uint32_t n = ci >> 1u, h = ci & 1u;
            const uint8_t* sb = rowd + (size_t)s * PD_KQ6_DATA;
            const uint4 qa = __ldcs((const uint4*)(sb + n * 64u + h * 16u));
            const uint4 qb = __ldcs((const uint4*)(sb + n * 64u + 32u + h * 16u));
            const uint4 hv = __ldcs((const uint4*)(sb + 128u + n * 32u + h * 16u));
            const uint8_t* rec = rows + (size_t)s * PD_KQ_SCB;
            __half hd;
            memcpy(&hd, rec, 2u);
            const float d = __half2float(hd);
            const int8_t* sc = (const int8_t*)rec + 4;
            const uint32_t xb = s * 256u + n * 128u + h * 16u;
            const int4 x00 = *(const int4*)(sxq + (xb >> 2u));           // a=0 lo
            const int4 x10 = *(const int4*)(sxq + ((xb + 32u) >> 2u));   // a=1 lo
            const int4 x01 = *(const int4*)(sxq + ((xb + 64u) >> 2u));   // a=0 hi
            const int4 x11 = *(const int4*)(sxq + ((xb + 96u) >> 2u));   // a=1 hi
            const uint32_t q0[4] = {qa.x, qa.y, qa.z, qa.w};
            const uint32_t q1[4] = {qb.x, qb.y, qb.z, qb.w};
            const uint32_t hw[4] = {hv.x, hv.y, hv.z, hv.w};
            const int xw00[4] = {x00.x, x00.y, x00.z, x00.w};
            const int xw10[4] = {x10.x, x10.y, x10.z, x10.w};
            const int xw01[4] = {x01.x, x01.y, x01.z, x01.w};
            const int xw11[4] = {x11.x, x11.y, x11.z, x11.w};
            int d00 = 0, d10 = 0, d01 = 0, d11 = 0;
            #pragma unroll
            for (uint32_t v = 0; v < 4u; ++v) {
                const uint32_t lo0 = q0[v] & 0x0F0F0F0Fu;
                const uint32_t hi0 = (q0[v] >> 4u) & 0x0F0F0F0Fu;
                const uint32_t lo1 = q1[v] & 0x0F0F0F0Fu;
                const uint32_t hi1 = (q1[v] >> 4u) & 0x0F0F0F0Fu;
                const int w00 = (int)__vsub4(
                    lo0 | ((hw[v] & 0x03030303u) << 4u), 0x20202020u);
                const int w10 = (int)__vsub4(
                    lo1 | (((hw[v] >> 2u) & 0x03030303u) << 4u), 0x20202020u);
                const int w01 = (int)__vsub4(
                    hi0 | (((hw[v] >> 4u) & 0x03030303u) << 4u), 0x20202020u);
                const int w11 = (int)__vsub4(
                    hi1 | (((hw[v] >> 6u) & 0x03030303u) << 4u), 0x20202020u);
                d00 = __dp4a(w00, xw00[v], d00);
                d10 = __dp4a(w10, xw10[v], d10);
                d01 = __dp4a(w01, xw01[v], d01);
                d11 = __dp4a(w11, xw11[v], d11);
            }
            const float f00 = d * (float)sc[n * 8u + h];
            const float f10 = d * (float)sc[n * 8u + 2u + h];
            const float f01 = d * (float)sc[n * 8u + 4u + h];
            const float f11 = d * (float)sc[n * 8u + 6u + h];
            acc += f00 * (sxs[xb >> 5u] * (float)d00);
            acc += f10 * (sxs[(xb + 32u) >> 5u] * (float)d10);
            acc += f01 * (sxs[(xb + 64u) >> 5u] * (float)d01);
            acc += f11 * (sxs[(xb + 96u) >> 5u] * (float)d11);
        }
    } else if (dtype == PD_KQ_IQ4XS) {
        // chunk (s, ib): 16 qs bytes = the whole 32-weight sub-block; prmt
        // register codebook straight to packed s8 (no float build needed -
        // dp4a eats the s8 directly).
        const uint32_t nch = ns_row << 3u;
        for (uint32_t c = tt; c < nch; c += TPR) {
            const uint32_t s = c >> 3u, ib = c & 7u;
            const uint4 qv = __ldcs((const uint4*)(rowd + (size_t)s * PD_IQ4_DATA + ib * 16u));
            const uint8_t* rec = rows + (size_t)s * PD_KQ_SCB;
            __half hd;
            memcpy(&hd, rec, 2u);
            const float f = __half2float(hd) * (float)((const int8_t*)rec)[4u + ib];
            const uint32_t xb = s * 256u + ib * 32u;
            const int4 xa = *(const int4*)(sxq + (xb >> 2u));
            const int4 xc = *(const int4*)(sxq + ((xb + 16u) >> 2u));
            const uint32_t qw[4] = {qv.x, qv.y, qv.z, qv.w};
            const int xwl[4] = {xa.x, xa.y, xa.z, xa.w};
            const int xwh[4] = {xc.x, xc.y, xc.z, xc.w};
            int dl = 0, dh = 0;
            #pragma unroll
            for (uint32_t v = 0; v < 4u; ++v) {
                dl = __dp4a(pd_kq_iq4_prmt(qw[v] & 0x0F0F0F0Fu), xwl[v], dl);
                dh = __dp4a(pd_kq_iq4_prmt((qw[v] >> 4u) & 0x0F0F0F0Fu), xwh[v], dh);
            }
            const float xsc = sxs[xb >> 5u];
            acc += f * (xsc * (float)dl);
            acc += f * (xsc * (float)dh);
        }
    } else {  // Q4_K / Q5_K / Q4_0
        // chunk (s, g, h): 16 qs bytes = 16 weights of sub-block 2g (lo
        // nibbles) + 16 of 2g+1 (hi); Q5's qh bytes ride bits 2g / 2g+1.
        const bool q5 = dtype == PD_KQ_Q5K;
        const bool q40 = dtype == PD_KQ_Q40;
        const uint32_t nch = ns_row << 3u;
        for (uint32_t c = tt; c < nch; c += TPR) {
            const uint32_t s = c >> 3u, ci = c & 7u;
            const uint32_t g = ci >> 1u, h = ci & 1u;
            const uint8_t* sb = rowd + (size_t)s * datab;
            const uint4 qv = __ldcs((const uint4*)(sb + g * 32u + h * 16u));
            uint4 hv = make_uint4(0u, 0u, 0u, 0u);
            if (q5) hv = __ldcs((const uint4*)(sb + 128u + h * 16u));
            const uint8_t* rec = rows + (size_t)s * PD_KQ_SCB;
            __half hd, hm;
            memcpy(&hd, rec, 2u);
            memcpy(&hm, rec + 2u, 2u);
            const float d = __half2float(hd), dmin = __half2float(hm);
            const uint32_t j1 = 2u * g, j2 = 2u * g + 1u;
            const float cf = q5 ? 16.0f : 8.0f;
            const float dj1 = q40 ? pd_kq40_dj(rec, j1) : d * (float)rec[4u + j1];
            const float mu1 = q40 ? 0.0f : cf * dj1 - dmin * (float)rec[12u + j1];
            const float dj2 = q40 ? pd_kq40_dj(rec, j2) : d * (float)rec[4u + j2];
            const float mu2 = q40 ? 0.0f : cf * dj2 - dmin * (float)rec[12u + j2];
            const uint32_t C = q5 ? 0x10101010u : 0x08080808u;
            const uint32_t xb = s * 256u + g * 64u + h * 16u;
            const int4 xa = *(const int4*)(sxq + (xb >> 2u));
            const int4 xc = *(const int4*)(sxq + ((xb + 32u) >> 2u));
            const uint32_t qw[4] = {qv.x, qv.y, qv.z, qv.w};
            const uint32_t hw[4] = {hv.x, hv.y, hv.z, hv.w};
            const int xwl[4] = {xa.x, xa.y, xa.z, xa.w};
            const int xwh[4] = {xc.x, xc.y, xc.z, xc.w};
            int dl = 0, dh = 0;
            #pragma unroll
            for (uint32_t v = 0; v < 4u; ++v) {
                uint32_t lo = qw[v] & 0x0F0F0F0Fu;
                uint32_t hi = (qw[v] >> 4u) & 0x0F0F0F0Fu;
                if (q5) {
                    lo |= ((hw[v] >> j1) & 0x01010101u) << 4u;
                    hi |= ((hw[v] >> j2) & 0x01010101u) << 4u;
                }
                dl = __dp4a((int)__vsub4(lo, C), xwl[v], dl);
                dh = __dp4a((int)__vsub4(hi, C), xwh[v], dh);
            }
            const float xs1 = sxs[xb >> 5u], xs2 = sxs[(xb >> 5u) + 1u];
            acc += dj1 * (xs1 * (float)dl);
            acc += dj2 * (xs2 * (float)dh);
            acc += mu1 * (xs1 * ssm[xb >> 4u]);
            acc += mu2 * (xs2 * ssm[(xb >> 4u) + 2u]);
        }
    }
    return acc;
}

template <uint32_t ROWS, uint32_t NT = 256u>
__global__ void __launch_bounds__(NT) pd_kquant_gemv_w4a8_kernel(
        const uint8_t* __restrict__ data, const uint8_t* __restrict__ scales,
        const int8_t* __restrict__ xq, const float* __restrict__ xs,
        const float* __restrict__ xsums, float* __restrict__ y,
        uint32_t in_dim, uint32_t out_dim, uint32_t dtype) {
    //  PDL cascade (launched via pd_pdl_go): every input here (xq/xs/
    // xsums) is the predecessor quantize's output, so the arm sits at the very
    // top - no dep-free prologue to hide, the win is launch-ramp overlap.
    // No-op under plain launches and below sm_90.
    PD_PDL_ARM();
    constexpr uint32_t TPR = NT / ROWS;            // threads per row
    const uint32_t tid = threadIdx.x;
    const uint32_t lr = tid / TPR;                 // row-in-block
    const uint32_t o = blockIdx.x * ROWS + lr;
    const uint32_t tt = tid % TPR;                 // thread-in-row
    const uint32_t n_super = in_dim >> 8u;
    const uint32_t datab = pd_kq_datab(dtype);
    const bool mu = pd_kq_has_mu(dtype);
    // packed int8 x | per-32 scales | per-16 sums (mu formats only)
    extern __shared__ int pd_kg8_sh[];
    int* sxq = pd_kg8_sh;
    float* sxs = (float*)(pd_kg8_sh + (in_dim >> 2u));
    float* ssm = sxs + (in_dim >> 5u);
    for (uint32_t i = tid; i < (in_dim >> 4u); i += NT)
        ((int4*)sxq)[i] = ((const int4*)xq)[i];
    for (uint32_t i = tid; i < (in_dim >> 5u); i += NT) sxs[i] = xs[i];
    if (mu)
        for (uint32_t i = tid; i < (in_dim >> 4u); i += NT) ssm[i] = xsums[i];
    __syncthreads();
    // dead rows (out_dim % 4 tail) walk zero chunks but stay for the final
    // cross-warp barrier (the f32 GEMV's discipline)
    const uint32_t ns_row = o < out_dim ? n_super : 0u;
    const uint8_t* rowd = data + (size_t)o * n_super * datab;
    const uint8_t* rows = scales + (size_t)o * n_super * PD_KQ_SCB;
    float acc = pd_kq_w4a8_row_acc<TPR>(rowd, rows, sxq, sxs, ssm, tt, ns_row,
                                        dtype, datab);

    // rows are warp-aligned: warp-reduce, then one thread per row folds its
    // TPR/32 warp partials. 16 slots, not 8: NT=512 runs 16 warps, and the
    // old wsum8[8] sent warps 8-15 into the first 32 B of the dynamic-shared
    // staging region - dead bytes by then, so it read back correctly by
    // accident (found while building the multi sibling; output
    // bit-identical either way, this just makes the slot count real).
    static_assert(NT / 32u <= 16u, "wsum8 covers 16 warps");
    __shared__ float wsum8[16];
    for (uint32_t sdown = 16; sdown > 0; sdown >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, sdown);
    const uint32_t warp = tid >> 5u, lane = tid & 31u;
    if (lane == 0) wsum8[warp] = acc;
    __syncthreads();
    if (tid < ROWS) {
        const uint32_t ro = blockIdx.x * ROWS + tid;
        if (ro < out_dim) {
            constexpr uint32_t WPR = TPR / 32u;
            float v = 0.0f;
            #pragma unroll
            for (uint32_t w = 0; w < WPR; ++w) v += wsum8[tid * WPR + w];
            y[ro] = v;
        }
    }
}

// cached per-SM shared memory for the resident-thread NT election below
// (102400 B on sm_120a per the probed table; queried so the rule travels).
// Not PD_EXPORT: internal-linkage helper - MSVC refuses dllexport on a
// static, and nothing resolves it by name (the Windows build gate)
static inline uint32_t pd_kq_smem_per_sm() {
    static int v = 0;
    if (v == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev);
        if (v <= 0) v = 102400;
    }
    return (uint32_t)v;
}

int pd_kquant_gemv_w4a8(const void* data, const void* scales, const void* xq,
                        const void* xs, const void* xsums, void* y,
                        uint32_t in_dim, uint32_t out_dim, uint32_t dtype,
                        void* stream) {
    if (out_dim == 0) return 0;
    if (pd_kq_valid_iq(dtype))
        return pd_kq_iq_dense_dp4a(data, scales, xq, xs, xsums, y, in_dim, out_dim, 1u, dtype, stream);
    if ((in_dim & 255u) != 0u) return cudaErrorInvalidValue;
    if (!pd_kq_valid(dtype)) return cudaErrorInvalidValue;
    const bool mu = pd_kq_has_mu(dtype);
    if (mu && xsums == nullptr) return cudaErrorInvalidValue;
    // int8 x + per-32 f32 scales (+ per-16 f32 sums): 1.375*in_dim B max
    const uint32_t smem = in_dim + (in_dim >> 3u) + (mu ? in_dim >> 2u : 0u);
    cudaStream_t st = (cudaStream_t)stream;
    if (out_dim >= 2048u) {
        // Q4_K/Q5_K (mu formats) at 256 threads cap at 6 blk/SM on sm_120a's
        // real 1536 maxThreadsPerMultiProcessor (not 2048 -- see this file's
        // Q8_0 GEMV occupancy-ceiling entry); 128 threads doubles that to 12
        // and measured -12-13% on granite-4.1-30b's q/o (real decode shapes,
        // DRAM-cold). Q6_K/IQ4XS do not
        // get this: `down` (Q6_K, in_dim=32768) is already shared-memory-
        // bound at occ=2 regardless of thread count there, and narrowing hurt
        // it (+4.9%) instead -- less per-thread parallelism on an unchanged
        // occupancy tier, the same "TPR narrowing" trap this file's laguna
        // note already names.
        //
        // The NT=128 election is not unconditional for mu shapes (
        // the granite-30b c1 loss): it was derived on in_dim=4096 shapes
        // where occupancy is THREAD-capped, but Q4_K_M mixes dtypes per
        // layer and ~half of granite-30b's down planes are Q4_K at
        // in_dim=32768, where smem (1.375*in = 45KB) pins blocks at 2/SM
        // regardless of NT -- there NT=128 strands the SM at 256 resident
        // threads and ran 845 GB/s vs 1345 at NT=512 (bench downQ4 row;
        // live trace 798 GB/s = the entire c1 GEMV deficit vs llama.cpp).
        // Election: maximize resident threads min(smem_blocks, 1536/NT)*NT,
        // tie-break to the SMALLEST NT (more blocks hides latency better;
        // the in4096 tie measured in NT=128's favor). NT change regroups
        // the f32 scale accumulation -- the same sanctioned TPR-regrouping
        // numeric class as the k/v multi election, gated by the parity
        // suite's exact-int anchor.
        if (mu) {
            const uint32_t bsm = pd_kq_smem_per_sm() / smem;
            uint32_t nt_best = 128u, res_best = 0u;
            for (uint32_t nt = 128u; nt <= 512u; nt <<= 1u) {
                const uint32_t cap = 1536u / nt;
                const uint32_t res = (bsm < cap ? bsm : cap) * nt;
                if (res > res_best) { res_best = res; nt_best = nt; }
            }
            if (nt_best == 512u) {
                pd_pdl_go(pd_kquant_gemv_w4a8_kernel<4u, 512u>, (out_dim + 3u) / 4u, 512, smem, st,
                    (const uint8_t*)data, (const uint8_t*)scales, (const int8_t*)xq,
                    (const float*)xs, (const float*)xsums, (float*)y, in_dim, out_dim,
                    dtype);
            } else if (nt_best == 256u) {
                pd_pdl_go(pd_kquant_gemv_w4a8_kernel<4u, 256u>, (out_dim + 3u) / 4u, 256, smem, st,
                    (const uint8_t*)data, (const uint8_t*)scales, (const int8_t*)xq,
                    (const float*)xs, (const float*)xsums, (float*)y, in_dim, out_dim,
                    dtype);
            } else {
                pd_pdl_go(pd_kquant_gemv_w4a8_kernel<4u, 128u>, (out_dim + 3u) / 4u, 128, smem, st,
                    (const uint8_t*)data, (const uint8_t*)scales, (const int8_t*)xq,
                    (const float*)xs, (const float*)xsums, (float*)y, in_dim, out_dim,
                    dtype);
            }
        } else {
            pd_pdl_go(pd_kquant_gemv_w4a8_kernel<4u, 256u>, (out_dim + 3u) / 4u, 256, smem, st,
                (const uint8_t*)data, (const uint8_t*)scales, (const int8_t*)xq,
                (const float*)xs, (const float*)xsums, (float*)y, in_dim, out_dim,
                dtype);
        }
    } else {
        // 4 rows/512 threads beat the old 2 rows/256 threads on granite-30b's
        // k (-12%, ROWS=4/NT=512 vs ROWS=2/NT=256) and tied on v (both real
        // shapes, same bench) -- a dtype-agnostic win in this bucket, unlike
        // the >=2048 bucket above where Q4_K and Q6_K want different NT.
        pd_pdl_go(pd_kquant_gemv_w4a8_kernel<4u, 512u>, (out_dim + 3u) / 4u, 512, smem, st,
            (const uint8_t*)data, (const uint8_t*)scales, (const int8_t*)xq,
            (const float*)xs, (const float*)xsums, (float*)y, in_dim, out_dim,
            dtype);
    }
    return pd_launch_status();
}

// ---- W4A8 multi-segment decode GEMV (QKV / gate|up one-launch merge) --------
// Same launch economics as deltanet/core.cuh's pd_q8_0_gemv_repacked_multi,
// on the k-quant family: granite-30b's split k/v launches put 256 blocks on a
// die that seats 2256 at NT=128 and run latency-floored at ~6.1 us regardless
// of bytes (406/576 GB/s vs the die's 1531 practical ceiling), and q's solo
// 1024-block grid streams at 966 GB/s where the same config on gate/up's
// 8192-block grid does 1394 - pure ramp/drain amortization, bench-proven
// (gran30b_kquant_gemv_bench). One launch over up to 3 same-in_dim planes
// sharing one staged activation; segments may differ in k-quant dtype (the
// Q4_K_M file mixes q/k Q4_K with v Q6_K). At TPR=32 a row is a warp, so the
// per-row dtype branch never diverges inside a warp.
//
// Numeric class: the identical inner walk (pd_kq_w4a8_row_acc) at the merged
// launch's TPR. Planes whose solo election is already <4,128> (q, gate, up)
// are bit-identical to the split launch; k/v ran <4,512> solo, so merged rows
// are the same commutative-regrouping class at a different TPR (the header's
// "different TPR = different chunk->thread partition" law) - gated by the
// parity suite and the bench's memcmp-at-same-config section, not by a
// byte-diff against the differently-grouped solo config.
//
// Param-struct STACK trap (deltanet/core.cuh): segment resolve is
// a CONSTANT-INDEX if-chain; `segs.s[si]` with runtime si spills the whole
// by-value struct to local (STACK:120, measured -19/-42% on the Q8_0 twin).
struct PdKqGemvSeg {
    const uint8_t* data;
    const uint8_t* scales;
    float* y;
    uint32_t out_dim;
    uint32_t dtype;
};
struct PdKqGemvSegs3 { PdKqGemvSeg s[3]; };

template <uint32_t ROWS, uint32_t NT>
__global__ void __launch_bounds__(NT) pd_kquant_gemv_w4a8_multi_kernel(
        PdKqGemvSegs3 segs, const int8_t* __restrict__ xq,
        const float* __restrict__ xs, const float* __restrict__ xsums,
        uint32_t in_dim) {
    PD_PDL_ARM();
    constexpr uint32_t TPR = NT / ROWS;
    const uint32_t tid = threadIdx.x;
    const uint32_t lr = tid / TPR;
    const uint32_t o = blockIdx.x * ROWS + lr;
    const uint32_t tt = tid % TPR;
    const uint32_t n_super = in_dim >> 8u;
    // staging identical to the single kernel; sums plane only when a mu
    // segment exists (launcher passes xsums=null otherwise and sizes smem)
    extern __shared__ int pd_kg8m_sh[];
    int* sxq = pd_kg8m_sh;
    float* sxs = (float*)(pd_kg8m_sh + (in_dim >> 2u));
    float* ssm = sxs + (in_dim >> 5u);
    for (uint32_t i = tid; i < (in_dim >> 4u); i += NT)
        ((int4*)sxq)[i] = ((const int4*)xq)[i];
    for (uint32_t i = tid; i < (in_dim >> 5u); i += NT) sxs[i] = xs[i];
    if (xsums != nullptr)
        for (uint32_t i = tid; i < (in_dim >> 4u); i += NT) ssm[i] = xsums[i];
    __syncthreads();
    // constant-index segment resolve; grid-tail dead rows land past the last
    // real segment (the launcher pads slot 2 with out_dim=0 valid pointers),
    // walk zero chunks, and stay for the cross-warp barrier
    uint32_t so = o;
    const uint8_t* sd;
    const uint8_t* ss;
    uint32_t sod, sdt;
    if (o < segs.s[0].out_dim) {
        sd = segs.s[0].data; ss = segs.s[0].scales;
        sod = segs.s[0].out_dim; sdt = segs.s[0].dtype;
    } else if (o < segs.s[0].out_dim + segs.s[1].out_dim) {
        so = o - segs.s[0].out_dim;
        sd = segs.s[1].data; ss = segs.s[1].scales;
        sod = segs.s[1].out_dim; sdt = segs.s[1].dtype;
    } else {
        so = o - (segs.s[0].out_dim + segs.s[1].out_dim);
        sd = segs.s[2].data; ss = segs.s[2].scales;
        sod = segs.s[2].out_dim; sdt = segs.s[2].dtype;
    }
    const uint32_t datab = pd_kq_datab(sdt);
    const uint32_t ns_row = so < sod ? n_super : 0u;
    const uint8_t* rowd = sd + (size_t)so * n_super * datab;
    const uint8_t* rows = ss + (size_t)so * n_super * PD_KQ_SCB;
    float acc = pd_kq_w4a8_row_acc<TPR>(rowd, rows, sxq, sxs, ssm, tt, ns_row,
                                        sdt, datab);

    // same 16-slot discipline as the single kernel above (NT=128 uses 4)
    static_assert(NT / 32u <= 16u, "wsum8 covers 16 warps");
    __shared__ float wsum8[16];
    for (uint32_t sdown = 16; sdown > 0; sdown >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, sdown);
    const uint32_t warp = tid >> 5u, lane = tid & 31u;
    if (lane == 0) wsum8[warp] = acc;
    __syncthreads();
    if (tid < ROWS) {
        // the writer thread's row differs from the one its lane walked -
        // re-resolve the segment for the row it writes (same constant-index
        // discipline)
        const uint32_t ro = blockIdx.x * ROWS + tid;
        uint32_t wo = ro, wod;
        float* wy;
        if (ro < segs.s[0].out_dim) {
            wy = segs.s[0].y; wod = segs.s[0].out_dim;
        } else if (ro < segs.s[0].out_dim + segs.s[1].out_dim) {
            wo = ro - segs.s[0].out_dim;
            wy = segs.s[1].y; wod = segs.s[1].out_dim;
        } else {
            wo = ro - (segs.s[0].out_dim + segs.s[1].out_dim);
            wy = segs.s[2].y; wod = segs.s[2].out_dim;
        }
        if (wo < wod) {
            constexpr uint32_t WPR = TPR / 32u;
            float v = 0.0f;
            #pragma unroll
            for (uint32_t w = 0; w < WPR; ++w) v += wsum8[tid * WPR + w];
            wy[wo] = v;
        }
    }
}

PD_EXPORT
int pd_kquant_gemv_w4a8_multi(
        const void* d0, const void* s0, void* y0, uint32_t od0, uint32_t dt0,
        const void* d1, const void* s1, void* y1, uint32_t od1, uint32_t dt1,
        const void* d2, const void* s2, void* y2, uint32_t od2, uint32_t dt2,
        const void* xq, const void* xs, const void* xsums,
        uint32_t in_dim, uint32_t n_segs, void* stream) {
    if (n_segs < 2u || n_segs > 3u) return cudaErrorInvalidValue;
    if (pd_kq_valid_iq(dt0) || pd_kq_valid_iq(dt1) || (n_segs == 3u && pd_kq_valid_iq(dt2))) {
        // one launch per segment: the i-quant ones on their lane, the rest on
        // the single-plane W4A8 GEMV (the fused walk's per-column math)
        const void* ds[3] = {d0, d1, d2};
        const void* ss[3] = {s0, s1, s2};
        void* ys[3] = {y0, y1, y2};
        const uint32_t ods[3] = {od0, od1, od2};
        const uint32_t dts[3] = {dt0, dt1, dt2};
        for (uint32_t i = 0; i < n_segs; ++i) {
            if (ods[i] == 0u) continue;
            const int rc = pd_kq_valid_iq(dts[i])
                ? pd_kq_iq_dense_dp4a(ds[i], ss[i], xq, xs, xsums, ys[i], in_dim, ods[i], 1u, dts[i], stream)
                : pd_kquant_gemv_w4a8(ds[i], ss[i], xq, xs, xsums, ys[i], in_dim, ods[i], dts[i], stream);
            if (rc != 0) return rc;
        }
        return 0;
    }
    if ((in_dim & 255u) != 0u) return cudaErrorInvalidValue;
    PdKqGemvSegs3 segs;
    segs.s[0] = PdKqGemvSeg{(const uint8_t*)d0, (const uint8_t*)s0, (float*)y0, od0, dt0};
    segs.s[1] = PdKqGemvSeg{(const uint8_t*)d1, (const uint8_t*)s1, (float*)y1, od1, dt1};
    // unused slot 2 keeps VALID pointers at out_dim=0: grid-tail dead rows
    // compute (never dereference) a row pointer from whatever sits there
    segs.s[2] = n_segs == 3u
        ? PdKqGemvSeg{(const uint8_t*)d2, (const uint8_t*)s2, (float*)y2, od2, dt2}
        : PdKqGemvSeg{(const uint8_t*)d1, (const uint8_t*)s1, (float*)y1, 0u, dt1};
    bool mu = false;
    uint32_t total = 0u;
    for (uint32_t i = 0; i < n_segs; ++i) {
        if (!pd_kq_valid(segs.s[i].dtype) || segs.s[i].out_dim == 0u)
            return cudaErrorInvalidValue;
        mu = mu || pd_kq_has_mu(segs.s[i].dtype);
        total += segs.s[i].out_dim;
    }
    if (mu && xsums == nullptr) return cudaErrorInvalidValue;
    const uint32_t smem = in_dim + (in_dim >> 3u) + (mu ? in_dim >> 2u : 0u);
    // <4,128> unconditionally: the mu out_dim>=2048 election (q/gate/up
    // dominate every merged row count this serves; NT=128 doubles blocks/SM
    // on sm_120a's 1536-thread SM - see the single launcher's note). k/v ride
    // at TPR=32 instead of their solo 128 - merged-grid economics beat the
    // small planes' solo election (bench: split QKV 22.6 us, merged measured
    // in the bench's multi section).
    pd_pdl_go(pd_kquant_gemv_w4a8_multi_kernel<4u, 128u>, (total + 3u) / 4u, 128,
              smem, (cudaStream_t)stream, segs, (const int8_t*)xq,
              (const float*)xs, (const float*)(mu ? xsums : nullptr), in_dim);
    return pd_launch_status();
}

// ---- W4A8 fused-GLU decode GEMV (gate+up+SwiGLU, one launch) ---------------
// llama.cpp's fused GLU mmvq (studied as reference -
// their gate+up+silu*mul runs as one kernel/layer at 1449 GB/s with no
// separate swiglu; original implementation here) applied to the multi
// kernel's row walk: each block owns ROWS rows of the GLU OUTPUT and walks
// the gate row AND the matching up row against the same staged activation
// (weight bytes unchanged, activation staged once for both dots, the
// 2*out_dim f32 intermediate + the swiglu launch + its round-trip gone).
//
// Numeric class: BIT-EXACT vs the split path - each dot is the identical
// pd_kq_w4a8_row_acc walk at the multi's <4,128> TPR, and the epilogue is
// character-identical to pd_swiglu_kernel's (g / (1 + expf(-g))) * u. The
// parity test memcmps against multi+swiglu directly.
template <uint32_t ROWS, uint32_t NT>
__global__ void __launch_bounds__(NT) pd_kquant_gemv_w4a8_glu_kernel(
        const uint8_t* __restrict__ gd, const uint8_t* __restrict__ gs,
        const uint8_t* __restrict__ ud, const uint8_t* __restrict__ us,
        float* __restrict__ y, const int8_t* __restrict__ xq,
        const float* __restrict__ xs, const float* __restrict__ xsums,
        uint32_t in_dim, uint32_t out_dim, uint32_t dtg, uint32_t dtu) {
    PD_PDL_ARM();
    constexpr uint32_t TPR = NT / ROWS;
    const uint32_t tid = threadIdx.x;
    const uint32_t lr = tid / TPR;
    const uint32_t o = blockIdx.x * ROWS + lr;
    const uint32_t tt = tid % TPR;
    const uint32_t n_super = in_dim >> 8u;
    extern __shared__ int pd_kgg_sh[];
    int* sxq = pd_kgg_sh;
    float* sxs = (float*)(pd_kgg_sh + (in_dim >> 2u));
    float* ssm = sxs + (in_dim >> 5u);
    for (uint32_t i = tid; i < (in_dim >> 4u); i += NT)
        ((int4*)sxq)[i] = ((const int4*)xq)[i];
    for (uint32_t i = tid; i < (in_dim >> 5u); i += NT) sxs[i] = xs[i];
    if (xsums != nullptr)
        for (uint32_t i = tid; i < (in_dim >> 4u); i += NT) ssm[i] = xsums[i];
    __syncthreads();
    const uint32_t ns_row = o < out_dim ? n_super : 0u;
    const uint32_t gb = pd_kq_datab(dtg), ub = pd_kq_datab(dtu);
    float ag = pd_kq_w4a8_row_acc<TPR>(gd + (size_t)o * n_super * gb,
                                       gs + (size_t)o * n_super * PD_KQ_SCB,
                                       sxq, sxs, ssm, tt, ns_row, dtg, gb);
    float au = pd_kq_w4a8_row_acc<TPR>(ud + (size_t)o * n_super * ub,
                                       us + (size_t)o * n_super * PD_KQ_SCB,
                                       sxq, sxs, ssm, tt, ns_row, dtu, ub);
    static_assert(NT / 32u <= 16u, "wsum covers 16 warps");
    __shared__ float wsg[16], wsu[16];
    for (uint32_t sdown = 16; sdown > 0; sdown >>= 1) {
        ag += __shfl_down_sync(0xffffffffu, ag, sdown);
        au += __shfl_down_sync(0xffffffffu, au, sdown);
    }
    const uint32_t warp = tid >> 5u, lane = tid & 31u;
    if (lane == 0) { wsg[warp] = ag; wsu[warp] = au; }
    __syncthreads();
    if (tid < ROWS) {
        const uint32_t ro = blockIdx.x * ROWS + tid;
        if (ro < out_dim) {
            constexpr uint32_t WPR = TPR / 32u;
            float g = 0.0f, u = 0.0f;
            #pragma unroll
            for (uint32_t w = 0; w < WPR; ++w) {
                g += wsg[tid * WPR + w];
                u += wsu[tid * WPR + w];
            }
            y[ro] = (g / (1.0f + expf(-g))) * u;
        }
    }
}

PD_EXPORT
int pd_kquant_gemv_w4a8_glu(
        const void* gate_data, const void* gate_scales,
        const void* up_data, const void* up_scales,
        const void* xq, const void* xs, const void* xsums, void* y,
        uint32_t in_dim, uint32_t out_dim, uint32_t dtg, uint32_t dtu,
        void* stream) {
    if (out_dim == 0u) return cudaErrorInvalidValue;
    if (pd_kq_valid_iq(dtg) || pd_kq_valid_iq(dtu))
        return pd_kq_iq_dense_glu(gate_data, gate_scales, up_data, up_scales, xq, xs, xsums, y,
                                  in_dim, out_dim, dtg, dtu, stream);
    if ((in_dim & 255u) != 0u) return cudaErrorInvalidValue;
    if (!pd_kq_valid(dtg) || !pd_kq_valid(dtu)) return cudaErrorInvalidValue;
    const bool mu = pd_kq_has_mu(dtg) ||
                    pd_kq_has_mu(dtu);
    if (mu && xsums == nullptr) return cudaErrorInvalidValue;
    const uint32_t smem = in_dim + (in_dim >> 3u) + (mu ? in_dim >> 2u : 0u);
    // <4,128> to match the multi's election on the same in_dim class (the
    // gate|up activation is embd-wide, thread-bound; the resident-thread
    // rule from the single launcher picks 128 there too).
    pd_pdl_go(pd_kquant_gemv_w4a8_glu_kernel<4u, 128u>, (out_dim + 3u) / 4u, 128,
              smem, (cudaStream_t)stream,
              (const uint8_t*)gate_data, (const uint8_t*)gate_scales,
              (const uint8_t*)up_data, (const uint8_t*)up_scales,
              (float*)y, (const int8_t*)xq, (const float*)xs,
              (const float*)(mu ? xsums : nullptr), in_dim, out_dim, dtg, dtu);
    return pd_launch_status();
}

// ---- W4A8 multi-column decode GEMV (the spec-verify r-class, r = 2..5) --------
// The spec verify runs the backbone at r = k_draft+1 rows; those rode the
// K-split mma (455-496 GB/s weight-effective at r=4) while the b=1 GEMV above
// reads the same weights at 631-648 - a ~25% verify tax on every spec round.
// This is the mmvq ncols<=8 design point (llama.cpp's own draft-verify class):
// the same weight walk, row geometry and unpack as the single-column kernel,
// but each 16/32-weight window unpacks once into registers and dots against
// NCOLS activation vectors - weight traffic identical to b=1, r-fold flop
// reuse. ~25 ALU/cycle at the 620 GB/s stream rate vs the 64 available, so
// the weight stream stays the bound (the f32 GEMV's issue-bound failure mode
// needed ~4 ops per WEIGHT; this pays ~r per 4).
//
// Activations are the STRIDED pd_quantize_q8 layout (token-major rows - the
// same buffers the dp4a/mma_ks ladder eats), staged in shared through
// PD_KGN_WIN-elem windows (~28 KB at r=5 -> 3-4 blocks/SM; the full-plane
// v1 hit 1 block/SM at r>=4 and went latency-bound, 266-423 GB/s).
// Staging in shared is load-bearing: the GEMV-derived walk with per-chunk
// GLOBAL x windows measured L2-transaction-bound at ~110 GB/s (see the dp4a
// kernel's header) - do not "simplify" this to direct reads.
//
// Numeric class: per column identical expressions in identical chunk order to
// the single-column GEMV (exact int8 dots, f32 scale folds) - column t's
// output equals a b=1 GEMV run on that column. Deterministic for fixed shape.
//
// x planes are XOR-swizzled at 16 B granularity within each 128 B group:
// chunk-strided threads read the plane at a 128 B stride, which lands every
// lane of a warp on the same 4 banks (32-way serialization). At r=1 the
// conflicted LDS hides under the weight stream; at r=4/5 it is the bound
// (measured 271-502 GB/s pre-swizzle vs the 600+ target). The swizzle is a
// pure within-group permutation applied at store AND load - bit-inert.
__device__ __forceinline__ uint32_t pd_kgn_swz(uint32_t i4) {
    return i4 ^ ((i4 >> 3u) & 7u);
}

// x-window elems per staging span: a multiple of TPR*64 for both ROWS
// variants so every window's chunk range starts on a whole thread residue
// cycle, and small enough that r=5 stays under the 48 KB default window.
#define PD_KGN_WIN 4096u

template <uint32_t ROWS, uint32_t NCOLS>
__global__ void __launch_bounds__(256) pd_kquant_gemv_w4a8_nc_kernel(
        const uint8_t* __restrict__ data, const uint8_t* __restrict__ scales,
        const int8_t* __restrict__ xq, const float* __restrict__ xs,
        const float* __restrict__ xsums, float* __restrict__ y,
        uint32_t in_dim, uint32_t out_dim, uint32_t dtype) {
    constexpr uint32_t TPR = 256u / ROWS;          // threads per row
    // NCOLS >= 4: the Q4/Q5 mu term folds once per (row, sub-block) in a
    // separate pass instead of inline per chunk - the inline fold's 2 ssm
    // loads + mul/FMA pair per COLUMN per chunk is what collapsed the mu
    // formats at r>=4 (230-360 GB/s across three inline variants). The fold
    // is a summation REORDER: mu_j·(xsc_j·(S_2j+S_2j+1)) instead of two
    // h-split contributions in chunk order - same exact int dots, different
    // f32 fold order, so this class is gated by tolerance parity, not the
    // bit-identity gate. NCOLS 2..3 keep the inline fold; they walk the b=1
    // GEMV's chunk sequence at TPR 64 (ROWS=4) / 128 (ROWS=2), so they are
    // bit-identical to it exactly when the b=1 launcher lands the same TPR.
    // Since the mu NT election (b=1 picks NT per shape and die),
    // that is guaranteed only for the non-mu formats - for Q4K/Q5K it is a
    // per-shape accident, and the parity gate holds mu formats to the
    // TPR-regrouping tolerance class + exact-int anchor at every NCOLS
    constexpr bool MUFOLD = NCOLS >= 4u;
    const uint32_t tid = threadIdx.x;
    const uint32_t lr = tid / TPR;                 // row-in-block
    const uint32_t o = blockIdx.x * ROWS + lr;
    const uint32_t tt = tid % TPR;                 // thread-in-row
    const uint32_t n_super = in_dim >> 8u;
    const uint32_t datab = pd_kq_datab(dtype);
    const bool mu = pd_kq_has_mu(dtype);
    const uint32_t nb32 = in_dim >> 5u, nb16 = in_dim >> 4u;
    // Windowed x staging (PD_KGN_WIN elems per span): the full-plane variant
    // at r=4/5 cost 67-84 KB -> 1 block/SM and went latency-bound (266-423
    // GB/s); 4096-elem windows keep smem <= 28 KB -> 3-4 blocks/SM. Each
    // thread's chunk sequence (c = tt mod TPR ascending) is UNCHANGED by the
    // windows - the fold order, and so the bit-identity with the
    // single-column kernel, is preserved. in_dim <= win degenerates to the
    // single-stage shape exactly.
    constexpr uint32_t WIN = PD_KGN_WIN;
    // The plane strides follow the STAGED window, min(in_dim, WIN) - the
    // launcher sizes dynamic shared from that same expression. Deriving
    // them from WIN put every in_dim < 4096 plane's xs / sums planes past
    // the allocation: ILLEGAL_ADDRESS on a 2048-wide Q5_K shexp plane at
    // ncols 2 (issue #19), on sm_86 and sm_120 alike, since the windowed
    // staging landed. The 4096+ production planes never reached it.
    const uint32_t win = in_dim < WIN ? in_dim : WIN;
    const uint32_t wxi = win >> 2u;                // ints per window x plane
    const uint32_t w32 = win >> 5u, w16 = win >> 4u;
    extern __shared__ int pd_kgn_sh[];
    int* sxq = pd_kgn_sh;
    float* sxs = (float*)(pd_kgn_sh + NCOLS * wxi);
    float* ssm = sxs + NCOLS * w32;
    const uint32_t ns_row = o < out_dim ? n_super : 0u;
    const uint8_t* rowd = data + (size_t)o * n_super * datab;
    const uint8_t* rows = scales + (size_t)o * n_super * PD_KQ_SCB;
    float acc[NCOLS] = {};

    // per-format chunk size in elems (the walk granularities below)
    const uint32_t ce = dtype == PD_KQ_Q6K ? 64u : 32u;
    const uint32_t nch =
        dtype == PD_KQ_Q6K ? (ns_row << 2u) : (ns_row << 3u);

    for (uint32_t e0 = 0; e0 < in_dim; e0 += WIN) {
        const uint32_t we = (in_dim - e0) < WIN ? (in_dim - e0) : WIN;
        #pragma unroll
        for (uint32_t tc = 0; tc < NCOLS; ++tc) {
            const int4* gx = (const int4*)(xq + (size_t)tc * in_dim + e0);
            int4* sx = (int4*)(sxq + tc * wxi);
            for (uint32_t i = tid; i < (we >> 4u); i += 256u)
                sx[pd_kgn_swz(i)] = gx[i];
            const float* gs = xs + (size_t)tc * nb32 + (e0 >> 5u);
            float* ss = sxs + tc * w32;
            for (uint32_t i = tid; i < (we >> 5u); i += 256u) ss[i] = gs[i];
            if (mu) {
                const float* gm = xsums + (size_t)tc * nb16 + (e0 >> 4u);
                float* sm = ssm + tc * w16;
                for (uint32_t i = tid; i < (we >> 4u); i += 256u) sm[i] = gm[i];
            }
        }
        __syncthreads();
        // this window's chunks, this thread's residue class (c ≡ tt mod TPR,
        // ascending - the global sequence the single-column kernel walks)
        const uint32_t clo = e0 / ce;
        uint32_t chi = (e0 + we) / ce;
        if (chi > nch) chi = nch;
        const uint32_t cst = clo + ((tt + TPR - (clo % TPR)) % TPR);

    if (dtype == PD_KQ_Q6K) {
        // merged (s, n, h) chunk: 4 sixteen-weight windows per 3 loads (the
        // single-column kernel's mapping); weights unpack once, dot NCOLS x.
        for (uint32_t c = cst; c < chi; c += TPR) {
            const uint32_t s = c >> 2u, ci = c & 3u;
            const uint32_t n = ci >> 1u, h = ci & 1u;
            const uint8_t* sb = rowd + (size_t)s * PD_KQ6_DATA;
            const uint4 qa = __ldcs((const uint4*)(sb + n * 64u + h * 16u));
            const uint4 qb = __ldcs((const uint4*)(sb + n * 64u + 32u + h * 16u));
            const uint4 hv = __ldcs((const uint4*)(sb + 128u + n * 32u + h * 16u));
            const uint8_t* rec = rows + (size_t)s * PD_KQ_SCB;
            __half hd;
            memcpy(&hd, rec, 2u);
            const float d = __half2float(hd);
            const int8_t* sc = (const int8_t*)rec + 4;
            const uint32_t xb = s * 256u + n * 128u + h * 16u;
            const uint32_t q0[4] = {qa.x, qa.y, qa.z, qa.w};
            const uint32_t q1[4] = {qb.x, qb.y, qb.z, qb.w};
            const uint32_t hw[4] = {hv.x, hv.y, hv.z, hv.w};
            int w00[4], w10[4], w01[4], w11[4];
            #pragma unroll
            for (uint32_t v = 0; v < 4u; ++v) {
                const uint32_t lo0 = q0[v] & 0x0F0F0F0Fu;
                const uint32_t hi0 = (q0[v] >> 4u) & 0x0F0F0F0Fu;
                const uint32_t lo1 = q1[v] & 0x0F0F0F0Fu;
                const uint32_t hi1 = (q1[v] >> 4u) & 0x0F0F0F0Fu;
                w00[v] = (int)__vsub4(lo0 | ((hw[v] & 0x03030303u) << 4u), 0x20202020u);
                w10[v] = (int)__vsub4(lo1 | (((hw[v] >> 2u) & 0x03030303u) << 4u), 0x20202020u);
                w01[v] = (int)__vsub4(hi0 | (((hw[v] >> 4u) & 0x03030303u) << 4u), 0x20202020u);
                w11[v] = (int)__vsub4(hi1 | (((hw[v] >> 6u) & 0x03030303u) << 4u), 0x20202020u);
            }
            const float f00 = d * (float)sc[n * 8u + h];
            const float f10 = d * (float)sc[n * 8u + 2u + h];
            const float f01 = d * (float)sc[n * 8u + 4u + h];
            const float f11 = d * (float)sc[n * 8u + 6u + h];
            const uint32_t xl = xb - e0;  // window-local elem offset
            // unroll 1: full unroll hoisted every column's x loads into live
            // registers (~80+ regs at NCOLS=5 -> 1 block/SM, the same
            // latency-bound cliff the smem windows just fixed)
            #pragma unroll 1
            for (uint32_t tc = 0; tc < NCOLS; ++tc) {
                const int4* px4 = (const int4*)(sxq + tc * wxi);
                const float* pxs = sxs + tc * w32;
                const int4 x00 = px4[pd_kgn_swz(xl >> 4u)];
                const int4 x10 = px4[pd_kgn_swz((xl >> 4u) + 2u)];
                const int4 x01 = px4[pd_kgn_swz((xl >> 4u) + 4u)];
                const int4 x11 = px4[pd_kgn_swz((xl >> 4u) + 6u)];
                const int xw00[4] = {x00.x, x00.y, x00.z, x00.w};
                const int xw10[4] = {x10.x, x10.y, x10.z, x10.w};
                const int xw01[4] = {x01.x, x01.y, x01.z, x01.w};
                const int xw11[4] = {x11.x, x11.y, x11.z, x11.w};
                int d00 = 0, d10 = 0, d01 = 0, d11 = 0;
                #pragma unroll
                for (uint32_t v = 0; v < 4u; ++v) {
                    d00 = __dp4a(w00[v], xw00[v], d00);
                    d10 = __dp4a(w10[v], xw10[v], d10);
                    d01 = __dp4a(w01[v], xw01[v], d01);
                    d11 = __dp4a(w11[v], xw11[v], d11);
                }
                acc[tc] += f00 * (pxs[xl >> 5u] * (float)d00);
                acc[tc] += f10 * (pxs[(xl + 32u) >> 5u] * (float)d10);
                acc[tc] += f01 * (pxs[(xl + 64u) >> 5u] * (float)d01);
                acc[tc] += f11 * (pxs[(xl + 96u) >> 5u] * (float)d11);
            }
        }
    } else if (dtype == PD_KQ_IQ4XS) {
        for (uint32_t c = cst; c < chi; c += TPR) {
            const uint32_t s = c >> 3u, ib = c & 7u;
            const uint4 qv = __ldcs((const uint4*)(rowd + (size_t)s * PD_IQ4_DATA + ib * 16u));
            const uint8_t* rec = rows + (size_t)s * PD_KQ_SCB;
            __half hd;
            memcpy(&hd, rec, 2u);
            const float f = __half2float(hd) * (float)((const int8_t*)rec)[4u + ib];
            const uint32_t xb = s * 256u + ib * 32u;
            const uint32_t qw[4] = {qv.x, qv.y, qv.z, qv.w};
            int wl[4], wh[4];
            #pragma unroll
            for (uint32_t v = 0; v < 4u; ++v) {
                wl[v] = pd_kq_iq4_prmt(qw[v] & 0x0F0F0F0Fu);
                wh[v] = pd_kq_iq4_prmt((qw[v] >> 4u) & 0x0F0F0F0Fu);
            }
            const uint32_t xl = xb - e0;
            #pragma unroll 1
            for (uint32_t tc = 0; tc < NCOLS; ++tc) {
                const int4* px4 = (const int4*)(sxq + tc * wxi);
                const int4 xa = px4[pd_kgn_swz(xl >> 4u)];
                const int4 xc = px4[pd_kgn_swz((xl >> 4u) + 1u)];
                const int xwl[4] = {xa.x, xa.y, xa.z, xa.w};
                const int xwh[4] = {xc.x, xc.y, xc.z, xc.w};
                int dl = 0, dh = 0;
                #pragma unroll
                for (uint32_t v = 0; v < 4u; ++v) {
                    dl = __dp4a(wl[v], xwl[v], dl);
                    dh = __dp4a(wh[v], xwh[v], dh);
                }
                const float xsc = sxs[tc * w32 + (xl >> 5u)];
                acc[tc] += f * (xsc * (float)dl);
                acc[tc] += f * (xsc * (float)dh);
            }
        }
    } else {  // Q4_K / Q5_K / Q4_0
        const bool q5 = dtype == PD_KQ_Q5K;
        const bool q40 = dtype == PD_KQ_Q40;
        for (uint32_t c = cst; c < chi; c += TPR) {
            const uint32_t s = c >> 3u, ci = c & 7u;
            const uint32_t g = ci >> 1u, h = ci & 1u;
            const uint8_t* sb = rowd + (size_t)s * datab;
            const uint4 qv = __ldcs((const uint4*)(sb + g * 32u + h * 16u));
            uint4 hv = make_uint4(0u, 0u, 0u, 0u);
            if (q5) hv = __ldcs((const uint4*)(sb + 128u + h * 16u));
            const uint8_t* rec = rows + (size_t)s * PD_KQ_SCB;
            __half hd;
            memcpy(&hd, rec, 2u);
            const float d = __half2float(hd);
            const uint32_t j1 = 2u * g, j2 = 2u * g + 1u;
            const float dj1 = q40 ? pd_kq40_dj(rec, j1) : d * (float)rec[4u + j1];
            const float dj2 = q40 ? pd_kq40_dj(rec, j2) : d * (float)rec[4u + j2];
            float mu1 = 0.0f, mu2 = 0.0f;
            if (!MUFOLD) {
                __half hm;
                memcpy(&hm, rec + 2u, 2u);
                const float dmin = __half2float(hm);
                const float cf = q5 ? 16.0f : 8.0f;
                mu1 = q40 ? 0.0f : cf * dj1 - dmin * (float)rec[12u + j1];
                mu2 = q40 ? 0.0f : cf * dj2 - dmin * (float)rec[12u + j2];
            }
            const uint32_t C = q5 ? 0x10101010u : 0x08080808u;
            const uint32_t xb = s * 256u + g * 64u + h * 16u;
            const uint32_t qw[4] = {qv.x, qv.y, qv.z, qv.w};
            const uint32_t hw[4] = {hv.x, hv.y, hv.z, hv.w};
            int wl[4], wh[4];
            #pragma unroll
            for (uint32_t v = 0; v < 4u; ++v) {
                uint32_t lo = qw[v] & 0x0F0F0F0Fu;
                uint32_t hi = (qw[v] >> 4u) & 0x0F0F0F0Fu;
                if (q5) {
                    lo |= ((hw[v] >> j1) & 0x01010101u) << 4u;
                    hi |= ((hw[v] >> j2) & 0x01010101u) << 4u;
                }
                wl[v] = (int)__vsub4(lo, C);
                wh[v] = (int)__vsub4(hi, C);
            }
            const uint32_t xl = xb - e0;
            #pragma unroll 1
            for (uint32_t tc = 0; tc < NCOLS; ++tc) {
                const int4* px4 = (const int4*)(sxq + tc * wxi);
                const int4 xa = px4[pd_kgn_swz(xl >> 4u)];
                const int4 xc = px4[pd_kgn_swz((xl >> 4u) + 2u)];
                const int xwl[4] = {xa.x, xa.y, xa.z, xa.w};
                const int xwh[4] = {xc.x, xc.y, xc.z, xc.w};
                int dl = 0, dh = 0;
                #pragma unroll
                for (uint32_t v = 0; v < 4u; ++v) {
                    dl = __dp4a(wl[v], xwl[v], dl);
                    dh = __dp4a(wh[v], xwh[v], dh);
                }
                const float xs1 = sxs[tc * w32 + (xl >> 5u)];
                const float xs2 = sxs[tc * w32 + (xl >> 5u) + 1u];
                acc[tc] += dj1 * (xs1 * (float)dl);
                acc[tc] += dj2 * (xs2 * (float)dh);
                if (!MUFOLD) {
                    acc[tc] += mu1 * (xs1 * ssm[tc * w16 + (xl >> 4u)]);
                    acc[tc] += mu2 * (xs2 * ssm[tc * w16 + (xl >> 4u) + 2u]);
                }
            }
        }
        if (MUFOLD) {
            // per-row mu fold over this window's supers: (s, j) tasks strided
            // by the thread residue, each task one sub-block's mu against the
            // weight-independent xsc_j*(S_2j + S_2j+1). Task order is fixed
            // per thread and feeds acc before the warp reduce, so the output
            // stays deterministic for a fixed shape. Rec re-reads hit the
            // lines the chunk loop just pulled (L1/L2). ~128 tasks per
            // window vs the chunk loop's 128*NCOLS dot chains - noise.
            const uint32_t slo = e0 >> 8u;
            uint32_t shi = (e0 + we) >> 8u;
            if (shi > ns_row) shi = ns_row;
            const uint32_t ntask = shi > slo ? (shi - slo) << 3u : 0u;
            const float cf = q5 ? 16.0f : 8.0f;
            for (uint32_t ti = tt; ti < ntask; ti += TPR) {
                const uint32_t s = slo + (ti >> 3u), j = ti & 7u;
                const uint8_t* rec = rows + (size_t)s * PD_KQ_SCB;
                const bool q40f = dtype == PD_KQ_Q40;
                __half hd, hm;
                memcpy(&hd, rec, 2u);
                memcpy(&hm, rec + 2u, 2u);
                const float dj = q40f ? pd_kq40_dj(rec, j)
                                      : __half2float(hd) * (float)rec[4u + j];
                const float muj = q40f
                    ? 0.0f
                    : cf * dj - __half2float(hm) * (float)rec[12u + j];
                const uint32_t bl = (s << 3u) + j - (e0 >> 5u);  // window 32-blk
                #pragma unroll 1
                for (uint32_t tc = 0; tc < NCOLS; ++tc) {
                    const float xsc = sxs[tc * w32 + bl];
                    const float sw = ssm[tc * w16 + 2u * bl]
                                   + ssm[tc * w16 + 2u * bl + 1u];
                    acc[tc] += muj * (xsc * sw);
                }
            }
        }
    }
        __syncthreads();  // window planes reload next span
    }

    // per-column warp reduce, then one thread per row folds warp partials
    __shared__ float wsumn[8u * NCOLS];
    const uint32_t warp = tid >> 5u, lane = tid & 31u;
    #pragma unroll
    for (uint32_t tc = 0; tc < NCOLS; ++tc) {
        float a = acc[tc];
        for (uint32_t sd = 16; sd > 0; sd >>= 1)
            a += __shfl_down_sync(0xffffffffu, a, sd);
        if (lane == 0) wsumn[warp * NCOLS + tc] = a;
    }
    __syncthreads();
    if (tid < ROWS) {
        const uint32_t ro = blockIdx.x * ROWS + tid;
        if (ro < out_dim) {
            constexpr uint32_t WPR = TPR / 32u;
            #pragma unroll
            for (uint32_t tc = 0; tc < NCOLS; ++tc) {
                float v = 0.0f;
                #pragma unroll
                for (uint32_t w = 0; w < WPR; ++w)
                    v += wsumn[(tid * WPR + w) * NCOLS + tc];
                y[(size_t)tc * out_dim + ro] = v;
            }
        }
    }
}

PD_EXPORT
int pd_kquant_gemv_w4a8_nc(const void* data, const void* scales, const void* xq,
                           const void* xs, const void* xsums, void* y,
                           uint32_t in_dim, uint32_t out_dim, uint32_t ncols,
                           uint32_t dtype, void* stream) {
    if (out_dim == 0) return 0;
    if (pd_kq_valid_iq(dtype))
        return pd_kq_iq_dense_dp4a(data, scales, xq, xs, xsums, y, in_dim, out_dim, ncols, dtype, stream);
    if ((in_dim & 255u) != 0u) return cudaErrorInvalidValue;
    if (!pd_kq_valid(dtype)) return cudaErrorInvalidValue;
    if (ncols < 2u || ncols > 5u) return cudaErrorInvalidValue;
    const bool mu = pd_kq_has_mu(dtype);
    if (mu && xsums == nullptr) return cudaErrorInvalidValue;
    // windowed staging: smem is bounded by the WINDOW, not in_dim (max
    // 5 * 5632 B = 27.5 KB - under the 48 KB default, no opt-in needed)
    const uint32_t w = in_dim < PD_KGN_WIN ? in_dim : PD_KGN_WIN;
    const uint32_t smem = ncols * (w + (w >> 3u) + (mu ? w >> 2u : 0u));
    cudaStream_t st = (cudaStream_t)stream;
    #define PD_KGN_LAUNCH(RV, NCV)                                                \
        do {                                                                      \
            pd_kquant_gemv_w4a8_nc_kernel<RV, NCV>                                \
                <<<(out_dim + RV - 1u) / RV, 256, smem, st>>>(                    \
                    (const uint8_t*)data, (const uint8_t*)scales,                 \
                    (const int8_t*)xq, (const float*)xs, (const float*)xsums,     \
                    (float*)y, in_dim, out_dim, dtype);                           \
        } while (0)
    #define PD_KGN_ROWS(NCV)                                                      \
        do {                                                                      \
            if (out_dim >= 2048u) PD_KGN_LAUNCH(4u, NCV);                         \
            else PD_KGN_LAUNCH(2u, NCV);                                          \
        } while (0)
    switch (ncols) {
        case 2u: PD_KGN_ROWS(2u); break;
        case 3u: PD_KGN_ROWS(3u); break;
        case 4u: PD_KGN_ROWS(4u); break;
        default: PD_KGN_ROWS(5u); break;
    }
    #undef PD_KGN_ROWS
    #undef PD_KGN_LAUNCH
    return pd_launch_status();
}

// ring smem bytes for <DT, BN, ST> - launcher and kernel share this sum (the
// kernel static_asserts its plane offsets against it). Pads: weight rows
// +16 B (raw row stride is a multiple of 128 B - unpadded, every row lands
// on the same banks); xs rows 8->12 f32 and sums rows 16->20 f32 (the
// 2t-strided compute reads would otherwise 4-way conflict).
__host__ __device__ constexpr uint32_t pd_km_smem_bytes(uint32_t dt, uint32_t bn,
                                                        uint32_t st) {
    return st * (64u * ((dt == PD_KQ_Q6K   ? PD_KQ6_DATA
                         : dt == PD_KQ_Q5K ? PD_KQ5_DATA
                                           : PD_KQ4_DATA) + 16u)
                 + 64u * PD_KQ_SCB
                 + bn * (PD_KM_BSTR * 4u)
                 + bn * 48u
                 + ((pd_kq_has_mu(dt)) ? bn * 80u : 0u));
}

template <uint32_t DT, uint32_t BN, uint32_t ST>
__global__ void __launch_bounds__(256) pd_kquant_mma_ks_kernel(
        const uint8_t* __restrict__ data, const uint8_t* __restrict__ scales,
        const int8_t* __restrict__ xq, const float* __restrict__ xs,
        const float* __restrict__ xsums, float* __restrict__ y,
        uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
#if PD_MMA_OK
    constexpr bool MU = (DT == PD_KQ_Q4K || DT == PD_KQ_Q5K || DT == PD_KQ_Q40);
    constexpr bool K16 = (DT == PD_KQ_Q6K);
    constexpr uint32_t CPW = BN / 2u;    // 8 warps = 4 row x 2 col
    constexpr uint32_t NSUB = CPW / 8u;  // 16x8 col sub-tiles per warp
    constexpr uint32_t DATAB = DT == PD_KQ_Q6K ? PD_KQ6_DATA
                             : DT == PD_KQ_Q5K ? PD_KQ5_DATA : PD_KQ4_DATA;
    constexpr uint32_t WSTR = DATAB + 16u;  // padded raw weight row stride
    static_assert(ST >= 1u && ST <= 4u, "stage count");

    // ST-deep ring planes: raw weights | raw scale recs | activations (final
    // int8, PD_KM_BSTR stride) | per-32 xs (12-f32 rows) | per-16 sums
    // (20-f32 rows, MU only)
    constexpr uint32_t W_PL = 64u * WSTR, R_PL = 64u * PD_KQ_SCB;
    constexpr uint32_t B_PL = BN * (PD_KM_BSTR * 4u);
    constexpr uint32_t XS_PL = BN * 48u, SU_PL = BN * 80u;
    constexpr uint32_t OFF_R = ST * W_PL, OFF_B = OFF_R + ST * R_PL;
    constexpr uint32_t OFF_XS = OFF_B + ST * B_PL;
    constexpr uint32_t OFF_SU = OFF_XS + ST * XS_PL;
    static_assert(OFF_SU + (MU ? ST * SU_PL : 0u) == pd_km_smem_bytes(DT, BN, ST),
                  "smem layout matches the launcher's size");
    extern __shared__ __align__(16) unsigned char pd_km_sh[];
    auto rw = [&](uint32_t buf) { return pd_km_sh + buf * W_PL; };
    auto rrec = [&](uint32_t buf) { return pd_km_sh + OFF_R + buf * R_PL; };
    auto rb = [&](uint32_t buf) {
        return (const int*)(pd_km_sh + OFF_B + buf * B_PL);
    };
    auto rxs = [&](uint32_t buf) {
        return (const float*)(pd_km_sh + OFF_XS + buf * XS_PL);
    };
    auto rsu = [&](uint32_t buf) {
        return (const float*)(pd_km_sh + OFF_SU + buf * SU_PL);
    };

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5u;
    const uint32_t g = lane >> 2u, t = lane & 3u;
    const uint32_t wr = (warp & 3u) * 16u;
    const uint32_t wc = (warp >> 2u) * CPW;
    const uint32_t row_base = blockIdx.x * 64u;
    const uint32_t col_base = blockIdx.y * BN;
    const uint32_t n_super = in_dim >> 8u;
    const uint32_t nb32 = in_dim >> 5u, nb16 = in_dim >> 4u;

    // K-split in SUPERS: the launcher picks nz so every z-slice is non-empty
    // and combines the partial planes in fixed z order. gridDim.z == 1 keeps
    // the single-plane walk writing y direct.
    uint32_t s_lo = 0u, s_hi = n_super;
    if (gridDim.z > 1u) {
        const uint32_t per = (n_super + gridDim.z - 1u) / gridDim.z;
        s_lo = blockIdx.z * per;
        s_hi = s_lo + per < n_super ? s_lo + per : n_super;
        y += (size_t)blockIdx.z * out_dim * batch;
    }

    float acc[NSUB][4];
    #pragma unroll
    for (uint32_t s = 0; s < NSUB; ++s) {
        #pragma unroll
        for (uint32_t i = 0; i < 4u; ++i) acc[s][i] = 0.0f;
    }

    // stage super kt's five planes into ring buffer `buf` - every copy is
    // cp.async; commit happens at the call site (Q8 ring discipline)
    auto stage = [&](uint32_t kt, uint32_t buf) {
        constexpr uint32_t WI4 = DATAB / 16u;  // 16 B chunks per weight row
        for (uint32_t i = tid; i < 64u * WI4; i += 256u) {
            const uint32_t row = i / WI4, c = i % WI4;
            const bool ok = (row_base + row) < out_dim;
            pd_mma_cpa16p(rw(buf) + row * WSTR + c * 16u,
                          data + ((size_t)(row_base + row) * n_super + kt) * DATAB
                              + c * 16u, ok);
        }
        for (uint32_t i = tid; i < 64u * 3u; i += 256u) {  // recs: 3 x 8 B
            const uint32_t row = i / 3u, c = i % 3u;
            const bool ok = (row_base + row) < out_dim;
            pd_kq_cpa8p(rrec(buf) + row * PD_KQ_SCB + c * 8u,
                        scales + ((size_t)(row_base + row) * n_super + kt) *
                            PD_KQ_SCB + c * 8u, ok);
        }
        const uint32_t k0 = kt * 256u;
        for (uint32_t i = tid; i < BN * 16u; i += 256u) {
            const uint32_t col = i >> 4u, k16i = i & 15u;
            const bool ok = (col_base + col) < batch;
            pd_mma_cpa16p((unsigned char*)rb(buf) + col * (PD_KM_BSTR * 4u)
                              + k16i * 16u,
                          xq + (size_t)(col_base + col) * in_dim + k0 + k16i * 16u,
                          ok);
        }
        for (uint32_t i = tid; i < BN * 2u; i += 256u) {  // per-32 scales
            const uint32_t col = i >> 1u, h = i & 1u;
            const bool ok = (col_base + col) < batch;
            pd_mma_cpa16p((unsigned char*)rxs(buf) + col * 48u + h * 16u,
                          xs + (size_t)(col_base + col) * nb32 + kt * 8u + h * 4u,
                          ok);
        }
        if (MU) {
            for (uint32_t i = tid; i < BN * 4u; i += 256u) {  // per-16 sums
                const uint32_t col = i >> 2u, h = i & 3u;
                const bool ok = (col_base + col) < batch;
                pd_mma_cpa16p((unsigned char*)rsu(buf) + col * 80u + h * 16u,
                              xsums + (size_t)(col_base + col) * nb16 + kt * 16u
                                  + h * 4u, ok);
            }
        }
    };

    // compute the staged super in `buf`: fragments unpack inline from the raw
    // strips; scale records expand per-thread in registers. Zero-filled dead
    // rows unpack to nonzero s8 (e.g. -8 for Q4_K) but their zero-filled recs
    // make every scale 0, so their contributions vanish exactly as v1's
    // explicit zero staging did (and the store guards skip them anyway).
    auto compute = [&](uint32_t buf) {
        const uint8_t* w0p = rw(buf) + (wr + g) * WSTR;
        const uint8_t* w8p = w0p + 8u * WSTR;
        const uint8_t* re0 = rrec(buf) + (wr + g) * PD_KQ_SCB;
        const uint8_t* re8 = re0 + 8u * PD_KQ_SCB;
        const int* rbv = rb(buf);
        const float* rxsv = rxs(buf);
        const float* rsuv = MU ? rsu(buf) : nullptr;

        // per-row scale material -> registers, extracted per kk below.
        // d/dmin as halfs at rec+0; scale/min bytes at rec+4/+12 (Q4/Q5),
        // 16 s8 at rec+4 (Q6), 8 s8 at rec+4 (IQ4) - same fields v1 read.
        float df0, dx0, df8, dx8;  // d and (Q4/Q5) dmin per row group
        uint32_t sw0[4], sw8[4];
        {
            const uint32_t h0 = *(const uint32_t*)re0;
            const uint32_t h8 = *(const uint32_t*)re8;
            df0 = __half2float(__ushort_as_half((unsigned short)(h0 & 0xFFFFu)));
            df8 = __half2float(__ushort_as_half((unsigned short)(h8 & 0xFFFFu)));
            dx0 = MU ? __half2float(__ushort_as_half((unsigned short)(h0 >> 16u)))
                     : 0.0f;
            dx8 = MU ? __half2float(__ushort_as_half((unsigned short)(h8 >> 16u)))
                     : 0.0f;
            #pragma unroll
            for (uint32_t j = 0; j < 4u; ++j) {
                // Q4/Q5: [0..1] 8 scale bytes, [2..3] 8 min bytes;
                // Q6: 16 s8 scales; IQ4: 8 s8 scales in [0..1]
                sw0[j] = *(const uint32_t*)(re0 + 4u + 4u * j);
                sw8[j] = *(const uint32_t*)(re8 + 4u + 4u * j);
            }
        }

        // inline fragment unpack: 4 centered s8 for (raw row, byte pos k4).
        // Index math inverts the repack layouts exactly as v1's staging tasks
        // did - same nibble sources, same __vsub4 centering, same LUT.
        auto unp = [&](const uint8_t* wrow, uint32_t k4) -> int {
            if (DT == PD_KQ_Q6K) {
                const uint32_t n = k4 >> 7u, r2 = k4 & 127u;
                const bool lo = r2 < 64u;
                const uint32_t rr = lo ? r2 : r2 - 64u;
                const uint32_t qw = *(const uint32_t*)(wrow + n * 64u + rr);
                const uint32_t hw =
                    *(const uint32_t*)(wrow + 128u + n * 32u + (rr & 31u));
                const uint32_t sh = 2u * (rr >> 5u) + (lo ? 0u : 4u);
                const uint32_t nib = (lo ? qw : qw >> 4u) & 0x0F0F0F0Fu;
                return (int)__vsub4(nib | (((hw >> sh) & 0x03030303u) << 4u),
                                    0x20202020u);
            } else if (DT == PD_KQ_IQ4XS) {
                const uint32_t ib = k4 >> 5u, r = k4 & 31u;
                const bool lo = r < 16u;
                const uint32_t qw = *(const uint32_t*)(wrow + ib * 16u + (r & 15u));
                return pd_kq_iq4_prmt((lo ? qw : qw >> 4u) & 0x0F0F0F0Fu);
            } else {  // Q4_K / Q5_K
                const uint32_t gq = k4 >> 6u, r = k4 & 63u;
                const bool lo = r < 32u;
                const uint32_t rr = lo ? r : r - 32u;
                const uint32_t qw = *(const uint32_t*)(wrow + gq * 32u + rr);
                uint32_t nib = (lo ? qw : qw >> 4u) & 0x0F0F0F0Fu;
                if (DT == PD_KQ_Q5K) {
                    const uint32_t hw = *(const uint32_t*)(wrow + 128u + rr);
                    nib |= ((hw >> (2u * gq + (lo ? 0u : 1u))) & 0x01010101u) << 4u;
                }
                return (int)__vsub4(
                    nib, DT == PD_KQ_Q5K ? 0x10101010u : 0x08080808u);
            }
        };

        #pragma unroll
        for (uint32_t kk = 0; kk < 8u; ++kk) {
            const uint32_t ko = kk * 8u;
            const uint32_t k4a = kk * 32u + t * 4u;
            const int a0 = unp(w0p, k4a);
            const int a1 = unp(w8p, k4a);
            const int a2 = unp(w0p, k4a + 16u);
            const int a3 = unp(w8p, k4a + 16u);
            float d0s = 0.0f, d8s = 0.0f, m0s = 0.0f, m8s = 0.0f;
            float s0lo = 0.0f, s0hi = 0.0f, s8lo = 0.0f, s8hi = 0.0f;
            if (K16) {
                // per-16 s8 scales: bytes 2kk / 2kk+1 of the 16-byte run
                s0lo = df0 * (float)(int8_t)(sw0[kk >> 1u] >> (8u * ((2u * kk) & 3u)));
                s0hi = df0 * (float)(int8_t)(sw0[kk >> 1u] >> (8u * ((2u * kk + 1u) & 3u)));
                s8lo = df8 * (float)(int8_t)(sw8[kk >> 1u] >> (8u * ((2u * kk) & 3u)));
                s8hi = df8 * (float)(int8_t)(sw8[kk >> 1u] >> (8u * ((2u * kk + 1u) & 3u)));
            } else if (DT == PD_KQ_IQ4XS) {
                d0s = df0 * (float)(int8_t)(sw0[kk >> 2u] >> (8u * (kk & 3u)));
                d8s = df8 * (float)(int8_t)(sw8[kk >> 2u] >> (8u * (kk & 3u)));
            } else if (DT == PD_KQ_Q40) {
                // {f16 dsub[8]}: read straight off the staged record (kk is
                // the sub-block); zero-filled dead-row recs give dsub = 0.
                __half h0v, h8v;
                memcpy(&h0v, re0 + 2u * kk, 2u);
                memcpy(&h8v, re8 + 2u * kk, 2u);
                d0s = __half2float(h0v);
                d8s = __half2float(h8v);
                // value is the centered d*(q-8) already: mu stays 0
            } else {  // Q4_K / Q5_K: unsigned scale/min bytes + mu fold
                const uint32_t sh_ = 8u * (kk & 3u);
                const uint32_t i2 = kk >> 2u;
                const float Cf = DT == PD_KQ_Q5K ? 16.0f : 8.0f;
                d0s = df0 * (float)((sw0[i2] >> sh_) & 0xFFu);
                d8s = df8 * (float)((sw8[i2] >> sh_) & 0xFFu);
                m0s = Cf * d0s - dx0 * (float)((sw0[2u + i2] >> sh_) & 0xFFu);
                m8s = Cf * d8s - dx8 * (float)((sw8[2u + i2] >> sh_) & 0xFFu);
            }
            #pragma unroll
            for (uint32_t sub = 0; sub < NSUB; ++sub) {
                const uint32_t csub = wc + sub * 8u;
                const int b0 = rbv[(csub + g) * PD_KM_BSTR + ko + t];
                const int b1 = rbv[(csub + g) * PD_KM_BSTR + ko + 4u + t];
                const float xc0 = rxsv[(csub + 2u * t) * 12u + kk];
                const float xc1 = rxsv[(csub + 2u * t + 1u) * 12u + kk];
                if (K16) {
                    // per-16 scales: two k16 mmas, halves scaled apart
                    int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                    int e0 = 0, e1 = 0, e2 = 0, e3 = 0;
                    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
                        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                        : "r"(a0), "r"(a1), "r"(b0));
                    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
                        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                        : "+r"(e0), "+r"(e1), "+r"(e2), "+r"(e3)
                        : "r"(a2), "r"(a3), "r"(b1));
                    acc[sub][0] += xc0 * (s0lo * (float)d0 + s0hi * (float)e0);
                    acc[sub][1] += xc1 * (s0lo * (float)d1 + s0hi * (float)e1);
                    acc[sub][2] += xc0 * (s8lo * (float)d2 + s8hi * (float)e2);
                    acc[sub][3] += xc1 * (s8lo * (float)d3 + s8hi * (float)e3);
                } else {
                    int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
                    acc[sub][0] += d0s * xc0 * (float)d0;
                    acc[sub][1] += d0s * xc1 * (float)d1;
                    acc[sub][2] += d8s * xc0 * (float)d2;
                    acc[sub][3] += d8s * xc1 * (float)d3;
                    if (MU) {
                        // mu operand rebuilt inline: same xc*(s16a+s16b)
                        // product v1 staged into tile_sx - identical f32 ops
                        const float sx0 =
                            xc0 * (rsuv[(csub + 2u * t) * 20u + 2u * kk]
                                   + rsuv[(csub + 2u * t) * 20u + 2u * kk + 1u]);
                        const float sx1 =
                            xc1 * (rsuv[(csub + 2u * t + 1u) * 20u + 2u * kk]
                                   + rsuv[(csub + 2u * t + 1u) * 20u + 2u * kk + 1u]);
                        acc[sub][0] += m0s * sx0;
                        acc[sub][1] += m0s * sx1;
                        acc[sub][2] += m8s * sx0;
                        acc[sub][3] += m8s * sx1;
                    }
                }
            }
        }
    };

    // ST-deep ring, Q8 mma_ks discipline: one commit group per iteration
    // always (empty groups are legal PTX and complete immediately) so the
    // wait immediate stays uniform; the trailing barrier after compute is the
    // write-hazard fence for the next issue into the just-read buffer.
    #pragma unroll
    for (uint32_t s = 0; s + 1u < ST; ++s) {
        const uint32_t kt = s_lo + s;
        if (kt < s_hi) stage(kt, s);
        pd_attn_cpa_commit();
    }
    uint32_t p = 0;
    for (uint32_t kt = s_lo; kt < s_hi; ++kt) {
        const uint32_t pre = kt + (ST - 1u);
        if (pre < s_hi) stage(pre, (p + ST - 1u) % ST);
        pd_attn_cpa_commit();
        pd_mma_cpa_waitN<(int)ST - 1>();
        __syncthreads();
        compute(p);
        __syncthreads();
        p = (p + 1u) % ST;
    }

    // store: (row, tok) -> y[tok*out_dim + row] (partial plane when nz > 1)
    const uint32_t or0 = row_base + wr + g, or8 = or0 + 8u;
    #pragma unroll
    for (uint32_t sub = 0; sub < NSUB; ++sub) {
        const uint32_t c0 = col_base + wc + sub * 8u + 2u * t;
        const uint32_t c1 = c0 + 1u;
        if (or0 < out_dim) {
            if (c0 < batch) y[(size_t)c0 * out_dim + or0] = acc[sub][0];
            if (c1 < batch) y[(size_t)c1 * out_dim + or0] = acc[sub][1];
        }
        if (or8 < out_dim) {
            if (c0 < batch) y[(size_t)c0 * out_dim + or8] = acc[sub][2];
            if (c1 < batch) y[(size_t)c1 * out_dim + or8] = acc[sub][3];
        }
    }
#else
    (void)data; (void)scales; (void)xq; (void)xs; (void)xsums; (void)y;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}

PD_EXPORT
int pd_kquant_gemm_mma_ks(const void* data, const void* scales, const void* xq,
                          const void* xs, const void* xsums, void* part, void* y,
                          uint32_t in_dim, uint32_t out_dim, uint32_t batch,
                          uint32_t dtype, void* stream) {
    if (out_dim == 0 || batch == 0) return 0;
    if (pd_kq_valid_iq(dtype))
        return pd_kq_iq_dense_dp4a(data, scales, xq, xs, xsums, y, in_dim, out_dim, batch, dtype, stream);
    if ((in_dim & 255u) != 0u) return cudaErrorInvalidValue;
    if (!pd_kq_valid(dtype)) return cudaErrorInvalidValue;
    if (batch > 64u) return cudaErrorInvalidValue;  // >64 is the mmq tile's rung
    if ((pd_kq_has_mu(dtype)) && xsums == nullptr)
        return cudaErrorInvalidValue;
    auto st = (cudaStream_t)stream;
    static int nsm = 0;
    if (nsm == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
        if (nsm <= 0) nsm = 128;
    }
    const uint32_t tiles = (out_dim + 63u) / 64u;
    const uint32_t n_super = in_dim >> 8u;
    uint32_t nz = ((uint32_t)nsm * 2u + tiles - 1u) / tiles;
    if (nz > 8u) nz = 8u;
    if (nz > n_super) nz = n_super;
    if (nz < 1u) nz = 1u;
    // slice in supers, then recompute nz from the slice size so no z-range
    // is empty (an empty range would leave its partial plane unwritten)
    const uint32_t per = (n_super + nz - 1u) / nz;
    nz = (n_super + per - 1u) / per;
    float* dst = nz > 1u ? (float*)part : (float*)y;
    // dynamic smem per (DT, BN): the ring blows the 48 KB static window on
    // the wider/heavier variants (Q4K BN64 ~71 KB), so opt in per
    // instantiation. BN16/BN32 Q4K/IQ4 stay under 48 KB -> 2 blocks/SM.
    #define PD_KM_LAUNCH(DTV, BNV)                                                \
        do {                                                                      \
            constexpr uint32_t smem = pd_km_smem_bytes(DTV, BNV, 2u);             \
            if (smem > 48u * 1024u) {                                             \
                static cudaError_t attr = cudaFuncSetAttribute(                   \
                    (const void*)pd_kquant_mma_ks_kernel<DTV, BNV, 2u>,           \
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);      \
                if (attr != cudaSuccess) return attr;                             \
            }                                                                     \
            pd_kquant_mma_ks_kernel<DTV, BNV, 2u>                                 \
                <<<dim3(tiles, 1u, nz), 256, smem, st>>>(                         \
                    (const uint8_t*)data, (const uint8_t*)scales,                 \
                    (const int8_t*)xq, (const float*)xs, (const float*)xsums,     \
                    dst, in_dim, out_dim, batch);                                 \
        } while (0)
    #define PD_KM_BN(BNV)                                                         \
        switch (dtype) {                                                          \
            case PD_KQ_Q40: PD_KM_LAUNCH(PD_KQ_Q40, BNV); break;                  \
            case PD_KQ_Q4K: PD_KM_LAUNCH(PD_KQ_Q4K, BNV); break;                  \
            case PD_KQ_Q5K: PD_KM_LAUNCH(PD_KQ_Q5K, BNV); break;                  \
            case PD_KQ_Q6K: PD_KM_LAUNCH(PD_KQ_Q6K, BNV); break;                  \
            default: PD_KM_LAUNCH(PD_KQ_IQ4XS, BNV); break;                       \
        }
    if (batch <= 16u) { PD_KM_BN(16u); }
    else if (batch <= 32u) { PD_KM_BN(32u); }
    else { PD_KM_BN(64u); }
    #undef PD_KM_BN
    #undef PD_KM_LAUNCH
    if (nz > 1u) {
        const uint32_t n = out_dim * batch;
        pd_q8_0_gemm_mma_ks_combine_kernel<<<(n + 255u) / 256u, 256, 0, st>>>(
            (const float*)part, nullptr, (float*)y, n, nz, out_dim);
    }
    return pd_launch_status();
}

// ---- pipelined (cp.async single-buffer overlap) W4A8 GEMM - the >64-batch rung
// Same 128x128 tile, warp/lane geometry, and per-DT unpack math as
// pd_kquant_w4a8_kernel above ("v1") - only the byte SOURCE changes. v1 issues
// a synchronous vectorized global load per thread for every super-block's
// weight+scale bytes, then unpacks in registers straight into tile_x: every
// warp parks on DRAM latency once per kt. Profiling a live granite-4.1-30b
// server (8-way concurrent prefill, sm_120a) showed this as
// 79.6% of all prefill GPU time - the exact profile Q8_0's own mmq kernel had
// before it grew `_hi`/`_pipe` siblings. K-quant never got that pass.
//
// This ports pd_kquant_mma_ks_kernel's already-proven technique (above - built
// for the 17..64-batch decode rung, Marlin's design point: raw compressed
// strips ride cp.async, unpack happens off the shared copy, never touching
// global memory synchronously) onto this file's 128x128 prefill tile.
//
// Not a textbook double buffer: a first cut ring-buffered both the raw
// weight+scale bytes AND both activation halves 2-deep and blew sm_120's
// ~101,376 B opt-in shared-memory cap for Q5_K/Q6_K (Q6_K's DATAB=192 alone
// needed ~117 KB). The fix that actually fits: only the weight+scale bytes
// (the real DRAM-latency source; activations/xsums are already final int8/f32
// and cheap to reload synchronously, exactly like v1) get a single raw buffer,
// and the overlap comes from re-ordering the loop, not from a second buffer -
// stage(kt+1) is issued right after build_tilex(kt) finishes reading the
// current buffer (safe: all reads done), so kt+1's fetch runs in the
// background for the entire h=0/h=1 MMA compute of kt, the actual slow part.
// Total dynamic shared drops to ~28-38 KB (worst case Q6_K ~89 KB all-in with
// tile_x), comfortably under the cap.
//
// tile_x's build math is UNCHANGED from v1 - only where the raw bytes come
// from (a prefetched shared buffer instead of global `data`/`scales`) moves.
// A dead row's buffer bytes zero-fill (cp.async's ok=false path) and unpack
// to a nonzero-but-deterministic s8 value (e.g. -8 for Q4_K); its
// ALSO-zero-filled scale record makes that row's contribution 0 regardless -
// the same invariant pd_kquant_mma_ks_kernel already relies on, so v1's
// explicit `live` guards are kept here rather than relied upon (belt and
// suspenders, not a numerics dependency).
// Everything (tile_x, tile_y, tile_s, the raw weight+scale buffer) lives in
// one dynamic (extern) shared allocation, opted in via cudaFuncSetAttribute:
// static `__shared__` arrays are capped at CUDA's hard 48 KB default with no
// opt-in override (that override only raises the DYNAMIC ceiling, up to
// sm_120's ~101,376 B) - tile_x alone (43,008 B) already leaves no room for
// tile_y/tile_s/the raw buffer under 48 KB if any of them are static.
__host__ __device__ __forceinline__ uint32_t pd_kwp_smem_bytes(uint32_t dt) {
    const uint32_t datab = pd_kq_datab(dt);
    const bool mu = pd_kq_has_mu(dt);
    const uint32_t tile_x_sz = 128u * PD_KW_XK * 4u;
    const uint32_t tile_y_sz = 128u * PD_MMQ_YK * 4u;
    const uint32_t tile_s_sz = mu ? 128u * 4u * 4u : 16u;  // 16B-aligned placeholder when unused
    const uint32_t raw_w = 128u * datab;
    const uint32_t raw_r = 128u * PD_KQ_SCB;
    return tile_x_sz + tile_y_sz + tile_s_sz + raw_w + raw_r;
}

template <uint32_t DT>
__global__ void __launch_bounds__(256, 1) pd_kquant_w4a8_pipe_kernel(
        const uint8_t* __restrict__ data, const uint8_t* __restrict__ scales,
        const uint8_t* __restrict__ yq, const float* __restrict__ xsums,
        float* __restrict__ y, uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
#if PD_MMA_OK
    constexpr bool MU = (DT == PD_KQ_Q4K || DT == PD_KQ_Q5K || DT == PD_KQ_Q40);
    constexpr bool K16 = (DT == PD_KQ_Q6K);
    constexpr uint32_t DATAB = DT == PD_KQ_Q6K ? PD_KQ6_DATA
                             : DT == PD_KQ_Q5K ? PD_KQ5_DATA : PD_KQ4_DATA;
    constexpr uint32_t RAW_W = 128u * DATAB;
    constexpr uint32_t TILE_X_SZ = 128u * PD_KW_XK * 4u;
    constexpr uint32_t TILE_Y_SZ = 128u * PD_MMQ_YK * 4u;
    constexpr uint32_t TILE_S_SZ = MU ? 128u * 4u * 4u : 16u;  // 16B-aligned placeholder
    constexpr uint32_t OFF_TILEY = TILE_X_SZ;
    constexpr uint32_t OFF_TILES = OFF_TILEY + TILE_Y_SZ;
    constexpr uint32_t OFF_RAWW = OFF_TILES + TILE_S_SZ;
    constexpr uint32_t OFF_RAWR = OFF_RAWW + RAW_W;

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5u;
    const uint32_t g = lane >> 2u, t = lane & 3u;
    const uint32_t i0 = (warp >> 1u) * 32u;
    const uint32_t joff = (warp & 1u) * 8u;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t n_super = in_dim >> 8u;
    const uint32_t nct = batch_pad >> 7u;
    const uint32_t tile = blockIdx.x;
    const uint32_t row_base = (tile / nct) * 128u;
    const uint32_t col_base = (tile % nct) * 128u;

    __shared__ int s_lut[16];
    if (DT == PD_KQ_IQ4XS) {
        if (tid < 16u) s_lut[tid] = (int)PD_KQ_IQ4NL[tid];
        __syncthreads();
    }

    // Everything below is carved out of one dynamic shared allocation (see
    // pd_kwp_smem_bytes) - tile_x: single-buffered unpacked weight tile,
    // rebuilt fresh from the raw buffer each kt, same PD_KW_XK layout as v1
    // (64 payload + 16 scale/mu f32 + 4 pad). tile_y/tile_s: v1's own
    // activation staging, reloaded synchronously per h - no latency-hiding
    // needed there (small, cheap). raw_w/raw_r: the single raw weight+scale
    // buffer this pipe kernel adds.
    extern __shared__ __align__(16) unsigned char pd_kwp_sh[];
    int* const tile_x = (int*)pd_kwp_sh;
    int* const tile_y = (int*)(pd_kwp_sh + OFF_TILEY);
    float* const tile_s = (float*)(pd_kwp_sh + OFF_TILES);
    unsigned char* const raw_w = pd_kwp_sh + OFF_RAWW;
    unsigned char* const raw_r = pd_kwp_sh + OFF_RAWR;
    float acc[16][4] = {};

    // stage super `kt`'s raw weight+scale bytes into the (single) raw
    // buffer - every copy is cp.async, commit at the call site.
    auto stage = [&](uint32_t kt) {
        constexpr uint32_t WCH = DATAB / 16u;  // 16 B chunks per weight row
        for (uint32_t i = tid; i < 128u * WCH; i += 256u) {
            const uint32_t row = i / WCH, c = i % WCH;
            const bool ok = (row_base + row) < out_dim;
            pd_mma_cpa16p(raw_w + row * DATAB + c * 16u,
                          data + ((size_t)(row_base + row) * n_super + kt) * DATAB
                              + c * 16u, ok);
        }
        for (uint32_t i = tid; i < 128u * 3u; i += 256u) {  // scale rec: 3 x 8 B
            const uint32_t row = i / 3u, c = i % 3u;
            const bool ok = (row_base + row) < out_dim;
            pd_kq_cpa8p(raw_r + row * PD_KQ_SCB + c * 8u,
                        scales + ((size_t)(row_base + row) * n_super + kt) *
                            PD_KQ_SCB + c * 8u, ok);
        }
    };

    // v1's own per-h activation load (unchanged) - synchronous, no ring.
    auto load_act = [&](uint32_t kt, uint32_t h) {
        const uint32_t chunk = kt * 2u + h;
        const int* by = (const int*)(yq + ((size_t)chunk * batch_pad + col_base) * 144u);
        #pragma unroll
        for (uint32_t it = 0; it < 18u; ++it) {  // 128*36 == 18*256 exactly
            const uint32_t l = it * 256u + tid;
            tile_y[l] = by[l];
        }
        if (MU) {
            #pragma unroll
            for (uint32_t it = 0; it < 2u; ++it) {
                const uint32_t l = it * 256u + tid;
                tile_s[l] = xsums[((size_t)chunk * batch_pad + col_base) * 4u + l];
            }
        }
    };

    // rebuild tile_x from the (already-landed) raw buffer - identical
    // arithmetic to v1, only the byte source changed. Callers must have
    // already cp.async-waited for this kt's `stage()` before calling.
    auto build_tilex = [&]() {
        #pragma unroll
        for (uint32_t it = 0; it < 4u; ++it) {
            const uint32_t i = it * 256u + tid;
            const uint32_t row = i >> 3u, ci = i & 7u;
            const bool live = (row_base + row) < out_dim;
            const uint8_t* sb = raw_w + row * DATAB;
            int out[8] = {};
            uint32_t obase, hioff;
            if (DT == PD_KQ_Q6K) {
                const uint32_t n = ci >> 2u, a = (ci >> 1u) & 1u, h = ci & 1u;
                obase = n * 32u + a * 8u + h * 4u;
                hioff = 16u;
                if (live) {
                    const uint4 qv = *(const uint4*)(sb + n * 64u + a * 32u + h * 16u);
                    const uint4 hv = *(const uint4*)(sb + 128u + n * 32u + h * 16u);
                    const uint32_t qw[4] = {qv.x, qv.y, qv.z, qv.w};
                    const uint32_t hw[4] = {hv.x, hv.y, hv.z, hv.w};
                    const uint32_t sh1 = 2u * a, sh2 = 2u * a + 4u;
                    #pragma unroll
                    for (uint32_t wv = 0; wv < 4u; ++wv) {
                        const uint32_t lo = (qw[wv] & 0x0F0F0F0Fu)
                            | (((hw[wv] >> sh1) & 0x03030303u) << 4u);
                        const uint32_t hi = ((qw[wv] >> 4u) & 0x0F0F0F0Fu)
                            | (((hw[wv] >> sh2) & 0x03030303u) << 4u);
                        out[wv] = (int)__vsub4(lo, 0x20202020u);
                        out[4u + wv] = (int)__vsub4(hi, 0x20202020u);
                    }
                }
            } else if (DT == PD_KQ_IQ4XS) {
                obase = ci * 8u;
                hioff = 4u;
                if (live) {
                    const uint4 qv = *(const uint4*)(sb + ci * 16u);
                    const uint32_t qw[4] = {qv.x, qv.y, qv.z, qv.w};
                    #pragma unroll
                    for (uint32_t wv = 0; wv < 4u; ++wv) {
                        uint32_t lo = 0u, hi = 0u;
                        #pragma unroll
                        for (uint32_t b = 0; b < 4u; ++b) {
                            const uint32_t qb = (qw[wv] >> (8u * b)) & 0xFFu;
                            lo |= ((uint32_t)(uint8_t)s_lut[qb & 0xFu]) << (8u * b);
                            hi |= ((uint32_t)(uint8_t)s_lut[qb >> 4u]) << (8u * b);
                        }
                        out[wv] = (int)lo;
                        out[4u + wv] = (int)hi;
                    }
                }
            } else {  // Q4_K / Q5_K
                const uint32_t gq = ci >> 1u, h = ci & 1u;
                obase = gq * 16u + h * 4u;
                hioff = 8u;
                if (live) {
                    const uint4 qv = *(const uint4*)(sb + gq * 32u + h * 16u);
                    uint4 hv = make_uint4(0u, 0u, 0u, 0u);
                    if (DT == PD_KQ_Q5K) hv = *(const uint4*)(sb + 128u + h * 16u);
                    const uint32_t qw[4] = {qv.x, qv.y, qv.z, qv.w};
                    const uint32_t hw[4] = {hv.x, hv.y, hv.z, hv.w};
                    const uint32_t j1 = 2u * gq, j2 = 2u * gq + 1u;
                    const uint32_t C = DT == PD_KQ_Q5K ? 0x10101010u : 0x08080808u;
                    #pragma unroll
                    for (uint32_t wv = 0; wv < 4u; ++wv) {
                        uint32_t lo = qw[wv] & 0x0F0F0F0Fu;
                        uint32_t hi = (qw[wv] >> 4u) & 0x0F0F0F0Fu;
                        if (DT == PD_KQ_Q5K) {
                            lo |= ((hw[wv] >> j1) & 0x01010101u) << 4u;
                            hi |= ((hw[wv] >> j2) & 0x01010101u) << 4u;
                        }
                        out[wv] = (int)__vsub4(lo, C);
                        out[4u + wv] = (int)__vsub4(hi, C);
                    }
                }
            }
            int* dst = tile_x + row * PD_KW_XK + obase;
            #pragma unroll
            for (uint32_t wv = 0; wv < 4u; ++wv) {
                dst[wv] = out[wv];
                dst[hioff + wv] = out[4u + wv];
            }
        }
        if (tid < 128u) {
            const uint32_t row = tid;
            float* sc = (float*)(tile_x + row * PD_KW_XK + 64u);
            const bool live = (row_base + row) < out_dim;
            const uint8_t* rec = raw_r + row * PD_KQ_SCB;
            if (DT == PD_KQ_Q6K) {
                float d = 0.0f;
                if (live) {
                    __half hd;
                    memcpy(&hd, rec, 2u);
                    d = __half2float(hd);
                }
                #pragma unroll
                for (uint32_t j = 0; j < 16u; ++j)
                    sc[j] = live ? d * (float)((const int8_t*)rec)[4u + j] : 0.0f;
            } else if (DT == PD_KQ_IQ4XS) {
                float d = 0.0f;
                if (live) {
                    __half hd;
                    memcpy(&hd, rec, 2u);
                    d = __half2float(hd);
                }
                #pragma unroll
                for (uint32_t j = 0; j < 8u; ++j)
                    sc[j] = live ? d * (float)((const int8_t*)rec)[4u + j] : 0.0f;
            } else {
                float d = 0.0f, dmin = 0.0f;
                if (live) {
                    __half hd, hm;
                    memcpy(&hd, rec, 2u);
                    memcpy(&hm, rec + 2u, 2u);
                    d = __half2float(hd);
                    dmin = __half2float(hm);
                }
                const float C = DT == PD_KQ_Q5K ? 16.0f : 8.0f;
                #pragma unroll
                for (uint32_t j = 0; j < 8u; ++j) {
                    const float dj = live ? (DT == PD_KQ_Q40 ? pd_kq40_dj(rec, j)
                                                             : d * (float)rec[4u + j])
                                          : 0.0f;
                    sc[j] = dj;
                    sc[8u + j] = (live && DT != PD_KQ_Q40)
                                     ? C * dj - dmin * (float)rec[12u + j] : 0.0f;
                }
            }
        }
        __syncthreads();  // tile_x fully built before any warp reads it
    };

    // MMA off tile_x (built by build_tilex) and tile_y/tile_s (just loaded by
    // load_act for this h) - identical arithmetic to v1.
    auto mma_h = [&](uint32_t h) {
        {
            const uint32_t k00 = h * 32u;
            int A[2][4][4];
            float dA[2][2][4];
            float muA[2][2][4];
            float sA[2][2][8];
            #pragma unroll
            for (uint32_t n = 0; n < 2u; ++n) {
                const uint32_t r0 = (i0 + n * 16u + g) * PD_KW_XK;
                const uint32_t r8 = (i0 + n * 16u + 8u + g) * PD_KW_XK;
                #pragma unroll
                for (uint32_t kk = 0; kk < 4u; ++kk) {
                    const uint32_t ko = k00 + kk * 8u;
                    A[n][kk][0] = tile_x[r0 + ko + t];
                    A[n][kk][1] = tile_x[r8 + ko + t];
                    A[n][kk][2] = tile_x[r0 + ko + 4u + t];
                    A[n][kk][3] = tile_x[r8 + ko + 4u + t];
                    if (K16) {
                        const uint32_t so = 64u + (k00 >> 2u) + 2u * kk;
                        sA[n][0][2u * kk] = ((const float*)tile_x)[r0 + so];
                        sA[n][0][2u * kk + 1u] = ((const float*)tile_x)[r0 + so + 1u];
                        sA[n][1][2u * kk] = ((const float*)tile_x)[r8 + so];
                        sA[n][1][2u * kk + 1u] = ((const float*)tile_x)[r8 + so + 1u];
                    } else {
                        dA[n][0][kk] = ((const float*)tile_x)[r0 + 64u + (k00 >> 3u) + kk];
                        dA[n][1][kk] = ((const float*)tile_x)[r8 + 64u + (k00 >> 3u) + kk];
                        if (MU) {
                            muA[n][0][kk] = ((const float*)tile_x)[r0 + 72u + (k00 >> 3u) + kk];
                            muA[n][1][kk] = ((const float*)tile_x)[r8 + 72u + (k00 >> 3u) + kk];
                        }
                    }
                }
            }
            #pragma unroll
            for (uint32_t j0 = 0; j0 < 128u; j0 += 16u) {
                const uint32_t jc = j0 + joff;
                #pragma unroll
                for (uint32_t kk = 0; kk < 4u; ++kk) {
                    const uint32_t ko = kk * 8u;
                    const int b0 = tile_y[(jc + g) * PD_MMQ_YK + 4u + ko + t];
                    const int b1 = tile_y[(jc + g) * PD_MMQ_YK + 4u + ko + 4u + t];
                    const float dB0 = ((const float*)tile_y)[(jc + 2u * t) * PD_MMQ_YK + kk];
                    const float dB1 = ((const float*)tile_y)[(jc + 2u * t + 1u) * PD_MMQ_YK + kk];
                    float S0 = 0.0f, S1 = 0.0f;
                    if (MU) {
                        S0 = tile_s[(jc + 2u * t) * 4u + kk];
                        S1 = tile_s[(jc + 2u * t + 1u) * 4u + kk];
                    }
                    #pragma unroll
                    for (uint32_t n = 0; n < 2u; ++n) {
                        if (K16) {
                            int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                            int e0 = 0, e1 = 0, e2 = 0, e3 = 0;
                            asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
                                "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                                : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                                : "r"(A[n][kk][0]), "r"(A[n][kk][1]), "r"(b0));
                            asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
                                "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                                : "+r"(e0), "+r"(e1), "+r"(e2), "+r"(e3)
                                : "r"(A[n][kk][2]), "r"(A[n][kk][3]), "r"(b1));
                            const float sa0 = sA[n][0][2u * kk], sa1 = sA[n][0][2u * kk + 1u];
                            const float sb0_ = sA[n][1][2u * kk], sb1_ = sA[n][1][2u * kk + 1u];
                            acc[(j0 >> 3) + n][0] += dB0 * (sa0 * (float)d0 + sa1 * (float)e0);
                            acc[(j0 >> 3) + n][1] += dB1 * (sa0 * (float)d1 + sa1 * (float)e1);
                            acc[(j0 >> 3) + n][2] += dB0 * (sb0_ * (float)d2 + sb1_ * (float)e2);
                            acc[(j0 >> 3) + n][3] += dB1 * (sb0_ * (float)d3 + sb1_ * (float)e3);
                        } else {
                            int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                            asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                                : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                                : "r"(A[n][kk][0]), "r"(A[n][kk][1]), "r"(A[n][kk][2]),
                                  "r"(A[n][kk][3]), "r"(b0), "r"(b1));
                            acc[(j0 >> 3) + n][0] += dA[n][0][kk] * dB0 * (float)d0;
                            acc[(j0 >> 3) + n][1] += dA[n][0][kk] * dB1 * (float)d1;
                            acc[(j0 >> 3) + n][2] += dA[n][1][kk] * dB0 * (float)d2;
                            acc[(j0 >> 3) + n][3] += dA[n][1][kk] * dB1 * (float)d3;
                            if (MU) {
                                acc[(j0 >> 3) + n][0] += muA[n][0][kk] * S0;
                                acc[(j0 >> 3) + n][1] += muA[n][0][kk] * S1;
                                acc[(j0 >> 3) + n][2] += muA[n][1][kk] * S0;
                                acc[(j0 >> 3) + n][3] += muA[n][1][kk] * S1;
                            }
                        }
                    }
                }
            }
        }
    };

    stage(0u);
    pd_attn_cpa_commit();

    for (uint32_t kt = 0; kt < n_super; ++kt) {
        pd_mma_cpa_waitN<0>();  // this kt's raw bytes fully landed
        __syncthreads();
        build_tilex();  // reads the raw buffer - must finish before it's reused
        __syncthreads();
        if (kt + 1u < n_super) {
            // now safe to overwrite: issued here so it flies in the
            // background for the entire h=0/h=1 MMA compute below, the slow
            // part v1 used to pay DRAM latency on every single kt.
            stage(kt + 1u);
            pd_attn_cpa_commit();
        }
        #pragma unroll
        for (uint32_t h = 0; h < 2u; ++h) {
            load_act(kt, h);
            __syncthreads();
            mma_h(h);
            __syncthreads();  // tile_y/tile_s free before the next h/kt reloads them
        }
    }

    #pragma unroll
    for (uint32_t j0 = 0; j0 < 128u; j0 += 16u) {
        const uint32_t c0 = col_base + j0 + joff + 2u * t;
        #pragma unroll
        for (uint32_t n = 0; n < 2u; ++n) {
            const uint32_t r0 = row_base + i0 + n * 16u + g;
            const uint32_t r8 = r0 + 8u;
            if (r0 < out_dim) {
                if (c0 < batch) y[(size_t)c0 * out_dim + r0] = acc[(j0 >> 3) + n][0];
                if (c0 + 1u < batch) y[(size_t)(c0 + 1u) * out_dim + r0] = acc[(j0 >> 3) + n][1];
            }
            if (r8 < out_dim) {
                if (c0 < batch) y[(size_t)c0 * out_dim + r8] = acc[(j0 >> 3) + n][2];
                if (c0 + 1u < batch) y[(size_t)(c0 + 1u) * out_dim + r8] = acc[(j0 >> 3) + n][3];
            }
        }
    }
#else
    (void)data; (void)scales; (void)yq; (void)xsums; (void)y;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}

PD_EXPORT
int pd_kquant_gemm_w4a8_pipe(const void* data, const void* scales, const void* yq,
                             const void* xsums, void* y, uint32_t in_dim,
                             uint32_t out_dim, uint32_t batch, uint32_t dtype,
                             void* stream) {
    if (out_dim == 0 || batch == 0) return 0;
    if (pd_kq_valid_iq(dtype))
        return pd_kquant_gemm_w4a8_pipe2(data, scales, yq, xsums, y, in_dim, out_dim,
                                         batch, dtype, stream);
    if ((in_dim & 255u) != 0u) return cudaErrorInvalidValue;
    if (!pd_kq_valid(dtype)) return cudaErrorInvalidValue;
    const bool mu = pd_kq_has_mu(dtype);
    if (mu && xsums == nullptr) return cudaErrorInvalidValue;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t ntiles = ((out_dim + 127u) / 128u) * (batch_pad >> 7u);
    #define PD_KWP_LAUNCH(DTV)                                                   \
        do {                                                                    \
            const uint32_t smem = pd_kwp_smem_bytes(DTV);                       \
            static cudaError_t attr = cudaFuncSetAttribute(                     \
                (const void*)pd_kquant_w4a8_pipe_kernel<DTV>,                   \
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);        \
            if (attr != cudaSuccess) return attr;                               \
            pd_kquant_w4a8_pipe_kernel<DTV><<<ntiles, 256, smem,                \
                                              (cudaStream_t)stream>>>(          \
                (const uint8_t*)data, (const uint8_t*)scales,                   \
                (const uint8_t*)yq, (const float*)xsums, (float*)y,            \
                in_dim, out_dim, batch);                                        \
        } while (0)
    switch (dtype) {
        case PD_KQ_Q40: PD_KWP_LAUNCH(PD_KQ_Q40); break;
        case PD_KQ_Q4K: PD_KWP_LAUNCH(PD_KQ_Q4K); break;
        case PD_KQ_Q5K: PD_KWP_LAUNCH(PD_KQ_Q5K); break;
        case PD_KQ_Q6K: PD_KWP_LAUNCH(PD_KQ_Q6K); break;
        default: PD_KWP_LAUNCH(PD_KQ_IQ4XS); break;
    }
    #undef PD_KWP_LAUNCH
    return pd_launch_status();
}


// ---- pipe2: genuine 2-deep double buffering, occupancy=1 (the hard floor) -
// The "hi" tile once here (128-wide K-chunk unpack, __launch_bounds__(256,2))
// hit its register target (REG:128) but profiling showed occupancy never
// rose: sm_120's shared memory per SM (102,400 B) is barely above its own
// single-block opt-in max (101,376 B) - two of that kernel's 61,440 B
// blocks need 122,880 B, which doesn't fit, so Block Limit Shared Mem
// stayed 1 regardless of the register fix. That's a hard ceiling on this
// chip (unlike datacenter Blackwell/Hopper's 227 KB/SM), not a tuning
// problem -. Reverted (kernel/ABI/dispatch/
// tests removed) rather than kept as dead code.
//
// Reading llama.cpp's own mmq.cuh main loop (b10271) found
// their k-quant kernel doesn't use cp.async at all: a plain
// load_tiles -> __syncthreads() -> vec_dot -> __syncthreads() loop, same
// occupancy=1 target as ours. So occupancy=1 is the genuinely-settled
// state on both engines here - the lever that remains is reducing the
// STALL inside that one resident block, not chasing more blocks.
//
// pd_kquant_w4a8_pipe_kernel's own overlap is real but partial: it only
// issues super-block kt+1's load after build_tilex(kt) finishes reading
// the single raw buffer (has to - there's nowhere else for kt+1's bytes to
// land), so build_tilex's own cost sits outside the overlap window: the
// load only hides behind mma_h, not behind build_tilex+mma_h. A genuine
// 2-deep buffer removes that ordering constraint - issue kt+1's load the
// MOMENT kt's own load lands, not when kt's consumer frees the buffer -
// extending the overlap window to the full build+compute phase. That
// wasn't affordable with the FULL-width tile_x (84 int32/row); the
// half-width tile_x from the reverted "hi" kernel above (already bit-exact
// verified) frees enough room for two raw copies AND stays under sm_120's
// 101,376 B single-block cap for all four types:
//   Q4_K 79,872 B (21,504 headroom) | Q5_K 88,064 B (13,312 headroom)
//   Q6_K 94,224 B (7,152 headroom)  | IQ4_XS 77,840 B (23,536 headroom)
// Ping-pong idiom matches this pack's own established pattern
// (gemm/dense_fp4_w8.cuh's PD_BSP_ISSUE_W/Y ring): issue the next chunk's
// loads into the other buffer, commit, wait_group(1) (guarantees the OLDER
// of the two outstanding commit groups - this kt's own data - has landed),
// consume; wait_group(0) drains the last chunk once there's no next one.
// The i-quant family + Q2_K / Q3_K / IQ4_NL ride the same kernel: the raw
// ring is sized per type (quant/iquant.cuh pd_iq_datab / pd_iq_scb), the
// half tile_x is built through the shared 16-weight window unpack
// (pd_iq_win_unpack_t) into 48-int rows - 32 payload words in k order, the 8
// windows' f32 scales at [32, 40) and, for Q2_K, their per-16 min at
// [40, 48) - and consumed on the Q6_K (per-16 scale, m16n8k16) path. Q2_K's
// min term multiplies per-16 activation sums the kernel builds itself from
// the mmq tile (the launcher's per-32 xsums do not serve a per-16 min), 8
// f32 per column in tile_s. The format's codebook is staged in shared after
// tile_s (2-16 KB). Every type stays under sm_120's 101,376 B single-block
// cap: IQ4_NL is the largest at 79,872 B, IQ1_S/M carry the 16 KB grid at
// 72,704 B.
__host__ __device__ __forceinline__ uint32_t pd_kwh2_smem_bytes(uint32_t dt) {
    if (pd_kq_valid_iq(dt)) {
        const uint32_t raw_w2 = 2u * 128u * pd_iq_datab(dt);
        const uint32_t raw_r2 = 2u * 128u * pd_iq_scb(dt);
        const uint32_t tile_x_sz = 128u * 48u * 4u;
        const uint32_t tile_y_sz = 128u * PD_MMQ_YK * 4u;
        const uint32_t tile_s_sz = dt == PD_KQ_Q2K_ID ? 128u * 8u * 4u : 16u;
        return raw_w2 + raw_r2 + tile_x_sz + tile_y_sz + tile_s_sz + pd_iq_tabs_bytes(dt);
    }
    const uint32_t datab = pd_kq_datab(dt);
    const bool mu = pd_kq_has_mu(dt);
    const uint32_t raw_w2 = 2u * 128u * datab;
    const uint32_t raw_r2 = 2u * 128u * PD_KQ_SCB;
    const uint32_t tile_x_sz = 128u * 40u * 4u;       // 32 payload + 8 scale/mu
    const uint32_t tile_y_sz = 128u * PD_MMQ_YK * 4u;
    const uint32_t tile_s_sz = mu ? 128u * 4u * 4u : 16u;
    return raw_w2 + raw_r2 + tile_x_sz + tile_y_sz + tile_s_sz;
}

template <uint32_t DT>
__global__ void __launch_bounds__(256, 1) pd_kquant_w4a8_pipe2_kernel(
        const uint8_t* __restrict__ data, const uint8_t* __restrict__ scales,
        const uint8_t* __restrict__ yq, const float* __restrict__ xsums,
        float* __restrict__ y, uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
#if PD_MMA_OK
    constexpr bool IQ = pd_kq_valid_iq(DT);
    constexpr bool MU = (DT == PD_KQ_Q4K || DT == PD_KQ_Q5K || DT == PD_KQ_Q40);
    constexpr bool MU16 = (DT == PD_KQ_Q2K_ID);  // per-16 min (the window unpack's g)
    constexpr bool K16 = (DT == PD_KQ_Q6K) || IQ;
    constexpr uint32_t DATAB = IQ ? pd_iq_datab(DT)
                             : DT == PD_KQ_Q6K ? PD_KQ6_DATA
                             : DT == PD_KQ_Q5K ? PD_KQ5_DATA : PD_KQ4_DATA;
    constexpr uint32_t SCB = IQ ? pd_iq_scb(DT) : PD_KQ_SCB;
    constexpr uint32_t RAW_W = 128u * DATAB;
    constexpr uint32_t RAW_R = 128u * SCB;
    // half-superblock tile_x row: 32 payload + 8 scale/mu (k-quant) or + 8
    // window scales + 8 window mins (i-quant)
    constexpr uint32_t KWH_XK = IQ ? 48u : 40u;
    constexpr uint32_t TS_F = MU ? 128u * 4u : MU16 ? 128u * 8u : 4u;  // tile_s floats

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5u;
    const uint32_t g = lane >> 2u, t = lane & 3u;
    const uint32_t i0 = (warp >> 1u) * 32u;
    const uint32_t joff = (warp & 1u) * 8u;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t n_super = in_dim >> 8u;
    const uint32_t nct = batch_pad >> 7u;
    const uint32_t tile = blockIdx.x;
    const uint32_t row_base = (tile / nct) * 128u;
    const uint32_t col_base = (tile % nct) * 128u;

    __shared__ int s_lut[16];
    if (DT == PD_KQ_IQ4XS) {
        if (tid < 16u) s_lut[tid] = (int)PD_KQ_IQ4NL[tid];
        __syncthreads();
    }

    extern __shared__ __align__(16) unsigned char pd_kwh2_sh[];
    unsigned char* const raw_w0 = pd_kwh2_sh;
    unsigned char* const raw_w1 = raw_w0 + RAW_W;
    unsigned char* const raw_r0 = raw_w1 + RAW_W;
    unsigned char* const raw_r1 = raw_r0 + RAW_R;
    int* const tile_x = (int*)(raw_r1 + RAW_R);
    int* const tile_y = tile_x + 128 * KWH_XK;
    float* const tile_s = (float*)(tile_y + 128 * PD_MMQ_YK);
    PdIqTabs tabs = {nullptr, nullptr};
    if (IQ) {
        tabs = pd_iq_tabs_stage(DT, (uint8_t*)(tile_s + TS_F));
        __syncthreads();
    }
    float acc[16][4] = {};

    // Same byte layout as pd_kquant_w4a8_pipe_kernel's stage(), parameterized
    // by which of the 2 raw buffers to fill.
    auto stage = [&](uint32_t buf, uint32_t kt) {
        unsigned char* const rw = buf ? raw_w1 : raw_w0;
        unsigned char* const rr = buf ? raw_r1 : raw_r0;
        constexpr uint32_t WCH = DATAB / 16u;
        for (uint32_t i = tid; i < 128u * WCH; i += 256u) {
            const uint32_t row = i / WCH, c = i % WCH;
            const bool ok = (row_base + row) < out_dim;
            pd_mma_cpa16p(rw + row * DATAB + c * 16u,
                          data + ((size_t)(row_base + row) * n_super + kt) * DATAB
                              + c * 16u, ok);
        }
        if (IQ) {
            constexpr uint32_t RCH = SCB / 4u;
            for (uint32_t i = tid; i < 128u * RCH; i += 256u) {
                const uint32_t row = i / RCH, c = i % RCH;
                const bool ok = (row_base + row) < out_dim;
                pd_kq_cpa4p(rr + row * SCB + c * 4u,
                            scales + ((size_t)(row_base + row) * n_super + kt) * SCB
                                + c * 4u, ok);
            }
        } else {
            for (uint32_t i = tid; i < 128u * 3u; i += 256u) {
                const uint32_t row = i / 3u, c = i % 3u;
                const bool ok = (row_base + row) < out_dim;
                pd_kq_cpa8p(rr + row * PD_KQ_SCB + c * 8u,
                            scales + ((size_t)(row_base + row) * n_super + kt) *
                                PD_KQ_SCB + c * 8u, ok);
            }
        }
    };

    // Identical to pd_kquant_w4a8_pipe_kernel's load_act().
    auto load_act = [&](uint32_t kt, uint32_t h) {
        const uint32_t chunk = kt * 2u + h;
        const int* by = (const int*)(yq + ((size_t)chunk * batch_pad + col_base) * 144u);
        #pragma unroll
        for (uint32_t it = 0; it < 18u; ++it) {
            const uint32_t l = it * 256u + tid;
            tile_y[l] = by[l];
        }
        if (MU) {
            #pragma unroll
            for (uint32_t it = 0; it < 2u; ++it) {
                const uint32_t l = it * 256u + tid;
                tile_s[l] = xsums[((size_t)chunk * batch_pad + col_base) * 4u + l];
            }
        }
        if (MU16) {
            // per-16 activation sums x their per-32 scale, straight off the
            // mmq tile: 128 columns x 8 windows, four per thread
            #pragma unroll
            for (uint32_t it = 0; it < 4u; ++it) {
                const uint32_t i = it * 256u + tid;
                const uint32_t col = i >> 3u, w = i & 7u;
                const int* pw = by + col * PD_MMQ_YK + 4u + 4u * w;
                int sm = __dp4a(pw[0], 0x01010101, 0);
                sm = __dp4a(pw[1], 0x01010101, sm);
                sm = __dp4a(pw[2], 0x01010101, sm);
                sm = __dp4a(pw[3], 0x01010101, sm);
                const float scl = ((const float*)by)[col * PD_MMQ_YK + (w >> 1u)];
                tile_s[col * 8u + w] = scl * (float)sm;
            }
        }
    };

    // Build the HALF-width tile_x for super `kt`'s `half` (0 or 1) out of
    // raw buffer `buf` - same per-DT unpack math as v1/pipe's build_tilex,
    // restricted to ci in [half*4, half*4+4) and rebased so the write lands
    // in [0,32) either way (obase_rebased = obase - half*32; verified true
    // for all four types by construction of their gq/n/ci formulas). Same
    // scale-record read, restricted to j in [half*4,half*4+4)
    // (Q4_K/Q5_K/IQ4_XS) or [half*8,half*8+8) (Q6_K), rebased to
    // sc[0..4) / sc[0..8).
    auto build_half = [&](uint32_t half, uint32_t buf) {
        unsigned char* const rw = buf ? raw_w1 : raw_w0;
        unsigned char* const rr = buf ? raw_r1 : raw_r0;
        if (IQ) {
            // 128 rows x 8 windows of 16 through the shared window unpack:
            // the 4 packed-s8 words land in k order at [wl*4, wl*4+4), the
            // window's f32 scale at [32 + wl], Q2_K's per-16 min at [40 + wl].
            // Dead rows (zero-filled raw) write zeros.
            #pragma unroll
            for (uint32_t it = 0; it < 4u; ++it) {
                const uint32_t i = it * 256u + tid;
                const uint32_t row = i >> 3u, wl = i & 7u;
                const bool live = (row_base + row) < out_dim;
                int wq[4] = {0, 0, 0, 0};
                float f = 0.0f, gm = 0.0f;
                if (live)
                    pd_iq_win_unpack_t(DT, rw + row * DATAB, rr + row * SCB, half * 8u + wl,
                                       wq, &f, &gm, tabs);
                int* const xr = tile_x + row * KWH_XK;
                *reinterpret_cast<int4*>(xr + wl * 4u) = make_int4(wq[0], wq[1], wq[2], wq[3]);
                reinterpret_cast<float*>(xr)[32u + wl] = f;
                if (MU16) reinterpret_cast<float*>(xr)[40u + wl] = gm;
            }
            __syncthreads();
            return;
        }
        if (IQ) return;  // (the k-quant build below is not instantiated for IQ)
        #pragma unroll
        for (uint32_t it = 0; it < 2u; ++it) {  // 128 rows * 4 ci = 512 = 2*256
            const uint32_t i = it * 256u + tid;
            const uint32_t row = i >> 2u, ci_local = i & 3u;
            const uint32_t ci = half * 4u + ci_local;
            const bool live = (row_base + row) < out_dim;
            const uint8_t* sb = rw + row * DATAB;
            int out[8] = {};
            uint32_t obase, hioff;
            if (DT == PD_KQ_Q6K) {
                const uint32_t n = ci >> 2u, a = (ci >> 1u) & 1u, h_bit = ci & 1u;
                obase = n * 32u + a * 8u + h_bit * 4u - half * 32u;
                hioff = 16u;
                if (live) {
                    const uint4 qv = *(const uint4*)(sb + n * 64u + a * 32u + h_bit * 16u);
                    const uint4 hv = *(const uint4*)(sb + 128u + n * 32u + h_bit * 16u);
                    const uint32_t qw[4] = {qv.x, qv.y, qv.z, qv.w};
                    const uint32_t hw[4] = {hv.x, hv.y, hv.z, hv.w};
                    const uint32_t sh1 = 2u * a, sh2 = 2u * a + 4u;
                    #pragma unroll
                    for (uint32_t wv = 0; wv < 4u; ++wv) {
                        const uint32_t lo = (qw[wv] & 0x0F0F0F0Fu)
                            | (((hw[wv] >> sh1) & 0x03030303u) << 4u);
                        const uint32_t hi = ((qw[wv] >> 4u) & 0x0F0F0F0Fu)
                            | (((hw[wv] >> sh2) & 0x03030303u) << 4u);
                        out[wv] = (int)__vsub4(lo, 0x20202020u);
                        out[4u + wv] = (int)__vsub4(hi, 0x20202020u);
                    }
                }
            } else if (DT == PD_KQ_IQ4XS) {
                obase = ci * 8u - half * 32u;
                hioff = 4u;
                if (live) {
                    const uint4 qv = *(const uint4*)(sb + ci * 16u);
                    const uint32_t qw[4] = {qv.x, qv.y, qv.z, qv.w};
                    #pragma unroll
                    for (uint32_t wv = 0; wv < 4u; ++wv) {
                        uint32_t lo = 0u, hi = 0u;
                        #pragma unroll
                        for (uint32_t b = 0; b < 4u; ++b) {
                            const uint32_t qb = (qw[wv] >> (8u * b)) & 0xFFu;
                            lo |= ((uint32_t)(uint8_t)s_lut[qb & 0xFu]) << (8u * b);
                            hi |= ((uint32_t)(uint8_t)s_lut[qb >> 4u]) << (8u * b);
                        }
                        out[wv] = (int)lo;
                        out[4u + wv] = (int)hi;
                    }
                }
            } else {  // Q4_K / Q5_K
                const uint32_t gq = ci >> 1u, h_bit = ci & 1u;
                obase = gq * 16u + h_bit * 4u - half * 32u;
                hioff = 8u;
                if (live) {
                    const uint4 qv = *(const uint4*)(sb + gq * 32u + h_bit * 16u);
                    uint4 hv = make_uint4(0u, 0u, 0u, 0u);
                    if (DT == PD_KQ_Q5K) hv = *(const uint4*)(sb + 128u + h_bit * 16u);
                    const uint32_t qw[4] = {qv.x, qv.y, qv.z, qv.w};
                    const uint32_t hw[4] = {hv.x, hv.y, hv.z, hv.w};
                    const uint32_t j1 = 2u * gq, j2 = 2u * gq + 1u;
                    const uint32_t C = DT == PD_KQ_Q5K ? 0x10101010u : 0x08080808u;
                    #pragma unroll
                    for (uint32_t wv = 0; wv < 4u; ++wv) {
                        uint32_t lo = qw[wv] & 0x0F0F0F0Fu;
                        uint32_t hi = (qw[wv] >> 4u) & 0x0F0F0F0Fu;
                        if (DT == PD_KQ_Q5K) {
                            lo |= ((hw[wv] >> j1) & 0x01010101u) << 4u;
                            hi |= ((hw[wv] >> j2) & 0x01010101u) << 4u;
                        }
                        out[wv] = (int)__vsub4(lo, C);
                        out[4u + wv] = (int)__vsub4(hi, C);
                    }
                }
            }
            int* dst = tile_x + row * KWH_XK + obase;
            #pragma unroll
            for (uint32_t wv = 0; wv < 4u; ++wv) {
                dst[wv] = out[wv];
                dst[hioff + wv] = out[4u + wv];
            }
        }
        if (tid < 128u) {
            const uint32_t row = tid;
            float* sc = (float*)(tile_x + row * KWH_XK + 32u);
            const bool live = (row_base + row) < out_dim;
            const uint8_t* rec = rr + row * PD_KQ_SCB;
            if (DT == PD_KQ_Q6K) {
                float d = 0.0f;
                if (live) {
                    __half hd;
                    memcpy(&hd, rec, 2u);
                    d = __half2float(hd);
                }
                #pragma unroll
                for (uint32_t jl = 0; jl < 8u; ++jl) {
                    const uint32_t j = half * 8u + jl;
                    sc[jl] = live ? d * (float)((const int8_t*)rec)[4u + j] : 0.0f;
                }
            } else if (DT == PD_KQ_IQ4XS) {
                float d = 0.0f;
                if (live) {
                    __half hd;
                    memcpy(&hd, rec, 2u);
                    d = __half2float(hd);
                }
                #pragma unroll
                for (uint32_t jl = 0; jl < 4u; ++jl) {
                    const uint32_t j = half * 4u + jl;
                    sc[jl] = live ? d * (float)((const int8_t*)rec)[4u + j] : 0.0f;
                }
            } else {
                float d = 0.0f, dmin = 0.0f;
                if (live) {
                    __half hd, hm;
                    memcpy(&hd, rec, 2u);
                    memcpy(&hm, rec + 2u, 2u);
                    d = __half2float(hd);
                    dmin = __half2float(hm);
                }
                const float C = DT == PD_KQ_Q5K ? 16.0f : 8.0f;
                #pragma unroll
                for (uint32_t jl = 0; jl < 4u; ++jl) {
                    const uint32_t j = half * 4u + jl;
                    const float dj = live ? (DT == PD_KQ_Q40 ? pd_kq40_dj(rec, j)
                                                             : d * (float)rec[4u + j])
                                          : 0.0f;
                    sc[jl] = dj;
                    sc[4u + jl] = (live && DT != PD_KQ_Q40)
                                      ? C * dj - dmin * (float)rec[12u + j] : 0.0f;
                }
            }
        }
        __syncthreads();
    };

    // MMA off the just-built half tile_x and this h's tile_y/tile_s -
    // identical arithmetic to v1/pipe's mma_h, with k00 always 0 (the half
    // tile_x is already rebased) so `ko = kk*8` directly, and the scale
    // offsets fixed at 32 (Q4_K/Q5_K/IQ4_XS dj) / 36 (mu) / 32 (Q6_K sA)
    // instead of `64 + (k00>>3)`-style h-dependent offsets.
    auto mma_half = [&]() {
        int A[2][4][4];
        float dA[2][2][4];
        float muA[2][2][4];
        float sA[2][2][8];
        float mA16[2][2][8];
        #pragma unroll
        for (uint32_t n = 0; n < 2u; ++n) {
            const uint32_t r0 = (i0 + n * 16u + g) * KWH_XK;
            const uint32_t r8 = (i0 + n * 16u + 8u + g) * KWH_XK;
            #pragma unroll
            for (uint32_t kk = 0; kk < 4u; ++kk) {
                const uint32_t ko = kk * 8u;
                A[n][kk][0] = tile_x[r0 + ko + t];
                A[n][kk][1] = tile_x[r8 + ko + t];
                A[n][kk][2] = tile_x[r0 + ko + 4u + t];
                A[n][kk][3] = tile_x[r8 + ko + 4u + t];
                if (K16) {
                    const uint32_t so = 32u + 2u * kk;
                    sA[n][0][2u * kk] = ((const float*)tile_x)[r0 + so];
                    sA[n][0][2u * kk + 1u] = ((const float*)tile_x)[r0 + so + 1u];
                    sA[n][1][2u * kk] = ((const float*)tile_x)[r8 + so];
                    sA[n][1][2u * kk + 1u] = ((const float*)tile_x)[r8 + so + 1u];
                    if (MU16) {
                        mA16[n][0][2u * kk] = ((const float*)tile_x)[r0 + so + 8u];
                        mA16[n][0][2u * kk + 1u] = ((const float*)tile_x)[r0 + so + 9u];
                        mA16[n][1][2u * kk] = ((const float*)tile_x)[r8 + so + 8u];
                        mA16[n][1][2u * kk + 1u] = ((const float*)tile_x)[r8 + so + 9u];
                    }
                } else {
                    dA[n][0][kk] = ((const float*)tile_x)[r0 + 32u + kk];
                    dA[n][1][kk] = ((const float*)tile_x)[r8 + 32u + kk];
                    if (MU) {
                        muA[n][0][kk] = ((const float*)tile_x)[r0 + 36u + kk];
                        muA[n][1][kk] = ((const float*)tile_x)[r8 + 36u + kk];
                    }
                }
            }
        }
        #pragma unroll
        for (uint32_t j0 = 0; j0 < 128u; j0 += 16u) {
            const uint32_t jc = j0 + joff;
            #pragma unroll
            for (uint32_t kk = 0; kk < 4u; ++kk) {
                const uint32_t ko = kk * 8u;
                const int b0 = tile_y[(jc + g) * PD_MMQ_YK + 4u + ko + t];
                const int b1 = tile_y[(jc + g) * PD_MMQ_YK + 4u + ko + 4u + t];
                const float dB0 = ((const float*)tile_y)[(jc + 2u * t) * PD_MMQ_YK + kk];
                const float dB1 = ((const float*)tile_y)[(jc + 2u * t + 1u) * PD_MMQ_YK + kk];
                float S0 = 0.0f, S1 = 0.0f;
                if (MU) {
                    S0 = tile_s[(jc + 2u * t) * 4u + kk];
                    S1 = tile_s[(jc + 2u * t + 1u) * 4u + kk];
                }
                float S0a = 0.0f, S0b = 0.0f, S1a = 0.0f, S1b = 0.0f;
                if (MU16) {
                    S0a = tile_s[(jc + 2u * t) * 8u + 2u * kk];
                    S0b = tile_s[(jc + 2u * t) * 8u + 2u * kk + 1u];
                    S1a = tile_s[(jc + 2u * t + 1u) * 8u + 2u * kk];
                    S1b = tile_s[(jc + 2u * t + 1u) * 8u + 2u * kk + 1u];
                }
                #pragma unroll
                for (uint32_t n = 0; n < 2u; ++n) {
                    if (K16) {
                        int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                        int e0 = 0, e1 = 0, e2 = 0, e3 = 0;
                        asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
                            "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                            : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                            : "r"(A[n][kk][0]), "r"(A[n][kk][1]), "r"(b0));
                        asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
                            "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                            : "+r"(e0), "+r"(e1), "+r"(e2), "+r"(e3)
                            : "r"(A[n][kk][2]), "r"(A[n][kk][3]), "r"(b1));
                        const float sa0 = sA[n][0][2u * kk], sa1 = sA[n][0][2u * kk + 1u];
                        const float sb0_ = sA[n][1][2u * kk], sb1_ = sA[n][1][2u * kk + 1u];
                        acc[(j0 >> 3) + n][0] += dB0 * (sa0 * (float)d0 + sa1 * (float)e0);
                        acc[(j0 >> 3) + n][1] += dB1 * (sa0 * (float)d1 + sa1 * (float)e1);
                        acc[(j0 >> 3) + n][2] += dB0 * (sb0_ * (float)d2 + sb1_ * (float)e2);
                        acc[(j0 >> 3) + n][3] += dB1 * (sb0_ * (float)d3 + sb1_ * (float)e3);
                        if (MU16) {
                            const float ma0 = mA16[n][0][2u * kk], mb0 = mA16[n][0][2u * kk + 1u];
                            const float ma8 = mA16[n][1][2u * kk], mb8 = mA16[n][1][2u * kk + 1u];
                            acc[(j0 >> 3) + n][0] += ma0 * S0a + mb0 * S0b;
                            acc[(j0 >> 3) + n][1] += ma0 * S1a + mb0 * S1b;
                            acc[(j0 >> 3) + n][2] += ma8 * S0a + mb8 * S0b;
                            acc[(j0 >> 3) + n][3] += ma8 * S1a + mb8 * S1b;
                        }
                    } else {
                        int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                        asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                            : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                            : "r"(A[n][kk][0]), "r"(A[n][kk][1]), "r"(A[n][kk][2]),
                              "r"(A[n][kk][3]), "r"(b0), "r"(b1));
                        acc[(j0 >> 3) + n][0] += dA[n][0][kk] * dB0 * (float)d0;
                        acc[(j0 >> 3) + n][1] += dA[n][0][kk] * dB1 * (float)d1;
                        acc[(j0 >> 3) + n][2] += dA[n][1][kk] * dB0 * (float)d2;
                        acc[(j0 >> 3) + n][3] += dA[n][1][kk] * dB1 * (float)d3;
                        if (MU) {
                            acc[(j0 >> 3) + n][0] += muA[n][0][kk] * S0;
                            acc[(j0 >> 3) + n][1] += muA[n][0][kk] * S1;
                            acc[(j0 >> 3) + n][2] += muA[n][1][kk] * S0;
                            acc[(j0 >> 3) + n][3] += muA[n][1][kk] * S1;
                        }
                    }
                }
            }
        }
    };

    stage(0u, 0u);
    pd_attn_cpa_commit();

    for (uint32_t kt = 0; kt < n_super; ++kt) {
        const uint32_t cur = kt & 1u;
        if (kt + 1u < n_super) {
            // issued the instant kt's own load lands (not after kt's build
            // frees a shared buffer) - overlaps the entire build_half+mma_half
            // phase of kt, not just mma_half like the single-buffer pipe
            // kernel above.
            stage(cur ^ 1u, kt + 1u);
            pd_attn_cpa_commit();
            pd_mma_cpa_waitN<1>();  // kt's own data (buf `cur`) landed; kt+1 still in flight
        } else {
            pd_mma_cpa_waitN<0>();  // drain: no next chunk to keep in flight
        }
        __syncthreads();
        #pragma unroll
        for (uint32_t half = 0; half < 2u; ++half) {
            load_act(kt, half);
            build_half(half, cur);  // has its own trailing __syncthreads()
            mma_half();
            __syncthreads();  // tile_x/tile_y/tile_s free before the next half/kt reloads them
        }
    }

    #pragma unroll
    for (uint32_t j0 = 0; j0 < 128u; j0 += 16u) {
        const uint32_t c0 = col_base + j0 + joff + 2u * t;
        #pragma unroll
        for (uint32_t n = 0; n < 2u; ++n) {
            const uint32_t r0 = row_base + i0 + n * 16u + g;
            const uint32_t r8 = r0 + 8u;
            if (r0 < out_dim) {
                if (c0 < batch) y[(size_t)c0 * out_dim + r0] = acc[(j0 >> 3) + n][0];
                if (c0 + 1u < batch) y[(size_t)(c0 + 1u) * out_dim + r0] = acc[(j0 >> 3) + n][1];
            }
            if (r8 < out_dim) {
                if (c0 < batch) y[(size_t)c0 * out_dim + r8] = acc[(j0 >> 3) + n][2];
                if (c0 + 1u < batch) y[(size_t)(c0 + 1u) * out_dim + r8] = acc[(j0 >> 3) + n][3];
            }
        }
    }
#else
    (void)data; (void)scales; (void)yq; (void)xsums; (void)y;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}

PD_EXPORT
int pd_kquant_gemm_w4a8_pipe2(const void* data, const void* scales, const void* yq,
                              const void* xsums, void* y, uint32_t in_dim,
                              uint32_t out_dim, uint32_t batch, uint32_t dtype,
                              void* stream) {
    if (out_dim == 0 || batch == 0) return 0;
    if ((in_dim & 255u) != 0u) return cudaErrorInvalidValue;
    if (!pd_kq_valid(dtype) && !pd_kq_valid_iq(dtype)) return cudaErrorInvalidValue;
    // Q2_K's per-16 min term is built in-kernel off the mmq tile; the
    // per-32 xsums are neither needed nor read for the i-quant family
    const bool mu = pd_kq_has_mu(dtype) && !pd_kq_valid_iq(dtype);
    if (mu && xsums == nullptr) return cudaErrorInvalidValue;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t ntiles = ((out_dim + 127u) / 128u) * (batch_pad >> 7u);
    #define PD_KWH2_LAUNCH(DTV)                                                 \
        do {                                                                    \
            const uint32_t smem = pd_kwh2_smem_bytes(DTV);                      \
            static cudaError_t attr = cudaFuncSetAttribute(                     \
                (const void*)pd_kquant_w4a8_pipe2_kernel<DTV>,                  \
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);        \
            if (attr != cudaSuccess) return attr;                               \
            pd_kquant_w4a8_pipe2_kernel<DTV><<<ntiles, 256, smem,               \
                                              (cudaStream_t)stream>>>(          \
                (const uint8_t*)data, (const uint8_t*)scales,                   \
                (const uint8_t*)yq, (const float*)xsums, (float*)y,            \
                in_dim, out_dim, batch);                                        \
        } while (0)
    switch (dtype) {
        case PD_KQ_Q40: PD_KWH2_LAUNCH(PD_KQ_Q40); break;
        case PD_KQ_Q4K: PD_KWH2_LAUNCH(PD_KQ_Q4K); break;
        case PD_KQ_Q5K: PD_KWH2_LAUNCH(PD_KQ_Q5K); break;
        case PD_KQ_Q6K: PD_KWH2_LAUNCH(PD_KQ_Q6K); break;
        case PD_KQ_IQ2XXS: PD_KWH2_LAUNCH(PD_KQ_IQ2XXS); break;
        case PD_KQ_IQ2XS: PD_KWH2_LAUNCH(PD_KQ_IQ2XS); break;
        case PD_KQ_IQ2S: PD_KWH2_LAUNCH(PD_KQ_IQ2S); break;
        case PD_KQ_IQ3XXS: PD_KWH2_LAUNCH(PD_KQ_IQ3XXS); break;
        case PD_KQ_IQ3S: PD_KWH2_LAUNCH(PD_KQ_IQ3S); break;
        case PD_KQ_IQ1S: PD_KWH2_LAUNCH(PD_KQ_IQ1S); break;
        case PD_KQ_IQ1M: PD_KWH2_LAUNCH(PD_KQ_IQ1M); break;
        case PD_KQ_IQ4NL_ID: PD_KWH2_LAUNCH(PD_KQ_IQ4NL_ID); break;
        case PD_KQ_Q2K_ID: PD_KWH2_LAUNCH(PD_KQ_Q2K_ID); break;
        case PD_KQ_Q3K_ID: PD_KWH2_LAUNCH(PD_KQ_Q3K_ID); break;
        default: PD_KWH2_LAUNCH(PD_KQ_IQ4XS); break;
    }
    #undef PD_KWH2_LAUNCH
    return pd_launch_status();
}

// Capability marker (slot 580): the >64-row W4A8 tile GEMM serves the
// i-quant family + Q2_K / Q3_K / IQ4_NL - the engine's prefill can keep a
// dense i-quant plane on the tile rung instead of the per-token dp4a walk.
PD_EXPORT int pd_kquant_iq_tile(void) { return 0; }
