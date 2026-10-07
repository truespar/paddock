// quant/kquant_w4a8_ring.cuh - the Q4_K prefill tile with its ACTIVATION
// tile on the cp.async ring (slot 750, pd_kquant_gemm_w4a8_pipe3).
//
// pipe2 (quant/kquant_w4a8.cuh) double-buffers the raw weight strips, but
// loads each half-super's activation tile (tile_y + tile_s) with synchronous
// global loads in front of the half's barrier. At this tile's one CTA an SM
// (248 registers, ~80 KB of shared) the two warps a scheduler sit on that
// latency every half: on GB10 ncu put lg_throttle and long_scoreboard at the
// top of pipe2's stalls, issue at 35% and the int8 tensor pipe at 24.5%.
// Here the activations get two buffers and ride cp.async one half-super
// ahead - step s = (kt, half) issues step s+1's activations (and, at half 0,
// super kt+1's weights) before waiting on its own. The unpack and the MMA
// epilogue are pipe2's verbatim, so every byte of y matches pipe2 (and v1).
// Q4_K only: two activation buffers on pipe2's Q4_K layout come to 100,352 B
// of the 101,376 B single-block cap; Q5_K / Q6_K's wider raw rings do not
// fit, so the export hands every other type to pipe2.
// GB10, Kolibri 1's planes at 2048 rows (bench/kq_w4a8_tile_gb10_bench.cu):
// [2560 -> 6144] 60.0 -> 68.7 TOPS, [6144 -> 2560] 62.5 -> 73.1, [2560 -> 512]
// 45.0 -> 49.4 (the die's int8 mma.sync ceiling measures 247).

constexpr uint32_t PD_KWP3_RAW_W = 128u * PD_KQ4_DATA, PD_KWP3_RAW_R = 128u * PD_KQ_SCB;
constexpr uint32_t PD_KWP3_TX = 128u * 40u * 4u, PD_KWP3_TY = 128u * PD_MMQ_YK * 4u;
constexpr uint32_t PD_KWP3_TS = 128u * 4u * 4u;
constexpr uint32_t PD_KWP3_SMEM = 2u * PD_KWP3_RAW_W + 2u * PD_KWP3_RAW_R + PD_KWP3_TX
                                + 2u * PD_KWP3_TY + 2u * PD_KWP3_TS;

__global__ void __launch_bounds__(256, 1) pd_kquant_w4a8_pipe3_kernel(
        const uint8_t* __restrict__ data, const uint8_t* __restrict__ scales,
        const uint8_t* __restrict__ yq, const float* __restrict__ xsums,
        float* __restrict__ y, uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
#if PD_MMA_OK
    constexpr uint32_t DATAB = PD_KQ4_DATA, KWH_XK = 40u;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5u;
    const uint32_t g = lane >> 2u, t = lane & 3u;
    const uint32_t i0 = (warp >> 1u) * 32u;
    const uint32_t joff = (warp & 1u) * 8u;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t n_super = in_dim >> 8u;
    const uint32_t nct = batch_pad >> 7u;
    const uint32_t row_base = (blockIdx.x / nct) * 128u;
    const uint32_t col_base = (blockIdx.x % nct) * 128u;

    extern __shared__ __align__(16) unsigned char pd_kwp3_sh[];
    unsigned char* const raw_w0 = pd_kwp3_sh;
    unsigned char* const raw_w1 = raw_w0 + PD_KWP3_RAW_W;
    unsigned char* const raw_r0 = raw_w1 + PD_KWP3_RAW_W;
    unsigned char* const raw_r1 = raw_r0 + PD_KWP3_RAW_R;
    int* const tile_x = (int*)(raw_r1 + PD_KWP3_RAW_R);
    unsigned char* const ty0 = (unsigned char*)(tile_x + 128 * KWH_XK);
    unsigned char* const ty1 = ty0 + PD_KWP3_TY;
    unsigned char* const ts0 = ty1 + PD_KWP3_TY;
    unsigned char* const ts1 = ts0 + PD_KWP3_TS;
    float acc[16][4] = {};

    auto stage_w = [&](uint32_t buf, uint32_t kt) {
        unsigned char* const rw = buf ? raw_w1 : raw_w0;
        unsigned char* const rr = buf ? raw_r1 : raw_r0;
        constexpr uint32_t WCH = DATAB / 16u;
        for (uint32_t i = tid; i < 128u * WCH; i += 256u) {
            const uint32_t row = i / WCH, c = i % WCH;
            const bool ok = (row_base + row) < out_dim;
            pd_mma_cpa16p(rw + row * DATAB + c * 16u,
                          data + ((size_t)(row_base + row) * n_super + kt) * DATAB + c * 16u, ok);
        }
        for (uint32_t i = tid; i < 128u * 3u; i += 256u) {
            const uint32_t row = i / 3u, c = i % 3u;
            const bool ok = (row_base + row) < out_dim;
            pd_kq_cpa8p(rr + row * PD_KQ_SCB + c * 8u,
                        scales + ((size_t)(row_base + row) * n_super + kt) * PD_KQ_SCB + c * 8u, ok);
        }
    };
    // one half-super's activations: 128 cols x 144 B contiguous + 128 x 16 B sums
    auto stage_act = [&](uint32_t buf, uint32_t step) {
        unsigned char* const ty = buf ? ty1 : ty0;
        unsigned char* const ts = buf ? ts1 : ts0;
        const unsigned char* by = yq + ((size_t)step * batch_pad + col_base) * 144u;
        for (uint32_t i = tid; i < PD_KWP3_TY / 16u; i += 256u)
            pd_mma_cpa16p(ty + i * 16u, by + i * 16u, true);
        const unsigned char* bs =
            (const unsigned char*)(xsums + ((size_t)step * batch_pad + col_base) * 4u);
        for (uint32_t i = tid; i < PD_KWP3_TS / 16u; i += 256u)
            pd_mma_cpa16p(ts + i * 16u, bs + i * 16u, true);
    };

    auto build_half = [&](uint32_t half, uint32_t buf) {
        unsigned char* const rw = buf ? raw_w1 : raw_w0;
        unsigned char* const rr = buf ? raw_r1 : raw_r0;
        #pragma unroll
        for (uint32_t it = 0; it < 2u; ++it) {
            const uint32_t i = it * 256u + tid;
            const uint32_t row = i >> 2u, ci_local = i & 3u;
            const uint32_t ci = half * 4u + ci_local;
            const bool live = (row_base + row) < out_dim;
            const uint8_t* sb = rw + row * DATAB;
            int out[8] = {};
            const uint32_t gq = ci >> 1u, h_bit = ci & 1u;
            const uint32_t obase = gq * 16u + h_bit * 4u - half * 32u, hioff = 8u;
            if (live) {
                const uint4 qv = *(const uint4*)(sb + gq * 32u + h_bit * 16u);
                const uint32_t qw[4] = {qv.x, qv.y, qv.z, qv.w};
                #pragma unroll
                for (uint32_t wv = 0; wv < 4u; ++wv) {
                    const uint32_t lo = qw[wv] & 0x0F0F0F0Fu;
                    const uint32_t hi = (qw[wv] >> 4u) & 0x0F0F0F0Fu;
                    out[wv] = (int)__vsub4(lo, 0x08080808u);
                    out[4u + wv] = (int)__vsub4(hi, 0x08080808u);
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
            float d = 0.0f, dmin = 0.0f;
            if (live) {
                __half hd, hm;
                memcpy(&hd, rec, 2u);
                memcpy(&hm, rec + 2u, 2u);
                d = __half2float(hd);
                dmin = __half2float(hm);
            }
            #pragma unroll
            for (uint32_t jl = 0; jl < 4u; ++jl) {
                const uint32_t j = half * 4u + jl;
                const float dj = live ? d * (float)rec[4u + j] : 0.0f;
                sc[jl] = dj;
                sc[4u + jl] = live ? 8.0f * dj - dmin * (float)rec[12u + j] : 0.0f;
            }
        }
        __syncthreads();
    };

    auto mma_half = [&](const int* tile_y, const float* tile_s) {
        int A[2][4][4];
        float dA[2][2][4], muA[2][2][4];
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
                dA[n][0][kk] = ((const float*)tile_x)[r0 + 32u + kk];
                dA[n][1][kk] = ((const float*)tile_x)[r8 + 32u + kk];
                muA[n][0][kk] = ((const float*)tile_x)[r0 + 36u + kk];
                muA[n][1][kk] = ((const float*)tile_x)[r8 + 36u + kk];
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
                const float S0 = tile_s[(jc + 2u * t) * 4u + kk];
                const float S1 = tile_s[(jc + 2u * t + 1u) * 4u + kk];
                #pragma unroll
                for (uint32_t n = 0; n < 2u; ++n) {
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
                    acc[(j0 >> 3) + n][0] += muA[n][0][kk] * S0;
                    acc[(j0 >> 3) + n][1] += muA[n][0][kk] * S1;
                    acc[(j0 >> 3) + n][2] += muA[n][1][kk] * S0;
                    acc[(j0 >> 3) + n][3] += muA[n][1][kk] * S1;
                }
            }
        }
    };

    const uint32_t nsteps = 2u * n_super;
    stage_w(0u, 0u);
    stage_act(0u, 0u);
    pd_attn_cpa_commit();
    for (uint32_t s = 0; s < nsteps; ++s) {
        const uint32_t kt = s >> 1u, half = s & 1u, cur = kt & 1u;
        if (s + 1u < nsteps) {
            stage_act((s + 1u) & 1u, s + 1u);
            if (half == 0u && kt + 1u < n_super) stage_w(cur ^ 1u, kt + 1u);
            pd_attn_cpa_commit();
            pd_mma_cpa_waitN<1>();  // step s's group landed; s+1's in flight
        } else {
            pd_mma_cpa_waitN<0>();
        }
        __syncthreads();
        build_half(half, cur);  // trailing __syncthreads inside
        mma_half((const int*)((s & 1u) ? ty1 : ty0), (const float*)((s & 1u) ? ts1 : ts0));
        __syncthreads();  // tile_x and this step's activation buffer free
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
int pd_kquant_gemm_w4a8_pipe3(const void* data, const void* scales, const void* yq,
                              const void* xsums, void* y, uint32_t in_dim,
                              uint32_t out_dim, uint32_t batch, uint32_t dtype,
                              void* stream) {
    if (out_dim == 0 || batch == 0) return 0;
    static const cudaError_t attr = cudaFuncSetAttribute(
        (const void*)pd_kquant_w4a8_pipe3_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)PD_KWP3_SMEM);
    if (dtype != PD_KQ_Q4K || attr != cudaSuccess || (in_dim & 255u) != 0u || xsums == nullptr)
        return pd_kquant_gemm_w4a8_pipe2(data, scales, yq, xsums, y, in_dim, out_dim, batch,
                                         dtype, stream);
    const uint32_t ntiles = ((out_dim + 127u) / 128u) * (((batch + 127u) & ~127u) >> 7u);
    pd_kquant_w4a8_pipe3_kernel<<<ntiles, 256, PD_KWP3_SMEM, (cudaStream_t)stream>>>(
        (const uint8_t*)data, (const uint8_t*)scales, (const uint8_t*)yq, (const float*)xsums,
        (float*)y, in_dim, out_dim, batch);
    return pd_launch_status();
}
