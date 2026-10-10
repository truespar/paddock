// quant/kquant.cuh (formerly 18_kquant.cuh) - GGUF k-quant (Q4_K / Q5_K / Q6_K / IQ4_XS) weight support:
// full-tensor dequant, load-time repack to kernel-friendly streams, exact fused
// decode GEMV, and an embedding row-gather. Stage 1 of the W4A8 route (
// quantization strategy): weights stay 4-6.6 bpw RESIDENT; batch/prefill GEMMs
// ride the dequant+f32-GEMM interim (pd_kquant_dequant_rp below) until the
// stage-2 int8-MMA (QServe-class) kernels land.
//
// All kernels in-house. The bit layouts and dequant semantics below are the GGUF
// k-quant FORMAT SPEC, verified against the pinned llama.cpp b9895 source
// (ggml-common.h / ggml-quants.c) per the Track B study-only rule:
//   Q4_K: super-block 256 = { f16 d, f16 dmin, u8 scales[12] (packed 6-bit
//         sc/min x8 sub-blocks of 32), u8 qs[128] } = 144 B. value = d*sc*q - dmin*m,
//         q = nibble; per 64-weight group, bytes q[0..32) hold weights g*64+l (low
//         nibble) and g*64+32+l (high nibble).
//   Q5_K: Q4_K + u8 qh[32] (5th bit): q = nibble + (qh[l]>>j & 1 ? 16 : 0), where
//         j is the sub-block index (u1/u2 shifted by 2 per 64-group in the ref).
//         Block = 176 B. NOTE the source field ORDER: { d, dmin, scales[12],
//         qh[32], qs[128] } - qh PRECEDES qs (unlike our repacked stream, which
//         keeps qs first). Getting this backwards decodes structured garbage
//         that still round-trips through any self-consistent test - the
//         UD bring-up hit exactly that.
//   Q6_K: super-block 256 = { u8 ql[128], u8 qh[64], i8 scales[16], f16 d } = 210 B.
//         16 groups of 16; per 128-half n and l in [0,32): rows 0..3 are
//         q = (ql[n*64 + (row&1)*32 + l] nibble(row<2 ? low : high))
//             | ((qh[n*32 + l] >> 2*row) & 3) << 4, minus 32;
//         value = d * scales[n*8 + row*2 + l/16] * q.
//   IQ4_XS: super-block 256 = { f16 d, u16 scales_h, u8 scales_l[4], u8 qs[128] }
//         = 136 B (4.25 bpw). Per 32-weight sub-block ib: 6-bit scale
//         ls = scales_l nibble | (scales_h >> 2*ib & 3) << 4, value =
//         d*(ls-32) * KVALUES[q] with the shared nonlinear 16-entry codebook;
//         qs byte j of the sub-block: low nibble -> weight j, high -> 16+j.
//
// dtype tags on every launcher are the GGUF raw type ids (Q4_K=12, Q5_K=13,
// Q6_K=14, IQ4_XS=23) so the engine passes GgmlType through without a mapping
// table.
//
// REPACKED layouts (ours, chosen for coalesced GEMV reads - same philosophy as
// pd_q8_0_repack's data/scale split):
//   Q4_K: data = qs verbatim, 128 B/sb (int4-aligned).
//         scales = 24 B/sb { f16 d, f16 dmin, u8 sc[8], u8 m[8], 4B pad }.
//   Q5_K: data = qs (128) then qh (32) = 160 B/sb; scales as Q4_K.
//   Q6_K: data = ql (128) then qh (64) = 192 B/sb;
//         scales = 24 B/sb { f16 d, 2B pad, i8 sc[16], 4B pad }.
//   IQ4_XS: data = qs verbatim, 128 B/sb;
//         scales = 24 B/sb { f16 d, 2B pad, i8 sc[8] (ls-32 pre-unpacked), 12B pad }.
//   Q4_0: data = nibbles PERMUTED into the Q4_K convention (group g byte l =
//         sub-block 2g weight l low | sub-block 2g+1 weight l high), 128 B/sb;
//         scales = 24 B/sb { f16 dsub[8], 8B pad } - eight INDEPENDENT f16
//         block scales, which Q4_K's shared-d 6-bit integer sub-scales cannot
//         express exactly. Consumers: dj = dsub[j], mu = 8*dj (the -8 center),
//         no dmin/m term. Every product stays exact per term.
// EXACTNESS: d/dmin stay f16 in the stream; every product (d*sc, dmin*m) is
// computed in f32 IN-KERNEL, matching the reference dequant bit-for-bit per term
// (reduction order differs - the same class as the repacked Q8_0 GEMV note).

//   Q4_0: legacy 32-weight block { f16 d, u8 qs[16] } = 18 B (4.5 bpw);
//         value = d*(q-8), byte k: low nibble -> weight k, high -> k+16.
//         Served for the QAT lineage (Google's Gemma QAT
//         checkpoints TRAIN at Q4_0, making it a native low-bit format there
//         rather than a dominated PTQ pick - the "decode what the
//         lab shipped" trendline). Rides the k-quant streams as a degenerate
//         super-block: 8 blocks = 256 weights = 144 B raw, same as Q4_K.
//
#define PD_KQ_Q4K 12u
#define PD_KQ_Q5K 13u
#define PD_KQ_Q6K 14u
#define PD_KQ_IQ4XS 23u
#define PD_KQ_Q40 2u

#define PD_KQ40_SRC 144u  // 8 x 18-byte blocks per 256-weight super-block

#define PD_KQ4_SRC 144u
#define PD_KQ5_SRC 176u
#define PD_KQ6_SRC 210u
#define PD_IQ4_SRC 136u
#define PD_KQ4_DATA 128u
#define PD_KQ5_DATA 160u
#define PD_KQ6_DATA 192u
#define PD_IQ4_DATA 128u
#define PD_KQ_SCB 24u   // repacked scale-record bytes (the k-quant family + IQ4_XS + Q4_0)

// i-quant DENSE lanes (quant/iquant_dense.cuh, included after quant/kquant_w4a8.cuh):
// the k-quant exports below dispatch the IQ1/IQ2/IQ3 family + IQ4_NL to
// these by type, so the k-quant kernels stay as they are.
int pd_kq_iq_dense_dp4a(const void* data, const void* scales, const void* xq,
                        const void* xs, const void* xsums, void* y, uint32_t in_dim,
                        uint32_t out_dim, uint32_t batch, uint32_t dtype, void* stream);
int pd_kq_iq_dense_glu(const void* gd, const void* gs, const void* ud, const void* us,
                       const void* xq, const void* xs, const void* xsums, void* y,
                       uint32_t in_dim, uint32_t out_dim, uint32_t dtg, uint32_t dtu,
                       void* stream);
int pd_kq_iq_dense_gemv_f32(const void* data, const void* scales, const void* x, void* y,
                            uint32_t in_dim, uint32_t out_dim, uint32_t dtype, void* stream);
int pd_kq_iq_dense_gather(const void* data, const void* scales, const void* tokens, void* out,
                          uint32_t embd, uint32_t n_tokens, uint32_t dtype, void* stream);
// Record stride by type: the k-quant family's fixed 24 bytes, the i-quant
// family's slimmer records (quant/iquant.cuh). Every lane an i-quant seat can
// reach (repack, repacked dequant, the token-batched MoE pair, the dense gemv
// and gather) indexes records through this; the W4A8 / mma / sorted lanes
// never see an i-quant type and keep PD_KQ_SCB.
__host__ __device__ __forceinline__ uint32_t pd_kq_scb(uint32_t dt) {
    return pd_kq_valid_iq(dt) ? pd_iq_scb(dt) : PD_KQ_SCB;
}
// Formats with a per-16 MIN (mu) term: the int8 lanes need the per-16
// activation sums for these (Q4_K, Q5_K, Q4_0's folded -8, Q2_K's dmin*m).
__host__ __device__ constexpr bool pd_kq_has_mu(uint32_t dt) {
    return dt == PD_KQ_Q4K || dt == PD_KQ_Q5K || dt == PD_KQ_Q40 || dt == PD_KQ_Q2K_ID ||
           dt == PD_KQ_Q51_ID;
}

__host__ __device__ __forceinline__ bool pd_kq_valid(uint32_t dtype) {
    return dtype == PD_KQ_Q4K || dtype == PD_KQ_Q5K || dtype == PD_KQ_Q6K ||
           dtype == PD_KQ_IQ4XS || dtype == PD_KQ_Q40;
}

// IQ4_NL/IQ4_XS nonlinear 4-bit codebook (OCP-adjacent format spec, ggml table).
__device__ __constant__ int8_t PD_KQ_IQ4NL[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

// get_scale_min_k4 port: unpack the packed 6-bit (scale, min) pair j (0..7)
// from the 12-byte scales field. Format spec, not borrowed code.
__device__ __forceinline__ void pd_kq_scmin(const uint8_t* s, uint32_t j,
                                            uint32_t* sc, uint32_t* m) {
    if (j < 4u) {
        *sc = s[j] & 63u;
        *m = s[j + 4u] & 63u;
    } else {
        *sc = (s[j + 4u] & 0xFu) | ((s[j - 4u] >> 6u) << 4u);
        *m = (s[j + 4u] >> 4u) | ((s[j] >> 6u) << 4u);
    }
}

// Q4_0's repacked scale record is {f16 dsub[8]}: the per-32-block scale for
// sub-block j. Shared by every consumer arm so the read stays one shape.
__device__ __forceinline__ float pd_kq40_dj(const uint8_t* rec, uint32_t j) {
    __half h;
    memcpy(&h, rec + 2u * j, 2u);
    return __half2float(h);
}

// ---- full-tensor dequant (load-time: upload()'s f32 side-copies) -----------
// One thread per super-block, grid-stride; perf non-critical (runs once).
__global__ void pd_kquant_dequant_kernel(const uint8_t* __restrict__ src,
                                         float* __restrict__ dst,
                                         uint64_t n_super, uint32_t dtype) {
    for (uint64_t b = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; b < n_super;
         b += (uint64_t)gridDim.x * blockDim.x) {
        float* y = dst + b * 256u;
        if (pd_kq_valid_iq(dtype)) {
            pd_iq_dequant_super(dtype, src + b * pd_iq_srcb(dtype), y);
            continue;
        }
        if (dtype == PD_KQ_Q6K) {
            const uint8_t* s = src + b * PD_KQ6_SRC;
            const uint8_t* ql = s;
            const uint8_t* qh = s + 128u;
            const int8_t* sc = (const int8_t*)(s + 192u);
            __half hd;
            memcpy(&hd, s + 208u, 2u);
            const float d = __half2float(hd);
            for (uint32_t n = 0; n < 2u; ++n) {
                for (uint32_t l = 0; l < 32u; ++l) {
                    const uint32_t is = l >> 4u;
                    const uint8_t qlo = ql[n * 64u + l], qlh = ql[n * 64u + 32u + l];
                    const uint8_t h = qh[n * 32u + l];
                    const int q1 = (int)((qlo & 0xFu) | (((h >> 0u) & 3u) << 4u)) - 32;
                    const int q2 = (int)((qlh & 0xFu) | (((h >> 2u) & 3u) << 4u)) - 32;
                    const int q3 = (int)((qlo >> 4u) | (((h >> 4u) & 3u) << 4u)) - 32;
                    const int q4 = (int)((qlh >> 4u) | (((h >> 6u) & 3u) << 4u)) - 32;
                    y[n * 128u + l] = d * (float)sc[n * 8u + is] * (float)q1;
                    y[n * 128u + 32u + l] = d * (float)sc[n * 8u + 2u + is] * (float)q2;
                    y[n * 128u + 64u + l] = d * (float)sc[n * 8u + 4u + is] * (float)q3;
                    y[n * 128u + 96u + l] = d * (float)sc[n * 8u + 6u + is] * (float)q4;
                }
            }
            continue;
        }
        if (dtype == PD_KQ_IQ4XS) {
            const uint8_t* s = src + b * PD_IQ4_SRC;
            __half hd;
            memcpy(&hd, s, 2u);
            const float d = __half2float(hd);
            uint16_t sh;
            memcpy(&sh, s + 2u, 2u);
            const uint8_t* sl = s + 4u;
            const uint8_t* qs = s + 8u;
            for (uint32_t ib = 0; ib < 8u; ++ib) {
                const int ls = (int)(((sl[ib >> 1u] >> (4u * (ib & 1u))) & 0xFu) |
                                     (((sh >> (2u * ib)) & 3u) << 4u)) - 32;
                const float dl = d * (float)ls;
                const uint8_t* q = qs + ib * 16u;
                for (uint32_t j = 0; j < 16u; ++j) {
                    y[ib * 32u + j] = dl * (float)PD_KQ_IQ4NL[q[j] & 0xFu];
                    y[ib * 32u + 16u + j] = dl * (float)PD_KQ_IQ4NL[q[j] >> 4u];
                }
            }
            continue;
        }
        if (dtype == PD_KQ_Q40) {
            const uint8_t* s = src + b * PD_KQ40_SRC;
            for (uint32_t j = 0; j < 8u; ++j) {
                const uint8_t* blk = s + j * 18u;
                __half hd;
                memcpy(&hd, blk, 2u);
                const float d = __half2float(hd);
                for (uint32_t l = 0; l < 16u; ++l) {
                    y[j * 32u + l] = d * (float)((int)(blk[2u + l] & 0xFu) - 8);
                    y[j * 32u + 16u + l] = d * (float)((int)(blk[2u + l] >> 4u) - 8);
                }
            }
            continue;
        }
        const uint32_t srcb = dtype == PD_KQ_Q5K ? PD_KQ5_SRC : PD_KQ4_SRC;
        const uint8_t* s = src + b * srcb;
        __half hd, hm;
        memcpy(&hd, s, 2u);
        memcpy(&hm, s + 2u, 2u);
        const float d = __half2float(hd), dmin = __half2float(hm);
        const uint8_t* scales = s + 4u;
        // Q5_K source order: qh[32] at 16, then qs[128] at 48 (see header note)
        const uint8_t* q = s + (dtype == PD_KQ_Q5K ? 48u : 16u);
        const uint8_t* qh = s + 16u;    // Q5_K only
        for (uint32_t j = 0; j < 8u; ++j) {  // sub-block of 32
            uint32_t sc, m;
            pd_kq_scmin(scales, j, &sc, &m);
            const float dj = d * (float)sc, mj = dmin * (float)m;
            const uint8_t* qg = q + (j >> 1u) * 32u;
            const bool hi = (j & 1u) != 0u;
            for (uint32_t l = 0; l < 32u; ++l) {
                uint32_t v = hi ? (qg[l] >> 4u) : (qg[l] & 0xFu);
                if (dtype == PD_KQ_Q5K) v += ((qh[l] >> j) & 1u) ? 16u : 0u;
                y[j * 32u + l] = dj * (float)v - mj;
            }
        }
    }
}

PD_EXPORT
int pd_kquant_dequant(const void* src, void* dst, uint64_t n_super, uint32_t dtype,
                      void* stream) {
    if (n_super == 0) return 0;
    if (!pd_kq_valid(dtype) && !pd_kq_valid_iq(dtype)) return cudaErrorInvalidValue;
    uint32_t threads = 256;
    uint64_t blocks = (n_super + threads - 1) / threads;
    if (blocks > 65535u) blocks = 65535u;
    pd_kquant_dequant_kernel<<<(uint32_t)blocks, threads, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)src, (float*)dst, n_super, dtype);
    return pd_launch_status();
}

// ---- load-time repack --------------------------------------------------------
// One thread per super-block: split the interleaved block into the aligned data
// stream + fixed-stride scale records (d/dmin f16 + unpacked sc/m bytes).
__global__ void pd_kquant_repack_kernel(const uint8_t* __restrict__ src,
                                        uint8_t* __restrict__ dst_data,
                                        uint8_t* __restrict__ dst_scales,
                                        uint64_t n_super, uint32_t dtype) {
    uint64_t b = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_super) return;
    uint8_t* rec = dst_scales + b * pd_kq_scb(dtype);
    if (pd_kq_valid_iq(dtype)) {
        pd_iq_repack_super(dtype, src + b * pd_iq_srcb(dtype), dst_data + b * pd_iq_datab(dtype), rec);
        return;
    }
    if (dtype == PD_KQ_Q6K) {
        const uint8_t* s = src + b * PD_KQ6_SRC;
        uint8_t* d = dst_data + b * PD_KQ6_DATA;
        for (uint32_t i = 0; i < 192u; ++i) d[i] = s[i];      // ql + qh verbatim
        rec[0] = s[208]; rec[1] = s[209];                      // f16 d
        rec[2] = 0; rec[3] = 0;
        for (uint32_t i = 0; i < 16u; ++i) rec[4u + i] = s[192u + i];  // i8 scales
        rec[20] = rec[21] = rec[22] = rec[23] = 0;
        return;
    }
    if (dtype == PD_KQ_IQ4XS) {
        const uint8_t* s = src + b * PD_IQ4_SRC;
        uint8_t* d = dst_data + b * PD_IQ4_DATA;
        for (uint32_t i = 0; i < 128u; ++i) d[i] = s[8u + i];  // qs verbatim
        rec[0] = s[0]; rec[1] = s[1];                          // f16 d
        rec[2] = 0; rec[3] = 0;
        uint16_t sh;
        memcpy(&sh, s + 2u, 2u);
        const uint8_t* sl = s + 4u;
        for (uint32_t ib = 0; ib < 8u; ++ib) {
            const int ls = (int)(((sl[ib >> 1u] >> (4u * (ib & 1u))) & 0xFu) |
                                 (((sh >> (2u * ib)) & 3u) << 4u)) - 32;
            ((int8_t*)rec)[4u + ib] = (int8_t)ls;
        }
        for (uint32_t i = 12u; i < 24u; ++i) rec[i] = 0;
        return;
    }
    if (dtype == PD_KQ_Q40) {
        const uint8_t* s = src + b * PD_KQ40_SRC;
        uint8_t* d = dst_data + b * PD_KQ4_DATA;
        // permute into the Q4_K data convention (group g byte l = sub 2g
        // weight l low nibble | sub 2g+1 weight l high); a Q4_0 block's byte
        // k holds weights k (low) and k+16 (high)
        for (uint32_t g = 0; g < 4u; ++g) {
            const uint8_t* a = s + (2u * g) * 18u + 2u;
            const uint8_t* c = s + (2u * g + 1u) * 18u + 2u;
            for (uint32_t l = 0; l < 32u; ++l) {
                const uint32_t lo = l < 16u ? (a[l] & 0xFu) : (uint32_t)(a[l - 16u] >> 4u);
                const uint32_t hi = l < 16u ? (c[l] & 0xFu) : (uint32_t)(c[l - 16u] >> 4u);
                d[g * 32u + l] = (uint8_t)(lo | (hi << 4u));
            }
        }
        for (uint32_t j = 0; j < 8u; ++j) {  // {f16 dsub[8]}
            rec[2u * j] = s[j * 18u];
            rec[2u * j + 1u] = s[j * 18u + 1u];
        }
        for (uint32_t i = 16u; i < 24u; ++i) rec[i] = 0;
        return;
    }
    const uint32_t srcb = dtype == PD_KQ_Q5K ? PD_KQ5_SRC : PD_KQ4_SRC;
    const uint32_t datab = dtype == PD_KQ_Q5K ? PD_KQ5_DATA : PD_KQ4_DATA;
    const uint8_t* s = src + b * srcb;
    uint8_t* d = dst_data + b * datab;
    // Q5_K source order is qh then qs (see header note); repacked stays qs-first
    const uint32_t qs_off = dtype == PD_KQ_Q5K ? 48u : 16u;
    for (uint32_t i = 0; i < 128u; ++i) d[i] = s[qs_off + i];  // qs
    if (dtype == PD_KQ_Q5K)
        for (uint32_t i = 0; i < 32u; ++i) d[128u + i] = s[16u + i];   // qh
    rec[0] = s[0]; rec[1] = s[1];                              // f16 d
    rec[2] = s[2]; rec[3] = s[3];                              // f16 dmin
    for (uint32_t j = 0; j < 8u; ++j) {
        uint32_t sc, m;
        pd_kq_scmin(s + 4u, j, &sc, &m);
        rec[4u + j] = (uint8_t)sc;
        rec[12u + j] = (uint8_t)m;
    }
    rec[20] = rec[21] = rec[22] = rec[23] = 0;
}

PD_EXPORT
int pd_kquant_repack(const void* src, void* dst_data, void* dst_scales,
                     uint64_t n_super, uint32_t dtype, void* stream) {
    if (n_super == 0) return 0;
    if (!pd_kq_valid(dtype) && !pd_kq_valid_iq(dtype)) return cudaErrorInvalidValue;
    uint32_t threads = 256;
    uint64_t blocks = (n_super + threads - 1) / threads;
    pd_kquant_repack_kernel<<<(uint32_t)blocks, threads, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)src, (uint8_t*)dst_data, (uint8_t*)dst_scales, n_super, dtype);
    return pd_launch_status();
}

// ---- fused decode GEMV (the stage-1 payoff: weight stream at 4.25-6.6 bpw) ----
// One block per output row (pd_q8_0_gemv_repacked's shape). VECTORIZED walk:
// every thread strides 16-BYTE data chunks (one uint4 load = 32 nibble weights
// on the 4-bit formats), unpacks in registers, and runs f32 dot/sum pairs -
// the first landing's per-byte scalar loop was instruction-bound at ~88 GB/s
// effective (16 tok/s on the 9B vs llama's 83); the chunk walk is ~2 FMA +
// ~2 logic ops per weight with fully coalesced loads. EXACTNESS unchanged:
// per-term f32 products identical to the reference dequant (only the
// commutative summation grouping differs - the class every GEMV here has).
// 4 ADJACENT rows per block (their repacked streams are contiguous - one
// block covers a 4-row DRAM span), 64 threads per row: at embd-class in_dims
// a one-row block had fewer chunks than threads (zero ILP, latency-bound at
// ~150 GB/s); the 4-row tile gives each thread 2-6 chunks and 4x the bytes
// in flight.
//
// Round 2, three bit-exact reshapes:
//  * x staged in TILES of 4096 floats, not whole-in_dim: a 12288-wide ffn_down
//    staged 50 KB and capped the SM at one block (25% occupancy, measured 60%
//    of the Q8 ref); ~17 KB/tile restores 5 blocks/SM and lifts the in_dim
//    ceiling entirely. Tile = 16 supers, so no chunk straddles; the per-thread
//    chunk sequence (stride 64, ascending) is unchanged -> identical sums.
//  * magic-number int->float (0x4B000000|v reads as 8388608+v; one FSUB later
//    it's float(v) exactly for v < 2^22): replaces the per-weight I2F, a
//    low-throughput-pipe op on sm_86, with LOP3+FADD-class work.
#define PD_KQ_GEMV_TILE 4096u  // x floats staged per shared tile (multiple of 256)
__global__ void pd_kquant_gemv_kernel(const uint8_t* __restrict__ data,
                                      const uint8_t* __restrict__ scales,
                                      const float* __restrict__ x,
                                      float* __restrict__ y,
                                      uint32_t in_dim, uint32_t out_dim, uint32_t dtype) {
    const uint32_t tid = threadIdx.x;
    const uint32_t lr = tid >> 6u;                 // row-in-block 0..3
    const uint32_t o = blockIdx.x * 4u + lr;
    const uint32_t tt = tid & 63u;                 // thread-in-row
    const uint32_t n_super = in_dim >> 8u;
    const uint32_t datab = dtype == PD_KQ_Q6K ? PD_KQ6_DATA
                         : dtype == PD_KQ_Q5K ? PD_KQ5_DATA : PD_KQ4_DATA;
    const uint8_t* rowd = data + (size_t)o * n_super * datab;
    const uint32_t scb = pd_kq_scb(dtype);
    const uint8_t* rows = scales + (size_t)o * n_super * scb;
    // x staged into PADDED shared, one tile per pass (coalesced global read,
    // shared by the 4 rows). The chunk walk's per-thread x runs are 32-float
    // strided across lanes - as global loads that is 32 L1 transactions per
    // load instruction, measured 160-215 GB/s vs the Q8 GEMV's 585-665. The
    // +1-float-per-32 pad spreads the strided runs across banks.
    extern __shared__ float xsh[];
    float acc = 0.0f;
    // every format walks n_super * 8 sixteen-byte chunks per row; a
    // past-the-end row (out_dim % 4 tail) walks zero and writes nothing -
    // but every thread still runs the (uniform) tile loop and its barriers
    const uint32_t n_chunk = o < out_dim ? (n_super << 3u) : 0u;
    const uint32_t nth = 64u;
    for (uint32_t x0 = 0; x0 < in_dim; x0 += PD_KQ_GEMV_TILE) {
        const uint32_t tf = min(PD_KQ_GEMV_TILE, in_dim - x0);
        if (x0 != 0u) __syncthreads();  // previous tile fully consumed
        for (uint32_t i = tid; i < tf; i += 256u) xsh[i + (i >> 5u)] = x[x0 + i];
        __syncthreads();
        // this tile's chunk range: 16 supers per tile, chunks never straddle
        const uint32_t tc0 = (x0 >> 8u) << 3u;
        const uint32_t tc1 = min(n_chunk, tc0 + ((tf >> 8u) << 3u));
    if (dtype == PD_KQ_Q6K && in_dim > PD_KQ_GEMV_TILE) {
        // MERGED task (s, n, h) for MULTI-TILE rows: both ql 32-halves a=0/1
        // in one pass - 3 loads + 1 rec read where the (n, a, h) walk paid 4
        // + 2 (the qh double-read itself was warp-coalesced; the win is the
        // instruction count). Measured: ffn_down [12288x4096] 427 -> ~530
        // GB/s. On SINGLE-TILE rows the same merge measured ~-18% (the head
        // [4096x248320] 604 -> ~490: one task per thread, nothing left to
        // overlap the longer unpack chain), so those keep the original walk
        // below. Same per-term products and per-chunk expressions; only which
        // THREAD folds the a=1 terms changes (the commutative-grouping class
        // every GEMV here has).
        const uint32_t mc0 = tc0 >> 1u, mc1 = tc1 >> 1u;  // 4 tasks per super
        for (uint32_t c = mc0 + tt; c < mc1; c += nth) {
            const uint32_t s = c >> 2u, ci = c & 3u;
            const uint32_t n = ci >> 1u, h = ci & 1u;
            const uint8_t* sb = rowd + (size_t)s * PD_KQ6_DATA;
            const uint4 qa = *(const uint4*)(sb + n * 64u + h * 16u);
            const uint4 qb_ = *(const uint4*)(sb + n * 64u + 32u + h * 16u);
            const uint4 hv = *(const uint4*)(sb + 128u + n * 32u + h * 16u);
            const uint8_t* rec = rows + (size_t)s * scb;
            __half hd;
            memcpy(&hd, rec, 2u);
            const float d = __half2float(hd);
            const int8_t* sc = (const int8_t*)rec + 4;
            const float sc10 = (float)sc[n * 8u + h];        // a=0 lo (row 0)
            const float sc20 = (float)sc[n * 8u + 4u + h];   // a=0 hi (row 2)
            const float sc11 = (float)sc[n * 8u + 2u + h];   // a=1 lo (row 1)
            const float sc21 = (float)sc[n * 8u + 6u + h];   // a=1 hi (row 3)
            const uint32_t xb = s * 256u + n * 128u + h * 16u - x0;
            const float* xa0 = xsh + xb + (xb >> 5u);
            const float* xh0 = xa0 + 66u; // +64 weights (row 0 -> 2) +2 pad
            const float* xa1 = xa0 + 33u; // +32 weights (a=0 -> a=1) +1 pad
            const float* xh1 = xa1 + 66u;
            float s10 = 0.0f, s20 = 0.0f, s11 = 0.0f, s21 = 0.0f;
            const uint32_t q0[4] = {qa.x, qa.y, qa.z, qa.w};
            const uint32_t q1[4] = {qb_.x, qb_.y, qb_.z, qb_.w};
            const uint32_t hw[4] = {hv.x, hv.y, hv.z, hv.w};
            #pragma unroll
            for (uint32_t wi = 0; wi < 4u; ++wi) {
                #pragma unroll
                for (uint32_t b = 0; b < 4u; ++b) {
                    const uint32_t k = wi * 4u + b;
                    const uint32_t b0 = (q0[wi] >> (8u * b)) & 0xFFu;
                    const uint32_t b1 = (q1[wi] >> (8u * b)) & 0xFFu;
                    const uint32_t hb = hw[wi] >> (8u * b);
                    // magic: 0x4B000000|v = 8388608+v; -8388640 -> (v-32) exact
                    const float f10 = __uint_as_float(
                        0x4B000000u | (b0 & 0xFu) | ((hb & 3u) << 4u)) - 8388640.0f;
                    const float f20 = __uint_as_float(
                        0x4B000000u | (b0 >> 4u) | (((hb >> 4u) & 3u) << 4u)) - 8388640.0f;
                    const float f11 = __uint_as_float(
                        0x4B000000u | (b1 & 0xFu) | (((hb >> 2u) & 3u) << 4u)) - 8388640.0f;
                    const float f21 = __uint_as_float(
                        0x4B000000u | (b1 >> 4u) | (((hb >> 6u) & 3u) << 4u)) - 8388640.0f;
                    s10 = fmaf(f10, xa0[k], s10);
                    s20 = fmaf(f20, xh0[k], s20);
                    s11 = fmaf(f11, xa1[k], s11);
                    s21 = fmaf(f21, xh1[k], s21);
                }
            }
            acc += d * fmaf(sc10, s10, sc20 * s20);
            acc += d * fmaf(sc11, s11, sc21 * s21);
        }
    } else if (dtype == PD_KQ_Q6K) {
        // single-tile Q6K: the original (s, n, a, h) chunk walk (see above)
        for (uint32_t c = tc0 + tt; c < tc1; c += nth) {
            const uint32_t s = c >> 3u, ci = c & 7u;
            const uint32_t n = ci >> 2u, a = (ci >> 1u) & 1u, h = ci & 1u;
            const uint8_t* sb = rowd + (size_t)s * PD_KQ6_DATA;
            const uint4 qv = *(const uint4*)(sb + n * 64u + a * 32u + h * 16u);
            const uint4 hv = *(const uint4*)(sb + 128u + n * 32u + h * 16u);
            const uint8_t* rec = rows + (size_t)s * scb;
            __half hd;
            memcpy(&hd, rec, 2u);
            const float d = __half2float(hd);
            const int8_t* sc = (const int8_t*)rec + 4;
            const float sc1 = (float)sc[n * 8u + a * 2u + h];
            const float sc2 = (float)sc[n * 8u + (a + 2u) * 2u + h];
            const uint32_t xb = s * 256u + n * 128u + a * 32u + h * 16u - x0;
            const float* x1 = xsh + xb + (xb >> 5u);
            const float* x2 = x1 + 66u; // +64 weights (rows a -> a+2) +2 pad
            float s1 = 0.0f, s2 = 0.0f;
            const uint32_t sh1 = 2u * a, sh2 = 2u * a + 4u;
            const uint32_t qw[4] = {qv.x, qv.y, qv.z, qv.w};
            const uint32_t hw[4] = {hv.x, hv.y, hv.z, hv.w};
            #pragma unroll
            for (uint32_t wi = 0; wi < 4u; ++wi) {
                #pragma unroll
                for (uint32_t b = 0; b < 4u; ++b) {
                    const uint32_t k = wi * 4u + b;
                    const uint32_t qb = (qw[wi] >> (8u * b)) & 0xFFu;
                    const uint32_t hb = hw[wi] >> (8u * b);
                    // magic: 0x4B000000|v = 8388608+v; -8388640 -> (v-32) exact
                    const float q1 = __uint_as_float(
                        0x4B000000u | (qb & 0xFu) | (((hb >> sh1) & 3u) << 4u)) - 8388640.0f;
                    const float q2 = __uint_as_float(
                        0x4B000000u | (qb >> 4u) | (((hb >> sh2) & 3u) << 4u)) - 8388640.0f;
                    s1 = fmaf(q1, x1[k], s1);
                    s2 = fmaf(q2, x2[k], s2);
                }
            }
            acc += d * fmaf(sc1, s1, sc2 * s2);
        }
    } else if (dtype == PD_KQ_IQ4XS) {
        // chunk c -> super s, sub-block ib: 16 qs bytes; lo nibble -> weight
        // ib*32+k, hi -> ib*32+16+k; shared-mem codebook per nibble.
        for (uint32_t c = tc0 + tt; c < tc1; c += nth) {
            const uint32_t s = c >> 3u, ib = c & 7u;
            const uint4 qv = *(const uint4*)(rowd + (size_t)s * PD_IQ4_DATA + ib * 16u);
            const uint8_t* rec = rows + (size_t)s * scb;
            __half hd;
            memcpy(&hd, rec, 2u);
            const float dl =
                __half2float(hd) * (float)((const int8_t*)rec)[4u + ib];
            const uint32_t xb = s * 256u + ib * 32u - x0;
            const float* x1 = xsh + xb + (xb >> 5u);
            const float* x2 = x1 + 16u; // same 32-block: same pad offset
            // Codebook via BIASED prmt lookup (value+128 packed per byte, the
            // ks-v2 register-codebook trick) + magic-number float build - no
            // LDS on the weight side at all. The shared-LUT versions measured
            // 421-424 GB/s vs Q4K's ~550 on the same byte count: every weight
            // paid a LUT load on TOP of its x load, doubling LDS traffic
            // (bank-replicating the LUT killed conflicts but not the issue
            // pressure). float(value) is exact (integers < 2^8), so the
            // per-term products are unchanged. Two accumulator chains.
            float s1 = 0.0f, s2 = 0.0f;
            const uint32_t qw[4] = {qv.x, qv.y, qv.z, qv.w};
            #pragma unroll
            for (uint32_t wi = 0; wi < 4u; ++wi) {
                // 4 lo-nibble + 4 hi-nibble indices -> biased codebook bytes.
                // Constants = IQ4NL values + 128 (bytes idx0..3, 4..7, 8..11,
                // 12..15); prmt's msb-replicate mode garbles exactly where
                // bit 3 selects the hi table, and the mask discards those.
                const uint32_t lo4 = qw[wi] & 0x0F0F0F0Fu;
                const uint32_t hi4 = (qw[wi] >> 4u) & 0x0F0F0F0Fu;
                const uint32_t cl = (lo4 | (lo4 >> 4u)) & 0x00FF00FFu;
                const uint32_t sl = (cl | (cl >> 8u)) & 0xFFFFu;
                const uint32_t ml = ((lo4 >> 3u) & 0x01010101u) * 0xFFu;
                const uint32_t pkl = (__byte_perm(0xA6998D81u, 0xF1D9C5B5u, sl & 0x7777u) & ml)
                                   | (__byte_perm(0x3F2D1801u, 0x766A5D4Fu, sl) & ~ml);
                const uint32_t ch = (hi4 | (hi4 >> 4u)) & 0x00FF00FFu;
                const uint32_t sh = (ch | (ch >> 8u)) & 0xFFFFu;
                const uint32_t mh = ((hi4 >> 3u) & 0x01010101u) * 0xFFu;
                const uint32_t pkh = (__byte_perm(0xA6998D81u, 0xF1D9C5B5u, sh & 0x7777u) & mh)
                                   | (__byte_perm(0x3F2D1801u, 0x766A5D4Fu, sh) & ~mh);
                #pragma unroll
                for (uint32_t b = 0; b < 4u; ++b) {
                    const uint32_t k = wi * 4u + b;
                    // magic: 0x4B000000|(v+128) = 8388736+v -> FSUB is exact
                    const float f1 = __uint_as_float(
                        0x4B000000u | ((pkl >> (8u * b)) & 0xFFu)) - 8388736.0f;
                    const float f2 = __uint_as_float(
                        0x4B000000u | ((pkh >> (8u * b)) & 0xFFu)) - 8388736.0f;
                    s1 = fmaf(f1, x1[k], s1);
                    s2 = fmaf(f2, x2[k], s2);
                }
            }
            acc += dl * (s1 + s2);
        }
    } else {
        // Q4_K / Q5_K: chunk c -> super s, 64-group g, 16-half h. The 16 qs
        // bytes hold sub-block 2g (lo nibbles) and 2g+1 (hi); Q5's qh bytes
        // (per-l, shared across groups) ride bits 2g / 2g+1.
        const bool q5 = dtype == PD_KQ_Q5K;
        const bool q40 = dtype == PD_KQ_Q40;
        for (uint32_t c = tc0 + tt; c < tc1; c += nth) {
            const uint32_t s = c >> 3u, ci = c & 7u;
            const uint32_t g = ci >> 1u, h = ci & 1u;
            const uint8_t* sb = rowd + (size_t)s * datab;
            const uint4 qv = *(const uint4*)(sb + g * 32u + h * 16u);
            uint4 hv = make_uint4(0u, 0u, 0u, 0u);
            if (q5) hv = *(const uint4*)(sb + 128u + h * 16u);
            const uint8_t* rec = rows + (size_t)s * scb;
            __half hd, hm;
            memcpy(&hd, rec, 2u);
            memcpy(&hm, rec + 2u, 2u);
            const float d = __half2float(hd), dmin = __half2float(hm);
            const uint32_t j1 = 2u * g, j2 = 2u * g + 1u;
            // Q4_0: dj = dsub[j], the offset is 8*dj (value = dsub*(q-8))
            const float dj1 = q40 ? pd_kq40_dj(rec, j1) : d * (float)rec[4u + j1];
            const float dj2 = q40 ? pd_kq40_dj(rec, j2) : d * (float)rec[4u + j2];
            const float mj1 = q40 ? 8.0f * dj1 : dmin * (float)rec[12u + j1];
            const float mj2 = q40 ? 8.0f * dj2 : dmin * (float)rec[12u + j2];
            const uint32_t xb = s * 256u + j1 * 32u + h * 16u - x0;
            const float* x1 = xsh + xb + (xb >> 5u);
            const float* x2 = x1 + 33u; // +32 weights (sub 2g -> 2g+1) +1 pad
            // sx sums ride the registers already loaded for the dot - ~3% of
            // FMA capacity; a precomputed-window variant traded them for a
            // strided global sweep and measured 20% slower on Q4K
            float s1 = 0.0f, sx1 = 0.0f, s2 = 0.0f, sx2 = 0.0f;
            const uint32_t qw[4] = {qv.x, qv.y, qv.z, qv.w};
            const uint32_t hw[4] = {hv.x, hv.y, hv.z, hv.w};
            #pragma unroll
            for (uint32_t wi = 0; wi < 4u; ++wi) {
                #pragma unroll
                for (uint32_t b = 0; b < 4u; ++b) {
                    const uint32_t k = wi * 4u + b;
                    const uint32_t qb = (qw[wi] >> (8u * b)) & 0xFFu;
                    const uint32_t hb = hw[wi] >> (8u * b);
                    // magic: 0x4B000000|v = 8388608+v; -8388608 -> float(v) exact
                    const float f1 = __uint_as_float(
                        0x4B000000u | (qb & 0xFu) | (((hb >> j1) & 1u) << 4u)) - 8388608.0f;
                    const float f2 = __uint_as_float(
                        0x4B000000u | (qb >> 4u) | (((hb >> j2) & 1u) << 4u)) - 8388608.0f;
                    const float xa = x1[k], xc = x2[k];
                    s1 = fmaf(f1, xa, s1);
                    sx1 += xa;
                    s2 = fmaf(f2, xc, s2);
                    sx2 += xc;
                }
            }
            acc += fmaf(dj1, s1, -mj1 * sx1) + fmaf(dj2, s2, -mj2 * sx2);
        }
    }
    }
    // rows are warp-aligned (64 threads = warps 2*lr, 2*lr+1): warp-reduce,
    // then one thread per row folds its two warp partials.
    __shared__ float wsum[8];
    for (uint32_t sdown = 16; sdown > 0; sdown >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, sdown);
    const uint32_t warp = tid >> 5u, lane = tid & 31u;
    if (lane == 0) wsum[warp] = acc;
    __syncthreads();
    if (tid < 4u) {
        const uint32_t ro = blockIdx.x * 4u + tid;
        if (ro < out_dim) y[ro] = wsum[2u * tid] + wsum[2u * tid + 1u];
    }
}

PD_EXPORT
int pd_kquant_gemv(const void* data, const void* scales, const void* x, void* y,
                   uint32_t in_dim, uint32_t out_dim, uint32_t dtype, void* stream) {
    if (out_dim == 0) return 0;
    if (pd_kq_valid_iq(dtype))
        return pd_kq_iq_dense_gemv_f32(data, scales, x, y, in_dim, out_dim, dtype, stream);
    if ((in_dim & 255u) != 0u) return cudaErrorInvalidValue;
    if (!pd_kq_valid(dtype)) return cudaErrorInvalidValue;
    // padded x staging for one tile: min(in_dim, tile) floats + 1 pad per 32.
    // Caps at ~17 KB so the SM fits 5 blocks at any in_dim (the old whole-row
    // staging hit 50 KB on a 12288-wide ffn_down -> 1 block/SM).
    const uint32_t tf0 = in_dim < PD_KQ_GEMV_TILE ? in_dim : PD_KQ_GEMV_TILE;
    const uint32_t smem = (tf0 + (tf0 >> 5u)) * 4u;
    pd_kquant_gemv_kernel<<<(out_dim + 3u) / 4u, 256, smem, (cudaStream_t)stream>>>(
        (const uint8_t*)data, (const uint8_t*)scales, (const float*)x, (float*)y,
        in_dim, out_dim, dtype);
    return pd_launch_status();
}

// ---- embedding row-gather (token_embd stays k-quant resident) ----------------
// One block per token: dequant that row from the REPACKED streams into
// out[i*embd ..). Threads stride the row's sub-blocks/groups with the same
// exact math as the GEMV.
__global__ void pd_kquant_gather_kernel(const uint8_t* __restrict__ data,
                                        const uint8_t* __restrict__ scales,
                                        const unsigned int* __restrict__ tokens,
                                        float* __restrict__ out,
                                        uint32_t embd, uint32_t n_tokens, uint32_t dtype) {
    const uint32_t i = blockIdx.x;
    if (i >= n_tokens) return;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    const uint32_t n_super = embd >> 8u;
    const uint32_t datab = dtype == PD_KQ_Q6K ? PD_KQ6_DATA
                         : dtype == PD_KQ_Q5K ? PD_KQ5_DATA : PD_KQ4_DATA;
    const size_t row = tokens[i];
    const uint8_t* rowd = data + row * n_super * datab;
    const uint32_t scb = pd_kq_scb(dtype);
    const uint8_t* rows = scales + row * n_super * scb;
    float* y = out + (size_t)i * embd;
    if (dtype == PD_KQ_Q6K) {
        const uint32_t n_grp = embd >> 4u;
        for (uint32_t g = tid; g < n_grp; g += nth) {
            const uint32_t s = g >> 4u, gi = g & 15u;
            const uint32_t n = gi >> 3u, r = gi & 7u;
            const uint32_t rw = r >> 1u, hl = r & 1u;
            const uint8_t* sb = rowd + (size_t)s * PD_KQ6_DATA;
            const uint8_t* ql = sb + n * 64u + ((rw & 1u) ? 32u : 0u);
            const uint8_t* qh = sb + 128u + n * 32u;
            const uint8_t* rec = rows + (size_t)s * scb;
            __half hd;
            memcpy(&hd, rec, 2u);
            const float d = __half2float(hd);
            const float scg = (float)((const int8_t*)rec)[4u + n * 8u + rw * 2u + hl];
            const bool hi = rw >= 2u;
            const uint32_t shift = rw * 2u;
            for (uint32_t t = 0; t < 16u; ++t) {
                const uint32_t l = hl * 16u + t;
                const uint32_t nib = hi ? (ql[l] >> 4u) : (ql[l] & 0xFu);
                const int q = (int)(nib | (((qh[l] >> shift) & 3u) << 4u)) - 32;
                y[s * 256u + n * 128u + rw * 32u + hl * 16u + t] = d * scg * (float)q;
            }
        }
        return;
    }
    if (dtype == PD_KQ_IQ4XS) {
        const uint32_t n_sub = embd >> 5u;
        for (uint32_t jj = tid; jj < n_sub; jj += nth) {
            const uint32_t s = jj >> 3u, ib = jj & 7u;
            const uint8_t* q = rowd + (size_t)s * PD_IQ4_DATA + ib * 16u;
            const uint8_t* rec = rows + (size_t)s * scb;
            __half hd;
            memcpy(&hd, rec, 2u);
            const float dl =
                __half2float(hd) * (float)((const int8_t*)rec)[4u + ib];
            float* yg = y + s * 256u + ib * 32u;
            for (uint32_t j = 0; j < 16u; ++j) {
                yg[j] = dl * (float)PD_KQ_IQ4NL[q[j] & 0xFu];
                yg[16u + j] = dl * (float)PD_KQ_IQ4NL[q[j] >> 4u];
            }
        }
        return;
    }
    const bool q5 = dtype == PD_KQ_Q5K;
    const bool q40 = dtype == PD_KQ_Q40;
    const uint32_t n_sub = embd >> 5u;
    for (uint32_t jj = tid; jj < n_sub; jj += nth) {
        const uint32_t s = jj >> 3u, j = jj & 7u;
        const uint8_t* sb = rowd + (size_t)s * datab;
        const uint8_t* qg = sb + (j >> 1u) * 32u;
        const uint8_t* qh = sb + 128u;
        const uint8_t* rec = rows + (size_t)s * scb;
        __half hd, hm;
        memcpy(&hd, rec, 2u);
        memcpy(&hm, rec + 2u, 2u);
        const float dj = q40 ? pd_kq40_dj(rec, j)
                             : __half2float(hd) * (float)rec[4u + j];
        const float mj = q40 ? 8.0f * dj
                             : __half2float(hm) * (float)rec[12u + j];
        const bool hi = (j & 1u) != 0u;
        for (uint32_t l = 0; l < 32u; ++l) {
            uint32_t v = hi ? (qg[l] >> 4u) : (qg[l] & 0xFu);
            if (q5) v += ((qh[l] >> j) & 1u) ? 16u : 0u;
            y[s * 256u + j * 32u + l] = dj * (float)v - mj;
        }
    }
}

PD_EXPORT
int pd_kquant_gather(const void* data, const void* scales, const void* tokens,
                     void* out, uint32_t embd, uint32_t n_tokens, uint32_t dtype,
                     void* stream) {
    if (n_tokens == 0) return 0;
    if (pd_kq_valid_iq(dtype))
        return pd_kq_iq_dense_gather(data, scales, tokens, out, embd, n_tokens, dtype, stream);
    if ((embd & 255u) != 0u) return cudaErrorInvalidValue;
    if (!pd_kq_valid(dtype)) return cudaErrorInvalidValue;
    pd_kquant_gather_kernel<<<n_tokens, 256, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)data, (const uint8_t*)scales, (const unsigned int*)tokens,
        (float*)out, embd, n_tokens, dtype);
    return pd_launch_status();
}

// ---- Q8_0 embedding row-gather from the REPACKED streams --------------------
// pd_embed_gather_q8's twin over the layout pd_q8_0_repack writes: the GGUF
// block stream split into an int8 plane (32 per block) and an f16 scale plane
// (one per block), block order unchanged, so row r of a [embd, rows] table is
// data[r*embd ..) and scales[r*embd/32 ..). It lets a TIED embedding gather
// from the LM head's own repacked plane instead of holding a second, raw copy
// of the same matrix (248320 x 2560 is 675 MB of Q8_0 on LightOnOCR-3-4B).
// Same product order as the raw kernel, q * d * scale, so the two are
// bit-identical on the same table. Graph-capturable: token ids from device.
__global__ void pd_embed_gather_q8r_kernel(const int8_t* __restrict__ data,
                                           const __half* __restrict__ scales,
                                           const uint32_t* __restrict__ tokens,
                                           float* __restrict__ out, uint32_t embd,
                                           float scale) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t t = blockIdx.y;
    if (i >= embd) return;
    const size_t row = (size_t)tokens[t];
    const float d = __half2float(scales[row * (embd >> 5u) + (i >> 5u)]);
    out[(size_t)t * embd + i] = (float)data[row * embd + i] * d * scale;
}

PD_EXPORT
int pd_embed_gather_q8r(const void* data, const void* scales, const void* tokens, void* out,
                        uint32_t embd, uint32_t n_tokens, float scale, void* stream) {
    if (embd == 0 || n_tokens == 0) return 0;
    if ((embd & 31u) != 0u) return cudaErrorInvalidValue;
    const uint32_t threads = 256;
    dim3 grid((embd + threads - 1) / threads, n_tokens);
    pd_embed_gather_q8r_kernel<<<grid, threads, 0, (cudaStream_t)stream>>>(
        (const int8_t*)data, (const __half*)scales, (const uint32_t*)tokens, (float*)out, embd,
        scale);
    return pd_launch_status();
}

// ---- dequant from the REPACKED streams (per-use, prefill interim) ------------
// The raw GGUF upload is freed after repack, so the batch-GEMM interim
// (dequant whole weight into an f32 scratch -> f32 GEMM, exact values) dequants
// from what is actually resident. One super-block per CUDA block (grid-stride),
// one weight per thread; bandwidth-bound and transient - the stage-2 W4A8 MMA
// replaces this whole path.
__global__ void pd_kquant_dequant_rp_kernel(const uint8_t* __restrict__ data,
                                            const uint8_t* __restrict__ scales,
                                            float* __restrict__ dst,
                                            uint64_t n_super, uint32_t dtype) {
    const uint32_t t = threadIdx.x;  // 256 = one weight each
    for (uint64_t b = blockIdx.x; b < n_super; b += gridDim.x) {
        const uint8_t* rec = scales + b * pd_kq_scb(dtype);
        float* y = dst + b * 256u;
        if (pd_kq_valid_iq(dtype)) {
            // the i-quant family off its repacked (payload + record) form:
            // one 16-weight window per 16 threads through the same unpack the
            // serving lanes read
            const uint8_t* sb = data + b * pd_iq_datab(dtype);
            const uint32_t w = t >> 4u, j = t & 15u;
            int wq[4];
            float f, g;
            pd_iq_win_unpack(dtype, sb, rec, w, wq, &f, &g);
            y[t] = f * (float)((int8_t)((wq[j >> 2u] >> (8u * (j & 3u))) & 0xffu)) + g;
            continue;
        }
        if (dtype == PD_KQ_Q6K) {
            const uint8_t* sb = data + b * PD_KQ6_DATA;
            const uint32_t n = t >> 7u, idx = t & 127u;
            const uint32_t row = idx >> 5u, l = idx & 31u;
            const uint8_t qlb = sb[n * 64u + ((row & 1u) ? 32u : 0u) + l];
            const uint8_t h = sb[128u + n * 32u + l];
            const uint32_t nib = (row >= 2u) ? (qlb >> 4u) : (qlb & 0xFu);
            const int q = (int)(nib | (((h >> (2u * row)) & 3u) << 4u)) - 32;
            __half hd;
            memcpy(&hd, rec, 2u);
            const float scg =
                (float)((const int8_t*)rec)[4u + n * 8u + row * 2u + (l >> 4u)];
            y[n * 128u + row * 32u + l] = __half2float(hd) * scg * (float)q;
            continue;
        }
        if (dtype == PD_KQ_IQ4XS) {
            const uint8_t* sb = data + b * PD_IQ4_DATA;
            const uint32_t ib = t >> 5u, l = t & 31u;
            const uint8_t qb = sb[ib * 16u + (l & 15u)];
            const uint32_t nib = (l >= 16u) ? (qb >> 4u) : (qb & 0xFu);
            __half hd;
            memcpy(&hd, rec, 2u);
            const float dl =
                __half2float(hd) * (float)((const int8_t*)rec)[4u + ib];
            y[t] = dl * (float)PD_KQ_IQ4NL[nib];
            continue;
        }
        const bool q5 = dtype == PD_KQ_Q5K;
        const bool q40 = dtype == PD_KQ_Q40;
        const uint8_t* sb = data + b * (q5 ? PD_KQ5_DATA : PD_KQ4_DATA);
        const uint32_t j = t >> 5u, l = t & 31u;
        const uint8_t qb = sb[(j >> 1u) * 32u + l];
        uint32_t v = (j & 1u) ? (qb >> 4u) : (qb & 0xFu);
        if (q5) v += ((sb[128u + l] >> j) & 1u) ? 16u : 0u;
        __half hd, hm;
        memcpy(&hd, rec, 2u);
        memcpy(&hm, rec + 2u, 2u);
        const float dj = q40 ? pd_kq40_dj(rec, j)
                             : __half2float(hd) * (float)rec[4u + j];
        const float mj = q40 ? 8.0f * dj
                             : __half2float(hm) * (float)rec[12u + j];
        y[t] = dj * (float)v - mj;
    }
}

// Capability marker (slot 478): present iff every kquant kernel in this pack
// serves PD_KQ_Q40. The engine gates its Q4_0 -> kquant route on this slot and
// falls back to the exact Q8_0 transcode against an older pack, because the
// dtype rides existing entry points and slot presence alone cannot say.
PD_EXPORT
int pd_kquant_q40(void) { return 0; }

// Capability marker (slot 600): present iff the i-quant lanes serve the flat
// 32-weight block formats Q5_1 and Q8_0 (quant/iquant.cuh) - the routed
// expert types UD-Q4_K_XL exports mix in. Same reason as the Q4_0 marker:
// the dtypes ride existing entry points.
PD_EXPORT
int pd_kquant_flat32(void) { return 0; }

// Capability marker (slot 655): present iff the flat 32-weight lanes above
// also serve Q5_0 (GGUF raw id 6) - the third flat type, which the quantizer
// puts on rows a 256-block recipe cannot encode (the gemma-4 A4B's 704-wide
// expert down and 2112-wide shared down in a Q4_K_M file). Same reason as
// slot 600: the dtype rides existing entry points.
PD_EXPORT
int pd_kquant_q50(void) { return 0; }

// Capability marker (slot 577): present iff the k-quant repack, dequant and
// token-batched MoE pair serve the ggml i-quant family (IQ1_S/M, IQ2_XXS/XS/S,
// IQ3_XXS/S) and IQ4_NL - see quant/iquant.cuh. Same reason as the Q4_0
// marker: the dtypes ride existing entry points.
PD_EXPORT
int pd_kquant_iq(void) { return 0; }

PD_EXPORT
int pd_kquant_dequant_rp(const void* data, const void* scales, void* dst,
                         uint64_t n_super, uint32_t dtype, void* stream) {
    if (n_super == 0) return 0;
    if (!pd_kq_valid(dtype) && !pd_kq_valid_iq(dtype)) return cudaErrorInvalidValue;
    uint64_t blocks = n_super < 65535u ? n_super : 65535u;
    pd_kquant_dequant_rp_kernel<<<(uint32_t)blocks, 256, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)data, (const uint8_t*)scales, (float*)dst, n_super, dtype);
    return pd_launch_status();
}
