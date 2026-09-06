// In-house f16xf16->f32 tensor-core GEMM (PADDOCK_INHOUSE_F16).
// Replaces the cuBLAS class-B helper gemm_f16_f32_beta (cublasGemmEx
// f16xf16->f32, COMPUTE_32F, beta in {0,1}) so cublas/cublasLt (507 MiB / 81%
// of the win-x64 runtime download) can eventually be dropped. Semantics match
// exactly: OP_T weight [out_dim,in_dim] x OP_N acts [batch,in_dim] -> y
// [batch,out_dim] f32, i.e. y[b][o] = beta*y[b][o] + sum_i w[o][i]*x[b][i].
// Both w and x are row-major/K-contiguous, which maps straight onto the
// tensor-core GEMM with no transposes.
//
// Three code paths, one entry point (pd_f16_gemm):
//   - pd_f16_gemm_tc5d_kernel : the sm_100a large-regular arm - tcgen05
//     cta_group::2 kind::f16, persistent cluster pairs, two col-adjacent
//     256x256 tiles in flight per cluster sharing the W slab (D0=tmem cols
//     0-255, D1=256-511). Batch >= 256 and in_dim%8==0 only; bit-identical
//     to cuBLAS (same instruction class, same accumulation order class).
//     Why it exists: warp-mma m16n8k16 ISSUES at Ampere-parity on Blackwell
//     (measured 554 TF pure-mma ceiling = 148 SM x 2048 flop/clk x 1.83GHz)
//     while tcgen05 kind::f16 sustains 2.2 PF hot - cuBLAS's nvjet kernels
//     are 2cta tcgen05, so only this class can approach/beat them. Measured
//     745-1260 TF by shape (cuBLAS 1.3-1.7 PF; the residual is tile-count
//     quantization at 74 clusters + ~330ns/slab loop latency).
//   - pd_f16_gemm_mma_kernel  : the portable-fast twin (sm_80+). Hand-rolled
//     ldmatrix + mma.sync m16n8k16 (f16xf16->f32), ST-deep cp.async ring.
//     Modeled bolt-for-bolt on int8_mma.cuh's pd_q8_0_gemm_mma_kernel (same
//     M=out/N=batch/K=in tiling, same ldmatrix lane->address map, same store)
//     with all the Q8_0 scale machinery removed (f16 needs none) and the tile
//     stepped k16 instead of k32. Zero-padded staging makes arbitrary M/N/K
//     correct (muse vision has K=588, N=972 - not 16-multiples) and beta is a
//     store-time read-modify-write (each output element is written by exactly
//     one thread, so no cross-CTA race).
//   - pd_f16_gemm_wmma_kernel : the portable first cut (sm_70+, one 64x64 tile,
//     no pipeline). Kept as the correctness reference and the escape hatch -
//     PADDOCK_F16_WMMA=1 forces it for A/B.
//   - pd_f16_gemm_tc5g_kernel : the sm_100a SKINNY arm (batch <= 128 - decode/
//     draft rows, cuBLAS's nvjet-host weak regime). ::1 no-cluster M128 row
//     tiles, per-chunk W+X TMA ring, TMA-store/STG epilogue election, K-split
//     via the shared pd_f16ks_flags protocol. pd_f16_gemm_tc5gp_kernel is its
//     paired-tile variant (two row tiles share one X slab), elected only when
//     the paired grid covers the machine (U0p >= SM count). Against cuBLAS on
//     the same hardware it runs 1.68-1.74x on the whisper head, ties at M72,
//     and wins qkv/M72 at kernel level on laguna (the residual there is
//     launch ramp + boundary, which the PDL/graph serve layer erases).

#pragma once
#include <mma.h>
#include <atomic>
#include <cstdlib>
#include <cuda.h>

// PD_MMA_OK is defined by int8_mma.cuh in the pack blob (this header includes
// after it). Define it here too so the standalone correctness test - which sets
// PD_MMA_OK=1 before the include - and any other include order still compile.
#ifndef PD_MMA_OK
#if defined(__CUDACC__)
#define PD_MMA_OK (__CUDA_ARCH__ >= 800)
#else
#define PD_MMA_OK 0
#endif
#endif

// ---------------------------------------------------------------- wmma fallback
// K=in_dim, M=out_dim, N=batch. W=[M,K] row-major, X=[N,K] row-major,
// Y=[N,M] row-major (== C[M,N] column-major, ld=M). One 16x16 output frag per
// warp; CTA covers BM=WARPS_M*16 out-rows x BN=WARPS_N*16 batch-cols.
template <int WARPS_M, int WARPS_N>
__global__ void __launch_bounds__(WARPS_M* WARPS_N * 32) pd_f16_gemm_wmma_kernel(
        const __half* __restrict__ W, const __half* __restrict__ X,
        float* __restrict__ Y, float beta, uint32_t K, uint32_t M, uint32_t N) {
#if PD_MMA_OK && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    using namespace nvcuda;
    constexpr int TM = 16, TN = 16, TK = 16;
    constexpr int BM = WARPS_M * TM, BN = WARPS_N * TN, BK = TK;
    constexpr int NWARP = WARPS_M * WARPS_N, NTH = NWARP * 32;
    __shared__ __half sA[BM * BK];        // weight tile [out-row][k]
    __shared__ __half sB[BN * BK];        // acts tile   [batch-col][k]
    __shared__ float sC[NWARP][TM * TN];  // epilogue / beta scratch, per warp

    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const uint32_t wm = warp / WARPS_N, wn = warp % WARPS_N;
    const uint32_t m0 = blockIdx.x * BM + wm * TM;  // warp out-row base
    const uint32_t n0 = blockIdx.y * BN + wn * TN;  // warp batch-col base
    const __half zero = __float2half(0.0f);

    wmma::fragment<wmma::accumulator, TM, TN, TK, float> acc;
    if (beta != 0.0f) {
        // guarded load of the Y tile (col-major: Y[n*M+m]) into per-warp scratch
        for (uint32_t i = lane; i < TM * TN; i += 32u) {
            const uint32_t mm = i % TM, nn = i / TM;
            const uint32_t gm = m0 + mm, gn = n0 + nn;
            sC[warp][i] = (gm < M && gn < N) ? Y[(size_t)gn * M + gm] : 0.0f;
        }
        __syncwarp();
        wmma::load_matrix_sync(acc, &sC[warp][0], TM, wmma::mem_col_major);
    } else {
        wmma::fill_fragment(acc, 0.0f);
    }

    wmma::fragment<wmma::matrix_a, TM, TN, TK, __half, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, TM, TN, TK, __half, wmma::col_major> fb;
    for (uint32_t k0 = 0; k0 < K; k0 += BK) {
        for (uint32_t i = tid; i < BM * BK; i += NTH) {
            const uint32_t r = i / BK, kk = i % BK;
            const uint32_t gm = blockIdx.x * BM + r, gk = k0 + kk;
            sA[i] = (gm < M && gk < K) ? W[(size_t)gm * K + gk] : zero;
        }
        for (uint32_t i = tid; i < BN * BK; i += NTH) {
            const uint32_t c = i / BK, kk = i % BK;
            const uint32_t gn = blockIdx.y * BN + c, gk = k0 + kk;
            sB[i] = (gn < N && gk < K) ? X[(size_t)gn * K + gk] : zero;
        }
        __syncthreads();
        wmma::load_matrix_sync(fa, &sA[wm * TM * BK], BK);  // rows[wm*16..], row-major
        wmma::load_matrix_sync(fb, &sB[wn * TN * BK], BK);  // cols[wn*16..], col-major
        wmma::mma_sync(acc, fa, fb, acc);
        __syncthreads();
    }

    wmma::store_matrix_sync(&sC[warp][0], acc, TM, wmma::mem_col_major);
    __syncwarp();
    for (uint32_t i = lane; i < TM * TN; i += 32u) {
        const uint32_t mm = i % TM, nn = i / TM;
        const uint32_t gm = m0 + mm, gn = n0 + nn;
        if (gm < M && gn < N) Y[(size_t)gn * M + gm] = sC[warp][i];
    }
#else
    (void)W; (void)X; (void)Y; (void)beta; (void)K; (void)M; (void)N;
#endif
}

// ---------------------------------------------------------------- mma helpers
// Self-named (pd_f16_*) so this header stays self-contained: the pack blob
// already defines equivalents (pd_mma_ldm_*, pd_af2_mma, pd_attn_cpa_*) but the
// standalone test includes only this file, so it cannot borrow them. All are
// trivial PTX wrappers; the codegen is identical to the pack's own.
#if PD_MMA_OK
__device__ __forceinline__ void pd_f16_ldm_x4(const void* p, uint32_t& r0,
                                              uint32_t& r1, uint32_t& r2,
                                              uint32_t& r3) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(a));
}
__device__ __forceinline__ void pd_f16_ldm_x2(const void* p, uint32_t& r0,
                                              uint32_t& r1) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                 : "=r"(r0), "=r"(r1)
                 : "r"(a));
}
// predicated 16B cp.async: ok==false issues a size-0 copy that zero-fills the
// shared destination without reading gmem (OOB source never dereferenced).
__device__ __forceinline__ void pd_f16_cpa16(void* smem, const void* gmem, bool ok) {
    const unsigned sm = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(sm),
                 "l"(gmem), "r"(ok ? 16u : 0u));
}
__device__ __forceinline__ void pd_f16_cpa_commit() {
    asm volatile("cp.async.commit_group;");
}
template <int N>
__device__ __forceinline__ void pd_f16_cpa_waitN() {
    asm volatile("cp.async.wait_group %0;" ::"n"(N));
}
// D = A*B + D, m16n8k16 f16xf16->f32. A = 4 regs (8 f16), B = 2 regs (4 f16),
// D = 4 f32. Fragment maps are the architectural PTX ones (same as prefill.cuh
// pd_af2_mma): A a0=(m=g,k=2t) a1=(m+8,2t) a2=(m,8+2t) a3=(m+8,8+2t);
// B b0=(n=g,k=2t) b1=(n,8+2t); D d0=(m=g,n=2t) d1=(m,2t+1) d2=(m+8,2t)
// d3=(m+8,2t+1); g=lane/4, t=lane%4.
__device__ __forceinline__ void pd_f16_mma(float d[4], const uint32_t a[4],
                                           const uint32_t b[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
#endif

// ---------------------------------------------------------------- mma GEMM
// Shared-tiled MMA GEMM over the output tile BM out-rows x BN batch-cols with
// NWARP warps. K is staged KT elements per barrier; ST cp.async buffers deep.
// Structure mirrors pd_q8_0_gemm_mma_kernel exactly (the ring math is data
// agnostic), minus Q8_0's per-block int scales - f16 accumulates straight in
// f32. Every stage is zero-padded past M/N/K so ragged tiles and ragged K (not
// a 16-multiple) are correct; the tail int4 of a ragged-K row falls to a scalar
// fill so it never reads a neighbouring row.
// RG x CG = the per-warp register micro-tile (RG row-groups of 16 out-rows x CG
// col-groups of 8 batch-cols). Each k16 step loads RG A-fragments + CG
// B-fragments and issues RG*CG mmas, so every fragment feeds many mmas: the
// mma:ldmatrix ratio is RG*CG/(RG+CG) instead of the strip layout's ~1:1. That
// ratio is what frees the tensor pipe from the LSU/ldmatrix bottleneck (the
// strip kernel ran the LSU as hard as the tensor cores at ~10% of SOL).
template <uint32_t BM, uint32_t BN, uint32_t NWARP, uint32_t ST, uint32_t KT,
          uint32_t RG, uint32_t CG, bool KS = false>
__global__ void __launch_bounds__(NWARP * 32) pd_f16_gemm_mma_kernel(
        const __half* __restrict__ W, const __half* __restrict__ X,
        float* __restrict__ Y, float beta, uint32_t K, uint32_t M, uint32_t N,
        float* __restrict__ Part = nullptr, uint32_t slab = 0u) {
#if PD_MMA_OK
    constexpr uint32_t NTH = NWARP * 32u;
    constexpr uint32_t WM = RG * 16u;      // warp tile rows
    constexpr uint32_t WN = CG * 8u;       // warp tile cols
    constexpr uint32_t WR = BM / WM;       // warp-rows in the CTA tile
    constexpr uint32_t WC = BN / WN;       // warp-cols in the CTA tile
    constexpr uint32_t NSUBK = KT / 16u;   // k16 sub-tiles per stage
    constexpr uint32_t KPAD = KT + 8u;     // padded shared K-stride (halfs)
    constexpr uint32_t H8PR = KT / 8u;     // int4 (8-half) loads per staged row

    // K-split: this CTA owns the KT-aligned slab [k_lo, k_hi) named by
    // blockIdx.z and writes its own partial plane; the combine sums the planes
    // in fixed z order, so the result stays deterministic. slab is a multiple
    // of KT, so only the final slab is ragged and the existing K-tail path in
    // stage() covers it unchanged. Every bound below is k_hi -- the row STRIDES
    // stay K, since the operands are not resliced, only the reduction is.
    const uint32_t k_lo = KS ? blockIdx.z * slab : 0u;
    const uint32_t k_hi = KS ? (K < k_lo + slab ? K : k_lo + slab) : K;
    static_assert(WR * WC == NWARP, "warp grid");
    static_assert(WM * WR == BM && WN * WC == BN, "tile cover");
    static_assert(KT % 16u == 0u, "KT k16-multiple");
    static_assert(ST >= 1u && ST <= 4u, "stage count");

    // dynamic smem: [ST][BM*KPAD] weights then [ST][BN*KPAD] acts. Dynamic (not
    // static) so KT/ST can scale past the 48 KB static window into B200's 228 KB
    // opt-in budget - bytes-in-flight is the sm_100 GEMM lever (int8_mma saga).
    // The 2D pointer casts keep every sh_a[buf][idx] access below unchanged.
    extern __shared__ __align__(16) __half pd_f16_dyn[];
    auto sh_a = reinterpret_cast<__half(*)[BM * KPAD]>(pd_f16_dyn);
    auto sh_b = reinterpret_cast<__half(*)[BN * KPAD]>(pd_f16_dyn + ST * BM * KPAD);

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, t = lane & 3u;
    const uint32_t wr = (warp % WR) * WM;   // warp row base within tile
    const uint32_t wc = (warp / WR) * WN;   // warp col base within tile
    const uint32_t row_base = blockIdx.x * BM;
    const uint32_t col_base = blockIdx.y * BN;
    const __half zero = __float2half(0.0f);

    // stage kt's A/B planes into buffer `buf`. async 16B when ST>=2 (commit at
    // the call site); synchronous int4 stores when ST=1.
    auto stage = [&](uint32_t k0, uint32_t buf) {
        #pragma unroll
        for (uint32_t i = tid; i < BM * H8PR; i += NTH) {
            const uint32_t row = i / H8PR, h8 = (i % H8PR) * 8u, gk = k0 + h8;
            const bool rowok = (row_base + row) < M;
            __half* dst = &sh_a[buf][row * KPAD + h8];
            const __half* src = W + (size_t)(row_base + row) * K + gk;
            if (rowok && gk + 8u <= k_hi) {
                if (ST >= 2u) pd_f16_cpa16(dst, src, true);
                else *reinterpret_cast<int4*>(dst) = *reinterpret_cast<const int4*>(src);
            } else if (!rowok || gk >= k_hi) {
                if (ST >= 2u) pd_f16_cpa16(dst, src, false);
                else *reinterpret_cast<int4*>(dst) = make_int4(0, 0, 0, 0);
            } else {  // ragged K tail: rowok && gk < K < gk+8
                #pragma unroll
                for (uint32_t e = 0; e < 8u; ++e)
                    dst[e] = (gk + e < k_hi) ? W[(size_t)(row_base + row) * K + gk + e] : zero;
            }
        }
        #pragma unroll
        for (uint32_t i = tid; i < BN * H8PR; i += NTH) {
            const uint32_t col = i / H8PR, h8 = (i % H8PR) * 8u, gk = k0 + h8;
            const bool colok = (col_base + col) < N;
            __half* dst = &sh_b[buf][col * KPAD + h8];
            const __half* src = X + (size_t)(col_base + col) * K + gk;
            if (colok && gk + 8u <= k_hi) {
                if (ST >= 2u) pd_f16_cpa16(dst, src, true);
                else *reinterpret_cast<int4*>(dst) = *reinterpret_cast<const int4*>(src);
            } else if (!colok || gk >= k_hi) {
                if (ST >= 2u) pd_f16_cpa16(dst, src, false);
                else *reinterpret_cast<int4*>(dst) = make_int4(0, 0, 0, 0);
            } else {
                #pragma unroll
                for (uint32_t e = 0; e < 8u; ++e)
                    dst[e] = (gk + e < k_hi) ? X[(size_t)(col_base + col) * K + gk + e] : zero;
            }
        }
    };

    // ldmatrix per-lane offsets. A x4 covers a 16x16 tile (rows +0/+8 by lane&8,
    // k +0/+8 by lane&16); B x2 covers an 8x16 tile (cols by lane&7, k +0/+8 by
    // lane&8) - lanes 16+ pass a benign in-range address. a_roff/b_coff are the
    // per-lane row/col within a group; the rg*16 / cg*8 group base is added below.
    const uint32_t l7 = lane & 7u;
    const uint32_t a_roff = ((lane & 8u) ? 8u : 0u) + l7;
    const uint32_t a_kof = (lane & 16u) ? 8u : 0u;
    const uint32_t b_kof = (lane & 8u) ? 8u : 0u;

    float acc[RG][CG][4] = {};
    auto compute = [&](uint32_t buf) {
        #pragma unroll
        for (uint32_t sk = 0; sk < NSUBK; ++sk) {
            const uint32_t ko = sk * 16u;
            // load this warp's RG A-fragments and CG B-fragments once...
            uint32_t a[RG][4];
            #pragma unroll
            for (uint32_t rg = 0; rg < RG; ++rg)
                pd_f16_ldm_x4(&sh_a[buf][(wr + rg * 16u + a_roff) * KPAD + ko + a_kof],
                              a[rg][0], a[rg][1], a[rg][2], a[rg][3]);
            uint32_t b[CG][2];
            #pragma unroll
            for (uint32_t cg = 0; cg < CG; ++cg)
                pd_f16_ldm_x2(&sh_b[buf][(wc + cg * 8u + l7) * KPAD + ko + b_kof],
                              b[cg][0], b[cg][1]);
            // ...then the RG*CG outer-product mmas reuse them from registers
            #pragma unroll
            for (uint32_t rg = 0; rg < RG; ++rg)
                #pragma unroll
                for (uint32_t cg = 0; cg < CG; ++cg)
                    pd_f16_mma(acc[rg][cg], a[rg], b[cg]);
        }
    };

    if (ST >= 2u) {
        // ST-deep ring: buffer p computes while up to ST-1 stages stream in.
        // One commit group per iteration always (empty groups are legal PTX)
        // so the wait immediate stays uniform. Byte-identical to the int8
        // ring this is copied from.
        #pragma unroll
        for (uint32_t s = 0; s < ST - 1u; ++s) {
            const uint32_t k0 = k_lo + s * KT;
            if (k0 < k_hi) stage(k0, s);
            pd_f16_cpa_commit();
        }
        uint32_t p = 0;
        for (uint32_t k0 = k_lo; k0 < k_hi; k0 += KT) {
            const uint32_t pre = k0 + (ST - 1u) * KT;
            if (pre < k_hi) stage(pre, (p + ST - 1u) % ST);
            pd_f16_cpa_commit();
            pd_f16_cpa_waitN<(int)ST - 1>();
            __syncthreads();
            compute(p);
            __syncthreads();
            p = (p + 1u) % ST;
        }
    } else {
        for (uint32_t k0 = k_lo; k0 < k_hi; k0 += KT) {
            stage(k0, 0);
            __syncthreads();
            compute(0);
            __syncthreads();
        }
    }

    // store: element (m=out row, n=batch col) -> Y[n*M + m], beta accumulate.
    // D frag: d0=(m=g,n=2t) d1=(m,2t+1) d2=(m+8,2t) d3=(m+8,2t+1). Each element
    // is written by exactly one thread, so the beta read-modify-write is safe.
    // beta is a UNIFORM branch, never a per-element ternary: the store proves
    // *o dereferenceable, so `beta != 0 ? *o : 0` compiles to an unconditional
    // LDG+FSEL - every store becomes a dependent global round-trip even at
    // beta=0 (measured ~4us on a 6us kernel in the tc5g1 testbed)
    // In KS mode the destination is this z-slab's own plane and there is
    // nothing to accumulate onto -- beta is applied once, by the combine.
    float* __restrict__ out = KS ? (Part + (size_t)blockIdx.z * M * N) : Y;
    const float b = KS ? 0.0f : beta;
    if (b != 0.0f) {
        #pragma unroll
        for (uint32_t rg = 0; rg < RG; ++rg) {
            const uint32_t r0 = row_base + wr + rg * 16u + g;
            const uint32_t r8 = r0 + 8u;
            #pragma unroll
            for (uint32_t cg = 0; cg < CG; ++cg) {
                const uint32_t c0 = col_base + wc + cg * 8u + 2u * t;
                const uint32_t c1 = c0 + 1u;
                if (r0 < M) {
                    if (c0 < N) { float* o = &out[(size_t)c0 * M + r0]; *o = acc[rg][cg][0] + b * *o; }
                    if (c1 < N) { float* o = &out[(size_t)c1 * M + r0]; *o = acc[rg][cg][1] + b * *o; }
                }
                if (r8 < M) {
                    if (c0 < N) { float* o = &out[(size_t)c0 * M + r8]; *o = acc[rg][cg][2] + b * *o; }
                    if (c1 < N) { float* o = &out[(size_t)c1 * M + r8]; *o = acc[rg][cg][3] + b * *o; }
                }
            }
        }
    } else {
        #pragma unroll
        for (uint32_t rg = 0; rg < RG; ++rg) {
            const uint32_t r0 = row_base + wr + rg * 16u + g;
            const uint32_t r8 = r0 + 8u;
            #pragma unroll
            for (uint32_t cg = 0; cg < CG; ++cg) {
                const uint32_t c0 = col_base + wc + cg * 8u + 2u * t;
                const uint32_t c1 = c0 + 1u;
                if (r0 < M) {
                    if (c0 < N) out[(size_t)c0 * M + r0] = acc[rg][cg][0];
                    if (c1 < N) out[(size_t)c1 * M + r0] = acc[rg][cg][1];
                }
                if (r8 < M) {
                    if (c0 < N) out[(size_t)c0 * M + r8] = acc[rg][cg][2];
                    if (c1 < N) out[(size_t)c1 * M + r8] = acc[rg][cg][3];
                }
            }
        }
    }
#else
    (void)W; (void)X; (void)Y; (void)beta; (void)K; (void)M; (void)N;
#endif
}

// ------------------------------------------------- tcgen05 ::2 duo (sm_100a)
// PD_TC5_OK and pd_tc5_sdesc come from ../tma_desc.cuh, included right after
// abi.cuh - i.e. before this file. This segment used to carry its own
// PD_TC5_OK guard (identical condition) and a bit-identical pd_tc5_sdesc,
// because the only definition then lived in dense_fp4_w8.cuh, included after
// this one. The guard was deliberate and correct at the time: an undefined
// macro compiles `#if` bodies away silently rather than failing (the pf5
// lesson). Hoisting the definition is what makes the copy unnecessary.

// Persistent ::2 duo GEMM. One cluster pair per 2 SMs walking contiguous
// 256-row x 512-col tile GROUPS (cols inner, so the pair's W k-slabs stay
// L2-hot); per group, two col-adjacent 256x256 tiles run concurrently: the
// two independent D accumulation chains interleave in the tensor pipe (the
// single-chain loop measures 332ns/slab solo vs the 281ns mma floor - one
// K-chain cannot absorb the slot-recycle latency) and the W slab stages once
// for both (48KB/CTA-slab feeds 2x8.39 MF of pair work). Probed ::2 contract
// (reconfirmed bit-exact for kind::f16 here):
//   - A: rank-local compact 128-row halves; B: global N=256, each rank
//     staging its own canonical 128-col tile
//   - D lanes land XOR-64 within each rank (epilogue relabels lane l -> l^64)
// The epilogue is exposed per group (D fills all 512 tmem cols - no
// ping-pong); beta is a store-time read-modify-write as in the mma twin.
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 900)
template <uint32_t S>
__global__ void __launch_bounds__(320) __cluster_dims__(2, 1, 1)
pd_f16_gemm_tc5d_kernel(
    const __grid_constant__ PdTmap wmap, const __grid_constant__ PdTmap xmap,
    float* __restrict__ y, float beta, uint32_t in_dim, uint32_t out_dim,
    uint32_t batch) {
#if PD_TC5_OK
    extern __shared__ __align__(1024) unsigned char pd_f16t_sh[];
    unsigned char* wt = pd_f16t_sh;                    // S x 16KB W (both tiles)
    unsigned char* xa = pd_f16t_sh + S * 16384u;       // S x 16KB X col half, tile A
    unsigned char* xb = pd_f16t_sh + 2u * S * 16384u;  // S x 16KB X col half, tile B
    uint64_t* bfull  = (uint64_t*)(pd_f16t_sh + 3u * S * 16384u);
    uint64_t* bempty = bfull + S;
    uint64_t* bpeer  = bempty + S;
    uint64_t* tfull  = bpeer + S;    // [1] group D ready
    uint64_t* tempty = tfull + 1;    // [1] group D drained (16 arrivals)
    __shared__ uint32_t tmem_slot[1];

    const uint32_t tid = threadIdx.x;
    uint32_t crank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(crank));
    const uint32_t nk = (in_dim * 2u + 127u) / 128u;   // 128B K-slabs = 64 halves
    const uint32_t row_tiles = (out_dim + 127u) >> 7;
    const uint32_t row_pairs = (row_tiles + 1u) >> 1;
    const uint32_t n_cols = (batch + 255u) >> 8;
    const uint32_t T = row_pairs * n_cols;
    const uint32_t n_clusters = gridDim.x >> 1;
    const uint32_t cid = blockIdx.x >> 1;
    const uint32_t per = (T + n_clusters - 1u) / n_clusters;
    const uint32_t t0 = cid * per < T ? cid * per : T;
    const uint32_t t1 = t0 + per < T ? t0 + per : T;

    if (tid == 0) {
        #pragma unroll
        for (uint32_t s = 0; s < S; ++s) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bfull[s])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bempty[s])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bpeer[s])));
        }
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                     ::"r"((uint32_t)__cvta_generic_to_shared(&tfull[0])));
        // 16 = 8 epilogue warps x 2 ranks, all arriving at the leader
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 16;"
                     ::"r"((uint32_t)__cvta_generic_to_shared(&tempty[0])));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();
    asm volatile("barrier.cluster.arrive;");
    asm volatile("barrier.cluster.wait;");
    if (tid < 32)
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], 512;"
                     ::"r"((uint32_t)__cvta_generic_to_shared(tmem_slot)));
    __syncthreads();
    const uint32_t tmem = tmem_slot[0];

    auto bar_wait = [&](uint64_t* bar, uint32_t parity) {
        const uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
        asm volatile("{\n\t.reg .pred p;\nW%=:\n\t"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                     "@!p bra W%=;\n\t}" ::"r"(a), "r"(parity));
    };
    // back-off variant for the heavy spinners (producer/watcher/epilogue)
    auto bar_wait_slow = [&](uint64_t* bar, uint32_t parity) {
        const uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
        asm volatile("{\n\t.reg .pred p;\nW%=:\n\t"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                     "@p bra D%=;\n\t"
                     "nanosleep.u32 128;\n\t"
                     "bra W%=;\nD%=:\n\t}" ::"r"(a), "r"(parity));
    };
    auto peer_addr = [&](void* p) -> uint32_t {
        const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
        uint32_t pa;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                     : "=r"(pa) : "r"(a), "r"(crank ^ 1u));
        return pa;
    };
    // tiles (t, t+1) fuse into a duo group when same row pair and both owned
    auto duo_of = [&](uint32_t t) -> bool {
        return t + 1u < t1 && (t / n_cols) == ((t + 1u) / n_cols);
    };

    if (tid == 0) {
        // producer: continuous TMA ring across every (group, kt)
        uint32_t n = 0, eph = 0;
        for (uint32_t t = t0; t < t1; t += duo_of(t) ? 2u : 1u) {
            const bool duo = duo_of(t);
            const uint32_t pair = t / n_cols;
            const uint32_t col = t % n_cols;
            const uint32_t row_base = pair * 256u + crank * 128u;
            const uint32_t ca = col * 256u + crank * 128u;  // own B col half
            const uint32_t cb = ca + 256u;
            for (uint32_t kt = 0; kt < nk; ++kt, ++n) {
                const uint32_t s = n % S;
                if (n >= S) { bar_wait_slow(&bempty[s], (eph >> s) & 1u); eph ^= 1u << s; }
                const uint32_t m = (uint32_t)__cvta_generic_to_shared(&bfull[s]);
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                             ::"r"(m), "r"(duo ? 49152u : 32768u));
                const int ck = (int)(kt * 128u);
                asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                             " [%0], [%1, {%2, %3}], [%4];"
                             ::"r"((uint32_t)__cvta_generic_to_shared(wt + s * 16384u)),
                               "l"(&wmap), "r"(ck), "r"((int)row_base), "r"(m) : "memory");
                asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                             " [%0], [%1, {%2, %3}], [%4];"
                             ::"r"((uint32_t)__cvta_generic_to_shared(xa + s * 16384u)),
                               "l"(&xmap), "r"(ck), "r"((int)ca), "r"(m) : "memory");
                if (duo)
                    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                                 " [%0], [%1, {%2, %3}], [%4];"
                                 ::"r"((uint32_t)__cvta_generic_to_shared(xb + s * 16384u)),
                                   "l"(&xmap), "r"(ck), "r"((int)cb), "r"(m) : "memory");
            }
        }
    } else if (tid == 32 && crank == 1u) {
        // rank1 watcher: forward slab-ready to the leader
        uint32_t fph = 0, total = 0;
        for (uint32_t t = t0; t < t1; t += duo_of(t) ? 2u : 1u) total += nk;
        for (uint32_t n = 0; n < total; ++n) {
            const uint32_t s = n % S;
            bar_wait_slow(&bfull[s], (fph >> s) & 1u); fph ^= 1u << s;
            asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];"
                         ::"r"(peer_addr(&bpeer[s])) : "memory");
        }
    } else if (tid == 32 && crank == 0u) {
        // issuer (leader only)
        uint32_t n = 0, fph = 0, pph = 0, g = 0;
        for (uint32_t t = t0; t < t1; t += duo_of(t) ? 2u : 1u, ++g) {
            const bool duo = duo_of(t);
            if (g >= 1u) bar_wait(&tempty[0], (g - 1u) & 1u);
            asm volatile("tcgen05.fence::after_thread_sync;");
            for (uint32_t kt = 0; kt < nk; ++kt, ++n) {
                const uint32_t s = n % S;
                bar_wait(&bfull[s], (fph >> s) & 1u); fph ^= 1u << s;
                bar_wait(&bpeer[s], (pph >> s) & 1u); pph ^= 1u << s;
                const uint32_t w16 = (uint32_t)__cvta_generic_to_shared(wt + s * 16384u) >> 4;
                const uint32_t a16 = (uint32_t)__cvta_generic_to_shared(xa + s * 16384u) >> 4;
                const uint32_t b16 = (uint32_t)__cvta_generic_to_shared(xb + s * 16384u) >> 4;
                const uint32_t id = (1u << 4) | ((256u >> 3) << 17) | ((256u >> 4) << 24);
                #pragma unroll
                for (uint32_t kb = 0; kb < 4u; ++kb) {  // 4 x 32B chunks (K=16 each)
                    const uint64_t ad = pd_tc5_sdesc(w16 + kb * 2u);
                    const uint32_t en = (kt > 0 || kb > 0) ? 1u : 0u;
                    asm volatile("{\n\t.reg .pred p;\n\t"
                                 "setp.ne.b32 p, %4, 0;\n\t"
                                 "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n\t}"
                                 ::"r"(tmem), "l"(ad),
                                   "l"(pd_tc5_sdesc(a16 + kb * 2u)), "r"(id), "r"(en));
                    if (duo)
                        asm volatile("{\n\t.reg .pred p;\n\t"
                                     "setp.ne.b32 p, %4, 0;\n\t"
                                     "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n\t}"
                                     ::"r"(tmem + 256u), "l"(ad),
                                       "l"(pd_tc5_sdesc(b16 + kb * 2u)), "r"(id), "r"(en));
                }
                asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                             ::"r"((uint32_t)__cvta_generic_to_shared(&bempty[s])));
                asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                             ::"r"(peer_addr(&bempty[s])));
            }
            asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&tfull[0])));
            asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                         ::"r"(peer_addr(&tfull[0])));
        }
    } else if (tid >= 64) {
        // epilogue: 8 warps (4 row bands x 2 col halves), own-rank D rows
        const uint32_t ewarp = (tid - 64u) >> 5, lane = tid & 31u;
        const uint32_t warp = ewarp & 3u, chalf = ewarp >> 2;
        uint32_t g = 0;
        for (uint32_t t = t0; t < t1; t += duo_of(t) ? 2u : 1u, ++g) {
            const bool duo = duo_of(t);
            bar_wait_slow(&tfull[0], g & 1u);
            asm volatile("tcgen05.fence::after_thread_sync;");
            const uint32_t pair = t / n_cols;
            // ::2 D lanes land 64-half-swapped within the rank
            const uint32_t rl = ((warp * 32u + lane) ^ 64u) & 127u;
            const uint32_t row = pair * 256u + crank * 128u + rl;
            #pragma unroll
            for (uint32_t dt = 0; dt < 2u; ++dt) {
                if (dt == 1u && !duo) break;
                const uint32_t col0 = (t % n_cols + dt) * 256u;
                #pragma unroll
                for (uint32_t ci = 0; ci < 4u; ++ci) {
                    const uint32_t cc = chalf * 4u + ci;
                    uint32_t r[32];
                    const uint32_t taddr = tmem + dt * 256u + ((warp * 32u) << 16) + cc * 32u;
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
                        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
                        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
                        : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
                          "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),"=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15]),
                          "=r"(r[16]),"=r"(r[17]),"=r"(r[18]),"=r"(r[19]),"=r"(r[20]),"=r"(r[21]),"=r"(r[22]),"=r"(r[23]),
                          "=r"(r[24]),"=r"(r[25]),"=r"(r[26]),"=r"(r[27]),"=r"(r[28]),"=r"(r[29]),"=r"(r[30]),"=r"(r[31])
                        : "r"(taddr));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    if (row < out_dim) {
                        // uniform beta branch - the ternary form compiles to
                        // an unconditional LDG+FSEL RMW (see mma twin note)
                        if (beta != 0.0f) {
                            #pragma unroll
                            for (uint32_t j = 0; j < 32u; ++j) {
                                const uint32_t c = col0 + cc * 32u + j;
                                if (c < batch) {
                                    float* o = y + (size_t)c * out_dim + row;
                                    *o = __uint_as_float(r[j]) + beta * *o;
                                }
                            }
                        } else {
                            #pragma unroll
                            for (uint32_t j = 0; j < 32u; ++j) {
                                const uint32_t c = col0 + cc * 32u + j;
                                if (c < batch)
                                    y[(size_t)c * out_dim + row] =
                                        __uint_as_float(r[j]);
                            }
                        }
                    }
                }
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            if (lane == 0) {
                if (crank == 0u)
                    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
                                 ::"r"((uint32_t)__cvta_generic_to_shared(&tempty[0])) : "memory");
                else
                    asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];"
                                 ::"r"(peer_addr(&tempty[0])) : "memory");
            }
        }
    }
    __syncthreads();
    asm volatile("barrier.cluster.arrive;");
    asm volatile("barrier.cluster.wait;");
    if (tid < 32)
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, 512;" ::"r"(tmem));
#else
    (void)wmap; (void)xmap; (void)y; (void)beta;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}
#endif  // __cluster_dims__ guard (host pass || sm_90+)

// -------------------------------------- tcgen05 ::2 ping-pong wide (sm_100a)
// The duo's exposed epilogue and the store-vs-chain interference ladder
// shaped this second tcgen05
// arm: one 256xNT tile per cluster whose D ping-pongs across 2x256 tmem cols
// (drain hides behind the next unit's chain), 2-slab ring slots loaded with
// one 3D-box TMA per operand (8 mmas per wait/commit pair - the issuer-loop
// amortization the duo got from its second tile), and a staged TMA-store
// epilogue (st.global retirement measurably poisons the chain's SMSP; STS +
// cp.async.bulk store does not). NT elects 256/192 (attn-class wave packing);
// KS>1 splits each tile's K-chain into KS wave-packing units at SLOT (2-slab)
// granularity with a STRIDED unit walk: slice 0 stores beta*y+P0 (or reduces
// at beta=1) then releases pd_f16ks_flags[t]; slices >0 spin-acquire and
// cp.reduce their partials; the last consumer leader self-cleans the flag, so
// the array needs no per-launch zeroing. Requires in_dim%64 (3D-box K rule),
// out_dim%4 (ymap 16B gstride), beta in {0,1} (reduce path).
__device__ uint32_t pd_f16ks_flags[4096];

#if PD_TC5P_STAMPS
// F-decomposition probe instrumentation: 8 globaltimer stamps per
// cluster, crank-0 threads only, last launch wins. Gated - production builds
// compile without this and are bit-identical. Exit-adjacent stamps carry a
// "memory" clobber (banked lesson: SASS hoists bare timer reads over barriers).
__device__ unsigned long long pd_tc5p_stamps[128 * 8];
#define PD_TC5P_STAMP(cid_, sl_) do { \
    unsigned long long t_; \
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_) :: "memory"); \
    if ((cid_) < 128u) pd_tc5p_stamps[(cid_) * 8u + (sl_)] = t_; } while (0)
#else
#define PD_TC5P_STAMP(cid_, sl_) do {} while (0)
#endif

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 900)
template <uint32_t S, uint32_t KS, uint32_t NT>
__global__ void __launch_bounds__(320) __cluster_dims__(2, 1, 1)
pd_f16_gemm_tc5p_kernel(
    const __grid_constant__ PdTmap wmap, const __grid_constant__ PdTmap xmap,
    const __grid_constant__ PdTmap ymap, float beta, uint32_t in_dim,
    uint32_t out_dim, uint32_t batch) {
#if PD_TC5_OK
    constexpr uint32_t XSL = NT * 128u;        // X bytes per 2-slab slot/rank
    extern __shared__ __align__(1024) unsigned char pd_f16p_sh[];
    unsigned char* wt = pd_f16p_sh;            // S x 32KB own-rank W, 2 slabs
    unsigned char* xt = pd_f16p_sh + S * 32768u;  // S x XSL own-rank X
    uint64_t* bfull  = (uint64_t*)(pd_f16p_sh + S * (32768u + XSL));
    uint64_t* bempty = bfull + S;
    uint64_t* bpeer  = bempty + S;
    uint64_t* tfull  = bpeer + S;    // [2] per ping-pong D buffer
    uint64_t* tempty = tfull + 2;    // [2] 16 arrivals each
    __shared__ uint32_t tmem_slot[1];

    const uint32_t tid = threadIdx.x;
    uint32_t crank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(crank));
    if (tid == 32u && crank == 0u) PD_TC5P_STAMP(blockIdx.x >> 1, 0);
    const uint32_t nk = (in_dim * 2u + 127u) / 128u;
    const uint32_t row_tiles = (out_dim + 127u) >> 7;
    const uint32_t row_pairs = (row_tiles + 1u) >> 1;
    const uint32_t n_cols = (batch + NT - 1u) / NT;
    const uint32_t U = row_pairs * n_cols * KS;
    // K splits at SLOT granularity (2 slabs): a partial slot may only sit at
    // the true K tail, where TMA OOB zero-fill makes the phantom slab exact
    const uint32_t nslot_all = (nk + 1u) >> 1;
    const uint32_t kbase = nslot_all / KS, krem = nslot_all % KS;
    const uint32_t n_clusters = gridDim.x >> 1;
    const uint32_t cid = blockIdx.x >> 1;
    // unit walk: contiguous chunk at KS==1 (W stays L2-hot across the col
    // sweep); strided at KS>1 (a consumer and its slice-0 must share a round
    // or chunk boundaries cascade-serialize whole clusters)
    uint32_t ucount, ustart, ustep;
    if (KS > 1u) {
        ustart = cid; ustep = n_clusters;
        ucount = cid < U ? (U - 1u - cid) / n_clusters + 1u : 0u;
    } else {
        const uint32_t per = (U + n_clusters - 1u) / n_clusters;
        ustart = cid * per < U ? cid * per : U;
        const uint32_t u1 = ustart + per < U ? ustart + per : U;
        ucount = u1 - ustart; ustep = 1u;
    }

    if (tid == 0) {
        #pragma unroll
        for (uint32_t s = 0; s < S; ++s) {
            // rank0 bfull merges peer readiness: own TMA arrive.expect_tx +
            // the rank1 watcher's remote arrive (was a separate bpeer wait -
            // the second per-slot spin on the issuer's critical path)
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bfull[s])),
                           "r"(crank == 0u ? 2u : 1u));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bempty[s])));
        }
        #pragma unroll
        for (uint32_t b = 0; b < 2u; ++b) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&tfull[b])));
            // 16 = 8 epilogue warps x 2 ranks, all arriving at the leader
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 16;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&tempty[b])));
        }
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();
    asm volatile("barrier.cluster.arrive;");
    asm volatile("barrier.cluster.wait;");
    if (tid < 32)
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], 512;"
                     ::"r"((uint32_t)__cvta_generic_to_shared(tmem_slot)));
    __syncthreads();
    const uint32_t tmem = tmem_slot[0];

    auto bar_wait = [&](uint64_t* bar, uint32_t parity) {
        const uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
        asm volatile("{\n\t.reg .pred p;\nW%=:\n\t"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                     "@!p bra W%=;\n\t}" ::"r"(a), "r"(parity));
    };
    auto bar_wait_slow = [&](uint64_t* bar, uint32_t parity) {
        const uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
        asm volatile("{\n\t.reg .pred p;\nW%=:\n\t"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                     "@p bra D%=;\n\t"
                     "nanosleep.u32 128;\n\t"
                     "bra W%=;\nD%=:\n\t}" ::"r"(a), "r"(parity));
    };
    auto peer_addr = [&](void* p) -> uint32_t {
        const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
        uint32_t pa;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                     : "=r"(pa) : "r"(a), "r"(crank ^ 1u));
        return pa;
    };
    auto slice_k = [&](uint32_t slice, uint32_t& ks0, uint32_t& kcnt) {
        const uint32_t ss0 = slice * kbase + (slice < krem ? slice : krem);
        const uint32_t scnt = kbase + (slice < krem ? 1u : 0u);
        ks0 = ss0 * 2u;
        kcnt = ks0 + scnt * 2u > nk ? nk - ks0 : scnt * 2u;
    };

    if (tid == 0) {
        // producer: one 3D-box TMA per operand per 2-slab slot
        uint32_t n = 0, eph = 0;
        for (uint32_t k = 0; k < ucount; ++k) {
            const uint32_t u = ustart + k * ustep;
            const uint32_t t = u / KS;
            const uint32_t pair = t / n_cols, col = t % n_cols;
            const uint32_t row_base = pair * 256u + crank * 128u;
            const uint32_t cb = col * NT + crank * (NT >> 1);
            uint32_t ks0, kcnt; slice_k(u % KS, ks0, kcnt);
            const uint32_t nslot = (kcnt + 1u) >> 1;
            for (uint32_t j = 0; j < nslot; ++j, ++n) {
                const uint32_t s = n % S;
                if (n >= S) { bar_wait_slow(&bempty[s], (eph >> s) & 1u); eph ^= 1u << s; }
                const uint32_t m = (uint32_t)__cvta_generic_to_shared(&bfull[s]);
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                             ::"r"(m), "r"(32768u + XSL));
                const int ck = (int)(ks0 + 2u * j);
                asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes"
                             " [%0], [%1, {0, %2, %3}], [%4];"
                             ::"r"((uint32_t)__cvta_generic_to_shared(wt + s * 32768u)),
                               "l"(&wmap), "r"((int)row_base), "r"(ck), "r"(m) : "memory");
                asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes"
                             " [%0], [%1, {0, %2, %3}], [%4];"
                             ::"r"((uint32_t)__cvta_generic_to_shared(xt + s * XSL)),
                               "l"(&xmap), "r"((int)cb), "r"(ck), "r"(m) : "memory");
            }
        }
    } else if (tid == 32 && crank == 1u) {
        // rank1 watcher: forward slot-ready to the leader (remote-mbarrier TMA
        // completion is not deliverable across ranks - probed, it hangs).
        // Arrives the LEADER'S bfull[s] (count 2 there), not a separate bpeer.
        uint32_t fph = 0, total = 0;
        for (uint32_t k = 0; k < ucount; ++k) {
            uint32_t ks0, kcnt; slice_k((ustart + k * ustep) % KS, ks0, kcnt);
            total += (kcnt + 1u) >> 1;
        }
        for (uint32_t n = 0; n < total; ++n) {
            const uint32_t s = n % S;
            // tight spin: the forward is on the issuer's critical path now
            // (bfull count 2) - nanosleep wake quantization costs real slope
            bar_wait(&bfull[s], (fph >> s) & 1u); fph ^= 1u << s;
            asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];"
                         ::"r"(peer_addr(&bfull[s])) : "memory");
        }
    } else if (tid == 32 && crank == 0u) {
        // issuer: 8 mmas per slot; a K-tail phantom slab is zero-filled by
        // the TMA OOB rule -> harmless zero accumulate. One bar_wait and one
        // multicast commit per slot: two commits/slot exhausted the HW's
        // outstanding-watermark depth and drained the tensor pipe at every
        // slot boundary (+214ns/slot vs nvjet's +40 - round-6 chain-law fit;
        // ring depth S=4 measured a null, see attack ledger P19/P26).
        PD_TC5P_STAMP(blockIdx.x >> 1, 1);
        uint32_t n = 0, fph = 0, teph = 0, tc = 0;
        for (uint32_t k = 0; k < ucount; ++k, ++tc) {
            const uint32_t u = ustart + k * ustep;
            const uint32_t buf = tc & 1u;
            uint32_t ks0, kcnt; slice_k(u % KS, ks0, kcnt);
            const uint32_t nslot = (kcnt + 1u) >> 1;
            if (tc >= 2u) { bar_wait(&tempty[buf], (teph >> buf) & 1u); teph ^= 1u << buf; }
            asm volatile("tcgen05.fence::after_thread_sync;");
            for (uint32_t j = 0; j < nslot; ++j, ++n) {
                const uint32_t s = n % S;
                bar_wait(&bfull[s], (fph >> s) & 1u); fph ^= 1u << s;
                if (n == 0u) PD_TC5P_STAMP(blockIdx.x >> 1, 2);
                const uint32_t w16 = (uint32_t)__cvta_generic_to_shared(wt + s * 32768u) >> 4;
                const uint32_t x16 = (uint32_t)__cvta_generic_to_shared(xt + s * XSL) >> 4;
                const uint32_t id = (1u << 4) | ((NT >> 3) << 17) | ((256u >> 4) << 24);
                // One asm block for the whole 8-mma burst: eight separate
                // statements each got their own ptxas ELECT/reconverge guard
                // plus hi-field rematerialization (~100 uniform instr per
                // 282ns slot - the measured +128ns/slot, see SASS audit).
                // sdesc is linear in the addr field, so descriptors advance
                // by pure add.u64: W +2 per k16, +1024 per row-half group;
                // X +2 per k16, +NT*4 per group. Same order/operands/enables
                // as the unrolled loop -> bit-exact.
                const uint64_t ad0 = pd_tc5_sdesc(w16);
                const uint64_t bd0 = pd_tc5_sdesc(x16);
                asm volatile(
                    "{\n\t.reg .pred p, q;\n\t.reg .b64 a, b;\n\t"
                    "setp.ne.b32 p, %4, 0;\n\t"
                    "setp.eq.b32 q, 0, 0;\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n\t"
                    "add.u64 a, %1, 2;\n\t"
                    "add.u64 b, %2, 2;\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], a, b, %3, q;\n\t"
                    "add.u64 a, a, 2;\n\t"
                    "add.u64 b, b, 2;\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], a, b, %3, q;\n\t"
                    "add.u64 a, a, 2;\n\t"
                    "add.u64 b, b, 2;\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], a, b, %3, q;\n\t"
                    "add.u64 a, %1, 1024;\n\t"
                    "add.u64 b, %2, %5;\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], a, b, %3, q;\n\t"
                    "add.u64 a, a, 2;\n\t"
                    "add.u64 b, b, 2;\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], a, b, %3, q;\n\t"
                    "add.u64 a, a, 2;\n\t"
                    "add.u64 b, b, 2;\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], a, b, %3, q;\n\t"
                    "add.u64 a, a, 2;\n\t"
                    "add.u64 b, b, 2;\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], a, b, %3, q;\n\t}"
                    ::"r"(tmem + buf * 256u), "l"(ad0), "l"(bd0), "r"(id),
                      "r"(j), "n"((uint64_t)(NT * 4u)));
                asm volatile(
                    "tcgen05.commit.cta_group::2.mbarrier::arrive::one"
                    ".shared::cluster.multicast::cluster.b64 [%0], %1;"
                    ::"r"((uint32_t)__cvta_generic_to_shared(&bempty[s])),
                      "h"((unsigned short)3u));
                if (n == 0u) PD_TC5P_STAMP(blockIdx.x >> 1, 3);
            }
            asm volatile(
                "tcgen05.commit.cta_group::2.mbarrier::arrive::one"
                ".shared::cluster.multicast::cluster.b64 [%0], %1;"
                ::"r"((uint32_t)__cvta_generic_to_shared(&tfull[buf])),
                  "h"((unsigned short)3u));
        }
        PD_TC5P_STAMP(blockIdx.x >> 1, 4);
    } else if (tid >= 64) {
        // epilogue: 8 warps = 4 row bands x 2 col halves of own-rank D.
        // NS = NT/32 slices of 16 cols, double-buffered 8KB staging, the next
        // slice's tcgen05.ld issued while this one stages -> a short pipelined
        // burst whose TMA stores fly during the next unit's chain.
        const uint32_t ewarp = (tid - 64u) >> 5, lane = tid & 31u;
        const uint32_t warp = ewarp & 3u, chalf = ewarp >> 2;
        const uint32_t rl = ((warp * 32u + lane) ^ 64u) & 127u;
        constexpr uint32_t NS = NT / 32u;
        unsigned char* stage0 = pd_f16p_sh + S * (32768u + XSL) + 1024u + chalf * 16384u;
        const bool leader = (warp == 0u && lane == 0u);
        uint32_t tc = 0, gsl = 0;
        for (uint32_t k = 0; k < ucount; ++k, ++tc) {
            const uint32_t u = ustart + k * ustep;
            const uint32_t buf = tc & 1u;
            const uint32_t t = u / KS, slice = u % KS;
            bar_wait_slow(&tfull[buf], (tc >> 1) & 1u);
            asm volatile("tcgen05.fence::after_thread_sync;");
            if (tid == 64u && crank == 0u && k + 1u == ucount)
                PD_TC5P_STAMP(blockIdx.x >> 1, 5);
            const uint32_t pair = t / n_cols;
            const uint32_t row0 = pair * 256u + crank * 128u;
            const uint32_t col0 = (t % n_cols) * NT + chalf * (NT >> 1);
            // reduce (accumulate into y) unless this is the beta=0 base store
            const bool red = (KS > 1u && slice > 0u) || beta != 0.0f;
            if (KS > 1u && slice > 0u && leader) {
                // partial consumer: acquire slice 0's base store (4 leaders)
                uint32_t v;
                do {
                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
                                 : "=r"(v) : "l"(pd_f16ks_flags + t) : "memory");
                    if ((v & 0xffu) >= 4u) break;
                    asm volatile("nanosleep.u32 64;");
                } while (true);
                asm volatile("fence.proxy.async.global;");
            }
            auto tld = [&](uint32_t si, uint32_t* r) {
                const uint32_t taddr = tmem + buf * 256u + ((warp * 32u) << 16)
                                     + chalf * (NT >> 1) + si * 16u;
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                    "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                    : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),
                      "=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
                      "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),
                      "=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15])
                    : "r"(taddr));
            };
            {
                uint32_t ra[16], rb[16];
                tld(0, ra);
                #pragma unroll 1
                for (uint32_t si = 0; si < NS; ++si, ++gsl) {
                    // staging reuse gate: same-parity TMA two slices back has
                    // read its buffer once <=1 group remains outstanding
                    if (leader && gsl >= 2u)
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                    asm volatile("bar.sync %0, 128;" ::"r"(1u + chalf));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    uint32_t* r = (si & 1u) ? rb : ra;
                    if (si + 1u < NS) tld(si + 1u, (si & 1u) ? ra : rb);
                    unsigned char* stg = stage0 + (gsl & 1u) * 8192u;
                    const uint32_t sb = (uint32_t)__cvta_generic_to_shared(stg) + rl * 4u;
                    #pragma unroll
                    for (uint32_t jj = 0; jj < 16u; ++jj)
                        asm volatile("st.shared.b32 [%0], %1;"
                                     ::"r"(sb + jj * 512u), "r"(r[jj]) : "memory");
                    asm volatile("fence.proxy.async.shared::cta;");
                    asm volatile("bar.sync %0, 128;" ::"r"(1u + chalf));
                    if (leader) {
                        const int cr = (int)row0, cc = (int)(col0 + si * 16u);
                        const uint32_t ss = (uint32_t)__cvta_generic_to_shared(stg);
                        if (red)
                            asm volatile("cp.reduce.async.bulk.tensor.2d.global.shared::cta"
                                         ".add.tile.bulk_group [%0, {%1, %2}], [%3];"
                                         ::"l"(&ymap), "r"(cr), "r"(cc), "r"(ss) : "memory");
                        else
                            asm volatile("cp.async.bulk.tensor.2d.global.shared::cta"
                                         ".tile.bulk_group [%0, {%1, %2}], [%3];"
                                         ::"l"(&ymap), "r"(cr), "r"(cc), "r"(ss) : "memory");
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                }
            }
            if (tid == 64u && crank == 0u && k + 1u == ucount)
                PD_TC5P_STAMP(blockIdx.x >> 1, 6);
            if (KS > 1u && leader) {
                asm volatile("cp.async.bulk.wait_group 0;");
                asm volatile("fence.proxy.async.global;");
                if (slice == 0u) {
                    asm volatile("red.release.gpu.global.add.u32 [%0], 1;"
                                 ::"l"(pd_f16ks_flags + t) : "memory");
                } else {
                    // done-mark; the last consumer leader self-cleans the flag
                    uint32_t old;
                    asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 256;"
                                 : "=r"(old) : "l"(pd_f16ks_flags + t) : "memory");
                    if ((old >> 8) + 1u == (KS - 1u) * 4u)
                        asm volatile("st.relaxed.gpu.global.u32 [%0], 0;"
                                     ::"l"(pd_f16ks_flags + t) : "memory");
                }
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            if (lane == 0) {
                if (crank == 0u)
                    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
                                 ::"r"((uint32_t)__cvta_generic_to_shared(&tempty[buf])) : "memory");
                else
                    asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];"
                                 ::"r"(peer_addr(&tempty[buf])) : "memory");
            }
        }
        if (leader)
            asm volatile("cp.async.bulk.wait_group 0;");
        if (tid == 64u && crank == 0u) PD_TC5P_STAMP(blockIdx.x >> 1, 7);
    }
    __syncthreads();
    asm volatile("barrier.cluster.arrive;");
    asm volatile("barrier.cluster.wait;");
    if (tid < 32)
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, 512;" ::"r"(tmem));
#else
    (void)wmap; (void)xmap; (void)ymap; (void)beta;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}
#endif  // __cluster_dims__ guard (host pass || sm_90+)

// ---------------------------------------------------------------- tc5g skinny
// ::1 NO-CLUSTER skinny arm (batch <= 128 - decode/draft rows, cuBLAS's nvjet
// host path costs 4.3-8.3us there). Single-CTA M128 row tiles, S-deep per-
// chunk W+X TMA ring, one tcgen05 ::1 K-chain per tile (the mma side is fully
// hidden behind the TMA stream - measured with a ring-only probe), tmem
// alloc 2*nto cols + relinquish so co-resident CTAs stream independently.
// Epilogue elects per launch: TMA path (out%4==0 && beta in {0,1}: STS 16-col
// slices -> cp.async.bulk / cp.reduce, self-cleaning pd_f16ks_flags K-split
// protocol, thr=1) or STG path (any beta, UNIFORM beta branch - never the
// ternary RMW). Census against cuBLAS (warm/cold): whisper head 1.68/1.74,
// laguna M72 0.95-1.00, qkv 0.66, laguna r16-r128
// 0.46-0.69 (kernel-level span already wins qkv/M72; the residual is launch
// ramp+boundary the PDL/graph serve layer erases while cuBLAS keeps paying
// its dispatch).
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 900)
template <uint32_t S>
__global__ void __launch_bounds__(192) pd_f16_gemm_tc5g_kernel(
    const __grid_constant__ PdTmap wmap, const __grid_constant__ PdTmap xmap,
    const __grid_constant__ PdTmap ymap,
    float* __restrict__ y, float beta,
    uint32_t in_dim, uint32_t out_dim, uint32_t batch, uint32_t nto,
    uint32_t KS, uint32_t ncols) {
#if PD_TC5_OK
    extern __shared__ __align__(1024) unsigned char shG[];
    // stage smem exists only when the TMA epilogue is elected - the STG path
    // (e.g. whisper's out%4!=0 head) keeps the smaller footprint and its
    // 3-CTA/SM single-wave fit
    const bool useTma = ((out_dim & 3u) == 0u) &&
                        (beta == 0.0f || beta == 1.0f) &&
                        (((uintptr_t)y & 15u) == 0u);  // ym box needs 16B rows
    unsigned char* wt = shG;                      // S x 16KB W
    unsigned char* xt = shG + S * 16384u;         // S x nto*128 X
    unsigned char* stage = xt + S * (size_t)(nto * 128u);   // 2 x 8KB Y staging
    uint64_t* bfull  = (uint64_t*)(stage + (useTma ? 16384u : 0u));
    uint64_t* bempty = bfull + S;
    uint64_t* tfull  = bempty + S;   // [2] tile ping-pong
    uint64_t* tempty = tfull + 2;    // [2] 4 arrivals
    __shared__ uint32_t tmem_slot[1];

    const uint32_t tid = threadIdx.x;
    const uint32_t nk = (in_dim * 2u + 127u) / 128u;
    const uint32_t xsl = nto * 128u;              // X bytes per slot
    const uint32_t row_tiles = (out_dim + 127u) >> 7;
    const uint32_t U = row_tiles * KS;
    const uint32_t kbase = nk / KS, krem = nk % KS;
    const uint32_t n_ctas = gridDim.x;
    const uint32_t cid = blockIdx.x;
    uint32_t ucount, ustart, ustep;
    if (KS > 1u) {
        ustart = cid; ustep = n_ctas;
        ucount = cid < U ? (U - 1u - cid) / n_ctas + 1u : 0u;
    } else {
        const uint32_t per = (U + n_ctas - 1u) / n_ctas;
        ustart = cid * per < U ? cid * per : U;
        const uint32_t u1 = ustart + per < U ? ustart + per : U;
        ucount = u1 - ustart; ustep = 1u;
    }

    if (tid == 0) {
        #pragma unroll
        for (uint32_t s = 0; s < S; ++s) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bfull[s])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bempty[s])));
        }
        #pragma unroll
        for (uint32_t b = 0; b < 2u; ++b) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&tfull[b])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 4;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&tempty[b])));
        }
        // publish the generic-proxy inits to the async proxy before the first
        // cp.async.bulk complete_tx references them (PTX fence.mbarrier_init;
        // the mmaf twin measurably corrupted without it -)
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();
    // tmem alloc is deferred to the ISSUER WARP so its latency overlaps the
    // producer's first TMA flight instead of stacking in front of it

    auto bar_wait = [&](uint64_t* bar, uint32_t parity) {
        const uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
        asm volatile("{\n\t.reg .pred p;\nW%=:\n\t"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                     "@!p bra W%=;\n\t}" ::"r"(a), "r"(parity));
    };
    auto slice_k = [&](uint32_t slice, uint32_t& ks0, uint32_t& kcnt) {
        ks0 = slice * kbase + (slice < krem ? slice : krem);
        const uint32_t scnt = kbase + (slice < krem ? 1u : 0u);
        kcnt = ks0 + scnt > nk ? nk - ks0 : scnt;
    };

    if (tid == 0) {
        // producer: one 16KB W slab + one X slab per slot
        uint32_t n = 0, eph = 0;
        for (uint32_t k = 0; k < ucount; ++k) {
            const uint32_t u = ustart + k * ustep;
            const uint32_t t = u / KS;
            const int rb0 = (int)(t * 128u);
            uint32_t ks0, kcnt; slice_k(u % KS, ks0, kcnt);
            for (uint32_t j = 0; j < kcnt; ++j, ++n) {
                const uint32_t s = n % S;
                if (n >= S) { bar_wait(&bempty[s], (eph >> s) & 1u); eph ^= 1u << s; }
                const uint32_t m = (uint32_t)__cvta_generic_to_shared(&bfull[s]);
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                             ::"r"(m), "r"(16384u + xsl));
                const int ck = (int)(ks0 + j);
                asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes"
                             " [%0], [%1, {0, %2, %3}], [%4];"
                             ::"r"((uint32_t)__cvta_generic_to_shared(wt + s * 16384u)),
                               "l"(&wmap), "r"(rb0), "r"(ck), "r"(m) : "memory");
                asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes"
                             " [%0], [%1, {0, 0, %2}], [%3];"
                             ::"r"((uint32_t)__cvta_generic_to_shared(xt + (size_t)s * xsl)),
                               "l"(&xmap), "r"(ck), "r"(m) : "memory");
            }
        }
    } else if (tid >= 32 && tid < 64) {
        // warp1: collective tmem alloc (overlapped with slot-0 TMA), then
        // lane 32 issues 4 K16 mmas per slot, single chain (KS shortens it;
        // dual-chain interleave measured NULL - the mma side is hidden)
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                     ::"r"((uint32_t)__cvta_generic_to_shared(tmem_slot)), "r"(ncols));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
        if (tid == 32) {
        const uint32_t tmem = tmem_slot[0];
        uint32_t n = 0, fph = 0, teph = 0, tc = 0;
        const uint32_t id = (1u << 4) | ((nto >> 3) << 17) | ((128u >> 4) << 24);
        for (uint32_t k = 0; k < ucount; ++k, ++tc) {
            const uint32_t u = ustart + k * ustep;
            const uint32_t buf = tc & 1u;
            uint32_t ks0, kcnt; slice_k(u % KS, ks0, kcnt);
            if (tc >= 2u) { bar_wait(&tempty[buf], (teph >> buf) & 1u); teph ^= 1u << buf; }
            asm volatile("tcgen05.fence::after_thread_sync;");
            for (uint32_t j = 0; j < kcnt; ++j, ++n) {
                const uint32_t s = n % S;
                bar_wait(&bfull[s], (fph >> s) & 1u); fph ^= 1u << s;
                const uint32_t w16 = (uint32_t)__cvta_generic_to_shared(wt + s * 16384u) >> 4;
                const uint32_t x16 = (uint32_t)__cvta_generic_to_shared(xt + (size_t)s * xsl) >> 4;
                const uint32_t acc = tmem + buf * nto;
                #pragma unroll
                for (uint32_t kb = 0; kb < 4u; ++kb) {
                    const uint32_t en = (j > 0 || kb > 0) ? 1u : 0u;
                    asm volatile(
                        "{\n\t.reg .pred p;\n\t"
                        "setp.ne.b32 p, %4, 0;\n\t"
                        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}"
                        ::"r"(acc), "l"(pd_tc5_sdesc(w16 + kb * 2u)),
                          "l"(pd_tc5_sdesc(x16 + kb * 2u)), "r"(id), "r"(en));
                }
                asm volatile(
                    "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                    ::"r"((uint32_t)__cvta_generic_to_shared(&bempty[s])));
            }
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                ::"r"((uint32_t)__cvta_generic_to_shared(&tfull[buf])));
        }
        }
    } else if (tid >= 64) {
        // epilogue: 4 warps, lane=row identity. tcgen05.ld reads the
        // EXECUTING warp's physical subpartition (warpid%4) regardless of the
        // taddr lane field - warps 2..5 claim rows (warpid&3)*32
        const uint32_t lane = tid & 31u;
        const uint32_t pw = (tid >> 5) & 3u;
        const uint32_t rl = pw * 32u + lane;
        const uint32_t thrKS = useTma ? 1u : 4u;
        uint32_t gsl = 0;
        uint32_t tc = 0;
        for (uint32_t k = 0; k < ucount; ++k, ++tc) {
            const uint32_t u = ustart + k * ustep;
            const uint32_t buf = tc & 1u;
            const uint32_t t = u / KS, slice = u % KS;
            bar_wait(&tfull[buf], (tc >> 1) & 1u);
            asm volatile("tcgen05.fence::after_thread_sync;");
            const uint32_t tmem = tmem_slot[0];
            if (KS > 1u && slice > 0u) {
                if (lane == 0) {
                    uint32_t v; uint32_t poll = 0;
                    do {
                        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
                                     : "=r"(v) : "l"(pd_f16ks_flags + t) : "memory");
                        if ((v & 0xffu) >= thrKS) break;
                        // hot-poll first (small-KS chains wake within ~us),
                        // then back off - up to KS-1/KS of the grid reaches
                        // this spin together while slice 0 drains
                        if (++poll > 8u) asm volatile("nanosleep.u32 256;");
                    } while (true);
                }
                __syncwarp();
                if (useTma) asm volatile("fence.proxy.async.global;");
            }
            const uint32_t row = t * 128u + rl;
            if (useTma) {
                const bool red = (KS > 1u && slice > 0u) || beta == 1.0f;
                const uint32_t NS = nto >> 4;
                auto tld = [&](uint32_t si, uint32_t co, uint32_t* r) {
                    const uint32_t taddr = tmem + co + ((pw * 32u) << 16)
                                         + si * 16u;
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                        : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),
                          "=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
                          "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),
                          "=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15])
                        : "r"(taddr));
                };
                uint32_t ra[16];
                #pragma unroll 1
                for (uint32_t si = 0; si < NS; ++si, ++gsl) {
                    if (tid == 64 && gsl >= 2u)
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                    asm volatile("bar.sync 1, 128;");
                    tld(si, buf * nto, ra);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    unsigned char* stg = stage + (gsl & 1u) * 8192u;
                    const uint32_t sb = (uint32_t)__cvta_generic_to_shared(stg) + rl * 4u;
                    #pragma unroll
                    for (uint32_t jj = 0; jj < 16u; ++jj)
                        asm volatile("st.shared.b32 [%0], %1;"
                                     ::"r"(sb + jj * 512u), "r"(ra[jj]) : "memory");
                    asm volatile("fence.proxy.async.shared::cta;");
                    asm volatile("bar.sync 1, 128;");
                    if (tid == 64) {
                        const int cr = (int)(t * 128u), cc = (int)(si * 16u);
                        const uint32_t ss = (uint32_t)__cvta_generic_to_shared(stg);
                        if (red)
                            asm volatile("cp.reduce.async.bulk.tensor.2d.global.shared::cta"
                                         ".add.tile.bulk_group [%0, {%1, %2}], [%3];"
                                         ::"l"(&ymap), "r"(cr), "r"(cc), "r"(ss) : "memory");
                        else
                            asm volatile("cp.async.bulk.tensor.2d.global.shared::cta"
                                         ".tile.bulk_group [%0, {%1, %2}], [%3];"
                                         ::"l"(&ymap), "r"(cr), "r"(cc), "r"(ss) : "memory");
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                }
            } else {
            // simple per-chunk drain; beta is a UNIFORM 3-way branch (the
            // per-element ternary compiles to an unconditional LDG+FSEL RMW)
            #pragma unroll 1
            for (uint32_t cchunk = 0; cchunk * 16u < nto; ++cchunk) {
                const uint32_t c0 = cchunk * 16u;
                uint32_t r[16];
                const uint32_t taddr = tmem + buf * nto + ((pw * 32u) << 16) + c0;
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                    "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                    : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),
                      "=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
                      "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),
                      "=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15])
                    : "r"(taddr));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                if (row < out_dim) {
                    if (KS > 1u && slice > 0u) {
                        #pragma unroll
                        for (uint32_t q = 0; q < 16u; ++q)
                            if (c0 + q < batch)
                                asm volatile("red.global.add.f32 [%0], %1;"
                                             ::"l"(y + (size_t)(c0 + q) * out_dim + row),
                                               "f"(__uint_as_float(r[q])) : "memory");
                    } else if (beta != 0.0f) {
                        #pragma unroll
                        for (uint32_t q = 0; q < 16u; ++q)
                            if (c0 + q < batch) {
                                float* o = y + (size_t)(c0 + q) * out_dim + row;
                                *o = __uint_as_float(r[q]) + beta * *o;
                            }
                    } else {
                        #pragma unroll
                        for (uint32_t q = 0; q < 16u; ++q)
                            if (c0 + q < batch)
                                y[(size_t)(c0 + q) * out_dim + row] = __uint_as_float(r[q]);
                    }
                }
            }
            }
            if (KS > 1u && useTma) {
                if (tid == 64) {
                    asm volatile("cp.async.bulk.wait_group 0;");
                    asm volatile("fence.proxy.async.global;");
                    if (slice == 0u) {
                        asm volatile("red.release.gpu.global.add.u32 [%0], 1;"
                                     ::"l"(pd_f16ks_flags + t) : "memory");
                    } else {
                        uint32_t old;
                        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 256;"
                                     : "=r"(old) : "l"(pd_f16ks_flags + t) : "memory");
                        if ((old >> 8) + 1u == (KS - 1u) * 1u)
                            asm volatile("st.relaxed.gpu.global.u32 [%0], 0;"
                                         ::"l"(pd_f16ks_flags + t) : "memory");
                    }
                }
            } else if (KS > 1u) {
                __syncwarp();
                if (slice == 0u) {
                    if (lane == 0)
                        asm volatile("red.release.gpu.global.add.u32 [%0], 1;"
                                     ::"l"(pd_f16ks_flags + t) : "memory");
                } else if (lane == 0) {
                    uint32_t old;
                    asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 256;"
                                 : "=r"(old) : "l"(pd_f16ks_flags + t) : "memory");
                    if ((old >> 8) + 1u == (KS - 1u) * 4u)
                        asm volatile("st.relaxed.gpu.global.u32 [%0], 0;"
                                     ::"l"(pd_f16ks_flags + t) : "memory");
                }
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            if (lane == 0)
                asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
                             ::"r"((uint32_t)__cvta_generic_to_shared(&tempty[buf])) : "memory");
        }
        if (tid == 64)
            asm volatile("cp.async.bulk.wait_group 0;");
    }
    __syncthreads();
    if (tid >= 32 && tid < 64)
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                     ::"r"(tmem_slot[0]), "r"(ncols));
#else
    (void)wmap; (void)xmap; (void)ymap; (void)y; (void)beta;
    (void)in_dim; (void)out_dim; (void)batch; (void)nto; (void)KS; (void)ncols;
#endif
}

// paired-tile variant (TMA epilogue only): two adjacent 128-row tiles share
// one X slab per chunk - X traffic halves (X/W byte ratio is nto/128, 50% of
// the whole stream at nto=128). 8 mmas/chunk into two tmem accumulators
// (A=tmem, B=tmem+nto); Y staging REUSES W ring slot 0 - legal only because
// the epilogue starts after tfull confirms every mma consumed its slabs AND
// the launcher always launches grid==U (ucount==1, no producer refill during
// the epilogue). Elected only when the paired grid still covers the machine
// on its own (U0p >= SM count): at laguna geometry (U0p~48) the X saving is
// eaten 1:1 by the combine RMW needed to refill the grid via extra K-slices.
template <uint32_t S>
__global__ void __launch_bounds__(192) pd_f16_gemm_tc5gp_kernel(
    const __grid_constant__ PdTmap wmap, const __grid_constant__ PdTmap xmap,
    const __grid_constant__ PdTmap ymap,
    float* __restrict__ y, float beta,
    uint32_t in_dim, uint32_t out_dim, uint32_t batch, uint32_t nto,
    uint32_t KS, uint32_t ncols) {
#if PD_TC5_OK
    extern __shared__ __align__(1024) unsigned char shG[];
    unsigned char* wt = shG;                       // S x 32KB W (A|B)
    unsigned char* xt = shG + S * 32768u;          // S x nto*128 X
    unsigned char* stage = shG;                    // reuse W slot 0 post-mma
    uint64_t* bfull  = (uint64_t*)(xt + S * (size_t)(nto * 128u));
    uint64_t* bempty = bfull + S;
    uint64_t* tfull  = bempty + S;
    uint64_t* tempty = tfull + 2;
    __shared__ uint32_t tmem_slot[1];

    const uint32_t tid = threadIdx.x;
    const uint32_t nk = (in_dim * 2u + 127u) / 128u;
    const uint32_t xsl = nto * 128u;
    const uint32_t pair_tiles = (out_dim + 255u) >> 8;
    const uint32_t U = pair_tiles * KS;
    const uint32_t kbase = nk / KS, krem = nk % KS;
    const uint32_t n_ctas = gridDim.x;
    const uint32_t cid = blockIdx.x;
    uint32_t ucount, ustart, ustep;
    if (KS > 1u) {
        ustart = cid; ustep = n_ctas;
        ucount = cid < U ? (U - 1u - cid) / n_ctas + 1u : 0u;
    } else {
        const uint32_t per = (U + n_ctas - 1u) / n_ctas;
        ustart = cid * per < U ? cid * per : U;
        const uint32_t u1 = ustart + per < U ? ustart + per : U;
        ucount = u1 - ustart; ustep = 1u;
    }

    if (tid == 0) {
        #pragma unroll
        for (uint32_t s = 0; s < S; ++s) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bfull[s])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bempty[s])));
        }
        #pragma unroll
        for (uint32_t b = 0; b < 2u; ++b) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&tfull[b])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 4;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&tempty[b])));
        }
        // publish the generic-proxy inits to the async proxy before the first
        // cp.async.bulk complete_tx references them (PTX fence.mbarrier_init;
        // the mmaf twin measurably corrupted without it -)
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();

    auto bar_wait = [&](uint64_t* bar, uint32_t parity) {
        const uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
        asm volatile("{\n\t.reg .pred p;\nW%=:\n\t"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                     "@!p bra W%=;\n\t}" ::"r"(a), "r"(parity));
    };
    auto slice_k = [&](uint32_t slice, uint32_t& ks0, uint32_t& kcnt) {
        ks0 = slice * kbase + (slice < krem ? slice : krem);
        const uint32_t scnt = kbase + (slice < krem ? 1u : 0u);
        kcnt = ks0 + scnt > nk ? nk - ks0 : scnt;
    };

    if (tid == 0) {
        uint32_t n = 0, eph = 0;
        for (uint32_t k = 0; k < ucount; ++k) {
            const uint32_t u = ustart + k * ustep;
            const uint32_t t = u / KS;
            const int rb0 = (int)(t * 256u);
            const bool hasB = t * 256u + 128u < out_dim;
            uint32_t ks0, kcnt; slice_k(u % KS, ks0, kcnt);
            for (uint32_t j = 0; j < kcnt; ++j, ++n) {
                const uint32_t s = n % S;
                if (n >= S) { bar_wait(&bempty[s], (eph >> s) & 1u); eph ^= 1u << s; }
                const uint32_t m = (uint32_t)__cvta_generic_to_shared(&bfull[s]);
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                             ::"r"(m), "r"((hasB ? 32768u : 16384u) + xsl));
                const int ck = (int)(ks0 + j);
                asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes"
                             " [%0], [%1, {0, %2, %3}], [%4];"
                             ::"r"((uint32_t)__cvta_generic_to_shared(wt + s * 32768u)),
                               "l"(&wmap), "r"(rb0), "r"(ck), "r"(m) : "memory");
                if (hasB)
                    asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes"
                                 " [%0], [%1, {0, %2, %3}], [%4];"
                                 ::"r"((uint32_t)__cvta_generic_to_shared(wt + s * 32768u + 16384u)),
                                   "l"(&wmap), "r"(rb0 + 128), "r"(ck), "r"(m) : "memory");
                asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes"
                             " [%0], [%1, {0, 0, %2}], [%3];"
                             ::"r"((uint32_t)__cvta_generic_to_shared(xt + (size_t)s * xsl)),
                               "l"(&xmap), "r"(ck), "r"(m) : "memory");
            }
        }
    } else if (tid >= 32 && tid < 64) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                     ::"r"((uint32_t)__cvta_generic_to_shared(tmem_slot)), "r"(ncols));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
        if (tid == 32) {
        const uint32_t tmem = tmem_slot[0];
        uint32_t n = 0, fph = 0, teph = 0, tc = 0;
        const uint32_t id = (1u << 4) | ((nto >> 3) << 17) | ((128u >> 4) << 24);
        for (uint32_t k = 0; k < ucount; ++k, ++tc) {
            const uint32_t u = ustart + k * ustep;
            const uint32_t buf = tc & 1u;
            const uint32_t t = u / KS;
            const bool hasB = t * 256u + 128u < out_dim;
            uint32_t ks0, kcnt; slice_k(u % KS, ks0, kcnt);
            if (tc >= 2u) { bar_wait(&tempty[buf], (teph >> buf) & 1u); teph ^= 1u << buf; }
            asm volatile("tcgen05.fence::after_thread_sync;");
            for (uint32_t j = 0; j < kcnt; ++j, ++n) {
                const uint32_t s = n % S;
                bar_wait(&bfull[s], (fph >> s) & 1u); fph ^= 1u << s;
                const uint32_t wa16 = (uint32_t)__cvta_generic_to_shared(wt + s * 32768u) >> 4;
                const uint32_t x16 = (uint32_t)__cvta_generic_to_shared(xt + (size_t)s * xsl) >> 4;
                #pragma unroll
                for (uint32_t kb = 0; kb < 4u; ++kb) {
                    const uint32_t en = (j > 0 || kb > 0) ? 1u : 0u;
                    asm volatile(
                        "{\n\t.reg .pred p;\n\t"
                        "setp.ne.b32 p, %4, 0;\n\t"
                        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}"
                        ::"r"(tmem), "l"(pd_tc5_sdesc(wa16 + kb * 2u)),
                          "l"(pd_tc5_sdesc(x16 + kb * 2u)), "r"(id), "r"(en));
                }
                if (hasB) {
                    const uint32_t wb16 = wa16 + (16384u >> 4);
                    #pragma unroll
                    for (uint32_t kb = 0; kb < 4u; ++kb) {
                        const uint32_t en = (j > 0 || kb > 0) ? 1u : 0u;
                        asm volatile(
                            "{\n\t.reg .pred p;\n\t"
                            "setp.ne.b32 p, %4, 0;\n\t"
                            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}"
                            ::"r"(tmem + nto), "l"(pd_tc5_sdesc(wb16 + kb * 2u)),
                              "l"(pd_tc5_sdesc(x16 + kb * 2u)), "r"(id), "r"(en));
                    }
                }
                asm volatile(
                    "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                    ::"r"((uint32_t)__cvta_generic_to_shared(&bempty[s])));
            }
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                ::"r"((uint32_t)__cvta_generic_to_shared(&tfull[buf])));
        }
        }
    } else if (tid >= 64) {
        const uint32_t lane = tid & 31u;
        const uint32_t pw = (tid >> 5) & 3u;
        const uint32_t rl = pw * 32u + lane;
        uint32_t gsl = 0;
        uint32_t tc = 0;
        for (uint32_t k = 0; k < ucount; ++k, ++tc) {
            const uint32_t u = ustart + k * ustep;
            const uint32_t buf = tc & 1u;
            const uint32_t t = u / KS, slice = u % KS;
            const uint32_t ntiles = (t * 256u + 128u < out_dim) ? 2u : 1u;
            bar_wait(&tfull[buf], (tc >> 1) & 1u);
            asm volatile("tcgen05.fence::after_thread_sync;");
            const uint32_t tmem = tmem_slot[0];
            if (KS > 1u && slice > 0u) {
                if (lane == 0) {
                    uint32_t v; uint32_t poll = 0;
                    do {
                        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
                                     : "=r"(v) : "l"(pd_f16ks_flags + t) : "memory");
                        if ((v & 0xffu) >= 1u) break;
                        if (++poll > 8u) asm volatile("nanosleep.u32 256;");
                    } while (true);
                }
                __syncwarp();
                asm volatile("fence.proxy.async.global;");
            }
            const bool red = (KS > 1u && slice > 0u) || beta == 1.0f;
            const uint32_t NS = nto >> 4;
            auto tld = [&](uint32_t si, uint32_t co, uint32_t* r) {
                const uint32_t taddr = tmem + co + ((pw * 32u) << 16) + si * 16u;
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                    "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                    : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),
                      "=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
                      "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),
                      "=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15])
                    : "r"(taddr));
            };
            uint32_t ra[16];
            for (uint32_t tp = 0; tp < ntiles; ++tp) {
                #pragma unroll 1
                for (uint32_t si = 0; si < NS; ++si, ++gsl) {
                    if (tid == 64 && gsl >= 2u)
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                    asm volatile("bar.sync 1, 128;");
                    tld(si, tp * nto, ra);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    unsigned char* stg = stage + (gsl & 1u) * 8192u;
                    const uint32_t sb = (uint32_t)__cvta_generic_to_shared(stg) + rl * 4u;
                    #pragma unroll
                    for (uint32_t jj = 0; jj < 16u; ++jj)
                        asm volatile("st.shared.b32 [%0], %1;"
                                     ::"r"(sb + jj * 512u), "r"(ra[jj]) : "memory");
                    asm volatile("fence.proxy.async.shared::cta;");
                    asm volatile("bar.sync 1, 128;");
                    if (tid == 64) {
                        const int cr = (int)(t * 256u + tp * 128u), cc = (int)(si * 16u);
                        const uint32_t ss = (uint32_t)__cvta_generic_to_shared(stg);
                        if (red)
                            asm volatile("cp.reduce.async.bulk.tensor.2d.global.shared::cta"
                                         ".add.tile.bulk_group [%0, {%1, %2}], [%3];"
                                         ::"l"(&ymap), "r"(cr), "r"(cc), "r"(ss) : "memory");
                        else
                            asm volatile("cp.async.bulk.tensor.2d.global.shared::cta"
                                         ".tile.bulk_group [%0, {%1, %2}], [%3];"
                                         ::"l"(&ymap), "r"(cr), "r"(cc), "r"(ss) : "memory");
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                }
            }
            if (KS > 1u) {
                if (tid == 64) {
                    asm volatile("cp.async.bulk.wait_group 0;");
                    asm volatile("fence.proxy.async.global;");
                    if (slice == 0u) {
                        asm volatile("red.release.gpu.global.add.u32 [%0], 1;"
                                     ::"l"(pd_f16ks_flags + t) : "memory");
                    } else {
                        uint32_t old;
                        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 256;"
                                     : "=r"(old) : "l"(pd_f16ks_flags + t) : "memory");
                        if ((old >> 8) + 1u == (KS - 1u) * 1u)
                            asm volatile("st.relaxed.gpu.global.u32 [%0], 0;"
                                         ::"l"(pd_f16ks_flags + t) : "memory");
                    }
                }
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            if (lane == 0)
                asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
                             ::"r"((uint32_t)__cvta_generic_to_shared(&tempty[buf])) : "memory");
        }
        if (tid == 64)
            asm volatile("cp.async.bulk.wait_group 0;");
    }
    __syncthreads();
    if (tid >= 32 && tid < 64)
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                     ::"r"(tmem_slot[0]), "r"(ncols));
#else
    (void)wmap; (void)xmap; (void)ymap; (void)y; (void)beta;
    (void)in_dim; (void)out_dim; (void)batch; (void)nto; (void)KS; (void)ncols;
#endif
}
#endif  // tc5g arch guard (host pass || sm_90+)

// ---------------------------------------------------------------- launchers
// PD_EXPORT is the pack's visibility-default extern "C" (abi.cuh); when this
// header is compiled standalone (correctness test) it falls back to extern "C".
#ifndef PD_EXPORT
#define PD_EXPORT extern "C"
#endif

// A/B escape: PADDOCK_F16_WMMA=1 forces the portable wmma path. Dev knob only
// (getenv, not the UCRT-safe Rust-set-var reader) - it is never a shipped tuned
// default, so the Windows environment-visibility caveat does not apply.
static inline bool pd_f16_wmma_forced() {
    static const bool v = [] {
        const char* e = std::getenv("PADDOCK_F16_WMMA");
        return e != nullptr && e[0] != '\0' && !(e[0] == '0' && e[1] == '\0');
    }();
    return v;
}

// ---- K-split for deep-K / skinny-N -----------------------------------------
// The tile election picks the best of two tiles, but at deep K with a small
// batch both underfill: qwen3-asr's proj is K=7680 M=1024 N=104, which is 32
// narrow CTAs on 188 SMs, and it measured 4.6x off cuBLAS even on the better
// tile. There is no tile that fixes that -- the parallelism simply is not in
// the M/N plane, it is in K.
//
// Same construction bf16_dense.cuh already uses: grid.z K-slabs into a static
// partials plane, then one fixed-order combine. Fixed order (z ascending) is
// what keeps the result deterministic run to run; a tree or atomic reduction
// would not be.
//
// Plane budget. 8 * 4096 * 64 floats = 8 MB. Sized deliberately and no larger:
// the shapes that need the split are skinny (M*N ~ 105k for the audio tower),
// and nz clamps itself down rather than reaching for a bigger plane, so a wide
// shape declines the split instead of growing the footprint.
#define PD_F16KS_ELEMS (8u * 4096u * 64u)

// Allocated LAZILY, so a model that never takes the split pays nothing. Every
// other scratch in this pack is a __device__ array, which is resident from
// module load whether or not its path is ever used; this is the one deviation
// from that idiom, and it is deliberate - most models on a box never issue an
// f16 GEMM that splits, and 8 MB of unconditional residency is not free on a
// card whose VRAM budget we have been fighting for.
//
// Keyed to the device current at allocation: a cudaMalloc pointer is valid
// only there. On any other device we return null, and null is not a failure -
// the launcher reads it as "stay unsplit", which is always a correct answer.
// Same on allocation failure: we clear the sticky error (the launcher returns
// cudaGetLastError(), and an OOM here must not surface as a launch error) and
// fall back to the unsplit path.
//
// Note for VRAM accounting: this lives outside the engine's pool, so it does
// not appear in settled_mem_used()'s residency split. One-shot, 8 MB, and only
// on models that actually split.
static float* pd_f16ks_scr() {
    struct Scr {
        float* p = nullptr;
        int dev = -1;
        Scr() {
            if (cudaGetDevice(&dev) != cudaSuccess) { dev = -1; return; }
            void* q = nullptr;
            if (cudaMalloc(&q, (size_t)PD_F16KS_ELEMS * sizeof(float)) == cudaSuccess)
                p = (float*)q;
            else
                cudaGetLastError();  // swallow it; unsplit is a fine outcome
        }
    };
    static Scr s;  // C++11 magic static: constructed once, thread-safe
    int cur = -1;
    if (cudaGetDevice(&cur) != cudaSuccess || cur != s.dev) return nullptr;
    return s.p;
}

// nz policy. The grid-fill test decides whether to split; the count is a pure
// function of K (512-element slabs, a multiple of every config's KT), capped
// at 8 and then clamped to fit the plane. 0 = stay unsplit.
// Kill: PADDOCK_NO_F16_KSPLIT.
static uint32_t pd_f16ks_nz(uint32_t blocks2d, uint32_t in_dim,
                            uint32_t out_m, uint32_t batch) {
    static const bool off = pd_env("PADDOCK_NO_F16_KSPLIT") != nullptr;
    if (off) return 0u;
    static int nsm = 0;
    if (nsm == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
        if (nsm <= 0) nsm = 128;
    }
    // Split only when the 2D grid leaves the machine genuinely empty. bf16 uses
    // 2 CTAs/SM here; measured on this tile that is far too permissive -- the
    // audio tower's ffn-up runs 128 narrow CTAs (two thirds of 188 SMs), where
    // nz=2 buys no parallelism and the extra combine pass is pure cost: 0.012
    // -> 0.016 ms. Half the SM count is where the split starts paying, and it
    // is the same threshold the tile election crosses at, for the same reason.
    if (blocks2d >= (uint32_t)nsm / 2u) return 0u;   // grid already fills it
    const uint32_t nb = (in_dim + 511u) / 512u;
    uint32_t nz = nb > 8u ? 8u : nb;
    while (nz >= 2u && (size_t)nz * out_m * batch > (size_t)PD_F16KS_ELEMS)
        --nz;
    return nz >= 2u ? nz : 0u;
}

// Fixed-order combine: sum the nz planes, then apply beta once.
__global__ void pd_f16_ks_combine_kernel(const float* __restrict__ part,
                                         float* __restrict__ y, float beta,
                                         uint32_t n, uint32_t nz) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = 0.0f;
    for (uint32_t z = 0; z < nz; ++z) acc += part[(size_t)z * n + i];
    y[i] = (beta != 0.0f) ? acc + beta * y[i] : acc;
}

static int pd_f16_gemm_wmma_launch(const __half* w, const __half* x, float* y,
                                   float beta, unsigned int in_dim,
                                   unsigned int out_dim, unsigned int batch,
                                   cudaStream_t st) {
    constexpr int WM = 4, WN = 4;  // 16 warps, 64x64 output tile
    dim3 grid((out_dim + WM * 16u - 1u) / (WM * 16u),
              (batch + WN * 16u - 1u) / (WN * 16u));
    pd_f16_gemm_wmma_kernel<WM, WN><<<grid, WM * WN * 32u, 0, st>>>(
            w, x, y, beta, in_dim, out_dim, batch);
    return (int)cudaGetLastError();
}

// One templated config launcher: sets the opt-in dynamic-smem cap once per
// instantiation, then launches with the computed byte count. Exposed so the
// perf sweep can drive arbitrary (tile, ST, KT) without editing the dispatch.
template <uint32_t BM, uint32_t BN, uint32_t NW, uint32_t ST, uint32_t KT,
          uint32_t RG, uint32_t CG>
static int pd_f16_mma_cfg(const __half* w, const __half* x, float* y, float beta,
                          unsigned in_dim, unsigned out_dim, unsigned batch,
                          cudaStream_t st) {
    constexpr uint32_t KPAD = KT + 8u;
    constexpr unsigned smem = 2u * ST * (BM + BN) * KPAD;  // bytes
    static bool set = false;
    if (!set) {
        cudaFuncSetAttribute(pd_f16_gemm_mma_kernel<BM, BN, NW, ST, KT, RG, CG>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        cudaFuncSetAttribute(
                pd_f16_gemm_mma_kernel<BM, BN, NW, ST, KT, RG, CG, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        set = true;
    }
    dim3 grid((out_dim + BM - 1u) / BM, (batch + BN - 1u) / BN);

    // K-split arm: same tile, grid.z slabs into the partials plane, then one
    // fixed-order combine. Self-gates on grid fill, slab depth and plane fit,
    // so a shape that already covers the machine falls straight through.
    const uint32_t nz = pd_f16ks_nz(grid.x * grid.y, in_dim, out_dim, batch);
    float* part = (nz >= 2u) ? pd_f16ks_scr() : nullptr;
    if (part) {
        // slab is KT-aligned so only the last one is ragged.
        const uint32_t slab = (((in_dim + nz - 1u) / nz + KT - 1u) / KT) * KT;
        dim3 gz(grid.x, grid.y, nz);
        pd_f16_gemm_mma_kernel<BM, BN, NW, ST, KT, RG, CG, true>
                <<<gz, NW * 32u, smem, st>>>(w, x, y, beta, in_dim, out_dim,
                                             batch, part, slab);
        const uint32_t n = out_dim * batch;
        pd_f16_ks_combine_kernel<<<(n + 255u) / 256u, 256u, 0, st>>>(
                part, y, beta, n, nz);
        return (int)cudaGetLastError();
    }

    pd_f16_gemm_mma_kernel<BM, BN, NW, ST, KT, RG, CG><<<grid, NW * 32u, smem, st>>>(
            w, x, y, beta, in_dim, out_dim, batch);
    return (int)cudaGetLastError();
}

static int pd_f16_gemm_mma_launch(const __half* w, const __half* x, float* y,
                                  float beta, unsigned int in_dim,
                                  unsigned int out_dim, unsigned int batch,
                                  cudaStream_t st) {
    // Register-blocked warp micro-tile (RG row-groups x CG col-groups) so each
    // A/B fragment feeds RG*CG mmas - that ratio is what unbinds the tensor pipe
    // from the LSU. Config picked by the (tile,ST,KT,RG,CG) sweep. Small batch
    // keeps a narrow tile (no wasted cols); the skinny-N/GEMV arm is the follow-up.
    // Elect by MACHINE FILL, not by batch. `batch <= 64` was a proxy for "is
    // the wide tile worth its coarser granularity", and it misses by an order
    // of magnitude whenever out_dim is small: qwen3-asr's audio tower runs
    // out_dim=1024 batch=104, which is 8 wide CTAs on 188 SMs and measured
    // 2.4x slower than the narrow tile. What actually predicts the winner is
    // the wide grid's own CTA count.
    //
    // Measured crossover (p64_f16_audio_gemm sweep, 36 synthetic shapes x {M 480,
    // 1024, 4096} x {K 1024, 4096} plus the 8 shapes of one ASR request): at
    // <= 64 wide-CTAs the narrow tile's 4x finer grid wins every shape; at
    // >= 128 the wide tile's 4x4 register blocking wins every shape. Half the
    // SM count sits in that gap and calls all 44 correctly. Deliberately a
    // measured crossover and not a derived one -- the two tiles differ in
    // register blocking as well as footprint, so there is no clean occupancy
    // argument to appeal to, only where they actually cross.
    static int nsm = 0;
    if (nsm == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
        if (nsm <= 0) nsm = 128;
    }
    const uint32_t ctas_wide = ((out_dim + 127u) / 128u) * ((batch + 127u) / 128u);
    if (ctas_wide < (uint32_t)nsm / 2u)
        return pd_f16_mma_cfg<64u, 64u, 8u, 3u, 32u, 2u, 2u>(w, x, y, beta, in_dim, out_dim, batch, st);
    return pd_f16_mma_cfg<128u, 128u, 8u, 3u, 32u, 4u, 4u>(w, x, y, beta, in_dim, out_dim, batch, st);
}

// ---- GEMV band twin (batch <= 8) ------------------------------------------
// Plain LDG dot-product kernel, all arches. At decode-band shapes (M=1280..
// 5120, W 3-13MB, N<=8) the tc5g ring pays a flat ~7.8us serve-graph span
// (ramp + tmem drain + TMA-store completion + K-split flags) against nvjet's
// ~4.5us; at N<=8 the FMA rate needed to hold DRAM pace fits the f32 cores,
// so the ring buys nothing. Measured against cuBLAS:
// wo b1 1.44x, fc2 b1 1.32x, qkv b1 1.05x; b<=4 wins everywhere, b5-8 wins
// only at out_dim <= ~2560 (the launcher's election mirrors that).
//
// smem X stage is transposed to [k][b] and XOR-swizzled on the 16B unit
// index: the naive layout puts every lane's read at l*NB*16B, which is
// 0 mod 128B at NB=8 - a 32-way bank conflict measured at 24.6us on a 0.8us
// kernel. The swizzle permutes units within 1KB blocks (alloc rounds up), and
// the NB halfs of one k stay contiguous so vector LDS survives.
__device__ __forceinline__ uint32_t pd_f16_gemv_sw(uint32_t h) {
    const uint32_t u = h >> 3, o = h & 7u;
    return ((u ^ ((u >> 3) & 7u)) << 3) + o;
}

template <uint32_t NB>
__global__ void __launch_bounds__(256) pd_f16_gemv_kernel(
        const __half* __restrict__ w, const __half* __restrict__ x,
        float* __restrict__ y, float beta,
        uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
    extern __shared__ __half pd_gv_xt[];  // [in_dim][NB], b minor, swizzled
    for (uint32_t b = 0; b < batch; ++b) {
        const uint4* xr = reinterpret_cast<const uint4*>(x + (size_t)b * in_dim);
        for (uint32_t p = threadIdx.x; p < in_dim / 8u; p += blockDim.x) {
            const uint4 v = xr[p];
            const __half* h = reinterpret_cast<const __half*>(&v);
#pragma unroll
            for (uint32_t j = 0; j < 8; ++j)
                pd_gv_xt[pd_f16_gemv_sw((p * 8u + j) * NB + b)] = h[j];
        }
    }
    for (uint32_t b = batch; b < NB; ++b)
        for (uint32_t i = threadIdx.x; i < in_dim; i += blockDim.x)
            pd_gv_xt[pd_f16_gemv_sw(i * NB + b)] = __float2half(0.f);
    __syncthreads();
    const uint32_t o = blockIdx.x * 8u + (threadIdx.x >> 5);
    if (o >= out_dim) return;
    const uint32_t lane = threadIdx.x & 31u;
    const __half* wr = w + (size_t)o * in_dim;
    float acc[NB];
#pragma unroll
    for (uint32_t b = 0; b < NB; ++b) acc[b] = 0.f;
    // lane-strided 16B W packs, ILP bounded at 8 outstanding loads (a fully
    // dynamic loop at K=5120 spills the W buffers and serializes cold)
    const uint32_t chunks = in_dim / 256u + (lane * 8u < in_dim % 256u ? 1u : 0u);
    uint32_t c = 0;
    while (c < chunks) {
        uint4 buf[8];
        const uint32_t n = chunks - c < 8u ? chunks - c : 8u;
#pragma unroll
        for (uint32_t u = 0; u < 8; ++u)
            if (u < n)
                buf[u] = *reinterpret_cast<const uint4*>(
                        wr + (size_t)(c + u) * 256u + lane * 8u);
#pragma unroll
        for (uint32_t u = 0; u < 8; ++u) {
            if (u >= n) break;
            const uint32_t i0 = (c + u) * 256u + lane * 8u;
            const __half* wh = reinterpret_cast<const __half*>(&buf[u]);
            // the lane's 8-k X window is exactly NB swizzle units (i0*NB is
            // 8-aligned): NB vector LDS fetch every (k,b) of this W pack -
            // one 16B LDS total at NB=1
            __half xwin[NB * 8u];
#pragma unroll
            for (uint32_t t = 0; t < NB; ++t)
                *reinterpret_cast<uint4*>(xwin + t * 8u) =
                        *reinterpret_cast<const uint4*>(
                                pd_gv_xt + pd_f16_gemv_sw(i0 * NB + t * 8u));
#pragma unroll
            for (uint32_t d = 0; d < 8; ++d) {
                // f32 products (the COMPUTE_32F class) - rounding the product
                // to f16 first fails the 5e-3 parity gate
                const float wf = __half2float(wh[d]);
#pragma unroll
                for (uint32_t b = 0; b < NB; ++b)
                    acc[b] = fmaf(wf, __half2float(xwin[d * NB + b]), acc[b]);
            }
        }
        c += n;
    }
#pragma unroll
    for (uint32_t b = 0; b < NB; ++b)
        for (uint32_t s = 16; s; s >>= 1)
            acc[b] += __shfl_xor_sync(~0u, acc[b], s);
    if (lane < batch) {
        float* o_ = y + (size_t)lane * out_dim + o;
        if (beta == 0.0f) *o_ = acc[lane];
        else              *o_ = acc[lane] + beta * *o_;
    }
}

static int pd_f16_gemm_gemv_launch(const __half* w, const __half* x, float* y,
                                   float beta, unsigned in_dim,
                                   unsigned out_dim, unsigned batch,
                                   cudaStream_t st) {
    if (batch > 8u || (in_dim & 7u) || ((uintptr_t)w & 15u) || ((uintptr_t)x & 15u))
        return (int)cudaErrorInvalidValue;
    const uint32_t nb = batch <= 1u ? 1u : batch <= 2u ? 2u : batch <= 4u ? 4u : 8u;
    // swizzle permutes within 1KB blocks: round the alloc up so the top
    // partial block cannot relocate out of range
    const uint32_t smem = (in_dim * nb * 2u + 1023u) & ~1023u;
    // per-arch opt-in smem ceiling (consumer arches sit near 100KB)
    static const int cap = [] {
        int dev = 0, v = 0;
        cudaGetDevice(&dev);
        if (cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                   dev) != cudaSuccess)
            v = 48 * 1024;
        for (auto k : {(const void*)pd_f16_gemv_kernel<1>,
                       (const void*)pd_f16_gemv_kernel<2>,
                       (const void*)pd_f16_gemv_kernel<4>,
                       (const void*)pd_f16_gemv_kernel<8>})
            cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, v);
        return v;
    }();
    if (smem > (uint32_t)cap) return (int)cudaErrorInvalidValue;
    const uint32_t grid = (out_dim + 7u) / 8u;
    switch (nb) {
        case 1u: pd_f16_gemv_kernel<1><<<grid, 256, smem, st>>>(w, x, y, beta, in_dim, out_dim, batch); break;
        case 2u: pd_f16_gemv_kernel<2><<<grid, 256, smem, st>>>(w, x, y, beta, in_dim, out_dim, batch); break;
        case 4u: pd_f16_gemv_kernel<4><<<grid, 256, smem, st>>>(w, x, y, beta, in_dim, out_dim, batch); break;
        default: pd_f16_gemv_kernel<8><<<grid, 256, smem, st>>>(w, x, y, beta, in_dim, out_dim, batch); break;
    }
    return (int)cudaGetLastError();
}

#ifdef PD_TC5_HOST
// The host-side tensor-map encode (pd_tmap_encode) and the 128x128B box
// builder (pd_tmap_2d) come from ../tma_desc.cuh. This segment used to define
// its own byte-identical pair, self-named to dodge the include order back when
// the only other copy lived in dense_fp4_w8.cuh, defined after this file.
// Note tma_desc.cuh's builders are gated `PD_BS_HOST || PD_TC5_HOST` precisely
// so this lane reaches them.

// tcgen05 route gate: exact cc 10.0 (the fatbin's sm_103/110 targets compile
// the body empty - the build.sh exact-match convention), tmap encode
// resolvable, kill switch PADDOCK_NO_F16TC5 (dev knob, getenv like the
// PADDOCK_F16_WMMA A/B).
static inline bool pd_f16_tc5_on() {
    static const bool v = [] {
        int dev = 0, ccM = 0, ccm = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&ccM, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&ccm, cudaDevAttrComputeCapabilityMinor, dev);
        const char* e = std::getenv("PADDOCK_NO_F16TC5");
        const bool kill = e != nullptr && e[0] != '\0' && !(e[0] == '0' && e[1] == '\0');
        return ccM == 10 && ccm == 0 && pd_tmap_encode() != nullptr && !kill;
    }();
    return v;
}

// same guard as the kernel: device passes below sm_90 never see the kernel
// symbol, so the referencing launcher must vanish with it
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 900)
template <uint32_t S>
static int pd_f16_gemm_tc5d_launch(const __half* w, const __half* x, float* y,
                                   float beta, unsigned in_dim, unsigned out_dim,
                                   unsigned batch, cudaStream_t st) {
    CUtensorMap wm, xm;
    if (!pd_tmap_2d(&wm, w, (uint64_t)in_dim * 2u, out_dim) ||
        !pd_tmap_2d(&xm, x, (uint64_t)in_dim * 2u, batch))
        return (int)cudaErrorInvalidValue;
    const uint32_t smem = 3u * S * 16384u + (3u * S + 2u) * 8u;
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute((const void*)pd_f16_gemm_tc5d_kernel<S>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        attr = true;
    }
    static const uint32_t sgrid = [] {
        int dev = 0, nn = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&nn, cudaDevAttrMultiProcessorCount, dev);
        return (uint32_t)(nn & ~1);
    }();
    pd_f16_gemm_tc5d_kernel<S><<<sgrid, 320, smem, st>>>(
        wm, xm, y, beta, in_dim, out_dim, batch);
    return (int)cudaGetLastError();
}
#endif  // launcher arch guard

// 3D wide-slot map for the tc5p arm: dims {128B, rows, k-chunks} with
// NON-MONOTONIC strides {kbytes, 128} (legal; probed) so a {128, rows_box, 2}
// box lands as two canonical 16KB slabs stacked at +16KB - the chunk-inner
// ordering breaks the SWIZZLE_128B phase (rows would advance 256B). Requires
// kbytes%128 (in_dim%64) so only the true K tail is OOB zero-fill.
static bool pd_f16t_tmap_3d(CUtensorMap* map, const void* base, uint64_t kbytes,
                            uint64_t rows, uint32_t rows_box,
                            uint32_t kchunks = 2u) {
    pd_tmap_encode_fn enc = pd_tmap_encode();
    if (!enc || ((uintptr_t)base & 15u) || (kbytes & 127u)) return false;
    const cuuint64_t gdim[3] = {128u, rows, kbytes >> 7};
    const cuuint64_t gstride[2] = {kbytes, 128u};
    const cuuint32_t box[3] = {128u, rows_box, kchunks};
    const cuuint32_t estride[3] = {1u, 1u, 1u};
    return enc(map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3u, (void*)base, gdim, gstride,
               box, estride, CU_TENSOR_MAP_INTERLEAVE_NONE,
               CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 900)
template <uint32_t S, uint32_t KS, uint32_t NT>
static int pd_f16_gemm_tc5p_launch(const __half* w, const __half* x, float* y,
                                   float beta, unsigned in_dim, unsigned out_dim,
                                   unsigned batch, cudaStream_t st) {
    const uint32_t nk = (in_dim * 2u + 127u) / 128u;
    if (((nk + 1u) >> 1) < KS) return (int)cudaErrorInvalidValue;
    if (in_dim % 64u || out_dim % 4u) return (int)cudaErrorInvalidValue;
    if (beta != 0.0f && beta != 1.0f) return (int)cudaErrorInvalidValue;
    const uint32_t T = ((out_dim + 255u) >> 8) * ((batch + NT - 1u) / NT);
    if (KS > 1u && T > 4096u) return (int)cudaErrorInvalidValue;  // flags cap
    CUtensorMap wm, xm, ym;
    if (!pd_f16t_tmap_3d(&wm, w, (uint64_t)in_dim * 2u, out_dim, 128u) ||
        !pd_f16t_tmap_3d(&xm, x, (uint64_t)in_dim * 2u, batch, NT >> 1))
        return (int)cudaErrorInvalidValue;
    {
        pd_tmap_encode_fn enc = pd_tmap_encode();
        if (!enc || ((uintptr_t)y & 15u)) return (int)cudaErrorInvalidValue;
        const cuuint64_t gdim[2] = {out_dim, batch};
        const cuuint64_t gstride[1] = {(cuuint64_t)out_dim * 4u};
        const cuuint32_t box[2] = {128u, 16u};
        const cuuint32_t estride[2] = {1u, 1u};
        // FLOAT32 dtype: cp.reduce's add is typed by the map (beta=1, K-split)
        if (enc(&ym, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2u, (void*)y, gdim,
                gstride, box, estride, CU_TENSOR_MAP_INTERLEAVE_NONE,
                CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
            return (int)cudaErrorInvalidValue;
    }
    const uint32_t smem = S * (32768u + NT * 128u) + 1024u + 32768u;
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute((const void*)pd_f16_gemm_tc5p_kernel<S, KS, NT>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        attr = true;
    }
    static const uint32_t sgrid = [] {
        int dev = 0, nn = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&nn, cudaDevAttrMultiProcessorCount, dev);
        return (uint32_t)(nn & ~1);
    }();
    pd_f16_gemm_tc5p_kernel<S, KS, NT><<<sgrid, 320, smem, st>>>(
        wm, xm, ym, beta, in_dim, out_dim, batch);
    return (int)cudaGetLastError();
}

// per-shape election, anchored to a shape census: NT192 wins
// short-K shapes whose 192-col tiling packs the same wave count with less
// per-wave work (muse attn 24.6->20.9us); KS3 wins long-K 2-wave shapes
// (mlp-down 100->83.9us); everything else NT256 KS1. All six vision shapes
// beat the duo arm (1.13-1.52x cuBLAS vs 1.35-1.80x).
//
// Low-fill rescue (whisper encoder class): a small out_dim at
// moderate batch makes so few units that most clusters idle and the wall is
// one unit's chain - finer columns raise fill directly. M=1280 N=1500:
// wo-class 12.60->11.50, fc2 (K=5120) 31.30->25.69 measured; K-splits lose
// here (combine tax > fill gain, ks2 26.83). Fires only when NT128 still
// lands one unit per cluster and NT256 would idle half the machine, which
// no muse vision shape (N=3888) or fused-plane shape (out>=2560) can hit.
// S=4 here: after the one-commit fix unmasked ring
// cover, the 4th slot pays at NT128's 282ns/slot cadence (fc2 26.6->24.6,
// bit-identical outputs - same smem 230400 as <3,1,256>). NT192/256 keep
// S=3 (their slot compute already hides the ring; S=4 doesn't fit anyway).
static int pd_f16_gemm_tc5p_elect(const __half* w, const __half* x, float* y,
                                  float beta, unsigned in_dim, unsigned out_dim,
                                  unsigned batch, cudaStream_t st) {
    static const uint32_t ncl = [] {
        int dev = 0, nn = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&nn, cudaDevAttrMultiProcessorCount, dev);
        return (uint32_t)(nn & ~1) / 2u;
    }();
    const uint32_t nk = (in_dim * 2u + 127u) / 128u;
    const uint32_t rp = (((out_dim + 127u) >> 7) + 1u) >> 1;
    const uint32_t u256 = rp * ((batch + 255u) >> 8);
    const uint32_t w256 = (u256 + ncl - 1u) / ncl;
    const uint32_t u128 = rp * ((batch + 127u) >> 7);
    if (u128 <= ncl && u256 * 2u <= ncl)
        return pd_f16_gemm_tc5p_launch<4u, 1u, 128u>(w, x, y, beta, in_dim,
                                                     out_dim, batch, st);
    if (nk >= 100u && w256 <= 2u && ((nk + 1u) >> 1) >= 3u)
        return pd_f16_gemm_tc5p_launch<3u, 3u, 256u>(w, x, y, beta, in_dim,
                                                     out_dim, batch, st);
    if (nk <= 32u && rp * ((batch + 191u) / 192u) <= w256 * ncl)
        return pd_f16_gemm_tc5p_launch<3u, 1u, 192u>(w, x, y, beta, in_dim,
                                                     out_dim, batch, st);
    return pd_f16_gemm_tc5p_launch<3u, 1u, 256u>(w, x, y, beta, in_dim,
                                                 out_dim, batch, st);
}
#endif  // launcher arch guard (tc5p)

// tc5g skinny launcher. Declines (nonzero) on in_dim%64 or batch>128 - the
// entry falls through to the mma twin. Policy comes out of the shape
// census: KS is latency-sized (chain <= ~4 chunks), capped by a PATH-
// dependent combine budget (TMA bulk-reduce: W/8, W/16 at nto=128; STG
// per-element red.global: absolute 2MB) and a 2-CTA/SM grid target; ring
// depth S is the deepest that fits the 111KB 2-CTA/SM co-residency cliff
// (8KB-granular smem rounding on the 228KB SM: 104KB co-resides, 112KB
// strands half the grid into a second serial wave) - grids that cannot
// exceed 1 CTA/SM get the full smem instead (deep ring is free there).
// Co-residency gate (2026-08-29). The tc5g/tc5gp K-split is a CROSS-CTA
// producer/consumer: slice 0 stores, slices >0 SPIN on pd_f16ks_flags[t] until
// it does. The split factor is elected from `2*nsm / U0` - i.e. on the
// assumption that this launch OWNS the machine. It does not when the caller
// runs a side stream: a slice>0 CTA then holds an SM waiting for a slice-0 CTA
// that will never be scheduled, and the device hangs at 100% with no progress
// and no error.
//
// Reproduced on qwen4_exp: batch 32 with every dense plane on this lane hangs
// forever; `PADDOCK_Q38FN_FORK=0` on the same binary clears it; batches 8 and
// 16 survive only because they land on the mmaf arm, which has no flag
// protocol. Not a plane shape (all 14 run clean at b32 in isolation), not
// graph capture, not the arm election.
//
// 0 = "another kernel may be resident, do not create cross-CTA dependencies".
// The launchers then clamp KS to 1, which is exactly the `KS > 1u` guard on
// every flag access, so the protocol is not merely avoided but unreachable.
// Read at DISPATCH time like the mmaf gate above, so a graph captured while it
// is clear bakes the KS=1 election.
static std::atomic<int> pd_f16_ks_gate{1};
PD_EXPORT int pd_f16_ksplit_set(int on) {
    pd_f16_ks_gate.store(on ? 1 : 0, std::memory_order_relaxed);
    return 0;
}

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 900)
static int pd_f16_gemm_tc5g_launch(const __half* w, const __half* x, float* y,
                                   float beta, unsigned in_dim, unsigned out_dim,
                                   unsigned batch, cudaStream_t st) {
    if (in_dim % 64u || batch > 128u) return (int)cudaErrorInvalidValue;
    static const uint32_t nsm = [] {
        int dev = 0, nn = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&nn, cudaDevAttrMultiProcessorCount, dev);
        return (uint32_t)nn;
    }();
    const uint32_t nto = batch <= 16u ? 16u : (batch + 15u) & ~15u;
    const uint32_t nk = (in_dim * 2u + 127u) / 128u;
    const bool tmaY = (out_dim & 3u) == 0u && !((uintptr_t)y & 15u) &&
                      (beta == 0.0f || beta == 1.0f);
    // paired-tile route only when the paired grid still covers the machine
    // on its own - at smaller U0p the X saving is eaten 1:1 by the combine
    // RMW needed to refill the grid (testbed: r64 16.4->18.9 worse)
    const bool pair = tmaY && nto >= 64u && out_dim > 128u &&
                      ((out_dim + 255u) >> 8) >= nsm;
    const uint32_t U0 = pair ? (out_dim + 255u) >> 8 : (out_dim + 127u) >> 7;
    if (U0 > 4096u) return (int)cudaErrorInvalidValue;   // flags cap
    uint32_t KS = (nk + 3u) / 4u;
    const uint64_t wBytes = (uint64_t)out_dim * in_dim * 2u;
    const uint64_t wFrac = wBytes / (nto >= 128u ? 16u : 8u);
    const uint64_t redBudget = tmaY && wFrac > 2000000ull ? wFrac : 2000000ull;
    const uint64_t redPer = (uint64_t)out_dim * nto * 4u;
    const uint32_t ksRed = 1u + (uint32_t)(redBudget / redPer);
    if (KS > ksRed) KS = ksRed;
    const uint32_t occ = U0 < 2u * nsm ? 2u * nsm / U0 : 1u;
    if (KS > occ) KS = occ;
    if (KS > nk) KS = nk;
    if (KS > 16u) KS = 16u;
    if (KS < 1u) KS = 1u;
    // Co-resident callers get KS=1: every flag access above is guarded by
    // `KS > 1u`, so this makes the cross-CTA spin unreachable rather than
    // merely unlikely. See pd_f16_ksplit_set.
    if (!pd_f16_ks_gate.load(std::memory_order_relaxed)) KS = 1u;
    CUtensorMap wm, xm, ym;
    if (!pd_f16t_tmap_3d(&wm, w, (uint64_t)in_dim * 2u, out_dim, 128u, 1u) ||
        !pd_f16t_tmap_3d(&xm, x, (uint64_t)in_dim * 2u, batch, nto, 1u))
        return (int)cudaErrorInvalidValue;
    if ((out_dim & 3u) == 0u && !((uintptr_t)y & 15u)) {
        pd_tmap_encode_fn enc = pd_tmap_encode();
        const cuuint64_t gdim[2] = {out_dim, batch};
        const cuuint64_t gstride[1] = {(cuuint64_t)out_dim * 4u};
        const cuuint32_t box[2] = {128u, 16u};
        const cuuint32_t estride[2] = {1u, 1u};
        // FLOAT32 dtype: cp.reduce's add is typed by the map (beta=1, K-split)
        if (!enc || enc(&ym, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2u, (void*)y, gdim,
                        gstride, box, estride, CU_TENSOR_MAP_INTERLEAVE_NONE,
                        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
            ym = xm;   // TMA epilogue self-disables in-kernel
    } else {
        ym = xm;       // dummy; the kernel elects the STG path
    }
    const uint32_t stageB = (tmaY && !pair) ? 16384u : 0u;
    const uint32_t slotB = (pair ? 32768u : 16384u) + nto * 128u;
    const uint32_t sBudget = U0 * KS <= nsm ? 224u * 1024u : 111u * 1024u;
    uint32_t S = 4u;
    while (S > 2u && S * slotB + stageB + (2u * S + 4u) * 8u > sBudget) --S;
    if (pair) S = 2u;    // only the <2> instantiation is dispatched for pair
    const uint32_t smem = S * slotB + stageB + (2u * S + 4u) * 8u;
    uint32_t ncols = 2u * nto;                    // pow2 >= 32
    while (ncols & (ncols - 1u)) ncols += ncols & (~ncols + 1u);
    if (ncols < 32u) ncols = 32u;
    static bool attr = false;
    if (!attr) {
        const int mx = (int)(4u * (16384u + 128u * 128u) + 16384u + 12u * 8u);
        cudaFuncSetAttribute((const void*)pd_f16_gemm_tc5g_kernel<2u>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, mx);
        cudaFuncSetAttribute((const void*)pd_f16_gemm_tc5g_kernel<3u>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, mx);
        cudaFuncSetAttribute((const void*)pd_f16_gemm_tc5g_kernel<4u>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, mx);
        cudaFuncSetAttribute((const void*)pd_f16_gemm_tc5gp_kernel<2u>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, mx);
        attr = true;
    }
    const uint32_t grid = U0 * KS;
    if (pair) {
        pd_f16_gemm_tc5gp_kernel<2u><<<grid, 192, smem, st>>>(
            wm, xm, ym, y, beta, in_dim, out_dim, batch, nto, KS, ncols);
        return (int)cudaGetLastError();
    }
    switch (S) {
    case 2u:
        pd_f16_gemm_tc5g_kernel<2u><<<grid, 192, smem, st>>>(
            wm, xm, ym, y, beta, in_dim, out_dim, batch, nto, KS, ncols);
        break;
    case 3u:
        pd_f16_gemm_tc5g_kernel<3u><<<grid, 192, smem, st>>>(
            wm, xm, ym, y, beta, in_dim, out_dim, batch, nto, KS, ncols);
        break;
    default:
        pd_f16_gemm_tc5g_kernel<4u><<<grid, 192, smem, st>>>(
            wm, xm, ym, y, beta, in_dim, out_dim, batch, nto, KS, ncols);
        break;
    }
    return (int)cudaGetLastError();
}
#endif  // launcher arch guard (tc5g)

// ---------------------------------------------------------------------------
// mmaf: fine-M mma.sync arm for the decode band, batch 5-32.
// Premise chain, all measured: every cross-CTA K-split combine
// protocol floors at ~2.6us (flag release vs spinner read-storm), while
// nvjet wins this band with fine 16-row tiles and no K-split. So: keep the
// tc5g producer (3D-box TMA ring, 16KB SW128 slabs) but consume with
// mma.sync m16n8k16 at a 32-row tile - grid fills via M, X stays whole in
// smem, K accumulates in registers, direct STG. Batch 17-32 splits across
// grid.y (tokens are disjoint outputs - zero protocol); NSLOT=4 halves the
// ring so two CTAs co-reside per SM, killing the second wave for grids up
// to 2*nsm (the CTA is ~pure latency: a second wave doubles the wall -
// fc1's 160-CTA grid measured exactly 2x before this).
// Kernel times vs the tc5g route (B200, whisper decode shapes):
//   wo b16 4.35 vs 6.95 (nvjet 4.46 - BEATEN)   wo b32 4.49 vs 7.40
//   qkv b32 6.71 vs 7.80                        fc1 b8 5.82 vs 7.71
// 1-deep ring bubble measured +0.2us (forced-N4 discrimination); the 2/SM
// residual is co-residency crowding (2x X-stage + ldmatrix per SM) - the
// ROWS=64 tile is the named follow-up arm.
// Structure: warp 0 of the producer warp issues every slab's TMA as soon as
// its slot frees (all of K<=6 slabs in flight at once); 8 compute warps =
// 4 warp-pair channels (pair p consumes slabs j%4==p, rowgroup rg splits
// the 32 rows); K-partials park in the pair's own primary ring slot and
// merge once behind bar.sync (intra-CTA, no protocol). ldmatrix A rides the
// TMA SW128 swizzle; X is staged [nto][K+8] (the +8 kills B-side bank
// conflicts, K%8 given in%64).
// The ROWS=64/CH=2 twin (nvjet's row geometry) keeps SLAB at 16KB (64 rows x
// 128B x 2 chunks, KEXT=128 halves); each warp owns RG=2 rowgroups with the
// B fragments hoisted and REUSED across both - per-row B-ldmatrix traffic
// halves and the grid halves, returning co-scheduled configs to 1 CTA/SM
// (qkv b32 6.71 -> 5.81, fc1 b8 5.82 -> 5.24, fc1 b32 newly electable 7.62
// vs tc5g 8.29).
template <uint32_t NSLOT, // 8: 2-deep channels, 1 CTA/SM; 4: 1-deep, 2/SM
          uint32_t ROWS>  // 32 (CH=4) or 64 (CH=2) out-rows per CTA
__global__ void __launch_bounds__(288, NSLOT == 4u ? 2 : 1)
pd_f16_gemm_mmaf_kernel(
    const __grid_constant__ PdTmap wmap,
    const __half* __restrict__ X, float* __restrict__ Y, float beta,
    uint32_t K, uint32_t M, uint32_t N, uint32_t nto) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && PD_MMA_OK
    constexpr uint32_t NCG = 2u;                         // B colgroups (nto16)
    constexpr uint32_t CH = ROWS == 32u ? 4u : 2u;       // 128B chunks per slab
    constexpr uint32_t RG = ROWS / 32u;                  // rowgroups per warp
    constexpr uint32_t SLAB = ROWS * 128u * CH;          // 16KB both geometries
    constexpr uint32_t KEXT = CH * 64u;                  // slab K-extent, halfs
    constexpr uint32_t NPAIR = 4u;
    const uint32_t KPAD = K + 8u;
    {   // batch-split: this CTA owns tokens [blockIdx.y*nto, ...)
        const uint32_t coff = blockIdx.y * nto;
        X += (size_t)coff * K; Y += (size_t)coff * M; N -= coff;
    }
    extern __shared__ __align__(1024) unsigned char shF[];
    unsigned char* wring = shF;                          // NSLOT x 16KB
    __half* shx = (__half*)(shF + NSLOT * SLAB);         // [nto][KPAD]
    uint64_t* bfull = (uint64_t*)(shx + (size_t)nto * KPAD);
    uint64_t* bempty = bfull + NSLOT;

    const uint32_t tid = threadIdx.x;
    const uint32_t nslab = (K + KEXT - 1u) / KEXT;
    if (tid == 0) {
        asm volatile("prefetch.tensormap [%0];" ::"l"(&wmap) : "memory");
        #pragma unroll
        for (uint32_t s = 0; s < NSLOT; ++s) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bfull[s])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 2;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bempty[s])));
        }
        // the generic-proxy init must reach the async proxy before the first
        // cp.async.bulk references these barriers (PTX fence.mbarrier_init) -
        // without it the TMA unit can credit complete_tx against the stale
        // pre-init word, a consumer's try_wait passes before data lands, and
        // the ldmatrix reads the previous launch's leftover smem (
        // ~1-per-100k stale-slab reads under co-resident tc5p TMA pressure,
        // the whisper overlap '!!!!' corruption)
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();

    auto bar_wait = [&](uint64_t* bar, uint32_t parity) {
        const uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
        // explicit acquire + memory clobber: the plain form leaves the wait
        // relaxed enough for the slab ldmatrix to be hoisted over the spin
        // (probed: stale-slab reads under co-resident tc5p TMA pressure)
        asm volatile("{\n\t.reg .pred p;\nW%=:\n\t"
                     "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%0], %1;\n\t"
                     "@!p bra W%=;\n\t}" ::"r"(a), "r"(parity) : "memory");
    };
    auto slot_of = [&](uint32_t j) {
        return NSLOT == 8u ? (j & 3u) * 2u + ((j >> 2) & 1u) : (j & 3u);
    };

    if (tid < 32) {
        // producer: every slab's TMA goes out as soon as its slot is free -
        // at K<=1536 (<=6 slabs) the whole W tile is in flight at once
        if (tid == 0) {
            const int rb = (int)(blockIdx.x * ROWS);
            uint32_t eph = 0;
            for (uint32_t j = 0; j < nslab; ++j) {
                const uint32_t s = slot_of(j);
                if (j >= NSLOT) { bar_wait(&bempty[s], (eph >> s) & 1u); eph ^= 1u << s; }
                const uint32_t m = (uint32_t)__cvta_generic_to_shared(&bfull[s]);
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                             ::"r"(m), "r"(SLAB));
                asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes"
                             " [%0], [%1, {0, %2, %3}], [%4];"
                             ::"r"((uint32_t)__cvta_generic_to_shared(wring + s * SLAB)),
                               "l"(&wmap), "r"(rb), "r"((int)(j * CH)), "r"(m) : "memory");
            }
        }
        return;                                          // producer warp done
    }

    // ---- compute warps: pair p = channel, rowgroup rg ----------------------
    const uint32_t ct = tid - 32u;                       // 0 .. 255
    const uint32_t lane = ct & 31u, w = ct >> 5;
    const uint32_t pair = w >> 1, rg = w & 1u;
    // X stage: whole [nto][KPAD], zero-pad cols >= N (K%8==0 given in%64)
    {
        const uint32_t kh8 = K / 8u;
        for (uint32_t i = ct; i < nto * kh8; i += 256u) {
            const uint32_t col = i / kh8, h8 = (i % kh8) * 8u;
            pd_f16_cpa16(&shx[(size_t)col * KPAD + h8],
                         X + (size_t)col * K + h8, col < N);
        }
        pd_f16_cpa_commit();
        pd_f16_cpa_waitN<0>();
        // MEMORY CLOBBER, not decoration. Without it the compiler may move
        // shared accesses across this barrier: racecheck (bench/mmaf_race.cu,
        // 2 iters) reported the merge read racing the park stores at 15360
        // hazards per site in <NSLOT=4,ROWS=64> and 7680 in <8,32>, and the
        // bit gate failed on 155 of 600 (plane,batch) runs. The bar_wait
        // helper above already carries the clobber for the same reason.
        asm volatile("bar.sync 1, 256;" ::: "memory");
    }

    const uint32_t g = lane >> 2, t = lane & 3u, l7 = lane & 7u;
    const uint32_t a_roff = ((lane & 8u) ? 8u : 0u) + l7;   // row within group
    const uint32_t a_kof = (lane & 16u) ? 8u : 0u;          // k +0/+8
    const uint32_t b_kof = (lane & 8u) ? 8u : 0u;

    float acc[RG][NCG][4] = {};
    uint32_t fph = 0;
    for (uint32_t j = pair; j < nslab; j += NPAIR) {
        const uint32_t s = slot_of(j);
        bar_wait(&bfull[s], (fph >> s) & 1u); fph ^= 1u << s;
        const uint32_t steps = (K - j * KEXT) >= KEXT ? KEXT / 16u
                                                      : (K - j * KEXT) / 16u;
        const uint32_t slotb = (uint32_t)__cvta_generic_to_shared(wring + s * SLAB);
        #pragma unroll 4
        for (uint32_t kk = 0; kk < steps; ++kk) {
            const uint32_t kh = kk * 16u + a_kof;           // half offset in slab
            const uint32_t c = kh >> 6;                     // 128B chunk
            const uint32_t u = (kh & 63u) >> 3;             // 16B unit in chunk
            const uint32_t gk = j * KEXT + kk * 16u + b_kof;
            // B first, once - every rowgroup of this warp reuses the fragments
            uint32_t b0[NCG], b1[NCG];
            #pragma unroll
            for (uint32_t cg = 0; cg < NCG; ++cg) {
                const uint32_t col = cg * 8u + l7;
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                             : "=r"(b0[cg]), "=r"(b1[cg])
                             : "r"((uint32_t)__cvta_generic_to_shared(
                                       &shx[(size_t)col * KPAD + gk])));
            }
            #pragma unroll
            for (uint32_t q = 0; q < RG; ++q) {
                const uint32_t r = q * 32u + rg * 16u + a_roff;
                uint32_t a0, a1, a2, a3;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                             : "r"(slotb + r * 128u + c * (ROWS * 128u) +
                                   ((u ^ (r & 7u)) << 4)));
                #pragma unroll
                for (uint32_t cg = 0; cg < NCG; ++cg) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+f"(acc[q][cg][0]), "+f"(acc[q][cg][1]),
                          "+f"(acc[q][cg][2]), "+f"(acc[q][cg][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                          "r"(b0[cg]), "r"(b1[cg]));
                }
            }
        }
        if (lane == 0)
            asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bempty[s]))
                         : "memory");
    }

    // K-partial merge: pairs 1-3 park their partials in their own primary
    // ring slot (they are done with it; the producer never refills past
    // nslab), pair 0 sums and stores. Fragment coords are identical across
    // pairs, so the merge is element-exact register addition.
    // A pair is two warps (rg=0,1) sharing the pair's ring slot, and the park
    // below OVERWRITES that slot. The loop's only synchronisation is the
    // per-slot mbarrier pair, which orders each warp against the PRODUCER --
    // nothing orders the two sibling warps against each other at loop exit.
    // So rg=0 could finish its last slab and start parking while rg=1 was
    // still issuing ldmatrix against the same slot:
    //   racecheck: Read f16_dense.cuh:2468 vs Writes 2499-2502,
    //   15360 hazards in <NSLOT=4,ROWS=64>, 7680 in <8,32>.
    // Worst where the slab walk is shortest and most uneven -- hc up is
    // K=320/KEXT=128 = 3 slabs across 4 pairs, and failed 31 of 40 at b16.
    // The barrier costs one CTA-wide sync per launch, off the K loop.
    asm volatile("bar.sync 1, 256;" ::: "memory");
    const uint32_t r0l = rg * 16u + g;                   // local row, this lane
    float* park = (float*)(wring + pair * (NSLOT / 4u) * SLAB);
    if (pair != 0u) {
        #pragma unroll
        for (uint32_t q = 0; q < RG; ++q)
            #pragma unroll
            for (uint32_t cg = 0; cg < NCG; ++cg) {
                const uint32_t c0 = cg * 8u + 2u * t, rq = q * 32u + r0l;
                park[(size_t)rq * nto + c0] = acc[q][cg][0];
                park[(size_t)rq * nto + c0 + 1u] = acc[q][cg][1];
                park[(size_t)(rq + 8u) * nto + c0] = acc[q][cg][2];
                park[(size_t)(rq + 8u) * nto + c0 + 1u] = acc[q][cg][3];
            }
    }
    // orders the park STORES above against the merge LOADS below -- the pair
    // that races here writes acc into its ring slot while pair 0 reads it.
    asm volatile("bar.sync 1, 256;" ::: "memory");
    if (pair != 0u) return;
    #pragma unroll
    for (uint32_t pp = 1; pp < NPAIR; ++pp) {
        const float* pk = (const float*)(wring + pp * (NSLOT / 4u) * SLAB);
        #pragma unroll
        for (uint32_t q = 0; q < RG; ++q)
            #pragma unroll
            for (uint32_t cg = 0; cg < NCG; ++cg) {
                const uint32_t c0 = cg * 8u + 2u * t, rq = q * 32u + r0l;
                acc[q][cg][0] += pk[(size_t)rq * nto + c0];
                acc[q][cg][1] += pk[(size_t)rq * nto + c0 + 1u];
                acc[q][cg][2] += pk[(size_t)(rq + 8u) * nto + c0];
                acc[q][cg][3] += pk[(size_t)(rq + 8u) * nto + c0 + 1u];
            }
    }

    // store: d0=(m=g,n=2t) d1=(m,2t+1) d2=(m+8,2t) d3=(m+8,2t+1); beta uniform
    #pragma unroll
    for (uint32_t q = 0; q < RG; ++q) {
        const uint32_t r0 = blockIdx.x * ROWS + q * 32u + rg * 16u + g;
        const uint32_t r8 = r0 + 8u;
        if (beta != 0.0f) {
            #pragma unroll
            for (uint32_t cg = 0; cg < NCG; ++cg) {
                const uint32_t c0 = cg * 8u + 2u * t, c1 = c0 + 1u;
                if (r0 < M) {
                    if (c0 < N) { float* o = &Y[(size_t)c0 * M + r0]; *o = acc[q][cg][0] + beta * *o; }
                    if (c1 < N) { float* o = &Y[(size_t)c1 * M + r0]; *o = acc[q][cg][1] + beta * *o; }
                }
                if (r8 < M) {
                    if (c0 < N) { float* o = &Y[(size_t)c0 * M + r8]; *o = acc[q][cg][2] + beta * *o; }
                    if (c1 < N) { float* o = &Y[(size_t)c1 * M + r8]; *o = acc[q][cg][3] + beta * *o; }
                }
            }
        } else {
            #pragma unroll
            for (uint32_t cg = 0; cg < NCG; ++cg) {
                const uint32_t c0 = cg * 8u + 2u * t, c1 = c0 + 1u;
                if (r0 < M) {
                    if (c0 < N) Y[(size_t)c0 * M + r0] = acc[q][cg][0];
                    if (c1 < N) Y[(size_t)c1 * M + r0] = acc[q][cg][1];
                }
                if (r8 < M) {
                    if (c0 < N) Y[(size_t)c0 * M + r8] = acc[q][cg][2];
                    if (c1 < N) Y[(size_t)c1 * M + r8] = acc[q][cg][3];
                }
            }
        }
    }
#else
    (void)wmap; (void)X; (void)Y; (void)beta; (void)K; (void)M; (void)N; (void)nto;
#endif
}

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 900)
static int pd_f16_gemm_mmaf_launch(const __half* w, const __half* x, float* y,
                                   float beta, unsigned in_dim, unsigned out_dim,
                                   unsigned batch, cudaStream_t st) {
    if (in_dim % 64u || batch == 0u || batch > 32u)
        return (int)cudaErrorInvalidValue;
    static const uint32_t nsm = [] {
        int dev = 0, nn = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&nn, cudaDevAttrMultiProcessorCount, dev);
        return (uint32_t)nn;
    }();
    const uint32_t nto = 16u;
    const uint32_t ns = (batch + nto - 1u) / nto;          // 1 or 2 token blocks
    const uint32_t xb = nto * (in_dim + 8u) * 2u + 16u * 8u; // X + barriers
    const bool s8 = 8u * 16384u + xb <= 227u * 1024u;
    const bool s4 = 4u * 16384u + xb <= 113u * 1024u;
    const uint32_t g32 = ((out_dim + 31u) / 32u) * ns;
    const uint32_t g64 = ((out_dim + 63u) / 64u) * ns;
    // wave model (validated on fc1: 160 CTAs -> exactly 2x wall): a second
    // wave doubles the ~pure-latency CTA. R32/N8 is the proven fast point at
    // 1/SM; R64 halves the grid AND the per-row B traffic when R32 would
    // co-schedule; R64/N4 half-ring co-residency is the last resort.
    uint32_t rows, nslot;
    if      (g32 <= nsm && s8)      { rows = 32u; nslot = 8u; }
    else if (g64 <= nsm && s8)      { rows = 64u; nslot = 8u; }
    else if (g64 <= 2u * nsm && s4) { rows = 64u; nslot = 4u; }
    else return (int)cudaErrorInvalidValue;
    const uint32_t smem = nslot * 16384u + xb;
    CUtensorMap wm;
    if (!pd_f16t_tmap_3d(&wm, w, (uint64_t)in_dim * 2u, out_dim, rows,
                         rows == 32u ? 4u : 2u))
        return (int)cudaErrorInvalidValue;
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute((const void*)pd_f16_gemm_mmaf_kernel<8u, 32u>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 232448);
        cudaFuncSetAttribute((const void*)pd_f16_gemm_mmaf_kernel<8u, 64u>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 232448);
        cudaFuncSetAttribute((const void*)pd_f16_gemm_mmaf_kernel<4u, 64u>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 232448);
        attr = true;
    }
    const dim3 grid((out_dim + rows - 1u) / rows, ns);
    if (rows == 32u)
        pd_f16_gemm_mmaf_kernel<8u, 32u><<<grid, 288u, smem, st>>>(
            wm, x, y, beta, in_dim, out_dim, batch, nto);
    else if (nslot == 8u)
        pd_f16_gemm_mmaf_kernel<8u, 64u><<<grid, 288u, smem, st>>>(
            wm, x, y, beta, in_dim, out_dim, batch, nto);
    else
        pd_f16_gemm_mmaf_kernel<4u, 64u><<<grid, 288u, smem, st>>>(
            wm, x, y, beta, in_dim, out_dim, batch, nto);
    return (int)cudaGetLastError();
}
#endif  // launcher arch guard (mmaf)
#endif  // PD_TC5_HOST

#ifndef PD_TC5_HOST
// No tc5g lane on this pack, so the K-split gate above has nothing to guard.
// The slot still has to resolve: exports.cuh names it unconditionally, and a
// build without sm_100 (a 4090, a 5090) otherwise dies at link with an
// undefined pd_f16_ksplit_set. A no-op keeps the table dense and the engine
// path identical whichever pack it loads.
PD_EXPORT int pd_f16_ksplit_set(int) { return 0; }
#endif  // !PD_TC5_HOST


// Capture-time mmaf election gate (dual-graph routing). Whisper's
// encode/decode overlap captures a SECOND decode-graph variant with the mmaf
// arm declined, replayed only on ticks whose admission encode is in flight on
// the side stream - the mmaf x tc5p stale-slab HW interaction (P39: ldmatrix
// reads pre-TMA slab content under cross-kernel bulk-DMA pressure, below the
// PTX contract, both fence remedies null) makes that pairing WER-poison while
// every other decode lane is overlap-clean. Read at DISPATCH time, so the
// election bakes into whatever graph is being captured; the env kills below
// still win when set. Set + read on the engine's serving thread; atomic only
// so a cross-thread caller is defined behavior, not because one exists.
static std::atomic<int> pd_f16_mmaf_gate{1};
PD_EXPORT int pd_f16_mmaf_set(int on) {
    pd_f16_mmaf_gate.store(on ? 1 : 0, std::memory_order_relaxed);
    return 0;
}

// Host entry - signature mirrors gemm_f16_f32_beta(w16, x16, y, in_dim,
// out_dim, batch, beta). Returns cudaGetLastError() (0 == success).
PD_EXPORT int pd_f16_gemm(const void* w, const void* x, void* y, float beta,
                           unsigned int in_dim, unsigned int out_dim,
                           unsigned int batch, void* stream) {
    if (out_dim == 0u || batch == 0u) return 0;
    const __half* W = (const __half*)w;
    const __half* X = (const __half*)x;
    float* Y = (float*)y;
    cudaStream_t st = (cudaStream_t)stream;
    // The mma path stages with 16B cp.async, which needs 16-byte-aligned f16
    // rows -> in_dim must be a multiple of 8. Ragged in_dim (muse's 588 vision
    // patch-embed, once per image) falls to the wmma path, which stages per
    // element and has no alignment constraint. PADDOCK_F16_WMMA forces it too.
    if (pd_f16_wmma_forced() || (in_dim & 7u) != 0u)
        return pd_f16_gemm_wmma_launch(W, X, Y, beta, in_dim, out_dim, batch, st);
#if defined(PD_TC5_HOST) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 900))
    // mmaf fine-tile arm (0.18.7), tried first in the 5-32 decode band: at
    // these rows it beats both the GEMV band (wo b5-8: 4.35 vs 7.0) and the
    // tc5g route (wo b16 4.35 vs 6.95 - beats nvjet's 4.46;
    // wo b32 4.49 vs 7.40; qkv b32 6.71 vs 7.80; fc1 b8 5.82 vs 7.71). The
    // launcher declines by grid/smem envelope (fc2's K=5120 X, the ~52K
    // head, fc1 b17-32's 320-CTA grid) and falls through to the chain below.
    // out_dim >= 1280 keeps the unmeasured tiny-M b5-8 class on GEMV.
    // Kill: PADDOCK_NO_F16MMAF.
    if (batch >= 5u && batch <= 32u && out_dim >= 1280u &&
        (in_dim % 64u) == 0u && pd_f16_tc5_on()) {
        static const bool moff = [] {
            const char* e = std::getenv("PADDOCK_NO_F16MMAF");
            return e != nullptr && e[0] != '\0' && !(e[0] == '0' && e[1] == '\0');
        }();
        if (!moff && pd_f16_mmaf_gate.load(std::memory_order_relaxed) != 0 &&
            pd_f16_gemm_mmaf_launch(W, X, Y, beta, in_dim, out_dim, batch, st) == 0)
            return 0;
    }
#endif
    // GEMV band (all arches): decode/draft rows where every tiled path pays
    // more fixed span than the whole W stream costs. Election from the
    // shape census: b<=4 always, b5-8 only small out_dim AND
    // small in_dim - the FMA-issue wall scales with in_dim*batch, and deep-K
    // b5-8 (whisper fc2: K=5120, GEMV ~17us vs tc5g 7.8 vs cuBLAS 9.6) was a
    // shipped regression in 0.18.5. The head class (out ~52K) stays on tc5g
    // (1.57x there) - its per-CTA X restage makes GEMV a loss at very wide
    // out. Kill: PADDOCK_NO_F16GEMV.
    if (batch <= 8u &&
        (batch <= 4u || (out_dim <= 2560u && in_dim <= 2048u)) &&
        out_dim <= 16384u) {
        static const bool voff = [] {
            const char* e = std::getenv("PADDOCK_NO_F16GEMV");
            return e != nullptr && e[0] != '\0' && !(e[0] == '0' && e[1] == '\0');
        }();
        if (!voff &&
            pd_f16_gemm_gemv_launch(W, X, Y, beta, in_dim, out_dim, batch, st) == 0)
            return 0;
    }
#if defined(PD_TC5_HOST) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 900))
    // Skinny ::1 arm on B200 (decode/draft rows, cuBLAS's weak regime):
    // batch <= 128, in_dim%64, any beta. Declines cleanly otherwise and
    // falls through to the mma twin. Dev kill: PADDOCK_NO_F16TC5G.
    if (batch <= 128u && pd_f16_tc5_on()) {
        static const bool goff = [] {
            const char* e = std::getenv("PADDOCK_NO_F16TC5G");
            return e != nullptr && e[0] != '\0' && !(e[0] == '0' && e[1] == '\0');
        }();
        if (!goff &&
            pd_f16_gemm_tc5g_launch(W, X, Y, beta, in_dim, out_dim, batch, st) == 0)
            return 0;
    }
    // Large-regular on B200: the ::2 duo arm. batch >= 256 keeps the 256-col
    // tile halves meaningfully full; below that the mma twin wins anyway.
    // A tmap-encode failure (unaligned base) falls through to the mma twin -
    // pd_f16_gemm_tc5d_launch returns before launching anything in that case.
    // The ::2 duo carries the same cross-CTA flag protocol AND a persistent
    // cluster grid sized to the machine, so a co-resident caller must not take
    // it either. Today this is unreachable in that state (batch >= 256 is
    // prefill, which does not fork), so the guard is belt-and-braces against a
    // future caller that batches wider.
    if (batch >= 256u && pd_f16_tc5_on() &&
        pd_f16_ks_gate.load(std::memory_order_relaxed)) {
        // ping-pong wide arm first (needs in_dim%64, out_dim%4, beta in {0,1}
        // - its launcher declines cleanly otherwise); the duo arm remains the
        // fallback for the %8-but-not-%64 K class.
        if (pd_f16_gemm_tc5p_elect(W, X, Y, beta, in_dim, out_dim, batch, st) == 0)
            return 0;
        if (pd_f16_gemm_tc5d_launch<4u>(W, X, Y, beta, in_dim, out_dim, batch, st) == 0)
            return 0;
    }
#endif
    return pd_f16_gemm_mma_launch(W, X, Y, beta, in_dim, out_dim, batch, st);
}
