// gemm/dense_f8_decode.cuh - e4m3 decode/GEMV lane: the non-tcgen05 decode
// kernels, the e4m3 GEMV keystone, the mma_ks twins and the per-row-scaled
// prefill class.
// Textually-included segment of the single pack translation unit.
// Not standalone-compilable: include order is defined by ../pack.cu.
//
// Split out of gemm/dense_fp4_w8.cuh (see gemm/dense_tc5.cuh for the why).
// Everything here is deliberately not tcgen05: v5 is a wide-grid
// rowwise e4m3 GEMV and v6 is warp-level mma.sync with no tensor memory, which
// is why the cut lands where it does.
//
// ORDER is LOAD-BEARING: must be included after gemm/dense_tc5.cuh - it uses
// pd_quantize_e4m3_row2_kernel, pd_rowmax_part_kernel, pd_rowq_chunks,
// pd_rowq_scr, pd_tc5p_fctr and pd_tc5q_ctr from there.
// ---- v5 decode GEMV: wide-grid rowwise e4m3, no tcgen05 -------------------
// From the B200 bring-up. Every tcgen05 CTA claims all 512
// TMEM columns for its epilogue ping-pong, so the die runs one CTA/SM:
// tc5q profiles at sm__warps_active 6.23% of peak (4 warps of 64) and
// dram__throughput 37.5%, and forcing a second CTA/SM collapses the tick 65x
// on TMEM contention. That ceiling is structural, not a tuning miss -- the
// dispatch heuristics around it all measured correct (NO_KSFOLD -8%,
// NO_NZ1W -16%, NO_L2EF -2%, ring depth flat).
//
// At batch <= 4 the mma earns nothing anyway: arithmetic intensity is ~2
// flops/byte, so the GEMM is pure streaming and the tensor cores idle. This
// route therefore drops tcgen05 and spends the die on memory parallelism --
// one WARP per output row, 8 warps/CTA, grid wide enough for >= 8 CTAs/SM.
// A plain-load probe on this die: 1.33 TB/s at the tcgen05 geometry
// (148x128), 6.93 TB/s at 1184x256.
//
// It consumes the same SW128 tile image, no second plane and no extra VRAM.
// The repacker's XOR permutes only the eight 16 B chunks within one 128 B
// span, since off16 = (r>>3)*64 + (r&7)*8 + (c ^ (r&7)) and c ^ (r&7) is a
// permutation of 0..7: so a row's 128 e4m3 bytes for one k-tile are 128
// CONTIGUOUS bytes, an 8-lane group loads them as a single coalesced
// transaction, and each lane recovers its own k range with the same XOR.
// x is [batch][in_dim] e4m3 and is read by every row, i.e. L2-hot (<= 70 KB).
// Scale fold matches tc5p exactly (y = D * wrs[row] * xrs[col]), which
// commutes with the grid.y K-split, so the ks combine contract is unchanged.
// Not bit-identical to tc5p: the K accumulation order differs (warp-strided
// 16-element chunks + shfl tree vs the mma's slab order).
//
// MEASURED, AND it loses -- off by DEFAULT. qwen3.6-27b c1, same binary,
// coherence character-identical to the tcgen05 route (so the tile-image
// indexing above is right):
//     tcgen05 baseline          reference
//     this route, x from L1     -27%
//     this route, x via smem    -35%
// Profiling the wide grid (2048x256, the in_qkv shape) says why the premise was
// wrong: Compute (SM) throughput 54.3% against DRAM 25.3%, 61% of stall
// cycles on L1TEX scoreboard. Dropping tcgen05 does buy the CTAs, but at b=1
// the CUDA-core path spends ~66 instructions per 16 MACs -- only 16 of them
// FMAs, the rest e4m3->f32 conversion -- so it goes ALU-bound long before it
// goes bandwidth-bound. The 6.93 TB/s a plain-load probe reaches is
// unreachable once every byte must be converted. Staging x in smem cut the
// conversions but cost more than it saved: the stage/compute phases
// serialize on two __syncthreads per k-chunk with no double buffering.
//
// So the occupancy ceiling is real but this is the wrong lever for it. What
// the numbers actually argue for is tensor cores AND high occupancy at once,
// i.e. warp-level mma.sync.aligned.m16n8k32...e4m3.e4m3.f32, which converts
// in the MMA datapath for free and uses no tensor memory, so it is not
// pinned to 1 CTA/SM the way tcgen05 is. Second candidate: take f32
// activations instead of e4m3 (W8A32) -- halves the inner-loop conversions
// AND deletes the rowmax+quantize pair, which costs 1.27 ms/tick.
__device__ __forceinline__ void pd_f8x16_to_f32(const uint4 v, float* __restrict__ f) {
    const unsigned short* p = (const unsigned short*)&v;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        __half2 h;
        *reinterpret_cast<__half2_raw*>(&h) =
            __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)p[i], __NV_E4M3);
        const float2 g = __half22float2(h);
        f[2 * i] = g.x;
        f[2 * i + 1] = g.y;
    }
}

// x is staged once per CTA, pre-converted to f32, and reused by all WARPS
// rows: v1 converted it per warp and the profile called that out (SM throughput
// 54.3% vs DRAM 25.3%, 61% of stall cycles on L1TEX scoreboard) -- at b=1 the
// naive form runs ~66 instructions per 16 MACs, only 16 of them the FMA.
// Lane jj owns k-chunk jj (contiguous k) and reads W slot jj^x8, so the eight
// lanes still cover one 128 B span in a single transaction while the x index
// becomes lane-linear. Chunks are padded to 17 floats: addresses are
// const + 17n + i for n = 0..31 and gcd(17,32) = 1, so the 32 lanes land on
// 32 distinct banks -- conflict-free (the natural 16-float stride puts all 32
// lanes on 2 banks, a 16-way conflict).
template <uint32_t WARPS, uint32_t KC>
__global__ void __launch_bounds__(WARPS * 32) pd_f8row_gemv_wide_kt(
    const unsigned char* __restrict__ wtiles, const float* __restrict__ wrs,
    const unsigned char* __restrict__ xq, const float* __restrict__ xrs,
    float* __restrict__ y, uint32_t in_dim, uint32_t out_dim, uint32_t batch,
    uint32_t nz) {
    constexpr uint32_t XS = KC * 8u * 17u;         // padded floats per batch row
    __shared__ float xs[XS * 4u];
    const uint32_t tid = threadIdx.x, lane = tid & 31u;
    const uint32_t row = blockIdx.x * WARPS + (tid >> 5);
    const bool live = row < out_dim;
    const uint32_t nkt = in_dim >> 7;
    const uint32_t ktpz = (nkt + nz - 1u) / nz;
    const uint32_t kt0 = blockIdx.y * ktpz;                 // uniform per CTA
    const uint32_t kt1 = (kt0 + ktpz) < nkt ? (kt0 + ktpz) : nkt;
    if (kt0 >= kt1) return;

    const uint32_t r128 = row & 127u, x8 = r128 & 7u;
    const size_t rowbase = ((size_t)(row >> 7) * nkt) << 14;
    const uint32_t rowoff = (((r128 >> 3) * 64u) + (x8 * 8u)) << 4;
    const uint32_t g = lane >> 3, jj = lane & 7u;
    const uint32_t slot = (jj ^ x8) << 4;   // where chunk jj physically lives

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t kb = kt0; kb < kt1; kb += KC) {
        const uint32_t ktn = (kb + KC) < kt1 ? KC : (kt1 - kb);
        __syncthreads();
        for (uint32_t ci = tid; ci < ktn * 8u * batch; ci += WARPS * 32u) {
            const uint32_t c = ci / (ktn * 8u), r = ci - c * ktn * 8u;
            const uint4 xv = *(const uint4*)(xq + (size_t)c * in_dim
                                             + (size_t)kb * 128u + r * 16u);
            float f[16];
            pd_f8x16_to_f32(xv, f);
            float* d = xs + c * XS + r * 17u;
            #pragma unroll
            for (int i = 0; i < 16; ++i) d[i] = f[i];
        }
        __syncthreads();
        if (!live) continue;
        for (uint32_t t = 0; t < ktn; t += 4u) {
            const uint32_t tg = t + g;
            if (tg >= ktn) continue;
            const uint4 wv = *(const uint4*)(wtiles + rowbase
                                             + ((size_t)(kb + tg) << 14) + rowoff + slot);
            float wf[16];
            pd_f8x16_to_f32(wv, wf);
            #pragma unroll
            for (uint32_t c = 0; c < 4u; ++c) {
                if (c < batch) {
                    const float* xr = xs + c * XS + (tg * 8u + jj) * 17u;
                    // four chains: a single accumulator serializes 16 FMAs
                    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < 16; i += 4) {
                        s0 = fmaf(wf[i + 0], xr[i + 0], s0);
                        s1 = fmaf(wf[i + 1], xr[i + 1], s1);
                        s2 = fmaf(wf[i + 2], xr[i + 2], s2);
                        s3 = fmaf(wf[i + 3], xr[i + 3], s3);
                    }
                    acc[c] += (s0 + s1) + (s2 + s3);
                }
            }
        }
    }
    if (!live) return;
    #pragma unroll
    for (uint32_t c = 0; c < 4u; ++c)
        #pragma unroll
        for (uint32_t sh = 16; sh > 0; sh >>= 1)
            acc[c] += __shfl_xor_sync(0xffffffffu, acc[c], sh);
    if (lane == 0) {
        const float ws = wrs[row];
        float* dst = y + (size_t)blockIdx.y * out_dim * batch;
        for (uint32_t c = 0; c < batch; ++c)
            dst[(size_t)c * out_dim + row] = acc[c] * ws * xrs[c];
    }
}

// ---- v6 decode GEMM: warp-level mma.sync e4m3, no tensor memory ----------
// The synthesis of the two B200 bring-up measurements. tcgen05 (tc5p/tc5q)
// gives tensor cores but every CTA claims all 512 TMEM columns, pinning the
// die to one CTA/SM -- sm__warps_active 6.23% of peak, dram 37.5%. The
// CUDA-core GEMV above gives the occupancy but goes ALU-bound converting
// e4m3->f32 by hand (SM 54.3% vs dram 25.3%, -27% end to end).
// mma.sync.aligned.m16n8k32...e4m3.e4m3.f32 is the one primitive that gives
// both: the conversion happens inside the MMA datapath for free, and it is a
// warp-level op on plain registers, so it touches no tensor memory and the
// occupancy is set by smem/registers like any ordinary kernel.
//
// Mapping: M=16 output rows, N=8 batch columns (the whole serving band in one
// tile), K=32. A warp owns 16 rows; the CTA owns WARPS*16.
//
// The tile image cooperates twice over. First, rows [16a, 16a+16) occupy
// exactly the 2 KB at (r>>3)*1024 inside each 16 KB tile, so a warp's whole
// k-tile slab is CONTIGUOUS and stages with four fully coalesced 16 B/lane
// loads. Second, the fragment reads out of that slab are conflict-free by
// construction: a-lane address/4 mod 32 is ((j0^gid)*4 + tig) mod 32, which
// is a bijection over the 32 lanes because gid indexes the XOR. x gets a
// 144 B padded stride so its 8 columns land on 4*gid + tig -- also distinct.
//
// Scale fold is tc5p's (y = D * wrs[row] * xrs[col]) and commutes with the
// grid.y K-split, so `part` and the ks combine keep their contract. Not
// bit-identical to tc5p: different K accumulation order.
//
// MEASURED, AND it wins on the big fused plane. qwen3.6-27b c1, same binary,
// coherence character-identical to tcgen05 throughout:
//     tcgen05 baseline                        reference
//     wmma on all f8t shapes                  -3.7%
//     wmma at tiles >= 100 (adds qkv/in_qkv)  +2.8%
//     wmma at tiles >= 256 (gate_up only)     +5.9%
//     + f8 lm_head at b=1                     +7.7%
// Profiling the gate_up launch (256x5 and 544x3, 128 thr): sm__warps_active
// 48.4% of peak against tcgen05's 6.23%, dram 40.7% against tc5q's 37.5%.
// The occupancy premise held; what limits it now is grid size, not TMEM --
// launch__waves_per_multiprocessor is 0.92 on gate_up and only 0.36 on the
// 40-tile shapes, which is why the shape gate exists: ctas = out/64 and the
// K-split cannot grow past the 8 planes `part` is sized for, so small planes
// never fill the die and tc5p keeps them.
//
// nz is deliberately auto (target 8 CTAs/SM) rather than "as large as
// possible" -- the wave quantization is sharp: nz=3 is 0.92 waves and wins,
// nz=4 is 1.22 waves (a nearly empty second wave) and drops below even nz=2.
// Forcing it is PADDOCK_F8T_WMMA_NZ.
//
// PIPELINING. The k-tile loop is now a cp.async two-stage
// pipeline (MODE 1, default). Three forms live in one binary so the A/B is
// same-binary; PADDOCK_F8T_WMMA_DB picks. Final, with minBlocks pinned:
//     MODE 0  single buffer, direct copy     reference
//     MODE 1  cp.async two-stage             +1.5%
//     MODE 2  register-prefetch two-stage    ~-13%  (spills, see below)
//
// The instructive part is that pipelining only pays once its register cost is
// capped, and the first two attempts lost for that reason alone. Before
// pinning minBlocks:
//     mode  regs/thr  blocks/SM  warps_active   dram
//       0        39        12        65.3%     45.0%
//       1        51         9        45.5%     51.3%
//       2        64         8        40.8%     45.4%
// cp.async did exactly what it was meant to -- dram 45.0 -> 51.3% -- but paid
// 12 registers for it, and on this kernel occupancy was worth more, so it
// came out behind (98.88 vs 99.58). __launch_bounds__(128, 12) makes ptxas
// budget 65536/(12*128) = 42 registers instead of spending freely, which buys
// the pipelining back at full occupancy. Two traps found on the way:
//   - routing the single-buffer copy through a uint4 staging array costs 16
//     live registers and 2.5% end to end; MODE 0 copies global->smem direct.
//   - `minBlocks = 1` is not the same as omitting minBlocks. Spelling it out
//     for MODE 0 changed ptxas's heuristics and cost 2.7%. The bound is flat
//     at 12 for every mode instead.
// MODE 2 (plain loads prefetched via registers) is kept only as the measured
// counterexample: at 64 regs/thread the 42-register budget spills it.
#if PD_MMA_OK && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
#define PD_WMMA_DEV_OK 1
#else
#define PD_WMMA_DEV_OK 0
#endif

// W is a pure stream -- .cg (bypass L1, it is never reused). x is the same
// 1 KB for every CTA on the die, so it wants to stay in L1: .ca.
static __device__ __forceinline__ void pd_wmma_cpa16(void* smem, const void* gmem) {
    const unsigned sm = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(sm), "l"(gmem)
                 : "memory");
}

static __device__ __forceinline__ void pd_wmma_cpa16_ca(void* smem, const void* gmem) {
    const unsigned sm = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(sm), "l"(gmem)
                 : "memory");
}

// MODE 0 = single buffer, plain loads.
// MODE 1 = cp.async two-stage pipeline.
// MODE 2 = two-stage pipeline with PLAIN loads prefetched through registers:
//          the global fetch for tile kt+1 is issued before the mmas for tile
//          kt and only consumed (stored to smem) after them, so its latency
//          hides under the mma stream without cp.async.
// The pipelined modes are worth registers only if they keep the occupancy
// that makes this kernel win: MODE 1 lifts dram 45.0 -> 51.3%
// but costing 39 -> 51 registers, which drops 12 blocks/SM to 9 and warps
// 65.3% -> 45.5%. Pin minBlocks so ptxas budgets 65536/(12*128) = 42
// registers instead of spending them freely. MODE 0 already fits at 39.
// NT = mma n-tiles per warp, i.e. ceil(batch/8), 1..4 for batch 1..32. The A
// fragment is the weight tile -- the entire bandwidth cost -- and it does not
// depend on the n-tile, so every extra n-tile rides the same W fetch for 4 more
// accumulator registers and one more mma issue. Arithmetic intensity therefore
// scales with NT at constant DRAM traffic, which is the whole reason this beats
// handing the batch to tc5q.
// That is what keeps batch 9..32 on tensor cores instead of falling off this
// route at the N=8 instruction boundary: the b=8 -> b=9 step measured an 18%
// cliff in per-request rate against the 1-3% per-step decline either side of
// it, and at c32 the whole band was running tc5q, which profiles at 1 CTA/SM
// because each CTA claims all 512 TMEM cols.
// Wider n-tilings cost smem, and smem is what sets minBlocks here. Per CTA the
// buffers are NB*WARPS*WSL (16384 for the shipped WARPS=4, MODE!=0) plus
// NB*NT*8*XST, against the die's 233472 B:
//   NT=1  18688 B -> 12 CTAs   NT=3  23296 B -> 10 CTAs
//   NT=2  20992 B -> 11 CTAs   NT=4  25600 B ->  9 CTAs
// All four stay under the 48 KB STATIC shared limit, so no arch needs
// cudaFuncSetAttribute and the kernel remains portable to the ~100 KB/SM dies
// (8.9, 12.x) even though the host dispatch only elects it on sm_100.
// NT=1 keeps the shipped 12 and its exact codegen.
template <uint32_t WARPS, int MODE, uint32_t NT>
__global__ void __launch_bounds__(WARPS * 32,
                                  NT == 1u ? 12 : NT == 2u ? 11 : NT == 3u ? 10 : 9)
pd_f8row_gemm_wmma_kt(
    const unsigned char* __restrict__ wtiles, const float* __restrict__ wrs,
    const unsigned char* __restrict__ xq, const float* __restrict__ xrs,
    float* __restrict__ y, uint32_t in_dim, uint32_t out_dim, uint32_t batch,
    uint32_t nz) {
#if PD_WMMA_DEV_OK
    constexpr uint32_t WSL = 2048u;   // W bytes per warp per k-tile (16 rows)
    constexpr uint32_t XST = 144u;    // padded x bytes per batch column
    constexpr uint32_t XBUF = NT * 8u * XST;
    constexpr uint32_t NB = MODE == 0 ? 1u : 2u;   // MODE 0 aliases buf1->buf0
    __shared__ __align__(16) unsigned char sh[NB * WARPS * WSL + NB * XBUF];
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const uint32_t gid = lane >> 2, tig = lane & 3u;   // mma fragment coords
    unsigned char* const wb0 = sh + warp * WSL;
    unsigned char* const wb1 = sh + (NB == 2u ? (WARPS + warp) : warp) * WSL;
    unsigned char* const xb0 = sh + NB * WARPS * WSL;
    unsigned char* const xb1 = xb0 + (NB == 2u ? XBUF : 0u);

    const uint32_t rblk = (blockIdx.x * WARPS + warp) * 16u;
    const uint32_t nkt = in_dim >> 7;
    const uint32_t ktpz = (nkt + nz - 1u) / nz;
    const uint32_t kt0 = blockIdx.y * ktpz;
    const uint32_t kt1 = (kt0 + ktpz) < nkt ? (kt0 + ktpz) : nkt;
    if (kt0 >= kt1) return;                       // uniform: blockIdx.y only
    const bool live = rblk < out_dim;
    const size_t trbase = ((size_t)(rblk >> 7) * nkt) << 14;
    const uint32_t blkoff = ((rblk & 127u) >> 3) << 10;

    // x columns >= batch are zero for every k-tile, so zero both buffers once
    // here and let the per-tile cp.async touch only the live columns (cp.async
    // cannot synthesise zeros, and re-zeroing per tile would need a second
    // barrier inside the pipeline)
    for (uint32_t i = tid; i < (NB * XBUF) >> 2; i += WARPS * 32u)
        ((uint32_t*)xb0)[i] = 0u;
    __syncthreads();

    // x staging is one 16 B chunk per (column, eighth of the 128 B k-tile), so
    // batch*8 chunks over NTH threads. Through NT=2 that is at most one chunk
    // per thread and XIT is 1, which reproduces the single-pass form exactly
    // (s=0, c=tid). Wider n-tilings take XIT passes rather than more threads:
    // growing WARPS instead would halve the CTAs per output tile, and the CTA
    // count is what this route wins on.
    constexpr uint32_t NTH = WARPS * 32u;
    constexpr uint32_t XIT = (NT * 64u + NTH - 1u) / NTH;

    auto astage = [&](uint32_t kt, unsigned char* wd, unsigned char* xd) {
        #pragma unroll
        for (uint32_t s = 0; s < XIT; ++s) {
            const uint32_t c = tid + s * NTH;
            if (c < batch * 8u) {
                const uint32_t xn = c >> 3, xqi = c & 7u;
                pd_wmma_cpa16_ca(xd + xn * XST + xqi * 16u,
                                 xq + (size_t)xn * in_dim + (size_t)kt * 128u
                                     + xqi * 16u);
            }
        }
        if (live) {
            const unsigned char* src = wtiles + trbase + ((size_t)kt << 14) + blkoff;
            #pragma unroll
            for (uint32_t i = 0; i < 4u; ++i)
                pd_wmma_cpa16(wd + i * 512u + lane * 16u, src + i * 512u + lane * 16u);
        }
    };
    auto gload = [&](uint32_t kt, uint4* wr, uint4* xr) {
        if (live) {
            const unsigned char* src = wtiles + trbase + ((size_t)kt << 14) + blkoff;
            #pragma unroll
            for (uint32_t i = 0; i < 4u; ++i)
                wr[i] = *(const uint4*)(src + i * 512u + lane * 16u);
        }
        #pragma unroll
        for (uint32_t s = 0; s < XIT; ++s) {
            const uint32_t c = tid + s * NTH;
            if (c < batch * 8u) {
                const uint32_t xn = c >> 3, xqi = c & 7u;
                xr[s] = *(const uint4*)(xq + (size_t)xn * in_dim
                                        + (size_t)kt * 128u + xqi * 16u);
            }
        }
    };
    auto sstore = [&](unsigned char* wd, unsigned char* xd, const uint4* wr,
                      const uint4* xr) {
        if (live) {
            #pragma unroll
            for (uint32_t i = 0; i < 4u; ++i)
                *(uint4*)(wd + i * 512u + lane * 16u) = wr[i];
        }
        #pragma unroll
        for (uint32_t s = 0; s < XIT; ++s) {
            const uint32_t c = tid + s * NTH;
            if (c < batch * 8u) {
                const uint32_t xn = c >> 3, xqi = c & 7u;
                *(uint4*)(xd + xn * XST + xqi * 16u) = xr[s];
            }
        }
    };

    float d[NT][4];
    #pragma unroll
    for (uint32_t t = 0; t < NT; ++t) {
        d[t][0] = 0.f; d[t][1] = 0.f; d[t][2] = 0.f; d[t][3] = 0.f;
    }
    uint4 wr[4];
    uint4 xr[XIT];
    #pragma unroll
    for (uint32_t s = 0; s < XIT; ++s) xr[s] = make_uint4(0u, 0u, 0u, 0u);
    if (MODE == 1) {
        astage(kt0, wb0, xb0);
        asm volatile("cp.async.commit_group;" ::: "memory");
    } else if (MODE == 2) {
        gload(kt0, wr, xr);
        sstore(wb0, xb0, wr, xr);
        __syncthreads();
    }
    for (uint32_t kt = kt0; kt < kt1; ++kt) {
        const uint32_t par = MODE == 0 ? 0u : ((kt - kt0) & 1u);
        const unsigned char* const wc = par ? wb1 : wb0;
        const unsigned char* const xc = par ? xb1 : xb0;
        const bool more = (kt + 1u) < kt1;      // CTA-uniform
        if (MODE == 0) {
            // this barrier is the one the pipelined paths put at the BOTTOM
            // of the loop: it retires the previous tile's readers before the
            // buffer is refilled, so every mode costs 2 barriers per tile
            __syncthreads();
            // DIRECT global->smem copy, deliberately not via gload/sstore:
            // routing through the wr[4] staging array costs 16 live registers
            // and this kernel is occupancy-limited by registers (12
            // blocks/SM), so the prefetch buffer would pay for itself twice
            #pragma unroll
            for (uint32_t s = 0; s < XIT; ++s) {
                const uint32_t c = tid + s * NTH;
                if (c < batch * 8u) {
                    const uint32_t xn = c >> 3, xqi = c & 7u;
                    *(uint4*)(xb0 + xn * XST + xqi * 16u) = *(const uint4*)(
                        xq + (size_t)xn * in_dim + (size_t)kt * 128u + xqi * 16u);
                }
            }
            if (live) {
                const unsigned char* src = wtiles + trbase + ((size_t)kt << 14) + blkoff;
                #pragma unroll
                for (uint32_t i = 0; i < 4u; ++i)
                    *(uint4*)(wb0 + i * 512u + lane * 16u) =
                        *(const uint4*)(src + i * 512u + lane * 16u);
            }
            __syncthreads();
        } else if (MODE == 1) {
            // issue tile kt+1 before consuming kt so the fetch overlaps the
            // mmas; `more` is CTA-uniform, so the wait immediate is too
            if (more) {
                astage(kt + 1u, par ? wb0 : wb1, par ? xb0 : xb1);
                asm volatile("cp.async.commit_group;" ::: "memory");
                asm volatile("cp.async.wait_group 1;" ::: "memory");
            } else {
                asm volatile("cp.async.wait_group 0;" ::: "memory");
            }
            __syncthreads();
        } else if (more) {
            gload(kt + 1u, wr, xr);   // in flight across the mmas below
        }
        if (live) {
            #pragma unroll
            for (uint32_t j0 = 0; j0 < 8u; j0 += 2u) {
                const uint32_t o0 = gid * 128u + ((j0 ^ gid) << 4) + tig * 4u;
                const uint32_t o1 = gid * 128u + (((j0 + 1u) ^ gid) << 4) + tig * 4u;
                const uint32_t a0 = *(const uint32_t*)(wc + o0);
                const uint32_t a1 = *(const uint32_t*)(wc + 1024u + o0);
                const uint32_t a2 = *(const uint32_t*)(wc + o1);
                const uint32_t a3 = *(const uint32_t*)(wc + 1024u + o1);
                // a0..a3 are n-tile invariant: both mmas consume the same W
                // fragment, so the second n-tile adds issue slots, not DRAM
                #pragma unroll
                for (uint32_t t = 0; t < NT; ++t) {
                    const uint32_t xbo = (gid + t * 8u) * XST + j0 * 16u + tig * 4u;
                    const uint32_t b0 = *(const uint32_t*)(xc + xbo);
                    const uint32_t b1 = *(const uint32_t*)(xc + xbo + 16u);
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+f"(d[t][0]), "+f"(d[t][1]), "+f"(d[t][2]), "+f"(d[t][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
                }
            }
        }
        // the other buffer is refilled now (MODE 2) or two iterations on
        // (MODE 1); either way every reader must clear first
        if (MODE == 1) {
            __syncthreads();
        } else if (MODE == 2) {
            __syncthreads();
            if (more) {
                sstore(par ? wb0 : wb1, par ? xb0 : xb1, wr, xr);
                __syncthreads();
            }
        }
    }
    if (!live) return;
    const uint32_t r0 = rblk + gid, r1 = r0 + 8u;
    const float w0 = wrs[r0], w1 = wrs[r1];
    float* dst = y + (size_t)blockIdx.y * out_dim * batch;
    #pragma unroll
    for (uint32_t t = 0; t < NT; ++t) {
        const uint32_t c0 = t * 8u + tig * 2u, c1 = c0 + 1u;
        if (c0 < batch) {
            const float s = xrs[c0];
            dst[(size_t)c0 * out_dim + r0] = d[t][0] * w0 * s;
            dst[(size_t)c0 * out_dim + r1] = d[t][2] * w1 * s;
        }
        if (c1 < batch) {
            const float s = xrs[c1];
            dst[(size_t)c1 * out_dim + r0] = d[t][1] * w0 * s;
            dst[(size_t)c1 * out_dim + r1] = d[t][3] * w1 * s;
        }
    }
#else
    (void)wtiles; (void)wrs; (void)xq; (void)xrs; (void)y;
    (void)in_dim; (void)out_dim; (void)batch; (void)nz;
#endif
}

// tc5q ring depth vs CTA occupancy (from the B200 bring-up). The shipped
// S=6 ring is 6 x 24 KB = 144 KB of the die's 228 KB smem, so
// launch__occupancy_limit_shared_mem pins tc5q at one CTA/SM: 128 threads =
// 4 warps of the 64 slots, and sm__warps_active measures 6.23% of
// peak with dram__throughput at 37.5%. A shallower ring fits more CTAs/SM
// (S=4 -> 96 KB -> 2; S=3 -> 72 KB -> 3) and trades stream length for warp
// parallelism. Which side wins is a per-die question and the persistent
// item loop is grid-agnostic (the dry-claim reset keys off gridDim.x), so
// both axes are selectable: PADDOCK_TC5Q_S and PADDOCK_TC5Q_CTA.
// pdist stays S-1, the shipped S=6/QD=5 relation.
template <uint32_t S_>
static inline void pd_tc5q_launch(bool ef, uint32_t grid, cudaStream_t st,
                                  const unsigned char* wtiles, const CUtensorMap& ym,
                                  const float* wrs, const float* xrs, float* dst,
                                  uint32_t in_dim, uint32_t out_dim, uint32_t batch,
                                  uint32_t nzq, uint32_t* qctr) {
#if defined(PD_BS_HOST) && defined(PD_TC5_HOST)
    const uint32_t sm_ = S_ * 24576u + 2u * S_ * 8u + 16u;
    static bool at = false;
    if (!at) {
        cudaFuncSetAttribute((const void*)pd_f8row_gemm_tc5q_kt<S_, true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm_);
        cudaFuncSetAttribute((const void*)pd_f8row_gemm_tc5q_kt<S_, false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm_);
        at = true;
    }
    if (ef)
        pd_f8row_gemm_tc5q_kt<S_, true><<<grid, 128, sm_, st>>>(
            wtiles, ym, wrs, xrs, dst, in_dim, out_dim, batch, nzq, S_ - 1u, qctr);
    else
        pd_f8row_gemm_tc5q_kt<S_, false><<<grid, 128, sm_, st>>>(
            wtiles, ym, wrs, xrs, dst, in_dim, out_dim, batch, nzq, S_ - 1u, qctr);
#else
    (void)ef; (void)grid; (void)st; (void)wtiles; (void)ym; (void)wrs; (void)xrs;
    (void)dst; (void)in_dim; (void)out_dim; (void)batch; (void)nzq; (void)qctr;
#endif
}

// tc5p ring depth, selectable. Same smem-vs-depth trade as tc5q's, but with the
// OPPOSITE answer available, because tc5p's grid is not pinned to nsm. tc5q is
// persistent at exactly one CTA per SM, so a shallower ring buys nothing (the
// S=3 probe left sm__warps_active at 6.23% and got slower). tc5p's grid is
// tiles*nz: at batch 32 the K-split election gives out_dim 5120 a 40x7 = 280
// CTA grid, so shrinking the ring can put a SECOND CTA on each SM instead of
// leaving 64-4 = 60 warp slots idle.
//
// Why it is worth trying: tc5p profiles at
// launch__occupancy_limit_shared_mem=1, sm__warps_active 6.2%, and 3.03 TB/s
// effective against tc5q's 4.37 on the same die -- it is the weakest GEMM
// route we have and worth ~1.1-1.5 ms of the 2.1 ms/step the decode tick
// needs to shed.
//   S=6 147568 B -> 1 CTA/SM      S=4  98368 B -> 2 CTA/SM
//   S=5 122960 B -> 1 CTA/SM      S=3  73776 B -> 3 CTA/SM
template <uint32_t S_>
static inline void pd_tc5p_launch(bool ef, bool pdl, uint32_t tiles, uint32_t nz,
                                  cudaStream_t st, const unsigned char* wtiles,
                                  const CUtensorMap& ym, const float* wrs,
                                  const float* xrs, float* dst, uint32_t in_dim,
                                  uint32_t out_dim, uint32_t batch, float* y,
                                  uint32_t* fctr, uint32_t l2pf) {
#if defined(PD_BS_HOST) && defined(PD_TC5_HOST)
    constexpr uint32_t D_ = S_ - 1u;
    const uint32_t smem = S_ * 24576u + 2u * S_ * 8u;
    static bool at = false;
    if (!at) {
        cudaFuncSetAttribute((const void*)pd_f8row_gemm_tc5p_kt<S_, true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        cudaFuncSetAttribute((const void*)pd_f8row_gemm_tc5p_kt<S_, false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        cudaFuncSetAttribute((const void*)pd_f8row_gemm_tc5p_kt<S_, true, true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        cudaFuncSetAttribute((const void*)pd_f8row_gemm_tc5p_kt<S_, false, true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        at = true;
    }
    bool launched = false;
    if (pdl) {
        cudaLaunchConfig_t cfg = {};
        cudaLaunchAttribute a[1];
        a[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        a[0].val.programmaticStreamSerializationAllowed = 1;
        cfg.gridDim = dim3(tiles, nz); cfg.blockDim = dim3(128);
        cfg.dynamicSmemBytes = smem; cfg.stream = st;
        cfg.attrs = a; cfg.numAttrs = 1;
        const cudaError_t le = ef
            ? cudaLaunchKernelEx(&cfg, pd_f8row_gemm_tc5p_kt<S_, true, true>,
                  wtiles, ym, wrs, xrs, dst, in_dim, out_dim, batch, nz, D_,
                  y, fctr, l2pf)
            : cudaLaunchKernelEx(&cfg, pd_f8row_gemm_tc5p_kt<S_, false, true>,
                  wtiles, ym, wrs, xrs, dst, in_dim, out_dim, batch, nz, D_,
                  y, fctr, l2pf);
        launched = le == cudaSuccess;
    }
    if (!launched) {
        if (ef)
            pd_f8row_gemm_tc5p_kt<S_, true><<<dim3(tiles, nz), 128, smem, st>>>(
                wtiles, ym, wrs, xrs, dst, in_dim, out_dim, batch, nz, D_, y, fctr, 0u);
        else
            pd_f8row_gemm_tc5p_kt<S_, false><<<dim3(tiles, nz), 128, smem, st>>>(
                wtiles, ym, wrs, xrs, dst, in_dim, out_dim, batch, nz, D_, y, fctr, 0u);
    }
#else
    (void)ef; (void)pdl; (void)tiles; (void)nz; (void)st; (void)wtiles; (void)ym;
    (void)wrs; (void)xrs; (void)dst; (void)in_dim; (void)out_dim; (void)batch;
    (void)y; (void)fctr; (void)l2pf;
#endif
}

// Both K-split elections (wmma here, tc5p/tc5q below) price the partial planes
// the split creates, a term that scales with batch. Shared kill switch.
static inline int pd_nzbatch_on() {
    static int v = -1;
    if (v < 0) v = pd_env("PADDOCK_NO_NZBATCH") ? 0 : 1;
    return v;
}

static int pd_f8t_gemm_impl(const void* wtiles, const void* wrs, const void* xq,
                            const void* xrs, void* part, void* y, uint32_t in_dim,
                            uint32_t out_dim, uint32_t batch, uint32_t no_combine,
                            uint32_t* out_nz, void* stream) {
#if !defined(PD_BS_HOST) || !defined(PD_TC5_HOST)
    (void)wtiles; (void)wrs; (void)xq; (void)xrs; (void)part; (void)y;
    (void)in_dim; (void)out_dim; (void)batch; (void)stream;
    return cudaErrorNotSupported;
#else
    if (out_dim == 0 || batch == 0) return 0;
    if ((in_dim & 127u) || (out_dim & 127u)) return cudaErrorInvalidValue;
    auto st = (cudaStream_t)stream;
    // Warp-level mma.sync e4m3 route (see pd_f8row_gemm_wmma_kt): tensor
    // cores without tensor memory, so it keeps tcgen05's free e4m3 conversion
    // while escaping the 1-CTA/SM pin. N=8 covers the whole serving band.
    // Opt-in during bring-up: PADDOCK_F8T_WMMA=1.
    {
        // DEFAULT-ON since the PPL gate, which measured this route
        // numerically free against tcgen05 (ppl 5.63039 vs 5.63580 -- very
        // slightly better, i.e. pure accumulation-order noise). Only reachable
        // on sm_100 anyway: the tile plane and this launcher are both behind
        // PD_TC5_HOST. Kill: PADDOCK_NO_F8T_WMMA.
        static int wm_on = -1;
        if (wm_on < 0) wm_on = pd_env("PADDOCK_NO_F8T_WMMA") ? 0 : 1;
        // Shape gate, profiled on qwen3.6-27b c1: this route beats tc5q on the big
        // fused plane (gate_up, 272 tiles: dram 40.7% vs tc5q's 37.5%) but
        // starves on the small ones (out 5120, 40 tiles: dram 21.7%, 0.36
        // waves/SM) because ctas = out/64 and the K-split cannot grow past
        // the 8 planes `part` is sized for. Default 256 tiles = exactly the
        // tc5q election, so this replaces tc5q and leaves tc5p alone.
        // BATCH-DEPENDENT. The gate was tuned at b=1, where 256
        // (gate_up only) wins and opening it to every shape costs ~1.6%. At
        // b=4 the election inverts: 40 (every shape) is +1.7% and 256 leaves
        // 9.6% on the table, because the mma's N=8 tile is half-idle at b=1
        // on the narrow planes but earns its keep once the batch fills it.
        //   c1: 256 -> 122.66, 40 -> 120.66
        //   c4:  40 -> 405.11, 112 -> 393.93, 256 -> 398.17
        // PADDOCK_F8T_WMMA_MINTILES overrides both.
        static int wm_mint_env = -2;
        if (wm_mint_env == -2) {
            const char* e = pd_env("PADDOCK_F8T_WMMA_MINTILES");
            wm_mint_env = e ? atoi(e) : -1;
        }
        const uint32_t wm_mint =
            wm_mint_env >= 0 ? (uint32_t)wm_mint_env : (batch >= 4u ? 40u : 256u);
        // Batch ceiling. NT=1..4 can carry batch to 32, but wider is not
        // automatically better: each n-tile adds 2304 B of smem, and smem is
        // what sets occupancy here (19712 B -> 11 CTAs/SM at NT=1, 26624 B -> 8
        // at NT=4). Past some batch the CTA count this route wins on is spent,
        // and tc5q's tcgen05 mma -- far more efficient per issue, even pinned at
        // 1 CTA/SM by its TMEM claim -- takes back the lead.
        // MEASURED (qwen3.6-27b Q8_0, B200, 32 slots, aggregate rate):
        //         c17     c20     c24     c32
        //   16   1134.5  1301.8  1502.8  1853 (8-round median)
        //   24   1164.3  1328.3  1512.8  1863
        //   32   1162.0  1330.6  1505.7  1740
        // 24 wins: +2.6% / +2.0% / +0.7% over the old 16 ceiling in the band it
        // opens, and level at c32. Letting batch 25..32 onto this route COSTS
        // 4% -- NT=4's 26624 B/CTA drops occupancy to 8 CTAs/SM, and past ~24
        // tc5q's tcgen05 mma wins even pinned at 1 CTA/SM by its TMEM claim.
        // NT=4 is kept compiled and reachable (PADDOCK_F8T_WMMA_BMAX=32) as the
        // measured counterexample, the same way MODE 2 is.
        static int wm_bmax = -2;
        if (wm_bmax == -2) {
            const char* e = pd_env("PADDOCK_F8T_WMMA_BMAX");
            wm_bmax = e ? atoi(e) : -1;
            if (wm_bmax > 32) wm_bmax = 32;
        }
        const uint32_t wm_bm = wm_bmax >= 0 ? (uint32_t)wm_bmax : 24u;
        if (wm_on && batch <= wm_bm && (out_dim >> 7) >= wm_mint) {
            constexpr uint32_t WARPS = 4u;         // 64 output rows per CTA
            static int nsm_w = 0;
            if (nsm_w == 0) {
                int d = 0;
                cudaGetDevice(&d);
                cudaDeviceGetAttribute(&nsm_w, cudaDevAttrMultiProcessorCount, d);
                if (nsm_w <= 0) nsm_w = 148;
            }
            const uint32_t ctas = (out_dim + WARPS * 16u - 1u) / (WARPS * 16u);
            const uint32_t nkt = in_dim >> 7;
            static int wnz = -2;
            if (wnz == -2) {
                const char* e = pd_env("PADDOCK_F8T_WMMA_NZ");
                wnz = e ? atoi(e) : 0;
            }
            // n-tiles, needed here because occupancy (smem-limited) depends on it
            uint32_t ntn = (batch + 7u) >> 3;                      // 1..4
            {
                static int nt_pin = -2;
                if (nt_pin == -2) {
                    const char* e = pd_env("PADDOCK_F8T_WMMA_NT");
                    nt_pin = e ? atoi(e) : 0;
                }
                if (nt_pin > (int)ntn && nt_pin <= 4) ntn = (uint32_t)nt_pin;
            }
            // (measured CTAs/SM by ntn: 1->11, 2->10, 3->9, 4->8 - kept as a
            // note; the nz policy below sizes off ctas/nsm_w, not occupancy)
            uint32_t nz = 1u;
            if (wnz > 0) {
                nz = (uint32_t)wnz;
            } else if (part && ctas < (uint32_t)nsm_w * 8u) {
                // Deliberately not the cost model used for tc5p/tc5q below.
                // Tried and reverted: the same waves/penalty model
                // here picks nz=8 at b=1 on the gate_up shape where this tuned
                // heuristic picks 3, and measurement says 3 is right -- c1 fell
                // 102.6 -> 101.4 (-1.2%) for +0.3% at c32. The model's wave term
                // assumes the full smem-theoretical occupancy and ignores the
                // per-CTA prologue (the x buffers are zeroed once per CTA), both
                // of which flatter large z. c1 is a cell we win; do not trade it
                // for a third of a percent at c32.
                //
                // The batch-blind split is still wrong in principle here, just
                // not by enough to matter: the wmma route only runs to batch 24,
                // where the partial planes are a quarter the size they reach on
                // the tc5p path. Revisit with a wave term fitted to measured
                // occupancy rather than the smem bound.
                nz = ((uint32_t)nsm_w * 8u + ctas - 1u) / ctas;
            }
            if (nz > 8u) nz = 8u;            // `part` is out_dim*batch*8 floats
            if (nz > nkt) nz = nkt;
            if (nz > 1u && !part) nz = 1u;
            float* dst = nz > 1u ? (float*)part : (float*)y;
            // Pipelining mode, all three in one binary so the A/B is clean:
            // 0 = single buffer, 1 = cp.async two-stage, 2 = register-prefetch
            // two-stage on plain loads. PADDOCK_F8T_WMMA_DB selects.
            static int wm_db = -1;
            if (wm_db < 0) {
                const char* e = pd_env("PADDOCK_F8T_WMMA_DB");
                wm_db = e ? atoi(e) : 1;
                if (wm_db < 0 || wm_db > 2) wm_db = 1;
            }
            #define PD_WMMA_LAUNCH(M, NT_)                                         \
                pd_f8row_gemm_wmma_kt<WARPS, M, NT_>                               \
                    <<<dim3(ctas, nz), WARPS * 32u, 0, st>>>(                      \
                    (const unsigned char*)wtiles, (const float*)wrs,               \
                    (const unsigned char*)xq, (const float*)xrs, dst,              \
                    in_dim, out_dim, batch, nz)
            #define PD_WMMA_MODES(NT_)                                             \
                do {                                                               \
                    if (wm_db == 0)      PD_WMMA_LAUNCH(0, NT_);                   \
                    else if (wm_db == 1) PD_WMMA_LAUNCH(1, NT_);                   \
                    else                 PD_WMMA_LAUNCH(2, NT_);                   \
                } while (0)
            // PADDOCK_F8T_WMMA_NT=N forces N n-tiles where fewer would do. That
            // is the correctness gate: any tile past ceil(batch/8) reads x
            // columns zeroed at entry and has every epilogue store skipped by
            // `c < batch`, so a forced-wider NT must be BIT-IDENTICAL to the
            // elected one -- which makes the b=1 ppl harness a direct proof of
            // each new instantiation. Forcing NARROWER is ignored deliberately:
            // it would silently drop the columns above NT*8.
            // ntn was elected above the K-split (occupancy depends on it)
            if (ntn <= 1u)      PD_WMMA_MODES(1u);
            else if (ntn == 2u) PD_WMMA_MODES(2u);
            else if (ntn == 3u) PD_WMMA_MODES(3u);
            else                PD_WMMA_MODES(4u);
            #undef PD_WMMA_MODES
            #undef PD_WMMA_LAUNCH
            if (nz > 1u && !no_combine) {
                const uint32_t n = out_dim * batch;
                pd_q8_0_gemm_mma_ks_combine_kernel<<<(n + 255u) / 256u, 256, 0, st>>>(
                    (const float*)part, nullptr, (float*)y, n, nz, out_dim);
            }
            if (out_nz) *out_nz = (nz > 1u && no_combine) ? nz : 1u;
            return cudaPeekAtLastError() == cudaSuccess ? 0 : -2;
        }
    }
    // Wide-grid GEMV route for the serving decode band (see
    // pd_f8row_gemv_wide_kt): trades the tcgen05 mma, which earns nothing at
    // ~2 flops/byte, for the CTA count the die needs to saturate HBM.
    // Opt-in during bring-up: PADDOCK_F8T_GEMV=1.
    {
        static int gemv_on = -1;
        if (gemv_on < 0) gemv_on = pd_env("PADDOCK_F8T_GEMV") ? 1 : 0;
        static int gemv_bmax = -1;
        if (gemv_bmax < 0) {
            const char* e = pd_env("PADDOCK_F8T_GEMV_BMAX");
            gemv_bmax = e ? atoi(e) : 4;
            if (gemv_bmax < 1 || gemv_bmax > 4) gemv_bmax = 4;
        }
        if (gemv_on && batch <= (uint32_t)gemv_bmax) {
            constexpr uint32_t WARPS = 8u;
            static int nsm_g = 0;
            if (nsm_g == 0) {
                int d = 0;
                cudaGetDevice(&d);
                cudaDeviceGetAttribute(&nsm_g, cudaDevAttrMultiProcessorCount, d);
                if (nsm_g <= 0) nsm_g = 148;
            }
            const uint32_t ctas = (out_dim + WARPS - 1u) / WARPS;
            const uint32_t nkt = in_dim >> 7;
            // K-split only when one z-plane cannot put 8 CTAs on every SM.
            // Capped at 4: `part` is documented as out_dim*batch*8 floats, so
            // 4 keeps a 2x margin on the caller's scratch.
            static int nzf = -2;
            if (nzf == -2) {
                const char* e = pd_env("PADDOCK_F8T_GEMV_NZ");
                nzf = e ? atoi(e) : 0;
            }
            uint32_t nz = 1u;
            if (nzf > 0) nz = (uint32_t)nzf;
            else if (part && ctas < (uint32_t)nsm_g * 8u)
                nz = ((uint32_t)nsm_g * 8u + ctas - 1u) / ctas;
            if (nz > 4u) nz = 4u;
            if (nz > nkt) nz = nkt;
            if (nz > 1u && !part) nz = 1u;
            float* dst = nz > 1u ? (float*)part : (float*)y;
            pd_f8row_gemv_wide_kt<WARPS, 8u><<<dim3(ctas, nz), WARPS * 32u, 0, st>>>(
                (const unsigned char*)wtiles, (const float*)wrs,
                (const unsigned char*)xq, (const float*)xrs, dst,
                in_dim, out_dim, batch, nz);
            if (nz > 1u && !no_combine) {
                const uint32_t n = out_dim * batch;
                pd_q8_0_gemm_mma_ks_combine_kernel<<<(n + 255u) / 256u, 256, 0, st>>>(
                    (const float*)part, nullptr, (float*)y, n, nz, out_dim);
            }
            if (out_nz) *out_nz = (nz > 1u && no_combine) ? nz : 1u;
            return cudaPeekAtLastError() == cudaSuccess ? 0 : -2;
        }
    }
    // tc5t persistent route for the 65..128-row band (P46): the
    // c32-spec verify tick (~96 rows) plus narrow prefill chunks leave
    // tc5r's cluster grids half-starved (42-84 CTAs on the o/down/qkv
    // planes at ~30 GB/s/CTA on a 148-SM die); the item loop fills it.
    // Opt-in while the P46 probe decides: PADDOCK_TC5T=1. Declines fall
    // through to tc5r unchanged.
    if (batch >= 65u && batch <= 128u && !(out_dim & 255u)) {
        static int t_on = -1;
        if (t_on < 0) {
            const char* e = pd_env("PADDOCK_TC5T");
            t_on = e && atoi(e) != 0 ? 1 : 0;
        }
        const uint32_t nk_t = (in_dim + 127u) / 128u;
        if (t_on && nk_t >= 8u && pd_tc5q_ctr()) {
            uint32_t* qctr = pd_tc5q_ctr();
            static int nsm_t = 0;
            if (nsm_t == 0) {
                int d = 0;
                cudaGetDevice(&d);
                cudaDeviceGetAttribute(&nsm_t, cudaDevAttrMultiProcessorCount, d);
                if (nsm_t <= 0) nsm_t = 148;
            }
            // 128-row-box ymap, own cache (the tc5r block stays independent)
            struct YMapT { const void* ptr; uint32_t in; CUtensorMap m; };
            static YMapT tcache[16];
            static uint32_t tn = 0;
            CUtensorMap* ymt = nullptr;
            for (uint32_t i = 0; i < tn; ++i)
                if (tcache[i].ptr == xq && tcache[i].in == in_dim) {
                    ymt = &tcache[i].m;
                    break;
                }
            if (!ymt) {
                if (tn >= 16u) tn = 0;
                if (pd_tmap_2d(&tcache[tn].m, xq, in_dim, 1u << 22)) {
                    tcache[tn].ptr = xq;
                    tcache[tn].in = in_dim;
                    ymt = &tcache[tn++].m;
                }
            }
            if (ymt) {
                // item K-split fills the die: want = ceil(nsm/ntiles), cap 8,
                // >= 4 k-tiles per z, clamped by the same partial-scratch
                // budget the loader publishes for tc5r (the partials share
                // `part`). P46 ledger elections at m<=128: qkv nz3, o/down
                // nz8, gu/lmh nz1 (>=148 items already).
                const uint32_t ntt = out_dim >> 8;
                static long tbud = -1;
                if (tbud < 0) {
                    const char* e = pd_env("PADDOCK_TC5R_NZ_BUDGET");
                    tbud = e ? atol(e) : 0;
                    if (tbud < 0) tbud = 0;
                }
                static int tnzf = -2;
                if (tnzf == -2) {
                    const char* e = pd_env("PADDOCK_TC5T_NZ");
                    tnzf = e ? atoi(e) : 0;
                }
                uint32_t nzt = 1u;
                if (part && ntt < (uint32_t)nsm_t) {
                    uint32_t want = tnzf > 0 ? (uint32_t)tnzf
                        : ((uint32_t)nsm_t + ntt - 1u) / ntt;
                    if (want > 8u) want = 8u;
                    while (want > 1u && want * 4u > nk_t) --want;
                    while (want > 1u &&
                           (size_t)out_dim * batch * want > (size_t)tbud)
                        --want;
                    if (want > 1u) {             // no empty z-planes
                        const uint32_t nkz_t = (nk_t + want - 1u) / want;
                        nzt = (nk_t + nkz_t - 1u) / nkz_t;
                    }
                }
                constexpr uint32_t TS_ = 4u;
                const uint32_t tsmem = TS_ * 49152u + (2u * TS_ + 2u) * 8u;
                static int no_eft = -1;
                if (no_eft < 0) no_eft = pd_env("PADDOCK_NO_L2EF") ? 1 : 0;
                static bool tattr = false;
                if (!tattr) {
                    cudaFuncSetAttribute((const void*)pd_f8row_gemm_tc5t_kt<TS_, true>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)tsmem);
                    cudaFuncSetAttribute((const void*)pd_f8row_gemm_tc5t_kt<TS_, false>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)tsmem);
                    tattr = true;
                }
                float* dst = nzt > 1u ? (float*)part : (float*)y;
                if (!no_eft)
                    pd_f8row_gemm_tc5t_kt<TS_, true><<<(uint32_t)nsm_t, 128, tsmem, st>>>(
                        (const unsigned char*)wtiles, *ymt, (const float*)wrs,
                        (const float*)xrs, dst, in_dim, out_dim, batch, nzt, 3u, qctr);
                else
                    pd_f8row_gemm_tc5t_kt<TS_, false><<<(uint32_t)nsm_t, 128, tsmem, st>>>(
                        (const unsigned char*)wtiles, *ymt, (const float*)wrs,
                        (const float*)xrs, dst, in_dim, out_dim, batch, nzt, 3u, qctr);
                if (nzt > 1u && !no_combine) {
                    const uint32_t n = out_dim * batch;
                    pd_q8_0_gemm_mma_ks_combine_kernel<<<(n + 255u) / 256u, 256, 0, st>>>(
                        (const float*)part, nullptr, (float*)y, n, nzt, out_dim);
                }
                if (out_nz) *out_nz = (nzt > 1u && no_combine) ? nzt : 1u;
                return pd_launch_status();
            }
        }
    }
    // tc5r 2-SM rowwise route (the whole 65+ band - class uniformity with
    // the <=64 tc5p/tc5q rungs; partial clusters at 65..255 pad to 256).
    // Kill PADDOCK_NO_TC5R falls back to the caller's f8w arm via the error.
    if (batch >= 65u) {
        static int no_r = -1;
        if (no_r < 0) no_r = pd_env("PADDOCK_NO_TC5R") ? 1 : 0;
        const uint32_t tiles_r = out_dim >> 7;
        if (no_r || (tiles_r & 1u)) return cudaErrorInvalidValue;
        // 128-row-box ymap (separate cache from the h64 decode maps); rows
        // bound is loose - the accessed rows stay inside the caller's
        // >=batch_pad activation buffer
        struct YMapR { const void* ptr; uint32_t in; CUtensorMap m; };
        static YMapR rcache[16];
        static uint32_t rn = 0;
        CUtensorMap* ymr = nullptr;
        for (uint32_t i = 0; i < rn; ++i)
            if (rcache[i].ptr == xq && rcache[i].in == in_dim) { ymr = &rcache[i].m; break; }
        if (!ymr) {
            if (rn >= 16u) rn = 0;
            if (!pd_tmap_2d(&rcache[rn].m, xq, in_dim, 1u << 22))
                return cudaErrorInvalidValue;
            rcache[rn].ptr = xq;
            rcache[rn].in = in_dim;
            ymr = &rcache[rn++].m;
        }
// tc5r kernel definition is compiled out below sm_90 (its __cluster_dims__
// guard); drop these references from those device passes too - the host pass
// keeps them, and this branch is only reached on cluster-capable devices
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 900)
        // Ring depth election. Profiling the verify band
        // (96 launches, locked clocks): schedulers ~90% idle (issue 8-12%),
        // barrier-stall dominant, and the 32/64/128-CTA planes all run ~70us
        // - duration tracks k-chain length, not CTA count or the mma stream,
        // so the band is latency-bound on the W/B ring, not issue-bound
        // (falsifies the NC=128 padding theory as the primary lever). S=6 x
        // 32KB/stage is the smem ceiling story, not an elected depth: S=7
        // (229544B) is the deepest NC=256 ring under sm_100's 232448B opt-in
        // cap. PADDOCK_TC5R_S=7 selects it for A/B; default 6 (ship parity)
        // until the battery decides. Same k order either way - bit-identical.
        constexpr uint32_t RNT = 1u;
        static int rsdep = 0;
        if (rsdep == 0) {
            const char* e = pd_env("PADDOCK_TC5R_S");
            rsdep = e && atoi(e) == 7 ? 7 : 6;
        }
        const uint32_t RS = (uint32_t)rsdep;
        const uint32_t rsmem = (1u + RNT) * RS * 16384u + 3u * RS * 8u;
        // Narrow-N election. The S=7 A/B read +1.6% (tie-class) -
        // depth is not the main term. Per-active-SM SM throughput ~62%
        // says the tensor pipes are the term: at batch <= 128 the N=256 mma
        // spends half its pipe cycles on padding cols (cutlass elects
        // narrow-N tiles - 128x64x256, 128x32x128 - on these same shapes).
        // The NW arm runs N=128: half the mma time per k-tile, same k order,
        // bit-identical real cols. PADDOCK_TC5R_N128=1 selects it for A/B.
        static int rn128 = -1;
        if (rn128 < 0) {
            const char* e = pd_env("PADDOCK_TC5R_N128");
            rn128 = e ? atoi(e) : 0;
        }
        const uint32_t nw = (rn128 > 0 && batch <= 128u) ? 1u : 0u;
        const uint32_t ncl = nw ? 128u : 256u;
        // KT=2 election: after n128 the per-STAGE sync round-trip
        // (bfull turnaround + bpeer hop + dual commit, 0.77-1.84us/tile as
        // shipped vs ~0.27us of narrow mma) dominates every plane -
        // gu barely moved when the mma halved. KT=2 at S=3 keeps the same 6
        // k-tiles in flight and halves the round-trips. PADDOCK_TC5R_K2=1
        // selects it for A/B (cutlass runs K=256 mainloops on these shapes).
        static int rk2 = -1;
        if (rk2 < 0) {
            const char* e = pd_env("PADDOCK_TC5R_K2");
            rk2 = e ? atoi(e) : 0;
        }
        const uint32_t k2 = rk2 > 0 ? 1u : 0u;
        // b64: under NW the 128-col B box streams 2x the bytes
        // the narrow mma reads. A pd_tmap_2d_h64 64-col box fetches exactly
        // this rank's real cols (8KB/k-tile) - halves B traffic AND frees
        // smem for S=9 (9 k-tiles in flight vs 6). PADDOCK_TC5R_B64=1 + NW.
        static int rb64 = -1;
        if (rb64 < 0) {
            const char* e = pd_env("PADDOCK_TC5R_B64");
            rb64 = e ? atoi(e) : 0;
        }
        const uint32_t b64 = (rb64 > 0 && nw) ? 1u : 0u;
        // O16 (f8t16): bf16 y on the prefill-chunk planes whose
        // out_dim the loader publishes (o/down = n_embd; their consumer's
        // p16 twin ships). Chunk widths only (batch >= 129 -> classic arm,
        // nz never fires). Absent env = arm dead, bit-identical build.
        static long ro16d = -1;
        if (ro16d < 0) {
            const char* e = pd_env("PADDOCK_TC5R_O16_DIM");
            ro16d = e ? atol(e) : 0;
            if (ro16d < 0) ro16d = 0;
        }
        // O16_IN narrows the election to one plane (wo: in = n_head*hd) so
        // the down plane - whose pipelined consumers cross pass iterations -
        // stays f32 until its own wiring lands.
        static long ro16i = -1;
        if (ro16i < 0) {
            const char* e = pd_env("PADDOCK_TC5R_O16_IN");
            ro16i = e ? atol(e) : 0;
            if (ro16i < 0) ro16i = 0;
        }
        const uint32_t o16 = (ro16d > 0 && (uint32_t)ro16d == out_dim
                              && (ro16i == 0 || (uint32_t)ro16i == in_dim)
                              && batch >= 129u) ? 1u : 0u;
        CUtensorMap* ym64 = nullptr;
        if (b64) {
            static YMapR r64cache[16];
            static uint32_t r64n = 0;
            for (uint32_t i = 0; i < r64n; ++i)
                if (r64cache[i].ptr == xq && r64cache[i].in == in_dim) {
                    ym64 = &r64cache[i].m;
                    break;
                }
            if (!ym64) {
                if (r64n >= 16u) r64n = 0;
                if (!pd_tmap_2d_h64(&r64cache[r64n].m, xq, in_dim, 1u << 22))
                    return cudaErrorInvalidValue;
                r64cache[r64n].ptr = xq;
                r64cache[r64n].in = in_dim;
                ym64 = &r64cache[r64n++].m;
            }
        }
        static bool rattr = false;
        if (!rattr) {
            const int sm6 = (int)((1u + RNT) * 6u * 16384u + 3u * 6u * 8u);
            const int sm7 = (int)((1u + RNT) * 7u * 16384u + 3u * 7u * 8u);
            cudaFuncSetAttribute((const void*)pd_f8t_gemm_tc5r_kt<6u, RNT>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, sm6);
            cudaFuncSetAttribute((const void*)pd_f8t_gemm_tc5r_kt<6u, RNT>,
                                 cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
            cudaFuncSetAttribute((const void*)pd_f8t_gemm_tc5r_kt<7u, RNT>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, sm7);
            cudaFuncSetAttribute((const void*)pd_f8t_gemm_tc5r_kt<7u, RNT>,
                                 cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
            cudaFuncSetAttribute((const void*)pd_f8t_gemm_tc5r_kt<6u, RNT, 1u>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, sm6);
            cudaFuncSetAttribute((const void*)pd_f8t_gemm_tc5r_kt<6u, RNT, 1u>,
                                 cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
            cudaFuncSetAttribute((const void*)pd_f8t_gemm_tc5r_kt<7u, RNT, 1u>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, sm7);
            cudaFuncSetAttribute((const void*)pd_f8t_gemm_tc5r_kt<7u, RNT, 1u>,
                                 cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
            const int smk2 = (int)((1u + RNT) * 3u * 2u * 16384u + 3u * 3u * 8u);
            cudaFuncSetAttribute((const void*)pd_f8t_gemm_tc5r_kt<3u, RNT, 0u, 2u>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, smk2);
            cudaFuncSetAttribute((const void*)pd_f8t_gemm_tc5r_kt<3u, RNT, 0u, 2u>,
                                 cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
            cudaFuncSetAttribute((const void*)pd_f8t_gemm_tc5r_kt<3u, RNT, 1u, 2u>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, smk2);
            cudaFuncSetAttribute((const void*)pd_f8t_gemm_tc5r_kt<3u, RNT, 1u, 2u>,
                                 cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
            const int smb64 = (int)(9u * (16384u + RNT * 8192u) + 3u * 9u * 8u);
            cudaFuncSetAttribute((const void*)pd_f8t_gemm_tc5r_kt<9u, RNT, 1u, 1u, 8192u>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, smb64);
            cudaFuncSetAttribute((const void*)pd_f8t_gemm_tc5r_kt<9u, RNT, 1u, 1u, 8192u>,
                                 cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
            rattr = true;
        }
        const uint32_t bpad = (batch + ncl - 1u) & ~(ncl - 1u);
        const uint32_t grid = (tiles_r / 2u) * (bpad / ncl) * 2u;
        // K-split election. Verify-width launches (one 256-col
        // block) leave 42-128 CTAs on the die at 1 CTA/SM: the o/down planes
        // measured 0.56 TB/s where a cutlass kernel does 1.5 on the same
        // shapes and die. grid.y z-planes stream
        // disjoint k-ranges into `part`; fixed-order combine finishes (same
        // numeric class as the tc5p/gemv nz routes). Off unless the loader
        // publishes its partial-scratch capacity via PADDOCK_TC5R_NZ_BUDGET
        // (floats) - scratch is per-model (qwen35's ks_part is a batch<=64
        // contract; gemma4's pf_skfix holds 12M). PADDOCK_TC5R_NZ forces
        // (1 = off) for A/B.
        static long rbud = -1;
        if (rbud < 0) {
            const char* e = pd_env("PADDOCK_TC5R_NZ_BUDGET");
            rbud = e ? atol(e) : 0;
            if (rbud < 0) rbud = 0;
        }
        static int rnzf = -2;
        if (rnzf == -2) {
            const char* e = pd_env("PADDOCK_TC5R_NZ");
            rnzf = e ? atoi(e) : 0;
        }
        static int nsm_r = 0;
        if (nsm_r == 0) {
            int d = 0;
            cudaGetDevice(&d);
            cudaDeviceGetAttribute(&nsm_r, cudaDevAttrMultiProcessorCount, d);
            if (nsm_r <= 0) nsm_r = 148;
        }
        uint32_t nzr = 1u;
        if (part && rbud > 0) {
            const uint32_t nkt_r = (in_dim + 127u) / 128u;
            // ~1.5 waves at 1 CTA/SM; each z keeps >= 4 k-tiles so the
            // S-deep ring still amortizes its prologue
            uint32_t want = 1u;
            if (rnzf > 0) want = (uint32_t)rnzf;
            else if (grid <= 48u && batch <= 128u)
                want = 2u;   // elected set + depth are both MEASURED, not
                             // derived (c32-spec A/B, 3 legs
                             // each): nz2 on grid<=48 (o/down 42 + 4096-out
                             // 32) at batch<=128 +3.7%; adding qkv 84 + mid
                             // 128 planes -3.6%; wave-filling nz6 -7.3%;
                             // without the batch gate (129-256 splits too)
                             // the win halves to +1.5%. Deeper splits, near-
                             // full grids and wide-batch partials lose to
                             // z-ramp + combine cost despite better fill.
            if (want > 8u) want = 8u;
            if (want > 1u && want * 4u > nkt_r)
                want = nkt_r / 4u ? nkt_r / 4u : 1u;
            while (want > 1u && (size_t)out_dim * batch * want > (size_t)rbud)
                --want;
            if (want > 1u) {                 // no empty z-planes: re-derive
                const uint32_t nkz_r = (nkt_r + want - 1u) / want;
                nzr = (nkt_r + nkz_r - 1u) / nkz_r;
            }
        }
        float* const rdst = nzr > 1u ? (float*)part : (float*)y;
        const uint32_t rsmem_k2 = (1u + RNT) * 3u * 2u * 16384u + 3u * 3u * 8u;
        const uint32_t rsmem_b64 = 9u * (16384u + RNT * 8192u) + 3u * 9u * 8u;
        if (o16 && nzr == 1u) {
            static bool o16attr = false;
            if (!o16attr) {
                cudaFuncSetAttribute(
                    (const void*)pd_f8t_gemm_tc5r_kt<6u, RNT, 0u, 1u, 16384u, 1u>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    (int)((1u + RNT) * 6u * 16384u + 3u * 6u * 8u));
                cudaFuncSetAttribute(
                    (const void*)pd_f8t_gemm_tc5r_kt<6u, RNT, 0u, 1u, 16384u, 1u>,
                    cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
                o16attr = true;
            }
            pd_f8t_gemm_tc5r_kt<6u, RNT, 0u, 1u, 16384u, 1u>
                <<<dim3(grid, 1u), 128, (1u + RNT) * 6u * 16384u + 3u * 6u * 8u, st>>>(
                (const unsigned char*)wtiles, *ymr, (const float*)wrs,
                (const float*)xrs, (float*)y, in_dim, out_dim, batch);
            if (out_nz) *out_nz = 1u;
            return pd_launch_status();
        }
        if (b64)
            pd_f8t_gemm_tc5r_kt<9u, RNT, 1u, 1u, 8192u>
                <<<dim3(grid, nzr), 128, rsmem_b64, st>>>(
                (const unsigned char*)wtiles, *ym64, (const float*)wrs,
                (const float*)xrs, rdst, in_dim, out_dim, batch);
        else if (k2 && nw)
            pd_f8t_gemm_tc5r_kt<3u, RNT, 1u, 2u><<<dim3(grid, nzr), 128, rsmem_k2, st>>>(
                (const unsigned char*)wtiles, *ymr, (const float*)wrs,
                (const float*)xrs, rdst, in_dim, out_dim, batch);
        else if (k2)
            pd_f8t_gemm_tc5r_kt<3u, RNT, 0u, 2u><<<dim3(grid, nzr), 128, rsmem_k2, st>>>(
                (const unsigned char*)wtiles, *ymr, (const float*)wrs,
                (const float*)xrs, rdst, in_dim, out_dim, batch);
        else if (nw && RS == 7u)
            pd_f8t_gemm_tc5r_kt<7u, RNT, 1u><<<dim3(grid, nzr), 128, rsmem, st>>>(
                (const unsigned char*)wtiles, *ymr, (const float*)wrs,
                (const float*)xrs, rdst, in_dim, out_dim, batch);
        else if (nw)
            pd_f8t_gemm_tc5r_kt<6u, RNT, 1u><<<dim3(grid, nzr), 128, rsmem, st>>>(
                (const unsigned char*)wtiles, *ymr, (const float*)wrs,
                (const float*)xrs, rdst, in_dim, out_dim, batch);
        else if (RS == 7u)
            pd_f8t_gemm_tc5r_kt<7u, RNT><<<dim3(grid, nzr), 128, rsmem, st>>>(
                (const unsigned char*)wtiles, *ymr, (const float*)wrs,
                (const float*)xrs, rdst, in_dim, out_dim, batch);
        else
            pd_f8t_gemm_tc5r_kt<6u, RNT><<<dim3(grid, nzr), 128, rsmem, st>>>(
                (const unsigned char*)wtiles, *ymr, (const float*)wrs,
                (const float*)xrs, rdst, in_dim, out_dim, batch);
        if (nzr > 1u) {
            const uint32_t n = out_dim * batch;
            pd_q8_0_gemm_mma_ks_combine_kernel<<<(n + 255u) / 256u, 256, 0, st>>>(
                (const float*)part, nullptr, (float*)y, n, nzr, out_dim);
        }
        if (out_nz) *out_nz = 1u;
        return pd_launch_status();
#else
        return cudaErrorNotSupported;
#endif
    }
    if (batch > 64u) return cudaErrorInvalidValue;
    static int nsm = 0;
    if (nsm == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
        if (nsm <= 0) nsm = 128;
    }
    // Y tensor maps keyed on (buffer, in_dim): the decode scratch is a fixed
    // address and in_dim takes 2-3 values per model, so a tiny append-only
    // cache kills the ~2 us/encode that would otherwise recur per launch.
    // Maps use 64-row boxes: the activation buffer must be >= 64 rows.
    struct YMapEnt { const void* ptr; uint32_t in; CUtensorMap m; };
    static YMapEnt cache[16];
    static uint32_t ncache = 0;
    CUtensorMap* ym = nullptr;
    for (uint32_t i = 0; i < ncache; ++i)
        if (cache[i].ptr == xq && cache[i].in == in_dim) { ym = &cache[i].m; break; }
    if (!ym) {
        if (ncache >= 16u) ncache = 0;   // wraparound: engine uses few shapes
        if (!pd_tmap_2d_h64(&cache[ncache].m, xq, in_dim, 64u))
            return cudaErrorInvalidValue;
        cache[ncache].ptr = xq;
        cache[ncache].in = in_dim;
        ym = &cache[ncache++].m;
    }
    const uint32_t tiles = out_dim >> 7, nk_all = in_dim >> 7;
    // tc5m election: PADDOCK_TC5M=1 routes every 64-aligned plane
    // at batch<=64 to the M64-tile arm (see kernel header) for A/B vs the
    // ship tc5p/tc5q routes. PADDOCK_TC5M_S picks the ring: 4 (3 CTAs/SM,
    // default) or 6 (2/SM). Output is final: no partials, no combine.
    {
        static int m64 = -1;
        if (m64 < 0) {
            const char* e = pd_env("PADDOCK_TC5M");
            m64 = e ? atoi(e) : 0;
        }
        if (m64 > 0 && !(out_dim & 63u) && nk_all >= 4u) {
            struct YM64 { const void* ptr; uint32_t in; CUtensorMap m; };
            static YM64 m64c[16];
            static uint32_t m64n = 0;
            CUtensorMap* ym64 = nullptr;
            for (uint32_t i = 0; i < m64n; ++i)
                if (m64c[i].ptr == xq && m64c[i].in == in_dim) {
                    ym64 = &m64c[i].m;
                    break;
                }
            if (!ym64) {
                if (m64n >= 16u) m64n = 0;
                if (!pd_tmap_2d_h64(&m64c[m64n].m, xq, in_dim, 1u << 22))
                    return cudaErrorInvalidValue;
                m64c[m64n].ptr = xq;
                m64c[m64n].in = in_dim;
                ym64 = &m64c[m64n++].m;
            }
            // S=13 = the cutlass recipe exactly
            // (MainloopSm100TmaUmmaWarpSpecialized<13,2,4>, 64x64x128
            // tiles, 1 CTA/SM at 208KB) - tc5m shares the tile geometry and
            // its S=4->6 probe trend was strongly positive (-23.5 -> -17.6).
            static int ms = -2;
            if (ms == -2) {
                const char* e = pd_env("PADDOCK_TC5M_S");
                const int v = e ? atoi(e) : 4;
                ms = (v == 6 || v == 10 || v == 13) ? v : 4;
            }
            static bool mattr = false;
            if (!mattr) {
                #define PD_M_ATTR(SS)                                            \
                    cudaFuncSetAttribute((const void*)pd_f8row_gemm_tc5m_kt<SS>, \
                        cudaFuncAttributeMaxDynamicSharedMemorySize,             \
                        (int)(SS * 16384u + 2u * SS * 8u + 8448u))
                PD_M_ATTR(4u); PD_M_ATTR(6u); PD_M_ATTR(10u); PD_M_ATTR(13u);
                #undef PD_M_ATTR
                mattr = true;
            }
            const uint32_t mgrid = out_dim >> 6;
            #define PD_M_GO(SS)                                                  \
                pd_f8row_gemm_tc5m_kt<SS><<<mgrid, 128,                          \
                    SS * 16384u + 2u * SS * 8u + 8448u, st>>>(                   \
                    (const unsigned char*)wtiles, *ym64, (const float*)wrs,      \
                    (const float*)xrs, (float*)y, in_dim, out_dim, batch)
            if (ms == 13)      PD_M_GO(13u);
            else if (ms == 10) PD_M_GO(10u);
            else if (ms == 6)  PD_M_GO(6u);
            else               PD_M_GO(4u);
            #undef PD_M_GO
            if (out_nz) *out_nz = 1u;
            return pd_launch_status();
        }
    }
    // persistent route for big fused planes (gu = 336 tiles): item streams of
    // ~2+ tiles per CTA reach the long-stream regime (fused-shape harness:
    // 3.1/2.9/2.7 TB/s at r=8/32/64 vs tc5p's 2.7/2.3/2.4 on the split
    // planes). Kill: PADDOCK_NO_TC5Q.
    // PADDOCK_TC5Q_MINTILES overrides the 256-tile admission (default keeps
    // the shipped behavior). Why it exists: the threshold encodes "~2 items
    // per persistent CTA", but that is a function of nsm and nzq, not a
    // constant - qwen3.5-9b's gu plane is 192 tiles and missed the route by
    // threshold alone, riding tc5p_m2 at 3.3 TB/s where tc5q clocks 4.37 on
    // the same die (b200-batch-gate doc; 9B c32 A/B below).
    static int no_q = -1;
    if (no_q < 0) no_q = pd_env("PADDOCK_NO_TC5Q") ? 1 : 0;
    static int q_min = -2;
    if (q_min == -2) {
        const char* e = pd_env("PADDOCK_TC5Q_MINTILES");
        q_min = e ? atoi(e) : 0;
        if (q_min <= 0) q_min = 256;
    }
    if (!no_q && tiles >= (uint32_t)q_min && nk_all >= 10u && pd_tc5q_ctr()) {
        uint32_t* qctr = pd_tc5q_ctr();
        constexpr uint32_t QS = 6u, QD = 5u;
        const uint32_t qsmem = QS * 24576u + 2u * QS * 8u + 16u;
        static int no_efq = -1;
        if (no_efq < 0) no_efq = pd_env("PADDOCK_NO_L2EF") ? 1 : 0;
        static bool qattr = false;
        if (!qattr) {
            cudaFuncSetAttribute((const void*)pd_f8row_gemm_tc5q_kt<QS, true>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)qsmem);
            cudaFuncSetAttribute((const void*)pd_f8row_gemm_tc5q_kt<QS, false>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)qsmem);
            qattr = true;
        }
        // K-SPLIT, BATCH-AWARE. This was a hard-coded 2, so every
        // tc5q launch wrote two partial planes of out_dim*batch floats and paid
        // a ks_combine to sum them -- the third site of the bug fixed for
        // tc5p (PADDOCK_F8T_NZ) and rejected for wmma. The split's cost scales
        // with BATCH (the planes are out_dim*batch); its benefit, CTA count,
        // does not, because batch is the kernel's n-dimension. At c32 the
        // tick still carried 209 combines per decode step costing 0.83
        // ms/step, and tc5q's fixed nzq=2 was 65 of them.
        // Split only while the batch is narrow enough that the partials are
        // cheap; above that write y directly and skip the combine entirely.
        static int nzq_env = -2;
        if (nzq_env == -2) {
            const char* e = pd_env("PADDOCK_TC5Q_NZ");
            nzq_env = e ? atoi(e) : 0;
            if (nzq_env < 0 || nzq_env > 2) nzq_env = 0;
        }
        const uint32_t nzq = nzq_env ? (uint32_t)nzq_env : (batch >= 16u ? 1u : 2u);
        // counter reset moved into the kernel's last-dry-CTA tail;
        // the memset node cost 3-8 us of kernel-idle per launch in the
        // captured decode graph. Kill: PADDOCK_NO_QRST restores it (the
        // in-kernel reset is unconditional, so the memset is redundant).
        static const bool no_qrst = pd_env("PADDOCK_NO_QRST") != nullptr;
        if (no_qrst) cudaMemsetAsync(qctr, 0, 4, st);
        // nzq==1 means one plane, so the kernel's output is final: write y and
        // skip both the partial staging and the combine below.
        float* dst = nzq > 1u ? (float*)part : (float*)y;
        // N=256 fused-pair arm: swap operands (A=Y M=64, B=W pair
        // N=256) - halves mma issue + Y stages. Wins where the pipe is
        // issue-bound: gu b=1 +5.4%, b=8 +4.0% (c1/c8 captures); flat b=33,
        // -2.6% b=64 (M=64 runs half the datapath) - so batch <= 8 only.
        // Not bit-identical to the N=64 route: the M=64 datapath rounds its
        // internal accumulation differently (maxabs 2.4e-4 @ signal rms 140,
        // few-ULP class). PADDOCK_TC5Q_N256=1 enables; even tile counts only
        // (items span 2 tiles). 40 KB slots cap the ring at S=5.
        static int n256 = -1;
        if (n256 < 0) n256 = pd_env("PADDOCK_TC5Q_N256") ? 1 : 0;
        if (n256 && !(tiles & 1u) && batch <= 8u) {
            constexpr uint32_t NS = 5u;
            // + 8448: the [64][33] f32 epilogue bounce tile after the barriers
            const uint32_t nsmem = NS * 40960u + 2u * NS * 8u + 16u + 8448u;
            static bool nattr = false;
            if (!nattr) {
                cudaFuncSetAttribute((const void*)pd_f8row_gemm_tc5q_kt<NS, true, true>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, (int)nsmem);
                cudaFuncSetAttribute((const void*)pd_f8row_gemm_tc5q_kt<NS, false, true>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, (int)nsmem);
                nattr = true;
            }
            if (!no_efq)
                pd_f8row_gemm_tc5q_kt<NS, true, true><<<(uint32_t)nsm, 128, nsmem, st>>>(
                    (const unsigned char*)wtiles, *ym, (const float*)wrs, (const float*)xrs,
                    dst, in_dim, out_dim, batch, nzq, QD, qctr);
            else
                pd_f8row_gemm_tc5q_kt<NS, false, true><<<(uint32_t)nsm, 128, nsmem, st>>>(
                    (const unsigned char*)wtiles, *ym, (const float*)wrs, (const float*)xrs,
                    dst, in_dim, out_dim, batch, nzq, QD, qctr);
        } else {
            // Ring depth / CTAs-per-SM knob (see pd_tc5q_launch). Defaults
            // reproduce the shipped launch exactly: S=6, grid = nsm.
            // !! PADDOCK_TC5Q_CTA > 1 PRODUCES wrong OUTPUT. !!
            //
            // tc5q is persistent and its CTAs-per-SM is the LAUNCH
            // grid (nsm * qcta), so raising it looks like free occupancy: at
            // c32 this kernel runs sm__warps_active 6.23% (4 warps of 64) and
            // dram__throughput 35% of peak = 2.7 TB/s on a die measured at
            // 6.93, and CTA=2 with a shallower ring came out +42%
            // end-to-end at c32.
            //
            // It is not free: the output is CORRUPT. A 32-prompt
            // column-correspondence check went 32/32 -> 0/32 on-topic, with
            // completions degenerating into repeated '!' and random-language
            // tokens within a few steps. The header comment claiming the item
            // loop is "grid-agnostic (the dry-claim reset keys off gridDim.x)"
            // does not hold for a grid wider than one CTA per SM.
            //
            // The +42% is real work the die can do, so the claim loop is worth
            // fixing  -- but until it is, CTA stays 1. Left selectable
            // only for that debugging.
            static int qs_env = -2;
            if (qs_env == -2) {
                const char* e = pd_env("PADDOCK_TC5Q_S");
                qs_env = e ? atoi(e) : 6;
                if (qs_env < 3 || qs_env > 9) qs_env = 6;
            }
            static int qcta = -2;
            if (qcta == -2) {
                const char* e = pd_env("PADDOCK_TC5Q_CTA");
                qcta = e ? atoi(e) : 1;
                if (qcta < 1 || qcta > 4) qcta = 1;
            }
            const int qs = qs_env;
            const uint32_t qgrid = (uint32_t)nsm * (uint32_t)qcta;
            const bool qef = !no_efq;
            switch (qs) {
                case 3: pd_tc5q_launch<3u>(qef, qgrid, st, (const unsigned char*)wtiles,
                            *ym, (const float*)wrs, (const float*)xrs, dst,
                            in_dim, out_dim, batch, nzq, qctr); break;
                case 4: pd_tc5q_launch<4u>(qef, qgrid, st, (const unsigned char*)wtiles,
                            *ym, (const float*)wrs, (const float*)xrs, dst,
                            in_dim, out_dim, batch, nzq, qctr); break;
                case 5: pd_tc5q_launch<5u>(qef, qgrid, st, (const unsigned char*)wtiles,
                            *ym, (const float*)wrs, (const float*)xrs, dst,
                            in_dim, out_dim, batch, nzq, qctr); break;
                // S=7..9: DeepGEMM's decode kernel runs the same
                // 1-CTA-per-SM, 148-wide grid we do, but with
                // 256 threads and an ELEVEN-stage ring in 206 KB of smem where
                // tc5q had six stages in 147 KB. Depth is the cheap half of
                // that difference: the slot is 24576 B, so the 227 KB opt-in
                // limit allows S=9 (221344 B).
                case 7: pd_tc5q_launch<7u>(qef, qgrid, st, (const unsigned char*)wtiles,
                            *ym, (const float*)wrs, (const float*)xrs, dst,
                            in_dim, out_dim, batch, nzq, qctr); break;
                case 8: pd_tc5q_launch<8u>(qef, qgrid, st, (const unsigned char*)wtiles,
                            *ym, (const float*)wrs, (const float*)xrs, dst,
                            in_dim, out_dim, batch, nzq, qctr); break;
                case 9: pd_tc5q_launch<9u>(qef, qgrid, st, (const unsigned char*)wtiles,
                            *ym, (const float*)wrs, (const float*)xrs, dst,
                            in_dim, out_dim, batch, nzq, qctr); break;
                default: pd_tc5q_launch<QS>(qef, qgrid, st, (const unsigned char*)wtiles,
                            *ym, (const float*)wrs, (const float*)xrs, dst,
                            in_dim, out_dim, batch, nzq, qctr); break;
            }
        }
        if (no_combine) {
            if (out_nz) *out_nz = nzq;
        } else if (nzq > 1u) {
            const uint32_t n = out_dim * batch;
            pd_q8_0_gemm_mma_ks_combine_kernel<<<(n + 255u) / 256u, 256, 0, st>>>(
                (const float*)part, nullptr, (float*)y, n, nzq, out_dim);
        } else if (out_nz) {
            *out_nz = 1u;                 // y is already final
        }
        return cudaPeekAtLastError() == cudaSuccess ? 0 : -2;
    }
    // K-split count by wave-quantization cost, not the old ceil heuristic:
    // wall ~ ceil(tiles*nz/nsm) waves x ceil(nk/nz) slabs per CTA (1 CTA/SM
    // at this smem). The ceil overshot at the wo/down shapes - 42x8 = 336 =
    // 2.27 waves left a 40-CTA third wave streaming at 27% of the die:
    // nz=7 lands 294 = 1.99 clean waves (tc5p_stall: wo -9%, down -15%).
    // Kill: PADDOCK_NO_NZOPT restores the ceil.
    static int no_nzopt = -1;
    if (no_nzopt < 0) no_nzopt = pd_env("PADDOCK_NO_NZOPT") ? 1 : 0;
    uint32_t nz = ((uint32_t)nsm * 2u + tiles - 1u) / tiles;
    if (nz > 8u) nz = 8u;
    if (nz > (nk_all + 1u) / 2u) nz = (nk_all + 1u) / 2u;
    if (nz < 1u) nz = 1u;
    // The K-split is priced on the GEMM's own serialized work -- `waves * per`,
    // in units of one CTA-wave processing one k-tile -- plus
    // what the split COSTS. Splitting K makes nz partial planes of
    // out_dim*batch floats that must be written and then read back by
    // pd_q8_0_gemm_mma_ks_combine, and that traffic scales with BATCH while the
    // CTA-count benefit does not (batch is the kernel's n-dimension; it adds no
    // CTAs). Without the penalty term the election returns the b=1 answer at
    // every batch: at c32 that put a combine after every one of the 192.7
    // GEMM launches per step, 0.79 ms/step = 7.3% of the tick.
    //
    // Units: one cost unit is nsm CTAs x one 128x128 k-tile = nsm*16384 weight
    // bytes. The split's extra traffic is ~8*out_dim*batch*z bytes (write the
    // planes, read them back). SC scales both so the penalty does not truncate
    // to zero at small batch. Kill: PADDOCK_NO_NZBATCH.
    if (!no_nzopt) {
        constexpr uint64_t SC = 16ull;
        const uint64_t unit = (uint64_t)nsm * 16384ull;
        uint64_t best = ~0ull;
        const uint32_t zmax = (nk_all + 1u) / 2u < 8u ? (nk_all + 1u) / 2u : 8u;
        for (uint32_t z = 1; z <= zmax; ++z) {
            const uint64_t waves = ((uint64_t)tiles * z + (uint32_t)nsm - 1u) / (uint32_t)nsm;
            const uint64_t per = (nk_all + z - 1u) / z;
            uint64_t cost = waves * per * SC;
            if (pd_nzbatch_on())
                cost += (8ull * out_dim * batch * z * SC + unit - 1ull) / unit;
            if (cost < best) { best = cost; nz = z; }   // ties keep smaller z
        }
    }
    // PADDOCK_F8T_NZ pins the K-split (unset/0 = elected above).
    //
    // The election prices only the GEMM's own wave/k-tile cost. It does not see
    // what the split creates: nz partial planes of out_dim*batch floats that get
    // written and then read back by pd_q8_0_gemm_mma_ks_combine. That term
    // scales with BATCH; the CTA-count benefit that motivates splitting does
    // not, because batch is the kernel's n-dimension and adds no CTAs. So the
    // right nz falls as batch rises, and at c32 the election is still picking
    // the b=1 answer -- a combine after every one of the 192.7 GEMM
    // launches per step, 0.79 ms/step or 7.3% of the tick.
    static int nz_pin = -2;
    if (nz_pin == -2) {
        const char* e = pd_env("PADDOCK_F8T_NZ");
        nz_pin = e ? atoi(e) : 0;
    }
    if (nz_pin > 0) {
        nz = (uint32_t)nz_pin;
        const uint32_t zc = (nk_all + 1u) / 2u;
        if (nz > zc) nz = zc;
        if (nz > 8u) nz = 8u;
        if (nz < 1u) nz = 1u;
    }
    // Ring depth: PADDOCK_TC5P_S, else elected by batch. See pd_tc5p_launch.
    // Only worth shrinking when the grid (tiles*nz) actually exceeds nsm, which
    // at narrow out_dim it does once the batch is wide.
    static int p_s_env = -2;
    if (p_s_env == -2) {
        const char* e = pd_env("PADDOCK_TC5P_S");
        p_s_env = e ? atoi(e) : 0;
        if (p_s_env && (p_s_env < 3 || p_s_env > 6)) p_s_env = 6;
    }
    static int no_ef = -1;
    if (no_ef < 0) no_ef = pd_env("PADDOCK_NO_L2EF") ? 1 : 0;
    //  arm (PADDOCK_C2COL=1): no-K-split cluster col-split for the M2
    // band - the probe showed M2's stream already rides the DRAM
    // floor and the rest is split-K overhead (partials + 7.45us/CTA epi +
    // combine). One row-tile per 2-CTA cluster, batch cols split 32/32, W+Y
    // multicast once per cluster, full-K streams, final y direct. Batch > 8
    // only (the fold path owns b<=8); 2 CTAs/tile must fit one wave; stream
    // long enough to amortize fill. Reorder class -> coherence gate.
    static int c2c = -1;
    if (c2c < 0) c2c = pd_env("PADDOCK_C2COL") ? 1 : 0;
    if (c2c && batch > 8u && tiles * 2u <= (uint32_t)nsm && nk_all >= 32u) {
// c2col kernel definition is compiled out below sm_90 (its __cluster_dims__
// guard) - drop these references from those device passes too, same as the
// tc5r launcher above; the arm is only reachable on cluster-capable devices
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 900)
        constexpr uint32_t CS = 6u;
        const uint32_t csmem = CS * 24576u + 3u * CS * 8u;
        static bool cattr = false;
        if (!cattr) {
            cudaFuncSetAttribute((const void*)pd_f8row_gemm_c2col_kt<CS, true>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)csmem);
            cudaFuncSetAttribute((const void*)pd_f8row_gemm_c2col_kt<CS, false>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)csmem);
            cattr = true;
        }
        if (!no_ef)
            pd_f8row_gemm_c2col_kt<CS, true><<<tiles * 2u, 128, csmem, st>>>(
                (const unsigned char*)wtiles, *ym, (const float*)wrs, (const float*)xrs,
                (float*)y, in_dim, out_dim, batch);
        else
            pd_f8row_gemm_c2col_kt<CS, false><<<tiles * 2u, 128, csmem, st>>>(
                (const unsigned char*)wtiles, *ym, (const float*)wrs, (const float*)xrs,
                (float*)y, in_dim, out_dim, batch);
        if (out_nz) *out_nz = 1u;
        return cudaPeekAtLastError() == cudaSuccess ? 0 : -2;
#else
        return cudaErrorNotSupported;
#endif
    }
    // in-kernel last-CTA fold replaces the combine launch (bit-equal fixed-z
    // sum) - but only at batch <= 8: fold work scales with batch while folder
    // parallelism is fixed (128 threads x tiles, all in the kernel tail), so
    // at b=33/64 the fold loses to the grid-parallel combine kernel even
    // with 8-chain ILP (wo 26.7/34.9 vs 20.5/24.6 us).
    // Kill: PADDOCK_NO_KSFOLD restores the separate combine everywhere.
    static int no_fold = -1;
    if (no_fold < 0) no_fold = pd_env("PADDOCK_NO_KSFOLD") ? 1 : 0;
    const bool want_fold = !no_fold && batch <= 8u && tiles <= 256u;
    //  (wo_ts_probe): the wave-cost model omits the launch fill-storm
    // (first slab lands 6 us in when 294 prologues fire at once) and the
    // wave-2 refill - wo b=1 measured nz=3 at 11.3 us vs the model's nz=7
    // at 15.2. On the fold path prefer the largest z that still fits one
    // wave: longest streams, no refill. Kill: PADDOCK_NO_NZ1W.
    static int no_nz1w = -1;
    if (no_nz1w < 0) no_nz1w = pd_env("PADDOCK_NO_NZ1W") ? 1 : 0;
    if (!no_nz1w && want_fold && tiles <= (uint32_t)nsm) {
        uint32_t z1 = (uint32_t)nsm / tiles;
        if (z1 > 8u) z1 = 8u;
        if (z1 > (nk_all + 1u) / 2u) z1 = (nk_all + 1u) / 2u;
        if (z1 >= 1u) nz = z1;
    }
    float* dst = nz > 1u ? (float*)part : (float*)y;
    uint32_t* fctr = want_fold ? pd_tc5p_fctr() : nullptr;
    // PDL: launch as a programmatic dependent so the kernel starts
    // inside the predecessor's execution - the dep-free W prologue streams
    // during the elementwise that produces our Y (triggers in the e4m3 row
    // kernels), and griddepcontrol.wait gates every dependent read on full
    // predecessor completion (probe: pdl_sem.cu). Same slabs, same mma
    // order -> bit-identical. tc5p_stall chain A/B: wo 13.6 -> 10.0 us
    // (-26%), down 23.1 -> 19.7 (-15%). Kill: PADDOCK_NO_PDL.
    static const bool no_pdl = pd_env("PADDOCK_NO_PDL") != nullptr;
    // L2-prefetch gate: whole-plane W prefetch only when the GEMM's
    // aggregate W fits L2 with headroom (wo 44MB / qkv 88MB yes; down 115MB /
    // gu 231MB would thrash the 126MB die). PDL launches only - the
    // early-launch window is what pays for it. Kill: PADDOCK_NO_L2PF.
    // FALSIFIED default (board v60 + repeated c1/dc4 A/B): whole-
    // stream prefetch was FLAT within noise at shallow batch (the TMA loads
    // already populate L2 through the early-launch window) and STOLE DRAM
    // from the co-resident attention band at c32 (1501.7 -> 1478.2).
    // Kept as explicit opt-in (PADDOCK_L2PF=1) for re-examination when the
    // co-execution picture changes (e.g. wide-spec verify ticks).
    static const bool want_l2pf = pd_env("PADDOCK_L2PF") != nullptr;
    const uint32_t l2pf =
        (want_l2pf && batch <= 16u
         && (size_t)tiles * ((in_dim + 127u) / 128u) * 16384u <= (100u << 20))
            ? 1u : 0u;
    // M2 route: two row-tiles per CTA when the pair grid still
    // fills >=75% of the die - halves CTA count, doubles stream length,
    // halves Y traffic; same nz as ::1 so it is bit-identical per row.
    // No-fold path only (the ksfold b<=8 regime keeps the ::1 form).
    // Kill: PADDOCK_NO_TC5M2.
    static const bool no_m2 = pd_env("PADDOCK_NO_TC5M2") != nullptr;
    if (!no_m2 && !fctr && (tiles & 1u) == 0u
        && (tiles / 2u) * nz * 4u >= (uint32_t)nsm * 3u) {
        constexpr uint32_t SM2 = 4u;
        const uint32_t smem2 = SM2 * 40960u + 2u * SM2 * 8u;
        static bool attr2 = false;
        if (!attr2) {
            cudaFuncSetAttribute((const void*)pd_f8row_gemm_tc5p_m2_kt<SM2, true, true>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem2);
            cudaFuncSetAttribute((const void*)pd_f8row_gemm_tc5p_m2_kt<SM2, false, true>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem2);
            cudaFuncSetAttribute((const void*)pd_f8row_gemm_tc5p_m2_kt<SM2, true>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem2);
            cudaFuncSetAttribute((const void*)pd_f8row_gemm_tc5p_m2_kt<SM2, false>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem2);
            attr2 = true;
        }
        bool m2_launched = false;
        if (!no_pdl) {
            cudaLaunchConfig_t cfg = {};
            cudaLaunchAttribute at[1];
            at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
            at[0].val.programmaticStreamSerializationAllowed = 1;
            cfg.gridDim = dim3(tiles / 2u, nz); cfg.blockDim = dim3(128);
            cfg.dynamicSmemBytes = smem2; cfg.stream = st;
            cfg.attrs = at; cfg.numAttrs = 1;
            const cudaError_t le = !no_ef
                ? cudaLaunchKernelEx(&cfg, pd_f8row_gemm_tc5p_m2_kt<SM2, true, true>,
                      (const unsigned char*)wtiles, *ym, (const float*)wrs,
                      (const float*)xrs, dst, in_dim, out_dim, batch, nz, 3u, l2pf)
                : cudaLaunchKernelEx(&cfg, pd_f8row_gemm_tc5p_m2_kt<SM2, false, true>,
                      (const unsigned char*)wtiles, *ym, (const float*)wrs,
                      (const float*)xrs, dst, in_dim, out_dim, batch, nz, 3u, l2pf);
            m2_launched = le == cudaSuccess;
        }
        if (!m2_launched) {
            if (!no_ef)
                pd_f8row_gemm_tc5p_m2_kt<SM2, true><<<dim3(tiles / 2u, nz), 128, smem2, st>>>(
                    (const unsigned char*)wtiles, *ym, (const float*)wrs, (const float*)xrs,
                    dst, in_dim, out_dim, batch, nz, 3u, 0u);
            else
                pd_f8row_gemm_tc5p_m2_kt<SM2, false><<<dim3(tiles / 2u, nz), 128, smem2, st>>>(
                    (const unsigned char*)wtiles, *ym, (const float*)wrs, (const float*)xrs,
                    dst, in_dim, out_dim, batch, nz, 3u, 0u);
        }
        if (nz > 1u && !no_combine) {
            const uint32_t n = out_dim * batch;
            pd_q8_0_gemm_mma_ks_combine_kernel<<<(n + 255u) / 256u, 256, 0, st>>>(
                (const float*)part, nullptr, (float*)y, n, nz, out_dim);
        }
        if (out_nz) *out_nz = (nz > 1u && no_combine) ? nz : 1u;
        return cudaPeekAtLastError() == cudaSuccess ? 0 : -2;
    }
    {
        // Shrink the ring only when there are spare CTAs to fill the SM with:
        // grid is tiles*nz, so this fires on the narrow planes at wide batch,
        // which is exactly where tc5p profiles at 3.03 TB/s and 4 warps/SM.
        // MEASURED: shrinking the ring does not pay, and the profile says
        // why. S=4 raises launch__occupancy_limit_shared_mem from 1 to 2 but
        // leaves sm__warps_active at 6.20%, unchanged -- because tc5p's grids
        // are (129,1,1) and (40,3,1), i.e. 120-129 CTAs on 148 SMs. There are
        // not enough CTAs to put two on any SM, so per-SM occupancy is not the
        // binding constraint; the die is not even filled once. c32 reads 2146
        // at S=6 against 2122 at S=4.
        // More CTAs can only come from K-split, and that is already optimal:
        // pinning nz=2 gives 2096 and nz=3 gives 2044, both below the elected
        // 2146. The real fix is finer output tiles (tc5p uses 128 rows; 64
        // would double the CTA count) -- .
        int ps = p_s_env ? p_s_env : 6;
        const bool pdl = !no_pdl, ef = !no_ef;
        #define PD_TC5P_GO(SS) pd_tc5p_launch<SS>(ef, pdl, tiles, nz, st,        \
            (const unsigned char*)wtiles, *ym, (const float*)wrs,               \
            (const float*)xrs, dst, in_dim, out_dim, batch, (float*)y, fctr, l2pf)
        switch (ps) {
            case 3:  PD_TC5P_GO(3u); break;
            case 4:  PD_TC5P_GO(4u); break;
            case 5:  PD_TC5P_GO(5u); break;
            default: PD_TC5P_GO(6u); break;
        }
        #undef PD_TC5P_GO
    }
    if (nz > 1u && !fctr && !no_combine) {
        const uint32_t n = out_dim * batch;
        pd_q8_0_gemm_mma_ks_combine_kernel<<<(n + 255u) / 256u, 256, 0, st>>>(
            (const float*)part, nullptr, (float*)y, n, nz, out_dim);
    }
    if (out_nz) *out_nz = (nz > 1u && !fctr && no_combine) ? nz : 1u;
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -2;
#endif
}

PD_EXPORT
int pd_f8t_gemm(const void* wtiles, const void* wrs, const void* xq, const void* xrs,
                void* part, void* y, uint32_t in_dim, uint32_t out_dim,
                uint32_t batch, void* stream) {
    return pd_f8t_gemm_impl(wtiles, wrs, xq, xrs, part, y, in_dim, out_dim, batch,
                            0u, nullptr, stream);
}

// no-combine form - leaves the nz partial planes in `part` and
// reports nz so an nz-aware CONSUMER (pd_addnorm_e4m3_nz /
// pd_quantize_e4m3_geglu2_nz) absorbs the fixed-z sum at full grid
// parallelism. *out_nz == 1 means y is already FINAL (tc5r direct, in-kernel
// b<=8 fold, or a single plane) and the consumer should read y with nzp 1.
PD_EXPORT
int pd_f8t_gemm2(const void* wtiles, const void* wrs, const void* xq, const void* xrs,
                 void* part, void* y, uint32_t in_dim, uint32_t out_dim,
                 uint32_t batch, uint32_t no_combine, uint32_t* out_nz, void* stream) {
    return pd_f8t_gemm_impl(wtiles, wrs, xq, xrs, part, y, in_dim, out_dim, batch,
                            no_combine, out_nz, stream);
}

PD_EXPORT
int pd_f8_gemm_w8(const void* data, const void* scale, const void* xq,
                  const void* xs, void* y, uint32_t in_dim, uint32_t out_dim,
                  uint32_t batch, void* stream) {
#ifndef PD_BS_HOST
    (void)data; (void)scale; (void)xq; (void)xs; (void)y; (void)in_dim;
    (void)out_dim; (void)batch; (void)stream;
    return cudaErrorNotSupported;
#else
    if (out_dim == 0 || batch == 0) return 0;
    if ((in_dim & 31u) != 0) return cudaErrorInvalidValue;
#ifdef PD_TC5_HOST
    // tcgen05 BLOCK-SCALE route (cc 10, this build carries sm_100a SASS):
    // the hardware ue8m0 fold the sw-fold _kt lacked - bit-exact vs the
    // reference per-32 model. Kill: PADDOCK_NO_TC5.
    static const bool tc5b = [] {
        int dev = 0, cc = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&cc, cudaDevAttrComputeCapabilityMajor, dev);
        return cc == 10 && pd_tmap_encode() != nullptr
            && pd_env("PADDOCK_NO_TC5") == nullptr;
    }();
    if (tc5b && (in_dim & 127u) == 0u) {
        CUtensorMap wm, ym;
        if (pd_tmap_2d(&wm, data, in_dim, out_dim) &&
            pd_tmap_2d(&ym, xq, in_dim, batch)) {
            // tc5s persistent ::2 election: the M-large band's
            // loss was wave quantization + per-tile overheads, and the
            // persistent cluster kernel beats tc5v by 8-47% per shape at
            // the c32 burst sizes - EXCEPT when the tile count quantizes
            // badly over the cluster count (down/wo at r~1013/2048 lose up
            // to 26%), so gate on ceil-share <= 1.3x. Kill: PADDOCK_NO_TC5S.
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 900)
            static const bool no_s5 = pd_env("PADDOCK_NO_TC5S") != nullptr;
            if (!no_s5 && batch >= 768u) {
                static const uint32_t sgrid = [] {
                    int dev = 0, nn = 0;
                    cudaGetDevice(&dev);
                    cudaDeviceGetAttribute(&nn, cudaDevAttrMultiProcessorCount, dev);
                    return (uint32_t)(nn & ~1);
                }();
                const uint32_t pairs = (((out_dim + 127u) >> 7) + 1u) >> 1;
                const uint32_t tcnt = pairs * ((batch + 255u) >> 8);
                const uint32_t ncl = sgrid >> 1;
                const uint32_t sper = (tcnt + ncl - 1u) / ncl;
                if (sgrid >= 4u && tcnt >= ncl
                    && (uint64_t)sper * ncl * 10u <= (uint64_t)tcnt * 13u) {
                    constexpr uint32_t SS = 6u;
                    const uint32_t ssm = 2u * SS * 16384u + 2u * SS * 1024u
                        + (3u * SS + 4u) * 8u + 64u;
                    // SF laws (both twins probe-refuted as speed levers):
                    // SFPF re-banking = bit-exact but slower (the pipe runs
                    // cp/mma strictly in issue order); batched 128x256b SF
                    // cps (the tc5sb arm, bit-exact) =
                    // -23-25%: the tensor pipe prices cps by SOURCE bytes
                    // (~8.4 B/ns, ~85ns op floor) and warpx4's 4x broadcast
                    // amplification is free, so classic already sits at the
                    // SF byte floor (1.5KB/step) AND the op floor (3 cps).
                    // Step ~597ns = ~255ns SF + ~340ns mma is the FORMAT
                    // floor for per-32 scales; DeepGEMM's ~2.8PF rides
                    // per-128 granularity. The mainloop road is CLOSED.
                    // What did move: ring depth S=4 beats S=6 - 143KB smem
                    // vs 209KB returns L1 to the producer's cp.async.ca
                    // scale streams. Real-entry m=2871 (stable over reps):
                    // down +15%, out/o +8%, gate +4%, attnqkv +1.4%, qkvz
                    // -0.3%, gu-fused -1.6%. Election: S4 iff out_dim <=
                    // 17408 (every plane but gu-fused; ~+3.3% on the wave
                    // mix). Force-all: PADDOCK_TC5S_S4=1; kill:
                    // PADDOCK_NO_TC5S_S4.
                    static const bool sfpf = pd_env("PADDOCK_TC5S_SFPF") != nullptr;
                    static const bool s4_all = pd_env("PADDOCK_TC5S_S4") != nullptr;
                    static const bool no_s4 = pd_env("PADDOCK_NO_TC5S_S4") != nullptr;
                    const bool s4 = !no_s4 && (s4_all || out_dim <= 17408u);
                    constexpr uint32_t S4 = 4u;
                    const uint32_t ssm4 = 2u * S4 * 16384u + 2u * S4 * 1024u
                        + (3u * S4 + 4u) * 8u + 64u;
                    static bool sattr = false;
                    if (!sattr) {
                        cudaFuncSetAttribute((const void*)pd_f8bs_gemm_tc5s_kt<SS, false>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)ssm);
                        cudaFuncSetAttribute((const void*)pd_f8bs_gemm_tc5s_kt<SS, false>,
                            cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
                        cudaFuncSetAttribute((const void*)pd_f8bs_gemm_tc5s_kt<SS, true>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)ssm);
                        cudaFuncSetAttribute((const void*)pd_f8bs_gemm_tc5s_kt<SS, true>,
                            cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
                        cudaFuncSetAttribute((const void*)pd_f8bs_gemm_tc5s_kt<S4, false>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)ssm4);
                        cudaFuncSetAttribute((const void*)pd_f8bs_gemm_tc5s_kt<S4, false>,
                            cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
                        sattr = true;
                    }
                    if (s4)
                        pd_f8bs_gemm_tc5s_kt<S4, false><<<sgrid, 320, ssm4, (cudaStream_t)stream>>>(
                            wm, ym, (const unsigned char*)scale, (const unsigned char*)xs,
                            (float*)y, in_dim, out_dim, batch);
                    else if (sfpf)
                        pd_f8bs_gemm_tc5s_kt<SS, true><<<sgrid, 320, ssm, (cudaStream_t)stream>>>(
                            wm, ym, (const unsigned char*)scale, (const unsigned char*)xs,
                            (float*)y, in_dim, out_dim, batch);
                    else
                        pd_f8bs_gemm_tc5s_kt<SS, false><<<sgrid, 320, ssm, (cudaStream_t)stream>>>(
                            wm, ym, (const unsigned char*)scale, (const unsigned char*)xs,
                            (float*)y, in_dim, out_dim, batch);
                    return pd_launch_status();
                }
            }
#endif
            // WIDE tile above 2 col-tiles of rows: the 128x128 tc5_kt is
            // L2-BW-bound and tc5w's shared W stage buys +40 %
            // (900 -> 1263 TF at the gate shape, bit-exact). Below 256 rows
            // the second Y tile would be pure OOB padding - keep tc5_kt.
            // Kill: PADDOCK_NO_TC5W.
            static const bool no_w = pd_env("PADDOCK_NO_TC5W") != nullptr;
            if (!no_w && batch >= 256u) {
                // NT=3 (N=384) pays when the tail Y tile amortizes (big r)
                // or the shape is row-tile-poor and wave-starved (down proj:
                // 42 row tiles, +40 % measured at r=1024). S=2 measured best
                // for NT=3, S=3 for NT=2 - the wider slab already gives the
                // TMA enough in-flight bytes.
                // (the down-shape NT3 special case won +40 % isolated at
                // r=1024 but cost ~1 % at c32 in serving - production chunks
                // interleave with decode and the isolated wave arithmetic
                // doesn't transfer; batch-only gate keeps the pf8 win)
                // NT3 also for the row-tile-poor down/wo shapes (out <=
                // in) from r>=1024: +48% isolated under tc5v (978->1448 at
                // out=5376/r=1024 - the serving loss for this special
                // case was a drain-regime artifact). Gate shapes (out > in)
                // keep the 1536 floor.
                // qkv-class exception (measured r=2048): out > in
                // but out <= 18432 (qkv 16384/12288) runs 768 NT3 CTAs =
                // 5.2 -> 6 ceil-waves (13.5% tail) vs 1024 NT2 = 6.9 -> 7
                // (1.2%): NT2 1676 TF vs NT3 1579. gu (43008) stays NT3
                // (2016 CTAs, 13.6 waves, 1802 TF).
                const bool nt3 = (batch >= 1536u
                                  && !(out_dim > in_dim && out_dim <= 18432u))
                    || (batch >= 1024u && out_dim <= in_dim);
                // Default: tc5v (pipelined issuer,  - +21-24% over
                // tc5w at every shape; S=3 pays once the per-stage drain is
                // gone). Kill: PADDOCK_NO_TC5V falls back to the tc5w
                // routes. (tc5z, the A-from-tmem variant, measured NEGATIVE
                // and is bench-only; tc5c/tc5n were neutral -.)
                static const bool no_v = pd_env("PADDOCK_NO_TC5V") != nullptr;
                if (nt3) {
                    constexpr uint32_t NT = 3u;
                    if (!no_v) {
                        constexpr uint32_t VS = 3u;
                        const uint32_t smem = (1u + NT) * VS * 16384u
                            + VS * (512u + NT * 512u) + 2u * VS * 8u;
                        static bool atv3 = false;
                        if (!atv3) {
                            cudaFuncSetAttribute((const void*)pd_f8bs_gemm_tc5v_kt<VS, NT>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
                            atv3 = true;
                        }
                        const uint32_t bp = (batch + NT * 128u - 1u) / (NT * 128u) * (NT * 128u);
                        const uint32_t nt = ((out_dim + 127u) / 128u) * (bp / (NT * 128u));
                        pd_f8bs_gemm_tc5v_kt<VS, NT><<<nt, 160, smem, (cudaStream_t)stream>>>(
                            wm, ym, (const unsigned char*)scale, (const unsigned char*)xs,
                            (float*)y, in_dim, out_dim, batch);
                        return pd_launch_status();
                    }
                    constexpr uint32_t WS = 2u;
                    const uint32_t smem = (1u + NT) * WS * 16384u
                        + WS * (512u + NT * 512u) + 2u * WS * 8u;
                    static bool atw3 = false;
                    if (!atw3) {
                        cudaFuncSetAttribute((const void*)pd_f8bs_gemm_tc5w_kt<WS, NT>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
                        atw3 = true;
                    }
                    const uint32_t bp = (batch + NT * 128u - 1u) / (NT * 128u) * (NT * 128u);
                    const uint32_t nt = ((out_dim + 127u) / 128u) * (bp / (NT * 128u));
                    pd_f8bs_gemm_tc5w_kt<WS, NT><<<nt, 128, smem, (cudaStream_t)stream>>>(
                        wm, ym, (const unsigned char*)scale, (const unsigned char*)xs,
                        (float*)y, in_dim, out_dim, batch);
                    return pd_launch_status();
                }
                constexpr uint32_t WS = 3u, NT = 2u;
                const uint32_t smem = (1u + NT) * WS * 16384u
                    + WS * (512u + NT * 512u) + 2u * WS * 8u;
                const uint32_t bp = (batch + NT * 128u - 1u) / (NT * 128u) * (NT * 128u);
                const uint32_t nt = ((out_dim + 127u) / 128u) * (bp / (NT * 128u));
                if (!no_v) {
                    static bool atv = false;
                    if (!atv) {
                        cudaFuncSetAttribute((const void*)pd_f8bs_gemm_tc5v_kt<WS, NT>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
                        atv = true;
                    }
                    pd_f8bs_gemm_tc5v_kt<WS, NT><<<nt, 160, smem, (cudaStream_t)stream>>>(
                        wm, ym, (const unsigned char*)scale, (const unsigned char*)xs,
                        (float*)y, in_dim, out_dim, batch);
                    return pd_launch_status();
                }
                static bool atw = false;
                if (!atw) {
                    cudaFuncSetAttribute((const void*)pd_f8bs_gemm_tc5w_kt<WS, NT>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
                    atw = true;
                }
                pd_f8bs_gemm_tc5w_kt<WS, NT><<<nt, 128, smem, (cudaStream_t)stream>>>(
                    wm, ym, (const unsigned char*)scale, (const unsigned char*)xs,
                    (float*)y, in_dim, out_dim, batch);
                return pd_launch_status();
            }
            constexpr uint32_t TS = 3u;
            const uint32_t smem = 2u * TS * 16384u + TS * 1024u + 2u * TS * 8u;
            static bool at = false;
            if (!at) {
                cudaFuncSetAttribute((const void*)pd_f8bs_gemm_tc5_kt<TS>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
                at = true;
            }
            const uint32_t bp = (batch + 127u) & ~127u;
            const uint32_t nt = ((out_dim + 127u) / 128u) * (bp >> 7);
            pd_f8bs_gemm_tc5_kt<TS><<<nt, 128, smem, (cudaStream_t)stream>>>(
                wm, ym, (const unsigned char*)scale, (const unsigned char*)xs,
                (float*)y, in_dim, out_dim, batch);
            return pd_launch_status();
        }
    }
#endif
    // STAGES pipeline depth (env PADDOCK_F8W8_STAGES, default 2 = original).
    static const uint32_t stages = [] {
        const char* e = pd_env("PADDOCK_F8W8_STAGES");
        uint32_t s = e ? (uint32_t)atoi(e) : 2u;
        return s < 2u ? 2u : (s > 4u ? 4u : s);
    }();
    // The env-selected variant ladder below (TMA/ws/n2/res/wy23/...) is all
    // sm_120a SASS (hardware block-scale mma); on any other device those
    // kernels are empty shells, so gate the whole ladder on the running cc
    // and let non-12 devices take the portable _kt ring at the bottom.
    static const bool sm120a = [] {
        int dev = 0, cc = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&cc, cudaDevAttrComputeCapabilityMajor, dev);
        return cc == 12;
    }();
    // TMA warp-specialized schedule (PADDOCK_F8W8_TMA=1): bulk-tensor loads +
    // mbarrier handoff, ~66 KB smem. Falls back to the cp.async ws kernel if
    // the driver can't hand out cuTensorMapEncodeTiled. The BASE tma_kt is
    // portable (sm_90+ TMA + sw ue8m0 fold); the tuning sub-variants below
    // are still 120a SASS and individually keep the sm120a guard. cc >= 9
    // required: Ada is f8w8-capable but has no TMA unit - its tma_kt body
    // compiles empty (PD_F8W8_TMA_OK), and launching it would be a silent
    // no-op (the encode call itself succeeds on Ada drivers).
    static const bool sm90p = [] {
        int dev = 0, cc = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&cc, cudaDevAttrComputeCapabilityMajor, dev);
        return cc >= 9;
    }();
    static const bool tma = sm90p && pd_env("PADDOCK_F8W8_TMA") != nullptr;
    // residency kernel (PADDOCK_F8W8_RES=1): cross-half fragment double
    // buffering at 320 threads - the clock64-model build
    static const bool fres = sm120a && pd_env("PADDOCK_F8W8_RES") != nullptr;
    if (tma && fres && pd_tmap_encode() != nullptr) {
        const uint32_t smemR = 67600u;
        static bool ares = false;
        if (!ares) {
            cudaFuncSetAttribute((const void*)pd_f8_gemm_w8_res_kt,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemR);
            ares = true;
        }
        CUtensorMap wm, ym;
        if (pd_tmap_2d(&wm, data, in_dim, out_dim) &&
            pd_tmap_2d(&ym, xq, in_dim, batch)) {
            const uint32_t bp = (batch + 127u) & ~127u;
            const uint32_t nt = ((out_dim + 127u) / 128u) * (bp >> 7);
            pd_f8_gemm_w8_res_kt<<<nt, 320, smemR, (cudaStream_t)stream>>>(
                wm, ym, (const unsigned char*)scale, (const unsigned char*)xs,
                (float*)y, in_dim, out_dim, batch);
            return pd_launch_status();
        }
    }
    // feed-depth rung (PADDOCK_F8W8_WY23=1): asymmetric W2/Y3 ring,
    // decoupled producer streams - see pd_f8_gemm_w8_wy23_kt
    static const bool wy23 = sm120a && pd_env("PADDOCK_F8W8_WY23") != nullptr;
    if (tma && wy23 && pd_tmap_encode() != nullptr) {
        const uint32_t smemW = 84544u;
        static bool awy = false;
        if (!awy) {
            cudaFuncSetAttribute((const void*)pd_f8_gemm_w8_wy23_kt,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemW);
            awy = true;
        }
        CUtensorMap wm, ym;
        if (pd_tmap_2d(&wm, data, in_dim, out_dim) &&
            pd_tmap_2d(&ym, xq, in_dim, batch)) {
            const uint32_t bp = (batch + 127u) & ~127u;
            const uint32_t nt = ((out_dim + 127u) / 128u) * (bp >> 7);
            pd_f8_gemm_w8_wy23_kt<<<nt, 384, smemW, (cudaStream_t)stream>>>(
                wm, ym, (const unsigned char*)scale, (const unsigned char*)xs,
                (float*)y, in_dim, out_dim, batch);
            return pd_launch_status();
        }
    }
    // feed-latency rung (PADDOCK_F8W8_SPLIT=1): W/Y arrive-early split
    static const bool tsplit = sm120a && pd_env("PADDOCK_F8W8_SPLIT") != nullptr;
    if (tma && tsplit && pd_tmap_encode() != nullptr) {
        const uint32_t smemS = 67648u;
        static bool asplit = false;
        if (!asplit) {
            cudaFuncSetAttribute((const void*)pd_f8_gemm_w8_tma2s_kt,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemS);
            asplit = true;
        }
        CUtensorMap wm, ym;
        if (pd_tmap_2d(&wm, data, in_dim, out_dim) &&
            pd_tmap_2d(&ym, xq, in_dim, batch)) {
            const uint32_t bp = (batch + 127u) & ~127u;
            const uint32_t nt = ((out_dim + 127u) / 128u) * (bp >> 7);
            pd_f8_gemm_w8_tma2s_kt<<<nt, 384, smemS, (cudaStream_t)stream>>>(
                wm, ym, (const unsigned char*)scale, (const unsigned char*)xs,
                (float*)y, in_dim, out_dim, batch);
            return pd_launch_status();
        }
    }
    // tile-geometry candidate B (PADDOCK_F8W8_TMA16=1): 16 consumer warps
    // of a 32x32 warp tile on the same 128x128 CTA tile - see the kernel
    static const bool tma16 = sm120a && pd_env("PADDOCK_F8W8_TMA16") != nullptr;
    if (tma && tma16 && pd_tmap_encode() != nullptr) {
        const uint32_t smem16 = 67600u;
        static bool atma16 = false;
        if (!atma16) {
            cudaFuncSetAttribute((const void*)pd_f8_gemm_w8_tma16_kt,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem16);
            atma16 = true;
        }
        CUtensorMap wm, ym;
        if (pd_tmap_2d(&wm, data, in_dim, out_dim) &&
            pd_tmap_2d(&ym, xq, in_dim, batch)) {
            const uint32_t bp = (batch + 127u) & ~127u;
            const uint32_t nt = ((out_dim + 127u) / 128u) * (bp >> 7);
            pd_f8_gemm_w8_tma16_kt<<<nt, 576, smem16, (cudaStream_t)stream>>>(
                wm, ym, (const unsigned char*)scale, (const unsigned char*)xs,
                (float*)y, in_dim, out_dim, batch);
            return pd_launch_status();
        }
    }
    // rung-1 pipeline (PADDOCK_F8W8_TMA4=1): 4-deep 64B ring + JIT B loads,
    // bit-identical accumulation - see pd_f8_gemm_w8_tma4_kt
    static const bool tma4 = sm120a && pd_env("PADDOCK_F8W8_TMA4") != nullptr;
    if (tma && tma4 && pd_tmap_encode() != nullptr) {
        const uint32_t smem4 = 67600u;
        static bool atma4 = false;
        if (!atma4) {
            cudaFuncSetAttribute((const void*)pd_f8_gemm_w8_tma4_kt,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem4);
            atma4 = true;
        }
        CUtensorMap wm, ym;
        if (pd_tmap_2d(&wm, data, in_dim, out_dim) &&
            pd_tmap_2d(&ym, xq, in_dim, batch)) {
            const uint32_t bp = (batch + 127u) & ~127u;
            const uint32_t nt = ((out_dim + 127u) / 128u) * (bp >> 7);
            pd_f8_gemm_w8_tma4_kt<<<nt, 384, smem4, (cudaStream_t)stream>>>(
                wm, ym, (const unsigned char*)scale, (const unsigned char*)xs,
                (float*)y, in_dim, out_dim, batch);
            return pd_launch_status();
        }
    }
    if (tma && pd_tmap_encode() != nullptr) {
        // write-windowing rung (PADDOCK_F8W8_WIN=1): measured -3% at
        // gate/2048 - see the kernel comment; opt-in for re-testing only
        static const bool twin = sm120a && pd_env("PADDOCK_F8W8_WIN") != nullptr;
        const uint32_t smem = 67600u;
        static bool atma = false;
        if (!atma) {
            cudaFuncSetAttribute((const void*)pd_f8_gemm_w8_tma_kt<false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
            cudaFuncSetAttribute((const void*)pd_f8_gemm_w8_tma_kt<true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
            atma = true;
        }
        CUtensorMap wm, ym;
        if (pd_tmap_2d(&wm, data, in_dim, out_dim) &&
            pd_tmap_2d(&ym, xq, in_dim, batch)) {
            const uint32_t bp = (batch + 127u) & ~127u;
            const uint32_t nt = ((out_dim + 127u) / 128u) * (bp >> 7);
            if (twin)
                pd_f8_gemm_w8_tma_kt<true><<<nt, 384, smem, (cudaStream_t)stream>>>(
                    wm, ym, (const unsigned char*)scale, (const unsigned char*)xs,
                    (float*)y, in_dim, out_dim, batch);
            else
                pd_f8_gemm_w8_tma_kt<false><<<nt, 384, smem, (cudaStream_t)stream>>>(
                    wm, ym, (const unsigned char*)scale, (const unsigned char*)xs,
                    (float*)y, in_dim, out_dim, batch);
            return pd_launch_status();
        }
    }
    // Warp-specialized schedule (PADDOCK_F8W8_WS=1): fixed 2-pair K-128 ring,
    // 80 KB smem, 384 threads. Wins over N2 when both set.
    static const bool ws = sm120a && pd_env("PADDOCK_F8W8_WS") != nullptr;
    if (ws) {
        const uint32_t smem = 2u * 4u * 128u * PD_BS_W8_ROW;  // 2 planes x 4 slabs
        static bool aws = false;
        if (!aws) {
            cudaFuncSetAttribute((const void*)pd_f8_gemm_w8_ws_kt,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
            aws = true;
        }
        const uint32_t bp = (batch + 127u) & ~127u;
        const uint32_t nt = ((out_dim + 127u) / 128u) * (bp >> 7);
        const cudaStream_t st = (cudaStream_t)stream;
        auto* dp = (const unsigned char*)data; auto* scp = (const unsigned char*)scale;
        auto* xqp = (const unsigned char*)xq; auto* xsp = (const unsigned char*)xs;
        pd_f8_gemm_w8_ws_kt<<<nt, 384, smem, st>>>(dp, scp, xqp, xsp, (float*)y, in_dim, out_dim, batch);
        return pd_launch_status();
    }
    static const bool n2 = sm120a && pd_env("PADDOCK_F8W8_N2") != nullptr;
    if (n2) {
        const uint32_t smem = stages * (128u + 256u) * PD_BS_W8_ROW;
        static bool a2 = false;
        if (!a2) {
            cudaFuncSetAttribute((const void*)pd_f8_gemm_w8_n2_kt<2u>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(2u * (128u + 256u) * PD_BS_W8_ROW));
            cudaFuncSetAttribute((const void*)pd_f8_gemm_w8_n2_kt<3u>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(3u * (128u + 256u) * PD_BS_W8_ROW));
            a2 = true;
        }
        const uint32_t bp2 = (batch + 255u) & ~255u;
        const uint32_t nt = ((out_dim + 127u) / 128u) * (bp2 >> 8);
        const cudaStream_t st = (cudaStream_t)stream;
        auto* dp = (const unsigned char*)data; auto* scp = (const unsigned char*)scale;
        auto* xqp = (const unsigned char*)xq; auto* xsp = (const unsigned char*)xs;
        if (stages >= 3u)
            pd_f8_gemm_w8_n2_kt<3u><<<nt, 256, smem, st>>>(dp, scp, xqp, xsp, (float*)y, in_dim, out_dim, batch);
        else
            pd_f8_gemm_w8_n2_kt<2u><<<nt, 256, smem, st>>>(dp, scp, xqp, xsp, (float*)y, in_dim, out_dim, batch);
        return pd_launch_status();
    }
    const uint32_t smem = 2u * stages * 128u * PD_BS_W8_ROW;
    static bool attr_done = false;
    if (!attr_done) {
        cudaFuncSetAttribute((const void*)pd_f8_gemm_w8_kt<2u>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(2u * 2u * 128u * PD_BS_W8_ROW));
        cudaFuncSetAttribute((const void*)pd_f8_gemm_w8_kt<3u>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(2u * 3u * 128u * PD_BS_W8_ROW));
        cudaFuncSetAttribute((const void*)pd_f8_gemm_w8_kt<4u>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(2u * 4u * 128u * PD_BS_W8_ROW));
        attr_done = true;
    }
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t ntiles = ((out_dim + 127u) / 128u) * (batch_pad >> 7);
    const cudaStream_t st = (cudaStream_t)stream;
    auto* dp = (const unsigned char*)data; auto* scp = (const unsigned char*)scale;
    auto* xqp = (const unsigned char*)xq; auto* xsp = (const unsigned char*)xs;
    if (stages == 4u)
        pd_f8_gemm_w8_kt<4u><<<ntiles, 256, smem, st>>>(dp, scp, xqp, xsp, (float*)y, in_dim, out_dim, batch);
    else if (stages == 3u)
        pd_f8_gemm_w8_kt<3u><<<ntiles, 256, smem, st>>>(dp, scp, xqp, xsp, (float*)y, in_dim, out_dim, batch);
    else
        pd_f8_gemm_w8_kt<2u><<<ntiles, 256, smem, st>>>(dp, scp, xqp, xsp, (float*)y, in_dim, out_dim, batch);
    return pd_launch_status();
#endif
}


// ---- e4m3 GEMV (the replace-design keystone) ------------------------------
// Decode-lane twin of pd_q8_0_gemv_repacked over the f8w planes (e4m3 bytes
// + ue8m0 per-32 scales): identical block/warp structure, identical fold
// order, one F2F convert per element where q8 does I2F. Purpose: batch<=8
// FFN GEMMs at the weight-bandwidth floor so the q8 FFN planes can be
// DROPPED (the TMA GEMM's 128-col tile pays 2x at tiny batch - a shape
// overhead, not a format one). Scale decode: ue8m0 byte s -> 2^(s-127)
// == __int_as_float(s << 23) (denormal-free by construction of q8_0_to_f8w).
// fp4 GEMV: pd_f8_gemv's structure over packed
// e2m1 nibbles - one 16B load = one full 32-block (split order: low nibble
// of byte j = elem j, high = elem j+16), so the weight stream HALVES vs
// e4m3 at the same loop shape. Decode via pd_e2m1_val (moe/decode_block_scale.cuh),
// scale = ue8m0 exponent add, f32 accumulate - same numeric class as the
// fp4 GEMM lanes (block-scale e2m1 x f32-exact activations).
__global__ void pd_fp4_gemv_kernel(
    const uint8_t* __restrict__ data, const uint8_t* __restrict__ scale,
    const float* __restrict__ bias, const float* __restrict__ x, float* __restrict__ y,
    uint32_t in_dim, uint32_t out_dim) {
#if PD_BS_OK
    uint32_t o = blockIdx.x;
    if (o >= out_dim) return;
    uint32_t tid = threadIdx.x, nth = blockDim.x;
    uint32_t n_blocks = in_dim >> 5;
    extern __shared__ float fp4gs[];
    const uint8_t* srow = scale + (size_t)o * n_blocks;
    for (uint32_t b = tid; b < n_blocks; b += nth)
        fp4gs[b] = __int_as_float(((int)srow[b]) << 23);
    __shared__ float wsum[32];
    __syncthreads();
    const uint8_t* row = data + (size_t)o * (in_dim >> 1);
    float acc = 0.0f;
    // prmt decode (v1 scalar ALU 609.9us, smem LUT 675.0 - both decode-
    // bound): e2m1 nibble = sign(bit3) | mag(bits 0..2); __byte_perm over
    // the 8-entry e4m3-byte magnitude table {0,0.5,1,1.5,2,3,4,6} =
    // {00,30,38,3C,40,44,48,4C} converts 4 nibbles per prmt (two OR-shift
    // folds compress the mag bytes into the 16-bit selector), the sign ORs
    // in at bit 7 (e2m1 grid embeds exactly in e4m3), and the hardware fp8
    // casts finish. No smem, ~2 int ops/elem on top of the e4m3 flow.
    constexpr uint32_t T0 = 0x3C383000u, T1 = 0x4C484440u;
    // element-major striding (16 elems/thread/iter like the e4m3 gemv):
    // a block-per-thread mapping idled 88 of 256 threads (168 blocks) and
    // left 174.6us; here the two threads on a block read the same 16B
    // (L1-served) and each decodes one nibble half
    for (uint32_t base = tid * 16u; base < in_dim; base += nth * 16u) {
        const uint32_t blk = base >> 5;
        const uint32_t half = (base >> 4) & 1u;
        int4 wv = *reinterpret_cast<const int4*>(row + (size_t)blk * 16u);
        const uint32_t* w32 = reinterpret_cast<const uint32_t*>(&wv);
        float s = 0.0f;
        #pragma unroll
        for (uint32_t q = 0; q < 4u; ++q) {
            const uint32_t v = (half ? (w32[q] >> 4) : w32[q]) & 0x0F0F0F0Fu;
            const uint32_t mag = v & 0x07070707u;
            const uint32_t t = (mag | (mag >> 4)) & 0x00FF00FFu;
            const uint32_t sel = (t | (t >> 8)) & 0xFFFFu;
            uint32_t e4 = __byte_perm(T0, T1, sel) | ((v & 0x08080808u) << 4);
            const __nv_fp8_e4m3* eb = reinterpret_cast<const __nv_fp8_e4m3*>(&e4);
            const float4 xv = *reinterpret_cast<const float4*>(x + base + q * 4u);
            s += (float)eb[0] * xv.x + (float)eb[1] * xv.y
               + (float)eb[2] * xv.z + (float)eb[3] * xv.w;
        }
        acc += fp4gs[blk] * s;
    }
    for (uint32_t s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, s);
    uint32_t warp = tid >> 5, lane = tid & 31u;
    if (lane == 0) wsum[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        float v = 0.0f;
        uint32_t nwarps = (nth + 31u) >> 5;
        for (uint32_t w = 0; w < nwarps; ++w) v += wsum[w];
        if (bias) v += bias[o];
        y[o] = v;
    }
#else
    (void)data; (void)scale; (void)bias; (void)x; (void)y; (void)in_dim; (void)out_dim;
#endif
}


// bf16-out launcher: tc5 route on sm_100 (P74) or the tma_kt route (the
// serving prefill class on sm_120): same contract as pd_f8_gemm_w8 but y is
// written bf16. Returns cudaErrorNotSupported when neither route covers the
// call - callers keep the f32 chain there (the engine probes availability
// once at plane build and gates o16 to wave-size passes on sm_100).
PD_EXPORT
int pd_f8_gemm_w8_o16(const void* data, const void* scale, const void* xq,
                      const void* xs, void* y, uint32_t in_dim, uint32_t out_dim,
                      uint32_t batch, void* stream) {
#ifndef PD_BS_HOST
    (void)data; (void)scale; (void)xq; (void)xs; (void)y; (void)in_dim;
    (void)out_dim; (void)batch; (void)stream;
    return cudaErrorNotSupported;
#else
    if (out_dim == 0 || batch == 0) return 0;
    if ((in_dim & 31u) != 0) return cudaErrorInvalidValue;
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 900)
    // tc5 bf16-store route - on B200 the TMA lane below is process-
    // killed (load.rs SM100_SET_KILL: it loses to tc5), so the o16 chain's
    // GEMM half runs the tc5s/tc5v twins instead. Mirrors pd_f8_gemm_w8's
    // election (incl. the S4 ring rule); covers batch >= 256 - the wave
    // passes' floor. Kill: PADDOCK_NO_TC5 (shared with the f32 lane).
    // VERDICT (serve ABBA + census): a WASH - tc5s pays ~+10%
    // for the 2B stores (the muse f16 STORE-POISON class; the "epilogue
    // fully hidden" law holds for 4B stores only) and the b16 consumers
    // hold no net win. Kept as parity-gated probe infra (pfshape mode 3);
    // the engine's PADDOCK_QWEN35_O16_TC5 opt-in is the only caller.
    static const bool tc5o = [] {
        int dev = 0, cc = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&cc, cudaDevAttrComputeCapabilityMajor, dev);
        return cc == 10 && pd_tmap_encode() != nullptr
            && pd_env("PADDOCK_NO_TC5") == nullptr;
    }();
    if (tc5o && (in_dim & 127u) == 0u && batch >= 256u) {
        CUtensorMap wm, ym;
        if (pd_tmap_2d(&wm, data, in_dim, out_dim) &&
            pd_tmap_2d(&ym, xq, in_dim, batch)) {
            static const bool no_s5 = pd_env("PADDOCK_NO_TC5S") != nullptr;
            if (!no_s5 && batch >= 768u) {
                static const uint32_t sgrid = [] {
                    int dev = 0, nn = 0;
                    cudaGetDevice(&dev);
                    cudaDeviceGetAttribute(&nn, cudaDevAttrMultiProcessorCount, dev);
                    return (uint32_t)(nn & ~1);
                }();
                const uint32_t pairs = (((out_dim + 127u) >> 7) + 1u) >> 1;
                const uint32_t tcnt = pairs * ((batch + 255u) >> 8);
                const uint32_t ncl = sgrid >> 1;
                const uint32_t sper = (tcnt + ncl - 1u) / ncl;
                if (sgrid >= 4u && tcnt >= ncl
                    && (uint64_t)sper * ncl * 10u <= (uint64_t)tcnt * 13u) {
                    static const bool s4_all = pd_env("PADDOCK_TC5S_S4") != nullptr;
                    static const bool no_s4 = pd_env("PADDOCK_NO_TC5S_S4") != nullptr;
                    const bool s4 = !no_s4 && (s4_all || out_dim <= 17408u);
                    constexpr uint32_t S6 = 6u, S4 = 4u;
                    const uint32_t ssm6 = 2u * S6 * 16384u + 2u * S6 * 1024u
                        + (3u * S6 + 4u) * 8u + 64u;
                    const uint32_t ssm4 = 2u * S4 * 16384u + 2u * S4 * 1024u
                        + (3u * S4 + 4u) * 8u + 64u;
                    static bool so16 = false;
                    if (!so16) {
                        cudaFuncSetAttribute((const void*)pd_f8bs_gemm_tc5s_kt<S6, false, true>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)ssm6);
                        cudaFuncSetAttribute((const void*)pd_f8bs_gemm_tc5s_kt<S6, false, true>,
                            cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
                        cudaFuncSetAttribute((const void*)pd_f8bs_gemm_tc5s_kt<S4, false, true>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)ssm4);
                        cudaFuncSetAttribute((const void*)pd_f8bs_gemm_tc5s_kt<S4, false, true>,
                            cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
                        so16 = true;
                    }
                    if (s4)
                        pd_f8bs_gemm_tc5s_kt<S4, false, true><<<sgrid, 320, ssm4, (cudaStream_t)stream>>>(
                            wm, ym, (const unsigned char*)scale, (const unsigned char*)xs,
                            (float*)y, in_dim, out_dim, batch);
                    else
                        pd_f8bs_gemm_tc5s_kt<S6, false, true><<<sgrid, 320, ssm6, (cudaStream_t)stream>>>(
                            wm, ym, (const unsigned char*)scale, (const unsigned char*)xs,
                            (float*)y, in_dim, out_dim, batch);
                    return pd_launch_status();
                }
            }
            static const bool no_v16 = pd_env("PADDOCK_NO_TC5V") != nullptr;
            if (!no_v16) {
                const bool nt3 = (batch >= 1536u
                                  && !(out_dim > in_dim && out_dim <= 18432u))
                    || (batch >= 1024u && out_dim <= in_dim);
                if (nt3) {
                    constexpr uint32_t VS = 3u, NT = 3u;
                    const uint32_t smem = (1u + NT) * VS * 16384u
                        + VS * (512u + NT * 512u) + 2u * VS * 8u;
                    static bool vo3 = false;
                    if (!vo3) {
                        cudaFuncSetAttribute((const void*)pd_f8bs_gemm_tc5v_kt<VS, NT, true>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
                        vo3 = true;
                    }
                    const uint32_t bp = (batch + NT * 128u - 1u) / (NT * 128u) * (NT * 128u);
                    const uint32_t nt = ((out_dim + 127u) / 128u) * (bp / (NT * 128u));
                    pd_f8bs_gemm_tc5v_kt<VS, NT, true><<<nt, 160, smem, (cudaStream_t)stream>>>(
                        wm, ym, (const unsigned char*)scale, (const unsigned char*)xs,
                        (float*)y, in_dim, out_dim, batch);
                    return pd_launch_status();
                }
                constexpr uint32_t VS = 3u, NT = 2u;
                const uint32_t smem = (1u + NT) * VS * 16384u
                    + VS * (512u + NT * 512u) + 2u * VS * 8u;
                static bool vo2 = false;
                if (!vo2) {
                    cudaFuncSetAttribute((const void*)pd_f8bs_gemm_tc5v_kt<VS, NT, true>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
                    vo2 = true;
                }
                const uint32_t bp = (batch + NT * 128u - 1u) / (NT * 128u) * (NT * 128u);
                const uint32_t nt = ((out_dim + 127u) / 128u) * (bp / (NT * 128u));
                pd_f8bs_gemm_tc5v_kt<VS, NT, true><<<nt, 160, smem, (cudaStream_t)stream>>>(
                    wm, ym, (const unsigned char*)scale, (const unsigned char*)xs,
                    (float*)y, in_dim, out_dim, batch);
                return pd_launch_status();
            }
        }
        return cudaErrorNotSupported;
    }
    if (tc5o) return cudaErrorNotSupported;
#endif
    static const bool tma = [] {
        int dev = 0, cma = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&cma, cudaDevAttrComputeCapabilityMajor, dev);
        return cma >= 9 && pd_tmap_encode() != nullptr
            && pd_env("PADDOCK_NO_F8W8_TMA") == nullptr;
    }();
    if (!tma || (in_dim & 127u) != 0u) return cudaErrorNotSupported;
    const uint32_t smem = 67600u;
    static bool ao16 = false;
    if (!ao16) {
        cudaFuncSetAttribute((const void*)pd_f8_gemm_w8_tma_kt<false, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        ao16 = true;
    }
    CUtensorMap wm, ym;
    if (!pd_tmap_2d(&wm, data, in_dim, out_dim) || !pd_tmap_2d(&ym, xq, in_dim, batch))
        return cudaErrorNotSupported;
    const uint32_t bp = (batch + 127u) & ~127u;
    const uint32_t nt = ((out_dim + 127u) / 128u) * (bp >> 7);
    pd_f8_gemm_w8_tma_kt<false, true><<<nt, 384, smem, (cudaStream_t)stream>>>(
        wm, ym, (const unsigned char*)scale, (const unsigned char*)xs,
        (float*)y, in_dim, out_dim, batch);
    return pd_launch_status();
#endif
}

PD_EXPORT
int pd_fp4_gemv(const void* data, const void* scale, const void* bias,
                const void* x, void* y, uint32_t in_dim, uint32_t out_dim,
                void* stream) {
    if (out_dim == 0) return 0;
    if ((in_dim & 31u) != 0) return cudaErrorInvalidValue;
    const uint32_t threads = 256u;
    const uint32_t shmem = (in_dim >> 5) * 4u;
    pd_fp4_gemv_kernel<<<out_dim, threads, shmem, (cudaStream_t)stream>>>(
        (const uint8_t*)data, (const uint8_t*)scale, (const float*)bias,
        (const float*)x, (float*)y, in_dim, out_dim);
    return pd_launch_status();
}

__global__ void pd_f8_gemv_kernel(
    const uint8_t* __restrict__ data, const uint8_t* __restrict__ scale,
    const float* __restrict__ bias, const float* __restrict__ x, float* __restrict__ y,
    uint32_t in_dim, uint32_t out_dim) {
#if PD_F8W8_OK
    uint32_t o = blockIdx.x;
    if (o >= out_dim) return;
    uint32_t tid = threadIdx.x, nth = blockDim.x;
    uint32_t n_blocks = in_dim >> 5;
    extern __shared__ float f8gs[];
    const uint8_t* srow = scale + (size_t)o * n_blocks;
    for (uint32_t b = tid; b < n_blocks; b += nth)
        f8gs[b] = __int_as_float(((int)srow[b]) << 23);
    __shared__ float wsum[32];
    __syncthreads();
    const uint8_t* row = data + (size_t)o * in_dim;
    float acc = 0.0f;
    for (uint32_t base = tid * 16u; base < in_dim; base += nth * 16u) {
        int4 wv = *reinterpret_cast<const int4*>(row + base);
        const __nv_fp8_e4m3* wb = reinterpret_cast<const __nv_fp8_e4m3*>(&wv);
        float4 x0 = *reinterpret_cast<const float4*>(x + base);
        float4 x1 = *reinterpret_cast<const float4*>(x + base + 4);
        float4 x2 = *reinterpret_cast<const float4*>(x + base + 8);
        float4 x3 = *reinterpret_cast<const float4*>(x + base + 12);
        float s = (float)wb[0] * x0.x + (float)wb[1] * x0.y + (float)wb[2] * x0.z + (float)wb[3] * x0.w
                + (float)wb[4] * x1.x + (float)wb[5] * x1.y + (float)wb[6] * x1.z + (float)wb[7] * x1.w
                + (float)wb[8] * x2.x + (float)wb[9] * x2.y + (float)wb[10] * x2.z + (float)wb[11] * x2.w
                + (float)wb[12] * x3.x + (float)wb[13] * x3.y + (float)wb[14] * x3.z + (float)wb[15] * x3.w;
        acc += f8gs[base >> 5] * s;
    }
    for (uint32_t s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, s);
    uint32_t warp = tid >> 5, lane = tid & 31u;
    if (lane == 0) wsum[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        float v = 0.0f;
        uint32_t nwarps = (nth + 31u) >> 5;
        for (uint32_t w = 0; w < nwarps; ++w) v += wsum[w];
        if (bias) v += bias[o];
        y[o] = v;
    }
#else
    (void)data; (void)scale; (void)bias; (void)x; (void)y; (void)in_dim; (void)out_dim;
#endif
}

PD_EXPORT
int pd_f8_gemv(const void* data, const void* scale, const void* bias,
               const void* x, void* y, uint32_t in_dim, uint32_t out_dim,
               void* stream) {
    if (out_dim == 0) return 0;
    uint32_t threads = 256;
    uint32_t shmem = (in_dim >> 5) * sizeof(float);
    pd_f8_gemv_kernel<<<out_dim, threads, shmem, (cudaStream_t)stream>>>(
        (const uint8_t*)data, (const uint8_t*)scale, (const float*)bias,
        (const float*)x, (float*)y, in_dim, out_dim);
    return pd_launch_status();
}

// Batched twin (2..16 rows): weights read once per block, one accumulator
// per row - covers the band between the gemv (b1) and the TMA GEMM (>=32,
// where its 128-col tile amortizes). x rows read direct from L2 (tiny, hot).
__global__ void pd_f8_gemv_batch_kernel(
    const uint8_t* __restrict__ data, const uint8_t* __restrict__ scale,
    const float* __restrict__ x, float* __restrict__ y,
    uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
#if PD_F8W8_OK
    uint32_t o = blockIdx.x;
    if (o >= out_dim) return;
    uint32_t tid = threadIdx.x, nth = blockDim.x;
    uint32_t n_blocks = in_dim >> 5;
    extern __shared__ float f8gbs[];
    const uint8_t* srow = scale + (size_t)o * n_blocks;
    for (uint32_t b = tid; b < n_blocks; b += nth)
        f8gbs[b] = __int_as_float(((int)srow[b]) << 23);
    __shared__ float wsum[32][16];
    __syncthreads();
    const uint8_t* row = data + (size_t)o * in_dim;
    float acc[16];
    #pragma unroll
    for (uint32_t r = 0; r < 16u; ++r) acc[r] = 0.f;
    for (uint32_t base = tid * 16u; base < in_dim; base += nth * 16u) {
        int4 wv = *reinterpret_cast<const int4*>(row + base);
        const __nv_fp8_e4m3* wb = reinterpret_cast<const __nv_fp8_e4m3*>(&wv);
        float wf[16];
        #pragma unroll
        for (uint32_t j = 0; j < 16u; ++j) wf[j] = (float)wb[j];
        const float sc = f8gbs[base >> 5];
        #pragma unroll
        for (uint32_t r = 0; r < 16u; ++r) {
            if (r >= batch) break;
            const float* xr = x + (size_t)r * in_dim + base;
            float4 x0 = *reinterpret_cast<const float4*>(xr);
            float4 x1 = *reinterpret_cast<const float4*>(xr + 4);
            float4 x2 = *reinterpret_cast<const float4*>(xr + 8);
            float4 x3 = *reinterpret_cast<const float4*>(xr + 12);
            float s = wf[0]*x0.x + wf[1]*x0.y + wf[2]*x0.z + wf[3]*x0.w
                    + wf[4]*x1.x + wf[5]*x1.y + wf[6]*x1.z + wf[7]*x1.w
                    + wf[8]*x2.x + wf[9]*x2.y + wf[10]*x2.z + wf[11]*x2.w
                    + wf[12]*x3.x + wf[13]*x3.y + wf[14]*x3.z + wf[15]*x3.w;
            acc[r] += sc * s;
        }
    }
    uint32_t warp = tid >> 5, lane = tid & 31u;
    #pragma unroll
    for (uint32_t r = 0; r < 16u; ++r) {
        if (r >= batch) break;
        float a = acc[r];
        for (uint32_t s = 16; s > 0; s >>= 1) a += __shfl_down_sync(0xffffffffu, a, s);
        if (lane == 0) wsum[warp][r] = a;
    }
    __syncthreads();
    if (tid < batch) {
        float v = 0.0f;
        uint32_t nwarps = (nth + 31u) >> 5;
        for (uint32_t w = 0; w < nwarps; ++w) v += wsum[w][tid];
        y[(size_t)tid * out_dim + o] = v;
    }
#else
    (void)data; (void)scale; (void)x; (void)y; (void)in_dim; (void)out_dim; (void)batch;
#endif
}

PD_EXPORT
int pd_f8_gemv_batch(const void* data, const void* scale, const void* x, void* y,
                     uint32_t in_dim, uint32_t out_dim, uint32_t batch, void* stream) {
    if (out_dim == 0 || batch == 0) return 0;
    if (batch > 16u) return cudaErrorInvalidValue;
    uint32_t shmem = (in_dim >> 5) * sizeof(float);
    pd_f8_gemv_batch_kernel<<<out_dim, 256, shmem, (cudaStream_t)stream>>>(
        (const uint8_t*)data, (const uint8_t*)scale, (const float*)x, (float*)y,
        in_dim, out_dim, batch);
    return pd_launch_status();
}

// ---- f8 mma_ks twin (the 4..31-row verify-band closer) ---------------------
// Structural clone of pd_q8_0_gemm_mma_kernel (gemm/int8_mma.cuh): same
// BM x BN shared tile, same PD_MMA_KT K-staging with the ST=2 cp.async
// double-buffer, same m16n8k32 fragment maps (e4m3 bytes sit in the exact
// byte positions s8 does) - but the per-32 ue8m0 scales ride the mxf8f6f4
// block-scale MMA in HARDWARE (pd_bs_mma_w8_kb), so the q8 kernel's fold
// epilogue (2 half converts + 2 f32 reads + 4 FMAs per mma) vanishes.
// Scale staging is cheaper too: a K-stage's 4 ue8m0 bytes are CONTIGUOUS in
// the plane -> one 4B word per row (q8 stages 2x4B half-pairs + 4B f32s).
// Scale-lane distribution is the TMA kernel's proven map: the A word comes
// from row (t&1 ? wr+8+g : wr+g), the B word from col csub+g, and the KB
// template byte-selects the sub-block. Purpose: the spec-verify FFN band
// (r=4..31) where the 128-col TMA tile pays ~2x vs this shape; from r=32 up
// the TMA GEMM already beats q8. K-split partial planes + fixed-z combine
// are shared with the q8 ks family (the combine kernel is format-blind).
// Same k-ascending chained-mma order as pd_f8_gemm_w8_kt -> bit-equal to it
// per element at nz==1; nz>1 regroups f32 partials (the ks class).
template <uint32_t BM, uint32_t BN, uint32_t NWARP, uint32_t ST = 1u>
__global__ void __launch_bounds__(NWARP * 32) pd_f8_gemm_mma_kernel(
        const uint8_t* __restrict__ data, const uint8_t* __restrict__ scale,
        const uint8_t* __restrict__ xq, const uint8_t* __restrict__ xs,
        float* __restrict__ y, uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
#if PD_F8W8_OK
    constexpr uint32_t NTH = NWARP * 32u;
    constexpr uint32_t WR = BM / 16u;          // warp rows (16 out-rows each)
    constexpr uint32_t WC = NWARP / WR;        // warp cols
    constexpr uint32_t CPW = BN / WC;          // cols per warp
    constexpr uint32_t NSUB = CPW / 8u;        // 16x8 col sub-tiles per warp
    constexpr uint32_t I4PR = PD_MMA_KT / 16u; // int4 loads per staged row
    static_assert(WR * WC == NWARP, "warp grid");
    static_assert(NSUB * 8u * WC == BN, "col cover");
    static_assert(ST == 1u || ST == 2u, "stage count");

    __shared__ __align__(16) uint8_t sh_a[ST][BM * PD_MMA_KPAD];
    __shared__ __align__(16) uint8_t sh_b[ST][BN * PD_MMA_KPAD];
    // one ue8m0 word (4 sub-block scales) per row/col per stage
    __shared__ __align__(4) uint32_t sh_ws[ST][BM];
    __shared__ __align__(4) uint32_t sh_xs[ST][BN];

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, t = lane & 3u;
    const uint32_t wr = (warp % WR) * 16u;
    const uint32_t wc = (warp / WR) * CPW;
    const uint32_t row_base = blockIdx.x * BM;
    const uint32_t col_base = blockIdx.y * BN;
    const uint32_t n_blocks = in_dim >> 5;

    // K-split: identical slicing to the q8 twin (per rounded to NSUBK, so a
    // slice's scale words stay 4B-aligned under the pipelined staging)
    const uint32_t nz = gridDim.z;
    uint32_t kt_lo = 0, kt_hi = n_blocks;
    if (nz > 1u) {
        const uint32_t per = ((n_blocks + nz - 1u) / nz + PD_MMA_NSUBK - 1u) /
                             PD_MMA_NSUBK * PD_MMA_NSUBK;
        kt_lo = blockIdx.z * per;
        kt_hi = kt_lo + per < n_blocks ? kt_lo + per : n_blocks;
        y += (size_t)blockIdx.z * out_dim * batch;
    }

    auto stage = [&](uint32_t kt, uint32_t buf) {
        #pragma unroll
        for (uint32_t i = tid; i < BM * I4PR; i += NTH) {
            uint32_t row = i / I4PR, k16 = (i % I4PR) * 16u, gk = kt * 32u + k16;
            const bool ok = gk < in_dim && (row_base + row) < out_dim;
            const uint8_t* src = data + (size_t)(row_base + row) * in_dim + gk;
            if (ST == 2u) {
                pd_mma_cpa16p(&sh_a[buf][row * PD_MMA_KPAD + k16], src, ok);
            } else {
                *reinterpret_cast<int4*>(&sh_a[buf][row * PD_MMA_KPAD + k16]) =
                    ok ? *reinterpret_cast<const int4*>(src) : make_int4(0, 0, 0, 0);
            }
        }
        #pragma unroll
        for (uint32_t i = tid; i < BN * I4PR; i += NTH) {
            uint32_t col = i / I4PR, k16 = (i % I4PR) * 16u, gk = kt * 32u + k16;
            const bool ok = gk < in_dim && (col_base + col) < batch;
            const uint8_t* src = xq + (size_t)(col_base + col) * in_dim + gk;
            if (ST == 2u) {
                pd_mma_cpa16p(&sh_b[buf][col * PD_MMA_KPAD + k16], src, ok);
            } else {
                *reinterpret_cast<int4*>(&sh_b[buf][col * PD_MMA_KPAD + k16]) =
                    ok ? *reinterpret_cast<const int4*>(src) : make_int4(0, 0, 0, 0);
            }
        }
        if (ST == 2u) {
            // whole-word scale copies (launcher enforces n_blocks % 4 == 0 for
            // ST=2, so kt..kt+3 is one aligned in-bounds word). Zero-filled
            // OOB words decode to 2^-127 against zero-filled data tiles -> 0.
            for (uint32_t i = tid; i < BM; i += NTH) {
                const bool ok = (row_base + i) < out_dim && kt < n_blocks;
                pd_mma_cpa4p(&sh_ws[buf][i],
                             scale + (size_t)(row_base + i) * n_blocks + kt, ok);
            }
            for (uint32_t i = tid; i < BN; i += NTH) {
                const bool ok = (col_base + i) < batch && kt < n_blocks;
                pd_mma_cpa4p(&sh_xs[buf][i],
                             xs + (size_t)(col_base + i) * n_blocks + kt, ok);
            }
        } else {
            // per-byte sync loads (odd n_blocks tails land here)
            for (uint32_t i = tid; i < BM * PD_MMA_NSUBK; i += NTH) {
                uint32_t row = i >> 2, sb = i & 3u;
                ((uint8_t*)&sh_ws[buf][row])[sb] =
                    ((kt + sb) < n_blocks && (row_base + row) < out_dim)
                        ? scale[(size_t)(row_base + row) * n_blocks + kt + sb] : 0u;
            }
            for (uint32_t i = tid; i < BN * PD_MMA_NSUBK; i += NTH) {
                uint32_t col = i >> 2, sb = i & 3u;
                ((uint8_t*)&sh_xs[buf][col])[sb] =
                    ((kt + sb) < n_blocks && (col_base + col) < batch)
                        ? xs[(size_t)(col_base + col) * n_blocks + kt + sb] : 0u;
            }
        }
    };

    float acc[NSUB][4] = {};
    auto compute = [&](uint32_t buf) {
#if PD_BS_OK
        // A scales: even-t lanes feed row wr+g, odd-t row wr+8+g (hw map)
        const uint32_t sa = sh_ws[buf][(t & 1u) ? wr + 8u + g : wr + g];
        uint32_t sbw[NSUB];
        #pragma unroll
        for (uint32_t sub = 0; sub < NSUB; ++sub)
            sbw[sub] = sh_xs[buf][wc + sub * 8u + g];
#endif
        // the KB byte-select is a PTX immediate -> the 4 sub-blocks unroll
        // as explicit template instantiations. Non-120a passes take the sw
        // fold, whose four scale bytes come off this thread's own quad rows/
        // cols (see pd_f8_mma_sw - no hw lane routing to imitate).
        #define PD_F8KS_SB(SB)                                                     \
        {                                                                          \
            const uint32_t ko = SB * 32u;                                          \
            uint32_t a0 = *reinterpret_cast<const uint32_t*>(                      \
                &sh_a[buf][(wr + g) * PD_MMA_KPAD + ko + t * 4u]);                 \
            uint32_t a1 = *reinterpret_cast<const uint32_t*>(                      \
                &sh_a[buf][(wr + 8u + g) * PD_MMA_KPAD + ko + t * 4u]);            \
            uint32_t a2 = *reinterpret_cast<const uint32_t*>(                      \
                &sh_a[buf][(wr + g) * PD_MMA_KPAD + ko + 16u + t * 4u]);           \
            uint32_t a3 = *reinterpret_cast<const uint32_t*>(                      \
                &sh_a[buf][(wr + 8u + g) * PD_MMA_KPAD + ko + 16u + t * 4u]);      \
            _Pragma("unroll")                                                      \
            for (uint32_t sub = 0; sub < NSUB; ++sub) {                            \
                const uint32_t csub = wc + sub * 8u;                               \
                uint32_t b0 = *reinterpret_cast<const uint32_t*>(                  \
                    &sh_b[buf][(csub + g) * PD_MMA_KPAD + ko + t * 4u]);           \
                uint32_t b1 = *reinterpret_cast<const uint32_t*>(                  \
                    &sh_b[buf][(csub + g) * PD_MMA_KPAD + ko + 16u + t * 4u]);     \
                PD_F8KS_MMA(SB, sub, a0, a1, a2, a3, b0, b1)                       \
            }                                                                      \
        }
#if PD_BS_OK
        #define PD_F8KS_MMA(SB, sub, a0, a1, a2, a3, b0, b1)                       \
            pd_bs_mma_w8_kb<SB>(acc[sub], a0, a1, a2, a3, b0, b1, sa, sbw[sub]);
#else
        #define PD_F8KS_MMA(SB, sub, a0, a1, a2, a3, b0, b1)                       \
            pd_f8_mma_sw(acc[sub], a0, a1, a2, a3, b0, b1,                         \
                ((const uint8_t*)&sh_ws[buf][wr + g])[SB],                         \
                ((const uint8_t*)&sh_ws[buf][wr + 8u + g])[SB],                    \
                ((const uint8_t*)&sh_xs[buf][wc + (sub) * 8u + 2u * t])[SB],       \
                ((const uint8_t*)&sh_xs[buf][wc + (sub) * 8u + 2u * t + 1u])[SB]);
#endif
        PD_F8KS_SB(0) PD_F8KS_SB(1) PD_F8KS_SB(2) PD_F8KS_SB(3)
        #undef PD_F8KS_MMA
        #undef PD_F8KS_SB
    };

    if (ST == 2u && kt_lo < kt_hi) {
        stage(kt_lo, 0);
        pd_attn_cpa_commit();
        uint32_t p = 0;
        for (uint32_t kt = kt_lo; kt < kt_hi; kt += PD_MMA_NSUBK) {
            const uint32_t nxt = kt + PD_MMA_NSUBK;
            if (nxt < kt_hi) {
                stage(nxt, p ^ 1u);
                pd_attn_cpa_commit();
                pd_attn_cpa_wait1();
            } else {
                pd_attn_cpa_wait0();
            }
            __syncthreads();
            compute(p);
            __syncthreads();
            p ^= 1u;
        }
    } else {
        for (uint32_t kt = kt_lo; kt < kt_hi; kt += PD_MMA_NSUBK) {
            stage(kt, 0);
            __syncthreads();
            compute(0);
            __syncthreads();
        }
    }

    const uint32_t r0 = row_base + wr + g, r8 = row_base + wr + 8u + g;
    #pragma unroll
    for (uint32_t sub = 0; sub < NSUB; ++sub) {
        const uint32_t c0 = col_base + wc + sub * 8u + 2u * t;
        const uint32_t c1 = c0 + 1u;
        if (r0 < out_dim) {
            if (c0 < batch) y[(size_t)c0 * out_dim + r0] = acc[sub][0];
            if (c1 < batch) y[(size_t)c1 * out_dim + r0] = acc[sub][1];
        }
        if (r8 < out_dim) {
            if (c0 < batch) y[(size_t)c0 * out_dim + r8] = acc[sub][2];
            if (c1 < batch) y[(size_t)c1 * out_dim + r8] = acc[sub][3];
        }
    }
#else
    (void)data; (void)scale; (void)xq; (void)xs; (void)y;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}

// ---- fp4 mma_ks twin ------------------------------------------------------
// pd_f8_gemm_mma_kernel with PACKED e2m1 A: the split nibble order maps
// PERFECTLY onto the m16n8k32 fragment shape - one u32 of packed bytes at
// (row, SB*16 + t*4) holds elems t*4..+3 in its LOW nibbles (the k-lo
// fragment half) and 16+t*4..+3 in its HIGH nibbles (k-hi), so the nibble
// expand yields a0..a3 from half the smem loads (2 u32/row vs 4). A-side
// staging halves too (64B packed per K-128 stage row, padded to 80 for
// 16B cp.async alignment). B (e4m3 activations) + scale staging identical;
// mma = pd_bs_mma_kb (e2m1.e4m3). sm_120a-only (no sw-fold twin: the fp4
// class is gated to the hw block-scale path).
#define PD_FP4_KPADP 80u
template <uint32_t BM, uint32_t BN, uint32_t NWARP, uint32_t ST = 1u>
__global__ void __launch_bounds__(NWARP * 32) pd_fp4_gemm_mma_kernel(
        const uint8_t* __restrict__ data, const uint8_t* __restrict__ scale,
        const uint8_t* __restrict__ xq, const uint8_t* __restrict__ xs,
        float* __restrict__ y, uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
#if PD_BS_OK
    constexpr uint32_t NTH = NWARP * 32u;
    constexpr uint32_t WR = BM / 16u;
    constexpr uint32_t WC = NWARP / WR;
    constexpr uint32_t CPW = BN / WC;
    constexpr uint32_t NSUB = CPW / 8u;
    constexpr uint32_t I4PR = PD_MMA_KT / 16u;   // B-side int4 loads/row
    constexpr uint32_t I4PRP = I4PR / 2u;        // packed A-side
    static_assert(WR * WC == NWARP, "warp grid");
    static_assert(NSUB * 8u * WC == BN, "col cover");
    static_assert(ST == 1u || ST == 2u, "stage count");

    __shared__ __align__(16) uint8_t sh_a[ST][BM * PD_FP4_KPADP];
    __shared__ __align__(16) uint8_t sh_b[ST][BN * PD_MMA_KPAD];
    __shared__ __align__(4) uint32_t sh_ws[ST][BM];
    __shared__ __align__(4) uint32_t sh_xs[ST][BN];

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, t = lane & 3u;
    const uint32_t wr = (warp % WR) * 16u;
    const uint32_t wc = (warp / WR) * CPW;
    const uint32_t row_base = blockIdx.x * BM;
    const uint32_t col_base = blockIdx.y * BN;
    const uint32_t n_blocks = in_dim >> 5;

    const uint32_t nz = gridDim.z;
    uint32_t kt_lo = 0, kt_hi = n_blocks;
    if (nz > 1u) {
        const uint32_t per = ((n_blocks + nz - 1u) / nz + PD_MMA_NSUBK - 1u) /
                             PD_MMA_NSUBK * PD_MMA_NSUBK;
        kt_lo = blockIdx.z * per;
        kt_hi = kt_lo + per < n_blocks ? kt_lo + per : n_blocks;
        y += (size_t)blockIdx.z * out_dim * batch;
    }

    auto stage = [&](uint32_t kt, uint32_t buf) {
        #pragma unroll
        for (uint32_t i = tid; i < BM * I4PRP; i += NTH) {
            uint32_t row = i / I4PRP, k16 = (i % I4PRP) * 16u;
            const uint32_t gk = kt * 32u + k16 * 2u;   // element index
            const bool ok = gk < in_dim && (row_base + row) < out_dim;
            const uint8_t* src = data + (size_t)(row_base + row) * (in_dim >> 1)
                               + kt * 16u + k16;
            if (ST == 2u) {
                pd_mma_cpa16p(&sh_a[buf][row * PD_FP4_KPADP + k16], src, ok);
            } else {
                *reinterpret_cast<int4*>(&sh_a[buf][row * PD_FP4_KPADP + k16]) =
                    ok ? *reinterpret_cast<const int4*>(src) : make_int4(0, 0, 0, 0);
            }
        }
        #pragma unroll
        for (uint32_t i = tid; i < BN * I4PR; i += NTH) {
            uint32_t col = i / I4PR, k16 = (i % I4PR) * 16u, gk = kt * 32u + k16;
            const bool ok = gk < in_dim && (col_base + col) < batch;
            const uint8_t* src = xq + (size_t)(col_base + col) * in_dim + gk;
            if (ST == 2u) {
                pd_mma_cpa16p(&sh_b[buf][col * PD_MMA_KPAD + k16], src, ok);
            } else {
                *reinterpret_cast<int4*>(&sh_b[buf][col * PD_MMA_KPAD + k16]) =
                    ok ? *reinterpret_cast<const int4*>(src) : make_int4(0, 0, 0, 0);
            }
        }
        if (ST == 2u) {
            for (uint32_t i = tid; i < BM; i += NTH) {
                const bool ok = (row_base + i) < out_dim && kt < n_blocks;
                pd_mma_cpa4p(&sh_ws[buf][i],
                             scale + (size_t)(row_base + i) * n_blocks + kt, ok);
            }
            for (uint32_t i = tid; i < BN; i += NTH) {
                const bool ok = (col_base + i) < batch && kt < n_blocks;
                pd_mma_cpa4p(&sh_xs[buf][i],
                             xs + (size_t)(col_base + i) * n_blocks + kt, ok);
            }
        } else {
            for (uint32_t i = tid; i < BM * PD_MMA_NSUBK; i += NTH) {
                uint32_t row = i >> 2, sb = i & 3u;
                ((uint8_t*)&sh_ws[buf][row])[sb] =
                    ((kt + sb) < n_blocks && (row_base + row) < out_dim)
                        ? scale[(size_t)(row_base + row) * n_blocks + kt + sb] : 0u;
            }
            for (uint32_t i = tid; i < BN * PD_MMA_NSUBK; i += NTH) {
                uint32_t col = i >> 2, sb = i & 3u;
                ((uint8_t*)&sh_xs[buf][col])[sb] =
                    ((kt + sb) < n_blocks && (col_base + col) < batch)
                        ? xs[(size_t)(col_base + col) * n_blocks + kt + sb] : 0u;
            }
        }
    };

    float acc[NSUB][4] = {};
    auto compute = [&](uint32_t buf) {
        const uint32_t sa = sh_ws[buf][(t & 1u) ? wr + 8u + g : wr + g];
        uint32_t sbw[NSUB];
        #pragma unroll
        for (uint32_t sub = 0; sub < NSUB; ++sub)
            sbw[sub] = sh_xs[buf][wc + sub * 8u + g];
        #define PD_FP4KS_SB(SB)                                                            {                                                                                      const uint32_t kp = SB * 16u + t * 4u;                                             const uint32_t p0 = *reinterpret_cast<const uint32_t*>(                                &sh_a[buf][(wr + g) * PD_FP4_KPADP + kp]);                                     const uint32_t p8 = *reinterpret_cast<const uint32_t*>(                                &sh_a[buf][(wr + 8u + g) * PD_FP4_KPADP + kp]);                                const uint32_t a0 = (p0 & 0x0F0F0F0Fu) << 2;                                       const uint32_t a1 = (p8 & 0x0F0F0F0Fu) << 2;                                       const uint32_t a2 = (p0 & 0xF0F0F0F0u) >> 2;                                       const uint32_t a3 = (p8 & 0xF0F0F0F0u) >> 2;                                       const uint32_t ko = SB * 32u;                                                      _Pragma("unroll")                                                                  for (uint32_t sub = 0; sub < NSUB; ++sub) {                                            const uint32_t csub = wc + sub * 8u;                                               uint32_t b0 = *reinterpret_cast<const uint32_t*>(                                      &sh_b[buf][(csub + g) * PD_MMA_KPAD + ko + t * 4u]);                           uint32_t b1 = *reinterpret_cast<const uint32_t*>(                                      &sh_b[buf][(csub + g) * PD_MMA_KPAD + ko + 16u + t * 4u]);                     pd_bs_mma_kb<SB>(acc[sub], a0, a1, a2, a3, b0, b1, sa, sbw[sub]);              }                                                                              }
        PD_FP4KS_SB(0) PD_FP4KS_SB(1) PD_FP4KS_SB(2) PD_FP4KS_SB(3)
        #undef PD_FP4KS_SB
    };

    if (ST == 2u && kt_lo < kt_hi) {
        stage(kt_lo, 0);
        pd_attn_cpa_commit();
        uint32_t p = 0;
        for (uint32_t kt = kt_lo; kt < kt_hi; kt += PD_MMA_NSUBK) {
            const uint32_t nxt = kt + PD_MMA_NSUBK;
            if (nxt < kt_hi) {
                stage(nxt, p ^ 1u);
                pd_attn_cpa_commit();
                pd_attn_cpa_wait1();
            } else {
                pd_attn_cpa_wait0();
            }
            __syncthreads();
            compute(p);
            __syncthreads();
            p ^= 1u;
        }
    } else {
        for (uint32_t kt = kt_lo; kt < kt_hi; kt += PD_MMA_NSUBK) {
            stage(kt, 0);
            __syncthreads();
            compute(0);
            __syncthreads();
        }
    }

    const uint32_t r0 = row_base + wr + g, r8 = row_base + wr + 8u + g;
    #pragma unroll
    for (uint32_t sub = 0; sub < NSUB; ++sub) {
        const uint32_t c0 = col_base + wc + sub * 8u + 2u * t;
        const uint32_t c1 = c0 + 1u;
        if (r0 < out_dim) {
            if (c0 < batch) y[(size_t)c0 * out_dim + r0] = acc[sub][0];
            if (c1 < batch) y[(size_t)c1 * out_dim + r0] = acc[sub][1];
        }
        if (r8 < out_dim) {
            if (c0 < batch) y[(size_t)c0 * out_dim + r8] = acc[sub][2];
            if (c1 < batch) y[(size_t)c1 * out_dim + r8] = acc[sub][3];
        }
    }
#else
    (void)data; (void)scale; (void)xq; (void)xs; (void)y;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}

PD_EXPORT
int pd_fp4_gemm_mma_ks(const void* data, const void* scale, const void* xq,
                       const void* xs, void* part, void* y, uint32_t in_dim,
                       uint32_t out_dim, uint32_t batch, void* stream) {
#if !defined(__CUDA_ARCH__)
    // host: mirror pd_f8_gemm_mma_ks's ladder / nz pick exactly
#endif
    if (out_dim == 0 || batch == 0) return 0;
    if ((out_dim & 15u) || (in_dim & 31u)) return cudaErrorInvalidValue;
    if (batch > 64u) return cudaErrorInvalidValue;
    auto st = (cudaStream_t)stream;
    static int nsm = 0;
    if (nsm == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
        if (nsm <= 0) nsm = 128;
    }
    const uint32_t tiles = (out_dim + 63u) / 64u;
    const uint32_t n_blocks = in_dim >> 5;
    uint32_t nz = ((uint32_t)nsm * 2u + tiles - 1u) / tiles;
    const uint32_t max_nz = (n_blocks + 3u) / 4u;
    if (nz > 8u) nz = 8u;
    if (nz > max_nz) nz = max_nz;
    if (nz < 1u) nz = 1u;
    float* dst = nz > 1u ? (float*)part : (float*)y;
    const bool pipe2 = (n_blocks & 3u) == 0u;
    dim3 grid(tiles, 1u, nz);
    const uint8_t* d = (const uint8_t*)data;
    const uint8_t* sc = (const uint8_t*)scale;
    const uint8_t* q = (const uint8_t*)xq;
    const uint8_t* s = (const uint8_t*)xs;
    if (batch <= 16u) {
        if (pipe2)
            pd_fp4_gemm_mma_kernel<64u, 16u, 8u, 2u><<<grid, 256, 0, st>>>(
                d, sc, q, s, dst, in_dim, out_dim, batch);
        else
            pd_fp4_gemm_mma_kernel<64u, 16u, 8u><<<grid, 256, 0, st>>>(
                d, sc, q, s, dst, in_dim, out_dim, batch);
    } else if (batch <= 32u) {
        if (pipe2)
            pd_fp4_gemm_mma_kernel<64u, 32u, 8u, 2u><<<grid, 256, 0, st>>>(
                d, sc, q, s, dst, in_dim, out_dim, batch);
        else
            pd_fp4_gemm_mma_kernel<64u, 32u, 8u><<<grid, 256, 0, st>>>(
                d, sc, q, s, dst, in_dim, out_dim, batch);
    } else {
        if (pipe2)
            pd_fp4_gemm_mma_kernel<64u, 64u, 8u, 2u><<<grid, 256, 0, st>>>(
                d, sc, q, s, dst, in_dim, out_dim, batch);
        else
            pd_fp4_gemm_mma_kernel<64u, 64u, 8u><<<grid, 256, 0, st>>>(
                d, sc, q, s, dst, in_dim, out_dim, batch);
    }
    if (nz > 1u) {
        uint32_t n = out_dim * batch;
        pd_q8_0_gemm_mma_ks_combine_kernel<<<(n + 255u) / 256u, 256, 0, st>>>(
            (const float*)part, nullptr, (float*)y, n, nz, out_dim);
    }
    return pd_launch_status();
}

// K-split launcher over the f8w planes: rung ladder + nz pick cloned from
// pd_q8_0_gemm_mma_ks_impl (BN16 <=16 / BN32 <=32 / BN64 <=64; >64 is the
// TMA GEMM's territory -> callers dispatch there). Bias-free (the gemma4
// FFN trio). Combine reuses the q8 family's format-blind partial-sum kernel.
PD_EXPORT
int pd_f8_gemm_mma_ks(const void* data, const void* scale, const void* xq,
                      const void* xs, void* part, void* y, uint32_t in_dim,
                      uint32_t out_dim, uint32_t batch, void* stream) {
    if (out_dim == 0 || batch == 0) return 0;
    if ((out_dim & 15u) || (in_dim & 31u)) return cudaErrorInvalidValue;
    if (batch > 64u) return cudaErrorInvalidValue;
    auto st = (cudaStream_t)stream;
    static int nsm = 0;
    if (nsm == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
        if (nsm <= 0) nsm = 128;
    }
    const uint32_t tiles = (out_dim + 63u) / 64u;
    const uint32_t n_blocks = in_dim >> 5;
    uint32_t nz = ((uint32_t)nsm * 2u + tiles - 1u) / tiles;
    const uint32_t max_nz = (n_blocks + 3u) / 4u;
    if (nz > 8u) nz = 8u;
    if (nz > max_nz) nz = max_nz;
    if (nz < 1u) nz = 1u;
    float* dst = nz > 1u ? (float*)part : (float*)y;
    // ST=2's whole-word scale copies need n_blocks % 4 == 0 (in_dim % 128)
    const bool pipe2 = (n_blocks & 3u) == 0u;
    dim3 grid(tiles, 1u, nz);
    const uint8_t* d = (const uint8_t*)data;
    const uint8_t* sc = (const uint8_t*)scale;
    const uint8_t* q = (const uint8_t*)xq;
    const uint8_t* s = (const uint8_t*)xs;
    if (batch <= 16u) {
        if (pipe2)
            pd_f8_gemm_mma_kernel<64u, 16u, 8u, 2u><<<grid, 256, 0, st>>>(
                d, sc, q, s, dst, in_dim, out_dim, batch);
        else
            pd_f8_gemm_mma_kernel<64u, 16u, 8u><<<grid, 256, 0, st>>>(
                d, sc, q, s, dst, in_dim, out_dim, batch);
    } else if (batch <= 32u) {
        if (pipe2)
            pd_f8_gemm_mma_kernel<64u, 32u, 8u, 2u><<<grid, 256, 0, st>>>(
                d, sc, q, s, dst, in_dim, out_dim, batch);
        else
            pd_f8_gemm_mma_kernel<64u, 32u, 8u><<<grid, 256, 0, st>>>(
                d, sc, q, s, dst, in_dim, out_dim, batch);
    } else {
        if (pipe2)
            pd_f8_gemm_mma_kernel<64u, 64u, 8u, 2u><<<grid, 256, 0, st>>>(
                d, sc, q, s, dst, in_dim, out_dim, batch);
        else
            pd_f8_gemm_mma_kernel<64u, 64u, 8u><<<grid, 256, 0, st>>>(
                d, sc, q, s, dst, in_dim, out_dim, batch);
    }
    if (nz > 1u) {
        uint32_t n = out_dim * batch;
        pd_q8_0_gemm_mma_ks_combine_kernel<<<(n + 255u) / 256u, 256, 0, st>>>(
            (const float*)part, nullptr, (float*)y, n, nz, out_dim);
    }
    return pd_launch_status();
}

// ---- per-ROW-scaled e4m3 GEMM (the sm_100 prefill class) ------------------
// The per-32 block-scale w8 GEMM needs the sm_120a hardware fold to be fast;
// the software fold (pd_f8_mma_sw) taxes every mma with scale reads + FMAs
// and regressed every measured shape on B200. This class removes
// the inner-loop fold entirely: weights requantized Q8_0 -> e4m3 with one
// power-of-2 scale per output row, activations e4m3 with one scale per token
// row, both applied in the GEMM epilogue - acc * ws[row] * xs[col], zero
// per-k32 work. Same scaling class vLLM's fp8 W8A8 ships (per-channel/
// per-tensor), so the quality precedent is established; our gates arbitrate.
// Coarser than per-32 (e4m3's 4-bit exponent absorbs the range) - lossy,
// quality-gated, PREFILL-shaped rows only (r >= 65); decode stays q8.

// one CTA per output row: pass 1 row amax over dequantized q8, pass 2 encode
__global__ void pd_q8_0_to_f8row_kernel(const int8_t* __restrict__ q8,
                                        const __half* __restrict__ s8,
                                        unsigned char* __restrict__ data,
                                        float* __restrict__ rscale,
                                        uint32_t in_dim, uint32_t out_dim) {
    const uint32_t row = blockIdx.x;
    if (row >= out_dim) return;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    const uint32_t nkb = in_dim >> 5;
    const int8_t* qr = q8 + (size_t)row * in_dim;
    const __half* sr = s8 + (size_t)row * nkb;
    __shared__ float wmax[32];
    __shared__ int s_e;
    float a = 0.0f;
    for (uint32_t i = tid; i < in_dim; i += nth)
        a = fmaxf(a, fabsf((float)qr[i] * __half2float(sr[i >> 5])));
    for (uint32_t sh = 16; sh > 0; sh >>= 1)
        a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, sh));
    if ((tid & 31u) == 0) wmax[tid >> 5] = a;
    __syncthreads();
    if (tid == 0) {
        float m = 0.0f;
        for (uint32_t w = 0; w < ((nth + 31u) >> 5); ++w) m = fmaxf(m, wmax[w]);
        int e = 0;
        if (m > 0.0f) {
            int ex;
            float fr = frexpf(m, &ex);
            e = ex - 9 + (fr > 0.875f ? 1 : 0);  // amax/2^e <= 448 = 0.875*2^9
        }
        s_e = e;
        rscale[row] = ldexpf(1.0f, e);
    }
    __syncthreads();
    const float inv = ldexpf(1.0f, -s_e);
    for (uint32_t i = tid; i < in_dim; i += nth)
        data[(size_t)row * in_dim + i] =
            __nv_fp8_e4m3((float)qr[i] * __half2float(sr[i >> 5]) * inv).__x;
}

PD_EXPORT
int pd_q8_0_to_f8row(const void* q8_data, const void* q8_scale, void* f8_data,
                     void* row_scale, uint32_t in_dim, uint32_t out_dim,
                     void* stream) {
#ifndef PD_BS_HOST
    (void)q8_data; (void)q8_scale; (void)f8_data; (void)row_scale;
    (void)in_dim; (void)out_dim; (void)stream;
    return cudaErrorNotSupported;
#else
    if (out_dim == 0) return 0;
    if (in_dim & 31u) return cudaErrorInvalidValue;
    pd_q8_0_to_f8row_kernel<<<out_dim, 256, 0, (cudaStream_t)stream>>>(
        (const int8_t*)q8_data, (const __half*)q8_scale,
        (unsigned char*)f8_data, (float*)row_scale, in_dim, out_dim);
    return pd_launch_status();
#endif
}

// activation twin: one CTA per token row, f32 in -> e4m3 bytes + f32 scale
// TI: f16 input plane for the attention streams (pd_ld4f exact expand).
template <typename TI = float>
__global__ void pd_quantize_e4m3_row_kernel(const TI* __restrict__ x,
                                            unsigned char* __restrict__ q,
                                            float* __restrict__ rscale,
                                            uint32_t n_dim) {
    // PDL: let the next (dependent-launched) GEMM start its dep-free W
    // prefetch while this kernel runs; its griddepcontrol.wait still gates
    // every dependent read on our full completion (probe-proven semantics).
    //  cascade: this kernel now also launches as a dependent, so gate
    // the body on full predecessor completion (no-op under plain launches).
    // PD_PDL_ARM (not raw asm): no-op below sm_90 - this kernel builds for
    // every arch and raw griddepcontrol breaks ptxas there.
    PD_PDL_ARM();

    // decode-band hot path: 4 calls/layer at r<=64 once the f8t attn planes
    // are on. float4/uchar4 vector lanes + 1024 threads - the 256-thread
    // scalar version cost 15.5 us/call at r=32 (32 CTAs on a 148-SM die).
    // The row max is exact regardless of fold order, so results stay
    // BIT-IDENTICAL to the scalar walk.
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    const TI* xr = x + (size_t)row * n_dim;
    __shared__ float wmax[32];
    __shared__ int s_e;
    const uint32_t n4 = n_dim >> 2;
    float a = 0.0f;
    for (uint32_t i = tid; i < n4; i += nth) {
        const float4 v = pd_ld4f(xr + (size_t)i * 4u);
        a = fmaxf(a, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)),
                           fmaxf(fabsf(v.z), fabsf(v.w))));
    }
    for (uint32_t sh = 16; sh > 0; sh >>= 1)
        a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, sh));
    if ((tid & 31u) == 0) wmax[tid >> 5] = a;
    __syncthreads();
    if (tid == 0) {
        float m = 0.0f;
        for (uint32_t w = 0; w < ((nth + 31u) >> 5); ++w) m = fmaxf(m, wmax[w]);
        int e = 0;
        if (m > 0.0f) {
            int ex;
            float fr = frexpf(m, &ex);
            e = ex - 9 + (fr > 0.875f ? 1 : 0);
        }
        s_e = e;
        rscale[row] = ldexpf(1.0f, e);
    }
    __syncthreads();
    const float inv = ldexpf(1.0f, -s_e);
    unsigned char* qr = q + (size_t)row * n_dim;
    for (uint32_t i = tid; i < n4; i += nth) {
        const float4 v = pd_ld4f(xr + (size_t)i * 4u);
        uchar4 o;
        o.x = __nv_fp8_e4m3(v.x * inv).__x;
        o.y = __nv_fp8_e4m3(v.y * inv).__x;
        o.z = __nv_fp8_e4m3(v.z * inv).__x;
        o.w = __nv_fp8_e4m3(v.w * inv).__x;
        *(uchar4*)(qr + (size_t)i * 4u) = o;
    }
}

// Single-PASS row quantize: one CTA per row, the row's values held in
// REGISTERS across the max reduction so x is read exactly once and every
// load is issued up front (V independent float4 per thread, no strided
// dependency chain). Bit-identical to both older forms -- same per-thread
// walk order, same warp fold, same exponent expression -- so this is a
// throughput change with no precision-class consequence.
//
// Why it matters (profiled on qwen3.6-27b c1): the activation quantize was
// the single largest remaining cost in the tick at +0.885 ms/step. It ran as
// two kernels (rowmax_part 6.39 us + quantize_row2 7.31 us, both grid 1x8)
// where a group-scaled quantizer needs one at ~1.9 us. The two-stage split
// exists because a PER-ROW scale needs the whole row's max before any
// element can be encoded; it was elected on a die where the single-block
// walk starved. On B200 at the serving batch that trade is inverted -- just
// electing the old single-block kernel is +2.1% end to end -- and holding
// the row in registers removes the second global pass on top of that.
template <typename TI, uint32_t V>
__global__ void __launch_bounds__(1024) pd_quantize_e4m3_row1p_kernel(
        const TI* __restrict__ x, unsigned char* __restrict__ q,
        float* __restrict__ rscale, uint32_t n_dim) {
    PD_PDL_ARM();
    const uint32_t row = blockIdx.x, tid = threadIdx.x, nth = blockDim.x;
    const TI* xr = x + (size_t)row * n_dim;
    __shared__ float wmax[32];
    __shared__ int s_e;
    const uint32_t n4 = n_dim >> 2;
    float4 v[V];
    float a = 0.0f;
    #pragma unroll
    for (uint32_t s = 0; s < V; ++s) {
        const uint32_t i = tid + s * nth;
        if (i < n4) {
            v[s] = pd_ld4f(xr + (size_t)i * 4u);
            a = fmaxf(a, fmaxf(fmaxf(fabsf(v[s].x), fabsf(v[s].y)),
                               fmaxf(fabsf(v[s].z), fabsf(v[s].w))));
        }
    }
    for (uint32_t sh = 16; sh > 0; sh >>= 1)
        a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, sh));
    if ((tid & 31u) == 0) wmax[tid >> 5] = a;
    __syncthreads();
    if (tid == 0) {
        float m = 0.0f;
        for (uint32_t w = 0; w < ((nth + 31u) >> 5); ++w) m = fmaxf(m, wmax[w]);
        int e = 0;
        if (m > 0.0f) {
            int ex;
            float fr = frexpf(m, &ex);
            e = ex - 9 + (fr > 0.875f ? 1 : 0);
        }
        s_e = e;
        rscale[row] = ldexpf(1.0f, e);
    }
    __syncthreads();
    const float inv = ldexpf(1.0f, -s_e);
    unsigned char* qr = q + (size_t)row * n_dim;
    #pragma unroll
    for (uint32_t s = 0; s < V; ++s) {
        const uint32_t i = tid + s * nth;
        if (i < n4) {
            uchar4 o;
            o.x = __nv_fp8_e4m3(v[s].x * inv).__x;
            o.y = __nv_fp8_e4m3(v[s].y * inv).__x;
            o.z = __nv_fp8_e4m3(v[s].z * inv).__x;
            o.w = __nv_fp8_e4m3(v[s].w * inv).__x;
            *(uchar4*)(qr + (size_t)i * 4u) = o;
        }
    }
}

// Chunked twin: grid (rows, C). Every block reads the whole row to derive the
// exact row max -- redundant, but the row is <= 68 KB and L2-hot after the
// first block touches it -- then quantizes only its own 1/C slice. Still one
// kernel and no cross-block dependency, so it keeps the single-pass form's
// advantage over the two-stage split while getting C x the CTAs.
//
// Why it exists: `grid = rows` starves at every serving width. At b=4 that is
// four CTAs on a 148-SM die, and act quantize measured 179.5 ms there - the
// single largest c4 decode cost. A per-token-GROUP quantizer sidesteps this
// by scaling grid with token*group instead of token alone; a per-ROW scale
// cannot, so this buys the parallelism back on the write side instead. Max
// is exact and order-free, so BIT-IDENTICAL.
template <typename TI, uint32_t V>
__global__ void __launch_bounds__(1024) pd_quantize_e4m3_row1pc_kernel(
        const TI* __restrict__ x, unsigned char* __restrict__ q,
        float* __restrict__ rscale, uint32_t n_dim) {
    PD_PDL_ARM();
    const uint32_t row = blockIdx.x, ch = blockIdx.y, C = gridDim.y;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    const TI* xr = x + (size_t)row * n_dim;
    __shared__ float wmax[32];
    __shared__ int s_e;
    const uint32_t n4 = n_dim >> 2;
    // pass 1: whole-row max (every block, identically)
    float a = 0.0f;
    #pragma unroll
    for (uint32_t s = 0; s < V; ++s) {
        const uint32_t i = tid + s * nth;
        if (i < n4) {
            const float4 v = pd_ld4f(xr + (size_t)i * 4u);
            a = fmaxf(a, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)),
                               fmaxf(fabsf(v.z), fabsf(v.w))));
        }
    }
    for (uint32_t sh = 16; sh > 0; sh >>= 1)
        a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, sh));
    if ((tid & 31u) == 0) wmax[tid >> 5] = a;
    __syncthreads();
    if (tid == 0) {
        float m = 0.0f;
        for (uint32_t w = 0; w < ((nth + 31u) >> 5); ++w) m = fmaxf(m, wmax[w]);
        int e = 0;
        if (m > 0.0f) {
            int ex;
            float fr = frexpf(m, &ex);
            e = ex - 9 + (fr > 0.875f ? 1 : 0);
        }
        s_e = e;
        if (ch == 0) rscale[row] = ldexpf(1.0f, e);
    }
    __syncthreads();
    // pass 2: this block's slice only
    const float inv = ldexpf(1.0f, -s_e);
    unsigned char* qr = q + (size_t)row * n_dim;
    const uint32_t per = (n4 + C - 1u) / C;
    const uint32_t i0 = ch * per, i1 = min(n4, i0 + per);
    for (uint32_t i = i0 + tid; i < i1; i += nth) {
        const float4 v = pd_ld4f(xr + (size_t)i * 4u);
        uchar4 o;
        o.x = __nv_fp8_e4m3(v.x * inv).__x;
        o.y = __nv_fp8_e4m3(v.y * inv).__x;
        o.z = __nv_fp8_e4m3(v.z * inv).__x;
        o.w = __nv_fp8_e4m3(v.w * inv).__x;
        *(uchar4*)(qr + (size_t)i * 4u) = o;
    }
}

// Elect the single-pass form when the row fits V<=8 float4 per thread
// (n_dim <= 32768, which covers every projection input in the fleet).
// Kill: PADDOCK_NO_ROWQ1P.
template <typename TI>
static inline bool pd_rowq_1p(const void* x, void* q, void* rscale,
                              uint32_t n_dim, uint32_t batch, cudaStream_t st) {
    static int off = -1;
    if (off < 0) off = pd_env("PADDOCK_NO_ROWQ1P") ? 1 : 0;
    if (off) return false;
    const uint32_t need = ((n_dim >> 2) + 1023u) / 1024u;
    if (need > 8u) return false;
    const uint32_t V = need <= 1u ? 1u : need <= 2u ? 2u : need <= 4u ? 4u : 8u;
    // grid = rows starves the die at serving widths (b=4 -> 4 CTAs of 148).
    // Split the WRITE across C blocks per row, each recomputing the exact max.
    // Kill: PADDOCK_NO_ROWQ1PC.
    static int nochunk = -1;
    if (nochunk < 0) nochunk = pd_env("PADDOCK_NO_ROWQ1PC") ? 1 : 0;
    static int nsm_q = 0;
    if (nsm_q == 0) {
        int d = 0;
        cudaGetDevice(&d);
        cudaDeviceGetAttribute(&nsm_q, cudaDevAttrMultiProcessorCount, d);
        if (nsm_q <= 0) nsm_q = 148;
    }
    uint32_t C = 1u;
    if (!nochunk && batch * 4u < (uint32_t)nsm_q) {
        C = ((uint32_t)nsm_q + batch * 4u - 1u) / (batch * 4u);
        if (C > 8u) C = 8u;
        if (C > need) C = need;          // no empty slices
        if (C < 1u) C = 1u;
    } else if (!nochunk && batch < (uint32_t)nsm_q * 2u) {
        // P44: the spec-verify width band (37..295 rows) fell
        // between the C-slice gate above and the 256-thread narrow form's
        // batch >= 2*nsm floor below - one 1024-thread CTA per row, 96 CTAs
        // on 148 SMs, single latency-bound wave, ~114 launches/verify tick.
        // Slice columns until the grid covers the 2-CTA/SM co-residency.
        // Same exact-max recipe, bit-identical. Default on since P44 ship
        // (gemma c32-spec verify -0.4ms; grid-shape only). Truthy gate:
        // PADDOCK_ROWQ_VSLICE=0 reverts to the one-CTA-per-row form.
        static int vslice = -1;
        if (vslice < 0) {
            const char* e = pd_env("PADDOCK_ROWQ_VSLICE");
            vslice = e ? (atoi(e) != 0) : 1;
        }
        if (vslice) {
            // P47 lever 1: at verify widths the 1024-thread
            // block pays a ~3x barrier + 32-warp-serial-scan tax vs the
            // 256-thread class on the same rows and bytes (in-graph:
            // rmsnorm_batch 4.7us at (128,1)x256 vs row1p 15.5us at
            // (128,1)x1024, 0.22 TB/s - not bandwidth). Launch the chunk
            // kernel at the widest block whose V=8 unrolled pass-1 walk
            // still covers the row, sliced to ~4 CTAs/SM. Max stays
            // exact and order-free => BIT-IDENTICAL. Truthy opt-in:
            // PADDOCK_ROWQ_VS256.
            static int vs2 = -1;
            if (vs2 < 0) {
                const char* e = pd_env("PADDOCK_ROWQ_VS256");
                vs2 = e ? (atoi(e) != 0) : 0;
            }
            if (vs2) {
                const uint32_t n4q = n_dim >> 2;
                const uint32_t nth = n4q <= 2048u ? 256u
                                   : n4q <= 4096u ? 512u : 1024u;
                const uint32_t nt = (n4q + nth - 1u) / nth;
                uint32_t C2 = ((uint32_t)nsm_q * 4u + batch - 1u) / batch;
                if (C2 > 8u) C2 = 8u;
                if (C2 > nt) C2 = nt;
                if (C2 < 1u) C2 = 1u;
                pd_pdl_go(pd_quantize_e4m3_row1pc_kernel<TI, 8u>,
                          dim3(batch, C2), nth, 0u, st, (const TI*)x,
                          (unsigned char*)q, (float*)rscale, n_dim);
                return true;
            }
            C = ((uint32_t)nsm_q * 2u + batch - 1u) / batch;
            if (C > 8u) C = 8u;
            if (C > need) C = need;      // no empty slices
            if (C < 1u) C = 1u;
        }
    }
    // Wide-grid narrow blocks: at wave widths (muse c32 = 5984
    // rows) the 1024-thread block caps co-residency at 2 CTAs/SM and the
    // grid serializes into ~20 latency-dominated CTA waves - DRAM
    // 27-40%, 25-29 warp cycles/issued instr. 256-thread blocks co-run 8/SM
    // with the same V*4-element register walk. Max is exact under the
    // changed element partition, so the narrow form is BIT-IDENTICAL.
    // Kill: PADDOCK_NO_ROWQ_NARROW.
    static int nonarrow = -1;
    if (nonarrow < 0) nonarrow = pd_env("PADDOCK_NO_ROWQ_NARROW") ? 1 : 0;
    const uint32_t need256 = ((n_dim >> 2) + 255u) / 256u;
    if (!nonarrow && C == 1u && batch >= (uint32_t)nsm_q * 2u && need256 <= 8u) {
        const uint32_t Vn = need256 <= 1u ? 1u
                          : need256 <= 2u ? 2u
                          : need256 <= 4u ? 4u : 8u;
        #define PD_RQ1PN(VV)                                                   \
            pd_pdl_go(pd_quantize_e4m3_row1p_kernel<TI, VV>, batch, 256,       \
                      0u, st, (const TI*)x, (unsigned char*)q,                 \
                      (float*)rscale, n_dim)
        if (Vn == 1u)      PD_RQ1PN(1u);
        else if (Vn == 2u) PD_RQ1PN(2u);
        else if (Vn == 4u) PD_RQ1PN(4u);
        else               PD_RQ1PN(8u);
        #undef PD_RQ1PN
        return true;
    }
    #define PD_RQ1P(VV)                                                        \
        do {                                                                   \
            if (C > 1u)                                                        \
                pd_pdl_go(pd_quantize_e4m3_row1pc_kernel<TI, VV>,              \
                          dim3(batch, C), 1024, 0u, st,                        \
                          (const TI*)x, (unsigned char*)q, (float*)rscale, n_dim); \
            else                                                               \
                pd_pdl_go(pd_quantize_e4m3_row1p_kernel<TI, VV>, batch, 1024,  \
                          0u, st, (const TI*)x, (unsigned char*)q,             \
                          (float*)rscale, n_dim);                              \
        } while (0)
    if (V == 1u)      PD_RQ1P(1u);
    else if (V == 2u) PD_RQ1P(2u);
    else if (V == 4u) PD_RQ1P(4u);
    else              PD_RQ1P(8u);
    #undef PD_RQ1P
    return true;
}

PD_EXPORT
int pd_quantize_e4m3_row(const void* x, void* q, void* rscale, uint32_t n_dim,
                         uint32_t batch, void* stream) {
    if (batch == 0) return 0;
    if (n_dim & 31u) return cudaErrorInvalidValue;
    if (pd_rowq_1p<float>(x, q, rscale, n_dim, batch, (cudaStream_t)stream))
        return pd_launch_status();
    {
        const uint32_t C = pd_rowq_chunks(batch);
        if (C > 1u) {
            float* scr = pd_rowq_scr();
            pd_pdl_go(pd_rowmax_part_kernel<float>, dim3(batch, C), 256, 0u,
                      (cudaStream_t)stream, (const float*)x, scr, n_dim, 1u);
            pd_pdl_go(pd_quantize_e4m3_row2_kernel<float>, dim3(batch, C), 256, 0u,
                      (cudaStream_t)stream, (const float*)x, (unsigned char*)q,
                      (float*)rscale, n_dim, (const float*)scr, 1u);
            return pd_launch_status();
        }
    }
    pd_pdl_go(pd_quantize_e4m3_row_kernel<float>, batch, 1024, 0u,
              (cudaStream_t)stream,
              (const float*)x, (unsigned char*)q, (float*)rscale, n_dim);
    return pd_launch_status();
}

// i16 twin for the attention streams: x is an f16 plane; same split
// election as the parent. Appended as its own export per the ABI growth
// rule.
PD_EXPORT
int pd_quantize_e4m3_row_i16(const void* x, void* q, void* rscale, uint32_t n_dim,
                             uint32_t batch, uint32_t i16, void* stream) {
    if (batch == 0) return 0;
    if (!i16)
        return pd_quantize_e4m3_row(x, q, rscale, n_dim, batch, stream);
    if (n_dim & 31u) return cudaErrorInvalidValue;
    if (pd_rowq_1p<__half>(x, q, rscale, n_dim, batch, (cudaStream_t)stream))
        return pd_launch_status();
    {
        const uint32_t C = pd_rowq_chunks(batch);
        if (C > 1u) {
            float* scr = pd_rowq_scr();
            pd_pdl_go(pd_rowmax_part_kernel<__half>, dim3(batch, C), 256, 0u,
                      (cudaStream_t)stream, (const __half*)x, scr, n_dim, 1u);
            pd_pdl_go(pd_quantize_e4m3_row2_kernel<__half>, dim3(batch, C), 256, 0u,
                      (cudaStream_t)stream, (const __half*)x, (unsigned char*)q,
                      (float*)rscale, n_dim, (const float*)scr, 1u);
            return pd_launch_status();
        }
    }
    pd_pdl_go(pd_quantize_e4m3_row_kernel<__half>, batch, 1024, 0u,
              (cudaStream_t)stream,
              (const __half*)x, (unsigned char*)q, (float*)rscale, n_dim);
    return pd_launch_status();
}

// raw e4m3 mma, accumulate in place - no scales anywhere near the K loop
__device__ __forceinline__ void pd_f8_mma_raw(float d[4], uint32_t a0, uint32_t a1,
                                              uint32_t a2, uint32_t a3, uint32_t b0,
                                              uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Structural clone of pd_f8_gemm_w8_kt with the scale machinery deleted:
// same STAGES-deep cp.async ring, same PD_BS_W8_ROW=80 row layout (the 16B
// scale prefix stays as dead pad so every offset and the ldmatrix swizzle
// carry over verbatim), same fragment maps - then a pure epilogue scale.
template <uint32_t STAGES, uint32_t CT = 128u>
__global__ void __launch_bounds__(256, (STAGES <= 2u) ? 2 : 1) pd_f8row_gemm_kt(
    const unsigned char* __restrict__ data, const float* __restrict__ wrs,
    const unsigned char* __restrict__ xq, const float* __restrict__ xrs,
    float* __restrict__ y, uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
#if PD_F8W8_OK
    PD_PDL_ARM();  // fp8-native chain cascade
    constexpr uint32_t ROW = PD_BS_W8_ROW;
    // CT = the COLUMN (batch) tile. 128 is the shipped shape and stays
    // bit-identical. CT=32 is the decode band's:  measured this
    // kernel occupancy-bound at batch<=32 because a 128-wide column tile
    // leaves warps 2..7 on columns that do not exist (c0w = 32/64/96 with
    // only 32 real columns). CT=32 gives every warp 16 real rows and the
    // one live column group: acc drops 64 -> 16 floats/thread and the Y
    // stage 128 -> 32 columns, which are the two limits found
    // pinning blocks/SM at 2. Same weights, same k order.
    constexpr uint32_t NS = CT / 32u;
    constexpr uint32_t YCOL = CT;
    extern __shared__ unsigned char pd_f8r_sh[];
    unsigned char* wring = pd_f8r_sh;
    unsigned char* yring = wring + STAGES * 128u * ROW;
    #define PD_F8R_WBUF(s) (wring + ((s) % STAGES) * 128u * ROW)
    #define PD_F8R_YBUF(s) (yring + ((s) % STAGES) * YCOL * ROW)

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, tq = lane & 3u;
    const uint32_t i0 = (CT == 128u) ? (warp & 1u) * 64u : warp * 16u;
    const uint32_t c0w = (CT == 128u) ? (warp >> 1) * 32u : 0u;
    const uint32_t nk = (in_dim + 63u) / 64u;
    const uint32_t batch_pad = (batch + (CT - 1u)) & ~(CT - 1u);
    const uint32_t nct = batch_pad / CT;
    const uint32_t tile = blockIdx.x;
    const uint32_t row_base = (tile / nct) * 128u;
    const uint32_t col_base = (tile % nct) * CT;
    // K-split: grid.y slabs each cover a contiguous kt range and
    // write their own plane at y + z*out_dim*batch (the launcher's partials
    // buffer, combined by the fixed-order ks_combine - the scales are a pure
    // per-(r,c) epilogue, so per-slab scaled partials sum exactly). A
    // gridDim.y == 1 launch keeps every bound and offset bit-identical to
    // the historical kernel. Decode-band grids (21-81 CTAs on a 188-SM die
    // at batch<=64) measured 23% of DRAM peak unsplit.
    const uint32_t nk_slab = (nk + gridDim.y - 1u) / gridDim.y;
    const uint32_t kt_lo = blockIdx.y * nk_slab;
    const uint32_t kt_hi = kt_lo + nk_slab < nk ? kt_lo + nk_slab : nk;
    y += (size_t)blockIdx.y * out_dim * batch;

    float acc[NS * 4u][4] = {};

    #define PD_F8R_ISSUE_W(dst, kt)                                                   \
        for (uint32_t u = tid; u < 512u; u += 256u) {                            \
            const uint32_t r = u >> 2, seg = u & 3u;                                  \
            const bool ok = (row_base + r) < out_dim && ((kt) * 4u + seg) * 16u < in_dim; \
            pd_cp_async16((int*)((dst) + r * ROW + 16u + seg * 16u),                  \
                          data + (size_t)(ok ? row_base + r : 0u) * in_dim +          \
                              (kt) * 64u + seg * 16u,                                 \
                          ok);                                                        \
        }
    #define PD_F8R_ISSUE_Y(dst, kt)                                                   \
        for (uint32_t u = tid; u < YCOL * 4u; u += 256u) {                                 \
            const uint32_t col = u >> 2, seg = u & 3u;                                \
            const bool ok = (col_base + col) < batch && ((kt) * 4u + seg) * 16u < in_dim; \
            pd_cp_async16((int*)((dst) + col * ROW + 16u + seg * 16u),                \
                          xq + (size_t)(ok ? col_base + col : 0u) * in_dim +          \
                              (kt) * 64u + seg * 16u,                                 \
                          ok);                                                        \
        }

    #pragma unroll
    for (uint32_t s = 0; s < STAGES - 1u; ++s) {
        if (kt_lo + s < kt_hi) { PD_F8R_ISSUE_W(PD_F8R_WBUF(kt_lo + s), kt_lo + s) PD_F8R_ISSUE_Y(PD_F8R_YBUF(kt_lo + s), kt_lo + s) }
        asm volatile("cp.async.commit_group;");
    }
    for (uint32_t kt = kt_lo; kt < kt_hi; ++kt) {
        unsigned char* tw = PD_F8R_WBUF(kt);
        unsigned char* ty = PD_F8R_YBUF(kt);
        const uint32_t pf = kt + STAGES - 1u;
        if (pf < kt_hi) { PD_F8R_ISSUE_W(PD_F8R_WBUF(pf), pf) PD_F8R_ISSUE_Y(PD_F8R_YBUF(pf), pf) }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group %0;" ::"n"(STAGES - 1u));
        __syncthreads();

        uint32_t am[NS][2][4];
        #pragma unroll
        for (uint32_t s = 0; s < NS; ++s) {
            const uint32_t rr = i0 + s * 16u + ((lane >> 3) & 1u) * 8u + (lane & 7u);
            #pragma unroll
            for (uint32_t kb = 0; kb < 2u; ++kb)
                pd_ldm_x4(am[s][kb], tw + rr * ROW + 16u + kb * 32u + (lane >> 4) * 16u);
        }
        #pragma unroll
        for (uint32_t j = 0; j < 4u; ++j) {
            uint32_t bm[4];
            pd_ldm_x4(bm, ty + (c0w + j * 8u + (lane & 7u)) * ROW + 16u +
                              (lane >> 3) * 16u);
            #pragma unroll
            for (uint32_t s = 0; s < NS; ++s) {
                pd_f8_mma_raw(acc[s * 4u + j], am[s][0][0], am[s][0][1],
                              am[s][0][2], am[s][0][3], bm[0], bm[1]);
                pd_f8_mma_raw(acc[s * 4u + j], am[s][1][0], am[s][1][1],
                              am[s][1][2], am[s][1][3], bm[2], bm[3]);
            }
        }
        __syncthreads();
    }
    #undef PD_F8R_ISSUE_W
    #undef PD_F8R_ISSUE_Y
    #undef PD_F8R_WBUF
    #undef PD_F8R_YBUF

    // epilogue: the only place scales exist. y[c*out+r] = acc * ws[r] * xs[c]
    #pragma unroll
    for (uint32_t j = 0; j < 4u; ++j) {
        const uint32_t c0 = col_base + c0w + j * 8u + 2u * tq;
        const float xs0 = c0 < batch ? xrs[c0] : 0.0f;
        const float xs1 = (c0 + 1u) < batch ? xrs[c0 + 1u] : 0.0f;
        #pragma unroll
        for (uint32_t s = 0; s < NS; ++s) {
            const uint32_t r0 = row_base + i0 + s * 16u + g;
            const uint32_t r8 = r0 + 8u;
            if (r0 < out_dim) {
                const float w0 = wrs[r0];
                if (c0 < batch) y[(size_t)c0 * out_dim + r0] = acc[s * 4u + j][0] * w0 * xs0;
                if (c0 + 1u < batch) y[(size_t)(c0 + 1u) * out_dim + r0] = acc[s * 4u + j][1] * w0 * xs1;
            }
            if (r8 < out_dim) {
                const float w8f = wrs[r8];
                if (c0 < batch) y[(size_t)c0 * out_dim + r8] = acc[s * 4u + j][2] * w8f * xs0;
                if (c0 + 1u < batch) y[(size_t)(c0 + 1u) * out_dim + r8] = acc[s * 4u + j][3] * w8f * xs1;
            }
        }
    }
#else
    (void)data; (void)wrs; (void)xq; (void)xrs; (void)y;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}

// TMA-ring window/wave GEMM on the f8row class.
// Both operands ride SWIZZLE_128B TMA 2D h64 boxes, S-stage mbarrier ring
// (single-thread bulk issue, one syncthreads per K-128 phase), fold-free f32
// accumulate, (r,c) scales in the epilogue - pd_f8row_gemm_mma's numerics
// (BIT-EQUAL vs its grid.y launch, 24/24 probe cells) on a cutlass-shaped
// pipeline. Grid (out/64, batch/64): weights re-read per 64-col batch tile
// and L2 serves the re-reads - the cutlass lesson; DRAM 55%, 2 blocks/SM,
// which is the ping-pong pipeline's own profile. Cold 12-plane measurement:
// wq M128 16.5us/1016 GB/s-wt against the old dispatch's 682; down M128
// 1143; the wave form is better at nearly every width.
template <uint32_t STAGES>
__global__ void __launch_bounds__(256) pd_f8row_gemm_tw_kernel(
    const __grid_constant__ PdTmap wmap,
    const __grid_constant__ PdTmap xmap,
    const float* __restrict__ wrs, const float* __restrict__ xrs,
    float* __restrict__ y, uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
#if PD_MMA_OK && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    constexpr uint32_t BM = 64u, BN = 64u, NWARP = 8u;
    constexpr uint32_t WR = BM / 16u, WC = NWARP / WR, CPW = BN / WC;
    constexpr uint32_t NSUB = CPW / 8u, NSUBK = 4u;
    constexpr uint32_t TB = BM * 128u; // 8KB per operand per stage
    // dynamic smem: 2*S*8KB clears the 48KB static cap at S=3 (ptxas 0xc080
    // > 0xc000 with the static layout); SW128 targets must stay
    // 1024-aligned, so round the dynamic base up (launch adds the slack)
    PD_PDL_ARM();  // fp8-native chain cascade
    extern __shared__ unsigned char pd_f8tw_dyn[];
    unsigned char* tw_base =
        (unsigned char*)(((uintptr_t)pd_f8tw_dyn + 1023u) & ~(uintptr_t)1023u);
    auto st_w = [&](uint32_t st) { return tw_base + (size_t)st * TB; };
    auto st_x = [&](uint32_t st) { return tw_base + (size_t)(STAGES + st) * TB; };
    __shared__ __align__(8) unsigned long long mb[STAGES];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, t = lane & 3u;
    const uint32_t wr = (warp % WR) * 16u, wc = (warp / WR) * CPW;
    // column-fastest raster: with a linearized grid (nrow*ncol, 1)
    // consecutive CTAs share one W row tile, so a W plane larger than L2
    // (30b FFN: 134 MB) streams from DRAM once and the other column tiles hit
    // L2 -- the 2-D grid (row tile fastest) streamed it once per column tile
    // (measured: g32k M=384 226us). A 2-D launch keeps the legacy
    // order (PADDOCK_NO_F8R_RASTER). Per-tile math unchanged => bit-equal.
    uint32_t rt = blockIdx.x, ct = blockIdx.y;
    if (gridDim.y == 1u) { const uint32_t ncol = (batch + BN - 1u) / BN; rt = blockIdx.x / ncol; ct = blockIdx.x % ncol; }
    const uint32_t row_base = rt * BM;
    const uint32_t col_base = ct * BN;
    const uint32_t nk = in_dim >> 7;
    const uint32_t mb0 = (uint32_t)__cvta_generic_to_shared(mb);
    if (tid == 0u) {
        #pragma unroll
        for (uint32_t s = 0; s < STAGES; ++s)
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(mb0 + s * 8u));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();
    auto issue = [&](uint32_t kt, uint32_t s) {
        const uint32_t m = mb0 + s * 8u;
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(m), "r"(2u * TB));
        const uint32_t wd = (uint32_t)__cvta_generic_to_shared(st_w(s));
        const uint32_t xd = (uint32_t)__cvta_generic_to_shared(st_x(s));
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%2, %3}], [%4];" ::"r"(wd), "l"(&wmap),
            "r"((int)(kt * 128u)), "r"((int)row_base), "r"(m) : "memory");
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%2, %3}], [%4];" ::"r"(xd), "l"(&xmap),
            "r"((int)(kt * 128u)), "r"((int)col_base), "r"(m) : "memory");
    };
    if (tid == 0u)
        for (uint32_t s = 0; s < STAGES && s < nk; ++s) issue(s, s);
    const uint32_t ldm_l7 = lane & 7u;
    const uint32_t ldm_arow = wr + ((lane & 8u) ? 8u : 0u) + ldm_l7;
    const uint32_t ca_hi = (lane & 16u) ? 1u : 0u;
    const uint32_t cb_hi = (lane & 8u) ? 1u : 0u;
    float acc[NSUB][4] = {};
    for (uint32_t ktp = 0; ktp < nk; ++ktp) {
        const uint32_t ph = ktp % STAGES;
        const uint32_t par = (ktp / STAGES) & 1u;
        {
            const uint32_t m = mb0 + ph * 8u;
            asm volatile(
                "{\n\t.reg .pred P;\n"
                "PD_F8TW_WAIT_%=:\n\t"
                "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
                "@!P bra PD_F8TW_WAIT_%=;\n\t}" ::"r"(m), "r"(par) : "memory");
        }
        const unsigned char* wp = st_w(ph);
        const unsigned char* xp = st_x(ph);
        #pragma unroll
        for (uint32_t sb = 0; sb < NSUBK; ++sb) {
            const uint32_t ca = sb * 2u + ca_hi;
            int a0, a1, a2, a3;
            pd_mma_ldm_x4(wp + ldm_arow * 128u + ((ca ^ (ldm_arow & 7u)) * 16u),
                          a0, a1, a2, a3);
            #pragma unroll
            for (uint32_t sub = 0; sub < NSUB; ++sub) {
                const uint32_t csub = wc + sub * 8u;
                const uint32_t col = csub + ldm_l7;
                const uint32_t cb = sb * 2u + cb_hi;
                int b0, b1;
                pd_mma_ldm_x2(xp + col * 128u + ((cb ^ (col & 7u)) * 16u), b0, b1);
                asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+f"(acc[sub][0]), "+f"(acc[sub][1]), "+f"(acc[sub][2]),
                      "+f"(acc[sub][3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            }
        }
        __syncthreads();
        if (tid == 0u && ktp + STAGES < nk) issue(ktp + STAGES, ph);
    }
    const uint32_t r0 = row_base + wr + g, r8 = row_base + wr + 8u + g;
    const float w0 = r0 < out_dim ? wrs[r0] : 0.0f;
    const float w8 = r8 < out_dim ? wrs[r8] : 0.0f;
    #pragma unroll
    for (uint32_t sub = 0; sub < NSUB; ++sub) {
        const uint32_t c0 = col_base + wc + sub * 8u + 2u * t, c1 = c0 + 1u;
        const float x0 = c0 < batch ? xrs[c0] : 0.0f;
        const float x1 = c1 < batch ? xrs[c1] : 0.0f;
        if (r0 < out_dim) {
            if (c0 < batch) y[(size_t)c0 * out_dim + r0] = acc[sub][0] * w0 * x0;
            if (c1 < batch) y[(size_t)c1 * out_dim + r0] = acc[sub][1] * w0 * x1;
        }
        if (r8 < out_dim) {
            if (c0 < batch) y[(size_t)c0 * out_dim + r8] = acc[sub][2] * w8 * x0;
            if (c1 < batch) y[(size_t)c1 * out_dim + r8] = acc[sub][3] * w8 * x1;
        }
    }
#else
    (void)wmap; (void)xmap; (void)wrs; (void)xrs; (void)y;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}


// tw4: dedicated producer warp (warp 8) + 8 consumer warps, full/empty
// mbarrier handshake, no __syncthreads in loop, deep ring at 1 block/SM.
template <uint32_t STAGES>
__global__ void __launch_bounds__(288) pd_f8row_gemm_tw4_kernel(
    const __grid_constant__ PdTmap wmap,
    const __grid_constant__ PdTmap xmap,
    const float* __restrict__ wrs, const float* __restrict__ xrs,
    float* __restrict__ y, uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
#if PD_MMA_OK && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    constexpr uint32_t BM = 64u, BN = 64u, NWARP = 8u;
    constexpr uint32_t WR = BM / 16u, WC = NWARP / WR, CPW = BN / WC;
    constexpr uint32_t NSUB = CPW / 8u, NSUBK = 4u;
    constexpr uint32_t TB = BM * 128u;
    PD_PDL_ARM();  // fp8-native chain cascade
    extern __shared__ unsigned char pd_f8tw4_dyn[];
    unsigned char* base = (unsigned char*)(((uintptr_t)pd_f8tw4_dyn + 1023u) & ~(uintptr_t)1023u);
    auto st_w = [&](uint32_t sl) { return base + (size_t)sl * TB; };
    auto st_x = [&](uint32_t sl) { return base + (size_t)(STAGES + sl) * TB; };
    __shared__ __align__(8) unsigned long long mbf[STAGES];
    __shared__ __align__(8) unsigned long long mbe[STAGES];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t nk = in_dim >> 7;
    const uint32_t row_base = blockIdx.x * BM;
    const uint32_t col_base = blockIdx.y * BN;
    const uint32_t mf0 = (uint32_t)__cvta_generic_to_shared(mbf);
    const uint32_t me0 = (uint32_t)__cvta_generic_to_shared(mbe);
    if (tid == 0u) {
        #pragma unroll
        for (uint32_t s = 0; s < STAGES; ++s) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(mf0 + s * 8u));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(me0 + s * 8u), "r"(NWARP));
        }
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();
    if (warp == NWARP) {
        if (lane == 0u) {
            auto issue = [&](uint32_t kt) {
                const uint32_t sl = kt % STAGES;
                const uint32_t m = mf0 + sl * 8u;
                const uint32_t wd = (uint32_t)__cvta_generic_to_shared(st_w(sl));
                const uint32_t xd = (uint32_t)__cvta_generic_to_shared(st_x(sl));
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(m), "r"(2u * TB));
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3}], [%4];" ::"r"(wd), "l"(&wmap),
                    "r"((int)(kt * 128u)), "r"((int)row_base), "r"(m) : "memory");
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3}], [%4];" ::"r"(xd), "l"(&xmap),
                    "r"((int)(kt * 128u)), "r"((int)col_base), "r"(m) : "memory");
            };
            for (uint32_t kt = 0; kt < nk && kt < STAGES; ++kt) issue(kt);
            for (uint32_t kt = STAGES; kt < nk; ++kt) {
                const uint32_t sl = kt % STAGES;
                const uint32_t par = ((kt / STAGES) - 1u) & 1u;
                const uint32_t m = me0 + sl * 8u;
                asm volatile(
                    "{\n\t.reg .pred P;\n"
                    "PD_TW4_WE_%=:\n\t"
                    "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
                    "@!P bra PD_TW4_WE_%=;\n\t}" ::"r"(m), "r"(par) : "memory");
                issue(kt);
            }
        }
        return;
    }
    const uint32_t g = lane >> 2, t = lane & 3u;
    const uint32_t wr = (warp % WR) * 16u, wc = (warp / WR) * CPW;
    const uint32_t ldm_l7 = lane & 7u;
    const uint32_t ldm_arow = wr + ((lane & 8u) ? 8u : 0u) + ldm_l7;
    const uint32_t ca_hi = (lane & 16u) ? 1u : 0u;
    const uint32_t cb_hi = (lane & 8u) ? 1u : 0u;
    float acc[NSUB][4] = {};
    for (uint32_t kt = 0; kt < nk; ++kt) {
        const uint32_t sl = kt % STAGES;
        const uint32_t par = (kt / STAGES) & 1u;
        {
            const uint32_t m = mf0 + sl * 8u;
            asm volatile(
                "{\n\t.reg .pred P;\n"
                "PD_TW4_WF_%=:\n\t"
                "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
                "@!P bra PD_TW4_WF_%=;\n\t}" ::"r"(m), "r"(par) : "memory");
        }
        const unsigned char* wp = st_w(sl);
        const unsigned char* xp = st_x(sl);
        #pragma unroll
        for (uint32_t sb = 0; sb < NSUBK; ++sb) {
            const uint32_t ca = sb * 2u + ca_hi;
            int a0, a1, a2, a3;
            pd_mma_ldm_x4(wp + ldm_arow * 128u + ((ca ^ (ldm_arow & 7u)) * 16u),
                          a0, a1, a2, a3);
            #pragma unroll
            for (uint32_t sub = 0; sub < NSUB; ++sub) {
                const uint32_t csub = wc + sub * 8u;
                const uint32_t col = csub + ldm_l7;
                const uint32_t cb = sb * 2u + cb_hi;
                int b0, b1;
                pd_mma_ldm_x2(xp + col * 128u + ((cb ^ (col & 7u)) * 16u), b0, b1);
                asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+f"(acc[sub][0]), "+f"(acc[sub][1]), "+f"(acc[sub][2]),
                      "+f"(acc[sub][3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            }
        }
        if (lane == 0u)
            asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(me0 + sl * 8u) : "memory");
    }
    const uint32_t r0 = row_base + wr + g, r8 = row_base + wr + 8u + g;
    const float w0 = r0 < out_dim ? wrs[r0] : 0.0f;
    const float w8 = r8 < out_dim ? wrs[r8] : 0.0f;
    #pragma unroll
    for (uint32_t sub = 0; sub < NSUB; ++sub) {
        const uint32_t c0 = col_base + wc + sub * 8u + 2u * t, c1 = c0 + 1u;
        const float x0 = c0 < batch ? xrs[c0] : 0.0f;
        const float x1 = c1 < batch ? xrs[c1] : 0.0f;
        if (r0 < out_dim) {
            if (c0 < batch) y[(size_t)c0 * out_dim + r0] = acc[sub][0] * w0 * x0;
            if (c1 < batch) y[(size_t)c1 * out_dim + r0] = acc[sub][1] * w0 * x1;
        }
        if (r8 < out_dim) {
            if (c0 < batch) y[(size_t)c0 * out_dim + r8] = acc[sub][2] * w8 * x0;
            if (c1 < batch) y[(size_t)c1 * out_dim + r8] = acc[sub][3] * w8 * x1;
        }
    }
#else
    (void)wmap; (void)xmap; (void)wrs; (void)xrs; (void)y;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}


// tw5: BM=128 x BN=128 tile, producer warp + 16 consumer warps, S-deep ring.
// Two h64 TMA boxes per operand per stage. Gate-shape geometry: 100 CTAs = 1 wave.
template <uint32_t STAGES>
__global__ void __launch_bounds__(544) pd_f8row_gemm_tw5_kernel(
    const __grid_constant__ PdTmap wmap,
    const __grid_constant__ PdTmap xmap,
    const float* __restrict__ wrs, const float* __restrict__ xrs,
    float* __restrict__ y, uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
#if PD_MMA_OK && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    constexpr uint32_t BM = 128u, BN = 128u, NWARP = 16u;
    constexpr uint32_t WR = 8u, WC = NWARP / WR, CPW = BN / WC;
    constexpr uint32_t NSUB = CPW / 8u, NSUBK = 4u;
    constexpr uint32_t TB = 64u * 128u;          // one h64 box
    constexpr uint32_t STG = 2u * TB;            // per-operand per-stage (2 boxes)
    PD_PDL_ARM();  // fp8-native chain cascade
    extern __shared__ unsigned char pd_f8tw5_dyn[];
    unsigned char* base = (unsigned char*)(((uintptr_t)pd_f8tw5_dyn + 1023u) & ~(uintptr_t)1023u);
    auto st_w = [&](uint32_t sl) { return base + (size_t)sl * STG; };
    auto st_x = [&](uint32_t sl) { return base + (size_t)(STAGES + sl) * STG; };
    __shared__ __align__(8) unsigned long long mbf[STAGES];
    __shared__ __align__(8) unsigned long long mbe[STAGES];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t nk = in_dim >> 7;
    // column-fastest raster: with a linearized grid (nrow*ncol, 1)
    // consecutive CTAs share one W row tile, so a W plane larger than L2
    // (30b FFN: 134 MB) streams from DRAM once and the other column tiles hit
    // L2 -- the 2-D grid (row tile fastest) streamed it once per column tile
    // (measured: g32k M=384 226us). A 2-D launch keeps the legacy
    // order (PADDOCK_NO_F8R_RASTER). Per-tile math unchanged => bit-equal.
    uint32_t rt = blockIdx.x, ct = blockIdx.y;
    if (gridDim.y == 1u) { const uint32_t ncol = (batch + BN - 1u) / BN; rt = blockIdx.x / ncol; ct = blockIdx.x % ncol; }
    const uint32_t row_base = rt * BM;
    const uint32_t col_base = ct * BN;
    const uint32_t mf0 = (uint32_t)__cvta_generic_to_shared(mbf);
    const uint32_t me0 = (uint32_t)__cvta_generic_to_shared(mbe);
    if (tid == 0u) {
        #pragma unroll
        for (uint32_t s = 0; s < STAGES; ++s) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(mf0 + s * 8u));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(me0 + s * 8u), "r"(NWARP));
        }
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();
    if (warp == NWARP) {
        if (lane == 0u) {
            auto issue = [&](uint32_t kt) {
                const uint32_t sl = kt % STAGES;
                const uint32_t m = mf0 + sl * 8u;
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(m), "r"(2u * STG));
                #pragma unroll
                for (uint32_t h = 0; h < 2u; ++h) {
                    const uint32_t wd = (uint32_t)__cvta_generic_to_shared(st_w(sl) + h * TB);
                    const uint32_t xd = (uint32_t)__cvta_generic_to_shared(st_x(sl) + h * TB);
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                        " [%0], [%1, {%2, %3}], [%4];" ::"r"(wd), "l"(&wmap),
                        "r"((int)(kt * 128u)), "r"((int)(row_base + h * 64u)), "r"(m) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                        " [%0], [%1, {%2, %3}], [%4];" ::"r"(xd), "l"(&xmap),
                        "r"((int)(kt * 128u)), "r"((int)(col_base + h * 64u)), "r"(m) : "memory");
                }
            };
            for (uint32_t kt = 0; kt < nk && kt < STAGES; ++kt) issue(kt);
            for (uint32_t kt = STAGES; kt < nk; ++kt) {
                const uint32_t sl = kt % STAGES;
                const uint32_t par = ((kt / STAGES) - 1u) & 1u;
                const uint32_t m = me0 + sl * 8u;
                asm volatile(
                    "{\n\t.reg .pred P;\n"
                    "PD_TW5_WE_%=:\n\t"
                    "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
                    "@!P bra PD_TW5_WE_%=;\n\t}" ::"r"(m), "r"(par) : "memory");
                issue(kt);
            }
        }
        return;
    }
    const uint32_t g = lane >> 2, t = lane & 3u;
    const uint32_t wr = (warp % WR) * 16u, wc = (warp / WR) * CPW;
    const uint32_t ldm_l7 = lane & 7u;
    const uint32_t arow = wr + ((lane & 8u) ? 8u : 0u) + ldm_l7;
    const unsigned char* wbox_off_zero = nullptr; (void)wbox_off_zero;
    const uint32_t ca_hi = (lane & 16u) ? 1u : 0u;
    const uint32_t cb_hi = (lane & 8u) ? 1u : 0u;
    float acc[NSUB][4] = {};
    for (uint32_t kt = 0; kt < nk; ++kt) {
        const uint32_t sl = kt % STAGES;
        const uint32_t par = (kt / STAGES) & 1u;
        {
            const uint32_t m = mf0 + sl * 8u;
            asm volatile(
                "{\n\t.reg .pred P;\n"
                "PD_TW5_WF_%=:\n\t"
                "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
                "@!P bra PD_TW5_WF_%=;\n\t}" ::"r"(m), "r"(par) : "memory");
        }
        const unsigned char* wp = st_w(sl) + (arow >> 6) * TB;
        const uint32_t ar = arow & 63u;
        #pragma unroll
        for (uint32_t sb = 0; sb < NSUBK; ++sb) {
            const uint32_t ca = sb * 2u + ca_hi;
            int a0, a1, a2, a3;
            pd_mma_ldm_x4(wp + ar * 128u + ((ca ^ (ar & 7u)) * 16u), a0, a1, a2, a3);
            #pragma unroll
            for (uint32_t sub = 0; sub < NSUB; ++sub) {
                const uint32_t col = wc + sub * 8u + ldm_l7;
                const unsigned char* xp = st_x(sl) + (col >> 6) * TB;
                const uint32_t cl = col & 63u;
                const uint32_t cb = sb * 2u + cb_hi;
                int b0, b1;
                pd_mma_ldm_x2(xp + cl * 128u + ((cb ^ (cl & 7u)) * 16u), b0, b1);
                asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+f"(acc[sub][0]), "+f"(acc[sub][1]), "+f"(acc[sub][2]),
                      "+f"(acc[sub][3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            }
        }
        if (lane == 0u)
            asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(me0 + sl * 8u) : "memory");
    }
    const uint32_t r0 = row_base + wr + g, r8 = row_base + wr + 8u + g;
    const float w0 = r0 < out_dim ? wrs[r0] : 0.0f;
    const float w8 = r8 < out_dim ? wrs[r8] : 0.0f;
    #pragma unroll
    for (uint32_t sub = 0; sub < NSUB; ++sub) {
        const uint32_t c0 = col_base + wc + sub * 8u + 2u * t, c1 = c0 + 1u;
        const float x0 = c0 < batch ? xrs[c0] : 0.0f;
        const float x1 = c1 < batch ? xrs[c1] : 0.0f;
        if (r0 < out_dim) {
            if (c0 < batch) y[(size_t)c0 * out_dim + r0] = acc[sub][0] * w0 * x0;
            if (c1 < batch) y[(size_t)c1 * out_dim + r0] = acc[sub][1] * w0 * x1;
        }
        if (r8 < out_dim) {
            if (c0 < batch) y[(size_t)c0 * out_dim + r8] = acc[sub][2] * w8 * x0;
            if (c1 < batch) y[(size_t)c1 * out_dim + r8] = acc[sub][3] * w8 * x1;
        }
    }
#else
    (void)wmap; (void)xmap; (void)wrs; (void)xrs; (void)y;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}


// Tensor-map cache: the TMA arms encoded two CUtensorMaps per
// launch (a driver call each, ~1-3 us on the host). The decode tick is eager
// (PDL cascade), so at 3 narrow planes x 64 layers that was several hundred
// encodes per tick -- the twdec A/B lost 1% on 8b (8.7 ms tick) while winning
// 0.8% on 30b (24.6 ms tick amortizes it). Weights are static and the decode
// activation scratch is a stable pointer per batch width, so (base, inner,
// outer) identifies a map for the process lifetime. Direct-mapped, linear
// probe, 2048 entries (30b: 64 layers x 4 planes + X maps per width).
// Launchers run under the engine's per-stream serialization => no locking.
struct PdTmapEntry { const void* base; uint64_t inner, outer; CUtensorMap map; };
#if defined(PD_BS_HOST) || defined(PD_TC5_HOST)  // pd_tmap_2d_h64 (tma_desc.cuh) exists only on those builds; so does the one caller
static bool pd_tmap_2d_h64_cached(CUtensorMap* out, const void* base, uint64_t inner, uint64_t outer) {
    static PdTmapEntry* tab = nullptr;
    static const uint32_t N = 2048u;
    if (!tab) { tab = new PdTmapEntry[N]; for (uint32_t i = 0; i < N; ++i) tab[i].base = nullptr; }
    uint64_t h = ((uint64_t)(uintptr_t)base >> 8) * 0x9E3779B97F4A7C15ull ^ (inner * 31u + outer);
    uint32_t i = (uint32_t)(h >> 20) & (N - 1u);
    for (uint32_t k = 0; k < 8u; ++k) {
        PdTmapEntry& e = tab[(i + k) & (N - 1u)];
        if (e.base == base && e.inner == inner && e.outer == outer) { *out = e.map; return true; }
        if (e.base == nullptr) {
            if (!pd_tmap_2d_h64(&e.map, base, inner, outer)) return false;
            e.base = base; e.inner = inner; e.outer = outer; *out = e.map; return true;
        }
    }
    return pd_tmap_2d_h64(out, base, inner, outer);   // probe run full: encode uncached
}
#endif  // PD_BS_HOST || PD_TC5_HOST


template <uint32_t STAGES, uint32_t BN>
__global__ void __launch_bounds__(288) pd_f8row_gemm_tw4d_kernel(
    const __grid_constant__ PdTmap wmap,
    const __grid_constant__ PdTmap xmap,
    const float* __restrict__ wrs, const float* __restrict__ xrs,
    float* __restrict__ y, uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
#if PD_MMA_OK && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    constexpr uint32_t BM = 64u, NWARP = 8u;
    constexpr uint32_t WR = BM / 16u, WC = NWARP / WR, CPW = BN / WC;
    constexpr uint32_t NSUB = CPW / 8u, NSUBK = 4u;
    constexpr uint32_t TBW = BM * 128u, TBX = BN * 128u;
    PD_PDL_ARM();  // fp8-native chain cascade
    extern __shared__ unsigned char pd_f8tw4d_dyn[];
    unsigned char* base = (unsigned char*)(((uintptr_t)pd_f8tw4d_dyn + 1023u) & ~(uintptr_t)1023u);
    auto st_w = [&](uint32_t sl) { return base + (size_t)sl * TBW; };
    auto st_x = [&](uint32_t sl) { return base + (size_t)STAGES * TBW + (size_t)sl * TBX; };
    __shared__ __align__(8) unsigned long long mbf[STAGES];
    __shared__ __align__(8) unsigned long long mbe[STAGES];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t n_blocks = in_dim >> 7;                 // K tiles of 128
    const uint32_t nz = gridDim.z;
    uint32_t kt_lo = 0, kt_hi = n_blocks;
    if (nz > 1u) {
        const uint32_t per = (n_blocks + nz - 1u) / nz;
        kt_lo = blockIdx.z * per;
        kt_hi = kt_lo + per < n_blocks ? kt_lo + per : n_blocks;
        y += (size_t)blockIdx.z * out_dim * batch;
    }
    const uint32_t nk = kt_hi > kt_lo ? kt_hi - kt_lo : 0u;
    const uint32_t row_base = blockIdx.x * BM;
    const uint32_t col_base = blockIdx.y * BN;
    const uint32_t mf0 = (uint32_t)__cvta_generic_to_shared(mbf);
    const uint32_t me0 = (uint32_t)__cvta_generic_to_shared(mbe);
    if (tid == 0u) {
        #pragma unroll
        for (uint32_t s = 0; s < STAGES; ++s) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(mf0 + s * 8u));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(me0 + s * 8u), "r"(NWARP));
        }
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();
    if (warp == NWARP) {
        if (lane == 0u) {
            auto issue = [&](uint32_t i) {
                const uint32_t sl = i % STAGES, kt = kt_lo + i;
                const uint32_t m = mf0 + sl * 8u;
                const uint32_t wd = (uint32_t)__cvta_generic_to_shared(st_w(sl));
                const uint32_t xd = (uint32_t)__cvta_generic_to_shared(st_x(sl));
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(m), "r"(TBW + TBX));
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3}], [%4];" ::"r"(wd), "l"(&wmap),
                    "r"((int)(kt * 128u)), "r"((int)row_base), "r"(m) : "memory");
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3}], [%4];" ::"r"(xd), "l"(&xmap),
                    "r"((int)(kt * 128u)), "r"((int)col_base), "r"(m) : "memory");
            };
            for (uint32_t i = 0; i < nk && i < STAGES; ++i) issue(i);
            for (uint32_t i = STAGES; i < nk; ++i) {
                const uint32_t sl = i % STAGES;
                const uint32_t par = ((i / STAGES) - 1u) & 1u;
                const uint32_t m = me0 + sl * 8u;
                asm volatile(
                    "{\n\t.reg .pred P;\n"
                    "PD_TW4D_WE_%=:\n\t"
                    "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
                    "@!P bra PD_TW4D_WE_%=;\n\t}" ::"r"(m), "r"(par) : "memory");
                issue(i);
            }
        }
        return;
    }
    const uint32_t g = lane >> 2, t = lane & 3u;
    const uint32_t wr = (warp % WR) * 16u, wc = (warp / WR) * CPW;
    const uint32_t ldm_l7 = lane & 7u;
    const uint32_t ldm_arow = wr + ((lane & 8u) ? 8u : 0u) + ldm_l7;
    const uint32_t ca_hi = (lane & 16u) ? 1u : 0u;
    const uint32_t cb_hi = (lane & 8u) ? 1u : 0u;
    float acc[NSUB][4] = {};
    for (uint32_t i = 0; i < nk; ++i) {
        const uint32_t sl = i % STAGES;
        const uint32_t par = (i / STAGES) & 1u;
        {
            const uint32_t m = mf0 + sl * 8u;
            asm volatile(
                "{\n\t.reg .pred P;\n"
                "PD_TW4D_WF_%=:\n\t"
                "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
                "@!P bra PD_TW4D_WF_%=;\n\t}" ::"r"(m), "r"(par) : "memory");
        }
        const unsigned char* wp = st_w(sl);
        const unsigned char* xp = st_x(sl);
        #pragma unroll
        for (uint32_t sb = 0; sb < NSUBK; ++sb) {
            const uint32_t ca = sb * 2u + ca_hi;
            int a0, a1, a2, a3;
            pd_mma_ldm_x4(wp + ldm_arow * 128u + ((ca ^ (ldm_arow & 7u)) * 16u), a0, a1, a2, a3);
            #pragma unroll
            for (uint32_t sub = 0; sub < NSUB; ++sub) {
                const uint32_t col = wc + sub * 8u + ldm_l7;
                const uint32_t cb = sb * 2u + cb_hi;
                int b0, b1;
                pd_mma_ldm_x2(xp + col * 128u + ((cb ^ (col & 7u)) * 16u), b0, b1);
                asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+f"(acc[sub][0]), "+f"(acc[sub][1]), "+f"(acc[sub][2]), "+f"(acc[sub][3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            }
        }
        if (lane == 0u)
            asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(me0 + sl * 8u) : "memory");
    }
    const uint32_t r0 = row_base + wr + g, r8 = row_base + wr + 8u + g;
    // K-split slabs write RAW partials (scales applied once by the caller's
    // epilogue is not available here) -- so apply the (r,c) scales per slab:
    // per-slab scaled partials sum exactly (scales are per-(r,c) constants).
    const float w0 = r0 < out_dim ? wrs[r0] : 0.0f;
    const float w8 = r8 < out_dim ? wrs[r8] : 0.0f;
    #pragma unroll
    for (uint32_t sub = 0; sub < NSUB; ++sub) {
        const uint32_t c0 = col_base + wc + sub * 8u + 2u * t, c1 = c0 + 1u;
        const float x0 = c0 < batch ? xrs[c0] : 0.0f;
        const float x1 = c1 < batch ? xrs[c1] : 0.0f;
        if (r0 < out_dim) {
            if (c0 < batch) y[(size_t)c0 * out_dim + r0] = acc[sub][0] * w0 * x0;
            if (c1 < batch) y[(size_t)c1 * out_dim + r0] = acc[sub][1] * w0 * x1;
        }
        if (r8 < out_dim) {
            if (c0 < batch) y[(size_t)c0 * out_dim + r8] = acc[sub][2] * w8 * x0;
            if (c1 < batch) y[(size_t)c1 * out_dim + r8] = acc[sub][3] * w8 * x1;
        }
    }
#else
    (void)wmap; (void)xmap; (void)wrs; (void)xrs; (void)y;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}

// h-parametrized X-box map (BN=32 column tile of the tw4d decode arm) + its
// cache; same contract as pd_tmap_2d_h64 / _cached (tma_desc.cuh).
static bool pd_tmap_2d_hN(CUtensorMap* map, const void* base, uint64_t inner, uint64_t rows, uint32_t h) {
    pd_tmap_encode_fn enc = pd_tmap_encode();
    if (!enc || ((uintptr_t)base & 15u) || (inner & 15u)) return false;
    const cuuint64_t gdim[2] = {inner, rows};
    const cuuint64_t gstride[1] = {inner};
    const cuuint32_t box[2] = {128u, h};
    const cuuint32_t estride[2] = {1u, 1u};
    return enc(map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2u, (void*)base, gdim, gstride,
               box, estride, CU_TENSOR_MAP_INTERLEAVE_NONE,
               CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}
static bool pd_tmap_2d_hN_cached(CUtensorMap* out, const void* base, uint64_t inner, uint64_t outer, uint32_t h) {
    static PdTmapEntry* tab = nullptr;
    static const uint32_t N = 1024u;
    if (!tab) { tab = new PdTmapEntry[N]; for (uint32_t i = 0; i < N; ++i) tab[i].base = nullptr; }
    const uint64_t okey = outer * 256u + h;   // h < 256 folded into the outer key
    uint64_t hh = ((uint64_t)(uintptr_t)base >> 8) * 0x9E3779B97F4A7C15ull ^ (inner * 31u + okey);
    uint32_t i = (uint32_t)(hh >> 20) & (N - 1u);
    for (uint32_t k = 0; k < 8u; ++k) {
        PdTmapEntry& e = tab[(i + k) & (N - 1u)];
        if (e.base == base && e.inner == inner && e.outer == okey) { *out = e.map; return true; }
        if (e.base == nullptr) {
            if (!pd_tmap_2d_hN(&e.map, base, inner, outer, h)) return false;
            e.base = base; e.inner = inner; e.outer = okey; *out = e.map; return true;
        }
    }
    return pd_tmap_2d_hN(out, base, inner, outer, h);
}


// ---- norm -> e4m3-row quant fusion (the fp8 decode tick) -----------------
// The fp8 lane paid rmsnorm (3.7 us) + quantize_row1p (1.9) as two launches
// per norm site, twice per layer, where a fused triton kernel does
// norm+quant in one (4.2-4.8 us) -- ~0.5 ms/tick at 30b, 0.35 at 8b. One CTA
// per row: the rmsnorm's own reduction (same ACC mode, same nth) and the same
// `v * inv * w` write, the normalized float4s held in registers exactly like
// pd_quantize_e4m3_row1p_kernel (V per thread), then its order-free row max
// and the same e4m3 expression => xn AND (q, rscale) are bit-identical to the
// unfused pair. Decode band only (batch < 64 at the 1024-wide norm walk);
// the launcher declines (100) elsewhere and the engine falls back.
template <int ACC, uint32_t V>
__global__ void __launch_bounds__(1024) pd_rmsnorm_quant_e4m3_row_kernel(
        const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ xn,
        unsigned char* __restrict__ q, float* __restrict__ rscale, uint32_t n, float eps) {
    using A = typename pd_acc_of<ACC>::type;
    PD_PDL_ARM();
    const uint32_t b = blockIdx.x, tid = threadIdx.x, nth = blockDim.x;
    const float* xb = x + (size_t)b * n;
    __shared__ A wsum[32];
    __shared__ float s_inv;
    __shared__ float wmax[32];
    __shared__ int s_e;
    A acc;
    if constexpr (ACC == PD_ACC_DF) { acc.hi = 0.0f; acc.lo = 0.0f; } else { acc = (A)0; }
    const uint32_t n4 = n >> 2;
    const float4* x4 = reinterpret_cast<const float4*>(xb);
    for (uint32_t i = tid; i < n4; i += nth) {
        float4 v = x4[i];
        if constexpr (ACC == PD_ACC_DF) {
            pd_df_add(acc, v.x * v.x); pd_df_add(acc, v.y * v.y);
            pd_df_add(acc, v.z * v.z); pd_df_add(acc, v.w * v.w);
        } else {
            acc += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
    }
    for (uint32_t sh = 16; sh > 0; sh >>= 1) {
        if constexpr (ACC == PD_ACC_DF) {
            pd_df o; o.hi = __shfl_down_sync(0xffffffffu, acc.hi, sh); o.lo = __shfl_down_sync(0xffffffffu, acc.lo, sh);
            acc = pd_df_merge(acc, o);
        } else {
            acc += __shfl_down_sync(0xffffffffu, acc, sh);
        }
    }
    const uint32_t warp = tid >> 5, lane = tid & 31u;
    if (lane == 0) wsum[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        const uint32_t nwarps = (nth + 31u) >> 5;
        double total;
        if constexpr (ACC == PD_ACC_DF) {
            pd_df sum; sum.hi = 0.0f; sum.lo = 0.0f;
            for (uint32_t wi = 0; wi < nwarps; ++wi) sum = pd_df_merge(sum, wsum[wi]);
            total = (double)sum.hi + (double)sum.lo;
        } else {
            A sum = (A)0;
            for (uint32_t wi = 0; wi < nwarps; ++wi) sum += wsum[wi];
            total = (double)sum;
        }
        s_inv = 1.0f / sqrtf((float)(total / (double)n) + eps);
    }
    __syncthreads();
    const float inv = s_inv;
    const float4* w4 = reinterpret_cast<const float4*>(w);
    float4* o4 = reinterpret_cast<float4*>(xn + (size_t)b * n);
    float4 r[V];
    float a = 0.0f;
    #pragma unroll
    for (uint32_t sidx = 0; sidx < V; ++sidx) {
        const uint32_t i = tid + sidx * nth;
        if (i < n4) {
            const float4 v = x4[i], wv = w4[i];
            r[sidx].x = v.x * inv * wv.x; r[sidx].y = v.y * inv * wv.y;
            r[sidx].z = v.z * inv * wv.z; r[sidx].w = v.w * inv * wv.w;
            o4[i] = r[sidx];
            a = fmaxf(a, fmaxf(fmaxf(fabsf(r[sidx].x), fabsf(r[sidx].y)), fmaxf(fabsf(r[sidx].z), fabsf(r[sidx].w))));
        }
    }
    for (uint32_t sh = 16; sh > 0; sh >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, sh));
    if (lane == 0) wmax[warp] = a;
    __syncthreads();
    if (tid == 0) {
        float m = 0.0f;
        for (uint32_t wi = 0; wi < ((nth + 31u) >> 5); ++wi) m = fmaxf(m, wmax[wi]);
        int e = 0;
        if (m > 0.0f) { int ex; float fr = frexpf(m, &ex); e = ex - 9 + (fr > 0.875f ? 1 : 0); }
        s_e = e;
        rscale[b] = ldexpf(1.0f, e);
    }
    __syncthreads();
    const float qi = ldexpf(1.0f, -s_e);
    unsigned char* qr = q + (size_t)b * n;
    #pragma unroll
    for (uint32_t sidx = 0; sidx < V; ++sidx) {
        const uint32_t i = tid + sidx * nth;
        if (i < n4) {
            uchar4 o;
            o.x = __nv_fp8_e4m3(r[sidx].x * qi).__x; o.y = __nv_fp8_e4m3(r[sidx].y * qi).__x;
            o.z = __nv_fp8_e4m3(r[sidx].z * qi).__x; o.w = __nv_fp8_e4m3(r[sidx].w * qi).__x;
            *(uchar4*)(qr + (size_t)i * 4u) = o;
        }
    }
}

// The FFN-site twin of pd_add_rmsnorm_batch_kernel (x += pscale*proj in place,
// f32 sumsq, 1/sqrtf(sum/n+eps)) with the same register-held quant tail.
template <uint32_t V>
__global__ void __launch_bounds__(1024) pd_add_rmsnorm_scaled_quant_e4m3_row_kernel(
        float* __restrict__ x, const float* __restrict__ proj, const float* __restrict__ w,
        float* __restrict__ xn, unsigned char* __restrict__ q, float* __restrict__ rscale,
        uint32_t n, float eps, float pscale) {
    PD_PDL_ARM();
    const uint32_t b = blockIdx.x, tid = threadIdx.x, nth = blockDim.x;
    float* xb = x + (size_t)b * n;
    const float* pb = proj + (size_t)b * n;
    __shared__ float wsum[32];
    __shared__ float s_inv;
    __shared__ float wmax[32];
    __shared__ int s_e;
    float acc = 0.0f;
    const uint32_t n4 = n >> 2;
    float4* x4 = reinterpret_cast<float4*>(xb);
    const float4* p4 = reinterpret_cast<const float4*>(pb);
    for (uint32_t i = tid; i < n4; i += nth) {
        float4 v = x4[i];
        const float4 pv = p4[i];
        v.x = fmaf(pscale, pv.x, v.x); v.y = fmaf(pscale, pv.y, v.y);
        v.z = fmaf(pscale, pv.z, v.z); v.w = fmaf(pscale, pv.w, v.w);
        x4[i] = v;
        acc += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    for (uint32_t sh = 16; sh > 0; sh >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, sh);
    const uint32_t warp = tid >> 5, lane = tid & 31u;
    if (lane == 0) wsum[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        float sum = 0.0f;
        const uint32_t nwarps = (nth + 31u) >> 5;
        for (uint32_t wi = 0; wi < nwarps; ++wi) sum += wsum[wi];
        s_inv = 1.0f / sqrtf(sum / (float)n + eps);
    }
    __syncthreads();
    const float inv = s_inv;
    // the unfused kernel's write walk is scalar (`xb[i] * inv * w[i]`): the
    // float4 form multiplies the same three operands in the same order per
    // element, so the values are identical
    const float4* w4 = reinterpret_cast<const float4*>(w);
    float4* o4 = reinterpret_cast<float4*>(xn + (size_t)b * n);
    float4 r[V];
    float a = 0.0f;
    #pragma unroll
    for (uint32_t sidx = 0; sidx < V; ++sidx) {
        const uint32_t i = tid + sidx * nth;
        if (i < n4) {
            const float4 v = x4[i], wv = w4[i];
            r[sidx].x = v.x * inv * wv.x; r[sidx].y = v.y * inv * wv.y;
            r[sidx].z = v.z * inv * wv.z; r[sidx].w = v.w * inv * wv.w;
            o4[i] = r[sidx];
            a = fmaxf(a, fmaxf(fmaxf(fabsf(r[sidx].x), fabsf(r[sidx].y)), fmaxf(fabsf(r[sidx].z), fabsf(r[sidx].w))));
        }
    }
    for (uint32_t sh = 16; sh > 0; sh >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, sh));
    if (lane == 0) wmax[warp] = a;
    __syncthreads();
    if (tid == 0) {
        float m = 0.0f;
        for (uint32_t wi = 0; wi < ((nth + 31u) >> 5); ++wi) m = fmaxf(m, wmax[wi]);
        int e = 0;
        if (m > 0.0f) { int ex; float fr = frexpf(m, &ex); e = ex - 9 + (fr > 0.875f ? 1 : 0); }
        s_e = e;
        rscale[b] = ldexpf(1.0f, e);
    }
    __syncthreads();
    const float qi = ldexpf(1.0f, -s_e);
    unsigned char* qr = q + (size_t)b * n;
    #pragma unroll
    for (uint32_t sidx = 0; sidx < V; ++sidx) {
        const uint32_t i = tid + sidx * nth;
        if (i < n4) {
            uchar4 o;
            o.x = __nv_fp8_e4m3(r[sidx].x * qi).__x; o.y = __nv_fp8_e4m3(r[sidx].y * qi).__x;
            o.z = __nv_fp8_e4m3(r[sidx].z * qi).__x; o.w = __nv_fp8_e4m3(r[sidx].w * qi).__x;
            *(uchar4*)(qr + (size_t)i * 4u) = o;
        }
    }
}

template <int ACC>
static int pd_nq_launch(const float* x, const float* w, float* xn, unsigned char* q, float* rs,
                        uint32_t n, float eps, uint32_t batch, uint32_t V, cudaStream_t st) {
    if (V == 1u) pd_pdl_go(pd_rmsnorm_quant_e4m3_row_kernel<ACC, 1u>, batch, 1024u, 0u, st, x, w, xn, q, rs, n, eps);
    else if (V == 2u) pd_pdl_go(pd_rmsnorm_quant_e4m3_row_kernel<ACC, 2u>, batch, 1024u, 0u, st, x, w, xn, q, rs, n, eps);
    else pd_pdl_go(pd_rmsnorm_quant_e4m3_row_kernel<ACC, 4u>, batch, 1024u, 0u, st, x, w, xn, q, rs, n, eps);
    return pd_launch_status();
}

// (x, w, xn, q, rscale, n, eps, batch, stream); 100 = declined (engine falls back)
PD_EXPORT
int pd_rmsnorm_quant_e4m3_row(const void* x, const void* w, void* xn, void* q, void* rscale,
                              uint32_t n, float eps, uint32_t batch, void* stream) {
    if (n == 0 || batch == 0) return 0;
    if (batch >= 64u || (n & 31u) || n > 16384u || pd_norm_decode_nth() != 1024u) return 100;
    const uint32_t n4 = n >> 2;
    const uint32_t V = n4 <= 1024u ? 1u : (n4 <= 2048u ? 2u : 4u);
    const int accm = pd_norm_acc_mode();
    if (accm == PD_ACC_DF)
        return pd_nq_launch<PD_ACC_DF>((const float*)x, (const float*)w, (float*)xn, (unsigned char*)q, (float*)rscale, n, eps, batch, V, (cudaStream_t)stream);
    if (accm == PD_ACC_F64)
        return pd_nq_launch<PD_ACC_F64>((const float*)x, (const float*)w, (float*)xn, (unsigned char*)q, (float*)rscale, n, eps, batch, V, (cudaStream_t)stream);
    return pd_nq_launch<PD_ACC_F32>((const float*)x, (const float*)w, (float*)xn, (unsigned char*)q, (float*)rscale, n, eps, batch, V, (cudaStream_t)stream);
}

// (x, proj, w, xn, q, rscale, n, eps, pscale, batch, stream); 100 = declined
PD_EXPORT
int pd_add_rmsnorm_scaled_quant_e4m3_row(void* x, const void* proj, const void* w, void* xn, void* q,
                                         void* rscale, uint32_t n, float eps, float pscale,
                                         uint32_t batch, void* stream) {
    if (n == 0 || batch == 0) return 0;
    if (batch >= 64u || (n & 31u) || n > 16384u || pd_norm_decode_nth() != 1024u) return 100;
    const uint32_t n4 = n >> 2;
    const uint32_t V = n4 <= 1024u ? 1u : (n4 <= 2048u ? 2u : 4u);
    cudaStream_t st = (cudaStream_t)stream;
    if (V == 1u) pd_pdl_go(pd_add_rmsnorm_scaled_quant_e4m3_row_kernel<1u>, batch, 1024u, 0u, st, (float*)x, (const float*)proj, (const float*)w, (float*)xn, (unsigned char*)q, (float*)rscale, n, eps, pscale);
    else if (V == 2u) pd_pdl_go(pd_add_rmsnorm_scaled_quant_e4m3_row_kernel<2u>, batch, 1024u, 0u, st, (float*)x, (const float*)proj, (const float*)w, (float*)xn, (unsigned char*)q, (float*)rscale, n, eps, pscale);
    else pd_pdl_go(pd_add_rmsnorm_scaled_quant_e4m3_row_kernel<4u>, batch, 1024u, 0u, st, (float*)x, (const float*)proj, (const float*)w, (float*)xn, (unsigned char*)q, (float*)rscale, n, eps, pscale);
    return pd_launch_status();
}


// fp8-native chain PDL: every f8row GEMM / K-split combine /
// rope-append in this file launched PLAIN, outside the cascade -- the
// c8 tick pays ~400 un-overlapped launch boundaries. The kernels now arm
// (trigger at top, then wait = full predecessor completion => bit-identical by
// construction) and launch as programmatic dependents. Kill (A/B only):
// PADDOCK_NO_F8R_PDL (plain launches; the arm is a no-op under them).
template <typename K, typename... Args>
// Args by const reference, not by value: a CUtensorMap argument is 128-aligned
// under /Zc:__cplusplus and MSVC refuses an over-aligned BY-VALUE parameter
// (C2719); the launch expression below still copies it into the kernel's
// PdTmap parameter (tma_desc.cuh).
static inline void pd_f8r_go(K kern, dim3 grid, dim3 block, uint32_t smem,
                             cudaStream_t st, const Args&... args) {
    static const bool off = pd_env("PADDOCK_NO_F8R_PDL") != nullptr;
    if (off) { kern<<<grid, block, smem, st>>>(args...); return; }
    pd_pdl_go(kern, grid, block, smem, st, args...);
}

// Two-segment decode GEMM: gate|up as one grid over two f8row
// planes of the same (in_dim, out_dim). Takes the mma decode arm's shapes
// (batch 2..64) UNSPLIT only; declines (returns 100) when the arm would
// K-split or the shape is not the mma tile's, and the caller falls back to
// two single GEMMs. Kill: PADDOCK_NO_F8R_GEMM2.
PD_EXPORT
int pd_f8row_gemm2(const void* d0, const void* w0, const void* d1, const void* w1,
                   const void* xq, const void* xrs, void* y0, void* y1,
                   uint32_t in_dim, uint32_t out_dim, uint32_t batch, void* stream) {
    // batch <= 16 only (interleaved A/B): the unsplit two-segment
    // grid wins on the BN=16 tile at r=8 (c8 +1.7%, ITL 8.5 -> 8.3) and loses
    // to the K-split path on the BN=32 tile at r=32 (c32 -2.2%).
    if (batch < 2u || batch > 32u || (out_dim & 15u) || ((in_dim >> 5) & 1u)) return 100;
    static const bool off = pd_env("PADDOCK_NO_F8R_GEMM2") != nullptr
        || pd_env("PADDOCK_NO_F8R_MMA64") != nullptr;
    // batch 17..32 by FILL (from the 30b c32 decode ledger): the unsplit
    // two-segment grid lost at r=32 on 8b because its 2 x 200 row tiles do not
    // fill the die against the K-split path; 30b's 2 x 512 tiles do. Admit
    // r > 16 only when the two-plane grid covers >= 2 CTAs per SM.
    // Kill for the band: PADDOCK_NO_F8R_GEMM2_FILL.
    if (batch > 16u) {
        static int nsm2 = 0;
        if (nsm2 == 0) { int d = 0; cudaGetDevice(&d); cudaDeviceGetAttribute(&nsm2, cudaDevAttrMultiProcessorCount, d); if (nsm2 <= 0) nsm2 = 148; }
        static const bool fill_off = pd_env("PADDOCK_NO_F8R_GEMM2_FILL") != nullptr;
        if (fill_off || 2u * ((out_dim + 63u) / 64u) < 2u * (uint32_t)nsm2) return 100;
    }
    if (off) return 100;
    cudaStream_t st = (cudaStream_t)stream;
    const uint32_t mtiles = (out_dim + 63u) / 64u;
    dim3 g(2u * mtiles, 1u, 1u);
    const unsigned char* dp0 = (const unsigned char*)d0; const float* wp0 = (const float*)w0;
    const unsigned char* dp1 = (const unsigned char*)d1; const float* wp1 = (const float*)w1;
    const unsigned char* xqp = (const unsigned char*)xq; const float* xsp = (const float*)xrs;
    if (batch <= 16u)
        pd_f8r_go(pd_f8row_gemm_mma2_kernel<16u>, g, 256, 0, st, dp0, wp0, (float*)y0, mtiles,
                  dp1, wp1, (float*)y1, xqp, xsp, in_dim, out_dim, batch);
    else if (batch <= 32u)
        pd_f8r_go(pd_f8row_gemm_mma2_kernel<32u>, g, 256, 0, st, dp0, wp0, (float*)y0, mtiles,
                  dp1, wp1, (float*)y1, xqp, xsp, in_dim, out_dim, batch);
    else
        pd_f8r_go(pd_f8row_gemm_mma2_kernel<64u>, g, 256, 0, st, dp0, wp0, (float*)y0, mtiles,
                  dp1, wp1, (float*)y1, xqp, xsp, in_dim, out_dim, batch);
    return pd_launch_status();
}

PD_EXPORT
int pd_f8row_gemm(const void* data, const void* w_rowscale, const void* xq,
                  const void* x_rowscale, void* y, uint32_t in_dim,
                  uint32_t out_dim, uint32_t batch, void* stream) {
#ifndef PD_BS_HOST
    (void)data; (void)w_rowscale; (void)xq; (void)x_rowscale; (void)y;
    (void)in_dim; (void)out_dim; (void)batch; (void)stream;
    return cudaErrorNotSupported;
#else
    if (out_dim == 0 || batch == 0) return 0;
    if ((in_dim & 31u) != 0) return cudaErrorInvalidValue;
#ifdef PD_TC5_HOST
    // tcgen05 route (this build carries sm_100a SASS): TMA-staged ring +
    // tensor-memory accumulate - ~15x the legacy skeleton's ceiling
    // (prototyped bit-exact at ~31 TF/SM issue rate). Numerically
    // identical class to the legacy path (same e4m3 products, f32
    // accumulate, epilogue scales); kill switch PADDOCK_NO_TC5.
    static const bool tc5 = [] {
        int dev = 0, cc = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&cc, cudaDevAttrComputeCapabilityMajor, dev);
        return cc == 10 && pd_tmap_encode() != nullptr
            && pd_env("PADDOCK_NO_TC5") == nullptr;
    }();
    if (tc5) {
        CUtensorMap wm, ym;
        if (pd_tmap_2d(&wm, data, in_dim, out_dim) &&
            pd_tmap_2d(&ym, xq, in_dim, batch)) {
            constexpr uint32_t TS = 3u;
            const uint32_t smem = 2u * TS * 16384u + 2u * TS * 8u;
            static bool at = false;
            if (!at) {
                cudaFuncSetAttribute((const void*)pd_f8row_gemm_tc5_kt<TS>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
                at = true;
            }
            const uint32_t bp = (batch + 127u) & ~127u;
            const uint32_t nt = ((out_dim + 127u) / 128u) * (bp >> 7);
            static int nsm = 0;
            if (nsm == 0) {
                int dev = 0;
                cudaGetDevice(&dev);
                cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
                if (nsm <= 0) nsm = 128;
            }
            // decode-shaped grids underfill (gate at r<=128 = 168 CTAs on a
            // 296-slot die) -> K-split to ~2 waves; prefill grids keep nz=1
            uint32_t nz = 1u;
            const uint32_t nk_all = (in_dim + 127u) / 128u;
            // batch >= 65: the K-split exists for underfilled PREFILL grids;
            // letting r=1..64 lanes (drafter steps, resume tails) in cost c1
            // ~2% to per-call scratch/combine overhead (v5 sweep)
            if (nt < (uint32_t)nsm * 2u && batch >= 65u) {
                nz = ((uint32_t)nsm * 2u + nt - 1u) / nt;
                if (nz > 8u) nz = 8u;
                const uint32_t max_nz = (nk_all + 2u) / 3u; // >= ~3 slabs each
                if (nz > max_nz) nz = max_nz;
                if (nz < 1u) nz = 1u;
            }
            if (nz > 1u) {
                // grow-once partial planes (nz * y); launchers run under the
                // engine's per-stream serialization, so the statics are safe
                static float* part = nullptr;
                static size_t part_sz = 0;
                const size_t need = (size_t)nz * out_dim * batch * 4u;
                if (need > part_sz) {
                    if (part) cudaFree(part);
                    if (cudaMalloc(&part, need) != cudaSuccess) { part = nullptr; part_sz = 0; }
                    else part_sz = need;
                }
                if (part) {
                    pd_f8row_gemm_tc5_kt<TS><<<dim3(nt, nz), 128, smem, (cudaStream_t)stream>>>(
                        wm, ym, (const float*)w_rowscale, (const float*)x_rowscale,
                        part, in_dim, out_dim, batch);
                    const uint32_t n = out_dim * batch;
                    pd_f8r_go(pd_q8_0_gemm_mma_ks_combine_kernel, (n + 255u) / 256u, 256, 0,
                                                         (cudaStream_t)stream, 
                        part, nullptr, (float*)y, n, nz, out_dim);
                    return pd_launch_status();
                }
            }
            pd_f8row_gemm_tc5_kt<TS><<<nt, 128, smem, (cudaStream_t)stream>>>(
                wm, ym, (const float*)w_rowscale, (const float*)x_rowscale,
                (float*)y, in_dim, out_dim, batch);
            return pd_launch_status();
        }
    }
#endif
    static const uint32_t stages = [] {
        const char* e = pd_env("PADDOCK_F8W8_STAGES");
        uint32_t s = e ? (uint32_t)atoi(e) : 2u;
        return s < 2u ? 2u : (s > 4u ? 4u : s);
    }();
    static bool attr_done = false;
    if (!attr_done) {
        cudaFuncSetAttribute((const void*)pd_f8row_gemm_kt<2u>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(2u * 2u * 128u * PD_BS_W8_ROW));
        cudaFuncSetAttribute((const void*)pd_f8row_gemm_kt<3u>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(2u * 3u * 128u * PD_BS_W8_ROW));
        cudaFuncSetAttribute((const void*)pd_f8row_gemm_kt<4u>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(2u * 4u * 128u * PD_BS_W8_ROW));
        // CT=32 stages 128 W rows + 32 Y columns, not 128+128
        cudaFuncSetAttribute((const void*)pd_f8row_gemm_kt<2u, 32u>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)(2u * (128u + 32u) * PD_BS_W8_ROW));
        attr_done = true;
    }
    const uint32_t smem = 2u * stages * 128u * PD_BS_W8_ROW;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t ntiles = ((out_dim + 127u) / 128u) * (batch_pad >> 7);
    const cudaStream_t st = (cudaStream_t)stream;
    // Decode-band COLUMN tile. At batch<=32 the 128-wide column
    // tile leaves warps 2..7 on columns that do not exist; CT=32 gives every
    // warp 16 real rows and the one live group. Measured on a clean GPU,
    // BIT-EXACT output (rel 0.0e+00 - same scale class, same k
    // order, only tile ownership moves):
    //     in_proj  2688x10304 (81 tiles): b16 26.69 -> 18.46, b32 26.68 -> 18.45
    //     out_proj 4096x2688  (21 tiles): b32 14.36 -> 24.83   <- worse
    // The split is GRID SIZE, and the mechanism is the opposite of the usual
    // one: CT=32 cuts per-CTA work 4x, which a 81-tile grid absorbs and a
    // 21-tile grid cannot - at 0.11 waves each CTA needs more work to cover
    // latency, and halving its parallelism halves bandwidth (894 -> 447
    // GB/s). Threshold is a two-point fit (81 wins, 21 loses); re-measure if
    // a third shape ever lands between them.
    auto* dp = (const unsigned char*)data; auto* wsp = (const float*)w_rowscale;
    auto* xqp = (const unsigned char*)xq; auto* xsp = (const float*)x_rowscale;
    // Decode-band K-split: batch<=64 pads to one 128-wide column
    // tile, so the grid is out_dim/128 CTAs - 21 (out_proj) to 81 (in_proj)
    // on a 188-SM die, measured at 23% of DRAM peak. grid.y K-slabs write
    // per-slab planes into a partials buffer, fixed-order ks_combine sums
    // them (f32 partial regroup - the same class the int8 ks path ships).
    // The buffer may only grow outside graph capture (decode ticks are
    // captured; the bulk-prefill calls that always precede the first capture
    // size it - in_proj's 8*out*64*4 is the high-water mark, and it never
    // moves after a graph has baked it); a capture that finds it short
    // falls back to the unsplit launch. Kill: PADDOCK_NO_F8R_KSPLIT.
    // CT=32 is not DISPATCHED - an isolated win, measured not to transfer.
    // In isolation on a clean GPU it had in_proj b32 at 18.45 us vs the
    // K-split path's 26.68, bit-exact. In the SERVE, with the kernel-name
    // proof from a capture (both instantiations visible):
    //     pd_f8row_gemm_kt<2,32>   19067 launches  mean 29.13 us
    //     pd_f8row_gemm_kt<2,128>  21551 launches  median 18.56 us
    // CT=32 lands at 29.13 - worse than isolation said AND worse than what it
    // replaced. A captured graph under a PDL cascade is not a standalone
    // launch: isolated measurements overstate, and a 3-leg A/B read -0.01%
    // before this capture explained it. The template param stays (default
    // 128, every call site bit-identical) because a test arm instantiates it;
    // the dispatch does not.
    //
    // NOTE for whoever revisits this: the first A/B of CT=32 was a null test -
    // the arm sat after the K-split block, which returns, so it never ran.
    // Presence of the env switch in the pack proved nothing. Prove the KERNEL
    // NAME appears in a capture; a distinct template instantiation makes that
    // free.
    // tw arm: batch >= 65 rides the TMA-ring
    // kernel at nearly every shape/width (probe matrix in the kernel header;
    // the one measured carve: 160+-tile gate/up planes at 257..539 keep kt,
    // 506 vs 447 GB/s-wt at M=384). Maps are built per call - the tc5
    // branch's own precedent - and a failed build falls through to the
    // mcol/kt arms. Kill: PADDOCK_NO_F8R_TW.
#if defined(PD_BS_HOST) || defined(PD_TC5_HOST)
    {
        static const bool tw_off = pd_env("PADDOCK_NO_F8R_TW") != nullptr;
        static const bool tw5_off = pd_env("PADDOCK_NO_F8R_TW5") != nullptr;
        static const bool tw4_off = pd_env("PADDOCK_NO_F8R_TW4") != nullptr;
        const uint32_t mtiles_tw = (out_dim + 63u) / 64u;
        // tw5/tw4 election (cold 12-plane probe): the
        // BM128xBN128 producer-warp tile wins fat-out cells (gate M128
        // 41.7us/1258 GB/s-wt vs tw 58.1) and gate
        // M>=256 (M384 80.5 vs 115.7 -- covers the old kt carve); the S6
        // producer-warp BM64 ring wins long-K narrow-out (down M128 41.5 vs
        // 45.8). BN128 pad waste keeps batch 129..255 on tw/kt; wq M768 and
        // down M>540 stay tw. Kills: PADDOCK_NO_F8R_TW5 / _TW4.
        // 30b shapes: out=32768 tw5 sweeps every batch incl the
        // 160..192 valley (M160 149 vs 178, M128 110.7 vs 131.9) -- fat-out
        // washes out the BN128 pad waste; unconditional above 16384.
        // batch >= 1024 on narrow-out planes (the 1024-row burst
        // tick): tw5+raster beats tw_s3 by 23% on d32k (493 vs 596us) and 8b
        // down (199 vs 264), 17% on wq/wo (76 vs 91); at M=768 tw_s3 still
        // wins (502 vs 525), so the band 541..1023 stays. Kill: PADDOCK_NO_F8R_TW5K.
        static const bool tw5k_off = pd_env("PADDOCK_NO_F8R_TW5K") != nullptr;
        const bool tw5_want = !tw5_off
            && ((out_dim >= 16384u)
                || (out_dim >= 8192u && (batch <= 128u || batch >= 256u))
                || (out_dim < 8192u && batch >= 384u && batch <= 540u)
                || (!tw5k_off && out_dim < 8192u && batch >= 1024u));
        const bool tw4_want = !tw4_off && out_dim < 8192u && in_dim >= 8192u
            && batch <= 128u;
        if (!tw_off && batch >= 65u && (out_dim & 63u) == 0u && (in_dim & 127u) == 0u
            && (tw5_want || tw4_want
                || !(mtiles_tw >= 160u && batch > 256u && batch < 540u))) {
            CUtensorMap wm, xm;
            if (pd_tmap_2d_h64_cached(&wm, data, in_dim, out_dim)
                && pd_tmap_2d_h64_cached(&xm, xq, in_dim, batch)) {
                constexpr uint32_t TW_SMEM = 2u * 3u * 64u * 128u + 1024u;
                constexpr uint32_t TW45_SMEM = 2u * 6u * 64u * 128u + 1024u;
                static int tw_attr = 0;
                if (!tw_attr) {
                    cudaFuncSetAttribute((const void*)pd_f8row_gemm_tw_kernel<3u>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         (int)TW_SMEM);
                    cudaFuncSetAttribute((const void*)pd_f8row_gemm_tw4_kernel<6u>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         (int)TW45_SMEM);
                    cudaFuncSetAttribute((const void*)pd_f8row_gemm_tw5_kernel<3u>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         (int)TW45_SMEM);
                    tw_attr = 1;
                }
                static const bool raster_off = pd_env("PADDOCK_NO_F8R_RASTER") != nullptr;
                // wave rule (probed at M=160/192): the column-fastest
                // linearized grid pays only when the grid spans several waves
                // (the >L2 W plane is then re-read per column tile); on a
                // SINGLE-wave grid (narrow-out at mixed-tick widths: 64 row
                // tiles x 3 col tiles) it only scrambles intra-wave order and
                // cost 25-30% (down M=160 84 vs 65us). Linearize iff tiles >
                // blocks/SM x SMs (tw5: 1/SM, tw: 2/SM). PADDOCK_F8R_RASTER_ALL
                // restores the unconditional raster (A/B).
                static const bool raster_all = pd_env("PADDOCK_F8R_RASTER_ALL") != nullptr;
                static int nsm_r = 0;
                if (nsm_r == 0) { int d = 0; cudaGetDevice(&d); cudaDeviceGetAttribute(&nsm_r, cudaDevAttrMultiProcessorCount, d); if (nsm_r <= 0) nsm_r = 148; }
                if (tw5_want) {
                    const uint32_t nr5 = (out_dim + 127u) / 128u, nc5 = (batch + 127u) / 128u;
                    dim3 tg(nr5, nc5);
                    if (!raster_off && (raster_all || nr5 * nc5 > (uint32_t)nsm_r)) tg = dim3(nr5 * nc5, 1u);
                    pd_f8r_go(pd_f8row_gemm_tw5_kernel<3u>, tg, 544, TW45_SMEM, st, 
                        wm, xm, (const float*)w_rowscale, (const float*)x_rowscale,
                        (float*)y, in_dim, out_dim, batch);
                } else if (tw4_want) {
                    dim3 tg(mtiles_tw, (batch + 63u) / 64u);
                    pd_f8r_go(pd_f8row_gemm_tw4_kernel<6u>, tg, 288, TW45_SMEM, st, 
                        wm, xm, (const float*)w_rowscale, (const float*)x_rowscale,
                        (float*)y, in_dim, out_dim, batch);
                } else {
                    const uint32_t ncw = (batch + 63u) / 64u;
                    dim3 tg(mtiles_tw, ncw);
                    if (!raster_off && (raster_all || mtiles_tw * ncw > 2u * (uint32_t)nsm_r)) tg = dim3(mtiles_tw * ncw, 1u);
                    pd_f8r_go(pd_f8row_gemm_tw_kernel<3u>, tg, 256, TW_SMEM, st, 
                        wm, xm, (const float*)w_rowscale, (const float*)x_rowscale,
                        (float*)y, in_dim, out_dim, batch);
                }
                return pd_launch_status();
            }
        }
    }
#endif
    // wide-window mcol arm: fold-free mcol as the
    // 65..=192 fallback where the TMA arm declines (ragged dims, no tmap
    // encoder). Own grow-once partials: the decode graphs bake the mma arm's
    // `part` pointer, and this arm is fused-tick/wave-only (always eager),
    // so the buffers must not be shared. Kill: PADDOCK_NO_F8R_MCOL.
    {
        static const bool mcol_off = pd_env("PADDOCK_NO_F8R_MCOL") != nullptr;
        const uint32_t mtiles0 = (out_dim + 63u) / 64u;
        if (!mcol_off && batch >= 65u && batch <= 192u && (out_dim & 15u) == 0u
            && ((in_dim >> 5) & 1u) == 0u && !(mtiles0 >= 160u && batch <= 128u)) {
            static int nsm2 = 0;
            if (nsm2 == 0) {
                int dev = 0;
                cudaGetDevice(&dev);
                cudaDeviceGetAttribute(&nsm2, cudaDevAttrMultiProcessorCount, dev);
                if (nsm2 <= 0) nsm2 = 128;
            }
            const uint32_t n_blocks2 = in_dim >> 5;
            uint32_t nz = ((uint32_t)nsm2 * 2u + mtiles0 - 1u) / mtiles0;
            const uint32_t max_nz = (n_blocks2 + 3u) / 4u;
            if (nz > max_nz) nz = max_nz;
            if (nz > 8u) nz = 8u;
            if (nz < 1u) nz = 1u;
            static float* part2 = nullptr;
            static size_t part2_sz = 0;
            cudaStreamCaptureStatus cap2 = cudaStreamCaptureStatusNone;
            cudaStreamIsCapturing(st, &cap2);
            const size_t want2 = (size_t)8u * out_dim * 192u * 4u;
            if (cap2 == cudaStreamCaptureStatusNone && nz > 1u && want2 > part2_sz) {
                if (part2) cudaFree(part2);
                if (cudaMalloc(&part2, want2) != cudaSuccess) { part2 = nullptr; part2_sz = 0; }
                else part2_sz = want2;
            }
            if (nz >= 2u && (!part2 || (size_t)nz * out_dim * batch * 4u > part2_sz))
                nz = 1u;
            constexpr uint32_t KPADm = PD_MMA_KT + 16u;
            constexpr uint32_t SM2 = 2u * ((64u * KPADm + 2u * 64u * KPADm + 15u) & ~15u);
            constexpr uint32_t SM3 = 2u * ((64u * KPADm + 3u * 64u * KPADm + 15u) & ~15u);
            static int mcol_attr = 0;
            if (!mcol_attr) {
                cudaFuncSetAttribute((const void*)pd_f8row_gemm_mcol_kernel<2u>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SM2);
                cudaFuncSetAttribute((const void*)pd_f8row_gemm_mcol_kernel<3u>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SM3);
                mcol_attr = 1;
            }
            float* dst2 = nz > 1u ? part2 : (float*)y;
            auto* dp2 = (const unsigned char*)data;
            auto* wsp2 = (const float*)w_rowscale;
            auto* xqp2 = (const unsigned char*)xq;
            auto* xsp2 = (const float*)x_rowscale;
            dim3 mg(mtiles0, 1u, nz);
            if (batch <= 128u)
                pd_f8row_gemm_mcol_kernel<2u><<<mg, 256, SM2, st>>>(
                    dp2, wsp2, xqp2, xsp2, dst2, in_dim, out_dim, batch);
            else
                pd_f8row_gemm_mcol_kernel<3u><<<mg, 256, SM3, st>>>(
                    dp2, wsp2, xqp2, xsp2, dst2, in_dim, out_dim, batch);
            if (nz > 1u) {
                const uint32_t n = out_dim * batch;
                pd_f8r_go(pd_q8_0_gemm_mma_ks_combine_kernel, (n + 255u) / 256u, 256, 0, st, 
                    part2, nullptr, (float*)y, n, nz, out_dim);
            }
            return pd_launch_status();
        }
    }
    {
        static float* part = nullptr;
        static size_t part_sz = 0;
        static int nsm = 0;
        if (nsm == 0) {
            int dev = 0;
            cudaGetDevice(&dev);
            cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
            if (nsm <= 0) nsm = 128;
        }
        static const bool off = pd_env("PADDOCK_NO_F8R_KSPLIT") != nullptr;
        if (!off) {
            cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
            cudaStreamIsCapturing(st, &cap);
            const size_t want = (size_t)8u * out_dim * 64u * 4u;
            if (cap == cudaStreamCaptureStatusNone && want > part_sz) {
                if (part) cudaFree(part);
                if (cudaMalloc(&part, want) != cudaSuccess) { part = nullptr; part_sz = 0; }
                else part_sz = want;
            }
            // mma-tile decode arm (mamba proj rung,
            // first in the ladder - the CT=32 null-A/B trap): the BM=64
            // f32-scale twin pd_f8row_gemm_mma_kernel. Rung 1 priced the
            // tile at in_proj 20.49 vs 26.81 us / out_proj 10.27 vs 14.36
            // (isolated, other scale class, timing only); this is that shape
            // on the f8row class itself - raw mma accumulate, scales a pure
            // epilogue, bit-identical to the kt kernel at nz=1. grid.z
            // slabs share the partials buffer + ks_combine above. Kill:
            // PADDOCK_NO_F8R_MMA64 (falls through to the kt K-split).
            // tw4dec: the tw4 producer-warp 6-stage TMA
            // ring at BN=32, UNSPLIT, for every narrow-out plane at batch 2..64
            // (probe PD_DEC=1 M=32 cold: wo 14.4 vs 17.5 us, qkv 20.2 vs 21.4,
            // down 40.4 vs 45.1, d32k 93.9 vs 112.3 = 1429 GB/s, i.e. at the
            // stream ceiling; ties fat gu). Its K-split arms lose: the deep ring already
            // saturates 64 CTAs. Bit-exact vs the grid.y launch (probe). Falls
            // through to the next decode arm. Kill: PADDOCK_NO_F8R_TW4DEC.
            static const bool tw4dec_off = pd_env("PADDOCK_NO_F8R_TW4DEC") != nullptr;
            if (!tw4dec_off && batch >= 2u && batch <= 64u && out_dim < 16384u
                && (out_dim & 63u) == 0u && (in_dim & 127u) == 0u) {
                constexpr uint32_t T4_S = 6u, T4_BN = 32u;
                constexpr uint32_t T4_SMEM = T4_S * (64u * 128u + T4_BN * 128u) + 1024u;
                static int t4_attr = 0;
                if (!t4_attr) {
                    cudaFuncSetAttribute((const void*)pd_f8row_gemm_tw4d_kernel<T4_S, T4_BN>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)T4_SMEM);
                    t4_attr = 1;
                }
                CUtensorMap wm4, xm4;
                if (pd_tmap_2d_h64_cached(&wm4, dp, in_dim, out_dim)
                    && pd_tmap_2d_hN_cached(&xm4, xqp, in_dim, batch, T4_BN)) {
                    dim3 tg((out_dim + 63u) / 64u, (batch + T4_BN - 1u) / T4_BN);
                    pd_f8r_go(pd_f8row_gemm_tw4d_kernel<T4_S, T4_BN>, tg, 288, T4_SMEM, st,
                        wm4, xm4, wsp, xsp, (float*)y, in_dim, out_dim, batch);
                    return pd_launch_status();
                }
            }
            static const bool mma_off = pd_env("PADDOCK_NO_F8R_MMA64") != nullptr;
            if (!mma_off && batch >= 2u && batch <= 64u && (out_dim & 15u) == 0u
                && ((in_dim >> 5) & 1u) == 0u) {
                const uint32_t mtiles = (out_dim + 63u) / 64u;
                const uint32_t n_blocks = in_dim >> 5;
                uint32_t nz = ((uint32_t)nsm * 2u + mtiles - 1u) / mtiles;
                const uint32_t max_nz = (n_blocks + 3u) / 4u;
                if (nz > max_nz) nz = max_nz;
                if (nz > 8u) nz = 8u;
                if (nz < 1u) nz = 1u;
                if (nz >= 2u && (!part || (size_t)nz * out_dim * batch * 4u > part_sz))
                    nz = 1u;   // partials don't fit: unsplit is still the better tile
                float* dst = nz > 1u ? part : (float*)y;
                dim3 mgrid(mtiles, 1u, nz);
                if (batch <= 16u)
                    pd_f8r_go(pd_f8row_gemm_mma_kernel<16u>, mgrid, 256, 0, st, 
                        dp, wsp, xqp, xsp, dst, in_dim, out_dim, batch);
                else if (batch <= 32u)
                    pd_f8r_go(pd_f8row_gemm_mma_kernel<32u>, mgrid, 256, 0, st, 
                        dp, wsp, xqp, xsp, dst, in_dim, out_dim, batch);
                else
                    pd_f8r_go(pd_f8row_gemm_mma_kernel<64u>, mgrid, 256, 0, st, 
                        dp, wsp, xqp, xsp, dst, in_dim, out_dim, batch);
                if (nz > 1u) {
                    const uint32_t n = out_dim * batch;
                    pd_f8r_go(pd_q8_0_gemm_mma_ks_combine_kernel, (n + 255u) / 256u, 256, 0, st, 
                        part, nullptr, (float*)y, n, nz, out_dim);
                }
                return pd_launch_status();
            }
            if (batch >= 2u && batch <= 64u && ntiles < (uint32_t)nsm * 2u) {
                uint32_t nz = ((uint32_t)nsm * 2u + ntiles - 1u) / ntiles;
                const uint32_t nk_all = (in_dim + 63u) / 64u;
                const uint32_t max_nz = nk_all / 6u; // keep >= ~6 K-steps/slab
                if (nz > max_nz) nz = max_nz;
                if (nz > 8u) nz = 8u;
                if (nz >= 2u && part && (size_t)nz * out_dim * batch * 4u <= part_sz) {
                    dim3 grid(ntiles, nz);
                    if (stages == 4u)
                        pd_f8r_go(pd_f8row_gemm_kt<4u>, grid, 256, smem, st, dp, wsp, xqp, xsp, part, in_dim, out_dim, batch);
                    else if (stages == 3u)
                        pd_f8r_go(pd_f8row_gemm_kt<3u>, grid, 256, smem, st, dp, wsp, xqp, xsp, part, in_dim, out_dim, batch);
                    else
                        pd_f8r_go(pd_f8row_gemm_kt<2u>, grid, 256, smem, st, dp, wsp, xqp, xsp, part, in_dim, out_dim, batch);
                    const uint32_t n = out_dim * batch;
                    pd_f8r_go(pd_q8_0_gemm_mma_ks_combine_kernel, (n + 255u) / 256u, 256, 0, st, 
                        part, nullptr, (float*)y, n, nz, out_dim);
                    return pd_launch_status();
                }
            }
        }
    }
    if (stages == 4u)
        pd_f8r_go(pd_f8row_gemm_kt<4u>, ntiles, 256, smem, st, dp, wsp, xqp, xsp, (float*)y, in_dim, out_dim, batch);
    else if (stages == 3u)
        pd_f8r_go(pd_f8row_gemm_kt<3u>, ntiles, 256, smem, st, dp, wsp, xqp, xsp, (float*)y, in_dim, out_dim, batch);
    else
        pd_f8r_go(pd_f8row_gemm_kt<2u>, ntiles, 256, smem, st, dp, wsp, xqp, xsp, (float*)y, in_dim, out_dim, batch);
    return pd_launch_status();
#endif
}


// ---- granite fused wqkv (f8row class) -------------------------------------
// one f8row mma over the load-time-concatenated q|k|v plane (out_dim =
// (n_heads + 2 n_kv) * head_dim) into K-split partial planes, then a single
// combine + NORM-rope + paged K/V append kernel. Replaces three underfilled
// GEMMs (wq 64 / wk 16 / wv 16 tiles read 25.2 MB at ~787 GB/s effective on
// the c32 tick) + the separate rope+append launch. The combine sums the nz
// z-planes in FIXED ascending order (pd_q8_0_gemm_mma_ks_combine's math on
// FINAL-VALUED f8row partials - the mma epilogue scales each slab) and then
// runs pd_rope_norm_qk_append_paged_kernel's NORM-convention body verbatim:
// pairs (2k, 2k+1), granite's rule (NEOX here = fluent text that degrades
// with position), roped k straight to the pool, v appended unrotated.
template <typename KV>
__global__ void pd_f8row_ks_rope_norm_qkv_append_paged_kernel(
    const float* __restrict__ part, float* __restrict__ q_out,
    KV* __restrict__ k_pool, KV* __restrict__ v_pool,
    const unsigned int* __restrict__ positions,
    const unsigned int* __restrict__ slots,
    const uint32_t* __restrict__ block_tables, uint32_t blocks_per_slot,
    uint32_t n_heads, uint32_t n_kv, uint32_t head_dim,
    float theta_scale, float freq_scale, float corr_low, float corr_high,
    float ext_factor, float mscale, uint32_t batch, uint32_t nz,
    float part_scale) {
    PD_PDL_ARM();
    const uint32_t qdim = n_heads * head_dim, kv_dim = n_kv * head_dim;
    const uint32_t rowd = qdim + 2u * kv_dim;
    const uint32_t npl = rowd * batch;
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t nslots = n_heads + 2u * n_kv;
    const uint32_t idx = blockIdx.x * (blockDim.x >> 5) + warp;
    if (idx >= batch * nslots) return;
    const uint32_t b = idx / nslots, si = idx % nslots;
    const uint32_t pos = positions[b];
    const uint32_t slot = slots ? slots[b] : b;
    auto sum_part = [&](uint32_t col) {
        float a = 0.0f;
        for (uint32_t z = 0; z < nz; ++z)
            a += part[(size_t)z * npl + (size_t)b * rowd + col];
        // per-tensor epilogue scale after the fixed-order fold -- the nvf4
        // raw-partials route (pd_nvf4_sk_reduce's exact order: fold, then
        // scale2); 1.0f on the f8row routes, where the mma epilogue already
        // scaled each slab (x * 1.0f is exact, so those stay bit-identical)
        return a * part_scale;
    };
    if (si >= n_heads + n_kv) {
        // v: combine + plain paged append
        const uint32_t h = si - n_heads - n_kv;
        const uint32_t blk = block_tables[(size_t)slot * blocks_per_slot + (pos >> 4)];
        KV* dst = v_pool + (size_t)blk * 16u * kv_dim + (size_t)(pos & 15u) * kv_dim
                + (size_t)h * head_dim;
        const uint32_t base = qdim + kv_dim + h * head_dim;
        for (uint32_t i = lane; i < head_dim; i += 32u) pd_kv_store(&dst[i], sum_part(base + i));
        return;
    }
    const bool is_k = si >= n_heads;
    const uint32_t h = is_k ? si - n_heads : si;
    const uint32_t base = (is_k ? qdim : 0u) + h * head_dim;
    KV* kdst = nullptr;
    if (is_k) {
        const uint32_t blk = block_tables[(size_t)slot * blocks_per_slot + (pos >> 4)];
        kdst = k_pool + (size_t)blk * 16u * kv_dim + (size_t)(pos & 15u) * kv_dim
             + (size_t)h * head_dim;
    }
    float* qdst = q_out + (size_t)b * qdim + (size_t)h * head_dim;
    const uint32_t half = head_dim / 2u;
    // per-warp theta chain - pd_rope_norm_qk_append_paged_kernel's, verbatim
    float theta = (float)pos;
    for (uint32_t i = 0; i < lane && i < half; ++i) theta *= theta_scale;
    for (uint32_t kk = lane; kk < half; kk += 32u) {
        float yv = ((float)kk - corr_low) / fmaxf(0.001f, corr_high - corr_low);
        float ramp = (1.0f - fminf(1.0f, fmaxf(0.0f, yv))) * ext_factor;
        float angle = (freq_scale * theta) * (1.0f - ramp) + theta * ramp;
        float sn = sinf(angle) * mscale;
        float cs = cosf(angle) * mscale;
        const uint32_t i0 = 2u * kk, i1 = 2u * kk + 1u;
        const float a = sum_part(base + i0);
        const float bb = sum_part(base + i1);
        const float r0 = a * cs - bb * sn;
        const float r1 = a * sn + bb * cs;
        if (is_k) {
            pd_kv_store(&kdst[i0], r0);
            pd_kv_store(&kdst[i1], r1);
        } else {
            qdst[i0] = r0;
            qdst[i1] = r1;
        }
        for (uint32_t i = 0; i < 32 && kk + i < half; ++i) theta *= theta_scale;
    }
}

// Rope-only twin: consume an ALREADY-COMPUTED fused-qkv plane (y layout,
// [token][qdim|kdim|vdim] col-major = the nz=1 partials layout) with the same
// combine+NORM-rope+paged-append kernel. The pf-side fused-qkv rung: the tw
// GEMM writes the fused plane into scratch at chunk widths (batch is not
// capped at 64 here - the kernel's grid is batch x head-slots warps).
PD_EXPORT
int pd_f8row_qkv_rope_norm_from_y_paged(
    const void* part, void* q_out, void* k_pool, void* v_pool,
    const void* positions, const void* slots, const void* block_tables,
    uint32_t blocks_per_slot, uint32_t n_heads, uint32_t n_kv,
    uint32_t head_dim, float theta_scale, float freq_scale, float corr_low,
    float corr_high, float ext_factor, float mscale, uint32_t batch,
    uint32_t kv_dtype, void* stream) {
#ifndef PD_BS_HOST
    (void)part; (void)q_out; (void)k_pool; (void)v_pool; (void)positions;
    (void)slots; (void)block_tables; (void)blocks_per_slot; (void)n_heads;
    (void)n_kv; (void)head_dim; (void)theta_scale; (void)freq_scale;
    (void)corr_low; (void)corr_high; (void)ext_factor; (void)mscale;
    (void)batch; (void)kv_dtype; (void)stream;
    return cudaErrorNotSupported;
#else
    if (batch == 0 || n_heads == 0 || n_kv == 0) return 0;
    auto st = (cudaStream_t)stream;
    const uint32_t warps = batch * (n_heads + 2u * n_kv);
    const uint32_t blocks = (warps * 32u + 255u) / 256u;
    if (kv_dtype == PD_KV_FP8_E4M3)
        pd_f8r_go(pd_f8row_ks_rope_norm_qkv_append_paged_kernel<__nv_fp8_e4m3>, blocks, 256, 0, st, 
            (const float*)part, (float*)q_out, (__nv_fp8_e4m3*)k_pool,
            (__nv_fp8_e4m3*)v_pool, (const unsigned int*)positions,
            (const unsigned int*)slots, (const uint32_t*)block_tables,
            blocks_per_slot, n_heads, n_kv, head_dim, theta_scale, freq_scale,
            corr_low, corr_high, ext_factor, mscale, batch, 1u, 1.0f);
    else
        pd_f8r_go(pd_f8row_ks_rope_norm_qkv_append_paged_kernel<__half>, blocks, 256, 0, st, 
            (const float*)part, (float*)q_out, (__half*)k_pool, (__half*)v_pool,
            (const unsigned int*)positions, (const unsigned int*)slots,
            (const uint32_t*)block_tables, blocks_per_slot, n_heads, n_kv,
            head_dim, theta_scale, freq_scale, corr_low, corr_high, ext_factor,
            mscale, batch, 1u, 1.0f);
    return pd_launch_status();
#endif
}

// Partials twin (granite NVFP4 fused qkv): consume `nz` RAW
// K-split partial planes ([z][token][qdim|kdim|vdim], slice stride
// batch*rowd -- pd_nvf4_gemm_f4c_raw's layout) and apply the checkpoint's
// per-tensor scale after the fixed-order fold. With nz=1, part_scale=1.0f this
// is pd_f8row_qkv_rope_norm_from_y_paged. Replaces pd_nvf4_sk_reduce + the
// separate rope+append launch on the nvf4 decode band: the fold that used to
// write y and the rope that re-read it become one pass over the partials.
PD_EXPORT
int pd_qkv_rope_norm_from_parts_paged(
    const void* part, uint32_t nz, float part_scale, void* q_out, void* k_pool,
    void* v_pool, const void* positions, const void* slots,
    const void* block_tables, uint32_t blocks_per_slot, uint32_t n_heads,
    uint32_t n_kv, uint32_t head_dim, float theta_scale, float freq_scale,
    float corr_low, float corr_high, float ext_factor, float mscale,
    uint32_t batch, uint32_t kv_dtype, void* stream) {
#ifndef PD_BS_HOST
    (void)part; (void)nz; (void)part_scale; (void)q_out; (void)k_pool;
    (void)v_pool; (void)positions; (void)slots; (void)block_tables;
    (void)blocks_per_slot; (void)n_heads; (void)n_kv; (void)head_dim;
    (void)theta_scale; (void)freq_scale; (void)corr_low; (void)corr_high;
    (void)ext_factor; (void)mscale; (void)batch; (void)kv_dtype; (void)stream;
    return cudaErrorNotSupported;
#else
    if (batch == 0 || n_heads == 0 || n_kv == 0) return 0;
    if (nz == 0) return cudaErrorInvalidValue;
    auto st = (cudaStream_t)stream;
    const uint32_t warps = batch * (n_heads + 2u * n_kv);
    const uint32_t blocks = (warps * 32u + 255u) / 256u;
    if (kv_dtype == PD_KV_FP8_E4M3)
        pd_f8r_go(pd_f8row_ks_rope_norm_qkv_append_paged_kernel<__nv_fp8_e4m3>, blocks, 256, 0, st,
            (const float*)part, (float*)q_out, (__nv_fp8_e4m3*)k_pool,
            (__nv_fp8_e4m3*)v_pool, (const unsigned int*)positions,
            (const unsigned int*)slots, (const uint32_t*)block_tables,
            blocks_per_slot, n_heads, n_kv, head_dim, theta_scale, freq_scale,
            corr_low, corr_high, ext_factor, mscale, batch, nz, part_scale);
    else
        pd_f8r_go(pd_f8row_ks_rope_norm_qkv_append_paged_kernel<__half>, blocks, 256, 0, st,
            (const float*)part, (float*)q_out, (__half*)k_pool, (__half*)v_pool,
            (const unsigned int*)positions, (const unsigned int*)slots,
            (const uint32_t*)block_tables, blocks_per_slot, n_heads, n_kv,
            head_dim, theta_scale, freq_scale, corr_low, corr_high, ext_factor,
            mscale, batch, nz, part_scale);
    return pd_launch_status();
#endif
}

PD_EXPORT
int pd_f8row_gemm_mma_qkv_norm_paged(
    const void* data, const void* wrs, const void* xq, const void* xrs,
    void* part, void* q_out, void* k_pool, void* v_pool,
    const void* positions, const void* slots, const void* block_tables,
    uint32_t blocks_per_slot, uint32_t in_dim, uint32_t n_heads,
    uint32_t n_kv, uint32_t head_dim, float theta_scale, float freq_scale,
    float corr_low, float corr_high, float ext_factor, float mscale,
    uint32_t batch, uint32_t kv_dtype, void* stream) {
#ifndef PD_BS_HOST
    (void)data; (void)wrs; (void)xq; (void)xrs; (void)part; (void)q_out;
    (void)k_pool; (void)v_pool; (void)positions; (void)slots;
    (void)block_tables; (void)blocks_per_slot; (void)in_dim; (void)n_heads;
    (void)n_kv; (void)head_dim; (void)theta_scale; (void)freq_scale;
    (void)corr_low; (void)corr_high; (void)ext_factor; (void)mscale;
    (void)batch; (void)kv_dtype; (void)stream;
    return cudaErrorNotSupported;
#else
    const uint32_t out_dim = (n_heads + 2u * n_kv) * head_dim;
    if (out_dim == 0 || batch == 0) return 0;
    if ((in_dim & 31u) || (out_dim & 15u)) return cudaErrorInvalidValue;
    if (batch < 2u || batch > 64u) return cudaErrorInvalidValue;
    // the mma body is __CUDA_ARCH__ >= 890 (empty below): refuse, don't no-op
    static const bool cc_ok = [] {
        int dev = 0, cma = 0, cmi = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&cma, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&cmi, cudaDevAttrComputeCapabilityMinor, dev);
        return cma > 8 || (cma == 8 && cmi >= 9);
    }();
    if (!cc_ok) return cudaErrorNotSupported;
    auto st = (cudaStream_t)stream;
    static int nsm = 0;
    if (nsm == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
        if (nsm <= 0) nsm = 128;
    }
    const uint32_t mtiles = (out_dim + 63u) / 64u;
    const uint32_t n_blocks = in_dim >> 5;
    uint32_t nz = ((uint32_t)nsm * 2u + mtiles - 1u) / mtiles;
    const uint32_t max_nz = (n_blocks + 3u) / 4u;
    if (nz > max_nz) nz = max_nz;
    if (nz > 8u) nz = 8u;
    if (nz < 1u) nz = 1u;
    // Always partials - the combine kernel is the single consumer, and at
    // nz=1 it is a pure rope+append read of the one plane
    dim3 mgrid(mtiles, 1u, nz);
    auto* dp = (const unsigned char*)data; auto* wsp = (const float*)wrs;
    auto* xqp = (const unsigned char*)xq; auto* xsp = (const float*)xrs;
    if (batch <= 16u)
        pd_f8r_go(pd_f8row_gemm_mma_kernel<16u>, mgrid, 256, 0, st, dp, wsp, xqp, xsp, (float*)part, in_dim, out_dim, batch);
    else if (batch <= 32u)
        pd_f8r_go(pd_f8row_gemm_mma_kernel<32u>, mgrid, 256, 0, st, dp, wsp, xqp, xsp, (float*)part, in_dim, out_dim, batch);
    else
        pd_f8r_go(pd_f8row_gemm_mma_kernel<64u>, mgrid, 256, 0, st, dp, wsp, xqp, xsp, (float*)part, in_dim, out_dim, batch);
    const uint32_t warps = batch * (n_heads + 2u * n_kv);
    const uint32_t blocks = (warps * 32u + 255u) / 256u;
    if (kv_dtype == PD_KV_FP8_E4M3)
        pd_f8r_go(pd_f8row_ks_rope_norm_qkv_append_paged_kernel<__nv_fp8_e4m3>, blocks, 256, 0, st, 
            (const float*)part, (float*)q_out, (__nv_fp8_e4m3*)k_pool,
            (__nv_fp8_e4m3*)v_pool, (const unsigned int*)positions,
            (const unsigned int*)slots, (const uint32_t*)block_tables,
            blocks_per_slot, n_heads, n_kv, head_dim, theta_scale, freq_scale,
            corr_low, corr_high, ext_factor, mscale, batch, nz, 1.0f);
    else
        pd_f8r_go(pd_f8row_ks_rope_norm_qkv_append_paged_kernel<__half>, blocks, 256, 0, st, 
            (const float*)part, (float*)q_out, (__half*)k_pool, (__half*)v_pool,
            (const unsigned int*)positions, (const unsigned int*)slots,
            (const uint32_t*)block_tables, blocks_per_slot, n_heads, n_kv,
            head_dim, theta_scale, freq_scale, corr_low, corr_high, ext_factor,
            mscale, batch, nz, 1.0f);
    return pd_launch_status();
#endif
}
