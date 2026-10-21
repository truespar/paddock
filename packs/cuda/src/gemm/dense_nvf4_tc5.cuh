// gemm/dense_nvf4_tc5.cuh - NVFP4 x NVFP4 tcgen05 GEMM (sm_100a): the W4A4 arm on B200.
// Textually-included segment of the single pack translation unit; needs only abi.cuh and
// tma_desc.cuh (PdTmap, pd_tmap_2d, PD_TC5_OK, pd_tc5_sdesc), so a bench probe can include
// it without the rest of the pack.
//
// Why this exists. On cc 10.0 the W4A4 family (pd_nvf4_gemm_f4*, warp-level mma.sync
// kind::mxf4nvf4) is NULLed - mma.sync has no fp4 form on datacenter Blackwell - so a
// checkpoint's NVFP4 FFN planes were re-encoded to e4m3 at load and served on the fp8
// tcgen05 lanes: twice the weight bytes per decode step and half the tensor rate of what
// the die does in fp4. The die's fp4 tensor core is tcgen05.mma kind::mxf4nvf4 with
// hardware block scaling (CUTLASS's KernelTmaWarpSpecialized1SmNvf4Sm100 measures it on
// this die - moe/nvf4_sorted.cuh); this is our own implementation of that class, consuming
// the checkpoint planes verbatim (the Nvf4Plane row layout: adjacent-packed nibbles, low
// nibble = even k, flat e4m3-per-16 scale plane) and the activations pd_quantize_nvf4 /
// pd_quantize_nvf4_swiglu write - the same operands, numeric class and epilogue contract
// as pd_nvf4_gemm_f4 (acc * scale2 + bias), so the engine's W4A4 chain serves it unchanged.
//
// Hardware facts this relies on (PTX ISA tcgen05; CUTLASS mma_sm100_desc.hpp as reference):
//   - kind::mxf4nvf4 reads PACKED e2m1 (two per byte) from smem - unlike kind::f8f6f4,
//     whose 4-bit operands sit one per byte. K = 64 per mma = 32 bytes of a K-major row,
//     so the SW128 descriptor walk is the e4m3 kernels' (+2 16B units per mma, 4 per slab).
//   - instruction descriptor: a/b format E2M1 is code 1 in the mxf4/mxf4nvf4 encoding
//     (mxf8f6f4 spells it 5 - a wrong code is an illegal instruction, not a wrong answer),
//     bit 23 = 0 for UE4M3 scales, sf ids 0: scale_vec::4X consumes all four bytes of a
//     scale cell per mma.
//   - scale cells in tensor memory: row m's cell at lane m%32, column sf + m/32, the four
//     bytes = the four 16-element k-blocks of that mma, duplicated to the four 32-lane
//     partitions by tcgen05.cp 32x128b.warpx4 from the canonical 512 B smem tile (lane l's
//     16 bytes at l*16, word c = row c*32+l) - the layout pd_f8bs_gemm_tc5_kt probed for
//     the ue8m0 class, where one tile serves four mmas by byte (sf id); here one tile
//     serves one mma, so a 256-K slab stages four tiles per operand.
//
// Structure (v1, correctness-first): pd_f8bs_gemm_tc5_kt's proven pipeline - TMA SW128
// W/X slabs on an S-deep ring, every thread stages its row's scale cells with 4 B
// cp.asyncs straight from the row-major scale planes into the canonical tiles (no repack,
// no transpose kernel) and arrives noinc on the slot barrier, tid 0 tcgen05.cp's the tiles
// and issues the mmas, commit -> bdone recycles the slot. A = W (M = 128 weight rows),
// B = X (N = 128 activation rows), so a decode-width batch is one column tile and the
// epilogue writes 128 consecutive outputs per activation row. grid.z splits K for the
// planes whose tile grid starves the die (down at decode: 40 tiles on 148 SMs); split
// launches write raw partials and pd_nv4tc5_reduce applies the epilogue in a fixed order.

#if PD_TC5_OK
__device__ __forceinline__ uint32_t pd_nv4tc5_idesc(uint32_t bn) {
    return (1u << 7) | (1u << 10)   // e2m1 x e2m1 (mxf4nvf4 code 1)
         | ((bn >> 3) << 17)        // N
         | (0u << 23)               // UE4M3 scales
         | ((128u >> 4) << 24);     // M = 128; a/b sf ids 0 (scale_vec::4X)
}
__device__ __forceinline__ void pd_nv4tc5_cpa4(void* smem, const void* gmem, bool ok) {
    const unsigned sm = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(sm), "l"(gmem),
                 "r"(ok ? 4u : 0u));
}
#endif

// smem: S x (16 KB W slab + 16 KB X slab + 4 KB scale tiles) + 2 S barriers
#define PD_NV4TC5_SMEM(S) ((S) * (16384u * 2u + 4096u) + 2u * (S) * 8u)

template <uint32_t S, bool RAW>
__global__ void __launch_bounds__(128) pd_nv4tc5_kt(
    const __grid_constant__ PdTmap wmap, const __grid_constant__ PdTmap xmap,
    const unsigned char* __restrict__ wsc, const unsigned char* __restrict__ xsc,
    float* __restrict__ y, float scale2, const float* __restrict__ bias,
    uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
#if PD_TC5_OK
    extern __shared__ __align__(1024) unsigned char pd_nv4tc5_sh[];
    unsigned char* wt = pd_nv4tc5_sh;
    unsigned char* xt = pd_nv4tc5_sh + S * 16384u;
    unsigned char* sfs = pd_nv4tc5_sh + 2u * S * 16384u;   // S x (4 SFA tiles | 4 SFB tiles)
    uint64_t* bfull = (uint64_t*)(sfs + S * 4096u);
    uint64_t* bdone = bfull + S;
    __shared__ uint32_t tmem_slot[1];

    const uint32_t tid = threadIdx.x;
    const uint32_t n_sb = in_dim >> 4;        // scale bytes per row
    const uint32_t nk_all = in_dim >> 8;      // 256-K slabs (launcher: in_dim % 256 == 0)
    const uint32_t per = (nk_all + gridDim.z - 1u) / gridDim.z;
    const uint32_t k0 = blockIdx.z * per;
    const uint32_t nk = k0 < nk_all ? (per < nk_all - k0 ? per : nk_all - k0) : 0u;
    const uint32_t row_base = blockIdx.x * 128u;   // weight rows = D lanes
    const uint32_t col_base = blockIdx.y * 128u;   // activation rows = D columns

    if (tid == 0) {
        #pragma unroll
        for (uint32_t s = 0; s < S; ++s) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 129;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bfull[s])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&bdone[s])));
        }
    }
    __syncthreads();
    // D: cols 0..127; scale ring: S x (16 SFA + 16 SFB) cols from 128
    if (tid < 32)
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 256;"
                     ::"r"((uint32_t)__cvta_generic_to_shared(tmem_slot)));
    __syncthreads();
    const uint32_t tmem = tmem_slot[0];
    const uint32_t sf_base = tmem + 128u;

    auto bar_wait = [&](uint64_t* bar, uint32_t parity) {
        const uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
        asm volatile("{\n\t.reg .pred p;\nW%=:\n\t"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                     "@!p bra W%=;\n\t}" ::"r"(a), "r"(parity));
    };
    auto tma_stage = [&](uint32_t kt, uint32_t s) {
        const uint32_t m = (uint32_t)__cvta_generic_to_shared(&bfull[s]);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 32768;" ::"r"(m));
        const uint32_t wd = (uint32_t)__cvta_generic_to_shared(wt + s * 16384u);
        const uint32_t xd = (uint32_t)__cvta_generic_to_shared(xt + s * 16384u);
        const int ck = (int)(kt * 128u);   // byte column of the slab
        asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                     " [%0], [%1, {%2, %3}], [%4];" ::"r"(wd), "l"(&wmap), "r"(ck),
                     "r"((int)row_base), "r"(m) : "memory");
        asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                     " [%0], [%1, {%2, %3}], [%4];" ::"r"(xd), "l"(&xmap), "r"(ck),
                     "r"((int)col_base), "r"(m) : "memory");
    };
    // this thread owns row `tid` of both scale operands: 4 cells each (one per mma),
    // into the canonical position of tile j
    auto sf_stage = [&](uint32_t kt, uint32_t s) {
        unsigned char* base = sfs + s * 4096u;
        const uint32_t off = (tid % 32u) * 16u + (tid / 32u) * 4u;
        const uint32_t rw = row_base + tid, rc = col_base + tid;
        const unsigned char* ws = wsc + (size_t)rw * n_sb + kt * 16u;
        const unsigned char* xs = xsc + (size_t)rc * n_sb + kt * 16u;
        #pragma unroll
        for (uint32_t j = 0; j < 4u; ++j) {
            pd_nv4tc5_cpa4(base + j * 512u + off, ws + j * 4u, rw < out_dim);
            pd_nv4tc5_cpa4(base + 2048u + j * 512u + off, xs + j * 4u, rc < batch);
        }
        const uint32_t m = (uint32_t)__cvta_generic_to_shared(&bfull[s]);
        asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(m) : "memory");
    };

    #pragma unroll
    for (uint32_t s = 0; s < S; ++s) {
        if (s < nk) {
            sf_stage(k0 + s, s);
            if (tid == 0) tma_stage(k0 + s, s);
        }
    }
    uint32_t fph = 0, dph = 0;
    for (uint32_t kt = 0; kt < nk; ++kt) {
        const uint32_t s = kt % S;
        if (tid == 0) {
            bar_wait(&bfull[s], (fph >> s) & 1u);
            fph ^= 1u << s;
            const uint32_t sfa_t = sf_base + s * 32u, sfb_t = sfa_t + 16u;
            const uint32_t v = (uint32_t)__cvta_generic_to_shared(sfs + s * 4096u) >> 4;
            #pragma unroll
            for (uint32_t j = 0; j < 4u; ++j) {
                // canonical tile descriptor (LBO 1, SBO 8 - probed in pd_f8bs_gemm_tc5_kt)
                const uint64_t da = ((uint64_t)((v + j * 32u) & 0x3FFFu)) | ((uint64_t)1u << 16)
                                  | ((uint64_t)8u << 32);
                const uint64_t db = ((uint64_t)((v + 128u + j * 32u) & 0x3FFFu))
                                  | ((uint64_t)1u << 16) | ((uint64_t)8u << 32);
                asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                             ::"r"(sfa_t + j * 4u), "l"(da) : "memory");
                asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                             ::"r"(sfb_t + j * 4u), "l"(db) : "memory");
            }
            const uint32_t w16 = (uint32_t)__cvta_generic_to_shared(wt + s * 16384u) >> 4;
            const uint32_t x16 = (uint32_t)__cvta_generic_to_shared(xt + s * 16384u) >> 4;
            #pragma unroll
            for (uint32_t j = 0; j < 4u; ++j) {
                const uint64_t ad = pd_tc5_sdesc(w16 + j * 2u);
                const uint64_t bd = pd_tc5_sdesc(x16 + j * 2u);
                const uint32_t en = (kt > 0 || j > 0) ? 1u : 0u;
                asm volatile(
                    "{\n\t.reg .pred p;\n\t"
                    "setp.ne.b32 p, %6, 0;\n\t"
                    "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X"
                    " [%0], %1, %2, %3, [%4], [%5], p;\n\t}"
                    ::"r"(tmem), "l"(ad), "l"(bd), "r"(pd_nv4tc5_idesc(128u)),
                      "r"(sfa_t + j * 4u), "r"(sfb_t + j * 4u), "r"(en));
            }
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                ::"r"((uint32_t)__cvta_generic_to_shared(&bdone[s])));
        }
        bar_wait(&bdone[s], (dph >> s) & 1u);
        dph ^= 1u << s;
        const uint32_t pf = kt + S;
        if (pf < nk) {
            sf_stage(k0 + pf, s);
            if (tid == 0) tma_stage(k0 + pf, s);
        }
    }
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");
    const uint32_t warp = tid >> 5, lane = tid & 31u;
    const uint32_t row = row_base + warp * 32u + lane;
    const float b0 = (!RAW && bias != nullptr && row < out_dim) ? bias[row] : 0.0f;
    float* yz = RAW ? y + (size_t)blockIdx.z * batch * out_dim : y;
    #pragma unroll
    for (uint32_t cc = 0; cc < 4u; ++cc) {
        uint32_t r[32];
        const uint32_t taddr = tmem + ((warp * 32u) << 16) + cc * 32u;
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
            #pragma unroll
            for (uint32_t j = 0; j < 32u; ++j) {
                const uint32_t col = col_base + cc * 32u + j;
                if (col < batch) {
                    // an empty split (nk == 0) never issued an mma: its partial is zero,
                    // not whatever tensor memory held
                    const float a = nk ? __uint_as_float(r[j]) : 0.0f;
                    yz[(size_t)col * out_dim + row] = RAW ? a : a * scale2 + b0;
                }
            }
        }
    }
    __syncthreads();
    if (tid < 32)
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 256;" ::"r"(tmem));
#else
    (void)wmap; (void)xmap; (void)wsc; (void)xsc; (void)y; (void)scale2; (void)bias;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}

// ---- v2: warp-specialized persistent kernel (pd_nv4tc5_ws_kt) ----------------
// v1 recycles a slot only after that slab's mmas completed AND every thread restaged
// its scale cells, and every 20-slab tile walk pays its own prologue and epilogue: on
// the decode planes it streamed ~2 TB/s of an 8 TB/s die. v2 is the warp-specialized
// form:
//   warp 4  producer: lane 0 issues the W/X TMA boxes and ONE 2 KB bulk copy of the
//           weight scale cells - pre-tiled at load into the canonical order by
//           pd_nvf4_tc5_tile_scales, so the 512 4-byte gathers per slab are gone - and
//           all 32 lanes gather the activation scale cells (BN rows x 4) with 4 B
//           cp.asyncs; the slot barrier counts 1 expect_tx arrival + 32 noinc ones.
//   warp 5  mma issuer (lane 0): tcgen05.cp the 8 scale tiles, 4 mmas, commit -> the
//           slot's empty barrier; after an item's last slab, commit -> dfull[buf].
//   warps 0-3 epilogue: wait dfull[buf], tcgen05.ld their lane quarter, release the
//           buffer (dempty), store. D ping-pongs between two BN-column buffers, so an
//           item's epilogue overlaps the next item's mma stream.
// Items (weight tile, activation tile, K split) are dealt round-robin to <= one CTA
// per SM - static, so nothing to reset between launches and nothing a second stream
// can share. BN = the activation tile (32/64/128 rows): a decode batch reads only the
// rows it has and the ring gets deeper (S = 8/7/6).
// F8 = the same pipeline over the block-scaled e4m3 planes (W8: e4m3 + ue8m0 per 32,
// row-major, and quantize_e4m3's activations): kind::mxf8f6f4.block_scale.scale_vec::1X,
// K = 128 per slab, ONE canonical scale tile per operand per slab serving its four mmas
// by sf-id byte (pd_f8bs_gemm_tc5_kt's probed layout), weight scale cells gathered by the
// producer lanes like the activation ones (4 per lane per slab - no load-time repack).
// It replaces the decode projections' mma.sync K-split arm (f8d_gemm_mma_ks) on B200.
// ROW (with F8) = the per-row-scale e4m3 planes (f8row: e4m3 + one f32 scale per weight
// row, quantize_e4m3_row's activations + one f32 per activation row): plain kind::f8f6f4,
// no scale tiles at all, the epilogue applies wrs[row] * xrs[col] (the qwen3.8 MLP layers
// the NVFP4 recipe kept at fp8, served by pd_f8row_gemm_tc5_kt at ~1.2 TB/s on B200).
#define PD_NV4WS_SF(F8, ROW) ((ROW) ? 0u : ((F8) ? 512u : 2048u))
#define PD_NV4WS_STAGE(BN, F8, ROW) (16384u + (BN) * 128u + 2u * PD_NV4WS_SF(F8, ROW))
#define PD_NV4WS_SMEM(S, BN, F8, ROW) ((S) * PD_NV4WS_STAGE(BN, F8, ROW) + (2u * (S) + 4u) * 8u)

// RAW (K split) launched as clusters of the splits: the CLUSTER K-split (`ck` below) -
// partials staged in the idle stage ring, reduced through DSMEM in rank order.
// RAW (K split) with `tctr`: SERIALIZED-K epilogue (f8_lin's KF scheme). Every split
// item has its own CTA in one wave; the mainloops run concurrently and only the epilogues
// order: split z waits for its tile's flag to read z, adds its accumulators into `yfin`
// (z = 0 stores), the last split applies scale2 + bias, then the flag passes on (and
// resets to 0 after the last). Fixed z order = deterministic; no partial planes, no
// reduce launch. Measured on the idle B200, down plane (40 tiles) at batch 64: unsplit
// 35.5 us, split + reduce 55 us. Without `tctr`, RAW leaves partials for
// pd_nv4tc5_reduce. The counters are the stream's (pd_nv4tc5_ctr).
// SWQ (unsplit nvf4, BN 128 only): the plane is an INTERLEAVED gate|up pair (rows 2j / 2j+1 =
// gate_j / up_j) and the epilogue writes the down GEMM's operand instead of y: silu(gate) * up from
// the accumulators (each value the unfused chain's `a * scale2 + 0` exactly), quantized as
// pd_quantize_nvf4_swiglu does (16-element blocks, pd_nvf4_scale + pd_e2m1_rn) into swq_q
// [batch][ff / 2] and swq_s [batch][ff / 16]. The mma runs operand-swapped (A = the activation
// tile, B = the weight tile, the scale tiles swapped with them): TMEM lanes are batch rows, so a
// 32-column load hands each thread one whole 16-pair block - silu, amax, scale and packing in
// registers, no shuffles. (Lanes = weight rows put a block across 16 lanes: ~120 dependent
// shuffle-bound instructions a value, 4x the unfused chain's time.)
// SWG, the swiglu-only landing (h = silu(g) * u in f32 from the epilogue, the plain quantizer
// after it - half the gate|up plane's bytes), falsified 2026-10-21: bit-identical, but the
// expf + IEEE divide a value doubles the 4-warp epilogue's instructions (2.48 -> 4.83 M) and it
// stops hiding behind the mainloop - 36 -> 62 us at 65 rows, 616 -> 1137 us (with the
// quantizer) against the pair at 4000. The separate pass, all SMs wide, is cheaper.
// K2 (KS = 2): a ring stage carries TWO 256-K slabs - two W and two X boxes, the 8 scale tiles of
// each operand in one 4 KB bulk copy (consecutive slabs' tiles are contiguous in both pre-tiled
// planes) - and the issuer runs the 8 mmas in the slabs' K order before ONE commit; the ring keeps
// S / 2 stages in the same smem. Measured at prefill widths (1760 rows, B200): stripped of every
// load, mma and store this kernel still took ~157 of its ~220 us - a ~300 ns floor a slab in the
// stage handshake (independent of S: 2, 3 and 6 time the same), not in the data. Halving the
// handshakes: gate|up 213 -> 173 us, down 117 -> 83 us at 1760 rows; 26 -> 25 us for gate|up at
// 120. Same mma sequence and operands, so the same bits (nv4_tc5_shape WIDE=1 DUMP=, K2 vs
// PADDOCK_NO_NV4TC5_K2: byte-identical; scales over 2^-9..2 make every sum round). Unsplit
// launches only: a K split's boundaries in 512s would move its partial sums (WIDE=1 shows it).
// Where K2 leaves it (ncu, gate|up 34816 x 5120 at 1760 rows, 173 us, 1.65 GHz under load):
// tensor pipe 45% active, L2 49%, DRAM 23%, smem banks 28% read / 28% write - no unit
// saturated. Not the issuer either (2026-10-21): the stage's 16 scale copies + 8 mmas + commit
// under one elect.sync each for copies and mmas, descriptors stepped in-asm, cut the kernel's
// instructions 34.3 -> 30.6 M (~97 a stage) byte-identically and timed the same (173.5 vs
// 174.0 us). The ceiling is the scale copies (bench/tc5_mma_rate.cu, smem-resident operands,
// B200 ~1.7 GHz): an mxf4nvf4 M128 N128 K64 mma alone issues every 64 cycles (7.9 PFLOP/s
// die-wide); with this kernel's 8 tcgen05.cp (32x128b.warpx4, 512 B) a slab it is 135 cycles
// (4.05 PFLOP/s) - this kernel's ~3.6 PFLOP/s is ~90% of that. Wider tiles amortize the SFA
// copies (N=256, 12 copies a slab: 198 cycles a 2x mma = 5.65 PFLOP/s in the bench) but did not
// carry over, 2026-10-21, both byte-identical to this kernel: a 1-CTA M128 N256 twin (8 epilogue
// warps, D single-buffered and released from registers; its D lanes land 64-half-swapped like
// tc5s's) ran 192 vs 174 us on gate|up at 1760 rows (179 with its stores stripped); a ::2
// M256 N256 pair (each rank staging both column groups' scales) ran 254 us - its ::2 copies
// cost ~80 us (175 us with them stripped).
// Decode widths (one tile a CTA) pay a fixed ~4.4 us a launch (fill + exposed epilogue). A lean
// config that would let the next GEMM's CTA co-reside and prefetch under this one's tail - a
// 3-stage ring (~110 KB smem), one accumulator, 256 TMEM columns - measured byte-identical but
// slower back to back under PDL (gate|up at 128 rows 25.3 -> 29.9 us, down 19.2 -> 19.7): the
// shallower ring (and no K2) costs more stream rate than the overlap returns.
template <uint32_t S, uint32_t BN, bool RAW, bool F8 = false, bool ROW = false, bool SWQ = false,
          uint32_t KS = 1u>
__global__ void __launch_bounds__(192, 1) pd_nv4tc5_ws_kt(
    const __grid_constant__ PdTmap wmap, const __grid_constant__ PdTmap xmap,
    const unsigned char* __restrict__ wsct, const unsigned char* __restrict__ xsc,
    float* __restrict__ y, float scale2, const float* __restrict__ bias,
    uint32_t in_dim, uint32_t out_dim, uint32_t batch, uint32_t n_kz,
    float* __restrict__ yfin = nullptr, uint32_t* __restrict__ tctr = nullptr,
    const unsigned char* __restrict__ xsw = nullptr, unsigned char* __restrict__ swq_q = nullptr,
    unsigned char* __restrict__ swq_s = nullptr) {
#if PD_TC5_OK
    static_assert(BN == 32u || BN == 64u || BN == 128u, "activation tile");
    static_assert(!SWQ || (!RAW && !F8 && !ROW && BN == 128u), "SWQ is the unsplit BN-128 nvf4 epilogue");
    // activation scales already in the 128x4 swizzled atom layout (pd_sf_swz128): one
    // 2 KB bulk copy a slab instead of BN x 4 four-byte cp.async gathers - the gather
    // held the producer to ~70% of this kernel's rate at the serve's 820-row ticks
    const bool xbulk = !F8 && !ROW && BN == 128u && xsw != nullptr;
    // the e4m3 twin (slot 882): wsct holds the weight scale cells pre-tiled (883) and xsw the
    // activation cells swizzled - both slabs' 512 B tiles by bulk copy, no 4 B gathers
    const bool f8bulk = F8 && !ROW && BN == 128u && xsw != nullptr;
    static_assert(!ROW || F8, "ROW is an e4m3 mode");
    constexpr uint32_t STG = PD_NV4WS_STAGE(BN, F8, ROW), SFB = PD_NV4WS_SF(F8, ROW);
    constexpr uint32_t SFC = ROW ? 0u : (F8 ? 8u : 32u);   // scale tmem cols per slab (A + B)
    // KS: 256-K slabs a ring stage (K2 above); the ring keeps S / KS stages in the same smem
    static_assert(KS == 1u || (KS == 2u && !ROW && !RAW && BN == 128u && S % KS == 0u),
                  "K2 is the unsplit BN-128 mainloop with bulk-copied scales (nvf4: swizzled "
                  "activation scales; e4m3: the tiled planes of slot 882)");
    constexpr uint32_t SE = S / KS;
    extern __shared__ __align__(1024) unsigned char pd_nv4ws_sh[];
    uint64_t* full = (uint64_t*)(pd_nv4ws_sh + S * STG);
    uint64_t* empty = full + S;
    uint64_t* dfull = empty + S;     // [2]
    uint64_t* dempty = dfull + 2;    // [2]
    __shared__ uint32_t tmem_slot[1];

    const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
    const uint32_t n_sb = F8 ? in_dim >> 5 : in_dim >> 4;   // scale bytes per row
    const uint32_t nk_all = (F8 ? in_dim >> 7 : in_dim >> 8) / KS; // stages of KS 128-byte slabs
    const uint32_t per = (nk_all + n_kz - 1u) / n_kz;
    const uint32_t n_wt = (out_dim + 127u) / 128u, n_ct = (batch + BN - 1u) / BN;
    const uint32_t items = n_wt * n_ct * n_kz;
    // CLUSTER K-split (RAW launched with cluster dims (n_kz,1,1), pd_nv4tc5_ck_go): one item a
    // CTA, a cluster = one tile's K splits (rank = split), and the epilogue reduces the
    // partials through DSMEM in rank order - no flag chain, no partial planes
    uint32_t ncr = 1u;
    if (RAW) asm("mov.u32 %0, %%cluster_nctarank;" : "=r"(ncr));
    const bool ck = RAW && ncr > 1u;
    const uint32_t id0 = ck ? (blockIdx.x % n_kz) * (n_wt * n_ct) + blockIdx.x / n_kz : blockIdx.x;

    if (tid == 0) {
        for (uint32_t s = 0; s < SE; ++s) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 33;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&full[s])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&empty[s])));
        }
        for (uint32_t b = 0; b < 2u; ++b) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&dfull[b])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 4;"
                         ::"r"((uint32_t)__cvta_generic_to_shared(&dempty[b])));
        }
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    if (warp == 5)
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;"
                     ::"r"((uint32_t)__cvta_generic_to_shared(tmem_slot)));
    asm volatile("tcgen05.fence::before_thread_sync;");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");
    const uint32_t tmem = tmem_slot[0];

    auto bar_wait = [&](uint64_t* bar, uint32_t parity) {
        const uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
        asm volatile("{\n\t.reg .pred p;\nW%=:\n\t"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                     "@!p bra W%=;\n\t}" ::"r"(a), "r"(parity) : "memory");
    };
    // activation tile innermost: the CTAs of a round share a few weight tiles across every
    // activation tile, so a W tile streams from DRAM about once while the (small) activation
    // plane stays L2-hot. Weight-innermost dealt 148 distinct W tiles a round and re-read the
    // plane ~20x at prefill widths (1.94 GB DRAM for a 100 MB gate|up plane at 4000 rows)
    // under the output stream's L2 churn. Which CTA owns a tile moves; no tile's bits do.
    auto item_at = [&](uint32_t id, uint32_t& wt, uint32_t& ct, uint32_t& k0, uint32_t& nk) {
        ct = id % n_ct;
        const uint32_t rest = id / n_ct;
        wt = rest % n_wt;
        const uint32_t kz = rest / n_wt;
        k0 = kz * per;
        nk = k0 < nk_all ? (per < nk_all - k0 ? per : nk_all - k0) : 0u;
    };

    // PDL: let the next kernel in the stream start its own pre-wait prologue as
    // our CTAs retire (a no-op when nothing launched as our dependent)
    asm volatile("griddepcontrol.launch_dependents;");
    if (warp == 4) {
        // ---- producer ----
        uint32_t s = 0, ph = 0;
        // The weight half of a slab (W box + its pre-tiled scales) depends on
        // nothing the predecessor writes: the first item's first S slabs issue
        // it BEFORE griddepcontrol.wait, so they stream while the predecessor
        // (the activation quantizer, the previous GEMM) drains - the same
        // dep-free W prologue tc5p runs. Only X and its scales wait. The early
        // slots are first uses, so skipping their empty wait changes nothing.
        // a stage: [W box u | ...][X box u | ...][SFA tiles][SFB tiles], u < KS - consecutive
        // slabs' scale tiles are contiguous in both pre-tiled planes, so one bulk copy each
        auto issue_w = [&](uint32_t st_i, uint32_t wt, uint32_t kt) {
            unsigned char* st = pd_nv4ws_sh + st_i * STG * KS;
            const uint32_t m = (uint32_t)__cvta_generic_to_shared(&full[st_i]);
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                         ::"r"(m), "r"(KS * (16384u + BN * 128u + (F8 ? 0u : 2048u) + (xbulk ? 2048u : 0u))
                                        + (f8bulk ? KS * 1024u : 0u)));
            #pragma unroll
            for (uint32_t u = 0; u < KS; ++u)
                asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                             " [%0], [%1, {%2, %3}], [%4];"
                             ::"r"((uint32_t)__cvta_generic_to_shared(st + u * 16384u)), "l"(&wmap),
                               "r"((int)((kt * KS + u) * 128u)), "r"((int)(wt * 128u)), "r"(m) : "memory");
            if (!F8) {
                const unsigned char* wsrc = wsct + ((size_t)wt * nk_all + kt) * KS * 2048u;
                asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                             " [%0], [%1], %2, [%3];"
                             ::"r"((uint32_t)__cvta_generic_to_shared(st + KS * (16384u + BN * 128u))),
                               "l"(wsrc), "r"(KS * 2048u), "r"(m) : "memory");
            } else if (f8bulk) {
                asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                             " [%0], [%1], %2, [%3];"
                             ::"r"((uint32_t)__cvta_generic_to_shared(st + KS * (16384u + BN * 128u))),
                               "l"(wsct + ((size_t)wt * nk_all + kt) * KS * 512u), "r"(KS * 512u), "r"(m)
                             : "memory");
            }
        };
        uint32_t pre = 0;
        if (id0 < items) {
            uint32_t wt, ct, k0, nk;
            item_at(id0, wt, ct, k0, nk);
            pre = nk < SE ? nk : SE;
            if (lane == 0)
                for (uint32_t i = 0; i < pre; ++i) issue_w(i, wt, k0 + i);
        }
        asm volatile("griddepcontrol.wait;" ::: "memory");
        for (uint32_t id = id0; id < items; id += gridDim.x) {
            uint32_t wt, ct, k0, nk;
            item_at(id, wt, ct, k0, nk);
            for (uint32_t kt = k0; kt < k0 + nk; ++kt) {
                const bool early = id == id0 && kt - k0 < pre;
                if (!early) bar_wait(&empty[s], ph ^ 1u);
                unsigned char* st = pd_nv4ws_sh + s * STG * KS;
                const uint32_t m = (uint32_t)__cvta_generic_to_shared(&full[s]);
                if (lane == 0) {
                    if (!early) issue_w(s, wt, kt);
                    #pragma unroll
                    for (uint32_t u = 0; u < KS; ++u)
                        asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                                     " [%0], [%1, {%2, %3}], [%4];"
                                     ::"r"((uint32_t)__cvta_generic_to_shared(st + KS * 16384u + u * BN * 128u)),
                                       "l"(&xmap), "r"((int)((kt * KS + u) * 128u)), "r"((int)(ct * BN)), "r"(m)
                                     : "memory");
                    if (xbulk)
                        asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                                     " [%0], [%1], %2, [%3];"
                                     ::"r"((uint32_t)__cvta_generic_to_shared(st + KS * (16384u + BN * 128u + 2048u))),
                                       "l"(xsw + ((size_t)ct * nk_all + kt) * KS * 2048u), "r"(KS * 2048u), "r"(m)
                                     : "memory");
                    else if (f8bulk)
                        asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                                     " [%0], [%1], %2, [%3];"
                                     ::"r"((uint32_t)__cvta_generic_to_shared(st + KS * (16384u + BN * 128u + 512u))),
                                       "l"(xsw + ((size_t)ct * nk_all + kt) * KS * 512u), "r"(KS * 512u), "r"(m)
                                     : "memory");
                }
                unsigned char* sfa = st + KS * (16384u + BN * 128u);
                unsigned char* sfb = sfa + SFB;
                if (ROW) {
                    // no scale cells: the epilogue applies the row scales
                } else if (F8 && (f8bulk || KS > 1u)) {
                    // both tiles came by bulk copy above (K2 runs only with them)
                } else if (F8) {
                    // one cell (4 k-blocks of 32) per row and operand -> one canonical tile each
                    #pragma unroll
                    for (uint32_t r = lane; r < 128u; r += 32u) {
                        const uint32_t row = wt * 128u + r;
                        pd_nv4tc5_cpa4(sfa + (r % 32u) * 16u + (r / 32u) * 4u,
                                       wsct + (size_t)row * n_sb + kt * 4u, row < out_dim);
                    }
                    #pragma unroll
                    for (uint32_t r = lane; r < BN; r += 32u) {
                        const uint32_t row = ct * BN + r;
                        pd_nv4tc5_cpa4(sfb + (r % 32u) * 16u + (r / 32u) * 4u,
                                       xsc + (size_t)row * n_sb + kt * 4u, row < batch);
                    }
                } else {
                    // activation scale cells -> the 4 canonical SFB tiles (rows >= BN never read)
                    if (!xbulk)
                    #pragma unroll
                    for (uint32_t i = lane; i < BN * 4u; i += 32u) {
                        const uint32_t r = i >> 2, j = i & 3u;
                        const uint32_t row = ct * BN + r;
                        pd_nv4tc5_cpa4(sfb + j * 512u + (r % 32u) * 16u + (r / 32u) * 4u,
                                       xsc + (size_t)row * n_sb + kt * 16u + j * 4u, row < batch);
                    }
                }
                asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(m) : "memory");
                if (++s == SE) { s = 0; ph ^= 1u; }
            }
        }
    } else if (warp == 5) {
        // ---- mma issuer: the WHOLE warp walks the loop and each tcgen05 op issues under
        // elect.sync. Run on lane 0 alone (if (lane == 0)), nvcc cannot prove the operands
        // uniform and wraps every UTCCP / UTCOMMA / UTCBAR in an ELECT + R2UR + BRA.U.ANY
        // waterfall: ~1180 issue cycles a 4-mma slab against the 268 the tensor pipe needs
        // (ncu, 17408 x 5120 x 820: tensor pipe 16.8% active, issuer samples flat over the
        // sequence). Converged, the descriptors live in uniform registers. ----
        {
            const uint32_t tmem_u = __shfl_sync(0xffffffffu, tmem, 0);
            const uint32_t sf_base_u = tmem_u + 2u * BN;   // S x SFC scale cols
            uint32_t s = 0, ph = 0, buf = 0, dph0 = 0, dph1 = 0;
            for (uint32_t id = id0; id < items; id += gridDim.x) {
                uint32_t wt, ct, k0, nk;
                item_at(id, wt, ct, k0, nk);
                bar_wait(&dempty[buf], (buf ? dph1 : dph0) ^ 1u);
                asm volatile("tcgen05.fence::after_thread_sync;");
                const uint32_t dt = tmem_u + buf * BN;
                for (uint32_t i = 0; i < nk; ++i) {
                    bar_wait(&full[s], ph);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    unsigned char* st = pd_nv4ws_sh + s * STG * KS;
                    const uint32_t sfa_t = sf_base_u + s * SFC * KS, sfb_t = sfa_t + SFC * KS / 2u;
                    const uint32_t va = (uint32_t)__cvta_generic_to_shared(st + KS * (16384u + BN * 128u)) >> 4;
                    // canonical tile descriptor (LBO 1, SBO 8 - probed in pd_f8bs_gemm_tc5_kt)
                    auto tdesc = [](uint32_t v) {
                        return ((uint64_t)(v & 0x3FFFu)) | ((uint64_t)1u << 16) | ((uint64_t)8u << 32);
                    };
                    constexpr uint32_t NSF = ROW ? 0u : (F8 ? 1u : 4u);   // scale tiles per operand
                    #pragma unroll
                    for (uint32_t j = 0; j + 1u <= NSF * KS; ++j) {
                        asm volatile("{\n\t.reg .pred e;\n\telect.sync _|e, 0xffffffff;\n\t"
                                     "@e tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;\n\t}"
                                     ::"r"(sfa_t + j * 4u), "l"(tdesc(va + j * 32u)) : "memory");
                        asm volatile("{\n\t.reg .pred e;\n\telect.sync _|e, 0xffffffff;\n\t"
                                     "@e tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;\n\t}"
                                     ::"r"(sfb_t + j * 4u), "l"(tdesc(va + SFB * KS / 16u + j * 32u)) : "memory");
                    }
                    #pragma unroll
                    for (uint32_t jk = 0; jk < 4u * KS; ++jk) {
                        // slab u of the stage, its mma j: the K order of KS separate slabs exactly
                        const uint32_t j = jk & 3u, u = jk >> 2;
                        const uint32_t w16 = (uint32_t)__cvta_generic_to_shared(st + u * 16384u) >> 4;
                        const uint32_t x16 =
                            (uint32_t)__cvta_generic_to_shared(st + KS * 16384u + u * BN * 128u) >> 4;
                        const uint32_t en = (i > 0 || jk > 0) ? 1u : 0u;
                        if (ROW)
                            // e4m3 x e4m3 -> f32, K-major both, M=128 N=BN (pd_tc5_idesc's fields)
                            asm volatile(
                                "{\n\t.reg .pred p, e;\n\telect.sync _|e, 0xffffffff;\n\t"
                                "setp.ne.b32 p, %4, 0;\n\t"
                                "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t}"
                                ::"r"(dt), "l"(pd_tc5_sdesc(w16 + j * 2u)), "l"(pd_tc5_sdesc(x16 + j * 2u)),
                                  "r"((1u << 4) | ((BN >> 3) << 17) | ((128u >> 4) << 24)), "r"(en));
                        else if (F8)
                            asm volatile(
                                "{\n\t.reg .pred p, e;\n\telect.sync _|e, 0xffffffff;\n\t"
                                "setp.ne.b32 p, %6, 0;\n\t"
                                "@e tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X"
                                " [%0], %1, %2, %3, [%4], [%5], p;\n\t}"
                                ::"r"(dt), "l"(pd_tc5_sdesc(w16 + j * 2u)), "l"(pd_tc5_sdesc(x16 + j * 2u)),
                                  "r"(pd_tc5_bs_idesc_bn(j, BN)), "r"(sfa_t + u * 4u), "r"(sfb_t + u * 4u),
                                  "r"(en));
                        else if (SWQ)
                            // operand-swapped: D[batch row][weight row] (M = N = 128, symmetric idesc)
                            asm volatile(
                                "{\n\t.reg .pred p, e;\n\telect.sync _|e, 0xffffffff;\n\t"
                                "setp.ne.b32 p, %6, 0;\n\t"
                                "@e tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X"
                                " [%0], %1, %2, %3, [%4], [%5], p;\n\t}"
                                ::"r"(dt), "l"(pd_tc5_sdesc(x16 + j * 2u)), "l"(pd_tc5_sdesc(w16 + j * 2u)),
                                  "r"(pd_nv4tc5_idesc(BN)), "r"(sfb_t + jk * 4u), "r"(sfa_t + jk * 4u),
                                  "r"(en));
                        else
                            asm volatile(
                                "{\n\t.reg .pred p, e;\n\telect.sync _|e, 0xffffffff;\n\t"
                                "setp.ne.b32 p, %6, 0;\n\t"
                                "@e tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X"
                                " [%0], %1, %2, %3, [%4], [%5], p;\n\t}"
                                ::"r"(dt), "l"(pd_tc5_sdesc(w16 + j * 2u)), "l"(pd_tc5_sdesc(x16 + j * 2u)),
                                  "r"(pd_nv4tc5_idesc(BN)), "r"(sfa_t + jk * 4u), "r"(sfb_t + jk * 4u),
                                  "r"(en));
                    }
                    asm volatile(
                        "{\n\t.reg .pred e;\n\telect.sync _|e, 0xffffffff;\n\t"
                        "@e tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n\t}"
                        ::"r"((uint32_t)__cvta_generic_to_shared(&empty[s])) : "memory");
                    if (++s == SE) { s = 0; ph ^= 1u; }
                }
                // the item's accumulator is complete once every mma above has retired
                asm volatile(
                    "{\n\t.reg .pred e;\n\telect.sync _|e, 0xffffffff;\n\t"
                    "@e tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n\t}"
                    ::"r"((uint32_t)__cvta_generic_to_shared(&dfull[buf])) : "memory");
                if (buf) dph1 ^= 1u; else dph0 ^= 1u;
                buf ^= 1u;
            }
        }
        __syncwarp();
    } else {
        // ---- epilogue (warps 0-3: lane quarter = warp) ----
        uint32_t buf = 0, dph0 = 0, dph1 = 0;
        for (uint32_t id = id0; id < items; id += gridDim.x) {
            uint32_t wt, ct, k0, nk;
            item_at(id, wt, ct, k0, nk);
            bar_wait(&dfull[buf], buf ? dph1 : dph0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            const uint32_t row = wt * 128u + warp * 32u + lane;
            const float b0 = (!RAW && bias != nullptr && row < out_dim) ? bias[row] : 0.0f;
            const uint32_t kz = id / (n_wt * n_ct);
            const bool kf = RAW && tctr != nullptr;
            float* yz = kf ? yfin : (RAW ? y + (size_t)kz * batch * out_dim : y);
            const bool kf_last = kf && kz + 1u == n_kz;
            if (kf) {
                // this split's turn: the previous split's y is complete and visible
                if (tid == 0) {
                    const uint32_t* f = tctr + (wt + n_wt * ct);
                    uint32_t v;
                    do {
                        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(f) : "memory");
                    } while (v != kz);
                }
                asm volatile("bar.sync 1, 128;" ::: "memory");   // epilogue warps only
            }
            #pragma unroll
            for (uint32_t cc = 0; cc < BN / 32u; ++cc) {
                uint32_t r[32];
                const uint32_t taddr = tmem + ((warp * 32u) << 16) + buf * BN + cc * 32u;
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
                if (cc + 1u == BN / 32u) {
                    // every column of this buffer is in registers: hand it back
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    __syncwarp();
                    if (lane == 0)
                        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
                                     ::"r"((uint32_t)__cvta_generic_to_shared(&dempty[buf])) : "memory");
                }
                if constexpr (SWQ) {
                    // thread = batch row; columns cc*32.. = weight rows = gate|up pairs of one block
                    const uint32_t ff = out_dim >> 1, b = ct * BN + warp * 32u + lane;
                    const uint32_t j0 = (wt * 128u + cc * 32u) >> 1;
                    float h[16];
                    float a = 0.0f;
                    #pragma unroll
                    for (uint32_t m = 0; m < 16u; ++m) {
                        const float g = (nk ? __uint_as_float(r[2u * m]) : 0.0f) * scale2 + 0.0f;
                        const float u = (nk ? __uint_as_float(r[2u * m + 1u]) : 0.0f) * scale2 + 0.0f;
                        h[m] = (g / (1.0f + expf(-g))) * u;
                        a = fmaxf(a, fabsf(h[m]));
                    }
                    float inv;
                    const unsigned sb = pd_nvf4_scale(a, &inv);
                    uint32_t p0 = 0u, p1 = 0u;
                    #pragma unroll
                    for (uint32_t m = 0; m < 8u; ++m) {
                        p0 |= pd_e2m1_rn(h[m] * inv) << (4u * m);
                        p1 |= pd_e2m1_rn(h[8u + m] * inv) << (4u * m);
                    }
                    if (b < batch) {
                        *(uint2*)(swq_q + (size_t)b * (ff >> 1) + (j0 >> 1)) = make_uint2(p0, p1);
                        swq_s[(size_t)b * (ff >> 4) + (j0 >> 4)] = (unsigned char)sb;
                    }
                    continue;
                }
                if (ck) {
                    // stage this split's partial, [col][row] (the stage ring is idle: every
                    // slab this CTA loaded has been consumed by the retired mmas)
                    float* pp = (float*)pd_nv4ws_sh + (cc * 32u) * 128u + warp * 32u + lane;
                    #pragma unroll
                    for (uint32_t j = 0; j < 32u; ++j) pp[j * 128u] = nk ? __uint_as_float(r[j]) : 0.0f;
                    continue;
                }
                if (row < out_dim) {
                    // KF: the previous split's 32 values for this chunk in flight at once (one L2
                    // round trip a chunk - a load-add-store chain per element cost ~0.5 us each
                    // and made the serialized epilogue slower than the reduce launch)
                    float prev[32];
                    if (kf) {
                        #pragma unroll
                        for (uint32_t j = 0; j < 32u; ++j) {
                            const uint32_t col = ct * BN + cc * 32u + j;
                            prev[j] = (kz && col < batch) ? __ldcg(yz + (size_t)col * out_dim + row) : 0.0f;
                        }
                    }
                    const float bl = kf_last && bias != nullptr ? bias[row] : 0.0f;
                    // ROW: wsct / xsc carry the f32 row scales
                    const float wr = ROW ? ((const float*)wsct)[row] : 1.0f;
                    #pragma unroll
                    for (uint32_t j = 0; j < 32u; ++j) {
                        const uint32_t col = ct * BN + cc * 32u + j;
                        if (col < batch) {
                            const float a = nk ? __uint_as_float(r[j]) : 0.0f;
                            const size_t o = (size_t)col * out_dim + row;
                            if (ROW) {
                                const float xr = ((const float*)xsc)[col];
                                if (kf) {
                                    const float v = prev[j] + a;
                                    yz[o] = kf_last ? v * wr * xr : v;
                                } else {
                                    yz[o] = a * wr * xr;   // unsplit only (no reduce for ROW)
                                }
                            } else if (kf) {
                                const float v = prev[j] + a;
                                yz[o] = kf_last ? v * scale2 + bl : v;
                            } else {
                                yz[o] = RAW ? a : a * scale2 + b0;
                            }
                        }
                    }
                }
            }
            if (kf) {
                // hand the tile to the next split (or back to 0 for the next launch)
                __threadfence();
                asm volatile("bar.sync 1, 128;" ::: "memory");
                if (tid == 0) {
                    uint32_t* f = tctr + (wt + n_wt * ct);
                    const uint32_t nv = kf_last ? 0u : kz + 1u;
                    asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(f), "r"(nv) : "memory");
                }
            }
            if (buf) dph1 ^= 1u; else dph0 ^= 1u;
            buf ^= 1u;
        }
    }
    if (ck) {
        // every split's partial is staged (release/acquire orders the smem stores for the
        // peers' DSMEM reads); this CTA sums its column slice over the splits in rank order
        // - deterministic - applies the unsplit epilogue's scaling and writes y
        asm volatile("barrier.cluster.arrive.release.aligned;\n\t"
                     "barrier.cluster.wait.acquire.aligned;" ::: "memory");
        const uint32_t rank = blockIdx.x % n_kz, tile = blockIdx.x / n_kz;
        const uint32_t ct = tile % n_ct, wt = tile / n_ct;   // item_at's order
        const uint32_t c0 = rank * BN / n_kz, c1 = (rank + 1u) * BN / n_kz;
        const uint32_t pbase = (uint32_t)__cvta_generic_to_shared(pd_nv4ws_sh);
        // float4s (4 rows) a lane, 4 of them in flight per rank: a scalar walk paid one
        // DSMEM round trip per element per rank (~7 us a CTA, ncu: L1TEX scoreboard 53%).
        // Every rank's loads in flight before the (rank-ordered) adds measured flat
        // 2026-10-21: bit-identical, down 5120 x 17408 at 128 rows 19.4 vs 19.5 us, W8
        // out-proj 15.1 vs 14.9 (nv4_tc5_shape COLD) - the reads are not this CTA's wall.
        const uint32_t n4 = (c1 - c0) * 32u;
        for (uint32_t i0 = tid; i0 < n4; i0 += 4u * 192u) {
            float4 acc[4];
            for (uint32_t z = 0; z < n_kz; ++z) {
                float4 v[4];
                #pragma unroll
                for (uint32_t u = 0; u < 4u; ++u) {
                    const uint32_t i = i0 + u * 192u;
                    if (i < n4) {
                        const uint32_t cl = c0 + (i >> 5), r4 = (i & 31u) * 4u;
                        uint32_t ra;
                        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                                     : "=r"(ra) : "r"(pbase + (cl * 128u + r4) * 4u), "r"(z));
                        asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];"
                                     : "=f"(v[u].x), "=f"(v[u].y), "=f"(v[u].z), "=f"(v[u].w)
                                     : "r"(ra) : "memory");
                    }
                }
                #pragma unroll
                for (uint32_t u = 0; u < 4u; ++u) {
                    if (z) {
                        acc[u].x += v[u].x; acc[u].y += v[u].y; acc[u].z += v[u].z; acc[u].w += v[u].w;
                    } else {
                        acc[u] = v[u];
                    }
                }
            }
            #pragma unroll
            for (uint32_t u = 0; u < 4u; ++u) {
                const uint32_t i = i0 + u * 192u;
                if (i >= n4) continue;
                const uint32_t cl = c0 + (i >> 5), r4 = (i & 31u) * 4u;
                const uint32_t row = wt * 128u + r4, col = ct * BN + cl;
                if (col >= batch) continue;
                const float a4[4] = {acc[u].x, acc[u].y, acc[u].z, acc[u].w};
                float o4[4];
                #pragma unroll
                for (uint32_t e = 0; e < 4u; ++e) {
                    const uint32_t rw = row + e < out_dim ? row + e : row;
                    o4[e] = ROW ? a4[e] * ((const float*)wsct)[rw] * ((const float*)xsc)[col]
                                : a4[e] * scale2 + (bias != nullptr ? bias[rw] : 0.0f);
                }
                float* yo = y + (size_t)col * out_dim + row;
                if ((out_dim & 3u) == 0u && row + 3u < out_dim) {
                    *reinterpret_cast<float4*>(yo) = make_float4(o4[0], o4[1], o4[2], o4[3]);
                } else {
                    #pragma unroll
                    for (uint32_t e = 0; e < 4u; ++e)
                        if (row + e < out_dim) yo[e] = o4[e];
                }
            }
        }
        // no CTA leaves (its smem reused) while a peer still reads it
        asm volatile("barrier.cluster.arrive.release.aligned;\n\t"
                     "barrier.cluster.wait.acquire.aligned;" ::: "memory");
    }
    asm volatile("tcgen05.fence::before_thread_sync;");
    __syncthreads();
    if (warp == 5)
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" ::"r"(tmem));
#else
    (void)wmap; (void)xmap; (void)wsct; (void)xsc; (void)y; (void)scale2; (void)bias;
    (void)in_dim; (void)out_dim; (void)batch; (void)n_kz;
#endif
}

// Weight scale cells, row-major [out][in/16] -> the canonical tile order the v2 producer
// bulk-copies: per (128-row tile t, 256-K slab k) 2 KB = 4 tiles (one per mma) x 512 B,
// byte l*16 + c*4 + b of tile j = scale[t*128 + c*32 + l][k*16 + 4j + b]; rows past
// out_dim are zero. One thread per 4-byte cell.
__global__ void pd_nv4tc5_tile_scales_kernel(const unsigned char* __restrict__ sc,
                                             unsigned char* __restrict__ tiled, uint32_t in_dim,
                                             uint32_t out_dim, uint32_t n_wt) {
    const uint32_t nk = in_dim >> 8;
    const uint64_t cell = blockIdx.x * 256ull + threadIdx.x;   // 4-byte cell index
    if (cell >= (uint64_t)n_wt * nk * 512u) return;
    const uint32_t within = (uint32_t)(cell % 512u);           // cell in the 2 KB block
    const uint64_t blk = cell / 512u;
    const uint32_t k = (uint32_t)(blk % nk), t = (uint32_t)(blk / nk);
    const uint32_t j = within / 128u, rem = within % 128u;      // tile j, cell (l, c)
    const uint32_t l = rem / 4u, c = rem % 4u;
    const uint32_t row = t * 128u + c * 32u + l;
    uint32_t v = 0;
    if (row < out_dim)
        v = *(const uint32_t*)(sc + (size_t)row * (in_dim >> 4) + k * 16u + j * 4u);
    *(uint32_t*)(tiled + blk * 2048u + j * 512u + l * 16u + c * 4u) = v;
}

// fixed-order split sum + the pd_nvf4_gemm_f4 epilogue (acc * scale2 + bias)
__global__ void pd_nv4tc5_reduce(const float* __restrict__ part, float* __restrict__ y,
                                 float scale2, const float* __restrict__ bias, uint32_t sk,
                                 uint32_t out_dim, uint32_t n) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");   // the split GEMM's partials
#endif
    const uint32_t i = blockIdx.x * 256u + threadIdx.x;
    if (i >= n) return;
    float a = 0.0f;
    for (uint32_t z = 0; z < sk; ++z) a += part[(size_t)z * n + i];
    y[i] = a * scale2 + (bias ? bias[i % out_dim] : 0.0f);
}

#if defined(PD_TC5_HOST)
static int pd_nv4tc5_nsm() {
    static int nsm = 0;
    if (nsm == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
        if (nsm <= 0) nsm = 148;
    }
    return nsm;
}
// split depth: only where the tile grid leaves most of the die idle, never past
// 4 slabs a split, and only when the caller's scratch holds the partials
// `bmax`: the widest batch the split still pays at. Idle-B200 probe, 40-tile planes
// (serialized-K epilogue vs unsplit): nvf4 down 21.6 vs 34.2 us at b=1-8, 30.2 vs 35.6 at
// b=64, 43.2 vs 38.3 at b=128; W8 o/out 15.6 vs 17.6 at b=1 but 23.2 vs 18.9 at b=37 -
// the epilogue chain grows with the batch columns while the mainloop gain does not. The
// CLUSTER K-split (pd_nv4tc5_ck_go) has no chain, so it pays at every decode width
// (pd_nv4tc5_split_bmax): down 11.0 vs 16.5 serialized at b=1, 17.9 vs 30.9 unsplit at
// b=124; W8 out 8.3 vs 13.5 at b=1, 13.2 vs 21.6 at b=124.
static uint32_t pd_nv4tc5_split(uint32_t tiles, uint32_t nk_all, size_t part_floats,
                                size_t plane, uint32_t batch = 0u, uint32_t bmax = ~0u) {
    const char* e = pd_env("PADDOCK_NV4TC5_SK");
    uint32_t sk = 1u;
    if (e) {
        sk = (uint32_t)atoi(e);
    } else if (batch <= bmax && tiles * 2u <= (uint32_t)pd_nv4tc5_nsm()) {
        sk = (uint32_t)pd_nv4tc5_nsm() / tiles;
    }
    if (sk > 8u) sk = 8u;
    if (sk > nk_all / 4u) sk = nk_all / 4u;
    if (sk < 1u) sk = 1u;
    while (sk > 1u && (size_t)sk * plane > part_floats) --sk;
    return sk;
}
#endif

// y[batch][out] = scale2 * (W4 . X4^T) (+ bias): the pd_nvf4_gemm_f4 contract on tcgen05.
// `part` (optional, part_floats floats) holds split-K partials; without it the GEMM
// runs unsplit. Declines (cudaErrorNotSupported) off cc 10.0 or without the TMA encode,
// cudaErrorInvalidValue for in_dim % 256 != 0 or misaligned planes, so the caller keeps
// its other arm.
PD_EXPORT
int pd_nvf4_gemm_tc5(const void* data, const void* scale, const void* bias, const void* xq,
                     const void* xs, void* y, float scale2, uint32_t in_dim, uint32_t out_dim,
                     uint32_t batch, void* part, uint64_t part_floats, void* stream) {
#if !defined(PD_TC5_HOST)
    (void)data; (void)scale; (void)bias; (void)xq; (void)xs; (void)y; (void)scale2;
    (void)in_dim; (void)out_dim; (void)batch; (void)part; (void)part_floats; (void)stream;
    return cudaErrorNotSupported;
#else
    if (out_dim == 0 || batch == 0) return 0;
    if ((in_dim & 255u) != 0u) return cudaErrorInvalidValue;
    static int ok = -1;
    if (ok < 0) {
        int dev = 0, ccM = 0, ccm = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&ccM, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&ccm, cudaDevAttrComputeCapabilityMinor, dev);
        ok = (ccM == 10 && ccm == 0 && pd_tmap_encode() != nullptr) ? 1 : 0;
    }
    if (!ok) return cudaErrorNotSupported;
    CUtensorMap wm, xm;
    if (!pd_tmap_2d(&wm, data, in_dim / 2u, out_dim) || !pd_tmap_2d(&xm, xq, in_dim / 2u, batch))
        return cudaErrorInvalidValue;
    constexpr uint32_t S = 4u;
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute((const void*)pd_nv4tc5_kt<S, false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)PD_NV4TC5_SMEM(S));
        cudaFuncSetAttribute((const void*)pd_nv4tc5_kt<S, true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)PD_NV4TC5_SMEM(S));
        attr = true;
    }
    const cudaStream_t st = (cudaStream_t)stream;
    const uint32_t tx = (out_dim + 127u) / 128u, ty = (batch + 127u) / 128u;
    const size_t plane = (size_t)batch * out_dim;
    const uint32_t sk = part ? pd_nv4tc5_split(tx * ty, in_dim >> 8, part_floats, plane) : 1u;
    if (sk == 1u) {
        pd_nv4tc5_kt<S, false><<<dim3(tx, ty, 1), 128, PD_NV4TC5_SMEM(S), st>>>(
            wm, xm, (const unsigned char*)scale, (const unsigned char*)xs, (float*)y, scale2,
            (const float*)bias, in_dim, out_dim, batch);
    } else {
        pd_nv4tc5_kt<S, true><<<dim3(tx, ty, sk), 128, PD_NV4TC5_SMEM(S), st>>>(
            wm, xm, (const unsigned char*)scale, (const unsigned char*)xs, (float*)part, 1.0f,
            nullptr, in_dim, out_dim, batch);
        const uint32_t n = (uint32_t)plane;
        pd_nv4tc5_reduce<<<(n + 255u) / 256u, 256, 0, st>>>(
            (const float*)part, (float*)y, scale2, (const float*)bias, sk, out_dim, n);
    }
    return (int)cudaPeekAtLastError();
#endif
}

// Bytes the v2 kernel's pre-tiled weight scale plane needs (pd_nvf4_tc5_tile_scales).
PD_EXPORT
uint64_t pd_nvf4_tc5_tiled_scale_bytes(uint32_t in_dim, uint32_t out_dim) {
    return (uint64_t)((out_dim + 127u) / 128u) * (in_dim >> 8) * 2048u;
}

#if defined(PD_TC5_HOST)
static void pd_nv4tc5_ctr_prepare();   // split-fold counter pool, below
#endif
// Load-time repack of a weight plane's e4m3 scale cells into the v2 tile order.
PD_EXPORT
int pd_nvf4_tc5_tile_scales(const void* scale, void* tiled, uint32_t in_dim, uint32_t out_dim,
                            void* stream) {
    if ((in_dim & 255u) != 0u || out_dim == 0) return cudaErrorInvalidValue;
    const uint32_t n_wt = (out_dim + 127u) / 128u;
    const uint64_t cells = (uint64_t)n_wt * (in_dim >> 8) * 512u;
#if defined(PD_TC5_HOST)
    pd_nv4tc5_ctr_prepare();   // load time, outside every capture
    pd_tc5_pool_prepare();     // and the per-stream swizzled-scale scratch (tc5_lanes.cuh)
#endif
    pd_nv4tc5_tile_scales_kernel<<<(uint32_t)((cells + 255u) / 256u), 256, 0, (cudaStream_t)stream>>>(
        (const unsigned char*)scale, (unsigned char*)tiled, in_dim, out_dim, n_wt);
    return (int)cudaPeekAtLastError();
}

#if defined(PD_TC5_HOST)
// 128B-swizzled byte-matrix map with a `box_rows`-row box; cached by every input of the
// encode (base, inner, rows, box) - a key missing a dimension is how the f4t cache once
// handed `down` the narrow `q` map (quant/nvf4.cuh). Encoding is host work per call; the
// cache keeps eager prefill from paying it per GEMM. 64 entries, round-robin.
static bool pd_nv4tc5_tmap(CUtensorMap* out, const void* base, uint64_t inner, uint64_t rows,
                           uint32_t box_rows) {
    struct E { const void* b; uint64_t in, rows; uint32_t box; CUtensorMap m; };
    static E cache[64];
    static uint32_t n = 0, next = 0;
    for (uint32_t i = 0; i < n; ++i)
        if (cache[i].b == base && cache[i].in == inner && cache[i].rows == rows && cache[i].box == box_rows) {
            *out = cache[i].m;
            return true;
        }
    pd_tmap_encode_fn enc = pd_tmap_encode();
    if (!enc || ((uintptr_t)base & 15u) || (inner & 15u)) return false;
    const cuuint64_t gdim[2] = {inner, rows};
    const cuuint64_t gstride[1] = {inner};
    const cuuint32_t box[2] = {128u, box_rows};
    const cuuint32_t estride[2] = {1u, 1u};
    CUtensorMap m;
    if (enc(&m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2u, (void*)base, gdim, gstride, box, estride,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
        return false;
    E& e = cache[n < 64u ? n++ : (next++ % 64u)];
    e = E{base, inner, rows, box_rows, m};
    *out = m;
    return true;
}

// Per-stream split-fold counters (graph-capture + two-lane safe, the tc5_lanes.cuh
// pattern): built outside any capture - at load by pd_nvf4_tc5_tile_scales, or on the
// first eager call - and claimed per stream with host bookkeeping only. 8192 tiles a
// lane covers every split shape (splits run only where the tile grid starves the die).
constexpr uint32_t PD_NV4TC5_CTR_TILES = 8192u, PD_NV4TC5_CTR_LANES = 8u;
static uint32_t* pd_nv4tc5_ctr_pool = nullptr;
static cudaStream_t pd_nv4tc5_ctr_st[PD_NV4TC5_CTR_LANES] = {};
static uint32_t pd_nv4tc5_ctr_used = 0;
static void pd_nv4tc5_ctr_prepare() {
    if (pd_nv4tc5_ctr_pool) return;
    uint32_t* p = nullptr;
    const size_t bytes = (size_t)PD_NV4TC5_CTR_LANES * PD_NV4TC5_CTR_TILES * 4u;
    if (cudaMalloc((void**)&p, bytes) != cudaSuccess) { cudaGetLastError(); return; }
    cudaMemset(p, 0, bytes);
    pd_nv4tc5_ctr_pool = p;
}
static uint32_t* pd_nv4tc5_ctr(cudaStream_t st) {
    if (!pd_nv4tc5_ctr_pool) {
        cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(st, &cs) != cudaSuccess || cs != cudaStreamCaptureStatusNone)
            return nullptr;
        pd_nv4tc5_ctr_prepare();
        if (!pd_nv4tc5_ctr_pool) return nullptr;
    }
    for (uint32_t i = 0; i < pd_nv4tc5_ctr_used; ++i)
        if (pd_nv4tc5_ctr_st[i] == st) return pd_nv4tc5_ctr_pool + (size_t)i * PD_NV4TC5_CTR_TILES;
    if (pd_nv4tc5_ctr_used >= PD_NV4TC5_CTR_LANES) return nullptr;
    pd_nv4tc5_ctr_st[pd_nv4tc5_ctr_used] = st;
    return pd_nv4tc5_ctr_pool + (size_t)(pd_nv4tc5_ctr_used++) * PD_NV4TC5_CTR_TILES;
}

// Cluster K-split launch (the RAW kernel's `ck` mode): grid = tiles x kz, cluster (kz,1,1), PDL
// as pd_pdl_go. False - the caller falls back - when the cluster cannot be resident at this
// smem footprint (probed once per kz) or PADDOCK_NV4TC5_NO_CK is set.
template <typename K, typename... Args>
static bool pd_nv4tc5_ck_go(K kern, uint32_t grid, uint32_t smem, uint32_t kz, cudaStream_t st,
                            const Args&... args) {
    static const bool no_ck = pd_env("PADDOCK_NV4TC5_NO_CK") != nullptr;
    static int fits[9] = {-1, -1, -1, -1, -1, -1, -1, -1, -1};
    if (no_ck || kz < 2u || kz > 8u) return false;
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(grid);
    cfg.blockDim = dim3(192);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = st;
    cudaLaunchAttribute at[2];
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = kz;
    at[0].val.clusterDim.y = 1;
    at[0].val.clusterDim.z = 1;
    cfg.attrs = at;
    cfg.numAttrs = 1;
    if (fits[kz] < 0) {
        int n = 0;
        fits[kz] = (cudaOccupancyMaxActiveClusters(&n, kern, &cfg) == cudaSuccess && n > 0) ? 1 : 0;
        cudaGetLastError();
    }
    if (!fits[kz]) return false;
    if (!(pd_pdl_off() || !pd_pdl_dev_ok() || pd_pdl_denied((const void*)kern))) {
        at[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        at[1].val.programmaticStreamSerializationAllowed = 1;
        cfg.numAttrs = 2;
    }
    return cudaLaunchKernelEx(&cfg, kern, args...) == cudaSuccess;
}

// Split depth past the serialized-K bound when the cluster K-split serves: its epilogue is
// one DSMEM pass, not a chain that grows with the batch columns.
static uint32_t pd_nv4tc5_split_bmax(uint32_t serial_bmax) {
    static const bool no_ck = pd_env("PADDOCK_NV4TC5_NO_CK") != nullptr;
    return no_ck ? serial_bmax : 128u;
}

template <uint32_t S, uint32_t BN, bool F8 = false, bool ROW = false>
static int pd_nv4tc5_ws_go(const CUtensorMap& wm, const CUtensorMap& xm, const void* wsct,
                           const void* xs, void* y, void* part, float scale2, const void* bias,
                           uint32_t in_dim, uint32_t out_dim, uint32_t batch, uint32_t kz,
                           cudaStream_t st, const unsigned char* xsw = nullptr) {
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute((const void*)pd_nv4tc5_ws_kt<S, BN, false, F8, ROW>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)PD_NV4WS_SMEM(S, BN, F8, ROW));
        cudaFuncSetAttribute((const void*)pd_nv4tc5_ws_kt<S, BN, true, F8, ROW>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)PD_NV4WS_SMEM(S, BN, F8, ROW));
        attr = true;
    }
    const uint32_t items = ((out_dim + 127u) / 128u) * ((batch + BN - 1u) / BN) * kz;
    const uint32_t grid = items < (uint32_t)pd_nv4tc5_nsm() ? items : (uint32_t)pd_nv4tc5_nsm();
    // programmatic dependent launches (abi.cuh's pd_pdl_go; PADDOCK_NO_PDL kills)
    const PdTmap wmp(wm), xmp(xm);
    // K2: two slabs a ring stage where the mainloop allows it (unsplit, BN 128, swizzled
    // activation scales, K in 512s) - same mma sequence, same bits; kill PADDOCK_NO_NV4TC5_K2
    constexpr bool k2_ok = !ROW && BN == 128u && S % 2u == 0u;
    static const bool no_k2 = pd_env("PADDOCK_NO_NV4TC5_K2") != nullptr;
    if (k2_ok && kz == 1u && xsw != nullptr && (in_dim & (F8 ? 255u : 511u)) == 0u && !no_k2) {
        constexpr uint32_t KS2 = k2_ok ? 2u : 1u;   // (k2_ok false: never launched)
        static bool attr2 = false;
        if (!attr2) {
            cudaFuncSetAttribute((const void*)pd_nv4tc5_ws_kt<S, BN, false, F8, ROW, false, KS2>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)PD_NV4WS_SMEM(S, BN, F8, ROW));
            attr2 = true;
        }
        pd_pdl_go(pd_nv4tc5_ws_kt<S, BN, false, F8, ROW, false, KS2>, dim3(grid), dim3(192),
                  PD_NV4WS_SMEM(S, BN, F8, ROW), st, wmp, xmp, (const unsigned char*)wsct,
                  (const unsigned char*)xs, (float*)y, scale2, (const float*)bias, in_dim, out_dim, batch, 1u,
                  (float*)nullptr, (uint32_t*)nullptr, xsw, (unsigned char*)nullptr, (unsigned char*)nullptr);
    } else if (kz == 1u) {
        pd_pdl_go(pd_nv4tc5_ws_kt<S, BN, false, F8, ROW>, dim3(grid), dim3(192), PD_NV4WS_SMEM(S, BN, F8, ROW), st,
                  wmp, xmp, (const unsigned char*)wsct, (const unsigned char*)xs, (float*)y, scale2,
                  (const float*)bias, in_dim, out_dim, batch, 1u, (float*)nullptr, (uint32_t*)nullptr, xsw,
                  (unsigned char*)nullptr, (unsigned char*)nullptr);
    } else if (pd_nv4tc5_ck_go(pd_nv4tc5_ws_kt<S, BN, true, F8, ROW>, items, PD_NV4WS_SMEM(S, BN, F8, ROW), kz,
                               st, wmp, xmp, (const unsigned char*)wsct, (const unsigned char*)xs, (float*)y,
                               scale2, (const float*)bias, in_dim, out_dim, batch, kz, (float*)nullptr,
                               (uint32_t*)nullptr, xsw, (unsigned char*)nullptr, (unsigned char*)nullptr)) {
        // cluster K-split: done
    } else {
        // serialized-K epilogue where the split grid is one wave (every split of a tile
        // resident - the flag chain needs that) and the stream has counters;
        // PADDOCK_NV4TC5_NO_KF keeps partials + reduce for A/B
        static const bool no_kf = pd_env("PADDOCK_NV4TC5_NO_KF") != nullptr;
        const uint32_t tiles = ((out_dim + 127u) / 128u) * ((batch + BN - 1u) / BN);
        uint32_t* ctr = (!no_kf && tiles <= PD_NV4TC5_CTR_TILES && items <= (uint32_t)pd_nv4tc5_nsm())
                            ? pd_nv4tc5_ctr(st) : nullptr;
        if (ROW && !ctr) return cudaErrorNotSupported;   // the caller keeps kz == 1 for ROW
        pd_pdl_go(pd_nv4tc5_ws_kt<S, BN, true, F8, ROW>, dim3(grid), dim3(192), PD_NV4WS_SMEM(S, BN, F8, ROW), st,
                  wmp, xmp, (const unsigned char*)wsct, (const unsigned char*)xs, (float*)part, scale2,
                  (const float*)bias, in_dim, out_dim, batch, kz, (float*)y, ctr, xsw,
                  (unsigned char*)nullptr, (unsigned char*)nullptr);
        if (!ctr) {
            const uint32_t n = batch * out_dim;
            pd_pdl_go(pd_nv4tc5_reduce, dim3((n + 255u) / 256u), dim3(256), 0u, st,
                      (const float*)part, (float*)y, scale2, (const float*)bias, kz, out_dim, n);
        }
    }
    return (int)cudaPeekAtLastError();
}

// The SWQ launch: one unsplit pass over the interleaved gate|up plane (out_dim = 2 ff) whose
// epilogue writes the down GEMM's nvf4 operand (q, qs) - no f32 landing, no swiglu-quant launch.
template <uint32_t S, uint32_t BN>
static int pd_nv4tc5_swq_go(const CUtensorMap& wm, const CUtensorMap& xm, const void* wsct,
                            const void* xs, void* q, void* qs, float scale2, uint32_t in_dim,
                            uint32_t out_dim, uint32_t batch, cudaStream_t st,
                            const unsigned char* xsw = nullptr) {
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute((const void*)pd_nv4tc5_ws_kt<S, BN, false, false, false, true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)PD_NV4WS_SMEM(S, BN, false, false));
        attr = true;
    }
    const uint32_t items = ((out_dim + 127u) / 128u) * ((batch + BN - 1u) / BN);
    const uint32_t grid = items < (uint32_t)pd_nv4tc5_nsm() ? items : (uint32_t)pd_nv4tc5_nsm();
    const PdTmap wmp(wm), xmp(xm);
    pd_pdl_go(pd_nv4tc5_ws_kt<S, BN, false, false, false, true>, dim3(grid), dim3(192),
              PD_NV4WS_SMEM(S, BN, false, false), st, wmp, xmp, (const unsigned char*)wsct,
              (const unsigned char*)xs, (float*)nullptr, scale2, (const float*)nullptr, in_dim, out_dim,
              batch, 1u, (float*)nullptr, (uint32_t*)nullptr, xsw, (unsigned char*)q, (unsigned char*)qs);
    return (int)cudaPeekAtLastError();
}
#endif

// Row-major scale cells (`n_w` 4-byte words a row, as the quantizers write them) into the
// 128x4 swizzled atom layout tcgen05.cp's 32x128b.warpx4 tile wants (CUTLASS's
// Sm1xxBlockScaledBasicChunk, TRT-LLM's "swizzled SF"): atom (row group g, word k) is 512 B,
// lane l's 16 B at l*16, word c = row g*128 + c*32 + l. A K-slab's tiles are then contiguous
// ([g][slab][4 x 512 B] for e4m3-per-16, one atom a slab for ue8m0-per-32) - the same order
// pd_nvf4_tc5_tile_scales gives the weights. Rows past `rows` in the last group are zero.
__global__ void pd_sf_swz128_kernel(const uint32_t* __restrict__ src, uint32_t* __restrict__ dst,
                                    uint32_t rows, uint32_t n_w, uint32_t n_out) {
    asm volatile("griddepcontrol.launch_dependents;");
    asm volatile("griddepcontrol.wait;" ::: "memory");
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_out) return;
    const uint32_t a = i >> 7, w = i & 127u;
    const uint32_t g = a / n_w, k = a % n_w;
    const uint32_t r = g * 128u + (w & 3u) * 32u + (w >> 2);
    dst[i] = r < rows ? __ldg(src + (size_t)r * n_w + k) : 0u;
}

// v2 W4A4 GEMM: the pd_nvf4_gemm_f4 contract, weight scales pre-tiled by
// pd_nvf4_tc5_tile_scales (`scale_tiled`), activation scales row-major as the
// quantizers write them. `part` (part_floats) holds K-split partials; none = unsplit.
PD_EXPORT
int pd_nvf4_gemm_tc5w(const void* data, const void* scale_tiled, const void* bias, const void* xq,
                      const void* xs, void* y, float scale2, uint32_t in_dim, uint32_t out_dim,
                      uint32_t batch, void* part, uint64_t part_floats, void* stream) {
#if !defined(PD_TC5_HOST)
    (void)data; (void)scale_tiled; (void)bias; (void)xq; (void)xs; (void)y; (void)scale2;
    (void)in_dim; (void)out_dim; (void)batch; (void)part; (void)part_floats; (void)stream;
    return cudaErrorNotSupported;
#else
    if (out_dim == 0 || batch == 0) return 0;
    if ((in_dim & 255u) != 0u) return cudaErrorInvalidValue;
    static int ok = -1;
    if (ok < 0) {
        int dev = 0, ccM = 0, ccm = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&ccM, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&ccm, cudaDevAttrComputeCapabilityMinor, dev);
        ok = (ccM == 10 && ccm == 0 && pd_tmap_encode() != nullptr) ? 1 : 0;
    }
    if (!ok) return cudaErrorNotSupported;
    const uint32_t bn = batch <= 32u ? 32u : (batch <= 64u ? 64u : 128u);
    CUtensorMap wm, xm;
    if (!pd_nv4tc5_tmap(&wm, data, in_dim / 2u, out_dim, 128u) ||
        !pd_nv4tc5_tmap(&xm, xq, in_dim / 2u, batch, bn))
        return cudaErrorInvalidValue;
    const uint32_t tiles = ((out_dim + 127u) / 128u) * ((batch + bn - 1u) / bn);
    const size_t plane = (size_t)batch * out_dim;
    const uint32_t kz = part ? pd_nv4tc5_split(tiles, in_dim >> 8, part_floats, plane, batch, pd_nv4tc5_split_bmax(64u)) : 1u;
    const cudaStream_t st = (cudaStream_t)stream;
    switch (bn) {
        case 32u: return pd_nv4tc5_ws_go<8u, 32u>(wm, xm, scale_tiled, xs, y, part, scale2, bias, in_dim, out_dim, batch, kz, st);
        case 64u: return pd_nv4tc5_ws_go<7u, 64u>(wm, xm, scale_tiled, xs, y, part, scale2, bias, in_dim, out_dim, batch, kz, st);
        default: {
            // the activation scales swizzled once into the stream's scratch, then bulk-copied
            // a slab at a time (kill PADDOCK_NO_NV4_XSW keeps the gather)
            static const bool no_xsw = pd_env("PADDOCK_NO_NV4_XSW") != nullptr;
            const uint32_t n_w = in_dim / 64u, n_rg = (batch + 127u) / 128u;
            unsigned char* xsw = no_xsw ? nullptr : pd_tc5_xsw(st, (size_t)n_rg * n_w * 512u);
            if (xsw) {
                const uint32_t n_out = n_rg * n_w * 128u;
                pd_pdl_go(pd_sf_swz128_kernel, dim3((n_out + 255u) / 256u), dim3(256), 0u, st,
                          (const uint32_t*)xs, (uint32_t*)xsw, batch, n_w, n_out);
            }
            return pd_nv4tc5_ws_go<6u, 128u>(wm, xm, scale_tiled, xs, y, part, scale2, bias, in_dim, out_dim,
                                             batch, kz, st, xsw);
        }
    }
#endif
}

// Slot 881: pd_nvf4_gemm_tc5w with the activation scales ALREADY in the swizzled atom layout (the
// producers' slots 879 / 880 write it), so no pd_sf_swz128 launch: xs holds ceil(batch / 128)
// * 128 rows' worth, read by the bulk scale copy as tc5w's own swizzle would have written it - the
// same bytes, the same mainloop. The BN-128 arm only: NotSupported at batch <= 64 (the narrow
// tiles gather row-major scales).
PD_EXPORT
int pd_nvf4_gemm_tc5w_sw(const void* data, const void* scale_tiled, const void* bias, const void* xq,
                         const void* xs, void* y, float scale2, uint32_t in_dim, uint32_t out_dim,
                         uint32_t batch, void* part, uint64_t part_floats, void* stream) {
#if !defined(PD_TC5_HOST)
    (void)data; (void)scale_tiled; (void)bias; (void)xq; (void)xs; (void)y; (void)scale2;
    (void)in_dim; (void)out_dim; (void)batch; (void)part; (void)part_floats; (void)stream;
    return cudaErrorNotSupported;
#else
    if (out_dim == 0 || batch == 0) return 0;
    if ((in_dim & 255u) != 0u) return cudaErrorInvalidValue;
    static int ok = -1;
    if (ok < 0) {
        int dev = 0, ccM = 0, ccm = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&ccM, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&ccm, cudaDevAttrComputeCapabilityMinor, dev);
        ok = (ccM == 10 && ccm == 0 && pd_tmap_encode() != nullptr) ? 1 : 0;
    }
    if (!ok || batch <= 64u) return cudaErrorNotSupported;
    CUtensorMap wm, xm;
    if (!pd_nv4tc5_tmap(&wm, data, in_dim / 2u, out_dim, 128u) ||
        !pd_nv4tc5_tmap(&xm, xq, in_dim / 2u, batch, 128u))
        return cudaErrorInvalidValue;
    const uint32_t tiles = ((out_dim + 127u) / 128u) * ((batch + 127u) / 128u);
    const size_t plane = (size_t)batch * out_dim;
    const uint32_t kz = part ? pd_nv4tc5_split(tiles, in_dim >> 8, part_floats, plane, batch, pd_nv4tc5_split_bmax(64u)) : 1u;
    return pd_nv4tc5_ws_go<6u, 128u>(wm, xm, scale_tiled, xs, y, part, scale2, bias, in_dim, out_dim,
                                     batch, kz, (cudaStream_t)stream, (const unsigned char*)xs);
#endif
}

// Slot 874: the interleaved gate|up GEMM with the swiglu + nvf4-quant epilogue (SWQ above) -
// slot 533's contract (q [batch][ff/2], qs [batch][ff/16], out_dim = 2 ff) on tcgen05, with the
// tc5w weight-scale tiling (`scale_tiled`). Bit-identical to {tc5w over the gate and up planes,
// quantize_nvf4_swiglu}: the same accumulators, value expression and quantizer math.
PD_EXPORT
int pd_nvf4_gemm_tc5_swq(const void* data, const void* scale_tiled, const void* xq, const void* xs,
                         void* q, void* qs, float scale2, uint32_t in_dim, uint32_t out_dim,
                         uint32_t batch, void* stream) {
#if !defined(PD_TC5_HOST)
    (void)data; (void)scale_tiled; (void)xq; (void)xs; (void)q; (void)qs; (void)scale2;
    (void)in_dim; (void)out_dim; (void)batch; (void)stream;
    return cudaErrorNotSupported;
#else
    if (out_dim == 0 || batch == 0) return 0;
    if ((in_dim & 255u) != 0u || (out_dim & 255u) != 0u) return cudaErrorInvalidValue;
    static int ok = -1;
    if (ok < 0) {
        int dev = 0, ccM = 0, ccm = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&ccM, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&ccm, cudaDevAttrComputeCapabilityMinor, dev);
        ok = (ccM == 10 && ccm == 0 && pd_tmap_encode() != nullptr) ? 1 : 0;
    }
    if (!ok) return cudaErrorNotSupported;
    // BN 128 at every width (the swapped mma's M): rows past the batch zero-fill and are not stored
    CUtensorMap wm, xm;
    if (!pd_nv4tc5_tmap(&wm, data, in_dim / 2u, out_dim, 128u) ||
        !pd_nv4tc5_tmap(&xm, xq, in_dim / 2u, batch, 128u))
        return cudaErrorInvalidValue;
    const cudaStream_t st = (cudaStream_t)stream;
    static const bool no_xsw = pd_env("PADDOCK_NO_NV4_XSW") != nullptr;
    const uint32_t n_w = in_dim / 64u, n_rg = (batch + 127u) / 128u;
    unsigned char* xsw = no_xsw ? nullptr : pd_tc5_xsw(st, (size_t)n_rg * n_w * 512u);
    if (xsw) {
        const uint32_t n_out = n_rg * n_w * 128u;
        pd_pdl_go(pd_sf_swz128_kernel, dim3((n_out + 255u) / 256u), dim3(256), 0u, st,
                  (const uint32_t*)xs, (uint32_t*)xsw, batch, n_w, n_out);
    }
    return pd_nv4tc5_swq_go<6u, 128u>(wm, xm, scale_tiled, xs, q, qs, scale2, in_dim, out_dim, batch, st,
                                      xsw);
#endif
}

// Block-scaled e4m3 GEMM on the same tcgen05 pipeline (F8 instantiation): y[batch][out] =
// W8 . X8^T with both operands' ue8m0-per-32 scales applied by the mma - the f8d_gemm_mma_ks
// contract (data/scale = the W8 plane, xq/xs = quantize_e4m3's output), for the decode and
// unified-tick projections that ran mma.sync on B200. in_dim % 128; `part` as in tc5w.
PD_EXPORT
int pd_f8_gemm_tc5x(const void* data, const void* scale, const void* xq, const void* xs, void* y,
                    uint32_t in_dim, uint32_t out_dim, uint32_t batch, void* part,
                    uint64_t part_floats, void* stream) {
#if !defined(PD_TC5_HOST)
    (void)data; (void)scale; (void)xq; (void)xs; (void)y; (void)in_dim; (void)out_dim;
    (void)batch; (void)part; (void)part_floats; (void)stream;
    return cudaErrorNotSupported;
#else
    if (out_dim == 0 || batch == 0) return 0;
    if ((in_dim & 127u) != 0u) return cudaErrorInvalidValue;
    static int ok = -1;
    if (ok < 0) {
        int dev = 0, ccM = 0, ccm = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&ccM, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&ccm, cudaDevAttrComputeCapabilityMinor, dev);
        ok = (ccM == 10 && ccm == 0 && pd_tmap_encode() != nullptr) ? 1 : 0;
    }
    if (!ok) return cudaErrorNotSupported;
    const uint32_t bn = batch <= 32u ? 32u : (batch <= 64u ? 64u : 128u);
    CUtensorMap wm, xm;
    if (!pd_nv4tc5_tmap(&wm, data, in_dim, out_dim, 128u) ||
        !pd_nv4tc5_tmap(&xm, xq, in_dim, batch, bn))
        return cudaErrorInvalidValue;
    const uint32_t tiles = ((out_dim + 127u) / 128u) * ((batch + bn - 1u) / bn);
    const size_t plane = (size_t)batch * out_dim;
    const uint32_t kz = part ? pd_nv4tc5_split(tiles, in_dim >> 7, part_floats, plane, batch, pd_nv4tc5_split_bmax(16u)) : 1u;
    const cudaStream_t st = (cudaStream_t)stream;
    switch (bn) {
        case 32u: return pd_nv4tc5_ws_go<8u, 32u, true>(wm, xm, scale, xs, y, part, 1.0f, nullptr, in_dim, out_dim, batch, kz, st);
        case 64u: return pd_nv4tc5_ws_go<8u, 64u, true>(wm, xm, scale, xs, y, part, 1.0f, nullptr, in_dim, out_dim, batch, kz, st);
        default:  return pd_nv4tc5_ws_go<6u, 128u, true>(wm, xm, scale, xs, y, part, 1.0f, nullptr, in_dim, out_dim, batch, kz, st);
    }
#endif
}

// Slot 883: the load-time repack of a W8 plane's row-major ue8m0-per-32 scale cells ([out][in / 32]
// bytes) into the tiles the e4m3 bulk path reads (slot 882): one 512 B canonical tile per (128-row
// weight tile, 128-K slab), laid (tile, slab)-major - pd_sf_swz128 with a slab's four cells as
// its word. tiled: ceil(out / 128) * (in / 128) * 512 bytes; rows past `out` are zero.
PD_EXPORT
int pd_f8_tc5_tile_scales(const void* scale, void* tiled, uint32_t in_dim, uint32_t out_dim,
                          void* stream) {
    if (out_dim == 0) return 0;
    if ((in_dim & 127u) != 0u) return cudaErrorInvalidValue;
#if defined(PD_TC5_HOST)
    pd_nv4tc5_ctr_prepare();   // load time, outside every capture: 882's split counters
    pd_tc5_pool_prepare();     // and the per-stream swizzled-scale scratch it swizzles into
#endif
    const uint32_t n_w = in_dim / 128u, n_out = (out_dim + 127u) / 128u * n_w * 128u;
    pd_sf_swz128_kernel<<<(n_out + 255u) / 256u, 256, 0, (cudaStream_t)stream>>>(
        (const uint32_t*)scale, (uint32_t*)tiled, out_dim, n_w, n_out);
    return (int)cudaPeekAtLastError();
}

// Slot 882: pd_f8_gemm_tc5x with the weight scales pre-tiled by 883 (`scale_tiled`) and the
// activation scales swizzled once into the stream's scratch: both operands' 512 B scale tiles
// a slab come by bulk copy instead of 4 B cp.async gathers per row (which held the producer:
// the qkvz decode plane 21.8 -> 16.2 us with them removed). The same tiles, the same mainloop
// and K-split election as tc5x, so the same bits. The BN-128 arm only: NotSupported at
// batch <= 64. K in 256s and unsplit: the K2 mainloop (two slabs a stage). K sweep at 128 rows
// (out 16384, B200, back-to-back): ~360 ns a slab against the gathers' ~480, but ~8 us fixed
// against ~4.4 - the swizzle launch; 887 takes scales its producers wrote swizzled (884-886).
// `xs_sw`: xs is already in the swizzled layout (887) - no swizzle launch.
static int pd_f8_gemm_tc5x_t_go(const void* data, const void* scale_tiled, const void* xq, const void* xs,
                                void* y, uint32_t in_dim, uint32_t out_dim, uint32_t batch, void* part,
                                uint64_t part_floats, void* stream, bool xs_sw) {
#if !defined(PD_TC5_HOST)
    (void)data; (void)scale_tiled; (void)xq; (void)xs; (void)y; (void)in_dim; (void)out_dim;
    (void)batch; (void)part; (void)part_floats; (void)stream; (void)xs_sw;
    return cudaErrorNotSupported;
#else
    if (out_dim == 0 || batch == 0) return 0;
    if ((in_dim & 127u) != 0u) return cudaErrorInvalidValue;
    static int ok = -1;
    if (ok < 0) {
        int dev = 0, ccM = 0, ccm = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&ccM, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&ccm, cudaDevAttrComputeCapabilityMinor, dev);
        ok = (ccM == 10 && ccm == 0 && pd_tmap_encode() != nullptr) ? 1 : 0;
    }
    if (!ok || batch <= 64u) return cudaErrorNotSupported;
    CUtensorMap wm, xm;
    if (!pd_nv4tc5_tmap(&wm, data, in_dim, out_dim, 128u) ||
        !pd_nv4tc5_tmap(&xm, xq, in_dim, batch, 128u))
        return cudaErrorInvalidValue;
    const cudaStream_t st = (cudaStream_t)stream;
    const uint32_t n_w = in_dim / 128u, n_rg = (batch + 127u) / 128u;
    const unsigned char* xsw = (const unsigned char*)xs;
    if (!xs_sw) {
        unsigned char* sw = pd_tc5_xsw(st, (size_t)n_rg * n_w * 512u);
        if (!sw) return cudaErrorNotSupported;
        const uint32_t n_out = n_rg * n_w * 128u;
        pd_pdl_go(pd_sf_swz128_kernel, dim3((n_out + 255u) / 256u), dim3(256), 0u, st,
                  (const uint32_t*)xs, (uint32_t*)sw, batch, n_w, n_out);
        xsw = sw;
    }
    const uint32_t tiles = ((out_dim + 127u) / 128u) * n_rg;
    const size_t plane = (size_t)batch * out_dim;
    const uint32_t kz = part ? pd_nv4tc5_split(tiles, in_dim >> 7, part_floats, plane, batch, pd_nv4tc5_split_bmax(16u)) : 1u;
    return pd_nv4tc5_ws_go<6u, 128u, true>(wm, xm, scale_tiled, xs, y, part, 1.0f, nullptr, in_dim, out_dim,
                                           batch, kz, st, xsw);
#endif
}

PD_EXPORT
int pd_f8_gemm_tc5x_t(const void* data, const void* scale_tiled, const void* xq, const void* xs,
                      void* y, uint32_t in_dim, uint32_t out_dim, uint32_t batch, void* part,
                      uint64_t part_floats, void* stream) {
    return pd_f8_gemm_tc5x_t_go(data, scale_tiled, xq, xs, y, in_dim, out_dim, batch, part, part_floats,
                                stream, false);
}

// 887: 882 with `xs_sw` already in the swizzled atom layout (884 / 885 / 886 wrote it; ceil(batch /
// 128) * 128 rows' worth) - no swizzle launch, the same bits.
PD_EXPORT
int pd_f8_gemm_tc5x_tsw(const void* data, const void* scale_tiled, const void* xq, const void* xs_sw,
                        void* y, uint32_t in_dim, uint32_t out_dim, uint32_t batch, void* part,
                        uint64_t part_floats, void* stream) {
    return pd_f8_gemm_tc5x_t_go(data, scale_tiled, xq, xs_sw, y, in_dim, out_dim, batch, part, part_floats,
                                stream, true);
}

// Per-row-scale e4m3 GEMM on the same pipeline (ROW instantiation): pd_f8row_gemm's
// contract - data [out][in] e4m3 + w_rowscale f32 [out], xq [batch][in] e4m3 +
// x_rowscale f32 [batch], y[batch][out] = sum(w*x) * w_rowscale[row] * x_rowscale[col].
// in_dim % 128. Splits only through the serialized-K epilogue (one wave, stream
// counters); otherwise unsplit.
PD_EXPORT
int pd_f8row_gemm_tc5x(const void* data, const void* w_rowscale, const void* xq,
                       const void* x_rowscale, void* y, uint32_t in_dim, uint32_t out_dim,
                       uint32_t batch, void* stream) {
#if !defined(PD_TC5_HOST)
    (void)data; (void)w_rowscale; (void)xq; (void)x_rowscale; (void)y; (void)in_dim;
    (void)out_dim; (void)batch; (void)stream;
    return cudaErrorNotSupported;
#else
    if (out_dim == 0 || batch == 0) return 0;
    if ((in_dim & 127u) != 0u) return cudaErrorInvalidValue;
    static int ok = -1;
    if (ok < 0) {
        int dev = 0, ccM = 0, ccm = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&ccM, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&ccm, cudaDevAttrComputeCapabilityMinor, dev);
        ok = (ccM == 10 && ccm == 0 && pd_tmap_encode() != nullptr) ? 1 : 0;
    }
    if (!ok) return cudaErrorNotSupported;
    const uint32_t bn = batch <= 32u ? 32u : (batch <= 64u ? 64u : 128u);
    CUtensorMap wm, xm;
    if (!pd_nv4tc5_tmap(&wm, data, in_dim, out_dim, 128u) ||
        !pd_nv4tc5_tmap(&xm, xq, in_dim, batch, bn))
        return cudaErrorInvalidValue;
    const uint32_t tiles = ((out_dim + 127u) / 128u) * ((batch + bn - 1u) / bn);
    const cudaStream_t st = (cudaStream_t)stream;
    // split only where the serialized-K epilogue can carry it (it needs no partial plane)
    uint32_t kz = pd_nv4tc5_split(tiles, in_dim >> 7, ~(size_t)0 >> 1, 0u, batch, pd_nv4tc5_split_bmax(16u));
    if (kz > 1u && (tiles * kz > (uint32_t)pd_nv4tc5_nsm() || !pd_nv4tc5_ctr(st))) kz = 1u;
    switch (bn) {
        case 32u: return pd_nv4tc5_ws_go<8u, 32u, true, true>(wm, xm, w_rowscale, x_rowscale, y, y, 1.0f, nullptr, in_dim, out_dim, batch, kz, st);
        case 64u: return pd_nv4tc5_ws_go<8u, 64u, true, true>(wm, xm, w_rowscale, x_rowscale, y, y, 1.0f, nullptr, in_dim, out_dim, batch, kz, st);
        default:  return pd_nv4tc5_ws_go<6u, 128u, true, true>(wm, xm, w_rowscale, x_rowscale, y, y, 1.0f, nullptr, in_dim, out_dim, batch, kz, st);
    }
#endif
}
