// gemm/nvf4_pf.cuh - the W4A16 prefill tile: NVFP4 row planes x bf16 rows on
// mma.m16n8k16, shaped for prefill widths.
// Textually-included segment of the single pack translation unit.
// Not standalone-compilable: include order is defined by ../pack.cu.
//
// Why: a checkpoint whose file marks its tensors 16-bit-activation (Flint's
// NVFP4 GGUF) prefills at the bf16 class. Its decode-shaped consumers - the
// tc GEMM's 128 x 32 tiles and the W16 expert pair's per-warp walk - re-decode
// every weight tile once per 32 tokens and ran ~30-37 TFLOPS on GB10 at 8K.
// This is the bf16 prefill tile's shape (pd_bf16_gemm_mma_kernel's RAST + XB
// arm: 128 weight rows x 128 token columns, 8 warps of 64 x 32, KT 64 on a
// cp.async ring, ldmatrix fragments) with one change on the weight side: the
// ring carries the plane's NVFP4 codes (32 B a row a stage) and e4m3 scales
// (4 B), and the CTA decodes each element ONCE into a bf16 tile before the
// MMAs (pd_nvpf_dq2 - e2m1 x e4m3 is exact in bf16). The plane's f32 factor
// (per tensor, or per expert) rides the epilogue.
//
// One kernel, four tasks (PdNvPfMode):
//   DENSE  - y[c][r] = (W x_c)_r * s2, f32, row c of a strided output.
//   GU     - silu(g) * u of two planes (gate, up) whose rows interleave in
//            the tile 8 at a time: tile rows 16i..16i+7 are gate channels
//            8i..8i+7 and 16i+8..16i+15 the same up channels, so a thread's
//            fragment rows g and g + 8 are one channel's gate and up and the
//            SwiGLU happens in registers. 64 channels a tile, bf16 out.
//   MOE_GU - GU over moe_align_bm's sorted 64-token blocks of one expert.
//   MOE_DN - the expert down over sorted (token, slot) rows of the GU output,
//            landing per-slot bf16 partials [rows][k][embd] with the top-k
//            weight and the expert factor folded in; moe_slot_combine_bf16
//            adds them onto the shared expert's f32 output in slot order (the
//            f32 [k + 1] partials cost ~400 MB of writes a layer at 8K).
//
// Row invariant: an output element is one weight row against one token
// column, accumulated over K in ascending 64-wide tiles of four k16 MMAs - no
// split-K, no dependence on which other tokens share the tile. A token's bits
// are the same alone, in a short tail or in a full chunk.

#define PD_NVPF_BM 128u      // weight rows a tile

// The decode needs e4m3 conversion (sm_89) and the MMAs bf16 (sm_80). The
// helpers are this file's own, named pd_nvpf_*, so the tile compiles in a
// light TU (bench/nvf4_pf_gb10_bench.cu); the dequant is pd_nvpf_dq2's
// arithmetic verbatim (moe/nvf4_w16.cuh), so both classes read the same bits.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
#define PD_NVPF_OK 1
#else
#define PD_NVPF_OK 0
#endif

#if PD_NVPF_OK
__device__ __forceinline__ void pd_nvpf_cpa16(void* smem, const void* gmem, bool ok) {
    const unsigned sm = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(sm), "l"(gmem),
                 "r"(ok ? 16u : 0u));
}
// weight-code copies with an L2 prefetch of the line around them: a stage
// reads 32 code bytes of a row (rows 1 KB apart at K 2048), so without it
// every DRAM burst served one stage's sliver of its line
#ifndef PD_NVPF_PF
#define PD_NVPF_PF "256B"
#endif
__device__ __forceinline__ void pd_nvpf_cpa16p(void* smem, const void* gmem, bool ok) {
    const unsigned sm = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global.L2::" PD_NVPF_PF " [%0], [%1], 16, %2;" ::"r"(sm),
                 "l"(gmem), "r"(ok ? 16u : 0u));
}
__device__ __forceinline__ void pd_nvpf_cpa4p(void* smem, const void* gmem, bool ok) {
    const unsigned sm = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global.L2::" PD_NVPF_PF " [%0], [%1], 4, %2;" ::"r"(sm),
                 "l"(gmem), "r"(ok ? 4u : 0u));
}
__device__ __forceinline__ void pd_nvpf_cpa8(void* smem, const void* gmem, bool ok) {
    const unsigned sm = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;" ::"r"(sm), "l"(gmem),
                 "r"(ok ? 8u : 0u));
}
__device__ __forceinline__ void pd_nvpf_cpa4(void* smem, const void* gmem, bool ok) {
    const unsigned sm = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(sm), "l"(gmem),
                 "r"(ok ? 4u : 0u));
}
__device__ __forceinline__ void pd_nvpf_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
__device__ __forceinline__ void pd_nvpf_wait() {
    asm volatile("cp.async.wait_group %0;" ::"n"(N));
}
__device__ __forceinline__ void pd_nvpf_ldm4(const void* p, uint32_t& r0, uint32_t& r1,
                                             uint32_t& r2, uint32_t& r3) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(a));
}
__device__ __forceinline__ void pd_nvpf_ldm2(const void* p, uint32_t& r0, uint32_t& r1) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                 : "=r"(r0), "=r"(r1)
                 : "r"(a));
}
__device__ __forceinline__ void pd_nvpf_mma(float d[4], const uint32_t a[4], const uint32_t b[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// one code byte (k even low) -> bf16x2 {k even, k odd} x the block scale
__device__ __forceinline__ uint32_t pd_nvpf_dq2(uint32_t b, uint32_t s2) {
    const uint32_t m0 = b & 7u, m1 = (b >> 4) & 7u;
    const uint32_t sel = m0 | (m1 << 4);
    const uint32_t hi = __byte_perm(0x3F3F3F00u, 0x40404040u, sel);
    const uint32_t lo = __byte_perm(0xC0800000u, 0xC0804000u, sel);
    uint32_t v = __byte_perm(lo, hi, 0x5140);
    v |= ((b & 0x08u) << 12) | ((b & 0x80u) << 24);
    uint32_t r;
#if __CUDA_ARCH__ >= 900
    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(v), "r"(s2));
#else
    asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(r) : "r"(v), "r"(s2), "r"(0x80008000u));
#endif
    return r;
}
// the e4m3 block scale as bf16x2 {s, s} (exact)
__device__ __forceinline__ uint32_t pd_nvpf_scale2(uint32_t sb) {
    const float s = (float)reinterpret_cast<const __nv_fp8_e4m3&>(sb);
    const uint32_t h = __float_as_uint(s) >> 16;
    return h | (h << 16);
}
#endif

// The lifted decode (sm_90+, where bf16x2 has a plain multiply): a code's
// sign and three magnitude bits dropped onto a bf16's sign and mantissa top
// read as e2m1 x 2^-126 exactly (0.5 is the subnormal 2^-127), so ONE
// multiply by the block scale x 2^119 gives the weight x 2^-7 - an exact
// product (<= 5 significant bits, no overflow at e4m3's 448, no underflow at
// its 2^-9), and the epilogue's factor carries the 2^7 back (exact). Element
// pair j of a code word is one byte-permute (byte 1 <- word << 4's byte j,
// byte 3 <- the word's byte j: the two codes land at bits 12-15 / 28-31)
// and two logic ops - ~5 ALU ops a pair where pd_nvpf_dq2 runs ~11, which
// was more of the tile's issue than its MMAs. The decoded value x 2^7 is
// pd_nvpf_dq2's, bit for bit (every scale byte x every code byte, checked).
#ifndef PD_NVPF_LIFT
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#define PD_NVPF_LIFT 1
#else
#define PD_NVPF_LIFT 0
#endif
#endif
#if PD_NVPF_OK && PD_NVPF_LIFT
__device__ __forceinline__ uint32_t pd_nvpf_lscale2(uint32_t sb) {
    const float s = (float)reinterpret_cast<const __nv_fp8_e4m3&>(sb) * 0x1p119f;
    const uint32_t h = __float_as_uint(s) >> 16;
    return h | (h << 16);
}
// pair j of code word w (w4 = w << 4) x the lifted scale
__device__ __forceinline__ uint32_t pd_nvpf_ldq(uint32_t w, uint32_t w4, uint32_t j, uint32_t s2) {
    const uint32_t q = __byte_perm(w, w4, ((4u + j) << 4) | (j << 12));
    const uint32_t v = (q & 0x80008000u) | ((q >> 6) & 0x01C001C0u);
    uint32_t r;
    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(v), "r"(s2));
    return r;
}
#endif

enum PdNvPfMode : uint32_t { PD_NVPF_DENSE = 0, PD_NVPF_GU = 1, PD_NVPF_MOE_GU = 2, PD_NVPF_MOE_DN = 3 };

struct PdNvPfArgs {
    // plane A (gate for GU), plane B (up, GU modes only)
    const uint8_t* da; const uint8_t* sa; const float* fa;   // codes, scales, factor(s)
    const uint8_t* db; const uint8_t* sb; const float* fb;
    float fa1, fb1;            // the per-tensor factors (dense modes)
    const __nv_bfloat16* x;    // bf16 rows
    uint32_t ldx, xoff;        // row stride / column offset of x (elements)
    float* y;                  // DENSE / MOE_DN output (f32)
    __nv_bfloat16* act;        // GU / MOE_GU output (bf16)
    uint32_t ldy, yoff;        // row stride / column offset of the output
    uint32_t K, M, ntok;       // in_dim, weight rows (channels for GU), token rows
    // MoE: moe_align_bm(64)'s sorting
    const uint32_t* srow; const uint32_t* sslot; const uint32_t* bexp;
    const float* topk_w; uint32_t k, n_blocks;
    // NR (slot 934's q|k|v grid): the q / k norm + rope in the epilogue, each
    // 128-row weight tile one head. nr 1 = q (f16 x qsc into q16 from row
    // dec on, f32 into y2 below it - a decode band's rows), 2 = k (f32 into
    // y); 0 = the plain epilogue (v)
    uint32_t nr, dec;
    const float* nw; const unsigned int* pos; __half* q16; float* y2;
    float eps, ts, fs, clo, chi, ext, msc, qsc;
};


// OVL: the codes run two stages ahead and the decode of stage k + 1 shares
// stage k's barrier interval with its MMAs (two bf16 tiles, a 3-deep code
// ring); off, a stage copies, decodes, then multiplies, each behind its own
// barrier. Same k walk either way - the bits do not move.
#ifndef PD_NVPF_OVL_DEFAULT
#define PD_NVPF_OVL_DEFAULT 0
#endif
#ifndef PD_NVPF_RING_DENSE
#define PD_NVPF_RING_DENSE 2
#endif
#ifndef PD_NVPF_BN_DENSE
#define PD_NVPF_BN_DENSE 64
#endif
#ifndef PD_NVPF_KT_DENSE
#define PD_NVPF_KT_DENSE 64u
#endif
#ifndef PD_NVPF_DN_MPER
#define PD_NVPF_DN_MPER 4u
#endif
// RING: the serial arm's ring depth (stage k + RING - 1 streams in while
// stage k decodes and multiplies).
// KT: the k-tile (64, or 32 for the 128-token dense tile: the same smem at
// twice the tokens a decoded weight feeds); rows pad 8 halves (ldmatrix
// conflict-free at both).
// CW: k-tiles a weight copy carries (the code ring's slots hold CW k-tiles
// of codes + scales, two slots; tokens keep the one-tile ring)
__host__ __device__ constexpr uint32_t pd_nvpf_smem(uint32_t bn, bool ovl, uint32_t ring,
                                                    uint32_t kt = 64u, bool rd = false,
                                                    uint32_t cw = 1u) {
    return (rd ? 0u : ovl ? 2u : 1u) * PD_NVPF_BM * (kt + 8u) * 2u   // bf16 weight tile(s)
           + (ovl ? 2u : ring) * bn * (kt + 8u) * 2u          // token ring
           + (ovl ? 3u : ring) * PD_NVPF_BM * (kt / 2u + 4u) * cw;  // code + scale ring
}

// BN token columns a tile (128 dense, 64 MoE: moe_align_bm's block).
// MPER (MoE only): a CTA walks MPER consecutive weight tiles of its block
// on one continuous cp.async ring - the routed down's K is 512, eight
// k-tiles, and a tile that paid its own prologue (the sorting's gathers)
// and ring fill for them was latency-bound (barrier + long-scoreboard
// stalls). Each output's K walk is unchanged, so the bits are too.
//
// RD (register decode): each warp owns 16 weight rows across all BN tokens
// and decodes its MMA A fragments straight from the staged codes into
// registers - no shared bf16 weight tile, no decode phase or its barrier,
// and no 4-way decode redundancy across column warps (each element is
// decoded once, by the lane whose fragment holds it). Lane (g, t) needs
// pair t of each code word of rows g and g + 8: one runtime byte-permute
// selector serves every word. Same MMAs in the same order - same bits.
// The tile's body on block `bx` of its plane's grid: pd_nvpf_kernel runs it on
// its own block, pd_nvpf3_kernel on the plane a fused grid's block falls in.
template <uint32_t MODE, uint32_t BN, bool OVL, uint32_t RING, uint32_t MPER = 1u,
          uint32_t KT = 64u, bool RD = false, uint32_t CW = 1u, bool NR = false>
__device__ __forceinline__ void pd_nvpf_body(const PdNvPfArgs& p, uint32_t bx) {
#if PD_NVPF_OK
    constexpr uint32_t BM = PD_NVPF_BM, KPAD = KT + 8u;
    // a row's codes a stage (CB bytes) go in two pieces, one thread each
    constexpr uint32_t CB = KT / 2u, PIECE = CB / 2u, PE = 2u * PIECE;
    static_assert(KT == 64u || KT == 32u, "k-tile");
    constexpr uint32_t NTH = 256u, ST = OVL ? 2u : RING, NA = RD ? 0u : OVL ? 2u : 1u,
                       NC = OVL ? 3u : RING;
    static_assert(RING >= 2u && RING <= 4u, "ring depth");
    // RD's decode is the lifted one: an RD body built below sm_90 computes
    // nothing, and the launchers never elect it there (pd_nvpf_rd_ok)
    static_assert(!RD || (!OVL && KT == 64u), "register decode: the serial arm, KT 64");
    static_assert(CW == 1u || (RD && RING == 2u && (CW == 2u || CW == 4u)), "wide code copies: RD, two slots");
    constexpr uint32_t RG = RD ? 1u : 4u, WR = RD ? 8u : 2u, WC = RD ? 1u : 4u, WN = BN / WC,
                       CG = WN / 8u;
    constexpr bool GU = MODE == PD_NVPF_GU || MODE == PD_NVPF_MOE_GU;
    constexpr bool MOE = MODE == PD_NVPF_MOE_GU || MODE == PD_NVPF_MOE_DN;
    static_assert(CG >= 1u && WN % 8u == 0u, "warp tile");
    static_assert(MPER == 1u || (MOE && !OVL), "the multi-tile walk is the MoE serial arm's");

    extern __shared__ __align__(16) unsigned char pd_nvpf_dyn[];
    __nv_bfloat16* sh_a = reinterpret_cast<__nv_bfloat16*>(pd_nvpf_dyn);   // [NA][BM][KPAD]
    __nv_bfloat16* sh_b = sh_a + NA * BM * KPAD;                           // [ST][BN][KPAD]
    uint8_t* sh_c = reinterpret_cast<uint8_t*>(sh_b + ST * BN * KPAD);     // [NC][BM][CB * CW]
    uint8_t* sh_s = sh_c + NC * BM * CB * CW;                              // [NC][BM][4 * CW]
    __shared__ const __nv_bfloat16* xrow[BN];
    __shared__ uint32_t orow[BN];   // output row (token, or token*(k+1)+slot) or PAD
    __shared__ float owt[BN];       // MOE_DN: top-k weight x expert factor

    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, t = lane & 3u;
    const uint32_t wr = (warp % WR) * (RG * 16u), wc = (warp / WR) * WN;

    // ---- the tile's weight rows and token columns
    uint32_t tile_m, tile_n = 0, e = 0;
    const uint32_t mt = GU ? p.M / 64u : (p.M + BM - 1u) / BM;   // weight tiles
    if constexpr (MOE) {
        const uint32_t mtc = mt / MPER;
        const uint32_t blk = bx / mtc;
        tile_m = (bx % mtc) * MPER;
        e = p.bexp[blk];
        if (e == PD_MOE_PAD) return;
        if (tid < BN) {
            const uint32_t s = blk * BN + tid;
            const uint32_t tok = p.srow[s];
            const bool live = tok != PD_MOE_PAD;
            const uint32_t slot = live ? p.sslot[s] : 0u;
            const uint32_t t0 = live ? tok : p.srow[blk * BN];
            if constexpr (MODE == PD_NVPF_MOE_GU) {
                xrow[tid] = p.x + (size_t)t0 * p.ldx + p.xoff;
                orow[tid] = live ? tok : PD_MOE_PAD;
                owt[tid] = (float)slot;   // GU: the slot rides here
            } else {
                xrow[tid] = p.x + (size_t)t0 * p.ldx + p.xoff + (size_t)slot * p.K;
                orow[tid] = live ? tok * p.k + slot : PD_MOE_PAD;
                owt[tid] = live ? p.topk_w[(size_t)tok * p.k + slot] * p.fa[e] : 0.0f;
            }
        }
    } else {
        // the grouped walk: every token tile of a ~8 MB group of activation
        // rows for one weight tile, then the next weight tile
        const uint32_t nt = (p.ntok + BN - 1u) / BN;
        const uint32_t row_bytes = BN * p.K * 2u;
        uint32_t grp = (8u << 20) / (row_bytes ? row_bytes : 1u);
        grp = grp < 1u ? 1u : (grp > nt ? nt : grp);
        const uint32_t per = grp * mt, gi = bx / per, in = bx % per;
        const uint32_t n0 = gi * grp, gsz = (nt - n0) < grp ? (nt - n0) : grp;
        tile_n = n0 + in % gsz;
        tile_m = in / gsz;
        if (tid < BN) {
            const uint32_t c = tile_n * BN + tid;
            const bool live = c < p.ntok;
            xrow[tid] = p.x + (size_t)(live ? c : 0u) * p.ldx + p.xoff;
            orow[tid] = live ? c : PD_MOE_PAD;
        }
    }
    __syncthreads();

    // tile row rr of weight tile tm -> (plane, plane row)
    auto plane_row = [&](uint32_t tm, uint32_t rr, const uint8_t*& d, const uint8_t*& s,
                         uint32_t& prow) {
        if constexpr (GU) {
            const uint32_t ch = tm * 64u + (rr >> 4) * 8u + (rr & 7u);
            const bool up = (rr & 8u) != 0u;
            d = up ? p.db : p.da;
            s = up ? p.sb : p.sa;
            prow = (MOE ? e * p.M : 0u) + ch;
        } else {
            d = p.da;
            s = p.sa;
            prow = (MOE ? e * p.M : 0u) + tm * BM + rr;
        }
    };
    const uint32_t kb = p.K >> 1, ks = p.K >> 4;   // code / scale bytes a row
    // weights: CB code bytes (two pieces) + the 4 scale bytes of the
    // 64-wide k span holding the stage (at KT 32 two stages read one word)
    auto stage_codes = [&](uint32_t tm, uint32_t k0, uint32_t buf) {
        if constexpr (CW > 1u) {
            // CW k-tiles: 32 * CW contiguous code bytes a row (GB10's DRAM moves
            // 32-byte row slivers at about half the rate of 64-byte pieces) and
            // their 4 * CW scale bytes
            for (uint32_t i = tid; i < BM * 2u * CW; i += NTH) {
                const uint32_t rr = i / (2u * CW), h = i % (2u * CW);
                const uint8_t *d, *s;
                uint32_t prow;
                plane_row(tm, rr, d, s, prow);
                const bool ok = GU || (tm * BM + rr) < p.M;
                pd_nvpf_cpa16(sh_c + (buf * BM + rr) * CB * CW + h * 16u,
                              d + (size_t)prow * kb + (k0 >> 1) + h * 16u, ok);
                if (h == 0u) {
                    uint8_t* sd = sh_s + (buf * BM + rr) * 4u * CW;
                    const uint8_t* sg = s + (size_t)prow * ks + (k0 >> 4);
                    if constexpr (CW == 4u) pd_nvpf_cpa16(sd, sg, ok);
                    else pd_nvpf_cpa8(sd, sg, ok);
                }
            }
            return;
        }
        for (uint32_t i = tid; i < BM * 2u; i += NTH) {
            const uint32_t rr = i >> 1, h = i & 1u;
            const uint8_t *d, *s;
            uint32_t prow;
            plane_row(tm, rr, d, s, prow);
            const bool ok = GU || (tm * BM + rr) < p.M;
            uint8_t* dst = sh_c + (buf * BM + rr) * CB + h * PIECE;
            const uint8_t* src = d + (size_t)prow * kb + (k0 >> 1) + h * PIECE;
            if constexpr (PIECE == 16u) pd_nvpf_cpa16p(dst, src, ok);
            else pd_nvpf_cpa8(dst, src, ok);
            if (h == 0u)
                pd_nvpf_cpa4p(sh_s + (buf * BM + rr) * 4u,
                              s + (size_t)prow * ks + ((k0 >> 4) & ~3u), ok);
        }
    };
    // tokens: KT bf16 = KT / 8 16 B pieces a row
    // Padding columns (a block's unfilled slots, rows past the chunk) are
    // skipped, not zero-filled: an MMA output column reads only its own token
    // column and a padding column is never stored, so whatever the slot holds
    // stays there. At 800 tokens 71% of a routed block's rows are padding and
    // their zero-fill copies were MIO queue pressure the weight stream waited
    // behind.
    auto stage_tokens = [&](uint32_t k0, uint32_t buf) {
        for (uint32_t i = tid; i < BN * (KT / 8u); i += NTH) {
            const uint32_t c = i / (KT / 8u), h8 = (i % (KT / 8u)) * 8u;
            if (orow[c] != PD_MOE_PAD)
                pd_nvpf_cpa16(&sh_b[(buf * BN + c) * KPAD + h8], xrow[c] + k0 + h8, true);
        }
    };
    // decode stage `buf` (k-tile at k0) into the bf16 tile: a thread takes
    // one piece of a row - PIECE code bytes, PE elements, PE / 16 scale
    // groups (their bytes sit at (k0 / 16) % 4 in the staged word)
    auto decode = [&](uint32_t buf, uint32_t abuf, uint32_t k0) {
        constexpr uint32_t NW = PIECE / 4u;   // code words a piece
        for (uint32_t i = tid; i < BM * 2u; i += NTH) {
            const uint32_t rr = i >> 1, h = i & 1u;
            uint32_t w[4];
            if constexpr (NW == 4u) {
                const uint4 c4 = *reinterpret_cast<const uint4*>(sh_c + (buf * BM + rr) * CB + h * PIECE);
                w[0] = c4.x; w[1] = c4.y; w[2] = c4.z; w[3] = c4.w;
            } else {
                const uint2 c2 = *reinterpret_cast<const uint2*>(sh_c + (buf * BM + rr) * CB + h * PIECE);
                w[0] = c2.x; w[1] = c2.y; w[2] = 0u; w[3] = 0u;
            }
            const uint32_t sw = *reinterpret_cast<const uint32_t*>(sh_s + (buf * BM + rr) * 4u);
            const uint32_t sb0 = ((k0 >> 4) & 3u) + h * (PE / 16u);
            const uint32_t s0 = (sw >> (8u * sb0)) & 0xFFu;
            const uint32_t s1 = (sw >> (8u * (sb0 + (PE / 16u) - 1u))) & 0xFFu;
            uint32_t o[16];
#if PD_NVPF_LIFT
            const uint32_t sg0 = pd_nvpf_lscale2(s0), sg1 = pd_nvpf_lscale2(s1);
#pragma unroll
            for (uint32_t q = 0; q < NW; ++q) {
                const uint32_t sc = q < NW / 2u ? sg0 : sg1, w4 = w[q] << 4;
#pragma unroll
                for (uint32_t j = 0; j < 4u; ++j) o[q * 4u + j] = pd_nvpf_ldq(w[q], w4, j, sc);
            }
#else
            const uint32_t sg0 = pd_nvpf_scale2(s0), sg1 = pd_nvpf_scale2(s1);
#pragma unroll
            for (uint32_t q = 0; q < NW; ++q) {
                const uint32_t sc = q < NW / 2u ? sg0 : sg1;
#pragma unroll
                for (uint32_t b = 0; b < 4u; ++b) o[q * 4u + b] = pd_nvpf_dq2((w[q] >> (8u * b)) & 0xFFu, sc);
            }
#endif
            uint4* dst = reinterpret_cast<uint4*>(&sh_a[(abuf * BM + rr) * KPAD + h * PE]);
#pragma unroll
            for (uint32_t q = 0; q < NW; ++q)
                dst[q] = make_uint4(o[4u * q], o[4u * q + 1u], o[4u * q + 2u], o[4u * q + 3u]);
        }
    };

    const uint32_t l7 = lane & 7u;
    const uint32_t a_roff = ((lane & 8u) ? 8u : 0u) + l7;
    const uint32_t a_kof = (lane & 16u) ? 8u : 0u;
    const uint32_t b_kof = (lane & 8u) ? 8u : 0u;
    float acc[RG][CG][4] = {};
    auto compute = [&](uint32_t abuf, uint32_t buf) {
#pragma unroll
        for (uint32_t sk = 0; sk < KT / 16u; ++sk) {
            const uint32_t ko = sk * 16u;
            uint32_t a[RG][4];
#pragma unroll
            for (uint32_t rg = 0; rg < RG; ++rg)
                pd_nvpf_ldm4(&sh_a[(abuf * BM + wr + rg * 16u + a_roff) * KPAD + ko + a_kof], a[rg][0],
                                a[rg][1], a[rg][2], a[rg][3]);
            uint32_t b[CG][2];
#pragma unroll
            for (uint32_t cg = 0; cg < CG; ++cg)
                pd_nvpf_ldm2(&sh_b[(buf * BN + wc + cg * 8u + l7) * KPAD + ko + b_kof], b[cg][0],
                                b[cg][1]);
#pragma unroll
            for (uint32_t rg = 0; rg < RG; ++rg)
#pragma unroll
                for (uint32_t cg = 0; cg < CG; ++cg) pd_nvpf_mma(acc[rg][cg], a[rg], b[cg]);
        }
    };

    // ---- epilogue: D d0=(m=g, n=2t) d1=(g, 2t+1) d2=(g+8, 2t) d3=(g+8, 2t+1)
    // the lifted decode's weights are 2^-7 of the true ones: the factor
    // takes 2^7 back (exact either way round)
    constexpr float LIFT = PD_NVPF_LIFT ? 128.0f : 1.0f;
    auto epilogue = [&](uint32_t tm) {
#pragma unroll
    for (uint32_t rg = 0; rg < RG; ++rg) {
#pragma unroll
        for (uint32_t cg = 0; cg < CG; ++cg) {
#pragma unroll
            for (uint32_t h = 0; h < 2u; ++h) {
                const uint32_t col = wc + cg * 8u + 2u * t + h;
                const uint32_t orw = orow[col];
                if (orw == PD_MOE_PAD) continue;
                if constexpr (GU) {
                    // rows g / g + 8 of m-tile (wr/16 + rg): one channel's gate and up
                    const uint32_t ch = tm * 64u + ((wr >> 4) + rg) * 8u + g;
                    const float gf = (MOE ? p.fa[e] : p.fa1) * LIFT;
                    const float uf = (MOE ? p.fb[e] : p.fb1) * LIFT;
                    const float gv = acc[rg][cg][h] * gf, uv = acc[rg][cg][2u + h] * uf;
                    const float v = gv * (1.0f / (1.0f + expf(-gv))) * uv;
                    const uint32_t slot = MOE ? (uint32_t)owt[col] : 0u;
                    p.act[(size_t)orw * p.ldy + p.yoff + slot * p.M + ch] = __float2bfloat16_rn(v);
                } else if constexpr (MOE) {
                    const uint32_t r0 = tm * BM + wr + rg * 16u + g;
                    const float f = owt[col] * LIFT;
                    __nv_bfloat16* yr = p.act + (size_t)orw * p.ldy;
                    yr[r0] = __float2bfloat16_rn(acc[rg][cg][h] * f);
                    yr[r0 + 8u] = __float2bfloat16_rn(acc[rg][cg][2u + h] * f);
                } else {
                    const uint32_t r0 = tm * BM + wr + rg * 16u + g;
                    const float f = p.fa1 * LIFT;
                    float* yr = p.y + (size_t)orw * p.ldy + p.yoff;
                    if (r0 < p.M) yr[r0] = acc[rg][cg][h] * f;
                    if (r0 + 8u < p.M) yr[r0 + 8u] = acc[rg][cg][2u + h] * f;
                }
            }
        }
    }
    };

    // NR: the f32 values the plain epilogue would store, 32 tokens at a time
    // through the freed ring ([32][132] f32: row stride 132 keeps the
    // fragment writes conflict-free), then per token the norm + rope v3 pass
    // (elementwise.cuh) runs on its own row: the same lane-strided registers,
    // shuffle tree, closed-form angle and stores - so the same bits as that
    // pass reading these values back from the f32 plane.
    auto epilogue_nr = [&](uint32_t tm) {
        static_assert(!NR || (MODE == PD_NVPF_DENSE && RD && BN % 32u == 0u && BM == 128u),
                      "the norm + rope epilogue: the dense register-decode tile, a head a tile");
        constexpr uint32_t SR = 132u, HD = 128u, HALF = 64u, HB = 2u;
        float* sm = reinterpret_cast<float*>(pd_nvpf_dyn);
        const float f = p.fa1 * LIFT;
        const uint32_t ld = p.ldy;
#pragma unroll
        for (uint32_t pass = 0; pass < BN / 32u; ++pass) {
            __syncthreads();
#pragma unroll
            for (uint32_t cg = 0; cg < CG; ++cg)
#pragma unroll
                for (uint32_t h = 0; h < 2u; ++h) {
                    const uint32_t col = wc + cg * 8u + 2u * t + h;
                    if ((col >> 5) != pass) continue;
                    const uint32_t r0 = wr + g;
                    sm[(col & 31u) * SR + r0] = acc[0][cg][h] * f;
                    sm[(col & 31u) * SR + r0 + 8u] = acc[0][cg][2u + h] * f;
                }
            __syncthreads();
            for (uint32_t cc = warp; cc < 32u; cc += 8u) {
                const uint32_t orw = orow[pass * 32u + cc];
                if (orw == PD_MOE_PAD) continue;   // warp-uniform
                float rv[4];
#pragma unroll
                for (uint32_t j = 0; j < 4u; ++j) rv[j] = sm[cc * SR + j * 32u + lane];
                float sum = 0.0f;
#pragma unroll
                for (uint32_t j = 0; j < 4u; ++j) {
                    float a2 = rv[j] * rv[j];
                    for (uint32_t sh = 16; sh > 0; sh >>= 1)
                        a2 += __shfl_down_sync(0xffffffffu, a2, sh);
                    a2 = __shfl_sync(0xffffffffu, a2, 0);
                    sum += a2;
                }
                const float inv = 1.0f / sqrtf(sum / (float)HD + p.eps);
#pragma unroll
                for (uint32_t j = 0; j < 4u; ++j) rv[j] = rv[j] * inv * p.nw[j * 32u + lane];
                const float l2ts = log2f(p.ts);
                const float pos_f = (float)p.pos[orw];
                for (uint32_t t2 = 0; t2 < HB; ++t2) {
                    const uint32_t kk = t2 * 32u + lane;
                    const float theta = pos_f * exp2f((float)kk * l2ts);
                    float y = ((float)kk - p.clo) / fmaxf(0.001f, p.chi - p.clo);
                    float ramp = (1.0f - fminf(1.0f, fmaxf(0.0f, y))) * p.ext;
                    float angle = (p.fs * theta) * (1.0f - ramp) + theta * ramp;
                    float sn = sinf(angle) * p.msc;
                    float cs = cosf(angle) * p.msc;
                    float a = rv[t2];
                    float bb = rv[t2 + HB];
                    if (p.nr == 1u && orw >= p.dec) {
                        __half* qh = p.q16 + (size_t)orw * ld + tm * HD;
                        const float o0 = a * cs - bb * sn, o1 = a * sn + bb * cs;
                        qh[kk] = __float2half_rn(o0 * p.qsc);
                        qh[kk + HALF] = __float2half_rn(o1 * p.qsc);
                    } else {
                        float* dst = (p.nr == 1u ? p.y2 : p.y) + (size_t)orw * ld + tm * HD;
                        dst[kk] = a * cs - bb * sn;
                        dst[kk + HALF] = a * sn + bb * cs;
                    }
                }
            }
        }
    };

    // RD: the A fragments of k16 step sk from the staged code words (rows g,
    // g + 8 of the warp's 16) - lifted decode, the scale x 2^119 per 16
    const uint32_t psel = ((4u + t) << 4) | (t << 12);
    // buf: the token stage; cs / sub: the code slot and the k-tile within it
    auto compute_rd = [&](uint32_t buf, uint32_t cs, uint32_t sub) {
#if PD_NVPF_LIFT
        const uint32_t rl = wr + g;
        const uint4* cl = reinterpret_cast<const uint4*>(sh_c + (cs * BM + rl) * CB * CW + sub * CB);
        const uint4* ch = reinterpret_cast<const uint4*>(sh_c + (cs * BM + rl + 8u) * CB * CW + sub * CB);
        const uint4 l0 = cl[0], l1 = cl[1], h0 = ch[0], h1 = ch[1];
        const uint32_t wl[8] = {l0.x, l0.y, l0.z, l0.w, l1.x, l1.y, l1.z, l1.w};
        const uint32_t wh[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
        const uint32_t sl = *reinterpret_cast<const uint32_t*>(sh_s + (cs * BM + rl) * 4u * CW + sub * 4u);
        const uint32_t sh = *reinterpret_cast<const uint32_t*>(sh_s + (cs * BM + rl + 8u) * 4u * CW + sub * 4u);
        auto pair = [&](uint32_t w, uint32_t s2) {
            const uint32_t q = __byte_perm(w, w << 4, psel);
            const uint32_t v = (q & 0x80008000u) | ((q >> 6) & 0x01C001C0u);
            uint32_t r;
            asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(v), "r"(s2));
            return r;
        };
#pragma unroll
        for (uint32_t sk = 0; sk < KT / 16u; ++sk) {
            const uint32_t ko = sk * 16u;
            const uint32_t scl = pd_nvpf_lscale2((sl >> (8u * sk)) & 0xFFu);
            const uint32_t sch = pd_nvpf_lscale2((sh >> (8u * sk)) & 0xFFu);
            uint32_t a[4];
            a[0] = pair(wl[2u * sk], scl);
            a[1] = pair(wh[2u * sk], sch);
            a[2] = pair(wl[2u * sk + 1u], scl);
            a[3] = pair(wh[2u * sk + 1u], sch);
#pragma unroll
            for (uint32_t cp = 0; cp < CG / 2u; ++cp) {
                uint32_t b0[2], b1[2];
                pd_nvpf_ldm4(&sh_b[(buf * BN + wc + cp * 16u + ((lane >> 4) & 1u) * 8u + l7) * KPAD +
                                   ko + ((lane >> 3) & 1u) * 8u],
                             b0[0], b0[1], b1[0], b1[1]);
                pd_nvpf_mma(acc[0][2u * cp], a, b0);
                pd_nvpf_mma(acc[0][2u * cp + 1u], a, b1);
            }
        }
#else
        (void)buf; (void)cs; (void)sub;
#endif
    };
    (void)psel; (void)compute_rd;

    const uint32_t nk = p.K / KT;
    if constexpr (OVL) {
        // codes two stages ahead, tokens one: iteration i decodes stage i + 1
        // into the other bf16 tile while it multiplies stage i
        stage_codes(tile_m, 0u, 0u);
        stage_tokens(0u, 0u);
        pd_nvpf_commit();
        if (nk > 1u) stage_codes(tile_m, KT, 1u);
        pd_nvpf_commit();
        pd_nvpf_wait<1>();
        __syncthreads();
        decode(0u, 0u, 0u);
        for (uint32_t i = 0; i < nk; ++i) {
            pd_nvpf_wait<0>();
            __syncthreads();
            if (i + 2u < nk) stage_codes(tile_m, (i + 2u) * KT, (i + 2u) % 3u);
            if (i + 1u < nk) stage_tokens((i + 1u) * KT, (i + 1u) & 1u);
            pd_nvpf_commit();
            if (i + 1u < nk) decode((i + 1u) % 3u, (i + 1u) & 1u, (i + 1u) * KT);
            compute(i & 1u, i & 1u);
        }
        epilogue(tile_m);
    } else if constexpr (CW > 1u) {
        // tokens one k-tile ahead, codes one CW-tile group ahead (the group
        // after the one being multiplied); one barrier a step both lands the
        // step's tiles and frees the slots the new copies stream into
        const uint32_t ns = MPER * nk, ng = ns / CW;
        auto stage_c = [&](uint32_t gi) {
            const uint32_t s0 = gi * CW;
            stage_codes(tile_m + s0 / nk, (s0 % nk) * KT, gi & 1u);
        };
        stage_c(0u);
        stage_tokens(0u, 0u);
        pd_nvpf_commit();
        for (uint32_t i = 0; i < ns; ++i) {
            pd_nvpf_wait<0>();
            __syncthreads();
            if (i + 1u < ns) stage_tokens(((i + 1u) % nk) * KT, (i + 1u) & 1u);
            if (i % CW == 0u && i / CW + 1u < ng) stage_c(i / CW + 1u);
            pd_nvpf_commit();
            compute_rd(i & 1u, (i / CW) & 1u, i % CW);
            if (i % nk == nk - 1u) {
                epilogue(tile_m + i / nk);
                if constexpr (MPER > 1u) {
#pragma unroll
                    for (uint32_t rg = 0; rg < RG; ++rg)
#pragma unroll
                        for (uint32_t cg = 0; cg < CG; ++cg)
#pragma unroll
                            for (uint32_t q = 0; q < 4u; ++q) acc[rg][cg][q] = 0.0f;
                }
            }
        }
    } else {
        // stage k + RING - 1 streams in while stage k decodes and multiplies;
        // step s is k-tile s % nk of weight tile tile_m + s / nk
        const uint32_t ns = MPER * nk;
        auto stage = [&](uint32_t s, uint32_t buf) {
            stage_codes(tile_m + s / nk, (s % nk) * KT, buf);
            stage_tokens((s % nk) * KT, buf);
        };
#pragma unroll
        for (uint32_t s = 0; s + 1u < RING; ++s) {
            if (s < ns) stage(s, s);
            pd_nvpf_commit();
        }
        for (uint32_t i = 0; i < ns; ++i) {
            const uint32_t pre = i + RING - 1u;
            if (pre < ns) stage(pre, pre % RING);
            pd_nvpf_commit();
            pd_nvpf_wait<(int)RING - 1>();
            __syncthreads();
            if constexpr (RD) {
                compute_rd(i % RING, i % RING, 0u);
            } else {
                decode(i % RING, 0u, (i % nk) * KT);
                __syncthreads();
                compute(0u, i % RING);
            }
            __syncthreads();
            if (i % nk == nk - 1u) {
                if constexpr (NR) {
                    if (p.nr != 0u) epilogue_nr(tile_m + i / nk);
                    else epilogue(tile_m + i / nk);
                } else {
                    epilogue(tile_m + i / nk);
                }
                if constexpr (MPER > 1u) {
#pragma unroll
                    for (uint32_t rg = 0; rg < RG; ++rg)
#pragma unroll
                        for (uint32_t cg = 0; cg < CG; ++cg)
#pragma unroll
                            for (uint32_t q = 0; q < 4u; ++q) acc[rg][cg][q] = 0.0f;
                }
            }
        }
    }

#else
    (void)p; (void)bx;
#endif
}

template <uint32_t MODE, uint32_t BN, bool OVL, uint32_t RING, uint32_t MPER = 1u,
          uint32_t KT = 64u, bool RD = false, uint32_t CW = 1u>
__global__ void __launch_bounds__(256, RD ? 3 : 1) pd_nvpf_kernel(const PdNvPfArgs p) {
    pd_nvpf_body<MODE, BN, OVL, RING, MPER, KT, RD, CW>(p, blockIdx.x);
}

// Up to three dense planes over the same bf16 rows in one grid (slot 931):
// plane i owns blocks [start[i], start[i + 1]) and runs the body on its own
// block index - the per-plane launch's tiles, walk and bits. One grid lets a
// thin plane's blocks (k / v: 52 at 0.8K tokens, under half a wave on GB10)
// fill the wide plane's last wave instead of paying a wave each.
struct PdNvPf3 {
    PdNvPfArgs a[3];
    uint32_t start[4];
};

template <uint32_t MODE, uint32_t BN, bool OVL, uint32_t RING, uint32_t MPER = 1u,
          uint32_t KT = 64u, bool RD = false, uint32_t CW = 1u, bool NR = false>
__global__ void __launch_bounds__(256, RD ? 3 : 1) pd_nvpf3_kernel(const PdNvPf3 P) {
    const uint32_t b = blockIdx.x;
    const uint32_t pl = b < P.start[1] ? 0u : (b < P.start[2] ? 1u : 2u);
    pd_nvpf_body<MODE, BN, OVL, RING, MPER, KT, RD, CW, NR>(P.a[pl], b - P.start[pl]);
}

// Two tiles of different modes in one grid (slots 932 / 933): blocks [0, na)
// run problem A (the routed experts), the rest problem B (the shared
// expert) on its own block index - each its own launch's bits. The shared
// expert's compute-bound blocks land in the routed pass's last wave instead
// of paying their own partial waves (208 blocks at 0.8K tokens on GB10's
// 144 slots). Register-decode arms only.
template <uint32_t MA, uint32_t PA, uint32_t CA, uint32_t MB, uint32_t PB, uint32_t CB_>
__global__ void __launch_bounds__(256, 3)
    pd_nvpf_pair_kernel(const PdNvPfArgs A, uint32_t na, const PdNvPfArgs B) {
    if (blockIdx.x < na)
        pd_nvpf_body<MA, 64u, false, 2u, PA, 64u, true, CA>(A, blockIdx.x);
    else
        pd_nvpf_body<MB, 64u, false, 2u, PB, 64u, true, CB_>(B, blockIdx.x - na);
}

template <uint32_t MA, uint32_t PA, uint32_t CA, uint32_t MB, uint32_t PB, uint32_t CB_>
static int pd_nvpf_launch_pair(const PdNvPfArgs& A, uint32_t na, const PdNvPfArgs& B,
                               uint32_t nb, cudaStream_t st) {
    constexpr uint32_t sa = pd_nvpf_smem(64u, false, 2u, 64u, true, CA);
    constexpr uint32_t sb = pd_nvpf_smem(64u, false, 2u, 64u, true, CB_);
    constexpr uint32_t smem = sa > sb ? sa : sb;
    static bool set = false;
    if (!set) {
        cudaFuncSetAttribute(pd_nvpf_pair_kernel<MA, PA, CA, MB, PB, CB_>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        set = true;
    }
    if (na + nb == 0u) return 0;
    pd_nvpf_pair_kernel<MA, PA, CA, MB, PB, CB_><<<na + nb, 256u, smem, st>>>(A, na, B);
    return pd_launch_status();
}

template <uint32_t MODE, uint32_t BN, uint32_t RING = 2u, bool OVL = PD_NVPF_OVL_DEFAULT,
          uint32_t MPER = 1u, uint32_t KT = 64u, bool RD = false, uint32_t CW = 1u>
static int pd_nvpf_launch(const PdNvPfArgs& a, uint32_t grid, cudaStream_t st) {
    constexpr uint32_t smem = pd_nvpf_smem(BN, OVL, RING, KT, RD, CW);
    static_assert(smem <= 101376u, "past the per-block opt-in shared memory");
    static bool set = false;
    if (!set) {
        cudaFuncSetAttribute(pd_nvpf_kernel<MODE, BN, OVL, RING, MPER, KT, RD, CW>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        set = true;
    }
    if (grid == 0u) return 0;
    pd_nvpf_kernel<MODE, BN, OVL, RING, MPER, KT, RD, CW><<<grid, 256u, smem, st>>>(a);
    return pd_launch_status();
}

template <uint32_t MODE, uint32_t BN, uint32_t RING = 2u, bool OVL = PD_NVPF_OVL_DEFAULT,
          uint32_t MPER = 1u, uint32_t KT = 64u, bool RD = false, uint32_t CW = 1u,
          bool NR = false>
static int pd_nvpf_launch3(const PdNvPf3& P, cudaStream_t st) {
    constexpr uint32_t smem0 = pd_nvpf_smem(BN, OVL, RING, KT, RD, CW);
    // NR stages 32 tokens x 132 f32 in the ring after the K loop
    constexpr uint32_t smem = NR && smem0 < 32u * 132u * 4u ? 32u * 132u * 4u : smem0;
    static bool set = false;
    if (!set) {
        cudaFuncSetAttribute(pd_nvpf3_kernel<MODE, BN, OVL, RING, MPER, KT, RD, CW, NR>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        set = true;
    }
    if (P.start[3] == 0u) return 0;
    pd_nvpf3_kernel<MODE, BN, OVL, RING, MPER, KT, RD, CW, NR>
        <<<P.start[3], 256u, smem, st>>>(P);
    return pd_launch_status();
}

// The register-decode arm (RD) on sm_90+ (its lifted decode's bf16x2
// multiply), for the dense planes and the routed gate|up - GB10 at Flint's
// 8K shapes: q 2.28 -> 1.94 ms, o 2.16 -> 1.82, shared gate|up 1.10 -> 0.95,
// routed gate|up 3.85 -> 3.34 (61 -> 72-75 / 53 -> 61 TF/s). The routed down
// (K 512) measured level and keeps the shared-tile arm. Same bits either
// way. Kill: PADDOCK_NO_NVPF_RD.
static inline bool pd_nvpf_rd_ok() {
    static const bool ok = [] {
        int dev = 0, major = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        return major >= 9 && pd_env("PADDOCK_NO_NVPF_RD") == nullptr;
    }();
    return ok;
}

static inline uint32_t pd_nvpf_dense_grid(uint32_t mt, uint32_t ntok, uint32_t bn) {
    return mt * ((ntok + bn - 1u) / bn);
}

// ---- launchers (ABI 920-923). Planes are row-major NVFP4 (codes [rows,
// in/2], e4m3 scales [rows, in/16]); in % 64 == 0; x rows bf16, 16-byte
// aligned at ldx / xoff. Dense planes M % 128 == 0 is not required (ragged
// rows zero-fill); GU channel counts are multiples of 64.

// 920: y[c * ldy + yoff + r] = (W x_c)_r * s2, c < ntok, r < out_dim.
PD_EXPORT
int pd_nvf4_pf_gemm(const void* data, const void* scale, float s2, const void* x,
                    uint32_t ldx, uint32_t xoff, void* y, uint32_t ldy, uint32_t yoff,
                    uint32_t in_dim, uint32_t out_dim, uint32_t ntok, void* stream) {
#ifndef PD_BS_HOST
    (void)data; (void)scale; (void)s2; (void)x; (void)ldx; (void)xoff; (void)y; (void)ldy;
    (void)yoff; (void)in_dim; (void)out_dim; (void)ntok; (void)stream;
    return cudaErrorNotSupported;
#else
    if (out_dim == 0u || ntok == 0u) return 0;
    if ((in_dim & 63u) != 0u || ((ldx | xoff) & 7u) != 0u) return cudaErrorInvalidValue;
    PdNvPfArgs a = {};
    a.da = (const uint8_t*)data; a.sa = (const uint8_t*)scale; a.fa1 = s2;
    a.x = (const __nv_bfloat16*)x; a.ldx = ldx; a.xoff = xoff;
    a.y = (float*)y; a.ldy = ldy; a.yoff = yoff;
    a.K = in_dim; a.M = out_dim; a.ntok = ntok;
    const uint32_t mt = (out_dim + PD_NVPF_BM - 1u) / PD_NVPF_BM;
    const uint32_t grid = pd_nvpf_dense_grid(mt, ntok, PD_NVPF_BN_DENSE);
    if (pd_nvpf_rd_ok() && PD_NVPF_KT_DENSE == 64u)
        return pd_nvpf_launch<PD_NVPF_DENSE, PD_NVPF_BN_DENSE, PD_NVPF_RING_DENSE, false, 1u, 64u,
                              true>(a, grid, (cudaStream_t)stream);
    return pd_nvpf_launch<PD_NVPF_DENSE, PD_NVPF_BN_DENSE, PD_NVPF_RING_DENSE,
                          PD_NVPF_OVL_DEFAULT, 1u, PD_NVPF_KT_DENSE>(a, grid, (cudaStream_t)stream);
#endif
}

// 931: up to three 920 calls over the same rows and in_dim in one launch
// (q|k|v): plane i is (data, scale, s2, y, ldy, yoff, out_dim), out_dim 0
// skips it. The register-decode arm only (sm_90+) - elsewhere
// cudaErrorNotSupported and the caller keeps three 920 launches. Each plane
// lands its 920 bits.
PD_EXPORT
int pd_nvf4_pf_gemm3(const void* d0, const void* s0, float f0, void* y0, uint32_t ldy0,
                     uint32_t n0, const void* d1, const void* s1, float f1, void* y1,
                     uint32_t ldy1, uint32_t n1, const void* d2, const void* s2, float f2,
                     void* y2, uint32_t ldy2, uint32_t n2, const void* x, uint32_t ldx,
                     uint32_t xoff, uint32_t in_dim, uint32_t ntok, void* stream) {
#ifndef PD_BS_HOST
    (void)d0; (void)s0; (void)f0; (void)y0; (void)ldy0; (void)n0; (void)d1; (void)s1; (void)f1;
    (void)y1; (void)ldy1; (void)n1; (void)d2; (void)s2; (void)f2; (void)y2; (void)ldy2;
    (void)n2; (void)x; (void)ldx; (void)xoff; (void)in_dim; (void)ntok; (void)stream;
    return cudaErrorNotSupported;
#else
    if (!pd_nvpf_rd_ok() || PD_NVPF_KT_DENSE != 64u) return cudaErrorNotSupported;
    if (ntok == 0u || n0 + n1 + n2 == 0u) return 0;
    if ((in_dim & 63u) != 0u || ((ldx | xoff) & 7u) != 0u) return cudaErrorInvalidValue;
    const void* ds[3] = {d0, d1, d2};
    const void* ss[3] = {s0, s1, s2};
    const float fs[3] = {f0, f1, f2};
    void* ys[3] = {y0, y1, y2};
    const uint32_t ls[3] = {ldy0, ldy1, ldy2}, ns[3] = {n0, n1, n2};
    PdNvPf3 P = {};
    for (int i = 0; i < 3; ++i) {
        PdNvPfArgs& a = P.a[i];
        a.da = (const uint8_t*)ds[i]; a.sa = (const uint8_t*)ss[i]; a.fa1 = fs[i];
        a.x = (const __nv_bfloat16*)x; a.ldx = ldx; a.xoff = xoff;
        a.y = (float*)ys[i]; a.ldy = ls[i]; a.yoff = 0u;
        a.K = in_dim; a.M = ns[i]; a.ntok = ntok;
        const uint32_t mt = (ns[i] + PD_NVPF_BM - 1u) / PD_NVPF_BM;
        P.start[i + 1] = P.start[i] + (ns[i] ? pd_nvpf_dense_grid(mt, ntok, PD_NVPF_BN_DENSE) : 0u);
    }
    return pd_nvpf_launch3<PD_NVPF_DENSE, PD_NVPF_BN_DENSE, PD_NVPF_RING_DENSE, false, 1u, 64u,
                           true>(P, (cudaStream_t)stream);
#endif
}

// 934: 931's q|k|v grid with the q / k norm + rope (the f32 v3 arm of
// pd_qkv_norm_rope_batch: NEOX, closed-form angle, head_dim 128, an unnormed
// V) in the epilogues - q lands as f16 x qscale in q16 (slot 929's halves)
// from row `dec` on and as f32 in qn below it, k in kn, v as projected.
// Each row's bits are the projection + slot 929 / v3 pass's. rope = theta
// scale, freq scale, corr low / high, ext factor, mscale. The register-decode
// arm only, the PADDOCK_NO_NRV2 / NRV3 / RTBL pins honored (cudaErrorNot-
// Supported - the caller keeps the separate passes).
PD_EXPORT
int pd_nvf4_pf_qkv_nr(const void* dq, const void* sq, float fq, const void* dk, const void* sk,
                      float fk, const void* dv, const void* sv, float fv, void* q16, void* qn,
                      void* kn, void* v, uint32_t n_head, uint32_t n_kv, const void* qw,
                      const void* kw, const void* positions, uint32_t dec, float eps,
                      const float* rope, float qscale, const void* x, uint32_t in_dim,
                      uint32_t ntok, void* stream) {
#ifndef PD_BS_HOST
    (void)dq; (void)sq; (void)fq; (void)dk; (void)sk; (void)fk; (void)dv; (void)sv; (void)fv;
    (void)q16; (void)qn; (void)kn; (void)v; (void)n_head; (void)n_kv; (void)qw; (void)kw;
    (void)positions; (void)dec; (void)eps; (void)rope; (void)qscale; (void)x; (void)in_dim;
    (void)ntok; (void)stream;
    return cudaErrorNotSupported;
#else
    static const bool off = pd_env("PADDOCK_NO_NRV2") || pd_env("PADDOCK_NO_NRV3")
                            || pd_env("PADDOCK_NO_RTBL") || pd_env("PADDOCK_NO_PF_NR");
    if (off || !pd_nvpf_rd_ok() || PD_NVPF_KT_DENSE != 64u || !rope)
        return cudaErrorNotSupported;
    if (ntok == 0u) return 0;
    if ((in_dim & 63u) != 0u || n_head == 0u || n_kv == 0u) return cudaErrorInvalidValue;
    const void* ds[3] = {dq, dk, dv};
    const void* ss[3] = {sq, sk, sv};
    const float fs[3] = {fq, fk, fv};
    const uint32_t ns[3] = {n_head * 128u, n_kv * 128u, n_kv * 128u};
    PdNvPf3 P = {};
    for (int i = 0; i < 3; ++i) {
        PdNvPfArgs& a = P.a[i];
        a.da = (const uint8_t*)ds[i]; a.sa = (const uint8_t*)ss[i]; a.fa1 = fs[i];
        a.x = (const __nv_bfloat16*)x; a.ldx = in_dim; a.xoff = 0u;
        a.ldy = ns[i]; a.yoff = 0u;
        a.K = in_dim; a.M = ns[i]; a.ntok = ntok;
        a.nr = i == 0 ? 1u : (i == 1 ? 2u : 0u);
        a.nw = (const float*)(i == 0 ? qw : kw);
        a.pos = (const unsigned int*)positions;
        a.dec = dec; a.eps = eps; a.qsc = qscale;
        a.ts = rope[0]; a.fs = rope[1]; a.clo = rope[2]; a.chi = rope[3]; a.ext = rope[4];
        a.msc = rope[5];
        const uint32_t mt = ns[i] / PD_NVPF_BM;
        P.start[i + 1] = P.start[i] + pd_nvpf_dense_grid(mt, ntok, PD_NVPF_BN_DENSE);
    }
    P.a[0].q16 = (__half*)q16; P.a[0].y2 = (float*)qn;
    P.a[1].y = (float*)kn;
    P.a[2].y = (float*)v;
    return pd_nvpf_launch3<PD_NVPF_DENSE, PD_NVPF_BN_DENSE, PD_NVPF_RING_DENSE, false, 1u, 64u,
                           true, 1u, true>(P, (cudaStream_t)stream);
#endif
}

// 921: act[c * lda + aoff + ch] = bf16(silu(g * gs2) * (u * us2)), ch < ff.
PD_EXPORT
int pd_nvf4_pf_gu(const void* gdata, const void* gscale, float gs2, const void* udata,
                  const void* uscale, float us2, const void* x, uint32_t ldx, void* act,
                  uint32_t lda, uint32_t aoff, uint32_t in_dim, uint32_t ff, uint32_t ntok,
                  void* stream) {
#ifndef PD_BS_HOST
    (void)gdata; (void)gscale; (void)gs2; (void)udata; (void)uscale; (void)us2; (void)x;
    (void)ldx; (void)act; (void)lda; (void)aoff; (void)in_dim; (void)ff; (void)ntok;
    (void)stream;
    return cudaErrorNotSupported;
#else
    if (ff == 0u || ntok == 0u) return 0;
    if ((in_dim & 63u) != 0u || (ff & 63u) != 0u || (ldx & 7u) != 0u)
        return cudaErrorInvalidValue;
    PdNvPfArgs a = {};
    a.da = (const uint8_t*)gdata; a.sa = (const uint8_t*)gscale; a.fa1 = gs2;
    a.db = (const uint8_t*)udata; a.sb = (const uint8_t*)uscale; a.fb1 = us2;
    a.x = (const __nv_bfloat16*)x; a.ldx = ldx;
    a.act = (__nv_bfloat16*)act; a.ldy = lda; a.yoff = aoff;
    a.K = in_dim; a.M = ff; a.ntok = ntok;
    const uint32_t grid = pd_nvpf_dense_grid(ff / 64u, ntok, PD_NVPF_BN_DENSE);
    if (pd_nvpf_rd_ok() && PD_NVPF_KT_DENSE == 64u)
        return pd_nvpf_launch<PD_NVPF_GU, PD_NVPF_BN_DENSE, PD_NVPF_RING_DENSE, false, 1u, 64u,
                              true>(a, grid, (cudaStream_t)stream);
    return pd_nvpf_launch<PD_NVPF_GU, PD_NVPF_BN_DENSE, PD_NVPF_RING_DENSE,
                          PD_NVPF_OVL_DEFAULT, 1u, PD_NVPF_KT_DENSE>(a, grid, (cudaStream_t)stream);
#endif
}

// 922 / 926: the routed gate|up. n_expert (926; 0 from 922) elects the
// weight copy width: below ~2.5 blocks an expert each expert's weights stream
// from DRAM for one or two blocks, and GB10's DRAM moves the 32-byte row
// slivers of one k-tile at about half the rate of two k-tiles' 64 bytes (CW 2:
// 2048 tokens 2.75 -> 1.81 ms, 800 tokens 2.55 -> 1.61); with more blocks an
// expert the weights come back from L2 and the extra slot costs a CTA of
// occupancy (8192 tokens 3.38 -> 3.70), so CW 1 there.
static int pd_nvf4_pf_moe_gu_impl(const void* gdata, const void* gscale, const void* gs2,
                                  const void* udata, const void* uscale, const void* us2,
                                  const void* sorted_row, const void* sorted_slot,
                                  const void* block_expert, const void* x, void* act,
                                  uint32_t in_dim, uint32_t ff, uint32_t aw, uint32_t n_blocks,
                                  uint32_t n_expert, void* stream) {
#ifndef PD_BS_HOST
    (void)gdata; (void)gscale; (void)gs2; (void)udata; (void)uscale; (void)us2;
    (void)sorted_row; (void)sorted_slot; (void)block_expert; (void)x; (void)act;
    (void)in_dim; (void)ff; (void)aw; (void)n_blocks; (void)n_expert; (void)stream;
    return cudaErrorNotSupported;
#else
    if (ff == 0u || n_blocks == 0u) return 0;
    if ((in_dim & 63u) != 0u || (ff & 63u) != 0u) return cudaErrorInvalidValue;
    PdNvPfArgs a = {};
    a.da = (const uint8_t*)gdata; a.sa = (const uint8_t*)gscale; a.fa = (const float*)gs2;
    a.db = (const uint8_t*)udata; a.sb = (const uint8_t*)uscale; a.fb = (const float*)us2;
    a.x = (const __nv_bfloat16*)x; a.ldx = in_dim;
    a.act = (__nv_bfloat16*)act; a.ldy = aw;
    a.K = in_dim; a.M = ff;
    a.srow = (const uint32_t*)sorted_row; a.sslot = (const uint32_t*)sorted_slot;
    a.bexp = (const uint32_t*)block_expert; a.n_blocks = n_blocks;
    const uint32_t grid = n_blocks * (ff / 64u);
    if (pd_nvpf_rd_ok()) {
        const bool thin = n_expert != 0u && 2u * n_blocks < 5u * n_expert && (in_dim / 64u) % 2u == 0u;
        if (thin)
            return pd_nvpf_launch<PD_NVPF_MOE_GU, 64u, 2u, false, 1u, 64u, true, 2u>(
                a, grid, (cudaStream_t)stream);
        return pd_nvpf_launch<PD_NVPF_MOE_GU, 64u, 2u, false, 1u, 64u, true>(
            a, grid, (cudaStream_t)stream);
    }
    return pd_nvpf_launch<PD_NVPF_MOE_GU, 64u>(a, grid, (cudaStream_t)stream);
#endif
}

// 922: act[t * aw + slot * ff + ch] over moe_align_bm(64)'s blocks; planes
// [n_expert * ff, in], factors [n_expert].
PD_EXPORT
int pd_nvf4_pf_moe_gu(const void* gdata, const void* gscale, const void* gs2,
                      const void* udata, const void* uscale, const void* us2,
                      const void* sorted_row, const void* sorted_slot, const void* block_expert,
                      const void* x, void* act, uint32_t in_dim, uint32_t ff, uint32_t aw,
                      uint32_t n_blocks, void* stream) {
    return pd_nvf4_pf_moe_gu_impl(gdata, gscale, gs2, udata, uscale, us2, sorted_row,
                                  sorted_slot, block_expert, x, act, in_dim, ff, aw, n_blocks,
                                  0u, stream);
}

// 926: 922 with the plane's expert count (the copy-width election above).
PD_EXPORT
int pd_nvf4_pf_moe_gu_ne(const void* gdata, const void* gscale, const void* gs2,
                         const void* udata, const void* uscale, const void* us2,
                         const void* sorted_row, const void* sorted_slot,
                         const void* block_expert, const void* x, void* act, uint32_t in_dim,
                         uint32_t ff, uint32_t aw, uint32_t n_blocks, uint32_t n_expert,
                         void* stream) {
    return pd_nvf4_pf_moe_gu_impl(gdata, gscale, gs2, udata, uscale, us2, sorted_row,
                                  sorted_slot, block_expert, x, act, in_dim, ff, aw, n_blocks,
                                  n_expert, stream);
}

// 923: the routed down over the same blocks: bf16 part[(t * k + slot) * embd
// + r] = (W_e act[t * aw + slot * ff])_r * topk_w[t][slot] * s2[e] - the
// layout moe_slot_combine_bf16 folds.
PD_EXPORT
int pd_nvf4_pf_moe_dn(const void* data, const void* scale, const void* s2,
                      const void* sorted_row, const void* sorted_slot, const void* block_expert,
                      const void* topk_w, const void* act, uint32_t aw, void* part, uint32_t ff,
                      uint32_t embd, uint32_t k, uint32_t n_blocks, void* stream) {
#ifndef PD_BS_HOST
    (void)data; (void)scale; (void)s2; (void)sorted_row; (void)sorted_slot;
    (void)block_expert; (void)topk_w; (void)act; (void)aw; (void)part; (void)ff; (void)embd;
    (void)k; (void)n_blocks; (void)stream;
    return cudaErrorNotSupported;
#else
    if (embd == 0u || n_blocks == 0u || k == 0u) return 0;
    if ((ff & 63u) != 0u || (embd & 127u) != 0u || (aw & 7u) != 0u)
        return cudaErrorInvalidValue;
    PdNvPfArgs a = {};
    a.da = (const uint8_t*)data; a.sa = (const uint8_t*)scale; a.fa = (const float*)s2;
    a.x = (const __nv_bfloat16*)act; a.ldx = aw;
    a.act = (__nv_bfloat16*)part; a.ldy = embd;
    a.K = ff; a.M = embd; a.k = k;
    a.srow = (const uint32_t*)sorted_row; a.sslot = (const uint32_t*)sorted_slot;
    a.bexp = (const uint32_t*)block_expert; a.topk_w = (const float*)topk_w;
    a.n_blocks = n_blocks;
    // four weight tiles a CTA where they divide (Flint: 16 of 128 rows); on
    // sm_90+ register decode with two k-tiles a weight copy (K 512: the
    // 32-byte row slivers were the whole cost at every chunk size - 8192
    // tokens 2.50 -> 2.17 ms, 2048 1.46 -> 1.04, 800 1.26 -> 0.77)
    const uint32_t mt = embd / PD_NVPF_BM;
    if (mt % PD_NVPF_DN_MPER == 0u) {
        const uint32_t grid = n_blocks * (mt / PD_NVPF_DN_MPER);
        if (pd_nvpf_rd_ok() && (ff / 64u) % 2u == 0u)
            return pd_nvpf_launch<PD_NVPF_MOE_DN, 64u, 2u, false, PD_NVPF_DN_MPER, 64u, true, 2u>(
                a, grid, (cudaStream_t)stream);
        return pd_nvpf_launch<PD_NVPF_MOE_DN, 64u, 2u, false, PD_NVPF_DN_MPER>(
            a, grid, (cudaStream_t)stream);
    }
    return pd_nvpf_launch<PD_NVPF_MOE_DN, 64u>(a, n_blocks * mt, (cudaStream_t)stream);
#endif
}

// 932: 926's routed gate|up and 921's shared gate|up (ff `sff`, landing at
// column `aoff` of the same act rows, `ntok` rows) in one grid - each its
// own launch's bits. Register-decode arm only: cudaErrorNotSupported
// elsewhere, and the caller keeps the two launches.
PD_EXPORT
int pd_nvf4_pf_moe_gu_sh(const void* gdata, const void* gscale, const void* gs2,
                         const void* udata, const void* uscale, const void* us2,
                         const void* sorted_row, const void* sorted_slot,
                         const void* block_expert, const void* x, void* act, uint32_t in_dim,
                         uint32_t ff, uint32_t aw, uint32_t n_blocks, uint32_t n_expert,
                         const void* sgdata, const void* sgscale, float sgs2, const void* sudata,
                         const void* suscale, float sus2, uint32_t sff, uint32_t aoff,
                         uint32_t ntok, void* stream) {
#ifndef PD_BS_HOST
    (void)gdata; (void)gscale; (void)gs2; (void)udata; (void)uscale; (void)us2;
    (void)sorted_row; (void)sorted_slot; (void)block_expert; (void)x; (void)act; (void)in_dim;
    (void)ff; (void)aw; (void)n_blocks; (void)n_expert; (void)sgdata; (void)sgscale;
    (void)sgs2; (void)sudata; (void)suscale; (void)sus2; (void)sff; (void)aoff; (void)ntok;
    (void)stream;
    return cudaErrorNotSupported;
#else
    if (!pd_nvpf_rd_ok() || PD_NVPF_KT_DENSE != 64u) return cudaErrorNotSupported;
    if ((in_dim & 63u) != 0u || (ff & 63u) != 0u || (sff & 63u) != 0u || (in_dim & 7u) != 0u)
        return cudaErrorInvalidValue;
    PdNvPfArgs a = {};
    a.da = (const uint8_t*)gdata; a.sa = (const uint8_t*)gscale; a.fa = (const float*)gs2;
    a.db = (const uint8_t*)udata; a.sb = (const uint8_t*)uscale; a.fb = (const float*)us2;
    a.x = (const __nv_bfloat16*)x; a.ldx = in_dim;
    a.act = (__nv_bfloat16*)act; a.ldy = aw;
    a.K = in_dim; a.M = ff;
    a.srow = (const uint32_t*)sorted_row; a.sslot = (const uint32_t*)sorted_slot;
    a.bexp = (const uint32_t*)block_expert; a.n_blocks = n_blocks;
    PdNvPfArgs b = {};
    b.da = (const uint8_t*)sgdata; b.sa = (const uint8_t*)sgscale; b.fa1 = sgs2;
    b.db = (const uint8_t*)sudata; b.sb = (const uint8_t*)suscale; b.fb1 = sus2;
    b.x = (const __nv_bfloat16*)x; b.ldx = in_dim;
    b.act = (__nv_bfloat16*)act; b.ldy = aw; b.yoff = aoff;
    b.K = in_dim; b.M = sff; b.ntok = ntok;
    const uint32_t na = ff ? n_blocks * (ff / 64u) : 0u;
    const uint32_t nb = sff && ntok ? pd_nvpf_dense_grid(sff / 64u, ntok, PD_NVPF_BN_DENSE) : 0u;
    // 926's copy-width election for the routed half
    const bool thin = n_expert != 0u && 2u * n_blocks < 5u * n_expert && (in_dim / 64u) % 2u == 0u;
    cudaStream_t st = (cudaStream_t)stream;
    if (thin)
        return pd_nvpf_launch_pair<PD_NVPF_MOE_GU, 1u, 2u, PD_NVPF_GU, 1u, 1u>(a, na, b, nb, st);
    return pd_nvpf_launch_pair<PD_NVPF_MOE_GU, 1u, 1u, PD_NVPF_GU, 1u, 1u>(a, na, b, nb, st);
#endif
}

// 933: 923's routed down (its MPER-4, two-k-tile-copy arm) and 920's shared
// down (K = sff, reading act from column `aoff`, into y [ntok][embd] f32) in
// one grid - each its own launch's bits. cudaErrorNotSupported off that arm.
PD_EXPORT
int pd_nvf4_pf_moe_dn_sh(const void* data, const void* scale, const void* s2,
                         const void* sorted_row, const void* sorted_slot,
                         const void* block_expert, const void* topk_w, const void* act,
                         uint32_t aw, void* part, uint32_t ff, uint32_t embd, uint32_t k,
                         uint32_t n_blocks, const void* sdata, const void* sscale, float ss2,
                         void* y, uint32_t sff, uint32_t aoff, uint32_t ntok, void* stream) {
#ifndef PD_BS_HOST
    (void)data; (void)scale; (void)s2; (void)sorted_row; (void)sorted_slot;
    (void)block_expert; (void)topk_w; (void)act; (void)aw; (void)part; (void)ff; (void)embd;
    (void)k; (void)n_blocks; (void)sdata; (void)sscale; (void)ss2; (void)y; (void)sff;
    (void)aoff; (void)ntok; (void)stream;
    return cudaErrorNotSupported;
#else
    const uint32_t mt = embd / PD_NVPF_BM;
    if (!pd_nvpf_rd_ok() || mt % PD_NVPF_DN_MPER != 0u || (ff / 64u) % 2u != 0u)
        return cudaErrorNotSupported;
    if ((ff & 63u) != 0u || (embd & 127u) != 0u || (aw & 7u) != 0u || (sff & 63u) != 0u
        || (aoff & 7u) != 0u)
        return cudaErrorInvalidValue;
    PdNvPfArgs a = {};
    a.da = (const uint8_t*)data; a.sa = (const uint8_t*)scale; a.fa = (const float*)s2;
    a.x = (const __nv_bfloat16*)act; a.ldx = aw;
    a.act = (__nv_bfloat16*)part; a.ldy = embd;
    a.K = ff; a.M = embd; a.k = k;
    a.srow = (const uint32_t*)sorted_row; a.sslot = (const uint32_t*)sorted_slot;
    a.bexp = (const uint32_t*)block_expert; a.topk_w = (const float*)topk_w;
    a.n_blocks = n_blocks;
    PdNvPfArgs b = {};
    b.da = (const uint8_t*)sdata; b.sa = (const uint8_t*)sscale; b.fa1 = ss2;
    b.x = (const __nv_bfloat16*)act; b.ldx = aw; b.xoff = aoff;
    b.y = (float*)y; b.ldy = embd; b.yoff = 0u;
    b.K = sff; b.M = embd; b.ntok = ntok;
    const uint32_t na = k ? n_blocks * (mt / PD_NVPF_DN_MPER) : 0u;
    const uint32_t nb = sff && ntok ? pd_nvpf_dense_grid(mt, ntok, PD_NVPF_BN_DENSE) : 0u;
    return pd_nvpf_launch_pair<PD_NVPF_MOE_DN, PD_NVPF_DN_MPER, 2u, PD_NVPF_DENSE, 1u, 1u>(
        a, na, b, nb, (cudaStream_t)stream);
#endif
}
