// ---- tile-linear f8 weight layout ----------------------------------------
//
// The decode-band f8 GEMM measured 1.33 TB/s cold-stream on every kernel
// geometry (nz ladder, occupancy matrix, TMA-2D staging: all flat) while a
// dumb reader at the same grid hit 1.52 TB/s. roof_probe isolated the cause:
// the row-major weight walk reads 64-row x 128B boxes at stride in_dim and
// that ACCESS PATTERN caps at 1323 GB/s; per-CTA contiguous slabs stream
// 1490. The kernel was at its pattern roof, not a scheduling wall.
//
// Fix: repack f8w planes once at load into "lin" boxes so every CTA's K-walk
// is one sequential stream. Measured 138.9us -> 129.0us on the fused
// gate_up shape, bit-exact vs the ks rung.
//
// LAYOUT (per plane, in_dim % 128 == 0, rows padded to 128):
//   box(rt, kt) at (rt * nk + kt) * 16896, nk = in_dim / 128:
//     [    0, 16384)  data: row r (0..127), 16B chunk c (0..7) of the K-128
//                     span kt stored at r*128 + ((c ^ (r & 7)) * 16) - the
//                     SW128 smem image pre-applied, so a plain 1D bulk copy
//                     reproduces exactly what TMA SWIZZLE_128B would stage
//                     and the ldmatrix consumers keep their addressing.
//     [16384, 16896)  scales: row r's 4 ue8m0 bytes for blocks kt*4 .. +3
//   Total = 132 B per row per K-128 = 1.03125 B/param - identical to the
//   row-major data+scale pair it replaces (VRAM-neutral, planes freed).
//
// The 128-row box height is chosen so the PREFILL tma_kt consumer keeps its
// row addressing verbatim (one box = one K-128 W pair); the decode kernel
// (BM=64) fetches half-boxes - data halves are contiguous at +0/+8192 and
// scale halves at +16384/+16640.
//
// K-split survives the layout: a CTA's split walks a contiguous RANGE of
// boxes, so nz > 1 keeps sequential streams (small-tile shapes: the fused
// down proj is 80 tiles on a 188-SM die and needs nz=8 to fill it).

#define PD_LIN_BOX 16896u
#define PD_LIN_DATA 16384u

#if PD_MMA_OK && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
#define PD_LIN_DEV_OK 1
#else
#define PD_LIN_DEV_OK 0
#endif

// repack: grid (nrt, nk), 256 threads. 1024 data chunks (4/thread) + 128
// scale words (threads 0..127). OOB rows (out padded to 128) zero-fill.
__global__ void pd_f8w_repack_lin_kernel(
        const unsigned char* __restrict__ data, const unsigned char* __restrict__ scale,
        unsigned char* __restrict__ dst, uint32_t in_dim, uint32_t out_dim) {
    const uint32_t rt = blockIdx.x, kt = blockIdx.y, nk = gridDim.y;
    const uint32_t tid = threadIdx.x;
    unsigned char* box = dst + ((size_t)rt * nk + kt) * PD_LIN_BOX;
    const uint32_t n_blocks = in_dim >> 5;
    #pragma unroll
    for (uint32_t i = tid; i < 128u * 8u; i += 256u) {
        const uint32_t r = i >> 3, c = i & 7u;
        const uint32_t row = rt * 128u + r;
        int4 v = make_int4(0, 0, 0, 0);
        if (row < out_dim)
            v = *reinterpret_cast<const int4*>(
                data + (size_t)row * in_dim + kt * 128u + c * 16u);
        *reinterpret_cast<int4*>(box + r * 128u + ((c ^ (r & 7u)) * 16u)) = v;
    }
    if (tid < 128u) {
        const uint32_t row = rt * 128u + tid;
        uint32_t sv = 0;
        if (row < out_dim)
            sv = *reinterpret_cast<const uint32_t*>(
                scale + (size_t)row * n_blocks + kt * 4u);
        *reinterpret_cast<uint32_t*>(box + PD_LIN_DATA + tid * 4u) = sv;
    }
}

PD_EXPORT
int pd_f8w_repack_lin(const void* data, const void* scale, void* dst,
                      uint32_t in_dim, uint32_t out_dim, void* stream) {
    if (out_dim == 0) return 0;
    if ((in_dim & 127u) || (out_dim & 15u)) return cudaErrorInvalidValue;
    const uint32_t nrt = (out_dim + 127u) / 128u, nk = in_dim >> 7;
    dim3 grid(nrt, nk);
    pd_f8w_repack_lin_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(
        (const unsigned char*)data, (const unsigned char*)scale,
        (unsigned char*)dst, in_dim, out_dim);
    return pd_launch_status();
}

// gu-interleave twin: same boxes, but the SOURCE row is remapped so a
// gate/up pair p lands at box rows (p>>3)*16+(p&7) and +8 within its tile.
// A kt3 acc fragment holds output rows r0 and r0+8 in the same thread, so
// with this layout the fused geglu+quant epilogue (pd_f8_gemm_lin_gu) pairs
// gate/up with no cross-thread traffic and each per-32 ff block lives inside
// one warp half. Downstream lin GEMMs are layout-blind (they compute the
// same rows, permuted); only the geglu stage changes addressing
// (pd_quantize_e4m3_geglu2i). Measured bit-identical to the unpermuted
// chain end-to-end, -13.9% on the b=1792 gu+geglu2 pair.
__global__ void pd_f8w_repack_lin_gui_kernel(
        const unsigned char* __restrict__ data, const unsigned char* __restrict__ scale,
        unsigned char* __restrict__ dst, uint32_t in_dim, uint32_t out_dim) {
    const uint32_t rt = blockIdx.x, kt = blockIdx.y, nk = gridDim.y;
    const uint32_t tid = threadIdx.x;
    unsigned char* box = dst + ((size_t)rt * nk + kt) * PD_LIN_BOX;
    const uint32_t n_blocks = in_dim >> 5;
    const uint32_t half = out_dim >> 1;
    // bijection on [0, out_dim) for out_dim % 16 == 0: pair index from the
    // boxed row, gate half below sub-row 8, up half above
    auto src_of = [&](uint32_t row) {
        const uint32_t p = (row >> 4) * 8u + (row & 7u);
        return (row & 15u) < 8u ? p : half + p;
    };
    #pragma unroll
    for (uint32_t i = tid; i < 128u * 8u; i += 256u) {
        const uint32_t r = i >> 3, c = i & 7u;
        const uint32_t row = rt * 128u + r;
        int4 v = make_int4(0, 0, 0, 0);
        if (row < out_dim)
            v = *reinterpret_cast<const int4*>(
                data + (size_t)src_of(row) * in_dim + kt * 128u + c * 16u);
        *reinterpret_cast<int4*>(box + r * 128u + ((c ^ (r & 7u)) * 16u)) = v;
    }
    if (tid < 128u) {
        const uint32_t row = rt * 128u + tid;
        uint32_t sv = 0;
        if (row < out_dim)
            sv = *reinterpret_cast<const uint32_t*>(
                scale + (size_t)src_of(row) * n_blocks + kt * 4u);
        *reinterpret_cast<uint32_t*>(box + PD_LIN_DATA + tid * 4u) = sv;
    }
}

PD_EXPORT
int pd_f8w_repack_lin_gui(const void* data, const void* scale, void* dst,
                          uint32_t in_dim, uint32_t out_dim, void* stream) {
    if (out_dim == 0) return 0;
    if ((in_dim & 127u) || (out_dim & 15u)) return cudaErrorInvalidValue;
    const uint32_t nrt = (out_dim + 127u) / 128u, nk = in_dim >> 7;
    dim3 grid(nrt, nk);
    pd_f8w_repack_lin_gui_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(
        (const unsigned char*)data, (const unsigned char*)scale,
        (unsigned char*)dst, in_dim, out_dim);
    return pd_launch_status();
}

// ---- block-scale (bs) variant: official-FP8 byte passthrough --------------
// The checkpoint's raw e4m3 bytes with one f32 scale per 128x128 block
// (Qwen *-FP8 `weight_scale_inv` grid) instead of the per-32 ue8m0 strip:
// boxes are data-only 16384 B = 1.000 B/param, 3.03% less weight traffic on
// the decode band that is 67.8% of c8 GPU time. Same SW128 pre-swizzle, same
// consumers; the scale plane is a separate f32 [out/128][in/128] grid.
#define PD_LINBS_BOX 16384u

// repack: raw row-major e4m3 -> data-only lin boxes (no strip). The scale
// plane converts host-side (tiny). OOB rows zero-fill.
__global__ void pd_f8w_repack_lin_bs_kernel(
        const unsigned char* __restrict__ data, unsigned char* __restrict__ dst,
        uint32_t in_dim, uint32_t out_dim) {
    const uint32_t rt = blockIdx.x, kt = blockIdx.y, nk = gridDim.y;
    const uint32_t tid = threadIdx.x;
    unsigned char* box = dst + ((size_t)rt * nk + kt) * PD_LINBS_BOX;
    #pragma unroll
    for (uint32_t i = tid; i < 128u * 8u; i += 256u) {
        const uint32_t r = i >> 3, c = i & 7u;
        const uint32_t row = rt * 128u + r;
        int4 v = make_int4(0, 0, 0, 0);
        if (row < out_dim)
            v = *reinterpret_cast<const int4*>(
                data + (size_t)row * in_dim + kt * 128u + c * 16u);
        *reinterpret_cast<int4*>(box + r * 128u + ((c ^ (r & 7u)) * 16u)) = v;
    }
}

PD_EXPORT
int pd_f8w_repack_lin_bs(const void* data, void* dst, uint32_t in_dim,
                         uint32_t out_dim, void* stream) {
    if (out_dim == 0) return 0;
    if ((in_dim & 127u) || (out_dim & 15u)) return cudaErrorInvalidValue;
    const uint32_t nrt = (out_dim + 127u) / 128u, nk = in_dim >> 7;
    dim3 grid(nrt, nk);
    pd_f8w_repack_lin_bs_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(
        (const unsigned char*)data, (unsigned char*)dst, in_dim, out_dim);
    return pd_launch_status();
}

// data-only twin of pd_f8w_repack_lin_gui: gu-interleaved boxes without the
// scale strip, for the rowwise (pc) plane class - the per-row exponents live
// in a separate wse byte vector the caller keeps (interleaved to box row
// order host-side, it is tiny).
__global__ void pd_f8w_repack_lin_bs_gui_kernel(
        const unsigned char* __restrict__ data, unsigned char* __restrict__ dst,
        uint32_t in_dim, uint32_t out_dim) {
    const uint32_t rt = blockIdx.x, kt = blockIdx.y, nk = gridDim.y;
    const uint32_t tid = threadIdx.x;
    unsigned char* box = dst + ((size_t)rt * nk + kt) * PD_LINBS_BOX;
    const uint32_t half = out_dim >> 1;
    auto src_of = [&](uint32_t row) {
        const uint32_t p = (row >> 4) * 8u + (row & 7u);
        return (row & 15u) < 8u ? p : half + p;
    };
    #pragma unroll
    for (uint32_t i = tid; i < 128u * 8u; i += 256u) {
        const uint32_t r = i >> 3, c = i & 7u;
        const uint32_t row = rt * 128u + r;
        int4 v = make_int4(0, 0, 0, 0);
        if (row < out_dim)
            v = *reinterpret_cast<const int4*>(
                data + (size_t)src_of(row) * in_dim + kt * 128u + c * 16u);
        *reinterpret_cast<int4*>(box + r * 128u + ((c ^ (r & 7u)) * 16u)) = v;
    }
}

PD_EXPORT
int pd_f8w_repack_lin_bs_gui(const void* data, void* dst, uint32_t in_dim,
                             uint32_t out_dim, void* stream) {
    if (out_dim == 0) return 0;
    if ((in_dim & 127u) || (out_dim & 15u)) return cudaErrorInvalidValue;
    const uint32_t nrt = (out_dim + 127u) / 128u, nk = in_dim >> 7;
    dim3 grid(nrt, nk);
    pd_f8w_repack_lin_bs_gui_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(
        (const unsigned char*)data, (unsigned char*)dst, in_dim, out_dim);
    return pd_launch_status();
}

// ---- decode-band lin GEMM (BM=64 half-boxes, 1D bulk + mbarrier ring) ----
// The bench-verified TMA hybrid: 256-thread barrier-loop consume (exact ks
// numerics - same ldmatrix fragments, same m16n8k32 e4m3 mma, same exponent
// fold, same accumulation order = bit-exact), single-thread bulk staging.
// One __syncthreads per K-tile (was two) and no per-thread cp.async issue.
// ~25 KB smem at BN=32 -> 4 CTAs/SM (the co-residency the +16-KPAD cp.async
// layout could not reach). B activations ride a SWIZZLE_128B TMA box (sh_b
// must stay 1024B-aligned - the swizzle atom; learned the hard way).
// KP (the mid-M lever ported to the decode frame): boxes per
// mbarrier PHASE. KP=1 is the incumbent ring (one wait + one syncthreads +
// one issue per K-128 box). KP=2 batches the ring at K-256 quanta - the
// slot count doubles (STAGES phases x KP slots), the sync ops per byte
// halve, per-CTA flight doubles; per-element K order is preserved (slots
// consumed kt-ascending inside a phase) so KP arms are BITWISE vs KP=1.
// KP>1 crosses the 48 KB static-smem line -> storage moves to dynamic
// smem (launch must pass pd_f8_lin_kp_smem() bytes and raise the func
// attribute); the static arrays degenerate to 1 byte in those
// instantiations so they cost nothing.
template <uint32_t BN, uint32_t STAGES, uint32_t MINB, bool BS = false, bool RW = false,
          uint32_t KP = 1u>
__global__ void __launch_bounds__(256, MINB) pd_f8_gemm_lin_kernel(
        const unsigned char* __restrict__ wlin, const __grid_constant__ CUtensorMap ymap,
        const unsigned char* __restrict__ xs, float* __restrict__ y,
        uint32_t in_dim, uint32_t out_dim, uint32_t batch,
        const float* __restrict__ wsc = nullptr,
        const unsigned char* __restrict__ wse = nullptr) {
#if PD_LIN_DEV_OK && (__CUDA_ARCH__ >= 900)
    constexpr uint32_t BM = 64u, NWARP = 8u, NTH = 256u;
    constexpr uint32_t WR = BM / 16u, WC = NWARP / WR, CPW = BN / WC;
    constexpr uint32_t NSUB = CPW / 8u, NSUBK = 4u;
    constexpr uint32_t HBOX = 8192u + 256u;  // half-box: data + scales
    constexpr uint32_t NSLOT = STAGES * KP;
    constexpr uint32_t SB_SZ = BN * 128u;
    __shared__ __align__(1024) unsigned char st_b[KP == 1u ? STAGES : 1u]
                                                [KP == 1u ? BN * 128u : 1u];
    __shared__ __align__(128) unsigned char st_a[KP == 1u ? STAGES : 1u]
                                               [KP == 1u ? HBOX : 1u];
    __shared__ __align__(16) unsigned char st_xs[KP == 1u ? STAGES : 1u]
                                                [KP == 1u ? BN * NSUBK : 1u];
    __shared__ __align__(8) unsigned long long mb[STAGES];
    extern __shared__ unsigned char pd_lin_kp_dyn[];
    unsigned char *sb_base, *sa_base, *sxs_base;
    if constexpr (KP == 1u) {
        sb_base = &st_b[0][0]; sa_base = &st_a[0][0]; sxs_base = &st_xs[0][0];
    } else {
        // dynamic base is not 1024-aligned by contract - round up (launch
        // over-allocates the slack); sa slots stay 128-aligned because
        // SB_SZ is a multiple of 1024 and HBOX of 128
        uintptr_t p = ((uintptr_t)pd_lin_kp_dyn + 1023u) & ~(uintptr_t)1023u;
        sb_base = (unsigned char*)p;
        sa_base = sb_base + (size_t)NSLOT * SB_SZ;
        sxs_base = sa_base + (size_t)NSLOT * HBOX;
    }
    auto slot_b = [&](uint32_t s) { return sb_base + (size_t)s * SB_SZ; };
    auto slot_a = [&](uint32_t s) { return sa_base + (size_t)s * HBOX; };
    auto slot_xs = [&](uint32_t s) { return sxs_base + (size_t)s * (BN * NSUBK); };
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, t = lane & 3u;
    const uint32_t wr = (warp % WR) * 16u, wc = (warp / WR) * CPW;
    const uint32_t rt64 = blockIdx.x;
    const uint32_t row_base = rt64 * BM;
    // col tiling: blockIdx.y owns batch cols [col_base, col_base+BN)
    // so the decode-band kernel reaches the wide-decode M (~128 spec rows)
    // the tiny-r lane's batch<=BN cap excluded. Legacy grids launch y=1 ->
    // col_base 0 -> bitwise-identical to the pre-tiling kernel.
    const uint32_t col_base = blockIdx.y * BN;
    const uint32_t n_blocks = in_dim >> 5;
    const uint32_t nk = in_dim >> 7;
    // K-split: box range [kt_lo, kt_hi) - contiguous stream per split
    const uint32_t nz = gridDim.z;
    uint32_t kt_lo = 0, kt_hi = nk;
    if (nz > 1u) {
        const uint32_t per = (nk + nz - 1u) / nz;
        kt_lo = blockIdx.z * per;
        kt_hi = kt_lo + per < nk ? kt_lo + per : nk;
        y += (size_t)blockIdx.z * out_dim * batch;
        if (kt_lo >= kt_hi) {
            // empty tail split still owns its partial plane: zero it so the
            // combine's sum stays exact (this tile's col range only)
            const uint32_t c_hi = col_base + BN < batch ? col_base + BN : batch;
            for (uint32_t i = tid; i < BM; i += NTH) {
                const uint32_t r = row_base + i;
                if (r < out_dim)
                    for (uint32_t c = col_base; c < c_hi; ++c) y[(size_t)c * out_dim + r] = 0.f;
            }
            return;
        }
    }
    // RW (rowwise pc plane): strip-free boxes like BS, but the row exponent
    // stays on the integer fold path - loaded once per row from wse
    constexpr uint32_t BOXSZ = (BS || RW) ? PD_LINBS_BOX : PD_LIN_BOX;
    const unsigned char* wbox0 = wlin + ((size_t)(rt64 >> 1) * nk) * BOXSZ;
    const uint32_t dhalf = (rt64 & 1u) * 8192u;
    const uint32_t shalf = PD_LIN_DATA + (rt64 & 1u) * 256u;
    // bs: one f32 scale per (row-tile, K-128) block - both 64-row halves of a
    // box share the row-tile's scale row
    const float* wsc_rt = BS ? wsc + (size_t)(rt64 >> 1) * nk : nullptr;
    const uint32_t mb0 = (uint32_t)__cvta_generic_to_shared(mb);
    if (tid == 0u) {
        #pragma unroll
        for (uint32_t s = 0; s < STAGES; ++s)
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(mb0 + s * 8u));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    auto stage_sc = [&](uint32_t kt, uint32_t slot) {
        unsigned char* sxs = slot_xs(slot);
        for (uint32_t i = tid; i < BN * NSUBK; i += NTH) {
            const uint32_t col = i >> 2, sb = i & 3u;
            const bool ok = col_base + col < batch && kt * NSUBK + sb < n_blocks;
            sxs[col * NSUBK + sb] = ok
                ? xs[(size_t)(col_base + col) * n_blocks + kt * NSUBK + sb] : 0u;
        }
    };
    // one arrive/expect_tx per PHASE covering its (up to) KP boxes; the
    // per-box TMA triple is today's issue body
    auto issue_ph = [&](uint32_t kt0, uint32_t ph) {
        const uint32_t m = mb0 + ph * 8u;
        uint32_t nb = kt_hi - kt0;
        if (nb > KP) nb = KP;
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                     ::"r"(m),
                     "r"((((BS || RW) ? 8192u : 8192u + 256u) + BN * 128u) * nb));
        for (uint32_t j = 0; j < nb; ++j) {
            const uint32_t kt = kt0 + j, slot = ph * KP + j;
            const uint32_t wd = (uint32_t)__cvta_generic_to_shared(slot_a(slot));
            const uint32_t yd = (uint32_t)__cvta_generic_to_shared(slot_b(slot));
            const unsigned char* box = wbox0 + (size_t)kt * BOXSZ;
            asm volatile(
                "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                " [%0], [%1], %2, [%3];" ::"r"(wd), "l"(box + dhalf), "r"(8192u),
                "r"(m) : "memory");
            if (!BS && !RW) {
                const uint32_t sd = wd + 8192u;
                asm volatile(
                    "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1], %2, [%3];" ::"r"(sd), "l"(box + shalf), "r"(256u),
                    "r"(m) : "memory");
            }
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                " [%0], [%1, {%2, %3}], [%4];" ::"r"(yd),
                "l"(&ymap), "r"((int)(kt * 128u)), "r"((int)col_base), "r"(m) : "memory");
        }
    };
    for (uint32_t j = 0; j < KP && kt_lo + j < kt_hi; ++j)
        stage_sc(kt_lo + j, j);
    __syncthreads();  // mbarrier init + first scales visible
    if (tid == 0u)
        for (uint32_t s = 0; s < STAGES && kt_lo + s * KP < kt_hi; ++s)
            issue_ph(kt_lo + s * KP, s);
    const uint32_t ldm_l7 = lane & 7u;
    const uint32_t ldm_arow = wr + ((lane & 8u) ? 8u : 0u) + ldm_l7;
    const uint32_t ca_hi = (lane & 16u) ? 1u : 0u;
    const uint32_t cb_hi = (lane & 8u) ? 1u : 0u;
    // RW: the row exponent is K-invariant - load once (wse padded to the
    // 128-row tail, so the pad rows read defined bytes)
    const int rwe0 = RW ? (int)wse[row_base + wr + g] : 0;
    const int rwe8 = RW ? (int)wse[row_base + wr + 8u + g] : 0;
    float acc[NSUB][4] = {};
    for (uint32_t ktp = kt_lo; ktp < kt_hi; ktp += KP) {
        const uint32_t i_ph = (ktp - kt_lo) / KP;
        const uint32_t ph = i_ph % STAGES;
        if (ktp + KP < kt_hi) {
            const uint32_t nph = (i_ph + 1u) % STAGES;
            #pragma unroll
            for (uint32_t j = 0; j < KP; ++j)
                if (ktp + KP + j < kt_hi) stage_sc(ktp + KP + j, nph * KP + j);
        }
        {
            const uint32_t m = mb0 + ph * 8u;
            const uint32_t par = (i_ph / STAGES) & 1u;
            asm volatile(
                "{\n\t.reg .pred P;\n"
                "PD_F8LIN_WAIT_%=:\n\t"
                "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
                "@!P bra PD_F8LIN_WAIT_%=;\n\t}" ::"r"(m), "r"(par) : "memory");
        }
        #pragma unroll
        for (uint32_t j = 0; j < KP; ++j) {
            const uint32_t kt = ktp + j;
            if (KP > 1u && kt >= kt_hi) break;
            const uint32_t slot = ph * KP + j;
            const unsigned char* wp = slot_a(slot);
            const unsigned char* bp = slot_b(slot);
            const unsigned char* sxs = slot_xs(slot);
            // bs: the whole K-128 box shares one weight scale - folded into
            // the per-32 activation exponent (f00==f80, per-row ue8m0 gone)
            const float fs = BS ? wsc_rt[kt] : 0.f;
            #pragma unroll
            for (uint32_t sb = 0; sb < NSUBK; ++sb) {
                const uint32_t ca = sb * 2u + ca_hi;
                int a0, a1, a2, a3;
                pd_mma_ldm_x4(wp + ldm_arow * 128u + ((ca ^ (ldm_arow & 7u)) * 16u),
                              a0, a1, a2, a3);
                const int ws0 = BS ? 0 : (RW ? rwe0 : (int)wp[8192u + (wr + g) * NSUBK + sb]);
                const int ws8 = BS ? 0 : (RW ? rwe8 : (int)wp[8192u + (wr + 8u + g) * NSUBK + sb]);
                #pragma unroll
                for (uint32_t sub = 0; sub < NSUB; ++sub) {
                    const uint32_t csub = wc + sub * 8u;
                    const uint32_t col = csub + ldm_l7;
                    const uint32_t cb = sb * 2u + cb_hi;
                    int b0, b1;
                    pd_mma_ldm_x2(bp + col * 128u + ((cb ^ (col & 7u)) * 16u), b0, b1);
                    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
                    asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
                    const int xc0 = (int)sxs[(csub + 2u * t) * NSUBK + sb];
                    const int xc1 = (int)sxs[(csub + 2u * t + 1u) * NSUBK + sb];
                    if (BS) {
                        const float fx0 = fs * __uint_as_float((uint32_t)xc0 << 23);
                        const float fx1 = fs * __uint_as_float((uint32_t)xc1 << 23);
                        acc[sub][0] += fx0 * d0;
                        acc[sub][1] += fx1 * d1;
                        acc[sub][2] += fx0 * d2;
                        acc[sub][3] += fx1 * d3;
                    } else {
                        const float f00 = __uint_as_float((uint32_t)(ws0 + xc0 - 127) << 23);
                        const float f01 = __uint_as_float((uint32_t)(ws0 + xc1 - 127) << 23);
                        const float f80 = __uint_as_float((uint32_t)(ws8 + xc0 - 127) << 23);
                        const float f81 = __uint_as_float((uint32_t)(ws8 + xc1 - 127) << 23);
                        acc[sub][0] += f00 * d0;
                        acc[sub][1] += f01 * d1;
                        acc[sub][2] += f80 * d2;
                        acc[sub][3] += f81 * d3;
                    }
                }
            }
        }
        __syncthreads();  // consumers done with the phase; next scales visible
        if (tid == 0u && ktp + STAGES * KP < kt_hi) issue_ph(ktp + STAGES * KP, ph);
    }
    const uint32_t r0 = row_base + wr + g, r8 = row_base + wr + 8u + g;
    #pragma unroll
    for (uint32_t sub = 0; sub < NSUB; ++sub) {
        const uint32_t c0 = col_base + wc + sub * 8u + 2u * t, c1 = c0 + 1u;
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
    (void)wlin; (void)ymap; (void)xs; (void)y;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}

// dynamic smem bytes a KP>1 instantiation needs at launch (slots x
// (b-tile + half-box + xs) + the 1024 alignment slack the kernel rounds
// away). KP==1 instantiations use static smem and take 0.
static inline uint32_t pd_f8_lin_kp_smem(uint32_t bn, uint32_t stages, uint32_t kp) {
    return stages * kp * (bn * 128u + 8448u + bn * 4u) + 1024u;
}

// activation tmap cache: (ptr, in_dim, rows) - decode xq scratch pointers
// are stable for the model's life, so CUDA-graph capture sees ready maps
struct PdLinYEnt { const void* p; uint32_t in; uint32_t rows; CUtensorMap m; };

template <bool BS, bool RW = false>
static int pd_f8_gemm_lin_go(const void* wlin, const void* wsc, const void* xq,
                             const void* xs, void* part, void* y, uint32_t in_dim,
                             uint32_t out_dim, uint32_t batch, void* stream,
                             const void* wse = nullptr) {
#ifndef PD_BS_HOST
    (void)wlin; (void)wsc; (void)xq; (void)xs; (void)part; (void)y; (void)in_dim;
    (void)out_dim; (void)batch; (void)stream; (void)wse;
    return cudaErrorNotSupported;
#else
    if (out_dim == 0 || batch == 0) return 0;
    if ((in_dim & 127u) || (out_dim & 15u) || batch > 64u)
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
    // nz: die-filling tile counts run split-free (bench: 544 CTAs = 129 us);
    // small-tile shapes (fused down: 80) split at box granularity to fill.
    // PD_LIN_NZ overrides for rebuild-free tuning.
    static int nz_env = -1;
    if (nz_env < 0) {
        const char* e = pd_env("PD_LIN_NZ");
        nz_env = e ? atoi(e) : 0;
    }
    uint32_t nz = 1u;
    // Depth re-election (honest cold-W - the
    // earlier nz4 election for the 80-tile shapes was made on L2-resident
    // planes: NC=3 x 32 MB fit the 126 MB L2 and read 1.7 "TB/s"). With
    // copies x plane > 220 MB the sub-die grids (tiles < SMs, 1 CTA/SM)
    // are FLIGHT-starved, not split-starved: a third ring stage at nz=1
    // beats both s2-nz1 and the nz4 split on every 80-tile shape (wo 29.1
    // vs 32.2/31.4, dnout 29.8 vs 32.1/30.9, down 69.9 vs 78.3/70.7 us,
    // bitwise - per-element K order is stage-invariant), and nz=1 deletes
    // the combine launch + partial-plane traffic from the decode graph
    // (~128 combines/tick on q36). Die-filling grids (qkv 224 / dnig 256 /
    // gu 544) measured best at s2-nz1 - MINB 4->3 costs them co-residency
    // (61.3/65.4/134.5 vs 58.9/64.8/132.2). Rule: tiles < SMs -> STAGES=3
    // at nz=1; else STAGES=2 at nz=1. K-split at b<=32 is dead (fiction-
    // elected). PD_LIN_NZ still overrides; PD_LIN_S3=0 reverts the depth.
    static int s3_env = -1;
    if (s3_env < 0) {
        const char* e = pd_env("PD_LIN_S3");
        s3_env = e ? atoi(e) : 1;
    }
    const bool s3 = s3_env != 0 && batch <= 32u && tiles < (uint32_t)nsm;
    // 64-TILE REFINEMENT (cold-W, per plane). The rule above sets the nz=1
    // boundary at `tiles < nsm` (188), but the split boundary is much lower:
    // measured per plane at 64-wide tiles, nz=2 wins everything at 64 tiles
    // and loses everything at >=80.
    //
    //   plane      tiles   nz=1 b32   nz=2 b32
    //   gran_o        64     17.4us     15.0us   -14%
    //   gran_down     64     47.4       41.5     -12%
    //   qwen_down     80     64.7       65.3
    //   gran_qkv      96     21.1       23.3
    //   gran_gu      400     74.7       79.8
    //   qwen_gu      544    125.4      135.5
    //
    // It holds across the arm's whole width range, most strongly at the top:
    // gran_down b64 81.1 -> 48.4us (-40%), gran_o b64 28.6 -> 21.2 (-26%).
    // This does not contradict the q36 re-election above, which measured
    // 80-tile shapes (and nz=4, not nz=2) and correctly kept them at nz=1 -
    // the two datasets agree the boundary sits between 64 and 80 tiles. Only
    // the nz choice moves; STAGES=3 still applies by the rule above. Callers
    // already size `part` for 8 splits (see gpu/fp8.rs), so raising nz here
    // cannot outrun the scratch. PD_LIN_NZ still overrides.
    if (tiles <= 64u) nz = 2u;
    if (nz_env >= 1 && nz_env <= 8) nz = (uint32_t)nz_env;
    // per-(ptr, shape) activation tmaps; wraparound table like the attention
    // launchers' caches
    // key on exact batch: a map encoded with gdim rows=b zero-fills every
    // row >= b, so it must never serve a later, larger batch
    static PdLinYEnt yc[64];
    static uint32_t yn = 0;
    CUtensorMap* ym = nullptr;
    for (uint32_t i = 0; i < yn; ++i)
        if (yc[i].p == xq && yc[i].in == in_dim && yc[i].rows == batch) {
            ym = &yc[i].m;
            break;
        }
    if (!ym) {
        if (yn >= 64u) yn = 0;
        bool ok = batch <= 32u
            ? pd_tmap_2d_h32(&yc[yn].m, xq, in_dim, batch)
            : pd_tmap_2d_h64(&yc[yn].m, xq, in_dim, batch);
        if (!ok) return cudaErrorNotSupported;
        yc[yn].p = xq; yc[yn].in = in_dim; yc[yn].rows = batch;
        ym = &yc[yn++].m;
    }
    float* dst = nz > 1u ? (float*)part : (float*)y;
    dim3 grid(tiles, 1, nz);
    if (batch <= 32u && s3)
        pd_f8_gemm_lin_kernel<32u, 3u, 3u, BS, RW><<<grid, 256, 0, st>>>(
            (const unsigned char*)wlin, *ym, (const unsigned char*)xs, dst,
            in_dim, out_dim, batch, (const float*)wsc,
            (const unsigned char*)wse);
    else if (batch <= 32u)
        pd_f8_gemm_lin_kernel<32u, 2u, 4u, BS, RW><<<grid, 256, 0, st>>>(
            (const unsigned char*)wlin, *ym, (const unsigned char*)xs, dst,
            in_dim, out_dim, batch, (const float*)wsc,
            (const unsigned char*)wse);
    else
        pd_f8_gemm_lin_kernel<64u, 2u, 2u, BS, RW><<<grid, 256, 0, st>>>(
            (const unsigned char*)wlin, *ym, (const unsigned char*)xs, dst,
            in_dim, out_dim, batch, (const float*)wsc,
            (const unsigned char*)wse);
    if (nz > 1u) {
        uint32_t n = out_dim * batch;
        pd_q8_0_gemm_mma_ks_combine_kernel<<<(n + 255u) / 256u, 256, 0, st>>>(
            (const float*)part, nullptr, (float*)y, n, nz, out_dim);
    }
    return pd_launch_status();
#endif
}

PD_EXPORT
int pd_f8_gemm_lin(const void* wlin, const void* xq, const void* xs,
                   void* part, void* y, uint32_t in_dim, uint32_t out_dim,
                   uint32_t batch, void* stream) {
    return pd_f8_gemm_lin_go<false>(wlin, nullptr, xq, xs, part, y, in_dim,
                                    out_dim, batch, stream);
}

// block-scale decode GEMM: data-only lin boxes + f32 [out/128][in/128] scale
// plane (official-FP8 byte passthrough - see the bs format note above)
PD_EXPORT
int pd_f8_gemm_lin_bs(const void* wlin, const void* wsc, const void* xq,
                      const void* xs, void* part, void* y, uint32_t in_dim,
                      uint32_t out_dim, uint32_t batch, void* stream) {
    return pd_f8_gemm_lin_go<true>(wlin, wsc, xq, xs, part, y, in_dim,
                                   out_dim, batch, stream);
}

// rowwise decode GEMM: data-only lin boxes + per-row ue8m0 byte vector (the
// pc plane class). wse must be padded to the 128-row tail.
PD_EXPORT
int pd_f8_gemm_lin_r(const void* wlin, const void* wse, const void* xq,
                     const void* xs, void* part, void* y, uint32_t in_dim,
                     uint32_t out_dim, uint32_t batch, void* stream) {
    return pd_f8_gemm_lin_go<false, true>(wlin, nullptr, xq, xs, part, y,
                                          in_dim, out_dim, batch, stream, wse);
}

// ---- prefill lin twin of pd_f8_gemm_w8_tma_kt ----------------------------
// Same warp-specialized producer/consumer structure, same fragments, same
// hw block-scale mma (or sw fold on non-120a) - the only deltas: W arrives
// as one 1D bulk of a lin box (16896 B, K-128 pair = one box, rows already
// the SW128 smem image) instead of a 2D tensor fetch, and W scales are read
// from the box tail instead of a producer-staged slab (wsc dropped; its
// 1 KB moved into the W ring). smem total stays 67600.
//
// Trailing `k` = kernel: the name must stay distinct from the exported host
// launcher `pd_f8_gemm_lin_kt` below. Sharing it made every
// `cudaFuncSetAttribute((const void*)pd_f8_gemm_lin_kt<...>, ...)` an
// ambiguous overload set under MSVC - a cast to `const void*` supplies no
// target signature, so cl.exe will not let the explicit template args pick
// the template over the non-template launcher (clang/gcc do). That broke
// every Windows build carrying sm_120, because build.ps1 only defines
// PD_BS_HOST for that arch and these casts live in its #else branch, so it
// went unnoticed for a long while.
template <bool WIN = false, bool O16 = false>
__global__ void __launch_bounds__(384, 1) pd_f8_gemm_lin_ktk(
    const unsigned char* __restrict__ wlin, const __grid_constant__ CUtensorMap ymap,
    const unsigned char* __restrict__ xs, float* __restrict__ y,
    uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
#if PD_F8W8_TMA_OK
    constexpr uint32_t PAIR16 = 16384u;
    extern __shared__ __align__(128) unsigned char pd_lin_sh[];
    unsigned char* wdat = pd_lin_sh;                // 2 pairs x 16896 (box)
    unsigned char* ydat = pd_lin_sh + 33792u;       // 2 pairs x 16 KB
    unsigned char* ysc = pd_lin_sh + 66560u;        // 2 pairs x 512 B
    unsigned long long* mb = (unsigned long long*)(pd_lin_sh + 67584u);

    const uint32_t tid = threadIdx.x;
    const uint32_t n_kb = in_dim >> 5;
    const uint32_t nk = (in_dim + 63u) / 64u;
    const uint32_t nsp = (nk + 1u) >> 1;
    const uint32_t nkboxes = in_dim >> 7;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t nct = batch_pad >> 7;
    const uint32_t tile = blockIdx.x;
    const uint32_t row_base = (tile / nct) * 128u;
    const uint32_t col_base = (tile % nct) * 128u;
    const unsigned char* wboxes = wlin + (size_t)(row_base >> 7) * nkboxes * PD_LIN_BOX;

    if (tid == 0u) {
        const uint32_t m0 = (uint32_t)__cvta_generic_to_shared(mb);
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 128;" ::"r"(m0));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 128;" ::"r"(m0 + 8u));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    asm volatile("bar.sync 0, 384;");

    if (tid >= 256u) {
        // ---------------- producer warps 8-11 ----------------
        const uint32_t ptid = tid - 256u;
        unsigned char syr[2][2];
        for (uint32_t sp = 0; sp < nsp; ++sp) {
            const uint32_t b = sp % 2u;
            #pragma unroll
            for (uint32_t h = 0; h < 2u; ++h) {
                #pragma unroll
                for (uint32_t kb = 0; kb < 2u; ++kb) {
                    const uint32_t kt = sp * 2u + h;
                    const bool yok = (col_base + ptid) < batch && kt * 2u + kb < n_kb;
                    syr[h][kb] = yok ? xs[(size_t)(col_base + ptid) * n_kb + kt * 2u + kb] : 0u;
                }
            }
            if (sp >= 2u) asm volatile("bar.sync %0, 384;" ::"r"(1u + b));
            #pragma unroll
            for (uint32_t h = 0; h < 2u; ++h) {
                #pragma unroll
                for (uint32_t kb = 0; kb < 2u; ++kb)
                    ysc[b * 512u + h * 256u + ptid * 2u + kb] = syr[h][kb];
            }
            const uint32_t m = (uint32_t)__cvta_generic_to_shared(mb) + b * 8u;
            if (ptid == 0u) {
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 33280;" ::"r"(m));
                const uint32_t wd = (uint32_t)__cvta_generic_to_shared(wdat + b * PD_LIN_BOX);
                const uint32_t yd = (uint32_t)__cvta_generic_to_shared(ydat + b * PAIR16);
                asm volatile(
                    "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1], %2, [%3];" ::"r"(wd),
                    "l"(wboxes + (size_t)sp * PD_LIN_BOX), "r"(PD_LIN_BOX), "r"(m)
                    : "memory");
                const int ck = (int)(sp * 128u);
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3}], [%4];" ::"r"(yd),
                    "l"(&ymap), "r"(ck), "r"((int)col_base), "r"(m)
                    : "memory");
            } else {
                asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(m));
            }
        }
        return;
    }

    // ---------------- consumer warps 0-7 ----------------
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, tq = lane & 3u;
    const uint32_t i0 = (warp & 1u) * 64u;
    const uint32_t c0w = (warp >> 1) * 32u;

    float acc[16][4] = {};
    uint32_t ph0 = 0u, ph1 = 0u;

    for (uint32_t sp = 0; sp < nsp; ++sp) {
        const uint32_t b = sp % 2u;
        const uint32_t m = (uint32_t)__cvta_generic_to_shared(mb) + b * 8u;
        const uint32_t ph = (b == 0u) ? ph0 : ph1;
        asm volatile(
            "{\n\t.reg .pred P;\n"
            "PD_LINKT_WAIT_%=:\n\t"
            "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
            "@!P bra PD_LINKT_WAIT_%=;\n\t}" ::"r"(m), "r"(ph) : "memory");
        if (b == 0u) ph0 ^= 1u; else ph1 ^= 1u;

        const unsigned char* wp = wdat + b * PD_LIN_BOX;
        const unsigned char* yp = ydat + b * PAIR16;
        #pragma unroll
        for (uint32_t h = 0; h < 2u; ++h) {
            const uint32_t kt = sp * 2u + h;
            if (kt >= nk) break;

            uint32_t am[4][2][4];
#if PD_BS_OK
            uint32_t sa[4];
#endif
            #pragma unroll
            for (uint32_t s = 0; s < 4u; ++s) {
                const uint32_t rr = i0 + s * 16u + ((lane >> 3) & 1u) * 8u + (lane & 7u);
                #pragma unroll
                for (uint32_t kb = 0; kb < 2u; ++kb) {
                    const uint32_t c = h * 4u + kb * 2u + (lane >> 4);
                    pd_ldm_x4(am[s][kb], wp + rr * 128u + ((c ^ (rr & 7u)) * 16u));
                }
#if PD_BS_OK
                const uint32_t r0 = i0 + s * 16u + g;
                const uint32_t rs = (tq & 1u) ? r0 + 8u : r0;
                sa[s] = *(const unsigned short*)(wp + PD_LIN_DATA + rs * 4u + h * 2u);
#endif
            }
            uint32_t bmj[4][4];
#if PD_BS_OK
            uint32_t sbj[4];
#endif
            #pragma unroll
            for (uint32_t j = 0; j < 4u; ++j) {
                const uint32_t col = c0w + j * 8u + (lane & 7u);
                const uint32_t c = h * 4u + (lane >> 3);
                pd_ldm_x4(bmj[j], yp + col * 128u + ((c ^ (col & 7u)) * 16u));
#if PD_BS_OK
                sbj[j] = *(const unsigned short*)(ysc + b * 512u + h * 256u + (c0w + j * 8u + g) * 2u);
#endif
            }
#if PD_BS_OK
            if (WIN && (h == 1u || sp * 2u + h + 1u >= nk))
                asm volatile("bar.arrive %0, 384;" ::"r"(1u + b));
            #pragma unroll
            for (uint32_t j = 0; j < 4u; ++j)
                #pragma unroll
                for (uint32_t s = 0; s < 4u; ++s)
                    pd_bs_mma_w8_kb<0>(acc[s * 4u + j], am[s][0][0], am[s][0][1],
                                       am[s][0][2], am[s][0][3], bmj[j][0], bmj[j][1],
                                       sa[s], sbj[j]);
            #pragma unroll
            for (uint32_t j = 0; j < 4u; ++j)
                #pragma unroll
                for (uint32_t s = 0; s < 4u; ++s)
                    pd_bs_mma_w8_kb<1>(acc[s * 4u + j], am[s][1][0], am[s][1][1],
                                       am[s][1][2], am[s][1][3], bmj[j][2], bmj[j][3],
                                       sa[s], sbj[j]);
#else
            #pragma unroll
            for (uint32_t j = 0; j < 4u; ++j) {
                const uint32_t c0 = c0w + j * 8u + 2u * tq;
                #pragma unroll
                for (uint32_t s = 0; s < 4u; ++s) {
                    const uint32_t r0 = i0 + s * 16u + g;
                    #pragma unroll
                    for (uint32_t kb = 0; kb < 2u; ++kb) {
                        const unsigned char* wtail = wp + PD_LIN_DATA;
                        const unsigned char* ysb = ysc + b * 512u + h * 256u;
                        pd_f8_mma_sw(acc[s * 4u + j], am[s][kb][0], am[s][kb][1],
                                     am[s][kb][2], am[s][kb][3], bmj[j][kb * 2u],
                                     bmj[j][kb * 2u + 1u], wtail[r0 * 4u + h * 2u + kb],
                                     wtail[(r0 + 8u) * 4u + h * 2u + kb], ysb[c0 * 2u + kb],
                                     ysb[(c0 + 1u) * 2u + kb]);
                    }
                }
            }
#endif
        }
#if PD_BS_OK
        if (!WIN) asm volatile("bar.arrive %0, 384;" ::"r"(1u + b));
#else
        asm volatile("bar.arrive %0, 384;" ::"r"(1u + b));
#endif
    }

    #pragma unroll
    for (uint32_t j = 0; j < 4u; ++j) {
        const uint32_t c0 = col_base + c0w + j * 8u + 2u * tq;
        #pragma unroll
        for (uint32_t s = 0; s < 4u; ++s) {
            const uint32_t r0 = row_base + i0 + s * 16u + g;
            const uint32_t r8 = r0 + 8u;
            __nv_bfloat16* yh = (__nv_bfloat16*)y;
            if (r0 < out_dim) {
                if (c0 < batch) { if (O16) yh[(size_t)c0 * out_dim + r0] = __float2bfloat16(acc[s * 4u + j][0]); else y[(size_t)c0 * out_dim + r0] = acc[s * 4u + j][0]; }
                if (c0 + 1u < batch) { if (O16) yh[(size_t)(c0 + 1u) * out_dim + r0] = __float2bfloat16(acc[s * 4u + j][1]); else y[(size_t)(c0 + 1u) * out_dim + r0] = acc[s * 4u + j][1]; }
            }
            if (r8 < out_dim) {
                if (c0 < batch) { if (O16) yh[(size_t)c0 * out_dim + r8] = __float2bfloat16(acc[s * 4u + j][2]); else y[(size_t)c0 * out_dim + r8] = acc[s * 4u + j][2]; }
                if (c0 + 1u < batch) { if (O16) yh[(size_t)(c0 + 1u) * out_dim + r8] = __float2bfloat16(acc[s * 4u + j][3]); else y[(size_t)(c0 + 1u) * out_dim + r8] = acc[s * 4u + j][3]; }
            }
        }
    }
#else
    (void)wlin; (void)ymap; (void)xs; (void)y;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}

// ---- kt3: 3-deep ring, the stage-period cut -------------------------------
// The early-arrive A/B falsified slot-free timing as the stage-period term
// (bit-exact, 921 -> 936 us): consumers spin at the MBARRIER (barrier stall
// 3.66/14.7) - with a 2-slot ring each 33 KB TMA gets only ~one stage of
// math to land. Deepening to three K-128 slots fits the 99 KB opt-in only
// with the ysc staging moved out of smem: consumers read x-scales straight
// from xs (L2-resident, ~0.5 MB; prefetched into registers before the phase
// wait so the latency rides under the spin), which also collapses the
// producer to its floor - One warp: bar.sync -> expect_tx -> two TMA
// issues. Layout (99864 B): Y ring [3][16 KB] at 0 (1024-aligned for the
// SW128 image), W boxes [3][16896] at 49152, mbarriers at 99840. Same
// fragments, same per-element K order -> bit-exact vs kt. Immediate
// barrier ids per slot (a register-valued bar id reserves all 16 hw
// barriers - kt64 lesson), 288 threads, late consumer arrive.
#if PD_F8W8_TMA_OK && PD_BS_OK
// kt3 pipeline stages: one half-pair's fragment loads / one half-pair's mma
// block, factored so the rotated loop below can interleave them. Same
// addressing and mma order as kt - the bitwise gate depends on it.
static __device__ __forceinline__ void pd_kt3_ldh(
    const unsigned char* wp, const unsigned char* yp, uint32_t h,
    uint32_t lane, uint32_t i0, uint32_t c0w,
    uint32_t (&am)[4][2][4], uint32_t (&bm)[4][4], uint32_t (&sa)[4]) {
    const uint32_t g = lane >> 2, tq = lane & 3u;
    #pragma unroll
    for (uint32_t s = 0; s < 4u; ++s) {
        const uint32_t rr = i0 + s * 16u + ((lane >> 3) & 1u) * 8u + (lane & 7u);
        #pragma unroll
        for (uint32_t kb = 0; kb < 2u; ++kb) {
            const uint32_t c = h * 4u + kb * 2u + (lane >> 4);
            pd_ldm_x4(am[s][kb], wp + rr * 128u + ((c ^ (rr & 7u)) * 16u));
        }
        const uint32_t r0 = i0 + s * 16u + g;
        const uint32_t rs = (tq & 1u) ? r0 + 8u : r0;
        sa[s] = *(const unsigned short*)(wp + PD_LIN_DATA + rs * 4u + h * 2u);
    }
    #pragma unroll
    for (uint32_t j = 0; j < 4u; ++j) {
        const uint32_t col = c0w + j * 8u + (lane & 7u);
        const uint32_t c = h * 4u + (lane >> 3);
        pd_ldm_x4(bm[j], yp + col * 128u + ((c ^ (col & 7u)) * 16u));
    }
}

static __device__ __forceinline__ void pd_kt3_mma(
    float (&acc)[16][4], const uint32_t (&am)[4][2][4],
    const uint32_t (&bm)[4][4], const uint32_t (&sa)[4],
    const uint32_t (&sb)[4]) {
    #pragma unroll
    for (uint32_t j = 0; j < 4u; ++j)
        #pragma unroll
        for (uint32_t s = 0; s < 4u; ++s)
            pd_bs_mma_w8_kb<0>(acc[s * 4u + j], am[s][0][0], am[s][0][1],
                               am[s][0][2], am[s][0][3], bm[j][0], bm[j][1],
                               sa[s], sb[j]);
    #pragma unroll
    for (uint32_t j = 0; j < 4u; ++j)
        #pragma unroll
        for (uint32_t s = 0; s < 4u; ++s)
            pd_bs_mma_w8_kb<1>(acc[s * 4u + j], am[s][1][0], am[s][1][1],
                               am[s][1][2], am[s][1][3], bm[j][2], bm[j][3],
                               sa[s], sb[j]);
}

// pd_kt3_ldh minus the sa scale-strip loads (addressing otherwise verbatim).
// Consumers: the kt4 scale-free family AND the rowwise (RW) arms below,
// where sa is loop-invariant (pc planes: one exponent per row) and rides in
// registers loaded once from the per-row wse vector.
static __device__ __forceinline__ void pd_kt4a_ldh(
    const unsigned char* wp, const unsigned char* yp, uint32_t h,
    uint32_t lane, uint32_t i0, uint32_t c0w,
    uint32_t (&am)[4][2][4], uint32_t (&bm)[4][4]) {
    #pragma unroll
    for (uint32_t s = 0; s < 4u; ++s) {
        const uint32_t rr = i0 + s * 16u + ((lane >> 3) & 1u) * 8u + (lane & 7u);
        #pragma unroll
        for (uint32_t kb = 0; kb < 2u; ++kb) {
            const uint32_t c = h * 4u + kb * 2u + (lane >> 4);
            pd_ldm_x4(am[s][kb], wp + rr * 128u + ((c ^ (rr & 7u)) * 16u));
        }
    }
    #pragma unroll
    for (uint32_t j = 0; j < 4u; ++j) {
        const uint32_t col = c0w + j * 8u + (lane & 7u);
        const uint32_t c = h * 4u + (lane >> 3);
        pd_ldm_x4(bm[j], yp + col * 128u + ((c ^ (col & 7u)) * 16u));
    }
}

// rowwise sa words for the block-scale mma: pc planes carry one ue8m0
// exponent per row, so the strip words pd_kt3_ldh would read are constant
// across K - resolve them once from wse (per-row bytes; PADDED to the box
// row count, so no bounds branch). Packing matches the strip's 16-bit load:
// both k-block bytes equal.
static __device__ __forceinline__ void pd_rw_sa(
    const unsigned char* __restrict__ wse, uint32_t row_base, uint32_t lane,
    uint32_t i0, uint32_t (&sa_row)[4]) {
    const uint32_t g = lane >> 2, tq = lane & 3u;
    #pragma unroll
    for (uint32_t s = 0; s < 4u; ++s) {
        const uint32_t r0 = i0 + s * 16u + g;
        const uint32_t rs = (tq & 1u) ? r0 + 8u : r0;
        const uint32_t e = wse[row_base + rs];
        sa_row[s] = e | (e << 8);
    }
}
#endif

// ktzf flag lane: one release/acquire word per row
// tile for the serialized-K fused reduction (KF below) - each split's
// epilogue advances tile's word z -> z+1, the last z resets it to 0, so
// the next launch (and every graph replay) starts clean with no memset
// side kernel. Zero-initialized at module load. Contract: one fused-KS
// lin launch in flight at a time - the ktz part buffer already imposes
// exactly this single-stream contract on the unfused path. 512 words
// cover the single-wave gate on any die we target.
__device__ unsigned pd_lin_ktzf_flag[512];

// ktz partial-plane scratch, shared by the strip and rowwise kt launchers
// (grow-only, graph-capture rules - see the allocation site)
static void* pd_lin_ktz_part = nullptr;
static size_t pd_lin_ktz_cap = 0;

// KF (with KS): serialized-K IN-KERNEL reduction - a zero-side-kernel
// combine on our frame. Chains still run fully parallel; only the
// epilogues serialize per tile: z0 stores, z>0 spins for flag==z then ADDS
// into y - the combine's exact left-assoc z-order sum, so the output is
// BITWISE vs partial+combine whenever no split is empty (an empty z>0
// skips its +0.0f, which the combine did pay - that can flip a -0.0; none
// of the gated decode shapes has empty splits). Deletes the combine
// launch and the partial-plane traffic. Single-wave grids only (every
// split of a tile must be resident for the flag chain to advance).
template <bool O16 = false, bool KS = false, bool KF = false, bool RW = false,
          bool SPLIT = false>
__global__ void __launch_bounds__(288, 1) pd_f8_gemm_lin_kt3(
    const unsigned char* __restrict__ wlin, const __grid_constant__ CUtensorMap ymap,
    const unsigned char* __restrict__ xs, float* __restrict__ y,
    uint32_t in_dim, uint32_t out_dim, uint32_t batch,
    const unsigned char* __restrict__ wse = nullptr,
    float* __restrict__ y2 = nullptr, uint32_t ncut = 0u) {
    // SPLIT (q36 DN): kt4<SEG>'s two-buffer epilogue on the kt3
    // frame - f32-out single-chain only; the other arms never fuse.
    static_assert(!(SPLIT && (O16 || KS || KF)), "SPLIT is plain-kt3 only");
#if PD_F8W8_TMA_OK
    // RW (rowwise): pc planes in strip-free 16,384B boxes - the per-32
    // strip was the row exponent repeated (3.03% dead bytes); sa rides in
    // registers from wse instead. Bit-exact vs the strip arm (0 mismatches,
    // -2.55% b=128 / -3.40% b=1792).
    constexpr uint32_t BOX = RW ? PD_LINBS_BOX : PD_LIN_BOX;
    constexpr uint32_t PAIR16 = 16384u;
    extern __shared__ __align__(128) unsigned char pd_lin_sh3[];
    unsigned char* ydat = pd_lin_sh3;                 // 3 slots x 16 KB
    unsigned char* wdat = pd_lin_sh3 + 49152u;        // 3 slots x BOX
    unsigned long long* mb = (unsigned long long*)(pd_lin_sh3 + 49152u + 3u * BOX);

    const uint32_t tid = threadIdx.x;
    const uint32_t n_kb = in_dim >> 5;
    const uint32_t nk = (in_dim + 63u) / 64u;
    const uint32_t nsp = (nk + 1u) >> 1;
    const uint32_t nkboxes = in_dim >> 7;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t nct = batch_pad >> 7;
    const uint32_t tile = blockIdx.x;
    const uint32_t row_base = (tile / nct) * 128u;
    const uint32_t col_base = (tile % nct) * 128u;
    const unsigned char* wboxes = wlin + (size_t)(row_base >> 7) * nkboxes * BOX;

    // K-split arm (KS, "ktz"): blockIdx.z owns the contiguous stage range
    // [sp_lo, sp_hi) and writes an f32 partial plane at z*out_dim*batch; the
    // combine kernel sums the planes (and does the bf16 cast when the caller
    // wanted o16 - KS is only instantiated with O16=false). Under-filled
    // mid-M grids only (see the launcher): wall clock there is the per-CTA
    // stage CHAIN (down 285 us at rows=544 vs 300 at
    // 1024 - M-invariant), so nz slices cut the chain by nz while filling
    // the die. Not bit-exact vs the single-chain kernel (nz partial sums) -
    // the same accepted numeric class as the decode ks combine.
    uint32_t sp_lo = 0u, sp_hi = nsp;
    if (KS) {
        const uint32_t nz = gridDim.z;
        const uint32_t per = (nsp + nz - 1u) / nz;
        sp_lo = blockIdx.z * per;
        sp_hi = sp_lo + per < nsp ? sp_lo + per : nsp;
        if (!KF) y += (size_t)blockIdx.z * out_dim * batch;
        if (sp_lo >= sp_hi) {
            if constexpr (KF) {
                // an empty split still owns a link in the tile's flag
                // chain: z0 zeroes y (the sum of the live chains then
                // starts from 0, like the combine's zeroed plane), z>0
                // adds nothing; both pass the flag. Consumers only - the
                // producer warp has no epilogue role.
                if (tid >= 256u) return;
                unsigned* fp = &pd_lin_ktzf_flag[blockIdx.x];
                if (blockIdx.z != 0u) {
                    unsigned v;
                    do {
                        asm volatile("ld.acquire.gpu.b32 %0, [%1];"
                                     : "=r"(v) : "l"(fp) : "memory");
                    } while (v < blockIdx.z);
                } else {
                    for (uint32_t i = tid; i < 128u * 128u; i += 256u) {
                        const uint32_t r = row_base + (i & 127u);
                        const uint32_t c = col_base + (i >> 7);
                        if (r < out_dim && c < batch) y[(size_t)c * out_dim + r] = 0.f;
                    }
                }
                __threadfence();
                asm volatile("bar.sync 4, 256;");
                if (tid == 0u) {
                    if (blockIdx.z + 1u == gridDim.z)
                        asm volatile("st.relaxed.gpu.b32 [%0], %1;" ::"l"(fp), "r"(0u) : "memory");
                    else
                        asm volatile("st.release.gpu.b32 [%0], %1;" ::"l"(fp), "r"(blockIdx.z + 1u) : "memory");
                }
                return;
            } else {
                // empty tail split still owns its plane region: zero this tile's
                // rows/cols so the combine's sum stays exact
                for (uint32_t i = tid; i < 128u * 128u; i += 288u) {
                    const uint32_t r = row_base + (i & 127u);
                    const uint32_t c = col_base + (i >> 7);
                    if (r < out_dim && c < batch) y[(size_t)c * out_dim + r] = 0.f;
                }
                return;
            }
        }
    }

    if (tid == 0u) {
        const uint32_t m0 = (uint32_t)__cvta_generic_to_shared(mb);
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 32;" ::"r"(m0));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 32;" ::"r"(m0 + 8u));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 32;" ::"r"(m0 + 16u));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    asm volatile("bar.sync 0, 288;");

    if (tid >= 256u) {
        // ---------------- producer: one warp, TMA issue only ----------------
        for (uint32_t sp = sp_lo; sp < sp_hi; ++sp) {
            const uint32_t sq = sp - sp_lo;
            const uint32_t b = sq % 3u;
            if (sq >= 3u) {
                if (b == 0u)      asm volatile("bar.sync 1, 288;");
                else if (b == 1u) asm volatile("bar.sync 2, 288;");
                else              asm volatile("bar.sync 3, 288;");
            }
            const uint32_t m = (uint32_t)__cvta_generic_to_shared(mb) + b * 8u;
            if (tid == 256u) {
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                             ::"r"(m), "r"(BOX + PAIR16));
                const uint32_t wd = (uint32_t)__cvta_generic_to_shared(wdat + b * BOX);
                const uint32_t yd = (uint32_t)__cvta_generic_to_shared(ydat + b * PAIR16);
                asm volatile(
                    "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1], %2, [%3];" ::"r"(wd),
                    "l"(wboxes + (size_t)sp * BOX), "r"(BOX), "r"(m)
                    : "memory");
                const int ck = (int)(sp * 128u);
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3}], [%4];" ::"r"(yd),
                    "l"(&ymap), "r"(ck), "r"((int)col_base), "r"(m)
                    : "memory");
            } else {
                asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(m));
            }
        }
        return;
    }

    // ---------------- consumer warps 0-7 ----------------
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, tq = lane & 3u;
    const uint32_t i0 = (warp & 1u) * 64u;
    const uint32_t c0w = (warp >> 1) * 32u;
    // M-TAIL TILE: in the last batch tile the pad
    // cols are free mma work today - at the serving default (2048 span +
    // 32 riders = 2080 rows) the 17th tile is 75% dead. This warp's whole
    // 32-col group past the real batch => skip its Y/W fragment loads and
    // mma entirely (warp-uniform branch); the ring protocol (phase waits +
    // slot arrives) stays on all 288 threads, and the real cols' K chains
    // are untouched -> bitwise-exact vs the unskipped kernel. Dead warps'
    // stores were already guarded. Single-wave mid-M grids gain nothing
    // (chain-bound - the staircase), the multi-wave default does.
    const bool warp_live = col_base + c0w < batch;

    float acc[16][4] = {};
    uint32_t ph0 = 0u, ph1 = 0u, ph2 = 0u;

#if PD_BS_OK
    // RW: loop-invariant sa words, once from the per-row vector
    uint32_t sa_row[4] = {};
    if (RW && warp_live) pd_rw_sa(wse, row_base, lane, i0, sa_row);
    // Rotated software pipeline (probed: smem-fed mma
    // ceiling 850 TF; the flat loop measured 0.68 PF = 80% of it - the
    // missing fifth is tensor-pipe idle in the ldmatrix windows, because
    // the phase wait releases every warp in lockstep). h=1 fragments load
    // under the h=0 mma, the next slot's h=0 fragments load under the h=1
    // mma (the 3-ring pre-signals that wait), x-scales ride ahead from L2.
    // 288 threads leave a 224-reg budget so both fragment sets fit - kt's
    // 384 threads could not afford this. Same mma order per acc ->
    // bit-exact vs kt. nk is even (host rejects in_dim%128), so every sp
    // has both halves and the flat loop's kt>=nk break is dead.
    // Fragment-pipelining falsified both ways (all bit-exact):
    // cross-stage rotation spilled (3 live sets -> 168 regs + 200 B stack,
    // 2549 us) and within-stage up-front loads measured 940.6 vs 903.3 us -
    // with 2 warps/scheduler the two warps' load windows already interleave
    // with each other's mma, so there was no idle window to recover. The
    // flat loop below is the right shape at this occupancy.
    for (uint32_t sp = sp_lo; sp < sp_hi; ++sp) {
        const uint32_t b = (sp - sp_lo) % 3u;
        // x-scales straight from L2 - issued before the phase wait so the
        // latency rides under the spin; OOB cols clamp the ADDRESS (their
        // accs are never written back)
        uint32_t sbj[2][4];
        if (warp_live) {
            #pragma unroll
            for (uint32_t h = 0; h < 2u; ++h)
                #pragma unroll
                for (uint32_t j = 0; j < 4u; ++j) {
                    const uint32_t col = col_base + c0w + j * 8u + g;
                    const uint32_t ccol = col < batch ? col : (batch - 1u);
                    sbj[h][j] = *(const unsigned short*)(
                        xs + (size_t)ccol * n_kb + (sp * 2u + h) * 2u);
                }
        }
        const uint32_t m = (uint32_t)__cvta_generic_to_shared(mb) + b * 8u;
        const uint32_t ph = (b == 0u) ? ph0 : (b == 1u) ? ph1 : ph2;
        asm volatile(
            "{\n\t.reg .pred P;\n"
            "PD_LINKT3_WAIT_%=:\n\t"
            "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
            "@!P bra PD_LINKT3_WAIT_%=;\n\t}" ::"r"(m), "r"(ph) : "memory");
        if (b == 0u) ph0 ^= 1u; else if (b == 1u) ph1 ^= 1u; else ph2 ^= 1u;

        const unsigned char* wp = wdat + b * BOX;
        const unsigned char* yp = ydat + b * PAIR16;
        if (warp_live) {
            #pragma unroll
            for (uint32_t h = 0; h < 2u; ++h) {
                uint32_t am[4][2][4], bm[4][4];
                if (RW) {
                    pd_kt4a_ldh(wp, yp, h, lane, i0, c0w, am, bm);
                    pd_kt3_mma(acc, am, bm, sa_row, sbj[h]);
                } else {
                    uint32_t sa[4];
                    pd_kt3_ldh(wp, yp, h, lane, i0, c0w, am, bm, sa);
                    pd_kt3_mma(acc, am, bm, sa, sbj[h]);
                }
            }
        }
        // free slot b: in-order issue means the h=1 mma above already
        // consumed every smem read from it (late arrive - early falsified)
        if (b == 0u)      asm volatile("bar.arrive 1, 288;");
        else if (b == 1u) asm volatile("bar.arrive 2, 288;");
        else              asm volatile("bar.arrive 3, 288;");
    }
#else
    // non-block-scale arches (sm_90/100 fatbin passes): flat loop, sw fold
    // with scale bytes read straight from L2 - correctness path, the env
    // arm targets sm_120a
    for (uint32_t sp = sp_lo; sp < sp_hi; ++sp) {
        const uint32_t b = (sp - sp_lo) % 3u;
        const uint32_t m = (uint32_t)__cvta_generic_to_shared(mb) + b * 8u;
        const uint32_t ph = (b == 0u) ? ph0 : (b == 1u) ? ph1 : ph2;
        asm volatile(
            "{\n\t.reg .pred P;\n"
            "PD_LINKT3_WAIT_%=:\n\t"
            "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
            "@!P bra PD_LINKT3_WAIT_%=;\n\t}" ::"r"(m), "r"(ph) : "memory");
        if (b == 0u) ph0 ^= 1u; else if (b == 1u) ph1 ^= 1u; else ph2 ^= 1u;

        const unsigned char* wp = wdat + b * BOX;
        const unsigned char* yp = ydat + b * PAIR16;
        #pragma unroll
        for (uint32_t h = 0; h < 2u; ++h) {
            const uint32_t kt = sp * 2u + h;
            if (kt >= nk) break;
            if (!warp_live) continue; // M-tail: whole 32-col group is pad

            uint32_t am[4][2][4];
            #pragma unroll
            for (uint32_t s = 0; s < 4u; ++s) {
                const uint32_t rr = i0 + s * 16u + ((lane >> 3) & 1u) * 8u + (lane & 7u);
                #pragma unroll
                for (uint32_t kb = 0; kb < 2u; ++kb) {
                    const uint32_t c = h * 4u + kb * 2u + (lane >> 4);
                    pd_ldm_x4(am[s][kb], wp + rr * 128u + ((c ^ (rr & 7u)) * 16u));
                }
            }
            uint32_t bmj[4][4];
            #pragma unroll
            for (uint32_t j = 0; j < 4u; ++j) {
                const uint32_t col = c0w + j * 8u + (lane & 7u);
                const uint32_t c = h * 4u + (lane >> 3);
                pd_ldm_x4(bmj[j], yp + col * 128u + ((c ^ (col & 7u)) * 16u));
            }
            #pragma unroll
            for (uint32_t j = 0; j < 4u; ++j) {
                const uint32_t c0 = c0w + j * 8u + 2u * tq;
                const uint32_t gc0 = col_base + c0 < batch ? col_base + c0 : batch - 1u;
                const uint32_t gc1 = col_base + c0 + 1u < batch ? col_base + c0 + 1u : batch - 1u;
                #pragma unroll
                for (uint32_t s = 0; s < 4u; ++s) {
                    const uint32_t r0 = i0 + s * 16u + g;
                    #pragma unroll
                    for (uint32_t kb = 0; kb < 2u; ++kb) {
                        // RW: strip-free box - the row exponent comes from
                        // wse (correctness path; the env arm targets sm_120a)
                        const unsigned char* wtail = wp + PD_LIN_DATA;
                        const uint32_t se0 = RW ? wse[row_base + r0]
                            : wtail[r0 * 4u + h * 2u + kb];
                        const uint32_t se8 = RW ? wse[row_base + r0 + 8u]
                            : wtail[(r0 + 8u) * 4u + h * 2u + kb];
                        pd_f8_mma_sw(acc[s * 4u + j], am[s][kb][0], am[s][kb][1],
                                     am[s][kb][2], am[s][kb][3], bmj[j][kb * 2u],
                                     bmj[j][kb * 2u + 1u], se0, se8,
                                     xs[(size_t)gc0 * n_kb + kt * 2u + kb],
                                     xs[(size_t)gc1 * n_kb + kt * 2u + kb]);
                    }
                }
            }
        }
        if (b == 0u)      asm volatile("bar.arrive 1, 288;");
        else if (b == 1u) asm volatile("bar.arrive 2, 288;");
        else              asm volatile("bar.arrive 3, 288;");
    }
#endif

    if (KF) {
        // serialized-K epilogue: spin for this tile's turn, z0 stores /
        // z>0 accumulates (the combine's exact left-assoc z-order sum),
        // pass the flag. f32-out only (the launcher keeps o16 on the
        // combine's cast path). Bar id 4, consumers only - the producer
        // warp returned at issue end.
        unsigned* fp = &pd_lin_ktzf_flag[blockIdx.x];
        if (blockIdx.z != 0u) {
            unsigned v;
            do {
                asm volatile("ld.acquire.gpu.b32 %0, [%1];"
                             : "=r"(v) : "l"(fp) : "memory");
            } while (v < blockIdx.z);
        }
        #pragma unroll
        for (uint32_t j = 0; j < 4u; ++j) {
            const uint32_t c0 = col_base + c0w + j * 8u + 2u * tq;
            #pragma unroll
            for (uint32_t s = 0; s < 4u; ++s) {
                const uint32_t r0 = row_base + i0 + s * 16u + g;
                const uint32_t r8 = r0 + 8u;
                if (r0 < out_dim) {
                    if (c0 < batch) { if (blockIdx.z) y[(size_t)c0 * out_dim + r0] += acc[s * 4u + j][0]; else y[(size_t)c0 * out_dim + r0] = acc[s * 4u + j][0]; }
                    if (c0 + 1u < batch) { if (blockIdx.z) y[(size_t)(c0 + 1u) * out_dim + r0] += acc[s * 4u + j][1]; else y[(size_t)(c0 + 1u) * out_dim + r0] = acc[s * 4u + j][1]; }
                }
                if (r8 < out_dim) {
                    if (c0 < batch) { if (blockIdx.z) y[(size_t)c0 * out_dim + r8] += acc[s * 4u + j][2]; else y[(size_t)c0 * out_dim + r8] = acc[s * 4u + j][2]; }
                    if (c0 + 1u < batch) { if (blockIdx.z) y[(size_t)(c0 + 1u) * out_dim + r8] += acc[s * 4u + j][3]; else y[(size_t)(c0 + 1u) * out_dim + r8] = acc[s * 4u + j][3]; }
                }
            }
        }
        __threadfence();
        asm volatile("bar.sync 4, 256;");
        if (tid == 0u) {
            if (blockIdx.z + 1u == gridDim.z)
                asm volatile("st.relaxed.gpu.b32 [%0], %1;" ::"l"(fp), "r"(0u) : "memory");
            else
                asm volatile("st.release.gpu.b32 [%0], %1;" ::"l"(fp), "r"(blockIdx.z + 1u) : "memory");
        }
        return;
    }

    // SPLIT: resolve the destination side once per CTA (ncut is a
    // 128-multiple and row_base is 128-aligned, so a tile never straddles
    // - kt4<SEG>'s exact resolve; per-element store order unchanged, so
    // the merged launch is bit-exact vs the two split launches).
    float* ysp = y;
    uint32_t sp_dim = out_dim, sp_base = 0u;
    if (SPLIT) {
        if (row_base >= ncut) { ysp = y2; sp_dim = out_dim - ncut; sp_base = ncut; }
        else sp_dim = ncut;
    }
    #pragma unroll
    for (uint32_t j = 0; j < 4u; ++j) {
        const uint32_t c0 = col_base + c0w + j * 8u + 2u * tq;
        #pragma unroll
        for (uint32_t s = 0; s < 4u; ++s) {
            const uint32_t r0 = row_base + i0 + s * 16u + g;
            const uint32_t r8 = r0 + 8u;
            __nv_bfloat16* yh = (__nv_bfloat16*)y;
            if (r0 < out_dim) {
                if (c0 < batch) { if (O16) yh[(size_t)c0 * out_dim + r0] = __float2bfloat16(acc[s * 4u + j][0]); else ysp[(size_t)c0 * sp_dim + (r0 - sp_base)] = acc[s * 4u + j][0]; }
                if (c0 + 1u < batch) { if (O16) yh[(size_t)(c0 + 1u) * out_dim + r0] = __float2bfloat16(acc[s * 4u + j][1]); else ysp[(size_t)(c0 + 1u) * sp_dim + (r0 - sp_base)] = acc[s * 4u + j][1]; }
            }
            if (r8 < out_dim) {
                if (c0 < batch) { if (O16) yh[(size_t)c0 * out_dim + r8] = __float2bfloat16(acc[s * 4u + j][2]); else ysp[(size_t)c0 * sp_dim + (r8 - sp_base)] = acc[s * 4u + j][2]; }
                if (c0 + 1u < batch) { if (O16) yh[(size_t)(c0 + 1u) * out_dim + r8] = __float2bfloat16(acc[s * 4u + j][3]); else ysp[(size_t)(c0 + 1u) * sp_dim + (r8 - sp_base)] = acc[s * 4u + j][3]; }
            }
        }
    }
#else
    (void)wlin; (void)ymap; (void)xs; (void)y;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}


// ktd: kt3 with the two operand streams DECOUPLED
// into independent rings - W (the DRAM stream) 4 slots deep, Y 2 slots
// deep, separate mbarriers and separate per-slot named free barriers.
// Fragments, per-element K order, epilogue: kt3 verbatim -> BITWISE vs
// kt3. What changes is TIME. kt3's single 3-slot ring ties W issue to Y
// frees and leaves the producer ~1 stage of real slack, so short decode-M
// chains (10-42 boxes) spend 39-46% of their cycles not streaming
// (54-61% DRAM-busy vs 81% on long chains) - ramp plus the
// ~2.4us per-phase floor (mbarrier round trip + TMA issue on the critical
// path). Here the producer pre-issues W0-W3 at chain start (67.6 KB of
// DRAM flight vs kt3's ~50 KB, none of it gated on Y) and then runs W two
// iterations ahead of Y; every producer wait gates on the consumer stage
// two back, so the free round trip and the TMA issue hide under two full
// stage periods instead of sitting on the period. Y is L2-RESIDENT at
// decode-M (the whole activation plane is <= 2.75 MB, shared by every row
// tile), so 2 slots of Y cover its ~0.5us refill with a stage to spare.
// This is the kt3-frame rebuild of the property that lets cutlass's sm120
// stages-(5,2) pingpong stream ~0.5us per K-128 (deep small stages + a
// producer decoupled from the consumer phase) - study-only, no code
// taken. Decode-M only (PADDOCK_LIN_KTD under the kt3 cc12 gate, batch
// <= 128 = one L2-cheap col tile); mid-M keeps kt3 - its binding term is
// mma issue, not the phase floor.
template <bool O16 = false, bool KS = false>
__global__ void __launch_bounds__(288, 1) pd_f8_gemm_lin_ktd(
    const unsigned char* __restrict__ wlin, const __grid_constant__ CUtensorMap ymap,
    const unsigned char* __restrict__ xs, float* __restrict__ y,
    uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
#if PD_F8W8_TMA_OK && PD_BS_OK
    constexpr uint32_t PAIR16 = 16384u;
    extern __shared__ __align__(128) unsigned char pd_lin_shd[];
    unsigned char* ydat = pd_lin_shd;                  // 2 slots x 16 KB (1024-aligned for SW128)
    unsigned char* wdat = pd_lin_shd + 32768u;         // 4 slots x 16896 (box)
    unsigned long long* mb = (unsigned long long*)(pd_lin_shd + 100352u); // wb[4] then yb[2]

    const uint32_t tid = threadIdx.x;
    const uint32_t n_kb = in_dim >> 5;
    const uint32_t nk = (in_dim + 63u) / 64u;
    const uint32_t nsp = (nk + 1u) >> 1;
    const uint32_t nkboxes = in_dim >> 7;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t nct = batch_pad >> 7;
    const uint32_t tile = blockIdx.x;
    const uint32_t row_base = (tile / nct) * 128u;
    const uint32_t col_base = (tile % nct) * 128u;
    const unsigned char* wboxes = wlin + (size_t)(row_base >> 7) * nkboxes * PD_LIN_BOX;

    // K-split arm: kt3's ktz contract verbatim (contiguous stage range,
    // f32 partial plane per z, empty splits zero their region)
    uint32_t sp_lo = 0u, sp_hi = nsp;
    if (KS) {
        const uint32_t nz = gridDim.z;
        const uint32_t per = (nsp + nz - 1u) / nz;
        sp_lo = blockIdx.z * per;
        sp_hi = sp_lo + per < nsp ? sp_lo + per : nsp;
        y += (size_t)blockIdx.z * out_dim * batch;
        if (sp_lo >= sp_hi) {
            for (uint32_t i = tid; i < 128u * 128u; i += 288u) {
                const uint32_t r = row_base + (i & 127u);
                const uint32_t c = col_base + (i >> 7);
                if (r < out_dim && c < batch) y[(size_t)c * out_dim + r] = 0.f;
            }
            return;
        }
    }

    if (tid == 0u) {
        const uint32_t m0 = (uint32_t)__cvta_generic_to_shared(mb);
        // count 1: only the issuing producer thread arrives (tid 256 on
        // the W barriers, tid 257 on the Y barriers) - the TMA tx count
        // carries the data side
        #pragma unroll
        for (uint32_t i = 0; i < 6u; ++i)
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(m0 + i * 8u));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    asm volatile("bar.sync 0, 288;");

    if (tid >= 256u) {
        // ---- producer: one warp; W runs two iterations ahead of Y ----
        // Iteration sq issues Y(sq) and W(sq+2); both slots were freed by
        // consumer stage sq-2 ((sq+2)&3 == (sq-2)&3, sq&1 == (sq-2)&1), so
        // one wait point per iteration gates both issues and the W ring
        // stays 4 deep the whole chain.
        const uint32_t n = sp_hi - sp_lo;
        const uint32_t m0 = (uint32_t)__cvta_generic_to_shared(mb);
        if (tid == 256u) {
            // prologue: W0/W1 into empty slots, no gate
            #pragma unroll
            for (uint32_t p = 0; p < 2u; ++p)
                if (p < n) {
                    const uint32_t m = m0 + p * 8u;
                    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 16896;" ::"r"(m));
                    const uint32_t wd = (uint32_t)__cvta_generic_to_shared(wdat + p * PD_LIN_BOX);
                    asm volatile(
                        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                        " [%0], [%1], %2, [%3];" ::"r"(wd),
                        "l"(wboxes + (size_t)(sp_lo + p) * PD_LIN_BOX), "r"(PD_LIN_BOX), "r"(m)
                        : "memory");
                }
        }
        for (uint32_t sq = 0; sq < n; ++sq) {
            if (sq >= 2u) {
                // frees of consumer stage sq-2 (immediate bar ids - a
                // register-valued id reserves all 16 hw barriers, kt64
                // lesson): its W slot, then its Y slot
                switch ((sq + 2u) & 3u) {
                case 0u:  asm volatile("bar.sync 1, 288;"); break;
                case 1u:  asm volatile("bar.sync 2, 288;"); break;
                case 2u:  asm volatile("bar.sync 3, 288;"); break;
                default:  asm volatile("bar.sync 4, 288;"); break;
                }
                if (sq & 1u) asm volatile("bar.sync 6, 288;");
                else         asm volatile("bar.sync 5, 288;");
            }
            if (tid == 256u && sq + 2u < n) {
                const uint32_t b = (sq + 2u) & 3u;
                const uint32_t m = m0 + b * 8u;
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 16896;" ::"r"(m));
                const uint32_t wd = (uint32_t)__cvta_generic_to_shared(wdat + b * PD_LIN_BOX);
                asm volatile(
                    "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1], %2, [%3];" ::"r"(wd),
                    "l"(wboxes + (size_t)(sp_lo + sq + 2u) * PD_LIN_BOX), "r"(PD_LIN_BOX), "r"(m)
                    : "memory");
            }
            if (tid == 257u) {
                const uint32_t b = sq & 1u;
                const uint32_t m = m0 + 32u + b * 8u;
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 16384;" ::"r"(m));
                const uint32_t yd = (uint32_t)__cvta_generic_to_shared(ydat + b * PAIR16);
                const int ck = (int)((sp_lo + sq) * 128u);
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3}], [%4];" ::"r"(yd),
                    "l"(&ymap), "r"(ck), "r"((int)col_base), "r"(m)
                    : "memory");
            }
        }
        return;
    }

    // ---------------- consumer warps 0-7: kt3 verbatim except the two
    // decoupled waits (Y first - it ran ahead from L2 - then W, the DRAM
    // gate) and the two per-slot frees ----------------
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, tq = lane & 3u;
    const uint32_t i0 = (warp & 1u) * 64u;
    const uint32_t c0w = (warp >> 1) * 32u;
    const bool warp_live = col_base + c0w < batch;

    float acc[16][4] = {};
    uint32_t phw = 0u, phy = 0u;

    for (uint32_t sp = sp_lo; sp < sp_hi; ++sp) {
        const uint32_t sq = sp - sp_lo;
        const uint32_t bw = sq & 3u, by = sq & 1u;
        uint32_t sbj[2][4];
        if (warp_live) {
            #pragma unroll
            for (uint32_t h = 0; h < 2u; ++h)
                #pragma unroll
                for (uint32_t j = 0; j < 4u; ++j) {
                    const uint32_t col = col_base + c0w + j * 8u + g;
                    const uint32_t ccol = col < batch ? col : (batch - 1u);
                    sbj[h][j] = *(const unsigned short*)(
                        xs + (size_t)ccol * n_kb + (sp * 2u + h) * 2u);
                }
        }
        const uint32_t m0 = (uint32_t)__cvta_generic_to_shared(mb);
        asm volatile(
            "{\n\t.reg .pred P;\n"
            "PD_LINKTD_YW_%=:\n\t"
            "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
            "@!P bra PD_LINKTD_YW_%=;\n\t}" ::"r"(m0 + 32u + by * 8u),
            "r"((phy >> by) & 1u) : "memory");
        phy ^= 1u << by;
        asm volatile(
            "{\n\t.reg .pred P;\n"
            "PD_LINKTD_WW_%=:\n\t"
            "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
            "@!P bra PD_LINKTD_WW_%=;\n\t}" ::"r"(m0 + bw * 8u),
            "r"((phw >> bw) & 1u) : "memory");
        phw ^= 1u << bw;

        const unsigned char* wp = wdat + bw * PD_LIN_BOX;
        const unsigned char* yp = ydat + by * PAIR16;
        if (warp_live) {
            #pragma unroll
            for (uint32_t h = 0; h < 2u; ++h) {
                uint32_t am[4][2][4], bm[4][4], sa[4];
                pd_kt3_ldh(wp, yp, h, lane, i0, c0w, am, bm, sa);
                pd_kt3_mma(acc, am, bm, sa, sbj[h]);
            }
        }
        // per-slot frees: in-order issue means the h=1 mma consumed every
        // smem read from both slots (late arrive - early falsified)
        switch (bw) {
        case 0u:  asm volatile("bar.arrive 1, 288;"); break;
        case 1u:  asm volatile("bar.arrive 2, 288;"); break;
        case 2u:  asm volatile("bar.arrive 3, 288;"); break;
        default:  asm volatile("bar.arrive 4, 288;"); break;
        }
        if (by) asm volatile("bar.arrive 6, 288;");
        else    asm volatile("bar.arrive 5, 288;");
    }

    #pragma unroll
    for (uint32_t j = 0; j < 4u; ++j) {
        const uint32_t c0 = col_base + c0w + j * 8u + 2u * tq;
        #pragma unroll
        for (uint32_t s = 0; s < 4u; ++s) {
            const uint32_t r0 = row_base + i0 + s * 16u + g;
            const uint32_t r8 = r0 + 8u;
            __nv_bfloat16* yh = (__nv_bfloat16*)y;
            if (r0 < out_dim) {
                if (c0 < batch) { if (O16) yh[(size_t)c0 * out_dim + r0] = __float2bfloat16(acc[s * 4u + j][0]); else y[(size_t)c0 * out_dim + r0] = acc[s * 4u + j][0]; }
                if (c0 + 1u < batch) { if (O16) yh[(size_t)(c0 + 1u) * out_dim + r0] = __float2bfloat16(acc[s * 4u + j][1]); else y[(size_t)(c0 + 1u) * out_dim + r0] = acc[s * 4u + j][1]; }
            }
            if (r8 < out_dim) {
                if (c0 < batch) { if (O16) yh[(size_t)c0 * out_dim + r8] = __float2bfloat16(acc[s * 4u + j][2]); else y[(size_t)c0 * out_dim + r8] = acc[s * 4u + j][2]; }
                if (c0 + 1u < batch) { if (O16) yh[(size_t)(c0 + 1u) * out_dim + r8] = __float2bfloat16(acc[s * 4u + j][3]); else y[(size_t)(c0 + 1u) * out_dim + r8] = acc[s * 4u + j][3]; }
            }
        }
    }
#else
    (void)wlin; (void)ymap; (void)xs; (void)y;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}


// kt3g: kt3 with the FUSED geglu + per-32 e4m3/ue8m0 quant epilogue, for gu
// planes repacked by pd_f8w_repack_lin_gui (gate/up pair p at tile rows
// (p>>3)*16+(p&7) and +8). The main loop is kt3 verbatim; the y writeback is
// replaced: a thread's acc entry holds gate ([0]/[1], row r0) and up
// ([2]/[3], row r0+8) of the same pair, so the geglu product is in-register;
// a 32-block of pairs is one warp half's 4 s-entries x 8 g-lanes, amax rides
// 3 shfl_xor (16/8/4 keep the tq group), and the e-pick + SATFINITE cvt
// chain is pd_e4m3_quant4's exactly. y (f32, 308 MB at the churn chunk) is
// never written and the separate geglu2 pass (231 us) disappears.
// Measured (in 5376 / out 43008, RTX PRO 6000): BIT-equal q
// and scales vs kt3 -> pd_quantize_e4m3_geglu2 at b=1792 AND b=128 (max is
// order-free, same f32 acc values); pair time 1637.6 -> 1409.7 us (-13.9%)
// at b=1792, 187.5 -> 168.5 (-10.1%) at b=128 (the e4m3 store is 4x lighter
// than the f32 y write, so the fused kernel beats even the BARE gemm there).
// 160 regs, no spill. Rotated-pipeline consumer only (elected under the
// kt3(cc12) gate); other passes stub. No KS twin: gu grids never enter the
// ktz band (336 row tiles alone exceed 1.5x SMs).
// ACT selects the gate nonlinearity folded into the epilogue (pd_glu_act,
// abi.cuh); PD_ACT_GELU reproduces the kernel byte-for-byte as it shipped.
template <bool RW = false, int ACT = PD_ACT_GELU>
__global__ void __launch_bounds__(288, 1) pd_f8_gemm_lin_kt3g(
    const unsigned char* __restrict__ wlin, const __grid_constant__ CUtensorMap ymap,
    const unsigned char* __restrict__ xs, unsigned char* __restrict__ q,
    unsigned char* __restrict__ qs, uint32_t in_dim, uint32_t out_dim,
    uint32_t batch, const unsigned char* __restrict__ wse = nullptr) {
#if PD_F8W8_TMA_OK && PD_BS_OK
    // RW: strip-free boxes + register sa from wse (see kt3's RW note)
    constexpr uint32_t BOX = RW ? PD_LINBS_BOX : PD_LIN_BOX;
    constexpr uint32_t PAIR16 = 16384u;
    extern __shared__ __align__(128) unsigned char pd_lin_sh3g[];
    unsigned char* ydat = pd_lin_sh3g;
    unsigned char* wdat = pd_lin_sh3g + 49152u;
    unsigned long long* mb = (unsigned long long*)(pd_lin_sh3g + 49152u + 3u * BOX);

    const uint32_t tid = threadIdx.x;
    const uint32_t n_kb = in_dim >> 5;
    const uint32_t nk = (in_dim + 63u) / 64u;
    const uint32_t nsp = (nk + 1u) >> 1;
    const uint32_t nkboxes = in_dim >> 7;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t nct = batch_pad >> 7;
    const uint32_t tile = blockIdx.x;
    const uint32_t row_base = (tile / nct) * 128u;
    const uint32_t col_base = (tile % nct) * 128u;
    const unsigned char* wboxes = wlin + (size_t)(row_base >> 7) * nkboxes * BOX;

    if (tid == 0u) {
        const uint32_t m0 = (uint32_t)__cvta_generic_to_shared(mb);
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 32;" ::"r"(m0));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 32;" ::"r"(m0 + 8u));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 32;" ::"r"(m0 + 16u));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    asm volatile("bar.sync 0, 288;");

    if (tid >= 256u) {
        for (uint32_t sp = 0; sp < nsp; ++sp) {
            const uint32_t b = sp % 3u;
            if (sp >= 3u) {
                if (b == 0u)      asm volatile("bar.sync 1, 288;");
                else if (b == 1u) asm volatile("bar.sync 2, 288;");
                else              asm volatile("bar.sync 3, 288;");
            }
            const uint32_t m = (uint32_t)__cvta_generic_to_shared(mb) + b * 8u;
            if (tid == 256u) {
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                             ::"r"(m), "r"(BOX + PAIR16));
                const uint32_t wd = (uint32_t)__cvta_generic_to_shared(wdat + b * BOX);
                const uint32_t yd = (uint32_t)__cvta_generic_to_shared(ydat + b * PAIR16);
                asm volatile(
                    "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1], %2, [%3];" ::"r"(wd),
                    "l"(wboxes + (size_t)sp * BOX), "r"(BOX), "r"(m)
                    : "memory");
                const int ck = (int)(sp * 128u);
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3}], [%4];" ::"r"(yd),
                    "l"(&ymap), "r"(ck), "r"((int)col_base), "r"(m)
                    : "memory");
            } else {
                asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(m));
            }
        }
        return;
    }

    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, tq = lane & 3u;
    const uint32_t i0 = (warp & 1u) * 64u;
    const uint32_t c0w = (warp >> 1) * 32u;
    const bool warp_live = col_base + c0w < batch;

    float acc[16][4] = {};
    uint32_t ph0 = 0u, ph1 = 0u, ph2 = 0u;
    // RW: loop-invariant sa words, once from the per-row vector
    uint32_t sa_row[4] = {};
    if (RW && warp_live) pd_rw_sa(wse, row_base, lane, i0, sa_row);

    for (uint32_t sp = 0; sp < nsp; ++sp) {
        const uint32_t b = sp % 3u;
        uint32_t sbj[2][4];
        if (warp_live) {
            #pragma unroll
            for (uint32_t h = 0; h < 2u; ++h)
                #pragma unroll
                for (uint32_t j = 0; j < 4u; ++j) {
                    const uint32_t col = col_base + c0w + j * 8u + g;
                    const uint32_t ccol = col < batch ? col : (batch - 1u);
                    sbj[h][j] = *(const unsigned short*)(
                        xs + (size_t)ccol * n_kb + (sp * 2u + h) * 2u);
                }
        }
        const uint32_t m = (uint32_t)__cvta_generic_to_shared(mb) + b * 8u;
        const uint32_t ph = (b == 0u) ? ph0 : (b == 1u) ? ph1 : ph2;
        asm volatile(
            "{\n\t.reg .pred P;\n"
            "PD_LINKT3G_WAIT_%=:\n\t"
            "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
            "@!P bra PD_LINKT3G_WAIT_%=;\n\t}" ::"r"(m), "r"(ph) : "memory");
        if (b == 0u) ph0 ^= 1u; else if (b == 1u) ph1 ^= 1u; else ph2 ^= 1u;

        const unsigned char* wp = wdat + b * BOX;
        const unsigned char* yp = ydat + b * PAIR16;
        if (warp_live) {
            #pragma unroll
            for (uint32_t h = 0; h < 2u; ++h) {
                uint32_t am[4][2][4], bm[4][4];
                if (RW) {
                    pd_kt4a_ldh(wp, yp, h, lane, i0, c0w, am, bm);
                    pd_kt3_mma(acc, am, bm, sa_row, sbj[h]);
                } else {
                    uint32_t sa[4];
                    pd_kt3_ldh(wp, yp, h, lane, i0, c0w, am, bm, sa);
                    pd_kt3_mma(acc, am, bm, sa, sbj[h]);
                }
            }
        }
        if (b == 0u)      asm volatile("bar.arrive 1, 288;");
        else if (b == 1u) asm volatile("bar.arrive 2, 288;");
        else              asm volatile("bar.arrive 3, 288;");
    }

    if (!warp_live) return;
    const uint32_t n_ffk = out_dim >> 1;
    const uint32_t nsb = n_ffk >> 5;
    const uint32_t pb = (row_base >> 1) + (i0 >> 1) + g;
    const uint32_t sblk = (row_base >> 6) + (i0 >> 6);
    #pragma unroll
    for (uint32_t j = 0; j < 4u; ++j) {
        const uint32_t c0 = col_base + c0w + j * 8u + 2u * tq;
        float v[4][2];
        float am0 = 0.f, am1 = 0.f;
        #pragma unroll
        for (uint32_t s = 0; s < 4u; ++s) {
            const float g0 = acc[s * 4u + j][0], g1 = acc[s * 4u + j][1];
            const float u0 = acc[s * 4u + j][2], u1 = acc[s * 4u + j][3];
            const float w0 = pd_glu_act<ACT>(g0) * u0;
            const float w1 = pd_glu_act<ACT>(g1) * u1;
            v[s][0] = w0; v[s][1] = w1;
            am0 = fmaxf(am0, fabsf(w0));
            am1 = fmaxf(am1, fabsf(w1));
        }
        #pragma unroll
        for (uint32_t off = 16; off >= 4; off >>= 1) {
            am0 = fmaxf(am0, __shfl_xor_sync(0xffffffffu, am0, off));
            am1 = fmaxf(am1, __shfl_xor_sync(0xffffffffu, am1, off));
        }
        int e0 = 0, e1 = 0;
        if (am0 > 0.0f) { int ex; float mm = frexpf(am0, &ex); e0 = ex - 9 + (mm > 0.875f ? 1 : 0); }
        if (am1 > 0.0f) { int ex; float mm = frexpf(am1, &ex); e1 = ex - 9 + (mm > 0.875f ? 1 : 0); }
        const float inv0 = ldexpf(1.0f, -e0), inv1 = ldexpf(1.0f, -e1);
        if (c0 < batch) {
            #pragma unroll
            for (uint32_t s = 0; s < 4u; ++s)
                q[(size_t)c0 * n_ffk + pb + s * 8u] = __nv_fp8_e4m3(v[s][0] * inv0).__x;
            if (g == 0u) qs[(size_t)c0 * nsb + sblk] = (unsigned char)(e0 + 127);
        }
        if (c0 + 1u < batch) {
            #pragma unroll
            for (uint32_t s = 0; s < 4u; ++s)
                q[(size_t)(c0 + 1u) * n_ffk + pb + s * 8u] = __nv_fp8_e4m3(v[s][1] * inv1).__x;
            if (g == 0u) qs[(size_t)(c0 + 1u) * nsb + sblk] = (unsigned char)(e1 + 127);
        }
    }
#else
    (void)wlin; (void)ymap; (void)xs; (void)q; (void)qs;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}


// kt4 rung A production twin (prototyped at -15.4% BIT-EQUAL at b=1792
// gu): kt3g with the SCALE-FREE mainloop - plain kind::f8f6f4 mma
// accumulating in place, no sa/sb scale-operand loads; per-token (as, f32
// from the row quantizer) x per-channel (wsg/wsu, f32 from the pc plane's
// ue8m0 exponents) scales applied once in the epilogue before geglu. Serves
// the per-channel plane class (PADDOCK_G4_PC): plane bytes quantized on a
// per-row POW2 grid whose exponent also fills the per-32 strip, so kt3g/
// mma_ks paths on the same plane dequantize identically - one plane, all
// bands consistent. The scale operands cost ~15% issue/LSU on the dependent
// loop (register-fed mma rates are equal - mma_rate); this twin returns it
// at chunk shapes. Elected by pd_f8_gemm_lin_gu_pc at r >= 256.
#if PD_F8W8_TMA_OK && PD_BS_OK
// plain e4m3 mma, f32 in-place accumulate - no block-scale operands
static __device__ __forceinline__ void pd_pd_kt4a_mma1(float d[4], uint32_t a0, uint32_t a1,
                                                 uint32_t a2, uint32_t a3, uint32_t b0,
                                                 uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

static __device__ __forceinline__ void pd_kt4a_mma(
    float (&acc)[16][4], const uint32_t (&am)[4][2][4], const uint32_t (&bm)[4][4]) {
    #pragma unroll
    for (uint32_t j = 0; j < 4u; ++j)
        #pragma unroll
        for (uint32_t s = 0; s < 4u; ++s)
            pd_pd_kt4a_mma1(acc[s * 4u + j], am[s][0][0], am[s][0][1], am[s][0][2],
                      am[s][0][3], bm[j][0], bm[j][1]);
    #pragma unroll
    for (uint32_t j = 0; j < 4u; ++j)
        #pragma unroll
        for (uint32_t s = 0; s < 4u; ++s)
            pd_pd_kt4a_mma1(acc[s * 4u + j], am[s][1][0], am[s][1][1], am[s][1][2],
                      am[s][1][3], bm[j][2], bm[j][3]);
}

// kt4d: block-scale mma with UNIT weight scale (0x7F = 2^0) - the activation
// exponent (sb, per-32 from xs) stays in-loop, the weight scale moves to the
// epilogue. The down twin's compromise: its input is kt4a's fused per-32
// output, and a per-token requant pass would cost more than the sa removal
// saves (the duplication lesson).
static __device__ __forceinline__ void pd_kt4d_mma(
    float (&acc)[16][4], const uint32_t (&am)[4][2][4], const uint32_t (&bm)[4][4],
    const uint32_t (&sb)[4]) {
    #pragma unroll
    for (uint32_t j = 0; j < 4u; ++j)
        #pragma unroll
        for (uint32_t s = 0; s < 4u; ++s)
            pd_bs_mma_w8_kb<0>(acc[s * 4u + j], am[s][0][0], am[s][0][1], am[s][0][2],
                               am[s][0][3], bm[j][0], bm[j][1], 0x7F7Fu, sb[j]);
    #pragma unroll
    for (uint32_t j = 0; j < 4u; ++j)
        #pragma unroll
        for (uint32_t s = 0; s < 4u; ++s)
            pd_bs_mma_w8_kb<1>(acc[s * 4u + j], am[s][1][0], am[s][1][1], am[s][1][2],
                               am[s][1][3], bm[j][2], bm[j][3], 0x7F7Fu, sb[j]);
}

#endif

// kt3g clone, scale-free mainloop; as (per-token) x wsg/wsu (per-channel)
// applied in the epilogue before geglu. RW: strip-free 16,384B boxes - the
// mainloop never read the strip, so the twin is pure byte diet.
template <bool RW = false, int ACT = PD_ACT_GELU>
__global__ void __launch_bounds__(288, 1) pd_f8_gemm_lin_kt4a(
    const unsigned char* __restrict__ wlin, const __grid_constant__ CUtensorMap ymap,
    unsigned char* __restrict__ q, unsigned char* __restrict__ qs,
    const float* __restrict__ as, const float* __restrict__ wsg,
    const float* __restrict__ wsu, uint32_t in_dim, uint32_t out_dim,
    uint32_t batch) {
#if PD_F8W8_TMA_OK && PD_BS_OK
    constexpr uint32_t BOX = RW ? PD_LINBS_BOX : PD_LIN_BOX;
    constexpr uint32_t PAIR16 = 16384u;
    extern __shared__ __align__(128) unsigned char pd_lin_sh4a[];
    unsigned char* ydat = pd_lin_sh4a;
    unsigned char* wdat = pd_lin_sh4a + 49152u;
    unsigned long long* mb = (unsigned long long*)(pd_lin_sh4a + 49152u + 3u * BOX);

    const uint32_t tid = threadIdx.x;
    const uint32_t nk = (in_dim + 63u) / 64u;
    const uint32_t nsp = (nk + 1u) >> 1;
    const uint32_t nkboxes = in_dim >> 7;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t nct = batch_pad >> 7;
    const uint32_t tile = blockIdx.x;
    const uint32_t row_base = (tile / nct) * 128u;
    const uint32_t col_base = (tile % nct) * 128u;
    const unsigned char* wboxes = wlin + (size_t)(row_base >> 7) * nkboxes * BOX;

    if (tid == 0u) {
        const uint32_t m0 = (uint32_t)__cvta_generic_to_shared(mb);
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 32;" ::"r"(m0));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 32;" ::"r"(m0 + 8u));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 32;" ::"r"(m0 + 16u));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    asm volatile("bar.sync 0, 288;");

    if (tid >= 256u) {
        for (uint32_t sp = 0; sp < nsp; ++sp) {
            const uint32_t b = sp % 3u;
            if (sp >= 3u) {
                if (b == 0u)      asm volatile("bar.sync 1, 288;");
                else if (b == 1u) asm volatile("bar.sync 2, 288;");
                else              asm volatile("bar.sync 3, 288;");
            }
            const uint32_t m = (uint32_t)__cvta_generic_to_shared(mb) + b * 8u;
            if (tid == 256u) {
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                             ::"r"(m), "r"(BOX + PAIR16));
                const uint32_t wd = (uint32_t)__cvta_generic_to_shared(wdat + b * BOX);
                const uint32_t yd = (uint32_t)__cvta_generic_to_shared(ydat + b * PAIR16);
                asm volatile(
                    "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1], %2, [%3];" ::"r"(wd),
                    "l"(wboxes + (size_t)sp * BOX), "r"(BOX), "r"(m)
                    : "memory");
                const int ck = (int)(sp * 128u);
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3}], [%4];" ::"r"(yd),
                    "l"(&ymap), "r"(ck), "r"((int)col_base), "r"(m)
                    : "memory");
            } else {
                asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(m));
            }
        }
        return;
    }

    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, tq = lane & 3u;
    const uint32_t i0 = (warp & 1u) * 64u;
    const uint32_t c0w = (warp >> 1) * 32u;
    const bool warp_live = col_base + c0w < batch;

    float acc[16][4] = {};
    uint32_t ph0 = 0u, ph1 = 0u, ph2 = 0u;

    for (uint32_t sp = 0; sp < nsp; ++sp) {
        const uint32_t b = sp % 3u;
        const uint32_t m = (uint32_t)__cvta_generic_to_shared(mb) + b * 8u;
        const uint32_t ph = (b == 0u) ? ph0 : (b == 1u) ? ph1 : ph2;
        asm volatile(
            "{\n\t.reg .pred P;\n"
            "PD_LINKT4A_WAIT_%=:\n\t"
            "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
            "@!P bra PD_LINKT4A_WAIT_%=;\n\t}" ::"r"(m), "r"(ph) : "memory");
        if (b == 0u) ph0 ^= 1u; else if (b == 1u) ph1 ^= 1u; else ph2 ^= 1u;

        const unsigned char* wp = wdat + b * BOX;
        const unsigned char* yp = ydat + b * PAIR16;
        if (warp_live) {
            #pragma unroll
            for (uint32_t h = 0; h < 2u; ++h) {
                uint32_t am[4][2][4], bm[4][4];
                pd_kt4a_ldh(wp, yp, h, lane, i0, c0w, am, bm);
                pd_kt4a_mma(acc, am, bm);
            }
        }
        if (b == 0u)      asm volatile("bar.arrive 1, 288;");
        else if (b == 1u) asm volatile("bar.arrive 2, 288;");
        else              asm volatile("bar.arrive 3, 288;");
    }

    if (!warp_live) return;
    const uint32_t n_ffk = out_dim >> 1;
    const uint32_t nsb = n_ffk >> 5;
    const uint32_t pb = (row_base >> 1) + (i0 >> 1) + g;
    const uint32_t sblk = (row_base >> 6) + (i0 >> 6);
    #pragma unroll
    for (uint32_t j = 0; j < 4u; ++j) {
        const uint32_t c0 = col_base + c0w + j * 8u + 2u * tq;
        const float a0 = c0 < batch ? as[c0] : 1.0f;
        const float a1 = c0 + 1u < batch ? as[c0 + 1u] : 1.0f;
        float v[4][2];
        float am0 = 0.f, am1 = 0.f;
        #pragma unroll
        for (uint32_t s = 0; s < 4u; ++s) {
            const uint32_t ch = pb + s * 8u;
            const float sg = wsg[ch], su = wsu[ch];
            const float g0 = acc[s * 4u + j][0] * sg * a0;
            const float g1 = acc[s * 4u + j][1] * sg * a1;
            const float u0 = acc[s * 4u + j][2] * su * a0;
            const float u1 = acc[s * 4u + j][3] * su * a1;
            const float w0 = pd_glu_act<ACT>(g0) * u0;
            const float w1 = pd_glu_act<ACT>(g1) * u1;
            v[s][0] = w0; v[s][1] = w1;
            am0 = fmaxf(am0, fabsf(w0));
            am1 = fmaxf(am1, fabsf(w1));
        }
        #pragma unroll
        for (uint32_t off = 16; off >= 4; off >>= 1) {
            am0 = fmaxf(am0, __shfl_xor_sync(0xffffffffu, am0, off));
            am1 = fmaxf(am1, __shfl_xor_sync(0xffffffffu, am1, off));
        }
        int e0 = 0, e1 = 0;
        if (am0 > 0.0f) { int ex; float mm = frexpf(am0, &ex); e0 = ex - 9 + (mm > 0.875f ? 1 : 0); }
        if (am1 > 0.0f) { int ex; float mm = frexpf(am1, &ex); e1 = ex - 9 + (mm > 0.875f ? 1 : 0); }
        const float inv0 = ldexpf(1.0f, -e0), inv1 = ldexpf(1.0f, -e1);
        if (c0 < batch) {
            #pragma unroll
            for (uint32_t s = 0; s < 4u; ++s)
                q[(size_t)c0 * n_ffk + pb + s * 8u] = __nv_fp8_e4m3(v[s][0] * inv0).__x;
            if (g == 0u) qs[(size_t)c0 * nsb + sblk] = (unsigned char)(e0 + 127);
        }
        if (c0 + 1u < batch) {
            #pragma unroll
            for (uint32_t s = 0; s < 4u; ++s)
                q[(size_t)(c0 + 1u) * n_ffk + pb + s * 8u] = __nv_fp8_e4m3(v[s][1] * inv1).__x;
            if (g == 0u) qs[(size_t)(c0 + 1u) * nsb + sblk] = (unsigned char)(e1 + 127);
        }
    }
#else
    (void)wlin; (void)ymap; (void)q; (void)qs; (void)as; (void)wsg; (void)wsu;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}

// kt4: the scale-free lin GEMM twin for the qkv/wo chunk classes.
// Mainloop = kt4a verbatim (plain f8f6f4 mma, no scale-operand loads);
// epilogue = kt3's y writeback with as[token] x ws[out-row] applied. Serves
// the pc planes (per-row pow2 grid whose exponent fills the per-32 strip, so
// the w8/lin/gemv consumers of the same plane dequantize identically).
// Elected by pd_f8_gemm_w8_pc at chunk shapes; -2 elsewhere.
// RW: strip-free boxes (mainloop never read the strip - pure byte diet).
// SEG: one launch over the whole fused qkv plane, epilogue
// scattered to the three dense per-projection outputs (y=q / yk / yv at
// q_dim/kv_dim/kv_dim row strides). The segment boundaries are 128-multiples
// and row_base is 128-aligned, so a tile never straddles - the resolve is
// one branch per CTA and the mainloop/accumulate order is untouched
// (bit-exact vs three split launches over the same plane). The point is
// grid width: at admission-M the split grids run 1.5-3 waves at 1 CTA/SM
// and pay a full deep-K chain on each straggler wave (q 571 / kv 564 TF vs
// kt4a's 627 on 16-wave grids); one 128-tile-wide grid pays one tail.
template <bool O16 = false, bool RW = false, bool SEG = false>
__global__ void __launch_bounds__(288, 1) pd_f8_gemm_lin_kt4(
    const unsigned char* __restrict__ wlin, const __grid_constant__ CUtensorMap ymap,
    float* __restrict__ y, const float* __restrict__ as,
    const float* __restrict__ ws, uint32_t in_dim, uint32_t out_dim,
    uint32_t batch, float* __restrict__ yk, float* __restrict__ yv,
    uint32_t q_dim, uint32_t kv_dim) {
#if PD_F8W8_TMA_OK && PD_BS_OK
    constexpr uint32_t BOX = RW ? PD_LINBS_BOX : PD_LIN_BOX;
    constexpr uint32_t PAIR16 = 16384u;
    extern __shared__ __align__(128) unsigned char pd_lin_sh4[];
    unsigned char* ydat = pd_lin_sh4;
    unsigned char* wdat = pd_lin_sh4 + 49152u;
    unsigned long long* mb = (unsigned long long*)(pd_lin_sh4 + 49152u + 3u * BOX);

    const uint32_t tid = threadIdx.x;
    const uint32_t nk = (in_dim + 63u) / 64u;
    const uint32_t nsp = (nk + 1u) >> 1;
    const uint32_t nkboxes = in_dim >> 7;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t nct = batch_pad >> 7;
    const uint32_t tile = blockIdx.x;
    const uint32_t row_base = (tile / nct) * 128u;
    const uint32_t col_base = (tile % nct) * 128u;
    const unsigned char* wboxes = wlin + (size_t)(row_base >> 7) * nkboxes * BOX;

    if (tid == 0u) {
        const uint32_t m0 = (uint32_t)__cvta_generic_to_shared(mb);
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 32;" ::"r"(m0));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 32;" ::"r"(m0 + 8u));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 32;" ::"r"(m0 + 16u));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    asm volatile("bar.sync 0, 288;");

    if (tid >= 256u) {
        for (uint32_t sp = 0; sp < nsp; ++sp) {
            const uint32_t b = sp % 3u;
            if (sp >= 3u) {
                if (b == 0u)      asm volatile("bar.sync 1, 288;");
                else if (b == 1u) asm volatile("bar.sync 2, 288;");
                else              asm volatile("bar.sync 3, 288;");
            }
            const uint32_t m = (uint32_t)__cvta_generic_to_shared(mb) + b * 8u;
            if (tid == 256u) {
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                             ::"r"(m), "r"(BOX + PAIR16));
                const uint32_t wd = (uint32_t)__cvta_generic_to_shared(wdat + b * BOX);
                const uint32_t yd = (uint32_t)__cvta_generic_to_shared(ydat + b * PAIR16);
                asm volatile(
                    "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1], %2, [%3];" ::"r"(wd),
                    "l"(wboxes + (size_t)sp * BOX), "r"(BOX), "r"(m)
                    : "memory");
                const int ck = (int)(sp * 128u);
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3}], [%4];" ::"r"(yd),
                    "l"(&ymap), "r"(ck), "r"((int)col_base), "r"(m)
                    : "memory");
            } else {
                asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(m));
            }
        }
        return;
    }

    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, tq = lane & 3u;
    const uint32_t i0 = (warp & 1u) * 64u;
    const uint32_t c0w = (warp >> 1) * 32u;
    const bool warp_live = col_base + c0w < batch;

    float acc[16][4] = {};
    uint32_t ph0 = 0u, ph1 = 0u, ph2 = 0u;

    for (uint32_t sp = 0; sp < nsp; ++sp) {
        const uint32_t b = sp % 3u;
        const uint32_t m = (uint32_t)__cvta_generic_to_shared(mb) + b * 8u;
        const uint32_t ph = (b == 0u) ? ph0 : (b == 1u) ? ph1 : ph2;
        asm volatile(
            "{\n\t.reg .pred P;\n"
            "PD_LINKT4A_WAIT_%=:\n\t"
            "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
            "@!P bra PD_LINKT4A_WAIT_%=;\n\t}" ::"r"(m), "r"(ph) : "memory");
        if (b == 0u) ph0 ^= 1u; else if (b == 1u) ph1 ^= 1u; else ph2 ^= 1u;

        const unsigned char* wp = wdat + b * BOX;
        const unsigned char* yp = ydat + b * PAIR16;
        if (warp_live) {
            #pragma unroll
            for (uint32_t h = 0; h < 2u; ++h) {
                uint32_t am[4][2][4], bm[4][4];
                pd_kt4a_ldh(wp, yp, h, lane, i0, c0w, am, bm);
                pd_kt4a_mma(acc, am, bm);
            }
        }
        if (b == 0u)      asm volatile("bar.arrive 1, 288;");
        else if (b == 1u) asm volatile("bar.arrive 2, 288;");
        else              asm volatile("bar.arrive 3, 288;");
    }

    if (!warp_live) return;
    // SEG: resolve the destination segment once per CTA (row_base is
    // 128-aligned, q_dim/kv_dim are 128-multiples - no tile straddles).
    // ws stays absolute over the fused plane; only the y target and its
    // row stride change, so the split-launch writes are reproduced exactly.
    float* yseg = y;
    uint32_t seg_dim = out_dim, seg_base = 0u;
    if (SEG) {
        if (row_base >= q_dim + kv_dim) { yseg = yv; seg_dim = kv_dim; seg_base = q_dim + kv_dim; }
        else if (row_base >= q_dim)     { yseg = yk; seg_dim = kv_dim; seg_base = q_dim; }
        else                            { seg_dim = q_dim; }
    }
    #pragma unroll
    for (uint32_t j = 0; j < 4u; ++j) {
        const uint32_t c0 = col_base + c0w + j * 8u + 2u * tq;
        const float a0 = c0 < batch ? as[c0] : 1.0f;
        const float a1 = c0 + 1u < batch ? as[c0 + 1u] : 1.0f;
        #pragma unroll
        for (uint32_t s = 0; s < 4u; ++s) {
            const uint32_t r0 = row_base + i0 + s * 16u + g;
            const uint32_t r8 = r0 + 8u;
            const float w0 = r0 < out_dim ? ws[r0] : 0.0f;
            const float w8s = r8 < out_dim ? ws[r8] : 0.0f;
            const uint32_t sr0 = r0 - seg_base, sr8 = r8 - seg_base;
            __nv_bfloat16* yh = (__nv_bfloat16*)yseg;
            if (r0 < out_dim) {
                if (c0 < batch) { const float v = acc[s * 4u + j][0] * a0 * w0; if (O16) yh[(size_t)c0 * seg_dim + sr0] = __float2bfloat16(v); else yseg[(size_t)c0 * seg_dim + sr0] = v; }
                if (c0 + 1u < batch) { const float v = acc[s * 4u + j][1] * a1 * w0; if (O16) yh[(size_t)(c0 + 1u) * seg_dim + sr0] = __float2bfloat16(v); else yseg[(size_t)(c0 + 1u) * seg_dim + sr0] = v; }
            }
            if (r8 < out_dim) {
                if (c0 < batch) { const float v = acc[s * 4u + j][2] * a0 * w8s; if (O16) yh[(size_t)c0 * seg_dim + sr8] = __float2bfloat16(v); else yseg[(size_t)c0 * seg_dim + sr8] = v; }
                if (c0 + 1u < batch) { const float v = acc[s * 4u + j][3] * a1 * w8s; if (O16) yh[(size_t)(c0 + 1u) * seg_dim + sr8] = __float2bfloat16(v); else yseg[(size_t)(c0 + 1u) * seg_dim + sr8] = v; }
            }
        }
    }

#else
    (void)wlin; (void)ymap; (void)y; (void)as; (void)ws;
    (void)in_dim; (void)out_dim; (void)batch;
    (void)yk; (void)yv; (void)q_dim; (void)kv_dim;
#endif
}

// kt4d: the down twin - weights per-channel (ws epilogue, no sa
// strip loads), activations per-32 IN-LOOP (unit-sfa block-scale mma, sb from
// xs = the fused gu epilogue's own scales; no producer change, no requant
// pass). Half the rung-A machinery removed - sized accordingly.
// RW: strip-free boxes (mainloop never read the strip - pure byte diet).
template <bool O16 = false, bool RW = false>
__global__ void __launch_bounds__(288, 1) pd_f8_gemm_lin_kt4d(
    const unsigned char* __restrict__ wlin, const __grid_constant__ CUtensorMap ymap,
    const unsigned char* __restrict__ xs, float* __restrict__ y,
    const float* __restrict__ ws, uint32_t in_dim, uint32_t out_dim,
    uint32_t batch) {
#if PD_F8W8_TMA_OK && PD_BS_OK
    constexpr uint32_t BOX = RW ? PD_LINBS_BOX : PD_LIN_BOX;
    constexpr uint32_t PAIR16 = 16384u;
    extern __shared__ __align__(128) unsigned char pd_lin_sh4d[];
    unsigned char* ydat = pd_lin_sh4d;
    unsigned char* wdat = pd_lin_sh4d + 49152u;
    unsigned long long* mb = (unsigned long long*)(pd_lin_sh4d + 49152u + 3u * BOX);

    const uint32_t tid = threadIdx.x;
    const uint32_t n_kb = in_dim >> 5;
    const uint32_t nk = (in_dim + 63u) / 64u;
    const uint32_t nsp = (nk + 1u) >> 1;
    const uint32_t nkboxes = in_dim >> 7;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t nct = batch_pad >> 7;
    const uint32_t tile = blockIdx.x;
    const uint32_t row_base = (tile / nct) * 128u;
    const uint32_t col_base = (tile % nct) * 128u;
    const unsigned char* wboxes = wlin + (size_t)(row_base >> 7) * nkboxes * BOX;

    if (tid == 0u) {
        const uint32_t m0 = (uint32_t)__cvta_generic_to_shared(mb);
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 32;" ::"r"(m0));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 32;" ::"r"(m0 + 8u));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 32;" ::"r"(m0 + 16u));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    asm volatile("bar.sync 0, 288;");

    if (tid >= 256u) {
        for (uint32_t sp = 0; sp < nsp; ++sp) {
            const uint32_t b = sp % 3u;
            if (sp >= 3u) {
                if (b == 0u)      asm volatile("bar.sync 1, 288;");
                else if (b == 1u) asm volatile("bar.sync 2, 288;");
                else              asm volatile("bar.sync 3, 288;");
            }
            const uint32_t m = (uint32_t)__cvta_generic_to_shared(mb) + b * 8u;
            if (tid == 256u) {
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                             ::"r"(m), "r"(BOX + PAIR16));
                const uint32_t wd = (uint32_t)__cvta_generic_to_shared(wdat + b * BOX);
                const uint32_t yd = (uint32_t)__cvta_generic_to_shared(ydat + b * PAIR16);
                asm volatile(
                    "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1], %2, [%3];" ::"r"(wd),
                    "l"(wboxes + (size_t)sp * BOX), "r"(BOX), "r"(m)
                    : "memory");
                const int ck = (int)(sp * 128u);
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3}], [%4];" ::"r"(yd),
                    "l"(&ymap), "r"(ck), "r"((int)col_base), "r"(m)
                    : "memory");
            } else {
                asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(m));
            }
        }
        return;
    }

    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, tq = lane & 3u;
    const uint32_t i0 = (warp & 1u) * 64u;
    const uint32_t c0w = (warp >> 1) * 32u;
    const bool warp_live = col_base + c0w < batch;

    float acc[16][4] = {};
    uint32_t ph0 = 0u, ph1 = 0u, ph2 = 0u;

    for (uint32_t sp = 0; sp < nsp; ++sp) {
        const uint32_t b = sp % 3u;
        uint32_t sbj[2][4];
        if (warp_live) {
            #pragma unroll
            for (uint32_t h = 0; h < 2u; ++h)
                #pragma unroll
                for (uint32_t j = 0; j < 4u; ++j) {
                    const uint32_t col = col_base + c0w + j * 8u + g;
                    const uint32_t ccol = col < batch ? col : (batch - 1u);
                    sbj[h][j] = *(const unsigned short*)(
                        xs + (size_t)ccol * n_kb + (sp * 2u + h) * 2u);
                }
        }
        const uint32_t m = (uint32_t)__cvta_generic_to_shared(mb) + b * 8u;
        const uint32_t ph = (b == 0u) ? ph0 : (b == 1u) ? ph1 : ph2;
        asm volatile(
            "{\n\t.reg .pred P;\n"
            "PD_LINKT4A_WAIT_%=:\n\t"
            "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
            "@!P bra PD_LINKT4A_WAIT_%=;\n\t}" ::"r"(m), "r"(ph) : "memory");
        if (b == 0u) ph0 ^= 1u; else if (b == 1u) ph1 ^= 1u; else ph2 ^= 1u;

        const unsigned char* wp = wdat + b * BOX;
        const unsigned char* yp = ydat + b * PAIR16;
        if (warp_live) {
            #pragma unroll
            for (uint32_t h = 0; h < 2u; ++h) {
                uint32_t am[4][2][4], bm[4][4];
                pd_kt4a_ldh(wp, yp, h, lane, i0, c0w, am, bm);
                pd_kt4d_mma(acc, am, bm, sbj[h]);
            }
        }
        if (b == 0u)      asm volatile("bar.arrive 1, 288;");
        else if (b == 1u) asm volatile("bar.arrive 2, 288;");
        else              asm volatile("bar.arrive 3, 288;");
    }

    if (!warp_live) return;
    #pragma unroll
    for (uint32_t j = 0; j < 4u; ++j) {
        const uint32_t c0 = col_base + c0w + j * 8u + 2u * tq;
        #pragma unroll
        for (uint32_t s = 0; s < 4u; ++s) {
            const uint32_t r0 = row_base + i0 + s * 16u + g;
            const uint32_t r8 = r0 + 8u;
            const float w0 = r0 < out_dim ? ws[r0] : 0.0f;
            const float w8s = r8 < out_dim ? ws[r8] : 0.0f;
            __nv_bfloat16* yh = (__nv_bfloat16*)y;
            if (r0 < out_dim) {
                if (c0 < batch) { const float v = acc[s * 4u + j][0] * w0; if (O16) yh[(size_t)c0 * out_dim + r0] = __float2bfloat16(v); else y[(size_t)c0 * out_dim + r0] = v; }
                if (c0 + 1u < batch) { const float v = acc[s * 4u + j][1] * w0; if (O16) yh[(size_t)(c0 + 1u) * out_dim + r0] = __float2bfloat16(v); else y[(size_t)(c0 + 1u) * out_dim + r0] = v; }
            }
            if (r8 < out_dim) {
                if (c0 < batch) { const float v = acc[s * 4u + j][2] * w8s; if (O16) yh[(size_t)c0 * out_dim + r8] = __float2bfloat16(v); else y[(size_t)c0 * out_dim + r8] = v; }
                if (c0 + 1u < batch) { const float v = acc[s * 4u + j][3] * w8s; if (O16) yh[(size_t)(c0 + 1u) * out_dim + r8] = __float2bfloat16(v); else y[(size_t)(c0 + 1u) * out_dim + r8] = v; }
            }
        }
    }

#else
    (void)wlin; (void)ymap; (void)xs; (void)y; (void)ws;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}
// ktz partial-plane combine: y[i] = sum_z part[z][i]. The KS kernel always
// writes f32 partials; the o16 flag does the bf16 cast the fused epilogue
// would have done. No TMA/BS dependence - defined unguarded.
__global__ void pd_lin_ktz_combine_kernel(const float* __restrict__ part,
    float* __restrict__ y, size_t n, uint32_t nz, uint32_t o16) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = part[i];
    for (uint32_t z = 1u; z < nz; ++z) s += part[(size_t)z * n + i];
    if (o16) ((__nv_bfloat16*)y)[i] = __float2bfloat16(s);
    else y[i] = s;
}

// kt3c: kt3 with PERSISTENT TILE CHAINING for the 1.5-3-wave band. The
// target case: gu (out 43008) at r<=128 is
// nt=336 = 1.79 waves at 1 CTA/SM - the busiest SMs run two 42-stage
// chains back-to-back with a ring re-fill between them while the 40-SM
// tail idles half the launch, and the W stream (231 MB, the binding
// resource at this r) loses ~20% of its consumers. One CTA here runs its
// tiles (blockIdx.x, +gridDim.x, ...) on a CONTINUOUS stage counter: the
// 3-slot ring never drains across the tile boundary, chains are balanced
// by construction (launcher picks grid = ceil(nt/tpc) <= 1 wave), and the
// per-tile K order is exactly kt3's -> bit-identical output tile-by-tile
// (no partials, no combine - unlike ktz this is a bitwise-gated arm).
// Rotated-pipeline consumer only: the arm is elected under the kt3(cc12)
// gate where PD_BS_OK holds; other passes get the stub body below.
template <bool O16 = false>
__global__ void __launch_bounds__(288, 1) pd_f8_gemm_lin_kt3c(
    const unsigned char* __restrict__ wlin, const __grid_constant__ CUtensorMap ymap,
    const unsigned char* __restrict__ xs, float* __restrict__ y,
    uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
#if PD_F8W8_TMA_OK && PD_BS_OK
    constexpr uint32_t PAIR16 = 16384u;
    extern __shared__ __align__(128) unsigned char pd_lin_sh3c[];
    unsigned char* ydat = pd_lin_sh3c;                 // 3 slots x 16 KB
    unsigned char* wdat = pd_lin_sh3c + 49152u;        // 3 slots x 16896 (box)
    unsigned long long* mb = (unsigned long long*)(pd_lin_sh3c + 99840u);

    const uint32_t tid = threadIdx.x;
    const uint32_t n_kb = in_dim >> 5;
    const uint32_t nk = (in_dim + 63u) / 64u;
    const uint32_t nsp = (nk + 1u) >> 1;
    const uint32_t nkboxes = in_dim >> 7;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t nct = batch_pad >> 7;
    const uint32_t nrt = out_dim >> 7;
    const uint32_t nt = nrt * nct;

    if (tid == 0u) {
        const uint32_t m0 = (uint32_t)__cvta_generic_to_shared(mb);
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 32;" ::"r"(m0));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 32;" ::"r"(m0 + 8u));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 32;" ::"r"(m0 + 16u));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    asm volatile("bar.sync 0, 288;");

    if (tid >= 256u) {
        // producer: one warp, TMA issue only - q ticks across all this
        // CTA's tiles so the slot parity/ring protocol never resets
        uint32_t q = 0u;
        for (uint32_t tile = blockIdx.x; tile < nt; tile += gridDim.x) {
            const uint32_t row_base = (tile / nct) * 128u;
            const uint32_t col_base = (tile % nct) * 128u;
            const unsigned char* wboxes =
                wlin + (size_t)(row_base >> 7) * nkboxes * PD_LIN_BOX;
            for (uint32_t sp = 0u; sp < nsp; ++sp, ++q) {
                const uint32_t b = q % 3u;
                if (q >= 3u) {
                    if (b == 0u)      asm volatile("bar.sync 1, 288;");
                    else if (b == 1u) asm volatile("bar.sync 2, 288;");
                    else              asm volatile("bar.sync 3, 288;");
                }
                const uint32_t m = (uint32_t)__cvta_generic_to_shared(mb) + b * 8u;
                if (tid == 256u) {
                    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 33280;" ::"r"(m));
                    const uint32_t wd = (uint32_t)__cvta_generic_to_shared(wdat + b * PD_LIN_BOX);
                    asm volatile(
                        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                        " [%0], [%1], %2, [%3];" ::"r"(wd),
                        "l"(wboxes + (size_t)sp * PD_LIN_BOX), "r"(PD_LIN_BOX), "r"(m)
                        : "memory");
                    const uint32_t yd = (uint32_t)__cvta_generic_to_shared(ydat + b * PAIR16);
                    const int ck = (int)(sp * 128u);
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                        " [%0], [%1, {%2, %3}], [%4];" ::"r"(yd),
                        "l"(&ymap), "r"(ck), "r"((int)col_base), "r"(m)
                        : "memory");
                } else {
                    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(m));
                }
            }
        }
        return;
    }

    // consumer warps 0-7 - same geometry/fragments/mma order as kt3; only
    // the outer tile loop and the continuous phase counter differ
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, tq = lane & 3u;
    const uint32_t i0 = (warp & 1u) * 64u;
    const uint32_t c0w = (warp >> 1) * 32u;
    uint32_t ph0 = 0u, ph1 = 0u, ph2 = 0u;
    uint32_t q = 0u;
    for (uint32_t tile = blockIdx.x; tile < nt; tile += gridDim.x) {
        const uint32_t row_base = (tile / nct) * 128u;
        const uint32_t col_base = (tile % nct) * 128u;
        const bool warp_live = col_base + c0w < batch;
        float acc[16][4] = {};
        for (uint32_t sp = 0u; sp < nsp; ++sp, ++q) {
            const uint32_t b = q % 3u;
            uint32_t sbj[2][4];
            if (warp_live) {
                #pragma unroll
                for (uint32_t h = 0; h < 2u; ++h)
                    #pragma unroll
                    for (uint32_t j = 0; j < 4u; ++j) {
                        const uint32_t col = col_base + c0w + j * 8u + g;
                        const uint32_t ccol = col < batch ? col : (batch - 1u);
                        sbj[h][j] = *(const unsigned short*)(
                            xs + (size_t)ccol * n_kb + (sp * 2u + h) * 2u);
                    }
            }
            const uint32_t m = (uint32_t)__cvta_generic_to_shared(mb) + b * 8u;
            const uint32_t ph = (b == 0u) ? ph0 : (b == 1u) ? ph1 : ph2;
            asm volatile(
                "{\n\t.reg .pred P;\n"
                "PD_LINKT3C_WAIT_%=:\n\t"
                "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
                "@!P bra PD_LINKT3C_WAIT_%=;\n\t}" ::"r"(m), "r"(ph) : "memory");
            if (b == 0u) ph0 ^= 1u; else if (b == 1u) ph1 ^= 1u; else ph2 ^= 1u;

            const unsigned char* wp = wdat + b * PD_LIN_BOX;
            const unsigned char* yp = ydat + b * PAIR16;
            if (warp_live) {
                #pragma unroll
                for (uint32_t h = 0; h < 2u; ++h) {
                    uint32_t am[4][2][4], bm[4][4], sa[4];
                    pd_kt3_ldh(wp, yp, h, lane, i0, c0w, am, bm, sa);
                    pd_kt3_mma(acc, am, bm, sa, sbj[h]);
                }
            }
            if (b == 0u)      asm volatile("bar.arrive 1, 288;");
            else if (b == 1u) asm volatile("bar.arrive 2, 288;");
            else              asm volatile("bar.arrive 3, 288;");
        }
        // per-tile epilogue: the producer streams the next tile under these
        // stores (its ring slots were already freed by the arrives above)
        #pragma unroll
        for (uint32_t j = 0; j < 4u; ++j) {
            const uint32_t c0 = col_base + c0w + j * 8u + 2u * tq;
            #pragma unroll
            for (uint32_t s = 0; s < 4u; ++s) {
                const uint32_t r0 = row_base + i0 + s * 16u + g;
                const uint32_t r8 = r0 + 8u;
                __nv_bfloat16* yh = (__nv_bfloat16*)y;
                if (r0 < out_dim) {
                    if (c0 < batch) { if (O16) yh[(size_t)c0 * out_dim + r0] = __float2bfloat16(acc[s * 4u + j][0]); else y[(size_t)c0 * out_dim + r0] = acc[s * 4u + j][0]; }
                    if (c0 + 1u < batch) { if (O16) yh[(size_t)(c0 + 1u) * out_dim + r0] = __float2bfloat16(acc[s * 4u + j][1]); else y[(size_t)(c0 + 1u) * out_dim + r0] = acc[s * 4u + j][1]; }
                }
                if (r8 < out_dim) {
                    if (c0 < batch) { if (O16) yh[(size_t)c0 * out_dim + r8] = __float2bfloat16(acc[s * 4u + j][2]); else y[(size_t)c0 * out_dim + r8] = acc[s * 4u + j][2]; }
                    if (c0 + 1u < batch) { if (O16) yh[(size_t)(c0 + 1u) * out_dim + r8] = __float2bfloat16(acc[s * 4u + j][3]); else y[(size_t)(c0 + 1u) * out_dim + r8] = acc[s * 4u + j][3]; }
                }
            }
        }
    }
#else
    (void)wlin; (void)ymap; (void)xs; (void)y;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}

#if PD_F8W8_TMA_OK && PD_BS_OK
// ktw stage helpers: 16-warp geometry - each warp owns 32 rows x 32 cols,
// so s runs 0..1 and acc is [8][4]. Same addressing pattern as kt/kt3.
static __device__ __forceinline__ void pd_ktw_ldh(
    const unsigned char* wp, const unsigned char* yp, uint32_t h,
    uint32_t lane, uint32_t i0, uint32_t c0w,
    uint32_t (&am)[2][2][4], uint32_t (&bm)[4][4], uint32_t (&sa)[2]) {
    const uint32_t g = lane >> 2, tq = lane & 3u;
    #pragma unroll
    for (uint32_t s = 0; s < 2u; ++s) {
        const uint32_t rr = i0 + s * 16u + ((lane >> 3) & 1u) * 8u + (lane & 7u);
        #pragma unroll
        for (uint32_t kb = 0; kb < 2u; ++kb) {
            const uint32_t c = h * 4u + kb * 2u + (lane >> 4);
            pd_ldm_x4(am[s][kb], wp + rr * 128u + ((c ^ (rr & 7u)) * 16u));
        }
        const uint32_t r0 = i0 + s * 16u + g;
        const uint32_t rs = (tq & 1u) ? r0 + 8u : r0;
        sa[s] = *(const unsigned short*)(wp + PD_LIN_DATA + rs * 4u + h * 2u);
    }
    #pragma unroll
    for (uint32_t j = 0; j < 4u; ++j) {
        const uint32_t col = c0w + j * 8u + (lane & 7u);
        const uint32_t c = h * 4u + (lane >> 3);
        pd_ldm_x4(bm[j], yp + col * 128u + ((c ^ (col & 7u)) * 16u));
    }
}

static __device__ __forceinline__ void pd_ktw_mma(
    float (&acc)[8][4], const uint32_t (&am)[2][2][4],
    const uint32_t (&bm)[4][4], const uint32_t (&sa)[2],
    const uint32_t (&sb)[4]) {
    #pragma unroll
    for (uint32_t j = 0; j < 4u; ++j)
        #pragma unroll
        for (uint32_t s = 0; s < 2u; ++s)
            pd_bs_mma_w8_kb<0>(acc[s * 4u + j], am[s][0][0], am[s][0][1],
                               am[s][0][2], am[s][0][3], bm[j][0], bm[j][1],
                               sa[s], sb[j]);
    #pragma unroll
    for (uint32_t j = 0; j < 4u; ++j)
        #pragma unroll
        for (uint32_t s = 0; s < 2u; ++s)
            pd_bs_mma_w8_kb<1>(acc[s * 4u + j], am[s][1][0], am[s][1][1],
                               am[s][1][2], am[s][1][3], bm[j][2], bm[j][3],
                               sa[s], sb[j]);
}
#endif

// ---- ktw: kt3's ring with SIXTEEN consumer warps -------------------------
// After kt3 the kernel is mma-ISSUE-bound: wait 4.23 + math_pipe 2.66 of
// 10.66 cycles/inst, tensor pipe 76.2%, and only 2.23 active warps per
// scheduler - two consumer warps/scheduler cannot cover the mma dependency
// latency. More CTAs is falsified (kt64: stage-latency, W restage); this
// variant raises ILP inside the CTA instead: 16 consumer warps, each 32
// rows x 32 cols (acc[8][4] = 32 regs), 4 warps/scheduler + producer.
// smem reads/stage rise 96 -> 128 KB (each W row-group and Y col-group now
// read by 4 warps) - acceptable, the smem pipe was ~38% utilized. Ring,
// L2-direct x-scales, one-warp producer, late arrive: all kt3's. Same
// per-element K order (warp partition does not change accumulation order)
// -> bit-exact vs kt. 544 threads caps the budget at ~120 regs/thread -
// acc halves to 32 and the fragment file to 34, so it fits without spill.
template <bool O16 = false>
__global__ void __launch_bounds__(544, 1) pd_f8_gemm_lin_ktw(
    const unsigned char* __restrict__ wlin, const __grid_constant__ CUtensorMap ymap,
    const unsigned char* __restrict__ xs, float* __restrict__ y,
    uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
#if PD_F8W8_TMA_OK
    constexpr uint32_t PAIR16 = 16384u;
    extern __shared__ __align__(128) unsigned char pd_lin_shw[];
    unsigned char* ydat = pd_lin_shw;                 // 3 slots x 16 KB
    unsigned char* wdat = pd_lin_shw + 49152u;        // 3 slots x 16896 (box)
    unsigned long long* mb = (unsigned long long*)(pd_lin_shw + 99840u);

    const uint32_t tid = threadIdx.x;
    const uint32_t n_kb = in_dim >> 5;
    const uint32_t nk = (in_dim + 63u) / 64u;
    const uint32_t nsp = (nk + 1u) >> 1;
    const uint32_t nkboxes = in_dim >> 7;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t nct = batch_pad >> 7;
    const uint32_t tile = blockIdx.x;
    const uint32_t row_base = (tile / nct) * 128u;
    const uint32_t col_base = (tile % nct) * 128u;
    const unsigned char* wboxes = wlin + (size_t)(row_base >> 7) * nkboxes * PD_LIN_BOX;

    if (tid == 0u) {
        const uint32_t m0 = (uint32_t)__cvta_generic_to_shared(mb);
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 32;" ::"r"(m0));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 32;" ::"r"(m0 + 8u));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 32;" ::"r"(m0 + 16u));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    asm volatile("bar.sync 0, 544;");

    if (tid >= 512u) {
        // ---------------- producer: one warp, TMA issue only ----------------
        for (uint32_t sp = 0; sp < nsp; ++sp) {
            const uint32_t b = sp % 3u;
            if (sp >= 3u) {
                if (b == 0u)      asm volatile("bar.sync 1, 544;");
                else if (b == 1u) asm volatile("bar.sync 2, 544;");
                else              asm volatile("bar.sync 3, 544;");
            }
            const uint32_t m = (uint32_t)__cvta_generic_to_shared(mb) + b * 8u;
            if (tid == 512u) {
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 33280;" ::"r"(m));
                const uint32_t wd = (uint32_t)__cvta_generic_to_shared(wdat + b * PD_LIN_BOX);
                const uint32_t yd = (uint32_t)__cvta_generic_to_shared(ydat + b * PAIR16);
                asm volatile(
                    "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1], %2, [%3];" ::"r"(wd),
                    "l"(wboxes + (size_t)sp * PD_LIN_BOX), "r"(PD_LIN_BOX), "r"(m)
                    : "memory");
                const int ck = (int)(sp * 128u);
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3}], [%4];" ::"r"(yd),
                    "l"(&ymap), "r"(ck), "r"((int)col_base), "r"(m)
                    : "memory");
            } else {
                asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(m));
            }
        }
        return;
    }

    // ---------------- consumer warps 0-15 ----------------
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, tq = lane & 3u;
    const uint32_t i0 = (warp & 3u) * 32u;
    const uint32_t c0w = (warp >> 2) * 32u;

    float acc[8][4] = {};
    uint32_t ph0 = 0u, ph1 = 0u, ph2 = 0u;

#if PD_BS_OK
    for (uint32_t sp = 0; sp < nsp; ++sp) {
        const uint32_t b = sp % 3u;
        // x-scales straight from L2, issued before the phase wait
        uint32_t sbj[2][4];
        #pragma unroll
        for (uint32_t h = 0; h < 2u; ++h)
            #pragma unroll
            for (uint32_t j = 0; j < 4u; ++j) {
                const uint32_t col = col_base + c0w + j * 8u + g;
                const uint32_t ccol = col < batch ? col : (batch - 1u);
                sbj[h][j] = *(const unsigned short*)(
                    xs + (size_t)ccol * n_kb + (sp * 2u + h) * 2u);
            }
        const uint32_t m = (uint32_t)__cvta_generic_to_shared(mb) + b * 8u;
        const uint32_t ph = (b == 0u) ? ph0 : (b == 1u) ? ph1 : ph2;
        asm volatile(
            "{\n\t.reg .pred P;\n"
            "PD_LINKTW_WAIT_%=:\n\t"
            "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
            "@!P bra PD_LINKTW_WAIT_%=;\n\t}" ::"r"(m), "r"(ph) : "memory");
        if (b == 0u) ph0 ^= 1u; else if (b == 1u) ph1 ^= 1u; else ph2 ^= 1u;

        const unsigned char* wp = wdat + b * PD_LIN_BOX;
        const unsigned char* yp = ydat + b * PAIR16;
        #pragma unroll
        for (uint32_t h = 0; h < 2u; ++h) {
            uint32_t am[2][2][4], bm[4][4], sa[2];
            pd_ktw_ldh(wp, yp, h, lane, i0, c0w, am, bm, sa);
            pd_ktw_mma(acc, am, bm, sa, sbj[h]);
        }
        if (b == 0u)      asm volatile("bar.arrive 1, 544;");
        else if (b == 1u) asm volatile("bar.arrive 2, 544;");
        else              asm volatile("bar.arrive 3, 544;");
    }
#else
    // non-block-scale arches: flat sw-fold loop, scale bytes from L2
    for (uint32_t sp = 0; sp < nsp; ++sp) {
        const uint32_t b = sp % 3u;
        const uint32_t m = (uint32_t)__cvta_generic_to_shared(mb) + b * 8u;
        const uint32_t ph = (b == 0u) ? ph0 : (b == 1u) ? ph1 : ph2;
        asm volatile(
            "{\n\t.reg .pred P;\n"
            "PD_LINKTW_SWAIT_%=:\n\t"
            "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
            "@!P bra PD_LINKTW_SWAIT_%=;\n\t}" ::"r"(m), "r"(ph) : "memory");
        if (b == 0u) ph0 ^= 1u; else if (b == 1u) ph1 ^= 1u; else ph2 ^= 1u;

        const unsigned char* wp = wdat + b * PD_LIN_BOX;
        const unsigned char* yp = ydat + b * PAIR16;
        #pragma unroll
        for (uint32_t h = 0; h < 2u; ++h) {
            const uint32_t kt = sp * 2u + h;
            if (kt >= nk) break;

            uint32_t am[2][2][4];
            #pragma unroll
            for (uint32_t s = 0; s < 2u; ++s) {
                const uint32_t rr = i0 + s * 16u + ((lane >> 3) & 1u) * 8u + (lane & 7u);
                #pragma unroll
                for (uint32_t kb = 0; kb < 2u; ++kb) {
                    const uint32_t c = h * 4u + kb * 2u + (lane >> 4);
                    pd_ldm_x4(am[s][kb], wp + rr * 128u + ((c ^ (rr & 7u)) * 16u));
                }
            }
            uint32_t bmj[4][4];
            #pragma unroll
            for (uint32_t j = 0; j < 4u; ++j) {
                const uint32_t col = c0w + j * 8u + (lane & 7u);
                const uint32_t c = h * 4u + (lane >> 3);
                pd_ldm_x4(bmj[j], yp + col * 128u + ((c ^ (col & 7u)) * 16u));
            }
            #pragma unroll
            for (uint32_t j = 0; j < 4u; ++j) {
                const uint32_t c0 = c0w + j * 8u + 2u * tq;
                const uint32_t gc0 = col_base + c0 < batch ? col_base + c0 : batch - 1u;
                const uint32_t gc1 = col_base + c0 + 1u < batch ? col_base + c0 + 1u : batch - 1u;
                #pragma unroll
                for (uint32_t s = 0; s < 2u; ++s) {
                    const uint32_t r0 = i0 + s * 16u + g;
                    #pragma unroll
                    for (uint32_t kb = 0; kb < 2u; ++kb) {
                        const unsigned char* wtail = wp + PD_LIN_DATA;
                        pd_f8_mma_sw(acc[s * 4u + j], am[s][kb][0], am[s][kb][1],
                                     am[s][kb][2], am[s][kb][3], bmj[j][kb * 2u],
                                     bmj[j][kb * 2u + 1u], wtail[r0 * 4u + h * 2u + kb],
                                     wtail[(r0 + 8u) * 4u + h * 2u + kb],
                                     xs[(size_t)gc0 * n_kb + kt * 2u + kb],
                                     xs[(size_t)gc1 * n_kb + kt * 2u + kb]);
                    }
                }
            }
        }
        if (b == 0u)      asm volatile("bar.arrive 1, 544;");
        else if (b == 1u) asm volatile("bar.arrive 2, 544;");
        else              asm volatile("bar.arrive 3, 544;");
    }
#endif

    #pragma unroll
    for (uint32_t j = 0; j < 4u; ++j) {
        const uint32_t c0 = col_base + c0w + j * 8u + 2u * tq;
        #pragma unroll
        for (uint32_t s = 0; s < 2u; ++s) {
            const uint32_t r0 = row_base + i0 + s * 16u + g;
            const uint32_t r8 = r0 + 8u;
            __nv_bfloat16* yh = (__nv_bfloat16*)y;
            if (r0 < out_dim) {
                if (c0 < batch) { if (O16) yh[(size_t)c0 * out_dim + r0] = __float2bfloat16(acc[s * 4u + j][0]); else y[(size_t)c0 * out_dim + r0] = acc[s * 4u + j][0]; }
                if (c0 + 1u < batch) { if (O16) yh[(size_t)(c0 + 1u) * out_dim + r0] = __float2bfloat16(acc[s * 4u + j][1]); else y[(size_t)(c0 + 1u) * out_dim + r0] = acc[s * 4u + j][1]; }
            }
            if (r8 < out_dim) {
                if (c0 < batch) { if (O16) yh[(size_t)c0 * out_dim + r8] = __float2bfloat16(acc[s * 4u + j][2]); else y[(size_t)c0 * out_dim + r8] = acc[s * 4u + j][2]; }
                if (c0 + 1u < batch) { if (O16) yh[(size_t)(c0 + 1u) * out_dim + r8] = __float2bfloat16(acc[s * 4u + j][3]); else y[(size_t)(c0 + 1u) * out_dim + r8] = acc[s * 4u + j][3]; }
            }
        }
    }
#else
    (void)wlin; (void)ymap; (void)xs; (void)y;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}


// ---- kt64: the occupancy rebuild of kt (stall anatomy) -------------------
// kt is warp-parallelism-starved, not memory-bound: 140 regs/thread + 67.6 KB
// smem pin it at one 12-warp CTA/SM (No-Eligible 80.2%, wait 4.56 +
// barrier 3.66 of 14.7 cycles/inst; long_scoreboard 0.87 - DRAM hidden).
// Geometry: 64 rows x 128 cols. The first cut (64x64) went 2x on both
// staging streams and the extra W restages came from DRAM - measured 1.5x
// SLOWER. 64x128 restages only Y (X is L2-resident, ~10 MB << L2): W bytes
// stay 1x. smem: Y ring [2][16 KB] + single W half-box 8448 B + scales =
// 42.3 KB -> 2 CTAs/SM, 20 resident warps. The single W buffer serializes
// its 8.4 KB stage against the previous pair's consumers (one bar wait);
// the Y ring keeps the deep stream pipelined. Fragments/swizzle/block-scale
// mma are kt's exactly, same per-element K order -> bit-exact vs kt.
// Immediate barrier ids (a register-valued bar.sync makes ptxas reserve all
// 16 hw barriers -> occupancy 1; the Block-Limit-Barriers cliff).
template <bool O16 = false>
__global__ void __launch_bounds__(320, 2) pd_f8_gemm_lin_kt64(
    const unsigned char* __restrict__ wlin, const __grid_constant__ CUtensorMap ymap,
    const unsigned char* __restrict__ xs, float* __restrict__ y,
    uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
#if PD_F8W8_TMA_OK
    constexpr uint32_t HDAT = 8192u;
    constexpr uint32_t PAIR16 = 16384u; // Y: 128 cols x 128 B
    extern __shared__ __align__(128) unsigned char pd_lin64_sh[];
    unsigned char* ydat = pd_lin64_sh;                 // 2 x 16384
    unsigned char* wdat = pd_lin64_sh + 32768u;        // 1 x 8448
    unsigned char* ysc = pd_lin64_sh + 41216u;         // 2 x 512
    unsigned long long* mb = (unsigned long long*)(pd_lin64_sh + 42240u);

    const uint32_t tid = threadIdx.x;
    const uint32_t n_kb = in_dim >> 5;
    const uint32_t nk = (in_dim + 63u) / 64u;
    const uint32_t nsp = (nk + 1u) >> 1;
    const uint32_t nkboxes = in_dim >> 7;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t nct = batch_pad >> 7;
    const uint32_t tile = blockIdx.x;
    const uint32_t row_base = (tile / nct) * 64u;
    const uint32_t col_base = (tile % nct) * 128u;
    const unsigned char* wboxes = wlin + (size_t)(row_base >> 7) * nkboxes * PD_LIN_BOX;
    const uint32_t rh = (row_base >> 6) & 1u;

    if (tid == 0u) {
        const uint32_t m0 = (uint32_t)__cvta_generic_to_shared(mb);
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 64;" ::"r"(m0));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 64;" ::"r"(m0 + 8u));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    asm volatile("bar.sync 0, 320;");

    if (tid >= 256u) {
        // ------------- producer warps 8-9 (64 lanes, two cols each) ---------
        const uint32_t ptid = tid - 256u;
        unsigned char syr[2][2][2];
        for (uint32_t sp = 0; sp < nsp; ++sp) {
            const uint32_t b = sp % 2u;
            #pragma unroll
            for (uint32_t cc = 0; cc < 2u; ++cc) {
                #pragma unroll
                for (uint32_t h = 0; h < 2u; ++h) {
                    #pragma unroll
                    for (uint32_t kb = 0; kb < 2u; ++kb) {
                        const uint32_t kt = sp * 2u + h;
                        const uint32_t col = col_base + cc * 64u + ptid;
                        const bool yok = col < batch && kt * 2u + kb < n_kb;
                        syr[cc][h][kb] = yok ? xs[(size_t)col * n_kb + kt * 2u + kb] : 0u;
                    }
                }
            }
            // Single W buffer ordering: Y(b) free = pair sp-2 consumed,
            // but W still holds pair sp-1 while its consumers read it - the
            // producer must also wait the other parity (pair sp-1 done)
            // before the W copy lands. This serializes the W stage against
            // the previous pair (the single-buffer price; Y keeps depth 2).
            if (sp >= 2u) {
                if (b == 0u) asm volatile("bar.sync 1, 320;");
                else asm volatile("bar.sync 2, 320;");
            }
            // dedicated W-free barrier (id 3, one cycle per pair): the two
            // parity barriers each complete once per two pairs and cannot
            // also express the depth-1 W wait (first attempt deadlocked)
            if (sp >= 1u) asm volatile("bar.sync 3, 320;");
            #pragma unroll
            for (uint32_t cc = 0; cc < 2u; ++cc)
                #pragma unroll
                for (uint32_t h = 0; h < 2u; ++h)
                    #pragma unroll
                    for (uint32_t kb = 0; kb < 2u; ++kb)
                        ysc[b * 512u + h * 256u + (cc * 64u + ptid) * 2u + kb] = syr[cc][h][kb];
            const uint32_t m = (uint32_t)__cvta_generic_to_shared(mb) + b * 8u;
            if (ptid == 0u) {
                // tx: Y 16384 + W data 8192 + W scales 256 = 24832
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 24832;" ::"r"(m));
                const uint32_t yd = (uint32_t)__cvta_generic_to_shared(ydat + b * PAIR16);
                const int ck = (int)(sp * 128u);
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3}], [%4];" ::"r"(yd),
                    "l"(&ymap), "r"(ck), "r"((int)col_base), "r"(m)
                    : "memory");
                const uint32_t wd = (uint32_t)__cvta_generic_to_shared(wdat);
                const unsigned char* box = wboxes + (size_t)sp * PD_LIN_BOX;
                asm volatile(
                    "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1], %2, [%3];" ::"r"(wd),
                    "l"(box + (size_t)rh * HDAT), "r"(HDAT), "r"(m)
                    : "memory");
                asm volatile(
                    "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1], %2, [%3];" ::"r"(wd + HDAT),
                    "l"(box + PD_LIN_DATA + (size_t)rh * 256u), "r"(256u), "r"(m)
                    : "memory");
            } else {
                asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(m));
            }
        }
        return;
    }

    // ------------- consumer warps 0-7 (8 fragments each) --------------------
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, tq = lane & 3u;
    const uint32_t i0 = (warp & 1u) * 32u;
    const uint32_t c0w = (warp >> 1) * 32u;

    float acc[8][4] = {};
    uint32_t ph0 = 0u, ph1 = 0u;

    for (uint32_t sp = 0; sp < nsp; ++sp) {
        const uint32_t b = sp % 2u;
        const uint32_t m = (uint32_t)__cvta_generic_to_shared(mb) + b * 8u;
        const uint32_t ph = (b == 0u) ? ph0 : ph1;
        asm volatile(
            "{\n\t.reg .pred P;\n"
            "PD_LINKT64_WAIT_%=:\n\t"
            "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
            "@!P bra PD_LINKT64_WAIT_%=;\n\t}" ::"r"(m), "r"(ph) : "memory");
        if (b == 0u) ph0 ^= 1u; else ph1 ^= 1u;

        const unsigned char* wp = wdat;
        const unsigned char* yp = ydat + b * PAIR16;
        #pragma unroll
        for (uint32_t h = 0; h < 2u; ++h) {
            const uint32_t kt = sp * 2u + h;
            if (kt >= nk) break;

            uint32_t am[2][2][4];
#if PD_BS_OK
            uint32_t sa[2];
#endif
            #pragma unroll
            for (uint32_t s = 0; s < 2u; ++s) {
                const uint32_t rr = i0 + s * 16u + ((lane >> 3) & 1u) * 8u + (lane & 7u);
                #pragma unroll
                for (uint32_t kb = 0; kb < 2u; ++kb) {
                    const uint32_t c = h * 4u + kb * 2u + (lane >> 4);
                    pd_ldm_x4(am[s][kb], wp + rr * 128u + ((c ^ (rr & 7u)) * 16u));
                }
#if PD_BS_OK
                const uint32_t r0 = i0 + s * 16u + g;
                const uint32_t rs = (tq & 1u) ? r0 + 8u : r0;
                sa[s] = *(const unsigned short*)(wp + HDAT + rs * 4u + h * 2u);
#endif
            }
            uint32_t bmj[4][4];
#if PD_BS_OK
            uint32_t sbj[4];
#endif
            #pragma unroll
            for (uint32_t j = 0; j < 4u; ++j) {
                const uint32_t col = c0w + j * 8u + (lane & 7u);
                const uint32_t c = h * 4u + (lane >> 3);
                pd_ldm_x4(bmj[j], yp + col * 128u + ((c ^ (col & 7u)) * 16u));
#if PD_BS_OK
                sbj[j] = *(const unsigned short*)(ysc + b * 512u + h * 256u + (c0w + j * 8u + g) * 2u);
#endif
            }
#if PD_BS_OK
            #pragma unroll
            for (uint32_t j = 0; j < 4u; ++j)
                #pragma unroll
                for (uint32_t s = 0; s < 2u; ++s)
                    pd_bs_mma_w8_kb<0>(acc[s * 4u + j], am[s][0][0], am[s][0][1],
                                       am[s][0][2], am[s][0][3], bmj[j][0], bmj[j][1],
                                       sa[s], sbj[j]);
            #pragma unroll
            for (uint32_t j = 0; j < 4u; ++j)
                #pragma unroll
                for (uint32_t s = 0; s < 2u; ++s)
                    pd_bs_mma_w8_kb<1>(acc[s * 4u + j], am[s][1][0], am[s][1][1],
                                       am[s][1][2], am[s][1][3], bmj[j][2], bmj[j][3],
                                       sa[s], sbj[j]);
#else
            #pragma unroll
            for (uint32_t j = 0; j < 4u; ++j) {
                const uint32_t c0 = c0w + j * 8u + 2u * tq;
                #pragma unroll
                for (uint32_t s = 0; s < 2u; ++s) {
                    const uint32_t r0 = i0 + s * 16u + g;
                    #pragma unroll
                    for (uint32_t kb = 0; kb < 2u; ++kb) {
                        const unsigned char* wtail = wp + HDAT;
                        const unsigned char* ysb = ysc + b * 512u + h * 256u;
                        pd_f8_mma_sw(acc[s * 4u + j], am[s][kb][0], am[s][kb][1],
                                     am[s][kb][2], am[s][kb][3], bmj[j][kb * 2u],
                                     bmj[j][kb * 2u + 1u], wtail[r0 * 4u + h * 2u + kb],
                                     wtail[(r0 + 8u) * 4u + h * 2u + kb], ysb[c0 * 2u + kb],
                                     ysb[(c0 + 1u) * 2u + kb]);
                    }
                }
            }
#endif
        }
        if (b == 0u) asm volatile("bar.arrive 1, 320;");
        else asm volatile("bar.arrive 2, 320;");
        asm volatile("bar.arrive 3, 320;");
    }

    #pragma unroll
    for (uint32_t j = 0; j < 4u; ++j) {
        const uint32_t c0 = col_base + c0w + j * 8u + 2u * tq;
        #pragma unroll
        for (uint32_t s = 0; s < 2u; ++s) {
            const uint32_t r0 = row_base + i0 + s * 16u + g;
            const uint32_t r8 = r0 + 8u;
            __nv_bfloat16* yh = (__nv_bfloat16*)y;
            if (r0 < out_dim) {
                if (c0 < batch) { if (O16) yh[(size_t)c0 * out_dim + r0] = __float2bfloat16(acc[s * 4u + j][0]); else y[(size_t)c0 * out_dim + r0] = acc[s * 4u + j][0]; }
                if (c0 + 1u < batch) { if (O16) yh[(size_t)(c0 + 1u) * out_dim + r0] = __float2bfloat16(acc[s * 4u + j][1]); else y[(size_t)(c0 + 1u) * out_dim + r0] = acc[s * 4u + j][1]; }
            }
            if (r8 < out_dim) {
                if (c0 < batch) { if (O16) yh[(size_t)c0 * out_dim + r8] = __float2bfloat16(acc[s * 4u + j][2]); else y[(size_t)c0 * out_dim + r8] = acc[s * 4u + j][2]; }
                if (c0 + 1u < batch) { if (O16) yh[(size_t)(c0 + 1u) * out_dim + r8] = __float2bfloat16(acc[s * 4u + j][3]); else y[(size_t)(c0 + 1u) * out_dim + r8] = acc[s * 4u + j][3]; }
            }
        }
    }
#else
    (void)wlin; (void)ymap; (void)xs; (void)y;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}

// ktp: kt with the producer shrunk 4 warps -> 2 (320 threads). Profiling
// put stalled_barrier 3.66/14.7 warp-cycles largely on parked producer warps
// occupying scheduler slots; kt2 proved 2 producer warps suffice. Same
// consumer code, same fragments, same K order -> bit-identical outputs.
template <bool O16 = false>
__global__ void __launch_bounds__(320, 1) pd_f8_gemm_lin_ktp(
    const unsigned char* __restrict__ wlin, const __grid_constant__ CUtensorMap ymap,
    const unsigned char* __restrict__ xs, float* __restrict__ y,
    uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
#if PD_F8W8_TMA_OK
    constexpr uint32_t PAIR16 = 16384u;
    extern __shared__ __align__(128) unsigned char pd_lin_sh[];
    unsigned char* wdat = pd_lin_sh;                // 2 pairs x 16896 (box)
    unsigned char* ydat = pd_lin_sh + 33792u;       // 2 pairs x 16 KB
    unsigned char* ysc = pd_lin_sh + 66560u;        // 2 pairs x 512 B
    unsigned long long* mb = (unsigned long long*)(pd_lin_sh + 67584u);

    const uint32_t tid = threadIdx.x;
    const uint32_t n_kb = in_dim >> 5;
    const uint32_t nk = (in_dim + 63u) / 64u;
    const uint32_t nsp = (nk + 1u) >> 1;
    const uint32_t nkboxes = in_dim >> 7;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t nct = batch_pad >> 7;
    const uint32_t tile = blockIdx.x;
    const uint32_t row_base = (tile / nct) * 128u;
    const uint32_t col_base = (tile % nct) * 128u;
    const unsigned char* wboxes = wlin + (size_t)(row_base >> 7) * nkboxes * PD_LIN_BOX;

    if (tid == 0u) {
        const uint32_t m0 = (uint32_t)__cvta_generic_to_shared(mb);
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 64;" ::"r"(m0));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 64;" ::"r"(m0 + 8u));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    asm volatile("bar.sync 0, 320;");

    if (tid >= 256u) {
        // ---------------- producer warps 8-11 ----------------
        const uint32_t ptid = tid - 256u;
        unsigned char syr[2][2][2];
        for (uint32_t sp = 0; sp < nsp; ++sp) {
            const uint32_t b = sp % 2u;
            #pragma unroll
            for (uint32_t cc = 0; cc < 2u; ++cc) {
                const uint32_t col = ptid + cc * 64u;
                #pragma unroll
                for (uint32_t h = 0; h < 2u; ++h) {
                    #pragma unroll
                    for (uint32_t kb = 0; kb < 2u; ++kb) {
                        const uint32_t kt = sp * 2u + h;
                        const bool yok = (col_base + col) < batch && kt * 2u + kb < n_kb;
                        syr[cc][h][kb] = yok ? xs[(size_t)(col_base + col) * n_kb + kt * 2u + kb] : 0u;
                    }
                }
            }
            if (sp >= 2u) asm volatile("bar.sync %0, 320;" ::"r"(1u + b));
            #pragma unroll
            for (uint32_t cc = 0; cc < 2u; ++cc) {
                const uint32_t col = ptid + cc * 64u;
                #pragma unroll
                for (uint32_t h = 0; h < 2u; ++h) {
                    #pragma unroll
                    for (uint32_t kb = 0; kb < 2u; ++kb)
                        ysc[b * 512u + h * 256u + col * 2u + kb] = syr[cc][h][kb];
                }
            }
            const uint32_t m = (uint32_t)__cvta_generic_to_shared(mb) + b * 8u;
            if (ptid == 0u) {
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 33280;" ::"r"(m));
                const uint32_t wd = (uint32_t)__cvta_generic_to_shared(wdat + b * PD_LIN_BOX);
                const uint32_t yd = (uint32_t)__cvta_generic_to_shared(ydat + b * PAIR16);
                asm volatile(
                    "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1], %2, [%3];" ::"r"(wd),
                    "l"(wboxes + (size_t)sp * PD_LIN_BOX), "r"(PD_LIN_BOX), "r"(m)
                    : "memory");
                const int ck = (int)(sp * 128u);
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3}], [%4];" ::"r"(yd),
                    "l"(&ymap), "r"(ck), "r"((int)col_base), "r"(m)
                    : "memory");
            } else {
                asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(m));
            }
        }
        return;
    }

    // ---------------- consumer warps 0-7 ----------------
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, tq = lane & 3u;
    const uint32_t i0 = (warp & 1u) * 64u;
    const uint32_t c0w = (warp >> 1) * 32u;

    float acc[16][4] = {};
    uint32_t ph0 = 0u, ph1 = 0u;

    for (uint32_t sp = 0; sp < nsp; ++sp) {
        const uint32_t b = sp % 2u;
        const uint32_t m = (uint32_t)__cvta_generic_to_shared(mb) + b * 8u;
        const uint32_t ph = (b == 0u) ? ph0 : ph1;
        asm volatile(
            "{\n\t.reg .pred P;\n"
            "PD_LINKT_WAIT_%=:\n\t"
            "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
            "@!P bra PD_LINKT_WAIT_%=;\n\t}" ::"r"(m), "r"(ph) : "memory");
        if (b == 0u) ph0 ^= 1u; else ph1 ^= 1u;

        const unsigned char* wp = wdat + b * PD_LIN_BOX;
        const unsigned char* yp = ydat + b * PAIR16;
        #pragma unroll
        for (uint32_t h = 0; h < 2u; ++h) {
            const uint32_t kt = sp * 2u + h;
            if (kt >= nk) break;

            uint32_t am[4][2][4];
#if PD_BS_OK
            uint32_t sa[4];
#endif
            #pragma unroll
            for (uint32_t s = 0; s < 4u; ++s) {
                const uint32_t rr = i0 + s * 16u + ((lane >> 3) & 1u) * 8u + (lane & 7u);
                #pragma unroll
                for (uint32_t kb = 0; kb < 2u; ++kb) {
                    const uint32_t c = h * 4u + kb * 2u + (lane >> 4);
                    pd_ldm_x4(am[s][kb], wp + rr * 128u + ((c ^ (rr & 7u)) * 16u));
                }
#if PD_BS_OK
                const uint32_t r0 = i0 + s * 16u + g;
                const uint32_t rs = (tq & 1u) ? r0 + 8u : r0;
                sa[s] = *(const unsigned short*)(wp + PD_LIN_DATA + rs * 4u + h * 2u);
#endif
            }
            uint32_t bmj[4][4];
#if PD_BS_OK
            uint32_t sbj[4];
#endif
            #pragma unroll
            for (uint32_t j = 0; j < 4u; ++j) {
                const uint32_t col = c0w + j * 8u + (lane & 7u);
                const uint32_t c = h * 4u + (lane >> 3);
                pd_ldm_x4(bmj[j], yp + col * 128u + ((c ^ (col & 7u)) * 16u));
#if PD_BS_OK
                sbj[j] = *(const unsigned short*)(ysc + b * 512u + h * 256u + (c0w + j * 8u + g) * 2u);
#endif
            }
#if PD_BS_OK
            #pragma unroll
            for (uint32_t j = 0; j < 4u; ++j)
                #pragma unroll
                for (uint32_t s = 0; s < 4u; ++s)
                    pd_bs_mma_w8_kb<0>(acc[s * 4u + j], am[s][0][0], am[s][0][1],
                                       am[s][0][2], am[s][0][3], bmj[j][0], bmj[j][1],
                                       sa[s], sbj[j]);
            #pragma unroll
            for (uint32_t j = 0; j < 4u; ++j)
                #pragma unroll
                for (uint32_t s = 0; s < 4u; ++s)
                    pd_bs_mma_w8_kb<1>(acc[s * 4u + j], am[s][1][0], am[s][1][1],
                                       am[s][1][2], am[s][1][3], bmj[j][2], bmj[j][3],
                                       sa[s], sbj[j]);
#else
            #pragma unroll
            for (uint32_t j = 0; j < 4u; ++j) {
                const uint32_t c0 = c0w + j * 8u + 2u * tq;
                #pragma unroll
                for (uint32_t s = 0; s < 4u; ++s) {
                    const uint32_t r0 = i0 + s * 16u + g;
                    #pragma unroll
                    for (uint32_t kb = 0; kb < 2u; ++kb) {
                        const unsigned char* wtail = wp + PD_LIN_DATA;
                        const unsigned char* ysb = ysc + b * 512u + h * 256u;
                        pd_f8_mma_sw(acc[s * 4u + j], am[s][kb][0], am[s][kb][1],
                                     am[s][kb][2], am[s][kb][3], bmj[j][kb * 2u],
                                     bmj[j][kb * 2u + 1u], wtail[r0 * 4u + h * 2u + kb],
                                     wtail[(r0 + 8u) * 4u + h * 2u + kb], ysb[c0 * 2u + kb],
                                     ysb[(c0 + 1u) * 2u + kb]);
                    }
                }
            }
#endif
        }
        asm volatile("bar.arrive %0, 320;" ::"r"(1u + b));
    }

    #pragma unroll
    for (uint32_t j = 0; j < 4u; ++j) {
        const uint32_t c0 = col_base + c0w + j * 8u + 2u * tq;
        #pragma unroll
        for (uint32_t s = 0; s < 4u; ++s) {
            const uint32_t r0 = row_base + i0 + s * 16u + g;
            const uint32_t r8 = r0 + 8u;
            __nv_bfloat16* yh = (__nv_bfloat16*)y;
            if (r0 < out_dim) {
                if (c0 < batch) { if (O16) yh[(size_t)c0 * out_dim + r0] = __float2bfloat16(acc[s * 4u + j][0]); else y[(size_t)c0 * out_dim + r0] = acc[s * 4u + j][0]; }
                if (c0 + 1u < batch) { if (O16) yh[(size_t)(c0 + 1u) * out_dim + r0] = __float2bfloat16(acc[s * 4u + j][1]); else y[(size_t)(c0 + 1u) * out_dim + r0] = acc[s * 4u + j][1]; }
            }
            if (r8 < out_dim) {
                if (c0 < batch) { if (O16) yh[(size_t)c0 * out_dim + r8] = __float2bfloat16(acc[s * 4u + j][2]); else y[(size_t)c0 * out_dim + r8] = acc[s * 4u + j][2]; }
                if (c0 + 1u < batch) { if (O16) yh[(size_t)(c0 + 1u) * out_dim + r8] = __float2bfloat16(acc[s * 4u + j][3]); else y[(size_t)(c0 + 1u) * out_dim + r8] = acc[s * 4u + j][3]; }
            }
        }
    }
#else
    (void)wlin; (void)ymap; (void)xs; (void)y;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}


// ---- kt2: the 256-row prefill tile ---------------------------------------
// The 128x128 kt tile measures at its operand-traffic ceiling: each block
// pulls 675 KB W + 655 KB Y through the TMA/L2 path for 168 MFLOP (126
// FLOP/byte), and multi-K-row calls sustain ~5.3 TB/s aggregate => 0.67 PF,
// shape-independent (isolated bench; block-scale-mma throughput probed at
// full plain-mma rate, so the instruction is not the limit). Doubling the
// out-rows per block - Two lin boxes per K-stage - halves Y bytes per
// output and lifts the traffic ceiling to ~1.3 PF. Fragments, scale reads
// and per-row K order are identical to kt (each 128-row half runs kt's
// exact sequence), so outputs are bit-identical per row. 320 threads: the
// 8 consumer warps carry 128 acc regs/thread (256x128 tile / 256 lanes),
// so the producer shrinks to 2 warps to keep the 204-reg budget. smem:
// 2 stages x 2 boxes + 2 x 16 KB Y + one 512 B ysc = 100864 B (the
// per-block budget is 102400 minus CUDA's 1 KB reserve, so the double
// ysc did not fit; consumers register-load their Y scales at stage entry
// and free ysc via named barrier 3, keeping the pipeline depth intact).
template <bool O16 = false>
__global__ void __launch_bounds__(320, 1) pd_f8_gemm_lin_kt2(
    const unsigned char* __restrict__ wlin, const __grid_constant__ CUtensorMap ymap,
    const unsigned char* __restrict__ xs, float* __restrict__ y,
    uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
#if PD_F8W8_TMA_OK
    constexpr uint32_t PAIR16 = 16384u;
    constexpr uint32_t WSTG = 2u * PD_LIN_BOX;
    extern __shared__ __align__(128) unsigned char pd_lin2_sh[];
    unsigned char* wdat = pd_lin2_sh;                // 2 stages x 2 boxes
    unsigned char* ydat = pd_lin2_sh + 67584u;       // 2 stages x 16 KB
    unsigned char* ysc = pd_lin2_sh + 100352u;       // one 512 B buffer
    __shared__ __align__(8) unsigned long long mb2[2];

    const uint32_t tid = threadIdx.x;
    const uint32_t n_kb = in_dim >> 5;
    const uint32_t nk = (in_dim + 63u) / 64u;
    const uint32_t nsp = (nk + 1u) >> 1;
    const uint32_t nkboxes = in_dim >> 7;
    const uint32_t batch_pad = (batch + 127u) & ~127u;
    const uint32_t nct = batch_pad >> 7;
    const uint32_t tile = blockIdx.x;
    const uint32_t row_base = (tile / nct) * 256u;
    const uint32_t col_base = (tile % nct) * 128u;
    const unsigned char* wb0 = wlin + (size_t)(row_base >> 7) * nkboxes * PD_LIN_BOX;
    const unsigned char* wb1 = wb0 + (size_t)nkboxes * PD_LIN_BOX;

    if (tid == 0u) {
        const uint32_t m0 = (uint32_t)__cvta_generic_to_shared(mb2);
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 64;" ::"r"(m0));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 64;" ::"r"(m0 + 8u));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    asm volatile("bar.sync 0, 320;");

    if (tid >= 256u) {
        // ------------- producer warps 8-9 (64 threads) -------------
        // each lane stages the activation-scale bytes for cols ptid and
        // ptid+64 (kt's 128-lane producer staged one col per lane)
        const uint32_t ptid = tid - 256u;
        unsigned char syr[2][2][2];
        for (uint32_t sp = 0; sp < nsp; ++sp) {
            const uint32_t b = sp % 2u;
            #pragma unroll
            for (uint32_t cc = 0; cc < 2u; ++cc) {
                const uint32_t col = ptid + cc * 64u;
                #pragma unroll
                for (uint32_t h = 0; h < 2u; ++h) {
                    #pragma unroll
                    for (uint32_t kb = 0; kb < 2u; ++kb) {
                        const uint32_t kt = sp * 2u + h;
                        const bool yok = (col_base + col) < batch && kt * 2u + kb < n_kb;
                        syr[cc][h][kb] =
                            yok ? xs[(size_t)(col_base + col) * n_kb + kt * 2u + kb] : 0u;
                    }
                }
            }
            if (sp >= 2u) asm volatile("bar.sync %0, 320;" ::"r"(1u + b));
            // single ysc buffer: wait for the previous stage's consumers to
            // have register-loaded their scales (named barrier 3)
            if (sp >= 1u) asm volatile("bar.sync 3, 320;");
            #pragma unroll
            for (uint32_t cc = 0; cc < 2u; ++cc) {
                const uint32_t col = ptid + cc * 64u;
                #pragma unroll
                for (uint32_t h = 0; h < 2u; ++h)
                    #pragma unroll
                    for (uint32_t kb = 0; kb < 2u; ++kb)
                        ysc[h * 256u + col * 2u + kb] = syr[cc][h][kb];
            }
            const uint32_t m = (uint32_t)__cvta_generic_to_shared(mb2) + b * 8u;
            if (ptid == 0u) {
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 50176;" ::"r"(m));
                const uint32_t w0 = (uint32_t)__cvta_generic_to_shared(wdat + b * WSTG);
                const uint32_t w1 = w0 + PD_LIN_BOX;
                const uint32_t yd = (uint32_t)__cvta_generic_to_shared(ydat + b * PAIR16);
                asm volatile(
                    "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1], %2, [%3];" ::"r"(w0),
                    "l"(wb0 + (size_t)sp * PD_LIN_BOX), "r"(PD_LIN_BOX), "r"(m)
                    : "memory");
                asm volatile(
                    "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1], %2, [%3];" ::"r"(w1),
                    "l"(wb1 + (size_t)sp * PD_LIN_BOX), "r"(PD_LIN_BOX), "r"(m)
                    : "memory");
                const int ck = (int)(sp * 128u);
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3}], [%4];" ::"r"(yd),
                    "l"(&ymap), "r"(ck), "r"((int)col_base), "r"(m)
                    : "memory");
            } else {
                asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(m));
            }
        }
        return;
    }

    // ------------- consumer warps 0-7 -------------
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t g = lane >> 2, tq = lane & 3u;
    const uint32_t i0 = (warp & 1u) * 64u;
    const uint32_t c0w = (warp >> 1) * 32u;

    float acc[2][16][4] = {};
    uint32_t ph0 = 0u, ph1 = 0u;

    for (uint32_t sp = 0; sp < nsp; ++sp) {
        const uint32_t b = sp % 2u;
        const uint32_t m = (uint32_t)__cvta_generic_to_shared(mb2) + b * 8u;
        const uint32_t ph = (b == 0u) ? ph0 : ph1;
        asm volatile(
            "{\n\t.reg .pred P;\n"
            "PD_LINKT2_WAIT_%=:\n\t"
            "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
            "@!P bra PD_LINKT2_WAIT_%=;\n\t}" ::"r"(m), "r"(ph) : "memory");
        if (b == 0u) ph0 ^= 1u; else ph1 ^= 1u;

        const unsigned char* yp = ydat + b * PAIR16;
        // register-load this stage's Y scales for both k-halves up front,
        // then free the single ysc buffer for the producer (barrier 3)
#if PD_BS_OK
        uint32_t sbj2[2][4];
        #pragma unroll
        for (uint32_t h = 0; h < 2u; ++h)
            #pragma unroll
            for (uint32_t j = 0; j < 4u; ++j)
                sbj2[h][j] = *(const unsigned short*)(ysc + h * 256u +
                                                      (c0w + j * 8u + g) * 2u);
#else
        unsigned char ysb2[2][2][8];
        #pragma unroll
        for (uint32_t h = 0; h < 2u; ++h)
            #pragma unroll
            for (uint32_t j = 0; j < 4u; ++j) {
                const uint32_t c0 = c0w + j * 8u + 2u * tq;
                #pragma unroll
                for (uint32_t kb = 0; kb < 2u; ++kb) {
                    ysb2[h][kb][j * 2u] = ysc[h * 256u + c0 * 2u + kb];
                    ysb2[h][kb][j * 2u + 1u] = ysc[h * 256u + (c0 + 1u) * 2u + kb];
                }
            }
#endif
        asm volatile("bar.arrive 3, 320;");
        #pragma unroll
        for (uint32_t h = 0; h < 2u; ++h) {
            const uint32_t kt = sp * 2u + h;
            if (kt >= nk) break;

            uint32_t bmj[4][4];
#if PD_BS_OK
            uint32_t sbj[4];
#endif
            #pragma unroll
            for (uint32_t j = 0; j < 4u; ++j) {
                const uint32_t col = c0w + j * 8u + (lane & 7u);
                const uint32_t c = h * 4u + (lane >> 3);
                pd_ldm_x4(bmj[j], yp + col * 128u + ((c ^ (col & 7u)) * 16u));
#if PD_BS_OK
                sbj[j] = sbj2[h][j];
#endif
            }
            // the two 128-row halves run kt's exact fragment/mma sequence
            // back to back - same K order per output row => bit-identical
            #pragma unroll
            for (uint32_t hf = 0; hf < 2u; ++hf) {
                const unsigned char* wp = wdat + b * WSTG + hf * PD_LIN_BOX;
                // per-s fragment slot (was all four held live: 168-reg cap
                // spilled 80 B; per-element K order kb0->kb1 is unchanged,
                // so this stays bit-identical)
                #pragma unroll
                for (uint32_t s = 0; s < 4u; ++s) {
                    uint32_t am[2][4];
                    const uint32_t rr = i0 + s * 16u + ((lane >> 3) & 1u) * 8u + (lane & 7u);
                    #pragma unroll
                    for (uint32_t kb = 0; kb < 2u; ++kb) {
                        const uint32_t c = h * 4u + kb * 2u + (lane >> 4);
                        pd_ldm_x4(am[kb], wp + rr * 128u + ((c ^ (rr & 7u)) * 16u));
                    }
#if PD_BS_OK
                    const uint32_t r0 = i0 + s * 16u + g;
                    const uint32_t rs = (tq & 1u) ? r0 + 8u : r0;
                    const uint32_t sa = *(const unsigned short*)(wp + PD_LIN_DATA + rs * 4u + h * 2u);
                    #pragma unroll
                    for (uint32_t j = 0; j < 4u; ++j)
                        pd_bs_mma_w8_kb<0>(acc[hf][s * 4u + j], am[0][0], am[0][1],
                                           am[0][2], am[0][3], bmj[j][0], bmj[j][1],
                                           sa, sbj[j]);
                    #pragma unroll
                    for (uint32_t j = 0; j < 4u; ++j)
                        pd_bs_mma_w8_kb<1>(acc[hf][s * 4u + j], am[1][0], am[1][1],
                                           am[1][2], am[1][3], bmj[j][2], bmj[j][3],
                                           sa, sbj[j]);
#else
                    const uint32_t r0s = i0 + s * 16u + g;
                    #pragma unroll
                    for (uint32_t j = 0; j < 4u; ++j) {
                        #pragma unroll
                        for (uint32_t kb = 0; kb < 2u; ++kb) {
                            const unsigned char* wtail = wp + PD_LIN_DATA;
                            pd_f8_mma_sw(acc[hf][s * 4u + j], am[kb][0], am[kb][1],
                                         am[kb][2], am[kb][3], bmj[j][kb * 2u],
                                         bmj[j][kb * 2u + 1u], wtail[r0s * 4u + h * 2u + kb],
                                         wtail[(r0s + 8u) * 4u + h * 2u + kb], ysb2[h][kb][j * 2u],
                                         ysb2[h][kb][j * 2u + 1u]);
                        }
                    }
#endif
                }
            }
        }
        asm volatile("bar.arrive %0, 320;" ::"r"(1u + b));
    }

    #pragma unroll
    for (uint32_t hf = 0; hf < 2u; ++hf) {
        #pragma unroll
        for (uint32_t j = 0; j < 4u; ++j) {
            const uint32_t c0 = col_base + c0w + j * 8u + 2u * tq;
            #pragma unroll
            for (uint32_t s = 0; s < 4u; ++s) {
                const uint32_t r0 = row_base + hf * 128u + i0 + s * 16u + g;
                const uint32_t r8 = r0 + 8u;
                __nv_bfloat16* yh = (__nv_bfloat16*)y;
                if (r0 < out_dim) {
                    if (c0 < batch) { if (O16) yh[(size_t)c0 * out_dim + r0] = __float2bfloat16(acc[hf][s * 4u + j][0]); else y[(size_t)c0 * out_dim + r0] = acc[hf][s * 4u + j][0]; }
                    if (c0 + 1u < batch) { if (O16) yh[(size_t)(c0 + 1u) * out_dim + r0] = __float2bfloat16(acc[hf][s * 4u + j][1]); else y[(size_t)(c0 + 1u) * out_dim + r0] = acc[hf][s * 4u + j][1]; }
                }
                if (r8 < out_dim) {
                    if (c0 < batch) { if (O16) yh[(size_t)c0 * out_dim + r8] = __float2bfloat16(acc[hf][s * 4u + j][2]); else y[(size_t)c0 * out_dim + r8] = acc[hf][s * 4u + j][2]; }
                    if (c0 + 1u < batch) { if (O16) yh[(size_t)(c0 + 1u) * out_dim + r8] = __float2bfloat16(acc[hf][s * 4u + j][3]); else y[(size_t)(c0 + 1u) * out_dim + r8] = acc[hf][s * 4u + j][3]; }
                }
            }
        }
    }
#else
    (void)wlin; (void)ymap; (void)xs; (void)y;
    (void)in_dim; (void)out_dim; (void)batch;
#endif
}

// prefill lin launcher: same contract shape as pd_f8_gemm_w8_o16 but weights
// come as a lin plane (row_off pre-applied by the caller as whole boxes) and
// `o16` selects the bf16 epilogue at runtime. cudaErrorNotSupported when the
// TMA route is off - the engine only builds lin planes after probing this.
PD_EXPORT
int pd_f8_gemm_lin_kt(const void* wlin, const void* xq, const void* xs,
                      void* y, uint32_t in_dim, uint32_t out_dim,
                      uint32_t batch, uint32_t o16, void* stream) {
#ifndef PD_BS_HOST
    (void)wlin; (void)xq; (void)xs; (void)y; (void)in_dim;
    (void)out_dim; (void)batch; (void)o16; (void)stream;
    return cudaErrorNotSupported;
#else
    if (out_dim == 0 || batch == 0) return 0;
    if ((in_dim & 127u) != 0) return cudaErrorInvalidValue;
    static const bool tma = [] {
        int dev = 0, cma = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&cma, cudaDevAttrComputeCapabilityMajor, dev);
        return cma >= 9 && pd_tmap_encode() != nullptr
            && pd_env("PADDOCK_NO_F8W8_TMA") == nullptr;
    }();
    if (!tma) return cudaErrorNotSupported;
    // kt5 decode-M arm (OPT-IN, PADDOCK_LIN_KT5=1; FALSIFIED - kept as the
    // iteration vehicle): routes the wide-decode shapes (batch <= 256) onto
    // the col-tiled decode-band kernel - 64-row half-box tiles, BN=64 col
    // tiles - chasing the 64x64x128-tile sm120 cutlass recipe. At o16=0
    // rows=128, best nz per shape vs kt3+ktz: g4gu 314 vs 180.5us, down
    // 111.2 vs 94.7 (61.7 warm), q +9%, wo +15%, kv +27% - Worse
    // everywhere; kt64 (kt-protocol 64-tile) loses 2x at the same M. That
    // small tile wins on a ~0.5us-period pingpong pipeline, not on tile
    // shape: at our stage period, halving stage bytes just doubles the
    // stage count. The next lever is the period itself (K-256 per mbarrier
    // phase on the kt3 frame - the mid-M lever.
    // Numeric class: ks (scale-fold-in-f32), verified vs the default path
    // at 1e-5-class rel diff.
    static const int kt5 = [] {
        const char* e = pd_env("PADDOCK_LIN_KT5");
        return e ? atoi(e) : 0;
    }();
    if (kt5 && !o16 && batch <= 256u && (out_dim & 63u) == 0u) {
        static int sms5 = 0;
        if (!sms5) {
            int dev = 0;
            cudaGetDevice(&dev);
            cudaDeviceGetAttribute(&sms5, cudaDevAttrMultiProcessorCount, dev);
            if (sms5 <= 0) sms5 = 188;
        }
        CUtensorMap ym5;
        if (pd_tmap_2d_h64(&ym5, xq, in_dim, batch)) {
            const uint32_t tiles = out_dim >> 6;
            const uint32_t colt = (batch + 63u) >> 6;
            const uint32_t nk5 = in_dim >> 7;
            static int nz_env5 = -1;
            if (nz_env5 < 0) {
                const char* e = pd_env("PD_LIN_KT5_NZ");
                nz_env5 = e ? atoi(e) : 0;
            }
            uint32_t nz = 1u;
            if (nz_env5 >= 1) nz = (uint32_t)nz_env5;
            else if (tiles * colt * 10u < (uint32_t)sms5 * 13u)
                nz = ((uint32_t)sms5 * 3u + tiles * colt - 1u) / (tiles * colt);
            if (nz > 4u) nz = 4u;
            const uint32_t max_nz = (nk5 + 1u) / 2u;  // >= 2 boxes per split
            if (nz > max_nz) nz = max_nz;
            if (nz < 1u) nz = 1u;
            float* dst = (float*)y;
            if (nz > 1u) {
                // partial planes: grow-only static scratch under ktz's graph
                // rules - pre-size to the arm's ceiling, never free, never
                // alloc mid-capture (fall to nz=1 there: correct, unsplit)
                static void* part5 = nullptr;
                static size_t cap5 = 0;
                const size_t need = (size_t)out_dim * batch * 4u * nz;
                cudaStreamCaptureStatus ccs = cudaStreamCaptureStatusNone;
                cudaStreamIsCapturing((cudaStream_t)stream, &ccs);
                if (need > cap5 && ccs == cudaStreamCaptureStatusNone) {
                    // sub-wave gate bounds tiles*colt at ~1.5*SMs -> out*colt
                    // <= ~18k rows of 64; 256 batch x 4 splits caps the plane
                    size_t want = (size_t)18048u * 256u * 4u * 4u;
                    if (want < need) want = need;
                    void* np = nullptr;
                    if (cudaMalloc(&np, want) == cudaSuccess) { part5 = np; cap5 = want; }
                }
                if (part5 && need <= cap5) dst = (float*)part5;
                else nz = 1u;
            }
            dim3 g5(tiles, colt, nz);
            pd_f8_gemm_lin_kernel<64u, 2u, 2u, false><<<g5, 256, 0,
                (cudaStream_t)stream>>>(
                (const unsigned char*)wlin, ym5, (const unsigned char*)xs, dst,
                in_dim, out_dim, batch, nullptr);
            if (nz > 1u) {
                const uint32_t n = out_dim * batch;
                pd_q8_0_gemm_mma_ks_combine_kernel<<<(n + 255u) / 256u, 256, 0,
                    (cudaStream_t)stream>>>(
                    (const float*)dst, nullptr, (float*)y, n, nz, out_dim);
            }
            return pd_launch_status();
        }
    }
    const uint32_t smem = 67600u;
    static bool alin = false;
    if (!alin) {
        cudaFuncSetAttribute((const void*)pd_f8_gemm_lin_ktk<false, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        cudaFuncSetAttribute((const void*)pd_f8_gemm_lin_ktk<false, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        alin = true;
    }
    CUtensorMap ym;
    if (!pd_tmap_2d(&ym, xq, in_dim, batch)) return cudaErrorNotSupported;
    const uint32_t bp = (batch + 127u) & ~127u;
    // 256-row kt2 tile (OPT-IN, PADDOCK_LIN_KT2=1): built to halve Y traffic
    // per output, and it did - but measured 0.64 PF vs the 128-tile's 0.67,
    // which FALSIFIES aggregate-L2-traffic as the primary limiter (two
    // kernels, different traffic, same ceiling). Both are stall-bound
    // per stage (~1.5-3us vs ~0.85-1.7us ideal); the next step is stall
    // attribution on the isolated shapes. Kept for that iteration.
    static const bool kt2 = [] {
        const char* e = pd_env("PADDOCK_LIN_KT2");
        return e && atoi(e) != 0;
    }();
    static const bool ktp = [] {
        const char* e = pd_env("PADDOCK_LIN_KTP");
        return e && atoi(e) != 0;
    }();
    if (ktp) {
        static bool alinp = false;
        if (!alinp) {
            cudaFuncSetAttribute((const void*)pd_f8_gemm_lin_ktp<true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
            cudaFuncSetAttribute((const void*)pd_f8_gemm_lin_ktp<false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
            alinp = true;
        }
        const uint32_t ntp = ((out_dim + 127u) / 128u) * (bp >> 7);
        if (o16)
            pd_f8_gemm_lin_ktp<true><<<ntp, 320, smem, (cudaStream_t)stream>>>(
                (const unsigned char*)wlin, ym, (const unsigned char*)xs,
                (float*)y, in_dim, out_dim, batch);
        else
            pd_f8_gemm_lin_ktp<false><<<ntp, 320, smem, (cudaStream_t)stream>>>(
                (const unsigned char*)wlin, ym, (const unsigned char*)xs,
                (float*)y, in_dim, out_dim, batch);
        return pd_launch_status();
    }
    // kt64 (OPT-IN, PADDOCK_LIN_KT64=1): the occupancy rebuild -
    // 64x64 tiles, 33.8 KB smem + 16-reg acc file so 2 CTAs/SM fit (kt is
    // pinned to one by both 140 regs/thread and 67.6 KB smem; the stall
    // anatomy showed warp-parallelism starvation - wait 4.56 + barrier 3.66
    // of 14.7 cycles/inst - with DRAM already hidden). Same fragments, same
    // block-scale mma, same per-element K order -> bit-exact vs kt. The
    // quadrupled grid's X re-reads are L2-resident (X ~10 MB << L2).
    static const bool kt64 = [] {
        const char* e = pd_env("PADDOCK_LIN_KT64");
        return e && atoi(e) != 0;
    }();
    if (kt64 && (out_dim & 127u) == 0u) {
        CUtensorMap ym64;
        if (pd_tmap_2d(&ym64, xq, in_dim, batch)) {
            const uint32_t smem64 = 42256u;
            static bool alin64 = false;
            if (!alin64) {
                // opt into the max-shared carveout or the SM's default budget
                // caps residency at one 33.8 KB CTA (occupancy_limit_
                // shared_mem 1 with registers already allowing 2)
                pd_prefer_max_shared(pd_f8_gemm_lin_kt64<true>);
                pd_prefer_max_shared(pd_f8_gemm_lin_kt64<false>);
                alin64 = true;
            }
            const uint32_t bp64 = (batch + 127u) & ~127u;
            const uint32_t nt64 = (out_dim >> 6) * (bp64 >> 7);
            if (o16)
                pd_f8_gemm_lin_kt64<true><<<nt64, 320, smem64, (cudaStream_t)stream>>>(
                    (const unsigned char*)wlin, ym64, (const unsigned char*)xs,
                    (float*)y, in_dim, out_dim, batch);
            else
                pd_f8_gemm_lin_kt64<false><<<nt64, 320, smem64, (cudaStream_t)stream>>>(
                    (const unsigned char*)wlin, ym64, (const unsigned char*)xs,
                    (float*)y, in_dim, out_dim, batch);
            return pd_launch_status();
        }
    }
    if (kt2 && (out_dim & 255u) == 0u) {
        const uint32_t smem2 = 100864u;
        static bool alin2 = false;
        if (!alin2) {
            cudaFuncSetAttribute((const void*)pd_f8_gemm_lin_kt2<true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem2);
            cudaFuncSetAttribute((const void*)pd_f8_gemm_lin_kt2<false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem2);
            alin2 = true;
        }
        const uint32_t nt2 = (out_dim >> 8) * (bp >> 7);
        if (o16)
            pd_f8_gemm_lin_kt2<true><<<nt2, 320, smem2, (cudaStream_t)stream>>>(
                (const unsigned char*)wlin, ym, (const unsigned char*)xs,
                (float*)y, in_dim, out_dim, batch);
        else
            pd_f8_gemm_lin_kt2<false><<<nt2, 320, smem2, (cudaStream_t)stream>>>(
                (const unsigned char*)wlin, ym, (const unsigned char*)xs,
                (float*)y, in_dim, out_dim, batch);
        return pd_launch_status();
    }
    const uint32_t nt = ((out_dim + 127u) / 128u) * (bp >> 7);
    // ktw (OPT-IN, PADDOCK_LIN_KTW=1): kt3's ring with 16 consumer warps -
    // the mma-issue-ILP experiment (2.23 active warps/scheduler after kt3).
    static const bool ktw = [] {
        const char* e = pd_env("PADDOCK_LIN_KTW");
        return e && atoi(e) != 0;
    }();
    if (ktw) {
        const uint32_t smemw = 99864u;
        static bool alinw2 = false;
        if (!alinw2) {
            cudaFuncSetAttribute((const void*)pd_f8_gemm_lin_ktw<true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemw);
            cudaFuncSetAttribute((const void*)pd_f8_gemm_lin_ktw<false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemw);
            pd_prefer_max_shared(pd_f8_gemm_lin_ktw<true>);
            pd_prefer_max_shared(pd_f8_gemm_lin_ktw<false>);
            alinw2 = true;
        }
        if (o16)
            pd_f8_gemm_lin_ktw<true><<<nt, 544, smemw, (cudaStream_t)stream>>>(
                (const unsigned char*)wlin, ym, (const unsigned char*)xs,
                (float*)y, in_dim, out_dim, batch);
        else
            pd_f8_gemm_lin_ktw<false><<<nt, 544, smemw, (cudaStream_t)stream>>>(
                (const unsigned char*)wlin, ym, (const unsigned char*)xs,
                (float*)y, in_dim, out_dim, batch);
        return pd_launch_status();
    }
    // kt3 (DEFAULT on for the sm_120 class; PADDOCK_LIN_KT3=0
    // reverts): 3-deep ring stage-period cut - ysc staging moved out of smem
    // (consumers read x-scales from L2 directly), producer collapsed to one
    // warp. Bit-exact vs kt, never slower isolated (gate/up -2%, qkv -2%,
    // gatez -1%, down ~par), serve-neutral (+0.2%, in-noise). Other cc keep
    // kt: kt3's non-block-scale arm is unmeasured on real Hopper/DC-Blackwell.
    // See the kernel comment for the falsification trail.
    static const bool kt3 = [] {
        const char* e = pd_env("PADDOCK_LIN_KT3");
        if (e) return atoi(e) != 0;
        int dev = 0, cma = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&cma, cudaDevAttrComputeCapabilityMajor, dev);
        return cma == 12;
    }();
    if (kt3) {
        const uint32_t smem3 = 99864u;
        static bool alin3 = false;
        if (!alin3) {
            cudaFuncSetAttribute((const void*)pd_f8_gemm_lin_kt3<true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem3);
            cudaFuncSetAttribute((const void*)pd_f8_gemm_lin_kt3<false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem3);
            pd_prefer_max_shared(pd_f8_gemm_lin_kt3<true>);
            pd_prefer_max_shared(pd_f8_gemm_lin_kt3<false>);
            alin3 = true;
        }
        // ktd (OPT-IN, PADDOCK_LIN_KTD=1): decoupled dual-ring mainloop -
        // W 4-deep / Y 2-deep, producer two iterations ahead (see the
        // kernel). Decode-M only: batch <= 128 keeps Y one L2-resident col
        // tile. BITWISE vs kt3 (same fragments, same K order), so the gate
        // is a plain A/B. Dispatch (grid, nz) is kt3's verbatim - this is
        // the period lever, not a knob.
        static const bool ktd = [] {
            const char* e = pd_env("PADDOCK_LIN_KTD");
            return e && atoi(e) != 0;
        }();
        const bool ktd_on = ktd && batch <= 128u;
        const uint32_t smemd = 100400u;
        if (ktd_on) {
            static bool alind = false;
            if (!alind) {
                cudaFuncSetAttribute((const void*)pd_f8_gemm_lin_ktd<true>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemd);
                cudaFuncSetAttribute((const void*)pd_f8_gemm_lin_ktd<false>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemd);
                cudaFuncSetAttribute((const void*)pd_f8_gemm_lin_ktd<false, true>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemd);
                pd_prefer_max_shared(pd_f8_gemm_lin_ktd<true>);
                pd_prefer_max_shared(pd_f8_gemm_lin_ktd<false>);
                pd_prefer_max_shared(pd_f8_gemm_lin_ktd<false, true>);
                alind = true;
            }
        }
        // ktz K-split (OPT-IN, PADDOCK_LIN_KTZ=1): mid-M narrow-out shapes
        // (down/o_proj/dnout/gatez at rows <= 1024) run near-single-wave
        // grids whose wall clock is the per-CTA K stage chain, M-invariant
        // (down 285 us at 544 rows vs 300 at 1024).
        // nz contiguous K-slices fill the die and cut the chain by nz;
        // partial f32 planes summed by the combine (decode-ks numeric
        // class, not bitwise vs single-chain - serve-gated).
        // DEFAULT-ON for the sm_120 class: serve-measured on both families
        // and up on every shape (wide-decode r~128 ticks are
        // chain-latency-bound; down nt=84 is dead-centre in the win
        // region), nothing regressed. PADDOCK_LIN_KTZ=0 reverts; other
        // cc stay opt-in (unmeasured there, and kt3 itself is cc12-only by
        // default).
        static const bool ktz = [] {
            const char* e = pd_env("PADDOCK_LIN_KTZ");
            if (e) return atoi(e) != 0;
            int dev = 0, cma = 0;
            cudaGetDevice(&dev);
            cudaDeviceGetAttribute(&cma, cudaDevAttrComputeCapabilityMajor, dev);
            return cma == 12;
        }();
        if (ktz && batch <= 1024u) {
            static int sms = 0;
            if (!sms) {
                int dev = 0;
                cudaGetDevice(&dev);
                cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
            }
            const uint32_t nsp = (((in_dim + 63u) / 64u) + 1u) >> 1;
            // measured-win region only: nt <= 1.5x
            // SMs - the 544/768-row narrow-out shapes (down -17%, o_proj
            // -8%, dnout -9%, gatez -4% at 544). Beyond it (everything at
            // 1024 rows, gatez at 768) the extra waves of half-chains
            // measured worse than the pipelined single-chain grid.
            uint32_t nz = 0;
            if (nt <= (3u * (uint32_t)sms) / 2u) {
                nz = (2u * (uint32_t)sms) / nt;
                if (nz < 2u) nz = 2u;
                if (nz > 4u) nz = 4u;
                // decode-M refit (o16=0 rows=128, all six g4 shapes): the
                // memory-bound decode grids saturate
                // DRAM at ~2/3 of the die - the old 2x-SMs target OVERSPLITS
                // short-K shapes, paying ramp-heavy half-chains for CTAs the
                // stream can't feed (10-21-box chains run 54-61% DRAM
                // vs 81% at 42 boxes). Fill-to-(2*sms/3), never beyond 4:
                // down/wo/kv z4 (unchanged), q z2 (-17%), qkv/qkvg z1
                // (-22%/-13% - and z1 returns those launches to the BITWISE
                // single-chain class, deleting their combine launches).
                // Decode-M only (batch <= 128): the 544-row mid-M band
                // measured the old target's wins (down -17% at 544) - its Y
                // traffic regime is different; do not touch it.
                // PADDOCK_LIN_KTZ_OLD=1 reverts to the pre-refit 2x-SMs
                // target - the serve A/B's leg-0 arm (same binary, env gate)
                static const bool ktz_old = [] {
                    const char* e = pd_env("PADDOCK_LIN_KTZ_OLD");
                    return e && atoi(e) != 0;
                }();
                // spec-verify band refit (mid-M f32-out ladder):
                // at batch 129..256 (2 col tiles) the col split
                // already doubles the chain count, so the old 2x-SMs target
                // oversplits the narrow-out shapes - down/dnout at z4 pay
                // double partial-plane traffic for chains the stream can't
                // feed. Isolated warm at M=256: dnout z2 39.2 vs z4 44.9 us
                // (-12.7%), down z2 92.2 vs 94.0 (-1.9%); q38inp/q38qkvz
                // already run z2 and measured worse at z1 (chain latency), so
                // cap at 2 rather than extending the b<=128 fill formula.
                if (!ktz_old && batch > 128u && batch <= 256u && nz > 2u) nz = 2u;
                if (!ktz_old && batch <= 128u) {
                    // 128 is the measured boundary on the 188-SM part: q
                    // (nt=64) needs target <= 128 for its z2 win, down/wo
                    // (nt=42) need > 126 to keep their proven z4 - ktz is
                    // cc12-default-only, so the constant binds this die;
                    // PADDOCK_LIN_KTZ_NZ re-tunes without a rebuild.
                    nz = (128u + nt - 1u) / nt;
                    if (nz > 4u) nz = 4u;
                    // chain-length floor (OPT-IN,
                    // PADDOCK_LIN_KTZ_CFLOOR=1): the fill target's
                    // z4 chops wo/kv into 10.5-stage chains whose ring
                    // fill is 28% of the chain - at SERVE (W streams from
                    // DRAM; the isolated sweep's L2-resident W flatters
                    // exactly these short-K shapes) wo runs 36.3us vs a
                    // 20.6 floor. Cap nz so chains keep >= 16 stages
                    // (ramp <= 3/16): wo/kv z4 -> z2 at 84/64 CTAs, q z2
                    // unchanged, down (nsp=168) keeps its serve-winning
                    // z4. Serve ABAB is the gate.
                    static const bool cfloor = [] {
                        const char* e = pd_env("PADDOCK_LIN_KTZ_CFLOOR");
                        return e && atoi(e) != 0;
                    }();
                    if (cfloor) {
                        uint32_t zmax = nsp / 16u;
                        if (zmax < 1u) zmax = 1u;
                        if (nz > zmax) nz = zmax;
                    }
                }
                static int nzo = -1;
                if (nzo < 0) {
                    const char* e = pd_env("PADDOCK_LIN_KTZ_NZ");
                    nzo = e ? atoi(e) : 0;
                }
                if (nzo >= 1 && nzo <= 16) nz = (uint32_t)nzo;
                if (nz > nsp) nz = nsp;
            }
            if (nz > 1u) {
                // ktzf (OPT-IN, PADDOCK_LIN_KTZF=1): serialized-K in-kernel
                // reduction - z epilogues add into y in z order behind a
                // per-tile release/acquire flag (kt3's KF arm), deleting
                // the partial planes AND the combine launch. The add order
                // is the combine's exact left-assoc sum -> bitwise-gated.
                // Single-wave grids only: every split of a tile must be
                // RESIDENT for the flag chain to advance (nt*nz <= SMs;
                // the decode-M refit grids all qualify). f32-out only -
                // o16 keeps the combine's cast path. No allocs here, so
                // the arm is graph-capture-safe unconditionally.
                static const bool ktzf = [] {
                    const char* e = pd_env("PADDOCK_LIN_KTZF");
                    return e && atoi(e) != 0;
                }();
                if (ktzf && !o16 && nt * nz <= (uint32_t)sms && nt <= 512u) {
                    static bool alinzf = false;
                    if (!alinzf) {
                        cudaFuncSetAttribute((const void*)pd_f8_gemm_lin_kt3<false, true, true>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem3);
                        pd_prefer_max_shared(pd_f8_gemm_lin_kt3<false, true, true>);
                        alinzf = true;
                    }
                    dim3 gf(nt, 1, nz);
                    pd_f8_gemm_lin_kt3<false, true, true><<<gf, 288, smem3, (cudaStream_t)stream>>>(
                        (const unsigned char*)wlin, ym, (const unsigned char*)xs,
                        (float*)y, in_dim, out_dim, batch);
                    return pd_launch_status();
                }
                const size_t plane = (size_t)out_dim * batch;
                const size_t need = (size_t)out_dim * 1024u * 4u * nz;
                // GRAPH-SAFETY: decode ticks are stream-captured
                // (THREAD_LOCAL) - a sync or alloc here mid-capture INVALIDATES
                // the graph being recorded (measured: 6 begins, 5 instantiates,
                // and the failed tick killed its generations). Rules: (a) the
                // scratch is pre-sized once, eagerly, to the dispatch gate's
                // own ceiling (nt <= 1.5*sms and nz <= 4 bound need at
                // 1.5*sms*64 * 1024 * 4 * 4 - ~296 MB on 188 SMs), so steady
                // state never grows; (b) growth, if it ever fires, is
                // GROW-ONLY - never free the old buffer, captured graphs bake
                // the pointer they saw and a free turns replays into
                // use-after-free; (c) under an active capture we do not touch
                // the allocator - the call falls through to the single-chain
                // kernel inside that graph (correct, just unsplit there).
                void*& part = pd_lin_ktz_part;
                size_t& cap = pd_lin_ktz_cap;
                cudaStreamCaptureStatus ccs = cudaStreamCaptureStatusNone;
                cudaStreamIsCapturing((cudaStream_t)stream, &ccs);
                if (need > cap && ccs == cudaStreamCaptureStatusNone) {
                    size_t want = (size_t)((3u * (uint32_t)sms) / 2u) * 64u * 1024u * 4u
                        * (nz > 4u ? nz : 4u); // ceiling tracks the nz override
                    if (want < need) want = need; // belt-and-braces
                    void* np = nullptr;
                    if (cudaMalloc(&np, want) == cudaSuccess) {
                        part = np; // old buffer (if any) intentionally stays alive
                        cap = want;
                    }
                }
                if (part && need <= cap) {
                    static bool alinz = false;
                    if (!alinz) {
                        cudaFuncSetAttribute((const void*)pd_f8_gemm_lin_kt3<false, true>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem3);
                        pd_prefer_max_shared(pd_f8_gemm_lin_kt3<false, true>);
                        alinz = true;
                    }
                    dim3 g3(nt, 1, nz);
                    if (ktd_on)
                        pd_f8_gemm_lin_ktd<false, true><<<g3, 288, smemd, (cudaStream_t)stream>>>(
                            (const unsigned char*)wlin, ym, (const unsigned char*)xs,
                            (float*)part, in_dim, out_dim, batch);
                    else
                        pd_f8_gemm_lin_kt3<false, true><<<g3, 288, smem3, (cudaStream_t)stream>>>(
                            (const unsigned char*)wlin, ym, (const unsigned char*)xs,
                            (float*)part, in_dim, out_dim, batch);
                    pd_lin_ktz_combine_kernel<<<(uint32_t)((plane + 255u) / 256u), 256, 0,
                        (cudaStream_t)stream>>>(
                        (const float*)part, (float*)y, plane, nz, o16);
                    return pd_launch_status();
                }
            }
        }
        // kt3c persistent chaining (DEFAULT-ON; kill
        // PADDOCK_NO_LIN_KT3C): the 1.5-4-wave band ktz's gate excludes
        // (nt > 1.5*sms) runs partial waves at 1 CTA/SM - wide-decode gu
        // (nt=336 at r<=128) and the qwen3.8 prefill GDN-out plane (nt=640)
        // are the tenants. Chained tiles on ceil(nt/tpc) CTAs balance the
        // chains and keep the TMA ring warm across tile boundaries;
        // bit-exact vs kt3 per tile (no partials), so the gate was a plain
        // serve A/B: same-run, +0.9% output throughput at the 4-wave band
        // cap.
        static const bool kt3c = [] {
            if (pd_env("PADDOCK_NO_LIN_KT3C") != nullptr) return false;
            const char* e = pd_env("PADDOCK_LIN_KT3C");
            return e == nullptr || atoi(e) != 0;
        }();
        if (kt3c) {
            static int sms_c = 0;
            if (!sms_c) {
                int dev = 0;
                cudaGetDevice(&dev);
                cudaDeviceGetAttribute(&sms_c, cudaDevAttrMultiProcessorCount, dev);
            }
            const uint32_t s = (uint32_t)sms_c;
            // band cap raised 3->4 waves: the qwen3.8
            // prefill GDN-out plane (out 5120, M-span 2048 -> nt 640 = 3.4
            // waves on 188 SMs) ran kt3 at 217 us vs a ~129 us K=4096 floor -
            // the short-K ring ramp paid 3.4x. nt=640 chains as tpc=4 on 160
            // CTAs, still <= 1 wave of chains.
            if (nt > (3u * s) / 2u && nt <= 4u * s) {
                static bool alin3c = false;
                if (!alin3c) {
                    cudaFuncSetAttribute((const void*)pd_f8_gemm_lin_kt3c<true>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem3);
                    cudaFuncSetAttribute((const void*)pd_f8_gemm_lin_kt3c<false>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem3);
                    pd_prefer_max_shared(pd_f8_gemm_lin_kt3c<true>);
                    pd_prefer_max_shared(pd_f8_gemm_lin_kt3c<false>);
                    alin3c = true;
                }
                const uint32_t tpc = (nt + s - 1u) / s;      // 2 or 3 in-band
                const uint32_t gc = (nt + tpc - 1u) / tpc;   // <= 1 wave of chains
                if (o16)
                    pd_f8_gemm_lin_kt3c<true><<<gc, 288, smem3, (cudaStream_t)stream>>>(
                        (const unsigned char*)wlin, ym, (const unsigned char*)xs,
                        (float*)y, in_dim, out_dim, batch);
                else
                    pd_f8_gemm_lin_kt3c<false><<<gc, 288, smem3, (cudaStream_t)stream>>>(
                        (const unsigned char*)wlin, ym, (const unsigned char*)xs,
                        (float*)y, in_dim, out_dim, batch);
                return pd_launch_status();
            }
        }
        if (ktd_on) {
            if (o16)
                pd_f8_gemm_lin_ktd<true><<<nt, 288, smemd, (cudaStream_t)stream>>>(
                    (const unsigned char*)wlin, ym, (const unsigned char*)xs,
                    (float*)y, in_dim, out_dim, batch);
            else
                pd_f8_gemm_lin_ktd<false><<<nt, 288, smemd, (cudaStream_t)stream>>>(
                    (const unsigned char*)wlin, ym, (const unsigned char*)xs,
                    (float*)y, in_dim, out_dim, batch);
        } else if (o16)
            pd_f8_gemm_lin_kt3<true><<<nt, 288, smem3, (cudaStream_t)stream>>>(
                (const unsigned char*)wlin, ym, (const unsigned char*)xs,
                (float*)y, in_dim, out_dim, batch);
        else
            pd_f8_gemm_lin_kt3<false><<<nt, 288, smem3, (cudaStream_t)stream>>>(
                (const unsigned char*)wlin, ym, (const unsigned char*)xs,
                (float*)y, in_dim, out_dim, batch);
        return pd_launch_status();
    }
    // early-arrive A/B (OPT-IN, PADDOCK_LIN_KTEA=1): the win=true consumer
    // path - bar.arrive right after the pair's fragment loads instead of
    // after the mma tail, so the producer's bar.sync (and thus the next TMA
    // issue) completes one mma-block earlier. Stage-period lever from the
    // stage-latency model; same loads/math order -> bit-exact.
    static const bool ktea = [] {
        const char* e = pd_env("PADDOCK_LIN_KTEA");
        return e && atoi(e) != 0;
    }();
    if (ktea) {
        static bool alinw = false;
        if (!alinw) {
            cudaFuncSetAttribute((const void*)pd_f8_gemm_lin_ktk<true, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
            cudaFuncSetAttribute((const void*)pd_f8_gemm_lin_ktk<true, false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
            alinw = true;
        }
        if (o16)
            pd_f8_gemm_lin_ktk<true, true><<<nt, 384, smem, (cudaStream_t)stream>>>(
                (const unsigned char*)wlin, ym, (const unsigned char*)xs,
                (float*)y, in_dim, out_dim, batch);
        else
            pd_f8_gemm_lin_ktk<true, false><<<nt, 384, smem, (cudaStream_t)stream>>>(
                (const unsigned char*)wlin, ym, (const unsigned char*)xs,
                (float*)y, in_dim, out_dim, batch);
        return pd_launch_status();
    }
    if (o16)
        pd_f8_gemm_lin_ktk<false, true><<<nt, 384, smem, (cudaStream_t)stream>>>(
            (const unsigned char*)wlin, ym, (const unsigned char*)xs,
            (float*)y, in_dim, out_dim, batch);
    else
        pd_f8_gemm_lin_ktk<false, false><<<nt, 384, smem, (cudaStream_t)stream>>>(
            (const unsigned char*)wlin, ym, (const unsigned char*)xs,
            (float*)y, in_dim, out_dim, batch);
    return pd_launch_status();
#endif
}

// q36 DN: the fused in_qkv|gate lin plane as one kt3 grid with a
// two-buffer epilogue (y gets rows [0,ncut), y2 gets [ncut,out_dim), each
// at its own row stride - the conv/gate consumers keep their layouts).
// Motivation is pure wave arithmetic: the split gate launch is a 48-tile
// grid (192/384/768 CTAs at r=512/1024/2048 on a 188-SM die = a ~1.0x
// fractional wave every time); the merged 128-tile grid pays one tail
// (pair -> merged -20/-13/-7% at 512/1024/2048).
// Deliberately NARROW: plain kt3 only - no ktz/ktd/opt-in variants (the
// merged grid is never in ktz's win region) and no o16. Returns -2 when
// the route can't engage so the caller keeps its two-launch pair - the
// pd_f8_gemm_lin_gu convention.
PD_EXPORT
int pd_f8_gemm_lin_kt_split(const void* wlin, const void* xq, const void* xs,
                            void* y, void* y2, uint32_t ncut, uint32_t in_dim,
                            uint32_t out_dim, uint32_t batch, void* stream) {
#ifndef PD_BS_HOST
    (void)wlin; (void)xq; (void)xs; (void)y; (void)y2; (void)ncut;
    (void)in_dim; (void)out_dim; (void)batch; (void)stream;
    return -2;
#else
    if (out_dim == 0 || batch == 0) return 0;
    if ((in_dim & 127u) != 0 || (out_dim & 127u) != 0 || (ncut & 127u) != 0
        || ncut == 0u || ncut >= out_dim)
        return -2;
    static const bool tma = [] {
        int dev = 0, cma = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&cma, cudaDevAttrComputeCapabilityMajor, dev);
        return cma >= 9 && pd_tmap_encode() != nullptr
            && pd_env("PADDOCK_NO_F8W8_TMA") == nullptr;
    }();
    // ride the kt3 election exactly like the gu fusion: the SPLIT arm is
    // kt3's epilogue twin, unmeasured anywhere kt3 isn't the incumbent
    static const bool kt3 = [] {
        const char* e = pd_env("PADDOCK_LIN_KT3");
        if (e) return atoi(e) != 0;
        int dev = 0, cma = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&cma, cudaDevAttrComputeCapabilityMajor, dev);
        return cma == 12;
    }();
    if (!tma || !kt3) return -2;
    const uint32_t smem3 = 99864u;
    static bool asp = false;
    if (!asp) {
        cudaFuncSetAttribute(
            (const void*)pd_f8_gemm_lin_kt3<false, false, false, false, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem3);
        pd_prefer_max_shared(pd_f8_gemm_lin_kt3<false, false, false, false, true>);
        asp = true;
    }
    CUtensorMap ym;
    if (!pd_tmap_2d(&ym, xq, in_dim, batch)) return -2;
    const uint32_t bp = (batch + 127u) & ~127u;
    const uint32_t nt = (out_dim >> 7) * (bp >> 7);
    pd_f8_gemm_lin_kt3<false, false, false, false, true>
        <<<nt, 288, smem3, (cudaStream_t)stream>>>(
        (const unsigned char*)wlin, ym, (const unsigned char*)xs, (float*)y,
        in_dim, out_dim, batch, nullptr, (float*)y2, ncut);
    return pd_launch_status();
#endif
}

// Fused gu GEMM + geglu + per-32 e4m3 quant on an INTERLEAVED lin plane
// (pd_f8w_repack_lin_gui). q gets [batch][out_dim/2] e4m3 bytes, qs the
// ue8m0 scale bytes - exactly what quantize_e4m3 would hand the down GEMM,
// bit-identical to the kt3 -> geglu2 chain. Returns -2 when the route can't
// engage (no TMA, kt3 off for this cc/env) so the caller keeps the 2-launch
// chain - same convention as attn_spec_batch_fin.
template <int ACT>
static inline int pd_f8_gemm_lin_gu_launch(
                      const void* wlin, const void* xq, const void* xs,
                      void* q, void* qs, uint32_t in_dim, uint32_t out_dim,
                      uint32_t batch, void* stream) {
#ifndef PD_BS_HOST
    (void)wlin; (void)xq; (void)xs; (void)q; (void)qs;
    (void)in_dim; (void)out_dim; (void)batch; (void)stream;
    return -2;
#else
    if (out_dim == 0 || batch == 0) return 0;
    // full 128-row tiles only: the epilogue assumes every tile is 64 whole
    // pairs (gemma4 gu 43008 qualifies); odd shapes keep the 2-launch chain
    if ((in_dim & 127u) != 0 || (out_dim & 127u) != 0) return -2;
    static const bool tma = [] {
        int dev = 0, cma = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&cma, cudaDevAttrComputeCapabilityMajor, dev);
        return cma >= 9 && pd_tmap_encode() != nullptr
            && pd_env("PADDOCK_NO_F8W8_TMA") == nullptr;
    }();
    // ride the kt3 election: the fused kernel is a kt3 clone (BS pipeline);
    // where kt3 isn't the incumbent the fused arm is unmeasured
    static const bool kt3 = [] {
        const char* e = pd_env("PADDOCK_LIN_KT3");
        if (e) return atoi(e) != 0;
        int dev = 0, cma = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&cma, cudaDevAttrComputeCapabilityMajor, dev);
        return cma == 12;
    }();
    if (!tma || !kt3) return -2;
    const uint32_t smem3 = 99864u;
    static bool alin3g = false;
    if (!alin3g) {
        cudaFuncSetAttribute((const void*)pd_f8_gemm_lin_kt3g<false, ACT>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem3);
        pd_prefer_max_shared(pd_f8_gemm_lin_kt3g<false, ACT>);
        alin3g = true;
    }
    CUtensorMap ym;
    if (!pd_tmap_2d(&ym, xq, in_dim, batch)) return -2;
    const uint32_t bp = (batch + 127u) & ~127u;
    const uint32_t nt = (out_dim >> 7) * (bp >> 7);
    pd_f8_gemm_lin_kt3g<false, ACT><<<nt, 288, smem3, (cudaStream_t)stream>>>(
        (const unsigned char*)wlin, ym, (const unsigned char*)xs,
        (unsigned char*)q, (unsigned char*)qs, in_dim, out_dim, batch);
    return pd_launch_status();
#endif
}

PD_EXPORT
int pd_f8_gemm_lin_gu(const void* wlin, const void* xq, const void* xs,
                      void* q, void* qs, uint32_t in_dim, uint32_t out_dim,
                      uint32_t batch, void* stream) {
    return pd_f8_gemm_lin_gu_launch<PD_ACT_GELU>(wlin, xq, xs, q, qs, in_dim,
                                                 out_dim, batch, stream);
}

// SiLU twin (muse-glimmer). Same plane, same election, same quant epilogue -
// only the gate nonlinearity folded into the writeback differs.
PD_EXPORT
int pd_f8_gemm_lin_gu_silu(const void* wlin, const void* xq, const void* xs,
                           void* q, void* qs, uint32_t in_dim, uint32_t out_dim,
                           uint32_t batch, void* stream) {
    return pd_f8_gemm_lin_gu_launch<PD_ACT_SILU>(wlin, xq, xs, q, qs, in_dim,
                                                 out_dim, batch, stream);
}

// per-channel gu launcher (kt4a twin): as_row = f32 per-token scales from the
// row quantizer, ws = f32 per-channel scales [2*n_ff] (gate at 0, up at n_ff)
// decoded from the pc plane's ue8m0 exponents at load. Same election guards
// as pd_f8_gemm_lin_gu; -2 = shape/route not covered (caller pre-checks make
// this unreachable in the pc lane - treat as error there).
template <int ACT>
static inline int pd_f8_gemm_lin_gu_pc_launch(
                         const void* wlin, const void* xq, const void* as_row,
                         const void* ws, void* q, void* qs, uint32_t in_dim,
                         uint32_t out_dim, uint32_t batch, void* stream) {
#ifndef PD_BS_HOST
    (void)wlin; (void)xq; (void)as_row; (void)ws; (void)q; (void)qs;
    (void)in_dim; (void)out_dim; (void)batch; (void)stream;
    return -2;
#else
    if (out_dim == 0 || batch == 0) return 0;
    if ((in_dim & 127u) != 0 || (out_dim & 127u) != 0) return -2;
    static const bool tma = [] {
        int dev = 0, cma = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&cma, cudaDevAttrComputeCapabilityMajor, dev);
        return cma >= 9 && pd_tmap_encode() != nullptr
            && pd_env("PADDOCK_NO_F8W8_TMA") == nullptr;
    }();
    if (!tma) return -2;
    const uint32_t smem4 = 99864u;
    static bool alin4a = false;
    if (!alin4a) {
        cudaFuncSetAttribute((const void*)pd_f8_gemm_lin_kt4a<false, ACT>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem4);
        pd_prefer_max_shared(pd_f8_gemm_lin_kt4a<false, ACT>);
        alin4a = true;
    }
    CUtensorMap ym;
    if (!pd_tmap_2d(&ym, xq, in_dim, batch)) return -2;
    const uint32_t bp = (batch + 127u) & ~127u;
    const uint32_t nt = (out_dim >> 7) * (bp >> 7);
    const float* wsf = (const float*)ws;
    pd_f8_gemm_lin_kt4a<false, ACT><<<nt, 288, smem4, (cudaStream_t)stream>>>(
        (const unsigned char*)wlin, ym, (unsigned char*)q, (unsigned char*)qs,
        (const float*)as_row, wsf, wsf + (out_dim >> 1), in_dim, out_dim, batch);
    return pd_launch_status();
#endif
}

int pd_f8_gemm_lin_gu_pc(const void* wlin, const void* xq, const void* as_row,
                         const void* ws, void* q, void* qs, uint32_t in_dim,
                         uint32_t out_dim, uint32_t batch, void* stream) {
    return pd_f8_gemm_lin_gu_pc_launch<PD_ACT_GELU>(wlin, xq, as_row, ws, q, qs,
                                                    in_dim, out_dim, batch, stream);
}

int pd_f8_gemm_lin_gu_pc_silu(const void* wlin, const void* xq, const void* as_row,
                              const void* ws, void* q, void* qs, uint32_t in_dim,
                              uint32_t out_dim, uint32_t batch, void* stream) {
    return pd_f8_gemm_lin_gu_pc_launch<PD_ACT_SILU>(wlin, xq, as_row, ws, q, qs,
                                                    in_dim, out_dim, batch, stream);
}

// pc lin launcher for the qkv/wo classes: `row_off` slices a fused plane
// (multiple of 128 - boxes are row-tile-major, one pointer offset), `ws` is
// the per-channel scale vector already sliced to the segment (engine passes
// base + ws_off). as_row = f32 per-token scales. -2 = route not covered.
int pd_f8_gemm_w8_pc(const void* wlin, uint32_t row_off, const void* xq,
                     const void* as_row, const void* ws, void* y,
                     uint32_t in_dim, uint32_t out_dim, uint32_t batch,
                     uint32_t o16, void* stream) {
#ifndef PD_BS_HOST
    (void)wlin; (void)row_off; (void)xq; (void)as_row; (void)ws; (void)y;
    (void)in_dim; (void)out_dim; (void)batch; (void)o16; (void)stream;
    return -2;
#else
    if (out_dim == 0 || batch == 0) return 0;
    if ((in_dim & 127u) != 0 || (out_dim & 127u) != 0 || (row_off & 127u) != 0)
        return -2;
    static const bool tma = [] {
        int dev = 0, cma = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&cma, cudaDevAttrComputeCapabilityMajor, dev);
        return cma >= 9 && pd_tmap_encode() != nullptr
            && pd_env("PADDOCK_NO_F8W8_TMA") == nullptr;
    }();
    if (!tma) return -2;
    const uint32_t smem4 = 99864u;
    static bool alin4 = false;
    if (!alin4) {
        cudaFuncSetAttribute((const void*)pd_f8_gemm_lin_kt4<false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem4);
        pd_prefer_max_shared(pd_f8_gemm_lin_kt4<false>);
        cudaFuncSetAttribute((const void*)pd_f8_gemm_lin_kt4<true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem4);
        pd_prefer_max_shared(pd_f8_gemm_lin_kt4<true>);
        alin4 = true;
    }
    CUtensorMap ym;
    if (!pd_tmap_2d(&ym, xq, in_dim, batch)) return -2;
    const uint32_t nkboxes = in_dim >> 7;
    const unsigned char* wp = (const unsigned char*)wlin
        + (size_t)(row_off >> 7) * nkboxes * PD_LIN_BOX;
    const uint32_t bp = (batch + 127u) & ~127u;
    const uint32_t nt = (out_dim >> 7) * (bp >> 7);
    if (o16)
        pd_f8_gemm_lin_kt4<true><<<nt, 288, smem4, (cudaStream_t)stream>>>(
            wp, ym, (float*)y, (const float*)as_row, (const float*)ws,
            in_dim, out_dim, batch, nullptr, nullptr, 0u, 0u);
    else
        pd_f8_gemm_lin_kt4<false><<<nt, 288, smem4, (cudaStream_t)stream>>>(
            wp, ym, (float*)y, (const float*)as_row, (const float*)ws,
            in_dim, out_dim, batch, nullptr, nullptr, 0u, 0u);
    return pd_launch_status();
#endif
}

// down-twin launcher (kt4d): per-32 activation scales stay (xs), weights
// per-channel via segment-sliced ws. -2 = route not covered.
int pd_f8_gemm_w8_pcd(const void* wlin, uint32_t row_off, const void* xq,
                      const void* xs, const void* ws, void* y,
                      uint32_t in_dim, uint32_t out_dim, uint32_t batch,
                      uint32_t o16, void* stream) {
#ifndef PD_BS_HOST
    (void)wlin; (void)row_off; (void)xq; (void)xs; (void)ws; (void)y;
    (void)in_dim; (void)out_dim; (void)batch; (void)o16; (void)stream;
    return -2;
#else
    if (out_dim == 0 || batch == 0) return 0;
    if ((in_dim & 127u) != 0 || (out_dim & 127u) != 0 || (row_off & 127u) != 0)
        return -2;
    static const bool tma = [] {
        int dev = 0, cma = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&cma, cudaDevAttrComputeCapabilityMajor, dev);
        return cma >= 9 && pd_tmap_encode() != nullptr
            && pd_env("PADDOCK_NO_F8W8_TMA") == nullptr;
    }();
    if (!tma) return -2;
    const uint32_t smem4 = 99864u;
    static bool alin4d = false;
    if (!alin4d) {
        cudaFuncSetAttribute((const void*)pd_f8_gemm_lin_kt4d<false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem4);
        pd_prefer_max_shared(pd_f8_gemm_lin_kt4d<false>);
        cudaFuncSetAttribute((const void*)pd_f8_gemm_lin_kt4d<true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem4);
        pd_prefer_max_shared(pd_f8_gemm_lin_kt4d<true>);
        alin4d = true;
    }
    CUtensorMap ym;
    if (!pd_tmap_2d(&ym, xq, in_dim, batch)) return -2;
    const uint32_t nkboxes = in_dim >> 7;
    const unsigned char* wp = (const unsigned char*)wlin
        + (size_t)(row_off >> 7) * nkboxes * PD_LIN_BOX;
    const uint32_t bp = (batch + 127u) & ~127u;
    const uint32_t nt = (out_dim >> 7) * (bp >> 7);
    if (o16)
        pd_f8_gemm_lin_kt4d<true><<<nt, 288, smem4, (cudaStream_t)stream>>>(
            wp, ym, (const unsigned char*)xs, (float*)y, (const float*)ws,
            in_dim, out_dim, batch);
    else
        pd_f8_gemm_lin_kt4d<false><<<nt, 288, smem4, (cudaStream_t)stream>>>(
            wp, ym, (const unsigned char*)xs, (float*)y, (const float*)ws,
            in_dim, out_dim, batch);
    return pd_launch_status();
#endif
}

// ---- rowwise (strip-free) launchers ---------------------------------------
// The pc plane class in data-only 16,384B boxes + a per-row ue8m0 byte
// vector (wse, PADDED to the 128-row tail). Same elections as the strip
// twins; bit-exact against them. wse rides L2 (~tens of KB per
// plane) instead of 3.03% of the weight stream.

// gu fused twin on the rowwise interleaved plane (kt3g<RW>). wse in BOX ROW
// order (interleaved) - the engine builds it with the gui src_of map.
template <int ACT>
static inline int pd_f8_gemm_lin_gu_r_launch(
                        const void* wlin, const void* wse, const void* xq,
                        const void* xs, void* q, void* qs, uint32_t in_dim,
                        uint32_t out_dim, uint32_t batch, void* stream) {
#ifndef PD_BS_HOST
    (void)wlin; (void)wse; (void)xq; (void)xs; (void)q; (void)qs;
    (void)in_dim; (void)out_dim; (void)batch; (void)stream;
    return -2;
#else
    if (out_dim == 0 || batch == 0) return 0;
    if ((in_dim & 127u) != 0 || (out_dim & 127u) != 0) return -2;
    static const bool tma = [] {
        int dev = 0, cma = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&cma, cudaDevAttrComputeCapabilityMajor, dev);
        return cma >= 9 && pd_tmap_encode() != nullptr
            && pd_env("PADDOCK_NO_F8W8_TMA") == nullptr;
    }();
    static const bool kt3 = [] {
        const char* e = pd_env("PADDOCK_LIN_KT3");
        if (e) return atoi(e) != 0;
        int dev = 0, cma = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&cma, cudaDevAttrComputeCapabilityMajor, dev);
        return cma == 12;
    }();
    if (!tma || !kt3) return -2;
    const uint32_t smem3r = 98328u;
    static bool alin3gr = false;
    if (!alin3gr) {
        cudaFuncSetAttribute((const void*)pd_f8_gemm_lin_kt3g<true, ACT>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem3r);
        pd_prefer_max_shared(pd_f8_gemm_lin_kt3g<true, ACT>);
        alin3gr = true;
    }
    CUtensorMap ym;
    if (!pd_tmap_2d(&ym, xq, in_dim, batch)) return -2;
    const uint32_t bp = (batch + 127u) & ~127u;
    const uint32_t nt = (out_dim >> 7) * (bp >> 7);
    pd_f8_gemm_lin_kt3g<true, ACT><<<nt, 288, smem3r, (cudaStream_t)stream>>>(
        (const unsigned char*)wlin, ym, (const unsigned char*)xs,
        (unsigned char*)q, (unsigned char*)qs, in_dim, out_dim, batch,
        (const unsigned char*)wse);
    return pd_launch_status();
#endif
}

PD_EXPORT
int pd_f8_gemm_lin_gu_r(const void* wlin, const void* wse, const void* xq,
                        const void* xs, void* q, void* qs, uint32_t in_dim,
                        uint32_t out_dim, uint32_t batch, void* stream) {
    return pd_f8_gemm_lin_gu_r_launch<PD_ACT_GELU>(wlin, wse, xq, xs, q, qs,
                                                   in_dim, out_dim, batch, stream);
}

PD_EXPORT
int pd_f8_gemm_lin_gu_r_silu(const void* wlin, const void* wse, const void* xq,
                             const void* xs, void* q, void* qs, uint32_t in_dim,
                             uint32_t out_dim, uint32_t batch, void* stream) {
    return pd_f8_gemm_lin_gu_r_launch<PD_ACT_SILU>(wlin, wse, xq, xs, q, qs,
                                                   in_dim, out_dim, batch, stream);
}

// kt3-band rowwise launcher: the DEFAULT cc12 serve route only (kt3 + ktz
// K-split with the decode-M refit + opt-in ktzf). The falsified opt-in
// iteration arms (kt5/kt2/ktp/kt64/ktw/ktea/kt3c/ktd) have no rowwise twins
// - rowwise planes never route there.
PD_EXPORT
int pd_f8_gemm_lin_kt_r(const void* wlin, const void* wse, const void* xq,
                        const void* xs, void* y, uint32_t in_dim,
                        uint32_t out_dim, uint32_t batch, uint32_t o16,
                        void* stream) {
#ifndef PD_BS_HOST
    (void)wlin; (void)wse; (void)xq; (void)xs; (void)y; (void)in_dim;
    (void)out_dim; (void)batch; (void)o16; (void)stream;
    return cudaErrorNotSupported;
#else
    if (out_dim == 0 || batch == 0) return 0;
    if ((in_dim & 127u) != 0) return cudaErrorInvalidValue;
    static const bool tma = [] {
        int dev = 0, cma = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&cma, cudaDevAttrComputeCapabilityMajor, dev);
        return cma >= 9 && pd_tmap_encode() != nullptr
            && pd_env("PADDOCK_NO_F8W8_TMA") == nullptr;
    }();
    static const bool kt3 = [] {
        const char* e = pd_env("PADDOCK_LIN_KT3");
        if (e) return atoi(e) != 0;
        int dev = 0, cma = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&cma, cudaDevAttrComputeCapabilityMajor, dev);
        return cma == 12;
    }();
    if (!tma || !kt3) return cudaErrorNotSupported;
    const uint32_t smem3r = 98328u;
    static bool alin3r = false;
    if (!alin3r) {
        cudaFuncSetAttribute((const void*)pd_f8_gemm_lin_kt3<false, false, false, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem3r);
        cudaFuncSetAttribute((const void*)pd_f8_gemm_lin_kt3<true, false, false, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem3r);
        cudaFuncSetAttribute((const void*)pd_f8_gemm_lin_kt3<false, true, false, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem3r);
        pd_prefer_max_shared(pd_f8_gemm_lin_kt3<false, false, false, true>);
        pd_prefer_max_shared(pd_f8_gemm_lin_kt3<true, false, false, true>);
        pd_prefer_max_shared(pd_f8_gemm_lin_kt3<false, true, false, true>);
        alin3r = true;
    }
    CUtensorMap ym;
    if (!pd_tmap_2d(&ym, xq, in_dim, batch)) return cudaErrorNotSupported;
    const uint32_t bp = (batch + 127u) & ~127u;
    const uint32_t nt = ((out_dim + 127u) / 128u) * (bp >> 7);
    const unsigned char* wsep = (const unsigned char*)wse;
    // ktz election: verbatim from the strip route (envs shared, scratch
    // shared) - see pd_f8_gemm_lin_kt for the measured trail
    static const bool ktz = [] {
        const char* e = pd_env("PADDOCK_LIN_KTZ");
        if (e) return atoi(e) != 0;
        int dev = 0, cma = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&cma, cudaDevAttrComputeCapabilityMajor, dev);
        return cma == 12;
    }();
    if (ktz && batch <= 1024u) {
        static int sms = 0;
        if (!sms) {
            int dev = 0;
            cudaGetDevice(&dev);
            cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        }
        const uint32_t nsp = (((in_dim + 63u) / 64u) + 1u) >> 1;
        uint32_t nz = 0;
        if (nt <= (3u * (uint32_t)sms) / 2u) {
            nz = (2u * (uint32_t)sms) / nt;
            if (nz < 2u) nz = 2u;
            if (nz > 4u) nz = 4u;
            static const bool ktz_old = [] {
                const char* e = pd_env("PADDOCK_LIN_KTZ_OLD");
                return e && atoi(e) != 0;
            }();
            if (!ktz_old && batch <= 128u) {
                nz = (128u + nt - 1u) / nt;
                if (nz > 4u) nz = 4u;
                static const bool cfloor = [] {
                    const char* e = pd_env("PADDOCK_LIN_KTZ_CFLOOR");
                    return e && atoi(e) != 0;
                }();
                if (cfloor) {
                    uint32_t zmax = nsp / 16u;
                    if (zmax < 1u) zmax = 1u;
                    if (nz > zmax) nz = zmax;
                }
            }
            static int nzo = -1;
            if (nzo < 0) {
                const char* e = pd_env("PADDOCK_LIN_KTZ_NZ");
                nzo = e ? atoi(e) : 0;
            }
            if (nzo >= 1 && nzo <= 16) nz = (uint32_t)nzo;
            if (nz > nsp) nz = nsp;
        }
        if (nz > 1u) {
            static const bool ktzf = [] {
                const char* e = pd_env("PADDOCK_LIN_KTZF");
                return e && atoi(e) != 0;
            }();
            if (ktzf && !o16 && nt * nz <= (uint32_t)sms && nt <= 512u) {
                static bool alinzfr = false;
                if (!alinzfr) {
                    cudaFuncSetAttribute((const void*)pd_f8_gemm_lin_kt3<false, true, true, true>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem3r);
                    pd_prefer_max_shared(pd_f8_gemm_lin_kt3<false, true, true, true>);
                    alinzfr = true;
                }
                dim3 gf(nt, 1, nz);
                pd_f8_gemm_lin_kt3<false, true, true, true><<<gf, 288, smem3r, (cudaStream_t)stream>>>(
                    (const unsigned char*)wlin, ym, (const unsigned char*)xs,
                    (float*)y, in_dim, out_dim, batch, wsep);
                return pd_launch_status();
            }
            const size_t plane = (size_t)out_dim * batch;
            const size_t need = (size_t)out_dim * 1024u * 4u * nz;
            void*& part = pd_lin_ktz_part;
            size_t& cap = pd_lin_ktz_cap;
            cudaStreamCaptureStatus ccs = cudaStreamCaptureStatusNone;
            cudaStreamIsCapturing((cudaStream_t)stream, &ccs);
            if (need > cap && ccs == cudaStreamCaptureStatusNone) {
                size_t want = (size_t)((3u * (uint32_t)sms) / 2u) * 64u * 1024u * 4u
                    * (nz > 4u ? nz : 4u);
                if (want < need) want = need;
                void* np = nullptr;
                if (cudaMalloc(&np, want) == cudaSuccess) {
                    part = np;
                    cap = want;
                }
            }
            if (part && need <= cap) {
                dim3 g3(nt, 1, nz);
                pd_f8_gemm_lin_kt3<false, true, false, true><<<g3, 288, smem3r, (cudaStream_t)stream>>>(
                    (const unsigned char*)wlin, ym, (const unsigned char*)xs,
                    (float*)part, in_dim, out_dim, batch, wsep);
                pd_lin_ktz_combine_kernel<<<(uint32_t)((plane + 255u) / 256u), 256, 0,
                    (cudaStream_t)stream>>>(
                    (const float*)part, (float*)y, plane, nz, o16);
                return pd_launch_status();
            }
        }
    }
    if (o16)
        pd_f8_gemm_lin_kt3<true, false, false, true><<<nt, 288, smem3r, (cudaStream_t)stream>>>(
            (const unsigned char*)wlin, ym, (const unsigned char*)xs,
            (float*)y, in_dim, out_dim, batch, wsep);
    else
        pd_f8_gemm_lin_kt3<false, false, false, true><<<nt, 288, smem3r, (cudaStream_t)stream>>>(
            (const unsigned char*)wlin, ym, (const unsigned char*)xs,
            (float*)y, in_dim, out_dim, batch, wsep);
    return pd_launch_status();
#endif
}

// per-channel gu chunk launcher on the rowwise plane (kt4a<RW> - the
// mainloop never read the strip; same as_row/ws epilogue contract)
template <int ACT>
static inline int pd_f8_gemm_lin_gu_pc_r_launch(
                           const void* wlin, const void* xq, const void* as_row,
                           const void* ws, void* q, void* qs, uint32_t in_dim,
                           uint32_t out_dim, uint32_t batch, void* stream) {
#ifndef PD_BS_HOST
    (void)wlin; (void)xq; (void)as_row; (void)ws; (void)q; (void)qs;
    (void)in_dim; (void)out_dim; (void)batch; (void)stream;
    return -2;
#else
    if (out_dim == 0 || batch == 0) return 0;
    if ((in_dim & 127u) != 0 || (out_dim & 127u) != 0) return -2;
    static const bool tma = [] {
        int dev = 0, cma = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&cma, cudaDevAttrComputeCapabilityMajor, dev);
        return cma >= 9 && pd_tmap_encode() != nullptr
            && pd_env("PADDOCK_NO_F8W8_TMA") == nullptr;
    }();
    if (!tma) return -2;
    const uint32_t smem4r = 98328u;
    static bool alin4ar = false;
    if (!alin4ar) {
        cudaFuncSetAttribute((const void*)pd_f8_gemm_lin_kt4a<true, ACT>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem4r);
        pd_prefer_max_shared(pd_f8_gemm_lin_kt4a<true, ACT>);
        alin4ar = true;
    }
    CUtensorMap ym;
    if (!pd_tmap_2d(&ym, xq, in_dim, batch)) return -2;
    const uint32_t bp = (batch + 127u) & ~127u;
    const uint32_t nt = (out_dim >> 7) * (bp >> 7);
    const float* wsf = (const float*)ws;
    pd_f8_gemm_lin_kt4a<true, ACT><<<nt, 288, smem4r, (cudaStream_t)stream>>>(
        (const unsigned char*)wlin, ym, (unsigned char*)q, (unsigned char*)qs,
        (const float*)as_row, wsf, wsf + (out_dim >> 1), in_dim, out_dim, batch);
    return pd_launch_status();
#endif
}

PD_EXPORT
int pd_f8_gemm_lin_gu_pc_r(const void* wlin, const void* xq, const void* as_row,
                           const void* ws, void* q, void* qs, uint32_t in_dim,
                           uint32_t out_dim, uint32_t batch, void* stream) {
    return pd_f8_gemm_lin_gu_pc_r_launch<PD_ACT_GELU>(wlin, xq, as_row, ws, q, qs,
                                                      in_dim, out_dim, batch, stream);
}

PD_EXPORT
int pd_f8_gemm_lin_gu_pc_r_silu(const void* wlin, const void* xq, const void* as_row,
                                const void* ws, void* q, void* qs, uint32_t in_dim,
                                uint32_t out_dim, uint32_t batch, void* stream) {
    return pd_f8_gemm_lin_gu_pc_r_launch<PD_ACT_SILU>(wlin, xq, as_row, ws, q, qs,
                                                      in_dim, out_dim, batch, stream);
}

// pc qkv/wo chunk launcher on the rowwise plane (kt4<RW>)
PD_EXPORT
int pd_f8_gemm_w8_pc_r(const void* wlin, uint32_t row_off, const void* xq,
                       const void* as_row, const void* ws, void* y,
                       uint32_t in_dim, uint32_t out_dim, uint32_t batch,
                       uint32_t o16, void* stream) {
#ifndef PD_BS_HOST
    (void)wlin; (void)row_off; (void)xq; (void)as_row; (void)ws; (void)y;
    (void)in_dim; (void)out_dim; (void)batch; (void)o16; (void)stream;
    return -2;
#else
    if (out_dim == 0 || batch == 0) return 0;
    if ((in_dim & 127u) != 0 || (out_dim & 127u) != 0 || (row_off & 127u) != 0)
        return -2;
    static const bool tma = [] {
        int dev = 0, cma = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&cma, cudaDevAttrComputeCapabilityMajor, dev);
        return cma >= 9 && pd_tmap_encode() != nullptr
            && pd_env("PADDOCK_NO_F8W8_TMA") == nullptr;
    }();
    if (!tma) return -2;
    const uint32_t smem4r = 98328u;
    static bool alin4r = false;
    if (!alin4r) {
        cudaFuncSetAttribute((const void*)pd_f8_gemm_lin_kt4<false, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem4r);
        cudaFuncSetAttribute((const void*)pd_f8_gemm_lin_kt4<true, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem4r);
        pd_prefer_max_shared(pd_f8_gemm_lin_kt4<false, true>);
        pd_prefer_max_shared(pd_f8_gemm_lin_kt4<true, true>);
        alin4r = true;
    }
    CUtensorMap ym;
    if (!pd_tmap_2d(&ym, xq, in_dim, batch)) return -2;
    const uint32_t nkboxes = in_dim >> 7;
    const unsigned char* wp = (const unsigned char*)wlin
        + (size_t)(row_off >> 7) * nkboxes * PD_LINBS_BOX;
    const uint32_t bp = (batch + 127u) & ~127u;
    const uint32_t nt = (out_dim >> 7) * (bp >> 7);
    if (o16)
        pd_f8_gemm_lin_kt4<true, true><<<nt, 288, smem4r, (cudaStream_t)stream>>>(
            wp, ym, (float*)y, (const float*)as_row, (const float*)ws,
            in_dim, out_dim, batch, nullptr, nullptr, 0u, 0u);
    else
        pd_f8_gemm_lin_kt4<false, true><<<nt, 288, smem4r, (cudaStream_t)stream>>>(
            wp, ym, (float*)y, (const float*)as_row, (const float*)ws,
            in_dim, out_dim, batch, nullptr, nullptr, 0u, 0u);
    return pd_launch_status();
#endif
}

// down-twin chunk launcher on the rowwise plane (kt4d<RW>)
PD_EXPORT
int pd_f8_gemm_w8_pcd_r(const void* wlin, uint32_t row_off, const void* xq,
                        const void* xs, const void* ws, void* y,
                        uint32_t in_dim, uint32_t out_dim, uint32_t batch,
                        uint32_t o16, void* stream) {
#ifndef PD_BS_HOST
    (void)wlin; (void)row_off; (void)xq; (void)xs; (void)ws; (void)y;
    (void)in_dim; (void)out_dim; (void)batch; (void)o16; (void)stream;
    return -2;
#else
    if (out_dim == 0 || batch == 0) return 0;
    if ((in_dim & 127u) != 0 || (out_dim & 127u) != 0 || (row_off & 127u) != 0)
        return -2;
    static const bool tma = [] {
        int dev = 0, cma = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&cma, cudaDevAttrComputeCapabilityMajor, dev);
        return cma >= 9 && pd_tmap_encode() != nullptr
            && pd_env("PADDOCK_NO_F8W8_TMA") == nullptr;
    }();
    if (!tma) return -2;
    const uint32_t smem4r = 98328u;
    static bool alin4dr = false;
    if (!alin4dr) {
        cudaFuncSetAttribute((const void*)pd_f8_gemm_lin_kt4d<false, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem4r);
        cudaFuncSetAttribute((const void*)pd_f8_gemm_lin_kt4d<true, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem4r);
        pd_prefer_max_shared(pd_f8_gemm_lin_kt4d<false, true>);
        pd_prefer_max_shared(pd_f8_gemm_lin_kt4d<true, true>);
        alin4dr = true;
    }
    CUtensorMap ym;
    if (!pd_tmap_2d(&ym, xq, in_dim, batch)) return -2;
    const uint32_t nkboxes = in_dim >> 7;
    const unsigned char* wp = (const unsigned char*)wlin
        + (size_t)(row_off >> 7) * nkboxes * PD_LINBS_BOX;
    const uint32_t bp = (batch + 127u) & ~127u;
    const uint32_t nt = (out_dim >> 7) * (bp >> 7);
    if (o16)
        pd_f8_gemm_lin_kt4d<true, true><<<nt, 288, smem4r, (cudaStream_t)stream>>>(
            wp, ym, (const unsigned char*)xs, (float*)y, (const float*)ws,
            in_dim, out_dim, batch);
    else
        pd_f8_gemm_lin_kt4d<false, true><<<nt, 288, smem4r, (cudaStream_t)stream>>>(
            wp, ym, (const unsigned char*)xs, (float*)y, (const float*)ws,
            in_dim, out_dim, batch);
    return pd_launch_status();
#endif
}

// fused qkv chunk launcher on the rowwise plane (kt4<RW,SEG>):
// one launch over the whole q‖k‖v plane, epilogue scattered to the three
// dense per-projection outputs. Bit-exact vs the three split launches (same
// mainloop per tile, same absolute boxes/ws rows); the win is grid economics
// - at admission-M the split grids are 1.5-3 waves at 1 CTA/SM and each
// pays its own ramp + straggler-wave chain; the 128-tile-wide fused grid
// pays one. Rowwise plane only (the strip pc lane keeps split launches).
static int pd_f8_gemm_w8_pc_qkv_impl(const void* wlin, const void* xq,
                                     const void* as_row, const void* ws,
                                     void* yq, void* yk, void* yv,
                                     uint32_t in_dim, uint32_t q_dim,
                                     uint32_t kv_dim, uint32_t batch,
                                     uint32_t o16, void* stream) {
#ifndef PD_BS_HOST
    (void)wlin; (void)xq; (void)as_row; (void)ws; (void)yq; (void)yk;
    (void)yv; (void)in_dim; (void)q_dim; (void)kv_dim; (void)batch;
    (void)o16; (void)stream;
    return -2;
#else
    if (q_dim == 0 || kv_dim == 0 || batch == 0) return 0;
    if ((in_dim & 127u) != 0 || (q_dim & 127u) != 0 || (kv_dim & 127u) != 0)
        return -2;
    static const bool tma = [] {
        int dev = 0, cma = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&cma, cudaDevAttrComputeCapabilityMajor, dev);
        return cma >= 9 && pd_tmap_encode() != nullptr
            && pd_env("PADDOCK_NO_F8W8_TMA") == nullptr;
    }();
    if (!tma) return -2;
    const uint32_t smem4r = 98328u;
    static bool alin4q = false;
    if (!alin4q) {
        cudaFuncSetAttribute((const void*)pd_f8_gemm_lin_kt4<false, true, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem4r);
        pd_prefer_max_shared(pd_f8_gemm_lin_kt4<false, true, true>);
        cudaFuncSetAttribute((const void*)pd_f8_gemm_lin_kt4<true, true, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem4r);
        pd_prefer_max_shared(pd_f8_gemm_lin_kt4<true, true, true>);
        alin4q = true;
    }
    CUtensorMap ym;
    if (!pd_tmap_2d(&ym, xq, in_dim, batch)) return -2;
    const uint32_t out_dim = q_dim + 2u * kv_dim;
    const uint32_t bp = (batch + 127u) & ~127u;
    const uint32_t nt = (out_dim >> 7) * (bp >> 7);
    if (o16)
        pd_f8_gemm_lin_kt4<true, true, true><<<nt, 288, smem4r, (cudaStream_t)stream>>>(
            (const unsigned char*)wlin, ym, (float*)yq, (const float*)as_row,
            (const float*)ws, in_dim, out_dim, batch, (float*)yk, (float*)yv,
            q_dim, kv_dim);
    else
        pd_f8_gemm_lin_kt4<false, true, true><<<nt, 288, smem4r, (cudaStream_t)stream>>>(
            (const unsigned char*)wlin, ym, (float*)yq, (const float*)as_row,
            (const float*)ws, in_dim, out_dim, batch, (float*)yk, (float*)yv,
            q_dim, kv_dim);
    return pd_launch_status();
#endif
}

PD_EXPORT
int pd_f8_gemm_w8_pc_qkv_r(const void* wlin, const void* xq,
                           const void* as_row, const void* ws, void* yq,
                           void* yk, void* yv, uint32_t in_dim,
                           uint32_t q_dim, uint32_t kv_dim, uint32_t batch,
                           void* stream) {
    return pd_f8_gemm_w8_pc_qkv_impl(wlin, xq, as_row, ws, yq, yk, yv,
                                     in_dim, q_dim, kv_dim, batch, 0u, stream);
}

// o16 twin (the chunk band's 16-bit output stream): identical mainloop
// and epilogue math, the final store converts to bf16 (kt4<O16,RW,SEG>).
// Appended as its own export per the ABI growth rule.
PD_EXPORT
int pd_f8_gemm_w8_pc_qkv_r2(const void* wlin, const void* xq,
                            const void* as_row, const void* ws, void* yq,
                            void* yk, void* yv, uint32_t in_dim,
                            uint32_t q_dim, uint32_t kv_dim, uint32_t batch,
                            uint32_t o16, void* stream) {
    return pd_f8_gemm_w8_pc_qkv_impl(wlin, xq, as_row, ws, yq, yk, yv,
                                     in_dim, q_dim, kv_dim, batch, o16, stream);
}



// ---- b=1 GEMV over the lin boxes ------------------------------------------
//
// Why this exists: the lin planes were built for the WIDTH GEMM, so the small
// bands kept reading the Q8_0 originals and the model carried both formats
// (~26 GB of duplicate residency on the 27B). pd_f8_gemv cannot read boxes,
// and a row-major f8w twin would cost the width GEMM ~11% (this file's header
// records the 1323-vs-1490 GB/s access-pattern fact). So the GEMV comes to
// the boxes instead.
//
// CTA (bt, z) owns a BM-row slice of row-tile rt = bt/(128/BM) and walks its
// boxes over K range [z*kpc, ...) - a contiguous byte range at box
// granularity, which is the per-CTA-sequential pattern the layout was made
// for. Three geometry facts, each measured worth ~5-8%:
//   - x is staged in WINDOWS of 48 boxes (24 KB smem), not per box: per-box
//     staging stutters the weight stream on __syncthreads.
//   - the K-split combine is a fused last-CTA ticket, not a second launch:
//     a combine launch costs a ~3 us floor, which was the whole gap at nz>1.
//     atomicInc wraps at nz-1 so the counter self-resets, and the combine
//     sums z in fixed order - same bytes every run, which serving requires.
//   - BM=64 (the layout's own contiguous half-boxes) beats BM=128 on skinny
//     planes; quarter-boxes add nothing.
// Rate on the FFN/head shapes: 1400-1519 GB/s, at or past the tuned Q8_0
// GEMV. The skinny PROJECTIONS stay on Q8 (only 40 row-tiles: 128-row box
// granularity starves the die where the Q8 kernel runs one CTA per output
// row).
#define PD_LINV_XWIN 48u
template <int THREADS, int BM>
__global__ void pd_f8lin_gemv_kernel(
        const unsigned char* __restrict__ boxes, const float* __restrict__ x,
        float* __restrict__ part, float* __restrict__ y,
        unsigned int* __restrict__ ticket, uint32_t in_dim, uint32_t out_dim,
        uint32_t kpc) {
#if PD_LIN_DEV_OK
    const uint32_t nk = in_dim >> 7;
    const uint32_t bt = blockIdx.x, z = blockIdx.y, nz = gridDim.y;
    const uint32_t sub = 128u / BM;
    const uint32_t rt = bt / sub, slice = bt % sub;
    const uint32_t k0 = z * kpc, k1 = min(nk, k0 + kpc);
    if (k0 >= k1) return;
    extern __shared__ float xs[];   // min(PD_LINV_XWIN, kpc) boxes
    const uint32_t tid = threadIdx.x;
    // THREADS == 2*BM: two lanes per row, each taking 4 of the 8 16 B chunks.
    // THREADS == BM: one lane per row, all 8.
    const uint32_t r = (THREADS == 2 * BM) ? (tid >> 1) : tid;
    const uint32_t c_lo = (THREADS == 2 * BM) ? ((tid & 1u) * 4u) : 0u;
    const uint32_t c_n = (THREADS == 2 * BM) ? 4u : 8u;
    const uint32_t rb = slice * BM + r;
    float acc = 0.0f;
    for (uint32_t w0 = k0; w0 < k1; w0 += PD_LINV_XWIN) {
        const uint32_t w1 = min(k1, w0 + PD_LINV_XWIN);
        __syncthreads();
        for (uint32_t i = tid; i < (w1 - w0) * 32u; i += THREADS)
            *reinterpret_cast<float4*>(xs + i * 4u) =
                *reinterpret_cast<const float4*>(x + w0 * 128u + i * 4u);
        __syncthreads();
        const unsigned char* box = boxes + (size_t)(rt * nk + w0) * PD_LIN_BOX;
        for (uint32_t kt = w0; kt < w1; ++kt, box += PD_LIN_BOX) {
            const uint32_t sw = *reinterpret_cast<const uint32_t*>(
                box + PD_LIN_DATA + rb * 4u);
            const float* xw = xs + (kt - w0) * 128u;
            #pragma unroll
            for (uint32_t ci = 0; ci < c_n; ++ci) {
                const uint32_t c = c_lo + ci;
                const int4 wv = *reinterpret_cast<const int4*>(
                    box + rb * 128u + ((c ^ (rb & 7u)) * 16u));
                const __nv_fp8_e4m3* wb =
                    reinterpret_cast<const __nv_fp8_e4m3*>(&wv);
                const float* xv = xw + c * 16u;
                float s = 0.0f;
                #pragma unroll
                for (uint32_t j = 0; j < 16u; ++j) s += (float)wb[j] * xv[j];
                acc += __int_as_float(((int)((sw >> ((c >> 1) * 8u)) & 0xffu)) << 23) * s;
            }
        }
    }
    const uint32_t row = rt * 128u + rb;
    // nz==1 has no combine, so the rows are the answer and go straight to y -
    // the kernel elects the split, so the kernel owns the destination. (A
    // caller that passed part=y hid this: the engine passes real scratch, and
    // every nz==1 plane then returned stale memory. Caught in serving, not
    // by a unit test.)
    float* __restrict__ dst = (nz == 1) ? y : part;
    if (THREADS == 2 * BM) {
        acc += __shfl_xor_sync(0xffffffffu, acc, 1);
        if ((tid & 1u) == 0 && row < out_dim)
            dst[(size_t)z * out_dim + row] = acc;
    } else if (row < out_dim) {
        dst[(size_t)z * out_dim + row] = acc;
    }
    if (nz == 1) return;
    __threadfence();
    __shared__ unsigned int last;
    __syncthreads();
    if (tid == 0) last = atomicInc(&ticket[bt], nz - 1u);
    __syncthreads();
    if (last != nz - 1u) return;
    for (uint32_t rr = tid; rr < BM; rr += THREADS) {
        const uint32_t o = rt * 128u + slice * BM + rr;
        if (o >= out_dim) continue;
        float v = 0.0f;
        for (uint32_t zz = 0; zz < nz; ++zz) v += part[(size_t)zz * out_dim + o];
        y[o] = v;
    }
#else
    (void)boxes; (void)x; (void)part; (void)y; (void)ticket;
    (void)in_dim; (void)out_dim; (void)kpc;
#endif
}

// `part` needs nz * out_dim floats; `ticket` needs 2*ceil(out_dim/128) u32,
// ZEROED once at allocation and then owned by launches of this shape (the
// wrap value is nz, so a buffer must not be shared across differing nz).
// ticket == NULL pins nz = 1 and `part` may alias `y`.
PD_EXPORT
int pd_f8lin_gemv(const void* wlin, const void* x, void* part, void* y,
                  void* ticket, uint32_t in_dim, uint32_t out_dim,
                  void* stream) {
    if (out_dim == 0) return 0;
    if ((in_dim & 127u) || (out_dim & 15u)) return cudaErrorInvalidValue;
    static int nsm = 0;
    if (!nsm) {
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) return cudaErrorInvalidValue;
        if (cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev)
            != cudaSuccess)
            return cudaErrorInvalidValue;
    }
    const uint32_t nk = in_dim >> 7, nrt = (out_dim + 127u) / 128u;
    // BM=64 always (measured >= BM=128 on every shape that isn't starved, and
    // strictly better when starved).
    const uint32_t ctas = nrt * 2u;
    // K-split election. Three rules were measured end-to-end in serving on
    // qwen3.8-27B c1 (state-matched A/Bs) and the ranking is monotonic in
    // less splitting:
    //     CTA/SM floor (nz 1..5)   fastest   (-0.84% vs the Q8 chain)
    //     wave fill    (nz 5..7)             (-1.1%)
    //     max split    (nz = nk)             (-1.4%)
    // So the split is not free: partials land in the shared ks scratch (71 MB
    // on this model, also used by the attention/DN ks GEMMs), and every extra
    // z evicts more of the other lane's working set than the added parallelism
    // buys. Keep the split to what the die actually needs to be busy.
    //
    // NOTE this does not close the gap to the Q8 chain, and geometry is not
    // why: the Q8 GEMV overlaps its neighbours (norm 55.8
    // ms, swiglu 19.7, itself 16.2 per profiled window) and this kernel
    // overlaps nothing at any of the three geometries. That is a structural
    // property of one fused call replacing two independent ones, not a
    // tuning parameter.
    uint32_t nz = 1;
    if (ticket) {
        const uint32_t want = (2u * (uint32_t)nsm + ctas - 1u) / ctas;
        nz = want < 1u ? 1u : (want > 16u ? 16u : want);
        if (nz > nk) nz = nk;
    }
    const uint32_t kpc = (nk + nz - 1u) / nz;
    // every z plane must own a non-empty K range or the ticket never fills
    const uint32_t nz_eff = (nk + kpc - 1u) / kpc;
    dim3 grid(ctas, nz_eff);
    const uint32_t win = kpc < PD_LINV_XWIN ? kpc : PD_LINV_XWIN;
    const uint32_t shm = win * 128u * (uint32_t)sizeof(float);
    pd_f8lin_gemv_kernel<128, 64><<<grid, 128, shm, (cudaStream_t)stream>>>(
        (const unsigned char*)wlin, (const float*)x, (float*)part, (float*)y,
        (unsigned int*)ticket, in_dim, out_dim, kpc);
    return pd_launch_status();
}
