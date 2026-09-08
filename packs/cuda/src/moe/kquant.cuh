// moe/kquant.cuh (formerly 20_kquant_moe.cuh) - k-quant routed-expert MoE, token-batched (decode class)
// Textually-included segment of the single pack translation unit.
// Not standalone-compilable: include order is defined by ../pack.cu.
//
// The W4A8 arm for MoE expert seats (qwen3.6-A3B class): expert weights stay
// in the repacked k-quant streams (quant/kquant.cuh layouts, 4-6.6 bpw) instead of
// being requantized to Q8_0 at load - ~0.55x the expert DRAM per decode step
// on this weight-bandwidth-bound FFN, and ~half the expert VRAM. Kernel
// shapes clone the Q8 token-batched pair (moe/q8.cuh): grid (ff, n_active, B)
// for gate+up+SwiGLU, grid (embd, B) warp-per-slot for down+combine; the
// per-lane 16-weight window unpack is the dense dp4a MT kernel's
// (quant/kquant_w4a8.cuh) - one window = one k-quant sub-block half, one (scale,
// mu) pair per window, Q6_K's per-16 scales native.
//
// Numeric class: identical to the dense k-quant batched ladder and the Q8
// MoE pair - exact int8 dots, f32 per-block scale application; the Q4_K/Q5_K
// mu term rides per-16 activation sums (pd_q8_sums_strided).
//
// Two classes in this file:
// - token-batched pair (below): the decode class - grid fills the die from
//   B=1, but re-reads routed expert rows per token, hopeless at prefill.
// - sorted mma pair (pd_kquant_moe_*_mma, end of file): the prefill/serving
//   class - moe_align blocks read each touched expert's weights once per
//   pass, tensor-core int8 mma straight off the RAW k-quant strips (the ks
//   v2 ring + inline nibble unpack from quant/kquant_w4a8.cuh, Marlin's design
//   point). The engine picks sorted past the same pair-count boundary the
//   Q8 seats use.

// pd_kq_datab / pd_kq_win_unpack (the per-lane 16-weight window unpack this
// file's token-batched pair is built on) moved to quant/kquant_w4a8.cuh - the
// W4A8 b=1 GEMV there consumes them too, and 19 precedes this file.

// Fused gate+up+SwiGLU over routed k-quant experts, token-batched: grid
// (ff, n_active, batch), one block per (out row, slot, token) - the Q8
// pair's geometry (512 x 8 x B blocks fills the die from B=1). Weight row
// for (expert e, out o) sits at (e*ff + o) in the repacked stream. gate and
// up may be different k-quant types (UD files mix per-tensor).
//
// The block is exactly as wide as the row has 16-weight windows (see the
// launcher): every thread walks one window, so a 256-thread launch is only
// right when in_dim >= 4096. Laguna's in_dim is 2048 - half the block used
// to walk zero windows and still pay the __syncthreads and the tid-0 fold.
// LIST: the expert-major prefill's form - grid (ff, P) with a fixed P, each
// block striding over a device-counted list of (token*n_active + slot) pair
// indices (the wave's in-wave pairs, moe/offload.cuh's mask+compact). The
// plain form's grid enumerates every pair, and on a 2000-token prompt that
// is billions of blocks per prefill exiting on the absent check; the list
// form launches P blocks per output row and does only the wave's work.
// Same body, same per-pair math and fold order (a pair's output does not
// depend on which block computed it).
template <bool LIST>
__global__ void __launch_bounds__(256) pd_kquant_moe_gate_up_kernel(
    // cascade (laguna chain): xq/idx are the quantize and
    // topk predecessors' outputs - armed at top, launched via pd_pdl_go
    const uint8_t* __restrict__ gd, const uint8_t* __restrict__ gsc,
    const uint8_t* __restrict__ ud_, const uint8_t* __restrict__ usc,
    const unsigned int* __restrict__ idx, const int8_t* __restrict__ xq,
    const float* __restrict__ xs, const float* __restrict__ xsums,
    float* __restrict__ out, uint32_t in_dim, uint32_t ff, uint32_t n_active,
    uint32_t gdt, uint32_t udt, const unsigned int* __restrict__ list,
    const unsigned int* __restrict__ n_list) {
    PD_PDL_ARM();
    // PLAIN grid is (batch, n_active, ff): the out row is the SLOW axis, so a
    // row's whole pair set dispatches together. What the grouped kernel below
    // left this one is the DECODE band (routed rows <= n_expert), and that is
    // where the order pays - measured on Flash-Next IQ3_XXS 2026-09-07, same
    // pack otherwise: c32 260.2 vs 248.0 out_tok/s, imax 277.6 vs 244.0. At
    // prefill widths it was neutral either way (the grouped kernel serves
    // those now). Index mapping only: same block, same math, same fold,
    // bit-identical output.
    const uint32_t o = LIST ? blockIdx.x : blockIdx.z;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    const uint32_t p_hi = LIST ? *n_list : 1u;
    for (uint32_t p = LIST ? blockIdx.y : 0u; p < p_hi; p += LIST ? gridDim.y : 1u) {
    uint32_t slot, b;
    if (LIST) {
        const uint32_t pr = list[p];
        b = pr / n_active;
        slot = pr - b * n_active;
    } else {
        slot = blockIdx.y;
        b = blockIdx.x;
    }
    const uint32_t e = idx[(size_t)b * n_active + slot];
    // Absent pair (the expert-major prefill's out-of-wave sentinel,
    // PD_MOE_CACHE_NONE from moe/offload.cuh): this block has no expert to
    // read. Leave `out` alone - the down kernel skips the same pair - and
    // skip before anything block-collective; e is block-uniform so the
    // skip is too. Measured reason: without it a wave's cost scaled with
    // EVERY routed pair (10 ms a launch on a 435-token prompt), not its own.
    if (e == 0xFFFFFFFFu) continue;
    const uint32_t gdb = pd_kq_datab(gdt), udb = pd_kq_datab(udt);
    const uint32_t gscb = pd_kq_scb(gdt), uscb = pd_kq_scb(udt);
    // row strides by type: whole superblocks, or IQ4_NL's flat 32-block rows
    // (a row that is not a whole number of superblocks carries no padding)
    const uint8_t* grow = gd + ((size_t)e * ff + o) * pd_kq_row_datab(gdt, in_dim);
    const uint8_t* grec = gsc + ((size_t)e * ff + o) * pd_kq_row_scb(gdt, in_dim);
    const uint8_t* urow = ud_ + ((size_t)e * ff + o) * pd_kq_row_datab(udt, in_dim);
    const uint8_t* urec = usc + ((size_t)e * ff + o) * pd_kq_row_scb(udt, in_dim);
    const int8_t* xrow = xq + (size_t)b * in_dim;
    const float* xsc = xs + (size_t)b * (in_dim >> 5);
    const float* xsm = xsums + (size_t)b * (in_dim >> 4);
    const bool gmu = pd_kq_has_mu(gdt);
    const bool umu = pd_kq_has_mu(udt);

    float accg = 0.0f, accu = 0.0f;
    for (uint32_t base = tid * 16u; base < in_dim; base += nth * 16u) {
        const uint32_t s = base >> 8, w = (base >> 4) & 15u;
        const float x_s = xsc[base >> 5];
        const int4 xv = *reinterpret_cast<const int4*>(xrow + base);
        int wq[4];
        float f, g;
        pd_kq_win_unpack(gdt, grow + (size_t)s * gdb,
                         grec + (size_t)s * gscb, w, wq, &f, &g);
        int si = __dp4a(wq[0], xv.x, 0);
        si = __dp4a(wq[1], xv.y, si);
        si = __dp4a(wq[2], xv.z, si);
        si = __dp4a(wq[3], xv.w, si);
        accg += f * (x_s * (float)si);
        if (gmu) accg += g * (x_s * xsm[base >> 4]);
        pd_kq_win_unpack(udt, urow + (size_t)s * udb,
                         urec + (size_t)s * uscb, w, wq, &f, &g);
        si = __dp4a(wq[0], xv.x, 0);
        si = __dp4a(wq[1], xv.y, si);
        si = __dp4a(wq[2], xv.z, si);
        si = __dp4a(wq[3], xv.w, si);
        accu += f * (x_s * (float)si);
        if (umu) accu += g * (x_s * xsm[base >> 4]);
    }
    __shared__ float wsum[2][8];
    const uint32_t lane = tid & 31u, warp = tid >> 5, nwarps = (nth + 31u) >> 5;
    for (uint32_t s2 = 16; s2 > 0; s2 >>= 1) {
        accg += __shfl_down_sync(0xffffffffu, accg, s2);
        accu += __shfl_down_sync(0xffffffffu, accu, s2);
    }
    if (lane == 0) { wsum[0][warp] = accg; wsum[1][warp] = accu; }
    __syncthreads();
    if (tid == 0) {
        float g = 0.0f, u = 0.0f;
        for (uint32_t w = 0; w < nwarps; ++w) { g += wsum[0][w]; u += wsum[1][w]; }
        // silu(g) * u - same epilogue as the Q8 pair
        out[((size_t)b * n_active + slot) * ff + o] = (g / (1.0f + __expf(-g))) * u;
    }
    // LIST: wsum is reused by the next pair this block takes - tid 0 must be
    // done reading it before any lane writes again
    if (LIST) __syncthreads();
    }
}

PD_EXPORT
int pd_kquant_moe_gate_up(const void* gate_data, const void* gate_scales,
                          const void* up_data, const void* up_scales,
                          const void* idx, const void* xq, const void* xs,
                          const void* xsums, void* out, uint32_t in_dim,
                          uint32_t ff, uint32_t n_active, uint32_t batch,
                          uint32_t gdt, uint32_t udt, void* stream) {
    if (ff == 0 || n_active == 0 || batch == 0) return 0;
    if ((in_dim & 31u) != 0) return cudaErrorInvalidValue;
    if ((in_dim & 255u) != 0 && !(gdt == PD_KQ_IQ4NL_ID && udt == PD_KQ_IQ4NL_ID))
        return cudaErrorInvalidValue;   // partial superblocks: IQ4_NL's flat rows only
    if (!(pd_kq_valid(gdt) || pd_kq_valid_iq(gdt)) || !(pd_kq_valid(udt) || pd_kq_valid_iq(udt)))
        return cudaErrorInvalidValue;
    const bool mu = pd_kq_has_mu(gdt) ||
                    pd_kq_has_mu(udt);
    if (mu && xsums == nullptr) return cudaErrorInvalidValue;
    dim3 grid(batch, n_active, ff);   // out row SLOW - see the kernel note
    // Block width = one thread per 16-weight window, warp-rounded, capped at
    // 256. BIT-EXACT against the old flat-256 launch: a thread's window is
    // tid*16 either way, so the surviving threads hold the same partial in the
    // same lane, the 32-lane tree is the same tree, and the warps that drop
    // out contributed exactly 0.0f to the tid-0 fold. Measured on the laguna
    // XS-2.1 decode shape (in 2048, ff 512, top-8 of 256 experts, sm_86):
    // 25.66 -> 19.90 us, 388 -> 501 GB/s, maxrel 0.
    // A warp-per-row rewrite (one lane walking 4 windows, no smem, no
    // syncthreads) was benched alongside and lands in the same place (19.33)
    // while costing a summation-order change - not worth the parity vetting.
    uint32_t nth = (in_dim >> 4) < 256u ? (((in_dim >> 4) + 31u) & ~31u) : 256u;
    if (nth < 32u) nth = 32u;
    pd_pdl_go(pd_kquant_moe_gate_up_kernel<false>, grid, nth, 0u, (cudaStream_t)stream,
        (const uint8_t*)gate_data, (const uint8_t*)gate_scales,
        (const uint8_t*)up_data, (const uint8_t*)up_scales,
        (const unsigned int*)idx, (const int8_t*)xq, (const float*)xs,
        (const float*)xsums, (float*)out, in_dim, ff, n_active, gdt, udt,
        (const unsigned int*)nullptr, (const unsigned int*)nullptr);
    return pd_launch_status();
}

// slot 583: the LIST form (see the kernel note). `pairs`/`n_pairs` are the
// wave's compacted in-wave pair indices and their device count; the grid is
// (ff, PD_KQ_MOE_LIST_P) regardless of the count.
#define PD_KQ_MOE_LIST_P 96u
PD_EXPORT
int pd_kquant_moe_gate_up_list(const void* gate_data, const void* gate_scales,
                               const void* up_data, const void* up_scales,
                               const void* idx, const void* xq, const void* xs,
                               const void* xsums, void* out, uint32_t in_dim,
                               uint32_t ff, uint32_t n_active, uint32_t batch,
                               uint32_t gdt, uint32_t udt, const void* pairs,
                               const void* n_pairs, void* stream) {
    if (ff == 0 || n_active == 0 || batch == 0) return 0;
    if ((in_dim & 31u) != 0) return cudaErrorInvalidValue;
    if ((in_dim & 255u) != 0 && !(gdt == PD_KQ_IQ4NL_ID && udt == PD_KQ_IQ4NL_ID))
        return cudaErrorInvalidValue;
    if (!(pd_kq_valid(gdt) || pd_kq_valid_iq(gdt)) || !(pd_kq_valid(udt) || pd_kq_valid_iq(udt)))
        return cudaErrorInvalidValue;
    if ((pd_kq_has_mu(gdt) || pd_kq_has_mu(udt)) && xsums == nullptr) return cudaErrorInvalidValue;
    if (pairs == nullptr || n_pairs == nullptr) return cudaErrorInvalidValue;
    dim3 grid(ff, PD_KQ_MOE_LIST_P, 1u);
    uint32_t nth = (in_dim >> 4) < 256u ? (((in_dim >> 4) + 31u) & ~31u) : 256u;
    if (nth < 32u) nth = 32u;
    pd_pdl_go(pd_kquant_moe_gate_up_kernel<true>, grid, nth, 0u, (cudaStream_t)stream,
        (const uint8_t*)gate_data, (const uint8_t*)gate_scales,
        (const uint8_t*)up_data, (const uint8_t*)up_scales,
        (const unsigned int*)idx, (const int8_t*)xq, (const float*)xs,
        (const float*)xsums, (float*)out, in_dim, ff, n_active, gdt, udt,
        (const unsigned int*)pairs, (const unsigned int*)n_pairs);
    return pd_launch_status();
}

// ---- expert-grouped gate+up (the prefill class) ----------------------------
// The pair kernel above unpacks a weight window per ROUTED ROW: at 512
// experts x top-10 a 200-token prompt routes ~3.9 rows to each expert, so
// every expert's rows were unpacked 3.9 times over. ncu on the shipped pair
// kernel: SM 52% of peak against DRAM 23% - i-quant UNPACK, not bandwidth, is
// what a prefill spends its MoE time on, and a grid transpose (L2 reuse) moved
// it 0%. Here a block owns (expert group, out row) from the moe_align layout:
// each thread unpacks its 16-weight window ONCE and walks the group's up-to-T
// routed rows against it. T=8 covers a whole expert at this density (one block
// per expert), so the unpack cost falls by the rows-per-expert factor while the
// dp4a work - which is per row either way - is unchanged.
//
// BIT-IDENTICAL to the pair kernel: same per-window math in the same order,
// same 32-lane shuffle tree per row, same ascending-warp fold. A row's output
// does not depend on which block computed it, and PAD lanes write nothing.
template <uint32_t T>
__global__ void __launch_bounds__(256, 4) pd_kquant_moe_gate_up_grp_kernel(
    const uint8_t* __restrict__ gd, const uint8_t* __restrict__ gsc,
    const uint8_t* __restrict__ ud_, const uint8_t* __restrict__ usc,
    const unsigned int* __restrict__ sorted_row,
    const unsigned int* __restrict__ sorted_slot,
    const unsigned int* __restrict__ block_expert, const int8_t* __restrict__ xq,
    const float* __restrict__ xs, const float* __restrict__ xsums,
    float* __restrict__ out, uint32_t in_dim, uint32_t ff, uint32_t n_active,
    uint32_t gdt, uint32_t udt) {
    PD_PDL_ARM();
    const uint32_t blk = blockIdx.x, o = blockIdx.y;
    const uint32_t e = block_expert[blk];
    if (e == PD_MOE_PAD) return;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;

    __shared__ unsigned int srow_sh[T], sslot_sh[T];
    if (tid < T) {
        srow_sh[tid] = sorted_row[(size_t)blk * T + tid];
        sslot_sh[tid] = sorted_slot[(size_t)blk * T + tid];
    }
    __syncthreads();

    const uint32_t gdb = pd_kq_datab(gdt), udb = pd_kq_datab(udt);
    const uint32_t gscb = pd_kq_scb(gdt), uscb = pd_kq_scb(udt);
    const uint8_t* grow = gd + ((size_t)e * ff + o) * pd_kq_row_datab(gdt, in_dim);
    const uint8_t* grec = gsc + ((size_t)e * ff + o) * pd_kq_row_scb(gdt, in_dim);
    const uint8_t* urow = ud_ + ((size_t)e * ff + o) * pd_kq_row_datab(udt, in_dim);
    const uint8_t* urec = usc + ((size_t)e * ff + o) * pd_kq_row_scb(udt, in_dim);
    const bool gmu = pd_kq_has_mu(gdt);
    const bool umu = pd_kq_has_mu(udt);

    // The group's rows read `xq` straight from global: staging them in shared
    // was measured (2026-09-08) and LOSES here - at in_dim 2560 a 16-row group
    // is 46 KB of shared, which costs more occupancy than the re-reads cost
    // traffic (8.97 -> 11.24 ms a layer at a 2114-row wave). The down half,
    // whose rows are ff = 640 wide, stages and wins.
    float accg[T], accu[T];
    #pragma unroll
    for (uint32_t i = 0; i < T; ++i) { accg[i] = 0.0f; accu[i] = 0.0f; }

    for (uint32_t base = tid * 16u; base < in_dim; base += nth * 16u) {
        const uint32_t s = base >> 8, w = (base >> 4) & 15u;
        int wqg[4], wqu[4];
        float fg, gg, fu, gu;
        pd_kq_win_unpack(gdt, grow + (size_t)s * gdb, grec + (size_t)s * gscb, w, wqg, &fg, &gg);
        pd_kq_win_unpack(udt, urow + (size_t)s * udb, urec + (size_t)s * uscb, w, wqu, &fu, &gu);
        for (uint32_t i = 0; i < T; ++i) {
            const unsigned int b = srow_sh[i];
            if (b == PD_MOE_PAD) continue;
            const int4 xv = *reinterpret_cast<const int4*>(xq + (size_t)b * in_dim + base);
            const float x_s = xs[(size_t)b * (in_dim >> 5) + (base >> 5)];
            int si = __dp4a(wqg[0], xv.x, 0);
            si = __dp4a(wqg[1], xv.y, si);
            si = __dp4a(wqg[2], xv.z, si);
            si = __dp4a(wqg[3], xv.w, si);
            accg[i] += fg * (x_s * (float)si);
            if (gmu) accg[i] += gg * (x_s * xsums[(size_t)b * (in_dim >> 4) + (base >> 4)]);
            si = __dp4a(wqu[0], xv.x, 0);
            si = __dp4a(wqu[1], xv.y, si);
            si = __dp4a(wqu[2], xv.z, si);
            si = __dp4a(wqu[3], xv.w, si);
            accu[i] += fu * (x_s * (float)si);
            if (umu) accu[i] += gu * (x_s * xsums[(size_t)b * (in_dim >> 4) + (base >> 4)]);
        }
    }

    __shared__ float wsum[2][T][8];
    const uint32_t lane = tid & 31u, warp = tid >> 5, nwarps = (nth + 31u) >> 5;
    for (uint32_t i = 0; i < T; ++i) {
        float g = accg[i], u = accu[i];
        for (uint32_t s2 = 16; s2 > 0; s2 >>= 1) {
            g += __shfl_down_sync(0xffffffffu, g, s2);
            u += __shfl_down_sync(0xffffffffu, u, s2);
        }
        if (lane == 0) { wsum[0][i][warp] = g; wsum[1][i][warp] = u; }
    }
    __syncthreads();
    if (tid < T) {
        const unsigned int b = srow_sh[tid];
        if (b != PD_MOE_PAD) {
            float g = 0.0f, u = 0.0f;
            for (uint32_t w2 = 0; w2 < nwarps; ++w2) {
                g += wsum[0][tid][w2];
                u += wsum[1][tid][w2];
            }
            out[((size_t)b * n_active + sslot_sh[tid]) * ff + o] =
                (g / (1.0f + __expf(-g))) * u;
        }
    }
}

// slot 586: the grouped form. `sorted_row`/`sorted_slot`/`block_expert` are a
// pd_moe_align_bm(bm = group) layout over the SAME idx the pair kernel reads;
// `group` is 8, 16 or 32 (the caller elects it from rows/n_expert - a whole
// expert per block is the point). Output layout is the pair kernel's, so the
// down kernel and the quantize between them are unchanged.
PD_EXPORT
int pd_kquant_moe_gate_up_grp(const void* gate_data, const void* gate_scales,
                              const void* up_data, const void* up_scales,
                              const void* sorted_row, const void* sorted_slot,
                              const void* block_expert, const void* xq,
                              const void* xs, const void* xsums, void* out,
                              uint32_t in_dim, uint32_t ff, uint32_t n_active,
                              uint32_t max_blocks, uint32_t group, uint32_t gdt,
                              uint32_t udt, void* stream) {
    if (ff == 0 || n_active == 0 || max_blocks == 0) return 0;
    if ((in_dim & 31u) != 0) return cudaErrorInvalidValue;
    if ((in_dim & 255u) != 0 && !(gdt == PD_KQ_IQ4NL_ID && udt == PD_KQ_IQ4NL_ID))
        return cudaErrorInvalidValue;
    if (!(pd_kq_valid(gdt) || pd_kq_valid_iq(gdt)) || !(pd_kq_valid(udt) || pd_kq_valid_iq(udt)))
        return cudaErrorInvalidValue;
    if ((pd_kq_has_mu(gdt) || pd_kq_has_mu(udt)) && xsums == nullptr)
        return cudaErrorInvalidValue;
    if (sorted_row == nullptr || sorted_slot == nullptr || block_expert == nullptr)
        return cudaErrorInvalidValue;
    uint32_t nth = (in_dim >> 4) < 256u ? (((in_dim >> 4) + 31u) & ~31u) : 256u;
    if (nth < 32u) nth = 32u;
    dim3 grid(max_blocks, ff);
#define PD_KQ_GRP_GO(TV)                                                       \
    pd_pdl_go(pd_kquant_moe_gate_up_grp_kernel<TV>, grid, nth, 0u,             \
        (cudaStream_t)stream, (const uint8_t*)gate_data,                       \
        (const uint8_t*)gate_scales, (const uint8_t*)up_data,                  \
        (const uint8_t*)up_scales, (const unsigned int*)sorted_row,            \
        (const unsigned int*)sorted_slot, (const unsigned int*)block_expert,   \
        (const int8_t*)xq, (const float*)xs, (const float*)xsums, (float*)out, \
        in_dim, ff, n_active, gdt, udt)
    switch (group) {
        case 8u: PD_KQ_GRP_GO(8u); break;
        case 16u: PD_KQ_GRP_GO(16u); break;
        case 32u: PD_KQ_GRP_GO(32u); break;
        default: return cudaErrorInvalidValue;
    }
#undef PD_KQ_GRP_GO
    return pd_launch_status();
}

// Routed k-quant down + weighted combine: out[b][o] = sum_slot topk_w *
// dot(down[e][o], fused_q[b][slot]). grid (embd, batch); warp w owns slot w
// (n_active <= 16, launcher sizes the block to 32*n_active), lanes stride ff
// in 16-byte windows - at ff = 512 one pass covers it. Plain write (caller
// adds shared expert + residual). 16 matches pd_moe_topk_warp's existing
// top-k ceiling (sel_logit[16]) - was hard-capped at 8 (XS-2.1's top-8);
// Laguna S-2.1's top-10 MoE hit the cap.
// LIST: grid (embd, R), each block striding over a device-counted list of
// TOKENS that hold at least one in-wave pair; the token's absent pairs
// contribute zero and the block ACCUMULATES into out (one writer per
// element per wave, waves in sequence - deterministic). See the gate_up note.
template <bool LIST>
__global__ void __launch_bounds__(512) pd_kquant_moe_down_kernel(
    const uint8_t* __restrict__ dd, const uint8_t* __restrict__ dsc,
    const unsigned int* __restrict__ idx, const float* __restrict__ topk_w,
    const int8_t* __restrict__ fq, const float* __restrict__ fs,
    const float* __restrict__ fsums, float* __restrict__ out, uint32_t ff,
    uint32_t embd, uint32_t n_active, uint32_t ddt,
    const unsigned int* __restrict__ rows_list, const unsigned int* __restrict__ n_rows) {
    // cascade: fq/topk_w are the gate_up-quantize and topk outputs
    PD_PDL_ARM();
    // PLAIN grid is (batch, embd) for the same reason as the gate_up pair
    // above: the output column is the slow axis, and what this kernel still
    // serves is the narrow decode band.
    const uint32_t o = LIST ? blockIdx.x : blockIdx.y;
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const uint32_t ddb = pd_kq_datab(ddt);
    const bool mu = pd_kq_has_mu(ddt);
    __shared__ float sh[16];
    const uint32_t i_hi = LIST ? *n_rows : 1u;
    for (uint32_t i = LIST ? blockIdx.y : 0u; i < i_hi; i += LIST ? gridDim.y : 1u) {
    const uint32_t b = LIST ? rows_list[i] : blockIdx.x;
    if (warp < n_active) {
        const size_t srow = (size_t)b * n_active + warp;
        const uint32_t e = idx[srow];
        if (e == 0xFFFFFFFFu) {
            // absent pair (expert-major prefill sentinel): contributes zero
            if (lane == 0) sh[warp] = 0.0f;
        } else {
        const uint32_t dscb = pd_kq_scb(ddt);
        // row strides by type (IQ4_NL: flat 32-block rows, no padding)
        const uint8_t* row = dd + ((size_t)e * embd + o) * pd_kq_row_datab(ddt, ff);
        const uint8_t* rrec = dsc + ((size_t)e * embd + o) * pd_kq_row_scb(ddt, ff);
        const int8_t* xrow = fq + srow * ff;
        const float* xsc = fs + srow * (ff >> 5);
        const float* xsm = fsums + srow * (ff >> 4);
        float acc = 0.0f;
        for (uint32_t base = lane * 16u; base < ff; base += 32u * 16u) {
            const uint32_t s = base >> 8, w = (base >> 4) & 15u;
            const int4 xv = *reinterpret_cast<const int4*>(xrow + base);
            int wq[4];
            float f, g;
            pd_kq_win_unpack(ddt, row + (size_t)s * ddb,
                             rrec + (size_t)s * dscb, w, wq, &f, &g);
            int si = __dp4a(wq[0], xv.x, 0);
            si = __dp4a(wq[1], xv.y, si);
            si = __dp4a(wq[2], xv.z, si);
            si = __dp4a(wq[3], xv.w, si);
            const float x_s = xsc[base >> 5];
            acc += f * (x_s * (float)si);
            if (mu) acc += g * (x_s * xsm[base >> 4]);
        }
        for (uint32_t s2 = 16; s2 > 0; s2 >>= 1)
            acc += __shfl_down_sync(0xffffffffu, acc, s2);
        if (lane == 0) sh[warp] = topk_w[srow] * acc;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float v = 0.0f;
        for (uint32_t w = 0; w < n_active; ++w) v += sh[w];
        if (LIST) out[(size_t)b * embd + o] += v;
        else out[(size_t)b * embd + o] = v;
    }
    if (LIST) __syncthreads();   // sh is reused by this block's next token
    }
}

PD_EXPORT
int pd_kquant_moe_down(const void* down_data, const void* down_scales,
                       const void* idx, const void* topk_w, const void* fq,
                       const void* fs, const void* fsums, void* out,
                       uint32_t ff, uint32_t embd, uint32_t n_active,
                       uint32_t batch, uint32_t ddt, void* stream) {
    if (embd == 0 || n_active == 0 || batch == 0) return 0;
    if ((ff & 31u) != 0 || n_active > 16u) return cudaErrorInvalidValue;
    if ((ff & 255u) != 0 && ddt != PD_KQ_IQ4NL_ID)
        return cudaErrorInvalidValue;   // partial superblocks: IQ4_NL's flat rows only
    if (!pd_kq_valid(ddt) && !pd_kq_valid_iq(ddt)) return cudaErrorInvalidValue;
    if ((pd_kq_has_mu(ddt)) && fsums == nullptr)
        return cudaErrorInvalidValue;
    dim3 grid(batch, embd);   // column SLOW - see the kernel note
    pd_pdl_go(pd_kquant_moe_down_kernel<false>, grid, 32u * n_active, 0u, (cudaStream_t)stream,
        (const uint8_t*)down_data, (const uint8_t*)down_scales,
        (const unsigned int*)idx, (const float*)topk_w, (const int8_t*)fq,
        (const float*)fs, (const float*)fsums, (float*)out, ff, embd, n_active,
        ddt, (const unsigned int*)nullptr, (const unsigned int*)nullptr);
    return pd_launch_status();
}

// ---- column-tiled down (the prefill class, slot 587) -----------------------
// The plain down kernel above computes ONE output float per block: 10 warps
// each walk their slot's 640-weight row with 1.25 windows per lane, then the
// block folds and writes. ncu: SM 32% of peak, DRAM 21%, 57% warps active -
// nothing is saturated, the block is waiting on its own dependent loads with
// no other work to hide them. Here a block owns COLS columns instead of one:
// the activation window is loaded ONCE and reused across the COLS weight rows,
// so each lane carries COLS independent dot chains and the row loads overlap.
//
// BIT-IDENTICAL to the plain kernel: per (token, column, slot) the same
// lane-strided window walk in the same order, the same 32-lane tree, and the
// same ascending slot fold. Columns are independent, so grouping them changes
// no sum.
template <uint32_t COLS>
__global__ void __launch_bounds__(512) pd_kquant_moe_down_cols_kernel(
    const uint8_t* __restrict__ dd, const uint8_t* __restrict__ dsc,
    const unsigned int* __restrict__ idx, const float* __restrict__ topk_w,
    const int8_t* __restrict__ fq, const float* __restrict__ fs,
    const float* __restrict__ fsums, float* __restrict__ out, uint32_t ff,
    uint32_t embd, uint32_t n_active, uint32_t ddt) {
    PD_PDL_ARM();
    const uint32_t b = blockIdx.x;
    const uint32_t o0 = blockIdx.y * COLS;
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const uint32_t ddb = pd_kq_datab(ddt), dscb = pd_kq_scb(ddt);
    const bool mu = pd_kq_has_mu(ddt);
    __shared__ float sh[COLS][16];
    if (warp < n_active) {
        const size_t srow = (size_t)b * n_active + warp;
        const uint32_t e = idx[srow];
        float acc[COLS];
        #pragma unroll
        for (uint32_t c = 0; c < COLS; ++c) acc[c] = 0.0f;
        if (e != 0xFFFFFFFFu) {
            const int8_t* xrow = fq + srow * ff;
            const float* xsc = fs + srow * (ff >> 5);
            const float* xsm = fsums + srow * (ff >> 4);
            for (uint32_t base = lane * 16u; base < ff; base += 32u * 16u) {
                const uint32_t s = base >> 8, w = (base >> 4) & 15u;
                const int4 xv = *reinterpret_cast<const int4*>(xrow + base);
                const float x_s = xsc[base >> 5];
                const float x_m = mu ? xsm[base >> 4] : 0.0f;
                #pragma unroll
                for (uint32_t c = 0; c < COLS; ++c) {
                    const uint32_t o = o0 + c;
                    if (o >= embd) continue;
                    const uint8_t* row = dd + ((size_t)e * embd + o) * pd_kq_row_datab(ddt, ff);
                    const uint8_t* rrec = dsc + ((size_t)e * embd + o) * pd_kq_row_scb(ddt, ff);
                    int wq[4];
                    float f, g;
                    pd_kq_win_unpack(ddt, row + (size_t)s * ddb, rrec + (size_t)s * dscb, w, wq, &f, &g);
                    int si = __dp4a(wq[0], xv.x, 0);
                    si = __dp4a(wq[1], xv.y, si);
                    si = __dp4a(wq[2], xv.z, si);
                    si = __dp4a(wq[3], xv.w, si);
                    acc[c] += f * (x_s * (float)si);
                    if (mu) acc[c] += g * (x_s * x_m);
                }
            }
        }
        #pragma unroll
        for (uint32_t c = 0; c < COLS; ++c) {
            float v = acc[c];
            for (uint32_t s2 = 16; s2 > 0; s2 >>= 1) v += __shfl_down_sync(0xffffffffu, v, s2);
            if (lane == 0) sh[c][warp] = (e == 0xFFFFFFFFu) ? 0.0f : topk_w[(size_t)b * n_active + warp] * v;
        }
    }
    __syncthreads();
    if (threadIdx.x < COLS) {
        const uint32_t o = o0 + threadIdx.x;
        if (o < embd) {
            float v = 0.0f;
            for (uint32_t w = 0; w < n_active; ++w) v += sh[threadIdx.x][w];
            out[(size_t)b * embd + o] = v;
        }
    }
}

// slot 587: the column-tiled down. Same arguments as slot 495 plus `cols`
// (4, 8 or 16); the caller takes it at prefill widths, where the grid stays
// full at COLS times fewer blocks. Numerics are slot 495's exactly.
PD_EXPORT
int pd_kquant_moe_down_cols(const void* down_data, const void* down_scales,
                            const void* idx, const void* topk_w, const void* fq,
                            const void* fs, const void* fsums, void* out,
                            uint32_t ff, uint32_t embd, uint32_t n_active,
                            uint32_t batch, uint32_t cols, uint32_t ddt,
                            void* stream) {
    if (embd == 0 || n_active == 0 || batch == 0) return 0;
    if ((ff & 31u) != 0 || n_active > 16u) return cudaErrorInvalidValue;
    if ((ff & 255u) != 0 && ddt != PD_KQ_IQ4NL_ID) return cudaErrorInvalidValue;
    if (!pd_kq_valid(ddt) && !pd_kq_valid_iq(ddt)) return cudaErrorInvalidValue;
    if (pd_kq_has_mu(ddt) && fsums == nullptr) return cudaErrorInvalidValue;
    cudaStream_t st = (cudaStream_t)stream;
#define PD_KQ_DNC_GO(CV)                                                       \
    do {                                                                       \
        dim3 grid(batch, (embd + (CV) - 1u) / (CV));                           \
        pd_pdl_go(pd_kquant_moe_down_cols_kernel<CV>, grid, 32u * n_active, 0u, \
            st, (const uint8_t*)down_data, (const uint8_t*)down_scales,         \
            (const unsigned int*)idx, (const float*)topk_w, (const int8_t*)fq,  \
            (const float*)fs, (const float*)fsums, (float*)out, ff, embd,       \
            n_active, ddt);                                                     \
    } while (0)
    switch (cols) {
        case 4u: PD_KQ_DNC_GO(4u); break;
        case 8u: PD_KQ_DNC_GO(8u); break;
        case 16u: PD_KQ_DNC_GO(16u); break;
        default: return cudaErrorInvalidValue;
    }
#undef PD_KQ_DNC_GO
    return pd_launch_status();
}

// ---- REGISTER-TILED grouped gate+up (the wave-prefill class, slot 592) ----
// The grouped pair kernel above pays one block per (expert group, OUT ROW), so
// it re-reads the group's activations for every one of the ff output rows: 47
// GB a layer at a 2114-row wave with in_dim 2560 and a 16-row group, which is
// what it was actually waiting on. (The evidence that it is TRAFFIC and not
// unpack: a 32-row group unpacks 237M windows against a 16-row group's 369M
// and is SLOWER - 10.80 vs 8.97 ms a layer - because it moves 60 GB instead
// of 47.)
//
// Here a block owns BM routed rows x BN output columns and stages BOTH sides
// of a BK slice once: the activations land in shared, and the weight windows
// are UNPACKED into shared as int8 with their per-window scales. Every thread
// then owns TN whole columns of one row, so its dots need no cross-thread
// reduction at all - the shuffle tree and the ascending warp fold of the pair
// kernels disappear with it. Activation traffic falls by BN.
//
// NUMERIC CLASS: a thread accumulates its K windows in ascending order, where
// the pair kernels split K across the block and folded the pieces. Same
// per-window math, different association - the prefill-vs-decode split this
// lane already carries for its dense planes, and the wave prefill has never
// been bit-equal to the token walk anyway.
#define PD_KQT_BM 16u
#define PD_KQT_BN 64u
#define PD_KQT_BK 128u
#define PD_KQT_XW (PD_KQT_BK + 16u)   // padded row strides: a 128-byte stride
#define PD_KQT_WW (PD_KQT_BK + 16u)   // puts every lane in the same bank set

template <bool MU>
__global__ void __launch_bounds__(256, 4) pd_kquant_moe_gate_up_tile_kernel(
    const uint8_t* __restrict__ gd, const uint8_t* __restrict__ gsc,
    const uint8_t* __restrict__ ud_, const uint8_t* __restrict__ usc,
    const unsigned int* __restrict__ sorted_row,
    const unsigned int* __restrict__ sorted_slot,
    const unsigned int* __restrict__ block_expert, const int8_t* __restrict__ xq,
    const float* __restrict__ xs, const float* __restrict__ xsums,
    float* __restrict__ out, uint32_t in_dim, uint32_t ff, uint32_t n_active,
    uint32_t gdt, uint32_t udt) {
    PD_PDL_ARM();
    const uint32_t blk = blockIdx.x;
    const uint32_t e = block_expert[blk];
    if (e == PD_MOE_PAD) return;
    const uint32_t o0 = blockIdx.y * PD_KQT_BN;
    const uint32_t tid = threadIdx.x;

    __shared__ int8_t sx[PD_KQT_BM * PD_KQT_XW];
    __shared__ float sxs[PD_KQT_BM][PD_KQT_BK / 32u];
    __shared__ float sxm[MU ? PD_KQT_BM : 1u][MU ? PD_KQT_BK / 16u : 1u];
    __shared__ int8_t swg[PD_KQT_BN * PD_KQT_WW];
    __shared__ int8_t swu[PD_KQT_BN * PD_KQT_WW];
    __shared__ float sfg[PD_KQT_BN][PD_KQT_BK / 16u];
    __shared__ float sfu[PD_KQT_BN][PD_KQT_BK / 16u];
    __shared__ float sgg[MU ? PD_KQT_BN : 1u][MU ? PD_KQT_BK / 16u : 1u];
    __shared__ float sgu[MU ? PD_KQT_BN : 1u][MU ? PD_KQT_BK / 16u : 1u];
    __shared__ unsigned int srow[PD_KQT_BM], sslt[PD_KQT_BM];

    if (tid < PD_KQT_BM) {
        srow[tid] = sorted_row[(size_t)blk * PD_KQT_BM + tid];
        sslt[tid] = sorted_slot[(size_t)blk * PD_KQT_BM + tid];
    }
    __syncthreads();

    // this thread: row `r`, the four columns starting at `c0`
    const uint32_t r = tid & (PD_KQT_BM - 1u);
    const uint32_t c0 = (tid / PD_KQT_BM) * 4u;
    float accg[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float accu[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    const uint32_t gdb = pd_kq_datab(gdt), udb = pd_kq_datab(udt);
    const uint32_t gscb = pd_kq_scb(gdt), uscb = pd_kq_scb(udt);
    const uint32_t grb = pd_kq_row_datab(gdt, in_dim), grs = pd_kq_row_scb(gdt, in_dim);
    const uint32_t urb = pd_kq_row_datab(udt, in_dim), urs = pd_kq_row_scb(udt, in_dim);

    for (uint32_t k0 = 0; k0 < in_dim; k0 += PD_KQT_BK) {
        __syncthreads();
        // ---- stage the activations: BM rows x BK bytes, 8 bytes a thread ----
        {
            const uint32_t rr = tid >> 4, off = (tid & 15u) * 8u;
            if (rr < PD_KQT_BM) {
                const unsigned int b = srow[rr];
                const bool live = b != PD_MOE_PAD;
                const int2 v = live
                    ? *reinterpret_cast<const int2*>(xq + (size_t)b * in_dim + k0 + off)
                    : make_int2(0, 0);
                *reinterpret_cast<int2*>(&sx[rr * PD_KQT_XW + off]) = v;
                if ((tid & 15u) < PD_KQT_BK / 32u) {
                    const uint32_t sbk = tid & 15u;
                    sxs[rr][sbk] = live ? xs[(size_t)b * (in_dim >> 5) + (k0 >> 5) + sbk] : 0.0f;
                }
                if (MU && (tid & 15u) < PD_KQT_BK / 16u) {
                    const uint32_t mb = tid & 15u;
                    sxm[rr][mb] = live ? xsums[(size_t)b * (in_dim >> 4) + (k0 >> 4) + mb] : 0.0f;
                }
            }
        }
        // ---- stage the weights: BN columns x BK, UNPACKED, 2 windows a thread ----
        #pragma unroll
        for (uint32_t t = 0; t < 2u; ++t) {
            const uint32_t idx = tid + t * 256u;              // 512 (column, window) pairs
            const uint32_t cc = idx / (PD_KQT_BK / 16u);
            const uint32_t jj = idx % (PD_KQT_BK / 16u);
            const uint32_t o = o0 + cc;
            const uint32_t base = k0 + jj * 16u;
            const uint32_t sb = base >> 8, w = (base >> 4) & 15u;
            const uint32_t oc = o < ff ? o : ff - 1u;
            int wq[4];
            float f, g;
            const size_t row_e = (size_t)e * ff + oc;
            pd_kq_win_unpack(gdt, gd + row_e * grb + (size_t)sb * gdb,
                             gsc + row_e * grs + (size_t)sb * gscb, w, wq, &f, &g);
            *reinterpret_cast<int4*>(&swg[cc * PD_KQT_WW + jj * 16u]) =
                make_int4(wq[0], wq[1], wq[2], wq[3]);
            sfg[cc][jj] = (o < ff) ? f : 0.0f;
            if (MU) sgg[cc][jj] = (o < ff) ? g : 0.0f;
            pd_kq_win_unpack(udt, ud_ + row_e * urb + (size_t)sb * udb,
                             usc + row_e * urs + (size_t)sb * uscb, w, wq, &f, &g);
            *reinterpret_cast<int4*>(&swu[cc * PD_KQT_WW + jj * 16u]) =
                make_int4(wq[0], wq[1], wq[2], wq[3]);
            sfu[cc][jj] = (o < ff) ? f : 0.0f;
            if (MU) sgu[cc][jj] = (o < ff) ? g : 0.0f;
        }
        __syncthreads();
        // ---- the tile: this row's BK windows against this thread's 4 columns ----
        if (srow[r] != PD_MOE_PAD) {
            #pragma unroll
            for (uint32_t j = 0; j < PD_KQT_BK / 16u; ++j) {
                const int4 xv = *reinterpret_cast<const int4*>(&sx[r * PD_KQT_XW + j * 16u]);
                const float x_s = sxs[r][j >> 1];
                const float x_m = MU ? sxm[r][j] : 0.0f;
                #pragma unroll
                for (uint32_t cc = 0; cc < 4u; ++cc) {
                    const uint32_t c = c0 + cc;
                    const int4 wg = *reinterpret_cast<const int4*>(&swg[c * PD_KQT_WW + j * 16u]);
                    int si = __dp4a(wg.x, xv.x, 0);
                    si = __dp4a(wg.y, xv.y, si);
                    si = __dp4a(wg.z, xv.z, si);
                    si = __dp4a(wg.w, xv.w, si);
                    accg[cc] += sfg[c][j] * (x_s * (float)si);
                    if (MU) accg[cc] += sgg[c][j] * (x_s * x_m);
                    const int4 wu = *reinterpret_cast<const int4*>(&swu[c * PD_KQT_WW + j * 16u]);
                    si = __dp4a(wu.x, xv.x, 0);
                    si = __dp4a(wu.y, xv.y, si);
                    si = __dp4a(wu.z, xv.z, si);
                    si = __dp4a(wu.w, xv.w, si);
                    accu[cc] += sfu[c][j] * (x_s * (float)si);
                    if (MU) accu[cc] += sgu[c][j] * (x_s * x_m);
                }
            }
        }
    }

    const unsigned int b = srow[r];
    if (b == PD_MOE_PAD) return;
    #pragma unroll
    for (uint32_t cc = 0; cc < 4u; ++cc) {
        const uint32_t o = o0 + c0 + cc;
        if (o >= ff) continue;
        const float g = accg[cc], u = accu[cc];
        out[((size_t)b * n_active + sslt[r]) * ff + o] = (g / (1.0f + __expf(-g))) * u;
    }
}

// The register-tiled DOWN twin (slot 593). Same tile as the gate+up pair
// above - BM routed rows x BN output columns, both operands staged per BK
// slice, a thread owning whole dots - but its activations are the SORTED
// pairs' swiglu rows (`fq`, pair-major) and its output is the per-(pair,
// column) partial that `pd_moe_part_fold_at` folds in slot order. The
// grouped down it replaces still read the staged rows once per column and
// reduced across a warp; here neither happens.
template <bool MU>
__global__ void __launch_bounds__(256, 4) pd_kquant_moe_down_tile_kernel(
    const uint8_t* __restrict__ dd, const uint8_t* __restrict__ dsc,
    const unsigned int* __restrict__ sorted_row,
    const unsigned int* __restrict__ sorted_slot,
    const unsigned int* __restrict__ block_expert, const float* __restrict__ topk_w,
    const int8_t* __restrict__ fq, const float* __restrict__ fs,
    const float* __restrict__ fsums, float* __restrict__ part, uint32_t ff,
    uint32_t embd, uint32_t o0, uint32_t ocols, uint32_t n_active, uint32_t ddt) {
    PD_PDL_ARM();
    const uint32_t blk = blockIdx.x;
    const uint32_t e = block_expert[blk];
    if (e == PD_MOE_PAD) return;
    const uint32_t cbase = blockIdx.y * PD_KQT_BN;
    const uint32_t tid = threadIdx.x;

    __shared__ int8_t sx[PD_KQT_BM * PD_KQT_XW];
    __shared__ float sxs[PD_KQT_BM][PD_KQT_BK / 32u];
    __shared__ float sxm[MU ? PD_KQT_BM : 1u][MU ? PD_KQT_BK / 16u : 1u];
    __shared__ int8_t swd[PD_KQT_BN * PD_KQT_WW];
    __shared__ float sfd[PD_KQT_BN][PD_KQT_BK / 16u];
    __shared__ float sgd[MU ? PD_KQT_BN : 1u][MU ? PD_KQT_BK / 16u : 1u];
    __shared__ unsigned int srow[PD_KQT_BM], sslt[PD_KQT_BM];

    if (tid < PD_KQT_BM) {
        srow[tid] = sorted_row[(size_t)blk * PD_KQT_BM + tid];
        sslt[tid] = sorted_slot[(size_t)blk * PD_KQT_BM + tid];
    }
    __syncthreads();

    const uint32_t r = tid & (PD_KQT_BM - 1u);
    const uint32_t c0 = (tid / PD_KQT_BM) * 4u;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const uint32_t ddb = pd_kq_datab(ddt), dscb = pd_kq_scb(ddt);
    const uint32_t drb = pd_kq_row_datab(ddt, ff), drs = pd_kq_row_scb(ddt, ff);

    for (uint32_t k0 = 0; k0 < ff; k0 += PD_KQT_BK) {
        __syncthreads();
        {   // the group's swiglu rows: BM x BK, 8 bytes a thread
            const uint32_t rr = tid >> 4, off = (tid & 15u) * 8u;
            if (rr < PD_KQT_BM) {
                const unsigned int b = srow[rr];
                const bool live = b != PD_MOE_PAD;
                const size_t pair = live ? (size_t)b * n_active + sslt[rr] : 0u;
                const int2 v = live
                    ? *reinterpret_cast<const int2*>(fq + pair * ff + k0 + off)
                    : make_int2(0, 0);
                *reinterpret_cast<int2*>(&sx[rr * PD_KQT_XW + off]) = v;
                if ((tid & 15u) < PD_KQT_BK / 32u) {
                    const uint32_t sbk = tid & 15u;
                    sxs[rr][sbk] = live ? fs[pair * (ff >> 5) + (k0 >> 5) + sbk] : 0.0f;
                }
                if (MU && (tid & 15u) < PD_KQT_BK / 16u) {
                    const uint32_t mb = tid & 15u;
                    sxm[rr][mb] = live ? fsums[pair * (ff >> 4) + (k0 >> 4) + mb] : 0.0f;
                }
            }
        }
        #pragma unroll
        for (uint32_t t = 0; t < 2u; ++t) {
            const uint32_t idx = tid + t * 256u;
            const uint32_t cc = idx / (PD_KQT_BK / 16u);
            const uint32_t jj = idx % (PD_KQT_BK / 16u);
            const uint32_t c = cbase + cc;
            const uint32_t o = o0 + c;
            const bool live = c < ocols && o < embd;
            const uint32_t oc = live ? o : embd - 1u;
            const uint32_t base = k0 + jj * 16u;
            const uint32_t sb = base >> 8, w = (base >> 4) & 15u;
            int wq[4];
            float f, g;
            const size_t row_e = (size_t)e * embd + oc;
            pd_kq_win_unpack(ddt, dd + row_e * drb + (size_t)sb * ddb,
                             dsc + row_e * drs + (size_t)sb * dscb, w, wq, &f, &g);
            *reinterpret_cast<int4*>(&swd[cc * PD_KQT_WW + jj * 16u]) =
                make_int4(wq[0], wq[1], wq[2], wq[3]);
            sfd[cc][jj] = live ? f : 0.0f;
            if (MU) sgd[cc][jj] = live ? g : 0.0f;
        }
        __syncthreads();
        if (srow[r] != PD_MOE_PAD) {
            #pragma unroll
            for (uint32_t j = 0; j < PD_KQT_BK / 16u; ++j) {
                const int4 xv = *reinterpret_cast<const int4*>(&sx[r * PD_KQT_XW + j * 16u]);
                const float x_s = sxs[r][j >> 1];
                const float x_m = MU ? sxm[r][j] : 0.0f;
                #pragma unroll
                for (uint32_t cc = 0; cc < 4u; ++cc) {
                    const uint32_t c = c0 + cc;
                    const int4 wv = *reinterpret_cast<const int4*>(&swd[c * PD_KQT_WW + j * 16u]);
                    int si = __dp4a(wv.x, xv.x, 0);
                    si = __dp4a(wv.y, xv.y, si);
                    si = __dp4a(wv.z, xv.z, si);
                    si = __dp4a(wv.w, xv.w, si);
                    acc[cc] += sfd[c][j] * (x_s * (float)si);
                    if (MU) acc[cc] += sgd[c][j] * (x_s * x_m);
                }
            }
        }
    }

    const unsigned int b = srow[r];
    if (b == PD_MOE_PAD) return;
    const size_t pair = (size_t)b * n_active + sslt[r];
    #pragma unroll
    for (uint32_t cc = 0; cc < 4u; ++cc) {
        const uint32_t c = cbase + c0 + cc;
        if (c >= ocols || o0 + c >= embd) continue;
        part[pair * ocols + c] = topk_w[pair] * acc[cc];
    }
}

// slot 593: the register-tiled down over one column chunk. `ff` must be a
// multiple of the tile's BK (128) and the CSR must be a moe_align_bm(16).
PD_EXPORT
int pd_kquant_moe_down_tile(const void* down_data, const void* down_scales,
                            const void* sorted_row, const void* sorted_slot,
                            const void* block_expert, const void* topk_w,
                            const void* fq, const void* fs, const void* fsums,
                            void* part, uint32_t ff, uint32_t embd, uint32_t o0,
                            uint32_t ocols, uint32_t n_active, uint32_t max_blocks,
                            uint32_t ddt, void* stream) {
    if (embd == 0 || n_active == 0 || max_blocks == 0 || ocols == 0) return 0;
    if ((ff % PD_KQT_BK) != 0 || n_active > 16u) return cudaErrorInvalidValue;
    if (!pd_kq_valid(ddt) && !pd_kq_valid_iq(ddt)) return cudaErrorInvalidValue;
    const bool mu = pd_kq_has_mu(ddt);
    if (mu && fsums == nullptr) return cudaErrorInvalidValue;
    if (sorted_row == nullptr || sorted_slot == nullptr || block_expert == nullptr)
        return cudaErrorInvalidValue;
    dim3 grid(max_blocks, (ocols + PD_KQT_BN - 1u) / PD_KQT_BN);
    cudaStream_t st = (cudaStream_t)stream;
    if (mu) {
        pd_pdl_go(pd_kquant_moe_down_tile_kernel<true>, grid, 256u, 0u, st,
            (const uint8_t*)down_data, (const uint8_t*)down_scales,
            (const unsigned int*)sorted_row, (const unsigned int*)sorted_slot,
            (const unsigned int*)block_expert, (const float*)topk_w,
            (const int8_t*)fq, (const float*)fs, (const float*)fsums, (float*)part,
            ff, embd, o0, ocols, n_active, ddt);
    } else {
        pd_pdl_go(pd_kquant_moe_down_tile_kernel<false>, grid, 256u, 0u, st,
            (const uint8_t*)down_data, (const uint8_t*)down_scales,
            (const unsigned int*)sorted_row, (const unsigned int*)sorted_slot,
            (const unsigned int*)block_expert, (const float*)topk_w,
            (const int8_t*)fq, (const float*)fs, (const float*)fsums, (float*)part,
            ff, embd, o0, ocols, n_active, ddt);
    }
    return pd_launch_status();
}

// slot 592: the register-tiled pair. Takes a `pd_moe_align_bm(bm = 16)` CSR
// (BM is the tile, so the group size is the kernel's, not the caller's) and
// writes the token-batched kernel's PAIR-major output, so the quantize and
// the down half after it are unchanged.
PD_EXPORT
int pd_kquant_moe_gate_up_tile(const void* gate_data, const void* gate_scales,
                               const void* up_data, const void* up_scales,
                               const void* sorted_row, const void* sorted_slot,
                               const void* block_expert, const void* xq,
                               const void* xs, const void* xsums, void* out,
                               uint32_t in_dim, uint32_t ff, uint32_t n_active,
                               uint32_t max_blocks, uint32_t gdt, uint32_t udt,
                               void* stream) {
    if (ff == 0 || n_active == 0 || max_blocks == 0) return 0;
    if ((in_dim % PD_KQT_BK) != 0) return cudaErrorInvalidValue;
    if (!(pd_kq_valid(gdt) || pd_kq_valid_iq(gdt)) || !(pd_kq_valid(udt) || pd_kq_valid_iq(udt)))
        return cudaErrorInvalidValue;
    const bool mu = pd_kq_has_mu(gdt) || pd_kq_has_mu(udt);
    if (mu && xsums == nullptr) return cudaErrorInvalidValue;
    if (sorted_row == nullptr || sorted_slot == nullptr || block_expert == nullptr)
        return cudaErrorInvalidValue;
    dim3 grid(max_blocks, (ff + PD_KQT_BN - 1u) / PD_KQT_BN);
    cudaStream_t st = (cudaStream_t)stream;
    if (mu) {
        pd_pdl_go(pd_kquant_moe_gate_up_tile_kernel<true>, grid, 256u, 0u, st,
            (const uint8_t*)gate_data, (const uint8_t*)gate_scales,
            (const uint8_t*)up_data, (const uint8_t*)up_scales,
            (const unsigned int*)sorted_row, (const unsigned int*)sorted_slot,
            (const unsigned int*)block_expert, (const int8_t*)xq, (const float*)xs,
            (const float*)xsums, (float*)out, in_dim, ff, n_active, gdt, udt);
    } else {
        pd_pdl_go(pd_kquant_moe_gate_up_tile_kernel<false>, grid, 256u, 0u, st,
            (const uint8_t*)gate_data, (const uint8_t*)gate_scales,
            (const uint8_t*)up_data, (const uint8_t*)up_scales,
            (const unsigned int*)sorted_row, (const unsigned int*)sorted_slot,
            (const unsigned int*)block_expert, (const int8_t*)xq, (const float*)xs,
            (const float*)xsums, (float*)out, in_dim, ff, n_active, gdt, udt);
    }
    return pd_launch_status();
}

// ---- expert-GROUPED down + slot fold (the prefill class, slots 589/590) ----
// The column-tiled kernel above still unpacks every routed pair's expert row
// per output column: a 2114-row wave walk is 2114 x 2560 x 400 window unpacks
// a layer, and down owned 42% of it. Here a block owns (expert group, column
// tile) off the same moe_align CSR the grouped gate_up reads: the group's
// activation rows are staged ONCE into shared, and each column's weight row is
// unpacked ONCE and walked against all of them. Unpack falls by the rows per
// group, the fq re-reads by the columns per block.
//
// A grouped block holds pairs of DIFFERENT tokens, so it cannot fold the slots
// itself - it writes one partial per (pair, column) and `pd_moe_part_fold_at`
// sums them in ascending slot order, which is the fold order the ungrouped
// kernel used inside its block. Per (pair, column) the dot keeps the plain
// kernel's lane->window mapping (32 lanes striding 512) and its 32-lane tree,
// so every number here is bit-identical to slot 495's.
//
// `o0`/`ocols` are a COLUMN CHUNK: the partials plane is [pairs, ocols], so
// the caller walks the output in chunks instead of sizing a [pairs, embd]
// plane (419 MB at a 4096-row wave; 42 MB at ocols = 256).
template <uint32_t T, uint32_t CPW>
__global__ void __launch_bounds__(256, 4) pd_kquant_moe_down_grp_kernel(
    const uint8_t* __restrict__ dd, const uint8_t* __restrict__ dsc,
    const unsigned int* __restrict__ sorted_row,
    const unsigned int* __restrict__ sorted_slot,
    const unsigned int* __restrict__ block_expert, const float* __restrict__ topk_w,
    const int8_t* __restrict__ fq, const float* __restrict__ fs,
    const float* __restrict__ fsums, float* __restrict__ part, uint32_t ff,
    uint32_t embd, uint32_t o0, uint32_t ocols, uint32_t n_active, uint32_t ddt) {
    PD_PDL_ARM();
    const uint32_t blk = blockIdx.x;
    const uint32_t e = block_expert[blk];
    if (e == PD_MOE_PAD) return;
    const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
    const uint32_t nsb = ff >> 5;
    // dynamic shared: the group's fq rows, their per-32 scales, the row list
    extern __shared__ char pd_kdg_sh[];
    int8_t* sx = reinterpret_cast<int8_t*>(pd_kdg_sh);            // [T][ff]
    float* sxs = reinterpret_cast<float*>(sx + (size_t)T * ff);   // [T][ff/32]
    unsigned int* srow = reinterpret_cast<unsigned int*>(sxs + (size_t)T * nsb);
    unsigned int* sslt = srow + T;
    if (tid < T) {
        srow[tid] = sorted_row[(size_t)blk * T + tid];
        sslt[tid] = sorted_slot[(size_t)blk * T + tid];
    }
    __syncthreads();
    for (uint32_t i = 0; i < T; ++i) {
        const unsigned int b = srow[i];
        if (b == PD_MOE_PAD) continue;
        const size_t pair = (size_t)b * n_active + sslt[i];
        for (uint32_t j = tid * 4u; j < ff; j += blockDim.x * 4u)
            *reinterpret_cast<int*>(&sx[i * ff + j]) =
                *reinterpret_cast<const int*>(fq + pair * ff + j);
        for (uint32_t j = tid; j < nsb; j += blockDim.x)
            sxs[i * nsb + j] = fs[pair * nsb + j];
    }
    __syncthreads();

    const uint32_t ddb = pd_kq_datab(ddt), dscb = pd_kq_scb(ddt);
    const bool mu = pd_kq_has_mu(ddt);
    // One column at a time per warp, and the row loop is NOT unrolled: with
    // the group's rows unrolled (or held as a register tile of columns) the
    // kernel compiled to 128 registers, which is 2 blocks an SM and 33% of
    // warps - and ncu then put it at 39% of L1 throughput with DRAM at 3.5%,
    // i.e. waiting on its own operand traffic with nothing to hide it. The
    // 4-column register tile was measured too: 22.6 ms a layer against this
    // shape's 13.4.
    for (uint32_t cw = 0; cw < CPW; ++cw) {
        const uint32_t c = blockIdx.y * (8u * CPW) + cw * 8u + warp;
        if (c >= ocols) break;
        const uint32_t o = o0 + c;
        if (o >= embd) break;
        const uint8_t* row = dd + ((size_t)e * embd + o) * pd_kq_row_datab(ddt, ff);
        const uint8_t* rrec = dsc + ((size_t)e * embd + o) * pd_kq_row_scb(ddt, ff);
        float acc[T];
        #pragma unroll
        for (uint32_t i = 0; i < T; ++i) acc[i] = 0.0f;
        // the plain kernel's walk: lane owns window `lane`, stride 32 windows
        for (uint32_t base = lane * 16u; base < ff; base += 32u * 16u) {
            const uint32_t sb = base >> 8, w = (base >> 4) & 15u;
            int wq[4];
            float f, g;
            pd_kq_win_unpack(ddt, row + (size_t)sb * ddb, rrec + (size_t)sb * dscb, w, wq, &f, &g);
            for (uint32_t i = 0; i < T; ++i) {
                if (srow[i] == PD_MOE_PAD) continue;
                const int4 xv = *reinterpret_cast<const int4*>(&sx[i * ff + base]);
                int si = __dp4a(wq[0], xv.x, 0);
                si = __dp4a(wq[1], xv.y, si);
                si = __dp4a(wq[2], xv.z, si);
                si = __dp4a(wq[3], xv.w, si);
                const float x_s = sxs[i * nsb + (base >> 5)];
                acc[i] += f * (x_s * (float)si);
                if (mu) {
                    const size_t pair = (size_t)srow[i] * n_active + sslt[i];
                    acc[i] += g * (x_s * fsums[pair * (ff >> 4) + (base >> 4)]);
                }
            }
        }
        for (uint32_t i = 0; i < T; ++i) {
            float v = acc[i];
            for (uint32_t s2 = 16; s2 > 0; s2 >>= 1) v += __shfl_down_sync(0xffffffffu, v, s2);
            if (lane == 0 && srow[i] != PD_MOE_PAD) {
                const size_t pair = (size_t)srow[i] * n_active + sslt[i];
                part[pair * ocols + c] = topk_w[pair] * v;
            }
        }
    }
}

// Fold a column chunk's per-(token, slot) partials into `out` in ASCENDING
// slot order - the same order the ungrouped down kernel summed inside its
// block. One writer per (token, column).
__global__ void pd_moe_part_fold_at_kernel(const float* __restrict__ part,
                                           float* __restrict__ out, uint32_t embd,
                                           uint32_t o0, uint32_t ocols,
                                           uint32_t n_active) {
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ocols) return;
    const uint32_t b = blockIdx.y;
    float v = 0.0f;
    for (uint32_t s = 0; s < n_active; ++s)
        v += part[((size_t)b * n_active + s) * ocols + c];
    out[(size_t)b * embd + o0 + c] = v;
}

// slot 589: grouped down over one column chunk. `group` is 8 or 16 (the CSR's
// bm); `ocols` is the chunk width, and `part` must hold rows*n_active*ocols
// floats. PAD lanes write nothing - the fold reads only real pairs because
// every (token, slot) of the walk is present exactly once in the CSR.
PD_EXPORT
int pd_kquant_moe_down_grp(const void* down_data, const void* down_scales,
                           const void* sorted_row, const void* sorted_slot,
                           const void* block_expert, const void* topk_w,
                           const void* fq, const void* fs, const void* fsums,
                           void* part, uint32_t ff, uint32_t embd, uint32_t o0,
                           uint32_t ocols, uint32_t n_active, uint32_t max_blocks,
                           uint32_t group, uint32_t ddt, void* stream) {
    if (embd == 0 || n_active == 0 || max_blocks == 0 || ocols == 0) return 0;
    if ((ff & 31u) != 0 || n_active > 16u) return cudaErrorInvalidValue;
    if ((ff & 255u) != 0 && ddt != PD_KQ_IQ4NL_ID) return cudaErrorInvalidValue;
    if (!pd_kq_valid(ddt) && !pd_kq_valid_iq(ddt)) return cudaErrorInvalidValue;
    if (pd_kq_has_mu(ddt) && fsums == nullptr) return cudaErrorInvalidValue;
    if (sorted_row == nullptr || sorted_slot == nullptr || block_expert == nullptr)
        return cudaErrorInvalidValue;
    constexpr uint32_t CPW = 2u;   // 8 warps x 2 columns = 16 a block
    cudaStream_t st = (cudaStream_t)stream;
#define PD_KQ_DGRP_GO(TV)                                                      \
    do {                                                                       \
        const uint32_t smem = (uint32_t)((size_t)(TV) * ff                      \
            + (size_t)(TV) * (ff >> 5) * sizeof(float) + 2u * (TV) * sizeof(unsigned int)); \
        if (smem > 48u * 1024u) return cudaErrorInvalidValue;                   \
        dim3 grid(max_blocks, (ocols + 8u * CPW - 1u) / (8u * CPW));            \
        pd_pdl_go(pd_kquant_moe_down_grp_kernel<TV, CPW>, grid, 256u, smem, st,  \
            (const uint8_t*)down_data, (const uint8_t*)down_scales,             \
            (const unsigned int*)sorted_row, (const unsigned int*)sorted_slot,  \
            (const unsigned int*)block_expert, (const float*)topk_w,            \
            (const int8_t*)fq, (const float*)fs, (const float*)fsums,           \
            (float*)part, ff, embd, o0, ocols, n_active, ddt);                  \
    } while (0)
    switch (group) {
        case 8u: PD_KQ_DGRP_GO(8u); break;
        case 16u: PD_KQ_DGRP_GO(16u); break;
        case 32u: PD_KQ_DGRP_GO(32u); break;
        default: return cudaErrorInvalidValue;
    }
#undef PD_KQ_DGRP_GO
    return pd_launch_status();
}

// slot 590: the fold that turns slot 589's chunk of partials into `out`.
PD_EXPORT
int pd_moe_part_fold_at(const void* part, void* out, uint32_t embd, uint32_t o0,
                        uint32_t ocols, uint32_t n_active, uint32_t rows,
                        void* stream) {
    if (rows == 0 || ocols == 0) return 0;
    dim3 grid((ocols + 255u) / 256u, rows);
    pd_moe_part_fold_at_kernel<<<grid, 256u, 0, (cudaStream_t)stream>>>(
        (const float*)part, (float*)out, embd, o0, ocols, n_active);
    return pd_launch_status();
}

// ---- token-major down for the expert-major prefill (slot 584) --------------
// The (column, token) geometry above is wrong for a wave: with top-10 over
// seven waves nearly every token holds an in-wave pair in every wave, so a
// per-(column, token) block ran once per wave with nine of its ten warps
// skipping - 18 ms a launch at 435 tokens, 57% of the prefill (nsys).
// Here a block owns ONE token from the wave's token list and walks that
// token's in-wave pairs in order (1-3 of them): the pair's fq row + scales
// are staged to smem once, every thread computes a strided set of output
// columns over the whole 640-deep row, and the topk-weighted sum lands in
// registers. One writer per (token, column) per wave, pair order fixed by
// the routing, waves in sequence: deterministic, and a wave costs its own
// pairs. A thread's column dot is a serial window walk (the plain kernel's
// 32-lane tree is a different f32 order; the wave-order reassociation is
// already the class this path carries).
template <uint32_t NTH>
__global__ void __launch_bounds__(NTH) pd_kquant_moe_down_rows_kernel(
    const uint8_t* __restrict__ dd, const uint8_t* __restrict__ dsc,
    const unsigned int* __restrict__ idx, const float* __restrict__ topk_w,
    const int8_t* __restrict__ fq, const float* __restrict__ fs,
    const float* __restrict__ fsums, float* __restrict__ out, uint32_t ff,
    uint32_t embd, uint32_t n_active, uint32_t ddt,
    const unsigned int* __restrict__ rows_list, const unsigned int* __restrict__ n_rows) {
    PD_PDL_ARM();
    extern __shared__ __align__(16) unsigned char pd_dnr_sh[];
    int8_t* sx = (int8_t*)pd_dnr_sh;                         // ff int8
    float* sxs = (float*)(pd_dnr_sh + ff);                   // ff/32 scales
    float* sxm = sxs + (ff >> 5);                            // ff/16 sums
    const uint32_t ddb = pd_kq_datab(ddt);
    const uint32_t dscb = pd_kq_scb(ddt);
    const bool mu = pd_kq_has_mu(ddt);
    const uint32_t rowb = pd_kq_row_datab(ddt, ff), rows_b = pd_kq_row_scb(ddt, ff);
    const uint32_t nwin = ff >> 4;
    // columns per thread: embd / NTH rounded up, compile-time bound 16
    constexpr uint32_t MAXC = 16u;
    const uint32_t i_hi = *n_rows;
    for (uint32_t i = blockIdx.x; i < i_hi; i += gridDim.x) {
        const uint32_t b = rows_list[i];
        float acc[MAXC];
        #pragma unroll
        for (uint32_t c = 0; c < MAXC; ++c) acc[c] = 0.0f;
        for (uint32_t j = 0; j < n_active; ++j) {
            const size_t srow = (size_t)b * n_active + j;
            const uint32_t e = idx[srow];
            if (e == 0xFFFFFFFFu) continue;           // block-uniform
            // stage this pair's quantized activation row
            __syncthreads();                          // previous pair's readers done
            for (uint32_t t = threadIdx.x; t < (ff >> 4); t += NTH)
                ((int4*)sx)[t] = ((const int4*)(fq + srow * ff))[t];
            for (uint32_t t = threadIdx.x; t < (ff >> 5); t += NTH) sxs[t] = fs[srow * (ff >> 5) + t];
            if (mu)
                for (uint32_t t = threadIdx.x; t < (ff >> 4); t += NTH) sxm[t] = fsums[srow * (ff >> 4) + t];
            __syncthreads();
            const float wj = topk_w[srow];
            const uint8_t* erow = dd + (size_t)e * embd * rowb;
            const uint8_t* erec = dsc + (size_t)e * embd * rows_b;
            #pragma unroll
            for (uint32_t c = 0; c < MAXC; ++c) {
                const uint32_t o = threadIdx.x + c * NTH;
                if (o >= embd) break;
                const uint8_t* row = erow + (size_t)o * rowb;
                const uint8_t* rrec = erec + (size_t)o * rows_b;
                float a = 0.0f;
                for (uint32_t w16 = 0; w16 < nwin; ++w16) {
                    const uint32_t base = w16 << 4;
                    const uint32_t s = base >> 8, w = (base >> 4) & 15u;
                    const int4 xv = *reinterpret_cast<const int4*>(sx + base);
                    int wq[4];
                    float f, g;
                    pd_kq_win_unpack(ddt, row + (size_t)s * ddb, rrec + (size_t)s * dscb, w, wq, &f, &g);
                    int si = __dp4a(wq[0], xv.x, 0);
                    si = __dp4a(wq[1], xv.y, si);
                    si = __dp4a(wq[2], xv.z, si);
                    si = __dp4a(wq[3], xv.w, si);
                    const float x_s = sxs[base >> 5];
                    a += f * (x_s * (float)si);
                    if (mu) a += g * (x_s * sxm[base >> 4]);
                }
                acc[c] += wj * a;
            }
        }
        #pragma unroll
        for (uint32_t c = 0; c < MAXC; ++c) {
            const uint32_t o = threadIdx.x + c * NTH;
            if (o < embd) out[(size_t)b * embd + o] += acc[c];
        }
    }
}

// slot 584: the token-major form - accumulates into `out` over the tokens
// in `rows`/`n_rows` (the caller zeroes `out` before the first wave).
#define PD_KQ_MOE_LIST_R 512u
PD_EXPORT
int pd_kquant_moe_down_list(const void* down_data, const void* down_scales,
                            const void* idx, const void* topk_w, const void* fq,
                            const void* fs, const void* fsums, void* out,
                            uint32_t ff, uint32_t embd, uint32_t n_active,
                            uint32_t batch, uint32_t ddt, const void* rows,
                            const void* n_rows, void* stream) {
    if (embd == 0 || n_active == 0 || batch == 0) return 0;
    if ((ff & 31u) != 0 || n_active > 16u) return cudaErrorInvalidValue;
    if ((ff & 255u) != 0 && ddt != PD_KQ_IQ4NL_ID) return cudaErrorInvalidValue;
    if (!pd_kq_valid(ddt) && !pd_kq_valid_iq(ddt)) return cudaErrorInvalidValue;
    if ((pd_kq_has_mu(ddt)) && fsums == nullptr) return cudaErrorInvalidValue;
    if (rows == nullptr || n_rows == nullptr) return cudaErrorInvalidValue;
    if (embd > 16u * 256u) return cudaErrorInvalidValue;   // MAXC columns per thread
    const uint32_t smem = ff + (ff >> 5) * 4u + (ff >> 4) * 4u;
    dim3 grid(PD_KQ_MOE_LIST_R);
    pd_pdl_go(pd_kquant_moe_down_rows_kernel<256u>, grid, 256u, smem, (cudaStream_t)stream,
        (const uint8_t*)down_data, (const uint8_t*)down_scales,
        (const unsigned int*)idx, (const float*)topk_w, (const int8_t*)fq,
        (const float*)fs, (const float*)fsums, (float*)out, ff, embd, n_active,
        ddt, (const unsigned int*)rows, (const unsigned int*)n_rows);
    return pd_launch_status();
}

// ---- sorted k-quant MoE mma (the prefill/serving class) ---------------------
// The ks v2 machinery (quant/kquant_w4a8.cuh's pd_kquant_mma_ks_kernel: ST-deep
// cp.async ring holding RAW compressed strips + 24 B scale records, nibbles
// unpacked inline at fragment-load time) applied to the sorted moe_align
// layout of the Q8 mma pair (moe/q8.cuh): one CTA per (32-token sorted block,
// 64-row output strip), block -> expert via block_expert, activation columns
// gathered through sorted_row (PAD -> cp.async zero-fill, contributions
// vanish through zero scales exactly like the dense kernel's dead rows).
//
// GU=true runs the ring twice (gate then up - the Q8 pair re-stages
// activations per mat too), then a SwiGLU + per-32 in-register quantize
// epilogue writes fq/fs SORTED-CONTIGUOUS for the down half. The ks fragment
// map scatters a token's 32-row output block across warps, so the quantize
// amax bounces the fused f32s through the (dead by then) ring smem first.
// GU=false is the down half over K = ff: activations are the sorted fq rows
// (direct index, no gather), epilogue scatters topk_w-weighted partials to
// (token, slot) rows for pd_moe_slot_combine - one writer per element,
// deterministic (the Q8 down_mma discipline).
//
// Single dtype for the gate/up pair (the mat loop shares one template DT) -
// UD 35B-A3B: gate.ty == up.ty on all 40 layers; the engine falls back to
// the token-batched pair on a file that ever mixes the pair.
//
// Numeric class: identical expressions in identical K-fold order as the
// dense ks v2 (super-ascending, kk 0..7) - exact int8 dots, f32 scale
// application, deterministic for a fixed sorted layout.
template <uint32_t DT, bool GU>
__global__ void __launch_bounds__(256) pd_kq_moe_mma_kernel(
        const uint8_t* __restrict__ wd0, const uint8_t* __restrict__ ws0,
        const uint8_t* __restrict__ wd1, const uint8_t* __restrict__ ws1,
        const unsigned int* __restrict__ sorted_row,
        const unsigned int* __restrict__ sorted_slot,
        const unsigned int* __restrict__ block_expert,
        const float* __restrict__ topk_w, const int8_t* __restrict__ xq,
        const float* __restrict__ xs, const float* __restrict__ xsums,
        int8_t* __restrict__ fq, float* __restrict__ fs,
        float* __restrict__ part, uint32_t in_dim, uint32_t out_dim,
        uint32_t n_active) {
#if PD_MMA_OK
    constexpr bool MU = (DT == PD_KQ_Q4K || DT == PD_KQ_Q5K || DT == PD_KQ_Q40);
    constexpr bool K16 = (DT == PD_KQ_Q6K);
    constexpr uint32_t BN = 32u, ST = 2u;
    constexpr uint32_t CPW = BN / 2u;    // 8 warps = 4 row x 2 col
    constexpr uint32_t NSUB = CPW / 8u;
    constexpr uint32_t DATAB = DT == PD_KQ_Q6K ? PD_KQ6_DATA
                             : DT == PD_KQ_Q5K ? PD_KQ5_DATA : PD_KQ4_DATA;
    constexpr uint32_t WSTR = DATAB + 16u;

    // ring planes - the dense ks v2 layout verbatim (same size helper)
    constexpr uint32_t W_PL = 64u * WSTR, R_PL = 64u * PD_KQ_SCB;
    constexpr uint32_t B_PL = BN * (PD_KM_BSTR * 4u);
    constexpr uint32_t XS_PL = BN * 48u, SU_PL = BN * 80u;
    constexpr uint32_t OFF_R = ST * W_PL, OFF_B = OFF_R + ST * R_PL;
    constexpr uint32_t OFF_XS = OFF_B + ST * B_PL;
    constexpr uint32_t OFF_SU = OFF_XS + ST * XS_PL;
    static_assert(OFF_SU + (MU ? ST * SU_PL : 0u) == pd_km_smem_bytes(DT, BN, ST),
                  "smem layout matches the launcher's size");
    // the GU epilogue bounce (BN cols x 65-f32 stride) reuses the ring space
    static_assert(!GU || BN * 65u * 4u <= pd_km_smem_bytes(DT, BN, ST),
                  "fused bounce fits the ring");
    extern __shared__ __align__(16) unsigned char pd_kqm_sh[];
    auto rw = [&](uint32_t buf) { return pd_kqm_sh + buf * W_PL; };
    auto rrec = [&](uint32_t buf) { return pd_kqm_sh + OFF_R + buf * R_PL; };
    auto rb = [&](uint32_t buf) {
        return (const int*)(pd_kqm_sh + OFF_B + buf * B_PL);
    };
    auto rxs = [&](uint32_t buf) {
        return (const float*)(pd_kqm_sh + OFF_XS + buf * XS_PL);
    };
    auto rsu = [&](uint32_t buf) {
        return (const float*)(pd_kqm_sh + OFF_SU + buf * SU_PL);
    };

    const uint32_t blk = blockIdx.x;  // token block (fast axis: L2 strip reuse)
    const uint32_t e = block_expert[blk];
    if (e == PD_MOE_PAD) return;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5u;
    const uint32_t g = lane >> 2u, t = lane & 3u;
    const uint32_t wr = (warp & 3u) * 16u;
    const uint32_t wc = (warp >> 2u) * CPW;
    const uint32_t row_base = blockIdx.y * 64u;
    const uint32_t n_super = in_dim >> 8u;
    const uint32_t nb32 = in_dim >> 5u, nb16 = in_dim >> 4u;
    const size_t wrow0 = (size_t)e * out_dim + row_base;

    // token map: GU gathers activations through it (PAD -> zero-fill); down
    // reads sorted-contiguous fq rows and uses it only in the scatter.
    __shared__ unsigned int tok[BN];
    __shared__ unsigned int slt[GU ? 1u : BN];
    for (uint32_t i = tid; i < BN; i += 256u) {
        tok[i] = sorted_row[(size_t)blk * BN + i];
        if (!GU) slt[i] = sorted_slot[(size_t)blk * BN + i];
    }
    __syncthreads();

    float acc_g[NSUB][4] = {}, acc_u[NSUB][4] = {};
    #pragma unroll
    for (uint32_t mat = 0; mat < (GU ? 2u : 1u); ++mat) {
        const uint8_t* wd = mat ? wd1 : wd0;
        const uint8_t* ws = mat ? ws1 : ws0;
        float acc[NSUB][4] = {};

        // stage super kt's planes into ring buffer `buf` - all cp.async,
        // commit at the call site (the dense ks ring discipline)
        auto stage = [&](uint32_t kt, uint32_t buf) {
            constexpr uint32_t WI4 = DATAB / 16u;
            for (uint32_t i = tid; i < 64u * WI4; i += 256u) {
                const uint32_t row = i / WI4, c = i % WI4;
                const bool ok = (row_base + row) < out_dim;
                pd_mma_cpa16p(rw(buf) + row * WSTR + c * 16u,
                              wd + ((wrow0 + row) * n_super + kt) * DATAB + c * 16u,
                              ok);
            }
            for (uint32_t i = tid; i < 64u * 3u; i += 256u) {  // recs: 3 x 8 B
                const uint32_t row = i / 3u, c = i % 3u;
                const bool ok = (row_base + row) < out_dim;
                pd_kq_cpa8p(rrec(buf) + row * PD_KQ_SCB + c * 8u,
                            ws + ((wrow0 + row) * n_super + kt) * PD_KQ_SCB + c * 8u,
                            ok);
            }
            const uint32_t k0 = kt * 256u;
            for (uint32_t i = tid; i < BN * 16u; i += 256u) {
                const uint32_t col = i >> 4u, k16i = i & 15u;
                // GU: gather by token, clamp PAD to row 0 (address formed but
                // never read at src-size 0). Down: sorted fq rows, always live
                // (PAD rows hold the zeros the GU epilogue wrote).
                const bool ok = !GU || tok[col] != PD_MOE_PAD;
                const size_t ar = GU ? (size_t)(ok ? tok[col] : 0u)
                                     : (size_t)blk * BN + col;
                pd_mma_cpa16p((unsigned char*)rb(buf) + col * (PD_KM_BSTR * 4u)
                                  + k16i * 16u,
                              xq + ar * in_dim + k0 + k16i * 16u, ok);
            }
            for (uint32_t i = tid; i < BN * 2u; i += 256u) {  // per-32 scales
                const uint32_t col = i >> 1u, h = i & 1u;
                const bool ok = !GU || tok[col] != PD_MOE_PAD;
                const size_t ar = GU ? (size_t)(ok ? tok[col] : 0u)
                                     : (size_t)blk * BN + col;
                pd_mma_cpa16p((unsigned char*)rxs(buf) + col * 48u + h * 16u,
                              xs + ar * nb32 + kt * 8u + h * 4u, ok);
            }
            if (MU) {
                for (uint32_t i = tid; i < BN * 4u; i += 256u) {  // per-16 sums
                    const uint32_t col = i >> 2u, h = i & 3u;
                    const bool ok = !GU || tok[col] != PD_MOE_PAD;
                    const size_t ar = GU ? (size_t)(ok ? tok[col] : 0u)
                                         : (size_t)blk * BN + col;
                    pd_mma_cpa16p((unsigned char*)rsu(buf) + col * 80u + h * 16u,
                                  xsums + ar * nb16 + kt * 16u + h * 4u, ok);
                }
            }
        };

        // compute the staged super in `buf` - the dense ks v2 compute verbatim
        // (inline fragment unpack off the raw strips, per-thread scale-record
        // expansion; zero-filled dead rows/cols vanish through zero scales)
        auto compute = [&](uint32_t buf) {
            const uint8_t* w0p = rw(buf) + (wr + g) * WSTR;
            const uint8_t* w8p = w0p + 8u * WSTR;
            const uint8_t* re0 = rrec(buf) + (wr + g) * PD_KQ_SCB;
            const uint8_t* re8 = re0 + 8u * PD_KQ_SCB;
            const int* rbv = rb(buf);
            const float* rxsv = rxs(buf);
            const float* rsuv = MU ? rsu(buf) : nullptr;

            float df0, dx0, df8, dx8;
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
                    sw0[j] = *(const uint32_t*)(re0 + 4u + 4u * j);
                    sw8[j] = *(const uint32_t*)(re8 + 4u + 4u * j);
                }
            }

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
                    s0lo = df0 * (float)(int8_t)(sw0[kk >> 1u] >> (8u * ((2u * kk) & 3u)));
                    s0hi = df0 * (float)(int8_t)(sw0[kk >> 1u] >> (8u * ((2u * kk + 1u) & 3u)));
                    s8lo = df8 * (float)(int8_t)(sw8[kk >> 1u] >> (8u * ((2u * kk) & 3u)));
                    s8hi = df8 * (float)(int8_t)(sw8[kk >> 1u] >> (8u * ((2u * kk + 1u) & 3u)));
                } else if (DT == PD_KQ_IQ4XS) {
                    d0s = df0 * (float)(int8_t)(sw0[kk >> 2u] >> (8u * (kk & 3u)));
                    d8s = df8 * (float)(int8_t)(sw8[kk >> 2u] >> (8u * (kk & 3u)));
                } else if (DT == PD_KQ_Q40) {
                    // {f16 dsub[8]} off the staged record; zero recs -> 0
                    __half h0v, h8v;
                    memcpy(&h0v, re0 + 2u * kk, 2u);
                    memcpy(&h8v, re8 + 2u * kk, 2u);
                    d0s = __half2float(h0v);
                    d8s = __half2float(h8v);
                    // value is the centered d*(q-8) already: mu stays 0
                } else {  // Q4_K / Q5_K
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

        // ST-deep ring, one commit group per iteration always (the dense ks
        // discipline); trailing barrier = write hazard fence for the next
        // issue into the just-read buffer - and, after the last iteration,
        // for the next mat's prologue into the same ring.
        #pragma unroll
        for (uint32_t s = 0; s + 1u < ST; ++s) {
            if (s < n_super) stage(s, s);
            pd_attn_cpa_commit();
        }
        uint32_t p = 0;
        for (uint32_t kt = 0; kt < n_super; ++kt) {
            const uint32_t pre = kt + (ST - 1u);
            if (pre < n_super) stage(pre, (p + ST - 1u) % ST);
            pd_attn_cpa_commit();
            pd_mma_cpa_waitN<(int)ST - 1>();
            __syncthreads();
            compute(p);
            __syncthreads();
            p = (p + 1u) % ST;
        }
        #pragma unroll
        for (uint32_t sub = 0; sub < NSUB; ++sub) {
            #pragma unroll
            for (uint32_t i = 0; i < 4u; ++i)
                (mat ? acc_u : acc_g)[sub][i] = acc[sub][i];
        }
    }

    if (GU) {
        // SwiGLU + per-32 quantize epilogue. PAD columns carry exact-zero
        // accs (zero-filled activations AND scales), so their fq/fs rows
        // write zeros - the flat fsums pass over the sorted rows needs that.
        float* sf = (float*)pd_kqm_sh;  // BN cols x 65-f32 stride (bank skew)
        #pragma unroll
        for (uint32_t sub = 0; sub < NSUB; ++sub) {
            const uint32_t c0 = wc + sub * 8u + 2u * t;
            #pragma unroll
            for (uint32_t q = 0; q < 4u; ++q) {
                const uint32_t c = c0 + (q & 1u);
                const uint32_t rl = wr + g + (q & 2u ? 8u : 0u);
                const float gv = acc_g[sub][q], uv = acc_u[sub][q];
                sf[c * 65u + rl] = (gv / (1.0f + __expf(-gv))) * uv;
            }
        }
        __syncthreads();
        const uint32_t n_sb = out_dim >> 5u;
        if (tid < BN * 2u) {
            const uint32_t col = tid >> 1u, half = tid & 1u;
            const uint32_t r0 = row_base + half * 32u;
            if (r0 < out_dim) {
                float amax = 0.0f;
                #pragma unroll
                for (uint32_t j = 0; j < 32u; ++j)
                    amax = fmaxf(amax, fabsf(sf[col * 65u + half * 32u + j]));
                const float scl = amax * (1.0f / 127.0f);
                const float inv = scl > 0.0f ? 1.0f / scl : 0.0f;
                const size_t frow = (size_t)blk * BN + col;
                #pragma unroll
                for (uint32_t j = 0; j < 32u; ++j) {
                    int qi = __float2int_rn(sf[col * 65u + half * 32u + j] * inv);
                    qi = qi < -127 ? -127 : (qi > 127 ? 127 : qi);
                    fq[frow * out_dim + r0 + j] = (int8_t)qi;
                }
                fs[frow * n_sb + (r0 >> 5u)] = scl;
            }
        }
    } else {
        // deterministic partials scatter: one writer per (token, slot, row)
        const uint32_t or0 = row_base + wr + g, or8 = or0 + 8u;
        #pragma unroll
        for (uint32_t sub = 0; sub < NSUB; ++sub) {
            const uint32_t c0 = wc + sub * 8u + 2u * t;
            #pragma unroll
            for (uint32_t q = 0; q < 4u; ++q) {
                const uint32_t c = c0 + (q & 1u);
                const uint32_t r = q & 2u ? or8 : or0;
                const unsigned int token = tok[c];
                if (token == PD_MOE_PAD || r >= out_dim) continue;
                const size_t pair = (size_t)token * n_active + slt[c];
                part[pair * out_dim + r] = topk_w[pair] * acc_g[sub][q];
            }
        }
    }
#else
    (void)wd0; (void)ws0; (void)wd1; (void)ws1; (void)sorted_row;
    (void)sorted_slot; (void)block_expert; (void)topk_w; (void)xq; (void)xs;
    (void)xsums; (void)fq; (void)fs; (void)part; (void)in_dim; (void)out_dim;
    (void)n_active;
#endif
}

// dynamic-smem opt-in per instantiation (Q5K/Q6K rings exceed the 48 KB
// static window; Q4K/IQ4 stay under -> 2 CTA/SM)
#define PD_KQM_LAUNCH(DTV, GUV, ...)                                          \
    do {                                                                      \
        constexpr uint32_t smem = pd_km_smem_bytes(DTV, 32u, 2u);             \
        if (smem > 48u * 1024u) {                                             \
            static cudaError_t attr = cudaFuncSetAttribute(                   \
                (const void*)pd_kq_moe_mma_kernel<DTV, GUV>,                  \
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);      \
            if (attr != cudaSuccess) return attr;                             \
        }                                                                     \
        pd_kq_moe_mma_kernel<DTV, GUV><<<grid, 256, smem, st>>>(__VA_ARGS__); \
    } while (0)

PD_EXPORT
int pd_kquant_moe_gate_up_mma(const void* gate_data, const void* gate_scales,
                              const void* up_data, const void* up_scales,
                              const void* sorted_row, const void* block_expert,
                              const void* xq, const void* xs, const void* xsums,
                              void* fq, void* fs, uint32_t in_dim, uint32_t ff,
                              uint32_t max_blocks, uint32_t dtype, void* stream) {
    if (ff == 0 || max_blocks == 0) return 0;
    if ((in_dim & 255u) != 0 || (ff & 31u) != 0) return cudaErrorInvalidValue;
    if (!pd_kq_valid(dtype)) return cudaErrorInvalidValue;
    if ((pd_kq_has_mu(dtype)) && xsums == nullptr)
        return cudaErrorInvalidValue;
    dim3 grid(max_blocks, (ff + 63u) / 64u);
    cudaStream_t st = (cudaStream_t)stream;
    switch (dtype) {
        #define PD_KQM_GU(DTV)                                                \
            PD_KQM_LAUNCH(DTV, true, (const uint8_t*)gate_data,               \
                (const uint8_t*)gate_scales, (const uint8_t*)up_data,         \
                (const uint8_t*)up_scales, (const unsigned int*)sorted_row,   \
                nullptr, (const unsigned int*)block_expert, nullptr,          \
                (const int8_t*)xq, (const float*)xs, (const float*)xsums,     \
                (int8_t*)fq, (float*)fs, nullptr, in_dim, ff, 0u)
        case PD_KQ_Q40: PD_KQM_GU(PD_KQ_Q40); break;
        case PD_KQ_Q4K: PD_KQM_GU(PD_KQ_Q4K); break;
        case PD_KQ_Q5K: PD_KQM_GU(PD_KQ_Q5K); break;
        case PD_KQ_Q6K: PD_KQM_GU(PD_KQ_Q6K); break;
        default: PD_KQM_GU(PD_KQ_IQ4XS); break;
        #undef PD_KQM_GU
    }
    return pd_launch_status();
}

PD_EXPORT
int pd_kquant_moe_down_mma(const void* down_data, const void* down_scales,
                           const void* sorted_row, const void* sorted_slot,
                           const void* block_expert, const void* topk_w,
                           const void* fq, const void* fs, const void* fsums,
                           void* part, uint32_t ff, uint32_t embd,
                           uint32_t n_active, uint32_t max_blocks,
                           uint32_t dtype, void* stream) {
    if (embd == 0 || max_blocks == 0) return 0;
    if ((ff & 255u) != 0 || (embd & 31u) != 0) return cudaErrorInvalidValue;
    if (!pd_kq_valid(dtype)) return cudaErrorInvalidValue;
    if ((pd_kq_has_mu(dtype)) && fsums == nullptr)
        return cudaErrorInvalidValue;
    dim3 grid(max_blocks, (embd + 63u) / 64u);
    cudaStream_t st = (cudaStream_t)stream;
    switch (dtype) {
        #define PD_KQM_DN(DTV)                                                \
            PD_KQM_LAUNCH(DTV, false, (const uint8_t*)down_data,              \
                (const uint8_t*)down_scales, nullptr, nullptr,                \
                (const unsigned int*)sorted_row,                              \
                (const unsigned int*)sorted_slot,                             \
                (const unsigned int*)block_expert, (const float*)topk_w,      \
                (const int8_t*)fq, (const float*)fs, (const float*)fsums,     \
                nullptr, nullptr, (float*)part, ff, embd, n_active)
        case PD_KQ_Q40: PD_KQM_DN(PD_KQ_Q40); break;
        case PD_KQ_Q4K: PD_KQM_DN(PD_KQ_Q4K); break;
        case PD_KQ_Q5K: PD_KQM_DN(PD_KQ_Q5K); break;
        case PD_KQ_Q6K: PD_KQM_DN(PD_KQ_Q6K); break;
        default: PD_KQM_DN(PD_KQ_IQ4XS); break;
        #undef PD_KQM_DN
    }
    return pd_launch_status();
}
#undef PD_KQM_LAUNCH
