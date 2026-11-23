// dflash.cuh - Muse Glimmer DFlash2 drafter kernels (stage B).
// Textually-included segment of the single pack translation unit.
// Not standalone-compilable: include order is defined by ../pack.cu.
//
// DFlash2 wraps every drafter sublayer in a GROUPED DYNAMIC CONVOLUTION: a
// depthwise conv along the TOKEN axis whose coefficients are a per-channel
// static (`base`) plus a per-token, per-GROUP delta predicted from the
// sublayer's own input by a projection GEMM. Groups are `group_size` adjacent
// channels sharing one delta, so the dynamic half costs embd/group_size
// numbers per token per tap instead of embd (muse: 416 instead of 6656).
//
//   out[row][c] = sum_t (base[side][t][c] + delta[row][side][t][g])
//                       * h[row - t][c] * (pos_in_block(row) >= t)
//
// with g = c / group_size. It runs twice per sublayer - `side` 0 before,
// `side` 1 after - off one projection whose row splits [2][taps][groups].
//
// The MASK is LOAD-BEARING, not an edge case. Row 0 of a block has no
// in-block predecessor, and the row physically before it in the plane belongs
// to a different SLOT's block, so an unmasked tap would convolve one
// sequence's draft into another's. The reference masks with
// `& (block_size - 1)` because it always runs the full trained block; a
// paddock draft round runs a RUNTIME block of `rows` = k+1 <= block_size
// (see dflash_draft_batch), so the mask here is `row % rows`. The two agree
// exactly when rows == block_size and rows is a power of two.
//
// The kernel reads h[row-1] while writing out[row], so `out` must not alias
// `h` - the caller passes a separate plane (there is no ordering between
// blocks to make an in-place walk safe).

// float4 down the channel axis: embd and group_size are both multiples of 4
// on every shipped DFlash2 geometry, so a quad never straddles two groups and
// one delta load serves all four channels. The launcher rc -2's anything else
// rather than silently taking a slower scalar path nothing has measured.
__global__ void pd_dflash_conv_kernel(const float4* __restrict__ h,
                                      float4* __restrict__ out,
                                      const float* __restrict__ base,
                                      const float* __restrict__ delta,
                                      uint32_t side, uint32_t e4,
                                      uint32_t taps, uint32_t ng,
                                      uint32_t gs4, uint32_t rows,
                                      uint32_t r) {
    PD_PDL_ARM();  // h is the predecessor GEMM's / rmsnorm's output
    uint32_t c4 = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y;
    if (c4 >= e4 || row >= r) return;
    uint32_t g = c4 / gs4;
    uint32_t pos = row % rows;
    uint32_t embd = e4 * 4u;
    size_t dstride = (size_t)2u * taps * ng;
    // `pos + 1` caps the tap walk instead of a per-tap predicate: tap 0 is
    // always in range and tap t needs t committed predecessors in-block.
    uint32_t tmax = taps < pos + 1u ? taps : pos + 1u;
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (uint32_t t = 0; t < tmax; ++t) {
        float d = delta[(size_t)row * dstride +
                        (size_t)(side * taps + t) * ng + g];
        const float* bt = base + (size_t)embd * (t + taps * side) + c4 * 4u;
        float4 b = *reinterpret_cast<const float4*>(bt);
        float4 hv = h[(size_t)(row - t) * e4 + c4];
        acc.x += (b.x + d) * hv.x;
        acc.y += (b.y + d) * hv.y;
        acc.z += (b.z + d) * hv.z;
        acc.w += (b.w + d) * hv.w;
    }
    out[(size_t)row * e4 + c4] = acc;
}

PD_EXPORT
int pd_dflash_conv(const void* h, void* out, const void* base,
                   const void* delta, uint32_t side, uint32_t embd,
                   uint32_t taps, uint32_t num_groups, uint32_t group_size,
                   uint32_t rows, uint32_t r, void* stream) {
    if (r == 0u || embd == 0u) return 0;
    if (rows == 0u || taps == 0u || side > 1u) return cudaErrorInvalidValue;
    // Geometry the float4 walk assumes, and the group split it assumes.
    if ((embd & 3u) || (group_size & 3u)) return -2;
    if (num_groups == 0u || num_groups * group_size != embd) return -2;
    if (h == out) return cudaErrorInvalidValue;  // reads row-1, writes row
    uint32_t e4 = embd >> 2;
    dim3 grid((e4 + 255u) / 256u, r, 1);
    pd_pdl_go(pd_dflash_conv_kernel, grid, 256u, 0u, (cudaStream_t)stream,
              (const float4*)h, (float4*)out, (const float*)base,
              (const float*)delta, side, e4, taps, num_groups,
              group_size >> 2, rows, r);
    return pd_launch_status();
}

// ---- DFlash2 candidate selector (stage C) -----------------------------------
//
// v1 drafts each row independently: argmax of that row's logits. Rows inside a
// block are conditionally independent given the context, so the block can be
// individually plausible and jointly incoherent - which is exactly what a
// verifier rejects. The selector answers that by scoring a PATH.
//
// Per position it keeps the top-K candidates and scores an EDGE from the
// candidate chosen at the previous position to each candidate here:
//
//   edge[p][c] = unary[c] + sum_r pred[prev_p][r] * hidden[r] * succ[cand_c][r]
//
// a rank-r bilinear form over two vocab-sized codebooks. The walk is GREEDY
// forward, not Viterbi: take the row of the edge matrix for the predecessor we
// actually chose, argmax it, carry the index. So only one row of the K x K
// matrix is ever needed per position, which is why this kernel is small.
//
// `unary` is the drafter's own logit epilogue applied to the raw top-K logit
// (scale then cap). Greedy per-row drafting could skip it - both halves are
// monotone, so an argmax does not care - but the selector ADDS the unary to a
// bilinear term, and addition does not commute with a softcap. Skipping it
// here would silently mis-weight every edge.

// Extract candidate ids out of pd_topk_rows' interleaved (id, logit-bits)
// pairs into the flat u32 plane pd_kquant_gather wants, and append each
// block's ANCHOR token at the tail so one gather serves both. The anchor is
// row 0 of the block - the committed token the block extends.
__global__ void pd_dflash_cand_ids_kernel(const uint32_t* __restrict__ topk,
                                          const uint32_t* __restrict__ toks,
                                          uint32_t* __restrict__ ids,
                                          uint32_t k, uint32_t rows,
                                          uint32_t r, uint32_t vocab) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t nk = r * k;
    if (i < nk) {
        uint32_t t = topk[(size_t)i * 2u];
        // pd_topk_rows pads a short row with 0xFFFFFFFF; a vocab this size
        // never pads, but a gather must not be handed one either way.
        ids[i] = t < vocab ? t : 0u;
    } else if (i < nk + r / rows) {
        ids[i] = toks[(size_t)(i - nk) * rows];
    }
}

// One CTA per block, walking that block's positions in order. Rows are
// [block][position]; position 0 is the committed anchor whose logits are
// discarded, so the walk covers positions 1..rows-1 and the first predecessor
// is the anchor's codebook row (parked at the tail of `pred` by the extract
// above).
__global__ void pd_dflash_select_kernel(const uint32_t* __restrict__ topk,
                                        const float* __restrict__ pred,
                                        const float* __restrict__ succ,
                                        const float* __restrict__ hs,
                                        uint32_t* __restrict__ out,
                                        float scale, float cap,
                                        uint32_t rank, uint32_t k,
                                        uint32_t rows, uint32_t r) {
    const uint32_t b = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5, nwarp = blockDim.x >> 5;
    extern __shared__ float s_score[];   // k floats
    const uint32_t nk = r * k;
    uint32_t previous = 0;
    for (uint32_t j = 1; j < rows; ++j) {
        const uint32_t row = b * rows + j;
        // predecessor: the anchor at the first position, else the candidate
        // this walk actually committed to one position back.
        const float* pe = (j == 1) ? pred + (size_t)(nk + b) * rank
                                   : pred + ((size_t)(row - 1) * k + previous) * rank;
        const float* h = hs + (size_t)row * rank;
        // Every warp iterates on a warp-UNIFORM bound, so a warp that runs out
        // of candidates skips the shuffles as a whole and never strands a
        // partial mask (see the warp-collective liveness rule).
        for (uint32_t c = warp; c < k; c += nwarp) {
            const float* se = succ + ((size_t)row * k + c) * rank;
            float acc = 0.0f;
            for (uint32_t t = lane; t < rank; t += 32u) acc += pe[t] * h[t] * se[t];
            for (uint32_t s = 16; s > 0; s >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, s);
            if (lane == 0) {
                float u = __uint_as_float(topk[((size_t)row * k + c) * 2u + 1u]) * scale;
                if (cap > 0.0f) u = tanhf(u / cap) * cap;
                s_score[c] = u + acc;
            }
        }
        __syncthreads();
        if (tid == 0) {
            uint32_t best = 0;
            float bv = s_score[0];
            for (uint32_t c = 1; c < k; ++c) {
                if (s_score[c] > bv) { bv = s_score[c]; best = c; }
            }
            out[row] = topk[((size_t)row * k + best) * 2u];
            s_score[k] = __uint_as_float(best);   // publish the carry
        }
        __syncthreads();
        previous = (uint32_t)__float_as_uint(s_score[k]);
        __syncthreads();
    }
}

PD_EXPORT
int pd_dflash_cand_ids(const void* topk, const void* toks, void* ids,
                       uint32_t k, uint32_t rows, uint32_t r, uint32_t vocab,
                       void* stream) {
    if (r == 0u || k == 0u || rows == 0u) return 0;
    if (r % rows) return cudaErrorInvalidValue;
    uint32_t total = r * k + r / rows;
    pd_dflash_cand_ids_kernel<<<(total + 255u) / 256u, 256u, 0, (cudaStream_t)stream>>>(
        (const uint32_t*)topk, (const uint32_t*)toks, (uint32_t*)ids, k, rows, r, vocab);
    return pd_launch_status();
}

PD_EXPORT
int pd_dflash_select(const void* topk, const void* pred, const void* succ,
                     const void* hs, void* out, float scale, float cap,
                     uint32_t rank, uint32_t k, uint32_t rows, uint32_t r,
                     void* stream) {
    if (r == 0u || rows < 2u) return 0;
    if (r % rows) return cudaErrorInvalidValue;
    if (k == 0u || k > 64u || (rank & 31u)) return -2;
    uint32_t n = r / rows;
    // k floats for the scores + 1 for the carry the walk publishes.
    pd_dflash_select_kernel<<<n, 256u, (k + 1u) * 4u, (cudaStream_t)stream>>>(
        (const uint32_t*)topk, (const float*)pred, (const float*)succ,
        (const float*)hs, (uint32_t*)out, scale, cap, rank, k, rows, r);
    return pd_launch_status();
}

// slot 974: the whole selector LATTICE, for a tree of drafts. The greedy walk
// above needs one row of each position's K x K edge matrix (the predecessor
// it committed to); a tree keeps several predecessors alive, so this scores
// every edge: scores[(blk * (rows - 1) + j - 1) * k * k + p * k + c] = the
// walk's edge (unary + rank-r bilinear form, the same expression and order)
// from candidate p one position back - the anchor at the first position,
// where every p row is the anchor's - to candidate c at position j, and
// ids[(blk * (rows - 1) + j - 1) * k + c] = candidate c's token. One CTA a
// (block, position), a warp a (p, c) edge with its lanes across the rank.
__global__ void pd_dflash_lattice_kernel(const uint32_t* __restrict__ topk,
                                         const float* __restrict__ pred,
                                         const float* __restrict__ succ,
                                         const float* __restrict__ hs,
                                         float* __restrict__ scores,
                                         uint32_t* __restrict__ ids,
                                         float scale, float cap,
                                         uint32_t rank, uint32_t k,
                                         uint32_t rows, uint32_t r) {
    const uint32_t e = blockIdx.x;                  // (block, position) edge set
    const uint32_t b = e / (rows - 1u), j = e % (rows - 1u) + 1u;
    const uint32_t row = b * rows + j;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5, nwarp = blockDim.x >> 5;
    const uint32_t nk = r * k;
    const float* h = hs + (size_t)row * rank;
    float* out = scores + (size_t)e * k * k;
    if (tid < k) ids[(size_t)e * k + tid] = topk[((size_t)row * k + tid) * 2u];
    for (uint32_t pc = warp; pc < k * k; pc += nwarp) {
        const uint32_t p = pc / k, c = pc % k;
        const float* pe = (j == 1u) ? pred + (size_t)(nk + b) * rank
                                    : pred + ((size_t)(row - 1u) * k + p) * rank;
        const float* se = succ + ((size_t)row * k + c) * rank;
        float acc = 0.0f;
        for (uint32_t t = lane; t < rank; t += 32u) acc += pe[t] * h[t] * se[t];
        for (uint32_t s = 16; s > 0; s >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, s);
        if (lane == 0) {
            float u = __uint_as_float(topk[((size_t)row * k + c) * 2u + 1u]) * scale;
            if (cap > 0.0f) u = tanhf(u / cap) * cap;
            out[pc] = u + acc;
        }
    }
}

PD_EXPORT
int pd_dflash_lattice(const void* topk, const void* pred, const void* succ, const void* hs,
                      void* scores, void* ids, float scale, float cap, uint32_t rank,
                      uint32_t k, uint32_t rows, uint32_t r, void* stream) {
    if (r == 0u || rows < 2u) return 0;
    if (r % rows) return cudaErrorInvalidValue;
    if (k == 0u || k > 64u || (rank & 31u)) return -2;
    const uint32_t edges = (r / rows) * (rows - 1u);
    pd_dflash_lattice_kernel<<<edges, 256u, 0, (cudaStream_t)stream>>>(
        (const uint32_t*)topk, (const float*)pred, (const float*)succ, (const float*)hs,
        (float*)scores, (uint32_t*)ids, scale, cap, rank, k, rows, r);
    return pd_launch_status();
}

// ---- rung G: SAMPLED selector walk + 16-candidate
// canonical rejection sampling -------------------------------------------
//
// Serving samples at temperature 0.7 (top-k 20 / top-p 0.95), and
// there the greedy walk above + the verify's "sample the target, accept on
// equality" rule accepts a draft with probability p(argmax q). The reference
// DFlash2 proposer (vLLM's dflash2/speculator.py `_selector_walk_
// kernel`) Gumbel-SAMPLES the walk at the request temperature and verifies
// with Leviathan/Chen rejection sampling against the cached draft
// distribution, accepting with min(1, p/q) - sum_x min(p(x), q(x)) per
// position instead of max p: measured 39% against 32% per-draft acceptance
// for the greedy form on the same cell. Lossless either way (the emitted stream is the
// target's distribution by construction); the lever is acceptance only.
//
// Two kernels, both DFlash-shaped: q lives on the row's K (=16) selector
// candidates and nowhere else, so the draft distribution is K floats per
// row, not a vocab-wide store (the gemma4 Phase-67 arm's fp16 q-store is
// the full-softmax MTP shape - a different drafter).
//
// Uniforms: one u32 seed per block from the slot's own seed stream (the
// service's RS chain draw), expanded on device by a counter hash over
// (seed, position, candidate). The graph bakes the seed BUFFER, the stage
// rewrites it per round like d_toks.

__device__ __forceinline__ uint32_t pd_dflash_mix32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}
// uniform in (0, 1): 24 mantissa bits, never exactly 0 so -log(-log u)
// stays finite
__device__ __forceinline__ float pd_dflash_uni(uint32_t seed, uint32_t j, uint32_t c) {
    const uint32_t h = pd_dflash_mix32(seed ^ pd_dflash_mix32(j * 0x9e3779b9u + c * 0x85ebca6bu + 0x1234567u));
    return ((h >> 8) + 0.5f) * (1.0f / 16777216.0f);
}

// Sampled twin of pd_dflash_select_kernel: per block, invt[b] > 0 runs the
// walk as Gumbel-max over s/T (== sampling softmax(s/T) position by
// position, conditioned on the chosen predecessor) and writes that row's
// full K-way q into q16[row*k + c]; invt[b] <= 0 is the greedy walk with q
// = one-hot (so a temperature-0 slot resolves under the classic rule
// exactly). The edge scoring is the greedy kernel's, verbatim.
__global__ void pd_dflash_select_rs_kernel(const uint32_t* __restrict__ topk,
                                           const float* __restrict__ pred,
                                           const float* __restrict__ succ,
                                           const float* __restrict__ hs,
                                           const float* __restrict__ invt,
                                           const uint32_t* __restrict__ seeds,
                                           uint32_t* __restrict__ out,
                                           float* __restrict__ q16,
                                           float scale, float cap,
                                           uint32_t rank, uint32_t k,
                                           uint32_t rows, uint32_t r) {
    const uint32_t b = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5, nwarp = blockDim.x >> 5;
    extern __shared__ float s_score[];   // k floats + carry
    const uint32_t nk = r * k;
    const float it = invt[b];
    const uint32_t seed = seeds[b];
    uint32_t previous = 0;
    for (uint32_t j = 1; j < rows; ++j) {
        const uint32_t row = b * rows + j;
        const float* pe = (j == 1) ? pred + (size_t)(nk + b) * rank
                                   : pred + ((size_t)(row - 1) * k + previous) * rank;
        const float* h = hs + (size_t)row * rank;
        for (uint32_t c = warp; c < k; c += nwarp) {
            const float* se = succ + ((size_t)row * k + c) * rank;
            float acc = 0.0f;
            for (uint32_t t = lane; t < rank; t += 32u) acc += pe[t] * h[t] * se[t];
            for (uint32_t s = 16; s > 0; s >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, s);
            if (lane == 0) {
                float u = __uint_as_float(topk[((size_t)row * k + c) * 2u + 1u]) * scale;
                if (cap > 0.0f) u = tanhf(u / cap) * cap;
                s_score[c] = u + acc;
            }
        }
        __syncthreads();
        if (tid == 0) {
            uint32_t best = 0;
            float* qrow = q16 + (size_t)row * k;
            if (it <= 0.0f) {
                float bv = s_score[0];
                for (uint32_t c = 1; c < k; ++c)
                    if (s_score[c] > bv) { bv = s_score[c]; best = c; }
                for (uint32_t c = 0; c < k; ++c) qrow[c] = (c == best) ? 1.0f : 0.0f;
            } else {
                float m = s_score[0];
                for (uint32_t c = 1; c < k; ++c) m = fmaxf(m, s_score[c]);
                float z = 0.0f, bg = -3.402823466e+38f;
                for (uint32_t c = 0; c < k; ++c) {
                    const float zc = (s_score[c] - m) * it;
                    const float e = expf(zc);
                    qrow[c] = e;            // unnormalized for now
                    z += e;
                    const float g = -logf(-logf(pd_dflash_uni(seed, j, c)));
                    const float v = zc + g;
                    if (v > bg) { bg = v; best = c; }
                }
                const float inv = (z > 0.0f) ? 1.0f / z : 0.0f;
                for (uint32_t c = 0; c < k; ++c) qrow[c] *= inv;
                if (!(z > 0.0f)) {          // all -inf: degenerate, one-hot at best
                    for (uint32_t c = 0; c < k; ++c) qrow[c] = (c == best) ? 1.0f : 0.0f;
                }
            }
            out[row] = topk[((size_t)row * k + best) * 2u];
            s_score[k] = __uint_as_float(best);
        }
        __syncthreads();
        previous = (uint32_t)__float_as_uint(s_score[k]);
        __syncthreads();
    }
}

PD_EXPORT
int pd_dflash_select_rs(const void* topk, const void* pred, const void* succ,
                        const void* hs, const void* invt, const void* seeds,
                        void* out, void* q16, float scale, float cap,
                        uint32_t rank, uint32_t k, uint32_t rows, uint32_t r,
                        void* stream) {
    if (r == 0u || rows < 2u) return 0;
    if (r % rows) return cudaErrorInvalidValue;
    if (k == 0u || k > 64u || (rank & 31u)) return -2;
    uint32_t n = r / rows;
    pd_dflash_select_rs_kernel<<<n, 256u, (k + 1u) * 4u, (cudaStream_t)stream>>>(
        (const uint32_t*)topk, (const float*)pred, (const float*)succ,
        (const float*)hs, (const float*)invt, (const uint32_t*)seeds,
        (uint32_t*)out, (float*)q16, scale, cap, rank, k, rows, r);
    return pd_launch_status();
}

// Verify resolve for the K-candidate draft distribution, TRUNCATION-AWARE.
// One CTA per verify row; rows whose PdSampleRow.mode != 7 are left alone
// (the sample_rows family owns them). Layout: verify rows are [block][j] of
// width k1; row j carries the target distribution that judges the draft at
// row j+1 (`toks[row+1]`, the device-assembled verify token), whose q is
// the DRAFTER row (srow*drows + j + 1) - srow = the block's index in the
// chain order (meta[n_blocks + i], the spec_toks meta plane). Head build +
// nucleus are pd_sample_rows_t_kernel's (top-K by scaled value, head
// softmax, min_p take-while, top_p INCLUSIVE cum) so p here is the
// distribution the dense sampler draws from - that equality is what makes
// the scheme lossless against our own non-spec path. Then:
//   accept  iff  u1 * q(d) < p(d)            (p(d) = 0 off the kept head)
//   reject  ->   sample  max(p - q, 0) / Zr   over the kept head with u2,
//                masked-argmax fallback when Zr rounds to 0 (never re-emit d:
//                the accept walk would book it as a match).
// par: PdSampleRow {inv_t, u1, mode=7, u2 bits}; trunc: PdSampleTrunc.
// 1024-thread launch => must fit in 64 regs/thread (65536 per SM); unbounded
// nvcc chose 96 and the launch was refused with 701. Same omission as task
// the earlier pd_sample_rows_t_kernel, found by the same register audit.
__global__ void __launch_bounds__(1024)
pd_dflash_rs_resolve_kernel(const float* __restrict__ logits,
                            const PdSampleRow* __restrict__ ps,
                            const PdSampleTrunc* __restrict__ pt,
                            const uint32_t* __restrict__ meta,
                            const uint32_t* __restrict__ toks,
                            const uint32_t* __restrict__ cand,
                            const float* __restrict__ q16,
                            unsigned int* __restrict__ out,
                            uint32_t n_blocks, uint32_t k1,
                            uint32_t drows, uint32_t k,
                            uint32_t n) {
    const uint32_t row = blockIdx.x, tid = threadIdx.x;
    if (ps[row].mode != 7u) return;
    const uint32_t bi = row / k1, j = row % k1;
    // unreachable by contract (the engine marks mode 7 only on rows with a
    // draft at j+1 inside the drafter's block); a safe real id, never the
    // draft, so a violated contract reads as a miss instead of an accept
    if (j + 1u >= k1 || j + 1u >= drows) { if (tid == 0) out[row] = toks[row]; return; }
    const float* x = logits + (size_t)row * n;
    const uint32_t kk = min(max(pt[row].k, 1u), PD_TOPK_HEAD);
    const uint32_t K = min(kk, n);

    // head build: pd_sample_rows_t_kernel verbatim (two-level histogram
    // threshold + gather + exact top-K selection by (okey desc, id asc))
    __shared__ uint32_t s_hist[2048];
    __shared__ uint32_t s_b1, s_b2, s_cg;
    for (uint32_t i = tid; i < 2048u; i += blockDim.x) s_hist[i] = 0;
    __syncthreads();
    for (uint32_t i = tid; i < n; i += blockDim.x)
        atomicAdd(&s_hist[pd_okey(x[i]) >> 21], 1u);
    __syncthreads();
    if (tid == 0) {
        uint32_t above = 0, b = 2047;
        for (;; --b) {
            if (above + s_hist[b] >= K || b == 0) break;
            above += s_hist[b];
        }
        s_b1 = b;
        s_cg = above;
    }
    __syncthreads();
    const uint32_t b1 = s_b1;
    const uint32_t cg1 = s_cg;
    for (uint32_t i = tid; i < 2048u; i += blockDim.x) s_hist[i] = 0;
    __syncthreads();
    for (uint32_t i = tid; i < n; i += blockDim.x) {
        const uint32_t key = pd_okey(x[i]);
        if ((key >> 21) == b1) atomicAdd(&s_hist[(key >> 10) & 0x7FFu], 1u);
    }
    __syncthreads();
    if (tid == 0) {
        uint32_t above = cg1, b = 2047;
        for (;; --b) {
            if (above + s_hist[b] >= K || b == 0) break;
            above += s_hist[b];
        }
        s_b2 = b;
    }
    __syncthreads();
    const uint32_t bfloor = (b1 << 11) | s_b2;
    __shared__ uint32_t s_ids[PD_TOPK_HEAD + PD_TOPK_ECAP];
    __shared__ uint32_t s_keys[PD_TOPK_HEAD + PD_TOPK_ECAP];
    __shared__ uint32_t s_gn, s_en;
    if (tid == 0) { s_gn = 0; s_en = 0; }
    __syncthreads();
    for (uint32_t i = tid; i < n; i += blockDim.x) {
        const uint32_t key = pd_okey(x[i]);
        const uint32_t top22 = key >> 10;
        if (top22 > bfloor) {
            const uint32_t p = atomicAdd(&s_gn, 1u);
            if (p < PD_TOPK_HEAD) { s_ids[p] = i; s_keys[p] = key; }
        } else if (top22 == bfloor) {
            const uint32_t p = atomicAdd(&s_en, 1u);
            if (p < PD_TOPK_ECAP) {
                s_ids[PD_TOPK_HEAD + p] = i;
                s_keys[PD_TOPK_HEAD + p] = key;
            }
        }
    }
    __syncthreads();
    if (tid != 0) return;

    const uint32_t gn = min(s_gn, (uint32_t)PD_TOPK_HEAD);
    const uint32_t en = min(s_en, (uint32_t)PD_TOPK_ECAP);
    uint32_t cid[PD_TOPK_HEAD];
    float cval[PD_TOPK_HEAD];
    uint32_t cn = 0;
    uint32_t used_mark = 0xFFFFFFFFu;
    for (; cn < K; ++cn) {
        uint32_t best = used_mark, bkey = 0, bid = used_mark;
        for (uint32_t a = 0; a < gn + en; ++a) {
            const uint32_t idx = a < gn ? a : PD_TOPK_HEAD + (a - gn);
            if (s_ids[idx] == used_mark) continue;
            if (best == used_mark || s_keys[idx] > bkey
                || (s_keys[idx] == bkey && s_ids[idx] < bid)) {
                best = idx;
                bkey = s_keys[idx];
                bid = s_ids[idx];
            }
        }
        if (best == used_mark) break;
        s_ids[best] = used_mark;
        cid[cn] = bid;
        cval[cn] = x[bid];
    }
    const uint32_t d = toks[row + 1u];
    if (cn == 0) { out[row] = (d != 0u) ? 0u : 1u; return; }
    const float inv_t = ps[row].inv_t;
    const float m = cval[0] * inv_t;
    if (!isfinite(m)) { out[row] = (cid[0] == d) ? d : cid[0]; return; }
    float p[PD_TOPK_HEAD];
    float head_sum = 0.0f;
    for (uint32_t a = 0; a < cn; ++a) {
        p[a] = expf(cval[a] * inv_t - m);
        head_sum += p[a];
    }
    if (!(head_sum > 0.0f)) { out[row] = (cid[0] == d) ? d : cid[0]; return; }
    for (uint32_t a = 0; a < cn; ++a) p[a] /= head_sum;
    uint32_t keep = cn;
    const float min_p = pt[row].min_p;
    if (min_p > 0.0f) {
        const float thresh = min_p * p[0];
        uint32_t s = 0;
        while (s < cn && p[s] >= thresh) ++s;
        keep = s;
    }
    const float top_p = pt[row].top_p;
    if (top_p < 1.0f) {
        float cum = 0.0f;
        uint32_t kp = keep;
        for (uint32_t a = 0; a < keep; ++a) {
            cum += p[a];
            if (cum >= top_p) { kp = a + 1u; break; }
        }
        keep = kp;
    }
    if (keep == 0u) keep = 1u;
    float total = 0.0f;
    for (uint32_t a = 0; a < keep; ++a) total += p[a];
    const float inv_total = (total > 0.0f) ? 1.0f / total : 0.0f;

    // the drafter row's q over its K candidates
    const uint32_t srow = meta[n_blocks + bi];
    const size_t qrow = (size_t)srow * drows + (j + 1u);
    const uint32_t* cr = cand + qrow * k;
    const float* qr = q16 + qrow * k;
    float qd = 0.0f;
    for (uint32_t c = 0; c < k; ++c) if (cr[c] == d) { qd = qr[c]; break; }
    float pd = 0.0f;
    for (uint32_t a = 0; a < keep; ++a) if (cid[a] == d) { pd = p[a] * inv_total; break; }
    const float u1 = ps[row].u;
    const float u2 = __uint_as_float(ps[row]._pad);
    // accept test, div-free: u1 < p(d)/q(d)  <=>  u1 * q(d) < p(d)
    if (pd > 0.0f && u1 * qd < pd) { out[row] = d; return; }

    // residual over the kept head: max(p - q, 0)
    float rres[PD_TOPK_HEAD];
    float zr = 0.0f;
    for (uint32_t a = 0; a < keep; ++a) {
        float qa = 0.0f;
        for (uint32_t c = 0; c < k; ++c) if (cr[c] == cid[a]) { qa = qr[c]; break; }
        const float rv = p[a] * inv_total - qa;
        rres[a] = rv > 0.0f ? rv : 0.0f;
        zr += rres[a];
    }
    if (!(zr > 0.0f)) {
        // rounding dust: masked argmax - the best kept token that is not d
        uint32_t pick = cid[0];
        if (pick == d) pick = (keep > 1u) ? cid[1] : ((d + 1u < n) ? d + 1u : d - 1u);
        out[row] = pick;
        return;
    }
    float rr = u2 * zr;
    uint32_t pick = 0xFFFFFFFFu;
    for (uint32_t a = 0; a < keep; ++a) {
        if (rres[a] <= 0.0f) continue;
        pick = cid[a];
        rr -= rres[a];
        if (rr <= 0.0f) break;
    }
    if (pick == 0xFFFFFFFFu || pick == d) {
        // last-with-mass backstop never lands on d (its residual is 0 on a
        // reject); the guard is belt-and-braces for NaN inputs
        pick = (cid[0] != d) ? cid[0] : ((keep > 1u) ? cid[1] : ((d + 1u < n) ? d + 1u : d - 1u));
    }
    out[row] = pick;
}

PD_EXPORT
int pd_dflash_rs_resolve(const void* logits, const void* params, const void* trunc,
                         const void* meta, const void* toks, const void* cand,
                         const void* q16, void* out, uint32_t rows,
                         uint32_t n_blocks, uint32_t k1, uint32_t drows,
                         uint32_t k, uint32_t n, void* stream) {
    if (rows == 0u || n == 0u) return 0;
    if (k1 == 0u || drows == 0u || k == 0u || k > 64u) return cudaErrorInvalidValue;
    pd_dflash_rs_resolve_kernel<<<rows, 1024u, 0, (cudaStream_t)stream>>>(
        (const float*)logits, (const PdSampleRow*)params, (const PdSampleTrunc*)trunc,
        (const uint32_t*)meta, (const uint32_t*)toks, (const uint32_t*)cand,
        (const float*)q16, (unsigned int*)out, n_blocks, k1, drows, k, n);
    return pd_launch_status();
}
