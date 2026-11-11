// attn/dash.cuh - Unlimited-OCR-2's SR-SWA ("DASH" in the checkpoint's code):
// the page-2D rope (slot 935), the router's block summaries (936) and
// route / select (937), the sparse attention over a paged f16 pool (938), and
// the page-count guard on the logits (939).
//
// One router per layer, single-head, shared by all attention heads. Per query:
//   q_r, k_r      the router's own 128-wide projections of the attention input,
//                 Gemma-RMS-normed, roped with the main branch's cos / sin
//   kbar_c        block c's summary: softmax_i(k_r[i] . head_cls / sqrt(128))
//                 weighted sum of the block's router keys (real positions only)
//   x_c           clamp(q_r . kbar_c / sqrt(128), +-1e4) * score_scale over the
//                 query's candidate blocks (remote: c <= qblk - local_blocks)
//   select        the top M (16) candidates, 1.5-entmax over them, the first K
//                 (12) read with bias log(max(p, floor)) - mean, / sigma
//   attend        the local window (128 keys, no bias) U the selected blocks
//                 (their bias added; a key the window covers is read once,
//                 unbiased), one softmax - fp32 throughout.
// All math fp32; the reference rounds q / P to bf16 in its prefill kernel, so
// agreement with it is at the bf16 level by construction.

#define PD_DASH_BS 32u    // keys a router block
#define PD_DASH_D 128u    // head dim == router index dim
#define PD_DASH_KCAP 16u  // capacity of a row's selected-block list (K <= 16)

// ---- 935: page-2D rope (+ the router's norms) -------------------------------
// Rotates, in place, every head of q and k and the router's q / k with the
// angles of the row's (tok, page parity) coordinates: pair j < 63 turns by
// tok * inv_tok[j], pair 63 by the parity (the one page pair's frequency is
// page_theta^0 = 1). Halves convention: dims (j, j + 64) form pair j. The two
// router planes are Gemma-RMS-normed first (x * rsqrt(mean(x^2) + eps), then
// * (1 + w), each step clamped to +-1e4 as the reference clamps), `rqw` / `rkw`
// carrying the folded 1 + w.
__global__ void pd_dash_rope2d_kernel(float* __restrict__ q, float* __restrict__ k,
                                      float* __restrict__ rq, float* __restrict__ rk,
                                      const float* __restrict__ rqw,
                                      const float* __restrict__ rkw,
                                      const uint32_t* __restrict__ coords,
                                      const float* __restrict__ inv_tok, uint32_t n_head,
                                      uint32_t rstride, float eps) {
    __shared__ float red[2];
    const uint32_t row = blockIdx.x, plane = blockIdx.y, j = threadIdx.x;  // j < 64
    float* x;
    const float* nw = nullptr;
    if (plane < n_head) x = q + ((size_t)row * n_head + plane) * PD_DASH_D;
    else if (plane < 2u * n_head) x = k + ((size_t)row * n_head + plane - n_head) * PD_DASH_D;
    else if (plane == 2u * n_head) { x = rq + (size_t)row * rstride; nw = rqw; }
    else { x = rk + (size_t)row * rstride; nw = rkw; }
    float x1 = x[j], x2 = x[j + 64u];
    if (nw != nullptr) {   // uniform across the block: router planes only
        float ss = x1 * x1 + x2 * x2;
        #pragma unroll
        for (uint32_t o = 16u; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        if ((j & 31u) == 0) red[j >> 5] = ss;
        __syncthreads();
        const float inv = rsqrtf((red[0] + red[1]) / (float)PD_DASH_D + eps);
        x1 = fminf(fmaxf(fminf(fmaxf(x1 * inv, -1e4f), 1e4f) * nw[j], -1e4f), 1e4f);
        x2 = fminf(fmaxf(fminf(fmaxf(x2 * inv, -1e4f), 1e4f) * nw[j + 64u], -1e4f), 1e4f);
    }
    const uint32_t tok = coords[2u * row], par = coords[2u * row + 1u];
    const float ang = j < 63u ? (float)tok * inv_tok[j] : (float)par;
    float s, c;
    sincosf(ang, &s, &c);
    x[j] = x1 * c - x2 * s;
    x[j + 64u] = x2 * c + x1 * s;
}

PD_EXPORT
int pd_dash_rope2d(void* q, void* k, void* rq, void* rk, const void* rqw, const void* rkw,
                   const void* coords, const void* inv_tok, uint32_t rows, uint32_t n_head,
                   uint32_t rstride, float eps, void* stream) {
    if (rows == 0) return 0;
    pd_dash_rope2d_kernel<<<dim3(rows, 2u * n_head + 2u), 64, 0, (cudaStream_t)stream>>>(
        (float*)q, (float*)k, (float*)rq, (float*)rk, (const float*)rqw, (const float*)rkw,
        (const uint32_t*)coords, (const float*)inv_tok, n_head, rstride, eps);
    return (int)cudaGetLastError();
}

// ---- 936: block summaries ---------------------------------------------------
// One CTA per item = one (slot, block) the pass touched. The block's keys are
// its first n_prev positions from the slot's staging rows (keys an earlier pass
// wrote; staging row = position mod 32) followed by n_chunk chunk rows from
// rk at row0. kbar is rewritten whole; with `stage` set the chunk's keys are
// copied into staging afterwards, so the block can be completed later.
// items: [slot, blk, row0, n_chunk, n_prev, stage] u32.
__global__ void __launch_bounds__(128) pd_dash_summary_kernel(
    const float* __restrict__ rk, uint32_t rstride, float* __restrict__ staging,
    float* __restrict__ kbar, const float* __restrict__ head_cls,
    const uint32_t* __restrict__ items, uint32_t kbar_blocks) {
    __shared__ float key[PD_DASH_BS][PD_DASH_D];
    __shared__ float aw[PD_DASH_BS];
    const uint32_t* it = items + (size_t)blockIdx.x * 6u;
    const uint32_t slot = it[0], blk = it[1], row0 = it[2], n_chunk = it[3], n_prev = it[4];
    const bool stage = it[5] != 0u;
    const uint32_t n = n_prev + n_chunk, d = threadIdx.x, warp = d >> 5, lane = d & 31u;
    if (n_chunk == 0u) return;   // a decode row with nothing to summarize
    float* st = staging + (size_t)slot * PD_DASH_BS * PD_DASH_D;
    for (uint32_t i = 0; i < n; ++i)
        key[i][d] = i < n_prev ? st[i * PD_DASH_D + d]
                               : rk[(size_t)(row0 + i - n_prev) * rstride + d];
    __syncthreads();
    for (uint32_t i = warp; i < n; i += 4u) {
        float s = 0.0f;
        #pragma unroll
        for (uint32_t c = 0; c < 4u; ++c) s += key[i][lane * 4u + c] * head_cls[lane * 4u + c];
        #pragma unroll
        for (uint32_t o = 16u; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (lane == 0) aw[i] = fminf(fmaxf(s * 0.08838834764831845f, -1e4f), 1e4f);
    }
    __syncthreads();
    if (warp == 0) {
        const float a = lane < n ? aw[lane] : -INFINITY;
        float m = a;
        #pragma unroll
        for (uint32_t o = 16u; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        const float e = lane < n ? expf(a - m) : 0.0f;
        float sum = e;
        #pragma unroll
        for (uint32_t o = 16u; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
        if (lane < n) aw[lane] = e / fmaxf(sum, 1e-6f);
    }
    __syncthreads();
    float acc = 0.0f;
    for (uint32_t i = 0; i < n; ++i) acc += aw[i] * key[i][d];
    kbar[((size_t)slot * kbar_blocks + blk) * PD_DASH_D + d] = acc;
    if (stage)
        for (uint32_t i = n_prev; i < n; ++i) st[i * PD_DASH_D + d] = key[i][d];
}

PD_EXPORT
int pd_dash_summary(const void* rk, uint32_t rstride, void* staging, void* kbar,
                    const void* head_cls, const void* items, uint32_t n_items,
                    uint32_t kbar_blocks, void* stream) {
    if (n_items == 0) return 0;
    pd_dash_summary_kernel<<<n_items, 128, 0, (cudaStream_t)stream>>>(
        (const float*)rk, rstride, (float*)staging, (float*)kbar, (const float*)head_cls,
        (const uint32_t*)items, kbar_blocks);
    return (int)cudaGetLastError();
}

// ---- 937: route / select ----------------------------------------------------
// One warp per row. rinfo: [slot, pos, lo, hi, nblk_m, cap] u32 - the row's
// page block range (lo > hi: every block), the key length's block count that
// sizes the entmax window (prefill: the whole prompt's; decode: the cache's),
// and the last block any candidate may be (a decode row's kept prompt).
// Candidates are [lo, hi] clipped to the remote blocks, or every remote block
// when that leaves none. Scores sit in dynamic shared memory (kbar_blocks
// floats a row).
__global__ void __launch_bounds__(32) pd_dash_route_kernel(
    const float* __restrict__ rq, uint32_t rstride, const float* __restrict__ kbar,
    uint32_t kbar_blocks, const uint32_t* __restrict__ rinfo, int32_t* __restrict__ out_blk,
    float* __restrict__ out_bias, uint32_t* __restrict__ out_n, uint32_t local_blocks,
    uint32_t M, uint32_t K, float score_scale, float floor_p, float inv_sigma) {
    extern __shared__ float pd_dash_sc[];
    __shared__ float qs[PD_DASH_D];
    const uint32_t row = blockIdx.x, lane = threadIdx.x;
    const uint32_t* ri = rinfo + (size_t)row * 6u;
    const uint32_t slot = ri[0], pos = ri[1], lo = ri[2], hi = ri[3], nblk_m = ri[4], cap = ri[5];
    int qmax = (int)(pos / PD_DASH_BS) - (int)local_blocks;
    if (cap < kbar_blocks && qmax > (int)cap) qmax = (int)cap;
    if (qmax < 0) {
        if (lane == 0) out_n[row] = 0u;
        return;
    }
    uint32_t clo = 0, chi = (uint32_t)qmax;
    if (lo <= hi) {
        const uint32_t h = hi < chi ? hi : chi;
        if (lo <= h) { clo = lo; chi = h; }
    }
    const uint32_t nc = chi - clo + 1u;
    for (uint32_t d = lane; d < PD_DASH_D; d += 32u) qs[d] = rq[(size_t)row * rstride + d];
    __syncwarp();
    const float* kb = kbar + (size_t)slot * kbar_blocks * PD_DASH_D;
    for (uint32_t c = lane; c < nc; c += 32u) {
        const float* kr = kb + (size_t)(clo + c) * PD_DASH_D;
        float s = 0.0f;
        for (uint32_t d = 0; d < PD_DASH_D; ++d) s += qs[d] * kr[d];
        pd_dash_sc[c] = fminf(fmaxf(s * 0.08838834764831845f, -1e4f), 1e4f) * score_scale;
    }
    __syncwarp();
    const uint32_t m_eff = M < nblk_m ? M : nblk_m;
    const uint32_t m = m_eff < nc ? m_eff : nc;
    const uint32_t k_eff = K < m_eff ? K : m_eff;
    const uint32_t n_sel = k_eff < nc ? k_eff : nc;
    float xv = -INFINITY;   // lane r holds the r-th largest
    uint32_t xi = 0;
    for (uint32_t r = 0; r < m; ++r) {
        float bv = -INFINITY;
        uint32_t bi = 0xFFFFFFFFu;
        for (uint32_t c = lane; c < nc; c += 32u) {
            const float v = pd_dash_sc[c];
            if (v > bv) { bv = v; bi = c; }   // ascending c: the first max wins
        }
        #pragma unroll
        for (uint32_t o = 16u; o; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const uint32_t oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == r) { xv = bv; xi = bi; }
        __syncwarp();
        if (lane == 0) pd_dash_sc[bi] = -INFINITY;
        __syncwarp();
    }
    // 1.5-entmax over the m sorted values (the reference's sort-based form)
    const float x0 = __shfl_sync(0xffffffffu, xv, 0);
    const float X = lane < m ? (xv - x0) * 0.5f : 0.0f;
    float cs = X, cs2 = X * X;
    #pragma unroll
    for (uint32_t o = 1u; o < 32u; o <<= 1) {
        const float a = __shfl_up_sync(0xffffffffu, cs, o);
        const float b = __shfl_up_sync(0xffffffffu, cs2, o);
        if (lane >= o) { cs += a; cs2 += b; }
    }
    const float rho = (float)(lane + 1u);
    const float mean = cs / rho, msq = cs2 / rho;
    const float ss = rho * (msq - mean * mean);
    const float delta = (1.0f - ss) / rho;
    const float tau = mean - sqrtf(fmaxf(delta, 0.0f));
    const uint32_t supp = __ballot_sync(0xffffffffu, lane < m && tau <= X);
    const uint32_t s = max(1u, (uint32_t)__popc(supp));
    const float taus = __shfl_sync(0xffffffffu, tau, s - 1u);
    const float y = fmaxf(X - taus, 0.0f);
    const float p = y * y;
    const float logp = lane < n_sel ? logf(fmaxf(p, floor_p)) : 0.0f;
    float sum = logp;
    #pragma unroll
    for (uint32_t o = 16u; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    const float mu = sum / (float)max(1u, n_sel);
    if (lane < n_sel) {
        out_blk[(size_t)row * PD_DASH_KCAP + lane] = (int32_t)(clo + xi);
        out_bias[(size_t)row * PD_DASH_KCAP + lane] = (logp - mu) * inv_sigma;
    }
    if (lane == 0) out_n[row] = n_sel;
}

PD_EXPORT
int pd_dash_route(const void* rq, uint32_t rstride, const void* kbar, uint32_t kbar_blocks,
                  const void* rinfo, void* out_blk, void* out_bias, void* out_n, uint32_t rows,
                  uint32_t local_blocks, uint32_t M, uint32_t K, float score_scale,
                  float floor_p, float inv_sigma, void* stream) {
    if (rows == 0) return 0;
    if (K > PD_DASH_KCAP || M > 32u || M < K) return cudaErrorInvalidValue;
    const size_t smem = (size_t)kbar_blocks * sizeof(float);
    static size_t opted = 48u * 1024u;
    if (smem > opted) {
        if (cudaFuncSetAttribute(pd_dash_route_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)smem) != cudaSuccess)
            return cudaErrorInvalidValue;
        opted = smem;
    }
    pd_dash_route_kernel<<<rows, 32, smem, (cudaStream_t)stream>>>(
        (const float*)rq, rstride, (const float*)kbar, kbar_blocks, (const uint32_t*)rinfo,
        (int32_t*)out_blk, (float*)out_bias, (uint32_t*)out_n, local_blocks, M, K, score_scale,
        floor_p, inv_sigma);
    return (int)cudaGetLastError();
}

// ---- 938: the sparse attention ----------------------------------------------
// One CTA per (row, head), 128 threads. Key slot i < 128: the window key
// w0 + i (w0 = max(0, pos - window + 1)); i >= 128: key u of selected block s,
// kept only if it lies before the window (the window reads it unbiased).
//
// Interim, knowingly: SIMT fp32 over at most 512 keys a row, the same kernel
// for prompt and decode rows. Decode reads ~10 x 512 keys a step and is not
// where the time goes; PREFILL is - every prompt row gathers its own 512 keys
// per head, nothing shared between neighbouring rows. The SOTA target is the
// block-sparse tensor-core class (NSA / DeepSeek sparse-attention kernels): a
// query tile's rows share their window and mostly their selected blocks (the
// router's candidates are the PAGE's blocks), so the union of a tile's blocks
// loads once into smem and runs as mma with a per-row mask + bias.
// ainfo: [slot, pos, keep, ring] u32 - logical position j lives at pool row
// j < keep ? j : keep + (j - keep) % ring (the ring after the kept prompt).
__device__ __forceinline__ uint32_t pd_dash_phys(uint32_t j, uint32_t keep, uint32_t ring) {
    return j < keep ? j : keep + (j - keep) % ring;
}

__global__ void __launch_bounds__(128) pd_dash_attn_f16_kernel(
    const float* __restrict__ q, const __half* __restrict__ kpool,
    const __half* __restrict__ vpool, float* __restrict__ out,
    const uint32_t* __restrict__ block_tables, uint32_t bps, const uint32_t* __restrict__ ainfo,
    const int32_t* __restrict__ sel_blk, const float* __restrict__ sel_bias,
    const uint32_t* __restrict__ sel_n, uint32_t n_head, float scale, uint32_t window) {
    constexpr uint32_t NK = 128u + PD_DASH_KCAP * PD_DASH_BS;
    __shared__ float qs[PD_DASH_D];
    __shared__ float lg[NK];
    __shared__ uint32_t rowof[NK];
    __shared__ float red[4];
    const uint32_t row = blockIdx.x, h = blockIdx.y, t = threadIdx.x;
    const uint32_t* ai = ainfo + (size_t)row * 4u;
    const uint32_t slot = ai[0], pos = ai[1], keep = ai[2], ring = ai[3];
    const uint32_t kv_dim = n_head * PD_DASH_D;
    const uint32_t w0 = pos + 1u > window ? pos + 1u - window : 0u;
    const uint32_t nsel = sel_n[row];
    qs[t] = q[((size_t)row * n_head + h) * PD_DASH_D + t];
    __syncthreads();
    const uint32_t* bt = block_tables + (size_t)slot * bps;
    float mx = -INFINITY;
    for (uint32_t i = t; i < NK; i += 128u) {
        uint32_t j = 0;
        bool ok;
        float b = 0.0f;
        if (i < 128u) {
            j = w0 + i;
            ok = j <= pos;
        } else {
            const uint32_t s = (i - 128u) / PD_DASH_BS, u = (i - 128u) % PD_DASH_BS;
            ok = s < nsel;
            if (ok) {
                j = (uint32_t)sel_blk[(size_t)row * PD_DASH_KCAP + s] * PD_DASH_BS + u;
                ok = j < w0;
                b = sel_bias[(size_t)row * PD_DASH_KCAP + s];
            }
        }
        float l = -INFINITY;
        uint32_t pr = 0;
        if (ok) {
            const uint32_t ph = pd_dash_phys(j, keep, ring);
            pr = bt[ph >> 4] * 16u + (ph & 15u);
            const uint4* kp = reinterpret_cast<const uint4*>(kpool + (size_t)pr * kv_dim + h * PD_DASH_D);
            float dot = 0.0f;
            #pragma unroll 4
            for (uint32_t c = 0; c < PD_DASH_D / 8u; ++c) {
                const uint4 w = kp[c];
                const __half2* hw = reinterpret_cast<const __half2*>(&w);
                #pragma unroll
                for (uint32_t e = 0; e < 4u; ++e) {
                    const float2 f = __half22float2(hw[e]);
                    dot += qs[c * 8u + 2u * e] * f.x + qs[c * 8u + 2u * e + 1u] * f.y;
                }
            }
            l = dot * scale + b;
        }
        lg[i] = l;
        rowof[i] = ok ? pr : 0xFFFFFFFFu;
        mx = fmaxf(mx, l);
    }
    #pragma unroll
    for (uint32_t o = 16u; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    if ((t & 31u) == 0) red[t >> 5] = mx;
    __syncthreads();
    mx = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
    __syncthreads();
    float sum = 0.0f;
    for (uint32_t i = t; i < NK; i += 128u) {
        const float e = rowof[i] != 0xFFFFFFFFu ? expf(lg[i] - mx) : 0.0f;
        lg[i] = e;
        sum += e;
    }
    #pragma unroll
    for (uint32_t o = 16u; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    if ((t & 31u) == 0) red[t >> 5] = sum;
    __syncthreads();
    sum = red[0] + red[1] + red[2] + red[3];
    float acc = 0.0f;
    for (uint32_t i = 0; i < NK; ++i) {
        const uint32_t pr = rowof[i];
        if (pr == 0xFFFFFFFFu) continue;
        acc += lg[i] * __half2float(vpool[(size_t)pr * kv_dim + h * PD_DASH_D + t]);
    }
    out[((size_t)row * n_head + h) * PD_DASH_D + t] = acc / sum;
}

PD_EXPORT
int pd_dash_attn_f16(const void* q, const void* kpool, const void* vpool, void* out,
                     const void* block_tables, uint32_t bps, const void* ainfo,
                     const void* sel_blk, const void* sel_bias, const void* sel_n, uint32_t rows,
                     uint32_t n_head, float scale, uint32_t window, void* stream) {
    if (rows == 0) return 0;
    if (window != 128u) return cudaErrorInvalidValue;  // the window slots are 128 wide
    pd_dash_attn_f16_kernel<<<dim3(rows, n_head), 128, 0, (cudaStream_t)stream>>>(
        (const float*)q, (const __half*)kpool, (const __half*)vpool, (float*)out,
        (const uint32_t*)block_tables, bps, (const uint32_t*)ainfo, (const int32_t*)sel_blk,
        (const float*)sel_bias, (const uint32_t*)sel_n, n_head, scale, window);
    return (int)cudaGetLastError();
}

// ---- 939: the page-count guard ----------------------------------------------
// The reference's guard, per row: 0 nothing, 1 EOS banned (fewer pages opened
// than the prompt holds), 2 EOS only (more pages than the prompt), 3 <PAGE>
// only (a page hit its token cap with pages still to open).
__global__ void pd_dash_guard_kernel(float* __restrict__ logits, const uint32_t* __restrict__ mode,
                                     uint32_t vocab, uint32_t eos, uint32_t page) {
    const uint32_t row = blockIdx.y, v = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t md = mode[row];
    if (md == 0u || v >= vocab) return;
    float* l = logits + (size_t)row * vocab;
    if (md == 1u) {
        if (v == eos) l[v] = -INFINITY;
    } else if (md == 2u) {
        if (v != eos) l[v] = -INFINITY;
    } else if (md == 3u) {
        if (v != page) l[v] = -INFINITY;
    }
}

PD_EXPORT
int pd_dash_guard(void* logits, const void* mode, uint32_t rows, uint32_t vocab, uint32_t eos,
                  uint32_t page, void* stream) {
    if (rows == 0) return 0;
    pd_dash_guard_kernel<<<dim3((vocab + 255u) / 256u, rows), 256, 0, (cudaStream_t)stream>>>(
        (float*)logits, (const uint32_t*)mode, vocab, eos, page);
    return (int)cudaGetLastError();
}
