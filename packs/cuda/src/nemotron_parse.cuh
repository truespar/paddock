// nemotron_parse.cuh - Nemotron 3.5 Parse's own kernels (slots 890-892).
//
// The tower and the decoder run on kernels the pack already has (the f16
// tower GEMMs, the half attention, the DINOv3 seams, the paged decoder
// attention); what this family adds is small and specific:
//
// 890 pd_np_patch_u8 - the processor's pixel math on the device, for a
//     RECTANGULAR canvas (every other patch stem here is square or fixed):
//     u8 RGB -> /255 -> (x - mean) / std -> the 16x16 patch flattened
//     (c, ky, kx), one f16 row a patch. A page crosses the bus as 3 bytes a
//     pixel instead of 1536 per patch row. The reference does the same math in
//     f32 eager ops (`div_(255.0)`, then `(page - mean) / std`); each step is
//     its own rounding there, so each is an explicit IEEE intrinsic here -
//     nothing may contract or approximate a divide.
//
// 891 pd_np_pen_argmax - the checkpoint's repetition penalty (its
//     generation_config: 1.1) applied to a row of logits, then the greedy pick.
//     HF semantics (`RepetitionPenaltyLogitsProcessor`): every DISTINCT id the
//     sequence has seen is penalized once - a logit < 0 is multiplied by the
//     penalty, any other is divided. "Seen" is a per-slot bitmap (the prompt and
//     every committed token) plus up to `n_extra` per-row ids: a spec verify
//     row's own uncommitted chunk tokens, which the reference would already have
//     appended before scoring that position. Ties pick the lowest id, as
//     torch.argmax does. With `write_back` the penalized logits replace the
//     row, so a host reader sees exactly what the device picked from.
//
// 892 pd_np_seen_or - set the bits of committed ids in their slots' bitmaps.
//
// Why the penalty lives in the model and not in the service's sampler: the
// service takes any penalized sequence off the device and speculative paths
// (a penalty mutates logits from history). For this checkpoint the penalty
// is part of its published decoding contract, so the family owns it, on the
// device, and the service sees a greedy request - MTP verify rows stay exact.

#define PD_NP_ARGMAX_THREADS 1024u

__global__ void __launch_bounds__(256) pd_np_patch_u8_kernel(
    const uint8_t* __restrict__ canvas, __half* __restrict__ out, uint32_t h, uint32_t w,
    float m0, float m1, float m2, float s0, float s1, float s2) {
    const uint32_t cols = w / 16u;
    const uint32_t p = blockIdx.x;
    const uint32_t pr = p / cols, pc = p - pr * cols;
    const float mean[3] = {m0, m1, m2};
    const float sd[3] = {s0, s1, s2};
    __half* orow = out + (size_t)p * 768u;
    for (uint32_t i = threadIdx.x; i < 768u; i += blockDim.x) {
        const uint32_t c = i >> 8, ky = (i >> 4) & 15u, kx = i & 15u;
        const uint32_t y = pr * 16u + ky, x = pc * 16u + kx;
        const float u = (float)canvas[((size_t)y * w + x) * 3u + c];
        const float v = __fdiv_rn(u, 255.0f);
        orow[i] = __float2half_rn(__fdiv_rn(__fsub_rn(v, mean[c]), sd[c]));
    }
}

PD_EXPORT
int pd_np_patch_u8(const void* canvas, void* out, uint32_t h, uint32_t w, float m0, float m1,
                   float m2, float s0, float s1, float s2, void* stream) {
    if (h == 0 || w == 0) return 0;
    if ((h % 16u) != 0u || (w % 16u) != 0u) return cudaErrorInvalidValue;
    const uint32_t n = (h / 16u) * (w / 16u);
    pd_np_patch_u8_kernel<<<n, 256, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)canvas, (__half*)out, h, w, m0, m1, m2, s0, s1, s2);
    return pd_launch_status();
}

__global__ void __launch_bounds__(PD_NP_ARGMAX_THREADS) pd_np_pen_argmax_kernel(
    float* __restrict__ logits, const uint32_t* __restrict__ seen,
    const uint32_t* __restrict__ row_slot, const uint32_t* __restrict__ extra, uint32_t n_extra,
    float penalty, uint32_t vocab, uint32_t words, uint32_t write_back,
    uint32_t* __restrict__ out_ids) {
    const uint32_t row = blockIdx.x;
    float* lr = logits + (size_t)row * vocab;
    const uint32_t* sb = seen + (size_t)row_slot[row] * words;
    const uint32_t* ex = extra + (size_t)row * n_extra;
    float best = -INFINITY;
    uint32_t bi = 0xffffffffu;
    for (uint32_t v = threadIdx.x; v < vocab; v += blockDim.x) {
        float l = lr[v];
        bool pen = (sb[v >> 5] >> (v & 31u)) & 1u;
        for (uint32_t e = 0; e < n_extra; ++e) pen |= ex[e] == v;
        if (pen) {
            l = l < 0.0f ? __fmul_rn(l, penalty) : __fdiv_rn(l, penalty);
            if (write_back) lr[v] = l;
        }
        // strictly greater keeps the lowest id among equals within a thread
        // (its ids rise); the reduction below breaks ties toward the lower id
        if (l > best) {
            best = l;
            bi = v;
        }
    }
    __shared__ float s_v[32];
    __shared__ uint32_t s_i[32];
    for (uint32_t o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, best, o);
        const uint32_t oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > best || (ov == best && oi < bi)) {
            best = ov;
            bi = oi;
        }
    }
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    if (lane == 0) {
        s_v[warp] = best;
        s_i[warp] = bi;
    }
    __syncthreads();
    if (warp == 0) {
        const uint32_t nw = blockDim.x >> 5;
        best = lane < nw ? s_v[lane] : -INFINITY;
        bi = lane < nw ? s_i[lane] : 0xffffffffu;
        for (uint32_t o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, best, o);
            const uint32_t oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > best || (ov == best && oi < bi)) {
                best = ov;
                bi = oi;
            }
        }
        // an all-NaN row has no pick; id 0 keeps the output defined
        if (lane == 0) out_ids[row] = bi == 0xffffffffu ? 0u : bi;
    }
}

PD_EXPORT
int pd_np_pen_argmax(void* logits, const void* seen, const void* row_slot, const void* extra,
                     uint32_t n_extra, float penalty, uint32_t vocab, uint32_t words,
                     uint32_t rows, uint32_t write_back, void* out_ids, void* stream) {
    if (rows == 0) return 0;
    if (vocab == 0 || words * 32u < vocab || !(penalty > 0.0f)) return cudaErrorInvalidValue;
    pd_np_pen_argmax_kernel<<<rows, PD_NP_ARGMAX_THREADS, 0, (cudaStream_t)stream>>>(
        (float*)logits, (const uint32_t*)seen, (const uint32_t*)row_slot,
        (const uint32_t*)extra, n_extra, penalty, vocab, words, write_back,
        (uint32_t*)out_ids);
    return pd_launch_status();
}

__global__ void pd_np_seen_or_kernel(uint32_t* __restrict__ seen,
                                     const uint32_t* __restrict__ slots,
                                     const uint32_t* __restrict__ toks, uint32_t n,
                                     uint32_t words) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t t = toks[i];
    // an id past the bitmap is a caller bug; dropping it beats a stray write
    if (t >= words * 32u) return;
    atomicOr(seen + (size_t)slots[i] * words + (t >> 5), 1u << (t & 31u));
}

PD_EXPORT
int pd_np_seen_or(void* seen, const void* slots, const void* toks, uint32_t n, uint32_t words,
                  void* stream) {
    if (n == 0) return 0;
    pd_np_seen_or_kernel<<<(n + 255u) / 256u, 256, 0, (cudaStream_t)stream>>>(
        (uint32_t*)seen, (const uint32_t*)slots, (const uint32_t*)toks, n, words);
    return pd_launch_status();
}

// 893 pd_np_vit_attn - the tower's self-attention at head_dim 72 EXACTLY.
//
// The shared vision kernel (620) pads 72 to 80 (and FA2's head-dim table
// jumps 64 -> 96, the padding vLLM's RADIO pays); 72 is 4 k16 steps plus one
// k8 step of the QK mma and 9 n8 tiles of the PV mma, so nothing here
// multiplies a padded dim. It is FA2's forward for an unmasked sequence:
//   - 8 warps x 16 query rows = 128-row blocks: each staged K/V tile feeds
//     twice the queries of 620's 64-row block;
//   - K/V tiles of 64 keys double-buffered through cp.async, the next tile's
//     load in flight under this tile's math;
//   - the score in log2 units: the caller folds log2(e)/sqrt(72) into q
//     before q's one f16 round (the split kernel's qscale), so softmax is
//     ex2(s - m) with no multiply; only the last key tile is masked.
// q, k, v: rows `in_stride` halves apart, head h's 72 dims at h*72 - either
// three [rows][heads][72] planes, or straight off the fused qkv GEMM's
// [rows][3][heads][72] landing (in_stride 3 * heads * 72, k and v at +d and
// +2d). With `bq` the q fragment is f16((q + bq) * qscale), the split
// kernel's arithmetic, done once at the fragment load; with `bv` the value
// bias is added in the epilogue - P sums to 1 after normalization, so
// P (V + 1 bv^T) / l = P V / l + bv; the key bias needs nothing at all
// (one constant per query row, which softmax ignores). That is the whole
// split pass gone. out: [rows][heads][72] f16 for the wo GEMM. Plain
// pointers for q/k/v: a predecessor kernel writes them, so no read-only
// (__restrict__ / ldg) loads of them under PDL.
#define PD_NPA_QW 8u
#define PD_NPA_KT 64u
#define PD_NPA_HD 72u
#define PD_NPA_RS 88u   // smem row stride (halves): 176 B = 11 16-byte units, odd, so the 8 rows an ldmatrix reads sit on 8 distinct bank groups (80 would put rows 0 and 4 together)

__device__ __forceinline__ void pd_np_mma8(float d[4], uint32_t a0, uint32_t a1, uint32_t b0) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(b0));
}

__device__ __forceinline__ float pd_np_ex2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ void pd_np_cp16(void* dst, const void* src, bool ok) {
    const uint32_t d = (uint32_t)__cvta_generic_to_shared(dst);
    const int n = ok ? 16 : 0;   // src-size 0 = zero-fill, no global read
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(d), "l"(src), "r"(n));
}

__global__ void __launch_bounds__(32u * PD_NPA_QW, 2) pd_np_vit_attn_kernel(
    const __half* q, const __half* k, const __half* v, __half* __restrict__ out,
    uint32_t n, uint32_t n_heads, uint32_t in_stride, const float* __restrict__ bq,
    const float* __restrict__ bv, float qscale) {
#if PD_FA_OK
    constexpr uint32_t KT = PD_NPA_KT, HD = PD_NPA_HD, RS = PD_NPA_RS, NT = 32u * PD_NPA_QW;
    __shared__ __align__(16) __half sh[2][2][KT * RS];   // [stage][K|V][key*RS + dim]
    const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
    const uint32_t g8 = lane >> 2, t4 = lane & 3u, lg = lane >> 3;
    const uint32_t h = blockIdx.y;
    const uint32_t row0 = blockIdx.x * (PD_NPA_QW * 16u) + warp * 16u;
    const size_t rstride = in_stride, ostride = (size_t)n_heads * HD;

    // stage one K|V tile (64 keys x 72 dims each) into `st`
    auto stage = [&](uint32_t st, uint32_t t0) {
        constexpr uint32_t CH = HD / 8u;                // 9 16-byte chunks a row
        for (uint32_t u = tid; u < 2u * KT * CH; u += NT) {
            const uint32_t isv = u >= KT * CH;
            const uint32_t ur = isv ? u - KT * CH : u;
            const uint32_t kk = ur / CH, c = ur - kk * CH;
            const uint32_t key = t0 + kk;
            const bool ok = key < n;
            const __half* src = (isv ? v : k) + (size_t)(ok ? key : 0u) * rstride + h * HD + c * 8u;
            pd_np_cp16(&sh[st][isv][kk * RS + c * 8u], src, ok);
        }
        asm volatile("cp.async.commit_group;");
    };

    // Q fragments: 4 k16 steps (dims 0..63) + one k8 step (64..71)
    uint32_t qa[4][4], q8[2];
    {
        const uint32_t ra = row0 + g8, rb = ra + 8u;
        const bool oka = ra < n, okb = rb < n;
        const __half* pa = q + (size_t)(oka ? ra : 0u) * rstride + h * HD;
        const __half* pb = q + (size_t)(okb ? rb : 0u) * rstride + h * HD;
        auto ld = [&](const __half* p, bool ok, uint32_t c) -> uint32_t {
            if (!ok) return 0u;
            __half2 x = *reinterpret_cast<const __half2*>(p + c);
            if (bq) {
                const float2 f = __half22float2(x);
                x = __floats2half2_rn((f.x + bq[h * HD + c]) * qscale,
                                      (f.y + bq[h * HD + c + 1u]) * qscale);
            }
            return *reinterpret_cast<const uint32_t*>(&x);
        };
#pragma unroll
        for (uint32_t d0 = 0; d0 < 4u; ++d0) {
            qa[d0][0] = ld(pa, oka, d0 * 16u + 2u * t4);
            qa[d0][1] = ld(pb, okb, d0 * 16u + 2u * t4);
            qa[d0][2] = ld(pa, oka, d0 * 16u + 8u + 2u * t4);
            qa[d0][3] = ld(pb, okb, d0 * 16u + 8u + 2u * t4);
        }
        q8[0] = ld(pa, oka, 64u + 2u * t4);
        q8[1] = ld(pb, okb, 64u + 2u * t4);
    }

    float m_st[2] = {-1e30f, -1e30f}, l_st[2] = {0.f, 0.f};
    float o[9][4];
#pragma unroll
    for (uint32_t i = 0; i < 9u; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;

    const uint32_t n_tiles = (n + KT - 1u) / KT;
    stage(0u, 0u);
    for (uint32_t t = 0; t < n_tiles; ++t) {
        const uint32_t st = t & 1u, t0 = t * KT;
        if (t + 1u < n_tiles) {
            stage(st ^ 1u, t0 + KT);
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();
        const __half* sk = sh[st][0];
        const __half* sv = sh[st][1];

        // S = Q K^T over 64 keys: 8 n8 tiles
        float s[8][4];
#pragma unroll
        for (uint32_t i = 0; i < 8u; ++i) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.f;
#pragma unroll
        for (uint32_t np = 0; np < 4u; ++np) {
#pragma unroll
            for (uint32_t d0 = 0; d0 < 4u; ++d0) {
                const __half* kp = sk + (np * 16u + (lg >> 1) * 8u + (lane & 7u)) * RS
                                   + d0 * 16u + (lg & 1u) * 8u;
                uint32_t b[4];
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3])
                             : "r"((uint32_t)__cvta_generic_to_shared(kp)));
                pd_fa_mma16(s[np * 2u], qa[d0][0], qa[d0][1], qa[d0][2], qa[d0][3], b[0], b[1]);
                pd_fa_mma16(s[np * 2u + 1u], qa[d0][0], qa[d0][1], qa[d0][2], qa[d0][3], b[2], b[3]);
            }
            const __half* kp8 = sk + (np * 16u + ((lane >> 3) & 1u) * 8u + (lane & 7u)) * RS + 64u;
            uint32_t b0, b1;
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                         : "=r"(b0), "=r"(b1)
                         : "r"((uint32_t)__cvta_generic_to_shared(kp8)));
            pd_np_mma8(s[np * 2u], q8[0], q8[1], b0);
            pd_np_mma8(s[np * 2u + 1u], q8[0], q8[1], b1);
        }
        // the last tile's keys past n are zero-filled - mask them
        if (t0 + KT > n) {
#pragma unroll
            for (uint32_t i = 0; i < 8u; ++i) {
                const uint32_t kb = t0 + i * 8u + 2u * t4;
#pragma unroll
                for (uint32_t e = 0; e < 4u; ++e)
                    if (kb + (e & 1u) >= n) s[i][e] = -1e30f;
            }
        }
        // online softmax in log2 units (rows g8 and g8 + 8)
        float mn[2] = {m_st[0], m_st[1]};
#pragma unroll
        for (uint32_t i = 0; i < 8u; ++i) {
            mn[0] = fmaxf(mn[0], fmaxf(s[i][0], s[i][1]));
            mn[1] = fmaxf(mn[1], fmaxf(s[i][2], s[i][3]));
        }
#pragma unroll
        for (uint32_t x = 1; x <= 2u; x <<= 1) {
            mn[0] = fmaxf(mn[0], __shfl_xor_sync(0xffffffffu, mn[0], x));
            mn[1] = fmaxf(mn[1], __shfl_xor_sync(0xffffffffu, mn[1], x));
        }
        float ws[2] = {0.f, 0.f};
#pragma unroll
        for (uint32_t i = 0; i < 8u; ++i) {
#pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) {
                const float p = pd_np_ex2(s[i][e] - mn[e >> 1]);
                s[i][e] = p;
                ws[e >> 1] += p;
            }
        }
#pragma unroll
        for (uint32_t x = 1; x <= 2u; x <<= 1) {
            ws[0] += __shfl_xor_sync(0xffffffffu, ws[0], x);
            ws[1] += __shfl_xor_sync(0xffffffffu, ws[1], x);
        }
        float corr[2];
#pragma unroll
        for (uint32_t r = 0; r < 2u; ++r) {
            corr[r] = pd_np_ex2(m_st[r] - mn[r]);
            l_st[r] = l_st[r] * corr[r] + ws[r];
            m_st[r] = mn[r];
        }
#pragma unroll
        for (uint32_t i = 0; i < 9u; ++i) {
            o[i][0] *= corr[0];
            o[i][1] *= corr[0];
            o[i][2] *= corr[1];
            o[i][3] *= corr[1];
        }
        // O += P V: 4 k16 steps over keys, 9 n8 tiles over dims
#pragma unroll
        for (uint32_t kf = 0; kf < 4u; ++kf) {
            const uint32_t c0 = 2u * kf, c1 = c0 + 1u;
            const __half2 a0 = __floats2half2_rn(s[c0][0], s[c0][1]);
            const __half2 a1 = __floats2half2_rn(s[c0][2], s[c0][3]);
            const __half2 a2 = __floats2half2_rn(s[c1][0], s[c1][1]);
            const __half2 a3 = __floats2half2_rn(s[c1][2], s[c1][3]);
            const uint32_t pa0 = *reinterpret_cast<const uint32_t*>(&a0);
            const uint32_t pa1 = *reinterpret_cast<const uint32_t*>(&a1);
            const uint32_t pa2 = *reinterpret_cast<const uint32_t*>(&a2);
            const uint32_t pa3 = *reinterpret_cast<const uint32_t*>(&a3);
            const __half* vp = sv + (kf * 16u + (lg & 1u) * 8u + (lane & 7u)) * RS + (lg >> 1) * 8u;
#pragma unroll
            for (uint32_t nt = 0; nt < 8u; nt += 2u) {
                uint32_t b[4];
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3])
                             : "r"((uint32_t)__cvta_generic_to_shared(vp + nt * 8u)));
                pd_fa_mma16(o[nt], pa0, pa1, pa2, pa3, b[0], b[1]);
                pd_fa_mma16(o[nt + 1u], pa0, pa1, pa2, pa3, b[2], b[3]);
            }
            const __half* vp8 = sv + (kf * 16u + ((lane >> 3) & 1u) * 8u + (lane & 7u)) * RS + 64u;
            uint32_t b0, b1;
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                         : "=r"(b0), "=r"(b1)
                         : "r"((uint32_t)__cvta_generic_to_shared(vp8)));
            pd_fa_mma16(o[8], pa0, pa1, pa2, pa3, b0, b1);
        }
        __syncthreads();   // this stage is read; the next iteration's load may overwrite it
    }

    const float nrm[2] = {l_st[0] > 0.f ? 1.f / l_st[0] : 0.f, l_st[1] > 0.f ? 1.f / l_st[1] : 0.f};
#pragma unroll
    for (uint32_t i = 0; i < 9u; ++i) {
        const uint32_t dcol = i * 8u + 2u * t4;
#pragma unroll
        for (uint32_t r = 0; r < 2u; ++r) {
            const uint32_t row = row0 + g8 + r * 8u;
            if (row >= n) continue;
            __half* op = out + (size_t)row * ostride + h * HD + dcol;
            const float b0 = bv ? bv[h * HD + dcol] : 0.f, b1 = bv ? bv[h * HD + dcol + 1u] : 0.f;
            *reinterpret_cast<__half2*>(op) =
                __floats2half2_rn(o[i][2u * r] * nrm[r] + b0, o[i][2u * r + 1u] * nrm[r] + b1);
        }
    }
#else
    (void)q; (void)k; (void)v; (void)out; (void)n; (void)n_heads; (void)in_stride; (void)bq;
    (void)bv; (void)qscale;
#endif
}

PD_EXPORT
int pd_np_vit_attn(const void* q, const void* k, const void* v, void* out, uint32_t n,
                   uint32_t n_heads, uint32_t in_stride, const void* bq, const void* bv,
                   float qscale, void* stream) {
    if (n == 0 || n_heads == 0) return 0;
    if (in_stride < n_heads * PD_NPA_HD || (in_stride % 8u) != 0u) return cudaErrorInvalidValue;
    int dev = 0, cc = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&cc, cudaDevAttrComputeCapabilityMajor, dev);
    if (cc < 8) return cudaErrorInvalidValue;
    dim3 grid((n + PD_NPA_QW * 16u - 1u) / (PD_NPA_QW * 16u), n_heads, 1);
    pd_np_vit_attn_kernel<<<grid, 32u * PD_NPA_QW, 0, (cudaStream_t)stream>>>(
        (const __half*)q, (const __half*)k, (const __half*)v, (__half*)out, n, n_heads,
        in_stride, (const float*)bq, (const float*)bv, qscale);
    return pd_launch_status();
}
