// Greedy pick with its confidence readouts (slot 343): the runner-up, the
// chosen token's log-probability, one probed token's probability and a
// tail-suppressed entropy, in one pass over each logits row. Whisper's
// transcripts read it at every decode step. It sits beside stage2_sample's
// `pd_argmax_rows`, whose tie rule it keeps bit for bit.

// Same greedy pick, plus everything a transcript needs to say how sure the
// model was - the RUNNER-UP included, which is what turns "confidence" from a
// number into something a reader can act on.
//
// Four readouts, all from one pass over the row:
//   log p(top1)  how much mass the chosen token took
//   log p(top2)  the road not taken. `p1 - p2` is the MARGIN, and margin is
//                the signal that separates "torn between two words" (where ASR
//                errors actually live) from "merely diffuse". The runner-up's
//                ID rides out alongside so the UI can name the alternative.
//   p(probe)     one nominated token's probability - whisper's `<|nospeech|>`,
//                which OpenAI defines at the first decode step. Reading it
//                host-side would mean copying back a 51866-float row per
//                window just to index one element of it.
//   H2           Renyi-2 (collision) entropy, -log sum(p^2), in nats.
//
// Why H2 and not Shannon: over a 51866-token vocabulary the Shannon sum is
// dominated by the TAIL - tens of thousands of near-zero terms - so it ranks a
// decisive row as more uncertain than a genuine two-way tie, which is
// backwards for the one job it has. The ASR confidence literature's answer is a
// tail-suppressed entropy (Tsallis/Renyi with alpha > 1; Laptev & Ginsburg,
// "Fast Entropy-Based Methods of Word-Level Confidence Estimation for E2E ASR",
// SLT 2022 - technique studied, implementation ours). alpha = 2 specifically,
// because sum(p^2) is the one alpha that falls out of an online softmax for
// free: square the same exponential the sum already computed. Any other alpha
// costs a second pass over the row plus a powf per element, which this kernel
// sits inside a graph-captured decode tick to avoid.
//
// This is `pd_argmax_rows` walking an online softmax and a top-2 alongside the
// max, so it stays one read of [rows, vocab] rather than three.
//
// Tie rule is the argmax kernel's exactly (lowest index wins) at both ranks, so
// the token this returns is bit-identical to what `pd_argmax_rows` would have
// picked - asking for confidence must never move the transcript.
//
// `mask` (<= PD_A2_MASK_MAX ids) is the reference's `SuppressTokens`: those ids
// are left out of the pick, the runner-up and the normaliser, exactly as
// setting them to -inf before the log-softmax would - whisper's decode must
// never emit `<|nospeech|>` or a prompt token as transcript (openai/whisper
// `decoding.py`, whisper.cpp `whisper_process_logits`; technique theirs, code
// ours). The PROBE alone reads the UNSUPPRESSED distribution: `no_speech_prob`
// is defined on the model's own logits before any suppression, and masking the
// probed token itself would otherwise read it as 0.
#define PD_A2_MASK_MAX 8u
#define PD_A2_BETTER(AM, AI, BM, BI) ((AM) > (BM) || ((AM) == (BM) && (AI) < (BI)))
// "no runner-up yet". Loses every comparison on value alone, so it also works
// as the initial champion for a thread that owns no elements at all.
#define PD_A2_NONE 0xffffffffu

__global__ void pd_argmax_top2_rows_kernel(const float* __restrict__ logits,
                                           unsigned int* __restrict__ out,
                                           unsigned int* __restrict__ alt,
                                           float* __restrict__ stats,
                                           uint32_t probe,
                                           const uint32_t* __restrict__ mask,
                                           uint32_t n_mask, uint32_t n) {
    const uint32_t row = blockIdx.x, tid = threadIdx.x, nth = blockDim.x;
    const float* x = logits + (size_t)row * n;
    __shared__ uint32_t smask[PD_A2_MASK_MAX];
    if (tid < n_mask) smask[tid] = mask[tid];
    __syncthreads();
    // -FLT_MAX rather than -inf: the merges below take differences of two
    // running maxima, and (-inf) - (-inf) is a nan that would poison the sum.
    float m1 = -3.402823466e+38f, m2 = -3.402823466e+38f;
    float s = 0.f, s2 = 0.f;
    uint32_t i1 = PD_A2_NONE, i2 = PD_A2_NONE;
    for (uint32_t i = tid; i < n; i += nth) {
        bool suppressed = false;
        for (uint32_t k = 0; k < n_mask; ++k) suppressed |= (i == smask[k]);
        if (suppressed) continue;
        const float v = x[i];
        if (v > m1) {
            // the outgoing champion is by construction the new runner-up: it
            // beat everything this thread has seen except the value replacing
            // it. Strict `>` on an ascending walk keeps the LOWEST index at
            // both ranks without a tie test.
            const float d = __expf(m1 - v);
            s = s * d + 1.f;
            s2 = s2 * d * d + 1.f;
            m2 = m1;
            i2 = i1;
            m1 = v;
            i1 = i;
        } else {
            const float e = __expf(v - m1);
            s += e;
            s2 += e * e;
            if (v > m2) {
                m2 = v;
                i2 = i;
            }
        }
    }
    // Merge two (top-2, sum, sum-of-squares) states: rescale the loser's sums
    // onto the winner's max. Each side is already sorted internally, so the
    // merged runner-up is one comparison, never a re-sort of four candidates.
    // Fixed lane order, so the result is run-to-run stable.
#define PD_A2_MERGE(OM1, OI1, OM2, OI2, OS, OS2)                              \
    do {                                                                      \
        const float om1_ = (OM1), om2_ = (OM2), os_ = (OS), os2_ = (OS2);     \
        const uint32_t oi1_ = (OI1), oi2_ = (OI2);                            \
        if (PD_A2_BETTER(om1_, oi1_, m1, i1)) {                               \
            /* their champion takes the row; ours drops into the runner-up   \
               race against theirs (our own m2 lost to our m1 already) */     \
            if (PD_A2_BETTER(m1, i1, om2_, oi2_)) {                           \
                m2 = m1;                                                      \
                i2 = i1;                                                      \
            } else {                                                          \
                m2 = om2_;                                                    \
                i2 = oi2_;                                                    \
            }                                                                 \
            const float d_ = __expf(m1 - om1_);                               \
            s = s * d_ + os_;                                                 \
            s2 = s2 * d_ * d_ + os2_;                                         \
            m1 = om1_;                                                        \
            i1 = oi1_;                                                        \
        } else {                                                              \
            /* ours holds; only their champion can outrank our runner-up,    \
               since their m2 sits below their m1 */                          \
            if (PD_A2_BETTER(om1_, oi1_, m2, i2)) {                           \
                m2 = om1_;                                                    \
                i2 = oi1_;                                                    \
            }                                                                 \
            const float d_ = __expf(om1_ - m1);                               \
            s += os_ * d_;                                                    \
            s2 += os2_ * d_ * d_;                                             \
        }                                                                     \
    } while (0)
#pragma unroll
    for (uint32_t off = 16; off > 0; off >>= 1) {
        PD_A2_MERGE(__shfl_down_sync(0xffffffffu, m1, off),
                    __shfl_down_sync(0xffffffffu, i1, off),
                    __shfl_down_sync(0xffffffffu, m2, off),
                    __shfl_down_sync(0xffffffffu, i2, off),
                    __shfl_down_sync(0xffffffffu, s, off),
                    __shfl_down_sync(0xffffffffu, s2, off));
    }
    __shared__ float sm1[8], sm2[8], ss[8], ss2[8];
    __shared__ uint32_t si1[8], si2[8];
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    if (lane == 0) {
        sm1[warp] = m1;
        sm2[warp] = m2;
        ss[warp] = s;
        ss2[warp] = s2;
        si1[warp] = i1;
        si2[warp] = i2;
    }
    __syncthreads();
    if (tid != 0) return;
    for (uint32_t w = 1; w < ((nth + 31u) >> 5); ++w)
        PD_A2_MERGE(sm1[w], si1[w], sm2[w], si2[w], ss[w], ss2[w]);
#undef PD_A2_MERGE
    // A row whose every logit is exactly -FLT_MAX never promotes anything, so
    // index 0 stands in - which is what a host argmax over an all-equal row
    // returns anyway. Never let the sentinel out as a token id.
    out[row] = (i1 == PD_A2_NONE) ? 0u : i1;
    // n means "there wasn't one" - the same out-of-range convention `probe`
    // uses, so a caller has one rule to remember. Only a 1-token row can hit it.
    if (alt) alt[row] = (i2 == PD_A2_NONE) ? n : i2;
    if (!stats) return;
    // s is already sum(exp(x - m1)), so log p(v) = (v - m1) - log s.
    // logf, not __logf: it runs once per row and the values are user-visible.
    const float lse = (s > 0.f) ? logf(s) : 0.f;
    float* o = stats + (size_t)row * 4u;
    o[0] = (s > 0.f) ? -lse : 0.f;
    // the probe's normaliser puts the suppressed ids back: an online-softmax
    // merge of at most PD_A2_MASK_MAX terms onto (m1, s), max-first so a
    // suppressed token that outranks every allowed one cannot overflow
    float mf = m1, sf = s;
    for (uint32_t k = 0; k < n_mask; ++k) {
        const uint32_t id = smask[k];
        if (id >= n) continue;
        const float v = x[id];
        if (v > mf) {
            sf = sf * __expf(mf - v) + 1.f;
            mf = v;
        } else {
            sf += __expf(v - mf);
        }
    }
    o[1] = (probe < n && sf > 0.f) ? (__expf(x[probe] - mf) / sf) : 0.f;
    o[2] = (i2 != PD_A2_NONE && s > 0.f) ? (m2 - m1 - lse) : 0.f;
    // sum(p^2) = s2 / s^2, so H2 = -log(s2/s^2) = 2 log s - log s2. s2 >= 1
    // always (the max's own term is exactly 1), so the log is safe and H2
    // lands in [0, log n]: 0 for a one-token row, log n for a uniform one.
    o[3] = (s2 > 0.f) ? (2.f * lse - logf(s2)) : 0.f;
}
#undef PD_A2_BETTER
#undef PD_A2_NONE

// 343: greedy pick + the runner-up + the row's confidence readouts, one pass.
//   logits f32 [rows, n]
//   out    u32 [rows]      argmax (same tie rule as pd_argmax_rows)
//   alt    u32 [rows]      the runner-up's id, or n for "none"; NULL to skip
//   stats  f32 [rows, 4]   {log p(top1), p(probe), log p(top2), H2}; NULL to skip
//   probe  token id to report, or >= n for "none" (writes 0)
//   mask   u32 [n_mask]    ids never picked (suppressed before the softmax);
//                          NULL with n_mask 0 for none, at most PD_A2_MASK_MAX
PD_EXPORT
int pd_argmax_top2_rows(const void* logits, void* out, void* alt, void* stats,
                        uint32_t probe, const void* mask, uint32_t n_mask, uint32_t rows,
                        uint32_t n, void* stream) {
    if (rows == 0 || n == 0) return 0;
    if (n_mask > PD_A2_MASK_MAX || (n_mask != 0 && mask == NULL)) return cudaErrorInvalidValue;
    pd_argmax_top2_rows_kernel<<<rows, 256, 0, (cudaStream_t)stream>>>(
        (const float*)logits, (unsigned int*)out, (unsigned int*)alt, (float*)stats,
        probe, (const uint32_t*)mask, n_mask, n);
    return pd_launch_status();
}
#undef PD_A2_MASK_MAX
