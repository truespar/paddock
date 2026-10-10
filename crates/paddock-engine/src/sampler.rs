//! Token sampling. Deterministic given a seed (OpenAI `seed` param): same
//! seed + same logits => same token, which is also what the parity harness
//! needs. Greedy is the special case temperature == 0.
//!
//! Filter order matches the de-facto pipeline (temperature -> top-k -> top-p ->
//! min-p -> sample); repetition penalty is applied to logits before that.
//!
//! Performance contract: sampling runs on the host once per slot per decode
//! tick, so it must stay far under the batched GPU step (~15 ms at B=32).
//! No path may sort the full vocab (201k logits sort ≈ 6 ms - it was the
//! serving bottleneck at every concurrency). The no-truncation path draws
//! sort-free in two O(V) passes; truncation paths select_nth an O(V) head
//! and sort only that. Every stochastic path consumes exactly one uniform
//! per token, so a seed's ChaCha stream stays aligned regardless of which
//! path a step takes. (Same distribution as before; the seed->token mapping
//! differs from pre-rewrite builds, which was never a stable contract.)

use rand_chacha::ChaCha8Rng;
use rand_core::SeedableRng;

#[derive(Debug, Clone)]
pub struct SamplingParams {
    /// 0.0 = greedy/argmax; >0 scales logits before softmax.
    pub temperature: f32,
    /// keep only the top-k logits (0 = disabled).
    pub top_k: usize,
    /// nucleus: keep the smallest set whose cumulative prob ≥ top_p (1.0 = off).
    pub top_p: f32,
    /// min-p: drop tokens below min_p × p(top) (0 = off).
    pub min_p: f32,
    /// divide the logit of already-seen tokens (1.0 = off).
    pub repeat_penalty: f32,
    /// how many trailing tokens the penalties consider.
    pub repeat_last_n: usize,
    /// OpenAI presence penalty: subtract once per token that appeared in the
    /// window (0 = off).
    pub presence_penalty: f32,
    /// OpenAI frequency penalty: subtract per occurrence in the window (0 = off).
    pub frequency_penalty: f32,
    pub seed: u64,
    /// OpenAI logit_bias: (token id, bias in -100..100) added to the logits
    /// before penalties and sampling (empty = off). Sequences with a bias
    /// cannot ride the speculative path (device argmaxes never see it).
    pub logit_bias: Vec<(u32, f32)>,
    /// Sliding-window no-repeat n-gram guard, `(n, window, stop)`; `(0, 0, _)`
    /// = off. The DeepSeek-OCR family's required repetition control - same
    /// math as the reference's `SlidingWindowNoRepeatNgramProcessor` (and
    /// SGLang's `DeepseekOCRNoRepeatNGramLogitProcessor`): ban every token
    /// that would complete an `n`-gram whose (n-1)-token prefix equals the
    /// current tail, searching occurrences that START within the trailing
    /// `window` tokens of prompt+output. History-dependent, so an armed ban
    /// keeps the sequence off the device/spec paths, like the penalties above.
    ///
    /// `stop` turns the ban into a STOP: the logits are never touched (the
    /// sampler stays whatever it is, greedy device rows included) and the
    /// sequence ends, before the token, where the token it picked is one the
    /// ban would have masked - `repeat_stop_hit`. A degenerate loop is cut
    /// at its first repeated n-gram instead of steered into variations.
    pub no_repeat_ngram: (usize, usize, bool),
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            seed: 0,
            logit_bias: Vec::new(),
            no_repeat_ngram: (0, 0, false),
        }
    }
}

/// Incremental output constraint (JSON schema / tool-call grammar): the
/// engine asks which tokens are legal, commits the accepted one, and learns
/// when the constrained region may end. Implementations live server-side
/// (they need the tokenizer); the engine only drives the seam.
pub trait TokenConstraint: Send {
    /// is `id` a legal next token in the current state?
    fn allows(&self, id: u32) -> bool;
    /// commit a token previously reported legal (never a stop token).
    fn accept(&mut self, id: u32);
    /// the constrained output may end now - stop tokens become legal.
    fn may_stop(&self) -> bool;
    /// The constraint is in a free phase right now: `allows` returns true for
    /// every token (reasoning before the gate arms, prose outside a tool
    /// call), so unconstrained and constrained sampling are the same
    /// distribution and speculative rounds are exact here - provided each
    /// committed token is `accept`ed and the run is CUT at the first token
    /// after which this turns false (the machine armed; later drafts were
    /// never checked).
    ///
    /// How this relates to the field: vLLM V1
    /// composes spec+grammar with a bitmask per DRAFT POSITION (PR #14702)
    /// and recently had to special-case trimming grammar advance at the
    /// reasoning boundary (PR #44297) - the boundary our gate handles by
    /// construction; SGLang/xgrammar advance the matcher over drafts and
    /// roll back on rejection; llama.cpp runs the grammar in both the draft
    /// and target sampler chains. All three keep speculating inside the
    /// constrained region via masks or matcher rollback. Ours composes
    /// there too, by a fourth mechanism: the host verify walk samples each
    /// pick through the machine (pick_next's exact semantics at the verify
    /// position), so active regions are grammar-legal by construction with
    /// drafts as accept-while-match acceleration - no bitmask, no rollback,
    /// because only committed picks ever advance the machine. `free_now`
    /// remains the cheap probe for rounds that cannot host a machine (the
    /// device/strip lanes, where acceptance resolves on device). Default
    /// false keeps unknown implementors on the dense path.
    fn free_now(&self) -> bool {
        false
    }
}

/// A per-token sampling plan the GPU can execute on the logits row in place -
/// no host readback. Only paths whose logits need no host mutation qualify:
/// active penalties, logit_bias, truncation filters (top-k/top-p/min-p),
/// constraints and logprobs all stay on the host pipeline.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DevicePlan {
    /// pure argmax; the device tie-break (lowest index) matches the host scan
    Greedy,
    /// temperature-only categorical (the OpenAI-default hot path): softmax
    /// over logits × inv_t, pick the u-quantile walking the vocab in index
    /// order. The device sums exp mass in a different ORDER than the host
    /// (chunked vs serial), so the same u can pick a boundary-adjacent token -
    /// identical distribution; the seed->token mapping is not a contract.
    Categorical { inv_t: f32, u: f32 },
    /// P65 host-head truncation plan: the device runs only the top-K
    /// SELECTION (pd_topk_rows, K = 64 superset head); the backend then
    /// host-samples the compact head with [`sample_trunc_head`] - exact
    /// build_nucleus semantics - and writes the token into `step.ids` like
    /// any device row. Emitted only by `device_plan_trunc`/`peek_..._trunc`
    /// at service sites whose finish path implements the head readback;
    /// `is_device_plannable` stays false for truncation rows deliberately -
    /// the zero-host decode pipes cannot feed a host-sampled id forward.
    TruncCat {
        inv_t: f32,
        u: f32,
        k: u32,
        top_p: f32,
        min_p: f32,
    },
    /// Canonical rejection-sampling verify row (PADDOCK_SPEC_RS):
    /// the row's draft was SAMPLED from the drafter softmax; the device
    /// accepts it with probability min(1, p/q) (u1) and recovers from the
    /// residual max(p-q, 0) on reject (u2). Same emitted distribution as
    /// `Categorical` + accept-while-match - higher expected acceptance when
    /// p and q are broad and close. Rows carrying this plan are SKIPPED by
    /// the row sampler (mode 0); the RS resolve kernel writes them.
    RsVerify { inv_t: f32, u1: f32, u2: f32 },
    /// Truncation-aware canonical rejection-sampling verify row (rung G,
    /// the same accept-w.p.-min(1, p/q) / residual recovery as
    /// `RsVerify`, but p is the mode-5 NUCLEUS (top-k `k`, `top_p`, `min_p`
    /// - exactly the distribution the dense sampler draws from, so the
    ///   scheme stays lossless against our own non-spec path) and q is the
    ///   drafter's K-candidate distribution the backend recorded when it
    ///   sampled the draft. Emitted only for backends that answer
    ///   `supports_spec_rs_trunc`; consumed by the DFlash2 resolve (slot 471).
    RsTrunc {
        inv_t: f32,
        u1: f32,
        u2: f32,
        k: u32,
        top_p: f32,
        min_p: f32,
    },
}

pub struct Sampler {
    params: SamplingParams,
    rng: ChaCha8Rng,
    // reused candidate buffer for the truncation paths - building a fresh
    // vocab-sized Vec per token is ~1.6 MB of alloc churn per slot per tick
    scratch: Vec<(u32, f32)>,
}

impl Sampler {
    pub fn new(params: SamplingParams) -> Self {
        let rng = ChaCha8Rng::seed_from_u64(params.seed);
        Self {
            params,
            rng,
            scratch: Vec::new(),
        }
    }

    /// True when sampling degenerates to pure argmax over RAW logits - no
    /// temperature, no history-dependent penalty. Only such sequences can
    /// ride the speculative batch: its per-row picks are device argmaxes,
    /// computed before any host-side logit mutation could apply.
    pub fn is_pure_greedy(&self) -> bool {
        let p = &self.params;
        p.temperature <= 0.0
            && p.logit_bias.is_empty()
            && !self.ngram_armed()
            && (p.repeat_last_n == 0
                || (p.repeat_penalty == 1.0
                    && p.presence_penalty == 0.0
                    && p.frequency_penalty == 0.0))
    }

    /// The no-repeat-ngram BAN is on (both halves non-zero, matching the
    /// reference's `no_repeat_ngram_size > 0 and ngram_window > 0` gate). The
    /// stop mode masks nothing, so it never arms this.
    fn ngram_armed(&self) -> bool {
        let (n, w, stop) = self.params.no_repeat_ngram;
        n > 0 && w > 0 && !stop
    }

    /// The guard in stop mode, asked of the token about to be committed: would
    /// `next` complete an `n`-gram already started within the window? Then
    /// the sequence ends here, without it - the output up to this token is
    /// the unguarded output, byte for byte.
    pub fn repeat_stop_hit(&self, history: &[u32], next: u32) -> bool {
        let (n, w, stop) = self.params.no_repeat_ngram;
        let mut hit = false;
        if stop {
            for_each_ngram_ban((n, w), history, |tok| hit |= tok == next);
        }
        hit
    }

    /// True when sampling is ROW-LOCAL - each token's distribution depends
    /// only on its own logits (temperature/top-k/top-p/min-p fine), not on
    /// generated history (penalties) or logit_bias. Such sequences can ride
    /// the SAMPLED speculative round: with deterministic (greedy-MTP) drafts,
    /// "sample each verify row with this sampler, accept while the sample
    /// equals the draft, first mismatch's sample is the replacement" is exact
    /// Leviathan rejection sampling - the emitted distribution is identical
    /// to the dense path's. (Penalties would need round-internal history
    /// updates; bias is cheap but host-mutates logits the greedy device path
    /// never sees - both stay dense.)
    pub fn is_spec_safe(&self) -> bool {
        let p = &self.params;
        p.logit_bias.is_empty()
            && !self.ngram_armed()
            && (p.repeat_last_n == 0
                || (p.repeat_penalty == 1.0
                    && p.presence_penalty == 0.0
                    && p.frequency_penalty == 0.0))
    }

    /// Non-consuming twin of `device_plan`'s eligibility: every draw this
    /// sampler makes can execute on device (greedy, or temperature-only
    /// categorical). Spec rounds check this before drawing per-row plans -
    /// `device_plan` itself consumes a uniform per Categorical plan.
    pub fn is_device_plannable(&self) -> bool {
        let p = &self.params;
        self.is_spec_safe()
            && (p.temperature <= 0.0 || (p.top_k == 0 && p.top_p >= 1.0 && p.min_p <= 0.0))
    }

    /// The device-executable plan for this token, or `None` when the token
    /// needs the host pipeline. Drawing a `Categorical` plan consumes the
    /// token's one uniform from the seed stream, exactly like the host draw
    /// would - call it once per token, and only when the plan will be used.
    pub fn device_plan(&mut self) -> Option<DevicePlan> {
        if !self.params.logit_bias.is_empty() || self.ngram_armed() {
            return None;
        }
        let p = &self.params;
        // penalties mutate logits from history - host only
        if p.repeat_last_n > 0
            && (p.repeat_penalty != 1.0 || p.presence_penalty != 0.0 || p.frequency_penalty != 0.0)
        {
            return None;
        }
        if p.temperature <= 0.0 {
            return Some(DevicePlan::Greedy);
        }
        if p.top_k == 0 && p.top_p >= 1.0 && p.min_p <= 0.0 {
            let u = self.next_uniform();
            return Some(DevicePlan::Categorical {
                inv_t: 1.0 / self.params.temperature,
                u,
            });
        }
        None
    }

    /// `device_plan` extended with the P65 host-head truncation plan: same
    /// preconditions, but rows whose only host dependency is a truncation
    /// filter with top_k in 1..=64 get `TruncCat` instead of `None`. Only
    /// call from service sites whose backend + finish path implement the
    /// head readback (`supports_host_head`) - see the TruncCat doc.
    pub fn device_plan_trunc(&mut self) -> Option<DevicePlan> {
        if let Some(p) = self.device_plan() {
            return Some(p);
        }
        let p = &self.params;
        if !p.logit_bias.is_empty()
            || self.ngram_armed()
            || (p.repeat_last_n > 0
                && (p.repeat_penalty != 1.0
                    || p.presence_penalty != 0.0
                    || p.frequency_penalty != 0.0))
        {
            return None;
        }
        // The whole coverable truncation space rides a
        // device plan - top_k 1..=64 executes as mode 5 (64-head kernel),
        // top_k == 0 with top_p/min_p as mode 6 (histogram quantile walk;
        // the untruncated temp>0 case already returned Categorical above).
        // top_k in 65..vocab stays host: no elected profile uses it (the
        // OpenAI API does not even expose top_k) and the head-partial
        // machinery it needs is not worth building unmeasured.
        if p.temperature > 0.0 && p.top_k <= 64 {
            let (k, top_p, min_p) = (p.top_k as u32, p.top_p, p.min_p);
            let inv_t = 1.0 / p.temperature;
            let u = self.next_uniform();
            return Some(DevicePlan::TruncCat {
                inv_t,
                u,
                k,
                top_p,
                min_p,
            });
        }
        None
    }

    /// True when every draw is either classically device-plannable OR a
    /// truncation plan a full-device backend can execute (mode 5 for
    /// top_k 1..=64, mode 6 for the k-less top-p/min-p space) - the
    /// pipe/overlap admission twin of `is_device_plannable`
    /// for backends that pass `supports_device_trunc`. Mirrors the
    /// `device_plan_trunc` eligibility exactly (twin-consistency test).
    pub fn is_trunc_plannable(&self) -> bool {
        let p = &self.params;
        self.is_spec_safe() && (p.temperature <= 0.0 || p.top_k <= 64)
    }

    /// Non-consuming twin of [`Self::device_plan_trunc`] (see
    /// [`Self::peek_device_plan`] for the peek/commit contract).
    pub fn peek_device_plan_trunc(&self) -> Option<DevicePlan> {
        let mut tmp = Self {
            params: self.params.clone(),
            rng: self.rng.clone(),
            scratch: Vec::new(),
        };
        tmp.device_plan_trunc()
    }

    /// Canonical-RS twin of `device_plan` for DRAFTED verify rows: consumes
    /// two uniforms (accept test + residual recovery). Only meaningful when
    /// the slot is device-plannable at temperature > 0 - greedy slots keep
    /// the classic exact-match rule, which already is canonical rejection
    /// sampling for a point (argmax) proposal.
    pub fn rs_verify_plan(&mut self) -> Option<DevicePlan> {
        if !self.is_device_plannable() || self.params.temperature <= 0.0 {
            return None;
        }
        let u1 = self.next_uniform();
        let u2 = self.next_uniform();
        Some(DevicePlan::RsVerify {
            inv_t: 1.0 / self.params.temperature,
            u1,
            u2,
        })
    }

    /// Truncation-aware twin of [`Self::rs_verify_plan`] (rung G): the
    /// slot's elected nucleus (top_k 1..=64, top_p, min_p) rides along so
    /// the resolve judges the draft against the same distribution the
    /// dense mode-5 sampler draws from. None for greedy slots, host-only
    /// sampling features, or top_k outside the 64-head kernel - the caller
    /// falls back to the classic plan, which stays lossless.
    pub fn rs_trunc_plan(&mut self) -> Option<DevicePlan> {
        let p = &self.params;
        if p.temperature <= 0.0 || p.top_k > 64 {
            return None;
        }
        if !p.logit_bias.is_empty()
            || self.ngram_armed()
            || (p.repeat_last_n > 0
                && (p.repeat_penalty != 1.0
                    || p.presence_penalty != 0.0
                    || p.frequency_penalty != 0.0))
        {
            return None;
        }
        let (k, top_p, min_p) = (p.top_k as u32, p.top_p, p.min_p);
        let inv_t = 1.0 / p.temperature;
        let u1 = self.next_uniform();
        let u2 = self.next_uniform();
        Some(DevicePlan::RsTrunc {
            inv_t,
            u1,
            u2,
            k,
            top_p,
            min_p,
        })
    }

    /// RS chain draws for one spec round: the drafter-softmax inverse
    /// temperature (0 = greedy/argmax chain rows) plus one draft-draw
    /// uniform per potential chain step. The proposal runs at the REQUEST
    /// temperature - acceptance is maximized when q tracks p's entropy.
    pub fn rs_chain_draw(&mut self, k: usize) -> (f32, Vec<f32>) {
        if self.params.temperature <= 0.0 {
            return (0.0, vec![0.0; k]);
        }
        let inv_t = 1.0 / self.params.temperature;
        let us = (0..k).map(|_| self.next_uniform()).collect();
        (inv_t, us)
    }

    /// Non-consuming twin of [`Self::device_plan`] for plans that may not
    /// execute: the mixed tick peeks a plan for every chunk-prefilling slot,
    /// but only the slot(s) that actually FINISH this tick consume it - the
    /// generator decides that internally from its budget. The uniform comes
    /// from a CLONE of the rng; call [`Self::commit_device_plan`] when a
    /// `FinishSample::Sampled` confirms the plan ran, so the seed stream
    /// advances exactly as the host draw would have.
    pub fn peek_device_plan(&self) -> Option<DevicePlan> {
        let mut tmp = Self {
            params: self.params.clone(),
            rng: self.rng.clone(),
            scratch: Vec::new(),
        };
        tmp.device_plan()
    }

    /// Advance the seed stream for an executed peeked plan (greedy consumed
    /// nothing; categorical consumed one uniform).
    pub fn commit_device_plan(&mut self, plan: &DevicePlan) {
        if matches!(
            plan,
            DevicePlan::Categorical { .. } | DevicePlan::TruncCat { .. }
        ) {
            let _ = self.next_uniform();
        }
    }

    /// logit_bias, before penalties and filtering (the server validates ids
    /// against the vocab; out-of-range is skipped defensively).
    fn apply_logit_bias(&self, logits: &mut [f32]) {
        for &(id, b) in &self.params.logit_bias {
            if let Some(l) = logits.get_mut(id as usize) {
                *l += b;
            }
        }
    }

    fn apply_repeat_penalty(&self, logits: &mut [f32], history: &[u32]) {
        let p = &self.params;
        if p.repeat_last_n == 0 {
            return;
        }
        let start = history.len().saturating_sub(p.repeat_last_n);
        let window = &history[start..];
        if p.repeat_penalty != 1.0 {
            for &tok in window {
                if let Some(l) = logits.get_mut(tok as usize) {
                    // llama.cpp convention: divide if positive, multiply if negative
                    *l = if *l > 0.0 {
                        *l / p.repeat_penalty
                    } else {
                        *l * p.repeat_penalty
                    };
                }
            }
        }
        // OpenAI-style additive penalties over the same window
        if p.presence_penalty != 0.0 || p.frequency_penalty != 0.0 {
            let mut counts: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
            for &tok in window {
                *counts.entry(tok).or_insert(0) += 1;
            }
            for (tok, n) in counts {
                if let Some(l) = logits.get_mut(tok as usize) {
                    *l -= p.presence_penalty + p.frequency_penalty * n as f32;
                }
            }
        }
    }

    /// The sliding-window no-repeat-ngram ban, a line-for-line mirror of the
    /// reference processor (`modeling_unlimitedocr.py`,
    /// `SlidingWindowNoRepeatNgramProcessor.__call__`): `history` plays its
    /// `input_ids` row. The reference's sequence is prompt+generated and so is
    /// ours - for a multimodal prompt ours carries the TEXT ids where the
    /// reference carries ~900 copies of the image placeholder too, but a run
    /// of one repeated placeholder id can never match a generated text
    /// (n-1)-gram at n=35, so the ban sets are identical where it matters.
    fn apply_no_repeat_ngram(&self, logits: &mut [f32], history: &[u32]) {
        let (n, w, stop) = self.params.no_repeat_ngram;
        if stop {
            return; // the stop mode never touches the logits
        }
        for_each_ngram_ban((n, w), history, |tok| {
            if let Some(l) = logits.get_mut(tok as usize) {
                *l = f32::NEG_INFINITY;
            }
        });
    }

    /// Would the no-repeat-ngram guard mask anything at this history? The
    /// device-sampling eligibility probe: a greedy row whose
    /// guard bans nothing this tick samples as a raw argmax, which the
    /// device row sampler matches bit-exactly - so the scheduler grants it
    /// `DevicePlan::Greedy` and only ban-live ticks pay the host readback.
    /// Shares `for_each_ngram_ban` with the mask so the two can't diverge.
    pub fn ngram_would_ban(&self, history: &[u32]) -> bool {
        let mut any = false;
        let (n, w, stop) = self.params.no_repeat_ngram;
        if !stop {
            for_each_ngram_ban((n, w), history, |_| any = true);
        }
        any
    }

    /// True when the no-repeat-ngram guard is the only thing keeping this
    /// sampler off the device greedy path: temperature 0, no bias, neutral
    /// penalties - `is_pure_greedy` minus its !ngram clause. Rows in this
    /// class flip between Device(Greedy) and Host per tick on
    /// `ngram_would_ban`.
    pub fn is_greedy_ngram_only(&self) -> bool {
        let p = &self.params;
        p.temperature <= 0.0
            && p.logit_bias.is_empty()
            && self.ngram_armed()
            && (p.repeat_last_n == 0
                || (p.repeat_penalty == 1.0
                    && p.presence_penalty == 0.0
                    && p.frequency_penalty == 0.0))
    }

    /// Like `sample`, but only tokens passing `legal` can win. Greedy masks
    /// the argmax and retries (exact). Stochastic filters the surviving
    /// candidate set by `legal` before the draw, falling back to the highest-
    /// probability legal token overall. `None` = no legal token exists
    /// (constraint deadlock) - the caller must fail the sequence.
    pub fn sample_constrained(
        &mut self,
        logits: &mut [f32],
        history: &[u32],
        legal: &mut dyn FnMut(u32) -> bool,
    ) -> Option<u32> {
        self.apply_logit_bias(logits);
        self.apply_repeat_penalty(logits, history);
        self.apply_no_repeat_ngram(logits, history);

        if self.params.temperature <= 0.0 {
            loop {
                let best = argmax(logits);
                if logits[best as usize] == f32::NEG_INFINITY {
                    return None; // every token masked
                }
                if legal(best) {
                    return Some(best);
                }
                logits[best as usize] = f32::NEG_INFINITY;
            }
        }

        let u = self.next_uniform();
        let n = self.build_nucleus(logits);
        // constraint filter on the surviving nucleus (probs stay as computed
        // over the pre-filter set, matching the unconstrained pipeline)
        self.scratch.truncate(n);
        self.scratch.retain(|c| legal(c.0));
        if self.scratch.is_empty() {
            // nucleus fully illegal: highest-probability legal token overall.
            // Walk argmax-and-mask in descending order - grammars typically
            // legalize within a few tokens, so this beats sorting the vocab.
            loop {
                let best = argmax(logits);
                if logits[best as usize] == f32::NEG_INFINITY {
                    return None;
                }
                if legal(best) {
                    return Some(best);
                }
                logits[best as usize] = f32::NEG_INFINITY;
            }
        }
        Some(draw(&self.scratch, u))
    }

    /// Pick the next token id. `history` is the tokens so far (for the
    /// repetition penalty). `logits` is consumed as scratch.
    pub fn sample(&mut self, logits: &mut [f32], history: &[u32]) -> u32 {
        self.apply_logit_bias(logits);
        self.apply_repeat_penalty(logits, history);
        self.apply_no_repeat_ngram(logits, history);

        if self.params.temperature <= 0.0 {
            return argmax(logits);
        }

        let u = self.next_uniform();
        let p = &self.params;
        if p.top_k == 0 && p.top_p >= 1.0 && p.min_p <= 0.0 {
            // no truncation filter: sort-free categorical draw
            return sample_all(logits, 1.0 / p.temperature, u);
        }
        let n = self.build_nucleus(logits);
        draw(&self.scratch[..n], u)
    }

    /// One uniform in [0,1) per sampled token - every stochastic path draws
    /// exactly once so the seed's RNG stream stays path-independent. next_u32
    /// is on the low-level `Rng` trait (rand_core 0.10: RngCore is an alias
    /// `RngCore: Rng`), stable across the rand/rand_chacha version boundary.
    fn next_uniform(&mut self) -> f32 {
        use rand_core::Rng as _;
        self.rng.next_u32() as f32 / (u32::MAX as f32 + 1.0)
    }

    /// Run temperature -> top-k -> softmax -> min-p -> top-p over a candidate
    /// HEAD (top_k wide, or 512 grown geometrically until it provably brackets
    /// every survivor), leaving the surviving candidates in `self.scratch[..n]`
    /// as (id, prob) sorted by descending probability. Probabilities are
    /// normalized over the survivors' parent set exactly like the classic
    /// pipeline: over the top-k head when top_k is set, over the full vocab
    /// otherwise (nucleus semantics) - the tail's mass is a second O(V) pass,
    /// never a vocab sort.
    fn build_nucleus(&mut self, logits: &[f32]) -> usize {
        let p = &self.params;
        let inv_t = 1.0 / p.temperature;
        let n = logits.len();
        let mut width = if p.top_k > 0 {
            p.top_k.min(n)
        } else {
            512.min(n)
        };
        // full-vocab softmax normalizer (shift by the global max; the head
        // always contains the argmax so the shift matches the head's)
        let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) * inv_t;
        let full_sum: f32 = if p.top_k > 0 {
            0.0 // unused: top-k renormalizes over the head
        } else {
            logits.iter().map(|&l| (l * inv_t - m).exp()).sum()
        };
        loop {
            let cand = &mut self.scratch;
            cand.clear();
            cand.extend(
                logits
                    .iter()
                    .enumerate()
                    .map(|(i, &l)| (i as u32, l * inv_t)),
            );
            if width < cand.len() {
                cand.select_nth_unstable_by(width - 1, |a, b| b.1.total_cmp(&a.1));
                cand.truncate(width);
            }
            cand.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

            let mut head_sum = 0.0f32;
            for c in cand.iter_mut() {
                c.1 = (c.1 - m).exp();
                head_sum += c.1;
            }
            let denom = if p.top_k > 0 { head_sum } else { full_sum };
            // `!(x > 0.0)` deliberately: a NaN denominator must land here too,
            // and `x <= 0.0` would let it through
            #[allow(clippy::neg_cmp_op_on_partial_ord)]
            if !(denom > 0.0) {
                // degenerate distribution (all mass underflowed): argmax head
                self.scratch.truncate(1);
                return self.scratch.len();
            }
            for c in cand.iter_mut() {
                c.1 /= denom;
            }

            let mut keep = cand.len();
            if p.min_p > 0.0 {
                let thresh = p.min_p * cand[0].1;
                let surviving = cand.iter().take_while(|c| c.1 >= thresh).count();
                // head fully above threshold with vocab left outside -> the
                // boundary may lie beyond the head; widen and retry
                if p.top_k == 0 && surviving == cand.len() && width < n {
                    width = (width * 4).min(n);
                    continue;
                }
                keep = surviving;
            }
            if p.top_p < 1.0 {
                let mut cum = 0.0f32;
                let mut kp = keep;
                let mut reached = false;
                for (i, c) in cand[..keep].iter().enumerate() {
                    cum += c.1;
                    if cum >= p.top_p {
                        kp = i + 1;
                        reached = true;
                        break;
                    }
                }
                // nucleus mass not reached inside the head -> widen and retry
                if p.top_k == 0 && !reached && width < n {
                    width = (width * 4).min(n);
                    continue;
                }
                keep = kp;
            }
            return keep.max(1);
        }
    }
}

/// Sort-free categorical draw over the full distribution: one O(V) pass for
/// the softmax normalizer, one to walk to the u-quantile. This is the hot
/// serving path (OpenAI defaults: temperature only, top_p = 1) - the old
/// implementation built and SORTED a vocab-sized candidate list per token
/// (~6 ms at 201k vocab), which capped every serving concurrency at ~140
/// aggregate tok/s while the GPU sat idle.
fn sample_all(logits: &[f32], inv_t: f32, u: f32) -> u32 {
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) * inv_t;
    if !m.is_finite() {
        return argmax(logits);
    }
    let mut sum = 0.0f32;
    for &l in logits.iter() {
        sum += (l * inv_t - m).exp();
    }
    // `!(x > 0.0)` deliberately: a NaN sum must fall back to argmax too
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(sum > 0.0) {
        return argmax(logits);
    }
    let mut r = u * sum;
    let mut last = 0u32;
    for (i, &l) in logits.iter().enumerate() {
        let e = (l * inv_t - m).exp();
        if e > 0.0 {
            last = i as u32;
            r -= e;
            if r <= 0.0 {
                return i as u32;
            }
        }
    }
    last // fp round-off tail: the highest-index token with mass
}

/// Draw from (id, prob) candidates with the caller's uniform (probs need not
/// sum to 1 - truncation leaves the surviving mass unnormalized).
/// P65: host-side nucleus sampling over a device-selected top-K HEAD -
/// the exact `build_nucleus` (top_k > 0 branch) + `draw` pipeline on the
/// compact candidates the pd_topk_rows kernel returned. Valid whenever the
/// head is a superset of the top_k candidates and contains the row argmax
/// (K = 64 ≥ k guarantees both): selection order is f32 total order on
/// scaled logits exactly like the host's total_cmp (inv_t > 0 monotonic),
/// m is the global max (the argmax is in the head), and the denominator is
/// the top-k head sum - the top_k > 0 semantics verbatim.
pub fn sample_trunc_head(
    head: &[(u32, f32)],
    inv_t: f32,
    u: f32,
    k: u32,
    top_p: f32,
    min_p: f32,
) -> u32 {
    let mut cand: Vec<(u32, f32)> = head.iter().map(|&(id, l)| (id, l * inv_t)).collect();
    cand.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    cand.truncate((k as usize).max(1));
    let m = cand[0].1;
    let mut head_sum = 0.0f32;
    for c in cand.iter_mut() {
        c.1 = (c.1 - m).exp();
        head_sum += c.1;
    }
    // `!(x > 0.0)` deliberately: a NaN head sum must take the argmax path too
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(head_sum > 0.0) {
        return cand[0].0; // degenerate: argmax head, like build_nucleus
    }
    for c in cand.iter_mut() {
        c.1 /= head_sum;
    }
    let mut keep = cand.len();
    if min_p > 0.0 {
        let thresh = min_p * cand[0].1;
        keep = cand.iter().take_while(|c| c.1 >= thresh).count();
    }
    if top_p < 1.0 {
        let mut cum = 0.0f32;
        let mut kp = keep;
        for (i, c) in cand[..keep].iter().enumerate() {
            cum += c.1;
            if cum >= top_p {
                kp = i + 1;
                break;
            }
        }
        keep = kp;
    }
    draw(&cand[..keep.max(1)], u)
}

fn draw(cand: &[(u32, f32)], u: f32) -> u32 {
    let total: f32 = cand.iter().map(|c| c.1).sum();
    let mut r = u * total;
    for c in cand {
        r -= c.1;
        if r <= 0.0 {
            return c.0;
        }
    }
    cand.last().map(|c| c.0).unwrap_or(0)
}

/// Walk the tokens the sliding-window no-repeat-ngram guard bans at this
/// history - a line-for-line mirror of the reference processor
/// (`modeling_unlimitedocr.py`, `SlidingWindowNoRepeatNgramProcessor`),
/// factored out so the logits mask (`apply_no_repeat_ngram`) and the
/// device-eligibility probe (`ngram_would_ban`) share one scan and can
/// never diverge.
fn for_each_ngram_ban(no_repeat_ngram: (usize, usize), history: &[u32], mut ban: impl FnMut(u32)) {
    let (n, window) = no_repeat_ngram;
    if n == 0 || window == 0 || history.len() < n {
        return;
    }
    let search_start = history.len().saturating_sub(window);
    let search_end = history.len() - n + 1; // exclusive, like the range()
    if search_end <= search_start {
        return;
    }
    let prefix = &history[history.len() - (n - 1)..]; // empty when n == 1
    for idx in search_start..search_end {
        let gram = &history[idx..idx + n];
        if n == 1 || gram[..n - 1] == *prefix {
            ban(gram[n - 1]);
        }
    }
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, v) in logits.iter().enumerate() {
        if *v > logits[best] {
            best = i;
        }
    }
    best as u32
}

/// log_softmax stats of `logits` for the chosen token + top-k alternatives.
/// Runs on the RAW logits, before any penalty/temperature mutation.
pub(crate) fn compute_logprobs(
    logits: &[f32],
    chosen: u32,
    top_n: u8,
) -> crate::service::TokenLogprobs {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let lse = max + logits.iter().map(|&l| (l - max).exp()).sum::<f32>().ln();
    let mut top: Vec<(u32, f32)> = Vec::new();
    if top_n > 0 {
        let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
        let k = (top_n as usize).min(idx.len());
        idx.select_nth_unstable_by(k - 1, |&a, &b| {
            logits[b as usize].total_cmp(&logits[a as usize])
        });
        idx.truncate(k);
        idx.sort_unstable_by(|&a, &b| logits[b as usize].total_cmp(&logits[a as usize]));
        top = idx
            .into_iter()
            .map(|i| (i, logits[i as usize] - lse))
            .collect();
    }
    crate::service::TokenLogprobs {
        chosen: logits
            .get(chosen as usize)
            .copied()
            .unwrap_or(f32::NEG_INFINITY)
            - lse,
        top,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_is_deterministic_argmax() {
        let mut s = Sampler::new(SamplingParams::default());
        let mut l = vec![0.1, 0.9, 0.3, 0.2];
        assert_eq!(s.sample(&mut l, &[]), 1);
    }

    #[test]
    fn same_seed_same_sequence() {
        let params = SamplingParams {
            temperature: 1.0,
            seed: 42,
            ..Default::default()
        };
        let logits = vec![1.0f32, 2.0, 0.5, 1.5, 0.2];
        let mut a = Sampler::new(params.clone());
        let mut b = Sampler::new(params);
        for _ in 0..20 {
            assert_eq!(
                a.sample(&mut logits.clone(), &[]),
                b.sample(&mut logits.clone(), &[])
            );
        }
    }

    #[test]
    fn presence_and_frequency_penalties_shift_repeats() {
        let params = SamplingParams {
            presence_penalty: 0.5,
            frequency_penalty: 0.25,
            ..Default::default()
        };
        let mut s = Sampler::new(params);
        // token 1 leads raw, but appears 3x in the window:
        // 0.9 - (0.5 + 0.25*3) = -0.35 < 0.3
        let mut l = vec![0.1, 0.9, 0.3, 0.2];
        assert_eq!(s.sample(&mut l, &[1, 1, 1]), 2);
        // and the penalized sampler is not spec-safe
        assert!(!s.is_pure_greedy());
    }

    #[test]
    fn top_k_one_is_argmax_even_hot() {
        let params = SamplingParams {
            temperature: 5.0,
            top_k: 1,
            seed: 7,
            ..Default::default()
        };
        let mut s = Sampler::new(params);
        let l = vec![0.1, 5.0, 0.3];
        for _ in 0..10 {
            assert_eq!(s.sample(&mut l.clone(), &[]), 1);
        }
    }

    #[test]
    fn no_truncation_draws_match_softmax() {
        // the sort-free path must still be categorical sampling: empirical
        // frequencies over many draws ≈ softmax probabilities
        let params = SamplingParams {
            temperature: 1.0,
            seed: 3,
            ..Default::default()
        };
        let mut s = Sampler::new(params);
        let logits = vec![2.0f32, 1.0, 0.0, -1.0, -2.0];
        let max = 2.0f32;
        let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let mut counts = [0u32; 5];
        let n = 40_000;
        for _ in 0..n {
            counts[s.sample(&mut logits.clone(), &[]) as usize] += 1;
        }
        for (i, &e) in exps.iter().enumerate() {
            let expect = e / sum;
            let got = counts[i] as f32 / n as f32;
            assert!(
                (got - expect).abs() < 0.01,
                "token {i}: got {got:.4}, want {expect:.4}"
            );
        }
    }

    #[test]
    fn top_p_truncates_the_tail() {
        // top token holds ~0.84 of the mass; top_p=0.5 must always pick it
        let params = SamplingParams {
            temperature: 1.0,
            top_p: 0.5,
            seed: 9,
            ..Default::default()
        };
        let mut s = Sampler::new(params);
        let logits = vec![3.0f32, 1.0, 0.5, 0.0];
        for _ in 0..200 {
            assert_eq!(s.sample(&mut logits.clone(), &[]), 0);
        }
    }

    #[test]
    fn nucleus_wider_than_first_head_grows() {
        // near-uniform 4096-token vocab with top_p=0.99: the nucleus spans
        // ~4055 tokens - far beyond the 512 first head, exercising the
        // widen-and-retry path. Draws must cover far more than 512 ids and
        // exclude nothing valid.
        let params = SamplingParams {
            temperature: 1.0,
            top_p: 0.99,
            seed: 5,
            ..Default::default()
        };
        let mut s = Sampler::new(params);
        let logits = vec![0.0f32; 4096];
        let mut seen = std::collections::HashSet::new();
        for _ in 0..20_000 {
            seen.insert(s.sample(&mut logits.clone(), &[]));
        }
        assert!(seen.len() > 2000, "only {} distinct ids drawn", seen.len());
    }

    #[test]
    fn min_p_beyond_first_head_grows() {
        // 2000 equal-logit tokens: every one passes any min_p < 1, so the
        // survivor set must span the whole vocab, not the first 512 head
        let params = SamplingParams {
            temperature: 1.0,
            min_p: 0.5,
            seed: 11,
            ..Default::default()
        };
        let mut s = Sampler::new(params);
        let logits = vec![1.0f32; 2000];
        let mut hi = 0u32;
        for _ in 0..5_000 {
            hi = hi.max(s.sample(&mut logits.clone(), &[]));
        }
        assert!(hi > 1024, "draws capped at {hi} - head never widened");
    }

    /// The reference processor's own math, executed in Python on this exact
    /// LCG sequence, produced these ban sets (generator:
    /// `s = s*6364136223846793005 + 1442695040888963407; tok = (s>>40) % 7`,
    /// seed 0xC0FFEE, 64 tokens). The Rust ban pass must agree exactly.
    #[test]
    fn no_repeat_ngram_matches_the_reference_math() {
        let mut s: u64 = 0xC0FFEE;
        let seq: Vec<u32> = (0..64)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 40) % 7) as u32
            })
            .collect();
        let cases: &[((usize, usize), &[u32])] = &[
            ((3, 64), &[1, 4]),
            ((3, 16), &[]),
            ((2, 64), &[0, 1, 3, 4, 5]),
            ((1, 8), &[0, 2, 3, 4, 5, 6]),
            ((4, 32), &[]),
        ];
        for &((n, w), banned) in cases {
            let s = Sampler::new(SamplingParams {
                no_repeat_ngram: (n, w, false),
                ..Default::default()
            });
            let mut logits = vec![0.0f32; 7];
            s.apply_no_repeat_ngram(&mut logits, &seq);
            let got: Vec<u32> = (0..7u32)
                .filter(|&t| logits[t as usize] == f32::NEG_INFINITY)
                .collect();
            assert_eq!(got, banned, "ban set diverged at (n={n}, window={w})");
        }
    }

    #[test]
    fn no_repeat_ngram_bans_the_repeat_completion() {
        // prefix [1,2] occurred at idx 0 (followed by 3) and idx 4 (followed
        // by 4): both continuations banned, greedy falls to token 5
        let params = SamplingParams {
            no_repeat_ngram: (3, 10, false),
            ..Default::default()
        };
        let mut s = Sampler::new(params);
        let hist = [1u32, 2, 3, 9, 1, 2, 4, 7, 1, 2];
        let mut l = vec![0.0, 0.0, 0.0, 0.9, 0.8, 0.7];
        assert_eq!(s.sample(&mut l, &hist), 5);
        // an armed guard is history-dependent: off every device/spec path
        assert!(!s.is_pure_greedy());
        assert!(!s.is_spec_safe());
        assert!(s.device_plan().is_none());
    }

    #[test]
    fn repeat_stop_fires_where_the_ban_would_and_touches_nothing() {
        // the stop mode: same predicate as the ban, asked of the picked token
        let stop = Sampler::new(SamplingParams {
            no_repeat_ngram: (3, 10, true),
            ..Default::default()
        });
        let ban = Sampler::new(SamplingParams {
            no_repeat_ngram: (3, 10, false),
            ..Default::default()
        });
        // tail (1, 2) occurred before followed by 3: 3 would repeat (1, 2, 3)
        let hist = [1u32, 2, 3, 9, 1, 2];
        assert!(stop.repeat_stop_hit(&hist, 3));
        assert!(!stop.repeat_stop_hit(&hist, 4));
        assert!(!ban.repeat_stop_hit(&hist, 3), "the ban never stops");
        // it masks nothing, so a greedy stop-mode sampler stays a device row
        assert!(stop.is_pure_greedy() && !stop.ngram_would_ban(&hist));
        assert!(!ban.is_pure_greedy() && ban.ngram_would_ban(&hist));
        let mut l = vec![0.0, 0.0, 0.0, 0.9, 0.8, 0.7];
        let mut stop = stop;
        assert_eq!(stop.sample(&mut l, &hist), 3, "the logits are untouched");
        // outside the window, no stop
        let far = [1u32, 2, 3, 9, 9, 9, 9, 9, 9, 9, 9, 9, 1, 2];
        assert!(
            !Sampler::new(SamplingParams {
                no_repeat_ngram: (3, 10, true),
                ..Default::default()
            })
            .repeat_stop_hit(&far, 3)
        );
    }

    #[test]
    fn no_repeat_ngram_window_bounds_the_search() {
        // same history, window 5: both [1,2,_] occurrences START before
        // search_start (10-5=5), so nothing is banned - the reference indexes
        // occurrences by their start, and so must we
        let params = SamplingParams {
            no_repeat_ngram: (3, 5, false),
            ..Default::default()
        };
        let mut s = Sampler::new(params);
        let hist = [1u32, 2, 3, 9, 1, 2, 4, 7, 1, 2];
        let mut l = vec![0.0, 0.0, 0.0, 0.9, 0.8, 0.7];
        assert_eq!(s.sample(&mut l, &hist), 3);
        // (0, _) and (_, 0) are off - the reference gates on both being > 0
        for off in [(0, 128, false), (35, 0, false)] {
            let mut s = Sampler::new(SamplingParams {
                no_repeat_ngram: off,
                ..Default::default()
            });
            let mut l = vec![0.0, 0.0, 0.0, 0.9, 0.8, 0.7];
            assert_eq!(s.sample(&mut l, &hist), 3);
            assert!(s.is_pure_greedy());
        }
    }

    /// The device-eligibility probe must agree with the mask at every
    /// history: `ngram_would_ban` == "apply_no_repeat_ngram writes -inf
    /// somewhere". A false negative would let a Device(Greedy) row skip a
    /// live ban (silent wrong token); a false positive only costs a readback.
    #[test]
    fn ngram_would_ban_agrees_with_the_mask() {
        for (n, w) in [(1usize, 4usize), (2, 6), (3, 5), (3, 128), (35, 128)] {
            let s = Sampler::new(SamplingParams {
                no_repeat_ngram: (n, w, false),
                ..Default::default()
            });
            // deterministic pseudo-random histories over a tiny vocab so
            // repeated grams actually occur
            let mut x = 0x2545f491u32;
            for len in [0usize, 1, 2, 5, 34, 35, 60, 200] {
                let hist: Vec<u32> = (0..len)
                    .map(|_| {
                        x ^= x << 13;
                        x ^= x >> 17;
                        x ^= x << 5;
                        x % 7
                    })
                    .collect();
                let mut logits = vec![1.0f32; 8];
                s.apply_no_repeat_ngram(&mut logits, &hist);
                let masked = logits.contains(&f32::NEG_INFINITY);
                assert_eq!(
                    s.ngram_would_ban(&hist),
                    masked,
                    "probe/mask divergence at n={n} w={w} len={len}"
                );
            }
        }
    }

    #[test]
    fn logit_bias_shifts_and_bans() {
        let params = SamplingParams {
            logit_bias: vec![(0, 6.0), (1, -100.0)],
            ..Default::default()
        };
        let mut s = Sampler::new(params);
        // a biased sequence must fall off the device-argmax spec path
        assert!(!s.is_pure_greedy());
        // +6 flips the greedy winner from 1 to 0; -100 bans token 1 outright
        let mut l = vec![1.0, 5.0, 2.0];
        assert_eq!(s.sample(&mut l, &[]), 0);
        // out-of-range ids are skipped defensively
        let params = SamplingParams {
            logit_bias: vec![(99, 50.0)],
            ..Default::default()
        };
        let mut s = Sampler::new(params);
        let mut l = vec![1.0, 5.0, 2.0];
        assert_eq!(s.sample(&mut l, &[]), 1);
    }

    /// `is_trunc_plannable` must equal "`device_plan_trunc()` is Some" at
    /// every parameter corner - a false positive would hand a pipe a Hole
    /// where it expected a plan, a false negative silently sends the row
    /// back to host sampling - the whole cliff this exists to avoid.
    #[test]
    fn trunc_plannable_twins_the_plan() {
        let corners: &[SamplingParams] = &[
            SamplingParams::default(), // greedy
            SamplingParams {
                temperature: 0.7,
                ..Default::default()
            },
            SamplingParams {
                temperature: 0.7,
                top_k: 20,
                top_p: 0.95,
                ..Default::default()
            },
            SamplingParams {
                temperature: 1.0,
                top_k: 64,
                ..Default::default()
            },
            SamplingParams {
                temperature: 1.0,
                top_k: 65,
                ..Default::default()
            },
            SamplingParams {
                temperature: 1.0,
                top_p: 0.9,
                ..Default::default()
            }, // p-only (mode 6)
            SamplingParams {
                temperature: 1.0,
                min_p: 0.05,
                ..Default::default()
            }, // minp-only (mode 6)
            SamplingParams {
                temperature: 0.8,
                top_p: 0.95,
                min_p: 0.02,
                ..Default::default()
            },
            SamplingParams {
                temperature: 0.7,
                top_k: 20,
                min_p: 0.05,
                ..Default::default()
            },
            SamplingParams {
                temperature: 0.0,
                top_k: 20,
                ..Default::default()
            }, // greedy+k
            SamplingParams {
                temperature: 0.7,
                top_k: 20,
                repeat_penalty: 1.1,
                ..Default::default()
            },
            SamplingParams {
                temperature: 0.7,
                top_k: 20,
                logit_bias: vec![(1, 2.0)],
                ..Default::default()
            },
            SamplingParams {
                temperature: 0.7,
                top_k: 20,
                no_repeat_ngram: (3, 64, false),
                ..Default::default()
            },
            SamplingParams {
                temperature: 0.7,
                top_k: 20,
                presence_penalty: 0.5,
                ..Default::default()
            },
        ];
        for (i, p) in corners.iter().enumerate() {
            let mut s = Sampler::new(p.clone());
            let plannable = s.is_trunc_plannable();
            let peeked = s.peek_device_plan_trunc().is_some();
            let planned = s.device_plan_trunc().is_some();
            assert_eq!(plannable, planned, "corner {i}: plannable != plan");
            assert_eq!(peeked, planned, "corner {i}: peek != plan");
        }
    }

    #[test]
    fn trunc_head_matches_full_host_pipeline() {
        // P65 contract: TruncCat (device top-64 head + sample_trunc_head)
        // must draw the same token as the full host pipeline for the same
        // seed position - the head is a superset of the top-k candidates,
        // the u comes from the same stream slot, and the nucleus math is
        // the top_k>0 branch verbatim. 200 seeded trials, 4k vocab.
        for trial in 0u64..200 {
            let params = SamplingParams {
                temperature: 0.85,
                top_k: 20,
                top_p: 0.95,
                seed: 1000 + trial,
                ..Default::default()
            };
            // deterministic pseudo-random logits (LCG), varied per trial
            let n = 4096usize;
            let mut x = 0x9e3779b97f4a7c15u64 ^ trial.wrapping_mul(0x2545f4914f6cdd1d);
            let logits: Vec<f32> = (0..n)
                .map(|_| {
                    x = x
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((x >> 40) as f32 / (1u64 << 24) as f32) * 12.0 - 6.0
                })
                .collect();
            let mut host = Sampler::new(params.clone());
            let host_tok = host.sample(&mut logits.clone(), &[]);

            let mut dev = Sampler::new(params);
            let plan = dev.device_plan_trunc().expect("trunc-plannable");
            let crate::sampler::DevicePlan::TruncCat {
                inv_t,
                u,
                k,
                top_p,
                min_p,
            } = plan
            else {
                panic!("expected TruncCat, got {plan:?}");
            };
            // host-reference top-64 selection (what pd_topk_rows returns:
            // f32 total order on raw logits, ids arbitrary order)
            let mut idx: Vec<u32> = (0..n as u32).collect();
            idx.sort_unstable_by(|&a, &b| logits[b as usize].total_cmp(&logits[a as usize]));
            let head: Vec<(u32, f32)> =
                idx[..64].iter().map(|&i| (i, logits[i as usize])).collect();
            let dev_tok = sample_trunc_head(&head, inv_t, u, k, top_p, min_p);
            assert_eq!(host_tok, dev_tok, "trial {trial}");
        }
    }
}
