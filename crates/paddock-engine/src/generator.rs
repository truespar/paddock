//! The generation seam: anything that turns a token into next-token logits.
//! One trait over the CPU reference models and the GPU models, so the engine
//! service (and the parity harness) drive them identically.

/// A stateful autoregressive model with a KV cache. Not object-safe across
/// threads by itself - the engine service owns one on its dedicated thread.
pub trait Generator: Send {
    /// Clear the KV cache / position for a fresh sequence.
    fn reset(&mut self);

    /// Feed one token at the current position; return full vocab logits.
    fn forward(&mut self, token: u32) -> Result<Vec<f32>, GenError>;

    fn vocab(&self) -> usize;

    /// The model's context window (KV capacity). Prompts longer than this are
    /// rejected before prefill so an over-length prompt returns a clean error
    /// instead of tripping a deep assert and killing the engine thread. Default
    /// unbounded - the CPU reference models don't cap.
    fn max_context(&self) -> usize {
        usize::MAX
    }

    /// Enable batched decode over up to `max_batch` concurrent sequences, each
    /// pinned to a KV slot (batch row). Returns the capacity actually enabled;
    /// the default `Ok(1)` means this backend can't batch and the engine falls
    /// back to the serial loop.
    fn enable_batch(&mut self, _max_batch: usize) -> Result<usize, GenError> {
        Ok(1)
    }

    /// True if this backend has a working speculative-decode drafter (a loaded MTP
    /// head with spec enabled). The service routes even single-user (max_batch 1)
    /// serving through the batched loop when this holds - spec decode, the big
    /// single-stream speedup, only runs there. Default false (no drafter).
    fn spec_capable(&self) -> bool {
        false
    }

    /// Measured device bytes this generator holds (weights + KV/state pools),
    /// for telemetry. None on CPU backends or when the driver can't say.
    /// Device bytes held by model weights (all serving classes), where the
    /// family accounts for them. Default None = not reported.
    fn weights_mem_bytes(&self) -> Option<u64> {
        None
    }

    /// Device bytes held by the KV cache right now, where the family can
    /// compute them exactly. Default None = not reported.
    fn kv_mem_bytes(&self) -> Option<u64> {
        None
    }

    fn device_mem_used(&self) -> Option<u64> {
        None
    }

    /// One batched decode step: `tokens[i]` at `positions[i]` drives KV slot i.
    /// Returns [rows, vocab] logits (row i = slot i's next-token logits). Only
    /// valid after `enable_batch` returned > 1.
    fn forward_batch(&mut self, _tokens: &[u32], _positions: &[u32]) -> Result<Vec<f32>, GenError> {
        Err(GenError::Backend("batched decode not supported".into()))
    }

    /// True when the backend can sample decode rows on device
    /// (`forward_batch_sampled`): eligible rows come back as bare token ids
    /// and the [rows, vocab] logits readback disappears. Capability probe -
    /// the scheduler checks this before drawing any per-row uniforms, so a
    /// slot's seed stream never pays for a path that won't run.
    fn supports_device_sampling(&self) -> bool {
        false
    }

    /// `forward_batch` + on-device sampling: row i executes `plans[i]`.
    /// `Device` rows return their token in `ids[i]`; `Host` rows (penalties,
    /// filters, constraints, logprobs) get their full logits in `host_rows`
    /// (row order); `Hole` rows return nothing. Only called when
    /// `supports_device_sampling`.
    fn forward_batch_sampled(
        &mut self,
        _tokens: &[u32],
        _positions: &[u32],
        _plans: &[RowSample],
    ) -> Result<SampledStep, GenError> {
        Err(GenError::Backend("device sampling not supported".into()))
    }

    /// True when the backend can run PIPELINED pure-decode ticks: tick N+1 is
    /// enqueued (its inputs advanced on device from tick N's sampled ids)
    /// before tick N's ids reach the host, so the scheduler's commit/SSE work
    /// overlaps the GPU instead of gapping it between steps.
    fn supports_decode_pipe(&self) -> bool {
        false
    }

    /// Post-miss draft-depth floor for the spec round that just ran, when the
    /// backend's drafting regime wants a non-classic value. A block drafter
    /// drafts its whole block in one forward, so depth is nearly free and the
    /// classic post-miss rule (k -> accepted, floor 1) makes it spend most
    /// rounds re-climbing - the measured k death-spiral. A chain drafter pays
    /// per depth, so the classic floor is right for it. This is per-ROUND
    /// (read right after the round, before the controller update): a hybrid
    /// backend answers by which drafter actually ran, so attaching a block
    /// drafter no longer pollutes the chain regime's controller (the old
    /// attach-time PADDOCK_SPEC_K_MISS_FLOOR env default did exactly that -
    /// MTP rounds at live 4..8 re-drafted 7 deep after every miss).
    /// None = the service default (PADDOCK_SPEC_K_MISS_FLOOR, classic 1).
    fn spec_k_miss_floor(&self) -> Option<usize> {
        None
    }

    /// The block width of an attached BLOCK drafter (DFlash: every round
    /// drafts `block - 1` positions in one forward, so the round's natural
    /// verify width is `live * block`).
    /// The service's low-live row tier sizes itself from this instead of
    /// the chain drafter's 32-row pin - at a rejection-sampling acceptance
    /// of ~0.88/draft the deeper round keeps paying all the way to the
    /// block. None = no block drafter.
    fn spec_block_width(&self) -> Option<usize> {
        None
    }

    /// True when this backend may receive `Device(Greedy)` plans for rows
    /// whose sampler is greedy + no-repeat-ngram ONLY. The scheduler grants
    /// the plan per tick, only after verifying the guard would ban nothing at
    /// the row's current history - a no-op mask leaves raw logits, so the
    /// device argmax is bit-exact; ban-live ticks fall back to the Host row.
    /// The scheduler additionally refuses this whenever the decode pipe is
    /// active: pipe plans are drawn before the previous token lands, where
    /// the per-tick ban check would go stale.
    fn device_greedy_ngram_ok(&self) -> bool {
        false
    }

    /// Start a pipelined decode over the same rows `forward_batch_sampled`
    /// would take - but `plans` must be all Device/Hole (no Host rows). No
    /// ids return here; the first `decode_pipe_next` yields tick 0's.
    fn decode_pipe_begin(
        &mut self,
        _tokens: &[u32],
        _positions: &[u32],
        _plans: &[RowSample],
    ) -> Result<(), GenError> {
        Err(GenError::Backend("pipelined decode not supported".into()))
    }

    /// Enqueue the next tick with `plans` (same rows as begin), returning the
    /// ids of the oldest in-flight tick. `ids[i]` is meaningful where that
    /// tick's plan was Device.
    fn decode_pipe_next(&mut self, _plans: &[RowSample]) -> Result<Vec<u32>, GenError> {
        Err(GenError::Backend("pipelined decode not supported".into()))
    }

    /// End the pipe: return the last in-flight tick's ids without enqueueing
    /// more work. Must be called before any other forward call.
    fn decode_pipe_drain(&mut self) -> Result<Vec<u32>, GenError> {
        Err(GenError::Backend("pipelined decode not supported".into()))
    }

    /// True when the backend can OVERLAP a prefill span with decode ticks on
    /// a second execution lane (route B): decode lane forked +
    /// unified spans + slot-mapped pipes available. Gates the overlapped
    /// scheduler branch.
    fn supports_overlap(&self) -> bool {
        false
    }

    /// `decode_pipe_begin` over an ARBITRARY slot set: row i drives
    /// `slots[i]` (the overlap scheduler's churn-phase decode set - never
    /// contiguous). Same contract otherwise.
    fn decode_pipe_begin_slots(
        &mut self,
        _slots: &[u32],
        _tokens: &[u32],
        _positions: &[u32],
        _plans: &[RowSample],
    ) -> Result<(), GenError> {
        Err(GenError::Backend(
            "slot-mapped pipelined decode not supported".into(),
        ))
    }

    /// Launch a prefill-only unified span without waiting for it (overlap
    /// scheduler); returns false (launching nothing) when the chunk queue is
    /// empty. `unified_span_finish` must run before any other forward call
    /// except decode-pipe ticks, which are the point of the split.
    fn unified_span_launch(
        &mut self,
        _budget: usize,
        _fin_plans: &[(usize, RowSample)],
    ) -> Result<bool, GenError> {
        Err(GenError::Backend(
            "unified span launch not supported".into(),
        ))
    }

    /// Non-blocking: has the in-flight span's GPU work completed? True when
    /// no span is in flight.
    fn unified_span_done(&self) -> bool {
        true
    }

    /// Complete the in-flight span (drain, finisher ids, chunk-queue
    /// advance); returns the finished prompts, same tuple contract as
    /// `forward_unified_sampled`'s second return.
    fn unified_span_finish(&mut self) -> Result<Vec<(usize, FinishSample, usize)>, GenError> {
        Err(GenError::Backend(
            "unified span finish not supported".into(),
        ))
    }

    /// Prefill a whole prompt into `slot`'s KV cache in one (chunked) pass and
    /// return the last token's next-token logits. The batched TTFT path.
    fn forward_prefill(&mut self, _slot: usize, _tokens: &[u32]) -> Result<Vec<f32>, GenError> {
        Err(GenError::Backend("batched prefill not supported".into()))
    }

    /// Prefill several `(slot, tokens)` prompts together (one weight-amortized pass
    /// over their concatenated cache-divergent tails), returning each prompt's last
    /// logits in order. Default: fall back to prefilling one at a time.
    fn forward_prefill_batch(
        &mut self,
        items: &[(usize, Vec<u32>)],
    ) -> Result<Vec<Vec<f32>>, GenError> {
        items
            .iter()
            .map(|(slot, tokens)| self.forward_prefill(*slot, tokens))
            .collect()
    }

    /// Bulk-prefill a whole prompt for the SINGLE-STREAM serial path. The model
    /// was just `reset` (fresh sequence at position 0); prefill the whole prompt
    /// in one pass and return the last token's next-token logits - decode then
    /// continues via `forward`, exactly the reset->prefill->forward contract
    /// `forward_multimodal` uses for images. The default feeds token-by-token
    /// (correct, but one forward pass per token - a long prompt prefills at decode
    /// speed); GPU models override with a single batched prefill (the same class
    /// of kernels the batched TTFT path uses).
    fn forward_prefill_stream(&mut self, tokens: &[u32]) -> Result<Vec<f32>, GenError> {
        let mut logits = Vec::new();
        for &t in tokens {
            logits = self.forward(t)?;
        }
        Ok(logits)
    }

    /// Exclusive multimodal prefill: resets all sequence state and prefills the
    /// interleaved text/image chunks from position 0, returning the last row's
    /// logits AND the row count it prefilled; decode then continues via
    /// `forward`. `Ok(None)` = backend has no vision path. The engine drains
    /// every batch slot before calling this and admits nothing new until the
    /// request completes.
    ///
    /// The row count is not decoration: it is what `usage.prompt_tokens` must
    /// report, and it is not the prompt's token count - one `<image>` chunk
    /// becomes the picture's whole row run. The slot lane returns it from
    /// `forward_prefill_multimodal` for exactly the same reason; it used to be
    /// logits-only, which is why a serial-engine image request billed as its
    /// text alone.
    fn forward_multimodal(
        &mut self,
        _chunks: &[crate::service::MmChunk],
    ) -> Result<Option<(Vec<f32>, usize)>, GenError> {
        Ok(None)
    }

    /// One speculative batched decode round over ragged per-slot chunks:
    /// `reqs[i] = (slot, start_pos, chunk)` with `chunk[0]` the slot's
    /// committed pending token and `chunk[1..]` drafts. Returns each row's
    /// GREEDY pick, flat in request order (callers accept per slot while
    /// `chunk[j+1] == picks[base + j]`). `Ok(None)` = backend doesn't
    /// support speculation - the engine falls back to `forward_batch`.
    /// Greedy only: picks are argmaxes, so sampling slots cannot ride this.
    fn forward_spec_batch(
        &mut self,
        _reqs: &[(usize, usize, Vec<u32>)],
    ) -> Result<Option<Vec<u32>>, GenError> {
        Ok(None)
    }

    /// Model-side speculative drafts for the serving spec round (MTP draft
    /// heads etc). `pendings[i] = (slot, pending token)`, one entry per live
    /// slot in round order; returns up to `k` drafts per entry (fewer is
    /// fine). `Ok(None)` = no model drafter (or its per-slot state is stale
    /// this tick) - the service falls back to its n-gram drafter. Called at
    /// most once per round, immediately before `forward_spec_batch` with
    /// chunks built from these drafts.
    fn spec_draft_batch(
        &mut self,
        _pendings: &[(usize, u32)],
        _k: usize,
    ) -> Result<Option<Vec<Vec<u32>>>, GenError> {
        Ok(None)
    }

    /// Async spec round, phase 1: enqueue the drafter chain and
    /// return its effective draft depth plus a per-pendings-entry kept flag
    /// without reading drafts back. The service then calls
    /// `forward_spec_batch` with placeholder chunk VALUES - kept entries at
    /// their capped draft length, un-kept (chain-cold) entries as length-1
    /// chunks, exactly the lengths the synchronous path would build (row
    /// counts are graph keys; divergence causes capture storms) - and pairs
    /// every armed call with `spec_draft_fetch` after the verify for the
    /// accept replay. `Ok(None)` = can't arm; fall back to the synchronous
    /// `spec_draft_batch` round.
    fn spec_draft_begin(
        &mut self,
        _pendings: &[(usize, u32)],
        _k: usize,
    ) -> Result<Option<(usize, Vec<bool>)>, GenError> {
        Ok(None)
    }

    /// Async spec round, phase 2: drafts of the armed `spec_draft_begin`
    /// call, indexed by its pendings order (cold slots empty). Clears the
    /// armed state; `Ok(None)` if nothing was armed.
    fn spec_draft_fetch(&mut self) -> Result<Option<Vec<Vec<u32>>>, GenError> {
        Ok(None)
    }

    /// True when this backend runs the canonical rejection-sampling spec arm
    /// (PADDOCK_SPEC_RS): drafts SAMPLED from the drafter softmax
    /// + full-q accept/recover on verify. The service then stashes per-slot
    ///   chain draws before each `spec_draft_begin` and marks drafted verify
    ///   rows with `DevicePlan::RsVerify`.
    fn supports_spec_rs(&self) -> bool {
        false
    }

    /// Rung G: true when this backend's drafter SAMPLES its
    /// drafts at the request temperature and its verify resolves drafted
    /// rows with the truncation-aware rejection sampler (pack slots
    /// 470/471). The service then emits `DevicePlan::RsTrunc` for drafted
    /// verify rows of nucleus-sampling slots (and still stashes the per-slot
    /// chain draws, whose inv_t + first uniform seed the sampled walk).
    fn supports_spec_rs_trunc(&self) -> bool {
        false
    }

    /// True when this backend implements the P65 host-head sampling finish:
    /// rows planned `DevicePlan::TruncCat` get the device top-K prefilter
    /// (pack slot 434) and the backend host-samples the compact head into
    /// `step.ids` like any device row. Service plan sites emit TruncCat
    /// only when this holds - a backend without the finish would leave the
    /// row unsampled.
    fn supports_host_head(&self) -> bool {
        false
    }

    /// P67b: true when TruncCat rows execute fully on device (pack slot 435
    /// mode 5 - head build + nucleus draw in-kernel, token in the sampled-id
    /// plane). Only then are truncation rows pipe/overlap admissible: the
    /// zero-host decode pipes feed tokens forward device-side and cannot
    /// host-sample.
    fn supports_device_trunc(&self) -> bool {
        false
    }

    /// Stash the RS chain draws for the round about to be drafted (one entry
    /// per pending slot; consumed by the next `spec_draft_begin`). No-op on
    /// backends without the RS arm.
    fn spec_rs_stash(&mut self, _draws: Vec<SpecRsDraw>) {}

    /// Ensure the slot's draft-head (MTP) state is warm through its decode
    /// cursor before a spec round. Dense/mixed ticks advance the backbone KV
    /// but not the draft-head warm; this re-syncs the gap (`committed` is the
    /// slot's committed token sequence; `want_pos` the KV-space position of
    /// its last committed row - equal to `committed.len()-1` for pure-text
    /// slots, offset past it by the image rows on multimodal slots). Returns
    /// whether the slot is warm and spec-ready. Default backends have no
    /// draft head -> false.
    fn spec_ensure_warm(
        &mut self,
        _slot: usize,
        _committed: &[u32],
        _want_pos: u32,
    ) -> Result<bool, GenError> {
        Ok(false)
    }

    /// True when `spec_draft_batch` decides warmth per SLOT - it filters cold
    /// slots itself and returns an empty draft list for each, so the caller
    /// does not have to require every live slot be warm before asking.
    ///
    /// This matters at batch. The scheduler's `all_warm` is a conjunction over
    /// live slots, so P(all warm) decays with slot count: measured on muse at
    /// 8 slots, the model drafter was reached once in 145 decode ticks while
    /// every tick had a healthy k budget - one cold slot sent all eight to the
    /// n-gram fallback. A drafter that already reports per slot should not be
    /// gated that way. Token-replay drafters must keep the conjunction: a cold
    /// slot there desyncs the chain rather than simply drafting nothing.
    fn spec_draft_per_slot_warm(&self) -> bool {
        false
    }

    /// True when the drafter drafts in KV-row space: warmth is checked
    /// against the slot's KV position, so multimodal slots (image rows push
    /// pos past the token history) can spec too. Token-replay drafters
    /// (state synced by re-running committed tokens) can't bridge an image
    /// gap and return false - the scheduler then reports such slots cold.
    fn spec_draft_kv_space(&self) -> bool {
        false
    }

    /// Scheduler hint: whether prefills should eagerly warm the draft head.
    /// The scheduler turns this off while the live count exceeds its spec
    /// engagement cap - warming then is pure prefill cost (no spec round will
    /// ever consume it). Backends without draft-head warming ignore it.
    fn spec_warm_hint(&mut self, _on: bool) {}

    /// Does a per-slot RING drafter (DFlash) own the round at this live
    /// count? When it does, a ring-cold slot simply drafts nothing and rides
    /// the verify, so the scheduler must not pay `spec_ensure_warm`'s
    /// token-replay gap re-warm for it (see `spec_ring_warm`). Backends
    /// without a ring drafter: false (every slot goes through ensure_warm).
    fn spec_ring_owns_round(&self, _live: usize) -> bool {
        false
    }

    /// Pure ring-warmth probe for the round the ring drafter owns - no
    /// re-warm side effect. `None` = no ring drafter attached (use
    /// `spec_ensure_warm`).
    ///
    /// Why: the hybrid seam in `spec_ensure_warm` falls through to the MTP
    /// chain's gap re-warm for a ring-cold slot - a serial single-slot
    /// backbone re-run plus a full DeltaNet state save/restore, ~25 ms - and
    /// the scheduler called it for every live slot every tick. A DFlash round
    /// never advances the MTP cursor, so the gap re-opened every tick: one
    /// admission tick over the drafter's fusion cap wiped all 32 rings, and
    /// the serve then ran 1.0-1.8 s ticks at a quarter of its healthy rate
    /// with the GPU at 95% and clocks green. The re-warm bought nothing:
    /// DFlash owned every round.
    fn spec_ring_warm(&mut self, _slot: usize, _want_pos: u32) -> Option<bool> {
        None
    }

    /// Per-tick: will this width speculate at all? A drafter whose features
    /// are fused from every forward (DFlash) otherwise pays that fusion on
    /// ticks that will never draft - a few percent of the tick, which is the
    /// whole margin at c4/c8.
    ///
    /// The trade is real and deliberate: a slot that decodes through a wide
    /// stretch stops being fed, and `dflash_warm` demands coverage ending
    /// exactly at `p`, so it will not speculate again for the rest of that
    /// request. That is the right side of the trade - at those widths
    /// speculation measured a LOSS anyway - but it is not free, and it is why
    /// this is a hint about width rather than a kill switch.
    fn spec_fuse_hint(&mut self, _on: bool) {}

    /// The backend's spec-round live capacity: rounds with more live slots
    /// than this will decline (e.g. a VRAM-degraded draft-state allocation).
    /// The scheduler clamps its own spec engagement cap by it so warm/draft
    /// attempts aren't made for cohorts the backend can never serve - those
    /// attempts aren't free (ensure_warm re-warms a gap that no round will
    /// ever re-sync, every tick). usize::MAX = no backend limit.
    fn spec_live_cap(&self) -> usize {
        usize::MAX
    }

    /// DEVICE-SAMPLED speculative round: verify the chunks, sample every row
    /// on device with the pre-drawn per-row plans (flat in request order, one
    /// plan per chunk row - exact for greedy/temperature-only samplers, the
    /// same `sample_rows` semantics as the dense device-sampled step), commit
    /// internally with accept-while-match, and return the sampled picks in
    /// the service's flat layout. No logits readback - this is the sampled
    /// path that scales past the host round's live cap. `Ok(None)` = can't
    /// this tick (fall back / cool down).
    fn forward_spec_batch_plans(
        &mut self,
        _reqs: &[(usize, usize, Vec<u32>)],
        _plans: &[crate::sampler::DevicePlan],
    ) -> Result<Option<Vec<u32>>, GenError> {
        Ok(None)
    }

    /// Strip-mode verify for ARMED async rounds - the
    /// accept-while-match walk runs on device right after the tick and the
    /// backend returns one compact per-slot result (accepted count, next
    /// pending, the emitted tokens) in reqs order. No picks, no drafts
    /// fetch, no host replays; the armed plan is consumed. `Ok(None)` =
    /// round declined (caller falls back exactly as with the plans entry;
    /// any armed plan is already cleared or must be fetch-discarded).
    /// Gate on `supports_spec_strip`.
    fn forward_spec_batch_strip(
        &mut self,
        _reqs: &[(usize, usize, Vec<u32>)],
        _plans: &[crate::sampler::DevicePlan],
    ) -> Result<Option<Vec<SpecAccepted>>, GenError> {
        Ok(None)
    }

    /// True when the backend can run strip-mode verify rounds.
    fn supports_spec_strip(&self) -> bool {
        false
    }

    /// Arm the one-ahead spec pipeline on the shape of
    /// the strip round that just ran. False = not steady / not supported.
    fn spec_pipe_arm(&mut self) -> bool {
        false
    }

    /// Enqueue one pipelined round (chain -> verify -> accept/prep, no host
    /// uploads, never syncs). `par` = this round's sampler params, row-major
    /// (n*k1 rows x 4 u32, the batch d_par layout).
    fn spec_pipe_round(&mut self, _par: &[u32]) -> Result<(), GenError> {
        Err(GenError::Backend("spec pipe not supported".into()))
    }

    /// Read a pipelined round's per-slot results (event-gated; syncs only
    /// that round). `half` alternates 0/1 with the enqueue order.
    fn spec_pipe_strip(&mut self, _half: usize) -> Result<Vec<SpecAccepted>, GenError> {
        Err(GenError::Backend("spec pipe not supported".into()))
    }

    /// Pre-grow the paged KV tables to a per-slot position horizon (host-
    /// side, evicting; called between pipelined rounds so the in-flight
    /// graphs never allocate).
    fn spec_pipe_ensure(&mut self, _slots: &[u32], _positions: &[u32]) -> Result<(), GenError> {
        Ok(())
    }

    /// Drain the pipeline (waits out queued rounds; host mirrors are the
    /// caller's, resynced from the strips it consumed).
    fn spec_pipe_drain(&mut self) -> Result<(), GenError> {
        Ok(())
    }

    /// SAMPLED speculative round, phase 1: verify the ragged chunks and
    /// return the RAW per-row logits, concatenated in request order
    /// ([sum(chunk.len()), vocab] flat). The service samples each row with
    /// the slot's own sampler (exact rejection sampling for deterministic
    /// drafts) and then calls `spec_commit` with the per-request committed
    /// row counts. `Ok(None)` = can't spec this tick (fall back to dense).
    fn forward_spec_verify(
        &mut self,
        _reqs: &[(usize, usize, Vec<u32>)],
    ) -> Result<Option<Vec<f32>>, GenError> {
        Ok(None)
    }

    /// SAMPLED speculative round, phase 2: commit `committed[i]` chunk rows
    /// for request i of the round opened by the last `forward_spec_verify`
    /// (state rollback + draft-head catchup). Must be called exactly once
    /// per successful verify, before any other forward.
    fn spec_commit(&mut self, _committed: &[u32]) -> Result<(), GenError> {
        Ok(())
    }

    /// Prompt tokens the last prefill of `slot` served from a prefix cache
    /// (usage reporting; taken - resets to 0). 0 = no cache / no reuse.
    fn take_prefill_reused(&mut self, _slot: usize) -> usize {
        0
    }

    /// KV-tier maintenance: advance the tier transport -
    /// collect completions, release demote pins, start queued IO. Called
    /// once per scheduler pass; default no-op for untiered backends. Without
    /// a tick-driven pump the demote queue only advances when a request
    /// happens to touch the prefix cache, and an otherwise-idle machine never
    /// drains its own demotes (found live on the first granite probe).
    fn tier_pump(&mut self) {}

    /// Park/wake (KVFlow): true when `tokens`' prefix is being restored
    /// from the KV tier for `slot` - the scheduler skips the request this
    /// tick and the batch runs other work; the per-pass `tier_pump` resolves
    /// the flow and the request re-enters admission, where a normal radix
    /// match adopts the published prefix. The first call STARTS the restore
    /// (reservation-first - the destination is seated before any IO).
    /// Untiered backends never park.
    fn tier_prefix_loading(&mut self, _slot: usize, _tokens: &[u32]) -> bool {
        false
    }

    /// Feed a measured prefill span to the tier's cost model (tokens
    /// computed, wall us - decode interleaving included, which is the
    /// honest recompute cost under load). Untiered backends ignore it.
    fn tier_observe_prefill(&mut self, _tokens: u32, _wall_us: f64) {}

    /// The KV tier's snapshot, when one is armed. Polled by the service
    /// once per pass into the metrics gauges.
    fn tier_stats(&self) -> Option<crate::kv_tier::TierStats> {
        None
    }

    /// The tier's report - decisions and their reasons, not just
    /// occupancy. None when the model has no tier armed.
    fn tier_report(&self) -> Option<crate::kv_tier::TierReport> {
        None
    }

    /// Free-on-completion hook (P5b): `occupied[k]` is whether slot `k` still
    /// holds a live sequence. A paged backend returns idle slots' KV blocks to
    /// the shared pool so the freed memory is immediately available to new
    /// admissions, instead of being pinned until the slot is next reused. Called
    /// once per batched tick; idempotent. Default no-op.
    fn release_inactive_slots(&mut self, _occupied: &[bool]) {}

    /// Free KV blocks in the paged budget pool, or `None` when the backend has
    /// no pool (admit by slots only). Drives the P5b watermark: the scheduler
    /// stops admitting new sequences while the pool is nearly full and resumes as
    /// free-on-completion returns blocks (bit-exact - admission timing only).
    fn pool_free_blocks(&self) -> Option<usize> {
        None
    }

    /// True when the backend supports CHUNKED prefill (prefill_begin +
    /// forward_mixed): an admission advances a budget of rows per tick
    /// alongside the live decode rows instead of stalling every stream for
    /// the whole prompt. Capability probe - the scheduler checks this before
    /// taking the mixed path.
    fn supports_chunked_prefill(&self) -> bool {
        false
    }

    /// Begin a chunked prefill on `slot` (only when `supports_chunked_prefill`).
    /// The backend matches + loads any cached prefix immediately; the rest of
    /// the prompt advances via `forward_mixed`. Several may be in flight
    /// (backend-bounded); the backend errors when its queue is full.
    fn prefill_begin(&mut self, _slot: usize, _tokens: Vec<u32>) -> Result<(), GenError> {
        Err(GenError::Backend("chunked prefill not supported".into()))
    }

    /// `prefill_begin` with checkpoint hints: positions inside `tokens` where
    /// the scheduler knows OTHER queued prompts diverge from this one. The
    /// backend snapshots resumable state as the prefill crosses each, so those
    /// prompts adopt the shared prefix instead of re-prefilling it beside this
    /// one. Backends without hint support ignore them. Returns the resume
    /// point (leading tokens served from the prefix cache; 0 = cold).
    fn prefill_begin_hinted(
        &mut self,
        slot: usize,
        tokens: Vec<u32>,
        _hints: &[usize],
    ) -> Result<usize, GenError> {
        self.prefill_begin(slot, tokens).map(|_| 0)
    }

    /// The shared-prefix length (tokens) from which a prompt queued behind an
    /// in-flight prompt would resume off that prompt's published prefix once
    /// it lands. None = no prefix cache: the scheduler holds nothing back.
    fn prefix_share_floor(&self) -> Option<usize> {
        None
    }

    /// Abandon slot `slot`'s in-flight chunked prefill (client hung up).
    /// Returns true when the backend actually dropped it - false means "not
    /// now" (e.g. a fused span referencing the chunk is still in flight) and
    /// the scheduler should retry next tick. Default: backends without abort
    /// support just run the prefill to completion (the older status quo),
    /// so returning false forever is safe, only wasteful.
    fn prefill_abort(&mut self, _slot: usize) -> bool {
        false
    }

    /// One mixed decode+prefill tick: decode rows `decodes[i] = (slot, token,
    /// pos)` plus up to `budget` rows spread over every in-flight chunked
    /// prompt (FIFO) in one weight-amortized pass. Returns the decode logits
    /// (flat [decodes.len(), vocab], input order) and one `(slot, last-token
    /// logits, prompt KV rows)` per prompt that finished this tick.
    fn forward_mixed(
        &mut self,
        _decodes: &[(usize, u32, u32)],
        _budget: usize,
    ) -> Result<(Vec<f32>, Vec<(usize, Vec<f32>, usize)>), GenError> {
        Err(GenError::Backend(
            "mixed decode+prefill not supported".into(),
        ))
    }

    /// `forward_mixed` with fused device sampling for the decode rows - the
    /// mixed-tick twin of `forward_batch_sampled`. Backends without it keep
    /// the unsampled mixed pass (full logits readback). `fin_plans` maps slot
    /// -> sampling plan for chunk-prefilling slots: a backend may device-
    /// sample a finishing prompt's first token with its plan and return
    /// `FinishSample::Sampled` (no logits readback); backends that don't look
    /// at it keep returning `FinishSample::Logits`. Plans carry PEEKED
    /// uniforms - the scheduler advances the slot's seed stream only for
    /// plans that actually executed (`Sampled` came back).
    fn forward_mixed_sampled(
        &mut self,
        _decodes: &[(usize, u32, u32)],
        _budget: usize,
        _plans: &[RowSample],
        _fin_plans: &[(usize, RowSample)],
    ) -> Result<(SampledStep, Vec<(usize, FinishSample, usize)>), GenError> {
        Err(GenError::Backend(
            "mixed device sampling not supported".into(),
        ))
    }

    /// Speculative MIXED tick: the decode rows ride VERIFY chunks
    /// ([pending, drafts...] per slot, device-planned) while the prompt
    /// chunk streams in the same tick. Returns (picks, finished-prefill).
    /// A decline (Ok((None, empty))) must leave the queued prompt chunk
    /// untouched so the scheduler can fall back to the plain mixed tick.
    /// `fin_plans`: per-slot finisher sampling plans - a backend
    /// that honors a `RowSample::Device` entry returns
    /// `FinishSample::Sampled` (no [1, vocab] logits readback for that
    /// finisher); ignoring them and returning `Logits` stays correct.
    fn forward_mixed_spec_plans(
        &mut self,
        _reqs: &[(usize, usize, Vec<u32>)],
        _budget: usize,
        _plans: &[crate::sampler::DevicePlan],
        _fin_plans: &[(usize, RowSample)],
    ) -> Result<(Option<Vec<u32>>, Vec<(usize, FinishSample, usize)>), GenError> {
        Ok((None, Vec::new()))
    }

    /// Issue-ahead: enqueue the mixed spec round and return before
    /// the picks readback. true = launched (the caller must then run
    /// `forward_mixed_spec_finish` on this tick; deferred host work goes
    /// between the two - that window overlaps the round's GPU time). false
    /// = not launched: call `forward_mixed_spec_plans` as before (a backend
    /// that produced a fallback result stashes it for that call, so nothing
    /// runs twice).
    fn forward_mixed_spec_begin(
        &mut self,
        _reqs: &[(usize, usize, Vec<u32>)],
        _budget: usize,
        _plans: &[crate::sampler::DevicePlan],
        _fin_plans: &[(usize, RowSample)],
    ) -> Result<bool, GenError> {
        Ok(false)
    }
    fn forward_mixed_spec_finish(
        &mut self,
    ) -> Result<(Option<Vec<u32>>, Vec<(usize, FinishSample, usize)>), GenError> {
        Err(GenError::Backend("no mixed spec round in flight".into()))
    }

    /// True unified prefill+decode tick: decode rows + one queued prompt fused
    /// into a single weight-amortized forward (vs `forward_mixed_sampled`'s two
    /// forwards). Same contract. Backends without it fall back to the mixed tick,
    /// so the scheduler can call this unconditionally under the opt-in.
    fn forward_unified_sampled(
        &mut self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
        plans: &[RowSample],
        fin_plans: &[(usize, RowSample)],
    ) -> Result<(SampledStep, Vec<(usize, FinishSample, usize)>), GenError> {
        self.forward_mixed_sampled(decodes, budget, plans, fin_plans)
    }

    /// True when the backend can prefill multimodal requests into batch
    /// slots (vision attached + batching on). When false the scheduler runs
    /// them EXCLUSIVELY via `forward_multimodal` (the pre-S8 path).
    fn supports_mm_slots(&self) -> bool {
        false
    }

    /// Multimodal prefill into a batch slot: the last row's logits plus the
    /// slot's KV row count (image rows included - it differs from the
    /// prompt's token count). Only called when `supports_mm_slots`.
    fn forward_prefill_multimodal(
        &mut self,
        _slot: usize,
        _chunks: &[crate::service::MmChunk],
    ) -> Result<(Vec<f32>, usize), GenError> {
        Err(GenError::Backend(
            "multimodal slot prefill not supported".into(),
        ))
    }

    /// True when a `rows`-row prefill pass can run without reallocating the
    /// shared scratch (an overlapped admission must never trigger a realloc -
    /// it would drop decode graphs that still have queued replays). Default
    /// false = the backend does not support overlapped admissions.
    fn prefill_scratch_fits(&self, _rows: usize) -> bool {
        false
    }

    /// Prefill every pending multimodal request of this scheduler pass.
    /// Default = the historical serial per-slot loop; backends with a batched
    /// vision encode override this so concurrent image requests share one
    /// tower pass (otherwise concurrent image requests see a TTFT staircase)
    /// even though the slot prefills still run per-slot. Per-slot results,
    /// order preserved.
    fn forward_prefill_multimodal_batch(
        &mut self,
        items: Vec<(usize, Vec<crate::service::MmChunk>)>,
    ) -> Vec<(usize, Result<(Vec<f32>, usize), GenError>)> {
        items
            .into_iter()
            .map(|(k, chunks)| {
                let r = self.forward_prefill_multimodal(k, &chunks);
                (k, r)
            })
            .collect()
    }

    /// True when image prompts can join the CHUNKED prefill queue rather than
    /// taking a blocking whole-prompt pass. Implies `supports_chunked_prefill`
    /// - it is the same queue and the same mixed ticks.
    fn supports_chunked_multimodal(&self) -> bool {
        false
    }

    /// Admit a whole wave of MULTIMODAL prompts onto the chunked queue.
    ///
    /// The backend encodes every pending request's images (one pass where it
    /// can) and enqueues each slot's ROW PLAN - image rows included - so the
    /// pictures advance under the same row budget as text and never hold the
    /// tick. Returns immediately; completions arrive through `forward_mixed*`
    /// like any other chunked prompt. Per-slot results, order preserved.
    ///
    /// Why it is a wave rather than one slot at a time: the vision encode is
    /// the expensive half and batches across requests, so admitting a wave
    /// one-by-one would give up exactly what `forward_prefill_multimodal_batch`
    /// exists for.
    /// A backend that answers `Encoding` here takes OWNERSHIP of those slots'
    /// chunks: it holds them until an `encode_step` reports the slot, and the
    /// scheduler must neither re-offer the slot nor route it to the text lane
    /// in the meantime (its `mm` is gone, so the text lane would prefill an
    /// empty prompt).
    fn prefill_begin_multimodal(
        &mut self,
        items: Vec<(usize, Vec<crate::service::MmChunk>)>,
    ) -> Vec<(usize, MmAdmit)> {
        items
            .into_iter()
            .map(|(k, _)| {
                (
                    k,
                    MmAdmit::Failed(GenError::Backend(
                        "chunked multimodal prefill not supported".into(),
                    )),
                )
            })
            .collect()
    }

    /// Spend one ENCODER BUDGET on whatever vision work the backend is holding
    /// and report the slots that finished it.
    ///
    /// Called once per tick while `encoding_pending` is true. Empty result =
    /// still going. The verdicts are the same ones `prefill_begin_multimodal`
    /// returns, minus `Encoding` - a slot reported here has left that state.
    ///
    /// The default backend never says `Encoding`, so it never has anything to
    /// step and this is dead weight for it - which is the point: an encoder
    /// budget is a backend capability, not a scheduler policy.
    fn encode_step(&mut self) -> Vec<(usize, MmAdmit)> {
        Vec::new()
    }

    /// True while the backend holds slots mid-encode. The scheduler keeps
    /// calling `encode_step` until it is false.
    fn encoding_pending(&self) -> bool {
        false
    }

    /// The largest image this endpoint's vision tower can use, from the loaded
    /// mmproj. `None` = this model does not take images.
    ///
    /// Every caller that sizes an image - the API's `detail` handling, the
    /// Studio's per-image picker - reads it from here, so there is exactly one
    /// place the number can be wrong and it is next to the file it came from.
    fn vision_budget(&self) -> Option<VisionBudget> {
        None
    }
}

/// How large an image this endpoint's vision tower can actually use.
///
/// Every field is computed by the family from what it LOADED - never from the
/// arch string, and never from llama.cpp's defaults. That distinction is not
/// pedantry: llama.cpp caps Qwen-VL at `set_limit_image_tokens(8, 4096)` while
/// its own comment cites a `preprocessor_config.json` that allows 16384, and
/// we had inherited the quarter-sized cap. The model's published spec is the
/// authority; a serving cap on top of it is a policy that has to be visible.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VisionBudget {
    /// Source pixels above which the encoder's own resize discards detail.
    /// Sending more than this is wasted bytes, not a better answer.
    pub max_pixels: u64,
    /// Pixels below which the encoder upsamples. Sending less than this throws
    /// away detail the model would have used.
    pub min_pixels: u64,
    /// Longest usable edge, when the family bounds one. Granite's AnyRes
    /// pinpoint list does (3840); the smart-resize families bound area only,
    /// so this is None for them and an aspect-preserving area fit is the whole
    /// rule.
    pub max_edge: Option<u32>,
    /// Pixels one vision token covers, so a caller can price an image before
    /// sending it: `tokens ≈ ceil(w * h / pixels_per_token)`, clamped to
    /// `[min_tokens, max_tokens]`. This is what lets the Studio show a real
    /// cost per detail level instead of a guess.
    pub pixels_per_token: u64,
    pub max_tokens: u32,
    pub min_tokens: u32,
}

/// Vision tokens one image may cost under `detail: auto` - the DEFAULT policy
/// on every API surface, so this number is what an ordinary request spends.
///
/// 4096 is not a fresh invention: it is exactly the cap qwen used to serve
/// under (llama.cpp's `set_limit_image_tokens(8, 4096)`, which we had
/// inherited as a hard-coded `max_pixels`), and it sits above granite's
/// 2353-row and gemma4's 280-row ceilings. So `auto` reproduces the resolution
/// every client was already getting, on every family we serve, and `high` is
/// what opens the rest of the model's published spec.
///
/// It lives here rather than in the runner because the Studio's picker and the
/// API's `detail` handling have to agree on it, and a default that two callers
/// can spell differently is a default that will drift.
pub const AUTO_MAX_TOKENS: u32 = 4096;

impl VisionBudget {
    /// Fit `(w, h)` to this budget, preserving aspect. Never upsamples past
    /// `max_pixels` and never returns a zero side.
    pub fn fit(&self, w: u32, h: u32) -> (u32, u32) {
        self.fit_px(w, h, self.max_pixels)
    }

    /// Fit `(w, h)` so the image costs at most `tokens` vision rows - the
    /// `detail` levels resolve to exactly this.
    pub fn fit_tokens(&self, w: u32, h: u32, tokens: u32) -> (u32, u32) {
        self.fit_px(w, h, self.pixels_for_tokens(tokens))
    }

    /// Source pixels a token count is worth here, clamped to what the tower can
    /// actually do. Asking for fewer rows than `min_tokens` cannot make the
    /// encoder emit fewer - it only throws away detail before the encoder sees
    /// it - so the floor is real, not defensive.
    pub fn pixels_for_tokens(&self, tokens: u32) -> u64 {
        let t = tokens.clamp(self.min_tokens, self.max_tokens) as u64;
        t.saturating_mul(self.pixels_per_token)
            .clamp(self.min_pixels, self.max_pixels)
    }

    /// Aspect-preserving area fit under `max_px`, with the family's edge cap
    /// applied first when it has one.
    fn fit_px(&self, w: u32, h: u32, max_px: u64) -> (u32, u32) {
        let (mut w, mut h) = (w.max(1) as f64, h.max(1) as f64);
        if let Some(edge) = self.max_edge {
            let longest = w.max(h);
            if longest > edge as f64 {
                let s = edge as f64 / longest;
                w *= s;
                h *= s;
            }
        }
        let px = w * h;
        if px > max_px as f64 {
            let s = (max_px as f64 / px).sqrt();
            w *= s;
            h *= s;
        }
        // FLOOR, not round. Rounding a scaled-down pair can land back over the
        // ceiling it just enforced - 8000x6000 into qwen's 16,777,216 rounds to
        // 4730x3547 = 16,777,310 - and every caller here treats the result as a
        // hard bound. Half a pixel of aspect drift is the cheap side of that
        // trade.
        ((w as u32).max(1), (h as u32).max(1))
    }

    /// Vision tokens `(w, h)` would cost at this budget - the number the
    /// picker shows next to each detail level.
    pub fn tokens_for(&self, w: u32, h: u32) -> u32 {
        self.tokens_for_capped(w, h, self.max_tokens)
    }

    /// Vision tokens `(w, h)` costs once fitted under a `tokens` ceiling.
    pub fn tokens_for_capped(&self, w: u32, h: u32, tokens: u32) -> u32 {
        let (w, h) = self.fit_tokens(w, h, tokens);
        let n = (w as u64 * h as u64).div_ceil(self.pixels_per_token.max(1));
        (n as u32).clamp(self.min_tokens, self.max_tokens)
    }
}

/// What multimodal admission did with one slot's image prompt.
///
/// Three outcomes rather than a `Result`, because "not yet" is a normal and
/// common answer once there is an encoder budget, and the scheduler has to tell
/// it from both success and failure. Collapsing it into either is how a slot
/// ends up prefilled from an empty prompt (read as queued) or an image request
/// dies for no reason (read as failed).
pub enum MmAdmit {
    /// On the chunked prefill queue - an ordinary chunked prompt from here on.
    Queued,
    /// Still encoding under the backend's encoder budget. The BACKEND owns this
    /// slot's chunks until a later `encode_step` reports it.
    Encoding,
    Failed(GenError),
}

/// Per-slot result of a strip-mode spec round (rung B1): what the device
/// accept emitted, in reqs order.
#[derive(Clone, Debug)]
pub struct SpecAccepted {
    /// accepted rows (a+1): the slot advances this many positions
    pub accepted: usize,
    /// the new pending token (the accepted-final row's sample)
    pub pending: u32,
    /// the emitted tokens (sampled[0..accepted]) - pushed to the slot's
    /// draft window and streamed
    pub tokens: Vec<u32>,
}

/// Per-slot chain draws for one canonical-RS spec round (PADDOCK_SPEC_RS):
/// the drafter-softmax inverse temperature (0 = greedy/argmax chain rows)
/// plus one draft-draw uniform per potential chain step, drawn from the
/// slot's seed stream by the service (the backend has no sampler access).
#[derive(Clone, Debug)]
pub struct SpecRsDraw {
    /// the slot these draws belong to (pendings are keep-filtered downstream)
    pub slot: usize,
    /// 1/T for the drafter softmax; 0 keeps the classic argmax chain
    pub inv_t: f32,
    /// per-chain-step draft-draw uniforms (length >= the round's k_use)
    pub u: Vec<f32>,
}

/// Per-row instruction for a device-sampled decode step.
#[derive(Debug, Clone, Copy)]
pub enum RowSample {
    /// dummy row below the high-water mark - no output wanted
    Hole,
    /// the device executes this plan; the row returns a bare token id
    Device(crate::sampler::DevicePlan),
    /// the row needs its full logits on the host
    Host,
}

/// Result of one device-sampled decode step.
pub struct SampledStep {
    /// per-row sampled token; meaningful only where the plan was `Device`
    pub ids: Vec<u32>,
    /// full logits for each `Host` row, ascending row order
    pub host_rows: Vec<(usize, Vec<f32>)>,
}

/// A finishing prompt's first-token result from a sampled mixed tick.
pub enum FinishSample {
    /// last-row logits for host sampling (the classic readback path)
    Logits(Vec<f32>),
    /// sampled on device with the slot's `fin_plans` entry - the scheduler
    /// must advance the slot's seed stream (`commit_device_plan`)
    Sampled(u32),
}

#[derive(Debug, thiserror::Error)]
pub enum GenError {
    #[error("generation failed: {0}")]
    Backend(String),
    /// A forward pass hit `CUDA_ERROR_OUT_OF_MEMORY`, classified by numeric
    /// result code at the driver boundary (`gpu::from_driver`) and threaded up
    /// here as a type - Not re-sniffed from rendered text. The API layer maps
    /// this to a retryable `Overloaded` capacity error (`EngineError::from_gen`)
    /// naming the levers the caller controls (fewer/smaller image or PDF pages),
    /// because an OOM on a large multimodal prefill is a capacity event, not an
    /// engine bug. The `#[error]` text still carries the `CUDA_ERROR_OUT_OF_MEMORY`
    /// signature so the funnel's text fallback catches any OOM that reaches it
    /// through an untyped `Backend(_)` path (e.g. cuBLAS) as well.
    #[error("GPU out of memory (CUDA_ERROR_OUT_OF_MEMORY)")]
    OutOfMemory,
    /// The paged KV pool ran out of blocks mid-step. The scheduler preempts a
    /// victim sequence and retries, rather than failing the batch (P5b-3).
    #[error("KV pool exhausted")]
    PoolExhausted,
    /// A provably infeasible startup config (e.g. --max-ctx × --max-batch beyond
    /// the KV budget). FATAL at startup: the engine spawn fails with the message
    /// instead of width-halving or falling back to the serial loop.
    #[error("{0}")]
    Config(String),
}

/// Map a GPU backend error to the scheduler error, preserving the pool-exhaustion
/// signal so the scheduler can preempt (P5b-3) instead of failing every sequence.
fn to_gen_err(e: crate::gpu_model::gpt_oss::GpuModelError) -> GenError {
    use crate::gpu_model::gpt_oss::GpuModelError;
    match e {
        GpuModelError::PoolExhausted => GenError::PoolExhausted,
        GpuModelError::Config(msg) => GenError::Config(msg),
        // A device OOM classified at the driver boundary arrives here as a typed
        // Gpu(OutOfMemory); keep it typed so the API maps it to a retryable
        // capacity error rather than a 500. Every other GpuError stays a Backend
        // string - its Display is diagnostic, not caller-actionable.
        GpuModelError::Gpu(crate::gpu::GpuError::OutOfMemory) => GenError::OutOfMemory,
        other => GenError::Backend(other.to_string()),
    }
}

impl Generator for crate::gpu_model::qwen35::GpuQwen35 {
    fn tier_prefix_loading(&mut self, slot: usize, tokens: &[u32]) -> bool {
        crate::gpu_model::qwen35::GpuQwen35::tier_consult_impl(self, slot, tokens)
    }
    fn tier_observe_prefill(&mut self, tokens: u32, wall_us: f64) {
        crate::gpu_model::qwen35::GpuQwen35::tier_observe_prefill_impl(self, tokens, wall_us);
    }
    fn tier_pump(&mut self) {
        crate::gpu_model::qwen35::GpuQwen35::tier_pump_impl(self);
    }
    fn tier_stats(&self) -> Option<crate::kv_tier::TierStats> {
        crate::gpu_model::qwen35::GpuQwen35::tier_stats_impl(self)
    }
    fn tier_report(&self) -> Option<crate::kv_tier::TierReport> {
        crate::gpu_model::qwen35::GpuQwen35::tier_report_impl(self)
    }
    fn reset(&mut self) {
        crate::gpu_model::qwen35::GpuQwen35::reset(self);
    }
    fn forward(&mut self, token: u32) -> Result<Vec<f32>, GenError> {
        self.forward_one(token)
            .map_err(|e| GenError::Backend(e.to_string()))
    }
    fn vocab(&self) -> usize {
        self.vocab
    }
    fn max_context(&self) -> usize {
        self.max_ctx
    }
    fn kv_mem_bytes(&self) -> Option<u64> {
        crate::gpu_model::qwen35::GpuQwen35::kv_mem_bytes(self)
    }

    fn weights_mem_bytes(&self) -> Option<u64> {
        crate::gpu_model::qwen35::GpuQwen35::weights_mem_bytes(self)
    }

    fn device_mem_used(&self) -> Option<u64> {
        crate::gpu_model::qwen35::GpuQwen35::device_mem_used(self)
    }
    fn enable_batch(&mut self, max_batch: usize) -> Result<usize, GenError> {
        // to_gen_err so a Config reject stays typed (fatal at startup, never
        // width-halved into a config the user didn't ask for)
        crate::gpu_model::qwen35::GpuQwen35::enable_batch(self, max_batch).map_err(to_gen_err)
    }
    fn spec_capable(&self) -> bool {
        crate::gpu_model::qwen35::GpuQwen35::serve_spec_on(self)
    }
    fn forward_batch(&mut self, tokens: &[u32], positions: &[u32]) -> Result<Vec<f32>, GenError> {
        crate::gpu_model::qwen35::GpuQwen35::forward_batch(self, tokens, positions)
            .map_err(to_gen_err)
    }
    fn supports_device_sampling(&self) -> bool {
        crate::gpu_model::qwen35::GpuQwen35::supports_device_sampling(self)
    }
    fn forward_batch_sampled(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        plans: &[RowSample],
    ) -> Result<SampledStep, GenError> {
        crate::gpu_model::qwen35::GpuQwen35::forward_batch_sampled(self, tokens, positions, plans)
            .map_err(to_gen_err)
    }
    fn supports_decode_pipe(&self) -> bool {
        crate::gpu_model::qwen35::GpuQwen35::supports_decode_pipe(self)
    }
    fn spec_k_miss_floor(&self) -> Option<usize> {
        crate::gpu_model::qwen35::GpuQwen35::spec_k_miss_floor_mtp(self)
    }
    fn spec_block_width(&self) -> Option<usize> {
        crate::gpu_model::qwen35::GpuQwen35::dflash_block_width(self)
    }
    fn decode_pipe_begin(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        plans: &[RowSample],
    ) -> Result<(), GenError> {
        crate::gpu_model::qwen35::GpuQwen35::decode_pipe_begin(self, tokens, positions, plans)
            .map_err(|e| GenError::Backend(e.to_string()))
    }
    fn decode_pipe_next(&mut self, plans: &[RowSample]) -> Result<Vec<u32>, GenError> {
        crate::gpu_model::qwen35::GpuQwen35::decode_pipe_next(self, plans)
            .map_err(|e| GenError::Backend(e.to_string()))
    }
    fn decode_pipe_drain(&mut self) -> Result<Vec<u32>, GenError> {
        crate::gpu_model::qwen35::GpuQwen35::decode_pipe_drain(self)
            .map_err(|e| GenError::Backend(e.to_string()))
    }
    fn supports_overlap(&self) -> bool {
        crate::gpu_model::qwen35::GpuQwen35::overlap_ready(self)
    }
    fn decode_pipe_begin_slots(
        &mut self,
        slots: &[u32],
        tokens: &[u32],
        positions: &[u32],
        plans: &[RowSample],
    ) -> Result<(), GenError> {
        crate::gpu_model::qwen35::GpuQwen35::decode_pipe_begin_slots(
            self, slots, tokens, positions, plans,
        )
        .map_err(|e| GenError::Backend(e.to_string()))
    }
    fn unified_span_launch(
        &mut self,
        budget: usize,
        fin_plans: &[(usize, RowSample)],
    ) -> Result<bool, GenError> {
        crate::gpu_model::qwen35::GpuQwen35::unified_span_launch(self, budget, fin_plans)
            .map_err(to_gen_err)
    }
    fn spec_draft_begin(
        &mut self,
        pendings: &[(usize, u32)],
        k: usize,
    ) -> Result<Option<(usize, Vec<bool>)>, GenError> {
        crate::gpu_model::qwen35::GpuQwen35::spec_draft_begin_mtp(self, pendings, k)
            .map_err(to_gen_err)
    }
    fn spec_draft_fetch(&mut self) -> Result<Option<Vec<Vec<u32>>>, GenError> {
        crate::gpu_model::qwen35::GpuQwen35::spec_draft_fetch_mtp(self).map_err(to_gen_err)
    }
    // Rung G: the DFlash2 lane samples its selector walk at the
    // request temperature and resolves drafted rows with the truncation-
    // aware rejection sampler. Both answers are the same predicate - the
    // service stashes the chain draws (inv_t + seed) exactly when it will
    // also mark drafted rows RsTrunc.
    fn supports_spec_rs(&self) -> bool {
        self.dflash_rs_available()
    }
    fn supports_spec_rs_trunc(&self) -> bool {
        self.dflash_rs_available()
    }
    fn spec_rs_stash(&mut self, draws: Vec<SpecRsDraw>) {
        self.spec_rs_stash_draws(draws);
    }
    fn forward_mixed_spec_plans(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
        budget: usize,
        plans: &[crate::sampler::DevicePlan],
        fin_plans: &[(usize, RowSample)],
    ) -> Result<(Option<Vec<u32>>, Vec<(usize, FinishSample, usize)>), GenError> {
        crate::gpu_model::qwen35::GpuQwen35::forward_mixed_spec_plans_mtp(
            self, reqs, budget, plans, fin_plans,
        )
        .map_err(to_gen_err)
    }
    fn unified_span_done(&self) -> bool {
        crate::gpu_model::qwen35::GpuQwen35::unified_span_done(self)
    }
    fn unified_span_finish(&mut self) -> Result<Vec<(usize, FinishSample, usize)>, GenError> {
        crate::gpu_model::qwen35::GpuQwen35::unified_span_finish(self).map_err(to_gen_err)
    }
    fn supports_chunked_prefill(&self) -> bool {
        crate::gpu_model::qwen35::GpuQwen35::supports_chunked_prefill(self)
    }
    fn prefill_begin(&mut self, slot: usize, tokens: Vec<u32>) -> Result<(), GenError> {
        crate::gpu_model::qwen35::GpuQwen35::prefill_begin(self, slot, tokens)
            .map_err(|e| GenError::Backend(e.to_string()))
    }
    fn prefill_begin_hinted(
        &mut self,
        slot: usize,
        tokens: Vec<u32>,
        hints: &[usize],
    ) -> Result<usize, GenError> {
        crate::gpu_model::qwen35::GpuQwen35::prefill_begin_hinted(self, slot, tokens, hints)
            .map_err(|e| GenError::Backend(e.to_string()))
    }
    fn prefix_share_floor(&self) -> Option<usize> {
        crate::gpu_model::qwen35::GpuQwen35::prefix_share_floor(self)
    }
    fn prefill_abort(&mut self, slot: usize) -> bool {
        crate::gpu_model::qwen35::GpuQwen35::prefill_abort(self, slot)
    }
    fn forward_mixed(
        &mut self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
    ) -> Result<(Vec<f32>, Vec<(usize, Vec<f32>, usize)>), GenError> {
        crate::gpu_model::qwen35::GpuQwen35::forward_mixed(self, decodes, budget)
            .map_err(to_gen_err)
    }
    fn supports_host_head(&self) -> bool {
        self.host_head_supported()
    }
    fn supports_device_trunc(&self) -> bool {
        self.device_trunc_supported()
    }
    fn forward_mixed_sampled(
        &mut self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
        plans: &[RowSample],
        fin_plans: &[(usize, RowSample)],
    ) -> Result<(SampledStep, Vec<(usize, FinishSample, usize)>), GenError> {
        crate::gpu_model::qwen35::GpuQwen35::forward_mixed_sampled(
            self, decodes, budget, plans, fin_plans,
        )
        .map_err(to_gen_err)
    }
    fn forward_unified_sampled(
        &mut self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
        plans: &[RowSample],
        fin_plans: &[(usize, RowSample)],
    ) -> Result<(SampledStep, Vec<(usize, FinishSample, usize)>), GenError> {
        crate::gpu_model::qwen35::GpuQwen35::forward_unified_sampled(
            self, decodes, budget, plans, fin_plans,
        )
        .map_err(to_gen_err)
    }
    fn forward_prefill(&mut self, slot: usize, tokens: &[u32]) -> Result<Vec<f32>, GenError> {
        self.forward_prefill_slot(slot, tokens).map_err(to_gen_err)
    }
    fn forward_spec_batch(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
    ) -> Result<Option<Vec<u32>>, GenError> {
        crate::gpu_model::qwen35::GpuQwen35::forward_spec_batch_mtp(self, reqs).map_err(to_gen_err)
    }
    fn spec_draft_batch(
        &mut self,
        pendings: &[(usize, u32)],
        k: usize,
    ) -> Result<Option<Vec<Vec<u32>>>, GenError> {
        crate::gpu_model::qwen35::GpuQwen35::spec_draft_batch_mtp(self, pendings, k)
            .map_err(to_gen_err)
    }
    fn spec_ensure_warm(
        &mut self,
        slot: usize,
        committed: &[u32],
        want_pos: u32,
    ) -> Result<bool, GenError> {
        // Hybrid drafters: either warmth serves - the routing picks the
        // drafter per round, so a dflash-cold slot (post-finisher ramp,
        // prefix-resume trim) must still count warm through the MTP re-warm
        // or the service never offers spec at all (measured: c4 fell to the
        // dense-classic rate, ITL 24ms, zero rounds - the warm seam was
        // judging every slot by ring coverage alone).
        // a full warm pass gap-syncs the MTP cursor below: the chain may draft
        self.set_spec_ring_probed(false);
        if self.dflash_attached() && self.dflash_ensure_warm(slot, want_pos) {
            return Ok(true);
        }
        // token-replay drafter: warmth is token-space (spec_draft_kv_space
        // stays false - the scheduler never sends multimodal slots here)
        crate::gpu_model::qwen35::GpuQwen35::spec_ensure_warm_mtp(self, slot, committed)
            .map_err(to_gen_err)
    }
    fn spec_warm_hint(&mut self, on: bool) {
        self.set_spec_warm_wanted(on);
    }
    fn spec_ring_owns_round(&self, live: usize) -> bool {
        crate::gpu_model::qwen35::GpuQwen35::dflash_owns_round(self, live)
    }
    fn spec_draft_per_slot_warm(&self) -> bool {
        // With the block drafter attached, dflash_draft_batch filters warmth
        // per slot (a cold slot gets an empty draft list and rides the
        // verify), so the scheduler's all-warm conjunction must not gate the
        // round on it - same as gemma4. The MTP chain (which does need the
        // conjunction) is only reached after a full warm pass: a ring-probed
        // round declines the chain (spec_ring_probed), and the full pass
        // reports any chain-cold slot as cold, so the conjunction still
        // protects the chain exactly as before.
        self.dflash_attached()
    }
    fn spec_ring_warm(&mut self, slot: usize, want_pos: u32) -> Option<bool> {
        if !self.dflash_attached() {
            return None;
        }
        // the MTP cursors are not gap-synced on a ring-probed round; the chain
        // must not draft from them (see spec_draft_batch_mtp's guard)
        self.set_spec_ring_probed(true);
        Some(self.dflash_ensure_warm(slot, want_pos))
    }
    fn spec_live_cap(&self) -> usize {
        crate::gpu_model::qwen35::GpuQwen35::spec_live_cap_mtp(self)
    }
    fn forward_spec_verify(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
    ) -> Result<Option<Vec<f32>>, GenError> {
        crate::gpu_model::qwen35::GpuQwen35::forward_spec_verify_mtp(self, reqs).map_err(to_gen_err)
    }
    fn forward_spec_batch_plans(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
        plans: &[crate::sampler::DevicePlan],
    ) -> Result<Option<Vec<u32>>, GenError> {
        crate::gpu_model::qwen35::GpuQwen35::forward_spec_batch_plans_mtp(self, reqs, plans)
            .map_err(to_gen_err)
    }
    fn spec_commit(&mut self, committed: &[u32]) -> Result<(), GenError> {
        crate::gpu_model::qwen35::GpuQwen35::spec_commit_mtp(self, committed).map_err(to_gen_err)
    }
    fn forward_prefill_batch(
        &mut self,
        items: &[(usize, Vec<u32>)],
    ) -> Result<Vec<Vec<f32>>, GenError> {
        crate::gpu_model::qwen35::GpuQwen35::forward_prefill_batch(self, items).map_err(to_gen_err)
    }
    fn forward_prefill_stream(&mut self, tokens: &[u32]) -> Result<Vec<f32>, GenError> {
        // batched: slot 0 through the served lane; serial: the token spine
        crate::gpu_model::qwen35::GpuQwen35::prefill_stream(self, tokens).map_err(to_gen_err)
    }
    fn forward_multimodal(
        &mut self,
        chunks: &[crate::service::MmChunk],
    ) -> Result<Option<(Vec<f32>, usize)>, GenError> {
        // loaded without an mmproj -> a helpful Err (not None: the arch can)
        crate::gpu_model::qwen35::GpuQwen35::forward_multimodal_chunks(self, chunks)
            .map(Some)
            .map_err(|e| GenError::Backend(e.to_string()))
    }
    fn take_prefill_reused(&mut self, slot: usize) -> usize {
        crate::gpu_model::qwen35::GpuQwen35::take_prefill_reused(self, slot)
    }
    fn release_inactive_slots(&mut self, occupied: &[bool]) {
        crate::gpu_model::qwen35::GpuQwen35::release_inactive_slots(self, occupied);
    }
    fn pool_free_blocks(&self) -> Option<usize> {
        crate::gpu_model::qwen35::GpuQwen35::pool_free_blocks(self)
    }
    fn supports_mm_slots(&self) -> bool {
        self.has_vision()
    }
    fn vision_budget(&self) -> Option<VisionBudget> {
        self.vision_model().map(|v| v.budget())
    }
    fn prefill_scratch_fits(&self, rows: usize) -> bool {
        crate::gpu_model::qwen35::GpuQwen35::prefill_scratch_fits(self, rows)
    }
    fn forward_prefill_multimodal(
        &mut self,
        slot: usize,
        chunks: &[crate::service::MmChunk],
    ) -> Result<(Vec<f32>, usize), GenError> {
        self.forward_prefill_slot_mm(slot, chunks)
            .map_err(|e| GenError::Backend(e.to_string()))
    }

    fn forward_prefill_multimodal_batch(
        &mut self,
        items: Vec<(usize, Vec<crate::service::MmChunk>)>,
    ) -> Vec<(usize, Result<(Vec<f32>, usize), GenError>)> {
        // One cache-aware batched tower pass over every pending request's
        // images, then one batched prefill pass over every request's rows -
        // the full vi8 fix (encode batching alone left a serial-prefill
        // TTFT plateau)
        let refs: Vec<&[crate::service::MmChunk]> =
            items.iter().map(|(_, c)| c.as_slice()).collect();
        let t0 = std::time::Instant::now();
        match self.encode_images_for_requests(&refs) {
            Ok(per_req) => {
                if paddock_models::dev_var_os!("PADDOCK_ROUTE_WITNESS").is_some() {
                    eprintln!(
                        "pd route: mm encode {} reqs in {:.1}ms",
                        refs.len(),
                        t0.elapsed().as_secs_f64() * 1e3
                    );
                }
                let t1 = std::time::Instant::now();
                let ks: Vec<usize> = items.iter().map(|(k, _)| *k).collect();
                let reqs: Vec<_> = items
                    .into_iter()
                    .zip(per_req)
                    .map(|((k, chunks), images)| (k, chunks, images))
                    .collect();
                let n = reqs.len();
                let res = self.forward_prefill_batch_mm(reqs);
                if paddock_models::dev_var_os!("PADDOCK_ROUTE_WITNESS").is_some() {
                    eprintln!(
                        "pd route: mm prefill {} reqs in {:.1}ms",
                        n,
                        t1.elapsed().as_secs_f64() * 1e3
                    );
                }
                match res {
                    Ok(res) => ks.into_iter().zip(res).map(|(k, lr)| (k, Ok(lr))).collect(),
                    Err(e) => {
                        let msg = e.to_string();
                        ks.into_iter()
                            .map(|k| (k, Err(GenError::Backend(msg.clone()))))
                            .collect()
                    }
                }
            }
            // a batched-encode failure is systemic (alloc/driver): report it
            // on every pending slot rather than half-serving the wave
            Err(e) => {
                let msg = e.to_string();
                items
                    .into_iter()
                    .map(|(k, _)| (k, Err(GenError::Backend(msg.clone()))))
                    .collect()
            }
        }
    }
}

impl Generator for crate::gpu_model::gpt_oss::GpuGptOss {
    fn reset(&mut self) {
        crate::gpu_model::gpt_oss::GpuGptOss::reset(self);
    }
    fn forward(&mut self, token: u32) -> Result<Vec<f32>, GenError> {
        self.forward_one(token)
            .map_err(|e| GenError::Backend(e.to_string()))
    }
    fn vocab(&self) -> usize {
        self.vocab
    }
    fn max_context(&self) -> usize {
        self.max_ctx
    }
    fn device_mem_used(&self) -> Option<u64> {
        crate::gpu_model::gpt_oss::GpuGptOss::device_mem_used(self)
    }

    fn weights_mem_bytes(&self) -> Option<u64> {
        crate::gpu_model::gpt_oss::GpuGptOss::weights_mem_bytes(self)
    }

    fn kv_mem_bytes(&self) -> Option<u64> {
        crate::gpu_model::gpt_oss::GpuGptOss::kv_mem_bytes(self)
    }
    fn enable_batch(&mut self, max_batch: usize) -> Result<usize, GenError> {
        crate::gpu_model::gpt_oss::GpuGptOss::enable_batch(self, max_batch).map_err(to_gen_err)?;
        Ok(max_batch)
    }
    fn forward_batch(&mut self, tokens: &[u32], positions: &[u32]) -> Result<Vec<f32>, GenError> {
        crate::gpu_model::gpt_oss::GpuGptOss::forward_batch(self, tokens, positions)
            .map_err(to_gen_err)
    }
    fn supports_device_sampling(&self) -> bool {
        crate::gpu_model::gpt_oss::GpuGptOss::supports_device_sampling(self)
    }

    fn supports_device_trunc(&self) -> bool {
        self.device_trunc_supported()
    }
    fn forward_batch_sampled(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        plans: &[RowSample],
    ) -> Result<SampledStep, GenError> {
        crate::gpu_model::gpt_oss::GpuGptOss::forward_batch_sampled(self, tokens, positions, plans)
            .map_err(to_gen_err)
    }
    fn supports_decode_pipe(&self) -> bool {
        // G4a: the decode pipe advances positions on device across ticks, so the
        // host can't re-upload the pool block table between them - pool mode
        // forces the per-tick host-driven decode (forward_batch_sampled).
        !crate::gpu_model::gpt_oss::GpuGptOss::pool_active(self)
            && crate::gpu_model::gpt_oss::GpuGptOss::supports_decode_pipe(self)
    }
    fn decode_pipe_begin(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        plans: &[RowSample],
    ) -> Result<(), GenError> {
        crate::gpu_model::gpt_oss::GpuGptOss::decode_pipe_begin(self, tokens, positions, plans)
            .map_err(|e| GenError::Backend(e.to_string()))
    }
    fn decode_pipe_next(&mut self, plans: &[RowSample]) -> Result<Vec<u32>, GenError> {
        crate::gpu_model::gpt_oss::GpuGptOss::decode_pipe_next(self, plans)
            .map_err(|e| GenError::Backend(e.to_string()))
    }
    fn decode_pipe_drain(&mut self) -> Result<Vec<u32>, GenError> {
        crate::gpu_model::gpt_oss::GpuGptOss::decode_pipe_drain(self)
            .map_err(|e| GenError::Backend(e.to_string()))
    }
    fn forward_prefill(&mut self, slot: usize, tokens: &[u32]) -> Result<Vec<f32>, GenError> {
        crate::gpu_model::gpt_oss::GpuGptOss::forward_prefill(self, slot, tokens)
            .map_err(to_gen_err)
    }
    fn forward_prefill_batch(
        &mut self,
        items: &[(usize, Vec<u32>)],
    ) -> Result<Vec<Vec<f32>>, GenError> {
        crate::gpu_model::gpt_oss::GpuGptOss::forward_prefill_batch(self, items).map_err(to_gen_err)
    }
    fn forward_spec_batch(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
    ) -> Result<Option<Vec<u32>>, GenError> {
        // P3: spec verify now grows the pool for its draft span before the baked
        // block-table read (see forward_spec_batch_inner), so it runs under the
        // budget pool. PADDOCK_NO_POOL_SPEC pins the old fall-back-to-plain-decode.
        if crate::gpu_model::gpt_oss::GpuGptOss::pool_active(self)
            && paddock_models::dev_var_os!("PADDOCK_NO_POOL_SPEC").is_some()
        {
            return Ok(None);
        }
        crate::gpu_model::gpt_oss::GpuGptOss::forward_spec_batch(self, reqs)
            .map(Some)
            .map_err(|e| GenError::Backend(e.to_string()))
    }
    fn take_prefill_reused(&mut self, slot: usize) -> usize {
        crate::gpu_model::gpt_oss::GpuGptOss::take_prefill_reused(self, slot)
    }
    fn release_inactive_slots(&mut self, occupied: &[bool]) {
        crate::gpu_model::gpt_oss::GpuGptOss::release_inactive_slots(self, occupied);
    }
    fn pool_free_blocks(&self) -> Option<usize> {
        crate::gpu_model::gpt_oss::GpuGptOss::pool_free_blocks(self)
    }
    fn supports_chunked_prefill(&self) -> bool {
        true
    }
    fn prefill_begin(&mut self, slot: usize, tokens: Vec<u32>) -> Result<(), GenError> {
        crate::gpu_model::gpt_oss::GpuGptOss::prefill_begin(self, slot, tokens)
            .map_err(|e| GenError::Backend(e.to_string()))
    }
    fn forward_mixed(
        &mut self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
    ) -> Result<(Vec<f32>, Vec<(usize, Vec<f32>, usize)>), GenError> {
        crate::gpu_model::gpt_oss::GpuGptOss::forward_mixed(self, decodes, budget)
            .map_err(to_gen_err)
    }
    fn forward_mixed_sampled(
        &mut self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
        plans: &[RowSample],
        _fin_plans: &[(usize, RowSample)],
    ) -> Result<(SampledStep, Vec<(usize, FinishSample, usize)>), GenError> {
        crate::gpu_model::gpt_oss::GpuGptOss::forward_mixed_sampled(self, decodes, budget, plans)
            .map_err(to_gen_err)
            .map(|(step, fin)| {
                let fin = fin
                    .into_iter()
                    .map(|(k, l, r)| (k, FinishSample::Logits(l), r))
                    .collect();
                (step, fin)
            })
    }
}

#[cfg(test)]
mod vision_budget_tests {
    use super::*;

    /// Qwen3.5/3.6's real numbers: 16.7 Mpx over 32x32-pixel tokens.
    fn qwen() -> VisionBudget {
        VisionBudget {
            max_pixels: 16_777_216,
            min_pixels: 65_536,
            max_edge: None,
            pixels_per_token: 1024,
            max_tokens: 16_384,
            min_tokens: 64,
        }
    }

    /// gemma4's: a fixed 280 rows per image whatever you send it.
    fn gemma4() -> VisionBudget {
        VisionBudget {
            max_pixels: 645_120,
            min_pixels: 92_160,
            max_edge: None,
            pixels_per_token: 2304,
            max_tokens: 280,
            min_tokens: 40,
        }
    }

    #[test]
    fn fit_preserves_aspect_and_never_upsamples() {
        let b = qwen();
        // already inside the budget: untouched, so the family's own
        // preprocessing stays the only resample on the ordinary path
        assert_eq!(b.fit(1920, 1080), (1920, 1080));
        // over the ceiling: scaled down, aspect held
        let (w, h) = b.fit(8000, 6000);
        assert!(
            (w as u64) * (h as u64) <= b.max_pixels,
            "{w}x{h} over budget"
        );
        let (src, out) = (8000.0 / 6000.0, w as f64 / h as f64);
        assert!((src - out).abs() < 0.01, "aspect drifted {src} -> {out}");
    }

    /// The edge cap runs before the area fit - granite's pinpoint list bounds a
    /// single side at 3840, and a 10000x200 banner is inside the area budget
    /// while being far outside the list.
    #[test]
    fn an_edge_cap_binds_even_when_the_area_fits() {
        let b = VisionBudget {
            max_edge: Some(3840),
            ..qwen()
        };
        let (w, h) = b.fit(10_000, 200);
        assert_eq!(w, 3840);
        assert_eq!(h, 76); // 200 * 3840/10000 = 76.8, floored
    }

    /// The three detail levels, as the API resolves them.
    #[test]
    fn detail_levels_are_a_token_cap_not_a_pixel_one() {
        let b = qwen();
        let (w, h) = (6000, 4000); // 24 Mpx, over even qwen's ceiling

        // high = the model's published spec
        let hi = b.fit_tokens(w, h, b.max_tokens);
        assert!((hi.0 as u64) * (hi.1 as u64) <= b.max_pixels);
        assert!((hi.0 as u64) * (hi.1 as u64) > 16_000_000, "{hi:?}");

        // auto = the conservative default, exactly 4096 rows' worth
        let auto = b.fit_tokens(w, h, AUTO_MAX_TOKENS);
        assert!((auto.0 as u64) * (auto.1 as u64) <= 4_194_304, "{auto:?}");
        assert!(b.tokens_for_capped(w, h, AUTO_MAX_TOKENS) <= AUTO_MAX_TOKENS);

        // low = the smallest the tower will really encode
        let lo = b.fit_tokens(w, h, b.min_tokens);
        assert!((lo.0 as u64) * (lo.1 as u64) <= b.min_pixels, "{lo:?}");

        // and they are strictly ordered - a UI listing them can say so
        assert!(lo.0 < auto.0 && auto.0 < hi.0);
    }

    /// Asking for fewer rows than the tower's floor cannot buy fewer rows; it
    /// only throws pixels away. The floor is enforced, so `low` on a model with
    /// a high minimum is a no-op rather than a quiet quality cut.
    #[test]
    fn a_token_request_under_the_floor_clamps_to_the_floor() {
        let b = qwen();
        assert_eq!(b.pixels_for_tokens(0), b.min_pixels);
        assert_eq!(b.pixels_for_tokens(1), b.min_pixels);
        assert_eq!(b.pixels_for_tokens(u32::MAX), b.max_pixels);
    }

    /// gemma4 charges 280 rows per image whatever it is sent, so every detail
    /// level collapses to the same request there. Worth a test rather than a
    /// comment: a picker that offers three choices on gemma4 is offering one,
    /// and that should be a known fact instead of a support question.
    #[test]
    fn gemma4_prices_every_detail_level_the_same() {
        let b = gemma4();
        let (w, h) = (4000, 3000);
        let auto = b.tokens_for_capped(w, h, AUTO_MAX_TOKENS.min(b.max_tokens));
        let high = b.tokens_for_capped(w, h, b.max_tokens);
        assert_eq!(auto, high, "auto is already gemma4's whole ceiling");
        assert_eq!(high, b.max_tokens);
        // its floor still differs, so `low` remains a real (small) choice
        assert!(b.tokens_for_capped(w, h, b.min_tokens) < high);
    }
}
